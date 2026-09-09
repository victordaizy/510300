"""解析并审计合法取得的上交所股票期权历史快照文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sse_option_historical_snapshot_import_v1.yaml"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _case_column_map(frame: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if key in result:
            raise ValueError(f"CSV列名忽略大小写后重复：{column}")
        result[key] = str(column)
    return result


def _scalar_column(frame: pd.DataFrame, name: str) -> str:
    mapping = _case_column_map(frame)
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"官方期权快照缺少标量字段：{name}")
    return mapping[key]


def _split_array_value(value: Any, levels: int, field: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        pieces = list(value)
    else:
        text = str(value).strip().strip("[](){}")
        pieces = [piece for piece in re.split(r"[,;|\s]+", text) if piece]
    if len(pieces) != levels:
        raise ValueError(f"{field}五档数组长度不是{levels}：{value}")
    values = [float(piece) for piece in pieces]
    if not all(pd.notna(values)):
        raise ValueError(f"{field}五档数组存在无效数字")
    return values


def extract_levels(frame: pd.DataFrame, base: str, levels: int) -> pd.DataFrame:
    """兼容官方数组列及常见明确展开列；禁止按列位置推断。"""

    mapping = _case_column_map(frame)
    expanded: list[str] = []
    matched_levels: list[int] = []
    for level in range(1, levels + 1):
        candidates = (
            f"{base}{level}",
            f"{base}[{level}]",
            f"{base}_{level}",
            f"{base}{level:02d}",
        )
        matches = [mapping[item.lower()] for item in candidates if item.lower() in mapping]
        if len(matches) > 1:
            raise ValueError(f"{base}第{level}档存在多个候选列：{matches}")
        if matches:
            expanded.append(matches[0])
            matched_levels.append(level)
    if len(expanded) == levels:
        result = frame[expanded].apply(pd.to_numeric, errors="coerce").copy()
        result.columns = [f"{base}{level}" for level in range(1, levels + 1)]
        return result
    aggregate_candidates = [f"{base}[{levels}]", base]
    aggregate = [
        mapping[item.lower()] for item in aggregate_candidates if item.lower() in mapping
    ]
    if matched_levels == [levels] and aggregate == [expanded[0]]:
        expanded = []
        matched_levels = []
    elif expanded:
        raise ValueError(f"{base}展开列不完整：只找到{len(expanded)}/{levels}档")
    if len(aggregate) != 1:
        raise ValueError(f"无法唯一识别{base}五档数组列：{aggregate}")
    values = frame[aggregate[0]].map(lambda value: _split_array_value(value, levels, base))
    return pd.DataFrame(
        values.tolist(),
        index=frame.index,
        columns=[f"{base}{level}" for level in range(1, levels + 1)],
    )


def parse_datetime(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    digits = text.str.replace(r"\D", "", regex=True)
    if digits.str.len().lt(14).any():
        raise ValueError("官方期权快照DateTime少于14位")
    parsed = pd.to_datetime(digits.str[:14], format="%Y%m%d%H%M%S", errors="coerce")
    if parsed.isna().any():
        raise ValueError("官方期权快照DateTime无法解析")
    return parsed


def active_510300_contracts(master: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    required = {
        "contract_code",
        "underlying_code",
        "option_type",
        "list_date",
        "expiry_date",
        "delist_date",
        "strike",
        "contract_unit",
        "is_adjusted",
    }
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"510300点时合约主表缺少字段：{missing}")
    data = master.loc[master["underlying_code"].astype(str).eq("510300.SH")].copy()
    for column in ("list_date", "expiry_date", "delist_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce").dt.normalize()
    date = trade_date.normalize()
    active = data.loc[
        data["list_date"].le(date)
        & (data["delist_date"].isna() | data["delist_date"].ge(date))
    ].copy()
    if active.empty:
        raise ValueError("点时主表没有当天活跃510300期权")
    if active["contract_code"].duplicated().any():
        raise ValueError("点时主表存在重复合约代码")
    return active.sort_values("contract_code").reset_index(drop=True)


def normalize_sse_snapshot(
    raw: pd.DataFrame,
    master: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    levels = int(config["schema"]["levels"])
    scalar_names = list(config["schema"]["scalar_required"])
    scalar_columns = {name: _scalar_column(raw, name) for name in scalar_names}
    data = raw[[scalar_columns[name] for name in scalar_names]].copy()
    data.columns = scalar_names
    data["source_row_number"] = range(2, len(data) + 2)
    for base in config["schema"]["array_required"]:
        data = pd.concat([data, extract_levels(raw, base, levels)], axis=1)
    data["snapshot_timestamp"] = parse_datetime(data["DateTime"])
    data["trade_date"] = data["snapshot_timestamp"].dt.normalize()
    expected = trade_date.normalize()
    if not data["trade_date"].eq(expected).all():
        raise ValueError("官方快照文件含非预期交易日记录")
    data["contract_code"] = (
        data["SecurityID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True) + ".SH"
    )
    start = pd.Timestamp(
        f"{expected.date()} {config['selection']['earliest_accepted_snapshot_time']}"
    )
    end = pd.Timestamp(
        f"{expected.date()} {config['selection']['latest_accepted_snapshot_time']}"
    )
    window = data.loc[data["snapshot_timestamp"].between(start, end, inclusive="both")].copy()
    if window.empty:
        raise ValueError("官方快照在冻结收盘选择窗口内没有记录")
    window = window.sort_values(
        ["contract_code", "snapshot_timestamp", "source_row_number"],
        kind="mergesort",
    ).drop_duplicates("contract_code", keep="last")
    active = active_510300_contracts(master, expected)
    expected_codes = set(active["contract_code"].astype(str))
    selected = window.loc[window["contract_code"].isin(expected_codes)].copy()
    actual_codes = set(selected["contract_code"].astype(str))
    selected = selected.merge(
        active[
            [
                "contract_code",
                "option_type",
                "expiry_date",
                "strike",
                "contract_unit",
                "is_adjusted",
                "list_date",
                "delist_date",
            ]
        ],
        on="contract_code",
        how="left",
        validate="one_to_one",
    )
    numeric_map = {
        "PreClosePx": "pre_close",
        "OpenPx": "open",
        "HighPx": "high",
        "LowPx": "low",
        "LastPx": "last_price",
        "TotalLongPosition": "open_interest",
        "TotalVolumeTrade": "volume",
        "TotalValueTrade": "amount",
        "AvgPx": "average_price",
        "PreSettlePx": "pre_settle",
        "SettlePx": "settle",
    }
    selected = selected.rename(columns=numeric_map)
    for column in numeric_map.values():
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    for level in range(1, levels + 1):
        rename = {
            f"BidPrice{level}": f"bid{level}",
            f"BidOrderQty{level}": f"bid{level}_volume",
            f"OfferPx{level}": f"ask{level}",
            f"OfferQty{level}": f"ask{level}_volume",
        }
        selected = selected.rename(columns=rename)
        for column in rename.values():
            selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.rename(columns={"PhaseCode": "phase_code"})
    selected["expected_trade_date"] = expected
    selected["source"] = "sseinfo.official_historical_option_snapshot"
    selected["source_interface_version"] = config["official_source"]["interface_version"]
    selected["selection_window_start"] = start
    selected["selection_window_end"] = end
    coverage = len(expected_codes & actual_codes) / len(expected_codes)
    positive_bid = float(selected["bid1"].gt(0).mean()) if len(selected) else 0.0
    positive_ask = float(selected["ask1"].gt(0).mean()) if len(selected) else 0.0
    inverted = int(selected["bid1"].gt(selected["ask1"]).sum())
    duplicates = int(selected["contract_code"].duplicated().sum())
    gates = {
        "active_contract_coverage": coverage
        >= float(config["quality"]["active_contract_coverage_minimum"]),
        "exact_active_contract_set": actual_codes == expected_codes,
        "positive_bid1_ratio": positive_bid
        >= float(config["quality"]["positive_bid1_ratio_minimum"]),
        "positive_ask1_ratio": positive_ask
        >= float(config["quality"]["positive_ask1_ratio_minimum"]),
        "no_inverted_quotes": inverted <= int(config["quality"]["maximum_inverted_quotes"]),
        "no_duplicate_contracts": duplicates
        <= int(config["quality"]["maximum_duplicate_contracts"]),
    }
    audit = {
        "status": "PASS" if all(gates.values()) else "NO_VIEW",
        "expected_contract_count": len(expected_codes),
        "actual_contract_count": len(actual_codes),
        "active_contract_coverage": coverage,
        "positive_bid1_ratio": positive_bid,
        "positive_ask1_ratio": positive_ask,
        "inverted_quote_count": inverted,
        "duplicate_contract_count": duplicates,
        "gates": gates,
    }
    return selected.sort_values("contract_code").reset_index(drop=True), audit


def validate_license_evidence(path: Path, trade_date: pd.Timestamp, config: dict) -> dict:
    return validate_license_evidence_at_root(path, trade_date, config, ROOT)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_license_evidence_at_root(
    path: Path,
    trade_date: pd.Timestamp,
    config: dict,
    project_root: Path,
) -> dict:
    licensed_root = (project_root / config["paths"]["licensed_input_root"]).resolve()
    evidence_path = path.resolve()
    if not _is_within(evidence_path, licensed_root):
        raise PermissionError("许可证明JSON必须位于冻结的许可数据根目录内")
    if not evidence_path.exists():
        raise PermissionError("缺少合法数据许可证明文件")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    required = {
        "provider",
        "license_holder",
        "authorized_product",
        "valid_start",
        "valid_end",
        "research_use_permitted",
        "evidence_document_path",
        "evidence_document_sha256",
    }
    missing = sorted(required.difference(evidence))
    if missing:
        raise PermissionError(f"许可证明缺少字段：{missing}")
    if evidence["provider"] != config["official_source"]["provider"]:
        raise PermissionError("许可提供方不是上证所信息网络有限公司")
    if evidence["authorized_product"] != config["quality"]["authorized_product_required"]:
        raise PermissionError("许可授权产品不是股票期权历史行情")
    if not str(evidence["license_holder"]).strip():
        raise PermissionError("许可持有人不能为空")
    if not evidence["research_use_permitted"]:
        raise PermissionError("许可未明确允许研究用途")
    date = trade_date.normalize()
    if not pd.Timestamp(evidence["valid_start"]) <= date <= pd.Timestamp(evidence["valid_end"]):
        raise PermissionError("交易日不在许可有效期内")
    expected_hash = str(evidence["evidence_document_sha256"]).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise PermissionError("许可证明文档SHA-256格式无效")
    document_path = (licensed_root / str(evidence["evidence_document_path"])).resolve()
    if not _is_within(document_path, licensed_root):
        raise PermissionError("许可证明原件必须位于冻结的许可数据根目录内")
    if not document_path.is_file():
        raise PermissionError("许可证明原件不存在")
    if sha256(document_path) != expected_hash:
        raise PermissionError("许可证明原件SHA-256不匹配")
    return evidence


def validate_input_path(
    input_path: Path,
    trade_date: pd.Timestamp,
    config: dict,
    project_root: Path = ROOT,
) -> Path:
    """只接受许可根目录下官方冻结层级的当日Snapshot.csv。"""

    licensed_root = (project_root / config["paths"]["licensed_input_root"]).resolve()
    resolved = input_path.resolve()
    expected = (
        licensed_root
        / "sho"
        / trade_date.strftime("%Y%m%d")
        / "Snapshot.csv"
    ).resolve()
    if resolved != expected or not _is_within(resolved, licensed_root):
        raise PermissionError(
            f"输入文件必须严格位于许可目录：{expected}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"官方历史快照不存在：{resolved}")
    return resolved


def validate_master_path(
    master_path: Path,
    config: dict,
    project_root: Path = ROOT,
) -> Path:
    expected = (project_root / config["paths"]["contract_master"]).resolve()
    resolved = master_path.resolve()
    if resolved != expected:
        raise PermissionError(f"合约主表路径必须使用冻结值：{expected}")
    if not resolved.is_file():
        raise FileNotFoundError(f"冻结的510300合约主表不存在：{resolved}")
    return resolved


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入合法取得的上交所510300期权历史快照")
    parser.add_argument("--trade-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--input", required=True, help="官方sho/YYYYMMDD/Snapshot.csv路径")
    parser.add_argument(
        "--master",
        default="data/raw/return_tail/options/510300_contract_master.parquet",
        help="点时510300合约主表",
    )
    return parser.parse_args()


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.freeze_sse_option_historical_snapshot_import_v1 import verify_protocol

    config, manifest = verify_protocol()
    args = parse_args()
    trade_date = pd.Timestamp(args.trade_date).normalize()
    status_path = ROOT / config["paths"]["status"]
    base = {
        "project_id": config["protocol"]["project_id"],
        "trade_date": str(trade_date.date()),
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "manifest_frozen_at": manifest["frozen_at"],
        "strategy_or_return_test_authorized": False,
    }
    try:
        evidence = validate_license_evidence(
            ROOT / config["paths"]["license_evidence"], trade_date, config
        )
        input_path = validate_input_path(Path(args.input), trade_date, config)
        master_path = validate_master_path(ROOT / args.master, config)
        raw = pd.read_csv(input_path, encoding=config["official_source"]["encoding"])
        master = pd.read_parquet(master_path)
        normalized, audit = normalize_sse_snapshot(raw, master, trade_date, config)
        if audit["status"] != "PASS":
            payload = {**base, "status": "NO_VIEW_QUALITY_GATE", "audit": audit}
            _atomic_json(payload, status_path)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 3
        output_path = ROOT / config["paths"]["output_root"] / f"{trade_date:%Y%m%d}.parquet"
        if output_path.exists():
            raise FileExistsError("规范化历史快照已存在，禁止覆盖")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".parquet.tmp")
        if temporary.exists():
            raise FileExistsError("存在未处理的临时快照文件，禁止覆盖")
        normalized.to_parquet(temporary, index=False)
        if output_path.exists():
            temporary.unlink(missing_ok=True)
            raise FileExistsError("并发导入已生成同日快照，禁止覆盖")
        temporary.replace(output_path)
        payload = {
            **base,
            "status": "PASS_IMPORT_ONLY",
            "input": str(input_path),
            "input_sha256": sha256(input_path),
            "contract_master": master_path.relative_to(ROOT).as_posix(),
            "contract_master_sha256": sha256(master_path),
            "license_holder": evidence["license_holder"],
            "output": output_path.relative_to(ROOT).as_posix(),
            "output_sha256": sha256(output_path),
            "audit": audit,
        }
        _atomic_json(payload, status_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        payload = {
            **base,
            "status": "NO_VIEW_IMPORT_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        }
        _atomic_json(payload, status_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
