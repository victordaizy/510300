"""冻结并构建 510300 的 60D/120D 总回报成分账本。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.total_return_component_ledger_v1 import (  # noqa: E402
    build_component_ledger,
    build_etf_total_return_series,
    build_forward_schedule,
)


PROGRAM_ID = "510300_TOTAL_RETURN_COMPONENT_LEDGER_V1"
CONFIG_PATH = ROOT / "config/510300_total_return_component_ledger_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/total_return_component_ledger_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("总回报成分账本配置的 PROGRAM_ID 不匹配")
    return config


def project_path(relative: str) -> Path:
    return ROOT / Path(str(relative).replace("/", os.sep))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json_new(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    _atomic_bytes_new(path, (payload + "\n").encode("utf-8"))


def atomic_text_new(path: Path, value: str) -> None:
    _atomic_bytes_new(path, value.encode("utf-8"))


def atomic_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _source_files(config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    source = config["source_contract"]
    files: dict[str, tuple[str, str]] = {
        source["price_index"]["path"]: (
            source["price_index"]["expected_sha256"],
            "CSI300_PRICE_INDEX",
        ),
        source["total_return_index"]["path"]: (
            source["total_return_index"]["expected_sha256"],
            "CSI300_TOTAL_RETURN_AND_AGGREGATE_PE",
        ),
        source["etf_price"]["path"]: (
            source["etf_price"]["expected_sha256"],
            "510300_UNADJUSTED_PRICE",
        ),
        source["etf_dividends"]["path"]: (
            source["etf_dividends"]["expected_sha256"],
            "510300_OFFICIAL_CASH_DISTRIBUTIONS",
        ),
        source["component_total_return"]["path"]: (
            source["component_total_return"]["expected_sha256"],
            "COMPONENT_TOTAL_RETURN_CLOSE",
        ),
        source["component_state"]["path"]: (
            source["component_state"]["expected_sha256"],
            "PIT_COMPONENT_FINANCIALS_AND_PROXY_WEIGHTS",
        ),
    }
    for year, specification in source["component_raw_price_yearly"].items():
        files[specification["path"]] = (
            specification["expected_sha256"],
            f"COMPONENT_RAW_PRICE_{year}",
        )
    return files


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent_contract"]
    manifest = json.loads(
        project_path(parent["power_audit_manifest"]).read_text(encoding="utf-8")
    )
    status = json.loads(
        project_path(parent["power_audit_status"]).read_text(encoding="utf-8")
    )
    if manifest.get("manifest_payload_sha256") != parent[
        "expected_manifest_payload_sha256"
    ]:
        raise ValueError("功效审计协议哈希不一致")
    if status.get("status_payload_sha256") != parent[
        "expected_status_payload_sha256"
    ]:
        raise ValueError("功效审计状态哈希不一致")
    if status.get("status") != parent["expected_status"]:
        raise ValueError("功效审计状态不允许进入成分账本")
    if status.get("portfolio_evaluation_allowed") is not False:
        raise ValueError("功效审计意外开放了组合评估")
    for relative, (expected, label) in _source_files(config).items():
        path = project_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"{label} 缺失：{path}")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} 哈希不一致")
    return status


def _registered_inputs(config: dict[str, Any]) -> dict[str, str]:
    parent = config["parent_contract"]
    registered = {
        parent["power_audit_manifest"]: "FROZEN_POWER_AUDIT_PROTOCOL",
        parent["power_audit_status"]: "FROZEN_POWER_AUDIT_STATUS",
    }
    registered.update(
        {relative: role for relative, (_, role) in _source_files(config).items()}
    )
    return registered


def _output_paths(config: dict[str, Any]) -> list[Path]:
    return [
        project_path(value)
        for key, value in config["artifacts"].items()
        if key != "protocol_manifest"
    ]


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"成分账本冻结清单已存在：{manifest_path}")
    existing = [str(path) for path in _output_paths(config) if path.exists()]
    if existing:
        raise RuntimeError(f"冻结前发现同名成分账本输出：{existing}")
    parent_status = validate_inputs(config)
    registered: dict[str, Any] = {}
    for relative, role in _registered_inputs(config).items():
        path = project_path(relative)
        registered[Path(relative).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_COMPONENT_LEDGER_VALUE_READ",
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered,
        "parent_status_payload_sha256": parent_status["status_payload_sha256"],
        "horizons_scopes_identities_coverage_gates_frozen": True,
        "historical_official_weight_vintage_verified": False,
        "component_pure_dividend_archive_available": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json_new(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("成分账本协议尚未冻结")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if canonical_hash(payload) != expected_hash:
        raise ValueError("成分账本清单自身哈希失败")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("成分账本配置在冻结后发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("成分账本配置语义在冻结后发生漂移")
    for relative, expected in manifest["implementation_files"].items():
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"成分账本实现文件发生漂移：{relative}")
    for relative, metadata in manifest["registered_inputs"].items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"成分账本冻结输入发生漂移或缺失：{relative}")
    validate_inputs(config)
    return manifest


def _load_filtered_raw_prices(
    config: dict[str, Any], needed_dates: set[pd.Timestamp]
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for specification in config["source_contract"][
        "component_raw_price_yearly"
    ].values():
        frame = pd.read_parquet(
            project_path(specification["path"]),
            columns=["stock_code", "date", "raw_close"],
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        filtered = frame.loc[frame["date"].isin(needed_dates)].copy()
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        raise ValueError("所需 origin/target 日期没有成分原始价格")
    return pd.concat(frames, ignore_index=True)


def _scope_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    return (
        ledger.groupby(["scope", "horizon_market_days"], as_index=False)
        .agg(
            row_count=("origin", "count"),
            first_origin=("origin", "min"),
            last_origin=("origin", "max"),
            identity_pass_count=(
                "identity_multiplicative_residual",
                lambda values: int(values.abs().le(1.0e-10).sum()),
            ),
            median_market_weight_coverage=("market_weight_coverage", "median"),
            minimum_market_weight_coverage=("market_weight_coverage", "min"),
            median_financial_weight_coverage=("financial_weight_coverage", "median"),
            minimum_financial_weight_coverage=("financial_weight_coverage", "min"),
        )
        .sort_values(["scope", "horizon_market_days"], kind="stable")
        .reset_index(drop=True)
    )


def _build_report(
    manifest: dict[str, Any],
    ledger: pd.DataFrame,
    coverage: pd.DataFrame,
    identity: pd.DataFrame,
) -> str:
    summary = _scope_summary(ledger)
    financial_counts = coverage["financial_coverage_status"].value_counts().to_dict()
    identity_counts = identity["identity_status"].value_counts().to_dict()
    lines = [
        "# 510300_TOTAL_RETURN_COMPONENT_LEDGER_V1",
        "",
        "## 裁决",
        "",
        "60D/120D 总回报成分账本已完成。它是会计与归因产物，不是收益预测、策略回测或仓位映射。",
        "",
        f"- 协议哈希：`{manifest['manifest_payload_sha256']}`",
        f"- 账本：`{len(ledger)}` 行；覆盖率：`{len(coverage)}` 行；恒等式审计：`{len(identity)}` 行。",
        f"- 财务覆盖状态：`{json.dumps(financial_counts, ensure_ascii=False, sort_keys=True)}`",
        f"- 恒等式状态：`{json.dumps(identity_counts, ensure_ascii=False, sort_keys=True)}`",
        "- `RETURN_PREDICTION_ALLOWED=false`，`PORTFOLIO_EVALUATION_ALLOWED=false`。",
        "",
        "## 三种口径",
        "",
        "| 口径 | horizon | 行数 | 恒等式通过 | 市场覆盖中位数 | 市场覆盖最低 | 财务覆盖中位数 | 财务覆盖最低 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        median_financial = (
            f"{row.median_financial_weight_coverage:.2%}"
            if np.isfinite(row.median_financial_weight_coverage)
            else "NA"
        )
        minimum_financial = (
            f"{row.minimum_financial_weight_coverage:.2%}"
            if np.isfinite(row.minimum_financial_weight_coverage)
            else "NA"
        )
        lines.append(
            f"| {row.scope} | {int(row.horizon_market_days)}D | {int(row.row_count)} | {int(row.identity_pass_count)} | {row.median_market_weight_coverage:.2%} | {row.minimum_market_weight_coverage:.2%} | {median_financial} | {minimum_financial} |"
        )
    lines.extend(
        [
            "",
            "## 成分定义与限制",
            "",
            "1. 实际指数链式口径：`H00300/000300` 给出隐含股息再投资因子；`000300/PE_TTM` 给出链式盈利代理，`PE_TTM` 变化给出倍数重估。恒等式精确，但历史 PE 并非首次发布版本档案，因此盈利/倍数状态为 `PARTIAL`。",
            "2. origin 固定成分与权重口径：以冻结 origin 的市值代理权重买入并持有。总回报与原始价格之比只能识别“分配及公司行动”合并项；缺少成分股逐笔分配档案，纯股息项为 `NO_VIEW`。",
            "3. 固定财务覆盖 cohort：只在同一 origin 固定持股、起止 PIT 财务事实、有效股本和起止价格均可用的覆盖 cohort 上，将价格精确分解为盈利增长×倍数重估。低于 80% 权重覆盖时不填数。",
            "4. `H00300/固定 cohort 总回报` 被登记为“成分/权重＋指数方法＋代理权重残差”，不能冒充纯粹的换仓收益。历史官方权重版本仍未验证。",
            "5. 跟踪残差使用含官方现金分红的 510300 总回报与 H00300 总回报之比。",
            "",
            "## 精确恒等式",
            "",
            "- 实际指数：`指数隐含股息 × 链式盈利 × 链式倍数 × ETF跟踪 = ETF总回报`。",
            "- 固定市场 cohort：`分配及公司行动 × 未拆分价格 × 成分权重方法残差 × ETF跟踪 = ETF总回报`。",
            "- 固定财务 cohort：`分配及公司行动 × 盈利增长 × 倍数重估 × 成分权重方法残差 × ETF跟踪 = ETF总回报`。",
            "",
            "恒等式通过只证明账本算术闭合，不证明任何成分可预测。",
            "",
            "## 下一步",
            "",
            "分别检验 CF→盈利/现金流/breadth、DR→倍数/ERP gap、RC→波动/下行半方差/相关性/流动性冲击的测量有效性。任何模块都不得直接改回总回报目标。当前仍为 `ABSTAIN / POSITION_UNSET`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_ledger(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(config)
    source = config["source_contract"]
    program = config["program"]
    contract = config["ledger_contract"]
    price_index = pd.read_parquet(project_path(source["price_index"]["path"]))
    total_return_index = pd.read_parquet(
        project_path(source["total_return_index"]["path"])
    )
    etf_price = pd.read_parquet(project_path(source["etf_price"]["path"]))
    dividends = pd.read_csv(project_path(source["etf_dividends"]["path"]))
    component_state = pd.read_parquet(
        project_path(source["component_state"]["path"])
    )
    etf_total_return = build_etf_total_return_series(etf_price, dividends)
    schedule = build_forward_schedule(
        component_state["origin"],
        total_return_index["date"],
        horizons_market_days=[int(value) for value in program["horizons_market_days"]],
        origin_start=str(program["origin_start"]),
        observation_cutoff=str(program["observation_cutoff"]),
    )
    pass_schedule = schedule.loc[
        schedule["schedule_status"].eq("PASS_EXACT_MARKET_DAY_TARGET")
    ]
    needed_dates = set(
        pd.to_datetime(
            pd.concat([pass_schedule["origin"], pass_schedule["target_date"]]),
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
    )
    raw_price = _load_filtered_raw_prices(config, needed_dates)
    component_total_return = pd.read_parquet(
        project_path(source["component_total_return"]["path"]),
        columns=["date", "con_code", "total_return_close"],
    )
    component_total_return["date"] = pd.to_datetime(
        component_total_return["date"], errors="coerce"
    ).dt.normalize()
    component_total_return = component_total_return.loc[
        component_total_return["date"].isin(needed_dates)
    ].copy()
    ledger, coverage, identity = build_component_ledger(
        schedule=schedule,
        price_index=price_index,
        total_return_index=total_return_index,
        etf_total_return=etf_total_return,
        component_state=component_state,
        component_raw_price=raw_price,
        component_total_return=component_total_return,
        effective_share_columns=list(contract["effective_share_columns"]),
        market_minimum_weight_coverage=float(
            contract["market_component_minimum_weight_coverage"]
        ),
        financial_pass_minimum_weight_coverage=float(
            contract["financial_pass_minimum_weight_coverage"]
        ),
        financial_partial_minimum_weight_coverage=float(
            contract["financial_partial_minimum_weight_coverage"]
        ),
        identity_tolerance=float(contract["identity_tolerance"]),
    )
    if ledger.empty or coverage.empty or identity.empty:
        raise ValueError("总回报成分账本或审计输出为空")

    artifacts = config["artifacts"]
    paths = {name: project_path(relative) for name, relative in artifacts.items()}
    pending = [path for name, path in paths.items() if name != "protocol_manifest" and path.exists()]
    if pending:
        raise RuntimeError(f"成分账本输出已存在，禁止覆盖：{pending}")
    atomic_parquet_new(paths["ledger_panel"], ledger)
    atomic_parquet_new(paths["coverage_panel"], coverage)
    atomic_parquet_new(paths["identity_audit"], identity)

    summary = _scope_summary(ledger)
    identity_counts = {
        str(key): int(value)
        for key, value in identity["identity_status"].value_counts().items()
    }
    financial_counts = {
        str(key): int(value)
        for key, value in coverage["financial_coverage_status"].value_counts().items()
    }
    schedule_counts = {
        str(key): int(value)
        for key, value in schedule["schedule_status"].value_counts().items()
    }
    failed_identity = identity.loc[~identity["identity_pass"]]
    ledger_json = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "LEDGER_COMPLETE_EXACT_IDENTITIES_WITH_EXPLICIT_SOURCE_LIMITATIONS",
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "schedule_status_counts": schedule_counts,
        "scope_summary": dataframe_records(summary),
        "financial_coverage_status_counts": financial_counts,
        "identity_status_counts": identity_counts,
        "failed_or_no_view_identity_count": int(len(failed_identity)),
        "ledger_records": dataframe_records(ledger),
        "coverage_records": dataframe_records(coverage),
        "historical_official_weight_vintage_verified": False,
        "component_pure_dividend_archive_available": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    ledger_json["payload_sha256"] = canonical_hash(ledger_json)
    atomic_json_new(paths["ledger_json"], ledger_json)
    report = _build_report(manifest, ledger, coverage, identity)
    atomic_text_new(paths["final_report"], report)

    exact_pass_count = int(identity["identity_pass"].sum())
    status = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "LEDGER_COMPLETE_EXACT_IDENTITIES_WITH_EXPLICIT_SOURCE_LIMITATIONS",
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "ledger_panel": {
            "path": paths["ledger_panel"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["ledger_panel"]),
            "rows": len(ledger),
            "scope_counts": {
                str(key): int(value)
                for key, value in ledger["scope"].value_counts().items()
            },
        },
        "coverage_panel": {
            "path": paths["coverage_panel"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["coverage_panel"]),
            "rows": len(coverage),
            "financial_coverage_status_counts": financial_counts,
            "minimum_market_weight_coverage": float(
                coverage["market_weight_coverage"].min()
            ),
            "minimum_financial_weight_coverage": float(
                coverage["financial_weight_coverage"].min()
            ),
        },
        "identity_audit": {
            "path": paths["identity_audit"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["identity_audit"]),
            "rows": len(identity),
            "identity_pass_count": exact_pass_count,
            "identity_not_pass_count": int(len(identity) - exact_pass_count),
            "identity_status_counts": identity_counts,
            "tolerance": float(contract["identity_tolerance"]),
        },
        "component_admission": {
            "actual_index_dividend_component": "PASS_IMPLIED_FROM_TOTAL_RETURN_VS_PRICE_INDEX",
            "fixed_component_pure_dividend": "NO_VIEW_NO_COMPONENT_DISTRIBUTION_ARCHIVE",
            "fixed_component_distribution_and_corporate_action": "ADMITTED_BROADER_ACCOUNTING_COMPONENT",
            "fixed_financial_earnings_and_multiple": "PASS_OR_PARTIAL_BY_EXPLICIT_WEIGHT_COVERAGE",
            "membership_and_weight": "PARTIAL_INCLUDES_METHOD_AND_PROXY_WEIGHT_RESIDUAL",
            "tracking_residual": "PASS_ETF_CASH_DISTRIBUTION_ADJUSTED",
        },
        "historical_official_weight_vintage_verified": False,
        "mathematical_impossibility_proven": False,
        "accessible_free_information_feasibility_validated": False,
        "next_research_action": "TEST_CF_DR_RC_OWN_OBJECT_MEASUREMENT_VALIDITY",
        "portfolio_action": "ABSTAIN",
        "position_state": "POSITION_UNSET",
        "current_investable_strategy": "NONE",
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "nav_calculated": False,
        "sharpe_calculated": False,
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json_new(paths["authoritative_status"], status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 510300 总回报成分账本 V1")
    parser.add_argument("--phase", choices=["freeze", "build", "all", "verify"], default="all")
    args = parser.parse_args()
    config = load_config()
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"build", "all"}:
        run_ledger(config, manifest)
    if args.phase == "verify":
        print("总回报成分账本协议与全部冻结输入校验通过。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
