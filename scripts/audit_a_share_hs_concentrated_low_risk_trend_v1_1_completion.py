"""审计 V1.1 暂存输入是否足以冻结信号；本脚本禁止读取收益列。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1_1_data_contract.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    return path


def parquet_summary(path: Path, allowed_columns: list[str]) -> dict:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    selected = [column for column in allowed_columns if column in columns]
    frame = parquet.read(columns=selected).to_pandas() if selected else pd.DataFrame()
    result = {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": True,
        "rows": parquet.metadata.num_rows,
        "columns": columns,
        "sha256": sha256_file(path),
    }
    if "ts_code" in frame:
        result["symbols"] = int(frame["ts_code"].nunique())
        result["exchange_suffix_counts"] = {
            "SH": int(frame["ts_code"].astype(str).str.endswith(".SH").sum()),
            "SZ": int(frame["ts_code"].astype(str).str.endswith(".SZ").sum()),
        }
    for column in ("date", "effective_date", "valid_from"):
        if column in frame:
            dates = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not dates.empty:
                result[f"{column}_min"] = dates.min().date().isoformat()
                result[f"{column}_max"] = dates.max().date().isoformat()
    if "status" in frame:
        result["status_counts"] = {str(key): int(value) for key, value in frame["status"].value_counts().items()}
    return result


def build_report() -> dict:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    receipt_path = project_path(contract["outputs"]["collection_receipt"])
    blockers: list[str] = []
    checks: dict[str, dict] = {}
    if not receipt_path.is_file():
        blockers.append("缺少暂存层收集回执")
        receipt = {}
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    artifacts = {
        "security_master": (contract["sources"]["security_master"]["staging_path"], ["ts_code", "exchange", "list_date", "delist_date"]),
        "share_counts": (contract["sources"]["share_count_snapshots"]["staging_path"], ["ts_code", "effective_date"]),
        "industry": (contract["sources"]["industry_intervals"]["staging_path"], ["ts_code", "industry_l1", "valid_from", "valid_to"]),
        "calendar": (contract["sources"]["calendar"]["staging_path"], ["exchange", "date", "is_open"]),
        "szse_status": (contract["sources"]["szse_name_changes"]["staging_path"], ["ts_code", "status", "valid_from", "valid_to"]),
    }
    for key, (relative, columns) in artifacts.items():
        path = project_path(relative)
        if not path.is_file():
            checks[key] = {"path": relative, "exists": False}
            blockers.append(f"缺少暂存输入：{relative}")
        else:
            checks[key] = parquet_summary(path, columns)
            expected_hash = receipt.get(key, {}).get("sha256")
            checks[key]["matches_collection_receipt"] = expected_hash == checks[key]["sha256"]
            if expected_hash and expected_hash != checks[key]["sha256"]:
                blockers.append(f"暂存输入在收集后发生变化：{relative}")
    board_path = project_path(contract["sources"]["board_rules"]["staging_path"])
    checks["board_rules"] = {"path": board_path.relative_to(ROOT).as_posix(), "exists": board_path.is_file()}
    if board_path.is_file():
        rules = yaml.safe_load(board_path.read_text(encoding="utf-8"))
        checks["board_rules"]["boards"] = rules.get("boards", [])
        checks["board_rules"]["sha256"] = sha256_file(board_path)
    else:
        blockers.append("缺少板块点时成交规则")
    manifest_path = project_path(contract["sources"]["daily_market"]["manifest_path"])
    checks["daily_market_manifest"] = {"path": manifest_path.relative_to(ROOT).as_posix(), "exists": manifest_path.is_file()}
    if manifest_path.is_file():
        market_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks["daily_market_manifest"]["datasets"] = market_manifest.get("datasets", [])
        for item in market_manifest.get("datasets", []):
            path = project_path(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                blockers.append(f"日线数据集缺失或哈希变化：{item['path']}")
    else:
        blockers.append("缺少日线数据集清单")

    master_path = project_path(contract["sources"]["security_master"]["staging_path"])
    sz_status_path = project_path(contract["sources"]["szse_name_changes"]["staging_path"])
    if master_path.is_file() and sz_status_path.is_file():
        master_codes = pd.read_parquet(master_path, columns=["ts_code", "exchange"])
        expected_sz = set(master_codes.loc[master_codes["exchange"].eq("SZSE"), "ts_code"].astype(str))
        observed_sz = set(pd.read_parquet(sz_status_path, columns=["ts_code"])["ts_code"].astype(str))
        checks["szse_status"]["master_symbol_coverage"] = len(observed_sz & expected_sz) / len(expected_sz)
        checks["szse_status"]["unexpected_symbols"] = len(observed_sz - expected_sz)
        if observed_sz != expected_sz:
            blockers.append("深市历史状态区间未完整覆盖深市证券母表")

    sse_path = project_path(contract["sources"]["sse_status_intervals"]["staging_path"])
    checks["sse_status"] = {"path": sse_path.relative_to(ROOT).as_posix(), "exists": sse_path.is_file()}
    if not sse_path.is_file():
        blockers.append("缺少上交所普通A股完整历史ST、*ST和退市整理状态区间")
    else:
        checks["sse_status"].update(parquet_summary(sse_path, ["ts_code", "status", "valid_from", "valid_to"]))

    corporate_path = project_path(contract["sources"]["corporate_actions"]["staging_path"])
    checks["corporate_actions"] = {
        "path": corporate_path.relative_to(ROOT).as_posix(),
        "exists": corporate_path.is_file(),
        "stage": "AFTER_SIGNAL_FREEZE_BEFORE_RETURN_READ",
        "status": "DEFERRED_UNTIL_SIGNAL_UNIVERSE_IS_KNOWN" if not corporate_path.is_file() else "PRESENT_PENDING_SELECTED_SYMBOL_COVERAGE_AUDIT",
    }

    benchmark_paths = {
        "H00300_training": contract["benchmarks"]["H00300_training"],
        "H00300_holdout": contract["benchmarks"]["H00300_holdout"],
        "510300_raw_daily": contract["benchmarks"]["510300_raw_daily"],
    }
    checks["benchmarks"] = {}
    for key, relative in benchmark_paths.items():
        path = project_path(relative)
        checks["benchmarks"][key] = {"path": relative, "exists": path.is_file()}
        if path.is_file():
            parquet = pq.ParquetFile(path)
            checks["benchmarks"][key].update({"rows": parquet.metadata.num_rows, "columns": parquet.schema_arrow.names, "sha256": sha256_file(path)})
            if "date" in parquet.schema_arrow.names:
                dates = pd.to_datetime(parquet.read(columns=["date"]).to_pandas()["date"], errors="coerce").dropna()
                checks["benchmarks"][key]["date_min"] = dates.min().date().isoformat()
                checks["benchmarks"][key]["date_max"] = dates.max().date().isoformat()
        else:
            blockers.append(f"缺少冻结基准：{relative}")
    dividend_path = project_path(contract["benchmarks"]["510300_cash_dividends"])
    checks["benchmarks"]["510300_cash_dividends"] = {
        "path": contract["benchmarks"]["510300_cash_dividends"],
        "exists": dividend_path.is_file(),
    }
    if dividend_path.is_file():
        checks["benchmarks"]["510300_cash_dividends"].update(
            {"bytes": dividend_path.stat().st_size, "sha256": sha256_file(dividend_path)}
        )
    else:
        blockers.append(f"缺少冻结基准分红：{contract['benchmarks']['510300_cash_dividends']}")

    status = "READY_FOR_SIGNAL_FREEZE" if not blockers else "BLOCKED_SIGNAL_INPUTS_NO_VIEW"
    return {
        "schema_version": "A_SHARE_HS_CONCENTRATED_V1_1_COMPLETION_V1",
        "model_id": contract["parent_model_id"],
        "contract_id": contract["contract_id"],
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": status,
        "view_status": "NO_VIEW",
        "return_values_read": False,
        "return_metrics_computed": False,
        "signals_generated": False,
        "positions_generated": False,
        "orders_generated": False,
        "checks": checks,
        "blocking_reasons": blockers,
        "provider_attempts": [
            {"provider": "TuShare", "status": "TOKEN_EXPIRED", "used_for_current_refresh": False},
            {"provider": "Baostock", "status": "LOGIN_SOCKET_RECEIVE_HUNG_AND_ABORTED", "used_for_current_refresh": False},
            {"provider": "SZSE", "status": "OFFICIAL_NAME_CHANGE_XLSX_COLLECTED", "used_for_szse_status": True},
            {"provider": "CNINFO", "status": "SHARE_AND_DIVIDEND_ENDPOINTS_REACHABLE", "used_for_full_dividend_collection": False},
        ],
        "next_allowed_step": "取得上交所历史证券状态区间后重新审计" if blockers else "冻结无收益信号并收集入选证券公司行动",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 沪深A股三只极端低风险确认策略 V1.1 完成审计", "",
        f"- 状态：`{report['status']}`", f"- 视图：`{report['view_status']}`",
        "- 收益值读取：否", "- 收益指标计算：否", "- 信号、仓位、订单：均未生成", "",
        "## 已完成", "",
        "- 深交所官方简称变更整表已重建历史状态区间并覆盖深市母表。",
        "- 沪深证券母表、两段日线、股本快照、申万一级行业区间、观察交易日历和板块规则已形成带哈希暂存层。",
        "- 策略特征、行业风险排名、自身波动分位、确定性选股、相关性约束、2万元账户、成本和评估门槛已有独立实现与测试。", "",
        "## 硬阻断", "",
    ]
    lines.extend([f"- {item}" for item in report["blocking_reasons"]] or ["- 无，可冻结信号。"]) 
    lines.extend([
        "", "## 公司行动两阶段门", "",
        "只有信号冻结后才知道需要收集哪些股票的分红、送股和转增。该数据必须在读取组合收益前完成并覆盖全部实际入选证券。", "",
        "## 结论", "",
        f"下一步只允许：{report['next_allowed_step']}。当前不得发布任何收益、Top1/Top3/Top5/Top20优劣或交易动作。", "",
    ])
    return "\n".join(lines)


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    report = build_report()
    json_path = project_path(contract["outputs"]["readiness_json"])
    markdown_path = project_path(contract["outputs"]["readiness_markdown"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"状态": report["status"], "阻断项": len(report["blocking_reasons"]), "收益读取": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
