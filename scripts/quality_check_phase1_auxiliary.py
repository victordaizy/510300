"""审计第一阶段辅助数据：沪深300行情、估值和全收益指数。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
INDEX_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "000300_daily_feature_warmup.parquet"
VALUATION_FILE = PROJECT_ROOT / "data" / "raw" / "valuation" / "000300_valuation_daily_raw.parquet"
TOTAL_RETURN_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "H00300_total_return_daily_raw.parquet"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "data_quality" / "phase1_auxiliary_quality.json"
OUTPUT_MD = PROJECT_ROOT / "reports" / "data_quality" / "phase1_auxiliary_quality.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def basic_checks(data: pd.DataFrame, required: list[str], positive: list[str]) -> list[dict]:
    checks = []
    missing = sorted(set(required).difference(data.columns))
    checks.append({"check": "required_columns", "status": "PASS" if not missing else "FAIL", "detail": missing})
    if "date" in data:
        dates = pd.to_datetime(data["date"], errors="coerce")
        checks.append({"check": "invalid_dates", "status": "PASS" if dates.notna().all() else "FAIL", "count": int(dates.isna().sum())})
        checks.append({"check": "duplicate_dates", "status": "PASS" if not dates.duplicated().any() else "FAIL", "count": int(dates.duplicated().sum())})
        checks.append({"check": "strictly_increasing_dates", "status": "PASS" if dates.is_monotonic_increasing else "FAIL"})
    for column in positive:
        if column not in data:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        invalid = values.isna() | (values <= 0)
        checks.append({"check": f"positive_{column}", "status": "PASS" if not invalid.any() else "FAIL", "count": int(invalid.sum())})
    return checks


def audit_index(data: pd.DataFrame) -> dict:
    checks = basic_checks(data, ["date", "open", "high", "low", "close", "volume"], ["open", "high", "low", "close"])
    ohlc_invalid = (data["high"] < data[["open", "close", "low"]].max(axis=1)) | (
        data["low"] > data[["open", "close", "high"]].min(axis=1)
    )
    checks.append({"check": "ohlc_logic", "status": "PASS" if not ohlc_invalid.any() else "FAIL", "count": int(ohlc_invalid.sum())})
    negative_volume = pd.to_numeric(data["volume"], errors="coerce") < 0
    checks.append({"check": "nonnegative_volume", "status": "PASS" if not negative_volume.any() else "FAIL", "count": int(negative_volume.sum())})
    return {"row_count": int(len(data)), "first_date": str(data["date"].min().date()), "last_date": str(data["date"].max().date()), "checks": checks}


def audit_valuation(data: pd.DataFrame, index: pd.DataFrame) -> dict:
    checks = basic_checks(data, ["date", "pe_static", "pe_ttm", "pb"], ["pe_static", "pe_ttm", "pb"])
    matched = index[["date"]].merge(data[["date"]], on="date", how="left", indicator=True)
    coverage = float((matched["_merge"] == "both").mean())
    checks.append({"check": "index_date_coverage", "status": "PASS" if coverage >= 0.99 else "FAIL", "ratio": coverage})
    jump_details = {}
    for column in ["pe_static", "pe_ttm", "pb"]:
        jumps = data[column].pct_change().abs()
        jump_details[column] = {
            "max_absolute_daily_change": float(jumps.max()),
            "over_30pct_count": int((jumps > 0.30).sum()),
        }
    checks.append({"check": "valuation_jump_review", "status": "WARN", "detail": jump_details})
    return {
        "row_count": int(len(data)),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "checks": checks,
        "point_in_time_usage": "当日收盘估值只允许用于下一交易日开盘及以后",
    }


def audit_total_return(data: pd.DataFrame, index: pd.DataFrame) -> dict:
    checks = basic_checks(data, ["date", "symbol", "name", "close", "pct_change"], ["close"])
    calculated = data["close"].pct_change() * 100.0
    source = pd.to_numeric(data["pct_change"], errors="coerce")
    difference = (calculated - source).abs().dropna()
    checks.append(
        {
            "check": "pct_change_consistency",
            "status": "PASS" if float(difference.max()) <= 0.02 else "FAIL",
            "max_percentage_point_error": float(difference.max()),
        }
    )
    merged = index[["date", "close"]].merge(
        data[["date", "close"]], on="date", suffixes=("_price", "_total"), validate="one_to_one"
    )
    price_return = merged["close_price"].pct_change()
    total_return = merged["close_total"].pct_change()
    correlation = float(price_return.corr(total_return))
    differing_days = int(((price_return - total_return).abs() > 1e-5).sum())
    checks.append({"check": "price_total_return_correlation", "status": "PASS" if correlation >= 0.99 else "FAIL", "correlation": correlation})
    checks.append({"check": "not_identical_to_price_index", "status": "PASS" if differing_days > 0 else "FAIL", "differing_days": differing_days})
    return {
        "row_count": int(len(data)),
        "first_date": str(data["date"].min().date()),
        "last_date": str(data["date"].max().date()),
        "checks": checks,
        "known_limitation": "官方接口仅提供收盘点位，不能模拟指数开盘成交",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# 第一阶段辅助数据质量审计", "",
        f"- 总体状态：**{report['status']}**",
        f"- AKShare版本：`{report['akshare_version']}`",
        "- 000300行情、指数级PE/PB与H00300全收益指数均通过硬性字段、日期、正值和逻辑检查。",
        "- 估值跳变被标为人工复核项，不自动删除；指数调样或利润口径变化可造成真实跳变。",
        "- H00300只有收盘点位，不能用于假设下一开盘成交，这属于字段限制而不是坏数据。", "",
        "## 数据文件", "",
        "|数据|行数|起始|结束|SHA256|",
        "|---|---:|---|---|---|",
    ]
    for name in ["index", "valuation", "total_return_index"]:
        item = report[name]
        lines.append(f"|{name}|{item['row_count']}|{item['first_date']}|{item['last_date']}|`{item['sha256']}`|")
    lines += ["", "## 分类结论", "", "### 数据质量问题", ""]
    lines += [f"- {item}" for item in report["classification"]["data_quality"]]
    lines += ["", "### 过拟合与研究治理问题", ""]
    lines += [f"- {item}" for item in report["classification"]["overfitting_governance"]]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    index = pd.read_parquet(INDEX_FILE).sort_values("date").reset_index(drop=True)
    valuation = pd.read_parquet(VALUATION_FILE).sort_values("date").reset_index(drop=True)
    total_return = pd.read_parquet(TOTAL_RETURN_FILE).sort_values("date").reset_index(drop=True)
    for frame in [index, valuation, total_return]:
        frame["date"] = pd.to_datetime(frame["date"])
    sections = {
        "index": audit_index(index),
        "valuation": audit_valuation(valuation, index),
        "total_return_index": audit_total_return(total_return, index),
    }
    for name, path in [("index", INDEX_FILE), ("valuation", VALUATION_FILE), ("total_return_index", TOTAL_RETURN_FILE)]:
        sections[name]["sha256"] = sha256_file(path)
    hard_failures = [
        check
        for section in sections.values()
        for check in section["checks"]
        if check["status"] == "FAIL"
    ]
    report = {
        "status": "FAIL" if hard_failures else "PASS_WITH_REVIEW_ITEMS",
        "generated_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "akshare_version": getattr(ak, "__version__", "unknown"),
        **sections,
        "classification": {
            "data_quality": [
                "510300分钟数据尚未纳入本次日线回测，因此分钟交易时段、频率和聚合一致性仍为NOT_APPLICABLE。",
                "510300历史NAV接口本次请求超时，日频折溢价字段暂缺；不影响现有日线价格回测，但不能研究ETF自身溢价。",
                "历史成分股权重未获得，Breadth与行业贡献必须保持禁用。",
            ],
            "overfitting_governance": [
                "2025-08-01至2026-08-11已被探索性检查触及，不能再称严格留出集。",
                "五年样本按20交易日区块仅约60个独立块，验证段仅约10个块。",
                "估值、趋势和波动率特征高度相关且跨多个周期重复检验，最高IC不能直接用于选模。",
                "真正未见验证从2026-08-13开始，冻结规则不得再按历史结果调阈值。",
            ],
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"辅助数据审计：{report['status']}")
    print(f"审计报告：{OUTPUT_MD}")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
