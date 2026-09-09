"""下载并审计中国月度货币供应量；仅用于宏观发现研究。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "data" / "raw" / "macro" / "china_money_supply_monthly.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "china_money_supply_monthly_status.json"

NORMALIZED_COLUMNS = [
    "month_raw",
    "m2_stock_100m_cny",
    "m2_yoy_pct",
    "m2_mom_pct",
    "m1_stock_100m_cny",
    "m1_yoy_pct",
    "m1_mom_pct",
    "m0_stock_100m_cny",
    "m0_yoy_pct",
    "m0_mom_pct",
]

PRIMARY_REFERENCES = [
    {
        "description": "人民银行2018年统计报告：2018年调整M2货币市场基金口径，并回溯2017年月度增速",
        "url": "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/370befa3ce2042dfbc414b74f3b397b7/index.html",
    },
    {
        "description": "人民银行2020年货币供应量表",
        "url": "https://www.pbc.gov.cn/diaochatongjisi/fileDir/resource/cms/2021/01/2021011909524819782.pdf",
    },
    {
        "description": "人民银行2024年上半年金融统计数据报告",
        "url": "https://xining.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212554139432/index.html",
    },
    {
        "description": "人民银行2024年第四季度货币政策执行报告",
        "url": "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/5587716/2025022618190099812.pdf",
    },
]

OFFICIAL_CHECKPOINTS = [
    {"month": "2017-01", "column": "m2_yoy_pct", "expected": 10.7, "tolerance": 0.06},
    {"month": "2017-12", "column": "m2_yoy_pct", "expected": 8.1, "tolerance": 0.06},
    {"month": "2020-12", "column": "m2_stock_100m_cny", "expected": 2186795.89, "tolerance": 0.1},
    {"month": "2020-12", "column": "m2_yoy_pct", "expected": 10.1, "tolerance": 0.06},
    {"month": "2024-06", "column": "m2_stock_100m_cny", "expected": 3050161.54, "tolerance": 0.1},
    {"month": "2024-06", "column": "m2_yoy_pct", "expected": 6.2, "tolerance": 0.06},
    {"month": "2024-12", "column": "m2_stock_100m_cny", "expected": 3135322.30, "tolerance": 0.1},
    {"month": "2024-12", "column": "m2_yoy_pct", "expected": 7.3, "tolerance": 0.06},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_month(value: object) -> pd.Timestamp:
    numbers = re.findall(r"\d+", str(value))
    if len(numbers) < 2:
        raise ValueError(f"无法识别月份：{value!r}")
    year = int(numbers[0])
    month = int(numbers[1])
    if year < 1990 or not 1 <= month <= 12:
        raise ValueError(f"月份超出合理范围：{value!r}")
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def normalize_money_supply(raw: pd.DataFrame, retrieved_at: datetime) -> pd.DataFrame:
    """按接口稳定列序标准化；月份只依赖数字，避免供应商中文编码影响。"""

    if raw.shape[1] < len(NORMALIZED_COLUMNS):
        raise ValueError(f"货币供应量接口列数不足：{raw.shape[1]}")
    result = raw.iloc[:, : len(NORMALIZED_COLUMNS)].copy()
    result.columns = NORMALIZED_COLUMNS
    result["month"] = result["month_raw"].map(_parse_month)
    numeric_columns = NORMALIZED_COLUMNS[1:]
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="coerce")
    result = result.loc[result["month"].ge(pd.Timestamp("2017-01-31"))].copy()
    result = result.dropna(subset=["month", "m2_stock_100m_cny", "m2_yoy_pct"])
    result = result.sort_values("month").drop_duplicates("month", keep="last")
    if result["month"].duplicated().any():
        raise ValueError("货币供应量月份重复")
    if not (result["m2_stock_100m_cny"] > 0).all():
        raise ValueError("M2余额存在非正值")
    result["source"] = "sina.macro.via_akshare.macro_china_money_supply"
    result["retrieved_at"] = retrieved_at
    return result[
        [
            "month",
            "m2_stock_100m_cny",
            "m2_yoy_pct",
            "m2_mom_pct",
            "m1_stock_100m_cny",
            "m1_yoy_pct",
            "m1_mom_pct",
            "m0_stock_100m_cny",
            "m0_yoy_pct",
            "m0_mom_pct",
            "source",
            "retrieved_at",
        ]
    ].reset_index(drop=True)


def check_official_checkpoints(data: pd.DataFrame) -> list[dict[str, object]]:
    indexed = data.set_index(data["month"].dt.to_period("M"))
    results: list[dict[str, object]] = []
    for checkpoint in OFFICIAL_CHECKPOINTS:
        period = pd.Period(str(checkpoint["month"]), freq="M")
        if period not in indexed.index:
            raise ValueError(f"缺少官方抽查月份：{period}")
        actual = float(indexed.loc[period, str(checkpoint["column"])])
        expected = float(checkpoint["expected"])
        difference = actual - expected
        passed = abs(difference) <= float(checkpoint["tolerance"])
        results.append(
            {
                **checkpoint,
                "actual": actual,
                "difference": difference,
                "passed": passed,
            }
        )
    return results


def main() -> int:
    retrieved_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    raw = ak.macro_china_money_supply()
    data = normalize_money_supply(raw, retrieved_at)
    checkpoints = check_official_checkpoints(data)
    complete_months = pd.period_range(data["month"].min(), data["month"].max(), freq="M")
    observed_months = set(data["month"].dt.to_period("M"))
    missing_months = [str(period) for period in complete_months if period not in observed_months]
    status = "PASS_DISCOVERY_ONLY" if not missing_months and all(item["passed"] for item in checkpoints) else "FAILED"
    if len(data) < 100:
        status = "FAILED"

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_FILE, index=False)
    report = {
        "status": status,
        "purpose": "宏观大因子发现；不是现行510300策略输入",
        "retrieved_at": retrieved_at.isoformat(),
        "source": "AKShare转接新浪宏观货币供应量",
        "primary_source_spot_checks": PRIMARY_REFERENCES,
        "rows": int(len(data)),
        "first_month": str(data["month"].min().date()),
        "last_month": str(data["month"].max().date()),
        "missing_months": missing_months,
        "official_checkpoint_results": checkpoints,
        "known_methodology": {
            "2018_revision": "2018年1月人民银行调整货币市场基金统计方法，并回溯更新2017年M2增速。",
            "evaluation_rule": "正式发现检验从2018年8月后开始，使6个月变化的两端均处于2018新口径。",
            "vintage_limitation": "本文件是当前历史版本，不是逐月公告原始快照；只能支持DISCOVERY_ONLY，不能直接授权Paper或实盘。",
        },
        "availability_rule_for_research": "每个统计月统一延迟到次月20日，之后首个交易日收盘形成信号，下一交易日开盘执行。",
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)).replace("\\", "/"),
        "output_sha256": sha256_file(OUTPUT_FILE),
        "formal_point_in_time_evidence": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "live_trading_authorized": False,
    }
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "rows": len(data), "last_month": report["last_month"]}, ensure_ascii=False))
    return 0 if status == "PASS_DISCOVERY_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
