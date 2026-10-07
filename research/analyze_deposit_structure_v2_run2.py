"""解析全国存款年度表，作事后结构分解；不合并股票未来收益。"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_macro_volatility_mechanisms_v2_run2"


def table_rows(path: Path) -> list[list[str]]:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    return [[unicodedata.normalize("NFKC", c.get_text(" ", strip=True)) for c in row.find_all(["td", "th"], recursive=False)] for row in soup.select("tr")]


def parse(path: Path, url: str) -> list[dict]:
    rows = table_rows(path)
    header = next(row for row in rows if len(row) >= 13 and re.fullmatch(r"20\d{2}\.01", row[1]))
    months = [value.replace(".", "-") for value in header[1:13]]
    result = [{"stat_month": month, "source_path": str(path.relative_to(STUDY)), "source_url": url, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "vintage_status": "CURRENT_ANNUAL_SNAPSHOT_NOT_PIT", "historical_decision_admission": "EXCLUDED"} for month in months]
    scope = None
    for row_number, row in enumerate(rows, start=1):
        if len(row) < 13:
            continue
        label = re.sub(r"\s+", "", row[0])
        if "各项贷款" in label:
            break
        key = None
        if "住户存款" in label:
            key, scope = "household_total", "household"
        elif "非金融企业存款" in label:
            key, scope = "corporate_total", "corporate"
        elif "活期存款" in label and scope:
            key = scope + "_demand"
        elif "定期及其他存款" in label and scope:
            key = scope + "_time_other"
        elif "机关团体存款" in label:
            key, scope = "organization_total", None
        elif "财政性存款" in label:
            key = "fiscal_total"
        elif "非银行业金融机构存款" in label:
            key = "nonbank_total"
        if key:
            for record, value in zip(result, row[1:13], strict=True):
                record[key + "_100m"] = float(value.replace(",", ""))
                record[key + "_source_row"] = row_number
    assert len(result) == 12
    return result


def run() -> None:
    receipts = json.loads((STUDY / "sources/source_receipts.json").read_text(encoding="utf-8"))
    rows = []
    for year in [2023, 2024, 2025]:
        source = next(row for row in receipts if row["id"] == f"deposit_{year}")
        assert source["status"] in ["FETCHED", "REUSED_SAVED"]
        rows.extend(parse(STUDY / source["local_path"], source["url"]))
    data = pd.DataFrame(rows).sort_values("stat_month").reset_index(drop=True)
    checks = {}
    for sector in ["household", "corporate"]:
        difference = data[f"{sector}_total_100m"] - data[f"{sector}_demand_100m"] - data[f"{sector}_time_other_100m"]
        checks[sector + "_rounding_error_max_100m"] = float(difference.abs().max())
        assert difference.abs().max() <= 0.011, "原表分项和与总数不符"
        data[sector + "_demand_share"] = data[f"{sector}_demand_100m"] / data[f"{sector}_total_100m"]
    columns = [col for col in data if col.endswith("_100m")]
    for col in columns:
        data["month_change_" + col] = data[col].diff().round(2)
        data["yoy_change_" + col] = data[col].diff(12).round(2)
    facts = pd.read_csv(STUDY / "results/104个月_余额基数分解_不含未来标签.csv")
    data = data.merge(facts[["stat_month", "training_regime", "m1_balance_100m", "m0_balance_100m", "m1_ex_m0_100m"]], on="stat_month", how="left", validate="one_to_one")
    data["old_m1_unit_demand_less_corporate_100m"] = (data.m1_ex_m0_100m - data.corporate_demand_100m).where(data.training_regime.str.startswith("M1_OLD"))
    data["new_m1_mixed_residual_100m"] = (data.m1_ex_m0_100m - data.corporate_demand_100m - data.household_demand_100m).where(data.training_regime == "M1_NEW2025")
    data["residual_warning"] = "公告M1/M0为百亿精度，企业/住户表为当前年度快照；残差混合其他单位活期、支付备付金及口径/修订差异，不是独立测得分项"
    data.to_csv(STUDY / "results/全国存款结构_2023至2025_事后快照.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    indexed = data.set_index("stat_month")
    windows = [("2023-12", "2024-03"), ("2024-03", "2024-06"), ("2024-06", "2024-09"), ("2024-09", "2024-12"), ("2023-12", "2024-12"), ("2024-12", "2025-03"), ("2025-03", "2025-06"), ("2025-06", "2025-09"), ("2025-09", "2025-12"), ("2024-12", "2025-12")]
    records = []
    for start, end in windows:
        a, b = indexed.loc[start], indexed.loc[end]
        row = {"start_month": start, "end_month": end, "vintage_status": "CURRENT_ANNUAL_SNAPSHOT_NOT_PIT"}
        for col in columns:
            row["change_" + col] = round(b[col] - a[col], 2)
        row["corporate_demand_share_start"] = a.corporate_demand_share
        row["corporate_demand_share_end"] = b.corporate_demand_share
        row["m1_change_100m"] = b.m1_balance_100m - a.m1_balance_100m if a.training_regime == b.training_regime else np.nan
        row["m0_change_100m"] = b.m0_balance_100m - a.m0_balance_100m
        row["m1_ex_m0_change_100m"] = b.m1_ex_m0_100m - a.m1_ex_m0_100m if a.training_regime == b.training_regime else np.nan
        row["causal_identification"] = "NOT_IDENTIFIED_AGGREGATE_STOCK_DIFFERENCES"
        records.append(row)
    periods = pd.DataFrame(records)
    periods.to_csv(STUDY / "results/全国存款结构_固定季度与年度变化.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    (STUDY / "verification_deposits.json").write_text(json.dumps({"status": "PASS", "monthly_rows": len(data), "identity_checks": checks, "historical_decision_admission": "EXCLUDED", "interpretation": "余额差可分解总变化，不能追踪同一笔钱从活期搬向何处。"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(periods[["start_month", "end_month", "change_corporate_demand_100m", "change_corporate_time_other_100m", "change_corporate_total_100m", "m1_ex_m0_change_100m"]].to_string(index=False))


if __name__ == "__main__":
    run()
