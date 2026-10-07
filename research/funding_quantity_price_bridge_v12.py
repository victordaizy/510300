"""连接货币数量、融资价格与股票既有定价；保留原月度观察时钟。"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_funding_quantity_price_bridge_v12"
INPUTS = {
    "monthly.csv": "reports/research/510300_macro_transmission_context_v4/results/104个月_多层证据与原后续路径.csv",
    "fixing.parquet": "reports/research/510300_repo_fixing_segmentation_source_v1/fixing_segmentation.parquet",
    "policy.parquet": "reports/research/510300_macro_transmission_context_v4/inputs/policy_rate.parquet",
    "bond.parquet": "reports/research/510300_macro_transmission_context_v4/inputs/bond_yield.parquet",
    "valuation.parquet": "reports/research/510300_macro_transmission_context_v4/inputs/valuation.parquet",
    "previous.json": "reports/research/510300_holiday_money_credit_bridge_v11/result.json",
}
SOURCES = {
    "2020年第四季度货币政策报告_发布页.html": "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/4021036/8d39aa9d730046c896289d17c09346d2/index.html",
    "2020年第四季度货币政策执行报告.pdf": "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/4021036/8d39aa9d730046c896289d17c09346d2/2021020821282167078.pdf",
    "中国货币网_回购定盘定义.html": "https://www.chinamoney.com.cn/chinese/bkfrr/",
}
CASES = ["2020-06", "2021-01", "2024-08", "2025-07", "2025-08"]


def stamp():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    for part in ["inputs", "sources", "results", "code", "figures"]:
        (OUT / part).mkdir(parents=True, exist_ok=True)
    if (OUT / "freeze.json").exists():
        return
    protocol = {
        "at": stamp(), "study_id": "510300_FUNDING_QUANTITY_PRICE_BRIDGE_V12",
        "previous_turn_classification": "PROGRESS：V11金额、春节时点和期限结构已复算并交付。",
        "question": "货币信贷数量改善时，资金成本、机构分层和长期利率是否也同向改善？",
        "scope": "原104个月逐点补充当时可见融资价格；五个既有病例并列，不按新结果选案例。",
        "cases": CASES, "selection": "历史结果已知，本轮是机制解释，不是独立验证或新选股择时检验。",
        "fixing": "FDR007与FR007分别是银银间与银行间上午定盘；既有档案按当日12点计划可见，历史首版送达未认证；不冒充全天加权DR007、R007。",
        "funding_windows": "每个观察点取最近已可见的20条定盘记录，另取之前20条不重叠记录；为银行间记录日，不冒充A股交易日。",
        "quantity_vs_price": "FDR007减同期已公布7天逆回购利率；FR007减FDR007是不同样本定盘差，不是纯非银融资成本。",
        "bond": "沿用既有下一A股开盘可见口径；最新值与20条之前值对照，不回填当日尚不可见收盘值。",
        "valuation": "只沿用原始PE及其计划可见时钟；100/PE减10年国债只称收益率差代理，不等于预期权益风险溢价。",
        "policy_pdf": "2020年第四季度报告2021年2月8日22:20:39公开，先于本例2月9日21点；当前下载版不等于历史首版认证。",
        "no_new_return_groups": True, "new_models": 0, "new_accounts": 0, "orders_authorized": False,
        "goal_achieved": False, "independent_validation": False,
    }
    save("protocol.json", protocol)
    (ROOT / "config/510300_funding_quantity_price_bridge_v12.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    files = []
    for name, source in INPUTS.items():
        shutil.copy2(ROOT / source, OUT / "inputs" / name)
        files.append({"name": name, "source": source, "sha256": digest(OUT / "inputs" / name)})
    save("freeze.json", {"at": stamp(), "protocol_sha256": digest(OUT / "protocol.json"), "files": files})
    print("已固定原观察点、资金口径及比较窗口。", flush=True)


def sources():
    receipts = []
    for name, url in SOURCES.items():
        path = OUT / "sources" / name
        record = path.with_suffix(path.suffix + ".receipt.json")
        if record.exists():
            receipts.append(json.loads(record.read_text(encoding="utf-8")))
            continue
        response = requests.get(url, timeout=(15, 45), headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        if name.endswith(".pdf"):
            assert response.content.startswith(b"%PDF")
        path.write_bytes(response.content)
        rec = {"name": name, "url": url, "retrieved_at": stamp(), "sha256": digest(path), "historical_first_version_authenticated": False}
        record.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        receipts.append(rec)
    save("source_receipts.json", receipts)
    pdf_path = OUT / "sources/2020年第四季度货币政策执行报告.pdf"
    with pdfplumber.open(pdf_path) as pdf:
        pages = [{"page": i + 1, "text": p.extract_text() or ""} for i, p in enumerate(pdf.pages)]
    save("sources/政策报告逐页文本.json", pages)
    selected = [p for p in pages if any(k in p["text"].replace(" ", "") for k in ["不应过度关注", "4.61%", "35.2%", "5.03%", "不宜过度关注"])]
    save("results/报告机制定位页.json", selected)
    print(json.dumps({"原文数": len(receipts), "报告页数": len(pages), "定位页": [p["page"] for p in selected]}, ensure_ascii=False), flush=True)


def build():
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("已保存的本轮计算不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for f in frozen["files"]:
        assert digest(OUT / "inputs" / f["name"]) == f["sha256"]
    monthly = pd.read_csv(OUT / "inputs/monthly.csv")
    fixing = pd.read_parquet(OUT / "inputs/fixing.parquet").sort_values("date").reset_index(drop=True)
    fixing["available_at"] = pd.to_datetime(fixing.available_at, utc=True).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    fixing["date"] = pd.to_datetime(fixing.date)
    policy = pd.read_parquet(OUT / "inputs/policy.parquet").sort_values("published_at")
    policy["policy_known_at"] = pd.to_datetime(policy.published_at).dt.tz_localize("Asia/Shanghai").dt.as_unit("ns")
    fixing = pd.merge_asof(fixing.sort_values("available_at"), policy[["policy_known_at", "seven_day_rate_percent", "raw_path", "raw_sha256"]].rename(columns={"raw_path": "policy_raw_path", "raw_sha256": "policy_raw_sha256"}), left_on="available_at", right_on="policy_known_at", direction="backward")
    fixing["fdr_policy_gap_bp"] = 100 * (fixing.fdr007_percent - fixing.seven_day_rate_percent)
    fixing["fr_fdr_gap_bp"] = 100 * (fixing.fr007_percent - fixing.fdr007_percent)
    fixing.to_parquet(OUT / "results/逐日定盘与当时政策利率.parquet", index=False)
    bond = pd.read_parquet(OUT / "inputs/bond.parquet").sort_values("available_at").reset_index(drop=True)
    bond["available_at"] = pd.to_datetime(bond.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    bond["observation_date"] = pd.to_datetime(bond.observation_date)
    rows, details = [], []
    original_keys = ["stat_month", "published_at", "snapshot_at", "observation_date", "m1_yoy_pp", "m2_yoy_pp", "spread_pp", "delta3_spread_pp", "past_return20", "past_return60", "v_downside20", "valuation_original_pe_official", "valuation_period", "E0_20_entry_date", "E0_20_exit_date", "E0_20_return", "E1_20_return"]
    for _, original in monthly.iterrows():
        row = {k: original[k] for k in original_keys if k in original.index}
        snapshot = pd.Timestamp(original.snapshot_at)
        available = fixing[fixing.available_at.le(snapshot)]
        if len(available) < 40 or (snapshot.date() - available.iloc[-1].date.date()).days > 7:
            row["funding_status"] = "NO_VIEW_INSUFFICIENT_OR_STALE_FIXING"
        else:
            history = available.tail(40)
            recent, old = history.tail(20), history.head(20)
            latest = recent.iloc[-1]
            row.update(funding_status="AVAILABLE_DIAGNOSTIC_ONLY", fixing_date=latest.date.date().isoformat(), fixing_known_at=latest.available_at.isoformat(), funding_recent_start=recent.iloc[0].date.date().isoformat(), funding_previous_start=old.iloc[0].date.date().isoformat(), funding_previous_end=old.iloc[-1].date.date().isoformat())
            for col in ["fdr007_percent", "fr007_percent", "seven_day_rate_percent", "fdr_policy_gap_bp", "fr_fdr_gap_bp"]:
                row[col + "_latest"] = float(latest[col])
                row[col + "_recent20_mean"] = float(recent[col].mean())
                row[col + "_previous20_mean"] = float(old[col].mean())
                row[col + "_mean_change"] = float(recent[col].mean() - old[col].mean())
            assert history[["fdr007_percent", "fr007_percent", "seven_day_rate_percent"]].notna().all().all()
            for j, (_, item) in enumerate(history.iterrows()):
                details.append({"stat_month": original.stat_month, "window": "PREVIOUS20" if j < 20 else "RECENT20", "date": item.date.date().isoformat(), "available_at": item.available_at.isoformat(), "fdr007_percent": item.fdr007_percent, "fr007_percent": item.fr007_percent, "policy_rate_percent": item.seven_day_rate_percent, "fdr_policy_gap_bp": item.fdr_policy_gap_bp, "fr_fdr_gap_bp": item.fr_fdr_gap_bp, "raw_path": item.raw_path, "raw_sha256": item.raw_sha256})
        b = bond[bond.available_at.le(snapshot)]
        if len(b) > 20 and (snapshot.date() - b.iloc[-1].observation_date.date()).days <= 7:
            last, before = b.iloc[-1], b.iloc[-21]
            row.update(bond_status="AVAILABLE_DIAGNOSTIC_ONLY", bond_date=last.observation_date.date().isoformat(), bond_known_at=last.available_at.isoformat(), bond_10y_percent=float(last.china_10y_yield), bond_previous20_date=before.observation_date.date().isoformat(), bond_change20_bp=100 * float(last.china_10y_yield - before.china_10y_yield))
        else:
            row["bond_status"] = "NO_VIEW_INSUFFICIENT_OR_STALE_BOND"
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "results/104个月_货币数量与融资价格.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details).to_csv(OUT / "results/资金窗口明细.csv", index=False, encoding="utf-8-sig")
    cases = result[result.stat_month.isin(CASES)]
    cases.to_csv(OUT / "results/五个既有病例_融资与定价.csv", index=False, encoding="utf-8-sig")
    # 同日央行公告的月加权利率是独立层次，不能与上午定盘序列拼接。
    source = ROOT / "reports/research/510300_holiday_money_credit_bridge_v11/sources"
    for month in ["2020-12", "2021-01"]:
        shutil.copy2(source / (month + "_金融统计原文.html"), OUT / "sources" / (month + "_金融统计原文.html"))
    save("results/build_receipt.json", {"at": stamp(), "status": "FUNDING_QUANTITY_PRICE_CONTEXT_BUILT", "monthly_rows": len(result), "funding_status_counts": result.funding_status.value_counts().to_dict(), "bond_status_counts": result.bond_status.value_counts().to_dict(), "known_before_case_snapshot": True, "new_models": 0, "new_accounts": 0, "goal_achieved": False})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(cases[["stat_month", "fixing_date", "fdr_policy_gap_bp_latest", "fdr_policy_gap_bp_recent20_mean", "fdr_policy_gap_bp_mean_change", "fr_fdr_gap_bp_latest", "bond_10y_percent", "bond_change20_bp", "E0_20_return"]].to_json(orient="records", force_ascii=False), flush=True)


if __name__ == "__main__":
    freeze()
    sources()
    build()
