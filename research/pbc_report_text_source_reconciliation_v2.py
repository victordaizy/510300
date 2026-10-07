"""仅修正已保存央行原件的汉字年份/同义章节与日历起点，不重采集。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from research.pbc_report_text_case_alignment_v1 import clock_upper
from research.pbc_report_text_source_v1 import CN_QUARTERS, EXPECTED, POLICY_HEADINGS, compact, first_known_origins, similarity

ROOT = Path(__file__).absolute().parents[1]
SOURCE = ROOT / "reports/research/510300_pbc_report_text_source_v1"
OUT = ROOT / "reports/research/510300_pbc_report_text_source_reconciliation_v2"
CALENDAR = ROOT / "reports/research/510300_all_factor_macro_earnings_joint_v1/results/全部3488共同源视图_不足保留.parquet"
CASES = ROOT / "reports/research/510300_all_factor_joint_scorecard_v1/results/全部30历史案例_联合解释分与覆盖.parquet"
MANIFEST = SOURCE / "全部59季度_原件章节公布钟与未知.parquet"
YEAR_DIGITS = str.maketrans({"二": "2", "○": "0", "〇": "0", "零": "0", "Ｏ": "0", "一": "1",
                            "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"})
ECONOMIC_HEADINGS = ("第四部分宏观经济分析", "第四部分宏观经济形势")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def cover_identity(first_page, quarter):
    cover = compact(first_page)
    matches = re.findall(r"((?:20\d{2})|(?:[二○〇零Ｏ一三四五六七八九]{4}))年第([一二三四1234])季度", cover)
    identities = {f"{int(year.translate(YEAR_DIGITS))}Q{int(CN_QUARTERS.get(q, q))}" for year, q in matches}
    return identities == {quarter}


def split_sections(full_text):
    text = compact(full_text)
    economic_start, economic_heading = max((text.rfind(name), name) for name in ECONOMIC_HEADINGS)
    forecast_start = text.rfind("第五部分货币政策趋势")
    policy_start, policy_heading = max((text.rfind(name), name) for name in POLICY_HEADINGS)
    economic = text[economic_start:forecast_start] if 0 <= economic_start < forecast_start else None
    guidance = text[policy_start:] if forecast_start >= 0 and policy_start > forecast_start else None
    return {"economic": economic, "guidance": guidance,
            "economic_heading": economic_heading if economic else None,
            "policy_heading": policy_heading if guidance else None}


def bind_origins(reports, dates):
    """直接按真实公布上界和15:05排序，日历起点前以实际最新报告为准。"""
    if reports.quarter.tolist() != EXPECTED:
        raise ValueError("原59季度范围被删改。")
    reports = reports.copy()
    reports["publication_upper"] = [clock_upper(row) for row in reports.to_dict("records")]
    if reports.publication_upper.isna().any():
        raise ValueError("完整公布日历未知，不允许自动沿用旧报告。")
    reports = first_known_origins(reports, dates)
    reports["publication_timestamp"] = pd.to_datetime(reports.publication_upper, utc=True).dt.tz_convert("Asia/Shanghai")
    if reports.publication_timestamp.duplicated().any():
        raise ValueError("不同报告真实公布上界完全相同，须保留冲突。")
    fields = ["quarter", "status", "previous_quarter", "publication_upper", "publication_timestamp", "first_known_origin",
              "economic_similarity", "guidance_similarity", "historical_first_vintage", "pdf_path"]
    right = reports[fields].rename(columns={key: "pbc_" + key for key in fields}).sort_values("pbc_publication_timestamp")
    calendar = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    left = pd.DataFrame({"date": calendar, "origin_at": calendar.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)})
    result = pd.merge_asof(left, right, left_on="origin_at", right_on="pbc_publication_timestamp", direction="backward")
    result["pbc_two_channel_known"] = (result.pbc_status.eq("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS")
                                       & result.pbc_economic_similarity.notna() & result.pbc_guidance_similarity.notna())
    for part in ("economic", "guidance"):
        result["pbc_" + part + "_change"] = (1 - result["pbc_" + part + "_similarity"]).where(result.pbc_two_channel_known)
    result["pbc_report_age_calendar_days"] = (result.origin_at - result.pbc_publication_timestamp).dt.total_seconds() / 86400
    result["pbc_numeric_direction"] = "NONE_NOT_TRADING_SCORE"
    result["pbc_independent_validation"] = "NOT_ESTABLISHED"
    return reports, result.drop(columns="origin_at")


def freeze():
    original = pd.read_parquet(MANIFEST)
    inputs = [MANIFEST, SOURCE / "protocol.json", SOURCE / "summary.json", CALENDAR, CASES,
              ROOT / "research/pbc_report_text_source_v1.py", ROOT / "research/pbc_report_text_case_alignment_v1.py"]
    for row in original.to_dict("records"):
        if row["pdf_received"]:
            inputs += [ROOT / row["pdf_path"], SOURCE / "texts" / (row["quarter"] + "_full.txt")]
    save("protocol.json", {"at": now(), "registration": "TECH.R265", "decision": "TECH.R266",
          "hypothesis": "首版部分NO_VIEW由原件格式识别错误造成；修正应恢复已有章节，不能恢复真正缺失原件或产生预测优势。",
          "cause_evidence": "首版df8e38/7568b9保存原件证据；2011Q4/2015Q1/2024Q2封面已视觉核对；2015等章标题宏观经济形势与旧宏观经济分析同一第四部分。",
          "changes": ["只在首封面识别汉字四位年份及原数字年份", "接受第四部分宏观经济形势/宏观经济分析两个明确标题",
                      "首次原点不把两份不同公布时刻的旧报告误当同钟冲突；按真实公布上界取当时最新报告"],
          "unchanged": "原59季度、58已有原件、章节角色、100字符底线、立即前季比较、中文二元余弦、真实公布日末/15:05、原30案例与3488日、目标与全部冻结策略",
          "new_network_requests": 0, "source_failure": "2025Q4 TLS原失败保留，不重试、不补原件或前季相似度、不回退旧文本。",
          "original_run_archived_unchanged": True, "new_accounts": 0, "new_fits": 0, "new_labels": 0,
          "financial_admission": "NOT_ADMITTED_PREDICTIVE_METHOD_NOT_REGISTERED", "first_vintage_authenticated": False,
          "goal_achieved": False, "orders_authorized": False,
          "code_sha256": digest(Path(__file__)),
          "input_sha256": {path.absolute().relative_to(ROOT).as_posix(): digest(path) for path in inputs}})
    print("已冻结纯本地原件格式修正；保留首版与真实缺件，0新HTTP和收益计算。", flush=True)


def run():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    if digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("格式修正登记后源码变化。")
    for name, expected in protocol["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)
    save("run_started.json", {"at": now()})
    original = pd.read_parquet(MANIFEST)
    results, texts = [], {}
    for i, original_row in enumerate(original.to_dict("records")):
        row = dict(original_row)
        quarter = row["quarter"]
        row.update(original_v1_status=row["status"], status="NO_VIEW_SOURCE", economic_similarity=None, guidance_similarity=None)
        if row["pdf_received"]:
            full_text = (SOURCE / "texts" / (quarter + "_full.txt")).read_text(encoding="utf-8")
            row["cover_identity_valid"] = cover_identity(full_text.split("\f")[0], quarter)
            sections = split_sections(full_text)
            row.update(economic_characters=len(sections["economic"] or ""), guidance_characters=len(sections["guidance"] or ""),
                       economic_heading=sections["economic_heading"], policy_heading=sections["policy_heading"])
            row["sections_valid"] = row["economic_characters"] >= 100 and row["guidance_characters"] >= 100
            row["status"] = "ORIGINAL_DOCUMENT_AND_TWO_SECTIONS" if row["cover_identity_valid"] and row["sections_valid"] else "NO_VIEW_IDENTITY_OR_SECTION"
            for part in ("economic", "guidance"):
                if sections[part]:
                    (OUT / "texts").mkdir(exist_ok=True)
                    (OUT / "texts" / (quarter + "_" + part + ".txt")).write_text(sections[part], encoding="utf-8")
            if row["status"] == "ORIGINAL_DOCUMENT_AND_TWO_SECTIONS":
                texts[quarter] = sections
        previous = EXPECTED[i - 1] if i else None
        row["previous_quarter"] = previous
        for part in ("economic", "guidance"):
            if quarter in texts and previous in texts:
                row[part + "_similarity"] = similarity(texts[quarter][part], texts[previous][part])
        results.append(row)
    dates = pd.read_parquet(CALENDAR, columns=["date"]).date
    reports, origins = bind_origins(pd.DataFrame(results), dates)
    cases = pd.read_parquet(CASES)
    if len(cases) != 30 or len(origins) != 3488:
        raise ValueError("原完整范围变化。")
    bound = cases.merge(origins, on="date", how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(cases, bound[cases.columns], check_exact=True)
    for name, frame in (("全部59报告_修正格式并保留缺件", reports), ("全部3488原点_最新公布两章节变化与未知", origins),
                        ("原全部30案例_当时报告与原解释等级", bound)):
        frame.to_parquet(OUT / (name + ".parquet"), index=False)
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    summary = {"at": now(), "decision": "TECH.R266", "status": "COMPLETED_LOCAL_SOURCE_FORMAT_FIX_AND_ALL_CASE_ALIGNMENT",
               "expected_reports": len(EXPECTED), "original_documents": int(reports.pdf_received.sum()),
               "two_sections_qualified": int(reports.status.eq("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS").sum()),
               "paired_two_channel_changes": int((reports.economic_similarity.notna() & reports.guidance_similarity.notna()).sum()),
               "all_quarters_originals_complete": bool(reports.status.eq("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS").all()),
               "v1_format_false_unknown_restored": int((reports.original_v1_status.ne("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS") & reports.status.eq("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS")).sum()),
               "all_original_cases": len(cases), "cases_two_channel_known": int(bound.pbc_two_channel_known.sum()),
               "unique_case_reports_carried": int(bound.pbc_quarter.nunique()),
               "all_daily_origins": len(origins), "origins_two_channel_known": int(origins.pbc_two_channel_known.sum()),
               "unique_origin_reports_carried": int(origins.pbc_quarter.nunique()),
               "missing_origins": int((~origins.pbc_two_channel_known).sum()),
               "first_vintage_authenticated": False, "original_cases_and_grades_preserved": True,
               "original_v1_inputs_and_failure_preserved": True, "new_network_requests": 0,
               "financial_admission": "NOT_ADMITTED_PREDICTIVE_METHOD_NOT_REGISTERED", "new_accounts": 0, "new_fits": 0,
               "new_labels": 0, "new_trading_scores": 0, "independent_validation": "NOT_ESTABLISHED",
               "goal_achieved": False, "orders_authorized": False}
    save("summary.json", summary)
    save("run_completed.json", {"at": now(), "terminal": True, "status": summary["status"]})
    print(f"原件{summary['original_documents']}/59，双章节{summary['two_sections_qualified']}，前季双通道{summary['paired_two_channel_changes']}；原30案例可见{summary['cases_two_channel_known']}，0新金融结果。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="只修正央行已保存原件的格式与首次日历对应。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()


if __name__ == "__main__":
    main()
