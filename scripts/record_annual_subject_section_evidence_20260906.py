"""根据年报明确公司信息栏目补充身份核对，保留原封面未识别记录。"""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_financial_ttm_dependencies_v1"
CASES = [
    ("1219376072", 331, ["法定名称", "中国平安保险（集团）股份有限公司", "601318", "二零二三年年报"]),
    ("1204547754", 5, ["公司信息", "中文名稱： 交通銀行股份有限公司", "601328", "二零一七年年度報告 A 股"]),
    ("1212669927", 101, ["公司基本信息", "公司法定中文名称 中国人寿保险股份有限公司", "二零二一年年报"]),
]


def main():
    rows = []
    for aid, page, strings in CASES:
        path = OUT / "page_texts" / (aid + ".json")
        data = json.loads(path.read_text("utf-8"))
        text = data["pages"][page-1]
        for item in strings:
            if item.replace(" ", "") not in text.replace(" ", "").replace("\r", "").replace("\n", ""):
                raise ValueError(f"公司信息栏原始文字未匹配：{aid} {item}")
        evidence = [{"page": page, "raw_text": text}]
        if aid == "1212669927":
            code_page = data["pages"][101]
            if "股票代码" not in code_page or "A股 上海证券交易所 中国人寿 601628" not in code_page:
                raise ValueError("中国人寿股票代码栏目未匹配")
            evidence.append({"page": 102, "raw_text": code_page})
        rows.append({"announcement_id": aid, "source": data["source"], "cached_pages_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "status": "PASS_EXPLICIT_ANNUAL_COMPANY_INFORMATION_SECTION", "evidence": evidence,
                     "financial_facts_admitted": False, "original_front_only_failure_preserved": True})
    result = {"recorded_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "status": "THREE_ANNUAL_SUBJECTS_CONFIRMED_IN_EXPLICIT_COMPANY_SECTIONS",
              "basis": "人工定位后按明确公司名称、年份、A股证券代码栏目核对；只确认主体，不接纳年报财务指标。",
              "initial_front_passed": 12, "additional_section_passed": 3, "total_subject_identified": 15, "rows": rows}
    with (OUT / "annual_subject_sections_supplement.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print("三份年报已从明确公司信息栏目确认主体，原封面失败记录保留，财务提取继续。")


if __name__ == "__main__":
    main()
