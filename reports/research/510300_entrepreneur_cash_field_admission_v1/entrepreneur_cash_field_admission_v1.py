"""提取企业家原季度收款与周转判断，按各自公布时钟保留可用性。"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.entrepreneur_cash_source_v1 as source
import research.banker_survey_availability_completion_v1 as banking
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_entrepreneur_cash_field_admission_v1"
STUDY = "510300_ENTREPRENEUR_CASH_FIELD_ADMISSION_V1"
PANEL = OUT / "released_entrepreneur_cash.parquet"
FIELDS = {"sales_revenue_collection_index": "销货款回笼指数", "fund_turnover_index": "资金周转指数"}


def identity(original):
    record = dict(original)
    if record["title_text_matches"]:
        return record
    assert record["quarter"] in ["2018Q2", "2020Q4"], "另一个标题未辨认，必须先核对原页。"
    q = record["quarter"]
    page_date, printed_date, title = {
        "2018Q2": ("2018-06-15", "2018年6月15日", "2018年第二季度企业家问卷调查报告"),
        "2020Q4": ("2021-01-19", "2021年1月19日", "2020年第四季度企业家问卷调查报告")}[q]
    assert record["published_at"].startswith(page_date)
    images = [source.OUT / "texts" / f"{q}_title_check.png"]
    if q == "2020Q4":
        images.append(source.OUT / "texts/2020Q4_table_check.png")
    assert all(path.exists() for path in images)
    record["visual_identity_completion"] = {
        "title": title, "printed_date": printed_date,
        "images": {path.relative_to(ROOT).as_posix(): digest(path) for path in images},
        "reason": "原页已渲染并阅读，PDF对象的文本顺序使标题不连续；原始字节和最初问题标记保留。"}
    return record


def parse(original):
    original = identity(original)
    assert digest(ROOT / original["pdf_raw_path"]) == original["pdf_sha256"]
    document = read(ROOT / original["text_path"])
    assert document["pdf_sha256"] == original["pdf_sha256"]
    with pdfplumber.open(ROOT / original["pdf_raw_path"]) as pdf:
        pages = [p.dedupe_chars(tolerance=1, extra_attrs=()) for p in pdf.pages]
        texts = [p.extract_text() or "" for p in pages]
        selected = [p for p, text in zip(pages, texts) if "企业家调查问卷指数表" in re.sub(r"\s+", "", text)]
        assert len(selected) == 1, "企业家附表页不唯一。"
        tables = selected[0].extract_tables()
    full = "\n".join(texts)
    compact = re.sub(r"\s+", "", full)
    quarter = original["quarter"].replace("Q", ".Q")
    rows = re.findall(r"(?m)^\s*" + re.escape(quarter) + r"\s+([^\n]+)$", full)
    assert len(rows) == 1, "原季度附表当季行不唯一。"
    values = re.findall(r"\d+(?:\.\d+)?", rows[0])
    assert len(values) in [9, 10], f"附表字段数量未识别：{len(values)}。"
    matches = []
    for table in tables:
        header = [re.sub(r"\s+", "", cell or "") for cell in table[0]]
        if all(any(label in cell for cell in header) for label in FIELDS.values()):
            matches.append((table, header))
    assert len(matches) == 1, "目标表头没有唯一匹配。"
    table, header = matches[0]
    assert "热度指数" in header[0] and len(header) == len(values) - 1
    mapped, evidence = {}, {}
    for field, label in FIELDS.items():
        positions = [i for i, cell in enumerate(header) if label in cell]
        assert len(positions) == 1, (label, "表头未唯一识别")
        position = positions[0]
        value = float(values[position])
        assert value == float(str(table[-1][position]).splitlines()[-1]), "附表行与同名单元格不一致。"
        body = re.findall(re.escape(label) + r"为(\d+(?:\.\d+)?)%", compact)
        assert len(body) == 1 and float(body[0]) == value, "正文当季值与附表不一致。"
        definitions = list(re.finditer(r"(\d+)[、.．]" + re.escape(label) + r"[：:]", compact))
        assert len(definitions) == 1, "指标定义未唯一识别。"
        definition = definitions[0]
        excerpt = compact[definition.start():definition.start() + 280]
        assert all(word in excerpt for word in ["扩散指数", "良好", "一般", "0.5"]), "定义未符合收款或周转判断。"
        assert 0 <= value <= 100
        mapped[field] = value
        evidence[field] = {"header": header[position], "column_number_one_based": position + 1,
                           "definition_number": int(definition.group(1)), "definition_excerpt": excerpt,
                           "body_current_value": float(body[0]), "cell_current_value": value}
    expected = (8, 7) if len(values) == 10 else (7, 6)
    assert tuple(evidence[field]["column_number_one_based"] for field in FIELDS) == expected
    return {**original, **mapped, "status": "CURRENT_QUARTER_CASH_PERCEPTION_EXTRACTED",
            "table_field_count": len(values), "current_quarter_row_evidence": quarter + " " + rows[0],
            "field_definition_evidence": evidence, "actual_corporate_cashflow_measurement": False}


def run():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("企业收款字段准入已固定，不重复执行。")
    originals = read(source.OUT / "saved_originals.json")
    assert read(source.OUT / "unresolved_sources.json") == []
    assert len(originals) == 32
    # 解析只读原文，不读取股票收益；全部字段核清后再固定本阶段的输出。
    records = [parse(original) for original in originals]
    frame = pd.DataFrame([{k: v for k, v in row.items() if k not in ["field_definition_evidence", "visual_identity_completion"]}
                          for row in records])
    frame["known_at"] = pd.to_datetime(frame.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    frame["quarter_end"] = pd.PeriodIndex(frame.quarter, freq="Q").end_time.normalize()
    frame["publication_days_from_quarter_end"] = (frame.known_at.dt.tz_localize(None).dt.normalize() - frame.quarter_end).dt.days
    frame = frame.sort_values(["known_at", "quarter"]).reset_index(drop=True)
    states = frame.drop_duplicates("known_at", keep="last")
    dates = pd.read_parquet(ROOT / "reports/research/510300_repo_segmentation_daily_v1/inputs/market.parquet", columns=["date"])
    dates["date"] = pd.to_datetime(dates.date).dt.as_unit("ns")
    dates["decision_time"] = dates.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)
    daily = pd.merge_asof(dates, states[["known_at", "quarter"]], left_on="decision_time", right_on="known_at", direction="backward")
    daily["age_days"] = (daily.decision_time - daily.known_at).dt.total_seconds() / 86400
    daily["available_within_120_days"] = daily.age_days.between(0, 120)
    bank = pd.read_parquet(banking.PANEL, columns=["quarter", "known_at"])
    bank["known_at"] = pd.to_datetime(bank.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    clock_comparison = frame[["quarter", "known_at"]].merge(bank, on="quarter", suffixes=("_enterprise", "_bank"), validate="one_to_one")
    current = daily[daily.date.between("2020-01-02", "2026-09-24")]
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "fields": FIELDS,
        "purpose": "原季度原文、同名附表单元格与正文三处一致后准入，只保存当季值。",
        "source_universe": "全国工业企业调查，不等于沪深300企业样本，个体答卷及成员历史不可得。",
        "semantics": "销货款回笼和资金周转分别为良好比例加0.5乘一般比例，都是扩散判断；不是企业财务现金流金额、应收账款周转天数或股票资金流。",
        "single_strategy_feature_intent": "后续只检验销货款回笼一项，资金周转留作定义核对，不自动入模。",
        "clock": "每份报告按自己的页面时间保存，以当日日末保守可用。同日补发仅最新统计季成为之后状态，不回填此前日期。",
        "table_schema": "有企业家信心列时10列、删除后9列，按实际表头定位；编号不能替代表头。",
        "past_values": "后续报告附表中的重列旧值不用于覆盖当时已公布的原季值。",
        "age_limit_for_source_diagnostic_days": 120,
        "pretested_without_strategy_returns": True, "historical_first_vintage_verified": False,
        "new_strategy_returns": 0, "new_accounts": 0, "orders_authorized": False, "goal_achieved": False}, True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "code_sha256": digest(Path(__file__)), "source_manifest_sha256": digest(source.OUT / "saved_originals.json"),
        "source_result_sha256": digest(source.OUT / "result.json"), "bank_panel_sha256": digest(banking.PANEL)}, True)
    save(OUT / "extracted_current_quarters.json", records, True)
    frame.to_parquet(PANEL, index=False)
    daily.to_parquet(OUT / "daily_source_availability.parquet", index=False)
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(OUT / "result.json", {"at": now(), "study_id": STUDY, "status": "32_CURRENT_QUARTER_CASH_REPORTS_READY_STRATEGY_NOT_RUN",
        "quarters": len(frame), "first_quarter": frame.quarter.min(), "last_quarter": frame.quarter.max(),
        "unique_effective_release_days": len(states), "table_field_counts": frame.table_field_count.value_counts().to_dict(),
        "same_day_releases": frame.loc[frame.known_at.duplicated(keep=False), ["quarter", "known_at"]].to_dict("records"),
        "publication_delay_min_days": int(frame.publication_days_from_quarter_end.min()),
        "publication_delay_max_days": int(frame.publication_days_from_quarter_end.max()),
        "delays_over_90_days": frame.loc[frame.publication_days_from_quarter_end.gt(90), ["quarter", "published_at", "publication_days_from_quarter_end"]].to_dict("records"),
        "bank_and_enterprise_clock_differences": clock_comparison.loc[clock_comparison.known_at_bank.ne(clock_comparison.known_at_enterprise)].to_dict("records"),
        "evaluated_market_days": len(current), "available_days_within_120": int(current.available_within_120_days.sum()),
        "latest_information_asof_last_market_day": current.iloc[-1].to_dict(),
        "historical_first_vintage_verified": False, "new_accounts": 0, "new_strategy_returns": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False}, True)
    print(f"企业收款与周转原季字段提取{len(frame)}季、{len(states)}个有效公布日；尚未运行策略。", flush=True)


if __name__ == "__main__":
    run()
