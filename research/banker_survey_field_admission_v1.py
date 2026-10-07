"""只提取银行家原报告的当季值，并计量季度数据在日更训练中的独立时点。"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.banker_survey_source_completion_v1 as source
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_banker_survey_field_admission_v1"
STUDY = "510300_BANKER_SURVEY_FIELD_ADMISSION_V1"
FIELDS = {"loan_demand_index": "贷款总体需求指数", "loan_approval_index": "银行贷款审批指数",
          "monetary_policy_perception_index": "货币政策感受指数"}


def input_originals():
    records = read(source.OUT / "saved_originals.json")
    missing = read(source.OUT / "unresolved_sources.json")
    assert missing == [{"quarter": "2020Q4", "error": ""}], "只补全已经看过原页的2020Q4标题。"
    initial = source.prior
    item = next(r for r in read(initial.OUT / "catalogue.json") if r["quarter"] == "2020Q4")
    html = read(initial.OUT / "requests/2020Q4_page.json")
    pdf = read(initial.OUT / "requests/2020Q4_pdf.json")
    title_image = initial.OUT / "texts/2020Q4_title_check.png"
    table_image = initial.OUT / "texts/2020Q4_table_check.png"
    assert title_image.exists() and table_image.exists()
    clocks = re.findall(r"文章来源[：:]?\s*(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", initial.soup_for(html).get_text(" ", strip=True))
    assert len(clocks) == 1 and clocks[0].startswith("2021-01-19")
    record = {**item, "published_at": clocks[0] + "+08:00", "conservative_known_at": clocks[0][:10] + "T23:59:59+08:00",
              "page_raw_path": html["raw_path"], "page_sha256": html["sha256"], "pdf_url": pdf["url"],
              "pdf_raw_path": pdf["raw_path"], "pdf_sha256": pdf["sha256"], "page_count": 5,
              "text_path": (initial.OUT / "texts/2020Q4.json").relative_to(ROOT).as_posix(),
              "historical_first_vintage_verified": False,
              "visual_identity_completion": {"title_read": "2020年第四季度银行家问卷调查报告",
                  "title_image": title_image.relative_to(ROOT).as_posix(), "title_image_sha256": digest(title_image),
                  "table_image": table_image.relative_to(ROOT).as_posix(), "table_image_sha256": digest(table_image),
                  "reason": "原PDF文本对象的变换使标题在提取文本中不连续，原页渲染确认标题与2021年1月19日页眉；当季表格单元格正常。"}}
    records.append(record)
    records.sort(key=lambda r: r["quarter"])
    assert len(records) == len({r["quarter"] for r in records}) == 32
    return records


def parse(original):
    document = read(ROOT / original["text_path"])
    assert document["pdf_sha256"] == original["pdf_sha256"]
    assert digest(ROOT / original["pdf_raw_path"]) == original["pdf_sha256"]
    with pdfplumber.open(ROOT / original["pdf_raw_path"]) as pdf:
        pages = [p.dedupe_chars(tolerance=1, extra_attrs=()) for p in pdf.pages]
        texts = [p.extract_text() or "" for p in pages]
        table_pages = [p for p, text in zip(pages, texts) if "附件：银行家问卷调查指数表" in re.sub(r"\s+", "", text)]
        assert len(table_pages) == 1
        tables = table_pages[0].extract_tables()
    full = "\n".join(texts)
    compact = re.sub(r"\s+", "", full)
    quarter = original["quarter"].replace("Q", ".Q")
    candidates = re.findall(r"(?m)^\s*" + re.escape(quarter) + r"\s+([^\n]+)$", full)
    assert len(candidates) == 1, "原报告当季附表行不唯一。"
    amounts = re.findall(r"\d+(?:\.\d+)?", candidates[0])
    assert len(amounts) in [11, 12], f"附表字段数未识别：{len(amounts)}。"
    assert "附件：银行家问卷调查指数表" in compact and "编制说明" in compact
    # 按实际表头定位。某些年份的编制说明没有列出全部行业，编号不能充当列号。
    qualified = []
    for table in tables:
        header = [re.sub(r"\s+", "", cell or "") for cell in table[0]]
        if all(any(label in cell for cell in header) for label in FIELDS.values()):
            qualified.append((table, header))
    assert len(qualified) == 1, "含三个目标表头的原始表不唯一。"
    table, header = qualified[0]
    assert "热度指数" in header[0] and len(header) == len(amounts) - 1, "表格边界不符合已辨认的首末列结构。"
    mapped, evidence = {}, {}
    for field, label in FIELDS.items():
        matches = list(re.finditer(r"(\d+)[.．]" + label + r"[：:]", compact))
        assert len(matches) == 1, (label, "编制编号不唯一")
        match = matches[0]
        positions = [i for i, cell in enumerate(header) if label in cell]
        assert len(positions) == 1
        position = positions[0]
        assert 0 <= position < len(amounts)
        mapped[field] = float(amounts[position])
        cell_value = float(str(table[-1][position]).splitlines()[-1])
        assert cell_value == mapped[field], "当季文本行与同名原表单元格不一致。"
        evidence[field] = {"column_number_one_based": position + 1,
                           "header": header[position], "method_definition_number": int(match.group(1)),
                           "definition_excerpt": compact[match.start():match.start() + 180]}
    assert all(0 <= value <= 100 for value in mapped.values())
    expected = (3, 4, 10) if len(amounts) == 12 else (2, 3, 9)
    actual = tuple(evidence[f]["column_number_one_based"] for f in ["monetary_policy_perception_index", "loan_demand_index", "loan_approval_index"])
    assert expected == actual, "当前列序与已辨认表头不一致。"
    assert "放松" in evidence["loan_approval_index"]["definition_excerpt"]
    assert "基本不变" in evidence["loan_approval_index"]["definition_excerpt"]
    return {**original, **mapped, "table_field_count": len(amounts), "current_quarter_row_evidence": quarter + " " + candidates[0],
            "field_definition_evidence": evidence, "status": "CURRENT_QUARTER_MEASUREMENTS_EXTRACTED",
            "loan_approval_is_diffusion_not_approval_rate": True}


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("银行家字段准入已有终态。")
    out = read(source.OUT / "result.json")
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "source_status": out["status"],
        "extraction": "仅提取每份原季度报告的当季附表行。按实际表头定位并与同名单元格核对，适配11/12列。2019年至2020年部分编制说明缺少行业条目，定义编号不可当列号。重叠字形在1点坐标内去重，不按相邻字符删数字。",
        "fields": FIELDS,
        "semantics": "贷款需求为银行家对需求增长/不变的扩散判断；审批为条件放松/不变的扩散判断；政策感受亦为扩散指数。三者均不等于实际贷款数或批准比例，不能相减当作经济缺口。",
        "clock": "沿用每份原报告页面公布日末保守可用。季度末可能在公布日前也可能在其后，季度调查不冒充季度结束后才有的全季实现统计。",
        "same_day": "同日公布多个季度时，当日结束后的最新状态取统计季度最新者；旧季当时才公布的值不回填过去决策。",
        "availability_measurement": "按已有市场日期09:00形成逐日元数据，只计资料可用性及最近两年可观察季度状态数，不读取股票收益或拟合策略。",
        "age_diagnostic_days": 120, "age_is_measurement_limit_not_validated_trade_rule": True,
        "new_strategy_returns": 0, "new_accounts": 0, "source_parser_pretested_without_stock_returns": True,
        "historical_first_vintage_verified": False, "goal_achieved": False}, True)
    originals = input_originals()
    save(OUT / "source_identity_completion.json", originals, True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "code_sha256": digest(Path(__file__)),
        "source_manifest_sha256": digest(source.OUT / "saved_originals.json"), "source_result_sha256": digest(source.OUT / "result.json"),
        "completed_original_manifest_sha256": digest(OUT / "source_identity_completion.json")}, True)
    records, failures = [], []
    for original in originals:
        try:
            records.append(parse(original))
        except (AssertionError, ValueError, KeyError) as exc:
            failures.append({"quarter": original["quarter"], "error": str(exc), "source_url": original["source_url"]})
    save(OUT / "extracted_current_quarters.json", records, True)
    save(OUT / "unresolved_fields.json", failures, True)
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "field_definition_evidence"} for r in records])
    assert len(frame) > 0
    frame["known_at"] = pd.to_datetime(frame.conservative_known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    frame["survey_quarter_end"] = pd.PeriodIndex(frame.quarter, freq="Q").end_time.normalize()
    frame["publication_days_from_quarter_end"] = (frame.known_at.dt.tz_localize(None).dt.normalize() - frame.survey_quarter_end).dt.days
    frame = frame.sort_values(["known_at", "quarter"]).reset_index(drop=True)
    repeated = frame[frame.known_at.duplicated(keep=False)][["quarter", "published_at", "known_at"]]
    frame.to_parquet(OUT / "released_banker_survey.parquet", index=False)
    states = frame.drop_duplicates("known_at", keep="last")
    dates = pd.read_parquet(ROOT / "reports/research/510300_repo_segmentation_daily_v1/inputs/market.parquet", columns=["date"])
    dates["decision_time"] = dates.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)
    daily = pd.merge_asof(dates, states[["known_at", "quarter"]], left_on="decision_time", right_on="known_at", direction="backward")
    daily["age_days"] = (daily.decision_time - daily.known_at).dt.total_seconds() / 86400
    daily["available_within_120_days"] = daily.age_days.between(0, 120)
    samples = []
    for i, row in daily.iterrows():
        prior = daily.iloc[:i]
        mask = prior.date.ge(row.date - pd.DateOffset(years=2)) & prior.available_within_120_days
        samples.append(int(prior.loc[mask, "quarter"].nunique()))
    daily["distinct_prior_two_year_quarter_states"] = samples
    daily.to_parquet(OUT / "daily_availability_only.parquet", index=False)
    current = daily[daily.date.ge("2020-01-02")]
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "CURRENT_QUARTER_SOURCE_READY_STRATEGY_NOT_RUN" if not failures else "PARTIAL_QUARTER_FIELDS_UNRESOLVED_STRATEGY_NOT_RUN",
        "source_quarters": len(frame), "unresolved_quarters": len(failures), "fields": list(FIELDS),
        "first_quarter": frame.quarter.min(), "last_quarter": frame.quarter.max(),
        "same_day_released_quarters": repeated.to_dict("records"), "unique_effective_release_days": len(states),
        "table_field_counts": frame.table_field_count.value_counts().to_dict(),
        "publication_delay_min_days": int(frame.publication_days_from_quarter_end.min()),
        "publication_delay_max_days": int(frame.publication_days_from_quarter_end.max()),
        "delays_over_90_days": frame.loc[frame.publication_days_from_quarter_end.gt(90), ["quarter", "published_at", "publication_days_from_quarter_end"]].to_dict("records"),
        "evaluated_market_days": len(current), "available_days_within_120": int(current.available_within_120_days.sum()),
        "prior_two_year_distinct_quarter_states_min": int(current.distinct_prior_two_year_quarter_states.min()),
        "prior_two_year_distinct_quarter_states_max": int(current.distinct_prior_two_year_quarter_states.max()),
        "latest_information_asof_last_market_day": current.iloc[-1].to_dict(),
        "historical_first_vintage_verified": False, "new_accounts": 0, "new_strategy_returns": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False}, True)
    (OUT / "source_admission_code.py").write_bytes(Path(__file__).read_bytes())
    print(f"银行家当季字段已提取{len(frame)}季、{len(states)}个有效公布日；未运行策略。", flush=True)


if __name__ == "__main__":
    run()
