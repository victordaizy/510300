"""把两家成分公司的计划与实际回购披露转换为时点事实账本。"""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import research.corporate_repurchase_public_completion_v1 as source
import research.corporate_repurchase_source_probe_v1 as probe
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_corporate_repurchase_fact_ledger_v1"
STUDY = "510300_CORPORATE_REPURCHASE_FACT_LEDGER_V1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
MARKET = ROOT / "reports/research/510300_repo_segmentation_daily_v1/inputs/market.parquet"
NUM = r"([\d,]+(?:\.\d+)?)"
FULL_DATE = r"(20\d{2})年(\d{1,2})月(\d{1,2})日"
SCHEMES = {
    ("600519.SH", 30, 60): "600519_20240921",
    ("600519.SH", 15, 30): "600519_20251106",
    ("300750.SZ", 20, 30): "300750_20231031",
    ("300750.SZ", 40, 80): "300750_20250407",
    ("300750.SZ", 200, 400): "300750_20260725",
}


def cents(value, unit="元"):
    multiplier = {"元": 100, "万元": 1_000_000, "亿元": 10_000_000_000}[unit]
    number = Decimal(str(value).replace(",", "")) * multiplier
    if number != number.to_integral_value():
        raise ValueError("金额不是整数分，不能默默舍入。")
    return int(number)


def date_value(parts):
    return pd.Timestamp(*map(int, parts))


def classify(title):
    if "限制性股票" in title:
        return "EXCLUDED_RESTRICTED_SHARE_CANCELLATION"
    if "股东持股" in title or "持股情况" in title:
        return "EXCLUDED_HOLDER_LIST"
    if "债权人" in title:
        return "EXCLUDED_CREDITOR_NOTICE"
    if "价格上限" in title:
        return "PRICE_CAP_AMENDMENT_NOT_NEW_CASH"
    if "结果" in title:
        return "COMPLETION"
    if "首次回购" in title:
        return "FIRST_EXECUTION"
    if "进展" in title:
        return "PROGRESS"
    if "方案" in title or "报告书" in title:
        return "PLAN_DISCLOSURE"
    raise ValueError("未分类的回购公告：" + title)


def extract(row, cross_reference):
    text = re.sub(r"\s+", "", "".join(read(source.OUT / row["text_path"])))
    kind = classify(row["title"])
    record = {"symbol": row["symbol"], "document_id": row["document_id"], "title": row["title"],
        "source_url": row["source_url"], "raw_path": str((source.OUT / row["raw_path"]).relative_to(ROOT)),
        "raw_sha256": row["sha256"], "classification": kind, "catalogue_timestamp": row["catalogue_timestamp"],
        "catalogue_date": row["catalogue_date"]}
    assert digest(ROOT / record["raw_path"]) == record["raw_sha256"]
    if kind.startswith("EXCLUDED_") or kind == "PRICE_CAP_AMENDMENT_NOT_NEW_CASH":
        return record
    # 这是明确限定的两家公司五方案解析器；相同预算的新方案不能自动合并。
    limits = set(re.findall(r"不低于(?:人民币)?" + NUM + r"亿元.{0,18}?不超过(?:人民币)?" + NUM + r"亿元", text))
    if len(limits) != 1:
        raise ValueError(f"方案金额上下限不唯一：{row['document_id']} / {limits}")
    floor, cap = next(iter(limits))
    key = (row["symbol"], int(Decimal(floor)), int(Decimal(cap)))
    if key not in SCHEMES:
        raise ValueError("出现未预定的新方案，需建立新的方案身份。")
    scheme = SCHEMES[key]
    if scheme == "300750_20231031":
        assert "2023年10月30日" in text
    elif scheme == "300750_20250407":
        assert "2025年4月7日" in text
    elif scheme == "300750_20260725":
        assert "2026年7月24日" in text or "2026年7月25日" in text
    elif scheme == "600519_20240921":
        assert any(v in text for v in ["2024年9月20日", "2024年11月27日", "2024/9/21"])
    elif scheme == "600519_20251106":
        assert any(v in text for v in ["2025年11月4日", "2025年11月28日", "2025/11/6"]), row["document_id"]
    signed = list(re.finditer(FULL_DATE, text[-220:]))
    if not signed:
        raise ValueError("没有取得文末签章日期：" + row["document_id"])
    signed_date = date_value(signed[-1].groups())
    dates = [pd.Timestamp(row["catalogue_date"]), signed_date]
    sse = cross_reference.get(row["document_id"])
    if sse:
        dates.append(pd.Timestamp(sse["sse_date"]))
        record["sse_source_url"] = sse["sse_url"]
    available_date = max(dates)
    # 不依赖历史目录毫秒字段证明实时送达，保守使用日期末。
    known_at = available_date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    record.update(scheme=scheme, signed_date=signed_date, known_at=known_at,
        planned_floor_cents=cents(floor, "亿元"), planned_cap_cents=cents(cap, "亿元"),
        funding_description="OWN_OR_SELF_RAISED" if "自筹" in text else "OWN_FUNDS",
        loan_possibility_mentioned="股票回购专项贷款" in text,
        actual_loan_drawdown_proven=False, conditional_plan_not_cash_inflow=True,
        cumulative_cents=None, economic_cutoff=None, economic_cutoff_precision=None)
    if kind == "PLAN_DISCLOSURE":
        record["approval_still_required"] = "尚需提交" in text and ("股东会" in text or "股东大会" in text)
        return record
    # 先取摘要中的实际累计值；避免误取期内数、股东增持或回购预算。
    summary = re.findall(r"(?:累计已回购金额|实际回购金额)" + NUM + r"元", text)
    if summary:
        amounts = set(cents(value) for value in summary)
        assert len(amounts) == 1
        amount = next(iter(amounts))
        method = "EXPLICIT_CUMULATIVE_SUMMARY"
    elif "尚未实施股份回购" in text:
        amount, method = 0, "EXPLICIT_NOT_STARTED"
    else:
        # 宁德时代结果公告先回顾首次买入，最后一个成交总金额对应完整累计段。
        matches = list(re.finditer(r"成交总金额为(?:人民币)?" + NUM + r"元", text))
        if not matches:
            raise ValueError("缺少实际累计金额：" + row["document_id"])
        amount = cents(matches[-1].group(1))
        context = text[max(0, matches[-1].start() - 230):matches[-1].end()]
        if kind != "FIRST_EXECUTION":
            assert "累计回购" in context
        method = "FINAL_CUMULATIVE_EXECUTION_PARAGRAPH"
    assert 0 <= amount <= record["planned_cap_cents"]
    cutoff = None
    precision = None
    matches = list(re.finditer(r"截至" + FULL_DATE, text))
    if matches:
        cutoff, precision = date_value(matches[-1].groups()), "EXPLICIT_AS_OF_DAY"
    if cutoff is None:
        matches = list(re.finditer(r"截至(20\d{2})年(\d{1,2})月底", text))
        if matches:
            year, month = map(int, matches[-1].groups())
            cutoff, precision = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0), "EXPLICIT_MONTH_END"
    if cutoff is None and kind == "PROGRESS":
        matches = list(re.finditer(r"(20\d{2})年(\d{1,2})月份?，公司通过集中竞价", text))
        if matches:
            year, month = map(int, matches[-1].groups())
            cutoff, precision = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0), "MONTHLY_EXECUTION_END"
    if cutoff is None and kind == "COMPLETION":
        matches = list(re.finditer(FULL_DATE + r"，公司回购股份实施完[成毕]", text))
        if matches:
            cutoff, precision = date_value(matches[-1].groups()), "EXPLICIT_COMPLETION_DAY"
        else:
            assert "截至本公告披露日" in text
            cutoff, precision = signed_date, "AS_OF_ANNOUNCEMENT_NOT_LAST_BUY_DAY"
    if cutoff is None and kind == "FIRST_EXECUTION":
        matches = list(re.finditer(FULL_DATE + r"，公司(?:通过|首次实施)", text))
        if matches:
            cutoff, precision = date_value(matches[-1].groups()), "EXPLICIT_FIRST_BUY_DAY"
    if cutoff is None:
        raise ValueError("没有解析执行截止日期：" + row["document_id"])
    assert cutoff <= available_date
    record.update(cumulative_cents=amount, amount_method=method, economic_cutoff=cutoff,
        economic_cutoff_precision=precision, approval_still_required=False)
    return record


def attach_increments(records):
    """只对同方案相邻的实际累计披露求差，期前基数保持未知。"""
    previous, output = {}, []
    for item in sorted(records, key=lambda row: (row["known_at"], row["document_id"])):
        row = dict(item)
        key, amount = row["scheme"], row["cumulative_cents"]
        row["newly_disclosed_increment_cents"] = None
        row["prior_actual_document_id"] = None
        row["increment_status"] = "NOT_AN_EXECUTION_DISCLOSURE"
        if amount is not None:
            if key in previous:
                old = previous[key]
                row["newly_disclosed_increment_cents"] = amount - old["cumulative_cents"]
                row["prior_actual_document_id"] = old["document_id"]
                assert row["newly_disclosed_increment_cents"] >= 0
                row["increment_status"] = "POSITIVE_DISCLOSED_CHANGE" if amount > old["cumulative_cents"] else "UNCHANGED_CUMULATIVE_NOT_NEW_FLOW"
            elif row["classification"] == "FIRST_EXECUTION" or amount == 0:
                row["newly_disclosed_increment_cents"] = amount
                row["increment_status"] = "EXPLICIT_FIRST_EXECUTION" if amount else "EXPLICIT_ZERO_BASELINE"
            else:
                row["increment_status"] = "PRIOR_BASELINE_UNKNOWN"
            previous[key] = row
        output.append(row)
    return output


def asof_states(events, dates):
    """每日只保留已经公开的事实状态；未报告的日内买入额始终未知。"""
    output, state, position = [], {}, 0
    events = sorted(events, key=lambda row: (row["known_at"], row["document_id"]))
    for day in dates:
        decision = pd.Timestamp(day).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)
        while position < len(events) and events[position]["known_at"] < decision:
            row = events[position]
            old = state.get(row["scheme"], {})
            cumulative = row["cumulative_cents"] if row["cumulative_cents"] is not None else old.get("cumulative_cents")
            state[row["scheme"]] = {**row, "cumulative_cents": cumulative}
            position += 1
        for symbol in ["600519.SH", "300750.SZ"]:
            selected = [row for row in state.values() if row["symbol"] == symbol]
            actual = [row for row in selected if row["cumulative_cents"] is not None]
            output.append({"date": day, "decision_at": decision, "symbol": symbol,
                "information_status": "KNOWN_DISCLOSURE_HISTORY" if selected else "NO_VIEW_NO_PRIOR_SELECTED_DISCLOSURE",
                "known_schemes": len(selected), "known_completed_schemes": sum(row["classification"] == "COMPLETION" for row in selected),
                "sum_latest_cumulative_cents": sum(row["cumulative_cents"] for row in actual) if actual else None,
                "current_day_actual_buy_cents": None,
                "latest_known_at": max((row["known_at"] for row in selected), default=None),
                "latest_document_ids": "|".join(row["document_id"] for row in selected)})
    return pd.DataFrame(output)


def implementation_checks():
    assert cents("1,234.56") == 123456 and cents("1.2", "亿元") == 12_000_000_000
    assert classify("关于部分限制性股票回购注销完成的公告").startswith("EXCLUDED_")
    base = {"symbol": "300750.SZ", "planned_cap_cents": 10000, "cumulative_cents": 3000,
        "known_at": pd.Timestamp("2025-01-02 23:59:59", tz="Asia/Shanghai"),
        "scheme": "A", "document_id": "1", "classification": "FIRST_EXECUTION"}
    rows = [base, {**base, "document_id": "2", "known_at": base["known_at"] + pd.Timedelta(days=1), "classification": "PROGRESS"},
        {**base, "document_id": "3", "known_at": base["known_at"] + pd.Timedelta(days=2), "classification": "COMPLETION"},
        {**base, "scheme": "B", "document_id": "4", "known_at": base["known_at"] + pd.Timedelta(days=3), "cumulative_cents": 500}]
    changes = attach_increments(rows)
    assert [row["newly_disclosed_increment_cents"] for row in changes] == [3000, 0, 0, 500]
    missing = attach_increments([{**base, "classification": "PROGRESS"}])
    assert missing[0]["newly_disclosed_increment_cents"] is None
    dates = pd.date_range("2025-01-02", periods=7)
    full = asof_states(changes, dates)
    prefix = asof_states(changes[:2], dates[:3])
    pd.testing.assert_frame_equal(full[full.date.isin(dates[:3])].reset_index(drop=True), prefix)
    assert full[full.date.eq("2025-01-02")].information_status.str.startswith("NO_VIEW").all()
    assert full.current_day_actual_buy_cents.isna().all()
    return {"money_units_exact_cents": True, "unchanged_cumulative_is_zero_increment": True,
        "schemes_not_spliced": True, "unknown_starting_baseline_not_zero": True,
        "restriction_cancellation_excluded": True, "future_disclosure_does_not_change_past_state": True,
        "daily_actual_purchases_not_fabricated": True}


def run():
    tests = implementation_checks()
    for folder in ["code", "results"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    documents = read(source.OUT / "results/resolved_documents.json")
    assert len(documents) == 57 and all(row["status"] == "PDF_TEXT_SAVED" for row in documents)
    cross = read(source.OUT / "results/sse_catalogue_cross_reference.json")
    links = {row["candidate_document_ids"][0]: row for row in cross if row["unique_match"]}
    protocol = {"at": now(), "study_id": STUDY,
        "purpose": "先修复股票实际需求的计量，再决定是否具备完整指数与两年滚动预测所需证据。",
        "source_rows": 57, "source_companies": ["600519.SH", "300750.SZ"],
        "period": [probe.START, probe.END], "scheme_mapping": [{"symbol": k[0], "floor_100m": k[1], "cap_100m": k[2], "scheme": v} for k, v in SCHEMES.items()],
        "rules": "计划与执行分开；累计值仅同方案求差；期前累计基数不计成新增；完成方案的剩余上限不再作为潜在买入；限制性股票回购注销排除；持股清单、债权人通知、价格上限调整不增加实际金额。",
        "time": "以目录日期、签章日期和可匹配的交易所日期三者最大值的当日末为已知上界，次日交易时点方可使用；按经济截止日期回填禁止。",
        "daily_panel": "只生成过去已披露累计状态；不平均摊日、不产生当日实际买入额，不将每个前向填充日当新观察。",
        "validation_limit": "两家公司、五方案只验证数据处理可行性，不代表全指数；本轮无新账户或预测拟合。",
        "first_vintage_delivery_authenticated": False, "new_independent_forward_observations": 0,
        "implementation_checks": tests, "code_sha256": digest(Path(__file__)),
        "source_manifest_sha256": digest(source.OUT / "results/resolved_documents.json"),
        "new_accounts": 0, "new_fits": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "protocol.json", protocol, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    parsed = [extract(row, links) for row in documents]
    included = attach_increments([row for row in parsed if row.get("scheme")])
    excluded = [row for row in parsed if not row.get("scheme")]
    save(OUT / "results/facts.json", included, True)
    save(OUT / "results/excluded_documents.json", excluded, True)
    frame = pd.DataFrame(included)
    for field in ["planned_floor_cents", "planned_cap_cents", "cumulative_cents", "newly_disclosed_increment_cents"]:
        frame[field] = pd.array([row.get(field) for row in included], dtype="Int64")
    frame.to_parquet(OUT / "results/facts.parquet", index=False)
    dates = pd.read_parquet(MARKET, columns=["date"])
    dates = dates.loc[dates.date.between(probe.START, probe.END), "date"]
    daily = asof_states(included, dates)
    daily.to_parquet(OUT / "results/known_disclosure_daily.parquet", index=False)
    summaries = []
    for scheme, group in frame.groupby("scheme", sort=True):
        group = group.sort_values(["known_at", "document_id"])
        actual = group[group.cumulative_cents.notna()]
        last = actual.iloc[-1]
        first = group.iloc[0]
        closed = last.classification == "COMPLETION"
        positive = actual.newly_disclosed_increment_cents.gt(0).fillna(False)
        unchanged = actual.increment_status.eq("UNCHANGED_CUMULATIVE_NOT_NEW_FLOW")
        known_increments = int(actual.newly_disclosed_increment_cents.sum())
        starting = int(actual.iloc[0].cumulative_cents) if actual.iloc[0].increment_status == "PRIOR_BASELINE_UNKNOWN" else 0
        assert starting + known_increments == int(last.cumulative_cents)
        first_purchase = actual[actual.classification.eq("FIRST_EXECUTION")]
        summaries.append({"scheme": scheme, "symbol": last.symbol, "first_observed_known_at": first.known_at,
            "last_known_at": last.known_at, "plan_floor_cents": int(last.planned_floor_cents), "plan_cap_cents": int(last.planned_cap_cents),
            "actual_cumulative_cents": int(last.cumulative_cents), "closed": closed,
            "unused_cap_cents": int(last.planned_cap_cents - last.cumulative_cents),
            "closed_scheme_future_capacity_cents": 0 if closed else None,
            "execution_disclosures": len(actual), "positive_disclosed_changes": int(positive.sum()),
            "unchanged_cumulative_disclosures": int(unchanged.sum()), "unknown_starting_baseline_cents": starting,
            "known_disclosed_increment_sum_cents": known_increments,
            "naive_sum_of_cumulative_cents": int(actual.cumulative_cents.sum()),
            "first_execution_date": first_purchase.economic_cutoff.iloc[0] if len(first_purchase) else None,
            "days_first_observed_plan_to_first_execution": int((first_purchase.economic_cutoff.iloc[0] - pd.Timestamp(first.known_at).tz_localize(None).normalize()).days) if len(first_purchase) and first.classification == "PLAN_DISCLOSURE" else None})
    save(OUT / "results/scheme_summaries.json", summaries, True)
    actual = frame[frame.cumulative_cents.notna()]
    naive = int(actual.cumulative_cents.sum())
    end_total = sum(row["actual_cumulative_cents"] for row in summaries)
    result = {"at": now(), "study_id": STUDY, "status": "COMPLETED_ISSUER_DEMAND_MEASUREMENT_NOT_INDEX_STRATEGY",
        "source_documents": len(documents), "companies": 2, "schemes": len(summaries),
        "fact_events": len(included), "excluded_documents": len(excluded),
        "classification_counts": pd.Series([row["classification"] for row in parsed]).value_counts().to_dict(),
        "execution_disclosures": len(actual), "increment_status_counts": actual.increment_status.value_counts().to_dict(),
        "naive_sum_of_cumulative_cny": naive / 100, "sum_latest_scheme_cumulative_cny": end_total / 100,
        "naive_overcount_multiple": naive / end_total,
        "sum_known_disclosed_increment_cny": sum(row["known_disclosed_increment_sum_cents"] for row in summaries) / 100,
        "unknown_starting_baseline_cny": sum(row["unknown_starting_baseline_cents"] for row in summaries) / 100,
        "scheme_summaries": summaries, "daily_state_rows": len(daily), "daily_actual_buy_amount_known_rows": 0,
        "sse_catalogue_unique_matches": len(links), "source_origins_preserved": True,
        "implementation_checks": tests, "new_accounts": 0, "new_fits": 0,
        "whole_CSI300_coverage_established": False, "first_vintage_delivery_authenticated": False,
        "independent_forward_observations": 0, "current_market_view": "NO_VIEW",
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print(f"实际需求事实账本完成：{len(included)}个信息事件、{len(actual)}次执行披露、{len(summaries)}个方案；没有把它们当作完整指数策略。", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print(implementation_checks())
        rows = read(source.OUT / "results/resolved_documents.json")
        cross = read(source.OUT / "results/sse_catalogue_cross_reference.json")
        links = {r["candidate_document_ids"][0]: r for r in cross if r["unique_match"]}
        output = [extract(row, links) for row in rows]
        print({"已解析": len(output), "事实事件": sum(bool(row.get("scheme")) for row in output)})
    else:
        run()
