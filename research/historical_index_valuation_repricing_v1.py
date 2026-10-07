"""连接固定历史信用发布、指数整体估值及已保存收益，保留盈利口径缺口。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from historical_index_liquidity_transmission_v1 import clean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_valuation_repricing_v1"
RATES = ROOT / "reports/research/510300_historical_index_rates_policy_transmission_v1"
EXPECTATIONS = ROOT / "reports/research/510300_historical_index_credit_expectation_gap_v1"
BRIDGE = ROOT / "reports/research/510300_macro_earnings_pricing_bridge_v5"
STUDY = "510300_HISTORICAL_INDEX_VALUATION_REPRICING_V1"
INPUTS = {
    "pe": str((BRIDGE / "inputs/pe.parquet").relative_to(ROOT)),
    "index_price": str((BRIDGE / "inputs/index_price.parquet").relative_to(ROOT)),
    "vendor_valuation": "data/raw/valuation/000300_valuation_daily_raw.parquet",
    "old_pe": "data/raw/valuation/000300_pe_official_raw.parquet",
    "old_arithmetic": "reports/research/510300_index_repricing_chronology_v26/results/104个月_定价前后便览.csv",
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("已有范围，不覆盖。")
    events = json.loads((RATES / "protocol.json").read_text(encoding="utf-8"))["events"]
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "mode": "HISTORICAL_DISCOVERY_ONLY", "research_unit": "沪深300整体，ETF成本后事件收益单列",
        "question": "2019年1至4月，信用信息出现时指数整体估值已变化多少；旧PE隐含分母与真实已披露盈利能否区分？",
        "events": events, "monthends": ["2018-12", "2019-01", "2019-02", "2019-03", "2019-04"],
        "input_files": INPUTS,
        "existing_work_reused": "V5与V26已做价格/PE恒等式；本轮不声称发明新分解，只连接四次历史调查预期、利率和成本后收益，并检查定义缺口。",
        "known_history": "这些历史价格、部分估值和结果已在旧研究出现，不是盲检或独立验证。",
        "identity": "100*log(P1/P0)=100*log(PE1/PE0)+100*log((P1/PE1)/(P0/PE0))；仅代数恒等式。",
        "asset_clock": "价格和PE只使用同日期000300；不能把510300价格、全收益指数或不同日期估值混入恒等式。",
        "pre_release_context": "各金融数据公开日21:00，选择旧PE账本available_at不晚于该时刻的最近可用值；可用日期与数据日期分开。",
        "lookbacks": [20, 60], "lookback_origin": "直接沿用V5、V26既有20/60交易日价格描述，不选表现最好的窗口。",
        "monthend_role": "月末同日期价格和PE只作历史分解，保留PE假设可用时间；不假称月末收盘即可按收盘下单。",
        "csi_source_probe": {"url": "https://www.csindex.com.cn/csindex-home/perf/index-perf",
                             "params": {"indexCode": "000300", "startDate": "20181228", "endDate": "20190516"},
                             "purpose": "一次有界请求核对官方历史字段；无响应时保留缺口，不批量重试或用现价替代。"},
        "cross_provider": "乐咕静态与TTM整体PE分别作为口径对照，不与中证peg平均、拼接或挑选有利值。当前供应商历史未认证首次版本。",
        "earnings_interpretation": "P/PE是隐含分母；股本、样本、除数、亏损处理、报告期和修订均可能改变它。未取得一致口径真实盈利前，不命名为已披露或预期EPS。",
        "old_failure": "VAL04维持HISTORICAL_REJECTED_FROZEN_NO_STRATEGY_BACKTEST；本轮不复活PB-ROE回归、调阈值或组合权重。",
        "returns": "复用四个既有5/20日净收益，不更改入场、退出或费用；不把解释窗口收益折算为策略夏普。",
        "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None,
        "goal_achieved": False, "orders_authorized": False,
    })
    print("已固定五个月末和四次发布，复用既有恒等式与收益，只补定价及盈利口径证据。", flush=True)


def source_probe():
    if (OUT / "source_probe.json").exists():
        raise RuntimeError("已保存来源请求，不重复。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    spec = protocol["csi_source_probe"]
    row = {"recorded_at": now(), **spec}
    path = OUT / "sources/csi_2019_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(spec["url"], params=spec["params"], timeout=(8, 25),
                                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"})
        path.with_suffix(".raw").write_bytes(response.content)
        row.update(http_status=response.status_code, bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest())
        response.raise_for_status()
        payload = response.json()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data:
            raise ValueError("官方接口未返回非空历史行。")
        row.update(status="RETRIEVED", rows=len(data), fields=list(data[0]), first=data[0], last=data[-1],
                   note="字段名称或第三方翻译不足以认证2019年股本、滚动或静态口径；这里只登记原响应。")
    except Exception as exc:
        row.update(status="MISSING_SOURCE", error=str(exc))
    save("source_probe.json", row)
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)


def build():
    if (OUT / "event_pricing.json").exists():
        raise RuntimeError("已有定价结果，不覆盖。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    pe = pd.read_parquet(ROOT / INPUTS["pe"]).copy()
    price = pd.read_parquet(ROOT / INPUTS["index_price"]).copy()
    vendor = pd.read_parquet(ROOT / INPUTS["vendor_valuation"]).copy()
    pe["date"] = pd.to_datetime(pe["observation_date"])
    pe["available_at"] = pd.to_datetime(pe["available_at"], utc=True).dt.tz_convert("Asia/Shanghai")
    price["date"] = pd.to_datetime(price["date"])
    vendor["date"] = pd.to_datetime(vendor["date"])
    assert not pe["date"].duplicated().any() and not price["date"].duplicated().any() and not vendor["date"].duplicated().any()
    series = price[["date", "close"]].merge(pe[["date", "available_at", "original_pe_official"]], on="date", how="left")
    series = series.merge(vendor[["date", "pe_static", "pe_ttm", "pb", "index_close_pe_source"]], on="date", how="left")
    series = series.sort_values("date").set_index("date")
    series["implied_denominator"] = series["close"] / series["original_pe_official"]
    series["ey_percent"] = 100 / series["original_pe_official"]
    scoped = series.loc["2018-12-03":"2019-05-16"].copy()
    scoped.to_parquet(OUT / "daily_valuation.parquet")
    save("input_receipts.json", {"recorded_at": now(), "inputs": [{"id": k, "path": v,
         "sha256": hashlib.sha256((ROOT / v).read_bytes()).hexdigest()} for k, v in INPUTS.items()]})

    def snap(date):
        row = series.loc[date]
        return {"date": date, **{k: row[k] for k in series.columns},
                "status": "OBSERVED" if pd.notna(row.original_pe_official) else "MISSING_PE"}

    def split(start, end):
        a, b = series.loc[start], series.loc[end]
        if pd.isna(a.original_pe_official) or pd.isna(b.original_pe_official):
            return {"status": "MISSING_PE", "start": start, "end": end}
        lp, lm = np.log(b.close / a.close) * 100, np.log(b.original_pe_official / a.original_pe_official) * 100
        ld = np.log(b.implied_denominator / a.implied_denominator) * 100
        assert abs(lp - lm - ld) < 1e-9
        return {"status": "EXACT_ALGEBRA_NOT_EARNINGS_ATTRIBUTION", "start": start, "end": end,
                "price_simple_return": b.close / a.close - 1, "pe_simple_change": b.original_pe_official / a.original_pe_official - 1,
                "implied_denominator_simple_change": b.implied_denominator / a.implied_denominator - 1,
                "price_log_points": lp, "pe_log_points": lm, "implied_denominator_log_points": ld,
                "identity_error": lp - lm - ld}

    months, month_dates = [], []
    for month in protocol["monthends"]:
        dates = series.index[series.index.to_period("M") == pd.Period(month)]
        date = dates[-1]
        months.append(snap(date))
        month_dates.append(date)
    month_changes = [split(a, b) for a, b in zip(month_dates[:-1], month_dates[1:])]
    total = split(month_dates[0], month_dates[-1])
    old = json.loads((EXPECTATIONS / "comparison.json").read_text(encoding="utf-8"))["rows"]
    rates = pd.read_parquet(RATES / "daily_rates.parquet").set_index("date")
    events = []
    for event in protocol["events"]:
        cutoff = pd.Timestamp(event["date"], tz="Asia/Shanghai") + pd.Timedelta(hours=21)
        eligible = series.loc[series["available_at"].le(cutoff) & series["original_pe_official"].notna()]
        if eligible.empty:
            events.append({"event": event, "status": "NO_VIEW"})
            continue
        end = eligible.index[-1]
        row = {"stat_month": event["id"], "release_date": event["date"], "cutoff": cutoff,
               "latest_pricing": snap(end), "status": "RETROSPECTIVE_RECONSTRUCTION", "lookbacks": []}
        pos = series.index.get_loc(end)
        for horizon in protocol["lookbacks"]:
            row["lookbacks"].append({"horizon": horizon, **split(series.index[pos - horizon], end)})
        observations = [r for r in old if r["stat_month"] == event["id"] and r["series"] == "tsf_flow"]
        row["tsf_surprises"] = [{k: r.get(k) for k in ["provider", "expected", "actual", "surprise_raw", "evidence_tier"]} for r in observations]
        old_return = next(r for r in old if r["stat_month"] == event["id"])
        row.update(net_return_5d=old_return["net_return_5d"], net_return_20d=old_return["net_return_20d"], entry_date=old_return["entry_date"])
        if end in rates.index:
            row["same_date_cgb_10y"] = float(rates.loc[end, "cgb_10y"])
            row["ey_minus_cgb_pp_descriptive_only"] = float(series.loc[end, "ey_percent"] - rates.loc[end, "cgb_10y"])
        events.append(row)
    save("monthend_pricing.json", {"computed_at": now(), "snapshots": months, "monthly_changes": month_changes, "full_period": total})
    save("event_pricing.json", {"computed_at": now(), "rows": events})
    save("result.json", {"study_id": STUDY, "computed_at": now(), "classification": "PROGRESS_INDEX_VALUATION_CLOCK_AND_REPRICING",
                         "events": len(events), "monthends": len(months), "earnings_series_verified": False,
                         "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None,
                         "goal_achieved": False, "orders_authorized": False})
    print(pd.DataFrame(months)[["date", "close", "original_pe_official", "pe_static", "pe_ttm", "implied_denominator"]].to_string(index=False), flush=True)
    print(json.dumps(clean(total), ensure_ascii=False, indent=2), flush=True)
    print(pd.DataFrame([{**{k: r[k] for k in ["release_date", "net_return_20d"]}, "pe_date": r["latest_pricing"]["date"],
                         "pe": r["latest_pricing"]["original_pe_official"],
                         "past60_pe_log": next(q["pe_log_points"] for q in r["lookbacks"] if q["horizon"] == 60),
                         "past60_price_return": next(q["price_simple_return"] for q in r["lookbacks"] if q["horizon"] == 60)} for r in events]).to_string(index=False), flush=True)


def definition_evidence():
    """只核对已取得的历史响应，不把一致性当作盈利定义证明。"""
    if (OUT / "definition_evidence.json").exists():
        raise RuntimeError("已保存口径对照，不覆盖。")
    daily = pd.read_parquet(OUT / "daily_valuation.parquet")
    probe = json.loads((OUT / "source_probe.json").read_text(encoding="utf-8"))
    official = {"source_status": probe["status"], "first_vintage_verified": False,
                "definition_verified": False, "independent_provider": False}
    if probe["status"] == "RETRIEVED":
        rows = json.loads((OUT / "sources/csi_2019_history.json").read_text(encoding="utf-8"))["data"]
        source = pd.DataFrame(rows)
        source["date"] = pd.to_datetime(source["tradeDate"].astype(str))
        source["peg"] = pd.to_numeric(source["peg"])
        source["close"] = pd.to_numeric(source["close"])
        joined = source[["date", "peg", "close", "consNumber"]].merge(
            daily.reset_index(), on="date", suffixes=("_new_official", "_old"), validate="one_to_one")
        official.update(rows=len(joined),
                        pe_max_absolute_difference=float((joined.peg - joined.original_pe_official).abs().max()),
                        price_max_absolute_difference=float((joined.close_new_official - joined.close_old).abs().max()),
                        constituent_counts=sorted(joined.consNumber.dropna().unique().tolist()),
                        note="同一机构两个接口的数值核对；点位差不超过0.005，符合旧价格三位小数和新响应两位小数的精度差。300只不证明样本、股本或盈利口径恒定。")
        assert len(joined) == len(source)
        assert official["pe_max_absolute_difference"] < 1e-10
        assert official["price_max_absolute_difference"] <= 0.00500001

    definitions = []
    endpoints = [pd.Timestamp("2018-12-28"), pd.Timestamp("2019-04-30")]
    for field, price_field, label in [
        ("original_pe_official", "close", "中证原字段peg：沿用账本PE名称，未认证2019年静态/滚动及股本口径"),
        ("pe_static", "index_close_pe_source", "乐咕整体静态PE：供应商字段定义，历史首次版本未认证"),
        ("pe_ttm", "index_close_pe_source", "乐咕整体TTM PE：供应商字段定义，历史首次版本未认证"),
    ]:
        a, b = (daily.loc[t] for t in endpoints)
        price_log = 100 * np.log(b[price_field] / a[price_field])
        pe_log = 100 * np.log(b[field] / a[field])
        denom = (b[price_field] / b[field]) / (a[price_field] / a[field]) - 1
        definitions.append({"field": field, "definition": label, "price_field": price_field,
                            "start": endpoints[0], "end": endpoints[1],
                            "start_pe": float(a[field]), "end_pe": float(b[field]),
                            "pe_simple_change": float(b[field] / a[field] - 1),
                            "price_log_points": float(price_log), "pe_log_points": float(pe_log),
                            "implied_denominator_change_not_profit_growth": float(denom),
                            "implied_denominator_log_points": float(100 * np.log1p(denom)),
                            "verified_earnings": False})

    events = json.loads((OUT / "event_pricing.json").read_text(encoding="utf-8"))["rows"]
    full_price = pd.read_parquet(ROOT / INPUTS["index_price"])
    full_price["date"] = pd.to_datetime(full_price["date"])
    full_price = full_price.sort_values("date").set_index("date")
    clock_rows = []
    prior = json.loads((EXPECTATIONS / "comparison.json").read_text(encoding="utf-8"))["rows"]
    for row in events:
        latest = full_price.loc[full_price.index < pd.Timestamp(row["release_date"])].iloc[-1]
        position = full_price.index.get_loc(latest.name)
        paired_date = pd.Timestamp(row["latest_pricing"]["date"])
        assert pd.Timestamp(row["latest_pricing"]["available_at"]) <= pd.Timestamp(row["cutoff"])
        old_return = next(r for r in prior if r["stat_month"] == row["stat_month"])
        assert row["net_return_20d"] == old_return["net_return_20d"]
        assert row["net_return_5d"] == old_return["net_return_5d"]
        clock_rows.append({"release_date": row["release_date"], "paired_valuation_date": paired_date,
                           "previous_index_trading_date": latest.name,
                           "previous_index_close": float(latest.close),
                           "latest_price_change_since_paired_date": float(latest.close / full_price.loc[paired_date, "close"] - 1),
                           "previous_close_past60_price_return": float(latest.close / full_price.iloc[position - 60].close - 1),
                           "note": "本行只说明发布日前已发生的价格；不为旧PE改写可得时间，不替换原配对窗口。"})
    first, last = events[0], events[-1]
    gap = {"start": first["latest_pricing"]["date"], "end": last["latest_pricing"]["date"],
           "start_gap_pp": first["ey_minus_cgb_pp_descriptive_only"],
           "end_gap_pp": last["ey_minus_cgb_pp_descriptive_only"],
           "ey_change_pp": last["latest_pricing"]["ey_percent"] - first["latest_pricing"]["ey_percent"],
           "cgb_change_pp": last["same_date_cgb_10y"] - first["same_date_cgb_10y"],
           "is_expected_return_or_identified_risk_premium": False}
    save("definition_evidence.json", {"computed_at": now(), "official_reconciliation": official,
         "definition_comparisons": definitions, "publication_clock": clock_rows, "descriptive_yield_gap": gap,
         "earnings_series_verified": False,
         "remaining_gap": "没有取得2019年这四次发布时点、与指数口径一致的首次披露汇总利润序列；不能用P/PE或后来完整年报代替。",
         "checks": {"same_date_identity": "PASS", "official_values": "PASS" if probe["status"] == "RETRIEVED" else "NOT_RUN",
                    "paired_data_before_cutoff": "PASS", "saved_event_returns_unchanged": "PASS"}})
    print(json.dumps(clean({"official": official, "definitions": definitions, "clocks": clock_rows, "gap": gap}), ensure_ascii=False, indent=2), flush=True)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    daily = pd.read_parquet(OUT / "daily_valuation.parquet").loc["2018-12-28":"2019-05-16"]
    events = json.loads((OUT / "event_pricing.json").read_text(encoding="utf-8"))["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor("#f7f7f3")
    for ax in axes:
        ax.set_facecolor("#f7f7f3")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#ddded8", lw=.7)
        ax.set_axisbelow(True)
    for field, label, color, width, style in [
        ("close", "沪深300价格指数", "#222b37", 2.7, "-"),
        ("original_pe_official", "中证PE原字段", "#ba5638", 2.2, "-"),
        ("pe_ttm", "乐咕整体TTM PE", "#337f89", 1.5, "--"),
        ("pe_static", "乐咕整体静态PE", "#879444", 1.5, ":"),
    ]:
        axes[0].plot(daily.index, (daily[field] / daily.iloc[0][field] - 1) * 100,
                     label=label, color=color, lw=width, ls=style)
    for i, row in enumerate(events, 1):
        stamp = pd.Timestamp(row["release_date"])
        axes[0].axvline(stamp, color="#92958e", ls=":", lw=.9)
        axes[0].text(stamp, -.01, f"{i}", transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=9)
    axes[0].set_title("指数上涨与估值扩张：相对2018年末", loc="left", fontsize=13, pad=13)
    axes[0].set_ylabel("累计变化（%）")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    x = np.arange(len(events))
    before = [100 * next(v["price_simple_return"] for v in row["lookbacks"] if v["horizon"] == 60) for row in events]
    after = [100 * row["net_return_20d"] for row in events]
    for positions, values, color, label in [(x - .18, before, "#aaa997", "此前60日指数涨幅（估值配对日止）"),
                                           (x + .18, after, "#337f89", "原20日ETF事件净收益")]:
        bars = axes[1].bar(positions, values, .33, color=color, label=label)
        for bar, value in zip(bars, values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + (.65 if value >= 0 else -.65),
                         f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    axes[1].axhline(0, color="#59606a", lw=.8)
    axes[1].set_xticks(x, [f"{i+1}  {r['release_date'][5:]}\n配对至{r['latest_pricing']['date'][5:10]}" for i, r in enumerate(events)])
    axes[1].set_ylabel("涨跌幅 / 事件净收益（%）")
    axes[1].set_ylim(-14, 39)
    axes[1].set_title("信息出现前后的价格位置", loc="left", fontsize=13, pad=13)
    axes[1].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.suptitle("2019年1—4月：信用信息须结合此前指数定价解释", fontsize=17, x=.06, ha="left", y=.98)
    fig.text(.06, .062, "估值分解仅为恒等式；P/PE不是已核实利润。PE沿用下一交易日可用的原账本规则，3月事件配对至3月7日。", fontsize=9, color="#4d545b")
    fig.text(.06, .027, "ETF收益复用原10万元预算、成本与固定退出窗口；事件相互重叠，不是独立样本或完整账户。全图仅描述历史。", fontsize=9, color="#4d545b")
    fig.subplots_adjust(left=.065, right=.975, top=.86, bottom=.19, wspace=.24)
    fig.savefig(OUT / "指数估值与历史收益.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    months = json.loads((OUT / "monthend_pricing.json").read_text(encoding="utf-8"))
    events = json.loads((OUT / "event_pricing.json").read_text(encoding="utf-8"))["rows"]
    evidence = json.loads((OUT / "definition_evidence.json").read_text(encoding="utf-8"))
    total = months["full_period"]
    gap = evidence["descriptive_yield_gap"]
    local = lambda path, label: f"[{label}](<{path.as_posix()}>)"
    source_url = "https://www.csindex.com.cn/csindex-home/perf/index-perf?indexCode=000300&startDate=20181228&endDate=20190516"
    labels = {"2018-12": "社融调查预期缺失", "2019-01": "路透公布前调查：高于预期13900亿元",
              "2019-02": "第一财经公布前调查：低于预期6270亿元", "2019-03": "路透公布后回述：高于所述预期11160亿元"}
    lines = ["# 510300历史发现：整体估值、信用信息与已经发生的定价", "",
             "**本轮结论：指数整体已发生的重估，是解释消息后收益时不可省略的背景；这段历史仍不足以给出交易阈值。**", "",
             "沿用2019年1—4月四次金融数据公开日，以及既有20/60交易日描述。V5和V26已经做过价格/PE恒等式，本轮复用这项工作，将其连接到新补出的历史调查预期、利率、政策原因及原成本后收益；不把旧恒等式称为新发现。研究单位是沪深300整体，不扩展个股。", "",
             f"2018年12月28日至2019年4月30日，沪深300价格指数上涨{total['price_simple_return']:.2%}，原中证PE字段由10.71升至13.32，增加{total['pe_simple_change']:.2%}。按同日价格与PE恒等式，价格对数变化{total['price_log_points']:.2f}点，可写为PE对数变化{total['pe_log_points']:.2f}点，加P/PE隐含分母变化{total['implied_denominator_log_points']:.2f}点。它表明价格上涨主要伴随该估值倍数扩张，不能解释成‘这部分由某项政策造成’。", "",
             "**四次信息公开时，市场处在不同价格位置**", "",
             "| 发布日 | 社融信息证据 | 当时可用的PE配对日 | 原中证PE | 配对日前60交易日指数涨幅 | 原5日事件净收益 | 原20日事件净收益 |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in events:
        lookback = next(v for v in row["lookbacks"] if v["horizon"] == 60)
        lines.append(f"| {row['release_date']} | {labels[row['stat_month']]} | {row['latest_pricing']['date'][:10]} | {row['latest_pricing']['original_pe_official']:.2f} | {lookback['price_simple_return']:+.2%} | {row['net_return_5d']:+.2%} | {row['net_return_20d']:+.2%} |")
    march_clock = next(r for r in evidence["publication_clock"] if r["release_date"] == "2019-03-10")
    lines += ["",
              "2月的强信用数据，出现在此前60日指数涨幅约5%的位置；4月强信用报道出现时，配对指数此前60日已上涨约30%，估值倍数也明显抬升。相同方向的信用消息，并不意味着相同的剩余收益空间。这是历史背景差异，尚未证明前期涨幅或PE导致了收益差异。", "",
              "3月是必须保留的反例：调查所示信用低于预期，指数此前也已经上涨，但随后原20日净收益仍为正。因而不能把本表翻译成‘超预期就买’、‘PE高于某值就卖’，也不能凭4月这一例拟合涨幅或估值阈值。", "",
              "上述PE沿用旧账本‘下一ETF交易日开盘可用’的保守规则，在各公开日21:00截取；这不是对当年网站实际发布时间的认证。3月10日为周日，旧规则令3月8日PE到3月11日才可用，所以配对日期停在3月7日。"
              f"然而3月8日指数收盘价早已形成，较3月7日下跌{-march_clock['latest_price_change_since_paired_date']:.2%}；以3月8日收盘单看此前60日价格，涨幅为{march_clock['previous_close_past60_price_return']:.2%}。两种时间口径并列保留，不能把配对日价格误说成公告前的最新价格。", "",
              "原事件收益仍是510300次一交易日开盘进入、原5/20日终点退出的成本后收益，使用10万元预算并以实际投入为分母；佣金、滑点和退出均未重算或更改。它们不是20万元完整账户结果，彼此有窗口重叠，也不是四次独立成功/失败试验。4月窗口完整保留5月贸易冲击，不能把全部下跌归因于4月信用消息。", "",
              "**利率变化不能单独解释估值变化**", "",
              f"在1月14日至4月11日这两个配对节点，100/PE由{events[0]['latest_pricing']['ey_percent']:.2f}%降至{events[-1]['latest_pricing']['ey_percent']:.2f}%，同日期10年国债由{events[0]['same_date_cgb_10y']:.4f}%升至{events[-1]['same_date_cgb_10y']:.4f}%；两者相减由{gap['start_gap_pp']:.2f}降至{gap['end_gap_pp']:.2f}个百分点。这里主要变化是股价相对账面PE分母已经抬升，并非同期10年国债利率一路下行。", "",
              "100/PE与国债收益率之差仅是描述性对照。股票利润并非保证分红，增长、风险和期限也不同，因此本轮不把它命名为已识别的风险溢价、未来收益率或买入信号。", "",
              "上一轮已经定位：税期、现金回笼、期限替换与财政支出都影响央行操作量；3月31日PMI改善早于4月信用发布，国债利率的一部分上行也已提前发生。把这些材料与本轮价格接起来，更合理的历史解释是：融资改善、增长判断、后续宽松必要性和市场已经付出的价格相互作用。现有证据尚不能分离各自因果贡献。详见" + local(RATES / "历史发现_资金价格与指数政策传导.md", "资金价格与政策历史研究") + "。", "",
              "**盈利是否已经改善：现有数据不足以作同口径确认**", "",
              "| 估值口径 | 2018年末PE | 2019年4月末PE | PE变化 | 同源价格/PE隐含分母变化（不是利润增速） |",
              "|---|---:|---:|---:|---:|"]
    names = {"original_pe_official": "中证原字段peg", "pe_static": "乐咕整体静态PE", "pe_ttm": "乐咕整体TTM PE"}
    for row in evidence["definition_comparisons"]:
        lines.append(f"| {names[row['field']]} | {row['start_pe']:.2f} | {row['end_pe']:.2f} | {row['pe_simple_change']:+.2%} | {row['implied_denominator_change_not_profit_growth']:+.2%} |")
    lines += ["",
              "几套PE都上升，支持‘整体价格上涨伴随估值扩张’的描述；它们却给出明显不同的隐含分母变化。因此，不能把中证这一列约4.51%的P/PE上升直接叫作‘沪深300利润增长4.51%’。静态与TTM的报告期不同，股本、样本、亏损处理、除数和更新时点也可能影响结果；本轮没有把各口径平均、拼接或选择最符合结论的一列。", "",
              "本轮中证官网请求取得2018年12月28日至2019年5月16日共89行，peg与旧PE记录逐行一致，点位差最多0.005，符合显示精度差。但响应没有提供2019年PE的股本与盈利分母定义；同机构两个接口一致，也不等于独立来源或当年首次版本认证。当前字段被第三方翻译为滚动PE、或与另一日期事实表数值相同，都不足以倒推2019年的完整定义。[中证历史接口原址](" + source_url + ")。", "",
              "目前没有取得四次公开时点、与指数口径一致的首次披露汇总利润序列。旧公司财务输入存在后来更新的依赖，也不能直接拿来补齐。当年5月或更晚的完整年度/季度总结，即使描述2018年度，也不能变成1—4月已经知道的盈利。这里保留缺口，停止继续泛搜财报，不将个股研究重新变成主线。", "",
              "**对下一项历史发现的具体约束**", "",
              "保留一个待对照的机制判断：政策与信用变化的作用，要同时考虑增长信息、后续宽松空间和此前指数重估程度。它还不是可交易规则。此后应换一个固定日历阶段，而不是继续给2019年4月失败案例增加解释或筛选条件。", "",
              "下一段固定考察2018年7—12月这六个月的金融数据发布：复用已经保存的全部发布事件和20日成本后收益，补出公布前调查证据及指数整体定价；调查缺失就保留缺失。同时只对关键转折补当时的政策理由，比较同样的信用/宽松信息在此前估值变化不同的阶段是否留下不同收益，不按后续涨跌分组、不拟合阈值。此阶段属于既有历史再发现，不能标成独立验证。", "",
              "本轮仅完成日期、同日恒等式、89行官网数值及原收益保留的必要核对；没有新增全账户回测、参数拟合或达标声明。VAL04既有估值策略失败保持原状态。夏普1.2和年化10%的完整账户联合目标仍未实现。", "",
              "文件：" + "、".join([local(OUT / "event_pricing.json", "四次发布定价"), local(OUT / "monthend_pricing.json", "月末分解"),
                                           local(OUT / "definition_evidence.json", "口径对照与时间说明"), local(OUT / "指数估值与历史收益.png", "历史图"),
                                           local(ROOT / "research/historical_index_valuation_repricing_v1.py", "研究脚本")]) + "。", ""]
    name = "历史发现_指数整体估值与已发生定价.md"
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")
    draw()
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    result.update(report=name, report_completed_at=now(), official_historical_rows=evidence["official_reconciliation"].get("rows", 0),
                  discovery="2019年1至4月指数涨约30%伴随多口径整体PE扩张；4月强信用之前已大幅重估，但3月弱信用后仍上涨，说明预期差须连同先前定价及政策反应解释。P/PE隐含分母不能认证实际利润增长。",
                  next_historical_question="固定2018年7至12月六个金融数据公开月，复用全部已存事件、成本后收益及指数整体价格/估值，补公布前调查与关键政策原因；与2019年1至4月对照，不按收益分期或拟合阈值，仍非独立验证。",
                  hypothesis_status="MECHANISM_CANDIDATE_NOT_TRADING_RULE", causal_contribution_identified=False)
    save("result.json", result)
    print("已生成指数整体估值报告与历史图；保留盈利口径缺失和3月反例，未新增策略规则。", flush=True)


def record_progress():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    assert (OUT / result["report"]).is_file()
    prefix = "reports/research/510300_historical_index_valuation_repricing_v1/"
    stamp, changes = now(), []
    for name in ["510300_historical_cause_discovery_v1.json", "510300_existing_data_training_mandate_v1.json"]:
        path = ROOT / "config" / name
        cfg = json.loads(path.read_text(encoding="utf-8"))
        before = dict(cfg)
        if name == "510300_historical_cause_discovery_v1.json":
            cfg.update(current_study=prefix + "protocol.json", latest_completed_study=prefix + "result.json", latest_report=prefix + result["report"], updated_at=stamp)
        else:
            cfg.update(current_round=STUDY, latest_progress_receipt=prefix + "result.json",
                       latest_historical_index_valuation_repricing=prefix + "result.json",
                       latest_continuation_report=prefix + result["report"], latest_historical_report=prefix + result["report"],
                       latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp, goal_status="active", goal_achieved=False,
                       local_goal_work_status="ACTIVE_HISTORICAL_ONLY", last_research_result=result["discovery"],
                       last_source_result="中证官网89行2019年历史数值与旧记录一致；三套整体PE分别对照，尚未认证当时同口径实际盈利。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": str(path.relative_to(ROOT)), "fields": {k: {"before": before.get(k), "after": v} for k, v in cfg.items() if before.get(k) != v}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "study_id": STUDY, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": "PROGRESS_COMPLETED_INDEX_VALUATION_HISTORY", "changes": changes,
                                  "goal_achieved": False, "orders_authorized": False})
    print("已登记指数估值历史发现；完整账户目标保持未达成。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="指数整体估值与历史信息时序")
    parser.add_argument("mode", choices=["prepare", "source-probe", "build", "definitions", "report", "record-progress"])
    args = parser.parse_args()
    {"prepare": prepare, "source-probe": source_probe, "build": build, "definitions": definition_evidence,
     "report": report, "record-progress": record_progress}[args.mode]()


if __name__ == "__main__":
    main()
