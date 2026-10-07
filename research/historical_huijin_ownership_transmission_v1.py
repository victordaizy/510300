"""复用历史原件，分解ETF持有变化、申赎口径和价格响应先后。"""
from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_huijin_ownership_transmission_v1"
HOLDER_ROOT = ROOT / "reports/research/510300_participant_identity_clock_v1"
QUARTER_ROOT = ROOT / "reports/research/510300_original_fund_subscription_reports_v1"
QUARTER_PDFS = ROOT / "data/raw/510300_original_fund_subscription_reports_v1/quarterly_pdfs"
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
SHARES = ROOT / "data/raw/flow/510300_etf_share_premium_level_full_v1.parquet"
NAV = ROOT / "data/raw/fund/510300_nav_daily_raw.parquet"
OLD_ACCOUNT = ROOT / "reports/research/510300_huijin_etf_event_saved_completion_v1/result.json"
D = Decimal
HUNDRED_MILLION = D(100_000_000)

# 源PDF第13页逐行转录；序号按各期原表，不能当成永久身份。
# 字段为：季度、序号、20%区间、期初、原表申购、原表赎回、期末、期末占比。
ANONYMOUS_ROWS = [
    ("2024Q1", 1, "20240116-20240331", "6247371001", "26355945058", "0", "32603316059", "58.93"),
    ("2024Q2", 1, "20240401-20240630", "32603316059", "3051282800", "0.00", "35654598859", "58.70"),
    ("2024Q3", 1, "20240701-20240930", "35654598859", "0.00", "0.00", "35654598859", "36.79"),
    ("2024Q3", 2, "20240730-20240930", "728041500", "25893286844", "0.00", "26621328344", "27.47"),
    ("2024Q4", 1, "20241001-20241231", "35654598859", "0.00", "0.00", "35654598859", "39.89"),
    ("2024Q4", 2, "20241001-20241231", "26621328344", "0.00", "0.00", "26621328344", "29.78"),
    ("2025Q1", 1, "20250101-20250331", "35654598859", "0.00", "0.00", "35654598859", "41.92"),
    ("2025Q1", 2, "20250101-20250331", "26621328344", "362999800", "0.00", "26984328144", "31.72"),
    ("2025Q2", 1, "20250401-20250630", "26984328144", "10874146830", "-", "37858474974", "40.26"),
    ("2025Q2", 2, "20250401-20250630", "35654598859", "-", "-", "35654598859", "37.91"),
    ("2025Q3", 1, "20250701-20250930", "37858474974", "-", "-", "37858474974", "42.20"),
    ("2025Q3", 2, "20250701-20250930", "35654598859", "-", "-", "35654598859", "39.74"),
    ("2025Q4", 1, "20251001-20251231", "37858474974", "-", "-", "37858474974", "42.62"),
    ("2025Q4", 2, "20251001-20251231", "35654598859", "-", "-", "35654598859", "40.14"),
]


def write_json(name: str, value: object) -> None:
    def encode(x):
        if isinstance(x, D):
            return format(x, "f")
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        raise TypeError(type(x).__name__)
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False) + "\n", encoding="utf-8")


def dec(value: object) -> Decimal:
    return D(str(value))


def billion_units(value: object) -> str:
    return f"{dec(value) / HUNDRED_MILLION:+.2f}"


def interval_text(lo: D, hi: D) -> str:
    return billion_units(lo) if lo == hi else f"{billion_units(lo)} 至 {billion_units(hi)}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    holders = pd.read_parquet(HOLDER_ROOT / "facts/持有结构与公开时钟.parquet")
    holders = holders[pd.to_datetime(holders.period_end).between("2023-12-31", "2025-12-31")].sort_values("period_end")
    quarters = pd.read_parquet(QUARTER_ROOT / "quarterly_share_flow_facts_v1_1.parquet")
    quarters = quarters[quarters.period.str.match(r"202[45]Q[1-4]")].sort_values("period").set_index("period")
    market = pd.read_parquet(MARKET).sort_values("date")
    market["date"] = pd.to_datetime(market.date)

    def price_return(start: str, end: str) -> dict:
        a = market[market.date <= pd.Timestamp(start)].iloc[-1]
        b = market[market.date <= pd.Timestamp(end)].iloc[-1]
        return {"requested_start": start, "requested_end": end,
                "start_session": a.date.strftime("%Y-%m-%d"), "end_session": b.date.strftime("%Y-%m-%d"),
                "total_return_before_cost": float(b.wealth / a.wealth - 1),
                "measurement": "含分红价格回报；未扣交易费；不是完整账户"}

    snapshots = []
    sources = []
    for row in holders.to_dict("records"):
        n, inst, personal, feeder = [dec(row[k]) for k in ["total_units", "institution_excluding_feeder_units", "direct_personal_units", "feeder_units"]]
        assert n == inst + personal + feeder, "持有结构分项未对齐"
        lo, hi = dec(row["huijin_direct_lower"]), dec(row["huijin_direct_upper"])
        snapshots.append({"period_end": row["period_end"], "total_units": n,
            "huijin_lower": lo, "huijin_upper": hi,
            "other_lower": n - hi, "other_upper": n - lo,
            "other_institutions_lower": inst - hi, "other_institutions_upper": inst - lo,
            "personal_units": personal, "feeder_units": feeder,
            "huijin_weight_lower": lo / n, "huijin_weight_upper": hi / n,
            "publication_date": row["publication_date_upper_bound"], "first_available_session": row["first_available_session"],
            "source_url": row["source_url"], "holder_page": int(row["holder_section_page"]),
            "source_pdf": str((HOLDER_ROOT / row["source_pdf"]).relative_to(ROOT))})
        sources.append({"period": row["period_end"], "url": row["source_url"],
                        "file": str((HOLDER_ROOT / row["source_pdf"]).relative_to(ROOT)),
                        "page": int(row["holder_section_page"]), "publication_date": row["publication_date_upper_bound"]})
    halfyears = []
    for a, b in zip(snapshots, snapshots[1:]):
        dn = b["total_units"] - a["total_units"]
        h_lo, h_hi = b["huijin_lower"] - a["huijin_upper"], b["huijin_upper"] - a["huijin_lower"]
        oi_lo = b["other_institutions_lower"] - a["other_institutions_upper"]
        oi_hi = b["other_institutions_upper"] - a["other_institutions_lower"]
        dp, df = b["personal_units"] - a["personal_units"], b["feeder_units"] - a["feeder_units"]
        assert dn - h_hi == oi_lo + dp + df and dn - h_lo == oi_hi + dp + df
        halfyears.append({"period": b["period_end"][:4] + ("H1" if b["period_end"][5:7] == "06" else "H2"),
            "start": a["period_end"], "end": b["period_end"], "total_change": dn,
            "huijin_change_lower": h_lo, "huijin_change_upper": h_hi,
            "other_change_lower": dn - h_hi, "other_change_upper": dn - h_lo,
            "other_institution_change_lower": oi_lo, "other_institution_change_upper": oi_hi,
            "personal_change": dp, "feeder_change": df, "end_huijin_weight": b["huijin_weight_lower"],
            "full_identity_available_session": b["first_available_session"],
            "contemporaneous_price_return": price_return(a["period_end"], b["period_end"])})

    anon = []
    for period, number, interval, begin, inc, red, end, weight in ANONYMOUS_ROWS:
        q = quarters.loc[period]
        inc_d, red_d = D(0) if inc == "-" else D(inc), D(0) if red == "-" else D(red)
        assert D(begin) + inc_d - red_d == D(end), "匿名表转录未对齐"
        pdfs = list(QUARTER_PDFS.glob(period + "_*.pdf"))
        assert len(pdfs) == 1
        anon.append({"period": period, "reported_institution_number": number, "at_least_20_percent_interval": interval,
            "begin_units": D(begin), "reported_subscription_raw": inc, "reported_redemption_raw": red,
            "reported_subscription_numeric": inc_d, "reported_redemption_numeric": red_d,
            "end_units": D(end), "reported_end_percent": weight,
            "net_holding_change": D(end) - D(begin),
            "identity_status": "ANONYMOUS_IN_QUARTERLY_REPORT_QUANTITY_MATCH_IS_INFERENCE",
            "source_url": q.source_url, "source_pdf": str(pdfs[0].relative_to(ROOT)), "source_page": 13,
            "publication_date": q.publication_date, "first_available_session": q.feature_available_session})
    qrecords = []
    for period, q in quarters.iterrows():
        g = [r for r in anon if r["period"] == period]
        dn = dec(q.ending_units) - dec(q.beginning_units)
        gross, redeemed = dec(q.gross_subscription_units), dec(q.gross_redemption_units)
        assert dn == gross - redeemed and dec(q.split_delta_units) == 0
        dg = sum((r["net_holding_change"] for r in g), D(0))
        calendar = pd.Period(period, freq="Q")
        qrecords.append({"period": period, "total_change": dn, "gross_fund_subscriptions": gross,
            "gross_fund_redemptions": redeemed, "disclosed_large_holder_count": len(g),
            "disclosed_large_holder_change": dg, "remaining_holder_change": dn - dg,
            "cohort_caveat": "逐季披露的大户集合；2024上半年前两季只列一户，余项不是全部非汇金持有",
            "contemporaneous_price_return": price_return((calendar.start_time - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), calendar.end_time.strftime("%Y-%m-%d")),
            "publication_date": q.publication_date, "first_available_session": q.feature_available_session,
            "source_url": q.source_url})
        sources.append({"period": period, "url": q.source_url, "file": g[0]["source_pdf"],
                        "pages": [12, 13], "publication_date": q.publication_date})

    share_frame = pd.read_parquet(SHARES)
    checkpoint = share_frame[pd.to_datetime(share_frame.date).eq(pd.Timestamp("2024-07-30"))]
    assert len(checkpoint) == 1
    checkpoint = checkpoint.iloc[0]
    g3 = next(r for r in anon if r["period"] == "2024Q3" and r["reported_institution_number"] == 2)
    checkpoint_n = dec(checkpoint.fund_shares)
    minimum_holding = checkpoint_n * D("0.2")
    minimum_increase = minimum_holding - g3["begin_units"]
    timing = {"date": "2024-07-30", "total_etf_units": checkpoint_n,
        "share_data_source": checkpoint.share_source, "share_retrieved_at": str(checkpoint.share_retrieved_at),
        "large_holder_minimum_units": minimum_holding, "quarter_start_units": g3["begin_units"],
        "minimum_net_increase_by_date": minimum_increase,
        "quarter_net_increase": g3["net_holding_change"],
        "minimum_fraction_of_quarter_net_increase": minimum_increase / g3["net_holding_change"],
        "original_report_publication_date": g3["publication_date"],
        "not_known_on_checkpoint_date_from_this_report": True,
        "assumptions": "采用报告20%区间与当日总份额同一统计口径；历史日份额存在精度差，展示到0.01亿份，不把计算余位当精确下界。只约束累计净持有变化，不还原逐日买卖。",
        "identity_inference": "该匿名行期初728041500份及期末26621328344份分别与2024中报及年报汇金资管实名量吻合；季报未披露姓名。"}
    one = next(r for r in anon if r["period"] == "2024Q1")
    scope_difference = {"period": "2024Q1", "single_holder_subscription_column": one["reported_subscription_numeric"],
        "whole_fund_gross_subscription": dec(quarters.loc["2024Q1", "gross_subscription_units"]),
        "difference": one["reported_subscription_numeric"] - dec(quarters.loc["2024Q1", "gross_subscription_units"]),
        "finding": "单一持有人原列大于全基金总申购，不能把两栏直接看成同一一级申购口径或据此计算新增现金；具体买入或转入路线仍未识别。"}
    price_windows = [price_return(a, b) for a, b in [
        ("2024-06-28", "2024-07-30"), ("2024-07-30", "2024-09-23"),
        ("2024-06-28", "2024-09-23"), ("2024-09-23", "2024-09-30")]]
    nav = pd.read_parquet(NAV)
    nav["date"] = pd.to_datetime(nav.date)
    nav = nav.set_index("date")
    a, b = nav.loc[pd.Timestamp("2024-09-23")], nav.loc[pd.Timestamp("2024-09-30")]
    assert market[market.date.between("2024-09-24", "2024-09-30")].dividend.eq(0).all()
    nav_growth = float(b.unit_nav / a.unit_nav - 1)
    price_growth = price_windows[-1]["total_return_before_cost"]
    premium_effect = float((1 + b.close_premium_to_nav) / (1 + a.close_premium_to_nav) - 1)
    assert abs((1 + nav_growth) * (1 + premium_effect) - (1 + price_growth)) < 1e-9
    price_decomposition = {"start": "2024-09-23", "end": "2024-09-30", "dividend_during_window": 0,
        "start_unit_nav": float(a.unit_nav), "end_unit_nav": float(b.unit_nav),
        "nav_growth": nav_growth, "market_price_growth": price_growth,
        "start_premium": float(a.close_premium_to_nav), "end_premium": float(b.close_premium_to_nav),
        "multiplicative_premium_growth": premium_effect,
        "price_return_minus_nav_return_percentage_points": (price_growth - nav_growth) * 100,
        "identity": "1+价格收益=(1+净值收益)*(1+期末溢价)/(1+期初溢价)；本段无分红",
        "source_file": str(NAV.relative_to(ROOT)), "primary_source": str(a.source_primary),
        "secondary_source": str(a.source_secondary), "source_retrieved_at": str(a.retrieved_at),
        "purpose": "解释固定历史区间的价格口径，不新选收益区间或交易参数。"}
    old = json.loads(OLD_ACCOUNT.read_text(encoding="utf-8"))
    inherited = [{k: row[k] for k in ["policy", "cost", "days", "sharpe", "annual_return", "max_drawdown", "completed_cycles", "historical_point_targets_met"]}
                 for row in old["all_accounts"] if row["cost"] == "STRESS"]
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    result = {"study_id": protocol["study_id"], "completed_at": now,
        "status": "HISTORICAL_OWNERSHIP_COMPOSITION_AND_BUYING_PRICE_TIMING_DISTINGUISHED",
        "halfyear_periods": len(halfyears), "quarter_periods": len(qrecords), "quarterly_anonymous_records": len(anon),
        "named_snapshots": len(snapshots), "timing_bound": timing, "incompatible_subscription_columns": scope_difference,
        "price_windows": price_windows, "price_nav_premium_decomposition": price_decomposition,
        "prior_frozen_account_source": str(OLD_ACCOUNT.relative_to(ROOT)),
        "prior_stress_accounts_unchanged": inherited,
        "reused_inputs": [str(p.relative_to(ROOT)) for p in [
            HOLDER_ROOT / "facts/持有结构与公开时钟.parquet",
            QUARTER_ROOT / "quarterly_share_flow_facts_v1_1.parquet", MARKET, SHARES, NAV, OLD_ACCOUNT]],
        "new_models": 0, "new_parameter_searches": 0, "new_full_accounts": 0, "new_prospective_forecasts": 0,
        "minimal_checks": {"halfyear_component_identities": True, "eight_quarter_creation_redemption_identities": True,
                           "fourteen_anonymous_rows_reconciled": True, "nine_source_pages_visually_read": True,
                           "fixed_window_price_nav_premium_identity": True},
        "goal_achieved": False, "orders_authorized": False,
        "next_question": "停止用总份额替代全市场风险偏好。转查2024年1—2月原始杠杆约束、强制减仓及政策化解资料，区分主动持有减少和融资约束下的被动卖出；先复用已有融资研究，不改旧失败模型。"}
    for filename, obj in [("期末实名持有及公开日期.json", snapshots), ("四个半年持有变化分解.json", halfyears),
                          ("八份季报匿名大户原列.json", anon), ("八个季度申赎与持有变化.json", qrecords),
                          ("购买发生期与价格响应.json", {"bound": timing, "price_windows": price_windows,
                           "price_nav_premium_decomposition": price_decomposition}),
                          ("source_index.json", sources), ("result.json", result)]:
        write_json(filename, obj)

    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1.15]})
    x = np.arange(4)
    for offset, keys, color, label in [(-0.24, ("total_change", "total_change"), "#708497", "ETF总份额变化"),
            (0, ("huijin_change_lower", "huijin_change_upper"), "#007F77", "两家汇金持有变化"),
            (0.24, ("other_change_lower", "other_change_upper"), "#C3754C", "其余持有人变化")]:
        lo = np.array([float(r[keys[0]] / HUNDRED_MILLION) for r in halfyears])
        hi = np.array([float(r[keys[1]] / HUNDRED_MILLION) for r in halfyears])
        mid = (lo + hi) / 2
        axes[0].bar(x + offset, mid, width=0.22, color=color, label=label, yerr=(hi-lo)/2, capsize=3)
        for xp, value in zip(x + offset, mid):
            axes[0].text(xp, value + (7 if value >= 0 else -7), f"{value:+.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    axes[0].axhline(0, color="#7D8791", linewidth=.8)
    axes[0].set_xticks(x, [r["period"].replace("H1", "上半年").replace("H2", "下半年") for r in halfyears])
    axes[0].set_ylabel("亿份（不是金额）")
    axes[0].set_ylim(-100, 375)
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    axes[0].set_title("同样的份额增长，背后可以是不同持有人在相反方向变化", loc="left", fontweight="bold", pad=14)
    axes[0].text(0, -0.17, "2024上半年保留2023年末未进前十的持有界限；柱上数字为区间中点，原始区间见报告。", transform=axes[0].transAxes, fontsize=9, color="#596573")
    curve = market[market.date.between("2024-06-28", "2024-09-30")].copy()
    axes[1].plot(curve.date, curve.wealth / curve.wealth.iloc[0] * 100, color="#007F77", linewidth=2.2)
    for date, color in [("2024-07-30", "#C3754C"), ("2024-09-24", "#708497")]:
        axes[1].axvline(pd.Timestamp(date), color=color, linestyle="--", linewidth=1)
    axes[1].axhline(100, color="#AAB3BC", linewidth=.7)
    axes[1].set_title("2024年三季度：7月底已有大额持有，价格响应仍需另找原因", loc="left", fontweight="bold", pad=14)
    axes[1].text(.025, .93, f"季报约束：截至7/30，匿名第二大户净增至少 {minimum_increase/HUNDRED_MILLION:.2f} 亿份\n达到本季净增加量的至少 {minimum_increase/g3['net_holding_change']:.2%}；这条证据10/25才披露", transform=axes[1].transAxes, va="top", fontsize=10)
    axes[1].text(.025, .50, f"9/24政策发布；后段多项政策共同影响\n9/30收盘溢价约{b.close_premium_to_nav:.2%}，价格涨幅含溢价扩大", transform=axes[1].transAxes, ha="left", va="bottom", fontsize=9, color="#596573")
    axes[1].set_ylabel("含分红价格指数（6/28 = 100）")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axes[1].grid(axis="y", alpha=.18)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("510300历史机制发现｜持有者、申赎与价格的三个时钟", x=.075, ha="left", fontsize=17, fontweight="bold")
    fig.text(.075, .02, "资料：基金年报、中报、8份季报及既有历史行情。价格未扣交易成本；本图不是策略账户或当时已知的信号。", fontsize=9, color="#596573")
    fig.subplots_adjust(left=.075, right=.98, top=.91, bottom=.075, hspace=.48)
    fig.savefig(OUT / "持有变化与价格先后.png", dpi=170)
    plt.close(fig)

    text = ["# 510300历史发现：谁在增持、谁在减少持有，以及价格何时响应", "",
        "本轮按用户要求只做历史检验与发现。新增的是持有结构、季度发生期与价格响应的分解，没有新建前瞻任务，也没有把旧公告策略改参数重跑。", "",
        f"**最有区分力的发现：2024年三季报的第二大匿名机构，7月30日已达到20%持有比例。由当日总份额推得，它相对季初至少净增 {minimum_increase/HUNDRED_MILLION:.2f} 亿份，至少占本季净增加量的 {minimum_increase/g3['net_holding_change']:.2%}。大额持有增加早于9月24日政策发布；其后的价格表现不能只用季度末增持量解释。**", "",
        "![历史持有分解和价格先后](持有变化与价格先后.png)", "",
        "用期末数量恒等式分清：ETF总份额变化 = 两家汇金直接实名持有变化 + 其余持有人变化。下表单位均为亿份，不是资金金额。", "",
        "| 固定日历期 | ETF总份额变化 | 两家汇金持有变化 | 其余持有人变化 | 期末汇金占比 | 同期含分红价格回报 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in halfyears:
        text.append(f"| {r['period']} | {billion_units(r['total_change'])} | {interval_text(r['huijin_change_lower'],r['huijin_change_upper'])} | {interval_text(r['other_change_lower'],r['other_change_upper'])} | {r['end_huijin_weight']:.2%} | {r['contemporaneous_price_return']['total_return_before_cost']:+.2%} |")
    text += ["", "2023年末汇金资管不在前十名，旧实名表只支持0至3.07567448亿份的范围，未直接填零。2024年年报匿名大户表给出某行期初为零，与后续实名数量吻合；由于身份仍由数量匹配推断，本表继续使用保守界限。2025年下半年总份额减少而两家持有不变，是既有研究已确认的事实，本轮补充完整四期分解，不把它重复记成新发现。", "",
        "2025年上半年，总份额增加46.557亿份由两股相反变化合成：两家汇金增加112.37146630亿份，其余持有人减少65.81446630亿份。进一步拆开，其余机构减少35.66480828亿份，个人直接持有减少31.11738814亿份，联接基金增加0.96773012亿份。这里的‘个人’仅指直接登记个人账户；不能推及个人通过其他机构产品的全部权益暴露。", "",
        "这说明持有结构向特定大户集中，不能据此把当时的总份额上升描述成普遍风险偏好回升。也不能直接断言汇金接走了哪一批卖单：大户在二级市场买入、其他人赎回、大户申购、非交易转移等路线，都可能产生相近的期末数量。", "",
        "把半年拆到全部八个季度，能看到持有增加发生的阶段。匿名大户表只披露曾达到20%的投资者，2024年前两个季度各一户，其余季度各两户；其余变化按当季已披露集合求剩余，集合并不固定。", "",
        "| 季度 | 全基金总申购 | 全基金总赎回 | 净份额变化 | 披露大户持有净变化 | 其余持有净变化 | 同期含分红价格回报 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in qrecords:
        text.append(f"| {r['period']} | {r['gross_fund_subscriptions']/HUNDRED_MILLION:.2f} | {r['gross_fund_redemptions']/HUNDRED_MILLION:.2f} | {billion_units(r['total_change'])} | {billion_units(r['disclosed_large_holder_change'])} | {billion_units(r['remaining_holder_change'])} | {r['contemporaneous_price_return']['total_return_before_cost']:+.2%} |")
    text += ["", "2024年三季度第二行期初7.280415亿份、期末266.21328344亿份，分别与当年中报和年报汇金资管实名数量吻合，这是身份推断而非季报明示姓名。2025年二季度机构序号发生交换，本轮按数量衔接，未把序号当作固定账户。二季度新增108.74146830亿份占2025上半年两家净增量约96.77%，但季度表不能把这一数量全部归到4月7日单日公告。", "",
        f"2024年一季度尤其暴露口径问题：全基金总申购为 {scope_difference['whole_fund_gross_subscription']/HUNDRED_MILLION:.2f} 亿份，单个机构持有变化表的‘申购’却为 {scope_difference['single_holder_subscription_column']/HUNDRED_MILLION:.8f} 亿份，后者大 {scope_difference['difference']/HUNDRED_MILLION:.8f} 亿份。已逐页核对第12、13页；不是把两个表误拼造成。它至少否定‘该持有人栏全部等于同口径一级申购’的读法，不能进一步把差额直接叫作已识别的二级净买入或资金流入。", "",
        f"7月30日约束使用的历史总份额约为{checkpoint_n/HUNDRED_MILLION:.2f}亿份，20%约为{minimum_holding/HUNDRED_MILLION:.2f}亿份，再减季初7.280415亿份，得到约{minimum_increase/HUNDRED_MILLION:.2f}亿份净增下界。日份额源有精度差，不把计算余位当精确下界。总份额来自后来回补的数据；匿名比例区间在10月25日季报才公开，不能把这个下界提前当作7月交易信号。", "",
        "| 固定时间段 | 含分红价格回报 |", "| --- | ---: |"]
    for r in price_windows:
        text.append(f"| {r['start_session']}收盘至{r['end_session']}收盘 | {r['total_return_before_cost']:+.2%} |")
    text += ["", f"后段还需要拆价格与净值：9月23日至30日基金单位净值从{a.unit_nav:.4f}升至{b.unit_nav:.4f}，增长{nav_growth:.2%}；收盘溢价从{a.close_premium_to_nav:.3%}扩大至{b.close_premium_to_nav:.3%}，所以成交价格增长{price_growth:.2%}。价格收益高于净值收益约{(price_growth-nav_growth)*100:.2f}个百分点，包含溢价变化及乘法交互，不能全部解释成底层股票上涨。本段没有分红，乘法恒等式已对齐；净值复用既有东方财富与新浪双源表。", "",
        "以上时间段由报告区间和9月24日政策日期确定，没有搜索最佳转折日。它们用于描述先后，不是可交易信号收益。9月24日政策组合及其后9月26日政治局会议共同构成竞争解释，不能将后段涨幅全部归给某个工具或某类买盘。[9月24日国新办发布会原文](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)。", "",
        "机制链条应当写成：压力与政策目标影响特定主体的购买；基金总份额只显示申赎净额，持有人之间还有转移；其他持有人可能继续减少持有；价格同时反映卖方约束、政策信息与盈利预期。历史证据支持把这些环节分开，但没有识别‘若没有汇金购买，价格本会是多少’的反事实，也没有查明每个减持者的动机。", "",
        "已公开的汇金说明将市场稳定作为目标，并列出自有资金、分红、市场融资及央行流动性支持等资金来源。因此存在逆周期购买、与其他持有者意愿相反的机制基础；不能把持有量直接等同于盈利预期上调。[2025年4月8日汇金答记者问](https://www.huijin-inv.cn/huijin-inv/SC20252/2025-04/1002847.shtml)。", "",
        "交易层仍复用原有公告账户的结果。该研究从2015年初至2026年9月24日保留2852个交易日、20万元完整账户与原风险约束；这些长区间指标只是既有证据，本轮没有要求因子再做十年检验。压力成本下：", "",
        "| 既有冻结规则 | 净夏普 | 年化收益 | 最大回撤 | 完整周期 |", "| --- | ---: | ---: | ---: | ---: |"]
    names = {"DISCLOSURE_ONLY": "公告触发", "DISCLOSURE_PRICE_CONFIRMATION": "公告后价格确认"}
    for r in inherited:
        text.append(f"| {names[r['policy']]} | {r['sharpe']:.4f} | {r['annual_return']:.4%} | {r['max_drawdown']:.2%} | {r['completed_cycles']} |")
    text += ["", "本轮的实用结论是：大户增持可以是市场压力下的承接行为，持有占比上升也可能来自其余持有人退出；它们与价格立即上涨没有固定对应。使用因子前要问谁改变了持有、是新增购买还是分母变化、变化何时发生、价格之后为何仍走弱或转强。旧公告规则尚无费用后优势，不能靠本次事后分解改名为有效策略。", "",
        "目标仍未达到：新账户0、参数搜索0、前瞻预测0。本轮只核对分项恒等式及关键原页，不添加无关审计。下一环追查历史卖方的融资约束与被动减仓原因，先查已有证据，避免继续围绕同一个份额因子换表达反复测试。", "",
        "来源、全部原值及公开日期分别保存在`source_index.json`、`期末实名持有及公开日期.json`、`八份季报匿名大户原列.json`；脚本为`research/historical_huijin_ownership_transmission_v1.py`。所有原报告与既有失败结果保持原样。", "",
        "| 原报告 | 相关页 |", "| --- | --- |"]
    for src in sources:
        pages = src.get("pages", [src.get("page")])
        text.append(f"| [{src['period']}]({src['url']}) | {', '.join(str(p) for p in pages)} |")
    (OUT / "历史发现_大户增持与价格响应的先后.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"状态": result["status"], "7月30日净增下界_亿份": float(minimum_increase/HUNDRED_MILLION),
                      "本季净增量下界占比": float(minimum_increase/g3["net_holding_change"]),
                      "固定时段价格回报": price_windows, "目标达到": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
