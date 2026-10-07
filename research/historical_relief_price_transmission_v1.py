"""历史缓释案例的净值、份额与固定成分归因；不生成交易信号或未来判断。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_relief_price_transmission_v1"
PARENT = ROOT / "reports/research/510300_historical_collateral_relief_cases_v1"
INPUT = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs"
SW = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1"
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
HIST_NAV = ROOT / "data/external_validation/510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2/snapshots/20260828T125705_0800/510300_nav_daily.parquet"
NEW_NAV = ROOT / "data/raw/fund/510300_nav_daily_raw.parquet"
SHARE = ROOT / "data/raw/flow/510300_etf_share_premium_level_full_v1.parquet"
WEIGHT = ROOT / "data/raw/constituents/000300_historical_weights.parquet"
STUDY = "510300_HISTORICAL_RELIEF_PRICE_TRANSMISSION_V1"
DELAY = 6
FOCUS = ["P18_OCT", "M24_CONTROL"]
SEGMENTS = [("PRE_ENTRY", "信息前至原入场前收盘", "anchor", "entry_previous"),
            ("FIRST_FIVE", "原入场前收盘至观察五日末", "entry_previous", "five_close"),
            ("LATER", "观察五日末至原退出前收盘", "five_close", "exit_previous")]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, np.generic):
        return clean(x.item())
    if x is pd.NA or x is pd.NaT or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return x


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本轮设定已保存，不覆盖。")
    parent = json.loads((PARENT / "protocol.json").read_text(encoding="utf-8"))
    events = [{k: r[k] for k in ["id", "date", "available_date", "label"]} for r in parent["events"]]
    events.append({"id": "M24_CONTROL", "date": "2024-02-05", "available_date": "2024-02-05", "label": "两融及质押缓释说明（既有对照）"})
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "research_mode": "HISTORICAL_ONLY",
        "question": "融资活动同向改善的2018年10月与2024年2月，净值、份额和固定成分贡献有何差异，等待观察后还剩多少收益？",
        "events": events, "focus": FOCUS, "sample_selection": "保留上一轮五节点及2024既有对照；重点两例按已知收益差异选取，只是发现性比较，不能算独立验证。",
        "prior_returns_already_seen": True, "before_new_decomposition_and_delay_returns": True,
        "independent_validation": False, "new_parameters_fitted": 0,
        "observation": "原入场日起5个完整交易日，包含原入场日；随后再留一个完整交易日。",
        "delay_sessions": DELAY, "exit_sessions": 20,
        "delay_meaning": "统一在原入场索引加6日开盘买入，原索引加20日开盘退出。这是固定日期机会成本算例，不以事后融资/净值/行业结果筛选交易，也不证明数据已在当时发布。",
        "price_decomposition": "各段同日收盘对收盘：(P1-P0+D)/P0=(NAV1-NAV0+D)/P0+[(P1-NAV1)-(P0-NAV0)]/P0。不是盘中折价套利或因果贡献。",
        "share_meaning": "基金份额数量及其变化，绝不等同现金净流入、全市场股票净买入或某机构成交。",
        "constituent_universe": "每例最早信息日前一交易日固定300成分，不用事后新成分替换。",
        "reference_weights": "锚日前最近月末权重，最多62个自然日；未认证当时发布版本，仅为固定篮子描述。没有权重则不补值。",
        "constituent_returns": "复用已分类的股东总回报，仅使用有效交易或官方停牌状态；窗口任何日缺失即该证券窗口缺失，不补零、不重标有效权重。",
        "industry": "按官方历史变更表的有效区间做事后归因，锁定锚日分类；记录更新时间晚于事件的数量。拒绝旧PIT表以过期行业替代未知现行行业的做法。2021代码表仅作名称展示，旧版缺失名称保留代码。",
        "costs": parent["costs"], "new_accounts": 0, "new_prospective_forecasts": 0,
        "goal_achieved": False, "orders_authorized": False,
        "existing_failures_preserved": ["510300_etf_discount_compensation_probe_v1", "510300_fund_share_publication_receipts_closure_v1", "510300_factor96_rapid_breadth_speed_v1"],
        "additional_source": {"url": "https://www.csrc.gov.cn/csrc/c100028/c7462111/content.shtml", "date": "2024-02-06", "role": "同日新增ETF增持信息的竞争解释；不是2月6日开盘前已知信息。"},
    })
    print("已固定六个既有节点、两个重点案例、三段价格分解及统一延迟日期。", flush=True)


def read_inputs():
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date)
    a, b = pd.read_parquet(HIST_NAV), pd.read_parquet(NEW_NAV)
    for frame in [a, b]:
        frame["date"] = pd.to_datetime(frame.date)
    nav = pd.concat([a[a.date.lt("2021-08-12")], b[b.date.ge("2021-08-12")]], ignore_index=True)
    assert not nav.date.duplicated().any()
    shares = pd.read_parquet(SHARE)
    shares["date"] = pd.to_datetime(shares.date)
    merged = market.merge(nav[["date", "unit_nav", "close", "source_primary", "source_secondary"]].rename(columns={"close": "nav_file_close"}), on="date", how="left", validate="one_to_one")
    merged = merged.merge(shares[["date", "fund_shares", "share_source", "close_premium_to_nav"]].rename(columns={"close_premium_to_nav": "old_merged_premium"}), on="date", how="left", validate="one_to_one")
    merged["premium"] = merged.close / merged.unit_nav - 1
    return merged


def decomposition(frame, start, end):
    a, b = frame.iloc[start], frame.iloc[end]
    assert end >= start
    window = frame.iloc[start:end+1]
    assert window[["close", "unit_nav", "fund_shares"]].notna().all().all()
    d = float(frame.iloc[start+1:end+1].dividend.sum())
    price = (b.close - a.close + d) / a.close
    nav = (b.unit_nav - a.unit_nav + d) / a.close
    residual = ((b.close-b.unit_nav)-(a.close-a.unit_nav)) / a.close
    error = float(price-nav-residual)
    assert abs(error) < 1e-12
    return {"start": a.date, "end": b.date, "sessions": end-start,
            "close_return": float(price), "nav_component": float(nav), "premium_component": float(residual),
            "nav_return": float((b.unit_nav-a.unit_nav+d)/a.unit_nav), "cash_dividend": d,
            "premium_start": float(a.premium), "premium_end": float(b.premium),
            "shares_start": float(a.fund_shares), "shares_end": float(b.fund_shares),
            "share_change": float(b.fund_shares-a.fund_shares), "share_change_ratio": float(b.fund_shares/a.fund_shares-1),
            "price_decomposition_error": error,
            "nav_file_close_max_absolute_difference": float((window.close-window.nav_file_close).abs().max()),
            "old_premium_max_difference": float((window.premium-window.old_merged_premium).abs().max())}


def classify_members(members, anchor, history, codebook):
    eligible = history[history.effective_date.le(anchor) & (history.out_date.isna() | history.out_date.gt(anchor))]
    assert not eligible.symbol.duplicated().any(), "有效区间重叠。"
    columns = ["symbol", "industry_code", "industry_l1_code", "effective_date", "out_date", "record_updated_at", "classification_standard"]
    result = members.merge(eligible[columns], on="symbol", how="left", validate="one_to_one")
    names = codebook.drop_duplicates("industry_l1_code").set_index("industry_l1_code").industry_l1_name.to_dict()
    result["industry_name"] = result.industry_l1_code.map(names)
    result["industry_name"] = result.industry_name.fillna("旧版代码未对应展示名称")
    result["industry_l1_code"] = result.industry_l1_code.fillna("UNKNOWN")
    result["industry_updated_after_anchor"] = result.record_updated_at.gt(anchor+pd.Timedelta(hours=15))
    return result


def constituent_analysis(specs, market):
    members = pd.read_parquet(INPUT / "membership.parquet")
    members["membership_date"] = pd.to_datetime(members.membership_date)
    history = pd.read_parquet(SW / "sw_industry_history_events_through_20260814.parquet")
    for column in ["effective_date", "out_date", "record_updated_at"]:
        history[column] = pd.to_datetime(history[column])
    codebook = pd.read_parquet(SW / "sw_2021_codebook.parquet")
    weights = pd.read_parquet(WEIGHT)
    weights["trade_date"] = pd.to_datetime(weights.trade_date)
    returns = pd.read_parquet(INPUT / "classified.parquet", columns=["date", "symbol", "constituent_return_state", "daily_total_shareholder_return", "return_is_usable"])
    returns["date"] = pd.to_datetime(returns.date)
    valid = returns.return_is_usable & returns.constituent_return_state.isin(["TRADED_VALID", "OFFICIAL_SUSPENSION"]) & np.isfinite(returns.daily_total_shareholder_return) & returns.daily_total_shareholder_return.gt(-1)
    returns.loc[~valid, "daily_total_shareholder_return"] = np.nan
    summaries, security_rows, group_rows, map_rows = [], [], [], []
    for spec in specs:
        if spec["id"] not in FOCUS:
            continue
        anchor = market.iloc[spec["indices"]["anchor"]].date
        universe = members[members.membership_date.eq(anchor)][["symbol"]].copy()
        assert len(universe) == 300 and universe.symbol.nunique() == 300
        universe = classify_members(universe, anchor, history, codebook)
        prior_dates = weights.loc[weights.trade_date.lt(anchor), "trade_date"]
        weight_date = prior_dates.max()
        assert pd.notna(weight_date) and (anchor-weight_date).days <= 62
        w = weights[weights.trade_date.eq(weight_date)][["con_code", "weight"]].rename(columns={"con_code": "symbol"})
        assert not w.symbol.duplicated().any() and abs(w.weight.sum()-100) < .02
        universe = universe.merge(w, on="symbol", how="left", validate="one_to_one")
        universe["weight"] = universe.weight / 100
        universe["event_id"] = spec["id"]
        universe["anchor"] = anchor
        map_rows.extend(universe.to_dict("records"))
        for key, label, left, right in SEGMENTS:
            start, end = spec["indices"][left], spec["indices"][right]
            dates = market.iloc[start+1:end+1].date
            selected = returns[returns.date.isin(dates) & returns.symbol.isin(universe.symbol)]
            assert not selected.duplicated(["date", "symbol"]).any()
            daily = selected.pivot(index="date", columns="symbol", values="daily_total_shareholder_return").reindex(index=dates, columns=universe.symbol)
            cumulative = (1+daily).prod(axis=0, skipna=False)-1
            detail = universe.merge(cumulative.rename("total_return"), left_on="symbol", right_index=True, how="left", validate="one_to_one")
            detail["contribution"] = detail.weight * detail.total_return
            detail["segment"] = key
            usable = detail.total_return.notna()
            positive = usable & detail.total_return.gt(0)
            contribution = detail.contribution.sum(min_count=1)
            window = decomposition(market, start, end)
            row = {"event_id": spec["id"], "segment": key, "label": label, "start": market.iloc[start].date, "end": market.iloc[end].date,
                   "weight_date": weight_date, "weight_sum": float(detail.weight.sum(min_count=1)), "usable_weight": float(detail.loc[usable, "weight"].sum(min_count=1)),
                   "members": 300, "usable_members": int(usable.sum()), "positive_members": int(positive.sum()),
                   "positive_fraction_of_usable": float(positive.sum()/usable.sum()) if usable.any() else None,
                   "median_security_return": float(detail.total_return.median()), "equal_weight_return": float(detail.total_return.mean()),
                   "fixed_weight_return": float(contribution), "etf_nav_return": window["nav_return"],
                   "etf_nav_minus_reference_basket": float(window["nav_return"]-contribution),
                   "industry_updated_after_anchor": int(detail.industry_updated_after_anchor.sum()),
                   "industry_unknown_members": int(detail.industry_l1_code.eq("UNKNOWN").sum()),
                   "weighted_method": "月末固定权重参考篮子，无缺失重标；不是官方指数或基金实际持仓归因。"}
            summaries.append(row)
            security_rows.extend(detail.to_dict("records"))
            for (code, name), group in detail.groupby(["industry_l1_code", "industry_name"], dropna=False):
                group_rows.append({"event_id": spec["id"], "segment": key, "industry_code": code, "industry_name": name,
                                   "members": len(group), "usable_members": int(group.total_return.notna().sum()),
                                   "weight": float(group.weight.sum(min_count=1)), "contribution": float(group.contribution.sum(min_count=1)),
                                   "median_return": float(group.total_return.median())})
    save("固定成分归因汇总.json", summaries)
    save("固定行业贡献.json", group_rows)
    save("当日有效行业与更新时间.json", map_rows)
    pd.DataFrame(security_rows).to_parquet(OUT / "固定成分分段回报.parquet", index=False)
    return summaries, group_rows


def chart(events, market, groups):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for col, event_id in enumerate(FOCUS):
        row = next(r for r in events if r["id"] == event_id)
        i, j = row["indices"]["anchor"], row["indices"]["exit_previous"]
        frame = market.iloc[i:j+1].copy()
        d = frame.dividend.copy()
        d.iloc[0] = 0
        x = np.arange(len(frame))
        axes[0, col].plot(x, ((frame.close+d.cumsum())/frame.close.iloc[0]-1)*100, label="ETF收盘价格及股息", color="#1a6b77", lw=2)
        axes[0, col].plot(x, ((frame.unit_nav+d.cumsum())/frame.unit_nav.iloc[0]-1)*100, label="单位净值及股息", color="#b27835", ls="--", lw=1.7)
        axes[0, col].axvline(row["indices"]["five_close"]-i, color="#999999", ls=":", label="五日观察结束")
        axes[0, col].set(title=row["date"]+" 历史价格与净值", ylabel="相对信息前收盘（%）", xlabel="实际交易日序号；起点为信息前收盘")
        axes[0, col].legend(fontsize=9)
        axes[0, col].text(.02, .97, f"观察五日期间份额变化 {row['segments'][1]['share_change']/1e8:+.2f} 亿份\n统一延迟算例净收益 {row['delayed_trade']['net_return']:+.2%}", transform=axes[0, col].transAxes, va="top", fontsize=9)
        g = pd.DataFrame([r for r in groups if r["event_id"] == event_id and r["segment"] == "FIRST_FIVE"])
        g = g.reindex(g.contribution.abs().sort_values(ascending=False).index).head(8).sort_values("contribution")
        axes[1, col].barh(g.industry_name+"（"+g.industry_code+"）", g.contribution*100, color=np.where(g.contribution.ge(0), "#1a6b77", "#b26049"))
        axes[1, col].set(title="五日观察期间：绝对贡献最大的八个行业", xlabel="固定月末权重参考贡献（百分点）")
        axes[1, col].axvline(0, color="#777777", lw=.7)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15, axis="x")
    fig.suptitle("融资回暖之后：价格来自净值，差异需要看持仓与需求渠道", fontsize=16)
    fig.text(.02, .015, "两例按已见收益差异选取，只作历史发现。份额不等于现金流；行业使用事后有效分类，参考权重未认证首发版本。", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(OUT / "两段历史的净值份额与行业贡献.png", dpi=170)
    plt.close(fig)


def report(events, summaries, groups):
    date = lambda value: pd.Timestamp(value).strftime("%Y-%m-%d")
    by_id = {r["id"]: r for r in events}
    a, b = by_id["P18_OCT"], by_id["M24_CONTROL"]
    def segment(row):
        return next(s for s in row["segments"] if s["key"] == "FIRST_FIVE")
    def summary(row):
        return next(s for s in summaries if s["event_id"] == row["id"] and s["segment"] == "FIRST_FIVE")
    def contribution(row, code):
        return next(s["contribution"] for s in groups if s["event_id"] == row["id"] and s["segment"] == "FIRST_FIVE" and s["industry_code"] == code)
    first_a, first_b, sum_a, sum_b = segment(a), segment(b), summary(a), summary(b)
    focus_table = "\n".join([
        f"| 观察区间（收盘至收盘） | {date(first_a['start'])}—{date(first_a['end'])} | {date(first_b['start'])}—{date(first_b['end'])} |",
        f"| ETF收盘价格变化 | {first_a['close_return']:+.2%} | {first_b['close_return']:+.2%} |",
        f"| 净值变化贡献 | {first_a['nav_component']*100:+.3f}个百分点 | {first_b['nav_component']*100:+.3f}个百分点 |",
        f"| 价格与净值差额的贡献 | {first_a['premium_component']*100:+.3f}个百分点 | {first_b['premium_component']*100:+.3f}个百分点 |",
        f"| ETF份额变化 | {first_a['share_change']/1e8:+.3f}亿份 / {first_a['share_change_ratio']:+.2%} | {first_b['share_change']/1e8:+.3f}亿份 / {first_b['share_change_ratio']:+.2%} |",
        f"| 固定300只中上涨数量 | {sum_a['positive_members']}只 | {sum_b['positive_members']}只 |",
        f"| 个股回报中位数 | {sum_a['median_security_return']:+.2%} | {sum_b['median_security_return']:+.2%} |",
        f"| 非银金融参考贡献 | {contribution(a, '49')*100:+.3f}个百分点 | {contribution(b, '49')*100:+.3f}个百分点 |",
        f"| 银行参考贡献 | {contribution(a, '48')*100:+.3f}个百分点 | {contribution(b, '48')*100:+.3f}个百分点 |",
        f"| 食品饮料参考贡献 | {contribution(a, '34')*100:+.3f}个百分点 | {contribution(b, '34')*100:+.3f}个百分点 |",
    ])
    returns_table = "\n".join(f"| {r['date']} {r['label']} | {date(r['original_trade']['entry_date'])} | {r['original_trade']['net_return']:+.2%} | {date(r['delayed_trade']['entry_date'])} | {date(r['delayed_trade']['exit_date'])} | {r['delayed_trade']['net_return']:+.2%} |" for r in events)
    all_nav = "\n".join(f"| {r['date']} | {segment(r)['close_return']:+.2%} | {segment(r)['nav_component']*100:+.3f} | {segment(r)['premium_component']*100:+.3f} | {segment(r)['share_change_ratio']:+.2%} |" for r in events)
    timing = "\n".join(f"| {r['date']} | {r['timing_bridge']['entry_gap']:+.2%} | {r['timing_bridge']['entry_open_to_five_close']:+.2%} | {r['timing_bridge']['next_open_gap']:+.2%} | {r['timing_bridge']['net5']:+.2%} |" for r in [a,b])
    text = f"""# 历史发现：缓释以后，谁在买、哪些权重仍在跌

本轮只研究已经发生的六个节点。2018年10月与2024年2月都出现融资买入恢复、净偿还收窄，但篮子内部、同步需求事件和入场价格不同。价格与净值分解能定位差异，尚未识别单一政策的因果效应，也未产生完整账户夏普1.2的结果。

2018年10月的金融、地产贡献为正，食品饮料却明显拖累；2024年2月上涨扩散到固定300只中的297只。两个时期都增加ETF份额，且2018年的相对增幅更大，所以不能用“份额增加”单独解释收益差异。2015年7月的份额大增与价格大跌同时出现，构成同一观察集内的反例。

| 同一口径的五日观察 | 2018年10月 | 2024年2月 |
|---|---:|---:|
{focus_table}

净值贡献与价差贡献采用相同的期初ETF价格作分母，二者精确相加等于收盘价格回报。它们是价格恒等式，不能解释为某项政策的因果贡献。两段价格的主要变化来自底层净值，2024年价格相对净值的变化反而抵消约0.486个百分点涨幅，不支持把该段收益说成溢价扩张。

行业贡献用锚日前月末的固定权重计算，2018年为2018-09-28，2024年为2024-01-31；不是基金实际逐日持仓，也不是官方指数归因。两例五日均有完整300只有效回报。2018年后续阶段有4只回报缺失，未补零、未重新放大剩余权重，有效权重约99.183%。

2018年五日最大负贡献个股包括600519、600887、002304及000858。这里已经回答“跌在哪些权重”，但仅凭价格还不能回答“盈利预期为何变化”；后续查公司披露时必须按公布日期解释，不能拿五日观察结束之后的财报回填此前下跌原因。

2024年2月6日还出现独立的需求渠道信息：证监会就汇金增持ETF公告表态支持，并表示推动机构入市。这是两融债务缓释以外的同期竞争解释。[证监会原文](https://www.csrc.gov.cn/csrc/c100028/c7462111/content.shtml)只给出当日日期，本轮不将该文视为2月6日开盘前已知信息，也不把全部510300份额增加认定为汇金购买。

ETF一级市场申购可以交付证券篮子、现金替代及差额，因此份额数量变化不等于新增现金买股；二级市场买入也不必然增加基金份额。[上交所ETF问答](https://www.sse.com.cn/assortment/fund/etf/question/)解释了这些交易与申赎机制。份额上升表示基金份额数量扩大，但不能单独识别买方身份、资金是否新入市或价格冲击方向。

| 既有节点 | 五日收盘价格变化 | 净值贡献（百分点） | 价差贡献（百分点） | 五日份额变化 |
|---|---:|---:|---:|---:|
{all_nav}

2015年7月价差变化更明显；但本轮未取得同刻可执行股票篮子报价，收盘净值不能当作盘中可成交价格。这段分解不能升级为折价套利策略。

为什么2018年观察期收盘上涨，原来的五日算例仍亏损？观察起点是10月19日收盘，算例买入是10月22日开盘，买入前跳空已经消耗一部分上涨；第五日后按10月29日开盘卖出，还经历一次隔夜变化。下面将这些时点逐一接起来，避免混用收盘归因和开盘成交收益。

| 节点 | 原入场开盘相对前收盘 | 入场开盘至第五日收盘 | 下一开盘相对第五日收盘 | 原五日费用后参考收益 |
|---|---:|---:|---:|---:|
{timing}

为观察“看清五日之后还剩多少”，本轮在计算前固定了一个日历算例：观察原入场日起五个完整交易日，再留一个完整交易日，在第七个交易日开盘买入，仍于原入场索引加20日的开盘退出。所有六个节点统一执行，不根据融资、份额或行业表现挑选。它只衡量等待的机会成本，不是一套已经可用的信号。

| 信息节点 | 原买入日 | 原20日净收益 | 统一延迟买入日 | 固定原退出日 | 延迟后净收益 |
|---|---|---:|---|---|---:|
{returns_table}

等待不是统一改善：2018年10月延迟算例收益较高，是因为同一日历规则恰好跨过后续下跌后买入；2024年2月等待则错过部分上涨，参考收益由12.20%降至3.68%。2015年6月延迟后仍亏13.31%。不能据此事后选择“2018年晚买、2024年早买”，也不能把六例拼成年化夏普。每例仅按10万元参考预算、0.04%单边佣金且最低5元、0.1%单边滑点、0.001元最小价位和100份整数手测算；20万元完整账户、空仓期、风险预算与重叠事件尚未模拟。

实用上，这组历史把三个层次分开了：债权条款改变影响被迫卖出约束；投资者实际购买渠道影响承接；权重公司的经营与估值变化仍能抵消政策方向。这是下一步解释个股盈利披露与买方行为的依据，而不是“看到纾困或份额上升就买”的结论。

本轮复用了既有价格、净值、份额和成分数据，只做必要核对。原六节点20日净收益得到逐项复现，价格拆分与行业加总恒等式通过。原始份额、NAV的历史首次发布时钟没有因此补齐。行业按官方变更表的生效区间事后映射，2018年有177只、2024年有158只对应记录更新晚于锚日，不充当当时交易输入；2021代码表只用于名称展示，跨版本口径不完全相同。已有折价、份额发布和宽度策略失败结果保持不变。

本轮设置见[protocol.json](protocol.json)，逐事件结果见[六节点分段价格与延迟收益.json](六节点分段价格与延迟收益.json)，行业明细见[固定行业贡献.json](固定行业贡献.json)，全部可复算数据路径见[source_index.json](source_index.json)。没有新增参数拟合、完整策略账户、未来判断或交易授权。目标状态仍为未达到。

![净值、份额与行业贡献](两段历史的净值份额与行业贡献.png)
"""
    (OUT / "历史发现_净值份额与权重传导.md").write_text(text, encoding="utf-8")


def verify(events, summaries, groups):
    old = json.loads((PARENT / "五个历史节点的收益与融资分解.json").read_text(encoding="utf-8"))
    control = json.loads((PARENT / "2024年既有对照.json").read_text(encoding="utf-8"))["event"]
    old.append(dict(control, id="M24_CONTROL"))
    mapping = {r["id"]: r for r in old}
    return_errors, group_errors, price_errors = [], [], []
    for event in events:
        original = mapping[event["id"]]
        return_errors.extend([abs(event["original_trade"]["net_return"]-original["holding_20"]["net_return"]),
                              abs(event["timing_bridge"]["net5"]-original["holding_5"]["net_return"])])
        assert pd.Timestamp(event["original_trade"]["exit_date"]) == pd.Timestamp(original["holding_20"]["exit_date"])
        b = event["timing_bridge"]
        assert abs((1+b["entry_open_to_five_close"])*(1+b["next_open_gap"])-1-b["gross5"]) < 1e-12
        assert event["delayed_trade"]["holding_sessions"] == 14
        assert event["delayed_trade"]["shares"] % 100 == 0
        price_errors.extend(abs(s["price_decomposition_error"]) for s in event["segments"])
    for row in summaries:
        group_sum = sum(r["contribution"] for r in groups if r["event_id"] == row["event_id"] and r["segment"] == row["segment"])
        group_errors.append(abs(group_sum-row["fixed_weight_return"]))
    assert max(return_errors) < 1e-12 and max(group_errors) < 1e-12
    save("calculation_checks.json", {"status": "PASS_ACCOUNTING_AND_PRIOR_RETURN_RECONCILIATION", "checked_at": now(),
                                     "prior_returns_compared": len(return_errors), "max_prior_return_error": max(return_errors),
                                     "max_price_decomposition_error": max(price_errors), "max_group_addition_error": max(group_errors),
                                     "scope": "仅检查运算和既有结果一致，不构成因果识别、交易有效性或独立验证。"})


def source_index():
    paths = [MARKET, HIST_NAV, NEW_NAV, SHARE, WEIGHT, INPUT / "classified.parquet", INPUT / "membership.parquet",
             SW / "sw_industry_history_events_through_20260814.parquet", SW / "sw_2021_codebook.parquet",
             ROOT / "data/reference/510300_dividends.csv", PARENT / "五个历史节点的收益与融资分解.json", PARENT / "2024年既有对照.json"]
    sources = [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mode": "EXISTING_LOCAL_INPUT"} for p in paths]
    save("source_index.json", {"recorded_at": now(), "local_inputs": sources,
                               "protocol_sha256": hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
                               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               "official_web_source": {"url": "https://www.csrc.gov.cn/csrc/c100028/c7462111/content.shtml", "publication_date": "2024-02-06",
                                                       "verified_with": "WEB_TOOL_DIRECT_PAGE", "retrieval_date": "2026-09-30", "local_raw_archive": "NOT_ACQUIRED_HTTP403",
                                                       "claim": "证监会同日回应汇金增持ETF公告并支持机构入市；未公布510300逐日购买量。",
                                                       "entry_information": False},
                               "etf_mechanism_source": "https://www.sse.com.cn/assortment/fund/etf/question/",
                               "share_units": "份；2018来源为SSE历史规模，2024为Tushare兼容服务的fund_share整理数据，历史首发时钟未补齐。",
                               "nav_sources": ["eastmoney.f10.lsjz", "sina.CaihuiFundInfoService.getNav"],
                               "not_a_fully_independent_official_data_chain": True})


def analyze():
    cfg = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert cfg["delay_sessions"] == DELAY and cfg["exit_sessions"] == 20 and cfg["focus"] == FOCUS
    market = read_inputs()
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividends = dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    events, paths = [], []
    for event in cfg["events"]:
        entry_i = int(market.date.searchsorted(pd.Timestamp(event["available_date"]), side="right"))
        anchor_i = int(market.date.searchsorted(pd.Timestamp(event["date"]), side="left"))-1
        indices = {"anchor": anchor_i, "entry_previous": entry_i-1, "entry": entry_i, "five_close": entry_i+4, "delayed_entry": entry_i+DELAY, "exit_previous": entry_i+19, "exit": entry_i+20}
        row = dict(event, indices=indices, dates={key: market.iloc[value].date for key, value in indices.items()})
        row["segments"] = [dict(key=key, label=label, **decomposition(market, indices[left], indices[right])) for key, label, left, right in SEGMENTS]
        for key, start in [("original_trade", entry_i), ("delayed_trade", entry_i+DELAY)]:
            a, b = market.iloc[start], market.iloc[entry_i+20]
            cash = float(dividends.loc[dividends.record_date.ge(a.date) & dividends.record_date.lt(b.date), "cash_dividend_per_share"].sum())
            trade = round_trip(a.open, b.open, cash, cfg["costs"])
            row[key] = {"entry_date": a.date, "exit_date": b.date, "holding_sessions": entry_i+20-start,
                        "cash_dividend_per_share": cash, "entry_open": float(a.open), "exit_open": float(b.open), **trade}
        previous, entry, five, exit5 = (market.iloc[x] for x in [entry_i-1, entry_i, entry_i+4, entry_i+5])
        cash5 = float(dividends.loc[dividends.record_date.ge(entry.date) & dividends.record_date.lt(exit5.date), "cash_dividend_per_share"].sum())
        assert cash5 == 0 and market.iloc[entry_i:entry_i+6].dividend.sum() == 0, "桥接窗口有股息，需要显式处理。"
        row["timing_bridge"] = {"entry_gap": float(entry.open/previous.close-1),
                                "entry_open_to_five_close": float(five.close/entry.open-1),
                                "next_open_gap": float(exit5.open/five.close-1),
                                "gross5": float(exit5.open/entry.open-1),
                                "net5": round_trip(entry.open, exit5.open, cash5, cfg["costs"])["net_return"],
                                "scope": "复核既有五日收益的日期差异，不新增期限选择。"}
        events.append(row)
        path = market.iloc[anchor_i:entry_i+21][["date", "open", "close", "dividend", "unit_nav", "premium", "fund_shares", "share_source", "source_primary", "source_secondary"]].copy()
        path["event_id"] = event["id"]
        paths.append(path)
    summaries, groups = constituent_analysis(events, market)
    save("六节点分段价格与延迟收益.json", events)
    pd.concat(paths, ignore_index=True).to_parquet(OUT / "六节点价格净值份额日线.parquet", index=False)
    result = {"study_id": STUDY, "status": "COMPLETED_HISTORICAL_PRICE_NAV_SHARE_AND_CONSTITUENT_COMPARISON", "completed_at": now(),
              "goal_turn_classification": "PROGRESS_HISTORICAL_RELIEF_TRANSMISSION_AND_DELAYED_REMAINING_RETURN",
              "research_mode": "HISTORICAL_ONLY", "event_count": len(events), "focus": FOCUS,
              "independent_validation": False, "new_parameters_fitted": 0, "new_accounts": 0, "new_prospective_forecasts": 0,
              "goal_achieved": False, "strategy_status": "NOT_CREATED_DESCRIPTIVE_ACCOUNTING_AND_DATE_BASED_OPPORTUNITY_COST",
              "source_limits": ["历史NAV及份额首次发布时钟仍未建立", "固定月末权重非实际每日持仓", "行业按事后有效区间解释", "同日多政策和多需求冲击无法因果分离"],
              "returns": [{"id": r["id"], "original_net20": r["original_trade"]["net_return"], "delayed_entry": r["delayed_trade"]["entry_date"], "same_exit_delayed_net": r["delayed_trade"]["net_return"]} for r in events],
              "focus_constituent_summary": summaries,
              "next_historical_question": "将固定行业贡献及ETF购买渠道追溯到当时已公开的盈利与实际持有人变化，区分宏观/盈利冲击和库存承接；不由本轮结果生成筛选条件。"}
    save("result.json", result)
    verify(events, summaries, groups)
    source_index()
    chart(events, market, groups)
    report(events, summaries, groups)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="历史净值、份额和行业传导比较")
    parser.add_argument("mode", choices=["prepare", "analyze"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        analyze()


if __name__ == "__main__":
    main()
