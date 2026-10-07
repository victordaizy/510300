"""发布两个固定病例的公司传导解释、完整行业图和保存结果复算回执。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_constituent_external_bridge_v16"
PREV = ROOT / "reports/research/510300_external_discount_clock_v15"


def load(name):
    return pd.read_csv(OUT / "results" / name)


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_saved_outputs(summary, stock, sector):
    """从保存的成分原字段重新递推股数和现金，再核对三个聚合层。"""
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha256((OUT / "protocol.json").read_bytes()).hexdigest() == frozen["protocol_sha256"]
    for row in frozen["inputs"]:
        assert sha256((OUT / "inputs" / row["name"]).read_bytes()).hexdigest() == row["sha256"]
    ledger = pd.read_parquet(OUT / "results/12000个成分日_持股现金账本.parquet")
    assert len(ledger) == 12000 and len(stock) == 600
    errors, action_rows = [], []
    for case in summary.index:
        raw = load(case + "_实际采用的6000行成分原字段.csv").sort_values(["symbol", "date"])
        saved_stock = stock[stock.stat_month == case].set_index("symbol")
        saved_days = ledger[ledger.stat_month == case].set_index(["symbol", "date"])
        days = sorted(raw.date.unique())
        weighted_changes = np.zeros((20, 300))
        weights = pd.read_parquet(OUT / "inputs/weights.parquet")
        weights = weights[weights.trade_date.astype(str).str[:10] == summary.loc[case, "weight_date"]].set_index("con_code").weight
        assert len(weights) == 300
        weights = weights / weights.sum()
        for j, (symbol, a) in enumerate(raw.groupby("symbol", sort=True)):
            a = a.sort_values("date")
            assert len(a) == 20
            unit, cash, previous_wealth = 1 / a.iloc[0].previous_unadjusted_close, 0.0, 1.0
            assert abs(saved_stock.loc[symbol, "weight"] - weights[symbol]) < 1e-11
            for t, row in enumerate(a.itertuples()):
                marked = row.unadjusted_close
                if not np.isfinite(marked):
                    assert row.official_suspension
                    marked = row.previous_unadjusted_close
                cash += unit * (row.cash_distribution_per_pre_event_share - row.subscription_cash_outflow_per_pre_event_share)
                unit *= row.post_to_pre_share_ratio
                wealth = unit * marked + cash
                saved = saved_days.loc[(symbol, row.date)]
                errors.extend([abs(saved.stock_wealth - wealth), abs(saved.cash_per_initial_stock_yuan - cash),
                               abs(saved.units_per_initial_stock_yuan - unit)])
                weighted_changes[t, j] = weights[symbol] * (wealth - previous_wealth)
                errors.append(abs(saved.daily_contribution_to_initial_basket - weighted_changes[t, j]))
                previous_wealth = wealth
                if row.cash_distribution_per_pre_event_share != 0 or row.post_to_pre_share_ratio != 1:
                    action_rows.append({"stat_month": case, "symbol": symbol, "date": row.date,
                                        "cash_per_pre_share": row.cash_distribution_per_pre_event_share,
                                        "post_to_pre_ratio": row.post_to_pre_share_ratio})
            errors.append(abs(saved_stock.loc[symbol, "stock_return20_cc"] - (wealth - 1)))
            first_close = a.iloc[0].unadjusted_close
            if not np.isfinite(first_close):
                assert a.iloc[0].official_suspension
                first_close = a.iloc[0].previous_unadjusted_close
            unit19, cash19 = 1 / first_close, 0.0
            for row in a.iloc[1:].itertuples():
                cash19 += unit19 * (row.cash_distribution_per_pre_event_share - row.subscription_cash_outflow_per_pre_event_share)
                unit19 *= row.post_to_pre_share_ratio
            close_exit = a.iloc[-1].unadjusted_close
            if not np.isfinite(close_exit):
                assert a.iloc[-1].official_suspension
                close_exit = a.iloc[-1].previous_unadjusted_close
            errors.append(abs(saved_stock.loc[symbol, "stock_return19_entryclose"] - (unit19 * close_exit + cash19 - 1)))
        contributions = weighted_changes.sum(axis=1)
        previous_value = 1 + np.r_[0, np.cumsum(contributions[:-1])]
        returns = contributions / previous_value
        normalized = weighted_changes / previous_value[:, None]
        variance = float(np.var(returns, ddof=1) * 252)
        downside2 = float(np.minimum(returns, 0).dot(np.minimum(returns, 0)) * 252 / 20)
        vc = np.array([np.cov(normalized[:, j], returns, ddof=1)[0, 1] * 252 for j in range(300)])
        dc = np.sum(normalized * (returns * (returns < 0))[:, None], axis=0) * 252 / 20
        ordered = saved_stock.sort_index()
        errors.extend(np.abs(vc - ordered.variance_contribution.to_numpy()))
        errors.extend(np.abs(dc - ordered.downside_squared_contribution.to_numpy()))
        assert abs(contributions.sum() * 100 - summary.loc[case, "basket20_cc_return_percent"]) < 1e-8
        assert abs(np.sqrt(variance) * 100 - summary.loc[case, "basket_rv20_percent"]) < 1e-8
        assert abs(np.sqrt(downside2) * 100 - summary.loc[case, "basket_downside20_percent"]) < 1e-8
        by_industry = ordered.groupby("industry_reference")[["weight", "contribution20_pp", "variance_contribution", "downside_squared_contribution"]].sum()
        existing = sector[sector.stat_month == case].set_index("industry_reference").reindex(by_industry.index)
        assert np.allclose(by_industry.to_numpy(), existing[by_industry.columns].to_numpy(), atol=1e-9, rtol=1e-8)
        assert int((ordered.stock_return20_cc > 0).sum()) == summary.loc[case, "positive_stock_count"]
        segmented = load("全部成分_消息分段贡献.csv")
        summed = segmented[segmented.stat_month == case].groupby("symbol").contribution_pp.sum().reindex(ordered.index)
        assert np.allclose(summed, ordered.contribution20_pp, atol=1e-9, rtol=1e-8)
        q = pd.read_csv(OUT / "inputs/market.csv").set_index("date")
        orig = summary.loc[case]
        # 收盘观测、延迟至入场收盘、原E0开盘三种口径分别核对。
        for begin, denom, div_days, col in [
            (orig.origin_date, q.loc[orig.origin_date, "close"], days, "etf20_cc_return_percent"),
            (orig.entry_date, q.loc[orig.entry_date, "close"], days[1:], "etf19_entryclose_return_percent"),
            (orig.entry_date, q.loc[orig.entry_date, "open"], days[1:], "etf_original_E0_percent")]:
            actual = (q.loc[orig.exit_date, "close"] + q.loc[div_days, "dividend"].sum()) / denom - 1
            assert abs(actual * 100 - orig[col]) < 1e-8, (case, begin, col, actual, orig[col])
    assert np.isfinite(errors).all() and max(errors) < 1e-8, max(errors)
    assert any(x["symbol"] == "002648.SZ" and x["post_to_pre_ratio"] == 1.4 for x in action_rows)
    facts = load("十份中报_30项财务成对原数.csv")
    for row in facts.itertuples():
        pages = json.loads((OUT / "sources" / row.document).with_suffix(".pages.json").read_text(encoding="utf-8"))
        text = "".join(pages[row.pdf_page - 1]["text"].split())
        assert row.current_raw_token in text and row.prior_raw_token in text
        assert row.published_date < summary.loc[row.stat_month, "origin_date"]
        if np.isfinite(row.reported_yoy_percent):
            exact = (row.current_value / row.prior_value - 1) * 100
            assert abs(exact - row.recomputed_yoy_percent) < 1e-6
    save_json("results/分红送转逐笔复算记录.json", action_rows)
    return {"saved_stock_daily_rows_recomputed": len(ledger), "stock_case_rows_recomputed": len(stock),
            "maximum_account_and_risk_error": float(max(errors)), "corporate_action_rows": len(action_rows),
            "financial_pairs_checked_against_saved_pages": len(facts), "all_constituents_present": True,
            "etf_three_return_clocks_checked": True, "new_external_independent_validation": False}


def make_chart(summary, sector):
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(),
                         "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False})
    fig = plt.figure(figsize=(14, 14.4), facecolor="#f8f7f2")
    fig.text(.06, .968, "相近的总波动，不同的下行结构", fontsize=23, color="#172b3b", weight="bold")
    fig.text(.06, .94, "固定两个8月货币公告病例 · 全部300只成分 · 原持股现金账本", fontsize=12, color="#4d5d66")
    for i, case in enumerate(["2020-08", "2022-08"]):
        x = .06 if i == 0 else .56
        row = summary.loc[case]
        fig.text(x, .894, case + " 数据公布后", fontsize=17, color="#172b3b", weight="bold")
        fig.text(x, .867, f"参考篮子收益 {row.basket20_cc_return_percent:+.2f}%    上涨 {int(row.positive_stock_count)}/300", fontsize=14, color="#172b3b")
        fig.text(x, .842, f"总波动 {row.basket_rv20_percent:.2f}%    下行波动 {row.basket_downside20_percent:.2f}%", fontsize=13, color="#385e74")
        fig.text(x, .819, f"起点 {row.origin_date} 收盘 → {row.exit_date} 收盘", fontsize=10, color="#65727a")
        ax = fig.add_axes([x + .06, .108, .32, .676], facecolor="#f8f7f2")
        a = sector[sector.stat_month == case].sort_values("contribution20_pp", ascending=False)
        y = np.arange(len(a))
        vals = a.contribution20_pp.to_numpy()
        ax.barh(y, vals, color=["#b65a40" if v > 0 else "#386981" for v in vals], height=.72)
        ax.set_yticks(y, a.industry_reference, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(-1.05, 1.1)
        ax.set_xticks([-.8, -.4, 0, .4, .8])
        ax.axvline(0, color="#a1aaac", linewidth=.8)
        ax.xaxis.grid(True, color="#e0e2dc", linewidth=.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=7)
        for pos, val in zip(y, vals):
            ax.text(val + (.025 if val >= 0 else -.025), pos, f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=9, color="#243c4a")
        ax.set_xlabel("行业收益贡献（占起点本金的百分点）", fontsize=11, labelpad=12)
    fig.text(.06, .051, "图中为事后20日结果；参考权重不是510300真实持仓。分红留现金，送转调整股数。", fontsize=10, color="#53626b")
    fig.text(.06, .032, "行业使用各自起点附近的参考分类，保留未知；两年不强行合并标签。总波动与下行波动均为年化尺度。", fontsize=10, color="#53626b")
    fig.savefig(OUT / "figures/全部行业贡献与下行结构.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def report(summary, sector, company, checks):
    macro = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    external = pd.read_csv(PREV / "results/两个病例_外部条件与原收益对照.csv").set_index("stat_month")
    receipts = {r["name"]: r for r in json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8")) if r["status"].startswith("SAVED")}
    facts = load("十份中报_30项财务成对原数.csv")
    scalars = load("公司传导_37项原始数值.csv")
    qualitative = load("公司传导_14项原文定位与解释.csv")
    def cite(label, doc, pages):
        return f"[{label}]({receipts[doc]['url']})（PDF第{pages}页）"
    def link(label, path):
        return f"[{label}](<{path.as_posix()}>)"
    two = ["2020-08", "2022-08"]
    context_rows = []
    for label, getter, fmt in [
        ("M1／M2同比", lambda c: (macro.loc[c, "m1_yoy_pp"], macro.loc[c, "m2_yoy_pp"]), lambda v: f"{v[0]:.1f}%／{v[1]:.1f}%"),
        ("剪刀差三个月变化", lambda c: macro.loc[c, "delta3_spread_pp"], lambda v: f"{v:+.1f}个百分点"),
        ("当月较上月剪刀差变化", lambda c: macro.loc[c, "d1_spread_ordinary_pp"], lambda v: f"{v:+.1f}个百分点"),
        ("前三个月当期相对余额分量C", lambda c: macro.loc[c, "current_component"], lambda v: f"{v:+.3f}对数百分点"),
        ("企业累计新增贷款同比", lambda c: macro.loc[c, "corporate_total_yoy_percent"], lambda v: f"{v:+.2f}%"),
        ("企业中长期累计新增贷款同比多增", lambda c: macro.loc[c, "loan_corporate_long_ytd_yoy_change_yi"], lambda v: f"{v:+,.0f}亿元"),
        ("居民中长期累计新增贷款同比多增", lambda c: macro.loc[c, "loan_household_long_ytd_yoy_change_yi"], lambda v: f"{v:+,.0f}亿元"),
        ("制造业新订单", lambda c: macro.loc[c, "orders_first_release_value"], lambda v: f"{v:.1f}"),
        ("此前20／60日510300收益", lambda c: (macro.loc[c, "pre_return20_pp"], macro.loc[c, "pre_return60_pp"]), lambda v: f"{v[0]:+.2f}%／{v[1]:+.2f}%"),
        ("起点510300总／下行波动", lambda c: (macro.loc[c, "rv20"]*100, macro.loc[c, "downside20"]*100), lambda v: f"{v[0]:.2f}%／{v[1]:.2f}%"),
        ("起点可见美债2年／10年利率", lambda c: (external.loc[c, "origin_ust2_percent"], external.loc[c, "origin_ust10_percent"]), lambda v: f"{v[0]:.2f}%／{v[1]:.2f}%"),
        ("美债2年／10年前20条记录变化", lambda c: (external.loc[c, "origin_ust2_delta20_bp"], external.loc[c, "origin_ust10_delta20_bp"]), lambda v: f"{v[0]:+.0f}／{v[1]:+.0f}基点"),
    ]:
        context_rows.append([label] + [fmt(getter(c)) for c in two])
    context = md_table(["起点已知背景", "2020年8月数据", "2022年8月数据"], context_rows)
    outcomes = md_table(["后续观察结果", "2020病例", "2022病例"], [
        [label] + [fmt(summary.loc[c, field]) for c in two] for label, field, fmt in [
            ("原E0：次日开盘起20日510300毛收益", "etf_original_E0_percent", lambda v: f"{v:+.2f}%"),
            ("起点收盘起510300同口径收益", "etf20_cc_return_percent", lambda v: f"{v:+.2f}%"),
            ("起点收盘起参考篮子收益", "basket20_cc_return_percent", lambda v: f"{v:+.2f}%"),
            ("参考篮子－510300同口径", "basket_minus_etf20_pp", lambda v: f"{v:+.3f}个百分点"),
            ("再到原入场收盘起19日参考篮子收益", "basket19_entryclose_return_percent", lambda v: f"{v:+.2f}%"),
            ("上涨公司数", "positive_stock_count", lambda v: f"{int(v)}／300"),
            ("上涨公司起点权重合计", "positive_stock_initial_weight_percent", lambda v: f"{v:.2f}%"),
            ("参考篮子事后总波动", "basket_rv20_percent", lambda v: f"{v:.2f}%"),
            ("参考篮子事后下行波动", "basket_downside20_percent", lambda v: f"{v:.2f}%"),
            ("起点前五名权重合计", "top5_initial_weight_percent", lambda v: f"{v:.2f}%"),
            ("起点前五名收益贡献", "top5_contribution_pp", lambda v: f"{v:+.3f}个百分点"),
            ("起点前五名下行平方贡献份额", "top5_downside_squared_share_percent", lambda v: f"{v:.2f}%"),
        ]])
    company_rows = [[r.stat_month, int(r.reference_rank), r.name, f"{r.weight*100:.2f}%", f"{r.revenue_yoy_percent:+.2f}%", f"{r.net_profit_yoy_percent:+.2f}%", f"{r.stock_return20_cc*100:+.2f}%", f"{r.contribution20_pp:+.3f}"] for r in company.itertuples()]
    company_table = md_table(["病例", "事前权重序", "公司", "参考权重", "上半年收入同比", "归母利润同比", "随后收盘口径收益", "篮子贡献/百分点"], company_rows)
    a22 = sector[(sector.stat_month == "2022-08") & (sector.industry_reference != "行业缺失")]
    neg_industries = int((a22.contribution20_pp < 0).sum())
    loss_share = summary.loc["2022-08", "top5_contribution_pp"] / summary.loc["2022-08", "basket20_cc_return_percent"] * 100
    source_rows = []
    for doc in facts.document.drop_duplicates():
        q = facts[facts.document == doc]
        pages = sorted(set(q.pdf_page.tolist() + scalars[scalars.document == doc].pdf_page.tolist() + qualitative[qualitative.document == doc].pdf_page.tolist()))
        source_rows.append([q.iloc[0]["name"], doc.split("_")[1], receipts[doc]["published_date"], "、".join(map(str, pages)), f"[原报告]({receipts[doc]['url']})"])
    source_table = md_table(["公司", "报告期", "公开日期", "本轮引用PDF页", "来源"], source_rows)
    text = f"""# 第十六轮：资金结构怎样进入公司，再进入510300

**本轮最明确的进展，是把货币结构、经营压力和股价结果连到了同一张可核对的表上：2022年剪刀差三个月改善时，代表性公司的回款、息差、利润率与新业务已经显露不同的压力；其后下跌又覆盖多数成分。低波动和已公布利润增长，都不足以证明这条传导已经转好。**

这仍是两个固定病例的机制核对。能够确认其中发生了什么，尚不能从两个病例估计某类背景下一定涨跌的概率。总目标保持进行中，尚无确定性预测结论。

## 一、这次比较的起点和口径

继续使用第十四、十五轮已固定的2020年8月、2022年8月货币公告病例，没有根据本轮公司结果重新选月。公告分别在2020-09-11和2022-09-09收盘后发布，观察快照为当日21:00。原E0分别从2020-09-14、2022-09-13开盘到第20个交易日收盘，终点为2020-10-19、2022-10-17。

两例都存在三个月剪刀差改善及正的当期相对余额分量，企业累计新增贷款同比也都增长超过三成；这些相似点没有消除其余条件的差别。尤其2022年三个月改善0.4个百分点，**当月较上月却恶化0.8个百分点**，不能笼统写作“持续改善”。C是M1相对M2余额变化的对数分解量，不是经营活力的直接测量，也不是普通同比百分点。

{context}

企业中长期贷款的累计差额不能替代当月变化：原资料中，2022年8月当月企业中长期贷款较上年同月多增2,138亿元，2020年对应多增2,967亿元；2022年的当月边际改善应当保留。居民中长期、订单与资金活化的偏弱也应保留。数据来源沿冻结的月度底表和第十五轮外部时钟结果，见{link('前轮完整报告', PREV / '第十五轮_信贷改善与外部利率消息时序.md')}。美国日度收益率沿前轮保守次日可用时间，不把中国收盘后才形成的美国利率当作中国当日已知值。

公司样本按起点前最近完整参考权重的前五名选取：每例五家，共十个公司报告期、六家不同公司。报告均已在起点前公开，属于已知的上半年背景，**不等于9月起点的新信息，也不是市场对未来盈利的预期**。

## 二、2022年的下跌覆盖多数成分，整体波动却与2020年相近

{outcomes}

2022年300只成分只有48只最终上涨，上涨公司的起点权重仅11.88%；{len(a22)}个有名称的参考行业中，{neg_industries}个贡献为负。电子、非银金融、电力设备、食品饮料、银行分别贡献约−0.813、−0.778、−0.743、−0.610、−0.540个百分点。医药生物贡献+0.113个百分点，社会服务贡献约+0.001个百分点，反向情况完整保留。

起点前五名合计占16.32%参考权重，贡献了全篮子净跌幅的{loss_share:.2f}%及下行平方的18.47%。因此该段下跌不能只用少数大权重公司的跌价说明。2020年则有169只上涨，非银金融、银行、家用电器等提供正贡献；食品饮料整体贡献并不是主要上涨来源。

两个篮子的事后总波动分别18.95%、18.80%，但下行尺度分别9.86%、15.04%。这是“同样一个总波动数，内部路径可能很不同”的直接例子。**这里的未来20日波动与涨跌参与面是结果，不能倒填成入场条件。** 2022年起点原510300的波动较低、下行波动近5日也在减弱，随后依然出现广泛下跌；“降波”本身没有证明需求恢复或卖压永久结束。

{link('查看全部行业贡献图', OUT / 'figures/全部行业贡献与下行结构.png')}

## 三、已经增长的利润，掩盖不了不同的经营压力

{company_table}

表内公司收益为起点收盘到原退出收盘的含分红现金观察收益。公司财务是当年上半年同比，日期与股价收益期不同。没有把两列解释为同一期间的因果效应，也没有将五家归母利润直接加总为全指数每股盈利。

**招商银行：资金结构能够通过资产、负债两边同时压缩息差。** 2022H1归母利润同比+13.52%，净利息收益率却从上年同期2.49%降至2.44%；第二季度2.37%，比第一季度2.51%低14个基点。管理层把原因具体落到：零售贷款结构偏弱，贷款需求及定价承压，企业结算活期增长受限，以及居民资金向定期存款转移。它为“资金未充分进入交易周转—存款结构变化—银行盈利空间承压”提供了公司层面的证据。它没有量化这家银行造成全国M1/M2变化的份额，也不能单凭该说明确定之后20日股价下跌的因果比例。来源：{cite('招商银行2022年中报', '600036_2022H1.pdf', '7、9、15、36—37')}。

2020年招行净利息收益率也同比下降，归母利润同比−1.63%，但本轮之后收盘收益+7.38%。这保留了一个反例：同一项已公布经营压力，在不同预期、价格位置和后续消息下，不必给出相同短期股价方向。来源：{cite('招商银行2020年中报', '600036_2020H1_巨潮.pdf', '7、9')}。

**五粮液：同样现金流下降，必须区分结算时点与经销商资金压力。** 2020H1经营现金净额同比−86.04%，公司解释涉及春节较早令回款落在上一年第四季度，以及当期税金支付。2022H1归母利润同比+14.38%，经营现金净额同比−78.33%，公司则说明调整预收现金比例、订单管理以缓解经销商资金压力，并叠加上年票据到期收现较高的基数。这里更接近用户所说“钱为什么这样变化”，而不是给负现金流变化机械贴标签。经销商压力的说明支持渠道环节偏弱，不等于已证明全部终端销量下降。来源：{cite('五粮液2020年中报', '000858_2020H1.pdf', '5—6、10')}；{cite('五粮液2022年中报', '000858_2022H1.pdf', '7、14')}。

**宁德时代：需求放量与单位盈利承压可以并存。** 2022H1收入同比+156.32%、归母利润+82.17%，整体毛利率18.68%，同比下降8.58个百分点；动力电池毛利率15.04%，同比下降7.96个百分点。公司明确提到部分上游材料成本上升。境外电池业务收入占合并营业收入19.70%，这只是已列示的境外电池业务，不能称全部海外收入比例。其财务费用项下当期汇兑收益约0.625亿元，但同时有外币资金、外币债务、采购相关套期及计入其他综合收益的套期变动；不能据人民币汇率一个方向断定综合损益。上半年已发生汇兑结果也不能外推为9—10月已知收益。来源：{cite('宁德时代2022年中报', '300750_2022H1.pdf', '7、13—14、135、140—141、147')}。

**中国平安：集团利润回升不等于新保单价值回升。** 2022H1归母利润同比+3.9%，寿险及健康险新业务价值按原列报数同比−28.5%；使用上年末假设与方法重列比较则为−20.3%，两种口径均保留。总投资收益率3.1%，低于上年同期3.5%，净投资收益率3.9%则高于上年同期3.8%。保险资产与负债两侧、存量业务与新业务、经常性与市场波动必须拆开。公司还披露投资收益率与风险贴现率同时上调50基点时，模型中的集团内含价值上升；这不是美债利率单独冲击的试验，也不是其股票应上涨的证据。来源：{cite('中国平安2022年中报', '601318_2022H1.pdf', '10、26、43、68')}。

2020H1平安归母利润同比−29.7%，比较基期含104.53亿元税收政策一次性损益；扣非利润仍同比−21.2%，不能把全部下降归为基数。本轮其后收盘收益却+7.05%，进一步说明旧利润增速不是短期方向的充分条件。来源：{cite('中国平安2020年中报', '601318_2020H1.pdf', '10、26')}。

**贵州茅台：要防止把所有外部压力都写成融资成本或海外经营恶化。** 2022H1收入+17.38%、归母利润+20.85%；国外收入占主营业务收入约3.59%，报告期末无浮息负债。此后本轮收盘收益−6.85%。因此，不能仅凭美国利率上行，就将该跌幅解释为该公司浮息借款成本增加；少量国外收入也不能证明其主要经营全部暴露于汇率损失。合并经营现金净额约−0.112亿元，报告解释主要涉及财务子公司客户及同业资金变化，不能直接等同白酒经营现金枯竭。来源：{cite('贵州茅台2022年中报', '600519_2022H1_巨潮.pdf', '5、7—8、80')}。

茅台2020H1利润亦增长而本轮其后股价下跌；恒瑞2020H1收入、利润和经营现金净额均增长，本轮其后收益仍略负。恒瑞报告期没有借款，但这不意味着股票不受折现率、政策或需求预期影响。来源：{cite('贵州茅台2020年中报', '600519_2020H1.pdf', '5、7')}；{cite('恒瑞医药2020年中报', '600276_2020H1.pdf', '5、92')}。

## 四、这使我们的结论具体到了哪一步

已经获得公司原文支持的链条是：**贷款与存款的结构变化，要通过实际需求、客户结算、利息收支、成本和新业务，才能转化为各类成分的盈利与现金流；这些环节的方向可以不同。** 招行的存款活期化不足与贷款定价、五粮液的渠道结算安排、宁德时代的原料成本、平安的新业务，给出了具体的中间环节。宏观总量、三个月剪刀差、已公布利润、单一波动数，都不能代替它们。

2022年起点同时存在正的当期相对余额分量、企业总信贷扩张，以及上述经营压力；随后美国利率与人民币中间价路径继续变化，参考篮子又出现大范围下跌。这支持继续检验“国内传导不足，叠加外部折现与风险偏好压力”的解释，而不是把一切归因于M1、M2或美债中的一个数。

**尚未识别的是股价变化在未来盈利预期、股票折现率、风险溢价及其他消息之间的比例。** 公司报告告诉我们已经发生的经营变化，无法提供当时市场完整预期。直接融资通道不足以解释某公司的下跌，也不自动证明剩余部分全是估值压缩。原20日窗口内还有新的国内金融数据与其他公司消息，本轮没有建立完整消息反事实，不能把公告前后全部收益归给单一事件。

下一步仍在历史资料范围内，把本轮得到的传导预期写清楚，再检查剩余固定月份的同向与反向案例：若主张需求修复，应在订单、真实融资用途、回款或利润率看到对应变化；若主张只是资金与存款迁移，应保留经营端未改善的可能。开始扩大验证前固定判据，不能用已经知道的涨跌回头挑选最合意因素。尚未通过独立验证的解释不提升为交易规则。

## 五、数据口径和本轮实际完成范围

两例起点各300只历史成分完整保留，权重使用此前8月31日参考快照并归一化；它不是ETF真实每日持仓，也不是已认证的历史首版权重。行业使用起点附近月度参考标签，保留行业未知权重2020年0.723%、2022年0.456%，不把2020、2022不同分类强行拼接。

成分可靠数据是收盘价，因此主拆解从公告当日收盘到原20日终点收盘；这个起点早于公告，**只用于相同价格口径的事后观察，不是公告后可成交的入场**。原E0开盘收益另列，并加上原入场日收盘至终点的19日敏感性结果。篮子与510300同口径差额保留为未解释差额，没有分摊成某个行业的真实ETF贡献。

每股从固定参考本金出发持有，现金分红不再投资，送转调整股数。股数变化包含2022-10-13荣盛石化1.4倍送转及每原股0.4元分红。现金分红按除权日权益近似入账，不模拟登记到账、费用、整手、委托及可交易账户。以每日个股本金变动除以上日篮子净值，计算与篮子收益的协方差贡献及下行平方贡献；二者都能加总回完整篮子。下行贡献可能为负，意味着相关股票在篮子下跌日对冲部分跌幅，不意味着它期末必须上涨。

唯一新增价格状态补证是国金证券2020-10-12仍停牌：{cite('国金证券复牌公告', '600109_20201013_复牌_巨潮.pdf', '1')}说明9月21日起停牌、10月13日复牌，复牌前收与停牌前有效价格15.29元一致。该后出公告仅用于历史结果复原，没有进入2020-09-11的已知信息，也未修改父数据及旧研究结论。

已完成12000个成分日、600个公司病例的持股现金递推与风险加总复算；核对10份中报的30项成对财务原数、37项专项数值和14项原因定位，保存33张原页渲染。已查看主要传导页图像，确认表格、单位和脚注。公开日期均早于各自起点；当前下载的原PDF不等同已认证不可修订的当年首版。保存结果复算通过并不等于外部独立验证。

{source_table}

## 六、文件导航

- {link('全部成分和行业的数值结果目录', OUT / 'results')}
- {link('十家公司经营背景与后续收益', OUT / 'results/事前前五名_经营背景与后续收益.csv')}
- {link('全部行业贡献与下行结构图', OUT / 'figures/全部行业贡献与下行结构.png')}
- {link('原因原文定位与解释', OUT / 'results/公司传导_14项原文定位与解释.csv')}
- {link('冻结观察范围', OUT / 'protocol.json')}
- {link('来源、日期和下载回执', OUT / 'source_receipts.json')}
- {link('本轮完成回执', OUT / 'completion_receipt.json')}

研究状态：本轮有实质进展；总目标尚未达成；新增模型0、新增策略账户0；没有给出当前市场观点或仓位。
"""
    (OUT / "第十六轮_成分股盈利与外部压力的实际传导.md").write_text(text, encoding="utf-8")


def main():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮已完成，禁止覆盖完成记录。")
    summary = load("两个病例_完整篮子与510300同口径比较.csv").set_index("stat_month")
    stock = load("600个公司病例_全部收益及风险贡献.csv")
    sector = load("全部行业_收益与下行平方贡献.csv")
    company = load("事前前五名_经营背景与后续收益.csv")
    checks = check_saved_outputs(summary, stock, sector)
    make_chart(summary, sector)
    report(summary, sector, company, checks)
    save_json("results/保存结果复算.json", checks)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("成分原字段、股数现金、行业加总、三种ETF起点口径及财务原数复算通过。")
    print("报告和完整行业图已生成，等待图像检查后记录本轮完成状态。")


if __name__ == "__main__":
    main()
