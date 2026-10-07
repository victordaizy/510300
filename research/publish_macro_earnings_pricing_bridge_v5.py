"""发布第五轮的具体传导结论、原公告核对和可复查图表。"""
from pathlib import Path
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.macro_earnings_pricing_bridge_v5 import OUT, PARENT, read, save, now, digest


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |", *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def pct(value, scale=100):
    return "缺失" if pd.isna(value) else f"{value * scale:+.2f}%"


def primary_check():
    income = read("income.parquet")
    primary = []
    for code, title, current, previous, file, pages in [
        ("601318.SH", "中国平安", 68047000000, 74619000000, "平安_2025中期业绩演示原文.pdf", [9, 39, 40]),
        ("600030.SH", "中信证券", 13719079187.22, 10569764458.88, "中信证券_2025半年报摘要原文.pdf", [3]),
    ]:
        row = income[income.stock_code.eq(code) & pd.to_datetime(income.report_end).eq("2025-06-30")].iloc[0]
        item = {"stock_code": code, "company": title, "report_end": "2025-06-30", "pdf_pages_one_based": pages, "source_file": file, "source_sha256": digest(OUT / "sources" / file), "primary_parent_net_profit_ytd": current, "primary_previous_ytd": previous, "primary_profit_yoy": current / previous - 1, "aggregator_current_value": float(row.parent_net_profit), "aggregator_nominal_notice": str(row.available_at), "aggregator_update_date": str(row.update_date), "aggregator_minus_primary": float(row.parent_net_profit) - current, "relative_difference": float(row.parent_net_profit) / current - 1, "scope": "仅两家公司当期归母利润及原公告定义核对，不代表300家公司版本均已核验，也不替换已冻结汇总。"}
        if code == "601318.SH":
            item.update(primary_operating_profit_2025=77732000000, primary_operating_profit_2024=74986000000, operating_profit_yoy=77732 / 74986 - 1, operating_profit_assumption="两期均按2024年末4.0%长期投资回报假设；短期投资波动与一次性非经营项目另列。", warning="原文集团调整项与归母净利润不能忽略少数股东口径后强行求和。")
        primary.append(item)
    save(OUT / "results/两家公司_原公告经营口径核对.json", {"at": now(), "selected_after_reviewing_aggregate": True, "purpose": "检验TTM与新报告、营运与净利润的含义及现实版本差异；不是预选预测样本。", "pdf_relevant_pages_visually_checked": True, "records": primary})
    return primary


def figure(cases, periods):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(14.8, 10.6))
    fig.patch.set_facecolor("#f7f8f7")
    fig.subplots_adjust(left=.075, right=.97, bottom=.13, top=.84, hspace=.48, wspace=.30)
    fig.text(.075, .955, "剪刀差改善之后，还要辨认哪一层真正变了", fontsize=23, weight="bold", color="#153d43")
    fig.text(.075, .907, "固定病例：2025年9月12日，观察8月货币数据。利润、现金流与定价并不同步。", fontsize=12)
    fig.text(.075, .876, "财务：现有版本的历史重建，含后来修订；此图用于解释，不能作为当时已知的预测证据。", fontsize=11, color="#875327")
    case = cases.set_index("stat_month").loc["2025-08"]
    group = periods[(periods.stat_month == "2025-08") & (periods.group_type == "金融分层")].set_index("group")
    labels = ["银行\n24家公司", "非银金融\n27家公司", "非金融\n247家公司"]
    names = ["银行", "非银金融", "非金融"]
    x = np.arange(3)
    ax = axs[0, 0]
    for offset, col, name, color, scale in [(-.18, "ttm_growth_pp", "过去十二个月", "#176f78", 1), (.18, "recent_ytd_profit_yoy", "最新半年", "#d7984c", 100)]:
        bars = ax.bar(x + offset, [group.loc[g, col] * scale for g in names], .33, label=name, color=color)
        ax.bar_label(bars, fmt="%+.1f", padding=4, fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(-12, 80)
    ax.set_ylabel("归母利润同比（%）")
    ax.set_title("① 增长发生在哪个行业、哪段时间", loc="left", pad=13)
    ax.legend(frameon=False, loc="upper left")
    ax.axhline(0, color="#7a898b", linewidth=.8)

    ax = axs[0, 1]
    recent = group.loc["非银金融", "recent_period_contribution_pp"]
    old = group.loc["非银金融", "remainder_period_contribution_pp"]
    bars = ax.bar([0, 1, 2], [recent, old, recent + old], color=["#d7984c", "#6b91a4", "#176f78"], width=.56)
    ax.bar_label(bars, fmt="%+.2f", padding=5)
    ax.set_xticks([0, 1, 2], ["2025上半年\n对比2024上半年", "2024下半年\n对比2023下半年", "合计\nTTM同比"])
    ax.set_ylim(0, 80)
    ax.set_ylabel("对非银金融TTM同比的贡献（百分点）")
    ax.set_title("② 约八成利润增量来自更早的半年", loc="left", pad=13)
    ax.text(.02, .93, "12.70 + 50.26 = 62.96", transform=ax.transAxes, color="#486168", fontsize=11)

    ax = axs[1, 0]
    values = [case.financial_nonfinancial_total_operating_revenue_yoy * 100, case.financial_nonfinancial_parent_net_profit_yoy * 100, case.financial_nonfinancial_cash_cohort_cashflow_yoy * 100]
    bars = ax.bar([0, 1, 2], values, color=["#b85f55", "#b85f55", "#176f78"], width=.55)
    ax.bar_label(bars, fmt="%+.2f", padding=5)
    ax.set_xticks([0, 1, 2], ["营业收入", "归母利润", "经营现金流净额"])
    ax.axhline(0, color="#7a898b", linewidth=.8)
    ax.set_ylim(-6, 18)
    ax.set_ylabel("非金融同组公司TTM同比（%）")
    ax.set_title("③ 现金流改善，收入和利润仍未同向", loc="left", pad=13)
    ax.text(.02, .90, "现金流变好不能直接认定为订单扩张或财政清欠", transform=ax.transAxes, fontsize=10, color="#486168")

    ax = axs[1, 1]
    vals = [case.price_bridge_20_log_pe_pp, case.price_bridge_20_log_implied_denominator_pp, case.price_bridge_20_log_price_pp]
    bars = ax.bar([0, 1, 2], vals, color=["#6b91a4", "#b85f55", "#176f78"], width=.55)
    ax.bar_label(bars, fmt="%+.2f", padding=5)
    ax.set_xticks([0, 1, 2], ["PE变化", "价格/PE的\n隐含分母变化", "价格变化"])
    ax.axhline(0, color="#7a898b", linewidth=.8)
    ax.set_ylim(-5, 15)
    ax.set_ylabel("过去20交易日的对数变化（×100）")
    ax.set_title("④ 价格已经反映了多少重新评价", loc="left", pad=13)
    ax.text(.02, .90, "截至9月11日；价格简单收益 +8.98%", transform=ax.transAxes, fontsize=10, color="#486168")
    for a in axs.flat:
        a.set_facecolor("#f7f8f7")
        a.yaxis.grid(True, alpha=.15)
        a.set_axisbelow(True)
    fig.text(.075, .055, "口径：TTM为滚动十二个月；行业利润为公司归母金额汇总，包含亏损，并非指数权重收益贡献。另有2家公司行业缺失。\n价格/PE仅为代数隐含分母，不等于官方指数EPS；PE变化不能单独归因于利率、流动性或风险偏好。", fontsize=10, color="#486168", linespacing=1.7)
    fig.savefig(OUT / "figures/2025年8月_剪刀差背后的盈利期间现金流与定价.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def run():
    verification = json.loads((OUT / "verification_receipt.json").read_text(encoding="utf-8"))
    assert verification["status"].startswith("PASS_")
    cases = pd.read_csv(OUT / "results/五个固定病例_盈利与定价.csv")
    periods = pd.read_csv(OUT / "results/五个固定病例_全部行业利润增长期间来源.csv")
    aggregate = pd.read_csv(OUT / "results/逐观察日_全部行业盈利与非金融现金流.csv")
    primary = primary_check()
    figure(cases, periods)

    financial_rows, price_rows, timing_rows = [], [], []
    for r in cases.to_dict("records"):
        m = r["stat_month"]
        financial_rows.append([m + " / " + r["observation_date"], pct(r["financial_bank_parent_net_profit_yoy"]), pct(r["financial_nonbank_parent_net_profit_yoy"]), pct(r["financial_nonfinancial_parent_net_profit_yoy"]), pct(r["financial_nonfinancial_cash_cohort_cashflow_yoy"])])
        price_rows.append([m, pct(r["price_bridge_60_price_return"]), f'{r["price_bridge_60_log_price_pp"]:+.2f}', f'{r["price_bridge_60_log_pe_pp"]:+.2f}', f'{r["price_bridge_60_log_implied_denominator_pp"]:+.2f}', pct(r["E0_20_return"]), pct(r["E1_20_return"])])
        v = periods[(periods.stat_month == m) & (periods.group_type == "金融分层") & (periods.group == "非银金融")].iloc[0]
        timing_rows.append([m, f'{v.ttm_growth_pp:+.2f}%', pct(v.recent_ytd_profit_yoy), f'{v.recent_period_contribution_pp:+.2f}', f'{v.remainder_period_contribution_pp:+.2f}', v.report_periods])
    flow = pd.read_csv(PARENT / "results/2025年6至8月_社融多增来源.csv")
    sept = cases.set_index("stat_month").loc["2025-08"]
    all_sept = aggregate[(aggregate.observation_date == "2025-09-12") & (aggregate.group_type.isin(["总样本", "金融分层"]))].set_index("group")
    report = f'''# 第五轮：钱的来源、利润发生的时间与价格已经走过的路

这轮将用户要求的“立体看待”落实为可以逐段核对的事实链：**货币为什么变 → 信贷由谁扩张、用于什么 → 影响哪些企业、哪一段经营 → 价格已反映多少 → 波动和市场内部参与说明市场怎样消化信息。**价格也会反过来影响金融企业投资收益和融资条件，不能把同一个过程产生的几个指标重复当成独立验证。

目前新增证据支持一个更具体的判断：**2025年夏季的剪刀差改善，不能直接解释为私人信用扩张推动全面盈利复苏。资金来源、企业现金流、不同金融行业的利润、利润所属期间与市场重估，呈现明显分化。**这是对传导解释的限制，还不是未来必涨或必跌的结论。

本轮完整保留104个月、445周和既定五个病例，另保存两个完整历史段共28个月。公司财务连接覆盖503个不同交易日、150,900条公司观察；同一天出现在月度和周度表中只构成一次公司财务观察。没有新增收益模型、账户回测、阈值或仓位。

**一、先从造成剪刀差变化的来源出发。**

前两轮的余额/隐含基数分解已显示：2025年7、8月，剪刀差三个月变化分别为+3.3、+2.8个百分点，余额相对变化项仅约+0.29、+0.22个对数百分点，基数/修订项约+2.93、+2.50个对数百分点。普通同比百分点与对数百分点不能直接相加；本结论来自各自完整恒等式，不把“基数/修订”进一步解释成已追踪的企业资金行为。

此前直接核过人民银行原文：2025年1—7月、1—8月社融累计同比多增中，政府债券多增分别解释约{flow.loc[flow.stat_month.eq('2025-07'), 'government_share_of_tsf_yoy_increase'].iloc[0] * 100:.2f}%和{flow.loc[flow.stat_month.eq('2025-08'), 'government_share_of_tsf_yoy_increase'].iloc[0] * 100:.2f}%。这是“同比多增的来源”，不是政府债占全部社融的比例。同期企业和居民中长期贷款累计新增均少于上年同期，制造业新订单PMI分别49.4、49.5。来源与全文见[第四轮报告](../510300_macro_transmission_context_v4/第四轮_立体传导与明确结论.md)。

这些事实足以否定“社融多增就是企业主动扩大借贷”这一简单解释。但政府融资究竟经由项目支出、置换负债、偿还欠款还是其他渠道传导，仍需要对应资金用途的直接证据；不能从总量反推钱已经流入股市。

**二、把宏观工业利润换成沪深300实际成分的行业观察后，分化更明显。**

下表均为观察日历史成分集合、同一公司本期与上年TTM归母利润之和的同比，包含亏损。现金流只汇总同报告期数据完整的非金融公司；没有把银行的经营现金流解释成实体企业回款。

{table(['货币统计月 / 观察日','银行利润TTM','非银金融利润TTM','非金融利润TTM','非金融经营现金流TTM'], financial_rows)}

按当前库回顾重建，2025年9月12日的300家公司归母利润合计同比+{sept.financial_all_parent_net_profit_yoy * 100:.2f}%。其中非银金融对这个公司金额汇总增速贡献{all_sept.loc['非银金融', 'profit_change_contribution_to_all_growth_pp']:+.2f}个百分点，银行贡献{all_sept.loc['银行', 'profit_change_contribution_to_all_growth_pp']:+.2f}个百分点，非金融贡献{all_sept.loc['非金融', 'profit_change_contribution_to_all_growth_pp']:+.2f}个百分点，另有行业缺失2家公司贡献{all_sept.loc['行业缺失', 'profit_change_contribution_to_all_growth_pp']:+.2f}个百分点。总体正增长不能证明各类企业一起改善。

这里不是指数上涨归因：沪深300使用调整股本和相应价格构造指数，公司集团归母利润占比不等于指数权重；A/H分配、上市母子公司合并范围重叠也会影响解释。历史权重未完成原版本认证，本轮只作不归一化的参考。参见[中证沪深300编制方案](https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300_Index_Methodology_cn.pdf)。

**三、即使同一个行业“利润高增长”，也要继续问：增长是哪一段时间发生的？**

对于半年报，滚动十二个月利润=本年上半年利润+上年下半年利润。因此滚动同比金额增量=本年上半年相对上年上半年的增量+上年下半年相对更早下半年的增量。这个分解没有设置收益阈值；补充方案在进一步计算前单独保存，明确记录已经看过TTM汇总及平安半年报检索结果，不伪称完全盲检。

{table(['货币月','非银金融TTM同比','最新累计期同比','最新累计期贡献/百分点','更早剩余期贡献/百分点','最新报告期与公司数'], timing_rows)}

2025年8月病例对应9月12日观察，27家非银金融公司全部已经进入2025年半年报：**TTM同比+62.96%=最新半年贡献12.70个百分点+更早半年贡献50.26个百分点**。后者约占TTM利润增量79.83%。最新半年利润本身同比约+17.66%，不应被叙述成当下经营增长了63%。7月病例观察时，这27家公司仍全部在一季报，不能把两个观察点视作同一期财报变化。

2020年6月病例也有这种差别：非银金融TTM同比+26.03%，但最新一季度累计利润同比-23.69%。并非一种“高增长”标签可以概括。

“更早剩余期贡献”只是金额与期间的精确分解，不等于已经证明那个期间的利润来自资本市场行情，也不等于不可持续。其原因仍需投资收益、承保、佣金、息差、减值等业务原文逐项解释。

**四、非金融企业现金流改善，并不自动意味着订单、盈利和风险偏好同步改善。**

2025年9月12日，247家非金融同组公司的TTM收入同比{sept.financial_nonfinancial_total_operating_revenue_yoy * 100:+.2f}%、归母利润{sept.financial_nonfinancial_parent_net_profit_yoy * 100:+.2f}%、经营现金流净额{sept.financial_nonfinancial_cash_cohort_cashflow_yoy * 100:+.2f}%；经营现金流/归母利润约{sept.financial_nonfinancial_cash_cohort_cashflow_to_profit:.2f}倍。最新半年累计利润同比约-0.24%，与TTM的-2.56%又是两个时间层次。

这支持“现金流与损益不同步”的事实，但没有识别现金增加来自销售收款加快、存货变化、应付款变化、税费还是其他因素。经营净现金流不是销售收现率，不能据此认定财政清欠已经使这些公司获益。这里也保留了重要的正面信息：现金流改善确实存在，只是尚不足以证明利润和需求已经同步扩张。

**五、原公司公告说明，净利润、营运利润与金融市场变化需要分别解释。**

平安2025年中期原演示资料第9、39、40页：上半年归母净利润680.47亿元，上年同期746.19亿元，同比约-8.8%；归母营运利润777.32亿元，上年749.86亿元，同比约+3.7%。原文将短期投资波动和一次性非经营项目另列，营运利润两期采用2024年末4.0%的长期投资回报假设。不能把营运利润等同于现金利润，或将不同股东口径的调整项直接拼到归母净利润。[平安官方原文](https://group.pingan.com/resource/pingan/IR-Docs/2025/pingan-interim25-results-presentation.pdf)

中信证券2025年半年报摘要纸面第2页（PDF第3页）披露归母净利润137.19亿元，同比+29.80%。两家非银公司的最新半年方向不同，再次说明“非银金融整体增长”需要继续拆业务和公司。[中信证券官方原文](https://www.cs.ecitic.com/newsite/tzzgx/ggyth/gg/aggg/202508/P020250828660220719823.pdf)

原公告核对还发现一个实际版本差异：当前财务库的中信证券2025年上半年归母利润约137.64亿元，字段更新日为2026年8月21日，比上述2025年公告高约0.445亿元、0.324%。本轮完整保留两者，没有拿当前库数字冒充首次公告值，也没有只替换一个对结论有利的数据点。所有行业汇总因此统一保留“当前版本回顾重建”身份。两家公司原表只能核实这两份公告及口径，不能认证整段历史。

**六、盈利传导之外，还要看价格已经完成多少重新评价。**

为避免把ETF价格与指数PE混用，本轮改用同日期的000300价格指数和原中证PE来源；只有PE在观察截止前已可用才接入。过去60交易日的对数变化满足：价格变化=PE变化+价格/PE隐含分母变化。下表的三个对数列均为100×对数变化，不是简单收益百分比。

{table(['货币月','此前CSI300价格60日简单收益','价格对数变化','PE对数变化','隐含分母对数变化','随后510300 E0/20日','随后510300 E1/20日'], price_rows)}

例如2021年1月数据公布时，可用的同日期价格/PE窗口截至2月8日：此前60交易日价格简单收益+14.57%，对数变化13.60，其中PE对数变化15.56、隐含分母变化-1.96。此时再把信贷和剪刀差改善理解成“市场尚未反映的全面盈利改善”，证据不足。原E0、E1路径后来仍分别为-11.09%、-13.96%，作为反例保留。

2025年8月数据公布时，价格/PE窗口截至9月11日：此前20交易日价格简单收益+8.98%，对数变化8.60=PE变化10.71+隐含分母变化-2.11。过去60交易日价格已涨18.34%，随后原E0、E1的20日收益分别约+0.39%、+1.68%。这些数值描述价格变化的先后，**没有证明PE升高必然导致随后回报下降**。

价格/PE仅是代数隐含分母；PE口径、样本调整、亏损剔除和财务更新均可能改变它，不能直接称作官方已披露EPS。PE上涨也可能同时包含盈利预期、无风险利率、风险补偿等变化，不能从恒等式分配各自的因果份额。来源接口存在总股本PE与计算用股本PE等不同口径，见[原数据接口说明](https://github.com/akfamily/akshare/blob/main/docs/data/index/index.md)。本轮不将未核明口径的分母升级为指数盈利序列。

**七、波动应放在这条传导链上，而不是另设一个“好或坏”的标签。**

第三轮已拆开上涨和下跌对波动的贡献，以及窗口中旧跌幅移出与最近新跌幅进入的差别。第四轮的2024年9月连续观察也保留：剪刀差同为-13.6个百分点、下行波动尺度下降的情况下，政策新信息和价格急涨不断改变已知条件，后续入场窗口结果发生明显变化。

因此后续观察至少要同时讲清：低波动来自没有新增卖压、成交沉寂还是旧波动移出；上涨是否已提前反映政策与盈利预期；受益是多数行业还是少数权重行业；利润改善是新经营、基数、旧利润滚入还是金融市场反馈。这里保留因果问题和时间顺序，不把它们压成一个总分，也不用事后牛熊标签替代解释。

![既定病例的四层对照](figures/2025年8月_剪刀差背后的盈利期间现金流与定价.png)

**本轮证据能支持到哪里。**

可以支持：货币与信贷的表面同向不代表融资主体和用途相同；成分公司总利润改善不代表非金融企业全面改善；TTM增长不能替代最新经营改善；现金净流入改善不能替代利润改善；价格和PE可能在宏观与财报确认前已有明显变化。这些是有完整表和原公告支撑的具体限制。

仍未证明：资金唯一去向、每类冲击对价格的因果贡献、当时市场一致预期与实际公告的差，以及跨不同历史阶段仍可使用的方向优势。五个病例和28个月完整段原本已经看过后续表现，本轮不是独立样本预测验证。E0/E1为原固定股数、含持有期分红的观察标签，未扣交易成本，不等于可执行账户结果。

数据边界明确：149,091条有效公司财务观察中，142,609条至少一个TTM依赖的更新日晚于观察日。名义公告时钟通过检查，不消除后来修订的风险；未标记晚更新的其余记录也未因此获得首次版本认证。行业继承此前月度状态，参考权重未完成历史版本核验。成分名单截止2026年8月14日，后续4个周度原点维持缺失，2026年8月货币数据仍晚于原行情截止日而为NO_VIEW。行业财务不应作为新模型的历史可交易输入。

已完成必要复算：549个原点的38,089个原标签单元保持一致；五病例的11,474个TTM数值从保存的原始报表另算；149,091个公司利润期间恒等式和1,084个同日期价格/PE窗口通过。输入专用表同时删除E0/E1和公布后首次反应字段。此检查只证明所述计算一致，没有认证未来盈利或完成外部独立验证。

目标状态继续为active，未宣称找到未来确定收益。下一步围绕两段完整历史的少数明确传导问题，核对最新报告期的收入、成本、息差、减值和投资收益如何变化，以及消息出现前价格已经反映多少；先补足可区分解释的证据，再形成预先固定、可被新数据否定的命题。不会继续在旧104个月里反复切小组直到找到漂亮收益。
'''
    (OUT / "第五轮_盈利期间现金流与定价传导.md").write_text(report, encoding="utf-8")
    navigation = '''# 第五轮阅读入口

先读《第五轮_盈利期间现金流与定价传导.md》，其中区分已核实事实、回顾性财务重建和待证明的传导原因。

- `results/五个固定病例_盈利与定价.csv`：宏观、信贷、订单、波动、价格位置、行业财务、价格/PE及原E0/E1后续路径。
- `results/五个固定病例_全部行业利润增长期间来源.csv`：TTM增长拆成最新累计期与更早剩余期；保留全部行业。
- `results/两个固定历史段_完整28个月_盈利定价连接.csv`：不只看五病例，检查2020-04至2021-02及2024-04至2025-08完整段。
- `results/104个月_盈利定价与宏观波动完整连接.csv`与`445周_盈利定价与宏观波动完整连接.csv`：全部原点。
- `results/逐观察日_全部行业盈利与非金融现金流.csv`：全部日期、全部行业；公司金额汇总不是指数价格归因。
- `results/全部公司_利润增长期间来源.parquet`：公司级依赖、公告及更新时点、有效性和缺失。
- `results/两家公司_原公告经营口径核对.json`与`sources/`：两家公司的原公告、截图和真实版本差异。
- `protocol.json`、补充方案及`code/`：完整计算口径和可复算代码；依次运行主脚本、extend、verify、publish。主脚本拒绝覆盖完成记录。

文件名中的“不含未来标签”仅表示不含后续价格结果，不表示财务数值已经得到历史首次版本认证。没有新增预测模型、收益优化、仓位或订单。
'''
    (OUT / "00_阅读说明.md").write_text(navigation, encoding="utf-8")
    save(OUT / "goal_progress.json", {"updated_at": now(), "goal_status": "active", "goal_achieved": False, "classification": "PROGRESS_IN_EARNINGS_PERIODS_CASHFLOW_AND_PRIOR_PRICING", "completed": "历史成分行业财务连接、利润期间分解、同日期价格PE桥、两个原公司公告口径核对和全部原点保留。", "remaining": "财务首次版本与当时预期差、资金用途至企业业务的原因连接、独立新验证；未证明可交易方向优势。", "next_bounded_action": "围绕两段完整历史核对最新报告期的经营变动和此前定价；不按既知后续收益选公司、拼阶段或调参。", "blocking_condition_repeated": False, "orders": 0})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "completion_receipt.json", {"at": now(), "study_status": "ROUND_COMPLETE_GOAL_REMAINS_ACTIVE", "report_sha256": digest(OUT / "第五轮_盈利期间现金流与定价传导.md"), "protocol_sha256": digest(OUT / "protocol.json"), "financial_is_strict_pit": False, "new_models": 0, "new_accounts": 0, "orders": 0, "goal_achieved": False})
    print("第五轮报告、图表和目标进度已保存；目标未标记完成。")


if __name__ == "__main__":
    run()
