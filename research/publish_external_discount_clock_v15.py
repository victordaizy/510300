"""从落盘结果生成两个八月病例的图表及中文机制说明。"""
from pathlib import Path
import shutil
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_external_discount_clock_v15"


def link(path, label):
    return f"[{label}](<{path.as_posix()}>)"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def main():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮报告已完成，不覆盖。")
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    wide = pd.read_csv(OUT / "results/两个病例_外部条件与原收益对照.csv").set_index("stat_month")
    daily = pd.read_csv(OUT / "results/原40个交易日_完整路径及当时可见外部信息.csv")
    segments = pd.read_csv(OUT / "results/全部消息分段_同一入场本金贡献.csv")
    nodes = pd.read_csv(OUT / "results/官方消息_北京时间与信息阶段.csv")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfbf8", "axes.facecolor": "#fcfbf8", "axes.labelcolor": "#34424b"})
    fig, axes = plt.subplots(4, 2, figsize=(13.3, 13.8), sharex="col", sharey="row")
    fig.subplots_adjust(left=.09, right=.96, top=.87, bottom=.11, hspace=.23, wspace=.14)
    fig.suptitle("两次剪刀差改善后的外部条件与510300路径", fontsize=20, x=.08, ha="left", y=.975, color="#172f3b")
    fig.text(.08, .942, "全部20个交易日｜竖线标出官方公告公布后，第一个中国交易日", color="#5a6770", fontsize=11)
    for col, case in enumerate(["2020-08", "2022-08"]):
        q = daily[daily.stat_month == case]
        w, m = wide.loc[case], monthly.loc[case]
        x = q.day_number.to_numpy()
        ax = axes[:, col]
        ax[0].set_title(f"{case}统计月\n起点 {w.snapshot_at[:10]}　原入场 {w.entry_date}", loc="left", fontsize=12, pad=42, color="#172f3b")
        ax[0].plot(x, q.path_return * 100, color="#174e62", lw=2.6)
        ax[0].fill_between(x, q.path_return * 100, 0, color="#174e62", alpha=.08)
        ax[0].annotate(f"{w.E0_return_percent:+.2f}%", (x[-1], q.path_return.iloc[-1] * 100), xytext=(-5, 10), textcoords="offset points", ha="right", color="#174e62", fontweight="bold")
        for tenor, color, label in [(2, "#b7562d", "美国2年期"), (10, "#2c638d", "美国10年期")]:
            y = (q[f"ust{tenor}_percent"] - w[f"origin_ust{tenor}_percent"]) * 100
            delayed = (q[f"ust_delay1_{tenor}_percent"] - w[f"origin_ust_delay1_{tenor}_percent"]) * 100
            ax[1].plot(x, y, color=color, lw=2, label=label)
            ax[1].fill_between(x, y, delayed, color=color, alpha=.17)
        ax[1].legend(loc="upper left", frameon=False, fontsize=9)
        fy = (q.usdcny_midpoint / w.origin_usdcny_midpoint - 1) * 100
        ax[2].plot(x, fy, color="#81692b", lw=2.2)
        ax[2].annotate(f"{fy.iloc[-1]:+.2f}%", (x[-1], fy.iloc[-1]), xytext=(-5, -15 if fy.iloc[-1] < 0 else 10), textcoords="offset points", ha="right", color="#81692b")
        ax[3].plot(x, q.v_downside20 * 100, color="#795177", lw=2, label="下行波动 D20")
        ax[3].plot(x, q.v_rv20 * 100, color="#648c8d", lw=1.6, ls="--", label="总波动 RV20")
        ax[3].axhline(m.v_downside20 * 100, color="#795177", lw=.8, ls=":", alpha=.7)
        ax[3].legend(loc="upper left", frameon=False, fontsize=9)
        active = nodes[(nodes.stat_month == case) & (nodes.role == "AFTER_ENTRY")]
        for stamp, group in active.groupby("first_china_open", sort=True):
            match = q[q.date == stamp[:10]]
            assert len(match) == 1
            day = int(match.day_number.iloc[0])
            label = "利率决定" if "FOMC" in group.kind.tolist() else "CPI"
            for a in ax:
                a.axvline(day, color="#8c8c81", alpha=.45, lw=.8, ls=":")
            ax[0].text(day, 1.025, label + "\n" + stamp[5:10], transform=ax[0].get_xaxis_transform(), fontsize=8.5, ha="center", va="bottom", color="#696d67")
        for a in ax:
            a.grid(axis="y", color="#dadfdc", alpha=.65, lw=.7)
            a.set_xlim(.7, 20.6)
            a.tick_params(length=0, labelsize=9)
        for a in ax[:3]:
            a.axhline(0, color="#8c9696", lw=.8)
        ax[0].set_ylim(-10.5, 7)
        ax[1].set_ylim(-10, 115)
        ax[2].set_ylim(-2.7, 3.5)
        ax[3].set_ylim(4, 26)
        ticks = [1, 5, 10, 15, 20]
        ax[3].set_xticks(ticks, [f"第{d}日\n{q[q.day_number == d].date.iloc[0][5:]}" for d in ticks])
    labels = ["原入场以来累计收益（%）", "相对起点的美债收益率变化（基点）", "相对起点的美元兑人民币中间价变化（%）", "年化波动（%）"]
    for ax, label in zip(axes[:, 0], labels):
        ax.set_ylabel(label, labelpad=9)
    fig.text(.09, .053, "美债按美国数据日结束后可见的假设对齐，淡带为额外滞后一条美债记录的结果。\n汇率上升表示人民币中间价走弱；该序列不是可成交汇率。收益为原固定份额、含已获得分红的毛收益。", fontsize=9, color="#59656d", linespacing=1.6)
    figure = OUT / "figures/外部利率汇率与原二十日路径.png"
    fig.savefig(figure, dpi=175)
    fig.savefig(figure.with_suffix(".pdf"))
    plt.close(fig)
    w0, w2 = wide.loc["2020-08"], wide.loc["2022-08"]
    m0, m2 = monthly.loc["2020-08"], monthly.loc["2022-08"]
    source20 = "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/be8e91f117384eac8934a50d21a75e42/index.html"
    source22 = "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/b2dfd3ad0adc4d3ca62f1adacea5fc09/index.html"
    body = f"""# 信贷改善怎样面对外部利率与后续新消息

本轮把2020年8月与2022年8月两个既定病例，沿着货币余额、信贷结构、国内资金价格、海外利率、人民币中间价、此前价格路径和后续新消息逐层对齐。**较明确的结论是：国内货币与资金价格的改善，可以与外部融资及折现环境趋紧同时发生；起点的低波动不能替代对这些压力的判断。** 两个病例已经看过结果，因此以下是历史机制诊断，尚未形成可独立验证的方向预测规则。

2022年这个病例的矛盾在起点就已可见：企业信贷总量同比多增、国内资金偏松，但累计信贷结构与居民中长期贷款偏弱，美国国债收益率已在上升，人民币中间价在走弱，价格仍在下行的200日均线下方。之后又发生新的核心通胀和政策信息。2020年的累计信贷组成、海外利率水平与变化、汇率方向以及此前中期价格路径均不同。因此，把两者只归为“剪刀差改善”或“降波”会丢掉有意义的信息。

## 起点已经知道的不同条件

统计月是8月，观察起点分别是2020年9月11日21时与2022年9月9日21时。两次中国金融数据均已在当天公布。原入场分别为9月14日和9月13日；2022年9月12日不是A股交易日。

{table(['起点维度','2020年8月病例','2022年8月病例'], [
['剪刀差过去3个月改善', f'{m0.delta3_spread_pp:+.2f}个百分点', f'{m2.delta3_spread_pp:+.2f}个百分点'],
['当期M1相对M2余额的3个月贡献', f'{m0.current_component:+.3f}对数百分点', f'{m2.current_component:+.3f}对数百分点'],
['企业贷款累计新增额同比', f'{m0.corporate_total_yoy_percent:+.2f}%', f'{m2.corporate_total_yoy_percent:+.2f}%'],
['企业中长期贷款累计同比多增', '+19057亿元', '−3340亿元'],
['票据融资累计同比多增', '−8510亿元', '+19032亿元'],
['居民中长期贷款累计同比多增', '+3181亿元', '−22789亿元'],
['8月当月企业中长期贷款同比多增', '+2967亿元', '+2138亿元'],
['8月当月居民中长期贷款同比多增', '+1031亿元', '−1601亿元'],
['制造业新订单指数', '52.0，较前月+0.3', '49.2，较前月+0.7'],
['国内FDR007相对7天政策利率的近20记录均值', f'{m0.funding_gap_bp:+.2f}基点', f'{m2.funding_gap_bp:+.2f}基点'],
['美国2年期收益率及此前20记录变化', f'{w0.origin_ust2_percent:.2f}%，{w0.origin_ust2_delta20_bp:+.0f}基点', f'{w2.origin_ust2_percent:.2f}%，{w2.origin_ust2_delta20_bp:+.0f}基点'],
['美国10年期收益率及此前20记录变化', f'{w0.origin_ust10_percent:.2f}%，{w0.origin_ust10_delta20_bp:+.0f}基点', f'{w2.origin_ust10_percent:.2f}%，{w2.origin_ust10_delta20_bp:+.0f}基点'],
['人民币中间价每美元折人民币元', f'{w0.origin_usdcny_midpoint:.4f}', f'{w2.origin_usdcny_midpoint:.4f}'],
['中间价此前20记录变化', f'{w0.origin_usdcny_delta20_percent:+.2f}%，人民币走强', f'{w2.origin_usdcny_delta20_percent:+.2f}%，人民币走弱'],
['中国减美国10年期指示利差', f'{w0.origin_cgb10_minus_ust10_bp:+.2f}基点', f'{w2.origin_cgb10_minus_ust10_bp:+.2f}基点'],
['此前60个交易日含分红收益', f'{m0.pre_return60_pp:+.2f}%', f'{m2.pre_return60_pp:+.2f}%'],
['相对200日均线位置及均线近20日方向', f'{m0.price_ma200_gap*100:+.2f}%，均线上行', f'{m2.price_ma200_gap*100:+.2f}%，均线下行'],
['起点20日年化总波动', f'{m0.v_rv20*100:.2f}%', f'{m2.v_rv20*100:.2f}%'],
['起点20日年化下行波动', f'{m0.v_downside20*100:.2f}%，未下降', f'{m2.v_downside20*100:.2f}%，下降且近5日损失项也下降'],
['此前20日上涨成分股占比', f'{m0.internal_breadth20*100:.2f}%', f'{m2.internal_breadth20*100:.2f}%']])}

信贷和国内背景沿用已保存的月表与上一轮单月补充，原文见[2020年8月金融统计](<{source20}>)、[2022年8月金融统计](<{source22}>)。同一比较里，累计和当月必须同时保留：2022年企业中长期贷款当月已经同比多增，不能因为累计少增就写成“当月需求继续全面恶化”。同样，2020年此前20日的成分股广度只有35%，也不支持把它笼统写成全面强势。

美债起点值分别来自美国9月10日和9月8日，而不是中国观察日稍后才形成的美国当天数据。中美10年期利差是各自保守可用的指示值之差，不代表同一时刻、对冲汇率后的可成交套利收益。FDR007是上午定盘利率，不能冒充全日成交加权DR007。普通剪刀差的百分点与相对余额的对数百分点也不直接相加。

## 这些差别怎样影响传导

货币余额改善与信贷增长能否进入企业盈利，仍需经过资金用途、订单、回款、利润率及每股盈利分配等环节。本轮没有贷款逐笔用途数据，不能认定哪一笔贷款直接形成了M1，也不能把信贷增长全部解释为最终需求恢复。前面的企业现金流与银行盈利诊断继续适用。

**本轮新增的机制解释是：国内短期资金价格只覆盖传导链的一部分。** 起点时，2022年的国内资金松与海外利率升、人民币中间价弱并存。推断上，外部利率变化可能影响跨市场资产比较及折现要求，汇率变化又可能改变不同企业的收入、成本与外币负债效果；对510300的净影响还取决于成分行业和企业暴露。本轮没有把美国收益率直接代入人民币股票的折现率，也没有测得“资本外流额”或“盈利损失额”。

2020年美债收益率低且此前变化小，人民币中间价走强；2022年美债已明显上行，中美10年期指示利差为负。这些是起点背景差异的直接证据。它们支持分开解释两个病例，尚不能识别每个因素的独立因果贡献，更不能保证所有相似情形的下一段收益方向。

## 起点之后出现了哪些新信息

官方公告明确的美东时间已逐份换算为北京时间。美国数据日、美东公告时间、中国可见时间与第一个中国交易日分别保存；下表中的CPI均为美国数据。

{table(['北京时间','公告内容','在本研究中的信息位置'], [
['2020-09-11 20:30','8月CPI同比1.3%，核心同比1.7%；两者环比均0.4%','在9月11日21时快照之前，已经知道'],
['2020-09-17 02:00','联邦基金目标区间维持0至0.25%；预测中位数中的2020至2022年末政策利率均0.1%','原入场以后；第4个交易日开盘前知道'],
['2020-10-13 20:30','9月CPI同比1.4%，核心同比1.7%；两者环比均0.2%','原入场以后；10月14日起进入中国交易日信息集'],
['2022-08-10 20:30','7月CPI同比8.5%，核心同比5.9%；环比分别0.0%与0.3%','9月9日起点最新已公布CPI'],
['2022-09-13 20:30','8月CPI同比8.3%，核心同比6.3%；环比分别0.1%与0.6%','原入场日收盘后公布；9月14日起可响应'],
['2022-09-22 02:00','目标区间升至3至3.25%；参与者上调政策利率预测，下调增长预测','原入场以后；9月22日起可响应'],
['2022-10-13 20:30','9月CPI同比8.2%，核心同比6.6%；环比分别0.4%与0.6%','原入场以后；10月14日起可响应']])}

**2022年总体通胀回落与核心通胀上升同时发生。** 不能用8.5%降到8.3%再降到8.2%就概括成外部紧缩压力已解除。原文见[9月13日CPI公告](https://www.bls.gov/news.release/archives/cpi_09132022.pdf)、[10月13日CPI公告](https://www.bls.gov/news.release/archives/cpi_10132022.pdf)。这些环比是季调值，同比是未季调值；本轮没有合格的事前市场一致预期，未把实际值与上一期之差称为“超预期”。

9月的经济预测中，FOMC参与者对2022、2023年末政策利率的中位数分别由6月的3.4%、3.8%变为4.4%、4.6%；对应实际GDP增速中位数由1.7%、1.7%变为0.2%、1.2%。这是政策参与者预测路径的变化，同时包含利率与增长的信息。它在9月9日尚未公布，不能倒填进起点。SEP不是市场一致预期，也不是政策承诺。原文见[6月预测第2页](https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20220615.pdf)、[9月预测第2页](https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20220921.pdf)及[9月利率决定](https://www.federalreserve.gov/newsevents/pressreleases/monetary20220921a.htm)。

同样重要的是，原窗口里还有新的中国金融数据：2020年9月数据于10月14日16:31:34公布，2022年9月数据于10月11日20:30:40公布。它们分别从10月15日、10月12日的中国交易日起成为新增信息。本轮公告目录覆盖固定的FOMC、SEP与CPI，不是全部国内外事件清单。

## 完整路径保留了支持与不支持直觉的部分

![两个八月病例的全部路径](<{figure.as_posix()}>)

{table(['从起点快照至原20日退出时','2020年病例','2022年病例'], [
['美国2年期收益率变化',f'{w0.origin_to_exit_ust2_bp:+.0f}基点',f'{w2.origin_to_exit_ust2_bp:+.0f}基点'],
['美国10年期收益率变化',f'{w0.origin_to_exit_ust10_bp:+.0f}基点',f'{w2.origin_to_exit_ust10_bp:+.0f}基点'],
['美元兑人民币中间价变化',f'{w0.origin_to_exit_usdcny_percent:+.2f}%',f'{w2.origin_to_exit_usdcny_percent:+.2f}%'],
['中国10年期收益率变化',f'{w0.origin_to_exit_cgb10_bp:+.2f}基点',f'{w2.origin_to_exit_cgb10_bp:+.2f}基点'],
['原20日窗口',f'{w0.entry_date}至{w0.exit_date}',f'{w2.entry_date}至{w2.exit_date}'],
['510300原20日毛收益',f'{w0.E0_return_percent:+.2f}%',f'{w2.E0_return_percent:+.2f}%'],
['原方案延后一天入场的20日毛收益',f'{w0.E1_return_percent:+.2f}%',f'{w2.E1_return_percent:+.2f}%']])}

这是观察期间的共同变化，不是一个因素对另一个因素的贡献率。2020年中国10年期收益率也上升了，股票最终仍涨；2022年后段美国利率及人民币中间价压力增加，也不能据此认定所有跌幅都由它们造成。定价取决于利率变化的原因、现金流与风险补偿、之前已经反映多少以及之后新出现的信息。

下面把每段涨跌统一除以原入场本金。因此各段可以直接相加回原20日收益，分段数值是对原窗口收益的百分点贡献，不是每段自身独立复利收益。FOMC与SEP同一时刻公布的文件合并为一个节点，避免重复计入。

{table(['病例','分段交易日期','段首已加入的信息','交易日数','对原20日收益贡献'], [[r.stat_month, f'{r.start_date}至{r.end_date}',r.start_after_announcement,int(r.days),f'{r.contribution_pp:+.4f}个百分点'] for r in segments.itertuples()])}

2022年9月13日当日贡献为+0.1680个百分点，新CPI在当晚公布，之后9月14日至21日贡献−5.0888个百分点。这说明主要损失不在最初那个白天发生，但时间先后并不能把损失全部归因于美国CPI。9月22日至10月13日又包含利率决定、经济预测及中国下一期金融数据等变化。

反过来，2022年10月13日晚核心CPI同比继续升到6.6%，随后本窗口剩余两个交易日却贡献+2.0883个百分点。2020年10月CPI环比放慢，之后剩余四日贡献−1.7208个百分点。**这些反例需要保留：公布值的好坏不与下一段价格涨跌一一对应。** 缺少合格事前预期、全量同时发生的消息及成分股暴露，不能把消息表变成确定性的买卖表。

## 时钟假设与可以复核的边界

美国财政部日度CMT来自约美东15:30的指示性买价，经曲线插值形成，并非实际成交收益率。本轮按美国数据日23:59:59才可用的保守研究假设对齐；不是历史首发回执认证。美东夏令时下，该时刻对应次日北京时间11:59:59，因此中国早盘可能仍沿用更早记录。额外滞后一条美债记录后，从起点至退出的2年期与10年期变化分别为：2020年{w0.origin_to_exit_ust_delay1_2_bp:+.0f}、{w0.origin_to_exit_ust_delay1_10_bp:+.0f}基点；2022年{w2.origin_to_exit_ust_delay1_2_bp:+.0f}、{w2.origin_to_exit_ust_delay1_10_bp:+.0f}基点。两种对齐结果都保留，没有按股价走势选择时钟。

财政部在2021年12月6日改变了曲线插值方法，本轮保留两个年份各自方法，不把它们拼成新预测模型的连续同质样本。定义见[美国财政部日度利率说明](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve)。人民币序列使用外汇交易中心中间价及原09:15可用假设，已逐值对回现有JSON原件；它不能代替在岸即期收盘、离岸汇率、订单流或不可变历史首版。中国国债按数据日之后第一个A股开盘才可用。

本轮新增保存13份官方公告，提取63条数值记录；两个美国国债年份共500条原始记录，实际使用的140条人民币中间价及其20记录比较基准对回原件。两个病例共40个交易日的每日贡献、累计收益与原终值均直接用原市场价格及分红重新核算，误差低于十亿分之一。该复算确认的是数据对齐和会计恒等式，不是外部独立验证，也不是成本后账户表现。

## 结论与下一条需要补实的传导

目前可以明确排除“国内资金便宜、剪刀差改善且降波，就足以确认未来上涨”这种读法。更贴近两个病例的解释需要同时保留：信贷来自哪些部门和期限、累计与当月是否一致、国内与国外利率是否同向、人民币在怎样变化、股价与内部广度此前已反映多少，以及后续是否出现新信息。

现有证据仍没有直接量出哪类成分股通过现金流、外币资产负债、估值或风险补偿承担了外部变化。下一条可补实的历史机制，是将这些暴露与原有成分股的收益贡献对应起来，同时保留行业盈利和国内事件的竞争解释；先核对当时可得的企业证据，再讨论是否存在可复用的条件。当前不把两个病例升级为预测模型，不新增阈值或账户策略，目标仍未达成。

详细数据：{link(OUT / 'results/两个病例_起点入场与退出快照.csv', '起点入场退出快照')}；{link(OUT / 'results/官方消息_北京时间与信息阶段.csv', '全部公告时钟')}；{link(OUT / 'results/原40个交易日_完整路径及当时可见外部信息.csv', '40个交易日完整路径')}；{link(OUT / 'results/全部消息分段_同一入场本金贡献.csv', '全部分段贡献')}。冻结范围和计算代码随报告保存在同一目录。
"""
    report = OUT / "第十五轮_信贷改善与外部利率消息时序.md"
    report.write_text(body, encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(f"已生成报告：{report}")
    print(f"已生成图表：{figure}")


if __name__ == "__main__":
    main()
