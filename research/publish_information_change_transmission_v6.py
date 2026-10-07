"""形成第六轮可阅读结论与独立图，区分原因、预期和后续新增信息。"""
from pathlib import Path
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.information_change_transmission_v6 import OUT, save, sha, now


def location(path):
    return "C:/Users/戴周阳/Documents/New project 8/" + Path(path).relative_to(ROOT).as_posix()


def link(path, label):
    return f"[{label}](<{location(path)}>)"


def show_number(value, digits=2, signed=False):
    if pd.isna(value):
        return "缺失"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def make_figure():
    frame = pd.read_csv(OUT / "results/原E0二十日_全部逐日贡献与波动变化.csv")
    f = frame[frame.stat_month.eq("2024-08")].copy()
    f["date"] = pd.to_datetime(f.date)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(15.6, 10.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.1]})
    fig.patch.set_facecolor("#f7f8fa")
    fig.subplots_adjust(left=.082, right=.94, top=.825, bottom=.16, hspace=.17)
    fig.text(.082, .95, "同一份货币数据，后续经历了不同的信息与波动阶段", fontsize=21, weight="bold", color="#182f46")
    fig.text(.082, .91, "2024年8月病例｜9月13日公布 M1：−7.3%，M2：6.3%｜原E0窗口：9月18日开盘至10月22日收盘", fontsize=11.5, color="#526372")
    fig.text(.082, .877, "价格路径、政策公布、下一次月度数据与波动构成按时间对齐；图中先后关系不能直接量化因果。", fontsize=11, color="#526372")
    policy_day = pd.Timestamp("2024-09-24")
    next_release = pd.Timestamp("2024-10-14 17:00:50")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#e2e7ed", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#bbc7d0")
        ax.axvline(policy_day, color="#8b5b98", linestyle="--", linewidth=1.3, alpha=.85)
        ax.axvline(next_release, color="#778a9a", linestyle=":", linewidth=1.3)
    ax = axes[0]
    ax.plot(f.date, f.path_return * 100, color="#1e5676", linewidth=2.7, marker="o", markersize=3.3)
    ax.fill_between(f.date, 0, f.path_return * 100, color="#c4dfe9", alpha=.3)
    ax.axhline(0, color="#aab5be", linewidth=.8)
    ax.set_ylabel("相对原入场金额的收益（%）")
    ax.set_ylim(-2, 44)
    ax.text(pd.Timestamp("2024-09-25"), 39, "9月24日：降准降息及资本市场工具等政策发布", color="#785085", fontsize=10.7)
    ax.text(pd.Timestamp("2024-10-15"), 34, "10月14日收盘后\n9月货币数据公布", color="#657889", fontsize=10)
    pre = f.set_index("date").loc[pd.Timestamp("2024-09-23")]
    final = f.iloc[-1]
    ax.annotate(f"至9月23日：{pre.path_return*100:.2f}%", xy=(pd.Timestamp("2024-09-23"), pre.path_return * 100), xytext=(pd.Timestamp("2024-09-18"), 12), arrowprops={"arrowstyle": "-", "color": "#526372"}, fontsize=10.5, color="#233e52")
    ax.annotate(f"原20日：{final.path_return*100:.2f}%", xy=(final.date, final.path_return * 100), xytext=(pd.Timestamp("2024-10-16"), 13), arrowprops={"arrowstyle": "-", "color": "#526372"}, fontsize=10.5, color="#233e52")
    ax = axes[1]
    for col, label, color in [("v_rv20", "20日总波动 RV20", "#74828e"), ("v_upside20", "上行波动项 U20", "#0b897f"), ("v_downside20", "下行波动项 D20", "#bf5c3e")]:
        ax.plot(f.date, f[col] * 100, label=label, color=color, linewidth=2.4, marker="o", markersize=3)
    ax.set_ylabel("年化波动构成（%）")
    ax.set_ylim(0, 72)
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=10.5)
    by_date = f.set_index("date")
    for date, text_date, text_y in [("2024-09-23", "2024-09-18", 22), ("2024-09-30", "2024-09-28", 19), ("2024-10-09", "2024-10-10", 21)]:
        row = by_date.loc[pd.Timestamp(date)]
        ax.annotate(f"{pd.Timestamp(date):%m/%d} 下行项 {row.v_downside20*100:.2f}%", xy=(pd.Timestamp(date), row.v_downside20 * 100), xytext=(pd.Timestamp(text_date), text_y), arrowprops={"arrowstyle": "-", "color": "#bf5c3e"}, fontsize=10, color="#a34a31")
    ticks = pd.to_datetime(["2024-09-18", "2024-09-23", "2024-09-24", "2024-09-30", "2024-10-08", "2024-10-09", "2024-10-14", "2024-10-22"])
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", rotation=40)
    ax.set_xlim(pd.Timestamp("2024-09-17"), pd.Timestamp("2024-10-24"))
    fig.text(.082, .085, "9月23日、9月30日、10月9日收盘时，最近公布的仍是同一份8月M1/M2数据，波动结构却已经不同。", fontsize=10.7, color="#233e52")
    fig.text(.082, .052, "U20、D20为正负日收益平方项开方，不能直接相加为RV20；收益按固定股数计入持有期分红，未扣交易成本。", fontsize=9.8, color="#657889")
    fig.text(.082, .027, "图示为已知历史的解释；政策目录不完整，日线无法分离盘中消息与价格；未生成新策略。", fontsize=9.8, color="#657889")
    dest = OUT / "figures/2024年8月_新信息与波动构成的时间路径.png"
    dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return dest


def run():
    checked = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert checked["status"] == "PASS_SAVED_JOIN_AND_PATH_RECOMPUTATION"
    assert checked["daily_publication_clock_rows_verified"] == 2060
    cases = pd.read_csv(OUT / "results/五个固定病例_新信息与传导全表.csv").set_index("stat_month")
    clocks = pd.read_csv(OUT / "results/五病例_原五日与原二十日时钟并列.csv").set_index("stat_month")
    credit = pd.read_csv(OUT / "results/三例同一调查_贷款社融预期与实际.csv")
    splits = pd.read_csv(OUT / "results/原E0二十日_已记录后续政策与路径分段.csv").set_index("stat_month")
    split = splits.loc["2024-08"]
    stage = pd.read_csv(OUT / "results/2024年8月病例_图示三日波动构成与已知货币数据.csv")
    comparison = json.loads((OUT / "results/2025年8月_三种比较基准不可混用.json").read_text(encoding="utf-8"))
    info = json.loads((OUT / "results/information_diagnostics.json").read_text(encoding="utf-8"))
    verdict = json.loads((OUT / "inputs/prior_verdict.json").read_text(encoding="utf-8"))
    monthly_clock = json.loads((OUT / "results/daily_information_clock_receipt.json").read_text(encoding="utf-8"))
    figure = make_figure()
    case_lines = []
    for month, row in cases.iterrows():
        observed = str(row.snapshot_at)[:10]
        case_lines.append(f"| {month}／{observed} | {row.delta3_spread_pp:+.1f} | {show_number(row.expectation_spread_surprise_proxy_pp, 1, True)} | {row.past_return60*100:+.2f}% | {row.valuation_original_pe_official:.2f} | {row.E0_20_return*100:+.2f}% |")
    credit_lines = []
    for row in credit.itertuples():
        visible = "原观察时已公开的来源" if row.visible_using_this_actual_source_at_original_snapshot else "本轮实际值为后来官方转载，不能回填原时点"
        credit_lines.append(f"| {row.stat_month} | {row.metric}（{row.period}） | {row.forecast_yi:,.0f} | {row.actual_yi:,.0f} | {row.actual_minus_forecast_yi:+,.0f} | {row.display_sign_description}；{visible} |")
    stage_lines = []
    for row in stage.itertuples():
        stage_lines.append(f"| {row.date} | {row.latest_published_stat_month} | {row.v_rv20*100:.2f}% | {row.v_upside20*100:.2f}% | {row.v_downside20*100:.2f}% |")
    info_lines = []
    for row in info:
        name = "旧M1口径" if "OLD" in row["regime"] else "新M1口径"
        info_lines.append(f"| {name} | {row['both_available']} | {row['improved_over_three_months']} | {row['improved_but_below_expectation']} | {row['improved_and_matches_expectation']} |")
    m2025 = cases.loc["2025-08"]
    report = f"""**第六轮：把预期、信用来源和后续信息放回同一条传导链**

本轮已经把104个月、445周的原因背景与事前调查连接起来，并把103个原20日窗口展开为2060个交易日。得到的明确结论是：**剪刀差改善不等于实体融资同步改善，也不等于比预期更好；总波动升降不直接给出方向；同一份货币数据公布后的20日收益，还会混入后来的政策和下一次月度数据。** 这些是关于信息含义与传导过程的结论。510300持续、可验证的未来方向优势仍未证明，长期目标保持进行中。

这里的观察顺序是：货币余额与基数如何变化 → 贷款和社融由谁推动 → 订单、收入、利润和现金流是否承接 → 公布前的价格、估值和市场广度已经走到哪里 → 本次公布比事前调查增加了什么信息 → 后来是否出现新的驱动及不同的上行、下行波动。每一环保留自己的统计期间和公布时点，不把这些量压成一分数。

**2025年8月病例：货币改善、财政融资、企业和住户融资与股价重估并没有同步。**

这批数据在2025年9月12日公开。按公布精度，M1同比6.0%，与9月11日事前报告记录的6.0%相同；M2同比8.8%，调查记录8.7%。剪刀差较三个月前改善2.8个百分点，但相对这份调查的剪刀差代理只差−0.1个百分点。四个展示到0.1个百分点的数相减，保守展示精度边界为±0.2个百分点，因此不把−0.1称为显著负面意外。

沿用{link(ROOT/'reports/research/510300_macro_volatility_mechanisms_v2_run2/第二轮机制观察结论.md', '第二轮同口径余额的回顾分解')}，这一期剪刀差三个月变化的对数分解为：当期相对余额项约+0.223、去年同期基数项约+2.500对数百分点。基数项占主导，不能把同比剪刀差的改善全部理解成当前交易性资金加速。这两个对数项也不能直接与上面的普通百分点2.8相加。

同一份报告同一页还提供了新增贷款和社融的调查值。2025年前8个月人民币贷款新增，预期135,710亿元、实际134,600亿元，低1,110亿元；社融预期264,907亿元、实际265,600亿元，高693亿元。两者方向相反，已经不适合再用一个剪刀差数字概括“信用全面变好”。它们是本报告中的调查代理，也不保证代表临公布前最后一刻的全部市场预期。

社融总量的同比多增46,600亿元中，政府债同比多增46,300亿元，占{comparison['government_share_of_tsf_yoy_increase']*100:.2f}%；对实体经济发放的人民币贷款则同比少增4,851亿元。这个比例的分母是**累计同比多增**，不是社融总量，也不是高于预期的693亿元。缺少政府债分项的同口径事前预期，无法认定社融超预期的693亿元由政府债贡献多少。[人民银行2025年8月社融增量报告](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/523c260b344c4f1390664430295064a9/index.html)支持这里的实际量和同比结构；事前预期见{link(OUT/'sources/Bls250911.pdf', '9月11日调查原页，第15页')}。

向下看融资结构与经营承接：企业中长期贷款前8个月同比少增13,200亿元，住户中长期少增2,300亿元，票据融资多增1,181亿元；当时最新制造业新订单指数49.5。向上看沪深300经营与定价：V5回顾版本的非金融同组公司TTM利润同比−2.56%、经营现金流净额+12.87%，经营净现金流改善不能等同为利润扩张，也尚未区分回款、存货、预收款及付款节奏的贡献；非银TTM利润同比+62.96%，其中约79.83%的利润增量来自更早的2024年下半年与2023年下半年之差，最新半年利润实际同比约+17.66%。这些财务数有后来修订风险，不能作为严格历史预测输入；企业贷款也不能直接视为民营企业融资或已用于扩产。

公布前的价格也已经变化。ETF前60日涨幅{m2025.past_return60*100:.2f}%；在V5能够对齐的沪深300指数与官方PE日期中，前20日价格的对数变化{m2025.price_bridge_20_log_price_pp:+.2f}个百分点，可以恒等拆成PE变化{m2025.price_bridge_20_log_pe_pp:+.2f}与隐含P/PE分母变化{m2025.price_bridge_20_log_implied_denominator_pp:+.2f}。这是重估与隐含分母的算术桥，不是对真实EPS或上涨因果的精确分摊。

因此，这一病例呈现的是**基数项推动同比改善、政府融资增加、经营净现金流改善和股价重估并存，但企业和住户新增中长期贷款同比少增，非金融利润尚未转为同比增长**。可以排除“看到剪刀差改善，就认定企业借钱扩张、利润兑现并继续推升价格”这一跳步；还不能识别每一笔融资如何进入M1，或从这个组合确定下一阶段涨跌。原20日收益为{m2025.E0_20_return*100:+.2f}%，不能拿这个事后结果反向制造阈值。

**五个固定病例仍全部保留，既看改善也看承接和预先定价。**

| 统计月／实际观察日 | 剪刀差三个月变化，百分点 | 本次相对调查差，百分点 | 公布前ETF60日收益 | 当时官方PE | 原E0后20日收益 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(case_lines)}

2020年6月、2025年7月缺少合格配对预期，仍为缺失；不能拿“没有预期记录”当成“没有意外”。2021年1月和2024年8月使用旧M1口径，2025年8月使用新口径，表格只并列病例，不合并拟合。统计月与可观察月份往往相差一个月。

同一事前报告中的贷款与社融补充，单位统一为亿元。2021年1月的当月与年内累计相同，另两例均比较1至8月累计，绝不以累计值减去当月预期。

| 统计月 | 指标及期间 | 调查记录 | 实际公布 | 实际减调查 | 展示精度及来源时钟 |
|---|---|---:|---:|---:|---|
{chr(10).join(credit_lines)}

2024年社融实际值在转载中展示为21.9万亿元，保守取±500亿元舍入边界，所以+125亿元不足以认定明确正向差异。2021年和2024年本轮追加的社融实际量采用较晚的官方转载，只能核实历史数值；分别来自[国家发改委2月22日转载](https://www.ndrc.gov.cn/fggz/fgzh/gnjjjc/hbjr/202102/t20210222_1267615.html)与[上海金融管理部门9月14日转载](https://jrj.sh.gov.cn/SCGK194/20240914/3184186167db42939ec38aa8500a9953.html)，没有倒填到2月9日、9月13日的原观察时点。后者本轮网页可读而直接下载403，状态已记录。两份调查原页分别为{link(OUT/'sources/Bls210208.pdf', '2021年2月8日报告，第12页')}、{link(OUT/'sources/Bls240912.pdf', '2024年9月12日报告，第13页')}。

2021年1月病例尤其说明传导不能停在“利好”。剪刀差相对调查高5.0个百分点，企业和住户中长期贷款同比多增，订单指数52.3；同时ETF前60日已涨15.95%、距200日均线约21.80%、官方PE18.32，20日广度约40.33%。首次完整交易日反应为+2.02%，原五日研究在随后开盘进入的五日收益为−8.37%，原E0二十日为−11.09%。这是“较好数据、短暂正向反应、随后负收益”的已知历史，不能凭单个病例证明某个高估值门槛，也不能把前两项机械延伸成第三项。

**2024年8月病例：把后来的政策与波动变化放回发生的日期。**

起点可知的是弱货币与信用、订单48.9、公布前60日ETF下跌8.19%、官方PE10.91、广度19.13%。剪刀差比事前调查低0.7个百分点。但起点以后出现了9月24日的新政策，随后还有9月27日利率落地、10月12日财政发布、10月18日工具推进等。[9月24日联合发布会原文](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)是政策内容的来源。这里保留来源显示时间及其上界，不把今天的网页时间戳说成已经认证的历史首版。

原E0从9月18日开盘持有至10月22日收盘，收益{split.original_20d_return*100:.2f}%。第一批已记录后续政策在9月24日，目录中同一时钟包含降准降息与住房资金支持两个节点，不能只挑一个命名为全部变化的原因。按该日以前最近收盘9月23日分割，同一原入场金额的路径是：**{split.contribution_before_node*100:.2f}个百分点 + {split.contribution_after_boundary_to_exit*100:.2f}个百分点 = {split.original_20d_return*100:.2f}%**。若从9月23日收盘权益重新计算后段收益，则为{split.after_boundary_equity_return*100:.2f}%，与相对原入场金额的贡献不是同一分母。

这只能说明大部分价格变化发生在新政策发布以后，不能把{split.contribution_after_boundary_to_exit*100:.2f}个百分点全部归因于政策。日线不能分离盘中消息、政策预期形成和同时发生的其他变化。目录24个节点也不是全部消息，空记录不代表无新信息。

更直接对应波动的证据是下面三个历史日期：最近公开的始终是同一份8月M1/M2，波动结构却明显不同。9月23日至9月30日，下行项从9.28%略降至8.80%，总波动却从11.24%上升到44.82%，上升主要伴随上行项放大；10月9日下行项已经升至31.98%。**在这一条路径内，“波动上升”先对应大幅上涨，后来又包含明显下跌冲击，不能固定译为看多或看空。**

| 收盘日期 | 最近已公开的统计月 | 总波动RV20 | 上行项U20 | 下行项D20 |
|---|---|---:|---:|---:|
{chr(10).join(stage_lines)}

U20、D20是正、负日收益平方贡献的年化开方，与带均值修正的总波动不是可直接相加的三项。三个图示日是在看到历史路径后选取，用于解释，不构成新的预测样本筛选。所有20日的完整数据同时保留；9月货币数据直到10月14日收盘后才公开，已经纳入后续每日信息记录。

![2024年8月病例的信息与波动时间路径](<{location(figure)}>)

**全样本能确认的是“变化”与“新信息”必须分开，尚未确认预测优势。**

在三个月变化与合格预期同时可用的月中，以下只是公布数值符号的描述，数值零容差1e−10用于处理浮点数，不是经济显著性或展示精度检验，也不是胜率。

| 口径 | 两者可用月数 | 三个月改善月数 | 改善但名义上低于调查 | 改善且与调查相同 |
|---|---:|---:|---:|---:|
{chr(10).join(info_lines)}

原有货币预期差五日研究的结果保持不变：旧口径24次评价中，加入预期差后的均方误差较原基线高{-verdict[0]['relative_MSE_improvement']*100:.2f}%，前后两半都没有改善，状态REJECTED_FROZEN_NO_RELIABLE_MESSAGE_INCREMENT；新口径只有16个成熟独立事件，未过原60个样本门，状态NOT_RUN_SAMPLE_GATE。本轮连接原因背景和逐日路径没有重拟合模型，不能将这次更完整的解释冒充新策略验证，更不能挽救旧失败。

原五日试验先观察公布后的首个完整交易日，再从下一交易日开盘计算五日；本主线原E0二十日按原21时快照后下一开盘计，两者时钟不同。例如2021年E0从2月10日进入，原五日试验在春节后2月18日进入。所有收益均为固定股数、含应得分红的毛收益，没有成本后全账户结果。

**交付与核对。**

已核对549个起点原列不变，其中{checked['preserved_nonempty_outcome_cells']:,}个非空后续标签单元格不变；103个原20日窗口、2060个逐日金额恒等式与源行情复算一致；2060个逐日已公开货币状态按公布时钟重新核对；6条贷款社融预期与PDF原页核对。原两段完整28个月保留；77个月已有配对预期中76个月落在本轮市场截止之前，2026年8月数据公布晚于冻结行情截止仍不进入可用输入。

全体原窗口中，{monthly_clock['original_windows_with_later_monthly_release']}个还包含起点之后公开的其他月度货币数据；这些数据和政策仅进入后续解释表，不能回到起点输入。去除后续价格标签的表也仍含后来修订风险的财务数据，不能称为严格历史可得模型底表。验证通过表示连接、时钟和算术一致，没有建立全部来源的历史首版身份，也不是独立研究验证。

- {link(OUT/'results/104个月_原因盈利定价与预期差完整连接.csv', '104个月完整连接表')}；{link(OUT/'results/445周_原因盈利定价与预期差完整连接.csv', '445周完整连接表')}。
- {link(OUT/'results/两个固定历史段_完整28个月_新信息与传导.csv', '两段完整28个月')}；{link(OUT/'results/三例同一调查_贷款社融预期与实际.csv', '同一调查的贷款与社融预期及实际量')}。
- {link(OUT/'results/原E0二十日_全部逐日贡献与波动变化.csv', '全部原20日逐日价格、分红及波动路径')}；{link(OUT/'results/原E0二十日_每日收盘已经公开的货币状态.csv', '每日当时已公开的货币状态')}。
- {link(OUT/'verification.json', '计算核对记录')}；{link(OUT/'protocol.json', '本轮事先记录的连接规则')}；{link(OUT/'补充范围_同一消息包的信贷与社融预期.json', '读过结果后追加贷款社融核对的范围说明')}。

下一步尚待解决的是：在已知起点的融资主体、经营承接和预先定价下，哪些新的驱动有证据改变后续利润或风险溢价，变化能否在进入前识别并在独立的后续样本中成立。不能继续把“政府债同比多增”“M1活化”“现金流改善”和“股价上涨”之间尚未识别的箭头当成既定因果。缺少资金去向或分项预期时应明确保留未知；也不能靠新增高低位或牛熊阈值，把这几个已知病例拟合成确定性结论。
"""
    target = OUT / "第六轮_预期信用来源与信息时序.md"
    target.write_text(report, encoding="utf-8")
    for name in ["information_change_transmission_v6.py", "extend_credit_expectations_v6.py", "extend_daily_information_clock_v6.py", "verify_information_change_transmission_v6.py", Path(__file__).name]:
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    save("goal_progress.json", {"at": now(), "goal_status": "active", "turn_classification": "PROGRESS", "completed_this_round": ["549起点连接事前调查与既有原因背景", "同一调查追加三例贷款社融预期并保留真实来源时钟", "103个原20日窗口逐日收益和波动分解", "后续政策和其他月度公布按各自时钟归位", "形成混合信用信息与不同波动阶段的明确历史结论"], "not_achieved": "尚未建立510300独立验证的未来方向或成本后账户优势；不能承诺确定性上涨。", "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False})
    save("completion_receipt.json", {"at": now(), "round_status": "RESEARCH_DELIVERABLE_COMPLETE_GOAL_STILL_ACTIVE", "report": target.name, "report_sha256": sha(target), "figure": str(figure.relative_to(OUT)), "figure_sha256": sha(figure), "verification_sha256": sha(OUT / "verification.json"), "round_achieved": True, "goal_achieved": False})
    print("第六轮报告与时间路径图已生成。研究目标继续，未生成新模型、账户结果或交易指令。")
    print(location(target))
    print(location(figure))


if __name__ == "__main__":
    run()
