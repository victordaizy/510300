"""发布第二轮机制观察：可读结论、图表和有时钟的证据表。"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_macro_volatility_mechanisms_v2_run2"
GROUP_CURRENT = "改善_当期相对余额同向"
GROUP_BASE = "改善_当期相对余额未同向"
REGIMES = ["M1_OLD_M2_MMF2018", "M1_NEW2025"]


def csv(data: pd.DataFrame, name: str) -> None:
    data.to_csv(STUDY / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    return "|" + "|".join(headers) + "|\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("|" + "|".join(row) + "|" for row in rows)


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:+.{digits}f}%" if pd.notna(value) else "缺失"


def source_clock(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clocks = {"理财半年原报告": "2024-07-30", "手工补息整治官方转载": "2024-08-05", "央行三季度报告迁移解释": "2024-11-08", "M1修订答记者问官方转载": "2024-12-03"}
    for row in frame.to_dict("records"):
        published = pd.Timestamp(row["available_at_upper_bound"])
        known = [label for label, date in clocks.items() if pd.Timestamp(date + "T23:59:59+08:00") <= published]
        rows.append({"stat_month": row["stat_month"], "available_at_upper_bound": row["available_at_upper_bound"], "current_balance_arithmetic": "RECONSTRUCTED_RELEASE_WITH_VINTAGE_LIMIT", "official_context_public_by_this_release": ";".join(known) if known else "本轮未取得更早的对应机制资料", "context_is_monthly_cause": False, "annual_deposit_table_exists": "2023-01" <= row["stat_month"] <= "2025-12", "annual_table_historical_admission": "EXCLUDED_NO_ORIGINAL_MONTHLY_VINTAGE", "economic_cause": "未知/可混合", "cause_status": "NOT_IDENTIFIED", "meaning": "官方资料只确认制度或一般机制；没有把当前年度表和后出报告前移，也没有把制度存在等同逐月因果。"})
    return pd.DataFrame(rows)


def figures(frame: pd.DataFrame) -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": font_name, "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fafaf7", "axes.facecolor": "#fafaf7", "savefig.facecolor": "#fafaf7"})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=True)
    for ax, year in zip(axes, [2024, 2025], strict=True):
        part = frame[frame.stat_month.str.startswith(str(year))]
        x = np.arange(len(part))
        current = part.d3_relative_current_log_pp.to_numpy()
        base = part.d3_relative_base_revision_log_pp.to_numpy()
        total = part.d3_relative_growth_log_pp.to_numpy()
        ax.bar(x - 0.18, current, width=0.34, color="#1d756e", label="当期相对余额项")
        ax.bar(x + 0.18, base, width=0.34, color="#bb7840", label="隐含基数/修订项")
        ax.plot(x, total, "D-", color="#222d3c", markersize=4, linewidth=1.6, label="两项合计：同比对数差的三月变化")
        ax.axhline(0, color="#76808a", linewidth=0.8)
        ax.set_xticks(x, [month[-2:] + "月" for month in part.stat_month])
        ax.set_ylabel("对数百分点")
        ax.grid(axis="y", color="#d8dcd9", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_title(f"{year}年 · {'旧' if year == 2024 else '新'}M1口径", loc="left", fontweight="bold")
        if year == 2025:
            ax.text(0.03, 0.9, "1—3月跨口径，保持缺失", transform=ax.transAxes, color="#606a72", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False, fontsize=10)
    fig.suptitle("同比读数改善，不等于当期相对余额扩张", x=0.08, y=0.995, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.08, 0.018, "基数由每次公告余额和同比反推，包含舍入及修订影响；柱为相对M1/M2项，未作季节调整。\n统计月不是公告日。图中合计采用对数变换，与普通剪刀差百分点数值略有不同。", fontsize=9, color="#505c66")
    fig.subplots_adjust(top=0.83, bottom=0.13, left=0.08, right=0.98, hspace=0.38)
    fig.savefig(STUDY / "figures/当期余额与基数分解.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, regime in zip(axes, REGIMES, strict=True):
        for index, group in enumerate([GROUP_CURRENT, GROUP_BASE]):
            part = frame[(frame.training_regime == regime) & (frame.arithmetic_mechanism_group == group) & frame.E0_20_return.notna()]
            values = part.E0_20_return.to_numpy() * 100
            offsets = np.linspace(-0.13, 0.13, len(part)) if len(part) > 1 else np.zeros(len(part))
            ax.scatter(np.full(len(part), index) + offsets, values, s=39, alpha=0.75, color="#1d756e" if index == 0 else "#bb7840", edgecolors="white", linewidth=0.6)
            ax.hlines(np.mean(values), index - 0.2, index + 0.2, color="#1b2730", linewidth=2.5)
            ax.text(index, -14.4, f"n={len(part)}", ha="center", fontsize=10)
        ax.set_xticks([0, 1], ["当期相对余额同向", "当期相对余额未同向"], fontsize=10)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(-16, 14)
        ax.axhline(0, color="#76808a", linewidth=0.8)
        ax.set_ylabel("E0后20日毛收益 / %")
        ax.set_title("旧M1口径" if regime == REGIMES[0] else "新M1口径", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#d8dcd9", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("同为剪刀差改善，后续分布仍然重叠", x=0.08, y=0.99, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.08, 0.022, "每点为一个统计月，黑线为均值；横向错开仅便于阅读。旧口径季节构成失衡，新口径同向仅2个月。\n样本依赖且有窗口重叠，未调整季节或既有价格表现；图示为历史描述，不代表独立预测效果。", fontsize=9, color="#505c66")
    fig.subplots_adjust(top=0.82, bottom=0.19, left=0.08, right=0.98, wspace=0.24)
    fig.savefig(STUDY / "figures/改善月份后20日分布.png", dpi=170)
    plt.close(fig)


def run() -> None:
    (STUDY / "figures").mkdir(exist_ok=True)
    frame = pd.read_csv(STUDY / "results/104个月_机制分解与原冻结标签.csv")
    summary = pd.read_csv(STUDY / "results/全部机制分组_描述分布.csv")
    csv(source_clock(frame), "104个月_机制证据时钟.csv")
    selected = frame[frame.arithmetic_mechanism_group.isin([GROUP_CURRENT, GROUP_BASE])].copy()
    selected["calendar_month"] = selected.stat_month.str[-2:]
    selected["calendar_year"] = selected.stat_month.str[:4]
    for calendar in ["calendar_month", "calendar_year"]:
        counts = selected.groupby(["training_regime", "arithmetic_mechanism_group", calendar]).size().reset_index(name="months")
        csv(counts, f"改善分组_{calendar}_样本构成.csv")
    case_months = ["2019-12", "2021-01", "2024-06", "2024-11", "2025-06", "2025-07", "2025-08"]
    cases = frame[frame.stat_month.isin(case_months)].copy()
    cases["selection_role"] = "POST_RESULT_EXPLANATION_NOT_VALIDATION"
    csv(cases, "解释用例_含亏损与基数反例.csv")
    figures(frame)
    table_rows = []
    for regime in REGIMES:
        for group in [GROUP_CURRENT, GROUP_BASE]:
            a = summary[(summary.training_regime == regime) & (summary.group == group) & (summary.entry == "E0") & (summary.horizon == 20)].iloc[0]
            b = summary[(summary.training_regime == regime) & (summary.group == group) & (summary.entry == "E1") & (summary.horizon == 20)].iloc[0]
            table_rows.append(["旧" if regime == REGIMES[0] else "新", "同向" if group == GROUP_CURRENT else "未同向", str(a.mature_months), pct(a.return_mean), pct(a.return_median), pct(a.worst_path_min), pct(b.return_mean)])
    main_table = markdown(["M1口径", "当期相对余额", "成熟月数", "E0后20日均值", "中位数", "组内最差途中收盘路径", "E1后20日均值"], table_rows)
    auxiliary_rows = []
    for regime in REGIMES:
        for group in [GROUP_CURRENT, GROUP_BASE]:
            cells = ["旧" if regime == REGIMES[0] else "新", "同向" if group == GROUP_CURRENT else "未同向"]
            for horizon in [5, 20, 60]:
                v = summary[(summary.training_regime == regime) & (summary.group == group) & (summary.entry == "E0") & (summary.horizon == horizon)].iloc[0]
                cells.append(pct(v.return_mean))
            auxiliary_rows.append(cells)
    auxiliary = markdown(["口径", "当期相对余额", "后5日均值", "后20日均值", "后60日均值"], auxiliary_rows)
    report = f"""# 第二轮机制观察：同比改善中，多少来自当期余额？

**本轮找到了可复算的区别，但没有得到可直接入场的规律。**旧口径31个剪刀差改善月份中，17个伴随M1余额相对M2上升，14个没有；新口径9个改善月份中只有2个相对余额上升。旧组的后20日分布存在差别，但季节构成严重不平衡；新组仅2个月，且仍主要受隐含基数项推动。

本轮在首轮结果已经看过的基础上，先冻结这次分解和分组，再读取新分组结果。因此它是后续探索，不能叫作事前独立验证。行情仍冻结到2026-09-11；保留104个月、E0/E1和5/20/60日标签，不新增行情。

## 1. 这次多识别了什么

普通剪刀差是M1同比减M2同比。“当期相对余额同向”只表示三个月中M1余额增幅超过M2余额增幅，**不表示企业需求已改善，也不要求这项贡献超过基数项**。两侧分别下降时也可能相对改善；本轮保留各自绝对余额变化，避免只看相对数。

对公告余额M和公告同比g，反推与其相容的基数 B*=M/(1+g/100)。B*包含舍入和潜在修订，并非已认证的去年原始余额。计算恒等式：

Δ₃ln(1+g/100)=Δ₃ln(M)−Δ₃ln(B*)。

两侧相减后得到M1相对M2的当期余额项及隐含基数/修订项。本轮用对数百分点保存这两个可相加部分，普通剪刀差百分点另列，不混用。仅在普通和对数变化都为正时按当期相对余额项正负分组。3月分解有效95个月，另外9个月因历史不足或定义断点保持缺失；1月分解有效101个月。2022年12月数字人民币及2025年新M1边界沿用首轮严格口径。

![当期余额与基数分解](figures/当期余额与基数分解.png)

两个例子能说明区别：2021年1月普通剪刀差三月改善6.7个百分点，但相对当期余额项为−0.246、隐含基数项为+6.252对数百分点。公告后20日E0为−11.09%，途中最差−12.74%。2025年6月普通剪刀差改善1.7个百分点，相对当期项仍为−0.884、基数项+2.587；其后20日却上涨2.94%。因此“基数支持改善”本身也没有预设涨跌方向。

2025年7月、8月是新口径仅有的两个同向月，相对当期项分别只有+0.288、+0.223，基数项分别+2.934、+2.500对数百分点。**即使相对余额同向，改善也未必主要由当期余额贡献。**

## 2. 同样是剪刀差改善，后续分布怎样

{main_table}

E0为公告可得后的首个21:00快照之后下一交易日开盘；E1再推迟一交易日开盘。持有窗口包含入场日，收益复用首轮固定份额的分红权益近似。途中最差为入场开盘起的每日收盘路径并与零取较小值，不是盘中最低价、账户最大回撤或保证损失上限。上述均为未扣费用的观察毛收益。

旧口径两组E0均值相差约2.09个百分点，E1相差约2.15个百分点，方向一致。但存在三个不能跳过的事实：

- 未同向组14个月中10个月来自1—3月，同向组17个月中1—3月为0。它们并非处在相同季节条件下，不能把均值差全归给余额来源。
- 同向组17个月有7个月来自2020年，行情阶段集中；相邻公告及20日窗口也可能重叠。17或14不是独立实验次数。
- 同向组仍有−11.87%的最差途中收盘路径。2019年12月这例最后20日收益只有−0.17%，但中间跌幅很大；2024年11月同向改善后仍亏5.23%。这是“方向读对但路径危险/仍亏损”的明确反例。

新口径两个同向月就是2025年7、8月；公告前20日510300分别已涨4.77%、7.59%，前60日平均已涨14.03%。后20日分别为+6.40%和+0.39%，不是同一种剩余空间。两个月都来自同一段行情，无法证明独立于先前上涨的增量。

![改善月份后20日分布](figures/改善月份后20日分布.png)

辅助期限完整保留，不挑选最漂亮的期限：

{auxiliary}

旧口径同向组后5日均值仍为负；新口径两组后60日均值都为正。只用20日表讲一个稳定故事会遗漏这些差异。没有新增阈值、拟合、匹配或重抽样，不为这次探索追加貌似精确的显著性结论。

## 3. 2024年存款结构的确变化了，但还不能追踪同一笔钱

从央行全国人民币信贷收支表的当前快照重新拆分，单位为万亿元：

|期间|非金融企业活期变化|定期及其他变化|企业存款总额变化|
|---|---:|---:|---:|
|2024年二季度（3月末至6月末）|−3.1115|+1.4427|−1.6688|
|2024年三季度（6月末至9月末）|−2.0127|+1.3457|−0.6669|
|2024年全年|−3.7373|+3.3266|−0.4107|

活期余额下降不能全解释为企业总存款等额消失。活期与定期及其他的反向变化支持结构迁移候选，但上述是不同科目的净余额变化，**不能认定增加的定期全由减少的活期直接转入**，更不能把总额剩余下降自动归给理财、消费或还贷。[2024年官方表](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111817274927346.htm)、[2023年末基数](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111817163388840.htm)。

2025年全年，企业活期净增加1.5932万亿元，定期及其他增加0.6013万亿元。结构与2024年不同，仍不能直接命名为需求复苏。[2025年官方表](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2026/01/2026011516053141218.htm)。

2024年3月至6月，旧M1扣M0的单位活期余额近似减少2.57万亿元，与企业活期减少3.11万亿元并不相等：单位活期不只含非金融企业，且公告精度和年度表版本也不同。2025年新M1还包含个人活期与支付备付金，扣M0后不能再称企业或单位活期。新口径的2024年回溯值未前移填入2024年观察。

这三张年度表均在本轮取得，原始逐月发布时间和历史修订版本没有被认证，**全部标为事后结构资料，不进入历史决策变量**。

## 4. 理财迁移证据支持到哪里

理财登记托管中心原报告给出的2024年6月末规模为28.52万亿元，较年初增6.43%；上半年33.68万亿元募集金额包含开放式产品各周期累计申购，不能当作净流入。固定收益类产品占96.88%，权益类产品占0.25%；这是产品分类，不是底层股票仓位，更不能由此推算流入510300的金额。机构投资者1.26%是人数占比，也不是资金占比。[发行者原报告公开媒体副本，印刷页8、11、18](https://att.dahecube.com/f/240730/2c8d5fcda96e49f57648f5a66c4e951b)。

该原报告已在线读取相关页；本地下载两次超时后停止，没有声称保存PDF本体或取得其哈希。报告封面到2024年7月，7月30日新闻能辅助确认它当日已公开，不把这一日期伪装成发行者原始秒级发布回执。

2024年8月5日官方转载确认手工补息整治已实施；11月8日央行三季度执行报告专栏4解释，存款流向资管、非银资产负债表调整以及证券客户保证金回流，都可能改变M2。这支持“货币也会响应金融投资”的机制，不能唯一量化每月原因。后出的11月解释不进入4—9月历史输入。[8月政策背景](https://jrj.sh.gov.cn/ZXYW178/20240805/e7a428bfdf2a4c0bb3e05437a5998e3f.html)、[11月执行报告](https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/5347949/afbfa5df25ee45889d916a2819b60a43/2024110815410752868.pdf)。已保存央行PDF，并核对专栏印刷页28—29的完整原页。

## 5. 对510300留下的可验证预期

|机制候选|对下一段观察的具体要求|什么会使解释失效|
|---|---|---|
|同比基数主导改善|继续分别记当期M1、M2余额与同比，观察基数项消退后是否仍有同口径当期支持|余额仍弱却只因同比变好就命名经营恢复|
|活期与定期及其他结构变化|用同期可得分项和经营资料核对；涉及银行时另看贷款收益、存款成本及期限|只有总量净余额，没有流向证据，却断言资金流入股票或全部退出实体|
|经营与支付改善|需要同一信息截止日的单位活期、订单/回款/贷款等互证，并检查510300此前已涨多少、参与是否扩散|仅现金或金融投资账户变化，或上涨贡献与所称经营链条无关|
|金融投资反馈|比较政策/价格变化与存款读数先后；将宏观作为可能同步或滞后的环境信息|把股市上涨后形成的保证金增加反过来解释为事前预测|

已有波动与内部参与字段继续随104个月保留，不新增交叉切格。以新口径仅有的两个同向月为例，2025年7、8月的20日成员上涨比例代理约75.5%、70.6%，但下行尺度5日变化均未下降；它们不能同时充当“下行已缓和”的支持证据。该代理沿用首轮历史成员与滞后一日处理，缺失保持缺失；尚无全历史行业权重贡献。

104个月唯一经济原因仍保持“未知/可混合”。本轮进展是把算术、存款结构和一般制度机制分开证明，而不是把无法识别的因果填满。旧轮未通过的条件比较保持原结论，新轮收益差只列为探索线索。进入进一步验证前，必须先解决季节可比、同口径新样本和历史分项资料可得性；缺少这些，就停留在环境观察。

## 文件与复算

- [104个月分解及原标签](results/104个月_机制分解与原冻结标签.csv)：可逐月检查M0、M1、M2、当期项、基数项、先前价格与后续路径。
- [不含未来标签的分解底表](results/104个月_余额基数分解_不含未来标签.csv)：先生成再连接原标签，留有摘要记录。
- [全国存款结构36个月](results/全国存款结构_2023至2025_事后快照.csv)与[固定季度/年度差分](results/全国存款结构_固定季度与年度变化.csv)。
- [机制证据时钟](results/104个月_机制证据时钟.csv)、[全部分组分布](results/全部机制分组_描述分布.csv)、[解释与亏损用例](results/解释用例_含亏损与基数反例.csv)。
- [冻结方案](protocol.json)、[来源回执](sources/source_receipts.json)、[复算说明](00_阅读与复算.md)。

已核对104份源HTML身份，提取104个M0，复核余额/同比恒等式，48列原E0/E1标签逐值复用，36个月存款分项和总额只相差原表的亿元小数舍入。检查证明本轮实现一致，不证明历史不可修订版本或策略有效性。账户收益、仓位、夏普均未计算；未制作ZIP。
"""
    (STUDY / "第二轮机制观察结论.md").write_text(report, encoding="utf-8")
    readme = """# 阅读与复算

先读第二轮机制观察结论.md。图、原文、CSV直接保存在此目录；本轮没有ZIP，未对外发送。

复算入口均在项目根目录的research目录。项目根目录是 C:\\Users\\戴周阳\\Documents\\New project 8；使用该项目.venv下的Python，依次执行：

1. research\\macro_volatility_mechanisms_v2_run2.py：从首轮104份原文和已保存结果生成余额分解，再复用原标签。
2. research\\analyze_deposit_structure_v2_run2.py：只读已保存的官方年度表。
3. research\\publish_macro_mechanisms_v2_run2.py：生成可读结论、图和证据时钟。

research\\collect_macro_mechanism_sources_v2_run2.py只是定向资料采集的实现留档；日常复算不需要运行网络采集。理财CDN已失败两次并明确停止，该报告只留下在线阅读事实及地址。其他七份新来源已本地保存。

code目录存放以上脚本的本轮副本用于查看。入口以项目根目录research脚本为准；本目录不是独立环境安装包，不声称完整离线便携。

原始行情和48列E0/E1标签来自首轮，截止2026-09-11；2026年8月公告之后没有行情观察快照。年度存款表是2026-09-29获取的当前版本，只供事后解释。B*=M/(1+g)是隐含基数，不能当作同版本真实历史余额。E0/E1收益为观察毛收益，未做费用、整手、T+1、资金占用和盘中成交模拟。

主比较只看同口径三月剪刀差改善月份，当期M1/M2余额比率三月变化大于0为同向，否则未同向。未改善、口径不足及可能的普通/对数方向不一致均保留。零容差1e-10；未尝试其他阈值。新研究是在已看过首轮结果之后冻结，所有分组仍为探索性描述。

北京时间与原始发布日、统计月份分别保留。任何官方一般机制资料已公开，只表示其可作背景阅读，不表示它认证了每个月的唯一原因。
"""
    (STUDY / "00_阅读与复算.md").write_text(readme, encoding="utf-8")
    for name in ["macro_volatility_mechanisms_v2_run2.py", "collect_macro_mechanism_sources_v2_run2.py", "analyze_deposit_structure_v2_run2.py", "publish_macro_mechanisms_v2_run2.py"]:
        shutil.copy2(ROOT / "research" / name, STUDY / "code" / name)
    print("第二轮结论、2幅研究图、机制时钟与样本构成已保存。")
    print(main_table)


if __name__ == "__main__":
    run()
