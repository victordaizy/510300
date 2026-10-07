"""将第三轮证据整理为明确结论，不将历史关联升级成确定收益。"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_volatility_money_decomposition_v3"
OLD, NEW = "M1_OLD_M2_MMF2018", "M1_NEW2025"


def run() -> None:
    associated = pd.read_csv(STUDY / "results/波动与后续风险收益关联.csv")
    weekly = pd.read_csv(STUDY / "results/波动分解_固定周度_完整观察.csv")
    nonoverlap = pd.read_csv(STUDY / "results/不重叠20日窗口_关联复核.csv")
    distributions = pd.read_csv(STUDY / "results/全部波动来源与宏观背景分布.csv")
    source = associated[(associated.feature == "v_rv20") & (associated.entry == "E0")]
    selected_periods = [(OLD, "统计月2018-2021"), (OLD, "统计月2022-2024"), (NEW, "统计月2025-2026")]
    rows, vol, down, ret, disjoint = [], [], [], [], []
    for regime, period in selected_periods:
        p = source[(source.regime == regime) & (source.period == period)].set_index("outcome")
        v, d, r = p.loc["future_rv", "weighted_spearman"], p.loc["future_downside", "weighted_spearman"], p.loc["return", "weighted_spearman"]
        q = nonoverlap[(nonoverlap.regime == regime) & (nonoverlap.period == period) & (nonoverlap.outcome == "future_rv")].iloc[0]
        rows.append(f"|{period.removeprefix('统计月')}|{int(p.loc['future_rv','weeks'])} / {int(p.loc['future_rv','cycles'])}|{v:+.3f}|{d:+.3f}|{r:+.3f}|{q.spearman:+.3f}（{q.n_nonoverlapping_windows}窗）|")
        vol.append(v); down.append(d); ret.append(r); disjoint.append(q.spearman)
    table = "|统计月时期|周数 / 发布周期|与未来20日总波动|与未来20日下行尺度|与未来20日收益|去掉未来窗口重叠后，与总波动|\n|---|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(), "axes.unicode_minus": False, "font.size": 11, "figure.facecolor": "#fafaf7", "axes.facecolor": "#fafaf7", "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "#fafaf7"})
    fig, ax = plt.subplots(figsize=(11.8, 6))
    x = np.arange(3)
    ax.bar(x - 0.24, vol, 0.22, label="当前波动 → 未来总波动", color="#20796e")
    ax.bar(x, down, 0.22, label="当前波动 → 未来下行尺度", color="#82aba0")
    ax.bar(x + 0.24, ret, 0.22, label="当前波动 → 未来收益", color="#bc7b42")
    for shift, values in [(-0.24, vol), (0, down), (0.24, ret)]:
        for i, value in enumerate(values):
            ax.text(i + shift, value + (0.018 if value >= 0 else -0.04), f"{value:+.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, ["2018—2021\n48个发布周期", "2022—2024\n36个发布周期", "2025—2026新口径\n19个成熟发布周期"])
    ax.set_ylabel("按发布周期等权的秩相关")
    ax.set_ylim(-0.38, 0.59)
    ax.axhline(0, linewidth=0.8, color="#707b84")
    ax.grid(axis="y", color="#d7ded9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=10)
    fig.suptitle("风险大小的延续较稳定，收益方向会变", x=0.09, y=0.99, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.09, 0.02, "全部使用观察后下一开盘起20日固定份额路径。时期按宏观统计月份划分；相关不是上涨概率。\n不重叠窗口核对后的风险关联仍为+0.427、+0.269、+0.377；这是历史证据，不是独立前向样本。", fontsize=9, color="#525f68")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.21, top=0.79)
    fig.savefig(STUDY / "figures/风险惯性与方向不稳定.png", dpi=170)
    plt.close(fig)
    certainties = [
        {"claim": "当前波动大小与随后波动大小存在重复出现的正向历史关联", "status": "SUPPORTED_HISTORICAL_RISK_PERSISTENCE", "evidence": "3个既定时期和不重叠窗口方向均正；旧口径总体相关0.473，描述性区间[0.334,0.566]", "limit": "不是确定未来幅度或任何单次风险上限"},
        {"claim": "当前波动率高低能够稳定判断后20日涨跌", "status": "NOT_SUPPORTED_DIRECTION_CHANGES_ACROSS_PERIODS", "evidence": "三个时期收益相关为-0.150,+0.156,-0.259；旧口径总体0.030，区间跨零", "limit": "不能从未证实方向优势推导反向交易"},
        {"claim": "D20下降等于最近5天下跌幅度下降", "status": "FALSIFIED_BY_EXACT_WINDOW_IDENTITY", "evidence": "旧69/192、新15/46个D20下降周的近期5日下行平方和反而上升", "limit": "窗口错读被否定，不证明该状态后续一定更差"},
        {"claim": "剪刀差改善加上涨主导波动上升足以构成高胜率", "status": "NOT_SUPPORTED_IN_OLD_SAMPLE", "evidence": "旧25周17周期，20日按周期等权上涨比例48.04%；新10周6周期83.33%但样本很小", "limit": "历史状态分布，不是执行策略、独立事件胜率或未来概率"},
        {"claim": "已达到用户要求的510300确定方向结论", "status": "NOT_ACHIEVED_GOAL_ACTIVE", "evidence": "风险规律已有证据，方向优势和独立验证仍缺", "limit": "保持原目标，不把完成一轮观察当成总目标完成"},
    ]
    pd.DataFrame(certainties).to_csv(STUDY / "results/结论与证据等级.csv", index=False, encoding="utf-8-sig")
    examples = []
    for state in ["D20下降_近期5日反而增强", "D20下降_近期5日同步缓和"]:
        data = weekly[(weekly.downside_window_state == state) & weekly.stat_month.notna()].dropna(subset=["E0_20_return"])
        for selection, index in [("最差", data.E0_20_return.idxmin()), ("最好", data.E0_20_return.idxmax())]:
            row = data.loc[index].to_dict()
            row["case_selection"] = "事后说明_" + selection
            examples.append(row)
    pd.DataFrame(examples).to_csv(STUDY / "results/窗口效应解释与反例.csv", index=False, encoding="utf-8-sig", float_format="%.15g")
    report = f"""# 第三轮明确结论：风险有惯性，方向优势仍未成立

**这轮最明确的正面发现是510300风险大小的延续性。**当前波动率与未来20日波动率的正相关，在旧口径前后期、新口径短样本以及剔除未来窗口重叠后均出现。相同的当前波动率与未来收益的关系会变号。已有证据支持用波动理解风险环境，尚不支持用它稳定决定涨跌方向。

本轮按用户原目标继续，不把目标改成只证明一个容易成立的恒等式。已完成新分解和复核，但“510300确定方向/稳定入场优势”仍未达到，目标保持进行中。所有历史数据先前已见，以下均不冒充独立前向验证。

## 1. 波动的哪些内容已经相对清楚

{table}

这里的相关不是胜率。周度主统计按宏观发布周期等权，每周期内有效周等权，防止同一宏观信息被重复周数放大。三个时期沿用统计月分段，因此2024年12月公告后的部分2025年初交易日仍归旧口径末段。

旧口径全体357周、84个发布周期：当前RV20与后20日RV的相关为+0.473，6周期时间块的95%描述区间为[+0.334,+0.566]；与未来收益只有+0.030，区间[-0.118,+0.202]。与未来下行尺度为+0.339，但与最差途中路径仅−0.050，区间跨零。**风险振幅的惯性，不等于能精确预测最坏回撤。**

为核对相邻周20日结果重叠的影响，结果出来后另行登记一个只看日期的固定检查：每口径从首个成熟窗口开始，下一窗口入场必须晚于前一窗口退出，不尝试不同偏移。得到旧口径79窗、新口径17窗；风险关联仍为+0.436、+0.377。这是重叠敏感性核对，仍使用同一段历史，不能称为新独立样本。

![风险惯性与方向不稳定](figures/风险惯性与方向不稳定.png)

复算此前冻结的LOG_RISK_RIDGE风险预测账也得到一致分工：140个配对月中，按原固定风险损失ln(预测值)+实际风险/预测值，2015—2019和2020—2026两段都优于历史均值和最近下行风险两个基准。这里没有重新拟合。这个损失指标上的改善并不表示所有指标都改善；例如早期相对最近下行基准的风险MSE并未改善，原账户目标也未达到。

## 2. “下行波动下降”约三分之一并非近期下跌减弱

对20日窗口，下行平方尺度D²的5日变化，严格等于“新进入窗口的5日负收益平方和”减“刚移出窗口的5日负收益平方和”，再乘固定年化系数。它没有要求新5日比紧邻的上一组5日更小。

|口径|D20下降的全部观察周|其中最近5日下行平方和反而增大|占比|
|---|---:|---:|---:|
|旧|192|69|35.94%|
|新|46|15|32.61%|

因此，可以确定排除“D20下降就代表近期下跌持续缓和”这个读法。例：2021-02-19，旧大跌移出使D20下降，但最近5日负收益平方和已高于此前5日；之后20日E0收益−13.62%，最差途中收盘−13.93%。它是事后反例，不用于重选参数。

不过，进一步将两种下降状态比较，旧口径“近期反而增强”相对“同步缓和”的后20日收益差仅+0.30个百分点，描述区间[-0.72,+1.79]；最差路径差+0.09个百分点，区间[-0.37,+0.70]。**把指标读对了，还没有自动得到涨跌预测。**

## 3. 上涨推高波动，也不能直接命名“好波动”

这次没有把总波动简单等同上行加下行，而是严格保留样本方差中的均值修正。n=20时：RV²=20/19×[上行平方尺度+下行平方尺度−252×平均日收益²]。总波动上升的三项贡献都保存，均值项主导和并列分别保留。

在总波动上升周，上行项主导相对下行项主导：

|口径|两组有效发布周期|E0后20日收益差|最差途中路径差|E1收益差|
|---|---|---:|---:|---:|
|旧|55 / 56|−0.51个百分点|−0.72个百分点|−0.58个百分点|
|新|11 / 13|−2.04个百分点|−1.61个百分点|−2.08个百分点|

旧口径主比较区间均跨零；新口径样本少、周期有关联，不能因一个未校正多重比较的区间未跨零就升级为卖出规律。结果至少没有支持“由上涨造成的波动扩张，之后自然更安全/更容易续涨”。仍需区分已经完成的上涨与未来剩余空间。

## 4. 把M1/M2背景加回来，结论是否改变

没有只看波动平均数。全部波动组都按剪刀差三月改善、未改善和缺失分别保留，未选择最漂亮的交叉组。

“剪刀差改善+上行项主导波动上升”在旧口径是25周、17个发布周期，后20日平均毛收益+1.08%，但周期等权上涨比例仅48.04%，平均最差途中收盘路径−3.76%。平均为正主要说明收益分布的正负幅度不同，不能叫作高胜率确定节点。新口径10周、6周期对应+1.47%、83.33%，不足以把短样本比例当未来胜率。

月度原始关联和控制后的样本关联也分开检查。旧口径78个有效三月变化月，剪刀差变化与后20日收益原始秩相关+0.094；控制此前20/60日收益、logRV20、下行平方占比和统计月份后，线性残差秩相关+0.118。与未来总波动的原始+0.039在控制后为−0.028。剪刀差水平与未来总波动原始+0.295，控制后只剩+0.058。这些是样本内描述，不是新的预测成绩。

新口径三月变化只有16个成熟月；加入既定月份和价格/波动控制后没有剩余自由度，本轮保持不可估计，没有删掉控制项来制造漂亮结论。原始收益相关+0.582仍原样保留。

更直接的时间顺序证据来自原先冻结的M1/M2增量预测账：53个配对月中，价格与波动基准的MSE为0.005694，加入剪刀差后为0.005752，误差增加约1.02%；前后半段增量异号。此次逐值复算原预测和损失，没有重训或改门槛。它否定的是这次固定表示的可靠增量，不是宣称任何宏观信息都无用。

## 5. 当前可采用的研究判断与尚缺的部分

|判断|证据状态|对继续研究的影响|
|---|---|---|
|当前波动大小有助于理解随后风险大小|三个时期与不重叠核对重复出现正向关联|保留为风险环境信息，不赋予收益正负含义|
|D20下降能直接解释为近期下跌缓和|被精确窗口分解和实际反例否定|必须同时看新5日、上一组5日和移出5日|
|上涨主导波动扩张天然有后续优势|未得到支持|不能因为“涨出来的波动”就忽略后续回撤|
|剪刀差改善和好看波动状态组合成确定节点|旧样本与冻结增量检验未支持，新样本太短|后续只能检验具体原因、先前定价和传播过程，不反复组合阈值|

下一项仍有实质研究价值的问题是：**同一条宏观信息已公开且不变的期间，新的价格/波动变化是否只是对过去冲击的机械反映，还是包含当时可定位的新原因及尚未完成的价格反应。**需要在同一个信息周期内比较，并核对政策/需求/金融产品迁移的时间，避免用不同月份的市场差异代替机制。这项尚未完成，原目标继续保留。

研究边界：行情截至2026-09-11，445周、104月保留；新风险标签从E0/E1开盘起固定份额财富路径计算，第一天为开盘到收盘，随后为相邻财富点变动。收益为未扣费用的观察毛收益，非账户收益。官方历史公告仍有版本不可修订证明不足；新旧M1不拼接；2026-08公告之后没有新行情快照。没有产生仓位、夏普或订单。

## 方法和文件

上/下行平方和分解与半方差研究的方向一致，但本轮使用日线固定窗口，不冒充高频连续时间估计。[Shephard作者文献页](https://shephard.scholars.harvard.edu/publications/measuring-downside-risk-realised-semivariance)。跨尺度波动延续的相关研究见[Corsi作者文献页](https://people.unipi.it/fulvio_corsi/pubblicazioni/)；510300的上述结论来自本地复算，未从论文直接套用。

主要文件：[445周波动分解](results/波动分解_固定周度_完整观察.csv)、[104月波动与宏观](results/波动分解_104个月_完整观察.csv)、[全部宏观背景分组](results/全部波动来源与宏观背景分布.csv)、[两项固定比较](results/主20日_两项固定比较.csv)、[不重叠窗口核对](results/不重叠20日窗口_关联复核.csv)、[结论证据等级](results/结论与证据等级.csv)、[冻结协议](protocol.json)。

已复核方差和窗口恒等式、状态定义、48列原标签、全部120条风险关联、固定时间块抽样以及旧风险/宏观预测损失。多重比较未校正，区间只作描述。检查通过不等于总目标完成。
"""
    (STUDY / "第三轮明确结论.md").write_text(report, encoding="utf-8")
    readme = """# 阅读与复算

从第三轮明确结论.md开始。它将可确认的风险规律、可否定的指标读法和仍未证实的方向优势分开，目标保持进行中。

所有复算从 C:\\Users\\戴周阳\\Documents\\New project 8 项目根目录使用 .venv\\Scripts\\python.exe，依次执行 research\\volatility_money_decomposition_v3.py、research\\verify_volatility_money_v3.py、research\\publish_volatility_money_v3.py。

主脚本复用第一轮104月/445周和已保存行情；不联网下载。旧风险预测和M1/M2增量账只做保存值复算，不重新拟合。code目录是对应脚本的阅读副本，复算入口使用项目根目录research文件。

非重叠核对是在初步结果出来后针对窗口重叠问题登记，使用仅依赖日期的唯一贪心规则，没有搜索起始偏移。它不替代原主结果，也不提供新的独立历史样本。风险部分的“当前波动”与“未来波动”窗口没有同日收益重叠。

误差区间采用6个连续宏观发布周期的时间块、2000次、固定种子；原始抽样索引及次数矩阵均已保存。不同状态可能出现在同一发布周期，不能将组间周期数相加当独立样本。新口径不足24周期均有样本不足标识。

线性残差仅用于样本内条件相关；控制矩阵剩余自由度不足12时不计算，避免饱和拟合。按统计月的季节哑变量意味着分时期表使用统计月范围，不等同严格自然年行情区间。

报告图表已经检查后再交付；原始CSV保留小数与缺失状态。全目标尚未达到，任何本轮完成记录只指本轮事实计算和材料保存。
"""
    (STUDY / "00_阅读与复算.md").write_text(readme, encoding="utf-8")
    for name in ["volatility_money_decomposition_v3.py", "verify_volatility_money_v3.py", "publish_volatility_money_v3.py"]:
        shutil.copy2(ROOT / "research" / name, STUDY / "code" / name)
    progress = {"updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "goal_objective": "我需要更明显的结论，继续拆解波动率和m1,m2剪刀差，直到找到510300确定性的结论", "goal_status": "active", "goal_achieved": False, "turn_classification": "PROGRESS_NEW_VOLATILITY_IDENTITIES_AND_RISK_PERSISTENCE_EVIDENCE", "requirements": [{"item": "继续拆解波动率", "status": "本轮已完成上行/下行/均值项和窗口退出效应"}, {"item": "继续拆解M1/M2与波动关系", "status": "本轮完成宏观背景交叉分布和季节/价格/波动控制关联；原因未唯一识别"}, {"item": "更明确结论", "status": "已区分风险惯性、被否定的指标读法、未成立的方向优势"}, {"item": "找到510300确定方向或稳定优势", "status": "未证实，独立新证据不足；不能宣告完成"}], "next_bounded_action": "同一宏观发布周期内的价格/波动变化配对，分离新信息、已发生行情与窗口机械变化；先冻结，不调参救援。"}
    (STUDY / "goal_progress.json").write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table)
    print("第三轮明确结论、风险图与完整证据已保存，原目标保持active。")


if __name__ == "__main__":
    run()
