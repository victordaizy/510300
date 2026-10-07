"""汇总两轮真实训练和固定节点试验，保留未达标事实及完整年度记录。"""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter


ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"reports/research/510300_dense_probability_payoff_nodes_v1"
B=ROOT/"reports/research/510300_node_distribution_calibration_v1"
OUT=ROOT/"reports/research/510300_probability_payoff_progress_20260924"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value.strip()+"\n",encoding="utf-8")


def plot(accounts,annual):
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":10})
    fig,(top,bottom)=plt.subplots(2,1,figsize=(11,8),gridspec_kw={"height_ratios":[1.35,1]})
    colors=["#2a6f97","#d17a22","#52796f"]
    for (label,frame),color in zip(accounts,colors):
        top.plot(pd.to_datetime(frame.date),frame.equity_cny/200000,label=label,color=color,lw=1.9)
    top.set_title("510300：完整账户表现与逐年交易次数",fontproperties=font,loc="left",fontsize=16,pad=15)
    top.set_ylabel("净值（起始20万元）",fontproperties=font)
    top.grid(axis="y",alpha=.18)
    top.legend(prop=font,loc="upper left",frameon=False)
    years=list(range(2019,2026))
    x=np.arange(len(years))
    for offset,(label,frame),color in zip([-.25,0,.25],annual,colors):
        values=frame.set_index("year").completed_cycles_by_entry_year.reindex(years).to_numpy()
        bottom.bar(x+offset,values,width=.23,color=color,label=label)
        for xi,value in zip(x+offset,values):
            bottom.text(xi,value+.16,str(value),ha="center",fontsize=9,color=color)
    bottom.axhline(5,color="#9b2226",ls="--",lw=1.3,label="每个完整年份至少5次")
    bottom.set_xticks(x,years)
    bottom.set_ylabel("完整持仓周期数",fontproperties=font)
    bottom.set_ylim(0,13)
    bottom.grid(axis="y",alpha=.15)
    bottom.legend(prop=font,loc="upper right",frameon=False)
    for ax in [top,bottom]:
        ax.spines[["top","right"]].set_visible(False)
    fig.text(.085,.018,"2018-05-02至2026-08-26，压力成本，全程计入空仓日。2018与2026是不完整年份，未列入年度门槛。\n扩点原模型于2021年触发原定账户回撤停止，后续空仓；节点校准仍未同时达到夏普、回撤和频率目标。",fontproperties=font,fontsize=9,color="#555555")
    fig.tight_layout(rect=[0,.075,1,1])
    fig.savefig(OUT/"账户与年度次数.png",dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sa,sb=load(A/"summary.json"),load(B/"summary.json")
    ma=pd.read_csv(A/"results/账户联合评价.csv")
    mb=pd.read_csv(B/"results/账户联合评价.csv")
    rows=[("原IF节点＋日样本模型",ma[(ma.model=="DENSE_IF_VOL_GAMMA")&(ma.node_stage=="S0_IF_ONLY")&(ma.scenario=="STRESS")].iloc[0]),
          ("加入转正节点",ma[(ma.model=="DENSE_IF_VOL_GAMMA")&(ma.node_stage=="S1_PLUS_FORECAST_ONSET")&(ma.scenario=="STRESS")].iloc[0]),
          ("同一节点池＋逐期校准",mb[(mb.model=="TYPE_OFFSET_CALIBRATION")&(mb.node_stage=="S1_PLUS_FORECAST_ONSET")&(mb.scenario=="STRESS")].iloc[0])]
    table=["| 固定阶段 | 成交周期 | 实际胜率 | 平均盈利/平均亏损金额 | 净夏普 | 最大回撤 | 账户净利润 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for label,r in rows:
        table.append(f"| {label} | {r.opportunities} | {r.win_rate:.2%} | {r.realized_cash_payoff_ratio:.2f} | {r.net_sharpe:.3f} | {r.max_drawdown:.2%} | {r.net_profit_cny:,.2f}元 |")
    annual_a=pd.read_csv(A/"results/完整自然年交易次数.csv")
    annual_b=pd.read_csv(B/"results/完整自然年交易次数.csv")
    nav_a=pd.read_csv(A/"results/完整逐日账户.csv")
    nav_b=pd.read_csv(B/"results/完整逐日账户.csv")
    nav_groups,annual_groups=[],[]
    for (label,r),nav,annual in zip(rows,[nav_a,nav_a,nav_b],[annual_a,annual_a,annual_b]):
        nav_groups.append((label,nav[(nav.model==r.model)&(nav.node_stage==r.node_stage)&(nav.scenario=="STRESS")]))
        annual_groups.append((label,annual[(annual.model==r.model)&(annual.node_stage==r.node_stage)&(annual.scenario=="STRESS")]))
    plot(nav_groups,annual_groups)
    annual_table=["| 完整自然年 | 原IF筛选 | 加入转正节点 | 同节点池逐期校准 | 要求 |","|---|---:|---:|---:|---:|"]
    for year in range(2019,2026):
        values=[int(f.loc[f.year==year,"completed_cycles_by_entry_year"].iloc[0]) for _,f in annual_groups]
        annual_table.append(f"| {year} | {values[0]} | {values[1]} | {values[2]} | ≥5 |")
    pred=pd.read_csv(B/"results/完整节点预测.csv")
    rank=[]
    for name,sample in pred.loc[~pred.if_event].groupby("model"):
        y=sample.target_profit.to_numpy(int)
        positive=int(y.sum())
        auc=float((rankdata(sample.p_win)[y==1].sum()-positive*(positive+1)/2)/(positive*(len(y)-positive)))
        rank.append({"model":name,"scope":"ADDED_FORECAST_NODES","n":len(sample),"auc":auc,
            "expectation_spearman":float(spearmanr(sample.expected_net_proxy,sample.target_net_proxy).statistic),
            "diagnostic_role":"POSTHOC_EXPLANATION_ONLY_NO_NEW_SELECTION"})
    pd.DataFrame(rank).to_csv(OUT/"节点区分度补充诊断.csv",index=False,encoding="utf-8-sig")
    effective_fits=sa["new_probability_fits"]+sa["new_conditional_mean_fits"]+sb["new_probability_fits"]+sb["new_amplitude_fits"]
    report=f"""
# 510300：先训练概率与盈亏幅度，再逐类增加节点

**本次连续完成两轮实际训练及账户比较，仍未达到净夏普1.3、最大回撤10%、每个完整自然年至少5次的联合目标。** 日样本训练改善了部分盈亏幅度预测，但新增转正节点没有形成合格交易优势；后续节点校准提高了账户胜率，却降低了盈亏比和夏普。保留训练中有效的幅度改进，不把这些账户标为可用策略。

## 实际完成的工作

使用已有2510个有效日样本，范围2016-04-14至2026-08-12，沿用10日收益期限及24bp名义压力成本代理。新保存并核验的有效拟合结果共{effective_fits}个：概率{sa['new_probability_fits']+sb['new_probability_fits']}个、盈利或亏损条件幅度{sa['new_conditional_mean_fits']+sb['new_amplitude_fits']}个。这是逐期模型头数量，不是独立策略或独立证据数量。两轮新生成{sa['new_accounts']+sb['new_accounts']}个账户；第二轮另复用第一轮4个原账户，不能重复算成新回测。没有新采集。

第一轮在100个既有月末原点更新模型，随后逐日固定参数。训练只纳入当时已经结束的十日标签，按平均逆重叠数加权。末次月末有2492行成熟日样本，重叠权重合计约250.1；这一合计也不是独立样本数。7个预先声明的模型/对照，共14091条预测，实际覆盖2013个预测日。

第二轮固定全部148个候选节点，使用每个节点之前已成熟的节点结果，分别拟合公共校正与节点类型校正。没有成熟节点或某方向尚无样本时沿用原预测，保留早期空仓日期，未另选更有利的账户起点。

## 训练阶段真正改善了什么

在同一100次月末评价中，日样本IF模型相对月末抽样IF模型，概率Brier误差改善0.21%，盈利幅度MSE下降6.04%，亏损幅度MSE下降22.76%，整体预期收益MSE下降3.14%。盈利幅度改善在前半段为8.01%、后半段为0.58%；亏损幅度分别改善16.81%、26.69%，均已保留在完整表中。

增加已知波动尺度后，月末亏损幅度MSE又下降3.08%，但盈利幅度MSE上升1.71%。因此只能说部分幅度预测改善，不能说三项预测已经同时可靠，更不能把预测误差下降直接换算成账户夏普。

## 逐步扩点及完整账户结果

原10个IF耗竭节点日期不变。新增一类“预测联合状态转正”的候选节点：主模型p>0.5、预计盈利幅度>预计亏损幅度且预期期望为正，由不满足转为满足的首日。新增138个日期，其中5个发生于月末模型更新日，与原IF节点没有同日重复，合计148节点。第一次取得模型或数据缺口后的首次状态不自动构成新节点。

这个节点表示预测状态变化，并未证明是市场变盘。p>0.5且盈利幅度>亏损幅度已蕴含正期望，三条判断不是三份相互独立的证据；0.5和幅度比1是固定方向性分界，不是用户要求的“高胜率”数值标准。

以下均是20万元完整账户、2018-05-02至2026-08-26共2022个交易日、压力成本结果。压力成本为单边佣金2bp且最低5元，滑点10bp；仓位沿用事前10%年化波动预算及100份整手约束。所有空仓日计入夏普；完整周期一买一卖计一次。

{chr(10).join(table)}

第一行只有1笔亏损，盈亏比3.38的证据很薄，不能将事后85.71%胜率称为下次交易的确定概率。扩点账户2021年触发原定8%回撤停止，后续保持空仓；没有重启资金曲线或删去后续空仓日。该账户最大盈利交易占总净利润约86.66%，收益集中度也较高。

节点校准将扩点账户胜率提高至65.38%，但平均盈利金额仍小于平均亏损金额；其按单笔投入归一后的收益率盈亏比为1.035，与金额口径0.876分列，不能挑选较高口径掩盖金额损益。校准账户并未触发永久回撤停止，仍未达年度次数要求。

![完整账户与年度次数](账户与年度次数.png)

## 每个完整年份分别验收

{chr(10).join(annual_table)}

2018和2026是样本两端的不完整自然年，单独保留在原始表中，未用年化次数代替完整年度要求。三组结果的年度最小完整周期数均为0；两轮全部新账户中，没有同时通过夏普、回撤、逐年频率的账户。

## 新节点为何没有达标

138个新增节点的十日净收益代理实际获利比例为51.45%，平均净代理收益为-0.1939%；原模型平均预测为+0.2947%。实际盈利节点的平均幅度约2.56%，模型约3.06%；实际亏损节点平均幅度约3.11%，模型约2.65%。这一差异是继续做节点分布校准的明确依据，不能通过只保留已成交或盈利的节点把它隐藏。

逐期校准将新增节点的平均预计亏损幅度移至约3.14%，接近实际均值；但亏损预测MSE几乎未改善，概率Brier误差反而上升约0.91%，盈利幅度MSE上升约10.58%。均值偏差缩小没有建立更好的逐节点识别能力。

补充的事后解释性诊断显示，原新增节点概率排序AUC约0.535，校准后约0.481。它们只用于解释已保存的结果，没有用来改门槛、重新选节点或运行新账户，也不是独立显著性检验。

## 当前判断及后续边界

用户的“达到目标再停止”保留为总体任务要求，当前goal_achieved=false。完成这两轮不等于完成整个策略目标。日样本幅度训练可作为后续研究基线；本次新增节点与两种节点校准均不晋升策略。固定模型、节点日期、成本、持有期及完整失败账户封存。

当前缺口是可验证的交易信息和节点区分能力。继续在已经反复查看的历史上更改阈值、起点或模型，直到显示夏普1.3，会增加选择偏差，不能作为“拒绝过拟合”的验收证据。后续有明确不同的信息依据时可以继续现有数据研究；独立验收仍需没有参与这些模型与节点选择的证据。此处没有假设已经存在这样的证据，也没有恢复采集。

## 直接文件及复算

第一轮见01_日样本训练与扩点下的protocol.json、freeze.json、models、results；第二轮见02_固定节点分布校准。两轮verify入口都只复算保存参数、预测、账户现金流和年度计数，不重新拟合或生成账户。

第一轮求解过程中两次中断；诊断证明一个L-BFGS停止标志异常，但最大梯度约1.02e-8，已小于原1e-6标准。保留原代码及修正记录后以原凸目标梯度判据完成，未改目标、特征或经济规则；中断重复拟合不计独立试验。第二轮的实现修正只移动输入副本的保存位置，保证verify不写结果文件，发生于本轮拟合前。

第一轮协议继承的旧account_end字段为2026-08-18，本轮实际按照同一协议新写明的“末次可用IF信号十日结束”规则计算至2026-08-26，旧字段未被本轮程序读取；解释见协议字段说明.json，账户结果没有据此重新选择。第二轮继承的若干父模型展示字段同样不参与计算，实际三项对照在其models_description和代码MODELS中明确列出。

这些是历史探索结果。所有行情此前已经被查看，第二轮也明确受第一轮发现启发，不能冒称独立前向验证。没有新的交易授权，没有启动采集或任何下单操作。
"""
    write(OUT/"训练进展与结果.md",report)
    write(A/"阶段结果.md",f"# 日样本概率幅度训练与一类扩点\n\n有效保存202个概率头及606个条件幅度头。日样本IF版相对月末版的盈利/亏损MSE分别改善6.04%/22.76%，加入138个新候选节点后压力账户30笔、胜率60%、盈亏金额比1.047、夏普0.371、回撤8.383%、逐年最少0笔。新增节点不晋升策略；模型误差改进与账户失败同时保留。\n\n研究总报告见同批交付的训练进展与结果.md。全部预测、148节点母集、24账户及逐年次数都在本目录。")
    write(B/"阶段结果.md","# 固定节点的概率幅度校准\n\n148节点固定，保存290个概率校准头、578个幅度校准头，另20个无成熟条件组的头保持原预测。新生成8个账户，复用父级4个对照。节点类型校准后的压力账户26笔、胜率65.38%、金额盈亏比0.876、夏普0.304、回撤6.444%，逐年次数仍失败。未证明校准带来预测或账户优势；本次表示不晋升。")
    shutil.copyfile(A/"verification/numerical_probe.json",A/"数值中断诊断.json")
    clarification={"recorded_at":sa["completed_at"],"changes_results":False,
        "first_study":{"inherited_unused_account_end":"2026-08-18","active_predeclared_rule":"最后可用IF信号的第十个持有交易日结束","actual_end":"2026-08-26","date_computed_by":"dense_probability_payoff_nodes_v1.rows"},
        "second_study":{"inherited_unused_presentation_fields":["comparison_models","fixed_comparisons","model_clock","amplitude_variants"],
            "actual_models":["UNCHANGED_PREDICTIONS","SHARED_OFFSET_CALIBRATION","TYPE_OFFSET_CALIBRATION"],"active_definitions":["models_description","fitting","objective","node_dates","account_rules"]}}
    save(OUT/"协议字段说明.json",clarification)
    save(A/"协议字段说明.json",clarification["first_study"])
    save(B/"协议字段说明.json",clarification["second_study"])
    decision={"recorded_at":sb["completed_at"],"status":"TWO_FIXED_RESEARCH_ROUNDS_COMPLETE_TARGET_NOT_MET",
        "user_goal_remains_unfulfilled":True,"goal_achieved":False,"new_saved_probability_heads":492,"new_saved_amplitude_heads":1184,
        "new_saved_fitted_heads_total":effective_fits,"new_accounts":32,"reused_accounts":4,"new_candidate_node_types":1,
        "new_unique_candidate_dates":138,"all_numerical_targets_passed_accounts":0,"new_collection":0,
        "retained_increment":"密集成熟样本在月末分项预测中的幅度误差改善。",
        "rejected_for_strategy_promotion":["仅凭预测状态转正增加节点","固定节点的公共或节点类型offset校准"],
        "independent_validation_established":False,"reason_goal_not_complete":"未同时达到夏普1.3、回撤10%和逐年5周期，独立验证亦未建立。",
        "background_training_running_after_delivery":False,"orders_authorized":False}
    save(OUT/"continuation_status.json",decision)
    write(OUT/"00_README_FIRST.md","""
# 阅读顺序

先读训练进展与结果.md和continuation_status.json。目标未完成，本包没有将高胜率或分项误差下降当作策略达标。

01_日样本训练与扩点：2510个已有日样本、808个有效拟合头、138个新增日期、24个完整账户。

02_固定节点分布校准：固定148节点、868个有效拟合头、8个新账户加4个原样复用账户。

所有直接输入、冻结协议、实现修正、参数、逐日预测、节点、成交和年度次数都在相应目录。协议字段说明.json解释从父级继承但未使用的展示字段，保留原冻结协议。

离线复算：用Python执行第一目录code/dense_probability_payoff_nodes_v1.py verify --root 第一目录；第二目录执行code/node_distribution_calibration_v1.py verify --root 第二目录。只核对保存参数和结果，0拟合、0新账户、0采集。完整参数和包文件身份由根FILE_INDEX.csv索引。
""")
    write(OUT/"用户要求.md","""
# 当前用户要求

“请继续，达到目标再停止。”承接“先训练胜率与盈亏幅度，再逐步增加节点”。

仅510300和现金，20万元，成本后净夏普至少1.3，最大回撤不超过10%，每个完整自然年至少5个完整持仓周期。追求高胜率、高盈亏比和有依据的节点，拒绝过拟合。用户未设定胜率或盈亏比的具体数值门槛。

不新采集、使用已有数据直接训练。频率用于完整策略最终验收，不作为开始概率和幅度训练的门槛。没有下单授权，旧终止策略不恢复。
""")
    write(OUT/"GPT审阅提示词.md","""
请审阅这两轮510300实际训练与节点研究。目标：20万元、净夏普1.3、回撤10%、每个完整自然年至少5周期；先训练概率和盈亏幅度，再逐类扩点，拒绝过拟合。当前目标明确未达，不能把1676个拟合头当作独立试验或独立证据。

请重点检查：

1. 月末抽样与成熟日样本的比较是否同日、同标签、同训练时钟？十日标签重叠权重是否合理，是否错误宣称样本独立？
2. Gamma及已知波动尺度模型的分项改进是否真实、前后半段是否一致？概率改善很小，是否仍夸大确定性？
3. 138个新增转正节点是否严格只用当时特征、固定月末参数和相邻交易日状态？模型更新触发、首次状态、数据缺口是否正确处理？
4. 后续校准是否仅用各预测时点以前成熟的节点，是否保留全部148节点及早期无样本状态？校准虽然修正均值，却恶化部分误差和夏普，报告是否完整保留反例？
5. 完整现金流、分红应收与付款、T+1、佣金滑点、8%回撤永久停止及空仓日是否一致？年度次数是否按完整买卖周期而非成交边数计算？
6. 两轮同一历史上的发现身份、求解标志修正、未使用的父级展示字段是否清楚，哪些属于数值核验而不是独立验证？
7. 请给下一步优先级及停止条件：需要何种明确不同的信息才能改善节点区分，而不是继续换门槛或回测起点直到出现1.3？若现有条件不足以完成目标，请直接指出证据缺口。

先写结论，再引用具体文件和行号。不要因为数据量或ZIP检查通过就宣称科学有效、已被外部评审、能够实盘或获准交易。
""")
    write(OUT/"EXCLUSIONS.md","""
# 范围与排除

包括两轮全部直接数值输入、原行情、分红、已保存IF特征、月末历史模型、节点、所有新拟合参数、预测、成交、完整账户及年度次数。IF原始合约来源不重复打包，本轮从既有已核对特征开始；旧获取过程不因此被重新验证。

排除验证解压目录、缓存、外部工具环境和包生成后的最终回执。第一轮数值中断诊断另外保留在阶段目录，原冻结代码及实现修正都包含。

本包结构检查与保存结果复算，不等于外部审阅、独立前向验证或真实成交。所有研究输入均来自已有本地数据。
""")
    for name in ["510300_existing_data_training_mandate_v1.json","510300_sparse_node_training_contract_v1.json"]:
        path=ROOT/"config"/name
        config=load(path)
        config.update(current_round="510300_NODE_DISTRIBUTION_CALIBRATION_V1",goal_achieved=False,
            latest_progress_receipt="reports/research/510300_probability_payoff_progress_20260924/continuation_status.json",
            last_research_result="两轮有效保存1676拟合头、32新账户，全部未达完整目标；日样本幅度改进保留，新增节点及校准不晋升。")
        save(path,config)
        save(OUT/name,config)
    status=ROOT/"RESEARCH_STATUS.md"
    text=status.read_text(encoding="utf-8-sig")
    marker="# 510300 研究权威状态"
    if marker not in text:
        raise ValueError("缺少研究状态插入点。")
    entry="> 2026-09-24 按用户“达到目标再停止、先训练胜率与盈亏幅度再扩点”连续完成两轮：2510成熟日样本，保存492概率头和1184幅度头；新增138个转正候选节点，新生成32账户，另复用4个。日样本较月末抽样的盈利/亏损幅度MSE改善6.04%/22.76%；原IF压力账户7笔、85.71%胜率、金额盈亏比3.376、夏普0.351。扩点账户30笔、夏普0.371、回撤8.383%；节点校准后26笔、65.38%胜率、金额盈亏比0.876、夏普0.304、回撤6.444%。全部年度最少0笔，无账户同时达到1.3/10%/逐年5次。完整结果与失败保留，goal_achieved=false，不采集、不恢复旧策略。见[两轮训练进展](reports/research/510300_probability_payoff_progress_20260924/训练进展与结果.md)。"
    if entry not in text:
        status.write_text(text.replace(marker,marker+"\n\n"+entry,1),encoding="utf-8")
    print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
