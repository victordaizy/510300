"""汇总完整候选清单与已完成的两个机制，不把未跑项目填成零收益。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from research.factor96_margin_repair_v1 import OUT,save,now,digest

ACTIVE=OUT/"completed_run"


def progress():
    factors=json.loads((OUT/"factor_registry.json").read_text(encoding="utf-8"))
    strategies=json.loads((OUT/"strategy_registry.json").read_text(encoding="utf-8"))
    mapping={
        "T01":("NOT_RUN_OVERLAP_REVIEW", "与已失败突破回踩方向重叠；新增ER/保留率必须证明机制差别，不能只改窗口救援。"),
        "T02":("NOT_RUN_MEMBER_NEW_LOW_FEATURES", "已有点时成员和广度20资料；尚未构建同成员20日新低覆盖及上次有效测试对照。"),
        "T03":("COMPLETE_PRIMARY_NO_EVENTS", "固定主时钟在2017—2025年无符合全部条件的买入；夏普未定义。额外延迟对照主期2次，仍未达标。"),
        "T04":("NOT_RUN_FIXED_CORE_CONTRIBUTIONS", "需点时行业贡献及每月提前固定领先3组；不能拿每日最强行业代替。"),
        "T05":("COMPLETE_TARGET_FAILED", "固定测试未达夏普目标；主期20万元压力夏普-1.058448；未调参。"),
        "T06":("NOT_RUN_OVERLAY_BASELINE_AND_MEDIAN", "需固定价格基准及同成员中位收益；覆盖层不作为独立入场策略。"),
        "T07":("NOT_RUN_SYSTEM_ETF_UNIT_CLOCK", "需要同指数产品完整份额、分拆、NAV和历史首次可得时钟。"),
        "T08":("NOT_RUN_SYNCHRONIZED_QUOTES", "需要可核验同步IOPV及同指数ETF分钟报价；未以日收盘溢折价替代。"),
        "T09":("NOT_RUN_CARRY_ADJUSTED_BASIS", "需逐合约价差、已知待分红与持有成本；旧持仓量失败不代表新基差已检验。"),
        "T10":("NOT_RUN_CROSS_BORDER_RESPONSE_MODEL", "本地有部分海外ETF日线；尚需跨时区、汇率、首轮反应和剩余信息合同。"),
        "T11":("NOT_RUN_EARNINGS_CASHFLOW_CLOCK", "需同口径首次盈利事实、现金质量及权重时钟，不使用后来修订的财务数。"),
        "T12":("NOT_RUN_REPURCHASE_PROGRESS_CLOCK", "需实际实施回购新增额、用途和可比前公告，不将计划额当实际流入。"),
        "T13":("NOT_RUN_KNOWN_SUPPLY_WINDOW", "需当时已公开供给条款及结束窗口；现有旧解禁检验不等于这项新定义。"),
        "T14":("LOCAL_SOURCE_FOUND_NOT_RUN", "已找到2015—2026年DR007.IB原始缓存及25条7天操作利率记录；后者不是完整首次政策宣布序列，需先明确操作利率时钟。"),
        "T15":("NOT_RUN_OPTION_DATA_AND_SCOPE", "完整同期限期权链未准入；本轮未开展期权模型或交易。"),
        "T16":("NOT_RUN_ACCEPTED_BASELINE_REQUIRED", "待固定可检验的基准持仓后评价风险覆盖；减仓本身不是收益策略。"),
        "T17":("NOT_RUN_NO_ACCEPTED_COMPONENTS", "本批无通过的机制，不拼接两个失败账户。未来组合须真实单账户净额执行。"),
        "T18":("NOT_RUN_MATURE_CALIBRATION_REQUIRED", "需成熟历史预测与事前季度更新合同；本轮没有按近绩效择优或软启停。"),
    }
    current=[]
    for r in strategies:
        status,reason=mapping[r["id"]]
        current.append({**r,"current_status":status,"current_evidence":reason,"individual_performance":None})
    pd.DataFrame(current).to_csv(OUT/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"strategy_progress.json",current)
    implemented={"B04":"已实现负2倍波动冲击、10日去重和确认事件；卡片中的rv3衰减幅度未作独立筛选。",
        "B06":"已实现速度差与超过前日高点的首次确认，用于T05及价格对照。",
        "C01":"已实现20日涨跌价格冲击中位数比，正负日各至少5个。",
        "F01":"按两市融资余额恒等式推导5日净变化；发布时间至少滞后1交易日。",
        "F02":"两段不重叠5日净扩张率之差，用于T03。",
        "F06":"隐含5日偿还活动已实现；不是强平字段，E02配对未实现。"}
    records=[]
    for r in factors:
        related=[s["id"] for s in strategies if r["id"] in str(s["factor_ids"]).split(",")]
        state="PARTIAL_COMPONENT_TESTED" if r["id"] in implemented else "NOT_RUN"
        hint=[]
        for p in r["repository_hint"].split(";"):
            p=p.strip()
            hint.append({"path":p,"exists":(ROOT/p).exists()})
        records.append({**r,"current_status":state,"strategy_dependencies":",".join(related),
            "current_note":implemented.get(r["id"],"保留原假设；尚未完成该项独立定义、数据可得性和收益检验。"),
            "standalone_net_sharpe":None,"path_hint_check":json.dumps(hint,ensure_ascii=False),
            "data_grade_is_attachment_classification":True})
    pd.DataFrame(records).to_csv(OUT/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"factor_progress.json",records)
    evidence=[]
    for path in sorted(set(p.strip() for r in factors for p in r["repository_hint"].split(";"))):
        folder=ROOT/path
        if not folder.is_dir():
            continue
        for name in ["status.json","result.json","acceptance_outcome.json"]:
            item=folder/name
            if item.is_file():
                j=json.loads(item.read_text(encoding="utf-8"))
                target=OUT/"legacy_status_snapshots"/folder.name/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(item,target)
                keys=["status","study_id","goal_achieved","independent_validation","historical_point_target_met"]
                evidence.append({"original_path":item.relative_to(ROOT).as_posix(),"sha256":digest(item),
                    **{k:j[k] for k in keys if isinstance(j,dict) and k in j}})
                break
    save(OUT/"overlap_status_inventory.json",evidence)
    next_steps=[{"priority":1,"strategy":"T14","next_action":"先核验DR007和7天操作利率的日期、值与可得时间，再冻结压力回落事件；当前未运行收益。"},
        {"priority":2,"strategy":"T02","next_action":"从已保存点时成分重建20日新低覆盖，固定前次测试事件，比较价格对照。"},
        {"priority":3,"strategy":"T04","next_action":"确认点时行业及贡献口径，月末固定领先组，后续参与扩散只向前计算。"},
        {"priority":4,"strategy":"T07/T09","next_action":"先补体系份额公开时钟或待分红基差；不把缺数据填成策略零收益。"},
        {"priority":5,"strategy":"T10/T11/T12/T13","next_action":"按各自独立信息源建立完整发布事件母集及首轮观察窗口，避免只挑有行情的案例。"}]
    save(OUT/"next_actions.json",next_steps)


def render_plot(metrics):
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(2,1,figsize=(12.5,8.5),sharex=True,gridspec_kw={"height_ratios":[1.3,1]})
    styles={"T03":("T03 融资缓和：无交易","#159895","-"),
        "T03_PRICE":("T03 价格对照","#159895","--"),"T05":("T05 偿还活动","#ba3b46","-"),
        "T05_PRICE":("T05 价格对照","#875eaa","--"),"BUY_HOLD_50":("期初50%买入持有","#7188a5","-")}
    for policy,(name,color,style) in styles.items():
        z=pd.read_parquet(ACTIVE/f"accounts/MAIN/200000/STRESS/LAG1/{policy}/ledger.parquet")
        nav=z.equity/200000
        dates=pd.concat([pd.Series([z.date.iloc[0]-pd.Timedelta(days=1)]),z.date],ignore_index=True)
        full=pd.Series(np.r_[1.,nav.to_numpy()])
        axes[0].plot(dates,full,label=name,color=color,ls=style,lw=1.5)
        axes[1].plot(dates,(full/full.cummax()-1)*100,color=color,ls=style,lw=1.5)
    axes[0].set_title("510300 候选库首批机制检验",fontproperties=font,fontsize=18,loc="left",pad=56)
    axes[0].set_ylabel("完整账户净值",fontproperties=font)
    axes[1].set_ylabel("自账户峰值回撤（%）",fontproperties=font)
    axes[1].set_xlabel("2021—2025年；每日均保留现金及未平仓损益",fontproperties=font)
    axes[0].legend(prop=font,loc="lower left",bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=10)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.text(.09,.02,"20万元 · 压力成本：每边万4，最低5元，10bp滑点并按0.001价位取整 · 非独立前向验证",fontproperties=font,fontsize=10,color="#555555")
    fig.tight_layout(rect=[.02,.05,.99,.98])
    fig.savefig(OUT/"主期完整账户净值与回撤.png",dpi=160)
    plt.close(fig)


def main():
    progress()
    metrics=pd.read_csv(ACTIVE/"metrics.csv")
    render_plot(metrics)
    x=pd.read_parquet(ACTIVE/"daily_features_lag1.parquet")
    stages=[]
    for period,lo,hi in [("EARLY","2017-01-03","2020-12-31"),("MAIN","2021-01-04","2025-12-31")]:
        q=x[x.date.between(lo,hi)]
        stages.append({"period":period,"days":len(q),"T03_price_confirmations":int(q.T03_price_signal.sum()),
            "T03_and_financing_contraction":int((q.T03_price_signal & q.F01.lt(0)).sum()),
            "T03_all_conditions":int((q.T03_price_signal & q.F01.lt(0) & q.F02.gt(0)).sum()),
            "T05_price_confirmations":int(q.T05_price_signal.sum()),
            "T05_all_conditions":int((q.T05_price_signal & q.F06_implied_repay5.gt(q.F06_q80)).sum())})
    save(OUT/"signal_stage_counts.json",stages)
    names={"T03":"T03 急跌修复＋融资缓和","T03_PRICE":"T03 价格对照","T05":"T05 高偿还＋价格韧性","T05_PRICE":"T05 价格对照","BUY_HOLD_50":"期初半仓买入持有"}
    table=[]
    for policy in names:
        r=metrics[(metrics.period=="MAIN") & (metrics.capital==200000) & (metrics.cost=="STRESS") & (metrics.lag==1) & (metrics.policy==policy)].iloc[0]
        s="未定义（无交易）" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        table.append(f"| {names[policy]} | {s} | {r.cagr:.2%} | {r.max_drawdown:.2%} | {int(r.closed_cycles)} |")
    body="\n".join(table)
    report=f"""# 510300：96项候选库首批实测结果

截至2026年9月27日，尚未找到满足成本后夏普1.2的合格策略。本轮完整导入96项因子、18套策略，实际完成T03和T05两条融资机制及其价格对照，共64个账户情景。T03主时钟没有符合全部条件的交易，夏普未定义；T05主期20万元压力夏普为-1.058，年化-1.00%。两项均不进入组合，旧失败和已终止策略保持原状态。总目标继续，交易权限没有变化。

## 本轮实际完成了什么

来源Excel、HTML与粘贴说明已原样保存并核对全部ID和名称。来源分类为G1 29项、G2 51项、G3 16项；这是候选库的资料分层，不是96项均已获数据准入。已计算的B04事件部分、B06、C01、F01、F02及F06隐含偿还共同用于两条策略，不把它们当作6个分别有效的独立因子。其余项目保持NOT_RUN。

先通过7项实现测试，再冻结进入、退出、成本、时钟、两年成熟风险窗口和统计方法。首次执行因旧融资表缺11个深市日，在计算标签和策略账户前停止。随后从深交所官方历史接口定点取得这11日，与已有沪市官方原值合并，融资日历恢复为2015-01-05至2025-12-31共2674日。原2663行余额与买入额不变；原输入、原冻结与失败记录均保留。补齐后重新冻结数据，交易代码和阈值未改。

## 主要结果

主期为2021-01-04至2025-12-31，共1212个交易日。以下均为20万元完整现金账户、压力成本、主披露时钟；年化使用242个交易日，252转换另列且不用于改变判定。现金日、分红应收、未平仓损益及末端压力退出成本准备均计入。

| 策略或对照 | 净夏普 | 复合年化 | 最大回撤 | 完整交易周期 |
|---|---:|---:|---:|---:|
{body}

T03的层层条件为：9次价格修复确认，其中6次确认时融资仍净收缩，再要求收缩速度缓和后为0次。这个结果说明当前固定合同没有交易机会，不能把夏普填成0或1.2。T05有21次满足条件的信号，实际完成18个周期，胜率27.78%；信号与完整周期数量不同，原因包括已有仓位、价格退出和资金/风险条件。

2万元对照独立按100份及最低5元手续费模拟：T05主期压力夏普-1.155、年化-1.06%、最大回撤6.05%；T03仍无交易。2017—2020年早期对照中，T03同样没有交易，T05的20万元压力夏普-0.127、年化-0.19%，没有支持稳定的历史收益。

预先登记的额外延迟一天情景中，T03在主期出现2个周期，20万元压力夏普0.593、年化0.48%；T05压力夏普-1.056。不能在看见这两个周期后换用较好时钟：这既没有达到夏普线，也不是独立证据。

## 融资信息到底增加了什么

每条候选与仅移除融资条件的价格对照比较，其余成本、风险预算和退出相同。主期20万元压力日收益差按20交易日区块、4000份保存抽样检验。T03年化算术收益增量为-0.045个百分点，95%区间约[-1.010,1.031]个百分点；T05为+0.014个百分点，95%区间约[-1.512,1.508]个百分点。两项区间都跨零，家族内Holm调整后的单侧p值均约0.954，没有确认可靠增量。

这些区块结果只针对本批两项比较，不能清除96项候选和历史大量研究的选择偏差。两个历史时期都已经被研究过，不能称为独立样本外或前向验证。

## 规则与数据边界

只模拟510300.SH和人民币现金，不加杠杆、不卖空。收盘确认后下一开盘成交，100份整手，0.001报价单位，T+1，涨跌停方向保守不成交。基础成本每边万2、最低5元、5bp滑点；压力成本万4、最低5元、10bp。它们是研究假设，没有声称是已核实券商报价。

沿用当前账户尾部合同：目标仓位上限50%，条件五日ES95预算2.5%，单次-10%跳空假设损失预算5%，并约束剩余回撤空间。条件风险每日用此前两年且已成熟的五日结果重估，仅管理仓位，不额外根据回测好坏择时。回撤触发后的退出受T+1、开盘价格和涨跌停约束，因此预算不是最大损失保证。

融资净变化由两市余额变化计算，偿还活动由“买入额减余额变化”推导，报告明确称为隐含偿还，含权益调整，不能解释成强平。依据是[上交所融资汇总字段定义](https://www.sse.com.cn/market/othersdata/margin/sum/)及补齐的深交所官方原响应。股票ETF的T+1约束已按[上交所说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)处理。

沪市历史有全段交叉核对，深市以批量来源、样本核对及官方补缺组成；此次新增11日的官网展示精度约100万元。数据为回溯取得，没有逐日首次送达或修订版本的不可变证据。采用至少一个交易日滞后及额外延迟对照降低时钟风险，不能把这种假设写成已证明的历史首次公开记录。

本轮没有新增独立前向观察、外部GPT审阅、Paper/Shadow信号或实盘委托。没有选择获利年份拼接，也没有重启SELECTED_MIX_BAND10_SIMPLE2。

## 其余16套策略与下一步

完整进度见“18策略当前进度.csv”和“96因子当前进度.csv”。T14已找到DR007.IB本地原始缓存及25条7天操作利率记录；25条是操作记录而非完整首次政策宣布母集，例如2024年9月存在宣布、实施、表内记录不同日的问题，因此尚未将其直接冒充完整可交易政策时钟。下一阶段先完成这个定义及数据衔接，再冻结资金压力回落事件。

之后分别推进T02的点时成分新低收缩、T04的提前固定领先组传播；T07、T09先解决体系份额和基差的真实数据合同。T01仍需与既有突破回踩失败做定义差异核对。T06/T16是覆盖层；T17/T18是组合与适应框架。当前没有通过的组成策略，不将它们强行组合成一个好看的账户。

停止条件：已经冻结且失败的两项不换阈值、窗口或退出补救；缺少必要时间戳的机制不得写成独立验证通过。只有另有机制信息、正式新协议和完整结果时才启动新的候选。夏普1.2仍是验收线，不能作为调参后的填充值。

## 阅读顺序

1. 本文件和“主期完整账户净值与回撤.png”。
2. “18策略当前进度.csv”“96因子当前进度.csv”。
3. completed_run/protocol.json、result.json、metrics.csv及paired_increment.json。
4. completed_run/accounts内64份完整账户、交易周期和逐日决策。
5. data_repair官方原响应、原始来源快照、两次冻结与首次失败记录。
6. saved_verification_receipt.json、完整文件索引和独立保存结果复算脚本。
"""
    (OUT/"研究结论.md").write_text(report,encoding="utf-8")
    (OUT/"00_README_FIRST.md").write_text("# 510300候选库首批研究\n\n本包用于审阅已完成的两条融资机制，不代表96项全部完成，更不代表目标达成。先读研究结论，再看策略进度、协议、全量指标与账户。\n\n原第一次执行于数据日历检查停止，没有结果；completed_run为官方11日补齐后唯一完成的账户执行。两历史时期都不是独立前向验证。\n",encoding="utf-8")
    prompt="""请评审本包的研究有效性，并提出下一步，不要把结构验证当成策略通过。
用户目标是只操作510300与人民币现金，完整账户成本后夏普至少1.2；既有合同同时检验年化10%和回撤10%。
先读研究结论、两份完整进度表、completed_run/protocol.json和result.json，再核对全量metrics、逐日决策与账户。
本轮只检验T03/T05及价格对照，64个账户情景不是64套独立候选。其余策略和因子不得推断为已回测。
请重点检查：金融字段含义；首次可得时钟与历史回取的区别；11日官方补齐是否只修数据；T03稀疏/无交易；T05相对价格对照的收益、尾部和成本；五日成熟标签；分红、T+1、现金及末端成本；多重选择和已看过历史。
明确哪些结论支持、哪些反驳、哪些缺证据。不要对冻结失败进行参数救援，也不要恢复已经终止的策略。
优先评估T14/T02/T04以及不同信息家族如何形成真正新增证据，给出具体所需输入、固定实验、验收线和停止条件。
没有外部审阅或独立前向结果，不授权订单、Paper/Shadow或实盘。
"""
    (OUT/"01_GPT_REVIEW_PROMPT.txt").write_text(prompt,encoding="utf-8")
    save(OUT/"round_status.json",{"updated_at":now(),"goal_achieved":False,"goal_status":"active",
        "registered_factors":96,"registered_strategies":18,"completed_strategies":["T03","T05"],
        "completed_account_scenarios":64,"qualified_strategies":[],"remaining_strategies":16,
        "next_research":"T14_SOURCE_CLOCK_THEN_FROZEN_EVENT_TEST","external_review":"NOT_PERFORMED",
        "position_impact":0,"orders_authorized":False})
    status=ROOT/"RESEARCH_STATUS.md"
    marker="<!-- FACTOR96_MARGIN_REPAIR_ROUND_20260927 -->"
    previous=status.read_text(encoding="utf-8")
    if marker not in previous:
        entry=marker+"\n\n> 2026-09-27 导入96因子/18策略候选库，首批T03/T05完成64个账户情景，共同目标通过0项；T03主时钟无交易，T05主期20万元压力夏普-1.058448。先前缺11日深市已从官方补齐，原冻结和数据门失败保留，交易规则未改。其余16套保持未运行。目标ACTIVE，独立前向0，仓位影响0。见[本轮结论](reports/research/510300_factor96_mechanism_batch_v1/研究结论.md)。\n\n"
        status.write_text(entry+previous,encoding="utf-8")
    print("研究结论、全候选进度和完整净值图已生成；目标仍未达成。")


if __name__=="__main__":
    main()
