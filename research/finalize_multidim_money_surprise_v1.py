"""核对保存的事件评分与账户，交付历史多维评分结果。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import multidim_money_surprise_score_v1 as study


ROOT, OUT = study.ROOT, study.OUT


def check_saved_results() -> dict:
    predictions = pd.read_csv(OUT / "逐事件联合评分.csv")
    inputs = pd.read_parquet(OUT / "月度事件联合状态及标签.parquet").set_index("stat_month")
    models = study.read(OUT / "saved_models.json")
    errors = []
    for saved in models:
        current = inputs.loc[saved["stat_month"]]
        pool = inputs.loc[saved["training_months"]]
        assert pool.exit_at.le(current.decision_at).all()
        assert pool.state_idx.ge(current.state_idx-504).all()
        row = predictions[predictions.stat_month.eq(saved["stat_month"])].iloc[0]
        assert int(row.training_events) == len(pool)
        assert abs(row.actual_net5-current.actual_net5) < 1e-12
        for family in ["state", "joint"]:
            tree = saved[family+"_tree"]
            values = current[tree["features"]].to_numpy(float)
            node = 0
            while tree["children_left"][node] != -1:
                feature = tree["feature"][node]
                value = float(np.float32(values[feature]))
                node = tree["children_left"][node] if value <= tree["threshold"][node] else tree["children_right"][node]
            errors.append(abs(tree["value"][node]-row[family+"_prediction"]))
    assert max(errors) < 1e-12
    accounts = study.read(OUT / "account_result.json")
    metric_errors = []
    for case_id in {row["case_id"] for row in accounts["accounts"]}:
        ledger = pd.read_parquet(OUT / "accounts" / (case_id+".parquet"))
        for saved in [row for row in accounts["accounts"] if row["case_id"] == case_id]:
            if saved["period"].startswith("完整期"):
                left, right = "2021-01-01", "2025-12-31"
            elif saved["period"].startswith("较早期"):
                left, right = "2021-01-01", "2023-12-31"
            else:
                left, right = "2024-01-01", "2025-12-31"
            positions = np.flatnonzero(ledger.date.between(left,right))
            initial = saved["capital"] if positions[0] == 0 else ledger.equity.iloc[positions[0]-1]
            nav = np.r_[initial, ledger.equity.iloc[positions].to_numpy(float)]
            returns = nav[1:]/nav[:-1]-1
            sd = returns.std(ddof=1)
            sharpe = np.sqrt(242)*returns.mean()/sd if sd > 1e-15 else None
            cagr = (nav[-1]/initial)**(242/len(returns))-1
            drawdown = 1-np.min(nav/np.maximum.accumulate(nav))
            for calculated, recorded in [(sharpe,saved["net_sharpe"]),(cagr,saved["cagr"]),(drawdown,saved["max_drawdown"])]:
                if calculated is None:
                    assert recorded is None
                else:
                    metric_errors.append(abs(calculated-recorded))
    assert max(metric_errors) < 1e-12
    recent = predictions[predictions.era.eq("2024—2025")]
    np.testing.assert_allclose(recent.state_prediction,recent.joint_prediction,rtol=0,atol=1e-12)
    np.testing.assert_allclose(recent.state_score,recent.joint_score,rtol=0,atol=1e-12)
    result = {"checked_at":study.common.now(),"saved_tree_predictions_recomputed":len(errors),
              "maximum_saved_tree_error":max(errors),"account_period_metric_rows_recomputed":len(accounts["accounts"]),
              "maximum_account_metric_error":max(metric_errors),"mature_training_labels_only":True,
              "primary_state_and_joint_predictions_identical":True,
              "primary_state_and_joint_scores_identical":True,
              "independent_validation":False,"current_view":"NO_VIEW"}
    study.save("必要结果核对.json",result)
    return result


def percent(value, digits=2):
    return "未定义" if value is None or not np.isfinite(value) else f"{value:.{digits}%}"


def finish() -> None:
    if (OUT / "交付结果.json").exists():
        raise RuntimeError("结果已交付，不覆盖。")
    checked = check_saved_results()
    score_result = study.read(OUT / "result.json")
    account_result = study.read(OUT / "account_result.json")
    table = pd.DataFrame(account_result["accounts"])
    primary = table[table.period.eq("主要期2024—2025")].set_index("case_id")
    main = primary.loc["joint_200000"]
    small = primary.loc["joint_20000"]
    full_main = table[table.case_id.eq("joint_200000") & table.period.eq("完整期2021—2025")].iloc[0]
    cycles = pd.read_csv(OUT / "全部账户自然交易周期.csv",parse_dates=["entry","exit"])
    recent_cycles = cycles[cycles.case_id.eq("joint_200000") & cycles.entry.ge("2024-01-01")].copy()
    ledger = pd.read_parquet(OUT / "accounts/joint_200000.parquet")
    after_0924 = float(ledger.loc[ledger.date.eq("2024-09-25"),"equity"].iloc[0]-ledger.loc[ledger.date.eq("2024-09-23"),"equity"].iloc[0])
    biggest = recent_cycles.loc[recent_cycles.profit.idxmax()]
    concentration = {"largest_cycle_entry":biggest.entry,"largest_cycle_exit":biggest.exit,
                     "largest_cycle_profit_cny":float(biggest.profit),"share_of_primary_net_profit":float(biggest.profit/main.net_profit),
                     "nav_increase_from_20240923_close_to_20240925_exit":after_0924,
                     "share_of_largest_cycle_profit_after_20240923_close":after_0924/float(biggest.profit),
                     "interpretation":"固定交易本来已在持有。9月24日政策是在9月13日判断之后的新信息；这里是日期区间损益拆分，不识别政策因果贡献。没有删除该笔、改变退出或重新回测。"}
    study.save("主要期收益集中度.json",concentration)
    make_plot(ledger,recent_cycles,main)

    def link(name,label):
        return f"[{label}](<{(OUT/name).as_posix()}>)"
    account_rows=[]
    for key,label in [("joint_200000","联合评分 20万元"),("joint_20000","联合评分 2万元"),("state_200000","同事件状态树 20万元"),("state_20000","同事件状态树 2万元"),("buy_hold50_200000","期初50%预算被动持有参考"),("cash_200000","全现金参考")]:
        row=primary.loc[key]
        sharpe="未定义" if pd.isna(row.net_sharpe) else f"{row.net_sharpe:.3f}"
        cycles_text="未平仓" if key.startswith("buy_hold") else str(row.completed_cycles)
        account_rows.append(f"| {label} | {sharpe} | {row.cagr:.2%} | {row.max_drawdown:.2%} | {row.net_profit:+,.2f} | {cycles_text} |")
    cycle_rows=[f"| {row.entry:%Y-%m-%d} | {row.exit:%Y-%m-%d} | {row.profit:+,.2f} | {int(row.initial_quantity):,} |" for row in recent_cycles.itertuples()]
    predictions=pd.read_csv(OUT/"逐事件联合评分.csv")
    original = predictions[predictions.era.eq("2024—2025")]
    high=original[original.joint_score.ge(80)&original.joint_prediction.gt(0)]
    low=original[original.joint_score.lt(80)]
    report_name="预期偏差与指数状态联合评分_历史账户结果.md"
    report=f"""# 510300 预期偏差与指数状态联合评分的历史账户结果

**2024—2025年的成本后账户夏普已出现超过1.2的历史点值，但完整目标仍未实现。** 九变量联合评分的20万元账户净夏普为{main.net_sharpe:.3f}，2万元账户为{small.net_sharpe:.3f}；对应年化收益仅{main.cagr:.2%}、{small.cagr:.2%}，低于既定10%目标。2024年自然完成3次、2025年2次交易，均少于每年5次要求。收益还集中在一笔交易，不能把这个局部结果视为拒绝过拟合目标已经满足。

本轮研究的是指数整体。在八项订单、资金利差、融资、趋势、波动和成交状态上，增加M2公布值相对事前报告预期的偏差，用最多两层条件分支一起评分。同事件的八变量状态树作为删除新增信息的对照，同九变量线性模型和历史均值另作参考。没有给M2正偏差预设“利好加分”，也没有调整已保存的80分门槛。

原预期资料有77个月，按本轮公布截止2025年末可用68次；2021—2025最终得到47次逐事件评分，其中2024—2025有20次。每次月度公布仅用一行，训练限于此前504个ETF交易日内已经退出成熟的12至24次事件；没有把同一消息重复扩成日样本。M1没有进入本轮，2025年M1定义变化不通过拼接被忽略。M2预期来自此前固定选择的银行或券商报告，保留为“事前预期代理”，不是完整市场共识或不可变首版快照。

![主要期净值与五笔交易利润](<{(OUT/'联合评分_账户净值与收益集中.png').as_posix()}>)

**主要历史阶段2024—2025的完整账户指标**如下。所有现金日都进入夏普和年化计算，现金利率和无风险利率均取零，年化使用242交易日。账户自2021年起连续运行，没有为最近两年重置资金：20万元联合账户在2024年初的权益为{main.start_equity:,.2f}元。表中利润是2024—2025这段的实际模拟净值增加。

| 模型和本金 | 净夏普 | 复合年化 | 最大回撤 | 期间净利润 元 | 自然完成周期 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(account_rows)}

50%预算被动持有只作描述性参考，不实施候选的动态尾部限额和止损，不能视为符合当前风险合同的候选。它在2024年的仓位和现金继承自2021年的原账户，不是2024年新建半仓。全现金波动为零，夏普未定义。

主账户实际最高收盘仓位44.13%，主要期平均仓位仅{main.mean_exposure:.2%}。低平均仓位来自机会稀少和现金等待；净夏普越过1.2并没有带来10%的资本增长。不能通过提高仓位或放宽风险限额把年化收益直接补到目标。

**五次高分机会的分布仍然很偏。** 固定10000份的评分标签平均净收益{high.actual_net5.mean():+.2%}，中位数{high.actual_net5.median():+.2%}，胜率{high.actual_net5.gt(0).mean():.0%}。其他15次已评分事件均值为{low.actual_net5.mean():+.2%}。这是已经观察过的历史差异，不是独立验证，也不能仅凭五次事件判断规律稳定。

| 主账户入场 | 自然退出 | 成本后利润 元 | 初始份额 |
|---|---|---:|---:|
{chr(10).join(cycle_rows)}

2024年9月18日至25日这一笔赚了{biggest.profit:,.2f}元，占主要期全部净利润的{biggest.profit/main.net_profit:.2%}。9月23日收盘至25日退出，账户增值{after_0924:,.2f}元，占该笔利润{after_0924/biggest.profit:.2%}。9月24日公开的降准降息等安排，晚于9月13日这次模型判断；它可以作为之后发生的新事实，不能写回成模型入场时已经知道的理由。这个拆分只确认损益发生时间，没有识别政策造成了多少收益。[9月24日官方发布会实录](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)

**M2预期偏差没有为主要期增加预测信息。** 全部47次评分中，新增变量只进入一次树；2024—2025的20次预测与分数，与删除该变量的状态树逐项相同，五次高分日期也完全一样。因此最近两年的正收益证据属于“特定货币公布日期上的指数状态评分”，不能归为“M2超预期策略有效”。两套账户的净夏普与利润有细微差别，来自此前交易造成的权益、峰值和整手份额路径差异，不能当作本期消息增量。

模型的历史分支也要按当时理解。例如2024年9月13日的树，第一层使用此前五日融资余额变化，另外一支再看订单月度变化；M2偏差并未进入。保存的分支只是该历史训练池形成的映射，不将某个精确阈值另立为新买点。

较早阶段2021—2023的联合模型只有两笔交易，均亏损，净夏普−0.228。整个2021—2025连续账户净夏普为{full_main.net_sharpe:.3f}、年化{full_main.cagr:.2%}。这些较早结果用于展示阶段差异，并不要求一个因子在所有时期方向相同；即使只评价主要两年，收益、次数与集中度仍不足以完成当前目标。

本轮费用是单边佣金万四且最低5元，单边滑点千一。买入份额在开盘前按已知价格上限及风险预算固定，实际开盘不用于重新选择数量。最高目标仓位50%，五日ES预算2.5%，10%缺口压力预算5%，并保留半回撤剩余空间约束；登记、除息、到账分别记账，实行T+1。全部六个模拟账户共7272行已保存，账户恒等式误差低于0.000001元；94个保存树预测和18行分段账户指标已从保存数据重算一致。这里仍是日线成交代理，没有逐笔订单簿或实际成交证明。

后续优先辨别：这组融资、订单与价格条件是否在货币公布日期上才有剩余收益，还是普通日期也有同样表现。当前证据没有理由继续堆叠M2衍生项，也不足以把月度规则直接扩大为每日买点。保留本轮规则与全部反例，不反号、不改80分门槛、不删除9月盈利交易，也不把新口径缺失当成零。

资料入口：{link('逐事件联合评分.csv','47次事件评分')}、{link('全部104月的本轮覆盖.csv','完整月份及缺失')}、{link('完整账户指标.csv','全部账户指标')}、{link('全部账户自然交易周期.csv','所有账户交易周期')}、{link('account_protocol.json','账户合同')}、{link('必要结果核对.json','必要结果核对')}。预期代理例子可在原来源记录中定位：2024年1月11日报告对2023年12月M2预期为10.1%，1月12日官方公布9.7%；本轮使用的偏差为−0.4个百分点。[原事前报告](https://research2.bualuang.co.th/upload/Bls240111.pdf)、[央行当期数据](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/1a6fe9c99e0b4b3ba35bbd038ef8535c/index.html)

当前只完成历史研究，没有当前行情观点、交易指令或前瞻跟踪任务。完整目标保持未实现。
"""
    (OUT/report_name).write_text(report,encoding="utf-8")
    score_result.update(status="COMPLETED_JOINT_EVENT_SCORE_ACCOUNT_POINT_SHARPE_ONLY",finalized_at=study.common.now(),
                        report=report_name,figure="联合评分_账户净值与收益集中.png",new_accounts=6,
                        account_net_sharpe=main.net_sharpe,account_primary_cagr=main.cagr,
                        account_primary_drawdown=main.max_drawdown,primary_point_sharpe_at_least_1_2=True,
                        all_financial_and_frequency_targets_pass=False,
                        message_increment_primary=False,largest_cycle_share_of_primary_net_profit=biggest.profit/main.net_profit,
                        account_result="account_result.json",goal_achieved=False)
    study.save("result.json",score_result)
    relative=OUT.relative_to(ROOT).as_posix()
    summary=f"月度M2预期代理与八项指数状态完成47次评分和6个账户。2024—2025主账户净夏普{main.net_sharpe:.3f}、年化{main.cagr:.2%}、回撤{main.max_drawdown:.2%}；2024年3次、2025年2次，最大一笔占利润82.26%。M2在主要期未改变任何预测，删除它的状态树亦有同样五个日期。夏普历史点值超过1.2，但收益、频率和稳健性目标未实现。"
    for path in [ROOT/"config/510300_historical_cause_discovery_v1.json",ROOT/"config/510300_existing_data_training_mandate_v1.json"]:
        value=study.read(path)
        if "historical_cause" in path.name:
            value.update(latest_completed_study=relative+"/result.json",latest_report=relative+"/"+report_name,
                         current_study=relative+"/protocol.json",latest_result_summary=summary,updated_at=study.common.now(),
                         latest_completed_branch_boundary="主要两年净夏普点值超过1.2，但年化不足10%、每年不足5次且利润集中；M2信息未建立增量，不把此局部结果升格为完成。",
                         next_historical_question="先去重，辨别货币公布日这个时间条件与融资、订单和量价状态的增量；普通日期对照须按当时信息构造，不能事后扩大当前月度规则。")
        else:
            value.update(current_round=study.STUDY,latest_progress_receipt=relative+"/result.json",last_research_result=summary,
                         latest_continuation_report=relative+"/"+report_name,latest_continuation_classification="PROGRESS_JOINT_EVENT_SCORE_AND_FULL_ACCOUNT",
                         goal_status="active",goal_achieved=False)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    study.save("交付结果.json",{"at":study.common.now(),"status":"SAVED_PENDING_VISUAL_REVIEW","report":report_name,
                             "report_sha256":study.sha(OUT/report_name),"necessary_calculation_checks":checked,
                             "goal_achieved":False,"orders_authorized":False})
    print("已保存研究报告、账户曲线与集中度；夏普点值达到1.2，完整目标仍未达到。",flush=True)


def make_plot(ledger,cycles,main):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.font_manager import FontProperties,fontManager
    font=Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font))
    plt.rcParams.update({"font.family":FontProperties(fname=str(font)).get_name(),"axes.unicode_minus":False,"font.size":10.5})
    fig,axes=plt.subplots(1,2,figsize=(13.4,6.4),gridspec_kw={"width_ratios":[1.15,1]})
    for name,label,color in [("joint_200000","联合评分主账户","#147b8c"),("buy_hold50_200000","期初50%预算被动持有参考","#a1adb9")]:
        d=ledger if name.startswith("joint") else pd.read_parquet(OUT/"accounts"/(name+".parquet"))
        mask=d.date.ge("2024-01-01")
        first=int(np.flatnonzero(mask)[0])
        nav=d.loc[mask,"equity"]/float(d.equity.iloc[first-1])*100
        axes[0].plot(d.loc[mask,"date"],nav,label=label,color=color,linewidth=2.1 if name.startswith("joint") else 1.6)
    axes[0].axhline(100,color="#5c6570",linestyle="--",linewidth=.8,label="全现金")
    axes[0].set_title("2024—2025完整账户净值",loc="left",fontsize=12,pad=14)
    axes[0].set_ylabel("2024年初权益归一为100")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].legend(loc="upper left",frameon=False,fontsize=9)
    colors=["#db913d" if value==cycles.profit.max() else ("#147b8c" if value>0 else "#ac5c61") for value in cycles.profit]
    bars=axes[1].bar(np.arange(len(cycles)),cycles.profit,color=colors,width=.62)
    for bar in bars:
        y=bar.get_height()
        axes[1].text(bar.get_x()+bar.get_width()/2,y+(100 if y>=0 else -100),f"{y:+,.0f}",ha="center",va="bottom" if y>=0 else "top",fontsize=10.5)
    axes[1].set_xticks(np.arange(len(cycles)),[d.strftime("%Y-%m") for d in cycles.entry])
    axes[1].set_title("主账户五次自然完成交易的净利润",loc="left",fontsize=12,pad=14)
    axes[1].set_ylabel("元")
    axes[1].set_ylim(-1050,7700)
    axes[1].axhline(0,color="#5c6570",linewidth=.8)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("历史夏普超过1.2，收益与机会数仍不足",x=.065,y=.97,ha="left",fontsize=17)
    fig.text(.065,.9,f"主账户净夏普 {main.net_sharpe:.3f}  ·  年化收益 {main.cagr:.2%}  ·  最大回撤 {main.max_drawdown:.2%}",fontsize=12,color="#254d59")
    fig.text(.065,.095,"2024年3次、2025年2次；2024年9月一笔占两年净利润82.26%。M2预期偏差未改变主要期的评分或机会。",fontsize=10.5,color="#42505d")
    fig.text(.065,.045,"全部现金日和压力费用计入指标。被动持有参考不实施动态尾部预算；本图为已使用历史的发现结果。",fontsize=10.5,color="#65717d")
    fig.tight_layout(rect=(.02,.14,.99,.85))
    fig.savefig(OUT/"联合评分_账户净值与收益集中.png",dpi=160,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    finish()
