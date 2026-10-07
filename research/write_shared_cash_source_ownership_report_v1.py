"""归因所有共同账户、真实补充及原进入改变，不重跑金融。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import shared_cash_source_ownership_study_v1 as study
from research.write_core_actual_acceptance_report_v1 import markdown,fmt

ROOT,OUT = study.ROOT,study.OUT
REPORT = "共同现金来源归属_完整金融与新增点位失败归因.md"
COLORS = {study.SAVED_CORE:"#344d63",study.SAVED_SUPPORT:"#ac9871",study.SAVED_ACCEPTANCE:"#8688a7",
          study.rules.POLICIES[0]:"#bd5146",study.rules.POLICIES[1]:"#3294a0",study.rules.POLICIES[2]:"#5c9d72"}


def frozen_exact():
    records = study.read(OUT / "protocol.json")["sources"]
    for item in records:
        if study.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原冻结来源改变："+item["path"])
    return len(records)


def load_account(period,cost,policy):
    if policy in study.SAVED:
        return study.saved_account(period,cost,policy)
    p = OUT / "accounts" / period / cost / policy
    result = {n:pd.read_parquet(p / f"{n}.parquet") for n in (*study.ACCOUNT_TABLES,"ownership_evidence")}
    result["terminal"] = study.read(p / "terminal.json")
    return result


def inventory():
    files = []
    for period in study.PERIODS:
        for cost in study.COSTS:
            for policy in study.rules.POLICIES:
                p = OUT / "accounts" / period / cost / policy
                for n in (*study.ACCOUNT_TABLES,"ownership_evidence"):
                    f = p / f"{n}.parquet"
                    files.append({"path":study.relative(f),"sha256":study.digest(f)})
                f = p / "terminal.json"
                files.append({"path":study.relative(f),"sha256":study.digest(f)})
    if len(files) != 84:
        raise ValueError("十二金融账户七文件不完整。")
    return files


def labeled(frame,period,cost,policy):
    f = frame.copy()
    for name,value in (("policy",policy),("cost",cost),("period",period)):
        f.insert(0,name,value)
    return f


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("金融报告已输出，不重复。")
    frozen = frozen_exact()
    result = study.read(OUT / "summary.json")
    if len(result["metrics"]) != 24 or len(result["comparisons"]) != 20 or result["new_accounts"] != 12:
        raise ValueError("完整账户或控制范围不全。")
    files = inventory()
    metrics = pd.DataFrame(result["metrics"])
    accounts = {(p,c,k):load_account(p,c,k) for p in study.PERIODS for c in study.COSTS for k in (*study.SAVED,*study.rules.POLICIES)}
    cycles,evidence,diagnoses,owners,matching,matches = [],[],[],[],[],[]
    for (p,c,k),a in accounts.items():
        cycles.append(labeled(a["trades"],p,c,k))
        if k not in study.rules.POLICIES:
            continue
        evidence.append(labeled(a["ownership_evidence"],p,c,k))
        d,t = a["daily"],a["trades"]
        stats = metrics.loc[metrics.period.eq(p) & metrics.cost.eq(c) & metrics.policy.eq(k)].iloc[0]
        gross = float((d.price_pnl+d.dividend_accrual).sum())
        fees = float(d.commission.sum()+d.slippage.sum())
        bound_curve = 200000.+(d.price_pnl+d.dividend_accrual).cumsum().to_numpy(float)
        bound_returns = bound_curve/np.r_[200000.,bound_curve[:-1]]-1
        bound = study.previous.parent.measurements.return_statistics(bound_returns)
        diagnoses.append({"period":p,"cost":c,"policy":k,"actual_gross_pnl_fixed_quantities":gross,
            "commission_and_slippage":fees,"net_total_pnl":float(d.equity.iloc[-1]-200000.),
            "fees_as_share_positive_gross":fees/gross if gross > 0 else None,
            "fixed_quantity_zero_fee_explanatory_cagr":bound["net_cagr"],"fixed_quantity_zero_fee_explanatory_sharpe":bound["net_sharpe"],
            "bound_role":"EXPLANATORY_FIXED_ACTUAL_QUANTITY_NOT_EXECUTABLE_NEW_STRATEGY",
            "mean_exposure":stats.mean_exposure,"account_stopped":a["terminal"]["stopped"],"open_pnl_cny":a["terminal"]["open_pnl_cny"],
            "event_consumed_counts":a["decisions"].loc[a["decisions"].entry_event,"current_complement_event_role"].value_counts().to_dict(),
            "actual_exit_reasons":a["orders"].loc[a["orders"].side.eq("SELL"),"reason"].value_counts().to_dict(),
            "all_rejections":a["rejections"].to_dict("records")})
        for owner in ("CORE","COMPLEMENT"):
            subset = t.loc[t.owner.eq(owner)]
            done = subset.loc[subset.status.eq("COMPLETE")]
            owners.append({"period":p,"cost":c,"policy":k,"owner":owner,"cycles":len(subset),"complete":len(done),
                "open":int(subset.status.eq("RIGHT_CENSORED").sum()),"wins":int(done.net_pnl.gt(0).sum()),
                "losses":int(done.net_pnl.lt(0).sum()),"completed_actual_pnl_cny":float(done.net_pnl.sum()),
                "role":"ACTUAL_COMPONENT_PNL_WITH_SHARED_CASH_FEEDBACK_NOT_STANDALONE_ACCOUNT"})
        old = accounts[(p,c,study.SAVED_CORE)]["trades"]
        cols = ["entry_date","entry_origin","exit_date","status","buy_debit","net_pnl","net_return"]
        left,right = old[cols].copy(),t[[*cols,"owner"]].copy()
        for column in ("entry_date","entry_origin","exit_date"):
            left[column] = pd.to_datetime(left[column]).astype("datetime64[ns]")
            right[column] = pd.to_datetime(right[column]).astype("datetime64[ns]")
        joined = left.merge(right,on="entry_date",how="outer",suffixes=("_original_A","_common_account"),indicator=True,validate="one_to_one")
        joined["entry_identity_role"] = joined.pop("_merge").astype(str)
        matching.append(labeled(joined,p,c,k))
        matches.append({"period":p,"cost":c,"policy":k,"matched_actual_entry_dates":int(joined.entry_identity_role.eq("both").sum()),
            "original_only_actual_entry_dates":int(joined.entry_identity_role.eq("left_only").sum()),
            "new_only_actual_entry_dates":int(joined.entry_identity_role.eq("right_only").sum()),
            "note":"不同真实日期含吸收或迟到及现金反馈，不能只相加补充周期损益"})
    all_cycles,all_evidence = pd.concat(cycles,ignore_index=True),pd.concat(evidence,ignore_index=True)
    owner_table = pd.DataFrame(owners)
    study.table("全部24账户所有周期与开放_费用复本非独立样本",all_cycles)
    study.table("全部12新账户当前来源消费与真实归属",all_evidence)
    study.table("全部12新账户毛优势费用风险与拒绝归因",pd.DataFrame(diagnoses))
    study.table("全部12新账户CORE与补充实际完成盈亏",owner_table)
    study.table("全部12新账户与原A真实进入外连接_现金数量反馈",pd.concat(matching,ignore_index=True))
    study.table("全部12新账户原A进入吸收迟到与新增统计",pd.DataFrame(matches))
    actual_complements = all_cycles.loc[all_cycles.policy.eq(study.rules.POLICIES[0]) & all_cycles.owner.eq("COMPLEMENT")].copy()
    study.table("主机制全部真实支持补充_赢亏与费用复本完整",actual_complements)
    stress_new = actual_complements.loc[actual_complements.cost.eq("STRESS")].sort_values("entry_date")
    if len(stress_new) != 5 or len(actual_complements) != 10:
        raise ValueError("主机制全部实际补充范围改变。")
    intervals = pd.read_parquet(OUT / "results/全部五对照两尺度40净增量区间.parquet")
    if len(intervals) != 40:
        raise ValueError("五对照所有区间未完成。")
    diagnosis = {"at":study.previous.parent.original.now(),"all_new_account_diagnoses":diagnoses,
        "all_owner_component_results":owners,"all_original_A_entry_matches":matches,
        "all_actual_primary_stress_complements":stress_new.to_dict("records"),
        "accepted_unique_primary_complements":5,"actual_complement_cost_replicates":10,
        "independent_validation":"NOT_ESTABLISHED","overfitting_removed":False,"goal_achieved":False}
    study.write(OUT / "post_run_diagnosis.json",diagnosis)
    proposal = {"at":study.previous.parent.original.now(),"status":"PROPOSED_NOT_REGISTERED_NOT_ADMITTED_NOT_RUN",
        "name":"SOURCE_EXPECTATION_AND_ETF_TRANSMISSION_EVIDENCE_ADMISSION_REVIEW",
        "question":"能否取得事前可知的政策预期/实际增量及主线传播到ETF的不同信息，解释已有支持反弹持续与失败，而非根据输赢改技术门槛？",
        "reason":"实际来源互补增加次数但近段净收益/Sharpe降低；价格接受无支持加频也明显无优势。仅重复宽松公告/技术柱不建立新增持续优势。",
        "scope":"先核对当前7券商来源及已保存原政策65节点、所有真实主補充5例及反例；期待信息若无可靠公开上界/原版本则NO_VIEW/NOT_ADMITTED，不反填完整预期。",
        "separation":"新源可知的市场预期与政策实际变化、指数成分可观察传播、解释与预测分开；同公告确认不新分、不用ETF成交量当流入。",
        "stop":"只有事后解读、不能提前取得、重复已用信息或靠已知亏损PMI/窗口调参则不准入新金融。",
        "next_action":"来源可用性与用途审查，先不拟合/回测；不同机制还须事前唯一固定全部动作和必要对照。",
        "financial_admission":"NOT_ADMITTED","new_financial_runs":0,"new_accounts":0,
        "no_rescue":"R240/R232/R236固定终态不改，不升格近期固定退出对照，不拼接时期；原E03独立前瞻保持。",
        "independent_validation":"NOT_ESTABLISHED","goal_achieved":False}
    study.write(OUT / "next_source_expectation_transmission_review_proposal.json",proposal)
    plt.rcParams.update({"font.family":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,"font.size":9})
    picture = OUT / "figures"
    picture.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(2,2,figsize=(16,9),layout="constrained")
    for column,period in enumerate(study.PERIODS):
        for policy in (*study.SAVED,*study.rules.POLICIES):
            d = accounts[(period,"STRESS",policy)]["daily"]
            axes[0,column].plot(d.date,d.equity/10000,color=COLORS[policy],lw=1.4,label=study.NAMES[policy])
            axes[1,column].plot(d.date,d.drawdown*100,color=COLORS[policy],lw=1.1)
        axes[0,column].set_title(period+"｜压力费用｜各20万元独立开始")
        axes[0,column].set_ylabel("净资产：万元")
        axes[1,column].set_ylabel("收盘回撤：%")
        for ax in axes[:,column]:
            ax.grid(alpha=.15)
        axes[0,column].legend(fontsize=7,loc="upper left")
    fig.suptitle("共同现金、固定来源归属：增加机会仍须同时改善完整收益和夏普",fontsize=14)
    equity_path = picture / "两时期全部六政策_完整压力净值与回撤.png"
    fig.savefig(equity_path,dpi=145)
    plt.close(fig)
    fig,axes = plt.subplots(5,1,figsize=(16,16),layout="constrained")
    data = pd.read_parquet(study.previous.DAILY)
    for ax,t in zip(axes,stress_new.itertuples()):
        start,end = t.entry_origin-pd.Timedelta(days=10),t.exit_date+pd.Timedelta(days=10)
        segment = data.loc[data.date.between(start,end)]
        ax.plot(segment.date,segment.ac,color="#6d6d6d",lw=1.1,label="现金平移收盘")
        ax.axvline(t.entry_date,color="#aa5146",ls="--",label="共同支持真实进入")
        ax.axvline(t.exit_date,color="#5c865c",ls=":",label="共同支持真实退出")
        positions = ax.twinx()
        for policy,color,label in ((study.rules.POLICIES[0],"#b34c3f","共同账户真实库存"),(study.SAVED_CORE,"#344d63","原A真实库存")):
            d = accounts[(t.period,"STRESS",policy)]["daily"]
            d = d.loc[d.date.between(start,end)]
            positions.step(d.date,d.shares/10000,where="post",color=color,lw=1.5,label=label)
        ax.set_title(f"{t.entry_date:%Y-%m-%d}—{t.exit_date:%Y-%m-%d}｜真实初买{t.entry_quantity:,}份｜净损益{t.net_pnl:+,.2f}元｜{t.exit_reason}")
        positions.set_ylabel("真实库存：万份")
        ax.set_ylabel("现金平移收盘")
        ax.grid(alpha=.15)
        ax.legend(fontsize=7,loc="upper left",ncol=3)
        positions.legend(fontsize=7,loc="upper right")
    fig.suptitle("主机制全部5个真实补充：来源拥有库存，CORE在占用期间无法另用20万元",fontsize=13)
    position_path = picture / "全部五个真实补充_价格与原A库存占用.png"
    fig.savefig(position_path,dpi=145)
    plt.close(fig)
    def metric_view(cost):
        return markdown(pd.DataFrame([{"时期":x.period,"账户":study.NAMES[x.policy],"净年化":fmt(x.net_cagr,True),
            "净夏普":fmt(x.net_sharpe),"最大回撤":fmt(x.max_drawdown,True),"完成/开放":f"{x.completed_cycles}/{x.unfinished_cycles}",
            "赢/亏":f"{x.wins}/{x.losses}","净pB":fmt(x.p_times_b),"标准净期望":fmt(x.standard_expectancy_loss_units),
            "完整年均次数":fmt(x.average_full_year_cycles)} for x in metrics.loc[metrics.cost.eq(cost)].itertuples()]))
    new_view = stress_new[["period","entry_origin","entry_date","exit_date","entry_quantity","net_pnl","exit_reason"]].copy()
    new_view.columns = ["时期","当时决定","真实买入","真实退出","初买份数","实际净损益元","退出原因"]
    risk = metrics.loc[metrics.cost.eq("STRESS"),["period","policy","worst_day","worst_trade","mean_exposure",
        "gross_turnover_over_initial_capital","total_commission","total_slippage","largest_winner_fraction_of_all_wins","open_pnl_cny"]].copy()
    risk["policy"] = risk.policy.map(study.NAMES)
    main_owner = owner_table.loc[owner_table.cost.eq("STRESS") & owner_table.policy.eq(study.rules.POLICIES[0])]
    main_matching = pd.DataFrame(matches).loc[lambda f:f.cost.eq("STRESS") & f.policy.eq(study.rules.POLICIES[0])]
    original_intervals = intervals.loc[intervals.control.eq(study.SAVED_CORE)]
    report = f"""# 共同现金来源归属：完整金融、真实补充与失败归因

TECH.R239—R240已完成唯一配置：12新完整账户、12保存对照、24指标、五对照20比较及40两尺度区间。四经济门通过{sum(x['economic_passed'] for x in result['gates'])}/4、稳定{result['historical_stability_passed']}；终态{result['status']}。本固定配置不晋升，收益夏普同时提高尚未实现，独立验证/去过拟合未建立。没有结果后改规则、挑时期或金融重跑。

新增机会实际执行：压力主早期23完成/1开放，近期35完成/0开放；原A22完成/1开放和32完成/0开放。完整年均次数从4.4/4.1667提高到4.6/4.6667。早期净年化2.7025%/夏普0.606417虽高于原A1.8371%/0.438043，实际pB0.824837未>1；近期3.6362%/1.096314低于原A3.9908%/1.216910，实际pB0.983214亦未>1。

## 压力费用全部结果

{metric_view('STRESS')}

## 基础费用全部结果

{metric_view('BASE')}

两档费用共同报告，不能选择近期基础pB1.083567替代压力0.983214。净pB为实际完成净回报胜率乘真实平均赢/平均亏，不用计划2ATR赔率；标准期望为pB减亏损概率，所以pB<1时标准期望仍可能正。用户要求的较强pB>1与完整净收益夏普同时提高都保持。原独立支持早期三胜零亏，B/pB不定义，不填无穷或声称100%真实成功概率。

## 全部五个实际支持补充

{markdown(new_view)}

2019真正共同账户初买11500份、实际净利7225.05元，区别于R238原A背景单次容量11600份和旧独立支持29700份/16105.95元。较早2015补充已经改变现金、峰值和风险，资金反馈继续影响2019。2023/2025的真实共同损失3203.09/2201.58元也高于旧独立损失2058.31/1150.68元，因为资金路径/预算和数量不同。不能保留大赢家的独立规模同时只算小亏损的独立规模。

{markdown(main_owner)}

这张组件表来自同一真实账户，已受共享现金和以前交易影响；CORE损益也会因来源占用和新数量变化。组件可以解释实际总损益，不是两个独立账户的预测收益相加。

## 原A实际进入的保留、吸收与迟到

{markdown(main_matching)}

逐笔完整外连接另存所有原A和新真实进入，日期不同可能是补充吸收原CORE、退出后较迟CORE进入或现金改变机会，不直接将原A所有盈亏加五笔补充。主持仓来源固定，没有在2019盈利后倒换身份，也没有在2024已持CORE时再买一套支持仓位。原A及原支持关闭另一功能的八个适配账户，日净值、真实订单和周期都精确复现；旧账户经济动作没有因本实验偷偷改变。

## 更多交易、持有延长与成本分开

仅价格补充将完整年均次数提到早10.0/近10.1667，但压力净年化−0.1687%/0.0765%、净夏普−0.021661/0.040139，缺少收益优势。支持信息对比价格控制有开发点值区别，也不足以令本完整主机制通过。近期固定退出归因3.9325%/1.174684高于主3.6362%/1.096314，仍低于同期原A3.9908%/1.216910；不能看结果后将对照改名为合格策略或只拼某时期。

所有12新账户的实际数量毛损益、佣金/滑点、拒绝、占用及原目标身份均另表保存。固定实际数量去费路径只作解释上界，未重计算资金风险或改变交易，不是可执行零成本策略。完整风险与集中度如下，包含所有空仓日、开放损益及实际换手。

{markdown(risk)}

![全部完整净值回撤](figures/两时期全部六政策_完整压力净值与回撤.png)

![五个实际补充与库存占用](figures/全部五个真实补充_价格与原A库存占用.png)

第二图的线为收盘价格、阶梯为真实日末库存，竖线是真实开盘进入/退出日期；开盘成交价在订单，不将收盘线当成交价。所有五个实际补充含失败均显示，量价/日MACD/原PMI及决定前锚已在[R238完整五例图](../510300_core_support_complement_description_v1/implementation_v1_0_1/原A与支持来源互补_全部点位时钟容量和持仓冲突.md)解释，不从两亏损的PMI反推新过滤门槛。

## 所有不确定性及冻结记录

主对原A的全部既定区间如下；五对照全40行在CSV及summary一并保存。20和252日尺度及基础/压力均报告，不能事后只取有利短窗口。

{markdown(original_intervals)}

11必要合成测试、19原CORE前缀通过；R238原31来源/资格/已发生开盘前缀复用且不重复跑。八成功原账户适配、12新现金复算和12来源/归属时序通过，原{frozen}冻结来源及84新账户文件保存。登记前合成订单未来首个卖出引出17/18列模式差异，随后原A适配出现object/StringDtype差异；两失败、原代码及先前测试回执保留。统一声明输出字段/字符串类型在登记前完成，逐值测试不放宽，0新候选金融重跑。

首版/全国政策覆盖未认证，历史截至2026-09-30全部已看开发，独立验证NOT_ESTABLISHED，DSR/PBO及搜索选择校正NOT_COMPUTED。历史多次研究选择不因一次冻结自动消失；开发点值和区块区间不能代替新的独立样本。原13项E03前瞻保持，本结果不生成当前市场建议、持仓或订单。

下一优先用途是审查不同的事前信息：政策实际变化相对此前市场预期、主线传播到ETF的可观察证据。先核对7原券商来源和65原节点的来源/时钟可用性；如果只有事后故事、未知首版或重复当前已用信号，就NO_VIEW/NOT_ADMITTED。它是待做来源审查提案，0新金融准入，不按亏损改PMI/窗口或救R240。原独立前瞻继续按其原合同。

依据：[唯一冻结规则](../../../docs/510300_SHARED_CASH_SOURCE_OWNERSHIP_V1.md)、[固定登记](protocol.json)、[完整实际结果](summary.json)、[全部归因](post_run_diagnosis.json)、[全部24指标](results/全部24完整账户_共同现金来源归属.csv)、[逐年完整次数](results/全部逐年完整次数与净收益.csv)、[所有区间](results/全部五对照两尺度40净增量区间.csv)、[全部实际进入外连接](results/全部12新账户与原A真实进入外连接_现金数量反馈.csv)、[下一来源审查](next_source_expectation_transmission_review_proposal.json)。
"""
    report_path = OUT / REPORT
    with report_path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(report)
    study.write(OUT / "delivery_receipt.json",{"at":study.previous.parent.original.now(),"report":study.relative(report_path),
        "report_sha256":study.digest(report_path),"original_frozen_sources_exact":frozen,"new_account_files":files,
        "new_account_files_exact":len(files),"all_metrics":24,"all_interval_rows":40,"new_accounts":12,
        "new_financial_run_starts":1,"financial_replays_after_result":0,
        "figures":[{"path":study.relative(p),"sha256":study.digest(p)} for p in (equity_path,position_path)],
        "figure_panels":9,"all_actual_primary_stress_complements":5,"goal_achieved":False})
    print("共同现金完整金融、全部新增/吸收/迟到与毛成本归因已保存；两图9面板，未重跑金融。",flush=True)


if __name__ == "__main__":
    main()
