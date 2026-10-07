"""只复算保存账户并交付阶段研究结论，不重跑或调整策略。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import broker_stage_policy_study_v1 as study
from research import broker_stage_policy_inputs_v1 as stage
from research import point_first_passage_study_v1 as original
from research.point_account_nr7_complement_v1 import verify_account

ROOT, OUT = study.ROOT, study.OUT
KEY_DATES = pd.to_datetime(["2015-06-24","2015-06-25","2015-06-29","2015-06-30","2019-01-08","2019-01-14","2019-01-18",
                           "2020-03-27","2020-04-01","2020-04-23","2020-05-28","2020-05-29","2020-06-08",
                           "2024-09-23","2024-09-24","2024-09-26","2024-09-30"])


def saved_account(period, cost, policy):
    folder = OUT/"accounts"/period/cost/policy
    result = {name: pd.read_parquet(folder/f"{name}.parquet") for name in ("daily","orders","trades","decisions","rejections")}
    result["terminal"] = study.read(folder/"terminal.json")
    return result


def source_rows(trades, period):
    rows = []
    for source, group in trades.groupby("source"):
        stats = study.measurements.trade_statistics(group)
        rows.append({"period": period, "cost": "STRESS", "source": source, **stats})
    return rows


def main():
    if (OUT/"delivery_receipt.json").exists():
        raise RuntimeError("保存账户交付已完成，不重复追加。")
    summary = study.read(OUT/"summary.json")
    metrics = pd.read_parquet(OUT/"results/十六完整账户_同资金风险成本比较.parquet")
    observed = pd.read_parquet(OUT/"results/全部3488事前阶段资格与上一完整周结构位.parquet")
    cases = pd.read_parquet(ROOT/"reports/research/510300_volume_lead_price_confirm_explanation_v1/results/四原案例逐日全部量价及量先恢复状态.parquet")
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei"], "axes.unicode_minus":False,
                         "font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    colors = {"A_SAVED_WEIGHT":"#3b4354", "STAGE_ENTRY_AND_EXIT":"#267969", "SAME_ENTRY_FIXED_EXIT":"#c17a35", "RAW_ENTRY_SAME_EXIT":"#6b86ac"}
    names = {"A_SAVED_WEIGHT":"原A", **stage.NAMES}
    checks, source_stats, key_rows, case_rows, decompositions, figures = [], [], [], [], [], []
    for period, (start,end) in study.PERIODS.items():
        for cost in study.COSTS:
            fig, axes = plt.subplots(2,1,figsize=(12.5,7.3),sharex=True,gridspec_kw={"height_ratios":[2,1]})
            accounts = {}
            for policy in ("A_SAVED_WEIGHT", *stage.POLICIES):
                if policy == "A_SAVED_WEIGHT":
                    daily = pd.read_parquet(original.CONTROL/period/cost/policy/"daily.parquet")
                    account = None
                else:
                    account = saved_account(period,cost,policy)
                    accounts[policy] = account
                    daily = account["daily"]
                    checks.append({"period":period,"cost":cost,"policy":policy,**verify_account(account)})
                    yearly = study.annual_rows(account,period,policy,cost)
                    stats = {**study.measurements.metrics(account), **study.extra_metrics(account,yearly)}
                    saved = metrics.loc[metrics.period.eq(period)&metrics.cost.eq(cost)&metrics.policy.eq(policy)].iloc[0]
                    for key,value in stats.items():
                        if isinstance(value,(float,np.floating,int,np.integer)) and not isinstance(value,(bool,np.bool_)):
                            np.testing.assert_allclose(value,saved[key],atol=1e-10,rtol=0,equal_nan=True)
                axes[0].plot(daily.date,daily.equity/200000,label=names[policy],color=colors[policy],linewidth=1.5)
                axes[1].plot(daily.date,-100*daily.drawdown,color=colors[policy],linewidth=1.1)
            axes[0].set_ylabel("完整账户净值 / 初始20万元")
            axes[1].set_ylabel("账户回撤 / %")
            axes[0].legend(loc="upper left",ncol=2,fontsize=9)
            for ax in axes:
                ax.grid(axis="y",color="#dddddd",alpha=.7)
            fig.suptitle(f"{period} · {cost}：阶段政策与两归因对照、原A\n全部现金日及真实摩擦计入；未平仓自然计价",fontsize=13)
            fig.tight_layout(rect=(0,0,1,.93))
            path=OUT/"figures"/f"{period}_{cost}_完整净值与回撤.png"
            path.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(path,dpi=140)
            plt.close(fig)
            figures.append(str(path.absolute().relative_to(ROOT)))
            if cost != "STRESS":
                continue
            primary = accounts[stage.POLICIES[0]]
            source_stats.extend(source_rows(primary["trades"],period))
            row = metrics.loc[metrics.period.eq(period)&metrics.cost.eq(cost)&metrics.policy.eq(stage.POLICIES[0])].iloc[0]
            gross_end = 200000+float(row.gross_at_actual_quantities_pnl)
            decompositions.append({"period":period,"cost":cost,"gross_pnl_cny":row.gross_at_actual_quantities_pnl,
                                   "friction_cny":row.total_commission+row.total_slippage,"net_pnl_cny":row.ending_equity-200000,
                                   "gross_no_friction_cagr_fixed_executed_quantities":(gross_end/200000)**(252/len(primary["daily"]))-1,
                                   "counterfactual_role":"解释上界，不再优化成交数量或风险，不是可执行无费策略"})
            local_keys = observed.loc[observed.date.isin(KEY_DATES)&observed.date.between(pd.Timestamp(start),pd.Timestamp(end))].copy()
            local_cases = cases.loc[cases.date.between(pd.Timestamp(start),pd.Timestamp(end)),["date","original_episode_id"]].copy()
            for policy, account in accounts.items():
                dec = account["decisions"].rename(columns={"origin":"date", "holding_stage":"decision_holding_stage"})
                pos = account["daily"][["date","shares","source","holding_stage","drawdown","equity"]].rename(columns={"shares":"actual_close_shares"})
                local = local_keys.merge(dec,on="date",how="left",validate="one_to_one",suffixes=("","_decision")).merge(pos,on="date",how="left",validate="one_to_one")
                local["period"],local["cost"],local["policy"] = period,cost,policy
                key_rows.append(local)
                dates=local_cases.copy()
                dates["date"]=dates.date.astype(observed.date.dtype)
                values=dates.merge(observed,on="date",how="left",validate="many_to_one").merge(dec,on="date",how="left",validate="many_to_one",suffixes=("","_decision")).merge(pos,on="date",how="left",validate="many_to_one")
                values["period"],values["cost"],values["policy"] = period,cost,policy
                case_rows.append(values)
    study.table("十二保存账户交付复算",pd.DataFrame(checks))
    source_frame=pd.DataFrame(source_stats)
    study.table("主政策全部实际进入来源_压力完整与未完成分开",source_frame)
    study.table("主政策实际数量毛收益与摩擦分解_无费仅解释",pd.DataFrame(decompositions))
    study.table("原17关键日_三政策实际资格请求与持仓",pd.concat(key_rows,ignore_index=True))
    study.table("四原案例240行_三政策全部实际状态",pd.concat(case_rows,ignore_index=True))
    chosen=metrics.loc[metrics.cost.eq("STRESS")]
    lines=["# 分阶段进场与持有：完整结果、具体点位及失败归因","",
           "2026-10-05。TECH.R211一次登记、TECH.R212一次完整账户检验。原A四账户精确复现，主政策四账户和两归因对照八账户共12新账户；0新模型、标签或行情请求。历史为开发/校准。",
           "",
           "结论：本固定政策拒绝，四个经济门0/4，历史稳定门失败，提高完整收益夏普目标未达。较早阶段收益夏普和次数有局部改善，但实际pB仍不达标；近期整体负收益，低于原A，不能由2024成功单笔推成有效策略。", "",
           "## 相同资金、风险和压力费用的比较", "",
           "| 时期 | 政策 | 净年化 | 全日历夏普 | 最大回撤 | 完成周期 | 胜率 | 实际B | pB | 标准期望 | 完整年均次数 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in chosen.itertuples(index=False):
        lines.append(f"| {row.period} | {names[row.policy]} | {row.net_cagr:.4%} | {row.net_sharpe:.5f} | {row.max_drawdown:.4%} | {row.completed_cycles} | {row.win_rate:.2%} | {row.payoff:.5f} | {row.p_times_b:.5f} | {row.standard_expectancy_loss_units:.5f} | {row.average_full_year_cycles:.3f} |")
    lines += ["", "基础费用也完整保留。主政策较早CAGR2.5598%/Sharpe0.60680/pB0.91248；近期CAGR−0.0071%/Sharpe0.01367/pB0.78452。近期基础算术平均收益略正而复合年化略负，两者不混同。", "",
              "## 与具体上涨的连接", "",
              "主政策实际捕捉2024启动：9月24日形成REPRICING事件，9月25日开盘买入，9月30日根据已知周柱和融资确认晋为TREND，10月17日开盘按结构失效退出。压力费用后实际完成净回报12.1365%，净盈利3647.23568元。它是一个实际完成周期，不是用事后最低点/最高点成交，也不代表全年或全期优势。", "",
              "原17关键日及四原案例的三政策资格、次日请求、实际收盘份额、持有阶段和失效均已保存。原四案例240行在三政策下为720行，17日为51行；没发生买入和宏观未知均保留。", "",
              "## 信号、成本与阶段用途", "",
              "| 时期 | 保存实际数量毛损益 | 实际摩擦 | 净损益 | 加回摩擦的固定数量年化 |", "|---|---:|---:|---:|---:|"]
    for row in decompositions:
        lines.append(f"| {row['period']}压力 | {row['gross_pnl_cny']:.2f}元 | {row['friction_cny']:.2f}元 | {row['net_pnl_cny']:.2f}元 | {row['gross_no_friction_cagr_fixed_executed_quantities']:.4%} |")
    lines += ["", "无费只是保持所有已经成交的份额、时点和风险后的说明上界，不是另一策略。近期毛收益本身只有约0.3043%年化，即使不计显性摩擦也远低于原A实际扣费3.9908%。费用把微弱毛优势变成净亏损，但不能把问题仅归因于费用；信号与机会区分仍然不足。", "",
              "阶段持有相对同进入的固定退出，在两时期两费用四场景的CAGR和Sharpe点值均改善；多数配对95%区间仍跨零，不能宣称稳定性。较早压力对A年化增量约0.3500个百分点，20/252日区块区间均跨零；近期对A年化差约−4.2308个百分点，两个区间上界均负。全部比较和两种区块尺度见summary.json，不能只挑有利尺度。", "",
              "| 实际进入来源 | 时期 | 完成 | 赢/亏 | 完成净损益 | 实际pB |", "|---|---|---:|---:|---:|---:|"]
    for row in source_frame.itertuples(index=False):
        p_b = f"{row.p_times_b:.5f}" if pd.notna(row.p_times_b) else "UNKNOWN（无盈利笔，B不可估）"
        lines.append(f"| {row.source} | {row.period}压力 | {row.completed_cycles} | {row.wins}/{row.losses} | {row.completed_cycle_net_pnl:.2f}元 | {p_b} |")
    lines += ["", "近期回踩8笔全部亏损；早期重新定价来源也有显著亏损。宏观/阶段资格的净收益作用并未跨时期稳定。这里不据结果删除回踩、反转指标或保留成功年份救援；来源统计仅作解释。", "",
              "## 风险、频率与未平仓", ""]
    for row in chosen.loc[chosen.policy.eq(stage.POLICIES[0])].itertuples(index=False):
        lines.append(f"- {row.period}压力：平均暴露{row.mean_exposure:.2%}，最坏日{row.worst_day:.4%}，最坏完成周期{row.worst_trade:.4%}，最大盈利占全部完成盈利{row.largest_winner_fraction_of_all_wins:.2%}，最长空仓{row.longest_flat_sessions}日，双边原价成交额/初始资金{row.gross_turnover_over_initial_capital:.3f}，未平仓损益{row.open_pnl_cny:.2f}元，应收股息{row.unpaid_dividend_cny:.2f}元。")
    lines += ["", "较早27完成之外另有1个自然未平仓周期，盈利3784.16360元已进完整账户净值但未计完成胜率或B。近期无未平仓。周期回报与全账户回报不混合；风险减仓不算新增交易，2026部分年另列。", "",
              "## 研究决策与下一步", "",
              "接受：保留阶段持有优于本固定退出的历史局部点值与2024真实进出事实，继续研究不同信息角色与机会区分。拒绝：把当前三个手工阶段条件整体晋升为有效策略；四经济门、实际pB与稳定性均不满足。不得由一个成功行情重新制定资格或止损。", "",
              "下一优先方向是主线与510300的匹配及轮动/宽基扩散信息：先核对当时行业分类、成分权重与日线的完整来源合同，再研究资金和产业共识是否对宽基形成传导。该不同用途不更改本配置，也不依据近期8笔回踩亏损删路线。已有主线资料尚未取得金融准入，相关字段不能直接填入评分。", "",
              "独立验证NOT_ESTABLISHED，全项目历史搜索选择校正NOT_COMPUTED，首版NOT_CERTIFIED。当前目标active且未达；原13项前瞻与旧策略保持。", "",
              "可复算资料：[协议](protocol.json)、[全部16账户指标](results/十六完整账户_同资金风险成本比较.csv)、[逐年完整周期](results/全部逐年实际完成周期与账户收益.csv)、[实际来源](results/主政策全部实际进入来源_压力完整与未完成分开.csv)、[17关键日](results/原17关键日_三政策实际资格请求与持仓.csv)、[毛收益/摩擦](results/主政策实际数量毛收益与摩擦分解_无费仅解释.csv)、[全部区间和裁决](summary.json)。", ""]
    for figure in figures:
        relative=Path(figure).name
        lines.append(f"![{relative}](figures/{relative})")
        lines.append("")
    (OUT/"阶段点位_完整结果与失败归因.md").write_text("\n".join(lines),encoding="utf-8")
    study.write(OUT/"delivery_receipt.json", {"at":original.now(),"status":"SAVED_ACCOUNTS_RECOMPUTED_REPORT_AND_FIGURES_WRITTEN",
        "saved_accounts_recomputed":len(checks), "four_cases_three_policies_rows":sum(len(frame) for frame in case_rows),
        "seventeen_dates_three_policies_rows":sum(len(frame) for frame in key_rows), "figures":figures,
        "new_accounts":0,"new_fits":0,"new_labels":0,"goal_achieved":False})
    print("12保存账户指标与账本复算通过，51关键日/720案例行及4图已生成；没有新回测。",flush=True)


if __name__ == "__main__":
    main()
