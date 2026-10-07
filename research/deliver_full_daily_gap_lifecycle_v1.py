"""一次交付R178全体生命周期解释，并更新四长期事实，金融R177保持。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from research.full_daily_gap_lifecycle_attribution_v1 import OUT,FINANCIAL,EXPLANATION,STATE,PERIODS
from research.finalize_downtrend_break_state_v1 import FORWARD,DOCS
from research.point_first_passage_study_v1 import read,write_json,digest,now,require


def run():
    require(not (OUT/"delivery_receipt.json").exists() and not (OUT/"project_state_update_receipt.json").exists(),
            "R178全体交付或事实更新已经完成，不重复。")
    summary,protocol=read(OUT/"summary.json"),read(OUT/"protocol.json")
    require(summary["status"]=="COMPLETED_ALL_SAVED_GAP_LIFECYCLE_AND_EXIT_WEALTH_IDENTITIES",
            "全体生命周期和真实资金时钟未完成。")
    for source in protocol["sources"]:
        require(digest(ROOT/source["path"])==source["sha256"],"固定诊断输入已变化。")
    cycles=pd.read_parquet(OUT/"results/全部130周期_已知抬线回补及真实资金时钟.parquet")
    holding=pd.read_parquet(OUT/"results/全部持有原点_下沿更新回补与原决定精确核对.parquet")
    grouped=pd.read_parquet(OUT/"results/全部六组两时期两费用_最高账面与最终结果仅解释.parquet")
    aligned=pd.read_parquet(OUT/"results/全部97出生_原上涨段位置仅事后对齐.parquet")
    cases=pd.read_parquet(OUT/"results/四原案例全部相交压力周期_真实时钟与资金.parquet")
    quotes=pd.read_parquet(EXPLANATION/"results/全部3488已知缺口与量价_前2015A未知保留.parquet").set_index("date")
    original=pd.read_parquet(FINANCIAL/"results/全部实际进出点位_已知缺口量价及真实资金.parquet")
    require(len(cycles)==130 and len(cases)==11 and len(grouped)==24,"保存全体数量不同。")
    require(cycles.loc[cycles.status.ne("COMPLETE"),"actual_net_return"].isna().all(),"开放周期被填完成收益。")
    for row in cycles.itertuples(index=False):
        reference=original.loc[original.period.eq(row.period)&original.cost.eq(row.cost)&original.cycle_id.eq(row.cycle_id)]
        require(len(reference)==1,"诊断周期没有唯一原金融周期。")
        expected=reference.iloc[0]
        for key in ("entry_origin","entry_date","entry_quantity","status"):
            require(getattr(row,key)==expected[key],"原周期身份或库存改变。")
        if row.status=="COMPLETE":
            require(abs(row.actual_net_return-expected.net_return)<=1e-12 and abs(row.actual_net_pnl-expected.net_pnl)<=1e-8,
                    "诊断修改了原实际收益。")
    for scenario in summary["all_scenario_diagnostics"]:
        selected=cycles.loc[cycles.period.eq(scenario["period"])&cycles.cost.eq(scenario["cost"])&cycles.status.eq("COMPLETE")]
        reconstructed=float((selected.exit_origin_marked_pnl_cny+selected.final_open_gap_gross_cny+
                             selected.dividend_accrual_after_final_origin_cny-selected.final_sell_commission-selected.final_sell_slippage).sum())
        require(abs(reconstructed-selected.actual_net_pnl.sum())<=1e-6,"全组退出身份不符。")
        require(selected.cycle_observed_until.le(pd.Timestamp(PERIODS[scenario["period"]][1])).all(),"观察超过自身账户末日。")
    selected_cases=cases.loc[cases.entry_origin.isin(pd.to_datetime(["2015-07-10","2019-01-09","2019-04-01","2020-03-25","2020-06-01","2020-07-06","2024-09-25"]))]
    recent=next(row for row in summary["all_scenario_diagnostics"] if row["period"]=="2020_2026" and row["cost"]=="STRESS")
    metrics=pd.read_parquet(FINANCIAL/"results/完整账户共同口径比较.parquet")
    old_a=metrics.loc[metrics.period.eq("2020_2026")&metrics.cost.eq("STRESS")&metrics.policy.eq("A_SAVED_WEIGHT")].iloc[0]
    actual_gap_pnl=recent["completed_pnl_cny"]
    actual_a_difference=float(old_a.completed_cycle_net_pnl-actual_gap_pnl)
    next_boundary={
        "at":now(),"status":"ALL_FIXED_GAP_LIFECYCLE_DIAGNOSTICS_COMPLETE_NO_NEW_FINANCIAL_CANDIDATE",
        "latest_diagnostic":"TECH.R178","latest_actual_financial":"TECH.R177",
        "accepted_new_facts":"全130周期/1560持有决定/128退出财富身份完成。压力64完成中20曾真实收盘盈利但最终亏损，另26从未在持有收盘盈利；19周期只在抬高后的下沿回补，真实入场和持有时钟差异已解释。",
        "finite_rejected_explanations":["入场后出现收盘盈利就证明最后可赢","事后峰值可以直接用作可成交退出","从本全体诊断反选阶段/量/MACD/RV过滤","仅把原最终卖出的价差/摩擦当作全部对A差距来源"],
        "next_required_input":"先提出实质不同、当时可观察的日线点位信息及完整用途卡：对象、可知时钟、与旧表达和旧完整用途区别、全体失败的验证方法；核旧实际结果后才决定金融准入。真正独立新样本或真实来源/实现错误也可构成入口。不能把事后最高财富/底峰位置或已知赢家变成输入。",
        "closed_current_policy":"R177固定完整缺口全账户拒绝保持，不能改阈值/回补/成本/风险预算、加旧指标或混A营救；R173和全部旧终态保持。",
        "no_additional_same_diagnostic_split_planned":True,"new_admitted_unrun_candidates":0,
        "next_new_financial_experiment":"NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "new_accounts":0,"independent_validation":"NOT_ESTABLISHED","goal_achieved":False,
        "scope_limit":"本诊断及固定政策关闭不是全项目暂停或否定全部技术分析；原真实前瞻不改，历史重复研究不创造新样本。",
    }
    write_json(OUT/"next_information_admission_boundary.json",next_boundary,exclusive=True)
    lines=["# 全部完整缺口生命周期、真实资金和退出时钟","",
           "R178只读取R177四个保存账户，完成97出生、194资格所对应的130实际周期、1560持有决定和128完成退出的真实财富身份；没有新增金融账户、拟合、训练标签或采集。所有失败、已有持仓、取消及2开放周期保持。最高账面损益及原波段位置均为事后解释，不能作为当前交易输入。","",
           "## 具体上涨和失败的顺序","",
           "2019-01-09缺口在原事后上涨段底后5个交易日出现、距峰66日，时间位置约7.04%；初始下沿3.110元（当时原报价）。1月16日除息后原现金平移下沿换回原价为3.051元，经济下沿没有下降。2月25日和3月4日两个新缺口提高下沿，3月8日日低点回补，3月11日真实开盘卖出。真实持仓最高收盘账面盈利8889.04元，最终净6789.32元。4月1日另一缺口已在原同段底后58日、距峰13日，曾真实收盘盈利2500.42元，最终−1645.50元。时序说明两次信号处于不同位置，底峰位置仍然是事后标签。","",
           "2020-03-25缺口在原段底后2日、距峰72日出现，3月26日入场，4月7日/17日抬下沿，4月21日回补、22日退出，最终+1409.67元。6月1日缺口已在同段底后46日，6月8日新缺口将下沿提高，6月11日回补；该原点真实库存账面仍+591.17元，6月12日开盘价差−824.10元、卖出佣金31.82/滑点80.40元，最终−345.16元。回补时收盘仍在下沿上不等于未失效，也不保证下一开盘仍盈利。7月6日缺口在原段底后69日、距峰仅5日，次开追入后最高真实收盘账面仅+701.96元，8月20日才回补抬高后的线，21日出场−804.30元。不能把后来原波段38.54%总涨幅算给各次入点。","",
           "2024-09-25完整缺口到达后，09-26入场，09-27新缺口抬下沿；10月8日最高真实收盘账面+12765.11元，之后较长时间没有更高新完整缺口，下沿维持当时3.643元，允许价格回撤但未触线。2025年3月6日新缺口才再次抬高下沿至3.999元，3月10日回补、11日出场，最终+7004.21元。原9月25日日柱已正而DIF/上一周柱负，后续动量、量和波动先扩张再退潮可解释路径，但没有证明可事前识别10月8日就是最高资金日。","",
           "2015-07-10缺口实际属于原第19段的底后2日，而第18段06-29/30短反弹没有此信号。它与第18段固定案例窗相交所以仍展示，不混称同一上涨段。真实07-13买入后曾收盘+1316.64元，07-27回补初始线、28日卖出−3718.20元，未出现抬线。当天仍属于高波动修复，不能以出现缺口或曾盈利为稳定点位证明。","",
           "## 七个具体周期的真实时钟","",
           "| 出生原点 | 入场 | 回补确认 | 实际退出 | 抬线次数 | 最高真实收盘账面/元 | 退出前原点账面/元 | 次开价差/元 | 最终净利润/元 | 最终净收益率 |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in selected_cases.itertuples(index=False):
        lines.append(f"| {row.entry_origin:%Y-%m-%d} | {row.entry_date:%Y-%m-%d} | {row.first_known_low_fill_origin:%Y-%m-%d} | {row.exit_date:%Y-%m-%d} | {row.floor_raise_count} | {row.best_observed_holding_close_pnl_cny:.2f} | {row.exit_origin_marked_pnl_cny:.2f} | {row.final_open_gap_gross_cny:.2f} | {row.actual_net_pnl:.2f} | {row.actual_net_return:.2%} |")
    lines.extend(["","最高账面值使用真实逐日剩余份额、已实现卖出净现金、原买入支出和当时已发生股息应收，已经包含当时买入/减仓摩擦。它不是日内最高价成交、初始份额一直不减仓、等额资金或新增退出账户。各周期最高账面出现在不同日期，不能将其拼为策略收益或夏普。","",
                  "## 全体归因","",
                  "压力64已完成周期中，18个最后盈利、46个最后亏损。38个曾在真实持有收盘盈利，其中20个最后亏损；另外26个在持有收盘从未盈利，全部最后亏损。早期是8赢家/9曾盈利最终输/12未盈利输，近期10/11/14。BASE对应曾盈利最终输为10/11个，不只挑压力结果；六事后组×四场景共24单元均报告，空组均值未知。这个分组依赖后来实际结果，不能直接做入场/退出过滤。","",
                  "| 时期 | 费用 | 完成/开放 | 曾收盘盈利最终亏损 | 只有抬高线触及，初始线未触及 | 完成周期有抬线 | 最终次开价差/元 | 最终卖出摩擦/元 | 全部完成净利润/元 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in summary["all_scenario_diagnostics"]:
        lines.append(f"| {row['period']} | {row['cost']} | {row['completed']}/{row['open']} | {row['completed_positive_close_then_loss']} | {row['exit_on_raised_floor_only']} | {row['cycles_with_floor_raise']} | {row['final_open_gap_gross_cny']:.2f} | {row['final_exit_friction_cny']:.2f} | {row['completed_pnl_cny']:.2f} |")
    lines.extend(["",
                  "128完成周期的真实身份逐项核对：最终实际净利润−最终卖出前原点周期账面损益 = 原点剩余份额×(卖出日原开盘−原点收盘)+其后该周期已发生股息应收−最后卖出佣金−滑点。最大逐笔残差小于2e−11元，本次后续股息项均为0；股息资格由原登记日库存还原，未省略或事后回填。","",
                  f"近期STRESS最终开盘价差合计−3219.10元、最终卖出摩擦3400.12元，合计时钟差6619.22元；原A完成净利润{old_a.completed_cycle_net_pnl:.2f}元与缺口账户{actual_gap_pnl:.2f}元差{actual_a_difference:.2f}元。仅原最后一次卖出的这些项目不能单独说明全部对A差距。这是已有真实数量的财富差额，不是允许在回补当天收盘卖出的账户，不是任意新退出策略的收益上界；换规则会同时改变后续资金、入点和风险，不能据此声称能提高夏普。","",
                  "全部97出生与原61段底至峰对齐有87匹配、10不在任何原上升段，未知保留。原49正式上涨段不变；未挑最有利波段。位置比例是底至峰的交易日时间比例，不是已捕获涨幅，也不代表当时能知道未来峰。原四图用真实下沿及实际周期资金，图窗扩展到相交周期自身观察末日，原固定案例窗仍标出；11压力周期、所有失败同时展示。","",
                  "## 裁决与下一步","",
                  "接受全体已知下沿/回补/库存/现金/股息身份及具体早晚和退出差额解释。排除‘曾盈利就能稳定获利’、‘最高账面可当可成交退出’和‘原最终卖出的价差/摩擦解释全部对A差距’。这不证明任何新退出信号，也不证明量/MACD/RV可以区分赢家。R177固定完整政策拒绝、R173和全部旧终态不变；没有新的收益/夏普提升。","",
                  "本固定生命周期归因一次完成，不继续将同一结果切分成新过滤或阈值搜索。下一先提出实质不同、当时可观察的信息及完整用途卡，明确对象、可知时钟、与旧字段/旧用途差别、全体失败验证方法；核旧实际结果后才决定准入。真实新独立样本或真实来源/实现错误也是入口。当前无已准入待跑金融候选，不能把事后峰值、分段位置或已知赢家变成模型输入。原A/POINT真实前瞻十二状态和1008实际交易日口径保持。","",
                  "全部历史DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED，global DSR/PBO NOT_COMPUTED，去过拟合及提高完整收益夏普目标未达，目标active。最新技术解释R178、实际金融R177/登记R176、预测R158/原退出R145保持，其他分支不改。44冻结来源、0新必要测试/账户/拟合/训练标签/采集；本核对为解释所需的保存账本身份，不宣称新回测或外部评审。","",
                  "## 文件","",
                  "- [全部130真实周期及退出身份](results/全部130周期_已知抬线回补及真实资金时钟.csv)",
                  "- [全部1560持有原点的抬线与原决定核对](results/全部持有原点_下沿更新回补与原决定精确核对.csv)",
                  "- [逐日真实库存、现金流、股息应收与财富](results/全部周期逐日真实库存现金应收及收盘财富.csv)",
                  "- [全部97出生与原波段的事后位置](results/全部97出生_原上涨段位置仅事后对齐.csv)",
                  "- [六事后组全部24单元](results/全部六组两时期两费用_最高账面与最终结果仅解释.csv)",
                  "- [四原案例全部11压力周期](results/四原案例全部相交压力周期_真实时钟与资金.csv)",
                  "- [固定归因口径](protocol.json)、[全部结果](summary.json)、[下一信息准入边界](next_information_admission_boundary.json)",
                  "- [原R177完整金融结果](../510300_full_daily_gap_study_v1/研究结果与下一步.md)",""])
    for case_id in (18,37,42,55):
        lines.extend([f"![原案例{case_id}真实周期时钟](案例{case_id}_真实缺口生命周期与资金.png)",""])
    report=OUT/"研究结果与下一步.md"
    with report.open("x",encoding="utf-8",newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT/"delivery_receipt.json",{
        "at":now(),"status":"PASS_ALL_SAVED_LIFECYCLE_QUANTITY_DIVIDEND_AND_EXIT_IDENTITIES_DELIVERED",
        "technical_decision":"TECH.R178","actual_cycle_rows":130,"holding_origins_verified":1560,
        "completed_exit_identities_verified":128,"unchanged_actual_returns_checked":128,"open_cycles_unknown_preserved":2,
        "case_cycle_rows":11,"charts_viewed":4,"all_retrospective_group_cells":24,"source_freeze_count":44,
        "new_accounts":0,"new_model_fits":0,"new_training_labels":0,"new_market_requests":0,"necessary_new_tests":0,
        "latest_financial_strategy_decision_preserved":"TECH.R177","goal_achieved":False,
        "sources":[{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in [Path(__file__),OUT/"summary.json",report]],
    },exclusive=True)
    state=read(STATE)
    require(state["latest_technical_decision"]=="TECH.R177" and state["latest_financial_strategy_decision"]=="TECH.R177",
            "项目阶段不再是R177，不能覆盖另一阶段。")
    original_forward={key:state[key] for key in FORWARD}
    original_next=state["next_experiment"]
    original_trials=state["actual_candidate_trials"]
    write_json(OUT/"project_state_before.json",state,exclusive=True)
    marker="> 技术线当前事实 TECH.R178（优先于下方技术线历史快照，2026-10-05）："
    top=marker+(
        "全97出生/130周期、1560持有决定及128退出财富身份完成，0新账户。压力64完成中20曾真实收盘盈利最终亏损、26持有收盘从未盈利；19只在抬高后的线回补。"
        "2020-06-11回补原点账面+591.17元，次开价差−824.10元/最后卖出摩擦112.22元，最终−345.16元；最高账面及原波段位置仅事后解释。"
        "近期最后卖出时钟差6619.22元，对A净差46172.90元，不能仅解释全部差距或推新退出策略。"
        "固定诊断完成，不从分组救R177；金融R177/登记R176拒绝和原前瞻保持。下一不同信息及完整用途准入，待跑0；收益夏普/独立/去过拟合未达，目标active。"
        " [全体生命周期与具体上涨解释](../reports/research/510300_full_daily_gap_lifecycle_attribution_v1/研究结果与下一步.md)。\n\n")
    detailed="""
## TECH.R178：全体完整缺口生命周期与真实资金退出时钟（2026-10-05）

**假设**：R177交易次数增加而完整收益夏普不足，可能涉及入点在修复/上涨中途的不同位置、后续缺口抬失效线、低点回补及次开成交时钟，需要全体解释。

**验证方法**：固定44来源、一次只读97出生/194资格对应四保存账户130周期（128完成/2开放）。1560持有原点还原初始下沿、后续新完整缺口只抬线、已知日low回补及原决定库存/份额；逐日实际BUY/SELL现金、登记日真实份额的已发生股息应收与剩余份额乘原收盘还原1688财富行。128最终实际净利润−退出前原点账面损益=原点份额×(真实退出开盘−原点收盘)+后续该周期股息应收−最终卖出佣金−滑点，逐笔残差小于2e−11元。六事后组×四场景全部24单元，空组均值未知；全部97与原底至峰分段对齐87匹配/10无相交未知，无后验筛选。四原案例全部11压力周期及图已查看，原固定案例窗保留、图至相交周期自身观察末日；0新账户/测试/拟合/训练标签/采集。

**结果**：压力64完成周期18最后盈利/46最后亏损；38曾真实持有收盘盈利，其中20最后亏损，26持有收盘从未盈利且最后全亏。早8赢/9曾盈利后输/12未盈利输，近10/11/14；BASE曾盈利最终输早10/近11同时报告。压力完成周期有抬线早7/近13，其中只触抬高线且初始线未触早6/近13共19。2019-01-09在原底后5日/距峰66日，2次抬线、03-08回补/11出，最高真实收盘账面8889.04元、最终6789.32元；04-01原底后58日/距峰13日，曾2500.42元后亏1645.50元。原底峰位置仅事后时间标签，不是捕获涨幅或决策输入。

2020-03-25底后2日，04-07/17抬线、21回补/22出净1409.67元。06-01底后46日，06-08抬线/11回补；回补原点账面+591.16512元，次开价差−824.10元、最终佣金31.82232/滑点80.40，最终−345.15720元，显示次开可能翻转净符号。07-06底后69日/距峰5日，最高真实收盘账面701.96元，08-20回补/21出−804.30元。2024-09-25后09-27抬线，10-08真实最高账面12765.11元；下沿3.643元长期保持，2025-03-06新缺口升至3.999元，03-10回补/11出7004.21元。2015-07-10实际对齐第19段，和第18段固定窗相交不混称一段，曾1316.64元后亏3718.20元。

近期压力最终开盘价差−3219.10元/最后卖出摩擦3400.12元，共6619.22元；与原A完成净利润57847.49元对本规则11674.60元差46172.90元相比，原最后一次卖出的这些项目不能单独解释全部差距。较早压力相应−1695.10/2444.01元、完成净−2006.35元，开放周期不填完成收益。它们是已有真实库存财富身份，不是提前收盘成交账户或任意新退出策略的收益上界，也不计算另一夏普。

**为什么接受/拒绝**：接受全体可知时钟、库存/现金/股息和具体路径解释，排除“曾盈利即可稳定获利”“把事后最高账面当可成交价”“原最终卖出价差和摩擦是全部对A差距”等解释。没有接受可预测阶段或新的退出/过滤，不能按这些事后分组营救R177。R177四金融经济门/稳定败及原R173/所有旧终态保持。最高账面日由后来路径选择，底峰位置依赖事后分段，均不能冒充当时信息。历史DEVELOPMENT_CALIBRATION、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED；去过拟合与提高完整收益夏普目标未达，目标active/本轮progress/阻塞0。

**是否需要重新验证/下一步**：该固定全体诊断完成，不再将同一账本切分成参数或指标门；仅真实来源/实现错误复核并保留原证据。下一先提出实质不同、当时可观察的日线信息及完整用途卡：对象、可知时钟、与旧表达/用途区别、全体失败验证；核旧实际结果后才决定准入。真正独立新样本同样是入口。当前新金融用途未定义/登记、已准入待跑0；不能把峰值、分段位置或已知赢家作为输入。最新技术诊断R178，金融R177/登记R176、实际预测R158/原退出R145保持；原十二真实前瞻和1008实际交易日计划及其他分支不改。

依据：[全体生命周期与四例](../reports/research/510300_full_daily_gap_lifecycle_attribution_v1/研究结果与下一步.md)、[固定口径](../reports/research/510300_full_daily_gap_lifecycle_attribution_v1/protocol.json)、[实际全体身份](../reports/research/510300_full_daily_gap_lifecycle_attribution_v1/summary.json)、[全部130周期](../reports/research/510300_full_daily_gap_lifecycle_attribution_v1/results/全部130周期_已知抬线回补及真实资金时钟.csv)、[下一信息准入](../reports/research/510300_full_daily_gap_lifecycle_attribution_v1/next_information_admission_boundary.json)、[原R177金融](../reports/research/510300_full_daily_gap_study_v1/summary.json)。
"""
    for name in DOCS:
        path=ROOT/"docs"/name
        text=path.read_text(encoding="utf-8-sig")
        require(marker not in text,"事实文件已有R178，不重复追加。")
        first,remainder=text.split("\n",1)
        path.write_text(first+"\n\n"+top+remainder.lstrip("\n")+"\n"+detailed,encoding="utf-8",newline="\n")
    relative=OUT.relative_to(ROOT).as_posix()
    work={"scope":"TECH_R178_ONE_FIXED_ALL_SAVED_GAP_LIFECYCLE_AND_EXIT_IDENTITY_DIAGNOSTIC",
          "all_gap_births":97,"original_qualification_rows":194,"actual_cycle_rows":130,"completed_cycles":128,"open_cycles":2,
          "holding_decisions_verified":1560,"real_cycle_wealth_rows":1688,"exit_wealth_identities_verified":128,
          "pressure_completed_cycles":64,"pressure_positive_holding_close_final_losses":20,"retrospective_group_cells":24,
          "original_episodes":61,"original_admitted_waves":49,"case_pressure_cycle_rows":11,"charts_viewed":4,
          "frozen_sources":44,"necessary_new_tests":0,"new_accounts":0,"new_model_fits":0,"new_training_labels":0,"new_market_requests":0}
    next_action=next_boundary["next_required_input"]
    state.update({"at":now(),"updated_at":now(),"latest_completed_study":relative,"latest_result":relative+"/summary.json",
        "latest_report":relative+"/研究结果与下一步.md","latest_research_report":relative+"/研究结果与下一步.md",
        "latest_research_status":summary["status"],"latest_progress":top,"latest_overall_summary":top,"latest_continuation_outcome":top,
        "current_study":"510300_FULL_DAILY_GAP_SAVED_LIFECYCLE_ATTRIBUTION_V1",
        "current_phase":"ALL_SAVED_FULL_GAP_LIFECYCLE_AND_EXIT_IDENTITIES_COMPLETE_FINANCIAL_R177_PRESERVED",
        "current_direction":"具体上涨、全部失败及下沿/回补/次开真实财富解释完成；事后峰值/位置不可用作输入，固定政策不营救。",
        "current_priority":next_action,"next_research_action":next_action,"next_available_action":next_action,
        "next_research_plan":relative+"/next_information_admission_boundary.json",
        "next_experiment_status":"DISTINCT_KNOWN_INFORMATION_PURPOSE_OR_TRUE_NEW_SAMPLE_REQUIRED_NO_READY_FINANCIAL_POLICY",
        "next_candidate_field_status":"NO_NEW_FIELD_OR_FINANCIAL_POLICY_ADMITTED_AFTER_ALL_LIFECYCLE_DIAGNOSTICS",
        "next_financial_experiment":"NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status":"NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY","current_admitted_unrun_numeric_candidates":0,
        "latest_technical_decision":"TECH.R178","latest_actual_model_decision":"TECH.R177",
        "latest_financial_strategy_decision":"TECH.R177","latest_registration_decision":"TECH.R176",
        "latest_financial_registration_decision":"TECH.R176","latest_actual_prediction_model_decision":"TECH.R158",
        "latest_original_exit_decision":"TECH.R145","latest_continuation_receipt":relative+"/delivery_receipt.json",
        "latest_full_gap_lifecycle_attribution":relative+"/summary.json",
        "latest_full_gap_all_lifecycle_points":relative+"/results/全部130周期_已知抬线回补及真实资金时钟.csv",
        "latest_new_information_admission_boundary":relative+"/next_information_admission_boundary.json",
        "current_phase_trial_accounting":work,"current_goal_turn_actual_work":work,
        "current_phase_source_freeze_count":44,"current_phase_known_daily_rows":3488,
        "current_phase_information_scope":"ALL_SAVED_FULL_GAP_LIFECYCLE_REAL_INVENTORY_AND_CLOSE_WEALTH_EXPLANATION_ONLY",
        "actual_candidate_trials_role":"LATEST_FINANCIAL_R177_TRIAL_ACCOUNTING_PRESERVED_NOT_R178_DIAGNOSTIC",
        "new_accounts_this_continuation":0,"new_accounts_in_current_phase":0,"new_primary_accounts_this_continuation":0,
        "new_financial_candidate_accounts_this_continuation":0,"new_investment_account_evaluations_this_continuation":0,
        "internal_reference_replays_this_continuation":0,"saved_account_controls_replayed_this_continuation":0,
        "saved_accounts_checked_this_continuation":4,"necessary_tests_passed_this_continuation":0,"necessary_tests_passed_in_current_phase":0,
        "actual_prefix_checks_this_continuation":0,"new_model_fits_this_continuation":0,"historical_model_refits_this_continuation":0,
        "new_return_labels_this_continuation":0,"new_training_labels_this_continuation":0,"new_model_training_labels_this_continuation":0,
        "new_market_requests_this_continuation":0,"new_financial_result_computed_this_continuation":False,
        "new_allocation_method_admitted":False,"new_method_admitted":"SAVED_FULL_GAP_LIFECYCLE_DIAGNOSTIC_ONLY_NO_NEW_POLICY",
        "new_market_information_admitted":False,"current_fields_admitted_this_continuation":0,"current_information_mechanisms_admitted_this_continuation":0,
        "returns_and_sharpe_improved":False,"return_and_sharpe_improved_this_continuation":False,
        "original_strategy_source_files_changed_this_continuation":0,
        "code_files_added_this_continuation":["research/full_daily_gap_lifecycle_attribution_v1.py","research/deliver_full_daily_gap_lifecycle_v1.py"],
        "previous_goal_turn_classification":"progress","current_goal_turn_classification":"progress",
        "goal_turn_progress_classification":"NEW_ALL_SAVED_LIFECYCLE_AND_ACTUAL_EXIT_WEALTH_EVIDENCE_NO_PARAMETER_RESCUE",
        "current_goal_turn_classification_reason":"全97/130/1560/128真实身份明确20曾盈利后亏、19仅触抬高线、次开符号翻转及全部原波段位置，排除事后峰值/仅卖出差额解释，改变下一信息准入入口。",
        "goal_status":"active","goal_tool_status_confirmed":"active","goal_achieved":False,
        "blocked_audit_count":0,"consecutive_blocked_goal_turns":0,"blocked_reason":None,"blocked_audit_key":None,"blocking_decision":None,
        "whole_model_overfitting_removed":False,"overfitting_removed":False,"overfit_removed":False,
        "independent_validation_status":"NOT_ESTABLISHED","global_DSR_PBO":"NOT_COMPUTED",
        "current_unmet_evidence":"R177四金融经济门/稳定失败保持；生命周期诊断不能生成事前预测或新高夏普策略，独立/去过拟合未建立，无新待跑金融候选。",
        "validation_method_this_continuation":"一次44来源，只读四保存账本，97出生对齐/130周期实际收益保持/1560持有决定/1688真实财富/128退出身份/24单元/11案例四图完成，0账户重跑或新测试。"})
    require({key:state[key] for key in FORWARD}==original_forward,"十二真实前瞻改变。")
    require(state["next_experiment"]==original_next and state["actual_candidate_trials"]==original_trials,
            "原1008日计划或金融R177试验账改变。")
    write_json(STATE,state)
    write_json(OUT/"project_state_update_receipt.json",{
        "at":now(),"status":"PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_diagnostic":"TECH.R178","latest_financial_preserved":"TECH.R177","registration_preserved":"TECH.R176",
        "actual_prediction_preserved":"TECH.R158","original_exit_preserved":"TECH.R145",
        "forward_values_unchanged":original_forward,"existing_forward_next_experiment_unchanged":original_next,
        "actual_financial_trial_accounting_unchanged":original_trials,"goal_status":"active","goal_achieved":False,
        "new_accounts":0,"new_admitted_unrun_candidates":0,
        "sources":[{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in [Path(__file__),STATE,*(ROOT/"docs"/name for name in DOCS)]],
    },exclusive=True)
    print("R178全体生命周期已交付，四长期事实一次更新；金融R177及原前瞻保持，目标active。",flush=True)


if __name__=="__main__":
    run()
