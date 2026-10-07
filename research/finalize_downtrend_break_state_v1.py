"""一次更新首次下降结构突破的四份长期事实与当前状态。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.downtrend_break_study_v1 import OUT, EXPLANATION, PRIMARY
from research.point_first_passage_study_v1 import read, write_json, digest, now, require

STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD = (
    "forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
    "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at",
    "current_validated_candidates", "forward_account_comparison_protocol", "latest_forward_account_check",
    "new_prospective_sessions_this_continuation", "new_prospective_cycles_this_continuation",
)
DOCS = ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md")


def run():
    require(not (OUT / "project_state_update_receipt.json").exists(), "本轮长期事实已经更新，不重复写入。")
    summary, explanation, delivery = read(OUT / "summary.json"), read(EXPLANATION / "summary.json"), read(OUT / "delivery_receipt.json")
    require(summary["technical_decision"] == "TECH.R173" and summary["new_primary_accounts"] == 4,
            "本轮四新完整账户未完成。")
    require(delivery["all_four_actual_point_quality_passed"] and delivery["individual_economic_gate_passes"] == 1,
            "实际点位质量或场景通过数量与交付不一致。")
    require(delivery["open_case_windows_clipped_at_own_account_end"], "案例观察末日未按各账户截断。")
    for path in (OUT / "protocol.json", EXPLANATION / "protocol.json"):
        for source in read(path)["sources"]:
            require(digest(ROOT / source["path"]) == source["sha256"], "首次突破冻结来源发生变化。")
    state = read(STATE)
    forward_before = {key: state[key] for key in FORWARD}
    next_forward_before = state["next_experiment"]
    texts = {name: (ROOT / "docs" / name).read_text(encoding="utf-8-sig") for name in DOCS}
    marker = "> 技术线当前事实 TECH.R173（优先于下方技术线历史快照，2026-10-05）："
    for text in texts.values():
        require(marker not in text, "事实文件已经包含本轮，不重复追加。")
    write_json(OUT / "project_state_before.json", state, exclusive=True)
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    primary = metrics.loc[metrics.policy.eq(PRIMARY)]
    earlier = primary.loc[primary.period.eq("2015_2019") & primary.cost.eq("STRESS")].iloc[0]
    recent = primary.loc[primary.period.eq("2020_2026") & primary.cost.eq("STRESS")].iloc[0]
    overview = (
        "按先解释具体上涨/失败的量价和指标、再反推当时可知点位，完成TECH.R171全体首次下降结构突破解释、R172唯一完整政策登记与R173四实际账户。"
        "原严格收盘2/2确认不变，上一已知双高双低下降中的最后高点被收盘首次跨越，次开尝试，被突破旧高为初始失效、持有只上移至确认低点。"
        "66资格原点，每费用53真实周期（52完成/1开放）、11次开取消/2已有持仓，132资格及106实际周期完整交付；原61/49段、旧2846/2826/20标签及9尾部保持。"
        "四场景实际pB均>1、平均净收益正：压力较早胜率25%/B4.245752/pB1.061438、净年化1.890622%/夏普0.610192/DD7.0600%；"
        "近期18.75%/B5.645901/pB1.058607、0.112558%/0.056596/DD6.9612%。"
        "只有较早STRESS通过单场景经济门，整体四场景准入和历史稳定未过，近期收益夏普远低于原A，固定完整政策拒绝关闭。"
        "四输入/四账户测试、四行情前缀、八保存控制精确复现与四主账户一次运行/保存核对完成，初次dtype/浮点表示失败保留并在结果前修复，41解释/81金融来源保持。"
        "四资金恒等式保存；近期STRESS均值项6647.31及资金收益对应项−5181.32还原净1465.99元，费用5227.61元；不能推等额配仓或挑赢家。"
        "最新金融R173、登记R172、预测R158/原退出R145保持，独立/去过拟合未建立、global DSR/PBO未计算、完整目标未达；目标active/本轮progress/阻塞0。"
        "下一仅全66信号原预算与实际减仓归因，无新账户或策略准入；原十二真实前瞻值及既定1008实际交易日计划、其他分支保持。"
    )
    top = marker + (
        "已完成R171解释/R172登记/R173四账户。首次下降结构突破的四场景实际pB均>1；压力较早净年化1.8906%/夏普0.6102，近期0.1126%/0.0566。"
        "只有较早STRESS单场景通过，整体和稳定门未过，固定政策关闭；点位质量为开发事实，完整收益夏普/独立/去过拟合目标未达。"
        "132资格/106周期及全部失败保留；下一全66信号原资金归因，不增加账户策略，原A/所有旧终态/真实前瞻保持。"
        " [具体上涨解释与完整结果](../reports/research/510300_downtrend_break_study_v1/研究结果与下一步.md)。\n\n"
    )
    technical_state = """
## TECH.R171—R173：具体上涨、首次下降结构突破与完整账户（2026-10-05）

当前目标继续是510300.SH日线及上一完整周的可提前识别多头点位，真实交易净p×B>1且标准期望pB−q>0，并在原20万元、成本及风险预算下提高全日历收益与夏普，次数为软目标。先解释具体上涨和失败，再固定完整入出规则；不使用分钟、其他资产交易或期权收益。

原严格日收盘2/2确认及因果交替枢轴保持。上一原点两高两低已确认且均下降，上一收盘<=同一旧H2、当前>H2为首次突破；同锚只首次。空仓次真实开一次尝试，开盘现金平移价<=旧H2取消，不追入。初始失效为旧H2，持有按每个已知原点max(旧失效,最新确认低点)只上移，收盘<=线次合法开退出；未知不迫使卖出、风险只减，无加仓、A混合、固定2R或20日。该完整用途在连接旧结果标签前固定，不能把原退出MSE/115成员门误用于此规则；原退出R145及预测R158终态不变。

2019-01-18突破早于01-23双抬升，压力01-21入/03-11出净+14.79%；03-18另一次突破03-19入/03-26出净−2.94%同样保留。2020-05-29当日柱/上一周柱仍负，06-01入/07-27出净+17.36%，不声称比旧04-01/04-27结构统一早。2024-09-24放量3.338、日柱正而DIF/上一周柱负，突破早在08-28确认的旧高，09-25入/10-31出净+13.25%；旧完整抬升至10-21才成立。该日A库存0但已知目标31.92%，不能说A无信号。2015-06-29/30放量反弹没有突破原已知下降高，整个固定案例窗没有新的合格点。原61段/49正式波段、3488状态/941确认、2855当期原点、66首次突破/2同锚重复及66收盘失效路径完整保留。

有限旧完整用途实际核对：简单量反转六候选COMPLETED_TARGET_NOT_MET；日线供给测试REJECTED_FROZEN_NO_PARAMETER_RESCUE；图谱/海龟实际reports/backtest/graph_regime_martin_turtle_v2.json的六轨道4 FAIL/2 INSUFFICIENT_EVIDENCE、registered_pass_count=0，证据HISTORICALLY_CONTAMINATED；快速structure实际是IF-OI/ETF份额迁移，worth_followup=[]，不等同价格枢轴。不是全项目穷尽查重，更不重启旧失败；原图谱成本门没有据未核结果另作终态判断。R170完整双抬升失败保持。

R172原20万元/252日/现金0、两时期两费用/50%上限、原ES/跳空/DD、T+1/100份/.001/股息口径不变。四必要输入及四账户测试最终通过、四实际前缀精确一致、八A/纯价格日账/订单/周期精确复现；41解释及81金融来源冻结，四主账户一次运行、保存指标与资金/库存/T+1核对PASS。初次字符串dtype和开盘现金平移浮点表示失败原源码/输出保留：仅显式nullable字符串及open+已发生累计分红，断言/规则不变，修复时新主金融结果读取0。

实际四场景完成质量均正：早期BASE净年化2.0719%/夏普0.6580/pB1.1483，STRESS1.8906%/0.6102/pB1.0614；近期BASE0.3309%/0.1327/pB1.1844，STRESS0.1126%/0.0566/pB1.0586。压力胜率早25%/B4.2458，近18.75%/B5.6459，DD7.0600%/6.9612%；只有早期STRESS单场景经济门通过，早期BASE年化略低A，近期两费用均远低A（压力A3.9908%/1.2169）。全四场景共同经济门及全部比较历史稳定门未过，R173固定完整政策拒绝关闭；不能误写四个单独场景都失败。全部20/252日配对区块各2000次、固定seed510300154及两比较区间保存，仍属开发描述。

每费用66资格=53真实周期（52完成/1早期开放）+11次开失效取消+2已有持仓不另生周期。全部132资格/106周期，104完成/2开放；未成交不填收益，开放不人工清仓/填入完成胜率，案例只到自身账户末日。压力完整年均次数4.0/4.6667，原A4.4/4.1667；不说两个时期次数都提高，也不把同样近期32完成周期称32新增机会。原2846旧标签/2826成熟/20删失和9尾部无新标签不变，旧20日66事件pB约0.494不是该动态失效政策收益。

全部四场景完成利润恒等拆解ΣQr=ΣQ×平均r+Σ[(Q−平均Q)(r−平均r)]，Q实际买入支出。近期压力6647.31−5181.32=1465.99元，毛6693.60/佣金1458.71/滑点3768.90元；平均资金暴露5.1342%，现金日纳入夏普。早期压力7688.01+9217.25=16905.27元完成净利润，账户净增18966.48元，开放差额保留。早期最大赢家2015-02-12净21495.81元、近期2020-05-29净10502.89元均超过所属全部完成净利润，只有5/6个赢家；这是全体事后集中度事实，不是赢家过滤、等额或放大仓位策略。

接受因果时钟、具体案例的提前/滞后解释和四场景交易质量点估计；拒绝固定政策作为完整收益/夏普改进，不加MACD/量/RV门、调窗口/退出/费用/时期/预算或混A营救。全部历史DEVELOPMENT_CALIBRATION，来源first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED；去过拟合与完整目标未达。目标active、本轮progress、阻塞0，最新金融R173/登记R172，原预测R158/退出R145及十二真实前瞻值不变。

下一具体实验：只归因全部66已知信号的原计划份额、次开现金/风险可买上限、ES/负10%跳空/剩余回撤预算和持有减仓；两费用全部成交/取消/已有持仓/失败/开放保留，原A库存和已知目标分别解释。四资金恒等式已有，不能把旧R166/R167预算机制重说未试；本次是针对R173新结果的解释，无新账户/拟合/标签/采集，完整新策略未定义或准入、待跑候选0。已有真实A/POINT前瞻1008实际交易日协议和最早2026-10-08 15:05资格新收盘保持，其他宏观/盘口分支保持。

依据：[结果与全部点位](../reports/research/510300_downtrend_break_study_v1/研究结果与下一步.md)、[R171解释](../reports/research/510300_downtrend_break_explanation_v1/summary.json)、[R172冻结金融定义](../reports/research/510300_downtrend_break_study_v1/protocol.json)、[R173实际结果及全部区间](../reports/research/510300_downtrend_break_study_v1/summary.json)、[保存核对](../reports/research/510300_downtrend_break_study_v1/saved_result_verification.json)、[下一全体资金归因](../reports/research/510300_downtrend_break_study_v1/next_all_signal_budget_attribution_proposal.json)。
"""
    decisions = """
## TECH.R171：具体上涨及失败的首次下降结构突破解释（2026-10-05）

**假设**：完整双高双低抬升确认可能错过急涨，上一原点已确认下降结构的最后高点首次被收盘突破可以有不同的到达时钟，需同时解释慢涨、急涨与失败。

**验证方法**：复用原严格收盘2/2、因果交替组件及已发生分红平移；引用上一原点已确认双高双低均下降中的H2，上收盘<=同H2、当前>H2且同锚首次。未来完整次开/旧高初始失效/只上移确认低点用途在接旧标签前固定。核旧量反转、日供给、图谱海龟实际终态及快速structure真实信息用途。四输入测试和四实际前缀通过后冻结41来源，解释全体3488/2855状态、941确认、66突破/2同锚重复、66收盘失效、原61/49段与四案例。0新账户/拟合/训练标签/采集。

**结果**：2019-01-18早于旧双抬升01-23；2024-09-24早于10-21且可提前到主升段；2020只到05-29，不统一早于旧04-01/04-27，且日柱/上一周柱仍负；2015反弹全案例窗未形成合格跨越。49正式段有27段确认后至高点没有新突破。旧20日66事件描述pB约0.494、均值−0.1587%，全部旧标签/删失/尾部保留，不作为动态退出策略收益门。初次字符串dtype表示失败保存，仅显式dtype修复，原断言不改。旧图谱实际六轨4 FAIL/2 INSUFFICIENT_EVIDENCE，快速OI/ETF信息用途不等同价格枢轴。

**为什么接受/拒绝**：接受当时确认时钟、提前与滞后的具体解释和完整负例；没有接受指标预测、因果上涨或策略优越。旧用途有限查重不代表全局新颖，也不能复活终态。完整金融检验另由R172固定/R173执行。

**是否需要重新验证**：固定解释已完成，只因真实来源/实现错误复核，不变窗口或据后验收益选过滤。金融登记已经履行，独立样本尚未建立。

## TECH.R172：唯一首次下降结构突破完整账户用途登记（2026-10-05）

**假设**：首次跨越已知下降H2、用被突破旧高及只上移确认低点管理失效，可能将修复/急涨点位转化为完整账户收益夏普。

**验证方法**：空仓次真实开一次尝试，开盘平移价<=旧H2取消不追；持有floor=max(旧floor,新已确认低点)，已知收盘<=floor次合法开退出，未知不退出、风险只减，无加仓/A/2R/20日。四输入和四账户行为测试、四实际前缀及八原控制精确复现；81金融来源在四主账户结果前冻结。原20万元、两时期/两费用/252日/现金0/50%/ES/跳空/DD/T+1/100份/.001/股息、期末和比较不变。全部四场景净CAGR和全日历夏普>0且同时优于A与纯价格、DD<=10%、真实pB>1和pB−q>0，次数软目标；20/252日各2000固定配对区块、seed510300154全比较。初次等于旧H2开盘的浮点表示失败保存，只将open+此前已发生累计分红相加，断言与规则不改，金融结果读取0。

**结果**：一固定新政策/四主账户计划，四新账户已由R173一次完成，八控制仅复用保存数据。0拟合/新训练标签/采集，登记不再是待跑候选。

**为什么接受/拒绝**：接受一次不同完整信息用途的开发测量，不接受收益优势或独立样本资格，不把原退出MSE/成员门误用于规则政策。原R145/R158/R170及所有旧终态保持，不改预算或回填中心日。

**是否需要重新验证**：无需重登记/重跑，金融结论见R173；登记不授权实盘、其他资产交易、分钟或期权收益。

## TECH.R173：四场景交易质量为正，完整账户政策拒绝并关闭（2026-10-05）

**假设**：R172固定完整用途须在全部时期/费用中同时改进净收益、全日历夏普和真实交易质量，不能只依靠解释好的案例。

**验证方法**：四主账户一次运行、八原控制复用；保存指标与资金/库存/T+1/成本/股息核对PASS。132资格及106周期全部记录：104完成/2开放，每费用53实际周期+11次开取消+2已有持仓=66资格；不为未成交填收益或期末人工结算。四案例按各账户自身观察末日相交，原A库存与已知目标分别记录。四场景全部资金净利润恒等式只从已保存Q/r拆解，无额外账户或收益反事实。

**结果**：四费用时期完成pB均>1/标准期望>0/平均净收益>0。压力较早20完成/1开放，胜率25%、B4.245752、pB1.061438，净年化1.890622%/夏普0.610192/DD7.0600%；近期32完成，18.75%/B5.645901/pB1.058607，0.112558%/0.056596/DD6.9612%。BASE早2.0719%/0.6580/pB1.1483、近0.3309%/0.1327/pB1.1844。只有早期STRESS单场景通过，早期BASE年化略低A，近期两费用显著低A；不能说四个各自都失败，整体全四经济门与历史稳定门均未过。压力完整年均4.0/4.6667，原A4.4/4.1667，次数不是两个时期都增。

压力实际2019-01-21至03-11净+14.79%、03-19至03-26净−2.94%、2020-06-01至07-27净+17.36%、2024-09-25至10-31净+13.25%与全部其他失败同时保留。2024信号日A库存0但已知目标31.92%，不是A完全漏掉。近期压力毛6693.60元/摩擦5227.61元、完成净1465.99元；资金均值项6647.31元加对应项−5181.32元还原，不能称等额账户。早最大赢家2015-02-12净21495.81元超过全体完成16905.27元，近最大2020-05-29净10502.89元超过1465.99元，赢家仅5/6个。

**为什么接受/拒绝**：接受四场景点位质量及具体早晚解释的开发事实；拒绝固定完整政策作为收益/夏普改进，关闭保存全部失败。不得从结果加MACD/量/RV门、换窗口/退出/费用/时期/预算或混A营救。first-vintage未认证、全部历史开发复用、独立NOT_ESTABLISHED/global DSR/PBO NOT_COMPUTED，去过拟合和完整目标仍未达。原A、全部其他终态及真实前瞻不变。

**是否需要重新验证/下一步**：不在同数据上参数救援或重跑，只遇真实实现/来源错误、实质新信息或真正新样本才独立另验并保留此终态。下一只解释全部66资格的原资金预算/开盘可买/持有减仓及A覆盖，两费用所有取消/已有持仓/失败/开放保留；旧R166/R167及配仓失败不重启，新金融政策未定义准入、待跑0。目标active、本轮progress/阻塞0；最新金融R173/登记R172、预测R158/退出R145及十二真实前瞻值保持。

依据：[完整结果与全部点位](../reports/research/510300_downtrend_break_study_v1/研究结果与下一步.md)、[固定解释](../reports/research/510300_downtrend_break_explanation_v1/protocol.json)、[金融冻结定义](../reports/research/510300_downtrend_break_study_v1/protocol.json)、[实际四账户和全部区间](../reports/research/510300_downtrend_break_study_v1/summary.json)、[下一全信号资金归因](../reports/research/510300_downtrend_break_study_v1/next_all_signal_budget_attribution_proposal.json)。
"""
    routing = "\n## 技术线TECH.R171—R173当前更新（2026-10-05）\n\n" + overview + "\n\n共享宏观/盘口分支的独立范围与终态保持；下面三项按假设→验证→结果→处置→重验记录，不依赖聊天。\n"
    for name, text in texts.items():
        first, remainder = text.split("\n", 1)
        append = decisions if name in ("RESEARCH_DECISIONS.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md") else technical_state if name == "PROJECT_STATE_TECHNICAL_LINE.md" else routing
        (ROOT / "docs" / name).write_text(first + "\n\n" + top + remainder.lstrip("\n") + "\n" + append,
                                         encoding="utf-8", newline="\n")
    out_relative, explanation_relative = OUT.relative_to(ROOT).as_posix(), EXPLANATION.relative_to(ROOT).as_posix()
    next_action = "只归因全66合格信号的原计划份额/次开可买/ES/跳空/DD及持有减仓与A库存/已知目标，两费用全部失败取消开放保留；不加过滤/混A/放宽预算营救R173，无新金融策略准入。"
    accounting = {
        "scope": "TECH_R171_R172_R173_ONE_FIXED_DOWNTREND_BREAK_COMPLETE_POLICY",
        "candidate_configurations": 1, "new_policy_accounts": 4, "primary_accounts": 4,
        "saved_original_A_replays": 4, "saved_price_control_replays": 4, "reused_control_scenarios": 8,
        "all_known_daily_rows": 3488, "origins_since_2015": 2855, "confirmed_arrivals": 941,
        "first_break_events": 66, "same_anchor_repeat_crosses_not_admitted": 2,
        "all_qualification_rows_two_costs": 132, "all_actual_cycle_rows_two_costs": 106,
        "actual_cycles_per_cost": 53, "completed_per_cost": 52, "open_per_cost": 1,
        "open_cancelled_per_cost": 11, "already_holding_no_new_cycle_per_cost": 2,
        "original_episode_rows": 61, "original_admitted_waves": 49,
        "necessary_tests_final_passed": 8, "pre_registration_initial_failures_preserved": 2,
        "actual_prefix_checks": 4, "charts": 4, "explanation_frozen_sources": 41, "financial_frozen_sources": 81,
        "capital_identity_cells": 4, "all_four_point_quality_passed": True, "single_scenario_economic_passes": 1,
        "overall_economic_gate_passed": False, "historical_stability_gate_passed": False,
        "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0, "risk_budget_changes": 0,
    }
    added = [
        "research/downtrend_break_inputs_v1.py", "research/downtrend_break_explanation_v1.py",
        "research/downtrend_break_account_v1.py", "research/downtrend_break_study_v1.py",
        "research/deliver_downtrend_break_results_v1.py", "research/finalize_downtrend_break_state_v1.py",
        "tests/test_downtrend_break_inputs_v1.py", "tests/test_downtrend_break_account_v1.py",
    ]
    state.update({
        "at": now(), "updated_at": now(), "latest_completed_study": out_relative,
        "latest_result": out_relative + "/summary.json", "latest_report": out_relative + "/研究结果与下一步.md",
        "latest_research_report": out_relative + "/研究结果与下一步.md", "latest_research_status": summary["status"],
        "latest_progress": overview, "latest_overall_summary": overview, "latest_continuation_outcome": overview,
        "current_study": "510300_DOWNTREND_BREAK_STUDY_V1",
        "current_phase": "DOWNTREND_BREAK_POINT_QUALITY_POSITIVE_FULL_ACCOUNT_POLICY_REJECTED",
        "current_direction": "首次下降结构突破全体解释及完整账户完成，点估计正但整体经济和稳定未过；下一全信号原资金归因。",
        "current_priority": next_action, "next_research_action": next_action, "next_available_action": next_action,
        "next_research_plan": out_relative + "/next_all_signal_budget_attribution_proposal.json",
        "next_experiment_status": "ALL_SIGNAL_SAVED_BUDGET_ATTRIBUTION_PROPOSED_NO_NEW_POLICY",
        "next_candidate_field_status": "NO_NEW_ADMITTED_STRATEGY_AFTER_FIXED_FULL_ACCOUNT_REJECTION",
        "next_financial_experiment": "NOT_DEFINED_OR_REGISTERED_NO_READY_CANDIDATE",
        "next_strategy_increment_status": "NO_NEW_ADMITTED_UNRUN_NUMERIC_STRATEGY", "current_admitted_unrun_numeric_candidates": 0,
        "latest_technical_decision": "TECH.R173", "latest_actual_model_decision": "TECH.R173",
        "latest_registration_decision": "TECH.R172", "latest_financial_registration_decision": "TECH.R172",
        "latest_financial_strategy_decision": "TECH.R173", "latest_financial_strategy_result": out_relative + "/summary.json",
        "new_account_return_sharpe_result": out_relative + "/summary.json",
        "latest_actual_model_kind": "COMPLETE_RULE_POLICY_WITHOUT_ESTIMATED_PREDICTION_MODEL",
        "latest_actual_prediction_model_decision": "TECH.R158", "latest_original_exit_decision": "TECH.R145",
        "latest_continuation_receipt": out_relative + "/delivery_receipt.json",
        "latest_downtrend_break_explanation": explanation_relative + "/summary.json",
        "latest_downtrend_break_protocol": out_relative + "/protocol.json",
        "latest_downtrend_break_saved_verification": out_relative + "/saved_result_verification.json",
        "latest_downtrend_break_actual_points": out_relative + "/results/全部实际进出点位_已知下降结构量价及资金.csv",
        "latest_all_population_capital_identity": out_relative + "/results/全部四场景完成周期_实际资金恒等式.csv",
        "economic_stage_status": "FIXED_POLICY_REJECTED_OVERALL_ONE_OF_FOUR_SCENARIO_GATES_PASSED",
        "current_full_account_economic_gate_passed": False, "current_historical_stability_gate_passed": False,
        "current_all_four_actual_point_quality_passed": True, "current_point_quality_evidence_role": "ARCHIVED_DEVELOPMENT_ESTIMATE_NOT_STRATEGY_PROMOTION",
        "current_individual_economic_gate_passes": 1, "returns_and_sharpe_improved": False,
        "return_and_sharpe_improved_this_continuation": False, "new_financial_result_computed_this_continuation": True,
        "actual_candidate_trials": accounting, "current_phase_trial_accounting": accounting,
        "current_goal_turn_actual_work": accounting, "actual_candidate_trials_role": "LATEST_FINANCIAL_R173_TRIAL_ACCOUNTING",
        "current_executed_unique_new_entry_events": 53, "current_structural_signal_events": 66,
        "current_trade_map_cycles": 53, "current_trade_map_complete_cycles": 52, "current_trade_map_open_cycles": 1,
        "current_trade_map_scope": "ONE_COST_FULL_POPULATION_SAVED_R173_PRIMARY_ACCOUNTS",
        "current_new_policy_complete_cycles": 52, "current_new_policy_open_cycles": 1,
        "current_new_policy_unexecuted_origins": 13,
        "necessary_tests_passed_this_continuation": 8, "necessary_tests_passed_in_current_phase": 8,
        "actual_prefix_checks_this_continuation": 4, "new_accounts_this_continuation": 4, "new_accounts_in_current_phase": 4,
        "new_primary_accounts_this_continuation": 4, "new_financial_candidate_accounts_this_continuation": 4,
        "new_investment_account_evaluations_this_continuation": 4, "internal_reference_replays_this_continuation": 8,
        "saved_account_controls_replayed_this_continuation": 8, "saved_accounts_checked_this_continuation": 12,
        "new_model_fits_this_continuation": 0, "historical_model_refits_this_continuation": 0,
        "new_return_labels_this_continuation": 0, "new_training_labels_this_continuation": 0,
        "new_model_training_labels_this_continuation": 0,
        "new_allocation_method_admitted": False,
        "new_method_admitted": "ONE_FIXED_DOWNTREND_BREAK_COMPLETE_POLICY_EXECUTED_AND_REJECTED",
        "new_market_information_admitted": False, "new_market_requests_this_continuation": 0,
        "original_strategy_source_files_changed_this_continuation": 0, "code_files_added_this_continuation": added,
        "pre_registration_new_source_representation_fixes": 2,
        "current_goal_turn_classification": "progress", "previous_goal_turn_classification": "progress",
        "goal_turn_progress_classification": "NEW_EXPLANATION_AND_ACTUAL_POINT_QUALITY_EVIDENCE_FULL_ACCOUNT_REJECTED",
        "current_goal_turn_classification_reason": "四例与全体下降突破完整测量产生四场景实际pB正的新事实，同时完整账户未达；四资金恒等式与全66执行状态改变下一诊断问题。",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0,
        "blocked_reason": None, "blocked_audit_key": None, "blocking_decision": None,
        "goal_status": "active", "goal_tool_status_confirmed": "active", "goal_achieved": False,
        "whole_model_overfitting_removed": False, "overfitting_removed": False, "overfit_removed": False,
        "independent_validation_status": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "current_unmet_evidence": "R173实际交易质量四场景均正，但完整收益/夏普共同经济及历史稳定未达，近期明显低于A，独立样本/去过拟合未建立；无新待跑账户策略。",
        "validation_method_this_continuation": "四输入/四账户测试，四实际行情前缀，八原A/价格控制精确复现，四主账户一次运行及保存核对；两初次表示失败结果前修复并保留，不改金融定义。",
    })
    require({key: state[key] for key in FORWARD} == forward_before, "十二真实前瞻值意外变化。")
    require(state["next_experiment"] == next_forward_before, "已有真实前瞻1008日计划意外变化。")
    write_json(STATE, state)
    write_json(OUT / "project_state_update_receipt.json", {
        "at": now(), "status": "PASS_FOUR_DURABLE_FACT_FILES_AND_CURRENT_STATE_UPDATED_ONCE",
        "latest_technical_and_financial": "TECH.R173", "financial_registration": "TECH.R172",
        "actual_prediction_preserved": "TECH.R158", "original_exit_preserved": "TECH.R145",
        "goal_status": "active", "goal_achieved": False, "admitted_unrun_candidates": 0,
        "forward_values_unchanged": forward_before, "existing_forward_next_experiment_unchanged": next_forward_before,
        "all_four_actual_point_quality_passed": True, "individual_economic_gate_passes": 1,
        "overall_full_account_gate_passed": False, "historical_stability_gate_passed": False,
        "original_strategy_source_changes": 0, "new_primary_accounts": 4, "control_replays": 8,
        "sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in
                    [Path(__file__), STATE, *(ROOT / "docs" / name for name in DOCS)]],
    }, exclusive=True)
    print("四长期事实与当前状态已一次更新至R173；四场景交易质量为正、整体账户拒绝，原十二前瞻值保持，目标active。", flush=True)


if __name__ == "__main__":
    run()
