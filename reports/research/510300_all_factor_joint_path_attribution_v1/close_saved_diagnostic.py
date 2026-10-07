"""保存全因素总览、已有结果交叉核对和一次项目状态更新。"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import all_factor_joint_path_attribution_v1 as diagnostic

OUT = diagnostic.OUT
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FINANCIAL_KEYS = ["latest_actual_financial_decision", "latest_actual_financial_result",
    "latest_actual_financial_status", "current_new_strategy_return_sharpe",
    "latest_actual_financial_primary_four_scene_metrics"]
FORWARD_KEYS = ["forward_protocol", "forward_registry", "new_prospective_observations",
    "earliest_future_exchange_session", "registered_candidate_intents", "new_prospective_completed_points",
    "next_new_close_eligible_at", "current_validated_candidates", "forward_account_comparison_protocol",
    "latest_forward_account_check", "new_prospective_sessions_this_continuation",
    "new_prospective_cycles_this_continuation", "next_experiment"]
DOCS = ["PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"]


def git_status():
    return subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT)


def verify_saved_metrics():
    """用上一金融测量的独立保存指标交叉核对本轮所有财富差。"""
    summary = diagnostic.read(OUT / "summary.json")
    metric = pd.read_parquet(diagnostic.METRICS).set_index(["period", "cost", "policy"])
    maximum = 0.
    for row in summary["all_comparisons"]:
        main = metric.loc[(row["period"], row["cost"], diagnostic.PRIMARY)]
        control = metric.loc[(row["period"], row["cost"], row["control"])]
        expected = {
            "net_wealth_gap_cny": main.ending_equity - control.ending_equity,
            "gross_pnl_gap_cny": main.gross_at_actual_quantities_pnl - control.gross_at_actual_quantities_pnl,
            "commission_gap_cny": main.total_commission - control.total_commission,
            "slippage_gap_cny": main.total_slippage - control.total_slippage,
        }
        for key, value in expected.items():
            error = abs(float(row[key]) - float(value))
            maximum = max(maximum, error)
            diagnostic.require(error < 1e-6, "路径分解与旧金融指标不一致：" + key)
    diagnostic.write(OUT / "verification_receipt.json", {
        "at": diagnostic.now(), "meaningful_tests_passed": 4, "pytest_chunk": "e6a42e",
        "test_scope": "亏损与未完成周期保留、双重财富差闭合、质量许可与持仓/成交区分、重复进入不能强配。",
        "saved_financial_metric_rows": len(metric), "all_bridges_verified": 16,
        "maximum_saved_metric_cross_check_error_cny": maximum,
        "maximum_double_bridge_error_cny": summary["maximum_bridge_error_cny"],
        "all_sources_unchanged": all(diagnostic.sha(ROOT / source["path"]) == source["sha256"]
            for source in diagnostic.read(OUT / "protocol.json")["sources"]),
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "independent_validation": "NOT_ESTABLISHED",
    }, exclusive=True)


def render_bridge():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    summary = diagnostic.read(OUT / "summary.json")
    row = next(r for r in summary["all_comparisons"] if r["period"] == "2020_2026" and r["cost"] == "STRESS" and r["control"] == "TECH_COMMON")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), dpi=150)
    panels = [
        ([row["gross_pnl_gap_cny"], row["net_fee_savings_cny"]], ["实际毛损益差", "佣金与滑点节省"], "按实际毛损益与费用分解"),
        ([row["minus_control_only_closed_net_component"], row["joint_only_closed_net_component"], row["matched_entry_closed_net_difference"], row["open_and_residual_net_difference"]],
         ["负的遗漏周期\n净损益", "新增周期\n净损益", "共同进入\n路径差", "开放与残差"], "按全部实际周期分解（已经扣费）"),
    ]
    for ax, (values, labels, title) in zip(axes, panels):
        running = 0.
        for i, value in enumerate(values):
            after = running + value
            ax.bar(i, abs(value), bottom=min(running, after), width=.66,
                   color="#c25755" if value < 0 else "#338977", zorder=3)
            ax.text(i, min(running, after) - 1350, f"{value:+,.2f}", ha="center", va="top", fontsize=10, fontproperties=font)
            ax.plot([i + .33, i + .67], [after, after], color="#81929f", lw=1)
            running = after
        total = row["net_wealth_gap_cny"]
        ax.bar(len(values), -total, bottom=total, width=.66, color="#2d4d67", zorder=3)
        ax.text(len(values), total - 1350, f"{total:+,.2f}", ha="center", va="top", fontsize=10, fontproperties=font)
        ax.set_xticks(range(len(values) + 1), labels + ["完整净财富差"], fontproperties=font, fontsize=10)
        ax.set_title(title, fontproperties=font, fontsize=12, pad=14)
        ax.set_ylim(-37000, 4500)
        ax.axhline(0, color="#567080", lw=.9)
        ax.grid(axis="y", color="#e3e8ed", zorder=0)
        ax.set_ylabel("联合模型减同池价量模型（元）", fontproperties=font, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("联合评分为什么降低收益？同池、同日历、20万元、压力费用", fontproperties=font, fontsize=15, y=.96)
    fig.text(.5, .02, "2020—2026完整账户；两种分解是同一个差额，不能相加。路径差不是单一因果；没有新模型或零费策略。",
             ha="center", fontproperties=font, fontsize=10, color="#526675")
    fig.subplots_adjust(top=.84, bottom=.16, wspace=.30)
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "近期STRESS_联合评分与价量全部净差.png", facecolor="white")
    fig.savefig(folder / "近期STRESS_联合评分与价量全部净差.svg", facecolor="white")
    plt.close(fig)


def factor_overview():
    catalogue_path = ROOT / "reports/research/510300_all_factor_joint_scorecard_v1/results/全部因素目录_同源角色与未绑定.parquet"
    catalogue = pd.read_parquet(catalogue_path)
    diagnostic.require(len(catalogue) == 83 and catalogue.group.nunique() == 12, "原全因素讨论目录改变。")
    roles = {
        "价格结构": ("支撑阻力、突破、回踩、缺口、现金股息调整价", "原支撑锚与价格保留；新增主模型使用原技术特征", "部分可评分；所有结构型机会尚未认证", "股息调整锚与原始交易价不同，不能混用。"),
        "动量与技术": ("日/已完成周MACD、动量、EMA、RSI、KDJ、ADX、背离", "MACD/动量等原技术8字段已进共同模型", "有开发分和真实账户；增益未成立", "同一价格衍生指标重复不算多份独立证据；周线转强可能较晚。"),
        "量价参与": ("相对量、上涨量占比、成交金额、实体效率、OBV、MFI、换手", "相对量、上涨量平衡与实体效率已用", "有部分数值；其余未绑定保持未知", "量放大可能是启动或拥挤，不能统一给正分。"),
        "波动与风险": ("ATR、实现波动、ES、下行波动、止损距离、扣费目标空间", "旧风险预算、全账户回撤、真实数量和费用已用", "风险约束单列，不充当看涨票", "模型计划2ATR/1ATR不等于实际成交盈亏比。"),
        "宏观周期": ("PMI、M1/M2、社融、PPI/CPI、生产消费出口、财政", "PMI订单等公布钟合格部分已入核分", "固定联合用途被拒绝，宏观解释仍保留", "低PMI不排除政策驱动修复；统计背景与当日新催化分开。"),
        "政策与预期兑现": ("降准、降息、财政、事前预期、宣布—落地—价格接受", "12支持正例、84月双期限LPR与9/24公布案例保存", "解释部分可用；完整催化/预期评分未建立", "正例目录不等于全部事件日历；缺记录不能当无政策，没一致预期不能说超预期。"),
        "资金价格与融资": ("DR007—政策利率差、差的变化、融资余额/买入、信用利差", "资金/融资原6宏观中的相应字段已用", "已实际对照，当前固定用途未提升收益夏普", "资金宽松或融资增长可能滞后；同来源信号不可重复投票。"),
        "真实流量与订单流": ("ETF份额、申赎、溢折价、PCF/IOPV、CVD/L2、北向", "当前官方份额只有基准版本，新增可比变化0", "UNKNOWN；未来版本观察不等于历史分数", "日线成交量不能恢复逐笔主动买卖；份额变化也不是所有二级资金流。"),
        "指数主线与广度": ("当时成员权重、行业相对强弱、上涨扩散、持续性、对510300贡献", "财报广度使用当时成员；行业权重只得到滞后披露上下文", "历史主线贡献评分尚未准入", "2026/8/31行业快照不能回填历史或当作实时ETF持仓。"),
        "估值与盈利": ("PE/PB、股息率、股债差、利润现金流改善、EPS与修订", "原始财报16字段加入；当前官方估值为滞后披露上下文", "盈利已有共同模型结果；估值历史首版未认证", "原始财报等权广度不等于指数EPS；供应商晚更新不准回填。"),
        "博弈与情绪": ("拥挤、持仓、IF基差、期权IV/PCR/偏斜、涨跌停、再平衡", "券商资金博弈研究用于提出机制；完整历史输入未建立", "未验证项UNKNOWN，不指定拍脑袋权重", "期权可作观察，当前实测对象仍是标的多头点位与完整账户。"),
        "外部环境与事件": ("人民币、美元、美债、海外收盘、商品油价、突发事件", "讨论保留；本次共同模型未纳入完整可用序列", "UNKNOWN；不据已知上涨反推方向或窗口", "海外收盘、公布时差和突发事件原可用钟必须逐条明确。"),
    }
    rows = []
    for group in catalogue.group.drop_duplicates():
        representative, current, score_role, limits = roles[group]
        rows.append({"因素组": group, "原目录项数": int(catalogue.group.eq(group).sum()),
            "主要内容": representative, "当前实证覆盖": current, "本次评分处理": score_role, "限制": limits})
    diagnostic.table("12因素组_全部讨论与评分状态", pd.DataFrame(rows))
    context = pd.read_parquet(diagnostic.CONTEXT)
    prediction = pd.read_parquet(diagnostic.SCORES)
    selected = context[context.date.isin(pd.to_datetime(["2024-09-24", "2024-09-27", "2024-10-08", "2025-05-06"]))].copy()
    selected["relative_volume"] = np.exp(selected.log_relative_volume)
    selected["realized_volatility_ratio"] = np.exp(selected.log_rv_ratio)
    selected["pmi_orders_original"] = selected.pmi_orders_level + 50
    selected = selected.merge(prediction.loc[prediction.policy.eq(diagnostic.PRIMARY),
        ["date", "score", "fit_index", "candidate_quality_pass", "earnings_fields_on_path", "macro_fields_on_path"]],
        on="date", how="left", validate="one_to_one")
    diagnostic.table("固定案例_量价背景与当时联合路径", selected)
    lines = ["# 全因素研究总览：讨论目录、解释分与实测联合模型", "",
        "按照用户‘所有因素全部加进来讨论、进行打分’的要求，统一保存12类83项讨论目录。它是一份有限的研究目录，不宣称穷尽所有市场因素。",
        "原30案例逐项2490观察、原解释等级以及所有未知仍保留；原目录的绑定数量是旧解释卡的事实，原始财报与滞后官方披露的后续绑定分别记录，不重写旧结论。", "",
        "| 因素组 | 原目录项数 | 当前实证覆盖 | 评分处理 |", "|---|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['因素组']} | {row['原目录项数']} | {row['当前实证覆盖']} | {row['本次评分处理']} |")
    lines += ["", "逐组的具体因素、限制及来源状态见同名CSV/Parquet。缺失为未知，不计成0分；同源MACD/RSI/KDJ或同一次政策的多篇报道不增加独立票数。",
        "", "解释分与预测分分开：原解释等级0/20/40/60/80描述已知价量、支持锚和传导条件；例如2024/9/25原解释分60，2019/1/9为80。它们不是胜率或入场指令。新原始财报16＋宏观6＋技术8的共同模型有528原点开发分和实测完整账户；全83项预测总分仍未计算。",
        "", "固定联合模型近期压力费用结果：净年化-0.7125%、净Sharpe -0.440314、实际净p×盈亏比0.219372、最大回撤4.8811%、7完整交易，目标未通过。三个匹配去组对照与原A一并保留，不能因某对照更好而改称成功。",
        "", "## 具体上涨段：同一指标的意义随阶段改变", "",
        "| 原点 | 原始收盘 | 日MACD/ATR | 前已完成周MACD/ATR | 相对量 | 短长实现波动比 | PMI订单原值 | 融资五日变化 | 主联合开发分 | 固定月度拟合索引 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected.itertuples():
        lines.append(f"| {row.date:%Y-%m-%d} | {row.close:.3f} | {row.daily_hist_atr:.4f} | {row.weekly_hist_atr:.4f} | {row.relative_volume:.2f} | {row.realized_volatility_ratio:.2f} | {row.pmi_orders_original:.1f} | {row.financing_net_change5:.3%} | {row.score:.4f} | {row.fit_index:.0f} |")
    lines += ["", "2024/9/24：日线动量/成交已增强，前一已完成周仍弱；PMI与财报背景未明显转好，但有当天公开政策催化。9/27动量和量价延续，主联合仍拒绝。10/8周线、PMI与融资转强，波动和相对量也更高；此时主联合放行、次日实际进入，随后亏损。",
        "注意：10/8已换成原固定规则下的另一个月度拟合，主模型决策路径此时没有宏观/盈利字段，不能把分数跳升单独归因于PMI。价格已大涨后的趋势确认与此前启动是不同阶段；‘全部变好统一加分’的用途没有成立。",
        "", "因此新的讨论主线是公开催化、缓慢背景、资金传导、价量启动/延续和拥挤/失效分开，再研究共同状态。它仍是待检验机制，不是已获利策略；完整事件来源覆盖和事前预期不足，目前不准入数值模型。",
        "", "## 可以继续与已经关闭的用途", "",
        "接受：全因素讨论表、真实点位的可用钟解释、原始财报与技术/宏观的严格共同对照、完整实际路径和成本分解。",
        "拒绝：本次固定30字段树的完整收益/Sharpe用途；旧双期限LPR早退用途；已冻结的旧线性/非线性宏观、同覆盖成分、收益尾部等失败用途，均不是换阈值/权重再试的对象。",
        "待证：新增公开催化的完整台账、预期与实际、历史指数主线/真实流量/拥挤状态，以及真正新样本的独立账户。未验证信息既不当利好票，也不当利空票。",
        "", "下一有限实验是来源可用性检验：保留成功/失败/无动作/时钟不明，对公开催化按同一事件去重，报告完整分母与缺口；原12正例不能提供全样本无事件状态。来源门通过后才冻结唯一共同状态用途和全部同池对照。",
        "", "本总览不更改原策略，不新增拟合/账户/标签，不授权交易。"]
    (OUT / "全因素研究总览_解释分与实测结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnostic.write(OUT / "next_source_purpose.json", {
        "created_at": diagnostic.now(), "purpose": "新增公开催化、事前预期/实际落地与价量阶段的来源可用性检验。",
        "status": "NOT_ADMITTED_SOURCE_COVERAGE_AND_EXPECTATIONS_UNPROVEN", "numeric_configurations": 0,
        "source_role": "当时公布、共同事件去重、全样本分母、成功/失败/无动作/未知保留；旧正例不等于覆盖。",
        "interpretation": "旧统计背景不替代新催化；启动、延续、拥挤/失效是待证共同状态，不直接手填上涨权重。",
        "first_experiment": "先盘点原公开源事件与原时钟、事前同工具调查和缺失；不得根据已知盈亏补事件或给缺失记无动作。",
        "later_numeric_gate": "来源完整性通过后另登记唯一不同用途、全部同池去组对照、原完整两时期两费用和新独立验证。",
        "stop_condition": "来源不足即以UNKNOWN/NOT_ADMITTED关闭此来源实验，不以降低覆盖或后见标签救。",
        "not_allowed": ["旧30字段树调参", "以新阶段名重命名旧失败", "挑上涨段当全部事件", "事后权重", "逆转失败信号", "当前快照回填历史"],
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "goal_achieved": False,
    }, exclusive=True)


def update_once():
    diagnostic.write(OUT / "close_started.json", {"at": diagnostic.now(), "state_updates_planned": 1}, exclusive=True)
    original = diagnostic.read(STATE)
    old_git = git_status()
    financial = {k: copy.deepcopy(original.get(k)) for k in FINANCIAL_KEYS}
    forward = {k: copy.deepcopy(original.get(k)) for k in FORWARD_KEYS}
    observer = copy.deepcopy(original.get("independent_official_share_observer"))
    diagnostic.require(financial["latest_actual_financial_decision"] == "TECH.R256", "最新实际金融不是上一轮R256。")
    diagnostic.write(OUT / "state_before_TECH_R258.json", original, exclusive=True)
    service = diagnostic.read(OUT / "goal_service_status_after_delivery.json")
    diagnostic.require(service["goal"]["status"] == "active", "目标服务不是预期active。")
    summary = diagnostic.read(OUT / "summary.json")
    updated = copy.deepcopy(original)
    at = diagnostic.now()
    updated.update({
        "updated_at": at, "status": "research_active", "goal_achieved": False,
        "latest_completed_study": OUT.absolute().relative_to(ROOT).as_posix(),
        "latest_result": "reports/research/510300_all_factor_joint_path_attribution_v1/summary.json",
        "latest_report": "reports/research/510300_all_factor_joint_path_attribution_v1/全因素共同评分_全部失误与资金费用路径.md",
        "latest_research_status": summary["status"], "latest_registration_decision": "TECH.R257",
        "latest_technical_decision": "TECH.R258", "current_study": summary["study_id"],
        "current_phase": "SAVED_JOINT_ALL_PATH_DIAGNOSTIC_COMPLETE_NEW_CATALYST_SOURCE_PURPOSE_NOT_ADMITTED",
        "new_accounts_in_current_phase": 0, "current_priority": "先完整盘点新增公开催化与事前预期来源，不把旧统计背景、正例台账或已知上涨当完整共同状态。",
        "next_research_question": "新的公开催化、其事前预期/实际落地和价量启动/延续/拥挤是否有完整当时可用证据？先做来源检验，未准入不新增评分配置。",
        "latest_progress": "实际22856请求路径、198周期外连接和16财富差闭合；联合模型劣化主要为真实评分错拒和新增亏损，费用节省不能解释失败。不是仅同步状态。",
        "latest_continuation_outcome": "R257—R258已保存全路径诊断，0新拟合/账户/标签；原R256固定拒绝保持，公开催化与背景/价量阶段用途尚未准入。",
        "current_direction": "12类83项统一讨论；缺失不记0。原联合完整用途关闭，继续不同进入前信息来源与真正独立点位验证。",
        "latest_gap_attribution": "reports/research/510300_all_factor_joint_path_attribution_v1/summary.json",
        "latest_actual_financial_failure_diagnosis": "reports/research/510300_all_factor_joint_path_attribution_v1/summary.json",
        "latest_all_factor_overview": "reports/research/510300_all_factor_joint_path_attribution_v1/全因素研究总览_解释分与实测结果.md",
        "next_all_factor_source_purpose": "reports/research/510300_all_factor_joint_path_attribution_v1/next_source_purpose.json",
        "latest_continuation_receipt": "reports/research/510300_all_factor_joint_path_attribution_v1/project_state_update_receipt.json",
        "latest_continuation_audit": "reports/research/510300_all_factor_joint_path_attribution_v1/verification_receipt.json",
        "previous_goal_turn_classification": diagnostic.read(diagnostic.FINANCE / "project_state_update_receipt.json")["classification"],
        "latest_goal_turn_classification": "PROGRESS_R257_R258_ACTUAL_ALL_SAVED_SCORE_AND_FINANCIAL_PATH_DIAGNOSTIC",
        "goal_tool_status_confirmed": "active",
        "latest_goal_tool_status_receipt": "reports/research/510300_all_factor_joint_path_attribution_v1/goal_service_status_after_delivery.json",
        "blocked_audit_count": 0, "goal_blocked_audit_count": 0, "blocked_reason": None,
        "blocked_audit_key": None, "current_goal_turn_blocker_id": None,
        "blocking_decision": "NONE_THIS_TURN_COMPLETE_NEW_PATH_DIAGNOSTIC_FUTURE_SOURCE_PURPOSE_NOT_ADMITTED",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "current_unmet_evidence": "最新R256金融四场景失败；R258查明真实评分与路径缺口但没有新独立样本或通过的不同完整用途，收益/Sharpe和去过拟合目标未完成。",
        "next_saved_account_diagnostic_status": "COMPLETED_R258_ALL_ORIGINS_AND_PATHS_CLOSED",
    })
    block = """### TECH.R257—R258：全部评分与真实周期路径归因（2026-10-06）

**最新实际金融仍为R256固定拒绝；本轮完成的是有新数值归因的保存数据诊断。0新拟合、账户、收益标签，独立完成点位0，提高收益与净Sharpe目标未完成，goal实际active。**

假设→固定三源共同模型劣化可能来自评分许可、持仓/风险/执行路径或费用；并检查旧统计背景与新增公开催化的表达缺口。
验证方法→冻结原3488原点、四模型、20保存账户、两时期两费用；22856真实决策记录分开评分门、空仓请求、已有持仓与实际成交；198周期外连接保留赢家、输家、期末未完成，16比较同时核对真实毛费净和全部周期/开放残差。
结果→近期STRESS主联合减同池纯价量净财富-28976.05元=负的遗漏完整周期净损益-25750.93+新增完整周期-2781.78+同进入路径差-443.33+近0开放残差。联合少做交易节省费用1239.65元，实际毛损益差-30215.70元，不能把劣化归费用。遗漏6盈利/1亏损周期，新增1盈利/2亏损周期；7遗漏全由主模型质量门拒绝。42许可日中35天已有持仓，实际7新周期；不是42个新机会。A、宏观及盈利去组的全部场景差和拒绝也保留。
具体上涨段→2024/9/24日MACD/ATR+0.3028、前完成周-0.2742、相对量3.34，公开降准/利率催化当天已公布；9/24、9/27联合分均0.311553，拦住价量对照9/25→9/27和9/30→10/8真实盈利5212.17/15507.05元。10/8前完成周转正、相对量6.85、波动比3.63，联合分84.213527，10/9入场后亏损。10/8同时换月度拟合且路径不含宏观/盈利，不能把分数变化归单个PMI或新闻原因。上涨启动与后续拥挤是待证不同状态。
接受/拒绝→接受本全路径诊断和12类83项完整讨论总览，原30案例及0/20/40/60/80解释等级不变，全83项预测总分NOT_COMPUTED。R256固定30字段共同树继续拒绝，历史分非可信胜率；不换树深、叶子、阈值、窗口或挑更好去组账户作为主候选。
重新验证/下一有限实验→新增公开催化、事前同工具预期/实际落地与价量阶段的来源可用性检验。原12支持记录只有正例，不是完整政策日历；未知不当无政策，没一致预期不称超预期。成功/失败/无动作/时钟不明、同事件多报道去重、完整分母必须保存。当前NOT_ADMITTED_SOURCE_COVERAGE_AND_EXPECTATIONS_UNPROVEN，数值配置0；来源通过后另登记唯一不同用途及真正新样本独立账户。来源不足关闭该来源用途，不降低覆盖营救。
必要验证→4测试通过，上一金融20保存指标交叉核对16比较，双重桥最大误差1.783e-10元；旧来源哈希、金融五字段、前瞻十三字段和官方份额观察合同均保持。10月8日15:05新收盘及10月8日23:05首份额采集窗等原日期不改，未创建自动采集或交易。
证据→`reports/research/510300_all_factor_joint_path_attribution_v1/summary.json`、`全因素共同评分_全部失误与资金费用路径.md`、`全因素研究总览_解释分与实测结果.md`、`results/全部周期外连接_盈利亏损和未完成保留.parquet`、`next_source_purpose.json`。

"""
    records = []
    for name in DOCS:
        path = ROOT / "docs" / name
        old = path.read_bytes()
        (OUT / (path.stem + "_before_TECH_R258.md")).write_bytes(old)
        title, remainder = old.split(b"\n", 1)
        insertion = ("\n" + block).encode("utf-8")
        content = title + b"\n" + insertion + remainder
        path.write_bytes(content)
        preserved = path.read_bytes() == content and content[len(title) + 1 + len(insertion):] == remainder
        diagnostic.require(preserved, "项目文档旧正文未保持。")
        records.append({"path": path.absolute().relative_to(ROOT).as_posix(), "old_body_preserved_exact": preserved, "sha256": diagnostic.sha(path)})
    diagnostic.write(STATE, updated)
    saved = diagnostic.read(STATE)
    diagnostic.require(all(saved.get(k) == v for k, v in financial.items()), "最新实际金融五字段改变。")
    diagnostic.require(all(saved.get(k) == v for k, v in forward.items()), "原前瞻十三字段改变。")
    diagnostic.require(saved.get("independent_official_share_observer") == observer, "官方份额观察合同改变。")
    diagnostic.require(git_status() == old_git, "已有跟踪文件工作区状态改变。")
    diagnostic.write(OUT / "project_state_update_receipt.json", {"at": at, "registration": "TECH.R257", "decision": "TECH.R258",
        "state_updates": 1, "classification": updated["latest_goal_turn_classification"], "goal_service_status": "active",
        "financial_five_preserved_exact_R256": True, "forward_thirteen_preserved_exact": True,
        "official_share_contract_preserved_exact": True, "tracked_git_status_preserved_exact": True,
        "documents": records, "new_fits": 0, "new_accounts": 0, "new_labels": 0,
        "new_independent_completed_points": 0, "goal_achieved": False, "archive_writing_itself_is_not_research_progress": True}, exclusive=True)
    print("已一次更新项目状态与四份长期事实文档；R256金融拒绝和原观察合同保持。")


if __name__ == "__main__":
    verify_saved_metrics()
    render_bridge()
    factor_overview()
    update_once()
