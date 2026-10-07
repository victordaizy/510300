"""一次裁决预期兑现退出，并保存具体量价解释与长期事实。"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from research import policy_expectation_thesis_exit_completion_v1_0_1 as study
from research.close_broker_stage_policy_v1 import prepared_prepend

OUT, ROOT, STATE = study.OUT, study.ROOT, study.STATE
REPORT = "信用预期兑现与支持持仓_完整结果.md"


def deliver():
    study.frozen_exact()
    raw = study.read(OUT / "financial_raw_summary.json")
    frame = pd.read_parquet(OUT / "results/全部12完整账户_四新与八保存对照.parquet")
    gates, comparisons = [], []
    for (period, cost), block in frame.groupby(["period", "cost"], sort=False):
        indexed = block.set_index("policy")
        primary = indexed.loc[study.POLICY]
        conditions = {"actual_net_pB_above_one": bool(primary.p_times_b > 1),
                      "standard_net_EV_positive": bool(primary.standard_expectancy_loss_units > 0),
                      "drawdown_at_most_ten_percent": bool(primary.max_drawdown <= .1)}
        for control in (study.BASELINE, study.parent.SAVED_CORE):
            reference = indexed.loc[control]
            conditions["beats_cagr_and_sharpe_" + control] = bool(primary.net_cagr > reference.net_cagr and primary.net_sharpe > reference.net_sharpe)
            comparisons.append({"period": period, "cost": cost, "control": control,
                                "cagr_delta": float(primary.net_cagr-reference.net_cagr),
                                "sharpe_delta": float(primary.net_sharpe-reference.net_sharpe)})
        gates.append({"period": period, "cost": cost, **conditions, "economic_passed": all(conditions.values())})
    cancellations = pd.read_parquet(OUT / "results/全部取消请求_真实来源归属与执行钟.parquet")
    orders = pd.read_parquet(OUT / "results/全部真实订单_完整现金反馈.parquet")
    executed = orders.loc[orders.reason.eq(study.EXIT_REASON)]
    unique_origins = sorted(str(pd.Timestamp(x).date()) for x in executed.origin.unique())
    recent_passed = all(x["economic_passed"] for x in gates if x["period"] == "2020_2026")
    all_passed = all(x["economic_passed"] for x in gates)
    status = "HISTORICAL_FIXED_EXIT_POINT_INCREMENT_NOT_INDEPENDENTLY_VALIDATED" if recent_passed else "REJECTED_FIXED_DUAL_CREDIT_EXIT_NET_RETURN_SHARPE_TARGETS_NOT_MET"
    summary = {**raw, "status": status, "all_four_scene_economic_gates_passed": all_passed,
               "predeclared_recent_two_scene_gates_passed": recent_passed, "gates": gates, "comparisons": comparisons,
               "cancellation_requests_across_fee_copies": len(cancellations), "actual_cancel_sell_orders_across_fee_copies": len(executed),
               "unique_historical_cancel_decisions": unique_origins, "independent_completed_points": 0,
               "historical_increment_stability": "NOT_ESTABLISHED_SINGLE_ORIGINAL_MESSAGE_IF_ONE_CANCEL",
               "search_selection_correction": "NOT_COMPUTED", "overfitting_removed": False,
               "goal_achieved": False, "report": study.rel(OUT / REPORT)}
    study.write(OUT / "summary.json", summary)
    study.export(pd.DataFrame(gates), "全部四场景门_既定近期与早期都披露")
    study.export(pd.DataFrame(comparisons), "全部八同账户净增量_不拼时期")
    source_roles = []
    roles = {
        "OCBC_2018_OUTLOOK": "中国章节预计进一步降准、对降基准利率相对保留；年度方向展望，无1月4日单次幅度或当时公开首版证明。",
        "OCBC_20190107": "1月4日降准后解释总释放、MLF到期与净流动性；可以解释传导，不能冒充1月4日降准前调查。",
        "ING_20230613": "7天逆回购已降之后讨论MLF、LPR可能跟进；后续条件预期，不能倒填本次OMO事前共识。",
        "MUFG_20230613": "预测后续各政策利率还有10—20bp降幅，带财政条件；多工具/未固定单次公布时点，不能作为精确LPR或OMO市场中位数。",
    }
    for sid, date, title, url in study.SOURCES:
        receipt = study.read(OUT / "sources" / sid / "receipt.json")
        source_roles.append({"id": sid, "internal_date": date, "url": url, "http_status": receipt.get("status"),
                             "historical_publication_upper": "NOT_ESTABLISHED", "role": roles[sid] if receipt.get("status") == 200 else "NO_VIEW",
                             "numeric_model_role": "EXCLUDED_BACKGROUND_ONLY"})
    study.write(OUT / "original_author_source_roles.json", {"sources": source_roles, "new_reports_not_used_as_numeric_model_inputs": True})
    lines = ["# 信用预期兑现与支持持仓：完整研究结果", "", f"固定裁决：**{status}**。目标未完成，独立完成点位仍0。",
             "", "策略可以变化；本次检验的是持仓论据随新增消息改变，所有初始技术进入和原风险费用保持。一次主用途四个新账户，不训练、不扫阈值、不择期拼接。",
             "", "| 时期 | 费用 | 完整账户 | 净年化 | 净夏普 | 回撤 | 胜率 | 实际盈亏比 | p×B | 完成次数 | 年均次数 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = {study.POLICY: "新：双期限兑现取消", study.BASELINE: "原：共同现金支持补充", study.parent.SAVED_CORE: "原A"}
    for row in frame.itertuples():
        lines.append(f"| {row.period} | {row.cost} | {names[row.policy]} | {row.net_cagr:.4%} | {row.net_sharpe:.6f} | {row.max_drawdown:.4%} | {row.win_rate:.2%} | {row.payoff:.4f} | {row.p_times_b:.6f} | {row.completed_cycles} | {row.average_full_year_cycles:.4f} |")
    lines.extend(["", "完整84个月包括56个保存调查和28个缺失；期限区间不能唯一识别时保留ABSTAIN。3个共同偏弱事件是2022-08、2023-06、2023-08，是否取消还必须由当时真实持仓归属决定。缺失和未识别月份没有被命名为政策中性。",
                  "", "数字输入复用当时调查的公布及修改时钟、原官方报价和严格边界。新取得的4份银行原报告只支持机制解释，内部日期不是历史公开上界。LPR调查没有传递给7天逆回购，个人预测没有当市场共识。",
                  "", "旧LPR五模型的20日收益/误差阈值进入和旧股债联合四预测均保留原结果；本次只在实际支持持仓中应用新消息失效。历史赢亏及2023年6月方向已在登记前看过，属于探索，不是盲测。",
                  "", "可执行机制：至少一个LPR期限明确低于原调查预期的宽松程度，另一期限一致或也偏弱；两个期限相反或不确定时不取消。政策公布可知后的第一个15:05收盘形成一次请求，次开盘按原T+1、整手、涨跌停与最低佣金执行。CORE持仓由原策略处理；旧支持进入事件不能因退出后重新使用。",
                  "", f"共有{len(executed)}笔费用复本卖单，但只有{len(unique_origins)}个不同原历史取消决定：{', '.join(unique_origins) or '无'}。费用复本不能增加样本量。前期LPR机制/调查覆盖不足，不称已检验新信息在所有时期有效。",
                  "", "近期两费用是否同时胜过原共同现金及原A、pB>1、EV>0、DD<=10%：" + str(recent_passed) + "；四场景完整经济门：" + str(all_passed) + "。即使局部点值增加，也没有独立增量、稳定性或去过拟合证明。",
                  "", "原作者报告解释与用途如下："])
    for source in source_roles:
        lines.append(f"\n- [{source['id']}]({source['url']})：{source['role']}历史公开上界仍未知，未进入数值模型。")
    lines.extend(["", "具体2023年6月解释：6月13日OMO降10bp后，价格接受形成6月15日收盘的原进入判断，6月16日开盘实际买入；6月19日新调查预期可知，6月20日两个LPR实际各降10bp。调查约束的一年中位数为降10bp；五年以上中位数至多为−12.5bp，因此五年以上实际至少比调查中位数少宽松2.5bp，不能擅自写成精确‘预期降20bp’。这只是信用兑现差异，不能据此认定股价下跌的因果。",
                  "", "下一步：保留本固定终态。不同信息进入模型仍需完整机制和对照，真正未来的调查原版本与官方份额可以另观察；不能把本次已知盈亏的退出提前日、期限权重或阈值重新扫描。当前原E03前瞻13项及独立官方份额观察合同精确保持。"])
    (OUT / REPORT).write_text("\n".join(lines)+"\n", encoding="utf-8")
    make_case_figure()
    files = [OUT / "summary.json", OUT / REPORT, OUT / "original_author_source_roles.json", OUT / "figures/2023年6月_价格接受与信用兑现.png"]
    study.write(OUT / "delivery_receipt.json", {"at": study.now(), "result": status,
                "files": [{"path": study.rel(p), "sha256": study.digest(p)} for p in files],
                "new_primary_accounts": 4, "saved_controls": 8, "independent_points": 0, "goal_achieved": False})
    print("全部结果、原作者用途与具体量价图已保存；没有根据结果调整固定退出。", flush=True)


def make_case_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator

    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((x for x in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC") if x in installed), "DejaVu Sans")
    plt.rcParams.update({"font.family": selected, "axes.unicode_minus": False, "font.size": 10})
    data, _, _, _, _ = study.parent.load()
    close = data.close.astype(float)
    dif = close.ewm(span=12, adjust=False).mean()-close.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    data = data.copy()
    data["illustrative_macd_hist"] = 2*(dif-dea)
    case = data.loc[data.date.between("2023-06-01", "2023-06-30")].copy().reset_index(drop=True)
    volume = next((x for x in ("volume", "vol", "成交量") if x in case.columns), None)
    if volume is None:
        raise ValueError("原日线没有可验证成交量列，不能编造量价图。")
    x = np.arange(len(case))
    figure, axes = plt.subplots(4, 1, figsize=(13.5, 10.0), sharex=True, gridspec_kw={"height_ratios": [2.6, 1, 1, 1]}, constrained_layout=True)
    axes[0].plot(x, case.close, color="#185b8d", marker="o", markersize=3, label="原日线收盘价")
    axes[0].set_ylabel("510300价格（元）")
    annotation = [("2023-06-13", "OMO降10bp", "#825a20"), ("2023-06-16", "原实际进入", "#237748"),
                  ("2023-06-20", "LPR兑现差异可知", "#9b3947"), ("2023-06-21", "新退出执行", "#9b3947"), ("2023-06-27", "原退出执行", "#525c64")]
    for date, label, color in annotation:
        idx = np.flatnonzero(case.date.eq(pd.Timestamp(date)))
        if len(idx):
            k = int(idx[0])
            axes[0].axvline(k, color=color, alpha=.45, linestyle="--")
            axes[0].annotate(label, (k, case.close.iloc[k]), xytext=(0, 26 if date != "2023-06-20" else -36), textcoords="offset points", color=color, ha="center", fontsize=9)
    colors = np.where(case.close >= case.open, "#b34c4c", "#27836b")
    axes[1].bar(x, case[volume].astype(float), color=colors, alpha=.8)
    axes[1].set_ylabel("原成交量单位")
    axes[2].bar(x, case.illustrative_macd_hist, color=np.where(case.illustrative_macd_hist >= 0, "#b34c4c", "#27836b"))
    axes[2].axhline(0, color="#66717b", linewidth=.6)
    axes[2].set_ylabel("MACD柱（12/26/9）")
    case["illustrative_ATR14_percent"] = case.atr14.astype(float)/case.close.astype(float)*100
    axes[3].plot(x, case.illustrative_ATR14_percent, color="#7b5a91", marker="o", markersize=3)
    axes[3].set_ylabel("ATR14/价格（%）")
    axes[3].set_xticks(x[::2], [d.strftime("%m-%d") for d in case.date.iloc[::2]])
    axes[0].set_title("2023年6月：初始价格接受后，新增信用兑现信息改变持仓论据\n量价与MACD仅作解释；新规则只用双期限调查边界及真实支持持仓", loc="left", fontsize=13)
    for axis in axes:
        axis.grid(axis="y", alpha=.18)
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.spines[["top", "right"]].set_visible(False)
    folder = OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / "2023年6月_价格接受与信用兑现.png", dpi=160)
    plt.close(figure)
    study.export(case[["date", "open", "high", "low", "close", volume, "illustrative_macd_hist", "illustrative_ATR14_percent"]], "2023年6月解释图原值_MACD不作新筛选")


def close():
    study.frozen_exact()
    result = study.read(OUT / "summary.json")
    study.read(OUT / "delivery_receipt.json")
    explanation = study.read(OUT / "report_explanation_addendum.json")
    if explanation["final_report_sha256"] != study.digest(OUT / REPORT):
        raise ValueError("最终具体解释报告与补充回执不一致。")
    service = study.read(OUT / "goal_service_status_after_delivery.json")
    if service["status"] != "active":
        raise ValueError("目标服务未在活动状态；不能写成活动研究。")
    guard = OUT / "project_state_update_started.json"
    study.write(guard, {"at": study.now(), "decision": study.DECISION})
    state_bytes = STATE.read_bytes()
    state = study.read(STATE)
    forward = {k: state[k] for k in study.facts.FORWARD}
    result_path, report_path = study.rel(OUT / "summary.json"), study.rel(OUT / REPORT)
    primary = result["primary_metrics"]
    recent = next(x for x in primary if x["period"] == "2020_2026" and x["cost"] == "STRESS")
    text = f"""### TECH.R245—R246：双期限信用兑现与支持持仓取消（2026-10-06）

**当前金融裁决：{result['status']}；提高净收益与净Sharpe的目标仍未完成。**

假设→支持和价格接受产生的持仓，在新公布两个LPR期限共同明确少于事前调查预期时取消，可能改善完整账户。
方法→84完整月/56保存事前调查/28缺失；区间、实际公布与修订钟保留，两期限共同状态，不传给OMO、不取中点；一次唯一退出，真实COMPLEMENT归属才触发，次开盘原20万元共同现金与风险费用；两时期两费用4新账户、8原保存对照。旧LPR预测训练与联合四模型、R240终态不改。四个关闭事件账户的daily/orders/trades精确复现R240；8必要测试通过，登记前测试替身闭包失败证据保留。

结果→3个共同兑现偏弱月份，真正改变历史退出的不同原决定只有{len(result['unique_historical_cancel_decisions'])}个；费用复本不增加独立样本。近期STRESS新净CAGR {recent['net_cagr']:.4%}、净Sharpe {recent['net_sharpe']:.6f}、DD {recent['max_drawdown']:.4%}、净pB {recent['p_times_b']:.6f}。近期两费用全部既定经济门={result['predeclared_recent_two_scene_gates_passed']}，四场景门={result['all_four_scene_economic_gates_passed']}。完整账户及原A对照、逐年、所有周期与真实订单已保存；不能称提高全目标或消除过拟合。

接受/拒绝→接受新增消息改变支持持仓论据的可执行检验及真实局部事实；按固定门保持本裁决，不调退出日/期限/阈值、不救旧模型或拼时期。4份原作者银行报告GET均保存，内部日期/历史公开首版未认证，只解释机制、未进入数值模型；年度降准观点与宣布后跟进预期不等于单次OMO事前共识。

再验证→只接受真实新样本、不同完整信息用途或真实实现错误；1个历史修改（若仅1）不建立稳定性/独立增量。当前待跑新金融0、独立完成点位0、去过拟合未达，goal active未完成。原E03前瞻13项及R244官方份额观察合同精确保持，自动任务未创建。

依据：[完整结果](../{report_path})、[固定登记](../{study.rel(OUT/'protocol.json')})、[全体账户](../{result_path})、[具体量价图](../{study.rel(OUT/'figures/2023年6月_价格接受与信用兑现.png')})。
"""
    documents = []
    prepared = []
    for name in ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md"):
        path = ROOT / "docs" / name
        prepared.append((path, *prepared_prepend(path, text)))
    state.update({"updated_at": study.now(), "latest_completed_study": study.rel(OUT), "latest_result": result_path,
        "latest_report": report_path, "latest_research_status": result["status"], "latest_progress": "唯一双期限兑现退出四新完整账户与八原对照、原作者四来源和具体量价图已完成。",
        "latest_technical_decision": study.DECISION, "latest_actual_financial_decision": study.DECISION,
        "latest_actual_financial_result": result_path, "latest_actual_financial_status": result["status"],
        "latest_actual_financial_primary_four_scene_metrics": primary, "current_new_strategy_return_sharpe": "COMPUTED_R246_COMPLETE_FIXED_EXPLORATORY_ACCOUNTS_NOT_INDEPENDENTLY_VALIDATED",
        "current_study": "510300_POLICY_EXPECTATION_THESIS_EXIT_V1", "current_phase": "FIXED_POLICY_THESIS_EXIT_EXPERIMENT_COMPLETED_NO_PROMOTION",
        "current_admitted_unrun_numeric_candidates": 0, "current_admitted_unrun_complete_uses": 0,
        "current_financial_candidate_admission": result["status"], "new_accounts_in_current_phase": 4,
        "necessary_tests_passed_in_current_phase": 8, "current_phase_trial_accounting": result,
        "current_goal_turn_actual_work": {"new_primary_accounts": 4, "muted_exact_control_accounts": 4, "saved_controls": 8, "new_broker_original_requests": 4, "new_fits": 0, "new_independent_points": 0},
        "goal_turn_classification": "PROGRESS_R245_R246_FOUR_NEW_FIXED_THESIS_EXIT_FULL_ACCOUNTS",
        "blocked_audit_count": 0, "consecutive_blocked_goal_turns": 0, "goal_achieved": False,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": service["observed_at_utc"],
        "latest_goal_tool_status_receipt": study.rel(OUT / "goal_service_status_after_delivery.json"),
        "next_research_question": "真正未来的支持/调查原版本与官方份额、价格响应能否建立持续性信息？不能调本固定历史取消用途。"})
    state["next_source_matching_proposal"] = {**state["next_source_matching_proposal"],
        "status": "COMPLETED_PRIOR_LPR_DIFFERENT_EXIT_USE_OTHER_TOOL_EXPECTATIONS_UNKNOWN",
        "policy_numeric_expectation_status": "R246_LPR_THESIS_EXIT_COMPLETED_OMO_EXPECTATION_NOT_ESTABLISHED",
        "policy_exit_result": result_path, "new_financial_runs": 1, "new_accounts": 4,
        "financial_admission": result["status"]}
    with (OUT / "state_before_TECH_R246.json").open("xb") as stream:
        stream.write(state_bytes)
    for path, old, new, body in prepared:
        if path.read_bytes() != old:
            raise ValueError("准备后长期事实发生变化。")
        with (OUT / f"{path.stem}_before_TECH_R246.md").open("xb") as stream:
            stream.write(old)
        path.write_bytes(new)
        if not path.read_bytes().endswith(body):
            raise AssertionError("原事实正文未保留。")
        documents.append({"path": study.rel(path), "old_body_preserved_exact": True})
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    saved = study.read(STATE)
    if {k: saved[k] for k in study.facts.FORWARD} != forward:
        raise AssertionError("原13项前瞻字段改变。")
    study.write(OUT / "project_state_update_receipt.json", {"at": study.now(), "decision": study.DECISION,
        "documents": documents, "actual_state_updates": 1, "original_forward_thirteen_exact": True,
        "latest_actual_financial_decision": saved["latest_actual_financial_decision"], "new_candidate_accounts": 4,
        "final_report_sha256": explanation["final_report_sha256"],
        "independent_validation": "NOT_ESTABLISHED", "goal_status": "active", "goal_achieved": False})
    print("R245—R246四份长期事实与状态已一次更新；原13前瞻和份额观察器保持。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="预期兑现退出完整交付与归档")
    parser.add_argument("action", choices=("deliver", "close"))
    {"deliver": deliver, "close": close}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
