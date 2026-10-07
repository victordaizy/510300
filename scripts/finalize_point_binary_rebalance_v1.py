"""整理单一仓位简化结果及已有训练机制，不重跑账户或拟合模型。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import upward_episode_anatomy_v1 as common
from research.learned_cycle_exit_v1 import training_rows

OUT = ROOT / "reports/research/510300_point_binary_rebalance_diagnostic_v1"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
OVERALL = ROOT / "reports/research/510300_trade_quality_and_sharpe_summary_20261001"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
USER = "去除过拟合，并把我们的模型完善做得更好，夏普率更高，收益率更高"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    if (OUT / "followup_delivery_receipt.json").exists():
        raise RuntimeError("结果整理已经完成，不重新覆盖。")
    result = read(OUT / "summary.json")
    if result["status"] != "REJECTED_FROZEN" or result["new_account_evaluations"] != 4:
        raise ValueError("整理内容只适用于本次已保存的四账户拒绝结果。")
    config_path = CURRENT / "inputs/config/learned_cycle_exit.json"
    sample_path = CURRENT / "inputs/old_training_samples.parquet"
    models_path = CURRENT / "inputs/ordinary_models.json"
    cfg, models = read(config_path), read(models_path)["models"]
    latest = [item for item in models if item["status"] == "FIT_COMPLETE"][-1]
    samples = pd.read_parquet(sample_path)
    rows, ids = training_rows(samples, latest["fit_index"], cfg)
    totals = rows.groupby("cycle_id").sample_weight.sum()
    np.testing.assert_allclose(totals, 1., rtol=0, atol=1e-12)
    if len(rows) != latest["training_rows"] or len(ids) != latest["training_cycle_count"]:
        raise ValueError("保存训练样本与最近月度记录不一致。")
    if not rows.exit_index.le(latest["fit_index"]).all():
        raise ValueError("训练成员包含当时尚未成熟周期。")
    old_reviews = []
    for stem, title in [("robust_cycle_exit", "稳健损失与周期截距"),
                        ("single_component_exit", "八项状态压缩为一个评分"),
                        ("sparse_vintage_exit", "稀疏化学习退出"),
                        ("cycle_serial_error_exit", "修正周期内相邻误差")]:
        path = ROOT / "reports/research" / ("510300_" + stem + "_v1") / "result.json"
        old = read(path)
        stress = [x for x in old["primary"] if x["cost"] == "STRESS"][0]
        old_reviews.append({"study": old["study_id"], "title": title, "result": relative(path),
                            "source_sha256": common.digest(path), "status": old["status"],
                            "historical_point_target_met": old["historical_point_target_met"],
                            "goal_achieved": old["goal_achieved"],
                            "independent_validation": old["independent_validation"],
                            "original_pressure_metrics": {key: stress[key] for key in (
                                "annualized_return", "net_sharpe", "max_drawdown", "trading_days")},
                            "comparison_limit": "原时期与原账户口径，不与本轮共同风险账户直接排名；未达各自预设目标，不声称该方法普遍无效。"})
    training_review = {
        "at": common.now(), "scope": "已存在的训练机制与四项相关固定研究的只读核对",
        "new_model_fits": 0, "new_account_evaluations": 0,
        "latest_fit_origin": latest["fit_origin"], "training_rows": len(rows),
        "distinct_training_cycles": len(ids), "minimum_total_weight_per_cycle": float(totals.min()),
        "maximum_total_weight_per_cycle": float(totals.max()),
        "mature_before_fit": True, "ridge_alpha_already_present": cfg["ridge_alpha"],
        "cycle_equal_weight_already_present": True,
        "independent_sample_count": "NOT_ESTABLISHED；20个自然周期不自动等于20个相互独立样本，更不能视790行为独立样本。",
        "finding": "没有发现把长周期每一行等权从而重复放大的训练错误；周期等权、岭正则和成熟时钟已存在，不能重新包装为本轮改进。",
        "input_sources": [{"path": relative(p), "sha256": common.digest(p)} for p in (config_path, sample_path, models_path,
                          ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py")],
        "prior_fixed_studies": old_reviews,
        "decision": "保留既有训练保护，不按当前历史指标追加惩罚强度、稀疏度或确认天数搜索；这些旧方向没有为本轮提供可直接采用的新版本。",
        "goal_achieved": False,
    }
    common.save_json(OUT / "model_training_review.json", training_review)

    lines = ["# 模型训练与改进判断", "",
             "这次检查服务于实际模型改进。已完成的四个新账户比较证明：在相同再平衡规则下，删除仓位强弱没有提高收益或夏普，固定简化版本拒绝。", "",
             f"退出模型最近一次保存拟合为{latest['fit_origin']}。本次重新选择当时已成熟成员，得到{len(rows)}行状态、{len(ids)}个完整周期；每个周期总权重已为1。岭惩罚已为{cfg['ridge_alpha']}，未结束的周期被排除。没有重新拟合，也没有更改这些规则。", "",
             "这里没有发现可用“添加周期等权”修复的训练错误。样本仍然少，而且来自连续市场；20个周期不能自动当作20个独立样本，790行尤其不能视为790个独立试验。", "",
             "| 已有改进方向 | 原固定研究判断 | 本轮处理 |", "|---|---|---|"]
    for item in old_reviews:
        lines.append(f"| {item['title']} | {'已达原目标' if item['historical_point_target_met'] else '未达原目标'}，独立验证未建立 | 复用原结论，不改参数重试 |")
    lines += ["", "这四项旧研究采用各自的时期和账户口径。保存的数字只说明各自固定版本结果，不与本轮20万元共同风险账户直接排名，也不说明相应技术在所有市场无效。", "",
              "当前有证据支持的取舍是：保留原仓位强弱为开发线索，拒绝本次统一目标版本；NR7等已失败的次数补充仍不加入。当前缺口在于跨期交易质量和独立证据，继续往旧模型叠加指标或搜索正则强度没有现成依据。", "",
              "后续若发现新的事前可用信息，应先固定一种能说明原因的改动，再在相同账户内看净年化、净夏普、实际净pB、回撤和自然次数。全部已使用历史继续标记为开发资料；既有未见期登记保持原方案，不依据这次结果替换候选。", "",
              "本次没有实现更高且更可靠的新模型；已经明确排除一种有收益代价的简化，并避免把既有训练保护误报为新成果。原方案本身仍没有取得跨期与独立验证上的通过结论。", ""]
    (OUT / "模型训练与改进判断.md").write_text("\n".join(lines), encoding="utf-8")

    metrics = pd.read_parquet(OUT / "results/统一账户比较.parquet")
    recent = metrics.loc[metrics.period.eq("2020_2026") & metrics.cost.eq("STRESS")].set_index("policy")
    earlier = metrics.loc[metrics.period.eq("2015_2019") & metrics.cost.eq("STRESS")].set_index("policy")
    a, b, c = [recent.loc[name] for name in ("POINT_BINARY", "SAVED_WEIGHT", "BINARY_REBALANCE")]
    before_report = OVERALL / "全部判断与收益夏普进展.md"
    archive = OVERALL / "archive_before_binary_rebalance"
    archive.mkdir(exist_ok=False)
    for src, name in ((before_report, before_report.name), (OVERALL / "summary.json", "summary.json"),
                      (CONTEXT / "state.json", "continuation_state.json"),
                      (CONTEXT / "active_goal_effective_requirements.json", "effective_requirements.json")):
        shutil.copyfile(src, archive / name)
    text = before_report.read_text(encoding="utf-8")
    insert = ["## 2026年10月2日：针对模型简化的实际比较", "",
              "最新要求是同时减少过拟合、提高净收益和净夏普。本轮新增一个固定简化版本、四条账户，复用八条对照；没有训练新模型或搜索参数。三项必要测试通过，十二份保存账户核对完成。此前已报告的16条新增账户不重复计入本轮。", "",
              "先前比较同时改变仓位强弱与持仓中再平衡。本轮控制账户引擎后，只把所有已知正目标改为原预算上限50%，零和未知保持原样，实际份额仍受原风险上限约束。", "",
              "| 近期压力成本版本 | 净年化 | 净夏普 | 回撤 | 完整周期 | 实际净pB | 平均仓位 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    labels = {"POINT_BINARY": "旧二值点位", "SAVED_WEIGHT": "保留仓位强弱", "BINARY_REBALANCE": "统一正目标、同样再平衡"}
    for name in ("POINT_BINARY", "SAVED_WEIGHT", "BINARY_REBALANCE"):
        r = recent.loc[name]
        insert.append(f"| {labels[name]} | {r.net_cagr:.2%} | {r.net_sharpe:.3f} | {r.max_drawdown:.2%} | {int(r.completed_cycles)} | {r.p_times_b:.3f} | {r.mean_exposure:.2%} |")
    reduced, retained = earlier.loc["BINARY_REBALANCE"], earlier.loc["SAVED_WEIGHT"]
    insert += ["", f"统一正目标比保留强弱少{b.ending_equity-c.ending_equity:,.2f}元期末权益；其中实际份额毛损益少{b.gross_at_actual_quantities_pnl-c.gross_at_actual_quantities_pnl:,.2f}元、费用多{c.total_commission+c.total_slippage-b.total_commission-b.total_slippage:,.2f}元。不是单纯费用造成。成交订单{int(b.orders)}→{int(c.orders)}，完整周期仍{int(c.completed_cycles)}，未增加真实机会。", "",
               f"较早2015—2019压力年化{retained.net_cagr:.2%}→{reduced.net_cagr:.2%}，夏普{retained.net_sharpe:.3f}→{reduced.net_sharpe:.3f}，实际净pB{retained.p_times_b:.3f}→{reduced.p_times_b:.3f}。两个时期、两档费用的收益和夏普均下降；统一目标版本固定拒绝，不继续搜索其他统一仓位或调整带。", "",
               "这补强了原仓位强弱的历史作用，却没有让原模型取得独立有效性。原模型较早实际净pB仍不足1，近期利润集中与既有反复选择的问题均保留。", "",
               "学习退出核对还确认：原训练已经按周期等权、有岭正则、只取自然成熟标签。最近790行状态来自20个周期，每周期总权重1；没有发现可用再次加入这些措施修复的错误。稳健损失、单一评分、稀疏学习与相邻误差修正四项方向已有未达各自固定目标的记录，本轮没有重跑或调参。", "",
               "目标仍未实现：本次没有形成收益和夏普均更高的新版本，没有把保留原模型称为去除过拟合。后续研究须针对新信息带来的事前净优势，旧历史不能反复改名为未见样本。", "",
               "详细入口：[本次受控账户比较](../510300_point_binary_rebalance_diagnostic_v1/研究结论.md)、[训练与改进判断](../510300_point_binary_rebalance_diagnostic_v1/模型训练与改进判断.md)。", ""]
    first_break = text.find("\n\n")
    text = text[:first_break+2] + "\n".join(insert) + "\n" + text[first_break+2:]
    text = text.replace("其来源是既有仓位强弱信息在不同日子的资金分配，以及相应成交成本变化。",
                        "该原比较同时改变了仓位强弱与再平衡，不能单凭原结果分离二者。10月2日新增的共同引擎比较补强了仓位强弱的作用，仍只是历史诊断。")
    text = text.replace("本轮新增16条账户测量", "10月1日前一账户阶段新增16条账户测量")
    text = text.replace("## 补充：去除过拟合后的当前判断", "## 10月1日：针对过拟合的否证结果")
    text = text.replace("最新一轮针对已有8份账户做否证", "该否证阶段针对已有8份账户做否证")
    before_report.write_text(text, encoding="utf-8")

    addendum = {"at": common.now(), "latest_user_instruction": USER,
                "joint_objective": "同时减少过拟合风险并提高扣费后全账户净年化与夏普；没有降低其中任一目标。",
                "research_universe": ["510300.SH", "CASH_CNY"], "capital_cny": 200000,
                "bars": ["DAILY", "PREVIOUS_COMPLETED_WEEK"], "priority": "LONG_FIRST",
                "quality_gate": "实际净pB>1且净均值>0；增加次数是软目标；共同资金成本风险口径。",
                "new_absolute_numeric_target": None,
                "completed_bounded_change": result["study"], "completed_status": result["status"],
                "new_accounts": 4, "reused_accounts": 8, "new_fits": 0,
                "all_existing_history_role": "DEVELOPMENT_AND_CALIBRATION",
                "no_zero_overfit_claim": True, "forward_protocol_unchanged": True,
                "goal_achieved": False, "orders_authorized": False}
    addendum_path = CONTEXT / "joint_model_quality_scope_addendum_20261002.json"
    common.save_json(addendum_path, addendum)
    requirements_path = CONTEXT / "active_goal_effective_requirements.json"
    requirements = read(requirements_path)
    requirements.update(at=common.now(), latest_user_instruction=USER,
                        latest_user_anti_overfitting_instruction=USER, latest_user_account_instruction=USER,
                        current_priority="ANTI_OVERFITTING_AND_NET_RETURN_SHARPE_IMPROVEMENT",
                        joint_model_quality_scope_addendum=addendum_path.name,
                        accepted_independent_candidates=[], goal_achieved=False)
    common.save_json(requirements_path, requirements)
    state = read(CONTEXT / "state.json")
    state.update(updated_at=common.now(), latest_user_instruction=USER,
                 latest_completed_study=result["study"], latest_result=relative(OUT / "summary.json"),
                 latest_report=relative(OUT / "研究结论.md"), latest_research_status=result["status"],
                 latest_progress="控制再平衡后完成唯一仓位简化比较；四个场景收益与夏普均下降，拒绝该版本。确认周期等权和岭正则已经存在，复用四项旧训练方向结论。",
                 current_priority="ANTI_OVERFITTING_AND_NET_RETURN_SHARPE_IMPROVEMENT",
                 current_phase="JOINT_MODEL_SIMPLIFICATION_AND_NET_PERFORMANCE",
                 current_study=result["study"], new_accounts_this_continuation=4,
                 new_accounts_in_current_phase=4, new_investment_account_evaluations_this_continuation=4,
                 new_model_fits_this_continuation=0, new_point_replays_this_continuation=0,
                 internal_reference_replays_this_continuation=0, historical_model_refits_this_continuation=0,
                 new_bars_this_continuation=0, necessary_tests_passed_this_continuation=3,
                 previous_goal_turn_classification="progress", blocked_audit_count=0,
                 joint_model_quality_scope_addendum=addendum_path.name,
                 current_validated_candidates=[], whole_model_overfitting_removed=False,
                 latest_training_review=relative(OUT / "model_training_review.json"),
                 current_unmet_evidence="仍没有同时满足跨期实际净pB、收益与夏普改善及独立验证的新版本。",
                 next_research_question="本次统一目标版本拒绝，原仓位强弱只保留开发线索。下一项模型研究须先定位新的事前可用信息或独立机制，不能重启已失败窗口、权重和学习退出参数搜索；已有前瞻比较继续按原真实日历接续。",
                 goal_achieved=False, orders_authorized=False)
    common.save_json(CONTEXT / "state.json", state)
    overall = read(OVERALL / "summary.json")
    overall.update(updated_at=common.now(), latest_user_instruction=USER,
                   prior_account_phase_new_accounts=overall.get("new_accounts_this_user_turn", 16),
                   new_accounts_this_user_turn=4, necessary_tests_passed=3,
                   latest_study=result["study"], latest_status=result["status"],
                   simplified_target_candidate="REJECTED_FROZEN", saved_controls_recomputed=8,
                   current_study=relative(OUT / "summary.json"), training_review=relative(OUT / "model_training_review.json"),
                   report_sha256=common.digest(before_report), whole_model_overfitting_removed=False,
                   final_strategy_validated=False, goal_achieved=False)
    common.save_json(OVERALL / "summary.json", overall)
    common.save_json(OUT / "followup_delivery_receipt.json", {
        "at": common.now(), "script_sha256": common.digest(Path(__file__)),
        "training_review_sha256": common.digest(OUT / "model_training_review.json"),
        "training_report_sha256": common.digest(OUT / "模型训练与改进判断.md"),
        "overall_report_sha256": common.digest(before_report), "original_report_snapshot": relative(archive),
        "additional_fits_or_account_replays": 0, "goal_achieved": False})
    print("原训练成员已核对：790行、20周期、周期总权重1；没有新拟合。")
    print("四项相关训练研究原结果已整理，没有重复回测。")
    print("本轮结果、总评与有效目标已更新；仍未达成可靠的收益与夏普共同改进。")


if __name__ == "__main__":
    main()
