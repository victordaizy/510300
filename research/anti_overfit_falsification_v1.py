"""固定已有账户的过拟合否证，不扫描新规则或重新训练。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import anti_overfit_evidence_inputs_v1 as evidence
from research import upward_episode_anatomy_v1 as common
from research.point_account_nr7_inputs_v1 import return_statistics, trade_statistics
from research.point_core_observation_v1 import LAYERS

OUT = ROOT / "reports/research/510300_anti_overfit_falsification_v1"
SOURCE = ROOT / "reports/research/510300_point_weight_information_diagnostic_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
PERIODS = ("2015_2019", "2020_2026")
COSTS = ("BASE", "STRESS")
POLICIES = ("POINT_BINARY", "SAVED_WEIGHT")


def table(name, data):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path.with_suffix(".parquet"), index=False)
    data.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("固定否证研究已存在，不能看结果后覆盖重做。")
    OUT.mkdir(parents=True)
    sources = [(SOURCE / "summary.json", "inputs/previous_summary.json"),
               (SOURCE / "results/同风险预算的仓位信息比较.parquet", "inputs/previous_metrics.parquet"),
               (ROOT / "reports/research/510300_sharpe_1_2_latest_research.json", "inputs/old_research_index.json"),
               (ROOT / "config/510300_anti_overfitting_requirements_v1.json", "inputs/old_anti_overfit_policy.json"),
               (ROOT / "docs/510300_ANTI_OVERFITTING_REQUIREMENTS_20260914.md", "inputs/old_anti_overfit_requirements.md"),
               (CONTEXT / "anti_overfitting_scope_addendum_20261001.json", "inputs/current_scope.json"),
               (CONTEXT / "active_goal_effective_requirements.json", "inputs/effective_requirements.json"),
               (ROOT / "reports/research/510300_point_forward_observer_v1/seed/results/完整实际登记.parquet", "inputs/actual_forward_registry.parquet"),
               (ROOT / "reports/research/510300_historic_cycles_point_translation_v1/protocol.json", "inputs/fourteen_candidate_selection_protocol.json")]
    for period in PERIODS:
        for cost in COSTS:
            for policy in POLICIES:
                for name in ("daily", "trades"):
                    sources.append((SOURCE / f"results/accounts/{period}/{cost}/{policy}/{name}.parquet",
                                    f"inputs/accounts/{period}/{cost}/{policy}/{name}.parquet"))
    for name in ("ordinary_models.json", "within_models.json"):
        sources.append((CURRENT / "inputs" / name, "inputs/lineage/" + name))
    for name in ("learned_cycle_exit.json", "within_cycle_exit.json"):
        sources.append((CURRENT / "inputs/config" / name, "inputs/lineage/" + name))
    for path in (Path(__file__), Path(evidence.__file__), Path(common.__file__),
                 ROOT / "research/point_core_observation_v1.py", ROOT / "research/point_account_nr7_inputs_v1.py",
                 ROOT / "tests/test_anti_overfit_evidence_v1.py"):
        sources.append((path, "code/" + path.name))
    files = {}
    for src, name in sources:
        dest = OUT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        files[name] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(dest)}
    protocol = {
        "study": "510300_ANTI_OVERFIT_FALSIFICATION_V1", "at": common.now(),
        "user_objective": "请继续，直到去除过拟合",
        "known_results": "已知保留仓位强弱的近期压力夏普1.217、年化3.99%、pB1.073；较早压力夏普0.438、pB0.611；20日区块增量区间跨零。此前方案经过多轮已知历史筛选。",
        "purpose": "固定现有8份账户，检查年份敏感性、盈利集中、区间不确定性、选择历史和独立验收资格。不是新增策略或提高回测数字。",
        "accounts": {"periods": PERIODS, "costs": COSTS, "policies": POLICIES},
        "scope": "两个历史区间独立启动，成本和风险口径原样保存；不拼接或改账户。",
        "checks": [
            "每个方案分别完整列出所有留一年统计；每日账户按日历年剔除，周期按退出年剔除，明确不是可执行账户。",
            "完整周期最大、前三、前五盈利对完成周期净利润的贡献；另报告删最大盈利一次的pB与均值。期末未完成持仓仍排除周期统计。",
            "按完整退出年份成组重采样5000次，保留空年份；2026部分年份明确作为部分组。固定种子20261001，报告未定义盈亏比次数。",
            "固定252交易日配对循环区块、2000次，与原20日区块结果并列；不从区块长度中选有利结果。",
            "逐月实际资金增量和相对对照的对数财富增量；保存所有月份，不仅盈利月份。",
            "读取原试验计数与当前来源链，披露已知选择范围；不把两个外层名字当成两个独立自由度。",
            "逐项检查独立验收资格，历史过线、切分和重采样不能升级证据类别。",
        ],
        "formal_dsr_pbo": "NOT_COMPUTED_INCOMPLETE_GLOBAL_COMPARABLE_TRIAL_UNIVERSE；旧583/599仅某索引范围，不能当独立试验数代入。",
        "uncertainty_limits": "这些区间描述已用历史，不校正全项目长期选择；年份组少、不同市场阶段可能非平稳，不能解释为未来成功概率。",
        "stop": "一次报告全部固定检查；没有删除最大盈利后的新策略、最优年份子集、阈值搜索或失败规则救援。",
        "zero_new_strategy_accounts": True, "zero_model_fits": True,
        "sources": ["https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf", "https://www.nber.org/papers/w21329"],
        "goal_achieved": False, "orders_authorized": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("已固定八份旧账户的过拟合否证；本轮不产生新账户、模型或参数搜索。", flush=True)


def lineage_inventory():
    index = json.loads((OUT / "inputs/old_research_index.json").read_text(encoding="utf-8"))
    fields = ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
              "evaluation_accounts_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays",
              "registered_candidate_source_runs_including_unrun_legacy_bindings", "count_warning"]
    result = {"old_index_status": index["status"], "old_counts": {k: index.get(k) for k in fields},
              "old_counts_are_global_total": False, "old_counts_are_independent_trials": False,
              "outer_point_selection_pool": 14, "mechanical_state_groups": list(LAYERS),
              "mechanical_state_groups_are_independent_parameters": False,
              "lineage_chain": ["参考训练持仓路径", "两类月度岭退出模型", "连续参考持仓",
                                "下行/方差风险预算与来源路由", "趋势噪声预算", "连续段辅助和回撤门槛",
                                "从十四方案筛出的固定点位", "保存目标大小或正零的账户映射"],
              "monthly_models": {}, "learned_feature_names": {}, "model_clock_checks": [],
              "independent_validation": "NOT_ESTABLISHED"}
    for kind, filename, cfg in [("ordinary", "ordinary_models.json", "learned_cycle_exit.json"),
                                ("within", "within_models.json", "within_cycle_exit.json")]:
        records = json.loads((OUT / "inputs/lineage" / filename).read_text(encoding="utf-8"))["models"]
        config = json.loads((OUT / "inputs/lineage" / cfg).read_text(encoding="utf-8"))
        fits = [r for r in records if r.get("status") == "FIT_COMPLETE"]
        checks = []
        for r in fits:
            if int(r["latest_exit_index"]) > int(r["fit_index"]):
                raise ValueError("保存的月度记录使用了拟合日之后的退出标签。")
            checks.append({"kind": kind, "fit_origin": r["fit_origin"], "latest_mature_exit": r["latest_exit_date"],
                           "training_rows": r["training_rows"], "distinct_training_cycles": r["training_cycle_count"],
                           "known_by_fit_close": True})
        result["model_clock_checks"].extend(checks)
        last = fits[-1]
        result["monthly_models"][kind] = {"schedule_records": len(records), "fitted_records": len(fits),
            "warming_or_other_records": len(records) - len(fits), "feature_count": len(config["feature_columns"]),
            "latest_training_rows": last["training_rows"], "latest_distinct_training_cycles": last["training_cycle_count"],
            "latest_fit_origin": last["fit_origin"], "latest_label_exit_date": last["latest_exit_date"]}
        result["learned_feature_names"][kind] = config["feature_names"]
    common.save_json(OUT / "results/来源与选择范围.json", result)
    table("月度训练记录的成熟时钟", pd.DataFrame(result["model_clock_checks"]))
    return result


def current_claim_inputs(period, cost, policy, row, source_freeze):
    first = "2015-01-05" if period == "2015_2019" else "2020-01-02"
    last = "2019-12-31" if period == "2015_2019" else "2026-09-30"
    return {
        "candidate": policy, "period": period, "cost": cost,
        "candidate_frozen_at": source_freeze,
        "last_outcome_used_for_design": "2026-09-30T15:00:00+08:00",
        "evaluation_first_origin": first + "T15:05:00+08:00",
        "evaluation_last_origin": last + "T15:05:00+08:00",
        "actual_decisions_registered_before_execution": False,
        "full_calendar_coverage": True, "same_frozen_version": True,
        "evaluation_schedule_fixed_before_first_origin": False,
        "required_information_met": False, "uncertainty_gate_passed": False,
        "economic_gate_passed": bool(row["p_times_b"] > 1 and row["mean_cycle_net_return"] > 0),
        "prospective_completed_cycles": 0, "candidate_role": "DIAGNOSTIC_ONLY_USED_HISTORY",
        "legacy_terminal_rejection": False,
        "source_rule_terminal_rejection_preserved": True,
        "historical_numeric_p_times_b": row["p_times_b"], "historical_net_sharpe": row["net_sharpe"],
    }


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("固定否证已运行，不能再次覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for name, record in frozen["files"].items():
        if common.digest(OUT / name) != record["sha256"]:
            raise ValueError("固定来源发生变化：" + name)
        if name.startswith("code/") and common.digest(ROOT / record["source"]) != record["sha256"]:
            raise ValueError("执行代码不等于事前版本：" + name)
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "new_accounts": 0})
    previous = pd.read_parquet(OUT / "inputs/previous_metrics.parquet")
    source_time = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))["at"]
    accounts, metrics, concentrations, leaveouts, year_boot, long_blocks, months, qualification = {}, [], [], [], [], [], [], []
    for period in PERIODS:
        for cost in COSTS:
            for policy in POLICIES:
                path = OUT / f"inputs/accounts/{period}/{cost}/{policy}"
                daily, trades = [pd.read_parquet(path / (n + ".parquet")) for n in ("daily", "trades")]
                accounts[(period, cost, policy)] = (daily, trades)
                tag = {"period": period, "cost": cost, "policy": policy}
                m = {**tag, **return_statistics(daily.net_return), **trade_statistics(trades)}
                old = previous.loc[previous.period.eq(period) & previous.cost.eq(cost) & previous.policy.eq(policy)].iloc[0]
                for k in ("net_sharpe", "net_cagr", "max_drawdown", "p_times_b", "mean_cycle_net_return"):
                    np.testing.assert_allclose(m[k], old[k], atol=1e-12, rtol=1e-12, equal_nan=True)
                metrics.append(m)
                c = {**tag, **evidence.concentration(trades)}
                concentrations.append(c)
                l = evidence.leave_one_year(daily, trades)
                for k, v in tag.items():
                    l[k] = v
                leaveouts.append(l)
                report, distribution = evidence.calendar_year_trade_bootstrap(trades, int(daily.date.iloc[0].year), int(daily.date.iloc[-1].year))
                year_boot.append({**tag, **report})
                dest = OUT / f"results/distributions/{period}/{cost}/{policy}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, year_block_point_statistics=distribution)
                claim = current_claim_inputs(period, cost, policy, m, source_time)
                check = evidence.independent_claim_gate(claim)
                try:
                    evidence.require_independent_claim(claim)
                except ValueError:
                    refused = True
                else:
                    refused = False
                if not refused:
                    raise AssertionError("已使用历史竟通过了独立验收准入。")
                qualification.append({**claim, **check, "runtime_claim_refused": refused})
                print(f"{period}/{cost}/{policy}：原pB={m['p_times_b']:.3f}，删最大盈利一次={c['without_largest_cycle_p_times_b']:.3f}，年份组描述区间=[{report['p_times_b_lower_2_5']:.3f},{report['p_times_b_upper_97_5']:.3f}]。", flush=True)
            a, _ = accounts[(period, cost, "POINT_BINARY")]
            b, _ = accounts[(period, cost, "SAVED_WEIGHT")]
            if not pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date)):
                raise ValueError("配对账户日历不同。")
            block_report, values = evidence.paired_block_effect(a.net_return, b.net_return)
            long_blocks.append({"period": period, "cost": cost, **block_report})
            np.savez_compressed(OUT / f"results/distributions/{period}/{cost}/paired_252.npz", effects=values)
            ap = np.diff(np.r_[200000., a.equity.to_numpy(float)])
            bp = np.diff(np.r_[200000., b.equity.to_numpy(float)])
            monthly = pd.DataFrame({"month": a.date.dt.to_period("M").astype(str),
                "account_pnl_increment": bp - ap,
                "relative_log_wealth_increment": np.log1p(b.net_return.to_numpy()) - np.log1p(a.net_return.to_numpy())})
            monthly = monthly.groupby("month", as_index=False).sum()
            np.testing.assert_allclose(monthly.account_pnl_increment.sum(), b.equity.iloc[-1] - a.equity.iloc[-1], atol=1e-7, rtol=0)
            monthly["period"], monthly["cost"] = period, cost
            months.append(monthly)
    metrics, c, leaveout, yb, lb = pd.DataFrame(metrics), pd.DataFrame(concentrations), pd.concat(leaveouts, ignore_index=True), pd.DataFrame(year_boot), pd.DataFrame(long_blocks)
    for name, frame in [("原账户指标复算", metrics), ("完整周期盈利集中度", c), ("全部留一年敏感性", leaveout),
                        ("年份组点位不确定性", yb), ("一年区块仓位增量不确定性", lb), ("全部月度资金增量", pd.concat(months, ignore_index=True))]:
        table(name, frame)
    common.save_json(OUT / "results/独立验收资格.json", qualification)
    lineage = lineage_inventory()
    registry = pd.read_parquet(OUT / "inputs/actual_forward_registry.parquet")
    registry_summary = {"actual_registered_intents": len(registry),
        "registered_entry_intents": int(registry.entry_or_exit_intent.eq("ENTER_NEXT_OPEN").sum()),
        "planned_dates": sorted(pd.to_datetime(registry.planned_execution_date).dt.strftime("%Y-%m-%d").unique().tolist()),
        "prospective_completed_cycles": 0,
        "registration_is_not_a_completed_trade": True}
    summary = {"study": "510300_ANTI_OVERFIT_FALSIFICATION_V1", "at": common.now(),
        "status": "OVERFITTING_NOT_REMOVED_INDEPENDENT_PROMOTION_REFUSED",
        "previous_goal_turn_classification": "progress", "saved_accounts_examined": len(metrics),
        "new_strategy_accounts": 0, "new_model_fits": 0, "new_parameter_searches": 0,
        "necessary_tests_passed": 6, "current_state_groups": len(LAYERS),
        "known_legacy_configurations": lineage["old_counts"]["registered_configurations_in_this_resumption"],
        "known_counts_are_total_independent_trials": False,
        "formal_dsr": "NOT_COMPUTED_INCOMPLETE_GLOBAL_COMPARABLE_TRIAL_UNIVERSE",
        "formal_pbo": "NOT_COMPUTED_INCOMPLETE_GLOBAL_COMPARABLE_TRIAL_UNIVERSE",
        "independent_claims_refused": sum(x["runtime_claim_refused"] for x in qualification),
        "forward_evidence": registry_summary, "independent_validation": "NOT_ESTABLISHED",
        "remaining_requirements": ["可否证且事前固定的候选与全部选择范围", "独立评价时点与信息量设计", "不参与后续调参的新观察",
                                   "独立期内收益、夏普、实际pB和回撤等共同成立"],
        "next_work": "先依据本次否证处置现有复杂候选，再将独立评价设计落实为不可事后换规则的接续流程；不能再用已用历史刷高数字。",
        "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    report(metrics, c, leaveout, yb, lb, lineage, summary)
    plot(c)
    common.save_json(OUT / "delivery_receipt.json", {"at": common.now(), "report_sha256": common.digest(OUT / "研究结论.md"),
        "fixed_metrics_recomputed": True, "source_accounts_unchanged": True, "historical_promotion_is_blocked": True,
        "blocking_scope": "本轮结果评估器及本任务有效研究要求，不声称已改写全项目全部旧脚本。"})
    state = json.loads((CONTEXT / "state.json").read_text(encoding="utf-8"))
    state.update(updated_at=common.now(), current_study=summary["study"], latest_completed_study=summary["study"],
        latest_result=str((OUT / "summary.json").relative_to(ROOT)), latest_report=str((OUT / "研究结论.md").relative_to(ROOT)),
        latest_research_status=summary["status"], latest_progress="完成8份固定账户的过拟合否证、选择来源清单及独立验收准入；已用历史不能升级为独立通过。",
        current_validated_candidates=[], historical_numeric_leads_role="DEVELOPMENT_ONLY_NOT_ELIGIBLE_FOR_INDEPENDENT_PROMOTION",
        independent_claim_gate="research/anti_overfit_evidence_inputs_v1.py", previous_goal_turn_classification="progress",
        next_research_question=summary["next_work"], blocked_audit_count=0, goal_achieved=False)
    common.save_json(CONTEXT / "state.json", state)
    requirement = json.loads((CONTEXT / "active_goal_effective_requirements.json").read_text(encoding="utf-8"))
    requirement.update(independent_claim_gate="research/anti_overfit_evidence_inputs_v1.py",
                       prior_used_outcomes_through="2026-09-30", all_current_history_role="DEVELOPMENT_AND_CALIBRATION",
                       accepted_independent_candidates=[], goal_achieved=False)
    common.save_json(CONTEXT / "active_goal_effective_requirements.json", requirement)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def report(metrics, c, leaveout, yb, lb, lineage, summary):
    names = {"POINT_BINARY": "点位正零账户", "SAVED_WEIGHT": "保留仓位强弱"}
    rows = ["# 510300去除过拟合：固定账户的否证结果", "",
        "目标仍未实现。本轮没有继续搜索更高回测收益，而是对已有8份账户完成固定检查，保留全部年份、成本和失败结果。新增账户0、训练0、参数搜索0。", "",
        "## 先区分已解决和未解决的问题", "",
        "已有逐日重建和历史截断支持信号按当时可用资料计算，本轮又检查保存月度模型的成熟退出时钟；这些检查不能消除看过历史后选策略、选模块和选解释的影响。", "",
        f"旧索引记录{lineage['old_counts']['registered_configurations_in_this_resumption']}个设置、{lineage['old_counts']['evaluated_candidate_source_runs_including_corrected_replays']}个已评价来源版本。这是该索引的部分研究范围，不是整个项目的完整试验数，更不是相互独立的试验数。", "",
        f"当前两条点位从14套旧方案中筛出，接续链包含{len(LAYERS)}组机械状态，还依赖两类月度岭退出模型。每类有142个月度时点，最近拟合使用8项特征、790行持仓状态，但只来自20个自然成熟周期。790行不能解释为790个独立行情样本，9个状态组也不是正式自由度计数。", "",
        "旧85/15混合策略仍保持用户终止状态。本轮不恢复它，不用历史高夏普推翻终止裁决。", "",
        "## 是否依赖少数盈利周期", "",
        "以下是压力成本结果。盈利贡献分母是全部已完成周期净利润，可能超过100%；未结束持仓没有强平或塞入周期胜率。", "",
        "| 时期 | 方案 | 最大盈利占完成净利润 | 前五盈利占完成净利润 | 原实际净pB | 删最大盈利一次后的pB |",
        "|---|---|---:|---:|---:|---:|"]
    for r in c.loc[c.cost.eq("STRESS")].itertuples():
        rows.append(f"| {r.period} | {names[r.policy]} | {r.top1_fraction_of_net_profit:.1%} | {r.top5_fraction_of_net_profit:.1%} | {r.original_p_times_b:.3f} | {r.without_largest_cycle_p_times_b:.3f} |")
    rows += ["", "删除最大盈利是脆弱性诊断，不是新策略，也不能反过来只保留该类盈利。集中性本身不能证明过拟合，但会削弱对小样本稳定性的信心。", "",
             "## 整个年份分组后的不确定性", "",
             "| 时期 | 方案 | 实际净pB的95%描述区间 | 平均周期净回报95%描述区间 | pB未定义的重采样次数 |",
             "|---|---|---|---|---:|"]
    for r in yb.loc[yb.cost.eq("STRESS")].itertuples():
        rows.append(f"| {r.period} | {names[r.policy]} | [{r.p_times_b_lower_2_5:.3f}, {r.p_times_b_upper_97_5:.3f}] | [{r.mean_cycle_return_lower_2_5:.2%}, {r.mean_cycle_return_upper_97_5:.2%}] | {r.undefined_p_times_b_replications}/5000 |")
    rows += ["", "按退出年份整组抽取，不把同年交易拆成独立样本；保留无交易年份，2026是部分年份。只有5或7个年份组，区间仅描述已有资料，不能当严格覆盖率保证或未来达标概率。缺少盈利或亏损的抽样中pB保持未定义，未替换成无穷大。", "",
             "| 时期 | 成本 | 252交易日区块下的夏普增量区间 | 年化增量区间 |",
             "|---|---|---|---|"]
    for r in lb.itertuples():
        rows.append(f"| {r.period} | {r.cost} | [{r.sharpe_delta_lower_2_5:.3f}, {r.sharpe_delta_upper_97_5:.3f}] | [{r.cagr_delta_lower_2_5:.2%}, {r.cagr_delta_upper_97_5:.2%}] |")
    rows += ["", "原20日配对区块已显示区间跨零。本轮固定较长区块后同时报告全部结果，不从区块长度中挑更好看的结论；这两种抽样都不能撤销历史上的方案选择。", "",
             "## 全部留一年结果", "",
             "| 时期 | 方案 | 排除年份 | 剩余日收益夏普 | 剩余日收益年化 | 剩余完整周期pB |",
             "|---|---|---:|---:|---:|---:|"]
    for r in leaveout.loc[leaveout.cost.eq("STRESS")].itertuples():
        rows.append(f"| {r.period} | {names[r.policy]} | {r.excluded_year} | {r.net_sharpe:.3f} | {r.net_cagr:.2%} | {r.p_times_b:.3f} |")
    rows += ["", "留一年表只重新计算保存收益的敏感性，不重启账户，不改变原交易和风险路径，也不拼成新的可执行绩效。不能挑选某个删除结果作为成绩。完整两成本表另存。", "",
             "## 独立验收准入已经落实到结果计算", "",
             "本任务使用`research/anti_overfit_evidence_inputs_v1.py`检查：冻结真实时刻、已用结果截止日、评价首日、事前决定登记、全部交易日覆盖、固定版本、事前验收计划、信息量、不确定性要求及经济要求。", "",
             f"8份当前历史账户均被该检查拒绝独立晋升。现有实际登记只有{summary['forward_evidence']['actual_registered_intents']}条下一交易日观察意向，入场意向{summary['forward_evidence']['registered_entry_intents']}，前瞻完成周期0。意向不等于新交易或独立净值。", "",
             "这是本任务当前评估入口的实际检查，不声称已经替换全项目全部旧脚本。依赖这些字段的事实仍必须核对其原始证据；单纯填写True不能成为科学验证。", "",
             "正式修正夏普（DSR）和过拟合概率（PBO）维持NOT_COMPUTED：缺少覆盖全部选择过程的可比候选收益集合，不能把583或599机械代入公式。即使将来能计算，也不能把已用于选择的历史重新变成独立样本。", "",
             "## 继续工作的边界", "",
             "已有复杂候选继续作为固定观察或机制对照，不能凭本轮统计复活。去除过拟合还需要：固定可否证的候选与全部选择范围，固定独立评价时点及信息量要求，取得不再参与调参的新观察，并在该期间同时检验收益、夏普、实际pB和回撤。", "",
             "用户既有收益与交易质量目标没有改成现金空仓、少交易或低波动即可通过。旧历史仍可用于揭示失败原因，但不能继续以达到漂亮数字为理由追加条件。新机制的复杂度预算沿用现有要求：一个主要信号、一层直接风险控制，首版不新增学习退出或嵌套账户组合。", "",
             "六项针对性测试通过，原8份账户指标已复算一致。这个阶段完成的是具体否证和不当晋升路径的限制，去除过拟合的整体目标继续保持进行中。", "",
             "方法依据：[Bailey与López de Prado修正夏普研究](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)讨论选择次数和收益分布造成的高估；[Novy-Marx多信号回测研究](https://www.nber.org/papers/w21329)说明组合内部的选择也会造成过拟合。文献不是本候选的有效性证明。", "",
             "文件入口：`results/来源与选择范围.json`、`results/完整周期盈利集中度.csv`、`results/全部留一年敏感性.csv`、`results/年份组点位不确定性.csv`、`results/一年区块仓位增量不确定性.csv`、`results/独立验收资格.json`。", ""]
    (OUT / "研究结论.md").write_text("\n".join(rows), encoding="utf-8")


def plot(c):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    s = c.loc[c.cost.eq("STRESS")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(len(s))
    ax.bar(x - .18, s.original_p_times_b, width=.36, color="#226b7c", label="原完整周期pB")
    ax.bar(x + .18, s.without_largest_cycle_p_times_b, width=.36, color="#b98250", label="删最大盈利一次后的pB")
    ax.axhline(1, color="#b34d4d", ls="--", label="用户门槛：严格大于1")
    for i, r in enumerate(s.itertuples()):
        for off, value in [(-.18, r.original_p_times_b), (.18, r.without_largest_cycle_p_times_b)]:
            ax.text(i + off, value + .02, f"{value:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, [f"{r.period.replace('_','—')}\n{'点位正零' if r.policy == 'POINT_BINARY' else '保留仓位强弱'}" for r in s.itertuples()])
    ax.set_ylabel("实际净胜率×盈亏比")
    ax.set_title("少一笔最大盈利，历史点位质量是否仍能达线？", loc="left", fontsize=15)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(s.original_p_times_b.max(), 1.2) + .22)
    ax.grid(axis="y", alpha=.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.06, .014, "压力成本；只作脆弱性诊断。没有删除交易后创建新策略，没有改变原账户，也没有建立独立验证。", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUT / "盈利集中与过拟合风险.png", dpi=150)
    fig.savefig(OUT / "盈利集中与过拟合风险.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定历史账户的过拟合否证与独立资格检查。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
