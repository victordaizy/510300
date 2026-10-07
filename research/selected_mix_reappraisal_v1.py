"""按新授权复核原85/15指数策略：固定原版复现、证据判读和前向资料状态。"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/research/510300_selected_mix_reappraisal_v1"
MODEL = "SELECTED_MIX_BAND10_SIMPLE2"
STUDY = "510300_SELECTED_MIX_REAPPRAISAL_V1"
LATEST = ROOT / "reports/research/510300_fixed_daily_continuation_v1/2026-09-16"
SEPTEMBER = ROOT / "reports/research/510300_september_monthly_continuation_v1"
OLD = ROOT / "reports/research/510300_incremental_selected_intent_mix_v1"
GRAPH = ROOT / "reports/research/510300_post_selection_extension_inputs_v1/dependency_graph.json"
REVIEW = ROOT / "reports/research/510300_strategy_review_diagnostics_v1"
NEIGHBOR = ROOT / "reports/research/510300_session_window_neighborhood_v1"
BOOTSTRAP = ROOT / "reports/research/510300_point_pass_fixed_diagnostic_v1"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def local_import_closure(seeds):
    pending, included = list(seeds), set()
    while pending:
        source = pending.pop()
        if source in included:
            continue
        included.add(source)
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                if name.startswith("research."):
                    target = ROOT.joinpath(*name.split(".")).with_suffix(".py")
                    if target.is_file() and target not in included:
                        pending.append(target)
    return included


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("本次原版复核已经冻结；请复用保存入口。")
    for folder in ["inputs", "code", "results"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"]
    save(root / "inputs/previous_mandate.json", mandate, True)
    instruction = "请给我实现它，我们之前不是有过一版过拟合的夏普1.2策略了吗，有没有可能它没有过拟合呢？"
    authority = {"at": now(), "user_instruction": instruction, "strategy": MODEL,
                 "authorization_interpretation": "重新检验并实现原策略的本地固定研究复现，不恢复参数搜索或交易执行。",
                 "prior_termination_preserved_as_historical_record": True, "fixed_original_replay_authorized": True,
                 "scope": ["510300.SH", "CASH_CNY"], "new_market_collection_enabled": False,
                 "parameter_optimization": False, "broker_orders": False, "automation_started": False,
                 "old_rule_exception": "原貌复核保留旧月度模型和最近20成熟周期规则；不改成两年日更后仍称原版。新策略的两年日更约束没有撤销。"}
    save(root / "authority_update.json", authority, True)
    shutil.copy2(__file__, root / "code/selected_mix_reappraisal_v1.py")
    graph = read(GRAPH)
    seeds = {ROOT / "research/strategy_review_full_graph_v1.py", ROOT / "research/post_selection_continuous_replay_v1.py"}
    source_files = local_import_closure(seeds)
    paths = source_files | {GRAPH, LATEST / "candidate_features.parquet", LATEST / "result.json",
             LATEST / "new_observed_daily_returns.csv", SEPTEMBER / "ridge_models.json", SEPTEMBER / "within_models.json",
             ROOT / "config/510300_incremental_selected_intent_mix_v1.json", ROOT / "config/510300_september_monthly_continuation_v1.json",
             ROOT / "config/510300_learned_cycle_exit_v1.json", ROOT / "config/510300_within_cycle_exit_v1.json",
             ROOT / "data/reference/510300_dividends.csv", ROOT / "data/reference/sse_trade_calendar_2026.csv",
             ROOT / "reports/research/510300_fixed_research_origin_v1/origin.json",
             NEIGHBOR / "portfolio_metrics.csv", NEIGHBOR / "protocol.json", NEIGHBOR / "early_baseline_clock_check.json",
             BOOTSTRAP / "bootstrap_intervals.csv", BOOTSTRAP / "bootstrap_two_cost_threshold_fractions.csv",
             REVIEW / "basic_receipt.json", REVIEW / "opening_price_sensitivity.csv",
             ROOT / "config/510300_anti_overfitting_requirements_v1.json",
             ROOT / "docs/510300_INCREMENTAL_SELECTED_INTENT_MIX_V1.md"}
    paths.update(ROOT / node["configuration"] for node in graph["nodes"])
    for cost in ["BASE", "STRESS"]:
        paths.update((LATEST / "accounts_run/accounts" / cost).glob("*/*.parquet"))
        paths.update((LATEST / "accounts_run/accounts" / cost).glob("*/checkpoint.json"))
        paths.add(OLD / "earlier_diagnostic" / cost / f"{MODEL}_ledger.parquet")
    protocol = {"study_id": STUDY, "at": now(), "strategy": MODEL,
                "primary_question": "高过拟合风险是否足以断言不存在真实优势，以及原版能否按固定规则完整复现。",
                "hypotheses": ["正期望存在，但历史筛选放大了绩效", "主要收益来自选择偶然性", "存在环境依赖且当前信息不足以区分"],
                "fixed_original": {"weights": [.85, .15], "outer_weight_band": .1, "signal_window": 60,
                                   "costs": "原BASE/STRESS", "annual_days": 242, "initial_capital": 200000,
                                   "model_coefficients": "原保存月度模型，按原成熟时钟使用，不重新拟合"},
                "reproduction": "主历史从各依赖节点的原起点重建22条账户至2026-09-16；与最后正式账本逐日比对，不从最终收益反推交易。较早四场景中的早期两账户直接复算。",
                "unchanged_original_end": "原来收盘估值的主历史仍用收盘；早期2019-12-31原终点开盘清算单独保留，不拼接两段净值。",
                "statistics": "复用已有5/20/60日区块区间及完整50/60/70日邻域；本次不再抽样择优，不运行删减模型。",
                "important_limits": ["两段历史都参与过选优，因此较早区间不是独立留出。",
                                     "完整邻域变体沿用60日版学习系数，只检验冻结系统输入扰动，不是重新训练后的50/70日算法。",
                                     "简单核心同时移除多个模块，不能替代完整原版的邻域结果或归因单一模块。",
                                     "原试验的选择家族与相关试验数未完整界定，不能给出策略无效的精确概率；DSR及PBO保持未计算。",
                                     "利润集中是脆弱性证据，不单独证明过拟合；历史正收益与过拟合膨胀可同时存在。"],
                "forward_check": "核对原研究起点后的真实记录日、按时到达日、有仓日、成交和完整周期；禁止把空仓零收益当交易能力验证。",
                "current_mandate_compatibility": "本轮原样复核具有窄范围历史例外；原策略不自动满足新提案的两年日更及50%库存预算，不能静默改规则。",
                "goal_achieved": False, "new_candidates": 0, "new_parameter_search": False,
                "new_market_downloads": 0, "orders_authorized": False}
    save(root / "protocol.json", protocol, True)
    save(root / "freeze.json", {"at": now(), "root": str(ROOT), "source_code_modules": len(source_files),
                                 "workspace_files": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)},
                                 "local_files": {p.relative_to(root).as_posix(): digest(p) for p in [root / "protocol.json", root / "authority_update.json", root / "code/selected_mix_reappraisal_v1.py"]}}, True)
    mandate.update(latest_user_instruction=instruction, current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "authority_update.json").relative_to(ROOT).as_posix(),
                   selected_strategy_reappraisal={"strategy": MODEL, "status": "FIXED_RESEARCH_REAPPRAISAL_AUTHORIZED",
                                                  "receipt": (root / "authority_update.json").relative_to(ROOT).as_posix(),
                                                  "historical_original_rules_exception": True, "live_or_automated_execution": False})
    mandate["historically_terminated_strategy"] = mandate.pop("terminal_strategy_not_revived", MODEL)
    save(mandate_path, mandate)
    print("原85/15策略已按最新授权登记固定复核；原参数、旧终止记录和其他研究约束保留。", flush=True)


def verify_sources(root):
    manifest = read(root / "freeze.json")
    for relative, expected in manifest["workspace_files"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError("原版复现来源已变化：" + relative)
    for relative, expected in manifest["local_files"].items():
        if digest(root / relative) != expected:
            raise RuntimeError("本轮固定复核文件变化：" + relative)
    assert digest(Path(__file__)) == digest(root / "code/selected_mix_reappraisal_v1.py")


def make_pipeline(root):
    from research.strategy_review_full_graph_v1 import DiagnosticPipeline
    from research.adaptive_allocation_v1 import normalize_dividends
    from research.simple_intraday_protection_v1 import make_rules

    class OriginalPipeline(DiagnosticPipeline):
        def __init__(self):
            self.variant = "ORIGINAL_FROZEN_REPRODUCTION"
            self.destination = root / "reproduction"
            self.destination.mkdir(exist_ok=True)
            self.cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
            self.cfg["data_cutoff"] = "2026-09-16"
            self.data = pd.read_parquet(LATEST / "candidate_features.parquet")
            self.data = self.data[self.data.date.le("2026-09-16")].reset_index(drop=True)
            self.div = normalize_dividends(pd.read_csv(ROOT / self.cfg["dividends"]))
            self.capital, self.start, self.next_date = 200000., "2020-01-02", "2026-09-17"
            self.first = int(np.flatnonzero(self.data.date.ge(self.start))[0])
            self.graph = {node["node"]: deepcopy(node) for node in read(GRAPH)["nodes"]}
            self.accounts, self.targets, self.settings, self.calls = {}, {}, {}, {}
            self.checks, self.target_checks = [], []
            self.within = read(SEPTEMBER / "within_models.json")["models"]
            self.ridge = read(SEPTEMBER / "ridge_models.json")["models"]
            self.session = make_rules(self.data)["D60_INTRA"]

    return OriginalPipeline()


def reproduce(root):
    verify_sources(root)
    save(root / "REPRODUCTION_STARTED.json", {"at": now()}, True)
    pipeline = make_pipeline(root).run()
    comparisons = []
    for (model, cost), (ledger, decisions, state) in pipeline.accounts.items():
        expected_path = LATEST / "accounts_run/accounts" / cost / model
        expected = pd.read_parquet(expected_path / "ledger.parquet")
        pd.testing.assert_frame_equal(ledger, expected, check_exact=True, check_dtype=True)
        previous = pd.read_parquet(expected_path / "decisions.parquet")
        common = [c for c in previous if c in decisions]
        pd.testing.assert_frame_equal(decisions[common], previous[common], check_exact=True, check_dtype=True)
        comparisons.append({"model": model, "cost": cost, "ledger_rows": len(ledger), "decision_rows": len(decisions),
                            "all_ledger_columns_exact": True, "common_decision_columns_exact": True,
                            "noncommon_decision_columns": sorted(set(previous.columns).symmetric_difference(decisions.columns)),
                            "max_accounting_error": float(ledger.accounting_error.abs().max())})
    receipt = {"at": now(), "status": "PASS_FROZEN_ORIGINAL_FULL_GRAPH_REPRODUCTION", "accounts": len(comparisons),
               "total_ledger_rows": sum(r["ledger_rows"] for r in comparisons), "results": comparisons,
               "new_model_fits": 0, "new_candidates": 0, "new_independent_observations": 0}
    save(root / "reproduction_receipt.json", receipt, True)
    print(f"原版22条账户、{receipt['total_ledger_rows']:,}行账本已从头复现；保存账户逐列一致。", flush=True)
    return receipt


def metrics(ledger):
    from research.strategy_review_diagnostics_v1 import metrics as original_metrics
    return original_metrics(ledger)


def adjudicate(root):
    verify_sources(root)
    receipt = read(root / "reproduction_receipt.json")
    assert receipt["status"] == "PASS_FROZEN_ORIGINAL_FULL_GRAPH_REPRODUCTION"
    from research.strategy_review_diagnostics_v1 import cycles
    rows, concentration = [], []
    for period in ["main", "earlier"]:
        for cost in ["BASE", "STRESS"]:
            path = root / "reproduction/accounts" / cost / MODEL / "ledger.parquet" if period == "main" else OLD / "earlier_diagnostic" / cost / f"{MODEL}_ledger.parquet"
            ledger = pd.read_parquet(path)
            m = {"period": period, "cost": cost, "start": ledger.date.min(), "end": ledger.date.max(), **metrics(ledger)}
            m["historical_point_targets_met"] = m["sharpe"] >= 1.2 and m["annual_return"] >= .1 and abs(m["max_drawdown"]) <= .1
            m["was_used_for_selection"] = True
            rows.append(m)
            complete, unfinished = cycles(ledger)
            sorted_cycles = complete.sort_values("profit", ascending=False)
            concentration.append({"period": period, "cost": cost, "complete_cycles": len(complete),
                                  "positive_cycles": int(complete.profit.gt(0).sum()),
                                  "largest_profit_share": float(sorted_cycles.profit.iloc[0] / m["profit"]),
                                  "top5_profit_share": float(sorted_cycles.profit.head(5).sum() / m["profit"]),
                                  "largest_cycle": sorted_cycles.iloc[0].to_dict(), "unfinished_cycle": unfinished})
    intervals = pd.read_csv(BOOTSTRAP / "bootstrap_intervals.csv")
    intervals = intervals[intervals.model.eq(MODEL)]
    neighbors = pd.read_csv(NEIGHBOR / "portfolio_metrics.csv")
    forward = pd.read_csv(LATEST / "new_observed_daily_returns.csv")
    origin = read(ROOT / "reports/research/510300_fixed_research_origin_v1/origin.json")
    nonduplicated = forward[forward.cost.eq("STRESS")]
    forward_count = {"origin_recorded_at": origin["recorded_at"], "latest_saved_price_date": "2026-09-16",
                     "post_origin_return_dates": len(nonduplicated),
                     "timely_all_source_decision_dates": int(nonduplicated.all_22_incoming_decisions_saved_before_open.sum()),
                     "holding_dates": int(nonduplicated.shares.gt(0).sum()),
                     "executed_orders": int(nonduplicated.filled_quantity.ne(0).sum()),
                     "complete_trade_cycles": 0, "all_returns_zero": bool(nonduplicated.net_return.eq(0).all()),
                     "independent_trade_edge_established": False,
                     "replay_today_is_new_forward_evidence": False}
    favorable = ["原版四个历史场景均达到数值门槛，原完整账户可逐日复现。",
                 "已有未作完整选择校正的区块区间下界为正，历史优势不只表现为一个高夏普点。",
                 "完整50和70日邻域仍为正收益，不能推断所有信号信息均为零。"]
    adverse = ["308条历史路径筛选及后续权重、调仓设置选择同时使用了两段历史；区块区间未消除这项选择偏差。",
               "主压力最大周期约占总盈利42%，前五约85%；少数机会和成交时点显著影响结果。",
               "完整邻域的收益和夏普明显下降，未稳定达到原门槛。",
               "冻结后现有三天均空仓，只有一天全部依赖决定在开盘前保存，尚无前向交易证据。"]
    result = {"study_id": STUDY, "at": now(), "strategy": MODEL,
              "verdict": "HISTORICAL_EDGE_PLAUSIBLE_SELECTION_INFLATION_UNRESOLVED",
              "not_proven_to_be_pure_overfit": True, "not_proven_free_of_overfitting": True,
              "historical_point_passes": sum(m["historical_point_targets_met"] for m in rows), "historical_metrics": rows,
              "profit_concentration": concentration, "existing_bootstrap_intervals": intervals.to_dict("records"),
              "existing_complete_neighbors": neighbors.to_dict("records"), "forward_evidence": forward_count,
              "favorable_evidence": favorable, "adverse_evidence": adverse,
              "formal_overfitting_probability": "NOT_IDENTIFIABLE_FROM_CURRENT_SELECTION_RECORD",
              "deflated_sharpe_ratio": "NOT_COMPUTED", "probability_of_backtest_overfitting": "NOT_COMPUTED",
              "new_full_graph_reproduction_accounts": 22, "new_model_fits": 0, "new_candidates": 0,
              "new_independent_observations": 0, "independent_performance_established": False,
              "compatible_with_new_two_year_daily_training_contract": False,
              "scope_of_compatibility_exception": "仅用户明确重问旧版后的原貌研究复核，不改变新模型规则",
              "goal_achieved": False, "orders_authorized": False}
    save(root / "result.json", result, True)
    contract = {"at": now(), "status": "IMPLEMENTED_FIXED_REFERENCE_WAITING_FOR_NEW_LOCAL_EVIDENCE",
                "fixed_strategy": MODEL, "configuration_and_code_freeze": "freeze.json",
                "assets": ["510300.SH", "CASH_CNY"], "no_options": True,
                "start": "本轮协议冻结后的首个有完整资料且全部决定可在下一开盘前保存的交易日；不回填缺失日期为及时证据",
                "decision_clock": "原收盘15:05，次一交易日开盘模拟成交",
                "data_requirements": ["官方交易日连续的日线与分红覆盖", "原特征及来源账本完整", "到原月度更新时具备全部已成熟周期和新月模型记录",
                                      "不可变决定保存时间早于执行开盘；每个数据文件有来源、取得时间和哈希"],
                "evaluation": "使用全部连续账户日，包括空仓、失败、未成交和缺失。按预先固定252个交易日阶段复核；不足20个完整周期时保留证据不足。不能首次点值过线即提前结束。",
                "acceptance": "预定阶段压力完整账户夏普>=1.2、复合年化>=10%、最大回撤<=10%，并披露区块不确定性与集中度；样本不足不确认目标",
                "training_clock_conflict": "若未来要求原版也改为两年日更，须视为新版本并独立冻结，不能沿用原版绩效。",
                "live_trading": False, "automation_enabled": False, "market_collection_enabled": False,
                "current_market_view": "NO_VIEW", "current_signal": None,
                "reason": "现有合格本地资料截至2026-09-16，没有本轮冻结后的可评价交易。"}
    save(root / "forward_evidence_contract.json", contract, True)
    print(json.dumps(clean({"结论": result["verdict"], "四场景历史点值通过": result["historical_point_passes"],
                            "前向证据": forward_count, "新独立观察": 0}), ensure_ascii=False, indent=2))
    return result


def status(root):
    output = {"study_id": STUDY, "strategy": MODEL, "protocol_frozen": (root / "freeze.json").exists(),
              "reproduction_complete": (root / "reproduction_receipt.json").exists(),
              "assessment_complete": (root / "result.json").exists(), "current_market_view": "NO_VIEW",
              "current_signal": None, "goal_achieved": False, "orders_authorized": False}
    if output["assessment_complete"]:
        result = read(root / "result.json")
        output.update(verdict=result["verdict"], historical_point_passes=result["historical_point_passes"],
                      forward_evidence=result["forward_evidence"])
    print(json.dumps(output, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="原85/15指数策略固定复核和可重复研究入口")
    parser.add_argument("command", choices=["freeze", "reproduce", "adjudicate", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    {"freeze": freeze, "reproduce": reproduce, "adjudicate": adjudicate, "status": status}[args.command](args.out)


if __name__ == "__main__":
    main()
