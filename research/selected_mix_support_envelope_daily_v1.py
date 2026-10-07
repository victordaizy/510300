"""两年日更学习退出的单项支持范围：越界时继续执行原价格失效边界。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from pathlib import Path
import shutil
import sys
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.selected_mix_daily_two_year_v1 as daily
import research.selected_mix_pooled_daily_v1 as pooled
import research.selected_mix_risk_before_band_v1 as execution
import research.post_selection_continuous_replay_v1 as original
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, make_pipeline, LATEST, MODEL, now
from research.learned_cycle_exit_v1 import FEATURES, state_values, predict
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction
from research.adaptive_allocation_v1 import normalize_dividends
from research.selected_mix_migration_factorial_v1 import account_checks
from research.strategy_review_diagnostics_v1 import metrics

OUT = ROOT / "reports/research/510300_selected_mix_support_envelope_daily_v1"
STUDY = "510300_SELECTED_MIX_SUPPORT_ENVELOPE_DAILY_V1"
PRIMARY = "D60_SUPPORT_ENVELOPE_RISK_BAND10"
TOLERANCE = 1e-12


def supported(values, envelope):
    lower, upper = np.asarray(envelope["lower"]), np.asarray(envelope["upper"])
    outside = (values < lower - TOLERANCE) | (values > upper + TOLERANCE)
    return not bool(outside.any()), [name for name, rejected in zip(FEATURES, outside) if rejected]


class SupportEnvelopeExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [record["fit_index"] for record in models]
        assert self.fit_indexes == sorted(set(self.fit_indexes))
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t) - 1
        record = self.models[index] if index >= 0 else None
        estimate, unrestricted, inside, outside = None, None, None, []
        status = "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            assert record["latest_exit_index"] < record["fit_index"] == t
            if np.isfinite(values).all():
                model = record["model"]
                unrestricted = within_cycle_prediction(model, values) if model["kind"] == "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE" else predict(model, values)
                inside, outside = supported(values, record["marginal_support"])
                estimate = unrestricted if inside else None
                status = "PREDICTION_AVAILABLE" if inside else "NO_VIEW_OUTSIDE_D60_MARGINAL_SUPPORT"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        self.negative_count = self.negative_count + 1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status,
                "continuation_prediction": estimate, "unrestricted_continuation_prediction": unrestricted,
                "inside_d60_marginal_support": inside, "outside_support_features": "；".join(outside),
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else pd.NaT,
                "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **dict(zip(FEATURES, values))}


def implementation_checks():
    envelope = {"lower": [-1.] * 8, "upper": [1.] * 8}
    assert supported(np.zeros(8), envelope) == (True, [])
    assert supported(np.ones(8), envelope) == (True, [])
    values = np.zeros(8)
    values[2] = -1.01
    assert supported(values, envelope) == (False, [FEATURES[2]])
    data = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=4), "mom5": [0., 0., 2., 0.],
                         "mom20": 0., "sma120": 0., "vol20": 0.})
    models = [{"fit_index": t, "latest_exit_index": t - 1, "status": "FIT_COMPLETE",
               "marginal_support": {"lower": [-10.] * 8, "upper": [1.8] * 8},
               "model": {"kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8,
                         "coefficients": [0.] * 8, "intercept": -.01, "feature_clip": 5.}} for t in range(4)]
    controller = SupportEnvelopeExitController(data, models)
    cycle = {"cycle_id": 1, "entry_index": 0, "entry_cost_cny": 100., "mode": 1}
    states = [controller(t, cycle, 100., 100.) for t in range(4)]
    assert [r["negative_confirmation_count"] for r in states] == [1, 2, 0, 1]
    assert states[2]["learning_status"] == "NO_VIEW_OUTSIDE_D60_MARGINAL_SUPPORT"
    assert states[2]["continuation_prediction"] is None
    assert states[2]["unrestricted_continuation_prediction"] == -.01
    return {"inclusive_marginal_boundaries": True, "outside_feature_identified": True,
            "two_negative_confirmation_and_outside_reset": True, "raw_prediction_retained": True}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("单项支持范围实验已经固定。")
    pooled.verify_sources(pooled.OUT)
    execution.verify_sources(execution.OUT)
    for name in ["code", "inputs", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {"study_id": STUDY, "at": now(), "primary": PRIMARY,
        "question": "两年日更的学习退出在当前八项状态超出D60成熟训练范围时暂不采用，是否改善完整账户。",
        "motivation": "父共享模型压力D60学习账户2020年起292个可预测日中，63日至少一项超出同窗D60各项范围；其中8日预测为负。这是已看历史的结构诊断，未据后续盈亏选特征。",
        "fixed_change": "使用每个当日模型已经登记的D60训练成员，保存原八项特征各自最小值和最大值。实际状态任一项超界则该日学习预测NO_VIEW并清零学习退出连续计数。所有项范围内才沿用原预测与连续两个负值确认。只用1e-12浮点容差，无分位数或距离阈值。",
        "interpretation": "各项范围内并不证明联合状态有充分支持，也不是概率置信区间；此规则只限制明显的单项外推。范围外仍正常执行原价格、止损和最长持有规则。",
        "training": "复用三类共享训练的全部已保存系数，不新增系数拟合。范围仅取当日已登记D60成熟成员，完整周期入场和观察在最近两日历年，退出索引严格小于当日。每天重新检查，不携带过期范围。",
        "unchanged": ["三类训练资料及八项特征、类别截距和共享斜率", "原价格入场、再入场和失效边界",
                      "两年每日参考协方差与下行预算", "当前末端尾部预算及风险投影后的10个百分点交易带",
                      "20万元、100份、T+1、佣金、滑点、分红及连续收盘结算"],
        "duplicate_review": {"deleted_cycle_round72": "逐次删周期要求一致负值，已完成；本轮不重做。",
                             "bounded_five_day_value": "未结束周期的五日成熟标签已经检验；本轮沿用原自然周期标签。",
                             "model_support_router": "旧路由只检查模型是否满足周期/行数，不检查当前状态值。",
                             "cycle_analogue_round122": "旧近邻用五个不同周期直接重估收益；本轮不改变预测。"},
        "period": [daily.START, daily.END], "capital": 200000, "annual_days": 242,
        "target": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "tail_contract": mandate["account_tail_risk_contract"], "primary_comparator": execution.PRIMARY,
        "new_full_accounts": 2, "new_internal_dependency_accounts": 22, "new_reference_accounts": 0,
        "new_coefficient_fits": 0, "new_parameter_grid": 0,
        "statistics": "完整同日收益差20日区块2000次，种子20261003；报告所有年份和全部滚动两年。",
        "checks": implementation_checks(), "goal_achieved": False, "orders_authorized": False,
        "new_market_collection": False, "evidence_class": "已研究历史的新固定开发假设，无独立新增资料"}
    save(root / "protocol.json", protocol, True)
    files = local_import_closure({Path(__file__)})
    files.update(ROOT / name for name in read(pooled.OUT / "freeze.json")["sources"])
    files.update([pooled.OUT / "results/daily_models.json", pooled.OUT / "training_reference/pooled_samples.parquet",
                  daily.OUT / "results/tail_forecasts.json", execution.OUT / "result.json"])
    for cost in ["BASE", "STRESS"]:
        files.add(execution.OUT / "accounts" / cost / execution.PRIMARY / "ledger.parquet")
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)}}, True)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("单项外推退出限制已固定，复用所有日更系数，原价格失效边界继续执行。", flush=True)


def verify_sources(root):
    record = read(root / "freeze.json")
    assert digest(Path(__file__)) == record["code_sha256"]
    assert digest(root / "protocol.json") == record["protocol_sha256"]
    for name, expected in record["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def attach_support(root):
    models = read(pooled.OUT / "results/daily_models.json")
    samples = pd.read_parquet(pooled.OUT / "training_reference/pooled_samples.parquet").set_index("sample_id", drop=False)
    envelopes = {}
    for record in models["ridge"]:
        if record["status"] != "FIT_COMPLETE":
            continue
        rows = samples.loc[record["sample_ids"]]
        rows = rows.loc[rows.signal.eq("D60_INTRA")]
        assert len(rows) and rows.exit_index.lt(record["fit_index"]).all()
        assert rows.entry_date.ge(pd.Timestamp(record["window_start"])).all()
        assert rows.origin.ge(pd.Timestamp(record["window_start"])).all()
        raw = rows[FEATURES].to_numpy(float)
        assert np.isfinite(raw).all()
        envelopes[record["fit_index"]] = {"fit_index": record["fit_index"], "fit_origin": record["fit_origin"],
            "window_start": record["window_start"], "training_rows": len(rows), "d60_cycles": int(rows.cycle_id.nunique()),
            "sample_ids": rows.sample_id.to_list(), "latest_exit_index": int(rows.exit_index.max()),
            "lower": raw.min(axis=0).tolist(), "upper": raw.max(axis=0).tolist()}
    original_ridge = {record["fit_index"]: record for record in models["ridge"]}
    for kind in models:
        for record in models[kind]:
            assert record["sample_ids"] == original_ridge[record["fit_index"]]["sample_ids"]
            if record["status"] == "FIT_COMPLETE":
                record["marginal_support"] = envelopes[record["fit_index"]]
    save(root / "results/daily_support_envelopes.json", list(envelopes.values()), True)
    print(f"已保存{len(envelopes)}个两年日更支持范围，无新增回归拟合。", flush=True)
    return models


def graph(root, models):
    destination = root / "internal_graphs/SUPPORT_ENVELOPE"
    destination.mkdir(parents=True)
    pipeline = make_pipeline(destination)
    pipeline.variant = "DAILY_D60_SUPPORT_ENVELOPE"
    pipeline.ridge, pipeline.within = models["ridge"], models["within"]
    bindings = dict(original.Pipeline.run.__globals__)
    bindings.update(EntryVintageExitController=SupportEnvelopeExitController, ExitController=SupportEnvelopeExitController,
                    continuous_minimum_variance_budget=daily.daily_min_variance,
                    joint_downside_budgets=daily.daily_joint_downside, support_choice=daily.daily_support)
    FunctionType(original.Pipeline.run.__code__, bindings, "单项范围支持的日更依赖图")(pipeline)
    save(destination / "bindings.json", {"changed_exit_controller": "SupportEnvelopeExitController",
         "models_reused_from": pooled.OUT.relative_to(ROOT).as_posix(), "internal_accounts": len(pipeline.accounts),
         "daily_reference_risk_unchanged": True, "original_files_changed": False}, True)
    return pipeline


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    models = attach_support(root)
    pipeline = graph(root, models)
    data = pipeline.data
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    tails = read(daily.OUT / "results/tail_forecasts.json")
    simulator = FunctionType(execution.simulate.__code__, {**execution.simulate.__globals__, "PRIMARY": PRIMARY}, "支持范围末端账户")
    accounts, comparisons = {}, []
    rng = np.random.default_rng(20261003)
    for cost in cfg["costs"]:
        ledger, decisions, checkpoint = simulator(data, dividends, cfg, pipeline.get(MODEL, cost), tails, cost)
        folder = root / "accounts" / cost / PRIMARY
        folder.mkdir(parents=True)
        ledger.to_parquet(folder / "ledger.parquet", index=False)
        decisions.to_parquet(folder / "decisions.parquet", index=False)
        save(folder / "checkpoint.json", checkpoint, True)
        account_checks(ledger, decisions)
        accounts[PRIMARY, cost] = ledger
        baseline = pd.read_parquet(execution.OUT / "accounts" / cost / execution.PRIMARY / "ledger.parquet")
        pd.testing.assert_series_equal(ledger.date, baseline.date)
        comparisons.append({"cost": cost, "left": PRIMARY, "right": execution.PRIMARY,
                            **daily.paired_interval(ledger, baseline, rng)})
        measurement = metrics(ledger)
        print(f"支持范围 {cost}：夏普{measurement['sharpe']:.6f}，年化{measurement['annual_return']:.2%}，回撤{measurement['max_drawdown']:.2%}。", flush=True)
    measurements, windows = daily.summarize_accounts(root, data, accounts)
    diagnostics = []
    for (node, cost), (_, decisions, _) in pipeline.accounts.items():
        if "inside_d60_marginal_support" not in decisions:
            continue
        actual = decisions.loc[decisions.origin.ge(daily.START)]
        outside = actual.learning_status.eq("NO_VIEW_OUTSIDE_D60_MARGINAL_SUPPORT")
        assert not actual.loc[outside, "learned_exit_requested"].any()
        assert actual.loc[outside, "negative_confirmation_count"].eq(0).all()
        assert actual.loc[outside, "continuation_prediction"].isna().all()
        diagnostics.append({"node": node, "cost": cost,
            "known_model_decisions": int(actual.unrestricted_continuation_prediction.notna().sum()),
            "outside_support": int(outside.sum()),
            "negative_raw_prediction_outside_support": int((outside & actual.unrestricted_continuation_prediction.lt(0)).sum()),
            "learning_exit_requests": int(actual.learned_exit_requested.fillna(False).sum())})
    save(root / "results/exit_support_diagnostics.json", diagnostics, True)
    save(root / "results/paired_comparisons.json", comparisons, True)
    verification = {"at": now(), "status": "PASS_SAVED_MODELS_CLOCK_SUPPORT_AND_ACCOUNT_IDENTITIES",
        "coefficient_fits": 0, "daily_support_envelopes": len(read(root / "results/daily_support_envelopes.json")),
        "controller_accounts_checked": len(diagnostics), "full_accounts_checked": 2,
        "outside_support_resets_learning_only": True}
    save(root / "verification.json", verification, True)
    primary = next(row for row in measurements if row["cost"] == "STRESS")
    save(root / "result.json", {"study_id": STUDY, "at": now(), "primary": primary, "all_accounts": measurements,
        "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_SUPPORT_ENVELOPE",
        "comparisons": comparisons, "diagnostics": diagnostics, "verification": verification,
        "new_full_accounts": 2, "new_internal_accounts": 22, "new_reference_accounts": 0, "new_coefficient_fits": 0,
        "rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in windows if row["cost"] == "STRESS"),
        "goal_achieved": False, "new_independent_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}, True)
    print("学习退出支持范围实验完成，原模型及父账户保持原记录。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="两年日更模型的单项外推退出限制")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
