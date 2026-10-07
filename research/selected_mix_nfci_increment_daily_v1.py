"""将新取得的NFCI实时版本加入两年日更退出；固定原价格规则和账户预算。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from copy import deepcopy
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
from research.learned_cycle_exit_v1 import FEATURES, state_values
from research.adaptive_allocation_v1 import normalize_dividends
from research.selected_mix_migration_factorial_v1 import account_checks
from research.strategy_review_diagnostics_v1 import metrics

OUT = ROOT / "reports/research/510300_selected_mix_nfci_increment_daily_v1"
SOURCE = ROOT / "reports/research/510300_nfci_graph_vintages_source_v1_3"
STUDY = "510300_SELECTED_MIX_NFCI_INCREMENT_DAILY_V1"
PRIMARY = "NFCI_COMPOSITION"
EXTRA = ["nfci_level", "nfci_change4", "nfci_credit_minus_risk_change4"]
SETS = {"BASELINE": list(FEATURES), "NFCI_AGGREGATE": [*FEATURES, *EXTRA[:2]], PRIMARY: [*FEATURES, *EXTRA]}


def align_vintages(data, versions):
    """每个收盘原点仅合并此前已可用的版本，不使用未来修订或下一次发布日期。"""
    source = versions.copy()
    values = source.pivot(index="vintage_date", columns="series", values="value")
    changes = source.pivot(index="vintage_date", columns="series", values="change4")
    clocks = source[source.series.eq("NFCI")].set_index("vintage_date")
    wide = pd.DataFrame({"nfci_level": values.NFCI, "nfci_change4": changes.NFCI,
                         "nfci_credit_minus_risk_change4": changes.NFCICREDIT - changes.NFCIRISK,
                         "macro_available_at": clocks.available_at, "macro_observation_date": clocks.observation_date})
    wide.index.name = "macro_vintage_date"
    wide = wide.reset_index().sort_values("macro_available_at")
    wide["macro_available_at"] = pd.to_datetime(wide.macro_available_at, utc=True).dt.tz_convert("Asia/Shanghai").astype("datetime64[ns, Asia/Shanghai]")
    wide["macro_observation_date"] = pd.to_datetime(wide.macro_observation_date).astype("datetime64[ns]")
    left = pd.DataFrame({"origin_index": np.arange(len(data)), "date": pd.to_datetime(data.date).astype("datetime64[ns]")})
    left["decision_time"] = left.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    merged = pd.merge_asof(left.sort_values("decision_time"), wide, left_on="decision_time", right_on="macro_available_at", direction="backward")
    known = merged.macro_available_at.notna()
    assert (merged.loc[known, "macro_available_at"] <= merged.loc[known, "decision_time"]).all()
    merged["macro_age_days"] = (merged.date - merged.macro_observation_date).dt.days
    merged["macro_known"] = known & merged[EXTRA].notna().all(axis=1) & merged.macro_age_days.between(0, 21)
    return merged.sort_values("origin_index").reset_index(drop=True)


def estimate(model, values):
    values = np.asarray(values, float)
    if len(values) != len(model["features"]) or not np.isfinite(values).all():
        raise ValueError("模型的完整特征不可用")
    z = np.clip((values - np.asarray(model["mean"])) / np.asarray(model["scale"]), -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + z @ np.asarray(model["coefficients"]))


class NFCIExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        assert self.indexes == sorted(set(self.indexes))
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        base = state_values(self.data, t, cycle, current_value, peak_value)
        k = bisect_right(self.indexes, t) - 1
        record = self.models[k] if k >= 0 else None
        value, status, names, values = None, "NO_VIEW_NO_MATURE_MODEL", list(FEATURES), base
        if record and record["status"] == "FIT_COMPLETE":
            assert record["latest_exit_index"] < record["fit_index"] == t
            names = record["model"]["features"]
            assert names[:len(FEATURES)] == FEATURES
            values = np.r_[base, [self.data.iloc[t][name] for name in names[len(FEATURES):]]]
            if np.isfinite(values).all():
                value, status = estimate(record["model"], values), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_FEATURES"
        self.negative_count = self.negative_count + 1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status,
                "continuation_prediction": value, "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else pd.NaT,
                "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days, **dict(zip(names, values))}


def implementation_checks():
    # 收盘原点的边界：15:05之后发布的下一版不能进入当日特征。
    rows = []
    for vintage, available, value in [("2020-03-18", "2020-03-19T13:00:00+08:00", 1.),
                                       ("2020-03-25", "2020-03-26T15:06:00+08:00", 99.)]:
        for series in ["NFCI", "NFCICREDIT", "NFCIRISK"]:
            rows.append({"vintage_date": pd.Timestamp(vintage), "available_at": pd.Timestamp(available).tz_convert("Asia/Shanghai"),
                         "observation_date": pd.Timestamp(vintage) - pd.Timedelta(days=5), "series": series, "value": value, "change4": value / 2})
    versions = pd.DataFrame(rows)
    data = pd.DataFrame({"date": pd.to_datetime(["2020-03-19", "2020-03-26", "2020-03-27"])})
    aligned = align_vintages(data, versions)
    assert aligned.nfci_level.to_list() == [1., 1., 99.]
    changed = versions.copy()
    changed.loc[changed.vintage_date.eq("2020-03-25"), "value"] = 9999.
    np.testing.assert_array_equal(align_vintages(data, changed).nfci_level.iloc[:2], aligned.nfci_level.iloc[:2])
    model = {"features": [*FEATURES, EXTRA[0]], "mean": [0.] * 9, "scale": [1.] * 9,
             "coefficients": [0.] * 8 + [.5], "intercept": -.2, "feature_clip": 5.}
    assert abs(estimate(model, [0.] * 8 + [2.]) - .8) < 1e-12
    return {"availability_boundary": True, "later_vintage_perturbation_does_not_change_past": True,
            "additional_feature_enters_prediction": True}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("实验已冻结，不覆盖固定协议。")
    result = read(SOURCE / "result.json")
    assert result["status"] == "COMPLETE_OFFICIAL_ARCHIVAL_VINTAGE_FEATURES"
    snapshot = ROOT / result["snapshot"]
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    versions = pd.read_parquet(snapshot / "vintage_features.parquet")
    aligned = align_vintages(data, versions)
    baseline = read(pooled.OUT / "results/daily_models.json")
    first = baseline["ridge"][0]["fit_index"]
    unavailable_origins = aligned.loc[first:][lambda x: ~x.macro_known]
    samples = pd.read_parquet(pooled.OUT / "training_reference/pooled_samples.parquet")
    assert aligned.macro_known.iloc[samples.origin_index.to_numpy(int)].all()
    for name in ["inputs", "results", "code", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(root / "inputs/point_in_time_macro.parquet", index=False)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    save(root / "inputs/source_result.json", result, True)
    protocol = {
        "study_id": STUDY, "at": now(), "primary": PRIMARY,
        "question": "在相同价格持仓状态及总体金融条件下，信用相对风险的变化能否改善继续持有与退出的判断。",
        "economic_hypothesis": "市场风险急升而信用条件较稳定，可能对应暂时风险厌恶；信用也持续收紧可能对应更持久压力。美国指标对510300的跨市场含义仍待检验，不等同国内信贷。",
        "new_evidence": result, "features": SETS,
        "new_feature_definitions": {"nfci_level": "当时已可见NFCI版本的最新周值",
            "nfci_change4": "该版本最新周值减同一版本严格四周前值",
            "nfci_credit_minus_risk_change4": "信用分项的同版四周变化减风险分项的同版四周变化；两者均为标准化指数，不能解释为货币金额"},
        "timing": "原点T收盘15:05；版本日芝加哥午夜加一自然日后可用，T+1开盘执行；逐原点保留版本日、观察日和可用时刻。",
        "new_design": "总指标对照只加两项；预指定主方案在其上只加一项信用与风险差异。",
        "baseline": "原两年每日共享退出+风险目标先行+固定10个百分点调仓带；基础、压力账户逐列复现。",
        "training": read(pooled.OUT / "protocol.json")["training"],
        "training_change": "只增加明确特征，不改成熟周期、样本权重、岭罚1、裁剪5、两年窗口和日更频率；新特征覆盖所有原有样本，否则不运行。",
        "missing_macro_policy": "最新观察超过21自然日时，新增宏观信息为NO_VIEW，该决策原点使用同一天原价格模型；不删除交易日、不放宽新鲜度、不补造数据。所有原训练样本仍须具备当时有效宏观信息。",
        "macro_unavailable_origins": unavailable_origins.to_dict("records"),
        "prefreeze_input_correction": "首次输入检查发现三个原点的官方档案过期，直接档案补查仍为旧值；在新策略拟合及账户收益计算前明确原价格模型回退。原失败记录保留。",
        "preserved": ["价格入场与价格失效边界", "两日负预测退出确认", "三类训练来源及依赖图", "每日风险分配及账户尾部预算", "账户费用、T+1、整手、现金与分红"],
        "period": [daily.START, daily.END], "capital": 200000, "annual_days": 242,
        "target": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "costs": read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")["costs"],
        "tail_contract": read(ROOT / "config/510300_existing_data_training_mandate_v1.json")["account_tail_risk_contract"],
        "statistics": "所有共同日收益差20日区块2000次，种子20260928；所有年份及滚动两年，不挑日期。区间未校正整个项目历史选择。",
        "continuation_gate": "预指定主方案压力成本相对总指标和原对照的完整账户算术收益差95%区间下界均>0，且至少两个完整年增量为正；达标仍需后续独立验证。",
        "planned_new_final_accounts": 4, "baseline_reproduction_accounts": 2,
        "new_parameter_search": 0, "goal_achieved": False,
        "evidence_class": "新取得的历史宏观版本配合已研究过的价格样本，只属于开发增量检验。",
        "checks": implementation_checks(), "orders_authorized": False,
    }
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update([LATEST / "candidate_features.parquet", ROOT / "data/reference/510300_dividends.csv",
                    ROOT / "config/510300_incremental_selected_intent_mix_v1.json", pooled.OUT / "results/daily_models.json",
                    pooled.OUT / "training_reference/pooled_samples.parquet", daily.OUT / "results/tail_forecasts.json",
                    snapshot / "vintage_features.parquet", SOURCE / "result.json"])
    for cost in ["BASE", "STRESS"]:
        sources.add(execution.OUT / "accounts" / cost / execution.PRIMARY / "ledger.parquet")
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(root / "protocol.json"),
        "aligned_input_sha256": digest(root / "inputs/point_in_time_macro.parquet"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    print("NFCI增量实验已固定：同一训练样本和账户约束，总体条件与一项信用/风险差异分开比较。", flush=True)


def train(root, data, macro, original_models):
    samples = pd.read_parquet(pooled.OUT / "training_reference/pooled_samples.parquet")
    for col in EXTRA:
        samples[col] = macro[col].iloc[samples.origin_index.to_numpy(int)].to_numpy()
    trained = {}
    verification = {"new_fits": 0, "daily_origins": 0, "baseline_coefficient_checks": 0, "sample_membership_checks": 0,
                    "macro_unavailable_fallback_origins": {}}
    for policy in ["NFCI_AGGREGATE", PRIMARY]:
        models = {kind: [] for kind in ["ridge", "within"]}
        verification["macro_unavailable_fallback_origins"][policy] = []
        fitter = FunctionType(pooled.pooled_fit.__code__, {**pooled.pooled_fit.__globals__, "FEATURES": SETS[policy]}, "新增宏观特征的固定岭拟合")
        for k, old in enumerate(original_models["ridge"]):
            t = old["fit_index"]
            rows, ids, left, per_signal, eligible = pooled.choose(data, samples, t)
            assert rows.sample_id.to_list() == old["sample_ids"] and ids == old["training_cycles"]
            assert eligible == (old["status"] == "FIT_COMPLETE")
            assert np.isfinite(rows[SETS[policy]].to_numpy(float)).all()
            assert rows.exit_index.lt(t).all() and rows.entry_date.ge(left).all()
            verification["sample_membership_checks"] += 1
            macro_known = bool(macro.macro_known.iloc[t])
            if not macro_known:
                verification["macro_unavailable_fallback_origins"][policy].append(int(t))
            if eligible and k % 500 == 0:
                reproduced = pooled.pooled_fit(rows, "ridge")
                np.testing.assert_allclose(reproduced["coefficients"], old["model"]["coefficients"], atol=1e-12, rtol=0)
                verification["baseline_coefficient_checks"] += 1
            for kind in models:
                record = deepcopy(original_models[kind][k])
                if eligible and macro_known:
                    record["model"] = fitter(rows, kind)
                elif not eligible:
                    record["model"] = None
                else:
                    assert record["model"] == original_models[kind][k]["model"]
                record["macro_status"] = "AVAILABLE" if macro_known else "NO_VIEW_STALE_BASELINE_FALLBACK"
                record["macro_vintage_date"] = macro.macro_vintage_date.iloc[t]
                record["macro_available_at"] = macro.macro_available_at.iloc[t]
                record["macro_age_days"] = macro.macro_age_days.iloc[t]
                models[kind].append(record)
                verification["new_fits"] += int(eligible and macro_known)
            if (k + 1) % 500 == 0:
                print(f"{policy} 两年逐日更新 {k+1}/{len(original_models['ridge'])}。", flush=True)
        save(root / "results" / f"{policy}_models.json", models, True)
        trained[policy] = models
        verification["daily_origins"] += len(models["ridge"])
    save(root / "results/training_checks.json", verification, True)
    return trained, verification


def graph(root, policy, models, macro):
    destination = root / "internal_graphs" / policy
    destination.mkdir(parents=True)
    pipeline = make_pipeline(destination)
    pipeline.variant = "NFCI_INCREMENT_" + policy
    pd.testing.assert_series_equal(pipeline.data.date.reset_index(drop=True), macro.date.reset_index(drop=True), check_names=False, check_dtype=False)
    for col in EXTRA:
        pipeline.data[col] = macro[col].to_numpy()
    pipeline.ridge, pipeline.within = models["ridge"], models["within"]
    bindings = {**original.Pipeline.run.__globals__, "EntryVintageExitController": NFCIExitController,
                "ExitController": NFCIExitController, "continuous_minimum_variance_budget": daily.daily_min_variance,
                "joint_downside_budgets": daily.daily_joint_downside, "support_choice": daily.daily_support}
    FunctionType(original.Pipeline.run.__code__, bindings, "宏观版本增量依赖图")(pipeline)
    return pipeline


def run(root):
    frozen = read(root / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(root / "protocol.json") == frozen["protocol_sha256"]
    assert digest(root / "inputs/point_in_time_macro.parquet") == frozen["aligned_input_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, "冻结来源变化：" + path
    save(root / "RUN_STARTED.json", {"at": now()}, True)
    macro = pd.read_parquet(root / "inputs/point_in_time_macro.parquet")
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    original_models = read(pooled.OUT / "results/daily_models.json")
    models, checks = train(root, data, macro, original_models)
    models["BASELINE"] = original_models
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    tails = read(daily.OUT / "results/tail_forecasts.json")
    accounts = {}
    for policy in SETS:
        pipeline = graph(root, policy, models[policy], macro)
        simulator = FunctionType(execution.simulate.__code__, {**execution.simulate.__globals__, "PRIMARY": policy}, "宏观增量末端账户")
        for cost in cfg["costs"]:
            ledger, decisions, checkpoint = simulator(pipeline.data, dividends, cfg, pipeline.get(MODEL, cost), tails, cost)
            account_checks(ledger, decisions)
            if policy == "BASELINE":
                old = pd.read_parquet(execution.OUT / "accounts" / cost / execution.PRIMARY / "ledger.parquet")
                pd.testing.assert_frame_equal(ledger, old, check_exact=False, atol=1e-8, rtol=0)
            folder = root / "accounts" / cost / policy
            folder.mkdir(parents=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            save(folder / "checkpoint.json", checkpoint, True)
            accounts[policy, cost] = ledger
            m = metrics(ledger)
            print(f"{policy}／{cost}：夏普{m['sharpe']:.6f}，年化{m['annual_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    measurements, windows = daily.summarize_accounts(root, data, accounts)
    comparisons = []
    rng = np.random.default_rng(20260928)
    for cost in cfg["costs"]:
        for left, right in [("NFCI_AGGREGATE", "BASELINE"), (PRIMARY, "NFCI_AGGREGATE"), (PRIMARY, "BASELINE")]:
            a, b = accounts[left, cost], accounts[right, cost]
            positive_years = []
            for year in sorted(set(a.date.dt.year)):
                mask = a.date.dt.year.eq(year)
                if year < 2026 and float((a.loc[mask, "net_return"].to_numpy() - b.loc[mask, "net_return"].to_numpy()).sum()) > 0:
                    positive_years.append(int(year))
            comparisons.append({"cost": cost, "left": left, "right": right, "positive_complete_years": positive_years,
                                **daily.paired_interval(a, b, rng)})
    primary = next(r for r in measurements if r["policy"] == PRIMARY and r["cost"] == "STRESS")
    gate_rows = [r for r in comparisons if r["cost"] == "STRESS" and r["left"] == PRIMARY]
    gate = all(r["lower_95"] > 0 and len(r["positive_complete_years"]) >= 2 for r in gate_rows)
    result = {"study_id": STUDY, "at": now(), "primary": primary, "all_accounts": measurements, "comparisons": comparisons,
              "continuation_gate": gate, "status": "DEVELOPMENT_INCREMENT_SUPPORTED" if gate else "FROZEN_NO_RELIABLE_NFCI_INCREMENT",
              "training_checks": checks, "new_final_accounts": 4, "baseline_accounts_reproduced": 2,
              "internal_dependency_accounts": 66, "new_reference_accounts": 0,
              "rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in windows if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
              "new_independent_market_observations": 0, "goal_achieved": False,
              "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(root / "result.json", result, True)
    print("新NFCI历史版本的固定增量比较完成，全部对照和失败结果保留。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFCI历史版本的两年日更策略增量。")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    args = parser.parse_args()
    if args.command == "status":
        print(read(OUT / "result.json") if (OUT / "result.json").exists() else "实验尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](OUT)
