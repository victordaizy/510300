"""三类固定价格过程共享退出斜率；两年每日训练、相同尾部预算。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.selected_mix_daily_two_year_v1 as parent
from research.selected_mix_reappraisal_v1 import digest, read, save, clean, local_import_closure, LATEST, MODEL
from research.september_monthly_training_v1 import reference_samples
from research.simple_intraday_protection_v1 import make_rules
from research.adaptive_allocation_v1 import normalize_dividends
from research.learned_cycle_exit_v1 import FEATURES
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_selected_mix_pooled_daily_v1"
PARENT = parent.OUT
STUDY = "510300_SELECTED_MIX_POOLED_DAILY_V1"
PRIMARY = "POOLED_DAILY_TWO_YEAR_TAIL"
SIGNALS = ["D60_INTRA", "S1_TREND_REBOUND", "R2_Z_CONFIRM"]


def freeze(root):
    for name in ["code", "inputs", "results", "accounts", "training_reference"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    if (root / "freeze.json").exists():
        raise RuntimeError("共享训练实验已冻结，请读取或执行原协议。")
    import shutil
    shutil.copy2(__file__, root / "code/selected_mix_pooled_daily_v1.py")
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    assert mandate["executable_assets"] == ["510300.SH", "CASH_CNY"]
    assert read(PARENT / "result.json")["status"] == "FROZEN_NO_QUALIFIED_TWO_YEAR_DAILY_MIGRATION"
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {
        "study_id": STUDY, "at": parent.now(), "primary": PRIMARY,
        "question": "仅D60成熟周期稀少时，三类原价格过程共享斜率且保留类别差异，能否改善两年每日退出。",
        "change_from_parent": "只更换学习退出的训练来源与估计结构；保持父实验价格信号、每日风险预算、最终仓位约束、费用和评价起止。",
        "reference_signals": SIGNALS, "new_reference_reconstructions": 2, "reused_D60_reference": True,
        "training": {"window": "最近两个日历年，完整周期入场>=窗口起点且退出索引<决策索引。",
                     "minimum_total_cycles": 10, "minimum_cycles_each_signal": 3, "minimum_rows": 100,
                     "state_features": FEATURES, "ridge_alpha": 1., "feature_clip": 5.,
                     "weights": "三类过程各占相同总权重，每类内每周期等权、周期内每状态等权；总权重仍为周期数。",
                     "ridge_model": "共同八项斜率，三类过程各有未惩罚截距；实际D60预测采用D60截距。",
                     "within_model": "共同八项周期内斜率，每周期有未惩罚截距；实际预测只取成熟D60周期截距等权均值。",
                     "refresh": "每交易日重新检查并拟合；资格失败不延续过期系数，按原价格退出。",
                     "sample_dependence": "三类过程可重叠且共享510300价格；周期数量不解释为独立样本，单独披露重叠组。"},
        "primary_comparators": ["父实验DAILY_TWO_YEAR_MIN5_TAIL", "父实验ORIGINAL_TAIL"],
        "target": {"net_sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "tail_contract": read(PARENT / "protocol.json")["tail"],
        "planned_new_accounts": 2, "planned_internal_graph_accounts": 22, "new_parameter_grid": 0,
        "statistics": "复用父实验20日区块、2000次，固定种子20260926，比较同日完整账户。",
        "evidence_class": "此前查看过历史后的新固定开发假设，不是独立验证。",
        "new_market_collection": False, "orders_authorized": False, "goal_achieved": False}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update(ROOT / name for name in read(PARENT / "freeze.json")["sources"])
    sources.update([PARENT / "protocol.json", PARENT / "result.json", PARENT / "results/tail_forecasts.json",
                    PARENT / "training_reference/samples.parquet"])
    for cost in ["BASE", "STRESS"]:
        for model in [parent.PRIMARY, "ORIGINAL_TAIL"]:
            sources.add(PARENT / "accounts" / cost / model / "ledger.parquet")
    save(root / "freeze.json", {"at": parent.now(), "code_sha256": digest(Path(__file__)),
         "protocol_sha256": digest(root / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("三类过程共享训练已固定：共同斜率、类别差异、两年每日时钟，相同末端风险预算。", flush=True)


def verify_sources(root):
    frozen = read(root / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(root / "protocol.json") == frozen["protocol_sha256"]
    for name, expected in frozen["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def make_samples(root, data, dividends):
    cfg = read(ROOT / "config/510300_learned_cycle_exit_v1.json")
    rules = make_rules(data)
    combined = []
    for code, signal in enumerate(SIGNALS):
        if signal == "D60_INTRA":
            samples = pd.read_parquet(PARENT / "training_reference/samples.parquet").copy()
        else:
            adapted = deepcopy(cfg)
            adapted["candidate_specs"]["D60_INTRA"] = cfg["candidate_specs"][signal]
            _, ledger, decisions, completed, checkpoint, samples = reference_samples(
                data, dividends, adapted, rules[signal], parent.END, parent.NEXT)
            samples = samples.merge(completed[["cycle_id", "entry_index"]], on="cycle_id", how="left", validate="many_to_one")
            samples["entry_date"] = pd.to_datetime(data.date.iloc[samples.entry_index.to_numpy(int)].to_numpy())
            folder = root / "training_reference" / signal
            folder.mkdir()
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            completed.to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "checkpoint.json", checkpoint, True)
        samples["source_cycle_id"] = samples.cycle_id
        samples["cycle_id"] = code * 100000 + samples.cycle_id.astype(int)
        samples["signal"] = signal
        combined.append(samples)
        print(f"共享参考 {signal}：{samples.cycle_id.nunique()}个有标签周期、{len(samples)}行状态。", flush=True)
    merged = pd.concat(combined, ignore_index=True)
    merged["sample_id"] = np.arange(len(merged), dtype=int)
    merged.to_parquet(root / "training_reference/pooled_samples.parquet", index=False)
    return merged


def choose(data, samples, t):
    rows, ids, left = parent.training_at(data, samples, t)
    counts = rows.groupby("signal").cycle_id.nunique().reindex(SIGNALS, fill_value=0).astype(int)
    total = len(ids)
    if len(rows):
        rows["sample_weight"] *= rows.signal.map({signal: total / (3 * n) for signal, n in counts.items() if n})
    eligible = total >= 10 and counts.ge(3).all() and len(rows) >= 100
    return rows, ids, left, counts, bool(eligible)


def pooled_fit(rows, kind):
    x = rows[FEATURES].to_numpy(float)
    y, w = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    assert np.isfinite(x).all() and np.isfinite(y).all() and (w > 0).all()
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x - mean) / scale, -5., 5.)
    group_values = rows.signal.to_numpy() if kind == "ridge" else rows.cycle_id.to_numpy()
    dx, dy = np.empty_like(z), np.empty_like(y)
    groups = []
    for group in sorted(set(group_values)):
        mask = group_values == group
        mz = np.average(z[mask], axis=0, weights=w[mask])
        my = float(np.average(y[mask], weights=w[mask]))
        dx[mask], dy[mask] = z[mask] - mz, y[mask] - my
        groups.append({"group": str(group), "signal": str(rows.signal.iloc[np.flatnonzero(mask)[0]]),
                       "rows": int(mask.sum()), "weight": float(w[mask].sum()),
                       "standardized_feature_mean": mz.tolist(), "target_mean": my})
    model = Ridge(alpha=1., fit_intercept=False, solver="svd")
    model.fit(dx, dy, sample_weight=w)
    for group in groups:
        group["intercept"] = float(group["target_mean"] - np.asarray(group["standardized_feature_mean"]) @ model.coef_)
    local = [group["intercept"] for group in groups if group["signal"] == "D60_INTRA"]
    assert len(local)
    return {"kind": "RIDGE" if kind == "ridge" else "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE",
            "estimator": "POOLED_SIGNAL_INTERCEPT_RIDGE" if kind == "ridge" else "POOLED_WITHIN_CYCLE_SLOPES_D60_INTERCEPT",
            "features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": model.coef_.tolist(), "intercept": float(np.mean(local)),
            "feature_clip": 5., "training_groups": groups, "prediction_signal": "D60_INTRA"}


def overlap_groups(rows):
    spans = rows[["cycle_id", "entry_index", "exit_index"]].drop_duplicates().sort_values(["entry_index", "exit_index"])
    end, groups = -1, 0
    for row in spans.itertuples():
        if row.entry_index > end:
            groups += 1
        end = max(end, row.exit_index)
    return groups


def train(root, data, samples):
    cfg = read(ROOT / "config/510300_learned_cycle_exit_v1.json")
    first = int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0]) - 1
    streams, receipts = {"ridge": [], "within": []}, []
    for t in range(first, len(data)):
        rows, ids, left, counts, eligible = choose(data, samples, t)
        base = {"fit_index": int(t), "fit_origin": str(data.date.iloc[t].date()),
                "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5), "window_start": str(left.date()),
                "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(rows),
                "cycles_each_signal": counts.to_dict(), "overlap_groups_descriptive": overlap_groups(rows),
                "sample_ids": rows.sample_id.to_list(), "sample_membership_hash": parent.frame_hash(rows[["sample_id", "sample_weight"]]),
                "eligible_for_fit": eligible, "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None,
                "earliest_entry_date": str(rows.entry_date.min().date()) if len(rows) else None,
                "missing_feature_rows": 0, "failure": None,
                "status": "FIT_COMPLETE" if eligible else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS"}
        assert np.isfinite(rows[FEATURES].to_numpy(float)).all()
        for kind in streams:
            streams[kind].append({**deepcopy(base), "model": pooled_fit(rows, kind) if eligible else None})
        receipts.append({k: v for k, v in base.items() if k not in {"sample_ids", "training_cycles", "cycles_each_signal"}} | {"cycles_" + k: v for k, v in counts.items()})
        if (t - first + 1) % 400 == 0:
            print(f"共享训练逐日更新：{t - first + 1}/{len(data) - first}。", flush=True)
    save(root / "results/daily_models.json", streams, True)
    pd.DataFrame(receipts).to_parquet(root / "results/training_receipts.parquet", index=False)
    return streams


def verify(root, data, samples, models):
    counts = {"daily_records": 0, "actual_model_fits": 0, "coefficient_recomputations": 0,
              "future_or_expired_label_checks": 0, "account_checks": 0}
    for kind, records in models.items():
        fitted = [r for r in records if r["status"] == "FIT_COMPLETE"]
        counts["daily_records"] += len(records)
        counts["actual_model_fits"] += len(fitted)
        for r in records:
            if r["latest_exit_index"] is not None:
                assert r["latest_exit_index"] < r["fit_index"]
                assert pd.Timestamp(r["earliest_entry_date"]) >= pd.Timestamp(r["window_start"])
        for r in fitted[::max(1, len(fitted) // 6)]:
            rows, ids, left, per_signal, eligible = choose(data, samples, r["fit_index"])
            assert eligible and ids == r["training_cycles"]
            assert clean(pooled_fit(rows, kind)) == clean(r["model"])
            per_type_weights = rows.groupby("signal").sample_weight.sum().to_numpy()
            np.testing.assert_allclose(per_type_weights, len(ids) / 3, atol=1e-10, rtol=0)
            changed = samples.copy()
            changed.loc[changed.entry_date.lt(left) | changed.exit_index.ge(r["fit_index"]), "target"] = 9999.
            again = choose(data, changed, r["fit_index"])[0]
            pd.testing.assert_frame_equal(rows, again, check_exact=True)
            counts["coefficient_recomputations"] += 1
            counts["future_or_expired_label_checks"] += 1
    for folder in (root / "accounts").glob("*/*"):
        ledger, decisions = pd.read_parquet(folder / "ledger.parquet"), pd.read_parquet(folder / "decisions.parquet")
        assert ledger.accounting_error.abs().max() < 1e-6
        assert ledger.cash.ge(-1e-8).all() and ledger.mark_clock.eq("CLOSE").all()
        feasible = decisions.loc[decisions.risk_plan_feasible]
        assert feasible.planned_target_exposure.le(.5 + 1e-10).all()
        assert (feasible.planned_tail_loss <= feasible.tail_budget + 1e-8).all()
        assert (feasible.planned_gap_loss <= feasible.gap_budget + 1e-8).all()
        assert decisions.loc[~decisions.risk_plan_feasible, "target_shares"].eq(0).all()
        counts["account_checks"] += 1
    assert counts["account_checks"] == 2
    save(root / "verification.json", {"status": "PASS_POOLED_TRAINING_TIME_WEIGHTS_AND_ACCOUNT_BUDGET", **counts}, True)
    return counts


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": parent.now()}, True)
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    data = data.loc[data.date.le(parent.END)].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    samples = make_samples(root, data, dividends)
    streams = train(root, data, samples)
    pipeline = parent.graph_run(root, "POOLED", streams)
    tails = read(PARENT / "results/tail_forecasts.json")
    accounts = {}
    for cost in cfg["costs"]:
        source = pipeline.get(MODEL, cost)
        ledger, decisions, checkpoint = parent.governed_account(data, dividends, cfg, source, tails, PRIMARY, cost)
        folder = root / "accounts" / cost / PRIMARY
        folder.mkdir(parents=True)
        ledger.to_parquet(folder / "ledger.parquet", index=False)
        decisions.to_parquet(folder / "decisions.parquet", index=False)
        save(folder / "checkpoint.json", checkpoint, True)
        accounts[PRIMARY, cost] = ledger
        m = metrics(ledger)
        print(f"共享训练 {cost}：夏普{m['sharpe']}，年化{m['annual_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    measurements, windows = parent.summarize_accounts(root, data, accounts)
    comparisons = []
    rng = np.random.default_rng(20260926)
    for cost in cfg["costs"]:
        for control in [parent.PRIMARY, "ORIGINAL_TAIL"]:
            other = pd.read_parquet(PARENT / "accounts" / cost / control / "ledger.parquet")
            comparisons.append({"cost": cost, "left": PRIMARY, "right": control,
                                **parent.paired_interval(accounts[PRIMARY, cost], other, rng)})
    save(root / "results/paired_comparisons.json", comparisons, True)
    saved_models = read(root / "results/daily_models.json")
    checks = verify(root, data, samples, saved_models)
    primary = next(r for r in measurements if r["cost"] == "STRESS")
    stress_windows = [r for r in windows if r["cost"] == "STRESS"]
    result = {"study_id": STUDY, "at": parent.now(),
              "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_POOLED_DAILY_EXIT",
              "primary": primary, "all_accounts": measurements, "comparisons": comparisons, "checks": checks,
              "new_external_accounts": 2, "new_internal_graph_accounts": 22, "new_reference_reconstructions": 2,
              "rolling_two_year_windows": len(stress_windows), "rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in stress_windows),
              "new_independent_observations": 0, "goal_achieved": False, "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(root / "result.json", result, True)
    print("三类过程共享退出实验完成，旧版与父实验结果保留。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="三类已有过程的两年每日共享训练")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "研究尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
