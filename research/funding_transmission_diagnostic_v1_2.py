"""只读既有预测和账户，定位资金信息到成交收益的断点。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.funding_afternoon_response_daily_v1 as prior
import research.repo_segmentation_daily_v1 as parent
import research.intraday_overnight_increment_v1 as engine
from research.selected_mix_reappraisal_v1 import read, save, digest, now

OUT = ROOT / "reports/research/510300_funding_transmission_diagnostic_v1_2"
COMPLETE = ROOT / "reports/research/510300_funding_afternoon_response_saved_completion_v1"
STUDY = "510300_FUNDING_TRANSMISSION_DIAGNOSTIC_V1_2"
POLICIES = ["PRICE", prior.FUNDING, prior.RESPONSE, prior.PRIMARY]
ECB_URL = "https://www.ecb.europa.eu/press/financial-stability-publications/fsr/special/html/ecb.fsrart202305_01~830184261b.en.html"


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("断点诊断已经固定，不能覆盖。")
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir(exist_ok=True)
    files = [Path(__file__), COMPLETE / "result.json", prior.OUT / "inputs/decision_information.parquet",
             prior.OUT / "results/predictions.parquet", prior.OUT / "results/update_receipts.parquet",
             prior.OUT / "results/saved_distributions.json", parent.OUT / "inputs/mature_labels.parquet"]
    for policy in POLICIES:
        files.extend(prior.OUT / "accounts/STRESS" / policy / name for name in ["ledger.parquet", "decisions.parquet"])
    save(OUT / "protocol.json", {
        "at": now(), "study_id": STUDY,
        "question": "资金信息到收益的断点出现在预测、弱势入场条件、成本效用、执行还是持有路径？",
        "scope": "只读四个已保存的压力账户、920日预测和原训练样本；不改参数、不生成新账户。",
        "flat_first_refusal": ["无预测", "均值非正", "价格不弱", "预算连一手也不允许", "扣费和方差惩罚后不值得进入", "计划买入但未成交", "实际买入"],
        "inventory_days": "已有库存另列增仓/不增仓，不能把维持库存重复算作新入场。",
        "prediction_groups": ["全部成熟预测", "事前五日下跌", "事前五日下跌且均值为正", "实际买入日", "全部未买日"],
        "fixed_periods": [[parent.START, "2023-12-31"], ["2024-01-01", parent.END]],
        "uncertainty": "每组五日收益均值给20个交易日循环区块2000次区间，seed2026092602；保持被选择日期掩码，不当作独立交易或全项目多重筛选校正。",
        "horizon": "实际买入日的原五日标的含息标签只作诊断，不等于固定五日账户；无挑选收益最好的持有期限。",
        "hold_path": "按原实际份额分拆旧仓隔夜、新旧仓日内、股息、费用；恒等式复算，不运行零成本策略。",
        "neighbor_diagnostic": "只统计选择比例、样本年龄、标签覆盖重叠和预测与同池HISTORY的差异，不据诊断调近邻数。",
        "source_context": {"url": ECB_URL, "publication": "2023-05", "used_as_alpha_evidence": False,
                           "note": "融资流动性与市场流动性不同；融资环境改善不自动等于510300存在可交易折价。"},
        "engineering_correction": "原V1使用仅在成交行填写的shares_after分解日内损益，核验报NaN。改用连续账本shares及其一日滞后；没有改账户、参数或分组。另在不重叠计数前重置索引，消除idx同时是列和索引的歧义；两次原失败保留。",
        "post_result_diagnostic": True, "new_fits": 0, "new_accounts": 0, "new_market_downloads": 0,
        "goal_achieved": False, "orders_authorized": False,
    }, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "freeze.json", {"at": now(), "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
                              "protocol_sha256": digest(OUT / "protocol.json")}, True)
    print("资金到交易收益的断点诊断已固定，不运行新策略。", flush=True)


def interval(values, mask, rng):
    """对完整日历行抽区块，避免把稀疏、重叠五日样本当独立样本。"""
    values, mask = np.asarray(values, float), np.asarray(mask, bool)
    valid = mask & np.isfinite(values)
    if not valid.any():
        return {"n": 0, "mean_return5": None, "lower95": None, "upper95": None}
    n, length = len(values), 20
    draws = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / length)))
        chosen = ((starts[:, None] + np.arange(length)) % n).ravel()[:n]
        accepted = chosen[valid[chosen]]
        if len(accepted):
            draws.append(float(values[accepted].mean()))
    bounds = np.quantile(draws, [.025, .975])
    return {"n": int(valid.sum()), "mean_return5": float(values[valid].mean()),
            "lower95": float(bounds[0]), "upper95": float(bounds[1]), "valid_bootstrap_draws": len(draws)}


def nonoverlap_count(rows):
    last, count = -1, 0
    for row in rows.reset_index(drop=True).sort_values("idx").itertuples():
        if int(row.idx) >= last:
            last, count = int(row.exit_idx), count + 1
    return count


def refusal(row, ledger_row):
    if not bool(row.prediction_available):
        return "NO_FORECAST"
    if row.shares_before_decision > 0:
        return "INVENTORY_ADD" if row.filled_quantity > 0 else "INVENTORY_NO_ADD"
    if row.mu5 <= 0:
        return "NONPOSITIVE_MEAN"
    if row.pressure5 <= 0:
        return "PRICE_NOT_WEAK"
    if row.pre_open_requested_quantity > 0:
        return "FILLED_NEW_ENTRY" if row.filled_quantity > 0 else "PLANNED_ENTRY_NOT_FILLED"
    context = row._asdict()
    cost = parent.distribution.COSTS["STRESS"]
    feasible = parent.risk_valid(engine, 100, row.reference_price, context, cost)
    affordable = engine.affordable_quantity(ledger_row.cash_before,
        engine.fill_price(row.reference_price, 1, cost, .001), cost, 100) >= 100
    return "NET_UTILITY_NO_ENTRY" if feasible and affordable else "NO_FEASIBLE_LOT"


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["files"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    predictions = pd.read_parquet(prior.OUT / "results/predictions.parquet")
    labels = pd.read_parquet(parent.OUT / "inputs/mature_labels.parquet")
    x = pd.read_parquet(prior.OUT / "inputs/decision_information.parquet")
    history = predictions[predictions.model.eq("HISTORY")][["idx", "mu5"]].rename(columns={"mu5": "history_mu5"})
    by_idx = labels.set_index("idx", drop=False)
    saved = read(prior.OUT / "results/saved_distributions.json")
    rng = np.random.default_rng(2026092602)
    all_gates, gate_summary, predictive, attribution, neighbor_summary = [], [], [], [], []
    for policy in POLICIES:
        folder = prior.OUT / "accounts/STRESS" / policy
        ledger = pd.read_parquet(folder / "ledger.parquet")
        decisions = pd.read_parquet(folder / "decisions.parquet")
        assert ledger.idx.equals(decisions.idx)
        gates = pd.DataFrame({"date": decisions.date, "idx": decisions.idx, "policy": policy,
            "first_refusal": [refusal(a, b) for a, b in zip(decisions.itertuples(), ledger.itertuples())]})
        gates["bought"] = decisions.filled_quantity.gt(0)
        all_gates.append(gates)
        for name, part in [("FULL", gates), ("2020_2023", gates[gates.date.lt("2024-01-01")]),
                           ("2024_END", gates[gates.date.ge("2024-01-01")])]:
            counts = {str(k): int(v) for k, v in part.first_refusal.value_counts().items()}
            assert sum(counts.values()) == len(part)
            gate_summary.append({"policy": policy, "period": name, "days": len(part), "counts": counts})
        scored = decisions[["idx", "date", "pressure5", "filled_quantity", "prediction_available"]].merge(
            predictions[predictions.model.eq(policy)][["idx", "mu5", "n_train", "n_selected"]], on="idx", how="left", validate="one_to_one")
        scored = scored.merge(labels[["idx", "exit_idx", "gross_return5"]], on="idx", how="left", validate="one_to_one")
        scored = scored.merge(history, on="idx", how="left", validate="one_to_one")
        masks = {"ALL": scored.prediction_available, "WEAK": scored.prediction_available & scored.pressure5.gt(0),
                 "WEAK_POSITIVE_MEAN": scored.prediction_available & scored.pressure5.gt(0) & scored.mu5.gt(0),
                 "ACTUAL_BUY": scored.filled_quantity.gt(0),
                 "NO_BUY": scored.prediction_available & scored.filled_quantity.le(0)}
        for subset, mask in masks.items():
            selected = scored[mask & scored.gross_return5.notna()]
            record = {"policy": policy, "subset": subset, **interval(scored.gross_return5, mask, rng),
                "greedy_nonoverlapping_labels": nonoverlap_count(selected),
                "mean_forecast": float(selected.mu5.mean()) if len(selected) else None,
                "mean_history_forecast": float(selected.history_mu5.mean()) if len(selected) else None,
                "positive_realized_fraction": float(selected.gross_return5.gt(0).mean()) if len(selected) else None}
            predictive.append(record)
        scored.to_parquet(OUT / "results" / f"{policy}_decision_outcomes.parquet", index=False)
        previous_close = ledger.mark.shift(1)
        first_idx = int(ledger.idx.iloc[0])
        market = pd.read_parquet(parent.OUT / "inputs/market.parquet", columns=["close"])
        previous_close.iloc[0] = float(market.close.iloc[first_idx - 1])
        # 无成交日的执行器 shares_after 为空；每日持仓以连续账本 shares 为准。
        old_shares = ledger.shares.shift(1, fill_value=0)
        overnight = old_shares * (ledger.open - previous_close)
        intraday = ledger.shares * (ledger.mark - ledger.open)
        np.testing.assert_allclose(overnight + intraday, ledger.price_pnl, atol=1e-7, rtol=0)
        costs = float((ledger.commission + ledger.slippage_cost).sum())
        dividends = float(ledger.dividend_recognized.sum())
        total = float(overnight.sum() + intraday.sum() + dividends - costs - ledger.terminal_exit_reserve.iloc[-1])
        np.testing.assert_allclose(total, ledger.equity.iloc[-1] - 200000, atol=1e-6, rtol=0)
        attribution.append({"policy": policy, "overnight_price_pnl": float(overnight.sum()),
            "intraday_price_pnl": float(intraday.sum()), "dividend_pnl": dividends,
            "costs": costs, "terminal_exit_reserve": float(ledger.terminal_exit_reserve.iloc[-1]), "net_pnl": total})
        neighbor_rows = []
        for model in saved:
            if model["model"] != policy:
                continue
            chosen = by_idx.loc[model["selected_indices"]]
            index = int(model["idx"])
            age = (x.date.iloc[index] - chosen.date).dt.days
            neighbor_rows.append({"idx": index, "selected_fraction": len(chosen) / len(model["training_indices"]),
                "mean_age_calendar_days": float(age.mean()), "youngest_age_calendar_days": int(age.min()),
                "greedy_nonoverlapping_training_labels": nonoverlap_count(chosen)})
        nr = pd.DataFrame(neighbor_rows)
        nr.to_parquet(OUT / "results" / f"{policy}_neighbor_support.parquet", index=False)
        pair = scored[scored.prediction_available]
        neighbor_summary.append({"policy": policy, "prediction_days": len(pair),
            "correlation_with_history_mean": float(pair.mu5.corr(pair.history_mu5)),
            "mean_absolute_increment_vs_history": float((pair.mu5 - pair.history_mu5).abs().mean()),
            "mean_selected_fraction": float(nr.selected_fraction.mean()),
            "mean_neighbor_age_calendar_days": float(nr.mean_age_calendar_days.mean()),
            "min_nonoverlapping_training_labels": int(nr.greedy_nonoverlapping_training_labels.min()),
            "median_nonoverlapping_training_labels": float(nr.greedy_nonoverlapping_training_labels.median())})
        print(f"{policy}：逐层拒绝原因、五日预测结果和实际持仓损益分解完成。", flush=True)
    pd.concat(all_gates, ignore_index=True).to_parquet(OUT / "results/decision_gates.parquet", index=False)
    for path, sha in frozen["files"].items():
        assert digest(ROOT / path) == sha, path
    result = {"at": now(), "study_id": STUDY, "status": "COMPLETED_SAVED_EVIDENCE_DIAGNOSTIC",
              "decision_rows": sum(len(g) for g in all_gates), "gate_summary": gate_summary,
              "five_day_outcomes": predictive, "fixed_share_path_attribution": attribution,
              "neighbor_support": neighbor_summary, "new_fits": 0, "new_accounts": 0,
              "new_market_downloads": 0, "original_files_unchanged": True,
              "post_result_diagnostic": True, "independent_validation": False, "goal_achieved": False}
    save(OUT / "result.json", result, True)
    print("断点诊断完成；原四个压力账户和全部参数保持原值。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            save(OUT / "RUN_FAILURE.json", {"at": now(), "type": type(exc).__name__, "message": str(exc)}, True)
            raise
