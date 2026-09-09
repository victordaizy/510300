"""核对保存分解与原账户结果、净值恒等式和原预测记录的一致性。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_corrected_account_attribution_v1_1 import OUT, ACCOUNTS, COMPONENTS, R11


def main():
    summary = pd.read_csv(OUT / "账户规模与时变损益分解.csv")
    pairs = pd.read_csv(OUT / "同期间策略增量分解.csv")
    config = read(ROOT / "config/510300_forward_eps_monthly_policy_v2.json")
    frames = {}
    max_cash_error = 0.0
    for row in summary.to_dict("records"):
        cost, period, key = row["cost_scenario"], row["period"], row["model"]
        folder, source_model, _ = ACCOUNTS[key]
        base = ROOT / "reports/research" / folder
        ledger = pd.read_parquet(base / "evaluation" / cost / (source_model + "_ledger.parquet"))
        frame = pd.read_parquet(OUT / "decomposition_ledgers" / f"{cost}_{period}_{key}.parquet")
        assert frame.date.tolist() == ledger.loc[ledger.date.ge(pd.Timestamp(row["start"])), "date"].tolist()
        original = ledger.set_index("date").loc[frame.date]
        assert np.array_equal(frame.q_on.to_numpy(), original.shares_before.to_numpy())
        assert np.array_equal(frame.q_day.to_numpy(), original.shares.to_numpy())
        np.testing.assert_allclose(frame.transaction_cost_paid_cny, original.commission.to_numpy() + original.slippage_cost.to_numpy(), atol=1e-10, rtol=0)
        np.testing.assert_allclose(frame.net_return, original.net_return.to_numpy(), atol=1e-12, rtol=0)
        np.testing.assert_allclose(frame.net_cny, original.pnl.to_numpy(), atol=1e-7, rtol=0)
        for suffix in ["return", "cny"]:
            np.testing.assert_allclose(frame[[x + "_" + suffix for x in COMPONENTS[:-1]]].sum(axis=1),
                                       frame["net_" + suffix], atol=1e-7 if suffix == "cny" else 1e-12, rtol=0)
        for component in COMPONENTS:
            np.testing.assert_allclose(frame[component + "_return"].mean() * config["annual_days"],
                                       row[component + "_annual_arithmetic"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(frame[component + "_cny"].sum(), row[component + "_cumulative_cny"], atol=1e-7, rtol=0)
        before_first = frame.previous_nav.iloc[0]
        account_change = float(original.equity.iloc[-1] - before_first)
        error = abs(account_change - float(frame.net_cny.sum()))
        max_cash_error = max(max_cash_error, error)
        assert error < 1e-6
        if period == "FULL_EVALUATION":
            stored = next(x for x in read(base / "result.json")["all_metrics"] if x["cost"] == cost and x["model"] == source_model)
            for field in ["net_sharpe", "annualized_return", "cumulative_return", "max_drawdown", "trade_count"]:
                np.testing.assert_allclose(row[field], stored[field], atol=1e-10, rtol=1e-10)
        old_frame = pd.read_parquet(ROOT / "reports/research/510300_forward_eps_corrected_account_attribution_v1/decomposition_ledgers" / f"{cost}_{period}_{key}.parquet")
        pd.testing.assert_frame_equal(frame[old_frame.columns], old_frame)
        frames[(cost, period, key)] = frame
    for row in pairs.to_dict("records"):
        left = frames[(row["cost_scenario"], row["period"], row["left"])]
        right = frames[(row["cost_scenario"], row["period"], row["right"])]
        for component in COMPONENTS:
            value = (left[component + "_return"] - right[component + "_return"]).mean() * config["annual_days"]
            np.testing.assert_allclose(row[component], value, atol=1e-12, rtol=0)
    forecasts = pd.read_csv(OUT / "已有预测与兑现结果.csv")
    signals = pd.read_parquet(ROOT / "reports/research" / R11 / "signals.parquet").set_index("date")
    labels = pd.read_parquet(ROOT / "reports/research" / R11 / "monthly_features_and_mature_labels.parquet").set_index("origin")
    receipts = read(ROOT / "reports/research" / R11 / "training_receipts.json")["rows"]
    receipts = {(x["model"], x["origin"]): x for x in receipts}
    for row in forecasts.to_dict("records"):
        day = pd.Timestamp(row["origin"])
        np.testing.assert_allclose(row["forecast"], signals.loc[day, row["model"]], atol=1e-12, rtol=0)
        receipt = receipts[(row["model"], str(day.date()))]
        assert receipt["status"] == "TRAINED_MONTHLY_MODEL"
        train = labels.loc[pd.to_datetime(receipt["training_months"])]
        assert train.label_exit_date.le(day).all() and train.index.to_series().lt(day).all()
        np.testing.assert_allclose(row["training_mean_y60"], train.Y60.mean(), atol=1e-12, rtol=0)
        actual = labels.loc[day, "Y60"]
        np.testing.assert_allclose(row["actual_y60"], actual, atol=1e-12, rtol=0, equal_nan=True)
        if np.isfinite(actual):
            np.testing.assert_allclose(row["squared_error"], (row["forecast"] - actual) ** 2, atol=1e-12, rtol=0)
        else:
            assert pd.isna(row.get("squared_error"))
    save(OUT / "saved_numerical_verification.json", {"checked_at": now(),
        "status": "PASS_SAVED_ACCOUNT_SOURCE_METRICS_DECOMPOSITION_AND_FORECAST_CLOCKS",
        "original_accounts_checked": len(ACCOUNTS) * 2, "account_periods_checked": len(summary),
        "paired_decompositions_checked": len(pairs), "saved_forecasts_checked": len(forecasts),
        "maximum_period_net_cash_difference": max_cash_error, "new_models_fit": 0,
        "new_accounts_generated": 0, "new_labels_generated": 0, "new_downloads": 0,
        "random_samples_regenerated": 0, "security_audit_performed": False,
        "all_original_v1_numerical_components_preserved": True, "positive_cost_paid_and_negative_cost_effect_separate": True}, exclusive=True)
    print("原18账户绩效、36期间损益分解、28组比较及108条预测时钟核对通过。", flush=True)


if __name__ == "__main__":
    main()
