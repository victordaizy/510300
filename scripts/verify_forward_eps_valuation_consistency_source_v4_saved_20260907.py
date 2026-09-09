"""核对PE关系条件、共同月份、成熟训练与全部进出场账户。"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_two_institution_features_v3 import SOURCES
from research.forward_eps_valuation_consistency_features_v2 import qualify_report, QUALIFIED, PARENT as ORIGINAL_FEATURE_SOURCE
from research.forward_eps_report_valuation_fields_v1 import parse_front_page
from research.forward_eps_guosen_history_v1 import identity
from research.forward_eps_valuation_consistency_policy_v2 import MODELS, CONTROLS, OUT, SOURCE, FEATURE_SOURCE, PARENT, CONFIG
from research.forward_eps_monthly_policy_v1 import mature_training
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import Account, choose_order, normalize_dividends, holding_total_return, interval


def equal_value(actual, expected):
    if isinstance(expected, (float, int, np.number)) and not isinstance(expected, bool):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
    else:
        assert actual == expected


def main():
    config = read(CONFIG)
    result = read(OUT / "result.json")
    monthly = pd.read_parquet(OUT / "monthly_features_and_mature_labels.parquet")
    feature_months = pd.read_parquet(FEATURE_SOURCE / "monthly_two_institution_features.parquet")
    institution = pd.read_parquet(FEATURE_SOURCE / "institution_company_month_evidence.parquet")
    company = pd.read_parquet(FEATURE_SOURCE / "pooled_company_month_features.parquet")
    old_institution = pd.read_parquet(ORIGINAL_FEATURE_SOURCE / "institution_company_month_evidence.parquet")
    old_company = pd.read_parquet(ORIGINAL_FEATURE_SOURCE / "pooled_company_month_features.parquet")
    old_months = pd.read_parquet(ORIGINAL_FEATURE_SOURCE / "monthly_two_institution_features.parquet")
    pd.testing.assert_frame_equal(institution[old_institution.columns], old_institution)
    pd.testing.assert_frame_equal(company[old_company.columns], old_company)
    pd.testing.assert_frame_equal(feature_months[old_months.columns], old_months)
    facts = {}
    for source, settings in SOURCES.items():
        frame = pd.read_parquet(ROOT / "reports/research" / settings["facts"] / "annual_eps_forecast_vintages.parquet")
        for aid, group in frame.groupby("report_id"):
            facts[(source, aid)] = group
    records = read(FEATURE_SOURCE / "report_qualifications.json")["rows"]
    lookup = {}
    for record in records:
        key = (record["institution"], record["report_id"])
        assert key not in lookup
        group = facts[key]
        quote_path = ROOT / record["quote_source_record"]["path"]
        assert identity(quote_path)["sha256"] == record["quote_source_record"]["sha256"]
        if record["institution"] == "guosen":
            observed = parse_front_page(read(quote_path)["pages"][0])["quote"]
            assert observed["value"] == record["reference_price_exact"]
            assert observed["raw"] == record["reference_price_raw"]
        else:
            observed = group.report_reference_close_exact.iloc[0]
            assert (pd.isna(observed) and record["reference_price_exact"] is None) or observed == record["reference_price_exact"]
        rebuilt = qualify_report(group.to_dict("records"), record["reference_price_exact"])
        assert {k: record[k] for k in rebuilt} == rebuilt
        lookup[key] = record
    assert len(lookup) == len(facts)
    for row in institution.itertuples():
        if not np.isfinite(row.reported_earnings_yield):
            assert not np.isfinite(getattr(row, QUALIFIED))
            continue
        record = lookup[(row.institution, row.report_id)]
        group = facts[(row.institution, row.report_id)]
        target = group.loc[group.target_fiscal_year.eq(row.origin.year + 1)]
        assert len(target) == 1
        equal_value(row.reported_earnings_yield, 1 / float(target.pe_value_exact.iloc[0]))
        assert pd.Timestamp(record["information_date"]) < row.origin
        assert row.pe_relation_status == record["status"]
        expected = row.reported_earnings_yield if record["eligible"] else np.nan
        equal_value(getattr(row, QUALIFIED), expected)
    rebuilt_company = institution.groupby(["origin", "ts_code"])[QUALIFIED].mean()
    actual_company = company.set_index(["origin", "ts_code"])[QUALIFIED].sort_index()
    pd.testing.assert_series_equal(actual_company, rebuilt_company.sort_index())
    for origin, group in company.groupby("origin"):
        saved = feature_months.loc[feature_months.origin.eq(origin)].iloc[0]
        values = group[QUALIFIED].dropna()
        assert saved.qualified_pe_company_count == len(values)
        equal_value(saved.pooled_qualified_reported_earnings_yield_median, values.median())
        columns = ["pooled_eps_growth_median", "pooled_profit_revision_median", "pooled_reported_earnings_yield_median", "pooled_qualified_reported_earnings_yield_median"]
        expected = saved.pooled_eps_growth_company_count >= 30 and saved.pooled_profit_revision_company_count >= 15 and len(values) >= 30 and np.isfinite(saved[columns].to_numpy(float)).all()
        assert bool(saved.common_quality_features_valid) == bool(expected)
    assert not institution.duplicated(["origin", "ts_code", "institution"]).any()
    assert institution.groupby("origin").size().eq(600).all()
    for origin, group in institution.groupby("origin", sort=True):
        current = group.loc[group.information_date.notna()]
        assert pd.to_datetime(current.information_date).lt(origin).all()
        assert pd.to_numeric(current.report_age_days).between(1, 180).all()
        prior = group.loc[group.prior_information_date.notna()]
        assert pd.to_datetime(prior.prior_information_date).lt(origin - pd.Timedelta(days=90)).all()
    for column in feature_months:
        pd.testing.assert_series_equal(feature_months[column], monthly[column])
    signals = pd.read_parquet(OUT / "signals.parquet")
    data = pd.read_parquet(PARENT / "features.parquet")
    original_signals = pd.read_parquet(SOURCE / "signals.parquet")
    assert signals.date.tolist() == data.date.tolist()
    pd.testing.assert_series_equal(signals.event_mask, original_signals.event_mask)
    dividends = normalize_dividends(pd.read_csv(ROOT / config["inputs"]["dividends"]))
    labels = 0
    for row in monthly.loc[monthly.Y60.notna()].itertuples():
        start = int(row.origin_index) + 1
        end = int(row.origin_index) + 61
        assert row.label_exit_date == data.date.iloc[end]
        equal_value(row.Y60, holding_total_return(data, dividends, start, end)[0])
        labels += 1
    receipts = read(OUT / "training_receipts.json")["rows"]
    groups = {}
    for receipt in receipts:
        groups.setdefault(receipt["origin"], []).append(receipt)
    for rows in groups.values():
        assert len(rows) == len(MODELS)
        assert len({tuple(x["training_months"]) for x in rows}) == 1
        assert len({x["status"] for x in rows}) == 1
    trained = 0
    for receipt in receipts:
        origin = pd.Timestamp(receipt["origin"])
        current = monthly.loc[monthly.origin.eq(origin)].iloc[0]
        train = mature_training(monthly, origin)
        assert receipt["training_months"] == train.origin.dt.strftime("%Y-%m-%d").tolist()
        assert receipt["training_samples"] == len(train)
        location = int(current.origin_index)
        if receipt["status"] != "TRAINED_MONTHLY_MODEL":
            assert not np.isfinite(signals.loc[location, receipt["model"]])
            continue
        assert current.all_features_valid and len(train) >= config["minimum_train_samples"]
        assert train.label_exit_date.le(origin).all() and train.origin.lt(origin).all()
        columns = MODELS[receipt["model"]]
        model = joblib.load(ROOT / receipt["model_file"]["path"])
        scaler = model.named_steps["standardscaler"]
        assert scaler.n_samples_seen_ == len(train)
        np.testing.assert_allclose(scaler.mean_, train[columns].mean().to_numpy(), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(scaler.var_, train[columns].var(ddof=0).to_numpy(), atol=1e-12, rtol=1e-12)
        assert model.named_steps["ridge"].alpha == config["ridge_alpha"]
        prediction = float(model.predict(pd.DataFrame([current[columns].to_dict()], columns=columns))[0])
        equal_value(prediction, receipt["prediction"])
        equal_value(prediction, signals.loc[location, receipt["model"]])
        trained += 1
    order_checks = exit_checks = entry_checks = 0
    evaluation_dates = data.loc[data.date.ge(pd.Timestamp(config["evaluation_start"])), "date"].tolist()
    for metric in result["all_metrics"]:
        cost = config["costs"][metric["cost"]]
        folder = OUT / "evaluation" / metric["cost"]
        name = metric["model"]
        ledger = pd.read_parquet(folder / (name + "_ledger.parquet"))
        decisions = pd.read_parquet(folder / (name + "_decisions.parquet"))
        assert ledger.date.tolist() == evaluation_dates
        assert not ledger.terminal_unliquidated.iloc[-1]
        assert ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
        for field, value in summarize(ledger, config).items():
            equal_value(metric[field], value)
        np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, atol=1e-7, rtol=0)
        np.testing.assert_allclose(ledger.pnl, ledger.price_pnl + ledger.dividend_recognized - ledger.commission - ledger.slippage_cost, atol=1e-7, rtol=0)
        previous = np.r_[config["initial_capital"], ledger.equity.to_numpy()[:-1]]
        np.testing.assert_allclose(ledger.net_return, ledger.equity.to_numpy() / previous - 1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(ledger.shares - ledger.shares_before, ledger.filled_quantity, atol=0, rtol=0)
        if name in CONTROLS:
            for suffix, frame in [("ledger", ledger), ("decisions", decisions)]:
                original = pd.read_parquet(SOURCE / "evaluation" / metric["cost"] / (CONTROLS[name] + "_" + suffix + ".parquet"))
                pd.testing.assert_frame_equal(original, frame)
            continue
        positions = ledger.set_index("date")
        for decision in decisions.itertuples():
            t = int(decision.origin_index)
            scheduled = bool(signals.event_mask.iloc[t])
            prediction = signals.loc[t, name]
            if not scheduled or not np.isfinite(prediction):
                assert decision.requested_quantity == 0
            else:
                if decision.origin in positions.index:
                    held = positions.loc[decision.origin]
                    account = Account(float(held.cash), int(held.shares), {0: float(held.dividend_receivable)})
                else:
                    account = Account(config["initial_capital"])
                chosen = choose_order(account, float(data.close.iloc[t]), float(prediction),
                                      60 * float(data.variance60.iloc[t]), cost, config)
                assert decision.requested_quantity == chosen["requested_quantity"]
                equal_value(decision.reference_weight, chosen["reference_weight"])
                equal_value(decision.score, chosen["score"])
                order_checks += 1
            execution = positions.loc[decision.execution_date]
            assert decision.execution_date == data.date.iloc[t + 1]
            if execution.mark_clock != "OPEN_TERMINAL":
                assert decision.requested_quantity == execution.requested_quantity
        nonterminal = ledger.loc[ledger.mark_clock.ne("OPEN_TERMINAL")]
        exit_checks += int((nonterminal.filled_quantity.lt(0) & nonterminal.shares.eq(0)).sum())
        entry_checks += int((nonterminal.filled_quantity.gt(0) & nonterminal.shares_before.eq(0)).sum())
    for cost in config["costs"]:
        for block in config["bootstrap_day_blocks"]:
            saved = pd.read_parquet(OUT / f"{cost}_block{block}_saved_bootstrap_statistics.parquet")
            assert len(saved) == config["bootstrap_repetitions"]
            for column in saved:
                rebuilt_interval = interval(saved[column].dropna().tolist())
                expected_interval = result["uncertainty"][cost][str(block)][column + "_95_interval"]
                if rebuilt_interval == [None, None]:
                    assert expected_interval == [None, None]
                else:
                    np.testing.assert_allclose(rebuilt_interval, expected_interval, atol=1e-12, rtol=0)
    save(OUT / "saved_numerical_verification.json", {
        "checked_at": now(), "status": "PASS_SAVED_PE_RELATIONS_MATCHED_FEATURES_MODELS_ORDERS_AND_ACCOUNTS",
        "institution_company_month_rows_checked": len(institution), "monthly_feature_rows_checked": len(monthly),
        "saved_labels_checked": labels, "saved_trained_models_checked": trained,
        "matched_training_calendars_identical": True, "source_report_qualifications_checked": len(records),
        "all_original_eps_profit_and_reported_pe_values_preserved": True, "valid_month_end_orders_recomputed_from_saved_account": order_checks,
        "autonomous_full_exits_checked_both_costs": exit_checks, "entries_checked_both_costs": entry_checks,
        "complete_accounts_checked": len(result["all_metrics"]), "reused_control_accounts_checked": 4,
        "saved_bootstrap_files_checked": 4, "new_models_fit": 0, "new_accounts_generated": 0,
        "new_downloads": 0, "random_samples_regenerated": 0, "security_audit_performed": False,
    }, exclusive=True)
    print("PE原件关系、原EPS保留、共同月份、保存模型和十条进出场账户核对通过。", flush=True)


if __name__ == "__main__":
    main()
