"""检验九项模型来源接入、连续状态时钟与新资金独立账户。"""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from research.continuous_interaction_budget_v1 import reference_factors, read_models
from research.profit_drawdown_interaction_inputs_v1 import InteractionExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.event_clock_account_v1 import simulate_event_account
from tests.test_continuous_reference_min_variance_v1 import fixture as reference_fixture
from tests.test_median_continuation_v1 import fixture as account_fixture
from tests.test_profit_drawdown_interaction_v1 import stored


def inputs():
    data, references, cfg = reference_fixture()
    references["INTERACTION_REFERENCE"] = references.pop("REARM_RIDGE")
    return data, references, cfg


def test_new_model_list_rejects_old_identity_or_future_maturity(tmp_path):
    p = tmp_path / "九项模型.json"
    m = stored(t=8)
    p.write_text(json.dumps({"models": [m]}), encoding="utf-8")
    assert read_models(p) == [m]
    m["model"]["kind"] = "RIDGE"
    p.write_text(json.dumps({"models": [m]}), encoding="utf-8")
    with pytest.raises(ValueError, match="九项交互模型"):
        read_models(p)
    p.write_text(json.dumps({"models": [stored(t=8, latest=9)]}), encoding="utf-8")
    with pytest.raises(ValueError, match="未来成熟"):
        read_models(p)


def test_new_continuous_reference_obeys_model_clock_and_prefix():
    args = account_fixture()
    model = stored(t=4)
    full, decisions, _ = simulate_rearmed_exit(*args, InteractionExitController(args[0], [model]))
    assert decisions.loc[decisions.origin_index.lt(4), "continuation_prediction"].dropna().empty
    assert full.loc[full.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[6]
    cut = copy.deepcopy(args)
    cut[0] = cut[0].iloc[:8]
    cut[5] = {"entry": cut[5]["entry"][:8], "exit": {1: cut[5]["exit"][1][:8]}}
    prefix, earlier, _ = simulate_rearmed_exit(*cut, InteractionExitController(cut[0], [model, stored(.9, t=9)]))
    assert_frame_equal(full.iloc[:6].reset_index(drop=True), prefix.iloc[:6].reset_index(drop=True))
    assert_frame_equal(decisions.iloc[:7].reset_index(drop=True), earlier.iloc[:7].reset_index(drop=True))


def test_new_reference_budget_uses_its_own_full_past_and_terminal_is_not_close():
    data, references, cfg = inputs()
    full = reference_factors(data, references, 1, cfg)
    prefix = reference_factors(data.iloc[:31], references, 1, cfg)
    assert_frame_equal(full.iloc[:30], prefix.iloc[:30])
    changed = copy.deepcopy(references)
    ledger, decisions = changed["INTERACTION_REFERENCE"]
    ledger.loc[ledger.date.ge(data.date.iloc[30]), "net_return"] = -.9
    decisions.loc[decisions.origin.ge(data.date.iloc[30]), "reference_weight"] = 0.
    assert_frame_equal(full.iloc[:30], reference_factors(data, changed, 1, cfg).iloc[:30])
    assert pd.isna(prefix.target.iloc[-1]) and not prefix.risk_update_scheduled.iloc[-1]
    references["INTERACTION_REFERENCE"][1].loc[12, "reference_weight"] = np.nan
    assert pd.isna(reference_factors(data, references, 1, cfg).target.iloc[12])


def test_new_cash_does_not_inherit_reference_capital_or_old_dividend_entitlements():
    data, references, cfg = inputs()
    factors = reference_factors(data, references, 1, cfg)
    dates = data.date
    data.loc[8:, ["open", "close", "previous_close"]] = 9.9
    data.loc[8, ["previous_close", "dividend"]] = [10., .1]
    div = pd.DataFrame({"record_date": [dates.iloc[7]], "ex_date": [dates.iloc[8]], "payment_date": [dates.iloc[15]], "cash_dividend_per_share": [.1]})
    account = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "weight_band": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    ledger, decisions = simulate_event_account(data, div, account, cost, str(dates.iloc[12].date()), "INTERACTION_NEW_CASH", targets=factors.target.to_numpy(), event_mask=np.ones(len(data), bool))
    assert ledger.shares_before.iloc[0] == 0 and 0 < ledger.shares.iloc[0] < 21000
    assert ledger.dividend_recognized.sum() == 0 and ledger.equity.iloc[0] < 200000.
    assert decisions.origin.iloc[0] == dates.iloc[11] and decisions.execution_date.iloc[0] == dates.iloc[12]
    assert ledger.shares.iloc[-1] == 0 and ledger.accounting_error.abs().max() < 1e-6
