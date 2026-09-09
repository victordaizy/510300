import numpy as np
import pandas as pd

from research.benchmark_relative_objective import (
    evaluate_objective_gates,
    evaluate_relative_ledger,
)


def test_relative_evaluation_detects_positive_excess() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    steps = np.arange(300)
    benchmark = pd.DataFrame({"date": dates, "close": 100.0 * 1.0005 ** steps})
    ledger = pd.DataFrame({"date": dates, "equity": 20_000.0 * 1.0008 ** steps})

    result = evaluate_relative_ledger(
        ledger, benchmark, 20_000.0, trading_days_per_year=242, rolling_window=20
    )

    assert result["annualized_excess"] > 0.0
    assert result["information_ratio"] > 0.0
    assert result["rolling_excess"]["positive_ratio"] == 1.0
    assert all(item["excess_return"] > 0.0 for item in result["chronological_halves"])


def test_gate_requires_stress_evidence() -> None:
    relative = {
        "elapsed_years": 5.0,
        "annualized_excess": 0.01,
        "information_ratio": 0.2,
        "maximum_relative_drawdown": -0.10,
        "rolling_excess": {"median": 0.01, "positive_ratio": 0.70},
        "chronological_halves": [{"excess_return": 0.02}, {"excess_return": 0.01}],
    }
    gates = {
        "minimum_evaluation_years": 4.5,
        "minimum_base_annualized_excess": 0.0,
        "minimum_stress_annualized_excess": 0.0,
        "minimum_rolling_excess_median": 0.0,
        "minimum_rolling_positive_ratio": 0.55,
        "minimum_information_ratio": 0.0,
        "maximum_relative_drawdown": -0.20,
    }

    without_stress = evaluate_objective_gates(relative, None, gates)
    with_stress = evaluate_objective_gates(relative, relative, gates)

    assert without_stress["status"] == "REJECTED_OR_INCOMPLETE"
    assert with_stress["status"] == "RETROSPECTIVE_PASS_FORWARD_ONLY"
