"""驱动周期深模块的无前视、状态与收益口径测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.index_driver_episode import (
    EPISODE_OBSERVE_ONLY,
    NO_EPISODE,
    NO_VIEW,
    EpisodeRules,
    EpisodeStructure,
    EvaluationRules,
    OutcomeRules,
    build_driver_episode_structure,
    build_forward_outcomes,
    evaluate_episode_candidates,
)


def _episode_rules(**overrides: object) -> EpisodeRules:
    values: dict[str, object] = {
        "data_cutoff": pd.Timestamp("2026-01-31"),
        "confirmation_lookback_days": 5,
        "maximum_core_industries": 3,
        "minimum_core_positive_contribution_share": 0.60,
        "minimum_core_positive_day_share": 0.80,
        "minimum_internal_advancer_weight_share": 0.55,
        "minimum_effective_positive_contributors": 5.0,
        "maximum_largest_positive_contributor_share": 0.35,
        "require_positive_index_contribution": True,
        "short_lookback_days": 3,
        "minimum_propagation_noncore_advancer_weight_share": 0.50,
        "minimum_propagation_positive_noncore_industry_share": 0.50,
        "exhaustion_minimum_hazard_count": 2,
        "exhaustion_core_slowdown_ratio": 0.50,
        "exhaustion_core_advancer_weight_share": 0.50,
        "exhaustion_minimum_core_positive_days": 2,
        "exhaustion_maximum_largest_positive_contributor_share": 0.50,
        "breakdown_minimum_core_positive_contribution_share": 0.35,
        "breakdown_confirmation_days": 2,
        "maximum_episode_valid_days": 60,
    }
    values.update(overrides)
    return EpisodeRules(**values)


def _synthetic_attribution(
    industry_returns: list[tuple[float, float]],
    no_view_positions: set[int] | None = None,
    concentrated_a: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2026-01-05", periods=len(industry_returns))
    no_view_positions = no_view_positions or set()
    cross_rows: list[dict[str, object]] = []
    industry_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for position, (return_a, return_b) in enumerate(industry_returns):
        date = dates[position]
        contribution_sum = 0.0
        for industry, industry_return in (("A", return_a), ("B", return_b)):
            contributions: list[float] = []
            for member in range(6):
                weight = 1.0 / 12.0
                member_return = industry_return
                if concentrated_a and industry == "A":
                    member_return = industry_return * 5.75 if member == 0 else industry_return * 0.05
                contribution = weight * member_return
                contributions.append(contribution)
                cross_rows.append(
                    {
                        "date": date,
                        "con_code": f"{industry}{member}",
                        "snapshot_weight": weight,
                        "constituent_return_1d": member_return,
                        "industry_l1": industry,
                        "return_available": True,
                        "industry_available": True,
                        "component_return_contribution_1d": contribution,
                    }
                )
            industry_contribution = float(sum(contributions))
            contribution_sum += industry_contribution
            industry_rows.append(
                {
                    "date": date,
                    "industry_l1": industry,
                    "weighted_return_contribution_1d": industry_contribution,
                }
            )
        valid = position not in no_view_positions
        daily_rows.append(
            {
                "date": date,
                "snapshot_weighted_constituent_return_1d": contribution_sum,
                "valid_for_attribution": valid,
                "output": "ATTRIBUTION" if valid else NO_VIEW,
                "failure_category": "PASS" if valid else "SYNTHETIC_MISSING",
            }
        )
    return pd.DataFrame(cross_rows), pd.DataFrame(industry_rows), pd.DataFrame(daily_rows)


def test_confirmation_waits_for_five_observed_days_and_does_not_backfill() -> None:
    inputs = _synthetic_attribution([(0.012, 0.002)] * 7)
    result = build_driver_episode_structure(*inputs, _episode_rules())
    first_four = result.daily_states.iloc[:4]
    assert first_four["research_output"].eq(NO_EPISODE).all()
    assert result.episodes.iloc[0]["confirmation_date"] == pd.Timestamp("2026-01-09")
    assert result.episodes.iloc[0]["formation_date"] == pd.Timestamp("2026-01-05")
    assert not result.daily_states.iloc[:4]["candidate_event"].any()


def test_single_security_dominance_rejects_false_sector_driver() -> None:
    inputs = _synthetic_attribution(
        [(0.012, 0.002)] * 5,
        concentrated_a=True,
    )
    rules = _episode_rules(minimum_effective_positive_contributors=1.0)
    result = build_driver_episode_structure(*inputs, rules)
    final = result.daily_states.iloc[-1]
    assert final["research_output"] == NO_EPISODE
    assert "SINGLE_CONTRIBUTOR_CONCENTRATION_ABOVE_GATE" in final["failure_category"]


def test_core_is_frozen_and_each_candidate_occurs_once_per_episode() -> None:
    returns = [(0.012, 0.002)] * 6 + [(0.006, 0.004)] * 6
    inputs = _synthetic_attribution(returns)
    result = build_driver_episode_structure(*inputs, _episode_rules())
    active = result.daily_states.loc[
        result.daily_states["research_output"].eq(EPISODE_OBSERVE_ONLY)
    ]
    assert active["core_industries"].eq("A").all()
    assert not result.events[["episode_id", "candidate_id"]].duplicated().any()
    assert result.events.loc[result.events["candidate_id"].eq("M1")].shape[0] == 1


def test_no_view_interrupts_episode_and_requires_fresh_confirmation_history() -> None:
    inputs = _synthetic_attribution(
        [(0.012, 0.002)] * 12,
        no_view_positions={6},
    )
    result = build_driver_episode_structure(*inputs, _episode_rules())
    interrupted = result.daily_states.iloc[6]
    assert interrupted["research_output"] == NO_VIEW
    assert result.episodes.iloc[0]["termination_reason"] == "NO_VIEW_INTERRUPTION"
    after = result.daily_states.iloc[7:11]
    assert after["research_output"].eq(NO_EPISODE).all()


def test_forward_outcome_starts_next_open_and_excludes_entry_day_dividend() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "open": [100.0, 100.0, 100.0],
            "low": [100.0, 90.0, 100.0],
            "close": [100.0, 100.0, 100.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "symbol": ["510300.SH", "510300.SH"],
            "ex_date": pd.to_datetime(["2026-01-06", "2026-01-07"]),
            "cash_dividend_per_share": [10.0, 10.0],
        }
    )
    rules = OutcomeRules(
        data_cutoff=pd.Timestamp("2026-01-07"),
        horizon_trading_days=2,
        cash_annual_rate=0.0,
        trading_days_per_year=242,
        middle_minimum_relative_return=0.0,
        bad20_terminal_threshold=-0.05,
        tail20_path_threshold=-0.08,
    )
    outcome = build_forward_outcomes(market, dividends, rules).iloc[0]
    assert np.isclose(outcome["x20"], 0.10)
    assert np.isclose(outcome["min_path20"], -0.10)
    assert bool(outcome["tail20"])
    assert not bool(outcome["middle20"])


def _evaluation_rules() -> EvaluationRules:
    return EvaluationRules(
        bootstrap_repetitions=200,
        bootstrap_event_block_length=3,
        random_seed=20260818,
        signal_delay_days=1,
        minimum_effect_retention=0.70,
        fish_minimum_events=20,
        fish_minimum_lift=1.20,
        fish_bootstrap_lower=1.0,
        fish_minimum_median_x20=0.0,
        fish_minimum_positive_rate=0.55,
        fish_maximum_tail_rate_ratio=1.0,
        fish_split_minimum_lift=1.0,
        tail_minimum_events=10,
        tail_minimum_target_events=5,
        tail_minimum_lift=2.0,
        tail_bootstrap_lower=1.0,
        tail_split_minimum_lift=1.0,
    )


def test_candidate_evaluation_is_deterministic_and_never_authorizes_trading() -> None:
    dates = pd.bdate_range("2025-01-02", periods=120)
    outcomes = pd.DataFrame(
        {
            "signal_date": dates,
            "x20": np.where(np.arange(120) % 2 == 0, 0.02, -0.01),
            "middle20": np.arange(120) % 2 == 0,
            "tail20": np.arange(120) % 12 == 0,
            "bad20": np.arange(120) % 10 == 0,
        }
    )
    fish_positions = list(range(0, 40, 2))
    tail_positions = list(range(80, 100, 2))
    for position in fish_positions:
        outcomes.loc[position : position + 1, ["x20", "middle20", "tail20"]] = [
            0.03,
            True,
            False,
        ]
    for position in tail_positions:
        outcomes.loc[position : position + 1, ["x20", "middle20", "tail20"]] = [
            -0.10,
            False,
            True,
        ]
    event_rows = [
        {
            "event_id": f"EP{index:04d}-M1",
            "candidate_id": "M1",
            "episode_id": f"EP{index:04d}",
            "event_date": dates[position],
        }
        for index, position in enumerate(fish_positions, start=1)
    ] + [
        {
            "event_id": f"EP{index:04d}-T1",
            "candidate_id": "T1",
            "episode_id": f"EP{index:04d}",
            "event_date": dates[position],
        }
        for index, position in enumerate(tail_positions, start=101)
    ]
    structure = EpisodeStructure(
        daily_states=pd.DataFrame(
            {"date": dates, "research_output": [NO_EPISODE] * len(dates)}
        ),
        episodes=pd.DataFrame(),
        events=pd.DataFrame(event_rows),
    )
    first = evaluate_episode_candidates(structure, outcomes, _evaluation_rules())
    second = evaluate_episode_candidates(structure, outcomes, _evaluation_rules())
    assert first == second
    assert not first["safety"]["forecast_eligible_emitted"]
    assert not first["safety"]["position_mapping_enabled"]
    assert not first["safety"]["order_generation_enabled"]
    assert not first["safety"]["broker_connection_enabled"]
