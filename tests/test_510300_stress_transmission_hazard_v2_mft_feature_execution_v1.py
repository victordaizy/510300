from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.stress_transmission_hazard_v2 import (
    ConstituentReturnState,
    build_internal_raw_features,
    build_macro_features,
)
from research.stress_transmission_hazard_v2_mft_features_v1 import (
    build_macro_availability_ledger,
    conservative_next_market_open,
    prepare_reverse_repo_7d_releases,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_mft_feature_execution_v1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contract_keeps_research_and_performance_closed() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    assert config["program"]["research_state"] == "DISCOVERY_ONLY"
    assert config["program"]["return_evaluation"] == "NOT_ALLOWED"
    assert config["program"]["label_read_allowed"] is False
    assert config["program"]["model_training_allowed"] is False
    assert config["program"]["portfolio_evaluation_allowed"] is False
    assert config["program"]["position_impact"] == 0
    assert config["no_view_chain"]["abstain_implies_position"] is False


def test_conservative_daily_release_clock_never_uses_same_session() -> None:
    market_dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    observations = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"]))
    available = conservative_next_market_open(observations, market_dates)
    expected = pd.to_datetime(
        [
            "2020-01-03T01:30:00Z",
            "2020-01-06T01:30:00Z",
            "2020-01-06T01:30:00Z",
        ],
        utc=True,
    )
    assert available.tolist() == expected.tolist()
    local_dates = available.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    assert local_dates.gt(observations.dt.normalize()).all()


def test_published_policy_after_close_is_not_selected_on_same_day() -> None:
    policy = pd.DataFrame(
        {
            "notice_date": ["2020-01-02", "2020-01-03"],
            "published_at": ["2020-01-02 09:20:00", "2020-01-03 15:30:00"],
            "has_seven_day_row": [True, True],
            "seven_day_rate_percent": [2.5, 2.4],
            "parse_status": ["PUBLISHED_7D_OPERATION_RATE"] * 2,
            "source_url": ["https://www.pbc.gov.cn/a", "https://www.pbc.gov.cn/b"],
            "raw_sha256": ["a" * 64, "b" * 64],
            "announcement_number": ["2020-1", "2020-2"],
        }
    )
    release = prepare_reverse_repo_7d_releases(policy, source_sha256="c" * 64)
    dates = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    constant = pd.DataFrame(
        {
            "observation_date": ["2019-12-31"],
            "available_at": ["2020-01-01T01:30:00Z"],
        }
    )
    macro = build_macro_features(
        market_dates=dates,
        earnings_yield_releases=constant.assign(csi300_earnings_yield=8.0),
        china_10y_releases=constant.assign(china_10y_yield=3.0),
        dr007_releases=constant.assign(dr007=2.7),
        reverse_repo_7d_releases=release,
    )
    assert macro["reverse_repo_7d_rate"].tolist() == [2.5, 2.5]
    availability = build_macro_availability_ledger(macro)
    assert availability["reverse_repo_7d_rate_available_by_close"].all()


def _synthetic_members_and_returns(
    dates: pd.DatetimeIndex,
    *,
    switch_position: int | None = None,
    extreme_future_from: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stable = [f"{number:06d}.SZ" for number in range(1, 9)]
    outgoing = ["000101.SZ", "000102.SZ"]
    incoming = ["000201.SZ", "000202.SZ"]
    all_symbols = stable + outgoing + incoming
    membership_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    benchmark_daily: list[float] = []
    for date_index, date in enumerate(dates):
        if switch_position is not None and date_index >= switch_position:
            current = stable + incoming
        else:
            current = stable + outgoing
        for symbol in current:
            membership_rows.append(
                {"date": date, "symbol": symbol, "index_code": "000300"}
            )
        values: list[float] = []
        common = 0.002 * np.sin(date_index / 3.0)
        for symbol_index, symbol in enumerate(all_symbols):
            value = common + 0.0004 * np.cos((date_index + symbol_index) / 5.0)
            if extreme_future_from is not None and date_index >= extreme_future_from:
                value += 0.05 * (-1.0 if symbol_index % 2 else 1.0)
            values.append(value)
            return_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "constituent_return_state": ConstituentReturnState.TRADED_VALID.value,
                    "daily_total_shareholder_return": value,
                    "return_is_usable": True,
                }
            )
        benchmark_daily.append(float(np.mean(values)))
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "close": 1000.0 * np.cumprod(1.0 + np.asarray(benchmark_daily)),
        }
    )
    return pd.DataFrame(membership_rows), pd.DataFrame(return_rows), benchmark


def test_new_members_use_public_price_history_but_not_nonmember_comovement_days() -> None:
    dates = pd.bdate_range("2020-01-02", periods=35)
    membership, returns, benchmark = _synthetic_members_and_returns(
        dates, switch_position=10
    )
    feature = build_internal_raw_features(
        membership=membership,
        classified_returns=returns,
        h00300_total_return_close=benchmark,
        expected_members_per_day=10,
    ).set_index("date")
    ten_member_days = feature.loc[dates[19]]
    fifteen_member_days = feature.loc[dates[24]]
    assert ten_member_days["return20_scoreable_member_count"] == 10
    assert ten_member_days["comovement_scoreable_member_ratio"] == pytest.approx(0.8)
    assert fifteen_member_days["comovement_scoreable_member_ratio"] == pytest.approx(1.0)


def test_internal_features_are_invariant_to_appended_future_prices() -> None:
    dates = pd.bdate_range("2020-01-02", periods=80)
    membership, returns, benchmark = _synthetic_members_and_returns(
        dates, extreme_future_from=70
    )
    full = build_internal_raw_features(
        membership=membership,
        classified_returns=returns,
        h00300_total_return_close=benchmark,
        expected_members_per_day=10,
    )
    prefix_dates = dates[:70]
    prefix = build_internal_raw_features(
        membership=membership.loc[membership["date"].isin(prefix_dates)],
        classified_returns=returns.loc[returns["date"].isin(prefix_dates)],
        h00300_total_return_close=benchmark.loc[benchmark["date"].isin(prefix_dates)],
        expected_members_per_day=10,
    )
    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True),
        full.iloc[:70].reset_index(drop=True),
        check_exact=True,
    )


def test_macro_features_are_invariant_to_future_releases() -> None:
    dates = pd.bdate_range("2020-01-02", periods=30)

    def releases(name: str, values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "observation_date": dates,
                "available_at": (
                    dates.tz_localize("Asia/Shanghai")
                    + pd.Timedelta(hours=9, minutes=30)
                ).tz_convert("UTC"),
                name: values,
            }
        )

    ey = releases("csi300_earnings_yield", np.linspace(6.0, 9.0, len(dates)))
    bond = releases("china_10y_yield", np.linspace(2.0, 3.0, len(dates)))
    dr = releases("dr007", np.linspace(1.8, 3.2, len(dates)))
    policy = releases("reverse_repo_7d_rate", np.linspace(1.7, 2.2, len(dates)))
    full = build_macro_features(
        market_dates=dates,
        earnings_yield_releases=ey,
        china_10y_releases=bond,
        dr007_releases=dr,
        reverse_repo_7d_releases=policy,
    )
    prefix = build_macro_features(
        market_dates=dates[:20],
        earnings_yield_releases=ey,
        china_10y_releases=bond,
        dr007_releases=dr,
        reverse_repo_7d_releases=policy,
    )
    pd.testing.assert_frame_equal(
        prefix.reset_index(drop=True),
        full.iloc[:20].reset_index(drop=True),
        check_exact=True,
    )


def test_core_contract_and_outputs_are_industry_free() -> None:
    config = load_config(DEFAULT_CONFIG)
    feature_text = str(config["feature_contract"]).casefold()
    assert "shenwan" not in feature_text
    for output_name in (
        "internal_features",
        "macro_features",
        "mft_feature_panel",
        "coverage_ledger",
    ):
        lowered = config["outputs"][output_name].casefold()
        assert "industry" not in lowered
        assert "sector" not in lowered
        assert "shenwan" not in lowered


def test_frozen_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / config["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        return
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["execution_id"] == config["program"]["execution_id"]
    assert manifest["actual_label_artifact_read"] is False
    assert manifest["actual_performance_artifact_read"] is False
    assert manifest["position_impact"] == 0
