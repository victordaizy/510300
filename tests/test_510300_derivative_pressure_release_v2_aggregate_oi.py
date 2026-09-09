from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.derivative_pressure_release_v2_aggregate_oi import (
    PROJECT_ID,
    aggregate_if_open_interest,
)
from research.episodic_alpha_library_v1 import prior_rolling_z
from research.frozen_protocol_support_v1 import (
    ProtocolError,
    acquire_immutable_claim,
    strict_json_dumps,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_derivative_pressure_release_v2_aggregate_oi.yaml"


def load_protocol() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_protocol_is_independent_one_shot_and_never_rescues_v1() -> None:
    config = load_protocol()
    protocol = config["protocol"]
    assert protocol["project_id"] == PROJECT_ID
    assert protocol["relation_to_v1"] == "INDEPENDENT_PREFROZEN_DATA_CONSTRUCTION_VERSION"
    assert protocol["v1_status"] == "NO_VIEW_FROZEN_UNCHANGED"
    assert protocol["rescue_v1"] is False
    assert protocol["max_research_runs"] == 1
    assert protocol["post_result_parameter_change"] == "FORBIDDEN"
    assert protocol["v3_or_later_if_not_pass"] == "FORBIDDEN"


def test_aggregate_oi_uses_all_positive_strictly_unexpired_contracts() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["IF2608", "IF2609", "IF2612", "IF2608", "IF2609"],
            "date": pd.to_datetime(["2026-08-21"] * 3 + ["2026-08-24"] * 2),
            "expiry_date": pd.to_datetime(
                ["2026-08-21", "2026-09-18", "2026-12-18", "2026-08-21", "2026-09-18"]
            ),
            "open_interest": [100.0, 200.0, 0.0, 999.0, 250.0],
        }
    )
    result = aggregate_if_open_interest(frame)
    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-21", "2026-08-24"]
    assert result["aggregate_open_interest"].tolist() == [200.0, 250.0]
    assert result["aggregate_contract_count"].tolist() == [1, 1]


def test_standardization_uses_exact_prior_252_and_excludes_current() -> None:
    series = pd.Series(range(253), dtype=float)
    z = prior_rolling_z(series, 252, 252)
    assert z.iloc[:252].isna().all()
    expected = (252.0 - series.iloc[:252].mean()) / series.iloc[:252].std(ddof=1)
    assert z.iloc[252] == pytest.approx(expected)


def test_expiry_day_and_missing_aggregate_rules_are_hard_frozen() -> None:
    config = load_protocol()
    branch = config["branches"]["derivative_pressure_release"]
    assert branch["expiry_day_new_signal"] is False
    assert branch["aggregate_oi_missing_status"] == "NO_VIEW"
    assert branch["aggregate_oi_formula"].startswith("SUM_OPEN_INTEREST_ALL_IF_CONTRACTS")
    assert config["rolling_rules"]["standard_z_window_days"] == 252
    assert config["rolling_rules"]["standard_z_minimum_history_days"] == 252


def test_all_trading_switches_remain_closed() -> None:
    config = load_protocol()
    assert not any(config["boundaries"].values())
    assert config["scope"]["position_impact"] == 0
    assert config["scope"]["live_trading_authorized"] is False


def test_one_shot_claim_cannot_be_created_twice(tmp_path: Path) -> None:
    path = tmp_path / "claim.json"
    payload = {"project_id": PROJECT_ID, "run": 1}
    acquire_immutable_claim(path, payload)
    assert json.loads(path.read_text(encoding="utf-8"))["run"] == 1
    with pytest.raises(FileExistsError):
        acquire_immutable_claim(path, payload)


def test_aggregate_rejects_incomplete_schema() -> None:
    with pytest.raises(ProtocolError):
        aggregate_if_open_interest(pd.DataFrame({"date": [pd.Timestamp("2026-08-21")]}))


def test_strict_json_uses_null_for_missing_and_nonfinite_scalars() -> None:
    payload = json.loads(
        strict_json_dumps(
            {
                "nat": pd.NaT,
                "missing": pd.NA,
                "nan": float("nan"),
                "infinite": float("inf"),
            }
        )
    )
    assert payload == {"nat": None, "missing": None, "nan": None, "infinite": None}
