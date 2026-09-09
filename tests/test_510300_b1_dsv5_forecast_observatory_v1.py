"""验证前向时钟、禁止仓位列、输入漂移及里程碑，不读取NBS收益。"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
import json
import yaml

from research import b1_dsv5_forecast_observatory_v1 as observer

from research.b1_dsv5_forecast_observatory_v1 import (FIELDS, check_prediction_record, compare_history,
                                                     compare_dividends, origins, review_metrics)
from research.nbs_v2_common import ContractError, TZ, identity, load_json, write_once


def record():
    result = dict.fromkeys(FIELDS)
    result.update(origin_date="2026-09-11", model_vintage_id="合成版本", training_end_date="2026-09-10",
                  b0_predicted_dsv5=.02, b1_predicted_dsv5=.03, q_b1_div_b0=1.5,
                  input_sha256="a"*64, prediction_generated_at="2026-09-11T19:30:01+08:00",
                  label_maturity_date="2026-09-18")
    return result


def test_forecast_schema_has_no_position_columns():
    check_prediction_record(record())
    bad = {**record(), "target_weight": 1.0}
    with pytest.raises(ContractError):
        check_prediction_record(bad)


@pytest.mark.parametrize("field,value", [("prediction_generated_at", "2026-09-12T19:30:01+08:00"),
                                          ("prediction_generated_at", "2026-09-11T15:30:01+08:00"),
                                          ("training_end_date", "2026-09-12")])
def test_historical_backfill_and_unmatured_training_rejected(field, value):
    row = record()
    row[field] = value
    with pytest.raises(ContractError):
        check_prediction_record(row)


def test_input_revision_is_detected():
    old = pd.DataFrame({"date": pd.to_datetime(["2026-09-01", "2026-09-02"]), "open": [4., 4.1], "close": [4.1, 4.2]})
    new = old.copy()
    compare_history(old, new)
    new.loc[1, "close"] = 4.3
    with pytest.raises(ContractError, match="DRIFT"):
        compare_history(old, new)


def test_dividend_history_revision_blocks_but_new_event_is_allowed():
    old = pd.DataFrame({"record_date": ["2026-01-15"], "ex_date": ["2026-01-16"],
                        "payment_date": ["2026-01-20"], "cash_dividend_per_share": [.10]})
    new = pd.concat([old, pd.DataFrame({"record_date": ["2026-10-15"], "ex_date": ["2026-10-16"],
                                      "payment_date": ["2026-10-20"], "cash_dividend_per_share": [.12]})])
    compare_dividends(old, new, pd.Timestamp("2026-08-14"))
    new.iloc[0, 3] = .11
    with pytest.raises(ContractError, match="DRIFT"):
        compare_dividends(old, new, pd.Timestamp("2026-08-14"))


def test_review_happens_only_at_52_or_104_complete_origins():
    assert not review_metrics(np.ones(51), 52)["passed"]
    assert not review_metrics(np.ones(26), 26)["passed"]
    assert review_metrics(np.ones(52), 52)["passed"]
    assert not review_metrics(np.r_[np.ones(26)*2, -np.ones(26)], 52)["passed"]
    assert review_metrics(np.ones(104), 104)["passed"]
    assert not review_metrics(np.r_[np.ones(52)*2, -np.ones(52)], 104)["passed"]


def test_fixed_five_trading_day_grid_never_reanchors_after_missing_day():
    cal = pd.bdate_range("2026-09-01", periods=20)
    config = {"schedule": {"anchor_date": "2026-09-01", "first_new_origin_date": "2026-09-08"}}
    grid = origins(cal, config)
    assert grid.equals(cal[5::5])


def test_synthetic_forward_prediction_maturity_and_idempotency(tmp_path, monkeypatch):
    """隔离时钟和数据源，真实执行风险拟合、到期标签与不可变记录链。"""
    cal = pd.bdate_range(end="2026-10-30", periods=700)
    rng = np.random.default_rng(1711)
    closes = 4 * np.exp(np.cumsum(rng.normal(0, .01, len(cal))))
    prices = pd.DataFrame({"date": cal, "open": closes*np.exp(rng.normal(0, .002, len(cal))),
                           "close": closes, "symbol": "510300.SH"})
    dividends = pd.DataFrame(columns=["symbol", "record_date", "ex_date", "payment_date",
                                      "cash_dividend_per_share", "source"])
    features = observer.parent.build_b1_daily_features(prices, dividends, annualization_days=252)
    seed = pd.DataFrame([{"offset": 0, "origin_date": cal[p], "horizon_end_date": cal[p+5],
                          "DSV5": .002*(.1+(p % 7))} for p in range(40, 440, 5)])
    seed.to_parquet(tmp_path / "seed.parquet", index=False)
    (tmp_path / "parent.yaml").write_text(yaml.safe_dump({"label": {"horizon_trading_days": 5,
                                                                     "annualization_days": 252}}), "utf-8")
    config = {"model_id": "SYNTHETIC_FORECAST_ONLY", "schedule": {"anchor_date": "2026-09-11",
               "first_new_origin_date": "2026-09-11"}, "inputs": {"parent_protocol": "parent.yaml"}}
    write_once(tmp_path / observer.MANIFEST, {"identities": [], "seed_mapping": {"seed_labels": "seed.parquet"}})
    clock = [datetime(2026, 9, 11, 19, 30, 1, tzinfo=TZ)]
    calls = []

    class SyntheticClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    def synthetic_input(root, config, manifest, day):
        calls.append(day)
        path = root / f"input_{day:%Y%m%d}.json"
        write_once(path, {"available_through": str(day.date())})
        return features.loc[features.date <= day].copy(), dividends, identity(root, path)

    monkeypatch.setattr(observer, "datetime", SyntheticClock)
    monkeypatch.setattr(observer, "now", lambda: clock[0].isoformat())
    monkeypatch.setattr(observer, "cfg", lambda root: config)
    monkeypatch.setattr(observer, "calendar", lambda root, cfg: cal)
    monkeypatch.setattr(observer, "risk_input", synthetic_input)
    first = observer.tick(tmp_path)
    assert first["state"] == "RECORDED_FORECAST_ONLY"
    prediction_path = tmp_path / observer.BASE / "predictions/2026-09-11.json"
    prediction = load_json(prediction_path)
    assert prediction["matured_actual_dsv5"] is None
    assert prediction["label_maturity_date"] == "2026-09-18"
    assert observer.tick(tmp_path)["state"] == "ALREADY_RECORDED_IMMUTABLE_FORECAST"
    assert len(calls) == 1
    clock[0] = datetime(2026, 9, 18, 19, 30, 1, tzinfo=TZ)
    second = observer.tick(tmp_path)
    assert second["state"] == "RECORDED_FORECAST_ONLY"
    matured = load_json(tmp_path / observer.BASE / "matured/2026-09-11.json")
    assert matured["matured_actual_dsv5"] >= 0
    assert matured["b1_minus_b0_loss_improvement"] == pytest.approx(matured["b0_qlike"]-matured["b1_qlike"])
    assert load_json(prediction_path) == prediction
    assert load_json(tmp_path / observer.BASE / "predictions/2026-09-18.json")["model_vintage_id"] == prediction["model_vintage_id"]
    ledger = pd.read_csv(tmp_path / observer.BASE / "forecast_ledger.csv")
    assert list(ledger.columns) == FIELDS
    assert len(ledger) == 2
