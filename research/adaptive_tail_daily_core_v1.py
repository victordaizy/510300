"""两年窗口内嵌套顺序验证：经验与波动尺度五日尾部预测的日更组合。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np
import pandas as pd

from research.nfci_tail_forecast_increment_v1 import fz0


@dataclass(frozen=True)
class Inputs:
    dates: pd.DatetimeIndex
    states: np.ndarray
    volatility: np.ndarray
    returns5: np.ndarray
    phase_origin: int


def inputs_from(data, labels):
    """缺少末端未成熟标签保持NaN，禁止用补值制造已知结果。"""
    returns = np.full(len(data), np.nan)
    indexes = labels.origin_index.to_numpy(int)
    np.testing.assert_array_equal(labels.exit_index.to_numpy(int), indexes + 6)
    returns[indexes] = labels.gross_return5.to_numpy(float)
    first = int(np.flatnonzero(data.date.ge("2015-01-05"))[0])
    states = np.where(data.mom20.notna(), data.mom20.gt(0).astype(int), -1)
    return Inputs(pd.DatetimeIndex(data.date), states,
                  data.vol20.to_numpy(float), returns, first)


def expert_pair(inputs, origin, window_left):
    """风险训练标签的退出必须严格早于预测原点；原点满足窗口左边界。"""
    start = int(inputs.dates.searchsorted(window_left))
    ids = np.arange(start, origin - 6, dtype=int)
    matching = ids[inputs.states[ids] == inputs.states[origin]] if inputs.states[origin] >= 0 else ids[:0]
    conditional = len(matching) >= 60
    ids = matching if conditional else ids
    if len(ids) < 60:
        return None
    realized, historic_vol = inputs.returns5[ids], inputs.volatility[ids]
    current_vol = inputs.volatility[origin]
    if not (np.isfinite(realized).all() and (realized > -1).all()
            and np.isfinite(historic_vol).all() and (historic_vol > 0).all()
            and np.isfinite(current_vol) and current_vol > 0):
        raise ValueError("风险样本缺失或尺度非法，不能移除不利场景")
    scaled = np.expm1(np.log1p(realized) * current_vol / historic_vol)
    if not np.isfinite(scaled).all():
        raise ValueError("尺度变换产生非有限预测")
    tail_count = max(1, int(np.ceil(.05 * len(ids))))
    pair = np.array([[np.quantile(values, .05), max(0., -np.sort(values)[:tail_count].mean())]
                     for values in [realized, scaled]], dtype=float)
    return {"values": pair, "label_indices": ids, "conditional": conditional,
            "training_rows": len(ids), "latest_label_exit_index": int(ids.max() + 6),
            "membership_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()}


def soft_weight(mean_historical_score, mean_filtered_score):
    """固定温度1的熵正则软权重；较低的联合风险损失获得较高权重。"""
    difference = float(mean_historical_score - mean_filtered_score)
    return float(np.exp(-np.logaddexp(0., -difference)))


def blend(pair, filtered_weight):
    return (1. - filtered_weight) * pair[0] + filtered_weight * pair[1]


def predict(inputs, origin):
    """当前专家与内层评分均完全限制在当前的两个日历年中。"""
    day = inputs.dates[origin]
    left = day - pd.DateOffset(years=2)
    current = expert_pair(inputs, origin, left)
    if current is None:
        raise ValueError("当前两年风险样本不足，不得直接跳过该预测日")
    first_validation = int(inputs.dates.searchsorted(day - pd.DateOffset(years=1)))
    validation_ids = np.arange(first_validation, origin - 6, dtype=int)
    validation_ids = validation_ids[(validation_ids - inputs.phase_origin) % 5 == 0]
    observations, losses, invalid = [], [], 0
    for inner_origin in validation_ids:
        pair = expert_pair(inputs, int(inner_origin), left)
        record = {"outer_origin_index": origin, "outer_origin": day,
                  "outer_window_start": left, "validation_origin_index": int(inner_origin),
                  "validation_origin": inputs.dates[inner_origin],
                  "validation_maturity_index": int(inner_origin + 6),
                  "validation_maturity": inputs.dates[inner_origin + 6],
                  "realized_return5": float(inputs.returns5[inner_origin])}
        if pair is None:
            record["status"] = "INSUFFICIENT_RISK_TRAINING_ROWS"
            observations.append(record)
            continue
        values = pair["values"]
        valid = (values[:, 1] > 0).all() and (-values[:, 1] <= values[:, 0]).all()
        record.update(training_rows=pair["training_rows"], conditional=pair["conditional"],
                      earliest_training_origin_index=int(pair["label_indices"].min()),
                      latest_training_origin_index=int(pair["label_indices"].max()),
                      latest_training_exit_index=pair["latest_label_exit_index"],
                      membership_sha256=pair["membership_sha256"],
                      historical_q05=values[0, 0], historical_es95=values[0, 1],
                      filtered_q05=values[1, 0], filtered_es95=values[1, 1])
        if not valid:
            invalid += 1
            record["status"] = "INVALID_JOINT_FORECAST_WHOLE_DAY_HALF_FALLBACK"
        else:
            score = fz0(inputs.returns5[inner_origin], values[:, 0], -values[:, 1])
            losses.append(score)
            record.update(status="SCORED", historical_fz0=score[0], filtered_fz0=score[1])
        observations.append(record)
    if invalid or len(losses) < 40:
        weight, mean = .5, np.array([np.nan, np.nan])
        status = "FIXED_HALF_INVALID_INNER_FORECAST" if invalid else "FIXED_HALF_INSUFFICIENT_MATURE_VALIDATION"
    else:
        mean = np.mean(losses, axis=0)
        weight = soft_weight(*mean)
        status = "DAILY_TWO_YEAR_NESTED_SCORE_WEIGHT"
    values = current["values"]
    metadata = {"origin_index": origin, "origin": day, "window_start": left,
                "training_rows": current["training_rows"], "conditional": current["conditional"],
                "label_indices": current["label_indices"].tolist(),
                "latest_label_exit_index": current["latest_label_exit_index"],
                "validation_origins": validation_ids.tolist(), "mature_validation_rows": len(losses),
                "invalid_validation_rows": invalid, "weight_status": status,
                "mean_historical_validation_fz0": mean[0], "mean_filtered_validation_fz0": mean[1],
                "filtered_weight": weight, "available": True}
    predictions = {"BASELINE": values[0], "VOL20_FILTERED": values[1],
                   "FIXED_HALF": blend(values, .5), "ADAPTIVE_TWO_YEAR": blend(values, weight)}
    return metadata, predictions, observations


def validate(inputs, baseline_records, filtered_records):
    """只检查信息边界与算术，不按检查结果选择收益最好的模型。"""
    t = next(r["origin_index"] for r in baseline_records if r["origin_index"] >= 1600)
    original = predict(inputs, t)
    left = inputs.dates[t] - pd.DateOffset(years=2)
    modified = inputs.returns5.copy()
    excluded = (inputs.dates < left) | (np.arange(len(inputs.dates)) + 6 >= t)
    modified[excluded] = 10000.
    unseen = predict(replace(inputs, returns5=modified), t)
    np.testing.assert_array_equal(list(original[1].values()), list(unseen[1].values()))
    assert original[0]["filtered_weight"] == unseen[0]["filtered_weight"]
    future_vol = inputs.volatility.copy()
    future_vol[t + 1:] = 9999.
    future = predict(replace(inputs, volatility=future_vol), t)
    np.testing.assert_array_equal(list(original[1].values()), list(future[1].values()))
    flat = predict(replace(inputs, volatility=np.full(len(inputs.dates), .2)), t)
    np.testing.assert_allclose(list(flat[1].values()), np.tile(flat[1]["BASELINE"], (4, 1)), atol=1e-14, rtol=0)
    assert abs(flat[0]["filtered_weight"] - .5) < 1e-14
    assert soft_weight(0., 0.) == .5
    assert soft_weight(0., 1.) < .5 < soft_weight(1., 0.)
    assert np.isfinite([soft_weight(-10000., 10000.), soft_weight(10000., -10000.)]).all()
    old_base = {r["origin_index"]: r for r in baseline_records}
    old_filtered = {r["origin_index"]: r for r in filtered_records}
    for origin in [t, baseline_records[0]["origin_index"], baseline_records[-1]["origin_index"]]:
        pair = expert_pair(inputs, origin, inputs.dates[origin] - pd.DateOffset(years=2))
        np.testing.assert_array_equal(pair["label_indices"], old_base[origin]["label_indices"])
        np.testing.assert_allclose(pair["values"], [[old_base[origin]["q05"], old_base[origin]["es95"]],
                                                   [old_filtered[origin]["q05"], old_filtered[origin]["es95"]]], atol=1e-14, rtol=0)
    for row in original[2]:
        assert row["validation_maturity_index"] < t
        if row["status"] == "SCORED":
            assert row["latest_training_exit_index"] < row["validation_origin_index"]
            assert inputs.dates[row["earliest_training_origin_index"]] >= left
    return {"expired_and_unmatured_labels_do_not_change_forecasts_or_weights": True,
            "future_volatility_does_not_change_forecasts_or_weights": True,
            "constant_volatility_equalizes_experts_and_half_weight": True,
            "weight_direction_and_numerical_stability": True,
            "three_dates_reproduce_both_saved_experts": True,
            "nested_training_and_validation_maturity_order": True}
