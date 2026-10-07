"""完整原样本的可选残差修正。字段名称显式传入，不修改既有冻结模块。"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import require
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction


KIND = "FIXED_CORE_OPTIONAL_WITHIN_CYCLE_RESIDUAL_RIDGE"


def design(rows: pd.DataFrame, model: dict) -> np.ndarray:
    fields = model["features"]
    raw = rows[fields].to_numpy(float)
    known = np.isfinite(raw).all(axis=1)
    require(np.array_equal(known, rows.auxiliary_available.to_numpy(bool)), "可选原值与分支不一致")
    phi = np.zeros_like(raw)
    if model["identified"] and known.any():
        z = np.clip((raw[known] - np.asarray(model["mean"])) / np.asarray(model["scale"]), -5., 5.)
        phi[known] = z - np.asarray(model["clip_center"])
    return phi


def system(rows: pd.DataFrame, core: dict, model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = design(rows, model)
    baseline = np.asarray([within_cycle_prediction(core, x) for x in rows[BASE_FEATURES].to_numpy(float)])
    residual = rows.target.to_numpy(float) - baseline
    weights = rows.sample_weight.to_numpy(float)
    ids = rows.cycle_id.to_numpy(int)
    dx, dy = np.empty_like(phi), np.empty_like(residual)
    for cycle_id in sorted(set(ids)):
        mask = ids == cycle_id
        require(np.allclose(weights[mask], 1. / mask.sum(), atol=1e-14, rtol=0), "原周期等权被改变")
        dx[mask] = phi[mask] - np.average(phi[mask], axis=0, weights=weights[mask])
        dy[mask] = residual[mask] - np.average(residual[mask], weights=weights[mask])
    return dx, dy, weights


def fit(rows: pd.DataFrame, core: dict, fields: list[str]) -> dict:
    require(len(fields) == 2 and len(set(fields)) == 2, "本模型固定两项可选组件")
    require(len(rows) > 0 and not rows.duplicated(["cycle_id", "origin_index"]).any(), "原训练身份为空或重复")
    require(np.isfinite(rows[BASE_FEATURES + ["target", "sample_weight"]].to_numpy(float)).all(), "原训练值缺失")
    require((rows.sample_weight > 0).all(), "原训练权重非法")
    raw = rows[fields].to_numpy(float)
    known = np.isfinite(raw).all(axis=1)
    require(np.array_equal(known, rows.auxiliary_available.to_numpy(bool)), "训练分支不一致")
    mean, scale, center = np.zeros(2), np.ones(2), np.zeros(2)
    if known.any():
        w = rows.sample_weight.to_numpy(float)[known]
        mean = np.average(raw[known], axis=0, weights=w)
        sd = np.sqrt(np.average((raw[known] - mean) ** 2, axis=0, weights=w))
        scale = np.where(sd > 1e-12, sd, 1.)
        center = np.average(np.clip((raw[known] - mean) / scale, -5., 5.), axis=0, weights=w)
    model = {"kind": KIND, "features": fields, "identified": bool(known.any()), "mean": mean.tolist(),
             "scale": scale.tolist(), "clip_center": center.tolist(), "global_intercept": 0.,
             "ridge_alpha": 1., "feature_clip": 5., "training_rows": len(rows),
             "known_training_rows": int(known.sum()), "unknown_training_rows_retained": int((~known).sum())}
    dx, dy, weights = system(rows, core, model)
    gram = dx.T @ (weights[:, None] * dx) + np.eye(2)
    rhs = dx.T @ (weights * dy)
    beta = np.linalg.solve(gram, rhs)
    require(np.isfinite(beta).all(), "可选修正系数非法")
    mean_error = float(np.max(np.abs(np.average(design(rows, model), axis=0, weights=weights))))
    require(mean_error < 1e-12, "可选修正增加全局截距")
    model.update(coefficients=beta.tolist(), global_design_mean_max=mean_error,
                 normal_equation_gradient_max=float(np.max(np.abs(gram @ beta - rhs))))
    return model


def predict(core: dict, model: dict, base_values: list | np.ndarray,
            raw_values: list | np.ndarray, fields: list[str]) -> tuple[float, float, str]:
    require(model["kind"] == KIND and model["features"] == fields, "可选修正身份或字段不同")
    baseline = within_cycle_prediction(core, base_values)
    raw = np.asarray(raw_values, float)
    require(raw.shape == (2,), "可选修正需要两个原值")
    if not np.isfinite(raw).all() or not model["identified"]:
        return baseline, baseline, "EXACT_CORE_FALLBACK"
    phi = np.clip((raw - np.asarray(model["mean"])) / np.asarray(model["scale"]), -5., 5.) - np.asarray(model["clip_center"])
    return baseline, baseline + float(phi @ np.asarray(model["coefficients"])), "OPTIONAL_CORRECTION_AVAILABLE"


def identity(rows: pd.DataFrame, core: dict, fields: list[str]) -> str:
    columns = ["cycle_id", "origin_index", "exit_index", "target", "sample_weight", "auxiliary_available"] + BASE_FEATURES + fields
    payload = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(np.uint64).tobytes()
    metadata = {"core": core, "fields": fields, "kind": KIND, "alpha": 1., "clip": 5.}
    return hashlib.sha256(payload + json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
