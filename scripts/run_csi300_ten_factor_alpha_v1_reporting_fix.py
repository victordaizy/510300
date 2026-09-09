"""十因子V1报告字段重名修复；不改变因子、模型、目标或交易。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_csi300_ten_factor_alpha_v1 as original
from scripts.freeze_csi300_ten_factor_alpha_v1 import sha256


FIX_MANIFEST = ROOT / "config" / "csi300_ten_factor_alpha_v1_reporting_fix_manifest.json"


def corrected_prediction_diagnostics(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> dict:
    """复用预测表已有标签，缺失时才合并，避免同名列加后缀。"""

    evaluated = predictions.loc[
        predictions["model_output"].eq("PREDICTION_AUDIT_ONLY")
    ].copy()
    target = "future_excess_log_return_10d"
    if target not in evaluated.columns:
        evaluated = evaluated.merge(
            outcomes[["date", "con_code", target]],
            on=["date", "con_code"],
            how="left",
            validate="one_to_one",
        )
    matured = evaluated.dropna(
        subset=["predicted_excess_log_return_10d", target]
    )
    information_coefficients = matured.groupby("date", sort=True).apply(
        lambda frame: frame["predicted_excess_log_return_10d"].corr(
            frame[target], method="spearman"
        ),
        include_groups=False,
    )
    top = matured.loc[matured["prediction_rank"].le(3)]
    return {
        "prediction_rows": int(len(predictions)),
        "model_ready_rows": int(predictions["model_output"].eq("PREDICTION_AUDIT_ONLY").sum()),
        "model_signal_dates": int(
            predictions.loc[
                predictions["model_output"].eq("PREDICTION_AUDIT_ONLY"), "date"
            ].nunique()
        ),
        "matured_evaluation_rows": int(len(matured)),
        "mean_daily_spearman_ic": float(information_coefficients.mean()),
        "median_daily_spearman_ic": float(information_coefficients.median()),
        "positive_daily_ic_ratio": float(information_coefficients.gt(0).mean()),
        "top3_mean_future_excess_log_return_10d": float(top[target].mean()),
        "top3_positive_excess_ratio": float(top[target].gt(0).mean()),
    }


def main() -> int:
    if not FIX_MANIFEST.exists():
        raise FileNotFoundError("报告修复尚未冻结")
    manifest = json.loads(FIX_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["reporting_fix_file"]
    current = sha256(Path(__file__).resolve())
    if current != expected:
        raise RuntimeError("报告修复脚本冻结后变化")
    original._prediction_diagnostics = corrected_prediction_diagnostics
    return original.main()


if __name__ == "__main__":
    raise SystemExit(main())
