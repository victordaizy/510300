"""仅读取117原成熟模型残差尺度，无新模型或策略收益。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.learned_cycle_exit_v1 import training_rows
from research.robust_cycle_exit_inputs_v1 import original_residual_scale

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_robust_cycle_scale_preflight_20260908"


def main():
    cfg = json.loads((ROOT / "config/510300_market_path_exit_v1.json").read_text(encoding="utf-8"))
    path = ROOT / "reports/research/510300_market_path_exit_v1/saved_models.json"
    originals = json.loads(path.read_text(encoding="utf-8"))["models"]
    samples = pd.read_parquet(ROOT / cfg["saved_market_samples"])
    checks = []
    for record in originals:
        if record["status"] != "FIT_COMPLETE":
            continue
        rows, ids = training_rows(samples, record["fit_index"], cfg)
        require(ids == record["training_cycles"] and len(rows) == record["training_rows"], "稳健尺度不是原成熟完整周期")
        state = original_residual_scale(rows, record["model"])
        delta = 1.345*state["scale"]
        checks.append({"拟合收盘": record["fit_origin"], "成熟周期": len(ids), "原状态行": len(rows), "原残差加权中位数": state["median"],
                       "原残差加权中位绝对偏差": state["mad"], "原残差稳健尺度": state["scale"], "拟采用胡伯边界": delta,
                       "原残差超过该边界行数": int((np.abs(state["residual"]) > delta).sum()),
                       "原残差超过该边界的基础权重比例": float(rows.sample_weight.to_numpy()[np.abs(state["residual"]) > delta].sum()/rows.sample_weight.sum())})
    require(len(checks) == 114 and all(r["原残差稳健尺度"] > 1e-12 for r in checks), "原成熟残差尺度不足，不能改尺度救回")
    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(checks).to_csv(OUT / "原成熟月份残差尺度.csv", index=False, encoding="utf-8-sig")
    result = {"recorded_at": now(), "status": "ORIGINAL_MATURE_RESIDUAL_SCALES_AVAILABLE_NO_NEW_MODELS_OR_ACCOUNTS", "mature_months": len(checks),
              "minimum_scale": min(r["原残差稳健尺度"] for r in checks), "maximum_scale": max(r["原残差稳健尺度"] for r in checks),
              "first_month": checks[0], "last_month": checks[-1], "new_models_or_accounts": 0, "new_reference_accounts": 0,
              "original_models_sha256": digest(path), "source_sha256": digest(Path(__file__)), "goal_achieved": False}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
