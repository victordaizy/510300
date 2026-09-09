"""只读取哈希训练组数据训练固定十因子模型。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS, ModelRules, build_features, build_outcomes, fit_model  # noqa: E402

CONFIG_FILE = ROOT / "config" / "a_share_hash_holdout_alpha_v1.yaml"


def rules_from_config(contract: dict, start_key: str, end_key: str) -> ModelRules:
    periods, model, universe = contract["periods"], contract["model"], contract["universe"]
    return ModelRules(
        signal_start=pd.Timestamp(periods[start_key]), signal_end=pd.Timestamp(periods[end_key]),
        rebalance_step=int(periods["rebalance_every_trading_days"]), horizon=int(periods["target_horizon_trading_days"]),
        minimum_amount=float(universe["minimum_20d_average_amount_cny"]), maximum_one_lot=float(universe["maximum_signal_price_for_one_lot_cny"]),
        learning_rate=float(model["learning_rate"]), max_iter=int(model["max_iter"]), max_leaf_nodes=int(model["max_leaf_nodes"]),
        min_samples_leaf=int(model["min_samples_leaf"]), l2_regularization=float(model["l2_regularization"]),
        max_bins=int(model["max_bins"]), random_state=int(model["random_state"]),
        target_clip=(float(model["target_clip"][0]), float(model["target_clip"][1])), half_life_days=float(model["training_half_life_calendar_days"]),
    )


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    paths = contract["paths"]
    status = json.loads((ROOT / paths["training_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("holdout_returns_downloaded"):
        raise RuntimeError("训练数据未通过或盲测收益已提前下载")
    master = pd.read_parquet(ROOT / paths["master"])
    master = master.loc[master["split_group"].eq("TRAIN")].copy()
    panel = pd.read_parquet(ROOT / paths["training_panel"])
    benchmark = pd.read_parquet(ROOT / paths["training_benchmark"])
    if set(panel["con_code"].astype(str)).difference(set(master["ts_code"].astype(str))):
        raise RuntimeError("训练面板混入非训练组证券")
    rules = rules_from_config(contract, "training_signal_start", "training_signal_end")
    features, calendar = build_features(panel, master, benchmark, rules)
    outcomes = build_outcomes(features, panel, benchmark, calendar, rules.horizon)
    model, training = fit_model(features, outcomes, rules)
    model_path = ROOT / paths["trained_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report = {
        "status": "TRAINED_ON_HASH_TRAINING_STOCKS_ONLY_AWAITING_MODEL_FREEZE",
        "trained_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "factor_count": len(FEATURE_COLUMNS), "training_split_only": True, "future_date_labels_read": False,
        "feature_rows": int(len(features)), "ready_rows": int(features["signal_output"].eq("SIGNAL_READY").sum()), **training,
    }
    report_path = ROOT / paths["training_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
