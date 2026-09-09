"""只读取2014—2021外部数据训练时间迁移模型。"""

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

from research.csi300_temporal_transfer_alpha_v1 import (  # noqa: E402
    FEATURE_COLUMNS,
    TransferRules,
    build_outcomes,
    build_price_factor_panel,
    fit_model,
)
from scripts.freeze_csi300_temporal_transfer_protocol_v1 import (  # noqa: E402
    CONFIG_FILE,
    FROZEN_FILES,
    PROTOCOL_MANIFEST,
    sha256,
)


def _verify(contract: dict) -> dict:
    manifest = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_TRAINING_OUTCOME_READ":
        raise RuntimeError("时间迁移协议尚未冻结")
    changed = [relative for relative in FROZEN_FILES if sha256(ROOT / relative) != manifest["frozen_files"][relative]]
    for relative, expected in manifest["training_inputs"].items():
        if sha256(ROOT / relative) != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError(f"协议冻结后变化：{sorted(set(changed))}")
    return manifest


def _rules(contract: dict) -> TransferRules:
    periods = contract["periods"]
    model = contract["model"]
    return TransferRules(
        start=pd.Timestamp(periods["training_feature_start"]),
        end=pd.Timestamp(periods["training_end"]),
        step=int(periods["rebalance_every_trading_days"]),
        horizon=int(periods["target_horizon_trading_days"]),
        features=FEATURE_COLUMNS,
        minimum_amount=float(contract["selection"]["minimum_20d_average_amount_cny"]),
        learning_rate=float(model["learning_rate"]), max_iter=int(model["max_iter"]),
        max_leaf_nodes=int(model["max_leaf_nodes"]), min_samples_leaf=int(model["min_samples_leaf"]),
        l2_regularization=float(model["l2_regularization"]), max_bins=int(model["max_bins"]),
        random_state=int(model["random_state"]),
        target_clip=(float(model["target_clip"][0]), float(model["target_clip"][1])),
        half_life_days=float(model["training_half_life_calendar_days"]),
    )


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = _verify(contract)
    inputs = contract["training_inputs"]
    panel = pd.read_parquet(ROOT / inputs["member_panel"])
    index = pd.read_parquet(ROOT / inputs["price_index"])
    benchmark = pd.read_parquet(ROOT / inputs["benchmark"])
    rules = _rules(contract)
    features, calendar = build_price_factor_panel(panel, panel, index, rules)
    outcomes = build_outcomes(features, panel, benchmark, calendar, rules.horizon)
    model, training = fit_model(features, outcomes, rules)
    if pd.Timestamp(training["last_maturity_date"]) > rules.end:
        raise AssertionError("训练模型读取了训练截止日之后成熟的标签")
    outputs = contract["outputs"]
    model_path = ROOT / outputs["trained_model"]
    feature_path = ROOT / outputs["training_features"]
    report_path = ROOT / outputs["training_report"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    features.to_parquet(feature_path, index=False)
    report = {
        "project_id": contract["protocol"]["project_id"],
        "state": "TRAINED_ON_EARLY_PERIOD_ONLY_AWAITING_MODEL_FREEZE",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_manifest_sha256": sha256(PROTOCOL_MANIFEST),
        "training_period": {"start": str(rules.start.date()), "end": str(rules.end.date())},
        "recent_evaluation_input_read": False,
        "feature_count": 10,
        "training": training,
        "artifacts": {
            outputs["trained_model"]: sha256(model_path),
            outputs["training_features"]: sha256(feature_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": report["state"], "训练行": training["training_rows"], "训练信号日": training["training_signal_dates"], "最后成熟标签": training["last_maturity_date"], "近期输入读取": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

