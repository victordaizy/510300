"""只用哈希训练组证券与2023年以前标签训练固定十因子模型。"""

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

from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS, build_outcomes, fit_model  # noqa: E402
from research.csi300_retrospective_hash_alpha_v1 import (  # noqa: E402
    build_master,
    build_point_in_time_features,
    load_continuous_benchmark,
    load_continuous_panel,
    read_codes,
)
from scripts.freeze_csi300_retrospective_hash_alpha_v1 import CONFIG_FILE, FROZEN_FILES, sha256  # noqa: E402
from scripts.run_csi300_etf_rotation_alpha_v1 import atomic_parquet, atomic_text  # noqa: E402
from scripts.train_a_share_hash_holdout_alpha_v1 import rules_from_config  # noqa: E402


def verify_protocol(contract: dict) -> dict:
    manifest_path = ROOT / contract["paths"]["protocol_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_HASH_SUBGROUP_FEATURE_AND_RETURN_EVALUATION":
        raise RuntimeError("回溯性哈希协议冻结状态错误")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != manifest["frozen_files"][name]]
    changed += [name for name, expected in manifest["input_files"].items() if sha256(ROOT / name) != expected]
    if changed:
        raise RuntimeError(f"协议冻结后文件变化：{sorted(set(changed))}")
    return manifest


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    protocol = verify_protocol(contract)
    paths = contract["paths"]
    model_manifest_path = ROOT / paths["model_manifest"]
    if model_manifest_path.exists():
        raise FileExistsError("训练模型清单已经存在，禁止覆盖或重新训练")
    inputs = contract["inputs"]
    source_paths = [ROOT / inputs["early_member_panel"], ROOT / inputs["recent_member_panel"]]
    master = build_master(read_codes(source_paths))
    train_master = master.loc[master["split_group"].eq("TRAIN")].copy()
    train_codes = train_master["ts_code"].astype(str).tolist()
    panel, panel_audit = load_continuous_panel(source_paths[0], source_paths[1], train_codes)
    if set(panel["con_code"].unique()).difference(train_codes):
        raise RuntimeError("训练面板混入哈希留出组证券")
    benchmark, benchmark_audit = load_continuous_benchmark(
        ROOT / inputs["early_benchmark"], ROOT / inputs["recent_benchmark"]
    )
    rules = rules_from_config(contract, "training_signal_start", "training_signal_end")
    features, calendar = build_point_in_time_features(panel, train_master, benchmark, rules)
    outcomes = build_outcomes(features, panel, benchmark, calendar, rules.horizon)
    model, training = fit_model(features, outcomes, rules)
    model_path = ROOT / paths["trained_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    feature_path = ROOT / paths["training_features"]
    atomic_parquet(features, feature_path)
    report = {
        "status": "TRAINED_AND_FROZEN_BEFORE_HASH_HOLDOUT_EVALUATION",
        "trained_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "evidence_class": contract["protocol"]["evidence_class"],
        "permanent_blind_claim_allowed": False,
        "training_split_only": True,
        "training_code_count": int(len(train_master)),
        "holdout_code_count": int(master["split_group"].eq("HOLDOUT").sum()),
        "factor_count": len(FEATURE_COLUMNS),
        "feature_rows": int(len(features)),
        "ready_rows": int(features["signal_output"].eq("SIGNAL_READY").sum()),
        "panel_audit": panel_audit,
        "benchmark_audit": benchmark_audit,
        **training,
    }
    report_path = ROOT / paths["training_report"]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), report_path)
    model_manifest = {
        "state": "RETROSPECTIVE_HASH_MODEL_FROZEN_BEFORE_HOLDOUT_SUBGROUP_EVALUATION",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_manifest_sha256": sha256(ROOT / paths["protocol_manifest"]),
        "trained_model_sha256": sha256(model_path),
        "training_features_sha256": sha256(feature_path),
        "training_report_sha256": sha256(report_path),
        "training_codes_sha256": __import__("hashlib").sha256("\n".join(sorted(train_codes)).encode("utf-8")).hexdigest(),
    }
    atomic_text(json.dumps(model_manifest, ensure_ascii=False, indent=2), model_manifest_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
