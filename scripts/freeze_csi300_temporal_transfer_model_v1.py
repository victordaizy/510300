"""冻结早期训练模型及近期评估输入，之后才允许读取近期收益。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.freeze_csi300_temporal_transfer_protocol_v1 import (
    CONFIG_FILE,
    FROZEN_FILES,
    PROTOCOL_MANIFEST,
    ROOT,
    sha256,
    tree_sha256,
)


MODEL_MANIFEST = ROOT / "config" / "csi300_temporal_transfer_alpha_v1_model_manifest.json"


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    changed = [relative for relative in FROZEN_FILES if sha256(ROOT / relative) != protocol["frozen_files"][relative]]
    if changed:
        raise RuntimeError(f"协议文件变化：{changed}")
    outputs = contract["outputs"]
    model_path = ROOT / outputs["trained_model"]
    training_report_path = ROOT / outputs["training_report"]
    training_features_path = ROOT / outputs["training_features"]
    for path in (model_path, training_report_path, training_features_path):
        if not path.exists():
            raise FileNotFoundError(f"训练产物缺失：{path}")
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training.get("recent_evaluation_input_read") is not False:
        raise RuntimeError("训练阶段近期输入隔离失败")
    if pd_timestamp(training["training"]["last_maturity_date"]) > pd_timestamp(contract["periods"]["training_end"]):
        raise RuntimeError("训练标签越过截止日")
    evaluation_hashes = {}
    for key, relative in contract["evaluation_inputs"].items():
        path = ROOT / relative
        if key == "component_history_cache":
            evaluation_hashes[relative] = tree_sha256(path)
        else:
            evaluation_hashes[relative] = sha256(path)
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "state": "TRAINED_MODEL_AND_EVALUATION_INPUTS_FROZEN",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_manifest_sha256": sha256(PROTOCOL_MANIFEST),
        "trained_model_sha256": sha256(model_path),
        "training_report_sha256": sha256(training_report_path),
        "training_features_sha256": sha256(training_features_path),
        "recent_evaluation_input_read_during_training": False,
        "evaluation_input_hashes": evaluation_hashes,
        "governance": contract["governance"],
    }
    MODEL_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "近期训练期读取": False, "模型已冻结": True}, ensure_ascii=False, indent=2))
    return 0


def pd_timestamp(value: str):
    import pandas as pd

    return pd.Timestamp(value)


if __name__ == "__main__":
    raise SystemExit(main())
