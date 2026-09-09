"""训练完成后封存模型与训练证据，随后才允许下载盲测收益。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts.freeze_a_share_hash_holdout_protocol_v1 import CONFIG_FILE, FROZEN_FILES, MANIFEST_FILE, sha256, tree_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = ROOT / "config" / "a_share_hash_holdout_alpha_v1_model_manifest.json"


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    protocol = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if protocol.get("state") != "FROZEN_BEFORE_STOCK_MASTER_AND_RETURNS":
        raise RuntimeError("初始协议冻结状态错误")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != protocol["frozen_files"][name]]
    if changed:
        raise RuntimeError(f"初始冻结后代码变化：{changed}")
    paths = contract["paths"]
    required = [paths[key] for key in ("master", "training_panel", "training_benchmark", "training_status", "trained_model", "training_report")]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"模型封存输入缺失：{missing}")
    status = json.loads((ROOT / paths["training_status"]).read_text(encoding="utf-8"))
    report = json.loads((ROOT / paths["training_report"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("holdout_returns_downloaded"):
        raise RuntimeError("训练数据状态不允许封存")
    if report.get("status") != "TRAINED_ON_HASH_TRAINING_STOCKS_ONLY_AWAITING_MODEL_FREEZE" or not report.get("training_split_only"):
        raise RuntimeError("模型训练证据不符合隔离要求")
    holdout_paths = [paths[key] for key in ("holdout_panel", "holdout_benchmark", "holdout_status", "result_json")]
    existing = [name for name in holdout_paths if (ROOT / name).exists()]
    if existing:
        raise RuntimeError(f"模型封存前盲测收益或结果已存在：{existing}")
    manifest = {
        "project_id": contract["protocol"]["project_id"], "state": "TRAINED_MODEL_FROZEN_BEFORE_HOLDOUT_RETURN_DOWNLOAD",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_manifest_sha256": sha256(MANIFEST_FILE),
        "trained_model_sha256": sha256(ROOT / paths["trained_model"]),
        "training_report_sha256": sha256(ROOT / paths["training_report"]),
        "training_status_sha256": sha256(ROOT / paths["training_status"]),
        "master_sha256": sha256(ROOT / paths["master"]),
        "training_panel_sha256": sha256(ROOT / paths["training_panel"]),
        "training_benchmark_sha256": sha256(ROOT / paths["training_benchmark"]),
        "training_checkpoint_tree_sha256": tree_sha256(ROOT / paths["training_checkpoint_directory"]),
        "holdout_returns_downloaded_before_freeze": False,
    }
    MODEL_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": manifest["state"], "盲测收益提前下载": False, "模型已封存": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
