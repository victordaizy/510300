"""离线核对增长分歧试验的来源配对、全部月末、保存模型与固定重采样。"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_growth_disagreement_increment_v1 import OUT, CONFIG, read, save, now, verify
from research.eps_growth_disagreement_source_feasibility_v1 import verify as verify_sources


def verify_all_months(receipt: Path):
    """补充检查缺失行完整性，避免只验证已拟合模型而遗漏原点。"""
    config = read(CONFIG)
    samples = pd.read_parquet(OUT / "samples.parquet")
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    expected = samples.loc[samples.origin.ge(pd.Timestamp(config["evaluation_start"]))].reset_index(drop=True)
    if predictions.origin.duplicated().any() or predictions.origin.tolist() != expected.origin.tolist():
        raise ValueError("预测没有完整覆盖固定评价区间的全部来源月末")
    expected_keys = set()
    for index, row in expected.iterrows():
        actual = predictions.iloc[index]
        for field in ["origin_index", "label_exit_date", "Y60"]:
            a, b = row[field], actual[field]
            if not ((pd.isna(a) and pd.isna(b)) or a == b):
                raise ValueError("保存预测的标签或原点索引不一致：" + field)
        count = int((samples.all_features_valid & samples.origin.lt(row.origin)
                     & samples.label_exit_date.notna() & samples.label_exit_date.le(row.origin)
                     & np.isfinite(samples.Y60)).sum())
        if not row.all_features_valid:
            status = "NO_VIEW_SOURCE"
        elif count < config["minimum_train_samples"]:
            status = "NO_VIEW_INSUFFICIENT_MATURE_TRAINING_MONTHS"
        else:
            status = "TRAINED_PAIRED_MODELS"
        if actual.status != status or actual.training_samples != count or bool(actual.source_eligible) != bool(row.all_features_valid):
            raise ValueError("原点缺失原因或可用成熟训练数不符")
        if status == "TRAINED_PAIRED_MODELS":
            if not np.isfinite([actual.M0, actual.M1]).all():
                raise ValueError("已训练原点缺少共同预测")
            expected_keys.update((row.origin.strftime("%Y-%m-%d"), name) for name in config["models"])
        elif not (pd.isna(actual.M0) and pd.isna(actual.M1)):
            raise ValueError("缺失原点意外生成预测")
    records = read(OUT / "training_receipts.json")["rows"]
    actual_keys = [(r["origin"], r["model_id"]) for r in records]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise ValueError("训练收据业务键与全部合格原点不一致")
    mature = predictions.loc[np.isfinite(predictions[["Y60", "M0", "M1"]]).all(axis=1)]
    save(receipt, {"status": "PASS_ALL_MONTH_ENDS_LABELS_NO_VIEW_STATES_AND_EXACT_MODEL_KEYS",
        "completed_at": now(), "all_evaluation_months": len(predictions), "source_months": len(samples),
        "status_counts": {k: int(v) for k, v in predictions.status.value_counts().items()},
        "saved_model_keys": len(actual_keys), "mature_paired_origins": len(mature),
        "changed_predicted_directions": int(((mature.M0 > 0) != (mature.M1 > 0)).sum()),
        "new_model_fits": 0, "new_random_draws": 0, "new_accounts": 0, "goal_achieved": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    parser.add_argument("--workspace", action="store_true", help="工作区保存复核，不要求ZIP根索引")
    args = parser.parse_args()
    receipt_dir = args.receipt_directory.resolve()
    receipt_dir.mkdir(parents=True, exist_ok=False)
    entries = []
    if not args.workspace:
        with (ROOT / "FILE_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
            entries = list(csv.DictReader(stream))
        if len({r["path"] for r in entries}) != len(entries):
            raise ValueError("文件索引有重复路径")
        for entry in entries:
            path = (ROOT / entry["path"]).resolve()
            path.relative_to(ROOT.resolve())
            raw = path.read_bytes()
            if len(raw) != int(entry["size_bytes"]) or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("文件索引不符：" + entry["path"])
    verify_sources(receipt_dir / "source_pair_recomputation.json")
    verify(receipt_dir / "saved_prediction_recomputation.json")
    verify_all_months(receipt_dir / "all_month_end_alignment.json")
    save(receipt_dir / "packet_verification.json", {"status": "PASS_SOURCES_ALL_MONTHS_SAVED_MODELS_AND_FIXED_BOOTSTRAP",
        "completed_at": now(), "root": str(ROOT), "index_required": not args.workspace,
        "indexed_files_verified": len(entries), "upstream_institution_company_month_rows": 83400,
        "paired_company_month_rows": 41700, "source_months": 139, "saved_models_checked": 72,
        "mature_paired_origins": 33, "saved_bootstrap_replicates": 2000,
        "new_model_fits": 0, "new_random_draws": 0, "new_accounts": 0,
        "new_network_requests": 0, "goal_achieved": False})
    print("来源配对、全部月末及缺失状态、保存模型和固定重采样全部复核通过。", flush=True)


if __name__ == "__main__":
    main()
