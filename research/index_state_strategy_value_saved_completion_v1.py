"""复用固定状态策略分配，修正单个机器精度边界并完成全部账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import FunctionType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.index_state_strategy_value_v1 as parent
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, now

OUT = ROOT / "reports/research/510300_index_state_strategy_value_saved_completion_v1"
STUDY = "510300_INDEX_STATE_STRATEGY_VALUE_SAVED_COMPLETION_V1"


def freeze(root):
    parent.verify_sources(parent.OUT)
    if (root / "freeze.json").exists():
        raise RuntimeError("保存结果完成流程已经固定。")
    source = pd.read_parquet(parent.OUT / "results/allocation_predictions.parquet")
    bad = source.loc[~source.raw_target.between(0, 1), ["idx", "date", "model", "raw_target"]]
    assert len(bad) == 1 and bad.raw_target.between(-1e-12, 1 + 1e-12).all()
    for name in ["code", "inputs", "results", "accounts"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    shutil.copytree(parent.OUT / "references", root / "references")
    for path in (parent.OUT / "results").iterdir():
        if path.is_file():
            shutil.copy2(path, root / "inputs" / path.name)
    protocol = {**read(parent.OUT / "protocol.json"), "study_id": STUDY, "at": now(),
                "parent_study": parent.STUDY, "parent_status": "INTERRUPTED_NUMERIC_BOUNDARY_AFTER_ONE_ACCOUNT",
                "correction": "原始连续分配产生一条1.0000000000000002。仅对绝对超界不超过1e-12的合成仓位截断到[0,1]；超出容差仍拒绝。",
                "original_bad_rows": bad.to_dict("records"),
                "economic_rules_changed": False, "new_allocation_fits": 0, "reused_allocation_fits": 4083,
                "reference_rebuilt": False, "new_reference_accounts": 0, "reused_reference_accounts": 3,
                "new_full_accounts": 7, "account_computations": 8,
                "account_recomputation": "所有8个账户在新目录完成，包括已有的1个等权基础账户重复校验；旧中断和旧文件保留。"}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update(ROOT / name for name in read(parent.OUT / "freeze.json")["sources"])
    sources.update(p for p in parent.OUT.rglob("*") if p.is_file())
    save(root / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
                                "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)},
                                "copies": {p.relative_to(root).as_posix(): digest(p) for folder in ["inputs", "references"] for p in (root / folder).rglob("*") if p.is_file()}}, True)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(mandate_path, mandate)
    print("机器精度边界修正已固定，复用已保存分配，不新增策略拟合。", flush=True)


def verify_sources(root):
    record = read(root / "freeze.json")
    assert digest(Path(__file__)) == record["code_sha256"]
    assert digest(root / "protocol.json") == record["protocol_sha256"]
    for name, expected in record["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("原始固定来源变化：" + name)
    for name, expected in record["copies"].items():
        assert digest(root / name) == expected


def saved_references(root, data, dividends):
    labels = pd.read_parquet(root / "inputs/expert_value_labels.parquet")
    targets = pd.read_parquet(root / "inputs/expert_current_targets.parquet")
    for name, frame in [("expert_value_labels", labels), ("expert_current_targets", targets)]:
        frame.to_parquet(root / "results" / (name + ".parquet"), index=False)
    return labels.set_index("idx", drop=False), targets.set_index("idx", drop=False)


def saved_allocations(root, labels, targets):
    frame = pd.read_parquet(root / "inputs/allocation_predictions.parquet")
    assert frame.raw_target.between(-1e-12, 1 + 1e-12).all()
    original = frame.raw_target.copy()
    frame["raw_target"] = frame.raw_target.clip(0, 1)
    changed = frame.loc[frame.raw_target.ne(original), ["idx", "date", "model", "raw_target"]].copy()
    changed["original_raw_target"] = original.loc[changed.index]
    assert len(changed) == 1
    np.testing.assert_array_equal(frame.raw_target.loc[frame.raw_target.eq(original)], original.loc[frame.raw_target.eq(original)])
    records = read(root / "inputs/saved_allocations.json")
    save(root / "results/numerical_correction.json", {"tolerance": 1e-12, "changes": changed.to_dict("records"),
         "unchanged_economic_decisions": True, "new_fits": 0}, True)
    frame.to_parquet(root / "results/allocation_predictions.parquet", index=False)
    save(root / "results/saved_allocations.json", records, True)
    return frame, records


def run(root):
    def save_completed(path, value, exclusive=False):
        if path == root / "result.json":
            value = {**value, "new_reference_accounts": 0, "reused_reference_accounts": 3,
                     "new_full_accounts": 7, "account_computations": 8,
                     "new_allocation_fits": 0, "reused_allocation_fits": value["daily_allocation_fits"],
                     "daily_allocation_fits": 0}
        save(path, value, exclusive)

    # 复用原固定账户及验证流程，只替换来源读取；不修改原函数、策略或冻结文件。
    namespace = dict(parent.run.__globals__)
    namespace.update(verify_sources=verify_sources, references=saved_references,
                     learn_allocations=saved_allocations, save=save_completed, STUDY=STUDY)
    completion = FunctionType(parent.run.__code__, namespace, parent.run.__name__)
    completion(root)
    baseline = pd.read_parquet(parent.OUT / "accounts/BASE/EQUAL_EXPERTS_ledger.parquet")
    recomputed = pd.read_parquet(root / "accounts/BASE/EQUAL_EXPERTS_ledger.parquet")
    pd.testing.assert_frame_equal(baseline, recomputed)
    save(root / "completion_receipt.json", {"at": now(), "status": "COMPLETED_SAVED_ALLOCATION_NO_REFIT",
        "new_hidden_model_fits": 0, "new_allocation_fits": 0, "reused_allocation_fits": 4083,
        "new_reference_accounts": 0, "reused_reference_accounts": 3, "account_computations": 8,
        "previously_completed_account_exactly_reproduced": True,
        "new_unique_full_accounts": 7, "goal_achieved": False}, True)
    print("全部固定账户完成，原已完成等权账户逐列重现，只有机器精度截断。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="完成已固定状态策略分配的数值边界修正")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "completion_receipt.json") if (args.out / "completion_receipt.json").exists() else "尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
