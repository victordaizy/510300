"""在独立目录补完已冻结的V3；只改输出位置与内部数据格式。"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_pattern_daily_state_learning_v3"
OUT = ROOT / "reports/research/510300_pattern_daily_state_learning_v3_saved_completion"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_internal_data(path: Path, value) -> None:
    """用户不需要展示表格，原脚本的数据记录保存为内部Parquet。"""
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    destination = path.with_suffix(".parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(destination), frame.reset_index(drop=True), check_dtype=False)
    else:
        frame.to_parquet(destination, index=False)


def main() -> None:
    if OUT.exists():
        raise RuntimeError("独立补完目录已经存在；不得覆盖或重新训练。")
    if (SOURCE / "summary.json").exists() or (SOURCE / "RUN_STARTED.json").exists():
        raise RuntimeError("原V3已经开始或完成；应先读取原结果，避免重复运行。")
    frozen = json.loads((SOURCE / "freeze.json").read_text(encoding="utf-8-sig"))
    for relative, expected in frozen["hashes"].items():
        if digest(SOURCE / relative) != expected:
            raise RuntimeError(f"原冻结输入发生变化：{relative}")
    OUT.mkdir(parents=True)
    for relative in [*frozen["hashes"], "freeze.json"]:
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, target)
    shutil.copy2(Path(__file__), OUT / "code/completion_launcher.py")
    receipt = {
        "started_at": timestamp(),
        "study_id": "510300_PATTERN_DAILY_STATE_LEARNING_V3",
        "purpose": "补完已有冻结实验，不登记为新的独立研究假设。",
        "original_directory": SOURCE.relative_to(ROOT).as_posix(),
        "output_directory": OUT.relative_to(ROOT).as_posix(),
        "original_freeze_sha256": digest(SOURCE / "freeze.json"),
        "launcher_sha256": digest(Path(__file__)),
        "model_or_account_logic_changed": False,
        "output_changes": ["输出目录隔离", "原CSV数据写入Parquet", "终端比较输出保存为内部执行日志"],
        "new_data_downloads": 0,
        "original_files_modified": False,
        "review_package_created": False,
        "orders_authorized": False,
    }
    save_json(OUT / "RUN_STARTED.json", receipt)
    sys.path.insert(0, str(OUT / "code"))
    source_script = OUT / "code/run_pattern_daily_state_learning_v3.py"
    specification = importlib.util.spec_from_file_location("frozen_state_v3_completion", source_script)
    if specification is None or specification.loader is None:
        raise RuntimeError("无法加载原冻结脚本。")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.OUT = OUT
    module.save_csv = save_internal_data
    print("正在按冻结参数补完普通日状态学习V3。", flush=True)
    with (OUT / "execution.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            module.run()
    receipt["completed_at"] = timestamp()
    receipt["summary_sha256"] = digest(OUT / "summary.json")
    receipt["all_original_source_hashes_unchanged"] = all(
        digest(SOURCE / relative) == expected for relative, expected in frozen["hashes"].items()
    )
    save_json(OUT / "execution_receipt.json", receipt)
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({key: summary[key] for key in ["status", "new_fitted_models", "refit_dates", "new_accounts", "shape_confirmations", "numeric_phase_account_pass"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
