"""等待正在运行的第十九轮结束，接续已固定的PE关系三方法对照。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity
from scripts.continue_forward_eps_two_institution_pipeline_20260907 import ExistingProcess

OUT = ROOT / "reports/research/510300_forward_eps_valuation_consistency_pipeline_v1"
DEPENDENCY = ROOT / "reports/research/510300_forward_eps_two_institution_pipeline_v1"
DEPENDENCY_PID = 3920
DEPENDENCY_CREATED = "2026-09-07T02:29:18.999112+08:00"
READY = [DEPENDENCY / "result.json", ROOT / "reports/research/510300_forward_eps_two_institution_policy_v2/saved_numerical_verification.json"]
STAGES = [
    ("完整两机构报告PE关系条件", "research/forward_eps_valuation_consistency_features_v1.py", ["--run"],
     "reports/research/510300_forward_eps_valuation_consistency_features_v1/result.json"),
    ("第二十轮三方法完整进出场账户", "research/forward_eps_valuation_consistency_policy_v1.py", ["--run"],
     "reports/research/510300_forward_eps_valuation_consistency_policy_v1/result.json"),
    ("保存原件关系模型与进出场核对", "scripts/verify_forward_eps_valuation_consistency_saved_20260907.py", [],
     "reports/research/510300_forward_eps_valuation_consistency_policy_v1/saved_numerical_verification.json"),
]


def status(stage: str, **fields):
    record = {"observed_at": now(), "stage": stage, "pipeline_pid": os.getpid(),
              "goal_achieved": False, "prepare_gpt_numerical_review_package": False, **fields}
    temporary = OUT / "status.next.json"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT / "status.json")


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *[ROOT / stage[1] for stage in STAGES],
             ROOT / "scripts/continue_forward_eps_two_institution_pipeline_20260907.py", DEPENDENCY / "manifest.json",
             ROOT / "reports/research/510300_forward_eps_valuation_consistency_features_v1/manifest.json",
             ROOT / "config/510300_forward_eps_valuation_consistency_policy_v1_manifest.json"]
    save(OUT / "manifest.json", {"registered_at": now(), "dependency_exec_session": 2240,
         "dependency_pid": DEPENDENCY_PID, "dependency_creation_time": DEPENDENCY_CREATED,
         "stages": STAGES, "restart_source_or_round_19": False,
         "files": [identity(p) for p in paths]}, exclusive=True)
    print("第二十轮顺序接续已固定，等待原第十九轮真实完成及保存核对。", flush=True)


def run():
    for item in read(OUT / "manifest.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "第二十轮接续登记改变"
    save(OUT / "RUN_STARTED.json", {"started_at": now(), "pid": os.getpid()}, exclusive=True)
    existing = None
    current_stage = "等待原第十九轮完整账户和保存核对"
    try:
        if not all(p.exists() for p in READY):
            existing = ExistingProcess(DEPENDENCY_PID, DEPENDENCY_CREATED)
            while not all(p.exists() for p in READY):
                if (DEPENDENCY / "failure.json").exists():
                    raise RuntimeError("原第十九轮已记录失败，先处理原失败，不自动重跑")
                if not existing.running():
                    raise RuntimeError("原第十九轮进程已结束但完成文件不足，须核对原因")
                status(current_stage, dependency_process_confirmed_running=True, dependency_pid=DEPENDENCY_PID,
                       dependency_observation=read(DEPENDENCY / "status.json"))
                time.sleep(20)
        if existing:
            existing.close()
        assert read(READY[0])["status"] == "REGISTERED_SOURCE_AND_ACCOUNT_PIPELINE_FINISHED"
        assert read(READY[1])["status"] == "PASS_SAVED_TWO_INSTITUTION_FEATURES_MODELS_ORDERS_AND_ACCOUNTS"
        save(OUT / "dependency_complete_receipt.json", {"ready_at": now(), "files": [identity(p) for p in READY]}, exclusive=True)
        completed = []
        for number, (title, script, arguments, expected) in enumerate(STAGES, 1):
            current_stage = title
            if (ROOT / expected).exists():
                raise FileExistsError("尚未由此接续运行但结果已经存在：" + expected)
            log_path = OUT / f"{number:02d}_运行日志.txt"
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(ROOT / script), *arguments], cwd=ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"}, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                print("第二十轮接续开始：", title, "子进程", process.pid, flush=True)
                while process.poll() is None:
                    status(title, child_pid=process.pid, child_confirmed_running=True, stage_number=number,
                           log_path=log_path.relative_to(ROOT).as_posix())
                    time.sleep(10)
            if process.returncode != 0 or not (ROOT / expected).exists():
                raise RuntimeError(f"第二十轮阶段未完成：{title}，退出码{process.returncode}，日志{log_path}")
            completed.append({"stage": title, "completed_at": now(), "result": identity(ROOT / expected)})
            save(OUT / f"{number:02d}_完成凭证.json", completed[-1], exclusive=True)
        save(OUT / "result.json", {"completed_at": now(), "status": "VALUATION_RELATION_ACCOUNT_PIPELINE_FINISHED",
             "completed_stages": completed, "new_strategies": 3, "new_accounts_generated": 6,
             "evaluation_accounts": 10, "user_delivery_pending": True, "goal_achieved": False,
             "prepare_gpt_numerical_review_package": False}, exclusive=True)
        status("第二十轮完成并核对，等待解释真实结果及更新交付", child_confirmed_running=False)
    except Exception as exc:
        save(OUT / "failure.json", {"failed_at": now(), "stage": current_stage, "error_type": type(exc).__name__,
             "error": str(exc), "traceback": traceback.format_exc(), "dependency_or_source_restarted": False,
             "goal_achieved": False}, exclusive=True)
        status("第二十轮接续失败，保留结果等待处理", error=str(exc), failed_stage=current_stage)
        raise
    finally:
        if existing:
            existing.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
