"""等待现有归档结束，顺序执行已登记的来源修正与第十九轮账户。"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
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

OUT = ROOT / "reports/research/510300_forward_eps_two_institution_pipeline_v1"
ORIGINALS = ROOT / "reports/research/510300_forward_eps_soochow_originals_v1"
OLD_FACTS = ROOT / "reports/research/510300_forward_eps_soochow_facts_v1"
SOURCE_PARENT_PID = 18952
SOURCE_PARENT_CREATION = "2026-09-07T01:42:44.015979+08:00"
STAGES = [
    ("完整东吴第三版EPS", "research/forward_eps_soochow_facts_v3.py", ["--run"], "reports/research/510300_forward_eps_soochow_facts_v3/result.json"),
    ("完整EPS原行与时钟核对", "research/verify_forward_eps_soochow_facts_v3.py", [], "reports/research/510300_forward_eps_soochow_facts_v3/saved_source_verification.json"),
    ("两机构公司等权因子", "research/forward_eps_two_institution_features_v2.py", ["--run"], "reports/research/510300_forward_eps_two_institution_features_v2/result.json"),
    ("第十九轮三方法完整账户", "research/forward_eps_two_institution_policy_v2.py", ["--run"], "reports/research/510300_forward_eps_two_institution_policy_v2/result.json"),
    ("保存因子模型进出场账户核对", "scripts/verify_forward_eps_two_institution_saved_20260907.py", [], "reports/research/510300_forward_eps_two_institution_policy_v2/saved_numerical_verification.json"),
]


class ExistingProcess:
    """持有原PowerShell的进程句柄，避免把重用的进程号误判为旧进程。"""
    def __init__(self, pid: int, expected_creation: str):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.WaitForSingleObject.restype = wintypes.DWORD
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not self.handle:
            raise RuntimeError("现有采集父进程已不存在或无法读取")
        fields = [wintypes.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle, *(ctypes.byref(x) for x in fields)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())
        ticks = (fields[0].dwHighDateTime << 32) | fields[0].dwLowDateTime
        created = datetime.fromtimestamp(ticks / 10000000 - 11644473600, timezone.utc)
        if abs((created - datetime.fromisoformat(expected_creation)).total_seconds()) > 0.01:
            self.close()
            raise RuntimeError("采集父进程号已被其他进程重用")

    def running(self) -> bool:
        state = self.kernel.WaitForSingleObject(self.handle, 0)
        if state not in [0, 258]:
            raise RuntimeError("无法读取采集父进程运行状态")
        return state == 258

    def close(self):
        if getattr(self, "handle", None):
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def status(stage: str, **fields):
    value = {"observed_at": now(), "stage": stage, "pipeline_pid": os.getpid(),
             "goal_achieved": False, "prepare_gpt_numerical_review_package": False, **fields}
    temporary = OUT / "status.next.json"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(OUT / "status.json")


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *[ROOT / stage[1] for stage in STAGES],
             ROOT / "config/510300_forward_eps_two_institution_policy_v2_manifest.json",
             ROOT / "reports/research/510300_forward_eps_two_institution_features_v2/manifest.json",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v3/manifest.json",
             ORIGINALS / "manifest.json", OLD_FACTS / "manifest.json"]
    save(OUT / "manifest.json", {"registered_at": now(), "source_exec_session": 47679,
         "source_parent_pid": SOURCE_PARENT_PID, "source_parent_creation": SOURCE_PARENT_CREATION,
         "restart_archive": False, "stages": STAGES, "source_process_already_running": True,
         "files": [identity(p) for p in paths]}, exclusive=True)
    print("现有归档的顺序接续已登记，不新建采集。", flush=True)


def run():
    manifest = read(OUT / "manifest.json")
    for item in manifest["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "接续登记文件改变"
    save(OUT / "RUN_STARTED.json", {"started_at": now(), "pid": os.getpid(),
         "manifest": identity(OUT / "manifest.json")}, exclusive=True)
    observed_process = None
    current_stage = "等待原采集及其自动第一版提取"
    try:
        if not all((folder / "result.json").exists() for folder in [ORIGINALS, OLD_FACTS]):
            observed_process = ExistingProcess(SOURCE_PARENT_PID, SOURCE_PARENT_CREATION)
            while not all((folder / "result.json").exists() for folder in [ORIGINALS, OLD_FACTS]):
                if not observed_process.running():
                    raise RuntimeError("原采集父进程已结束，但原件或第一版结果尚未完成；须检查原因，不自动重启")
                status(current_stage, dependency_process_confirmed_running=True,
                       source_parent_pid=SOURCE_PARENT_PID,
                       archived_receipt_files=sum(1 for _ in (ORIGINALS / "document_records").glob("*.json")),
                       original_result_exists=(ORIGINALS / "result.json").exists(),
                       old_fact_result_exists=(OLD_FACTS / "result.json").exists())
                time.sleep(20)
        if observed_process:
            observed_process.close()
        original_result = read(ORIGINALS / "result.json")
        if original_result["access_restriction_stop"]:
            raise RuntimeError("原来源存在访问限制，保留原进度并等待处理")
        if original_result["selected_reports"] != 6734:
            raise RuntimeError("固定原件队列数量变化")
        save(OUT / "completed_original_dependency_receipt.json", {"ready_at": now(),
             "files": [identity(folder / "result.json") for folder in [ORIGINALS, OLD_FACTS]]}, exclusive=True)
        completed = []
        for sequence, (title, script, arguments, expected) in enumerate(STAGES, 1):
            current_stage = title
            if (ROOT / expected).exists():
                raise FileExistsError("本接续尚未运行但阶段结果已存在，请核对其他执行者：" + expected)
            log_path = OUT / f"{sequence:02d}_运行日志.txt"
            environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(ROOT / script), *arguments], cwd=ROOT,
                                           env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                print("自动接续开始：", title, "子进程", process.pid, flush=True)
                while process.poll() is None:
                    status(title, child_pid=process.pid, child_confirmed_running=True, stage_number=sequence,
                           log_path=log_path.relative_to(ROOT).as_posix())
                    time.sleep(10)
            if process.returncode != 0 or not (ROOT / expected).exists():
                raise RuntimeError(f"阶段未成功完成：{title}，退出码{process.returncode}，请查看{log_path}")
            completed.append({"stage": title, "result": identity(ROOT / expected), "completed_at": now()})
            save(OUT / f"{sequence:02d}_完成凭证.json", completed[-1], exclusive=True)
            print("自动接续完成：", title, flush=True)
        save(OUT / "result.json", {"completed_at": now(), "status": "REGISTERED_SOURCE_AND_ACCOUNT_PIPELINE_FINISHED",
             "completed_stages": completed, "new_strategies": 3, "new_accounts_generated": 6,
             "evaluation_accounts": 10, "goal_achieved": False, "user_delivery_pending": True,
             "prepare_gpt_numerical_review_package": False}, exclusive=True)
        status("已完成第十九轮和保存核对，等待解释结果及更新交付", child_confirmed_running=False)
    except Exception as exc:
        save(OUT / "failure.json", {"failed_at": now(), "stage": current_stage,
             "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
             "archive_restarted": False, "goal_achieved": False}, exclusive=True)
        status("接续阶段失败，保留全部结果等待处理", failed_stage=current_stage, error=str(exc))
        raise
    finally:
        if observed_process:
            observed_process.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
