"""等待当前V2进程结束后，一次性完成已授权来源重算和本地交付。"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def now():
    return datetime.now().astimezone().isoformat()


def check_prepared_code():
    prepared = read(OUT / "prepared_delivery_pipeline.json")
    assert prepared["status"] == "PREPARED_NOT_EXECUTED"
    for record in prepared["files"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], record["path"]
    for file in ["parser_semantic_invalidation_addendum.json", "replay_started.json", "round_status.json", "delivery_receipt.json"]:
        assert not (OUT / file).exists(), "存在已开始工作或新反证，不允许重复启动：" + file


def wait_for_existing_batch(pid):
    assert os.name == "nt", "此完成流程使用Windows原生进程句柄"
    start = read(OUT / "batch_started.json")
    assert start["pid"] == pid
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        assert error == 87 and (OUT / "batch_complete.json").exists(), ("原批次句柄不可用且没有成功终态", error)
        return {"status": "ALREADY_EXITED_WITH_COMPLETION_RECORD", "pid": pid}
    try:
        created, ended, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        assert kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(ended), ctypes.byref(kernel_time), ctypes.byref(user_time))
        birth = (((created.dwHighDateTime << 32) | created.dwLowDateTime) - 116444736000000000) / 10000000
        batch_record_time = datetime.fromisoformat(start["at"]).timestamp()
        assert 0 <= batch_record_time - birth <= 120, "进程创建时刻不匹配，防止PID复用"
        save("pipeline_wait_handle_receipt.json", {"at": now(), "status": "LIVE_BATCH_HANDLE_CONFIRMED", "pid": pid,
            "creation_time_unix": birth, "batch_started_at": start["at"], "waiter_pid": os.getpid()})
        print(f"已取得批次进程{pid}的真实句柄，等待其结束；不重新启动或重发请求。", flush=True)
        while True:
            signal = kernel.WaitForSingleObject(handle, 30000)
            if signal == 0:
                break
            assert signal == 258, ("等待进程返回异常", signal)
        code = wintypes.DWORD()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        assert code.value == 0, ("V2来源批次未正常完成，停止后续流程", code.value)
        assert (OUT / "batch_complete.json").exists()
        return {"status": "BATCH_EXITED_SUCCESSFULLY", "pid": pid, "exit_code": code.value}
    finally:
        kernel.CloseHandle(handle)


def execute_stage(name, args):
    command = [sys.executable, "-X", "utf8", *args]
    print(f"开始阶段：{name}。", flush=True)
    with (OUT / ("pipeline_" + name + ".log")).open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
        save("pipeline_" + name + "_started.json", {"at": now(), "stage": name, "pid": process.pid, "command": command})
        code = process.wait()
    assert code == 0, (name, code, str(OUT / ("pipeline_" + name + ".log")))
    save("pipeline_" + name + "_complete.json", {"at": now(), "stage": name, "exit_code": code})
    print(f"阶段完成：{name}。", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-pid", type=int, required=True)
    args = parser.parse_args()
    assert not (OUT / "pipeline_started.json").exists()
    check_prepared_code()
    save("pipeline_started.json", {"at": now(), "pid": os.getpid(), "batch_pid": args.batch_pid,
        "status": "WAITING_EXISTING_V2_BATCH", "new_network_requests_from_pipeline": 0,
        "new_accounts": 0, "goal_achieved": False, "orders_authorized": False})
    stage = "wait"
    try:
        ending = wait_for_existing_batch(args.batch_pid)
        save("pipeline_batch_ended.json", {"at": now(), **ending})
        check_prepared_code()
        stage = "replay"
        execute_stage(stage, ["-m", "research.factor96_financial_parser_replay_v2", "replay"])
        assert read(OUT / "source_repair_result.json")["all_confirmed_regressions_passed"], "已确认原文反例尚未全部修正"
        assert not (OUT / "parser_semantic_invalidation_addendum.json").exists()
        stage = "report"
        execute_stage(stage, ["scripts/report_factor96_financial_parser_scope_v2.py"])
        stage = "package"
        execute_stage(stage, ["scripts/package_factor96_financial_parser_scope_v2.py"])
        receipt = read(OUT / "delivery_receipt.json")
        assert receipt["saved_output_recomputation"]["status"] == "PASS_SAVED_V2_SCOPE_REPAIR_AND_UNCHANGED_MEASUREMENT"
        save("pipeline_complete.json", {"at": now(), "status": "SOURCE_REPLAY_AND_LOCAL_DELIVERY_COMPLETE",
            "zip_path": receipt["zip_path"], "zip_sha256": receipt["sha256"], "T11": "NOT_RUN", "new_accounts": 0,
            "goal_achieved": False, "external_review": "NOT_PERFORMED", "orders_authorized": False})
        print("来源重算和本地交付已完成；T11账户仍未运行，夏普目标仍未达成。", flush=True)
    except Exception as error:
        save("pipeline_failed.json", {"at": now(), "stage": stage, "error_type": type(error).__name__, "error": str(error),
            "status": "FAILED_PRESERVE_ALL_ARTIFACTS_DO_NOT_RESTART_SOURCE_BATCH", "new_accounts": 0, "goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
