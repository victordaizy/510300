"""续算中断的V2纯本地解析；已保存结果不覆盖，HTTP来源不再请求。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil

from research import factor96_financial_parser_batch_v2 as batch
from research.factor96_financial_row_parser_v2 import PARSER_VERSION


ROOT, OUT = batch.ROOT, batch.OUT
RECOVERY = OUT / "execution_recovery_01"
read, save, now, digest = batch.read, batch.save, batch.now, batch.digest


def process_probe(pid):
    assert os.name == "nt"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        assert error == 87, ("无法确认原进程是否已退出", pid, error)
        return {"pid": pid, "state": "PROCESS_NOT_FOUND", "win32_error": error}
    try:
        code = wintypes.DWORD()
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        return {"pid": pid, "state": "STILL_ACTIVE" if code.value == 259 else "EXITED", "exit_code": code.value}
    finally:
        kernel.CloseHandle(handle)


def plan_missing(targets, out):
    expected = {target["sha256"] for target in targets}
    assert len(expected) == len(targets)
    assert {p.stem for p in (out / "fetch_receipts").glob("*.json")} == expected
    assert {p.stem for p in (out / "parsed_documents").glob("*.json")}.issubset(expected)
    completed, missing, unknown = [], [], []
    for target in targets:
        checksum = target["sha256"]
        source = read(out / "fetch_receipts" / (checksum + ".json"))
        assert source["expected_sha256"] == checksum
        assert str(source["announcement_id"]) == str(target["announcement_id"])
        path = out / "parsed_documents" / (checksum + ".json")
        if source["status"] not in batch.ACCEPTED:
            assert not path.exists(), "失败来源不得出现来源不明的解析结果"
            unknown.append(checksum)
            continue
        if not path.exists():
            missing.append(target)
            continue
        result = read(path)
        assert result["official_pdf_sha256"] == checksum
        assert str(result["announcement_id"]) == str(target["announcement_id"])
        assert isinstance(result["metrics"], list)
        assert result["status"] in {"PARSED", "NO_VIEW_PARSE_ERROR"}
        if result["status"] == "PARSED":
            assert result["parser_version"] == PARSER_VERSION
        else:
            assert not result["metrics"]
        completed.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size,
                          "sha256": digest(path), "source_pdf_sha256": checksum, "status": result["status"]})
    return {"completed": completed, "missing_targets": missing, "unknown_source_hashes": unknown}


def verify_frozen_code():
    for item in read(OUT / "batch_freeze.json")["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    for name in ["factor96_financial_parser_batch_v2.py", "factor96_financial_row_parser_v2.py"]:
        assert digest(ROOT / "research" / name) == digest(OUT / "code" / name)
    assert not (OUT / "parser_semantic_invalidation_addendum.json").exists()


def prepare():
    assert not RECOVERY.exists() and not (OUT / "batch_complete.json").exists()
    assert not (OUT / "replay_started.json").exists()
    probes = [process_probe(15084), process_probe(18352)]
    assert all(p["state"] in {"PROCESS_NOT_FOUND", "EXITED"} for p in probes)
    verify_frozen_code()
    assert (OUT / "download_complete.json").exists()
    targets = read(OUT / "batch_targets.json")
    plan = plan_missing(targets, OUT)
    assert len(targets) == 2112 and len(plan["completed"]) + len(plan["missing_targets"]) == 2085
    assert len(plan["unknown_source_hashes"]) == 27
    RECOVERY.mkdir()
    sources = [OUT / "download_complete.json", OUT / "batch_started.json", OUT / "batch_targets.json",
               *(OUT / "fetch_attempts").glob("*.json"), *(OUT / "fetch_receipts").glob("*.json")]
    source_manifest = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(sources)]
    save(RECOVERY / "source_io_before.json", {"files": source_manifest})
    save(RECOVERY / "resume_plan.json", plan)
    save(RECOVERY / "protocol.json", {"at": now(), "status": "FROZEN_EXECUTION_CONTINUATION_NO_SEMANTIC_CHANGE",
        "original_process_probes": probes, "original_exec_sessions": [2029, 36996],
        "observed_interruption": "原会话不可用、两个原进程均已不存在；没有batch_complete或pipeline失败回执。",
        "cause": "NOT_ESTABLISHED；系统启动时间未改变，近期应用错误事件未发现匹配项。",
        "completed_results_preserved": len(plan["completed"]), "missing_results_to_compute": len(plan["missing_targets"]),
        "unknown_sources_unchanged": len(plan["unknown_source_hashes"]),
        "method": "仅调用原冻结batch.parse_document；保留现有1839份结果字节，仅为不存在结果的246份原文续算。",
        "network": "不调用fetch_document，不重发HTTP；所有请求前记录、来源回执及download_complete必须前后字节一致。",
        "partial_result_rule": "已有结果若无法解析、身份不符或版本不符则停止，不覆盖或重试。",
        "workers": 4, "new_accounts": 0, "returns_read": False, "goal_achieved": False, "orders_authorized": False})
    code = [Path(__file__), ROOT / "scripts/resume_and_finish_factor96_financial_scope_v2_01.py",
            ROOT / "tests/test_factor96_financial_parser_resume_v2_01.py", ROOT / "scripts/finish_factor96_financial_scope_v2_when_ready.py"]
    for path in code:
        shutil.copy2(path, RECOVERY / path.name)
    frozen = [{"path": p.relative_to(RECOVERY).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
              for p in sorted(RECOVERY.iterdir()) if p.is_file()]
    save(RECOVERY / "freeze.json", {"at": now(), "files": frozen})
    print(json.dumps({"已保存结果": len(plan["completed"]), "待续算": len(plan["missing_targets"]), "新HTTP请求": 0}, ensure_ascii=False), flush=True)


def parse_missing(target):
    assert not (batch.OUT / "parsed_documents" / (target["sha256"] + ".json")).exists(), "已有结果禁止重写"
    return batch.parse_document(target)


def run():
    assert not (RECOVERY / "resume_started.json").exists() and not (OUT / "batch_complete.json").exists()
    for item in read(RECOVERY / "freeze.json")["files"]:
        assert digest(RECOVERY / item["path"]) == item["sha256"]
    assert digest(Path(__file__)) == digest(RECOVERY / Path(__file__).name)
    verify_frozen_code()
    plan = read(RECOVERY / "resume_plan.json")
    current = plan_missing(read(OUT / "batch_targets.json"), OUT)
    assert current == plan, "准备后结果发生变化，不能重复续算"
    save(RECOVERY / "resume_started.json", {"at": now(), "pid": os.getpid(), "pending": len(plan["missing_targets"]), "new_network_requests": 0})
    finished = 0
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(parse_missing, target) for target in plan["missing_targets"]]
        for future in as_completed(futures):
            future.result()
            finished += 1
            if finished % 20 == 0 or finished == len(futures):
                print(f"V2中断后本地续算完成{finished}/{len(futures)}，累计{len(plan['completed']) + finished}/2085。", flush=True)
    for item in plan["completed"] + read(RECOVERY / "source_io_before.json")["files"]:
        path = OUT / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], item["path"]
    targets = read(OUT / "batch_targets.json")
    after = plan_missing(targets, OUT)
    assert not after["missing_targets"] and len(after["completed"]) == 2085
    parsed = {p.stem: read(p) for p in (OUT / "parsed_documents").glob("*.json")}
    recovered = sum(len(set(t["target_metrics"]) & {m["metric_id"] for m in parsed[t["sha256"]]["metrics"]})
                    for t in targets if t["sha256"] in parsed)
    save(RECOVERY / "resume_complete.json", {"at": now(), "newly_parsed": finished,
        "previous_results_byte_identical": len(plan["completed"]), "source_io_files_byte_identical": len(read(RECOVERY / "source_io_before.json")["files"]),
        "new_network_requests": 0, "new_accounts": 0})
    save(OUT / "batch_complete.json", {"at": now(), "documents": len(targets), "download_receipts": len(targets),
        "parsed_documents": len(parsed), "parsed_status_counts": dict(Counter(r["status"] for r in parsed.values())),
        "recovered_target_fields": recovered, "execution_recovery": "execution_recovery_01/resume_complete.json",
        "new_accounts": 0, "returns_read": False})
    print("V2全部原文解析已齐；已完成结果及来源回执均保持原字节。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
