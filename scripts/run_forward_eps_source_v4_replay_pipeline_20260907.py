"""顺序运行修正来源的六个既有方法，并核对真实账户与进出场。"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity

OUT = ROOT / "reports/research/510300_forward_eps_source_v4_replay_pipeline_v1"
STAGES = [
    ("补齐字段后的两机构公司因子", "research/forward_eps_two_institution_features_v3.py", ["--run"], "510300_forward_eps_two_institution_features_v3/result.json"),
    ("原十九轮三方法来源重放", "research/forward_eps_two_institution_policy_v3.py", ["--run"], "510300_forward_eps_two_institution_policy_v3/result.json"),
    ("两机构模型进出场与完整账户核对", "scripts/verify_forward_eps_two_institution_source_v4_saved_20260907.py", [], "510300_forward_eps_two_institution_policy_v3/saved_numerical_verification.json"),
    ("补齐字段后的报告PE关系条件", "research/forward_eps_valuation_consistency_features_v2.py", ["--run"], "510300_forward_eps_valuation_consistency_features_v2/result.json"),
    ("原二十轮三方法来源重放", "research/forward_eps_valuation_consistency_policy_v2.py", ["--run"], "510300_forward_eps_valuation_consistency_policy_v2/result.json"),
    ("PE模型进出场与完整账户核对", "scripts/verify_forward_eps_valuation_consistency_source_v4_saved_20260907.py", [], "510300_forward_eps_valuation_consistency_policy_v2/saved_numerical_verification.json"),
]


def status(stage, **fields):
    value = {"observed_at": now(), "stage": stage, "pipeline_pid": os.getpid(),
             "goal_achieved": False, "prepare_gpt_numerical_review_package": False, **fields}
    temp = OUT / "status.next.json"
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(OUT / "status.json")


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT / "research/forward_eps_source_v4_replay_registration.py",
             ROOT / "docs/510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY.md",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v4/result.json",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v4/saved_source_verification.json",
             ROOT / "config/510300_forward_eps_two_institution_policy_v3_manifest.json",
             ROOT / "config/510300_forward_eps_valuation_consistency_policy_v2_manifest.json",
             ROOT / "reports/research/510300_forward_eps_two_institution_features_v3/manifest.json",
             ROOT / "reports/research/510300_forward_eps_valuation_consistency_features_v2/manifest.json",
             *[ROOT / stage[1] for stage in STAGES]]
    save(OUT / "manifest.json", {"registered_at": now(), "stages": STAGES,
         "new_distinct_methods": 0, "existing_methods_with_new_sources": 6,
         "new_accounts": 12, "evaluation_accounts": 20,
         "files": [identity(p) for p in files]}, exclusive=True)
    print("六种原方法的来源重放接续已登记。", flush=True)


def run():
    for item in read(OUT / "manifest.json")["files"]:
        assert identity(ROOT / item["path"])["sha256"] == item["sha256"], "来源重放登记文件改变"
    save(OUT / "RUN_STARTED.json", {"started_at": now(), "pid": os.getpid(),
         "manifest": identity(OUT / "manifest.json")}, exclusive=True)
    completed, current = [], "尚未开始"
    try:
        for sequence, (title, script, arguments, expected) in enumerate(STAGES, 1):
            current = title
            expected_path = ROOT / "reports/research" / expected
            if expected_path.exists():
                raise FileExistsError("阶段结果已经存在，不重复执行：" + expected)
            log_path = OUT / f"{sequence:02d}_运行日志.txt"
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(ROOT / script), *arguments], cwd=ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"}, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                print("来源重放开始：", title, "子进程", process.pid, flush=True)
                while process.poll() is None:
                    status(title, child_pid=process.pid, child_confirmed_running=True,
                           stage_number=sequence, log_path=log_path.relative_to(ROOT).as_posix())
                    time.sleep(10)
            if process.returncode != 0 or not expected_path.exists():
                raise RuntimeError(f"阶段失败：{title}，退出码{process.returncode}，日志{log_path}")
            item = {"stage": title, "completed_at": now(), "result": identity(expected_path)}
            save(OUT / f"{sequence:02d}_完成凭证.json", item, exclusive=True)
            completed.append(item)
        save(OUT / "result.json", {"study_id": "510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY",
            "completed_at": now(), "status": "SIX_EXISTING_METHOD_SOURCE_REPLAYS_AND_VERIFICATIONS_COMPLETE",
            "completed_stages": completed, "new_distinct_methods": 0, "new_source_versions": 6,
            "new_accounts_generated": 12, "evaluation_accounts": 20, "goal_achieved": False,
            "user_delivery_pending": True, "prepare_gpt_numerical_review_package": False}, exclusive=True)
        status("六个来源重放及二十评价账户已完成，等待比较和中文交付", child_confirmed_running=False)
    except Exception as exc:
        save(OUT / "failure.json", {"failed_at": now(), "stage": current, "error": str(exc),
             "traceback": traceback.format_exc(), "goal_achieved": False}, exclusive=True)
        status("来源重放失败，保留已完成阶段", failed_stage=current, error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    freeze() if args.freeze else run()
