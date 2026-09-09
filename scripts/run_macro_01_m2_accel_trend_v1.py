"""在冻结后执行MACRO-01唯一一次历史发现回测。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from research.daily_macro_01_m2_accel_trend_v1 import (
    ROOT,
    evaluate_protocol,
    json_default,
    load_config,
    markdown_report,
    sha256_file,
)


MANIFEST = ROOT / "config" / "macro_01_m2_accel_trend_v1_manifest.json"
FIX_MANIFEST = ROOT / "config" / "macro_01_m2_accel_trend_v1_execution_fix_manifest.json"
RUN_SCRIPT_RELATIVE = "scripts/run_macro_01_m2_accel_trend_v1.py"


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_verify_execution_fix(manifest: dict) -> dict:
    if not FIX_MANIFEST.exists():
        raise FileNotFoundError("缺少执行路径修正清单")
    fix = json.loads(FIX_MANIFEST.read_text(encoding="utf-8"))
    recorded_hash = fix.get("manifest_content_sha256")
    payload = dict(fix)
    payload.pop("manifest_content_sha256", None)
    if recorded_hash != canonical_hash(payload):
        raise ValueError("执行路径修正清单内容哈希不匹配")
    if fix.get("original_manifest_content_sha256") != manifest.get("manifest_content_sha256"):
        raise ValueError("执行路径修正清单未绑定当前原始冻结清单")
    if fix.get("old_run_script_sha256") != manifest["frozen_file_sha256"][RUN_SCRIPT_RELATIVE]:
        raise ValueError("执行路径修正清单的旧脚本哈希不匹配")
    if fix.get("failure_signature") != "ModuleNotFoundError: No module named 'research'":
        raise ValueError("执行路径修正原因不符合允许范围")
    return fix


def verify_manifest(manifest: dict) -> None:
    recorded_hash = manifest.get("manifest_content_sha256")
    payload = dict(manifest)
    payload.pop("manifest_content_sha256", None)
    if recorded_hash != canonical_hash(payload):
        raise ValueError("冻结清单内容哈希不匹配")
    if manifest.get("historical_run_completed"):
        raise RuntimeError("MACRO-01历史回测已经执行，协议禁止第二次运行")
    fix = load_and_verify_execution_fix(manifest)
    for relative, expected in manifest["frozen_file_sha256"].items():
        if relative == RUN_SCRIPT_RELATIVE:
            expected = fix["new_run_script_sha256"]
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"冻结实现被修改：{relative}，实际{actual}")
    for relative, expected in manifest["input_file_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"冻结输入被修改：{relative}，实际{actual}")


def main() -> int:
    if not MANIFEST.exists():
        raise FileNotFoundError("缺少冻结清单，拒绝查看真实收益")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    config = load_config()
    for output in config["outputs"].values():
        if (ROOT / output).exists():
            raise FileExistsError(f"历史输出已存在，拒绝覆盖：{output}")

    report, artifacts = evaluate_protocol(config)
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    outputs = config["outputs"]
    for relative in outputs.values():
        (ROOT / relative).parent.mkdir(parents=True, exist_ok=True)
    artifacts["base_ledger"].to_parquet(ROOT / outputs["base_ledger"], index=False)
    artifacts["base_trades"].to_parquet(ROOT / outputs["base_trades"], index=False)
    artifacts["stress_ledger"].to_parquet(ROOT / outputs["stress_ledger"], index=False)
    artifacts["stress_trades"].to_parquet(ROOT / outputs["stress_trades"], index=False)
    artifacts["signals"].to_parquet(ROOT / outputs["signals"], index=False)
    artifacts["intervals"].to_parquet(ROOT / outputs["intervals"], index=False)
    (ROOT / outputs["report_json"]).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8"
    )
    (ROOT / outputs["report_markdown"]).write_text(markdown_report(report), encoding="utf-8")

    manifest.update(
        {
            "historical_run_completed": True,
            "historical_run_completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "real_510300_future_returns_seen_after_freeze": True,
            "m2_factor_outcomes_seen_after_freeze": True,
            "historical_result_status": "PASS_DISCOVERY" if report["decision"].startswith("DISCOVERY_PASS") else "REJECT_DISCOVERY",
            "decision": report["decision"],
            "failed_gates": report["failed_gates"],
            "result_files": {relative: sha256_file(ROOT / relative) for relative in outputs.values()},
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        }
    )
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "failed_gates": report["failed_gates"],
                "base_cagr": report["account_paths"]["base_combined"]["cagr"],
                "base_sharpe": report["account_paths"]["base_combined"]["sharpe_excess_cash"],
                "base_profit_factor": report["comparisons"]["base_monthly_excess_profit_factor_vs_static_60"],
                "report": outputs["report_markdown"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
