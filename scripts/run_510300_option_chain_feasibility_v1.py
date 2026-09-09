"""执行一次510300期权链结果盲数据可行性审计。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.option_chain_feasibility_v1 import (  # noqa: E402
    load_protocol,
    render_markdown,
    run_feasibility_audit,
    sha256,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
PROTOCOL_PATH = ROOT / "config" / "510300_option_chain_feasibility_v1.yaml"
MANIFEST_PATH = ROOT / "config" / "510300_option_chain_feasibility_v1_manifest.json"


def atomic_json(payload: dict, path: Path) -> None:
    """原子写入JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_text(content: str, path: Path) -> None:
    """原子写入UTF-8文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def verify_manifest() -> tuple[dict, dict]:
    """验证预注册协议、实现和输入哈希，拒绝第二次数据审计运行。"""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("缺少期权链可行性冻结清单")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("freeze_state") != "PREREGISTERED_BEFORE_SURFACE_METRICS":
        raise RuntimeError("期权链可行性冻结状态不正确")
    if int(manifest.get("data_feasibility_runs_consumed", 0)) != 0:
        raise RuntimeError("期权链数据可行性唯一历史运行已经消费")
    protocol = load_protocol(PROTOCOL_PATH)
    if sha256(PROTOCOL_PATH) != manifest["protocol_sha256"]:
        raise RuntimeError("期权链可行性协议哈希漂移")
    for mapping_name in ("implementation_sha256", "input_sha256"):
        for relative, expected in manifest[mapping_name].items():
            path = ROOT / relative
            if not path.exists():
                raise FileNotFoundError(f"冻结文件缺失：{relative}")
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"冻结文件哈希漂移：{relative}")
    return protocol, manifest


def main() -> int:
    """运行审计、写入结果，并消费唯一一次数据可行性运行。"""

    protocol, manifest = verify_manifest()
    report, maturity_audit, daily_audit, official_receipt = run_feasibility_audit(
        ROOT, protocol
    )
    paths = protocol["paths"]
    result_json = ROOT / paths["result_json"]
    result_markdown = ROOT / paths["result_markdown"]
    daily_path = ROOT / paths["daily_audit"]
    maturity_path = ROOT / paths["maturity_audit"]
    receipt_path = ROOT / paths["official_source_receipt"]
    generated_at = datetime.now(TIMEZONE).isoformat()
    report["generated_at"] = generated_at
    report["protocol_sha256"] = manifest["protocol_sha256"]
    report["manifest_freeze_id"] = manifest["freeze_id"]
    atomic_parquet(maturity_audit, maturity_path)
    atomic_parquet(daily_audit, daily_path)
    atomic_json(official_receipt, receipt_path)
    atomic_json(report, result_json)
    atomic_text(render_markdown(report), result_markdown)
    result_paths = [
        result_json,
        result_markdown,
        daily_path,
        maturity_path,
        receipt_path,
    ]
    manifest["data_feasibility_runs_consumed"] = 1
    manifest["completed_at"] = generated_at
    manifest["terminal_status"] = report["status"]
    manifest["future_data_reads"] = report["outcome_blindness"]["future_data_reads"]
    manifest["result_sha256"] = {
        path.relative_to(ROOT).as_posix(): sha256(path) for path in result_paths
    }
    atomic_json(manifest, MANIFEST_PATH)
    print(
        json.dumps(
            {
                "study_id": report["study_id"],
                "status": report["status"],
                "valid_surface_day_ratio": report["coverage"][
                    "valid_surface_day_ratio"
                ],
                "failed_hard_gates": report["failed_hard_gates"],
                "future_data_reads": report["outcome_blindness"]["future_data_reads"],
                "prediction_model_created": report["governance"][
                    "prediction_model_created"
                ],
                "portfolio_evaluation_run": report["governance"][
                    "portfolio_evaluation_run"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

