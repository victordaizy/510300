"""不读取未来收益地审计并落盘X1共同尾部信号。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.x1_common_tail_prediction import (  # noqa: E402
    X1Rules,
    build_x1_common_tail_signal,
)


CONTRACT_FILE = ROOT / "config" / "x1_common_tail_prediction_v1.yaml"
STATUS_JSON = ROOT / "reports" / "data_quality" / "x1_common_tail_signal_v1_status.json"
STATUS_MD = ROOT / "reports" / "data_quality" / "x1_common_tail_signal_v1_status.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in directory.glob("*.parquet") if path.is_file())
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_history(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("完整个股历史缓存为空")
    frames = [
        pd.read_parquet(
            path, columns=["date", "con_code", "total_return_close"]
        )
        for path in files
    ]
    return pd.concat(frames, ignore_index=True)


def _render(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# X1共同尾部信号数据门 V1",
            "",
            f"- 状态：`{report['status']}`",
            f"- 审计时间：`{report['checked_at']}`",
            "- 未来收益读取：`false`",
            "- 预测模型拟合：`false`",
            f"- 个股历史文件：{report['component_history_file_count']}",
            f"- 点时成员日：{report['signal_day_count']}",
            f"- 有效信号日：{report['ready_signal_day_count']}",
            f"- 首个有效日：{report['first_ready_signal_date']}",
            f"- 最新有效日：{report['last_ready_signal_date']}",
            f"- 有效日最低权重覆盖：{report['minimum_ready_weight_coverage']}",
            f"- 最新权重覆盖：{report['latest_weight_coverage']}",
            "",
            "完整个股历史只用于形成各证券事前504日尾部分位；每天的X1分子和分母仍严格使用当天点时300只成员。",
        ]
    ) + "\n"


def main() -> int:
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    inputs = contract["inputs"]
    feasibility = json.loads(
        (ROOT / inputs["current_feasibility_status"]).read_text(encoding="utf-8")
    )
    if feasibility.get("branch_status", {}).get("CONSTITUENTS") != "PASS":
        raise RuntimeError("当前成分股分支数据门不是PASS")
    if feasibility.get("hypotheses_allowed_for_return_test") != ["X1"]:
        raise RuntimeError("当前收益测试许可不是唯一X1")
    history_directory = ROOT / inputs["component_history_cache"]
    history_files = sorted(history_directory.glob("*.parquet"))
    if len(history_files) != 493:
        raise RuntimeError(f"完整个股历史缓存文件数不是493：{len(history_files)}")
    rules = X1Rules.from_contract(contract)
    signal = build_x1_common_tail_signal(
        pd.read_parquet(ROOT / inputs["constituent_daily"]),
        _load_history(history_directory),
        pd.read_parquet(ROOT / inputs["constituent_weights"]),
        pd.read_parquet(ROOT / inputs["industry_intervals"]),
        rules,
    )
    ready = signal.loc[signal["signal_output"].eq("SIGNAL_READY")]
    latest = signal.iloc[-1]
    gates = {
        "minimum_probability_training_window": len(ready)
        >= rules.minimum_probability_training,
        "latest_signal_ready": latest["signal_output"] == "SIGNAL_READY",
        "minimum_ready_weight_coverage": ready["tail_signal_weight_coverage"].min()
        >= rules.minimum_signal_weight_coverage,
        "point_in_time_member_count": signal["member_count"].eq(300).all(),
        "industry_secondary_available": ready["industry_tail_breadth"].notna().all(),
    }
    status = "READY_FOR_OUTCOME_FREEZE" if all(gates.values()) else "NO_VIEW"
    signal_path = ROOT / inputs["x1_signal_feature"]
    _atomic_parquet(signal, signal_path)
    report = {
        "project_id": contract["protocol"]["project_id"],
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": status,
        "failure_category": "PASS" if status == "READY_FOR_OUTCOME_FREEZE" else "SIGNAL_DATA_GATE_FAILED",
        "future_return_read": False,
        "predictive_model_fitted": False,
        "component_history_file_count": len(history_files),
        "component_history_row_count": int(
            sum(len(pd.read_parquet(path, columns=["date"])) for path in history_files)
        ),
        "signal_day_count": int(len(signal)),
        "ready_signal_day_count": int(len(ready)),
        "no_view_signal_day_count": int(signal["signal_output"].eq("NO_VIEW").sum()),
        "warmup_signal_day_count": int(signal["signal_output"].eq("WARMUP").sum()),
        "first_ready_signal_date": str(ready["date"].min().date()),
        "last_ready_signal_date": str(ready["date"].max().date()),
        "minimum_ready_weight_coverage": float(
            ready["tail_signal_weight_coverage"].min()
        ),
        "latest_weight_coverage": float(latest["tail_signal_weight_coverage"]),
        "gates": {name: bool(value) for name, value in gates.items()},
        "input_hashes": {
            inputs["constituent_daily"]: sha256(ROOT / inputs["constituent_daily"]),
            inputs["constituent_weights"]: sha256(ROOT / inputs["constituent_weights"]),
            inputs["constituent_membership"]: sha256(ROOT / inputs["constituent_membership"]),
            inputs["industry_intervals"]: sha256(ROOT / inputs["industry_intervals"]),
            inputs["component_history_cache"]: tree_sha256(history_directory),
        },
        "output_hashes": {inputs["x1_signal_feature"]: sha256(signal_path)},
        "safety": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }
    _atomic_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        STATUS_JSON,
    )
    _atomic_text(_render(report), STATUS_MD)
    print(
        json.dumps(
            {
                "状态": status,
                "有效信号日": len(ready),
                "首个有效日": report["first_ready_signal_date"],
                "最新有效日": report["last_ready_signal_date"],
                "最低有效权重覆盖": report["minimum_ready_weight_coverage"],
                "未来收益读取": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
