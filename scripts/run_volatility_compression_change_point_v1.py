"""运行510300波动压缩变盘事件研究并生成审计图。"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.volatility_compression_change_point import (
    build_events,
    build_features,
    load_config,
    load_inputs,
    matched_control,
    recent_state,
    summarize_rule,
)


MANIFEST = ROOT / "config" / "volatility_compression_change_point_v1_implementation_manifest.json"
OUTPUT_DIR = ROOT / "data" / "processed" / "volatility_compression_change_point_v1"
REPORT_JSON = ROOT / "reports" / "discovery" / "volatility_compression_change_point_v1.json"
REPORT_MD = ROOT / "reports" / "discovery" / "volatility_compression_change_point_v1.md"
FIGURE = ROOT / "reports" / "figures" / "volatility_compression_change_point_v1.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise RuntimeError("NO_VIEW：波动压缩实现尚未冻结")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("outcome_calculation_allowed") is not True:
        raise RuntimeError("NO_VIEW：冻结清单未允许结果计算")
    for section in ("implementation_files", "input_files"):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError(f"NO_VIEW：冻结指纹变化：{relative_path}")
    return manifest


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _ratio(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}倍"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 510300波动压缩是否为变盘点：冻结事件研究V1",
        "",
        f"- 数据截止：{report['data_cutoff']}",
        "- 字面规则：当前20日平均振幅0.5%–1.0%，此前20日为1.5%–2.5%。",
        "- 自适应规则：RV5<RV20<RV60，且RV20位于此前三年20%低分位以下。",
        "- 事件连续出现只取第一天，间隔至少20个交易日。",
        "",
        "| 规则 | 20日成熟事件 | 未来20日振幅比中位数 | 扩张比例 | 首次向上/向下/无突破 | 波动准备 | 方向信号 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rule_id in ("LITERAL", "ADAPTIVE"):
        item = report["rules"][rule_id]
        counts = item.get("first_breakout_counts_20d", {})
        lines.append(
            "| {rule} | {count} | {ratio} | {frequency} | {up}/{down}/{none} | {vol} | {direction} |".format(
                rule=rule_id,
                count=item["mature_20d_event_count"],
                ratio=_ratio(item.get("future_amplitude_ratio_20_median")),
                frequency=_pct(item.get("expansion_frequency_20d")),
                up=counts.get("UP", 0),
                down=counts.get("DOWN", 0),
                none=counts.get("NONE", 0),
                vol="支持" if item.get("volatility_preparation_supported") else "不支持/不足",
                direction="支持" if item.get("directional_signal_supported") else "不支持",
            )
        )
    current = report["recent_state"]
    lines.extend(
        [
            "",
            "## 当前状态",
            "",
            f"- 日期：{current['date']}",
            f"- 当前20日平均振幅：{_pct(current['mean_amplitude_20'])}；此前20日：{_pct(current['prior_mean_amplitude_20'])}。",
            f"- RV5/RV20/RV60：{_pct(current['rv5'])} / {_pct(current['rv20'])} / {_pct(current['rv60'])}。",
            f"- 字面压缩条件：{current['literal_condition']}；自适应压缩条件：{current['adaptive_condition']}。",
            "",
            "波动压缩即使成立，也只表示可能进入波动准备区，不直接生成上涨或下跌交易方向。",
            "",
        ]
    )
    return "\n".join(lines)


def render_figure(features: pd.DataFrame, events: pd.DataFrame, path: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(4, 1, figsize=(15, 16), constrained_layout=True)
    literal = events.loc[events["rule_id"].eq("LITERAL")]
    adaptive = events.loc[events["rule_id"].eq("ADAPTIVE")]

    axes[0].plot(features["date"], features["adjusted_close"], color="#1f2937", linewidth=1.0)
    axes[0].scatter(
        literal["event_date"], literal["adjusted_close"], color="#f59e0b", marker="o", label="字面振幅压缩"
    )
    axes[0].scatter(
        adaptive["event_date"], adaptive["adjusted_close"], color="#2563eb", marker="^", label="自适应RV压缩"
    )
    axes[0].set_title("510300复权价格与波动压缩事件")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        features["date"], features["mean_amplitude_20"] * 100, color="#0f766e", label="当前20日平均振幅"
    )
    axes[1].plot(
        features["date"], features["prior_mean_amplitude_20"] * 100, color="#94a3b8", alpha=0.65, label="此前20日平均振幅"
    )
    for level, color in ((0.5, "#cbd5e1"), (1.0, "#f59e0b"), (1.5, "#fb7185"), (2.5, "#ef4444")):
        axes[1].axhline(level, color=color, linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("日振幅均值（%）")
    axes[1].set_title("对话字面规则：0.5%–1.0% 对此前1.5%–2.5%")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    axes[2].plot(features["date"], features["rv5"] * 100, label="RV5", alpha=0.7)
    axes[2].plot(features["date"], features["rv20"] * 100, label="RV20", linewidth=1.2)
    axes[2].plot(features["date"], features["rv60"] * 100, label="RV60", alpha=0.7)
    axes[2].plot(
        features["date"], features["rv20_trailing_q20"] * 100, label="此前三年RV20的20%分位", color="#dc2626", linestyle="--"
    )
    axes[2].set_ylabel("年化波动率（%）")
    axes[2].set_title("尺度无关的RV期限结构压缩")
    axes[2].legend(loc="upper right", ncol=2)
    axes[2].grid(alpha=0.2)

    mature = events.loc[events["mature_20d"].astype(bool)].copy()
    colors = mature["rule_id"].map({"LITERAL": "#f59e0b", "ADAPTIVE": "#2563eb"})
    axes[3].scatter(
        mature["event_date"], mature["future_amplitude_ratio_20d"], c=colors, s=45, alpha=0.85
    )
    axes[3].axhline(1.0, color="#111827", linestyle="--", linewidth=1.0, label="未来振幅=当前振幅")
    axes[3].axhline(1.5, color="#dc2626", linestyle=":", linewidth=1.0, label="扩大50%")
    axes[3].set_ylabel("未来20日/当前20日振幅比")
    axes[3].set_title("每个事件之后是否真正扩张")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    manifest = verify_freeze()
    config = load_config()
    market, dividends = load_inputs(ROOT, config)
    features = build_features(market, dividends, config)
    events = build_events(features, config)
    rules: dict[str, Any] = {}
    for rule_id in ("LITERAL", "ADAPTIVE"):
        matched = matched_control(features, events, rule_id, config)
        rules[rule_id] = summarize_rule(events, rule_id, matched, config)
    report = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "evidence_label": config["protocol"]["historical_evidence_label"],
        "data_cutoff": config["protocol"]["data_cutoff"],
        "true_forward_start": None,
        "rules": rules,
        "recent_state": recent_state(features, events),
        "implementation_manifest_sha256": sha256(MANIFEST),
        "freeze_stage": manifest["freeze_stage"],
        "live_position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    safe = _safe(report)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_DIR / "daily_volatility_features.parquet", index=False)
    events.to_csv(OUTPUT_DIR / "compression_event_audit.csv", index=False)
    REPORT_JSON.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(safe), encoding="utf-8")
    render_figure(features, events, FIGURE)
    print(
        json.dumps(
            {
                "report": str(REPORT_JSON),
                "figure": str(FIGURE),
                "events": {
                    rule: rules[rule]["mature_20d_event_count"] for rule in rules
                },
                "decisions": {rule: rules[rule]["decision"] for rule in rules},
                "recent_state": safe["recent_state"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
