"""统一审计现有510300策略相对沪深300全收益指数的净超额。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.benchmark_relative_objective import (
    evaluate_objective_gates,
    evaluate_relative_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "benchmark_relative_objective.yaml"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _render_markdown(payload: dict[str, Any]) -> str:
    def pct(value: float) -> str:
        return f"{value:.2%}"

    lines = [
        "# 510300相对沪深300全收益指数目标审计",
        "",
        "> 主目标是扣除账户成本后的H00300净超额，不再以策略绝对盈利作为通过依据。全部历史已参与研究，历史通过只允许纸面前向观察。",
        "",
        "## 统一排名",
        "",
        "| 排名 | 策略 | CAGR | 年化超额 | 信息比率 | 滚动一年超额中位数/跑赢比例 | 前半/后半超额 | 相对最大回撤 | 结论 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(payload["ranking"], start=1):
        base = item["base"]
        halves = base["chronological_halves"]
        lines.append(
            f"| {rank} | {item['strategy_id']} | {pct(base['strategy_cagr'])} | "
            f"{pct(base['annualized_excess'])} | {base['information_ratio']:.2f} | "
            f"{pct(base['rolling_excess']['median'])}/{pct(base['rolling_excess']['positive_ratio'])} | "
            f"{pct(halves[0]['excess_return'])}/{pct(halves[1]['excess_return'])} | "
            f"{pct(base['maximum_relative_drawdown'])} | `{item['decision']['status']}` |"
        )
    selected = payload["selection"]["forward_observation_candidate"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 历史门槛全部通过且具备压力成本账本：{payload['selection']['pass_count']}条。",
            f"- 纸面前向观察候选：`{selected}`。" if selected else "- 当前没有纸面前向观察候选。",
            "- 不授权自动下单；冻结后必须用新数据验证，不能继续用同一五年历史调参。",
            "",
            "## 通过门槛",
            "",
            "- 基础与15bp压力成本的年化超额均为正。",
            "- 前后两个时间段超额均为正。",
            "- 滚动242日超额中位数为正，至少55%的窗口跑赢。",
            "- 信息比率为正，相对净值最大回撤不超过20%。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    benchmark_path = ROOT / config["data"]["benchmark_file"]
    benchmark = pd.read_parquet(benchmark_path)
    evaluation = config["evaluation"]
    gates = {**config["gates"], "minimum_evaluation_years": evaluation["minimum_evaluation_years"]}
    results: list[dict[str, Any]] = []
    for candidate in config["strategies"]:
        base_path = ROOT / candidate["base_ledger"]
        if not base_path.exists():
            raise FileNotFoundError(f"缺少基础成本账本：{base_path}")
        stress_path = (
            ROOT / candidate["stress_ledger"] if candidate.get("stress_ledger") else None
        )
        if stress_path is not None and not stress_path.exists():
            raise FileNotFoundError(f"缺少压力成本账本：{stress_path}")
        base = evaluate_relative_ledger(
            pd.read_parquet(base_path),
            benchmark,
            float(evaluation["initial_cash_cny"]),
            int(evaluation["trading_days_per_year"]),
            int(evaluation["rolling_window_trading_days"]),
        )
        stress = (
            evaluate_relative_ledger(
                pd.read_parquet(stress_path),
                benchmark,
                float(evaluation["initial_cash_cny"]),
                int(evaluation["trading_days_per_year"]),
                int(evaluation["rolling_window_trading_days"]),
            )
            if stress_path is not None
            else None
        )
        decision = evaluate_objective_gates(base, stress, gates)
        if not bool(candidate["eligible_for_selection"]):
            decision["status"] = "AUDIT_ONLY_NOT_ELIGIBLE"
        results.append(
            {
                **candidate,
                "base": base,
                "stress": stress,
                "decision": decision,
            }
        )

    ranking = sorted(results, key=lambda item: item["base"]["annualized_excess"], reverse=True)
    passed = [
        item
        for item in ranking
        if item["eligible_for_selection"]
        and item["decision"]["status"] == "RETROSPECTIVE_PASS_FORWARD_ONLY"
    ]
    payload = {
        "status": "PASS_OBJECTIVE_AUDIT_COMPLETED",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")),
        "objective": config["objective"],
        "ranking": ranking,
        "selection": {
            "pass_count": len(passed),
            "forward_observation_candidate": passed[0]["strategy_id"] if passed else None,
            "live_trading_authorized": False,
        },
        "gates": gates,
        "governance": config["governance"],
    }
    ready = _json_ready(payload)
    json_path = ROOT / config["output"]["report_json"]
    markdown_path = ROOT / config["output"]["report_markdown"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(ready), encoding="utf-8")
    print(json.dumps(ready, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
