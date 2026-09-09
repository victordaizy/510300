"""运行冻结的广泛流动 ETF 流动性冲击反转可见期或封存期。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import (
    build_liquidity_shock_signals,
    evaluate_historical_returns,
    load_benchmark_series,
    load_contract,
    load_etf_inputs,
    run_portfolio_backtest,
)
from scripts.freeze_broad_liquid_etf_liquidity_shock_reversal_v1 import (
    verify as verify_manifest,
)


DEFAULT_CONFIG = ROOT / "config" / "broad_liquid_etf_liquidity_shock_reversal_v1.yaml"


def sha256_file(path: Path) -> str:
    """流式计算输入或配置文件哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    """以同目录临时文件原子写入文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """以同目录临时文件原子写入 Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _phase_paths(contract: dict[str, Any], phase: str) -> dict[str, Path]:
    prefix = "visible" if phase == "visible" else "sealed"
    outputs = contract["outputs"]
    return {
        "daily": ROOT / outputs[f"{prefix}_daily_returns"],
        "trades": ROOT / outputs[f"{prefix}_trades"],
        "json": ROOT / outputs[f"{prefix}_report_json"],
        "markdown": ROOT / outputs[f"{prefix}_report_markdown"],
    }


def _phase_dates(contract: dict[str, Any], phase: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    partition = contract["historical_partition"]
    if phase == "visible":
        return pd.Timestamp(partition["visible_start"]), pd.Timestamp(partition["visible_end"])
    return pd.Timestamp(partition["sealed_replication_start"]), pd.Timestamp(
        partition["sealed_replication_end"]
    )


def _require_visible_pass(contract: dict[str, Any]) -> None:
    visible_path = ROOT / contract["outputs"]["visible_report_json"]
    if not visible_path.is_file():
        raise RuntimeError("封存期仍关闭：缺少可见期报告")
    visible = json.loads(visible_path.read_text(encoding="utf-8"))
    if visible.get("status") != "VISIBLE_40PCT_HIGH_SHARPE_PASS_READY_FOR_SEALED_REPLICATION":
        raise RuntimeError(f"封存期仍关闭：可见期状态为{visible.get('status')}")
    if not bool(visible.get("evaluation", {}).get("all_visible_gates_pass")):
        raise RuntimeError("封存期仍关闭：可见期并非全部门槛通过")


def build_report(
    contract: dict[str, Any],
    config_path: Path,
    *,
    phase: str,
    bootstrap_repetitions_override: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """验证冻结状态并构建一个完整历史阶段报告。"""

    manifest = verify_manifest()
    if manifest["failure_count"]:
        raise RuntimeError(f"冻结清单验证失败：{manifest['failures']}")
    if phase == "sealed":
        _require_visible_pass(contract)
    start, end = _phase_dates(contract, phase)
    inputs = {key: ROOT / relative for key, relative in contract["inputs"].items()}
    panel, master = load_etf_inputs(
        inputs["etf_total_return_panel"],
        inputs["fund_master"],
    )
    benchmark = load_benchmark_series(
        inputs["benchmark_total_return"],
        inputs["benchmark_price_ohlc"],
    )
    signals, market, return_wide, signal_audit = build_liquidity_shock_signals(
        panel,
        master,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    daily, trades, portfolio_audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        return_wide,
        contract,
        start=start,
        end=end,
    )
    evaluation = evaluate_historical_returns(
        daily,
        contract,
        bootstrap_repetitions_override=bootstrap_repetitions_override,
    )
    if phase == "visible":
        status = (
            "VISIBLE_40PCT_HIGH_SHARPE_PASS_READY_FOR_SEALED_REPLICATION"
            if evaluation["all_visible_gates_pass"]
            else "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN"
        )
    else:
        status = (
            "SEALED_REPLICATION_PASS_RETROSPECTIVE_NOT_VERIFIED"
            if evaluation["all_visible_gates_pass"]
            else "REJECTED_SEALED_REPLICATION_FROZEN"
        )
    report = {
        "schema_version": "1.0.0",
        "report_id": f"BROAD_LIQUID_ETF_LIQUIDITY_SHOCK_REVERSAL_V1_{phase.upper()}",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": phase,
        "status": status,
        "goal_achieved": False,
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "period": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "objective": {
            "initial_capital_cny": float(contract["account"]["initial_capital_cny"]),
            "user_transaction_fee_rate_per_leg": float(
                contract["account"]["user_transaction_fee_rate_per_leg"]
            ),
            "minimum_annualized_net_excess": float(
                contract["visible_gates"]["minimum_annualized_net_excess"]
            ),
            "minimum_strategy_net_sharpe": float(
                contract["visible_gates"]["minimum_strategy_net_sharpe"]
            ),
        },
        "manifest_verification": manifest,
        "data_audit": {
            "liquidity_shock_signal": signal_audit,
            "portfolio": portfolio_audit,
        },
        "evaluation": evaluation,
        "historical_evidence_limits": contract["historical_evidence_limits"],
        "decision": {
            "sealed_replication_open": bool(
                phase == "visible" and evaluation["all_visible_gates_pass"]
            ),
            "candidate_may_be_reparameterized_after_failure": False,
            "historical_result_verifies_40pct_target": False,
            "next_step": (
                "运行冻结封存期复制；即使通过仍需726个前瞻交易日"
                if status == "VISIBLE_40PCT_HIGH_SHARPE_PASS_READY_FOR_SEALED_REPLICATION"
                else "冻结拒绝本候选，不打开封存期，不调参救回"
                if phase == "visible"
                else "即使封存复制通过仍只能进入前瞻观察，不能宣布目标已达成"
            ),
        },
        "inputs": {
            "config": {
                "path": config_path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(config_path),
            },
            **{
                key: {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for key, path in inputs.items()
            },
        },
        "outputs": {
            "daily_row_count": int(len(daily)),
            "trade_row_count": int(len(trades)),
        },
        "bootstrap_repetitions_override": bootstrap_repetitions_override,
        "safety": contract["safety"],
    }
    return report, daily, trades


def render_markdown(report: dict[str, Any]) -> str:
    """渲染便于人工复核的短报告。"""

    metrics = report["evaluation"]["metrics"]
    failed = [name for name, passed in report["evaluation"]["gates"].items() if not passed]
    signal = report["data_audit"]["liquidity_shock_signal"]
    portfolio = report["data_audit"]["portfolio"]
    lines = [
        f"# 广泛流动ETF流动性冲击反转 V1：{report['phase']}",
        "",
        f"- 状态：`{report['status']}`",
        "- 目标已达成：否。历史结果不能验证40个百分点目标。",
        f"- 区间：{report['period']['start']} 至 {report['period']['end']}。",
        "- 账户：500,000元；用户交易费率为每条买卖腿0.0001。",
        "",
        "## 成本后结果",
        "",
        f"- H00300年化：{metrics['benchmark_total_return_cagr']:.2%}",
        f"- 基础净年化：{metrics['strategy_base_net_cagr']:.2%}；净超额：{metrics['base_annualized_excess']:.2%}；净夏普：{metrics['base_strategy_net_sharpe']:.3f}；最大回撤：{metrics['base_maximum_drawdown']:.2%}",
        f"- 压力净年化：{metrics['strategy_stress_net_cagr']:.2%}；净超额：{metrics['stress_annualized_excess']:.2%}；净夏普：{metrics['stress_strategy_net_sharpe']:.3f}；最大回撤：{metrics['stress_maximum_drawdown']:.2%}",
        f"- 未通过门槛：{', '.join(failed) if failed else '无'}",
        "",
        "## 覆盖与执行",
        "",
        f"- 可执行信号：{signal['executable_signal_rows']}条；信号日：{signal['signal_day_count']}个；涉及ETF：{signal['unique_etf_count']}只。",
        f"- 买入成交：{portfolio['entry_transaction_count']}笔；卖出成交：{portfolio['exit_transaction_count']}笔；受阻退出尝试：{portfolio['blocked_exit_attempt_count']}次。",
        f"- 最大成交额占过去20日中位成交额：{portfolio['maximum_capacity_fraction']:.4%}。",
        f"- 最大持仓数：{portfolio['maximum_position_count']}；最大毛敞口：{portfolio['maximum_gross_exposure']:.3f}。",
        "",
        "## 证据限制",
        "",
        "- ETF行情和基金主表是当前归档，缺少逐日不可变供应商版本链。",
        "- 历史数据没有逐笔买卖价差；基础与压力滑点、冲击为冻结模型值。",
        "- 历史结果只用于淘汰或进入下一阶段，不是收益承诺、仓位、订单或实盘授权。",
        "",
        "## 决策",
        "",
        report["decision"]["next_step"] + "。",
        "",
    ]
    return "\n".join(lines)


def _write_failure(contract: dict[str, Any], phase: str, error: Exception) -> None:
    """保留数据或执行合同失败，不用成功报告覆盖。"""

    paths = _phase_paths(contract, phase)
    payload = {
        "schema_version": "1.0.0",
        "report_id": f"BROAD_LIQUID_ETF_LIQUIDITY_SHOCK_REVERSAL_V1_{phase.upper()}",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "phase": phase,
        "status": "FAILED_DATA_OR_EXECUTION_CONTRACT",
        "goal_achieved": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "sealed_replication_open": False,
        "safety": contract["safety"],
    }
    atomic_text(paths["json"], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    atomic_text(
        paths["markdown"],
        "\n".join(
            [
                f"# 广泛流动ETF流动性冲击反转 V1：{phase}",
                "",
                "- 状态：`FAILED_DATA_OR_EXECUTION_CONTRACT`",
                f"- 错误类型：`{type(error).__name__}`",
                f"- 错误：{error}",
                "- 封存期不开启；不回填、不替代、不调参。",
                "",
            ]
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="运行冻结广泛流动ETF冲击反转候选")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("visible", "sealed"), default="visible")
    parser.add_argument("--bootstrap-repetitions", type=int)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_repetitions is not None and not args.no_write:
        parser.error("Bootstrap次数覆盖只能与--no-write共同用于测试，不能写入权威报告")
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    contract = load_contract(config_path)
    try:
        report, daily, trades = build_report(
            contract,
            config_path,
            phase=args.phase,
            bootstrap_repetitions_override=args.bootstrap_repetitions,
        )
    except Exception as error:
        if not args.no_write:
            _write_failure(contract, args.phase, error)
        print(
            json.dumps(
                {
                    "状态": "FAILED_DATA_OR_EXECUTION_CONTRACT",
                    "错误类型": type(error).__name__,
                    "错误": str(error),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    if not args.no_write:
        paths = _phase_paths(contract, args.phase)
        atomic_parquet(paths["daily"], daily)
        atomic_parquet(paths["trades"], trades)
        atomic_text(
            paths["json"],
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        )
        atomic_text(paths["markdown"], render_markdown(report))
    metrics = report["evaluation"]["metrics"]
    print(
        json.dumps(
            {
                "状态": report["status"],
                "目标已达成": report["goal_achieved"],
                "基础年化净超额": metrics["base_annualized_excess"],
                "压力年化净超额": metrics["stress_annualized_excess"],
                "基础净夏普": metrics["base_strategy_net_sharpe"],
                "压力净夏普": metrics["stress_strategy_net_sharpe"],
                "全部可见门槛通过": report["evaluation"]["all_visible_gates_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
