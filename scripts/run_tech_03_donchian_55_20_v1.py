"""运行冻结的TECH_03唐奇安55/20历史诊断。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.tech_03_donchian_55_20_v1 import (
    active_return_frame,
    audit_and_load_inputs,
    build_buy_hold_signals,
    build_cycles,
    build_event_audit,
    build_fixed_mix_benchmark,
    build_h00300_benchmark,
    build_signals,
    evaluate_decision,
    load_config,
    make_cost_model,
    multiple_testing_diagnostics,
    period_diagnostics,
    relative_metrics,
    simulate_portfolio,
    summarize_ledger,
    validate_manifest,
)


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_serializable(item) for item in value.to_dict("records")]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _variant_slug(variant_id: str) -> str:
    return "primary_55_20" if variant_id == "TECH_03_DONCHIAN_55_20_V1" else "robustness_100_50"


def _layer_payload(
    signals: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    layer: str,
    stress: bool,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    fractional = layer == "theoretical"
    initial = (
        float(config["execution"]["theoretical"]["initial_wealth"])
        if fractional
        else float(config["execution"]["small_account"]["initial_cash_cny"])
    )
    costs = make_cost_model(config, layer, stress)
    strategy_ledger, strategy_trades = simulate_portfolio(
        signals, dividends, costs, initial, fractional=fractional
    )
    buy_hold_signals = build_buy_hold_signals(signals, f"{signals['variant_id'].iloc[0]}_BUY_HOLD")
    buy_hold_ledger, buy_hold_trades = simulate_portfolio(
        buy_hold_signals, dividends, costs, initial, fractional=fractional
    )
    h00300 = build_h00300_benchmark(
        signals,
        initial,
        cash_annual_rate=float(config["execution"]["cash_annual_rate"]),
        trading_days=int(config["execution"]["trading_days_per_year"]),
    )
    summary_strategy = summarize_ledger(
        strategy_ledger,
        strategy_trades,
        initial,
        costs.cash_annual_rate,
        costs.trading_days_per_year,
        costs.minimum_commission,
    )
    summary_buy_hold = summarize_ledger(
        buy_hold_ledger,
        buy_hold_trades,
        initial,
        costs.cash_annual_rate,
        costs.trading_days_per_year,
        costs.minimum_commission,
    )
    summary_h00300 = summarize_ledger(
        h00300,
        pd.DataFrame(),
        initial,
        costs.cash_annual_rate,
        costs.trading_days_per_year,
        0.0,
    )
    relative_510300, rolling_510300 = relative_metrics(
        strategy_ledger,
        buy_hold_ledger,
        costs.trading_days_per_year,
        config["evaluation"]["rolling_windows_trading_days"],
    )
    relative_h00300, rolling_h00300 = relative_metrics(
        strategy_ledger,
        h00300,
        costs.trading_days_per_year,
        config["evaluation"]["rolling_windows_trading_days"],
    )
    payload = {
        "layer": layer,
        "stress_cost": stress,
        "strategy": summary_strategy,
        "buy_hold": summary_buy_hold,
        "h00300": summary_h00300,
        "relative_vs_510300": relative_510300,
        "relative_vs_h00300": relative_h00300,
    }
    frames = {
        "strategy_ledger": strategy_ledger,
        "strategy_trades": strategy_trades,
        "buy_hold_ledger": buy_hold_ledger,
        "buy_hold_trades": buy_hold_trades,
        "h00300_ledger": h00300,
    }
    for window, frame in rolling_510300.items():
        frames[f"rolling_{window}d_vs_510300"] = frame
    for window, frame in rolling_h00300.items():
        frames[f"rolling_{window}d_vs_h00300"] = frame
    return payload, frames


def _fixed_benchmarks(
    signals: pd.DataFrame,
    dividends: pd.DataFrame,
    strategy_ledger: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    initial = float(config["execution"]["theoretical"]["initial_wealth"])
    cash_rate = float(config["execution"]["cash_annual_rate"])
    trading_days = int(config["execution"]["trading_days_per_year"])
    exposures = [float(value) for value in config["evaluation"]["fixed_exposure_benchmarks"]]
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, Any] = {}
    for exposure in exposures:
        name = f"fixed_{int(exposure * 100)}pct"
        frame = build_fixed_mix_benchmark(
            signals, dividends, exposure, initial, cash_rate, trading_days
        )
        frames[name] = frame
        summaries[name] = summarize_ledger(
            frame, pd.DataFrame(), initial, cash_rate, trading_days, 0.0
        )
    average_exposure = float(strategy_ledger["actual_position"].mean())
    same_average = build_fixed_mix_benchmark(
        signals, dividends, average_exposure, initial, cash_rate, trading_days
    )
    full = frames["fixed_100pct"]
    full_volatility = full["daily_return"].iloc[1:].std(ddof=1)
    strategy_volatility = strategy_ledger["daily_return"].iloc[1:].std(ddof=1)
    volatility_exposure = float(np.clip(strategy_volatility / full_volatility, 0.0, 1.0)) if full_volatility > 0 else 0.0
    same_volatility = build_fixed_mix_benchmark(
        signals, dividends, volatility_exposure, initial, cash_rate, trading_days
    )
    frames["same_average_exposure"] = same_average
    frames["same_volatility"] = same_volatility
    summaries["same_average_exposure"] = {
        "target_exposure": average_exposure,
        **summarize_ledger(same_average, pd.DataFrame(), initial, cash_rate, trading_days, 0.0),
        "relative": relative_metrics(
            strategy_ledger,
            same_average,
            trading_days,
            config["evaluation"]["rolling_windows_trading_days"],
        )[0],
    }
    summaries["same_volatility"] = {
        "target_exposure": volatility_exposure,
        **summarize_ledger(same_volatility, pd.DataFrame(), initial, cash_rate, trading_days, 0.0),
        "relative": relative_metrics(
            strategy_ledger,
            same_volatility,
            trading_days,
            config["evaluation"]["rolling_windows_trading_days"],
        )[0],
    }
    return summaries, frames


def _markdown_report(report: dict[str, Any]) -> str:
    primary = report["variants"]["TECH_03_DONCHIAN_55_20_V1"]
    robust = report["variants"]["TECH_03_DONCHIAN_100_50_ROBUSTNESS_V1"]
    base = primary["small_account_base"]
    stress = primary["small_account_stress"]
    rel = base["relative_vs_510300"]
    decision = report["decision"]
    pct = lambda value: "N/A" if value is None else f"{value:.2%}"
    num = lambda value: "N/A" if value is None else f"{value:.2f}"
    lines = [
        "# TECH_03_DONCHIAN_55_20_V1 历史诊断",
        "",
        f"> 结论：`{decision['decision']}`。`alpha_pass=false`；这是截至{report['data_cutoff']}的受污染历史，不是当前买卖指令。",
        "",
        "## 核心结果（2万元执行层）",
        "",
        "| 方案 | CAGR | 对510300年化主动收益 | IR | Sharpe | 最大回撤 | 平均仓位 | 成交笔数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 55/20基础成本 | {pct(base['strategy']['cagr'])} | {pct(rel['annualized_active_return'])} | {num(rel['information_ratio'])} | {num(base['strategy']['sharpe_excess_cash'])} | {pct(base['strategy']['maximum_drawdown'])} | {pct(base['strategy']['average_exposure'])} | {base['strategy']['trade_count']} |",
        f"| 55/20双倍成本 | {pct(stress['strategy']['cagr'])} | {pct(stress['relative_vs_510300']['annualized_active_return'])} | {num(stress['relative_vs_510300']['information_ratio'])} | {num(stress['strategy']['sharpe_excess_cash'])} | {pct(stress['strategy']['maximum_drawdown'])} | {pct(stress['strategy']['average_exposure'])} | {stress['strategy']['trade_count']} |",
        f"| 100/50稳健性 | {pct(robust['small_account_base']['strategy']['cagr'])} | {pct(robust['small_account_base']['relative_vs_510300']['annualized_active_return'])} | {num(robust['small_account_base']['relative_vs_510300']['information_ratio'])} | {num(robust['small_account_base']['strategy']['sharpe_excess_cash'])} | {pct(robust['small_account_base']['strategy']['maximum_drawdown'])} | {pct(robust['small_account_base']['strategy']['average_exposure'])} | {robust['small_account_base']['strategy']['trade_count']} |",
        f"| 510300含分红买入持有 | {pct(base['buy_hold']['cagr'])} | 0.00% | — | {num(base['buy_hold']['sharpe_excess_cash'])} | {pct(base['buy_hold']['maximum_drawdown'])} | {pct(base['buy_hold']['average_exposure'])} | {base['buy_hold']['trade_count']} |",
        f"| H00300全收益 | {pct(base['h00300']['cagr'])} | — | — | {num(base['h00300']['sharpe_excess_cash'])} | {pct(base['h00300']['maximum_drawdown'])} | 100.00% | — |",
        "",
        "## 判定",
        "",
        f"- 历史屏幕：`{'PASS' if decision['historical_screen_pass'] else 'FAIL'}`；风险覆盖历史门槛：`{'PASS' if decision['risk_overlay_historical_pass'] else 'FAIL'}`。",
        f"- Sharpe改善：{num(decision['sharpe_improvement'])}；最大回撤降幅：{pct(decision['drawdown_reduction'])}；上涨捕获率：{pct(rel['upside_capture'])}。",
        f"- 四段中主动收益为正：{primary['period_diagnostics']['positive_predefined_period_count']}/4；剔除最佳年份后年化主动收益：{pct(primary['period_diagnostics']['annualized_active_return_without_best_year'])}。",
        f"- 242日滚动超额中位数/正值比例：{pct(rel['rolling_active_returns']['242']['median'])} / {pct(rel['rolling_active_returns']['242']['positive_ratio'])}。",
        f"- 484日滚动超额中位数/正值比例：{pct(rel['rolling_active_returns']['484']['median'])} / {pct(rel['rolling_active_returns']['484']['positive_ratio'])}。",
        "",
        "## 多重检验",
        "",
        f"- 累计登记试验3个，实际收益候选2个；White Reality Check p={report['multiple_testing']['white_reality_check']['p_value']:.4f}。",
        f"- Hansen SPA区块自助近似 p={report['multiple_testing']['hansen_spa']['p_value']:.4f}；Deflated Sharpe概率={pct(report['multiple_testing']['deflated_sharpe_ratio']['probability'])}。",
        f"- PBO：`{report['multiple_testing']['pbo']['status']}`，原因：{report['multiple_testing']['pbo']['reason']}。",
        "- 相邻参数网格未运行，因为冻结预算禁止继续搜索；固定100/50已完整报告。",
        "",
        "## 数据与执行口径",
        "",
        f"- 数据闸门：`{report['data_gate']['status']}`；日期截止{report['data_cutoff']}。",
        "- H00300官方本地快照只有收盘；信号OHLC是用`H00300收盘/000300收盘`缩放000300 OHLC得到的冻结代理，不冒充官方H00300 OHLC。",
        "- T日收盘信号、T+1开盘执行；只在状态改变时交易；100份整手、普通调仓至少1000份、每腿最低5元、5bp滑点、T+1。",
        "- 理论层与2万元执行层分开保存；每个信号、执行和闭合周期均有独立证据文件。",
        "- 当前仓位映射、订单生成和券商连接均为关闭状态。",
        "",
        "## 历史门槛明细",
        "",
        "| 门槛 | 结果 |",
        "|---|---|",
    ]
    for name, passed in decision["historical_gate_checks"].items():
        lines.append(f"| `{name}` | `{'PASS' if passed else 'FAIL'}` |")
    return "\n".join(lines) + "\n"


def _save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        frame.to_parquet(path, index=False)


def main() -> int:
    config = load_config()
    manifest = validate_manifest(ROOT, config)
    market, dividends, audit = audit_and_load_inputs(ROOT, config)
    if audit["status"] != "PASS":
        raise RuntimeError("数据闸门未通过，只允许NO_VIEW")

    processed_dir = ROOT / config["artifacts"]["processed_directory"]
    variant_payloads: dict[str, Any] = {}
    variant_frames: dict[str, dict[str, pd.DataFrame]] = {}
    active_returns: dict[str, pd.DataFrame] = {}
    for variant in config["variants"]:
        variant_id = variant["id"]
        slug = _variant_slug(variant_id)
        signals = build_signals(market, variant)
        theoretical_base, theoretical_frames = _layer_payload(
            signals, dividends, config, "theoretical", stress=False
        )
        theoretical_stress, theoretical_stress_frames = _layer_payload(
            signals, dividends, config, "theoretical", stress=True
        )
        small_base, small_frames = _layer_payload(
            signals, dividends, config, "small_account", stress=False
        )
        small_stress, small_stress_frames = _layer_payload(
            signals, dividends, config, "small_account", stress=True
        )
        fixed_summaries, fixed_frames = _fixed_benchmarks(
            signals,
            dividends,
            theoretical_frames["strategy_ledger"],
            config,
        )
        cycles = build_cycles(
            small_frames["strategy_trades"], signals, dividends
        )
        events = build_event_audit(
            signals,
            small_frames["strategy_trades"],
            cycles,
            config["evaluation"]["event_forward_horizons_trading_days"],
        )
        period_result = period_diagnostics(
            small_frames["strategy_ledger"],
            small_frames["buy_hold_ledger"],
            config["evaluation"]["predefined_periods"],
            int(config["evaluation"]["annualization_days"]),
        )
        closed_cycles = cycles.loc[cycles["closed"]].copy() if not cycles.empty else cycles
        average_holding = float(closed_cycles["holding_trading_days"].mean()) if not closed_cycles.empty else None
        variant_payloads[variant_id] = {
            "role": variant["role"],
            "entry_lookback_trading_days": variant["entry_lookback_trading_days"],
            "exit_lookback_trading_days": variant["exit_lookback_trading_days"],
            "first_signal_eligible_date": str(signals["date"].iloc[0].date()),
            "signal_count": int(signals["signal_action"].isin(["ENTER", "EXIT"]).sum()),
            "closed_cycle_count": int(len(closed_cycles)),
            "average_holding_trading_days": average_holding,
            "theoretical_base": theoretical_base,
            "theoretical_stress": theoretical_stress,
            "small_account_base": small_base,
            "small_account_stress": small_stress,
            "fixed_exposure_and_matched_benchmarks": fixed_summaries,
            "period_diagnostics": period_result,
        }
        variant_frames[variant_id] = {
            "signals": signals,
            "cycles": cycles,
            "events": events,
            **{f"theoretical_base_{key}": value for key, value in theoretical_frames.items()},
            **{f"theoretical_stress_{key}": value for key, value in theoretical_stress_frames.items()},
            **{f"small_base_{key}": value for key, value in small_frames.items()},
            **{f"small_stress_{key}": value for key, value in small_stress_frames.items()},
            **{f"benchmark_{key}": value for key, value in fixed_frames.items()},
            "annual_relative": period_result["annual"],
        }
        active_returns[variant_id] = active_return_frame(
            small_frames["strategy_ledger"], small_frames["buy_hold_ledger"]
        )

    multiple_testing = multiple_testing_diagnostics(active_returns, config)
    decision = evaluate_decision(
        config,
        variant_payloads["TECH_03_DONCHIAN_55_20_V1"],
        variant_payloads["TECH_03_DONCHIAN_100_50_ROBUSTNESS_V1"],
        multiple_testing,
    )
    report = {
        "project_id": config["protocol"]["project_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_cutoff": config["protocol"]["historical_contamination_cutoff"],
        "evidence_label": "HISTORICALLY_CONTAMINATED",
        "true_forward_start": None,
        "research_only": True,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "manifest_sha256": manifest["manifest_content_sha256"],
        "data_gate": audit,
        "trial_registry": config["trial_registry"],
        "variants": variant_payloads,
        "multiple_testing": multiple_testing,
        "decision": decision,
    }
    serializable_report = _serializable(report)

    for variant_id, frames in variant_frames.items():
        variant_dir = processed_dir / _variant_slug(variant_id)
        for name, frame in frames.items():
            suffix = ".csv" if name in {"events", "cycles"} or name.endswith("trades") else ".parquet"
            _save_frame(variant_dir / f"{name}{suffix}", frame)
    report_json = ROOT / config["artifacts"]["report_json"]
    report_md = ROOT / config["artifacts"]["report_markdown"]
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(
        json.dumps(serializable_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_md.write_text(_markdown_report(serializable_report), encoding="utf-8")

    figure_path = ROOT / config["artifacts"]["report_figure"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    primary_frames = variant_frames["TECH_03_DONCHIAN_55_20_V1"]
    strategy = primary_frames["small_base_strategy_ledger"]
    buy_hold = primary_frames["small_base_buy_hold_ledger"]
    h00300 = primary_frames["small_base_h00300_ledger"]
    signals = primary_frames["signals"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for frame, label in (
        (strategy, "55/20策略（2万元）"),
        (buy_hold, "510300含分红买入持有"),
        (h00300, "H00300全收益"),
    ):
        axes[0].plot(frame["date"], frame["equity"] / frame["equity"].iloc[0], label=label)
    axes[0].set_ylabel("归一化净值")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].step(signals["date"], signals["target_position"], where="post", label="收盘后目标状态")
    axes[1].set_ylabel("研究状态")
    axes[1].set_yticks([0, 1])
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(json.dumps({
        "project_id": report["project_id"],
        "decision": decision,
        "report": str(report_md),
        "artifacts": str(processed_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
