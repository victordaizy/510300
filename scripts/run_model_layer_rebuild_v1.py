"""运行已冻结的510300模型层重建历史诊断。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.engine import run_long_cash_backtest
from research.model_layer_rebuild_v1 import (
    FractionalCosts,
    audit_and_load_inputs,
    build_all_technical_targets,
    build_buy_hold_targets,
    build_event_audit,
    build_fixed_mix_benchmark,
    build_h00300_benchmark,
    build_r5_component_targets,
    build_valuation_engineering_audit,
    classify_model,
    load_config,
    make_execution_costs,
    multiple_testing_diagnostics,
    relative_metrics,
    shapley_r5,
    simulate_fractional_targets,
    stability_diagnostics,
    summarize,
    validate_manifest,
)


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _execution_market(market: pd.DataFrame) -> pd.DataFrame:
    return market[["date", "etf_open", "etf_high", "etf_low", "etf_close"]].rename(
        columns={
            "etf_open": "open",
            "etf_high": "high",
            "etf_low": "low",
            "etf_close": "close",
        }
    )


def _run_execution(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    stress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    costs = make_execution_costs(config, stress)
    ledger, trades = run_long_cash_backtest(
        prices=_execution_market(market),
        dividends=dividends,
        targets=targets,
        initial_cash=float(config["execution"]["initial_cash_cny"]),
        costs=costs,
        start_date=config["protocol"]["evaluation_start"],
        end_date=config["protocol"]["evaluation_end"],
        minimum_trade_shares=int(config["execution"]["minimum_ordinary_trade_shares"]),
    )
    summary = summarize(
        ledger,
        trades,
        float(config["execution"]["initial_cash_cny"]),
        config,
        minimum_commission=costs.minimum_commission_cny,
    )
    return ledger, trades, summary


def _run_theoretical(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
    stress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    execution = config["execution"]
    multiplier = float(execution["stress_multiplier"]) if stress else 1.0
    costs = FractionalCosts(
        commission_rate=float(execution["commission_rate"]) * multiplier,
        slippage_bps=float(execution["slippage_bps_per_leg"]) * multiplier,
        cash_annual_rate=float(execution["cash_annual_rate"]),
        trading_days_per_year=int(execution["trading_days_per_year"]),
    )
    ledger, trades = simulate_fractional_targets(
        market,
        dividends,
        targets,
        float(execution["theoretical_initial_wealth"]),
        costs,
    )
    summary = summarize(
        ledger,
        trades,
        float(execution["theoretical_initial_wealth"]),
        config,
        minimum_commission=0.0,
    )
    return ledger, trades, summary


def _model_payload(
    identifier: str,
    targets: pd.DataFrame,
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    benchmarks: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    config: dict[str, Any],
    processed: Path,
    family: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    execution_base, trades_base, summary_base = _run_execution(
        market, dividends, targets, config, stress=False
    )
    execution_stress, trades_stress, summary_stress = _run_execution(
        market, dividends, targets, config, stress=True
    )
    theory_base, theory_trades, theory_summary = _run_theoretical(
        market, dividends, targets, config, stress=False
    )
    theory_stress, theory_stress_trades, theory_stress_summary = _run_theoretical(
        market, dividends, targets, config, stress=True
    )

    buy_execution_base, _, buy_execution_base_summary = benchmarks["execution_base"]
    buy_execution_stress, _, buy_execution_stress_summary = benchmarks["execution_stress"]
    buy_theory_base, _, buy_theory_base_summary = benchmarks["theoretical_base"]
    buy_theory_stress, _, buy_theory_stress_summary = benchmarks["theoretical_stress"]

    relative_base, active_frame = relative_metrics(execution_base, buy_execution_base, config)
    relative_stress, _ = relative_metrics(execution_stress, buy_execution_stress, config)
    theory_relative, _ = relative_metrics(theory_base, buy_theory_base, config)
    theory_stress_relative, _ = relative_metrics(theory_stress, buy_theory_stress, config)
    stability = stability_diagnostics(execution_base, buy_execution_base, config)
    initial_decision = classify_model(
        summary_base,
        buy_execution_base_summary,
        relative_base,
        relative_stress,
        stability,
        config,
    )

    slug = identifier.lower()
    targets.to_parquet(processed / f"{slug}_targets.parquet", index=False)
    execution_base.to_parquet(processed / f"{slug}_execution_base_ledger.parquet", index=False)
    trades_base.to_csv(processed / f"{slug}_execution_base_trades.csv", index=False, encoding="utf-8-sig")
    execution_stress.to_parquet(processed / f"{slug}_execution_stress_ledger.parquet", index=False)
    theory_base.to_parquet(processed / f"{slug}_theoretical_base_ledger.parquet", index=False)
    theory_trades.to_csv(processed / f"{slug}_theoretical_base_trades.csv", index=False, encoding="utf-8-sig")
    theory_stress.to_parquet(processed / f"{slug}_theoretical_stress_ledger.parquet", index=False)
    theory_stress_trades.to_csv(
        processed / f"{slug}_theoretical_stress_trades.csv", index=False, encoding="utf-8-sig"
    )
    event_audit = build_event_audit(targets, trades_base)
    event_audit.to_csv(processed / f"{slug}_event_audit.csv", index=False, encoding="utf-8-sig")

    payload = {
        "id": identifier,
        "family": family,
        "target_levels_observed": sorted(float(value) for value in targets["target_position"].dropna().unique()),
        "signal_change_count": int(targets["trade_allowed"].astype(bool).sum()),
        "execution_base": {
            "strategy": summary_base,
            "buy_hold": buy_execution_base_summary,
            "relative_vs_510300": relative_base,
        },
        "execution_stress": {
            "strategy": summary_stress,
            "buy_hold": buy_execution_stress_summary,
            "relative_vs_510300": relative_stress,
        },
        "theoretical_base": {
            "strategy": theory_summary,
            "buy_hold": buy_theory_base_summary,
            "relative_vs_510300": theory_relative,
        },
        "theoretical_stress": {
            "strategy": theory_stress_summary,
            "buy_hold": buy_theory_stress_summary,
            "relative_vs_510300": theory_stress_relative,
        },
        "implementation_gap": {
            "execution_minus_theoretical_cagr": summary_base["cagr"] - theory_summary["cagr"],
            "execution_minus_theoretical_active_return": relative_base["annualized_active_return"]
            - theory_relative["annualized_active_return"],
            "execution_minus_theoretical_average_exposure": summary_base["average_exposure"]
            - theory_summary["average_exposure"],
        },
        "stability": stability,
        "decision_before_multiple_testing": initial_decision,
    }
    return payload, active_frame, execution_base


def _apply_multiple_testing_decisions(
    models: dict[str, dict[str, Any]], diagnostics: dict[str, Any], config: dict[str, Any]
) -> None:
    settings = config["evaluation"]["multiple_testing"]
    family_checks = {
        "white_reality_check_5pct": diagnostics["white_reality_check"]["p_value"]
        <= float(settings["significance_level"]),
        "hansen_spa_5pct": diagnostics["hansen_spa"]["p_value"]
        <= float(settings["significance_level"]),
        "pbo_at_most_25pct": diagnostics["pbo"]["value"] <= float(settings["pbo_maximum"]),
    }
    for identifier, payload in models.items():
        dsr = diagnostics["deflated_sharpe_ratio"][identifier]
        checks = {
            **family_checks,
            "deflated_sharpe_probability_95pct": dsr["probability"]
            >= float(settings["deflated_sharpe_probability_minimum"]),
        }
        before = payload["decision_before_multiple_testing"]
        if before["status"] == "HISTORICAL_SCREEN_PASS_FORWARD_REQUIRED":
            status = (
                "HISTORICAL_SCREEN_PASS_FORWARD_REQUIRED"
                if all(checks.values())
                else "REJECT_HISTORICAL_MULTIPLE_TESTING"
            )
        else:
            status = before["status"]
        payload["decision"] = {
            **before,
            "status": status,
            "multiple_testing_checks": checks,
            "alpha_pass": False,
            "alpha_pass_blocker": "TRUE_OOS_NOT_STARTED",
        }


def _r5_interactions(values: dict[tuple[bool, bool, bool], float]) -> dict[str, float]:
    zero = values[(False, False, False)]
    pairwise = {
        "VxT": values[(True, True, False)] - values[(True, False, False)] - values[(False, True, False)] + zero,
        "VxR": values[(True, False, True)] - values[(True, False, False)] - values[(False, False, True)] + zero,
        "TxR": values[(False, True, True)] - values[(False, True, False)] - values[(False, False, True)] + zero,
    }
    three_way = (
        values[(True, True, True)]
        - values[(True, True, False)]
        - values[(True, False, True)]
        - values[(False, True, True)]
        + values[(True, False, False)]
        + values[(False, True, False)]
        + values[(False, False, True)]
        - zero
    )
    return {**pairwise, "VxTxR": three_way}


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    tech = report["technical_models"]
    r5 = report["r5_decomposition"]["variants"]
    lines = [
        "# 510300模型层重建V1：统一历史诊断",
        "",
        f"- 评价期：{report['evaluation_period']['start']} 至 {report['evaluation_period']['end']}（{report['evaluation_period']['trading_days']}日）",
        "- 证据标签：`HISTORICALLY_CONTAMINATED / TRUE_OOS_NOT_STARTED`",
        "- 研究边界：`RESEARCH_ONLY / NO_POSITION_CHANGE / NO_ORDER`",
        f"- 技术数据闸门：`{report['data_gate']['technical_gate']['status']}`",
        f"- 估值数据闸门：`{report['data_gate']['valuation_gate']['status']}`",
        f"- R5原始重放：`{report['data_gate']['r5_replay_gate']['raw_replay_status']}`；拆分输入：`{report['data_gate']['r5_replay_gate']['decomposition_input_status']}`",
        "",
        "## 结论先行",
        "",
        f"- 组合研究：`{report['combination_gate']['status']}`。{report['combination_gate']['reason']}",
        f"- 技术家族多重检验：White p={report['multiple_testing']['white_reality_check']['p_value']:.4f}，SPA p={report['multiple_testing']['hansen_spa']['p_value']:.4f}，PBO={report['multiple_testing']['pbo']['value']:.2%}。",
        "- VAL-01/VAL-02只完成公式和覆盖率审计，未生成信号、收益、IC或仓位结论。",
        "",
        "## 第一批独立技术模型（2万元执行层，净成本）",
        "",
        "| 模型 | CAGR | 年化主动 | IR | Sharpe | 最大回撤 | 曝险 | 交易数 | 双倍成本主动 | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for identifier, payload in tech.items():
        base = payload["execution_base"]
        strategy = base["strategy"]
        relative = base["relative_vs_510300"]
        stress = payload["execution_stress"]["relative_vs_510300"]
        lines.append(
            f"| {identifier} | {_format_pct(strategy['cagr'])} | {_format_pct(relative['annualized_active_return'])} | {relative['information_ratio']:.2f} | {strategy['sharpe_excess_cash']:.2f} | {_format_pct(strategy['maximum_drawdown'])} | {_format_pct(strategy['average_exposure'])} | {strategy['trade_count']} | {_format_pct(stress['annualized_active_return'])} | `{payload['decision']['status']}` |"
        )
    lines.extend(
        [
            "",
            "### 理论层与2万元包装差异",
            "",
            "| 模型 | 理论CAGR | 执行CAGR | CAGR差 | 理论主动 | 执行主动 | 主动差 | 曝险差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for identifier, payload in tech.items():
        theory = payload["theoretical_base"]
        execution = payload["execution_base"]
        gap = payload["implementation_gap"]
        lines.append(
            f"| {identifier} | {_format_pct(theory['strategy']['cagr'])} | {_format_pct(execution['strategy']['cagr'])} | {_format_pct(gap['execution_minus_theoretical_cagr'])} | {_format_pct(theory['relative_vs_510300']['annualized_active_return'])} | {_format_pct(execution['relative_vs_510300']['annualized_active_return'])} | {_format_pct(gap['execution_minus_theoretical_active_return'])} | {_format_pct(gap['execution_minus_theoretical_average_exposure'])} |"
        )
    lines.extend(
        [
            "",
            "### 2万元执行与成本",
            "",
            "| 模型 | 调仓腿数 | 平均持仓段（日） | 换手/平均权益 | 佣金（元） | 滑点（元） | 总成本（元） | 最低佣金占比 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for identifier, payload in tech.items():
        strategy = payload["execution_base"]["strategy"]
        lines.append(
            f"| {identifier} | {strategy['trade_count']} | {strategy['average_positive_exposure_spell_trading_days']:.1f} | {strategy['turnover_over_average_equity']:.2f} | {strategy['commission']:.2f} | {strategy['slippage_cost']:.2f} | {strategy['total_execution_cost']:.2f} | {_format_pct(strategy['minimum_commission_trade_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "### 稳定性与多重检验",
            "",
            "| 模型 | H1主动 | H2主动 | 正主动年度占比 | 剔除最佳年后年化主动 | 正贡献最大年占比 | 242日滚动为正 | 484日滚动为正 | DSR概率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    dsr = report["multiple_testing"]["deflated_sharpe_ratio"]
    for identifier, payload in tech.items():
        stability = payload["stability"]
        halves = stability["chronological_halves"]
        rolling = payload["execution_base"]["relative_vs_510300"]["rolling_active_returns"]
        lines.append(
            f"| {identifier} | {_format_pct(halves[0]['active_return'])} | {_format_pct(halves[1]['active_return'])} | {_format_pct(stability['positive_year_ratio'])} | {_format_pct(stability['annualized_active_return_without_best_year'])} | {_format_pct(stability['maximum_positive_year_contribution_share'])} | {_format_pct(rolling['242']['positive_ratio'])} | {_format_pct(rolling['484']['positive_ratio'])} | {_format_pct(dsr[identifier]['probability'])} |"
        )
    lines.extend(
        [
            "",
            "White和SPA是全家族门槛，不能解释为某个单模型的显著性；PBO使用8折CSCV的70个对称拆分。所有候选还被`TRUE_OOS_NOT_STARTED`统一阻断。",
            "",
            "### 逐模型未通过门槛",
            "",
        ]
    )
    for identifier, payload in tech.items():
        decision = payload["decision"]
        failed = [
            key
            for key, passed in {
                **decision["historical_checks"],
                **decision["risk_overlay_checks"],
                **decision["multiple_testing_checks"],
            }.items()
            if not passed
        ]
        lines.append(f"- `{identifier}`：{', '.join(failed)}")
    lines.extend(
        [
            "",
            "## R5八路拆分（相同2万元包装）",
            "",
            "| 变体 | CAGR | 年化主动 | IR | Sharpe | 最大回撤 | 平均曝险 | 交易数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for identifier, payload in r5.items():
        strategy = payload["execution_base"]["strategy"]
        relative = payload["execution_base"]["relative_vs_510300"]
        information_ratio = relative["information_ratio"]
        lines.append(
            f"| {identifier} | {_format_pct(strategy['cagr'])} | {_format_pct(relative['annualized_active_return'])} | {'—' if information_ratio is None else f'{information_ratio:.2f}'} | {strategy['sharpe_excess_cash']:.2f} | {_format_pct(strategy['maximum_drawdown'])} | {_format_pct(strategy['average_exposure'])} | {strategy['trade_count']} |"
        )
    lines.extend(
        [
            "",
            "### R5归因",
            "",
            "Shapley值按八个组合的年化主动收益计算；它是描述性归因，不是因果识别。",
            "",
        ]
    )
    for key, value in report["r5_decomposition"]["shapley_annualized_active_return"].items():
        lines.append(f"- {key}：{_format_pct(value)}")
    lines.extend(["", "交互项（组合主动收益的离散差分）：", ""])
    for key, value in report["r5_decomposition"]["interaction_annualized_active_return"].items():
        lines.append(f"- {key}：{_format_pct(value)}")
    lines.extend(
        [
            "",
            "## 估值线为何是NO_VIEW",
            "",
        ]
    )
    for reason in report["data_gate"]["valuation_gate"]["failure_categories"]:
        lines.append(f"- `{reason}`")
    lines.extend(
        [
            "",
            "日度供应商PE虽可计算公式，但不能证明每个历史截面是当时可见且未经未来修订；点时月度财务仅有60个月，无法在2021-08-12评价起点提供5年/7年暖机。因此任何估值回报或组合结果都会违反点时性闸门。",
            "",
            "## 固定仓位与主基准",
            "",
            "| 基准 | CAGR | Sharpe | 最大回撤 |",
            "|---|---:|---:|---:|",
        ]
    )
    for identifier, summary in report["benchmarks"]["theoretical_fixed_exposure"].items():
        lines.append(
            f"| {identifier} | {_format_pct(summary['cagr'])} | {summary['sharpe_excess_cash']:.2f} | {_format_pct(summary['maximum_drawdown'])} |"
        )
    lines.extend(
        [
            f"| 510300含分红买入持有（2万元执行） | {_format_pct(report['benchmarks']['execution_buy_hold_base']['cagr'])} | {report['benchmarks']['execution_buy_hold_base']['sharpe_excess_cash']:.2f} | {_format_pct(report['benchmarks']['execution_buy_hold_base']['maximum_drawdown'])} |",
            f"| 510300含分红买入持有（理论） | {_format_pct(report['benchmarks']['theoretical_buy_hold_base']['cagr'])} | {report['benchmarks']['theoretical_buy_hold_base']['sharpe_excess_cash']:.2f} | {_format_pct(report['benchmarks']['theoretical_buy_hold_base']['maximum_drawdown'])} |",
            f"| H00300全收益 | {_format_pct(report['benchmarks']['h00300']['cagr'])} | {report['benchmarks']['h00300']['sharpe_excess_cash']:.2f} | {_format_pct(report['benchmarks']['h00300']['maximum_drawdown'])} |",
            "",
            "## 解释限制",
            "",
            "- 整段历史已被研究过程观察，所有结果只能称历史诊断，不能称样本外Alpha。",
            "- R5八路拆分依赖冻结派生信号快照；当前原始成分市值源漂移导致无法原始重放。",
            "- 多重检验是近似实现；结论同时受真实前向尚未开始这一更强约束阻断。",
            "- 本报告不输出当日观点、目标仓位或订单。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_figure(
    ledgers: dict[str, pd.DataFrame], buy_hold: pd.DataFrame, output: Path
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    base = float(buy_hold["equity"].iloc[0])
    axes[0].plot(buy_hold["date"], buy_hold["equity"] / base, label="510300含分红买入持有", linewidth=2.2, color="black")
    for identifier, ledger in ledgers.items():
        axes[0].plot(ledger["date"], ledger["equity"] / ledger["equity"].iloc[0], label=identifier, alpha=0.8)
    axes[0].set_title("第一批独立技术模型：2万元执行层归一化净值")
    axes[0].set_ylabel("归一化净值")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    relative_base = buy_hold["equity"] / buy_hold["equity"].iloc[0]
    for identifier, ledger in ledgers.items():
        relative = (ledger["equity"] / ledger["equity"].iloc[0]) / relative_base - 1.0
        axes[1].plot(ledger["date"], relative, label=identifier, alpha=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("相对510300含分红买入持有")
    axes[1].set_ylabel("累计主动收益")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    config = load_config()
    manifest = validate_manifest(ROOT, config)
    datasets, audit = audit_and_load_inputs(ROOT, config)
    if audit["technical_gate"]["status"] != "PASS":
        raise RuntimeError("技术数据闸门未通过，禁止计算收益")
    if audit["valuation_gate"]["return_calculation_allowed"]:
        raise RuntimeError("估值闸门异常：本冻结版本不允许估值收益计算")

    artifacts = config["artifacts"]
    processed = ROOT / artifacts["processed_directory"]
    processed.mkdir(parents=True, exist_ok=True)
    (ROOT / artifacts["report_json"]).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / artifacts["figure"]).parent.mkdir(parents=True, exist_ok=True)
    data_gate_path = ROOT / artifacts["data_gate_json"]
    data_gate_path.parent.mkdir(parents=True, exist_ok=True)
    data_gate_path.write_text(
        json.dumps(_serializable(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    valuation_frame, valuation_audit = build_valuation_engineering_audit(datasets, config)
    valuation_frame[["date", "pe_ttm", "raw_ey", "normalized_growth", "normalized_ey", "cgb_10y", "normalized_ey_spread"]].to_parquet(
        processed / "valuation_formula_engineering_only_no_signal.parquet", index=False
    )

    market = datasets["evaluation_market"]
    dividends = datasets["dividends"]
    buy_targets = build_buy_hold_targets(market["date"])
    benchmarks = {
        "execution_base": _run_execution(market, dividends, buy_targets, config, False),
        "execution_stress": _run_execution(market, dividends, buy_targets, config, True),
        "theoretical_base": _run_theoretical(market, dividends, buy_targets, config, False),
        "theoretical_stress": _run_theoretical(market, dividends, buy_targets, config, True),
    }
    h00300 = build_h00300_benchmark(
        market,
        float(config["execution"]["theoretical_initial_wealth"]),
        float(config["execution"]["cash_annual_rate"]),
        int(config["execution"]["trading_days_per_year"]),
    )
    h00300_summary = summarize(
        h00300,
        pd.DataFrame(),
        float(config["execution"]["theoretical_initial_wealth"]),
        config,
    )
    fixed_summaries = {}
    for exposure in (0.25, 0.50, 0.75, 1.00):
        ledger = build_fixed_mix_benchmark(
            market,
            dividends,
            exposure,
            float(config["execution"]["theoretical_initial_wealth"]),
            float(config["execution"]["cash_annual_rate"]),
            int(config["execution"]["trading_days_per_year"]),
        )
        identifier = f"FIXED_{int(exposure * 100)}PCT"
        fixed_summaries[identifier] = summarize(
            ledger,
            pd.DataFrame(),
            float(config["execution"]["theoretical_initial_wealth"]),
            config,
        )
        ledger.to_parquet(processed / f"{identifier.lower()}_theoretical_ledger.parquet", index=False)

    technical_targets = build_all_technical_targets(datasets, config)
    technical_payloads: dict[str, dict[str, Any]] = {}
    active_frames: dict[str, pd.DataFrame] = {}
    technical_ledgers: dict[str, pd.DataFrame] = {}
    family_by_id = {item["id"]: item["family"] for item in config["technical_models"]}
    for identifier, targets in technical_targets.items():
        payload, active, ledger = _model_payload(
            identifier,
            targets,
            market,
            dividends,
            benchmarks,
            config,
            processed,
            family_by_id[identifier],
        )
        technical_payloads[identifier] = payload
        active_frames[identifier] = active
        technical_ledgers[identifier] = ledger
    multiple = multiple_testing_diagnostics(active_frames, config)
    _apply_multiple_testing_decisions(technical_payloads, multiple, config)

    r5_targets = build_r5_component_targets(datasets, config, ROOT)
    r5_payloads: dict[str, dict[str, Any]] = {}
    r5_values: dict[tuple[bool, bool, bool], float] = {}
    r5_by_id = {item["id"]: item for item in config["r5_decomposition"]["variants"]}
    for identifier, targets in r5_targets.items():
        payload, _, _ = _model_payload(
            identifier,
            targets,
            market,
            dividends,
            benchmarks,
            config,
            processed,
            "R5_COMPONENT_DECOMPOSITION",
        )
        variant = r5_by_id[identifier]
        mask = (bool(variant["valuation"]), bool(variant["trend"]), bool(variant["risk"]))
        r5_values[mask] = payload["execution_base"]["relative_vs_510300"]["annualized_active_return"]
        payload["component_mask"] = {"V": mask[0], "T": mask[1], "R": mask[2]}
        payload["decision"] = {
            "status": "HISTORICAL_COMPONENT_ATTRIBUTION_ONLY",
            "alpha_pass": False,
            "blockers": ["TRUE_OOS_NOT_STARTED", audit["r5_replay_gate"]["raw_replay_status"]],
        }
        r5_payloads[identifier] = payload

    report = {
        "project_id": config["protocol"]["project_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "evaluation_period": {
            "start": config["protocol"]["evaluation_start"],
            "end": config["protocol"]["evaluation_end"],
            "trading_days": int(len(market)),
        },
        "evidence_label": "HISTORICALLY_CONTAMINATED",
        "true_forward_start": None,
        "research_only": True,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "data_gate": audit,
        "valuation_models": valuation_audit,
        "benchmarks": {
            "execution_buy_hold_base": benchmarks["execution_base"][2],
            "execution_buy_hold_stress": benchmarks["execution_stress"][2],
            "theoretical_buy_hold_base": benchmarks["theoretical_base"][2],
            "h00300": h00300_summary,
            "theoretical_fixed_exposure": fixed_summaries,
        },
        "technical_trial_registry": {
            "registered_trial_count": int(config["evaluation"]["multiple_testing"]["registered_technical_trial_count"]),
            "return_tested_candidate_count": len(technical_payloads),
            "unrun_registered_trial": "T0_DONCHIAN_20_10_SUPERSEDED_BEFORE_RETURN_RUN",
        },
        "technical_models": technical_payloads,
        "multiple_testing": multiple,
        "r5_decomposition": {
            "source_status": audit["r5_replay_gate"],
            "variants": r5_payloads,
            "shapley_annualized_active_return": shapley_r5(r5_values),
            "interaction_annualized_active_return": _r5_interactions(r5_values),
        },
        "combination_gate": {
            "status": "STOPPED_NO_COMBINATION_TEST",
            "reason": "VAL-01与VAL-02均为NO_VIEW，独立估值线未通过数据闸门；按冻结协议禁止组合优化。",
            "combination_returns_calculated": False,
        },
        "safety": {
            "current_view_emitted": False,
            "position_target_emitted": False,
            "order_emitted": False,
            "broker_accessed": False,
        },
    }
    serializable = _serializable(report)
    (ROOT / artifacts["report_json"]).write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(serializable, ROOT / artifacts["report_markdown"])
    _save_figure(
        technical_ledgers,
        benchmarks["execution_base"][0],
        ROOT / artifacts["figure"],
    )
    print(
        json.dumps(
            {
                "status": "COMPLETED_RESEARCH_ONLY",
                "report": artifacts["report_markdown"],
                "technical_models": {
                    identifier: payload["decision"]["status"]
                    for identifier, payload in serializable["technical_models"].items()
                },
                "valuation_status": valuation_audit["formal_status"],
                "combination_status": serializable["combination_gate"]["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
