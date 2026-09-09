"""运行冻结的沪深300成分股2万元整仓轮换历史重缩放。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_final_alpha_strategy import (  # noqa: E402
    _benchmark_series,
    build_factor_panel,
    summarize,
)
from research.small_account_cross_sectional import (  # noqa: E402
    SmallAccountCosts,
    load_full_component_history,
    run_small_account_open_backtest,
)
from scripts.freeze_csi300_small_account_alpha_v2 import (  # noqa: E402
    CONFIG_FILE,
    FROZEN_FILES,
    MANIFEST_FILE,
    sha256,
    tree_sha256,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _verify_manifest(config: dict) -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("协议尚未冻结，禁止运行历史结果")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_SMALL_ACCOUNT_RERUN":
        raise RuntimeError("冻结清单状态错误")
    changed: list[str] = []
    for relative, expected in manifest["frozen_files"].items():
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected:
            changed.append(relative)
    if set(manifest["frozen_files"]) != set(FROZEN_FILES):
        changed.append("FROZEN_FILE_SET")
    for relative, expected in manifest["input_files"].items():
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected:
            changed.append(relative)
    history_dir = ROOT / config["inputs"]["component_history_cache"]
    if tree_sha256(history_dir) != manifest["component_history_cache"]:
        changed.append(config["inputs"]["component_history_cache"])
    if changed:
        raise RuntimeError(f"冻结后文件变化：{sorted(set(changed))}")
    return manifest


def _build_targets(factors: pd.DataFrame, config: dict) -> pd.DataFrame:
    start = pd.Timestamp(config["schedule"]["start_date"])
    end = pd.Timestamp(config["schedule"]["end_date"])
    calendar = pd.DatetimeIndex(
        sorted(factors.loc[factors["date"].between(start, end), "date"].unique())
    )
    step = int(config["schedule"]["rebalance_every_trading_days"])
    signal_dates = set(calendar[::step])
    eligible = factors.loc[
        factors["date"].isin(signal_dates)
        & factors["is_index_member"].astype(bool)
        & ~factors["is_suspended"].astype(bool)
        & factors["average_amount_20d"].ge(
            float(config["universe"]["minimum_20d_average_amount_cny"])
        )
    ].copy()
    eligible["selection_score"] = np.where(
        eligible["regime"].eq("BULL"),
        eligible["score_concentrated_bull"],
        eligible["score_defensive"],
    )
    holdings = int(config["selection"]["holdings"])
    selected = (
        eligible.dropna(subset=["selection_score"])
        .sort_values(
            ["date", "selection_score", "con_code"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=True)
        .head(holdings)
        .copy()
    )
    selected["selection_rank"] = selected.groupby("date").cumcount() + 1
    return selected[
        ["date", "con_code", "selection_rank", "selection_score", "regime"]
    ].rename(columns={"date": "signal_date"})


def _costs(config: dict, slippage_key: str) -> SmallAccountCosts:
    account = config["account"]
    costs = config["costs"]
    return SmallAccountCosts(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_sell_rate=float(costs["stamp_duty_sell_rate"]),
        stamp_duty_sell_rate_before_reduction=float(costs["stamp_duty_sell_rate_before_reduction"]),
        stamp_duty_reduction_effective_date=str(costs["stamp_duty_reduction_effective_date"]),
        slippage_bps_per_leg=float(costs[slippage_key]),
        cash_annual_rate=float(account["cash_annual_rate"]),
        lot_size=int(account["lot_size"]),
        minimum_trade_notional_cny=float(account["minimum_trade_notional_cny"]),
        maximum_positions=int(account["maximum_positions"]),
    )


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


def _render(report: dict) -> str:
    lines = [
        "# 沪深300成分股2万元小账户 Alpha V2 历史结果",
        "",
        f"- 总状态：`{report['status']}`",
        "- 证据标签：`HISTORICALLY_CONTAMINATED_RETROSPECTIVE`",
        f"- 本金：{report['account']['initial_cash_cny']:.0f}元",
        f"- 因子数：{report['factor_budget']['used']} / {report['factor_budget']['maximum']}",
        f"- 最低单腿成交：{report['account']['minimum_trade_notional_cny']:.0f}元",
        "",
        "|情景|策略年化|基准年化|年化超额|最大回撤|成交笔数|总成本|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("基础5bp", "base_cost"), ("压力15bp", "stress_cost")):
        item = report[key]
        lines.append(
            f"|{label}|{item['strategy']['cagr']:.2%}|{item['benchmark']['cagr']:.2%}|"
            f"{item['annualized_excess']:.2%}|{item['strategy']['maximum_drawdown']:.2%}|"
            f"{item['trade_rows']}|{item['total_cost_cny']:.2f}元|"
        )
    lines += [
        "",
        f"- 所有成交不低于5000元：`{report['execution_audit']['all_trades_at_least_minimum']}`",
        f"- 最小实际成交：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        f"- 平均持仓数：{report['base_cost']['average_position_count']:.2f}",
        f"- 平均资金暴露：{report['base_cost']['average_exposure']:.2%}",
        f"- 因低于5000元暂缓退出次数：{report['execution_audit']['blocked_small_sell_days']}",
        "",
        "## 结论边界",
        "",
        "这个结果只是对已污染历史的账户重缩放。即使年化超额达到20%，也不能称为严格样本外，不能据此自动生成真实仓位或订单。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = _verify_manifest(config)
    inputs = config["inputs"]
    member_panel = pd.read_parquet(ROOT / inputs["constituent_member_panel"])
    index = pd.read_parquet(ROOT / inputs["index_daily"])
    breadth = pd.read_parquet(ROOT / inputs["breadth"])
    benchmark_raw = pd.read_parquet(ROOT / inputs["benchmark_total_return"])
    factors = build_factor_panel(member_panel, index, breadth)
    targets = _build_targets(factors, config)

    history_dir = ROOT / inputs["component_history_cache"]
    execution_history = load_full_component_history(
        [str(path) for path in sorted(history_dir.glob("*.parquet"))]
    )
    start = pd.Timestamp(config["schedule"]["start_date"])
    end = pd.Timestamp(config["schedule"]["end_date"])
    calendar = pd.DatetimeIndex(
        sorted(member_panel.loc[member_panel["date"].between(start, end), "date"].unique())
    )
    initial_cash = float(config["account"]["initial_cash_cny"])
    gap = float(config["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        execution_history,
        targets,
        calendar,
        initial_cash,
        _costs(config, "base_slippage_bps_per_leg"),
        gap,
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        execution_history,
        targets,
        calendar,
        initial_cash,
        _costs(config, "stress_slippage_bps_per_leg"),
        gap,
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    base = summarize(base_ledger, base_trades, benchmark, initial_cash)
    stress = summarize(stress_ledger, stress_trades, benchmark, initial_cash)
    threshold = float(config["protocol"]["acceptance_minimum"])
    base_pass = base["annualized_excess"] >= threshold
    stress_pass = stress["annualized_excess"] >= threshold
    minimum_trade = float(config["account"]["minimum_trade_notional_cny"])
    all_minimum = bool(base_trades.empty or base_trades["notional"].ge(minimum_trade - 1e-8).all())
    report = _json_safe(
        {
            "project_id": config["protocol"]["project_id"],
            "version": config["protocol"]["version"],
            "status": "RETROSPECTIVE_PASS_BASE_AND_STRESS" if base_pass and stress_pass else (
                "RETROSPECTIVE_PASS_BASE_ONLY" if base_pass else "RETROSPECTIVE_FAIL"
            ),
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "evidence_label": config["governance"]["historical_result_label"],
            "manifest_sha256": sha256(MANIFEST_FILE),
            "frozen_at": manifest["frozen_at"],
            "data_period": {"start": str(start.date()), "end": str(end.date())},
            "factor_budget": config["factor_budget"],
            "account": config["account"],
            "costs": config["costs"],
            "acceptance": {
                "annualized_excess_minimum": threshold,
                "base_pass": base_pass,
                "stress_pass": stress_pass,
                "strict_holdout_pass": False,
            },
            "base_cost": base,
            "stress_cost": stress,
            "execution_audit": {
                "all_trades_at_least_minimum": all_minimum,
                "minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else None,
                "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()),
                "maximum_position_count": int(base_ledger["position_count"].max()),
                "signal_to_execution": "NEXT_TRADING_DAY_OPEN",
                "t_plus_one_enforced": True,
            },
            "safety": {
                "paper_signal": "DISABLED",
                "position_mapping": "DISABLED",
                "order_generation": "DISABLED",
                "broker_connection": "DISABLED",
                "live_trading": "NOT_AUTHORIZED",
            },
        }
    )
    outputs = config["outputs"]
    for frame, key in (
        (base_ledger, "base_ledger"),
        (base_trades, "base_trades"),
        (stress_ledger, "stress_ledger"),
        (stress_trades, "stress_trades"),
    ):
        _atomic_parquet(frame, ROOT / outputs[key])
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / outputs["result_json"])
    _atomic_text(_render(report), ROOT / outputs["result_markdown"])
    print(
        json.dumps(
            {
                "状态": report["status"],
                "基础年化超额": base["annualized_excess"],
                "压力年化超额": stress["annualized_excess"],
                "最小成交": report["execution_audit"]["minimum_trade_notional_cny"],
                "严格样本外通过": False,
                "结果": str(ROOT / outputs["result_json"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

