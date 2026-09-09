"""将已冻结的2万元父公式原样运行于2015至2021早期外部时段。"""

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
    _return_metrics,
    build_factor_panel,
    summarize,
)
from research.small_account_cross_sectional import (  # noqa: E402
    SmallAccountCosts,
    run_small_account_open_backtest,
)
from scripts.freeze_csi300_small_account_external_2015_2021 import (  # noqa: E402
    CONFIG_FILE,
    FROZEN_FILES,
    MANIFEST_FILE,
    sha256,
)
from scripts.run_csi300_small_account_alpha_v2 import _build_targets  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def build_external_breadth(panel: pd.DataFrame) -> pd.DataFrame:
    """用同日点时成员计算等权MA60广度。"""

    data = panel[["date", "con_code", "total_return_close", "is_index_member"]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(["con_code", "date"], inplace=True)
    data["ma60"] = (
        data.groupby("con_code", sort=False)["total_return_close"]
        .rolling(60, min_periods=60)
        .mean()
        .reset_index(level=0, drop=True)
    )
    data["above_ma60"] = data["total_return_close"].gt(data["ma60"])
    active = data.loc[data["is_index_member"].astype(bool)].copy()
    result = active.groupby("date", as_index=False).agg(
        equal_above_ma60_share=("above_ma60", "mean"),
        member_count=("con_code", "nunique"),
    )
    return result


def _verify_manifest(contract: dict) -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError("外部验证尚未冻结")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_EXTERNAL_PERIOD_RETURN_READ":
        raise RuntimeError("外部验证冻结状态错误")
    changed: list[str] = []
    for relative, expected in manifest["frozen_files"].items():
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected:
            changed.append(relative)
    if set(manifest["frozen_files"]) != set(FROZEN_FILES):
        changed.append("FROZEN_FILE_SET")
    for relative, expected in manifest["input_files"].items():
        if not (ROOT / relative).exists() or sha256(ROOT / relative) != expected:
            changed.append(relative)
    if changed:
        raise RuntimeError(f"外部验证冻结后变化：{sorted(set(changed))}")
    return manifest


def _costs(contract: dict, key: str) -> SmallAccountCosts:
    account = contract["account"]
    costs = contract["costs"]
    return SmallAccountCosts(
        commission_rate=float(costs["commission_rate"]),
        minimum_commission_cny=float(costs["minimum_commission_cny"]),
        stamp_duty_sell_rate=float(costs["stamp_duty_sell_rate"]),
        stamp_duty_sell_rate_before_reduction=float(costs["stamp_duty_sell_rate_before_reduction"]),
        stamp_duty_reduction_effective_date=str(costs["stamp_duty_reduction_effective_date"]),
        slippage_bps_per_leg=float(costs[key]),
        cash_annual_rate=float(account["cash_annual_rate"]),
        lot_size=int(account["lot_size"]),
        minimum_trade_notional_cny=float(account["minimum_trade_notional_cny"]),
        maximum_positions=int(account["maximum_positions"]),
    )


def _paired_bootstrap(strategy: pd.Series, benchmark: pd.Series, contract: dict) -> dict:
    repetitions = int(contract["evaluation"]["bootstrap_repetitions"])
    block = int(contract["evaluation"]["bootstrap_block_length_trading_days"])
    rng = np.random.default_rng(int(contract["evaluation"]["random_seed"]))
    left = strategy.to_numpy(dtype=float)
    right = benchmark.to_numpy(dtype=float)
    n = len(left)
    starts = np.arange(n - block + 1)
    values = np.empty(repetitions)
    count = int(np.ceil(n / block))
    for repetition in range(repetitions):
        chosen = rng.choice(starts, size=count, replace=True)
        indices = np.concatenate([np.arange(start, start + block) for start in chosen])[:n]
        strategy_cagr = np.expm1(np.log1p(left[indices]).mean() * 242)
        benchmark_cagr = np.expm1(np.log1p(right[indices]).mean() * 242)
        values[repetition] = strategy_cagr - benchmark_cagr
    return {
        "repetitions": repetitions,
        "block_length_trading_days": block,
        "median": float(np.median(values)),
        "interval_95pct": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def _periods(ledger: pd.DataFrame, benchmark: pd.DataFrame, contract: dict) -> dict:
    result = {}
    for period in contract["evaluation"]["predefined_periods"]:
        mask = ledger["date"].between(pd.Timestamp(period["start"]), pd.Timestamp(period["end"]))
        strategy = _return_metrics(ledger.loc[mask, "daily_return"], ledger.loc[mask, "date"])
        bench = _return_metrics(benchmark.loc[mask, "daily_return"], benchmark.loc[mask, "date"])
        result[period["id"]] = {
            "start": period["start"],
            "end": period["end"],
            "strategy": strategy,
            "benchmark": bench,
            "annualized_excess": strategy["cagr"] - bench["cagr"],
        }
    return result


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
        "# 沪深300小账户 Alpha V2 2015—2021外部时段结果",
        "",
        f"- 总状态：`{report['status']}`",
        "- 父公式修改：`false`",
        "- 正式验证期每日成员：300",
        "- 成员价格缺口：0",
        "",
        "|情景|策略年化|基准年化|年化超额|最大回撤|成交笔数|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("基础5bp", "base_cost"), ("压力15bp", "stress_cost")):
        item = report[key]
        lines.append(
            f"|{label}|{item['strategy']['cagr']:.2%}|{item['benchmark']['cagr']:.2%}|"
            f"{item['annualized_excess']:.2%}|{item['strategy']['maximum_drawdown']:.2%}|{item['trade_rows']}|"
        )
    lines += [
        "",
        f"- Bootstrap年化超额95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 最小成交：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{name}`" for name, value in report["gates"].items())
    lines += ["", "结果后禁止在该区间修改父公式补救；无论是否通过，均不自动生成仓位或订单。", ""]
    return "\n".join(lines)


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = _verify_manifest(contract)
    inputs = contract["inputs"]
    panel = pd.read_parquet(ROOT / inputs["external_member_panel"])
    index = pd.read_parquet(ROOT / inputs["external_price_index"])
    benchmark_raw = pd.read_parquet(ROOT / inputs["external_benchmark"])
    breadth = build_external_breadth(panel)
    if not breadth.loc[
        breadth["date"].between(pd.Timestamp(contract["schedule"]["start_date"]), pd.Timestamp(contract["schedule"]["end_date"])),
        "member_count",
    ].eq(300).all():
        raise RuntimeError("外部验证广度每日成员不是300")
    factors = build_factor_panel(panel, index, breadth)
    targets = _build_targets(factors, contract)
    start = pd.Timestamp(contract["schedule"]["start_date"])
    end = pd.Timestamp(contract["schedule"]["end_date"])
    calendar = pd.DatetimeIndex(sorted(panel.loc[panel["date"].between(start, end), "date"].unique()))
    initial_cash = float(contract["account"]["initial_cash_cny"])
    gap = float(contract["universe"]["maximum_open_total_return_gap_for_trade"])
    base_ledger, base_trades = run_small_account_open_backtest(
        panel, targets, calendar, initial_cash, _costs(contract, "base_slippage_bps_per_leg"), gap
    )
    stress_ledger, stress_trades = run_small_account_open_backtest(
        panel, targets, calendar, initial_cash, _costs(contract, "stress_slippage_bps_per_leg"), gap
    )
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    base = summarize(base_ledger, base_trades, benchmark, initial_cash)
    stress = summarize(stress_ledger, stress_trades, benchmark, initial_cash)
    bootstrap = _paired_bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    periods = _periods(base_ledger, benchmark, contract)
    positive_periods = sum(item["annualized_excess"] > 0 for item in periods.values())
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= float(contract["evaluation"]["annualized_excess_minimum"]),
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"]),
        "bootstrap_lower_bound_positive": bootstrap["interval_95pct"][0] > float(contract["evaluation"]["bootstrap_lower_bound_strictly_above"]),
        "positive_predefined_periods": positive_periods >= int(contract["evaluation"]["positive_predefined_periods_minimum"]),
        "all_trades_at_least_5000": bool(base_trades.empty or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()),
    }
    status = "EXTERNAL_PERIOD_PASS_AWAITING_TRUE_FORWARD" if all(gates.values()) else "EXTERNAL_PERIOD_REJECTED_FROZEN"
    report = _json_safe(
        {
            "project_id": contract["protocol"]["project_id"],
            "status": status,
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "manifest_sha256": sha256(MANIFEST_FILE),
            "frozen_at": manifest["frozen_at"],
            "parent_formula_unchanged": True,
            "data_gate": manifest["data_gate"],
            "base_cost": base,
            "stress_cost": stress,
            "bootstrap_annualized_excess": bootstrap,
            "predefined_periods": periods,
            "positive_predefined_periods": positive_periods,
            "gates": gates,
            "execution_audit": {
                "minimum_trade_notional_cny": float(base_trades["notional"].min()) if not base_trades.empty else None,
                "maximum_position_count": int(base_ledger["position_count"].max()),
                "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()),
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
    outputs = contract["outputs"]
    for frame, key in (
        (breadth, "breadth"),
        (base_ledger, "base_ledger"),
        (base_trades, "base_trades"),
        (stress_ledger, "stress_ledger"),
        (stress_trades, "stress_trades"),
    ):
        _atomic_parquet(frame, ROOT / outputs[key])
    _atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / outputs["result_json"])
    _atomic_text(_render(report), ROOT / outputs["result_markdown"])
    print(json.dumps({"状态": status, "基础年化超额": base["annualized_excess"], "压力年化超额": stress["annualized_excess"], "Bootstrap下界": bootstrap["interval_95pct"][0], "正超额分段": positive_periods, "结果": str(ROOT / outputs["result_json"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

