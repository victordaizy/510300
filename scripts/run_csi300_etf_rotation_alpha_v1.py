"""一次性评估已冻结的八因子ETF跨资产轮动。"""

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

from backtest.run_final_alpha_strategy import _benchmark_series, _return_metrics, summarize  # noqa: E402
from research.csi300_etf_rotation_alpha_v1 import FACTOR_COLUMNS, build_rotation_targets  # noqa: E402
from research.small_account_cross_sectional import SmallAccountCosts, run_small_account_open_backtest  # noqa: E402
from scripts.freeze_csi300_etf_rotation_protocol_v1 import (  # noqa: E402
    CONFIG_FILE,
    FROZEN_FILES,
    MANIFEST_FILE,
    sha256,
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def costs(contract: dict, slippage_key: str) -> SmallAccountCosts:
    account, fee = contract["account"], contract["costs"]
    return SmallAccountCosts(
        commission_rate=float(fee["commission_rate"]),
        minimum_commission_cny=float(fee["minimum_commission_cny"]),
        stamp_duty_sell_rate=float(fee["stamp_duty_sell_rate"]),
        stamp_duty_sell_rate_before_reduction=float(fee["stamp_duty_sell_rate_before_reduction"]),
        stamp_duty_reduction_effective_date=str(fee["stamp_duty_reduction_effective_date"]),
        slippage_bps_per_leg=float(fee[slippage_key]),
        cash_annual_rate=float(account["cash_annual_rate"]),
        lot_size=int(account["lot_size"]),
        minimum_trade_notional_cny=float(account["minimum_trade_notional_cny"]),
        maximum_positions=int(account["maximum_positions"]),
    )


def bootstrap(strategy: pd.Series, benchmark: pd.Series, contract: dict) -> dict:
    evaluation = contract["evaluation"]
    repetitions = int(evaluation["bootstrap_repetitions"])
    block = int(evaluation["bootstrap_block_length_trading_days"])
    rng = np.random.default_rng(int(evaluation["random_seed"]))
    left, right = strategy.to_numpy(float), benchmark.to_numpy(float)
    starts = np.arange(len(left) - block + 1)
    values = np.empty(repetitions)
    for index in range(repetitions):
        chosen = rng.choice(starts, size=int(np.ceil(len(left) / block)), replace=True)
        sampled = np.concatenate([np.arange(start, start + block) for start in chosen])[: len(left)]
        values[index] = np.expm1(np.log1p(left[sampled]).mean() * 242) - np.expm1(np.log1p(right[sampled]).mean() * 242)
    return {
        "repetitions": repetitions,
        "block_length_trading_days": block,
        "median": float(np.median(values)),
        "interval_95pct": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def period_results(ledger: pd.DataFrame, benchmark: pd.DataFrame, contract: dict) -> dict:
    result = {}
    for period in contract["evaluation"]["predefined_periods"]:
        mask = ledger["date"].between(pd.Timestamp(period["start"]), pd.Timestamp(period["end"]))
        strategy = _return_metrics(ledger.loc[mask, "daily_return"], ledger.loc[mask, "date"])
        bench = _return_metrics(benchmark.loc[mask, "daily_return"], benchmark.loc[mask, "date"])
        result[period["id"]] = {
            "start": period["start"], "end": period["end"], "strategy": strategy,
            "benchmark": bench, "annualized_excess": strategy["cagr"] - bench["cagr"],
        }
    return result


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def render(report: dict) -> str:
    base, stress = report["base_cost"], report["stress_cost"]
    lines = [
        "# CSI300 ETF 跨资产轮动 Alpha V1",
        "",
        f"- 状态：`{report['status']}`",
        f"- 因子：{report['factor_count']}个",
        "",
        "|成本|策略年化|H00300年化|年化超额|最大回撤|成交笔数|",
        "|---|---:|---:|---:|---:|---:|",
        f"|5bp|{base['strategy']['cagr']:.2%}|{base['benchmark']['cagr']:.2%}|{base['annualized_excess']:.2%}|{base['strategy']['maximum_drawdown']:.2%}|{base['trade_rows']}|",
        f"|15bp|{stress['strategy']['cagr']:.2%}|{stress['benchmark']['cagr']:.2%}|{stress['annualized_excess']:.2%}|{stress['strategy']['maximum_drawdown']:.2%}|{stress['trade_rows']}|",
        "",
        f"- Bootstrap 95%区间：{report['bootstrap_annualized_excess']['interval_95pct']}",
        f"- 最小成交：{report['execution_audit']['minimum_trade_notional_cny']:.2f}元",
        "",
        "## 门槛",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} `{key}`" for key, value in report["gates"].items())
    lines += ["", "历史研究结果不生成纸面仓位、订单或实盘连接。", ""]
    return "\n".join(lines)


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("state") != "FROZEN_BEFORE_MARKET_DATA_DOWNLOAD":
        raise RuntimeError("协议尚未在数据下载前冻结")
    changed = [name for name in FROZEN_FILES if sha256(ROOT / name) != manifest["frozen_files"][name]]
    if changed:
        raise RuntimeError(f"冻结后文件变化：{changed}")
    status = json.loads((ROOT / contract["inputs"]["data_status"]).read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or not status.get("formula_frozen_before_download"):
        raise RuntimeError("ETF数据审计未通过")
    if sha256(ROOT / contract["inputs"]["etf_panel"]) != status["hashes"]["panel"]:
        raise RuntimeError("ETF面板哈希变化")
    if sha256(ROOT / contract["inputs"]["benchmark"]) != status["hashes"]["benchmark"]:
        raise RuntimeError("基准哈希变化")

    panel = pd.read_parquet(ROOT / contract["inputs"]["etf_panel"])
    benchmark_raw = pd.read_parquet(ROOT / contract["inputs"]["benchmark"])
    features, targets, calendar = build_rotation_targets(panel, contract)
    execution = panel[["date", "con_code", "raw_open", "total_return_open", "total_return_close", "volume"]].copy()
    execution["is_suspended"] = execution["volume"].fillna(0.0).le(0.0)
    initial_cash = float(contract["account"]["initial_cash_cny"])
    base_ledger, base_trades = run_small_account_open_backtest(execution, targets, calendar, initial_cash, costs(contract, "base_slippage_bps_per_leg"))
    stress_ledger, stress_trades = run_small_account_open_backtest(execution, targets, calendar, initial_cash, costs(contract, "stress_slippage_bps_per_leg"))
    benchmark = _benchmark_series(benchmark_raw, base_ledger["date"], initial_cash)
    base = summarize(base_ledger, base_trades, benchmark, initial_cash)
    stress = summarize(stress_ledger, stress_trades, benchmark, initial_cash)
    boot = bootstrap(base_ledger["daily_return"], benchmark["daily_return"], contract)
    periods = period_results(base_ledger, benchmark, contract)
    positive_periods = sum(item["annualized_excess"] > 0 for item in periods.values())
    minimum_trade = float(base_trades["notional"].min()) if not base_trades.empty else np.nan
    threshold = float(contract["evaluation"]["annualized_excess_minimum"])
    gates = {
        "base_annualized_excess_20pct": base["annualized_excess"] >= threshold,
        "stress_annualized_excess_20pct": stress["annualized_excess"] >= float(contract["evaluation"]["stress_annualized_excess_minimum"]),
        "bootstrap_lower_bound_positive": boot["interval_95pct"][0] > float(contract["evaluation"]["bootstrap_lower_bound_strictly_above"]),
        "positive_predefined_periods": positive_periods >= int(contract["evaluation"]["positive_predefined_periods_minimum"]),
        "all_trades_at_least_5000": bool(base_trades.empty or base_trades["notional"].ge(float(contract["account"]["minimum_trade_notional_cny"]) - 1e-8).all()),
    }
    result_status = "ETF_ROTATION_HISTORICAL_PASS_AWAITING_FORWARD" if all(gates.values()) else "ETF_ROTATION_REJECTED_FROZEN"
    report = json_safe({
        "project_id": contract["protocol"]["project_id"], "status": result_status,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_manifest_sha256": sha256(MANIFEST_FILE), "formula_frozen_before_download": True,
        "factor_count": len(FACTOR_COLUMNS), "asset_count": len(contract["universe"]["risk_assets"]) + 1,
        "base_cost": base, "stress_cost": stress, "bootstrap_annualized_excess": boot,
        "predefined_periods": periods, "positive_predefined_periods": positive_periods, "gates": gates,
        "execution_audit": {
            "minimum_trade_notional_cny": minimum_trade, "maximum_position_count": int(base_ledger["position_count"].max()),
            "blocked_small_sell_days": int(base_ledger["blocked_small_sell_count"].gt(0).sum()), "t_plus_one_enforced": True,
        },
        "safety": contract["governance"],
    })
    outputs = contract["outputs"]
    for frame, key in ((features, "features"), (targets, "targets"), (base_ledger, "base_ledger"), (base_trades, "base_trades"), (stress_ledger, "stress_ledger"), (stress_trades, "stress_trades")):
        atomic_parquet(frame, ROOT / outputs[key])
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), ROOT / outputs["result_json"])
    atomic_text(render(report), ROOT / outputs["result_markdown"])
    print(json.dumps({"状态": result_status, "策略年化": base["strategy"]["cagr"], "基准年化": base["benchmark"]["cagr"], "基础年化超额": base["annualized_excess"], "压力年化超额": stress["annualized_excess"], "Bootstrap下界": boot["interval_95pct"][0], "正超额分段": positive_periods}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
