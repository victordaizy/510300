"""将冻结的主流量逐日概率按唯一0.5界映射为510300/现金精确账本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.intraday_binary_livermore_screen_v1 import (
    CostModel,
    build_benchmark_returns,
    simulate_candidate,
    summarize_period,
    trading_date_with_offset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_primary_flow_p05_etf_binary_screen_v1.yaml"


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """加载并核对不可调整的单候选协议。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"]["study_id"] != "510300_PRIMARY_FLOW_P05_ETF_BINARY_SCREEN_V1":
        raise ValueError("研究编号不匹配")
    if config["protocol"]["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("协议不是冻结前实现完成状态")
    if not bool(config["protocol"]["no_parameter_rescue"]):
        raise ValueError("协议必须禁止参数营救")
    if list(config["scope"]["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("持仓范围必须严格为510300和现金")
    if list(config["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if float(config["state_rule"]["threshold"]) != 0.50:
        raise ValueError("唯一概率界必须为0.50")
    if config["state_rule"]["threshold_search"] != "FORBIDDEN":
        raise ValueError("必须禁止阈值搜索")
    if float(config["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化超额硬门必须为20%")
    if float(config["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动超额硬门必须为20%")
    return config


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """核对冻结后的协议、程序、依赖和数据哈希。"""

    path = ROOT / config["artifacts"]["freeze_manifest"]
    if not path.exists():
        raise FileNotFoundError("冻结清单不存在")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if sha256_file(CONFIG_PATH) != manifest["protocol_sha256"]:
        raise ValueError("冻结后协议漂移")
    for group in ["source_artifacts", "input_artifacts"]:
        for relative_path, expected in manifest[group].items():
            artifact = ROOT / relative_path
            if not artifact.exists() or sha256_file(artifact) != expected:
                raise ValueError(f"冻结工件漂移：{relative_path}")
    return manifest


def _required_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    if missing := set(required) - set(frame.columns):
        raise ValueError(f"{name}缺少字段：{sorted(missing)}")


def audit_and_build_daily_states(
    market: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """审计递归时点并将概率严格按0.5扩展为每日收盘状态。"""

    required = {
        "signal_date",
        "model_fit_date",
        "latest_training_label_end_date",
        "training_sample_count",
        "direction_probability_positive_20d_net",
        "share_snapshot_date",
        "share_snapshot_age_calendar_days",
    }
    if missing := required - set(forecasts.columns):
        raise ValueError(f"冻结概率缺少字段：{sorted(missing)}")
    frame = forecasts.copy()
    for column in [
        "signal_date",
        "model_fit_date",
        "latest_training_label_end_date",
        "share_snapshot_date",
    ]:
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame["direction_probability_positive_20d_net"] = pd.to_numeric(
        frame["direction_probability_positive_20d_net"], errors="raise"
    )
    if frame["signal_date"].duplicated().any():
        raise ValueError("冻结概率信号日期重复")
    if not frame["direction_probability_positive_20d_net"].between(
        0.0, 1.0, inclusive="both"
    ).all():
        raise ValueError("冻结概率超出0到1")
    if (frame["latest_training_label_end_date"] > frame["model_fit_date"]).any():
        raise ValueError("父模型使用了拟合日之后的标签")
    if (frame["model_fit_date"] > frame["signal_date"]).any():
        raise ValueError("父模型拟合日在信号日之后")
    market_dates = pd.to_datetime(market["trade_date"]).dt.normalize()
    missing_market_dates = sorted(set(frame["signal_date"]) - set(market_dates))
    if missing_market_dates:
        raise ValueError(f"概率信号日不在510300交易日：{missing_market_dates[:5]}")
    signal_map = {
        pd.Timestamp(record["signal_date"]): float(
            record["direction_probability_positive_20d_net"]
        )
        for record in frame.to_dict("records")
    }
    current_state = 0
    state_values = np.zeros(len(market), dtype=np.int8)
    daily_rows: list[dict[str, Any]] = []
    for index, date_value in enumerate(market_dates):
        date = pd.Timestamp(date_value)
        probability = signal_map.get(date)
        if probability is not None:
            current_state = int(probability >= threshold)
        state_values[index] = current_state
        daily_rows.append(
            {
                "date": date,
                "signal_available": probability is not None,
                "direction_probability_positive_20d_net": probability,
                "desired_state_at_close": current_state,
            }
        )
    if not set(np.unique(state_values)).issubset({0, 1}):
        raise AssertionError("映射后状态不是严格二元")
    daily = pd.DataFrame(daily_rows)
    audit = {
        "forecast_rows": int(len(frame)),
        "first_signal_date": str(frame["signal_date"].min().date()),
        "last_signal_date": str(frame["signal_date"].max().date()),
        "minimum_training_samples": int(frame["training_sample_count"].min()),
        "maximum_training_samples": int(frame["training_sample_count"].max()),
        "training_label_after_fit_violations": int(
            (frame["latest_training_label_end_date"] > frame["model_fit_date"]).sum()
        ),
        "model_fit_after_signal_violations": int(
            (frame["model_fit_date"] > frame["signal_date"]).sum()
        ),
        "probability_minimum": float(
            frame["direction_probability_positive_20d_net"].min()
        ),
        "probability_maximum": float(
            frame["direction_probability_positive_20d_net"].max()
        ),
        "signal_full_share": float(
            (frame["direction_probability_positive_20d_net"] >= threshold).mean()
        ),
        "future_outcome_columns_read": 0,
    }
    return daily, state_values, audit


def _cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    costs = config["costs"]
    if scenario == "BASE":
        slippage = costs["base_slippage_bps_per_leg"]
    elif scenario == "STRESS":
        slippage = costs["stress_slippage_bps_per_leg"]
    else:
        raise ValueError(f"未知成本情景：{scenario}")
    return CostModel(
        scenario=scenario,
        commission_rate=float(costs["commission_rate_per_leg"]),
        minimum_commission=float(costs["minimum_commission_cny_per_leg"]),
        slippage_bps=float(slippage),
        cash_annual_rate=float(costs["cash_annual_rate"]),
        trading_days_per_year=int(config["objective"]["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """按哈希读取；概率文件仅打开许可列，未来结果列完全不读。"""

    contracts = config["data_contracts"]
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, contract in contracts.items():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"固定工件不存在：{path}")
        actual = sha256_file(path)
        if actual != contract["required_sha256"]:
            raise ValueError(f"固定工件哈希漂移：{contract['file']}")
        paths[name] = path
        hashes[contract["file"]] = actual

    permitted = list(contracts["frozen_forecasts"]["permitted_columns_to_read"])
    forecasts = pd.read_parquet(paths["frozen_forecasts"], columns=permitted)
    if len(forecasts) != int(contracts["frozen_forecasts"]["required_rows"]):
        raise ValueError("冻结概率行数不匹配")
    forecasts["signal_date"] = pd.to_datetime(forecasts["signal_date"]).dt.normalize()
    if str(forecasts["signal_date"].min().date()) != contracts["frozen_forecasts"][
        "required_first_signal_date"
    ]:
        raise ValueError("冻结概率首日不匹配")
    if str(forecasts["signal_date"].max().date()) != contracts["frozen_forecasts"][
        "required_last_signal_date"
    ]:
        raise ValueError("冻结概率末日不匹配")

    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    _required_columns(
        market,
        contracts["execution_daily"]["required_columns"],
        "510300日线",
    )
    _required_columns(
        benchmark,
        contracts["benchmark_total_return"]["required_columns"],
        "H00300基准",
    )
    if set(market["symbol"].dropna().unique()) != {
        contracts["execution_daily"]["required_symbol"]
    }:
        raise ValueError("510300代码不匹配")
    if set(benchmark["symbol"].dropna().unique()) != {
        contracts["benchmark_total_return"]["required_symbol"]
    }:
        raise ValueError("H00300代码不匹配")
    market = market.rename(columns={"date": "trade_date"}).copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
    market["bar_end"] = market["trade_date"] + pd.Timedelta(hours=15)
    market = market.sort_values("trade_date", kind="mergesort").drop_duplicates(
        "trade_date", keep="last"
    )
    benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.normalize()
    benchmark = benchmark.sort_values("date", kind="mergesort").drop_duplicates(
        "date", keep="last"
    )
    window_start = pd.Timestamp("2023-10-31")
    window_end = pd.Timestamp(config["evaluation"]["end_date"])
    market = market.loc[
        market["trade_date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    if sorted(set(market["trade_date"]) - set(benchmark["date"])):
        raise ValueError("H00300缺少510300执行交易日")
    if (market[["open", "close"]] <= 0.0).any().any() or (
        benchmark["close"] <= 0.0
    ).any():
        raise ValueError("执行或基准价格非法")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column]).dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if len(dividends) != int(contracts["cash_distributions"]["required_event_count"]):
        raise ValueError("分红事件数不匹配")
    audit = {
        "hashes": hashes,
        "forecast_columns_read": permitted,
        "forbidden_forecast_columns_read": [],
        "market_rows": int(len(market)),
        "benchmark_rows": int(len(benchmark)),
        "dividend_events": int(len(dividends)),
    }
    return {
        "forecasts": forecasts,
        "market": market,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def evaluate(
    config: dict[str, Any],
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    dividends: pd.DataFrame,
    states: np.ndarray,
) -> pd.DataFrame:
    """运行两档成本、三个时期和五个起点。"""

    benchmark_returns = build_benchmark_returns(benchmark)
    market_dates = [pd.Timestamp(value) for value in market["trade_date"].tolist()]
    rows: list[dict[str, Any]] = []
    for scenario in config["evaluation"]["cost_scenarios"]:
        costs = _cost_model(config, scenario)
        for period in config["evaluation"]["periods"]:
            for offset in config["evaluation"][
                "start_date_perturbations_trading_days"
            ]:
                start_date = trading_date_with_offset(
                    market_dates, pd.Timestamp(period["start_date"]), int(offset)
                )
                ledger, trades, diagnostics = simulate_candidate(
                    market,
                    dividends,
                    states,
                    start_date=start_date,
                    end_date=pd.Timestamp(period["end_date"]),
                    costs=costs,
                    initial_capital=float(config["scope"]["initial_capital_cny"]),
                )
                summary = summarize_period(
                    ledger,
                    trades,
                    benchmark_returns,
                    period_id=period["id"],
                    start_date=start_date,
                    end_date=pd.Timestamp(period["end_date"]),
                    objective=config["objective"],
                )
                rows.append(
                    {
                        "candidate_id": config["state_rule"]["candidate_id"],
                        "scenario": scenario,
                        "start_offset": int(offset),
                        **summary,
                        "t_plus_one_violations": int(
                            diagnostics["t_plus_one_violations"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"无法JSON序列化：{type(value).__name__}")


def _percent(value: float | None) -> str:
    return "不可计算" if value is None else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    result = report["summary"]
    return f"""# 510300 主流量概率0.5二元筛选 V1

状态：`{report['status']}`

父模型的未来实际收益列没有被读取；每日收盘概率只按预先固定的0.50界转成状态，次日开盘执行。

## 结果

- 压力最差年化净超额：{_percent(result['stress_min_annualized_excess'])}
- 压力最差滚动242日超额中位数：{_percent(result['stress_min_rolling_242d_excess_median'])}
- 压力综合期年化净超额：{_percent(result['stress_combined_annualized_excess_offset0'])}
- 压力综合期滚动242日超额中位数：{_percent(result['stress_combined_rolling_median_offset0'])}
- 全门通过：{'是' if result['hard_pass'] else '否'}
- 正向影子通过：{'是' if result['positive_shadow'] else '否'}

决策：{report['decision']}
"""


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """执行冻结后唯一一次0.5映射筛选。"""

    if config_path.resolve() != CONFIG_PATH.resolve():
        raise ValueError("只允许固定协议入口")
    config = load_config(config_path)
    manifest = verify_freeze_manifest(config)
    report_path = ROOT / config["artifacts"]["report_json"]
    if report_path.exists():
        raise FileExistsError("固定报告已存在，禁止重复运行")
    inputs, input_audit = load_inputs(config)
    daily_states, state_values, forecast_audit = audit_and_build_daily_states(
        inputs["market"],
        inputs["forecasts"],
        threshold=float(config["state_rule"]["threshold"]),
    )
    results = evaluate(
        config,
        inputs["market"],
        inputs["benchmark"],
        inputs["dividends"],
        state_values,
    )
    stress = results.loc[results["scenario"].eq("STRESS")]
    base = results.loc[results["scenario"].eq("BASE")]
    stress_combined = stress.loc[
        stress["period_id"].eq("COMBINED") & stress["start_offset"].eq(0)
    ].iloc[0]
    base_combined = base.loc[
        base["period_id"].eq("COMBINED") & base["start_offset"].eq(0)
    ].iloc[0]
    hard_pass = bool(stress["both_20pct_gates"].all())
    positive_shadow = bool(
        (stress["annualized_excess"] > 0.0).all()
        and (stress["rolling_242d_excess_median"] > 0.0).all()
    )
    summary = {
        "candidate_id": config["state_rule"]["candidate_id"],
        "hard_pass": hard_pass,
        "positive_shadow": positive_shadow,
        "stress_min_annualized_excess": float(stress["annualized_excess"].min()),
        "stress_min_rolling_242d_excess_median": float(
            stress["rolling_242d_excess_median"].min()
        ),
        "stress_combined_annualized_excess_offset0": float(
            stress_combined["annualized_excess"]
        ),
        "stress_combined_rolling_median_offset0": float(
            stress_combined["rolling_242d_excess_median"]
        ),
        "base_combined_annualized_excess_offset0": float(
            base_combined["annualized_excess"]
        ),
        "base_combined_rolling_median_offset0": float(
            base_combined["rolling_242d_excess_median"]
        ),
        "stress_trade_leg_count_total": int(stress["trade_leg_count"].sum()),
        "stress_execution_cost_cny_total": float(stress["execution_cost_cny"].sum()),
    }
    status = (
        config["governance"]["result_label_if_passed"]
        if hard_pass
        else config["governance"]["result_label_if_rejected"]
    )

    combined_period = next(
        period for period in config["evaluation"]["periods"] if period["id"] == "COMBINED"
    )
    ledger, trades, diagnostics = simulate_candidate(
        inputs["market"],
        inputs["dividends"],
        state_values,
        start_date=pd.Timestamp(combined_period["start_date"]),
        end_date=pd.Timestamp(combined_period["end_date"]),
        costs=_cost_model(config, "STRESS"),
        initial_capital=float(config["scope"]["initial_capital_cny"]),
    )
    output_dir = ROOT / config["artifacts"]["output_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "period_results": ROOT / config["artifacts"]["period_results_parquet"],
        "daily_states": ROOT / config["artifacts"]["daily_states_parquet"],
        "stress_daily_ledger": ROOT
        / config["artifacts"]["stress_daily_ledger_parquet"],
        "stress_trades": ROOT / config["artifacts"]["stress_trades_parquet"],
    }
    results.to_parquet(output_paths["period_results"], index=False)
    daily_states.to_parquet(output_paths["daily_states"], index=False)
    ledger.to_parquet(output_paths["stress_daily_ledger"], index=False)
    trades.to_parquet(output_paths["stress_trades"], index=False)
    report = {
        "status": status,
        "study_id": config["protocol"]["study_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": manifest["protocol_sha256"],
        "freeze_manifest_sha256": sha256_file(
            ROOT / config["artifacts"]["freeze_manifest"]
        ),
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "threshold": 0.50,
            "historical_evidence_label": config["protocol"][
                "historical_evidence_label"
            ],
        },
        "input_audit": input_audit,
        "forecast_audit": forecast_audit,
        "period_result_rows": int(len(results)),
        "maximum_t_plus_one_violations": int(results["t_plus_one_violations"].max()),
        "summary": summary,
        "combined_stress_diagnostics": diagnostics,
        "artifacts": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in output_paths.values()
        },
        "decision": (
            "回溯双门通过；只允许另行冻结246交易日前瞻验证。"
            if hard_pass
            else "固定0.50映射未通过；禁止改阈值、校准概率或加滞回营救。"
        ),
        "governance": config["governance"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    markdown_path = ROOT / config["artifacts"]["report_markdown"]
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "summary": summary,
                "maximum_t_plus_one_violations": report[
                    "maximum_t_plus_one_violations"
                ],
                "report_sha256": sha256_file(report_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
