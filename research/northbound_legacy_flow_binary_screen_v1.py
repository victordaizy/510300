"""筛选旧披露口径北向净流量映射的510300/现金固定二元规则。"""

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
CONFIG_PATH = ROOT / "config" / "510300_northbound_legacy_flow_binary_screen_v1.yaml"
STUDY_ID = "510300_NORTHBOUND_LEGACY_FLOW_BINARY_SCREEN_V1"


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def json_default(value: Any) -> Any:
    """将NumPy和Pandas标量安全转换为JSON。"""

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


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """读取并核对冻结前协议。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("协议未达到冻结前实现完成状态")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("协议必须禁止参数营救")
    if int(protocol["frozen_candidate_family_size"]) != 6:
        raise ValueError("候选家族数量必须固定为6")
    if list(config["scope"]["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("持仓范围必须严格为510300和现金")
    if list(config["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if float(config["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(config["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")
    candidate_ids = [item["candidate_id"] for item in config["candidates"]]
    if len(candidate_ids) != 6 or len(set(candidate_ids)) != 6:
        raise ValueError("候选标识必须是6个唯一值")
    return config


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """运行前核对协议、源码、依赖和输入均未漂移。"""

    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError("冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_FIRST_SCREEN_RUN":
        raise ValueError("冻结清单状态无效")
    if sha256_file(CONFIG_PATH) != manifest["protocol_sha256"]:
        raise ValueError("冻结后协议漂移")
    for group in ["source_artifacts", "input_artifacts"]:
        for relative, expected in manifest[group].items():
            path = ROOT / relative
            if not path.exists() or sha256_file(path) != expected:
                raise ValueError(f"冻结工件漂移：{relative}")
    return manifest


def rolling_last_percentile(
    values: pd.Series,
    *,
    window: int,
    minimum_observations: int,
) -> pd.Series:
    """计算当前值在仅含当前及过去观测的滚动经验分位。"""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(numeric), np.nan, dtype=float)
    for index, current in enumerate(numeric):
        if not np.isfinite(current):
            continue
        start = max(0, index - window + 1)
        history = numeric[start : index + 1]
        history = history[np.isfinite(history)]
        if len(history) >= minimum_observations:
            output[index] = float(np.mean(history <= current))
    return pd.Series(output, index=values.index, dtype=float)


def build_flow_features(
    flow: pd.DataFrame,
    *,
    rolling_window: int,
    minimum_observations: int,
) -> pd.DataFrame:
    """仅用当日及过去流量构造固定1日、5日、20日风险特征。"""

    required = {
        "date",
        "north_net_buy_100m_cny",
        "northbound_semantics",
        "usable_as_net_flow",
        "source",
    }
    if missing := required - set(flow.columns):
        raise ValueError(f"北向资金输入缺少字段：{sorted(missing)}")
    frame = flow.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["north_net_buy_100m_cny"] = pd.to_numeric(
        frame["north_net_buy_100m_cny"], errors="raise"
    )
    frame.sort_values("date", kind="mergesort", inplace=True)
    if frame["date"].duplicated().any():
        raise ValueError("北向资金日期重复")
    if frame["north_net_buy_100m_cny"].isna().any():
        raise ValueError("北向成交净买额存在缺失")
    if not frame["usable_as_net_flow"].astype(bool).all():
        raise ValueError("输入混入不可作为日净流量的记录")
    if set(frame["northbound_semantics"].astype(str).unique()) != {
        "LEGACY_DAILY_NET_BUY"
    }:
        raise ValueError("北向资金披露语义不是冻结的旧口径")
    if frame["date"].max() >= pd.Timestamp("2024-08-19"):
        raise ValueError("输入混入披露变更日及以后")
    frame["flow_1d"] = frame["north_net_buy_100m_cny"]
    frame["flow_5d"] = frame["flow_1d"].rolling(5, min_periods=5).sum()
    frame["flow_20d"] = frame["flow_1d"].rolling(20, min_periods=20).sum()
    for horizon in [1, 5, 20]:
        frame[f"rank_{horizon}d"] = rolling_last_percentile(
            frame[f"flow_{horizon}d"],
            window=rolling_window,
            minimum_observations=minimum_observations,
        )
    return frame.reset_index(drop=True)


def build_candidate_states(features: pd.DataFrame) -> pd.DataFrame:
    """按冻结定义生成六列严格0/1状态。"""

    frame = features[["date", "flow_1d", "flow_5d", "flow_20d", "rank_1d", "rank_5d", "rank_20d"]].copy()
    frame["NB_FLOW_SIGN"] = np.where(frame["flow_1d"] < 0.0, 0, 1)
    frame["NB1D_BOTTOM10_CASH"] = np.where(
        frame["rank_1d"].notna() & frame["rank_1d"].le(0.10), 0, 1
    )
    frame["NB5D_BOTTOM10_CASH"] = np.where(
        frame["rank_5d"].notna() & frame["rank_5d"].le(0.10), 0, 1
    )
    frame["NB20D_BOTTOM10_CASH"] = np.where(
        frame["rank_20d"].notna() & frame["rank_20d"].le(0.10), 0, 1
    )
    frame["NB1D_OR_5D_BOTTOM10_CASH"] = np.where(
        (frame["rank_1d"].notna() & frame["rank_1d"].le(0.10))
        | (frame["rank_5d"].notna() & frame["rank_5d"].le(0.10)),
        0,
        1,
    )
    frame["NB1D_AND_5D_BOTTOM20_CASH"] = np.where(
        (frame["rank_1d"].notna() & frame["rank_1d"].le(0.20))
        & (frame["rank_5d"].notna() & frame["rank_5d"].le(0.20)),
        0,
        1,
    )
    state_columns = [column for column in frame.columns if column.startswith("NB")]
    for column in state_columns:
        frame[column] = frame[column].astype(np.int8)
        if not set(frame[column].unique()).issubset({0, 1}):
            raise AssertionError(f"{column}不是严格二元状态")
    return frame


def align_states_to_market(
    market: pd.DataFrame,
    candidate_states: pd.DataFrame,
    candidate_ids: list[str],
) -> pd.DataFrame:
    """对齐交易日；无北向新披露的交易日明确回到满仓。"""

    aligned = market[["trade_date"]].rename(columns={"trade_date": "date"}).merge(
        candidate_states,
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    aligned["flow_observed"] = aligned["_merge"].eq("both")
    aligned.drop(columns="_merge", inplace=True)
    for candidate_id in candidate_ids:
        aligned[candidate_id] = aligned[candidate_id].fillna(1).astype(np.int8)
        if not set(aligned[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}对齐后不是严格二元状态")
    return aligned


def required_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    """核对数据字段。"""

    if missing := set(required) - set(frame.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """按固定哈希读取北向、510300、全收益基准和现金分红。"""

    contracts = config["data_contracts"]
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, contract in contracts.items():
        path = ROOT / contract["file"]
        if not path.exists():
            raise FileNotFoundError(f"固定工件不存在：{contract['file']}")
        actual = sha256_file(path)
        if actual != contract["required_sha256"]:
            raise ValueError(f"固定工件哈希漂移：{contract['file']}")
        paths[name] = path
        hashes[contract["file"]] = actual

    flow = pd.read_parquet(paths["northbound_flow"])
    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    required_columns(flow, contracts["northbound_flow"]["required_columns"], "北向资金")
    required_columns(market, contracts["execution_daily"]["required_columns"], "510300日线")
    required_columns(
        benchmark,
        contracts["benchmark_total_return"]["required_columns"],
        "H00300全收益基准",
    )
    required_columns(
        dividends,
        contracts["cash_distributions"]["required_columns"],
        "510300分红",
    )
    if len(flow) != int(contracts["northbound_flow"]["required_rows"]):
        raise ValueError("北向资金固定行数不匹配")
    flow["date"] = pd.to_datetime(flow["date"], errors="raise").dt.normalize()
    if str(flow["date"].min().date()) != contracts["northbound_flow"]["required_first_date"]:
        raise ValueError("北向资金首日不匹配")
    if str(flow["date"].max().date()) != contracts["northbound_flow"]["required_last_date"]:
        raise ValueError("北向资金末日不匹配")
    if set(market["symbol"].dropna().astype(str).unique()) != {
        contracts["execution_daily"]["required_symbol"]
    }:
        raise ValueError("510300代码不匹配")
    if set(benchmark["symbol"].dropna().astype(str).unique()) != {
        contracts["benchmark_total_return"]["required_symbol"]
    }:
        raise ValueError("H00300代码不匹配")

    market = market.rename(columns={"date": "trade_date"}).copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="raise").dt.normalize()
    market["bar_end"] = market["trade_date"] + pd.Timedelta(hours=15)
    market.sort_values("trade_date", kind="mergesort", inplace=True)
    market.drop_duplicates("trade_date", keep="last", inplace=True)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise").dt.normalize()
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    window_start = pd.Timestamp("2016-12-05")
    window_end = pd.Timestamp(config["evaluation"]["end_date"])
    market = market.loc[
        market["trade_date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    if sorted(set(market["trade_date"]) - set(benchmark["date"])):
        raise ValueError("H00300缺少510300执行交易日")
    if (market[["open", "close"]] <= 0.0).any().any() or (benchmark["close"] <= 0.0).any():
        raise ValueError("执行或基准价格非法")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if len(dividends) != int(contracts["cash_distributions"]["required_event_count"]):
        raise ValueError("分红事件数不匹配")
    audit = {
        "hashes": hashes,
        "flow_rows": int(len(flow)),
        "flow_first_date": flow["date"].min().date().isoformat(),
        "flow_last_date": flow["date"].max().date().isoformat(),
        "market_rows": int(len(market)),
        "benchmark_rows": int(len(benchmark)),
        "dividend_event_count": int(len(dividends)),
        "post_disclosure_change_flow_rows_read": int(
            flow["date"].ge(pd.Timestamp("2024-08-19")).sum()
        ),
    }
    return {
        "flow": flow,
        "market": market,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    """按协议构造基础或压力成本。"""

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


def evaluate(
    config: dict[str, Any],
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    dividends: pd.DataFrame,
    aligned_states: pd.DataFrame,
) -> pd.DataFrame:
    """运行六候选、两成本、三时期和五个起点。"""

    benchmark_returns = build_benchmark_returns(benchmark)
    market_dates = [pd.Timestamp(value) for value in market["trade_date"].tolist()]
    rows: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        candidate_id = candidate["candidate_id"]
        states = aligned_states[candidate_id].to_numpy(dtype=np.int8)
        for scenario in config["evaluation"]["cost_scenarios"]:
            costs = cost_model(config, scenario)
            for period in config["evaluation"]["periods"]:
                for offset in config["evaluation"]["start_date_perturbations_trading_days"]:
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
                            "candidate_id": candidate_id,
                            "scenario": scenario,
                            "start_offset": int(offset),
                            **summary,
                            "t_plus_one_violations": int(
                                diagnostics["t_plus_one_violations"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_candidates(
    config: dict[str, Any],
    results: pd.DataFrame,
    aligned_states: pd.DataFrame,
) -> list[dict[str, Any]]:
    """汇总每个固定规则的最差门槛和综合期表现。"""

    summaries: list[dict[str, Any]] = []
    for candidate in config["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_rows = results.loc[results["candidate_id"].eq(candidate_id)]
        stress = candidate_rows.loc[candidate_rows["scenario"].eq("STRESS")]
        base = candidate_rows.loc[candidate_rows["scenario"].eq("BASE")]
        combined_id = "COMBINED_2018_2024"
        stress_combined = stress.loc[
            stress["period_id"].eq(combined_id) & stress["start_offset"].eq(0)
        ].iloc[0]
        base_combined = base.loc[
            base["period_id"].eq(combined_id) & base["start_offset"].eq(0)
        ].iloc[0]
        hard_pass = bool(stress["both_20pct_gates"].all())
        positive_shadow = bool(
            (stress["annualized_excess"] > 0.0).all()
            and (stress["rolling_242d_excess_median"] > 0.0).all()
        )
        state_series = aligned_states[candidate_id]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "rule": candidate["rule"],
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
                "cash_state_share_full_input_window": float(state_series.eq(0).mean()),
                "state_change_count_full_input_window": int(state_series.diff().ne(0).sum() - 1),
                "maximum_t_plus_one_violations": int(
                    candidate_rows["t_plus_one_violations"].max()
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (
            item["hard_pass"],
            item["stress_min_annualized_excess"],
            item["stress_min_rolling_242d_excess_median"],
        ),
        reverse=True,
    )


def percent(value: float | None) -> str:
    """格式化百分比。"""

    return "不可计算" if value is None else f"{value:.2%}"


def render_markdown(report: dict[str, Any]) -> str:
    """生成简洁的人读报告。"""

    best = report["best_candidate"]
    return f"""# 510300 北向旧口径流量二元筛选 V1

状态：`{report['status']}`

只使用2024-08-19披露变化前的北向成交净买额；信号在T日收盘形成，T+1开盘执行，目标状态严格为满仓或空仓。

## 最佳固定规则

- 规则：`{best['candidate_id']}`
- 压力最差年化净超额：{percent(best['stress_min_annualized_excess'])}
- 压力最差滚动242日超额中位数：{percent(best['stress_min_rolling_242d_excess_median'])}
- 压力综合期年化净超额：{percent(best['stress_combined_annualized_excess_offset0'])}
- 压力综合期滚动242日超额中位数：{percent(best['stress_combined_rolling_median_offset0'])}
- 双20全门：{'通过' if best['hard_pass'] else '未通过'}

决策：{report['decision']}
"""


def run(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """执行冻结后的唯一一次固定家族筛选。"""

    if config_path.resolve() != CONFIG_PATH.resolve():
        raise ValueError("只允许固定协议入口")
    config = load_config(config_path)
    manifest = verify_freeze_manifest(config)
    report_path = ROOT / config["artifacts"]["report_json"]
    if report_path.exists():
        raise FileExistsError("固定报告已存在，禁止重复运行")
    inputs, input_audit = load_inputs(config)
    mechanism = config["mechanism"]
    features = build_flow_features(
        inputs["flow"],
        rolling_window=int(mechanism["rolling_percentile_window_observations"]),
        minimum_observations=int(
            mechanism["rolling_percentile_minimum_observations"]
        ),
    )
    candidate_states = build_candidate_states(features)
    candidate_ids = [item["candidate_id"] for item in config["candidates"]]
    aligned_states = align_states_to_market(
        inputs["market"], candidate_states, candidate_ids
    )
    results = evaluate(
        config,
        inputs["market"],
        inputs["benchmark"],
        inputs["dividends"],
        aligned_states,
    )
    summaries = summarize_candidates(config, results, aligned_states)
    passing = [item["candidate_id"] for item in summaries if item["hard_pass"]]
    positive = [item["candidate_id"] for item in summaries if item["positive_shadow"]]
    status = (
        "DISCOVERY_CANDIDATE_FOUND_REQUIRES_246D_PROSPECTIVE_FREEZE"
        if passing
        else "REJECTED_FIXED_NORTHBOUND_LEGACY_FLOW_BINARY_FAMILY_NO_RESCUE"
    )
    decision = (
        "固定家族存在双20通过者；只允许冻结新的246交易日前瞻验证，不得直接交易。"
        if passing
        else "六个预先固定规则均未过双20门槛；禁止改分位、改累计窗口或反转符号营救。"
    )
    state_path = ROOT / config["artifacts"]["daily_states"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    atomic_parquet(aligned_states, state_path)
    atomic_parquet(results, metrics_path)
    report = {
        "status": status,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": manifest["protocol_sha256"],
        "freeze_manifest_sha256": sha256_file(
            ROOT / config["artifacts"]["freeze_manifest"]
        ),
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "signal_time": "T_CLOSE",
            "execution_time": "T_PLUS_1_OPEN",
            "evidence_label": config["protocol"]["evidence_label"],
        },
        "input_audit": input_audit,
        "feature_audit": {
            "flow_feature_rows": int(len(features)),
            "first_rank_1d_date": features.loc[
                features["rank_1d"].notna(), "date"
            ].min().date().isoformat(),
            "first_rank_5d_date": features.loc[
                features["rank_5d"].notna(), "date"
            ].min().date().isoformat(),
            "first_rank_20d_date": features.loc[
                features["rank_20d"].notna(), "date"
            ].min().date().isoformat(),
            "market_days_without_new_flow": int((~aligned_states["flow_observed"]).sum()),
            "future_return_columns_read": 0,
        },
        "candidate_count": len(candidate_ids),
        "result_rows": int(len(results)),
        "passing_candidates": passing,
        "positive_shadow_candidates": positive,
        "best_candidate": summaries[0],
        "candidate_summaries": summaries,
        "maximum_t_plus_one_violations": int(results["t_plus_one_violations"].max()),
        "artifacts": {
            state_path.relative_to(ROOT).as_posix(): sha256_file(state_path),
            metrics_path.relative_to(ROOT).as_posix(): sha256_file(metrics_path),
        },
        "decision": decision,
        "boundaries": config["boundaries"],
    }
    atomic_json(report, report_path)
    markdown_path = ROOT / config["artifacts"]["report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "passing_candidates": passing,
                "positive_shadow_candidates": positive,
                "best_candidate": summaries[0],
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
    """命令行入口。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
