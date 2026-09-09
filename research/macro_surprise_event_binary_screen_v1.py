"""筛选五类中国宏观实际值相对预期意外映射的510300/现金二元规则。"""

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
    summarize_period,
    trading_date_with_offset,
)
from research.us_china_overnight_binary_screen_v1 import simulate_preopen_candidate


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_macro_surprise_event_binary_screen_v1.yaml"
CANDIDATE_CONTRACT_PATH = (
    ROOT / "config" / "510300_macro_surprise_event_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "60ffdf04bc4e7ec8bec83f8333de241f47d1d1b3fb53e6ad703eff71141130e7"
)
STUDY_ID = "510300_MACRO_SURPRISE_EVENT_BINARY_SCREEN_V1"
SERIES_IDS = ["PMI", "M2", "EXPORTS", "IMPORTS", "INDUSTRIAL_PRODUCTION"]
EXPECTED_CANDIDATES = [
    "MACRO_ANY_NEG_HOLD1",
    "MACRO_ANY_NEG_HOLD3",
    "MACRO_ANY_NEG_HOLD5",
    "MACRO_SEVERE_ZLE_M1_HOLD3",
    "MACRO_BREADTH_TWO_NEG_HOLD3",
    "MACRO_COMPOSITE_ZLE_M0P5_HOLD3",
    "MACRO_PMI_NEG_HOLD5",
    "MACRO_M2_NEG_HOLD5",
]
CANDIDATE_TRIGGER_AND_HOLD = {
    "MACRO_ANY_NEG_HOLD1": ("trigger_any_negative", 1),
    "MACRO_ANY_NEG_HOLD3": ("trigger_any_negative", 3),
    "MACRO_ANY_NEG_HOLD5": ("trigger_any_negative", 5),
    "MACRO_SEVERE_ZLE_M1_HOLD3": ("trigger_severe_negative", 3),
    "MACRO_BREADTH_TWO_NEG_HOLD3": ("trigger_two_negative", 3),
    "MACRO_COMPOSITE_ZLE_M0P5_HOLD3": ("trigger_composite_negative", 3),
    "MACRO_PMI_NEG_HOLD5": ("trigger_pmi_negative", 5),
    "MACRO_M2_NEG_HOLD5": ("trigger_m2_negative", 5),
}


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    """将NumPy和Pandas值转换为JSON可序列化类型。"""

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"无法JSON序列化：{type(value).__name__}")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """原子写入JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_candidate_contract() -> dict[str, Any]:
    """读取在事件下载前已经固定的候选合同。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("候选合同在事件数据获取后发生漂移")
    contract = yaml.safe_load(CANDIDATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol = contract["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("候选合同研究编号不匹配")
    if protocol["state"] != "CANDIDATE_FAMILY_FIXED_BEFORE_EVENT_DATA_ACQUISITION":
        raise ValueError("候选合同状态无效")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("候选合同必须禁止参数营救")
    candidate_ids = [item["candidate_id"] for item in contract["candidates"]]
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("候选合同的八条规则或顺序发生漂移")
    contract_series = [item["series_id"] for item in contract["event_contract"]["series"]]
    if contract_series != SERIES_IDS:
        raise ValueError("固定宏观序列或顺序发生漂移")
    if list(contract["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    return contract


def load_config(path: Path = CONFIG_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    """读取并核对完整冻结前协议。"""

    contract = load_candidate_contract()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("主协议研究编号不匹配")
    if protocol["state"] != "PREFREEZE_IMPLEMENTATION_COMPLETE":
        raise ValueError("主协议尚未达到冻结前实现完成状态")
    if protocol["candidate_contract_sha256"] != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("主协议候选合同哈希不匹配")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("主协议必须禁止参数营救")
    if float(contract["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(contract["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")
    return config, contract


def verify_freeze_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """运行前核对冻结协议、源码、测试、依赖与输入。"""

    manifest_path = ROOT / config["artifacts"]["freeze_manifest"]
    if not manifest_path.exists():
        raise FileNotFoundError("冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_BEFORE_FIRST_SCREEN_RUN":
        raise ValueError("冻结清单状态无效")
    if sha256_file(CONFIG_PATH) != manifest["protocol_sha256"]:
        raise ValueError("冻结后主协议漂移")
    if sha256_file(CANDIDATE_CONTRACT_PATH) != manifest["candidate_contract_sha256"]:
        raise ValueError("冻结后候选合同漂移")
    for group in ["source_artifacts", "input_artifacts"]:
        for relative, expected in manifest[group].items():
            path = ROOT / relative
            if not path.exists() or sha256_file(path) != expected:
                raise ValueError(f"冻结工件漂移：{relative}")
    return manifest


def build_event_features(
    raw_events: pd.DataFrame,
    *,
    standardization_window: int,
    minimum_prior_events: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造仅用此前事件尺度的逐事件意外与逐公布日聚合特征。"""

    required = {
        "series_id",
        "product",
        "release_date",
        "actual",
        "forecast",
        "previous",
        "source_function",
        "source",
        "retrieved_at",
    }
    if missing := required - set(raw_events.columns):
        raise ValueError(f"宏观事件输入缺少字段：{sorted(missing)}")
    events = raw_events.copy()
    events["release_date"] = pd.to_datetime(
        events["release_date"], errors="raise"
    ).dt.normalize()
    for column in ["actual", "forecast", "previous"]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    if set(events["series_id"].astype(str).unique()) != set(SERIES_IDS):
        raise ValueError("宏观事件输入序列集合不匹配")
    if events.duplicated(["series_id", "release_date"]).any():
        raise ValueError("宏观事件同序列同公布日重复")
    events["usable_event"] = events["actual"].notna() & events["forecast"].notna()
    events["raw_surprise"] = events["actual"] - events["forecast"]
    events.loc[~events["usable_event"], "raw_surprise"] = np.nan
    events.sort_values(["series_id", "release_date"], kind="mergesort", inplace=True)

    def prior_scale(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(
            standardization_window,
            min_periods=minimum_prior_events,
        ).std(ddof=0)

    events["prior_surprise_scale"] = events.groupby(
        "series_id", sort=False
    )["raw_surprise"].transform(prior_scale)
    events.loc[events["prior_surprise_scale"].le(0.0), "prior_surprise_scale"] = np.nan
    events["standardized_surprise"] = (
        events["raw_surprise"] / events["prior_surprise_scale"]
    )
    events["negative_surprise"] = events["usable_event"] & events["raw_surprise"].lt(0.0)
    events["severe_negative_surprise"] = (
        events["standardized_surprise"].notna()
        & events["standardized_surprise"].le(-1.0)
    )
    events.sort_values(["release_date", "series_id"], kind="mergesort", inplace=True)
    events.reset_index(drop=True, inplace=True)

    rows: list[dict[str, Any]] = []
    for release_date, group in events.loc[events["usable_event"]].groupby(
        "release_date", sort=True
    ):
        standardized = group["standardized_surprise"].dropna()
        negative_series = set(
            group.loc[group["negative_surprise"], "series_id"].astype(str)
        )
        rows.append(
            {
                "release_date": pd.Timestamp(release_date),
                "usable_series_count": int(len(group)),
                "negative_count": int(group["negative_surprise"].sum()),
                "severe_negative_count": int(
                    group["severe_negative_surprise"].sum()
                ),
                "mean_standardized_surprise": (
                    np.nan if standardized.empty else float(standardized.mean())
                ),
                "negative_series": "|".join(sorted(negative_series)),
                "trigger_any_negative": bool(len(negative_series) >= 1),
                "trigger_severe_negative": bool(
                    group["severe_negative_surprise"].any()
                ),
                "trigger_two_negative": bool(len(negative_series) >= 2),
                "trigger_composite_negative": bool(
                    not standardized.empty and standardized.mean() <= -0.5
                ),
                "trigger_pmi_negative": bool("PMI" in negative_series),
                "trigger_m2_negative": bool("M2" in negative_series),
            }
        )
    daily = pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)
    return events, daily


def build_candidate_states(
    market_dates: pd.Series,
    daily_events: pd.DataFrame,
) -> pd.DataFrame:
    """将事件映射到严格晚于公布日的中国交易日并生成八个二元状态。"""

    dates = pd.to_datetime(market_dates, errors="raise").dt.normalize().reset_index(
        drop=True
    )
    date_array = dates.to_numpy(dtype="datetime64[ns]")
    output = pd.DataFrame({"date": dates})
    for candidate_id in EXPECTED_CANDIDATES:
        trigger_column, hold_days = CANDIDATE_TRIGGER_AND_HOLD[candidate_id]
        states = np.ones(len(dates), dtype=np.int8)
        signal_asof = np.full(
            len(dates), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
        )
        triggered = daily_events.loc[daily_events[trigger_column].astype(bool)]
        for event in triggered.itertuples(index=False):
            release_date = np.datetime64(pd.Timestamp(event.release_date), "ns")
            first_index = int(np.searchsorted(date_array, release_date, side="right"))
            if first_index >= len(dates):
                continue
            last_index = min(len(dates), first_index + int(hold_days))
            states[first_index:last_index] = 0
            existing = signal_asof[first_index:last_index]
            replacement = np.full(last_index - first_index, release_date)
            signal_asof[first_index:last_index] = np.where(
                np.isnat(existing),
                replacement,
                np.maximum(existing, replacement),
            )
        output[candidate_id] = states
        output[f"{candidate_id}__signal_asof_event_date"] = pd.to_datetime(signal_asof)
        invalid = output[f"{candidate_id}__signal_asof_event_date"].ge(output["date"])
        if bool(invalid.fillna(False).any()):
            raise AssertionError(f"{candidate_id}使用了公布日当日或未来事件")
        if not set(output[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}不是严格二元状态")
    return output


def required_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    """核对输入字段。"""

    if missing := set(required) - set(frame.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """按固定哈希读取事件、510300执行、H00300全收益与分红。"""

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

    events = pd.read_parquet(paths["macro_events"])
    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    required_columns(events, contracts["macro_events"]["required_columns"], "宏观事件")
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
    if len(events) != int(contracts["macro_events"]["required_rows"]):
        raise ValueError("宏观事件固定行数不匹配")
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
    market.sort_values("trade_date", kind="mergesort", inplace=True)
    market.drop_duplicates("trade_date", keep="last", inplace=True)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise").dt.normalize()
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    window_start = pd.Timestamp(config["evaluation_window"]["feature_start_date"])
    window_end = pd.Timestamp(config["evaluation_window"]["end_date"])
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
        "raw_event_rows": int(len(events)),
        "raw_event_series": sorted(events["series_id"].astype(str).unique()),
        "market_rows": int(len(market)),
        "market_first_date": market["trade_date"].min().date().isoformat(),
        "market_last_date": market["trade_date"].max().date().isoformat(),
        "benchmark_rows": int(len(benchmark)),
        "dividend_event_count": int(len(dividends)),
    }
    return {
        "events": events,
        "market": market,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def cost_model(config: dict[str, Any], scenario: str) -> CostModel:
    """构造基础或压力成本。"""

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
        trading_days_per_year=int(config["annualization_trading_days"]),
        lot_size=int(costs["lot_size_shares"]),
    )


def evaluate(
    config: dict[str, Any],
    contract: dict[str, Any],
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    dividends: pd.DataFrame,
    states: pd.DataFrame,
) -> pd.DataFrame:
    """运行八候选、两成本、四时期和五个起点。"""

    benchmark_returns = build_benchmark_returns(benchmark)
    market_dates = [pd.Timestamp(value) for value in market["trade_date"].tolist()]
    rows: list[dict[str, Any]] = []
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        target_states = states[candidate_id].to_numpy(dtype=np.int8)
        signal_column = f"{candidate_id}__signal_asof_event_date"
        for scenario in contract["evaluation"]["cost_scenarios"]:
            costs = cost_model(config, scenario)
            for period in contract["evaluation"]["periods"]:
                for offset in contract["evaluation"][
                    "start_date_perturbations_trading_days"
                ]:
                    start_date = trading_date_with_offset(
                        market_dates, pd.Timestamp(period["start_date"]), int(offset)
                    )
                    ledger, trades, diagnostics = simulate_preopen_candidate(
                        market,
                        dividends,
                        target_states,
                        states[signal_column],
                        start_date=start_date,
                        end_date=pd.Timestamp(period["end_date"]),
                        costs=costs,
                        initial_capital=float(config["initial_capital_cny"]),
                    )
                    summary = summarize_period(
                        ledger,
                        trades,
                        benchmark_returns,
                        period_id=period["id"],
                        start_date=start_date,
                        end_date=pd.Timestamp(period["end_date"]),
                        objective=contract["objective"],
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
                            "same_or_future_signal_date_violations": int(
                                diagnostics["same_or_future_signal_date_violations"]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_candidates(
    contract: dict[str, Any],
    results: pd.DataFrame,
    states: pd.DataFrame,
) -> list[dict[str, Any]]:
    """汇总固定规则的最差门槛与综合期表现。"""

    summaries: list[dict[str, Any]] = []
    combined_id = "COMBINED_2017_2025"
    for candidate in contract["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_rows = results.loc[results["candidate_id"].eq(candidate_id)]
        stress = candidate_rows.loc[candidate_rows["scenario"].eq("STRESS")]
        base = candidate_rows.loc[candidate_rows["scenario"].eq("BASE")]
        stress_combined = stress.loc[
            stress["period_id"].eq(combined_id) & stress["start_offset"].eq(0)
        ].iloc[0]
        base_combined = base.loc[
            base["period_id"].eq(combined_id) & base["start_offset"].eq(0)
        ].iloc[0]
        hard_pass = bool(stress["both_20pct_gates"].all())
        positive_shadow = bool(
            (stress["annualized_excess"] > 0.0).all()
            & (stress["rolling_242d_excess_median"] > 0.0).all()
        )
        state_series = states[candidate_id]
        signal_series = states[f"{candidate_id}__signal_asof_event_date"]
        summaries.append(
            {
                "candidate_id": candidate_id,
                "rule": candidate["rule"],
                "mechanism": candidate["mechanism"],
                "hard_pass": hard_pass,
                "positive_shadow": positive_shadow,
                "stress_min_annualized_excess": float(
                    stress["annualized_excess"].min()
                ),
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
                "cash_trigger_event_count": int(signal_series.dropna().nunique()),
                "state_change_count_full_input_window": int(
                    max(0, state_series.diff().ne(0).sum() - 1)
                ),
                "maximum_t_plus_one_violations": int(
                    candidate_rows["t_plus_one_violations"].max()
                ),
                "maximum_same_or_future_signal_date_violations": int(
                    candidate_rows["same_or_future_signal_date_violations"].max()
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
    """生成人读报告。"""

    best = report["best_candidate"]
    return f"""# 510300 宏观预期意外事件二元筛选 V1

状态：`{report['status']}`

实际值与市场预期只在公布日之后生效；任何公布日当日成交均被禁止，下一交易日开盘执行，持仓严格为510300满仓或现金空仓。

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
    config, contract = load_config(config_path)
    manifest = verify_freeze_manifest(config)
    report_path = ROOT / config["artifacts"]["report_json"]
    if report_path.exists():
        raise FileExistsError("固定报告已存在，禁止重复运行")
    inputs, input_audit = load_inputs(config)
    events, daily_events = build_event_features(
        inputs["events"],
        standardization_window=int(
            contract["event_contract"]["standardization_window_events"]
        ),
        minimum_prior_events=int(
            contract["event_contract"]["standardization_minimum_prior_events"]
        ),
    )
    states = build_candidate_states(inputs["market"]["trade_date"], daily_events)
    results = evaluate(
        config,
        contract,
        inputs["market"],
        inputs["benchmark"],
        inputs["dividends"],
        states,
    )
    summaries = summarize_candidates(contract, results, states)
    passing = [item["candidate_id"] for item in summaries if item["hard_pass"]]
    positive = [
        item["candidate_id"] for item in summaries if item["positive_shadow"]
    ]
    status = (
        "DISCOVERY_CANDIDATE_FOUND_REQUIRES_246D_PROSPECTIVE_FREEZE"
        if passing
        else "REJECTED_FIXED_MACRO_SURPRISE_EVENT_BINARY_FAMILY_NO_RESCUE"
    )
    decision = (
        "固定家族存在双20通过者；只允许冻结新的246交易日前瞻验证，不得直接交易。"
        if passing
        else "八个预先固定规则均未过双20门槛；禁止改指标、改符号、改标准化或改持有期营救。"
    )
    event_path = ROOT / config["artifacts"]["event_features"]
    daily_event_path = ROOT / config["artifacts"]["daily_event_features"]
    state_path = ROOT / config["artifacts"]["daily_states"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    atomic_parquet(events, event_path)
    atomic_parquet(daily_events, daily_event_path)
    atomic_parquet(states, state_path)
    atomic_parquet(results, metrics_path)
    report = {
        "status": status,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": manifest["protocol_sha256"],
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "freeze_manifest_sha256": sha256_file(
            ROOT / config["artifacts"]["freeze_manifest"]
        ),
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "signal_cutoff": "strictly after release date",
            "execution_time": "first subsequent China trading day open",
            "evidence_label": contract["protocol"]["evidence_label"],
        },
        "input_audit": input_audit,
        "feature_audit": {
            "raw_event_rows": int(len(events)),
            "usable_event_rows": int(events["usable_event"].sum()),
            "standardized_event_rows": int(
                events["standardized_surprise"].notna().sum()
            ),
            "release_days_with_usable_events": int(len(daily_events)),
            "series_usable_counts": {
                series_id: int(
                    events.loc[events["series_id"].eq(series_id), "usable_event"].sum()
                )
                for series_id in SERIES_IDS
            },
            "same_release_date_execution_count": 0,
            "future_return_columns_read": 0,
        },
        "candidate_count": len(EXPECTED_CANDIDATES),
        "result_rows": int(len(results)),
        "passing_candidates": passing,
        "positive_shadow_candidates": positive,
        "best_candidate": summaries[0],
        "candidate_summaries": summaries,
        "maximum_t_plus_one_violations": int(results["t_plus_one_violations"].max()),
        "maximum_same_or_future_signal_date_violations": int(
            results["same_or_future_signal_date_violations"].max()
        ),
        "artifacts": {
            event_path.relative_to(ROOT).as_posix(): sha256_file(event_path),
            daily_event_path.relative_to(ROOT).as_posix(): sha256_file(daily_event_path),
            state_path.relative_to(ROOT).as_posix(): sha256_file(state_path),
            metrics_path.relative_to(ROOT).as_posix(): sha256_file(metrics_path),
        },
        "decision": decision,
        "boundaries": contract["boundaries"],
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
                "maximum_same_or_future_signal_date_violations": report[
                    "maximum_same_or_future_signal_date_violations"
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
