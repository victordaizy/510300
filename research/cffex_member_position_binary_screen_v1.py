"""筛选中金所IF前20名会员持仓映射的510300/现金固定二元规则。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.etf_share_premium_level_binary_screen_v1 import (
    atomic_json,
    atomic_parquet,
    percent,
    required_columns,
    rolling_last_percentile,
    sha256_file,
)
from research.if_true_term_structure_binary_screen_v1_0_1 import (
    evaluate,
    summarize_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_cffex_member_position_binary_screen_v1.yaml"
CANDIDATE_CONTRACT_PATH = (
    ROOT / "config" / "510300_cffex_member_position_binary_screen_v1_candidates.yaml"
)
CANDIDATE_CONTRACT_SHA256 = (
    "845f94c3eed3ef6991ff78312c1578b18baffcc2dd8039e915af4fe0d74ac739"
)
STUDY_ID = "510300_CFFEX_MEMBER_POSITION_BINARY_SCREEN_V1"
EXPECTED_CANDIDATES = [
    "CFFEX_ALL_NET_BOTTOM10_CASH",
    "CFFEX_ALL_NET_TOP10_CASH",
    "CFFEX_ALL_NET_CHANGE_BOTTOM10_CASH",
    "CFFEX_ALL_NET_CHANGE_TOP10_CASH",
    "CFFEX_NEAR_NET_BOTTOM10_CASH",
    "CFFEX_NEAR_NET_TOP10_CASH",
    "CFFEX_TOP5_CONCENTRATION_BOTTOM10_CASH",
    "CFFEX_TOP5_CONCENTRATION_TOP10_CASH",
]
FEATURE_COLUMNS = [
    "all_top20_net_ratio",
    "all_top20_net_change_ratio",
    "near_top20_net_ratio",
    "all_top5_concentration_spread",
]
RANK_COLUMNS = [f"rank_{column}" for column in FEATURE_COLUMNS]


def load_candidate_contract() -> dict[str, Any]:
    """读取并核对正式会员排名历史获取前固定的八规则合同。"""

    if sha256_file(CANDIDATE_CONTRACT_PATH) != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("正式会员排名历史获取前固定的候选合同发生漂移")
    contract = yaml.safe_load(CANDIDATE_CONTRACT_PATH.read_text(encoding="utf-8"))
    protocol = contract["protocol"]
    if protocol["study_id"] != STUDY_ID:
        raise ValueError("候选合同研究编号不匹配")
    if (
        protocol["state"]
        != "CANDIDATE_FAMILY_FIXED_BEFORE_CFFEX_MEMBER_RANK_HISTORY_ACQUISITION"
    ):
        raise ValueError("候选合同状态无效")
    if not bool(protocol["no_parameter_rescue"]):
        raise ValueError("候选合同必须禁止参数营救")
    candidate_ids = [item["candidate_id"] for item in contract["candidates"]]
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError("固定八规则或顺序发生漂移")
    if list(contract["scope"]["allowed_holdings"]) != ["510300.SH", "CASH_CNY"]:
        raise ValueError("持仓范围必须严格为510300和现金")
    if list(contract["scope"]["allowed_target_states"]) != [0, 1]:
        raise ValueError("目标状态必须严格为0和1")
    if float(contract["objective"]["minimum_annualized_excess"]) != 0.20:
        raise ValueError("年化净超额硬门必须为20%")
    if float(contract["objective"]["minimum_rolling_excess_median"]) != 0.20:
        raise ValueError("滚动净超额硬门必须为20%")
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
        raise ValueError("主协议必须禁止结果后参数营救")
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


def build_member_features(
    member_ranks: pd.DataFrame,
    expiry_table: pd.DataFrame,
    *,
    minimum_dte: int,
    minimum_rank_rows: int,
    maximum_rank: int,
    minimum_contracts: int,
    rolling_window: int,
    minimum_observations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """仅用当日已发布排名构造跨期限与近月会员持仓特征。"""

    required_columns(
        member_ranks,
        [
            "date",
            "symbol",
            "rank",
            "long_open_interest",
            "long_open_interest_change",
            "short_open_interest",
            "short_open_interest_change",
            "source_url",
            "raw_file_sha256",
        ],
        "中金所IF会员持仓排名",
    )
    required_columns(
        expiry_table,
        ["symbol", "expiry_date", "expiry_source", "active_at_ceiling"],
        "IF合约到期表",
    )
    ranks = member_ranks.copy()
    ranks["date"] = pd.to_datetime(ranks["date"], errors="raise").dt.normalize()
    ranks["symbol"] = ranks["symbol"].astype(str)
    ranks["rank"] = pd.to_numeric(ranks["rank"], errors="raise").astype(int)
    numeric_columns = [
        "long_open_interest",
        "long_open_interest_change",
        "short_open_interest",
        "short_open_interest_change",
    ]
    for column in numeric_columns:
        ranks[column] = pd.to_numeric(ranks[column], errors="raise")
    ranks.sort_values(["date", "symbol", "rank"], kind="mergesort", inplace=True)
    if ranks.duplicated(["date", "symbol", "rank"]).any():
        raise ValueError("会员排名输入日期、合约、名次重复")
    if not ranks["symbol"].str.fullmatch(r"IF\d{4}").all():
        raise ValueError("会员排名输入混入非IF代码")
    if ranks[["long_open_interest", "short_open_interest"]].lt(0).any().any():
        raise ValueError("会员多空持仓量出现负数")
    if not ranks["rank"].between(1, maximum_rank, inclusive="both").all():
        raise ValueError("会员排名超出冻结的1至20名范围")

    expiry = expiry_table.copy()
    expiry["symbol"] = expiry["symbol"].astype(str)
    expiry["expiry_date"] = pd.to_datetime(
        expiry["expiry_date"], errors="raise"
    ).dt.normalize()
    if expiry["symbol"].duplicated().any():
        raise ValueError("IF合约到期表代码重复")
    if set(ranks["symbol"].unique()) != set(expiry["symbol"].unique()):
        raise ValueError("会员排名与到期表代码集合不一致")
    active = expiry.loc[expiry["active_at_ceiling"].astype(bool)]
    if not active["expiry_source"].eq(
        "CFFEX_TRADING_PARAMETER_END_TRADING_DAY_AT_CEILING"
    ).all():
        raise ValueError("样本末日存续合约未使用官方END_TRADING_DAY")

    top20 = ranks.loc[ranks["rank"].le(maximum_rank)].copy()
    top20_summary = (
        top20.groupby(["date", "symbol"], sort=True)
        .agg(
            rank_count=("rank", "nunique"),
            long_top20=("long_open_interest", "sum"),
            short_top20=("short_open_interest", "sum"),
            long_change_top20=("long_open_interest_change", "sum"),
            short_change_top20=("short_open_interest_change", "sum"),
            source_url=("source_url", "first"),
            raw_file_sha256=("raw_file_sha256", "first"),
        )
        .reset_index()
    )
    top5_summary = (
        top20.loc[top20["rank"].le(5)]
        .groupby(["date", "symbol"], sort=True)
        .agg(
            long_top5=("long_open_interest", "sum"),
            short_top5=("short_open_interest", "sum"),
        )
        .reset_index()
    )
    summaries = top20_summary.merge(
        top5_summary,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    ).merge(
        expiry[["symbol", "expiry_date", "expiry_source", "active_at_ceiling"]],
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    summaries["calendar_days_to_expiry"] = (
        summaries["expiry_date"] - summaries["date"]
    ).dt.days
    summaries["rank_row_eligible"] = summaries["rank_count"].ge(minimum_rank_rows)
    summaries["dte_eligible"] = summaries["calendar_days_to_expiry"].ge(minimum_dte)
    summaries["contract_eligible"] = (
        summaries["rank_row_eligible"] & summaries["dte_eligible"]
    )
    summaries.sort_values(["date", "expiry_date", "symbol"], inplace=True)
    summaries.reset_index(drop=True, inplace=True)
    eligible = summaries.loc[summaries["contract_eligible"]].copy()

    feature_rows: list[dict[str, Any]] = []
    insufficient_dates: list[str] = []
    zero_denominator_dates: list[str] = []
    for date, group in eligible.groupby("date", sort=True):
        group = group.sort_values(["expiry_date", "symbol"], kind="mergesort")
        if len(group) < minimum_contracts:
            insufficient_dates.append(pd.Timestamp(date).date().isoformat())
            continue
        if group["expiry_date"].duplicated().any():
            raise ValueError(f"{pd.Timestamp(date).date()}出现重复IF到期日")
        all_long = float(group["long_top20"].sum())
        all_short = float(group["short_top20"].sum())
        near = group.iloc[0]
        near_long = float(near["long_top20"])
        near_short = float(near["short_top20"])
        if all_long <= 0.0 or all_short <= 0.0 or near_long + near_short <= 0.0:
            zero_denominator_dates.append(pd.Timestamp(date).date().isoformat())
            continue
        feature_rows.append(
            {
                "date": pd.Timestamp(date),
                "eligible_contract_count": int(len(group)),
                "near_symbol": str(near["symbol"]),
                "near_expiry_date": pd.Timestamp(near["expiry_date"]),
                "near_dte": int(near["calendar_days_to_expiry"]),
                "all_long_top20": all_long,
                "all_short_top20": all_short,
                "all_long_change_top20": float(group["long_change_top20"].sum()),
                "all_short_change_top20": float(group["short_change_top20"].sum()),
                "all_long_top5": float(group["long_top5"].sum()),
                "all_short_top5": float(group["short_top5"].sum()),
                "all_top20_net_ratio": (all_long - all_short)
                / (all_long + all_short),
                "all_top20_net_change_ratio": (
                    float(group["long_change_top20"].sum())
                    - float(group["short_change_top20"].sum())
                )
                / (all_long + all_short),
                "near_top20_net_ratio": (near_long - near_short)
                / (near_long + near_short),
                "all_top5_concentration_spread": float(group["long_top5"].sum())
                / all_long
                - float(group["short_top5"].sum())
                / all_short,
                "feature_asof": pd.Timestamp(date)
                + pd.Timedelta(hours=16, minutes=30),
                "execution_earliest": "NEXT_TRADING_DAY_OPEN",
            }
        )
    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise ValueError("会员持仓日特征为空")
    features.sort_values("date", kind="mergesort", inplace=True)
    features.reset_index(drop=True, inplace=True)
    if features["date"].duplicated().any():
        raise ValueError("会员持仓日特征日期重复")
    if features[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("会员持仓核心变量存在缺失")
    if not np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("会员持仓核心变量存在非有限值")
    for feature in FEATURE_COLUMNS:
        features[f"rank_{feature}"] = rolling_last_percentile(
            features[feature],
            window=rolling_window,
            minimum_observations=minimum_observations,
        )
    audit = {
        "raw_position_rows": int(len(ranks)),
        "raw_position_dates": int(ranks["date"].nunique()),
        "raw_position_contracts": int(ranks["symbol"].nunique()),
        "raw_date_contracts": int(len(top20_summary)),
        "contract_summary_rows": int(len(summaries)),
        "eligible_contract_summary_rows": int(len(eligible)),
        "feature_rows": int(len(features)),
        "feature_first_date": features["date"].min().date().isoformat(),
        "feature_last_date": features["date"].max().date().isoformat(),
        "minimum_eligible_contract_count": int(
            features["eligible_contract_count"].min()
        ),
        "maximum_eligible_contract_count": int(
            features["eligible_contract_count"].max()
        ),
        "insufficient_contract_dates_excluded": insufficient_dates,
        "zero_denominator_dates_excluded": zero_denominator_dates,
        "active_contract_metadata_rows": int(len(active)),
        "feature_publication_clock": "T日16:30",
        "future_price_or_return_columns_read": 0,
    }
    return summaries, features, audit


def build_candidate_states(features: pd.DataFrame) -> pd.DataFrame:
    """按冻结上下尾定义生成八列严格0/1状态。"""

    frame = features[
        ["date", *FEATURE_COLUMNS, *RANK_COLUMNS, "near_symbol"]
    ].copy()
    pairs = [
        ("CFFEX_ALL_NET_BOTTOM10_CASH", "rank_all_top20_net_ratio", "BOTTOM"),
        ("CFFEX_ALL_NET_TOP10_CASH", "rank_all_top20_net_ratio", "TOP"),
        (
            "CFFEX_ALL_NET_CHANGE_BOTTOM10_CASH",
            "rank_all_top20_net_change_ratio",
            "BOTTOM",
        ),
        (
            "CFFEX_ALL_NET_CHANGE_TOP10_CASH",
            "rank_all_top20_net_change_ratio",
            "TOP",
        ),
        ("CFFEX_NEAR_NET_BOTTOM10_CASH", "rank_near_top20_net_ratio", "BOTTOM"),
        ("CFFEX_NEAR_NET_TOP10_CASH", "rank_near_top20_net_ratio", "TOP"),
        (
            "CFFEX_TOP5_CONCENTRATION_BOTTOM10_CASH",
            "rank_all_top5_concentration_spread",
            "BOTTOM",
        ),
        (
            "CFFEX_TOP5_CONCENTRATION_TOP10_CASH",
            "rank_all_top5_concentration_spread",
            "TOP",
        ),
    ]
    for candidate_id, rank_column, tail in pairs:
        if tail == "BOTTOM":
            trigger = frame[rank_column].notna() & frame[rank_column].le(0.10)
        else:
            trigger = frame[rank_column].notna() & frame[rank_column].ge(0.90)
        frame[candidate_id] = np.where(trigger, 0, 1).astype(np.int8)
        if not set(frame[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}不是严格二元状态")
    return frame


def align_states_to_market(market: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """对齐510300交易日；会员特征缺失日明确回到满仓。"""

    aligned = market[["trade_date"]].rename(columns={"trade_date": "date"}).merge(
        states,
        on="date",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    aligned["member_feature_observed"] = aligned["_merge"].eq("both")
    aligned.drop(columns="_merge", inplace=True)
    for candidate_id in EXPECTED_CANDIDATES:
        aligned[candidate_id] = aligned[candidate_id].fillna(1).astype(np.int8)
        if not set(aligned[candidate_id].unique()).issubset({0, 1}):
            raise AssertionError(f"{candidate_id}对齐后不是严格二元状态")
    return aligned


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """按固定哈希读取会员排名、到期表、执行资产、基准和分红。"""

    contracts = config["data_contracts"]
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, item in contracts.items():
        path = ROOT / item["file"]
        if not path.exists():
            raise FileNotFoundError(f"固定工件不存在：{item['file']}")
        actual = sha256_file(path)
        if actual != item["required_sha256"]:
            raise ValueError(f"固定工件哈希漂移：{item['file']}")
        paths[name] = path
        hashes[item["file"]] = actual

    ranks = pd.read_parquet(paths["member_positions"])
    expiry = pd.read_parquet(paths["if_contract_expiry"])
    acquisition_audit = json.loads(
        paths["member_acquisition_audit"].read_text(encoding="utf-8")
    )
    market = pd.read_parquet(paths["execution_daily"])
    benchmark = pd.read_parquet(paths["benchmark_total_return"])
    dividends = pd.read_csv(paths["cash_distributions"])
    required_columns(
        ranks,
        contracts["member_positions"]["required_columns"],
        "中金所IF会员排名",
    )
    required_columns(
        expiry,
        contracts["if_contract_expiry"]["required_columns"],
        "IF合约到期表",
    )
    required_columns(
        market, contracts["execution_daily"]["required_columns"], "510300日线"
    )
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
    if len(ranks) != int(contracts["member_positions"]["required_rows"]):
        raise ValueError("会员排名固定行数不匹配")
    if len(expiry) != int(contracts["if_contract_expiry"]["required_rows"]):
        raise ValueError("IF到期表固定行数不匹配")
    if acquisition_audit["status"] != "SUCCESS_CFFEX_IF_MEMBER_POSITIONS_ACQUIRED":
        raise ValueError("会员排名官方输入获取审计状态无效")
    if bool(acquisition_audit["performance_returns_read"]):
        raise ValueError("会员排名输入获取阶段读取了绩效收益")
    if acquisition_audit["candidate_contract_sha256"] != CANDIDATE_CONTRACT_SHA256:
        raise ValueError("会员排名获取时的候选合同哈希不匹配")
    member_relative = contracts["member_positions"]["file"]
    if acquisition_audit["output"] != member_relative:
        raise ValueError("会员排名获取审计输出路径不一致")
    if acquisition_audit["output_sha256"] != hashes[member_relative]:
        raise ValueError("会员排名输入与获取审计记录不一致")

    for frame, symbol, label in [
        (market, "510300.SH", "510300执行资产"),
        (benchmark, "H00300", "H00300全收益基准"),
    ]:
        if set(frame["symbol"].dropna().astype(str).unique()) != {symbol}:
            raise ValueError(f"{label}代码不匹配")
    market = market.rename(columns={"date": "trade_date"}).copy()
    market["trade_date"] = pd.to_datetime(
        market["trade_date"], errors="raise"
    ).dt.normalize()
    market["bar_end"] = market["trade_date"] + pd.Timedelta(hours=15)
    market.sort_values("trade_date", kind="mergesort", inplace=True)
    market.drop_duplicates("trade_date", keep="last", inplace=True)
    benchmark["date"] = pd.to_datetime(
        benchmark["date"], errors="raise"
    ).dt.normalize()
    benchmark.sort_values("date", kind="mergesort", inplace=True)
    benchmark.drop_duplicates("date", keep="last", inplace=True)
    window_start = pd.Timestamp("2012-05-28")
    window_end = pd.Timestamp("2026-08-12")
    market = market.loc[
        market["trade_date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    benchmark = benchmark.loc[
        benchmark["date"].between(window_start, window_end, inclusive="both")
    ].reset_index(drop=True)
    if sorted(set(market["trade_date"]) - set(benchmark["date"])):
        raise ValueError("H00300缺少510300执行交易日")
    if (market[["open", "close"]] <= 0.0).any().any():
        raise ValueError("510300执行价格非法")
    if (benchmark["close"] <= 0.0).any():
        raise ValueError("H00300全收益指数价格非法")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(
            dividends[column], errors="raise"
        ).dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if len(dividends) != int(contracts["cash_distributions"]["required_event_count"]):
        raise ValueError("分红事件数不匹配")
    audit = {
        "hashes": hashes,
        "member_position_rows": int(len(ranks)),
        "member_position_dates": int(
            pd.to_datetime(ranks["date"]).dt.normalize().nunique()
        ),
        "member_position_contracts": int(ranks["symbol"].nunique()),
        "if_expiry_rows": int(len(expiry)),
        "official_acquisition_status": acquisition_audit["status"],
        "market_rows": int(len(market)),
        "benchmark_rows": int(len(benchmark)),
        "dividend_event_count": int(len(dividends)),
    }
    return {
        "ranks": ranks,
        "expiry": expiry,
        "market": market,
        "benchmark": benchmark,
        "dividends": dividends,
    }, audit


def render_markdown(report: dict[str, Any]) -> str:
    """生成简洁的人读报告。"""

    best = report["best_candidate"]
    feature_audit = report["feature_audit"]
    return f"""# 510300 中金所IF会员持仓二元筛选 V1

状态：`{report['status']}`

信息只来自中金所IF前20名会员持仓排名，T日约16:30发布后形成状态，T+1开盘执行510300；目标状态严格为满仓或空仓。

## 覆盖

- 官方排名原始行：{report['input_audit']['member_position_rows']:,}
- 合资格特征日：{feature_audit['feature_rows']:,}
- 510300交易日中缺少合资格会员特征：{feature_audit['market_days_without_member_feature']:,}
- 至少合资格合约数：{feature_audit['minimum_eligible_contract_count']}

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
    selection = contract["contract_and_rank_selection"]
    feature_contract = contract["features"]
    summaries_frame, features, feature_audit = build_member_features(
        inputs["ranks"],
        inputs["expiry"],
        minimum_dte=int(selection["minimum_calendar_days_to_expiry_inclusive"]),
        minimum_rank_rows=int(selection["minimum_unique_rank_rows_per_contract"]),
        maximum_rank=int(selection["maximum_rank_inclusive"]),
        minimum_contracts=int(selection["minimum_eligible_contracts_per_day"]),
        rolling_window=int(feature_contract["rolling_percentile_window_trading_rows"]),
        minimum_observations=int(
            feature_contract["rolling_percentile_minimum_rows"]
        ),
    )
    states = build_candidate_states(features)
    aligned_states = align_states_to_market(inputs["market"], states)
    results = evaluate(
        config,
        contract,
        inputs["market"],
        inputs["benchmark"],
        inputs["dividends"],
        aligned_states,
    )
    candidate_summaries = summarize_candidates(contract, results, aligned_states)
    passing = [
        item["candidate_id"] for item in candidate_summaries if item["hard_pass"]
    ]
    positive = [
        item["candidate_id"]
        for item in candidate_summaries
        if item["positive_shadow"]
    ]
    status = (
        "DISCOVERY_CANDIDATE_FOUND_REQUIRES_246D_PROSPECTIVE_FREEZE"
        if passing
        else "REJECTED_FIXED_CFFEX_MEMBER_POSITION_BINARY_FAMILY_NO_RESCUE"
    )
    decision = (
        "固定家族存在双20通过者；只允许冻结新的246交易日前瞻验证，不得直接交易。"
        if passing
        else "八个预先固定的会员持仓上下尾规则均未过双20门槛；禁止改方向、分位、最低合约数、到期筛选或排名范围营救。"
    )
    summary_path = ROOT / config["artifacts"]["contract_summaries"]
    feature_path = ROOT / config["artifacts"]["daily_features"]
    state_path = ROOT / config["artifacts"]["daily_states"]
    metrics_path = ROOT / config["artifacts"]["metrics"]
    atomic_parquet(summaries_frame, summary_path)
    atomic_parquet(features, feature_path)
    atomic_parquet(aligned_states, state_path)
    atomic_parquet(results, metrics_path)
    first_rank_dates = {
        column: (
            features.loc[features[column].notna(), "date"].min().date().isoformat()
            if features[column].notna().any()
            else None
        )
        for column in RANK_COLUMNS
    }
    report = {
        "status": status,
        "study_id": STUDY_ID,
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_sha256": manifest["protocol_sha256"],
        "freeze_manifest_sha256": sha256_file(
            ROOT / config["artifacts"]["freeze_manifest"]
        ),
        "candidate_contract_sha256": CANDIDATE_CONTRACT_SHA256,
        "scope": {
            "execution_asset": "510300.SH",
            "information_asset": "CFFEX IF top-20 member position ranking",
            "allowed_states": [0, 1],
            "benchmark": "H00300 total return",
            "signal_time": "T_CFFEX_MEMBER_RANKING_AROUND_16_30",
            "execution_time": "T_PLUS_1_510300_OPEN",
            "evidence_label": contract["protocol"]["evidence_label"],
        },
        "input_audit": input_audit,
        "feature_audit": {
            **feature_audit,
            "first_rank_dates": first_rank_dates,
            "market_days_without_member_feature": int(
                (~aligned_states["member_feature_observed"]).sum()
            ),
        },
        "candidate_count": len(EXPECTED_CANDIDATES),
        "result_rows": int(len(results)),
        "passing_candidates": passing,
        "positive_shadow_candidates": positive,
        "best_candidate": candidate_summaries[0],
        "candidate_summaries": candidate_summaries,
        "maximum_t_plus_one_violations": int(
            results["t_plus_one_violations"].max()
        ),
        "artifacts": {
            summary_path.relative_to(ROOT).as_posix(): sha256_file(summary_path),
            feature_path.relative_to(ROOT).as_posix(): sha256_file(feature_path),
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
                "best_candidate": candidate_summaries[0],
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
