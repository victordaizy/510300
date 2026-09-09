"""第一阶段：分别研究估值、趋势、波动率对510300未来收益的信息。"""

from __future__ import annotations

import json
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
ETF_FILE = PROJECT_ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = PROJECT_ROOT / "data" / "reference" / "510300_dividends.csv"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "000300_market_state_daily.parquet"
OUTPUT_DATA_FILE = PROJECT_ROOT / "data" / "features" / "510300_base_feature_outcomes.parquet"
OUTPUT_JSON_FILE = PROJECT_ROOT / "reports" / "research" / "510300_base_feature_research.json"
OUTPUT_MD_FILE = PROJECT_ROOT / "reports" / "research" / "510300_base_feature_research.md"

HORIZONS = (1, 3, 5, 10, 20)
FEATURE_GROUPS = {
    "valuation": [
        "signal_pe_ttm", "signal_pb", "signal_earnings_yield",
        "signal_pe_ttm_percentile_5y", "signal_pb_percentile_5y",
    ],
    "trend": ["signal_trend_ma20_over_ma60", "signal_trend_close_over_ma120"],
    "volatility": [
        "signal_rv_5", "signal_rv_20", "signal_rv_60", "signal_rv_120",
        "signal_downside_rv_20", "signal_parkinson_rv_20", "signal_vol_term_20_120",
    ],
}


def add_forward_outcomes(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    data = prices.copy().sort_values("date").reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"])
    events = dividends.copy()
    if events.empty:
        events = pd.DataFrame(columns=["ex_date", "cash_dividend_per_share"])
    events["ex_date"] = pd.to_datetime(events["ex_date"])
    events["cash_dividend_per_share"] = pd.to_numeric(events["cash_dividend_per_share"], errors="coerce")
    dividend_by_date = events.groupby("ex_date")["cash_dividend_per_share"].sum().to_dict()

    for horizon in horizons:
        entry_price = data["open"].shift(-1)
        exit_price = data["close"].shift(-horizon)
        data[f"label_end_date_{horizon}d"] = data["date"].shift(-horizon)
        dividends_received = []
        mae = []
        mfe = []
        for index in range(len(data)):
            end_index = index + horizon
            if end_index >= len(data):
                dividends_received.append(np.nan)
                mae.append(np.nan)
                mfe.append(np.nan)
                continue
            # t+1开盘买入，入场日若为除息日不享有该次分红；从t+2开始计入。
            eligible_dates = data.loc[index + 2 : end_index, "date"] if horizon >= 2 else pd.Series(dtype="datetime64[ns]")
            cash = float(sum(dividend_by_date.get(date, 0.0) for date in eligible_dates))
            dividends_received.append(cash)
            path = data.loc[index + 1 : end_index]
            mae.append(float(path["low"].min() / entry_price.iloc[index] - 1.0))
            mfe.append(float(path["high"].max() / entry_price.iloc[index] - 1.0))
        data[f"forward_dividend_{horizon}d"] = dividends_received
        data[f"forward_price_return_{horizon}d"] = exit_price / entry_price - 1.0
        data[f"forward_total_return_{horizon}d"] = (
            exit_price + data[f"forward_dividend_{horizon}d"]
        ) / entry_price - 1.0
        data[f"forward_mae_{horizon}d"] = mae
        data[f"forward_mfe_{horizon}d"] = mfe
        data[f"t1_executable_{horizon}d"] = horizon >= 2
    return data


def assign_split(date: pd.Timestamp, split_config: dict) -> str:
    ranges = {
        "development": (split_config["development_start"], split_config["development_end"]),
        "validation": (split_config["validation_start"], split_config["validation_end"]),
        "final_holdout": (split_config["final_holdout_start"], split_config["final_holdout_end"]),
    }
    for name, (start, end) in ranges.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return name
    return "purge_embargo"


def staggered_spearman(data: pd.DataFrame, feature: str, target: str, horizon: int) -> dict:
    correlations = []
    for offset in range(max(1, horizon)):
        sample = data.iloc[offset::max(1, horizon)][[feature, target]].dropna()
        if len(sample) >= 10 and sample[feature].nunique() > 1 and sample[target].nunique() > 1:
            correlations.append(float(sample[feature].corr(sample[target], method="spearman")))
    if not correlations:
        return {"median": None, "min": None, "max": None, "same_sign_ratio": None, "subsamples": 0}
    median = float(np.median(correlations))
    return {
        "median": median, "min": float(np.min(correlations)), "max": float(np.max(correlations)),
        "same_sign_ratio": float(np.mean(np.sign(correlations) == np.sign(median))),
        "subsamples": len(correlations),
    }


def moving_block_bootstrap_ic(
    data: pd.DataFrame,
    feature: str,
    target: str,
    horizon: int,
    iterations: int = 300,
) -> dict:
    clean = data[[feature, target]].dropna().reset_index(drop=True)
    block_size = max(20, horizon)
    if len(clean) < block_size * 3:
        return {
            "iterations": 0,
            "block_size": block_size,
            "median": None,
            "ci_2_5pct": None,
            "ci_97_5pct": None,
        }
    ranked_x = clean[feature].rank(method="average").to_numpy(dtype=float)
    ranked_y = clean[target].rank(method="average").to_numpy(dtype=float)
    maximum_start = len(clean) - block_size
    seed = zlib.crc32(f"{feature}|{target}|{len(clean)}".encode("utf-8"))
    generator = np.random.default_rng(seed)
    correlations = []
    blocks_needed = int(np.ceil(len(clean) / block_size))
    for _ in range(iterations):
        starts = generator.integers(0, maximum_start + 1, size=blocks_needed)
        indices = np.concatenate(
            [np.arange(start, start + block_size, dtype=int) for start in starts]
        )[: len(clean)]
        x_sample = ranked_x[indices]
        y_sample = ranked_y[indices]
        if np.std(x_sample) == 0 or np.std(y_sample) == 0:
            continue
        correlations.append(float(np.corrcoef(x_sample, y_sample)[0, 1]))
    if not correlations:
        return {
            "iterations": 0,
            "block_size": block_size,
            "median": None,
            "ci_2_5pct": None,
            "ci_97_5pct": None,
        }
    return {
        "iterations": len(correlations),
        "block_size": block_size,
        "median": float(np.median(correlations)),
        "ci_2_5pct": float(np.quantile(correlations, 0.025)),
        "ci_97_5pct": float(np.quantile(correlations, 0.975)),
    }


def quintile_edges(series: pd.Series) -> list[float]:
    clean = series.dropna()
    values = clean.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy(dtype=float)
    return sorted(set(values.tolist()))


def summarize_quintiles(data: pd.DataFrame, feature: str, target: str, mae: str, mfe: str, edges: list[float]) -> list[dict]:
    labels = list(range(1, len(edges) + 2))
    bucket = pd.cut(data[feature], bins=[-np.inf, *edges, np.inf], labels=labels, include_lowest=True)
    temp = data.assign(_quintile=bucket)
    output = []
    for group, frame in temp.groupby("_quintile", observed=True):
        clean = frame[[target, mae, mfe]].dropna()
        if clean.empty:
            continue
        output.append(
            {
                "quintile": int(group), "observations": int(len(clean)),
                "mean_return": float(clean[target].mean()), "median_return": float(clean[target].median()),
                "hit_rate": float((clean[target] > 0).mean()),
                "mean_mae": float(clean[mae].mean()), "mean_mfe": float(clean[mfe].mean()),
            }
        )
    return output


def evaluate_feature(data: pd.DataFrame, feature: str, horizon: int, split_config: dict) -> dict:
    target = f"forward_total_return_{horizon}d"
    mae = f"forward_mae_{horizon}d"
    mfe = f"forward_mfe_{horizon}d"
    end_date = f"label_end_date_{horizon}d"
    development = data.loc[
        (data["split"] == "development")
        & (data[end_date] <= pd.Timestamp(split_config["development_end"]))
    ].dropna(subset=[feature, target, mae, mfe])
    validation = data.loc[
        (data["split"] == "validation")
        & (data[end_date] <= pd.Timestamp(split_config["validation_end"]))
    ].dropna(subset=[feature, target, mae, mfe])
    edges = quintile_edges(development[feature])
    result = {"horizon": horizon, "feature": feature, "quintile_edges_from_development": edges, "splits": {}}
    for name, frame in [("development", development), ("validation", validation)]:
        ic = float(frame[feature].corr(frame[target], method="spearman")) if len(frame) >= 3 else None
        quintiles = summarize_quintiles(frame, feature, target, mae, mfe, edges)
        spread = None
        if quintiles and quintiles[0]["quintile"] == 1 and quintiles[-1]["quintile"] == len(edges) + 1:
            spread = quintiles[-1]["mean_return"] - quintiles[0]["mean_return"]
        result["splits"][name] = {
            "observations": int(len(frame)),
            "approx_independent_20d_blocks": int(len(frame) // max(20, horizon)),
            "spearman_ic_all_overlapping": ic, "staggered_non_overlapping_ic": staggered_spearman(frame, feature, target, horizon),
            "moving_block_bootstrap_ic": moving_block_bootstrap_ic(frame, feature, target, horizon),
            "top_minus_bottom_mean_return": spread, "quintiles": quintiles,
        }
    dev_ic = result["splits"]["development"]["staggered_non_overlapping_ic"]["median"]
    val_ic = result["splits"]["validation"]["staggered_non_overlapping_ic"]["median"]
    dev_spread = result["splits"]["development"]["top_minus_bottom_mean_return"]
    val_spread = result["splits"]["validation"]["top_minus_bottom_mean_return"]
    result["stability"] = {
        "ic_same_sign": bool(dev_ic is not None and val_ic is not None and np.sign(dev_ic) == np.sign(val_ic)),
        "quintile_spread_same_sign": bool(
            dev_spread is not None and val_spread is not None and np.sign(dev_spread) == np.sign(val_spread)
        ),
    }
    return result


def render_markdown(report: dict) -> str:
    forward_start = report["split"].get("true_forward_start") or "等待正式候选冻结后的下一个交易日"
    lines = [
        "# 510300 第一阶段基础特征研究（旧探索产物，未获批准）", "",
        "> 治理状态：`EXPLORATORY_NOT_APPROVED`。本报告生成于正式Target与因子讨论之前，不代表项目选择；当前正式结果以第二轮选择决定为准。", "",
        "## 研究约束", "",
        "- 三类特征并列研究：估值、趋势、波动率；未构造综合分数。",
        "- 所有信号以当日收盘为信息截止点；只能在下一交易日开盘或更晚入场。",
        "- 1日标签只用于信息研究，因股票ETF T+1，不作为新买仓位可执行策略收益。",
        "- 开发集确定分位边界，验证集只复用开发集边界。",
        "- 未来20日标签存在重叠，因此同时报告错位非重叠IC与近似独立样本数。",
        "- JSON报告另含20交易日移动区块Bootstrap的IC区间；多特征多周期检验仍存在数据窥探风险。",
        "- 2025-08-01至2026-08-11曾被探索性检查触及，已降级为受污染回顾测试集；本报告仍不展示其特征效果。", "",
        "## 样本切分", "",
        f"- 开发集：{report['split']['development_start']} 至 {report['split']['development_end']}",
        f"- 验证集：{report['split']['validation_start']} 至 {report['split']['validation_end']}",
        f"- 回顾测试集：{report['split']['final_holdout_start']} 至 {report['split']['final_holdout_end']}（受污染，不用于选参）",
        f"- 真正前向验证起点：{forward_start}", "",
        "## 稳定性摘要", "", "|特征组|特征|周期|开发IC|验证IC|IC同号|分位差同号|", "|---|---|---:|---:|---:|---|---|",
    ]
    for group, results in report["results"].items():
        for item in results:
            dev = item["splits"]["development"]["staggered_non_overlapping_ic"]["median"]
            val = item["splits"]["validation"]["staggered_non_overlapping_ic"]["median"]
            lines.append(
                f"|{group}|{item['feature']}|{item['horizon']}|{'' if dev is None else f'{dev:.4f}'}|"
                f"{'' if val is None else f'{val:.4f}'}|{item['stability']['ic_same_sign']}|"
                f"{item['stability']['quintile_spread_same_sign']}|"
            )
    lines += ["", "详细分位条件收益、命中率、MAE、MFE保存在同目录JSON报告中。", ""]
    return "\n".join(lines)


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    split_config = config["research_split"]
    if split_config.get("final_holdout_status") != "contaminated_by_exploratory_inspection":
        raise ValueError("必须显式记录回顾测试集的治理状态")
    etf = pd.read_parquet(ETF_FILE)
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    features = pd.read_parquet(FEATURE_FILE)
    outcomes = add_forward_outcomes(etf, dividends)
    data = features.merge(outcomes, on="date", how="inner", suffixes=("", "_etf"), validate="one_to_one")
    data["split"] = data["date"].apply(lambda value: assign_split(value, split_config))
    OUTPUT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTPUT_DATA_FILE, index=False)

    results: dict[str, list[dict]] = {}
    for group, feature_names in FEATURE_GROUPS.items():
        results[group] = [
            evaluate_feature(data, feature, horizon, split_config)
            for feature in feature_names for horizon in HORIZONS
        ]
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
        "split": split_config,
        "holdout_policy": "CONTAMINATED_NOT_USED_FOR_SELECTION",
        "holdout_row_count_only": int((data["split"] == "final_holdout").sum()),
        "label_definition": "t日收盘后形成信号，t+1开盘入场，t+h收盘计价；入场当日除息不计入持有人分红",
        "horizon_1_warning": "信息研究标签；新买510300受T+1限制，不能在入场当日卖出",
        "multiple_testing_warning": "共检验多组相关特征与五个周期；结果仅用于筛除弱假设，不可按最高IC挑模型。",
        "feature_groups": FEATURE_GROUPS,
        "results": results,
    }
    OUTPUT_JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MD_FILE.write_text(render_markdown(report), encoding="utf-8")
    print(f"研究数据：{OUTPUT_DATA_FILE}")
    print(f"研究报告：{OUTPUT_MD_FILE}")
    print(f"回顾测试集：CONTAMINATED_NOT_USED_FOR_SELECTION，行数仅记录为 {report['holdout_row_count_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
