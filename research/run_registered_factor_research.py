"""运行预注册的单标的时间序列因子检验。"""

from __future__ import annotations

import json
import math
import zlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry.yaml"
DATA_FILE = ROOT / "data" / "features" / "510300_registered_factor_dataset.parquet"
JSON_FILE = ROOT / "reports" / "research" / "registered_factor_research.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "registered_factor_research.md"


def _safe_correlation(frame: pd.DataFrame, x: str, y: str, method: str) -> float | None:
    clean = frame[[x, y]].dropna()
    if len(clean) < 10 or clean[x].nunique() < 2 or clean[y].nunique() < 2:
        return None
    value = clean[x].corr(clean[y], method=method)
    return None if pd.isna(value) else float(value)


def _non_overlapping_correlations(
    frame: pd.DataFrame,
    factor: str,
    target: str,
    horizon: int,
) -> dict:
    values: list[float] = []
    sizes: list[int] = []
    for offset in range(horizon):
        sample = frame.iloc[offset::horizon][[factor, target]].dropna()
        if len(sample) < 10 or sample[factor].nunique() < 2 or sample[target].nunique() < 2:
            continue
        correlation = sample[factor].corr(sample[target], method="spearman")
        if pd.notna(correlation):
            values.append(float(correlation))
            sizes.append(int(len(sample)))
    if not values:
        return {
            "subsamples": 0, "median": None, "minimum": None, "maximum": None,
            "same_sign_ratio": None, "sample_sizes": [],
        }
    median = float(np.median(values))
    return {
        "subsamples": len(values),
        "median": median,
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "same_sign_ratio": float(np.mean(np.sign(values) == np.sign(median))),
        "sample_sizes": sizes,
    }


def _hac_regression(frame: pd.DataFrame, factor: str, target: str, horizon: int) -> dict:
    clean = frame[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 30 or clean[factor].std(ddof=1) == 0:
        return {"observations": int(len(clean)), "beta": None, "t_value": None, "p_value": None}
    standardized = ((clean[factor] - clean[factor].mean()) / clean[factor].std(ddof=1)).to_numpy(dtype=float)
    y = clean[target].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(clean), dtype=float), standardized])
    xtx_inverse = np.linalg.inv(x.T @ x)
    coefficients = xtx_inverse @ x.T @ y
    residuals = y - x @ coefficients
    maximum_lag = max(1, horizon)
    meat = np.zeros((2, 2), dtype=float)
    for index in range(len(clean)):
        vector = x[index][:, None]
        meat += residuals[index] ** 2 * (vector @ vector.T)
    for lag in range(1, maximum_lag + 1):
        weight = 1.0 - lag / (maximum_lag + 1.0)
        gamma = np.zeros((2, 2), dtype=float)
        for index in range(lag, len(clean)):
            current = x[index][:, None]
            previous = x[index - lag][:, None]
            gamma += residuals[index] * residuals[index - lag] * (current @ previous.T)
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    t_value = None if standard_error == 0 else float(coefficients[1] / standard_error)
    p_value = None if t_value is None else float(math.erfc(abs(t_value) / math.sqrt(2.0)))
    return {
        "observations": int(len(clean)),
        "beta_per_factor_std": float(coefficients[1]),
        "standard_error_newey_west": standard_error,
        "t_value": t_value,
        "p_value": p_value,
        "hac_maxlags": maximum_lag,
    }


def _moving_block_bootstrap(
    frame: pd.DataFrame,
    factor: str,
    target: str,
    horizon: int,
    iterations: int = 500,
) -> dict:
    clean = frame[[factor, target]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    block_size = max(20, horizon)
    if len(clean) < block_size * 3:
        return {"iterations": 0, "block_size": block_size, "median": None, "ci_2_5pct": None, "ci_97_5pct": None}
    x = clean[factor].rank(method="average").to_numpy(dtype=float)
    y = clean[target].rank(method="average").to_numpy(dtype=float)
    maximum_start = len(clean) - block_size
    blocks_needed = int(np.ceil(len(clean) / block_size))
    seed = zlib.crc32(f"{factor}|{target}|{len(clean)}".encode("utf-8"))
    generator = np.random.default_rng(seed)
    correlations: list[float] = []
    for _ in range(iterations):
        starts = generator.integers(0, maximum_start + 1, size=blocks_needed)
        indices = np.concatenate([np.arange(start, start + block_size) for start in starts])[: len(clean)]
        sampled_x = x[indices]
        sampled_y = y[indices]
        if np.std(sampled_x) > 0 and np.std(sampled_y) > 0:
            correlations.append(float(np.corrcoef(sampled_x, sampled_y)[0, 1]))
    if not correlations:
        return {"iterations": 0, "block_size": block_size, "median": None, "ci_2_5pct": None, "ci_97_5pct": None}
    return {
        "iterations": len(correlations),
        "block_size": block_size,
        "median": float(np.median(correlations)),
        "ci_2_5pct": float(np.quantile(correlations, 0.025)),
        "ci_97_5pct": float(np.quantile(correlations, 0.975)),
    }


def _bucket_summary(
    frame: pd.DataFrame,
    factor: str,
    target: str,
    mae: str,
    mfe: str,
    edges: list[float],
    expected_sign: str,
) -> dict:
    labels = list(range(1, len(edges) + 2))
    bucket = pd.cut(frame[factor], [-np.inf, *edges, np.inf], labels=labels, include_lowest=True)
    temp = frame.assign(_bucket=bucket)
    buckets: list[dict] = []
    for group, values in temp.groupby("_bucket", observed=True):
        clean = values[[target, mae, mfe]].dropna()
        if clean.empty:
            continue
        buckets.append(
            {
                "bucket": int(group),
                "observations": int(len(clean)),
                "mean_net_return": float(clean[target].mean()),
                "median_net_return": float(clean[target].median()),
                "hit_rate": float((clean[target] > 0).mean()),
                "mean_mae": float(clean[mae].mean()),
                "mean_mfe": float(clean[mfe].mean()),
            }
        )
    by_bucket = {item["bucket"]: item for item in buckets}
    low = by_bucket.get(1)
    high = by_bucket.get(len(labels))
    effect = None
    minimum_extreme_observations = max(20, int(np.ceil(len(temp[[factor, target]].dropna()) * 0.10)))
    extreme_bucket_support = bool(
        low is not None and high is not None
        and low["observations"] >= minimum_extreme_observations
        and high["observations"] >= minimum_extreme_observations
    )
    if low is not None and high is not None:
        if expected_sign == "positive":
            effect = high["mean_net_return"] - low["mean_net_return"]
        else:
            effect = low["mean_net_return"] - high["mean_net_return"]
    return {
        "buckets": buckets,
        "favorable_minus_adverse_mean_net_return": effect,
        "minimum_required_extreme_bucket_observations": minimum_extreme_observations,
        "extreme_bucket_support_pass": extreme_bucket_support,
    }


def _year_direction_ratio(
    frame: pd.DataFrame,
    factor: str,
    target: str,
    expected_sign: str,
) -> dict:
    correlations: dict[str, float] = {}
    expected = 1.0 if expected_sign == "positive" else -1.0
    for year, values in frame.groupby(frame["date"].dt.year):
        correlation = _safe_correlation(values, factor, target, "spearman")
        if correlation is not None:
            correlations[str(year)] = correlation
    ratio = None if not correlations else float(np.mean([np.sign(value) == expected for value in correlations.values()]))
    return {"correlations": correlations, "expected_direction_ratio": ratio}


def _benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    valid = [(index, value) for index, value in enumerate(p_values) if value is not None and np.isfinite(value)]
    output: list[float | None] = [None] * len(p_values)
    if not valid:
        return output
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    adjusted = [0.0] * count
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        _, value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, float(value) * count / rank)
        adjusted[reverse_index] = min(running, 1.0)
    for (original_index, _), value in zip(ordered, adjusted):
        output[original_index] = value
    return output


def _direction_matches(value: float | None, expected_sign: str) -> bool:
    if value is None or not np.isfinite(value) or value == 0:
        return False
    return value > 0 if expected_sign == "positive" else value < 0


def _evaluate_hypothesis(
    dataset: pd.DataFrame,
    definition: dict,
    hypothesis: dict,
    split: dict,
) -> dict:
    factor = definition["column"]
    horizon = 20 if "20d" in hypothesis["target"] else 5
    target = hypothesis["target"]
    mae = f"exec_mae_{horizon}d"
    mfe = f"exec_mfe_{horizon}d"
    end_date = f"label_end_date_{horizon}d"
    development = dataset.loc[
        dataset["date"].between(pd.Timestamp(split["development_start"]), pd.Timestamp(split["development_end"]))
        & (dataset[end_date] <= pd.Timestamp(split["development_end"]))
    ].copy()
    pseudo_start = pd.Timestamp(split["validation_start"])
    pseudo_end = pd.Timestamp(split["validation_end"])
    pseudo = dataset.loc[
        dataset["date"].between(pseudo_start, pseudo_end)
        & (dataset[end_date] <= pseudo_end)
    ].copy()
    development_clean = development[[factor, target]].dropna()
    edges = [] if development_clean.empty else sorted(set(development_clean[factor].quantile([0.2, 0.4, 0.6, 0.8]).tolist()))
    result = {
        "hypothesis_id": hypothesis["hypothesis_id"],
        "factor_id": definition["factor_id"],
        "factor": factor,
        "factor_name_cn": definition["name_cn"],
        "target": target,
        "horizon": horizon,
        "expected_sign": definition["expected_sign"],
        "economic_hypothesis": definition["economic_hypothesis"],
        "development_bucket_edges": edges,
        "splits": {},
    }
    for name, frame in (("development", development), ("pseudo_oos", pseudo)):
        clean = frame[[factor, target, mae, mfe, "date"]].replace([np.inf, -np.inf], np.nan).dropna()
        result["splits"][name] = {
            "observations": int(len(clean)),
            "approx_independent_observations": int(len(clean) // horizon),
            "pearson": _safe_correlation(clean, factor, target, "pearson"),
            "spearman": _safe_correlation(clean, factor, target, "spearman"),
            "hac_regression": _hac_regression(clean, factor, target, horizon),
            "non_overlapping_spearman": _non_overlapping_correlations(clean, factor, target, horizon),
            "moving_block_bootstrap_spearman": _moving_block_bootstrap(clean, factor, target, horizon),
            "conditional_returns": _bucket_summary(clean, factor, target, mae, mfe, edges, definition["expected_sign"]),
            "year_stability": _year_direction_ratio(clean, factor, target, definition["expected_sign"]),
        }
    return result


def _assign_evidence(result: dict, q_value: float | None) -> None:
    expected = result["expected_sign"]
    development = result["splits"]["development"]
    pseudo = result["splits"]["pseudo_oos"]
    checks = {
        "development_spearman_direction": _direction_matches(development["spearman"], expected),
        "pseudo_oos_spearman_direction": _direction_matches(pseudo["spearman"], expected),
        "pseudo_oos_nonoverlap_direction": _direction_matches(pseudo["non_overlapping_spearman"]["median"], expected),
        "pseudo_oos_costed_bucket_effect_positive": (
            pseudo["conditional_returns"]["favorable_minus_adverse_mean_net_return"] is not None
            and pseudo["conditional_returns"]["favorable_minus_adverse_mean_net_return"] > 0
            and pseudo["conditional_returns"]["extreme_bucket_support_pass"]
        ),
        "pseudo_oos_year_direction_ratio_ge_60pct": (
            pseudo["year_stability"]["expected_direction_ratio"] is not None
            and pseudo["year_stability"]["expected_direction_ratio"] >= 0.60
        ),
        "pseudo_oos_bootstrap_median_direction": _direction_matches(
            pseudo["moving_block_bootstrap_spearman"]["median"], expected
        ),
        "pseudo_oos_hac_direction_and_q_le_20pct": (
            _direction_matches(pseudo["hac_regression"].get("beta_per_factor_std"), expected)
            and q_value is not None
            and q_value <= 0.20
        ),
    }
    score = int(sum(checks.values()))
    development_effect = development["conditional_returns"]["favorable_minus_adverse_mean_net_return"]
    pseudo_effect = pseudo["conditional_returns"]["favorable_minus_adverse_mean_net_return"]
    both_reversed = (
        development_effect is not None and development_effect < 0
        and pseudo_effect is not None and pseudo_effect < 0
    )
    extreme_support = pseudo["conditional_returns"]["extreme_bucket_support_pass"]
    if both_reversed or score <= 1:
        rating = "REJECTED"
    elif score <= 3:
        rating = "INCONCLUSIVE"
    elif score <= 5:
        rating = "WEAK"
    elif extreme_support:
        rating = "STRONG"
    else:
        rating = "WEAK"
    result["multiple_testing"] = {"pseudo_oos_hac_bh_q_value": q_value}
    result["evidence"] = {"score": score, "rating": rating, "checks": checks}


def _render_markdown(report: dict) -> str:
    lines = [
        "# 510300预注册因子时间序列研究", "",
        "## 治理口径", "",
        "- 这是单标的时间序列研究，不使用横截面Rank IC。",
        "- D20为主目标，D5为敏感性目标；标签包含分红、滑点、佣金、最低佣金和整手约束。",
        "- 同时报告重叠样本、错位非重叠样本、HAC/Newey-West和移动区块Bootstrap。",
        "- 历史时间顺序验证仅称pseudo-OOS；没有严格未见历史留出集。",
        "- 评级是证据强度，不是实盘批准。", "",
        "## 假设结果", "",
        "|ID|因子|目标|预期方向|开发Spearman|pseudo-OOS Spearman|非重叠中位数|成本后有利-不利|BH q值|得分|评级|",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        development = item["splits"]["development"]
        pseudo = item["splits"]["pseudo_oos"]
        values = [
            item["hypothesis_id"], item["factor_name_cn"], item["target"], item["expected_sign"],
            development["spearman"], pseudo["spearman"], pseudo["non_overlapping_spearman"]["median"],
            pseudo["conditional_returns"]["favorable_minus_adverse_mean_net_return"],
            item["multiple_testing"]["pseudo_oos_hac_bh_q_value"], item["evidence"]["score"], item["evidence"]["rating"],
        ]
        formatted = []
        for index, value in enumerate(values):
            if index in (4, 5, 6, 7, 8):
                formatted.append("" if value is None else f"{value:.4f}")
            else:
                formatted.append(str(value))
        lines.append("|" + "|".join(formatted) + "|")
    primary = [item for item in report["results"] if item["target"] == "exec_total_return_20d_net"]
    candidates = [item for item in primary if item["evidence"]["rating"] in {"WEAK", "STRONG"}]
    lines += ["", "## D20证据候选", ""]
    if candidates:
        for item in sorted(candidates, key=lambda value: value["evidence"]["score"], reverse=True):
            lines.append(f"- `{item['factor_id']}`：{item['factor_name_cn']}，{item['evidence']['rating']}（{item['evidence']['score']}/7）。")
    else:
        lines.append("- 本轮没有达到WEAK或STRONG的D20因子；不得强制选择赢家。")
    lines += ["", "详细条件分桶、年度稳定性、HAC和Bootstrap结果保存在同名JSON报告中。", ""]
    return "\n".join(lines)


def main() -> int:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"缺少注册因子数据集：{DATA_FILE}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    dataset = pd.read_parquet(DATA_FILE)
    dataset["date"] = pd.to_datetime(dataset["date"])
    for horizon in (5, 20):
        dataset[f"label_end_date_{horizon}d"] = pd.to_datetime(dataset[f"label_end_date_{horizon}d"])
    definitions = {item["factor_id"]: item for item in registry["factor_definitions"]}
    results = [
        _evaluate_hypothesis(dataset, definitions[item["factor_id"]], item, settings["research_split"])
        for item in registry["hypotheses"]
    ]
    for target in sorted({item["target"] for item in results}):
        indices = [index for index, item in enumerate(results) if item["target"] == target]
        p_values = [results[index]["splits"]["pseudo_oos"]["hac_regression"]["p_value"] for index in indices]
        q_values = _benjamini_hochberg(p_values)
        for index, q_value in zip(indices, q_values):
            _assign_evidence(results[index], q_value)
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "primary_target": registry["targets"]["primary"],
        "sensitivity_target": registry["targets"]["sensitivity"],
        "hypothesis_count": len(results),
        "method": "时间序列Pearson/Spearman、HAC/Newey-West、错位非重叠样本、条件收益和移动区块Bootstrap",
        "results": results,
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text(_render_markdown(report), encoding="utf-8")
    print(f"研究报告：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
