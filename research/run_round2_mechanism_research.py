"""运行第二轮价量机制拆解、IF环境检验与增量回归。"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from research.run_registered_factor_research import (
    _assign_evidence,
    _benjamini_hochberg,
    _evaluate_hypothesis,
)


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round2.yaml"
DATA_FILE = ROOT / "data" / "features" / "510300_round2_mechanism_dataset.parquet"
ROUND1_RESEARCH_FILE = ROOT / "reports" / "research" / "registered_factor_research.json"
JSON_FILE = ROOT / "reports" / "research" / "round2_mechanism_research.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "round2_mechanism_research.md"


def _split(dataset: pd.DataFrame, settings: dict, target: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    horizon = 20 if "20d" in target else 5
    split = settings["research_split"]
    label_end = f"label_end_date_{horizon}d"
    development_end = pd.Timestamp(split["development_end"])
    pseudo_end = pd.Timestamp(split["validation_end"])
    development = dataset.loc[
        dataset["date"].between(pd.Timestamp(split["development_start"]), development_end)
        & (dataset[label_end] <= development_end)
    ].copy()
    pseudo = dataset.loc[
        dataset["date"].between(pd.Timestamp(split["validation_start"]), pseudo_end)
        & (dataset[label_end] <= pseudo_end)
    ].copy()
    return development, pseudo, horizon


def _fit_standardized_ols(frame: pd.DataFrame, target: str, columns: list[str], horizon: int) -> dict:
    clean = frame[[target, *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max(40, len(columns) * 10):
        return {"observations": int(len(clean)), "status": "INSUFFICIENT_DATA"}
    means = clean[columns].mean()
    standard_deviations = clean[columns].std(ddof=1)
    if (standard_deviations == 0).any():
        return {"observations": int(len(clean)), "status": "ZERO_VARIANCE"}
    standardized = (clean[columns] - means) / standard_deviations
    x = np.column_stack([np.ones(len(clean)), standardized.to_numpy(dtype=float)])
    y = clean[target].to_numpy(dtype=float)
    xtx_inverse = np.linalg.pinv(x.T @ x)
    coefficients = xtx_inverse @ x.T @ y
    residuals = y - x @ coefficients
    maximum_lag = max(1, horizon)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for index in range(len(clean)):
        vector = x[index][:, None]
        meat += residuals[index] ** 2 * (vector @ vector.T)
    for lag in range(1, maximum_lag + 1):
        weight = 1.0 - lag / (maximum_lag + 1.0)
        gamma = np.zeros_like(meat)
        for index in range(lag, len(clean)):
            current = x[index][:, None]
            previous = x[index - lag][:, None]
            gamma += residuals[index] * residuals[index - lag] * (current @ previous.T)
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inverse @ meat @ xtx_inverse
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_values = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    p_values = [None if not np.isfinite(value) else float(math.erfc(abs(value) / math.sqrt(2.0))) for value in t_values]
    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    residual_sum_squares = float(np.sum(residuals**2))
    r_squared = None if total_sum_squares == 0 else 1.0 - residual_sum_squares / total_sum_squares
    adjusted_r_squared = None
    if r_squared is not None and len(clean) > len(columns) + 1:
        adjusted_r_squared = 1.0 - (1.0 - r_squared) * (len(clean) - 1) / (len(clean) - len(columns) - 1)
    terms = {
        column: {
            "beta_per_development_std": float(coefficients[index + 1]),
            "standard_error_newey_west": float(standard_errors[index + 1]),
            "t_value": None if not np.isfinite(t_values[index + 1]) else float(t_values[index + 1]),
            "p_value": p_values[index + 1],
        }
        for index, column in enumerate(columns)
    }
    return {
        "status": "PASS",
        "observations": int(len(clean)),
        "intercept": float(coefficients[0]),
        "means": {column: float(means[column]) for column in columns},
        "standard_deviations": {column: float(standard_deviations[column]) for column in columns},
        "coefficients": {column: float(coefficients[index + 1]) for index, column in enumerate(columns)},
        "terms": terms,
        "r_squared": None if r_squared is None else float(r_squared),
        "adjusted_r_squared": None if adjusted_r_squared is None else float(adjusted_r_squared),
        "hac_maxlags": maximum_lag,
    }


def _evaluate_fitted_model(model: dict, frame: pd.DataFrame, target: str, columns: list[str]) -> dict:
    if model.get("status") != "PASS":
        return {"status": "MODEL_NOT_FITTED", "observations": 0}
    clean = frame[[target, *columns]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"status": "INSUFFICIENT_DATA", "observations": 0}
    prediction = np.full(len(clean), float(model["intercept"]), dtype=float)
    for column in columns:
        standardized = (
            clean[column].to_numpy(dtype=float) - float(model["means"][column])
        ) / float(model["standard_deviations"][column])
        prediction += standardized * float(model["coefficients"][column])
    actual = clean[target].to_numpy(dtype=float)
    squared_error = float(np.sum((actual - prediction) ** 2))
    baseline_error = float(np.sum((actual - actual.mean()) ** 2))
    correlation = None
    if np.std(prediction) > 0 and np.std(actual) > 0:
        correlation = float(np.corrcoef(prediction, actual)[0, 1])
    return {
        "status": "PASS",
        "observations": int(len(clean)),
        "prediction_correlation": correlation,
        "rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))),
        "oos_r_squared_vs_pseudo_mean": None if baseline_error == 0 else float(1.0 - squared_error / baseline_error),
        "directional_accuracy": float(np.mean(np.sign(prediction) == np.sign(actual))),
    }


def _run_incremental_models(dataset: pd.DataFrame, settings: dict, registry: dict) -> list[dict]:
    output: list[dict] = []
    for target in (registry["targets"]["primary"], registry["targets"]["sensitivity"]):
        development, pseudo, horizon = _split(dataset, settings, target)
        previous_adjusted_r_squared: float | None = None
        for definition in registry["incremental_models"]:
            columns = list(definition["columns"])
            development_fit = _fit_standardized_ols(development, target, columns, horizon)
            pseudo_fit = _fit_standardized_ols(pseudo, target, columns, horizon)
            pseudo_evaluation = _evaluate_fitted_model(development_fit, pseudo, target, columns)
            adjusted = development_fit.get("adjusted_r_squared")
            delta = None
            if adjusted is not None and previous_adjusted_r_squared is not None:
                delta = float(adjusted - previous_adjusted_r_squared)
            if adjusted is not None:
                previous_adjusted_r_squared = float(adjusted)
            output.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "model_id": definition["model_id"],
                    "columns": columns,
                    "development_fit": development_fit,
                    "delta_development_adjusted_r_squared_vs_previous": delta,
                    "pseudo_oos_refit_diagnostic": pseudo_fit,
                    "pseudo_oos_frozen_development_model": pseudo_evaluation,
                }
            )
    return output


def _apply_posthoc_effective_sample_guard(result: dict) -> None:
    """审计重叠标签的有效样本；该规则在首次第二轮输出后追加。"""

    pseudo = result["splits"]["pseudo_oos"]
    effective_observations = int(pseudo["approx_independent_observations"])
    minimum = 20
    uncapped_rating = result["evidence"]["rating"]
    passes = effective_observations >= minimum
    audited_rating = uncapped_rating
    if not passes and uncapped_rating in {"WEAK", "STRONG"}:
        audited_rating = "WEAK_SAMPLE_LIMITED"
    result["evidence"]["uncapped_rating"] = uncapped_rating
    result["evidence"]["rating"] = audited_rating
    result["evidence"]["effective_sample_guard"] = {
        "approx_independent_observations": effective_observations,
        "minimum_required": minimum,
        "passes": passes,
        "governance": "ADDED_AFTER_FIRST_ROUND2_OUTPUT_AND_APPLIED_UNIFORMLY",
        "purpose": "防止D20重叠标签把约199行误读为约199个独立样本",
    }


def _add_studywide_multiple_testing_audit(results: list[dict]) -> dict:
    """把两轮全部日线假设作为同一研究族补做BH审计。"""

    if not ROUND1_RESEARCH_FILE.exists():
        return {"status": "NOT_AVAILABLE", "reason": f"缺少{ROUND1_RESEARCH_FILE}"}
    round1 = json.loads(ROUND1_RESEARCH_FILE.read_text(encoding="utf-8"))
    all_results = list(round1["results"]) + results
    p_values = [item["splits"]["pseudo_oos"]["hac_regression"]["p_value"] for item in all_results]
    q_values = _benjamini_hochberg(p_values)
    first_round_count = len(round1["results"])
    for item, q_value in zip(results, q_values[first_round_count:]):
        item["multiple_testing"]["studywide_bh_q_value_all_daily_hypotheses"] = q_value
        item["multiple_testing"]["studywide_hypothesis_count"] = len(all_results)
        item["multiple_testing"]["studywide_governance"] = "POSTHOC_AUDIT_ACROSS_ROUND1_AND_ROUND2"
    return {
        "status": "PASS",
        "method": "Benjamini-Hochberg",
        "round1_hypothesis_count": first_round_count,
        "round2_hypothesis_count": len(results),
        "studywide_hypothesis_count": len(all_results),
        "governance": "POSTHOC_AUDIT_ACROSS_ROUND1_AND_ROUND2",
    }


def _format(value: float | None, percentage: bool = False) -> str:
    if value is None:
        return ""
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def _render_markdown(report: dict) -> str:
    lines = [
        "# 510300第二轮价量机制与IF环境研究", "",
        "## 治理边界", "",
        "- 本轮由第一轮价量候选后验派生，只能做机制拆解和候选收缩，不能提供独立确认。",
        "- 成交量冲击严格使用此前20日均值和标准差，并固定截断至[-4,4]。",
        "- 高成交量趋势交互只在成交量冲击为正时生效，修正旧乘积把缩量下跌记成正值的问题。",
        "- IF主连相对强弱受换月影响，本轮仅作环境探索；原始持仓量变化未注册。", "",
        "- 首次输出后追加统一审计：pseudo-OOS折算独立观测少于20时，WEAK/STRONG封顶为WEAK_SAMPLE_LIMITED；该规则不伪装成事前登记。", "",
        "- 另报告第一、二轮合计28个日线假设的研究级BH q值；轮内q值继续保留，避免覆盖原始记录。", "",
        "## 单因子结果", "",
        "|ID|因子|目标|开发Spearman|pseudo-OOS Spearman|成本后有利-不利|轮内BH q|全研究BH q|得分|评级|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        development = item["splits"]["development"]
        pseudo = item["splits"]["pseudo_oos"]
        lines.append(
            f"|{item['hypothesis_id']}|{item['factor_name_cn']}|{item['target']}|"
            f"{_format(development['spearman'])}|{_format(pseudo['spearman'])}|"
            f"{_format(pseudo['conditional_returns']['favorable_minus_adverse_mean_net_return'], True)}|"
            f"{_format(item['multiple_testing']['pseudo_oos_hac_bh_q_value'])}|"
            f"{_format(item['multiple_testing'].get('studywide_bh_q_value_all_daily_hypotheses'))}|"
            f"{item['evidence']['score']}/7|{item['evidence']['rating']}|"
        )
    lines += ["", "## 增量回归", "", "|目标|模型|开发调整R²|相对前一模型增量|pseudo-OOS冻结模型R²|预测相关|", "|---|---|---:|---:|---:|---:|"]
    for model in report["incremental_models"]:
        development = model["development_fit"]
        pseudo = model["pseudo_oos_frozen_development_model"]
        lines.append(
            f"|{model['target']}|{model['model_id']}|{_format(development.get('adjusted_r_squared'))}|"
            f"{_format(model['delta_development_adjusted_r_squared_vs_previous'])}|"
            f"{_format(pseudo.get('oos_r_squared_vs_pseudo_mean'))}|{_format(pseudo.get('prediction_correlation'))}|"
        )
    interaction_rows = [item for item in report["incremental_models"] if item["model_id"] == "PRICE_VOLUME_INTERACTION"]
    lines += ["", "## 交互项判读", ""]
    for item in interaction_rows:
        column = "round2_etf_high_volume_trend_20d"
        development_term = item["development_fit"].get("terms", {}).get(column, {})
        pseudo_term = item["pseudo_oos_refit_diagnostic"].get("terms", {}).get(column, {})
        lines.append(
            f"- `{item['target']}`：开发期交互β={_format(development_term.get('beta_per_development_std'), True)}、"
            f"p={_format(development_term.get('p_value'))}；pseudo-OOS重拟合诊断β="
            f"{_format(pseudo_term.get('beta_per_development_std'), True)}、p={_format(pseudo_term.get('p_value'))}。"
        )
    lines += ["", "完整HAC、非重叠样本、Bootstrap、年度稳定性和冻结模型诊断保存在同名JSON中。", ""]
    return "\n".join(lines)


def main() -> int:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"缺少第二轮因子数据集：{DATA_FILE}")
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
            _apply_posthoc_effective_sample_guard(results[index])
    studywide_audit = _add_studywide_multiple_testing_audit(results)
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "derivation_note": registry["derivation_note"],
        "hypothesis_count": len(results),
        "results": results,
        "incremental_models": _run_incremental_models(dataset, settings, registry),
        "studywide_multiple_testing_audit": studywide_audit,
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text(_render_markdown(report), encoding="utf-8")
    print(f"第二轮机制报告：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
