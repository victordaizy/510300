"""NBS V2 严格前序 G2 实现；来源未通过时不得调用标签读取器。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

from research.nbs_v2_common import ContractError, committed, csv_bytes, git, identity, load_json, now, verify, write_once
from research.stk_mins_source_admission_v2 import normalize

MODEL = "config/510300_nbs_1000_negative_information_drift_v2.yaml"
SOURCE_CONTRACT = "config/510300_stk_mins_source_admission_v2.yaml"
SOURCE_RESULT = "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/source_adjudication.json"
OUT = "reports/research/510300_nbs_v2_g2_execution_v1"
MANIFEST = f"{OUT}/implementation_manifest.json"
PLAN = f"{OUT}/event_baseline_placebo_plan_pre_return.csv"
SCOPE = ["research/nbs_v2_g2_engine.py", "scripts/run_510300_nbs_v2_g2.py", "tests/test_510300_nbs_v2_g2.py", MODEL]


class StatisticalNoView(ContractError):
    """冻结样本不能识别原模型；不得通过删样本或替换尺度修复。"""


def admitted_loader(source: dict, loader: Callable):
    checks = source.get("checks", {})
    if (source.get("state") != "PASS_REFETCHED_DATA_ORIGINAL_V2_CONTRACT"
            or source.get("source_pass") is not True or not checks or not all(checks.values())
            or source.get("source_hard_gates_changed") is not False):
        raise ContractError("G0 未按原合同通过，禁止读取标签和运行 G2")
    return loader()


def make_plan(ledger: pd.DataFrame, complete_open_days: pd.DatetimeIndex) -> pd.DataFrame:
    """只分配日期和时钟，不读取价格、收益或特征。"""
    days = pd.DatetimeIndex(complete_open_days).normalize().sort_values().unique()
    family = set(pd.to_datetime(ledger.scheduled_date.dropna()).dt.normalize())
    non_event = [d for d in days if d not in family]
    selected = ledger.loc[ledger.model_pre_return_eligibility.eq(True)].sort_values("model_event_ordinal")
    if len(selected) != 86 or selected.model_event_ordinal.astype(int).tolist() != list(range(1, 87)):
        raise ContractError("模型事件必须是原账本固定 86 个事件，不得删选")
    if ledger.final_event_eligibility.eq(True).sum() != 88:
        raise ContractError("日程合格事件必须保持 88 个")
    used, rows = set(), []
    for e in selected.itertuples():
        event_day = pd.Timestamp(e.scheduled_date).normalize()
        prior = [d for d in non_event if d < event_day]
        available = [d for d in prior if d not in used]
        if len(prior) < 60 or not available:
            raise ContractError("真实事件缺少严格此前基线或安慰剂 A 匹配日")
        a_day = available[-1]
        a_prior = [d for d in non_event if d < a_day]
        if len(a_prior) < 60:
            raise ContractError("安慰剂 A 不足其自身此前 60 个非事件日，停止而不换匹配规则")
        used.add(a_day)
        rows.append({"event_id": e.event_id, "event_date": event_day.strftime("%Y-%m-%d"),
                     "ordinal": int(e.model_event_ordinal), "era": e.era,
                     "placebo_a_date": a_day.strftime("%Y-%m-%d"),
                     "main_and_b_baseline_dates": "|".join(d.strftime("%Y-%m-%d") for d in prior[-60:]),
                     "placebo_a_baseline_dates": "|".join(d.strftime("%Y-%m-%d") for d in a_prior[-60:])})
    return pd.DataFrame(rows)


def ols_hc3(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """带截距 OLS；HC3 等价于逐点删除后系数变化的外积和。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite([x, y]).all():
        raise StatisticalNoView("OLS 输入缺失、非有限或样本不足")
    design = np.column_stack([np.ones(len(x)), x])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2:
        raise StatisticalNoView("OLS 唯一特征退化，禁止加特征或伪造变异")
    bread = np.linalg.inv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, bread, design)
    if (1 - leverage <= 1e-12).any():
        raise StatisticalNoView("HC3 杠杆值退化，不能构造有效均值标准误")
    residual = y - design @ coef
    scores = design * (residual / (1 - leverage))[:, None]
    covariance = bread @ (scores.T @ scores) @ bread
    return coef, covariance


def prequential(frame: pd.DataFrame, initial: int = 36) -> pd.DataFrame:
    if len(frame) <= initial or not frame.observation_date.is_monotonic_increasing:
        raise ContractError("前序事件数或时间顺序不合格")
    records = []
    for i in range(initial, len(frame)):
        training = frame.iloc[:i]
        current = frame.iloc[i]
        if not (pd.to_datetime(training.maturity_at) < pd.Timestamp(current.origin_at)).all():
            raise ContractError("发现未成熟训练标签，停止而不改变训练样本")
        coef, covariance = ols_hc3(training.X.to_numpy(), training.Y.to_numpy())
        vector = np.array([1., float(current.X)])
        variance = float(vector @ covariance @ vector)
        if variance < -1e-15 or not np.isfinite(variance):
            raise StatisticalNoView("HC3 均值预测方差不合格")
        b0, b1 = float(training.Y.mean()), float(vector @ coef)
        se = float(np.sqrt(max(variance, 0)))
        record = current.to_dict()
        record.update({"training_events": i, "training_last_maturity": training.maturity_at.iloc[-1],
                       "alpha": coef[0], "beta": coef[1], "b0": b0, "b1": b1,
                       "mean_se_hc3": se, "lcb": b1 - 1.645 * se,
                       "b0_squared_error": (float(current.Y) - b0) ** 2,
                       "b1_squared_error": (float(current.Y) - b1) ** 2})
        records.append(record)
    return pd.DataFrame(records)


def bootstrap_slopes(x: np.ndarray, y: np.ndarray, indices: np.ndarray) -> np.ndarray:
    xb, yb = x[indices], y[indices]
    xc, yc = xb - xb.mean(axis=1, keepdims=True), yb - yb.mean(axis=1, keepdims=True)
    denominator = (xc * xc).sum(axis=1)
    return np.divide((xc * yc).sum(axis=1), denominator,
                     out=np.full(len(indices), np.nan), where=denominator > 0)


def g2_statistics(predictions: dict[str, pd.DataFrame], repetitions: int, seed: int) -> tuple[dict, pd.DataFrame]:
    main = predictions["MAIN"]
    if len(main) != 50 or main.era.value_counts().to_dict() != {"ERA_1": 17, "ERA_2": 17, "ERA_3": 16}:
        raise ContractError("评价样本必须保持 50 个以及 17/17/16 切分")
    indices = np.random.default_rng(seed).integers(0, len(main), size=(repetitions, len(main)))
    metrics, draws = {}, {}
    for arm, p in predictions.items():
        if p.event_id.tolist() != main.event_id.tolist():
            raise ContractError("安慰剂必须与主实验使用相同事件序号的成对重采样")
        coef, _ = ols_hc3(p.X.to_numpy(), p.Y.to_numpy())
        sampled = bootstrap_slopes(p.X.to_numpy(), p.Y.to_numpy(), indices)
        finite = np.isfinite(sampled).all()
        draws[arm] = sampled
        metrics[arm] = {"evaluation_events": len(p), "b0_mse": p.b0_squared_error.mean(),
                        "b1_mse": p.b1_squared_error.mean(),
                        "mse_improvement": p.b0_squared_error.mean() - p.b1_squared_error.mean(),
                        "relative_mse_improvement": 1 - p.b1_squared_error.mean() / p.b0_squared_error.mean()
                        if p.b0_squared_error.mean() > 0 else None,
                        "evaluation_beta": coef[1], "beta_lower_10pct": np.quantile(sampled, .10) if finite else None,
                        "degenerate_bootstrap_samples": int((~np.isfinite(sampled)).sum()),
                        "spearman_X_Y": float(spearmanr(p.X, p.Y).statistic),
                        "era_mse": p.groupby("era")[["b0_squared_error", "b1_squared_error"]].mean().to_dict("index")}
    superiority = {}
    for arm in ["A", "B"]:
        diff = draws["MAIN"] - draws[arm]
        valid = np.isfinite(diff).all()
        superiority[arm] = {"point_beta_difference": metrics["MAIN"]["evaluation_beta"] - metrics[arm]["evaluation_beta"],
                            "paired_beta_difference_lower_10pct": np.quantile(diff, .10) if valid else None,
                            "degenerate_paired_draws": int((~np.isfinite(diff)).sum()),
                            "pass": bool(valid and np.quantile(diff, .10) > 0)}
    m = metrics["MAIN"]
    era_better = {era: row["b1_squared_error"] < row["b0_squared_error"] for era, row in m["era_mse"].items()}
    gates = {"prequential_mse_improvement_positive": m["mse_improvement"] > 0,
             "evaluation_beta_positive": m["evaluation_beta"] > 0,
             "beta_bootstrap_lower_positive": m["beta_lower_10pct"] is not None and m["beta_lower_10pct"] > 0,
             "at_least_two_better_eras": sum(era_better.values()) >= 2,
             "latest_era_better": era_better["ERA_3"], "spearman_positive": m["spearman_X_Y"] > 0,
             "stronger_than_placebo_A": superiority["A"]["pass"],
             "stronger_than_placebo_B": superiority["B"]["pass"],
             "no_degenerate_bootstrap": all(v["degenerate_bootstrap_samples"] == 0 for v in metrics.values())}
    return {"g2_pass": all(gates.values()), "gates": gates, "arms": metrics,
            "paired_placebo_superiority": superiority, "bootstrap_repetitions": repetitions,
            "seed": seed, "resampling": "SAME_EVENT_INDICES_FOR_ALL_THREE_ARMS_NO_REDRAW"}, pd.DataFrame(draws)


def construct_labels(minute: pd.DataFrame, plan: pd.DataFrame, source: dict, model: dict) -> dict[str, pd.DataFrame]:
    """本函数只可在来源通过且独占标签 claim 落盘之后调用。"""
    frame = normalize(minute)
    windows = dict(source["windows"])
    windows.update({f"shift_{name}": model["placebos"]["B"][f"{name}_labels"] for name in ["pre", "reaction", "entry", "exit"]})
    table = {}
    for name, labels in windows.items():
        subset = frame.loc[frame.clock.isin(labels)]
        grouped = subset.groupby("date")
        totals = grouped[["amount", "vol"]].sum()
        if not grouped.size().eq(5).all() or (totals <= 0).any().any() or not np.isfinite(totals).all().all():
            raise ContractError("已准入来源出现无效窗口，停止而不替换价格")
        table[name] = totals.amount / totals.vol
    prices = pd.DataFrame(table)
    arms = {}
    for arm in ["MAIN", "A", "B"]:
        prefix = "shift_" if arm == "B" else ""
        g = np.log(prices[f"{prefix}reaction"] / prices[f"{prefix}pre"])
        rows = []
        for p in plan.itertuples():
            day = pd.Timestamp(p.placebo_a_date if arm == "A" else p.event_date)
            baseline_text = p.placebo_a_baseline_dates if arm == "A" else p.main_and_b_baseline_dates
            baseline_dates = pd.to_datetime(baseline_text.split("|"))
            if len(baseline_dates) != 60 or not (baseline_dates < day).all():
                raise ContractError("基线日期违反严格此前 60 日规则")
            history = g.reindex(baseline_dates).to_numpy()
            if not np.isfinite(history).all() or day not in prices.index:
                raise StatisticalNoView("基线或事件价格缺失，不得插值")
            median = float(np.median(history))
            scale = float(1.4826 * np.median(abs(history - median)))
            if scale <= 0:
                raise StatisticalNoView("原 60 日 MAD 尺度不为正，保持 NO_VIEW")
            z = float((g.loc[day] - median) / scale)
            x = max(-z, 0.)
            y = -float(np.log(prices.loc[day, f"{prefix}exit"] / prices.loc[day, f"{prefix}entry"]))
            origin_clock = "09:50:00" if arm == "B" else "10:05:00"
            rows.append({"arm": arm, "event_id": p.event_id, "ordinal": p.ordinal, "era": p.era,
                         "observation_date": day.strftime("%Y-%m-%d"),
                         "origin_at": f"{day:%Y-%m-%d}T{origin_clock}+08:00",
                         "maturity_at": f"{day:%Y-%m-%d}T14:55:00+08:00",
                         "g": g.loc[day], "baseline_median": median, "baseline_mad_scale": scale,
                         "X": x, "Y": y})
        arms[arm] = pd.DataFrame(rows)
    return arms


def freeze(root: Path) -> dict:
    model = yaml.safe_load((root / MODEL).read_text("utf-8"))
    source = yaml.safe_load((root / SOURCE_CONTRACT).read_text("utf-8"))
    verify(root, load_json(root / "config/510300_stk_mins_source_admission_v2_manifest.json")["identities"])
    initial = load_json(root / source["inputs"]["acquisition"])
    rec = load_json(root / initial["records"][0]["receipt_relative_path"])
    cal = pd.read_parquet(root / rec["normalized_relative_path"])
    days = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open.astype(int).eq(1), "cal_date"].astype(str)))
    ledger = pd.read_csv(root / model["sample"]["ledger"])
    plan = make_plan(ledger, days)
    write_once(root / PLAN, csv_bytes(plan))
    paths = SCOPE + [SOURCE_CONTRACT, PLAN, model["sample"]["ledger"],
                     "research/nbs_v2_common.py", "research/stk_mins_source_admission_v2.py",
                     "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/manifest.json"]
    result = {"state": "IMPLEMENTATION_FROZEN_WITH_ZERO_REAL_EVENT_RETURN_READS", "generated_at": now(),
              "identities": [identity(root, p) for p in paths], "events": len(plan),
              "era_counts": plan.era.value_counts().to_dict(),
              "source_required_state": "PASS_REFETCHED_DATA_ORIGINAL_V2_CONTRACT",
              "source_report": SOURCE_RESULT, "quantile_method": "NUMPY_DEFAULT_LINEAR",
              "hc3_reference": "https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.HC3_se.html",
              "event_return_values_read": False, "signal_density_read": False, "portfolio_run": False}
    write_once(root / MANIFEST, result)
    return {k: v for k, v in result.items() if k != "identities"}


def run(root: Path) -> dict:
    manifest = load_json(root / MANIFEST)
    verify(root, manifest["identities"])
    verify(root, load_json(root / "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/manifest.json")["identities"])
    committed(root, SCOPE + [MANIFEST, PLAN])
    source_result = load_json(root / SOURCE_RESULT)
    try:
        admitted_loader(source_result, lambda: None)
    except ContractError:
        blocked = {"state": "NOT_RUN_BLOCKED_BY_G0", "checked_at": now(),
                   "source": identity(root, SOURCE_RESULT), "implementation": identity(root, MANIFEST),
                   "event_return_values_read": False, "event_labels_created": 0,
                   "B0_B1_trained": False, "placebos_run": False, "G3": "NOT_RUN", "G4": "NOT_RUN",
                   "position_impact": 0}
        write_once(root / OUT / "gate_check_receipt.json", blocked)
        return blocked
    claim_path = root / OUT / "return_read_claim.json"
    if claim_path.exists():
        raise ContractError("G2 标签读取 claim 已领取，禁止重复运行或调整结果")
    committed(root, [SOURCE_RESULT,
                    "reports/data_quality/510300_nbs_v2_source_refetch_v1_2_20260905/acquisition_receipt.json"])
    verify(root, [source_result["candidate"]])
    claim = {"state": "CLAIMED_ONCE_FOR_FROZEN_G2_LABEL_CONSTRUCTION", "claimed_at": now(),
             "source": identity(root, SOURCE_RESULT), "candidate": source_result["candidate"],
             "implementation": identity(root, MANIFEST), "freeze_commit": git(root, "rev-parse", "HEAD")}
    write_once(claim_path, claim)
    model = yaml.safe_load((root / MODEL).read_text("utf-8"))
    source = yaml.safe_load((root / SOURCE_CONTRACT).read_text("utf-8"))
    plan = pd.read_csv(root / PLAN)
    arms = admitted_loader(source_result, lambda: construct_labels(
        pd.read_parquet(root / source_result["candidate"]["path"]), plan, source, model))
    for name, data in arms.items():
        write_once(root / OUT / f"labels_{name}.csv", csv_bytes(data))
    predictions = {name: prequential(data, model["models"]["initial_training"]) for name, data in arms.items()}
    stats, bootstrap = g2_statistics(predictions, model["statistics"]["bootstrap_repetitions"], model["statistics"]["seed"])
    for name, frame in predictions.items():
        write_once(root / OUT / f"predictions_{name}.csv", csv_bytes(frame))
    write_once(root / OUT / "paired_bootstrap_slopes.csv", csv_bytes(bootstrap))
    stats.update({"state": "PASS_G2_READY_FOR_FROZEN_G3" if stats["g2_pass"] else "REJECTED_FROZEN_NBS_FAMILY_NO_RESCUE",
                  "generated_at": now(), "event_return_values_read": True, "labels_created": 258,
                  "signal_density_read": False, "G3": "NOT_RUN", "G4": "NOT_RUN", "position_impact": 0,
                  "claim": identity(root, claim_path)})
    write_once(root / OUT / "G2_adjudication.json", stats)
    return stats
