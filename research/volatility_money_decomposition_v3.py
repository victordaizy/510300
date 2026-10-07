"""拆解上/下行波动、滚动窗口退出效应及货币背景；全部结果仅作研究观察。"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_volatility_money_decomposition_v3"
PARENT = ROOT / "reports/research/510300_macro_volatility_observation_v2_run1"
EPS = 1e-12
REGIMES = ["M1_OLD_M2_MMF2018", "M1_NEW2025"]
UP, DOWN, CENTER, TIE, FLAT = "总波动上升_上行项主导", "总波动上升_下行项主导", "总波动上升_均值项主导", "总波动上升_并列", "总波动未上升"
RELIEF, ROLLOFF, NO_RELIEF = "D20下降_近期5日同步缓和", "D20下降_近期5日反而增强", "D20未下降"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def jsave(name: str, obj) -> None:
    path = STUDY / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(STUDY / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def feature_frame(market: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame({"observation_date": market.date})
    r = (market.close + market.dividend) / market.close.shift() - 1
    mean = r.rolling(20).mean()
    down = r.clip(upper=0).pow(2)
    up = r.clip(lower=0).pow(2)
    data["v_up2"] = up.rolling(20).mean() * 252
    data["v_down2"] = down.rolling(20).mean() * 252
    data["v_mean_correction"] = -252 * mean.pow(2)
    data["v_total2"] = r.rolling(20).var(ddof=1) * 252
    data["v_upside20"] = np.sqrt(data.v_up2)
    data["v_downside20"] = np.sqrt(data.v_down2)
    data["v_rv20"] = np.sqrt(data.v_total2)
    data["v_down_fraction"] = data.v_down2 / (data.v_up2 + data.v_down2)
    for component in ["up2", "down2", "mean_correction"]:
        data["v_d5_" + component] = data["v_" + component].diff(5)
        data["v_variance_change_" + component] = data["v_d5_" + component] * 20 / 19
    data["v_variance_change_total"] = data.v_total2.diff(5)
    data["v_recent5_down_sum"] = down.rolling(5).sum()
    data["v_previous5_down_sum"] = down.rolling(5).sum().shift(5)
    data["v_exited5_down_sum"] = down.rolling(5).sum().shift(20)
    def volatility_label(row):
        if pd.isna(row.v_variance_change_total):
            return "历史不足"
        if row.v_variance_change_total <= EPS:
            return FLAT
        values = [row.v_variance_change_up2, row.v_variance_change_down2, row.v_variance_change_mean_correction]
        winners = [i for i, value in enumerate(values) if abs(value - max(values)) <= EPS]
        return [UP, DOWN, CENTER][winners[0]] if len(winners) == 1 else TIE
    def relief_label(row):
        if pd.isna(row.v_d5_down2):
            return "历史不足"
        if row.v_d5_down2 >= -EPS:
            return NO_RELIEF
        return ROLLOFF if row.v_recent5_down_sum - row.v_previous5_down_sum > EPS else RELIEF
    data["volatility_source_state"] = data.apply(volatility_label, axis=1)
    data["downside_window_state"] = data.apply(relief_label, axis=1)
    residual1 = data.v_total2 - 20 / 19 * (data.v_up2 + data.v_down2 + data.v_mean_correction)
    residual2 = data.v_d5_down2 - 252 / 20 * (data.v_recent5_down_sum - data.v_exited5_down_sum)
    assert residual1.abs().max() < EPS
    assert residual2.abs().max() < EPS
    jsave("evidence/identity_verification.json", {"variance_identity_max_abs": residual1.abs().max(), "rolling_identity_max_abs": residual2.abs().max(), "valid_variance_days": residual1.notna().sum(), "valid_rolling_days": residual2.notna().sum()})
    return data


def attach_future_risk(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    locations = {date: i for i, date in enumerate(market.date)}
    cache = {}
    for entry in ["E0", "E1"]:
        for horizon in [5, 20, 60]:
            prefix = f"{entry}_{horizon}"
            values = []
            for row in frame.to_dict("records"):
                start = row[f"{prefix}_entry_date"]
                end = row[f"{prefix}_exit_date"]
                if pd.isna(row[f"{prefix}_return"]):
                    values.append((np.nan, np.nan, np.nan))
                    continue
                key = (start, end)
                if key not in cache:
                    i, j = locations[start], locations[end]
                    assert j - i + 1 == horizon
                    part = market.iloc[i:j + 1]
                    dividends = part.dividend.to_numpy().copy()
                    dividends[0] = 0
                    wealth = part.close.to_numpy() + np.cumsum(dividends)
                    entry_open = float(part.open.iloc[0])
                    returns = wealth / np.r_[entry_open, wealth[:-1]] - 1
                    assert abs(wealth[-1] / entry_open - 1 - row[f"{prefix}_return"]) < 1e-12
                    assert abs(min(0.0, float(np.min(wealth / entry_open - 1))) - row[f"{prefix}_worst_path"]) < 1e-12
                    cache[key] = (float(np.std(returns, ddof=1) * np.sqrt(252)), float(np.sqrt(252 * np.mean(np.minimum(returns, 0) ** 2))), float(np.sqrt(252 * np.mean(np.maximum(returns, 0) ** 2))))
                values.append(cache[key])
            frame[[f"{prefix}_future_rv", f"{prefix}_future_downside", f"{prefix}_future_upside"]] = values
    return frame


def cycle_weights(frame: pd.DataFrame) -> np.ndarray:
    return 1.0 / frame.groupby("stat_month").stat_month.transform("count").to_numpy()


def weighted_rank(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # 同值使用累计权重中点秩，防止浮点尾数人为打破原数据并列。
    unique, inverse = np.unique(np.round(x, 12), return_inverse=True)
    sums = np.bincount(inverse, weights=weight, minlength=len(unique))
    return (np.cumsum(sums) - sums / 2)[inverse]


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    keep = w > 0
    x, y, w = x[keep], y[keep], w[keep]
    if len(x) < 3:
        return np.nan
    a, b = weighted_rank(x, w), weighted_rank(y, w)
    a = a - np.average(a, weights=w)
    b = b - np.average(b, weights=w)
    denom = math.sqrt(float(np.sum(w * a ** 2) * np.sum(w * b ** 2)))
    return float(np.sum(w * a * b) / denom) if denom > EPS else np.nan


def draw_cycles(frame: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    cycles = sorted(frame.stat_month.dropna().unique())
    n = len(cycles)
    rng = np.random.default_rng(20260929)
    block = min(6, n)
    starts = rng.integers(0, n - block + 1, size=(2000, math.ceil(n / block)))
    draws = (starts[:, :, None] + np.arange(block)).reshape(2000, -1)[:, :n]
    multiplicities = np.array([np.bincount(row, minlength=n) for row in draws])
    return cycles, multiplicities, draws


def interval(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {"lower_2_5": float(np.quantile(finite, 0.025)) if len(finite) else np.nan, "upper_97_5": float(np.quantile(finite, 0.975)) if len(finite) else np.nan, "valid_draws": len(finite), "missing_draws": len(values) - len(finite)}


def periods(frame: pd.DataFrame):
    yield "全部", frame
    for period, part in frame.groupby("analysis_period"):
        yield period, part


def distributions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, sample in frame.dropna(subset=["stat_month"]).groupby("training_regime"):
        for period, group in periods(sample):
            contexts = [("全部", group)] + list(group.groupby("macro_improvement"))
            for context, part in contexts:
                for category in ["volatility_source_state", "downside_window_state"]:
                    for label, selected in part.groupby(category):
                        for entry in ["E0", "E1"]:
                            for horizon in [5, 20, 60]:
                                prefix = f"{entry}_{horizon}"
                                valid = selected.dropna(subset=[f"{prefix}_return"])
                                if not len(valid):
                                    continue
                                weight = cycle_weights(valid)
                                row = {"regime": regime, "period": period, "macro_context": context, "category": category, "state": label, "entry": entry, "horizon": horizon, "weeks": len(valid), "cycles": valid.stat_month.nunique()}
                                for metric in ["return", "worst_path", "future_rv", "future_downside"]:
                                    x = valid[f"{prefix}_{metric}"].to_numpy()
                                    row[metric + "_cycle_weighted_mean"] = np.average(x, weights=weight)
                                    row[metric + "_min"] = x.min()
                                row["positive_return_fraction"] = np.average(valid[f"{prefix}_return"] > 0, weights=weight)
                                rows.append(row)
    return pd.DataFrame(rows)


def associations(frame: pd.DataFrame, draws: dict) -> pd.DataFrame:
    rows = []
    for regime, group in frame.dropna(subset=["stat_month"]).groupby("training_regime"):
        cycles, counts, _ = draws[regime]
        indices = {c: i for i, c in enumerate(cycles)}
        for period, sample in periods(group):
            for entry in ["E0", "E1"]:
                for feature in ["v_rv20", "v_upside20", "v_downside20"]:
                    for label in ["future_rv", "future_downside", "return", "worst_path"]:
                        target = f"{entry}_20_{label}"
                        valid = sample.dropna(subset=[feature, target])
                        w = cycle_weights(valid)
                        x, y = valid[feature].to_numpy(), valid[target].to_numpy()
                        row = {"regime": regime, "period": period, "entry": entry, "feature": feature, "outcome": label, "weeks": len(valid), "cycles": valid.stat_month.nunique(), "weighted_spearman": weighted_corr(x, y, w)}
                        if period == "全部" and entry == "E0":
                            ids = valid.stat_month.map(indices).to_numpy()
                            boot = np.array([weighted_corr(x, y, w * count[ids]) for count in counts])
                            row.update(interval(boot))
                        row["inference_status"] = "DESCRIPTIVE_BLOCK_INTERVAL" if row["cycles"] >= 24 else "SMALL_CYCLE_COUNT"
                        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(frame: pd.DataFrame, draws: dict) -> pd.DataFrame:
    pairs = [("上涨项主导减下跌项主导", "volatility_source_state", UP, DOWN), ("窗口缓和但近期增强减同步缓和", "downside_window_state", ROLLOFF, RELIEF)]
    rows = []
    for regime, group in frame.dropna(subset=["stat_month"]).groupby("training_regime"):
        cycles, counts, _ = draws[regime]
        for period, sample in periods(group):
            for label, column, positive, negative in pairs:
                for entry in ["E0", "E1"]:
                    for metric in ["return", "worst_path", "future_downside"]:
                        target = f"{entry}_20_{metric}"
                        vectors, sizes = [], []
                        for state in [positive, negative]:
                            valid = sample[sample[column] == state].dropna(subset=[target])
                            vector = valid.groupby("stat_month")[target].mean().reindex(cycles).to_numpy()
                            vectors.append(vector)
                            sizes.append((len(valid), valid.stat_month.nunique()))
                        row = {"regime": regime, "period": period, "comparison": label, "entry": entry, "outcome": metric, "positive_weeks": sizes[0][0], "positive_cycles": sizes[0][1], "negative_weeks": sizes[1][0], "negative_cycles": sizes[1][1], "positive_mean": np.nanmean(vectors[0]), "negative_mean": np.nanmean(vectors[1]), "difference": np.nanmean(vectors[0]) - np.nanmean(vectors[1])}
                        if period == "全部":
                            means = []
                            for vector in vectors:
                                finite = np.isfinite(vector)
                                denominator = counts @ finite.astype(float)
                                numerator = counts @ np.nan_to_num(vector)
                                means.append(np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0))
                            row.update(interval(means[0] - means[1]))
                        row["inference_status"] = "DESCRIPTIVE_ONLY" if min(sizes[0][1], sizes[1][1]) >= 24 else "SMALL_CYCLE_COUNT"
                        rows.append(row)
    return pd.DataFrame(rows)


def macro_associations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, group in frame.groupby("training_regime"):
        for period, sample in periods(group):
            for feature in ["spread_pp", "delta3_spread_pp"]:
                for entry in ["E0", "E1"]:
                    for label in ["return", "worst_path", "future_rv", "future_downside"]:
                        target = f"{entry}_20_{label}"
                        valid = sample.dropna(subset=[feature, target, "past_return20", "past_return60", "v_rv20", "v_down_fraction"])
                        if len(valid) < 3:
                            continue
                        x, y = valid[feature].to_numpy(), valid[target].to_numpy()
                        raw = weighted_corr(x, y, np.ones(len(valid)))
                        controls = np.c_[np.ones(len(valid)), valid[["past_return20", "past_return60"]], np.log(valid.v_rv20), valid.v_down_fraction, pd.get_dummies(valid.stat_month.str[-2:], drop_first=True, dtype=float)]
                        rank = np.linalg.matrix_rank(controls)
                        residual = np.nan
                        status = "NO_ESTIMATE_RESIDUAL_DF_LT12"
                        if len(valid) - rank >= 12:
                            xr = x - controls @ np.linalg.lstsq(controls, x, rcond=None)[0]
                            yr = y - controls @ np.linalg.lstsq(controls, y, rcond=None)[0]
                            residual = weighted_corr(xr, yr, np.ones(len(valid)))
                            status = "DESCRIPTIVE_LINEAR_RESIDUAL_ASSOCIATION"
                        rows.append({"regime": regime, "period": period, "entry": entry, "feature": feature, "outcome": label, "months": len(valid), "raw_spearman": raw, "residual_spearman": residual, "controls_rank": rank, "residual_df": len(valid) - rank, "status": status})
    return pd.DataFrame(rows)


def reuse_frozen_evidence() -> None:
    riskdir = ROOT / "reports/research/510300_monthly_downside_forecast_v1"
    macrodir = ROOT / "reports/research/510300_m1_m2_monthly_increment_v1"
    risk = pd.read_csv(riskdir / "paired_losses.csv")
    risk_metrics = []
    for comparator in ["PAST_MONTHLY_MEAN", "RECENT_DOWNSIDE20"]:
        for model in ["LOG_RISK_RIDGE", comparator]:
            calculated = np.log(risk[model]) + risk.observed_risk / risk[model]
            assert np.allclose(calculated, risk[model + "_loss"], atol=1e-10, rtol=0)
        improvement = risk[comparator + "_loss"] - risk.LOG_RISK_RIDGE_loss
        assert np.allclose(improvement, risk[comparator + "_improvement"], atol=1e-10, rtol=0)
        for period in ["early", "main"]:
            risk_metrics.append({"period": period, "comparator": comparator, "months": int((risk.period == period).sum()), "mean_loss_improvement": float(improvement[risk.period == period].mean())})
    files = [riskdir / name for name in ["prediction_result.json", "result.json", "period_metrics.csv", "paired_losses.csv", "account_metrics.csv"]] + [macrodir / name for name in ["prediction_result.json", "paired_M1_OLD_M2_MMF2018.csv"]]
    identities = []
    for source in files:
        destination = STUDY / "evidence" / f"{source.parent.name}_{source.name}"
        shutil.copy2(source, destination)
        identities.append({"original_path": str(source), "sha256": digest(source), "copy": str(destination.relative_to(STUDY))})
    macro = pd.read_csv(macrodir / "paired_M1_OLD_M2_MMF2018.csv")
    for model in ["baseline", "increment"]:
        calculated = (macro[model + "_prediction"] - macro.label) ** 2
        assert np.allclose(calculated, macro[model + "_loss"], atol=1e-12, rtol=0)
    assert np.allclose(macro.baseline_loss - macro.increment_loss, macro.loss_gain, atol=1e-12, rtol=0)
    assert (pd.to_datetime(macro.training_last_maturity) <= pd.to_datetime(macro.origin)).all()
    assert (pd.to_datetime(risk.training_last_maturity) <= pd.to_datetime(risk.origin)).all()
    jsave("evidence/reused_frozen_evidence.json", {"risk_losses_recomputed": True, "risk_period_metrics": risk_metrics, "macro_paired_months": len(macro), "macro_mse_baseline": macro.baseline_loss.mean(), "macro_mse_increment": macro.increment_loss.mean(), "macro_relative_mse_reduction": 1 - macro.increment_loss.mean() / macro.baseline_loss.mean(), "training_maturity_clock_check": "PASS", "new_model_fits": 0, "risk_forecast_is_not_return_prediction": True, "files": identities})


def run() -> None:
    for folder in ["results", "evidence", "inputs", "code", "figures"]:
        (STUDY / folder).mkdir(parents=True, exist_ok=True)
    frozen = json.loads((STUDY / "freeze_receipt.json").read_text(encoding="utf-8"))
    assert digest(ROOT / "config/510300_volatility_money_decomposition_v3.json") == frozen["protocol_sha256"]
    market = pd.read_csv(PARENT / "inputs/market_daily.csv")
    feature = feature_frame(market)
    csv(feature, "每日波动精确分解.csv")
    jsave("evidence/feature_receipt.json", {"saved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "future_new_risk_labels_joined": False, "sha256": digest(STUDY / "results/每日波动精确分解.csv")})
    frames = []
    for source in ["固定周度_完整观察.csv", "104个月_完整观察.csv"]:
        original = pd.read_csv(PARENT / "results" / source)
        copied = STUDY / "inputs" / source
        shutil.copy2(PARENT / "results" / source, copied)
        frame = original.merge(feature, on="observation_date", how="left", validate="many_to_one")
        for old, new in [("rv20", "v_rv20"), ("downside20", "v_downside20")]:
            assert np.allclose(frame[old], frame[new], atol=1e-12, rtol=0, equal_nan=True)
        frame = attach_future_risk(frame, market)
        frame["analysis_period"] = np.select([frame.stat_month < "2022-01", frame.stat_month < "2025-01"], ["统计月2018-2021", "统计月2022-2024"], default="统计月2025-2026")
        frame["macro_improvement"] = np.where(frame.delta3_spread_pp.isna(), "三月变化缺失", np.where(frame.delta3_spread_pp > EPS, "剪刀差改善", "剪刀差未改善"))
        csv(frame, "波动分解_" + source)
        frames.append(frame)
    weekly, monthly = frames
    draws = {regime: draw_cycles(weekly[weekly.training_regime == regime]) for regime in REGIMES}
    np.savez_compressed(STUDY / "results/固定时间块抽样.npz", **{regime + "_counts": values[1] for regime, values in draws.items()}, **{regime + "_indices": values[2] for regime, values in draws.items()})
    jsave("results/时间块周期顺序.json", {regime: values[0] for regime, values in draws.items()})
    csv(distributions(weekly), "全部波动来源与宏观背景分布.csv")
    print("已完成波动恒等式、未来风险路径、全部分组；开始固定区块区间。", flush=True)
    assoc = associations(weekly, draws)
    csv(assoc, "波动与后续风险收益关联.csv")
    comp = comparisons(weekly, draws)
    csv(comp, "主20日_两项固定比较.csv")
    macro = macro_associations(monthly)
    csv(macro, "月度剪刀差_价格波动季节控制关联.csv")
    reuse_frozen_evidence()
    shutil.copy2(Path(__file__), STUDY / "code" / Path(__file__).name)
    print(assoc[(assoc.period == "全部") & (assoc.entry == "E0") & (assoc.feature == "v_rv20")].to_string(index=False))
    print(comp[(comp.period == "全部") & (comp.entry == "E0")].to_string(index=False))
    print(weekly.groupby(["training_regime", "downside_window_state"]).size().to_string())
    print("第三轮计算完成；尚未将任何观察升级为独立预测证据。")


if __name__ == "__main__":
    run()
