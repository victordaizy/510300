"""现有分钟过程对日线的固定增量研究；历史条件模拟，不提交真实订单。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from research.intraday_overnight_increment_v1 import (
    Account, affordable_quantity, commission, execute_order, fill_price,
    normalize_dividends, normalize_prices, return_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_intraday_process_increment_v1.json"
MODELS = ("D", "A", "B", "C", "A_CONTROL")
ACCOUNTS = ("D", "A", "B", "C")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_times() -> list[str]:
    am = pd.date_range("2000-01-01 09:30", "2000-01-01 11:30", freq="min")
    pm = pd.date_range("2000-01-01 13:01", "2000-01-01 15:00", freq="min")
    return list(am.strftime("%H:%M")) + list(pm.strftime("%H:%M"))


def source_inventory(cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取真实数据；异常原样输出，未进入未来收益计算。"""
    paths = {k: ROOT / v for k, v in cfg["inputs"].items()}
    meta = json.loads(paths["minute_metadata"].read_text(encoding="utf-8"))
    price_receipt = json.loads(paths["price_receipt"].read_text(encoding="utf-8"))
    coverage = json.loads(paths["dividend_coverage"].read_text(encoding="utf-8"))
    require(digest(paths["minutes"]) == meta["files"]["parquet"]["sha256"], "分钟文件与来源回执身份不符")
    require(digest(paths["prices"]) == price_receipt["canonical_price"]["output_sha256"], "既有日线修正文件身份不符")
    require(digest(paths["dividends"]) == coverage["distribution_file_sha256"], "股息表身份不符")
    require(coverage["complete_history_confirmed"], "股息覆盖未确认")
    require(pd.Timestamp(coverage["coverage_end"]) >= pd.Timestamp(cfg["data_cutoff"]), "股息覆盖不足")
    m = pd.read_parquet(paths["minutes"]).sort_values("trade_time").reset_index(drop=True)
    require(set(m.ts_code) == {cfg["symbol"]}, "分钟证券不一致")
    m["date"] = pd.to_datetime(m.trade_time).dt.normalize()
    m["clock"] = m.trade_time.dt.strftime("%H:%M")
    finite = np.isfinite(m[["open", "high", "low", "close", "vol", "amount"]]).all(axis=1)
    m["price_valid"] = finite & m.low.gt(0) & m.high.ge(m[["open", "close", "low"]].max(axis=1)) & m.low.le(m[["open", "close", "high"]].min(axis=1))
    paired = m.vol.ge(0) & m.amount.ge(0) & m.vol.eq(0).eq(m.amount.eq(0))
    vwap = m.amount / m.vol.replace(0, np.nan)
    rounding = cfg["feature_contract"]["strict_amount_rounding_cny"] / m.vol.replace(0, np.nan)
    tol = cfg["account"]["tick"] * cfg["feature_contract"]["amount_vwap_tolerance_ticks"]
    m["vwap"] = vwap
    m["strict_amount_bad"] = ~paired | (m.vol.gt(0) & ((vwap < m.low - rounding) | (vwap > m.high + rounding)))
    m["amount_bad"] = ~paired | (m.vol.gt(0) & ((vwap < m.low - tol - 1e-12) | (vwap > m.high + tol + 1e-12)))
    p = normalize_prices(pd.read_parquet(paths["prices"]))
    p = p.loc[p.date.le(pd.Timestamp(cfg["data_cutoff"]))].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(paths["dividends"]))
    p["dividend"] = p.date.map(dividends.set_index("ex_date").cash_dividend_per_share).fillna(0.0)
    p["previous_close"] = p.close.shift(1)
    p["daily_total_return"] = (p.close + p.dividend) / p.previous_close - 1
    p["daily_intraday_return"] = np.log(p.close / p.open)
    p["daily_range"] = (p.high - p.low) / p.open
    p["daily_close_location"] = ((p.close - p.low) / (p.high - p.low).replace(0, np.nan)).where(p.high.gt(p.low), 0.5)
    p["log_amount"] = np.log(p.amount.where(p.amount.gt(0)))
    p["log_amount_relative20"] = np.log(p.amount / p.amount.shift().rolling(20, min_periods=20).median())
    p["rv20"] = p.daily_total_return.rolling(20, min_periods=20).std(ddof=1)
    p["log_rv20"] = np.log(p.rv20.where(p.rv20.gt(0)))
    p["momentum20"] = np.log1p(p.daily_total_return).rolling(20, min_periods=20).sum()
    calendar = pd.read_parquet(paths["calendar"])
    expected = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    expected = expected[(expected >= m.date.min()) & (expected <= m.date.max())]
    require(pd.DatetimeIndex(sorted(m.date.unique())).equals(expected), "分钟日期与独立交易日历不一致")
    aggregate = m.groupby("date").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("vol", "sum"), amount=("amount", "sum"))
    ref = p.set_index("date").reindex(aggregate.index)
    q = pd.DataFrame({"date": aggregate.index})
    groups = m.groupby("date", sort=True)
    q["rows"] = groups.size().to_numpy()
    q["complete_clock_grid"] = [g.clock.tolist() == expected_times() for _, g in groups]
    q["invalid_price_rows"] = groups.price_valid.apply(lambda s: int((~s).sum())).to_numpy()
    q["strict_amount_bad_rows"] = groups.strict_amount_bad.sum().to_numpy()
    q["amount_bad_rows"] = groups.amount_bad.sum().to_numpy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        q[col + "_aggregate_difference"] = (aggregate[col] - ref[col]).to_numpy()
    q["ohlc_daily_match"] = q[[c + "_aggregate_difference" for c in ["open", "high", "low", "close"]]].abs().max(axis=1).le(1e-10)
    q["daily_volume_amount_match"] = ((aggregate.volume - ref.volume).abs() / ref.volume < 1e-6).to_numpy() & ((aggregate.amount - ref.amount).abs() / ref.amount < 1e-6).to_numpy()
    q["price_day_valid"] = q.complete_clock_grid & q.invalid_price_rows.eq(0) & q.ohlc_daily_match & q.daily_volume_amount_match
    require(not m.trade_time.duplicated().any(), "分钟时间重复")
    q.to_csv(output / "02_逐日质量.csv", index=False, encoding="utf-8-sig")
    m.loc[m.strict_amount_bad | ~m.price_valid].to_parquet(output / "03_异常分钟原值.parquet", index=False)
    records = []
    for key, path in paths.items():
        row = {"用途": key, "路径": str(path), "SHA256": digest(path), "字节": path.stat().st_size}
        if path.suffix == ".parquet":
            pf = pq.ParquetFile(path)
            row.update(行数=pf.metadata.num_rows, 字段=", ".join(pf.schema_arrow.names))
        records.append(row)
    records[0].update(开始=str(m.date.min().date()), 结束=str(m.date.max().date()), 日期数=int(m.date.nunique()), 单位="价格元、数量份、金额元", 时间标签=meta["timestamp_meaning"], 使用权限="现有项目研究；第三方代理，未证明再分发或当前可免费重新下载", 本轮用途="盘后分钟过程代理，不认定历史送达或订单流")
    pd.DataFrame(records).to_csv(output / "01_实际数据清单.csv", index=False, encoding="utf-8-sig")
    quality = {"created_at": now(), "minute_rows": len(m), "minute_days": m.date.nunique(), "start": m.date.min(), "end": m.date.max(), "strict_vwap_bad_rows": int(m.strict_amount_bad.sum()), "strict_vwap_bad_days": int(m.loc[m.strict_amount_bad, "date"].nunique()), "strict_entire_day_valid_days": int(q.strict_amount_bad_rows.eq(0).sum()), "more_than_one_tick_bad_rows": int(m.amount_bad.sum()), "more_than_one_tick_bad_days": int(m.loc[m.amount_bad, "date"].nunique()), "price_invalid_rows": int((~m.price_valid).sum()), "price_qualified_days": int(q.price_day_valid.sum()), "zero_volume_rows": int(m.vol.eq(0).sum()), "timestamp_contract": meta["timestamp_meaning"], "historical_receive_proven": False, "permission": "EXISTING_PROJECT_RESEARCH_NO_REDISTRIBUTION_LICENSE_PROVEN", "one_tick_amount_precision_assumption": True, "all_anomalies_repaired": False, "daily_aggregation_proves_intraday_accuracy": False}
    write_json(output / "source_quality.json", quality)
    return m, p, dividends


def build_process(m: pd.DataFrame, p: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fc = cfg["feature_contract"]
    dates = pd.DatetimeIndex(sorted(m.date.unique()))
    q = m.groupby("date").price_valid.all().reindex(dates)
    continuous = m.loc[m.clock.ne("09:30")].copy()
    continuous["slot"] = continuous.groupby("date").cumcount() // 5
    b = continuous.groupby(["date", "slot"], sort=True).agg(open=("open", "first"), close=("close", "last"), high=("high", "max"), low=("low", "min"), amount=("amount", "sum"), volume=("vol", "sum"), amount_bad=("amount_bad", "any"), strict_amount_bad=("strict_amount_bad", "any"), end=("trade_time", "last")).reset_index()
    b["return5"] = np.log(b.close / b.open)
    b["regime"] = np.where(b.date.lt(pd.Timestamp(cfg["regime_change"])), "PRE_20260706", "POST_20260706")
    b["threshold"] = b.groupby(["regime", "slot"]).return5.transform(lambda s: s.shift().rolling(fc["prior_same_slot_days"], min_periods=fc["prior_same_slot_days"]).quantile(fc["shock_quantile"]))
    b["threshold"] = b.threshold.clip(upper=fc["shock_floor_log_return"])
    b["shock"] = b.return5.lt(b.threshold)
    events, daily_rows = [], []
    columns = sorted(set(sum(cfg["feature_groups"].values(), [])) - {"close_pressure_30m_60d"})
    for date, bars in b.groupby("date", sort=True):
        bars = bars.set_index("slot")
        minute_day = continuous.loc[continuous.date.eq(date)].reset_index(drop=True)
        warm = bars.threshold.notna().all()
        day_events = []
        lock = -1
        for slot in list(range(0, 18)) + list(range(24, 39)):
            row = bars.loc[slot]
            if slot < lock or not bool(row.shock):
                continue
            lock = slot + 1 + fc["observation_bars_5min"] + fc["quiet_bars_5min"]
            following = minute_day.iloc[(slot + 1) * 5:(slot + 7) * 5]
            require(len(following) == 30, "固定观察窗口不足30根")
            shock_size = float(row.open - row.close)
            require(shock_size > 0, "冲击价格幅度必须为正")
            repair = (following.close.to_numpy() - float(row.close)) / shock_size
            repaired = repair >= fc["repair_fraction"]
            first = np.flatnonzero(repaired)
            relapse = bool(len(first) and (following.close.to_numpy()[first[0] + 1:] < row.close).any())
            nxt = bars.loc[slot + 1]
            ratio = float(nxt.amount / row.amount) if row.amount > 0 else np.nan
            amount_valid = not bool(row.amount_bad or nxt.amount_bad) and row.amount > 0 and nxt.amount > 0
            similar = bool(amount_valid and fc["similar_amount_ratio"][0] <= ratio <= fc["similar_amount_ratio"][1])
            attenuation = (float(nxt.return5) - float(row.return5)) / abs(float(row.return5)) if similar else (0.0 if amount_valid else np.nan)
            event = {"date": date, "regime": row.regime, "shock_start_label": minute_day.iloc[slot * 5].trade_time, "shock_end_label": row.end, "observation_end_label": following.trade_time.iloc[-1], "known_by": date + pd.Timedelta(hours=15, minutes=5), "shock_return": float(row.return5), "threshold_past60": float(row.threshold), "shock_size": abs(float(row.return5)), "shock_open": float(row.open), "shock_end_price": float(row.close), "repair_5": float(repair[4]), "repair_30": float(repair[-1]), "repair_persistence": float(repaired.mean()), "repair_minutes": int(repaired.sum()), "first_repair_minute": int(first[0] + 1) if len(first) else np.nan, "relapse": relapse, "event_state": "REPAIRED_THEN_RELAPSED" if relapse else ("REPAIRED" if len(first) else "NEVER_REPAIRED"), "initial_amount": float(row.amount), "next_amount": float(nxt.amount), "amount_ratio": ratio, "initial_range": float(row.high - row.low), "next_range": float(nxt.high - nxt.low), "next_return": float(nxt.return5), "amount_valid": amount_valid, "strict_amount_valid": not bool(row.strict_amount_bad or nxt.strict_amount_bad), "similar_activity": similar, "impact_attenuation": attenuation}
            events.append(event)
            day_events.append(event)
        result = {"date": date, "regime": bars.regime.iloc[0], "event_count": len(day_events), "process_available": bool(warm and q.loc[date]), "B_amount_valid": all(e["amount_valid"] for e in day_events)}
        result.update({col: 0.0 if warm else np.nan for col in columns})
        if warm and day_events:
            result.update(shock_any=1.0, shock_size=np.mean([e["shock_size"] for e in day_events]), repair_5=np.mean([e["repair_5"] for e in day_events]), repair_30=np.mean([e["repair_30"] for e in day_events]), repair_persistence=np.mean([e["repair_persistence"] for e in day_events]), relapse_fraction=np.mean([e["relapse"] for e in day_events]), similar_activity_fraction=np.mean([e["similar_activity"] for e in day_events]), impact_attenuation=np.mean([e["impact_attenuation"] for e in day_events]))
        if not result["B_amount_valid"]:
            result["similar_activity_fraction"] = result["impact_attenuation"] = np.nan
        daily_rows.append(result)
    daily = pd.DataFrame(daily_rows)
    md = m.set_index(["date", "clock"])
    daily["late_return_30m"] = md.xs("15:00", level="clock").close.reindex(dates).to_numpy() / md.xs("14:30", level="clock").close.reindex(dates).to_numpy() - 1
    late = m.loc[m.clock.gt("14:30")].groupby("date")
    day_amount = m.groupby("date").amount.sum()
    daily["late_amount_share"] = (late.amount.sum() / day_amount).reindex(dates).to_numpy()
    daily["late_amount_valid"] = (~late.amount_bad.any()).reindex(dates).to_numpy()
    daily["late_amount_share_prior60"] = daily.groupby("regime").late_amount_share.transform(lambda s: s.shift().rolling(60, min_periods=60).median())
    prior_quality = daily.groupby("regime").late_amount_valid.transform(lambda s: s.astype(float).shift().rolling(60, min_periods=60).min())
    daily["C_amount_valid"] = daily.late_amount_valid & prior_quality.eq(1.0)
    daily["close_pressure_30m_60d"] = daily.late_return_30m * daily.late_amount_share / daily.late_amount_share_prior60
    raw_c = daily.close_pressure_30m_60d.copy()
    old = pd.read_parquet(ROOT / cfg["inputs"]["old_c_features"])
    old["date"] = pd.to_datetime(old.date)
    check = daily.loc[daily.regime.eq("PRE_20260706"), ["date", "close_pressure_30m_60d"]].merge(old[["date", "close_pressure_30m_60d"]], on="date", suffixes=("_new", "_old"))
    require(len(check) == int(daily.regime.eq("PRE_20260706").sum()), "旧C缓存日期覆盖不足")
    require(np.allclose(check.close_pressure_30m_60d_new, check.close_pressure_30m_60d_old, equal_nan=True, atol=1e-12), "复用C定义与旧保存值不一致")
    daily["C_original_value_before_quality_mask"] = raw_c
    daily.loc[~daily.C_amount_valid, "close_pressure_30m_60d"] = np.nan
    daily = daily.merge(p[["date", "open", "high", "low", "close", "volume", "amount"] + cfg["daily_features"]], on="date", validate="one_to_one")
    daily["late_location_change"] = ((daily.close - md.xs("14:30", level="clock").close.reindex(dates).to_numpy()) / (daily.high - daily.low).replace(0, np.nan)).fillna(0.0)
    all_features = cfg["daily_features"] + sorted(set(sum(cfg["feature_groups"].values(), [])))
    daily["common_valid"] = daily.process_available & np.isfinite(daily[all_features]).all(axis=1)
    daily["quality_state"] = np.select([~daily.process_available, ~daily.B_amount_valid, ~daily.C_amount_valid, daily.common_valid], ["NO_VIEW_PAST60_OR_PRICE", "NO_VIEW_B_AMOUNT", "NO_VIEW_C_AMOUNT_HISTORY", "AVAILABLE_CONDITIONAL_PROXY"], default="NO_VIEW_DAILY_INPUT")
    return daily, pd.DataFrame(events), b


def add_labels(process: pd.DataFrame, prices: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """只在冻结后读取未来；标签按登记权益处理，不用除息日简单相加代替资格。"""
    result = process.copy()
    lookup = {pd.Timestamp(d): i for i, d in enumerate(prices.date)}
    for horizon in (1, 2, 5):
        values, exits = [], []
        for row in result.itertuples():
            i = lookup[pd.Timestamp(row.date)]
            if i + horizon >= len(prices):
                values.append(np.nan)
                exits.append(pd.NaT)
                continue
            entry, end = prices.iloc[i + 1], prices.iloc[i + horizon]
            # 在计划退出收盘卖出，当天收盘不再持有，故登记日严格早于退出日。
            entitlement = dividends.loc[dividends.record_date.ge(entry.date) & dividends.record_date.lt(end.date), "cash_dividend_per_share"].sum()
            values.append((float(end.close) + float(entitlement)) / float(entry.open) - 1)
            exits.append(end.date)
        result[f"return_h{horizon}"] = values
        result[f"exit_h{horizon}"] = pd.to_datetime(exits)
    return result


def fit_ridge(train: pd.DataFrame, features: list[str], cfg: dict) -> dict:
    x = train[features].to_numpy(float)
    y = train.return_h2.to_numpy(float)
    center, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    z = np.clip((x - center) / scale, -cfg["model"]["standardized_clip"], cfg["model"]["standardized_clip"])
    # 裁剪后的设计重新居中，使未惩罚截距与训练均值一致。
    z_center, y_center = z.mean(axis=0), float(y.mean())
    zc = z - z_center
    coef = np.linalg.solve(zc.T @ zc + cfg["model"]["alpha"] * np.eye(len(features)), zc.T @ (y - y_center))
    return {"features": features, "center": center.tolist(), "scale": scale.tolist(), "coef": coef.tolist(), "intercept": float(y_center - z_center @ coef), "training_rows": len(train), "last_training_origin": train.date.max(), "last_training_label_exit": train.exit_h2.max()}


def predict(model: dict, row: pd.Series, cfg: dict) -> float:
    z = (row[model["features"]].to_numpy(float) - np.asarray(model["center"])) / np.asarray(model["scale"])
    z = np.clip(z, -cfg["model"]["standardized_clip"], cfg["model"]["standardized_clip"])
    return float(model["intercept"] + z @ np.asarray(model["coef"]))


def forecasts(data: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    models, last_fit = {}, {}
    fits, rows = [], []
    for i, row in data.iterrows():
        regime = row.regime
        result = {"date": row.date, "regime": regime, "common_valid": bool(row.common_valid), "quality_state": row.quality_state, "event_count": row.event_count, "forecast_state": "NO_VIEW_INPUT", "return_h1": row.return_h1, "return_h2": row.return_h2, "return_h5": row.return_h5, "exit_h2": row.exit_h2}
        for key in MODELS:
            result["prediction_" + key] = np.nan
        if row.common_valid:
            train = data.loc[data.common_valid & data.regime.eq(regime) & data.exit_h2.le(row.date) & data.return_h2.notna()]
            if len(train) >= cfg["model"]["minimum_training_days"]:
                if regime not in models or i - last_fit[regime] >= cfg["model"]["refit_interval_trading_days"]:
                    fitted = {}
                    for key in MODELS:
                        features = cfg["daily_features"] + ([] if key == "D" else cfg["feature_groups"][key])
                        model = fit_ridge(train, features, cfg)
                        require(pd.Timestamp(model["last_training_label_exit"]) <= row.date, "训练使用了尚未成熟标签")
                        fitted[key] = model
                        fits.append({"fit_date": row.date, "regime": regime, "model": key, **model})
                    models[regime], last_fit[regime] = fitted, i
                for key in MODELS:
                    result["prediction_" + key] = predict(models[regime][key], row, cfg)
                result["forecast_state"] = "AVAILABLE_HISTORICAL_PROXY"
                result["fit_date"] = fits[-1]["fit_date"]
                result["training_rows"] = models[regime]["D"]["training_rows"]
            else:
                result["forecast_state"] = "NO_VIEW_MATURE_TRAINING_HISTORY"
        rows.append(result)
    return pd.DataFrame(rows), fits


def bootstrap_indices(size: int, draws: int, block: int, seed: int) -> np.ndarray:
    require(size > 0, "无法对空样本构造区间")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, size, size=(draws, int(np.ceil(size / block))))
    return ((starts[..., None] + np.arange(block)) % size).reshape(draws, -1)[:, :size]


def interval(values: np.ndarray, alpha: float = 0.05) -> list[float]:
    return np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist()


def predictive_results(pred: pd.DataFrame, cfg: dict) -> dict:
    valid = pred.prediction_D.notna() & pred.return_h2.notna()
    require(int(valid.sum()) > 1, "成熟共同预测不足，不能启动账户比较")
    # 保留失效日间隔，20日块按交易日日历而非压缩后的合格观察序列抽样。
    positions = np.flatnonzero(valid.to_numpy())
    x = pred.iloc[positions[0]:positions[-1] + 1].reset_index(drop=True)
    inf = cfg["inference"]
    idx = bootstrap_indices(len(x), inf["draws"], inf["block_days"], inf["seed"])
    base_error = (x.return_h2 - x.prediction_D).to_numpy() ** 2
    result = {"days": int(valid.sum()), "calendar_span_days": len(x), "start": x.date.min(), "end": x.date.max(), "common_dates_all_models": True, "baseline_mse": np.nanmean(base_error), "comparisons": {}, "path_diagnostics": []}
    for key in ("A", "B", "C"):
        error = (x.return_h2 - x["prediction_" + key]).to_numpy() ** 2
        delta = base_error - error
        halves = [float(np.nanmean(part)) for part in np.array_split(delta, 2)]
        bounds = interval(np.nanmean(delta[idx], axis=1), inf["family_alpha"] / inf["family_comparisons"])
        result["comparisons"][key] = {"mse": np.nanmean(error), "mse_reduction": np.nanmean(delta), "relative_mse_reduction": np.nanmean(delta) / np.nanmean(base_error), "family_adjusted_interval": bounds, "half_mse_reductions": halves, "passes_fixed_increment_evidence": bool(valid.sum() >= inf["minimum_evaluation_days"] and bounds[0] > 0 and min(halves) > 0)}
    control_error = (x.return_h2 - x.prediction_A_CONTROL).to_numpy() ** 2
    full_error = (x.return_h2 - x.prediction_A).to_numpy() ** 2
    diff = control_error - full_error
    result["A_persistence_beyond_rebound"] = {"mse_reduction": np.nanmean(diff), "interval95": interval(np.nanmean(diff[idx], axis=1)), "all_days_are_sampling_units": True}
    # 各预测的排序在本行已经形成；之后路径只作固定诊断，不据此改交易。
    for key in ACCOUNTS:
        score = x["prediction_" + key]
        for horizon in (1, 2, 5):
            target = x[f"return_h{horizon}"]
            mask = target.notna() & score.notna()
            result["path_diagnostics"].append({"model": key, "horizon": horizon, "observations": int(mask.sum()), "correlation": score.loc[mask].corr(target.loc[mask]), "mean_return_when_prediction_positive": target.loc[score.gt(0) & mask].mean(), "executable_for_new_shares": horizon >= 2, "no_horizon_selection": True})
    return result


def constrained_execution(account: Account, request: int, price: float, previous_close: float,
                          dividend: float, volume: float, day_index: int, cost: dict, cfg: dict) -> dict:
    """请求先存在，容量和现金只约束执行；始终保留原请求。"""
    ac = cfg["account"]
    if not np.isfinite(volume) or not np.isfinite(price) or price <= 0:
        reason = "UNFILLED_EXECUTION_DATA_MISSING"
    elif volume <= 0:
        reason = "UNFILLED_NO_VOLUME"
    else:
        reason = ""
    if reason:
        return {"requested_quantity": int(request), "filled_quantity": 0, "open_price": price, "fill_price": None, "commission": 0.0, "slippage_cost": 0.0, "notional": 0.0, "status": reason, "capacity_limited": False, "cash_after": account.cash, "shares_after": account.shares}
    capacity = int(np.floor(volume * ac["auction_volume_participation_cap"] / ac["lot"])) * ac["lot"]
    bounded = int(np.sign(request)) * min(abs(request), capacity)
    if bounded == 0:
        return {"requested_quantity": int(request), "filled_quantity": 0, "open_price": price, "fill_price": None, "commission": 0.0, "slippage_cost": 0.0, "notional": 0.0, "status": "UNFILLED_CAPACITY", "capacity_limited": True, "cash_after": account.cash, "shares_after": account.shares}
    result = execute_order(account, bounded, price, previous_close, dividend, day_index, cost, ac)
    result["requested_quantity"] = int(request)
    result["capacity_limited"] = abs(bounded) < abs(request)
    if result["filled_quantity"] and abs(result["filled_quantity"]) < abs(request):
        result["status"] = "PARTIALLY_FILLED_CONSTRAINT"
    return result


def simulate_account(prices: pd.DataFrame, dividends: pd.DataFrame, minutes: pd.DataFrame,
                     pred: pd.DataFrame, capital: int, cost_name: str, model: str,
                     cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ac, cost = cfg["account"], cfg["costs"][cost_name]
    account = Account(float(capital))
    prediction_map = pred.set_index("date")["prediction_" + model].to_dict()
    first_origin = pred.loc[pred.prediction_D.notna(), "date"].min()
    terminal = minutes.date.max()
    lookup = {pd.Timestamp(date): i for i, date in enumerate(prices.date)}
    anchor, last = lookup[pd.Timestamp(first_origin)], lookup[pd.Timestamp(terminal)]
    execution_rows = minutes.loc[minutes.clock.isin(["09:30", "15:00"])].set_index(["date", "clock"])
    events = dividends.to_dict("records")
    ledger, orders, decisions, cycles = [], [], [], []
    previous_nav = float(capital)
    previous_mark = float(prices.iloc[anchor].close)
    cycle, pending = None, None

    def make_decision(i: int) -> dict:
        row = prices.iloc[i]
        prediction = prediction_map.get(pd.Timestamp(row.date), np.nan)
        decision = {"date": row.date, "model": model, "prediction": prediction, "cash": account.cash, "shares": account.shares, "requested_quantity": 0, "estimated_cost_return": np.nan, "state": "NO_VIEW_PREDICTION"}
        if i > last - 2:
            decision["state"] = "TERMINAL_NO_NEW_ENTRY"
        elif account.shares:
            decision["state"] = "HOLD_TO_FIXED_EXIT"
        elif np.isfinite(prediction):
            px = fill_price(float(row.close), 1, cost, ac["tick"])
            qty = affordable_quantity(account.cash, px, cost, ac["lot"])
            estimated = 2 * commission(qty, float(row.close), cost) / (qty * float(row.close)) + 2 * cost["slippage"] + 2 * ac["tick"] / float(row.close) if qty else np.inf
            decision.update(estimated_cost_return=estimated, state="NO_ENTRY_EDGE_BELOW_COST")
            if prediction > estimated:
                decision.update(requested_quantity=qty, state="REQUEST_NEXT_OPEN")
        decisions.append(decision)
        return decision if decision["requested_quantity"] else None

    pending = make_decision(anchor)
    for i in range(anchor + 1, last + 1):
        market = prices.iloc[i]
        date = pd.Timestamp(market.date)
        old_shares = account.shares
        recognized, paid, fees, slips, turnover = 0.0, 0.0, 0.0, 0.0, 0.0
        for key, event in enumerate(events):
            if event["ex_date"] == date:
                amount = account.entitlements.get(key, 0) * event["cash_dividend_per_share"]
                if amount:
                    account.receivables[key] = amount
                    recognized += amount
                    if cycle is not None:
                        cycle["dividend"] += amount
            if event["payment_date"] < date and key in account.receivables:
                amount = account.receivables.pop(key)
                account.cash += amount
                paid += amount

        def volume_at(clock: str) -> float:
            if (date, clock) not in execution_rows.index:
                return np.nan
            row = execution_rows.loc[(date, clock)]
            return float(row.vol) if bool(row.price_valid) and not bool(row.amount_bad) else np.nan

        if pending is not None:
            execution = constrained_execution(account, int(pending["requested_quantity"]), float(market.open), float(market.previous_close), float(market.dividend), volume_at("09:30"), i, cost, cfg)
            orders.append({"date": date, "clock": "OPEN_PROXY", "origin": pending["date"], "side": "BUY", "price_is_simulation_proxy": True, **execution})
            fees += execution["commission"]
            slips += execution["slippage_cost"]
            turnover += execution["notional"]
            quantity = execution["filled_quantity"]
            if quantity:
                require(cycle is None, "持仓中不能再次进入")
                cycle = {"entry_date": date, "entry_index": i, "planned_exit_date": prices.iloc[i + 1].date, "quantity": quantity, "entry_cost": execution["notional"] + execution["commission"], "entry_quote_notional": quantity * float(market.open), "sale_net_cash": 0.0, "sale_quote_notional": 0.0, "commission": execution["commission"], "slippage": execution["slippage_cost"], "dividend": 0.0}
        pending = None
        intraday_shares = account.shares
        if cycle is not None and i >= cycle["entry_index"] + 1:
            execution = constrained_execution(account, -account.shares, float(market.close), float(market.previous_close), float(market.dividend), volume_at("15:00"), i, cost, cfg)
            orders.append({"date": date, "clock": "CLOSE_PROXY", "origin": cycle["entry_date"], "side": "SELL", "price_is_simulation_proxy": True, **execution})
            fees += execution["commission"]
            slips += execution["slippage_cost"]
            turnover += execution["notional"]
            cycle["commission"] += execution["commission"]
            cycle["slippage"] += execution["slippage_cost"]
            cycle["sale_net_cash"] += execution["notional"] - execution["commission"]
            cycle["sale_quote_notional"] += -execution["filled_quantity"] * float(market.close)
            if account.shares == 0:
                cycle["exit_date"] = date
                cycle["holding_trading_days"] = i - cycle["entry_index"] + 1
                cycle["net_pnl"] = cycle["sale_net_cash"] + cycle["dividend"] - cycle["entry_cost"]
                cycle["gross_quote_pnl"] = cycle["sale_quote_notional"] + cycle["dividend"] - cycle["entry_quote_notional"]
                cycle["net_return"] = cycle["net_pnl"] / cycle["entry_cost"]
                require(abs(cycle["net_pnl"] - (cycle["gross_quote_pnl"] - cycle["commission"] - cycle["slippage"])) < 1e-6, "完成周期费用分解不守恒")
                cycles.append(dict(cycle))
                cycle = None
        for key, event in enumerate(events):
            if event["payment_date"] == date and key in account.receivables:
                amount = account.receivables.pop(key)
                account.cash += amount
                paid += amount
            if event["record_date"] == date:
                account.entitlements[key] = account.shares
        account.assert_valid()
        nav = account.value(float(market.close))
        price_pnl = old_shares * (float(market.open) - previous_mark) + intraday_shares * (float(market.close) - float(market.open))
        error = nav - previous_nav - (price_pnl + recognized - fees - slips)
        require(abs(error) < 1e-6, f"逐日账户恒等式失败：{date}，误差{error}")
        ledger.append({"date": date, "capital": capital, "cost": cost_name, "model": model, "cash": account.cash, "shares": account.shares, "receivable": account.receivable(), "close": float(market.close), "equity": nav, "net_return": nav / previous_nav - 1, "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid, "commission": fees, "slippage": slips, "traded_notional": turnover, "exposure": account.shares * float(market.close) / nav, "accounting_error": error, "prediction_state": "AVAILABLE" if np.isfinite(prediction_map.get(date, np.nan)) else "NO_VIEW", "terminal_unliquidated": bool(i == last and account.shares)})
        previous_nav, previous_mark = nav, float(market.close)
        if i < last:
            pending = make_decision(i)
    order_cols = ["date", "clock", "origin", "side", "requested_quantity", "filled_quantity", "open_price", "fill_price", "commission", "slippage_cost", "notional", "status", "capacity_limited", "price_is_simulation_proxy"]
    return pd.DataFrame(ledger), pd.DataFrame(orders).reindex(columns=order_cols), pd.DataFrame(decisions), pd.DataFrame(cycles)


def account_metrics(ledger: pd.DataFrame, orders: pd.DataFrame, cycles: pd.DataFrame, cfg: dict) -> dict:
    metrics = return_metrics(ledger.net_return.to_numpy(), cfg["account"]["annual_days"])
    cycle_returns = cycles.net_return.to_numpy() if len(cycles) else np.array([])
    winners, losers = cycle_returns[cycle_returns > 0], cycle_returns[cycle_returns < 0]
    n = len(cycle_returns)
    winp = len(winners) / n if n else np.nan
    lossp = len(losers) / n if n else np.nan
    gain = float(winners.mean()) if len(winners) else np.nan
    loss = float(-losers.mean()) if len(losers) else np.nan
    expected = float(cycle_returns.mean()) if n else np.nan
    identity = ((winp * (gain if len(winners) else 0)) - (lossp * (loss if len(losers) else 0))) if n else np.nan
    require(not n or abs(identity - expected) < 1e-12, "实际净胜率与盈亏比恒等式失败")
    worst = ledger.net_return.sort_values().head(max(1, int(np.ceil(len(ledger) * .05))))
    metrics.update(capital=int(ledger.capital.iloc[0]), cost=ledger.cost.iloc[0], model=ledger.model.iloc[0], start=ledger.date.min(), end=ledger.date.max(), account_days=len(ledger), completed_cycles=n, net_expectancy=expected, win_probability=winp, loss_probability=lossp, mean_gain=gain, mean_loss=loss, payoff_ratio=gain / loss if np.isfinite(gain) and np.isfinite(loss) and loss else np.nan, net_expectancy_identity=identity, worst_day=float(ledger.net_return.min()), daily_es5=float(worst.mean()), mean_exposure=float(ledger.exposure.mean()), cash_days=int(ledger.shares.eq(0).sum()), no_view_days=int(ledger.prediction_state.eq("NO_VIEW").sum()), commission_cny=float(ledger.commission.sum()), slippage_cny=float(ledger.slippage.sum()), order_requests=len(orders), unfilled_requests=int(orders.filled_quantity.eq(0).sum()), partial_requests=int((orders.filled_quantity.ne(0) & orders.filled_quantity.abs().lt(orders.requested_quantity.abs())).sum()), simulated_quantity_fill_ratio=float(orders.filled_quantity.abs().sum() / orders.requested_quantity.abs().sum()) if len(orders) else np.nan, ending_shares=int(ledger.shares.iloc[-1]), ending_equity=float(ledger.equity.iloc[-1]), max_accounting_error=float(ledger.accounting_error.abs().max()), completed_gross_quote_pnl=float(cycles.gross_quote_pnl.sum()) if n else 0.0, completed_net_pnl=float(cycles.net_pnl.sum()) if n else 0.0, actual_fills_verified=0)
    return metrics


def run_accounts(prices: pd.DataFrame, dividends: pd.DataFrame, minutes: pd.DataFrame,
                 pred: pd.DataFrame, cfg: dict, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    all_ledgers, all_orders, all_decisions, all_cycles, metrics = [], [], [], [], []
    for capital in cfg["account"]["initial_capitals"]:
        for cost in cfg["costs"]:
            for model in ACCOUNTS:
                ledger, orders, decisions, cycles = simulate_account(prices, dividends, minutes, pred, capital, cost, model, cfg)
                metrics.append(account_metrics(ledger, orders, cycles, cfg))
                all_ledgers.append(ledger)
                for frame, dest in [(orders, all_orders), (decisions, all_decisions), (cycles, all_cycles)]:
                    for col, value in [("capital", capital), ("cost", cost), ("model", model)]:
                        frame[col] = value
                    dest.append(frame)
    ledgers = pd.concat(all_ledgers, ignore_index=True)
    metric_df = pd.DataFrame(metrics)
    ledgers.to_parquet(output / "08_完整账户逐日账本.parquet", index=False)
    pd.concat(all_orders, ignore_index=True).to_csv(output / "09_全部模拟请求.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_decisions, ignore_index=True).to_parquet(output / "10_逐日决策.parquet", index=False)
    pd.concat(all_cycles, ignore_index=True).to_csv(output / "11_完成持有周期.csv", index=False, encoding="utf-8-sig")
    metric_df.to_csv(output / "12_全部账户指标.csv", index=False, encoding="utf-8-sig")
    comparisons = []
    for (capital, cost), part in ledgers.groupby(["capital", "cost"]):
        wide = part.pivot(index="date", columns="model", values="net_return")
        require(wide.notna().all().all(), "账户未保留共同完整日历")
        inf = cfg["inference"]
        idx = bootstrap_indices(len(wide), inf["draws"], inf["block_days"], inf["seed"])
        baseline = wide.D.to_numpy()
        for key in ("A", "B", "C"):
            values = wide[key].to_numpy()
            delta = values - baseline
            means = delta[idx].mean(axis=1) * cfg["account"]["annual_days"]
            bounds = interval(means, inf["family_alpha"] / inf["family_comparisons"])
            comparisons.append({"capital": capital, "cost": cost, "model": key, "paired_days": len(wide), "annualized_arithmetic_increment": float(delta.mean() * cfg["account"]["annual_days"]), "family_adjusted_lower": bounds[0], "family_adjusted_upper": bounds[1], "increment_lower_positive": bounds[0] > 0})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "13_账户相对日线增量.csv", index=False, encoding="utf-8-sig")
    annual = []
    for (capital, cost, model, year), part in ledgers.groupby(["capital", "cost", "model", ledgers.date.dt.year]):
        annual.append({"capital": capital, "cost": cost, "model": model, "year": year, "days": len(part), "cumulative_return": np.prod(1 + part.net_return) - 1, "cost_cny": (part.commission + part.slippage).sum(), "mean_exposure": part.exposure.mean()})
    pd.DataFrame(annual).to_csv(output / "14_逐年账户归因.csv", index=False, encoding="utf-8-sig")
    return metric_df, ledgers, {"comparisons": comparisons}


def chart(ledgers: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    names = {"D": "仅日线", "A": "日线＋修复过程", "B": "日线＋推进变化", "C": "日线＋旧尾盘代理"}
    for axis, capital in zip(axes, (200000, 20000)):
        for key in ACCOUNTS:
            part = ledgers.loc[ledgers.capital.eq(capital) & ledgers.cost.eq("BASE") & ledgers.model.eq(key)]
            axis.plot(part.date, part.equity / capital, label=names[key], linewidth=1.3)
        axis.set_title(f"{capital / 10000:g}万元账户 · 基础费用 · 全部现金日与持仓计价")
        axis.set_ylabel("净值")
        axis.grid(alpha=.2)
        axis.legend(ncol=2, fontsize=8)
    fig.suptitle("510300盘中过程增量：历史条件模拟，未验证真实成交", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "账户净值对比.png", dpi=150)
    plt.close(fig)


def render_report(cfg: dict, output: Path, process: pd.DataFrame, events: pd.DataFrame,
                  predictive: dict, metric_df: pd.DataFrame, comparison: dict) -> dict:
    quality = json.loads((output / "source_quality.json").read_text(encoding="utf-8"))
    old_c = json.loads((ROOT / cfg["inputs"]["old_c_result"]).read_text(encoding="utf-8"))
    economic = []
    for key in ("A", "B", "C"):
        row = metric_df.loc[metric_df.capital.eq(200000) & metric_df.cost.eq("BASE") & metric_df.model.eq(key)].iloc[0]
        comp = next(c for c in comparison["comparisons"] if c["capital"] == 200000 and c["cost"] == "BASE" and c["model"] == key)
        passed = bool(predictive["comparisons"][key]["passes_fixed_increment_evidence"] and comp["increment_lower_positive"] and pd.notna(row.net_expectancy) and row.net_expectancy > 0 and pd.notna(row.net_sharpe) and row.net_sharpe >= cfg["inference"]["sharpe_reference"] and row.ending_shares == 0)
        economic.append({"model": key, "fixed_historical_evidence_pass": passed, "independent_validation": "NOT_ESTABLISHED", "strategy_accepted_for_trading": False})
    verdict = "NO_CONFIRMED_INCREMENT_KEEP_FIXED_NEGATIVE_RESULTS" if not any(r["fixed_historical_evidence_pass"] for r in economic) else "HISTORICAL_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION"
    summary = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPLETED_FIXED_MINUTE_PROCESS_INCREMENT_STUDY", "verdict": verdict, "minute_rows": quality["minute_rows"], "minute_days": len(process), "event_rows": len(events), "event_state_counts": events.event_state.value_counts().to_dict(), "common_valid_days": int(process.common_valid.sum()), "quality_states": process.quality_state.value_counts().to_dict(), "forecast_days": predictive["days"], "account_scenarios": len(metric_df), "results": economic, "post_regime_days": int(process.regime.eq("POST_20260706").sum()), "post_regime_source_or_training_qualified": int(process.loc[process.regime.eq("POST_20260706"), "common_valid"].sum()), "old_C_verdict_preserved": old_c["status"], "fees_for_data": 0, "actual_orders": 0, "actual_fills_verified": 0, "new_market_downloads": 0, "new_goal_replaces_orderbook_gate_as_primary": True, "old_M1_M2_financial_result": "NOT_COMPUTED_RESERVE_ONLY", "positive_net_advantage_established": any(r["fixed_historical_evidence_pass"] for r in economic), "independent_validation": "NOT_ESTABLISHED_PREVIOUSLY_EXPOSED_HISTORY", "research_deliverables_completed": True}
    write_json(output / "summary.json", summary)
    def pct(x):
        return f"{100 * float(x):.3f}%" if pd.notna(x) else "未知"
    def num(x):
        return f"{float(x):.3f}" if pd.notna(x) else "未知"
    lines = ["# 510300盘中过程对日线的增量研究结果", "", "本轮按新目标完成现有分钟数据、A/B/C增量预测及16条费用后账户。结论：" + ("固定检验尚未建立可靠的跨日增量优势。" if verdict.startswith("NO_") else "出现有限历史候选，仍缺独立验证及真实执行证明。"), "", f"实际读取{quality['minute_rows']:,}根、{len(process):,}个交易日；登记{len(events):,}个固定冲击事件，全部未修复和再次下跌事件均保存。共同完整过程输入{int(process.common_valid.sum()):,}日，成熟滚动共同预测{predictive['days']:,}日。", "", "## 数据边界", "", f"主分钟文件OHLC日聚合全部吻合既有修正日线；严格金额VWAP越界{quality['strict_vwap_bad_rows']:,}根、{quality['strict_vwap_bad_days']:,}日。其中超过一个0.001元价位的{quality['more_than_one_tick_bad_rows']}根、{quality['more_than_one_tick_bad_days']}日按登记规则使相关量额窗口失效。严格全日无异常只有{quality['strict_entire_day_valid_days']}日，无法支持504日训练的独立严格金额验证。容许一档精度是主量价代理的假设，不能称原始异常已修复。", "", "第三方代理时间标签未承诺柱首/柱尾，没有逐日发布版本或真实送达。信号盘后、交易次日起，避免读取未结束柱，但不能据此认证历史实时可得。旧较长NBS分钟只获八个固定五分钟用途，未用于补长本轮；广度仅列入清单，没有填中性或加入模型。", "", f"2026-07-06新制度共有{summary['post_regime_days']}日，不足单独60日基线及504个成熟训练日；该段保留NO_VIEW和完整账户计价，不能作为新制度策略验证。", "", "## 三个固定增量比较", "", "三组使用同一日线基准、共同日期、同一T+1开盘至T+2收盘标签。正的MSE减少表示增量优于日线。置信区间按真实交易日20日分块，三组校正；失效日间隔保留。", "", "|组别|相对MSE减少|校正后MSE减少区间|前/后半段减少|固定增量证据|", "|---|---:|---|---|---|"]
    for key, item in predictive["comparisons"].items():
        lines.append(f"|{key}|{pct(item['relative_mse_reduction'])}|{item['family_adjusted_interval']}|{item['half_mse_reductions']}|{'通过有限历史条件' if item['passes_fixed_increment_evidence'] else '未通过'}|")
    a = predictive["A_persistence_beyond_rebound"]
    lines += ["", f"A持续性相对已知冲击幅度和5/30分钟反弹的额外MSE减少为{a['mse_reduction']:.8g}，95%区间{a['interval95']}。", "", f"旧C单变量研究裁决仍为{old_c['status']}；其AUC={old_c['evaluation']['auc']:.6f}，Brier skill={old_c['evaluation']['brier_skill_score']:.6f}。本轮未重跑或改写旧C，只做日线条件下的新用途归因；不把旧特征重新命名为发现。", "", "## 费用后完整账户", "", "下表为条件模拟，所有账户保留完整日历、现金、跳空、分红、部分/未成交和浮亏；成交价与容量均是代理，已验证真实成交数为0。", "", "|本金|费用|模型|年化收益|净夏普|最大回撤|完成周期|周期净期望|平均仓位|未成交/部分请求|", "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in metric_df.itertuples():
        lines.append(f"|{row.capital:,}|{row.cost}|{row.model}|{pct(row.annualized_return)}|{num(row.net_sharpe)}|{pct(row.max_drawdown)}|{row.completed_cycles}|{pct(row.net_expectancy)}|{pct(row.mean_exposure)}|{row.unfilled_requests}/{row.partial_requests}|")
    lines += ["", "|20万元基础增量|年化算术收益差|三组校正区间|", "|---|---:|---|"]
    for item in comparison["comparisons"]:
        if item["capital"] == 200000 and item["cost"] == "BASE":
            lines.append(f"|{item['model']}减日线|{pct(item['annualized_arithmetic_increment'])}|[{pct(item['family_adjusted_lower'])}, {pct(item['family_adjusted_upper'])}]|")
    lines += ["", "净期望直接来自同一完成周期的实际模拟买卖和分红净收益，等于盈利概率×平均净盈利减亏损概率×平均净亏损；不重复扣除已进入成交价的滑点。原执行路径的毛报价损益、佣金与滑点分别保存，供区分毛优势和成本。费用场景可能改变事前请求，不能将两场景差异全归因于同一交易的收费。", "", "## 收益持续时间与结论", "", "第1、2、5日固定路径见15_持续时间诊断.csv。第1日是买入当天收盘，新份额不能当日卖出，仅作已消失优势的诊断；未按路径结果重新选择持有期限。缺失造成少交易和低仓位造成小回撤均不作为净优势证据。", "", "保存正负结果后，停止本版参数探索。下一步应先判断原组别的毛预测增量、费用与容量/缺失各自影响；只有实质不同的信息用途及独立样本才值得另立实验，不能移动窗口、变号或延长持有期救回。历史被多次观察，任何局部通过都不建立独立金融验证或实盘授权。", "", "核心文件：01_实际数据清单.csv；02_逐日质量.csv；03_异常分钟原值.parquet；04_逐日盘中过程.csv/parquet；05_全部冲击事件.csv/parquet；06_滚动预测.csv；07_全部模型参数.json；08_完整账户逐日账本.parquet；09_全部模拟请求.csv；10_逐日决策.parquet；11_完成持有周期.csv；12_全部账户指标.csv；13_账户相对日线增量.csv；14_逐年账户归因.csv；15_持续时间诊断.csv。"]
    (output / "研究结论与下一步.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def freeze(cfg: dict, output: Path) -> None:
    paths = [CONFIG, Path(cfg["objective_file"]), ROOT / "research/intraday_process_increment_v1.py", ROOT / "scripts/run_510300_intraday_process_increment_v1.py", ROOT / "tests/test_intraday_process_increment_v1.py", ROOT / "docs/510300_INTRADAY_PROCESS_INCREMENT_V1.md", ROOT / "research/intraday_overnight_increment_v1.py"]
    paths += [ROOT / value for value in cfg["inputs"].values()]
    paths += [output / name for name in ["04_逐日盘中过程.parquet", "normalized_prices.parquet", "normalized_dividends.parquet"]]
    write_json(output / "freeze.json", {"frozen_at": now(), "future_labels_created": False, "new_fits_run": False, "new_accounts_run": False, "files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)} for path in paths], "no_same_sample_retuning": True})


def verify_saved(output: Path, cfg: dict) -> dict:
    """从保存账本重新计算金融数值；与真实行情/成交验证区分。"""
    ledger = pd.read_parquet(output / "08_完整账户逐日账本.parquet")
    metrics = pd.read_csv(output / "12_全部账户指标.csv")
    max_error = 0.0
    for key, part in ledger.groupby(["capital", "cost", "model"]):
        row = metrics.loc[metrics.capital.eq(key[0]) & metrics.cost.eq(key[1]) & metrics.model.eq(key[2])].iloc[0]
        equity = part.cash.to_numpy() + part.shares.to_numpy() * part.close.to_numpy() + part.receivable.to_numpy()
        require(np.allclose(equity, part.equity, atol=1e-7, rtol=0), "保存现金/持仓/应收恒等式失败")
        returns = equity / np.r_[key[0], equity[:-1]] - 1
        require(np.allclose(returns, part.net_return, atol=1e-12, rtol=0), "保存净值收益复算失败")
        expected = return_metrics(returns, cfg["account"]["annual_days"])
        for field in ["annualized_return", "max_drawdown", "net_sharpe"]:
            if expected[field] is None and pd.isna(row[field]):
                continue
            error = abs(float(expected[field]) - float(row[field]))
            max_error = max(max_error, error)
            require(error < 1e-10, f"保存指标复算失败：{key} {field}")
    orders = pd.read_csv(output / "09_全部模拟请求.csv")
    cycles = pd.read_csv(output / "11_完成持有周期.csv")
    if len(cycles):
        require((pd.to_datetime(cycles.exit_date) > pd.to_datetime(cycles.entry_date)).all(), "保存周期违反T+1")
        require(np.allclose(cycles.gross_quote_pnl - cycles.commission - cycles.slippage, cycles.net_pnl, atol=1e-6), "保存周期费用分解失败")
    require((orders.filled_quantity.abs() <= orders.requested_quantity.abs()).all(), "成交超过原请求")
    require((orders.filled_quantity % cfg["account"]["lot"] == 0).all(), "成交整手失败")
    report = {"verified_at": now(), "status": "PASS_SAVED_ACCOUNT_METRICS_RECOMPUTATION", "account_scenarios": len(metrics), "ledger_rows": len(ledger), "requests": len(orders), "completed_cycles": len(cycles), "maximum_metric_error": max_error, "maximum_accounting_error": float(ledger.accounting_error.abs().max()), "verified_actual_fills": 0, "independent_financial_validation": False}
    write_json(output / "saved_output_verification.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="510300盘中过程固定研究")
    parser.add_argument("--stage", choices=["prepare", "run", "verify"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["output"]
    if args.stage == "prepare":
        output.mkdir(parents=True, exist_ok=False)
        m, prices, dividends = source_inventory(cfg, output)
        process, events, bars = build_process(m, prices, cfg)
        process.to_parquet(output / "04_逐日盘中过程.parquet", index=False)
        process.to_csv(output / "04_逐日盘中过程.csv", index=False, encoding="utf-8-sig")
        events.to_parquet(output / "05_全部冲击事件.parquet", index=False)
        events.to_csv(output / "05_全部冲击事件.csv", index=False, encoding="utf-8-sig")
        bars.to_parquet(output / "fixed_5min_process_windows.parquet", index=False)
        prices.to_parquet(output / "normalized_prices.parquet", index=False)
        dividends.to_parquet(output / "normalized_dividends.parquet", index=False)
        freeze(cfg, output)
        print(json.dumps({"阶段": "过程表与代码冻结完成，未计算未来标签", "分钟日": len(process), "共同有效日": int(process.common_valid.sum()), "事件": len(events)}, ensure_ascii=False))
        return 0
    if args.stage == "verify":
        print(json.dumps(verify_saved(output, cfg), ensure_ascii=False))
        return 0
    require(not (output / "run_claim.json").exists(), "本版已经认领执行，禁止覆盖或重复调参运行")
    frozen = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        require(digest(Path(item["path"])) == item["sha256"], "冻结文件已变化：" + item["path"])
    write_json(output / "run_claim.json", {"claimed_at": now(), "status": "RUNNING_FIXED_EXPERIMENT"})
    process = pd.read_parquet(output / "04_逐日盘中过程.parquet")
    prices = pd.read_parquet(output / "normalized_prices.parquet")
    dividends = pd.read_parquet(output / "normalized_dividends.parquet")
    m = pd.read_parquet(ROOT / cfg["inputs"]["minutes"])
    m["date"] = m.trade_time.dt.normalize()
    m["clock"] = m.trade_time.dt.strftime("%H:%M")
    m["price_valid"] = np.isfinite(m[["open", "high", "low", "close"]]).all(axis=1) & m.low.gt(0)
    vwap = m.amount / m.vol.replace(0, np.nan)
    m["amount_bad"] = m.vol.eq(0).ne(m.amount.eq(0)) | m.vol.lt(0) | m.amount.lt(0) | (m.vol.gt(0) & ((vwap < m.low - .001 - 1e-12) | (vwap > m.high + .001 + 1e-12)))
    labeled = add_labels(process, prices, dividends)
    labeled.to_parquet(output / "process_with_fixed_labels.parquet", index=False)
    pred, fits = forecasts(labeled, cfg)
    pred.to_csv(output / "06_滚动预测.csv", index=False, encoding="utf-8-sig")
    pred.to_parquet(output / "06_滚动预测.parquet", index=False)
    write_json(output / "07_全部模型参数.json", fits)
    predictive = predictive_results(pred, cfg)
    write_json(output / "predictive_comparison.json", predictive)
    pd.DataFrame(predictive["path_diagnostics"]).to_csv(output / "15_持续时间诊断.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"阶段": "滚动预测完成", "共同预测日": predictive["days"], "模型拟合数": len(fits)}, ensure_ascii=False), flush=True)
    metric_df, ledgers, comparison = run_accounts(prices, dividends, m, pred, cfg, output)
    events = pd.read_parquet(output / "05_全部冲击事件.parquet")
    summary = render_report(cfg, output, process, events, predictive, metric_df, comparison)
    chart(ledgers, output)
    verification = verify_saved(output, cfg)
    write_json(output / "run_claim.json", {"claimed_at": json.loads((output / "run_claim.json").read_text(encoding="utf-8"))["claimed_at"], "completed_at": now(), "status": "COMPLETED_FIXED_EXPERIMENT", "verification": verification["status"]})
    files = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": digest(path)} for path in output.iterdir() if path.is_file() and path.name != "file_index.json"]
    total = sum(item["bytes"] for item in files)
    require(total <= cfg["maximum_output_bytes"], "输出超过冻结存储预算")
    write_json(output / "file_index.json", {"files": files, "total_bytes": total, "index_excludes_itself": True})
    print(json.dumps(clean({"状态": summary["status"], "结论": summary["verdict"], "账户数": len(metric_df), "保存复算": verification["status"], "新增字节": total}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
