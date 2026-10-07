"""冻结并检验增长状态对二十日收益及下行风险的独立增量。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_growth_state_increment_20d_v1"
REFRAME = ROOT / "reports/research/510300_macro_dynamic_reframe_v1"
STUDY = "510300_GROWTH_STATE_INCREMENT_20D_V1"
TZ = "Asia/Shanghai"
FEATURES = {"P0": ["price_m20", "price_log_sigma20"],
            "P1": ["price_m20", "price_log_sigma20", "growth_level", "growth_change"],
            "R0": ["price_m20", "price_log_down20"],
            "R1": ["price_m20", "price_log_down20", "growth_level", "growth_change"]}


def now() -> str:
    return datetime.now(ZoneInfo(TZ)).isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def freeze() -> None:
    if OUT.exists():
        raise FileExistsError("新实验目录已存在，不能覆盖冻结内容")
    for name in ("inputs", "code", "results", "figures", "evidence"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    for name in ("market.parquet", "pmi_new_orders.parquet"):
        shutil.copy2(REFRAME / "inputs" / name, OUT / "inputs" / name)
    shutil.copy2(ROOT / "data/reference/a_share_hs_trading_calendar_2010_2026_v1.parquet", OUT / "inputs/calendar.parquet")
    shutil.copy2(ROOT / "reports/data_quality/510300_macro_stress_inputs_2015_v2.json", OUT / "evidence/inherited_macro_quality.json")
    for name in ("growth_state_increment_20d_v1.py", "verify_growth_state_increment_20d_v1.py"):
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    cfg = {
        "study_id": STUDY, "frozen_at": now(), "hypothesis": "已公布新订单的水平和月度改善，分别是否在价格信息之外改善未来20日收益均值或下行波动预测。不是PMI预期差，也不代表全部宏观政策。",
        "parent_scope": "510300_MACRO_DYNAMIC_REFRAME_V1", "evidence_class": "FROZEN_NEW_COMPARISON_ON_PREVIOUSLY_OBSERVED_HISTORY",
        "prior_selection": "PMI历史已用于旧宏观压力门；价格、风险模型及20日持有期也有大量既往研究。本轮不是独立未见样本，不改变旧失败或NBS固定五分钟家族裁决。",
        "data_end": "2026-09-11", "last_macro_month": "2026-07", "last_decision_date": "2026-07-31",
        "horizon": 20, "lookback": 20, "minimum_train_months": 36, "minimum_evaluation_months": 24,
        "monthly_origin": "每份当次新订单完整可用后的第一个交易日收盘；从下一交易日开盘到第20个交易日收盘。",
        "training": "仅以月度公布后观察时点为训练原点，扩展训练；标签终点严格早于当前收盘。同一训练集合只拟合一次，按周及新公告时点更新预测输入。",
        "decision_grid": "官方交易日历每周最后交易日收盘，加新订单完整可用后的第一个收盘；同日去重。两类模型使用共同日历。",
        "primary_evaluation": "每月新订单公布后原点各一次；周度预测只作路径及逐期诊断，不扩大宏观有效样本量。",
        "features": FEATURES, "growth_features": "当次新订单减50；本月减上月。首次月份缺差值时不填零。",
        "price_features": "过去20个含分红日收益的复合回报；样本标准差对数；过去20日负收益平方均值的对数。均截至判断收盘。",
        "ridge_penalty": 1.0, "ridge_definition": "训练内总体标准差标准化；平均平方误差加1倍斜率平方和，截距不惩罚；不搜索系数惩罚、窗口或方向。",
        "return_target": "固定一份510300从次日开盘持有20日，获得入场之后除息产生的现金权利，不再投资；退出价格加累计分红除以入场价减1。",
        "risk_target": "同一持有区间每日含累计分红权益的财富变化，首日以开盘为起点；20个单日负收益平方的均值。无入场前跳空。",
        "risk_floor": 1e-8, "risk_model": "以log(max(目标下行方差,1e-8))作训练目标；预测指数乘训练残差指数均值作算术均值修正，保证正值。",
        "risk_loss": "QLIKE = log(预测下行方差) + 实际下行方差/预测下行方差；差值才有比较意义。另报告平方误差、实际/预测均值及低估频率。",
        "return_baselines": ["P0价格模型", "当时成熟月度标签均值", "恒定零收益"],
        "risk_baselines": ["R0价格风险模型", "当时成熟月度下行方差均值", "过去20日下行方差"],
        "gates": "收益、风险分别判定；新增模型相对各自三基准的损失改善均值和95%单侧下界均>0，相对价格模型前后两半改善均>0。至少24个月评价。任一通道失败不自动否决另一个。",
        "bootstrap": {"draws": 10000, "month_block": 6, "seed": 20260922, "lower_quantile": .05,
                      "method": "按连续月度评价原点，不循环移动块抽样；所有比较共用索引。两个独立主目标各用5%单侧门；不声称修正了整个仓库上游试验。"},
        "refresh_diagnostic": "仅在非周度复核日的新订单原点，固定同一组当时已成熟训练系数和当前价格输入，对比最新增长值与上一个周度复核时已知增长值；两者预测同一未来20日。诊断只衡量输入刷新，不等同真实连续账户。",
        "decision_diagnostic": "P0/P1预测是否超过按观察收盘、整手及最低佣金估算的满预算压力往返成本；R0/R1的10%风险预算参考仓位差。只是规则差异，不是实际成交或已实现账户收益。",
        "account_gate": "仅对通过自己预测门的模块开展独立账户阶段；收益失败不能用风险得分包装成功。20万元主账户、2万元成本对照。若有通道通过，下一账户执行合同须固定现金、分红到账、T+1、下一开盘及逐笔费用。",
        "account_mapping_if_admitted": "收益模块采用预测超过压力往返成本才持有，仓位上限为10%目标波动率/当时20日年化波动且不超过1；风险模块独立采用min(1,10%/sqrt(2*242*预测下行方差))，因子2是固定对称参照，不是假定分布已证实对称。共同周期及公告更新分别比较。",
        "cost": {"commission": .0002, "minimum": 5, "base_slippage": .0005, "stress_slippage": .001, "lot": 100, "annual_days": 242, "cash_rate": 0, "risk_free_rate": 0},
        "capital": [200000, 20000], "full_account_targets": {"net_sharpe": 1.2, "cagr": .10},
        "stop": "一轮固定历史检验，失败不改窗口、差分、方向、起点、惩罚或判据；不得从两个目标挑一个新名称救另一项失败。各宏观政策通道继续按自己的独立假设推进。",
        "new_20d_labels_before_freeze": 0, "independent_forward_events": 0, "new_accounts": 0}
    save(OUT / "protocol.json", cfg)
    frozen = [OUT / "protocol.json", *sorted((OUT / "inputs").glob("*")), *sorted((OUT / "code").glob("*.py"))]
    save(OUT / "freeze_receipt.json", {"frozen_at": now(), "new_target_returns_read": False,
        "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in frozen]})
    print("二十日增长状态协议、三份直接输入及两份代码已冻结；尚未读取新标签。")


def verify_freeze(study: Path) -> dict:
    receipt = json.loads((study / "freeze_receipt.json").read_text(encoding="utf-8"))
    for item in receipt["files"]:
        path = study / item["path"]
        if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("冻结身份不一致：" + item["path"])
    correction = study / "calendar_correction_receipt.json"
    if correction.exists():
        for item in json.loads(correction.read_text(encoding="utf-8"))["additional_frozen_files"]:
            path = study / item["path"]
            if sha(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("日历衔接修正身份不一致：" + item["path"])
    return json.loads((study / "protocol.json").read_text(encoding="utf-8"))


def label(prices: pd.DataFrame, origin: int, horizon: int) -> tuple[dict, list[float]]:
    entry, end = origin + 1, origin + horizon
    if end >= len(prices):
        raise ValueError("二十日标签未成熟，不能截短")
    holding = prices.iloc[entry:end + 1]
    cash = holding.dividend.to_numpy(float).copy()
    cash[0] = 0.0
    cumulative = np.cumsum(cash)
    wealth = holding.close.to_numpy(float) + cumulative
    entry_price = float(holding.open.iloc[0])
    returns = wealth / np.r_[entry_price, wealth[:-1]] - 1
    return {"entry_date": str(holding.date.iloc[0].date()), "exit_date": str(holding.date.iloc[-1].date()),
        "entry_at": (holding.date.iloc[0] + pd.Timedelta(hours=9, minutes=30)).isoformat(),
        "exit_at": (holding.date.iloc[-1] + pd.Timedelta(hours=15)).isoformat(),
        "entry_open": entry_price, "exit_close": float(holding.close.iloc[-1]), "earned_dividend": float(cumulative[-1]),
        "return_20d": float(wealth[-1] / entry_price - 1),
        "downside_variance_20d": float(np.square(np.minimum(returns, 0)).mean())}, returns.tolist()


def panel_and_origins(study: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(study / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date)
    pmi = pd.read_parquet(study / "inputs/pmi_new_orders.parquet").sort_values("reference_period").reset_index(drop=True)
    periods = pd.period_range(pmi.reference_period.iloc[0], pmi.reference_period.iloc[-1], freq="M").astype(str).tolist()
    if periods != pmi.reference_period.tolist() or len(pmi) != 139 or market.date.duplicated().any():
        raise ValueError("增长月份不连续或行情日期重复")
    pmi["growth_level"] = pmi.first_release_value - 50
    pmi["growth_change"] = pmi.first_release_value.diff()
    pmi["macro_at"] = pd.to_datetime(pmi.available_at, utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    market["decision_at"] = market.date + pd.Timedelta(hours=15)
    calculated_return = (market.close + market.dividend) / market.close.shift(1) - 1
    if not np.allclose(calculated_return.iloc[1:], market.total_simple.iloc[1:], atol=1e-12):
        raise ValueError("价格与分红日收益不一致")
    n = cfg["lookback"]
    market["price_m20"] = np.expm1(np.log1p(market.total_simple).rolling(n, min_periods=n).sum())
    market["sigma20"] = market.total_simple.rolling(n, min_periods=n).std(ddof=1)
    market["price_log_sigma20"] = np.log(market.sigma20)
    market["down20"] = market.total_simple.clip(upper=0).pow(2).rolling(n, min_periods=n).mean().clip(lower=cfg["risk_floor"])
    market["price_log_down20"] = np.log(market.down20)
    panel = pd.merge_asof(market, pmi[["macro_at", "reference_period", "growth_level", "growth_change"]], left_on="decision_at", right_on="macro_at", direction="backward")
    calendar_path = study / "inputs/calendar_extended.parquet"
    calendar = pd.read_parquet(calendar_path if calendar_path.exists() else study / "inputs/calendar.parquet")
    trade = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    expected = trade[(trade >= panel.date.min()) & (trade <= panel.date.max())]
    if list(expected) != panel.date.tolist():
        raise ValueError("行情与独立交易日历不同")
    calendar_frame = pd.DataFrame({"date": trade})
    weekly = set(calendar_frame.groupby(calendar_frame.date.dt.to_period("W-FRI")).date.max())
    release_positions = {}
    for row in pmi.itertuples():
        pos = int(np.searchsorted(panel.decision_at.to_numpy(), np.datetime64(row.macro_at), side="left"))
        if pos < len(panel):
            if pos in release_positions:
                raise ValueError("两个宏观发布映射到同一收盘，需先定义合并规则")
            release_positions[pos] = row.reference_period
    rows = []
    prior_weekly_pos = None
    all_features = sorted(set(sum(FEATURES.values(), [])))
    for pos, row in panel.iterrows():
        is_weekly = row.date in weekly
        is_release = pos in release_positions
        valid = row.date <= pd.Timestamp(cfg["last_decision_date"]) and row[all_features].notna().all()
        if valid and (is_weekly or is_release):
            outcome, _ = label(panel, pos, cfg["horizon"])
            item = {"origin_id": "GROWTH_" + row.date.strftime("%Y%m%d"), "origin_index": int(pos),
                "date": str(row.date.date()), "decision_at": row.decision_at.isoformat(),
                "macro_at": row.macro_at.isoformat(), "reference_period": row.reference_period,
                "is_release": is_release, "is_weekly": is_weekly, "previous_weekly_date": None,
                "stale_growth_level": np.nan, "stale_growth_change": np.nan, "close": float(row.close),
                "sigma20": float(row.sigma20), "down20": float(row.down20), **{c: float(row[c]) for c in all_features}, **outcome}
            if prior_weekly_pos is not None:
                previous = panel.iloc[prior_weekly_pos]
                item.update(previous_weekly_date=str(previous.date.date()), stale_growth_level=previous.growth_level, stale_growth_change=previous.growth_change)
            rows.append(item)
        if is_weekly:
            prior_weekly_pos = pos
    return panel, pd.DataFrame(rows)


def fit(train: pd.DataFrame, key: str, cfg: dict) -> dict:
    columns = FEATURES[key]
    x = train[columns].to_numpy(float)
    raw_target = train["return_20d" if key.startswith("P") else "downside_variance_20d"].to_numpy(float)
    y = raw_target if key.startswith("P") else np.log(np.maximum(raw_target, cfg["risk_floor"]))
    mean, scale = x.mean(0), x.std(0, ddof=0)
    scale = np.where(scale > 0, scale, 1.)
    z = (x - mean) / scale
    beta = np.linalg.solve(z.T @ z / len(train) + cfg["ridge_penalty"] * np.eye(len(columns)), z.T @ (y - y.mean()) / len(train))
    residual = y - (y.mean() + z @ beta)
    smear = 1. if key.startswith("P") else float(np.exp(residual).mean())
    return {"model": key, "columns": columns, "mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.tolist(),
        "intercept": float(y.mean()), "smearing": smear, "training_ids": train.origin_id.tolist(),
        "training_months": len(train), "latest_maturity": str(train.exit_at.max()), "first_used_at": None}


def predict(model: dict, row: pd.Series | dict) -> float:
    x = np.asarray([row[c] for c in model["columns"]], dtype=float)
    value = float(model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["beta"])
    if model["model"].startswith("R"):
        value = float(np.exp(value) * model["smearing"])
    if not np.isfinite(value):
        raise ValueError("预测产生非有限值，不能裁剪后继续")
    return value


def cost_threshold(close: float, capital: int, cfg: dict) -> float:
    c = cfg["cost"]
    buy_price, sell_price = close * (1 + c["stress_slippage"]), close * (1 - c["stress_slippage"])
    quantity = int(capital // (buy_price * c["lot"])) * c["lot"]
    while quantity > 0 and quantity * buy_price + max(c["minimum"], c["commission"] * quantity * buy_price) > capital:
        quantity -= c["lot"]
    if quantity <= 0:
        raise ValueError("本金不足整手")
    buy_fee = max(c["minimum"], c["commission"] * quantity * buy_price)
    sell_fee = max(c["minimum"], c["commission"] * quantity * sell_price)
    return float((quantity * (buy_price - sell_price) + buy_fee + sell_fee) / (quantity * close))


def walk_forward(origins: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    monthly = origins[origins.is_release].copy()
    all_models, cache, records = [], {}, []
    for _, row in origins.iterrows():
        train = monthly[monthly.exit_at < row.decision_at]
        if len(train) < cfg["minimum_train_months"]:
            continue
        training_key = tuple(train.origin_id)
        if training_key not in cache:
            fitted = {}
            for key in FEATURES:
                m = fit(train, key, cfg)
                m.update(fit_id=f"{key}_N{len(train):03d}", first_used_at=row.decision_at)
                all_models.append(m)
                fitted[key] = m
            cache[training_key] = fitted
        fitted = cache[training_key]
        out = row.to_dict()
        out.update(training_months=len(train), prediction_MEAN=float(train.return_20d.mean()), prediction_ZERO=0.,
                   prediction_RMEAN=float(train.downside_variance_20d.mean()), prediction_DOWN20=float(row.down20))
        for key, model in fitted.items():
            out["prediction_" + key] = predict(model, row)
            out["fit_" + key] = model["fit_id"]
        fresh_comparison = row.is_release and not row.is_weekly and np.isfinite(row.stale_growth_level) and np.isfinite(row.stale_growth_change)
        out["refresh_eligible"] = bool(fresh_comparison)
        out["prediction_P1_stale"] = np.nan
        out["prediction_R1_stale"] = np.nan
        if fresh_comparison:
            stale = row.copy()
            stale["growth_level"], stale["growth_change"] = row.stale_growth_level, row.stale_growth_change
            out["prediction_P1_stale"] = predict(fitted["P1"], stale)
            out["prediction_R1_stale"] = predict(fitted["R1"], stale)
        for capital in cfg["capital"]:
            threshold = cost_threshold(row.close, capital, cfg)
            out["threshold_" + str(capital)] = threshold
            for key in ("P0", "P1"):
                out[f"implied_entry_{key}_{capital}"] = bool(out["prediction_" + key] > threshold)
        for key in ("R0", "R1"):
            out["risk_budget_weight_" + key] = float(min(1., .10 / np.sqrt(2 * cfg["cost"]["annual_days"] * out["prediction_" + key])))
        records.append(out)
    return pd.DataFrame(records), all_models


def qlike(actual, forecast):
    return np.log(forecast) + actual / forecast


def adjudicate(pred: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    primary = pred[pred.is_release].reset_index(drop=True).copy()
    n = len(primary)
    if n < cfg["minimum_evaluation_months"]:
        raise ValueError("月度评价样本不足，停止而不缩小样本门")
    b = cfg["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    starts = rng.integers(0, n - b["month_block"] + 1, size=(b["draws"], math.ceil(n / b["month_block"])))
    indices = (starts[:, :, None] + np.arange(b["month_block"])).reshape(b["draws"], -1)[:, :n]
    metrics, comparisons, draws = [], [], {}
    for key in ("P0", "P1", "MEAN", "ZERO", "R0", "R1", "RMEAN", "DOWN20"):
        forecast = primary["prediction_" + key].to_numpy()
        is_risk = key in ("R0", "R1", "RMEAN", "DOWN20")
        actual = primary.downside_variance_20d.to_numpy() if is_risk else primary.return_20d.to_numpy()
        loss = qlike(actual, forecast) if is_risk else np.square(actual - forecast)
        primary["loss_" + key] = loss
        metrics.append({"model": key, "target": "下行方差" if is_risk else "收益均值", "evaluation_months": n,
            "mean_loss": float(loss.mean()), "mse": float(np.square(actual - forecast).mean()),
            "rmse_pp": None if is_risk else float(np.sqrt(loss.mean()) * 100),
            "underestimate_frequency": float((forecast < actual).mean()) if is_risk else None,
            "actual_to_prediction_mean_ratio": float(actual.mean() / forecast.mean()) if is_risk else None,
            "positive_predictions": None if is_risk else int((forecast > 0).sum())})
    verdicts = {}
    for candidate, references in [("P1", ["P0", "MEAN", "ZERO"]), ("R1", ["R0", "RMEAN", "DOWN20"])]:
        candidate_rows = []
        for reference in references:
            diff = (primary["loss_" + reference] - primary["loss_" + candidate]).to_numpy()
            means = diff[indices].mean(axis=1)
            name = reference + "_minus_" + candidate
            draws[name] = means
            gain = float(diff.mean())
            low = float(np.quantile(means, b["lower_quantile"]))
            halves = [float(diff[:n // 2].mean()), float(diff[n // 2:].mean())]
            item = {"candidate": candidate, "reference": reference, "mean_improvement": gain,
                "one_sided_95pct_lower": low, "two_sided_90pct_lower": low, "two_sided_90pct_upper": float(np.quantile(means, .95)),
                "first_half_improvement": halves[0], "second_half_improvement": halves[1],
                "gate_mean_and_lower_positive": bool(gain > 0 and low > 0)}
            comparisons.append(item)
            candidate_rows.append(item)
        passed = all(r["gate_mean_and_lower_positive"] for r in candidate_rows) and candidate_rows[0]["first_half_improvement"] > 0 and candidate_rows[0]["second_half_improvement"] > 0
        verdicts[candidate] = "PASS_HISTORICAL_PREDICTIVE_GATE_ACCOUNT_STAGE_REQUIRED" if passed else "REJECTED_FROZEN_NO_RELIABLE_INCREMENT"
    refresh = pred[pred.refresh_eligible].copy()
    refresh_summary = {"events": len(refresh), "evaluation": "共同起点与相同20日终点；只替换最新/上次周度已知增长值，价格与训练系数相同。", "is_continuous_account": False}
    for key in ("P1", "R1"):
        actual = refresh.return_20d if key == "P1" else refresh.downside_variance_20d
        fresh, stale = refresh["prediction_" + key], refresh["prediction_" + key + "_stale"]
        fresh_loss = np.square(actual - fresh) if key == "P1" else qlike(actual, fresh)
        stale_loss = np.square(actual - stale) if key == "P1" else qlike(actual, stale)
        refresh_summary[key + "_mean_loss_improvement"] = float((stale_loss - fresh_loss).mean())
        refresh_summary[key + "_fresh_better_count"] = int((fresh_loss < stale_loss).sum())
        refresh_summary[key + "_prediction_change_mean_abs"] = float((fresh - stale).abs().mean())
    refresh_summary["main_capital_implied_entry_changes"] = int(((refresh.prediction_P1 > refresh.threshold_200000) != (refresh.prediction_P1_stale > refresh.threshold_200000)).sum())
    return {"status": "COMPLETED_FROZEN_GROWTH_PREDICTION_EXPERIMENT", "verdicts": verdicts,
        "evaluation_months": n, "first_evaluation_origin": primary.date.iloc[0], "last_evaluation_origin": primary.date.iloc[-1],
        "all_review_origins": len(pred), "full_historical_period_already_observed": True,
        "refresh_diagnostic": refresh_summary, "new_accounts": 0, "independent_forward_events": 0,
        "account_status": {k: "PENDING_SEPARATE_ACCOUNT_EXECUTION" if v.startswith("PASS") else "NOT_RUN_OWN_PREDICTION_GATE" for k, v in verdicts.items()},
        "whole_macro_program_complete": False, "metrics": metrics, "comparisons": comparisons}, primary, pd.DataFrame(draws), pd.DataFrame(metrics), indices


def execute(study: Path = OUT, calendar_recovery: bool = False) -> None:
    cfg = verify_freeze(study)
    if calendar_recovery:
        if not (study / "calendar_correction_receipt.json").exists() or any((study / "results").iterdir()):
            raise ValueError("仅允许在无真实标签和结果时恢复已记录的日历衔接失败")
        start_file = study / "run_started_calendar_recovery.json"
    else:
        start_file = study / "run_started.json"
    if start_file.exists():
        raise FileExistsError("本实验执行已启动过；只允许只读复核，禁止覆盖研究结果")
    save(start_file, {"started_at": now(), "protocol_sha256": sha(study / "protocol.json"), "calendar_recovery": calendar_recovery})
    panel, origins = panel_and_origins(study, cfg)
    csv(origins, study / "results/全部周度与公布原点及20日标签.csv")
    csv(origins[origins.is_release], study / "results/全部月度训练原点.csv")
    panel.to_parquet(study / "results/每日已知输入.parquet", index=False)
    csv(panel, study / "results/每日已知输入.csv")
    pred, models = walk_forward(origins, cfg)
    csv(pred, study / "results/全部逐期预测.csv")
    save(study / "results/固定模型与训练集合.json", models)
    summary, primary, draws, metrics, indices = adjudicate(pred, cfg)
    summary.update(created_at=now(), study_id=STUDY, training_monthly_origins=int(origins.is_release.sum()),
                   model_fits=len(models), new_20d_labeled_origins=len(origins), price_daily_points=len(panel))
    csv(primary, study / "results/月度主要评价及损失.csv")
    csv(draws, study / "results/固定区块损失改善抽样.csv")
    np.savez_compressed(study / "results/固定区块抽样索引.npz", indices=indices.astype(np.int16))
    csv(metrics, study / "results/八项基准汇总.csv")
    csv(pd.DataFrame(summary["comparisons"]), study / "results/六项预定增量对照.csv")
    yearly = primary.assign(year=primary.date.str[:4]).groupby("year").agg(months=("origin_id", "size"),
        P0_loss=("loss_P0", "mean"), P1_loss=("loss_P1", "mean"), R0_loss=("loss_R0", "mean"), R1_loss=("loss_R1", "mean")).reset_index()
    yearly["return_improvement"] = yearly.P0_loss - yearly.P1_loss
    yearly["risk_improvement"] = yearly.R0_loss - yearly.R1_loss
    csv(yearly, study / "results/逐年预测贡献.csv")
    save(study / "results/summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("comparisons", "metrics")}, ensure_ascii=False, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="增长状态二十日收益与下行风险固定增量检验")
    parser.add_argument("action", choices=["freeze", "run", "resume-calendar"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        execute(calendar_recovery=args.action == "resume-calendar")
