"""财政实际执行、融资状态与持续更新：预定的二十日收益/下行风险检验。"""
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
OUT = ROOT / "reports/research/510300_fiscal_execution_state_20d_v1"
REF = ROOT / "reports/research/510300_macro_dynamic_reframe_v1"
GROWTH = ROOT / "reports/research/510300_growth_state_increment_20d_v1"
DR = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet"
STUDY = "510300_FISCAL_EXECUTION_STATE_20D_V1"
FEATURES = {}
for _prefix, _risk in [("P", "price_log_sigma20"), ("R", "price_log_down20")]:
    FEATURES[_prefix + "0"] = ["price_m20", _risk, "funding_spread20_pp"]
    FEATURES[_prefix + "1"] = FEATURES[_prefix + "0"] + ["general_model_yoy_pp", "fund_model_yoy_pp"]
    FEATURES[_prefix + "2"] = FEATURES[_prefix + "1"] + ["general_funding_interaction"]
FEATURES["P_REV"] = ["price_m20"]
FEATURES["P_MOM"] = ["price_m20"]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def freeze():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("协议已冻结，不能覆盖或反复选择参数")
    fiscal = pd.read_parquet(OUT / "fiscal_observations.parquet")
    expected = [str(p) for p in pd.period_range("2015-02", "2026-07", freq="M") if p.month != 1]
    if sorted(fiscal.reference_period) != expected:
        raise ValueError("原文月度母集未补齐，暂不冻结并运行模型")
    for folder in ["inputs", "code", "results", "figures", "evidence"]:
        (OUT / folder).mkdir(exist_ok=True)
    copies = {"fiscal.parquet": OUT / "fiscal_observations.parquet", "market.parquet": REF / "inputs/market.parquet", "operation_rates.parquet": REF / "inputs/operation_rate_records.parquet", "dr007.parquet": DR, "calendar.parquet": GROWTH / "inputs/calendar_extended.parquet"}
    for name, source in copies.items():
        shutil.copy2(source, OUT / "inputs" / name)
    for name in ["fiscal_execution_source_v1.py", "fiscal_execution_state_20d_v1.py"]:
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    config = {"study_id": STUDY, "frozen_at": now(), "evidence_class": "FIXED_NEW_MECHANISM_ON_ALREADY_OBSERVED_HISTORY_NOT_INDEPENDENT_VALIDATION",
        "mechanism": "财政已执行支出可能影响需求及银行体系流动性；检验一般公共预算、政府性基金各自累计可比增速在价格与融资条件之外的收益/风险信息，及一般公共预算增速与融资松紧的一项交互。既不假定支出增加必涨，也不等同新增政策冲击、市场预期差或到账企业利润。",
        "scope": "宏观主线中的财政执行子实验；不重新打开旧M1/M2五日、新订单二十日、操作数量、直接效用或终止85/15方案。融资状态仅作控制，不因本轮结果重宣其旧实现有效。",
        "features": FEATURES, "feature_rules": {"fiscal": "当次正文全国累计支出同比；明确公布同口径同比时使用同口径，否则用当次标题口径。一般公共预算与政府性基金分开，不相加；单位百分点；增长1.1倍是+110%。不拆1—2月，不以累计差分制造单月流量。",
        "funding": "每个A股交易收盘只使用日期严格早于当日的最近DR007加权利率，减截至收盘已公布的7天逆回购实际操作利率；取过去20个A股交易日平均，单位百分点。DR007是继承供应商历史，非逐日原始首发版本认证。",
        "interaction": "一般公共预算可比累计同比百分点乘融资利差20日均值百分点，允许两者的条件联系由成熟训练样本估计；不预设方向、不搜索阈值。",
        "price": "收盘已知的20个含分红日回报复合收益、标准差对数、负收益平方均值对数。"},
        "source_availability": "财政本页明确公开日、正文日期和目录日期取最大值，日期精度一律23:59:59上界；不用网址创建日或月份末。首个此后收盘观察，下一交易日开盘后评价。较晚转载保守延迟，不能当最早市场已知日。",
        "horizon": 20, "lookback": 20, "last_decision_date": "2026-07-31", "price_end": "2026-09-11", "minimum_train_events": 36, "minimum_eval_events": 24,
        "decision_grid": "每周最后A股交易日收盘，加财政原文首次合格可用收盘、7天逆回购新利率操作记录首次可用收盘；同日去重。财政公布原点仅每份原文一次用于训练和主要评价。",
        "training": "扩展的财政公布原点；标签20日终点严格早于当前判断收盘。所有模型共同训练集合；成熟集合相同仅拟合一次。周度和公告触发只更新当时已知输入。",
        "ridge_penalty": 1., "ridge": "训练内总体标准差标准化；平均平方误差加斜率平方和；截距不惩罚。没有窗口、方向、惩罚、配比搜索。",
        "return_target": "次日开盘买入一份到第20个交易日收盘，入场后除息的现金权利计入，不再投资；不获得入场前跳涨、入场当日除息权益。",
        "risk_target": "相同路径、含累计现金权益的20个日财富回报负值平方均值；首日从开盘起，不含已发生的公告跳空。",
        "risk_floor": 1e-8, "risk_model": "训练log(max(下行方差,1e-8))，指数预测乘训练残差指数均值；风险比较主损失为QLIKE。",
        "references": {"P1": ["P0", "MEAN", "ZERO", "P_REV", "P_MOM"], "P2": ["P1", "P0", "MEAN", "ZERO", "P_REV", "P_MOM"], "R1": ["R0", "RMEAN", "DOWN20"], "R2": ["R1", "R0", "RMEAN", "DOWN20"]},
        "reversal_baselines": "依据用户新增的均值回归问题，在财政拟合之前增加两个同训练集合、同惩罚的单变量对照：仅过去20日收益，P_REV斜率限制<=0，P_MOM斜率限制>=0；若触及边界，退化为成熟样本均值。不能用全历史选方向。与宏观相同的次日开盘后20日标签，不计当前收盘到入场开盘的跳空。",
        "known_reversal_diagnostic": "此前已在独立固定月末诊断观察170原点：81反转、88延续、1零，反转频数47.9%；该诊断不参与任何阈值、方向或系数选择，也不是完整账户证据。",
        "gates": "四个候选分别评价；每项对照损失改善均值>0且单侧98.75%区块下界>0，以Bonferroni保守分配四候选的5%错误额度。相对各自主基准前后两半改善均>0，至少24评价事件。不能在失败后改成另一指标宣称有效。此校正不覆盖仓库历次研究选择。",
        "bootstrap": {"draws": 10000, "block_events": 6, "seed": 20260922, "lower_quantile": .0125, "method": "非循环连续财政公布序列移动块，不将每周重复宏观值算新增观察；共享抽样索引。"},
        "refresh_diagnostic": "非周度且非首个周度以前的公告复核，用相同当时价格、相同训练系数、相同后续20日标签，比较最新财政和融资状态与上个周末已知状态；不是连续账户，也不收获已发生首轮反应。",
        "cost": {"annual_days": 242, "commission": .0003, "minimum": 5., "stress_slippage": .0005, "lot": 100, "cash_rate": 0., "risk_free_rate": 0.}, "capital": [200000, 20000],
        "account_gate": "本模块通过自身预定信息门才运行账户及动态配置；失败保留NOT_RUN，不填零收益。资金行为、其他宏观通道可独立继续，不被本次成败锁死。",
        "exit_boundary": "涨幅较大后再依回撤退出属于固定入场后的独立比较；本次没有搜卖点。", "strict_forward_events": 0, "new_orders_or_services": False}
    save(OUT / "protocol.json", config)
    identities = {str(p.relative_to(OUT)): sha(p) for p in [OUT / "protocol.json", *(OUT / "inputs").glob("*"), *(OUT / "code").glob("*.py")]}
    save(OUT / "freeze.json", {"frozen_at": now(), "identities": identities, "model_runs_before_freeze": 0, "data_rows": len(fiscal)})
    print("财政执行协议已冻结；尚未读取未来标签或运行模型。")


def label(market, pos, horizon):
    end = pos + horizon
    if end >= len(market):
        raise ValueError("标签未成熟，不截短持有区间")
    block = market.iloc[pos + 1:end + 1]
    dividends = block.dividend.to_numpy(float).copy()
    dividends[0] = 0.
    cash = dividends.cumsum()
    wealth = block.close.to_numpy(float) + cash
    entry = float(block.open.iloc[0])
    returns = wealth / np.r_[entry, wealth[:-1]] - 1
    return {"entry_date": str(block.date.iloc[0].date()), "exit_date": str(block.date.iloc[-1].date()), "exit_at": (block.date.iloc[-1] + pd.Timedelta(hours=15)).isoformat(), "entry_open": entry, "exit_close": float(block.close.iloc[-1]), "earned_dividend": float(cash[-1]), "return_20d": float(wealth[-1] / entry - 1), "downside_variance_20d": float(np.square(np.minimum(returns, 0.)).mean())}


def panel_and_origins(cfg):
    m = pd.read_parquet(OUT / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    m["date"] = pd.to_datetime(m.date).astype("datetime64[ns]")
    m["decision_at"] = m.date + pd.Timedelta(hours=15)
    if not np.allclose(((m.close + m.dividend) / m.close.shift() - 1).iloc[1:], m.total_simple.iloc[1:], atol=1e-12):
        raise ValueError("价格分红恒等式不成立")
    calendar = pd.read_parquet(OUT / "inputs/calendar.parquet")
    trades = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    expected = trades[(trades >= m.date.min()) & (trades <= m.date.max())]
    if list(expected) != m.date.tolist():
        raise ValueError("日频行情与交易日历不一致")
    f = pd.read_parquet(OUT / "inputs/fiscal.parquet").sort_values("available_at")
    f["fiscal_at"] = pd.to_datetime(f.available_at, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).astype("datetime64[ns]")
    f = f.sort_values(["fiscal_at", "reference_period"], kind="stable")
    d = pd.read_parquet(OUT / "inputs/dr007.parquet").sort_values("date").rename(columns={"date": "dr_date"})
    d["dr_date"] = pd.to_datetime(d.dr_date).astype("datetime64[ns]")
    r = pd.read_parquet(OUT / "inputs/operation_rates.parquet").sort_values("published_at").rename(columns={"published_at": "rate_at"})
    r["rate_at"] = pd.to_datetime(r.rate_at).astype("datetime64[ns]")
    m = pd.merge_asof(m, d[["dr_date", "dr007"]], left_on="date", right_on="dr_date", direction="backward", allow_exact_matches=False)
    m = pd.merge_asof(m, r[["rate_at", "seven_day_rate_percent"]], left_on="decision_at", right_on="rate_at", direction="backward")
    m["funding_spread_pp"] = m.dr007 - m.seven_day_rate_percent
    m["funding_spread20_pp"] = m.funding_spread_pp.rolling(20, min_periods=20).mean()
    m["price_m20"] = np.expm1(np.log1p(m.total_simple).rolling(20).sum())
    m["sigma20"] = m.total_simple.rolling(20).std(ddof=1)
    m["price_log_sigma20"] = np.log(m.sigma20)
    m["down20"] = m.total_simple.clip(upper=0).pow(2).rolling(20).mean().clip(lower=cfg["risk_floor"])
    m["price_log_down20"] = np.log(m.down20)
    m = pd.merge_asof(m, f[["fiscal_at", "reference_period", "general_model_yoy_pp", "fund_model_yoy_pp"]], left_on="decision_at", right_on="fiscal_at", direction="backward")
    m["general_funding_interaction"] = m.general_model_yoy_pp * m.funding_spread20_pp
    days = pd.DataFrame({"date": trades})
    weekly = set(days.groupby(days.date.dt.to_period("W-FRI")).date.max())
    fiscal_pos, rate_pos = set(), set()
    for frame, col, positions in [(f, "fiscal_at", fiscal_pos), (r, "rate_at", rate_pos)]:
        for time in frame[col]:
            pos = int(np.searchsorted(m.decision_at.to_numpy(), np.datetime64(time)))
            if pos < len(m):
                # 同一收盘只复核一次；merge_asof在相同可用时点保留最新所属期。
                positions.add(pos)
    required = sorted(set(sum(FEATURES.values(), [])))
    rows, last_week = [], None
    for pos, row in m.iterrows():
        is_weekly, is_fiscal, is_rate = row.date in weekly, pos in fiscal_pos, pos in rate_pos
        valid = row.date <= pd.Timestamp(cfg["last_decision_date"]) and row[required].notna().all()
        if valid and (is_weekly or is_fiscal or is_rate):
            item = {"origin_id": "FISCAL_" + row.date.strftime("%Y%m%d"), "origin_index": int(pos), "date": str(row.date.date()), "decision_at": row.decision_at.isoformat(), "fiscal_at": row.fiscal_at.isoformat(), "reference_period": row.reference_period, "rate_at": row.rate_at.isoformat(), "dr_date": str(row.dr_date.date()), "is_weekly": is_weekly, "is_fiscal": is_fiscal, "is_rate": is_rate, "close": float(row.close), "sigma20": float(row.sigma20), "down20": float(row.down20), "previous_weekly_date": None, **{k: float(row[k]) for k in required}, **label(m, pos, cfg["horizon"])}
            for k in ["general_model_yoy_pp", "fund_model_yoy_pp", "funding_spread20_pp", "general_funding_interaction"]:
                item["stale_" + k] = np.nan
            if last_week is not None:
                old = m.iloc[last_week]
                item["previous_weekly_date"] = str(old.date.date())
                for k in ["general_model_yoy_pp", "fund_model_yoy_pp", "funding_spread20_pp", "general_funding_interaction"]:
                    item["stale_" + k] = old[k]
            rows.append(item)
        if is_weekly:
            last_week = pos
    return m, pd.DataFrame(rows)


def fit(train, key, cfg):
    columns = FEATURES[key]
    x = train[columns].to_numpy(float)
    target = train["return_20d" if key.startswith("P") else "downside_variance_20d"].to_numpy(float)
    y = target if key.startswith("P") else np.log(np.maximum(target, cfg["risk_floor"]))
    mean, scale = x.mean(0), x.std(0, ddof=0)
    scale = np.where(scale > 0, scale, 1.)
    z = (x - mean) / scale
    beta = np.linalg.solve(z.T @ z / len(train) + cfg["ridge_penalty"] * np.eye(len(columns)), z.T @ (y - y.mean()) / len(train))
    if key == "P_REV":
        beta = np.minimum(beta, 0.)
    if key == "P_MOM":
        beta = np.maximum(beta, 0.)
    residual = y - (y.mean() + z @ beta)
    return {"key": key, "columns": columns, "mean": mean.tolist(), "scale": scale.tolist(), "beta": beta.tolist(), "intercept": float(y.mean()), "smearing": 1. if key.startswith("P") else float(np.exp(residual).mean()), "training_ids": train.origin_id.tolist(), "training_events": len(train), "latest_maturity": str(train.exit_at.max()), "fit_id": key + "_N" + str(len(train))}


def predict(model, row):
    x = np.asarray([row[c] for c in model["columns"]])
    value = float(model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["beta"])
    if model["key"].startswith("R"):
        value = float(np.exp(value) * model["smearing"])
    if not np.isfinite(value):
        raise ValueError("预测非有限，不能事后裁剪")
    return value


def cost_threshold(close, capital, cfg):
    c = cfg["cost"]
    buy, sell = close * (1 + c["stress_slippage"]), close * (1 - c["stress_slippage"])
    quantity = int(capital // (buy * c["lot"])) * c["lot"]
    while quantity > 0 and quantity * buy + max(c["minimum"], c["commission"] * quantity * buy) > capital:
        quantity -= c["lot"]
    if quantity == 0:
        raise ValueError("无可买整手")
    return (quantity * (buy - sell) + max(c["minimum"], c["commission"] * quantity * buy) + max(c["minimum"], c["commission"] * quantity * sell)) / (quantity * close)


def walk_forward(origins, cfg):
    monthly = origins[origins.is_fiscal]
    cache, models, rows = {}, [], []
    for _, row in origins.iterrows():
        train = monthly[monthly.exit_at < row.decision_at]
        if len(train) < cfg["minimum_train_events"]:
            continue
        signature = tuple(train.origin_id)
        if signature not in cache:
            cache[signature] = {}
            for key in FEATURES:
                model = fit(train, key, cfg)
                model["first_used_at"] = row.decision_at
                cache[signature][key] = model
                models.append(model)
        item = row.to_dict()
        item.update(training_events=len(train), prediction_MEAN=float(train.return_20d.mean()), prediction_ZERO=0., prediction_RMEAN=float(train.downside_variance_20d.mean()), prediction_DOWN20=row.down20)
        stale_cols = ["general_model_yoy_pp", "fund_model_yoy_pp", "funding_spread20_pp", "general_funding_interaction"]
        eligible = not row.is_weekly and all(np.isfinite(row["stale_" + c]) for c in stale_cols)
        item["refresh_eligible"] = bool(eligible)
        for key, model in cache[signature].items():
            item["prediction_" + key] = predict(model, row)
            item["fit_" + key] = model["fit_id"]
            item["prediction_" + key + "_stale"] = np.nan
            if eligible:
                stale = row.copy()
                for c in stale_cols:
                    stale[c] = row["stale_" + c]
                item["prediction_" + key + "_stale"] = predict(model, stale)
        for capital in cfg["capital"]:
            threshold = cost_threshold(row.close, capital, cfg)
            item["threshold_" + str(capital)] = threshold
            for key in ["P0", "P1", "P2"]:
                item[f"implied_entry_{key}_{capital}"] = bool(item["prediction_" + key] > threshold)
        rows.append(item)
    return pd.DataFrame(rows), models


def loss(actual, predicted, risk=False):
    return np.log(predicted) + actual / predicted if risk else np.square(actual - predicted)


def adjudicate(pred, cfg):
    primary = pred[pred.is_fiscal].reset_index(drop=True).copy()
    n = len(primary)
    if n < cfg["minimum_eval_events"]:
        raise ValueError("评价原点不足，不缩小验收门")
    b = cfg["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    starts = rng.integers(0, n - b["block_events"] + 1, size=(b["draws"], math.ceil(n / b["block_events"])))
    indices = (starts[:, :, None] + np.arange(b["block_events"])).reshape(b["draws"], -1)[:, :n]
    metrics, comparisons, verdicts = [], [], {}
    for key in list(FEATURES) + ["MEAN", "ZERO", "RMEAN", "DOWN20"]:
        risk = key.startswith("R") or key == "DOWN20"
        actual = primary.downside_variance_20d if risk else primary.return_20d
        forecast = primary["prediction_" + key]
        primary["loss_" + key] = loss(actual, forecast, risk)
        metrics.append({"model": key, "target": "下行方差" if risk else "收益均值", "evaluation_events": n, "mean_loss": float(primary["loss_" + key].mean()), "rmse_pp": None if risk else float(np.sqrt(primary["loss_" + key].mean()) * 100), "positive_predictions": None if risk else int((forecast > 0).sum()), "underestimate_fraction": float((forecast < actual).mean()) if risk else None, "mean_actual_to_forecast": float(actual.mean() / forecast.mean()) if risk else None})
    for candidate, refs in cfg["references"].items():
        passed = True
        for j, reference in enumerate(refs):
            gain = (primary["loss_" + reference] - primary["loss_" + candidate]).to_numpy()
            bootstrap_means = gain[indices].mean(1)
            low = float(np.quantile(bootstrap_means, b["lower_quantile"])); high = float(np.quantile(bootstrap_means, 1-b["lower_quantile"]))
            halves = [float(gain[:n//2].mean()), float(gain[n//2:].mean())]
            ok = bool(gain.mean() > 0 and low > 0 and (j != 0 or min(halves) > 0))
            passed = passed and ok
            comparisons.append({"candidate": candidate, "reference": reference, "mean_improvement": float(gain.mean()), "one_sided_98_75pct_lower": low, "two_sided_97_5pct_upper": high, "first_half_improvement": halves[0], "second_half_improvement": halves[1], "passed": ok})
        verdicts[candidate] = "PASS_HISTORICAL_PREDICTIVE_GATE_ACCOUNT_STAGE_REQUIRED" if passed else "REJECTED_FROZEN_NO_RELIABLE_INCREMENT"
    fresh = pred[pred.refresh_eligible].copy()
    refresh = []
    for key in FEATURES:
        risk = key.startswith("R")
        y = fresh.downside_variance_20d if risk else fresh.return_20d
        diff = loss(y, fresh["prediction_" + key + "_stale"], risk) - loss(y, fresh["prediction_" + key], risk)
        item = {"model": key, "events": len(fresh), "mean_refresh_improvement": float(diff.mean()), "fresh_better_events": int((diff > 0).sum()), "mean_absolute_prediction_change": float((fresh["prediction_" + key] - fresh["prediction_" + key + "_stale"]).abs().mean())}
        for capital in cfg["capital"]:
            item["implied_entry_changes_" + str(capital)] = None if risk else int(((fresh["prediction_" + key] > fresh["threshold_" + str(capital)]) != (fresh["prediction_" + key + "_stale"] > fresh["threshold_" + str(capital)])).sum())
        refresh.append(item)
    summary = {"study_id": STUDY, "created_at": now(), "status": "COMPLETED_FIXED_FISCAL_EXECUTION_AND_STATE_EXPERIMENT", "evaluation_events": n, "first_evaluation": str(primary.date.iloc[0]), "last_evaluation": str(primary.date.iloc[-1]), "verdicts": verdicts, "account_status": {k: "PENDING_ACCOUNT_STAGE" if v.startswith("PASS") else "NOT_RUN_OWN_PREDICTION_GATE" for k,v in verdicts.items()}, "model_families": len(FEATURES), "historical_data_previously_observed": True, "strict_forward_events": 0, "continuous_accounts": 0, "exit_rule_comparisons": 0, "whole_macro_program_complete": False, "metrics": metrics, "comparisons": comparisons, "refresh_diagnostic": refresh}
    return summary, primary, indices


def execute(recovery=False, simultaneous=False):
    identities = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))["identities"]
    for name, digest in identities.items():
        if sha(OUT / name) != digest:
            raise ValueError("冻结文件身份变化：" + name)
    start_file = OUT / ("run_started_simultaneous_recovery.json" if simultaneous else "run_started_clock_type_recovery.json" if recovery else "run_started.json")
    if start_file.exists():
        raise FileExistsError("已启动实验；禁止再次运行或覆盖结果")
    if recovery:
        receipt = OUT / ("simultaneous_clock_correction.json" if simultaneous else "clock_type_correction.json")
        if not receipt.exists() or any((OUT / "results").iterdir()):
            raise ValueError("仅允许在没有标签和结果的情况下恢复已记录的时间精度异常")
        if json.loads(receipt.read_text(encoding="utf-8"))["corrected_code_sha256"] != sha(Path(__file__)):
            raise ValueError("时间精度修正代码身份不符")
    cfg = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    save(start_file, {"started_at": now(), "protocol_sha256": sha(OUT / "protocol.json"), "clock_type_recovery": recovery})
    panel, origins = panel_and_origins(cfg)
    panel.to_parquet(OUT / "results/每日已知财政融资与价格.parquet", index=False)
    csv(panel, OUT / "results/每日已知财政融资与价格.csv")
    csv(origins, OUT / "results/全部周度公告原点及标签.csv")
    predictions, models = walk_forward(origins, cfg)
    csv(predictions, OUT / "results/全部逐期预测.csv")
    save(OUT / "results/固定模型参数与训练集合.json", models)
    summary, primary, indices = adjudicate(predictions, cfg)
    summary.update(training_fiscal_origins=int(origins.is_fiscal.sum()), all_review_origins=len(origins), predicted_review_origins=len(predictions), model_fits=len(models), price_daily_points=len(panel))
    csv(primary, OUT / "results/财政主要评价与损失.csv")
    csv(pd.DataFrame(summary["metrics"]), OUT / "results/全部模型及简单基准.csv")
    csv(pd.DataFrame(summary["comparisons"]), OUT / "results/预定增量对照.csv")
    csv(pd.DataFrame(summary["refresh_diagnostic"]), OUT / "results/公告更新共同起点诊断.csv")
    np.savez_compressed(OUT / "results/固定区块索引.npz", indices=indices.astype(np.int16))
    annual = primary.assign(year=primary.date.str[:4]).groupby("year").agg(events=("origin_id", "size"), **{key + "_loss": ("loss_" + key, "mean") for key in FEATURES}).reset_index()
    for key, base in [("P1","P0"),("P2","P1"),("R1","R0"),("R2","R1")]:
        annual[key + "_improvement"] = annual[base + "_loss"] - annual[key + "_loss"]
    csv(annual, OUT / "results/逐年贡献.csv")
    save(OUT / "results/summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ["metrics", "comparisons", "refresh_diagnostic"]}, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary["metrics"]).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="财政执行与融资状态固定检验")
    parser.add_argument("action", choices=["freeze", "run", "recover-clock-type", "recover-simultaneous"])
    args = parser.parse_args()
    {"freeze": freeze, "run": execute, "recover-clock-type": lambda: execute(True), "recover-simultaneous": lambda: execute(True, True)}[args.action]()
