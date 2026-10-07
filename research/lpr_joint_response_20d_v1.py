"""LPR公开变更与股债联合反应的固定二十日增量检验。"""
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
OUT = ROOT / "reports/research/510300_lpr_joint_response_20d_v1"
BASE = ["past20", "logvol20", "return1", "lpr_age20"]
RBASE = ["past20", "logdown20", "return1", "lpr_age20"]
LPR = ["lpr_delta1_pp", "lpr_delta5_pp"]
JOINT = ["yield_change_bp", "stock_yield_interaction"]
FEATURES = {"P0": BASE, "P_LPR": BASE + LPR, "P_JOINT": BASE + LPR + JOINT,
            "R0": RBASE, "R_LPR": RBASE + LPR, "R_JOINT": RBASE + LPR + JOINT,
            "P_REV": ["past20"], "P_MOM": ["past20"]}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def export(frame, name):
    frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)
    frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.17g")


def freeze():
    if (OUT / "freeze.json").exists():
        raise FileExistsError("研究已冻结，不得覆盖")
    observations = pd.read_parquet(OUT / "lpr_observations_precise.parquet")
    if len(observations) != 84 or observations.month.duplicated().any():
        raise ValueError("LPR固定完整月份未通过")
    sources = {
        "lpr.parquet": OUT / "lpr_observations_precise.parquet",
        "market.parquet": ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet",
        "calendar.parquet": ROOT / "reports/research/510300_growth_state_increment_20d_v1/inputs/calendar_extended.parquet",
        "bond.parquet": ROOT / "data/raw/macro/china_government_bond_yields_daily.parquet",
        "bond_prior_release_ledger.parquet": ROOT / "data/curated/510300_stress_transmission_hazard_v2_mft_feature_execution_v1/china_10y_yield_release_ledger.parquet",
    }
    for name, source in sources.items():
        shutil.copy2(source, OUT / "inputs" / name)
    for name in ["lpr_joint_response_20d_v1.py", "lpr_joint_response_sources_v1.py"]:
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    cfg = {
        "study_id": "510300_LPR_JOINT_RESPONSE_20D_V1", "frozen_at": now(),
        "hypotheses": {"LPR": "贷款定价的实际月度变化是否在价格和复核时钟之外提供增量；1年与5年以上分别表达，不施加系数相反约束。",
                       "JOINT": "在同样贷款定价背景下，当前股价与10年国债收益率的联合变化是否改变下一阶段收益或风险判断。",
                       "causal_limit": "日终曲线与日收益受多种消息影响，不是高频结构性货币政策意外识别，不把联合符号直接贴经济状态标签。"},
        "sources": {name: str(path) for name, path in sources.items()},
        "asset": "510300.SH", "cash": "CASH_CNY", "capital_main": 200000, "capital_comparison": 20000,
        "annual_days": 242, "cash_rate": 0, "risk_free_rate": 0,
        "anchor_start": "2019-10-01", "anchor_end": "2026-07-31", "horizon": 20,
        "start_reason": "2019年8月新LPR作为初始值，9月建立第一笔同口径变化；10月是有完整先前变化值的首个完整自然月。",
        "review_clock": "每周最后股票交易日，以及每份LPR公告上界之后第一个完整股市交易日，合并同日。开盘必须严格晚于公告上界。",
        "decision_clock": "复核日之后下一股票交易日09:00形成判断，09:30开盘作为收益起点；此时已取得复核日全部股价及日终中债曲线。",
        "lpr_clock": "只用复核日开盘前已公布的LPR，保留9:30/9:15/9:00历史原文时间；同日开盘恰与公告重合时保守顺延。",
        "bond_clock": "原日曲线定义为工作日日终17:30发布；本轮另用观测日23:59:59保守上界。原09:30下一股票日衍生账簿仅作数值交叉核对，不改写旧研究时钟。",
        "bond_same_interval": "收益率变化为复核日与前一股票交易日10年曲线差，单位bp；两日均须有原值，不前向填充或插值。不是可交易债券收益。",
        "features": FEATURES,
        "feature_definition": "过去20日含分红回报、当日含分红收益、20日对数标准差/对数下行方差；公告首个完整交易日以来股票交易日数/20控制信息年龄；两个LPR最近月度变化分别以百分点表示；唯一联合项=当前股票日回报乘同区间10年收益率bp变化。",
        "announcement_change_hold": "LPR月度变化与最近公告状态持续到下一次已公布月度报价；未变报价明确为0变化，未知预期仍为空。信息年龄进入共同基准，周内变化由实际新价格/债券和新公告更新。",
        "labels": "从下一开盘起20个收盘点加现金分红权益；不享有入场除息日分红、不再投资。下行风险为该权益路径20个日收益负部平方的均值。",
        "risk_floor": 1e-8,
        "training": "扩展窗口，只训练完整自然月的全部有效复核原点，且该月所有20日标签已于决策09:00前成熟。各月总权重相等，月内各原点等权。完整36个月后开始。训练集合不变时复用参数。",
        "minimum_train_months": 36, "minimum_eval_months": 24,
        "ridge_lambda": 1.0, "standardization": "仅当前成熟训练集按月等权均值和总体标准差；截距不惩罚；无裁剪、网格或窗口选择。",
        "risk_model": "对数下行方差为训练目标，按相同权重的训练残差指数均值作smearing回转；预测下限1e-8。",
        "sign_baselines": "P_REV单一past20斜率<=0，P_MOM>=0；边界归零时等于成熟月度等权均值，不称为方向策略胜出。",
        "primary_evaluation": "仅保留当月全部计划复核原点都有共同有效输入和预测的完整评价月；先在月内平均损失，再给各月相等权重。初始不完整评价月保留为诊断但不混入主要统计。",
        "bootstrap": {"draws": 10000, "block_months": 6, "seed": 20260922, "lower_quantile": .0125,
                      "rule": "沿时间排序的连续六个有效评价月非循环区块；若存在缺月保留，区块实际跨度可超过六个自然月，不补零。"},
        "references": {"P_LPR": ["P0", "MEAN", "ZERO", "P_REV", "P_MOM"],
                       "P_JOINT": ["P_LPR", "P0", "MEAN", "ZERO", "P_REV", "P_MOM"],
                       "R_LPR": ["R0", "RMEAN", "DOWN20"], "R_JOINT": ["R_LPR", "R0", "RMEAN", "DOWN20"]},
        "gate": "收益用MSE，风险用QLIKE；相对每一预定基准平均改善>0，单侧98.75%区块下界>0；对首列直接基准前后半段也都改善。四候选Bonferroni局部控制，不覆盖上游历史搜索。",
        "update_diagnostic": "非周度LPR复核日，在相同价格、债券、年龄、训练模型与未来标签下，仅将两个LPR变化值替换为上一周度复核时已知值，比较新旧报价内容；这是固定的保留旧内容对照，不是连续账户。",
        "account_gate": "未通过自身目标门的候选保持NOT_RUN；通过的候选必须继续比较周度与周度加公告更新的完整账户，收益候选固定入场后再进行一项涨后回撤退出比较。此预测结果本身不宣称账户达标。",
        "account_predefinition": {"commission_rate_per_side": .0002, "minimum_commission": 5,
                                  "slippage_base_per_side": .0005, "slippage_stress_per_side": .001,
                                  "lot_size": 100, "t_plus_one": True,
                                  "return_candidate_entry": "正的预期20日收益须超过按当时价格、整手与本金测得的往返费用；同一简单10%目标波动上限、仓位0至1。",
                                  "risk_candidate_budget": "恒定持有动机下，10%/sqrt(2)年化下行风险预算除以sqrt(242*预测下行方差)，仓位0至1，与R0同预算比较。",
                                  "exit_candidate": "沿用既有单一设定：入场前20日日波动σ，收盘浮盈达到2σ启动，从持有收盘高点回撤σ后次开盘退出；同次入场不重新启动，最迟固定20交易日收盘。"},
        "prior_families_preserved": ["MONEY_CONSENSUS_INCREMENT_V2", "GROWTH_STATE_INCREMENT_20D_V1", "FISCAL_EXECUTION_STATE_20D_V1", "POLICY_LIQUIDITY_QUANTITY_V1", "SELECTED_MIX_BAND10_SIMPLE2", "NBS_FIXED_5MIN_FAMILY"],
        "historical_prices_previously_observed": True, "historical_source_first_versions_authenticated": False,
        "strict_forward_events": 0, "whole_macro_objective_complete": False,
    }
    save(OUT / "protocol.json", cfg)
    paths = [OUT / "protocol.json", OUT / "source_plan.json"] + list((OUT / "inputs").glob("*")) + list((OUT / "code").glob("*"))
    paths += [OUT / p for p in observations.raw_path] + [OUT / "raw/chinabond_clock.html"]
    save(OUT / "freeze.json", {"frozen_at": now(), "before_this_study_labels_and_fits": True,
         "identities": {p.relative_to(OUT).as_posix(): sha(p) for p in sorted(set(paths))},
         "not_independent_unseen_history": True})
    print("收益、风险、月度权重、联合项、信息时钟及账户门已冻结。", flush=True)


def build(cfg):
    market = pd.read_parquet(OUT / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).astype("datetime64[ns]")
    calendar = pd.read_parquet(OUT / "inputs/calendar.parquet")
    calendar = calendar[calendar.is_open].copy()
    calendar["day"] = pd.to_datetime(calendar.trade_date).astype("datetime64[ns]")
    calendar = calendar.sort_values("day")
    week_ends = set(calendar.groupby(calendar.day.dt.to_period("W-SUN")).day.max())
    bonds = pd.read_parquet(OUT / "inputs/bond.parquet").sort_values("date")
    bonds["date"] = pd.to_datetime(bonds.date).astype("datetime64[ns]")
    if bonds.date.duplicated().any():
        raise ValueError("国债日期重复")
    old = pd.read_parquet(OUT / "inputs/bond_prior_release_ledger.parquet")
    old["date"] = pd.to_datetime(old.observation_date).astype("datetime64[ns]")
    joined = bonds.merge(old[["date", "china_10y_yield"]], on="date", how="inner")
    if not np.allclose(joined.cgb_10y, joined.china_10y_yield, rtol=0, atol=1e-12):
        raise ValueError("相同10年曲线继承值不一致")
    save(OUT / "evidence/bond_crosscheck.json", {"overlapping_dates": len(joined), "all_values_equal": True,
         "scope": "仅核对两个保存版本公共日期数值；不等于日曲线历史首版认证或交易报价。",
         "new_availability_rule": cfg["bond_clock"]})
    index = pd.DatetimeIndex(market.date)
    openings = index.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    lpr = pd.read_parquet(OUT / "inputs/lpr.parquet").sort_values("month").reset_index(drop=True)
    lpr["lpr_delta1_pp"] = lpr.lpr_1y_percent.diff()
    lpr["lpr_delta5_pp"] = lpr.lpr_5y_percent.diff()
    event_positions = []
    for item in lpr.itertuples():
        pos = int(openings.searchsorted(pd.Timestamp(item.available_at), side="right"))
        if pos >= len(index):
            raise ValueError("公告之后无市场日历")
        event_positions.append(pos)
    if len(event_positions) != len(set(event_positions)):
        raise ValueError("多个月度LPR映射同日，须保留数据缺口")
    lpr["review_index"] = event_positions
    lpr["first_full_session"] = [str(index[x].date()) for x in event_positions]
    export(lpr, "全部LPR公布与首次完整复核")
    bond_by_date = bonds.set_index("date").cgb_10y
    market["return1"] = market.total_simple.astype(float)
    market["past20"] = np.expm1(np.log1p(market.return1).rolling(20).sum())
    market["vol20"] = market.return1.rolling(20).std(ddof=1)
    market["logvol20"] = np.log(market.vol20.clip(lower=1e-8))
    market["down20"] = market.return1.clip(upper=0).pow(2).rolling(20).mean()
    market["logdown20"] = np.log(market.down20.clip(lower=cfg["risk_floor"]))
    market["yield10"] = market.date.map(bond_by_date)
    market["yield10_previous_stock_day"] = market.date.shift(1).map(bond_by_date)
    market["yield_change_bp"] = (market.yield10 - market.yield10_previous_stock_day) * 100
    market["stock_yield_interaction"] = market.return1 * market.yield_change_bp
    state_index = np.searchsorted(event_positions, np.arange(len(market)), side="right") - 1
    for column in ["lpr_month", "lpr_available_at", "lpr_first_full_session"]:
        market[column] = ""
    for column in ["lpr_delta1_pp", "lpr_delta5_pp", "lpr_age20"]:
        market[column] = np.nan
    for i, k in enumerate(state_index):
        if k < 0:
            continue
        row = lpr.iloc[k]
        market.loc[i, ["lpr_month", "lpr_available_at", "lpr_first_full_session"]] = [row.month, row.available_at, row.first_full_session]
        market.loc[i, ["lpr_delta1_pp", "lpr_delta5_pp", "lpr_age20"]] = [row.lpr_delta1_pp, row.lpr_delta5_pp, (i - row.review_index) / 20]
    market["is_weekly"] = market.date.isin(week_ends)
    market["is_lpr_review"] = market.index.isin(event_positions)
    market["feature_valid"] = np.isfinite(market[list(set(sum(FEATURES.values(), [])))]).all(axis=1)
    origins = []
    for i, row in market.iterrows():
        if not (row.is_weekly or row.is_lpr_review) or not (cfg["anchor_start"] <= str(row.date.date()) <= cfg["anchor_end"]):
            continue
        if i + cfg["horizon"] >= len(market):
            raise ValueError("固定20日标签未成熟，不能缩短期限")
        block = market.iloc[i + 1:i + 21]
        div = block.dividend.to_numpy(float).copy()
        div[0] = 0
        entry = float(block.open.iloc[0])
        wealth = np.r_[entry, block.close.to_numpy(float) + div.cumsum()]
        daily = wealth[1:] / wealth[:-1] - 1
        item = row.to_dict()
        item.update(origin_id=len(origins), anchor_index=i, date=str(row.date.date()), month=str(row.date.to_period("M")),
                    decision_at=(pd.Timestamp(block.date.iloc[0]).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9)).isoformat(),
                    entry_date=str(block.date.iloc[0].date()), entry_open=entry,
                    label_exit=str(block.date.iloc[-1].date()), label_exit_close=float(block.close.iloc[-1]), dividend_entitlement=float(div.sum()),
                    return_20d=float(wealth[-1] / entry - 1), downside_variance_20d=float(np.minimum(daily, 0).dot(np.minimum(daily, 0)) / 20))
        origins.append(item)
    frame = pd.DataFrame(origins)
    if len(frame) == 0:
        raise ValueError("无复核原点")
    for row in frame[frame.feature_valid].itertuples():
        if not pd.Timestamp(row.lpr_available_at) < pd.Timestamp(row.date + " 09:30:00", tz="Asia/Shanghai"):
            raise ValueError("LPR在完整复核日开盘时尚未可知")
        if not pd.Timestamp(row.date + " 23:59:59", tz="Asia/Shanghai") < pd.Timestamp(row.decision_at):
            raise ValueError("中债日终资料尚未可知")
    return market, frame


def train_model(train, key, cfg):
    columns = FEATURES[key]
    x = train[columns].to_numpy(float)
    risk = key in {"R0", "R_LPR", "R_JOINT"}
    y = train.downside_variance_20d.to_numpy(float) if risk else train.return_20d.to_numpy(float)
    if risk:
        y = np.log(np.maximum(y, cfg["risk_floor"]))
    counts = train.groupby("month").origin_id.transform("size").to_numpy(float)
    weights = 1 / counts
    weights /= weights.sum()
    mean = weights @ x
    scale = np.sqrt(weights @ ((x - mean) ** 2))
    scale[scale < 1e-12] = 1
    z = (x - mean) / scale
    intercept = float(weights @ y)
    beta = np.linalg.solve(z.T @ (weights[:, None] * z) + cfg["ridge_lambda"] * np.eye(len(columns)), z.T @ (weights * (y - intercept)))
    if key == "P_REV":
        beta[0] = min(float(beta[0]), 0)
    if key == "P_MOM":
        beta[0] = max(float(beta[0]), 0)
    fitted = intercept + z @ beta
    smear = float(weights @ np.exp(y - fitted)) if risk else 1.0
    return {"key": key, "features": columns, "mean": mean.tolist(), "scale": scale.tolist(), "coefficient": beta.tolist(),
            "intercept": intercept, "smearing": smear, "train_origin_ids": train.origin_id.astype(int).tolist(),
            "train_months": sorted(train.month.unique()), "train_weights": weights.tolist(),
            "latest_label_exit": train.label_exit.max(), "weighted_mean_return": float(weights @ train.return_20d.to_numpy(float)),
            "weighted_mean_risk": float(weights @ train.downside_variance_20d.to_numpy(float))}


def predict(model, row, cfg):
    x = np.array([row[c] for c in model["features"]], float)
    value = float(model["intercept"] + ((x - model["mean"]) / model["scale"]) @ np.array(model["coefficient"]))
    if model["key"] in {"R0", "R_LPR", "R_JOINT"}:
        value = max(cfg["risk_floor"], float(np.exp(value) * model["smearing"]))
    if not np.isfinite(value):
        raise ValueError("预测非有限数，不用事后裁剪修补")
    return value


def walk(origins, cfg):
    meta = origins.groupby("month").agg(last_label=("label_exit", "max"), feature_complete=("feature_valid", "all"))
    cache, models, predictions = {}, [], []
    for row in origins.to_dict("records"):
        if not row["feature_valid"]:
            continue
        ready = meta.index[(meta.last_label <= row["date"]) & meta.feature_complete & (meta.index < row["month"])]
        if len(ready) < cfg["minimum_train_months"]:
            continue
        train = origins[origins.month.isin(ready)]
        signature = tuple(train.origin_id.astype(int))
        if signature not in cache:
            group = {}
            for key in FEATURES:
                model = train_model(train, key, cfg)
                model.update(model_id=len(models), fitted_for_decision=row["decision_at"])
                models.append(model)
                group[key] = model
            cache[signature] = group
        group = cache[signature]
        result = dict(row)
        result["train_months"] = len(ready)
        for key, model in group.items():
            result["model_" + key] = model["model_id"]
            result["prediction_" + key] = predict(model, row, cfg)
        reference = group["P0"]
        result.update(prediction_MEAN=reference["weighted_mean_return"], prediction_ZERO=0.0,
                      prediction_RMEAN=max(cfg["risk_floor"], reference["weighted_mean_risk"]), prediction_DOWN20=max(cfg["risk_floor"], row["down20"]))
        old_week = origins[(origins.date < row["date"]) & origins.is_weekly & origins.feature_valid]
        result["refresh_diagnostic"] = bool(row["is_lpr_review"] and not row["is_weekly"] and len(old_week))
        result["previous_weekly_date"] = ""
        for key in ["P_LPR", "P_JOINT", "R_LPR", "R_JOINT"]:
            result["prediction_" + key + "_old_content"] = np.nan
        if result["refresh_diagnostic"]:
            old = old_week.iloc[-1]
            stale = dict(row)
            for column in LPR:
                stale[column] = float(old[column])
            result["previous_weekly_date"] = old.date
            for key in ["P_LPR", "P_JOINT", "R_LPR", "R_JOINT"]:
                result["prediction_" + key + "_old_content"] = predict(group[key], stale, cfg)
        predictions.append(result)
    return pd.DataFrame(predictions), models


def loss(y, forecast, risk):
    return np.log(forecast) + y / forecast if risk else (y - forecast) ** 2


def judge(origins, predictions, models, cfg):
    pred = predictions.copy()
    keys = list(FEATURES) + ["MEAN", "ZERO", "RMEAN", "DOWN20"]
    risk_keys = {"R0", "R_LPR", "R_JOINT", "RMEAN", "DOWN20"}
    for key in keys:
        risk = key in risk_keys
        y = pred.downside_variance_20d if risk else pred.return_20d
        pred["loss_" + key] = loss(y, pred["prediction_" + key], risk)
    complete = []
    month_counts = []
    for month, expected in origins.groupby("month", sort=True):
        actual = pred[pred.month == month]
        ok = len(actual) == len(expected) and expected.feature_valid.all()
        month_counts.append({"month": month, "scheduled_reviews": len(expected), "valid_reviews": int(expected.feature_valid.sum()), "predicted_reviews": len(actual), "primary_complete": bool(ok)})
        if ok:
            complete.append(month)
    primary = pred[pred.month.isin(complete)].copy()
    monthly = primary.groupby("month", sort=True).agg(reviews=("origin_id", "size"),
                 **{f"loss_{key}": (f"loss_{key}", "mean") for key in keys}).reset_index()
    n = len(monthly)
    if n < cfg["minimum_eval_months"]:
        raise ValueError("完整评价月份不足；不缩小最低要求")
    b = cfg["bootstrap"]
    rng = np.random.default_rng(b["seed"])
    starts = rng.integers(0, n - b["block_months"] + 1, size=(b["draws"], math.ceil(n / b["block_months"])))
    indices = (starts[:, :, None] + np.arange(b["block_months"])).reshape(b["draws"], -1)[:, :n]
    metrics, comparisons, verdicts = [], [], {}
    for key in keys:
        risk = key in risk_keys
        value = float(monthly["loss_" + key].mean())
        metrics.append({"model": key, "target": "下行方差" if risk else "收益均值", "evaluation_months": n,
                        "evaluation_reviews": len(primary), "mean_loss": value, "rmse_pp": None if risk else float(np.sqrt(value) * 100),
                        "positive_forecasts": None if risk else int((primary["prediction_" + key] > 0).sum())})
    for candidate, references in cfg["references"].items():
        passed = True
        for order, reference in enumerate(references):
            gain = (monthly["loss_" + reference] - monthly["loss_" + candidate]).to_numpy(float)
            samples = gain[indices].mean(axis=1)
            lower, upper = np.quantile(samples, [b["lower_quantile"], 1 - b["lower_quantile"]])
            first, second = float(gain[:n // 2].mean()), float(gain[n // 2:].mean())
            ok = bool(gain.mean() > 0 and lower > 0 and (order != 0 or min(first, second) > 0))
            passed &= ok
            comparisons.append({"candidate": candidate, "reference": reference, "mean_improvement": float(gain.mean()),
                                "one_sided_98_75pct_lower": float(lower), "central_97_5pct_upper": float(upper),
                                "first_half_improvement": first, "second_half_improvement": second, "passed": ok})
        verdicts[candidate] = "PASS_HISTORICAL_GATE_ACCOUNT_STAGE_REQUIRED" if passed else "REJECTED_FROZEN_NO_RELIABLE_INCREMENT"
    update = pred[pred.refresh_diagnostic].copy()
    updates = []
    for key in ["P_LPR", "P_JOINT", "R_LPR", "R_JOINT"]:
        risk = key in risk_keys
        y = update.downside_variance_20d if risk else update.return_20d
        gain = loss(y, update["prediction_" + key + "_old_content"], risk) - loss(y, update["prediction_" + key], risk)
        update["improvement_" + key] = gain
        updates.append({"model": key, "events": len(update), "contents_actually_changed": int((np.abs(update["prediction_" + key] - update["prediction_" + key + "_old_content"]) > 1e-12).sum()),
                        "mean_improvement": float(gain.mean()), "fresh_better": int((gain > 1e-12).sum()), "not_account": True})
    sign_changes = {key: int(((primary["prediction_" + key] > 0) != (primary.prediction_P0 > 0)).sum()) for key in ["P_LPR", "P_JOINT"]}
    summary = {"study_id": cfg["study_id"], "status": "COMPLETED_FROZEN_INFORMATION_GATE",
               "verdicts": verdicts, "account_status": {key: "PENDING_REQUIRED_ACCOUNT_COMPARISON" if value.startswith("PASS") else "NOT_RUN_OWN_PREDICTION_GATE" for key, value in verdicts.items()},
               "review_origins": len(origins), "valid_review_origins": int(origins.feature_valid.sum()), "predicted_origins": len(pred),
               "evaluation_months": n, "evaluation_reviews": len(primary), "first_eval_month": monthly.month.iloc[0], "last_eval_month": monthly.month.iloc[-1],
               "model_fits": len(models), "unique_training_sets": len(models) // len(FEATURES), "metrics": metrics, "comparisons": comparisons,
               "extra_lpr_review_diagnostic": updates, "positive_forecast_direction_changes_vs_P0": sign_changes,
               "raw_lpr_announcements": 84, "monthly_same_observation_not_daily_independence": True,
               "new_accounts": 0, "new_exit_strategy_runs": 0, "strict_forward_events": 0, "whole_macro_objective_complete": False}
    return summary, pred, primary, monthly, update, pd.DataFrame(month_counts), indices


def run():
    if (OUT / "run_started.json").exists():
        raise FileExistsError("本轮已经启动，禁止重复试验")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in frozen["identities"].items():
        if sha(OUT / name) != expected:
            raise ValueError("冻结输入或代码改变：" + name)
    if sha(Path(__file__)) != frozen["identities"]["code/lpr_joint_response_20d_v1.py"]:
        raise ValueError("执行代码与冻结版本不符")
    cfg = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    save(OUT / "run_started.json", {"started_at": now(), "frozen_code_sha256": sha(Path(__file__))})
    market, origins = build(cfg)
    export(market, "全日频股债LPR状态")
    export(origins, "全部计划复核原点与标签")
    predictions, models = walk(origins, cfg)
    save(OUT / "results/固定模型与月度训练权重.json", models)
    summary, pred, primary, monthly, updates, counts, indices = judge(origins, predictions, models, cfg)
    for frame, name in [(pred, "全部逐期预测与损失"), (primary, "完整评价月的全部预测"), (monthly, "主要月度损失"),
                        (updates, "额外LPR复核内容更新诊断"), (counts, "每月计划与准入状态"),
                        (pd.DataFrame(summary["metrics"]), "全部模型与简单基准"), (pd.DataFrame(summary["comparisons"]), "全部预定增量比较")]:
        export(frame, name)
    annual = monthly.assign(year=monthly.month.str[:4]).groupby("year", sort=True).agg(months=("month", "size"), **{key: ("loss_" + key, "mean") for key in FEATURES}).reset_index()
    for key, base in [("P_LPR", "P0"), ("P_JOINT", "P_LPR"), ("R_LPR", "R0"), ("R_JOINT", "R_LPR")]:
        annual[key + "_improvement"] = annual[base] - annual[key]
    export(annual, "逐年损失贡献")
    np.savez_compressed(OUT / "results/固定月度区块索引.npz", indices=indices.astype(np.int16))
    save(OUT / "results/summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"metrics", "comparisons"}}, ensure_ascii=False, indent=2), flush=True)
    print(pd.DataFrame(summary["metrics"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR与股债联合反应固定增量检验")
    parser.add_argument("action", choices=["freeze", "run"])
    {"freeze": freeze, "run": run}[parser.parse_args().action]()
