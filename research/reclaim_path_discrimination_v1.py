"""冻结破低母集，以一项固定成分破低参与变化检验收复后的分叉。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / "reports/research/510300_reclaim_path_discrimination_v1"
MAIN = WORKSPACE / "reports/research/510300_integrated_research_continuation_20260924"
PARENT = WORKSPACE / "reports/research/510300_all_research_abcd_increment_v1"
START, END = "2021-01-04", "2026-08-14"
STUDY = "510300_RECLAIM_PATH_DISCRIMINATION_V1"
CONTROLS = ["setup_trend_z", "recovery_atr", "confirmation_age"]
SOURCES = {
    "features.parquet": PARENT / "inputs/features.parquet",
    "cross_section.parquet": PARENT / "inputs/cross_section.parquet",
    "attribution_quality.parquet": PARENT / "inputs/attribution_quality.parquet",
    "dividends.csv": PARENT / "inputs/dividends.csv",
    "signals.parquet": PARENT / "inputs/signals.parquet",
    "parent_event_labels.parquet": PARENT / "inputs/parent_event_labels.parquet",
    "parent_engine.py": PARENT / "code/parent_engine.py",
    "prior_abcd_protocol.json": PARENT / "protocol.json",
    "known_information.parquet": WORKSPACE / "reports/research/510300_integrated_macro_micro_prediction_v1/results/model_inputs.parquet",
    "mandate.json": WORKSPACE / "config/510300_existing_data_training_mandate_v1.json",
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def engine(root):
    spec = importlib.util.spec_from_file_location("reclaim_frozen_parent", root / "inputs/parent_engine.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def share_bounds(prices, lows, weights):
    valid = np.isfinite(prices) & np.isfinite(lows)
    denominator = float(weights.sum())
    coverage = float(weights[valid].sum() / denominator)
    lower = float(weights[valid & (prices < lows)].sum() / denominator)
    return lower, lower + 1 - coverage, coverage


def tests():
    weights = np.array([.6, .4])
    lows = np.array([10., 20.])
    first = share_bounds(np.array([9., 19.]), lows, weights)
    second = share_bounds(np.array([11., 19.]), lows, weights)
    assert np.allclose(first, [1, 1, 1]) and np.allclose(second, [.4, .4, 1])
    missing = share_bounds(np.array([11., np.nan]), lows, weights)
    assert np.allclose(missing, [0, .4, .6])
    # 一只股票自身拆分比例在同一观察窗内同时作用于现价和锚点，不改变低点判别。
    scaled = share_bounds(np.array([110., 1.9]), lows * np.array([10., .1]), weights)
    assert np.allclose(scaled, second)
    return ["固定低点与固定权重", "缺失贡献区间而非填零", "同口径价格尺度不影响判别"]


def freeze():
    if (ROOT / "freeze.json").exists():
        raise RuntimeError("本轮已经冻结，禁止覆盖。")
    checks = tests()
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "code").mkdir(exist_ok=True)
    for name, path in SOURCES.items():
        shutil.copy2(path, ROOT / "inputs" / name)
    shutil.copy2(Path(__file__), ROOT / "code/reclaim_path_discrimination_v1.py")
    protocol = {
        "study_id": STUDY, "frozen_at": now(),
        "user_direction": "在相同形态存在分叉的前提下，固定全部破低、收复和失败案例，只增加一项内部信息，使用真实确认后价格，并区分单笔失败与关系退化。",
        "scope": "事件分叉与选择后果开发检验，不新增交易家族，不恢复旧失败分支。",
        "primary_period": [START, END],
        "mother_population": "完整复用冻结parent.detect的RECLAIM事件：含分红收盘破此前20日最低价减0.1前日ATR，原三日观察、原收复/继续下跌/到期定义和五日冷却均不变；未完成保留。",
        "single_increment": "事件准备日固定全部成分及权重；每只股票的参照低点为事件之前20个交易日含分红收盘的最低值，必须有20日资料。确认时仍低于自身固定低点的权重占比，相对准备日减少多少。",
        "difference_from_abcd_B": "旧B是准备日正贡献最高至多三个行业的固定核心成员三日上涨参与变化；本轮是全体固定成员相对各自事前低点的破低参与变化，不用行业赢家、每日涨跌家数或原B收益筛选。",
        "difference_from_old_breadth": "原MACD/均线广度、滚动新低指标与本轮事件固定个股低点的变化不相同；原失败结果保持。",
        "coverage": "快照日期严格早于准备日；沿用既有归因质量有效日。准备与确认的固定权重覆盖均>=98%；缺失不重归一化，输出占比上下界。历史快照可用性沿用原重建假设，并未认证首次发布版本。",
        "entry_information_gate": "仅当占比收缩下界>0才判定收缩已确认；无覆盖=NO_VIEW，下界<=0则没有足够证据确认收缩。阈值不搜索。",
        "labels": "复用原确认后次日可成交开盘，原结构失效与最多五日退出，BASE/STRESS实际费用、整手和T+1；未成交不计为收益零，未结束保留。等待确认前的涨幅单独记为机会成本，不计已赚收益。",
        "main_selection_measure": "在相同覆盖的成熟确认事件中，收缩过滤相对保留全部事件的每事件净收益差；同时披露避开亏损及错过盈利。等额事件收益不是连续账户收益。",
        "conditional_prediction": {"baseline": CONTROLS, "increment": "contraction_point", "target": "原STRESS事件净收益", "minimum_prior_mature_events": 3,
            "minimum_is_exploration_only": True, "basis_for_minimum": "沿用当前V2已授权最少3条成熟历史的探索口径，不代表统计充分。",
            "training": "扩展历史，只用退出idx严格小于当前入场idx的共同合格样本；当日09:00尚未知的09:30退出不得训练。", "ridge": 10, "training_standardization": True, "clip": 5, "parameter_grid": 0},
        "uncertainty": "按确认时间排列，三个相邻事件循环块重抽2000次，种子202609245。区间只作重复使用历史上的开发描述；不是全项目多重试验校正。没有决策差异时不把零区间作为证据。",
        "development_gate": "主要压力选择增量95%下界>0，逐期MSE增量95%下界>0，且两个时间半段选择和预测方向均改善；至少6个成熟逐期预测。通过也仅PASS_DISCOVERY_ONLY。",
        "drift_monitor": "每次只记录此前最多12个已成熟的逐期预测误差、亏损深度和净成本补偿；最少3条仍仅观察。先前没有已验证优势，不能将一次失败或一个观察报警解释为规律永久失效。",
        "rr3_admission": "原RECLAIM没有冻结事前盈利目标，3:1资格为NOT_DEFINED_REWARD_TARGET。保留用户3:1门槛，不由模型预测或事后反弹补出目标。",
        "full_account_and_shutdown_comparison": "NOT_RUN_NO_FROZEN_REWARD_TARGET；本轮事件选择不冒充满足用户3:1的可执行账户，不额外设计启停/恢复参数。",
        "macro_funding_holiday": "只保留既有执行日09:00可知的宏观、资金和假期/计划事件/公开冲击背景，不新增过滤条件。",
        "independent_forward_events": 0, "new_market_collection": False, "orders_authorized": False,
        "review_package": False, "pre_return_mechanism_checks": checks,
        "literature": [
            {"url": "https://www.nber.org/papers/w7613", "use": "形态条件分布可检验；美国股票结果不能迁移为510300有效证据。"},
            {"url": "https://www.nber.org/papers/w27959", "use": "流动性供给可能承担意外信息和波动风险；本轮不声称已识别真实卖方身份。"},
            {"url": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf", "use": "保留反复试验与选择偏差；本轮不计算没有完整试验分布支持的DSR。"},
            {"url": "https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml", "use": "股票ETF T+1约束。"},
        ],
    }
    save(ROOT / "protocol.json", protocol)
    authority = {"recorded_at": now(), "user_requested_specific_local_research": True,
                 "prior_blocked_receipt_preserved": True, "fresh_blocked_audit_count": 0,
                 "goal_tool_status_at_start": "blocked", "goal_tool_resume_not_available_to_assistant": True,
                 "new_collection_authorized": False, "minimum_planned_net_reward_risk": 3,
                 "user_excerpt": "把所有已经按当时规则识别出的破低、收复和失败案例放在一起；只提出一项差异假设；减少了多少失败交易，又错过了多少成功交易。"}
    save(ROOT / "authority_update.json", authority)
    paths = [ROOT / "protocol.json", ROOT / "authority_update.json", ROOT / "code/reclaim_path_discrimination_v1.py", *(ROOT / "inputs").iterdir()]
    save(ROOT / "freeze.json", {"frozen_at": now(), "before_new_postconfirmation_results": True,
                                "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]})
    print("已冻结：全部破低母集、一项固定低点参与变量、固定逐期比较；未定义盈利目标的账户保持未运行。")


def prepare(root, cutoff=END):
    p = engine(root)
    d = pd.read_parquet(root / "inputs/features.parquet")
    d = d[d.date.le(cutoff)].reset_index(drop=True)
    episodes, steps, signals = p.detect(d)
    episodes = episodes[episodes.family.eq("RECLAIM")].copy()
    steps = steps[steps.family.eq("RECLAIM")].copy()
    signals = signals[signals.family.eq("RECLAIM")].copy()
    raw = pd.read_parquet(root / "inputs/cross_section.parquet")
    quality = pd.read_parquet(root / "inputs/attribution_quality.parquet")
    for f in [raw, quality]:
        f["date"] = pd.to_datetime(f.date).dt.strftime("%Y-%m-%d")
        f["weight_snapshot_date"] = pd.to_datetime(f.weight_snapshot_date).dt.strftime("%Y-%m-%d")
    raw = raw[raw.date.le(cutoff)].copy()
    raw.loc[~raw.price_available, "constituent_total_return_close"] = np.nan
    prices = raw.pivot(index="date", columns="con_code", values="constituent_total_return_close").reindex(d.date).reset_index(drop=True)
    byday = dict(tuple(raw.groupby("date")))
    quality["usable"] = quality.valid_for_attribution & (quality.weight_snapshot_date < quality.date)
    good = quality.set_index("date").usable.reindex(d.date).fillna(False).to_numpy(bool)
    rows, member_rows = [], []
    for event in episodes.to_dict("records"):
        t = int(event["setup_idx"])
        cohort = byday.get(event["setup_date"])
        usable = cohort is not None and good[t] and (cohort.weight_snapshot_date < event["setup_date"]).all()
        initial, lows, weights, values = None, None, None, None
        if usable:
            cohort = cohort.loc[cohort.snapshot_weight.gt(0)].copy()
            weights = cohort.snapshot_weight.to_numpy(float)
            values = prices.reindex(columns=cohort.con_code).to_numpy(float)
            before = values[t - 20:t]
            valid_history = np.isfinite(before).all(axis=0)
            lows = np.min(np.where(np.isfinite(before), before, np.inf), axis=0)
            lows[~valid_history] = np.nan
            initial = share_bounds(values[t], lows, weights)
            for member, low in zip(cohort.itertuples(), lows):
                member_rows.append({"episode_id": event["episode_id"], "con_code": member.con_code,
                                    "weight": member.snapshot_weight, "weight_snapshot_date": member.weight_snapshot_date,
                                    "fixed_prior_low20": low})
        for step in steps[steps.episode_id.eq(event["episode_id"])].to_dict("records"):
            i = int(step["idx"])
            current = share_bounds(values[i], lows, weights) if usable else (np.nan, np.nan, 0.)
            known = bool(usable and initial[2] >= .98 - 1e-12 and current[2] >= .98 - 1e-12 and good[i])
            delta = initial[0] - current[0] if usable else np.nan
            lower = initial[0] - current[1] if usable else np.nan
            upper = initial[1] - current[0] if usable else np.nan
            rows.append({**step, "setup_idx": t, "setup_date": event["setup_date"],
                         "setup_under_low_lower": initial[0] if usable else np.nan,
                         "setup_under_low_upper": initial[1] if usable else np.nan,
                         "under_low_lower": current[0], "under_low_upper": current[1], "coverage": current[2],
                         "contraction_point": delta if known else np.nan,
                         "contraction_lower": lower if known else np.nan, "contraction_upper": upper if known else np.nan,
                         "internal_known": known, "contraction_confirmed": bool(known and lower > 1e-12),
                         "view_status": "KNOWN" if known else "NO_VIEW_FIXED_MEMBER_COVERAGE_OR_CLOCK"})
    context = pd.DataFrame(rows)
    confirmed = context[context.action.eq("CONFIRMED_RECLAIM")].copy()
    sample = signals.merge(confirmed[["episode_id", "contraction_point", "contraction_lower", "contraction_upper", "coverage", "internal_known", "contraction_confirmed", "view_status"]], on="episode_id", validate="one_to_one")
    sample["setup_trend_z"] = d.trend_z.iloc[sample.setup_idx.astype(int)].to_numpy()
    sample["recovery_atr"] = (d.ac.iloc[sample.signal_idx.astype(int)].to_numpy() - d.ac.iloc[sample.setup_idx.astype(int)].to_numpy()) / sample.atr_index
    sample["confirmation_age"] = sample.signal_idx - sample.setup_idx
    sample["primary"] = sample.signal_date.between(START, END)
    sample["entry_idx"] = sample.signal_idx + 1
    return p, d, episodes, context, sample, pd.DataFrame(member_rows)


def interval(values, seed=202609245):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return {"n": len(a), "mean": float(a.mean()) if len(a) else None, "ci95": None, "status": "INSUFFICIENT_EVENTS"}
    if np.all(np.abs(a) < 1e-14):
        return {"n": len(a), "mean": 0., "ci95": None, "status": "NOT_IDENTIFIABLE_ZERO_DIFFERENCE"}
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(a), size=(2000, int(np.ceil(len(a) / 3))))
    indices = ((starts[:, :, None] + np.arange(3)) % len(a)).reshape(2000, -1)[:, :len(a)]
    samples = a[indices].mean(axis=1)
    return {"n": len(a), "mean": float(a.mean()), "ci95": np.quantile(samples, [.025, .975]), "status": "DEVELOPMENT_BLOCK_INTERVAL_ONLY"}


def fit_predict(train, row, fields):
    x = train[fields].to_numpy(float)
    y = train.net_return.to_numpy(float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    z = np.clip((x - mean) / scale, -5, 5)
    centered_mean = z.mean(axis=0)
    zc = z - centered_mean
    intercept = float(y.mean())
    beta = np.linalg.solve(zc.T @ zc + 10 * np.eye(len(fields)), zc.T @ (y - intercept))
    current = np.clip((row[fields].to_numpy(float) - mean) / scale, -5, 5) - centered_mean
    prediction = float(intercept + current @ beta)
    return prediction, {"fields": fields, "mean": mean, "scale": scale, "centered_mean": centered_mean, "beta": beta, "intercept": intercept}


def selection_summary(frame):
    keep = frame.contraction_confirmed.astype(bool)
    delta = np.where(keep, 0., -frame.net_return.to_numpy(float))
    half = len(frame) // 2
    return {"common_mature_events": len(frame), "kept": int(keep.sum()), "excluded": int((~keep).sum()),
            "kept_winners": int((keep & frame.net_return.gt(0)).sum()), "kept_losers": int((keep & frame.net_return.lt(0)).sum()),
            "avoided_losers": int((~keep & frame.net_return.lt(0)).sum()), "missed_winners": int((~keep & frame.net_return.gt(0)).sum()),
            "baseline_mean_net_event_return": float(frame.net_return.mean()) if len(frame) else None,
            "filtered_mean_per_original_event": float(np.where(keep, frame.net_return, 0.).mean()) if len(frame) else None,
            "paired_event_increment": interval(delta),
            "first_half_increment": float(delta[:half].mean()) if half else None,
            "second_half_increment": float(delta[half:].mean()) if len(frame) > half else None,
            "sum_of_event_returns_is_account_return": False}


def run():
    if (ROOT / "RUN_STARTED.json").exists():
        raise RuntimeError("本轮已经开始，不得改参数重跑。")
    for item in json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))["files"]:
        assert digest(ROOT / item["path"]) == item["sha256"], item["path"]
    assert digest(Path(__file__)) == digest(ROOT / "code/reclaim_path_discrimination_v1.py")
    save(ROOT / "RUN_STARTED.json", {"started_at": now()})
    p, d, episodes, context, sample, members = prepare(ROOT)
    original = pd.read_parquet(ROOT / "inputs/signals.parquet")
    original = original[original.family.eq("RECLAIM") & original.signal_date.le(END)].reset_index(drop=True)
    pd.testing.assert_frame_equal(sample[original.columns].reset_index(drop=True), original, check_dtype=False)
    dividends = pd.read_csv(ROOT / "inputs/dividends.csv")
    labels = pd.DataFrame([p.trade_outcome(d, sig, cost) for sig in sample.to_dict("records") for cost in p.COSTS])
    keep_columns = ["signal_id", *CONTROLS, "contraction_point", "contraction_lower", "contraction_upper", "coverage", "internal_known", "contraction_confirmed", "primary", "stop_index", "setup_idx"]
    event = labels.merge(sample[keep_columns], on="signal_id", validate="many_to_one")
    event["planned_reward_risk_status"] = "NOT_DEFINED_REWARD_TARGET"
    event["planned_stop_loss_cny"] = np.nan
    event["realized_R_at_original_full_quantity"] = np.nan
    event["confirmation_wait_return_not_earned"] = np.nan
    facts = pd.read_parquet(ROOT / "inputs/known_information.parquet")
    fact_fields = ["decision_time", "pmi_level", "pmi_change3", "pmi_known", "pmi_available_at", "funding_gap_pp", "funding_change5_pp", "funding_known", "dr_available_at", "flow5", "flow_breadth5", "flow_known", "flow_available_at", "preholiday3", "known_scheduled_event", "global_shock"]
    for field in fact_fields:
        event[field] = [facts[field].iloc[int(i)] if int(i) < len(facts) else None for i in event.entry_idx]
    for row in event.itertuples():
        e, s, t = int(row.entry_idx), int(row.signal_idx), int(row.setup_idx)
        if e >= len(d):
            continue
        event.loc[row.Index, "confirmation_wait_return_not_earned"] = float(d.ao.iloc[e] / d.ac.iloc[t] - 1)
        if row.label_status != "MATURE":
            continue
        stop_raw = float(row.stop_index * d.close.iloc[s] / d.ac.iloc[s] - d.dividend.iloc[e])
        stress_buy = p.fill_price(float(d.open.iloc[e]), "BUY", "STRESS")
        stress_stop = p.fill_price(stop_raw, "SELL", "STRESS")
        q = int(row.quantity)
        loss = q * (stress_buy - stress_stop) + p.commission(q * stress_buy, "STRESS") + p.commission(q * stress_stop, "STRESS")
        if loss > 0:
            pnl = float(row.net_return) * (q * float(row.entry_price) + p.commission(q * float(row.entry_price), row.cost))
            event.loc[row.Index, "planned_stop_loss_cny"] = loss
            event.loc[row.Index, "realized_R_at_original_full_quantity"] = pnl / loss
        assert int(row.exit_idx) > e
    # 新的特征不得改变原事件标签；终点后才成熟的旧标签不参加比较。
    saved = pd.read_parquet(ROOT / "inputs/parent_event_labels.parquet")
    same = saved[saved.family.eq("RECLAIM") & saved.exit_idx.le(len(d) - 1) & saved.label_status.eq("MATURE")]
    checked = 0
    for row in same.itertuples():
        current = event[event.signal_id.eq(row.signal_id) & event.cost.eq(row.cost)].iloc[0]
        assert current.exit_idx == row.exit_idx and current.exit_reason == row.exit_reason
        assert abs(current.net_return - row.net_return) < 1e-12
        checked += 1
    stress = event[event.cost.eq("STRESS")].sort_values("entry_idx").reset_index(drop=True)
    stress["model_known"] = stress.internal_known & stress[[*CONTROLS, "contraction_point"]].notna().all(axis=1)
    forecasts, fits, monitors = [], [], []
    for _, row in stress.iterrows():
        historical = stress[stress.model_known & stress.label_status.eq("MATURE") & stress.exit_idx.lt(row.entry_idx)]
        mature_forecasts = [f for f in forecasts if f["actual_exit_idx"] is not None and f["actual_exit_idx"] < row.entry_idx and f["prediction_M1"] is not None]
        tail = mature_forecasts[-12:]
        monitors.append({"signal_id": row.signal_id, "decision_entry_idx": row.entry_idx, "prior_scored_events": len(tail),
                         "mean_prediction_bias": np.mean([f["prediction_M1"] - f["actual_net_return"] for f in tail]) if tail else None,
                         "mean_realized_net_return": np.mean([f["actual_net_return"] for f in tail]) if tail else None,
                         "worst_realized_R": min((f["realized_R"] for f in tail if f["realized_R"] is not None), default=None),
                         "status": "OBSERVATION_ONLY_NO_VALIDATED_EDGE" if len(tail) >= 3 else "WARMUP_OBSERVATION_ONLY",
                         "risk_budget_switch_tested": False})
        record = {"signal_id": row.signal_id, "signal_date": row.signal_date, "entry_idx": int(row.entry_idx), "primary": bool(row.primary),
                  "model_known": bool(row.model_known), "prior_mature_events": len(historical),
                  "actual_exit_idx": int(row.exit_idx) if row.label_status == "MATURE" else None,
                  "actual_net_return": float(row.net_return) if row.label_status == "MATURE" else None,
                  "realized_R": float(row.realized_R_at_original_full_quantity) if pd.notna(row.realized_R_at_original_full_quantity) else None,
                  "prediction_M0": None, "prediction_M1": None,
                  "prediction_status": "NO_VIEW_FEATURE" if not row.model_known else "WARMUP_LESS_THAN_3_MATURE"}
        if row.model_known and len(historical) >= 3:
            for name, fields in [("M0", CONTROLS), ("M1", [*CONTROLS, "contraction_point"])]:
                prediction, model = fit_predict(historical, row, fields)
                record[f"prediction_{name}"] = prediction
                fits.append({"signal_id": row.signal_id, "model": name, "train_ids": historical.signal_id.tolist(),
                             "maximum_train_exit_idx": int(historical.exit_idx.max()), "decision_entry_idx": int(row.entry_idx), **model})
            record["prediction_status"] = "SCORED_PREQUENTIAL_EXPLORATORY"
        forecasts.append(record)
    predictions = pd.DataFrame(forecasts)
    scored = predictions[predictions.primary & predictions.prediction_M1.notna() & predictions.actual_net_return.notna()].copy()
    if len(scored):
        error0 = (scored.actual_net_return - scored.prediction_M0).to_numpy(float) ** 2
        error1 = (scored.actual_net_return - scored.prediction_M1).to_numpy(float) ** 2
        half = len(scored) // 2
        predictive = {"scored_events": len(scored), "M0_mse": float(error0.mean()), "M1_mse": float(error1.mean()),
                      "relative_mse_improvement": float(1 - error1.mean() / error0.mean()),
                      "paired_loss_improvement": interval(error0 - error1),
                      "first_half_improvement": float((error0 - error1)[:half].mean()) if half else None,
                      "second_half_improvement": float((error0 - error1)[half:].mean())}
    else:
        predictive = {"scored_events": 0, "status": "NOT_IDENTIFIABLE_NO_MATURE_PREQUENTIAL_PREDICTIONS"}
    selection = {}
    for cost in p.COSTS:
        frame = event[event.primary & event.cost.eq(cost) & event.internal_known & event.label_status.eq("MATURE")].sort_values("entry_idx")
        selection[cost] = selection_summary(frame)
    main = episodes[episodes.setup_date.between(START, END)]
    ss = selection["STRESS"]
    ci = ss["paired_event_increment"]["ci95"]
    pci = predictive.get("paired_loss_improvement", {}).get("ci95")
    passed = bool(ci is not None and ci[0] > 0 and pci is not None and pci[0] > 0 and predictive["scored_events"] >= 6
                  and ss["first_half_increment"] > 0 and ss["second_half_increment"] > 0
                  and predictive["first_half_improvement"] > 0 and predictive["second_half_improvement"] > 0)
    result = {"study_id": STUDY, "completed_at": now(), "status": "PASS_DISCOVERY_ONLY" if passed else "FROZEN_NO_RELIABLE_RECLAIM_INTERNAL_INCREMENT",
              "mother_primary_count": len(main), "mother_primary_statuses": main.status.value_counts().to_dict(),
              "confirmed_primary_count": int(sample.primary.sum()), "confirmed_primary_internal_known": int((sample.primary & sample.internal_known).sum()),
              "primary_label_statuses": event[event.primary & event.cost.eq("STRESS")].label_status.value_counts().to_dict(),
              "selection": selection, "prediction": predictive,
              "model_fits": len(fits), "continuous_account_runs": 0,
              "account_comparison": "NOT_RUN_NO_FROZEN_REWARD_TARGET",
              "rr3_status": "NOT_DEFINED_REWARD_TARGET", "minimum_planned_net_reward_risk": 3,
              "shutdown_recovery_account_comparison": "NOT_RUN_NO_VALIDATED_EDGE_AND_NO_FROZEN_REWARD_TARGET",
              "monitor_rows": len(monitors), "monitor_is_rule_decay_proof": False,
              "new_market_collection": False, "independent_forward_events": 0, "goal_achieved": False,
              "formal_current_market_view": "NO_VIEW", "orders_authorized": False}
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for name, frame in [("all_breakdown_episodes", episodes), ("all_breakdown_steps", context), ("fixed_members", members),
                        ("confirmation_facts", sample), ("confirmed_event_outcomes", event), ("prequential_predictions", predictions), ("mature_error_monitor", pd.DataFrame(monitors))]:
        frame.to_parquet(out / f"{name}.parquet", index=False)
    save(out / "saved_models.json", fits)
    save(ROOT / "result.json", result)
    prefix_checks = []
    for cutoff in ["2022-12-30", "2024-12-31"]:
        _, _, _, partial, _, _ = prepare(ROOT, cutoff)
        expected = context[context.date.le(cutoff)].reset_index(drop=True)
        pd.testing.assert_frame_equal(partial.reset_index(drop=True), expected)
        prefix_checks.append({"cutoff": cutoff, "matched_step_rows": len(partial)})
    for fit in fits:
        assert fit["maximum_train_exit_idx"] < fit["decision_entry_idx"]
    save(ROOT / "verification.json", {"status": "PASS_IMPLEMENTATION_NOT_STRATEGY_VALIDATION", "source_frozen_files_unchanged": all(digest(ROOT / f["path"]) == f["sha256"] for f in json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))["files"]),
                                        "parent_labels_matched": checked, "prefix_checks": prefix_checks,
                                        "strict_maturity_checks": len(fits), "mechanism_checks": tests()})
    write_report(result)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


def write_report(result):
    selected, prediction = result["selection"]["STRESS"], result["prediction"]
    ci = selected["paired_event_increment"]["ci95"]
    delta = selected["paired_event_increment"]["mean"]
    text = f"""本轮按用户新增的具体问题，检查相同破低收复形态的后续分叉。正式结果为{result['status']}。这是一项使用已被研究历史的开发检验，未建立高夏普策略。

主样本2021-01-04至2026-08-14保留{result['mother_primary_count']}次原定义破低事件，状态分别为{json.dumps(result['mother_primary_statuses'], ensure_ascii=False)}。其中{result['confirmed_primary_count']}次完成收复确认，{result['confirmed_primary_internal_known']}次具有合格内部信息。未确认、继续下跌、未成交和资料不足的事件都保留，没有按后来的反弹挑选图形。

唯一新增信息是固定成员的破低参与变化。事件开始时固定全部成分及权重，每只股票自己的低点使用事件前20个交易日含分红收盘最低值；之后检查仍在该低点下方的权重是否减少。成员、权重和低点均不随之后表现调整。旧A/B/C/D的B关注固定核心行业的三日上涨参与，二者的对象和参照不同；原失败裁决继续保留。

准备日与确认日的固定权重覆盖均须至少98%，缺失不填零、不重新归一化。收缩下界大于零才判为有收缩证据。宏观、资金、节日及计划事件仅保留当时已知的背景，没有叠加进过滤条件。官方权重历史可用性沿用先前重建假设，尚不构成来源首次发布认证。

压力成本下，相同资料覆盖且已成熟的{selected['common_mature_events']}个确认事件，保留{selected['kept']}个，排除{selected['excluded']}个。保留事件中盈利{selected['kept_winners']}个、亏损{selected['kept_losers']}个；排除中避开亏损{selected['avoided_losers']}个，同时错过盈利{selected['missed_winners']}个。计算从原确认结束后次日可成交开盘开始，沿用原结构失效和最多五日退出，包含T+1、费用与分红。

过滤相对保留全部合格事件的平均净收益差为{delta if delta is not None else '未识别'}，三个相邻事件循环块重抽的95%开发区间为{clean(ci)}。这里的每事件结果不能拼成连续账户利润或夏普，未成交也不作为零收益混入。若差异全为零，则正式区间保留未识别。

两种逐期预测使用相同历史样本。基础模型只用准备日趋势、收复幅度/ATR与确认耗时，增量模型仅加上述参与变化；扩展训练、岭惩罚10、训练期标准化及截断均固定。最少3条已成熟历史沿用当前V2探索口径，样本很少，不能视作充分验证。每次只使用当日09:00之前已经结束的事件。本轮共{result['model_fits']}次固定拟合，主样本{prediction['scored_events']}个成熟逐期预测；结果为{json.dumps(clean(prediction), ensure_ascii=False)}。

已另存仅使用此前成熟预测的滚动误差观察记录，包括最多12个既往预测的偏差、净成本补偿和最重R亏损。观察记录不能区分每一次正常误差与关系改变，也不构成已经验证的降权、暂停或恢复规则。

原破低收复规则没有冻结事前盈利目标，因此本轮3:1资格为NOT_DEFINED_REWARD_TARGET。用户至少3:1的要求保持，完整账户及启停机制比较为NOT_RUN_NO_FROZEN_REWARD_TARGET。本轮完成的是事件分叉和过滤后果检验，没有新增账户，也没有用预测值或事后反弹替代事前目标。下一阶段需要先证明具体信息增量，并补上有依据的盈利目标，再评价风险预算及启停对完整账户的作用。

已核对原事件标签、两次真实截断下的特征一致性，以及训练标签成熟时钟。所有母事件和失败案例、模型与逐期预测均保存在本目录，无审核ZIP、用户汇总表或新市场资料采集。

文献支持这些问题值得检验：Lo等比较了形态条件下的收益分布，但美国股票结果不证明510300有效；Drechsler等讨论流动性供给所承担的信息及波动风险。它们不能替代本轮的本地结果。来源：[形态研究](https://www.nber.org/papers/w7613)、[Liquidity and Volatility](https://www.nber.org/papers/w27959)。反复试验的选择偏差仍保留，[DSR论文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)并未被用来补算缺失的全项目试验分布。股票ETF的T+1依据见[上交所说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)。
"""
    (ROOT / "研究结论.md").write_text(text, encoding="utf-8")
    status_path = MAIN / "current_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"updated_at": now(), "latest_goal_turn_classification": "PROGRESS_USER_REQUESTED_RECLAIM_DISCRIMINATION",
                   "same_condition_consecutive_no_progress_goal_turns": 0, "goal_achieved": False,
                   "latest_user_requested_study": str(ROOT.relative_to(WORKSPACE)),
                   "latest_user_requested_study_status": result["status"], "latest_research_workflow": "USER_REQUESTED_LOCAL_DIAGNOSTIC_COMPLETED",
                   "remaining_research_question": "依据本轮唯一内部信息的实际结果决定是否保留该用途；完整账户仍需事前盈利目标及3:1资格，启停比较未运行。"})
    save(status_path, status)
    main_report = MAIN / "最新研究结论.md"
    prefix = (f"用户新增‘相似形态分叉’的具体研究指令后，已完成原破低母集的一项内部参与增量检验：{result['mother_primary_count']}个主样本事件，{result['confirmed_primary_count']}次收复确认。正式状态{result['status']}，高夏普目标仍未完成。原RECLAIM没有事前盈利目标，3:1资格及完整账户比较未建立。\n\n"
              f"详见[相似破低形态的分叉检验](<{(ROOT / '研究结论.md').as_posix()}>)。先前受阻记录保留为当时范围下的历史状态，本次明确新增的本地研究没有沿用旧连续阻断次数。以下保留既有结果。\n\n")
    main_report.write_text(prefix + main_report.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结与执行破低收复分叉的单信息开发检验")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.command == "freeze" else run()
