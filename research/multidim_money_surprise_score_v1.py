"""按月度公布事件联合观察M2事前预期偏差与指数整体状态。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

import multidim_nonlinear_score_v1 as common


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_money_surprise_score_v1"
STUDY = "510300_MULTIDIM_MONEY_SURPRISE_SCORE_V1"
SOURCE = ROOT / "reports/research/510300_money_consensus_increment_v2/inputs/104个月共识选择.csv"
DAILY = ROOT / "reports/research/510300_multidim_nonlinear_score_v1/historical_inputs_and_labels.parquet"
BASE = common.FEATURES
NEWS = "M2公布相对事前预期"
FEATURES = [*BASE, NEWS]
TREE = {"max_depth": 2, "min_samples_leaf": 6, "random_state": 20261001}
LABELS = {"state": "八项指数状态", "joint": "指数状态加M2预期偏差", "linear": "同九变量线性参考", "mean": "同池历史平均"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name: str, value) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(common.clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    if (OUT / "protocol.json").exists():
        raise RuntimeError("月度事件联合评分已固定，不覆盖。")
    save("protocol.json", {
        "study_id": STUDY, "frozen_at": common.now(),
        "previous_goal_turn_classification": "PROGRESS_MULTIDIM_POLICY_TRANSMISSION_SCORE",
        "user_priority": "多维一起看待、一起打分、快速看到结果、研究非线性关系",
        "hypothesis": "M2公布值相对同一期事前预期的偏差，在不同订单需求、资金价格、融资与指数量价状态下可能对应不同剩余收益；不预设正偏差必然利好。",
        "scope": "仅历史指数整体研究，510300为价格与可交易观察，执行资产仍只有510300和现金。",
        "event_universe": "原104个月货币公布台账；本轮截至2025年12月31日已公布且有合格预期的事件。缺失月份完整保留，不填零，不向后复制同一消息生成额外训练样本。",
        "features_state": BASE, "features_joint": FEATURES,
        "surprise_definition": "当月M2公布增速减去原冻结来源选择表中的事前预期，单位百分点。原固定来源优先序不变，称为报告中的事前预期代理，不宣称完整市场共识。",
        "definition_boundary": "不使用M1或M1-M2差，避免2025年M1定义变化直接进入本轮；M2原口径版本作为元数据保留。数字人民币纳入的官方说明指出M1、M2增速无明显变化，不据此补造任何预期。",
        "clock": "公布日期23:59:59为本次研究判断截止；指数状态使用该截止前最近一个交易日16点的已保存八变量。预期报告日期必须早于公布日期。买入在严格晚于公布日期的第一个交易日开盘；退出在入场后第5个交易日开盘。",
        "training": {"lookback_sessions": 504, "minimum_mature_events": 12, "update": "每次合格月度公布", "sample_unit": "每次公布一行", "label_availability": "退出开盘时点已不晚于当前判断截止"},
        "model": TREE, "complexity_reason": "月度样本不可与前轮每日样本等量计算；最多两层、每叶至少六次公布，为首次固定的事件交互表达。",
        "comparisons": ["相同事件、训练池和复杂度的八项指数状态树", "增加M2预期偏差的九变量树", "同九变量标准化岭回归alpha10", "同池成熟事件净收益平均"],
        "label": read(ROOT / "reports/research/510300_multidim_nonlinear_score_v1/protocol.json")["label"],
        "primary_period": ["2024-01-01", "2025-12-31"], "earlier_context": ["2021-01-01", "2023-12-31"],
        "score": "当日模型预测在同训练池拟合预测内的中位百分位秩，0至100，不是上涨概率。",
        "evaluation": "固定五档0至20、20至40、40至60、60至80、80至100，空档保留；同事件比较误差和全部高分机会。",
        "entry_diagnostic": "分数>=80且预测净收益>0；退出前不增加新事件，固定10000份成本标签，仅机会诊断，不冒充账户。",
        "full_account_admission": "若主要期高分机会平均净收益非正或没有机会，不进入账户；为正亦不自动达标，须另行完整账户核对。",
        "closest_old_study": "reports/research/510300_money_consensus_increment_v2/protocol.json",
        "substantive_difference": "旧研究用三项价格状态加M1-M2预期差的扩展窗岭回归。本轮是需求、资金、融资和量价八项状态与单独M2预期偏差的两层事件树，并作删除预期偏差的同模型对照。原旧口径失败、缺失和账户NOT_RUN保留。",
        "selection_history": "既有预期源、所有原宏观结果与两轮日频非线性失败已被看过；本轮是历史开发，不恢复独立性，不搜索新参数或择优组合旧模型。",
        "not_comparable_directly": "本轮按月度事件重新训练；误差不可直接和前两轮1212个日频评分的误差大小比较。",
        "source_receipts": [{"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)} for path in [SOURCE, DAILY]],
        "new_downloads_required": False, "minute_data": False, "orders_authorized": False,
        "new_prospective_forecasts_enabled": False, "goal_achieved": False,
    })
    print("已固定每次公布一条样本的九变量联合评分，保留同事件状态树对照。", flush=True)


def event_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_parquet(DAILY).sort_values("idx").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily.date)
    days = pd.DatetimeIndex(daily.date)
    source = pd.read_csv(SOURCE)
    source["known_at"] = pd.to_datetime(source.available_at_upper_bound, utc=True).dt.tz_convert("Asia/Shanghai")
    rows = []
    for item in source.to_dict("records"):
        day = item["known_at"].normalize()
        row = {key: item.get(key) for key in ["stat_month", "source_url", "published_at", "available_at_upper_bound", "definition_version", "forecast_date", "forecast_source", "forecast_url", "forecast_file", "forecast_sha256", "m2_yoy_pp", "expected_m2_pp", "consensus_admitted"]}
        row["known_at"] = item["known_at"]
        row["decision_at"] = day + pd.Timedelta(hours=23, minutes=59, seconds=59)
        row["status"] = "READY"
        if day.year > 2025:
            row["status"] = "AFTER_FIXED_PERIOD"
            rows.append(row)
            continue
        if not item["consensus_admitted"] or not np.isfinite(item.get("m2_surprise_pp", np.nan)):
            row["status"] = "MISSING_ADMITTED_EXPECTATION"
            rows.append(row)
            continue
        forecast_day = pd.Timestamp(item["forecast_date"]).tz_localize("Asia/Shanghai")
        if forecast_day >= day:
            raise ValueError("预期报告日期没有早于公布日期。")
        i = int(days.searchsorted(day.tz_localize(None), side="right")-1)
        if i < 0 or i+6 >= len(daily):
            row["status"] = "MISSING_PRICE_WINDOW"
            rows.append(row)
            continue
        state = daily.iloc[i]
        if not bool(state.valid_features):
            row["status"] = "MISSING_STATE_FEATURES"
            rows.append(row)
            continue
        if state.decision_time > row["decision_at"]:
            raise ValueError("指数状态使用了判断时点以后的数据。")
        for feature in BASE:
            row[feature] = float(state[feature])
        row[NEWS] = float(item["m2_yoy_pp"]-item["expected_m2_pp"])
        assert abs(row[NEWS]-item["m2_surprise_pp"]) < 1e-10
        row.update(
            state_idx=i, state_date=state.date, state_available_at=state.decision_time,
            entry_date=daily.date.iloc[i+1], exit_date=daily.date.iloc[i+6], exit_idx=i+6,
            actual_net5=float(state.net_label),
            funding_available_at=state.funding_available_at, orders_available_at=state.orders_available_at,
        )
        row["entry_at"] = row["entry_date"].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=30)
        row["exit_at"] = row["exit_date"].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=30)
        if row["entry_at"] <= row["decision_at"]:
            raise ValueError("入场没有晚于历史判断截止。")
        assert row["known_at"] <= row["decision_at"]
        rows.append(row)
    coverage = pd.DataFrame(rows)
    coverage.to_csv(OUT / "全部104月的本轮覆盖.csv", index=False, encoding="utf-8-sig")
    events = coverage[coverage.status.eq("READY")].sort_values("decision_at").reset_index(drop=True)
    events["era"] = np.select([events.decision_at.dt.year.ge(2024), events.decision_at.dt.year.ge(2021)], ["2024—2025", "2021—2023"], default="训练预热2018—2020")
    events.to_parquet(OUT / "月度事件联合状态及标签.parquet", index=False)
    return events, daily


def fit_tree(pool: pd.DataFrame, current: pd.Series, features: list[str]) -> tuple[float, float, dict]:
    x, y = pool[features].to_numpy(float), pool.actual_net5.to_numpy(float)
    model = DecisionTreeRegressor(**TREE).fit(x, y)
    prediction = float(model.predict(current[features].to_numpy(float).reshape(1, -1))[0])
    fitted = model.predict(x)
    score = float(100*((fitted < prediction).mean()+.5*(fitted == prediction).mean()))
    tree = model.tree_
    saved = {
        "features": features, "children_left": tree.children_left.tolist(), "children_right": tree.children_right.tolist(),
        "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(), "value": tree.value.ravel().tolist(),
        "samples": tree.n_node_samples.tolist(), "used_features": [features[j] for j in sorted(set(tree.feature[tree.feature>=0]))],
        "leaves": model.get_n_leaves(), "text": export_text(model, feature_names=features, decimals=5),
    }
    return prediction, score, saved


def return_summary(values: pd.Series) -> dict:
    values = values.dropna()
    positive, negative = values[values.gt(0)], values[values.lt(0)]
    return {"n": len(values), "mean_net5": float(values.mean()), "median_net5": float(values.median()),
            "win_rate": float(values.gt(0).mean()) if len(values) else np.nan,
            "return_payoff": float(positive.mean()/-negative.mean()) if len(positive) and len(negative) else np.nan}


def run() -> None:
    if (OUT / "result.json").exists() or (OUT / "逐事件联合评分.csv").exists():
        raise RuntimeError("已有事件评分结果，不重跑。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["source_receipts"]:
        assert sha(ROOT/source["path"]) == source["sha256"]
    events, daily = event_inputs()
    rows, fitted_models, excluded = [], [], []
    for _, current in events.iterrows():
        if current.decision_at.year < 2021:
            continue
        pool = events[events.state_idx.ge(current.state_idx-504) & events.exit_at.le(current.decision_at)]
        if len(pool) < 12:
            excluded.append({"stat_month": current.stat_month, "decision_at": current.decision_at, "mature_events": len(pool), "reason": "不足首次固定的12次成熟公布"})
            continue
        assert pool.stat_month.ne(current.stat_month).all()
        state_prediction, state_score, state_model = fit_tree(pool, current, BASE)
        joint_prediction, joint_score, joint_model = fit_tree(pool, current, FEATURES)
        x, y = pool[FEATURES].to_numpy(float), pool.actual_net5.to_numpy(float)
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
        scale[scale < 1e-12] = 1.
        z = np.clip((x-mean)/scale, -5, 5)
        zmean = z.mean(axis=0)
        centered = z-zmean
        coef = np.linalg.solve(centered.T@centered+10*np.eye(len(FEATURES)), centered.T@(y-y.mean()))
        linear = float(y.mean()+(np.clip((current[FEATURES].to_numpy(float)-mean)/scale, -5, 5)-zmean)@coef)
        fields = current.to_dict()
        fields.update(
            training_events=len(pool), training_months=";".join(pool.stat_month), training_max_exit_at=pool.exit_at.max(),
            state_prediction=state_prediction, state_score=state_score,
            joint_prediction=joint_prediction, joint_score=joint_score, linear_prediction=linear, mean_prediction=float(y.mean()),
            surprise_used=NEWS in joint_model["used_features"], joint_features_used="；".join(joint_model["used_features"]),
        )
        rows.append(fields)
        fitted_models.append({"stat_month": current.stat_month, "decision_at": current.decision_at,
                              "training_months": pool.stat_month.tolist(), "state_tree": state_model, "joint_tree": joint_model,
                              "linear": {"mean": mean.tolist(), "scale": scale.tolist(), "zmean": zmean.tolist(), "coef": coef.tolist(), "ymean": float(y.mean())}})
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("没有满足固定训练要求的事件。")
    predictions.to_csv(OUT / "逐事件联合评分.csv", index=False, encoding="utf-8-sig")
    save("saved_models.json", fitted_models)
    save("未评分的成熟事件.json", excluded)
    buckets, errors, opportunities = [], [], []
    for model in ["state", "joint"]:
        next_decision = pd.Timestamp("1900-01-01", tz="Asia/Shanghai")
        for _, row in predictions.iterrows():
            if row.decision_at < next_decision or row[f"{model}_score"] < 80 or row[f"{model}_prediction"] <= 0:
                continue
            opportunities.append({"model": model, "stat_month": row.stat_month, "era": row.era,
                                  "signal_date": row.decision_at, "entry_date": row.entry_date, "exit_date": row.exit_date,
                                  "score": row[f"{model}_score"], "prediction": row[f"{model}_prediction"], "net_return": row.actual_net5,
                                  "m2_surprise_pp": row[NEWS], "forecast_url": row.forecast_url})
            next_decision = row.exit_at
        for era, block in predictions.groupby("era", sort=False):
            band_values = np.minimum((block[f"{model}_score"]/20).astype(int), 4)
            for band in range(5):
                selected = block.loc[band_values.eq(band), "actual_net5"]
                buckets.append({"model": model, "era": era, "score_band": f"{20*band}—{20*(band+1)}", **return_summary(selected)})
    for era, block in predictions.groupby("era", sort=False):
        for model in LABELS:
            errors.append({"era": era, "model": model, "mse": float(((block[f"{model}_prediction"]-block.actual_net5)**2).mean()), "n": len(block)})
    event_columns = ["model", "stat_month", "era", "signal_date", "entry_date", "exit_date", "score", "prediction", "net_return", "m2_surprise_pp", "forecast_url"]
    opportunities_frame = pd.DataFrame(opportunities, columns=event_columns)
    summaries, annual = [], []
    for model in ["state", "joint"]:
        for era in ["2021—2023", "2024—2025"]:
            block = opportunities_frame[opportunities_frame.model.eq(model) & opportunities_frame.era.eq(era)]
            summaries.append({"model": model, "era": era, **return_summary(block.net_return)})
        for year in range(2021, 2026):
            selected = opportunities_frame[opportunities_frame.model.eq(model) & pd.to_datetime(opportunities_frame.entry_date).dt.year.eq(year)]
            annual.append({"model": model, "year": year, **return_summary(selected.net_return)})
    pd.DataFrame(buckets).to_csv(OUT / "同事件非线性分层比较.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(OUT / "四项模型误差.csv", index=False, encoding="utf-8-sig")
    opportunities_frame.to_csv(OUT / "全部固定高分机会.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT / "高分机会比较.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "高分机会逐年次数.csv", index=False, encoding="utf-8-sig")
    save("result.json", {
        "study_id": STUDY, "completed_at": common.now(), "status": "COMPLETED_MONTHLY_JOINT_SCORE_PENDING_INTERPRETATION",
        "classification": "PROGRESS_MULTIDIM_PRERELEASE_EXPECTATION_AND_INDEX_STATE",
        "admitted_events_before_cutoff": len(events), "scored_events": len(predictions), "unscored_mature_events": len(excluded),
        "earlier_scored_events": int(predictions.era.eq("2021—2023").sum()), "primary_scored_events": int(predictions.era.eq("2024—2025").sum()),
        "new_model_families": 3, "fits_per_family": len(predictions), "parameter_grids": 0,
        "surprise_used_in_tree_events": int(predictions.surprise_used.sum()),
        "primary_surprise_used_events": int(predictions.loc[predictions.era.eq("2024—2025"), "surprise_used"].sum()),
        "buckets": buckets, "model_errors": errors, "opportunity_summary": summaries, "annual_opportunities": annual,
        "new_accounts": 0, "account_net_sharpe": None, "goal_achieved": False, "orders_authorized": False,
        "current_view": "NO_VIEW", "position": "UNSET", "new_prospective_forecasts_enabled": False,
    })
    print("逐事件非线性联合评分已完成。", flush=True)
    print(pd.DataFrame(buckets).to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(errors).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2事前预期与指数整体状态的固定联合评分")
    parser.add_argument("stage", choices=["prepare", "run"])
    stage = parser.parse_args().stage
    {"prepare": prepare, "run": run}[stage]()
