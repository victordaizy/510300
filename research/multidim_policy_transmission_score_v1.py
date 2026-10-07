"""用固定十变量模型检验政策利率与资金价格变化的历史联合评分增量。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_policy_transmission_score_v1"
PRIOR = ROOT / "reports/research/510300_multidim_nonlinear_score_v1"
RATES = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/operation_rate_records.parquet"
STUDY = "510300_MULTIDIM_POLICY_TRANSMISSION_SCORE_V1"
BASE_FEATURES = ["订单水平", "订单月度变化", "资金利率与政策利率差", "融资五日净变化", "融资买入活跃度", "指数趋势强度", "短长波动比", "成交活跃度"]
ADDED_FEATURES = ["已观察逆回购利率二十日变化", "资金利差二十日变化"]
FEATURES = BASE_FEATURES + ADDED_FEATURES
MODEL_ARGS = {"max_depth": 3, "min_samples_leaf": 60, "random_state": 20261001}
MODEL_COLUMNS = {
    "原八变量非线性": "base_nonlinear_prediction",
    "加入传导变化的十变量非线性": "nonlinear_prediction",
    "原八变量线性": "base_linear_prediction",
    "十变量线性": "linear_prediction",
    "同训练池平均": "mean_prediction",
}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(name: str, value) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    if (OUT / "protocol.json").exists():
        raise RuntimeError("该联合评分已固定，不覆盖设定。")
    prior_protocol = read(PRIOR / "protocol.json")
    assert all(prior_protocol["model"][key] == value for key, value in MODEL_ARGS.items())
    assert prior_protocol["features"] == BASE_FEATURES
    source_paths = [PRIOR / "historical_inputs_and_labels.parquet", PRIOR / "每日历史联合评分.csv", PRIOR / "result.json", PRIOR / "protocol.json", RATES]
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(),
        "user_priority": "我们可以多维一起看待，一起打分，我要快速看到结果，他们不可能是线性关系",
        "research_scope": "指数整体历史联合评分，仅510300和现金，不形成当前行情判断。",
        "hypothesis": "订单、融资与价格状态相同的情况下，最近利率动作及资金利差变化可能提供不同的收益分段条件；变化不预设为正或负，不能冒称原因已识别。",
        "features": FEATURES, "added_features": ADDED_FEATURES,
        "new_feature_definitions": {
            ADDED_FEATURES[0]: "截至每日16点在现有央行操作记录中已经观察到的最近7天逆回购利率，减去20个ETF交易日前同口径数值；单位百分点。",
            ADDED_FEATURES[1]: "原八变量中的已滞后一日资金利差减去20个ETF交易日前同口径利差；单位百分点。",
        },
        "economic_dimensions": ["订单需求状态", "已观察利率动作", "市场资金价格", "融资活动", "指数价格、波动及成交反映"],
        "source_clock_boundary": "按原操作记录published_at合并，naive时间按北京时间。2024年9月末现有记录首次看到1.5%是9月29日，对应9月30日评分；不倒填9月24日预告或9月27日实施。研究的是操作记录可观察变化，不是最早公告或政策意外。",
        "omitted_policy_series": "没有完整降准公告和事前政策共识序列，不把手选政策链节点加入模型，也不把缺失共识当成零。",
        "regime_caution": "2024年7月前后公开市场招标机制有变化，7天逆回购序列可观察但其政策角色并非全时期完全相同。",
        "model": MODEL_ARGS, "maximum_leaves": 8,
        "training": prior_protocol["training"], "label": prior_protocol["label"],
        "primary_period": prior_protocol["primary_period"], "earlier_context": prior_protocol["earlier_context"],
        "comparisons": "复用原八变量的保存预测；增加同复杂度十变量树和同alpha10的十变量岭回归，逐日训练池和标签必须相同。",
        "score": prior_protocol["score"], "evaluation": prior_protocol["evaluation"],
        "nonoverlap_opportunity": prior_protocol["nonoverlap_opportunity"],
        "diagnostic_context_table": "按事前已观察利率变化小于0/等于0/大于0、资金利差变化小于0/大于等于0固定交叉展示；不以收益改组，不作为第二套信号。",
        "anti_overfit": "本轮只有一个十变量树候选，不搜索窗口、深度、权重、门槛或年份；未把已知上涨事件编码为利好。此前已观察该段历史，本轮仍不构成独立验证。",
        "family_sequence": "这是看到八变量失败后的信息增量尝试；保留全家族尝试历史，不声称未被选择的独立实验。",
        "dedup": {"old_policy_quantity": "reports/research/510300_policy_liquidity_quantity_v1/result.json", "disposition": "旧16候选与账户失败原样保留；本轮不恢复该账户，仅对新八变量基线做固定信息增量比较。"},
        "sources": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in source_paths],
        "full_account_admission": "先报告固定高分机会的成本后收益及对原八变量的增量。若高分不重叠机会均值非正，直接不进入完整账户；正值亦不等于目标达成。",
        "new_full_accounts": 0, "orders_authorized": False, "new_prospective_forecasts_enabled": False,
        "current_view": "NO_VIEW", "position": "UNSET", "goal_achieved": False,
    })
    print("已固定十变量联合评分：仅增加利率动作与资金利差变化，不改变模型与门槛。", flush=True)


def inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(PRIOR / "historical_inputs_and_labels.parquet").sort_values("idx").reset_index(drop=True)
    baseline = pd.read_csv(PRIOR / "每日历史联合评分.csv")
    baseline["date"] = pd.to_datetime(baseline.date)
    rates = pd.read_parquet(RATES).sort_values("published_at").reset_index(drop=True)
    rates["rate_available_at"] = pd.to_datetime(rates.published_at).dt.tz_localize("Asia/Shanghai")
    selected = rates[["rate_available_at", "seven_day_rate_percent", "source_url", "raw_path"]].rename(columns={"source_url": "rate_source_url", "raw_path": "rate_raw_path"})
    d = pd.merge_asof(d.sort_values("decision_time"), selected.sort_values("rate_available_at"), left_on="decision_time", right_on="rate_available_at", direction="backward")
    d[ADDED_FEATURES[0]] = (d.seven_day_rate_percent - d.seven_day_rate_percent.shift(20)).round(10)
    # 20日前缺观测时，只沿用截至该日最后已知值，并单列陈旧天数。
    gap_known_idx = d.idx.where(d[BASE_FEATURES[2]].notna()).ffill()
    d["gap_lag20_additional_stale_sessions"] = d.idx-20-gap_known_idx.shift(20)
    d[ADDED_FEATURES[1]] = d[BASE_FEATURES[2]] - d[BASE_FEATURES[2]].ffill().shift(20)
    d["valid_ten_features"] = np.isfinite(d[FEATURES].to_numpy(float)).all(axis=1)
    d["observed_rate_regime"] = np.select([d[ADDED_FEATURES[0]].lt(0), d[ADDED_FEATURES[0]].gt(0)], ["观察到降息", "观察到升息"], default="未观察到利率变化")
    d["funding_gap_regime"] = np.where(d[ADDED_FEATURES[1]].lt(0), "资金利差收窄", "资金利差未收窄")
    valid = d.valid_ten_features
    assert (d.loc[valid, "rate_available_at"] <= d.loc[valid, "decision_time"]).all()
    assert d.idx.eq(np.arange(len(d))).all()
    assert d.loc[d.idx.isin(baseline.idx), "valid_ten_features"].all()
    d.to_parquet(OUT / "十变量历史输入及原标签.parquet", index=False)
    rates.to_csv(OUT / "沿用的利率变化记录.csv", index=False, encoding="utf-8-sig")
    return d, baseline


def summarize_returns(values: pd.Series) -> dict:
    values = values.dropna()
    wins, losses = values[values.gt(0)], values[values.lt(0)]
    return {
        "n": len(values), "mean_net5": float(values.mean()), "median_net5": float(values.median()),
        "win_rate": float(values.gt(0).mean()) if len(values) else np.nan,
        "return_payoff": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
    }


def run() -> None:
    if (OUT / "result.json").exists() or (OUT / "十变量每日联合评分.csv").exists():
        raise RuntimeError("已有评分结果，不重跑或调参。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("固定后源文件发生变化。")
    d, baseline = inputs()
    rows, fits, explanations = [], [], []
    for old in baseline.itertuples(index=False):
        i = int(old.idx)
        pool = d[d.idx.ge(i-504) & d.exit_idx.le(i) & d.valid_ten_features & d.net_label.notna()]
        assert len(pool) == old.n_train
        assert int(pool.exit_idx.max()) == old.max_label_exit_idx
        x, y = pool[FEATURES].to_numpy(float), pool.net_label.to_numpy(float)
        current = d.loc[i, FEATURES].to_numpy(float).reshape(1, -1)
        model = DecisionTreeRegressor(**MODEL_ARGS).fit(x, y)
        fitted, prediction = model.predict(x), float(model.predict(current)[0])
        score = float(100 * ((fitted < prediction).mean() + .5 * (fitted == prediction).mean()))
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
        scale[scale < 1e-12] = 1.
        z = np.clip((x-mean)/scale, -5, 5)
        zmean = z.mean(axis=0)
        centered = z-zmean
        coefficients = np.linalg.solve(centered.T @ centered + 10*np.eye(len(FEATURES)), centered.T @ (y-y.mean()))
        linear = float(y.mean() + (np.clip((current[0]-mean)/scale, -5, 5)-zmean) @ coefficients)
        assert abs(float(y.mean())-old.mean_prediction) < 1e-12
        assert abs(float(d.net_label.iloc[i])-old.actual_net5) < 1e-12
        tree = model.tree_
        path_nodes = model.decision_path(current).indices.tolist()
        path_features = [FEATURES[int(tree.feature[node])] for node in path_nodes if tree.feature[node] >= 0]
        used_features = [FEATURES[j] for j in sorted(set(tree.feature[tree.feature >= 0]))]
        rows.append({
            "idx": i, "date": old.date, "era": old.era, "n_train": len(pool), "max_label_exit_idx": int(pool.exit_idx.max()),
            "score": score, "score_bin": min(int(score//20), 4), "nonlinear_prediction": prediction, "linear_prediction": linear,
            "base_score": old.score, "base_nonlinear_prediction": old.nonlinear_prediction, "base_linear_prediction": old.linear_prediction,
            "mean_prediction": old.mean_prediction, "actual_net5": old.actual_net5,
            "rate_change20": float(d.loc[i, ADDED_FEATURES[0]]), "funding_gap_change20": float(d.loc[i, ADDED_FEATURES[1]]),
            "observed_rate_regime": d.loc[i, "observed_rate_regime"], "funding_gap_regime": d.loc[i, "funding_gap_regime"],
            "new_feature_used_in_tree": any(name in used_features for name in ADDED_FEATURES),
            "rate_change_used_in_tree": ADDED_FEATURES[0] in used_features,
            "funding_change_used_in_tree": ADDED_FEATURES[1] in used_features,
            "path_features": "；".join(path_features), "historical_only": True,
        })
        fits.append({
            "date": old.date, "features_used": used_features, "leaves": model.get_n_leaves(), "n_train": len(pool),
            "training_label_max_idx": int(pool.exit_idx.max()),
            "tree_children_left": tree.children_left.tolist(), "tree_children_right": tree.children_right.tolist(),
            "tree_feature": tree.feature.tolist(), "tree_threshold": tree.threshold.tolist(), "tree_value": tree.value.ravel().tolist(),
            "tree_samples": tree.n_node_samples.tolist(), "linear_coefficients": coefficients.tolist(),
            "linear_mean": mean.tolist(), "linear_scale": scale.tolist(), "linear_zmean": zmean.tolist(), "linear_target_mean": float(y.mean()),
        })
        if old.date.strftime("%Y-%m-%d") in {"2023-12-29", "2024-12-31", "2025-12-31"}:
            explanations.append({"date": old.date, "score": score, "expected_net5": prediction, "input_values": d.loc[i, FEATURES].to_dict(), "rule": export_text(model, feature_names=FEATURES, decimals=5)})
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "十变量每日联合评分.csv", index=False, encoding="utf-8-sig")
    save("saved_models.json", fits)
    save("固定年末评分说明.json", explanations)
    buckets, errors, events, contexts = [], [], [], []
    for family, score_column, prediction_column in [
        ("原八变量非线性", "base_score", "base_nonlinear_prediction"),
        ("加入传导变化的十变量非线性", "score", "nonlinear_prediction"),
    ]:
        next_decision = -1
        for row in p.itertuples(index=False):
            if row.idx < next_decision or getattr(row, score_column) < 80 or getattr(row, prediction_column) <= 0:
                continue
            events.append({
                "model": family, "era": row.era, "signal_idx": row.idx, "signal_date": row.date,
                "entry_date": d.date.iloc[row.idx+1], "exit_date": d.date.iloc[row.idx+6],
                "score": getattr(row, score_column), "prediction": getattr(row, prediction_column),
                "net_event_return": row.actual_net5, "rate_change20": row.rate_change20, "funding_gap_change20": row.funding_gap_change20,
            })
            next_decision = row.idx+6
        for era, block in p.groupby("era", sort=False):
            for band in range(5):
                mask = np.minimum((block[score_column]/20).astype(int), 4).eq(band)
                buckets.append({"model": family, "era": era, "score_band": f"{20*band}—{20*(band+1)}", **summarize_returns(block.loc[mask, "actual_net5"])})
    for era, block in p.groupby("era", sort=False):
        for model, column in MODEL_COLUMNS.items():
            errors.append({"era": era, "model": model, "mse": float(np.mean((block[column]-block.actual_net5)**2)), "days": len(block)})
        for rate_state in ["观察到降息", "未观察到利率变化", "观察到升息"]:
            for funding_state in ["资金利差收窄", "资金利差未收窄"]:
                block_state = block[block.observed_rate_regime.eq(rate_state) & block.funding_gap_regime.eq(funding_state)]
                contexts.append({"era": era, "rate_state": rate_state, "funding_state": funding_state, **summarize_returns(block_state.actual_net5)})
    bucket_frame, error_frame, event_frame = pd.DataFrame(buckets), pd.DataFrame(errors), pd.DataFrame(events)
    bucket_frame.to_csv(OUT / "八变量与十变量分层比较.csv", index=False, encoding="utf-8-sig")
    error_frame.to_csv(OUT / "五项基准误差.csv", index=False, encoding="utf-8-sig")
    event_frame.to_csv(OUT / "固定高分不重叠机会.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(contexts).to_csv(OUT / "政策利率与资金价格固定交叉.csv", index=False, encoding="utf-8-sig")
    event_summaries, annual = [], []
    for (model, era), block in event_frame.groupby(["model", "era"], sort=False):
        event_summaries.append({"model": model, "era": era, **summarize_returns(block.net_event_return)})
    for model in ["原八变量非线性", "加入传导变化的十变量非线性"]:
        for year in range(2021, 2026):
            block = event_frame[event_frame.model.eq(model) & event_frame.entry_date.dt.year.eq(year)]
            annual.append({"model": model, "year": year, **summarize_returns(block.net_event_return)})
    pd.DataFrame(event_summaries).to_csv(OUT / "不重叠机会比较.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "逐年自然完成机会.csv", index=False, encoding="utf-8-sig")
    save("result.json", {
        "study_id": STUDY, "completed_at": now(), "status": "COMPLETED_FIXED_TEN_FEATURE_SCORE_PENDING_INTERPRETATION",
        "classification": "PROGRESS_MULTIDIM_POLICY_TRANSMISSION_SCORE", "score_days": len(p), "features": FEATURES,
        "new_model_families": 2, "new_fits_per_family": len(p), "parameter_grids": 0,
        "buckets": buckets, "errors": errors, "events": event_summaries, "annual": annual,
        "tree_uses_new_feature_days": int(p.new_feature_used_in_tree.sum()),
        "tree_uses_rate_change_days": int(p.rate_change_used_in_tree.sum()),
        "tree_uses_funding_change_days": int(p.funding_change_used_in_tree.sum()),
        "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False,
        "current_view": "NO_VIEW", "position": "UNSET", "orders_authorized": False, "new_prospective_forecasts_enabled": False,
    })
    print("十变量非线性评分及固定比较已完成。", flush=True)
    print(bucket_frame.to_string(index=False))
    print(pd.DataFrame(event_summaries).to_string(index=False))
    print(error_frame.to_string(index=False))


def verify() -> None:
    p = pd.read_csv(OUT / "十变量每日联合评分.csv")
    baseline = pd.read_csv(PRIOR / "每日历史联合评分.csv")
    d = pd.read_parquet(OUT / "十变量历史输入及原标签.parquet")
    models = read(OUT / "saved_models.json")
    event_frame = pd.read_csv(OUT / "固定高分不重叠机会.csv")
    prior_events = pd.read_csv(PRIOR / "固定高分不重叠机会.csv")
    assert p.idx.tolist() == baseline.idx.tolist()
    np.testing.assert_allclose(p.actual_net5, baseline.actual_net5, rtol=0, atol=1e-12)
    assert (p.max_label_exit_idx <= p.idx).all()
    prediction_errors = []
    for i, model in enumerate(models):
        values = d.loc[int(p.idx.iloc[i]), FEATURES].to_numpy(float)
        node = 0
        while model["tree_children_left"][node] != -1:
            feature = model["tree_feature"][node]
            node = model["tree_children_left"][node] if float(np.float32(values[feature])) <= model["tree_threshold"][node] else model["tree_children_right"][node]
        prediction_errors.append(abs(model["tree_value"][node]-p.nonlinear_prediction.iloc[i]))
    assert max(prediction_errors) < 1e-12
    for model, frame in event_frame.groupby("model", sort=False):
        assert (pd.to_datetime(frame.entry_date).iloc[1:].reset_index(drop=True) > pd.to_datetime(frame.exit_date).iloc[:-1].reset_index(drop=True)).all()
    old = event_frame[event_frame.model.eq("原八变量非线性")].reset_index(drop=True)
    assert pd.to_datetime(old.signal_date).tolist() == pd.to_datetime(prior_events.signal_date).tolist()
    np.testing.assert_allclose(old.net_event_return, prior_events.net_event_return, rtol=0, atol=1e-12)
    buckets = pd.read_csv(OUT / "八变量与十变量分层比较.csv")
    assert buckets.groupby("model").n.sum().eq(len(p)).all()
    save("必要计算核对.json", {
        "checked_at": now(), "same_dates_labels_and_training_pools": True,
        "mature_labels_only": True, "saved_tree_predictions_recomputed": len(prediction_errors),
        "maximum_saved_prediction_error": max(prediction_errors), "old_fixed_opportunities_unchanged": len(old),
        "score_bin_observation_counts_match": True, "no_overlap_within_each_opportunity_series": True,
        "not_proven": ["独立验证", "因果关系", "完整账户夏普"],
    })
    print("已完成必要核对：日期、成熟标签、保存树预测、原机会一致及机会不重叠。", flush=True)


def finish() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    if not (OUT / "必要计算核对.json").exists():
        raise RuntimeError("尚未完成保存结果与时钟的必要核对。")
    result = read(OUT / "result.json")
    if result["status"] == "COMPLETED_FIXED_TEN_FEATURE_SCORE_NO_PRIMARY_INCREMENT":
        raise RuntimeError("该研究已经交付，不覆盖报告。")
    scores = pd.read_csv(OUT / "十变量每日联合评分.csv")
    buckets = pd.read_csv(OUT / "八变量与十变量分层比较.csv")
    errors = pd.read_csv(OUT / "五项基准误差.csv").pivot(index="era", columns="model", values="mse")
    opportunities = pd.read_csv(OUT / "不重叠机会比较.csv")
    coverage = pd.read_csv(OUT / "政策变化样本覆盖.csv")
    old_model, new_model = "原八变量非线性", "加入传导变化的十变量非线性"
    recent = opportunities[opportunities.model.eq(new_model) & opportunities.era.eq("2024—2025")].iloc[0]
    if recent.mean_net5 > 0:
        raise RuntimeError("固定高分机会收益为正，需要另行判断账户检验，不能套用失败报告。")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": FontProperties(fname=str(font_path)).get_name(), "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
    bands = [f"{20*i}—{20*(i+1)}" for i in range(5)]
    colors = ["#aab4c0", "#197d8c"]
    for ax, era in zip(axes, ["2021—2023", "2024—2025"]):
        for j, model in enumerate([old_model, new_model]):
            table = buckets[buckets.model.eq(model) & buckets.era.eq(era)].set_index("score_band").loc[bands]
            bars = ax.bar(np.arange(5)+(j-.5)*.34, table.mean_net5*100, width=.34, color=colors[j], label="原八变量" if j == 0 else "加入传导变化的十变量")
            if j == 1:
                for bar in bars:
                    value = bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2, value+(.035 if value >= 0 else -.035), f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=10)
        ax.axhline(0, color="#637080", linewidth=.9)
        ax.set_xticks(np.arange(5), bands)
        ax.set_title(era + ("  较早历史" if era == "2021—2023" else "  主要比较期"), loc="left", fontsize=13, pad=12)
        ax.set_xlabel("当日训练池内的相对分数")
        ax.set_ylim(-1.4, 1.6)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("随后五日平均净收益（已计压力费用与滑点）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.965,.975), frameon=False, fontsize=11)
    fig.suptitle("新增利率传导信息后，高分组仍未改善", x=.06, ha="left", fontsize=17, y=.97)
    fig.text(.06, .085, "分数是模型对净收益的相对排序，80分不代表80%胜率。五日窗口有重叠，观察日不能当作独立交易。", fontsize=10.5, color="#485563")
    fig.text(.06, .035, "2024—2025按固定条件去掉重叠：两套模型均为10次高分机会，平均净收益−0.81%，胜率40%。", fontsize=11, color="#254955")
    fig.tight_layout(rect=(.02,.13,.99,.88))
    figure_name = "多维联合评分_政策传导增量.png"
    fig.savefig(OUT / figure_name, dpi=160, facecolor="white")
    plt.close(fig)

    comparison_rows = []
    for era in ["2021—2023", "2024—2025"]:
        for model in [old_model, new_model]:
            row = opportunities[opportunities.model.eq(model) & opportunities.era.eq(era)].iloc[0]
            comparison_rows.append(f"| {era} | {'八变量' if model == old_model else '十变量'} | {int(row.n)} | {row.mean_net5:+.2%} | {row.win_rate:.2%} | {row.return_payoff:.2f} |")
    ranking_rows = []
    for band in bands:
        row8 = buckets[buckets.model.eq(old_model) & buckets.era.eq("2024—2025") & buckets.score_band.eq(band)].iloc[0]
        row10 = buckets[buckets.model.eq(new_model) & buckets.era.eq("2024—2025") & buckets.score_band.eq(band)].iloc[0]
        ranking_rows.append(f"| {band} | {row8.mean_net5:+.2%} | {row10.mean_net5:+.2%} | {row10.win_rate:.2%} | {int(row10.n)} |")
    increment = errors.loc["2024—2025", new_model]/errors.loc["2024—2025", old_model]-1
    versus_linear = errors.loc["2024—2025", new_model]/errors.loc["2024—2025", "十变量线性"]-1
    recent_use = int(scores.loc[scores.era.eq("2024—2025"), "funding_change_used_in_tree"].sum())
    earlier_use = int(scores.loc[scores.era.eq("2021—2023"), "funding_change_used_in_tree"].sum())
    allowed_primary = int(coverage.loc[coverage.era.eq("2024—2025"), "eligible_root_policy_split"].sum())
    recent_annual = [item for item in result["annual"] if item["model"] == new_model]
    count_text = "、".join(f"{item['year']}年{item['n']}次" for item in recent_annual)
    def link(name: str, label: str) -> str:
        return f"[{label}](<{(OUT / name).as_posix()}>)"
    report_name = "多维联合评分_政策传导增量结果.md"
    report = f"""# 510300 多维联合评分的政策传导增量结果

**已经完成十变量联合非线性评分。本轮新增的利率变化和资金利差变化，没有改善主要历史阶段的高分机会。夏普1.2尚未实现。** 2024—2025年的80分以上组，五日平均净收益仍为−0.90%；按原先固定条件去掉重叠后，10次机会平均净收益−0.81%、胜率40%。因此本轮没有继续运行完整账户。

研究对象是沪深300整体的状态传导，510300用于可交易价格观察。每个历史日同时看订单需求、已观察利率动作、实际资金价格、融资活动、指数价格与成交。分数由最多三层条件分支形成，一个因素的作用可以随另一个因素的状态而改变，没有把各项简单等权相加，也没有事先强制“降息必然加分”。非线性只是一种表示关系的方法，是否有效仍由后续可成交收益判断。

原八项为订单水平、订单月度变化、资金利差、融资五日变化、融资买入活跃度、指数趋势、短长波动比和成交活跃度。本轮只新增最近20个交易日的已观察7天逆回购利率变化、资金利差变化；仍使用同一1212个历史评分日、相同训练样本和五日收益标签。深度3、每叶至少60个历史观察、504交易日训练窗以及80分门槛均未改变。本轮拟合一套十变量树，并用同变量线性模型作参考；没有搜索配置。

![政策传导信息加入前后的五档收益](<{(OUT / figure_name).as_posix()}>)

**主要阶段2024—2025的分层结果**如下。净收益按信号后次日开盘买入、入场后第五个交易日开盘退出计算，佣金单边万四且最低5元、滑点单边千一，计入分红权益。计算统一使用10000份作为成本比较数量，并不代表20万元完整账户的仓位。

| 相对分数 | 原八变量五日均值 | 十变量五日均值 | 十变量胜率 | 十变量观察日数 |
|---|---:|---:|---:|---:|
{chr(10).join(ranking_rows)}

80分只是训练池拟合收益的相对排名，不是80%上涨概率。分层收益没有随分数上升而变好；20至40分组较好已经是事后可见结果，不能把它直接倒置成买入条件。观察日的五日持有窗口存在重叠。

**去掉重叠后，结果仍然没有支持高分组。** 规则沿用“分数至少80且模型预测净收益为正”；一笔持有结束后才允许下一笔判断，没有依据本轮结果修改门槛。

| 历史阶段 | 模型 | 自然完成机会 | 平均单次净收益 | 胜率 | 正收益均值与负收益绝对均值之比 |
|---|---|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

十变量机会逐年为{count_text}。2024年只有两次，尚不符合每个完整自然年至少五次的最终策略目标。这些固定数量事件也不能直接计算或替代全账户夏普。

新增变量的实际使用情况解释了结果的边界。资金利差变化进入了{earlier_use+recent_use}天的模型，其中较早阶段{earlier_use}天、主要阶段{recent_use}天；7天逆回购利率变化没有进入任何一天的树。主要阶段全部{allowed_primary}个训练窗口在根节点都有足够的状态日满足60行切分下限，故不能简单把未使用归因于全都不够样本。也不能把它解释为政策没有作用：相邻政策状态日源于少数政策事件，模型只衡量给定输入、给定训练池中的拟合改善。

2024—2025十变量非线性模型的均方误差较原八变量非线性模型增加{increment:.2%}，较同十变量线性参考高{versus_linear:.2%}。这是本轮的数值结果，不支持“只要改用非线性就能获得交易优势”。2021—2023不重叠机会均值虽由+0.03%变为+0.06%，仍不足以抵消主要阶段的失败或证明可用策略。

**政策动作、市场反应和预期差的口径必须区别清楚。** 本轮记录的是现有操作公告序列已经显示的利率变化，并没有完整的事前市场共识，不能称为“超预期降息”。已有序列在2024年9月29日才首次记录1.5%，因此9月30日评分才收到该变化；实际降息从9月27日生效。该数据不代表最早政策预告时点，本轮也没有给9月24日单独补一个利好标签。[央行9月27日公告的政府网转载](https://app.www.gov.cn/govdata/gov/202409/27/519932/article.html)

序列的经济含义也有时期区别：2024年7月22日起7天逆回购改为固定利率、数量招标，因此本轮统一称“已观察操作利率变化”，不声称它在此前所有时期都代表完全同样的政策信息。[央行7月22日公告](https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125469/5409998/index.html)

历史训练中的2020年3月有五行，恰好20个交易日前的资金观测缺失；采用截至该历史日最后可见值，并记录额外1至5日陈旧度。这一处理在首次拟合前确定，比较期1212个评分日本身没有这个陈旧问题。所有模型只使用退出日已经到来的收益标签；必要核对已重算1212个保存树的预测，并确认原八变量的36个固定机会不变、每套事件不重叠。历史已被研究过，不能称为独立验证。

本轮没有支持“把这两项变化加入当前低复杂度联合评分就足以改善高分机会”的假设。它不能排除其他非线性关系或证明任何宏观因果链。后续若继续研究，应寻找具备一致时点和事前预期口径的新信息，再与指数状态联合比较；当前十变量模型保留为失败结果，不加深树、不倒置分组、不调整门槛救回。

结果与来源入口：{link('十变量每日联合评分.csv','每日联合评分')}、{link('固定高分不重叠机会.csv','全部固定机会')}、{link('政策利率与资金价格固定交叉.csv','政策与资金状态交叉')}、{link('protocol.json','固定设定')}、{link('必要计算核对.json','必要计算核对')}。本轮不提供当前市场观点，不生成交易指令。
"""
    (OUT / report_name).write_text(report, encoding="utf-8")
    result.update(
        status="COMPLETED_FIXED_TEN_FEATURE_SCORE_NO_PRIMARY_INCREMENT",
        finalized_at=now(), report=report_name, figure=figure_name,
        primary_mse_relative_change_vs_base=increment,
        primary_mse_relative_change_vs_same_features_linear=versus_linear,
        primary_opportunity_status="NEGATIVE_NET_EXPECTANCY_UNCHANGED",
        full_account_status="NOT_RUN_NEGATIVE_FIXED_HIGH_SCORE_OPPORTUNITY_MEAN",
        source_clock_limitation="操作记录的已观察变化并非政策预告或最早实施时点，没有完整事前政策共识。",
        old_frozen_failure_preserved=True,
    )
    save("result.json", result)
    summary = "完成1212个历史日的十变量联合评分。新增利率动作及资金利差变化后，2024—2025高分组仍为-0.90%，10个固定不重叠机会仍平均-0.81%、胜率40%；主要期均方误差增加0.44%。资金变化进入165天模型，利率变化未被树采用；本轮无完整账户或达标夏普。"
    relative_out = str(OUT.relative_to(ROOT)).replace("\\", "/")
    cause_path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    cause = read(cause_path)
    cause.update(
        latest_completed_study=relative_out+"/result.json", latest_report=relative_out+"/"+report_name,
        current_study=relative_out+"/protocol.json", latest_result_summary=summary, updated_at=now(),
        latest_completed_branch_boundary="固定十变量联合评分没有改善主要期高分机会；不改深度、门槛或倒置有利分组。未采用利率变化不能解释为政策无效。",
        next_historical_question="先去重并确认已有统一事前预期口径，再将信息公布的偏差与指数状态联合比较；避免继续逐个追加衍生变量或修补已知行情。",
    )
    cause_path.write_text(json.dumps(cause, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(
        current_round=STUDY, latest_progress_receipt=relative_out+"/result.json", last_research_result=summary,
        latest_continuation_report=relative_out+"/"+report_name,
        latest_continuation_classification="PROGRESS_MULTIDIM_POLICY_TRANSMISSION_SCORE",
        goal_status="active", goal_achieved=False,
    )
    mandate_path.write_text(json.dumps(mandate, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("已保存对比图、结果报告与当前研究状态；夏普目标仍未实现。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定的指数多维非线性政策传导增量研究")
    parser.add_argument("stage", choices=["prepare", "run", "verify", "finish"])
    stage = parser.parse_args().stage
    {"prepare": prepare, "run": run, "verify": verify, "finish": finish}[stage]()
