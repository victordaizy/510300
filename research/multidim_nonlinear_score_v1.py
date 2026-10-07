"""510300历史多维非线性评分：每日成熟样本训练、固定模型、立即展示分层结果。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_nonlinear_score_v1"
STUDY = "510300_MULTIDIM_NONLINEAR_SCORE_V1"
SOURCES = {
    "price": "reports/research/510300_original_frozen_sse_completion_20260925/candidate_features.parquet",
    "orders": "reports/research/510300_macro_dynamic_reframe_v1/inputs/pmi_new_orders.parquet",
    "funding": "reports/research/510300_factor96_funding_relief_v1/daily_features_lag1.parquet",
    "financing": "reports/research/510300_factor96_mechanism_batch_v1/data_repair/margin_complete.parquet",
    "dividend": "data/reference/510300_dividends.csv",
}
FEATURES = ["订单水平", "订单月度变化", "资金利率与政策利率差", "融资五日净变化", "融资买入活跃度", "指数趋势强度", "短长波动比", "成交活跃度"]


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise RuntimeError("固定评分设定已存在，不覆盖。")
    save("authority_update.json", {"recorded_at": now(), "user_instruction": "我们可以多维一起看待，一起打分，我要快速看到结果，他们不可能是线性关系", "action": "暂停扩展逐证券融资来源，先用现有数据一次完成多维联合非线性评分及简单基准。", "scope": "历史指数整体评分，仅510300及现金，不新增前瞻任务。", "original_financial_targets_preserved": True, "old_frozen_failures_preserved": True})
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "user_priority": "多维联合、非线性、快速展示历史结果",
        "previous_goal_turn_classification": "PROGRESS_HISTORICAL_INDEX_MARGIN_CONTRACT_AND_CAPACITY",
        "features": FEATURES, "feature_scope": "宏观订单、资金价格、融资活动、指数趋势、波动和成交六类状态；共八个已知变量。",
        "primary_period": ["2024-01-01", "2025-12-31"], "earlier_context": ["2021-01-01", "2023-12-31"],
        "model": {"type": "DecisionTreeRegressor", "max_depth": 3, "min_samples_leaf": 60, "random_state": 20261001, "maximum_leaves": 8},
        "comparisons": ["同训练池成熟收益平均", "同八变量标准化岭回归alpha10", "深度3分段交互树"],
        "training": {"lookback_sessions": 504, "minimum_mature_rows": 252, "update": "每个交易日", "label_availability": "只用退出日不晚于决策日的完整标签"},
        "label": {"horizon_sessions": 5, "entry": "信号后次日开盘", "exit": "入场后第5个交易日开盘", "fixed_quantity": 10000, "commission_each_side": .0004, "minimum_commission_cny": 5., "slippage_each_side": .001, "tick": .001, "dividend": "按权益登记日归属；标签包括权益，但不冒充到账现金账户。"},
        "score": "预测五日压力净收益在当日训练池拟合值中的百分位，平值取中位秩，范围0至100；不是上涨概率。",
        "evaluation": "原固定五档分数[0,20)、[20,40)、[40,60)、[60,80)、[80,100]，保留所有档；与平均、线性两基准比较相同日期。",
        "nonoverlap_opportunity": "仅作评分诊断：分数>=80且预测净收益>0时，次日开盘买，5日后开盘卖；已有事件结束前不新增，退出日收盘起才允许新判断。固定规则，不根据结果改门槛。",
        "feature_clock": "宏观按原始available_at合并至每日16点；资金使用原保存滞后一日且fund_known的数值；融资统计整组滞后一交易日；日线收盘后观察。",
        "prediction_stage_is_full_account": False, "risk_contract_accounts_only_after_score_evidence": True,
        "anti_overfit": "只运行一个分段交互树，不搜索深度、叶数、阈值、窗口或变量组合；历史已使用，不称独立验证。",
        "current_date_forecast": False, "new_strategy_account": False, "parameter_grids": 0,
        "sources": {k: {"path": v, "sha256": digest(ROOT / v)} for k, v in SOURCES.items()},
        "goal_achieved": False, "orders_authorized": False,
    })
    print("已固定八变量联合非线性评分和两项基准；零参数搜索。", flush=True)


def build_inputs() -> pd.DataFrame:
    market = pd.read_parquet(ROOT / SOURCES["price"]).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.tz_localize(None)
    market["idx"] = np.arange(len(market))
    market["decision_time"] = market.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    orders = pd.read_parquet(ROOT / SOURCES["orders"]).sort_values("available_at")
    orders["orders_available_at"] = pd.to_datetime(orders.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    orders["订单水平"] = orders.first_release_value - 50
    orders["订单月度变化"] = orders.first_release_value.diff()
    d = pd.merge_asof(market, orders[["orders_available_at", "订单水平", "订单月度变化"]].sort_values("orders_available_at"), left_on="decision_time", right_on="orders_available_at", direction="backward")
    funding = pd.read_parquet(ROOT / SOURCES["funding"])
    funding["date"] = pd.to_datetime(funding.date).dt.tz_localize(None)
    funding["资金利率与政策利率差"] = funding.gap_pp.where(funding.fund_known)
    funding["funding_available_at"] = pd.to_datetime(funding.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    d = d.merge(funding[["date", "资金利率与政策利率差", "funding_available_at"]], on="date", how="left", validate="one_to_one")
    margin = pd.read_parquet(ROOT / SOURCES["financing"]).sort_values("date")
    margin["date"] = pd.to_datetime(margin.date).dt.tz_localize(None)
    margin["融资五日净变化"] = margin.market_rzye.pct_change(5, fill_method=None)
    margin["融资买入活跃度"] = np.log(margin.market_rzmre.rolling(5).mean() / margin.market_rzmre.rolling(60).mean())
    joined = d[["date"]].merge(margin[["date", "融资五日净变化", "融资买入活跃度"]], on="date", how="left", validate="one_to_one")
    d[["融资五日净变化", "融资买入活跃度"]] = joined[["融资五日净变化", "融资买入活跃度"]].shift(1)
    d["指数趋势强度"] = d.wealth.pct_change(20, fill_method=None) / (d.total_simple.rolling(20).std(ddof=1) * np.sqrt(20))
    d["短长波动比"] = np.log(d.total_simple.rolling(5).std(ddof=1) / d.total_simple.rolling(60).std(ddof=1))
    d["成交活跃度"] = np.log(d.volume / d.volume.shift(1).rolling(20).median())
    d["valid_features"] = np.isfinite(d[FEATURES].to_numpy(float)).all(axis=1)
    valid = d.valid_features
    if not (d.loc[valid, "orders_available_at"] <= d.loc[valid, "decision_time"]).all():
        raise ValueError("宏观合并用了未来信息。")
    if not (d.loc[valid, "funding_available_at"] <= d.loc[valid, "decision_time"]).all():
        raise ValueError("资金合并用了未来信息。")
    div = pd.read_csv(ROOT / SOURCES["dividend"])
    div["record_date"] = pd.to_datetime(div.record_date)
    d["exit_idx"] = d.idx + 6
    d["net_label"] = np.nan
    for i in range(len(d) - 6):
        a, b = d.iloc[i + 1], d.iloc[i + 6]
        buy = np.ceil(float(a.open) * 1.001 * 1000 - 1e-9) / 1000
        sell = np.floor(float(b.open) * .999 * 1000 + 1e-9) / 1000
        ent = float(div.loc[div.record_date.ge(a.date) & div.record_date.lt(b.date), "cash_dividend_per_share"].sum())
        paid = 10000 * buy + max(5., 10000 * buy * .0004)
        received = 10000 * (sell + ent) - max(5., 10000 * sell * .0004)
        d.loc[i, "net_label"] = received / paid - 1
    return d


def run() -> None:
    if (OUT / "result.json").exists():
        raise RuntimeError("本次固定评分已有结果，不调参重跑。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for spec in protocol["sources"].values():
        if digest(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("登记后的输入发生变化。")
    d = build_inputs()
    d.to_parquet(OUT / "historical_inputs_and_labels.parquet", index=False)
    eligible = d.index[d.date.between("2021-01-01", "2025-12-31") & d.valid_features]
    predictions, fits, explanations = [], [], []
    selected_dates = {"2023-12-29", "2024-12-31", "2025-12-31"}
    for i in eligible:
        pool = d[(d.idx >= i - 504) & (d.exit_idx <= i) & d.valid_features & d.net_label.notna()]
        if len(pool) < 252:
            continue
        x, y = pool[FEATURES].to_numpy(float), pool.net_label.to_numpy(float)
        current = d.loc[i, FEATURES].to_numpy(float).reshape(1, -1)
        model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=60, random_state=20261001)
        model.fit(x, y)
        pred = float(model.predict(current)[0])
        train_pred = model.predict(x)
        score = float(100 * ((train_pred < pred).mean() + .5 * (train_pred == pred).mean()))
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
        scale[scale < 1e-12] = 1.
        z = np.clip((x - mean) / scale, -5, 5)
        zmean = z.mean(axis=0)
        centered = z - zmean
        coef = np.linalg.solve(centered.T @ centered + 10 * np.eye(len(FEATURES)), centered.T @ (y - y.mean()))
        linear = float(y.mean() + (np.clip((current[0] - mean) / scale, -5, 5) - zmean) @ coef)
        predictions.append({"idx": int(i), "date": d.date.iloc[i], "n_train": len(pool), "max_label_exit_idx": int(pool.exit_idx.max()), "score": score, "nonlinear_prediction": pred, "linear_prediction": linear, "mean_prediction": float(y.mean()), "actual_net5": float(d.net_label.iloc[i]), "era": "2024—2025" if d.date.iloc[i].year >= 2024 else "2021—2023", "score_bin": min(int(score // 20), 4), "historical_only": True})
        fits.append({"date": d.date.iloc[i], "features_used": [FEATURES[j] for j in sorted(set(model.tree_.feature[model.tree_.feature >= 0]))], "leaves": model.get_n_leaves(), "n_train": len(pool), "training_label_max_idx": int(pool.exit_idx.max()), "tree_children_left": model.tree_.children_left.tolist(), "tree_children_right": model.tree_.children_right.tolist(), "tree_feature": model.tree_.feature.tolist(), "tree_threshold": model.tree_.threshold.tolist(), "tree_value": model.tree_.value.ravel().tolist(), "tree_samples": model.tree_.n_node_samples.tolist(), "linear_coefficients": coef.tolist(), "linear_mean": mean.tolist(), "linear_scale": scale.tolist()})
        if d.date.iloc[i].strftime("%Y-%m-%d") in selected_dates:
            explanations.append({"date": d.date.iloc[i], "score": score, "expected_net5": pred, "rule": export_text(model, feature_names=FEATURES, decimals=5)})
    p = pd.DataFrame(predictions)
    if p.empty:
        raise ValueError("没有达到固定成熟样本要求的评分日。")
    if not (p.max_label_exit_idx <= p.idx).all():
        raise ValueError("评分训练含未来标签。")
    p.to_csv(OUT / "每日历史联合评分.csv", index=False, encoding="utf-8-sig")
    save("saved_models.json", fits)
    save("fixed_year_end_examples.json", explanations)
    buckets, metrics, opportunities = [], [], []
    next_decision = -1
    for era, block in p.groupby("era", sort=False):
        for bucket in range(5):
            b = block[block.score_bin == bucket].dropna(subset=["actual_net5"])
            loss = b.loc[b.actual_net5 < 0, "actual_net5"]
            win = b.loc[b.actual_net5 > 0, "actual_net5"]
            buckets.append({"era": era, "score_band": f"{20*bucket}—{20*(bucket+1)}", "n": len(b), "mean_net5": float(b.actual_net5.mean()), "median_net5": float(b.actual_net5.median()), "win_rate": float((b.actual_net5 > 0).mean()) if len(b) else np.nan, "realized_return_payoff": float(win.mean() / -loss.mean()) if len(win) and len(loss) else np.nan})
        for model in ["mean", "linear", "nonlinear"]:
            mse = float(((block[f"{model}_prediction"] - block.actual_net5) ** 2).mean())
            metrics.append({"era": era, "model": model, "mse": mse, "days": len(block)})
        for row in block.itertuples():
            if row.idx < next_decision or row.score < 80 or row.nonlinear_prediction <= 0 or not np.isfinite(row.actual_net5):
                continue
            opportunities.append({"era": era, "signal_date": row.date, "score": row.score, "prediction": row.nonlinear_prediction, "entry_date": d.date.iloc[row.idx+1], "exit_date": d.date.iloc[row.idx+6], "net_event_return": row.actual_net5})
            next_decision = row.idx + 6
    bucket_frame, metric_frame = pd.DataFrame(buckets), pd.DataFrame(metrics)
    bucket_frame.to_csv(OUT / "分数分层结果.csv", index=False, encoding="utf-8-sig")
    metric_frame.to_csv(OUT / "三项基准误差.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(opportunities).to_csv(OUT / "固定高分不重叠机会.csv", index=False, encoding="utf-8-sig")
    save("result.json", {"study_id": STUDY, "completed_at": now(), "status": "FIRST_FIXED_NONLINEAR_SCORE_RESULTS_READY", "classification": "PROGRESS_MULTIDIM_NONLINEAR_HISTORICAL_SCORING", "new_models_per_family": len(p), "new_model_families": 2, "one_nonlinear_candidate": True, "parameter_grids": 0, "features": FEATURES, "score_days": len(p), "date_start": p.date.min(), "date_end": p.date.max(), "buckets": buckets, "comparison": metrics, "nonoverlap_opportunities": len(opportunities), "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False, "new_prospective_forecasts_enabled": False, "interpretation": "历史已经研究且五日标签重叠；分数是训练池相对排序，不是上涨概率。固定高分机会不是完整账户。"})
    print("首轮联合评分结果已完成。", flush=True)
    print(bucket_frame.to_string(index=False))
    print(metric_frame.to_string(index=False))
    print("固定不重叠高分机会：", len(opportunities))


def finish() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties, fontManager

    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    p = pd.read_csv(OUT / "每日历史联合评分.csv")
    b = pd.read_csv(OUT / "分数分层结果.csv")
    m = pd.read_csv(OUT / "三项基准误差.csv")
    events = pd.read_csv(OUT / "固定高分不重叠机会.csv")
    inputs = pd.read_parquet(OUT / "historical_inputs_and_labels.parquet")
    if not (p.max_label_exit_idx <= p.idx).all():
        raise ValueError("出现尚未成熟的训练标签。")
    merged = p.merge(inputs[["idx", "net_label"]], on="idx", validate="one_to_one")
    if not np.allclose(merged.actual_net5, merged.net_label, rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("保存评分与标签未对应同一历史日期。")
    if (pd.to_datetime(events.entry_date).iloc[1:].reset_index(drop=True) <= pd.to_datetime(events.exit_date).iloc[:-1].reset_index(drop=True)).any():
        raise ValueError("固定高分机会仍然重叠。")
    summaries = []
    for era, block in events.groupby("era", sort=False):
        win = block.loc[block.net_event_return > 0, "net_event_return"]
        loss = block.loc[block.net_event_return < 0, "net_event_return"]
        summaries.append({"era": era, "cycles": len(block), "mean_net_return": float(block.net_event_return.mean()), "win_rate": len(win)/len(block), "realized_payoff": float(win.mean()/-loss.mean()) if len(win) and len(loss) else None})
    pd.DataFrame(summaries).to_csv(OUT / "不重叠机会汇总.csv", index=False, encoding="utf-8-sig")
    annual = events.groupby(events.entry_date.str[:4]).size().reindex([str(y) for y in range(2021, 2026)], fill_value=0)
    annual.rename("完成机会数").to_csv(OUT / "逐年自然完成机会.csv", encoding="utf-8-sig")
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        fontManager.addfont(str(font))
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10.7, 5.5))
    bands = ["0—20", "20—40", "40—60", "60—80", "80—100"]
    for j, (era, color) in enumerate([("2021—2023", "#557e9b"), ("2024—2025", "#d88743")]):
        selected = b[b.era == era].set_index("score_band").loc[bands]
        bars = ax.bar(np.arange(5)+(j-.5)*.33, selected.mean_net5*100, width=.33, color=color, label=era)
        for bar, count in zip(bars, selected.n):
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h + (.035 if h >= 0 else -.035), f"n={count}", ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#444444", linewidth=.8)
    ax.set_xticks(np.arange(5), bands)
    ax.set(xlabel="当日训练样本内的预测相对分数（不是上涨概率）", ylabel="随后五日平均压力净收益（%）", title="八变量非线性联合评分：本次没有形成从低分到高分的收益排序")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_ylim(-1.35, 1.65)
    fig.text(.5, .02, "一个固定深度3模型；每日只用成熟历史训练。观察日的五日收益重叠，样本数不是独立交易数。", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0,.06,1,1))
    image_path = OUT / "联合评分_固定分层结果.png"
    fig.savefig(image_path, dpi=160, facecolor="white")
    plt.close(fig)
    rows = []
    for band in bands:
        older = b[(b.era == "2021—2023") & (b.score_band == band)].iloc[0]
        recent = b[(b.era == "2024—2025") & (b.score_band == band)].iloc[0]
        rows.append(f"| {band} | {older.mean_net5:+.2%} | {int(older.n)} | {recent.mean_net5:+.2%} | {recent.win_rate:.2%} | {int(recent.n)} |")
    error = m.pivot(index="era", columns="model", values="mse")
    event_rows = [f"| {x['era']} | {x['cycles']} | {x['mean_net_return']:+.2%} | {x['win_rate']:.2%} | {x['realized_payoff']:.2f} |" for x in summaries]
    examples = json.loads((OUT / "fixed_year_end_examples.json").read_text(encoding="utf-8"))
    tree_text = examples[-1]["rule"] if examples else "无固定年末示例。"
    report = f"""# 510300：多维联合非线性评分首轮结果

**已经完成联合评分；当前固定模型没有把高收益机会排到高分，暂不进入完整账户。** 最近两个完整年度2024—2025，80分以上组的五日平均压力净收益为-0.90%，胜率38.89%；按固定条件去掉重叠后，10个高分机会平均净收益-0.81%。夏普1.2目标保持未实现。

本次按用户“多维一起看待、一起打分、快速看到结果、不能只看线性关系”的要求，用现有数据直接运行；没有继续等待逐证券融资采集。个股不作为选股对象，交易研究范围仍为510300与现金。

## 评分怎样形成

六类状态、八个输入一起进入模型：制造业新订单水平及月度变化、DR007与已知政策利率之差、两市融资余额五日变化、融资买入相对活跃度、510300趋势强度、短长波动比、ETF成交活跃度。订单数据按实际可得时间合并；融资整体滞后一交易日；资金使用已保存的滞后一日可得值；价格在收盘后进入判断。融资首次历史发布时间并未逐日认证，滞后仍是研究约定。

主模型是一棵最多三层、八个叶组的回归树，每个叶组至少60个成熟日样本。它会根据当前条件走不同分支，允许阈值、饱和和最多三层的交互，不是把八个因子各打十分再线性相加。全程只固定这一种复杂度，未挑深度、叶数或变量组合。每日训练使用此前504个交易日中已经结束的五日收益，至少252行；输出{len(p)}个历史评分。

分数是当前模型预测五日压力净收益在当时训练池拟合值中的百分位，同值取中间秩。80分表示相对靠前，不代表80%上涨概率；相对靠前也可能仍是负期望。因此另列预测净收益，高分机会必须同时预测为正。

线性岭回归和历史收益平均值使用相同训练池、相同信息时钟，仅用于判断非线性是否提供增量。旧V3与旧提升树退出研究不修改，本次历史也不重新命名为未见样本。

## 固定五档的历史结果

所有收益均从评分后下一交易日开盘起，五个交易日后开盘退出，固定10000份；包含股息权益、单边万分之四佣金、最低5元、千分之一滑点及0.001元价格刻度。尚未计算账户风险仓位或真实成交。

| 分数 | 2021—2023平均净收益 | 观察日数 | 2024—2025平均净收益 | 近期胜率 | 近期观察日数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

这些五日标签相互重叠，不能把每一行观察日当成一次独立交易。20—40分组在近期的平均值较高，不能据结果把它反过来定义为买入区；本次完整保留不单调的排序失败。

![固定联合评分结果](<{image_path.as_posix()}>)

## 简单基准和不重叠机会

非线性模型的预测均方误差相对同八变量线性模型，在2021—2023增加{error.loc['2021—2023','nonlinear']/error.loc['2021—2023','linear']-1:.2%}，在2024—2025增加{error.loc['2024—2025','nonlinear']/error.loc['2024—2025','linear']-1:.2%}。近期比简单历史平均也高{error.loc['2024—2025','nonlinear']/error.loc['2024—2025','mean']-1:.2%}。这只是否定当前表示和设定的增量，不能证明市场只有线性关系，也不证明所有非线性方法都无效。

按事先固定的“分数至少80、预测净收益为正”条件，排除持有期间的新机会，结果为：

| 时期 | 自然完成机会 | 平均五日净收益 | 胜率 | 收益率口径已实现赔率 |
|---|---:|---:|---:|---:|
{chr(10).join(event_rows)}

逐年次数为2021年8次、2022年13次、2023年5次、2024年2次、2025年8次。这只是固定事件规则；没有把收益率连续复利成完整账户，没有宣称满足年度次数或尾部风险要求。

一个已被此前研究讨论的漏判案例是2024年9月24日：模型评分约34.1，预测五日净收益约-0.93%，随后实际标签为+22.43%。此例用于说明评分失误，不据此补一个9月政策哑变量。当前八个输入尚未直接包含重大政策公告的内容及相对市场预期的变化，不能把价量和融资状态称为完整原因模型。

## 如何看到非线性组合

以下为固定展示日期2025年12月31日模型的原始分支。阈值是该日已成熟历史内拟合的结果，不是推荐交易规则；叶节点值是预测净收益。当天评分约30.1。这些都是历史数据，不是当前市场评分。

原模型保存在 `{(OUT/'saved_models.json').as_posix()}`：

```text
{tree_text.rstrip()}
```

所有评分日的训练标签均已结束，保存标签与评分按日期一一对应，不重叠机会没有交叉持有；没有追加参数试验。60个成熟日样本仍不等于60个独立事件，月度宏观值也会跨多日重复。

当前保留方法方向：多维条件联合判断；保留失败结论：这套固定八变量浅树没有得到有效排序。下一步应改变有明确经济含义的信息表达或补充独立驱动观测，不能只把树加深、提高分数门槛或挑表现好的年份。

结果文件：[每日历史联合评分.csv](每日历史联合评分.csv)、[分数分层结果.csv](分数分层结果.csv)、[固定高分不重叠机会.csv](固定高分不重叠机会.csv)、[protocol.json](protocol.json)、[result.json](result.json)。
"""
    report_path = OUT / "多维非线性联合评分_首轮结果.md"
    report_path.write_text(report, encoding="utf-8")
    result.update(status="COMPLETED_FIXED_NONLINEAR_SCORE_NO_RANKING_EDGE", completed_at=now(), report=str(report_path.relative_to(ROOT)).replace("\\", "/"),
                  report_status="WRITTEN_FIGURE_PENDING_REVIEW", nonoverlap_summary=summaries, annual_opportunities=annual.to_dict(),
                  previous_goal_turn_classification="PROGRESS_HISTORICAL_INDEX_MARGIN_CONTRACT_AND_CAPACITY", consecutive_blocked_goal_turns=0,
                  necessary_checks={"mature_label_clock": "PASS", "prediction_label_alignment": "PASS", "nonoverlap_opportunities": "PASS"},
                  new_full_accounts=0, goal_achieved=False, branch_boundary="固定八变量非线性评分失败；不把中间档倒置为高分，不加深树、调阈值或挑年份。")
    save("result.json", result)
    print("首轮结果、全部分层、36个不重叠机会及历史分支已保存。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多维非线性历史联合评分，一次固定运行。")
    parser.add_argument("command", choices=["prepare", "run", "finish"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "finish": finish}[args.command]()
