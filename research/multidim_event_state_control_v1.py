"""复用已保存月度状态树，区分公布时点与指数条件组合的历史表现。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import multidim_nonlinear_score_v1 as common


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_event_state_control_v1"
PREVIOUS = ROOT / "reports/research/510300_multidim_money_surprise_score_v1"
DAILY = ROOT / "reports/research/510300_multidim_nonlinear_score_v1/historical_inputs_and_labels.parquet"
CALENDAR = ROOT / "reports/research/510300_money_consensus_increment_v2/inputs/104个月共识选择.csv"
STUDY = "510300_MULTIDIM_EVENT_STATE_CONTROL_V1"
FEATURES = common.FEATURES
END = pd.Timestamp("2025-12-31")
EVENT = "原可评分公布日"
ORDINARY = "普通日期"
OTHER = "其他已知公布日"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(common.clean(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    if (OUT / "protocol.json").exists():
        raise RuntimeError("公布时点对照设定已经固定，不覆盖。")
    paths = [DAILY, CALENDAR, PREVIOUS / "saved_models.json",
             PREVIOUS / "逐事件联合评分.csv", PREVIOUS / "月度事件联合状态及标签.parquet"]
    save("protocol.json", {
        "study_id": STUDY, "frozen_at": common.now(),
        "previous_goal_turn_classification": "PROGRESS_MONTHLY_SCORE_ACCOUNT_POINT_SHARPE_ONLY",
        "question": "此前高分月度事件的收益来自可重复的指数状态，还是仅出现在公布时点或少数新冲击附近？",
        "primary_period": ["2024-01-01", "2025-12-31"],
        "earlier_context": ["2021-01-01", "2023-12-31"],
        "model": "复用原47棵八变量状态树及各自训练池的预测分布；不重新拟合，不使用M2偏差，不修改阈值。",
        "ordinary_clock": "每个拟入场交易日前一自然日23:59:59判断；八项状态只取此前最近完整交易日16点快照。只使用在该截止前已经生成的最近一棵月度状态树。",
        "model_age": "原有模型在下一次合格月度更新前保持不变，记录实际年龄，不按事后盈亏增加过期阈值。",
        "announcement_class": "全部104月原公布台账按真实公布日期映射到严格次日后的首个交易日；该入场槽位均记公布日，包括缺预期的月份。其余槽位才是普通日期。周末与节假日公布并入随后首个开盘，不重复计算同一笔入场。",
        "missing_models": "第一棵原模型生成前不评分，覆盖表保留；退出超出2025年末的样本不进入本次已实现收益比较。",
        "score": "按原成熟月度训练池的拟合预测中位百分位秩；>=80且预测净收益>0为原固定高分口径。",
        "label": "完全沿用原固定10000份、下一交易日开盘买入、入场后第5个交易日开盘卖出的成本后标签，包括原佣金、滑点和权益登记口径。不是完整账户。",
        "nonoverlap": "各日期类别分别按判断时序保留第一个高分机会，原退出开盘以前不再进入；类别分别观察，不拼成组合。",
        "matched_control": {
            "lookback_sessions": 504, "nearest_nonoverlapping_dates": 3,
            "pool": "每个原公布事件之前已完成五日退出、八项状态齐全且判断时不是公布槽位的普通日期。",
            "distance": "用该历史池八项状态各自总体标准差归一化；八维差异的均方根距离；标准差为零时置1。按距离从小到大，平局取较近日期，贪心取三个持有区间互不重叠的日期。",
            "outcomes": "相似日期的选择只使用状态、时序和公布分类，不读取其收益作排名。选择完成后才附加收益。",
            "support": "不设结果驱动的相似度门槛；保存八维距离和逐维差异，匹配差时不声称控制充分。",
            "reused_dates": "不同公布事件可以重复匹配同一历史日期；按事件等权报告，明确唯一日期数，不把匹配次数当独立样本数。"
        },
        "causal_boundary": "普通日与公布日没有随机分配；匹配只含八个可观察状态且普通对照来自此前，时间环境、新政策和未观察信息仍可能不同。收益差只能作发现，不能命名为公告因果效应。",
        "future_news": "不因持有期间后来出现公告而剔除日期；未来新消息不能用于当时选择。",
        "selection_history": "这些历史已被研究；此前五笔主要期高分机会及9月利润集中已知，本轮属于事后归因与扩展诊断，不能恢复独立性。",
        "closest_old_studies": ["510300_MULTIDIM_MONEY_SURPRISE_SCORE_V1", "510300_MULTIDIM_NONLINEAR_SCORE_V1"],
        "substantive_difference": "此前月度模型只在公布事件评价，日频模型另行每天拟合。本轮保持月度模型及评分完全不变，只增加随后普通日期与先前相似日期的对照。",
        "new_fits": 0, "parameter_grids": 0, "new_accounts": 0,
        "account_boundary": "对照结果不扩大旧月度账户的交易授权，不把固定股数收益当账户夏普；原全部账户及失败结论保留。",
        "source_receipts": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in paths],
        "orders_authorized": False, "new_prospective_forecasts_enabled": False, "goal_achieved": False,
    })
    print("已固定原模型普通日期对照及三条历史相似日期比较；新增拟合为零。", flush=True)


def evaluate(tree: dict, vector: np.ndarray) -> tuple[float, int]:
    # sklearn树比较输入时使用float32；阈值仍保留原double精度。
    node = 0
    vector = vector.astype(np.float32)
    while tree["children_left"][node] != -1:
        feature = tree["feature"][node]
        node = (tree["children_left"][node] if float(vector[feature]) <= tree["threshold"][node]
                else tree["children_right"][node])
    return float(tree["value"][node]), node


def summary(values: pd.Series) -> dict:
    a = values.dropna().astype(float)
    positive, negative = a[a > 0], a[a < 0]
    return {"n": len(a), "mean_net5": float(a.mean()), "median_net5": float(a.median()),
            "win_rate": float((a > 0).mean()) if len(a) else np.nan,
            "return_payoff": float(positive.mean() / -negative.mean()) if len(positive) and len(negative) else np.nan,
            "minimum_net5": float(a.min()), "maximum_net5": float(a.max())}


def select_nonoverlap(frame: pd.DataFrame) -> pd.DataFrame:
    chosen = []
    last_exit = -1
    for index, row in frame.sort_values("decision_at").iterrows():
        if row.high_score and int(row.entry_idx) > last_exit:
            chosen.append(index)
            last_exit = int(row.exit_idx)
    return frame.loc[chosen].copy()


def load_inputs():
    daily = pd.read_parquet(DAILY).sort_values("idx").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily.date)
    assert np.array_equal(daily.idx.to_numpy(), np.arange(len(daily)))
    events = pd.read_parquet(PREVIOUS / "月度事件联合状态及标签.parquet")
    predictions = pd.read_csv(PREVIOUS / "逐事件联合评分.csv")
    models = read(PREVIOUS / "saved_models.json")
    calendar = pd.read_csv(CALENDAR)
    calendar["release_day"] = pd.to_datetime(calendar.available_at_upper_bound, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    days = pd.DatetimeIndex(daily.date)
    calendar["entry_idx"] = [int(days.searchsorted(t, side="right")) for t in calendar.release_day]
    calendar = calendar[calendar.release_day.le(END)]
    assert calendar.entry_idx.is_unique
    return daily, events, predictions, models, calendar


def run() -> None:
    if (OUT / "result.json").exists() or (OUT / "逐日对照评分.csv").exists():
        raise RuntimeError("已有本轮计算结果，不重跑或覆盖。")
    protocol = read(OUT / "protocol.json")
    for record in protocol["source_receipts"]:
        assert sha(ROOT / record["path"]) == record["sha256"]
    daily, events, original, models, calendar = load_inputs()
    model_times = pd.DatetimeIndex([pd.Timestamp(m["decision_at"]) for m in models])
    event_slots = {int(r.entry_idx): r.stat_month for r in calendar.itertuples()}
    original_slots = {int(r.state_idx) + 1: r.stat_month for r in original.itertuples()}
    fitted = {}
    reproduction_errors = []
    for model in models:
        pool = events[events.stat_month.isin(model["training_months"])]
        assert len(pool) == len(model["training_months"])
        assert pool.exit_at.le(pd.Timestamp(model["decision_at"])).all()
        fitted[model["stat_month"]] = np.array([evaluate(model["state_tree"], x)[0] for x in pool[FEATURES].to_numpy(float)])

    rows, coverage = [], []
    for i in range(len(daily) - 6):
        state = daily.iloc[i]
        entry, exit_day = daily.date.iloc[i + 1], daily.date.iloc[i + 6]
        if not pd.Timestamp("2021-01-01") <= entry <= END:
            continue
        cutoff = (entry - pd.Timedelta(days=1)).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        model_index = int(model_times.searchsorted(cutoff, side="right") - 1)
        row = {"state_idx": i, "state_date": state.date, "entry_idx": i + 1,
               "entry_date": entry, "exit_idx": i + 6, "exit_date": exit_day,
               "decision_at": cutoff, "status": "READY"}
        if model_index < 0:
            row["status"] = "NO_PRIOR_SAVED_MODEL"
        elif not state.valid_features:
            row["status"] = "MISSING_STATE"
        elif exit_day > END:
            row["status"] = "EXIT_AFTER_FIXED_CUTOFF"
        if row["status"] != "READY":
            coverage.append(row)
            continue
        model = models[model_index]
        prediction, leaf = evaluate(model["state_tree"], state[FEATURES].to_numpy(float))
        distribution = fitted[model["stat_month"]]
        score = 100 * ((distribution < prediction).mean() + .5 * (distribution == prediction).mean())
        group = EVENT if i + 1 in original_slots else OTHER if i + 1 in event_slots else ORDINARY
        assert state.decision_time <= cutoff < entry.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        assert pd.Timestamp(model["decision_at"]) <= cutoff
        row.update(group=group, stat_month=event_slots.get(i + 1), model_month=model["stat_month"],
                   model_decision_at=model["decision_at"], model_age_days=(cutoff - pd.Timestamp(model["decision_at"])).total_seconds()/86400,
                   prediction=prediction, score=float(score), leaf=leaf,
                   high_score=bool(score >= 80 and prediction > 0),
                   actual_net5=float(state.net_label),
                   era="2024—2025" if entry.year >= 2024 else "2021—2023")
        row.update({name: float(state[name]) for name in FEATURES})
        if group == EVENT:
            saved = original[original.stat_month.eq(row["stat_month"])].iloc[0]
            assert model["stat_month"] == row["stat_month"]
            errors = [abs(prediction - saved.state_prediction), abs(score - saved.state_score),
                      abs(row["actual_net5"] - saved.actual_net5)]
            assert max(errors) < 1e-10
            reproduction_errors.extend(errors)
        rows.append(row)
        coverage.append({k: row[k] for k in ["state_idx", "entry_date", "decision_at", "status"]})
    scored = pd.DataFrame(rows)
    assert scored.entry_idx.is_unique
    assert len(scored[scored.group.eq(EVENT)]) == len(original)
    scored.to_csv(OUT / "逐日对照评分.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(OUT / "全部日期覆盖.csv", index=False, encoding="utf-8-sig")
    comparisons, selected, bands = [], [], []
    for (era, group), block in scored.groupby(["era", "group"], sort=False):
        sparse = select_nonoverlap(block)
        selected.append(sparse)
        for kind, values in [("全部日期", block), ("全部高分标签", block[block.high_score]), ("不重叠高分机会", sparse)]:
            comparisons.append({"era": era, "group": group, "selection": kind, **summary(values.actual_net5)})
        for band in range(5):
            values = block[block.score.ge(20 * band) & (block.score.lt(20 * (band + 1)) if band < 4 else block.score.le(100))]
            bands.append({"era": era, "group": group, "score_band": f"{20*band}—{20*(band+1)}", **summary(values.actual_net5)})
    opportunities = pd.concat(selected, ignore_index=True).sort_values("decision_at")
    pd.DataFrame(comparisons).to_csv(OUT / "公布与普通日期比较.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(bands).to_csv(OUT / "公布与普通日期分层.csv", index=False, encoding="utf-8-sig")
    opportunities.to_csv(OUT / "全部不重叠高分机会.csv", index=False, encoding="utf-8-sig")

    # 相似日期匹配：先固定只含输入状态的排名，再读取所选日期的收益。
    pairs, event_matches, balance = [], [], []
    for event in original.to_dict("records"):
        state_idx = int(event["state_idx"])
        eligible = daily[daily.idx.ge(state_idx - 504) & daily.exit_idx.le(state_idx) & daily.valid_features].copy()
        eligible = eligible[~(eligible.idx + 1).isin(event_slots)]
        x = eligible[FEATURES].to_numpy(float)
        scale = x.std(axis=0, ddof=0)
        scale[scale < 1e-12] = 1
        current = np.array([event[f] for f in FEATURES], dtype=float)
        differences = (x - current) / scale
        eligible["distance"] = np.sqrt(np.mean(differences**2, axis=1))
        ranked = eligible[["idx", "exit_idx", "distance"]].sort_values(["distance", "idx"], ascending=[True, False])
        chosen = []
        for candidate in ranked.itertuples():
            entry_idx, exit_idx = int(candidate.idx) + 1, int(candidate.exit_idx)
            if all(entry_idx >= old_exit or exit_idx <= old_entry for _, old_entry, old_exit, _ in chosen):
                chosen.append((int(candidate.idx), entry_idx, exit_idx, float(candidate.distance)))
            if len(chosen) == 3:
                break
        assert len(chosen) == 3
        controls = []
        for rank, (idx, entry_idx, exit_idx, distance) in enumerate(chosen, 1):
            row = daily.iloc[idx]
            assert exit_idx <= state_idx and entry_idx not in event_slots
            control = {"stat_month": event["stat_month"], "event_entry_date": event["entry_date"],
                       "event_high_score": bool(event["state_score"] >= 80 and event["state_prediction"] > 0),
                       "era": event["era"], "rank": rank, "control_idx": idx,
                       "control_state_date": row.date, "control_entry_date": daily.date.iloc[entry_idx],
                       "control_exit_date": daily.date.iloc[exit_idx], "distance": distance,
                       "control_net5": float(row.net_label), "event_net5": float(event["actual_net5"])}
            pairs.append(control)
            controls.append(control)
        selected_states = daily.iloc[[c[0] for c in chosen]][FEATURES].to_numpy(float)
        for j, name in enumerate(FEATURES):
            balance.append({"stat_month": event["stat_month"], "feature": name,
                            "event_value": current[j], "control_mean": float(selected_states[:, j].mean()),
                            "pool_scale": scale[j], "standardized_difference": float((current[j] - selected_states[:, j].mean()) / scale[j])})
        event_matches.append({"stat_month": event["stat_month"], "entry_date": event["entry_date"],
                              "era": event["era"], "high_score": controls[0]["event_high_score"],
                              "event_net5": float(event["actual_net5"]),
                              "control_mean_net5": float(np.mean([c["control_net5"] for c in controls])),
                              "difference": float(event["actual_net5"] - np.mean([c["control_net5"] for c in controls])),
                              "mean_distance": float(np.mean([c["distance"] for c in controls])),
                              "max_distance": float(max(c["distance"] for c in controls)),
                              "pool_count": len(eligible),
                              "controls": ";".join(str(pd.Timestamp(c["control_entry_date"]).date()) for c in controls)})
    matched = pd.DataFrame(event_matches)
    paired = pd.DataFrame(pairs)
    matched_summaries = []
    for era, block in matched.groupby("era", sort=False):
        for kind, values in [("全部公布事件", block), ("原固定高分事件", block[block.high_score])]:
            joined = paired[paired.stat_month.isin(values.stat_month)]
            matched_summaries.append({"era": era, "selection": kind, "n_events": len(values),
                                      "n_control_uses": len(joined), "n_unique_controls": joined.control_idx.nunique(),
                                      "event_mean_net5": float(values.event_net5.mean()),
                                      "event_median_net5": float(values.event_net5.median()),
                                      "control_mean_net5": float(values.control_mean_net5.mean()),
                                      "mean_difference": float(values.difference.mean()),
                                      "median_difference": float(values.difference.median()),
                                      "fraction_event_above_controls": float(values.difference.gt(0).mean()),
                                      "mean_distance": float(values.mean_distance.mean()),
                                      "worst_distance": float(values.max_distance.max())})
    matched.to_csv(OUT / "逐公布事件的相似日期比较.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "全部历史相似日期.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(balance).to_csv(OUT / "八项状态匹配差异.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(matched_summaries).to_csv(OUT / "相似日期汇总.csv", index=False, encoding="utf-8-sig")
    checks = {"saved_event_rows_reproduced": len(original), "maximum_event_reproduction_error": max(reproduction_errors),
              "all_models_known_before_decision": True, "all_matched_labels_mature_before_event": True,
              "unique_entry_slots": bool(scored.entry_idx.is_unique), "no_future_announcement_exclusions": True,
              "new_fits": 0, "new_accounts": 0}
    save("必要计算核对.json", checks)
    save("result.json", {"study_id": STUDY, "completed_at": common.now(),
                         "status": "COMPLETED_EVENT_STATE_CONTROL_PENDING_INTERPRETATION",
                         "classification": "PROGRESS_EVENT_CLOCK_AND_STATE_SEPARATION",
                         "scored_dates": len(scored), "original_event_dates": len(original),
                         "ordinary_dates": int(scored.group.eq(ORDINARY).sum()),
                         "other_announcements": int(scored.group.eq(OTHER).sum()),
                         "maximum_model_age_days": float(scored.model_age_days.max()),
                         "comparisons": comparisons, "matched_summaries": matched_summaries,
                         "all_matched_control_uses": len(paired), "all_unique_matched_dates": int(paired.control_idx.nunique()),
                         "new_fits": 0, "parameter_grids": 0, "new_accounts": 0,
                         "account_net_sharpe": None, "goal_achieved": False, "independent_validation": False,
                         "current_view": "NO_VIEW", "position": "UNSET", "orders_authorized": False,
                         "new_prospective_forecasts_enabled": False})
    print("公布时点与指数状态对照完成，未重新拟合。", flush=True)
    print(pd.DataFrame(comparisons).query("selection == '不重叠高分机会'").to_string(index=False))
    print(pd.DataFrame(matched_summaries).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="历史公布时点与指数状态的固定对照")
    parser.add_argument("stage", choices=["prepare", "run"])
    action = parser.parse_args().stage
    {"prepare": prepare, "run": run}[action]()
