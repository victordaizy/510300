"""先检验NFCI状态对五日风险预测的增量；未通过预定门槛不计算新账户。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest, local_import_closure, LATEST
import research.selected_mix_daily_two_year_v1 as daily
from research.adaptive_allocation_v1 import normalize_dividends

OUT = ROOT / "reports/research/510300_nfci_tail_forecast_increment_v1"
MACRO = ROOT / "reports/research/510300_selected_mix_nfci_increment_daily_v1/inputs/point_in_time_macro.parquet"
STUDY = "510300_NFCI_TAIL_FORECAST_INCREMENT_V1"
PRIMARY = "RELATIVE_CREDIT"
POLICIES = {"AGGREGATE_TIGHTENING": "nfci_change4", PRIMARY: "nfci_credit_minus_risk_change4"}


def prepare():
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    macro = pd.read_parquet(MACRO)
    pd.testing.assert_series_equal(data.date.reset_index(drop=True), macro.date, check_names=False, check_dtype=False)
    labels = daily.five_day_labels(data, normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv")))
    labels["macro_known"] = macro.macro_known.iloc[labels.origin_index.to_numpy(int)].to_numpy()
    for policy, feature in POLICIES.items():
        labels[policy] = macro[feature].gt(0).iloc[labels.origin_index.to_numpy(int)].to_numpy()
    return data, macro, labels


def predict(data, macro, labels, t, policy, baseline=None):
    original = baseline if baseline is not None else daily.tail_at(data, labels, t)
    record = {**original, "policy": policy, "used_macro_condition": False,
              "macro_status": "NO_VIEW" if not macro.macro_known.iloc[t] else "AVAILABLE",
              "macro_vintage_date": macro.macro_vintage_date.iloc[t], "macro_available_at": macro.macro_available_at.iloc[t]}
    if not macro.macro_known.iloc[t] or not original["available"]:
        return record
    feature = POLICIES[policy]
    state = bool(macro[feature].iloc[t] > 0)
    candidates = labels[labels.origin_index.isin(original["label_indices"])]
    candidates = candidates[candidates.macro_known & candidates[policy].eq(state)]
    if len(candidates) < 60:
        record["macro_status"] = "INSUFFICIENT_MATCHED_ROWS_BASELINE_FALLBACK"
        return record
    assert candidates.exit_index.lt(t).all()
    assert candidates.origin.ge(pd.Timestamp(data.date.iloc[t]) - pd.DateOffset(years=2)).all()
    values = candidates.gross_return5.to_numpy(float)
    tail = np.sort(values)[:max(1, int(np.ceil(.05 * len(values))))]
    record.update(used_macro_condition=True, macro_state=state, training_rows=len(values),
                  label_indices=candidates.origin_index.to_list(), latest_label_exit_index=int(candidates.exit_index.max()),
                  q05=float(np.quantile(values, .05)), es95=max(0., -float(tail.mean())))
    return record


def fz0(realized, quantile, es_return):
    """Patton、Ziegel、Chen(2019)式(6)，收益符号的VaR及ES联合评分，越低越好。"""
    y, q, e = np.broadcast_arrays(np.asarray(realized, float), np.asarray(quantile, float), np.asarray(es_return, float))
    if not (np.isfinite(y).all() and np.isfinite(q).all() and np.isfinite(e).all() and (e < 0).all() and (e <= q).all()):
        raise ValueError("联合评分要求有限预测、ES为负且不高于VaR；不能删去不合格预测后美化评分")
    return -(y <= q).astype(float) * (q - y) / (.05 * e) + q / e + np.log(-e) - 1


def checks(data, macro, labels):
    t = int(data.index[data.date.ge("2019-01-01")][0])
    before = predict(data, macro, labels, t, PRIMARY)
    changed = labels.copy()
    outside = changed.exit_index.ge(t) | changed.origin.lt(data.date.iloc[t] - pd.DateOffset(years=2))
    changed.loc[outside, "gross_return5"] = 10000.
    after = predict(data, macro, changed, t, PRIMARY)
    assert before["label_indices"] == after["label_indices"]
    assert before["q05"] == after["q05"] and before["es95"] == after["es95"]
    missing = macro.copy()
    missing.loc[t, "macro_known"] = False
    fallback = predict(data, missing, labels, t, PRIMARY)
    baseline = daily.tail_at(data, labels, t)
    assert fallback["q05"] == baseline["q05"] and fallback["es95"] == baseline["es95"]
    y = np.array([-.06, -.01, .03])
    one = fz0(y, -.04, -.05) - fz0(y, -.035, -.045)
    two = fz0(2*y, -.08, -.1) - fz0(2*y, -.07, -.09)
    np.testing.assert_allclose(one, two, atol=1e-12, rtol=0)
    np.testing.assert_allclose(fz0([-.01, -.06], -.04, -.05), [.8+np.log(.05)-1, 8+.8+np.log(.05)-1])
    return {"future_and_expired_labels_excluded": True, "missing_macro_exact_baseline_fallback": True,
            "joint_score_hand_calculation": True, "score_difference_scale_invariance": True}


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("风险预测实验已固定，不覆盖协议")
    data, macro, labels = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir()
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    protocol = {
        "at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "控制原本地动量风险分组后，事前已知的金融条件变化能否改善五日VaR/ES预测。",
        "difference_from_failed_exit_use": "检验损失分布，不拟合继续持有均值，不改变退出模型。风险预测合格之前不运行新账户，也不以账户收益选择分组。",
        "known_prior_result": "NFCI持有/退出增量用途已经失败；本项是随后提出的另一明确用途，计入整个项目选择历史，不是独立外部验证。",
        "baseline": "精确复用两年内已成熟五日含分红开盘至开盘收益分布，匹配20日动量方向，至少60行。",
        "two_fixed_comparisons": {"AGGREGATE_TIGHTENING": "原组内再匹配当时已知NFCI四周变化是否>0", PRIMARY: "原组内再匹配信用减风险分项四周变化是否>0"},
        "sample_rule": "原风险样本中进一步匹配宏观状态，至少60行，否则回退原风险预测；宏观过期也回退。训练仅最近两个日历年、退出严格早于预测日、逐日更新。",
        "risk_measure": "5%经验分位数、原定义的最差ceil(5%N)收益均值；不改原估计公式。",
        "periods": {"earlier": ["2015-01-05", "2019-12-31"], "main": ["2020-01-02", "2026-09-16"]},
        "evaluation": "逐日保存预测及全日评分；主评价固定从首预测原点每隔五个交易日取一次，持有段不重叠，全部保留；只评价在各区间内已经成熟的结果，跨区间标签不计入前区间，另列所有年份。",
        "score": "FZ0联合VaR/ES评分，原始值越低越好；VaR分位数损失与实际跌破率另列。",
        "score_source": "https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf",
        "statistics": "固定4个非重叠观察组成20交易日区块，2000次，种子20260925；差值为新方案减原对照，未校正项目级反复检验。",
        "continuation_gate": "预指定RELATIVE_CREDIT在两个完整评价区间的FZ0均值差95%区间上界都<0，两个区间的分位数平均损失均不增加，且各区间VaR实际跌破率<=10%；否则该用途冻结，新账户NOT_RUN。",
        "orders_authorized": False, "new_model_parameter_search": 0, "new_regression_fits": 0,
        "planned_new_accounts_before_gate": 0, "goal_achieved": False, "checks": checks(data, macro, labels),
    }
    save(OUT / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update([LATEST / "candidate_features.parquet", MACRO, ROOT / "data/reference/510300_dividends.csv",
                    daily.OUT / "results/tail_forecasts.json"])
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(OUT / "protocol.json"),
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    print("风险预测用途已固定：仅两项宏观状态比较，先过预测门槛再研究账户。", flush=True)


def interval(difference, rng):
    values = np.asarray(difference, float)
    n = len(values)
    blocks = int(np.ceil(n / 4))
    starts = rng.integers(0, n, size=(2000, blocks))
    indexes = ((starts[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
    means = values[indexes].mean(axis=1)
    return {"mean_difference": float(values.mean()), "lower_95": float(np.quantile(means, .025)),
            "upper_95": float(np.quantile(means, .975)), "observations": n,
            "block_nonoverlap_observations": 4, "replications": 2000, "selection_adjusted": False}


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, "冻结来源发生变化：" + name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    data, macro, labels = prepare()
    first = int(data.index[data.date.ge("2015-01-05")][0])
    old = {r["origin_index"]: r for r in read(daily.OUT / "results/tail_forecasts.json")}
    forecasts, scored = [], []
    baseline_reproductions = 0
    for t in range(first, len(data)):
        base = daily.tail_at(data, labels, t)
        assert base["available"]
        if t in old:
            for key in ["label_indices", "es95", "q05", "training_rows", "latest_label_exit_index"]:
                assert base[key] == old[t][key]
            baseline_reproductions += 1
        predictions = {"BASELINE": {**base, "policy": "BASELINE", "used_macro_condition": False},
                       **{policy: predict(data, macro, labels, t, policy, base) for policy in POLICIES}}
        for policy, prediction in predictions.items():
            forecasts.append(prediction)
            observed = labels[labels.origin_index.eq(t)]
            if len(observed) and data.date.iloc[t].year < 2020 and observed.exit_date.iloc[0].year >= 2020:
                observed = observed.iloc[:0]
            if len(observed):
                y = float(observed.gross_return5.iloc[0])
                q, es = float(prediction["q05"]), float(prediction["es95"])
                scored.append({"policy": policy, "origin_index": t, "origin": data.date.iloc[t],
                    "maturity_index": int(observed.exit_index.iloc[0]), "maturity_date": observed.exit_date.iloc[0],
                    "period": "earlier" if data.date.iloc[t].year < 2020 else "main",
                    "fixed_nonoverlap_phase": (t-first) % 5 == 0, "return5": y, "q05": q, "es95": es,
                    "fz0": float(fz0(y, q, -es)), "pinball": float((.05 - (y < q)) * (y-q)),
                    "breach": y < q, "used_macro_condition": prediction["used_macro_condition"],
                    "training_rows": prediction["training_rows"]})
        if (t-first+1) % 500 == 0:
            print(f"五日风险逐日估计 {t-first+1}/{len(data)-first}，尚未执行账户。", flush=True)
    save(OUT / "forecasts.json", forecasts, True)
    score = pd.DataFrame(scored)
    score.to_parquet(OUT / "mature_forecast_scores.parquet", index=False)
    phase = score[score.fixed_nonoverlap_phase]
    periods, comparisons, annual = [], [], []
    rng = np.random.default_rng(20260925)
    for (period, policy), part in phase.groupby(["period", "policy"], sort=True):
        part = part.sort_values("origin_index")
        assert (part.origin_index.to_numpy()[1:] + 1 >= part.maturity_index.to_numpy()[:-1]).all()
        periods.append({"period": period, "policy": policy, "observations": len(part),
                        "mean_fz0": float(part.fz0.mean()), "mean_pinball": float(part.pinball.mean()),
                        "breach_rate": float(part.breach.mean()), "breaches": int(part.breach.sum()),
                        "macro_condition_used": int(part.used_macro_condition.sum())})
        if policy != "BASELINE":
            baseline = phase[phase.period.eq(period) & phase.policy.eq("BASELINE")].sort_values("origin_index")
            np.testing.assert_array_equal(part.origin_index, baseline.origin_index)
            comparisons.append({"period": period, "policy": policy,
                "pinball_mean_difference": float(part.pinball.mean() - baseline.pinball.mean()),
                "breach_rate": float(part.breach.mean()),
                **interval(part.fz0.to_numpy() - baseline.fz0.to_numpy(), rng)})
    for (year, policy), part in phase.groupby([phase.origin.dt.year, "policy"], sort=True):
        annual.append({"year": int(year), "policy": policy, "observations": len(part),
                       "mean_fz0": float(part.fz0.mean()), "mean_pinball": float(part.pinball.mean()),
                       "breach_rate": float(part.breach.mean())})
    save(OUT / "yearly_risk_scores.json", annual, True)
    primary = [r for r in comparisons if r["policy"] == PRIMARY]
    gate = len(primary) == 2 and all(r["upper_95"] < 0 and r["pinball_mean_difference"] <= 0 and r["breach_rate"] <= .10 for r in primary)
    result = {"at": now(), "study_id": STUDY, "status": "RISK_FORECAST_INCREMENT_SUPPORTED_ACCOUNT_PENDING" if gate else "FROZEN_NO_RELIABLE_NFCI_TAIL_INCREMENT",
              "continuation_gate": gate, "period_scores": periods, "comparisons": comparisons,
              "daily_origins": len(data)-first, "new_conditional_risk_estimates": 2*(len(data)-first),
              "new_regression_fits": 0, "reproduced_baseline_forecasts": baseline_reproductions,
              "new_accounts": 0, "account_status": "NOT_RUN_PENDING_SEPARATE_ACCOUNT_PROTOCOL" if gate else "NOT_RUN_FAILED_RISK_GATE",
              "new_independent_market_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFCI新历史版本的风险预测用途增量。")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
