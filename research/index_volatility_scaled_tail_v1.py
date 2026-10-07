"""同一两年风险样本按事前20日波动调整尺度，先评价风险预测再决定是否计算账户。"""
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
from research.nfci_tail_forecast_increment_v1 import fz0, interval

OUT = ROOT / "reports/research/510300_index_volatility_scaled_tail_v1"
PREVIOUS = ROOT / "reports/research/510300_nfci_tail_forecast_increment_v1"
STUDY = "510300_INDEX_VOLATILITY_SCALED_TAIL_V1"
PRIMARY = "VOL20_FILTERED"


def filtered(data, labels, t, baseline):
    rows = labels.loc[baseline["label_indices"]]
    assert rows.exit_index.lt(t).all()
    assert rows.origin.ge(data.date.iloc[t] - pd.DateOffset(years=2)).all()
    historical_vol = data.vol20.iloc[rows.origin_index.to_numpy(int)].to_numpy(float)
    current_vol = float(data.vol20.iloc[t])
    if not (np.isfinite(historical_vol).all() and (historical_vol > 0).all() and np.isfinite(current_vol) and current_vol > 0):
        raise ValueError("事前波动缺失，不能移除个别历史风险场景")
    returns = rows.gross_return5.to_numpy(float)
    if not (returns > -1).all():
        raise ValueError("含分红收益不支持对数风险场景")
    scaled = np.expm1(np.log1p(returns) * current_vol / historical_vol)
    if not np.isfinite(scaled).all():
        raise ValueError("尺度变换产生无效风险场景")
    tail = np.sort(scaled)[:max(1, int(np.ceil(.05 * len(scaled))))]
    return {**baseline, "policy": PRIMARY, "q05": float(np.quantile(scaled, .05)),
            "es95": max(0., -float(tail.mean())), "current_vol20": current_vol,
            "minimum_scale_ratio": float(np.min(current_vol/historical_vol)),
            "maximum_scale_ratio": float(np.max(current_vol/historical_vol))}


def prepare():
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    labels = daily.five_day_labels(data, normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))).set_index("origin_index", drop=False)
    baselines = [r for r in read(PREVIOUS / "forecasts.json") if r["policy"] == "BASELINE"]
    return data, labels, baselines


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("尺度预测实验已固定，不覆盖")
    data, labels, baselines = prepare()
    record = next(r for r in baselines if r["origin_index"] >= 1600)
    t = record["origin_index"]
    actual = filtered(data, labels, t, record)
    changed = labels.copy()
    outside = ~changed.origin_index.isin(record["label_indices"])
    changed.loc[outside, "gross_return5"] = 10000.
    assert filtered(data, changed, t, record) == actual
    future = data.copy()
    future.loc[t+1:, "vol20"] = 9999.
    assert filtered(future, labels, t, record) == actual
    flat_vol = data.copy()
    flat_vol["vol20"] = .2
    unchanged = filtered(flat_vol, labels, t, record)
    np.testing.assert_allclose([unchanged["q05"], unchanged["es95"]], [record["q05"], record["es95"]], atol=1e-14, rtol=0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir()
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    protocol = {
        "at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "原五日损失预测对市场波动变化反应不足，是否可由同一风险样本的事前波动尺度调整改善。",
        "economic_mechanism": "当前风险变化时，不直接把两年内高低波动场景的原幅度等同于今天的幅度。仍保留非正态的历史场景，跳空压力预算另行保留。",
        "formula": "r5_scenario_at_t = exp(log(1+r5_at_s) * vol20_at_t / vol20_at_s) - 1；s为历史决策原点，历史和当前vol20均只使用相应原点之前及当日已收盘数据。",
        "method_scope": "受过滤历史模拟启发的固定五日尺度近似，不声称精确估计未来五日波动路径。对数变换保持收益>-100%。",
        "old_related_work": "simple_core_window_diagnostic_v1曾测试按20日波动直接缩放账户仓位；dense_probability_payoff_nodes_v1曾测试十日幅度均值。此处只评价同一五日风险分布的VaR/ES，不重跑这些旧仓位或均值用途。",
        "fixed": ["最近两个日历年每日更新", "原动量条件和60行最低数", "同一标签和成熟时点", "同一5%分位数与尾部均值公式", "20日波动窗口固定，不扫描替代窗口"],
        "source": "https://www.bankofengland.co.uk/working-paper/2015/filtered-historical-simulation-value-at-risk-models-and-their-competitors",
        "evaluation": "与刚保存的BASELINE逐原点配对；2015-2019与2020-2026-09-16分别评价；固定五交易日相位，不选最有利相位；全部日期预测保留。",
        "gate": "两个区间FZ0联合风险评分差的95%区间上界均<0，平均分位数损失均不增加，且实际VaR跌破率均<=10%；通过后才另立账户检验，失败不运行账户。",
        "statistics": "4个非重叠观察的区块自助法，2000次，种子20260925；未校正项目级反复选择。",
        "selection_history": "本问题是在看到原风险模型前期校准偏差后提出，属于开发研究，不能改称独立确认。",
        "parameter_search": 0, "new_regression_fits": 0, "new_accounts_before_gate": 0,
        "checks": {"excluded_labels_cannot_change_prediction": True, "future_volatility_cannot_change_prediction": True,
                   "constant_volatility_reproduces_unscaled_distribution": True},
        "goal_achieved": False, "orders_authorized": False,
    }
    save(OUT / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update([LATEST / "candidate_features.parquet", ROOT / "data/reference/510300_dividends.csv",
                    PREVIOUS / "forecasts.json", PREVIOUS / "mature_forecast_scores.parquet"])
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    print("同一两年样本的20日波动尺度检验已固定；没有参数搜索或账户优化。", flush=True)


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, "来源改变：" + name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    data, labels, baselines = prepare()
    predictions = [filtered(data, labels, r["origin_index"], r) for r in baselines]
    save(OUT / "forecasts.json", predictions, True)
    base = pd.read_parquet(PREVIOUS / "mature_forecast_scores.parquet")
    base = base[base.policy.eq("BASELINE")].sort_values("origin_index").reset_index(drop=True)
    by_origin = {r["origin_index"]: r for r in predictions}
    new = base.copy()
    new["policy"] = PRIMARY
    for key in ["q05", "es95"]:
        new[key] = [by_origin[t][key] for t in new.origin_index]
    new["fz0"] = fz0(new.return5, new.q05, -new.es95)
    new["pinball"] = (.05 - new.return5.lt(new.q05).astype(float)) * (new.return5-new.q05)
    new["breach"] = new.return5.lt(new.q05)
    new["used_macro_condition"] = False
    score = pd.concat([base, new], ignore_index=True)
    score.to_parquet(OUT / "mature_forecast_scores.parquet", index=False)
    phase = score[score.fixed_nonoverlap_phase]
    statistics, comparisons, annual = [], [], []
    rng = np.random.default_rng(20260925)
    for period in ["earlier", "main"]:
        b = phase[phase.period.eq(period) & phase.policy.eq("BASELINE")].sort_values("origin_index")
        a = phase[phase.period.eq(period) & phase.policy.eq(PRIMARY)].sort_values("origin_index")
        np.testing.assert_array_equal(a.origin_index, b.origin_index)
        for part in [b, a]:
            statistics.append({"period": period, "policy": part.policy.iloc[0], "observations": len(part),
                "mean_fz0": float(part.fz0.mean()), "mean_pinball": float(part.pinball.mean()),
                "breaches": int(part.breach.sum()), "breach_rate": float(part.breach.mean())})
        comparisons.append({"period": period, "policy": PRIMARY,
            "pinball_mean_difference": float(a.pinball.mean()-b.pinball.mean()), "breach_rate": float(a.breach.mean()),
            **interval(a.fz0.to_numpy()-b.fz0.to_numpy(), rng)})
    for (year, policy), part in phase.groupby([phase.origin.dt.year, "policy"], sort=True):
        annual.append({"year": int(year), "policy": policy, "observations": len(part),
                       "mean_fz0": float(part.fz0.mean()), "mean_pinball": float(part.pinball.mean()),
                       "breach_rate": float(part.breach.mean())})
    save(OUT / "yearly_risk_scores.json", annual, True)
    gate = all(r["upper_95"] < 0 and r["pinball_mean_difference"] <= 0 and r["breach_rate"] <= .10 for r in comparisons)
    result = {"at": now(), "study_id": STUDY, "status": "RISK_FORECAST_INCREMENT_SUPPORTED_ACCOUNT_PENDING" if gate else "FROZEN_NO_RELIABLE_SCALED_TAIL_INCREMENT",
        "continuation_gate": gate, "period_scores": statistics, "comparisons": comparisons,
        "daily_risk_estimates": len(predictions), "new_regression_fits": 0, "new_accounts": 0,
        "account_status": "NOT_RUN_PENDING_SEPARATE_ACCOUNT_PROTOCOL" if gate else "NOT_RUN_FAILED_RISK_GATE",
        "goal_achieved": False, "orders_authorized": False, "new_independent_market_observations": 0}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="五日尾部风险的固定20日波动尺度检验。")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
