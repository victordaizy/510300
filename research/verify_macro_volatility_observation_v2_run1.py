"""以原始数组独立复算已保存观察标签和匹配差异，不新建策略或模型。"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT if (ROOT / "inputs/market_daily.csv").exists() else ROOT / "reports/research/510300_macro_volatility_observation_v2_run1"


def main():
    m = pd.read_csv(STUDY / "inputs/market_daily.csv")
    monthly = pd.read_csv(STUDY / "results/104个月_完整观察.csv")
    weekly = pd.read_csv(STUDY / "results/固定周度_完整观察.csv")
    a = pd.read_csv(STUDY / "inputs/official_releases.csv")
    dates = pd.to_datetime(m.date).to_numpy()
    close, opens, dividend = (m[x].to_numpy() for x in ["close", "open", "dividend"])
    returns = np.r_[np.nan, (close[1:] + dividend[1:]) / close[:-1] - 1]
    wealth = np.cumprod(1 + np.nan_to_num(returns))
    checked, mature, pending, features_checked = 0, 0, 0, 0
    errors = []

    def same(actual, expected, description):
        if pd.isna(expected) and pd.isna(actual):
            return
        if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > 2e-11:
            errors.append({"field": description, "actual": float(actual), "expected": float(expected)})

    for frame in [monthly, weekly]:
        for row in frame.to_dict("records"):
            snap = pd.Timestamp(row["snapshot_at"])
            if pd.notna(row.get("stat_month")):
                assert pd.Timestamp(row["available_at_upper_bound"]) <= snap
            if row["origin"] == "周度":
                eligible = a.loc[pd.to_datetime(a.available_at_upper_bound) <= snap]
                assert (pd.isna(row.get("stat_month")) and eligible.empty) or row["stat_month"] == eligible.iloc[-1].stat_month
            i = int(np.searchsorted(dates, np.datetime64(snap.tz_localize(None).normalize()), side="right") - 1)
            unavailable = snap.tz_localize(None).normalize() > pd.Timestamp(dates[-1])
            if unavailable:
                assert pd.isna(row["observation_date"]) and pd.isna(row["rv20"])
            else:
                past = returns[i - 19:i + 1]
                expected = {
                    "rv20": np.std(past, ddof=1) * math.sqrt(252),
                    "downside20": math.sqrt(np.minimum(past, 0).dot(np.minimum(past, 0)) * 252 / 20),
                    "past_return20": wealth[i] / wealth[i - 20] - 1,
                    "past_return60": wealth[i] / wealth[i - 60] - 1,
                }
                for key, value in expected.items():
                    same(row[key], value, row["origin_id"] + ":" + key)
                    features_checked += 1
            for delay in [0, 1]:
                entry = i + delay + 1 if not unavailable else len(m)
                for horizon in [5, 20, 60]:
                    prefix = f"E{delay}_{horizon}"
                    end = entry + horizon - 1
                    checked += 1
                    if end >= len(m):
                        assert pd.isna(row[prefix + "_return"])
                        assert row[prefix + "_status"] == "PENDING_MARKET_WINDOW"
                        pending += 1
                        continue
                    held_dividends = np.r_[0., np.cumsum(dividend[entry + 1:end + 1])]
                    path = (close[entry:end + 1] + held_dividends) / opens[entry]
                    anchored = np.r_[1., path]
                    values = {"return": path[-1] - 1, "worst_close": path.min() - 1,
                              "worst_path": min(path.min() - 1, 0), "best_path": max(path.max() - 1, 0),
                              "drawdown": (anchored / np.maximum.accumulate(anchored) - 1).min()}
                    for metric, value in values.items():
                        same(row[prefix + "_" + metric], value, row["origin_id"] + ":" + prefix + "_" + metric)
                    assert row[prefix + "_entry_date"] == m.at[entry, "date"]
                    assert row[prefix + "_exit_date"] == m.at[end, "date"]
                    mature += 1
    expected_weekly = pd.Series(pd.to_datetime(m.date)).loc[lambda x: x >= "2018-01-01"].groupby(lambda i: pd.Timestamp(dates[i]).to_period("W-FRI")).max().dt.strftime("%Y-%m-%d").tolist()
    assert expected_weekly == weekly.observation_date.tolist()
    pairs = pd.read_csv(STUDY / "results/历史近邻_全部配对.csv")
    matched = pd.read_csv(STUDY / "results/历史近邻_全部目标与缺口.csv")
    look = weekly.set_index("origin_id")
    for (question, target_id), part in pairs.groupby(["question", "target_id"]):
        target = look.loc[target_id]
        assert len(part) == 3 and part.control_cycle.nunique() == 3
        control = look.loc[part.control_id]
        assert control.training_regime.eq(target.training_regime).all()
        assert (part.control_label_end < target.observation_date).all()
        assert control.stat_month.ne(target.stat_month).all() and part.distance.le(1 + 1e-10).all()
        if question == "H2":
            assert target.downside_change5 < -1e-10 and control.downside_change5.ge(-1e-10).all()
        else:
            assert target.delta3_spread_pp > 1e-10 and control.delta3_spread_pp.le(1e-10).all()
        saved = matched.loc[(matched.question == question) & (matched.origin_id == target_id)].iloc[0]
        for delay in [0, 1]:
            for metric in ["return", "worst_close", "drawdown"]:
                col = f"E{delay}_20_{metric}"
                same(saved[col + "_difference"], target[col] - control[col].mean(), target_id + ":配对" + col)
    u = pd.read_csv(STUDY / "results/主20日_固定比较与区间.csv")
    draws = np.load(STUDY / "results/fixed_block_draws.npz")
    for row in u.to_dict("records"):
        universe = monthly.loc[monthly.training_regime == row["training_regime"]].set_index("stat_month")
        cycles = universe.index
        if row["question"] in ["H1", "H3"]:
            group, first, second = ("arithmetic_group", "M1侧主导改善", "M2侧主导改善") if row["question"] == "H1" else ("improvement_prior_group", "改善且此前上涨", "改善且此前未涨")
            col = row["delay"] + "_20_return"
            x = universe[col].where(universe[group] == first).to_numpy()
            y = universe[col].where(universe[group] == second).to_numpy()
        else:
            part = matched.loc[(matched.question == row["question"]) & (matched.training_regime == row["training_regime"]) & (matched.status == "MATCHED")]
            col = row["delay"] + "_20_" + row["metric"] + "_difference"
            x = part.groupby("stat_month")[col].mean().reindex(cycles).to_numpy()
            y = np.where(np.isfinite(x), 0., np.nan)
        same(row["mean_difference"], np.nanmean(x) - np.nanmean(y), row["question"] + ":周期等权差异")
        index = draws[row["training_regime"]]
        samples = []
        for selected in index:
            v1, v0 = x[selected], y[selected]
            if np.isfinite(v1).any() and np.isfinite(v0).any():
                samples.append(np.nanmean(v1) - np.nanmean(v0))
        same(row["ci_low"], np.quantile(samples, .025), row["question"] + ":保存抽样下界")
        same(row["ci_high"], np.quantile(samples, .975), row["question"] + ":保存抽样上界")
    reference = json.loads((STUDY / "results/reference_recomputation.json").read_text(encoding="utf-8"))
    assert not reference["mismatches"]
    correlation = pd.read_csv(STUDY / "results/全部连续关联.csv")
    for row in correlation.to_dict("records"):
        frame = monthly if row["origin"] == "月度" else weekly
        part = frame.loc[frame.training_regime == row["training_regime"]]
        if row["period"] != "全期":
            part = part.loc[part.period == row["period"]]
        z = part[[row["x"], row["y"]]].dropna()
        expected = z.iloc[:, 0].corr(z.iloc[:, 1], method="spearman") if len(z) >= 3 and z.iloc[:, 0].nunique() > 1 and z.iloc[:, 1].nunique() > 1 else np.nan
        same(row["spearman"], expected, "保存CSV关联:" + row["x"] + ":" + row["y"])
    receipt = {"status": "PASS" if not errors else "FAIL", "origin_rows": len(monthly) + len(weekly),
               "label_windows_checked": checked, "mature_windows": mature, "pending_windows": pending,
               "independent_feature_checks": features_checked, "historical_match_pairs": len(pairs),
               "saved_block_comparisons": len(u), "saved_csv_correlations": len(correlation), "errors": errors,
               "new_accounts": 0, "new_models": 0, "new_bootstrap_draws": 0, "downloads": 0,
               "scope": "复算保存标签、价格特征、时钟、匹配差异和固定抽样；不是外部审阅、独立验证或全源真实性认证。"}
    (STUDY / "verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    assert not errors


if __name__ == "__main__":
    main()
