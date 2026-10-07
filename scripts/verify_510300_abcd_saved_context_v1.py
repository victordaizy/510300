"""独立复算固定群体、日内隔夜定义与逐个决策；不新建研究账户。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(root):
    d = pd.read_parquet(root / "inputs/features.parquet")
    raw = pd.read_parquet(root / "inputs/cross_section.parquet")
    raw["date"] = pd.to_datetime(raw.date).dt.strftime("%Y-%m-%d")
    quality = pd.read_parquet(root / "inputs/attribution_quality.parquet")
    quality["date"] = pd.to_datetime(quality.date).dt.strftime("%Y-%m-%d")
    quality["weight_snapshot_date"] = pd.to_datetime(quality.weight_snapshot_date).dt.strftime("%Y-%m-%d")
    allowed = quality.set_index("date").valid_for_attribution & (quality.set_index("date").weight_snapshot_date < quality.set_index("date").index)
    allowed = allowed.reindex(d.date).fillna(False).to_numpy(bool)
    ctx = pd.read_parquet(root / "results/signal_daily_context.parquet")
    anchors = pd.read_csv(root / "results/signal_anchors.csv")
    members = pd.read_csv(root / "results/fixed_core_members.csv")
    returns = raw.pivot(index="date", columns="con_code", values="constituent_return_1d").reindex(d.date)
    returns.index = range(len(d))
    prior = d.close.shift(1)
    k = ((d.close - d.open) / prior - np.maximum(-(d.open - prior + d.dividend) / prior, 0)).rolling(3).mean()
    verified_context = verified_members = 0
    maximum_error = 0.0
    boundary_points = []
    for anchor in anchors.to_dict("records"):
        sample = ctx.loc[ctx.signal_id == anchor["signal_id"]]
        indices = sample.idx.to_numpy(int)
        want_c = k.iloc[indices].to_numpy() - k.iloc[int(anchor["setup_idx"])]
        assert np.allclose(want_c, sample.C_compensation_change, equal_nan=True, atol=1e-12)
        cohort = members.loc[members.signal_id == anchor["signal_id"]]
        if len(cohort):
            expected = raw.loc[(raw.date == anchor["setup_date"]) & raw.con_code.isin(cohort.con_code)].sort_values("con_code")
            cohort = cohort.sort_values("con_code")
            assert expected.con_code.tolist() == cohort.con_code.tolist()
            assert np.allclose(expected.snapshot_weight, cohort.snapshot_weight, atol=1e-12)
            assert expected.industry_l1.tolist() == cohort.industry_l1.tolist()
            verified_members += len(cohort)
        if anchor["anchor_status"] == "READY":
            w = cohort.snapshot_weight.to_numpy(float)
            a = returns[cohort.con_code.tolist()].to_numpy(float)
            known = np.isfinite(a)
            cov = (known * w).sum(axis=1) / w.sum()
            ratio = ((a > 0) * w).sum(axis=1) / w.sum()
            ratio[(cov < .98 - 1e-12) | ~allowed] = np.nan
            b = pd.Series(ratio).rolling(3).mean()
            expected_b = b.iloc[indices].to_numpy() - b.iloc[int(anchor["setup_idx"])]
            assert np.allclose(expected_b, sample.B_participation_change, equal_nan=True, atol=1e-12)
            finite = np.isfinite(expected_b)
            if finite.any():
                maximum_error = max(maximum_error, float(np.abs(expected_b[finite] - sample.B_participation_change.to_numpy()[finite]).max()))
            # 对实际确认和三个后续位置只截取截至当时的数据，逐点重算末三日。
            chosen = sorted(set([int(anchor["signal_idx"]), *[int(indices[j]) for j in [len(indices)//3, 2*len(indices)//3, len(indices)-1]]]))
            for i in chosen:
                prefix = pd.Series(ratio[:i + 1]).rolling(3).mean()
                x = prefix.iloc[-1] - prefix.iloc[int(anchor["setup_idx"])]
                y = sample.loc[sample.idx == i, "B_participation_change"].iloc[0]
                assert (pd.isna(x) and pd.isna(y)) or abs(x - y) < 1e-12
                boundary_points.append({"signal_id": anchor["signal_id"], "date": d.date.iloc[i], "max_input_idx": i, "pass": True})
        verified_context += len(sample)
    lookup = ctx.set_index(["signal_id", "idx"])
    decision_checks, summaries = 0, []
    for path in (root / "results/accounts").glob("*/*/*"):
        period, cost, policy = path.relative_to(root / "results/accounts").parts
        if policy not in ["A", "B", "C", "D", "A_COVERAGE"]:
            continue
        decision = pd.read_parquet(path / "decisions.parquet")
        for row in decision.loc[decision.decision_type == "ENTRY"].to_dict("records"):
            key = (row["signal_id"], int(row["idx"]))
            v = lookup.loc[key]
            if row["reason"] == "NO_VIEW_ENTRY":
                assert not bool(v.joint_available)
            elif row["reason"] == "INCREMENT_ENTRY_REJECTED":
                assert bool(v.joint_available) and not bool(v[policy])
            elif row["reason"] in ["ENTRY_ACCEPTED", "CAPITAL_OCCUPIED_OR_PRIORITY"] and policy != "A":
                assert bool(v.joint_available) and (policy == "A_COVERAGE" or bool(v[policy]))
            decision_checks += 1
        for (kind, reason), g in decision.groupby(["decision_type", "reason"]):
            summaries.append({"period": period, "cost": cost, "policy": policy, "decision_type": kind, "reason": reason, "count": len(g)})
    pd.DataFrame(summaries).to_csv(root / "results/decision_attribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(boundary_points).to_csv(root / "results/internal_prefix_checks.csv", index=False, encoding="utf-8-sig")
    receipt = {"status": "PASS_INDEPENDENT_SAVED_CONTEXT_AND_DECISION_RECOMPUTATION", "context_rows": verified_context,
               "frozen_member_rows": verified_members, "entry_decisions": decision_checks, "prefix_points": len(boundary_points),
               "maximum_B_recomputation_error": maximum_error, "new_accounts": 0, "new_model_fits": 0, "new_downloads": 0,
               "source_clock_limitation": "验证截至归因日的输入索引，不认证历史网页首发与月末快照发布时间。"}
    (root / "context_verification.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读复核510300增量研究")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "reports/research/510300_all_research_abcd_increment_v1")
    main(parser.parse_args().root.resolve())
