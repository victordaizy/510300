"""用已保存账本拆分回购信息、隔夜风险和五日以外续持，不重跑策略。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.funding_repurchase_demand_daily_v1 as trial
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_repurchase_horizon_loss_diagnostic_v1"
STUDY = "510300_REPURCHASE_HORIZON_LOSS_DIAGNOSTIC_V1"


def night_bucket(age):
    if age == 1:
        return "FIRST_NIGHT"
    return "NIGHTS_2_TO_5" if age <= 5 else "NIGHTS_AFTER_5"


def day_bucket(age):
    if age == 0:
        return "ENTRY_INTRADAY"
    return "INTRADAY_BEFORE_5_EXIT" if age < 5 else "INTRADAY_AFTER_5_EXIT"


def decompose(ledger, dividends):
    lots, entitlement = [], {}
    records = []
    prior_mark, prior_reserve = None, 0.
    events = dividends.to_dict("records")
    for row in ledger.itertuples():
        assert sum(q for _, q in lots) == row.shares_before
        pnl = defaultdict(float)
        if prior_mark is not None:
            for entry, quantity in lots:
                pnl[night_bucket(row.idx - entry)] += quantity * (row.open - prior_mark)
        dividend_total = 0.
        for k, event in enumerate(events):
            if event["ex_date"] == row.date:
                for entry, quantity in entitlement.get(k, []):
                    value = quantity * event["cash_dividend_per_share"]
                    pnl["DIVIDEND_" + night_bucket(row.idx - entry)] += value
                    dividend_total += value
        np.testing.assert_allclose(dividend_total, row.dividend_recognized, atol=1e-7, rtol=0)
        if row.filled_quantity > 0:
            lots.append((row.idx, row.filled_quantity))
        elif row.filled_quantity < 0:
            remaining, left = [], -row.filled_quantity
            for entry, quantity in lots:
                sold = min(quantity, left) if entry < row.idx else 0
                left -= sold
                if quantity > sold:
                    remaining.append((entry, quantity - sold))
            assert left == 0
            lots = remaining
        assert sum(q for _, q in lots) == row.shares
        for entry, quantity in lots:
            pnl[day_bucket(row.idx - entry)] += quantity * (row.mark - row.open)
        for k, event in enumerate(events):
            if event["record_date"] == row.date:
                entitlement[k] = list(lots)
        raw_price = sum(v for k, v in pnl.items() if not k.startswith("DIVIDEND_"))
        np.testing.assert_allclose(raw_price, row.price_pnl, atol=1e-7, rtol=0)
        expense = row.commission + row.slippage_cost + row.terminal_exit_reserve - prior_reserve
        net = sum(pnl.values()) - expense
        np.testing.assert_allclose(net, row.pnl, atol=1e-7, rtol=0)
        records.append({"date": row.date, "idx": row.idx, **pnl, "cost": expense,
                        "net_pnl": net, "max_remaining_lot_age": max((row.idx - entry for entry, _ in lots), default=0)})
        prior_mark, prior_reserve = row.mark, row.terminal_exit_reserve
    frame = pd.DataFrame(records).fillna(0.)
    numeric = frame.drop(columns=["date", "idx", "max_remaining_lot_age"])
    summary = numeric.sum().to_dict()
    summary["maximum_remaining_lot_age"] = int(frame.max_remaining_lot_age.max())
    summary["days_with_lot_after_five_day_exit"] = int(frame.max_remaining_lot_age.ge(5).sum())
    return frame, summary


def run():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("这份事后诊断已经固定，不能覆盖。")
    for folder in ["code", "results"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), trial.OUT / "result.json", trial.OUT / "results/scored_predictions.parquet",
             trial.demand.OUT / "results/confirmed_disclosure_events.json", trial.parent.DIVIDENDS]
    for name in trial.TRADED:
        for filename in ["ledger.parquet", "decisions.parquet", "cycles.parquet"]:
            paths.append(trial.OUT / "accounts/STRESS" / name / filename)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "status": "POST_RESULT_DIAGNOSTIC_NO_NEW_STRATEGY",
        "reason": "固定账户失败后，初步读取发现五日预测与43-59日持仓周期不一致、盘中盈利隔夜亏损，故拆分损失来源。",
        "preliminary_descriptive_results_already_seen": True,
        "method": "重建原引擎先进先出的成交批次；逐日拆首夜、第二至第五夜、第五夜以后及入场日/第五日退出前后盘中损益。分红按真实登记批次单独归属，费用与末端储备单列，逐日还原原账本。",
        "boundaries": "第五日指入场idx+5开盘。第五夜在原预测期内，第五日开盘后的盘中在原预测期外。批次归因不是提前可知信号，不是把旧份额提前卖出的收益反事实。",
        "source_question": "已确认信息中完成公告占比、报告截止日至可用日间隔和股票集中程度；间隔不是实际买入到披露的时长，后者通常未知。预算未用部分不推断为必然未来买盘。",
        "selection_question": "比较所有已知价格弱势、实际增仓和减仓原点的五日预测与成熟结果；这些条件集合不相同，不直接推断因果。",
        "new_accounts": 0, "new_fits": 0, "new_parameter_searches": 0, "orders_authorized": False,
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    events = pd.DataFrame([r for r in read(trial.demand.OUT / "results/confirmed_disclosure_events.json")
                           if r["normalization_status"] == "CONFIRMED_REPORTED_DEMAND_NORMALIZED"])
    events["lag_cutoff_to_decision_days"] = (pd.to_datetime(events.decision_date) - pd.to_datetime(events.economic_cutoff)).dt.days
    weights = events.groupby("symbol").strength.sum().sort_values(ascending=False)
    budgets = {"BELOW_REPORTED_FLOOR": 0, "AT_OR_ABOVE_REPORTED_FLOOR": 0, "BUDGET_UNKNOWN": 0}
    for row in events.to_dict("records"):
        budget = row["budget_terms"]
        key = "BUDGET_UNKNOWN" if not budget["unambiguous_pair"] else (
            "AT_OR_ABOVE_REPORTED_FLOOR" if row["cumulative_cents"] >= budget["floor_cents"] else "BELOW_REPORTED_FLOOR")
        budgets[key] += 1
    sources = {"normalized_disclosures": len(events), "classification_counts": events.classification.value_counts().to_dict(),
               "strength_by_classification": events.groupby("classification").strength.sum().to_dict(),
               "completion_share_of_strength": float(events.loc[events.classification.eq("COMPLETION"), "strength"].sum() / events.strength.sum()),
               "cutoff_date_known": int(events.economic_cutoff.notna().sum()),
               "cutoff_to_decision_lag_summary": events.lag_cutoff_to_decision_days.describe(percentiles=[.25, .5, .75, .9, .95]).to_dict(),
               "largest_company": weights.index[0], "largest_company_strength_fraction": float(weights.iloc[0] / weights.sum()),
               "top_five_strength_fraction": float(weights.iloc[:5].sum() / weights.sum()),
               "reported_budget_stage": budgets, "remaining_mandatory_buying_identified": False}
    scored = pd.read_parquet(trial.OUT / "results/scored_predictions.parquet")
    dividends = trial.engine.normalize_dividends(pd.read_csv(trial.parent.DIVIDENDS))
    attribution, selection = [], []
    for name in trial.TRADED:
        folder = trial.OUT / "accounts/STRESS" / name
        ledger = pd.read_parquet(folder / "ledger.parquet")
        decisions = pd.read_parquet(folder / "decisions.parquet")
        closed = pd.read_parquet(folder / "cycles.parquet")
        daily, totals = decompose(ledger, dividends)
        daily.to_parquet(OUT / "results" / f"{name}_daily_attribution.parquet", index=False)
        attribution.append({"model": name, **totals, "closed_cycles": len(closed),
                            "cycle_length_summary": closed.days.describe().to_dict() if len(closed) else None})
        joined = decisions.merge(scored[scored.model.eq(name)][["idx", "mu5", "gross_return5"]], on="idx", how="left", suffixes=("", "_prediction"))
        for subset, mask in [("KNOWN_PRICE_WEAKNESS", joined.pressure5.gt(0)),
                             ("ACTUAL_ADDITIONS", joined.filled_quantity.gt(0)),
                             ("ACTUAL_REDUCTIONS", joined.filled_quantity.lt(0))]:
            part = joined[mask & joined.gross_return5.notna()]
            selection.append({"model": name, "subset": subset, "n": len(part),
                              "mean_predicted_gross5": float(part.mu5_prediction.mean()),
                              "mean_mature_gross5": float(part.gross_return5.mean()),
                              "overlapping_labels_are_not_independent": True})
    save(OUT / "result.json", {"at": now(), "study_id": STUDY, "status": "DIAGNOSTIC_COMPLETED_NO_STRATEGY_PROMOTION",
        "source_diagnostics": sources, "account_attribution": attribution, "selection_diagnostics": selection,
        "daily_cash_identity_reconstructed_accounts": len(attribution), "new_accounts": 0, "new_fits": 0,
        "new_independent_forward_observations": 0, "orders_authorized": False, "goal_status": "active", "goal_achieved": False}, True)
    main = next(r for r in attribution if r["model"] == trial.PRIMARY)
    print({"逐日损益归因完成": len(attribution), "主账户": main, "来源": sources}, flush=True)


if __name__ == "__main__":
    run()
