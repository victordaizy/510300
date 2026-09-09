"""只比较保存实际持仓与保存参考的退出输入，不创建策略账户。"""
import json
from bisect import bisect_right
from pathlib import Path
import numpy as np
import pandas as pd
from research.learned_cycle_exit_v1 import FEATURES, state_values, predict
from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import now, require, write_json, digest

ROOT = Path(__file__).resolve().parents[1]
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
OUT = ROOT / "reports/research/510300_saved_reference_own_exit_states_20260908"


def main():
    require(not (OUT / "result.json").exists(), "保存退出状态诊断已完成")
    cfg = json.loads((ROOT / "config/510300_continuous_reference_min_variance_v1.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    dates = pd.DatetimeIndex(data.date)
    factors = pd.read_parquet(P91 / "evaluation_factors.parquet").set_index("date")
    references = pd.read_parquet(P91 / "continuous_references/BASE/REARM_RIDGE_decisions.parquet").set_index("origin")
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    fits = [m["fit_index"] for m in models]
    ex_events, record_events = {}, {}
    for key, event in enumerate(dividends.itertuples()):
        ex_events.setdefault(event.ex_date, []).append((key, event.cash_dividend_per_share))
        record_events.setdefault(event.record_date, []).append(key)
    rows, summary = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            ledger = pd.read_parquet(P91 / period / cost / "CONTINUOUS_REFERENCE_MIN_VARIANCE_ledger.parquet")
            owned_records, cycle, number, previous_shares = {}, None, 0, 0
            local = []
            for row in ledger.itertuples():
                t = dates.get_loc(row.date)
                # 先按当初真实登记的周期归属分红，不能把旧周期应收转给新买入。
                if cycle is not None:
                    for key, amount in ex_events.get(row.date, []):
                        owner, shares = owned_records.get(key, (None, 0))
                        if owner == cycle["cycle_id"]:
                            cycle["dividend"] += shares*amount
                if previous_shares == 0 and row.shares > 0:
                    number += 1
                    origin = factors.loc[row.origin]
                    pure = origin.panic_state == 0 and origin.learned_state == 1 and origin.learned_budget > 0
                    cycle = {"cycle_id": number, "entry_index": t, "entry_cost_cny": row.notional+row.commission,
                        "mode": 1, "entry_date": row.date, "entry_quantity": row.shares, "dividend": 0.,
                        "peak": row.notional+row.commission, "single_entry": True, "pure_learned_entry": bool(pure), "negative_count": 0}
                elif previous_shares > 0 and row.shares == 0:
                    cycle = None
                elif previous_shares > 0 and row.shares > 0 and row.filled_quantity != 0:
                    cycle["single_entry"] = False
                for key in record_events.get(row.date, []):
                    owned_records[key] = (cycle["cycle_id"] if cycle else None, row.shares)
                previous_shares = row.shares
                if cycle is None or row.mark_clock == "OPEN_TERMINAL":
                    continue
                value = row.shares*row.mark+cycle["dividend"]
                cycle["peak"] = max(cycle["peak"], value)
                ref = references.loc[row.date]
                record = {"period": period, "cost": cost, "date": row.date, "own_cycle": cycle["cycle_id"], "own_entry_date": cycle["entry_date"], "own_shares": row.shares,
                    "single_entry_so_far": cycle["single_entry"], "pure_learned_at_entry": cycle["pure_learned_entry"], "reference_learning_status": ref.learning_status,
                    "status": "NOT_COMPARABLE_MULTIPLE_FILLS_OR_OTHER_ENTRY", "own_prediction": np.nan, "reference_prediction": ref.continuation_prediction,
                    "sign_disagreement": False, "confirmed_exit_disagreement": False}
                eligible = cycle["single_entry"] and cycle["pure_learned_entry"] and pd.notna(ref.learning_cycle_id)
                if eligible:
                    stored_index = bisect_right(fits, t)-1
                    stored = models[stored_index] if stored_index >= 0 else None
                    values = state_values(data, t, cycle, value, cycle["peak"])
                    record.update({"own_"+name: float(v) for name, v in zip(FEATURES, values)})
                    if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
                        require(stored["fit_index"] <= t and stored["latest_exit_index"] <= stored["fit_index"], "保存退出模型成熟时钟错误")
                        estimate = predict(stored["model"], values)
                        reference_values = ref[FEATURES].to_numpy(float)
                        reference_estimate = predict(stored["model"], reference_values)
                        require(abs(reference_estimate-ref.continuation_prediction) < 1e-12, "原参考保存预测无法按原系数重现")
                        cycle["negative_count"] = cycle["negative_count"]+1 if estimate < 0 else 0
                        record.update(status="COMPARABLE_SAVED_MODEL_AND_SINGLE_ACTUAL_ENTRY", own_prediction=estimate,
                            reference_prediction=reference_estimate, own_negative_count=cycle["negative_count"], reference_negative_count=ref.negative_confirmation_count,
                            sign_disagreement=bool((estimate < 0) != (reference_estimate < 0)),
                            confirmed_exit_disagreement=bool((cycle["negative_count"] >= cfg["confirmation_days"]) != ref.learned_exit_requested),
                            prediction_difference=estimate-reference_estimate,
                            entry_age_difference=float(values[0]-reference_values[0]), return_input_difference=float(values[1]-reference_values[1]), drawdown_input_difference=float(values[2]-reference_values[2]))
                    else:
                        cycle["negative_count"] = 0
                        record["status"] = "NO_VIEW_NO_MATURE_MODEL_OR_FEATURE"
                else:
                    cycle["negative_count"] = 0
                local.append(record)
            rows.extend(local)
            frame = pd.DataFrame(local)
            valid = frame[frame.status.eq("COMPARABLE_SAVED_MODEL_AND_SINGLE_ACTUAL_ENTRY")]
            summary.append({"period": period, "cost": cost, "holding_closes": len(frame), "comparable_states": len(valid), "comparable_cycles": int(valid.own_cycle.nunique()),
                "sign_disagreements": int(valid.sign_disagreement.sum()), "confirmation_disagreements": int(valid.confirmed_exit_disagreement.sum()),
                "maximum_absolute_prediction_difference": float(valid.prediction_difference.abs().max()) if len(valid) else None,
                "status_counts": frame.status.value_counts().to_dict()})
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.DataFrame(rows)
    full.to_parquet(OUT / "saved_exit_state_comparison.parquet", index=False)
    full.to_csv(OUT / "全部实际与参考退出状态比较.csv", index=False, encoding="utf-8-sig")
    full[full.sign_disagreement | full.confirmed_exit_disagreement].to_csv(OUT / "已有预测判断差异.csv", index=False, encoding="utf-8-sig")
    result = {"completed_at": now(), "status": "SAVED_INPUT_DIFFERENCE_DIAGNOSTIC_ONLY", "accounts_read": 4, "new_accounts": 0, "new_model_fits": 0,
        "counterfactual_returns_computed": False, "summary": summary, "position_impact": 0, "goal_achieved": False,
        "interpretation": "仅比较当时只有一次实际买入且纯学习支路启动的可识别状态；有部分交易后不重新定义总市值回撤。差异不代表错误或改动会获利，没有反事实账户收益。", "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
