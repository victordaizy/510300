"""从保存账户归因，不重跑策略；包含原赢家被截断和原8描述案例的真实时钟。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import industry_leader_failure_study_v1 as study


def saved(period, cost, policy):
    if policy in study.CONTROLS[:2]:
        return study.saved_account(period, cost, policy)
    folder = study.OUT / "accounts" / period / cost / policy
    result = {name: pd.read_parquet(folder / f"{name}.parquet") for name in
        (*study.ACCOUNT_TABLES, "industry_checks", "industry_group_checks")}
    result["terminal"] = study.read(folder / "terminal.json")
    return result


def normalize(trades):
    frame = trades.copy()
    for name in ("entry_origin", "entry_date", "exit_date"):
        frame[name] = pd.to_datetime(frame[name]).astype("datetime64[ns]")
    if frame.entry_origin.duplicated().any():
        raise ValueError("周期原信号身份重复。")
    return frame


def main():
    out, parent = study.OUT, study.parent
    if (out / "post_run_diagnosis.json").exists():
        raise RuntimeError("实际进入行业失效诊断已经保存，不覆盖。")
    summary = study.read(out / "summary.json")
    accounts, attribution, changes, extra_rows, entry_types = {}, [], [], [], []
    for period in parent.PERIODS:
        for cost in parent.COSTS:
            original = saved(period, cost, study.CONTROL_STAGE)
            primary = saved(period, cost, study.candidate.POLICIES[0])
            accounts[(period, cost)] = primary
            t, old = normalize(primary["trades"]), normalize(original["trades"])
            columns = ["entry_origin", "cycle_id", "entry_date", "exit_date", "status", "net_pnl", "net_return", "entry_quantity", "source"]
            merged = t[columns].merge(old[columns], on="entry_origin", how="outer", suffixes=("_new", "_stage"), indicator=True, validate="one_to_one")
            merged["period"], merged["cost"] = period, cost
            changes.append(merged)
            check = primary["industry_checks"].merge(t[["cycle_id", "entry_origin", "status", "net_pnl", "exit_date"]], on="cycle_id", how="left", validate="one_to_one")
            check = check.merge(old[["entry_origin", "net_pnl", "exit_date", "status"]].rename(columns={
                "net_pnl": "stage_net_pnl", "exit_date": "stage_exit_date", "status": "stage_status"}), on="entry_origin", how="left", validate="one_to_one")
            extra = check.loc[check.extra_exit].assign(period=period, cost=cost)
            extra_rows.append(extra)
            earlier = (merged._merge.eq("both") & merged.status_new.eq("COMPLETE") & merged.status_stage.eq("COMPLETE")
                & merged.exit_date_new.lt(merged.exit_date_stage))
            cut_winners = merged.loc[earlier & merged.net_pnl_stage.gt(0)]
            completed_new = float(t.loc[t.status.eq("COMPLETE"), "net_pnl"].sum())
            completed_old = float(old.loc[old.status.eq("COMPLETE"), "net_pnl"].sum())
            open_delta = primary["terminal"]["open_pnl_cny"] - original["terminal"]["open_pnl_cny"]
            ending_delta = float(primary["daily"].equity.iloc[-1] - original["daily"].equity.iloc[-1])
            if abs(completed_new - completed_old + open_delta - ending_delta) > 1e-6:
                raise AssertionError("完成周期、开放损益与完整账户财富差不一致。")
            for route, values in t.loc[t.status.eq("COMPLETE")].groupby("source", sort=True):
                entry_types.append({"period": period, "cost": cost, "entry_route": route, "completed": len(values),
                    "wins": int(values.net_pnl.gt(0).sum()), "losses": int(values.net_pnl.lt(0).sum()),
                    "net_pnl_cny": float(values.net_pnl.sum()), "role": "原路线说明，不根据此表反选路线或组合"})
            d = primary["daily"]
            gross = d.price_pnl.to_numpy(float) + d.dividend_accrual.to_numpy(float)
            no_cost_nav = 200000 + np.cumsum(gross)
            no_cost_returns = no_cost_nav / np.r_[200000., no_cost_nav[:-1]] - 1
            bound = parent.measurements.return_statistics(no_cost_returns)
            fees = float(d.commission.sum() + d.slippage.sum())
            net = float(d.equity.iloc[-1] - 200000)
            if abs(gross.sum() - fees - net) > 1e-6:
                raise AssertionError("毛损益、摩擦和净财富归因错误。")
            attribution.append({"period": period, "cost": cost, "ending_equity_delta_vs_stage": ending_delta,
                "completed_pnl_delta_vs_stage": completed_new-completed_old, "open_pnl_delta_vs_stage": open_delta,
                "shared_entry_origins": int(merged._merge.eq("both").sum()), "new_entry_origins": int(merged._merge.eq("left_only").sum()),
                "omitted_stage_entry_origins": int(merged._merge.eq("right_only").sum()), "earlier_complete_exits": int(earlier.sum()),
                "cut_stage_winners": len(cut_winners), "cut_winner_origins": cut_winners.entry_origin.tolist(),
                "cut_winner_net_pnl_delta_cny": float((cut_winners.net_pnl_new-cut_winners.net_pnl_stage).sum()),
                "extra_requests": len(extra), "actual_gross_pnl": float(gross.sum()), "total_friction": fees, "actual_net_pnl": net,
                "zero_cost_fixed_actual_quantities_cagr": bound["net_cagr"], "zero_cost_fixed_actual_quantities_sharpe": bound["net_sharpe"],
                "zero_cost_role": "解释上界：固定实际数量和持有路径，非可执行策略或重新评估资金风险。",
                "actual_rejections": len(primary["rejections"]), "actual_account_risk_stopped": primary["terminal"]["stopped"],
                "actual_risk_reduction_orders": int(primary["orders"].reason.eq("PRIOR_CLOSE_RISK_REDUCTION").sum())})
    study.table("全部原信号周期与阶段对照_新增遗漏及开放保留", pd.concat(changes, ignore_index=True))
    study.table("全部实际额外退出_原周期盈亏与失效信息", pd.concat(extra_rows, ignore_index=True))
    study.table("全部四场景完整财富差与毛损益摩擦归因", pd.DataFrame(attribution))
    study.table("全部原进入路线完成周期说明_不反选路线", pd.DataFrame(entry_types))
    profiles = pd.read_parquet(study.description.OUT / "results/全部143事件三个固定时点_行业与原股票对照.parquet")
    old_eight = profiles.loc[profiles.original_period.eq("2020_2026") & profiles.original_account_context.eq("COMPLETE")
        & profiles.relative_session.eq(5) & profiles.descriptive_fixed_state.eq("ETF_POSITIVE_FIXED_LEADERS_NONPOSITIVE")].copy()
    if len(old_eight) != 8:
        raise ValueError("原八说明性周期范围改变。")
    actual = accounts[("2020_2026", "STRESS")]
    t = normalize(actual["trades"])
    check = actual["industry_checks"].merge(t[["cycle_id", "entry_origin"]], on="cycle_id", validate="one_to_one")
    check = check.rename(columns={name: "actual_" + name for name in check.columns if name != "entry_origin"})
    old_eight["entry_origin"] = pd.to_datetime(old_eight.anchor_date).astype("datetime64[ns]")
    check["entry_origin"] = pd.to_datetime(check.entry_origin).astype("datetime64[ns]")
    correspondence = old_eight.merge(check, on="entry_origin", how="left", validate="one_to_one")
    correspondence["actual_check_status"] = np.select([
        correspondence.actual_decision_date.isna(), correspondence.actual_extra_exit.eq(True),
        correspondence.actual_leader_view_allowed.eq(False)],
        ["NOT_REACHED_FIFTH_CHECK_OR_NOT_ENTERED", "ACTUAL_EXTRA_EXIT", "ACTUAL_LEADER_UNKNOWN"],
        default="ACTUAL_KNOWN_NO_EXTRA_EXIT")
    study.table("原八描述周期与实际进入五日检验_全部对应", correspondence)
    study.write(out / "post_run_diagnosis.json", {"at": parent.original.now(), "result": summary["status"],
        "attribution": attribution, "original_eight_actual_clock_states": correspondence.actual_check_status.value_counts().to_dict(),
        "original_eight_rows": len(correspondence), "all_entry_route_contexts": entry_types,
        "new_accounts_in_diagnosis": 0, "repeated_financial_runs": 0, "goal_achieved": False})
    print("已保存全部财富差/真实退出/原八周期时钟/进入路线及固定数量无费用解释，未重跑账户。", flush=True)


if __name__ == "__main__":
    main()
