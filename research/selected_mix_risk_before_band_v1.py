"""先形成风险可行目标，再应用固定调仓带；保留强制减险优先级。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.selected_mix_daily_two_year_v1 as parent
import research.selected_mix_pooled_daily_v1 as pooled
from research.selected_mix_reappraisal_v1 import read, save, digest, local_import_closure, LATEST
from research.adaptive_allocation_v1 import normalize_dividends
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account
from research.strategy_review_diagnostics_v1 import metrics

STUDY = "510300_SELECTED_MIX_RISK_BEFORE_BAND_V1"
PRIMARY = "POOLED_RISK_BEFORE_BAND10"
OUT = ROOT / "reports/research/510300_selected_mix_risk_before_band_v1"
BAND = .1


def fixed_choice(shares, nav, peak, price, raw, stopped, es95, cost, tick, lot):
    """交易带只能延迟可选择的变动，不能延迟风险超限或明确退出。"""
    known = bool(np.isfinite(raw))
    if known and not 0 <= raw <= 1:
        raise ValueError("保存信号仓位超出零至一范围。")
    desired = int(np.floor(raw * nav / price / lot)) * lot if known else shares
    if stopped:
        desired = 0
    projected = parent.risk_plan(shares, nav, peak, price, desired, es95, cost, tick, lot)
    held = parent.risk_plan(shares, nav, peak, price, shares, es95, cost, tick, lot)
    hold_feasible = held["risk_plan_feasible"] and held["target_shares"] == shares
    explicit_exit = bool(stopped or (known and raw == 0))
    distance = abs(projected["target_shares"] - shares) * price / nav
    band_suppressed = bool(shares > 0 and hold_feasible and not explicit_exit and distance < BAND
                           and projected["target_shares"] != shares)
    final = held if band_suppressed else projected
    return {**final, "source_target": raw, "source_known": known,
            "uncapped_signal_shares": desired, "projected_target_shares": projected["target_shares"],
            "projected_distance": distance, "holding_already_feasible": bool(hold_feasible),
            "risk_required_reduction": bool(shares > 0 and not hold_feasible),
            "band_suppressed": band_suppressed, "explicit_exit": explicit_exit,
            "shares_before_decision": shares, "decision_nav": nav, "decision_peak": peak,
            "risk_stopped": stopped, "used_band": BAND}


def mechanism_checks(cfg):
    cost = cfg["costs"]["STRESS"]
    args = {"nav": 200000., "peak": 200000., "price": 4., "stopped": False,
            "es95": .02, "cost": cost, "tick": cfg["tick"], "lot": cfg["lot"]}
    small = fixed_choice(shares=24000, raw=1., **args)
    assert small["band_suppressed"] and small["target_shares"] == 24000
    excess = fixed_choice(shares=26000, raw=1., **args)
    assert excess["risk_required_reduction"] and excess["target_shares"] < 26000
    assert not excess["band_suppressed"]
    exit_case = fixed_choice(shares=1000, raw=0., **args)
    assert exit_case["target_shares"] == 0 and not exit_case["band_suppressed"]
    stopped = fixed_choice(shares=1000, raw=1., **{**args, "stopped": True})
    assert stopped["target_shares"] == 0
    unknown = fixed_choice(shares=1000, raw=np.nan, **args)
    assert unknown["target_shares"] <= 1000
    missing_tail = fixed_choice(shares=0, raw=1., **{**args, "es95": None})
    assert missing_tail["target_shares"] == 0
    flat_entry = fixed_choice(shares=0, raw=.05, **args)
    assert flat_entry["target_shares"] > 0 and not flat_entry["band_suppressed"]
    return {"small_optional_change_held": True, "excess_risk_reduced": True,
            "zero_target_and_stop_exit": True, "missing_source_no_addition": True,
            "missing_tail_no_entry": True, "entry_not_blocked_by_band": True}


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("本轮已经固定，不能覆盖原协议。")
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    checks = mechanism_checks(cfg)
    for name in ["code", "inputs", "accounts", "results"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, root / "code" / Path(__file__).name)
    mandate = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    save(root / "inputs/previous_mandate.json", mandate, True)
    protocol = {
        "study_id": STUDY, "at": parent.now(), "primary": PRIMARY,
        "user_instruction": "不找到不中止   找到再停止", "scope": "已有数据、仅指数与现金的研究模拟",
        "question": "在风险裁剪之后再应用原10个百分点调仓带，能否改善完整账户的成本后表现。",
        "motivation": "父共享训练压力账户165次加仓申请中112次被风险上限裁为不足10个百分点的加仓；这是事后发现的执行交互，不是未见历史的新假设。",
        "fixed_rule": "按收盘权益和原信号算整手目标，先用不变的风险预算投影。已有持仓且保持原份额本身可行、并无零目标或回撤停止时，目标与当前仓位差小于10个百分点则保持。否则执行风险可行目标。",
        "priority": "风险超限减仓、原信号归零及回撤停止优先于调仓带；空仓初次进入不受调仓带拦截。",
        "unchanged": ["三类共享退出训练的全部保存模型与每日原目标", "两年每日训练与每日风险估计",
                      "50%目标上限、五日ES95为2.5%、10%跳空预算和回撤余量", "成本、账户时钟、T+1与现金、分红"],
        "source": (pooled.OUT / "result.json").relative_to(ROOT).as_posix(),
        "period": [parent.START, parent.END], "capital": 200000, "annual_days": 242,
        "costs": cfg["costs"], "target": {"sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "planned_new_full_accounts": 2, "new_models": 0, "new_parameter_grid": 0,
        "comparison": "与相同成本父共享训练账户比较；完整同日收益差、20日区块2000次、种子20260927。",
        "implementation_checks": checks, "goal_achieved": False, "orders_authorized": False,
        "evidence_limit": "全部历史已有研究选择，点值或比较改善均不能当作独立验证。"}
    save(root / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update([LATEST / "candidate_features.parquet", ROOT / "data/reference/510300_dividends.csv",
                    ROOT / "config/510300_incremental_selected_intent_mix_v1.json",
                    parent.OUT / "results/tail_forecasts.json", pooled.OUT / "protocol.json",
                    pooled.OUT / "result.json"])
    for cost in cfg["costs"]:
        sources.update([pooled.OUT / "accounts" / cost / pooled.PRIMARY / "decisions.parquet",
                        pooled.OUT / "accounts" / cost / pooled.PRIMARY / "ledger.parquet"])
    save(root / "freeze.json", {"at": parent.now(), "code_sha256": digest(Path(__file__)),
                                "protocol_sha256": digest(root / "protocol.json"),
                                "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    mandate.update(latest_user_instruction="不找到不中止   找到再停止", continuation_requested_at=parent.now(),
                   current_round=STUDY, latest_integrated_experiment=STUDY,
                   current_protocol=(root / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(root / "freeze.json").relative_to(ROOT).as_posix())
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", mandate)
    print("风险目标先行的固定执行比较已登记，模型与风险预算均不改变。", flush=True)


def verify_sources(root):
    frozen = read(root / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(root / "protocol.json") == frozen["protocol_sha256"]
    for name, expected in frozen["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("固定来源变化：" + name)


def simulate(data, dividends, cfg, source, tails, cost_name):
    peak, stopped = cfg["initial_capital"], False
    tail_map = {row["origin_index"]: row for row in tails}
    cost = cfg["costs"][cost_name]

    def policy(account, price, dummy, config, model, t):
        nonlocal peak, stopped
        nav = account.value(price)
        peak = max(peak, nav)
        drawdown = 1 - nav / peak
        stopped = stopped or drawdown >= .1
        tail = tail_map[t]
        chosen = fixed_choice(account.shares, nav, peak, price, float(source[t]), stopped,
                              tail["es95"], cost, config["tick"], config["lot"])
        q = chosen["target_shares"]
        return {**chosen, "requested_quantity": q - account.shares, "reference_weight": q * price / nav,
                "action": "风险可行目标按固定调仓带执行", "decision_drawdown": drawdown,
                "tail_sample_rows": tail["training_rows"], "tail_es95": tail["es95"],
                "tail_q05": tail["q05"], "conditional_tail": tail["conditional"]}

    ledger, decisions, state = simulate_indexed_request_account(
        data, dividends, cfg, cost, parent.START, PRIMARY, targets=np.zeros(len(data)),
        event_mask=np.ones(len(data), bool), request_policy=policy, next_execution_date=parent.NEXT)
    state["risk_governor"] = {"peak": peak, "stopped": stopped}
    return ledger, decisions, state


def verify_accounts(root, data, cfg):
    count = 0
    details = []
    for cost in cfg["costs"]:
        folder = root / "accounts" / cost / PRIMARY
        ledger = pd.read_parquet(folder / "ledger.parquet")
        decisions = pd.read_parquet(folder / "decisions.parquet")
        old = pd.read_parquet(pooled.OUT / "accounts" / cost / pooled.PRIMARY / "ledger.parquet")
        pd.testing.assert_series_equal(ledger.date, old.date)
        assert ledger.cash.ge(-1e-8).all() and ledger.mark_clock.eq("CLOSE").all()
        assert ledger.accounting_error.abs().max() < 1e-6
        nav = ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable
        np.testing.assert_allclose(nav, ledger.equity, atol=1e-7, rtol=0)
        previous = np.r_[200000., ledger.equity.to_numpy()[:-1]]
        np.testing.assert_allclose(ledger.equity / previous - 1, ledger.net_return, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(ledger.requested_quantity, decisions.requested_quantity.iloc[:-1])
        assert (decisions.execution_date > decisions.origin).all()
        feasible = decisions.loc[decisions.risk_plan_feasible]
        assert feasible.planned_target_exposure.le(.5 + 1e-10).all()
        assert (feasible.planned_tail_loss <= feasible.tail_budget + 1e-8).all()
        assert (feasible.planned_gap_loss <= feasible.gap_budget + 1e-8).all()
        assert decisions.loc[decisions.risk_required_reduction, "requested_quantity"].lt(0).all()
        held = decisions.loc[decisions.band_suppressed]
        assert held.requested_quantity.eq(0).all() and held.holding_already_feasible.all()
        assert (~held.explicit_exit).all() and held.projected_distance.lt(BAND).all()
        assert decisions.loc[~decisions.source_known, "requested_quantity"].le(0).all()
        count += 1
        details.append({"cost": cost, "band_held_decisions": len(held),
                        "risk_reductions": int(decisions.risk_required_reduction.sum()),
                        "planned_infeasible_exits": int((~decisions.risk_plan_feasible).sum()),
                        "old_fills": int(old.filled_quantity.ne(0).sum()),
                        "new_fills": int(ledger.filled_quantity.ne(0).sum()),
                        "old_fee": float((old.commission + old.slippage_cost).sum()),
                        "new_fee": float((ledger.commission + ledger.slippage_cost).sum())})
    result = {"status": "PASS_RISK_PRIORITY_AND_SAVED_ACCOUNT_RECOMPUTATION", "accounts": count,
              "mechanism_checks": mechanism_checks(cfg), "details": details}
    save(root / "verification.json", result, True)
    return result


def run(root):
    verify_sources(root)
    save(root / "RUN_STARTED.json", {"at": parent.now()}, True)
    cfg = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    data = data.loc[data.date.le(parent.END)].reset_index(drop=True)
    dividends = normalize_dividends(pd.read_csv(ROOT / "data/reference/510300_dividends.csv"))
    tails = read(parent.OUT / "results/tail_forecasts.json")
    accounts, comparisons = {}, []
    rng = np.random.default_rng(20260927)
    for cost in cfg["costs"]:
        old_folder = pooled.OUT / "accounts" / cost / pooled.PRIMARY
        source_decisions = pd.read_parquet(old_folder / "decisions.parquet")
        indices = source_decisions.origin_index.to_numpy(int)
        assert np.array_equal(indices, np.arange(indices[0], len(data)))
        np.testing.assert_array_equal(data.date.iloc[indices].to_numpy(), source_decisions.origin.to_numpy())
        source = np.full(len(data), np.nan)
        source[indices] = source_decisions.source_target.to_numpy(float)
        ledger, decisions, state = simulate(data, dividends, cfg, source, tails, cost)
        folder = root / "accounts" / cost / PRIMARY
        folder.mkdir(parents=True)
        ledger.to_parquet(folder / "ledger.parquet", index=False)
        decisions.to_parquet(folder / "decisions.parquet", index=False)
        save(folder / "checkpoint.json", state, True)
        accounts[PRIMARY, cost] = ledger
        comparisons.append({"cost": cost, "left": PRIMARY, "right": pooled.PRIMARY,
                            **parent.paired_interval(ledger, pd.read_parquet(old_folder / "ledger.parquet"), rng)})
        m = metrics(ledger)
        print(f"风险目标先行 {cost}：夏普{m['sharpe']:.6f}，年化{m['annual_return']:.2%}，回撤{m['max_drawdown']:.2%}。", flush=True)
    measurements, windows = parent.summarize_accounts(root, data, accounts)
    save(root / "results/paired_comparisons.json", comparisons, True)
    checks = verify_accounts(root, data, cfg)
    primary = next(row for row in measurements if row["cost"] == "STRESS")
    result = {"study_id": STUDY, "at": parent.now(),
              "status": "POINT_PASS_DEVELOPMENT_ONLY" if primary["historical_point_targets_met"] else "FROZEN_NO_QUALIFIED_RISK_BEFORE_BAND",
              "primary": primary, "all_accounts": measurements, "comparisons": comparisons, "checks": checks,
              "rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in windows if r["cost"] == "STRESS"),
              "new_full_accounts": 2, "new_model_fits": 0, "new_independent_observations": 0,
              "goal_achieved": False, "orders_authorized": False, "current_market_view": "NO_VIEW"}
    save(root / "result.json", result, True)
    print("风险裁剪与交易带顺序的有限比较完成，保存结果与父实验同时保留。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="风险可行目标先于调仓带的固定比较")
    parser.add_argument("command", choices=["freeze", "run", "status"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "status":
        print(read(args.out / "result.json") if (args.out / "result.json").exists() else "研究尚未完成")
    else:
        {"freeze": freeze, "run": run}[args.command](args.out)


if __name__ == "__main__":
    main()
