"""只读复用回购预测，比较禁止加仓与固定五日退出的完整账户。"""
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

import research.funding_repurchase_demand_daily_v1 as prior
import research.funding_forecast_maturity_account_v1 as maturity
import research.repurchase_horizon_loss_diagnostic_v1 as diagnosis
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_repurchase_fixed_maturity_account_v1"
STUDY = "510300_REPURCHASE_FIXED_MATURITY_ACCOUNT_V1"
VARIANTS = ["ROLL_NO_ADD", "FIXED5_NO_ADD"]
PRIMARY = prior.PRIMARY + "__FIXED5_NO_ADD"


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("回购期限检验已固定，不能覆盖。")
    assert read(prior.OUT / "result.json")["status"] == "FROZEN_NO_QUALIFIED_DISCLOSED_REPURCHASE_STRATEGY"
    assert read(diagnosis.OUT / "result.json")["daily_cash_identity_reconstructed_accounts"] == 4
    for folder in ["code", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "五日分布被每日续持转为长周期库存后造成的损失，能否通过事前固定同一期限改善；回购信息在相同期限下是否仍有增量？",
        "why_now": "回购信息试验已失败。随后逐批次归因发现主STRESS前五日毛损益4044.50元、后续续持毛损益-9228.60元；这只是实际路径归因，不能视为可赚取的反事实。",
        "models": prior.TRADED, "predictions": "复用上一轮全部逐日预测与训练池，不重新拟合、不重选特征或邻居；全部历史价格已被研究。",
        "variants": {"ROLL_NO_ADD": "空仓可按原信号和预算建立仓位，已有仓位不加仓；每日按原五日分布调整或退出。",
                     "FIXED5_NO_ADD": "空仓按完全相同信号及预算入场，期间不加仓；第五个后续交易日开盘退出，之前仅因尾部预算、回撤或信息失效减仓/退出；原尾部规则优先。"},
        "implementation": "完整复用已存在的funding_forecast_maturity_account_v1.simulate，不新增退出期限搜索；此次唯一变化是输入新的回购/资金共同样本预测。",
        "controls": "与同模型原每日库存账户及ROLL_NO_ADD比较期限作用；FIXED5_NO_ADD内部三个价格/资金对照检验回购信息增量。不能把所有模型共有的执行改善归功于回购。",
        "account": "16新账户，加8个只读原账户；20万元，510300.SH/CASH_CNY，T+1、100份、压力成本决策、最多50%仓位、ES及跳空和回撤规则全沿用。",
        "period": [prior.START, prior.END], "annual_days": 242, "costs": prior.distribution.COSTS,
        "targets": {"stress_sharpe": 1.2, "stress_cagr": .1, "max_drawdown": .1},
        "comparison": "主STRESS相对五个对照，20日循环区块2000次seed2026092604；完整账户及上一轮固定可用期均报告，后者不替代全期目标。逐年及全部滚动两年保留。",
        "development_gate": "主STRESS完整三目标通过、相对同期限三个信息对照配对差95%下界均正、至少两个日历年各126预测日且增量为正，才列开发候选；仍不代替独立验证。",
        "selection_boundary": "这是已看到原失败后提出的有限开发检验，不是未触碰留出样本。旧资金期限研究的失败保持原记录。",
        "new_full_accounts": 16, "reused_full_accounts": 8, "new_prediction_fits": 0,
        "new_parameter_searches": 0, "new_market_downloads": 0, "new_independent_forward_observations": 0,
        "goal_achieved": False, "orders_authorized": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    paths = [Path(__file__), Path(prior.__file__), Path(maturity.__file__), Path(prior.parent.__file__),
             Path(prior.engine.__file__), Path(prior.inventory.__file__), Path(prior.distribution.__file__),
             ROOT / "research/selected_mix_daily_two_year_v1.py", ROOT / "research/strategy_review_diagnostics_v1.py",
             prior.OUT / "result.json", prior.OUT / "inputs/decision_information.parquet", prior.OUT / "results/predictions.parquet",
             prior.parent.OUT / "inputs/market.parquet", prior.parent.DIVIDENDS, diagnosis.OUT / "result.json"]
    for model in prior.TRADED:
        for cost in prior.distribution.COSTS:
            for filename in ["ledger.parquet", "decisions.parquet"]:
                paths.append(prior.OUT / "accounts" / cost / model / filename)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
                               "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
                               "before_new_maturity_account_returns": True}, True)
    print("已固定16个禁止加仓/五日期限账户；复用预测，保留原失败账户。", flush=True)


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for path, sha in frozen["sources"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    market = pd.read_parquet(prior.parent.OUT / "inputs/market.parquet")
    x = pd.read_parquet(prior.OUT / "inputs/decision_information.parquet")
    forecasts = pd.read_parquet(prior.OUT / "results/predictions.parquet")
    dividends = prior.engine.normalize_dividends(pd.read_csv(prior.parent.DIVIDENDS))
    accounts, checks, cycle_counts = {}, [], []
    for model in prior.TRADED:
        for cost in prior.distribution.COSTS:
            policy = model + "__ORIGINAL"
            accounts[policy, cost] = pd.read_parquet(prior.OUT / "accounts" / cost / model / "ledger.parquet")
    # 两种已有模拟入口须在原规则下还原同一结果，才允许把差异解释为期限。
    original, _ = maturity.simulate(market, x, dividends, forecasts, prior.PRIMARY, "ORIGINAL", "STRESS")
    columns = ["date", "idx", "equity", "shares", "cash", "filled_quantity", "commission", "slippage_cost", "net_return"]
    pd.testing.assert_frame_equal(original[columns], accounts[prior.PRIMARY + "__ORIGINAL", "STRESS"][columns], check_exact=True)
    for model in prior.TRADED:
        for variant in VARIANTS:
            for cost in prior.distribution.COSTS:
                policy = model + "__" + variant
                ledger, decisions = maturity.simulate(market, x, dividends, forecasts, model, variant, cost)
                check = prior.parent.account_verification(ledger, decisions)
                assert not (ledger.shares_before.gt(0) & ledger.filled_quantity.gt(0)).any()
                if variant == "FIXED5_NO_ADD":
                    exits = decisions[decisions.reason.eq("FIVE_DAY_MATURITY")]
                    assert exits.idx.ge(exits.deadline_idx).all()
                folder = OUT / "accounts" / cost / policy
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / "ledger.parquet", index=False)
                decisions.to_parquet(folder / "decisions.parquet", index=False)
                closed, pending = cycles(ledger)
                closed.to_parquet(folder / "cycles.parquet", index=False)
                save(folder / "pending_cycle.json", pending, True)
                accounts[policy, cost] = ledger
                checks.append({"policy": policy, "cost": cost, **check, "no_add_to_open_cycle": True})
                cycle_counts.append({"policy": policy, "cost": cost, "closed": len(closed), "open": pending is not None})
                m = metrics(ledger)
                sh = "未定义" if m["sharpe"] is None else f"{m['sharpe']:.6f}"
                print(f"{policy}/{cost}：夏普{sh}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, market, accounts)
    old = read(prior.OUT / "result.json")
    period = old["available_period_diagnostics"][0]
    lo, hi = period["start"], period["end"]
    comparisons, available = [], []
    rng = np.random.default_rng(2026092604)
    controls = [(model + "__FIXED5_NO_ADD", "INFORMATION") for model in prior.TRADED[:-1]]
    controls += [(prior.PRIMARY + "__ROLL_NO_ADD", "MATURITY"), (prior.PRIMARY + "__ORIGINAL", "COMBINED_EXECUTION")]
    for cost in prior.distribution.COSTS:
        left = accounts[PRIMARY, cost]
        for control, purpose in controls:
            right = accounts[control, cost]
            mask = left.date.between(lo, hi)
            years = []
            for entry in old["calendar_coverage"]:
                year = entry["year"]
                take = left.date.dt.year.eq(year)
                years.append({**entry, "annual_arithmetic_difference": float((left.loc[take, "net_return"] - right.loc[take, "net_return"]).mean() * 242)})
            comparisons.append({"cost": cost, "control": control, "purpose": purpose,
                                "full_account": paired_interval(left, right, rng),
                                "fixed_available_period": paired_interval(left.loc[mask], right.loc[mask], rng), "years": years})
        for model in prior.TRADED:
            for variant in VARIANTS:
                policy = model + "__" + variant
                ledger = accounts[policy, cost]
                mask = ledger.date.between(lo, hi)
                begin = int(np.flatnonzero(mask)[0])
                capital = float(ledger.equity.iloc[begin - 1]) if begin else 200000.
                available.append({"policy": policy, "cost": cost, "start": lo, "end": hi,
                                  "only_coverage_diagnostic": True, **metrics(ledger.loc[mask], capital)})
    main = next(m for m in measures if m["policy"] == PRIMARY and m["cost"] == "STRESS")
    increments = [r for r in comparisons if r["cost"] == "STRESS" and r["purpose"] == "INFORMATION"]
    information_pass = all(r["full_account"]["lower_95"] > 0 for r in increments)
    cross_year_pass = old["cross_year_evidence_sufficient"] and all(y["annual_arithmetic_difference"] > 0 for r in increments for y in r["years"] if y["at_least_126"])
    ledger = accounts[PRIMARY, "STRESS"]
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    expense = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
    np.testing.assert_allclose(gross - expense, ledger.equity.iloc[-1] - 200000., atol=1e-6, rtol=0)
    candidate = main["historical_point_targets_met"] and information_pass and cross_year_pass
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if candidate else "FROZEN_NO_QUALIFIED_REPURCHASE_MATURITY_STRATEGY",
        "primary": main, "all_accounts": measures, "comparisons": comparisons, "available_period_diagnostics": available,
        "information_increment_gate": information_pass, "cross_year_evidence_gate": cross_year_pass,
        "new_full_accounts": 16, "reused_full_accounts": 8, "recomputed_original_identity_checks": 1,
        "original_account_recomputed_exactly": True, "new_prediction_fits": 0, "new_parameter_searches": 0,
        "account_checks": checks, "completed_and_pending_cycles": cycle_counts,
        "primary_cash_attribution": {"gross_pnl": gross, "cost_and_reserve": expense, "net_profit": float(ledger.equity.iloc[-1] - 200000.)},
        "primary_rolling_two_year_joint_passes": sum(r["joint_point_pass"] for r in rolling if r["policy"] == PRIMARY and r["cost"] == "STRESS"),
        "historical_prices_previously_researched": True, "new_independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False, "orders_authorized": False,
        "review_package_created": False}, True)
    print("回购预测的16个期限/加仓对照账户已完成，结果已保存。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="回购信息五日固定期限的完整账户对照")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            if not (OUT / "RUN_FAILURE.json").exists():
                save(OUT / "RUN_FAILURE.json", {"at": now(), "error": f"{type(exc).__name__}: {exc}"}, True)
            raise
