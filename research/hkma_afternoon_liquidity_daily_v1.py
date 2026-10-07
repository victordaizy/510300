"""离岸人民币设施下午使用变化的一项增量；两年日更、完整510300研究账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from types import FunctionType, SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.daily_liquidity_insurance_tail_v1 as distribution
import research.index_state_inventory_daily_v1 as inventory
import research.intraday_overnight_increment_v1 as engine
import research.repo_segmentation_daily_v1 as previous
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.selected_mix_daily_two_year_v1 import summarize_accounts, paired_interval
from research.strategy_review_diagnostics_v1 import metrics, cycles

OUT = ROOT / "reports/research/510300_hkma_afternoon_liquidity_daily_v1"
SOURCE = ROOT / "reports/research/510300_hkma_rmb_liquidity_source_v1"
PARENT = previous.OUT
STUDY = "510300_HKMA_AFTERNOON_LIQUIDITY_DAILY_V1"
PRIMARY = "PRICE_AND_AFTERNOON_FUNDING"
BASE = previous.BASE
MODELS = {"HISTORY": [], "PRICE": BASE, PRIMARY: [*BASE, "afternoon_usage_change"]}
START, END = previous.START, previous.END


def predictor():
    original = distribution.one_day_distribution
    return FunctionType(original.__code__, {**original.__globals__, "MODELS": MODELS}, original.__name__, original.__defaults__, original.__closure__)


def design():
    d = pd.read_parquet(PARENT / "inputs/market.parquet")
    x = pd.read_parquet(PARENT / "inputs/decision_information.parquet", columns=["date", "idx", "decision_time", *BASE])
    x["date"] = pd.to_datetime(x.date).astype("datetime64[ns]")
    x["decision_time"] = pd.to_datetime(x.decision_time).dt.as_unit("ns")
    source = pd.read_parquet(SOURCE / "rmb_liquidity_usage.parquet")
    source["source_date"] = pd.to_datetime(source.source_date).astype("datetime64[ns]")
    source["available_at"] = pd.to_datetime(source.available_at).dt.as_unit("ns")
    x = pd.merge_asof(x.sort_values("decision_time"), source.sort_values("available_at"),
                      left_on="decision_time", right_on="available_at", direction="backward")
    x["common_known"] = x[MODELS[PRIMARY]].notna().all(axis=1) & (x.date-x.source_date).dt.days.between(1, 7)
    known = x.common_known
    assert x.loc[known, "available_at"].le(x.loc[known, "decision_time"]).all()
    assert x.loc[known, "source_date"].lt(x.loc[known, "date"]).all()
    labels = pd.read_parquet(PARENT / "inputs/mature_labels.parquet")
    dividends = engine.normalize_dividends(pd.read_csv(previous.DIVIDENDS))
    return d, dividends, x, labels


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮已固定，禁止覆盖。")
    for folder in ["code", "inputs", "results", "accounts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    result = read(SOURCE / "result.json")
    assert result["status"] == "SOURCE_READY_INTRADAY_USAGE_CHANGE_ONLY"
    assert result["last_date"][:10] == END
    protocol = {"at": now(), "study_id": STUDY, "primary": PRIMARY, "control": "PRICE", "mean_reference": "HISTORY",
        "question": "在相似的A股价格弱势下，香港人民币日间回购安排在下午14至16时的借款余额变化能否区分随后修复与持续恶化？",
        "hypothesis": "下午增加借款可能伴随结算或融资需求，减少可能表示偿还；信息方向由仅过去两年的条件分布估计，不事先声称全部变化都是市场压力。",
        "single_new_information": "(intraday_repo_at_1600-intraday_repo_at_1400)/(1+intraday_repo_at_1600+intraday_repo_at_1400)，金额单位为百万元。原始二时点都为零时特征为零；不加入其他额度或期限。",
        "methodology": "设施额度、价格和可接受抵押品曾变化。固定用同日二时点变化避免机械跨日额度跳变，但不宣称彻底消除制度影响；2025-10-09前后单独描述，不依此选择区间。",
        "availability": "设施源日翌日00:00计划可用，合并至执行日09:00，来源年龄最多7自然日。香港与A股交易日分别处理，不使用执行日16:00的未来数据。历史首版与首次实际送达未认证。",
        "assets": ["510300.SH", "CASH_CNY"], "capital": 200000, "period": [START, END], "annual_days": 242,
        "targets": {"sharpe": 1.2, "cagr": .1, "max_drawdown": .1},
        "fixed_parent_methods": {"protocol": (PARENT / "protocol.json").relative_to(ROOT).as_posix(),
            "training": "最近两个日历年，每日更新，退出开盘严格早于执行日；最少252共同成熟原点。",
            "distribution": "训练内标准化截断[-5,5]、固定126近邻经验分布；价格3特征加一个新信息；HISTORY同池经验分布。",
            "account": "五日弱势时才新增库存，枚举100份目标，w*mu5-2*w^2*var5-调仓及退出费用；两候选两档成本共4连续账户。",
            "risk": "50%目标上限，5日ES95为2.5%，10%跳空预算为权益5%且不超过距90%权益峰值余量一半；先扣调仓费用再算预算；10%回撤下一可卖开盘退出且本轮不恢复。",
            "execution": "09:00判断、下一开盘模拟成交，现金、T+1、涨跌停、100份整手及分红分账；末端照常收盘计价，有仓计退出储备。"},
        "costs": distribution.COSTS, "new_final_accounts": 4, "new_parameter_searches": 0,
        "evaluation": "完整同日账户增量、20日循环区块2000次seed2026092508；固定2020-2023、2024-末端，以及全部滚动两年；五日预测含全部成熟与当时价格弱势两个已定义口径。",
        "monitor": "固定5日相位，只有已成熟结果进入最近60个标签的分位覆盖监控，突破率超过15%记录报警，不据结果再加启停参数。",
        "development_gate": "主压力账户三目标达标、收益增量95%下界>0且两个固定时期增量均正，才保留开发候选；仍无独立前向验证。",
        "duplicate_boundary": "此前大陆回购定盘差、DR007政策利差、央行操作总量和美国NFCI均保留原结果；本次新增香港官方日间设施同日下午变化，价格与账户方法沿用，不扫描宏观变量。",
        "prices_previously_researched": True, "new_independent_forward_observations": 0, "orders_authorized": False, "goal_achieved": False}
    save(OUT / "protocol.json", protocol, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "inputs/mandate.json", read(ROOT / "config/510300_existing_data_training_mandate_v1.json"), True)
    paths = [Path(__file__), Path(previous.__file__), Path(distribution.__file__), Path(inventory.__file__), Path(engine.__file__),
             ROOT / "research/strategy_review_diagnostics_v1.py", ROOT / "research/selected_mix_daily_two_year_v1.py",
             previous.DIVIDENDS, PARENT / "protocol.json", PARENT / "inputs/market.parquet",
             PARENT / "inputs/decision_information.parquet", PARENT / "inputs/mature_labels.parquet",
             SOURCE / "result.json", SOURCE / "rmb_liquidity_usage.parquet", SOURCE / "collection_plan.json",
             SOURCE / "known_methodology_change.json", SOURCE / "official_method_text.json"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_candidate_returns": True}, True)
    print("香港人民币设施下午变化的一项增量与四账户方案已固定。", flush=True)


def learn(x, labels):
    function = predictor()
    predictions, models, receipts = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, receipt, m = function(int(i), x, labels)
        predictions.extend(p)
        receipts.append(receipt)
        models.extend(m)
        if len(receipts) % 300 == 0:
            print(f"两年分布已更新至{x.date.iloc[i].date()}。", flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_parquet(OUT / "results/predictions.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(OUT / "results/update_receipts.parquet", index=False)
    save(OUT / "results/saved_distributions.json", models, True)
    return pred, models


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    d, dividends, x, labels = design()
    x.to_parquet(OUT / "inputs/decision_information.parquet", index=False)
    pred, models = learn(x, labels)
    original = previous.forecast_evaluation
    evaluate = FunctionType(original.__code__, {**original.__globals__, "OUT": OUT, "PRIMARY": PRIMARY}, original.__name__, original.__defaults__, original.__closure__)
    forecast, scored, monitor = evaluate(pred, labels, x)
    proxy = SimpleNamespace(START=START, END=END, COSTS=distribution.COSTS, plan=previous.plan, risk_valid=previous.risk_valid)
    accounts, checks, cycle_info = {}, [], []
    for model in ["PRICE", PRIMARY]:
        selected = pred[pred.model.eq(model)].copy()
        selected["model"] = "PRICE"
        for cost in distribution.COSTS:
            ledger, decisions = inventory.simulate(proxy, engine, d, x, dividends, selected, "PRICE_DOWN_ONLY", cost)
            ledger["policy"], decisions["policy"] = model, model
            folder = OUT / "accounts" / cost / model
            folder.mkdir(parents=True, exist_ok=True)
            ledger.to_parquet(folder / "ledger.parquet", index=False)
            decisions.to_parquet(folder / "decisions.parquet", index=False)
            closed, pending = cycles(ledger)
            closed.to_parquet(folder / "cycles.parquet", index=False)
            save(folder / "pending_cycle.json", pending, True)
            accounts[model, cost] = ledger
            checks.append({"model": model, "cost": cost, **previous.account_verification(ledger, decisions)})
            cycle_info.append({"model": model, "cost": cost, "closed": len(closed), "open": pending is not None})
            m = metrics(ledger)
            print(f"{model}/{cost}：夏普{m['sharpe']:.6f}、年化{m['annual_return']:.3%}、回撤{abs(m['max_drawdown']):.3%}。", flush=True)
    measures, rolling = summarize_accounts(OUT, d, accounts)
    rng = np.random.default_rng(2026092508)
    comparisons = []
    for cost in distribution.COSTS:
        left, right = accounts[PRIMARY, cost], accounts["PRICE", cost]
        periods = []
        for start, end in [(START, "2023-12-31"), ("2024-01-01", END)]:
            mask = left.date.between(start, end)
            periods.append({"start": start, "end": end, "annual_arithmetic_difference":
                            float((left.loc[mask, "net_return"]-right.loc[mask, "net_return"]).mean()*242)})
        comparisons.append({"cost": cost, **paired_interval(left, right, rng), "fixed_periods": periods})
    lookup, sample = pred.set_index(["idx", "model"]), labels.set_index("idx")
    for record in models:
        i = record["idx"]
        pool = sample.loc[record["training_indices"]]
        assert pool.date.ge(x.date.iloc[i]-pd.DateOffset(years=2)).all() and pool.exit_idx.lt(i).all()
        stats = distribution.empirical_statistics(sample.loc[record["selected_indices"], "gross_return5"])
        for key in ["mu5", "variance5", "q05", "es95"]:
            np.testing.assert_allclose(stats[key], lookup.loc[(i, record["model"]), key], rtol=0, atol=1e-14)
    for day in ["2022-12-30", "2025-10-10"]:
        i = int(x.date.searchsorted(day, side="right"))-1
        subset = labels[labels.date.le(x.date.iloc[i])].copy()
        subset.loc[subset.exit_idx.ge(i), "gross_return5"] = -9999.
        forecasts, _, _ = predictor()(i, x.iloc[:i+1], subset)
        for row in forecasts:
            for key in ["mu5", "variance5", "q05", "es95"]:
                assert row[key] == lookup.loc[(i, row["model"]), key]
    primary = next(row for row in measures if row["policy"] == PRIMARY and row["cost"] == "STRESS")
    pair = next(row for row in comparisons if row["cost"] == "STRESS")
    increment = pair["lower_95"] > 0 and all(row["annual_arithmetic_difference"] > 0 for row in pair["fixed_periods"])
    source = read(SOURCE / "result.json")
    policy_periods = []
    for model in ["PRICE", PRIMARY]:
        ledger = accounts[model, "STRESS"]
        for name, lo, hi in [("BEFORE_20251009_POLICY_CHANGE", START, "2025-10-08"), ("AFTER_20251009_POLICY_CHANGE", "2025-10-09", END)]:
            subset = ledger[ledger.date.between(lo, hi)]
            i = int(subset.index.min())
            capital = 200000 if i == 0 else float(ledger.equity.iloc[i-1])
            policy_periods.append({"model": model, "period": name, **metrics(subset, capital)})
    result = {"at": now(), "study_id": STUDY,
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if primary["historical_point_targets_met"] and increment else "FROZEN_NO_QUALIFIED_HKMA_LIQUIDITY_STRATEGY",
              "primary": primary, "all_accounts": measures, "comparisons": comparisons,
              "forecast_evaluation": forecast, "daily_distribution_updates": len(models),
              "unique_prediction_days": int(pred.idx.nunique()), "mature_prediction_days": int(scored[scored.gross_return5.notna()].idx.nunique()),
              "source_observations": source["rows"], "new_full_accounts": 4, "new_parameter_searches": 0,
              "completed_and_pending_cycles": cycle_info, "account_checks": checks,
              "saved_distributions_recomputed": len(models), "future_label_exclusion_checks": 2,
              "primary_rolling_two_year_joint_passes": sum(row["joint_point_pass"] for row in rolling if row["policy"] == PRIMARY and row["cost"] == "STRESS"),
              "known_policy_change_periods_descriptive": policy_periods,
              "observation_monitor_alert_counts": pd.DataFrame(monitor).groupby("model").alert.sum().to_dict(),
              "increment_gate_met": increment, "historical_first_vintage_verified": False,
              "new_independent_forward_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("香港人民币下午借款变化的固定研究已完成，原数据缺口已补充。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="香港人民币设施下午变化的两年每日研究")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
