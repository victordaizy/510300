"""接通两条固定点位候选的完整收盘计算链，并与保存的历史逐层对应。"""
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
from research import upward_episode_anatomy_v1 as common
from research import point_core_observation_inputs_v1 as engine
from research.point_state_reconstruction_v1 import MODELS, PERIODS
from research.adaptive_allocation_v1 import normalize_dividends

OUT = ROOT / "reports/research/510300_point_core_observation_v1"
READY = ROOT / "reports/research/510300_point_forward_readiness_v1"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
CONFIGS = ("entry_vintage_exit", "continuous_reference_min_variance", "downside_reference_risk",
           "joint_downside_reference_pair", "trend_noise_reference_blend")
LAYERS = {name.upper(): "510300_" + name + "_v1" for name in (
    "entry_vintage_exit", "vintage_reference_risk", "downside_reference_risk", "continuous_reference_min_variance",
    "joint_downside_reference_pair", "model_support_reference_router", "trend_noise_reference_blend")}
LAYERS.update(MODELS)
LEDGER_FIELDS = ("cash", "equity", "net_return", "shares", "filled_quantity", "commission", "dividend_recognized", "dividend_paid")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "freeze.json").exists(), "本次完整链已登记，不覆盖。")
    files = {}
    def copy(source, relative):
        destination = OUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[relative] = {"source": str(source.relative_to(ROOT)), "sha256": common.digest(destination)}

    for name in ("daily.parquet", "panic_daily.parquet", "dividends.csv", "within_models.json", "original_models.json", "already_inspected_daily.parquet"):
        copy(READY / "inputs" / name, "inputs/" + name)
    for key in CONFIGS:
        copy(ROOT / f"config/510300_{key}_v1.json", f"inputs/config/{key}.json")
    copy(CONTEXT / "active_goal_effective_requirements.json", "inputs/current_requirements.json")
    for period in PERIODS:
        for cost in ("BASE", "STRESS"):
            for model, study in LAYERS.items():
                copy(ROOT / f"reports/research/{study}/{period}/{cost}/{model}_decisions.parquet",
                     f"inputs/saved/{period}/{cost}/{model}_decisions.parquet")
        for model in ("DOWNSIDE_REFERENCE_RISK", "CONTINUOUS_REFERENCE_MIN_VARIANCE"):
            copy(ROOT / f"reports/research/{LAYERS[model]}/{period}/BASE/{model}_ledger.parquet",
                 f"inputs/saved/{period}/BASE/{model}_ledger.parquet")
    sources = (Path(__file__), Path(engine.__file__), ROOT / "tests/test_point_core_observation_inputs_v1.py",
               ROOT / "research/reference_observation_accounts_v1.py", ROOT / "research/point_state_reconstruction_v1.py",
               ROOT / "research/entry_vintage_exit_inputs_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
               ROOT / "research/downside_reference_risk_inputs_v1.py", ROOT / "research/joint_downside_reference_pair_inputs_v1.py",
               ROOT / "research/model_support_reference_router_inputs_v1.py", ROOT / "research/trend_noise_reference_blend_inputs_v1.py",
               ROOT / "research/two_policy_min_variance_inputs_v1.py")
    for source in sources:
        copy(source, "code/" + source.name)
    protocol = {
        "study": "510300_POINT_CORE_OBSERVATION_V1", "at": common.now(),
        "scope": "既有两条多头点位候选的计算接线与历史对应，不新增策略、盈利检验或参数搜索。",
        "user_gate": "p乘实际净盈亏比严格大于1、净均值大于0；年度次数为软目标。",
        "parents": "七层来源从真实已知价格、原模型、参考持仓及收益计算，不读取已存父目标作为输入。",
        "historical_reference": "已存36组目标只作为复现对照。每段末端前逐日对应；最后日按真实收盘独立计算，生成下一计划交易日意向。",
        "internal_reference_paths": "从2013年连续的急跌回升与原岭模型各一条；每个历史段固定入场参考BASE/STRESS各一条、下行风险BASE与连续方差组合BASE各一条，共十条内部参考路径。",
        "risk": "按原规则在月初第一个实际收盘更新242日预算，零预算来源若必需状态未知仍保留未知；不得填入统一末端强平收益。",
        "models": "沿用原月度模型与当时可用时钟，本次不拟合新月份或调整训练制度。",
        "last_history_date": "2026-08-14", "latest_already_inspected_date": "2026-09-16",
        "next_date": {"earlier_diagnostic": "2020-01-02", "evaluation": "2026-08-17"},
        "current_view": "末端输出只用于历史接线验证；没有覆盖研究日最新行情和后续月度模型，不构成当前信号。",
        "prospective_observations": 0, "new_model_fits": 0, "orders_authorized": False, "goal_achieved": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("完整核心链的输入、原规则、历史对照和末端观察范围已固定。", flush=True)


def compare_numeric(a, b, message, tolerance=1e-11):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)), message + "：形状或缺失状态不同。")
    require(np.allclose(a, b, rtol=0, atol=tolerance, equal_nan=True), message + "：数值不同。")
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.


def check_targets(period, cost, factors, data):
    rows = []
    start, end = map(pd.Timestamp, PERIODS[period])
    first = int(np.flatnonzero(data.date.ge(start))[0])
    expected_origins = pd.DatetimeIndex(data.date.iloc[first-1:-1])
    expected_execution = pd.DatetimeIndex(data.date.iloc[first:])
    for model in LAYERS:
        saved = pd.read_parquet(OUT / f"inputs/saved/{period}/{cost}/{model}_decisions.parquet")
        require(pd.DatetimeIndex(saved.origin).equals(expected_origins), model + "原来源日历不一致。")
        require(pd.DatetimeIndex(saved.execution_date).equals(expected_execution), model + "原下一开盘日历不一致。")
        indices = saved.origin_index.to_numpy(int)
        require(np.array_equal(indices, np.arange(first-1, len(data)-1)), model + "原日期索引不同。")
        error = compare_numeric(factors[model].iloc[indices], saved.reference_weight, f"{period}/{cost}/{model}")
        rows.append({"period": period, "cost": cost, "model": model, "compared_rows": len(saved),
                     "maximum_error": error, "unknown_states_match": True, "status": "PASS_HISTORICAL_TARGET_PARITY"})
    require(data.date.iloc[-1] == end, "复现历史边界错误。")
    return rows


def check_budgets(period, chain):
    rows = []
    mappings = {
        "variance_budget": ("CONTINUOUS_REFERENCE_MIN_VARIANCE", ("panic_budget", "learned_budget", "panic_sd", "learned_sd", "reference_covariance", "difference_variance", "risk_window_observations")),
        "joint_budget": ("JOINT_DOWNSIDE_REFERENCE_PAIR", ("downside_budget", "continuous_budget", "joint_downside_second_moment", "previous_budget_downside_second_moment", "downside_gradient", "risk_window_observations")),
    }
    for key, (model, columns) in mappings.items():
        saved = pd.read_parquet(OUT / f"inputs/saved/{period}/BASE/{model}_decisions.parquet")
        fresh = chain[key].iloc[saved.origin_index.to_numpy(int)].reset_index(drop=True)
        for column in columns:
            error = compare_numeric(fresh[column], saved[column], f"{period}/{model}/{column}")
            rows.append({"period": period, "source": model, "quantity": column, "compared_rows": len(saved), "maximum_error": error})
        for column in ("risk_status", "risk_update_scheduled", "risk_attempt_origin", "last_successful_risk_origin", "risk_window_start"):
            a, b = fresh[column], saved[column]
            require(((a.isna() & b.isna()) | a.eq(b)).all(), f"{period}/{model}/{column}时钟或状态不同。")
    saved = pd.read_parquet(OUT / f"inputs/saved/{period}/BASE/MODEL_SUPPORT_REFERENCE_ROUTER_decisions.parquet")
    fresh = chain["support"].iloc[saved.origin_index.to_numpy(int)].reset_index(drop=True)
    for column in ("selected_parent", "model_support_status", "decision_time"):
        require(((fresh[column].isna() & saved[column].isna()) | fresh[column].eq(saved[column])).all(), "模型支持选择与原时钟不同：" + column)
    compare_numeric(fresh.support_fit_index, saved.support_fit_index, "模型支持索引")
    return rows


def check_reference_ledgers(period, chain, last_date):
    rows = []
    for model in ("DOWNSIDE_REFERENCE_RISK", "CONTINUOUS_REFERENCE_MIN_VARIANCE"):
        ledger = chain["references"][model + "_BASE"][0]
        require(ledger.mark_clock.eq("CLOSE").all(), "完整链内部收益含人工终点强平。")
        saved = pd.read_parquet(OUT / f"inputs/saved/{period}/BASE/{model}_ledger.parquet")
        a, b = ledger.loc[ledger.date.lt(last_date)], saved.loc[saved.date.lt(last_date)]
        require(pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date)), "内部收益时钟不同。")
        errors = {column: compare_numeric(a[column], b[column], f"{period}/{model}/{column}", tolerance=1e-7) for column in LEDGER_FIELDS}
        rows.append({"period": period, "model": model, "compared_rows": len(a),
                     "maximum_money_error": max(errors[k] for k in ("cash", "equity", "commission", "dividend_recognized", "dividend_paid")),
                     "maximum_return_error": errors["net_return"], "maximum_share_error": max(errors["shares"], errors["filled_quantity"]),
                     "status": "PASS_REFERENCE_CLOSE_PREFIX"})
    return rows


def save_reference(name, reference):
    for kind, frame in zip(("ledger", "decisions", "cycles"), reference):
        save_table(name + "_" + kind, frame)


def crop_continuous(continuous, end):
    # 只供历史前缀对应使用。当前段之后的意向、收益和周期资料均不进入该段。
    return {name: (frames[0].loc[frames[0].date.le(end)].reset_index(drop=True),
                   frames[1].loc[frames[1].origin.le(end)].reset_index(drop=True))
            for name, frames in continuous.items()}


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "本次完整链已开始运行，先检查现有结果，不重复覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for relative, record in frozen["files"].items():
        require(common.digest(OUT / relative) == record["sha256"], "固定文件发生变化：" + relative)
        source = record.get("source")
        if source and relative.startswith("code/"):
            require(common.digest(ROOT / source) == record["sha256"], "执行来源与固定代码不同：" + source)
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "new_strategy_evaluations": 0})
    configs = {key: json.loads((OUT / f"inputs/config/{key}.json").read_text(encoding="utf-8")) for key in CONFIGS}
    data = pd.read_parquet(OUT / "inputs/daily.parquet")
    panic = pd.read_parquet(OUT / "inputs/panic_daily.parquet")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    models114 = json.loads((OUT / "inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    models31 = json.loads((OUT / "inputs/original_models.json").read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    next_dates = {"earlier_diagnostic": pd.Timestamp("2020-01-02"), "evaluation": pd.Timestamp("2026-08-17")}
    inspected = pd.read_parquet(OUT / "inputs/already_inspected_daily.parquet", columns=["date"])
    require(data.date.iloc[-1] == pd.Timestamp("2026-08-14"), "完整链历史边界与登记不同。")
    require(inspected.date.iloc[-1] == pd.Timestamp("2026-09-16"), "已研究的历史范围变化。")
    for period, next_date in next_dates.items():
        following = inspected.loc[inspected.date.gt(pd.Timestamp(PERIODS[period][1])), "date"].iloc[0]
        require(following == next_date, "下一计划日不对应真实历史交易日。")
    continuous = engine.continuous_references(data, panic, dividends, configs["continuous_reference_min_variance"], models31, next_dates["evaluation"])
    for name, reference in continuous.items():
        save_reference("continuous/" + name, reference)
    print("两条连续内部参考已按真实收盘生成，开始逐层核对两段历史。", flush=True)
    target_checks, budget_checks, reference_checks, tails = [], [], [], []
    for period, (start, end) in PERIODS.items():
        subset = data.loc[data.date.le(pd.Timestamp(end))].reset_index(drop=True)
        chain = engine.assemble_chain(subset, dividends, configs, models114, start, next_dates[period], crop_continuous(continuous, pd.Timestamp(end)))
        for key in ("variance_budget", "joint_budget", "support", "market", "risk"):
            save_table(f"{period}/{key}", chain[key])
        for name, reference in chain["references"].items():
            save_reference(f"{period}/references/{name}", reference)
        for cost, frame in chain["factors"].items():
            save_table(f"{period}/{cost}/full_factors", frame)
            target_checks.extend(check_targets(period, cost, frame, subset))
            last = frame.iloc[-1]
            for model in MODELS:
                value = float(last[model])
                require(np.isfinite(value), "历史末端组合仍缺少上游状态。")
                tails.append({"period": period, "cost": cost, "candidate": model, "origin": subset.date.iloc[-1],
                              "planned_execution_date": next_dates[period], "core_target": float(last.TREND_NOISE_REFERENCE_BLEND),
                              "auxiliary_raw": float(last.auxiliary_raw), "auxiliary_effective": float(last[model + "_effective_auxiliary"]),
                              "candidate_target": value, "direction_intent": "POSITIVE" if value > 0 else "ZERO",
                              "support_fit_origin": last.support_fit_origin, "record_type": "HISTORICAL_RECONSTRUCTION_ONLY"})
        budget_checks.extend(check_budgets(period, chain))
        reference_checks.extend(check_reference_ledgers(period, chain, pd.Timestamp(end)))
        print(f"{period}：七层来源和两条候选在两种费用下均已对应，真实最后收盘的目标完整。", flush=True)
    target_checks, budget_checks, reference_checks, tails = map(pd.DataFrame, (target_checks, budget_checks, reference_checks, tails))
    for name, frame in (("逐层目标对应", target_checks), ("月度预算对应", budget_checks), ("内部参考收益对应", reference_checks), ("历史末端完整候选", tails)):
        save_table(name, frame)
    summary = {
        "study": "510300_POINT_CORE_OBSERVATION_V1", "at": common.now(), "status": "COMPLETE_HISTORICAL_CORE_CHAIN_PARITY_CURRENT_DATA_PENDING",
        "target_groups_compared": len(target_checks), "target_rows_compared": int(target_checks.compared_rows.sum()),
        "target_maximum_error": float(target_checks.maximum_error.max()), "budget_quantity_comparisons": len(budget_checks),
        "budget_maximum_error": float(budget_checks.maximum_error.max()), "reference_groups_compared": len(reference_checks),
        "reference_close_rows_compared": int(reference_checks.compared_rows.sum()),
        "reference_maximum_money_error": float(reference_checks.maximum_money_error.max()),
        "reference_maximum_return_error": float(reference_checks.maximum_return_error.max()),
        "reference_maximum_share_error": float(reference_checks.maximum_share_error.max()),
        "internal_reference_replays": 10, "necessary_tests_passed": 5, "new_model_fits": 0,
        "new_point_strategy_evaluations": 0, "new_investment_account_evaluations": 0, "new_prospective_observations": 0,
        "historical_chain_ready": True, "current_signal": "NO_VIEW_STALE_PRICES_AND_MONTHLY_MODELS",
        "last_reconstructed_close": "2026-08-14", "last_already_inspected_close": "2026-09-16",
        "full_current_signal_chain_ready": False, "short_candidate_ready": False, "goal_achieved": False, "orders_authorized": False,
        "remaining_work": ["恢复末端未自然结束的训练参考与样本，按原制度补充后续月度模型",
                           "补齐真实日线股息并恢复当前状态；已看过的历史只能算状态恢复",
                           "从实际预先记录的意向形成并自然完成新点位，再计算pB、净均值与年度次数"],
    }
    common.save_json(OUT / "summary.json", summary)
    write_report(summary, tails)
    common.save_json(OUT / "verification_receipt.json", {"at": common.now(), "status": "PASS_COMPLETE_HISTORICAL_CHAIN_PARITY",
                     "summary_sha256": common.digest(OUT / "summary.json"), "profitability_claim": False})
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def write_report(summary, tails):
    names = {"CORE_AUXILIARY_DRAWDOWN_GATE": "回撤限制组合", "LAG_CONFIRMED_RUNS_AUXILIARY": "相关确认组合"}
    lines = ["# 固定候选的完整收盘计算链", "",
             "**两条既有多头候选的完整历史计算链已经接通：不借用保存的父目标，从价格、既定模型和内部参考状态即可逐层恢复到最后真实收盘。当前行情与后续月份模型仍需补齐；本轮没有新增独立交易结果，盈利目标尚未验证通过。**", "",
             "用户门槛仍为胜率p乘实际净盈亏比B严格大于1，且净均值为正。标准亏损单位期望为pB−q，其中q为亏损率。年度次数是软目标；期权合约与完整投资账户绩效不属于当前验收。", "",
             "## 已经完成什么", "",
             "计算链由原60日日线开收盘与隔夜相对强弱进入参考出发，分别形成普通波动和下行风险目标；另一来源连续维护急跌回升与原岭模型参考，并按原242日收益最小方差预算组合。随后使用原共同下行预算、当时的模型支持状态及120日趋势与20日波动得到核心。辅助仍为原60日连续段状态，由60日回撤或相邻收益相关确认限制。", "",
             "所有来源保持原规则、费用、训练记录和可用时钟。内部参考真实持仓与收益是计算依赖；本轮没有新做投资账户绩效验收，也没有搜索指标阈值或改善业绩。", "",
             f"比较了{summary['target_groups_compared']}组、{summary['target_rows_compared']}条逐日目标，覆盖两个历史段、两种费用、七层来源和两条最终候选；目标最大差异{summary['target_maximum_error']:.3g}。另有{summary['budget_quantity_comparisons']}组月度预算数值对应，最大差异{summary['budget_maximum_error']:.3g}，更新日历和模型选择状态亦一致。", "",
             f"共同下行预算直接依赖的四组内部参考共有{summary['reference_close_rows_compared']}条末端前记录，金额最大差异{summary['reference_maximum_money_error']:.3g}、收益最大差异{summary['reference_maximum_return_error']:.3g}、份额最大差异{summary['reference_maximum_share_error']:.3g}。五项必要测试验证末端月首预算更新、历史前缀不受后来数据影响、未知来源保留以及拒绝错误时钟与人工强平收益。", "",
             "## 最后真实收盘的输出", "",
             "原文件末日统一开盘清仓，使完整链此前无法直接续接。现在最后一天正常执行前一收盘意向、按真实收盘记账并计算下一交易日目标，没有追加假价格。以下仅列严格费用下的历史末端状态；目标正值只代表候选方向状态，是否新入场还取决于该候选既有点位是否已经持有。", "",
             "| 历史来源日 | 下一计划日 | 候选 | 核心 | 有效辅助 | 合计目标 |", "|---|---|---|---:|---:|---:|"]
    for row in tails.loc[tails.cost.eq("STRESS")].itertuples():
        lines.append(f"| {row.origin.date()} | {row.planned_execution_date.date()} | {names[row.candidate]} | {row.core_target:.6f} | {row.auxiliary_effective:.6f} | {row.candidate_target:.6f} |")
    lines += ["", "这些输出最晚只到2026年8月14日，属于历史计算接线验证，不是当前市场判断或实际交易指令。已参与上涨解剖的日线到9月16日，也不能在更新后改称独立新样本。", "",
              "## 仍需完成的研究", "",
              "下一步恢复未自然结束的训练参考与样本，核对原月度模型制度后补充后续月份，再用真实日线与股息恢复当前状态。能够计算当前候选后，只有实际预先保存的信号及其后自然完成的点位，才可计入新验证。既有未完成持仓单列，不能在开始登记时虚构平仓或新入场。", "",
              "两条线索此前四组历史点值曾达到用户门槛，但它们来自反复观察的历史，早期表现还有大盈利集中及小亏损样本改变pB的现象。完整代码复现只证明按原规则能算出一致信号，不能解决统计可信度，也没有产生空头达线候选。", "",
              "本轮新增模型拟合、策略盈利检验、前瞻点位结果均为零；研究继续，目标未完成。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定候选的完整真实收盘链复现。")
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as error:
            common.save_json(OUT / "failure.json", {"at": common.now(), "status": "IMPLEMENTATION_DIAGNOSIS_REQUIRED", "error": str(error)})
            raise
