"""唯一共同现金/持仓归属实验：适配后登记、一次完成所有场景。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import shared_cash_source_ownership_rules_v1 as rules
from research import shared_cash_source_ownership_account_v1 as execution
from research import core_actual_acceptance_study_v1 as previous
from research import core_support_complement_completion_1_0_1 as description

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_shared_cash_source_ownership_v1"
REGISTER,RESULT = "TECH.R239","TECH.R240"
PERIODS,COSTS = previous.PERIODS,previous.COSTS
SAVED_CORE,SAVED_SUPPORT,SAVED_ACCEPTANCE = previous.CONTROL_A,rules.support.POLICIES[0],previous.inputs.POLICIES[0]
SAVED = (SAVED_CORE,SAVED_SUPPORT,SAVED_ACCEPTANCE)
CONTROLS = (*rules.POLICIES[1:],*SAVED)
NAMES = {**rules.NAMES,SAVED_CORE:"原A",SAVED_SUPPORT:"原独立支持",SAVED_ACCEPTANCE:"R236原A接受保护/延续"}
ACCOUNT_TABLES = previous.ACCOUNT_TABLES
read,write,digest = previous.read,previous.write,previous.digest


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name,frame):
    p = OUT / "results" / name
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.with_suffix(".parquet").exists() or p.with_suffix(".csv").exists():
        raise RuntimeError("完整结果不覆盖："+name)
    frame.to_parquet(p.with_suffix(".parquet"),index=False)
    frame.to_csv(p.with_suffix(".csv"),index=False,encoding="utf-8-sig")


def saved_account(period,cost,policy):
    if policy == SAVED_CORE:
        p = previous.parent.original.CONTROL / period / cost / policy
    elif policy == SAVED_SUPPORT:
        p = previous.previous.OUT / "accounts" / period / cost / policy
    elif policy == SAVED_ACCEPTANCE:
        p = previous.OUT / "accounts" / period / cost / policy
    else:
        raise ValueError("未知保存对照。")
    result = {n:pd.read_parquet(p / f"{n}.parquet") for n in ACCOUNT_TABLES}
    result["terminal"] = read(p / "terminal.json")
    return result


def load():
    data,receipts,dividends,risks,parents = previous.load()
    if len(data) != 3488 or len(receipts) != 65 or not risks.latest_label_exit_idx.le(risks.idx).all():
        raise ValueError("原范围或风险成熟时钟改变。")
    return data,receipts,dividends,risks,parents


def preflight():
    if (OUT / "control_and_prefix_preflight.json").exists() or (OUT / "protocol.json").exists():
        raise RuntimeError("适配核对已开始或登记，不重复。")
    receipt = read(OUT / "tests_receipt.json")
    if receipt["passed"] != 11 or receipt["exit_code"] != 0:
        raise ValueError("十一项必要测试未通过。")
    for name,p in (("rules_sha256",Path(rules.__file__)),("account_sha256",Path(execution.__file__)),
                   ("tests_sha256",ROOT / "tests/test_shared_cash_source_ownership_v1.py")):
        if digest(p) != receipt[name]:
            raise ValueError("必要测试代码版本不同。")
    if not (description.OUT / "project_state_update_receipt.json").exists():
        raise ValueError("完整来源互补及一次归档尚未完成。")
    description.frozen_exact()
    data,_,dividends,risks,parents = load()
    prefixes = []
    for cut in pd.read_parquet(previous.KEYS).date:
        period = "2015_2019" if pd.Timestamp(cut).year < 2020 else "2020_2026"
        original = parents[period]
        at = original.loc[original.origin.le(cut)].copy().reset_index(drop=True)
        check = original.copy().set_index("origin").loc[:cut].reset_index()
        pd.testing.assert_frame_equal(at,check,check_exact=True)
        prefixes.append({"cut":str(pd.Timestamp(cut).date()),"raw_CORE_target_prefix_exact":True,
            "support_source_prefix_scope":"R238_ALL_DECISION_AND_SEEN_OPEN_31_PREFIXES_ALREADY_EXACT_NOT_REPEATED"})
    checks = []
    for period,(start,end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        for cost in COSTS:
            for adapter,reference in ((rules.BASELINE_CORE,SAVED_CORE),(rules.BASELINE_SUPPORT,SAVED_SUPPORT)):
                result = execution.account(local,dividends,parents[period],risks,adapter,cost,start)
                saved = saved_account(period,cost,reference)
                for name in ACCOUNT_TABLES[:3]:
                    previous.exact_saved_table(result[name],saved[name])
                checks.append({"period":period,"cost":cost,"adapter":adapter,"reference":reference,
                    "all_saved_daily_orders_trades_exact":True,**previous.parent.verify_account(result)})
                print(f"{period}/{cost}/{NAMES[adapter]}完整日净值/订单/周期精确复现。",flush=True)
    if len(checks) != 8 or len(prefixes) != 19:
        raise ValueError("全部适配/源前缀未完成。")
    write(OUT / "control_and_prefix_preflight.json",{"at":previous.parent.original.now(),"adapter_checks":checks,
        "all_original_parent_prefixes":prefixes,"saved_support_context_prefixes_reused":31,
        "synthetic_schema_failure_preserved":"tests_first_pre_freeze_failure.json","economic_rules_changed":False,
        "adapter_first_string_schema_failure_preserved":"preflight_first_string_schema_failure.json",
        "successful_original_adapter_accounts":8,"original_failed_adapter_attempts":1,
        "new_candidate_full_accounts":0,"new_fits":0,"new_labels":0})


def source_paths():
    paths = [Path(__file__),Path(rules.__file__),Path(execution.__file__),ROOT / "tests/test_shared_cash_source_ownership_v1.py",
        ROOT / "docs/510300_SHARED_CASH_SOURCE_OWNERSHIP_V1.md",Path(rules.support.__file__),Path(previous.__file__),
        Path(previous.parent.__file__),Path(previous.parent.measurements.__file__),Path(execution.budget.__file__),Path(previous.controls.__file__),
        previous.DAILY,previous.RECEIPTS,previous.KEYS,
        previous.parent.original.WEIGHT / "inputs/dividends.csv",previous.parent.original.WEIGHT / "inputs/risks.parquet",
        previous.parent.original.WEIGHT / "inputs/parent_signals.parquet",previous.parent.original.WEIGHT / "inputs/earlier_signals.parquet",
        previous.previous.OUT / "protocol.json",previous.previous.OUT / "implementation_v1_0_1/summary.json",
        previous.OUT / "protocol.json",previous.OUT / "summary.json",description.OUT / "protocol.json",description.OUT / "summary.json",
        description.OUT / "next_shared_cash_source_ownership_proposal.json",description.OUT / "post_run_diagnosis.json",
        description.OUT / "project_state_update_receipt.json",OUT / "tests_receipt.json",OUT / "control_and_prefix_preflight.json",
        OUT / "tests_first_pre_freeze_failure.json",OUT / "pre_freeze_original_account_schema.py",
        OUT / "preflight_first_string_schema_failure.json",OUT / "tests_receipt_before_string_schema_fix.json"]
    for period in PERIODS:
        for cost in COSTS:
            for policy in SAVED:
                if policy == SAVED_CORE:
                    p = previous.parent.original.CONTROL / period / cost / policy
                elif policy == SAVED_SUPPORT:
                    p = previous.previous.OUT / "accounts" / period / cost / policy
                else:
                    p = previous.OUT / "accounts" / period / cost / policy
                paths.extend(p / f"{n}.parquet" for n in ACCOUNT_TABLES)
                paths.append(p / "terminal.json")
    return list(dict.fromkeys(paths))


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("共同账户已登记，不重复。")
    pre = read(OUT / "control_and_prefix_preflight.json")
    state = read(ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json")
    if len(pre["adapter_checks"]) != 8 or len(pre["all_original_parent_prefixes"]) != 19:
        raise ValueError("唯一动作的原账户适配尚未完成。")
    if state["latest_technical_decision"] != "TECH.R238" or state["latest_actual_financial_decision"] != "TECH.R236":
        raise ValueError("当前项目状态已经推进。")
    protocol = {"at":previous.parent.original.now(),"registration":REGISTER,"decision":RESULT,
        "study":"510300_SHARED_CASH_SOURCE_OWNERSHIP_V1","primary":rules.POLICIES[0],"policies":list(rules.POLICIES),
        "saved_controls":list(SAVED),"all_comparison_controls":list(CONTROLS),"new_candidate_accounts":12,"saved_accounts_reused":12,
        "known_before_freeze":"12支持点、5潜在互补及三旧冲突/两合格出生全已知；原R232/R236完整失败已知，全部历史开发。",
        "rule_card":"docs/510300_SHARED_CASH_SOURCE_OWNERSHIP_V1.md","entry":"空仓CORE正优先/未知不补充，原CORE明确零才尝试当天支持接受；身份当日消费。",
        "ownership":"首次真实买入固定CORE/COMPLEMENT；CORE沿原A，补充沿R232各退出且不补买；持仓中不转身份、另开现金账户或同开复买。",
        "account":read(previous.previous.OUT / "protocol.json")["account"],
        "periods":PERIODS,"costs":read(previous.previous.OUT / "protocol.json")["costs"],
        "necessary_tests":11,"original_A_adapter_replays":4,"original_support_adapter_replays":4,
        "raw_CORE_target_prefixes":19,"source_context_prefixes_reused_not_recomputed":31,
        "economic_gate":"四场景主CAGR/Sharpe正且同时高于五对照，DD<=10%，实际净pB>1/标准期望正；次数软。",
        "stability":"五对照原20/252循环块各2000、种子510300154，所有CAGR/Sharpe增量95%下界正。",
        "parameter_grid":False,"financial_run_budget":1,"new_fits":0,"new_training_labels":0,"new_market_requests":0,
        "history_role":"DEVELOPMENT_CALIBRATION","first_vintage":"NOT_CERTIFIED","independent_validation":"NOT_ESTABLISHED",
        "search_selection_correction":"NOT_COMPUTED","goal_achieved":False,
        "no_rescue":"不按赢家/PMI另设门槛或拼时期，不修旧退出/周纯度，失败本配置不重跑；E03独立保持。",
        "sources":[{"path":relative(p),"sha256":digest(p)} for p in source_paths()]}
    write(OUT / "protocol.json",protocol)
    print("R239唯一共同现金/来源归属用途固定；12新完整账户尚未运行。",flush=True)


def check_timing(result,local):
    trades,orders,decisions = result["trades"],result["orders"],result["decisions"]
    if len(orders) and not orders.origin.lt(orders.date).all():
        raise AssertionError("实际订单没有先前收盘。")
    if len(trades) and not trades.entry_origin.lt(trades.entry_date).all():
        raise AssertionError("实际进入早于已知决定。")
    if not decisions.complement_event_consumed_today.eq(decisions.entry_event).all():
        raise AssertionError("接受事件未在当日消费。")
    for t in trades.itertuples():
        owned = orders.loc[orders.cycle_id.eq(t.cycle_id)]
        if not owned.owner.eq(t.owner).all():
            raise AssertionError("一个交易周期被另一个来源转归属。")
        if t.owner == "COMPLEMENT":
            d = decisions.loc[decisions.origin.eq(t.entry_origin)].iloc[0]
            if d.raw_saved_CORE_weight != 0 or d.shares_before != 0 or d.current_complement_event_role != "CURRENT_ZERO_CORE_COMPLEMENT_ATTEMPT":
                raise AssertionError("补充不是实际空仓且原源明确零。")
            if len(owned.loc[owned.side.eq("BUY")]) != 1:
                raise AssertionError("补充持有中补买。")
            at = local.loc[local.date.eq(t.entry_date)].iloc[0]
            if not at.open+at.cash_shift > d.support_at_origin+EPS:
                raise AssertionError("补充在开盘失效锚以下买入。")
    return {"next_open_and_fixed_owner_timing_passed":True,"cycles":len(trades),
        "complement_cycles":int(trades.owner.eq("COMPLEMENT").sum()),"core_cycles":int(trades.owner.eq("CORE").sum()),
        "accepted_events_consumed":int(decisions.entry_event.sum())}


EPS = 1e-10


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("共同账户金融已开始，不重复。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结来源改变："+item["path"])
    write(OUT / "RUN_STARTED.json",{"at":previous.parent.original.now(),"new_accounts_planned":12,"financial_run_number":1})
    data,_,dividends,risks,parents = load()
    accounts,metrics,annual,checks,timing = {},[],[],[],[]
    for period,(start,end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        for cost in COSTS:
            for policy in (*SAVED,*rules.POLICIES):
                if policy in rules.POLICIES:
                    result = execution.account(local,dividends,parents[period],risks,policy,cost,start)
                    checks.append({"period":period,"cost":cost,"policy":policy,**previous.parent.verify_account(result)})
                    timing.append({"period":period,"cost":cost,"policy":policy,**check_timing(result,local)})
                    p = OUT / "accounts" / period / cost / policy
                    p.mkdir(parents=True,exist_ok=True)
                    for name in (*ACCOUNT_TABLES,"ownership_evidence"):
                        result[name].to_parquet(p / f"{name}.parquet",index=False)
                    write(p / "terminal.json",result["terminal"])
                else:
                    result = saved_account(period,cost,policy)
                accounts[(period,cost,policy)] = result
                stats,years = previous.controls.statistics(result,period,cost,policy)
                metrics.append({"period":period,"cost":cost,"policy":policy,**stats})
                annual.extend(years)
                if policy in rules.POLICIES:
                    print(f"{period}/{cost}/{NAMES[policy]}：净年化{stats['net_cagr']:.4%}、净夏普{stats['net_sharpe']:.6f}、完成{stats['completed_cycles']}、pB{stats['p_times_b']:.6f}。",flush=True)
    frame = pd.DataFrame(metrics)
    table("全部24完整账户_共同现金来源归属",frame)
    table("全部逐年完整次数与净收益",pd.DataFrame(annual))
    table("全部12新账户现金复算",pd.DataFrame(checks))
    table("全部12新账户来源时钟和归属核对",pd.DataFrame(timing))
    write(OUT / "financial_accounts_completed.json",{"at":previous.parent.original.now(),"new_accounts_completed":12,
        "saved_accounts_reused":12,"metrics":frame.to_dict("records"),"all_accounts_saved_before_intervals":True})
    comparisons,gates,interval_rows = [],[],[]
    for period in PERIODS:
        for cost in COSTS:
            group = frame.loc[frame.period.eq(period) & frame.cost.eq(cost)].set_index("policy")
            primary = group.loc[rules.POLICIES[0]]
            conditions = {"positive_cagr_sharpe":bool(primary.net_cagr > 0 and primary.net_sharpe > 0),
                "drawdown_within_10pct":bool(primary.max_drawdown <= .1),"actual_pB_above_1":bool(primary.p_times_b > 1),
                "actual_standard_EV_positive":bool(primary.standard_expectancy_loss_units > 0)}
            for control in CONTROLS:
                reference = group.loc[control]
                conditions["beats_"+control] = bool(primary.net_cagr > reference.net_cagr and primary.net_sharpe > reference.net_sharpe)
                old,new = accounts[(period,cost,control)]["daily"],accounts[(period,cost,rules.POLICIES[0])]["daily"]
                np.testing.assert_array_equal(pd.to_datetime(old.date).astype("datetime64[ns]"),pd.to_datetime(new.date).astype("datetime64[ns]"))
                intervals = previous.parent.intervals(old.net_return.to_numpy(float),new.net_return.to_numpy(float))
                comparison = {"period":period,"cost":cost,"control":control,"cagr_delta":primary.net_cagr-reference.net_cagr,
                    "sharpe_delta":primary.net_sharpe-reference.net_sharpe,"intervals":intervals}
                comparisons.append(comparison)
                for value in intervals:
                    interval_rows.append({"period":period,"cost":cost,"control":control,"block":value["block"],"resamples":value["resamples"],
                        "cagr_delta":comparison["cagr_delta"],"sharpe_delta":comparison["sharpe_delta"],
                        "cagr_delta_lower":value["cagr_delta_95"][0],"cagr_delta_upper":value["cagr_delta_95"][1],
                        "sharpe_delta_lower":value["sharpe_delta_95"][0],"sharpe_delta_upper":value["sharpe_delta_95"][1]})
            gates.append({"period":period,"cost":cost,**conditions,"economic_passed":all(conditions.values())})
            print(f"{period}/{cost}五对照两尺度完整区间已保存。",flush=True)
    table("全部五对照两尺度40净增量区间",pd.DataFrame(interval_rows))
    table("全部四场景经济门",pd.DataFrame(gates))
    stable = all(x["cagr_delta_95"][0] > 0 and x["sharpe_delta_95"][0] > 0 for c in comparisons for x in c["intervals"])
    passed = all(x["economic_passed"] for x in gates)
    write(OUT / "summary.json",{"at":previous.parent.original.now(),"registration":REGISTER,"decision":RESULT,
        "status":"HISTORICAL_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passed and stable else "REJECTED_FIXED_SHARED_CASH_SOURCE_OWNERSHIP_FULL_ACCOUNT_GATES_NOT_MET",
        "all_economic_gates_passed":passed,"historical_stability_passed":stable,"gates":gates,"metrics":metrics,"comparisons":comparisons,
        "timing_checks":timing,"new_accounts":12,"saved_controls_reused":12,"saved_account_checks":len(checks),
        "original_A_adapter_replays":4,"original_support_adapter_replays":4,"necessary_tests_passed":11,
        "raw_CORE_target_prefixes":19,"saved_context_prefixes_reused":31,"new_fits":0,"new_training_labels":0,"new_market_requests":0,
        "history_role":"DEVELOPMENT_CALIBRATION","first_vintage":"NOT_CERTIFIED","independent_validation":"NOT_ESTABLISHED",
        "search_selection_correction":"NOT_COMPUTED","overfitting_removed":False,"goal_achieved":False})
    print("R240完整共同现金来源归属终态，原配置不按结果改变。",flush=True)


def main():
    parser = argparse.ArgumentParser(description="共同现金来源归属固定金融研究")
    parser.add_argument("action",choices=("preflight","freeze","run"))
    {"preflight":preflight,"freeze":freeze,"run":run}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
