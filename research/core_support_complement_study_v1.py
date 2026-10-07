"""一次完整描述三种保存账户背景中的来源互补与占用冲突，不跑金融。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research import core_support_complement_inputs_v1 as inputs
from research import core_actual_acceptance_study_v1 as previous

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_core_support_complement_description_v1"
REGISTER, RESULT = "TECH.R237", "TECH.R238"
SOURCE = previous.OUT
DAILY, RECEIPTS, KEYS = previous.DAILY, previous.RECEIPTS, previous.KEYS
PERIODS, COSTS = previous.PERIODS, previous.COSTS
read, write, digest = previous.read, previous.write, previous.digest


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def folder(period,cost,policy):
    if policy == inputs.ACCOUNTS[0]:
        return previous.parent.original.CONTROL / period / cost / policy
    if policy == inputs.ACCOUNTS[1]:
        return SOURCE / "accounts" / period / cost / policy
    return previous.previous.OUT / "accounts" / period / cost / policy


def load():
    data = pd.read_parquet(DAILY)
    receipts = pd.read_parquet(RECEIPTS)
    risks = pd.read_parquet(previous.parent.original.WEIGHT / "inputs/risks.parquet")
    if len(data) != 3488 or len(receipts) != 65 or not data.symbol.eq("510300.SH").all() or not risks.idx.is_unique:
        raise ValueError("原对象、来源、日历或风险身份改变。")
    if not risks.latest_label_exit_idx.le(risks.idx).all():
        raise ValueError("原风险含尚未成熟标签。")
    accounts = {}
    for p in PERIODS:
        for c in COSTS:
            for k in inputs.ACCOUNTS:
                path = folder(p,c,k)
                value = {n:pd.read_parquet(path / f"{n}.parquet") for n in previous.ACCOUNT_TABLES}
                value["terminal"] = read(path / "terminal.json")
                accounts[(p,c,k)] = value
    return data,receipts,risks,accounts


def table(name,frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.with_suffix(".parquet").exists():
        raise RuntimeError("描述输出已存在，不覆盖："+name)
    frame.to_parquet(path.with_suffix(".parquet"),index=False)
    frame.to_csv(path.with_suffix(".csv"),index=False,encoding="utf-8-sig")


def source_paths():
    paths = [Path(__file__),Path(inputs.__file__),ROOT / "tests/test_core_support_complement_v1.py",
        ROOT / "docs/510300_CORE_SUPPORT_COMPLEMENT_DESCRIPTION_V1.md",Path(previous.__file__),
        Path(inputs.budget.__file__),ROOT / "research/point_account_nr7_inputs_v1.py",DAILY,RECEIPTS,KEYS,
        previous.parent.original.WEIGHT / "inputs/risks.parquet",SOURCE / "summary.json",SOURCE / "protocol.json",
        SOURCE / "next_core_support_source_complement_description_proposal.json",
        previous.previous.OUT / "protocol.json",previous.previous.OUT / "implementation_v1_0_1/summary.json",
        ROOT / "research/macro_technical_first_passage_inputs_v1.py",OUT / "tests_receipt.json"]
    for p in PERIODS:
        for c in COSTS:
            for k in inputs.ACCOUNTS:
                path = folder(p,c,k)
                paths.extend(path / f"{n}.parquet" for n in previous.ACCOUNT_TABLES)
                paths.append(path / "terminal.json")
    return list(dict.fromkeys(paths))


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("来源互补描述已登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    if tests["passed"] != 5 or tests["exit_code"] != 0 or tests["inputs_sha256"] != digest(Path(inputs.__file__)):
        raise ValueError("五必要测试或实现版本不同。")
    state = read(ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json")
    if state["latest_technical_decision"] != "TECH.R236" or state["latest_actual_financial_decision"] != "TECH.R236":
        raise ValueError("项目已推进，不能在旧状态登记。")
    write(OUT / "protocol.json", {"at":previous.parent.original.now(),"study":"510300_CORE_SUPPORT_COMPLEMENT_DESCRIPTION_V1",
        "registration":REGISTER,"decision":RESULT,"rule_card":"docs/510300_CORE_SUPPORT_COMPLEMENT_DESCRIPTION_V1.md",
        "known_before_freeze":"R236全部金融失败和原12点位压力背景预览已知；5个原A潜在补充已看到，非盲或独立。",
        "scope":"原3488日/65节点、12支持接受点、55原A进入、三保存背景两时期两费用12账户；全部成功失败和开放。",
        "decision_role":"已有持仓消费事件，不同开盘重买；原CORE正优先、未知NO_VIEW，明确零且空仓/成熟风险/价格已知才有潜在补充尝试。",
        "open_role":"开盘发生后才核对涨跌停/已知锚及原背景一次容量；不更新共同账户现金或计算收益。",
        "entry_role":"原A实际进入开盘已发生后才形成身份；其事前当前持仓背景和支持来源身份仅用先前真实进入。",
        "result_annotations":"所有已完成损益只在输出另存参考，不进资格/容量/前缀，不能相加为组合收益。",
        "source_clock":"公布上界/首次ETF观察均<=固定锚，锚<接受；同源不独立投票、未知保留。",
        "periods":PERIODS,"costs":list(COSTS),"saved_accounts":12,"necessary_tests":5,"original_key_prefixes":19,
        "accepted_signal_cutoffs":12,"new_accounts":0,"new_fits":0,"new_training_labels":0,"new_market_requests":0,
        "parameter_grid":False,"history_role":"DEVELOPMENT_CALIBRATION","independent_validation":"NOT_ESTABLISHED",
        "new_strategy_return_sharpe":"NOT_COMPUTED","goal_achieved":False,
        "no_rescue":"不挑成功点位、拼接时期、重用已消费公告、改R232/R236、改变风险或费用；描述后才能提出不同整体机制。",
        "sources":[{"path":relative(p),"sha256":digest(p)} for p in source_paths()]})
    print("R237完整12点位/55原A入口来源互补描述已固定，0金融。",flush=True)


def verify_prefix(data,receipts,risks,accounts,points,contexts,entries,cut):
    prefix_data = data.loc[data.date.le(cut)].reset_index(drop=True)
    p,c,e = inputs.describe(prefix_data,receipts,accounts,risks,PERIODS)
    pd.testing.assert_frame_equal(p,points.loc[points.origin.le(cut)].reset_index(drop=True),check_exact=True)
    known = inputs.known_context_columns(contexts)
    pd.testing.assert_frame_equal(c[known],contexts.loc[contexts.origin.le(cut),known].reset_index(drop=True),check_exact=True)
    at = pd.Timestamp(cut).tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)
    seen = contexts.background_actual_next_order_observed_at.notna() & contexts.background_actual_next_order_observed_at.le(at)
    expected = contexts.loc[seen].reset_index(drop=True)
    actual = c.loc[c.background_actual_next_order_observed_at.notna() & c.background_actual_next_order_observed_at.le(at)].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    pd.testing.assert_frame_equal(e,entries.loc[entries.original_A_actual_entry_date.le(cut)].reset_index(drop=True),check_exact=True)
    current = c.loc[c.origin.eq(cut)]
    if len(current) and (current.background_actual_next_BUY_quantity.notna().any() or current.background_actual_next_order_observed_at.notna().any()):
        raise AssertionError("接受日截断显示了次日实际订单。")
    return {"cut":str(pd.Timestamp(cut).date()),"all_decision_features_exact":True,"all_already_observed_open_outcomes_exact":True,
        "all_already_observed_original_entries_exact":True,"decision_context_rows":len(c),"seen_open_context_rows":len(actual),
        "current_signal_next_open_unknown":True}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("来源互补描述已经开始，不重复。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结来源改变："+item["path"])
    write(OUT / "RUN_STARTED.json",{"at":previous.parent.original.now(),"new_financial_accounts":0})
    data,receipts,risks,accounts = load()
    points,contexts,entries = inputs.describe(data,receipts,accounts,risks,PERIODS)
    if len(points) != 12 or len(contexts) != 144 or len(entries) != 330:
        raise ValueError("原全部点位或账户背景范围不完整。")
    table("全部12支持接受_公布锚价格时钟与宏观原值",points)
    table("全部144点位背景_当时资格与后来一次容量分开",contexts)
    table("全部330原A实际进入背景_支持持有与冲突身份",entries)
    checks = []
    for cut in pd.read_parquet(KEYS).date:
        checks.append(verify_prefix(data,receipts,risks,accounts,points,contexts,entries,cut))
    signals = [verify_prefix(data,receipts,risks,accounts,points,contexts,entries,cut) for cut in points.origin]
    if len(checks) != 19 or len(signals) != 12:
        raise ValueError("原关键日或所有接受日时序核对不完整。")
    table("原19关键日_完整时序与已发生开盘前缀",pd.DataFrame(checks))
    table("全部12接受日_次开尚未发生必须未知",pd.DataFrame(signals))
    roles = contexts.groupby(["period","cost","background_policy","decision_role"],dropna=False).size().reset_index(name="contexts_not_independent_opportunities")
    table("全部背景角色与未知_费用复本不作独立机会",roles)
    refs = []
    for (p,c,k),account in accounts.items():
        t = inputs.normalized(account["trades"])
        for field,value in (("background_policy",k),("cost",c),("period",p)):
            t.insert(0,field,value)
        t["label_role"] = "FINAL_SAVED_RESULT_REFERENCE_NOT_PREDICTION_NOT_COMBINED_RETURN"
        refs.append(t)
    reference = pd.concat(refs,ignore_index=True)
    table("全部保存周期和开放_终态损益仅参考不进入资格",reference)
    eligible = contexts.loc[contexts.potential_attempt_at_decision]
    active_conflicts = entries.loc[entries.support_held_at_original_A_origin].copy()
    birth_map = contexts.loc[contexts.background_policy.eq(inputs.ACCOUNTS[0]),["period","cost","origin","potential_attempt_at_decision"]]
    active_conflicts = active_conflicts.merge(birth_map,left_on=["period","cost","support_cycle_entry_origin_at_Aorigin"],
        right_on=["period","cost","origin"],how="left",validate="many_to_one")
    table("全部支持独立持仓与原A冲突_原补充出生资格另列",active_conflicts)
    write(OUT / "summary.json",{"at":previous.parent.original.now(),"registration":REGISTER,"decision":RESULT,
        "status":"CORE_SUPPORT_SOURCE_COMPLEMENT_ALL_SAVED_CONTEXTS_DESCRIPTION_COMPLETED","saved_accounts":12,
        "original_observations":3488,"source_receipts":65,"all_support_points":12,"all_context_rows":len(contexts),
        "all_original_A_entry_context_rows":len(entries),"all_saved_cycle_reference_replicates":len(reference),
        "all_saved_complete_reference_replicates":int(reference.status.eq("COMPLETE").sum()),
        "all_saved_open_reference_replicates":int(reference.status.eq("RIGHT_CENSORED").sum()),
        "potential_complement_contexts_not_independent":len(eligible),"original_key_prefixes":checks,"all_signal_cutoffs":signals,
        "all_saved_support_A_conflict_context_rows":len(active_conflicts),"new_accounts":0,"new_fits":0,"new_training_labels":0,
        "new_market_requests":0,"necessary_tests_passed":5,"new_strategy_return_sharpe":"NOT_COMPUTED",
        "latest_actual_financial_decision_unchanged":"TECH.R236","independent_validation":"NOT_ESTABLISHED","goal_achieved":False})
    print("R238全12支持点位、144背景、330原A进入及31截断完成；0新金融。",flush=True)


def main():
    parser = argparse.ArgumentParser(description="CORE与支持来源完整时钟/资金冲突描述")
    parser.add_argument("action",choices=("freeze","run"))
    {"freeze":freeze,"run":run}[parser.parse_args().action]()


if __name__ == "__main__":
    main()
