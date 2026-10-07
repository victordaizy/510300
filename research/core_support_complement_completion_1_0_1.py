"""修正描述矩阵计数，不改原冻结经济动作或重跑金融。"""
from __future__ import annotations

from pathlib import Path
import traceback

import pandas as pd

from research import core_support_complement_study_v1 as original

ROOT = original.ROOT
OUT = original.OUT / "implementation_v1_0_1"
read, write, digest = original.read, original.write, original.digest


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.with_suffix(".parquet").exists() or path.with_suffix(".csv").exists():
        raise RuntimeError("已有描述表不覆盖："+name)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def frozen_exact():
    old = read(original.OUT / "protocol.json")
    for item in old["sources"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原冻结来源改变："+item["path"])
    if (OUT / "protocol.json").exists():
        for item in read(OUT / "protocol.json")["completion_sources"]:
            if digest(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("接续来源改变："+item["path"])
    return len(old["sources"])


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("接续已登记，不重复。")
    frozen_exact()
    diagnosis = read(original.OUT / "count_failure_diagnosis.json")
    tests = read(original.OUT / "tests_receipt.json")
    if diagnosis["actual_points"] != 12 or diagnosis["actual_contexts"] != 72 or diagnosis["actual_A_entry_contexts"] != 330:
        raise ValueError("实际身份范围不是原完整范围。")
    if tests["passed"] != 5 or tests["exit_code"] != 0:
        raise ValueError("原必要测试未通过。")
    failure = read(original.OUT / "RUN_FAILURE.json")
    if "原全部点位或账户背景范围不完整" not in failure["traceback"] or failure["financial_runs"] != 0:
        raise ValueError("原失败不是计数断言。")
    paths = [Path(__file__), ROOT / "docs/510300_CORE_SUPPORT_COMPLEMENT_COMPLETION_1_0_1.md",
             original.OUT / "protocol.json", original.OUT / "RUN_STARTED.json",
             original.OUT / "RUN_FAILURE.json", original.OUT / "count_failure_diagnosis.json"]
    write(OUT / "protocol.json", {"at":original.previous.parent.original.now(),
        "registration":original.REGISTER,"decision":original.RESULT,"version":"1.0.1_COUNT_IDENTITY_ONLY",
        "scope":"原12点、55原A进入、三背景两费用两时期；各事件仅属于一个时期。",
        "correction":"原错误144行修正为完整身份矩阵72行；资格/时钟/容量函数逐字未改。",
        "full_saved_outputs_before_completion":0,"original_in_memory_attempts":1,
        "diagnostic_reconstruction_only":1,"new_accounts":0,"new_financial_runs":0,
        "necessary_original_tests_passed":5,"prefix_scope":{"original_keys":19,"signal_days":12},
        "new_strategy_return_sharpe":"NOT_COMPUTED","independent_validation":"NOT_ESTABLISHED",
        "no_relaxation":"保持所有原字段逐值前缀及同日下一开盘未知；真实值失败保留，不调规则。",
        "completion_sources":[{"path":original.relative(p),"sha256":digest(p)} for p in paths]})
    print("来源描述1.0.1完整身份计数修正已登记，原失败留档，0金融。", flush=True)


def verify_identity(points, contexts, entries, accounts):
    expected = {(r.point_id,r.period,c,k) for r in points.itertuples()
                for c in original.COSTS for k in original.inputs.ACCOUNTS}
    actual = set(contexts[["point_id","period","cost","background_policy"]].itertuples(index=False,name=None))
    if len(contexts) != len(expected) or actual != expected:
        raise ValueError("点位完整背景身份不一致。")
    expected_entries = set()
    for (p,c,k),a in accounts.items():
        if k == original.inputs.ACCOUNTS[0]:
            expected_entries.update((p,c,r.cycle_id,b) for r in a["trades"].itertuples() for b in original.inputs.ACCOUNTS)
    actual_entries = set(entries[["period","cost","original_A_cycle_id","background_policy"]].itertuples(index=False,name=None))
    if len(entries) != len(expected_entries) or actual_entries != expected_entries:
        raise ValueError("全部原A进入背景身份不一致。")
    if len(points) != 12 or points.point_id.duplicated().any():
        raise ValueError("原完整点位身份改变。")
    return {"point_background_matrix_exact":True,"original_A_entry_background_matrix_exact":True,
            "point_rows":len(points),"context_rows":len(contexts),"entry_rows":len(entries)}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("接续已开始，不重复生成。")
    frozen = frozen_exact()
    write(OUT / "RUN_STARTED.json", {"at":original.previous.parent.original.now(),"new_accounts":0,"new_financial_runs":0})
    data,receipts,risks,accounts = original.load()
    points,contexts,entries = original.inputs.describe(data,receipts,accounts,risks,original.PERIODS)
    identity = verify_identity(points,contexts,entries,accounts)
    table("全部12支持接受_公布锚价格时钟与宏观原值",points)
    table("全部72点位背景_当时资格与后来一次容量分开",contexts)
    table("全部330原A实际进入背景_支持持有与冲突身份",entries)
    checks = [original.verify_prefix(data,receipts,risks,accounts,points,contexts,entries,cut)
              for cut in pd.read_parquet(original.KEYS).date]
    signals = [original.verify_prefix(data,receipts,risks,accounts,points,contexts,entries,cut) for cut in points.origin]
    if len(checks) != 19 or len(signals) != 12:
        raise ValueError("原完整截断范围不足。")
    table("原19关键日_完整时序与已发生开盘前缀",pd.DataFrame(checks))
    table("全部12接受日_次开尚未发生必须未知",pd.DataFrame(signals))
    roles = contexts.groupby(["period","cost","background_policy","decision_role"],dropna=False).size().reset_index(name="contexts_not_independent_opportunities")
    table("全部背景角色与未知_费用复本不作独立机会",roles)
    references = []
    for (p,c,k),account in accounts.items():
        t = original.inputs.normalized(account["trades"])
        for field,value in (("background_policy",k),("cost",c),("period",p)):
            t.insert(0,field,value)
        t["label_role"] = "FINAL_SAVED_RESULT_REFERENCE_NOT_PREDICTION_NOT_COMBINED_RETURN"
        references.append(t)
    reference = pd.concat(references,ignore_index=True)
    table("全部保存周期和开放_终态损益仅参考不进入资格",reference)
    active = entries.loc[entries.support_held_at_original_A_origin].copy()
    births = contexts.loc[contexts.background_policy.eq(original.inputs.ACCOUNTS[0]),
                         ["period","cost","origin","potential_attempt_at_decision"]]
    active = active.merge(births,left_on=["period","cost","support_cycle_entry_origin_at_Aorigin"],
                          right_on=["period","cost","origin"],how="left",validate="many_to_one")
    if len(active) and active.potential_attempt_at_decision.isna().any():
        raise ValueError("持有冲突未能匹配原事件出生资格。")
    table("全部支持独立持仓与原A冲突_原补充出生资格另列",active)
    write(OUT / "summary.json", {"at":original.previous.parent.original.now(),"registration":original.REGISTER,
        "decision":original.RESULT,"status":"CORE_SUPPORT_ALL_SAVED_CONTEXTS_DESCRIPTION_COMPLETED_COUNT_CORRECTION_ONLY",
        "implementation_version":"1.0.1_COUNT_IDENTITY_ONLY","original_failed_count_assertion_preserved":True,
        "saved_accounts":12,"original_observations":3488,"source_receipts":65,"all_support_points":len(points),
        "all_context_rows":len(contexts),"all_original_A_entry_context_rows":len(entries),"identity_matrix_checks":identity,
        "all_saved_cycle_reference_replicates":len(reference),
        "all_saved_complete_reference_replicates":int(reference.status.eq("COMPLETE").sum()),
        "all_saved_open_reference_replicates":int(reference.status.eq("RIGHT_CENSORED").sum()),
        "potential_complement_contexts_not_independent":int(contexts.potential_attempt_at_decision.sum()),
        "original_key_prefixes":checks,"all_signal_cutoffs":signals,
        "all_saved_support_A_conflict_context_rows":len(active),"frozen_original_sources_exact":frozen,
        "new_accounts":0,"new_financial_runs":0,"new_fits":0,"new_training_labels":0,"new_market_requests":0,
        "necessary_tests_passed":5,"full_saved_output_generations":1,
        "new_strategy_return_sharpe":"NOT_COMPUTED","latest_actual_financial_decision_unchanged":"TECH.R236",
        "independent_validation":"NOT_ESTABLISHED","goal_achieved":False})
    print("R238接续完成：12点/72背景/330原A进入、31截断全部精确，0金融。",flush=True)


def main():
    freeze()
    try:
        run()
    except Exception:
        write(OUT / "RUN_FAILURE.json", {"at":original.previous.parent.original.now(),
            "traceback":traceback.format_exc(),"new_financial_runs":0,"no_automatic_replay":True})
        raise


if __name__ == "__main__":
    main()
