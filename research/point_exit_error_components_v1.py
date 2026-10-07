"""原退出预测误差的恒等分解；未来周期均值仅用于归因，不能用于交易。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.point_account_cashflow_state_v1 import digest, now, require, write_json


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_point_exit_sign_calibration_v1"
OUT = PARENT / "error_components"
STUDY = "510300_POINT_EXIT_ERROR_COMPONENTS_V1"


def decompose(frame):
    """同一周期总权重一；总体误差方差使用ddof=0，保证逐周期精确恒等。"""
    known = frame.loc[np.isfinite(frame.prediction) & np.isfinite(frame.target)].copy()
    require(not known.duplicated(["cycle_id", "origin_index"]).any(), "原误差原点重复。")
    rows = []
    for cycle_id, group in known.groupby("cycle_id", sort=True):
        errors = group.target.to_numpy(float) - group.prediction.to_numpy(float)
        bias = float(errors.mean())
        total = float(np.mean(errors**2))
        bias_square = bias**2
        centered = float(np.mean((errors-bias)**2))
        identity = abs(total-bias_square-centered)
        require(identity <= 1e-12, "周期平均偏差与周期内误差未还原原MSE。")
        rows.append({"cycle_id": int(cycle_id), "rows": len(group), "entry_year": int(group.entry_year.iloc[0]),
                     "mean_prediction": float(group.prediction.mean()), "mean_target": float(group.target.mean()),
                     "target_minus_prediction_bias": bias, "bias_square": bias_square,
                     "centered_error_mean_square": centered, "total_mse": total, "identity_error": identity})
    cycles = pd.DataFrame(rows)
    summary = {"all_rows": len(frame), "eligible_rows": len(known), "unknown_rows": len(frame)-len(known),
               "cycles": len(cycles), "cycle_equal_total_mse": None, "cycle_equal_bias_square": None,
               "cycle_equal_centered_error_mean_square": None, "bias_share_of_mse": None, "max_identity_error": None}
    if len(cycles):
        total = float(cycles.total_mse.mean())
        bias = float(cycles.bias_square.mean())
        centered = float(cycles.centered_error_mean_square.mean())
        require(abs(total-bias-centered) <= 1e-12, "周期等权总体恒等不成立。")
        summary.update(cycle_equal_total_mse=total, cycle_equal_bias_square=bias,
                       cycle_equal_centered_error_mean_square=centered,
                       bias_share_of_mse=bias/total if total > 0 else None,
                       max_identity_error=float(cycles.identity_error.max()))
    return cycles, summary


def freeze():
    tests = json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 2, "两项误差分解必要测试未通过。")
    paths = [Path(__file__), ROOT/"tests/test_point_exit_error_components_v1.py",
             ROOT/"research/point_account_cashflow_state_v1.py", OUT/"tests_receipt.json",
             PARENT/"protocol.json", PARENT/"summary.json", PARENT/"results/自然参考原预测与标签.parquet"]
    protocol = {"study": STUDY, "frozen_at": now(), "linked_question": "E06附录：跨周期共同误差与周期内误差的贡献",
        "known_before_freeze": "E06的正负分组描述已知；本项偏差能量比例尚未计算。属于结果之后确定的诊断，不称原E06事前假设。",
        "fixed_views": ["CANONICAL_ALL", "CANONICAL_ACTIVE_MATCHED_STRESS_REFERENCE"],
        "fixed_periods": ["2015_2019", "2020_2026"],
        "definition": "e=原标签-原固定版本预测；每周期MSE=(周期e均值)^2+mean((e-周期e均值)^2)，周期等权汇总。",
        "missing": "原未知行保留计数，不填零；两个视图均不混费用、不拆年份择优。",
        "limit": "共同误差来自未来才能知道的同周期平均标签，既不证明预测误差方差，也不证明可以事前预测截距。原第115轮入场状态预测周期截距失败不重开。",
        "failure_exit": "只作精确恒等归因，没有预测通过门、模型选择、截距修正、反号、收益账户或可实现改善上界。",
        "new_model_fits": 0, "new_strategy_accounts": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "sources": [{"path": str(p.relative_to(ROOT)),"sha256":digest(p)} for p in paths]}
    write_json(OUT/"protocol.json", protocol, exclusive=True)
    write_json(OUT/"freeze.json", {"at": now(),"protocol_sha256":digest(OUT/"protocol.json")},exclusive=True)
    print("误差分解附录已冻结：两个固定视图、两个时期，没有模型或收益账户。",flush=True)


def check():
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    frozen=json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT/"protocol.json")==frozen["protocol_sha256"],"误差分解协议已变更。")
    for item in protocol["sources"]:
        require(digest(ROOT/item["path"])==item["sha256"],"误差分解来源已变更："+item["path"])
    return protocol


def run():
    protocol=check()
    write_json(OUT/"RUN_STARTED.json",{"at":now()},exclusive=True)
    frame=pd.read_parquet(PARENT/"results/自然参考原预测与标签.parquet")
    rows, diagnostics=[],[]
    for period in protocol["fixed_periods"]:
        subset=frame.loc[frame.period.eq(period)]
        views={"CANONICAL_ALL":subset,
               "CANONICAL_ACTIVE_MATCHED_STRESS_REFERENCE":subset.loc[subset.stress_reference_relation.eq("ACTIVE_MATCHED_REFERENCE_ENTRY")]}
        for view, part in views.items():
            cycles, summary=decompose(part)
            cycles["period"],cycles["view"]=period,view
            rows.append(cycles)
            diagnostics.append({"period":period,"view":view,**summary})
    results=pd.concat(rows,ignore_index=True)
    results.to_parquet(OUT/"逐周期误差分解.parquet",index=False)
    results.to_csv(OUT/"逐周期误差分解.csv",index=False,encoding="utf-8-sig")
    write_json(OUT/"summary.json",{"study":STUDY,"at":now(),"status":"COMPLETED_FIXED_ERROR_IDENTITY_DIAGNOSTIC",
        "views":diagnostics,"new_model_fits":0,"new_strategy_accounts":0,"account_return_sharpe":"NOT_COMPUTED",
        "independent_validation":"NOT_ESTABLISHED","goal_achieved":False},exclusive=True)
    check()
    files=[OUT/"逐周期误差分解.parquet",OUT/"逐周期误差分解.csv",OUT/"summary.json"]
    write_json(OUT/"verification_receipt.json",{"at":now(),"status":"PASS_CYCLE_EQUAL_MSE_IDENTITY",
        "sources_unchanged":len(protocol["sources"]),"artifacts":[{"path":str(p.relative_to(ROOT)),"sha256":digest(p)} for p in files]},exclusive=True)
    print("四项既定误差分解完成，原模型与原标签均保留。",flush=True)


def main():
    parser=argparse.ArgumentParser(description="原继续收益预测误差分解")
    parser.add_argument("action",choices=["freeze","run","check"])
    args=parser.parse_args()
    if args.action=="freeze":
        freeze()
    elif args.action=="run":
        run()
    else:
        check()
        print("误差分解冻结来源未变。",flush=True)


if __name__=="__main__":
    main()
