"""只检查历史原预测误差的成熟样本支持，不读取误差大小或创建偏差模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.learned_cycle_exit_v1 import training_rows
from research.point_account_cashflow_state_v1 import digest, now, require, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_prequential_bias_admission_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
E06 = ROOT / "reports/research/510300_point_exit_sign_calibration_v1"
STUDY = "510300_POINT_PREQUENTIAL_BIAS_ADMISSION_V1"


def freeze():
    cfg = json.loads((CURRENT/"inputs/config/within_cycle_exit.json").read_text(encoding="utf-8"))
    require([cfg[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows"]] == [20, 10, 100], "原样本门改变。")
    paths = [Path(__file__), ROOT/"research/learned_cycle_exit_v1.py", ROOT/"research/point_account_cashflow_state_v1.py",
             CURRENT/"inputs/within_models.json", CURRENT/"inputs/candidate_features.parquet",
             CURRENT/"inputs/config/within_cycle_exit.json", E06/"results/自然参考原预测与标签.parquet", E06/"protocol.json"]
    protocol = {"study": STUDY, "frozen_at": now(), "linked_question": "E07_仅成熟历史原预测误差能否支持偏差估计",
        "hypothesis": "预测固定于历史参考入场首次收盘，误差只在自然周期结束之后才可读；这样的历史预测周期能否在两时期达到原数量支持。",
        "clock": "复用原142个月首收盘及原training_rows；mature_date映射到真实价格索引，exit_index<=fit_index；不使用当前未结束周期。",
        "eligibility": "原预测非缺失的全部自然状态；不补未知，最近20个已自然成熟周期，至少10周期/100原点。原周期总权重一。",
        "failure_exit": "较早或近期全期间没有支持则拒绝作为跨期改善唯一改动；不降低数量、换窗口、用拟合残差补过去预测。",
        "history_role": "POINT_IN_TIME_RECONSTRUCTION_ON_REUSED_DEVELOPMENT_DATA_NOT_HISTORICAL_DEPLOYMENT_RECEIPTS",
        "error_values_read": False, "new_model_fits": 0, "new_strategy_accounts": 0,
        "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]}
    write_json(OUT/"protocol.json", protocol, exclusive=True)
    write_json(OUT/"freeze.json", {"at": now(), "protocol_sha256": digest(OUT/"protocol.json")}, exclusive=True)
    print("历史原预测误差的数量准入已冻结，尚未计算逐月支持。", flush=True)


def check():
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "样本准入协议改变。")
    for source in protocol["sources"]:
        require(digest(ROOT/source["path"]) == source["sha256"], "样本准入来源改变："+source["path"])
    return protocol


def run():
    check()
    write_json(OUT/"RUN_STARTED.json", {"at": now()}, exclusive=True)
    frame = pd.read_parquet(E06/"results/自然参考原预测与标签.parquet",
                            columns=["cycle_id", "origin_index", "entry_index", "mature_date", "prediction"])
    available = frame.loc[np.isfinite(frame.prediction)].copy()
    dates = pd.DatetimeIndex(pd.read_parquet(CURRENT/"inputs/candidate_features.parquet", columns=["date"]).date)
    require(dates.is_unique and dates.is_monotonic_increasing, "真实价格日期不唯一递增。")
    available["exit_index"] = dates.get_indexer(pd.to_datetime(available.mature_date))
    require(available.exit_index.ge(0).all() and available.origin_index.lt(available.exit_index).all(), "预测状态晚于成熟标签或成熟日期缺失。")
    records = json.loads((CURRENT/"inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    cfg = json.loads((CURRENT/"inputs/config/within_cycle_exit.json").read_text(encoding="utf-8"))
    rows, memberships = [], []
    for record in records:
        chosen, ids = training_rows(available, record["fit_index"], cfg)
        require(not len(chosen) or chosen.exit_index.le(record["fit_index"]).all(), "读取尚未自然成熟的误差周期。")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(chosen) >= cfg["minimum_rows"]
        day = pd.Timestamp(record["fit_origin"])
        period = "2015_2019" if pd.Timestamp("2015-01-05") <= day <= pd.Timestamp("2019-12-31") else "2020_2026" if pd.Timestamp("2020-01-02") <= day <= pd.Timestamp("2026-09-30") else "OUTSIDE_EVALUATION"
        rows.append({"period": period, "fit_origin": day, "fit_index": int(record["fit_index"]),
                     "mature_prediction_cycles": len(ids), "mature_prediction_rows": len(chosen),
                     "latest_mature_index": int(chosen.exit_index.max()) if len(chosen) else None,
                     "eligible": eligible, "status": "SUPPORTED_COUNTS_ONLY" if eligible else "NO_VIEW_INSUFFICIENT_MATURE_PREDICTION_CYCLES_OR_ROWS"})
        memberships.extend({"fit_origin": day, "cycle_id": int(g.cycle_id.iloc[0]), "exit_index": int(g.exit_index.iloc[0]),
                            "rows": len(g), "cycle_total_weight": float(g.sample_weight.sum())} for _, g in chosen.groupby("cycle_id"))
    monthly = pd.DataFrame(rows)
    support = []
    for period in ["2015_2019", "2020_2026"]:
        part = monthly.loc[monthly.period.eq(period)]
        eligible = part.loc[part.eligible]
        support.append({"period": period, "months": len(part), "supported_months": len(eligible),
                        "maximum_mature_cycles": int(part.mature_prediction_cycles.max()),
                        "maximum_mature_rows": int(part.mature_prediction_rows.max()),
                        "first_supported_origin": eligible.fit_origin.min() if len(eligible) else None})
    passes = all(x["supported_months"] > 0 for x in support)
    monthly.to_parquet(OUT/"逐月成熟预测支持.parquet", index=False)
    monthly.to_csv(OUT/"逐月成熟预测支持.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT/"逐月周期成员.parquet", index=False)
    summary = {"study": STUDY, "completed_at": now(), "status": "SUPPORTED_COUNTS_ONLY_NOT_FITTED" if passes else "REJECTED_FIXED_PREQUENTIAL_BIAS_ROUTE_INSUFFICIENT_CROSS_PERIOD_SUPPORT",
               "available_reference_prediction_rows": len(available), "available_reference_prediction_cycles": int(available.cycle_id.nunique()),
               "monthly_records": len(monthly), "period_support": support, "new_model_fits": 0, "new_strategy_accounts": 0,
               "error_values_read": False, "account_return_sharpe": "NOT_COMPUTED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write_json(OUT/"summary.json", summary, exclusive=True)
    check()
    paths = [OUT/"summary.json", OUT/"逐月成熟预测支持.parquet", OUT/"逐月成熟预测支持.csv", OUT/"逐月周期成员.parquet"]
    write_json(OUT/"verification_receipt.json", {"checked_at": now(), "status": "PASS_ORIGINAL_MATURITY_CLOCK_AND_CYCLE_WEIGHTS",
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]}, exclusive=True)
    print(json.dumps({"status": summary["status"], "period_support": support}, ensure_ascii=False, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser(description="只核对成熟历史预测误差的样本支持")
    parser.add_argument("action", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run()
    else:
        check()
        print("历史预测误差准入来源未变。", flush=True)


if __name__ == "__main__":
    main()
