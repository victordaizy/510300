"""新增状态进入退出模型前的有限样本准入；不拟合、不计算新交易收益。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.point_account_cashflow_state_v1 import (
    ROOT, OUT as CASHFLOW_OUT, BASE, COSTS, PERIODS, digest, now, require, write_json,
)


OUT = CASHFLOW_OUT / "model_increment_admission"
STUDY = "510300_POINT_CASHFLOW_INCREMENT_ADMISSION_V1"


def source_paths():
    paths = [Path(__file__).relative_to(ROOT), Path("research/point_account_cashflow_state_v1.py"),
             CASHFLOW_OUT.relative_to(ROOT) / "summary.json",
             CASHFLOW_OUT.relative_to(ROOT) / "results/持仓现金流状态与参考身份.parquet"]
    for period in PERIODS:
        for cost in COSTS:
            folder = BASE / f"controls/{period}/{cost}/A_SAVED_WEIGHT"
            paths.extend(folder / f"{name}.parquet" for name in ("daily", "trades"))
    return paths


def freeze():
    require(not OUT.exists(), "样本准入实验已存在，不能覆盖。")
    sources = [{"path": path.as_posix(), "sha256": digest(ROOT / path)} for path in source_paths()]
    protocol = {
        "study": STUDY, "frozen_at": now(), "role": "PRE_FIT_MAXIMUM_CAUSAL_SAMPLE_SUPPORT_ONLY",
        "preknown": "已完成现金流资格研究：1752个持仓原点，状态差异及稀疏阶段5/0已知；本准入不伪称收益读取前的独立预注册。",
        "proposed_single_field": "真实累计现金流收益率减固定版本参考cycle_return；只作为旧预测的有限残差增量候选，不直接替换原八项状态。",
        "eligible_origin": "实际有库存、参考cycle_return可用且原vintage_continuation_prediction为有限值；缺预测保留NO_VIEW，不填0。",
        "planned_target_status": "NOT_COMPUTED。拟议标签须按origin现有库存固定不加减仓，比较下一合法开盘卖出与原保存A自然退出日期卖出，原周期成熟后才可学习；本准入仅查最大样本支持，不声称已接纳该标签。",
        "maturity": "以该账户自然COMPLETE周期exit_date<=当前月首交易日为上界；只算至少一条eligible_origin且origin<exit_date的周期。未完成不训练。",
        "window": "最近20个符合条件、自然成熟的实际周期；每周期总权重一；不得混成本、拼时期或补参考周期。",
        "gate": {"minimum_cycles": 10, "minimum_rows": 100},
        "relation_to_original": "保留现模型最低成熟周期/行数要求，原策略本身不变；实际状态增量必须有同一账户定义下的成熟样本。",
        "failure_exit": "任一时期无可拟合月，则拒绝该固定单字段残差路线作为跨期共同改善的唯一改动；不降低周期数、不放宽缺失、不换窗口营救。",
        "economic_returns_computed": False, "new_model_fits": 0, "new_strategy_accounts": 0,
        "history_role": "DEVELOPMENT_ADMISSION_NOT_INDEPENDENT_VALIDATION", "sources": sources,
    }
    OUT.mkdir()
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"protocol_sha256": digest(OUT / "protocol.json"), "sources": sources}, exclusive=True)
    print("单字段残差路线样本准入已冻结，尚未读取新支持计数。", flush=True)


def check():
    record = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT / "protocol.json") == record["protocol_sha256"], "准入协议已变化。")
    for item in record["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "准入来源已变化：" + item["path"])
    return len(record["sources"])


def run():
    require(not (OUT / "summary.json").exists(), "准入已有裁决，不重跑。")
    check()
    states = pd.read_parquet(CASHFLOW_OUT / "results/持仓现金流状态与参考身份.parquet")
    states = states.loc[states.reference_state_available & states.vintage_continuation_prediction.notna()].copy()
    tables, summaries = [], []
    for period in PERIODS:
        for cost in COSTS:
            folder = ROOT / BASE / f"controls/{period}/{cost}/A_SAVED_WEIGHT"
            daily = pd.read_parquet(folder / "daily.parquet")
            trades = pd.read_parquet(folder / "trades.parquet")
            local = states.loc[states.period.eq(period) & states.cost.eq(cost)]
            require(local.return_difference.notna().all(), "新字段在预定可用原点缺失。")
            completed = trades.loc[trades.status.eq("COMPLETE"), ["cycle_id", "exit_date"]]
            paired = local.merge(completed, left_on="actual_cycle_id", right_on="cycle_id", validate="many_to_one")
            paired = paired.loc[paired.origin.lt(paired.exit_date)]
            cycles = paired.groupby("actual_cycle_id").agg(exit_date=("exit_date", "first"), rows=("origin", "size")).reset_index()
            require(cycles.exit_date.notna().all(), "成熟周期没有自然退出时点。")
            months = daily.groupby(daily.date.dt.to_period("M")).date.min()
            monthly = []
            for fit_origin in months:
                mature = cycles.loc[cycles.exit_date.le(fit_origin)].sort_values(["exit_date", "actual_cycle_id"]).tail(20)
                # exit_date成熟性同时保证其所有持仓原点在fit_origin以前。
                used = paired.loc[paired.actual_cycle_id.isin(mature.actual_cycle_id)]
                require(used.origin.le(fit_origin).all(), "准入读取了未来原点。")
                count_cycles, count_rows = len(mature), len(used)
                passed = count_cycles >= 10 and count_rows >= 100
                monthly.append({"period": period, "cost": cost, "fit_origin": fit_origin,
                                "mature_actual_cycles": count_cycles, "eligible_holding_rows": count_rows,
                                "status": "FIT_SUPPORT_AVAILABLE_TARGET_NOT_ADMITTED" if passed else "NO_VIEW_INSUFFICIENT_ACTUAL_CYCLES_OR_ROWS"})
            frame = pd.DataFrame(monthly)
            tables.append(frame)
            passed = frame.status.eq("FIT_SUPPORT_AVAILABLE_TARGET_NOT_ADMITTED")
            summaries.append({"period": period, "cost": cost, "eligible_origins": len(local),
                              "eventually_complete_eligible_cycles": len(cycles), "monthly_origins": len(frame),
                              "months_with_support": int(passed.sum()),
                              "max_mature_actual_cycles": int(frame.mature_actual_cycles.max()),
                              "max_eligible_holding_rows": int(frame.eligible_holding_rows.max()),
                              "first_supported_fit_origin": frame.loc[passed, "fit_origin"].min() if passed.any() else None})
    table = pd.concat(tables, ignore_index=True)
    table.to_parquet(OUT / "逐月样本支持.parquet", index=False)
    table.to_csv(OUT / "逐月样本支持.csv", index=False, encoding="utf-8-sig")
    rejected = any(item["months_with_support"] == 0 for item in summaries)
    result = {"study": STUDY, "at": now(),
              "status": "REJECTED_FIXED_CASHFLOW_RESIDUAL_ROUTE_INSUFFICIENT_CROSS_PERIOD_SUPPORT" if rejected else "SAMPLE_SUPPORT_AVAILABLE_TARGET_AND_PREDICTION_NOT_TESTED",
              "accounts": summaries, "sources_unchanged": check(), "new_model_fits": 0,
              "new_strategy_accounts": 0, "economic_returns_computed": False,
              "strategy_increment_status": "E02_NOT_REGISTERED_NOT_RUN",
              "goal_achieved": False,
              "interpretation": "日数不是独立周期；只按该账户完整且标签可成熟的周期计算。样本准入通过也不等于标签或预测通过；失败不放宽原成熟门槛。"}
    write_json(OUT / "summary.json", result, exclusive=True)
    write_json(OUT / "verification_receipt.json", {"at": now(), "sources_unchanged": result["sources_unchanged"],
               "protocol_sha256": digest(OUT / "protocol.json"), "summary_sha256": digest(OUT / "summary.json"),
               "table_sha256": digest(OUT / "逐月样本支持.parquet"), "economic_returns_computed": False}, exclusive=True)
    print(result["status"], flush=True)
    for item in summaries:
        print(f"{item['period']} {item['cost']}：最多{item['max_mature_actual_cycles']}个成熟周期，{item['months_with_support']}个月达到拟合样本要求。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="实际现金流残差字段的有限样本准入")
    parser.add_argument("action", choices=("freeze", "run", "check"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run()
    else:
        print(f"准入冻结来源未变：{check()}份。", flush=True)


if __name__ == "__main__":
    main()
