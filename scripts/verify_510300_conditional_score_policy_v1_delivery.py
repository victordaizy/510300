"""交付前复核日期、决策冻结、账本与统计汇总；不重训或重新搜索。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research.conditional_score_policy_v1 import metrics
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.run_510300_conditional_score_policy_v1 import verify_freeze


def clarify_decision_clock(output: Path, config: dict) -> dict:
    """展示表把当日已有成交和本日决策的次日成交明确分列。"""
    path = output / "每日决策表.csv"
    original = output / "每日决策表_原始导出.csv"
    require(not original.exists(), "每日决策展示表已处理，禁止重复覆盖原始导出")
    original.write_bytes(path.read_bytes())
    decisions = pd.read_csv(original, parse_dates=["origin", "execution_date", "date", "pending_original_origin"])
    executions = pd.concat([pd.read_parquet(output / "evaluation" / f"FULL_{cost}_ledger.parquet") for cost in ("BASE", "STRESS")], ignore_index=True)
    future = executions[["date", "cost", "model", "filled_quantity", "request_origin", "execution_status"]].rename(columns={
        "date": "execution_date", "filled_quantity": "下一开盘实际成交份额",
        "request_origin": "下一开盘请求原始决策日", "execution_status": "下一开盘执行结果"})
    table = decisions.merge(future, on=["execution_date", "cost", "model"], how="left", validate="one_to_one")
    initial = table["date"].isna()
    table["初始现金锚"] = initial
    table.loc[initial, ["cash", "equity"]] = config["initial_capital"]
    table.loc[initial, ["shares", "exposure"]] = 0
    table.loc[table.execution_date.isna(), "下一开盘执行结果"] = "固定样本截止_没有执行至样本外"
    table = table.rename(columns={"origin": "收盘决策日", "execution_date": "计划执行日",
        "filled_quantity": "当日开盘已成交份额_来自此前决策", "execution_status": "当日开盘执行结果",
        "shares": "收盘实际份额", "cash": "收盘现金", "equity": "收盘权益", "exposure": "收盘实际暴露",
        "score": "总分", "target_weight": "原始目标持仓", "adjusted_weight": "收盘执行规则调整目标",
        "requested_quantity": "本日冻结请求份额", "reason": "决策或未交易原因",
        "pending_quantity_after_close": "收盘后待执行份额", "pending_original_origin": "待执行请求原始决策日"})
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return {"display_rows": len(table), "original_export_sha256": digest(original), "clarified_display_sha256": digest(path),
            "change": "只补足初始现金账户的展示、区分当日成交与下一开盘成交；未改系数、目标、订单或经济账本"}


def main() -> int:
    output = ROOT / "reports/research/510300_conditional_score_policy_v1"
    config = json.loads((ROOT / "config/510300_conditional_score_policy_v1.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "execution_receipt.json").read_text(encoding="utf-8"))
    require(receipt["status"] == "COMPLETED", "缺少完成回执")
    verify_freeze(output)
    events = [json.loads(line) for line in (output / "stage_events.jsonl").read_text(encoding="utf-8").splitlines()]
    locked = next(i for i, row in enumerate(events) if row["event"] == "主评价前模型冻结完成")
    main_read = [i for i, row in enumerate(events) if row["event"] == "行情日期过滤读取" and row["main_evaluation"]]
    require(len(main_read) == 1 and main_read[0] > locked, "模型冻结与主评价读取顺序不符")
    fits = [row for row in events if row["event"] == "内层拟合完成"]
    require(len(fits) == 54, "内层拟合数不符")
    require(all(row["candidates"] <= 5832 and row["generations"] <= 80 for row in fits), "优化预算超限")
    calendar = pd.read_parquet(ROOT / config["inputs"]["calendar"], columns=["trade_date", "is_open"])
    expected = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"]))
    expected = expected[(expected >= pd.Timestamp(config["evaluation_start"])) & (expected <= pd.Timestamp(config["data_cutoff"]))].sort_values()
    account_checks = {}
    for model in ("BUY_HOLD", "SIMPLE", "FULL", "NO_REPAIR"):
        for cost in ("BASE", "STRESS"):
            key = f"{model}_{cost}"
            ledger = pd.read_parquet(output / "evaluation" / f"{key}_ledger.parquet")
            require(pd.DatetimeIndex(ledger.date).equals(expected), f"{key}日历行缺失或多余")
            require(np.all(ledger.cash >= -1e-7) and np.all(ledger.shares >= 0), f"{key}有杠杆或卖空")
            require(np.all(ledger.shares % 100 == 0) and np.all(ledger.filled_quantity % 100 == 0), "非整手份额")
            prior_equity = ledger.equity.shift(1).fillna(config["initial_capital"])
            require(np.allclose(ledger.net_return, ledger.equity / prior_equity - 1, atol=1e-14, rtol=0), "每日收益与权益不符")
            ref = metrics(ledger, config)
            for name, value in ref.items():
                saved = result["economics"][key][name]
                if value is None:
                    require(saved is None, "不可用夏普被替换")
                elif isinstance(value, (float, int)):
                    require(abs(float(value) - float(saved)) < 1e-10, f"{key}/{name}汇总不符")
            account_checks[key] = {"calendar_rows": len(ledger), "equity_sha256": digest(output / "evaluation" / f"{key}_ledger.parquet"),
                                   "maximum_accounting_error": ref["maximum_accounting_error"],
                                   "missing_dates": 0, "metrics_recomputed_from_saved_ledger": True}
        if model != "BUY_HOLD":
            base = pd.read_csv(output / "evaluation" / f"{model}_BASE_decisions.csv")
            stress = pd.read_csv(output / "evaluation" / f"{model}_STRESS_decisions.csv")
            require(np.allclose(base[["score", "target_weight"]].to_numpy(float), stress[["score", "target_weight"]].to_numpy(float), atol=1e-14, rtol=0, equal_nan=True), "情景间分数或目标不一致")
    require(receipt["result"]["sha256"] == digest(output / "result.json"), "结果回执哈希不一致")
    presentation = clarify_decision_clock(output, config)
    write_json(output / "delivery_verification.json", {"created_at": now(), "status": "PASS_DELIVERY_CHECKS",
               "historical_runs": 1, "inner_fits": len(fits), "main_reads_after_model_freeze": len(main_read),
               "saved_ledgers": account_checks, "model_scores_identical_across_cost_scenarios": True,
               "scope": "仅本轮输出的日期、会计、统计汇总及冻结顺序复核；未增加拟合或选择",
               "verifier_sha256": digest(Path(__file__)), "decision_display": presentation, "position_impact": 0}, exclusive=True)
    print("本轮8条账本、54次内层开发、主评价前模型冻结顺序及交付指标复核通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
