"""沿用原固定执行函数，为贷款预期字段计算四个连续账户。"""
from __future__ import annotations

import argparse
from types import FunctionType, SimpleNamespace

import pandas as pd

import multidim_loan_surprise_score_v1 as current
import multidim_money_surprise_account_v1 as original


ROOT, OUT = current.ROOT, current.OUT
OLD_OUT = ROOT / "reports/research/510300_multidim_money_surprise_score_v1"
OVERRIDE = ROOT / "config/510300_financing_source_correction_20240808_v1.json"


def prepare():
    if (OUT / "account_protocol.json").exists():
        raise RuntimeError("贷款偏差账户已固定，不覆盖。")
    result = current.read(OUT / "result.json")
    if not result["full_account_admitted"]:
        raise ValueError("未满足原先固定的账户准入条件。")
    daily = ROOT / current.read(OVERRIDE)["corrected_daily_path"]
    source = pd.read_csv(OUT / "全部高分机会.csv")
    # 原执行函数以state标识对照，不改变九变量对照信号或交易日期。
    source["model"] = source.model.replace({"control": "state"})
    source.to_csv(OUT / "全部固定高分机会.csv", index=False, encoding="utf-8-sig")
    protocol = current.read(OLD_OUT / "account_protocol.json")
    protocol.update({
        "registered_at": current.common.now(), "study_id": result["study_id"],
        "reason": "新增字段已改变6次主要期预测，联合高分均值为正，按已固定条件进行账户换算；主要期五个入场日未改变，不能据此声称新增机会。",
        "prior_seen": "评分结果全部已见；新增2023年2月13日机会亏损，主要期预测误差变差。账户阶段只换算既定信号，不改模型、分数阈值、持有期或风险预算。",
        "cases": [{"id": f"{label}_{capital}", "model": engine_model, "capital": capital}
                  for label, engine_model in [("control", "state"), ("joint", "joint")] for capital in [200000, 20000]],
        "model_label_mapping": {"state": "同事件九变量M2与指数状态对照", "joint": "十变量贷款偏差联合评分"},
        "benchmarks": "同日历旧现金及期初50%预算买入持有仅复用其描述结果，不重算两个重复账户，也不宣称为风险合规候选。",
        "daily_source": str(daily.relative_to(ROOT)).replace("\\", "/"),
        "source_receipts": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": current.sha(p)}
                            for p in [daily, original.RISK_PATH, original.DIVIDENDS, original.ENGINE_PATH,
                                      OUT / "逐事件贷款预期联合评分.csv", OUT / "全部固定高分机会.csv", ROOT / "research/multidim_money_surprise_account_v1.py"]],
        "goal_achieved": False
    })
    current.save("account_protocol.json", protocol)
    print("已固定4个连续账户，沿用原成交、尾部预算与五日退出。")


def run():
    protocol = current.read(OUT / "account_protocol.json")
    study = SimpleNamespace(ROOT=ROOT, OUT=OUT, DAILY=ROOT / protocol["daily_source"],
                            read=current.read, save=current.save, sha=current.sha, common=current.common)
    # 创建独立函数环境，保持旧模块、旧冻结账户和旧输出目录不变。
    function = FunctionType(original.run.__code__, {**original.run.__globals__, "OUT": OUT, "study": study},
                            original.run.__name__, original.run.__defaults__, original.run.__closure__)
    function()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="贷款预期联合评分的固定连续账户")
    parser.add_argument("stage", choices=["prepare", "run"])
    stage = parser.parse_args().stage
    {"prepare": prepare, "run": run}[stage]()
