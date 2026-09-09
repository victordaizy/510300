"""只确认既有风险上限来源和最终收盘目标，不计算新上限策略。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_final_target_cap_preflight_20260909"


def main():
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 137, "风险上限前序未完成137")
    old = json.loads((ROOT / "config/510300_vintage_reference_risk_v1.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "config/510300_joint_downside_reference_pair_v1.json").read_text(encoding="utf-8"))
    known = {str(Path(item["path"])): item["sha256"] for item in old["frozen_files"]}
    data = pd.read_parquet(ROOT / cfg["features"])
    sources = [ROOT / "config/510300_vintage_reference_risk_v1.json", ROOT / "config/510300_joint_downside_reference_pair_v1.json",
        ROOT / "reports/research/510300_joint_downside_reference_pair_v1/saved_verification_receipt.json",
        ROOT / "docs/510300_FINAL_TARGET_VOLATILITY_CAP_NEXT_20260909.md", ROOT / cfg["features"], ROOT / cfg["dividends"]]
    checks = []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        risk_path = ROOT / "reports/research/510300_conditional_variance_budget_v1" / f"{period}_factors.parquet"
        require(digest(risk_path) == known[str(risk_path.relative_to(ROOT))], "普通波动来源与131冻结记录不同")
        risk = pd.read_parquet(risk_path)
        require(pd.DatetimeIndex(risk.date).equals(pd.DatetimeIndex(frame.date)), "普通波动上限来源日历不同")
        cap = risk["ROLLING_VARIANCE_BUDGET_CONTROL_multiplier"].to_numpy(float)
        require(np.isfinite(cap).all() and ((cap >= 0) & (cap <= 1)).all(), "保存普通波动乘数非法")
        sources.append(risk_path)
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in cfg["costs"]:
            parent_path = ROOT / "reports/research/510300_joint_downside_reference_pair_v1" / period / cost / "JOINT_DOWNSIDE_REFERENCE_PAIR_decisions.parquet"
            parent = pd.read_parquet(parent_path)
            require(np.array_equal(parent.origin_index, indices), "最终合成目标索引不同")
            require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), "最终合成目标收盘不同")
            require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "最终合成目标不是下一开盘")
            targets = parent.reference_weight.to_numpy(float)
            require((np.isnan(targets) | (np.isfinite(targets) & (targets >= 0) & (targets <= 1))).all(), "原合成目标超出完整无融资范围")
            sources.append(parent_path)
            checks.append({"period": period, "cost": cost, "calendar_rows": len(frame), "existing_target_origins": len(parent),
                "unknown_parent_targets": int(np.isnan(targets).sum()), "cap_calendar_matches": True})
    receipt = {"checked_at": now(), "status": "EXISTING_VOLATILITY_CAP_AND_FINAL_TARGET_CLOCKS_READY",
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sources], "checks": checks,
        "new_targets_or_accounts_generated": 0, "candidate_registered": False, "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", receipt, exclusive=True)
    index["next_work"].update(status="FINAL_TARGET_VOLATILITY_CAP_INPUTS_READY", input_preflight=str((OUT / "result.json").relative_to(ROOT)),
        candidate_round=138, registered=False, new_targets_generated=False, planned_settings=1, planned_new_accounts=4)
    original = "较早2015与136账户相同，新增改善没有解决该年仍沿用初始各半预算的问题；但不能据此事后跳过2015。"
    corrected = "较早2015与136账户相同，当年仍使用预设各半预算，新增改善发生于其他年份；不能据此把预算初始化认定为失败原因，也不能事后跳过2015。"
    index["process_state_note"] = index["process_state_note"].replace(original, corrected)
    document = Path(index["deliveries"][-1]["main_document"])
    text = document.read_text(encoding="utf-8")
    require(original in text, "137文档预期解释位置不同")
    document.write_text(text.replace(original, corrected), encoding="utf-8")
    index["updated_at"] = now()
    write_json(index_path, index)
    print(json.dumps({"输入就绪": checks, "新目标或账户": 0, "登记": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
