"""确认原模型训练时钟及两策略收盘目标，暂不生成路由或新账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.model_support_reference_router_inputs_v1 import validate_support_records, MODELS
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_model_support_router_preflight_20260909"


def main():
    cfg = json.loads((ROOT / "config/510300_joint_downside_reference_pair_v1.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(ROOT / cfg["features"])
    source_path = ROOT / "reports/research/510300_within_cycle_exit_v1/saved_models.json"
    reused = json.loads((ROOT / "reports/research/510300_entry_vintage_exit_v1/reused_models_receipt.json").read_text(encoding="utf-8"))
    require(digest(source_path) == reused["source_sha256"], "原128已使用模型文件改变")
    records = json.loads(source_path.read_text(encoding="utf-8"))["models"]
    validate_support_records(data, records)
    require(len(records) == 141 and sum(r["status"] == "FIT_COMPLETE" for r in records) == 114, "原模型支持记录数量改变")
    sources = [source_path, ROOT / "config/510300_vintage_reference_risk_v1.json", ROOT / "config/510300_joint_downside_reference_pair_v1.json",
        ROOT / "reports/research/510300_entry_vintage_exit_v1/reused_models_receipt.json", ROOT / "reports/research/510300_final_target_volatility_cap_v1/saved_verification_receipt.json",
        ROOT / "docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_NEXT_20260909.md", ROOT / cfg["features"], ROOT / cfg["dividends"]]
    checks = []
    folders = [ROOT / "reports/research/510300_vintage_reference_risk_v1", ROOT / "reports/research/510300_joint_downside_reference_pair_v1"]
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in cfg["costs"]:
            for model, folder in zip(MODELS, folders):
                path = folder / period / cost / f"{model}_decisions.parquet"
                parent = pd.read_parquet(path)
                require(np.array_equal(parent.origin_index, indices), "训练支持父目标索引不同")
                require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), "训练支持父目标收盘不同")
                require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "训练支持父目标不是下一开盘")
                sources.append(path)
                checks.append({"period": period, "cost": cost, "model": model, "existing_origins": len(parent), "unknown_parent_targets": int(parent.reference_weight.isna().sum())})
    receipt = {"checked_at": now(), "status": "ORIGINAL_MODEL_1505_CLOCK_AND_PARENT_TARGETS_READY", "model_records": len(records),
        "mature_records": 114, "insufficient_records": 27, "first_mature_fit_time": next(r["fit_time"] for r in records if r["status"] == "FIT_COMPLETE"),
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sources], "checks": checks,
        "new_routes_targets_or_accounts_generated": 0, "candidate_registered": False, "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", receipt, exclusive=True)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 138, "模型支持前序轮次不符")
    index["next_work"].update(status="MODEL_SUPPORT_ROUTER_INPUT_CLOCKS_READY", input_preflight=str((OUT / "result.json").relative_to(ROOT)), registered=False)
    index["updated_at"] = now()
    write_json(index_path, index)
    print(json.dumps({"模型": {"总数": len(records), "成熟": 114, "首次": receipt["first_mature_fit_time"]}, "目标来源": checks, "新路由": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
