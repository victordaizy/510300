"""只确认下一项所需的保存信号和既有趋势，不生成新周期或账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.trend_reference_router_inputs_v1 import checked_trend, MODELS
from research.intraday_overnight_increment_v1 import now, require, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_episode_trend_reference_preflight_20260909"


def main():
    previous = ROOT / "config/510300_trend_reference_router_v1.json"
    cfg = json.loads(previous.read_text(encoding="utf-8"))
    bound = {str(Path(item["path"])): item["sha256"] for item in cfg["frozen_files"]}
    sources = [previous, ROOT / cfg["features"], ROOT / cfg["dividends"],
        ROOT / "docs/510300_EPISODE_TREND_REFERENCE_NEXT_20260909.md",
        ROOT / "reports/research/510300_trend_reference_router_v1/saved_verification_receipt.json",
        ROOT / "config/510300_vintage_reference_risk_v1.json", ROOT / "config/510300_model_support_reference_router_v1.json",
        ROOT / "research/trend_reference_router_inputs_v1.py"]
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == bound[str(Path(cfg[key]))], "下一项行情或分红不再等于已核对来源")
    data = pd.read_parquet(ROOT / cfg["features"])
    checked_trend(data)
    folders = [ROOT / "reports/research/510300_vintage_reference_risk_v1", ROOT / "reports/research/510300_model_support_reference_router_v1"]
    checks = []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in cfg["costs"]:
            for model, folder in zip(MODELS, folders):
                path = folder / period / cost / f"{model}_decisions.parquet"
                require(digest(path) == bound[str(path.relative_to(ROOT))], "下一项父目标与140已冻结来源不同")
                parent = pd.read_parquet(path)
                require(np.array_equal(parent.origin_index, indices), "下一项父目标索引不同")
                require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])) and
                    pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "下一项父目标时钟不同")
                require((parent.reference_weight.isna() | parent.reference_weight.between(0, 1)).all(), "下一项父目标超出无融资范围")
                sources.append(path)
                checks.append({"period": period, "cost": cost, "model": model, "origins": len(parent),
                    "unknown_targets": int(parent.reference_weight.isna().sum()), "unknown_trend_origins": int(frame.sma120.iloc[indices].isna().sum())})
    receipt = {"checked_at": now(), "status": "SAVED_PARENTS_AND_TRAILING_TREND_READY_NO_NEW_EPISODES",
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sources],
        "checks": checks, "new_routes_targets_or_accounts_generated": 0, "candidate_registered": False, "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", receipt, exclusive=True)
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 140, "信号周期固定来源的前序轮次不同")
    index["next_work"].update(status="EPISODE_TREND_REFERENCE_INPUTS_READY", input_preflight=str((OUT / "result.json").relative_to(ROOT)), registered=False)
    index["updated_at"] = now()
    write_json(path, index)
    print(json.dumps({"输入": checks, "新周期目标账户": 0, "下一项已登记": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
