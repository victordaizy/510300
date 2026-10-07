"""保留原冻结快照，用既有上交所年度日历补足遗漏的覆盖段。"""
from pathlib import Path
import shutil
import pandas as pd
from growth_state_increment_20d_v1 import OUT, ROOT, now, save, sha


def repair() -> None:
    if any((OUT / "results").iterdir()) or (OUT / "calendar_correction_receipt.json").exists():
        raise ValueError("已有结果或修正回执，不能重复补丁")
    import json
    metadata_source = ROOT / "data/reference/sse_trade_calendar_2026.metadata.json"
    source = ROOT / "data/reference/sse_trade_calendar_2026.csv"
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or sha(source) != metadata["sha256"]:
        raise ValueError("既有年度日历身份不一致")
    prior = pd.read_parquet(OUT / "inputs/calendar.parquet")
    prior["trade_date"] = pd.to_datetime(prior.trade_date)
    annual = pd.read_csv(source, parse_dates=["trade_date"])
    original_2026 = set(prior.loc[prior.trade_date.dt.year == 2026, "trade_date"])
    overlapping_2026 = set(annual.loc[annual.trade_date <= prior.trade_date.max(), "trade_date"])
    if original_2026 != overlapping_2026:
        raise ValueError("两份日历在重叠段不一致，不能拼接")
    market = pd.read_parquet(OUT / "inputs/market.parquet", columns=["date"])
    extra = annual[(annual.trade_date > prior.trade_date.max()) & (annual.trade_date <= market.date.max())].copy()
    supplement = pd.DataFrame({"exchange": "SSE", "trade_date": extra.trade_date, "is_open": True,
        "source": extra.source, "retrieved_at": metadata["retrieved_at"], "market_price_read": False, "future_return_read": False})
    joined = pd.concat([prior, supplement], ignore_index=True).sort_values("trade_date")
    dates = joined.loc[joined.trade_date.between(market.date.min(), market.date.max()), "trade_date"]
    if dates.tolist() != pd.to_datetime(market.date).tolist():
        raise ValueError("补齐后仍与实际日期不同")
    joined.to_parquet(OUT / "inputs/calendar_extended.parquet", index=False)
    shutil.copy2(source, OUT / "inputs/sse_trade_calendar_2026.csv")
    shutil.copy2(metadata_source, OUT / "evidence/sse_trade_calendar_2026.metadata.json")
    corrected = OUT / "code/corrected"
    corrected.mkdir(parents=True, exist_ok=True)
    for name in ("growth_state_increment_20d_v1.py", "verify_growth_state_increment_20d_v1.py", "repair_growth_calendar_20d_v1.py"):
        shutil.copy2(ROOT / "research" / name, corrected / name)
    additional = [OUT / "inputs/calendar_extended.parquet", OUT / "inputs/sse_trade_calendar_2026.csv", OUT / "evidence/sse_trade_calendar_2026.metadata.json", *corrected.glob("*.py")]
    save(OUT / "calendar_correction_receipt.json", {"corrected_at": now(),
        "original_attempt": "run_started.json", "failure": "行情与独立交易日历不同；参考日历止于2026-08-14，行情止于2026-09-11",
        "failure_stage": "标签生成前的日期准入", "original_attempt_new_labels": 0, "original_attempt_model_fits": 0,
        "original_protocol_changed": False, "original_frozen_files_changed": False, "added_dates": extra.trade_date.dt.strftime("%Y-%m-%d").tolist(),
        "supplement_source": metadata["official_source"], "independent_live_crosscheck": "https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml",
        "source_role": "按交易所年度休市安排生成的既有2026日历；重叠段相同；不根据510300收益补日期。",
        "additional_frozen_files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in additional]})
    print(f"原冻结快照保留；独立年度日历补齐{len(extra)}个日期，尚无新收益标签或模型。")


if __name__ == "__main__":
    repair()
