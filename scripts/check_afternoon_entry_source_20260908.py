"""保存已有分钟源覆盖与成交量单位核对，不创建策略或读取新收益。"""
import json
from pathlib import Path
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_afternoon_entry_source_feasibility_v1"
old_path = ROOT / "reports/research/510300_afternoon_learned_exit_v1/existing_minute_inventory.json"
inventory = json.loads(old_path.read_text(encoding="utf-8"))
minute_path = ROOT / inventory["source"]
daily_path = ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet"
require(digest(minute_path) == inventory["source_sha256"], "分钟源已变化，不能继承旧价格与时间记录核对")
minute = pd.read_parquet(minute_path, columns=["trade_time", "vol"])
minute["date"] = pd.to_datetime(minute.trade_time).dt.normalize()
daily = pd.read_parquet(daily_path, columns=["date", "volume"]).set_index("date").volume
paired = pd.concat([minute.groupby("date").vol.sum().rename("minute_sum"), daily.rename("daily_volume")], axis=1, sort=True).dropna()
paired["ratio"] = paired.minute_sum / paired.daily_volume
paired["absolute_difference"] = (paired.minute_sum - paired.daily_volume).abs()
OUT.mkdir(parents=True, exist_ok=True)
require(not (OUT / "source_feasibility.json").exists(), "来源核对已保存，不重复覆盖")
paired.to_csv(OUT / "minute_daily_volume_comparison.csv", encoding="utf-8-sig")
record = {"checked_at": now(), "status": "EXISTING_HISTORY_PROXY_ONLY_NO_LIVE_CLOCK_PROOF", "minute_file": str(minute_path.relative_to(ROOT)),
    "minute_sha256": digest(minute_path), "daily_sha256": digest(daily_path), "parent_inventory_sha256": digest(old_path),
    "minute_days": inventory["days"], "first": inventory["first"], "last": inventory["last"],
    "main_days_without_minute": inventory["main_days_without_minute"], "earlier_minute_days": inventory["earlier_minute_days"],
    "volume_comparison_days": len(paired), "volume_ratio_min": float(paired.ratio.min()), "volume_ratio_median": float(paired.ratio.median()),
    "volume_ratio_max": float(paired.ratio.max()), "max_absolute_volume_difference": float(paired.absolute_difference.max()),
    "volume_unit_interpretation": "与既有日线份额成交量同单位，最大14份舍入差，未发现100倍单位差",
    "source_limitations": inventory["source_caveats"], "official_definition_url": "https://tushare.pro/document/2?doc_id=387",
    "candidate_registered": False, "new_downloads": 0, "new_accounts": 0, "new_model_fits": 0, "strategy_returns_read": False}
write_json(OUT / "source_feasibility.json", record, exclusive=True)
print(json.dumps(record, ensure_ascii=False), flush=True)
