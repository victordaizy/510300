"""下载中证指数官方发布的当前沪深300成分与权重快照。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "constituents" / "000300_current_weights.parquet"
STATUS_FILE = PROJECT_ROOT / "reports" / "data_quality" / "000300_point_in_time_weights_status.json"


def main() -> int:
    data = ak.index_stock_cons_weight_csindex(symbol="000300").rename(
        columns={
            "日期": "effective_date", "指数代码": "index_code", "指数名称": "index_name",
            "成分券代码": "stock_code", "成分券名称": "stock_name", "交易所": "exchange",
            "权重": "weight_pct",
        }
    )
    required = ["effective_date", "index_code", "stock_code", "stock_name", "exchange", "weight_pct"]
    data = data[required].copy()
    data["effective_date"] = pd.to_datetime(data["effective_date"], errors="coerce")
    data["weight_pct"] = pd.to_numeric(data["weight_pct"], errors="coerce")
    if len(data) != 300 or data[required].isna().any().any() or data["stock_code"].duplicated().any():
        raise ValueError("当前成分权重快照结构异常")
    if not 99.0 <= data["weight_pct"].sum() <= 101.0:
        raise ValueError(f"权重合计异常：{data['weight_pct'].sum()}")
    data["symbol"] = data.apply(
        lambda row: f"{row['stock_code']}.SH" if "上海" in row["exchange"] else f"{row['stock_code']}.SZ", axis=1
    )
    data["source"] = "akshare.index_stock_cons_weight_csindex"
    data["retrieved_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.sort_values("weight_pct", ascending=False).to_parquet(OUTPUT_FILE, index=False)
    status = {
        "status": "BLOCKED_FOR_POINT_IN_TIME_BACKTEST",
        "current_snapshot_available": True,
        "current_snapshot_date": str(data["effective_date"].max().date()),
        "current_constituent_count": int(len(data)),
        "current_weight_sum_pct": float(data["weight_pct"].sum()),
        "historical_monthly_weights_available": False,
        "reason": "免费中证接口只返回当前快照，不能把当前权重倒填至五年历史。",
        "required_source": "Tushare index_weight 或其他可验证的历史点时权重源",
    }
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
