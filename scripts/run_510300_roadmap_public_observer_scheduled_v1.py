"""定时任务只在已登记交易日观察公开源；不运行研究模型或订单。"""
from pathlib import Path
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from research.roadmap_execution_v1 import digest, write_json
from research.roadmap_public_vintages_v1 import collect


def main() -> None:
    output = ROOT / "reports/research/510300_roadmap_execution_v1"
    current = datetime.now(ZoneInfo("Asia/Shanghai"))
    day = current.date().isoformat()
    result = {"recorded_at": current.isoformat(), "status": "SKIP_NONTRADING_DAY", "network_requests": 0,
              "new_model_fits": 0, "new_orders": 0}
    try:
        freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
        owned = {"config/510300_roadmap_execution_v1.json", "research/roadmap_public_vintages_v1.py",
                 "research/roadmap_execution_v1.py", "data/reference/sse_trade_calendar_2026.csv"}
        for item in freeze["files"]:
            if item["path"].replace("\\", "/") in owned and digest(ROOT / item["path"]) != item["sha256"]:
                raise ValueError("观察运行依赖身份已变化，停止自动取得数据")
        calendar = pd.read_csv(ROOT / "data/reference/sse_trade_calendar_2026.csv")
        if day > str(calendar.trade_date.max()):
            result["status"] = "STOP_CALENDAR_EXPIRED"
        elif day in set(calendar.trade_date.astype(str)):
            result = collect()
    except Exception as exc:
        result.update(status="OBSERVER_FAILED_NO_STRATEGY_ACTION", error_type=type(exc).__name__, error=str(exc)[:1200])
    # 没有变化时不发送通知；回执供后续有意义的研究进展或故障定位使用。
    write_json(output / "scheduled_observer_last_receipt.json", result)


if __name__ == "__main__":
    main()
