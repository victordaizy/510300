"""检查沪深300历史内部结构研究的输入就绪状态。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
CONTRACT_FILE = ROOT / "config" / "index_structure_data_contract.yaml"
WEIGHTS_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
DAILY_FILE = ROOT / "data" / "raw" / "constituents" / "000300_constituent_daily.parquet"
PROCESSOR_FILE = ROOT / "research" / "build_weighted_breadth_dataset.py"
REPORT_FILE = ROOT / "reports" / "data_quality" / "000300_index_structure_readiness.json"


def main() -> int:
    load_dotenv(ROOT / ".env")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    token_present = bool(os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN"))
    weights_present = WEIGHTS_FILE.exists()
    daily_present = DAILY_FILE.exists()
    if weights_present and daily_present:
        status = "READY_TO_BUILD_NEUTRAL_FEATURES"
        next_action = "运行python -m research.build_weighted_breadth_dataset"
    elif not token_present:
        status = "BLOCKED_MISSING_TUSHARE_CREDENTIAL"
        next_action = "在项目.env设置TUSHARE_TOKEN，并确认账户至少2000积分"
    elif not weights_present:
        status = "TOKEN_PRESENT_WEIGHT_ACCESS_NOT_VERIFIED"
        next_action = "运行历史权重下载器并验证index_weight权限"
    else:
        status = "BLOCKED_MISSING_CONSTITUENT_DAILY_HISTORY"
        next_action = "按数据合同生成包含总收益价格、停牌标志及点时市值/行业的成分股日线长表"
    report = {
        "status": status,
        "checked_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "tushare_token_present": token_present,
        "tushare_index_weight_minimum_points": 2000,
        "tushare_access_verified": False,
        "historical_weights_present": weights_present,
        "constituent_daily_history_present": daily_present,
        "weighted_breadth_processor_present": PROCESSOR_FILE.exists(),
        "data_contract_present": CONTRACT_FILE.exists(),
        "next_action": next_action,
        "security_note": "报告只记录凭据是否存在，不读取或输出Token内容。",
    }
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
