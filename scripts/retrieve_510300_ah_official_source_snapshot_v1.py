"""保存官网公开A/H图表及文档的新时间戳快照；不会覆盖冻结实验输入。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S_%f_0800")
    out = root / "data/raw/market/510300_ah_official_new_snapshots" / timestamp
    out.mkdir(parents=True, exist_ok=False)
    sources = {
        "directory.json": "/data/eng/index-series/directory.json",
        "chart.json": "/data/eng/indexes/01044.00/chart.json",
        "literatures.json": "/data/eng/index-series/ahpremium/literatures.json",
        "methodology.pdf": "/static/uploads/contents/en/dl_centre/methodologies/IM_ahpremiume.pdf",
        "factsheet.pdf": "/static/uploads/contents/en/dl_centre/factsheets/ahpremiume.pdf",
    }
    receipt = {"retrieved_at": timestamp, "purpose": "NEW_RAW_OBSERVATION_ONLY",
               "frozen_input_overwrite": False, "model_fits": 0, "accounts": 0, "sources": []}
    for name, endpoint in sources.items():
        item = {"url": "https://www.hsi.com.hk" + endpoint}
        try:
            response = requests.get(item["url"], timeout=(8, 20))
            (out / name).write_bytes(response.content)
            item.update({"http_status": response.status_code, "bytes": len(response.content),
                         "sha256": hashlib.sha256(response.content).hexdigest(),
                         "received_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                         "raw_path": (out / name).relative_to(root).as_posix()})
            response.raise_for_status()
            if name == "chart.json":
                payload = response.json()
                if payload.get("indexCode") != "01044.00":
                    raise ValueError("响应不是A/H溢价指数")
                item["source_last_update"] = payload.get("lastUpdate")
            item["status"] = "RAW_RESPONSE_SAVED_NOT_ADMITTED_TO_FROZEN_STUDY"
        except Exception as exc:
            item.update({"status": "EXTERNAL_FREE_SOURCE_FAILED", "error": str(exc)[:1000]})
        receipt["sources"].append(item)
        (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(item, ensure_ascii=False), flush=True)
    print("新增快照目录：", out)


if __name__ == "__main__":
    main()
