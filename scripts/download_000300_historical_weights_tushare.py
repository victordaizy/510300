"""用 Tushare 下载沪深300月度历史成分与权重，支持断点续传。"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "constituents" / "000300_historical_weights.parquet"
STATUS_FILE = PROJECT_ROOT / "reports" / "data_quality" / "000300_point_in_time_weights_status.json"


def get_token() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
    if not token:
        raise RuntimeError(
            "未设置TUSHARE_TOKEN；请写入项目.env或在当前PowerShell会话设置环境变量后重试；"
            "Tushare官方index_weight接口还要求账户至少2000积分"
        )
    return token


def validate(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    required = ["index_code", "con_code", "trade_date", "weight", "source", "retrieved_at"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"历史权重缺少字段：{missing}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("历史权重为空或存在空值")
    if data[["trade_date", "con_code"]].duplicated().any():
        raise ValueError("历史权重存在重复日期和成分代码")
    counts = data.groupby("trade_date")["con_code"].nunique()
    sums = data.groupby("trade_date")["weight"].sum()
    bad_count = counts[(counts < 250) | (counts > 350)]
    bad_sum = sums[(sums < 98.0) | (sums > 102.0)]
    if not bad_count.empty or not bad_sum.empty:
        raise ValueError(f"历史权重快照异常：成分数异常={len(bad_count)}，权重和异常={len(bad_sum)}")
    if data["trade_date"].min() > start + pd.Timedelta(days=31):
        raise ValueError("首个历史权重快照距回测起点超过一个月")
    if end - data["trade_date"].max() > pd.Timedelta(days=45):
        raise ValueError("最后历史权重快照过旧")
    return {
        "snapshot_count": int(data["trade_date"].nunique()),
        "unique_constituent_count": int(data["con_code"].nunique()),
        "min_constituents_per_snapshot": int(counts.min()),
        "max_constituents_per_snapshot": int(counts.max()),
        "min_weight_sum_pct": float(sums.min()),
        "max_weight_sum_pct": float(sums.max()),
    }


def main() -> int:
    try:
        import tushare as ts

        token = get_token()
        config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
        start = pd.Timestamp(config["project"]["feature_warmup_start"])
        end = pd.Timestamp(config["project"]["end_date"])
        api_url = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip()
        request_interval = float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8"))
        if not api_url.startswith(("https://", "http://")):
            raise ValueError("TUSHARE_API_URL必须是HTTP或HTTPS地址")
        if request_interval < 0.5:
            raise ValueError("请求间隔不得低于0.5秒")
        months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        existing = pd.read_parquet(OUTPUT_FILE) if OUTPUT_FILE.exists() else pd.DataFrame()
        completed = set(pd.to_datetime(existing.get("trade_date", pd.Series(dtype="datetime64[ns]"))).dt.to_period("M"))
        pro = ts.pro_api(token)
        pro._DataApi__http_url = api_url
        chunks = [existing] if not existing.empty else []
        for month in months:
            if month in completed:
                continue
            print(f"下载历史权重：{month}")
            part = pro.index_weight(
                index_code="399300.SZ",
                start_date=month.start_time.strftime("%Y%m%d"),
                end_date=month.end_time.strftime("%Y%m%d"),
            )
            if part is None or part.empty:
                if month == end.to_period("M"):
                    print(f"{month} 尚无月度权重快照，使用上一有效月并交由新鲜度门槛检查")
                    continue
                raise RuntimeError(f"{month} 未返回历史权重")
            part["trade_date"] = pd.to_datetime(part["trade_date"], errors="coerce")
            part["weight"] = pd.to_numeric(part["weight"], errors="coerce")
            part["source"] = "tushare.index_weight"
            part["retrieved_at"] = datetime.now(ZoneInfo(config["project"]["timezone"]))
            chunks.append(part)
            checkpoint = pd.concat(chunks, ignore_index=True).drop_duplicates(["trade_date", "con_code"], keep="last")
            checkpoint.sort_values(["trade_date", "con_code"]).to_parquet(OUTPUT_FILE, index=False)
            time.sleep(request_interval)
        data = pd.concat(chunks, ignore_index=True).drop_duplicates(["trade_date", "con_code"], keep="last")
        data = data.sort_values(["trade_date", "con_code"]).reset_index(drop=True)
        evidence = validate(data, start, end)
        data.to_parquet(OUTPUT_FILE, index=False)
        status = {
            "status": "PASS",
            "checked_at": datetime.now(ZoneInfo(config["project"]["timezone"])).isoformat(),
            "file": OUTPUT_FILE.relative_to(PROJECT_ROOT).as_posix(),
            "historical_monthly_weights_available": True,
            "api_host": api_url.split("//", 1)[-1].split("/", 1)[0],
            "request_interval_seconds": request_interval,
            "actual_first_date": str(data["trade_date"].min().date()),
            "actual_last_date": str(data["trade_date"].max().date()),
            **evidence,
        }
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"历史权重下载失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
