"""通过Tushare VIP接口下载并构建沪深300点时财务事件表。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
MEMBERSHIP_FILE = ROOT / "data" / "raw" / "constituents" / "000300_historical_membership_intervals.parquet"
CHECKPOINT_DIR = ROOT / "data" / "raw" / "fundamentals" / "vip_checkpoints"
OUTPUT_FILE = ROOT / "data" / "raw" / "fundamentals" / "csi300_financials_point_in_time.parquet"
REPORT_FILE = ROOT / "reports" / "data_quality" / "csi300_financials_point_in_time_status.json"

API_FIELDS = {
    "income_vip": [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "revenue", "n_income_attr_p",
    ],
    "balancesheet_vip": [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type",
        "total_hldr_eqy_exc_min_int", "total_share",
    ],
    "fina_indicator_vip": [
        "ts_code", "ann_date", "end_date", "eps", "dt_eps", "bps", "roe",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_token() -> str:
    """只从进程环境或被Git忽略的.env读取凭据。"""

    load_dotenv(ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
    if not token:
        raise RuntimeError("未设置TUSHARE_TOKEN或TS_TOKEN")
    return token


def build_quarter_periods(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """生成闭区间内所有季度末报告期。"""

    if end < start:
        raise ValueError("结束日期不能早于开始日期")
    return [period.strftime("%Y%m%d") for period in pd.date_range(start, end, freq="QE")]


def _download_with_retry(
    call: Callable[[], pd.DataFrame],
    label: str,
    attempts: int = 4,
) -> pd.DataFrame:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            result = call()
            if result is None or result.empty:
                raise ValueError("返回空数据")
            return result
        except Exception as exception:
            error = exception
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"{label}下载失败：{type(error).__name__}: {error}")


def _normalize_table(data: pd.DataFrame, api_name: str) -> pd.DataFrame:
    required = set(API_FIELDS[api_name])
    if missing := required - set(data.columns):
        raise ValueError(f"{api_name}缺少字段：{sorted(missing)}")
    result = data[list(API_FIELDS[api_name])].copy()
    result = result.rename(columns={"ts_code": "con_code", "end_date": "report_period"})
    for column in ("ann_date", "f_ann_date", "report_period"):
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    result["available_at"] = result.get("f_ann_date", result["ann_date"]).fillna(result["ann_date"])
    result = result.dropna(subset=["con_code", "report_period", "ann_date", "available_at"])
    numeric_columns = [
        column for column in result.columns
        if column not in {"con_code", "ann_date", "f_ann_date", "report_period", "available_at", "report_type"}
    ]
    result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if "report_type" in result:
        result["report_type"] = result["report_type"].astype("string")
        result["report_type_priority"] = result["report_type"].map({"1": 0, "4": 1, "5": 2}).fillna(9)
    else:
        result["report_type_priority"] = 0
    result = result.sort_values(
        ["con_code", "report_period", "available_at", "report_type_priority", "ann_date"]
    )
    result = result.drop_duplicates(
        ["con_code", "report_period", "available_at"], keep="first"
    ).drop(columns="report_type_priority")
    return result.reset_index(drop=True)


def build_point_in_time_events(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    indicator: pd.DataFrame,
    retrieved_at: datetime,
) -> pd.DataFrame:
    """在每个真实可得时点合并当时已知的三张财务表。"""

    income = _normalize_table(income, "income_vip").rename(
        columns={
            "revenue": "revenue_cny",
            "n_income_attr_p": "net_profit_parent_cny",
            "report_type": "income_report_type",
            "ann_date": "income_ann_date",
            "f_ann_date": "income_f_ann_date",
        }
    )
    balance = _normalize_table(balance, "balancesheet_vip").rename(
        columns={
            "total_hldr_eqy_exc_min_int": "equity_parent_cny",
            "total_share": "total_shares",
            "report_type": "balance_report_type",
            "ann_date": "balance_ann_date",
            "f_ann_date": "balance_f_ann_date",
        }
    )
    indicator = _normalize_table(indicator, "fina_indicator_vip").rename(
        columns={"ann_date": "indicator_ann_date"}
    )
    keys = ["con_code", "report_period", "available_at"]
    events = income.merge(balance, on=keys, how="outer", validate="one_to_one")
    events = events.merge(indicator, on=keys, how="outer", validate="one_to_one")
    events = events.sort_values(keys).reset_index(drop=True)
    value_columns = [
        "revenue_cny", "net_profit_parent_cny", "equity_parent_cny", "total_shares",
        "eps", "dt_eps", "bps", "roe", "income_report_type", "balance_report_type",
        "income_ann_date", "income_f_ann_date", "balance_ann_date", "balance_f_ann_date",
        "indicator_ann_date",
    ]
    existing_values = [column for column in value_columns if column in events.columns]
    events[existing_values] = events.groupby(
        ["con_code", "report_period"], sort=False
    )[existing_values].ffill()
    announcement_columns = [
        column for column in ("income_ann_date", "balance_ann_date", "indicator_ann_date")
        if column in events
    ]
    events["announcement_date"] = events[announcement_columns].max(axis=1)
    events["source"] = "tushare.income_vip+balancesheet_vip+fina_indicator_vip"
    events["retrieved_at"] = retrieved_at
    events = events.drop_duplicates(keys, keep="last")
    if events[keys].duplicated().any():
        raise ValueError("点时财务事件存在重复键")
    if (events["announcement_date"] > events["available_at"]).any():
        raise ValueError("公告日期晚于信息可得日期，点时语义异常")
    required_values = ["revenue_cny", "net_profit_parent_cny", "equity_parent_cny", "total_shares"]
    if events[required_values].notna().any(axis=1).sum() == 0:
        raise ValueError("点时财务事件没有任何核心财务值")
    return events.reset_index(drop=True)


def main() -> int:
    try:
        import tushare as ts

        token = get_token()
        settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
        timezone = ZoneInfo(settings["project"]["timezone"])
        retrieved_at = datetime.now(timezone)
        api_url = os.getenv("TUSHARE_API_URL", "https://api.tushare.pro").strip()
        request_interval = float(os.getenv("TUSHARE_REQUEST_INTERVAL_SECONDS", "0.8"))
        if request_interval < 0.5:
            raise ValueError("请求间隔不得低于0.5秒")
        periods = build_quarter_periods(pd.Timestamp("2015-07-01"), pd.Timestamp(settings["project"]["end_date"]))
        membership = pd.read_parquet(MEMBERSHIP_FILE)
        relevant_symbols = set(membership["symbol"].dropna().astype(str))
        pro = ts.pro_api(token)
        pro._DataApi__http_url = api_url
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        api_pieces: dict[str, list[pd.DataFrame]] = {api_name: [] for api_name in API_FIELDS}
        total_calls = len(periods) * len(API_FIELDS)
        completed_calls = 0
        for period in periods:
            for api_name, fields in API_FIELDS.items():
                checkpoint = CHECKPOINT_DIR / api_name / f"{period}.parquet"
                if checkpoint.exists():
                    part = pd.read_parquet(checkpoint)
                else:
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    print(f"下载{api_name}报告期{period}", flush=True)
                    part = _download_with_retry(
                        lambda api_name=api_name, period=period, fields=fields: pro.query(
                            api_name, period=period, fields=",".join(fields)
                        ),
                        f"{api_name}/{period}",
                    )
                    part.to_parquet(checkpoint, index=False)
                    time.sleep(request_interval)
                api_pieces[api_name].append(part)
                completed_calls += 1
                if completed_calls % 10 == 0 or completed_calls == total_calls:
                    print(f"财务下载进度 {completed_calls}/{total_calls}", flush=True)
        combined = {
            api_name: pd.concat(pieces, ignore_index=True)
            for api_name, pieces in api_pieces.items()
        }
        for api_name, data in combined.items():
            code_column = "ts_code"
            combined[api_name] = data.loc[data[code_column].isin(relevant_symbols)].copy()
        events = build_point_in_time_events(
            combined["income_vip"],
            combined["balancesheet_vip"],
            combined["fina_indicator_vip"],
            retrieved_at,
        )
        coverage = events["con_code"].nunique() / max(len(relevant_symbols), 1)
        if coverage < 0.95:
            raise ValueError(f"历史成员财务覆盖率仅{coverage:.2%}")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(OUTPUT_FILE, index=False)
        report = {
            "status": "PASS",
            "checked_at": retrieved_at.isoformat(),
            "api_host": api_url.split("//", 1)[-1].split("/", 1)[0],
            "request_interval_seconds": request_interval,
            "period_count": len(periods),
            "first_report_period": periods[0],
            "last_report_period": periods[-1],
            "relevant_symbol_count": len(relevant_symbols),
            "covered_symbol_count": int(events["con_code"].nunique()),
            "symbol_coverage": float(coverage),
            "event_count": int(len(events)),
            "first_available_at": str(events["available_at"].min().date()),
            "last_available_at": str(events["available_at"].max().date()),
            "output_file": OUTPUT_FILE.relative_to(ROOT).as_posix(),
            "output_sha256": _sha256(OUTPUT_FILE),
            "credential_note": "报告不保存或回显Token。",
        }
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        message = str(error)
        token = os.getenv("TUSHARE_TOKEN") or os.getenv("TS_TOKEN")
        if token:
            message = message.replace(token, "[已隐藏凭据]")
        print(f"点时财务下载失败：{type(error).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
