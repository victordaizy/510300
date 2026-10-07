"""只为当前点位任务补充已完成日线与股息覆盖，不调用旧投资账户或自动化。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import new_daily_input_adapter_v1 as adapter

OUT = ROOT / "reports/research/510300_point_current_inputs_20261001"
HISTORY = ROOT / "reports/research/510300_point_history_extension_v1"
OFFICIAL = "config/510300_official_dividend_coverage_refresh_v1.json"
CALENDAR_URL = "https://www.sse.com.cn/disclosure/announcement/general/c/c_20260915_10832273.shtml"


def task_relative(value):
    """reports目录联接到E盘，回执仍使用工作区可移植相对路径。"""
    path = Path(value).absolute()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        local = path.resolve().relative_to(OUT.resolve())
        return str(OUT.relative_to(ROOT) / local)


def run(attempt):
    if (OUT / "admission_receipt.json").exists():
        print("本次当前日线已接纳，读取既有回执即可，无需再次采集。", flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    previous = pd.read_parquet(HISTORY / "inputs/prices.parquet").date.iloc[-1]
    calendar = pd.to_datetime(pd.read_csv(ROOT / "data/reference/sse_trade_calendar_2026.csv").trade_date)
    dates, cutoff, next_day = adapter.completed_dates(calendar, previous, common.now())
    if cutoff != "2026-09-30" or next_day != "2026-10-08":
        raise ValueError("本次日期边界与已经核对的上交所国庆休市公告不同。")
    source = {"prices": str((HISTORY / "inputs/prices.parquet").relative_to(ROOT)),
              "features": str((HISTORY / "results/features.parquet").relative_to(ROOT)),
              "dividends": str((HISTORY / "inputs/dividends.csv").relative_to(ROOT)),
              "coverage": str((HISTORY / "inputs/dividend_coverage.json").relative_to(ROOT))}
    request = {"study": "510300_POINT_CURRENT_INPUTS_20261001", "at": common.now(), "attempt": attempt,
               "previous_known_close": str(previous.date()), "requested_cutoff": cutoff, "next_exchange_session": next_day,
               "expected_dates": [str(d.date()) for d in dates], "sources": source, "official_configuration": OFFICIAL,
               "calendar_official_notice": CALENDAR_URL, "calendar_verified_on": "2026-10-01",
               "scope": "新浪与腾讯未复权日线、基金管理人分红登记及上交所完整公告查询；保留旧前缀并只写本任务新目录。",
               "prospective_class": "BACKFILLED_HISTORY_NOT_PREOPEN_SIGNALS", "new_strategy_evaluations": 0, "orders_authorized": False,
               "source_hashes": {key: common.digest(ROOT / value) for key, value in source.items()},
               "program_sha256": common.digest(Path(__file__))}
    target = OUT / f"request_attempt{attempt}.json"
    if not target.exists():
        common.save_json(target, request)
    try:
        # 工作区根在C盘，reports目录联接到E盘；相对路径保留逻辑工作区入口。
        # 只替换本进程的路径显示函数，不改旧配置、源文件或任何旧研究状态。
        adapter.relative = task_relative
        receipt = adapter.collect_and_admit({"official_configuration": OFFICIAL}, source, OUT, dates, attempt)
        summary = {"status": "CURRENT_DAILY_INPUTS_ADMITTED", "at": common.now(), **receipt,
                   "next_exchange_session": next_day, "prospective_observations": 0, "current_candidate_signal_computed": False}
        common.save_json(OUT / "summary.json", summary)
        print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        summary = {"status": "NO_VIEW_CURRENT_INPUTS_NOT_ADMITTED", "at": common.now(), "attempt": attempt,
                   "error_type": type(error).__name__, "error": str(error), "old_inputs_preserved": True,
                   "requested_cutoff": cutoff, "last_admitted_close": str(previous.date()), "prospective_observations": 0,
                   "current_candidate_signal_computed": False, "orders_authorized": False}
        common.save_json(OUT / f"failure_attempt{attempt}.json", summary)
        common.save_json(OUT / "summary.json", summary)
        print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="补当前510300点位任务所需日线，不启动旧账户研究。")
    parser.add_argument("--attempt", type=int, default=1)
    run(parser.parse_args().attempt)
