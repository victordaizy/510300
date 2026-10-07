"""有限取得新报价年份，核对离岸末价与官方定盘偏离，不计算股票未来结果。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research import macro_technical_first_passage_study_v1 as common
from scripts.collect_rmb_residual_sources_v1 import normalize_dukascopy

ROOT = common.ROOT
OUT = ROOT / "reports/research/510300_cnh_fixing_deviation_source_preflight_v1"
CARD = ROOT / "docs/510300_CNH_FIXING_DEVIATION_SOURCE_PREFLIGHT_V1.md"
OLD = ROOT / "reports/research/510300_rmb_residual_state_v1"
QUOTES = OLD / "inputs/cnh.parquet"
FIXING = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/usdcny_midpoint_daily_2015_2026.parquet"
CALENDAR = common.BASE / "results/原点全部技术特征.parquet"
YEARS = (2015, 2018)
URL = "https://jetta.dukascopy.com/v1/candles/day/USD-CNH/BID/{year}"


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def canonical(frame):
    result = frame.copy()
    for column in result.columns:
        dtype = result[column].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            target = (pd.DatetimeTZDtype(unit="ns", tz=dtype.tz)
                      if isinstance(dtype, pd.DatetimeTZDtype) else "datetime64[ns]")
            converted = result[column].astype(target)
            pd.testing.assert_series_equal(converted.astype(dtype), result[column], check_exact=True)
            result[column] = converted
    return result


def table(name, frame):
    directory = OUT / "results"
    directory.mkdir(exist_ok=True)
    frame.to_parquet(directory / (name + ".parquet"), index=False)
    frame.to_csv(directory / (name + ".csv"), index=False, encoding="utf-8-sig")


def freeze():
    common.require(not OUT.exists(), "有限来源用途已存在，不覆盖。")
    quote = pd.read_parquet(QUOTES)
    common.require(not quote.date.dt.year.isin(YEARS).any(), "两个探测年份已在原报价缓存中，不重复采集。")
    prior = common.read(OLD / "result.json")
    common.require(prior["status"] == "SOURCE_GATE_NOT_PASSED" and prior["account_status"] == "NOT_RUN",
                   "旧人民币残差裁决改变，须先核对。")
    paths = [Path(__file__), CARD, QUOTES, FIXING, CALENDAR, OLD / "result.json",
             OLD / "sources/dukascopy_usdcnh_identity.json", ROOT / "scripts/collect_rmb_residual_sources_v1.py"]
    sources = [{"path": relative(path), "sha256": common.digest(path)} for path in paths]
    OUT.mkdir(parents=True)
    (OUT / "raw").mkdir()
    common.write_json(OUT / "protocol.json", {
        "at": common.now(), "study": "510300_CNH_FIXING_DEVIATION_SOURCE_PREFLIGHT_V1",
        "registration_decision": "TECH.R205", "result_decision": "TECH.R206", "purpose": relative(CARD),
        "request_years": YEARS, "requests_maximum": 2, "retry_count": 0, "request_timeout_seconds": 20,
        "tls_verification": True, "redirects": False, "credentials": False, "sources": sources,
        "source_same_day_match_required": True, "gap_formula": "10000*log(CNH_BID_CLOSE/CNY_FIXING)",
        "source_available_at": "统计日后第二自然日23:59上海，且晚于原报价柱末及中间价源钟",
        "maximum_age_calendar_days": 7, "observation_clock": "原3488日期16:00上海",
        "change5": "原ETF日历连续六个已知槽才计算五槽差",
        "new_training_labels": 0, "new_stock_outcomes": 0, "new_fits": 0, "new_accounts": 0,
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "original_residual_result_preserved": True}, exclusive=True)
    print("TECH.R205有限新信息用途已登记：离岸报价／官方定盘偏离，最多两个年度请求。", flush=True)


def obtain(year):
    url = URL.format(year=year)
    body = OUT / "raw" / f"USD_CNH_BID_{year}.response"
    receipt_path = OUT / "raw" / f"USD_CNH_BID_{year}.receipt.json"
    common.require(not receipt_path.exists(), "年度请求已有回执，不重复执行。")
    receipt = {"year": year, "url": url, "started_at": common.now(), "attempts": 1,
               "quote_side": "BID", "instrument": "USD-CNH", "success": False,
               "historical_first_delivery": "UNKNOWN", "historical_available_at": None}
    frame = None
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(url, timeout=20, verify=True, allow_redirects=False,
                                   headers={"User-Agent": "Mozilla/5.0"})
        body.write_bytes(response.content)
        receipt.update(http_status=response.status_code, response_url=response.url,
                       server_date=response.headers.get("Date"), raw_path=relative(body),
                       raw_sha256=common.digest(body))
        response.raise_for_status()
        frame = canonical(normalize_dukascopy(response.json()))
        common.require(len(frame) > 0 and frame.date.dt.year.eq(year).all(), "年度响应为空或包含不同年份。")
        common.require(frame.date.is_unique and frame.date.is_monotonic_increasing, "报价统计日不唯一或未排序。")
        frame["source_file"] = relative(body)
        receipt.update(success=True, rows=len(frame), first_date=frame.date.min().isoformat(),
                       last_date=frame.date.max().isoformat(), clock="原UTC柱起止保留")
    except Exception as error:
        frame = None
        receipt.update(error_type=type(error).__name__, error=str(error))
    receipt["completed_at"] = common.now()
    common.write_json(receipt_path, receipt, exclusive=True)
    print(f"{year}离岸年度响应：{'字段核对通过' if receipt['success'] else '失败已保存，不重试'}。", flush=True)
    return receipt, frame


def run():
    common.require(not (OUT / "RUN_STARTED.json").exists(), "来源核对已启动，不重复执行。")
    protocol = common.read(OUT / "protocol.json")
    for item in protocol["sources"]:
        common.require(common.digest(ROOT / item["path"]) == item["sha256"], "登记来源改变：" + item["path"])
    common.write_json(OUT / "RUN_STARTED.json", {"at": common.now(), "maximum_new_requests": 2}, exclusive=True)
    receipts, frames = [], [canonical(pd.read_parquet(QUOTES))]
    for year in YEARS:
        receipt, frame = obtain(year)
        receipts.append(receipt)
        if frame is not None:
            frames.append(frame)
            table(f"新取得{year}年离岸原报价及柱钟", frame)
    old = canonical(pd.read_parquet(QUOTES))
    quotes = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    common.require(quotes.date.is_unique, "新旧报价日期重叠，不事后择供应商版本。")
    inherited = quotes.loc[quotes.date.dt.year.isin(old.date.dt.year.unique()), old.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(old.reset_index(drop=True), inherited, check_exact=True)
    fixing = pd.read_parquet(FIXING).copy()
    fixing["date"] = pd.to_datetime(fixing.date).astype("datetime64[ns]")
    fixing["fixing_available_at"] = common.model.time_shanghai(fixing.available_at)
    fixing = fixing.rename(columns={"first_release_value": "fixing_value", "source_url": "fixing_url",
                                     "raw_path": "fixing_raw_path"})
    common.require(fixing.date.is_unique and fixing.fixing_value.gt(0).all(), "中间价重复或单位不合格。")
    pair = quotes.merge(fixing[["date", "fixing_value", "fixing_available_at", "fixing_url", "fixing_raw_path"]],
                        on="date", how="left", validate="one_to_one")
    pair["same_date_pair_known"] = pair.fixing_value.notna()
    pair["fixing_deviation_log_bp"] = 10000. * np.log(pair.close / pair.fixing_value)
    pair["conservative_available_at"] = (pair.date.dt.tz_localize("Asia/Shanghai") +
                                         pd.Timedelta(days=2, hours=23, minutes=59))
    bar_end = pair.bar_end_utc.dt.tz_convert("Asia/Shanghai").astype("datetime64[ns, Asia/Shanghai]")
    pair["conservative_available_at"] = pd.concat([pair.conservative_available_at, bar_end,
                                                    pair.fixing_available_at], axis=1).max(axis=1)
    pair["historical_first_vintage"] = "NOT_CERTIFIED"
    common.require((pair.conservative_available_at >= bar_end).all(), "报价柱尚未完成就可使用。")
    valid = pair.loc[pair.same_date_pair_known].copy().sort_values("conservative_available_at")
    common.require(valid.conservative_available_at.is_unique, "偏离源可得钟重复。")
    data = pd.read_parquet(CALENDAR)[["date", "daily_hist_atr", "weekly_hist_atr", "log_relative_volume"]].copy()
    data["date"] = pd.to_datetime(data.date).astype("datetime64[ns]")
    data["decision_time"] = data.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    right = valid.rename(columns={"date": "fx_stat_date", "close": "cnh_bid_close"})
    view = pd.merge_asof(data, right[["fx_stat_date", "cnh_bid_close", "fixing_value", "bar_end_utc",
                                      "fixing_available_at", "conservative_available_at", "fixing_deviation_log_bp"]],
                         left_on="decision_time", right_on="conservative_available_at", direction="backward")
    view["source_age_calendar_days"] = (view.date - view.fx_stat_date).dt.total_seconds() / 86400.
    view["deviation_known"] = (view.source_age_calendar_days.between(0., 7.) & view.fx_stat_date.lt(view.date)
                                & view.conservative_available_at.le(view.decision_time)
                                & np.isfinite(view.fixing_deviation_log_bp))
    view.loc[~view.deviation_known, "fixing_deviation_log_bp"] = np.nan
    view["fixing_deviation_change5_log_bp"] = view.fixing_deviation_log_bp.diff(5).where(
        view.fixing_deviation_log_bp.rolling(6, min_periods=6).count().eq(6))
    view["both_fields_known"] = view.deviation_known & view.fixing_deviation_change5_log_bp.notna()
    common.require(view.loc[view.deviation_known, "conservative_available_at"].le(
        view.loc[view.deviation_known, "decision_time"]).all(), "偏离使用了未可得数据。")
    common.require(np.allclose(view.loc[view.deviation_known, "fixing_deviation_log_bp"],
        10000. * np.log(view.loc[view.deviation_known, "cnh_bid_close"] /
                       view.loc[view.deviation_known, "fixing_value"]), rtol=0., atol=1e-10), "价差单位改变。")
    yearly = view.assign(year=view.date.dt.year).groupby("year").agg(
        observation_slots=("date", "size"), level_known=("deviation_known", "sum"),
        both_fields_known=("both_fields_known", "sum")).reset_index()
    table("全部离岸报价与同统计日中间价_缺失保留", pair)
    table("全部3488观察槽_离岸相对定盘偏离与保守钟", view)
    table("全部年份偏离来源覆盖_不计算股票收益", yearly)
    cases = view.loc[view.date.isin(pd.to_datetime(common.CASE_DATES))].copy()
    table("原17关键日_技术与不同步定盘背景不评分", cases)
    table("两个固定年度请求及全部失败", pd.DataFrame(receipts))
    successful = sum(receipt["success"] for receipt in receipts)
    summary = {"at": common.now(), "study": protocol["study"], "registration_decision": "TECH.R205",
        "decision": "TECH.R206", "status": "SOURCE_DEVIATION_FEASIBILITY_COMPLETED_NOT_FINANCIAL_ADMISSION",
        "original_quote_rows_exact": len(old), "new_quotes": len(quotes) - len(old),
        "native_requests": len(receipts), "successful_year_requests": successful,
        "requested_years": list(YEARS), "quote_rows_total": len(quotes),
        "same_stat_date_quote_fixing_pairs": int(pair.same_date_pair_known.sum()),
        "all_observation_slots": len(view), "level_known_slots": int(view.deviation_known.sum()),
        "both_fields_known_slots": int(view.both_fields_known.sum()),
        "original_case_days": len(cases), "source_same_day_market_prices": False,
        "source_definition": "离岸UTC日BID末价相对同统计日中国09:15官方中间价，二者实际时刻不同",
        "capital_flow_or_policy_surprise_identified": False,
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "new_training_labels": 0, "new_return_label_reads": 0, "new_stock_outcomes": 0,
        "new_fits": 0, "new_accounts": 0, "new_financial_metrics": "NOT_COMPUTED",
        "financial_admission": "NOT_REGISTERED_REQUIRES_DIFFERENT_COMPLETE_PURPOSE",
        "latest_actual_financial_decision": "TECH.R198", "goal_achieved": False, "overfitting_removed": False,
        "original_rmb_residual_result_preserved": True,
        "next_action": ("两个年度字段成功后，只核对该不同价差信息的完整可知成员和旧用途差别，"
                        "另登记唯一金融用途才拟合；不改变原宏观失败或未知状态。" if successful == 2 else
                        "两个年度未全部成功，关闭本次有限请求，不重试或补年份；当前不准入金融。")}
    common.write_json(OUT / "summary.json", summary, exclusive=True)
    text = "# 离岸人民币相对官方定盘偏离：有限来源结果\n\n"
    text += (f"TECH.R205—TECH.R206固定两年度请求完成，成功{successful}/2，"
             f"新增{len(quotes)-len(old)}条报价；原{len(old)}条缓存逐值和柱钟精确保持。"
             "本次不读取股票标签，不拟合或运行账户，不宣称改善收益夏普。\n\n")
    text += ("经济对象必须分清：本地在岸字段是官方中间价，而非在岸即期收盘；离岸为供应商UTC日BID末价。"
             "两者不是同步市场价差。该偏离是不同信息候选，不是资本流入、外生汇率意外或政策意外。"
             "旧人民币美元/利差残差SOURCE_GATE_NOT_PASSED及ACCOUNT_NOT_RUN保持。\n\n")
    text += ("源日期后第二自然日23:59上海的保守钟，晚于UTC柱末与定盘源钟；年龄超过7日或缺资料保留未知。"
             "数值为10000×log(离岸BID末价/官方定盘)，五槽变化只有连续六个ETF槽已知才可计算。"
             "保守时钟不能证明历史首版。\n\n")
    text += yearly.to_markdown(index=False) + "\n\n"
    text += "原固定17关键日全部保留，以下是当时可计算的技术及外汇背景，不是模型分数或买点。\n\n"
    text += cases[["date", "daily_hist_atr", "weekly_hist_atr", "fx_stat_date", "cnh_bid_close", "fixing_value",
                   "fixing_deviation_log_bp", "fixing_deviation_change5_log_bp", "both_fields_known"]].to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += summary["next_action"] + "\n\n"
    text += ("完整目标未达，最近金融仍TECH.R198；实际净pB、净期望、收益夏普、历史稳定和独立验证必须另测。"
             "本次没有授权重开旧人民币残差用途、改原树/窗口/阈值/退出或把未知交给原A填补。\n\n"
             "[所有年度请求与失败](results/两个固定年度请求及全部失败.csv)、"
             "[全部观察槽](results/全部3488观察槽_离岸相对定盘偏离与保守钟.csv)、[实际结果](summary.json)。\n")
    (OUT / "离岸相对定盘偏离_来源合同与全部案例.md").write_bytes(text.encode("utf-8"))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离岸相对官方定盘偏离有限来源核对")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as error:
            if OUT.exists() and not (OUT / "failure.json").exists():
                common.write_json(OUT / "failure.json", {"at": common.now(), "error_type": type(error).__name__,
                                  "error": str(error), "new_fits": 0, "new_accounts": 0}, exclusive=True)
            raise
