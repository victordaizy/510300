"""全因素估值、盈利、主线接入：历史解释与真实取得后的前瞻输入分开。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_all_factor_source_binding_v1"
TZ = ZoneInfo("Asia/Shanghai")
LOCAL = {
    "cases": "reports/research/510300_all_factor_joint_scorecard_v1/results/全部30历史案例_联合解释分与覆盖.parquet",
    "daily": "reports/research/510300_macro_technical_first_passage_v1_clock_adapter/results/全部3488当时已知技术与宏观_未知保留.parquet",
    "valuation": "reports/research/510300_historical_index_valuation_repricing_v1/daily_valuation.parquet",
    "valuation_definition": "reports/research/510300_historical_index_valuation_repricing_v1/definition_evidence.json",
    "earnings": "reports/research/510300_historical_index_earnings_population_v1/aggregates.parquet",
    "earnings_result": "reports/research/510300_historical_index_earnings_population_v1/result.json",
    "mainline": "reports/research/510300_broker_mainline_source_preflight_v1/summary.json",
    "industry_builder": "scripts/collect_a_share_hs_concentrated_low_risk_trend_v1_1_inputs.py",
}
QUOTE_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
FACTSHEET_URL = "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300factsheet.pdf"


def now():
    return datetime.now(TZ).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite(value):
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def normalize_quote(payload, expected_date):
    if payload.get("success") is not True or str(payload.get("code")) != "200":
        raise ValueError("官方响应未确认成功。")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("本次单日单指数请求未返回唯一原行。")
    row = rows[0]
    if str(row.get("indexCode")) != "000300" or str(row.get("tradeDate")) != expected_date:
        raise ValueError("指数或统计日不符，不能以其他行替代。")
    close, raw = finite(row.get("close")), finite(row.get("peg"))
    if close is None or close <= 0 or raw is None or raw <= 0:
        raise ValueError("官方原价格或peg字段不是有限正数。")
    return {"index_code": "000300", "statistics_date": pd.Timestamp(expected_date).strftime("%Y-%m-%d"),
            "index_close": close, "raw_peg": raw, "raw_peg_semantics": "RAW_FIELD_NOT_ASSUMED_PEG_RATIO_OR_TTM_PE"}


def parse_factsheet(text, expected_date):
    if "沪深300" not in text or "000300" not in text:
        raise ValueError("官方简表标的不符或无法识别。")
    dates = []
    for year, month, day in re.findall(r"(\d{4})[年/.\-]\s*(\d{1,2})[月/.\-]\s*(\d{1,2})日?", text[:2000]):
        try:
            dates.append(pd.Timestamp(int(year), int(month), int(day)))
        except ValueError:
            continue
    expected = pd.Timestamp(expected_date)
    if not dates or max(dates) != expected:
        raise ValueError("简表统计日期不等于本次目标日；不把旧简表当最新。")
    result = {"statistics_date": expected.strftime("%Y-%m-%d")}
    for label, key in [("滚动市盈率", "reported_ttm_pe"), ("市净率", "reported_pb"), ("股息率", "reported_dividend_yield_percent")]:
        match = re.search(re.escape(label) + r"\s*(?:[（(][^）)]*[）)])?\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            result[key] = float(match.group(1))
    if result.get("reported_ttm_pe", 0) <= 0:
        raise ValueError("简表没有可识别的滚动市盈率原值。")
    return result


def eligible_after_capture(snapshot, decision_time):
    available = pd.Timestamp(snapshot["available_at"])
    target = pd.Timestamp(decision_time)
    return (available.tzinfo is not None and target.tzinfo is not None
            and available <= target and snapshot.get("current_field_definition_status") == "OFFICIAL_FACTSHEET_LABEL_VERIFIED_CURRENT_VERSION_ONLY")


def required_period(date):
    if date.month <= 4:
        return str(date.year - 1) + "Q1Q3"
    if date.month <= 8:
        return str(date.year) + "Q1"
    if date.month <= 10:
        return str(date.year) + "H1"
    return str(date.year) + "Q1Q3"


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("本源接入用途已登记，不覆盖。")
    daily = pd.read_parquet(ROOT / LOCAL["daily"])
    date = pd.to_datetime(daily.date).max().strftime("%Y%m%d")
    protocol = {
        "study_id": "510300_ALL_FACTOR_SOURCE_BINDING_V1", "registered_at": now(),
        "registration": "TECH.R251", "decision": "TECH.R252",
        "purpose": "将全部30原案例的估值/盈利/主线实际覆盖、版本限制和未来真实官方输入分开；不把解释分当预测。",
        "sources": [{"key": key, "path": rel, "sha256": digest(ROOT / rel)} for key, rel in LOCAL.items()],
        "code_sha256": digest(Path(__file__).absolute()),
        "single_quote_request": {"url": QUOTE_URL, "params": {"indexCode": "000300", "startDate": date, "endDate": date}},
        "single_current_factsheet_request": FACTSHEET_URL,
        "network_attempts": "各一个本机请求，失败保留，不历史批量查询或换源重试。网页工具先前仅作定位。",
        "historical_binding": "估值仅上一实际交易日且原旧可得上界不晚于点位；只是当前版本解释。盈利只选当时最新完整报告期，后更新明确排除预测；行业生成时间戳不认证原公布。",
        "current_binding": "当前简表明确原统计日与指标标签；实际取得时间为使用上界，不回填9月30日或30历史案例。API原peg仅同日同值核对，不能把P/PE叫真实盈利。",
        "raw_history_role": "DEVELOPMENT_RETROSPECTIVE_CONTEXT_NOT_PIT_PREDICTIVE",
        "historical_model_admission": "NOT_ADMITTED", "new_models": 0, "new_accounts": 0,
        "goal_achieved": False,
    }
    write(OUT / "protocol.json", protocol)
    print("已登记三类来源接入与两个有界官方版本请求。")


def collect_one(key, url, params=None):
    folder = OUT / "current_official_sources"
    folder.mkdir(exist_ok=True)
    receipt = {"url": url, "params": params, "request_started_at": now(), "attempts": 1}
    try:
        response = requests.get(url, params=params, timeout=(8, 20), headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"})
        raw = folder / (key + (".pdf" if key == "factsheet" else ".json"))
        raw.write_bytes(response.content)
        receipt.update(http_status=response.status_code, received_at=now(), path=raw.absolute().relative_to(ROOT).as_posix(),
                       sha256=digest(raw), bytes=len(response.content), status="RECEIVED")
        response.raise_for_status()
        content = response.content
    except Exception as exc:
        receipt.update(received_at=now(), status="FAILED_SAVED_NO_RETRY", error=type(exc).__name__ + ": " + str(exc))
        content = None
    write(folder / (key + ".receipt.json"), receipt)
    return content, receipt


def bind_history(cases, daily, valuation, earnings):
    calendar = pd.DatetimeIndex(pd.to_datetime(daily.date)).sort_values()
    earnings = earnings[earnings.population.eq("DYNAMIC_300") & earnings.group_type.eq("全部") & earnings.measure.eq("YTD")]
    rows = []
    for case in cases.itertuples():
        date = pd.Timestamp(case.date)
        cutoff = pd.Timestamp(case.decision_at)
        previous_dates = calendar[calendar < date]
        previous = previous_dates[-1] if len(previous_dates) else None
        val = valuation.loc[previous] if previous is not None and previous in valuation.index else None
        if val is not None and pd.Timestamp(val.available_at) > cutoff:
            val = None
        period = required_period(date)
        company = earnings[earnings.period.eq(period)]
        company = company.iloc[0] if len(company) == 1 else None
        row = {"date": date, "decision_at": cutoff.isoformat(), "core_working_score_unchanged": case.core_explanatory_score,
               "valuation_statistics_date": previous if val is not None else pd.NaT,
               "official_original_pe": finite(val.original_pe_official) if val is not None else None,
               "vendor_static_pe": finite(val.pe_static) if val is not None else None,
               "vendor_ttm_pe": finite(val.pe_ttm) if val is not None else None,
               "vendor_pb": finite(val.pb) if val is not None else None,
               "valuation_role": "CURRENT_VERSION_RETROSPECTIVE_ONLY" if val is not None else "UNKNOWN_NO_MATCHING_PREVIOUS_SESSION",
               "valuation_historical_definition_verified": False, "valuation_first_vintage_verified": False,
               "required_earnings_period": period,
               "reconstructed_profit_yoy": finite(company.parent_net_profit_yoy) if company is not None else None,
               "earnings_valid_members": int(company.valid_members) if company is not None else None,
               "earnings_later_updated_members": int(company.late_updated_members) if company is not None else None,
               "earnings_role": "LATER_UPDATED_RECONSTRUCTION_NOT_PIT" if company is not None else "UNKNOWN_REQUIRED_PERIOD_NOT_PRESENT",
               "mainline_role": "GENERATED_CLASSIFICATION_CLOCK_NOT_FIRST_RELEASE_CERTIFICATION",
               "new_historical_predictive_factor_admission": False,
               "full_predictive_score": "NOT_COMPUTED", "new_trade_or_return": "NOT_COMPUTED"}
        rows.append(row)
    return pd.DataFrame(rows)


def run():
    p = read(OUT / "protocol.json")
    assert p["code_sha256"] == digest(Path(__file__).absolute())
    for source in p["sources"]:
        assert digest(ROOT / source["path"]) == source["sha256"], source["path"]
    if (OUT / "run_started.json").exists():
        raise FileExistsError("本轮接入已经启动，禁止覆盖重跑。")
    write(OUT / "run_started.json", {"started_at": now(), "new_accounts": 0})
    cases = pd.read_parquet(ROOT / LOCAL["cases"])
    history = bind_history(cases, pd.read_parquet(ROOT / LOCAL["daily"]), pd.read_parquet(ROOT / LOCAL["valuation"]), pd.read_parquet(ROOT / LOCAL["earnings"]))
    results = OUT / "results"
    results.mkdir()
    history.to_parquet(results / "全部30案例_估值盈利主线接入及用途.parquet", index=False)
    history.to_csv(results / "全部30案例_估值盈利主线接入及用途.csv", index=False, encoding="utf-8-sig")
    expected = p["single_quote_request"]["params"]["endDate"]
    quote_bytes, quote_receipt = collect_one("quote", QUOTE_URL, p["single_quote_request"]["params"])
    pdf_bytes, pdf_receipt = collect_one("factsheet", FACTSHEET_URL)
    snapshot = {"captured_at": now(), "available_at": now(), "role": "CURRENT_CAPTURE_BASELINE_NOT_HISTORICAL_OR_NEW_FORWARD_OUTCOME",
                "current_field_definition_status": "NOT_ESTABLISHED", "future_input_fields": [], "historical_use": False}
    if quote_bytes is not None:
        try:
            snapshot["quote"] = normalize_quote(json.loads(quote_bytes), expected)
        except Exception as exc:
            snapshot["quote_normalization_error"] = type(exc).__name__ + ": " + str(exc)
    if pdf_bytes is not None:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as document:
                text = "\n".join(page.extract_text() or "" for page in document.pages)
            (OUT / "current_official_sources/factsheet.txt").write_text(text, encoding="utf-8")
            snapshot["factsheet"] = parse_factsheet(text, expected)
            snapshot["current_field_definition_status"] = "OFFICIAL_FACTSHEET_LABEL_VERIFIED_CURRENT_VERSION_ONLY"
            snapshot["future_input_fields"] = [key for key in snapshot["factsheet"] if key != "statistics_date"]
            if "quote" in snapshot:
                snapshot["current_same_date_peg_matches_reported_ttm_pe"] = abs(snapshot["quote"]["raw_peg"] - snapshot["factsheet"]["reported_ttm_pe"]) <= .005
        except Exception as exc:
            snapshot["factsheet_normalization_error"] = type(exc).__name__ + ": " + str(exc)
    write(OUT / "current_official_snapshot.json", snapshot)
    summary = {"study_id": p["study_id"], "completed_at": now(), "status": "SOURCE_BINDING_COMPLETED_HISTORICAL_PREDICTIVE_FIELDS_NOT_ADMITTED",
               "historical_cases": len(history), "historical_valuation_context_cases": int(history.official_original_pe.notna().sum()),
               "historical_earnings_reconstruction_cases": int(history.reconstructed_profit_yoy.notna().sum()),
               "new_historical_predictive_factor_admissions": 0, "current_official_requests": 2,
               "current_request_statuses": {"quote": quote_receipt["status"], "factsheet": pdf_receipt["status"]},
               "current_explicit_reported_fields": snapshot["future_input_fields"],
               "current_available_at": snapshot["available_at"], "new_fits": 0, "new_accounts": 0,
               "new_independent_completed_points": 0, "new_strategy_Sharpe": "NOT_COMPUTED", "goal_achieved": False,
               "next_action": "当前真实版本若标签/时点合格，纳入之后可知的不同来源输入；预测和完整账户用途另登记，不用后修订历史救原联合模型。"}
    write(OUT / "summary.json", summary)
    lines = ["# 全因素评分的估值、盈利、主线接入", "",
             "全部30原案例已逐一检查。历史可解释数据与进场前可预测数据分开；原工作分没有改成新的胜率或交易。", "",
             "| 日期 | 上一交易日官方原PE | 第三方TTM PE | 盈利解释值 | 预测用途 |", "|---|---:|---:|---:|---|"]
    for row in history.itertuples():
        if pd.notna(row.official_original_pe) or pd.notna(row.reconstructed_profit_yoy):
            pe = "未知" if pd.isna(row.official_original_pe) else f"{row.official_original_pe:.2f}"
            ttm = "未知" if pd.isna(row.vendor_ttm_pe) else f"{row.vendor_ttm_pe:.2f}"
            profit = "未知" if pd.isna(row.reconstructed_profit_yoy) else f"后来重建{row.reconstructed_profit_yoy:.2%}"
            lines.append(f"| {row.date:%Y-%m-%d} | {pe} | {ttm} | {profit} | NOT_ADMITTED |")
    lines += ["", "原官方PE与第三方静态/TTM估值口径不同，P/PE只是代数分母，不能冒充实际指数利润。历史首版与原定义未认证，不据绝对数值给低估加分。", "",
              "2023Q1的动态300汇总有299个有效成员、299个后更新成员。其重建归母净利同比不代表当时首次披露可见的数值；预测明确排除。其他时期不把这一个季度向后填充。", "",
              "主线分类的available_at由程序根据valid_from生成；生效日期、历史首次公布和当前实际取得是不同时间。月末权重也不是每天真实披露的权重。", "",
              "## 当前实际官方版本", "", "一次单日指数API和一次官方简表请求，原响应及实际取得钟已保存。当前原文标签可验证的字段为：" + "、".join(snapshot["future_input_fields"]) + "。", "",
              "实际可用上界：" + snapshot["available_at"] + "。仅在这之后的研究决定使用；不能回填9月30日或30原案例。API的peg原字段与简表同日同值核对只适用于当前版本，不认证全历史口径。", "",
              "本轮0账户/拟合/新收益标签。完整收益/Sharpe目标、独立验证和预测用途尚未完成；原E03和原金融冻结结果保持。", "",
              "[官方简表](" + FACTSHEET_URL + ")；本机原件及回执位于current_official_sources。"]
    (OUT / "全因素接入_估值盈利主线与真实版本.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全因素来源接入与当前官方版本；不运行金融账户。")
    parser.add_argument("action", choices=["freeze", "run"])
    freeze() if parser.parse_args().action == "freeze" else run()
