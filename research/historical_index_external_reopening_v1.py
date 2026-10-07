"""历史指数共同驱动：通胀构成、政策路径、国内活动约束与统一账户。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import historical_index_reopening_constraints_v1 as domestic
from research import historical_index_rrr_full_account_v1 as checks
from research import factor96_bottleneck_diagnostic_v1 as account_model
from research import factor96_margin_repair_v1 as core
OUT = ROOT / "reports/research/510300_historical_index_external_reopening_v1"
V15 = ROOT / "reports/research/510300_external_discount_clock_v15"
PRIOR = ROOT / "reports/research/510300_historical_index_reopening_constraints_v1"
TZ = ZoneInfo("Asia/Shanghai")
SIGNAL = "external_or_domestic_public_before_open"


def serializable(value):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法序列化：{type(value)}")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=serializable), encoding="utf-8")
    temporary.replace(path)


def source_plan() -> list[dict]:
    protocol = read(OUT / "protocol.json")
    sources = []
    for date in protocol["cpi_dates"]:
        stamp = pd.Timestamp(date).strftime("%m%d%Y")
        sources.append({"id": "cpi_" + date, "filename": f"cpi_{stamp}.pdf", "kind": "CPI",
                        "url": f"https://www.bls.gov/news.release/archives/cpi_{stamp}.pdf",
                        "reuse": str(V15 / "sources" / f"CPI_{stamp}.pdf"), "date": date})
    for date in [protocol["prior_fomc_for_first_step"]] + protocol["fomc_dates"]:
        stamp = date.replace("-", "")
        sources.append({"id": "fed_" + date, "filename": f"fed_{stamp}.html", "kind": "FOMC",
                        "url": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{stamp}a.htm",
                        "reuse": str(V15 / "sources" / f"Fed_{stamp}.html"), "date": date})
    for date in ["2022-09-21", "2022-12-14"]:
        stamp = date.replace("-", "")
        sources.append({"id": "sep_" + date, "filename": f"sep_{stamp}.pdf", "kind": "SEP",
                        "url": f"https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl{stamp}.pdf",
                        "reuse": str(V15 / "sources" / f"SEP_{stamp}.pdf"), "date": date})
    sources.append({"id": "powell_2022-11-30", "filename": "powell_20221130.html", "kind": "SPEECH",
                    "url": "https://www.federalreserve.gov/newsevents/speech/powell20221130a.htm", "date": "2022-11-30"})
    for year in [2022, 2023]:
        for kind, api in [("nominal", "daily_treasury_yield_curve"), ("tips", "daily_treasury_real_yield_curve")]:
            sources.append({"id": f"{kind}_{year}", "filename": f"{kind}_{year}.xml", "kind": "TREASURY",
                            "url": f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data={api}&field_tdr_date_value={year}",
                            "reuse": str(ROOT / f"reports/research/510300_rmb_residual_state_v1/sources/treasury_{year}.xml") if kind == "nominal" else None})
    return sources


def collect() -> None:
    folder = OUT / "sources"
    folder.mkdir(exist_ok=True)
    manifest_path = OUT / "source_manifest.json"
    receipts = read(manifest_path) if manifest_path.exists() else []
    prior = {r["id"]: r for r in receipts if r["status"] in ["FETCHED", "REUSED"]}

    def fetch(source: dict) -> dict:
        record = {**source, "retrieved_or_reused_at": datetime.now(TZ).isoformat()}
        path = folder / source["filename"]
        try:
            old = Path(source["reuse"]) if source.get("reuse") else None
            if old and old.exists():
                raw = old.read_bytes()
                record["status"] = "REUSED"
            else:
                response = requests.get(source["url"], timeout=30, headers={"User-Agent": "Mozilla/5.0"})
                record.update(http_status=response.status_code, final_url=response.url)
                response.raise_for_status()
                raw = response.content
                record["status"] = "FETCHED"
            if path.suffix == ".pdf" and not raw.startswith(b"%PDF"):
                raise ValueError("返回内容不是PDF")
            path.write_bytes(raw)
            if path.suffix == ".pdf":
                import pdfplumber
                with pdfplumber.open(path) as doc:
                    # CPI前四页含摘要、表1与主要组成解释；SEP第一页含中位预测。
                    count = min(4 if source["kind"] == "CPI" else 2, len(doc.pages))
                    pages = [{"page": i + 1, "text": doc.pages[i].extract_text() or ""} for i in range(count)]
                save(path.with_suffix(".pages.json"), pages)
                path.with_suffix(".txt").write_text("\n\n".join(p["text"] for p in pages), encoding="utf-8")
            elif path.suffix == ".html":
                soup = BeautifulSoup(raw, "html.parser")
                path.with_suffix(".txt").write_text(soup.get_text(" ", strip=True), encoding="utf-8")
            record.update(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        except Exception as error:
            record.update(status="UNAVAILABLE", error=str(error))
        return record

    jobs = [s for s in source_plan() if s["id"] not in prior]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, s) for s in jobs]):
            record = future.result()
            receipts.append(record)
            save(manifest_path, receipts)
            print(record["id"], record["status"], flush=True)


def cpi_facts(protocol: dict) -> list[dict]:
    labels = {
        "headline": "All items", "core": "All items less food and energy",
        "core_goods": "Commodities less food and energy commodities",
        "core_services": "Services less energy services", "shelter": "Shelter",
        "energy": "Energy", "gasoline": "Gasoline (all types)",
        "used_cars": "Used cars and trucks",
    }
    records = []
    manifest = read(OUT / "source_manifest.json")
    existing_ids = {r["id"] for r in manifest}
    for date in protocol["cpi_dates"]:
        stamp = pd.Timestamp(date).strftime("%m%d%Y")
        source = OUT / "sources" / f"cpi_{stamp}.web.txt"
        raw = source.read_text(encoding="utf-8")
        text = re.sub(r"L\d+:\s?", "", raw)
        text = re.sub(r"cite.*?", "", text)
        table = text[text.index("Table A."):]
        assert "8:30 a.m." in text[:text.index("Table A.")], date
        record = {"event_id": "CPI_" + date.replace("-", ""), "kind": "CPI",
                  "date": date, "title": date + " 美国CPI",
                  "economic_month": str(pd.Timestamp(date).to_period("M") - 1),
                  "source_url": f"https://www.bls.gov/news.release/archives/cpi_{stamp}.htm",
                  "source_path": source.relative_to(ROOT).as_posix(),
                  "units": "百分数，环比季调、同比未季调", "market_consensus": None,
                  "actual_minus_consensus": None, "components": {}}
        for key, label in labels.items():
            pattern = r"\s*".join(re.escape(w) for w in label.split())
            match = re.search(pattern, table)
            assert match is not None, (date, label)
            # 原版排版省略小数前的零，需同时支持 .6、-.4 和 0.6。
            numbers = list(re.finditer(r"[-+]?(?:\d+\.\d+|\.\d+)", table[match.end():]))[:8]
            assert len(numbers) == 8, (date, label)
            values = [float(m.group()) for m in numbers]
            end = match.end() + numbers[-1].end()
            record["components"][key] = {
                "previous_mom": values[-3], "current_mom": values[-2], "current_yoy": values[-1],
                "seven_month_mom": values[:7], "evidence_row": table[match.start():end].strip(),
            }
        c = record["components"]["core"]
        record["trigger"] = c["current_mom"] < c["previous_mom"]
        record["trigger_reason"] = "同版公告核心环比下降" if record["trigger"] else "核心环比未下降"
        previous = records[-1] if records else None
        record["previous_month_core_first_reported"] = previous["components"]["core"]["current_mom"] if previous else None
        record["previous_core_differs_from_prior_release"] = bool(previous and c["previous_mom"] != previous["components"]["core"]["current_mom"])
        record["february_seasonal_update_release"] = pd.Timestamp(date).month == 2
        records.append(record)
        source_id = "cpi_web_" + date
        if source_id not in existing_ids:
            manifest.append({"id": source_id, "kind": "CPI", "date": date,
                             "url": record["source_url"], "path": record["source_path"],
                             "status": "WEB_TOOL_EXTRACTED_PRIMARY_SOURCE",
                             "recorded_at": datetime.now(TZ).isoformat(),
                             "file_saved_at": datetime.fromtimestamp(source.stat().st_mtime, TZ).isoformat(),
                             "receipt_is_original_publication_time": False,
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "bytes": source.stat().st_size})
    # 原文人工复核的核心数值，仅验证解析，不用来选择交易规则。
    reviewed = [(0.5, 0.6), (0.6, 0.6), (0.6, 0.5), (0.5, 0.3), (0.3, 0.6),
                (0.6, 0.6), (0.6, 0.7), (0.7, 0.3), (0.3, 0.6), (0.6, 0.6),
                (0.6, 0.3), (0.3, 0.2), (0.2, 0.3), (0.4, 0.4)]
    assert [(r["components"]["core"]["previous_mom"], r["components"]["core"]["current_mom"]) for r in records] == reviewed
    save(OUT / "source_manifest.json", manifest)
    save(OUT / "cpi_facts.json", records)
    return records


def rate_number(value: str) -> float:
    parts = value.strip().replace("‑", "-").split("-")
    result = 0.
    for part in parts:
        if "/" in part:
            a, b = part.split("/")
            result += float(a) / float(b)
        else:
            result += float(part)
    return result


def fomc_facts(protocol: dict) -> list[dict]:
    records, last_upper, last_positive_step = [], None, None
    for date in [protocol["prior_fomc_for_first_step"]] + protocol["fomc_dates"]:
        stamp = date.replace("-", "")
        source = OUT / "sources" / f"fed_{stamp}.txt"
        text = source.read_text(encoding="utf-8")
        assert "For release at 2:00 p.m." in text
        match = re.search(r"target range for the federal funds rate (?:at|to) ([\d/\-‑]+) to ([\d/\-‑]+) percent", text)
        assert match, date
        lower, upper = [rate_number(x) for x in match.groups()]
        step = round((upper - last_upper) * 100, 8) if last_upper is not None else None
        trigger = step is not None and step > 0 and last_positive_step is not None and step < last_positive_step
        row = {"event_id": "FOMC_" + stamp, "kind": "FOMC", "date": date,
               "title": date + " 美联储议息", "range_lower_pct": lower, "range_upper_pct": upper,
               "step_bp": step, "previous_positive_step_bp": last_positive_step,
               "trigger": trigger, "trigger_reason": "正加息步幅小于前次正加息" if trigger else "未满足加息减速",
               "range_evidence": match.group(), "source_path": source.relative_to(ROOT).as_posix(),
               "source_url": f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{stamp}a.htm",
               "market_consensus": None, "actual_minus_consensus": None}
        records.append(row)
        last_upper = upper
        if step is not None and step > 0:
            last_positive_step = step
    assert [r["step_bp"] for r in records[1:]] == [0, 25, 50, 75, 75, 75, 75, 50, 25]
    save(OUT / "fomc_facts_including_baseline.json", records)
    return records[1:]


def treasury_curve(path: Path, real: bool) -> pd.DataFrame:
    rows = []
    keys = {"TC_10YEAR": "real10"} if real else {"BC_2YEAR": "nominal2", "BC_10YEAR": "nominal10"}
    for element in ET.parse(path).getroot().iter():
        if element.tag.rsplit("}", 1)[-1] != "properties":
            continue
        values = {child.tag.rsplit("}", 1)[-1]: child.text for child in element}
        if not values.get("NEW_DATE"):
            continue
        row = {"date": pd.Timestamp(values["NEW_DATE"][:10])}
        row.update({name: float(values[key]) if values.get(key) else np.nan for key, name in keys.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date")


def yield_reactions(events: list[dict]) -> list[dict]:
    nominal = pd.concat([treasury_curve(OUT / "sources" / f"nominal_{y}.xml", False) for y in [2022, 2023]])
    real = pd.concat([treasury_curve(OUT / "sources" / f"tips_{y}.xml", True) for y in [2022, 2023]])
    curve = nominal.merge(real, on="date", how="inner", validate="one_to_one").sort_values("date").reset_index(drop=True)
    curve["inflation_compensation10"] = curve.nominal10 - curve.real10
    assert curve.date.is_unique
    curve.to_parquet(OUT / "treasury_curves.parquet", index=False)
    result = []
    for event in events:
        i = int(np.flatnonzero(curve.date.eq(event["date"]))[0])
        current, prior = curve.iloc[i], curve.iloc[i - 1]
        row = {"event_id": event["event_id"], "date": event["date"], "previous_us_observation_date": prior.date,
               "use": "事后利率反应，不进入首个中国开盘的信号", "change_units": "基点"}
        for key in ["nominal2", "nominal10", "real10", "inflation_compensation10"]:
            row[key + "_pct"] = float(current[key])
            row[key + "_change_bp"] = round(float(current[key] - prior[key]) * 100, 8)
        assert abs(row["nominal10_change_bp"] - row["real10_change_bp"] - row["inflation_compensation10_change_bp"]) < 1e-8
        result.append(row)
    save(OUT / "yield_reactions.json", result)
    return result


def event_clocks(events, market, features):
    opens = pd.DatetimeIndex(market.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    clocks = []
    for event in sorted(events, key=lambda r: r["date"]):
        time = "08:30" if event["kind"] == "CPI" else "14:00"
        public = pd.Timestamp(f"{event['date']} {time}", tz="America/New_York")
        china = public.tz_convert("Asia/Shanghai")
        i = int(np.flatnonzero(opens > china)[0])
        assert np.isfinite(features.es95.iloc[i - 1])
        row = {"event_id": event["event_id"], "kind": event["kind"], "title": event["title"],
               "date": event["date"], "trigger": event["trigger"], "trigger_reason": event["trigger_reason"],
               "public_at": public.isoformat(), "china_public_at": china.isoformat(),
               "entry_date": market.date.iloc[i], "entry_at": opens[i].isoformat(), "entry_idx": i,
               "es_observation_date": market.date.iloc[i - 1], "entry_es95": features.es95.iloc[i - 1],
               "planned_exit_date": market.date.iloc[i + 20],
               "pre_open_gap": (market.open.iloc[i] + market.dividend.iloc[i]) / market.close.iloc[i - 1] - 1,
               "pre_open_gap_role": "前收盘至可成交开盘的参考变化，新入场账户未获得此段"}
        clocks.append(row)
    save(OUT / "external_event_clocks.json", clocks)
    return clocks


def run_accounts(protocol, market, features, dividends, engine, external_clocks, domestic_clocks):
    all_stats, paths, all_checks, decisions, cycles = {}, {}, {}, {}, {}
    ledger_folder = OUT / "ledgers"
    ledger_folder.mkdir(exist_ok=True)
    signals = [row for row in external_clocks if row["trigger"]]
    trigger_sets = {"US_NEWS": signals, "DOMESTIC_US_UNION": signals + domestic_clocks}
    active = market.date.between(*protocol["account_period"])
    for candidate in protocol["new_account_candidates"]:
        clocks = sorted(trigger_sets[candidate], key=lambda r: r["entry_idx"])
        feature_case = features.copy()
        feature_case[SIGNAL] = False
        for clock in clocks:
            feature_case.loc[int(clock["entry_idx"]) - 1, SIGNAL] = True
        for capital in protocol["capitals_cny"]:
            case_id = f"{candidate}_{capital}_STRESS"
            case = {"case_id": case_id, "capital": capital, "policy": SIGNAL, "hold": 20,
                    "start": protocol["account_period"][0], "end": protocol["account_period"][1]}
            ledger = account_model.simulate(market, feature_case, dividends, case, "ORIGINAL", "STRESS", engine)
            assert ledger.date.reset_index(drop=True).equals(market.loc[active, "date"].reset_index(drop=True))
            ledger["reason"] = ledger.reason.replace({"前收盘信号入场": "公开消息满足固定条件后开盘入场"})
            ledger["es_observation_date"] = [market.date.iloc[int(i) - 1] for i in ledger.idx]
            ledger["prior_es95"] = [features.es95.iloc[int(i) - 1] for i in ledger.idx]
            ledger["fee_refund_equity"] = ledger.equity + (ledger.commission + ledger.slippage_cost).cumsum() + ledger.terminal_exit_reserve
            path = ledger_folder / f"{case_id}.parquet"
            ledger.to_parquet(path, index=False)
            saved = pd.read_parquet(path)
            checked = checks.check_account(saved, capital, clocks, True)
            checked["buy_dates_subset_of_frozen_signals"] = checked.pop("buy_dates_subset_of_frozen_six_entries")
            checked["verified_from_saved_ledger"] = True
            stats = core.metrics(saved, capital)
            stats.pop("sharpe_252_diagnostic", None)
            stats.pop("cagr_252_diagnostic", None)
            cycle = core.cycle_records(saved, capital)
            assert cycle.closed.all() and saved.shares.iloc[-1] == 0
            assert abs(cycle.profit.sum() - stats["net_profit"]) < 1e-6
            stats.update({"capital": capital, "candidate": candidate, "raw_trigger_count": len(clocks),
                          "unique_trigger_days": len({r["entry_idx"] for r in clocks}),
                          "entry_fills": int(saved.filled_quantity.gt(0).sum()),
                          "holding_close_days": int(saved.shares.gt(0).sum()),
                          "no_position_all_session_days": int((saved.shares.eq(0) & saved.shares_before.eq(0)).sum()),
                          "max_close_exposure": saved.exposure.max(),
                          "max_close_exposure_times_prior_ES": (saved.exposure * saved.prior_es95).max(),
                          "close_exposure_above_50pct_days": int(saved.exposure.gt(.5 + 1e-12).sum()),
                          "max_drawdown_stop_triggered": bool(saved.risk_stopped.any()),
                          "risk_reduction_fills": int((saved.reason.eq("风险预算减仓") & saved.filled_quantity.lt(0)).sum()),
                          "rejected_nonzero_orders": int((saved.requested_quantity.ne(0) & saved.filled_quantity.eq(0)).sum()),
                          "worst_day": saved.net_return.min(),
                          "turnover_traded_notional_over_initial_capital": saved.notional.sum() / capital,
                          "dividend_recognized": saved.dividend_recognized.sum(),
                          "same_holdings_fee_refund": checks.equity_metrics(saved.fee_refund_equity, capital),
                          "sample_joint_target_met": bool(stats["net_sharpe"] is not None and stats["net_sharpe"] >= 1.2
                              and stats["cagr"] >= .1 and stats["max_drawdown"] <= .1)})
            action_rows = []
            for clock in clocks:
                row = saved.loc[saved.date.eq(clock["entry_date"])].iloc[0]
                action = "入场" if row.filled_quantity > 0 else ("已有持仓，不加仓不延长" if row.shares_before > 0 else "风险或成交约束未入场")
                action_rows.append({"event_id": clock["event_id"], "entry_date": row.date,
                                    "shares_before": row.shares_before, "filled_quantity": row.filled_quantity,
                                    "action": action, "status": row.status, "reason": row.reason})
            all_stats[case_id], all_checks[case_id] = stats, checked
            paths[case_id] = path.relative_to(ROOT).as_posix()
            decisions[case_id], cycles[case_id] = action_rows, cycle.to_dict("records")
            print(f"完成账户：{case_id}，净利润{stats['net_profit']:.2f}元，夏普{stats['net_sharpe']:.6f}，年化{stats['cagr']:.4%}", flush=True)
    save(OUT / "account_checks.json", all_checks)
    save(OUT / "account_decisions.json", decisions)
    save(OUT / "account_cycles.json", cycles)
    return all_stats, paths


def run() -> None:
    protocol = read(OUT / "protocol.json")
    cpi, fomc = cpi_facts(protocol), fomc_facts(protocol)
    domestic_protocol = read(PRIOR / "protocol.json")
    market, features, dividends, engine, china_clocks = domestic.inputs(domestic_protocol)
    assert protocol["account_period"] == domestic_protocol["account_calendar"]
    clocks = event_clocks(cpi + fomc, market, features)
    assert len(clocks) == 23 and sum(c["trigger"] for c in clocks) == 7
    reactions = yield_reactions(cpi + fomc)
    windows = [domestic.fixed_event_window(market, dividends, engine, c, horizon) for c in clocks for horizon in [5, 20]]
    save(OUT / "event_returns.json", windows)
    stats, paths = run_accounts(protocol, market, features, dividends, engine, clocks, china_clocks)
    previous = read(PRIOR / "result.json")
    result = {"study_id": protocol["study_id"], "completed_at": datetime.now(TZ).isoformat(),
              "mode": "HISTORICAL_ONLY", "research_unit": "指数整体共同定价",
              "source_counts": {"cpi_original_releases": len(cpi), "fomc_decisions": len(fomc),
                                "fomc_baseline": 1, "sep": 2, "speech": 1, "treasury_year_series": 4},
              "external_event_count": len(clocks), "simple_trigger_count": 7,
              "event_window_rows": len(windows), "new_full_accounts": len(stats),
              "accounts": stats, "ledger_paths": paths,
              "domestic_baseline_result_path": (PRIOR / "result.json").relative_to(ROOT).as_posix(),
              "domestic_baseline_accounts": previous["accounts"],
              "expectation_surprises": "NOT_COMPUTED_NO_CONTEMPORANEOUS_CONSENSUS",
              "causal_contribution_identified": False, "independent_validation": False,
              "new_parameters_fitted": 0, "future_forecasts": False, "orders_authorized": False,
              "goal_achieved": False,
              "candidate_disposition": {candidate: "REJECTED_FROZEN" if not stats[f"{candidate}_200000_STRESS"]["sample_joint_target_met"] else "SAMPLE_ONLY_REQUIRES_OTHER_MANDATE_CONDITIONS"
                                        for candidate in protocol["new_account_candidates"]}}
    save(OUT / "result.json", result)
    print(f"完成：{len(clocks)}个公告，{len(windows)}个固定窗口，{len(stats)}个连续账户。", flush=True)


def enrich_saved_evidence() -> None:
    result = read(OUT / "result.json")
    cpi = read(OUT / "cpi_facts.json")
    events = read(OUT / "event_returns.json")
    clocks = read(OUT / "external_event_clocks.json")
    cycles = read(OUT / "account_cycles.json")
    cycle_details = {}
    for case_id, records in cycles.items():
        ledger = pd.read_parquet(ROOT / result["ledger_paths"][case_id])
        details = []
        for row in records:
            part = ledger.loc[ledger.date.between(pd.Timestamp(row["entry"]), pd.Timestamp(row["exit"]))]
            friction = float((part.commission + part.slippage_cost).sum())
            details.append({**row, "paid_friction_cny": friction,
                            "same_holdings_gross_profit_cny": row["profit"] + friction,
                            "mean_close_exposure": float(part.exposure.mean()),
                            "risk_reduction_fills": int((part.reason.eq("风险预算减仓") & part.filled_quantity.lt(0)).sum()),
                            "open_intervals": int(part.idx.iloc[-1] - part.idx.iloc[0])})
        cycle_details[case_id] = details
    save(OUT / "cycle_cost_and_exposure.json", cycle_details)
    windows20 = {r["event_id"]: r for r in events if r["horizon"] == 20}
    ids = [r["event_id"] for r in clocks if r["trigger"]]
    chosen = [windows20[key] for key in ids]
    result["trigger_window_diagnostic"] = {
        "count_including_overlap": len(chosen),
        "negative_count_including_overlap": sum(r["net_return"] < 0 for r in chosen),
        "independent_observation_count": None,
        "note": "12月14日与15日起算的窗口重叠，不能当作两份独立获利证据；完整账户只执行其中一笔。",
    }
    result["primary_us_news_closed_cycles"] = len(cycles["US_NEWS_200000_STRESS"])
    result["primary_us_news_losing_cycles"] = sum(r["profit"] < 0 for r in cycles["US_NEWS_200000_STRESS"])
    result["goal_turn_classification"] = "PROGRESS"
    result["classification"] = "PROGRESS_INDEX_INFLATION_COMPOSITION_POLICY_PATH_AND_ACCOUNT_INTERFERENCE"
    result["next_historical_question"] = (
        "从原23个固定美国公告入手，先检查已有数据是否包含当时的市场一致预期或议息前利率定价；"
        "把公布值较前月变化与真正的预期差分开，并与国内政策同窗信息逐一对时。"
        "找不到原始预期锚的窗口保留缺失；不按本次收益添加月份、利率反应或持有期过滤，不重救两个失败候选。"
    )
    result["discovery"] = [
        "核心通胀放缓的构成不同：春季能源与服务上涨、夏季短暂降温、年底商品回落及住房滞后不能合为同一种宽松。",
        "11月10日美国10年名义利率下降30基点，其中实质收益率下降27基点、通胀补偿下降3基点；国内政策方向同晚也已公开，不能识别中国指数收益的独立来源。",
        "12月加息步幅下降，同时2023年末政策利率中位预测提高；速度、水平及持续时间应分开。",
        "两项冻结候选全账户亏损，退还既定持仓成本后仍亏损；简单叠加消息还会改变已有国内政策持仓的退出时间。",
    ]
    result["report"] = (OUT / "历史发现_通胀构成与指数共同定价.md").relative_to(ROOT).as_posix()
    save(OUT / "result.json", result)
    cards = {
        "status": "HISTORICAL_EXPLANATION_AFTER_RESULTS_NO_NEW_FILTERS",
        "sep_2023_medians": {
            "september_2022": {"year_end_funds_rate_pct": 4.6, "core_pce_pct": 3.1, "real_gdp_growth_pct": 1.2, "unemployment_pct": 4.4},
            "december_2022": {"year_end_funds_rate_pct": 5.1, "core_pce_pct": 3.5, "real_gdp_growth_pct": .5, "unemployment_pct": 4.6},
            "source": "sources/sep_20221214.pdf", "page": 2,
            "source_url": "https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20221214.pdf",
            "meaning": "2023年末适当政策利率预测中位数，不是承诺、市场共识或精确的终点利率。",
            "visually_reviewed": True,
        },
        "powell_nov30_upstream_explanation": {
            "source": "sources/powell_20221130.html",
            "source_url": "https://www.federalreserve.gov/newsevents/speech/powell20221130a.htm",
            "goods": "疫情期商品需求与受损供给造成价格压力；当时供应链压力及进口、制造商支付价格已经缓解。",
            "housing": "新租约租金减速较快，存量租约缓慢更新使整体住房价格存在传导滞后。",
            "nonhousing_services": "工资是重要成本，劳动力供求紧张被视为服务通胀持续性的约束。",
            "policy": "讲话已提及最快12月减慢加息，并强调进一步加息幅度及维持限制的时长。",
            "scope": "讲话主要讨论PCE；本研究CPI分项不是PCE的同口径复算，也未定量归因供应链或工资贡献。",
        },
        "nov10_information_clock": [
            {"china_public_at": "2022-11-10T19:42:00+08:00", "event": "国内政策优化方向",
             "source_url": "https://www.mct.gov.cn/whzx/szyw/202211/t20221110_937380.htm"},
            {"china_public_at": "2022-11-10T21:30:00+08:00", "event": "美国10月CPI首次发布",
             "source_url": "https://www.bls.gov/news.release/archives/cpi_11102022.htm"},
            {"china_public_at": "2022-11-11T09:30:00+08:00", "event": "中国首个可成交开盘"},
        ],
        "rate_decomposition_limits": [
            "10年名义收益率=10年TIPS实质收益率+两者差额，仅为观测口径的恒等式。",
            "TIPS实质收益率仍含实质期限溢价和市场流动性影响，不能视为纯粹预期实质短端利率。",
            "差额称通胀补偿，含通胀风险溢价、相对流动性和计量差异；不是纯预期通胀。",
            "没有单独识别违约补偿、纯期限溢价或其他同时发生新闻的份额。",
        ],
        "seasonal_revision_example": {
            "economic_month": "2022-12", "first_published_on": "2023-01-12", "core_mom_first": .3,
            "shown_as_previous_on": "2023-02-14", "core_mom_revised": .4,
            "method": "每次均使用同一份当时公告中的前月及本月数值，避免跨版本拼接。",
        },
    }
    assert cpi[-1]["previous_core_differs_from_prior_release"]
    save(OUT / "mechanism_cards.json", cards)
    manifest = read(OUT / "source_manifest.json")
    if not any(row["id"] == "cpi_nov10_dol_pdf" for row in manifest):
        receipt = read(OUT / "sources/cpi_11102022_dol.receipt.json")
        path = OUT / "sources/cpi_11102022_dol.pdf"
        manifest.append({"id": "cpi_nov10_dol_pdf", **receipt, "status": "FETCHED_PRIMARY_DOL_MIRROR",
                         "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                         "table_a_page_2_visually_reviewed": True})
        save(OUT / "source_manifest.json", manifest)


def make_plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    cpi = read(OUT / "cpi_facts.json")
    yields = sorted(read(OUT / "yield_reactions.json"), key=lambda r: r["date"])
    result = read(OUT / "result.json")
    prior = read(PRIOR / "result.json")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 1, 1.2]})
    fig.suptitle("510300历史机制发现：消息的构成、利率反应与实际账户", x=.07, ha="left", fontsize=17)
    dates = pd.to_datetime([r["date"] for r in cpi])
    for key, label, color, style in [
        ("core", "核心CPI", "#263746", "-"), ("core_goods", "核心商品", "#218c8d", "-"),
        ("core_services", "核心服务（含住房）", "#cd8247", "-"), ("shelter", "住房", "#8b779b", "--"),
    ]:
        axes[0].plot(dates, [r["components"][key]["current_mom"] for r in cpi], style, color=color, marker="o", markersize=4, label=label)
    axes[0].axhline(0, color="#aab2b9", lw=.8)
    axes[0].set_title("同为核心通胀放慢，内部构成不同", loc="left", fontsize=13)
    axes[0].set_ylabel("季调月环比（%）")
    axes[0].set_xticks(dates, [d.strftime("%y-%m-%d") for d in dates], rotation=30, ha="right", fontsize=8)
    axes[0].set_xlabel("公告日期；每次使用当时版本，分项不可直接相加")
    axes[0].legend(frameon=False, ncol=4, fontsize=9, loc="upper right")
    x, width = np.arange(len(yields)), .34
    axes[1].bar(x - width / 2, [r["real10_change_bp"] for r in yields], width, color="#218c8d", label="10年TIPS实质收益率变化")
    axes[1].bar(x + width / 2, [r["inflation_compensation10_change_bp"] for r in yields], width, color="#cd8247", label="10年通胀补偿变化")
    axes[1].axhline(0, color="#aab2b9", lw=.8)
    axes[1].set_title("全部23个公告的美国利率同日反应：仅用于事后解释", loc="left", fontsize=13)
    axes[1].set_ylabel("相对前次观测（基点）")
    axes[1].set_xticks(x, [r["date"][2:] + (" C" if r["event_id"].startswith("CPI") else " F") for r in yields], rotation=60, ha="right", fontsize=8)
    axes[1].set_xlabel("C=CPI；F=议息。通胀补偿含风险及流动性影响，TIPS亦非纯预期实质利率。")
    axes[1].legend(frameon=False, ncol=2, fontsize=9, loc="lower left")
    for key, label, color, source in [
        ("US_NEWS_200000_STRESS", "外部消息规则：夏普−0.528", "#218c8d", result),
        ("DOMESTIC_US_UNION_200000_STRESS", "与国内政策取并集：夏普−0.857", "#bc6260", result),
        ("POLICY_200000_STRESS", "原国内政策规则：夏普0.566（旧结果）", "#718397", prior),
    ]:
        ledger = pd.read_parquet(ROOT / source["ledger_paths"][key])
        axes[2].plot(ledger.date, ledger.equity / 200000 - 1, color=color, lw=1.8, label=label)
    axes[2].axhline(0, color="#aab2b9", lw=.8)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_title("同一301交易日、20万元完整账户、压力成本及原风险约束", loc="left", fontsize=13)
    axes[2].set_ylabel("累计账户收益")
    axes[2].legend(frameon=False, fontsize=9, loc="lower left")
    axes[2].set_xlabel("历史日历，含全部空仓日；不是独立验证，不表示未来表现")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    fig.tight_layout(rect=[.015, .015, .995, .965], h_pad=1.8)
    fig.savefig(OUT / "通胀构成_利率反应_账户结果.png", dpi=160, facecolor="white")
    plt.close(fig)


def make_report() -> None:
    r = read(OUT / "result.json")
    cpi = read(OUT / "cpi_facts.json")
    fomc = read(OUT / "fomc_facts_including_baseline.json")[1:]
    clocks = read(OUT / "external_event_clocks.json")
    windows = {(x["event_id"], x["horizon"]): x for x in read(OUT / "event_returns.json")}
    curve = {x["event_id"]: x for x in read(OUT / "yield_reactions.json")}
    cycles = read(OUT / "cycle_cost_and_exposure.json")
    a, b = r["accounts"]["US_NEWS_200000_STRESS"], r["accounts"]["DOMESTIC_US_UNION_200000_STRESS"]
    def file_link(label, filename):
        return f"[{label}](<{(OUT / filename).as_posix()}>)"
    text = [
        "# 历史发现：通胀构成、政策路径与510300共同定价",
        "",
        f"本轮未找到达标策略。外部消息规则20万元完整账户净夏普{a['net_sharpe']:.3f}、年化{a['cagr']:.2%}；"
        f"与原国内政策信号合并后为{b['net_sharpe']:.3f}、{b['cagr']:.2%}。两项候选均固定拒绝，不根据亏损月份、利率反应或持有期反复修补。",
        "",
        "研究对象为指数整体：国内经济活动约束、全球贴现及风险补偿、公开消息到可成交价格的时差。范围为2022年1月至2023年2月全部14次CPI和9次FOMC，账户覆盖2022年1月1日至2023年3月31日。只做历史发现，不扩展个股。",
        "",
        "## 得到的机制区分",
        "",
        "1. **核心通胀放慢，需要解释是哪一部分变化。** 2022年4月12日公告里，核心环比由0.5%降至0.3%，但核心服务由0.5%升至0.6%，汽油环比18.3%、整体CPI环比1.2%。核心商品回落并没有消除能源及服务压力。8月10日核心环比0.7%降至0.3%，9月13日又回到0.6%。这些是不同构成和不同阶段，不是一个稳定的‘通胀下降’状态。[4月原公告](https://www.bls.gov/news.release/archives/cpi_04122022.htm)、[8月原公告](https://www.bls.gov/news.release/archives/cpi_08102022.htm)、[9月原公告](https://www.bls.gov/news.release/archives/cpi_09132022.htm)",
        "",
        "2. **上游原因及传导速度不同。** 11月30日鲍威尔解释，当时商品供应链、进口及厂商支付价格压力缓解；新租约降温向存量租约传递较慢；非住房服务仍受劳动力紧张和工资成本约束。这支持区分商品供给修复、住房滞后和服务成本，不能定量证明各自造成多少指数收益。讲话主要讨论PCE，本研究CPI分项不与PCE混作同一口径。[当时讲话](https://www.federalreserve.gov/newsevents/speech/powell20221130a.htm)",
        "",
        "3. **加息速度变慢，仍可伴随更高的利率路径。** 12月14日步幅75降至50基点，同时SEP中2023年末利率中位预测由9月4.6%提高至5.1%，2023年核心PCE预测3.1%升至3.5%、GDP增长1.2%降至0.5%。应分别看速度、水平与维持时间；2023年末预测不是承诺的精确加息终点。11月30日讲话已经提及最快12月减速，故12月步幅变化不能直接当成未预期的宽松。[12月声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20221214a.htm)、[12月SEP表1](https://www.federalreserve.gov/monetarypolicy/files/fomcprojtabl20221214.pdf)",
        "",
        "4. **同向的名义美债变化，内部可能相反。** 下表用同一观测日名义10年收益率减TIPS10年实质收益率得到通胀补偿。名义=实质+补偿是观测恒等式，不是结构因果识别。TIPS仍含实质期限溢价及流动性影响；通胀补偿也含风险溢价、相对流动性等，不能叫纯通胀预期。本轮没有把违约补偿单独估出来。",
        "",
        "|美国公告日|名义10年变化|TIPS10年变化|通胀补偿变化|首个可成交开盘后20日净收益|",
        "|---|---:|---:|---:|---:|",
    ]
    for event_id in ["CPI_20220310", "CPI_20220412", "CPI_20220810", "CPI_20221110", "FOMC_20221214", "FOMC_20230201"]:
        y, w = curve[event_id], windows[(event_id, 20)]
        text.append(f"|{y['date']} {'CPI' if event_id.startswith('CPI') else 'FOMC'}|{y['nominal10_change_bp']:+.0f}bp|{y['real10_change_bp']:+.0f}bp|{y['inflation_compensation10_change_bp']:+.0f}bp|{w['net_return']:+.2%}|")
    text += [
        "",
        "全部23个公告的利率结果均已保存。这里选取构成不同的个案作解释，不用于生成筛选规则。事件收益是10万元名义买入规模的独立固定窗口，含压力成本及分红；不是受仓位约束的全账户收益，重叠窗口不能累计。美国曲线缺少可靠的历史发布时点，本轮只用于事后反应，不进入中国首个开盘的信号。[美国财政部名义曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=2022)、[实质曲线](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_real_yield_curve&field_tdr_date_value=2022)",
        "",
        "5. **11月11日中国开盘前，是多条消息共同进入价格。** 11月10日19:42国内政策优化方向已公开；21:30美国CPI发布。随后美债名义10年下降30bp，其中TIPS下降27bp、通胀补偿下降3bp。510300次日开盘相对前收盘已高2.45%，这段不属于消息公开后开盘入场的收益。其后20日成本后窗口仍涨4.39%，但不能全部归为美国CPI或二十条细则的独立贡献。[国内19:42原文](https://www.mct.gov.cn/whzx/szyw/202211/t20221110_937380.htm)、[CPI原公告](https://www.bls.gov/news.release/archives/cpi_11102022.htm)",
        "",
        "## 固定规则与连续账户结果",
        "",
        "外部消息规则只有一个条件：当次公告里的核心CPI环比严格小于该公告所列前月，或正加息步幅严格小于前次正加息。共5次CPI及2次议息触发。CPI纽约08:30、FOMC纽约14:00按夏令时换算，随后首个中国开盘进入；持有20个开盘区间。合并规则再加入原四次国内政策触发，同日去重，持仓时不加仓、不延长。",
        "",
        "两个方案均先记录后计算，不搜索阈值。相同301个交易日包含所有空仓日；年化交易日242，现金收益及夏普无风险基准均为0。保留50%最大目标仓位、5日条件ES95预算2.5%、−10%跳空预算5%、10%回撤触发、T+1、整手及分红。压力成本为佣金万4且最低5元、单边滑点0.1%。风险约束使实际持仓通常低于50%。",
        "",
        "|规则／账户|净夏普|净年化|最大回撤|净利润|入场周期数|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [("US_NEWS_200000_STRESS", "外部消息／20万"), ("DOMESTIC_US_UNION_200000_STRESS", "国内外合并／20万"),
                       ("US_NEWS_20000_STRESS", "外部消息／2万"), ("DOMESTIC_US_UNION_20000_STRESS", "国内外合并／2万")]:
        q = r["accounts"][key]
        text.append(f"|{label}|{q['net_sharpe']:.3f}|{q['cagr']:.2%}|{q['max_drawdown']:.2%}|{q['net_profit']:+,.2f}元|{q['entry_fills']}|")
    q = r["domestic_baseline_accounts"]["POLICY_200000_STRESS"]
    text.append(f"|原国内政策／20万，复用旧结果|{q['net_sharpe']:.3f}|{q['cagr']:.2%}|{q['max_drawdown']:.2%}|{q['net_profit']:+,.2f}元|{q['entry_fills']}|")
    text += [
        "",
        "原国内政策规则本身也未达到夏普1.2且年化10%的联合目标，不能因为另外两个结果更差就升级为有效策略。",
        "",
        "**亏损来源可以明确分开。** 外部消息规则实际6个周期中4个亏损，4个对应的独立20日事件窗口在成本前也为负。完整账户手续费与滑点合计1,338.92元，退还后仍亏4,247.70元、夏普−0.395；合并规则退还1,361.54元后仍亏7,238.70元、夏普−0.716。因此不足首先体现在既定入场和持有区间的毛收益，费用是额外损耗。此退款只保持原仓位路径不变，不是一个免成本重新选仓的可执行策略。",
        "",
        "20万元外部账户平均收盘仓位12.72%，17次风险减仓；合并账户12.86%，22次风险减仓。没有非零订单被拒，没有触发10%回撤停机。风险预算确实改变仓位，但既定事件窗口本身已有亏损，不能简单把失败归因于仓位过低。未通过提高仓位或移除风控来挽救候选。",
        "",
        "**直接合并信号会改变交易过程。** 12月13日CPI使合并账户12月14日进入，随后12月15日议息及12月27日国内政策触发均因已有持仓被忽略，固定退出为2023年1月12日。原国内政策账户则12月27日进入、2月1日退出，并取得1月分红。两者是不同持有区间，不能把各自最好的一段拼起来。11月的更早入场也把退出从12月12日前移到12月9日。",
        "",
        "|外部消息20万元实际周期|退出|净利润|原仓位退还成本后利润|",
        "|---|---|---:|---:|",
    ]
    for q in cycles["US_NEWS_200000_STRESS"]:
        text.append(f"|{q['entry'][:10]}|{q['exit'][:10]}|{q['profit']:+,.2f}元|{q['same_holdings_gross_profit_cny']:+,.2f}元|")
    text += [
        "",
        f"![通胀构成、利率反应与连续账户](<{(OUT / '通胀构成_利率反应_账户结果.png').as_posix()}>)",
        "",
        "## 全部公告，不删反例",
        "",
        "下表的缺口是前收盘至消息后首个开盘的价格变化；新入场未获得它。5／20日窗口均从该开盘开始，含成本。它们不是对美国消息的纯因果估计。5日与20日分别表示不同长度的已实现路径，不据此挑出更优持有期。",
        "",
        "|公告（美国日期）|简单条件触发|中国入场日|已发生开盘缺口|后5日净收益|后20日净收益|",
        "|---|---|---|---:|---:|---:|",
    ]
    for q in clocks:
        text.append(f"|{q['date']} {q['kind']}|{'是' if q['trigger'] else '否'}|{q['entry_date'][:10]}|{q['pre_open_gap']:+.2%}|{windows[(q['event_id'], 5)]['net_return']:+.2%}|{windows[(q['event_id'], 20)]['net_return']:+.2%}|")
    text += [
        "",
        "还要保留相反例子：2022年6月10日CPI整体环比1.0%、2年美债当日升23bp，随后510300固定20日净收益却为+5.28%；11月2日仍加息75bp，随后窗口为+6.77%。这些并不推出‘通胀高／加息多就买入’，而是说明中国指数当时还有国内活动及政策等共同驱动，单一外部标签不足以决定方向。",
        "",
        "## 原始版本、预期差与使用边界",
        "",
        "14次CPI使用BLS当次历史公告Table A；部分原生PDF请求403，保留失败记录，数值从官方历史网页的工具提取文本读取。11月PDF另有美国劳工部镜像，Table A与12月SEP表1已渲染查看。没有把现在的下载时间误认为历史公开时间。",
        "",
        "2月季调版本变动单列：2022年12月核心环比在2023年1月12日首次报0.3%，到2月14日公告列前月时变为0.4%。本轮总是取同一份当时公告的前月与本月比较，未用最终修订序列回填历史信号。[1月原公告](https://www.bls.gov/news.release/archives/cpi_01122023.htm)、[2月原公告](https://www.bls.gov/news.release/archives/cpi_02142023.htm)",
        "",
        "本轮没有可信的逐公告市场一致预期，actual-minus-consensus仍为空。因此得到的是‘公布事实变化及市场随后反应’，尚未完成真正的预期差测量。名义利率反应或事后股价不能反过来冒充当时共识。部分2022年公告与价格已在旧研究出现，本轮不是独立验证；已见收益不用于新增筛选。",
        "",
        "仅保留必要核对：当时公告版本与时钟、原始表格关键数值、保存账本的资金恒等式、T+1及指标重算。没有增加多年扫描、参数搜索或繁琐的准入程序。",
        "",
        r["next_historical_question"],
        "",
        "可复查文件：" + "；".join([
            file_link("结果", "result.json"), file_link("固定方案", "protocol.json"),
            file_link("通胀分项与原文行", "cpi_facts.json"), file_link("议息数值", "fomc_facts_including_baseline.json"),
            file_link("利率分解", "yield_reactions.json"), file_link("全部事件窗口", "event_returns.json"),
            file_link("逐次账户决定", "account_decisions.json"), file_link("周期成本与仓位", "cycle_cost_and_exposure.json"),
            file_link("原文清单", "source_manifest.json"), file_link("必要账本核对", "account_checks.json"),
        ]) + "。",
        "",
        f"重算脚本：[{(ROOT / 'research/historical_index_external_reopening_v1.py').name}](<{(ROOT / 'research/historical_index_external_reopening_v1.py').as_posix()}>). 先运行 `--run` 计算固定结果，随后 `--report` 整理报告与图表；已有数据时不需再采集。",
        "",
        "目标仍未实现；没有实盘或自动下单授权。",
        "",
    ]
    (OUT / "历史发现_通胀构成与指数共同定价.md").write_text("\n".join(text), encoding="utf-8")


def report() -> None:
    enrich_saved_evidence()
    make_plot()
    make_report()
    print("已完成机制卡、账户分解、图表与历史研究报告。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="取得固定公告集合，优先复用原文")
    parser.add_argument("--run", action="store_true", help="按固定方案解析事实并计算历史账户")
    parser.add_argument("--report", action="store_true", help="根据保存结果整理机制解释、图表和报告")
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.run:
        run()
    elif args.report:
        report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
