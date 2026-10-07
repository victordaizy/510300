"""中国PMI历史预期偏差、构成变化与510300固定账户研究。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urljoin, urlparse
from zoneinfo import ZoneInfo

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

OUT = ROOT / "reports/research/510300_historical_index_pmi_expectation_v1"
ANCHOR = ROOT / "reports/research/510300_historical_index_expectation_anchor_v1"
CONTEXT = ROOT / "reports/research/510300_historical_index_domestic_constraint_clock_v1"
TZ = ZoneInfo("Asia/Shanghai")
RULES = {
    "PMI_POSITIVE_SURPRISE": "pmi_positive_surprise_before_open",
    "PMI_SURPRISE_DEMAND_IMPROVEMENT": "pmi_surprise_demand_improvement_before_open",
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value) -> None:
    core.save(path, value)


def now() -> str:
    return datetime.now(TZ).isoformat()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> list[dict]:
    protocol = read(OUT / "protocol.json")
    if "recorded_at" not in protocol:
        protocol["recorded_at"] = now()
        save(OUT / "protocol.json", protocol)
    archive = []
    for year in [2022, 2023]:
        path = ANCHOR / "sources" / f"archive_{year}.html"
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for link in soup.find_all("a", href=True):
            if "byshoweventarticle" not in link["href"]:
                continue
            row = link.find_parent("tr").get_text(" ", strip=True)
            date = re.search(r"(\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))$", row).group()
            day = pd.to_datetime(date, format="%m/%d/%Y" if len(date.split("/")[-1]) == 4 else "%m/%d/%y")
            fid = parse_qs(urlparse(link["href"]).query)["fid"][0]
            archive.append({"id": "econoday_" + fid, "fid": fid, "archive_date": day.strftime("%Y-%m-%d"), "url": urljoin("https://fidelity.econoday.com/", link["href"]), "title": link.get_text(" ", strip=True), "archive_path": rel(path)})
    source_by_id, mapping = {}, []
    actual = [x for x in read(CONTEXT / "monthly_pmi.json") if x["stat_month"] in protocol["months"]]
    assert len(actual) == 14
    for item in actual:
        public_date = item["published_at"][:10]
        candidate = max((a for a in archive if a["archive_date"] < public_date), key=lambda a: a["archive_date"])
        source_by_id[candidate["id"]] = candidate
        mapping.append({"event_id": "PMI_" + item["stat_month"].replace("-", ""), "stat_month": item["stat_month"], "actual_public_at": item["published_at"], "actual_known_at": item["known_at"], "source_id": candidate["id"], "archive_date": candidate["archive_date"]})
    save(OUT / "source_plan.json", list(source_by_id.values()))
    save(OUT / "event_source_map.json", mapping)
    return list(source_by_id.values())


def collect() -> None:
    sources = prepare()
    previous = {x["id"]: x for x in read(ANCHOR / "source_manifest.json") if x["status"] == "FETCHED"}
    existing = {x["id"]: x for x in read(OUT / "source_manifest.json")} if (OUT / "source_manifest.json").exists() else {}
    (OUT / "sources").mkdir(exist_ok=True)

    def one(item):
        if item["id"] in existing and existing[item["id"]]["status"] in ["FETCHED", "REUSED_NATIVE_SOURCE"]:
            old = existing[item["id"]]
            assert digest(ROOT / old["path"]) == old["sha256"]
            return old
        record = {**item, "recorded_at": now()}
        if item["id"] in previous:
            old = previous[item["id"]]
            record.update(status="REUSED_NATIVE_SOURCE", path=old["path"], sha256=digest(ROOT / old["path"]), original_retrieved_at=old["retrieved_at"])
            return record
        try:
            response = requests.get(item["url"], timeout=25)
            record.update(http_status=response.status_code, resolved_url=response.url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for tag in soup.select("script,style"):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            assert "Global Economics" in text, "响应没有周报正文"
            path = OUT / "sources" / (item["id"] + ".html")
            path.write_bytes(response.content)
            path.with_suffix(".txt").write_text(text, encoding="utf-8")
            record.update(status="FETCHED", path=rel(path), sha256=digest(path), retrieved_at=now())
        except (requests.RequestException, AssertionError) as exc:
            record.update(status="UNAVAILABLE", error=str(exc))
        return record

    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(one, item) for item in sources]):
            row = future.result()
            records.append(row)
            save(OUT / "source_manifest.json", sorted(records, key=lambda r: r["archive_date"]))
            print(f"周报 {row['id']}，日期 {row['archive_date']}，状态 {row['status']}", flush=True)


def previous_row(item: dict) -> dict:
    prior_month = pd.Period(item["stat_month"], freq="M") - 1
    label = f"{prior_month.year}年{prior_month.month}月"
    path = ROOT / item["raw_path"]
    assert digest(path) == item["raw_sha256"]
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    values = []
    for table in soup.find_all("table"):
        rows = [[re.sub(r"\s+", "", cell.get_text()) for cell in tr.find_all(["td", "th"])] for tr in table.find_all("tr")]
        heading = {cell for row in rows[:4] for cell in row}
        if not {"PMI", "生产", "供应商配送时间"}.issubset(heading):
            continue
        for row in rows:
            if row and row[0] == label:
                assert len(row) == 7
                values.append(tuple(float(x) for x in row[1:]))
    assert len(set(values)) == 1, (item["stat_month"], values)
    return dict(zip(["manufacturing_pmi", "production", "new_orders", "raw_inventory", "employment", "supplier_delivery"], values[0]))


def parse_facts() -> list[dict]:
    actual = {x["stat_month"]: x for x in read(CONTEXT / "monthly_pmi.json")}
    manifest = {x["id"]: x for x in read(OUT / "source_manifest.json")}
    result = []
    for mapping in read(OUT / "event_source_map.json"):
        current = actual[mapping["stat_month"]]
        prior_month = str(pd.Period(mapping["stat_month"], freq="M") - 1)
        prior_known = actual[prior_month]
        prior = previous_row(current)
        source = manifest[mapping["source_id"]]
        row = {**mapping, "actual": current["manufacturing_pmi"], "actual_nonmanufacturing": current["nonmanufacturing_activity"], "actual_services": current["services_activity"], "actual_url": current["url"], "actual_path": current["raw_path"], "prior_published_pmi": prior_known["manufacturing_pmi"], "prior_same_release_pmi": prior["manufacturing_pmi"], "prior_headline_revision": prior["manufacturing_pmi"] - prior_known["manufacturing_pmi"], "source_url": source["url"], "source_path": source.get("path"), "historical_first_vintage_verified": False, "expected": None, "expected_nonmanufacturing": None, "surprise": None, "forecast_status": "SOURCE_UNAVAILABLE", "trigger_base": False, "trigger_demand": False}
        publication = pd.Timestamp(current["published_at"])
        precision = "DAY" if len(current["published_at"]) == 10 else "MINUTE_OR_SECOND"
        if publication.tzinfo is None:
            publication = publication.tz_localize(TZ)
        row.update(actual_public_at=None if precision == "DAY" else publication.isoformat(), actual_public_date=current["published_at"][:10], actual_public_precision=precision, actual_public_lower_bound=publication.isoformat(), actual_public_upper_bound=current["known_at"] if precision == "DAY" else publication.isoformat(), actual_stored_publication_value=current["published_at"])
        weights = {"new_orders": 0.30, "production": 0.25, "employment": 0.20, "raw_inventory": 0.10, "supplier_delivery": -0.15}
        changes = {key: round(current[key] - prior[key], 10) for key in weights}
        contributions = {key: round(weights[key] * changes[key], 10) for key in weights}
        row.update(component_current={key: current[key] for key in weights}, component_previous_same_release=prior, component_changes=changes, component_contributions=contributions, orders_and_production_contribution=round(contributions["new_orders"] + contributions["production"], 10), observed_change_from_prior_published=round(current["manufacturing_pmi"] - prior_known["manufacturing_pmi"], 10), headline_same_release_change=round(current["manufacturing_pmi"] - prior["manufacturing_pmi"], 10), component_rounding_residual=round(current["manufacturing_pmi"] - prior["manufacturing_pmi"] - sum(contributions.values()), 10))
        if source["status"] not in ["FETCHED", "REUSED_NATIVE_SOURCE"]:
            result.append(row)
            continue
        soup = BeautifulSoup((ROOT / source["path"]).read_bytes(), "html.parser")
        for tag in soup.select("script,style"):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        header_match = re.search(r"Global Economics\s*[-–]\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", text)
        assert header_match, source["id"]
        body_day = pd.to_datetime(header_match[1], format="%B %d, %Y")
        title = soup.title.get_text(" ", strip=True)
        title_match = re.search(r"Global Economics\s+(\d+)\s+(\d+),\s*(\d{4})", title)
        assert title_match, (source["id"], title)
        title_day = pd.Timestamp(year=int(title_match[3]), month=int(title_match[1]), day=int(title_match[2]))
        bound_day = max(body_day, pd.Timestamp(source["archive_date"]))
        bound = bound_day.tz_localize("America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        strict_bound = max(bound_day, title_day).tz_localize("America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        row.update(body_article_date=body_day.date().isoformat(), page_title_date=title_day.date().isoformat(), source_public_upper_bound=bound.isoformat(), source_upper_bound_before_release=bool(bound < publication), three_date_upper_bound=strict_bound.isoformat(), three_date_upper_bound_before_release=bool(strict_bound < publication), source_date_to_release_hours=(publication - bound).total_seconds() / 3600)
        month_name = pd.Timestamp(mapping["stat_month"] + "-01").strftime("%B")
        heading = re.search(r"China:\s*CFLP\s+(?:Manufacturing PMI|PMIs)\s+for\s+" + month_name + r"\s*\(([^)]*)\)", text, flags=re.I)
        if not heading:
            row["forecast_status"] = "CONSENSUS_NOT_IN_SELECTED_REPORT"
            result.append(row)
            continue
        remainder = text[heading.end():heading.end() + 500]
        expected_match = re.match(r"\s*(?:Manufacturing PMI,\s*)?Consensus Forecast\s*:\s*([\d.]+)", remainder, flags=re.I)
        assert expected_match, (mapping["event_id"], remainder[:180])
        expected = float(expected_match[1].rstrip("."))
        nonmanufacturing_match = re.search(r"Non-manufacturing PMI,\s*Consensus Forecast\s*:\s*([\d.]+)", remainder, flags=re.I)
        nonmanufacturing = float(nonmanufacturing_match[1].rstrip(".")) if nonmanufacturing_match else None
        row.update(expected=expected, expected_nonmanufacturing=nonmanufacturing, surprise=round(current["manufacturing_pmi"] - expected, 10), expected_change_from_prior_published=round(expected - prior_known["manufacturing_pmi"], 10), nonmanufacturing_surprise=None if nonmanufacturing is None else round(current["nonmanufacturing_activity"] - nonmanufacturing, 10), source_schedule_text=heading[1], source_forecast_heading_position=heading.start(), provider="Econoday Global Economics周报；原站共识来源说明为Econoday及MNI", evidence_excerpt=f"Manufacturing PMI, Consensus Forecast: {expected:.1f}", measurement="周报共识偏差，非最后一刻共识；组成项仅有月度变化，没有组成项市场预期。")
        row["forecast_status"] = "ELIGIBLE" if bound < publication else "TIMING_UNRESOLVED"
        row["trigger_base"] = row["forecast_status"] == "ELIGIBLE" and row["surprise"] > 0
        row["trigger_demand"] = row["trigger_base"] and row["orders_and_production_contribution"] > 0
        result.append(row)
    assert len(result) == 14
    save(OUT / "pmi_expectation_facts.json", result)
    save(OUT / "source_clock_receipt.json", {"recorded_at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "script_sha256": digest(Path(__file__)), "actual_source_sha256": digest(CONTEXT / "monthly_pmi.json"), "events": len(result), "forecast_eligible": sum(r["forecast_status"] == "ELIGIBLE" for r in result), "base_triggers": sum(r["trigger_base"] for r in result), "demand_triggers": sum(r["trigger_demand"] for r in result), "returns_joined_at_this_stage": False})
    print(pd.DataFrame([{k: r.get(k) for k in ["stat_month", "actual", "expected", "surprise", "orders_and_production_contribution", "forecast_status", "trigger_base", "trigger_demand"]} for r in result]).to_string(index=False))
    return result


def run() -> None:
    facts = read(OUT / "pmi_expectation_facts.json")
    protocol = read(OUT / "protocol.json")
    old_protocol = read(domestic.OUT / "protocol.json")
    assert old_protocol["account_calendar"] == protocol["account_calendar"]
    market, features, dividends, engine, _ = domestic.inputs(old_protocol)
    clocks, event_returns = [], []
    for fact in facts:
        day = pd.Timestamp(fact["actual_public_date"])
        entry_idx = int(np.flatnonzero(market.date.gt(day))[0])
        entry_at = (market.date.iloc[entry_idx] + pd.Timedelta(hours=9, minutes=30)).tz_localize(TZ)
        assert pd.Timestamp(fact["actual_known_at"]) < entry_at
        assert np.isfinite(features.es95.iloc[entry_idx - 1])
        clock = {"event_id": fact["event_id"], "title": fact["stat_month"] + "官方制造业PMI", "stat_month": fact["stat_month"], "public_at": fact["actual_public_at"], "public_at_lower_bound": fact["actual_public_lower_bound"], "publication_precision": fact["actual_public_precision"], "public_at_upper_bound": fact["actual_known_at"], "entry_idx": entry_idx, "entry_date": market.date.iloc[entry_idx], "entry_at": entry_at, "planned_exit_date": market.date.iloc[entry_idx + 20], "entry_es95": features.es95.iloc[entry_idx - 1], "trigger_base": fact["trigger_base"], "trigger_demand": fact["trigger_demand"]}
        clocks.append(clock)
        for horizon in protocol["event_horizons"]:
            event_returns.append(domestic.fixed_event_window(market, dividends, engine, clock, horizon))
    assert len(event_returns) == 28 and all(x["status"] == "两端成交" for x in event_returns)
    save(OUT / "event_clocks.json", clocks)
    save(OUT / "event_returns.json", event_returns)
    return_by_key = {(x["event_id"], x["horizon"]): x for x in event_returns}
    descriptions = []
    for fact, clock in zip(facts, clocks):
        e = clock["entry_idx"]
        last_pre = int(np.flatnonzero(market.date.lt(pd.Timestamp(fact["actual_public_date"])))[-1])
        known_gap = float(market.open.iloc[e] / market.close.iloc[e - 1] - 1)
        pre_to_entry = float(market.open.iloc[e] / market.close.iloc[last_pre] - 1)
        descriptions.append({**fact, "entry_date": clock["entry_date"], "exit_date20": clock["planned_exit_date"], "last_pre_announcement_close_date": market.date.iloc[last_pre], "prior_20d_total_return": float(market.wealth.iloc[last_pre] / market.wealth.iloc[last_pre - 20] - 1), "pre_announcement_close_to_entry_open_return": pre_to_entry, "entry_overnight_gap": known_gap, "event_net_return5": return_by_key[(fact["event_id"], 5)]["net_return"], "event_net_return20": return_by_key[(fact["event_id"], 20)]["net_return"], "event_gross_return20": return_by_key[(fact["event_id"], 20)]["gross_return"]})
    save(OUT / "event_comparison.json", descriptions)
    accounts, paths, account_checks, all_cycles, decisions = {}, {}, {}, {}, {}
    (OUT / "ledgers").mkdir(exist_ok=True)
    for candidate, signal in RULES.items():
        gate = "trigger_base" if candidate == "PMI_POSITIVE_SURPRISE" else "trigger_demand"
        selected = [x for x in clocks if x[gate]]
        features[signal] = False
        for row in selected:
            features.loc[row["entry_idx"] - 1, signal] = True
        for capital in protocol["capitals_cny"]:
            case_id = f"{candidate}_{capital}_STRESS"
            case = {"case_id": case_id, "capital": capital, "policy": signal, "hold": 20, "start": protocol["account_calendar"][0], "end": protocol["account_calendar"][1]}
            ledger = account_model.simulate(market, features, dividends, case, "ORIGINAL", "STRESS", engine)
            ledger["reason"] = ledger.reason.replace({"前收盘信号入场": "已公开PMI预期偏差触发后的开盘入场"})
            ledger["es_observation_date"] = [market.date.iloc[int(i) - 1] for i in ledger.idx]
            ledger["prior_es95"] = [features.es95.iloc[int(i) - 1] for i in ledger.idx]
            ledger["fee_refund_equity"] = ledger.equity + (ledger.commission + ledger.slippage_cost).cumsum() + ledger.terminal_exit_reserve
            path = OUT / "ledgers" / (case_id + ".parquet")
            ledger.to_parquet(path, index=False)
            saved = pd.read_parquet(path)
            check = checks.check_account(saved, capital, selected, True)
            check["buy_dates_subset_of_registered_PMI_entries"] = check.pop("buy_dates_subset_of_frozen_six_entries")
            check["verified_from_saved_ledger"] = True
            stats = core.metrics(saved, capital)
            stats.pop("sharpe_252_diagnostic", None)
            stats.pop("cagr_252_diagnostic", None)
            cycles = core.cycle_records(saved, capital)
            records = []
            if len(cycles):
                assert cycles.closed.all() and saved.shares.iloc[-1] == 0
                assert abs(cycles.profit.sum() - stats["net_profit"]) < 1e-6
                for cycle in cycles.to_dict("records"):
                    part = saved.loc[saved.date.between(cycle["entry"], cycle["exit"])]
                    records.append({**cycle, "paid_friction_cny": float((part.commission + part.slippage_cost).sum()), "same_holdings_gross_profit": float(cycle["profit"] + part.commission.sum() + part.slippage_cost.sum()), "risk_reduction_fills": int((part.reason.eq("风险预算减仓") & part.filled_quantity.lt(0)).sum())})
            buys = saved.loc[saved.filled_quantity.gt(0)]
            annual = {str(year): int((buys.date.dt.year == year).sum()) for year in sorted(saved.date.dt.year.unique())}
            profits = [x["profit"] for x in records]
            wins, losses = [x for x in profits if x > 0], [x for x in profits if x < 0]
            stats.update(capital=capital, candidate=candidate, raw_trigger_count=len(selected), entry_fills=len(buys), entry_counts_by_year=annual, complete_years=[2022], partial_years=[2023], all_complete_years_entries_ge5=annual.get("2022", 0) >= 5, holding_close_days=int(saved.shares.gt(0).sum()), no_position_all_session_days=int((saved.shares.eq(0) & saved.shares_before.eq(0)).sum()), max_close_exposure=float(saved.exposure.max()), max_close_exposure_times_prior_ES=float((saved.exposure * saved.prior_es95).max()), max_drawdown_stop_triggered=bool(saved.risk_stopped.any()), risk_reduction_fills=int((saved.reason.eq("风险预算减仓") & saved.filled_quantity.lt(0)).sum()), rejected_nonzero_orders=int((saved.requested_quantity.ne(0) & saved.filled_quantity.eq(0)).sum()), worst_day=float(saved.net_return.min()), worst_cycle_profit=min(profits) if profits else None, best_cycle_profit=max(profits) if profits else None, winning_cycles=len(wins), losing_cycles=len(losses), cycle_win_rate=len(wins) / len(profits) if profits else None, average_win_loss_ratio=float(np.mean(wins) / abs(np.mean(losses))) if wins and losses else None, turnover_traded_notional_over_initial_capital=float(saved.notional.sum() / capital), dividend_recognized=float(saved.dividend_recognized.sum()), same_holdings_fee_refund=checks.equity_metrics(saved.fee_refund_equity, capital), sample_joint_target_met=bool(stats["net_sharpe"] is not None and stats["net_sharpe"] >= 1.2 and stats["cagr"] >= .1 and stats["max_drawdown"] <= .1))
            actions = []
            for row in clocks:
                day_row = saved.loc[saved.date.eq(row["entry_date"])].iloc[0]
                actions.append({"event_id": row["event_id"], "candidate_trigger": row[gate], "entry_date": day_row.date, "filled_quantity": day_row.filled_quantity, "shares_before": day_row.shares_before, "status": day_row.status, "reason": day_row.reason})
            accounts[case_id], account_checks[case_id], all_cycles[case_id], decisions[case_id] = stats, check, records, actions
            paths[case_id] = rel(path)
            print(f"{candidate}，本金{capital}元：净夏普{stats['net_sharpe']:.6f}，年化{stats['cagr']:.4%}，利润{stats['net_profit']:.2f}元，入场{len(buys)}次。", flush=True)
    save(OUT / "account_checks.json", account_checks)
    save(OUT / "account_cycles.json", all_cycles)
    save(OUT / "account_decisions.json", decisions)
    save(OUT / "result.json", {"study_id": protocol["study_id"], "computed_at": now(), "status": "CALCULATED_PENDING_REPORT", "event_count": 14, "fixed_event_windows": 28, "forecast_eligible": sum(x["forecast_status"] == "ELIGIBLE" for x in facts), "forecast_status_counts": pd.Series([x["forecast_status"] for x in facts]).value_counts().to_dict(), "new_full_accounts": 4, "parameters_fitted": 0, "accounts": accounts, "ledger_paths": paths, "benchmark_reused": {k: v for k, v in read(domestic.OUT / "result.json")["accounts"].items() if k in ["BUY_HOLD50_200000_REFERENCE", "CASH_200000_REFERENCE"]}, "sample_joint_target_met": any(x["sample_joint_target_met"] for x in accounts.values()), "independent_validation": False, "goal_achieved": False, "orders_authorized": False, "goal_turn_classification": "PROGRESS"})
    save(OUT / "research_receipt.json", {"computed_at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "script_sha256": digest(Path(__file__)), "source_facts_sha256": digest(OUT / "pmi_expectation_facts.json"), "account_checks_path": rel(OUT / "account_checks.json"), "input_limit": "全历史价格已在其他研究中使用；周报与PMI均为公开历史页面重建，非不可变首版快照；缺失月份不触发新交易。"})


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    facts = read(OUT / "event_comparison.json")
    result = read(OUT / "result.json")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9.5), gridspec_kw={"height_ratios": [1, 1.2]})
    positions = np.arange(len(facts))
    actual = np.array([x["observed_change_from_prior_published"] for x in facts])
    expected = np.array([np.nan if x.get("expected_change_from_prior_published") is None else x["expected_change_from_prior_published"] for x in facts])
    axes[0].bar(positions - .18, actual, .35, color="#177e89", label="实际PMI较前期变化")
    expected_bars = axes[0].bar(positions + .18, expected, .35, color="#a4b5bd", label="周报已经预计的变化")
    for i, fact in enumerate(facts):
        if fact["forecast_status"] == "TIMING_UNRESOLVED":
            expected_bars[i].set_hatch("///")
            expected_bars[i].set_edgecolor("#a36523")
        if fact["expected"] is None:
            axes[0].text(i, -2.2, "共识缺失", fontsize=8.5, ha="center", color="#8c5550")
    axes[0].axhline(0, color="#819097", lw=.8)
    axes[0].set_xticks(positions, [x["stat_month"] for x in facts], rotation=35, ha="right")
    axes[0].set_ylabel("PMI点数变化")
    axes[0].set_ylim(-2.55, 3.75)
    axes[0].set_title("经营恢复中，哪些变化已在这份周报的预期里", loc="left", fontsize=12)
    axes[0].legend(loc="upper left", ncol=2, frameon=False)
    axes[0].text(.99, .95, "4月斜纹：数值存在，但公开先后未充分确认，未触发入场", transform=axes[0].transAxes, ha="right", fontsize=9, color="#8a612f")
    labels = {"PMI_POSITIVE_SURPRISE": "只要求高于周报共识", "PMI_SURPRISE_DEMAND_IMPROVEMENT": "同时要求订单与生产改善"}
    colors = {"PMI_POSITIVE_SURPRISE": "#177e89", "PMI_SURPRISE_DEMAND_IMPROVEMENT": "#a25756"}
    for candidate in RULES:
        key = candidate + "_200000_STRESS"
        ledger = pd.read_parquet(ROOT / result["ledger_paths"][key])
        stats = result["accounts"][key]
        axes[1].plot(ledger.date, ledger.equity / 200000 - 1, color=colors[candidate], lw=2.2, label=f"{labels[candidate]}｜净夏普 {stats['net_sharpe']:.3f}")
    axes[1].axhline(0, color="#819097", lw=.9, linestyle="--", label="现金参考（收益为0）")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].set_ylabel("完整账户累计净收益")
    axes[1].set_title("20万元账户：包括空仓、风险减仓、成本和全部亏损", loc="left", fontsize=12)
    axes[1].legend(frameon=False, loc="upper left", fontsize=9.5)
    axes[1].set_ylim(-.067, .045)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("PMI改善、公告前共识与指数账户收益", x=.07, ha="left", fontsize=18, y=.98)
    fig.text(.07, .94, "固定14个月，11个月预期记录可用；两条规则各计算两个本金，未拟合参数", fontsize=11, color="#53616b")
    fig.text(.07, .02, "历史区间2022-01至2023-03；图中账户使用301个交易日。周报预期不等于公告前最后一刻预期，未达夏普1.2目标。", fontsize=9.5, color="#53616b")
    fig.subplots_adjust(left=.07, right=.98, top=.88, bottom=.09, hspace=.43)
    fig.savefig(OUT / "PMI预期偏差_月度变化与完整账户.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("已生成PMI变化、预期和完整账户对比图。")


def publish() -> None:
    result = read(OUT / "result.json")
    facts = read(OUT / "event_comparison.json")
    by_month = {x["stat_month"]: x for x in facts}
    cycles = read(OUT / "account_cycles.json")
    support = read(OUT / "support_source_manifest.json")
    assert all(x["status"] == "FETCHED" for x in support)
    base_key = "PMI_POSITIVE_SURPRISE_200000_STRESS"
    demand_key = "PMI_SURPRISE_DEMAND_IMPROVEMENT_200000_STRESS"
    base = result["accounts"][base_key]
    demand = result["accounts"][demand_key]
    source = {x["id"]: x for x in support}
    january = by_month["2023-01"]
    figures = OUT / "PMI预期偏差_月度变化与完整账户.png"
    assert figures.exists()

    def link(path: Path, label: str) -> str:
        return f"[{label}](<{path.as_posix()}>)"

    def table(headers, rows) -> str:
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])

    def value(x, precision=1, signed=False):
        if x is None:
            return "缺失"
        return f"{x:+.{precision}f}" if signed else f"{x:.{precision}f}"

    labels = {"PMI_POSITIVE_SURPRISE": "高于周报共识", "PMI_SURPRISE_DEMAND_IMPROVEMENT": "高于共识且订单/生产改善"}
    account_rows = []
    for x in result["accounts"].values():
        account_rows.append([labels[x["candidate"]], f"{x['capital']:,}", f"{x['net_profit']:+,.2f}", f"{x['net_sharpe']:.3f}", f"{x['cagr']:.2%}", f"{x['max_drawdown']:.2%}", x["entry_fills"], f"{x['winning_cycles']}/{x['entry_fills']}", f"{x['average_win_loss_ratio']:.2f}"])
    account_table = table(["固定规则", "本金元", "净利润元", "净夏普", "年化", "最大回撤", "完整交易", "盈利笔数", "平均盈亏比"], account_rows)
    event_rows = []
    for x in facts:
        eligibility = {"ELIGIBLE": "可用", "TIMING_UNRESOLVED": "时钟未确认", "CONSENSUS_NOT_IN_SELECTED_REPORT": "周报未覆盖"}[x["forecast_status"]]
        event_rows.append([x["stat_month"], x["actual_public_date"], value(x["actual"]), value(x["expected"]), value(x["surprise"], signed=True), value(x["orders_and_production_contribution"], 3, True), eligibility, x["entry_date"][:10], f"{x['event_net_return20']:+.2%}"])
    event_table = table(["统计月", "实际公布日", "实际PMI", "周报共识", "偏差点数", "订单+生产月度贡献", "预期资格", "次日或休市后开盘", "20日事件净收益"], event_rows)
    focus_rows = []
    for month in ["2022-05", "2022-06", "2023-01", "2023-02"]:
        x = by_month[month]
        focus_rows.append([month, f"{x['observed_change_from_prior_published']:+.1f}", f"{x['expected_change_from_prior_published']:+.1f}", f"{x['surprise']:+.1f}", f"{x['orders_and_production_contribution']:+.3f}", f"{x['component_contributions']['supplier_delivery']:+.3f}", f"{x['prior_20d_total_return']:+.2%}", f"{x['event_net_return20']:+.2%}"])
    focus_table = table(["统计月", "实际月变动", "周报预计月变动", "预期偏差", "订单/生产贡献", "交付反向贡献", "公告前20日指数总回报", "之后20日事件净收益"], focus_rows)
    stats = []
    for field, label in [("trigger_base", "高于周报共识"), ("trigger_demand", "同时要求订单/生产改善")]:
        selected = [x for x in facts if x[field]]
        stats.append({"label": label, "count": len(selected), "mean_event_gross_return20": float(np.mean([x["event_gross_return20"] for x in selected])), "mean_event_net_return20": float(np.mean([x["event_net_return20"] for x in selected])), "mean_event_net_return5": float(np.mean([x["event_net_return5"] for x in selected])), "is_full_account": False, "independent_samples": False})
    save(OUT / "event_group_descriptions.json", stats)
    findings = [
        {"month": "2022-05", "fact": "复工推进，物流受阻企业占比下降8个百分点；订单和生产贡献为正，交付恢复因反向计入抵消部分综合指数升幅。", "source": source["nbs_review_202205"]["url"], "equity_inference": "与经营恢复机制相符，单个盈利窗口不足以证明固定策略有效。"},
        {"month": "2022-06", "fact": "复工和此前受抑制的产需释放继续，但49.3%企业反映订单不足，出厂价格仍位于收缩区间。", "source": source["nbs_review_202206"]["url"], "equity_inference": "恢复速度、需求是否充足、售价与利润是不同变量；公告前指数已涨9.02%，但不能因此量化已完全定价。"},
        {"month": "2023-01", "fact": "PMI回升3.1点，周报已预计其中2.8点；公告前20日指数总回报9.74%。", "source": january["source_url"], "equity_inference": "这份周报早已预期大部分制造业回升；无法由此推断全部投资者已完全反映或已知未来所有消息。"},
        {"month": "2023-02", "fact": "复工复产与经营恢复加快，但反映订单不足的制造业和服务业企业占比仍超过50%。", "source": source["nbs_review_202302"]["url"], "equity_inference": "扩散指数相较上月改善，与绝对需求缺口仍大可以并存；对指数盈利的映射尚未识别。"},
    ]
    save(OUT / "mechanism_findings.json", {"selection": "账户结果后的原因对照，只解释，不新增交易条件。", "findings": findings, "main_inference": "经营改善、市场先前预期、需求缺口及指数盈利兑现需要分别度量。", "competing_explanations": ["指数在公告前后的价格变化同时包含其他国内外消息，不能用PMI单独归因。", "广泛企业PMI调查的覆盖、权重与股票指数不同；名义收入、成本和风险补偿可以抵消数量恢复。"], "causal_return_identified": False})
    report_path = OUT / "历史发现_PMI预期差与指数收益.md"
    report = f"""# 中国PMI预期差与指数收益：本轮完成结果

**结论：两个事先固定的入场规则、两个本金账户全部未达标。** 相对同一来源周报的制造业PMI正偏差，及其加上订单与生产改善的表达，都没有形成成本后的账户优势。20万元主账户净夏普{base['net_sharpe']:.3f}、年化{base['cagr']:.2%}；加条件账户更低。两套规则各自封存，不通过反号、选择月份或更换持有期营救。

这轮有实际新增：14个月原始预期来源对照、28个固定事件窗口、4个完整账户，以及经营恢复原因的原文。它不延伸上一轮已拒绝的美国CPI规则，不扩展个股，也不产生当前市场预测。

**首先区分三种量：经营比上月改善多少、市场此前已预期多少、买入之后实际还能赚多少。** 2023年1月PMI从47.0升至50.1，实际改善3.1点；1月27日周报已预计49.8，相当于预期改善2.8点。相对这份共识的意外只有0.3点，而不是3.1点。公告前20个交易日510300总回报已达{january['prior_20d_total_return']:.2%}，2月1日开盘之后20日事件净收益为{january['event_net_return20']:.2%}。这支持区分数据改善和新增信息，尚不能证明价格变化全部由PMI造成。[公告前周报]({january['source_url']})、[统计局当次发布]({january['actual_url']})。

{account_table}

账户区间为2022年1月1日至2023年3月31日，实际301个交易日，包含全部空仓日，按242个交易日年化，现金及无风险收益均取0。只持有510300和人民币现金。每个候选持有20个开盘间隔，期间同类新触发不加仓、不延长。沿用50%目标仓位上限、条件5日ES95预算2.5%、-10%冲击预算5%、10%回撤触发、T+1、整手、分红和压力成本；开盘成交是既有日线执行近似。两万元和二十万元分别计最低佣金与整手约束。

2022完整自然年，两种规则都实际开仓4次；2023年的2次仅覆盖第一季度，不年化成全年次数。主规则2022年虽然有5次原始触发，但一次在持仓期间被忽略，不能报成5笔新机会。完整账户频率同样未满足每完整年至少5次的当前记录要求。

**较高的单笔盈亏比没有弥补低胜率。** 主账户平均盈利/平均亏损约{base['average_win_loss_ratio']:.2f}倍，但6笔交易只有1笔盈利，成本后合计仍亏{abs(base['net_profit']):,.2f}元。这些比率只描述这6笔，不是已校准的未来概率。按这个已实现盈亏比做纯算术，收支平衡胜率约{1 / (1 + base['average_win_loss_ratio']):.1%}，高于本样本的{base['cycle_win_rate']:.1%}。

**从因子背后的原因看，恢复不等于需求缺口已被补足。**

{focus_table}

“订单/生产贡献”用官方固定权重乘当次公告内相对上月的变动：新订单30%、生产25%。交付按反向15%计算。其他就业、原料库存及舍入差另存。这里只有综合PMI的周报共识，没有各子项的市场共识，所以这些构成数是月度变化贡献，不能写成子项超预期贡献。

2022年5月的官方解释指向复工推进、上下游衔接改善，反映运输不畅的企业占比下降8个百分点。生产和订单的改善合计贡献+3.005点；交付恢复更快，反向计入后贡献-1.035点。综合PMI仍低于50，却已反映经营收缩显著缓和。6月1日入场的事件窗口净收益+8.58%，是主账户唯一盈利周期。[统计局5月原因说明]({source['nbs_review_202205']['url']})。

2022年6月PMI继续回升，订单与生产贡献仍正；官方同时记录49.3%的企业反映订单不足，出厂价格46.3也显示售价压力。**比上月恢复，与企业认为订单已充足，是不同问题。** 公告前指数已上涨9.02%，公告前收盘至7月1日入场又上涨1.39%；入场后20日事件净收益却为-5.53%。价格此前上涨与新增空间减少相容，但本轮没有识别可量化的“透支阈值”，也不能排除其间其他消息。[统计局6月原因说明]({source['nbs_review_202206']['url']})。

2023年1月的官方说明把产需回升和春节生产安排同时记录。政府网转载页是2月1日，本轮只将它作为原因解释，没有回填到当天开盘信号；信号使用1月31日统计局原始数值与此前周报。2月PMI进一步升至52.6，官方解释包括疫情影响减退、节后复工与稳经济政策，同时明确制造业和服务业中反映订单不足的企业占比仍超过50%。广泛改善与存量缺口可以同时存在，指数盈利还受售价、成本和行业权重等影响。[1月官方解释的政府网转载]({source['gov_nbs_review_202301']['url']})、[2月统计局原文]({source['nbs_review_202302']['url']})。

这四个月是账户结果后的原因对照，不是根据结果发明的可交易阶段标签。一个主要解释是经营数量恢复、需求缺口和价格预期的错位；两个竞争解释是同期其他国内外信息，以及PMI企业覆盖与指数盈利覆盖的差异。各解释对收益的独立贡献仍未识别。

**为什么加上更合理的条件，完整账户反而更差。**

订单与生产条件排除了2022年1月PMI信号。主规则因此已有2月7日至3月7日的持仓，而加条件账户在3月2日之前是空仓，能够响应2月PMI信号，持有至3月30日。两个20万元账户对应的第一笔损益分别为{cycles[base_key][0]['profit']:+,.2f}元和{cycles[demand_key][0]['profit']:+,.2f}元。后续账户本金和风险余量也随账本改变，因此加条件不是简单删掉一笔，再把其余旧利润原样相加。

两个账户的唯一盈利周期均来自5月PMI后的6月交易。主规则7个原始触发的20日事件净收益均值为{stats[0]['mean_event_net_return20']:.2%}，毛收益均值也为{stats[0]['mean_event_gross_return20']:.2%}。这些事件有重叠，不是可叠加账户收益；它们说明问题在进入完整账户之前已经存在。

**成本和风险约束没有掩盖一条足够强的毛收益信号。** 主账户佣金与滑点合计{base['commission'] + base['slippage']:,.2f}元。只把同一冻结持仓的全部成本返还，仍亏{abs(base['same_holdings_fee_refund']['net_profit']):,.2f}元，诊断夏普{base['same_holdings_fee_refund']['net_sharpe']:.3f}。加条件账户退款后也仍亏{abs(demand['same_holdings_fee_refund']['net_profit']):,.2f}元。两者都没有触发10%回撤停止，原始触发中的订单也未因非零委托被全部拒绝。风险减仓确实影响路径，但本轮不通过放宽上限或删除风险减仓寻找更好结果；毛收益与剩余持有优势不足是更直接的证据。

同区间既有50%买入持有参考的净夏普为-0.815、年化-6.83%、最大回撤13.79%；它不满足本轮全部风险预算。现金参考收益为0。候选比下跌市场少亏，仍没有达到自身的正收益和夏普目标。

**14个月全部保留，缺失和反例不删。**

{event_table}

表中5/20日事件计算为10万元名义交易，含压力费用，用于同口径比较；不是20万元风险账户。2022年4月和缺失月份也计算客观价格窗口，但不进入信号账户。10月PMI低于共识、之后指数却上涨，是原规则的反例，不能据此将规则反号。

**预期来源和时间限制。** 14份所选周报中，12份有目标月预测数值，11份能在本轮保守时钟下确认早于公告。11月、12月最近周报未覆盖目标发布；4月数值47.3存在，但周报正文只有纽约4月29日日精度、统计局复用页只有4月30日日精度，两个时间区间重叠，保留未确认，不入场。其余月份也不是公告前最后一刻共识。6月49.0、7月51.5等值已由各自周报数字栏及文字说明交叉核对，按同一来源原值保留。

目录与正文均为周五，页面标题多为随后周一；正文前瞻段包含周日前后待发布项目，说明页面标题不能机械当作上线时刻。本轮用目录与正文的较晚日期纽约日末，并保存标题。若一律采用三日期最晚值，则有预测的记录中只剩5个月满足时钟；这是日期解释的资料限制，不是选择盈利日期的开关。周报中个别星期、时区排期也与中国实际发布不一致，交易日期统一来自统计局资料。日精度的实际公告另外保存早晚界，不补写精确分钟。

原网页为目前取得的历史重建材料，不是本机当年保存的不可变首版；全部价格此前在其他研究中使用，本轮无独立验证。周报共识的样本机构和更新过程未完整取得，结论只针对上述周报锚及规则，不能外推为所有PMI信息无效。

**本轮处置：两条固定表达均拒绝并保留。** 新增原因材料解释了恢复速度与存量需求的差别，尚未给出能够弥补账户亏损的独立信息。夏普1.2与年化10%的联合目标仍未实现。后续需要补的是“宏观恢复怎样兑现为指数总体盈利”的独立证据，先核对已有指数研究覆盖，避免再围绕这14个月更改阈值、方向或持有期。

{link(figures, '查看月度变化、预期与完整账户对比图')}。

资料入口：{link(OUT / 'pmi_expectation_facts.json', '14个月预期与组成事实')}；{link(OUT / 'event_comparison.json', '完整事件比较')}；{link(OUT / 'account_cycles.json', '全部账户交易周期')}；{link(OUT / 'account_decisions.json', '触发与实际执行')}；{link(OUT / 'mechanism_findings.json', '原因与竞争解释')}；{link(OUT / 'source_manifest.json', '周报原文来源')}；{link(OUT / 'support_source_manifest.json', '统计局原因来源')}；{link(OUT / 'result.json', '完整结果')}；{link(ROOT / 'research/historical_index_pmi_expectation_v1.py', '完整计算脚本')}。

必要核对已覆盖：当月及当次公告内上月行、预期可用时间、全部14事件、保存账本的净值与损益复算、现金与整手、T+1和完整日历。没有增加参数搜索或长周期检验。完成时间：{now()}。
"""
    report_path.write_text(report, encoding="utf-8")
    summary = "14个月中国PMI有11个月合格周报共识；7个正偏差触发，加入订单/生产改善后6个触发。四账户全部未达标，20万元主规则夏普-0.634696、年化-2.2047%，加条件夏普-1.066353、年化-4.1935%；退款后仍亏，2022年均仅4次实际入场。恢复幅度、预先预期、存量需求不足与指数盈利兑现是不同信息，两个表达均拒绝封存。"
    next_question = "转离PMI阈值与持有期搜索：先核对已有指数层面研究对2022至2023年宏观恢复、名义收入/成本、总体盈利兑现的覆盖与缺口；只推进有独立新证据的传导环节，不回到个股扩展，不把事后赢家阶段拼入账户。"
    result.update(status="COMPLETED_HISTORICAL_STUDY_CANDIDATES_REJECTED", completed_at=now(), candidate_dispositions={key: "REJECTED_FROZEN" for key in RULES}, classification="PROGRESS_INDEX_PMI_EXPECTATION_COMPONENTS_AND_FOUR_ACCOUNTS", discovery=summary, next_historical_question=next_question, report=rel(report_path), figure=rel(figures), numerical_forecast_months=sum(x["expected"] is not None for x in facts), stricter_three_date_eligible_months=sum(x["expected"] is not None and x["three_date_upper_bound_before_release"] for x in facts), forecast_population_complete=False, inherited_source_first_vintage_verified=False, explanatory_reviews_after_results=True, report_status="WRITTEN_PENDING_FINAL_REVIEW", goal_achieved=False)
    save(OUT / "result.json", result)
    receipt = read(OUT / "research_receipt.json")
    receipt.update(report_completed_at=now(), script_sha256=digest(Path(__file__)), report_sha256=digest(report_path), supporting_source_manifest=rel(OUT / "support_source_manifest.json"))
    save(OUT / "research_receipt.json", receipt)
    for path in [ROOT / "config/510300_historical_cause_discovery_v1.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]:
        config = read(path)
        if path.name == "510300_historical_cause_discovery_v1.json":
            config.update(latest_completed_study=rel(OUT / "result.json"), latest_report=rel(report_path), current_study=rel(OUT / "protocol.json"), updated_at=now(), latest_result_summary=summary, next_historical_question=next_question, goal_achieved=False)
        else:
            config.update(current_round=result["study_id"], latest_progress_receipt=rel(OUT / "result.json"), latest_continuation_report=rel(report_path), latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=now(), latest_historical_report=rel(report_path), next_research_question=next_question, latest_historical_index_pmi_expectation=rel(OUT / "result.json"), goal_achieved=False)
        save(path, config)
    print(json.dumps({"报告": rel(report_path), "账户数量": 4, "目标完成": False}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="中国PMI预期偏差的固定历史研究")
    parser.add_argument("action", choices=["collect", "parse", "run", "plot", "publish"])
    action = parser.parse_args().action
    {"collect": collect, "parse": parse_facts, "run": run, "plot": plot, "publish": publish}[action]()
