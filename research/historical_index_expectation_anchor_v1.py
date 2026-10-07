"""历史指数研究：以同一来源的周前共识区分前月变化与预期偏差。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urljoin, parse_qs, urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import historical_index_external_reopening_v1 as previous
from research import historical_index_reopening_constraints_v1 as domestic
from research import historical_index_rrr_full_account_v1 as checks
from research import factor96_bottleneck_diagnostic_v1 as account_model
from research import factor96_margin_repair_v1 as core

OUT = ROOT / "reports/research/510300_historical_index_expectation_anchor_v1"
PRIOR = previous.OUT
TZ = ZoneInfo("Asia/Shanghai")
SIGNAL = "week_ahead_dovish_surprise_before_open"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    core.save(Path(path), value)


def source_plan():
    archives = []
    for year in [2022, 2023]:
        path = OUT / "sources" / f"archive_{year}.html"
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for link in soup.find_all("a", href=True):
            if "byshoweventarticle" not in link["href"]:
                continue
            text = link.find_parent("tr").get_text(" ", strip=True)
            date = re.search(r"(\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))$", text).group()
            day = pd.to_datetime(date, format="%m/%d/%Y" if len(date.split("/")[-1]) == 4 else "%m/%d/%y")
            fid = parse_qs(urlparse(link["href"]).query)["fid"][0]
            archives.append({"id": "econoday_" + fid, "fid": fid, "archive_date": day.strftime("%Y-%m-%d"),
                             "title": link.get_text(" ", strip=True),
                             "url": urljoin("https://fidelity.econoday.com/", link["href"]),
                             "file": f"econoday_{fid}.html"})
    protocol = read(OUT / "protocol.json")
    links, sources = [], {}
    for kind, dates in [("CPI", protocol["cpi_dates"]), ("FOMC", protocol["fomc_dates"])]:
        for day in dates:
            candidates = [r for r in archives if r["archive_date"] < day]
            selected = max(candidates, key=lambda r: r["archive_date"])
            assert 0 < (pd.Timestamp(day) - pd.Timestamp(selected["archive_date"])).days <= 10
            links.append({"event_id": kind + "_" + day.replace("-", ""), "kind": kind,
                          "event_date": day, "source_id": selected["id"], "archive_date": selected["archive_date"]})
            sources[selected["id"]] = selected
    save(OUT / "event_source_map.json", links)
    save(OUT / "source_plan.json", list(sources.values()))
    return list(sources.values())


def collect():
    sources = source_plan()
    path = OUT / "source_manifest.json"
    manifest = read(path) if path.exists() else []
    done = {r["id"] for r in manifest if r["status"] == "FETCHED"}
    def fetch(source):
        row = {**source, "retrieved_at": datetime.now(TZ).isoformat()}
        try:
            response = requests.get(source["url"], timeout=25)
            response.raise_for_status()
            target = OUT / "sources" / source["file"]
            target.write_bytes(response.content)
            soup = BeautifulSoup(response.content.decode("utf-8"), "html.parser")
            for tag in soup.select("script,style"):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            target.with_suffix(".txt").write_text(text, encoding="utf-8")
            assert "Global Economics" in text, "页面未含周报"
            row.update(status="FETCHED", bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest(),
                       path=target.relative_to(ROOT).as_posix(), http_status=response.status_code)
        except Exception as error:
            row.update(status="UNAVAILABLE", error=str(error))
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, s) for s in sources if s["id"] not in done]):
            row = future.result()
            manifest.append(row)
            save(path, manifest)
            print(row["id"], row["archive_date"], row["status"], flush=True)


def preview():
    sources = {r["id"]: r for r in read(OUT / "source_plan.json")}
    for item in read(OUT / "event_source_map.json"):
        source = sources[item["source_id"]]
        text = (OUT / "sources" / source["file"]).with_suffix(".txt").read_text(encoding="utf-8")
        compact = re.sub(r"\s+", " ", text)
        pattern = r"US CPI for " if item["kind"] == "CPI" else r"US (?:Federal Reserve|FOMC)"
        found = list(re.finditer(pattern, compact))
        print(item["event_id"], source["archive_date"], source["fid"])
        for match in found:
            print(compact[match.start():match.start() + 1350])
        if not found:
            print("未匹配标准前瞻段")


def parse_facts():
    source_lookup = {r["id"]: r for r in read(OUT / "source_plan.json")}
    actual_cpi = {r["event_id"]: r for r in read(PRIOR / "cpi_facts.json")}
    actual_fed = {r["event_id"]: r for r in read(PRIOR / "fomc_facts_including_baseline.json")}
    event_clocks = {r["event_id"]: r for r in read(PRIOR / "external_event_clocks.json")}
    rates = {r["event_id"]: r for r in read(PRIOR / "yield_reactions.json")}
    returns = {(r["event_id"], r["horizon"]): r for r in read(PRIOR / "event_returns.json")}
    facts = []
    for item in read(OUT / "event_source_map.json"):
        source = source_lookup[item["source_id"]]
        path = OUT / "sources" / source["file"]
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        header = re.search(r"Global Economics\s*-\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", text).group(1)
        title = soup.title.get_text(" ", strip=True)
        title_match = re.search(r"Global Economics\s+(\d+)\s+(\d+),\s*(\d{4})", title)
        title_day = pd.Timestamp(year=int(title_match[3]), month=int(title_match[1]), day=int(title_match[2]))
        header_day = pd.to_datetime(header, format="%B %d, %Y")
        # 标题为周一、正文为前周五；同时保存，采用两者与索引中较晚日期的纽约日终。
        upper_day = max(header_day, title_day, pd.Timestamp(source["archive_date"]))
        source_upper = upper_day.tz_localize("America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        clock = event_clocks[item["event_id"]]
        assert source_upper < pd.Timestamp(clock["public_at"]), item["event_id"]
        row = {**item, "source_url": source["url"], "source_path": path.relative_to(ROOT).as_posix(),
               "provider": "Econoday周报共识；网站标注共识源为Econoday及MNI",
               "body_article_date": header_day.strftime("%Y-%m-%d"),
               "page_title_date": title_day.strftime("%Y-%m-%d"),
               "historical_public_date_upper_bound": source_upper.isoformat(),
               "historical_immutable_snapshot": False, "measurement": "相对前周周报共识的偏差，不是公告前最后一刻定价",
               "actual_public_at": clock["public_at"], "entry_date": clock["entry_date"],
               "pre_open_gap": clock["pre_open_gap"], "old_deceleration_trigger": clock["trigger"],
               "net_return5": returns[(item["event_id"], 5)]["net_return"],
               "net_return20": returns[(item["event_id"], 20)]["net_return"],
               "gross_return20": returns[(item["event_id"], 20)]["gross_return"],
               "us_nominal2_change_bp": rates[item["event_id"]]["nominal2_change_bp"],
               "us_real10_change_bp": rates[item["event_id"]]["real10_change_bp"]}
        if item["kind"] == "CPI":
            number = r"([+-]?\d+(?:\.\d+)?)"
            pattern = (r"US CPI for ([A-Za-z]+)\s*\([^)]*\)\s*Consensus Forecast, Month over Month\s*:\s*" + number +
                       r"%\s*Consensus Forecast, Year over Year\s*:\s*" + number +
                       r"%\s*US CPI Core, Less Food & Energy\s*Consensus Forecast, Month over Month\s*:\s*" + number +
                       r"%\s*Consensus Forecast, Year over Year\s*:\s*" + number + r"%")
            match = re.search(pattern, text)
            assert match, item["event_id"]
            actual = actual_cpi[item["event_id"]]
            assert match[1] == pd.Timestamp(actual["economic_month"] + "-01").strftime("%B")
            values = dict(zip(["headline_mom", "headline_yoy", "core_mom", "core_yoy"], map(float, match.groups()[1:])))
            row["economic_month"] = actual["economic_month"]
            for key, expected in values.items():
                component, period = key.split("_")
                observed = actual["components"][component]["current_" + period]
                row[key + "_expected"] = expected
                row[key + "_actual"] = observed
                row[key + "_surprise_pp"] = round(observed - expected, 10)
            row["core_mom_previous_same_release"] = actual["components"]["core"]["previous_mom"]
            row["trigger"] = row["core_mom_surprise_pp"] < 0
            row["direction"] = "低于周报共识" if row["trigger"] else ("高于周报共识" if row["core_mom_surprise_pp"] > 0 else "符合周报共识")
            row["trigger_reason"] = "核心环比低于周报共识" if row["trigger"] else "核心环比不低于周报共识"
            row["evidence_excerpt"] = match.group()
        else:
            match = re.search(r"US Federal Reserve Announcement\s*\(([^)]*)\)\s*Consensus Forecast, Policy Rate Change:\s*(\d+) basis points", text)
            assert match, item["event_id"]
            actual = actual_fed[item["event_id"]]
            expected_step = float(match[2])
            expected_upper = actual["range_upper_pct"] - actual["step_bp"] / 100 + expected_step / 100
            range_match = re.match(r"\s*Consensus Forecast, Target Range:\s*([\d.]+)%? to ([\d.]+)", text[match.end():])
            if range_match:
                assert abs(expected_upper - float(range_match[2].rstrip("."))) < 1e-10
            row.update(expected_step_bp=expected_step, actual_step_bp=actual["step_bp"],
                       expected_upper_pct=expected_upper, actual_upper_pct=actual["range_upper_pct"],
                       target_surprise_bp=round((actual["range_upper_pct"] - expected_upper) * 100, 8),
                       source_schedule_text=match[1],
                       execution_clock_source="原始美联储14:00纽约时钟；周报多处15:00排期不用于交易对时。",
                       evidence_excerpt=match.group() + (range_match.group() if range_match else ""))
            row["trigger"] = row["target_surprise_bp"] < 0
            row["direction"] = "低于周报预期" if row["trigger"] else ("高于周报预期" if row["target_surprise_bp"] > 0 else "符合周报预期")
            row["trigger_reason"] = "目标上限低于周报预期" if row["trigger"] else "目标上限不低于周报预期"
        row["comparison_available"] = True
        facts.append(row)
    facts.sort(key=lambda r: r["event_date"])
    assert len(facts) == 23
    save(OUT / "expectation_comparison.json", facts)
    for r in facts:
        print(r["event_id"], r["direction"], "原触发" if r["old_deceleration_trigger"] else "原未触发",
              "现触发" if r["trigger"] else "现未触发", flush=True)
    return facts


def run_accounts(facts):
    protocol = read(OUT / "protocol.json")
    china_protocol = read(domestic.OUT / "protocol.json")
    market, features, dividends, engine, _ = domestic.inputs(china_protocol)
    assert protocol["account_period"] == china_protocol["account_calendar"]
    original_clocks = {r["event_id"]: r for r in read(PRIOR / "external_event_clocks.json")}
    clocks = [{**original_clocks[r["event_id"]], "trigger": True, "trigger_reason": r["trigger_reason"],
               "expectation_source_upper_bound": r["historical_public_date_upper_bound"]} for r in facts if r["trigger"]]
    save(OUT / "candidate_entry_clocks.json", clocks)
    features[SIGNAL] = False
    for row in clocks:
        features.loc[int(row["entry_idx"]) - 1, SIGNAL] = True
    accounts, paths, verification, cycles, decisions = {}, {}, {}, {}, {}
    (OUT / "ledgers").mkdir(exist_ok=True)
    for capital in protocol["capitals_cny"]:
        case_id = f"WEEK_AHEAD_SURPRISE_{capital}_STRESS"
        case = {"case_id": case_id, "capital": capital, "policy": SIGNAL, "hold": 20,
                "start": protocol["account_period"][0], "end": protocol["account_period"][1]}
        ledger = account_model.simulate(market, features, dividends, case, "ORIGINAL", "STRESS", engine)
        ledger["reason"] = ledger.reason.replace({"前收盘信号入场": "相对周报共识偏鸽后开盘入场"})
        ledger["es_observation_date"] = [market.date.iloc[int(i) - 1] for i in ledger.idx]
        ledger["prior_es95"] = [features.es95.iloc[int(i) - 1] for i in ledger.idx]
        ledger["fee_refund_equity"] = ledger.equity + (ledger.commission + ledger.slippage_cost).cumsum() + ledger.terminal_exit_reserve
        path = OUT / "ledgers" / f"{case_id}.parquet"
        ledger.to_parquet(path, index=False)
        saved = pd.read_parquet(path)
        checked = checks.check_account(saved, capital, clocks, True)
        checked["buy_dates_subset_of_frozen_surprise_entries"] = checked.pop("buy_dates_subset_of_frozen_six_entries")
        checked["verified_from_saved_ledger"] = True
        stats = core.metrics(saved, capital)
        stats.pop("sharpe_252_diagnostic", None)
        stats.pop("cagr_252_diagnostic", None)
        cycle = core.cycle_records(saved, capital)
        assert cycle.closed.all() and saved.shares.iloc[-1] == 0
        assert abs(cycle.profit.sum() - stats["net_profit"]) < 1e-6
        records = []
        for row in cycle.to_dict("records"):
            part = saved.loc[saved.date.between(row["entry"], row["exit"])]
            records.append({**row, "paid_friction_cny": (part.commission + part.slippage_cost).sum(),
                            "same_holdings_gross_profit": row["profit"] + part.commission.sum() + part.slippage_cost.sum(),
                            "mean_close_exposure": part.exposure.mean(),
                            "risk_reduction_fills": int((part.reason.eq("风险预算减仓") & part.filled_quantity.lt(0)).sum())})
        stats.update(capital=capital, raw_trigger_count=len(clocks), entry_fills=int(saved.filled_quantity.gt(0).sum()),
                     holding_close_days=int(saved.shares.gt(0).sum()),
                     no_position_all_session_days=int((saved.shares.eq(0) & saved.shares_before.eq(0)).sum()),
                     max_close_exposure=saved.exposure.max(),
                     max_close_exposure_times_prior_ES=(saved.exposure * saved.prior_es95).max(),
                     close_exposure_above_50pct_days=int(saved.exposure.gt(.5 + 1e-12).sum()),
                     max_drawdown_stop_triggered=bool(saved.risk_stopped.any()),
                     risk_reduction_fills=int((saved.reason.eq("风险预算减仓") & saved.filled_quantity.lt(0)).sum()),
                     rejected_nonzero_orders=int((saved.requested_quantity.ne(0) & saved.filled_quantity.eq(0)).sum()),
                     worst_day=saved.net_return.min(), worst_cycle_profit=cycle.profit.min(), best_cycle_profit=cycle.profit.max(),
                     turnover_traded_notional_over_initial_capital=saved.notional.sum() / capital,
                     dividend_recognized=saved.dividend_recognized.sum(),
                     same_holdings_fee_refund=checks.equity_metrics(saved.fee_refund_equity, capital),
                     sample_joint_target_met=bool(stats["net_sharpe"] is not None and stats["net_sharpe"] >= 1.2
                         and stats["cagr"] >= .1 and stats["max_drawdown"] <= .1))
        actions = []
        for row in clocks:
            day = saved.loc[saved.date.eq(pd.Timestamp(row["entry_date"]))].iloc[0]
            actions.append({"event_id": row["event_id"], "entry_date": day.date,
                            "filled_quantity": day.filled_quantity, "shares_before": day.shares_before,
                            "status": day.status, "reason": day.reason})
        accounts[case_id], verification[case_id], cycles[case_id], decisions[case_id] = stats, checked, records, actions
        paths[case_id] = path.relative_to(ROOT).as_posix()
        print(f"账户{capital}元：夏普{stats['net_sharpe']:.6f}，年化{stats['cagr']:.4%}，净利润{stats['net_profit']:.2f}元。", flush=True)
    save(OUT / "account_checks.json", verification)
    save(OUT / "account_cycles.json", cycles)
    save(OUT / "account_decisions.json", decisions)
    return accounts, paths


def group_diagnostics(facts):
    summaries = []
    for kind in ["CPI", "FOMC"]:
        rows = [r for r in facts if r["kind"] == kind]
        for direction in sorted({r["direction"] for r in rows}):
            group = [r for r in rows if r["direction"] == direction]
            summaries.append({"kind": kind, "direction": direction, "count": len(group),
                              "mean_pre_open_gap": np.mean([r["pre_open_gap"] for r in group]),
                              "mean_us_nominal2_change_bp": np.mean([r["us_nominal2_change_bp"] for r in group]),
                              "mean_us_real10_change_bp": np.mean([r["us_real10_change_bp"] for r in group]),
                              "mean_net_return5": np.mean([r["net_return5"] for r in group]),
                              "mean_net_return20": np.mean([r["net_return20"] for r in group]),
                              "positive_net_return20_count": sum(r["net_return20"] > 0 for r in group),
                              "is_independent_sample_or_full_account": False})
    save(OUT / "group_descriptions.json", summaries)
    return summaries


def run():
    facts = parse_facts()
    accounts, paths = run_accounts(facts)
    groups = group_diagnostics(facts)
    p = read(OUT / "protocol.json")
    primary = accounts["WEEK_AHEAD_SURPRISE_200000_STRESS"]
    result = {"study_id": p["study_id"], "completed_at": datetime.now(TZ).isoformat(),
              "mode": "HISTORICAL_ONLY", "source_weekly_reports": len(read(OUT / "source_plan.json")),
              "matched_events": len(facts), "cpi_events": 14, "fomc_events": 9,
              "new_full_accounts": 2, "parameters_fitted": 0,
              "new_trigger_count": sum(r["trigger"] for r in facts),
              "removed_previous_trigger_ids": [r["event_id"] for r in facts if r["old_deceleration_trigger"] and not r["trigger"]],
              "added_trigger_ids": [r["event_id"] for r in facts if not r["old_deceleration_trigger"] and r["trigger"]],
              "accounts": accounts, "ledger_paths": paths, "group_descriptions": groups,
              "sample_joint_target_met": primary["sample_joint_target_met"],
              "candidate_disposition": "REJECTED_FROZEN" if not primary["sample_joint_target_met"] else "SAMPLE_ONLY_OTHER_REQUIREMENTS_UNPROVEN",
              "independent_validation": False, "goal_achieved": False, "orders_authorized": False,
              "final_pre_announcement_consensus_complete": False,
              "classification": "PROGRESS_INDEX_WEEK_AHEAD_EXPECTATION_ANCHOR_AND_SIGNAL_COMPARISON"}
    save(OUT / "result.json", result)


def explanation_evidence():
    facts = read(OUT / "expectation_comparison.json")
    saved = ROOT / "reports/research/510300_historical_fomc_transmission_v1/us_events_before_etf_returns.parquet"
    raw = pd.read_parquet(saved).set_index("Date")
    rows = []
    for fact in facts:
        if fact["kind"] != "FOMC":
            continue
        item = raw.loc[pd.Timestamp(fact["event_date"])]
        rows.append({"event_id": fact["event_id"], "date": fact["event_date"],
                     "week_ahead_expected_step_bp": fact["expected_step_bp"], "actual_step_bp": fact["actual_step_bp"],
                     "relative_to_week_ahead_target_surprise_bp": fact["target_surprise_bp"],
                     "ois1y_change_bp": float(item.OIS1Y) * 100, "ois2y_change_bp": float(item.OIS2Y) * 100,
                     "sp500_window_return_pct": float(item.SP500),
                     "source_path": saved.relative_to(ROOT).as_posix(),
                     "source_url": "https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/",
                     "window": "声明前10分钟至记者会开始后60分钟，合计100分钟；不是整场逐字结束时刻。",
                     "used_in_candidate_signal": False, "historical_first_vintage": False,
                     "interpretation": "窄窗市场反应仍可能包含风险溢价及混合信息，不等于已隔离的外生冲击。"})
    assert len(rows) == 9
    save(OUT / "fomc_expectation_and_repricing.json", rows)
    support = read(OUT / "supporting_source_manifest.json")
    for row in list(support):
        path = OUT / "sources" / (row["id"] + ".web.txt")
        if path.exists() and not any(r["id"] == row["id"] + "_web" for r in support):
            support.append({"id": row["id"] + "_web", "url": row["url"], "status": "WEB_TOOL_PRIMARY_SOURCE_EXTRACT",
                            "path": path.relative_to(ROOT).as_posix(), "recorded_at": datetime.now(TZ).isoformat(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    save(OUT / "supporting_source_manifest.json", support)
    cards = {
        "lseg_june13": {"source_url": "https://lipperalpha.refinitiv.com/2022/06/will-the-fed-raise-by-75-basis-points/",
                         "economic_date": "2022-06-13", "initial_probability_75bp": .20, "updated_probability_75bp": .69,
                         "intraday_update_minute_known": False,
                         "meaning": "LSEG/Refinitiv当时报道自身IRPR数据；周报之后、6月15日决定之前的定价已经移动。不能把周报50bp与实际75bp之差全部称为宣布瞬间的冲击。",
                         "strategy_input": False},
        "nyfed_survey_publication": {"source_url": "https://www.newyorkfed.org/newsevents/events/regional_outreach/2012/0104_2012.html",
                                     "fact": "会前采集，结果在会议纪要公布的次日公开；不得用于本次会议后首个中国开盘的已知共识。",
                                     "strategy_input": False},
        "weekly_and_event_page_difference": {"event": "CPI_20230214", "weekly_headline_mom_consensus": .5,
                                             "event_page_headline_mom_consensus": .4, "core_mom_both": .3,
                                             "weekly_url": next(r["source_url"] for r in facts if r["event_id"] == "CPI_20230214"),
                                             "event_page_url": "https://www.cmegroup.com/education/events/econoday/2023/02/feed559226.html",
                                             "meaning": "相同Econoday来源的周报与历史事件页数值不同；可能涉及更新时间或记录口径。没有精确更新日志，不把差异确定归因于某条新闻，不用事件页数字覆盖固定周报输入。"},
        "limits": ["周报共识不等于全市场持仓加权预期。", "本轮数字相对固定周报版本计算；没有完整的公告前最后时点共识。",
                   "利率变化及指数收益用于历史机制解释，不能证明某条消息的独立贡献。", "收益已见，任何解释不再增设买入或退出条件。"],
    }
    save(OUT / "expectation_clock_cards.json", cards)
    return rows


def make_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10})
    facts = read(OUT / "expectation_comparison.json")
    cpi = [r for r in facts if r["kind"] == "CPI"]
    triggers = [r for r in facts if r["trigger"]]
    result = read(OUT / "result.json")
    old = read(PRIOR / "result.json")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [1, 1, 1.1]})
    fig.suptitle("510300历史预期锚：放慢、超预期与可成交收益", x=.08, ha="left", fontsize=17)
    x, w = np.arange(14), .36
    axes[0].bar(x-w/2, [r["core_mom_expected"] for r in cpi], w, color="#a3adb6", label="公告前周报共识")
    axes[0].bar(x+w/2, [r["core_mom_actual"] for r in cpi], w, color="#237d86", label="BLS当次首次发布值")
    axes[0].set_xticks(x, [r["event_date"][2:] for r in cpi], rotation=35, ha="right", fontsize=9)
    axes[0].set_title("14次核心CPI环比：比较同一预期来源，保留全部公告", loc="left", fontsize=13)
    axes[0].set_ylabel("季调月环比（%）")
    axes[0].legend(frameon=False, ncol=2)
    x, w = np.arange(len(triggers)), .25
    for offset, key, label, color in [(-w, "pre_open_gap", "入场前开盘缺口", "#a3adb6"),
                                      (0, "net_return5", "入场后5日净收益", "#c28a54"),
                                      (w, "net_return20", "入场后20日净收益", "#237d86")]:
        axes[1].bar(x + offset, [r[key] for r in triggers], w, label=label, color=color)
    axes[1].set_xticks(x, [r["entry_date"][:10] for r in triggers])
    axes[1].axhline(0, color="#9da5ab", lw=.8)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("4次低于核心通胀共识：固定20日净收益两正两负，均值仅+0.05%", loc="left", fontsize=13)
    axes[1].set_xlabel("首个中国开盘；缺口未计入新入场盈利，事件窗口不等于完整账户")
    axes[1].legend(frameon=False, ncol=3, fontsize=9)
    for source, key, label, color in [
        (result, "WEEK_AHEAD_SURPRISE_200000_STRESS", "周报预期偏差：夏普0.017", "#237d86"),
        (old, "US_NEWS_200000_STRESS", "原放慢／减速规则：夏普−0.528", "#a3adb6"),
    ]:
        ledger = pd.read_parquet(ROOT / source["ledger_paths"][key])
        axes[2].plot(ledger.date, ledger.equity / 200000 - 1, label=label, color=color, lw=1.8)
    axes[2].axhline(0, color="#9da5ab", lw=.8)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_title("同一301交易日、20万元完整账户；两个规则均未达标", loc="left", fontsize=13)
    axes[2].set_ylabel("累计净收益")
    axes[2].set_xlabel("含空仓、压力成本及原风险约束；历史重建，不是独立验证")
    axes[2].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
    fig.tight_layout(rect=[.01, .01, .99, .96], h_pad=1.8)
    fig.savefig(OUT / "预期锚_事件收益_账户结果.png", dpi=160, facecolor="white")
    plt.close(fig)


def make_report():
    result = read(OUT / "result.json")
    facts = read(OUT / "expectation_comparison.json")
    cpi, fed = [r for r in facts if r["kind"] == "CPI"], [r for r in facts if r["kind"] == "FOMC"]
    cycle = read(OUT / "account_cycles.json")["WEEK_AHEAD_SURPRISE_200000_STRESS"]
    intraday = {r["event_id"]: r for r in read(OUT / "fomc_expectation_and_repricing.json")}
    a = result["accounts"]["WEEK_AHEAD_SURPRISE_200000_STRESS"]
    def link(label, filename):
        return f"[{label}](<{(OUT / filename).as_posix()}>)"
    def ref(event_id, label):
        item = next(r for r in facts if r["event_id"] == event_id)
        return f"[{label}]({item['source_url']})"
    text = [
        "# 历史发现：周报预期锚、市场重估与指数可成交收益", "",
        f"本轮补齐22份Econoday历史周报，覆盖原14次CPI、9次FOMC。加入预期锚后，20万元完整账户净利润{a['net_profit']:.2f}元、"
        f"净夏普{a['net_sharpe']:.3f}、年化{a['cagr']:.4%}，仍未达到夏普1.2且年化10%的目标。"
        "新候选固定拒绝，原两项失败结果保留。", "",
        "最有用的进展是减少概念误判：前月改善、相对周报共识的偏差、公告前最新定价及随后市场反应是不同量。"
        "预期锚确实改变了信号，但这些美国消息在510300首个可成交开盘之后，仍没有形成足够收益。", "",
        "## 先说明本轮测到哪一层预期", "",
        "采用同一来源Econoday周报中的前瞻共识，不拼接不同机构的有利数字。每个事件取之前最近一篇覆盖该事件的周报。"
        "多数正文日期为周五，网页标题日期为下周一，索引日期偶有偏差；三者均留存，使用较晚日期的纽约日终仍早于公告。"
        "当前下载的旧网页是历史重建，不是本机当时保存的不可变快照。", "",
        "周报预期会滞后于之后的新信息，因此本轮称‘相对周报共识的偏差’，不称准确的公告瞬间外生冲击。"
        "纽约联储交易商调查虽在会前进行，结果却在纪要公布后才公开，未放入本次会议后首个开盘的信号。"
        "[纽约联储公开时间说明](https://www.newyorkfed.org/newsevents/events/regional_outreach/2012/0104_2012.html)", "",
        "## 哪些判断被预期锚改变", "",
        "- **2022年3月核心通胀放慢，已经在预期内。** 当次核心环比从0.6%降至0.5%，周报共识也是0.5%。"
        "原规则把它算作改善触发，新规则没有。" + ref("CPI_20220310", "3月公告前周报") , "",
        "- **2022年12月和2023年2月的加息减速，亦已在周报预期中。** 周报分别预计50和25基点，实际相同；"
        "不能仅凭它们小于上次步幅就叫额外宽松。" + ref("FOMC_20221214", "12月周报") + "、" + ref("FOMC_20230201", "次年2月会议前周报"), "",
        "- **真正低于本轮核心通胀共识的有四次，仍有不同构成。** 4月核心环比0.3%低于预期0.5%，但整体1.2%高于预期1.1%；"
        "8月、11月和12月则是核心与整体环比均低于各自周报共识。这里仅解释构成，没有据此追加‘两项同时低于预期’的筛选。", "",
        "|CPI公告日|整体环比：周报→实际|核心环比：周报→实际|相对同版前月核心变化|核心预期偏差|", "|---|---:|---:|---:|---:|",
    ]
    for r in cpi:
        text.append(f"|{r['event_date']}|{r['headline_mom_expected']:.1f}%→{r['headline_mom_actual']:.1f}%|{r['core_mom_expected']:.1f}%→{r['core_mom_actual']:.1f}%|{r['core_mom_actual']-r['core_mom_previous_same_release']:+.1f}个百分点|{r['core_mom_surprise_pp']:+.1f}个百分点|")
    text += ["", "每行均有周报原文、当时BLS版本与时间依据，见" + link("逐公告比较表", "expectation_comparison.json") + "。"
             "BLS实际值复用上一轮原始发布数，不用后来修订替换。2023年2月的前月核心环比为当次已修订的0.4%。", "",
        "## 周报之后，市场还在重新定价", "",
        "2022年6月周报预期加息50基点，实际75基点，看上去多了25基点。但Refinitiv在6月13日的原文记录，"
        "其IRPR系统给出的75基点概率当日由20%更新至69%。这说明周报后、决议前已经发生重估；"
        "缺精确更新分钟不影响它早于6月15日决议这一顺序。"
        "[LSEG／Refinitiv当时记录](https://lipperalpha.refinitiv.com/2022/06/will-the-fed-raise-by-75-basis-points/)", "",
        "复用已有旧金山联储USMPD原始变化字段，在声明前10分钟至记者会开始后60分钟的100分钟观察窗口内，"
        "6月15日1年OIS反而下降11.75基点，标普500上涨1.32%。与周前比‘多加25基点’和当场后续路径重估偏缓和可以同时成立。"
        "OIS仍含风险溢价及混合信息；此对照没有识别纯政策冲击，也不把美国股债组合自动转成A股买卖。"
        "[旧金山联储数据库](https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/)", "",
        "|FOMC日期|周报预计步幅|实际步幅|相对周报偏差|完整观察窗1年OIS变化|同窗标普500|", "|---|---:|---:|---:|---:|---:|",
    ]
    for r in fed:
        q = intraday[r["event_id"]]
        text.append(f"|{r['event_date']}|{r['expected_step_bp']:.0f}bp|{r['actual_step_bp']:.0f}bp|{r['target_surprise_bp']:+.0f}bp|{q['ois1y_change_bp']:+.2f}bp|{q['sp500_window_return_pct']:+.2f}%|")
    text += ["",
        "特别是12月与次年2月，步幅都符合周报预期，但1年OIS分别上升4.69与下降4.88基点。"
        "当次动作相同地‘符合预期’，不代表未来路径信息相同。12月SEP提高2023年末利率预测的原始证据在上一轮保留。", "",
        "这些高频字段来自已保存的当前历史数据库版本，本轮只用于解释，未新增交易输入，也未宣称已具备当年的实时报价访问。"
        "周报多处将会议排在15:00，本轮继续使用美联储原声明的14:00纽约时钟；全部交易仍在随后的中国开盘。", "",
        "还有一个版本差异：2023年2月14日整体CPI环比共识，前周周报为0.5%，CME转载的Econoday历史事件页为0.4%；"
        "核心环比共识两处均为0.3%。目前没有精确更新日志，不能确定差异由何时何种信息造成，也不能悄悄以事件页覆盖固定周报输入。"
        + ref("CPI_20230214", "周报") + "、[CME历史事件页](https://www.cmegroup.com/education/events/econoday/2023/02/feed559226.html)", "",
        "## 正面预期偏差，还剩多少指数收益", "",
        "固定候选只有一条：核心CPI环比低于周报共识，或目标利率上限低于周报预期。实际4次CPI触发，FOMC无触发。"
        "公告后首个中国开盘进入，持有20个开盘区间；已有持仓时不加仓、不顺延。没有按结果修改月份、阈值或持有期。", "",
        "|触发公告|首个入场开盘|此前开盘缺口|入场后5日净收益|入场后20日净收益|", "|---|---|---:|---:|---:|",
    ]
    for r in facts:
        if r["trigger"]:
            text.append(f"|{r['event_date']}|{r['entry_date'][:10]}|{r['pre_open_gap']:+.2%}|{r['net_return5']:+.2%}|{r['net_return20']:+.2%}|")
    text += ["",
        "这4次的美国2年名义利率同日平均下降15基点，但510300开盘后20日净收益两正两负、均值仅+0.05%；"
        "5日均值−0.32%。因此外部利率确有反应与中国指数后续收益不足并存。开盘缺口均值+0.63%主要受11月事件影响，"
        "不能把它计入公告后开盘入场收益，也不能全归美国消息。", "",
        "全部14次CPI和9次议息均保留在分组文件中。4月核心好于预期与能源压力同在；8月实质收益率没有随名义利率下降；"
        "11月国内政策优化方向与美国CPI同晚公开；12月处于国内活动约束改变而实际履约能力仍受感染冲击的阶段。"
        "这些是已有原文支持的竞争解释，没有用事后盈利给阶段命名或训练新的买入条件。", "",
        "## 完整账户与瓶颈", "",
        "相同2022年1月1日至2023年3月31日日历，共301个交易日；现金及夏普无风险基准均为0，年化交易日242。"
        "保留50%最大目标仓位、条件5日ES95预算2.5%、−10%跳空预算5%、10%回撤触发、T+1和整手。"
        "压力成本为单边佣金万4且最低5元、单边滑点0.1%。", "",
        "|账户|净夏普|净年化|最大回撤|净利润|周期数|", "|---|---:|---:|---:|---:|---:|",
    ]
    for capital in [200000, 20000]:
        q = result["accounts"][f"WEEK_AHEAD_SURPRISE_{capital}_STRESS"]
        text.append(f"|{capital/10000:.0f}万元|{q['net_sharpe']:.3f}|{q['cagr']:.4%}|{q['max_drawdown']:.2%}|{q['net_profit']:+,.2f}元|{q['entry_fills']}|")
    text += ["",
        "20万元账户的原持仓毛利润955.10元，手续费与滑点948.42元，净利润6.68元。"
        "费用几乎消耗了这点毛利润；但即使退还原路径全部成本，夏普也只有0.136、年化0.384%，距离目标仍很远。"
        "成本是本轮接近盈亏平衡的直接原因，毛优势不足则是无法靠降费达到目标的原因。退款只是解释用界限，不是另一条可执行策略。", "",
        "平均收盘仓位9.55%，实际持有80个收盘日，217天全日空仓；共有7次风险减仓，未出现非零订单拒绝，也未触发回撤停机。"
        "4次事件本身的20日收益也近乎抵消，不能只归因于低仓位或资金闲置。最大一笔盈利3,056.21元，另一笔亏损2,936.70元，净利润很薄。", "",
        "|20万元实际持仓|退出日|净利润|原仓位退还成本后利润|", "|---|---|---:|---:|",
    ]
    for q in cycle:
        text.append(f"|{q['entry'][:10]}|{q['exit'][:10]}|{q['profit']:+,.2f}元|{q['same_holdings_gross_profit']:+,.2f}元|")
    text += ["", f"![预期锚与账户对照](<{(OUT / '预期锚_事件收益_账户结果.png').as_posix()}>)", "",
        "## 结论的使用范围", "",
        "已经证实的是：原先7个放慢／减速触发中3个其实符合周报预期；引入同源预期锚后只余4次触发。"
        "还证实了：周前预期会被中间新消息改变，符合当次动作预期仍可出现不同未来路径重估。"
        "尚未证实的是：这些外部预期偏差能给510300留下达到目标的可成交优势。", "",
        "只新增这一项候选及20万／2万两个账户。没有延长到五年十年、没有网格搜索，也没有按是否获利挑选公告。"
        "完整公告的收益此前已见，所以本轮属于历史发现，不是独立验证。必要核对已完成：原预期段与实际值、公开时序、保存账本的资金恒等式及T+1、指标重算。", "",
        result["next_historical_question"], "",
        "材料：" + "；".join([link("固定方案", "protocol.json"), link("逐公告事实及原文", "expectation_comparison.json"),
            link("议息预期与市场重估", "fomc_expectation_and_repricing.json"), link("时序原因卡", "expectation_clock_cards.json"),
            link("全部分组描述", "group_descriptions.json"), link("完整结果", "result.json"), link("逐笔周期", "account_cycles.json"),
            link("必要核对", "account_checks.json"), link("22份原周报清单", "source_manifest.json")]) + "。", "",
        f"脚本：[{(ROOT / 'research/historical_index_expectation_anchor_v1.py').name}](<{(ROOT / 'research/historical_index_expectation_anchor_v1.py').as_posix()}>). "
        "已有原文下按 `--run` 计算固定结果、按 `--report` 整理报告；不必重复下载。", "",
        "目标保持进行中。本轮没有前瞻判断或实盘执行。", "",
    ]
    (OUT / "历史发现_预期锚与指数剩余收益.md").write_text("\n".join(text), encoding="utf-8")


def report():
    explanation_evidence()
    r = read(OUT / "result.json")
    r["group_descriptions"] = group_diagnostics(read(OUT / "expectation_comparison.json"))
    r["next_historical_question"] = (
        "比较原14次CPI事件当时国内可知的活动与政策约束，重点解释4次低于共识为何有不同可成交结果；"
        "先复用已有国内政策、PMI和物流证据，区别已解除的行政约束、实际履约能力、需求与价格此前反映。"
        "其余10次保留对照，不按收益命名阶段，不给本轮失败候选追加过滤；没有新的独立信息则结束该机制扩展。"
    )
    r["discovery"] = (
        "22份公告前周报覆盖23事件；3个旧触发符合周报预期，余4个核心通胀低于预期触发。"
        "4个20日净收益均值仅0.0499%，20万元净利润6.68元、夏普0.016809；退还原持仓成本后年化仍仅0.3838%。"
        "6月会前概率已移动、12月和次年2月当次步幅均符合预期而OIS方向相反，说明预期时间与期限维度不可省略。"
    )
    r["report"] = (OUT / "历史发现_预期锚与指数剩余收益.md").relative_to(ROOT).as_posix()
    r["goal_turn_classification"] = "PROGRESS"
    save(OUT / "result.json", r)
    make_plot()
    make_report()
    print("已整理预期时钟、既有高频反应对照、图表和完整报告。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect()
    elif args.preview:
        preview()
    elif args.run:
        run()
    elif args.report:
        report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
