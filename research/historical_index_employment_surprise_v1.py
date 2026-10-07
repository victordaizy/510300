"""按完整历史发布序列拆解美国就业预期差与沪深300反应。"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
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
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import historical_index_reopening_constraints_v1 as execution
from research import factor96_margin_repair_v1 as core

OUT = ROOT / "reports/research/510300_historical_index_employment_surprise_v1"
OLD = ROOT / "reports/research/510300_historical_index_expectation_anchor_v1"
TZ = ZoneInfo("Asia/Shanghai")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(name, value):
    core.save(OUT / name, value)


def collect_expectations():
    """复用已有周报，缺失的固定日期周报仅请求一次。"""
    source_dir = OUT / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_2021 = source_dir / "archive_2021.html"
    if not archive_2021.exists():
        response = requests.get("https://fidelity.econoday.com/articles?cust=fidelityFIPlus&year=2021&lid=0&archive=103", timeout=25)
        response.raise_for_status()
        archive_2021.write_bytes(response.content)
    archive_paths = [archive_2021, OLD / "sources/archive_2022.html", OLD / "sources/archive_2023.html"]
    archives = []
    for path in archive_paths:
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for link in soup.find_all("a", href=True):
            if "byshoweventarticle" not in link["href"]:
                continue
            row_text = link.find_parent("tr").get_text(" ", strip=True)
            match = re.search(r"(\d{1,2}/\d{1,2}/(?:\d{4}|\d{2}))$", row_text)
            if not match:
                continue
            day_text = match.group()
            day = pd.to_datetime(day_text, format="%m/%d/%Y" if len(day_text.split("/")[-1]) == 4 else "%m/%d/%y")
            fid = parse_qs(urlparse(link["href"]).query)["fid"][0]
            archives.append({"source_id": "econoday_" + fid, "archive_date": day.strftime("%Y-%m-%d"), "title": link.get_text(" ", strip=True),
                             "url": urljoin("https://fidelity.econoday.com/", link["href"])})
    plan = []
    for day in read(OUT / "protocol.json")["fixed_release_dates"]:
        options = [r for r in archives if r["archive_date"] < day]
        selected = max(options, key=lambda r: r["archive_date"])
        # 索引把11月4日正文列在11月3日，而标题为11月7日；该文章晚于本次发布。
        # 按此前最近的周报回退，原因在收益计算前记录，不依据涨跌选择来源。
        if day == "2022-11-04":
            selected = next(r for r in options if r["source_id"] == "econoday_561378")
        delay = (pd.Timestamp(day) - pd.Timestamp(selected["archive_date"])).days
        plan.append({"event_date": day, "event_id": "JOBS_" + day.replace("-", ""), **selected, "days_before": delay,
                     "source_plan_status": "ADMISSIBLE_ARCHIVE_DATE" if delay <= 10 else "MISSING_WEEK_AHEAD_ARCHIVE"})
    save("expectation_source_plan.json", plan)
    prior = {r["source_id"]: r for r in read(OUT / "expectation_source_manifest.json")} if (OUT / "expectation_source_manifest.json").exists() else {}

    def fetch(item):
        row = {**item, "retrieved_at": datetime.now(TZ).isoformat()}
        if item["source_plan_status"] != "ADMISSIBLE_ARCHIVE_DATE":
            return {**row, "status": "MISSING_WEEK_AHEAD_ARCHIVE"}
        if item["source_id"] in prior:
            return prior[item["source_id"]]
        existing = OLD / "sources" / (item["source_id"] + ".html")
        target = source_dir / (item["source_id"] + ".html")
        try:
            if existing.exists():
                target = existing
                status = "REUSED_EXISTING_BYTES"
            elif target.exists():
                status = "REUSED_LOCAL_BYTES"
            else:
                response = requests.get(item["url"], timeout=25)
                response.raise_for_status()
                target.write_bytes(response.content)
                status = "FETCHED"
            data = target.read_bytes()
            soup = BeautifulSoup(data.decode("utf-8"), "html.parser")
            for tag in soup.select("script,style"):
                tag.decompose()
            content = soup.get_text("\n", strip=True)
            assert "Global Economics" in content, "返回页面未含周报"
            if target.parent == source_dir:
                target.with_suffix(".txt").write_text(content, encoding="utf-8")
            return {**row, "status": status, "path": target.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(data).hexdigest()}
        except Exception as error:
            return {**row, "status": "UNAVAILABLE", "error": str(error)}

    manifest = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, item) for item in plan]):
            row = future.result()
            manifest.append(row)
            save("expectation_source_manifest.json", sorted(manifest, key=lambda r: r["event_date"]))
            print(row["event_date"], row["source_id"], row["status"], flush=True)


def preview_expectations():
    for row in read(OUT / "expectation_source_manifest.json"):
        print("\n发布日", row["event_date"], "周报日", row["archive_date"], row["status"])
        if "path" not in row:
            continue
        soup = BeautifulSoup((ROOT / row["path"]).read_text(encoding="utf-8"), "html.parser")
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        for match in re.finditer(r"US Employment Situation", text):
            print(text[match.start():match.start() + 1500])


def parse_expectations():
    rows = []
    for source in read(OUT / "expectation_source_manifest.json"):
        row = {**source, "expectation_available": False}
        if "path" not in source:
            rows.append(row)
            continue
        soup = BeautifulSoup((ROOT / source["path"]).read_text(encoding="utf-8"), "html.parser")
        content = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        header = re.search(r"Global Economics\s*-\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", content)
        title = re.search(r"Global Economics\s+(\d+)\s+(\d+),\s*(\d{4})", soup.title.get_text(" ", strip=True))
        assert header and title, source["event_id"]
        body_day = pd.to_datetime(header[1], format="%B %d, %Y")
        title_day = pd.Timestamp(year=int(title[3]), month=int(title[1]), day=int(title[2]))
        latest = max(body_day, title_day, pd.Timestamp(source["archive_date"]))
        available = latest.tz_localize("America/New_York") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        release = pd.Timestamp(source["event_date"]).tz_localize("America/New_York") + pd.Timedelta(hours=8, minutes=30)
        row.update(body_date=body_day.strftime("%Y-%m-%d"), title_date=title_day.strftime("%Y-%m-%d"),
                   expectation_available_upper=available.isoformat(), release_at=release.isoformat(),
                   public_at_shanghai=release.tz_convert("Asia/Shanghai").isoformat(),
                   immutable_historical_snapshot=False, expectation_measure="周前共识，不是公告前最后一分钟共识")
        if available >= release:
            row["expectation_status"] = "AFTER_RELEASE_TITLE_DATE"
            rows.append(row)
            continue
        match = re.search(r"US Employment Situation for ([A-Za-z]+)(.*?)(?=US: [A-Z]|US [A-Z]|Canadian [A-Z]|Canada [A-Z]|Important Legal Notice|$)", content)
        assert match, source["event_id"]
        body = match.group()
        economic_month = pd.Timestamp(source["event_date"]).to_period("M") - 1
        month = economic_month.strftime("%B")
        assert match[1] == month, (source["event_id"], match[1], month)
        patterns = {
            "payroll_expected": r"Consensus Forecast:\s*Change in Nonfarm Payrolls\s*:\s*([+-]?[\d,]+)",
            "unemployment_expected": r"Consensus Forecast:\s*Unemployment Rate\s*:\s*([\d.]+)%",
            "earnings_mom_expected": r"Consensus Forecast:\s*Average Hourly Earnings M/M\s*:\s*([\d.]+)%",
            "earnings_yoy_expected": r"Consensus Forecast:\s*Average Hourly Earnings Y/Y\s*:\s*([\d.]+)%"
        }
        for key, pattern in patterns.items():
            found = re.search(pattern, body)
            row[key] = float(found[1].replace(",", "")) if found else None
        if source["event_date"] == "2022-02-04":
            earnings = re.search(r"Average hourly earnings are expected to rise a substantial ([\d.]+) percent on the month and 5 tenths on the year to ([\d.]+) percent", body)
            assert earnings, source["event_id"]
            row.update(earnings_mom_expected=float(earnings[1]), earnings_yoy_expected=float(earnings[2]), earnings_expectation_from_narrative=True)
        assert all(row[k] is not None for k in ["payroll_expected", "earnings_mom_expected", "earnings_yoy_expected"]), (source["event_id"], body)
        row.update(expectation_available=True, expectation_status="ADMITTED_WEEK_AHEAD", evidence_excerpt=body,
                   economic_month=economic_month.strftime("%Y-%m"))
        rows.append(row)
    save("expectations.json", rows)
    print("已提取周前共识", sum(r["expectation_available"] for r in rows), "/", len(rows))


def bls_text(day):
    paths = sorted((OUT / "sources").glob("bls_" + day.replace("-", "") + "_*.txt"))
    lines = {}
    for path in paths:
        for match in re.finditer(r"^L(\d+): ?(.*)$", path.read_text(encoding="utf-8"), re.M):
            number = int(match[1])
            if 185 <= number <= 460:
                lines[number] = match[2]
    text = re.sub(r"\s+", " ", " ".join(lines[i].strip() for i in sorted(lines)))
    return text, [p.relative_to(ROOT).as_posix() for p in paths]


def parse_actuals():
    facts = []
    for day in read(OUT / "protocol.json")["fixed_release_dates"]:
        text, paths = bls_text(day)
        day_label = pd.Timestamp(day).strftime("%B %d, %Y").replace(" 0", " ")
        assert day_label in text, (day, "未核到原始发布日期")
        payroll = re.search(r"Total nonfarm payroll employment (?:rose|increased) by ([\d,]+)", text)
        unemployment = re.search(r"unemployment rate [^.]*?([\d]+\.[\d]+) percent", text)
        participation = re.search(r"labor force participation rate.*?([\d]+\.[\d]+) percent\b", text)
        assert payroll and unemployment and participation, day
        wage = re.search(r"[Aa]verage hourly earnings for all employees on private nonfarm payrolls(.{0,450})", text)
        assert wage, day
        w = wage.group()
        # 后半段的生产及非管理人员工资属于另一人群，不得代替全体私人雇员工资。
        subgroup = re.search(r"average hourly earnings of private", w)
        if subgroup:
            w = w[:subgroup.start()]
        yoy = re.search(r"Over the past 12 months, average hourly earnings have increased by ([\d.]+) percent", w)
        move = re.search(r"(?:increased|rose) by (\d+) cents(?:, or ([\d.]+) percent,)? to \$([\d.]+)", w)
        if move:
            cents, explicit, level = move[1], move[2], move[3]
        else:
            special = re.search(r"at \$([\d.]+).*?\(\+(\d+) cent\)", w)
            assert special, (day, w)
            level, cents, explicit = special[1], special[2], None
        current = Decimal(level.rstrip("."))
        prior = current - Decimal(cents) / 100
        mom = float(Decimal(explicit)) if explicit is not None else float(((current / prior - 1) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        revision = re.search(r"(?:With these revisions,|After revision,).*?combined (?:is|were) ([\d,]+) (higher|lower)", text)
        if day == "2022-02-04":
            assert "2021 combined is 709,000 higher" in text
            revision_value = 709000
            revision_excerpt = "年度季调模型与基准更新：2021年11、12月合计上修709000，6、7月合计下修807000；不等于近期需求突然新增709000。"
        else:
            assert revision, (day, "未提取两月修订")
            revision_value = int(revision[1].replace(",", "")) * (1 if revision[2] == "higher" else -1)
            revision_excerpt = revision.group()
        release = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=8, minutes=30)
        row = {"event_id": "JOBS_" + day.replace("-", ""), "event_date": day,
               "economic_month": (pd.Timestamp(day).to_period("M") - 1).strftime("%Y-%m"),
               "release_at": release.isoformat(), "public_at_shanghai": release.tz_convert("Asia/Shanghai").isoformat(),
               "payroll_initial": int(payroll[1].replace(",", "")), "unemployment_initial": float(unemployment[1]),
               "participation_initial": float(participation[1]), "earnings_mom_initial": mom,
               "earnings_mom_method": "原报告明示" if explicit is not None else "原报告当期美元工资及增量计算并四舍五入至一位百分数",
               "earnings_yoy_initial": float(yoy[1]), "earnings_level_initial": float(current),
               "earnings_prior_level_same_release": float(prior), "previous_two_months_revision": revision_value,
               "annual_establishment_benchmark": day in ["2022-02-04", "2023-02-03"],
               "household_population_control_change": day in ["2022-02-04", "2023-02-03"],
               "wage_excerpt": w, "revision_excerpt": revision_excerpt,
               "participation_excerpt": text[participation.start():participation.start() + 550],
               "source_url": "https://www.bls.gov/news.release/archives/empsit_" + pd.Timestamp(day).strftime("%m%d%Y") + ".htm",
               "source_capture_paths": paths, "source_representation": "浏览工具的官方归档页面文字片段，非原HTML字节；每次发布独立初值",
               "current_revised_series_used": False}
        facts.append(row)
    save("initial_release_facts.json", facts)
    print(pd.DataFrame(facts)[["event_date", "payroll_initial", "earnings_mom_initial", "earnings_yoy_initial", "unemployment_initial", "participation_initial", "previous_two_months_revision"]].to_string(index=False))


def analyze():
    actuals = read(OUT / "initial_release_facts.json")
    expected = {r["event_id"]: r for r in read(OUT / "expectations.json")}
    protocol = read(execution.OUT / "protocol.json")
    protocol["account_calendar"][1] = "2023-05-31"
    market, features, dividends, engine, _ = execution.inputs(protocol)
    rates = pd.read_parquet(ROOT / "reports/research/510300_historical_index_external_reopening_v1/treasury_curves.parquet").sort_values("date")
    dxy = pd.read_parquet(ROOT / "reports/research/510300_rmb_residual_state_v1/inputs/dxy.parquet").sort_values("date")
    rows, windows, clocks = [], [], []
    for actual in actuals:
        row = {**actual}
        expect = expected[row["event_id"]]
        row["expectation_available"] = expect["expectation_available"]
        if row["expectation_available"]:
            for name in ["payroll", "earnings_mom", "earnings_yoy", "unemployment"]:
                row[name + "_expected"] = expect[name + "_expected"]
                row[name + "_surprise"] = round(row[name + "_initial"] - expect[name + "_expected"], 8) if expect[name + "_expected"] is not None else None
            row.update(expectation_source_url=expect["url"], expectation_source_path=expect["path"],
                       expectation_available_upper=expect["expectation_available_upper"])
            payroll_up, wage_up = row["payroll_surprise"] > 0, row["earnings_mom_surprise"] > 0
            row["group"] = ("人数超预期" if payroll_up else "人数未超预期") + "／" + ("工资超预期" if wage_up else "工资未超预期")
        else:
            row["group"] = "周前共识缺失"
        available = pd.Timestamp(row["public_at_shanghai"]) + pd.Timedelta(minutes=5)
        opens = pd.to_datetime(market.date).dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        entry_idx = int(opens.loc[opens > available].index[0])
        entry_day = market.date.iloc[entry_idx]
        clock = {"event_id": row["event_id"], "title": row["event_date"] + "美国就业发布",
                 "public_at": row["public_at_shanghai"], "available_at": available.isoformat(),
                 "entry_idx": entry_idx, "entry_date": entry_day, "entry_at": opens.iloc[entry_idx].isoformat(),
                 "previous_A_close_date": market.date.iloc[entry_idx - 1]}
        assert opens.iloc[entry_idx] > available
        clocks.append(clock)
        row.update(entry_date=entry_day, previous_A_close_date=market.date.iloc[entry_idx - 1],
                   A_entry_calendar_gap_days=(entry_day - pd.Timestamp(row["event_date"])).days,
                   pre_entry_gap=market.open.iloc[entry_idx] / market.close.iloc[entry_idx - 1] - 1)
        for horizon in [5, 20]:
            window = execution.fixed_event_window(market, dividends, engine, clock, horizon)
            windows.append(window)
            assert window["status"] == "两端成交", window
            row[f"net_return{horizon}"] = window["net_return"]
            row[f"gross_return{horizon}"] = window["gross_return"]
        event_date = pd.Timestamp(row["event_date"])
        today = rates.loc[rates.date.eq(event_date)].iloc[0]
        yesterday = rates.loc[rates.date.lt(event_date)].iloc[-1]
        row["rate_previous_date"] = yesterday.date
        for name in ["nominal2", "nominal10", "real10", "inflation_compensation10"]:
            row[name + "_change_bp"] = round((today[name] - yesterday[name]) * 100, 6)
        dollar_today = dxy.loc[dxy.date.eq(event_date)].iloc[0]
        dollar_previous = dxy.loc[dxy.date.lt(event_date)].iloc[-1]
        row["dxy_change"] = dollar_today.dxy / dollar_previous.dxy - 1
        row["rate_dollar_reaction_scope"] = "前美国交易日到发布日收盘；包含全部当日新闻，只描述共同反应"
        row["rate_dollar_used_for_entry"] = False
        row["rate_historical_publication_clock"] = "未核准曲线首次发布时间；日度观察日期不能当作当年开盘前获取凭证"
        row["index_response_scope"] = "510300跟踪沪深300的ETF可成交近似，非沪深300官方指数本身；开盘前跳空已排除"
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "event_comparison.parquet", index=False)
    save("event_comparison.json", rows)
    save("event_windows.json", windows)
    save("event_clocks.json", clocks)
    groups = []
    order = ["人数超预期／工资超预期", "人数超预期／工资未超预期", "人数未超预期／工资超预期", "人数未超预期／工资未超预期", "周前共识缺失"]
    for label in order:
        part = frame.loc[frame.group.eq(label)]
        record = {"group": label, "n": len(part), "events": part.event_id.tolist()}
        for name in ["nominal2_change_bp", "real10_change_bp", "inflation_compensation10_change_bp", "dxy_change", "net_return5", "net_return20"]:
            record["mean_" + name] = float(part[name].mean()) if len(part) else None
            record["median_" + name] = float(part[name].median()) if len(part) else None
        record["rate_up_count"] = int(part.nominal2_change_bp.gt(0).sum())
        for h in [5, 20]:
            record[f"wins{h}"] = int(part[f"net_return{h}"].gt(0).sum())
            record[f"min_net{h}"] = float(part[f"net_return{h}"].min()) if len(part) else None
            record[f"max_net{h}"] = float(part[f"net_return{h}"].max()) if len(part) else None
        groups.append(record)
    save("descriptive_groups.json", groups)
    print(frame[["event_date", "group", "payroll_surprise", "earnings_mom_surprise", "previous_two_months_revision", "nominal2_change_bp", "real10_change_bp", "dxy_change", "net_return5", "net_return20"]].to_string(index=False))
    print("分组描述", json.dumps(groups, ensure_ascii=False, default=str))


def finish():
    """将固定分组的发现写成报告；不把事件窗口拼成策略账户。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    frame = pd.read_parquet(OUT / "event_comparison.parquet")
    groups = read(OUT / "descriptive_groups.json")
    by_day = frame.set_index("event_date")
    now = datetime.now(TZ).isoformat()
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 8.4), sharey=True, gridspec_kw={"width_ratios": [1.2, 1.1, 1.0, 1.8]})
    fig.patch.set_facecolor("#f7f8fa")
    y = np.arange(len(frame))
    colors = {True: "#216b86", False: "#bd653e"}
    columns = [("payroll_surprise", .0001, "就业人数与周前共识之差\n万人"),
               ("earnings_mom_surprise", 1, "工资环比与周前共识之差\n百分点"),
               ("nominal2_change_bp", 1, "当日美国2年期收益率变化\n基点")]
    for ax, (column, scale, title) in zip(axes, columns):
        values = frame[column] * scale
        for i, value in enumerate(values):
            if pd.isna(value):
                ax.text(0, i, "共识缺失", color="#90959d", fontsize=8, ha="center", va="center")
            else:
                ax.barh(i, value, height=.5, color=colors[value >= 0], alpha=.92)
                ax.text(value, i, f" {value:+.1f}" if value >= 0 else f"{value:+.1f} ", va="center", ha="left" if value >= 0 else "right", fontsize=8)
        lo = min(0., float(values.min()))
        hi = max(0., float(values.max()))
        pad = (hi - lo or .1) * .26
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title, loc="left", fontsize=11, pad=16)
    axes[3].barh(y - .16, frame.net_return5 * 100, height=.27, color="#afbfcb", label="5个交易日净收益")
    axes[3].barh(y + .16, frame.net_return20 * 100, height=.27, color="#355a75", label="20个交易日净收益")
    for i, value in enumerate(frame.net_return20 * 100):
        axes[3].text(value, i + .16, f" {value:+.2f}%" if value >= 0 else f"{value:+.2f}% ", ha="left" if value >= 0 else "right", va="center", fontsize=8)
    axes[3].set_xlim(-12.8, 12.4)
    axes[3].set_title("510300下一可成交开盘后的净收益\n百分比；含压力成本与分红", loc="left", fontsize=11, pad=16)
    axes[3].legend(loc="lower center", bbox_to_anchor=(.5, -.125), ncol=1, frameon=False, fontsize=9)
    axes[0].set_yticks(y, frame.event_date)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.axvline(0, color="#8a919a", linewidth=.75)
        ax.grid(axis="y", color="#e3e7eb", linewidth=.6)
        ax.set_axisbelow(True)
        ax.set_facecolor("#f7f8fa")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#c6cbd1")
    fig.suptitle("就业超预期后的利率与指数收益：方向并不一致", x=.075, y=.985, ha="left", fontsize=19, fontweight="bold", color="#1d3445")
    fig.text(.075, .932, "2022年1月至2023年2月全部14次发布｜11次周前共识可用｜BLS归档初值｜当日率汇反应仅供历史解释", fontsize=11, color="#596876")
    fig.text(.075, .029, "事件窗口使用510300的开盘成交近似；排除买入前跳空，保留缺失与反例。窗口会重叠，不能据此计算全账户夏普。", fontsize=10, color="#596876")
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.82, wspace=.18)
    figure = OUT / "就业信息_利率与指数反应.png"
    fig.savefig(figure, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    cause_cards = [
        {"event_date": "2022-02-04", "known_facts": "当月非农46.7万，工资环比0.7%；前两月合计上修70.9万。BLS说明年度基准和季调模型更新同时使更早的6、7月下修80.7万。人口控制更新造成的参与率上升0.3个百分点，剔除后实际当月参与率持平。",
         "interpretation": "headline强劲有真实就业恢复与统计重估两个来源；修订不能全部当成当月新增需求。",
         "identified_causal_share": False, "source_url": by_day.loc["2022-02-04", "source_url"]},
        {"event_date": "2022-09-02", "known_facts": "当月非农31.5万、周前共识29.3万；工资环比0.3%、共识0.4%；前两月下修10.7万。家庭调查就业增加44.2万、劳动力增加78.6万，参与率升0.3个百分点，失业率升0.2个百分点。",
         "interpretation": "更高失业率伴随更多人进入劳动力市场及就业增加，不能直接等同于裁员恶化；人数超预期中同时包含工资缓和与旧数据下修的信息。",
         "identified_causal_share": False, "source_url": by_day.loc["2022-09-02", "source_url"]},
        {"event_date": "2022-11-04", "known_facts": "机构调查非农增加26.1万；家庭调查就业减少32.8万、失业率升至3.7%，高于周前共识3.6%。工资同比4.7%符合共识，但环比0.4%高于0.3%共识。同日Collins公开强调政策重心由加息速度转向所需利率水平，并须权衡过度紧缩风险。",
         "interpretation": "岗位与就业人数的不同调查口径、不同工资比较期及政策言论并存。日线只能说明混合信息，不能把当天利率下降归因于一项工资读数或一位官员。",
         "identified_causal_share": False, "source_url": by_day.loc["2022-11-04", "source_url"],
         "competing_policy_source": "https://www.bostonfed.org/news-and-events/speeches/2022/perspectives-on-the-economy-and-monetary-policy.aspx"},
        {"event_date": "2023-02-03", "known_facts": "非农51.7万、高于18.5万周前共识，工资环比0.3%与同比4.4%均符合共识；前两月上修7.1万。政府就业增7.4万，其中州政府教育增加3.5万，BLS称后者反映罢工人员复岗。",
         "interpretation": "新增人数可增加增长和利率维持高位的判断，但不能称工资也超预期；部分增量来自复岗，且恰逢年度基准及行业分类更新。",
         "identified_causal_share": False, "source_url": by_day.loc["2023-02-03", "source_url"]}
    ]
    save("upstream_cause_cases.json", {"selection_purpose": "对固定全序列中的矛盾和统计构成作解释；案例不生成新筛选或交易规则", "cases": cause_cards})

    def file_link(label, path):
        return f"[{label}](<{Path(path).resolve().as_posix()}>)"

    group_table = ["| 公告构成 | 次数 | 2年收益率上涨次数 | 平均5日净收益 | 平均20日净收益 | 20日盈利次数 |",
                   "|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        if g["n"]:
            group_table.append(f"| {g['group']} | {g['n']} | {g['rate_up_count']}/{g['n']} | {g['mean_net_return5']:+.2%} | {g['mean_net_return20']:+.2%} | {g['wins20']}/{g['n']} |")
        else:
            group_table.append(f"| {g['group']} | 0 | — | — | — | — |")
    table = ["| 发布日 | 非农初值/共识（万） | 工资环比初值/共识 | 前两月修订（万） | 2年率变化 | 下一A股开盘 | 5日净收益 | 20日净收益 |",
             "|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in frame.to_dict("records"):
        payroll = f"{r['payroll_initial']/10000:.1f}/" + (f"{r['payroll_expected']/10000:.1f}" if r["expectation_available"] else "缺失")
        wage = f"{r['earnings_mom_initial']:.1f}%/" + (f"{r['earnings_mom_expected']:.1f}%" if r["expectation_available"] else "缺失")
        table.append(f"| {r['event_date']} | {payroll} | {wage} | {r['previous_two_months_revision']/10000:+.1f} | {r['nominal2_change_bp']:+.0f}bp | {pd.Timestamp(r['entry_date']):%Y-%m-%d} | {r['net_return5']:+.2%} | {r['net_return20']:+.2%} |")
    mixed = frame.loc[frame.group.eq("人数超预期／工资未超预期")]
    report = [
        "# 历史发现：就业信息的构成与沪深300整体反应",
        f"生成时间：{now}。研究对象为沪深300整体传导，交易收益使用跟踪ETF 510300的开盘近似；执行范围仍为510300.SH与人民币现金。",
        "**本轮发现了上游信息为什么不能简单贴上利多或利空标签，但没有发现新的指数买入优势。** 2022年1月至2023年2月共14次就业发布全部保留，11次有合格的周前共识。把人数与工资拆开后，同组事件仍有明显正反例；不增加参数、不挑月份、不反号构造策略。",
        "## 对指数研究的实际含义",
        "美国就业信息主要通过全球利率、美元和资产配置影响沪深300，不能直接代表沪深300自身的盈利变化。指数还同时承受国内整体订单、融资需求、政策兑现和风险补偿的变化。此次测量的是相同外部新闻构成之后的整体指数结果，不扩展到个股选取。",
        "分析链条保留四层：就业读数的来源（需求、劳动供给、复岗、统计重估）→相对当时预期的增量→率汇及其他信息共同定价→A股可成交价格之后的收益。日线能够描述共同变化，不能分离某一新闻的净因果贡献。",
        "## 三个有实际含义的历史区别",
        "**一，失业率上升不必然来自就业下降。** 2022年9月2日发布的8月数据中，家庭调查就业增加44.2万，劳动力增加78.6万，参与率升0.3个百分点；失业率仍升0.2个百分点。这里劳动供给扩张与就业增加并存。机构调查非农仍比周前共识多2.2万，但工资环比低0.1个百分点，前两月非农又下修10.7万。不能只用‘非农超预期’解释这一整包信息。[BLS原报告](https://www.bls.gov/news.release/archives/empsit_09022022.htm)、[8月26日Econoday周报](https://fidelity.econoday.com/byshoweventarticle?fid=556570&cust=fidelityFIPlus&year=2022&lid=0)。",
        "当天2年美债收益率下降11bp，510300下一个可成交开盘后的5日净收益+1.90%，20日却为−7.26%。这说明工资与供给信息可以改变对新闻的理解，但利率缓和没有保证指数持有20日获利。",
        "**二，统计重估和真实新增需求必须分开。** 2022年2月4日，前两月非农合计上修70.9万；BLS同时说明季调更新使更早的6、7月合计下修80.7万，年度整体变动修订小得多。家庭调查参与率表面提高0.3个百分点也来自人口控制更新，剔除这一影响后当月持平。不能把这些数字全部解释为最新需求猛增。[BLS年度调整说明及表A、表C](https://www.bls.gov/news.release/archives/empsit_02042022.htm)。",
        "**三，同一份报告也可以同时给出偏强与偏弱信息。** 2022年11月4日，机构调查非农增加26.1万，家庭调查就业却减少32.8万，失业率3.7%高于周前共识3.6%；工资同比4.7%符合共识，而环比0.4%高于0.3%共识。两个调查的覆盖和计量单位不同，不能相加或只选支持结论的一边。[BLS原报告及家庭调查表](https://www.bls.gov/news.release/archives/empsit_11042022.htm)、[10月28日Econoday周报](https://fidelity.econoday.com/byshoweventarticle?fid=561378&cust=fidelityFIPlus&year=2022&lid=0)。",
        "同日Collins公开强调政策重心从加息速度转向所需利率水平，也须权衡过度紧缩风险。当天2年收益率最终下降5bp，DXY下降约1.82%。这里只能确认就业与政策信息并存；没有把整个日收益归因于就业或该讲话。[波士顿联储原文](https://www.bostonfed.org/news-and-events/speeches/2022/perspectives-on-the-economy-and-monetary-policy.aspx)。",
        "2023年2月3日则提供另一个反例：新增就业51.7万高于周前共识18.5万，但工资环比0.3%和同比4.4%均符合共识；政府就业中州政府教育增加3.5万，BLS解释与罢工人员复岗有关。2年收益率仍上升21bp。工资没有超预期，不能推出利率必然下降。[BLS原报告](https://www.bls.gov/news.release/archives/empsit_02032023.htm)、[1月27日Econoday周报](https://fidelity.econoday.com/byshoweventarticle?fid=571335&cust=fidelityFIPlus&year=2023&lid=0)。",
        "## 固定分组结果",
        "分组在本研究查看事件收益前写入：人数是否高于周前共识 × 工资环比是否高于周前共识，未根据后续涨跌定义‘好时期’。等于预期归为未超预期。分组非常不平衡，10次人数超预期、1次未超预期，不能据此估计普适因果概率。",
        "\n".join(group_table),
        f"最大组‘人数超预期／工资未超预期’的7次事件，5日平均毛收益{mixed.gross_return5.mean():+.2%}、净收益{mixed.net_return5.mean():+.2%}；20日平均毛收益{mixed.gross_return20.mean():+.2%}、净收益{mixed.net_return20.mean():+.2%}。短窗口的微弱均值被成本吞没，20日毛收益本身已为负。仅降低费用不能把这一组解释成已经有效的指数策略。",
        "双超预期组仅3次，20日只有1次盈利；5日均值虽为正，仍不能据此挑一个更短持有期。此次不扩张参数搜索，也不启动没有新增优势依据的完整账户回测。",
        f"![就业信息与指数反应](<{figure.resolve().as_posix()}>)",
        "## 全部14次发布",
        "\n".join(table),
        "收益从公告之后首个A股开盘买入，到第5/20个交易日之后的开盘卖出。佣金单边0.04%、最低5元，滑点单边0.10%，100份一手，计入实际分红与T+1；统一10万元名义预算。收益分母是实际买入支付金额。",
        "表格排除了入场前跳空。春节、清明和国庆休市时，前A股收盘到入场间包含多日其他新闻，不能全归因于美国就业。日期、跳空、美元以及10年名义/实质收益率和通胀补偿变化均保存在明细中。",
        "事件窗口为可成交价格的描述性近似，份数使用观察到的开盘价按统一名义资金换算；未声称开盘前能精确按该价下单。它没有套用完整账户的50%仓位、尾部风险和回撤预算，窗口还会重叠，因此不计算或宣称全账户夏普。",
        "## 来源和适用范围",
        "BLS使用每次原始归档，不以今天修订后的时间序列倒填。2022年前四次发布未明示工资环比百分比，由该报告工资金额与增量计算并按一位小数表示；2022年3月4日使用全体私人雇员工资本月增加1美分，不能误用后文生产及非管理人员增加8美分的另一口径。",
        "共识来自同一Econoday周报体系，衡量相对周前预期的偏差，不能冒充公告前最后一分钟共识。2022年1月7日、12月2日及2023年1月6日没有10日以内合格周报，缺失保留。11月4日曾检索到索引日期早于正文的文章，因正文/标题晚于发布而排除，回退到10月28日周报；未按价格结果选择来源。",
        "财政部曲线及DXY仅描述当日共同反应，不进入交易信号；曲线历史首次发布时间未核准，观察日期不等于当年已经获取。名义与TIPS之差是通胀补偿，还混有风险和流动性溢价，不是纯通胀预期。[财政部名义曲线2022原数据](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=2022)、[实质曲线2022原数据](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_real_yield_curve&field_tdr_date_value=2022)。",
        "此次为有限历史发现。2023年2月3日的就业信息在上轮已经看到，该时段价格也用于其他研究；没有把本轮称为独立验证。日线无法拆出新闻净冲击，本轮没有采集或使用分钟数据。",
        "资料与复算入口：" + "；".join([file_link("固定研究规则", OUT / "protocol.json"), file_link("14次初值", OUT / "initial_release_facts.json"), file_link("11次共识和3次缺失", OUT / "expectations.json"), file_link("完整事件比较", OUT / "event_comparison.json"), file_link("28个成交窗口", OUT / "event_windows.json"), file_link("研究脚本", Path(__file__))]) + "。",
        "**夏普1.2目标仍未达到。** 本轮保留的是‘为何信息含义会变化’及其指数反例，不是新的买入规则。后续指数研究需要把外部条件放回国内整体需求与政策实施的历史背景，同时明确价格已反映了多少；不能靠再给失败信号增加条件来补救。"
    ]
    report_path = OUT / "历史发现_就业预期构成与指数传导.md"
    report_path.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    sources = []
    for path in sorted((OUT / "sources").iterdir()):
        if path.is_file():
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "bytes": path.stat().st_size, "capture_saved_at": datetime.fromtimestamp(path.stat().st_mtime, TZ).isoformat()})
    save("local_source_files.json", sources)
    result = {"study_id": "510300_HISTORICAL_INDEX_EMPLOYMENT_SURPRISE_V1", "completed_at": now,
              "status": "COMPLETED_HISTORICAL_CAUSE_DISCOVERY_NO_NEW_SIGNAL",
              "classification": "PROGRESS_INDEX_EMPLOYMENT_COMPOSITION_AND_COMPETING_INFORMATION",
              "report": report_path.relative_to(ROOT).as_posix(), "figure": figure.relative_to(ROOT).as_posix(),
              "release_count": 14, "admitted_week_ahead_consensus_count": 11, "missing_consensus_count": 3,
              "event_window_count": 28, "new_candidates": 0, "new_full_accounts": 0, "new_prospective_tasks": 0,
              "net_sharpe": None, "net_cagr": None, "goal_achieved": False, "orders_authorized": False,
              "causal_share_identified": False, "independent_validation": False,
              "discovery": "就业强弱取决于人数、工资、供给、修订及同期政策信息。14次固定发布中11次有周前共识，7次人数超预期而工资未超预期的20日平均毛收益-0.22%、净收益-0.53%；3次双超预期的20日净收益均值-1.06%。利率方向和A股持续收益仍反复异号，未形成新指数优势。",
              "next_historical_question": "把已有指数研究按固定历史阶段整理为原因、当时预期、价格反映和可成交结果的对照：重点解释外部条件相近时国内整体需求与政策实施的不同，不再生成美国新闻买入变种，不把已经失败的国内外联合信号换名重试。先确认是否存在尚未研究的指数整体信息增量。",
              "report_status": "PENDING_REVIEW", "figure_visually_reviewed": False,
              "limitations": ["周前共识不是最后时点预期", "3次共识缺失", "分组不平衡", "历史样本已经观察", "只有日线共同反应，无因果份额识别", "事件窗口非完整账户"]}
    save("result.json", result)
    print("报告、图表与结果已保存；未新增策略或完整账户。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collect", "preview", "parse", "actuals", "analyze", "finish"])
    args = parser.parse_args()
    {"collect": collect_expectations, "preview": preview_expectations, "parse": parse_expectations,
     "actuals": parse_actuals, "analyze": analyze, "finish": finish}[args.action]()


if __name__ == "__main__":
    main()
