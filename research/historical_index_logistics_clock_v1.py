"""交通物流历史运行与指数信息时钟；只做发现，不生成交易订单。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_logistics_clock_v1"
TZ = ZoneInfo("Asia/Shanghai")
WINDOWS = [("2022-05-01", "2022-07-31"), ("2022-10-01", "2023-02-28")]
FIELDS = {
    "rail_10kt": "铁路货物/万吨",
    "trucks_10k": "高速公路货车通行/万辆次",
    "port_10kt": "重点港口货物吞吐/万吨",
    "container_10kteu": "集装箱吞吐/万TEU",
    "cargo_flights": "民航货运航班/班",
    "pickup_100m": "邮政快递揽收/亿件",
    "delivery_100m": "邮政快递投递/亿件",
}
NEXT_QUESTION = "固定2022年原四次总体政策窗口，先核对既有研究，再检查国内政策与美国通胀/利率信息的先后顺序及指数共同定价；尤其核对11月11日开盘前已公开的信息，不把外部贴现变化全归入国内活动恢复，不按已知收益挑事件或阈值。"
SUPPORT = {
    "postal_staff_20221214": "https://www.spb.gov.cn/gjyzj/c100015/c100016/202212/e7a5f72a990e4a768997fe5a602f245d.shtml",
    "medical_capacity": "https://xxgk.mot.gov.cn/jigou/ysfws/202212/t20221229_3730781.html",
    "postal_capacity": "https://xxgk.mot.gov.cn/jigou/ysfws/202212/t20221229_3730780.html",
    "transport_restrictions": "https://xxgk.mot.gov.cn/jigou/zghssjzx/202212/t20221208_3720971.html",
    "holiday_2022": "https://app.www.gov.cn/govdata/gov/202110/25/477428/article.html",
    "holiday_2023": "https://app.www.gov.cn/govdata/gov/202212/08/495070/article.html",
}


def save_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def catalogue_entries() -> list[dict]:
    receipts = json.loads((OUT / "catalogue_receipts.json").read_text(encoding="utf-8-sig"))
    unique = {x["url"]: x for r in receipts for x in r.get("entries", [])}
    selected = []
    for entry in unique.values():
        match = re.search(r"(\d{1,2})月(\d{1,2})日全国", entry["title"])
        if not match:
            continue
        month, day = map(int, match.groups())
        published = pd.Timestamp(entry["catalogue_date"])
        year = published.year - int(month == 12 and published.month == 1)
        econ = f"{year:04d}-{month:02d}-{day:02d}"
        if any(lo <= econ <= hi for lo, hi in WINDOWS):
            selected.append({**entry, "catalogue_economic_date": econ, "kind": "daily"})
    return sorted(selected, key=lambda x: (x["catalogue_economic_date"], x["url"]))


def collect() -> None:
    entries = catalogue_entries()
    entries += [{"kind": "support", "source_id": k, "url": v} for k, v in SUPPORT.items()]
    receipts_path = OUT / "source_receipts.json"
    receipts = json.loads(receipts_path.read_text(encoding="utf-8")) if receipts_path.exists() else []
    prior = {r["url"]: r for r in receipts if r.get("status") == "FETCHED"}
    jobs = []
    for entry in entries:
        if entry["kind"] == "daily":
            stem = entry["url"].rsplit("/", 1)[-1].replace(".html", "")
            path = OUT / "sources/daily" / f"{stem}.html"
        else:
            path = OUT / "sources/support" / f"{entry['source_id']}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        if entry["url"] in prior and path.exists():
            continue
        jobs.append((entry, path))
    print(f"本轮目录范围内日报{len(catalogue_entries())}篇，需取得{len(jobs)}份原文。", flush=True)

    def fetch(job: tuple[dict, Path]) -> dict:
        entry, path = job
        record = {**entry, "retrieved_at": datetime.now(TZ).isoformat(), "path": path.relative_to(ROOT).as_posix()}
        try:
            response = requests.get(entry["url"], timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            record.update(http_status=response.status_code, final_url=response.url)
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")
            response.encoding = "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            plain = soup.get_text(" ", strip=True)
            if entry["kind"] == "daily" and "货车" not in plain:
                raise ValueError("原文未包含预期日报内容")
            path.write_bytes(response.content)
            path.with_suffix(".txt").write_text(plain, encoding="utf-8")
            record.update(status="FETCHED", sha256=hashlib.sha256(response.content).hexdigest())
        except Exception as exc:
            record.update(status="UNAVAILABLE", error=str(exc))
        return record

    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, future in enumerate(as_completed([pool.submit(fetch, j) for j in jobs]), 1):
            record = future.result()
            receipts.append(record)
            save_json(receipts_path, receipts)
            if i % 20 == 0 or i == len(jobs) or record["status"] != "FETCHED":
                print(f"原文进度{i}/{len(jobs)}：{record['status']} {record.get('catalogue_economic_date', record.get('source_id'))}", flush=True)


def first_number(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def parse_daily(record: dict) -> dict:
    soup = BeautifulSoup((ROOT / record["path"]).read_bytes(), "html.parser")
    node = soup.select_one(".detail-content")
    if node is None:
        raise ValueError(f"缺少正文节点：{record['url']}")
    body = re.sub(r"\s+", "", node.get_text(" ", strip=True)).replace("，", ",")
    title_node = soup.find("meta", attrs={"name": "ArticleTitle"})
    title = title_node.get("content", "") if title_node else record["title"]
    pub_node = soup.find("meta", attrs={"name": "PubDate"})
    published = str(pub_node.get("content", "")) if pub_node else ""
    visible = soup.select_one(".writerinfo")
    visible_text = visible.get_text(" ", strip=True) if visible else ""
    match = re.search(r"20\d\d-\d\d-\d\d\s+\d\d:\d\d(?::\d\d)?", visible_text)
    if match:
        published = match.group()
    econ = record["catalogue_economic_date"]
    expected_md = f"{int(econ[5:7])}月{int(econ[8:10])}日"
    rows = {**record, "article_title": title, "economic_date": econ, "published_at": published,
            "body_date_matches": expected_md in body, "body": body}
    num = r"([\d,]+(?:\.\d+)?)"
    rows["rail_10kt"] = first_number(body, [r"铁路(?:运输货物|货物发送量|货运量)" + num + r"万吨", r"铁路[^；>]{0,70}?(?:运输货物|完成)" + num + r"万吨"])
    rows["trucks_10k"] = first_number(body, [r"高速公路货车通行(?:量)?" + num + r"万辆"])
    rows["port_10kt"] = first_number(body, [r"港口完成货物吞吐量" + num + r"万吨"])
    rows["container_10kteu"] = first_number(body, [r"(?:完成)?集装箱吞吐量" + num + r"万(?:TEU|标箱)"])
    rows["cargo_flights"] = first_number(body, [r"(?:民航保障|其中(?:保障)?)货运航班(?:班)?" + num + r"班"])
    rows["pickup_100m"] = first_number(body, [r"(?:揽收(?:量)?)(?:预计)?(?:约为|约|为)?" + num + r"亿件"])
    rows["delivery_100m"] = first_number(body, [r"投递(?:量)?(?:预计)?(?:约为|约|为)?" + num + r"亿件"])
    rows["postal_provisional"] = bool(re.search(r"(?:揽收|投递)量?预计", body))
    for name, noun, verbs in [("closed_tolls", "收费站", "关闭"), ("closed_services", "服务区", "关停")]:
        value = first_number(body, [rf"临时{verbs}(?:的)?(?:高速公路)?{noun}{num}个", rf"{verbs}{noun}{num}个"])
        if re.search(rf"无(?:临时)?(?:关闭|关停)(?:的)?{noun}", body):
            value = 0.0
        if any(t in body for t in ["无临时关闭关停的收费站和服务区", "无临时关闭的高速公路收费站和服务区", "无临时关闭关停收费站和服务区", "无临时关闭关停收费站、服务区", "收费站和关停服务区已全部清零"]):
            value = 0.0
        rows[name] = value
    pub = pd.Timestamp(published) if published else pd.NaT
    economic_end = pd.Timestamp(econ) + pd.Timedelta(days=1)
    rows["clock_valid"] = bool(pd.notna(pub) and pub >= economic_end and rows["body_date_matches"])
    rows["available_after"] = str(pub.normalize() + pd.Timedelta(days=1)) if rows["clock_valid"] else None
    rows["clock_note"] = "按页面明示公开日结束后使用" if rows["clock_valid"] else "页面日期与全天统计时序矛盾或缺失；不猜测可用时点"
    return rows


def parse() -> None:
    receipts = json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8"))
    records = {r["url"]: r for r in receipts if r["status"] == "FETCHED" and r["kind"] == "daily"}
    daily = pd.DataFrame([parse_daily(r) for r in records.values()]).sort_values("economic_date")
    daily.to_parquet(OUT / "daily.parquet", index=False)
    print(f"已解析日报{len(daily)}篇。", flush=True)
    print("缺失数量：", daily[list(FIELDS) + ["closed_tolls", "closed_services"]].isna().sum().to_dict(), flush=True)
    print("时钟异常：", daily.loc[~daily.clock_valid, ["economic_date", "published_at", "url"]].to_dict("records"), flush=True)


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


HOLIDAYS = [
    ("2022-04-30", "2022-05-04", "劳动节"),
    ("2022-06-03", "2022-06-05", "端午节"),
    ("2022-10-01", "2022-10-07", "国庆节"),
    ("2022-12-31", "2023-01-02", "元旦"),
    ("2023-01-21", "2023-01-27", "春节"),
]
MAKEUP_DAYS = {"2022-05-07", "2022-10-08", "2022-10-09", "2023-01-28", "2023-01-29"}


def holiday_label(date: str) -> str:
    return "/".join(name for lo, hi, name in HOLIDAYS if lo <= date <= hi)


def weekly_data(daily: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for lo, hi in WINDOWS:
        calendar = pd.date_range(lo, hi, freq="D")
        for period, dates in pd.Series(calendar, index=calendar).groupby(calendar.to_period("W-SUN")):
            part = daily.loc[daily.date.between(period.start_time, period.end_time)]
            full = len(part) == 7 and part.date.nunique() == 7
            row = {"window": f"{lo}/{hi}", "week_start": period.start_time, "week_end": period.end_time.normalize(),
                   "calendar_dates_in_scope": len(dates), "observations": len(part), "complete_week": full,
                   "holiday_days": sum(bool(holiday_label(str(d.date()))) for d in dates),
                   "holiday": "/".join(sorted({holiday_label(str(d.date())) for d in dates if holiday_label(str(d.date()))})),
                   "makeup_days": sum(str(d.date()) in MAKEUP_DAYS for d in dates),
                   "publication_clock_valid": full and bool(part.clock_valid.all()),
                   "source_urls": part.url.tolist()}
            row["available_after"] = part.available_after.max() if row["publication_clock_valid"] else pd.NaT
            for field in FIELDS:
                row[field] = part[field].sum(min_count=7) if full else None
            for field in ["closed_tolls", "closed_services"]:
                row[field + "_report_days"] = int(part[field].notna().sum())
                row[field + "_max"] = part[field].max() if part[field].notna().any() else None
            row["postal_provisional_days"] = int(part.postal_provisional.sum())
            pieces.append(row)
    weekly = pd.DataFrame(pieces).sort_values("week_end").reset_index(drop=True)
    adjacent = weekly.week_end.diff().eq(pd.Timedelta(days=7))
    for field in FIELDS:
        weekly[field + "_wow"] = (weekly[field] / weekly[field].shift() - 1).where(adjacent)
    comparison_dates = []
    for i, row in weekly.iterrows():
        prior = weekly.iloc[i - 1] if i else None
        available = pd.NaT
        if i and adjacent.iloc[i] and row.publication_clock_valid and prior.publication_clock_valid:
            available = max(row.available_after, prior.available_after)
        comparison_dates.append(available)
    weekly["comparison_available_after"] = comparison_dates
    return weekly


def policy_snapshots(daily: pd.DataFrame, weekly: pd.DataFrame) -> list[dict]:
    prior = ROOT / "reports/research/510300_historical_index_reopening_constraints_v1"
    events = json.loads((prior / "event_clocks.json").read_text(encoding="utf-8-sig"))
    returns = json.loads((prior / "event_returns.json").read_text(encoding="utf-8-sig"))
    snapshots = []
    for event in events:
        entry = pd.Timestamp(event["entry_date"]) + pd.Timedelta(hours=9, minutes=30)
        available_daily = daily.loc[daily.clock_valid & (daily.available_after < entry)]
        available_weekly = weekly.loc[weekly.complete_week & (weekly.comparison_available_after < entry)]
        last_day = available_daily.sort_values("economic_date").iloc[-1]
        last_week = available_weekly.sort_values("week_end").iloc[-1]
        snapshots.append({"event_id": event["id"], "title": event["title"], "policy_date": event["date"],
                          "entry_at": str(entry), "latest_daily": records(last_day.to_frame().T.drop(columns=["body"]))[0],
                          "latest_comparable_week": records(last_week.to_frame().T)[0],
                          "existing_5_20_day_returns": [r for r in returns if r["event_id"] == event["id"]],
                          "use": "复盘入场前公开信息，不改变原账户和原交易规则"})
    return snapshots


def price_clock(weekly: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """在既有四笔固定窗口内标记物流信息到达时，原持仓已经实现的价格变化。"""
    prior = ROOT / "reports/research/510300_historical_index_reopening_constraints_v1"
    event_returns = json.loads((prior / "event_returns.json").read_text(encoding="utf-8-sig"))
    dividends = pd.read_csv(ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/dividends.csv")
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    rows = []
    for event in event_returns:
        if event["horizon"] != 20:
            continue
        entry, exit_date = pd.Timestamp(event["entry_date"]), pd.Timestamp(event["exit_date"])
        part = weekly.loc[weekly.complete_week & weekly.publication_clock_valid]
        for week in part.itertuples():
            trade = market.loc[market.date + pd.Timedelta(hours=9, minutes=30) > week.available_after].iloc[0]
            if not entry <= trade.date <= exit_date:
                continue
            # 在登记日收盘后才确认分红权利；这里只标记既有固定持仓的毛财富，未创建新成交。
            div = dividends.loc[(dividends.record_date >= entry) & (dividends.record_date < trade.date), "cash_dividend_per_share"].sum()
            comparison_known = pd.notna(week.comparison_available_after) and week.comparison_available_after < trade.date + pd.Timedelta(hours=9, minutes=30)
            rows.append({"event_id": event["event_id"], "original_entry": entry, "original_fixed_exit": exit_date,
                         "economic_week_end": week.week_end, "information_available_after": week.available_after,
                         "first_etf_open": trade.date, "etf_open": trade.open,
                         "gross_change_since_original_entry": (trade.open + div) / event["entry_open"] - 1,
                         "dividend_entitlement_per_share": div, "trucks_10k": week.trucks_10k,
                         "trucks_wow": week.trucks_10k_wow if comparison_known else None, "pickup_100m": week.pickup_100m,
                         "pickup_wow": week.pickup_100m_wow if comparison_known else None,
                         "week_on_week_publication_clock_valid": bool(comparison_known),
                         "description": "原固定持仓截至信息可用时的毛财富变化；不等于新信号的剩余收益"})
    return pd.DataFrame(rows)


def draw(weekly: pd.DataFrame, market: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    events = [("2022-06-28", "第九版"), ("2022-11-11", "二十条"), ("2022-12-07", "新十条"), ("2022-12-26", "乙类乙管")]
    for lo, hi in WINDOWS:
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
        w = weekly.loc[weekly.week_end.between(lo, hi) & weekly.complete_week].copy()
        x = w.week_end
        axes[0].plot(x, w.trucks_10k / 7, "o-", color="#245b81", label="高速货车通行日均（万辆次）")
        axes[0].set_ylabel("万辆次/日")
        axes[1].plot(x, w.pickup_100m / 7, "o-", color="#a64b21", label="揽收")
        axes[1].plot(x, w.delivery_100m / 7, "o-", color="#38775c", label="投递")
        axes[1].set_ylabel("亿件/日")
        p = market.loc[market.date.between(lo, hi)]
        base = market.loc[market.date < lo].iloc[-1].wealth
        axes[2].plot(p.date, (p.wealth / base - 1) * 100, color="#383443", linewidth=2, label="510300含分红再投资价格变化")
        axes[2].set_ylabel("区间累计（%）")
        for ax in axes:
            ax.grid(axis="y", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(loc="upper left", frameon=False)
            for start, stop, label in HOLIDAYS:
                if stop >= lo and start <= hi:
                    ax.axvspan(pd.Timestamp(start), pd.Timestamp(stop) + pd.Timedelta(days=1), color="#ccab69", alpha=0.16)
            for day, label in events:
                if lo <= day <= hi:
                    ax.axvline(pd.Timestamp(day), color="#858585", linestyle="--", alpha=0.6)
            ax.set_xlim(pd.Timestamp(lo), pd.Timestamp(hi))
        for day, label in events:
            if lo <= day <= hi:
                axes[2].annotate(label, (pd.Timestamp(day), 1), xycoords=("data", "axes fraction"),
                                 xytext=(3, -3), textcoords="offset points", ha="left", va="top", fontsize=9)
        axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        fig.suptitle(f"物流数量与指数并不总是同向：{lo} 至 {hi}", fontsize=16)
        fig.supxlabel("物流按经济周末绘制，信息尚须等待各篇日报公开；浅色带为法定假期。此图展示历史同步关系，不是交易信号。", fontsize=10)
        fig.savefig(OUT / f"物流与指数_{lo}_{hi}.png", dpi=150)
        plt.close(fig)


def write_report(daily: pd.DataFrame, weekly: pd.DataFrame, snapshots: list[dict], clock: pd.DataFrame, checks: dict) -> None:
    prior = ROOT / "reports/research/510300_historical_index_reopening_constraints_v1"
    previous_result = json.loads((prior / "result.json").read_text(encoding="utf-8-sig"))
    account = previous_result["accounts"]["POLICY_200000_STRESS"]
    source_receipts = json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8"))
    obtained = {r["url"]: r for r in source_receipts if r["status"] == "FETCHED"}
    w = weekly.set_index("week_end")
    weak = w.loc[pd.Timestamp("2022-12-25")]
    recovery = w.loc[pd.Timestamp("2023-01-08")]
    january_clock = clock.loc[(clock.event_id == "CLASSB_20221226") & (clock.economic_week_end == pd.Timestamp("2023-01-08"))].iloc[0]
    last_source = daily.set_index("economic_date")
    cards = [
        {"cause": "行政通行限制", "public_date": "2022-12-08", "source": SUPPORT["transport_restrictions"],
         "evidence": "取消相应查验、撤除检查点、保障物流设施运行；12月8日实际关闭收费站/服务区通报清零。",
         "limit": "关闭数量不度量司机到岗、网点承载、生产订单、货物价值或最终需求。"},
        {"cause": "到岗与末端履约能力", "public_date": "2022-12-14", "source": SUPPORT["postal_staff_20221214"],
         "evidence": "国家邮政局公开记录快件积压、人力不足、调配京外人员支持北京，并说明全国业务高峰与末端压力。",
         "limit": "这是供给约束的当时证据；北京调度不能直接代表全国份额，也不能定量解释全国道路货运下降。"},
        {"cause": "更晚的补充证据", "public_date": "2022-12-29", "source": SUPPORT["postal_capacity"],
         "signed_date": "2022-12-14", "evidence": "该原文专页公开日是12月29日，签发日不可回填为市场可得日。",
         "limit": "它不参与12月27日已知信息判断；当时的能力约束证据另有12月14日国家邮政局公开材料。"},
        {"cause": "假期与低基数", "public_date": "2022-12-08", "source": SUPPORT["holiday_2023"],
         "evidence": "元旦为12月31日至1月2日，春节为1月21日至27日，1月28/29日调休上班。",
         "limit": "完整自然周只能减少星期结构差异，没有消除春节提前停工、返岗时差及月内季节性。"},
        {"cause": "最终需求", "public_date": None, "source": None,
         "evidence": "交通通行、吞吐、揽投量可同时受供给与需求影响。",
         "limit": "本轮未观测载重率、订单取消、终端销售与库存组成；最终需求单独贡献未识别。"},
    ]
    save_json(OUT / "mechanism_cards.json", cards)
    result = {
        "study_id": "510300_HISTORICAL_INDEX_LOGISTICS_CLOCK_V1", "completed_at": datetime.now(TZ).isoformat(),
        "classification": "PROGRESS_INDEX_LOGISTICS_CAPACITY_HOLIDAY_AND_INFORMATION_CLOCK",
        "status": "COMPLETED_FIXED_HISTORICAL_DISCOVERY", "economic_date_windows": WINDOWS,
        "daily_reports": len(daily), "source_documents": len(obtained),
        "complete_calendar_weeks": checks["complete_calendar_weeks"],
        "weeks_with_valid_publication_clock": checks["weeks_with_valid_publication_clock"],
        "weeks_with_available_week_on_week_comparison": checks["weeks_with_available_week_on_week_comparison"],
        "policy_information_snapshots": len(snapshots), "information_price_marks": len(clock),
        "dec19_25_trucks_week_on_week": float(weak.trucks_10k_wow),
        "dec19_25_pickup_week_on_week": float(weak.pickup_100m_wow),
        "jan2_8_trucks_week_on_week": float(recovery.trucks_10k_wow),
        "jan2_8_pickup_week_on_week": float(recovery.pickup_100m_wow),
        "jan2_8_complete_week_first_executable_date": str(january_clock.first_etf_open.date()),
        "existing_dec27_holding_gross_change_at_that_open": float(january_clock.gross_change_since_original_entry),
        "finding": "物流日报在月度PMI前提供实际运行证据；道路清零后仍可能因履约能力和需求共同变化而减量，春节造成巨大季节波动。原四次政策入场前最新完整周道路量均下降，其后固定20日净收益却有正有负，不能用现量方向直接解释指数方向。",
        "factor_role": "RETAIN_AS_OPERATING_CONSTRAINT_CONTEXT_NOT_STANDALONE_INDEX_ENTRY_FILTER",
        "expectation_surprise_directly_measured": False, "causal_return_contribution_identified": False,
        "new_parameters_fitted": 0, "new_signal_filters": 0, "new_full_accounts": 0,
        "net_sharpe": None, "previous_full_account_net_sharpe_unchanged": account["net_sharpe"],
        "previous_full_account_cagr_unchanged": account["cagr"],
        "independent_validation": False, "goal_achieved": False, "orders_authorized": False,
        "next_historical_question": NEXT_QUESTION, "report": "历史发现_物流约束与指数定价时差.md",
    }
    save_json(OUT / "result.json", result)
    p = lambda value: "未算" if pd.isna(value) else f"{value:+.2%}"
    lines = [
        "**历史发现：物流约束变化与指数定价存在时差**", "",
        "本轮以510300代表沪深300的可交易价格，研究全国活动约束和共同定价，不扩展个股。只做已经结束的历史。",
        "",
        "**实际发现：物流日报有经营信息增量，但不能把业务量方向直接作为指数买卖方向。**在四次既有政策入场前，最新可得完整周的货车通行量均下降，固定20日窗口却有涨有跌。先辨别运输变化来自哪一层约束，再讨论它对指数的意义。",
        "",
        "固定日期为2022年5—7月、2022年10月—2023年2月，共243个自然日，取得241篇日报、33个完整自然周。5月1/2日缺失，不填零。32周有可核对的公开时钟，29周具备相邻完整周的可用比较时钟。所有七项数量均已抽取；原文的约数、预计数及港口监测范围仍限制经济解释精度。",
        "",
        "**限制解除、人员恢复、实际运输、指数重估是不同过程。**",
        "",
        f"[12月8日交通运输部通知]({SUPPORT['transport_restrictions']})要求解除相应交通限制、保障设施运行。原始日报显示，12月19—25日每日通报的临时关闭收费站、服务区均为零；该周货车通行4432.90万辆次，较前周{p(weak.trucks_10k_wow)}，揽收19.36亿件、{p(weak.pickup_100m_wow)}，投递21.12亿件、{p(weak.delivery_100m_wow)}，铁路和港口货物量分别{p(weak.rail_10kt_wow)}、{p(weak.port_10kt_wow)}。这是完整七天汇总，末日报见[12月25日运行通报]({last_source.loc['2022-12-25','url']})；全部组成原文在weekly.json内逐周列明。",
        "",
        f"[国家邮政局12月14日公开材料]({SUPPORT['postal_staff_20221214']})已经指出人员不足、积压和末端服务压力，并调配京外力量支持北京。这支持‘能力约束尚未同步恢复’的解释，也说明揽收需求与末端履约可以不同步；它不能定量说明全国运输减少有多少来自人员、有多少来自需求。",
        "",
        "向前追因应区分：是否允许运行、人员和网络能否运行、客户是否有货可运。国家铁路吨数、公路通行车次、港口吞吐吨数、民航航班及快递件数不是同一种经济量，不能把它们简单平均成已识别的总需求。",
        "",
        "**政策入场前，物流告诉了什么。**",
        "",
        "下表复用上一研究已经看过的四个事件及固定5/20交易日净收益。只恢复入场前信息，不新增过滤规则；这些窗口存在重叠，不能相加当作完整账户。周变化均为相邻完整自然周总量变化，非同比。",
        "",
        "| 政策及原入场日 | 当时最新可比完整周 | 货车周变化 | 揽收周变化 | 投递周变化 | 原5日净收益 | 原20日净收益 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for event in snapshots:
        week = event["latest_comparable_week"]
        ret = {r["horizon"]: r["net_return"] for r in event["existing_5_20_day_returns"]}
        period = week["week_start"][:10] + " 至 " + week["week_end"][:10]
        lines.append(f"| {event['title']} / {event['entry_at'][:10]} | {period} | {p(week['trucks_10k_wow'])} | {p(week['pickup_100m_wow'])} | {p(week['delivery_100m_wow'])} | {p(ret[5])} | {p(ret[20])} |")
    lines.extend([
        "",
        "12月27日入场前已知的是运输仍弱，后来固定20日净收益约+6.99%。这反对‘只有现量增长才可能上涨’的简单门槛；6月29日之后的亏损又反对‘现量下降就是修复机会’。四例是同一轮开发的解释材料，不能把反例改造成反向交易参数。",
        "",
        "**完整周的恢复早于月度PMI，但确认也会消耗价格空间。**",
        "",
        f"1月2—8日货车总量周增{p(recovery.trucks_10k_wow)}、揽收{p(recovery.pickup_100m_wow)}、投递{p(recovery.delivery_100m_wow)}。[最后一篇日报]({last_source.loc['2023-01-08','url']})在1月9日16:56公开，按本轮公开日结束后使用，完整周最早对应1月10日ETF开盘。此时原12月27日固定持仓的毛财富已变化{p(january_clock.gross_change_since_original_entry)}。这早于1月31日发布的1月PMI，是月内实际运行信息；但元旦改变了相邻两周的工作日安排，不能把这次增幅都归入趋势恢复。",
        "",
        "此后1月9—15日道路量又下降11.72%、揽收下降16.83%；到其信息对应的1月17日开盘，原12月27日持仓含分红权利毛财富已上涨6.78%。指数与当周现量不同步是观测事实；‘全由预期修复造成’仍是未被单独识别的解释。毛财富变化不是新策略收益，也不代表已测得有多少预期被价格吸收。",
        "",
        "| 经济周末 | 全周信息按规则可用日 | 首个ETF开盘 | 原12月27日持仓当时毛财富变化 | 道路周变化 | 揽收周变化 |",
        "|---|---|---|---:|---:|---:|",
    ])
    for row in clock.loc[clock.event_id == "CLASSB_20221226"].itertuples():
        lines.append(f"| {row.economic_week_end.date()} | {row.information_available_after.date()} | {row.first_etf_open.date()} | {p(row.gross_change_since_original_entry)} | {p(row.trucks_wow)} | {p(row.pickup_wow)} |")
    lines.extend([
        "",
        f"春节为1月21—27日，调休安排见[国务院通知]({SUPPORT['holiday_2023']})。1月16—22日道路量周减56.49%，1月23—29日再减31.47%；后一周揽收却增长63.59%、投递下降56.04%。节后第一完整周道路量增长163.91%，很大程度面对假期低基数。保留这些周，才能看到原始增速无法直接度量持续的盈利改善；本轮没有估计季节调整，也没有把恢复全归因于假期。",
        "",
        "**对指数研究的实用位置。**",
        "",
        "物流适合描述生产流通是否受阻、限制解除后恢复到哪一步，不适合独自代表沪深300全部盈利和风险溢价。指数行业之间存在不同暴露与抵消，还会共同受到贴现条件、风险承受能力和资金配置变化影响。当前证据无法单独测得市场事前共识、预期差大小或各驱动贡献，因此保留经营状态用途，不新增‘物流转好买、转弱卖’规则。",
        "",
        f"[上一研究完整账户](<{(prior / '历史发现_活动约束改变与指数剩余收益.md').as_posix()}>)的20万元结果仍为净夏普{account['net_sharpe']:.6f}、年化{account['cagr']:.4%}、最大回撤{account['max_drawdown']:.4%}。本轮没有新完整账户、没有拟合参数、没有改变退出；夏普1.2目标尚未达成。",
        "",
        "信息处理只保留必要约束：报告的经济日与公开日分开；11月18日原网页以早上时间标注当天24时统计，数量保留、可用时钟隔离；未报告关闭数量时留空。12月29日公开的文件不能按12月14/26日签发日回填。原文为当前取得的历史网页，并非当年逐日归档快照。",
        "",
        f"计算材料：[逐日原文与时钟](<{(OUT / 'daily.parquet').as_posix()}>)、[全部周及来源](<{(OUT / 'weekly.json').as_posix()}>)、[四个入场前快照](<{(OUT / 'policy_information_snapshots.json').as_posix()}>)、[原持仓的信息到达标记](<{(OUT / 'information_price_clock.json').as_posix()}>)、[来源取得结果](<{(OUT / 'source_receipts.json').as_posix()}>)。",
        "",
        f"[复现脚本](<{(ROOT / 'research/historical_index_logistics_clock_v1.py').as_posix()}>)使用项目Python加--parse --analyze，可在本地已有原文上重算。图：[5—7月](<{(OUT / '物流与指数_2022-05-01_2022-07-31.png').as_posix()}>)、[10月—次年2月](<{(OUT / '物流与指数_2022-10-01_2023-02-28.png').as_posix()}>)。",
        "",
        "后续历史问题：" + NEXT_QUESTION,
        "",
        "**全部完整自然周。**下表用于保留完整阶段，数量单位为万辆次和亿件；公开时钟异常的周仍可作事后数量描述，不能直接用于当时交易。",
        "",
        "| 周末 | 道路总量 | 道路周变化 | 揽收总量 | 投递总量 | 法定假期 | 全周公开时钟 |",
        "|---|---:|---:|---:|---:|---|---|",
    ])
    for row in weekly.loc[weekly.complete_week].itertuples():
        clock_text = str(row.available_after.date()) if row.publication_clock_valid else "不可确认"
        lines.append(f"| {row.week_end.date()} | {row.trucks_10k:.2f} | {p(row.trucks_10k_wow)} | {row.pickup_100m:.2f} | {row.delivery_100m:.2f} | {row.holiday or '无'} | {clock_text} |")
    (OUT / result["report"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze() -> None:
    daily = pd.read_parquet(OUT / "daily.parquet")
    daily["date"] = pd.to_datetime(daily.economic_date)
    daily["available_after"] = pd.to_datetime(daily.available_after)
    daily["holiday"] = daily.economic_date.map(holiday_label)
    daily["makeup_workday"] = daily.economic_date.isin(MAKEUP_DAYS)
    assert daily.date.is_unique, "同一经济日有多份原文，须先检查，不能自动平均"
    assert (daily[list(FIELDS)].dropna() >= 0).all().all(), "数量不得为负"
    expected = [str(d.date()) for lo, hi in WINDOWS for d in pd.date_range(lo, hi)]
    weekly = weekly_data(daily)
    market = pd.read_parquet(ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/market.parquet").sort_values("date")
    snapshots = policy_snapshots(daily, weekly)
    clock = price_clock(weekly, market)
    daily.to_parquet(OUT / "daily.parquet", index=False)
    weekly.to_parquet(OUT / "weekly.parquet", index=False)
    save_json(OUT / "weekly.json", records(weekly))
    save_json(OUT / "policy_information_snapshots.json", snapshots)
    save_json(OUT / "information_price_clock.json", records(clock))
    # 只保留影响研究结论的必要核对：完整周、数量、时点。未拟合参数、未遍历信号。
    checks = {"expected_days": len(expected), "obtained_days": len(daily),
              "missing_dates": sorted(set(expected) - set(daily.economic_date)),
              "missing_volume_fields": daily[list(FIELDS)].isna().sum().astype(int).to_dict(),
              "complete_calendar_weeks": int(weekly.complete_week.sum()),
              "weeks_with_valid_publication_clock": int(weekly.publication_clock_valid.sum()),
              "weeks_with_available_week_on_week_comparison": int(weekly.comparison_available_after.notna().sum()),
              "clock_anomalies": records(daily.loc[~daily.clock_valid, ["economic_date", "published_at", "url", "clock_note"]]),
              "postal_provisional_days": int(daily.postal_provisional.sum()),
              "missing_closure_counts": daily[["closed_tolls", "closed_services"]].isna().sum().astype(int).to_dict(),
              "first_and_last_dates": [str(daily.date.min().date()), str(daily.date.max().date())],
              "catalogue_tail": "现行原始目录止于2022年5月3日；5月1/2日缺失，相关不完整周不求总量。",
              "quantity_vs_clock": "11月18日数量保留作事后描述，页面却标注当日09:18且统计截止24时，公开时钟隔离，未按路径11月21日猜测。",
              "not_done": ["新交易过滤", "参数拟合", "新完整账户", "前瞻预测"]}
    save_json(OUT / "calculation_checks.json", checks)
    draw(weekly, market)
    write_report(daily, weekly, snapshots, clock, checks)
    print(json.dumps(checks, ensure_ascii=False), flush=True)
    print(clock[["event_id", "economic_week_end", "first_etf_open", "gross_change_since_original_entry", "trucks_wow", "pickup_wow"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true", help="取得已固定日期范围内原文，已有原文复用")
    parser.add_argument("--parse", action="store_true", help="解析原文与各自的公开时钟")
    parser.add_argument("--analyze", action="store_true", help="汇总固定自然周、当时可用信息与既有价格窗口")
    args = parser.parse_args()
    if args.collect:
        collect()
    if args.parse:
        parse()
    if args.analyze:
        analyze()
    if not (args.collect or args.parse or args.analyze):
        parser.print_help()


if __name__ == "__main__":
    main()
