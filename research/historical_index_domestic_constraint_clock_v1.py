"""历史指数研究：固定14次CPI事件的国内约束与公开信息时钟，不生成新交易。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_domestic_constraint_clock_v1"
ANCHOR = ROOT / "reports/research/510300_historical_index_expectation_anchor_v1"
POLICY = ROOT / "reports/research/510300_historical_index_reopening_constraints_v1"
LOGISTICS = ROOT / "reports/research/510300_historical_index_logistics_clock_v1"
PMI = ROOT / "reports/research/510300_manufacturing_price_transmission_source_v1"
MARKET = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/market.parquet"
TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def local_time(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(TZ) if stamp.tzinfo is None else stamp.tz_convert(TZ)


def fetch_sources() -> None:
    protocol = read(OUT / "protocol.json")
    if "recorded_at" not in protocol:
        protocol["recorded_at"] = now()
        save(OUT / "protocol.json", protocol)
    existing = {r["id"]: r for r in read(OUT / "source_manifest.json")} if (OUT / "source_manifest.json").exists() else {}

    def one(item: dict) -> dict:
        if item["id"] in existing and existing[item["id"]]["status"] == "FETCHED":
            old = existing[item["id"]]
            path = ROOT / old["path"]
            if path.exists() and digest(path) == old["sha256"]:
                return old
        record = {**item, "retrieved_at": now()}
        try:
            response = requests.get(item["url"], timeout=25, headers={"User-Agent": "Mozilla/5.0"})
            record.update(http_status=response.status_code, final_url=response.url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            body = soup.get_text("\n", strip=True)
            if "采购经理指数" not in body or item["expected_public_date"].replace("-", "") not in re.sub(r"\D", "", body):
                raise ValueError("响应缺少目标主题或原文日期，保留失败，不拼凑原文")
            path = OUT / "sources" / (item["id"] + ".html")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            path.with_suffix(".txt").write_text(body, encoding="utf-8")
            record.update(status="FETCHED", path=rel(path), sha256=digest(path), historical_immutable_snapshot=False)
        except (requests.RequestException, ValueError) as exc:
            record.update(status="FETCH_FAILED", error=str(exc))
        return record

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, read(OUT / "source_plan.json")))
    save(OUT / "source_manifest.json", records)
    print(json.dumps([{"来源": x["id"], "状态": x["status"]} for x in records], ensure_ascii=False))


def parse_pmi() -> list[dict]:
    months = read(OUT / "protocol.json")["pmi_months"]
    source = [r for r in read(PMI / "released_records.json") if r["stat_month"] in months]
    assert len(source) == len(months) == 15, "月份数量不完整"
    columns = {
        "manufacturing": ["manufacturing_pmi", "production", "new_orders", "raw_inventory", "employment", "supplier_delivery"],
        "manufacturing_other": ["new_export_orders", "imports", "purchases", "input_price", "output_price", "finished_inventory", "backlog", "business_expectation"],
        "nonmanufacturing": ["nonmanufacturing_activity", "nonmanufacturing_orders", "nonmanufacturing_input_price", "nonmanufacturing_output_price", "nonmanufacturing_employment", "nonmanufacturing_expectation"],
    }
    result = []
    for old in source:
        raw_path = ROOT / old["raw_path"]
        assert digest(raw_path) == old["raw_sha256"], "复用的PMI原文件已变化"
        soup = BeautifulSoup(raw_path.read_bytes(), "html.parser")
        text = compact(soup.get_text())
        year, month = old["stat_month"].split("-")
        label = f"{year}年{int(month)}月"
        candidates = {key: [] for key in columns}
        selected_columns = {}
        for table in soup.find_all("table"):
            rows = [[compact(cell.get_text()) for cell in tr.find_all(["td", "th"])] for tr in table.find_all("tr")]
            heading = {cell for row in rows[:4] for cell in row}
            if {"PMI", "生产", "新订单", "供应商配送时间"}.issubset(heading):
                kind = "manufacturing"
            elif {"主要原材料购进价格", "出厂价格", "新出口订单"}.issubset(heading):
                kind = "manufacturing_other"
            elif {"商务活动", "新订单", "投入品价格", "销售价格"}.issubset(heading):
                kind = "nonmanufacturing"
            else:
                continue
            selected = list(columns[kind])
            if kind == "manufacturing_other" and "生产经营活动预期" not in heading:
                selected = selected[:-1]
            if kind == "nonmanufacturing" and "业务活动预期" not in heading:
                selected = selected[:-1]
            if kind in selected_columns:
                assert selected_columns[kind] == selected
            selected_columns[kind] = selected
            for row in rows:
                if row and row[0] == label:
                    assert len(row) == len(selected) + 1, (old["stat_month"], kind, row)
                    candidates[kind].append(tuple(float(x) for x in row[1:]))
        item = {k: old[k] for k in ["stat_month", "published_at", "known_at", "url", "title", "raw_path", "raw_sha256", "original_snapshot_retrieved_at", "historical_first_vintage_verified"]}
        item["selected_current_month_rows"] = {}
        for kind, entries in candidates.items():
            assert len(set(entries)) == 1, (old["stat_month"], kind, entries)
            values = entries[0]
            item.update(dict(zip(selected_columns[kind], values)))
            item["selected_current_month_rows"][kind] = {"label": label, "columns": selected_columns[kind], "values": values, "matching_instances": len(entries)}
        item["columns_not_published_in_current_table"] = [name for names in columns.values() for name in names if name not in item]
        for name in item["columns_not_published_in_current_table"]:
            item[name] = None
        for key, pattern in {
            "services_activity": r"服务业商务活动指数为(\d+(?:\.\d+)?)%",
            "construction_activity": r"建筑业商务活动指数为(\d+(?:\.\d+)?)%",
            "composite_output": r"综合PMI产出指数为(\d+(?:\.\d+)?)%",
        }.items():
            values = set(float(x) for x in re.findall(pattern, text))
            assert len(values) == 1, (old["stat_month"], key, values)
            item[key] = next(iter(values))
        assert item["input_price"] == old["input_price_diffusion"]
        assert item["output_price"] == old["output_price_diffusion"]
        item["headline_component_arithmetic"] = (0.30 * item["new_orders"] + 0.25 * item["production"] + 0.20 * item["employment"] + 0.15 * (100 - item["supplier_delivery"]) + 0.10 * item["raw_inventory"])
        item["supplier_delivery_contribution_relative_to_neutral_50"] = 0.15 * (50 - item["supplier_delivery"])
        item["headline_rounding_residual"] = item["manufacturing_pmi"] - item["headline_component_arithmetic"]
        item["component_arithmetic_note"] = "依原文既定权重解释测量构成；交付反向计入。不是消除交付约束的经济反事实或新交易指标。"
        item["measure_type"] = "季调扩散指数，非产量增速或利润率；交付指数降低代表更慢"
        result.append(item)
    result.sort(key=lambda x: x["stat_month"])
    save(OUT / "monthly_pmi.json", result)
    return result


def policy_news() -> list[dict]:
    items = [
        ("PLAN9_20220628", "2022-06-28T23:59:59+08:00", "第九版：缩短特定隔离，仍保留防控约束", "plan9_notice", None),
        ("GUIDANCE_20221110", "2022-11-10T19:42:00+08:00", "二十条优化方向已公开，具体细则尚未公开", "prior_guidance_mct", None),
        ("TWENTY_20221111", "2022-11-11T23:59:59+08:00", "二十条细则：隔离和接触者认定进一步调整", "twenty_notice", None),
        ("RRR_RELAY_20221126", "2022-11-26T23:59:59+08:00", "降准报道：11月25日宣布，报道页11月26日公开，12月5日实施", "rrr_nov25", "2022-12-05"),
        ("TEN_20221207", "2022-12-07T23:59:59+08:00", "十条：减少核酸查验与跨地区流动限制", "ten_notice_beijing", None),
        ("TRANSPORT_20221208", "2022-12-08T23:59:59+08:00", "交通运输落实：取消相应查验、撤除防疫检查点", "transport_restrictions", None),
        ("POSTAL_20221214", "2022-12-14T23:59:59+08:00", "邮政局公开北京快件积压与人员不足，跨区补充人力", "postal_staff_20221214", None),
        ("CLASSB_20221226", "2022-12-26T23:59:59+08:00", "乙类乙管方案公布，次年1月8日实施", "classb_web_fact", "2023-01-08"),
        ("POSTAL_LATER_20221229", "2022-12-29T23:59:59+08:00", "交通运输部人员和末端能力安排：签发12月14日，原文公开12月29日", "postal_capacity", None),
    ]
    manifest = {x["id"]: x for x in read(POLICY / "source_manifest.json")}
    web = {x["id"]: x for x in read(POLICY / "web_resolved_facts.json")}
    support = {Path(x["path"]).stem: x for x in read(LOGISTICS / "source_receipts.json") if x.get("kind") == "support" and x.get("status") == "FETCHED"}
    result = []
    for event_id, known, title, source_id, effective in items:
        source = manifest.get(source_id) or web.get(source_id) or support.get(source_id)
        assert source is not None, source_id
        result.append({"id": event_id, "kind": "政策或履约证据", "known_at": known, "title": title, "url": source["url"], "path": source.get("path"), "source_status": source["status"], "effective_date": effective, "coverage_is_exhaustive": False})
    return result


def compute() -> None:
    protocol = read(OUT / "protocol.json")
    events = [x for x in read(ANCHOR / "expectation_comparison.json") if x["kind"] == "CPI"]
    assert [x["event_date"] for x in events] == protocol["cpi_dates"]
    pmi = parse_pmi()
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"])
    opens = market["date"].dt.tz_localize(TZ) + pd.Timedelta(hours=9, minutes=30)

    def first_open(known):
        candidates = np.flatnonzero((opens > local_time(known)).to_numpy())
        if not len(candidates):
            raise ValueError(f"没有覆盖公开时间之后的开盘：{known}")
        return int(candidates[0])

    def return_at_open(entry_idx: int, mark_idx: int) -> float:
        dividends = market.loc[entry_idx + 1:mark_idx, "dividend"].sum() if mark_idx > entry_idx else 0.0
        return float((market.at[mark_idx, "open"] + dividends) / market.at[entry_idx, "open"] - 1)

    news = policy_news()
    for p in pmi:
        news.append({"id": "PMI_" + p["stat_month"], "kind": "PMI", "known_at": p["known_at"], "title": f"{p['stat_month']} PMI：制造业{p['manufacturing_pmi']:.1f}、订单{p['new_orders']:.1f}、交付{p['supplier_delivery']:.1f}、服务{p['services_activity']:.1f}", "url": p["url"], "path": p["raw_path"], "stat_month": p["stat_month"]})
    for n in news:
        idx = first_open(n["known_at"])
        n.update(first_open_idx=idx, first_usable_open=opens.iloc[idx].isoformat())
    news.sort(key=lambda x: x["known_at"])
    save(OUT / "public_information_clock.json", news)
    weekly = pd.read_parquet(LOGISTICS / "weekly.parquet")
    weekly = weekly[weekly["complete_week"] & weekly["publication_clock_valid"] & weekly["comparison_available_after"].notna()].copy()
    weekly = weekly.sort_values("week_end")
    weekly_records = weekly.to_dict("records")
    assert len(weekly_records) > 0
    logistics_columns = ["week_start", "week_end", "available_after", "comparison_available_after", "trucks_10k_wow", "pickup_100m_wow", "delivery_100m_wow", "closed_tolls_report_days", "closed_tolls_max", "closed_services_report_days", "closed_services_max", "holiday_days", "holiday", "source_urls"]
    contexts, journeys, updates = [], [], []
    for event in events:
        entry_date = pd.Timestamp(event["entry_date"])
        indices = np.flatnonzero((market["date"] == entry_date).to_numpy())
        assert len(indices) == 1
        e = int(indices[0])
        exit_idx = e + 20
        eligible = [p for p in pmi if local_time(p["known_at"]) < opens.iloc[e]]
        assert eligible
        latest = max(eligible, key=lambda x: x["known_at"])
        latest_week = [w for w in weekly_records if local_time(w["comparison_available_after"]) < opens.iloc[e]]
        week = None
        if latest_week:
            w = latest_week[-1]
            week = {k: w[k] for k in logistics_columns}
            week["age_days_since_week_end"] = int((entry_date - pd.Timestamp(w["week_end"])).days)
            week["freshness_warning"] = "这里仅显示最后一份已有记录与资料年龄；不把跨过资料缺口的旧值称作当时最新经济情况。"
        row = {k: event[k] for k in ["event_id", "event_date", "entry_date", "actual_public_at", "direction", "trigger", "core_mom_expected", "core_mom_actual", "core_mom_surprise_pp", "pre_open_gap", "net_return5", "net_return20", "gross_return20", "us_nominal2_change_bp", "us_real10_change_bp"]}
        row.update(entry_idx=e, exit_idx=exit_idx, exit_date=market.at[exit_idx, "date"], entry_at=opens.iloc[e], pmi=latest, pmi_public_age_days=(entry_date.date() - local_time(latest["published_at"]).date()).days, prior_20d_total_return=float(market.at[e - 1, "wealth"] / market.at[e - 21, "wealth"] - 1), prior_20d_from=market.at[e - 21, "date"], prior_20d_to=market.at[e - 1, "date"], latest_observed_logistics_week=week, known_policy_and_capacity_ids=[n["id"] for n in news if n["kind"] != "PMI" and n["first_open_idx"] <= e])
        contexts.append(row)
        for n in news:
            idx = n["first_open_idx"]
            if e < idx <= exit_idx:
                updates.append({"event_id": event["event_id"], "new_information": n, "holding_open_interval": idx - e, "entry_to_this_open_gross_return": return_at_open(e, idx), "available_before_entry": False})
        for w in weekly_records:
            idx = first_open(w["comparison_available_after"])
            if e < idx <= exit_idx:
                updates.append({"event_id": event["event_id"], "new_information": {"id": "LOGISTICS_" + pd.Timestamp(w["week_end"]).date().isoformat(), "kind": "物流完整周", "known_at": local_time(w["comparison_available_after"]).isoformat(), "first_usable_open": opens.iloc[idx].isoformat(), "title": f"至{pd.Timestamp(w['week_end']).date()}完整周货车/揽收/投递变化", "week": {k: w[k] for k in logistics_columns}}, "holding_open_interval": idx - e, "entry_to_this_open_gross_return": return_at_open(e, idx), "available_before_entry": False})
        if event["trigger"]:
            for idx in range(e, exit_idx + 1):
                journeys.append({"event_id": event["event_id"], "entry_date": entry_date.date().isoformat(), "date": market.at[idx, "date"].date().isoformat(), "holding_open_interval": idx - e, "gross_holding_return": return_at_open(e, idx), "open": market.at[idx, "open"], "cumulative_dividend_entitlement": float(market.loc[e + 1:idx, "dividend"].sum()) if idx > e else 0.0})
    updates.sort(key=lambda x: (x["event_id"], x["holding_open_interval"], x["new_information"]["id"]))
    save(OUT / "event_contexts.json", contexts)
    save(OUT / "holding_period_updates.json", updates)
    save(OUT / "four_event_paths.json", journeys)
    flat = []
    for c in contexts:
        flat.append({"事件": c["event_id"], "开盘": str(c["entry_date"])[:10], "核心环比预期偏差pp": c["core_mom_surprise_pp"], "PMI月份": c["pmi"]["stat_month"], "制造业PMI": c["pmi"]["manufacturing_pmi"], "制造新订单": c["pmi"]["new_orders"], "供应商交付": c["pmi"]["supplier_delivery"], "服务业活动": c["pmi"]["services_activity"], "原料价格": c["pmi"]["input_price"], "出厂价格": c["pmi"]["output_price"], "此前20日总回报": c["prior_20d_total_return"], "入场前跳空": c["pre_open_gap"], "其后5日事件净收益": c["net_return5"], "其后20日事件净收益": c["net_return20"]})
    pd.DataFrame(flat).to_parquet(OUT / "event_contexts.parquet", index=False)
    selected = [c for c in contexts if c["trigger"]]
    assert len(contexts) == 14 and len(selected) == 4
    for c in contexts:
        assert local_time(c["pmi"]["known_at"]) < c["entry_at"]
        exact_path_return = return_at_open(c["entry_idx"], c["exit_idx"])
        assert abs(exact_path_return - c["gross_return20"]) < 1e-12, (c["event_id"], exact_path_return, c["gross_return20"])
    check = {"current_month_pmi_reports": len(pmi), "all_events_preserved": len(contexts), "fixed_favorable_events": len(selected), "pmi_known_before_entry_all": True, "original_20d_gross_returns_reproduced": True, "new_accounts": 0, "new_filters": 0, "parameters_fitted": 0, "checked_at": now()}
    save(OUT / "necessary_checks.json", check)
    save(OUT / "research_receipt.json", {"recorded_at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "script_sha256": digest(Path(__file__)), "inputs": [{"path": rel(p), "sha256": digest(p)} for p in [ANCHOR / "expectation_comparison.json", PMI / "released_records.json", LOGISTICS / "weekly.parquet", POLICY / "source_manifest.json", MARKET]], "method": "只核对当月提取、公开顺序和既有固定收益；未重新跑账户，未做参数搜索或额外统计检验。"})
    print(pd.DataFrame(flat).to_string(index=False))
    print("已完成14个事件的同口径国内背景和持有期信息时钟。")


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths = pd.DataFrame(read(OUT / "four_event_paths.json"))
    contexts = {x["event_id"]: x for x in read(OUT / "event_contexts.json") if x["trigger"]}
    updates = read(OUT / "holding_period_updates.json")
    annotations = {
        "CPI_20220412": {"PMI_2022-04": "4月PMI公布后首开盘"},
        "CPI_20220810": {"PMI_2022-08": "8月PMI公布后首开盘"},
        "CPI_20221110": {"TWENTY_20221111": "二十条细则后", "PMI_2022-11": "11月PMI后", "TEN_20221207": "十条后"},
        "CPI_20221213": {"POSTAL_20221214": "邮政人员约束公开后", "CLASSB_20221226": "乙类乙管方案后", "PMI_2022-12": "12月PMI后"},
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=False)
    colors = ["#b54740", "#bd7a2e", "#137f78", "#315f9c"]
    for ax, (event_id, series), color in zip(axes.flat, paths.groupby("event_id", sort=True), colors):
        c = contexts[event_id]
        ax.plot(series["holding_open_interval"], series["gross_holding_return"], color=color, linewidth=2.4)
        ax.axhline(0, color="#899397", linewidth=0.8)
        ax.set_xlim(-0.5, 20.5)
        ax.set_ylim(-0.115, 0.09)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(axis="y", alpha=0.15)
        ax.set_title(f"{c['entry_date'][:10]} 首开盘起｜20日事件净收益 {c['net_return20']:+.2%}", fontsize=11, loc="left")
        for i, u in enumerate(x for x in updates if x["event_id"] == event_id and x["new_information"]["id"] in annotations[event_id]):
            x = u["holding_open_interval"]
            y = u["entry_to_this_open_gross_return"]
            ax.axvline(x, color="#617178", linestyle=":", alpha=0.55)
            ax.scatter([x], [y], color=color, s=27, zorder=5)
            ax.annotate(annotations[event_id][u["new_information"]["id"]], (x, y), xytext=(max(1.0, min(x - 3.5, 12)), 0.073 - 0.027 * i), fontsize=8.5, arrowprops={"arrowstyle": "-", "color": "#617178", "lw": 0.6})
        p = c["pmi"]
        ax.text(0.02, 0.035, f"入场前已知{p['stat_month']}：订单{p['new_orders']:.1f} / 交付{p['supplier_delivery']:.1f} / 服务{p['services_activity']:.1f}", transform=ax.transAxes, fontsize=9, color="#48545a")
        ax.set_xlabel("原固定持有期内的开盘间隔")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("同为核心CPI低于周前共识0.2个百分点，国内约束与价格路径仍不同", fontsize=17, x=0.065, ha="left", y=0.985)
    fig.text(0.065, 0.94, "510300历史事件｜所有图使用同一纵轴；竖线按可用时间对齐，不代表识别出独立因果效应", fontsize=10, color="#53616b")
    fig.text(0.065, 0.02, "曲线：每份持有权益（含期间分红），未扣成本。标题净收益：复用固定20日压力成本事件计算。均非完整账户夏普。", fontsize=9, color="#53616b")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.89, bottom=0.09, wspace=0.16, hspace=0.30)
    fig.savefig(OUT / "四次预期偏差_国内约束与价格时钟.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("已生成四次事件的统一口径价格路径图。")


def publish() -> None:
    contexts = read(OUT / "event_contexts.json")
    selected = [c for c in contexts if c["trigger"]]
    updates = read(OUT / "holding_period_updates.json")
    pmi = read(OUT / "monthly_pmi.json")
    previous_result = read(ANCHOR / "result.json")
    main_account = previous_result["accounts"]["WEEK_AHEAD_SURPRISE_200000_STRESS"]
    manifest = read(OUT / "source_manifest.json")
    assert all(x["status"] == "FETCHED" for x in manifest)
    assert (OUT / "四次预期偏差_国内约束与价格时钟.png").exists()
    source = {x["id"]: x for x in manifest}

    def link(path: Path, label: str) -> str:
        return f"[{label}](<{path.as_posix()}>)"

    def official(source_id: str, label: str) -> str:
        return f"[{label}]({source[source_id]['url']})"

    def table(headers: list[str], rows: list[list[str]]) -> str:
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(row) + " |" for row in rows])

    key_rows = []
    for c in selected:
        p = c["pmi"]
        key_rows.append([c["entry_date"][:10], p["stat_month"], f"{p['manufacturing_pmi']:.1f}", f"{p['new_orders']:.1f}", f"{p['supplier_delivery']:.1f}", f"{p['services_activity']:.1f}", f"{p['input_price']:.1f}/{p['output_price']:.1f}", f"{c['net_return20']:+.2%}"])
    key_table = table(["首个中国开盘", "当时可知PMI月份", "制造业", "制造新订单", "供应商交付", "服务业活动", "原料/出厂价格", "20日事件净收益"], key_rows)
    all_rows = []
    for c in contexts:
        p = c["pmi"]
        all_rows.append([c["entry_date"][:10], f"{c['core_mom_surprise_pp']:+.1f}", p["stat_month"], f"{p['manufacturing_pmi']:.1f}", f"{p['new_orders']:.1f}", f"{p['supplier_delivery']:.1f}", f"{p['services_activity']:.1f}", f"{c['net_return5']:+.2%}", f"{c['net_return20']:+.2%}"])
    all_table = table(["中国开盘", "核心环比偏差pp", "PMI月份", "制造业", "订单", "交付", "服务", "5日事件净收益", "20日事件净收益"], all_rows)
    price_table = table(["入场日", "此前20日总回报", "入场前跳空", "美国2年收益率当日变化bp", "美国10年实际收益率当日变化bp", "原固定退出日"], [[c["entry_date"][:10], f"{c['prior_20d_total_return']:+.2%}", f"{c['pre_open_gap']:+.2%}", f"{c['us_nominal2_change_bp']:+.0f}", f"{c['us_real10_change_bp']:+.0f}", c["exit_date"][:10]] for c in selected])
    clock_ids = {
        "CPI_20220412": ["PMI_2022-04"],
        "CPI_20220810": ["PMI_2022-08"],
        "CPI_20221110": ["TWENTY_20221111", "PMI_2022-11", "TEN_20221207"],
        "CPI_20221213": ["POSTAL_20221214", "CLASSB_20221226", "LOGISTICS_2022-12-25", "PMI_2022-12", "LOGISTICS_2023-01-08"],
    }
    clock_rows = []
    for u in updates:
        if u["event_id"] not in clock_ids or u["new_information"]["id"] not in clock_ids[u["event_id"]]:
            continue
        n = u["new_information"]
        start = next(c["entry_date"][:10] for c in selected if c["event_id"] == u["event_id"])
        clock_rows.append([start, n["known_at"][:10], n["first_usable_open"][:10], n["title"], f"{u['entry_to_this_open_gross_return']:+.2%}"])
    clock_table = table(["原入场日", "本研究可用日晚界", "后续首个可用开盘", "这时新出现的证据", "从原入场至该开盘毛收益"], clock_rows)
    report_path = OUT / "历史发现_指数国内约束与信息时钟.md"
    report = f"""# 510300历史发现：相似外部预期偏差，国内约束与信息时钟不同

这轮围绕指数共同驱动完成了14次历史CPI事件的国内背景对照。固定原来的4次核心CPI环比低于周前共识0.2个百分点事件，补齐入场前的订单、交付、服务活动和政策材料，并把持有期间才公开的信息放回实际时间。

**结论：外部贴现率相关利好不足以决定中国指数的剩余收益；国内约束的性质、约束是否正在改变，以及价格已经反映的信息，都必须分开。** 这次没有形成新的合格交易信号。4次入场前制造业PMI全部低于50，但20日事件收益仍是两正两负；其中服务与交付读数最好的8月也未取得正的20日收益。把低PMI统一当坏信号，或把政策松动统一当经营恢复，都会遗漏关键环节。

本轮复用了15个月当月原始PMI公告、14个固定事件收益、既有物流与政策材料，补取6份统计局原因说明。新增交易账户0个、拟合参数0个。原收益此前已经被看到，这是一项历史原因与时间顺序的发现，不是独立策略验证。

## 四次相似的核心通胀预期偏差，国内状态并不相同

{key_table}

表中PMI数值为季调扩散指数。50代表相较前月的扩张/收缩分界，不是收入或利润增速；供应商交付指数降低表示交付变慢。PMI来自广泛企业调查，覆盖和权重也不同于沪深300，不能把它直接换算为指数盈利。事件净收益复用原10万元固定名义事件计算、压力成本和整手处理，既不是20万元完整账户收益，也不是本轮优化得到的收益。

**4月：生产、运输与成本约束同时存在。** 3月公开材料已经记录停工与人员不到岗、交付受阻，以及原料上涨给中下游带来的压力。外部利率回落没有自动解除这些约束。同期美国整体CPI环比实际1.2%、周报共识1.1%，所以4次核心环比同为低于共识，并不表示整个美国信息包相同。4月更差的PMI在4月30日才公布，不能用来提前解释4月13日当时“必然”该如何交易。{official('nbs_review_2022_03', '统计局3月原因说明')}、{official('nbs_review_2022_04', '4月后续说明')}。

**8月：更低的成本，与更弱的需求和售价并存。** 7月交付50.1、服务52.8，均好于另外三个事件的入场前读数；但新订单48.5，购进/出厂价格分别40.4/40.1。统计局同时记录淡季、高耗能行业走弱和多数企业反映需求不足。原料变便宜可能减轻部分成本，订单和售价压力也可能抵消它，不能只沿着成本一条线推断指数利润。8月高温、疫情等后续说明在8月31日才公开。{official('nbs_review_2022_07', '统计局7月原因说明')}、{official('nbs_review_2022_08', '8月后续说明')}。

**11月：经营存量偏弱，但约束变化的信息已经出现。** 10月订单48.1、服务47.0，并不是经营已经强劲。11月10日19:42的国内公开报道已涉及进一步优化防控的二十条方向，21:30又公布美国CPI。11月11日开盘前的2.45%跳空同时包含多种信息，无法划给某一个因素。具体二十条细则按11月11日日末可用，只有11月14日开盘起才列入本研究的已知信息。{official('nbs_review_2022_10', '统计局10月说明')}、[11月10日19:42公开报道](https://www.mct.gov.cn/whzx/szyw/202211/t20221110_937380.htm)。

**12月：行政通行、人员履约、经营需求的恢复不是同一个日期。** 12月14日入场前已知十条及交通运输落实安排，最新可比完整周为12月5日至11日，道路货运环比还上升16.68%、揽收上升32.93%。后来仍出现人员与末端积压压力：12月14日邮政局公开材料按本研究时钟从12月15日开盘才可用，不能充作入场前已知的坏消息；12月19日至25日完整周道路货运又下降11.59%、揽收下降21.56%。行政放松与短期经营回落可以并存，一周数量回升也不能证明能力持续修复。{official('nbs_review_2022_11', '统计局11月说明')}、[12月8日交通运输落实安排](https://xxgk.mot.gov.cn/jigou/zghssjzx/202212/t20221208_3720971.html)、[12月14日国家邮政局材料](https://www.spb.gov.cn/gjyzj/c100015/c100016/202212/e7a5f72a990e4a768997fe5a602f245d.shtml)。

这些是有原文支持的约束及其变化，随后到指数现金流、风险补偿和价格的连接仍属机制推断。没有识别各因素贡献多少收益，也没有证明“只选11月/12月”具有可重复优势。

## 因子本身的构成也要解释

PMI综合值按新订单30%、生产25%、就业20%、供应商交付15%、原材料库存10%加权，其中交付时间反向计入。由公开构成值重算，15个月与原综合值的最大差为{max(abs(p['headline_rounding_residual']) for p in pmi):.3f}点，处于已公布一位小数的舍入尺度。

以2022年4月为例，交付指数37.2显示延迟加重。由于反向计入，单就公式而言，它相对交付50会把综合值抬高1.92点。**这个正贡献是测量公式的结果，不是交付变差有利于经济。** 12月交付40.1对应同样的公式现象。这里没有另造“修正PMI”交易因子，也没有把算术替代值当作消除物流约束后的真实经济结果。[统计局当月原始公告及编制说明]({pmi[3]['url']})。

指数研究需要由此继续问：价格变化影响的是收入还是成本，物流变化来自需求还是供给，政策改变的是资金价格、法律限制还是执行能力。PMI调查企业的乐观/悲观也不是股票市场事前共识。本轮尚无国内PMI公告前市场共识，不能称为国内PMI超预期研究。

## 新证据公开时，价格可能已经走过一段

{price_table}

美国收益率为公告日期的日度变化，不能当作隔离其他消息后的高频冲击。8月两年收益率下降5个基点，十年实际收益率却上升2个基点；11月两者都下降27个基点。因此，核心CPI偏差相同，也不等于整个贴现率传导相同。此前20日涨跌是已实现价格路径，不能仅凭涨跌证明市场事前究竟预期了哪项政策。

下面按保守公开时钟展示固定持有期内的更新。PMI原文通常标注09:30，本研究沿用原资料公开日日末可用，再取下一中国开盘。日期精度的政策及人员材料也按日末处理。表内毛收益是原入场价开始的每份持有权益变化，含已发生分红权益、未扣成本；不是公告瞬间收益或新进出场计算。

{clock_table}

11月PMI继续转弱，但到其后首个可用开盘12月1日，11月11日入场的持有路径已上涨3.59%。12月读数更弱，进入可用信息时原12月14日持有路径仍下跌1.82%，随后才修复。1月PMI恢复则在1月31日才公布，已经晚于这次1月12日的固定退出。它们说明经营数据月份、公开时间和交易收益不是同一条时间轴；这些事实不等于“坏消息必涨”或已经证实“利空出尽”。[统计局12月解释](https://www.stats.gov.cn/xxgk/jd/sjjd2020/202212/t20221231_1891399.html)、[次年1月原始发布](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901737.html)。

物流资料覆盖2022年5月至7月及2022年10月至2023年2月的既有采集范围。4月入场没有相应完整周资料；8月入场时最新已有完整周止于7月31日，距入场11天，8月第一周资料未补齐。缺失不是零，也没有把跨资料空缺后的旧值描述成当时最新状态。自然周环比仍受节假日、供需混合和经营基数影响；本轮未测载重率、订单取消、终端销售，也未把北京人员情况推广为全国份额。

{link(OUT / '四次预期偏差_国内约束与价格时钟.png', '查看四次事件的价格路径与信息时钟对比图')}。

## 全部14次事件保留

{all_table}

保留反例后可以看到，2022年5月、6月核心通胀均高于周报共识，之后20日事件净收益却为正；2023年2月入场前制造业PMI已恢复50.1、服务54.0，随后20日净收益仍为负。它们只支持“不用单个好坏标签代替整个传导与定价过程”，不支持反号交易。事件窗口部分重叠，同属有限历史环境，不能把这些行当作独立随机样本计算高确信度。

## 对策略目标的实际影响

上一轮原规则的20万元完整账户仍为：净利润{main_account['net_profit']:.2f}元、净夏普{main_account['net_sharpe']:.6f}、年化{main_account['cagr']:.6%}、最大回撤{main_account['max_drawdown']:.2%}。即使只返还同一冻结持仓的交易成本，诊断年化也只有{main_account['same_holdings_fee_refund']['cagr']:.4%}。这不是本轮新回测，仍未达到净夏普1.2与年化10%的联合目标。

**本轮封存这条CPI规则的国内背景扩展。** 新增资料解释了约束的异同和信息先后，尚未产生可独立使用的入场、持有或退出信息。不在这4个已经看到结果的样本上选景气门槛、挑政策时期或缩短持有期，也不把解释更完整当成收益更高。

后续独立问题限定为：2022年1月至2023年2月的中国官方制造业PMI，在每次公告前是否有可核对的原始共识；当次偏差与新订单、生产及反向交付分量有何关系。先核验公开预期锚，水平与环比研究不能代替它；缺乏事前共识的月份保留缺失。该问题不接入已拒绝的CPI规则，也不因本轮几个收益结果选择月份。

## 资料与必要核对

仅完成当月原始行提取、14次公开先后与固定20日毛收益复算，未增加参数搜索或长期检验。原始网页目前下载并按页面历史日期重建，既有PMI记录也未证明不可变的历史首版快照；这项限制不会因哈希保存而消失。需要复查时可直接查阅下面的计算和材料。

- {link(OUT / 'event_contexts.json', '14次事件的完整国内背景')}，含PMI原文路径、可得时间及物流资料年龄。
- {link(OUT / 'holding_period_updates.json', '持有期内全部新信息时钟')}，包含本文表格未逐条展开的周度物流。
- {link(OUT / 'monthly_pmi.json', '15个月PMI当月行与测量构成')}。2022年12月原表未列经营预期的两个字段保留空值，没有从后期表格补值。
- {link(OUT / 'cause_findings.json', '四次原因解释与竞争解释')}、{link(OUT / 'source_manifest.json', '六份新增原文来源')}。
- {link(ROOT / 'research/historical_index_domestic_constraint_clock_v1.py', '完整计算脚本')}、{link(OUT / 'protocol.json', '本轮固定范围')}、{link(OUT / 'necessary_checks.json', '必要核对结果')}。
- {link(ANCHOR / 'result.json', '上一轮冻结完整账户结果')}。

完成时间：{now()}。仅历史研究，目标仍未完成。
"""
    report_path.write_text(report, encoding="utf-8")
    summary = "14次CPI对齐15个月国内PMI；4次核心环比同低于周报共识0.2pp且入场前制造业PMI都低于50，仍两正两负。4月供应与成本约束、8月需求与售价压力、11月政策方向和外部利率共同变化、12月行政放松与人员履约错位各有原文证据；未识别独立收益贡献或新交易优势，封存原CPI规则的国内背景扩展。"
    next_question = "单独核验2022年1月至2023年2月中国官方制造业PMI公告前原始共识，区分水平/环比与预期差，再对照新订单、生产和交付反向分量；从Econoday原周报可得性开始，所有月份保留，缺失不换代理，不接入已拒绝CPI规则。"
    result = {"study_id": "510300_HISTORICAL_INDEX_DOMESTIC_CONSTRAINT_CLOCK_V1", "completed_at": now(), "mode": "HISTORICAL_ONLY", "status": "COMPLETED_HISTORICAL_DISCOVERY_NO_NEW_SIGNAL", "goal_turn_classification": "PROGRESS", "classification": "PROGRESS_INDEX_DOMESTIC_CONSTRAINT_AND_INFORMATION_CLOCK", "events": 14, "fixed_favorable_events": 4, "pmi_current_month_reports": 15, "new_official_cause_explanations": 6, "new_full_accounts": 0, "new_filters": 0, "parameters_fitted": 0, "source_clock_and_fixed_returns_checked": True, "independent_validation": False, "causal_equity_return_identified": False, "underlying_candidate": "WEEK_AHEAD_DOVISH_SURPRISE", "underlying_candidate_disposition": "REJECTED_FROZEN", "domestic_context_extension_status": "CLOSED_NO_INCREMENTAL_EXECUTABLE_SIGNAL", "inherited_account_result_is_new_test": False, "inherited_full_account": {k: main_account[k] for k in ["capital", "net_profit", "net_sharpe", "cagr", "max_drawdown"]}, "inherited_account_result_path": rel(ANCHOR / "result.json"), "discovery": summary, "next_historical_question": next_question, "report": rel(report_path), "figure": rel(OUT / "四次预期偏差_国内约束与价格时钟.png"), "plot_visually_checked": True, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result)
    receipt = read(OUT / "research_receipt.json")
    receipt.update(report_completed_at=now(), script_sha256=digest(Path(__file__)), report_sha256=digest(report_path), necessary_checks_path=rel(OUT / "necessary_checks.json"))
    save(OUT / "research_receipt.json", receipt)
    cause_path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    cause = read(cause_path)
    cause.update(latest_completed_study=rel(OUT / "result.json"), latest_report=rel(report_path), updated_at=now(), current_study=rel(OUT / "protocol.json"), latest_result_summary=summary, next_historical_question=next_question, goal_achieved=False)
    save(cause_path, cause)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(current_round=result["study_id"], latest_progress_receipt=rel(OUT / "result.json"), latest_continuation_report=rel(report_path), latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=now(), latest_historical_report=rel(report_path), next_research_question=next_question, latest_historical_index_domestic_constraint_clock=rel(OUT / "result.json"), goal_achieved=False)
    save(mandate_path, mandate)
    print(json.dumps({"报告": rel(report_path), "状态": result["status"], "新增账户": 0, "目标完成": False}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定历史CPI事件的国内约束对照")
    parser.add_argument("action", choices=["fetch", "compute", "plot", "publish"])
    args = parser.parse_args()
    {"fetch": fetch_sources, "compute": compute, "plot": plot, "publish": publish}[args.action]()
