"""FOMC窄窗口消息组合与510300开盘后历史收益，不构建交易账户。"""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_fomc_transmission_v1"
sys.path.insert(0, str(ROOT / "research"))
from historical_price_gap_causes_v1 import round_trip

GROUPS = ["利率降股涨", "利率降股跌", "利率升股涨", "利率升股跌", "零变动或缺失"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def group(rate, stock):
    if pd.isna(rate) or pd.isna(stock) or rate == 0 or stock == 0:
        return GROUPS[-1]
    return ("利率降" if rate < 0 else "利率升") + ("股涨" if stock > 0 else "股跌")


def describe(values):
    values = pd.Series(values, dtype=float).dropna()
    return {"n": len(values), "mean": values.mean(), "median": values.median(),
            "positive": int(values.gt(0).sum()), "minimum": values.min(), "maximum": values.max(),
            "mean_without_best": values.drop(values.idxmax()).mean() if len(values) > 1 else np.nan}


def collect_cases(cases):
    destination = OUT / "sources"
    receipt_path = destination / "case_receipts.json"
    receipts = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else []
    known = {r["url"] for r in receipts}
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    for day in cases.Date.sort_values():
        stamp = day.strftime("%Y%m%d")
        for name, url, suffix in [
            ("声明", f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{stamp}a.htm", ".html"),
            ("记者会", f"https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{stamp}.pdf", ".pdf"),
        ]:
            path = destination / f"{stamp}_{name}{suffix}"
            if not path.exists():
                response = session.get(url, timeout=40)
                response.raise_for_status()
                path.write_bytes(response.content)
            if url not in known:
                receipts.append({"url": url, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                                 "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                known.add(url)
                save(receipt_path, receipts)
            text_path = path.with_suffix(".txt")
            if not text_path.exists() and suffix == ".pdf":
                document = pdfium.PdfDocument(str(path))
                pages = []
                for index in range(len(document)):
                    page = document[index]
                    text_page = page.get_textpage()
                    pages.append({"page": index + 1, "text": text_page.get_text_range()})
                    text_page.close()
                    page.close()
                document.close()
                save(path.with_suffix(".pages.json"), pages)
                text_path.write_text("\n\n".join(f"第{p['page']}页\n{p['text']}" for p in pages), encoding="utf-8")
            elif not text_path.exists():
                soup = BeautifulSoup(path.read_bytes(), "html.parser")
                article = soup.find("div", class_="col-xs-12 col-sm-8 col-md-8") or soup.find("main") or soup
                text_path.write_text(article.get_text("\n", strip=True), encoding="utf-8")
        print(f"{day:%Y-%m-%d} 的原声明和记者会已保存。")


def main():
    cfg = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    workbook = OUT / "sources/USMPD_20260930.xlsx"
    full = pd.read_excel(workbook, sheet_name="Monetary Events")
    statements = pd.read_excel(workbook, sheet_name="Statements").set_index("Date")
    conferences = pd.read_excel(workbook, sheet_name="Press Conferences").set_index("Date")
    events = full[full.Date.between("2022-01-01", "2025-12-31") & full.Unscheduled.eq(0)].copy()
    if len(events) != 32 or events.Date.duplicated().any() or not events.PC.eq(1).all():
        raise ValueError("2022至2025年32次预定会议及记者会覆盖与预期不同。")
    events["group"] = [group(r.OIS1Y, r.SP500) for r in events.itertuples()]
    cases = events[~events.group.eq(GROUPS[-1])].assign(abs_ois=lambda d: d.OIS1Y.abs()).sort_values(["abs_ois", "Date"], ascending=[False, True]).groupby("group").head(1)
    events.to_parquet(OUT / "us_events_before_etf_returns.parquet", index=False)
    cases.to_csv(OUT / "原因案例选择_未用A股收益.csv", index=False, encoding="utf-8-sig")
    collect_cases(cases)

    market = pd.read_parquet(ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet").sort_values("date").reset_index(drop=True)
    open_times = pd.DatetimeIndex(market.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividends = dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    rows = []
    for s in events.sort_values("Date").to_dict("records"):
        day = pd.Timestamp(s["Date"])
        pc = conferences.loc[day]
        statement = statements.loc[day]
        start_et = pd.Timestamp(s["date_time"]).tz_localize("America/New_York") - pd.Timedelta(minutes=10)
        end_et = pd.Timestamp(pc.date_time).tz_localize("America/New_York") + pd.Timedelta(minutes=60)
        start_sh, end_sh = start_et.tz_convert("Asia/Shanghai"), end_et.tz_convert("Asia/Shanghai")
        entry_i = open_times.searchsorted(end_sh, side="right")
        if entry_i == 0 or entry_i + 20 >= len(market):
            raise ValueError("所需A股开盘或收益终点缺失。")
        entry = market.iloc[entry_i]
        row = {
            "us_date": day, "phase": "2022-2023" if day.year < 2024 else "2024-2025", "group": s["group"],
            "rate_group": "利率下降" if s["OIS1Y"] < 0 else "利率上升" if s["OIS1Y"] > 0 else "零或缺失",
            "statement_at_et": pd.Timestamp(s["date_time"]).tz_localize("America/New_York").isoformat(),
            "window_start_shanghai": start_sh.isoformat(), "window_end_shanghai": end_sh.isoformat(),
            "entry_date": entry.date, "entry_at_shanghai": open_times[entry_i].isoformat(),
            "wait_hours": (open_times[entry_i] - end_sh).total_seconds() / 3600,
            "gap_total_return": (entry.open + entry.dividend) / market.iloc[entry_i - 1].close - 1,
            "entry_open": entry.open, "primary_horizon": 5,
            "OIS1Y_bp": s["OIS1Y"] * 100, "MP1_bp": s["MP1"] * 100,
            "MP2_bp": s["MP2"] * 100, "SP500_pct": s["SP500"], "DXY_pct": s["DXY"],
            "UST10Y_bp": s["UST10Y"] * 100, "TIPS10Y_bp": s["TIPS10Y"] * 100,
            "inflation_compensation_10y_bp": (s["UST10Y"] - s["TIPS10Y"]) * 100,
            "statement_OIS1Y_bp": statement.OIS1Y * 100, "statement_SP500_pct": statement.SP500,
            "pressconf_OIS1Y_bp": pc.OIS1Y * 100, "pressconf_SP500_pct": pc.SP500,
            "statement_group": group(statement.OIS1Y, statement.SP500),
            "cause_case_selected": day in set(cases.Date),
        }
        row["statement_vs_complete_group_changed"] = row["statement_group"] != row["group"]
        row["statement_vs_complete_rate_sign_changed"] = bool(statement.OIS1Y * s["OIS1Y"] < 0)
        for horizon in [5, 20]:
            end = market.iloc[entry_i + horizon]
            eligible = dividends.record_date.ge(entry.date) & dividends.record_date.lt(end.date)
            dividend = float(dividends.loc[eligible, "cash_dividend_per_share"].sum())
            trade = round_trip(entry.open, end.open, dividend, cfg["costs"])
            row.update({f"exit{horizon}_date": end.date, f"exit{horizon}_open": end.open,
                        f"dividend{horizon}_per_share": dividend,
                        f"gross_return{horizon}": (end.open + dividend) / entry.open - 1,
                        f"net_return{horizon}": trade["net_return"],
                        f"pnl{horizon}_cny": trade["net_pnl_cny"], f"paid{horizon}_cny": trade["paid_cny"],
                        f"shares{horizon}": trade["shares"], f"fees{horizon}_cny": trade["commissions_cny"],
                        f"buy{horizon}_fill": trade["buy_price"], f"sell{horizon}_fill": trade["sell_price"]})
        rows.append(row)
    result_events = pd.DataFrame(rows)
    result_events.to_parquet(OUT / "32次FOMC原因线索与开盘后收益.parquet", index=False)
    result_events.to_csv(OUT / "32次FOMC原因线索与开盘后收益.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    summaries = []
    for phase in ["全部", "2022-2023", "2024-2025"]:
        phase_data = result_events if phase == "全部" else result_events[result_events.phase.eq(phase)]
        for grouping, labels in [("group", GROUPS), ("rate_group", ["利率下降", "利率上升", "零或缺失"])]:
            for label in labels:
                data = phase_data[phase_data[grouping].eq(label)]
                for horizon in [5, 20]:
                    summaries.append({"phase": phase, "grouping": grouping, "label": label, "horizon": horizon,
                                      "gap_mean": data.gap_total_return.mean(), **describe(data[f"net_return{horizon}"])})
    save(OUT / "分组描述.json", summaries)
    pd.DataFrame(summaries).to_csv(OUT / "分组描述.csv", index=False, encoding="utf-8-sig")
    result = {
        "study_id": cfg["study_id"], "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "RETURNS_COMPUTED_ORIGINAL_CAUSE_TEXT_REVIEW_PENDING",
        "previous_goal_turn_classification": cfg["previous_goal_turn_classification"],
        "sample": {"events": len(result_events), "groups": result_events.group.value_counts().to_dict(),
                   "source_missing_ois_or_equity": int(events[["OIS1Y", "SP500"]].isna().any(axis=1).sum()),
                   "statement_full_rate_sign_reversals": int(result_events.statement_vs_complete_rate_sign_changed.sum()),
                   "statement_full_group_changes": int(result_events.statement_vs_complete_group_changed.sum()),
                   "entry_wait_over_24hours": int(result_events.wait_hours.gt(24).sum()),
                   "max_entry_wait_hours": result_events.wait_hours.max()},
        "all_event_net5": describe(result_events.net_return5), "all_event_net20": describe(result_events.net_return20),
        "primary_group_results": [s for s in summaries if s["phase"] == "全部" and s["grouping"] == "group" and s["horizon"] == 5],
        "cases_selected_without_etf_returns": result_events[result_events.cause_case_selected].to_dict("records"),
        "new_accounts": 0, "new_fitted_parameters": 0, "new_forecast_cards": 0, "goal_achieved": False,
        "independent_validation": False, "event_study_is_full_account": False,
        "historical_source_first_version_proven": False, "wait_for_future_data": False,
        "report": "reports/research/510300_historical_fomc_transmission_v1/历史发现_FOMC预期路径与A股剩余收益.md",
    }
    save(OUT / "calculation_result.json", result)
    make_figure(result_events, summaries)
    print(json.dumps(clean({k:v for k,v in result.items() if k != "cases_selected_without_etf_returns"}), ensure_ascii=False, indent=2))


def make_figure(events, summaries):
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "font.size": 11, "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.patch.set_facecolor("#f5f3ed")
    colors = {"2022-2023": "#c18751", "2024-2025": "#167d78"}
    for phase, data in events.groupby("phase"):
        axes[0].scatter(data.OIS1Y_bp, data.SP500_pct, s=45, color=colors[phase], label=phase,
                        edgecolors="white", linewidths=.6, alpha=.85)
    for row in events[events.cause_case_selected].itertuples():
        axes[0].annotate(f"{row.us_date:%Y-%m-%d}", (row.OIS1Y_bp, row.SP500_pct),
                         xytext=(6, 7), textcoords="offset points", fontsize=8.5)
    axes[0].axvline(0, color="#aaa79f", linewidth=.8)
    axes[0].axhline(0, color="#aaa79f", linewidth=.8)
    axes[0].set(xlabel="1年OIS利率变化（基点）", ylabel="同期标普500收益（%）", title="声明加记者会100分钟窗口内的市场反应")
    axes[0].margins(.23)
    axes[0].legend(frameon=False, fontsize=10)
    for offset, phase in [(-.18, "2022-2023"), (.18, "2024-2025")]:
        data = [next(s for s in summaries if s["phase"] == phase and s["grouping"] == "group" and s["label"] == g and s["horizon"] == 5) for g in GROUPS[:4]]
        values = [s["mean"] * 100 if s["n"] else 0 for s in data]
        bars = axes[1].barh(np.arange(4) + offset, values, height=.32, color=colors[phase], label=phase)
        for bar, stat, value in zip(bars, data, values):
            axes[1].text(value + (.06 if value >= 0 else -.06), bar.get_y() + bar.get_height()/2,
                         f"{value:+.2f}% · n={stat['n']}" if stat["n"] else "无样本",
                         ha="left" if value >= 0 else "right", va="center", fontsize=9)
    axes[1].set_yticks(np.arange(4), GROUPS[:4])
    axes[1].axvline(0, color="#aaa79f", linewidth=.8)
    axes[1].set(xlabel="下一可用开盘后5交易日净收益均值（%）", title="510300开盘后的收益是否延续")
    axes[1].margins(x=.43)
    axes[1].legend(frameon=False, fontsize=10, loc="lower right")
    for ax in axes:
        ax.set_facecolor("#fffdf8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#d8d5cd", alpha=.4, linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("2022—2025年全部32次预定FOMC会议：原因线索与可交易收益分开", fontsize=16)
    fig.text(.04, .026, "固定按利率与股票的正负分组，不使用A股收益选组。窄窗口仍可能混合多种冲击；当前版本数据用于历史重建。\n"
             "ETF收益含既定压力费用；图中为单独事件，不能当作完整账户夏普。", fontsize=9, color="#5f625e")
    fig.tight_layout(rect=[.02, .09, .99, .94])
    fig.savefig(OUT / "FOMC原因线索与A股剩余收益.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
