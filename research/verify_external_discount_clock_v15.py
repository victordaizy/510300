"""用保存的官方原件与市场底表复核消息数值、可用时钟和收益恒等式。"""
from datetime import datetime
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_external_discount_clock_v15"
CN = "Asia/Shanghai"


def read(name):
    return pd.read_csv(OUT / "results" / name)


def equal(a, b, tolerance=2e-9):
    assert abs(float(a) - float(b)) <= tolerance, (a, b)


def page_text(name, page):
    pages = json.loads((OUT / "sources" / name.replace(".pdf", ".pages.json")).read_text(encoding="utf-8"))
    return pages[page - 1]["text"]


def official_values():
    facts = read("全部官方公告_原始数值及页码.csv")
    checked = 0
    for name, rows in facts.groupby("document"):
        kind = rows.kind.iloc[0]
        src = OUT / "sources" / name
        assert sha256(src.read_bytes()).hexdigest() == rows.source_sha256.iloc[0]
        stamp = pd.Timestamp(rows.published_at_china.iloc[0])
        eastern = stamp.tz_convert("America/New_York")
        if kind == "SEP":
            text = page_text(name, int(rows.pdf_page_1based.iloc[0]))
            assert "2:00 p.m." in text and "EDT" in text.splitlines()[0]
            assert eastern.hour == 14 and stamp.hour == 2
            patterns = {"fed_funds": r"^Federal\s*funds\s*rate\s*(.*)",
                        "real_gdp": r"^Change\s*in\s*real\s*GDP\s*(.*)",
                        "core_pce": r"^Core\s*PCE\s*inflation4\s*(.*)"}
            for field, group in rows.groupby("field"):
                matches = [re.match(patterns[field], line) for line in text.splitlines()]
                matches = [x for x in matches if x]
                assert len(matches) == 1, (name, field)
                values = matches[0].group(1).split()[:len(group)]
                ordered = group.sort_values("statistic_period")
                for actual, row in zip(values, ordered.itertuples()):
                    equal(actual, row.value_percent)
                    checked += 1
        elif kind == "CPI":
            text = re.sub(r"\s+", " ", page_text(name, 1))
            assert "8:30 a.m." in text and eastern.hour == 8 and eastern.minute == 30 and stamp.hour == 20
            patterns = {
                "headline_mom_sa_percent": r"\(CPI-U\) (?:rose|increased) ([0-9.]+) percent",
                "headline_yoy_nsa_percent": r"Over the last 12 months, the all items index increased ([0-9.]+) percent",
                "core_mom_sa_percent": r"The index for all items less food and energy (?:rose|increased) ([0-9.]+) percent",
                "core_yoy_nsa_percent": r"(?:index for all items less food and energy|all items less food and energy index) (?:rose|increased) ([0-9.]+) percent over the last 12 months",
            }
            for row in rows.itertuples():
                if row.field == "headline_mom_sa_percent" and "(CPI-U) was unchanged" in text:
                    value = 0
                else:
                    match = re.search(patterns[row.field], text)
                    assert match, (name, row.field)
                    value = match.group(1)
                equal(value, row.value_percent)
                checked += 1
        else:
            text = re.sub(r"\s+", " ", src.with_suffix(".txt").read_text(encoding="utf-8"))
            assert "For release at 2:00 p.m. EDT" in text and eastern.hour == 14
            match = re.search(r"target range for the federal funds rate (?:at|to) ([\d/-]+) to ([\d/-]+) percent", text)
            assert match, name
            for token, field in zip(match.groups(), ["target_lower_percent", "target_upper_percent"]):
                value = sum(float(Fraction(p)) for p in token.split("-"))
                equal(value, rows.set_index("field").loc[field, "value_percent"])
                checked += 1
    assert checked == len(facts) == 63
    return checked


def main():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("已完成，不重复改写收据。")
    checks = {"official_numeric_facts_reextracted_from_source_text": official_values()}
    raw = []
    for year in [2020, 2022]:
        for prop in ET.parse(OUT / "sources" / f"Treasury_{year}.xml").iter():
            if prop.tag.endswith("}properties"):
                values = {n.tag.rsplit("}", 1)[-1]: n.text for n in prop}
                date = values["NEW_DATE"][:10]
                raw.append({"date": date, "ust2_percent": float(values["BC_2YEAR"]), "ust10_percent": float(values["BC_10YEAR"]),
                            "year": str(year), "known": pd.Timestamp(date + " 23:59:59", tz="America/New_York").tz_convert(CN)})
    u = pd.DataFrame(raw).sort_values("date").reset_index(drop=True)
    u["delay_known"] = u.groupby("year").known.shift(-1)
    fx = pd.read_parquet(OUT / "inputs/fx.parquet").sort_values("date").reset_index(drop=True)
    fx["date"] = pd.to_datetime(fx.date).dt.strftime("%Y-%m-%d")
    fx["known"] = pd.to_datetime(fx.available_at, utc=True).dt.tz_convert(CN)
    market = pd.read_csv(OUT / "inputs/market.csv").sort_values("date").set_index("date")
    trading_days = list(market.index)
    b = pd.read_parquet(OUT / "inputs/china_bonds.parquet").sort_values("date").reset_index(drop=True)
    b["date"] = pd.to_datetime(b.date).dt.strftime("%Y-%m-%d")
    # 通过排序查找下一交易日，独立于构建脚本的时区索引实现。
    from bisect import bisect_right
    def bond_clock(date):
        i = bisect_right(trading_days, date)
        return pd.Timestamp(trading_days[i] + " 09:30", tz=CN) if i < len(trading_days) else pd.NaT
    b["known"] = b.date.map(bond_clock)
    snaps = read("两个病例_起点入场与退出快照.csv")
    paths = read("原40个交易日_完整路径及当时可见外部信息.csv")
    for row in pd.concat([snaps, paths], ignore_index=True).itertuples():
        cutoff = pd.Timestamp(row.cutoff_at)
        for delayed, clock in [(False, "known"), (True, "delay_known")]:
            eligible = u[u[clock].notna() & (u[clock] <= cutoff)]
            index = eligible.index[-1]
            obs = u.loc[index]
            prefix = "ust_delay1_" if delayed else "ust"
            saved_date = row.ust_delay1_date if delayed else row.ust_date
            assert obs.date == saved_date
            for tenor in [2, 10]:
                field = f"ust{tenor}_percent"
                saved_field = f"{prefix}{tenor}_percent"
                equal(obs[field], getattr(row, saved_field))
                diff = (obs[field] - u.loc[index - 20, field]) * 100
                equal(diff, getattr(row, f"{prefix}{tenor}_delta20_bp"))
        eligible = fx[fx.known <= cutoff]
        index = eligible.index[-1]
        obs = fx.loc[index]
        assert obs.date == row.fx_date
        equal(obs.first_release_value, row.usdcny_midpoint)
        equal((obs.first_release_value / fx.loc[index - 20, "first_release_value"] - 1) * 100, row.usdcny_delta20_percent)
        cb = b[b.known.notna() & (b.known <= cutoff)].iloc[-1]
        assert cb.date == row.china_bond_date
        equal(cb.cgb_10y, row.cgb10_percent)
        equal((row.cgb10_percent - row.ust10_percent) * 100, row.cgb10_minus_ust10_bp)
    checks["raw_source_snapshot_recomputations"] = len(snaps) + len(paths)
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    segments = read("全部消息分段_同一入场本金贡献.csv")
    max_error = 0.0
    for case, p in paths.groupby("stat_month"):
        p = p.sort_values("day_number")
        m = monthly.loc[case]
        prices = market.loc[p.date]
        entry = float(prices.open.iloc[0])
        dividends = prices.dividend.to_numpy().copy()
        dividends[0] = 0
        values = (prices.close.to_numpy() - entry + np.cumsum(dividends)) / entry
        daily = np.diff(np.r_[0, values])
        for a, z in zip(values, p.path_return):
            max_error = max(max_error, abs(a-z))
            equal(a, z)
        equal(values[-1], m.E0_20_return)
        for seg in segments[segments.stat_month == case].itertuples():
            mask = (p.date >= seg.start_date) & (p.date <= seg.end_date)
            equal(daily[mask].sum()*100, seg.contribution_pp)
            assert mask.sum() == seg.days
            equal(seg.overnight_contribution_pp + seg.intraday_contribution_pp, seg.contribution_pp)
        equal(segments[segments.stat_month == case].contribution_pp.sum(), values[-1]*100)
        assert segments[segments.stat_month == case].days.sum() == 20
    checks["original_daily_rows_and_all_segments_recomputed"] = len(paths)
    checks["max_saved_path_error"] = max_error
    news = read("官方消息_北京时间与信息阶段.csv")
    for row in news.itertuples():
        stamp = pd.Timestamp(row.published_at_china)
        first_open = next(pd.Timestamp(day + " 09:30", tz=CN) for day in trading_days
                          if pd.Timestamp(day + " 09:30", tz=CN) >= stamp)
        assert first_open == pd.Timestamp(row.first_china_open)
        m = monthly.loc[row.stat_month]
        origin = pd.Timestamp(m.snapshot_at)
        entry = pd.Timestamp(m.E0_20_entry_date + " 09:30", tz=CN)
        exit_at = pd.Timestamp(m.E0_20_exit_date + " 15:00", tz=CN)
        expected = "PRE_ORIGIN" if stamp <= origin else "AFTER_ORIGIN_BEFORE_ENTRY" if stamp < entry else "AFTER_ENTRY" if stamp <= exit_at else "AFTER_EXIT"
        assert row.role == expected
    checks["news_clock_and_stage_rows"] = len(news)
    checks["treasury_extra_one_record_delay_preserved"] = True
    checks["source_vintage_not_promoted_to_authenticated_first_release"] = True
    payload = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_SOURCE_FACT_CLOCK_AND_PATH_RECOMPUTATION",
               "checks": checks, "scope_limit": "原件文本及会计恒等式复算；不证明因果关系或独立收益优势。"}
    (OUT / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
