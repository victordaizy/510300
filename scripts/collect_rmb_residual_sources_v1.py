"""采集离岸人民币、美元指数与国债利差来源；不计算510300事件收益。"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from research.mechanism_odds_open_contract_v1 import now, digest, save, read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_rmb_residual_state_v1"
SOURCES = OUT / "sources"


def fetch(item):
    name, url = item
    path = SOURCES / name
    receipt = path.with_suffix(path.suffix+".receipt.json")
    if path.exists():
        meta = read(receipt)
        assert meta["status"] == 200 and meta["url"] == url and meta["sha256"] == digest(path)
        return {"name": name, "status": "CACHED"}
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=35)
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout):
            if attempt:
                raise
            time.sleep(.5)
        except requests.HTTPError:
            save(path.with_suffix(path.suffix+".failure.json"), {"at": now(), "url": url,
                "status": response.status_code, "retry_after": response.headers.get("Retry-After")}, exclusive=False)
            delay = response.headers.get("Retry-After", "15")
            if response.status_code != 429 or attempt or not delay.isdigit() or int(delay) > 30:
                raise
            time.sleep(max(1, int(delay)))
    path.write_bytes(response.content)
    save(receipt, {"at": now(), "url": url, "final_url": response.url, "status": response.status_code,
        "sha256": digest(path), "bytes": len(response.content)})
    return {"name": name, "status": "DOWNLOADED"}


def download():
    jobs = [(f"dukascopy_cnh_{year}.json", f"https://jetta.dukascopy.com/v1/candles/day/USD-CNH/BID/{year}") for year in range(2016, 2026)]
    jobs += [(f"treasury_{year}.xml", "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value="+str(year)) for year in range(2016, 2026)]
    jobs += [("treasury_method.html", "https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/"),
        ("treasury_xml_manual.html", "https://home.treasury.gov/treasury-daily-interest-rate-xml-feed"),
        ("dukascopy_historical_service.html", "https://www.dukascopy.com/api/data/get/historical-data-export"),
        ("ice_dollar_definition.html", "https://www.ice.com/forex/usdx")]
    records = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for row in pool.map(fetch, jobs):
            records.append(row)
            print("来源已保存："+row["name"], flush=True)
    save(OUT / "download_receipt.json", {"at": now(), "results": records})


def normalize_dukascopy(payload):
    """按公开响应中的差分与倍率还原报价，不插入客户端生成的平坦K线。"""
    columns = ["opens", "highs", "lows", "closes", "volumes"]
    count = len(payload["times"])
    assert all(len(payload[c]) == count for c in columns)
    steps = np.asarray(payload["times"], float)
    assert np.isfinite(steps).all() and (steps >= 0).all() and np.equal(steps, np.floor(steps)).all()
    assert payload["shift"] == 86400000 and payload["multiplier"] > 0
    stamps = payload["timestamp"]+np.cumsum(steps)*payload["shift"]
    frame = pd.DataFrame({"timestamp_utc": pd.to_datetime(stamps, unit="ms", utc=True)})
    multiplier = float(payload["multiplier"])
    for field in ["open", "high", "low", "close"]:
        changes = np.asarray(payload[field+"s"], float)
        assert np.isfinite(changes).all() and np.equal(changes, np.floor(changes)).all()
        units = round(float(payload[field])/multiplier)+np.cumsum(changes)
        frame[field] = np.round(units*multiplier, 8)
    frame["quote_volume"] = np.asarray(payload["volumes"], float)
    frame["date"] = frame.timestamp_utc.dt.tz_localize(None).dt.normalize()
    assert frame.timestamp_utc.is_unique and frame.timestamp_utc.is_monotonic_increasing
    assert frame[["open", "high", "low", "close"]].gt(0).all().all()
    assert (frame.high+1e-8 >= frame[["open", "close", "low"]].max(axis=1)).all()
    assert (frame.low-1e-8 <= frame[["open", "close", "high"]].min(axis=1)).all()
    assert np.isfinite(frame.quote_volume).all() and frame.quote_volume.ge(0).all()
    frame["bar_end_utc"] = frame.timestamp_utc+pd.Timedelta(days=1)
    return frame


def normalize_treasury(payload):
    root = ET.fromstring(payload)
    rows = []
    for entry in root.iter():
        if entry.tag.split("}")[-1] != "properties":
            continue
        fields = {child.tag.split("}")[-1]: child.text for child in entry}
        if not fields.get("BC_10YEAR"):
            continue
        rows.append({"date": pd.Timestamp(fields["NEW_DATE"]).normalize(), "us_10y": float(fields["BC_10YEAR"])})
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    assert len(frame) and frame.date.is_unique and np.isfinite(frame.us_10y).all()
    return frame


def parse():
    assert not (OUT / "freeze.json").exists(), "冻结后不可重建输入"
    input_dir = OUT / "inputs"
    cnh, treasury, missing_fx_years = [], [], []
    for year in range(2016, 2026):
        fx_path = SOURCES / f"dukascopy_cnh_{year}.json"
        if fx_path.exists():
            fx = normalize_dukascopy(read(fx_path))
            assert fx.date.dt.year.eq(year).all()
            fx["source_file"] = f"sources/dukascopy_cnh_{year}.json"
            cnh.append(fx)
        else:
            missing_fx_years.append(year)
        rates = normalize_treasury((SOURCES / f"treasury_{year}.xml").read_bytes())
        assert rates.date.dt.year.eq(year).all()
        treasury.append(rates)
    cnh = pd.concat(cnh, ignore_index=True)
    treasury = pd.concat(treasury, ignore_index=True)
    assert cnh.date.is_unique and treasury.date.is_unique
    raw = read(SOURCES / "dxy_chart.json")["chart"]["result"][0]
    assert raw["meta"]["symbol"] == "DX-Y.NYB" and raw["meta"]["instrumentType"] == "INDEX"
    dates = pd.to_datetime(raw["timestamp"], unit="s", utc=True).tz_convert(raw["meta"]["exchangeTimezoneName"]).tz_localize(None).normalize()
    dxy = pd.DataFrame({"date": dates, "dxy": raw["indicators"]["quote"][0]["close"]})
    missing_dxy = dxy.loc[dxy.dxy.isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
    dxy = dxy.dropna().reset_index(drop=True)
    assert dxy.date.is_unique and dxy.dxy.gt(0).all()
    cn_path = ROOT / "data/raw/macro/china_government_bond_yields_daily.parquet"
    shutil.copy2(cn_path, input_dir / "cn_yield_source.parquet")
    cn = pd.read_parquet(input_dir / "cn_yield_source.parquet")[["date", "cgb_10y"]].rename(columns={"cgb_10y": "cn_10y"})
    cn.date = pd.to_datetime(cn.date)
    assert cn.date.is_unique
    all_dates = cnh[["date"]].merge(dxy[["date"]], how="outer").merge(treasury[["date"]], how="outer").merge(cn[["date"]], how="outer").sort_values("date")
    coverage = all_dates.loc[all_dates.date.between("2016-08-12", "2025-12-31")].copy()
    for name, frame in [("cnh", cnh), ("dxy", dxy), ("us_yield", treasury), ("cn_yield", cn)]:
        coverage[name+"_present"] = coverage.date.isin(frame.date)
    coverage.to_csv(OUT / "来源逐日覆盖.csv", index=False, encoding="utf-8-sig")
    common = cnh.merge(dxy, on="date").merge(treasury, on="date").merge(cn, on="date")
    common = common.loc[common.date.between("2016-08-12", "2025-12-31")].sort_values("date").reset_index(drop=True)
    assert common.date.is_unique and common[["close", "dxy", "us_10y", "cn_10y"]].notna().all().all()
    common["available_at"] = common.date.dt.tz_localize("Asia/Shanghai")+pd.Timedelta(days=2, hours=23, minutes=59)
    assert (common.bar_end_utc <= common.available_at.dt.tz_convert("UTC")).all()
    common["fx_change5"] = np.log(common.close).diff(5)
    common["usd_change5"] = np.log(common.dxy).diff(5)
    common["spread_change5"] = (common.cn_10y-common.us_10y).diff(5)
    common["window_start_date"] = common.date.shift(5)
    common["window_calendar_days"] = (common.date-common.window_start_date).dt.days
    common["window_crosses_missing_source_year"] = False
    for year in missing_fx_years:
        crosses = common.window_start_date.le(pd.Timestamp(year=year, month=12, day=31)) & common.date.ge(pd.Timestamp(year=year, month=1, day=1))
        common.loc[crosses, "window_crosses_missing_source_year"] = True
        common.loc[crosses, ["fx_change5", "usd_change5", "spread_change5"]] = np.nan
    common.to_parquet(input_dir / "common_sources.parquet", index=False)
    cnh.to_parquet(input_dir / "cnh.parquet", index=False)
    dxy.to_parquet(input_dir / "dxy.parquet", index=False)
    treasury.to_parquet(input_dir / "us_yield.parquet", index=False)
    shutil.copy2(ROOT / "reports/research/510300_mechanism_odds_open_contract_v1/inputs/market.parquet", input_dir / "market.parquet")
    shutil.copy2(ROOT / "config/510300_existing_data_training_mandate_v1.json", input_dir / "authority_snapshot.json")
    save(OUT / "source_quality_receipt.json", {"at": now(), "status": "PASS_AVAILABLE_QUOTES_WITH_DECLARED_YEAR_GAP" if missing_fx_years else "PASS_IDENTITY_AND_QUOTE_ALIGNMENT",
        "cnh_source": "DUKASCOPY_USD_CNH_BID", "cnh_daily_rows": len(cnh), "dxy_valid_rows": len(dxy),
        "dxy_missing_close_dates": missing_dxy, "us_rate_rows": len(treasury), "common_source_dates": len(common),
        "common_start": common.date.min(), "common_end": common.date.max(), "synthetic_flat_bars_inserted": 0,
        "all_bar_ends_before_available": True, "historical_first_seen_proven": False,
        "missing_fx_years": missing_fx_years, "windows_rejected_for_year_gap": int(common.window_crosses_missing_source_year.sum()),
        "same_time_quotes": False, "rate_methodologies_identical": False,
        "boundary": "这是不同市场日终来源的保守日期代理；前瞻抓取与同步价格未证明。覆盖门槛在模型周原点形成后另验。"})
    print(f"来源完成：CNH日柱{len(cnh)}，美元指数{len(dxy)}，美国利率{len(treasury)}，共同来源日期{len(common)}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离岸汇率固定检验的来源准备")
    parser.add_argument("stage", choices=["download", "parse"])
    args = parser.parse_args()
    {"download": download, "parse": parse}[args.stage]()
