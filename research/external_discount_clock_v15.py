"""对齐两个既定病例的外部利率、汇率与官方消息，保留原始收益窗口。"""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_external_discount_clock_v15"
CN = "Asia/Shanghai"
CASES = ["2020-08", "2022-08"]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(name, frame):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def at_china(day, clock):
    return pd.Timestamp(f"{str(day)[:10]} {clock}", tz=CN)


def source_gate():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for item in frozen["inputs"]:
        assert digest(OUT / "inputs" / item["name"]) == item["sha256"], item["name"]
    receipts = json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8"))
    receipts += json.loads((OUT / "cpi_pdf_receipts.json").read_text(encoding="utf-8"))
    saved = []
    for r in receipts:
        if r["status"] == "SAVED":
            assert digest(OUT / "sources" / r["name"]) == r["sha256"], r["name"]
            saved.append(r)
    assert len(saved) == 17
    return frozen


def read_treasury():
    rows = []
    for year in [2020, 2022]:
        path = OUT / "sources" / f"Treasury_{year}.xml"
        root = ET.parse(path).getroot()
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            prop = entry.find(".//{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}properties")
            values = {node.tag.split("}")[-1]: node.text for node in prop}
            day = values["NEW_DATE"][:10]
            available = pd.Timestamp(day + " 23:59:59", tz="America/New_York").tz_convert(CN)
            rows.append({"date": day, "ust2_percent": float(values["BC_2YEAR"]),
                         "ust10_percent": float(values["BC_10YEAR"]), "known_at": available,
                         "source_file": path.name, "source_sha256": digest(path),
                         "historical_first_vintage_verified": False})
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    assert not frame.date.duplicated().any()
    frame["year"] = frame.date.str[:4]
    # 额外滞后一条真实美债记录；不把跨年缺口当作一天。
    frame["known_at_delay1"] = frame.groupby("year").known_at.shift(-1)
    for field in ["ust2_percent", "ust10_percent"]:
        frame[field.replace("percent", "delta20_bp")] = frame.groupby("year")[field].diff(20) * 100
    frame["delta20_base_date"] = frame.groupby("year").date.shift(20)
    save_csv("美债两个年份_全部原始记录与可用时钟.csv", frame)
    return frame


def read_fx():
    frame = pd.read_parquet(OUT / "inputs/fx.parquet").sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    frame["known_at"] = pd.to_datetime(frame.available_at, utc=True).dt.tz_convert(CN)
    frame["usdcny_midpoint"] = frame.first_release_value
    frame["usdcny_delta20_percent"] = (frame.usdcny_midpoint / frame.usdcny_midpoint.shift(20) - 1) * 100
    frame["delta20_base_date"] = frame.date.shift(20)
    assert not frame.date.duplicated().any()
    return frame


def read_china_bonds(opens):
    frame = pd.read_parquet(OUT / "inputs/china_bonds.parquet").sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")
    known = []
    for day in frame.date:
        idx = opens.searchsorted(at_china(day, "23:59:59"), side="right")
        known.append(opens[idx] if idx < len(opens) else pd.NaT)
    frame["known_at"] = known
    frame["cgb10_delta20_bp"] = frame.cgb_10y.diff(20) * 100
    frame["delta20_base_date"] = frame.date.shift(20)
    assert not frame.date.duplicated().any()
    return frame


def last_row(frame, cutoff, clock="known_at"):
    selected = frame.loc[frame[clock].notna() & (frame[clock] <= cutoff)]
    assert len(selected), f"截止{cutoff}无可用记录"
    row = selected.iloc[-1]
    assert row[clock] <= cutoff
    return row


def snapshot(cutoff, treasury, fx, bonds):
    u = last_row(treasury, cutoff)
    lag = last_row(treasury, cutoff, "known_at_delay1")
    f = last_row(fx, cutoff)
    c = last_row(bonds, cutoff)
    result = {"cutoff_at": cutoff.isoformat(), "ust_date": u.date,
              "ust_known_at_assumed": u.known_at.isoformat(),
              "ust2_percent": u.ust2_percent, "ust10_percent": u.ust10_percent,
              "ust2_delta20_bp": u.ust2_delta20_bp, "ust10_delta20_bp": u.ust10_delta20_bp,
              "ust_delta20_base_date": u.delta20_base_date,
              "ust_delay1_date": lag.date, "ust_delay1_known_at_assumed": lag.known_at_delay1.isoformat(),
              "ust_delay1_2_percent": lag.ust2_percent, "ust_delay1_10_percent": lag.ust10_percent,
              "ust_delay1_2_delta20_bp": lag.ust2_delta20_bp, "ust_delay1_10_delta20_bp": lag.ust10_delta20_bp,
              "fx_date": f.date, "fx_known_at_assumed": f.known_at.isoformat(),
              "usdcny_midpoint": f.usdcny_midpoint, "usdcny_delta20_percent": f.usdcny_delta20_percent,
              "fx_delta20_base_date": f.delta20_base_date,
              "china_bond_date": c.date, "china_bond_known_at_assumed": c.known_at.isoformat(),
              "cgb10_percent": c.cgb_10y, "cgb10_delta20_bp": c.cgb10_delta20_bp,
              "cgb10_delta20_base_date": c.delta20_base_date,
              "cgb10_minus_ust10_bp": (c.cgb_10y - u.ust10_percent) * 100,
              "cgb10_minus_ust10_delay1_bp": (c.cgb_10y - lag.ust10_percent) * 100}
    return result


def facts(frozen, opens):
    cpi = {
        "09112020": ("2020-08", .4, 1.3, .4, 1.7),
        "10132020": ("2020-09", .2, 1.4, .2, 1.7),
        "08102022": ("2022-07", .0, 8.5, .3, 5.9),
        "09132022": ("2022-08", .1, 8.3, .6, 6.3),
        "10132022": ("2022-09", .4, 8.2, .6, 6.6),
    }
    fed = {"20200729": (0, .25), "20200916": (0, .25),
           "20220727": (2.25, 2.50), "20220921": (3, 3.25)}
    sep = {
        "20200610": {"fed_funds": {2020: .1, 2021: .1, 2022: .1}, "real_gdp": {2020: -6.5, 2021: 5.0, 2022: 3.5}},
        "20200916": {"fed_funds": {2020: .1, 2021: .1, 2022: .1, 2023: .1}, "real_gdp": {2020: -3.7, 2021: 4, 2022: 3, 2023: 2.5}},
        "20220615": {"fed_funds": {2022: 3.4, 2023: 3.8, 2024: 3.4}, "real_gdp": {2022: 1.7, 2023: 1.7, 2024: 1.9}, "core_pce": {2022: 4.3, 2023: 2.7, 2024: 2.3}},
        "20220921": {"fed_funds": {2022: 4.4, 2023: 4.6, 2024: 3.9, 2025: 2.9}, "real_gdp": {2022: .2, 2023: 1.2, 2024: 1.7, 2025: 1.8}, "core_pce": {2022: 4.5, 2023: 3.1, 2024: 2.3, 2025: 2.1}},
    }
    nodes, fact_rows = [], []
    for doc in frozen["documents"]:
        if doc["kind"] not in ["FOMC", "CPI", "SEP"]:
            continue
        name = doc["name"]
        if doc["kind"] == "CPI":
            name = name.replace(".html", ".pdf")
        src = OUT / "sources" / name
        assert src.exists()
        stamp = pd.Timestamp(doc["local_time"], tz="America/New_York").tz_convert(CN)
        first_open = opens[opens.searchsorted(stamp, side="left")]
        ident = name.split("_")[1].split(".")[0]
        page = 2 if doc["kind"] == "SEP" and ident.startswith("2022") else 1
        url = doc["url"].replace(".htm", ".pdf") if doc["kind"] == "CPI" else doc["url"]
        base = {"document": name, "kind": doc["kind"], "published_at_china": stamp.isoformat(),
                "first_china_open": first_open.isoformat(), "source_url": url,
                "source_sha256": digest(src), "pdf_page_1based": page if name.endswith(".pdf") else None,
                "historical_first_vintage_verified": False, "expected_consensus": "NOT_AVAILABLE"}
        nodes.append(base)
        if doc["kind"] == "CPI":
            stat, hm, hy, cm, cy = cpi[ident]
            for field, value in [("headline_mom_sa_percent", hm), ("headline_yoy_nsa_percent", hy),
                                 ("core_mom_sa_percent", cm), ("core_yoy_nsa_percent", cy)]:
                fact_rows.append({**base, "statistic_period": stat, "field": field, "value_percent": value,
                                  "interpretation": "当期官方公告值；环比季调，同比未季调；未计算市场意外"})
        elif doc["kind"] == "FOMC":
            for field, value in zip(["target_lower_percent", "target_upper_percent"], fed[ident]):
                fact_rows.append({**base, "statistic_period": doc["local_time"][:10], "field": field,
                                  "value_percent": value, "interpretation": "本次决定的联邦基金目标区间"})
        else:
            for field, values in sep[ident].items():
                for year, value in values.items():
                    fact_rows.append({**base, "statistic_period": str(year), "field": field,
                                      "value_percent": value,
                                      "interpretation": "FOMC参与者预测中位数；政策利率为年末，GDP/PCE为四季度同比；非市场一致预期或承诺"})
    return pd.DataFrame(nodes), pd.DataFrame(fact_rows)


def copy_verify_fx(used, fx):
    folder = OUT / "sources/original_fx"
    folder.mkdir(exist_ok=True)
    selected = fx[fx.date.isin(used)]
    assert len(selected) == len(used)
    for path, group in selected.groupby("raw_path"):
        raw = ROOT / path
        assert raw.exists()
        value = json.loads(raw.read_text(encoding="utf-8"))
        assert value["data"]["head"][0] == "USD/CNY"
        mapping = {r["date"]: float(r["values"][0]) for r in value["records"]}
        for r in group.itertuples():
            assert digest(raw) == r.source_hash
            assert abs(mapping[r.date] - r.usdcny_midpoint) < 1e-12, r.date
        target = folder / raw.name
        assert not target.exists() or digest(target) == digest(raw)
        shutil.copy2(raw, target)
    save_csv("实际使用人民币中间价_原件逐值复核.csv", selected)
    return len(selected)


def main():
    if (OUT / "result.json").exists():
        raise RuntimeError("结果已落盘，不覆盖已完成的研究。")
    frozen = source_gate()
    monthly = pd.read_csv(OUT / "inputs/monthly.csv")
    market = pd.read_csv(OUT / "inputs/market.csv").sort_values("date")
    paths = pd.read_csv(OUT / "inputs/daily_paths.csv")
    opens = pd.DatetimeIndex([at_china(d, "09:30:00") for d in market.date])
    treasury, fx, bonds = read_treasury(), read_fx(), read_china_bonds(opens)
    nodes, fact_rows = facts(frozen, opens)
    save_csv("全部官方公告_原始数值及页码.csv", fact_rows)
    snapshots, all_paths, classified, segments, wide, following_money, identities = [], [], [], [], [], [], []
    used_fx = set()
    for case in CASES:
        m = monthly.loc[monthly.stat_month == case].iloc[0]
        origin = pd.Timestamp(m.snapshot_at)
        entry, exit_at = at_china(m.E0_20_entry_date, "09:30:00"), at_china(m.E0_20_exit_date, "15:00:00")
        p = paths[paths.stat_month == case].sort_values("day_number").copy()
        assert len(p) == 20 and p.day_number.tolist() == list(range(1, 21))
        assert p.date.iloc[0] == m.E0_20_entry_date and p.date.iloc[-1] == m.E0_20_exit_date
        snap_by_phase = {}
        for phase, cutoff in [("ORIGIN", origin), ("ENTRY_OPEN", entry), ("EXIT_CLOSE", exit_at)]:
            snap = snapshot(cutoff, treasury, fx, bonds)
            snapshots.append({"stat_month": case, "phase": phase, **snap})
            snap_by_phase[phase] = snap
            used_fx.update([snap["fx_date"], snap["fx_delta20_base_date"]])
        first, last = snap_by_phase["ORIGIN"], snap_by_phase["EXIT_CLOSE"]
        rec = {"stat_month": case, "snapshot_at": origin.isoformat(), "entry_date": m.E0_20_entry_date,
               "exit_date": m.E0_20_exit_date, "E0_return_percent": m.E0_20_return * 100,
               "E1_return_percent": m.E1_20_return * 100}
        for k, v in first.items():
            rec["origin_" + k] = v
        for k, v in last.items():
            rec["exit_" + k] = v
        for tenor in [2, 10]:
            rec[f"origin_to_exit_ust{tenor}_bp"] = (last[f"ust{tenor}_percent"] - first[f"ust{tenor}_percent"]) * 100
            rec[f"origin_to_exit_ust_delay1_{tenor}_bp"] = (last[f"ust_delay1_{tenor}_percent"] - first[f"ust_delay1_{tenor}_percent"]) * 100
        rec["origin_to_exit_usdcny_percent"] = (last["usdcny_midpoint"] / first["usdcny_midpoint"] - 1) * 100
        rec["origin_to_exit_cgb10_bp"] = (last["cgb10_percent"] - first["cgb10_percent"]) * 100
        rec["origin_to_exit_spread_bp"] = last["cgb10_minus_ust10_bp"] - first["cgb10_minus_ust10_bp"]
        wide.append(rec)
        augmented = []
        for r in p.to_dict("records"):
            snap = snapshot(at_china(r["date"], "15:00:00"), treasury, fx, bonds)
            augmented.append({**r, **snap})
            used_fx.update([snap["fx_date"], snap["fx_delta20_base_date"]])
        all_paths.extend(augmented)
        start = (origin.tz_localize(None).normalize() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        for label, frame in [("美债", treasury), ("人民币中间价", fx), ("中国国债", bonds)]:
            selected = frame[(frame.date >= start) & (frame.date <= m.E0_20_exit_date)]
            save_csv(f"{case}_{label}_前40自然日至原退出日.csv", selected)
            if label == "人民币中间价":
                used_fx.update(selected.date.tolist())
                used_fx.update(selected.delta20_base_date.tolist())
        relevant = nodes[nodes.published_at_china.str.startswith(case[:4])].copy()
        for r in relevant.to_dict("records"):
            stamp = pd.Timestamp(r["published_at_china"])
            role = "PRE_ORIGIN" if stamp <= origin else "AFTER_ORIGIN_BEFORE_ENTRY" if stamp < entry else "AFTER_ENTRY" if stamp <= exit_at else "AFTER_EXIT"
            classified.append({"stat_month": case, "role": role, **r})
        active = relevant[(pd.to_datetime(relevant.published_at_china, utc=True) >= entry) &
                          (pd.to_datetime(relevant.published_at_china, utc=True) <= exit_at)]
        boundaries = [(entry, "原入场至首条新增消息前")]
        for stamp, group in active.groupby("first_china_open", sort=True):
            assert len(set(group.published_at_china)) == 1
            label = "+".join(group.kind.tolist()) + " " + group.published_at_china.iloc[0][:16]
            boundaries.append((pd.Timestamp(stamp), label))
        for i, (start_at, label) in enumerate(boundaries):
            end_at = boundaries[i + 1][0] if i + 1 < len(boundaries) else exit_at + pd.Timedelta(seconds=1)
            mask = p.date.map(lambda d: at_china(d, "15:00:00"))
            q = p[(mask >= start_at) & (mask < end_at)]
            assert len(q)
            segments.append({"stat_month": case, "segment": i + 1, "start_after_announcement": label,
                             "start_date": q.date.iloc[0], "end_date": q.date.iloc[-1], "days": len(q),
                             "contribution_pp": q.daily_contribution_to_entry_return.sum() * 100,
                             "overnight_contribution_pp": q.overnight_contribution_to_entry_return.sum() * 100,
                             "intraday_contribution_pp": q.intraday_contribution_to_entry_return.sum() * 100,
                             "causal_attribution": "NOT_IDENTIFIED"})
        new_money = monthly.copy()
        new_money["pub"] = pd.to_datetime(new_money.published_at, utc=True)
        for r in new_money[(new_money.pub > origin) & (new_money.pub <= exit_at)].itertuples():
            idx = opens.searchsorted(r.pub, side="left")
            following_money.append({"case": case, "stat_month": r.stat_month, "published_at": r.published_at,
                                    "first_china_open": opens[idx].isoformat(), "source_url": r.source_url,
                                    "source_sha256": r.source_sha256, "role": "后续新增国内信息_未用于起点分类"})
        z = market.set_index("date").loc[p.date].copy()
        entry_open = z.open.iloc[0]
        cash_div = z.dividend.copy()
        cash_div.iloc[0] = 0
        independent_path = (z.close - entry_open + cash_div.cumsum()) / entry_open
        independent_daily = independent_path.diff()
        independent_daily.iloc[0] = independent_path.iloc[0]
        errors = {"stat_month": case, "daily_max_absolute_error": float(np.max(np.abs(independent_daily.to_numpy() - p.daily_contribution_to_entry_return.to_numpy()))),
                  "path_max_absolute_error": float(np.max(np.abs(independent_path.to_numpy() - p.path_return.to_numpy()))),
                  "terminal_absolute_error": float(abs(independent_path.iloc[-1] - m.E0_20_return)),
                  "sum_contributions_absolute_error": float(abs(p.daily_contribution_to_entry_return.sum() - m.E0_20_return))}
        assert max(v for k, v in errors.items() if k != "stat_month") < 2e-8
        identities.append(errors)
    fx_checked = copy_verify_fx(used_fx, fx)
    save_csv("两个病例_起点入场与退出快照.csv", pd.DataFrame(snapshots))
    save_csv("两个病例_外部条件与原收益对照.csv", pd.DataFrame(wide))
    save_csv("原40个交易日_完整路径及当时可见外部信息.csv", pd.DataFrame(all_paths))
    save_csv("官方消息_北京时间与信息阶段.csv", pd.DataFrame(classified).sort_values(["stat_month", "published_at_china", "kind"]))
    save_csv("全部消息分段_同一入场本金贡献.csv", pd.DataFrame(segments))
    save_csv("窗口内新增国内金融数据.csv", pd.DataFrame(following_money))
    summary = {"study_id": "510300_EXTERNAL_DISCOUNT_CLOCK_V15", "case_count": 2, "daily_rows": len(all_paths),
               "official_document_count": len(nodes), "fact_row_count": len(fact_rows), "treasury_daily_rows": len(treasury),
               "fx_raw_value_checks": fx_checked, "path_identities": identities,
               "status": "DIAGNOSTIC_BUILT_PENDING_REPORT_REVIEW", "goal_achieved": False,
               "directional_predictive_validation": "NOT_RUN", "independent_validation": False,
               "interpretation": "已见结果的两个病例诊断；分段是会计分解，不能估计新闻的因果收益。"}
    save_json(OUT / "result.json", summary)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(summary, ensure_ascii=False))
    display = ["stat_month", "origin_ust_date", "origin_ust2_percent", "origin_ust10_percent", "origin_ust2_delta20_bp",
               "origin_ust10_delta20_bp", "origin_usdcny_midpoint", "origin_usdcny_delta20_percent",
               "origin_cgb10_percent", "origin_cgb10_minus_ust10_bp", "origin_to_exit_ust2_bp", "origin_to_exit_ust10_bp",
               "origin_to_exit_usdcny_percent", "origin_to_exit_cgb10_bp", "origin_to_exit_spread_bp",
               "origin_to_exit_ust_delay1_2_bp", "origin_to_exit_ust_delay1_10_bp", "E0_return_percent"]
    print(pd.DataFrame(wide)[display].to_string(index=False))
    print(pd.DataFrame(segments).to_string(index=False))


if __name__ == "__main__":
    main()
