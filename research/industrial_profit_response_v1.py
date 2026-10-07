"""工业利润原月报与股价联合反应的固定、单次毛收益初筛。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from research.mechanism_odds_open_contract_v1 import save, read, now, digest
from research.lpr_joint_response_v1 import response_indices, directional_limit, gross_return

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_industrial_profit_response_v1"
SOURCE = ROOT / "reports/research/510300_industrial_receivable_legacy_completion_v1/released_records.json"
MARKET = ROOT / "reports/research/510300_mechanism_odds_open_contract_v1/inputs/market.parquet"
PRIMARY = "PROFIT_POSITIVE_EQUITY_UP"
CONTROLS = ["EQUITY_UP_ONLY", "PROFIT_POSITIVE_ONLY", "ALL_RELEASES"]
HOLD = 20
PERIODS = {"FULL": ("2020-01-01", "2025-12-31"),
           "EARLY": ("2020-01-01", "2022-12-31"),
           "LATE": ("2023-01-01", "2025-12-31")}


def parse_profit(raw: bytes) -> dict:
    """使用原表精度；正文的倍数四舍五入单独核对，不混入两年平均。"""
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    compact = re.sub(r"\s+", "", soup.get_text(" ", strip=True))
    pattern = (r"全国规模以上工业企业实现利润总额(\d+(?:\.\d+)?)亿元[，,]"
               r"(?:同比|比上年)(?:(增长|下降)(\d+(?:\.\d+)?)(%|倍)|(持平))")
    bodies = set(re.findall(pattern, compact))
    assert len(bodies) == 1, "原文累计利润及同比不是唯一值。"
    amount, direction, number, unit, flat = next(iter(bodies))
    factor = 100. if unit == "倍" else 1.
    body_yoy = 0. if flat else float(number) * factor * (-1 if direction == "下降" else 1)
    decimals = len(number.split(".")[1]) if "." in number else 0
    tolerance = .000001 if flat else factor * .5 * 10 ** (-decimals) + .000001
    totals = set()
    for table in soup.find_all("table"):
        heading = re.sub(r"\s+", "", table.get_text())
        if not all(s in heading for s in ["营业收入", "营业成本", "利润总额"]):
            continue
        assert heading.index("营业收入") < heading.index("营业成本") < heading.index("利润总额")
        for tr in table.find_all("tr"):
            cells = [re.sub(r"\s+", "", cell.get_text())
                     for cell in tr.find_all(["td", "th"], recursive=False)]
            if len(cells) == 7 and cells[0] == "总计":
                assert all(re.fullmatch(r"-?\d+(?:\.\d+)?", s) for s in cells[1:]), "总计行含歧义字段。"
                totals.add((float(cells[-2]), float(cells[-1])))
    assert len(totals) == 1, "原表总计利润或同比不是唯一值。"
    total, yoy = next(iter(totals))
    assert abs(total - float(amount)) < .000001, "正文与总计利润不一致。"
    assert abs(yoy - body_yoy) <= tolerance, "正文与原表同比超出正文舍入精度。"
    assert "按可比口径计算" in compact, "缺少原报告同比口径说明。"
    return {"profit_ytd_cny_100million": total, "profit_ytd_reported_yoy_pct": yoy,
            "body_yoy_pct": body_yoy, "body_yoy_unit": unit or "持平",
            "body_rounding_tolerance_pp": tolerance, "body_table_agree_at_reported_precision": True}


def extract():
    assert not (OUT / "freeze.json").exists(), "来源已冻结，不能覆盖。"
    registration = read(OUT / "design_registration.json")
    records, failures = [], []
    for item in read(SOURCE):
        if item["stat_month"] not in registration["expected_stat_months"]:
            continue
        raw_path = ROOT / item["raw_path"]
        try:
            assert digest(raw_path) == item["raw_sha256"], "原文哈希与既有回执不符。"
            data = parse_profit(raw_path.read_bytes())
            records.append({**item, **data, "release_date": item["published_at"][:10]})
        except (AssertionError, ValueError, OSError) as exc:
            failures.append({"stat_month": item["stat_month"], "error": str(exc), "raw_path": item["raw_path"]})
    expected = registration["expected_stat_months"]
    missing = sorted(set(expected) - {r["stat_month"] for r in records})
    save(OUT / "released_profit_records.json", records)
    save(OUT / "source_finalization.json", {"at": now(),
        "status": "PASS_COMPLETE_ORIGINAL_RELEASE_FIELDS" if not missing and not failures else "SOURCE_GATE_NOT_PASSED",
        "expected": len(expected), "admitted": len(records), "missing": missing, "failures": failures,
        "legacy_page_only_clocks": sum(r.get("clock_support") == "ORIGINAL_PAGE_ONLY_LEGACY_ARCHIVE" for r in records),
        "body_multiplier_rounding_months": [r["stat_month"] for r in records if r["body_yoy_unit"] == "倍"],
        "historical_first_vintage_verified": False, "new_network_requests": 0,
        "new_return_reads": 0, "new_accounts": 0})
    print(f"原工业利润字段已解析{len(records)}/{len(expected)}份，缺口{len(missing)}份；尚未计算事件收益。", flush=True)


def build_events(market: pd.DataFrame, releases: list[dict]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(market.date)
    returns = np.log((market.close + market.dividend) / market.close.shift())
    rows = []
    for release in releases:
        base, observation, entry = response_indices(dates, release["release_date"])
        yoy = float(release["profit_ytd_reported_yoy_pct"])
        row = {"stat_month": release["stat_month"], "release_date": release["release_date"],
               "profit_ytd_reported_yoy_pct": yoy, "base_idx": base,
               "observation_idx": observation, "entry_idx": entry, "planned_exit_idx": entry + HOLD,
               "status": "NO_VIEW_INPUT", "equity_response": np.nan,
               "gross_return": np.nan, "stress_proportional_proxy_return": np.nan,
               "selected": False, "profit_positive": yoy > 0}
        if base < 0 or observation >= len(market) or not np.isfinite(yoy):
            rows.append(row)
            continue
        response = float(np.expm1(returns.iloc[base + 1:observation + 1].sum()))
        decision = dates[observation].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
        assert pd.Timestamp(release["known_at"]) < decision
        row.update(base_date=dates[base], observation_date=dates[observation], decision_at=decision,
                   equity_response=response, selected=yoy > 0 and response > 0)
        if entry >= len(market):
            row["status"] = "CENSORED_NO_ENTRY_PRICE"
            rows.append(row)
            continue
        row["entry_date"] = dates[entry]
        assert decision < dates[entry].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        if directional_limit(market, entry, 1):
            row.update(status="UNFILLED_UPPER_LIMIT", gross_return=0., stress_proportional_proxy_return=0.,
                       exit_idx=entry, exit_date=dates[entry])
            rows.append(row)
            continue
        end = entry + HOLD
        while end < len(market) and directional_limit(market, end, -1):
            end += 1
        if end >= len(market):
            row["status"] = "CENSORED_EXIT_AFTER_SAMPLE"
        else:
            value = gross_return(market, entry, end)
            row.update(status="MATURE", exit_idx=end, exit_date=dates[end],
                       exit_delay_sessions=end - entry - HOLD, gross_return=value,
                       stress_proportional_proxy_return=value - .0028)
        rows.append(row)
    return pd.DataFrame(rows)


def mask(events, group):
    if group == PRIMARY:
        return events.selected
    if group == "EQUITY_UP_ONLY":
        return events.equity_response.gt(0)
    if group == "PROFIT_POSITIVE_ONLY":
        return events.profit_positive
    if group == "ALL_RELEASES":
        return pd.Series(True, index=events.index)
    raise ValueError("未登记组别：" + group)


def statistics(events):
    rows = []
    periods = {**PERIODS, **{str(y): (f"{y}-01-01", f"{y}-12-31") for y in range(2020, 2026)}}
    for period, (start, end) in periods.items():
        part = events.loc[events.release_date.between(start, end) & events.gross_return.notna()]
        for name in [PRIMARY] + CONTROLS:
            sample = part.loc[mask(part, name)]
            values = sample.gross_return
            rows.append({"period": period, "group": name, "n": len(sample),
                         "filled": int(sample.status.eq("MATURE").sum()),
                         "mean_gross_return": values.mean(), "median_gross_return": values.median(),
                         "mean_stress_proportional_proxy_return": sample.stress_proportional_proxy_return.mean(),
                         "positive_fraction": values.gt(0).mean() if len(values) else np.nan,
                         "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame(rows)


def freeze():
    assert not (OUT / "freeze.json").exists(), "本版本已冻结。"
    source = read(OUT / "source_finalization.json")
    assert source["status"] == "PASS_COMPLETE_ORIGINAL_RELEASE_FIELDS", "来源未通过。"
    assert read(OUT / "prefreeze_tests.json")["exit_code"] == 0, "必要实现测试未通过。"
    (OUT / "inputs").mkdir(exist_ok=True)
    shutil.copy2(MARKET, OUT / "inputs/market.parquet")
    shutil.copy2(ROOT / "config/510300_existing_data_training_mandate_v1.json", OUT / "inputs/authority_snapshot.json")
    protocol = {**read(OUT / "design_registration.json"), "frozen_at": now(), "source_finalization": source,
                "implementation": str(Path(__file__).relative_to(ROOT)),
                "economic_boundary": "累计利润增长与股价反应是描述状态，不是宏观意外识别，不能说明因果或新信息未被定价。",
                "overlap": "月度20日事件标签可能重叠，毛收益均值不是可并行投资收益；后续账户如获准只持一笔。",
                "source_clock_limit": "18份旧月报只具原页历史时钟，所有历史首版不可变性未认证，不等于独立前向记录。"}
    save(OUT / "protocol.json", protocol)
    paths = [Path(__file__), ROOT / "tests/test_industrial_profit_response_v1.py",
             ROOT / "research/lpr_joint_response_v1.py", ROOT / "research/mechanism_odds_open_contract_v1.py",
             SOURCE, OUT / "design_registration.json", OUT / "protocol.json", OUT / "source_finalization.json",
             OUT / "prefreeze_tests.json", OUT / "released_profit_records.json"]
    paths.extend((OUT / "inputs").glob("*"))
    paths.extend(ROOT / r["raw_path"] for r in read(OUT / "released_profit_records.json"))
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]})
    print("65份原月报、零阈值、固定20日与两个日历子期已冻结。", flush=True)


def run():
    for row in read(OUT / "freeze.json")["files"]:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
    save(OUT / "run_started.json", {"at": now(), "stage": "FIXED_GROSS_SCREEN"})
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    market = market.loc[market.date.le("2025-12-31")].reset_index(drop=True)
    releases = read(OUT / "released_profit_records.json")
    events = build_events(market, releases)
    events.to_parquet(OUT / "events.parquet", index=False)
    events.to_csv(OUT / "全部公告与事件收益.csv", index=False, encoding="utf-8-sig")
    stats = statistics(events)
    stats.to_csv(OUT / "group_statistics.csv", index=False, encoding="utf-8-sig")
    full = stats.loc[stats.period.eq("FULL")].set_index("group")
    parts = stats.loc[stats.group.eq(PRIMARY)].set_index("period")
    gates = {"primary_mean_above_cost_proxy": full.loc[PRIMARY, "mean_gross_return"] > .0028,
             "primary_mean_above_equity_only": full.loc[PRIMARY, "mean_gross_return"] > full.loc["EQUITY_UP_ONLY", "mean_gross_return"],
             "early_positive": parts.loc["EARLY", "mean_gross_return"] > 0,
             "late_positive": parts.loc["LATE", "mean_gross_return"] > 0}
    passed = all(gates.values())
    rng = np.random.default_rng(20260928)
    n = len(events)
    indices = ((rng.integers(0, n, size=(2000, math.ceil(n / 4)))[:, :, None] + np.arange(4)) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=indices.astype(np.int16))
    values = events.gross_return.to_numpy(float)
    means = {}
    for name in [PRIMARY, "EQUITY_UP_ONLY"]:
        selected = mask(events, name).to_numpy(bool) & np.isfinite(values)
        selected_draws = selected[indices]
        totals = np.where(selected_draws, values[indices], 0.).sum(axis=1)
        count = selected_draws.sum(axis=1)
        means[name] = np.divide(totals, count, out=np.full(2000, np.nan), where=count > 0)
    uncertainty = {"method": "原65公告序列4次公告循环区块2000次；未成熟标签仍为空",
                   "primary_mean_95_interval": np.nanquantile(means[PRIMARY], [.025, .975]).tolist(),
                   "primary_minus_equity_only_95_interval": np.nanquantile(means[PRIMARY] - means["EQUITY_UP_ONLY"], [.025, .975]).tolist(),
                   "selection_bias_adjusted": False}
    save(OUT / "uncertainty.json", uncertainty)
    primary = events.loc[events.selected & events.gross_return.notna()]
    concentration = {"event_count": len(primary), "best_event": None, "mean_excluding_best": None}
    if len(primary):
        best = primary.gross_return.idxmax()
        total = primary.gross_return.sum()
        concentration.update(best_event=primary.loc[best].to_dict(),
                             best_share_of_arithmetic_sum=float(primary.loc[best, "gross_return"] / total) if total != 0 else None,
                             mean_excluding_best=primary.drop(index=best).gross_return.mean())
    save(OUT / "concentration.json", concentration)
    mature_primary = primary.loc[primary.status.eq("MATURE")].sort_values("entry_idx")
    overlaps = [[a.stat_month, b.stat_month] for a, b in zip(mature_primary.iloc[:-1].itertuples(), mature_primary.iloc[1:].itertuples()) if b.entry_idx < a.exit_idx]
    # 改变未来行情和未来公告字段，不得改变已经形成的决策。
    cutoff = pd.Timestamp("2023-01-01")
    altered_market = market.copy()
    altered_market.loc[altered_market.date.ge(cutoff), ["open", "high", "low", "close", "previous_close"]] *= 1.3
    altered_releases = [{**r, "profit_ytd_reported_yoy_pct": -r["profit_ytd_reported_yoy_pct"]}
                        if r["release_date"] >= "2023-01-01" else dict(r) for r in releases]
    altered = build_events(altered_market, altered_releases)
    prior = events.observation_date.lt(cutoff)
    pd.testing.assert_frame_equal(events.loc[prior, ["equity_response", "selected"]], altered.loc[prior, ["equity_response", "selected"]])
    discrepancies = []
    for row in events.loc[events.status.eq("MATURE")].itertuples():
        dividends = market.iloc[int(row.entry_idx) + 1:int(row.exit_idx) + 1].dividend.to_numpy(float)
        independently = (float(market.iloc[int(row.exit_idx)].open) + math.fsum(dividends)) / float(market.iloc[int(row.entry_idx)].open) - 1
        discrepancies.append(abs(independently - row.gross_return))
    assert max(discrepancies, default=0.) < 1e-12
    result = {"at": now(), "study_id": "510300_INDUSTRIAL_PROFIT_RESPONSE_V1",
              "status": "PASS_GROSS_SCREEN_ACCOUNT_REQUIRED" if passed else "REJECTED_FIXED_GROSS_SCREEN_NO_PARAMETER_RESCUE",
              "release_count": len(events), "statuses": events.status.value_counts().to_dict(),
              "primary": full.loc[PRIMARY].to_dict(), "equity_only": full.loc["EQUITY_UP_ONLY"].to_dict(),
              "fixed_subperiods": parts.loc[["EARLY", "LATE"]].to_dict("index"),
              "stage_two_gate": gates, "selected_overlap_pairs": overlaps,
              "account_status": "REQUIRED_NEXT_STAGE" if passed else "NOT_RUN_GROSS_SCREEN_FAILED",
              "net_sharpe": None, "net_cagr": None, "maximum_drawdown": None,
              "new_accounts": 0, "new_fits": 0, "parameter_searches": 0,
              "goal_achieved": False, "independent_forward_observations": 0, "orders_authorized": False}
    save(OUT / "result.json", result)
    save(OUT / "verification_receipt.json", {"at": now(), "status": "PASS_FROZEN_IMPLEMENTATION_CHECKS",
         "source_records": len(releases), "causal_prefix_checks": 1, "independently_recomputed_mature_labels": len(discrepancies),
         "maximum_label_difference": max(discrepancies, default=0.), "new_accounts": 0,
         "boundary": "来源和计算检查不等于经济通过、独立验证或真实成交。"})
    print(f"工业利润固定初筛完成：主组{len(primary)}次，毛均值{full.loc[PRIMARY, 'mean_gross_return']:.4%}，状态{result['status']}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工业利润原公告固定初筛")
    parser.add_argument("action", choices=["extract", "freeze", "run"])
    action = parser.parse_args().action
    {"extract": extract, "freeze": freeze, "run": run}[action]()
