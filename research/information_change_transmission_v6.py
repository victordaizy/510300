"""把事前预期、原因背景与持有期间的新信息按各自时钟连接。"""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_information_change_transmission_v6"
V5 = ROOT / "reports/research/510300_macro_earnings_pricing_bridge_v5"
CONSENSUS = ROOT / "reports/research/510300_money_consensus_increment_v2"
POLICY = ROOT / "reports/research/510300_policy_information_clock_v1"
INPUTS = {
    "monthly.csv": V5 / "results/104个月_盈利定价与宏观波动完整连接.csv",
    "weekly.csv": V5 / "results/445周_盈利定价与宏观波动完整连接.csv",
    "market.csv": V5 / "inputs/market.csv",
    "daily_volatility.csv": ROOT / "reports/research/510300_volatility_money_decomposition_v3/results/每日波动精确分解.csv",
    "forecasts.csv": CONSENSUS / "results/全部月度实际与事前预期_绘图数据.csv",
    "five_day_events.csv": CONSENSUS / "results/全部准入事件_固定五日.csv",
    "prior_predictions.csv": CONSENSUS / "results/A_B全部逐期预测.csv",
    "prior_verdict.json": CONSENSUS / "results/分口径裁决.json",
    "forecast_sources.json": CONSENSUS / "sources/selected_numeric_sources.json",
    "policy_nodes.csv": POLICY / "results/跨通道政策链_完整事实与时钟.csv",
}


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if x is pd.NaT or x is pd.NA:
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return x.isoformat()
    return x


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame, name):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def read(name):
    p = OUT / "inputs" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else pd.read_csv(p)


def local(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert("Asia/Shanghai")


def prepare():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha(OUT / "protocol.json") == frozen["protocol_sha256"]
    receipts = []
    for name, src in INPUTS.items():
        dest = OUT / "inputs" / name
        if dest.exists():
            assert sha(dest) == sha(src)
        else:
            shutil.copy2(src, dest)
        receipts.append({"name": name, "source": str(src.relative_to(ROOT)), "sha256": sha(dest)})
    save("input_receipts.json", {"at": now(), "items": receipts})
    source_records = []
    for row in read("forecast_sources.json"):
        if row["stat_month"] not in ["2021-01", "2024-08", "2025-08"]:
            continue
        src = ROOT / "reports/research/510300_money_consensus_source_extension_v1/raw" / row["filename"]
        assert sha(src) == row["sha256"]
        shutil.copy2(src, OUT / "sources" / src.name)
        source_records.append(row)
    save("sources/三个既定病例的事前报告.json", source_records)
    policy_receipts = []
    for row in read("policy_nodes.csv").to_dict("records"):
        src = POLICY / row["source_path"]
        assert sha(src) == row["source_sha256"]
        dest = OUT / "sources" / src.name
        if dest.exists():
            assert sha(dest) == sha(src)
        else:
            shutil.copy2(src, dest)
        policy_receipts.append({"node_id": row["node_id"], "source_file": dest.name, "source_sha256": sha(dest), "source_url": row["source_url"]})
    save("sources/政策节点原文身份.json", policy_receipts)


def information_join(panel):
    expectations = read("forecasts.csv")
    names = ["consensus_admitted", "forecast_date", "forecast_source", "forecast_url", "forecast_sha256", "forecast_file", "forecast_age_days", "expected_m1_pp", "expected_m2_pp", "expected_spread_proxy_pp", "spread_surprise_proxy_pp", "m1_surprise_pp", "m2_surprise_pp", "published_at", "available_at_upper_bound"]
    x = panel.merge(expectations[["stat_month", *names]].rename(columns={c: "expectation_" + c for c in names}), on="stat_month", how="left", validate="many_to_one").copy()
    x["expectation_admission_status"] = "NO_VIEW_NO_ADMITTED_FORECAST"
    admitted = x.expectation_consensus_admitted.eq(True)
    known = local(x.expectation_available_at_upper_bound)
    original_known = local(x.available_at_upper_bound)
    cutoff = local(x.snapshot_at)
    visible = admitted & known.le(cutoff) & original_known.le(cutoff)
    x.loc[visible, "expectation_admission_status"] = "RECONSTRUCTED_PRERELEASE_PROXY_VISIBLE_AT_ORIGIN"
    x.loc[admitted & ~visible, "expectation_admission_status"] = "NO_VIEW_NOT_YET_KNOWN_BY_ORIGIN"
    x.loc[x.context_snapshot_status.eq("NO_VIEW_AFTER_MARKET_CUTOFF"), "expectation_admission_status"] = "SOURCE_EXISTS_AFTER_FROZEN_MARKET_CUTOFF"
    x["expectation_release_clock_difference_seconds"] = (known - original_known).dt.total_seconds()
    x["expectation_surprise_usable"] = visible & x.context_snapshot_status.ne("NO_VIEW_AFTER_MARKET_CUTOFF")
    x["change_vs_surprise_description"] = "缺少同口径三个月变化或合格预期"
    delta = x.delta3_spread_pp
    surprise = x.expectation_spread_surprise_proxy_pp
    both = x.expectation_surprise_usable & delta.notna()
    tolerance = 1e-10
    dsign = np.sign(delta.where(delta.abs() > tolerance, 0))
    ssign = np.sign(surprise.where(surprise.abs() > tolerance, 0))
    x.loc[both & dsign.eq(ssign) & dsign.ne(0), "change_vs_surprise_description"] = "三个月变化与本次预期差同向"
    x.loc[both & (dsign * ssign).eq(-1), "change_vs_surprise_description"] = "三个月变化与本次预期差反向"
    x.loc[both & (dsign.eq(0) | ssign.eq(0)), "change_vs_surprise_description"] = "至少一项为零"
    return x


def daily_paths(monthly, market, policy):
    market = market.copy()
    market["date"] = pd.to_datetime(market.date)
    market = market.sort_values("date").set_index("date")
    market["close_time"] = market.index.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    policy = policy.copy()
    policy["known"] = local(policy.source_available_upper)
    policy = policy.sort_values(["known", "node_id"])
    paths, splits, links = [], [], []
    for r in monthly.to_dict("records"):
        if r.get("E0_20_status") != "MATURE":
            # 原状态字符串以实际标签非空及退出日期覆盖决定，不创造未成熟收益。
            if pd.isna(r.get("E0_20_return")):
                continue
        entry, end = pd.Timestamp(r["E0_20_entry_date"]), pd.Timestamp(r["E0_20_exit_date"])
        if entry not in market.index or end not in market.index:
            continue
        part = market.loc[entry:end].copy()
        assert len(part) == 20
        p0 = float(part.iloc[0].open)
        snapshot = pd.Timestamp(r["snapshot_at"])
        exit_time = market.loc[end, "close_time"]
        nodes = policy[(policy.known > snapshot) & (policy.known <= exit_time)]
        equity, cash, previous_close = p0, 0.0, p0
        for j, (day, q) in enumerate(part.iterrows()):
            dividend = float(q.dividend) if j > 0 else 0.0
            cash += dividend
            overnight = 0.0 if j == 0 else (float(q.open) - previous_close + dividend) / p0
            intraday = (float(q.close) - float(q.open)) / p0
            equity = float(q.close) + cash
            paths.append({"stat_month": r["stat_month"], "origin_id": r["origin_id"], "day_number": j + 1, "date": day, "entry_date": entry, "original_exit_date": end, "entry_open": p0, "open": q.open, "close": q.close, "earned_dividend_today": dividend, "accumulated_dividend": cash, "overnight_contribution_to_entry_return": overnight, "intraday_contribution_to_entry_return": intraday, "daily_contribution_to_entry_return": overnight + intraday, "path_return": equity / p0 - 1, "original_E0_20_return": r["E0_20_return"], "role": "AFTER_ORIGIN_OUTCOME_NOT_AN_ENTRY_FEATURE"})
            previous_close = float(q.close)
        original = float(r["E0_20_return"])
        np.testing.assert_allclose(equity / p0 - 1, original, atol=2e-13, rtol=2e-13)
        summary = {"stat_month": r["stat_month"], "original_entry_date": entry, "original_exit_date": end, "original_20d_return": original, "recorded_policy_node_count": len(nodes), "policy_catalog_scope": "24个人工核对节点；不完整；节点数不是独立冲击数", "first_node_status": "NO_RECORDED_NODE_NOT_NO_NEW_INFORMATION", "role": "后续解释，不能进入起点输入"}
        for node in nodes.to_dict("records"):
            links.append({"stat_month": r["stat_month"], "entry_date": entry, "exit_date": end, **{k: v for k, v in node.items() if k != "known"}, "role": "POST_ORIGIN_CONTEXT_NOT_AN_ENTRY_FEATURE"})
        if len(nodes):
            first = nodes.iloc[0]
            same_clock = nodes[nodes.known.eq(first.known)]
            before = part[part.close_time < first.known]
            summary.update(first_node_status="RECORDED_NODE_BETWEEN_ORIGIN_AND_ORIGINAL_EXIT", first_node_id=first.node_id, first_node_title=first.title, first_node_ids_same_clock=";".join(same_clock.node_id), first_node_titles_same_clock="；".join(same_clock.title), first_node_known=first.known, first_node_url=first.source_url, first_node_intraday_ambiguity=bool(pd.Timedelta(hours=9, minutes=30) < first.known - first.known.normalize() < pd.Timedelta(hours=15)))
            if len(before):
                d = before.index[-1]
                earned = part.loc[(part.index <= d) & (part.index > entry), "dividend"].sum()
                at_split = float(before.iloc[-1].close) + float(earned)
                pre_contribution = at_split / p0 - 1
                summary.update(split_close_date=d, split_fixed_share_equity=at_split, contribution_before_node=pre_contribution, contribution_after_boundary_to_exit=original - pre_contribution, after_boundary_equity_return=equity / at_split - 1, arithmetic_identity_error=original - pre_contribution - (original - pre_contribution), interpretation="金额路径的先后分解，不是政策因果贡献；盘中时点不能靠日线分解。")
            else:
                summary["split_status"] = "NO_PRE_NODE_HOLDING_CLOSE"
        splits.append(summary)
    daily = pd.DataFrame(paths)
    vol = read("daily_volatility.csv").rename(columns={"observation_date": "date"})
    vol["date"] = pd.to_datetime(vol.date)
    daily = daily.merge(vol, on="date", how="left", validate="many_to_one")
    csv(daily, "原E0二十日_全部逐日贡献与波动变化.csv")
    csv(pd.DataFrame(splits), "原E0二十日_已记录后续政策与路径分段.csv")
    csv(pd.DataFrame(links), "原E0二十日_全部后续政策节点.csv")
    return daily, pd.DataFrame(splits)


def run():
    assert not (OUT / "results/build_receipt.json").exists(), "第六轮已有完成记录，不覆盖。"
    prepare()
    monthly, weekly = information_join(read("monthly.csv")), information_join(read("weekly.csv"))
    for title, frame in [("104个月", monthly), ("445周", weekly)]:
        csv(frame, title + "_原因盈利定价与预期差完整连接.csv")
        future = [c for c in frame if c.startswith(("E0_", "E1_", "delay_change_", "first_reaction_"))]
        csv(frame.drop(columns=future), title + "_不含后续价格标签_财务仍为回顾版本.csv")
    cases = monthly[monthly.stat_month.isin(["2020-06", "2021-01", "2024-08", "2025-07", "2025-08"])]
    csv(cases, "五个固定病例_新信息与传导全表.csv")
    window = monthly[monthly.stat_month.between("2020-04", "2021-02") | monthly.stat_month.between("2024-04", "2025-08")]
    csv(window, "两个固定历史段_完整28个月_新信息与传导.csv")
    descriptions = monthly[monthly.expectation_surprise_usable & monthly.delta3_spread_pp.notna()].groupby(["training_regime", "change_vs_surprise_description"]).agg(months=("stat_month", "size"), stat_months=("stat_month", lambda s: ";".join(s))).reset_index()
    csv(descriptions, "剪刀差三个月变化与本次预期差_分口径描述.csv")
    usable = monthly.expectation_surprise_usable & monthly.delta3_spread_pp.notna()
    diagnostics = []
    for regime, part in monthly[usable].groupby("training_regime"):
        improved = part[part.delta3_spread_pp > 1e-10]
        diagnostics.append({"regime": regime, "both_available": len(part), "improved_over_three_months": len(improved), "improved_but_below_expectation": int((improved.expectation_spread_surprise_proxy_pp < -1e-10).sum()), "improved_and_matches_expectation": int((improved.expectation_spread_surprise_proxy_pp.abs() <= 1e-10).sum()), "purpose": "描述信息含义，不是后续胜率"})
    save("results/information_diagnostics.json", diagnostics)
    daily, splits = daily_paths(monthly, read("market.csv"), read("policy_nodes.csv"))
    # 原五日试验的初次反应和之后收益仅原样保存，不重新计算或扩展训练。
    original_five = read("five_day_events.csv")
    csv(cases[["stat_month", "delta3_spread_pp", "expectation_spread_surprise_proxy_pp", "E0_20_return", "E1_20_return"]].merge(original_five[["stat_month", "pre_date", "observation_date", "initial_response", "entry_date", "exit_date", "residual_5d_gross_return"]], on="stat_month", how="left"), "五病例_原五日与原二十日时钟并列.csv")
    save("results/build_receipt.json", {"at": now(), "study_id": "510300_INFORMATION_CHANGE_TRANSMISSION_V6", "monthly_origins": len(monthly), "weekly_origins": len(weekly), "admitted_source_months": int(read("forecasts.csv").consensus_admitted.sum()), "monthly_visible_contexts": int(monthly.expectation_surprise_usable.sum()), "recorded_policy_nodes": len(read("policy_nodes.csv")), "mature_original_20d_paths": len(splits), "daily_path_rows": len(daily), "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("第六轮已保存事前预期连接、全部原20日逐日路径及后续政策节点；未改动原预测裁决。")


if __name__ == "__main__":
    run()
