"""测量510300折价修复空间与成本，不把事后净值当作当时可成交报价。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.intraday_overnight_increment_v1 as engine
import research.repo_segmentation_daily_v1 as market_source
from research.selected_mix_reappraisal_v1 import read, save, digest, now

OUT = ROOT / "reports/research/510300_etf_discount_compensation_probe_v1"
STUDY = "510300_ETF_DISCOUNT_COMPENSATION_PROBE_V1"
HISTORICAL = ROOT / "data/external_validation/510300_etf_microstructure_dual_shadow_reverse_validation_v1_0_2/snapshots/20260828T125705_0800/510300_nav_daily.parquet"
CURRENT = ROOT / "data/raw/fund/510300_nav_daily_raw.parquet"
QUALITY = ROOT / "reports/data_quality/510300_etf_share_premium_level_full_v1.json"
SOURCE_MERGE = ROOT / "data/raw/flow/510300_etf_share_premium_level_full_v1.parquet"
SSE_URL = "https://etf.sse.com.cn/fund/learning/strategy/c/5704303.shtml"
PRIMARY_BUDGET = 100000.


def space(price, nav, cost):
    """假设标的净值不变并完全收敛：只比较报价空间，非策略收益上界。"""
    buy = engine.fill_price(float(price), 1, cost, .001)
    sell = engine.fill_price(float(nav), -1, cost, .001)
    quantity = engine.affordable_quantity(PRIMARY_BUDGET, buy, cost, 100)
    buy_fee, sell_fee = engine.commission(quantity, buy, cost), engine.commission(quantity, sell, cost)
    paid = quantity * buy + buy_fee
    gain = quantity * (sell - buy) - buy_fee - sell_fee
    return {"quantity": quantity, "buy_price": buy, "sell_price_at_unchanged_nav": sell,
            "budget_used": paid, "static_net_cny": gain, "static_net_return": gain / paid,
            "roundtrip_cost_cny": quantity * (buy - price + nav - sell) + buy_fee + sell_fee,
            "static_positive": gain > 0}


def implementation_checks():
    cost = market_source.distribution.COSTS["STRESS"]
    assert space(4., 4., cost)["static_net_cny"] < 0
    assert space(3.95, 4., cost)["static_net_cny"] > 0
    for price, nav in [(4., 4.), (3.95, 4.), (4.02, 4.)]:
        row = space(price, nav, cost)
        assert row["quantity"] % 100 == 0 and row["budget_used"] <= PRIMARY_BUDGET + 1e-8
        assert abs(row["static_net_cny"] - (row["quantity"] * (nav - price) - row["roundtrip_cost_cny"])) < 1e-7
    # 含现金分红的每份价格损益，必须精确等于净值损益和相对净值价差变化。
    p0, p1, n0, n1, cash = 4., 4.08, 4.02, 4.10, .1
    actual = (p1 - p0 + cash) / p0
    nav_part = (n1 - n0 + cash) / p0
    spread_part = ((p1 - n1) - (p0 - n0)) / p0
    np.testing.assert_allclose(actual, nav_part + spread_part, atol=1e-14)
    return {"static_cost_and_lot": True, "budget_not_exceeded": True, "dividend_decomposition_identity": True}


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("折价补偿测量已经固定，不能覆盖。")
    checks = implementation_checks()
    for folder in ["code", "results"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    quality = read(QUALITY)
    for path, sha in quality["source_hashes"].items():
        if path.endswith("nav_daily.parquet") or path.endswith("nav_daily_raw.parquet"):
            assert digest(ROOT / path) == sha
    assert digest(SOURCE_MERGE) == quality["output_sha256"]
    save(OUT / "protocol.json", {
        "at": now(), "study_id": STUDY,
        "question": "ETF相对基金净值的价格让步，是否足以覆盖当前压力费用；之后收益有多少来自净值上涨而非折价修复？",
        "source": "复用已保存的东财与新浪单位净值双来源及其原报价，2021-08-12以前取历史段、当日以后取当前段；按已有510300交易日历对齐。累计净值有拆分口径差异，不使用。",
        "canonical_quote": "采用已核对的现有market.parquet原始收盘价重新计算相对单位净值价差，同时保留原净值文件中的报价差异记录；不改原文件。",
        "clock": "旧feature_asof=15:00只是事后合并标签，不能证明当日单位净值于收盘前已发布。历史首次送达与同步开盘净值均未认证。",
        "static_scenario": "每个已观察收盘报价分别假设净值保持不变且价格完全回到净值，以10万元预算、100份整手、tick0.001、原BASE/STRESS双边费用计算空间。这个有利情景不是交易、不含等待风险，不是实际策略收益上界。",
        "threshold": "预先只使用完整原费用后静态空间>0，阈值由费用、tick和报价算出；不搜索折价分位或回归窗口。",
        "costs": market_source.distribution.COSTS,
        "observed_forward": "固定五个交易日收盘到收盘。每份ETF含息损益=单位净值含息变化+价格减净值的价差变化，全部除以同一初始价格；这是事后归因，不能在知道当日净值后回到当日收盘成交。",
        "delayed_price_probe": "另保存源日下一交易日开盘至第五个后续交易日开盘的实际含息标的收益；只描述已知历史，不认为净值已在该开盘前认证可用，也不当作完整账户。",
        "periods": [["2012-07-02", "2019-12-31"], ["2020-01-02", "2026-09-24"], ["2012-07-02", "2026-09-24"]],
        "subsets": ["全部共同日期", "相对单位净值折价", "BASE静态修复空间为正", "STRESS静态修复空间为正"],
        "dependence": "按完整交易日日历20日循环区块2000次、seed2026092604给平均归因区间；另给贪心五日不重叠数，不称独立样本量。",
        "primary_interpretation": "当前2020年起样本的STRESS空间为正组。全部日期、早期及BASE均完整披露；不选择收益最好的一组晋升。",
        "no_causal_claim": "折价变化不自动代表被迫卖出；净值也可能下跌。份额变化和融资变化的旧失败不借本轮改方向、分位或组合救回。",
        "primary_source_context": {"url": SSE_URL, "publication": "2022-06-23",
            "note": "上交所介绍的实时折价套利包含赎回和成分股交易，还须支付固定及冲击成本；当前只持有510300和现金的范围不包含该套利组合。"},
        "new_accounts": 0, "new_fits": 0, "new_market_downloads": 0,
        "strategy_status": "NOT_RUN_PRICE_INFORMATION_CLOCK_AND_SYNCHRONOUS_NAV_NOT_ESTABLISHED",
        "independent_validation": False, "goal_achieved": False, "orders_authorized": False}, True)
    save(OUT / "implementation_checks.json", checks, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    paths = [Path(__file__), Path(engine.__file__), Path(market_source.__file__), Path(market_source.distribution.__file__), HISTORICAL, CURRENT,
             QUALITY, SOURCE_MERGE, market_source.OUT / "inputs/market.parquet", market_source.OUT / "inputs/mature_labels.parquet"]
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "files": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "before_new_space_results": True}, True)
    print("折价空间、双边成本与五日收益归因已固定；本轮不产生交易账户。", flush=True)


def inputs():
    old, recent = pd.read_parquet(HISTORICAL), pd.read_parquet(CURRENT)
    seam = pd.Timestamp("2021-08-12")
    cols = ["date", "unit_nav", "nav_eastmoney", "nav_sina", "close", "close_premium_to_nav"]
    for frame in [old, recent]:
        frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
    joined = pd.concat([old.loc[old.date.lt(seam), cols], recent.loc[recent.date.ge(seam), cols]], ignore_index=True)
    assert joined.date.is_unique
    joined = joined.rename(columns={"close": "source_quote", "close_premium_to_nav": "source_premium"})
    joined["source_dual_difference"] = joined.nav_eastmoney - joined.nav_sina
    market = pd.read_parquet(market_source.OUT / "inputs/market.parquet")
    market["idx"] = np.arange(len(market))
    frame = market[["idx", "date", "close", "open", "dividend"]].merge(joined, on="date", how="left", validate="one_to_one")
    frame["nav_known_in_archive"] = frame.unit_nav.gt(0) & np.isfinite(frame.unit_nav)
    known = frame[frame.nav_known_in_archive]
    assert known.nav_eastmoney.notna().all() and known.nav_sina.notna().all()
    assert known.source_dual_difference.abs().le(1e-10).all()
    np.testing.assert_allclose(known.source_quote / known.unit_nav - 1, known.source_premium, atol=1e-12, rtol=0)
    frame["source_quote_difference"] = frame.close - frame.source_quote
    frame["premium"] = frame.close / frame.unit_nav - 1
    frame["discount_recovery_gross_fraction"] = frame.unit_nav / frame.close - 1
    for cost_name, cost in market_source.distribution.COSTS.items():
        records = [{"idx": int(row.idx), **space(row.close, row.unit_nav, cost)} for row in known.itertuples()]
        values = pd.DataFrame(records).rename(columns={k: f"{cost_name}_{k}" for k in records[0] if k != "idx"})
        frame = frame.merge(values, on="idx", how="left", validate="one_to_one")
    cash = frame.dividend.rolling(5, min_periods=5).sum().shift(-5)
    frame["exit_idx_close5"] = frame.idx.shift(-5)
    frame["exit_date_close5"] = frame.date.shift(-5)
    frame["etf_return_close5"] = (frame.close.shift(-5) - frame.close + cash) / frame.close
    frame["nav_component_close5"] = (frame.unit_nav.shift(-5) - frame.unit_nav + cash) / frame.close
    spread = frame.close - frame.unit_nav
    frame["spread_component_close5"] = (spread.shift(-5) - spread) / frame.close
    frame["complete_decomposition5"] = frame.nav_known_in_archive & frame.nav_known_in_archive.shift(-5, fill_value=False)
    valid = frame.complete_decomposition5
    np.testing.assert_allclose(frame.loc[valid, "etf_return_close5"],
        frame.loc[valid, "nav_component_close5"] + frame.loc[valid, "spread_component_close5"], atol=1e-12, rtol=0)
    labels = pd.read_parquet(market_source.OUT / "inputs/mature_labels.parquet")
    label_map = labels.set_index("idx")
    frame["delayed_entry_idx"] = frame.idx + 1
    frame["delayed_return_open5"] = frame.delayed_entry_idx.map(label_map.gross_return5)
    frame["delayed_exit_idx"] = frame.delayed_entry_idx.map(label_map.exit_idx)
    frame["delayed_exit_date"] = frame.delayed_entry_idx.map(label_map.exit_date)
    return frame


def greedy_count(indices, horizon):
    end, count = -1, 0
    for idx in np.sort(np.asarray(indices, int)):
        if idx >= end:
            count += 1
            end = idx + horizon
    return count


def mean_intervals(frame, mask, rng):
    columns = ["etf_return_close5", "nav_component_close5", "spread_component_close5", "delayed_return_open5"]
    values = frame[columns].to_numpy(float)
    eligible = np.asarray(mask, bool)[:, None] & np.isfinite(values)
    n = len(frame)
    boot = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / 20)))
        chosen = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
        denominator = eligible[chosen].sum(axis=0)
        numerator = np.where(eligible[chosen], values[chosen], 0.).sum(axis=0)
        boot.append(np.divide(numerator, denominator, out=np.full(len(columns), np.nan), where=denominator > 0))
    boot = np.asarray(boot)
    result = {}
    for j, col in enumerate(columns):
        active = eligible[:, j]
        b = boot[:, j][np.isfinite(boot[:, j])]
        result[col] = {"n": int(active.sum()), "mean": float(values[active, j].mean()) if active.any() else None,
            "lower95": float(np.quantile(b, .025)) if len(b) else None,
            "upper95": float(np.quantile(b, .975)) if len(b) else None,
            "valid_bootstrap_draws": len(b), "greedy_nonoverlap_count": greedy_count(frame.loc[active, "idx"], 5)}
    return result


def run():
    freeze_record = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == freeze_record["protocol_sha256"]
    for path, sha in freeze_record["files"].items():
        assert digest(ROOT / path) == sha, path
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    frame = inputs()
    frame.to_parquet(OUT / "results/observed_discount_and_attribution.parquet", index=False)
    known = frame[frame.nav_known_in_archive]
    mismatch = known[known.source_quote_difference.abs().gt(1e-10)]
    mismatch.to_parquet(OUT / "results/source_quote_differences.parquet", index=False)
    rng = np.random.default_rng(2026092604)
    periods, intervals = [], []
    for name, low, high in [("EARLY", "2012-07-02", "2019-12-31"), ("CURRENT", "2020-01-02", "2026-09-24"),
                            ("ALL", "2012-07-02", "2026-09-24")]:
        scope = frame[frame.date.between(low, high)].copy()
        available = scope.nav_known_in_archive
        observed = scope[available]
        counts = {"account_calendar_days": len(scope), "nav_observed_days": int(available.sum()),
            "nav_missing_days": int((~available).sum()), "discount_days": int(observed.premium.lt(0).sum()),
            "BASE_static_positive_days": int(observed.BASE_static_positive.sum()),
            "STRESS_static_positive_days": int(observed.STRESS_static_positive.sum())}
        source = {"period": name, "start": low, "end": high, **counts,
            "premium_quantiles_bps": {str(q): float(observed.premium.quantile(q) * 10000) for q in [.01, .05, .5, .95, .99]},
            "stress_positive_static_net_return_mean": float(observed.loc[observed.STRESS_static_positive.eq(True), "STRESS_static_net_return"].mean()),
            "stress_positive_static_net_return_median": float(observed.loc[observed.STRESS_static_positive.eq(True), "STRESS_static_net_return"].median()),
            "static_positive_by_year": {str(y): int(part.STRESS_static_positive.sum()) for y, part in observed.groupby(observed.date.dt.year)}}
        periods.append(source)
        masks = {"ALL_AVAILABLE": available, "DISCOUNT": available & scope.premium.lt(0),
            "BASE_STATIC_POSITIVE": available & scope.BASE_static_positive.eq(True),
            "STRESS_STATIC_POSITIVE": available & scope.STRESS_static_positive.eq(True)}
        # 固定时期内只统计在该时期已经结束的结果，不把下一时期结果带回。
        scope.loc[scope.exit_date_close5.gt(high), ["etf_return_close5", "nav_component_close5", "spread_component_close5"]] = np.nan
        scope.loc[scope.delayed_exit_date.gt(high), "delayed_return_open5"] = np.nan
        scope.loc[~scope.complete_decomposition5, ["etf_return_close5", "nav_component_close5", "spread_component_close5"]] = np.nan
        for subset, mask in masks.items():
            intervals.append({"period": name, "subset": subset, "signal_days": int(mask.sum()),
                              "outcomes": mean_intervals(scope, mask, rng)})
        print(f"{name}：{counts['nav_observed_days']}个净值共同日，压力费用后静态空间为正{counts['STRESS_static_positive_days']}日。", flush=True)
    for path, sha in freeze_record["files"].items():
        assert digest(ROOT / path) == sha, path
    result = {"at": now(), "study_id": STUDY, "status": "COMPLETED_DISCOUNT_COMPENSATION_MEASUREMENT",
        "first_nav_date": known.date.min(), "last_nav_date": known.date.max(), "nav_observed_days": len(known),
        "source_quote_difference_days": len(mismatch), "max_source_quote_difference": float(known.source_quote_difference.abs().max()),
        "dual_provider_nav_equal": True, "decomposition_identity_max_error": float((frame.etf_return_close5 - frame.nav_component_close5 - frame.spread_component_close5).abs().max()),
        "periods": periods, "forward_attribution": intervals, "new_accounts": 0, "new_fits": 0, "new_market_downloads": 0,
        "source_clock_verified": False, "synchronous_execution_nav_available": False,
        "strategy_status": "NOT_RUN_PRICE_INFORMATION_CLOCK_AND_SYNCHRONOUS_NAV_NOT_ESTABLISHED",
        "static_space_is_realized_profit": False, "static_space_is_strategy_return_upper_bound": False,
        "original_files_unchanged": True, "independent_forward_observations": 0, "goal_achieved": False,
        "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("折价补偿与五日收益来源测量完成，没有把收盘后净值变成同日可交易信号。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "freeze", "run"])
    action = parser.parse_args().action
    if action == "check":
        print(implementation_checks())
    elif action == "freeze":
        freeze()
    else:
        try:
            run()
        except Exception as exc:
            save(OUT / "RUN_FAILURE.json", {"at": now(), "type": type(exc).__name__, "message": str(exc)}, True)
            raise
