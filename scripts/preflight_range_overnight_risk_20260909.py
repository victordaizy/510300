"""确认现有日内高低价、除息和隔夜口径，不生成新风险或账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_range_overnight_risk_preflight_20260909"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"


def main():
    c143, c145 = ROOT / "config/510300_trend_noise_reference_blend_v1.json", ROOT / "config/510300_trend_coherence_blend_v1.json"
    base, previous = [json.loads(path.read_text(encoding="utf-8")) for path in [c143, c145]]
    bound = {str(Path(item["path"])): item["sha256"] for item in previous["frozen_files"]}
    sources = [c143, c145, ROOT / base["features"], ROOT / base["dividends"], ROOT / "docs/510300_RANGE_OVERNIGHT_RISK_NEXT_20260909.md",
        ROOT / "research/intraday_overnight_increment_v1.py", P143 / "saved_verification_receipt.json",
        ROOT / "reports/research/510300_trend_coherence_blend_v1/saved_verification_receipt.json"]
    for key in ["features", "dividends"]:
        require(digest(ROOT / base[key]) == bound[str(Path(base[key]))], "现有高低价或分红不再等于145冻结来源")
    data = pd.read_parquet(ROOT / base["features"])
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "现有高低价日历不完整递增")
    quotes = data[["open", "high", "low", "close"]]
    require(np.isfinite(quotes).all().all() and quotes.gt(0).all().all(), "现有开高低收存在缺失或非正值")
    require(data.high.ge(data[["open", "close", "low"]].max(axis=1)).all() and data.low.le(data[["open", "close", "high"]].min(axis=1)).all(), "最高最低与开收盘顺序矛盾")
    np.testing.assert_allclose(data.previous_close, data.close.shift(), atol=0, rtol=0, equal_nan=True)
    dividends = normalize_dividends(pd.read_csv(ROOT / base["dividends"]))
    events = dividends.groupby("ex_date").cash_dividend_per_share.sum()
    require(set(events.index).issubset(set(dates)), "已登记除息事件不在行情日历")
    expected_dividends = data.date.map(events).fillna(0)
    np.testing.assert_allclose(data.dividend, expected_dividends, atol=0, rtol=0)
    require(data.dividend.ge(0).all(), "现有每份分红出现负值")
    night = np.log((data.open+data.dividend)/data.previous_close)
    day = np.log((data.close+data.dividend)/(data.open+data.dividend))
    np.testing.assert_allclose(data.overnight_log, night, atol=1e-12, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(data.intraday_log, day, atol=1e-12, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(data.total_log, night+day, atol=1e-12, rtol=1e-12, equal_nan=True)
    require(data.overnight_log.isna().sum() == 1 and pd.isna(data.overnight_log.iloc[0]), "隔夜收益在首日以外缺失")
    checks = []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, base["evaluation_start"]) if period == "evaluation" else (data[data.date.le(base["earlier_terminal"])], base["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        require(first-1 >= 20, "新风险输入没有完整20日暖身")
        required = frame[["high", "low", "dividend", "overnight_log"]].notna().all(axis=1).astype(int).rolling(20, min_periods=20).sum()
        require(required.iloc[indices].eq(20).all(), "判断原点缺少完整20日已知输入")
        for cost in base["costs"]:
            path = P143 / period / cost / "TREND_NOISE_REFERENCE_BLEND_decisions.parquet"
            require(digest(path) == bound[str(path.relative_to(ROOT))], "143父目标不再等于145绑定来源")
            parent = pd.read_parquet(path, columns=["origin", "origin_index", "execution_date", "reference_weight"])
            require(np.array_equal(parent.origin_index, indices) and pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), "143父目标收盘日历不同")
            require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "143父目标不是下一开盘")
            require(parent.reference_weight.between(0, 1).all(), "143父目标未知或越界")
            sources.append(path)
            checks.append({"period": period, "cost": cost, "decision_origins": len(indices), "complete_20_day_input_origins": len(indices),
                "parent_targets_known": len(parent), "next_open_clock_checked": True})
    receipt = {"checked_at": now(), "status": "RANGE_OVERNIGHT_DIVIDEND_IDENTITIES_AND_PARENT_TARGET_CLOCKS_READY",
        "market_rows": len(data), "dividend_events": len(dividends), "quote_order_violations": 0,
        "overnight_intraday_total_log_identity_checked": True, "checks": checks,
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(sources))],
        "new_risk_multipliers_or_accounts": 0, "candidate_registered": False, "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", receipt, exclusive=True)
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 145, "高低区间风险的前序轮次不同")
    index["next_work"].update(status="RANGE_OVERNIGHT_RISK_INPUTS_READY", input_preflight=str((OUT / "result.json").relative_to(ROOT)), registered=False)
    index["updated_at"] = now()
    write_json(path, index)
    print(json.dumps({"行情行": len(data), "除息事件": len(dividends), "价格顺序及收益恒等式": "已确认", "输入": checks,
        "新增风险乘数或账户": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
