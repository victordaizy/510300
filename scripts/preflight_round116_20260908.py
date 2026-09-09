"""仅核对成交份额因子及原训练行覆盖，尚不拟合九因子模型或新账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.signed_volume_balance_inputs_v1 import signed_volume_balance
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "reports/research/510300_signed_volume_balance_preflight_20260908"
    feature_path = root / "reports/research/510300_adaptive_allocation_v1/features.parquet"
    sample_path = root / "reports/research/510300_learned_cycle_exit_v1/all_reference_samples.parquet"
    data = pd.read_parquet(feature_path)
    factors = signed_volume_balance(data)
    samples = pd.read_parquet(sample_path)
    samples = samples[samples.signal.eq("D60_INTRA")]
    values = factors.signed_volume_balance20.to_numpy()
    require(np.isfinite(values[samples.origin_index.to_numpy(int)]).all(), "原1461训练行的新增成交量因子不完整")
    np.testing.assert_allclose((data.close+data.dividend)/data.previous_close-1, data.total_simple, atol=0, rtol=0, equal_nan=True)
    check_count = 0
    for t in range(19, len(data)):
        frame = data.iloc[t-19:t+1]
        valid = frame[["close", "previous_close", "dividend", "volume"]].notna().all(axis=1)
        if not valid.all():
            require(np.isnan(values[t]), "新增因子越过缺失窗口")
            continue
        directions = []
        for row in frame.itertuples():
            diff = row.close+row.dividend-row.previous_close
            tolerance = 1e-12*max(1., abs(row.close+row.dividend), abs(row.previous_close))
            directions.append(0. if abs(diff) <= tolerance else 1. if diff > 0 else -1.)
        scalar = sum(direction*volume for direction, volume in zip(directions, frame.volume, strict=True))/frame.volume.sum()
        require(abs(scalar-values[t]) < 1e-12, "向量计算与逐日有符号份额不同")
        check_count += 1
    ex = data[data.dividend.gt(0)].copy()
    ex["raw_direction"] = np.sign(ex.close-ex.previous_close)
    ex["total_return_direction"] = factors.adjusted_return_direction.iloc[ex.index].to_numpy()
    ex["direction_changed_after_dividend"] = ex.raw_direction.ne(ex.total_return_direction)
    out.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(out / "factor_availability.parquet", index=False)
    ex[["date", "close", "previous_close", "dividend", "raw_direction", "total_return_direction", "direction_changed_after_dividend"]].to_csv(out / "除息方向差异.csv", index=False, encoding="utf-8-sig")
    status = {"checked_at": now(), "status": "FACTOR_DEFINITION_AND_ORIGINAL_SAMPLE_COVERAGE_COMPLETE_NOT_FITTED", "factor_window": 20,
              "factor_definition": "连续二十日：含分红上涨日记正成交份额、下跌记负、平盘记零；净有符号份额除全部成交份额。仅浮点平盘比较使用相对一万亿分之一容差。",
              "calendar_rows": len(data), "known_rows": int(np.isfinite(values).sum()), "unknown_rows": int(np.isnan(values).sum()),
              "scalar_windows_checked": check_count, "original_training_rows": len(samples), "known_original_training_rows": int(np.isfinite(values[samples.origin_index.to_numpy(int)]).sum()),
              "missing_or_negative_volume_rows": int((~np.isfinite(data.volume)|data.volume.lt(0)).sum()), "zero_volume_rows": int(data.volume.eq(0).sum()),
              "distribution_rows": len(ex), "dividend_adjustment_changes_direction_rows": int(ex.direction_changed_after_dividend.sum()),
              "new_models": 0, "new_accounts": 0, "new_reference_accounts": 0,
              "source_url": "https://www.tradingview.com/support/solutions/43000502593-on-balance-volume-obv/",
              "source_boundary": "官方OBV按每日涨跌给成交量正负；本项目额外采用含分红方向及固定二十日归一化，不能声称是标准累计OBV、主动买卖净额、公募申购或真实净流入。",
              "bounded_comparison": "原56按日内收盘位置加权成交份额并筛进入，原101按典型价方向汇总成交额并按14日占比恢复进出，原112用区间中点与量做14日零线上穿。当前候选拟以新成交份额输入增强114周期内继续价值，保留其平均截距和全部进入退出；旧三策略及115失败不重开。research及scripts中同名/符号计算命中只为收益方向准确率和其他无关代码，未见该公式。",
              "next_state": "ROUND116_NOT_REGISTERED_FIT_AND_EXECUTION_TESTS_PENDING",
              "sources": [{"path": str(path.relative_to(root)), "sha256": digest(path)} for path in [feature_path, sample_path, root / "research/signed_volume_balance_inputs_v1.py", Path(__file__)]]}
    write_json(out / "result.json", status, exclusive=True)
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
