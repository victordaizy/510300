"""生成单一抛物线转向完整账户入口，不运行历史。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "research/parabolic_reversal_v1.py"
assert not target.exists(), "本轮入口已存在，停止覆盖"
source = (ROOT / "research/atr_trend_bands_v1.py").read_text(encoding="utf-8")
source = source.replace("atr_trend_bands_inputs_v1 import band_frame", "parabolic_reversal_inputs_v1 import parabolic_frame")
source = source.replace("atr_trend_bands_v1", "parabolic_reversal_v1").replace("atr_trend_bands_inputs_v1", "parabolic_reversal_inputs_v1")
source = source.replace("ATR_TREND_BANDS", "PARABOLIC_REVERSAL").replace("平均真实波幅通道穿越进入与退出", "抛物线价格转向进入与退出")
source = source.replace("平均真实波幅通道", "抛物线价格转向").replace("通道", "转向规则")
source = source.replace('def freeze():\n', 'def freeze():\n    require(not CONFIG.exists(), "本轮已登记，不能重复冻结")\n')
source = source.replace('"config/510300_volatility_expert_router_v1.json"', '"config/510300_atr_trend_bands_v1.json"')
source = source.replace('round=66', 'round=74')
source = source.replace('atr_window=10, band_multiplier=3.0, seed="MEAN_FIRST_TEN_CONSECUTIVE_TRUE_RANGES_THEN_WILDER",\n        first_mature_direction="DOWN", missing="NO_VIEW_RESET_FULL_WARMUP_KEEP_ACTUAL_SHARES",',
    'acceleration=.02, maximum_acceleration=.2, seed="TWO_COMPLETE_BARS_DM_TIE_UP_THEN_SAME_BAR_TOUCH_CHECK",\n        first_mature_direction="DETERMINED_BY_FIXED_INITIALIZATION", missing="NO_VIEW_RESET_TWO_VALID_BARS_KEEP_ACTUAL_SHARES",')
source = source.replace('previous_goal_turn_classification="PROGRESS_ROUNDS64_65_COMPLETED"', 'previous_goal_turn_classification="PROGRESS_ROUND73_COMPLETED_AND_DELIVERED"')
source = source.replace('OUT / "tests_receipt.json", OUT / "wealth_scale_receipt.json", ROOT / "config/510300_research_authority_v6.json"]',
    'OUT / "tests_receipt.json", OUT / "wealth_scale_receipt.json", ROOT / "config/510300_research_authority_v6.json",\n        ROOT / "docs/reference/ta_lib_sar_20260908/ta_SAR.c", ROOT / "docs/reference/ta_lib_sar_20260908/LICENSE.txt", ROOT / "docs/reference/ta_lib_sar_20260908/source_receipt.json"]')
source = source.replace('    data = pd.read_parquet(ROOT / cfg["features"])\n    errors = {}',
    '    for item in old["frozen_files"]:\n        require(digest(ROOT / item["path"]) == item["sha256"], "原完整账户来源身份改变")\n    require(cfg["weight_band"] == .1, "原整手仓位调整带宽不是10个百分点")\n    data = pd.read_parquet(ROOT / cfg["features"])\n    errors = {}')
source = source.replace('第66轮一个10日波幅、3倍转向规则设置已冻结', '第74轮一个0.02步长、0.2上限转向设置已冻结')
source = source.replace('band_frame(data, cfg["atr_window"], cfg["band_multiplier"])', 'parabolic_frame(data, cfg["acceleration"], cfg["maximum_acceleration"])')
source = source.replace('        require(np.isfinite(target[anchor:-1]).all(), "实际评价的转向规则状态未完整成熟，保持无观点并停止")\n', '')
compile(source, str(target), "exec")
target.write_text(source, encoding="utf-8")
print("第74轮独立价格转向入口已生成，尚未登记或计算新账户。")
