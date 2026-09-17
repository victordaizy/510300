"""静态复用共同账户入口，替换为无训练的双向市场状态。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT/'research/median_slope_risk_v1.py').read_text(encoding='utf-8')
text = text.replace('median_slope_risk', 'sequential_return_state').replace('median_slope_risk_frames', 'sequential_return_frames')
text = text.replace('sequential_return_state_frames', 'sequential_return_frames')
text = text.replace('MEDIAN_SLOPE_RISK', 'SEQUENTIAL_RETURN_STATE')
text = text.replace('第155轮', '第160轮').replace('round=155', 'round=160').replace('中位数斜率', '双向累计收益状态').replace('成对斜率', '双向收益状态')
text = text.replace('单一中位数斜率', '单一双向收益状态')
text = text.replace('510300_boundary_rebalance_v1.json', '510300_median_slope_risk_v1.json')
text = text.replace('slope_window=60, slope_pair_count=1770, slope_estimator="MEDIAN_OF_ALL_LOG_WEALTH_PAIR_SLOPES", volatility_window=20,',
    'volatility_window=20, drift_allowance=.5, alarm_threshold=5., scale_clock="PREVIOUS_TWENTY_DAYS_EXCLUDING_TODAY", baseline_return=0., reset_rule="BOTH_ACCUMULATORS_ZERO_AFTER_EITHER_ALARM",')
text = text.replace('direction="POSITIVE_SLOPE_LONG_KNOWN_NONPOSITIVE_ZERO", equal_slope="KNOWN_ZERO_TARGET",',
    'direction="UPWARD_ALARM_LONG_DOWNWARD_ALARM_ZERO_ELSE_PREVIOUS_STATE", initial_state="KNOWN_NONPOSITIVE_AFTER_COMPLETE_SCALE",')
text = text.replace('missing_window="FULL_CONTIGUOUS_WINDOW_REQUIRED"', 'missing_window="UNKNOWN_TODAY_RESET_INTERNAL_STATE_REQUIRE_COMPLETE_PREVIOUS_TWENTY"')
text = text.replace('model_fit_count_scope="SUPERVISED_RETURN_MODELS_ONLY_NOT_ROLLING_DESCRIPTIVE_SLOPE_ESTIMATES"', 'model_fit_count_scope="ZERO_SUPERVISED_MODELS_DESCRIPTIVE_DAILY_RECURSION_ONLY"')
text = text.replace('PROGRESS_ROUND154_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET', 'PROGRESS_ROUND159_COMPLETED_CLOSED_AND_DELIVERED_FULL_GOAL_NOT_MET')
text = text.replace('NEXT_20260909.md', 'NEXT_20260910.md')
text = text.replace('第160轮一个固定成对斜率规则', '第160轮一个固定双向收益状态规则')
path = ROOT/'research/sequential_return_state_v1.py'
with path.open('x', encoding='utf-8') as stream:
    stream.write(text)
print('第160轮复用账户入口已生成，尚未冻结或计算历史账户。')
