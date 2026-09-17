"""为持续回撤双风险上限静态复用目标账户入口。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT/'research/sequential_return_state_v1.py').read_text(encoding='utf-8')
text = text.replace('sequential_return_state', 'drawdown_depth_risk').replace('sequential_return_frames', 'drawdown_depth_frames')
text = text.replace('SEQUENTIAL_RETURN_STATE', 'DRAWDOWN_DEPTH_RISK').replace('第160轮', '第161轮').replace('round=160', 'round=161')
text = text.replace('双向累计收益状态', '趋势与持续回撤双上限').replace('双向收益状态', '趋势与持续回撤双上限')
text = text.replace('510300_median_slope_risk_v1.json', '510300_sequential_return_state_v1.json')
start, end = text.index('        volatility_window=20,'), text.index('        new_model_fits=0,')
text = text[:start]+'''        trend_window=120, volatility_window=20, drawdown_window=60, risk_target=.1, drawdown_budget=.04,
        direction="POSITIVE_CURRENT_TOTAL_WEALTH_DEVIATION_FROM_120_DAY_MEAN", equal_trend="KNOWN_ZERO_TARGET",
        drawdown_method="EACH_WINDOW_PREFIX_PEAK_ROOT_MEAN_SQUARED_DRAWDOWNS", first_window_drawdown=0.,
        risk_combination="MINIMUM_OF_ORDINARY_VOLATILITY_AND_LOCAL_DRAWDOWN_CAPS", zero_drawdown_cap=1.,
        missing_window="FULL_CONTIGUOUS_INPUTS_KNOWN_NONPOSITIVE_TREND_ZERO_OTHERWISE_UNKNOWN",
''' + text[end:]
text = text.replace('ZERO_SUPERVISED_MODELS_DESCRIPTIVE_DAILY_RECURSION_ONLY', 'ZERO_SUPERVISED_MODELS_DESCRIPTIVE_WINDOW_STATISTICS_ONLY')
text = text.replace('PROGRESS_ROUND159_COMPLETED', 'PROGRESS_ROUND160_COMPLETED')
with (ROOT/'research/drawdown_depth_risk_v1.py').open('x', encoding='utf-8') as stream:
    stream.write(text)
print('第161轮持续回撤双上限入口已生成。')
