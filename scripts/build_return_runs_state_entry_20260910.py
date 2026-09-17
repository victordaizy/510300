"""复用四账户入口，保存固定收益强弱连续段规则。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'research/ordinal_entropy_v1.py').read_text(encoding='utf-8')
    for before, after in [('ordinal_entropy', 'return_runs_state'), ('ORDINAL_ENTROPY', 'RETURN_RUNS_STATE'),
        ('价格顺序分布与上涨动量', '收益强弱连续段与上涨动量'), ('第164轮', '第165轮'), ('round=164', 'round=165'),
        ('config/510300_price_volume_coherence_v1.json', 'config/510300_ordinal_entropy_v1.json'),
        ('PROGRESS_ROUND163_COMPLETED', 'PROGRESS_ROUND164_COMPLETED')]:
        text = text.replace(before, after)
    start = text.index('        entropy_window=60,')
    end = text.index('        new_model_fits=0,', start)
    text = text[:start]+'''        runs_window=60, momentum_window=60, volatility_window=20, risk_target=.1,
        entry_threshold=-1., exit_threshold=0., confirmation_closes=2,
        runs_method="FULL_WINDOW_MEDIAN_STRONG_WEAK_RUN_COUNT_NORMALIZED_WITHOUT_CONTINUITY_CORRECTION",
        exact_tie_method="EXCLUDE_EXACT_WINDOW_MEDIAN_KEEP_TIME_ORDER_NO_ROUNDING",
        undefined_statistic="NO_VIEW_RESET_DIRECTION_AND_COUNTERS",
        missing_window="WHOLE_WINDOW_UNKNOWN_RESET_DIRECTION_AND_ALL_THREE_COUNTERS",
        direction="TWO_LOW_RUNS_POSITIVE_MOMENTUM_OR_SEPARATE_NONNEGATIVE_RUNS_NONPOSITIVE_MOMENTUM_EXIT",
        initial_direction=0, signal_parameters_fitted=False,
'''+text[end:]
    with (ROOT/'research/return_runs_state_v1.py').open('x', encoding='utf-8') as stream:
        stream.write(text)
    print('第165轮收益强弱连续段冻结与四账户入口已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
