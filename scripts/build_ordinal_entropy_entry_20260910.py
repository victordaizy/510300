"""复用四账户入口，登记固定价格顺序分布方法。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'research/price_volume_coherence_v1.py').read_text(encoding='utf-8')
    for before, after in [('price_volume_coherence', 'ordinal_entropy'), ('PRICE_VOLUME_COHERENCE', 'ORDINAL_ENTROPY'),
        ('价量相关与上涨动量', '价格顺序分布与上涨动量'), ('第163轮', '第164轮'), ('round=163', 'round=164'),
        ('config/510300_session_signed_rank_v1.json', 'config/510300_price_volume_coherence_v1.json'),
        ('PROGRESS_ROUND162_COMPLETED', 'PROGRESS_ROUND163_COMPLETED')]:
        text = text.replace(before, after)
    start = text.index('        correlation_window=20,')
    end = text.index('        new_model_fits=0,', start)
    text = text[:start]+'''        entropy_window=60, pattern_order=3, momentum_window=60, volatility_window=20, risk_target=.1,
        entry_threshold=.9, exit_threshold=.95, confirmation_closes=2,
        entropy_method="NORMALIZED_SHANNON_OF_58_OVERLAPPING_THREE_DAY_WEALTH_ORDERS",
        exact_tie_method="OLDER_DATE_FIRST_NO_JITTER_NO_ROUNDING", momentum_complete_wealth_count=61,
        missing_window="WHOLE_WINDOW_UNKNOWN_RESET_DIRECTION_AND_ALL_THREE_COUNTERS",
        direction="TWO_LOW_ENTROPY_POSITIVE_MOMENTUM_OR_SEPARATE_HIGH_ENTROPY_NONPOSITIVE_MOMENTUM_EXIT",
        initial_direction=0, signal_parameters_fitted=False,
'''+text[end:]
    with (ROOT/'research/ordinal_entropy_v1.py').open('x', encoding='utf-8') as stream:
        stream.write(text)
    print('第164轮价格顺序分布冻结与四账户入口已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
