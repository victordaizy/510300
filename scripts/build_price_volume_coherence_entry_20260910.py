"""复制已有四账户运行入口，替换为一套固定价量相关方法。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'research/session_signed_rank_v1.py').read_text(encoding='utf-8')
    for before, after in [('session_signed_rank', 'price_volume_coherence'), ('SESSION_SIGNED_RANK', 'PRICE_VOLUME_COHERENCE'),
        ('日内隔夜带符号排序', '价量相关与上涨动量'), ('第162轮', '第163轮'), ('round=162', 'round=163'),
        ('config/510300_drawdown_depth_risk_v1.json', 'config/510300_session_signed_rank_v1.json'),
        ('PROGRESS_ROUND161_COMPLETED', 'PROGRESS_ROUND162_COMPLETED')]:
        text = text.replace(before, after)
    start = text.index('        rank_window=60,')
    end = text.index('        new_model_fits=0,', start)
    text = text[:start]+'''        correlation_window=20, momentum_window=20, volatility_window=20, risk_target=.1,
        entry_threshold=.2, exit_threshold=-.2, confirmation_closes=2,
        correlation_method="PEARSON_CONTEMPORANEOUS_TOTAL_LOG_RETURN_AND_SHARE_VOLUME_LOG_CHANGE",
        constant_window="UNDEFINED_NO_VIEW_RESET_STATE", volume_unit="share",
        missing_window="WHOLE_WINDOW_UNKNOWN_RESET_DIRECTION_AND_ALL_THREE_COUNTERS",
        direction="TWO_JOINT_ENTRY_OR_SEPARATELY_CONFIRMED_CORRELATION_OR_MOMENTUM_EXIT",
        initial_direction=0, signal_parameters_fitted=False,
'''+text[end:]
    with (ROOT/'research/price_volume_coherence_v1.py').open('x', encoding='utf-8') as stream:
        stream.write(text)
    print('第163轮价量相关的冻结与四账户入口已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
