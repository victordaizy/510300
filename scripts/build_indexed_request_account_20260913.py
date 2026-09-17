"""在既有申请回调中明确传入收盘索引，记账流程保持原样。"""
from pathlib import Path

from research.intraday_overnight_increment_v1 import require


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root/'research/event_account_request_callback_v1.py').read_text(encoding='utf-8')
    source = source.replace('def simulate_request_account(', 'def simulate_indexed_request_account(', 1)
    old = 'request_policy(account, price, value, config, model_id)'
    require(source.count(old) == 1, '回调签名不同')
    source = source.replace(old, 'request_policy(account, price, value, config, model_id, t)', 1)
    with (root/'research/event_account_indexed_request_v1.py').open('x', encoding='utf-8') as stream:
        stream.write(source)
    print('已创建明确传入当前收盘索引的申请入口。', flush=True)


if __name__ == '__main__':
    main()
