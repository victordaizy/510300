"""保留原记账与核对结构，只把申请数量策略改为明确传入的函数。"""
from pathlib import Path

from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    original = (ROOT/'research/event_clock_account_v1.py').read_text(encoding='utf-8')
    transformed = original.replace('def simulate_event_account(', 'def simulate_request_account(', 1)
    old = 'horizon: int = 1, event_mask: np.ndarray | None = None)'
    new = 'horizon: int = 1, event_mask: np.ndarray | None = None, *, request_policy)'
    require(transformed.count(old) == 1, '原账户函数签名不同')
    transformed = transformed.replace(old, new, 1)
    old = 'target_request(account, price, value, config)'
    require(transformed.count(old) == 1, '原目标调用位置不同')
    transformed = transformed.replace(old, 'request_policy(account, price, value, config, model_id)', 1)
    out = ROOT/'research/event_account_request_callback_v1.py'
    with out.open('x', encoding='utf-8') as stream:
        stream.write(transformed)
    source = (ROOT/'research/saved_target_account_checks_v1.py').read_text(encoding='utf-8')
    source = source.replace('def verify_saved_target_accounts(', 'def verify_requested_target_accounts(', 1)
    source = source.replace('expected_targets, comparison_models=()):', 'expected_targets, expected_requests, comparison_models=()):', 1)
    left = source.index('            known = np.isfinite(targets)')
    right = source.index('            execution_requests = requests.copy()', left)
    replacement = '''            requests = np.asarray(expected_requests(targets, prior, old_shares, prices, cfg), dtype=int)
            require(requests.shape == targets.shape, "独立申请数量长度不同")
            require(np.array_equal(decisions.requested_quantity, requests), "实际申请不符合本轮规则或未使用自身净值与份额")
'''
    source = source[:left]+replacement+source[right:]
    out = ROOT/'research/saved_requested_account_checks_v1.py'
    with out.open('x', encoding='utf-8') as stream:
        stream.write(source)
    print('已生成明确接收申请策略的账户和保存核对入口，原程序未修改。', flush=True)


if __name__ == '__main__':
    main()
