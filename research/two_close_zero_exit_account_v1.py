"""每条账户独立读取截至本收盘的连续零次数，不修改原参考账户。"""
import numpy as np

from research.event_account_indexed_request_v1 import simulate_indexed_request_account
from research.intraday_overnight_increment_v1 import require
from research.two_close_zero_exit_inputs_v1 import PRIMARY, confirmation_request, zero_streak


def simulate_confirmed_zero(data, dividends, cfg, cost, start, model, targets=None, prediction=None, horizon=1, event_mask=None):
    require(model == PRIMARY and targets is not None and prediction is None, '明确零退出只使用原保存目标')
    require(event_mask is not None and np.asarray(event_mask).all(), '本轮必须逐个收盘检查')
    counts = zero_streak(targets)

    def request(account, price, value, settings, model_id, origin_index):
        return confirmation_request(account, price, value, settings, model_id, int(counts[origin_index]))

    return simulate_indexed_request_account(data, dividends, cfg, cost, start, model,
        targets=targets, event_mask=event_mask, request_policy=request)
