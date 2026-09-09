"""两条有效的零一持仓状态取并集，缺失仍保持无观点。"""
import numpy as np

from research.intraday_overnight_increment_v1 import require


def union_targets(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    require(first.ndim == second.ndim == 1 and first.shape == second.shape, "合并信号长度或维数不符")
    for values in [first, second]:
        require(np.isin(values[np.isfinite(values)], [0., 1.]).all(), "专家状态不是原零一判断")
        require(not np.isinf(values).any(), "专家状态不能是无穷值")
    valid = np.isfinite(first) & np.isfinite(second)
    result = np.full(len(first), np.nan)
    result[valid] = np.maximum(first[valid], second[valid])
    return result
