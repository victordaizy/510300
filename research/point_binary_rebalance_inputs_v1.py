"""只删除来源仓位大小信息，保留已知/未知、正/零与原账户规则。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.point_account_nr7_inputs_v1 import PARENT_A


def binary_rebalance_parents(parents: pd.DataFrame) -> pd.DataFrame:
    """正目标统一请求原有50%预算；实际成交仍由原风险上限缩减。"""
    if not {"origin", PARENT_A}.issubset(parents.columns):
        raise ValueError("缺少来源日期或固定A目标。")
    if parents.origin.isna().any() or parents.origin.duplicated().any():
        raise ValueError("来源日期缺失或重复。")
    result = parents.copy(deep=True)
    values = pd.to_numeric(result[PARENT_A], errors="raise").astype(float)
    invalid = values.notna() & (~np.isfinite(values) | values.lt(0) | values.gt(1))
    if invalid.any():
        raise ValueError("原目标必须为0至1的有限数或未知值；不自动修补异常目标。")
    result[PARENT_A] = values.mask(values.gt(0), .5)
    return result
