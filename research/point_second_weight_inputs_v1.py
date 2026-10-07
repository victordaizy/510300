"""在同一账户引擎中选择原有来源，保留目标大小、零值和未知状态。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.point_account_nr7_inputs_v1 import PARENT_A, PARENT_B
from research.point_weight_information_inputs_v1 import weight_account


def source_parents(parents: pd.DataFrame, source: str) -> pd.DataFrame:
    if source not in (PARENT_A, PARENT_B):
        raise ValueError("只允许已登记的两条固定来源。")
    if not {"origin", source}.issubset(parents.columns):
        raise ValueError("指定来源或日期缺失，不回退到其他候选。")
    if parents.origin.isna().any() or parents.origin.duplicated().any():
        raise ValueError("来源日期缺失或重复。")
    values = pd.to_numeric(parents[source], errors="raise").astype(float)
    bad = values.notna() & (~np.isfinite(values) | values.lt(0) | values.gt(1))
    if bad.any():
        raise ValueError("来源目标必须为0至1的有限值，未知值保留，不自动修补。")
    # 旧引擎以A列名读取输入；这里仅适配字段，结果逐表附带真实来源身份。
    return pd.DataFrame({"origin": parents.origin.copy(), PARENT_A: values.copy()})


def source_account(data, dividends, parents, risks, cost, start, source):
    result = weight_account(data, dividends, source_parents(parents, source), risks, cost, start)
    for name in ("daily", "trades", "orders", "decisions", "rejections"):
        result[name]["signal_source"] = source
    result["terminal"]["signal_source"] = source
    return result
