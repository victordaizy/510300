"""把冻结双袖带父目标机械投影成三个小账户整仓目标。"""

from __future__ import annotations

import pandas as pd


def project_parent_targets(parent_targets: pd.DataFrame, holdings: int = 3) -> pd.DataFrame:
    required = {"signal_date", "con_code", "regime", "target_weight", "score"}
    if missing := required.difference(parent_targets.columns):
        raise ValueError(f"父目标缺少字段：{sorted(missing)}")
    selected = (
        parent_targets.sort_values(
            ["signal_date", "target_weight", "score", "con_code"],
            ascending=[True, False, False, True],
        )
        .groupby("signal_date", sort=True)
        .head(holdings)
        .copy()
    )
    selected["selection_rank"] = selected.groupby("signal_date").cumcount() + 1
    selected["parent_target_weight"] = selected["target_weight"]
    selected["parent_score"] = selected["score"]
    return selected[
        ["signal_date", "con_code", "selection_rank", "regime", "parent_target_weight", "parent_score"]
    ].reset_index(drop=True)
