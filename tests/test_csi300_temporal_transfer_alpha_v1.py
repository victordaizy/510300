"""十因子时间迁移的防前视与保留规则测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.csi300_temporal_transfer_alpha_v1 import FEATURE_COLUMNS, predict_and_select


class _ScoreModel:
    def predict(self, values):
        return values.iloc[:, 0].to_numpy()


def test_feature_budget_is_exactly_ten() -> None:
    assert len(FEATURE_COLUMNS) == 10
    assert len(set(FEATURE_COLUMNS)) == 10


def test_retention_buffer_keeps_existing_name_inside_top_ten() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-02-03"])
    rows = []
    for date_index, date in enumerate(dates):
        for rank in range(1, 13):
            score = 1.0 - rank / 100.0
            if date_index == 1 and rank == 3:
                score = 0.91
            row = {"date": date, "con_code": f"S{rank:02d}", "signal_output": "SIGNAL_READY"}
            for column_index, column in enumerate(FEATURE_COLUMNS):
                row[column] = score if column_index == 0 else 0.5
            rows.append(row)
    features = pd.DataFrame(rows)
    rules = type("Rules", (), {"features": FEATURE_COLUMNS})()
    _, selected = predict_and_select(_ScoreModel(), features, rules, holdings=3, retention_rank=10)
    second = selected.loc[selected["date"].eq(dates[1]), "con_code"].tolist()
    assert set(second) == {"S01", "S02", "S03"}

