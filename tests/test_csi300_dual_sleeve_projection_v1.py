import pandas as pd

from research.csi300_dual_sleeve_projection_v1 import project_parent_targets


def test_overlap_weight_is_selected_before_single_sleeve() -> None:
    frame = pd.DataFrame(
        [
            {"signal_date": "2025-01-01", "con_code": "OVERLAP", "regime": "BULL", "target_weight": 0.2083, "score": 0.8},
            {"signal_date": "2025-01-01", "con_code": "C1", "regime": "BULL", "target_weight": 0.1333, "score": 0.9},
            {"signal_date": "2025-01-01", "con_code": "C2", "regime": "BULL", "target_weight": 0.1333, "score": 0.7},
            {"signal_date": "2025-01-01", "con_code": "D1", "regime": "BULL", "target_weight": 0.0750, "score": 0.99},
        ]
    )
    result = project_parent_targets(frame, 3)
    assert result["con_code"].tolist() == ["OVERLAP", "C1", "C2"]
    assert result["selection_rank"].tolist() == [1, 2, 3]
