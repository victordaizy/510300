"""两套已知目标取较小值，任一未知保持未知。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "CONSENSUS_REFERENCE_TARGET"
MODELS = ["VINTAGE_REFERENCE_RISK", "MODEL_SUPPORT_REFERENCE_ROUTER"]


def consensus_target_frames(data, parents_by_cost, cfg, start):
    require(cfg["combination"] == "MINIMUM_OF_TWO_KNOWN_PARENT_TARGETS", "共同目标定义改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    summaries = []
    for cost, frame in frames.items():
        x, y = (frame[model+"_parent_target"].to_numpy(float) for model in MODELS)
        frame["target"] = np.minimum(x, y)
        frame["both_parents_positive"] = (x > 0) & (y > 0)
        frame["both_parents_known"] = np.isfinite(x) & np.isfinite(y)
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(eligible), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
