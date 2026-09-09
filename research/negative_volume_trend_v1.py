"""第129轮：单一缩量累计趋势，无新增拟合直接检验完整账户。"""
from pathlib import Path
from research.fixed_signal_account_round_v1 import freeze_signal, run_signal
from research.negative_volume_trend_inputs_v1 import negative_volume_factors, NegativeVolumeController

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_negative_volume_trend_v1"
CONFIG = ROOT / "config/510300_negative_volume_trend_v1.json"
PRIMARY = "NEGATIVE_VOLUME_TREND"


def freeze():
    additions = {"study_id": "510300_NEGATIVE_VOLUME_TREND_V1", "round": 129, "primary": PRIMARY, "name": "缩量日累计涨跌趋势",
        "signal_period": 255, "initial_index": 1000., "ema_seed": "FIRST_255_LEVELS_SIMPLE_MEAN_INCLUDING_INITIAL_1000",
        "recurrence": "ADD_100_TIMES_ECONOMIC_SIMPLE_RETURN_ONLY_ON_STRICT_VOLUME_DECREASE",
        "price_change": "DECIMAL_CLOSE_MINUS_PREVIOUS_PLUS_DIVIDEND_DIVIDED_BY_PREVIOUS",
        "volume": "ACTUAL_TRADED_SHARES", "equal_volume": "KEEP_INDEX", "equal_indicator_and_mean": "KEEP_ACTUAL_STATE",
        "source_gap": "NO_VIEW_REMAINDER_NO_RESEED", "old_cooldown_or_rearm": False,
        "rules": "docs/510300_NEGATIVE_VOLUME_TREND_V1.md", "input_receipt": "reports/research/510300_negative_volume_trend_preflight_20260909/result.json",
        "previous_goal_turn_classification": "PROGRESS_ROUND128_COMPLETED_AND_129_METHOD_REVIEW"}
    freeze_signal(ROOT, OUT, CONFIG, additions, [Path(__file__), ROOT / "research/negative_volume_trend_inputs_v1.py",
        ROOT / "tests/test_negative_volume_trend_v1.py"], 7)


def run():
    run_signal(ROOT, OUT, CONFIG, negative_volume_factors, NegativeVolumeController)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
