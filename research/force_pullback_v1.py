"""第125轮单一量价力度回调进入和反弹转弱退出。"""
from pathlib import Path
from research.fixed_signal_account_round_v1 import freeze_signal, run_signal
from research.force_pullback_inputs_v1 import force_factors, ForcePullbackController

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_force_pullback_v1"
CONFIG = ROOT / "config/510300_force_pullback_v1.json"
PRIMARY = "FORCE_PULLBACK"


def freeze():
    additions = {"study_id": "510300_FORCE_PULLBACK_V1", "round": 125, "primary": PRIMARY, "name": "长短量价力度回调交易",
        "fast_period": 2, "slow_period": 13, "ema_seed": "FIRST_N_COMPLETE_RAW_VALUES_SIMPLE_MEAN",
        "price_change": "DECIMAL_CLOSE_MINUS_PREVIOUS_PLUS_DIVIDEND", "volume": "ACTUAL_TRADED_SHARES",
        "source_gap": "NO_VIEW_REMAINDER_NO_RESEED", "old_cooldown_or_rearm": False,
        "rules": "docs/510300_FORCE_PULLBACK_V1.md", "input_receipt": "reports/research/510300_force_pullback_preflight_20260909/result.json",
        "previous_goal_turn_classification": "PROGRESS_ROUND124_COMPLETED_AND_125_METHOD_AND_SOURCE_REVIEW"}
    freeze_signal(ROOT, OUT, CONFIG, additions, [Path(__file__), ROOT / "research/force_pullback_inputs_v1.py", ROOT / "tests/test_force_pullback_v1.py"], 8)


def run():
    run_signal(ROOT, OUT, CONFIG, force_factors, ForcePullbackController)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
