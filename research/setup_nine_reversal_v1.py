"""第130轮固定四日比较九次准备事件，直接检验四个完整账户。"""
from pathlib import Path
from research.fixed_signal_account_round_v1 import freeze_signal, run_signal
from research.setup_nine_reversal_inputs_v1 import setup_nine_factors, SetupNineController

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_setup_nine_reversal_v1"
CONFIG = ROOT / "config/510300_setup_nine_reversal_v1.json"
PRIMARY = "SETUP_NINE_REVERSAL"


def freeze():
    additions = {"study_id": "510300_SETUP_NINE_REVERSAL_V1", "round": 130, "primary": PRIMARY,
        "name": "九次相对弱势反弹与强势退出", "comparison_days": 4, "event_count": 9,
        "price_comparison": "EXACT_FRACTION_PRODUCT_OF_FOUR_DIVIDEND_INCLUSIVE_SIMPLE_GROSSES_VS_ONE",
        "events": "FIRST_EXACT_NINTH_OBSERVATION_ONLY_NO_SATURATION", "equal_price": "RESET_BOTH_COUNTS",
        "failed_buy": "NO_RETRY_WITHOUT_NEW_NINTH_EVENT", "locked_exit": "PERSIST_UNTIL_ACTUAL_SALE",
        "source_gap": "NO_VIEW_REMAINDER_NO_RESTART", "complete_sequential_replication": False,
        "old_cooldown_or_rearm": False, "rules": "docs/510300_SETUP_NINE_REVERSAL_V1.md",
        "input_receipt": "reports/research/510300_setup_nine_reversal_preflight_20260909/result.json",
        "previous_goal_turn_classification": "PROGRESS_ROUND129_COMPLETED_AND_SHARED_SAVED_VERIFIER_DELIVERED"}
    freeze_signal(ROOT, OUT, CONFIG, additions, [Path(__file__), ROOT / "research/setup_nine_reversal_inputs_v1.py",
        ROOT / "tests/test_setup_nine_reversal_v1.py"], 7)


def run():
    run_signal(ROOT, OUT, CONFIG, setup_nine_factors, SetupNineController)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
