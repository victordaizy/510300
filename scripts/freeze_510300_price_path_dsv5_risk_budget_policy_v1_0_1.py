"""保留首版，冻结只涉及日期拼接的程序修正。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import price_path_dsv5_risk_budget_policy_v1 as base
from research.price_path_dsv5_risk_budget_policy_v1_0_1 import CORRECTION_MANIFEST, forward_calendar


def freeze() -> None:
    cfg, parent = base.verify(ROOT)
    for name in ["G0_G1.json", "historical_one_shot_claim.json"]:
        if (ROOT / base.REPORT / name).exists():
            raise base.ContractError("已经读取预测或消费收益，禁止此无收益技术修正")
    calendar = forward_calendar(ROOT, cfg)
    additions = [
        "research/price_path_dsv5_risk_budget_policy_v1_0_1.py",
        "scripts/run_510300_price_path_dsv5_risk_budget_policy_v1_0_1.py",
        "scripts/freeze_510300_price_path_dsv5_risk_budget_policy_v1_0_1.py",
        "tests/test_510300_price_path_dsv5_risk_budget_policy_v1_0_1.py",
        "docs/510300_PRICE_PATH_DSV5_RISK_BUDGET_POLICY_V1_0_1_TECHNICAL_CORRECTION_20260905.md",
        base.MANIFEST,
    ]
    correction = {**parent, "version": "1.0.1", "frozen_at": base.now(),
        "parent_program_version": "1.0.0", "parent_commit": base.git(ROOT, "rev-parse", "HEAD"),
        "correction_type": "PREDICTION_AND_RETURN_BLIND_CALENDAR_COVERAGE_JOIN",
        "policy_changed": False, "prediction_values_read": False, "portfolio_return_values_read": False,
        "calendar_coverage_end": str(calendar.max().date()),
        "implementation": parent["implementation"] + [base.identity(ROOT, p) for p in additions]}
    base.write_new(ROOT / CORRECTION_MANIFEST, correction)
    base.write_new(ROOT / base.REPORT / "v1_0_0_registration_failure.json", {
        "MODEL_ID": base.MODEL_ID, "STATE": "PROGRAM_FAILED_BEFORE_PREDICTION_READ",
        "version": "1.0.0", "recorded_at": base.now(),
        "cause": "历史日历截至2026-08-14，程序错误地要求其与全年官方日历完全相等",
        "forward_registration_created": False, "G0_G1_run": False,
        "prediction_values_read": False, "portfolio_return_values_read": False,
        "remediation_version": "1.0.1", "frozen_original_preserved": True})
    print("已冻结V1.0.1日期拼接修正，政策参数和首版文件保持原字节。")


if __name__ == "__main__":
    freeze()
