"""在读取任何收益结果前冻结沪深 A 股低风险 V2 协议和数据阻断证据。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config/a_share_sh_sz_low_risk_regime_entry_v2.yaml"
FROZEN_FILES = (
    "config/a_share_sh_sz_low_risk_regime_entry_v2.yaml",
    "docs/A_SHARE_SH_SZ_LOW_RISK_REGIME_ENTRY_V2_SPEC.md",
    "scripts/audit_a_share_sh_sz_low_risk_regime_entry_v2_inputs.py",
    "scripts/freeze_a_share_sh_sz_low_risk_regime_entry_v2.py",
    "tests/test_a_share_sh_sz_low_risk_regime_entry_v2.py",
    "reports/data_quality/a_share_sh_sz_low_risk_regime_entry_v2_data_readiness.json",
    "reports/data_quality/A_SHARE_SH_SZ_LOW_RISK_REGIME_ENTRY_V2_DATA_READINESS.md",
)
EXPECTED_BUCKET2_MANIFEST_SHA256 = (
    "5194175a32ba8c64ac8ef2b8ebf8e9ebb0586c5d04e9abe28b9101399345ce6a"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_protocol(contract: dict) -> None:
    protocol = contract["protocol"]
    if protocol["model_id"] != "A_SHARE_SH_SZ_LOW_RISK_REGIME_ENTRY_V2":
        raise ValueError("模型 ID 必须精确为沪深 A 股低风险 V2")
    if protocol["initial_status"] != "DISCOVERY_ONLY":
        raise ValueError("初始状态必须保持 DISCOVERY_ONLY")
    if protocol["current_view_status"] != "NO_VIEW":
        raise ValueError("数据门槛通过前必须保持 NO_VIEW")

    universe = contract["universe"]
    if universe["type"] != "SH_SZ_A_POINT_IN_TIME":
        raise ValueError("母池必须明确命名为点时沪深 A 股")
    if universe["exchanges"] != ["SSE", "SZSE"]:
        raise ValueError("本版本只能包含上交所和深交所")
    if universe["explicitly_out_of_scope_exchanges"] != ["BSE"]:
        raise ValueError("北交所必须显式标记为范围外")

    market_cap = contract["market_cap"]
    entry = market_cap["entry"]
    holding = market_cap["holding"]
    if (
        entry["min_total_mcap_cny"],
        entry["min_float_mcap_cny"],
        entry["min_float_mcap_cross_section_percentile"],
    ) != (10_000_000_000, 5_000_000_000, 0.30):
        raise ValueError("主模型进入市值三重门槛发生漂移")
    if (
        holding["min_total_mcap_cny"],
        holding["min_float_mcap_cny"],
        holding["min_float_mcap_cross_section_percentile"],
        holding["exit_confirmation_reviews"],
    ) != (8_000_000_000, 4_000_000_000, 0.25, 2):
        raise ValueError("持有市值迟滞规则发生漂移")

    liquidity = contract["liquidity"]
    if (
        liquidity["min_median_amount_20_cny"],
        liquidity["min_valid_trading_days_60"],
        liquidity["max_suspension_days_20"],
        liquidity["max_zero_volume_days_20"],
    ) != (100_000_000, 58, 0, 0):
        raise ValueError("主模型流动性门槛发生漂移")

    risk_score = contract["risk_score"]
    weights = [item["weight"] for item in risk_score["metrics"].values()]
    if weights != [0.25, 0.25, 0.25, 0.25] or sum(weights) != 1.0:
        raise ValueError("复合低风险四项权重必须各为 25%")
    if (risk_score["entry_max"], risk_score["holding_max"]) != (0.20, 0.40):
        raise ValueError("低风险进入或持有缓冲发生漂移")
    if not risk_score["market_cap_in_score_forbidden"]:
        raise ValueError("市值不得进入低风险得分")

    regime = contract["volatility_regime"]
    if (
        regime["history_lookback_market_days"],
        regime["min_valid_history"],
        regime["entry_max_percentile"],
        regime["exit_min_percentile"],
    ) != (504, 252, 0.40, 0.70):
        raise ValueError("自身波动状态规则发生漂移")
    if not regime["history_excludes_current_observation"]:
        raise ValueError("当前观察不得进入自身历史分位")

    review = contract["review"]
    if (
        review["frequency_trading_days"],
        review["minimum_holding_days"],
        review["signal_time"],
        review["execution_time"],
    ) != (20, 20, "T_CLOSE", "NEXT_TRADABLE_OPEN"):
        raise ValueError("检查或 T+1 执行规则发生漂移")

    portfolio = contract["research_portfolio"]
    if (
        portfolio["target_names"],
        portfolio["target_weight_per_name"],
        portfolio["max_names_per_industry_l1"],
    ) != (20, 0.05, 3):
        raise ValueError("20 只研究组合规则发生漂移")

    wrapper = contract["small_account_wrapper"]
    if (
        wrapper["capital_cny"],
        wrapper["target_names"],
        wrapper["lot_size_shares"],
        wrapper["minimum_trade_notional_cny"],
        wrapper["maximum_one_lot_notional_cny"],
    ) != (20_000, 3, 100, 5_000, 7_000):
        raise ValueError("2 万元包装规则发生漂移")

    governance = contract["governance"]
    disabled_values = {
        "modify_510300_position": False,
        "paper_signal": "DISABLED",
        "position_mapping": "DISABLED",
        "order_generation": "DISABLED",
        "broker_connection": "DISABLED",
        "automatic_order": "DISABLED",
        "live_trading": "NOT_AUTHORIZED",
    }
    for key, expected in disabled_values.items():
        if governance[key] != expected:
            raise ValueError(f"治理边界发生漂移：{key}")

    forbidden = set(contract["pre_return_calibration"]["forbidden_information"])
    required_forbidden = {
        "RETURN",
        "EXCESS_RETURN",
        "SHARPE",
        "DRAWDOWN",
        "WIN_RATE",
        "RANK_IC",
        "SELECTED_NAME_CONTRIBUTION",
    }
    if forbidden != required_forbidden:
        raise ValueError("冻结前禁止查看的收益信息集合发生漂移")


def _assert_bucket2_untouched(contract: dict) -> None:
    boundary = contract["research_boundary"]
    manifest_path = ROOT / boundary["parent_bucket2_evidence_manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError("桶2永久冻结证据清单缺失")
    observed = sha256_file(manifest_path)
    if observed != EXPECTED_BUCKET2_MANIFEST_SHA256:
        raise RuntimeError(
            "桶2永久冻结证据发生变化："
            f"expected={EXPECTED_BUCKET2_MANIFEST_SHA256}，actual={observed}"
        )
    if boundary["parent_bucket2_evidence_manifest_sha256"] != observed:
        raise RuntimeError("新协议记录的桶2证据哈希与磁盘不一致")
    if boundary["reuse_bucket2_results_for_tuning"]:
        raise ValueError("禁止使用桶2结果调参")
    if boundary["reopen_bucket2_analysis"]:
        raise ValueError("禁止重开桶2分析")


def _run_data_audit() -> dict:
    command = [
        sys.executable,
        "scripts/audit_a_share_sh_sz_low_risk_regime_entry_v2_inputs.py",
        "--replace-unfrozen",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "数据可行性审计失败：\n" + completed.stdout + "\n" + completed.stderr
        )
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    report_path = ROOT / contract["paths"]["data_readiness_json"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] not in {
        "BLOCKED_DATA_CONTRACT_INCOMPLETE",
        "READY_FOR_NON_RETURN_COVERAGE_AUDIT",
    }:
        raise RuntimeError(f"未知数据审计状态：{report['status']}")
    forbidden_true = [
        key
        for key in (
            "return_values_read",
            "return_metrics_computed",
            "selection_generated",
            "position_mapping_generated",
            "orders_generated",
        )
        if report[key]
    ]
    if forbidden_true:
        raise RuntimeError(f"冻结前发生禁止动作：{forbidden_true}")
    return report


def _run_tests() -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_a_share_sh_sz_low_risk_regime_entry_v2.py",
            "tests/test_bucket2_permanent_freeze.py",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("协议测试失败：\n" + completed.stdout + "\n" + completed.stderr)
    return completed.stdout.strip()


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = ROOT / contract["paths"]["protocol_manifest"]
    if manifest_path.exists():
        raise FileExistsError("沪深 A 股低风险 V2 协议清单已存在，禁止覆盖")

    _assert_protocol(contract)
    _assert_bucket2_untouched(contract)

    forbidden_results = [
        ROOT / contract["paths"]["eligible_universe_audit"],
        ROOT / contract["paths"]["result_json"],
        ROOT / contract["paths"]["result_markdown"],
    ]
    existing_results = [
        path.relative_to(ROOT).as_posix() for path in forbidden_results if path.exists()
    ]
    if existing_results:
        raise RuntimeError(f"协议冻结前已经存在禁止的审计或收益结果：{existing_results}")

    report = _run_data_audit()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    test_result = _run_tests()

    manifest = {
        "schema_version": "A_SHARE_SH_SZ_LOW_RISK_PROTOCOL_MANIFEST_V1",
        "state": "PROTOCOL_FROZEN_DATA_BLOCKED_NO_RETURN_VIEW"
        if report["status"] == "BLOCKED_DATA_CONTRACT_INCOMPLETE"
        else "PROTOCOL_FROZEN_READY_FOR_NON_RETURN_COVERAGE_AUDIT",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model_id": contract["protocol"]["model_id"],
        "research_scope": contract["protocol"]["system_scope"],
        "included_exchanges": ["SSE", "SZSE"],
        "explicitly_out_of_scope_exchanges": ["BSE"],
        "data_readiness_status": report["status"],
        "view_status": "NO_VIEW",
        "return_values_read_before_freeze": False,
        "return_metrics_computed_before_freeze": False,
        "eligible_universe_audit_run_before_freeze": False,
        "bucket2_result_reopened": False,
        "bucket2_result_used_for_tuning": False,
        "bucket2_evidence_manifest_sha256": EXPECTED_BUCKET2_MANIFEST_SHA256,
        "position_mapping_generated": False,
        "orders_generated": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
        "frozen_files": {
            relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES
        },
        "test_result": test_result,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "模型": manifest["model_id"],
                "冻结状态": manifest["state"],
                "数据状态": manifest["data_readiness_status"],
                "研究视图": manifest["view_status"],
                "收益值读取": False,
                "桶2重开": False,
                "交易授权": False,
                "测试": test_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
