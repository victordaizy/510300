"""冻结沪深 A 股三只极端低风险确认策略 V1，冻结前不读取收益。"""

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
CONFIG_FILE = ROOT / "config/a_share_hs_concentrated_low_risk_trend_v1.yaml"
SHARED_AUDITOR = "scripts/audit_a_share_sh_sz_low_risk_regime_entry_v2_inputs.py"
SHARED_AUDITOR_SHA256 = (
    "4edc523db0b9c85e1d0fa9fb02abb9bcc1cbc848377c1243ff7fd6cb1e0b4c42"
)
EXPECTED_BUCKET2_MANIFEST_SHA256 = (
    "5194175a32ba8c64ac8ef2b8ebf8e9ebb0586c5d04e9abe28b9101399345ce6a"
)
FROZEN_FILES = (
    "config/a_share_hs_concentrated_low_risk_trend_v1.yaml",
    "docs/A_SHARE_HS_CONCENTRATED_LOW_RISK_TREND_V1_SPEC.md",
    SHARED_AUDITOR,
    "scripts/freeze_a_share_hs_concentrated_low_risk_trend_v1.py",
    "tests/test_a_share_hs_concentrated_low_risk_trend_v1.py",
    "reports/data_quality/a_share_hs_concentrated_low_risk_trend_v1_data_readiness.json",
    "reports/data_quality/A_SHARE_HS_CONCENTRATED_LOW_RISK_TREND_V1_DATA_READINESS.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_protocol(contract: dict) -> None:
    protocol = contract["protocol"]
    if protocol["model_id"] != "A_SHARE_HS_CONCENTRATED_LOW_RISK_TREND_V1":
        raise ValueError("集中模型 ID 漂移")
    if protocol["initial_status"] != "DISCOVERY_ONLY" or protocol["current_view_status"] != "NO_VIEW":
        raise ValueError("集中模型必须从 DISCOVERY_ONLY/NO_VIEW 开始")
    draft = protocol["draft_change"]
    if (
        draft["volatility_regime_entry_max_percentile_before_freeze"],
        draft["volatility_regime_entry_max_percentile_frozen"],
    ) != (0.30, 0.10):
        raise ValueError("本次唯一阈值变化必须是 30% 草案到 10% 冻结")

    universe = contract["universe"]
    if universe["exchanges"] != ["SSE", "SZSE"]:
        raise ValueError("集中模型只能包含沪深两市")
    if universe["included_boards"] != ["SSE_MAIN", "SSE_STAR", "SZSE_MAIN", "SZSE_CHINEXT"]:
        raise ValueError("集中模型板块范围发生漂移")
    if universe["minimum_listed_trading_days"] != 400:
        raise ValueError("上市历史门槛发生漂移")

    entry = contract["market_cap"]["entry"]
    holding = contract["market_cap"]["holding"]
    if entry != {
        "min_total_mcap_20d_median_cny": 10_000_000_000,
        "min_float_mcap_20d_median_cny": 5_000_000_000,
        "min_float_mcap_percentile": 0.30,
    }:
        raise ValueError("进入市值门槛发生漂移")
    if holding != {
        "min_total_mcap_20d_median_cny": 8_000_000_000,
        "min_float_mcap_20d_median_cny": 4_000_000_000,
        "min_float_mcap_percentile": 0.25,
        "confirmation_reviews": 2,
    }:
        raise ValueError("持有市值门槛发生漂移")
    liquidity = contract["liquidity"]
    if (
        liquidity["min_amount_20d_median_cny"],
        liquidity["max_suspension_days_20"],
        liquidity["max_zero_volume_days_20"],
        liquidity["min_valid_returns_120"],
    ) != (100_000_000, 0, 0, 115):
        raise ValueError("流动性门槛发生漂移")

    metrics = contract["risk_score"]["metrics"]
    if list(metrics) != ["rv60", "rv120", "downside_vol60", "max_drawdown120", "parkinson_range_vol60"]:
        raise ValueError("五维风险指标集合发生漂移")
    if [item["weight"] for item in metrics.values()] != [0.20] * 5:
        raise ValueError("五维风险指标必须等权")
    if (
        contract["risk_score"]["entry_current_max"],
        contract["risk_score"]["entry_previous_20d_max"],
        contract["risk_score"]["entry_previous_40d_max"],
        contract["risk_score"]["stable_score_max"],
        contract["risk_score"]["worst_component_rank_max"],
        contract["risk_score"]["holding_max"],
    ) != (0.12, 0.20, 0.25, 0.18, 0.35, 0.35):
        raise ValueError("集中模型风险分数门槛发生漂移")

    stability = contract["risk_score_stability"]
    if (
        stability["current_weight"],
        stability["previous_20d_weight"],
        stability["previous_40d_weight"],
    ) != (0.50, 0.30, 0.20):
        raise ValueError("稳定低风险分数权重发生漂移")
    regime = contract["volatility_regime"]
    if (
        regime["lookback_days"],
        regime["min_valid_history"],
        regime["entry_max_percentile"],
        regime["normal_exit_percentile"],
        regime["emergency_exit_percentile"],
        regime["emergency_rv20_div_rv120"],
    ) != (504, 252, 0.10, 0.70, 0.90, 1.60):
        raise ValueError("10%自身波动状态及紧急退出规则发生漂移")
    if not regime["history_excludes_current_observation"]:
        raise ValueError("当前日不得进入自身历史分布")

    selection = contract["selection"]
    if (
        selection["target_names"],
        selection["max_names_per_industry"],
        selection["max_pairwise_correlation_120d"],
        selection["pairwise_rule"],
    ) != (3, 1, 0.80, "STRICTLY_LESS_THAN"):
        raise ValueError("Top3 行业和相关性约束发生漂移")
    if selection["relax_thresholds_when_insufficient"] or selection["insufficient_candidates"] != "HOLD_CASH":
        raise ValueError("候选不足必须持有现金")

    portfolio = contract["portfolio"]
    if portfolio["three_names"] != {"target_weight_each": 0.30, "cash_weight": 0.10}:
        raise ValueError("三只股票仓位阶梯发生漂移")
    if portfolio["two_names"] != {"target_weight_each": 0.40, "cash_weight": 0.20}:
        raise ValueError("两只股票仓位阶梯发生漂移")
    if portfolio["one_name"] != {"maximum_weight": 0.40, "cash_weight": 0.60}:
        raise ValueError("单只股票仓位阶梯发生漂移")
    if portfolio["zero_names"]["cash_weight"] != 1.0:
        raise ValueError("零候选必须持有 100% 现金")

    review = contract["review"]
    if (
        review["full_selection_frequency_trading_days"],
        review["risk_review_frequency_trading_days"],
        review["minimum_holding_days_for_normal_replacement"],
    ) != (20, 5, 20):
        raise ValueError("20 日选股、5 日风险检查或最低持有期发生漂移")
    if review["risk_review_can_select_replacement"]:
        raise ValueError("5 日风险检查不得临时选替代股票")

    execution = contract["execution"]
    if (
        execution["commission_rate_per_leg"],
        execution["minimum_commission_cny_per_leg"],
        execution["sell_stamp_duty_rate"],
        execution["one_way_slippage_rate"],
    ) != (0.0003, 5, 0.0005, 0.0005):
        raise ValueError("执行成本发生漂移")
    governance = contract["governance"]
    for key, value in {
        "old_bucket2_modification": "FORBIDDEN",
        "modify_510300_position": False,
        "paper_signal": "DISABLED",
        "position_mapping": "DISABLED",
        "order_generation": "DISABLED",
        "broker_connection": "DISABLED",
        "automatic_order": "DISABLED",
        "live_trading": "NOT_AUTHORIZED",
    }.items():
        if governance[key] != value:
            raise ValueError(f"治理边界发生漂移：{key}")


def assert_bucket2_untouched(contract: dict) -> None:
    boundary = contract["research_boundary"]
    path = ROOT / boundary["parent_bucket2_evidence_manifest"]
    if not path.is_file() or sha256_file(path) != EXPECTED_BUCKET2_MANIFEST_SHA256:
        raise RuntimeError("桶2永久冻结证据发生变化或缺失")
    if boundary["parent_bucket2_evidence_manifest_sha256"] != EXPECTED_BUCKET2_MANIFEST_SHA256:
        raise RuntimeError("新协议记录的桶2哈希不一致")
    if boundary["reuse_bucket2_results_for_tuning"] or boundary["reopen_bucket2_analysis"]:
        raise ValueError("集中模型不得使用或重开桶2")
    if sha256_file(ROOT / SHARED_AUDITOR) != SHARED_AUDITOR_SHA256:
        raise RuntimeError("共享数据审计器已经漂移")


def run_data_audit() -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_a_share_sh_sz_low_risk_regime_entry_v2_inputs.py",
            "--config",
            "config/a_share_hs_concentrated_low_risk_trend_v1.yaml",
            "--replace-unfrozen",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("共享数据审计失败：\n" + completed.stdout + "\n" + completed.stderr)
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    report = json.loads((ROOT / contract["paths"]["data_readiness_json"]).read_text(encoding="utf-8"))
    if report["model_id"] != contract["protocol"]["model_id"] or report["view_status"] != "NO_VIEW":
        raise RuntimeError("集中模型数据审计报告身份或视图状态错误")
    for key in ("return_values_read", "return_metrics_computed", "selection_generated", "position_mapping_generated", "orders_generated"):
        if report[key]:
            raise RuntimeError(f"冻结前发生禁止动作：{key}")
    return report


def run_tests() -> str:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_a_share_hs_concentrated_low_risk_trend_v1.py",
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
        raise RuntimeError("集中模型测试失败：\n" + completed.stdout + "\n" + completed.stderr)
    return completed.stdout.strip()


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    manifest_path = ROOT / contract["paths"]["protocol_manifest"]
    if manifest_path.exists():
        raise FileExistsError("集中模型 V1 协议清单已存在，禁止覆盖")
    assert_protocol(contract)
    assert_bucket2_untouched(contract)
    forbidden = [ROOT / contract["paths"]["result_json"], ROOT / contract["paths"]["result_markdown"]]
    existing = [path.relative_to(ROOT).as_posix() for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(f"冻结前已经存在收益结果：{existing}")
    report = run_data_audit()
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    test_result = run_tests()
    manifest = {
        "schema_version": "A_SHARE_HS_CONCENTRATED_LOW_RISK_PROTOCOL_MANIFEST_V1",
        "state": "PROTOCOL_FROZEN_DATA_BLOCKED_NO_RETURN_VIEW"
        if report["status"] == "BLOCKED_DATA_CONTRACT_INCOMPLETE"
        else "PROTOCOL_FROZEN_READY_FOR_NON_RETURN_COVERAGE_AUDIT",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model_id": contract["protocol"]["model_id"],
        "research_scope": contract["protocol"]["system_scope"],
        "decision_model": "TOP3",
        "frozen_entry_ts_vol_pct": 0.10,
        "draft_entry_ts_vol_pct": 0.30,
        "data_readiness_status": report["status"],
        "view_status": "NO_VIEW",
        "return_values_read_before_freeze": False,
        "return_metrics_computed_before_freeze": False,
        "selection_generated": False,
        "top1_shadow_started": False,
        "bucket2_result_reopened": False,
        "bucket2_result_used_for_tuning": False,
        "bucket2_evidence_manifest_sha256": EXPECTED_BUCKET2_MANIFEST_SHA256,
        "position_mapping_generated": False,
        "orders_generated": False,
        "broker_connection_enabled": False,
        "live_trading_authorized": False,
        "shared_data_auditor_sha256": SHARED_AUDITOR_SHA256,
        "frozen_files": {relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES},
        "test_result": test_result,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"模型": manifest["model_id"], "冻结状态": manifest["state"], "TSVolPct入场": 0.10, "数据状态": report["status"], "收益值读取": False, "测试": test_result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
