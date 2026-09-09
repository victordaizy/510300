"""生成DAILY_01分组支持硬失败报告并关闭一次性运行。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.daily_01_overnight_absorption_v1 import (
    audit_and_load_inputs,
    build_absorption_features,
    load_config,
    sha256_file,
)
from research.daily_01_overnight_absorption_v1_evaluation import (
    build_forward_labels,
    load_evaluation_config,
    verify_parent_protocol,
)
from research.daily_01_overnight_absorption_v1_failure_reporting import (
    build_group_support_failure_report,
)


EVALUATION_MANIFEST = ROOT / "config" / "daily_01_overnight_absorption_v1_evaluation_manifest.json"
REPORTING_MANIFEST = ROOT / "config" / "daily_01_overnight_absorption_v1_failure_reporting_manifest.json"


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    reporting_manifest = json.loads(REPORTING_MANIFEST.read_text(encoding="utf-8"))
    if reporting_manifest.get("failure_report_completed") is not False:
        raise RuntimeError("分组支持失败报告已完成，禁止重复生成")
    evaluation_manifest = json.loads(EVALUATION_MANIFEST.read_text(encoding="utf-8"))
    if evaluation_manifest.get("historical_run_completed") is not False:
        raise RuntimeError("一次性预测运行已经关闭")
    if sha256_file(EVALUATION_MANIFEST) != reporting_manifest["evaluation_manifest_sha256_before_reporting"]:
        raise ValueError("预测实现清单在失败报告冻结后发生变化")

    parent_config = load_config()
    evaluation_config = load_evaluation_config()
    verify_parent_protocol(ROOT, evaluation_config)
    market, dividends, audit = audit_and_load_inputs(ROOT, parent_config)
    if audit["status"] != "PASS":
        raise RuntimeError("数据闸门失败")
    features = build_absorption_features(market, dividends, parent_config)
    labeled = build_forward_labels(features, dividends, parent_config, evaluation_config)
    sample = evaluation_config["sample"]
    report = build_group_support_failure_report(
        labeled,
        favorable_threshold=float(sample["favorable_group_percentile_at_least"]),
        unfavorable_threshold=float(sample["unfavorable_group_percentile_at_most"]),
        minimum_group_observations=int(sample["minimum_group_observations"]),
        minimum_mature_enter_events=int(sample["minimum_mature_enter_events"]),
    )
    report["project_id"] = parent_config["protocol"]["project_id"]
    report["data_cutoff"] = parent_config["protocol"]["historical_evaluation_cutoff"]
    report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    report["observed_exception"] = "ValueError: 区块自助缺少有利组或不利组"
    report["known_prior_trial_lower_bound_before_daily_01"] = int(
        evaluation_config["statistics"]["known_prior_trial_lower_bound_before_daily_01"]
    )
    artifacts = evaluation_config["artifacts"]
    report_path = ROOT / artifacts["report_json"]
    _atomic_json(report, report_path)

    markdown = f"""# DAILY_01隔夜冲击—日内吸收预测检验

- 状态：`{report['status']}`
- 失败类别：`{report['failure_category']}`
- 数据截止：`{report['data_cutoff']}`
- 策略回测授权：`false`

## 支持度

- 有效候选日：{report['sample']['valid_candidate_rows']}
- 成熟候选日：{report['sample']['mature_candidate_rows']}
- 80%有利组：{report['sample']['favorable_group_observations']}
- 20%不利组：{report['sample']['unfavorable_group_observations']}
- 成熟ENTER事件：{report['sample']['mature_enter_events']}

预注册分组至少一侧少于20个观测，因此最低支持度门失败。未计算组间收益、区块Bootstrap、HAC增量回归或策略资产曲线；禁止修改分组、窗口、方向或期限补救。
"""
    markdown_path = ROOT / artifacts["report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text(markdown, encoding="utf-8")
    temporary.replace(markdown_path)

    evaluation_manifest["historical_run_completed"] = True
    evaluation_manifest["historical_run_completed_at"] = report["generated_at"]
    evaluation_manifest["historical_result_status"] = report["status"]
    evaluation_manifest["historical_run_termination"] = "GROUP_SUPPORT_FAILURE_BEFORE_STATISTICS"
    evaluation_manifest["report_file"] = artifacts["report_json"]
    evaluation_manifest["report_sha256"] = sha256_file(report_path)
    _atomic_json(evaluation_manifest, EVALUATION_MANIFEST)

    reporting_manifest["failure_report_completed"] = True
    reporting_manifest["failure_report_completed_at"] = report["generated_at"]
    reporting_manifest["result_status"] = report["status"]
    reporting_manifest["report_file"] = artifacts["report_json"]
    reporting_manifest["report_sha256"] = sha256_file(report_path)
    _atomic_json(reporting_manifest, REPORTING_MANIFEST)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
