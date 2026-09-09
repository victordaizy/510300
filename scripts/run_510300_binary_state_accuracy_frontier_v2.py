"""运行510300满仓/空仓短周期识别精度前沿V2。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.binary_state_accuracy_frontier_v2 import (  # noqa: E402
    CONFIG_PATH,
    build_report,
    evaluate_frontiers,
    load_config,
    load_source_inputs,
    render_markdown,
    validate_manifest,
)


def main() -> int:
    config = load_config(CONFIG_PATH)
    artifacts = config["artifacts"]
    report_json = ROOT / artifacts["report_json"]
    report_markdown = ROOT / artifacts["report_markdown"]
    output_directory = ROOT / artifacts["output_directory"]
    protected_outputs = [
        report_json,
        report_markdown,
        ROOT / artifacts["random_trials_parquet"],
        ROOT / artifacts["random_aggregates_parquet"],
        ROOT / artifacts["severity_frontier_parquet"],
        ROOT / artifacts["block_labels_parquet"],
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"V2冻结历史运行产物已存在，禁止覆盖：{existing}")

    manifest = validate_manifest(ROOT)
    market, dividends, data_audit, v1_manifest = load_source_inputs(ROOT, config)
    random_trials, random_aggregates, severity, block_labels, summary = evaluate_frontiers(
        market,
        dividends,
        config,
    )
    report = build_report(
        config,
        manifest,
        v1_manifest,
        data_audit,
        random_trials,
        random_aggregates,
        severity,
        block_labels,
        summary,
    )

    output_directory.mkdir(parents=True, exist_ok=False)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    random_trials.to_parquet(ROOT / artifacts["random_trials_parquet"], index=False)
    random_aggregates.to_parquet(ROOT / artifacts["random_aggregates_parquet"], index=False)
    severity.to_parquet(ROOT / artifacts["severity_frontier_parquet"], index=False)
    block_labels.to_parquet(ROOT / artifacts["block_labels_parquet"], index=False)
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "状态": report["status"],
                "目标达成": report["goal_achieved"],
                "随机模拟行数": report["row_counts"]["random_trials"],
                "报告": str(report_markdown.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
