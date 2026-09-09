"""运行510300满仓/空仓二元状态可行性前沿。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.binary_state_feasibility_v1 import (  # noqa: E402
    CONFIG_PATH,
    build_report,
    evaluate_accuracy_frontier,
    evaluate_oracle_offsets,
    load_and_audit_inputs,
    load_config,
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
        ROOT / artifacts["offset_results_parquet"],
        ROOT / artifacts["accuracy_frontier_parquet"],
        ROOT / artifacts["canonical_block_labels_parquet"],
    ]
    existing = [str(path.relative_to(ROOT)) for path in protected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"冻结历史运行产物已存在，禁止覆盖：{existing}")

    manifest = validate_manifest(ROOT)
    market, dividends, data_audit = load_and_audit_inputs(ROOT, config)
    offset_results, horizon_summaries, canonical_blocks = evaluate_oracle_offsets(
        market, dividends, config
    )
    accuracy_raw, accuracy_aggregates, accuracy_frontier = evaluate_accuracy_frontier(
        market, dividends, config, canonical_blocks
    )
    report = build_report(
        config,
        manifest,
        data_audit,
        offset_results,
        horizon_summaries,
        accuracy_aggregates,
        accuracy_frontier,
    )

    output_directory.mkdir(parents=True, exist_ok=False)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    offset_results.to_parquet(ROOT / artifacts["offset_results_parquet"], index=False)
    accuracy_raw.to_parquet(ROOT / artifacts["accuracy_frontier_parquet"], index=False)
    canonical_blocks.to_parquet(ROOT / artifacts["canonical_block_labels_parquet"], index=False)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "状态": report["status"],
        "目标达成": report["goal_achieved"],
        "完美方向可行周期": report["perfect_foresight_feasible_horizons"],
        "报告": str(report_markdown.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

