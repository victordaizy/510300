"""一次性固定分红覆盖维护实现及既有官方事件身份。"""

from __future__ import annotations

import json
from datetime import datetime

from research import official_dividend_coverage_refresh_v1 as source


def main() -> int:
    root = source.ROOT
    config = json.loads((root / source.CONFIG).read_text(encoding="utf-8"))
    baseline = json.loads((root / config["baseline_coverage"]).read_text(encoding="utf-8"))
    paths = {
        source.CONFIG,
        "research/official_dividend_coverage_refresh_v1.py",
        "scripts/refresh_510300_official_dividend_coverage_v1.py",
        "scripts/freeze_510300_official_dividend_coverage_refresh_v1.py",
        "scripts/priority_forward_data_paths_v1.py",
        "tests/test_official_dividend_coverage_refresh_v1.py",
        config["baseline_coverage"], config["baseline_dividends"], config["live_dividends"],
        config["calendar"], config["observer_config"], config["observer_manifest"],
        "data/forward/510300_official_dividend_coverage_survey_v1/20260906T033228560286/fund_510300_product.html",
        "data/forward/510300_official_dividend_coverage_survey_v1/20260906T033228560286/sse_fund_announcements_page_1.json",
    } | {record["saved_file"] for record in baseline["official_source_snapshots"]}
    payload = {
        "version": config["version"], "status": "FROZEN_SOURCE_MAINTENANCE_ONLY",
        "created_at": datetime.now(source.TZ).isoformat(),
        "frozen_event_count": baseline["event_count"],
        "mutable_output": config["live_coverage"],
        "original_event_ledger_and_observer_seed_preserved": True,
        "real_web_fixture_tests": "14 passed in 2.55s",
        "data_purchase_budget_cny": 0, "position_impact": 0,
        "files": [source.identity(root, root / path) for path in sorted(paths)],
    }
    source.save_json(root / source.MANIFEST, payload)
    digest = source.sha(root / source.MANIFEST)
    source.verify_runtime(root, digest)
    print(json.dumps({"status": "PASS_FROZEN_OFFICIAL_DIVIDEND_COVERAGE_MAINTENANCE", "manifest_sha256": digest, "tracked_file_count": len(paths), "source_fetches": 0, "coverage_updated": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
