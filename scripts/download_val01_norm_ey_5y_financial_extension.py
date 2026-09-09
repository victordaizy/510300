"""运行 VAL01_NORM_EY_5Y 财务历史扩展采集。"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val01_norm_ey_5y_financial_extension import (
    blocked_report,
    load_config,
    resolve_transport,
    run_acquisition,
    write_report,
)


def main() -> int:
    config = load_config()
    transport = None
    try:
        transport = resolve_transport(config)
        report = run_acquisition(config)
        write_report(report, config)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "target_universe": report["target_universe"],
                    "acquisition_range": report["acquisition_range"],
                    "extension_archive": report["extension_archive"],
                    "extended_archive": report["extended_archive"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        report = blocked_report(error, config, transport)
        write_report(report, config)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "failure_category": report["failure_category"],
                    "error": report["error"],
                    "frozen_inputs": report["frozen_inputs"]["status"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
