"""构造510300 UP20预测V1的冻结前因果特征。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from up20_rare_event_forecast_v1 import (  # noqa: E402
    ContractError,
    build_outcome_free_features,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造510300 UP20预测V1冻结前特征")
    parser.add_argument("--no-write", action="store_true", help="只复算，不写入特征产物")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        features, audit = build_outcome_free_features(write=not arguments.no_write)
    except (ContractError, FileNotFoundError, FileExistsError, ValueError) as exc:
        payload = {
            "status": "NO_VIEW_OUTCOME_FREE_FEATURE_DATA_OR_CONTRACT_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_outcome": "NOT_EVALUATED",
            "portfolio_evaluation": "NOT_ALLOWED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - 保留程序失败状态
        payload = {
            "status": "PROGRAM_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_outcome": "NOT_EVALUATED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 3
    summary = {
        "project_id": audit["project_id"],
        "status": audit["status"],
        "feature_rows": int(len(features)),
        "first_date": str(features["date"].iloc[0].date()),
        "last_date": str(features["date"].iloc[-1].date()),
        "historical_weights_used": False,
        "future_return_created": False,
        "model_outcome": "NOT_EVALUATED",
        "result_written": not arguments.no_write,
        "live_trading_authorized": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

