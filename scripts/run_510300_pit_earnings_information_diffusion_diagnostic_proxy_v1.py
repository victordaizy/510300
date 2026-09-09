"""运行510300 PIT盈余信息扩散的结果盲补齐或隔离代理诊断。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.pit_earnings_information_diffusion_diagnostic_proxy_v1 import (  # noqa: E402
    CONFIG_PATH,
    build_outcome_blind_data,
    load_config,
    run_unreliable_prediction_diagnostic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["data", "diagnostic"],
        help="data只构建结果盲面板；diagnostic读取冻结的未来标签并运行隔离诊断",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.stage == "data":
        result = build_outcome_blind_data(config)
        summary = {
            "stage": "OUTCOME_BLIND_DATA",
            "status": result["status"],
            "authoritative_candidate_status": result[
                "authoritative_candidate_status"
            ],
            "member_valid_weight_fact_count": result["prevalence"][
                "member_valid_weight_fact_count"
            ],
            "member_valid_weight_fact_count_2021_plus": result["prevalence"][
                "member_valid_weight_fact_count_2021_plus"
            ],
            "market_price_reads": result["market_price_reads"],
            "future_return_reads": result["future_return_reads"],
        }
    else:
        result = run_unreliable_prediction_diagnostic(config)
        summary = {
            "stage": "UNRELIABLE_INPUT_DIAGNOSTIC",
            "status": result["status"],
            "observed_prediction_status": result["evaluation"]["status"],
            "authoritative_candidate_status": result[
                "authoritative_candidate_status"
            ],
            "portfolio_evaluation_performed": result[
                "portfolio_evaluation_performed"
            ],
            "model_action": result["model_action"],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

