"""运行跨市场图形机器学习V1的海外闸门或唯一目标揭盲。"""

from __future__ import annotations

import argparse

from research.cross_market_chart_ml_v1 import (
    load_config,
    reveal_target_once,
    run_external_stage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行510300跨市场图形机器学习V1")
    parser.add_argument("--stage", choices=["external", "target"], required=True)
    args = parser.parse_args()
    config = load_config()
    if args.stage == "external":
        report = run_external_stage(config)
    else:
        report = reveal_target_once(config)
    print(f"阶段：{args.stage}")
    print(f"状态：{report['status']}")
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
