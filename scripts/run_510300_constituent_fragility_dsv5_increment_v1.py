"""运行 510300 成分脆弱性 DSV5 增量检验 V1。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.constituent_fragility_dsv5_increment_v1 import (  # noqa: E402
    DSV5ProtocolError,
    load_protocol,
    record_program_failure,
    run_full_execution,
    run_prelable_audit,
    verify_frozen_execution_scope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="先执行无标签G0/G1，提交后再消费一次性DSV5模型试验。"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="项目根目录，默认由脚本路径推导。",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("config/510300_constituent_fragility_dsv5_increment_v1.yaml"),
        help="相对项目根目录或绝对协议路径。",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="只读历史特征/价格的可用性，不构造DSV5，不拟合模型。",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="要求无标签审计已提交，随后创建claim并运行一次性预测实验。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    protocol_path = args.protocol
    if not protocol_path.is_absolute():
        protocol_path = project_root / protocol_path
    try:
        if args.preflight_only:
            protocol = load_protocol(protocol_path)
            verify_frozen_execution_scope(
                project_root=project_root,
                protocol=protocol,
                require_prelable_outputs_committed=False,
            )
            audit, *_ = run_prelable_audit(
                project_root=project_root,
                protocol_path=protocol_path,
                write_outputs=True,
            )
            print(
                json.dumps(
                    {
                        "状态": audit["status"],
                        "G0": audit["G0_DATA_CONTRACT"]["passed"],
                        "G1": audit["G1_NONOVERLAPPING_ORIGINS"],
                        "DSV5数值已读取": False,
                        "模型已训练": False,
                        "仓位影响": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if audit["status"] == "PASS_G0_G1_PRELABEL_ORIGIN_AUDIT" else 2

        result = run_full_execution(project_root=project_root, protocol_path=protocol_path)
        print(
            json.dumps(
                {
                    "最终状态": result["final_state"],
                    "G2通过": result["gates"]["G2_B1_VS_B0"]["passed"],
                    "G3状态": result["gates"]["G3_B2_VS_B1"]["status"],
                    "组合评价": result["portfolio_evaluation"],
                    "Sharpe": result["sharpe"],
                    "仓位影响": result["position_impact"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        failure_path = record_program_failure(
            project_root=project_root,
            protocol_path=protocol_path,
            error=error,
        )
        print(f"执行失败：{type(error).__name__}: {error}", file=sys.stderr)
        if failure_path is not None:
            print(f"一次性claim已消费，程序失败收据：{failure_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
