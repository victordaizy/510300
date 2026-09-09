from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.post_close_stale_price_capture_v2 import ProtocolError, render_g0_status


DEFAULT_CONFIG = ROOT / "config" / "510300_post_close_stale_price_capture_v2.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成510300盘后过时收盘价捕获V2的G0状态")
    parser.add_argument("--expected-manifest-sha256", required=True, help="冻结清单摘要")
    parser.add_argument("--g0-evidence", type=Path, required=True, help="G0证据回执")
    parser.add_argument("--root", type=Path, default=ROOT, help="项目根目录")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="冻结配置路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        status = render_g0_status(
            args.root.resolve(),
            args.config.resolve(),
            args.expected_manifest_sha256,
            args.g0_evidence.resolve(),
        )
    except (ProtocolError, OSError) as exc:
        print(json.dumps({"状态": "G0状态生成失败", "原因": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"状态": "G0状态已生成", "结果": status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
