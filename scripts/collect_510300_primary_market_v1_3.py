"""PCF/IOPV V1.3：严格 TLS 路线证据，不改变既有研究合同。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from market_data.etf_primary_market_strict_v1_3 import (
    SseEtfDailyCrosscheckProviderV1_3,
    SseIopvSnapshotProviderV1_3,
    SsePcfProviderV1_3,
)
from scripts import collect_510300_primary_market as base
from scripts import collect_510300_primary_market_v1_2 as v1_2
from scripts.free_source_storage_v1_2 import (
    persist_record_atomic,
    write_content_addressed_raw,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "primary_market_forward_v1_3.yaml"
EXPECTED_ENDPOINTS = {
    "query.sse.com.cn:443",
    "yunhq.sse.com.cn:32042",
}


def validate_transport_contract(config: dict[str, Any]) -> None:
    contract = config.get("transport_contract")
    if not isinstance(contract, dict):
        raise ValueError("PCF/IOPV V1.3 缺少严格传输合同")
    fallback = contract.get("direct_fallback")
    if not isinstance(fallback, dict):
        raise ValueError("PCF/IOPV V1.3 缺少严格直连回退合同")
    expected = {
        "schema_version": "SSE_STRICT_TRANSPORT_CONTRACT_V1",
        "tls_verification_required": True,
        "insecure_tls_allowed": False,
        "environment_route_first": True,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"严格传输合同字段不一致：{key}")
    fallback_expected = {
        "enabled": True,
        "trigger_only_on": "CERTIFICATE_VERIFICATION_FAILED",
        "maximum_attempts_per_request": 1,
        "trust_environment_proxy": False,
    }
    for key, value in fallback_expected.items():
        if fallback.get(key) != value:
            raise ValueError(f"严格直连回退字段不一致：{key}")
    endpoints = contract.get("allowed_https_endpoints")
    if not isinstance(endpoints, list) or set(map(str, endpoints)) != EXPECTED_ENDPOINTS:
        raise ValueError("严格传输合同端点白名单不一致")


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("PCF/IOPV V1.3 配置顶层必须是对象")
    validate_transport_contract(config)
    arguments = base._parser().parse_args()

    base.CONFIG_FILE = CONFIG_FILE
    base.SsePcfProvider = SsePcfProviderV1_3
    base.SseIopvSnapshotProvider = SseIopvSnapshotProviderV1_3
    base._write_raw = write_content_addressed_raw
    base._persist_record = persist_record_atomic
    exit_code = base.main()
    if exit_code != 0:
        return exit_code

    now = datetime.now(ZoneInfo(str(config["collector"]["timezone"])))
    if not v1_2._production_close_crosscheck_required(
        watch=arguments.watch,
        max_samples=arguments.max_samples,
        now=now,
        config=config,
    ):
        return 0
    try:
        payload = v1_2.collect_daily_crosscheck(
            config,
            now.date(),
            provider=SseEtfDailyCrosscheckProviderV1_3(),
        )
    except base.ExternalSourceAcquisitionError as exc:
        print(f"PCF/IOPV第二官方端点失败：{exc}", file=sys.stderr)
        return 3
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
