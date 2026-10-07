"""仅统一比较时钟的存储精度，逐值往返确认，不改变来源或金融定义。"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research import macro_2026_source_coverage_intake_v1 as source

PARENT = source.OUT
OUT = PARENT.with_name(PARENT.name + "_clock_adapter")
COMPARE = pd.testing.assert_frame_equal


def canonical_times(frame):
    result = frame.copy()
    for column in frame.columns:
        original = frame[column]
        if not pd.api.types.is_datetime64_any_dtype(original.dtype):
            continue
        zone = getattr(original.dtype, "tz", None)
        target = "datetime64[ns]" if zone is None else f"datetime64[ns, {zone}]"
        converted = original.astype(target)
        pd.testing.assert_series_equal(converted.astype(original.dtype), original, check_exact=True)
        result[column] = converted
    return result


def exact_clock_compare(left, right, **kwargs):
    kwargs.update(check_exact=True, check_dtype=True)
    return COMPARE(canonical_times(left), canonical_times(right), **kwargs)


def freeze():
    source.study.require(not OUT.exists(), "时钟比较适配已登记，不覆盖。")
    failure = source.study.read(PARENT / "implementation_failure.json")
    source.study.require(failure["phase"] == "ORIGINAL_3307_FUNDING_PREFIX_EXACT_COMPARISON" and
                         failure["new_accounts"] == 0 and failure["results_generated"] == 0, "非登记的无金融比较失败。")
    protocol = source.study.read(PARENT / "protocol.json")
    (OUT / "results").mkdir(parents=True)
    protocol.update(at=source.study.now(), original_failed_output=source.rel(PARENT),
                    clock_adapter="COMPARISON_ONLY_NS_WITH_EXACT_ROUNDTRIP_FOR_EVERY_DATETIME_COLUMN",
                    source_values_or_model_parameters_changed=False, original_failure_preserved=True)
    protocol["sources"] += [{"path": source.rel(p), "sha256": source.h(p)} for p in
                            [Path(__file__), PARENT / "protocol.json", PARENT / "implementation_failure.json"]]
    source.save_json(OUT / "protocol.json", protocol)
    print("来源用途不变，登记仅时钟精度比较适配；原失败保持。", flush=True)


def run():
    source.OUT = OUT
    pd.testing.assert_frame_equal = exact_clock_compare
    try:
        source.run()
    finally:
        pd.testing.assert_frame_equal = COMPARE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026源核对的精度比较适配")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
