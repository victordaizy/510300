"""按冻结 V1 规则构建 510300 压力传导危险率 V2 的点时 M/F/T 特征。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    normalize_project_relative_path,
)
from research.stress_transmission_hazard_v2 import NO_VIEW, VIEW_ALLOWED
from research.stress_transmission_hazard_v2_mft_features_v1 import (
    MFTFeatureArtifacts,
    build_mft_feature_artifacts,
)
from scripts.build_510300_stress_transmission_hazard_v2_four_state_ledger_v1 import (
    _write_parquet_new,
    _write_text_new,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_mft_feature_execution_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


TABLE_OUTPUTS = {
    "earnings_yield_release_ledger": "earnings_yield_releases",
    "china_10y_release_ledger": "china_10y_releases",
    "dr007_release_ledger": "dr007_releases",
    "reverse_repo_7d_release_ledger": "reverse_repo_7d_releases",
    "internal_features": "internal_features",
    "macro_features": "macro_features",
    "mft_feature_panel": "mft_feature_panel",
    "member_history_ledger": "member_history_ledger",
    "macro_availability_ledger": "macro_availability_ledger",
    "coverage_ledger": "coverage_ledger",
}


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def frame_semantic_sha256(frame: pd.DataFrame) -> str:
    """计算包含列顺序、dtype、行顺序与值的稳定语义摘要。"""

    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in frame.columns).encode("utf-8"))
    digest.update("\x1f".join(str(dtype) for dtype in frame.dtypes).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").tobytes())
    digest.update(str(len(frame)).encode("ascii"))
    return digest.hexdigest()


def load_and_build(config: Mapping[str, Any]) -> MFTFeatureArtifacts:
    """只读取冻结的特征输入并在内存中构建；不触碰标签或绩效文件。"""

    inputs = config["inputs"]
    membership = pd.read_parquet(_project_path(inputs["point_in_time_membership"]["path"]))
    classified = pd.read_parquet(
        _project_path(inputs["classified_constituent_returns"]["path"])
    )
    daily_coverage = pd.read_parquet(
        _project_path(inputs["four_state_daily_coverage"]["path"])
    )
    h00300 = pd.read_parquet(_project_path(inputs["h00300_total_return_close"]["path"]))
    pe = pd.read_parquet(_project_path(inputs["csi300_official_pe"]["path"]))
    bond = pd.read_parquet(_project_path(inputs["china_10y_yield"]["path"]))
    dr007 = pd.read_parquet(_project_path(inputs["dr007"]["path"]))
    policy = pd.read_parquet(_project_path(inputs["reverse_repo_7d_policy_rate"]["path"]))
    source_sha256 = {
        "pe": inputs["csi300_official_pe"]["sha256"],
        "bond": inputs["china_10y_yield"]["sha256"],
        "dr007": inputs["dr007"]["sha256"],
        "policy": inputs["reverse_repo_7d_policy_rate"]["sha256"],
    }
    return build_mft_feature_artifacts(
        membership=membership,
        classified_returns=classified,
        four_state_daily_coverage=daily_coverage,
        h00300_total_return_close=h00300,
        pe=pe,
        bond=bond,
        dr007=dr007,
        reverse_repo_7d=policy,
        source_sha256=source_sha256,
    )


def _output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: _project_path(str(relative))
        for name, relative in config["outputs"].items()
        if name != "curated_root"
    }


def _iso_date_or_none(values: pd.Series, *, first: bool) -> str | None:
    valid = pd.to_datetime(values, errors="coerce").dropna()
    if valid.empty:
        return None
    chosen = valid.min() if first else valid.max()
    return pd.Timestamp(chosen).date().isoformat()


def _build_report(artifacts: MFTFeatureArtifacts) -> str:
    panel = artifacts.mft_feature_panel
    model_view = panel["MODEL_STATE"].eq(VIEW_ALLOWED)
    b2_view = panel["B2_feature_state"].eq(VIEW_ALLOWED)
    first_view = _iso_date_or_none(panel.loc[model_view, "date"], first=True)
    last_view = _iso_date_or_none(panel.loc[model_view, "date"], first=False)
    reason_counts = panel.loc[~model_view, "NO_VIEW_REASON"].value_counts().to_dict()
    reason_lines = [f"  - `{reason}`：{int(count):,}" for reason, count in reason_counts.items()]
    return "\n".join(
        [
            "# 510300 压力传导危险率 V2：M/F/T 点时特征 V1",
            "",
            "## 数据与视图裁决",
            "",
            f"- 点时市场日：{len(panel):,}",
            f"- `F_state=VIEW_ALLOWED`：{int(panel['F_state'].eq(VIEW_ALLOWED).sum()):,}",
            f"- `M_state=VIEW_ALLOWED`：{int(panel['M_state'].eq(VIEW_ALLOWED).sum()):,}",
            f"- `T_state=VIEW_ALLOWED`：{int(panel['T_state'].eq(VIEW_ALLOWED).sum()):,}",
            f"- B2 特征可视日：{int(b2_view.sum()):,}",
            f"- 完整 M/F/T 可视日：{int(model_view.sum()):,}",
            f"- 完整 M/F/T `NO_VIEW` 日：{int((~model_view).sum()):,}",
            f"- 首个完整可视日：{first_view}",
            f"- 最后完整可视日：{last_view}",
            "",
            "## `NO_VIEW` 原因组合",
            "",
            *reason_lines,
            "",
            "## 时钟与经济方向",
            "",
            "- PE、10年国债和 DR007 均从严格下一点时交易日开盘起可用；央行逆回购按归档公告实际发布时间生效。",
            "- 被选择的宏观记录均满足 `available_at <= t日15:00` 且 `observation_date <= t`。",
            "- 九个风险分位通道均已按冻结经济方向独立重算，并完成实际样本前缀不变性检查。",
            "- 价格窗口允许使用当时公开的入指前历史；共同运动只累计真实点时成员日，非成员日不计。",
            "- 行业、申万或板块字段没有进入核心面板。",
            "",
            "本执行未读取实际 BAD10 标签、未来收益、AUC、Log Loss、Brier、警报、夏普、回撤、净值、仓位或订单。G0 仍等待全新进程重放和定向测试。",
            "",
        ]
    )


def build(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    outputs = _output_paths(config)
    build_owned = [
        *TABLE_OUTPUTS.keys(),
        "report",
        "receipt",
        "status",
    ]
    existing = [str(outputs[name]) for name in build_owned if outputs[name].exists()]
    if existing:
        raise EvidenceContractError(f"M/F/T 构建输出已存在，禁止覆盖：{existing}")

    print("开始按冻结时钟构建 M/F/T、来源可用性与 NO_VIEW 账本。", flush=True)
    artifacts = load_and_build(config)
    print("内存构建和无未来断言通过，正在原子写入新输出。", flush=True)
    semantic_hashes: dict[str, str] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        frame = getattr(artifacts, attribute_name)
        semantic_hashes[output_name] = frame_semantic_sha256(frame)
        _write_parquet_new(frame, outputs[output_name])

    report = _build_report(artifacts)
    _write_text_new(report, outputs["report"])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    table_evidence = {
        name: {
            **file_evidence(outputs[name], project_root=ROOT),
            "row_count": int(len(getattr(artifacts, attribute))),
            "semantic_sha256": semantic_hashes[name],
        }
        for name, attribute in TABLE_OUTPUTS.items()
    }
    panel = artifacts.mft_feature_panel
    view = panel["MODEL_STATE"].eq(VIEW_ALLOWED)
    no_view_reason_counts = {
        str(key): int(value)
        for key, value in panel.loc[~view, "NO_VIEW_REASON"].value_counts().items()
    }
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_FEATURE_BUILD_V1",
        "created_at": now,
        "status": "PASS_POINT_IN_TIME_M_F_T_WITH_EXPLICIT_NO_VIEW_PENDING_CLEAN_REPLAY",
        "outputs": {
            **table_evidence,
            "report": file_evidence(outputs["report"], project_root=ROOT),
        },
        "metrics": {
            "market_session_count": int(len(panel)),
            "F_view_allowed_day_count": int(panel["F_state"].eq(VIEW_ALLOWED).sum()),
            "M_view_allowed_day_count": int(panel["M_state"].eq(VIEW_ALLOWED).sum()),
            "T_view_allowed_day_count": int(panel["T_state"].eq(VIEW_ALLOWED).sum()),
            "B2_feature_view_allowed_day_count": int(
                panel["B2_feature_state"].eq(VIEW_ALLOWED).sum()
            ),
            "complete_mft_view_allowed_day_count": int(view.sum()),
            "complete_mft_no_view_day_count": int((~view).sum()),
            "first_complete_mft_view_date": _iso_date_or_none(
                panel.loc[view, "date"], first=True
            ),
            "last_complete_mft_view_date": _iso_date_or_none(
                panel.loc[view, "date"], first=False
            ),
            "days_with_new_members": int(
                panel["newly_entered_member_count"].gt(0).sum()
            ),
            "new_member_entries": int(panel["newly_entered_member_count"].sum()),
            "no_view_reason_counts": no_view_reason_counts,
        },
        "validation": artifacts.validation,
        "g0_status": "PENDING_CLEAN_REPLAY_AND_TARGETED_TESTS",
        "g1_through_g7_status": "NOT_RUN",
        "return_evaluation": "NOT_ALLOWED",
        "actual_label_artifact_read": False,
        "actual_future_return_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(outputs["receipt"], receipt)
    status = {
        "program_id": config["program"]["program_id"],
        "execution_id": config["program"]["execution_id"],
        "updated_at": now,
        "mft_feature_build": "PASS_WITH_EXPLICIT_NO_VIEW_DATES_RETAINED",
        "g0_status": "PENDING_CLEAN_REPLAY_AND_TARGETED_TESTS",
        "g1_through_g7_status": "NOT_RUN",
        "next_allowed_step": "RUN_FROZEN_MFT_CLEAN_REPLAY_AND_TARGETED_TESTS",
        "model_state": "NO_VIEW_EXCEPT_DATES_WITH_COMPLETE_M_F_T",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_results_read": False,
        "position_impact": 0,
        "receipt": file_evidence(outputs["receipt"], project_root=ROOT),
    }
    atomic_write_json_new(outputs["status"], status)
    print("M/F/T 点时特征已构建；G0 尚待干净重放，绩效与交易继续关闭。")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        build(args.config)
        return 0
    except (
        EvidenceContractError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"M/F/T 特征构建失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
