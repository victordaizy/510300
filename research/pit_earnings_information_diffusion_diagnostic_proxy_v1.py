"""510300 PIT 盈余信息扩散：结果盲补齐与隔离代理诊断。

正式候选的数据硬门不会因本模块的代理诊断而改变。模块分两阶段运行：

1. ``build_outcome_blind_data`` 只读取公告、成分、权重和交易日历；
2. ``run_unreliable_prediction_diagnostic`` 仅在独立冻结清单存在后读取价格和未来标签。

第二阶段是用户明确要求的低置信度诊断，不是候选通过、组合回测、Paper、
Shadow 或交易授权。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "config"
    / "510300_pit_earnings_information_diffusion_diagnostic_proxy_2021_v1.yaml"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    raise TypeError(f"无法序列化类型：{type(value)!r}")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if value is pd.NA:
        return None
    return value


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, encoded)


def atomic_text(path: Path, payload: str) -> None:
    _atomic_bytes(path, payload.encode("utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents and path != ROOT.resolve():
        raise ValueError(f"路径越出项目目录：{relative}")
    return path


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if path.resolve() != CONFIG_PATH.resolve():
        raise ValueError("只允许执行冻结的代理诊断配置")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if protocol["formal_candidate_id"] != (
        "510300_PIT_EARNINGS_INFORMATION_DIFFUSION_10D_RISK_V1"
    ):
        raise ValueError("正式候选编号不匹配")
    if protocol["research_stage"] != "DIAGNOSTIC_ONLY_UNRELIABLE_INPUTS":
        raise ValueError("研究阶段不得改写为正式候选")
    if protocol["trade_assets_if_ever_promoted"] != ["510300.SH", "CASH_CNY"]:
        raise ValueError("资产边界必须严格为510300和人民币现金")
    forbidden_true = [
        "formal_candidate_status_may_be_promoted",
        "paper_signal_allowed",
        "shadow_signal_allowed",
        "position_mapping_allowed",
        "portfolio_evaluation_allowed",
        "order_generation_allowed",
        "broker_connection_allowed",
        "live_trading_authorized",
    ]
    if any(bool(protocol[field]) for field in forbidden_true):
        raise ValueError("隔离诊断不得获得候选晋级、仓位或交易授权")
    return config


def verify_manifest(path: Path, expected_status: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"冻结清单不存在：{path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != expected_status:
        raise ValueError(
            f"冻结清单状态错误：{manifest.get('status')} != {expected_status}"
        )
    entries = list(manifest.get("frozen_files", [])) + list(
        manifest.get("frozen_inputs", [])
    )
    for entry in entries:
        target = project_path(str(entry["path"]))
        if not target.exists():
            raise FileNotFoundError(f"冻结证据缺失：{entry['path']}")
        actual = sha256_file(target)
        if actual != str(entry["sha256"]):
            raise ValueError(
                f"冻结证据哈希漂移：{entry['path']}，{actual} != {entry['sha256']}"
            )
        if "bytes" in entry and int(target.stat().st_size) != int(entry["bytes"]):
            raise ValueError(f"冻结证据字节数漂移：{entry['path']}")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label}缺少字段：{missing}")


def _open_dates(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    _required_columns(calendar, ["date", "is_open"], "交易日历")
    mask = calendar["is_open"].fillna(False).astype(bool)
    dates = pd.DatetimeIndex(
        pd.to_datetime(calendar.loc[mask, "date"], errors="raise")
        .dt.normalize()
        .unique()
    ).sort_values()
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("交易日历日期必须唯一且递增")
    return dates


def map_to_next_open(
    notice_dates: pd.Series, open_dates: pd.DatetimeIndex
) -> pd.Series:
    normalized = pd.to_datetime(notice_dates, errors="raise").dt.normalize()
    positions = open_dates.searchsorted(normalized.to_numpy(), side="right")
    mapped = np.full(
        len(normalized), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    valid = positions < len(open_dates)
    mapped[valid] = open_dates.to_numpy(dtype="datetime64[ns]")[positions[valid]]
    return pd.Series(mapped, index=notice_dates.index, dtype="datetime64[ns]")


def validate_daily_membership(membership: pd.DataFrame) -> dict[str, Any]:
    _required_columns(
        membership,
        ["membership_date", "index_code", "symbol", "source_contract"],
        "官方逐日成分",
    )
    frame = membership.copy()
    frame["membership_date"] = pd.to_datetime(
        frame["membership_date"], errors="raise"
    ).dt.normalize()
    duplicate_rows = int(frame.duplicated(["membership_date", "symbol"]).sum())
    counts = frame.groupby("membership_date")["symbol"].nunique()
    index_codes = sorted(frame["index_code"].astype(str).unique().tolist())
    if duplicate_rows or counts.min() != 300 or counts.max() != 300:
        raise ValueError("官方逐日成分必须每日恰好300只且无重复")
    if index_codes != ["000300"]:
        raise ValueError(f"逐日成分指数代码不匹配：{index_codes}")
    return {
        "rows": int(len(frame)),
        "sessions": int(len(counts)),
        "first_date": counts.index.min().date().isoformat(),
        "last_date": counts.index.max().date().isoformat(),
        "members_min": int(counts.min()),
        "members_max": int(counts.max()),
        "duplicate_date_symbol_rows": duplicate_rows,
        "source_contracts": sorted(frame["source_contract"].astype(str).unique().tolist()),
    }


def validate_weights(weights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    _required_columns(
        weights,
        ["con_code", "trade_date", "weight", "source", "retrieved_at"],
        "历史月度权重",
    )
    frame = weights.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["weight"] = pd.to_numeric(frame["weight"], errors="raise")
    duplicate_rows = int(frame.duplicated(["trade_date", "con_code"]).sum())
    counts = frame.groupby("trade_date")["con_code"].nunique()
    sums = frame.groupby("trade_date")["weight"].sum()
    if duplicate_rows:
        raise ValueError("历史权重存在重复日期证券")
    if counts.min() != 300 or counts.max() != 300:
        raise ValueError("历史权重每期必须恰好300只")
    if sums.min() < 98.0 or sums.max() > 102.0:
        raise ValueError("历史权重和超出98%至102%结构范围")
    return frame, {
        "rows": int(len(frame)),
        "snapshots": int(len(counts)),
        "first_snapshot": counts.index.min().date().isoformat(),
        "last_snapshot": counts.index.max().date().isoformat(),
        "members_min": int(counts.min()),
        "members_max": int(counts.max()),
        "weight_sum_min_percent": float(sums.min()),
        "weight_sum_max_percent": float(sums.max()),
        "duplicate_date_security_rows": duplicate_rows,
        "sources": frame["source"].astype(str).value_counts().to_dict(),
        "revision_vintage_proven": False,
        "reliability_status": "UNVERIFIED_HISTORICAL_WEIGHT_REVISION_VINTAGE",
    }


def build_member_event_panel(
    facts: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
    open_dates: pd.DatetimeIndex,
    *,
    maximum_weight_age_days: int,
    first_public_status: str,
    fact_status: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        "announcement_id",
        "ts_code",
        "announcement_type",
        "official_publication_date",
        "known_at",
        "financial_period",
        "cumulative_core_np_lower_yuan",
        "cumulative_core_np_upper_yuan",
        "fact_status",
        "first_public_status",
        "market_price_read",
        "future_return_read",
    ]
    _required_columns(facts, required, "首次公开事实")
    if facts["market_price_read"].fillna(False).any():
        raise ValueError("事实表含市场价格读取标记")
    if facts["future_return_read"].fillna(False).any():
        raise ValueError("事实表含未来收益读取标记")
    selected = facts.loc[
        facts["first_public_status"].eq(first_public_status)
        & facts["fact_status"].eq(fact_status),
        required,
    ].copy()
    selected["notice_date"] = pd.to_datetime(
        selected["official_publication_date"], errors="raise"
    ).dt.normalize()
    selected["availability_date"] = map_to_next_open(selected["notice_date"], open_dates)
    for column in [
        "cumulative_core_np_lower_yuan",
        "cumulative_core_np_upper_yuan",
    ]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected["core_profit_mid_yuan"] = (
        selected["cumulative_core_np_lower_yuan"]
        + selected["cumulative_core_np_upper_yuan"]
    ) / 2.0
    selected["negative_core_fact"] = selected[
        "cumulative_core_np_upper_yuan"
    ].lt(0.0)
    selected["positive_core_fact"] = selected[
        "cumulative_core_np_lower_yuan"
    ].gt(0.0)
    selected["core_sign_ambiguous"] = ~(
        selected["negative_core_fact"] | selected["positive_core_fact"]
    )

    member = membership[["membership_date", "symbol", "source_contract"]].copy()
    member["membership_date"] = pd.to_datetime(
        member["membership_date"], errors="raise"
    ).dt.normalize()
    joined = selected.merge(
        member,
        left_on=["availability_date", "ts_code"],
        right_on=["membership_date", "symbol"],
        how="inner",
        validate="many_to_one",
    )

    snapshot_dates = pd.DatetimeIndex(weights["trade_date"].unique()).sort_values()
    positions = snapshot_dates.searchsorted(
        joined["notice_date"].to_numpy(dtype="datetime64[ns]"), side="left"
    ) - 1
    joined["weight_snapshot_date"] = pd.NaT
    valid_position = positions >= 0
    joined.loc[valid_position, "weight_snapshot_date"] = snapshot_dates.to_numpy(
        dtype="datetime64[ns]"
    )[positions[valid_position]]
    joined["weight_snapshot_age_days"] = (
        joined["notice_date"] - joined["weight_snapshot_date"]
    ).dt.days
    weight_rows = weights.rename(columns={"con_code": "ts_code"})[
        ["ts_code", "trade_date", "weight", "source", "retrieved_at"]
    ].rename(
        columns={
            "trade_date": "weight_snapshot_date",
            "weight": "weight_percent",
            "source": "weight_source",
            "retrieved_at": "weight_retrieved_at",
        }
    )
    joined = joined.merge(
        weight_rows,
        on=["ts_code", "weight_snapshot_date"],
        how="left",
        validate="many_to_one",
    )
    joined["valid_strict_prior_weight"] = (
        joined["weight_percent"].notna()
        & joined["weight_snapshot_age_days"].ge(1)
        & joined["weight_snapshot_age_days"].le(maximum_weight_age_days)
    )
    joined["membership_source_reliability"] = "PASS_OFFICIAL_REPLAY_MEMBERSHIP"
    joined["weight_source_reliability"] = (
        "UNVERIFIED_HISTORICAL_WEIGHT_REVISION_VINTAGE"
    )
    joined["fact_source_reliability"] = (
        "OFFICIAL_FIRST_PUBLIC_PDF_AUTOMATION_A3_FAILED_HUMAN_REVIEW_PENDING"
    )
    joined["market_price_read"] = False
    joined["future_return_read"] = False
    joined.sort_values(
        ["availability_date", "announcement_id"], kind="mergesort", inplace=True
    )
    joined.reset_index(drop=True, inplace=True)

    summary = {
        "all_first_public_complete_fact_count": int(len(selected)),
        "member_first_public_complete_fact_count": int(len(joined)),
        "member_valid_weight_fact_count": int(joined["valid_strict_prior_weight"].sum()),
        "member_valid_weight_fact_count_2021_plus": int(
            (
                joined["valid_strict_prior_weight"]
                & joined["notice_date"].ge("2021-01-01")
            ).sum()
        ),
        "member_negative_fact_count_2021_plus": int(
            (
                joined["valid_strict_prior_weight"]
                & joined["notice_date"].ge("2021-01-01")
                & joined["negative_core_fact"]
            ).sum()
        ),
        "valid_weight_member_facts_by_year": {
            str(int(year)): int(count)
            for year, count in joined.loc[joined["valid_strict_prior_weight"]]
            .groupby(joined.loc[joined["valid_strict_prior_weight"], "notice_date"].dt.year)
            .size()
            .items()
        },
    }
    summary["years_with_at_least_20_valid_weight_member_facts"] = int(
        sum(count >= 20 for count in summary["valid_weight_member_facts_by_year"].values())
    )
    return joined, summary


def build_daily_outcome_blind_features(
    event_panel: pd.DataFrame,
    membership_dates: pd.Series,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        pd.to_datetime(membership_dates, errors="raise").dt.normalize().unique()
    ).sort_values()
    daily = pd.DataFrame({"date": dates})
    valid = event_panel.loc[event_panel["valid_strict_prior_weight"]].copy()
    valid["event_count"] = 1
    valid["negative_event_count"] = valid["negative_core_fact"].astype(int)
    valid["ambiguous_sign_event_count"] = valid["core_sign_ambiguous"].astype(int)
    valid["event_weight_percent"] = valid["weight_percent"].astype(float)
    valid["negative_event_weight_percent"] = (
        valid["weight_percent"].astype(float) * valid["negative_core_fact"].astype(float)
    )
    aggregate = (
        valid.groupby("availability_date", as_index=False)
        .agg(
            first_public_fact_count=("event_count", "sum"),
            negative_fact_count=("negative_event_count", "sum"),
            ambiguous_sign_fact_count=("ambiguous_sign_event_count", "sum"),
            first_public_weight_percent=("event_weight_percent", "sum"),
            negative_weight_percent=("negative_event_weight_percent", "sum"),
        )
        .rename(columns={"availability_date": "date"})
    )
    daily = daily.merge(aggregate, on="date", how="left", validate="one_to_one")
    count_columns = [
        "first_public_fact_count",
        "negative_fact_count",
        "ambiguous_sign_fact_count",
        "first_public_weight_percent",
        "negative_weight_percent",
    ]
    daily[count_columns] = daily[count_columns].fillna(0.0)
    for window in (5, 20):
        daily[f"first_public_fact_count_{window}d"] = daily[
            "first_public_fact_count"
        ].rolling(window, min_periods=1).sum()
        daily[f"negative_fact_count_{window}d"] = daily[
            "negative_fact_count"
        ].rolling(window, min_periods=1).sum()
        daily[f"first_public_weight_share_{window}d"] = (
            daily["first_public_weight_percent"].rolling(window, min_periods=1).sum()
            / 100.0
        )
        daily[f"negative_weight_share_{window}d"] = (
            daily["negative_weight_percent"].rolling(window, min_periods=1).sum()
            / 100.0
        )
    daily["fact_window_has_information_20d"] = daily[
        "first_public_fact_count_20d"
    ].gt(0.0)
    daily["negative_fact_breadth_20d"] = np.divide(
        daily["negative_fact_count_20d"],
        daily["first_public_fact_count_20d"],
        out=np.zeros(len(daily), dtype=float),
        where=daily["first_public_fact_count_20d"].to_numpy(dtype=float) > 0.0,
    )
    daily["negative_weight_breadth_20d"] = np.divide(
        daily["negative_weight_share_20d"],
        daily["first_public_weight_share_20d"],
        out=np.zeros(len(daily), dtype=float),
        where=daily["first_public_weight_share_20d"].to_numpy(dtype=float) > 0.0,
    )
    daily["data_reliability"] = "UNRELIABLE_SPARSE_NO_VIEW_DOMINATED_PROXY"
    daily["market_price_read"] = False
    daily["future_return_read"] = False
    daily["target_label_created"] = False
    return daily


def render_data_markdown(result: dict[str, Any]) -> str:
    prevalence = result["prevalence"]
    facts = result["facts_gate"]
    return "\n".join(
        [
            "# 510300 PIT 盈余信息扩散 V1.0.3：结果盲数据补齐",
            "",
            f"- 数据产物状态：`{result['status']}`",
            f"- 正式候选权威状态：`{result['authoritative_candidate_status']}`",
            "- 市场价格读取：`0`",
            "- 未来收益读取：`0`",
            "- 模型、标签、组合回测、仓位和订单：均未创建",
            "",
            "## 已补齐",
            "",
            f"- 官方逐日成分：{result['membership']['sessions']:,}个交易日、"
            f"{result['membership']['rows']:,}行，每日{result['membership']['members_min']}只。",
            f"- 结构完整月度权重：{result['weights']['snapshots']}期；"
            "历史修订版本来源仍不可证明。",
            f"- 官方PDF首次公开完整核心利润事实：全市场{prevalence['all_first_public_complete_fact_count']}条；"
            f"历史沪深300成员{prevalence['member_first_public_complete_fact_count']}条；"
            f"拥有严格前序权重{prevalence['member_valid_weight_fact_count']}条。",
            f"- 2021年以来拥有严格前序权重：{prevalence['member_valid_weight_fact_count_2021_plus']}条；"
            f"其中严格负利润区间{prevalence['member_negative_fact_count_2021_plus']}条。",
            "",
            "## 正式硬门",
            "",
            f"- 自动事实A3：{facts['a3_success_rate']:.2%}，要求至少{facts['required_rate']:.0%}，`FAIL`。",
            f"- 双人独立复核完成对数：{facts['actual_completed_human_review_pairs']}，`PENDING`。",
            f"- 满足每年至少20条事实的年份："
            f"{prevalence['years_with_at_least_20_valid_weight_member_facts']}，要求8年，`FAIL`。",
            "- 权重版本来源门：`FAIL_UNVERIFIED_HISTORICAL_WEIGHT_REVISION_VINTAGE`。",
            "",
            "全部缺失、模糊和不可读事实继续保留为 `NO_VIEW`，没有人工补值。"
            "本产物只证明数据已按现有来源完整汇总，不证明正式候选数据合格。",
            "",
        ]
    )


def build_outcome_blind_data(config: dict[str, Any]) -> dict[str, Any]:
    source_manifest = project_path(config["paths"]["source_manifest"])
    manifest = verify_manifest(
        source_manifest, "FROZEN_OUTCOME_BLIND_SOURCE_CORRECTION_BEFORE_OUTPUT"
    )
    inputs = config["inputs"]
    facts = pd.read_parquet(project_path(inputs["facts"]))
    facts_receipt = _read_json(project_path(inputs["facts_receipt"]))
    membership = pd.read_parquet(project_path(inputs["official_daily_membership"]))
    membership_audit = _read_json(project_path(inputs["membership_independent_audit"]))
    weights_raw = pd.read_parquet(project_path(inputs["monthly_weights"]))
    weight_receipt = _read_json(project_path(inputs["monthly_weights_status"]))
    calendar = pd.read_parquet(project_path(inputs["trading_calendar"]))

    membership_summary = validate_daily_membership(membership)
    if membership_audit.get("status") != (
        "PASS_MEMBERSHIP_ADMISSION_WEIGHT_BLOCK_PRESERVED"
    ):
        raise ValueError("逐日成分独立审计状态未通过")
    weights, weight_summary = validate_weights(weights_raw)
    open_dates = _open_dates(calendar)
    contract = config["outcome_blind_contract"]
    events, prevalence = build_member_event_panel(
        facts,
        membership,
        weights,
        open_dates,
        maximum_weight_age_days=int(contract["maximum_weight_age_calendar_days"]),
        first_public_status=str(contract["first_public_required_status"]),
        fact_status=str(contract["fact_required_status"]),
    )
    daily = build_daily_outcome_blind_features(events, membership["membership_date"])

    event_path = project_path(config["paths"]["member_event_panel"])
    daily_path = project_path(config["paths"]["daily_outcome_blind_features"])
    atomic_parquet(event_path, events)
    atomic_parquet(daily_path, daily)

    required_rate = float(
        config["formal_hard_gates"]["minimum_fact_extraction_success_rate"]
    )
    a3_rate = float(facts_receipt.get("a3_success_rate", 0.0))
    human_pairs = int(
        facts_receipt.get("formal_dual_review", {}).get(
            "actual_completed_human_pairs", 0
        )
    )
    formal_gates = {
        "official_daily_membership": {
            "passed": True,
            "actual": membership_summary,
            "required": "300_MEMBERS_EACH_OPEN_SESSION",
        },
        "version_proven_pit_weights": {
            "passed": False,
            "actual": weight_summary["reliability_status"],
            "required": "VERSION_PROVEN_PIT_WEIGHTS",
        },
        "fact_automation_a3": {
            "passed": bool(a3_rate >= required_rate),
            "actual": a3_rate,
            "required": required_rate,
        },
        "dual_human_review": {
            "passed": False,
            "actual_completed_pairs": human_pairs,
            "required": "TWO_REAL_INDEPENDENT_REVIEWERS_COMPLETE",
        },
        "first_public_weighted_fact_prevalence": {
            "passed": bool(
                prevalence["member_valid_weight_fact_count"]
                >= int(
                    config["formal_hard_gates"]
                    ["minimum_first_public_weighted_fact_count"]
                )
                and prevalence["years_with_at_least_20_valid_weight_member_facts"]
                >= int(
                    config["formal_hard_gates"]
                    ["minimum_years_with_20_first_public_weighted_facts"]
                )
            ),
            "actual": {
                "count": prevalence["member_valid_weight_fact_count"],
                "years_with_at_least_20": prevalence[
                    "years_with_at_least_20_valid_weight_member_facts"
                ],
            },
            "required": {
                "minimum_count": config["formal_hard_gates"]
                ["minimum_first_public_weighted_fact_count"],
                "minimum_years_with_at_least_20": config["formal_hard_gates"]
                ["minimum_years_with_20_first_public_weighted_facts"],
            },
        },
    }
    result = {
        "data_correction_id": config["protocol"]["data_correction_id"],
        "formal_candidate_id": config["protocol"]["formal_candidate_id"],
        "status": "PASS_OUTCOME_BLIND_PANEL_BUILT_WITH_EXPLICIT_LIMITATIONS",
        "authoritative_candidate_status": (
            "BLOCKED_NO_PIT_CSI300_MEMBERSHIP_OR_WEIGHTS"
        ),
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "source_manifest": {
            "path": config["paths"]["source_manifest"],
            "sha256": sha256_file(source_manifest),
            "status": manifest["status"],
        },
        "membership": membership_summary,
        "membership_independent_audit_status": membership_audit.get("status"),
        "weights": weight_summary,
        "weights_source_receipt_status": weight_receipt.get("status"),
        "facts_gate": {
            "automation_status": facts_receipt.get("status"),
            "a2_status": facts_receipt.get("A2_status"),
            "a3_status": facts_receipt.get("A3_status"),
            "a3_denominator_count": facts_receipt.get("a3_denominator_count"),
            "a3_success_count": facts_receipt.get("a3_success_count"),
            "a3_success_rate": a3_rate,
            "required_rate": required_rate,
            "actual_completed_human_review_pairs": human_pairs,
            "fact_status_counts": facts_receipt.get("fact_status_counts", {}),
            "first_public_status_counts": facts_receipt.get(
                "first_public_status_counts", {}
            ),
        },
        "prevalence": prevalence,
        "formal_gates": formal_gates,
        "formal_candidate_data_passed": False,
        "limitations": config["reliability_quarantine"]["required_labels"],
        "artifacts": {
            "member_event_panel": {
                "path": config["paths"]["member_event_panel"],
                "sha256": sha256_file(event_path),
                "rows": int(len(events)),
            },
            "daily_outcome_blind_features": {
                "path": config["paths"]["daily_outcome_blind_features"],
                "sha256": sha256_file(daily_path),
                "rows": int(len(daily)),
                "first_date": daily["date"].min().date().isoformat(),
                "last_date": daily["date"].max().date().isoformat(),
            },
        },
        "market_price_reads": 0,
        "future_return_reads": 0,
        "target_label_created": False,
        "prediction_model_created": False,
        "portfolio_evaluation_performed": False,
        "model_action": "ABSTAIN",
        "model_position_target": "UNSET",
        "paper_signal_allowed": False,
        "shadow_signal_allowed": False,
        "live_trading_authorized": False,
        "return_evaluation": "NOT_ALLOWED_IN_OUTCOME_BLIND_DATA_STAGE",
    }
    result_path = project_path(config["paths"]["data_result_json"])
    markdown_path = project_path(config["paths"]["data_result_markdown"])
    atomic_json(result_path, result)
    atomic_text(markdown_path, render_data_markdown(result))
    return result


def _normalize_market(frame: pd.DataFrame) -> pd.DataFrame:
    _required_columns(
        frame, ["date", "open", "high", "low", "close", "volume", "symbol"], "510300日线"
    )
    market = frame.copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market.sort_values("date", kind="mergesort", inplace=True)
    market.reset_index(drop=True, inplace=True)
    if market["date"].duplicated().any():
        raise ValueError("510300日线存在重复日期")
    if set(market["symbol"].dropna().astype(str).unique()) != {"510300.SH"}:
        raise ValueError("510300日线代码不匹配")
    for column in ["open", "high", "low", "close", "volume"]:
        market[column] = pd.to_numeric(market[column], errors="raise")
    if market[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("510300价格存在空值")
    if (market[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise ValueError("510300价格存在非正值")
    return market


def _normalize_dividends(frame: pd.DataFrame) -> pd.DataFrame:
    _required_columns(
        frame,
        ["symbol", "record_date", "ex_date", "payment_date", "cash_dividend_per_share"],
        "510300分红",
    )
    dividends = frame.copy()
    if set(dividends["symbol"].astype(str).unique()) != {"510300.SH"}:
        raise ValueError("分红输入混入非510300证券")
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    if dividends["ex_date"].duplicated().any():
        raise ValueError("分红除息日重复")
    return dividends


def prepare_market_features_and_labels(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    output = _normalize_market(market)
    dividend_map = dividends.groupby("ex_date")["cash_dividend_per_share"].sum()
    output["cash_dividend_per_share"] = output["date"].map(dividend_map).fillna(0.0)
    output["previous_close"] = output["close"].shift(1)
    output["daily_total_return"] = (
        (output["close"] + output["cash_dividend_per_share"])
        / output["previous_close"]
        - 1.0
    )
    wealth = (1.0 + output["daily_total_return"].fillna(0.0)).cumprod()
    output["trailing_5d_total_return"] = wealth / wealth.shift(5) - 1.0
    output["trailing_20d_realized_volatility"] = (
        output["daily_total_return"].rolling(20, min_periods=20).std(ddof=1)
        * math.sqrt(252.0)
    )
    downside_squared = output["daily_total_return"].clip(upper=0.0).pow(2)
    output["trailing_20d_downside_semideviation"] = (
        downside_squared.rolling(20, min_periods=20).mean().pow(0.5)
        * math.sqrt(252.0)
    )

    opens = output["open"].to_numpy(dtype=float)
    cash = output["cash_dividend_per_share"].to_numpy(dtype=float)
    cash_prefix = np.concatenate([[0.0], np.cumsum(cash)])
    future_return = np.full(len(output), np.nan, dtype=float)
    future_end = np.full(
        len(output), np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )
    next_day_return = np.full(len(output), np.nan, dtype=float)
    for decision_index in range(len(output)):
        entry_index = decision_index + 1
        exit_index = entry_index + int(horizon)
        if exit_index < len(output):
            dividend_cash = cash_prefix[exit_index + 1] - cash_prefix[entry_index + 1]
            future_return[decision_index] = (
                opens[exit_index] + dividend_cash
            ) / opens[entry_index] - 1.0
            future_end[decision_index] = output.iloc[exit_index]["date"].to_datetime64()
        next_exit = entry_index + 1
        if next_exit < len(output):
            dividend_cash = cash_prefix[next_exit + 1] - cash_prefix[entry_index + 1]
            next_day_return[decision_index] = (
                opens[next_exit] + dividend_cash
            ) / opens[entry_index] - 1.0
    output["future_10d_open_to_open_total_return"] = future_return
    output["future_10d_label_end_date"] = pd.to_datetime(future_end)
    output["target_10d_negative"] = np.where(
        np.isfinite(future_return), future_return < 0.0, pd.NA
    )
    output["next_day_open_to_open_total_return"] = next_day_return
    return output


@dataclass
class ProbabilityModel:
    scaler: StandardScaler | None
    model: LogisticRegression | None
    constant_probability: float | None
    calibration_model: LogisticRegression | None
    calibration_status: str

    def raw_probability(self, features: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full(len(features), self.constant_probability, dtype=float)
        assert self.scaler is not None and self.model is not None
        transformed = self.scaler.transform(features)
        return self.model.predict_proba(transformed)[:, 1]

    def probability(self, features: np.ndarray) -> np.ndarray:
        raw = self.raw_probability(features)
        if self.calibration_model is None:
            return raw
        logits = np.log(np.clip(raw, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw, 1e-6, 1.0))
        return self.calibration_model.predict_proba(logits.reshape(-1, 1))[:, 1]


def fit_probability_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_labels: np.ndarray,
    *,
    random_seed: int,
) -> ProbabilityModel:
    unique_train = np.unique(train_labels)
    if len(unique_train) < 2:
        probability = float(np.mean(train_labels))
        return ProbabilityModel(
            scaler=None,
            model=None,
            constant_probability=probability,
            calibration_model=None,
            calibration_status="NO_VIEW_SINGLE_CLASS_TRAINING_CONSTANT_PROBABILITY",
        )
    scaler = StandardScaler()
    transformed = scaler.fit_transform(train_features)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=int(random_seed),
    )
    model.fit(transformed, train_labels)
    shell = ProbabilityModel(
        scaler=scaler,
        model=model,
        constant_probability=None,
        calibration_model=None,
        calibration_status="UNCALIBRATED",
    )
    raw_calibration = shell.raw_probability(calibration_features)
    if len(np.unique(calibration_labels)) < 2 or float(np.std(raw_calibration)) < 1e-10:
        shell.calibration_status = "NO_VIEW_CALIBRATION_SINGLE_CLASS_OR_CONSTANT_SCORE"
        return shell
    logits = np.log(
        np.clip(raw_calibration, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - raw_calibration, 1e-6, 1.0)
    )
    calibration_model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=int(random_seed),
    )
    calibration_model.fit(logits.reshape(-1, 1), calibration_labels)
    shell.calibration_model = calibration_model
    shell.calibration_status = "PASS_FIXED_SIGMOID_CALIBRATION"
    return shell


def classification_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    predicted_bool = predicted.astype(bool)
    tp = int(np.sum(labels_bool & predicted_bool))
    fn = int(np.sum(labels_bool & ~predicted_bool))
    fp = int(np.sum(~labels_bool & predicted_bool))
    tn = int(np.sum(~labels_bool & ~predicted_bool))
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": float(recall),
        "false_positive_rate": float(fpr),
        "precision": float(precision),
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    grid_start: float,
    grid_stop: float,
    grid_step: float,
    recall_min: float,
    fpr_max: float,
) -> tuple[float, dict[str, Any]]:
    thresholds = np.round(
        np.arange(grid_start, grid_stop + grid_step / 2.0, grid_step), 10
    )
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = classification_metrics(labels, probabilities >= threshold)
        recall = float(metrics["recall"])
        fpr = float(metrics["false_positive_rate"])
        feasible = bool(
            np.isfinite(recall)
            and np.isfinite(fpr)
            and recall >= recall_min
            and fpr <= fpr_max
        )
        violation = max(0.0, recall_min - recall) + max(0.0, fpr - fpr_max)
        rows.append(
            {
                "threshold": float(threshold),
                **metrics,
                "feasible": feasible,
                "normalized_violation": float(violation),
            }
        )
    feasible_rows = [row for row in rows if row["feasible"]]
    if feasible_rows:
        chosen = sorted(
            feasible_rows,
            key=lambda row: (
                -float(row["precision"]),
                float(row["false_positive_rate"]),
                -float(row["threshold"]),
            ),
        )[0]
        selection_status = "PASS_CALIBRATION_RECALL_FPR_CONSTRAINT"
    else:
        chosen = sorted(
            rows,
            key=lambda row: (
                float(row["normalized_violation"]),
                -float(row["precision"]),
                float(row["false_positive_rate"]),
                -float(row["threshold"]),
            ),
        )[0]
        selection_status = "FAIL_NO_CALIBRATION_THRESHOLD_MET_RECALL_FPR_CONSTRAINT"
    return float(chosen["threshold"]), {**chosen, "status": selection_status}


def _quarter_key(dates: pd.Series) -> pd.Series:
    return dates.dt.to_period("Q").astype(str)


def anchored_quarterly_walk_forward(
    frame: pd.DataFrame,
    *,
    baseline_features: list[str],
    augmented_features: list[str],
    oos_start: pd.Timestamp,
    initial_training_observations: int,
    calibration_observations: int,
    random_seed: int,
    threshold_contract: dict[str, Any],
    prediction_gates: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy().reset_index(drop=True)
    required = list(dict.fromkeys(baseline_features + augmented_features))
    valid = (
        working[required].notna().all(axis=1)
        & working["target_10d_negative"].notna()
        & working["future_10d_label_end_date"].notna()
    )
    working = working.loc[valid].copy().reset_index(drop=True)
    working["target_10d_negative"] = working["target_10d_negative"].astype(bool)
    test_scope = working.loc[working["date"].ge(oos_start)].copy()
    prediction_parts: list[pd.DataFrame] = []
    refit_rows: list[dict[str, Any]] = []
    for quarter, test in test_scope.groupby(_quarter_key(test_scope["date"]), sort=True):
        test_start = pd.Timestamp(test["date"].min())
        calibration_candidates = working.loc[
            working["future_10d_label_end_date"].lt(test_start)
            & working["date"].lt(test_start)
        ].copy()
        if len(calibration_candidates) < calibration_observations:
            refit_rows.append(
                {
                    "quarter": quarter,
                    "status": "SKIPPED_INSUFFICIENT_CALIBRATION_HISTORY",
                    "test_start": test_start,
                    "test_end": test["date"].max(),
                    "calibration_candidate_count": int(len(calibration_candidates)),
                }
            )
            continue
        calibration = calibration_candidates.tail(calibration_observations).copy()
        calibration_start = pd.Timestamp(calibration["date"].min())
        training = working.loc[
            working["future_10d_label_end_date"].lt(calibration_start)
            & working["date"].lt(calibration_start)
        ].copy()
        if len(training) < initial_training_observations:
            refit_rows.append(
                {
                    "quarter": quarter,
                    "status": "SKIPPED_INSUFFICIENT_INITIAL_TRAINING_HISTORY",
                    "test_start": test_start,
                    "test_end": test["date"].max(),
                    "training_count": int(len(training)),
                    "required_training_count": int(initial_training_observations),
                }
            )
            continue

        y_train = training["target_10d_negative"].to_numpy(dtype=int)
        y_cal = calibration["target_10d_negative"].to_numpy(dtype=int)
        baseline_model = fit_probability_model(
            training[baseline_features].to_numpy(dtype=float),
            y_train,
            calibration[baseline_features].to_numpy(dtype=float),
            y_cal,
            random_seed=random_seed,
        )
        augmented_model = fit_probability_model(
            training[augmented_features].to_numpy(dtype=float),
            y_train,
            calibration[augmented_features].to_numpy(dtype=float),
            y_cal,
            random_seed=random_seed,
        )
        calibration_augmented = augmented_model.probability(
            calibration[augmented_features].to_numpy(dtype=float)
        )
        threshold, threshold_audit = select_threshold(
            y_cal,
            calibration_augmented,
            grid_start=float(threshold_contract["start"]),
            grid_stop=float(threshold_contract["stop"]),
            grid_step=float(threshold_contract["step"]),
            recall_min=float(prediction_gates["oos_event_recall_min"]),
            fpr_max=float(prediction_gates["oos_false_positive_rate_max"]),
        )
        predicted = test.copy()
        predicted["baseline_probability"] = baseline_model.probability(
            test[baseline_features].to_numpy(dtype=float)
        )
        predicted["augmented_probability"] = augmented_model.probability(
            test[augmented_features].to_numpy(dtype=float)
        )
        predicted["threshold"] = threshold
        predicted["high_risk_prediction"] = predicted[
            "augmented_probability"
        ].ge(threshold)
        predicted["refit_quarter"] = quarter
        predicted["calibration_threshold_status"] = threshold_audit["status"]
        prediction_parts.append(predicted)
        refit_rows.append(
            {
                "quarter": quarter,
                "status": "PASS_REFIT_DIAGNOSTIC_ONLY",
                "training_start": training["date"].min(),
                "training_end": training["date"].max(),
                "training_last_label_end": training[
                    "future_10d_label_end_date"
                ].max(),
                "training_count": int(len(training)),
                "training_positive_count": int(y_train.sum()),
                "calibration_start": calibration["date"].min(),
                "calibration_end": calibration["date"].max(),
                "calibration_last_label_end": calibration[
                    "future_10d_label_end_date"
                ].max(),
                "calibration_count": int(len(calibration)),
                "calibration_positive_count": int(y_cal.sum()),
                "test_start": test["date"].min(),
                "test_end": test["date"].max(),
                "test_count": int(len(test)),
                "baseline_calibration_status": baseline_model.calibration_status,
                "augmented_calibration_status": augmented_model.calibration_status,
                "selected_threshold": threshold,
                "threshold_status": threshold_audit["status"],
                "calibration_recall": threshold_audit["recall"],
                "calibration_false_positive_rate": threshold_audit[
                    "false_positive_rate"
                ],
                "calibration_precision": threshold_audit["precision"],
            }
        )
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    refits = pd.DataFrame(refit_rows)
    return predictions, refits


def block_bootstrap_brier_improvement(
    labels: np.ndarray,
    baseline_probability: np.ndarray,
    augmented_probability: np.ndarray,
    *,
    block_length: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    n = len(labels)
    if n == 0:
        return {
            "replicates": int(replicates),
            "block_length": int(block_length),
            "median_improvement": None,
            "ci_2_5": None,
            "ci_97_5": None,
            "positive_fraction": None,
        }
    starts = np.arange(0, n, block_length, dtype=int)
    blocks = [np.arange(start, min(start + block_length, n), dtype=int) for start in starts]
    rng = np.random.default_rng(seed)
    improvements = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled: list[np.ndarray] = []
        while sum(len(block) for block in sampled) < n:
            sampled.append(blocks[int(rng.integers(0, len(blocks)))])
        indices = np.concatenate(sampled)[:n]
        y = labels[indices]
        base = baseline_probability[indices]
        augmented = augmented_probability[indices]
        improvements[replicate] = float(
            np.mean((y - base) ** 2) - np.mean((y - augmented) ** 2)
        )
    return {
        "replicates": int(replicates),
        "block_length": int(block_length),
        "median_improvement": float(np.median(improvements)),
        "ci_2_5": float(np.quantile(improvements, 0.025)),
        "ci_97_5": float(np.quantile(improvements, 0.975)),
        "positive_fraction": float(np.mean(improvements > 0.0)),
    }


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    gates: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    if predictions.empty:
        return {
            "status": "PROGRAM_FAILED_NO_OOS_PREDICTIONS",
            "prediction_gates_passed": False,
            "gates": {},
        }
    labels = predictions["target_10d_negative"].to_numpy(dtype=bool)
    signal = predictions["high_risk_prediction"].to_numpy(dtype=bool)
    baseline = predictions["baseline_probability"].to_numpy(dtype=float)
    augmented = predictions["augmented_probability"].to_numpy(dtype=float)
    future_return = predictions[
        "future_10d_open_to_open_total_return"
    ].to_numpy(dtype=float)
    classification = classification_metrics(labels.astype(int), signal)
    baseline_brier = float(np.mean((labels.astype(float) - baseline) ** 2))
    augmented_brier = float(np.mean((labels.astype(float) - augmented) ** 2))
    brier_improvement = baseline_brier - augmented_brier
    brier_skill = 1.0 - augmented_brier / baseline_brier if baseline_brier > 0 else float("nan")
    loss_denominator = float(np.sum(-future_return[labels]))
    loss_weighted_recall = (
        float(np.sum(-future_return[labels & signal])) / loss_denominator
        if loss_denominator > 0.0
        else float("nan")
    )

    positive_next = predictions.loc[
        predictions["next_day_open_to_open_total_return"].gt(0.0)
    ].nlargest(10, "next_day_open_to_open_total_return")
    upside_denominator = float(positive_next["next_day_open_to_open_total_return"].sum())
    top_upside_miss_ratio = (
        float(
            positive_next.loc[
                positive_next["high_risk_prediction"],
                "next_day_open_to_open_total_return",
            ].sum()
        )
        / upside_denominator
        if upside_denominator > 0.0
        else float("nan")
    )

    ordered = predictions.sort_values("date", kind="mergesort").copy()
    previous_signal = ordered["high_risk_prediction"].shift(1, fill_value=False)
    ordered["risk_entry"] = ordered["high_risk_prediction"] & ~previous_signal
    round_trips_by_year = (
        ordered.groupby(ordered["date"].dt.year)["risk_entry"].sum().astype(int).to_dict()
    )
    maximum_annual_round_trips = int(max(round_trips_by_year.values(), default=0))

    yearly_rows: list[dict[str, Any]] = []
    for year, group in ordered.groupby(ordered["date"].dt.year, sort=True):
        y = group["target_10d_negative"].astype(float).to_numpy()
        base = group["baseline_probability"].to_numpy(dtype=float)
        aug = group["augmented_probability"].to_numpy(dtype=float)
        improvement = float(np.mean((y - base) ** 2) - np.mean((y - aug) ** 2))
        yearly_rows.append(
            {
                "year": int(year),
                "observations": int(len(group)),
                "baseline_brier": float(np.mean((y - base) ** 2)),
                "augmented_brier": float(np.mean((y - aug) ** 2)),
                "brier_improvement": improvement,
            }
        )
    positive_improvements = [
        row["brier_improvement"] for row in yearly_rows if row["brier_improvement"] > 0.0
    ]
    positive_years = len(positive_improvements)
    positive_total = float(sum(positive_improvements))
    maximum_year_share = (
        float(max(positive_improvements) / positive_total)
        if positive_total > 0.0
        else float("nan")
    )

    bootstrap_result = block_bootstrap_brier_improvement(
        labels.astype(float),
        baseline,
        augmented,
        block_length=int(bootstrap["block_length_trading_days"]),
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )
    gate_rows = {
        "oos_event_recall": {
            "passed": bool(classification["recall"] >= float(gates["oos_event_recall_min"])),
            "actual": classification["recall"],
            "required": gates["oos_event_recall_min"],
        },
        "oos_false_positive_rate": {
            "passed": bool(
                classification["false_positive_rate"]
                <= float(gates["oos_false_positive_rate_max"])
            ),
            "actual": classification["false_positive_rate"],
            "required_max": gates["oos_false_positive_rate_max"],
        },
        "oos_precision": {
            "passed": bool(classification["precision"] >= float(gates["oos_precision_min"])),
            "actual": classification["precision"],
            "required": gates["oos_precision_min"],
        },
        "brier_skill_vs_price_baseline": {
            "passed": bool(np.isfinite(brier_skill) and brier_skill > 0.0),
            "actual": brier_skill,
            "required": "STRICTLY_POSITIVE",
        },
        "loss_weighted_recall": {
            "passed": bool(
                np.isfinite(loss_weighted_recall)
                and loss_weighted_recall >= float(gates["loss_weighted_recall_min"])
            ),
            "actual": loss_weighted_recall,
            "required": gates["loss_weighted_recall_min"],
        },
        "top_upside_miss_ratio": {
            "passed": bool(
                np.isfinite(top_upside_miss_ratio)
                and top_upside_miss_ratio <= float(gates["top_upside_miss_ratio_max"])
            ),
            "actual": top_upside_miss_ratio,
            "required_max": gates["top_upside_miss_ratio_max"],
        },
        "annual_round_trips": {
            "passed": bool(maximum_annual_round_trips <= int(gates["annual_round_trips_max"])),
            "actual_max": maximum_annual_round_trips,
            "required_max": gates["annual_round_trips_max"],
        },
        "positive_brier_improvement_years": {
            "passed": bool(
                positive_years >= int(gates["minimum_positive_brier_improvement_years"])
            ),
            "actual": positive_years,
            "required": gates["minimum_positive_brier_improvement_years"],
        },
        "maximum_single_year_positive_improvement_share": {
            "passed": bool(
                np.isfinite(maximum_year_share)
                and maximum_year_share
                <= float(gates["maximum_single_year_positive_improvement_share"])
            ),
            "actual": maximum_year_share,
            "required_max": gates["maximum_single_year_positive_improvement_share"],
        },
    }
    return {
        "status": "PASS_DIAGNOSTIC_PREDICTION_GATES" if all(
            row["passed"] for row in gate_rows.values()
        ) else "FAIL_DIAGNOSTIC_PREDICTION_GATES",
        "observations": int(len(predictions)),
        "classification": classification,
        "baseline_brier": baseline_brier,
        "augmented_brier": augmented_brier,
        "brier_improvement": brier_improvement,
        "brier_skill_vs_price_baseline": brier_skill,
        "loss_weighted_recall": loss_weighted_recall,
        "top_10_upside_observations": int(len(positive_next)),
        "top_upside_miss_ratio": top_upside_miss_ratio,
        "round_trips_by_year": {str(int(k)): int(v) for k, v in round_trips_by_year.items()},
        "maximum_annual_round_trips": maximum_annual_round_trips,
        "yearly_brier": yearly_rows,
        "positive_brier_improvement_years": positive_years,
        "maximum_single_year_positive_improvement_share": maximum_year_share,
        "block_bootstrap_brier_improvement": bootstrap_result,
        "gates": gate_rows,
        "prediction_gates_passed": bool(all(row["passed"] for row in gate_rows.values())),
    }


def render_diagnostic_markdown(result: dict[str, Any]) -> str:
    evaluation = result["evaluation"]
    classification = evaluation.get("classification", {})
    return "\n".join(
        [
            "# 510300 PIT 盈余信息扩散：2021+ 不可靠输入代理诊断",
            "",
            f"- 隔离诊断状态：`{result['status']}`",
            f"- 预测门观察结果：`{evaluation.get('status')}`",
            f"- 正式候选权威状态：`{result['authoritative_candidate_status']}`",
            "- 组合回测：`NOT_PERFORMED`",
            "- 仓位、Paper、Shadow、订单与实盘：均未授权",
            "",
            "## 样本外预测观察",
            "",
            f"- 观察数：{evaluation.get('observations', 0):,}",
            f"- 召回率：{classification.get('recall', float('nan')):.2%}",
            f"- 误报率：{classification.get('false_positive_rate', float('nan')):.2%}",
            f"- 精确率：{classification.get('precision', float('nan')):.2%}",
            f"- 相对价格基线 Brier Skill："
            f"{evaluation.get('brier_skill_vs_price_baseline', float('nan')):.4f}",
            f"- 损失加权召回：{evaluation.get('loss_weighted_recall', float('nan')):.2%}",
            f"- 前10个关键上涨日遗漏贡献占比："
            f"{evaluation.get('top_upside_miss_ratio', float('nan')):.2%}",
            f"- 单年最大高风险往返：{evaluation.get('maximum_annual_round_trips', 0)}",
            "",
            "## 为什么不能解释为合格候选",
            "",
            "- 历史权重数值虽结构完整，但供应商历史修订版本不可证明。",
            "- 自动事实 A3 只有 55.42%，低于冻结的 90% 门槛。",
            "- 双人独立人工复核尚未完成。",
            "- 2021年以来正式沪深300加权首次事实只有46条，连续特征主要由零事件日构成。",
            "",
            "因此本结果只回答“强行继续时会观察到什么”，不能通过或拒绝正式候选，"
            "也不能触发组合评价。正式状态、现有持仓和交易授权均未改变。",
            "",
        ]
    )


def run_unreliable_prediction_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    diagnostic_manifest_path = project_path(config["paths"]["diagnostic_manifest"])
    manifest = verify_manifest(
        diagnostic_manifest_path,
        "FROZEN_DIAGNOSTIC_PROXY_BEFORE_FUTURE_OUTCOME_READ",
    )
    data_result_path = project_path(config["paths"]["data_result_json"])
    data_result = _read_json(data_result_path)
    if data_result.get("status") != (
        "PASS_OUTCOME_BLIND_PANEL_BUILT_WITH_EXPLICIT_LIMITATIONS"
    ):
        raise ValueError("结果盲数据面板状态不允许进入隔离诊断")
    if bool(data_result.get("formal_candidate_data_passed")):
        raise ValueError("隔离代理诊断只允许在正式数据未通过时运行")

    daily = pd.read_parquet(
        project_path(config["paths"]["daily_outcome_blind_features"])
    )
    if daily["market_price_read"].fillna(False).any() or daily[
        "future_return_read"
    ].fillna(False).any():
        raise ValueError("结果盲日面板含结果读取标记")
    market = pd.read_parquet(project_path(config["inputs"]["etf_daily"]))
    dividends = _normalize_dividends(
        pd.read_csv(project_path(config["inputs"]["dividends"]))
    )
    cutoff = pd.Timestamp(config["protocol"]["evidence_cutoff"])
    market = market.loc[
        pd.to_datetime(market["date"], errors="raise").dt.normalize().le(cutoff)
    ].copy()
    model_contract = config["diagnostic_proxy_model"]
    prepared = prepare_market_features_and_labels(
        market,
        dividends,
        horizon=int(model_contract["future_label_horizon_trading_days"]),
    )
    daily["date"] = pd.to_datetime(daily["date"], errors="raise").dt.normalize()
    frame = prepared.merge(daily, on="date", how="inner", validate="one_to_one")
    frame = frame.loc[
        frame["date"].ge(pd.Timestamp(config["protocol"]["diagnostic_training_start"]))
        & frame["date"].le(cutoff)
        & frame["future_10d_label_end_date"].le(cutoff)
    ].copy()

    baseline_features = list(model_contract["price_baseline_features"])
    earnings_features = list(model_contract["earnings_increment_features"])
    augmented_features = baseline_features + earnings_features
    predictions, refits = anchored_quarterly_walk_forward(
        frame,
        baseline_features=baseline_features,
        augmented_features=augmented_features,
        oos_start=pd.Timestamp(config["protocol"]["oos_start"]),
        initial_training_observations=int(model_contract["initial_training_observations"]),
        calibration_observations=int(model_contract["calibration_observations"]),
        random_seed=int(model_contract["random_seed"]),
        threshold_contract=model_contract["threshold_grid"],
        prediction_gates=model_contract["prediction_gates"],
    )
    evaluation = evaluate_predictions(
        predictions,
        gates=model_contract["prediction_gates"],
        bootstrap=model_contract["bootstrap"],
    )

    prediction_columns = [
        "date",
        "future_10d_label_end_date",
        "future_10d_open_to_open_total_return",
        "target_10d_negative",
        "next_day_open_to_open_total_return",
        *augmented_features,
        "baseline_probability",
        "augmented_probability",
        "threshold",
        "high_risk_prediction",
        "refit_quarter",
        "calibration_threshold_status",
    ]
    prediction_output = predictions[prediction_columns].copy()
    prediction_output["input_reliability"] = (
        "UNRELIABLE_FORMAL_DATA_GATES_FAILED_DIAGNOSTIC_ONLY"
    )
    prediction_output["portfolio_position"] = pd.NA
    prediction_output["paper_signal_allowed"] = False
    prediction_output["shadow_signal_allowed"] = False
    prediction_output["live_trading_authorized"] = False
    predictions_path = project_path(config["paths"]["prediction_rows"])
    refits_path = project_path(config["paths"]["refit_rows"])
    atomic_parquet(predictions_path, prediction_output)
    atomic_parquet(refits_path, refits)

    result = {
        "diagnostic_id": config["protocol"]["diagnostic_id"],
        "formal_candidate_id": config["protocol"]["formal_candidate_id"],
        "status": config["reliability_quarantine"]["result_status"],
        "authoritative_candidate_status": data_result[
            "authoritative_candidate_status"
        ],
        "created_at": datetime.now(TIMEZONE).isoformat(),
        "diagnostic_manifest": {
            "path": config["paths"]["diagnostic_manifest"],
            "sha256": sha256_file(diagnostic_manifest_path),
            "status": manifest["status"],
        },
        "scope": {
            "training_start": config["protocol"]["diagnostic_training_start"],
            "oos_start": config["protocol"]["oos_start"],
            "oos_end": (
                prediction_output["date"].max().date().isoformat()
                if not prediction_output.empty
                else None
            ),
            "price_asset": "510300.SH",
            "future_label_horizon_trading_days": model_contract[
                "future_label_horizon_trading_days"
            ],
            "initial_training_observations": model_contract[
                "initial_training_observations"
            ],
            "calibration_observations": model_contract[
                "calibration_observations"
            ],
            "label_embargo_trading_days": model_contract[
                "label_embargo_trading_days"
            ],
            "refit_frequency": model_contract["refit_frequency"],
            "test_mode": model_contract["test_mode"],
        },
        "input_data_gate": {
            "formal_candidate_data_passed": False,
            "member_valid_weight_fact_count": data_result["prevalence"][
                "member_valid_weight_fact_count"
            ],
            "member_valid_weight_fact_count_2021_plus": data_result["prevalence"][
                "member_valid_weight_fact_count_2021_plus"
            ],
            "a3_success_rate": data_result["facts_gate"]["a3_success_rate"],
            "human_review_pairs": data_result["facts_gate"][
                "actual_completed_human_review_pairs"
            ],
            "weight_revision_vintage_proven": False,
            "reliability_labels": config["reliability_quarantine"][
                "required_labels"
            ],
        },
        "refit_summary": {
            "rows": int(len(refits)),
            "completed_refits": int(
                refits.get("status", pd.Series(dtype=str))
                .eq("PASS_REFIT_DIAGNOSTIC_ONLY")
                .sum()
            ),
            "threshold_calibration_passes": int(
                refits.get("threshold_status", pd.Series(dtype=str))
                .eq("PASS_CALIBRATION_RECALL_FPR_CONSTRAINT")
                .sum()
            ),
        },
        "evaluation": evaluation,
        "formal_interpretation": (
            "NOT_ADMISSIBLE_REGARDLESS_OF_OBSERVED_DIAGNOSTIC_METRICS"
        ),
        "prediction_gate_can_authorize_portfolio": False,
        "portfolio_evaluation_performed": False,
        "return_evaluation": "DIAGNOSTIC_PREDICTION_ONLY_NO_PORTFOLIO",
        "model_action": "ABSTAIN",
        "model_position_target": "UNSET",
        "existing_holdings_override_allowed": False,
        "paper_signal_allowed": False,
        "shadow_signal_allowed": False,
        "order_generation_allowed": False,
        "broker_connection_allowed": False,
        "live_trading_authorized": False,
        "artifacts": {
            "predictions": {
                "path": config["paths"]["prediction_rows"],
                "sha256": sha256_file(predictions_path),
                "rows": int(len(prediction_output)),
            },
            "refits": {
                "path": config["paths"]["refit_rows"],
                "sha256": sha256_file(refits_path),
                "rows": int(len(refits)),
            },
        },
    }
    result_path = project_path(config["paths"]["diagnostic_result_json"])
    markdown_path = project_path(config["paths"]["diagnostic_result_markdown"])
    atomic_json(result_path, result)
    atomic_text(markdown_path, render_diagnostic_markdown(result))
    return result
