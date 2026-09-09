"""510300 成分脆弱性 DSV5 增量检验 V1。

模块严格分离“无标签 G0/G1 普查”和“一次性标签/模型执行”。前者只形成
日期、可用性与固定时代；后者必须在协议、实现及无标签审计均已提交后才能创建
一次性 claim 并读取 DSV5。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize


MODEL_ID = "510300_CONSTITUENT_FRAGILITY_DSV5_INCREMENT_V1"
VIEW_ALLOWED = "VIEW_ALLOWED"
NO_VIEW = "NO_VIEW"

PARENT_F_COMPONENTS = (
    "f1_breadth_risk_percentile",
    "f2_leadership_risk_percentile",
    "f3_comovement_risk_percentile",
)
PARENT_TC_COMPONENTS = (
    "t1_breadth_drop_risk_percentile",
    "t2_tail_diffusion_risk_percentile",
    "t3_comovement_accel_risk_percentile",
)
B1_FEATURES = ("LOG_RV20", "NEG5", "DD20")
B2_FEATURES = (*B1_FEATURES, "T_C", "T_C_X_F")


class DSV5ProtocolError(RuntimeError):
    """冻结合同、数据或顺序被违反。"""


@dataclass(frozen=True)
class FittedQLikeModel:
    """一个以训练样本标准化、斜率非负的指数链接 QLIKE Ridge 模型。"""

    feature_names: tuple[str, ...]
    intercept: float
    coefficients: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    objective_value: float
    iterations: int

    def predict(self, frame: pd.DataFrame, *, epsilon: float) -> np.ndarray:
        matrix = frame.loc[:, list(self.feature_names)].to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise DSV5ProtocolError("预测特征包含非有限值")
        standardized = (matrix - self.means) / self.scales
        eta = self.intercept + standardized @ self.coefficients
        prediction = np.exp(np.clip(eta, -60.0, 60.0))
        return np.maximum(prediction, epsilon)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    normalized = _to_builtin(payload)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any], hash_field: str) -> str:
    content = {key: value for key, value in payload.items() if key != hash_field}
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_if_absent_or_identical(path: Path, content: bytes) -> None:
    """不覆盖不同内容；相同内容保持文件原样。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise DSV5ProtocolError(f"冻结输出已存在且内容不同：{path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    write_bytes_if_absent_or_identical(path, canonical_json_bytes(payload))


def write_text_once(path: Path, content: str) -> None:
    normalized = content.replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    write_bytes_if_absent_or_identical(path, normalized.encode("utf-8"))


def write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DSV5ProtocolError(f"一次性 Parquet 输出已经存在：{path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DSV5ProtocolError("协议 YAML 顶层必须是映射")
    if payload.get("protocol", {}).get("model_id") != MODEL_ID:
        raise DSV5ProtocolError("MODEL_ID 与冻结程序不一致")
    return payload


def resolve_path(project_root: Path, logical_path: str) -> Path:
    path = Path(logical_path)
    return path if path.is_absolute() else project_root / path


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DSV5ProtocolError(f"{label}缺少必需列：{missing}")


def _normalize_dates(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    if parsed.isna().any():
        raise DSV5ProtocolError(f"{label}存在不可解析日期")
    return parsed


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicate = frame.duplicated(list(columns), keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, list(columns)].head(5).to_dict("records")
        raise DSV5ProtocolError(f"{label}存在重复键：{examples}")


def _require_finite_positive(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise DSV5ProtocolError(f"{label}存在缺失或非有限值")
    if numeric.le(0.0).any():
        raise DSV5ProtocolError(f"{label}必须全部大于0")
    return numeric


def assert_file_identity(project_root: Path, specification: Mapping[str, Any]) -> dict[str, Any]:
    path = resolve_path(project_root, str(specification["path"]))
    if not path.is_file():
        raise DSV5ProtocolError(f"缺少冻结输入：{path}")
    size = path.stat().st_size
    expected_size = int(specification["bytes"])
    if size != expected_size:
        raise DSV5ProtocolError(
            f"冻结输入字节数变化：{specification['path']}，预期{expected_size}，实际{size}"
        )
    digest = sha256_file(path)
    expected_digest = str(specification["sha256"]).lower()
    if digest != expected_digest:
        raise DSV5ProtocolError(
            f"冻结输入哈希变化：{specification['path']}，预期{expected_digest}，实际{digest}"
        )
    return {"path": str(specification["path"]), "bytes": size, "sha256": digest}


def prepare_dividends(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    required = [
        "symbol",
        "record_date",
        "ex_date",
        "payment_date",
        "cash_dividend_per_share",
        "source",
    ]
    _require_columns(frame, required, "510300分红")
    result = frame.loc[:, required].copy()
    result["symbol"] = result["symbol"].astype("string").str.strip().str.upper()
    if not result["symbol"].eq(symbol).all():
        raise DSV5ProtocolError("分红账本包含510300.SH之外的证券")
    for column in ("record_date", "ex_date", "payment_date"):
        result[column] = _normalize_dates(result[column], f"分红{column}")
    result["cash_dividend_per_share"] = _require_finite_positive(
        result["cash_dividend_per_share"], "每份现金分红"
    )
    if result["record_date"].gt(result["ex_date"]).any():
        raise DSV5ProtocolError("存在登记日晚于除息日的分红")
    if result["ex_date"].gt(result["payment_date"]).any():
        raise DSV5ProtocolError("存在除息日晚于支付日的分红")
    _require_unique(result, ["symbol", "ex_date"], "510300分红")
    return result.sort_values(["ex_date", "record_date"], kind="stable").reset_index(drop=True)


def prepare_etf_daily(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    _require_columns(frame, ["date", "open", "close", "symbol"], "510300日线")
    result = frame.loc[:, ["date", "open", "close", "symbol"]].copy()
    result["date"] = _normalize_dates(result["date"], "510300交易日")
    result["symbol"] = result["symbol"].astype("string").str.strip().str.upper()
    if not result["symbol"].eq(symbol).all():
        raise DSV5ProtocolError("日线包含510300.SH之外的证券")
    result["open"] = _require_finite_positive(result["open"], "510300官方开盘价")
    result["close"] = _require_finite_positive(result["close"], "510300未复权收盘价")
    _require_unique(result, ["date"], "510300日线")
    result = result.sort_values("date", kind="stable").reset_index(drop=True)
    if not result["date"].is_monotonic_increasing:
        raise DSV5ProtocolError("510300交易日没有严格递增")
    return result


def build_b1_daily_features(
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    *,
    annualization_days: int,
) -> pd.DataFrame:
    """只使用 t 日及以前信息构造冻结 B1；不构造未来标签。"""

    frame = etf_daily.copy()
    cash_by_ex_date = dividends.groupby("ex_date")["cash_dividend_per_share"].sum()
    frame["cash_dividend_on_ex_date"] = (
        frame["date"].map(cash_by_ex_date).fillna(0.0).astype(float)
    )
    frame["previous_close"] = frame["close"].shift(1)
    frame["daily_total_return"] = (
        (frame["close"] + frame["cash_dividend_on_ex_date"])
        / frame["previous_close"]
        - 1.0
    )
    invalid = frame["daily_total_return"].le(-1.0) & frame["daily_total_return"].notna()
    if invalid.any():
        raise DSV5ProtocolError("510300日总收益小于或等于-100%")
    frame["log_total_return"] = np.log1p(frame["daily_total_return"])
    squared = frame["log_total_return"].pow(2)
    rv20 = np.sqrt(
        float(annualization_days)
        / 20.0
        * squared.rolling(20, min_periods=20).sum()
    )
    frame["LOG_RV20"] = np.log(rv20.where(rv20.gt(0.0)))
    sum5 = frame["log_total_return"].rolling(5, min_periods=5).sum()
    frame["NEG5"] = (-sum5).clip(lower=0.0)

    gross = (1.0 + frame["daily_total_return"].fillna(0.0)).astype(float)
    frame["total_wealth_index"] = gross.cumprod()
    rolling_peak = frame["total_wealth_index"].rolling(20, min_periods=20).max()
    frame["DD20"] = (
        1.0 - frame["total_wealth_index"] / rolling_peak
    ).clip(lower=0.0)
    frame["B1_VALID"] = frame.loc[:, list(B1_FEATURES)].apply(
        lambda column: np.isfinite(pd.to_numeric(column, errors="coerce"))
    ).all(axis=1)
    return frame


def prepare_parent_features(frame: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    specification = protocol["inputs"]["parent_internal_features"]
    required = list(specification["required_columns"])
    _require_columns(frame, required, "父项目F/T特征")
    result = frame.loc[:, required].copy()
    result["date"] = _normalize_dates(result["date"], "父特征日期")
    _require_unique(result, ["date"], "父特征")
    result = result.sort_values("date", kind="stable").reset_index(drop=True)

    numeric_columns = [*PARENT_F_COMPONENTS, "F", *PARENT_TC_COMPONENTS]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
        finite_or_missing = result[column].isna() | np.isfinite(result[column])
        if not finite_or_missing.all():
            raise DSV5ProtocolError(f"父特征{column}包含无穷值")
        bounded = result[column].isna() | result[column].between(0.0, 1.0)
        if not bounded.all():
            raise DSV5ProtocolError(f"父特征{column}超出[0,1]")

    f_complete = result.loc[:, list(PARENT_F_COMPONENTS)].notna().all(axis=1)
    recomputed_f = result.loc[f_complete, list(PARENT_F_COMPONENTS)].median(axis=1)
    if not np.allclose(
        result.loc[f_complete, "F"].to_numpy(dtype=float),
        recomputed_f.to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise DSV5ProtocolError("持久化F不等于三个冻结分量的中位数")

    for column in (
        "return20_coverage_ratio",
        "tail_coverage_ratio",
        "comovement_scoreable_member_ratio",
    ):
        numeric = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not (numeric.isna() | numeric.between(0.0, 1.0)).all():
            raise DSV5ProtocolError(f"覆盖率{column}超出[0,1]")
        result[column] = numeric

    result["PARENT_COMPONENTS_COMPLETE"] = result[
        ["F", *PARENT_TC_COMPONENTS]
    ].notna().all(axis=1)
    result["PARENT_VIEW_ALLOWED"] = (
        result["internal_feature_state"].astype("string").eq(VIEW_ALLOWED)
        & result["four_state_daily_coverage_state"].astype("string").eq(VIEW_ALLOWED)
        & result["PARENT_COMPONENTS_COMPLETE"]
    )
    result["T_C"] = result.loc[:, list(PARENT_TC_COMPONENTS)].median(
        axis=1, skipna=False
    )
    result["T_C_X_F"] = result["T_C"] * result["F"]
    return result


def _split_era_ids(count: int, era_count: int) -> list[str]:
    if count < 0 or era_count <= 0:
        raise DSV5ProtocolError("时代划分参数非法")
    assignments = [""] * count
    for era_index, positions in enumerate(np.array_split(np.arange(count), era_count), 1):
        for position in positions.tolist():
            assignments[int(position)] = f"ERA_{era_index}"
    return assignments


def build_origin_schedule(
    *,
    parent_features: pd.DataFrame,
    b1_daily: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """只根据交易日、历史特征状态和未来日期是否存在形成固定原点。"""

    schedule_contract = protocol["origin_schedule"]
    cutoff = pd.Timestamp(protocol["scope"]["historical_cutoff_for_features"])
    calendar = pd.DatetimeIndex(b1_daily["date"])
    position_by_date = {date: index for index, date in enumerate(calendar)}
    b1_valid_by_date = b1_daily.set_index("date")["B1_VALID"].astype(bool)
    parent_by_date = parent_features.set_index("date")

    eligible_dates: list[pd.Timestamp] = []
    for date, row in parent_by_date.iterrows():
        if date > cutoff or date not in position_by_date:
            continue
        position = position_by_date[date]
        path_exists = position + int(protocol["label"]["horizon_trading_days"]) < len(calendar)
        if bool(row["PARENT_VIEW_ALLOWED"]) and bool(b1_valid_by_date.get(date, False)) and path_exists:
            eligible_dates.append(pd.Timestamp(date))
    if not eligible_dates:
        raise DSV5ProtocolError("没有任何无标签合格原点，无法建立固定网格")

    eligible_set = set(eligible_dates)
    anchor = min(eligible_dates)
    anchor_position = position_by_date[anchor]
    horizon = int(protocol["label"]["horizon_trading_days"])
    cadence = int(schedule_contract["cadence_trading_days"])
    rows: list[dict[str, Any]] = []
    for offset in [int(value) for value in schedule_contract["registered_offsets"]]:
        start = anchor_position + offset
        for position in range(start, len(calendar), cadence):
            origin = pd.Timestamp(calendar[position])
            if origin > cutoff:
                break
            path_exists = position + horizon < len(calendar)
            reasons: list[str] = []
            parent_row = parent_by_date.loc[origin] if origin in parent_by_date.index else None
            if parent_row is None:
                reasons.append("PARENT_DATE_MISSING")
            elif not bool(parent_row["PARENT_VIEW_ALLOWED"]):
                reasons.append("PARENT_F_T_C_NO_VIEW")
            if not bool(b1_valid_by_date.get(origin, False)):
                reasons.append("B1_HISTORY_NO_VIEW")
            if not path_exists:
                reasons.append("FIVE_DAY_PATH_DATE_NOT_MATURE")
            eligible = origin in eligible_set and not reasons
            rows.append(
                {
                    "offset": offset,
                    "origin_date": origin,
                    "entry_date": pd.Timestamp(calendar[position + 1]) if position + 1 < len(calendar) else pd.NaT,
                    "horizon_end_date": pd.Timestamp(calendar[position + horizon]) if path_exists else pd.NaT,
                    "eligible": bool(eligible),
                    "no_view_reason": "" if eligible else "|".join(reasons),
                    "sample_role": "NO_VIEW",
                    "era_id": "",
                }
            )
    schedule = pd.DataFrame(rows).sort_values(
        ["offset", "origin_date"], kind="stable"
    ).reset_index(drop=True)

    training_count = int(schedule_contract["first_training_origins"])
    primary_offset = int(schedule_contract["primary_offset"])
    era_count = int(schedule_contract["evaluation_eras"])
    offset_metrics: dict[str, Any] = {}
    for offset in [int(value) for value in schedule_contract["registered_offsets"]]:
        indices = schedule.index[(schedule["offset"].eq(offset)) & schedule["eligible"]].tolist()
        train_indices = indices[:training_count]
        evaluation_indices = indices[training_count:]
        schedule.loc[train_indices, "sample_role"] = "INITIAL_TRAINING"
        schedule.loc[evaluation_indices, "sample_role"] = "PREQUENTIAL_EVALUATION"
        if offset == primary_offset:
            era_ids = _split_era_ids(len(evaluation_indices), era_count)
            for index, era_id in zip(evaluation_indices, era_ids, strict=True):
                schedule.at[index, "era_id"] = era_id
        else:
            schedule.loc[evaluation_indices, "era_id"] = "ROBUSTNESS_OFFSET"
        selected = schedule.loc[indices]
        offset_metrics[f"OFFSET_{offset}"] = {
            "eligible_origin_count": len(indices),
            "initial_training_origin_count": len(train_indices),
            "prequential_evaluation_origin_count": len(evaluation_indices),
            "first_eligible_origin": (
                selected["origin_date"].min().date().isoformat() if not selected.empty else None
            ),
            "last_eligible_origin": (
                selected["origin_date"].max().date().isoformat() if not selected.empty else None
            ),
        }

    primary = schedule.loc[(schedule["offset"].eq(primary_offset)) & schedule["eligible"]]
    primary_evaluation = primary.loc[primary["sample_role"].eq("PREQUENTIAL_EVALUATION")]
    era_counts = {
        f"ERA_{index}": int(primary_evaluation["era_id"].eq(f"ERA_{index}").sum())
        for index in range(1, era_count + 1)
    }
    minimum_total = int(schedule_contract["minimum_primary_total_origins"])
    minimum_evaluation = int(
        schedule_contract["minimum_primary_prequential_evaluation_origins"]
    )
    minimum_each_era = int(schedule_contract["minimum_origins_each_era"])
    checks = {
        "primary_total_origins": len(primary) >= minimum_total,
        "initial_training_origins": min(len(primary), training_count) == training_count,
        "prequential_evaluation_origins": len(primary_evaluation) >= minimum_evaluation,
        "three_contiguous_eras": len(era_counts) == era_count,
        "minimum_each_era": all(count >= minimum_each_era for count in era_counts.values()),
    }
    g1_passed = all(checks.values())
    metrics = {
        "anchor_date": anchor.date().isoformat(),
        "primary_offset": primary_offset,
        "offsets": offset_metrics,
        "primary_era_counts": era_counts,
        "checks": checks,
        "passed": g1_passed,
        "status": (
            "PASS_NONOVERLAPPING_ORIGIN_IDENTIFIABILITY"
            if g1_passed
            else str(schedule_contract["insufficient_state"])
        ),
    }
    return schedule, metrics


def _schedule_csv_bytes(schedule: pd.DataFrame) -> bytes:
    output = schedule.copy()
    for column in ("origin_date", "entry_date", "horizon_end_date"):
        output[column] = pd.to_datetime(output[column], errors="coerce").dt.strftime("%Y-%m-%d")
    text = output.to_csv(index=False, lineterminator="\n")
    return text.encode("utf-8")


def run_prelable_audit(
    *,
    project_root: Path,
    protocol_path: Path,
    write_outputs: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行 G0/G1；该函数不调用也不接收任何 DSV5 标签构造器。"""

    protocol = load_protocol(protocol_path)
    input_specs = protocol["inputs"]
    parent_specs = protocol["immutable_parent"]
    identities: list[dict[str, Any]] = []
    for specification in (
        parent_specs["g0_1_manifest"],
        parent_specs["g0_1_clean_replay"],
        parent_specs["g1b_terminal_status"],
        input_specs["parent_internal_features"],
        input_specs["etf_unadjusted_daily"],
        input_specs["etf_cash_dividends"],
    ):
        identities.append(assert_file_identity(project_root, specification))

    terminal_path = resolve_path(
        project_root, str(parent_specs["g1b_terminal_status"]["path"])
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    for field, expected in parent_specs["g1b_terminal_status"]["required_fields"].items():
        if terminal.get(field) != expected:
            raise DSV5ProtocolError(
                f"父项目终态字段变化：{field}，预期{expected!r}，实际{terminal.get(field)!r}"
            )

    feature_path = resolve_path(
        project_root, str(input_specs["parent_internal_features"]["path"])
    )
    etf_path = resolve_path(project_root, str(input_specs["etf_unadjusted_daily"]["path"]))
    dividend_path = resolve_path(
        project_root, str(input_specs["etf_cash_dividends"]["path"])
    )
    raw_parent = pd.read_parquet(feature_path)
    raw_etf = pd.read_parquet(etf_path)
    raw_dividends = pd.read_csv(dividend_path, encoding="utf-8-sig")
    parent = prepare_parent_features(raw_parent, protocol)
    etf = prepare_etf_daily(raw_etf, symbol=str(input_specs["etf_unadjusted_daily"]["symbol"]))
    dividends = prepare_dividends(
        raw_dividends, symbol=str(input_specs["etf_cash_dividends"]["symbol"])
    )
    b1_daily = build_b1_daily_features(
        etf,
        dividends,
        annualization_days=int(protocol["label"]["annualization_days"]),
    )
    schedule, g1 = build_origin_schedule(
        parent_features=parent,
        b1_daily=b1_daily,
        protocol=protocol,
    )
    g0_checks = {
        "exact_input_hashes": True,
        "parent_terminal_state_preserved": True,
        "unique_trading_dates": bool(etf["date"].is_unique and parent["date"].is_unique),
        "finite_positive_etf_open_close": True,
        "parent_feature_median_identity": True,
        "parent_values_reused_without_recomputation": True,
        "industry_macro_nbs_option_minute_features_excluded": True,
    }
    g0_passed = all(g0_checks.values())
    audit: dict[str, Any] = {
        "model_id": MODEL_ID,
        "protocol_version": str(protocol["protocol"]["version"]),
        "status": (
            "PASS_G0_G1_PRELABEL_ORIGIN_AUDIT"
            if g0_passed and bool(g1["passed"])
            else str(protocol["origin_schedule"]["insufficient_state"])
        ),
        "G0_DATA_CONTRACT": {"passed": g0_passed, "checks": g0_checks},
        "G1_NONOVERLAPPING_ORIGINS": g1,
        "input_identities": identities,
        "parent_feature_value_contract": "REUSED_FROZEN_WITHOUT_NEW_252_DAY_OVERLAY",
        "dsv5_values_read": False,
        "future_total_variance_values_read": False,
        "model_trained": False,
        "portfolio_return_or_sharpe_read": False,
        "portfolio_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
    }
    audit["audit_payload_sha256"] = payload_sha256(audit, "audit_payload_sha256")

    if write_outputs:
        audit_path = resolve_path(project_root, str(protocol["outputs"]["g0_g1_audit"]))
        schedule_path = resolve_path(project_root, str(protocol["outputs"]["origin_schedule"]))
        write_json_once(audit_path, audit)
        write_bytes_if_absent_or_identical(schedule_path, _schedule_csv_bytes(schedule))
    return audit, schedule, parent, b1_daily, dividends


def _run_git(project_root: Path, arguments: Sequence[str], *, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise DSV5ProtocolError(f"Git检查失败：git {' '.join(arguments)}：{detail}")
    return completed.stdout.strip()


def git_head(project_root: Path) -> str:
    return _run_git(project_root, ["rev-parse", "HEAD"])


def git_branch(project_root: Path) -> str:
    return _run_git(project_root, ["branch", "--show-current"])


def assert_paths_committed_and_clean(project_root: Path, logical_paths: Iterable[str]) -> None:
    for logical_path in logical_paths:
        normalized = str(logical_path).replace("\\", "/")
        _run_git(project_root, ["ls-files", "--error-unmatch", "--", normalized])
        working = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", normalized],
            cwd=project_root,
            check=False,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "HEAD", "--", normalized],
            cwd=project_root,
            check=False,
        )
        if working.returncode != 0 or staged.returncode != 0:
            raise DSV5ProtocolError(f"冻结范围文件未提交或工作区有变化：{normalized}")


def verify_frozen_execution_scope(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    require_prelable_outputs_committed: bool,
) -> dict[str, Any]:
    required_branch = str(protocol["freeze"]["required_branch"])
    current_branch = git_branch(project_root)
    if current_branch != required_branch:
        raise DSV5ProtocolError(
            f"分支不符合冻结合同：预期{required_branch}，实际{current_branch}"
        )
    paths = [str(path) for path in protocol["freeze"]["implementation_files"]]
    paths.extend(
        [
            str(protocol["outputs"]["lineage_audit"]),
            str(protocol["outputs"]["manifest"]),
            str(protocol["outputs"]["freeze_receipt"]),
        ]
    )
    if require_prelable_outputs_committed:
        paths.extend(
            [
                str(protocol["outputs"]["g0_g1_audit"]),
                str(protocol["outputs"]["origin_schedule"]),
            ]
        )
    assert_paths_committed_and_clean(project_root, paths)

    manifest_path = resolve_path(project_root, str(protocol["outputs"]["manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_DSV5_LABEL_VALUE_READ":
        raise DSV5ProtocolError("冻结manifest状态不允许执行")
    expected_payload_hash = payload_sha256(manifest, "manifest_payload_sha256")
    if manifest.get("manifest_payload_sha256") != expected_payload_hash:
        raise DSV5ProtocolError("冻结manifest载荷哈希不匹配")
    for item in manifest.get("implementation_files", []):
        path = resolve_path(project_root, str(item["path"]))
        if path.stat().st_size != int(item["bytes"]) or sha256_file(path) != str(item["sha256"]):
            raise DSV5ProtocolError(f"冻结后实现文件变化：{item['path']}")
    return {
        "branch": current_branch,
        "head": git_head(project_root),
        "manifest_path": str(protocol["outputs"]["manifest"]),
        "manifest_sha256": sha256_file(manifest_path),
        "scope_paths_clean_and_committed": True,
    }


def build_dsv5_label_ledger(
    *,
    schedule: pd.DataFrame,
    b1_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    """在一次性 claim 之后构造 DSV5；不产生任何仓位或组合收益。"""

    horizon = int(protocol["label"]["horizon_trading_days"])
    annualization = float(protocol["label"]["annualization_days"])
    if horizon != 5:
        raise DSV5ProtocolError("V1标签期限必须严格为5个交易日")
    daily = b1_daily.sort_values("date", kind="stable").reset_index(drop=True)
    position_by_date = {date: index for index, date in enumerate(daily["date"])}
    dividends_by_ex_date: dict[pd.Timestamp, list[tuple[pd.Timestamp, float]]] = {}
    for row in dividends.itertuples(index=False):
        dividends_by_ex_date.setdefault(pd.Timestamp(row.ex_date), []).append(
            (pd.Timestamp(row.record_date), float(row.cash_dividend_per_share))
        )

    selected = schedule.loc[schedule["eligible"]].copy()
    rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        origin = pd.Timestamp(row.origin_date)
        position = position_by_date.get(origin)
        if position is None or position + horizon >= len(daily):
            raise DSV5ProtocolError(f"原点{origin.date()}的5日标签尚未成熟")
        entry_position = position + 1
        entry_date = pd.Timestamp(daily.at[entry_position, "date"])
        entry_open = float(daily.at[entry_position, "open"])
        entry_close = float(daily.at[entry_position, "close"])
        log_returns = [math.log(entry_close / entry_open)]
        entitled_cash_total = 0.0
        for step in range(2, horizon + 1):
            current_position = position + step
            current_date = pd.Timestamp(daily.at[current_position, "date"])
            current_close = float(daily.at[current_position, "close"])
            previous_close = float(daily.at[current_position - 1, "close"])
            entitled_cash = sum(
                amount
                for record_date, amount in dividends_by_ex_date.get(current_date, [])
                if record_date >= entry_date
            )
            entitled_cash_total += entitled_cash
            gross = (current_close + entitled_cash) / previous_close
            if not math.isfinite(gross) or gross <= 0.0:
                raise DSV5ProtocolError(f"原点{origin.date()}的总财富收益非法")
            log_returns.append(math.log(gross))
        vector = np.asarray(log_returns, dtype=float)
        if len(vector) != horizon or not np.isfinite(vector).all():
            raise DSV5ProtocolError(f"原点{origin.date()}的5日收益向量非法")
        dsv5 = annualization / horizon * float(np.square(np.minimum(vector, 0.0)).sum())
        v5 = annualization / horizon * float(np.square(vector).sum())
        rows.append(
            {
                "offset": int(row.offset),
                "origin_date": origin,
                "entry_date": entry_date,
                "horizon_end_date": pd.Timestamp(daily.at[position + horizon, "date"]),
                "sample_role": str(row.sample_role),
                "era_id": str(row.era_id),
                "DSV5": dsv5,
                "V5_TOTAL_VARIANCE_DIAGNOSTIC": v5,
                "negative_return_day_count": int(np.count_nonzero(vector < 0.0)),
                "entitled_cash_dividend_per_share_in_horizon": entitled_cash_total,
            }
        )
    ledger = pd.DataFrame(rows).sort_values(
        ["offset", "origin_date"], kind="stable"
    ).reset_index(drop=True)
    if ledger.empty or ledger["DSV5"].isna().any() or ledger["DSV5"].lt(0.0).any():
        raise DSV5ProtocolError("DSV5标签账本为空或包含非法值")
    return ledger


def assemble_model_panel(
    *,
    schedule: pd.DataFrame,
    parent_features: pd.DataFrame,
    b1_daily: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    selected = schedule.loc[schedule["eligible"]].copy()
    selected = selected[
        [
            "offset",
            "origin_date",
            "entry_date",
            "horizon_end_date",
            "sample_role",
            "era_id",
        ]
    ]
    b1 = b1_daily[["date", *B1_FEATURES]].rename(columns={"date": "origin_date"})
    parent = parent_features[["date", "F", "T_C", "T_C_X_F"]].rename(
        columns={"date": "origin_date"}
    )
    target = labels[
        ["offset", "origin_date", "DSV5", "V5_TOTAL_VARIANCE_DIAGNOSTIC"]
    ]
    panel = (
        selected.merge(b1, on="origin_date", how="left", validate="many_to_one")
        .merge(parent, on="origin_date", how="left", validate="many_to_one")
        .merge(target, on=["offset", "origin_date"], how="left", validate="one_to_one")
        .sort_values(["offset", "origin_date"], kind="stable")
        .reset_index(drop=True)
    )
    required_numeric = [*B2_FEATURES, "F", "DSV5"]
    numeric = panel.loc[:, required_numeric].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DSV5ProtocolError("模型面板存在缺失或非有限值")
    if panel["DSV5"].lt(0.0).any():
        raise DSV5ProtocolError("DSV5不得为负")
    return panel


def fit_nonnegative_qlike_ridge(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    target_name: str,
    l2_lambda: float,
    epsilon: float,
) -> FittedQLikeModel:
    names = tuple(str(name) for name in feature_names)
    matrix = frame.loc[:, list(names)].to_numpy(dtype=float)
    target = frame[target_name].to_numpy(dtype=float)
    if len(target) == 0 or not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise DSV5ProtocolError("QLIKE训练样本为空或包含非有限值")
    if (target < 0.0).any():
        raise DSV5ProtocolError("QLIKE目标不得为负")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    standardized = (matrix - means) / scales
    augmented = np.column_stack([np.ones(len(standardized)), standardized])
    actual = np.maximum(target + epsilon, epsilon)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = augmented @ parameters
        inverse_prediction = np.exp(np.clip(-eta, -60.0, 60.0))
        ratio = actual * inverse_prediction
        loss = float(np.mean(ratio + eta))
        penalty = float(l2_lambda) * float(np.square(parameters[1:]).sum())
        gradient_eta = 1.0 - ratio
        gradient = augmented.T @ gradient_eta / len(actual)
        gradient[1:] += 2.0 * float(l2_lambda) * parameters[1:]
        return loss + penalty, gradient

    initial = np.zeros(1 + len(names), dtype=float)
    initial[0] = math.log(max(float(actual.mean()), epsilon))
    optimized = minimize(
        fun=lambda parameters: objective(parameters)[0],
        x0=initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        bounds=[(None, None), *[(0.0, None) for _ in names]],
        options={"maxiter": 2000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if not optimized.success or not np.isfinite(optimized.x).all():
        raise DSV5ProtocolError(
            f"QLIKE优化未收敛：status={optimized.status} message={optimized.message}"
        )
    coefficients = np.maximum(np.asarray(optimized.x[1:], dtype=float), 0.0)
    return FittedQLikeModel(
        feature_names=names,
        intercept=float(optimized.x[0]),
        coefficients=coefficients,
        means=np.asarray(means, dtype=float),
        scales=np.asarray(scales, dtype=float),
        objective_value=float(optimized.fun),
        iterations=int(optimized.nit),
    )


def _coefficient_rows(
    *,
    model_name: str,
    fitted: FittedQLikeModel,
    offset: int,
    refit_id: str,
    origin_date: pd.Timestamp,
    training_count: int,
    maximum_training_horizon_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    common = {
        "offset": offset,
        "model": model_name,
        "refit_id": refit_id,
        "model_vintage_origin_date": origin_date,
        "training_count": training_count,
        "maximum_training_horizon_end": maximum_training_horizon_end,
        "objective_value": fitted.objective_value,
        "optimizer_iterations": fitted.iterations,
    }
    rows = [
        {
            **common,
            "feature": "INTERCEPT",
            "coefficient": fitted.intercept,
            "training_mean": np.nan,
            "training_scale": np.nan,
        }
    ]
    for name, coefficient, mean, scale in zip(
        fitted.feature_names,
        fitted.coefficients,
        fitted.means,
        fitted.scales,
        strict=True,
    ):
        rows.append(
            {
                **common,
                "feature": name,
                "coefficient": float(coefficient),
                "training_mean": float(mean),
                "training_scale": float(scale),
            }
        )
    return rows


def generate_b0_b1_predictions(
    panel: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_count = int(protocol["origin_schedule"]["first_training_origins"])
    refit_every = int(protocol["models"]["refit_every_evaluation_origins"])
    l2_lambda = float(protocol["models"]["l2_lambda"])
    epsilon = float(protocol["label"]["qlike_epsilon"])
    prediction_rows: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for offset in [int(value) for value in protocol["origin_schedule"]["registered_offsets"]]:
        data = panel.loc[panel["offset"].eq(offset)].sort_values(
            "origin_date", kind="stable"
        ).reset_index(drop=True)
        if len(data) <= training_count:
            continue
        fitted: FittedQLikeModel | None = None
        b0_prediction = math.nan
        refit_id = ""
        train_count_at_refit = 0
        max_horizon_at_refit = pd.NaT
        for position in range(training_count, len(data)):
            evaluation_index = position - training_count
            origin = pd.Timestamp(data.at[position, "origin_date"])
            if evaluation_index % refit_every == 0:
                candidates = data.iloc[:position].copy()
                mature = candidates.loc[
                    pd.to_datetime(candidates["horizon_end_date"]).le(origin)
                ].copy()
                if len(mature) < training_count:
                    raise DSV5ProtocolError(
                        f"OFFSET_{offset}在{origin.date()}只有{len(mature)}个成熟训练原点"
                    )
                refit_number = evaluation_index // refit_every + 1
                refit_id = f"OFFSET_{offset}_B1_REFIT_{refit_number:03d}"
                fitted = fit_nonnegative_qlike_ridge(
                    mature,
                    feature_names=B1_FEATURES,
                    target_name="DSV5",
                    l2_lambda=l2_lambda,
                    epsilon=epsilon,
                )
                b0_prediction = max(float(mature["DSV5"].mean()), epsilon)
                train_count_at_refit = len(mature)
                max_horizon_at_refit = pd.Timestamp(mature["horizon_end_date"].max())
                coefficients.append(
                    {
                        "offset": offset,
                        "model": "B0",
                        "refit_id": refit_id.replace("_B1_", "_B0_"),
                        "model_vintage_origin_date": origin,
                        "training_count": train_count_at_refit,
                        "maximum_training_horizon_end": max_horizon_at_refit,
                        "objective_value": np.nan,
                        "optimizer_iterations": 0,
                        "feature": "EXPANDING_MATURED_MEAN",
                        "coefficient": b0_prediction,
                        "training_mean": np.nan,
                        "training_scale": np.nan,
                    }
                )
                coefficients.extend(
                    _coefficient_rows(
                        model_name="B1",
                        fitted=fitted,
                        offset=offset,
                        refit_id=refit_id,
                        origin_date=origin,
                        training_count=train_count_at_refit,
                        maximum_training_horizon_end=max_horizon_at_refit,
                    )
                )
            if fitted is None:
                raise DSV5ProtocolError("B1模型在预测前未完成首次拟合")
            prediction_b1 = float(
                fitted.predict(data.iloc[[position]], epsilon=epsilon)[0]
            )
            prediction_rows.append(
                {
                    "offset": offset,
                    "origin_date": origin,
                    "entry_date": pd.Timestamp(data.at[position, "entry_date"]),
                    "horizon_end_date": pd.Timestamp(data.at[position, "horizon_end_date"]),
                    "era_id": str(data.at[position, "era_id"]),
                    "DSV5": float(data.at[position, "DSV5"]),
                    "predicted_B0": b0_prediction,
                    "predicted_B1": prediction_b1,
                    "b1_refit_id": refit_id,
                    "training_count_at_refit": train_count_at_refit,
                    "maximum_training_horizon_end_at_refit": max_horizon_at_refit,
                }
            )
    predictions = pd.DataFrame(prediction_rows).sort_values(
        ["offset", "origin_date"], kind="stable"
    ).reset_index(drop=True)
    coefficient_frame = pd.DataFrame(coefficients).sort_values(
        ["offset", "model_vintage_origin_date", "model", "feature"], kind="stable"
    ).reset_index(drop=True)
    return predictions, coefficient_frame


def generate_b2_predictions(
    panel: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_count = int(protocol["origin_schedule"]["first_training_origins"])
    refit_every = int(protocol["models"]["refit_every_evaluation_origins"])
    l2_lambda = float(protocol["models"]["l2_lambda"])
    epsilon = float(protocol["label"]["qlike_epsilon"])
    prediction_rows: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    for offset in [int(value) for value in protocol["origin_schedule"]["registered_offsets"]]:
        data = panel.loc[panel["offset"].eq(offset)].sort_values(
            "origin_date", kind="stable"
        ).reset_index(drop=True)
        if len(data) <= training_count:
            continue
        fitted: FittedQLikeModel | None = None
        refit_id = ""
        for position in range(training_count, len(data)):
            evaluation_index = position - training_count
            origin = pd.Timestamp(data.at[position, "origin_date"])
            if evaluation_index % refit_every == 0:
                candidates = data.iloc[:position].copy()
                mature = candidates.loc[
                    pd.to_datetime(candidates["horizon_end_date"]).le(origin)
                ].copy()
                if len(mature) < training_count:
                    raise DSV5ProtocolError(
                        f"OFFSET_{offset}在{origin.date()}没有足够成熟B2训练原点"
                    )
                refit_number = evaluation_index // refit_every + 1
                refit_id = f"OFFSET_{offset}_B2_REFIT_{refit_number:03d}"
                fitted = fit_nonnegative_qlike_ridge(
                    mature,
                    feature_names=B2_FEATURES,
                    target_name="DSV5",
                    l2_lambda=l2_lambda,
                    epsilon=epsilon,
                )
                coefficients.extend(
                    _coefficient_rows(
                        model_name="B2",
                        fitted=fitted,
                        offset=offset,
                        refit_id=refit_id,
                        origin_date=origin,
                        training_count=len(mature),
                        maximum_training_horizon_end=pd.Timestamp(
                            mature["horizon_end_date"].max()
                        ),
                    )
                )
            if fitted is None:
                raise DSV5ProtocolError("B2模型在预测前未完成首次拟合")
            prediction_rows.append(
                {
                    "offset": offset,
                    "origin_date": origin,
                    "predicted_B2": float(
                        fitted.predict(data.iloc[[position]], epsilon=epsilon)[0]
                    ),
                    "b2_refit_id": refit_id,
                }
            )
    return (
        pd.DataFrame(prediction_rows).sort_values(
            ["offset", "origin_date"], kind="stable"
        ).reset_index(drop=True),
        pd.DataFrame(coefficients).sort_values(
            ["offset", "model_vintage_origin_date", "model", "feature"],
            kind="stable",
        ).reset_index(drop=True),
    )


def qlike_deviance(actual: pd.Series, predicted: pd.Series, *, epsilon: float) -> np.ndarray:
    observed = np.maximum(pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float) + epsilon, epsilon)
    forecast = np.maximum(pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float), epsilon)
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise DSV5ProtocolError("QLIKE评价包含非有限值")
    ratio = observed / forecast
    return ratio - np.log(ratio) - 1.0


def circular_block_bootstrap_lower_bound(
    differences: np.ndarray,
    *,
    block_length: int,
    repetitions: int,
    lower_quantile: float,
    random_seed: int,
) -> float:
    values = np.asarray(differences, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise DSV5ProtocolError("Bootstrap输入为空或非法")
    if block_length <= 0 or repetitions <= 0 or not 0.0 < lower_quantile < 1.0:
        raise DSV5ProtocolError("Bootstrap冻结参数非法")
    rng = np.random.default_rng(random_seed)
    block_count = math.ceil(len(values) / block_length)
    means = np.empty(repetitions, dtype=float)
    offsets = np.arange(block_length)
    for repetition in range(repetitions):
        starts = rng.integers(0, len(values), size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % len(values)).ravel()[: len(values)]
        means[repetition] = float(values[indices].mean())
    return float(np.quantile(means, lower_quantile))


def _relative_improvement(baseline_loss: np.ndarray, model_loss: np.ndarray) -> float:
    baseline_mean = float(np.mean(baseline_loss))
    if not math.isfinite(baseline_mean) or baseline_mean <= 0.0:
        raise DSV5ProtocolError("基准QLIKE均值非正，无法定义相对改善")
    return float((baseline_mean - float(np.mean(model_loss))) / baseline_mean)


def _era_improvements(
    frame: pd.DataFrame,
    *,
    baseline_column: str,
    model_column: str,
    epsilon: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    era_ids = [value for value in frame["era_id"].astype(str).unique() if value.startswith("ERA_")]
    for era_id in sorted(era_ids):
        subset = frame.loc[frame["era_id"].eq(era_id)]
        base_loss = qlike_deviance(subset["DSV5"], subset[baseline_column], epsilon=epsilon)
        model_loss = qlike_deviance(subset["DSV5"], subset[model_column], epsilon=epsilon)
        metrics[era_id] = {
            "count": len(subset),
            "qlike_baseline": float(base_loss.mean()),
            "qlike_model": float(model_loss.mean()),
            "relative_improvement": _relative_improvement(base_loss, model_loss),
            "improved": bool(float(model_loss.mean()) < float(base_loss.mean())),
        }
    return metrics


def evaluate_g2(predictions: pd.DataFrame, protocol: Mapping[str, Any]) -> dict[str, Any]:
    primary_offset = int(protocol["origin_schedule"]["primary_offset"])
    frame = predictions.loc[predictions["offset"].eq(primary_offset)].copy()
    epsilon = float(protocol["label"]["qlike_epsilon"])
    loss_b0 = qlike_deviance(frame["DSV5"], frame["predicted_B0"], epsilon=epsilon)
    loss_b1 = qlike_deviance(frame["DSV5"], frame["predicted_B1"], epsilon=epsilon)
    differences = loss_b0 - loss_b1
    bootstrap = protocol["evaluation"]["bootstrap"]
    lower = circular_block_bootstrap_lower_bound(
        differences,
        block_length=int(bootstrap["block_length_origins"]),
        repetitions=int(bootstrap["repetitions"]),
        lower_quantile=1.0 - float(bootstrap["one_sided_confidence"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    eras = _era_improvements(
        frame,
        baseline_column="predicted_B0",
        model_column="predicted_B1",
        epsilon=epsilon,
    )
    latest_era = sorted(eras)[-1]
    gate = protocol["gates"]["G2_B1_VS_B0"]
    relative = _relative_improvement(loss_b0, loss_b1)
    checks = {
        "overall_qlike_improvement_positive": relative > 0.0,
        "bootstrap_90pct_lower_positive": lower > 0.0,
        "minimum_positive_eras": sum(bool(item["improved"]) for item in eras.values())
        >= int(gate["minimum_positive_eras"]),
        "latest_era_improves": bool(eras[latest_era]["improved"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evaluation_origin_count": len(frame),
        "qlike_B0": float(loss_b0.mean()),
        "qlike_B1": float(loss_b1.mean()),
        "relative_improvement": relative,
        "mean_loss_difference_B0_minus_B1": float(differences.mean()),
        "bootstrap_90pct_lower_mean_loss_difference": lower,
        "eras": eras,
        "latest_era": latest_era,
    }


def _risk_quintile_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(["predicted_B2", "origin_date"], kind="stable")
    groups = np.array_split(np.arange(len(ordered)), 5)
    means = [float(ordered.iloc[group]["DSV5"].mean()) for group in groups]
    denominator = max(means[0], 1.0e-12)
    return {
        "actual_dsv5_means_low_to_high": means,
        "highest_to_lowest_ratio": float(means[-1] / denominator),
    }


def evaluate_g3(predictions: pd.DataFrame, protocol: Mapping[str, Any]) -> dict[str, Any]:
    primary_offset = int(protocol["origin_schedule"]["primary_offset"])
    epsilon = float(protocol["label"]["qlike_epsilon"])
    gate = protocol["gates"]["G3_B2_VS_B1"]
    offset_metrics: dict[str, Any] = {}
    for offset in [int(value) for value in protocol["origin_schedule"]["registered_offsets"]]:
        subset = predictions.loc[predictions["offset"].eq(offset)].copy()
        if subset.empty or subset["predicted_B2"].isna().any():
            offset_metrics[f"OFFSET_{offset}"] = {
                "count": len(subset),
                "state": "NO_VIEW_INSUFFICIENT_OR_MISSING_B2_PREDICTIONS",
                "relative_improvement": None,
                "improved": False,
            }
            continue
        loss_b1 = qlike_deviance(subset["DSV5"], subset["predicted_B1"], epsilon=epsilon)
        loss_b2 = qlike_deviance(subset["DSV5"], subset["predicted_B2"], epsilon=epsilon)
        improvement = _relative_improvement(loss_b1, loss_b2)
        offset_metrics[f"OFFSET_{offset}"] = {
            "count": len(subset),
            "state": "EVALUATED",
            "qlike_B1": float(loss_b1.mean()),
            "qlike_B2": float(loss_b2.mean()),
            "relative_improvement": improvement,
            "improved": bool(float(loss_b2.mean()) < float(loss_b1.mean())),
        }

    frame = predictions.loc[predictions["offset"].eq(primary_offset)].copy()
    loss_b1 = qlike_deviance(frame["DSV5"], frame["predicted_B1"], epsilon=epsilon)
    loss_b2 = qlike_deviance(frame["DSV5"], frame["predicted_B2"], epsilon=epsilon)
    differences = loss_b1 - loss_b2
    bootstrap = protocol["evaluation"]["bootstrap"]
    lower = circular_block_bootstrap_lower_bound(
        differences,
        block_length=int(bootstrap["block_length_origins"]),
        repetitions=int(bootstrap["repetitions"]),
        lower_quantile=1.0 - float(bootstrap["one_sided_confidence"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    log_actual = np.log(frame["DSV5"].to_numpy(dtype=float) + epsilon)
    mse_b1 = float(np.square(log_actual - np.log(frame["predicted_B1"].to_numpy(dtype=float))).mean())
    mse_b2 = float(np.square(log_actual - np.log(frame["predicted_B2"].to_numpy(dtype=float))).mean())
    eras = _era_improvements(
        frame,
        baseline_column="predicted_B1",
        model_column="predicted_B2",
        epsilon=epsilon,
    )
    latest_era = sorted(eras)[-1]
    quintiles = _risk_quintile_diagnostic(frame)
    mean_ratio = float(frame["predicted_B2"].mean() / frame["DSV5"].mean())
    relative = _relative_improvement(loss_b1, loss_b2)
    positive_offsets = sum(bool(item["improved"]) for item in offset_metrics.values())
    checks = {
        "overall_qlike_improvement_at_least_2pct": relative
        >= float(gate["overall_qlike_relative_improvement_minimum"]),
        "log_dsv5_mse_improves": mse_b2 < mse_b1,
        "bootstrap_90pct_lower_positive": lower > 0.0,
        "minimum_positive_eras": sum(bool(item["improved"]) for item in eras.values())
        >= int(gate["minimum_positive_eras"]),
        "latest_era_improves": bool(eras[latest_era]["improved"]),
        "minimum_positive_registered_offsets": positive_offsets
        >= int(gate["minimum_positive_registered_offsets"]),
        "highest_to_lowest_actual_dsv5_ratio": quintiles["highest_to_lowest_ratio"]
        >= float(gate["highest_to_lowest_actual_dsv5_quintile_ratio_minimum"]),
        "mean_prediction_to_actual_ratio": float(
            gate["mean_predicted_to_actual_ratio_minimum"]
        )
        <= mean_ratio
        <= float(gate["mean_predicted_to_actual_ratio_maximum"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "evaluation_origin_count": len(frame),
        "qlike_B1": float(loss_b1.mean()),
        "qlike_B2": float(loss_b2.mean()),
        "relative_improvement": relative,
        "mean_loss_difference_B1_minus_B2": float(differences.mean()),
        "bootstrap_90pct_lower_mean_loss_difference": lower,
        "log_dsv5_mse_B1": mse_b1,
        "log_dsv5_mse_B2": mse_b2,
        "eras": eras,
        "latest_era": latest_era,
        "offsets": offset_metrics,
        "positive_offset_count": positive_offsets,
        "risk_quintiles": quintiles,
        "mean_predicted_to_actual_ratio": mean_ratio,
    }


def _markdown_result(result: Mapping[str, Any]) -> str:
    lines = [
        "# 510300 成分脆弱性 DSV5 增量检验 V1",
        "",
        f"- 最终状态：`{result['final_state']}`",
        f"- G0：`{result['gates']['G0_DATA_CONTRACT']['status']}`",
        f"- G1：`{result['gates']['G1_NONOVERLAPPING_ORIGINS']['status']}`",
        f"- G2（B1 对 B0）：`{'PASS' if result['gates']['G2_B1_VS_B0']['passed'] else 'FAIL'}`",
        f"- G3（B2 对 B1）：`{result['gates']['G3_B2_VS_B1']['status']}`",
        "- 证据类别：`RESEARCH_OBSERVED_PREQUENTIAL`，不是严格未观察样本外。",
        "- 仓位、净值、组合收益、Sharpe、Paper/Shadow、订单和实盘：均未生成。",
        "",
        "## 无标签原点门",
        "",
    ]
    g1 = result["gates"]["G1_NONOVERLAPPING_ORIGINS"]
    primary = g1["offsets"]["OFFSET_0"]
    lines.extend(
        [
            f"- 主偏移合格原点：`{primary['eligible_origin_count']}`",
            f"- 首次训练/严格前序评价：`{primary['initial_training_origin_count']}` / "
            f"`{primary['prequential_evaluation_origin_count']}`",
            "- 三个时代："
            + " / ".join(f"`{key}={value}`" for key, value in g1["primary_era_counts"].items()),
            "",
            "## G2：B1 对 B0",
            "",
        ]
    )
    g2 = result["gates"]["G2_B1_VS_B0"]
    lines.extend(
        [
            f"- 总体 QLIKE 相对改善：`{g2['relative_improvement']:.6%}`",
            f"- 13 原点块 Bootstrap 单侧 90% 下界："
            f"`{g2['bootstrap_90pct_lower_mean_loss_difference']:.10f}`",
            f"- 时代通过数：`{sum(bool(item['improved']) for item in g2['eras'].values())}/3`",
            f"- 最新时代改善：`{str(bool(g2['checks']['latest_era_improves'])).lower()}`",
            "",
        ]
    )
    g3 = result["gates"]["G3_B2_VS_B1"]
    if g3.get("status") == "NOT_RUN_BLOCKED_BY_G2":
        lines.extend(
            [
                "## G3：未运行",
                "",
                "B1 未通过冻结 G2，因此协议禁止拟合 B2。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## G3：B2 对 B1",
                "",
                f"- 总体 QLIKE 相对改善：`{g3['relative_improvement']:.6%}`（门槛 2%）",
                f"- log(DSV5+1e-8) MSE：B1=`{g3['log_dsv5_mse_B1']:.8f}`，"
                f"B2=`{g3['log_dsv5_mse_B2']:.8f}`",
                f"- Bootstrap 单侧 90% 下界："
                f"`{g3['bootstrap_90pct_lower_mean_loss_difference']:.10f}`",
                f"- 改善偏移：`{g3['positive_offset_count']}/5`",
                f"- 最高/最低预测风险五分位实际 DSV5 比："
                f"`{g3['risk_quintiles']['highest_to_lowest_ratio']:.6f}`",
                f"- 平均预测/平均实际：`{g3['mean_predicted_to_actual_ratio']:.6f}`",
                "",
                "### G3 原子门",
                "",
            ]
        )
        for key, passed in g3["checks"].items():
            lines.append(f"- `{key}`：`{'PASS' if passed else 'FAIL'}`")
        lines.append("")
    lines.extend(
        [
            "## 裁决",
            "",
            str(result["adjudication"]),
            "",
            "`POSITION_IMPACT=0`；本结果不得转换为当前仓位或订单。",
        ]
    )
    return "\n".join(lines) + "\n"


def _output_identity(project_root: Path, logical_path: str) -> dict[str, Any]:
    path = resolve_path(project_root, logical_path)
    return {
        "path": logical_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def create_one_shot_claim(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = protocol["outputs"]
    prohibited_existing = [
        outputs["label_ledger"],
        outputs["predictions"],
        outputs["coefficients"],
        outputs["result_json"],
        outputs["result_markdown"],
        outputs["execution_receipt"],
        outputs["final_status"],
    ]
    existing = [str(path) for path in prohibited_existing if resolve_path(project_root, str(path)).exists()]
    if existing:
        raise DSV5ProtocolError(f"一次性结果路径已存在，拒绝覆盖：{existing}")

    audit_path = resolve_path(project_root, str(outputs["g0_g1_audit"]))
    schedule_path = resolve_path(project_root, str(outputs["origin_schedule"]))
    claim: dict[str, Any] = {
        "model_id": MODEL_ID,
        "claim_state": "CLAIMED_BEFORE_FIRST_DSV5_VALUE_READ",
        "claimed_at": datetime.now().astimezone().isoformat(),
        "branch": str(scope["branch"]),
        "prelabel_lock_commit": str(scope["head"]),
        "freeze_manifest_sha256": str(scope["manifest_sha256"]),
        "g0_g1_audit": _output_identity(
            project_root, str(outputs["g0_g1_audit"])
        ),
        "origin_schedule": _output_identity(
            project_root, str(outputs["origin_schedule"])
        ),
        "dsv5_values_read_before_claim": False,
        "portfolio_return_or_sharpe_read": False,
        "position_impact": 0,
    }
    claim["claim_payload_sha256"] = payload_sha256(claim, "claim_payload_sha256")
    claim_path = resolve_path(project_root, str(outputs["one_shot_claim"]))
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(claim)
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise DSV5ProtocolError("一次性claim已经存在，本V1不得重跑") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return claim


def run_full_execution(
    *,
    project_root: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    scope = verify_frozen_execution_scope(
        project_root=project_root,
        protocol=protocol,
        require_prelable_outputs_committed=True,
    )
    audit, schedule, parent, b1_daily, dividends = run_prelable_audit(
        project_root=project_root,
        protocol_path=protocol_path,
        write_outputs=True,
    )
    if audit["status"] != "PASS_G0_G1_PRELABEL_ORIGIN_AUDIT":
        raise DSV5ProtocolError(
            "G0/G1未通过；协议禁止创建claim、读取DSV5或拟合模型"
        )
    claim = create_one_shot_claim(
        project_root=project_root,
        protocol=protocol,
        scope=scope,
    )
    labels = build_dsv5_label_ledger(
        schedule=schedule,
        b1_daily=b1_daily,
        dividends=dividends,
        protocol=protocol,
    )
    panel = assemble_model_panel(
        schedule=schedule,
        parent_features=parent,
        b1_daily=b1_daily,
        labels=labels,
    )
    predictions, coefficients = generate_b0_b1_predictions(panel, protocol=protocol)
    g2 = evaluate_g2(predictions, protocol)
    b2_was_run = False
    if bool(g2["passed"]):
        b2_predictions, b2_coefficients = generate_b2_predictions(panel, protocol=protocol)
        predictions = predictions.merge(
            b2_predictions,
            on=["offset", "origin_date"],
            how="left",
            validate="one_to_one",
        )
        coefficients = pd.concat([coefficients, b2_coefficients], ignore_index=True).sort_values(
            ["offset", "model_vintage_origin_date", "model", "feature"], kind="stable"
        ).reset_index(drop=True)
        g3: dict[str, Any] = evaluate_g3(predictions, protocol)
        g3["status"] = "PASS" if bool(g3["passed"]) else "FAIL"
        b2_was_run = True
    else:
        predictions["predicted_B2"] = np.nan
        predictions["b2_refit_id"] = "NOT_RUN_BLOCKED_BY_G2"
        g3 = {
            "passed": False,
            "status": "NOT_RUN_BLOCKED_BY_G2",
            "reason": "B1_FAILED_FROZEN_G2",
        }

    if not bool(g2["passed"]):
        final_state = str(protocol["gates"]["G2_B1_VS_B0"]["failure_state"])
        adjudication = (
            "510300自身价格风险B1未能稳定优于扩展均值B0；本风险预测家族在G2冻结关闭，"
            "B2未运行，组合评价不允许。"
        )
    elif not bool(g3["passed"]):
        final_state = str(protocol["gates"]["G3_B2_VS_B1"]["failure_state"])
        adjudication = (
            "B1通过，但成分脆弱性B2未满足全部增量门；该成分增量家族冻结拒绝，"
            "不得加入宏观、NBS或其他特征救援，组合评价不允许。"
        )
    else:
        final_state = "PASS_FROZEN_CONSTITUENT_FRAGILITY_DSV5_INCREMENT"
        adjudication = (
            "B2通过全部冻结预测增量门；这只保留风险测量模块。下一步仅可另行冻结"
            "510300_WEEKLY_DSV5_RISK_BUDGET_POLICY_V1，本次仍不得读取组合收益或Sharpe。"
        )

    result: dict[str, Any] = {
        "model_id": MODEL_ID,
        "protocol_version": str(protocol["protocol"]["version"]),
        "final_state": final_state,
        "evidence_class": "RESEARCH_OBSERVED_PREQUENTIAL",
        "new_alpha_family": False,
        "new_estimand": True,
        "parent_feature_values_reused": True,
        "dsv5_values_read_after_claim": True,
        "future_total_variance_used_for_training_or_selection": False,
        "gates": {
            "G0_DATA_CONTRACT": {"status": "PASS", "passed": True},
            "G1_NONOVERLAPPING_ORIGINS": audit["G1_NONOVERLAPPING_ORIGINS"],
            "G2_B1_VS_B0": g2,
            "G3_B2_VS_B1": g3,
        },
        "model_execution": {
            "B0": "RUN",
            "B1": "RUN",
            "B2": "RUN" if b2_was_run else "NOT_RUN_BLOCKED_BY_G2",
            "macro_extension": "NOT_ALLOWED",
        },
        "adjudication": adjudication,
        "portfolio_evaluation": "NOT_ALLOWED",
        "portfolio_return_read": False,
        "sharpe": "NOT_COMPUTED",
        "model_position_target": "UNSET",
        "position_impact": 0,
        "paper_or_shadow": "NOT_AUTHORIZED",
        "broker_connection": "NOT_AUTHORIZED",
        "order_generation": False,
        "live_trading_authorized": False,
        "claim_payload_sha256": claim["claim_payload_sha256"],
        "prelabel_lock_commit": scope["head"],
    }
    result["result_payload_sha256"] = payload_sha256(result, "result_payload_sha256")
    status: dict[str, Any] = {
        "MODEL_ID": MODEL_ID,
        "FINAL_STATE": final_state,
        "G0": "PASS",
        "G1": "PASS",
        "G2": "PASS" if bool(g2["passed"]) else "FAIL",
        "G3": (
            "PASS"
            if bool(g3["passed"])
            else ("NOT_RUN_BLOCKED_BY_G2" if not b2_was_run else "FAIL")
        ),
        "PORTFOLIO_EVALUATION": "NOT_ALLOWED",
        "SHARPE": "NOT_COMPUTED",
        "POSITION_IMPACT": 0,
        "NEXT_ALLOWED_STEP": (
            "FREEZE_510300_WEEKLY_DSV5_RISK_BUDGET_POLICY_V1"
            if bool(g3["passed"])
            else "STOP_NO_RESCUE"
        ),
    }
    status["status_payload_sha256"] = payload_sha256(status, "status_payload_sha256")

    outputs = protocol["outputs"]
    write_parquet_once(resolve_path(project_root, str(outputs["label_ledger"])), labels)
    write_parquet_once(resolve_path(project_root, str(outputs["predictions"])), predictions)
    write_parquet_once(resolve_path(project_root, str(outputs["coefficients"])), coefficients)
    write_json_once(resolve_path(project_root, str(outputs["result_json"])), result)
    write_text_once(
        resolve_path(project_root, str(outputs["result_markdown"])),
        _markdown_result(result),
    )
    write_json_once(resolve_path(project_root, str(outputs["final_status"])), status)

    output_keys = [
        "one_shot_claim",
        "label_ledger",
        "predictions",
        "coefficients",
        "result_json",
        "result_markdown",
        "final_status",
    ]
    receipt: dict[str, Any] = {
        "model_id": MODEL_ID,
        "status": "PASS_ONE_SHOT_EXECUTION_COMPLETED_WITH_FROZEN_ADJUDICATION",
        "completed_at": datetime.now().astimezone().isoformat(),
        "branch": scope["branch"],
        "prelabel_lock_commit": scope["head"],
        "freeze_manifest_sha256": scope["manifest_sha256"],
        "g0_g1_audit_sha256": sha256_file(
            resolve_path(project_root, str(outputs["g0_g1_audit"]))
        ),
        "outputs": [
            _output_identity(project_root, str(outputs[key])) for key in output_keys
        ],
        "final_state": final_state,
        "dsv5_values_read_after_claim": True,
        "B1_model_trained": True,
        "B2_model_trained": b2_was_run,
        "portfolio_return_or_sharpe_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = payload_sha256(receipt, "receipt_payload_sha256")
    write_json_once(resolve_path(project_root, str(outputs["execution_receipt"])), receipt)
    return result


def record_program_failure(
    *,
    project_root: Path,
    protocol_path: Path,
    error: BaseException,
) -> Path | None:
    """claim 已存在时留下终局程序失败收据；不删除或重置 claim。"""

    try:
        protocol = load_protocol(protocol_path)
        claim_path = resolve_path(project_root, str(protocol["outputs"]["one_shot_claim"]))
        if not claim_path.exists():
            return None
        path = project_root / "reports/audit/510300_constituent_fragility_dsv5_increment_v1_program_failure_receipt.json"
        payload: dict[str, Any] = {
            "model_id": MODEL_ID,
            "status": "PROGRAM_FAILED_AFTER_ONE_SHOT_CLAIM_NO_RERUN",
            "failed_at": datetime.now().astimezone().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "claim_sha256": sha256_file(claim_path),
            "portfolio_return_or_sharpe_read": False,
            "position_impact": 0,
            "next_step": "NEW_VERSION_REQUIRED_NO_SILENT_RERUN",
        }
        payload["receipt_payload_sha256"] = payload_sha256(payload, "receipt_payload_sha256")
        write_json_once(path, payload)
        return path
    except Exception:
        return None
