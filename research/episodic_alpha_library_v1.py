from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy import stats


PROJECT_ID = "510300_EPISODIC_ALPHA_LIBRARY_V1"
BRANCH_IDS = (
    "510300_INFORMED_CREATION_PREMIUM_V1",
    "510300_DERIVATIVE_PRESSURE_RELEASE_V1",
    "510300_OPENING_DISCOUNT_RECOVERY_V1",
)
GOVERNOR_ID = "510300_ALPHA_LIFECYCLE_GOVERNOR_V1"


class ProtocolError(RuntimeError):
    """冻结协议、输入快照或时钟约束不成立。"""


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ProtocolError(f"配置不是对象：{path}")
    if config.get("protocol", {}).get("project_id") != PROJECT_ID:
        raise ProtocolError("项目标识不匹配")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any], *, excluded_keys: Iterable[str] = ()) -> str:
    excluded = set(excluded_keys)
    material = {key: item for key, item in value.items() if key not in excluded}
    payload = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NaT:
        return None
    return value


def _configured_output_paths(root: Path, config: Mapping[str, Any]) -> list[Path]:
    artifacts = config["artifacts"]
    keys = (
        "data_admission_json",
        "result_json",
        "result_markdown",
        "opening_event_table",
        "derivative_feature_table",
        "derivative_event_table",
    )
    return [_resolve(root, artifacts[key]) for key in keys]


def _frozen_file_inventory(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for configured_path in config["freeze"]["files"]:
        path = _resolve(root, configured_path)
        if not path.is_file():
            raise ProtocolError(f"冻结文件不存在：{configured_path}")
        relative = _relative_path(root, path)
        entries[relative] = {
            "path": relative,
            "role": "PROTOCOL_IMPLEMENTATION",
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    if config["freeze"].get("include_all_data_contract_files", False):
        for contract_id, contract in config["data_contracts"].items():
            path = _resolve(root, contract["path"])
            if not path.is_file():
                if contract.get("optional", False):
                    continue
                raise ProtocolError(f"必需数据契约文件不存在：{contract_id} -> {contract['path']}")
            relative = _relative_path(root, path)
            entries[relative] = {
                "path": relative,
                "role": f"DATA_CONTRACT::{contract_id}",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return [entries[key] for key in sorted(entries)]


def freeze_bundle(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config["protocol"].get("state") != "PRE_OUTCOME_SIMULTANEOUS_FREEZE":
        raise ProtocolError("协议不是冻结前状态")
    if not config["protocol"].get("simultaneous_branch_freeze_required", False):
        raise ProtocolError("协议未要求三分支同时冻结")

    manifest_path = _resolve(root, config["artifacts"]["freeze_manifest"])
    if manifest_path.exists():
        raise ProtocolError(f"冻结清单已存在，禁止覆盖：{manifest_path}")

    existing_outputs = [str(path) for path in _configured_output_paths(root, config) if path.exists()]
    if existing_outputs:
        raise ProtocolError(f"冻结前已存在本项目结果，拒绝冻结：{existing_outputs}")

    branch_ids = tuple(branch["model_id"] for branch in config["branches"].values())
    if branch_ids != BRANCH_IDS:
        raise ProtocolError(f"分支顺序或标识不匹配：{branch_ids}")
    if config["lifecycle_governor"]["model_id"] != GOVERNOR_ID:
        raise ProtocolError("生命周期治理器标识不匹配")
    if any(bool(config["boundaries"][key]) for key in (
        "current_signal_created",
        "position_mapping_enabled",
        "paper_or_shadow_position_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    )):
        raise ProtocolError("冻结边界中存在被打开的交易开关")

    manifest: dict[str, Any] = {
        "schema_version": "510300_EPISODIC_ALPHA_LIBRARY_FREEZE_MANIFEST_V1",
        "project_id": PROJECT_ID,
        "protocol_version": config["protocol"]["version"],
        "bundle_status": "FROZEN_PRE_OUTCOME_SIMULTANEOUSLY",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_evidence_label": config["protocol"]["historical_evidence_label"],
        "branch_ids": list(BRANCH_IDS),
        "governor_id": GOVERNOR_ID,
        "results_observed_before_freeze": False,
        "pre_freeze_output_absence_verified": True,
        "live_trading_authorized": False,
        "frozen_files": _frozen_file_inventory(root, config),
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest, excluded_keys=("manifest_sha256",))
    _atomic_write_json(manifest_path, manifest)
    return {
        "status": "FROZEN_PRE_OUTCOME_SIMULTANEOUSLY",
        "manifest_path": _relative_path(root, manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "frozen_file_count": len(manifest["frozen_files"]),
        "branch_ids": list(BRANCH_IDS),
        "results_observed_before_freeze": False,
    }


def validate_frozen_bundle(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    config = load_config(config_path.resolve())
    manifest_path = _resolve(root, config["artifacts"]["freeze_manifest"])
    if not manifest_path.is_file():
        raise ProtocolError("冻结清单不存在")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_content_digest = canonical_json_sha256(manifest, excluded_keys=("manifest_sha256",))
    recorded_digest = manifest.get("manifest_sha256")
    if actual_content_digest != recorded_digest:
        raise ProtocolError("冻结清单自身内容摘要不匹配")
    if expected_manifest_sha256.lower() != str(recorded_digest).lower():
        raise ProtocolError("命令提供的冻结清单摘要不匹配")
    if manifest.get("bundle_status") != "FROZEN_PRE_OUTCOME_SIMULTANEOUSLY":
        raise ProtocolError("冻结清单状态无效")
    if tuple(manifest.get("branch_ids", [])) != BRANCH_IDS:
        raise ProtocolError("冻结清单没有同时覆盖三条分支")
    if manifest.get("governor_id") != GOVERNOR_ID:
        raise ProtocolError("冻结清单没有覆盖生命周期治理器")

    drift: list[dict[str, Any]] = []
    for entry in manifest.get("frozen_files", []):
        path = _resolve(root, entry["path"])
        if not path.is_file():
            drift.append({"path": entry["path"], "reason": "MISSING"})
            continue
        current_size = path.stat().st_size
        current_sha = sha256_file(path)
        if current_size != entry["size_bytes"] or current_sha != entry["sha256"]:
            drift.append({
                "path": entry["path"],
                "reason": "HASH_OR_SIZE_DRIFT",
                "expected_size": entry["size_bytes"],
                "actual_size": current_size,
                "expected_sha256": entry["sha256"],
                "actual_sha256": current_sha,
            })
    if drift:
        raise ProtocolError(f"冻结文件发生漂移：{json.dumps(drift, ensure_ascii=False)}")
    return config, manifest


def prior_rolling_z(series: pd.Series, window: int, minimum_history: int) -> pd.Series:
    history = series.shift(1)
    mean = history.rolling(window, min_periods=minimum_history).mean()
    std = history.rolling(window, min_periods=minimum_history).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def prior_rolling_quantile(
    series: pd.Series,
    quantile: float,
    window: int,
    minimum_history: int,
) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=minimum_history).quantile(quantile)


def robust_flow_surprise_z(
    flow: pd.Series,
    window: int = 252,
    minimum_history: int = 252,
    mad_scale: float = 1.4826,
) -> pd.Series:
    history = flow.shift(1)
    median = history.rolling(window, min_periods=minimum_history).median()
    mad = history.rolling(window, min_periods=minimum_history).apply(
        lambda values: float(np.median(np.abs(values - np.median(values)))),
        raw=True,
    )
    return (flow - median) / (mad_scale * mad).replace(0.0, np.nan)


def confirmation_from_minutes(day_bars: pd.DataFrame) -> dict[str, Any]:
    bars = day_bars.sort_values("trade_time").copy()
    bars["trade_time"] = pd.to_datetime(bars["trade_time"], errors="coerce")
    bars["clock"] = bars["trade_time"].dt.strftime("%H:%M:%S")
    pre = bars[(bars["clock"] >= "09:30:00") & (bars["clock"] <= "09:44:00")].copy()
    entry = bars[bars["clock"] == "09:45:00"].copy()
    expected = pd.date_range("2000-01-01 09:30:00", periods=15, freq="min").strftime("%H:%M:%S").tolist()
    observed = pre["clock"].tolist()
    complete = len(pre) == 15 and len(entry) == 1 and observed == expected
    result: dict[str, Any] = {
        "confirmation_complete": complete,
        "no_continued_new_low": False,
        "above_first15_vwap": False,
        "price_impact_declines": False,
        "first15_vwap": np.nan,
        "entry_price_0945": np.nan,
        "first10_low": np.nan,
        "last5_low": np.nan,
        "early_impact": np.nan,
        "late_impact": np.nan,
    }
    if not complete:
        return result

    numeric = ("open", "high", "low", "close", "vol", "amount")
    for column in numeric:
        pre[column] = pd.to_numeric(pre[column], errors="coerce")
    entry_open = float(pd.to_numeric(entry.iloc[0]["open"], errors="coerce"))
    amount_sum = float(pre["amount"].sum(min_count=1))
    volume_sum = float(pre["vol"].sum(min_count=1))
    vwap = amount_sum / volume_sum if volume_sum > 0 else np.nan
    first10_low = float(pre.iloc[:10]["low"].min())
    last5_low = float(pre.iloc[10:]["low"].min())

    previous_price = pre["close"].shift(1)
    previous_price.iloc[0] = pre.iloc[0]["open"]
    minute_return = (pre["close"] / previous_price - 1.0).abs()
    impact = minute_return / pre["amount"].where(pre["amount"] > 0)
    early_valid = impact.iloc[:5].dropna()
    late_valid = impact.iloc[10:].dropna()
    early_impact = float(early_valid.median()) if len(early_valid) >= 4 else np.nan
    late_impact = float(late_valid.median()) if len(late_valid) >= 4 else np.nan

    result.update({
        "no_continued_new_low": bool(last5_low > first10_low),
        "above_first15_vwap": bool(np.isfinite(vwap) and entry_open > vwap),
        "price_impact_declines": bool(
            np.isfinite(early_impact) and np.isfinite(late_impact) and late_impact < early_impact
        ),
        "first15_vwap": vwap,
        "entry_price_0945": entry_open,
        "first10_low": first10_low,
        "last5_low": last5_low,
        "early_impact": early_impact,
        "late_impact": late_impact,
    })
    return result


def _load_minute_bars(path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(path)
    required = {"ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ProtocolError(f"分钟数据缺少字段：{missing}")
    bars = bars[list(required)].copy()
    bars["trade_time"] = pd.to_datetime(bars["trade_time"], errors="coerce")
    if bars["trade_time"].isna().any():
        raise ProtocolError("分钟数据存在无法解析的时间")
    bars["trade_date"] = bars["trade_time"].dt.normalize()
    bars = bars.sort_values("trade_time").reset_index(drop=True)
    return bars


def _load_etf_daily(path: Path) -> pd.DataFrame:
    daily = pd.read_parquet(path)
    required = {"date", "open", "close"}
    missing = sorted(required.difference(daily.columns))
    if missing:
        raise ProtocolError(f"510300日线缺少字段：{missing}")
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()
    daily["open"] = pd.to_numeric(daily["open"], errors="coerce")
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["date", "open", "close"]).sort_values("date")
    daily = daily.drop_duplicates("date", keep="last").reset_index(drop=True)
    return daily


def _minute_day_lookup(bars: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    return {pd.Timestamp(day): group.copy() for day, group in bars.groupby("trade_date", sort=True)}


def _entry_open(day_lookup: Mapping[pd.Timestamp, pd.DataFrame], day: pd.Timestamp, clock: str) -> float:
    group = day_lookup.get(pd.Timestamp(day))
    if group is None:
        return np.nan
    row = group[group["trade_time"].dt.strftime("%H:%M:%S") == clock]
    if len(row) != 1:
        return np.nan
    return float(pd.to_numeric(row.iloc[0]["open"], errors="coerce"))


def _stress_net_return(gross_return: float, commission_rate: float, slippage_bps: float) -> float:
    if not np.isfinite(gross_return):
        return np.nan
    leg_cost = commission_rate + slippage_bps / 10000.0
    return (1.0 - leg_cost) * (1.0 + gross_return) * (1.0 - leg_cost) - 1.0


def _read_ex_dividend_dates(path: Path) -> set[pd.Timestamp]:
    frame = pd.read_csv(path)
    if "ex_date" not in frame.columns:
        raise ProtocolError("分红文件缺少 ex_date")
    dates = pd.to_datetime(frame["ex_date"], errors="coerce").dropna().dt.normalize()
    return set(pd.Timestamp(value) for value in dates)


def build_opening_discount_events(root: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    contracts = config["data_contracts"]
    daily = _load_etf_daily(_resolve(root, contracts["etf_daily"]["path"]))
    bars = _load_minute_bars(_resolve(root, contracts["etf_minute"]["path"]))
    day_lookup = _minute_day_lookup(bars)
    ex_dates = _read_ex_dividend_dates(_resolve(root, contracts["cash_distributions"]["path"]))

    daily = daily[daily["date"].isin(day_lookup)].copy().reset_index(drop=True)
    daily["previous_date"] = daily["date"].shift(1)
    daily["previous_close"] = daily["close"].shift(1)
    daily["calendar_gap_days"] = (daily["date"] - daily["previous_date"]).dt.days
    daily["gap"] = daily["open"] / daily["previous_close"] - 1.0
    rolling = config["rolling_rules"]
    daily["overnight_sigma60"] = daily["gap"].shift(1).rolling(
        int(rolling["overnight_sigma_window_days"]),
        min_periods=int(rolling["overnight_sigma_minimum_history_days"]),
    ).std(ddof=1)
    daily["gap_z"] = daily["gap"] / daily["overnight_sigma60"].replace(0.0, np.nan)
    daily["is_ex_dividend"] = daily["date"].isin(ex_dates)
    daily["is_long_holiday_first_day"] = daily["calendar_gap_days"] > 3
    threshold = float(config["branches"]["opening_discount_recovery"]["gap_z_maximum"])
    daily["candidate_preconfirmation"] = (
        daily["gap_z"].le(threshold)
        & ~daily["is_ex_dividend"]
        & ~daily["is_long_holiday_first_day"]
    )

    next_date = daily["date"].shift(-1)
    records: list[dict[str, Any]] = []
    costs = config["capital_and_costs"]
    repair_minimum = float(config["shared_clocks"]["repair_ratio_minimum"])
    for index, row in daily.iterrows():
        day = pd.Timestamp(row["date"])
        confirmation = confirmation_from_minutes(day_lookup[day])
        denominator = float(row["previous_close"] - row["open"]) if pd.notna(row["previous_close"]) else np.nan
        entry_price = float(confirmation["entry_price_0945"])
        recovery_ratio = (
            (entry_price - float(row["open"])) / denominator
            if np.isfinite(entry_price) and np.isfinite(denominator) and denominator > 0
            else np.nan
        )
        exit_day = pd.Timestamp(next_date.iloc[index]) if pd.notna(next_date.iloc[index]) else pd.NaT
        exit_price = _entry_open(day_lookup, exit_day, "09:45:00") if pd.notna(exit_day) else np.nan
        confirmed = bool(
            row["candidate_preconfirmation"]
            and confirmation["confirmation_complete"]
            and confirmation["no_continued_new_low"]
            and confirmation["above_first15_vwap"]
            and np.isfinite(recovery_ratio)
            and recovery_ratio >= repair_minimum
        )
        mature = bool(confirmed and np.isfinite(exit_price) and entry_price > 0 and exit_price > 0)
        gross = exit_price / entry_price - 1.0 if mature else np.nan
        stress_net = _stress_net_return(
            gross,
            float(costs["commission_rate_per_leg"]),
            float(costs["stress_slippage_bps_per_leg"]),
        )
        double_stress_net = _stress_net_return(
            gross,
            float(costs["commission_rate_per_leg"]),
            float(costs["double_stress_slippage_bps_per_leg"]),
        )
        group = day_lookup[day]
        close_rows = group[group["trade_time"].dt.strftime("%H:%M:%S") == "15:00:00"]
        same_day_close = (
            float(pd.to_numeric(close_rows.iloc[0]["close"], errors="coerce"))
            if len(close_rows) == 1
            else float(row["close"])
        )
        next_open = (
            float(daily.iloc[index + 1]["open"])
            if index + 1 < len(daily) and pd.notna(exit_day)
            else np.nan
        )
        same_day_after_entry_return = same_day_close / entry_price - 1.0 if mature else np.nan
        overnight_after_entry_return = next_open / same_day_close - 1.0 if mature and next_open > 0 else np.nan
        next_open_to_exit_return = exit_price / next_open - 1.0 if mature and next_open > 0 else np.nan
        records.append({
            "model_id": BRANCH_IDS[2],
            "trade_date": day,
            "previous_date": row["previous_date"],
            "exit_date": exit_day,
            "open": row["open"],
            "previous_close": row["previous_close"],
            "gap": row["gap"],
            "overnight_sigma60": row["overnight_sigma60"],
            "gap_z": row["gap_z"],
            "is_ex_dividend": bool(row["is_ex_dividend"]),
            "is_long_holiday_first_day": bool(row["is_long_holiday_first_day"]),
            "candidate_preconfirmation": bool(row["candidate_preconfirmation"]),
            **confirmation,
            "recovery_ratio": recovery_ratio,
            "repair_at_least_one_third": bool(np.isfinite(recovery_ratio) and recovery_ratio >= repair_minimum),
            "signal": confirmed,
            "mature_event": mature,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross,
            "stress_net_return": stress_net,
            "double_stress_net_return": double_stress_net,
            "same_day_after_entry_return": same_day_after_entry_return,
            "overnight_after_entry_return": overnight_after_entry_return,
            "next_open_to_exit_return": next_open_to_exit_return,
            "historical_evidence_label": "RESEARCH_OBSERVED_DISCOVERY",
            "position_impact": 0.0,
        })
    return pd.DataFrame.from_records(records)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values[valid].astype(float), weights=weights[valid].astype(float)))


def _build_option_pressure_panel(root: Path, config: Mapping[str, Any], daily: pd.DataFrame) -> pd.DataFrame:
    contracts = config["data_contracts"]
    eod = pd.read_parquet(_resolve(root, contracts["etf_option_eod"]["path"]))
    risk = pd.read_parquet(_resolve(root, contracts["etf_option_risk_indicators"]["path"]))
    required_eod = {
        "trade_date", "contract_code", "option_type", "expiry_date", "strike", "contract_unit",
        "is_adjusted", "open_interest", "underlying_close",
    }
    required_risk = {"trade_date", "contract_code", "implied_volatility"}
    if not required_eod.issubset(eod.columns) or not required_risk.issubset(risk.columns):
        raise ProtocolError("期权数据缺少构造 PutSkew/VRP 的字段")
    eod = eod[list(required_eod)].copy()
    risk = risk[list(required_risk)].copy()
    for frame in (eod, risk):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    eod["expiry_date"] = pd.to_datetime(eod["expiry_date"], errors="coerce").dt.normalize()
    numeric = ("strike", "contract_unit", "open_interest", "underlying_close")
    for column in numeric:
        eod[column] = pd.to_numeric(eod[column], errors="coerce")
    risk["implied_volatility"] = pd.to_numeric(risk["implied_volatility"], errors="coerce")
    options = eod.merge(risk, on=["trade_date", "contract_code"], how="inner", validate="one_to_one")
    adjusted = options["is_adjusted"].astype(str).str.lower().isin({"true", "1", "yes"})
    options["dte"] = (options["expiry_date"] - options["trade_date"]).dt.days
    branch = config["branches"]["derivative_pressure_release"]
    dte_min, dte_max = map(int, branch["put_skew_dte_calendar_days"])
    options = options[
        ~adjusted
        & options["contract_unit"].eq(10000)
        & options["dte"].between(dte_min, dte_max)
        & options["implied_volatility"].between(0.005, 5.0)
        & options["underlying_close"].gt(0)
        & options["strike"].gt(0)
    ].copy()
    if options.empty:
        raise ProtocolError("期权筛选后没有样本")
    nearest = options.groupby("trade_date", observed=True)["expiry_date"].transform("min")
    options = options[options["expiry_date"].eq(nearest)].copy()
    options["moneyness"] = options["strike"] / options["underlying_close"]

    pair = options.pivot_table(
        index=["trade_date", "expiry_date", "strike", "underlying_close"],
        columns="option_type",
        values="implied_volatility",
        aggfunc="first",
    ).reset_index()
    if "C" not in pair.columns or "P" not in pair.columns:
        raise ProtocolError("期权样本不能形成认购认沽平值配对")
    pair = pair.dropna(subset=["C", "P"])
    pair["atm_distance"] = (pair["strike"] / pair["underlying_close"] - 1.0).abs()
    pair = pair[pair["atm_distance"] <= float(branch["atm_moneyness_absolute_maximum"])]
    pair = pair.sort_values(["trade_date", "atm_distance", "strike"])
    atm = pair.groupby("trade_date", as_index=False, observed=True).first()
    atm["atm_pair_iv"] = (atm["C"] + atm["P"]) / 2.0

    otm_low, otm_high = map(float, branch["put_skew_otm_moneyness"])
    otm = options[
        options["option_type"].eq("P")
        & options["moneyness"].between(otm_low, otm_high)
        & options["open_interest"].gt(0)
    ].copy()
    otm_rows: list[dict[str, Any]] = []
    for day, group in otm.groupby("trade_date", sort=True):
        otm_rows.append({
            "trade_date": day,
            "otm_put_count": int(len(group)),
            "otm_put_iv": _weighted_mean(group["implied_volatility"], group["open_interest"]),
        })
    otm_daily = pd.DataFrame(otm_rows)
    panel = atm[["trade_date", "expiry_date", "atm_pair_iv", "atm_distance"]].merge(
        otm_daily,
        on="trade_date",
        how="left",
        validate="one_to_one",
    )
    panel.loc[panel["otm_put_count"].fillna(0) < 2, "otm_put_iv"] = np.nan
    panel["put_skew"] = panel["otm_put_iv"] - panel["atm_pair_iv"]

    rv = daily[["date", "close"]].copy()
    rv["log_return"] = np.log(rv["close"] / rv["close"].shift(1))
    rv["rv20"] = rv["log_return"].rolling(20, min_periods=20).std(ddof=1) * math.sqrt(
        float(config["evaluation"]["annualization_trading_days"])
    )
    panel = panel.merge(rv[["date", "rv20"]], left_on="trade_date", right_on="date", how="left")
    panel = panel.drop(columns=["date"])
    panel["vrp"] = panel["atm_pair_iv"].pow(2) - panel["rv20"].pow(2)
    return panel.sort_values("trade_date").reset_index(drop=True)


def build_derivative_pressure_events(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contracts = config["data_contracts"]
    if_daily = pd.read_parquet(_resolve(root, contracts["if_contract_daily"]["path"]))
    expiry = pd.read_parquet(_resolve(root, contracts["if_contract_expiry"]["path"]))
    spot = pd.read_parquet(_resolve(root, contracts["csi300_price_index_daily"]["path"]))
    etf = _load_etf_daily(_resolve(root, contracts["etf_daily"]["path"]))
    bars = _load_minute_bars(_resolve(root, contracts["etf_minute"]["path"]))
    day_lookup = _minute_day_lookup(bars)

    required_if = {"symbol", "date", "close", "open_interest"}
    required_expiry = {"symbol", "expiry_date"}
    if not required_if.issubset(if_daily.columns) or not required_expiry.issubset(expiry.columns):
        raise ProtocolError("IF合约或到期日面板字段不完整")
    if_daily = if_daily.copy()
    expiry = expiry.copy()
    if_daily["date"] = pd.to_datetime(if_daily["date"], errors="coerce").dt.normalize()
    expiry["expiry_date"] = pd.to_datetime(expiry["expiry_date"], errors="coerce").dt.normalize()
    if_daily["close"] = pd.to_numeric(if_daily["close"], errors="coerce")
    if_daily["open_interest"] = pd.to_numeric(if_daily["open_interest"], errors="coerce")
    if_daily = if_daily.merge(expiry[["symbol", "expiry_date"]], on="symbol", how="left", validate="many_to_one")
    if_daily["dte"] = (if_daily["expiry_date"] - if_daily["date"]).dt.days
    minimum_dte = int(config["branches"]["derivative_pressure_release"]["if_front_minimum_calendar_days_to_expiry"])
    eligible = if_daily[
        if_daily["dte"].ge(minimum_dte)
        & if_daily["close"].gt(0)
        & if_daily["open_interest"].gt(0)
    ].sort_values(["date", "expiry_date", "symbol"])
    front = eligible.groupby("date", as_index=False, observed=True).first()
    front = front.rename(columns={
        "symbol": "front_symbol",
        "close": "if_close",
        "open_interest": "front_open_interest",
    })

    spot = spot.copy()
    spot["date"] = pd.to_datetime(spot["date"], errors="coerce").dt.normalize()
    spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
    spot = spot.dropna(subset=["date", "close"]).drop_duplicates("date", keep="last")
    spot = spot[["date", "close"]].rename(columns={"close": "csi300_close"})

    feature = etf[["date", "open", "close"]].rename(columns={"open": "etf_open", "close": "etf_close"})
    feature = feature.merge(front[[
        "date", "front_symbol", "expiry_date", "dte", "if_close", "front_open_interest"
    ]], on="date", how="inner")
    feature = feature.merge(spot, on="date", how="inner", validate="one_to_one")
    feature = feature.sort_values("date").reset_index(drop=True)
    feature["if_annualized_basis"] = (
        (feature["if_close"] / feature["csi300_close"] - 1.0) * 365.0 / feature["dte"]
    )
    feature["same_front_contract"] = feature["front_symbol"].eq(feature["front_symbol"].shift(1))
    feature["roll_day"] = ~feature["same_front_contract"]
    feature["delta_front_oi"] = np.where(
        feature["same_front_contract"],
        np.log(feature["front_open_interest"] / feature["front_open_interest"].shift(1)),
        np.nan,
    )
    feature["etf_daily_return"] = feature["etf_close"] / feature["etf_close"].shift(1) - 1.0

    option_panel = _build_option_pressure_panel(root, config, etf)
    feature = feature.merge(option_panel, left_on="date", right_on="trade_date", how="left")
    feature = feature.drop(columns=["trade_date"])
    rolling = config["rolling_rules"]
    window = int(rolling["standard_z_window_days"])
    minimum = int(rolling["standard_z_minimum_history_days"])
    feature["basis_z"] = prior_rolling_z(feature["if_annualized_basis"], window, minimum)
    feature["delta_oi_z"] = prior_rolling_z(feature["delta_front_oi"], window, minimum)
    feature["put_skew_z"] = prior_rolling_z(feature["put_skew"], window, minimum)
    feature["vrp_z"] = prior_rolling_z(feature["vrp"], window, minimum)
    feature["pressure"] = (
        -feature["basis_z"] + feature["delta_oi_z"] + feature["put_skew_z"] + feature["vrp_z"]
    )
    feature["return_q05_prior252"] = prior_rolling_quantile(
        feature["etf_daily_return"], 0.05, window, minimum
    )
    feature["pressure_q95_prior252"] = prior_rolling_quantile(feature["pressure"], 0.95, window, minimum)
    feature["basis_weakens"] = (
        feature["same_front_contract"]
        & feature["if_annualized_basis"].lt(feature["if_annualized_basis"].shift(1))
    )
    feature["oi_increases_not_roll"] = feature["same_front_contract"] & feature["delta_front_oi"].gt(0)
    feature["candidate_preconfirmation"] = (
        feature["etf_daily_return"].le(feature["return_q05_prior252"])
        & feature["pressure"].ge(feature["pressure_q95_prior252"])
        & feature["basis_weakens"]
        & feature["oi_increases_not_roll"]
        & feature[["basis_z", "delta_oi_z", "put_skew_z", "vrp_z"]].notna().all(axis=1)
    )

    date_sequence = feature["date"].tolist()
    date_position = {pd.Timestamp(day): index for index, day in enumerate(date_sequence)}
    costs = config["capital_and_costs"]
    repair_minimum = float(config["shared_clocks"]["repair_ratio_minimum"])
    records: list[dict[str, Any]] = []
    for _, row in feature.iterrows():
        signal_day = pd.Timestamp(row["date"])
        position = date_position[signal_day]
        confirmation_day = pd.Timestamp(date_sequence[position + 1]) if position + 1 < len(date_sequence) else pd.NaT
        exit_day = pd.Timestamp(date_sequence[position + 2]) if position + 2 < len(date_sequence) else pd.NaT
        confirmation = (
            confirmation_from_minutes(day_lookup[confirmation_day])
            if pd.notna(confirmation_day) and confirmation_day in day_lookup
            else confirmation_from_minutes(bars.iloc[0:0].copy())
        )
        next_daily = feature.iloc[position + 1] if position + 1 < len(feature) else None
        next_open = float(next_daily["etf_open"]) if next_daily is not None else np.nan
        prior_close = float(row["etf_close"])
        entry_price = float(confirmation["entry_price_0945"])
        opening_drop_denominator = prior_close - next_open if np.isfinite(next_open) else np.nan
        if np.isfinite(opening_drop_denominator) and opening_drop_denominator > 0 and np.isfinite(entry_price):
            repair_ratio = (entry_price - next_open) / opening_drop_denominator
            repair_pass = repair_ratio >= repair_minimum
        elif np.isfinite(opening_drop_denominator) and opening_drop_denominator <= 0:
            repair_ratio = 1.0
            repair_pass = True
        else:
            repair_ratio = np.nan
            repair_pass = False
        exit_price = _entry_open(day_lookup, exit_day, "09:45:00") if pd.notna(exit_day) else np.nan
        confirmed = bool(
            row["candidate_preconfirmation"]
            and confirmation["confirmation_complete"]
            and confirmation["no_continued_new_low"]
            and confirmation["above_first15_vwap"]
            and repair_pass
            and confirmation["price_impact_declines"]
        )
        mature = bool(confirmed and np.isfinite(entry_price) and np.isfinite(exit_price) and entry_price > 0 and exit_price > 0)
        gross = exit_price / entry_price - 1.0 if mature else np.nan
        stress_net = _stress_net_return(
            gross,
            float(costs["commission_rate_per_leg"]),
            float(costs["stress_slippage_bps_per_leg"]),
        )
        double_stress_net = _stress_net_return(
            gross,
            float(costs["commission_rate_per_leg"]),
            float(costs["double_stress_slippage_bps_per_leg"]),
        )
        records.append({
            "model_id": BRANCH_IDS[1],
            "signal_date": signal_day,
            "confirmation_date": confirmation_day,
            "exit_date": exit_day,
            "front_symbol": row["front_symbol"],
            "if_annualized_basis": row["if_annualized_basis"],
            "delta_front_oi": row["delta_front_oi"],
            "put_skew": row["put_skew"],
            "vrp": row["vrp"],
            "basis_z": row["basis_z"],
            "delta_oi_z": row["delta_oi_z"],
            "put_skew_z": row["put_skew_z"],
            "vrp_z": row["vrp_z"],
            "pressure": row["pressure"],
            "return_q05_prior252": row["return_q05_prior252"],
            "pressure_q95_prior252": row["pressure_q95_prior252"],
            "etf_daily_return": row["etf_daily_return"],
            "basis_weakens": bool(row["basis_weakens"]),
            "oi_increases_not_roll": bool(row["oi_increases_not_roll"]),
            "candidate_preconfirmation": bool(row["candidate_preconfirmation"]),
            **confirmation,
            "opening_repair_ratio": repair_ratio,
            "opening_repair_pass": bool(repair_pass),
            "signal": confirmed,
            "mature_event": mature,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross,
            "stress_net_return": stress_net,
            "double_stress_net_return": double_stress_net,
            "historical_evidence_label": "RESEARCH_OBSERVED_DISCOVERY",
            "position_impact": 0.0,
        })
    return feature, pd.DataFrame.from_records(records)


def _safe_sharpe(mean: float, standard_deviation: float, annualizer: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return np.nan
    return float(mean / standard_deviation * annualizer)


def _hac_long_run_standard_deviation(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    centered = values - values.mean()
    gamma0 = float(np.mean(centered * centered))
    long_run_variance = gamma0
    maximum_lag = min(int(lag), len(values) - 1)
    for current_lag in range(1, maximum_lag + 1):
        weight = 1.0 - current_lag / (maximum_lag + 1.0)
        covariance = float(np.mean(centered[current_lag:] * centered[:-current_lag]))
        long_run_variance += 2.0 * weight * covariance
    return math.sqrt(max(long_run_variance, 0.0))


def event_metrics(events: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    mature = events[events["mature_event"]].copy()
    date_column = "exit_date"
    mature[date_column] = pd.to_datetime(mature[date_column], errors="coerce")
    mature = mature.sort_values(date_column).reset_index(drop=True)
    n = int(len(mature))
    metrics: dict[str, Any] = {
        "mature_event_count": n,
        "candidate_preconfirmation_count": int(events["candidate_preconfirmation"].sum()),
        "confirmed_signal_count": int(events["signal"].sum()),
        "historical_evidence_label": "RESEARCH_OBSERVED_DISCOVERY",
        "forward_mature_event_count": 0,
    }
    if n == 0:
        metrics.update({
            "mean_gross_bps": np.nan,
            "mean_stress_net_bps": np.nan,
            "stress_net_mean_lcb_bps": np.nan,
            "monthly_sharpe": np.nan,
            "event_interval_sharpe": np.nan,
            "lo_hac_event_sharpe": np.nan,
            "conservative_net_sharpe": np.nan,
            "historical_active_gate_diagnostics_pass": False,
        })
        return metrics

    gross = mature["gross_return"].astype(float).to_numpy()
    net = mature["stress_net_return"].astype(float).to_numpy()
    double_net = mature["double_stress_net_return"].astype(float).to_numpy()
    mean_gross = float(np.mean(gross))
    mean_net = float(np.mean(net))
    net_std = float(np.std(net, ddof=1)) if n >= 2 else np.nan
    confidence = float(config["evaluation"]["one_sided_confidence_level"])
    critical = float(stats.t.ppf(confidence, n - 1)) if n >= 2 else np.nan
    lcb = mean_net - critical * net_std / math.sqrt(n) if n >= 2 and np.isfinite(net_std) else np.nan

    first_date = mature[date_column].min()
    last_date = mature[date_column].max()
    span_years = max((last_date - first_date).days / 365.25, 1.0 / 365.25)
    events_per_year = n / span_years
    event_interval_sharpe = _safe_sharpe(mean_net, net_std, math.sqrt(events_per_year))

    month = mature.set_index(date_column)["stress_net_return"].groupby(pd.Grouper(freq="ME")).apply(
        lambda values: float(np.prod(1.0 + values.astype(float)) - 1.0)
    )
    if len(month):
        full_months = pd.date_range(month.index.min(), month.index.max(), freq="ME")
        month = month.reindex(full_months, fill_value=0.0)
    monthly_sharpe = _safe_sharpe(
        float(month.mean()) if len(month) else np.nan,
        float(month.std(ddof=1)) if len(month) >= 2 else np.nan,
        math.sqrt(12.0),
    )

    hac_lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if n > 0 else 0
    hac_std = _hac_long_run_standard_deviation(net, hac_lag)
    lo_hac = _safe_sharpe(mean_net, hac_std, math.sqrt(events_per_year))
    sharpe_values = [value for value in (monthly_sharpe, event_interval_sharpe, lo_hac) if np.isfinite(value)]
    conservative_sharpe = min(sharpe_values) if len(sharpe_values) == 3 else np.nan

    positive = np.clip(net, 0.0, None)
    positive_sum = float(positive.sum())
    max_contribution = float(positive.max() / positive_sum) if positive_sum > 0 else np.inf
    final_third_count = max(1, math.ceil(n / 3.0))
    final_third_sum = float(net[-final_third_count:].sum())
    recent20 = mature.tail(20)
    recent20_gross_bps = float(recent20["gross_return"].mean() * 10000.0) if len(recent20) else np.nan
    recent20_net_sum = float(recent20["stress_net_return"].sum()) if len(recent20) else np.nan

    active = config["lifecycle_governor"]["active_gates"]
    historical_gate_checks = {
        "mature_event_count_at_least_40": n >= int(config["evaluation"]["independent_mature_events_minimum"]),
        "mean_gross_edge_at_least_42bps": mean_gross * 10000.0 >= float(active["mean_gross_edge_bps_minimum"]),
        "stress_net_mean_lcb_above_zero": bool(np.isfinite(lcb) and lcb > float(active["one_sided_net_mean_lcb_minimum"])),
        "conservative_sharpe_at_least_1_2": bool(
            np.isfinite(conservative_sharpe)
            and conservative_sharpe >= float(active["conservative_sharpe_minimum"])
        ),
        "final_third_net_sum_positive": final_third_sum > float(active["recent_final_third_net_sum_minimum"]),
        "max_single_positive_contribution_at_most_20pct": max_contribution
        <= float(active["maximum_single_event_positive_contribution_fraction"]),
        "double_stress_net_sum_positive": float(double_net.sum()) > float(active["double_stress_net_sum_minimum"]),
    }
    metrics.update({
        "first_mature_exit_date": first_date,
        "last_mature_exit_date": last_date,
        "events_per_year": events_per_year,
        "mean_gross_bps": mean_gross * 10000.0,
        "median_gross_bps": float(np.median(gross) * 10000.0),
        "mean_stress_net_bps": mean_net * 10000.0,
        "median_stress_net_bps": float(np.median(net) * 10000.0),
        "stress_net_mean_lcb_bps": lcb * 10000.0 if np.isfinite(lcb) else np.nan,
        "stress_net_sum": float(net.sum()),
        "double_stress_net_sum": float(double_net.sum()),
        "discovery_stress_net_return_1pct_quantile": float(np.quantile(net, 0.01)),
        "monthly_sharpe": monthly_sharpe,
        "event_interval_sharpe": event_interval_sharpe,
        "lo_hac_event_sharpe": lo_hac,
        "hac_lag": hac_lag,
        "conservative_net_sharpe": conservative_sharpe,
        "final_third_event_count": final_third_count,
        "final_third_net_sum": final_third_sum,
        "maximum_single_event_positive_contribution_fraction": max_contribution,
        "recent20_mean_gross_bps": recent20_gross_bps,
        "recent20_net_sum": recent20_net_sum,
        "historical_active_gate_checks": historical_gate_checks,
        "historical_active_gate_diagnostics_pass": all(historical_gate_checks.values()),
    })
    return metrics


def historical_screen_decision(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    count = int(metrics.get("mature_event_count", 0))
    minimum = int(config["evaluation"]["independent_mature_events_minimum"])
    if count < minimum:
        return config["evaluation"]["result_if_fewer_than_40_events"]
    if bool(metrics.get("historical_active_gate_diagnostics_pass", False)):
        return config["evaluation"]["result_if_all_historical_gates_pass"]
    return config["evaluation"]["result_if_enough_events_and_any_active_gate_fails"]


def evaluate_active_eligibility(
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    independent_forward_mature_events: int,
    point_in_time_qualified: bool,
) -> dict[str, Any]:
    active = config["lifecycle_governor"]["active_gates"]
    checks = {
        "forward_events": independent_forward_mature_events >= int(active["independent_forward_mature_events_minimum"]),
        "point_in_time": bool(point_in_time_qualified),
        "mean_gross": float(metrics.get("mean_gross_bps", -np.inf)) >= float(active["mean_gross_edge_bps_minimum"]),
        "net_lcb": float(metrics.get("stress_net_mean_lcb_bps", -np.inf)) > 0.0,
        "sharpe": float(metrics.get("conservative_net_sharpe", -np.inf)) >= float(active["conservative_sharpe_minimum"]),
        "final_third": float(metrics.get("final_third_net_sum", -np.inf)) > 0.0,
        "contribution": float(metrics.get("maximum_single_event_positive_contribution_fraction", np.inf))
        <= float(active["maximum_single_event_positive_contribution_fraction"]),
        "double_stress": float(metrics.get("double_stress_net_sum", -np.inf)) > 0.0,
    }
    return {"eligible": all(checks.values()), "checks": checks}


def negative_page_hinkley_alarm(
    event_returns: Sequence[float],
    *,
    baseline_events: int = 40,
    drift_allowance_sigma: float = 0.5,
    alarm_threshold_sigma: float = 5.0,
) -> dict[str, Any]:
    values = np.asarray(event_returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= baseline_events:
        return {"alarm": False, "status": "INSUFFICIENT_POST_BASELINE_EVENTS", "statistic_sigma": 0.0}
    baseline = values[:baseline_events]
    mean = float(baseline.mean())
    sigma = float(baseline.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return {"alarm": True, "status": "ZERO_BASELINE_SCALE_FAIL_CLOSED", "statistic_sigma": np.inf}
    cumulative = 0.0
    worst = 0.0
    alarm_index: int | None = None
    for index, value in enumerate(values[baseline_events:], start=baseline_events):
        standardized = (value - mean) / sigma + drift_allowance_sigma
        cumulative = min(0.0, cumulative + standardized)
        negative_statistic = -cumulative
        worst = max(worst, negative_statistic)
        if negative_statistic >= alarm_threshold_sigma:
            alarm_index = index
            break
    return {
        "alarm": alarm_index is not None,
        "status": "ALARM" if alarm_index is not None else "NO_ALARM",
        "alarm_event_index_zero_based": alarm_index,
        "statistic_sigma": worst,
        "baseline_mean": mean,
        "baseline_sigma": sigma,
    }


def route_active_strategies(strategies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    priority = {model_id: index for index, model_id in enumerate(BRANCH_IDS)}
    eligible = [
        item for item in strategies
        if item.get("status") == "ACTIVE"
        and bool(item.get("current_signal", False))
        and float(item.get("conservative_net_edge", -np.inf)) > 0.0
        and not bool(item.get("hard_veto", False))
    ]
    if not eligible:
        return {
            "selected_model_id": None,
            "target_holding": "CASH_CNY",
            "target_position": 0.0,
            "reason": "NO_ELIGIBLE_ACTIVE_STRATEGY",
        }
    selected = sorted(
        eligible,
        key=lambda item: (-float(item["conservative_net_edge"]), priority.get(str(item["model_id"]), 999)),
    )[0]
    return {
        "selected_model_id": selected["model_id"],
        "target_holding": "510300.SH",
        "target_position": 1.0,
        "reason": "HIGHEST_CONSERVATIVE_NET_EDGE_WITH_FROZEN_PRIORITY_TIE_BREAK",
    }


def _admit_creation(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    contracts = config["data_contracts"]
    share_path = _resolve(root, contracts["actual_share_history_candidate"]["path"])
    iopv_path = _resolve(root, contracts["legacy_iopv_snapshots"]["path"])
    reasons: list[str] = []
    share_rows = 0
    share_columns: list[str] = []
    if not share_path.is_file():
        reasons.append("ACTUAL_SHARE_FILE_MISSING")
    else:
        shares = pd.read_parquet(share_path)
        share_rows = int(len(shares))
        share_columns = list(shares.columns)
        required = set(config["branches"]["informed_creation_premium"]["required_columns"]["actual_shares"])
        missing = sorted(required.difference(shares.columns))
        if missing:
            reasons.append("ACTUAL_SHARE_STRICT_AVAILABLE_AT_SCHEMA_MISSING:" + ",".join(missing))
        if "share_retrieved_at" in shares.columns and "date" in shares.columns:
            retrieved = pd.to_datetime(shares["share_retrieved_at"], errors="coerce", utc=True)
            event_date = pd.to_datetime(shares["date"], errors="coerce", utc=True)
            retrospective = (retrieved.dt.normalize() > event_date.dt.normalize()).mean()
            if retrospective > 0.01:
                reasons.append("HISTORICAL_SHARES_RETRIEVED_RETROSPECTIVELY_NOT_POINT_IN_TIME")

    quality_days = 0
    iopv_rows = 0
    iopv_columns: list[str] = []
    if not iopv_path.is_file():
        reasons.append("IOPV_FILE_MISSING")
    else:
        iopv = pd.read_parquet(iopv_path)
        iopv_rows = int(len(iopv))
        iopv_columns = list(iopv.columns)
        required_iopv = set(config["branches"]["informed_creation_premium"]["required_columns"]["iopv"])
        missing_iopv = sorted(required_iopv.difference(iopv.columns))
        if missing_iopv:
            reasons.append("IOPV_BID_ASK_OR_CLOCK_SCHEMA_MISSING:" + ",".join(missing_iopv))
        if {"trade_date", "exchange_timestamp", "last_price", "iopv"}.issubset(iopv.columns):
            iopv = iopv.copy()
            iopv["exchange_timestamp"] = pd.to_datetime(iopv["exchange_timestamp"], errors="coerce")
            iopv["trade_date"] = pd.to_datetime(iopv["trade_date"], errors="coerce").dt.normalize()
            iopv["last_price"] = pd.to_numeric(iopv["last_price"], errors="coerce")
            iopv["iopv"] = pd.to_numeric(iopv["iopv"], errors="coerce")
            valid = iopv[
                iopv["exchange_timestamp"].notna()
                & iopv["last_price"].gt(0)
                & iopv["iopv"].gt(0)
            ].drop_duplicates(["trade_date", "exchange_timestamp"])
            counts = valid.groupby("trade_date")["exchange_timestamp"].agg(["count", "max"])
            quality_days = int(
                (
                    counts["count"].ge(int(config["branches"]["informed_creation_premium"]["minimum_valid_iopv_snapshots"]))
                    & counts["max"].dt.strftime("%H:%M:%S").ge("14:59:00")
                ).sum()
            )
            if quality_days < 20:
                reasons.append(f"IOPV_COMPLETE_QUALITY_DAYS_BELOW_20:{quality_days}")
    return {
        "model_id": BRANCH_IDS[0],
        "status": "NO_VIEW_DATA_CONTRACT_FAILED" if reasons else "PASS_RESEARCH_OBSERVED_DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED" if reasons else "ALLOWED_DISCOVERY_ONLY",
        "point_in_time_qualified": False,
        "actual_share_rows": share_rows,
        "actual_share_columns": share_columns,
        "iopv_rows": iopv_rows,
        "iopv_columns": iopv_columns,
        "complete_iopv_quality_days": quality_days,
        "reasons": reasons,
    }


def _admit_derivative(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    requirements = {
        "if_contract_daily": {"symbol", "date", "close", "open_interest"},
        "if_contract_expiry": {"symbol", "expiry_date"},
        "csi300_price_index_daily": {"date", "close"},
        "etf_daily": {"date", "open", "close"},
        "etf_option_eod": {"trade_date", "contract_code", "expiry_date", "strike", "open_interest"},
        "etf_option_risk_indicators": {"trade_date", "contract_code", "implied_volatility"},
        "etf_minute": {"trade_time", "open", "high", "low", "close", "vol", "amount"},
    }
    reasons: list[str] = []
    inventory: dict[str, Any] = {}
    for contract_id, columns in requirements.items():
        path = _resolve(root, config["data_contracts"][contract_id]["path"])
        if not path.is_file():
            reasons.append(f"MISSING:{contract_id}")
            continue
        frame = pd.read_parquet(path)
        missing = sorted(columns.difference(frame.columns))
        inventory[contract_id] = {"rows": int(len(frame)), "columns": list(frame.columns)}
        if missing:
            reasons.append(f"SCHEMA:{contract_id}:{','.join(missing)}")
    if not reasons:
        if inventory["if_contract_daily"]["rows"] < 10000:
            reasons.append("IF_HISTORY_TOO_SHORT")
        if inventory["etf_option_risk_indicators"]["rows"] < 100000:
            reasons.append("OPTION_RISK_HISTORY_TOO_SHORT")
        if inventory["etf_minute"]["rows"] < 200000:
            reasons.append("MINUTE_HISTORY_TOO_SHORT")
    return {
        "model_id": BRANCH_IDS[1],
        "status": "PASS_RESEARCH_OBSERVED_DISCOVERY_ONLY" if not reasons else "NO_VIEW_DATA_CONTRACT_FAILED",
        "return_evaluation": "ALLOWED_DISCOVERY_ONLY" if not reasons else "NOT_ALLOWED",
        "point_in_time_qualified": False,
        "source_caveats": [
            "IF与上交所期权风险指标是官方历史值，但文件未保存逐日原始发布时间",
            "期权EOD合约与持仓字段来自事后代理文件，只能用于回溯发现",
            "不得声称复现主动买卖方向或做市商订单失衡",
        ],
        "inventory": inventory,
        "reasons": reasons,
    }


def _admit_opening(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    minute_path = _resolve(root, config["data_contracts"]["etf_minute"]["path"])
    daily_path = _resolve(root, config["data_contracts"]["etf_daily"]["path"])
    dividend_path = _resolve(root, config["data_contracts"]["cash_distributions"]["path"])
    reasons: list[str] = []
    if not minute_path.is_file() or not daily_path.is_file() or not dividend_path.is_file():
        if not minute_path.is_file():
            reasons.append("MINUTE_FILE_MISSING")
        if not daily_path.is_file():
            reasons.append("DAILY_FILE_MISSING")
        if not dividend_path.is_file():
            reasons.append("DIVIDEND_FILE_MISSING")
        return {
            "model_id": BRANCH_IDS[2],
            "status": "NO_VIEW_DATA_CONTRACT_FAILED",
            "return_evaluation": "NOT_ALLOWED",
            "point_in_time_qualified": False,
            "reasons": reasons,
        }
    bars = _load_minute_bars(minute_path)
    duplicate_count = int(bars.duplicated(["ts_code", "trade_time"]).sum())
    if duplicate_count:
        reasons.append(f"DUPLICATE_MINUTE_KEYS:{duplicate_count}")
    clocks = bars["trade_time"].dt.strftime("%H:%M:%S")
    opening = bars[clocks.between("09:30:00", "09:45:00")].copy()
    day_counts = opening.groupby("trade_date")["trade_time"].nunique()
    complete_days = int(day_counts.eq(16).sum())
    if complete_days < 1000:
        reasons.append(f"COMPLETE_OPENING_DAYS_BELOW_1000:{complete_days}")
    return {
        "model_id": BRANCH_IDS[2],
        "status": "PASS_RESEARCH_OBSERVED_DISCOVERY_ONLY" if not reasons else "NO_VIEW_DATA_CONTRACT_FAILED",
        "return_evaluation": "ALLOWED_DISCOVERY_ONLY" if not reasons else "NOT_ALLOWED",
        "point_in_time_qualified": False,
        "minute_rows": int(len(bars)),
        "first_minute": bars["trade_time"].min(),
        "last_minute": bars["trade_time"].max(),
        "opening_complete_days": complete_days,
        "duplicate_minute_keys": duplicate_count,
        "source_caveat": "分钟历史为事后取得数据，只能用于RESEARCH_OBSERVED_DISCOVERY",
        "reasons": reasons,
    }


def _opening_stop_diagnostics(events: pd.DataFrame) -> dict[str, Any]:
    mature = events[events["mature_event"]].sort_values("exit_date")
    recent20 = mature.tail(20)
    median_recovery = float(mature["recovery_ratio"].median()) if len(mature) else np.nan
    recent_net_sum = float(recent20["stress_net_return"].sum()) if len(recent20) else np.nan
    same_day = float(recent20["same_day_after_entry_return"].median()) if len(recent20) else np.nan
    overnight = float(recent20["overnight_after_entry_return"].median()) if len(recent20) else np.nan
    overnight_offsets = bool(
        len(recent20) >= 20
        and np.isfinite(same_day)
        and np.isfinite(overnight)
        and same_day > 0
        and overnight < -same_day
    )
    return {
        "median_recovery_ratio": median_recovery,
        "median_recovery_ratio_below_zero": bool(np.isfinite(median_recovery) and median_recovery < 0),
        "recent20_net_sum": recent_net_sum,
        "recent20_net_sum_below_zero": bool(len(recent20) >= 20 and recent_net_sum < 0),
        "recent20_median_same_day_after_entry_return": same_day,
        "recent20_median_overnight_after_entry_return": overnight,
        "recent20_overnight_loss_offsets_same_day_recovery": overnight_offsets,
        "news_shock_loss_cluster": "NOT_EVALUABLE_UNTIL_PUBLIC_EVENT_TAXONOMY_FROZEN",
    }


def _derivative_stop_diagnostics(events: pd.DataFrame, metrics: Mapping[str, Any]) -> dict[str, Any]:
    mature = events[events["mature_event"]].sort_values("exit_date")
    recent20 = mature.tail(20)
    median_gross = float(mature["gross_return"].median()) if len(mature) else np.nan
    recent20_gross_bps = float(recent20["gross_return"].mean() * 10000.0) if len(recent20) else np.nan
    return {
        "median_post_pressure_gross_return": median_gross,
        "post_pressure_no_reversal_or_continued_decline": bool(np.isfinite(median_gross) and median_gross <= 0),
        "basis_recovery_return_relation": "NOT_EVALUATED_WITHOUT_SEPARATE_FORWARD_BASIS_RECOVERY_CLOCK",
        "discovery_stress_net_return_1pct_quantile": metrics.get("discovery_stress_net_return_1pct_quantile"),
        "forward_tail_breach": "NOT_APPLICABLE_BEFORE_FORWARD_EVENTS",
        "recent20_mean_gross_bps": recent20_gross_bps,
        "recent20_mean_gross_below_28bps": bool(len(recent20) >= 20 and recent20_gross_bps < 28.0),
    }


def _branch_lifecycle_state(screen_decision: str) -> str:
    if screen_decision == "REJECTED_FROZEN":
        return "REJECTED_FROZEN"
    if screen_decision == "HISTORICAL_DISCOVERY_SCREEN_PASS_FORWARD_SHADOW_REQUIRED":
        return "SHADOW"
    return "CANDIDATE"


def _format_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "NOT_AVAILABLE"
    return f"{numeric:.{digits}f}"


def _render_markdown(result: Mapping[str, Any], admission: Mapping[str, Any]) -> str:
    branches = result["branches"]
    lines = [
        "# 510300 短周期情景 Alpha 库 V1：冻结后发现性结果",
        "",
        f"- 冻结清单：`{result['manifest_sha256']}`",
        "- 历史证据标签：`RESEARCH_OBSERVED_DISCOVERY`",
        "- 当前已验证高夏普策略：`NONE`",
        "- 当前路由：`CASH_CNY`，仓位影响为 0",
        "- 实盘交易授权：`FALSE`",
        "",
        "## 数据准入",
        "",
        "| 分支 | 准入状态 | 收益评价 | 点时资格 |",
        "|---|---|---|---|",
    ]
    for model_id in BRANCH_IDS:
        item = admission[model_id]
        lines.append(
            f"| `{model_id}` | `{item['status']}` | `{item['return_evaluation']}` | "
            f"`{str(item['point_in_time_qualified']).upper()}` |"
        )
    lines.extend(["", "## 分支裁决", ""])
    for model_id in BRANCH_IDS:
        item = branches[model_id]
        lines.extend([
            f"### {model_id}",
            "",
            f"- 研究裁决：`{item['research_decision']}`",
            f"- 生命周期状态：`{item['lifecycle_state']}`",
            f"- 成熟历史事件：`{item.get('metrics', {}).get('mature_event_count', 0)}`",
            f"- 平均毛边际：`{_format_metric(item.get('metrics', {}).get('mean_gross_bps'))} bp`",
            f"- 平均压力成本后净边际：`{_format_metric(item.get('metrics', {}).get('mean_stress_net_bps'))} bp`",
            f"- 保守净夏普：`{_format_metric(item.get('metrics', {}).get('conservative_net_sharpe'))}`",
            "- 前向成熟事件：`0`；历史事件不得用于 ACTIVE 晋级。",
            "",
        ])
    lines.extend([
        "## 生命周期与路由",
        "",
        "三条分支均没有取得基于新前向事件的 `ACTIVE` 资格。治理器因此不能把任何历史信号映射为仓位；当前唯一合法路由是现金。即使某条历史发现性筛选通过，也只能进入零仓位 `SHADOW`。",
        "",
        "## 固定边界",
        "",
        "`position_mapping_enabled=false`、`order_generation_enabled=false`、`broker_connection_enabled=false`、`live_trading_enabled=false`。完整期权订单失衡分支维持 `BLOCKED_NO_FREE_TRADE_SIGN_DATA`。",
        "",
    ])
    return "\n".join(lines)


def run_library(
    root: Path,
    config_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = root.resolve()
    config, manifest = validate_frozen_bundle(root, config_path, expected_manifest_sha256)

    creation_admission = _admit_creation(root, config)
    derivative_admission = _admit_derivative(root, config)
    opening_admission = _admit_opening(root, config)
    admission = {
        BRANCH_IDS[0]: creation_admission,
        BRANCH_IDS[1]: derivative_admission,
        BRANCH_IDS[2]: opening_admission,
    }

    branches: dict[str, Any] = {}
    branches[BRANCH_IDS[0]] = {
        "model_id": BRANCH_IDS[0],
        "research_decision": creation_admission["status"],
        "lifecycle_state": "CANDIDATE",
        "return_evaluation": creation_admission["return_evaluation"],
        "metrics": {"mature_event_count": 0, "forward_mature_event_count": 0},
        "reason": creation_admission["reasons"],
        "position_impact": 0.0,
    }

    if derivative_admission["return_evaluation"] == "ALLOWED_DISCOVERY_ONLY":
        derivative_features, derivative_events = build_derivative_pressure_events(root, config)
        derivative_metrics = event_metrics(derivative_events, config)
        derivative_decision = historical_screen_decision(derivative_metrics, config)
        derivative_stops = _derivative_stop_diagnostics(derivative_events, derivative_metrics)
        _atomic_write_parquet(_resolve(root, config["artifacts"]["derivative_feature_table"]), derivative_features)
        _atomic_write_parquet(_resolve(root, config["artifacts"]["derivative_event_table"]), derivative_events)
    else:
        derivative_metrics = {"mature_event_count": 0, "forward_mature_event_count": 0}
        derivative_decision = derivative_admission["status"]
        derivative_stops = {}
    branches[BRANCH_IDS[1]] = {
        "model_id": BRANCH_IDS[1],
        "research_decision": derivative_decision,
        "lifecycle_state": _branch_lifecycle_state(derivative_decision),
        "return_evaluation": derivative_admission["return_evaluation"],
        "metrics": derivative_metrics,
        "stop_diagnostics": derivative_stops,
        "active_eligibility": evaluate_active_eligibility(
            derivative_metrics,
            config,
            independent_forward_mature_events=0,
            point_in_time_qualified=False,
        ),
        "position_impact": 0.0,
    }

    if opening_admission["return_evaluation"] == "ALLOWED_DISCOVERY_ONLY":
        opening_events = build_opening_discount_events(root, config)
        opening_metrics = event_metrics(opening_events, config)
        opening_decision = historical_screen_decision(opening_metrics, config)
        opening_stops = _opening_stop_diagnostics(opening_events)
        _atomic_write_parquet(_resolve(root, config["artifacts"]["opening_event_table"]), opening_events)
    else:
        opening_metrics = {"mature_event_count": 0, "forward_mature_event_count": 0}
        opening_decision = opening_admission["status"]
        opening_stops = {}
    branches[BRANCH_IDS[2]] = {
        "model_id": BRANCH_IDS[2],
        "research_decision": opening_decision,
        "lifecycle_state": _branch_lifecycle_state(opening_decision),
        "return_evaluation": opening_admission["return_evaluation"],
        "metrics": opening_metrics,
        "stop_diagnostics": opening_stops,
        "active_eligibility": evaluate_active_eligibility(
            opening_metrics,
            config,
            independent_forward_mature_events=0,
            point_in_time_qualified=False,
        ),
        "position_impact": 0.0,
    }

    route = route_active_strategies([
        {
            "model_id": model_id,
            "status": item["lifecycle_state"],
            "current_signal": False,
            "conservative_net_edge": -np.inf,
            "hard_veto": True,
        }
        for model_id, item in branches.items()
    ])
    result: dict[str, Any] = {
        "schema_version": "510300_EPISODIC_ALPHA_LIBRARY_RESULT_V1",
        "project_id": PROJECT_ID,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest["manifest_sha256"],
        "historical_evidence_label": "RESEARCH_OBSERVED_DISCOVERY",
        "formal_status": config["formal_status"],
        "branches": branches,
        "lifecycle_governor": {
            "model_id": GOVERNOR_ID,
            "design_status": "AUTHORIZED_FOR_DESIGN",
            "historical_activation_allowed": False,
            "forward_mature_events_counted_for_activation": 0,
            "route": route,
        },
        "boundaries": config["boundaries"],
        "status_summary": {
            "project": "AUTHORIZED_FOR_DISCOVERY",
            "creation": branches[BRANCH_IDS[0]]["research_decision"],
            "derivative": branches[BRANCH_IDS[1]]["research_decision"],
            "opening": branches[BRANCH_IDS[2]]["research_decision"],
            "current_validated_high_sharpe_strategy": "NONE",
            "current_holding_route": "CASH_CNY",
            "position_impact": 0.0,
            "live_trading_authorized": False,
        },
    }

    data_admission_path = _resolve(root, config["artifacts"]["data_admission_json"])
    result_path = _resolve(root, config["artifacts"]["result_json"])
    markdown_path = _resolve(root, config["artifacts"]["result_markdown"])
    admission_document = {
        "schema_version": "510300_EPISODIC_ALPHA_LIBRARY_DATA_ADMISSION_V1",
        "project_id": PROJECT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "branches": admission,
        "historical_evidence_label": "RESEARCH_OBSERVED_DISCOVERY",
        "live_trading_authorized": False,
    }
    _atomic_write_json(data_admission_path, admission_document)
    _atomic_write_json(result_path, result)
    _atomic_write_text(markdown_path, _render_markdown(result, admission))
    return result
