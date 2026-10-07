"""从已有构建器提取纯本地成员与吸收率函数；不包含网络或采集函数。"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
ROOT = Path(__file__).resolve().parents[1]
class ContractError(ValueError):
    """已有数据未满足当前计算条件。"""
def project_path(value: str) -> Path:
    return ROOT / value


def _normalize_member_flags(values: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0"}
    unexpected = sorted(set(normalized.dropna().unique()).difference(allowed))
    if unexpected:
        raise ContractError(f"{label}存在无法识别的成分标记：{unexpected[:5]}")
    return normalized.isin({"true", "1"})


def load_sanitized_membership(config: dict[str, Any]) -> pd.DataFrame:
    specification = config["inputs"]["historical_membership"]
    frame = pd.read_parquet(project_path(specification["path"]))
    missing = sorted(set(specification["required_columns"]).difference(frame.columns))
    if missing:
        raise ContractError(f"历史成分区间缺少字段：{missing}")
    frame["opt_in"] = pd.to_datetime(frame["opt_in"], errors="coerce")
    frame["opt_out"] = pd.to_datetime(frame["opt_out"], errors="coerce")
    unresolved = frame.loc[frame["opt_in"].isna(), ["symbol", "opt_out"]].copy()
    observed = [
        {
            "symbol": str(row.symbol),
            "opt_out": pd.Timestamp(row.opt_out).date().isoformat(),
        }
        for row in unresolved.sort_values(["symbol", "opt_out"]).itertuples(index=False)
    ]
    if observed != config["data_contract"]["expected_unresolved_null_opt_in_intervals"]:
        raise ContractError("历史成分空纳入日期记录不符合冻结的4条未决区间")
    frame = frame.loc[frame["opt_in"].notna()].copy()
    for correction in config["data_contract"][
        "pre_factor_membership_interval_corrections"
    ]:
        mask = (
            frame["symbol"].astype(str).eq(str(correction["symbol"]))
            & frame["opt_in"].eq(pd.Timestamp(correction["original_opt_in"]))
            & frame["opt_out"].eq(pd.Timestamp(correction["original_opt_out"]))
        )
        if int(mask.sum()) != 1:
            raise ContractError(
                f"{correction['symbol']}待纠错成员区间不是唯一精确匹配"
            )
        frame.loc[mask, "opt_out"] = pd.Timestamp(correction["corrected_opt_out"])
    if frame[["symbol", "opt_in"]].isna().any().any():
        raise ContractError("清理后的历史成分仍有关键空值")
    if frame.duplicated(["symbol", "opt_in", "opt_out"]).any():
        raise ContractError("历史成分区间重复")
    return frame


def _load_panel_member_map(
    path: Path,
    label: str,
) -> dict[pd.Timestamp, list[str]]:
    frame = pd.read_parquet(path, columns=["date", "con_code", "is_index_member"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["con_code"].isna().any():
        raise ContractError(f"{label}日期或证券代码无效")
    frame["is_index_member"] = _normalize_member_flags(frame["is_index_member"], label)
    active = frame.loc[frame["is_index_member"], ["date", "con_code"]].copy()
    return {
        pd.Timestamp(date): sorted(group["con_code"].astype(str).unique())
        for date, group in active.groupby("date", sort=True)
    }


def _interval_members(membership: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    active = membership.loc[
        (membership["opt_in"] <= date)
        & (membership["opt_out"].isna() | (membership["opt_out"] > date)),
        "symbol",
    ]
    return sorted(active.astype(str).unique())


def build_daily_membership(
    calendar: pd.DatetimeIndex,
    membership: pd.DataFrame,
    external_map: dict[pd.Timestamp, list[str]],
    current_map: dict[pd.Timestamp, list[str]],
    config: dict[str, Any],
) -> tuple[dict[pd.Timestamp, list[str]], dict[pd.Timestamp, str]]:
    output: dict[pd.Timestamp, list[str]] = {}
    sources: dict[pd.Timestamp, str] = {}
    segments = config["data_contract"]["membership_segments"]
    for date in calendar:
        current = pd.Timestamp(date)
        if current <= pd.Timestamp(segments["HISTORICAL_INTERVALS"]["last_date"]):
            members = _interval_members(membership, current)
            minimum = int(segments["HISTORICAL_INTERVALS"]["minimum_active_members"])
            maximum = int(segments["HISTORICAL_INTERVALS"]["maximum_active_members"])
            if not minimum <= len(members) <= maximum:
                raise ContractError(
                    f"{current.date()}历史区间成分数{len(members)}不在{minimum}至{maximum}"
                )
            source = "HISTORICAL_INTERVALS"
        elif current <= pd.Timestamp(segments["EXTERNAL_POINT_IN_TIME_PANEL"]["last_date"]):
            if current not in external_map:
                raise ContractError(f"{current.date()}缺少外部点时成分面板")
            members = external_map[current]
            expected = int(segments["EXTERNAL_POINT_IN_TIME_PANEL"]["exact_active_members"])
            if len(members) != expected:
                raise ContractError(f"{current.date()}外部点时成分数不是{expected}")
            source = "EXTERNAL_POINT_IN_TIME_PANEL"
        else:
            if current not in current_map:
                raise ContractError(f"{current.date()}缺少当前点时成分面板")
            members = current_map[current]
            expected = int(segments["CURRENT_POINT_IN_TIME_PANEL"]["exact_active_members"])
            if len(members) != expected:
                raise ContractError(f"{current.date()}当前点时成分数不是{expected}")
            source = "CURRENT_POINT_IN_TIME_PANEL"
        output[current] = members
        sources[current] = source
    return output, sources


def load_calendar_dates_only(config: dict[str, Any]) -> pd.DatetimeIndex:
    specification = config["inputs"]["csi300_price_index"]
    frame = pd.read_parquet(project_path(specification["path"]), columns=["date"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ContractError("交易日历日期无效或重复")
    start = pd.Timestamp(config["dates"]["constituent_price_history_start"])
    end = pd.Timestamp(config["dates"]["factor_calculation_end"])
    dates = pd.DatetimeIndex(
        frame.loc[frame["date"].between(start, end), "date"].sort_values().unique()
    )
    if len(dates) < 500 + 250:
        raise ContractError("交易日历不足750日，无法完成冻结暖机")
    if dates[0] > start or dates[-1] < end:
        raise ContractError("交易日历没有覆盖冻结因子区间")
    return dates


def exponential_covariance_absorption_ratio(
    returns: np.ndarray,
    *,
    half_life: int,
    eigenvector_count: int,
) -> float:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 5:
        raise ContractError("吸收率收益矩阵形状无效")
    if not np.isfinite(values).all():
        raise ContractError("吸收率收益矩阵包含非有限值")
    ages = np.arange(values.shape[0] - 1, -1, -1, dtype=float)
    weights = np.power(0.5, ages / float(half_life))
    weights /= weights.sum()
    means = weights @ values
    centered = values - means
    covariance = (centered * np.sqrt(weights)[:, None]).T @ (
        centered * np.sqrt(weights)[:, None]
    )
    covariance = (covariance + covariance.T) / 2.0
    trace = float(np.trace(covariance))
    count = int(values.shape[1])
    top = int(eigenvector_count)
    if trace <= 0 or top < 1 or top >= count:
        raise ContractError("吸收率协方差迹或特征值数量无效")
    largest = eigvalsh(
        covariance,
        subset_by_index=[count - top, count - 1],
        check_finite=False,
        driver="evr",
    )
    ratio = float(np.maximum(largest, 0.0).sum() / trace)
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0 + 1e-10:
        raise ContractError("PCA吸收率不在有效区间")
    return min(max(ratio, 0.0), 1.0)


def compute_daily_factor(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    daily_members: dict[pd.Timestamp, list[str]],
    membership_sources: dict[pd.Timestamp, str],
    config: dict[str, Any],
) -> pd.DataFrame:
    symbols = sorted({symbol for members in daily_members.values() for symbol in members})
    expected_symbols = int(
        config["data_contract"]["expected_relevant_symbol_count_after_corrections"]
    )
    if len(symbols) != expected_symbols:
        raise ContractError(
            f"纠错后点时相关证券数量从{expected_symbols}漂移为{len(symbols)}"
        )
    matrix = prices.pivot(index="date", columns="con_code", values="total_return_close")
    matrix = matrix.reindex(index=calendar, columns=symbols).sort_index().ffill()
    log_returns = np.log(matrix).diff()
    window = int(config["factor"]["covariance_window_trading_days"])
    half_life = int(config["factor"]["exponential_half_life_trading_days"])
    minimum_count = int(config["data_contract"]["minimum_valid_member_count"])
    minimum_coverage = float(config["data_contract"]["minimum_valid_member_coverage"])
    column_positions = {symbol: index for index, symbol in enumerate(symbols)}
    values = log_returns.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    factor_dates = calendar[window:]
    for completed, date in enumerate(factor_dates, start=1):
        calendar_position = window + completed - 1
        members = daily_members[pd.Timestamp(date)]
        positions = np.array([column_positions[symbol] for symbol in members], dtype=int)
        trailing = values[calendar_position - window + 1 : calendar_position + 1, positions]
        valid_mask = np.isfinite(trailing).all(axis=0)
        valid_returns = trailing[:, valid_mask]
        valid_count = int(valid_mask.sum())
        active_count = int(len(members))
        coverage = float(valid_count / active_count) if active_count else 0.0
        eigenvectors = int(math.floor(valid_count / 5)) if valid_count else 0
        if valid_count >= minimum_count and coverage >= minimum_coverage:
            ratio = exponential_covariance_absorption_ratio(
                valid_returns,
                half_life=half_life,
                eigenvector_count=eigenvectors,
            )
        else:
            ratio = np.nan
        rows.append(
            {
                "date": pd.Timestamp(date),
                "active_member_count": active_count,
                "valid_member_count": valid_count,
                "valid_member_coverage": coverage,
                "eigenvector_count": eigenvectors,
                "absorption_ratio": ratio,
                "membership_source_segment": membership_sources[pd.Timestamp(date)],
            }
        )
        if completed % 100 == 0 or completed == len(factor_dates):
            observed = pd.DataFrame(rows)
            valid_rows = int(observed["absorption_ratio"].notna().sum())
            minimum_observed = float(observed["valid_member_coverage"].min())
            print(
                f"吸收率计算进度{completed}/{len(factor_dates)}，有效{valid_rows}，最低覆盖{minimum_observed:.4f}",
                flush=True,
            )
    factor = pd.DataFrame(rows)
    short = int(config["factor"]["short_average_trading_days"])
    long = int(config["factor"]["long_average_trading_days"])
    factor["absorption_ratio_mean_15"] = factor["absorption_ratio"].rolling(
        short, min_periods=short
    ).mean()
    factor["absorption_ratio_mean_250"] = factor["absorption_ratio"].rolling(
        long, min_periods=long
    ).mean()
    factor["absorption_ratio_std_250"] = factor["absorption_ratio"].rolling(
        long, min_periods=long
    ).std(ddof=int(config["factor"]["long_standard_deviation_ddof"]))
    denominator = factor["absorption_ratio_std_250"].where(
        factor["absorption_ratio_std_250"] > 0
    )
    factor["standardized_absorption_ratio_shift"] = (
        factor["absorption_ratio_mean_15"] - factor["absorption_ratio_mean_250"]
    ) / denominator
    return factor
