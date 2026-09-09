"""独立复核 510300 大盘状态识别 V1 的冻结输入、条件路径与裁决。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_regime_transition_router_v1.yaml"
NORMAL = "NORMAL_CARRY"
STRESS = "STRESS_DELEVERAGING"
RECOVERY = "RECOVERY_REPRICING"
UNCERTAIN = "UNCERTAIN"
STATES = [NORMAL, STRESS, RECOVERY, UNCERTAIN]


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n",
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _same_number(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    left_missing = left is None or (isinstance(left, float) and math.isnan(left))
    right_missing = right is None or (isinstance(right, float) and math.isnan(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))


def _contrast(frame: pd.DataFrame, minimum_days: int = 1) -> dict[str, Any]:
    state = frame["state"].astype(str)
    stress10 = frame.loc[state.eq(STRESS) & frame["future_return_10d"].notna()]
    normal10 = frame.loc[state.eq(NORMAL) & frame["future_return_10d"].notna()]
    recovery20 = frame.loc[state.eq(RECOVERY) & frame["future_return_20d"].notna()]
    stress20 = frame.loc[state.eq(STRESS) & frame["future_return_20d"].notna()]
    stress_valid = len(stress10) >= minimum_days and len(normal10) >= minimum_days
    recovery_valid = len(recovery20) >= minimum_days and len(stress20) >= minimum_days

    stress_mae = (
        float(
            stress10["future_mae_magnitude_10d"].mean()
            - normal10["future_mae_magnitude_10d"].mean()
        )
        if stress_valid
        else None
    )
    stress_q20 = (
        float(
            stress10["future_return_10d"].quantile(0.20)
            - normal10["future_return_10d"].quantile(0.20)
        )
        if stress_valid
        else None
    )
    recovery_return = (
        float(
            recovery20["future_return_20d"].mean()
            - stress20["future_return_20d"].mean()
        )
        if recovery_valid
        else None
    )
    recovery_positive = (
        float(
            recovery20["future_positive_wealth_fraction_20d"].mean()
            - stress20["future_positive_wealth_fraction_20d"].mean()
        )
        if recovery_valid
        else None
    )
    stress_mae_direction = bool(stress_mae is not None and stress_mae > 0.0)
    stress_q20_direction = bool(stress_q20 is not None and stress_q20 < 0.0)
    recovery_return_direction = bool(
        recovery_return is not None and recovery_return > 0.0
    )
    recovery_positive_direction = bool(
        recovery_positive is not None and recovery_positive > 0.0
    )
    stress_pass = stress_mae_direction and stress_q20_direction
    recovery_pass = recovery_return_direction and recovery_positive_direction
    return {
        "stress_days_10d": int(len(stress10)),
        "normal_days_10d": int(len(normal10)),
        "recovery_days_20d": int(len(recovery20)),
        "stress_days_20d": int(len(stress20)),
        "stress_mean_mae_minus_normal_10d": stress_mae,
        "stress_q20_return_minus_normal_10d": stress_q20,
        "recovery_mean_return_minus_stress_20d": recovery_return,
        "recovery_positive_fraction_minus_stress_20d": recovery_positive,
        "stress_mean_mae_direction": stress_mae_direction,
        "stress_q20_return_direction": stress_q20_direction,
        "recovery_mean_return_direction": recovery_return_direction,
        "recovery_positive_fraction_direction": recovery_positive_direction,
        "stress_directions_pass": stress_pass,
        "recovery_directions_pass": recovery_pass,
        "all_directions_pass": stress_pass and recovery_pass,
    }


def _compare_contrast(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    integer_keys = [
        "stress_days_10d",
        "normal_days_10d",
        "recovery_days_20d",
        "stress_days_20d",
    ]
    numeric_keys = [
        "stress_mean_mae_minus_normal_10d",
        "stress_q20_return_minus_normal_10d",
        "recovery_mean_return_minus_stress_20d",
        "recovery_positive_fraction_minus_stress_20d",
    ]
    boolean_keys = [
        "stress_mean_mae_direction",
        "stress_q20_return_direction",
        "recovery_mean_return_direction",
        "recovery_positive_fraction_direction",
        "stress_directions_pass",
        "recovery_directions_pass",
        "all_directions_pass",
    ]
    return bool(
        all(int(actual[key]) == int(expected[key]) for key in integer_keys)
        and all(_same_number(actual.get(key), expected.get(key)) for key in numeric_keys)
        and all(_as_bool(actual[key]) == bool(expected[key]) for key in boolean_keys)
    )


def _rebuild_episodes(state_panel: pd.DataFrame) -> pd.DataFrame:
    frame = state_panel[["date", "state"]].copy().reset_index(drop=True)
    state_text = frame["state"].astype(str)
    group_id = state_text.ne(state_text.shift(1)).cumsum()
    rows: list[dict[str, Any]] = []
    for episode_id, (_, group) in enumerate(frame.groupby(group_id), start=1):
        rows.append(
            {
                "episode_id": episode_id,
                "state": str(group["state"].iloc[0]),
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "signal_rows": int(len(group)),
                "left_censored": bool(group.index.min() == frame.index.min()),
                "right_censored": bool(group.index.max() == frame.index.max()),
            }
        )
    return pd.DataFrame(rows)


def _verify_episode_table(stored: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if len(stored) != len(expected):
        return False
    for column in ["start_date", "end_date"]:
        stored[column] = pd.to_datetime(stored[column], errors="raise").dt.normalize()
    columns = ["episode_id", "state", "start_date", "end_date", "signal_rows"]
    if not stored[columns].reset_index(drop=True).equals(
        expected[columns].reset_index(drop=True)
    ):
        return False
    for column in ["left_censored", "right_censored"]:
        if [_as_bool(value) for value in stored[column]] != expected[column].tolist():
            return False
    return True


def _verify_outcomes(
    config: dict[str, Any],
    state_panel: pd.DataFrame,
    outcome_panel: pd.DataFrame,
) -> tuple[bool, list[str]]:
    market_path = _project_path(config["inputs"]["etf_market"]["path"])
    dividend_path = _project_path(config["inputs"]["dividends"]["path"])
    market = pd.read_parquet(market_path).copy()
    market["date"] = pd.to_datetime(market["date"], errors="raise").dt.normalize()
    market = market.loc[
        market["date"].le(pd.Timestamp(config["dates"]["outcome_market_cutoff"]))
    ].sort_values("date").reset_index(drop=True)
    dividends = pd.read_csv(dividend_path)
    dividends["record_date"] = pd.to_datetime(
        dividends["record_date"], errors="raise"
    ).dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    record_map = (
        dividends.groupby("record_date")["cash_dividend_per_share"].sum().to_dict()
    )
    date_to_position = {date: index for index, date in enumerate(market["date"])}
    horizons = [int(value) for value in config["outcomes"]["horizons_trading_days"]]
    stored = outcome_panel.set_index("date", drop=False)
    mismatches: list[str] = []

    def mismatch(message: str) -> None:
        if len(mismatches) < 25:
            mismatches.append(message)

    for signal_date in state_panel["date"]:
        if signal_date not in stored.index:
            mismatch(f"条件路径缺少信号日 {signal_date.date().isoformat()}")
            continue
        row = stored.loc[signal_date]
        signal_position = date_to_position.get(signal_date)
        if signal_position is None or signal_position + 1 >= len(market):
            for horizon in horizons:
                if not _as_bool(row[f"future_censored_{horizon}d"]):
                    mismatch(f"{signal_date.date()} {horizon}日应删失")
                if not pd.isna(row[f"future_return_{horizon}d"]):
                    mismatch(f"{signal_date.date()} {horizon}日删失收益非空")
            continue

        entry_position = signal_position + 1
        entry_date = market.loc[entry_position, "date"]
        entry_open = float(market.loc[entry_position, "open"])
        if pd.Timestamp(row["entry_date"]) != entry_date:
            mismatch(f"{signal_date.date()} T+1 进入日期不一致")
        if not _same_number(row["entry_open"], entry_open):
            mismatch(f"{signal_date.date()} T+1 开盘价不一致")

        for horizon in horizons:
            prefix = f"{horizon}d"
            exit_position = entry_position + horizon - 1
            if exit_position >= len(market):
                if not _as_bool(row[f"future_censored_{prefix}"]):
                    mismatch(f"{signal_date.date()} {prefix}应删失")
                for metric in ["return", "mae_magnitude", "mfe", "positive_wealth_fraction"]:
                    if not pd.isna(row[f"future_{metric}_{prefix}"]):
                        mismatch(f"{signal_date.date()} {prefix}删失指标非空")
                continue

            cumulative_dividend = 0.0
            path: list[float] = []
            for position in range(entry_position, exit_position + 1):
                date = market.loc[position, "date"]
                cumulative_dividend += float(record_map.get(date, 0.0))
                path_value = float(market.loc[position, "close"]) + cumulative_dividend
                path.append(path_value / entry_open - 1.0)
            values = np.asarray(path, dtype=float)
            expected = {
                f"future_return_{prefix}": float(values[-1]),
                f"future_mae_magnitude_{prefix}": float(max(0.0, -values.min())),
                f"future_mfe_{prefix}": float(max(0.0, values.max())),
                f"future_positive_wealth_fraction_{prefix}": float(
                    (values > 0.0).mean()
                ),
            }
            if _as_bool(row[f"future_censored_{prefix}"]):
                mismatch(f"{signal_date.date()} {prefix}不应删失")
            if pd.Timestamp(row[f"future_exit_date_{prefix}"]) != market.loc[
                exit_position, "date"
            ]:
                mismatch(f"{signal_date.date()} {prefix}退出日期不一致")
            for key, expected_value in expected.items():
                if not _same_number(row[key], expected_value):
                    mismatch(f"{signal_date.date()} {key}不一致")
    return not mismatches, mismatches


def _period_metrics(
    merged: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
        "FULL": (
            pd.Timestamp(config["dates"]["evaluation_start"]),
            pd.Timestamp(config["dates"]["signal_cutoff"]),
        ),
        "EARLY": tuple(pd.Timestamp(value) for value in config["dates"]["early_period"]),
        "LATE": tuple(pd.Timestamp(value) for value in config["dates"]["late_period"]),
    }
    for cycle_id, cycle in config["pressure_cycles"].items():
        periods[f"CYCLE_{cycle_id}"] = (
            pd.Timestamp(cycle["start"]),
            pd.Timestamp(cycle["end"]),
        )
    rows: list[dict[str, Any]] = []
    for period_id, (start, end) in periods.items():
        sample = merged.loc[merged["date"].between(start, end)]
        for state_name in STATES:
            group = sample.loc[sample["state"].astype(str).eq(state_name)]
            rows.append(
                {
                    "period_id": period_id,
                    "start_date": start,
                    "end_date": end,
                    "state": state_name,
                    "signal_days": int(len(group)),
                    "complete_10d_days": int(group["future_return_10d"].notna().sum()),
                    "mean_return_10d": group["future_return_10d"].mean(),
                    "return_q20_10d": group["future_return_10d"].quantile(0.20),
                    "mean_mae_10d": group["future_mae_magnitude_10d"].mean(),
                    "complete_20d_days": int(group["future_return_20d"].notna().sum()),
                    "mean_return_20d": group["future_return_20d"].mean(),
                    "mean_positive_wealth_fraction_20d": group[
                        "future_positive_wealth_fraction_20d"
                    ].mean(),
                }
            )
    return pd.DataFrame(rows)


def _verify_period_metrics(stored: pd.DataFrame, expected: pd.DataFrame) -> bool:
    if len(stored) != len(expected):
        return False
    keys = ["period_id", "state"]
    stored = stored.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if not stored[keys].equals(expected[keys]):
        return False
    for column in ["start_date", "end_date"]:
        if not pd.to_datetime(stored[column]).dt.normalize().equals(expected[column]):
            return False
    for column in ["signal_days", "complete_10d_days", "complete_20d_days"]:
        if stored[column].astype(int).tolist() != expected[column].astype(int).tolist():
            return False
    numeric = [
        "mean_return_10d",
        "return_q20_10d",
        "mean_mae_10d",
        "mean_return_20d",
        "mean_positive_wealth_fraction_20d",
    ]
    return all(
        _same_number(left, right)
        for column in numeric
        for left, right in zip(stored[column], expected[column])
    )


def _verify_contrast_csv(
    stored: pd.DataFrame,
    expected_rows: list[dict[str, Any]],
    id_column: str,
) -> bool:
    if len(stored) != len(expected_rows):
        return False
    stored_map = {str(row[id_column]): row for _, row in stored.iterrows()}
    for expected in expected_rows:
        identifier = str(expected[id_column])
        if identifier not in stored_map:
            return False
        if not _compare_contrast(stored_map[identifier].to_dict(), expected):
            return False
    return True


def _evaluate_gates(
    config: dict[str, Any],
    state_panel: pd.DataFrame,
    outcome_panel: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    merged = state_panel[["date", "state"]].merge(
        outcome_panel, on="date", how="left", validate="one_to_one"
    )
    gates = config["state_gates"]
    state_counts = {
        state: int(state_panel["state"].astype(str).eq(state).sum()) for state in STATES
    }
    day_thresholds = {
        state: int(value) for state, value in gates["minimum_state_days"].items()
    }
    days_pass = all(
        state_counts[state] >= threshold
        for state, threshold in day_thresholds.items()
    )

    complete = episodes.loc[
        ~episodes["left_censored"].map(_as_bool)
        & ~episodes["right_censored"].map(_as_bool)
    ]
    episode_counts = {
        state: int(complete["state"].eq(state).sum()) for state in STATES
    }
    episode_thresholds = {
        state: int(value)
        for state, value in gates["minimum_state_episodes"].items()
    }
    episodes_pass = all(
        episode_counts[state] >= threshold
        for state, threshold in episode_thresholds.items()
    )
    stress_episodes = complete.loc[complete["state"].eq(STRESS)]
    recovery_episodes = complete.loc[complete["state"].eq(RECOVERY)]
    stress_median = (
        None if stress_episodes.empty else float(stress_episodes["signal_rows"].median())
    )
    recovery_median = (
        None
        if recovery_episodes.empty
        else float(recovery_episodes["signal_rows"].median())
    )
    one_day_share = (
        None
        if stress_episodes.empty
        else float(stress_episodes["signal_rows"].eq(1).mean())
    )
    persistence = gates["persistence"]
    persistence_pass = bool(
        stress_median is not None
        and stress_median >= int(persistence["minimum_stress_median_episode_rows"])
        and recovery_median is not None
        and recovery_median >= int(persistence["minimum_recovery_median_episode_rows"])
        and one_day_share is not None
        and one_day_share <= float(persistence["maximum_stress_one_day_episode_share"])
    )

    full = _contrast(merged)
    period_contrasts: dict[str, dict[str, Any]] = {}
    for period_id, values in [
        ("EARLY", config["dates"]["early_period"]),
        ("LATE", config["dates"]["late_period"]),
    ]:
        start, end = (pd.Timestamp(value) for value in values)
        period_contrasts[period_id] = _contrast(
            merged.loc[merged["date"].between(start, end)]
        )
    early_late_pass = all(
        values["all_directions_pass"] for values in period_contrasts.values()
    )

    minimum_cycle_days = int(gates["held_out_cycles"]["minimum_cycle_state_days"])
    cycle_rows: list[dict[str, Any]] = []
    for cycle_id, cycle in config["pressure_cycles"].items():
        sample = merged.loc[
            merged["date"].between(pd.Timestamp(cycle["start"]), pd.Timestamp(cycle["end"]))
        ]
        cycle_rows.append(
            {
                "cycle_id": cycle_id,
                "cycle_label": cycle["label"],
                "start_date": cycle["start"],
                "end_date": cycle["end"],
                **_contrast(sample, minimum_days=minimum_cycle_days),
            }
        )
    stress_cycle_count = sum(row["stress_directions_pass"] for row in cycle_rows)
    recovery_cycle_count = sum(row["recovery_directions_pass"] for row in cycle_rows)
    cycles_pass = bool(
        stress_cycle_count
        >= int(gates["held_out_cycles"]["minimum_cycles_passing_stress_directions"])
        and recovery_cycle_count
        >= int(gates["held_out_cycles"]["minimum_cycles_passing_recovery_directions"])
    )

    leave_rows: list[dict[str, Any]] = []
    for cycle_id, cycle in config["pressure_cycles"].items():
        remaining = merged.loc[
            ~merged["date"].between(
                pd.Timestamp(cycle["start"]), pd.Timestamp(cycle["end"])
            )
        ]
        leave_rows.append(
            {
                "deleted_cycle_id": cycle_id,
                "deleted_cycle_label": cycle["label"],
                "remaining_signal_days": int(len(remaining)),
                **_contrast(remaining),
            }
        )
    leave_pass = all(row["all_directions_pass"] for row in leave_rows)
    all_pass = bool(
        days_pass
        and episodes_pass
        and persistence_pass
        and full["all_directions_pass"]
        and early_late_pass
        and cycles_pass
        and leave_pass
    )
    result = {
        "state_day_counts": state_counts,
        "minimum_state_day_gate_passed": days_pass,
        "complete_episode_counts": episode_counts,
        "minimum_episode_gate_passed": episodes_pass,
        "persistence_gate_passed": persistence_pass,
        "stress_median_episode_rows": stress_median,
        "recovery_median_episode_rows": recovery_median,
        "stress_one_day_episode_share": one_day_share,
        "full_sample": full,
        "early_late": period_contrasts,
        "early_late_passed": early_late_pass,
        "stress_cycles_passing": int(stress_cycle_count),
        "recovery_cycles_passing": int(recovery_cycle_count),
        "held_out_cycles_passed": cycles_pass,
        "leave_one_cycle_out_passed": leave_pass,
        "phase_1_state_gate_passed": all_pass,
    }
    return result, cycle_rows, leave_rows


def _compare_report_gates(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    if stored["state_day_counts"] != expected["state_day_counts"]:
        return False
    if stored["complete_episode_counts"] != expected["complete_episode_counts"]:
        return False
    boolean_pairs = [
        (stored["minimum_state_day_gate"]["passed"], expected["minimum_state_day_gate_passed"]),
        (stored["minimum_episode_gate"]["passed"], expected["minimum_episode_gate_passed"]),
        (stored["persistence_gate"]["passed"], expected["persistence_gate_passed"]),
        (stored["full_sample_structure_gate"]["passed"], expected["full_sample"]["all_directions_pass"]),
        (stored["early_late_structure_gate"]["passed"], expected["early_late_passed"]),
        (stored["held_out_cycle_structure_gate"]["passed"], expected["held_out_cycles_passed"]),
        (stored["leave_one_cycle_out_gate"]["passed"], expected["leave_one_cycle_out_passed"]),
        (stored["phase_1_state_gate_passed"], expected["phase_1_state_gate_passed"]),
    ]
    if not all(_as_bool(left) == bool(right) for left, right in boolean_pairs):
        return False
    persistence = stored["persistence_gate"]
    for key in [
        "stress_median_episode_rows",
        "recovery_median_episode_rows",
        "stress_one_day_episode_share",
    ]:
        if not _same_number(persistence.get(key), expected[key]):
            return False
    if not _compare_contrast(stored["full_sample_structure_gate"], expected["full_sample"]):
        return False
    for period_id, contrast in expected["early_late"].items():
        if not _compare_contrast(
            stored["early_late_structure_gate"]["periods"][period_id], contrast
        ):
            return False
    cycle_gate = stored["held_out_cycle_structure_gate"]
    return bool(
        int(cycle_gate["stress_cycles_passing"]) == expected["stress_cycles_passing"]
        and int(cycle_gate["recovery_cycles_passing"])
        == expected["recovery_cycles_passing"]
    )


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# 510300 大盘状态识别 V1 独立输出复核",
        "",
        f"- 复核状态：`{audit['status']}`",
        f"- 正式裁决：`{audit['formal_result_status']}`",
        f"- 第一阶段状态门：`{audit['phase_1_state_gate_passed']}`",
        f"- 第二阶段：`{audit['phase_2_status']}`",
        f"- 失败检查数：{len(audit['failed_checks'])}",
        "",
        "## 检查清单",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
    ]
    for name, passed in audit["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "本复核不导入正式研究模块，独立重算 T+1 条件路径、现金分红登记日权益、状态区间、全样本/分段/周期/删除周期方向和最终门槛。第一阶段没有组合净值、仓位、Paper、Shadow、订单、券商连接或实盘授权。",
            "",
        ]
    )
    if audit["outcome_mismatches"]:
        lines.extend(["## 条件路径差异", ""])
        lines.extend(f"- {item}" for item in audit["outcome_mismatches"])
        lines.append("")
    return "\n".join(lines)


def audit() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    paths = {key: _project_path(value) for key, value in config["paths"].items()}
    manifest_path = paths["manifest"]
    required = [
        manifest_path,
        paths["freeze_receipt"],
        paths["input_audit"],
        paths["state_panel"],
        paths["outcome_panel"],
        paths["episodes"],
        paths["period_state_metrics"],
        paths["cycle_contrasts"],
        paths["leave_one_cycle_out"],
        paths["result_json"],
        paths["result_markdown"],
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少正式研究产物：{missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(paths["result_json"].read_text(encoding="utf-8"))
    input_audit = json.loads(paths["input_audit"].read_text(encoding="utf-8"))
    state_panel = pd.read_parquet(paths["state_panel"])
    outcome_panel = pd.read_parquet(paths["outcome_panel"])
    for frame in [state_panel, outcome_panel]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    episodes = pd.read_csv(paths["episodes"])
    period_metrics = pd.read_csv(paths["period_state_metrics"])
    cycle_contrasts = pd.read_csv(paths["cycle_contrasts"])
    leave_one_out = pd.read_csv(paths["leave_one_cycle_out"])

    hash_mismatches: dict[str, dict[str, str]] = {}
    for group in ["tracked_files", "input_files"]:
        for relative, expected_hash in manifest[group].items():
            path = _project_path(relative)
            actual_hash = _sha256(path) if path.exists() else "MISSING"
            if actual_hash != expected_hash:
                hash_mismatches[relative] = {
                    "expected": expected_hash,
                    "actual": actual_hash,
                }

    future_state_columns = [
        column
        for column in state_panel.columns
        if any(token in column.lower() for token in ["future", "forward", "target_return"])
    ]
    position_state_columns = [
        column
        for column in state_panel.columns
        if any(
            token in column.lower()
            for token in ["target_position", "portfolio_weight", "order_quantity"]
        )
    ]
    expected_rows = int(config["data_gates"]["expected_evaluation_rows"])
    outcome_ok, outcome_mismatches = _verify_outcomes(
        config, state_panel, outcome_panel
    )
    rebuilt_episodes = _rebuild_episodes(state_panel)
    episodes_ok = _verify_episode_table(episodes.copy(), rebuilt_episodes)
    merged = state_panel[["date", "state"]].merge(
        outcome_panel, on="date", how="left", validate="one_to_one"
    )
    expected_metrics = _period_metrics(merged, config)
    metrics_ok = _verify_period_metrics(period_metrics.copy(), expected_metrics)
    gates, expected_cycles, expected_leave = _evaluate_gates(
        config, state_panel, outcome_panel, rebuilt_episodes
    )
    cycle_ok = _verify_contrast_csv(
        cycle_contrasts, expected_cycles, "cycle_id"
    )
    leave_ok = _verify_contrast_csv(
        leave_one_out, expected_leave, "deleted_cycle_id"
    ) and all(
        int(stored) == int(expected["remaining_signal_days"])
        for stored, expected in zip(
            leave_one_out.sort_values("deleted_cycle_id")["remaining_signal_days"],
            sorted(expected_leave, key=lambda row: row["deleted_cycle_id"]),
        )
    )
    report_gate_ok = _compare_report_gates(result["state_gates"], gates)

    expected_status = (
        config["phase_2"]["if_state_gate_passes"]
        if gates["phase_1_state_gate_passed"]
        else config["phase_2"]["if_state_gate_fails"]
    )
    expected_phase_2 = (
        "ALLOWED_NEW_FROZEN_MODULE_STUDIES"
        if gates["phase_1_state_gate_passed"]
        else "SKIPPED_STATE_GATE_FAILED"
    )
    checks = {
        "manifest_frozen_before_outcome_read": bool(
            manifest.get("state") == "FROZEN_BEFORE_FIRST_OUTCOME_READ"
            and manifest.get("frozen_before_outcome_read") is True
            and manifest.get("outcome_read_before_freeze") is False
            and manifest.get("result_preexisted_at_freeze") is False
        ),
        "manifest_tracked_and_input_hashes": not hash_mismatches,
        "result_binds_current_manifest": result.get("manifest_sha256")
        == _sha256(manifest_path),
        "input_data_contract_passed": bool(
            input_audit.get("status") == "PASS_STATE_INPUT_DATA_CONTRACT"
            and input_audit.get("passed") is True
            and input_audit.get("boundaries", {}).get("future_outcomes_read") is False
        ),
        "state_panel_calendar_and_uniqueness": bool(
            len(state_panel) == expected_rows
            and state_panel["date"].is_monotonic_increasing
            and not state_panel["date"].duplicated().any()
            and state_panel["date"].min()
            == pd.Timestamp(config["dates"]["evaluation_start"])
            and state_panel["date"].max()
            == pd.Timestamp(config["dates"]["signal_cutoff"])
        ),
        "state_panel_has_only_frozen_states": set(state_panel["state"].astype(str).unique())
        <= set(STATES),
        "state_panel_has_no_future_fields": not future_state_columns,
        "state_panel_has_no_position_or_order_fields": not position_state_columns,
        "state_and_outcome_panels_are_physically_separate": bool(
            "state" not in outcome_panel.columns
            and outcome_panel["date"].equals(state_panel["date"])
        ),
        "all_t_plus_one_outcomes_recomputed": outcome_ok,
        "episodes_recomputed": episodes_ok,
        "period_state_metrics_recomputed": metrics_ok,
        "cycle_contrasts_recomputed": cycle_ok,
        "leave_one_cycle_out_recomputed": leave_ok,
        "all_state_gates_recomputed": report_gate_ok,
        "formal_status_matches_recomputed_gate": bool(
            result.get("status") == expected_status
            and result.get("phase_1_state_gate_passed")
            == gates["phase_1_state_gate_passed"]
            and result.get("phase_2_status") == expected_phase_2
        ),
        "no_strategy_or_execution_promotion": bool(
            result.get("strategy_backtest") == "NOT_RUN_IN_PHASE_1"
            and set(result.get("portfolio_metrics", {}).values()) == {"NOT_COMPUTED"}
            and result.get("boundaries", {}).get("position_mapping") == "DISABLED"
            and result.get("boundaries", {}).get("paper_signal") == "DISABLED"
            and result.get("boundaries", {}).get("shadow_signal") == "DISABLED"
            and result.get("boundaries", {}).get("order_generation") == "DISABLED"
            and result.get("boundaries", {}).get("broker_connection") == "DISABLED"
            and result.get("boundaries", {}).get("live_trading_authorized") is False
        ),
        "old_rejection_and_no_rescue_boundaries_preserved": bool(
            result.get("boundaries", {}).get("old_microstructure_status_overridden")
            is False
            and result.get("boundaries", {}).get("old_cash_rule_read") is False
            and result.get("boundaries", {}).get("recent_performance_routing_used")
            is False
            and result.get("boundaries", {}).get("same_history_rescue_allowed")
            is False
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    output_hashes = {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in required
        if path != paths["output_audit_json"]
    }
    audit_result = {
        "project_id": config["protocol"]["project_id"],
        "status": (
            "PASS_INDEPENDENT_OUTPUT_RECOMPUTATION"
            if not failed_checks
            else "FAIL_INDEPENDENT_OUTPUT_RECOMPUTATION"
        ),
        "formal_result_status": result["status"],
        "phase_1_state_gate_passed": gates["phase_1_state_gate_passed"],
        "phase_2_status": result["phase_2_status"],
        "checks": checks,
        "failed_checks": failed_checks,
        "hash_mismatches": hash_mismatches,
        "future_state_columns": future_state_columns,
        "position_state_columns": position_state_columns,
        "outcome_mismatches": outcome_mismatches,
        "independent_recomputation": gates,
        "artifact_sha256": output_hashes,
        "boundaries": {
            "formal_research_module_imported": False,
            "strategy_returns_recomputed": False,
            "position_mapping": "DISABLED",
            "paper_signal": "DISABLED",
            "shadow_signal": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(paths["output_audit_json"], audit_result)
    _atomic_text(paths["output_audit_markdown"], _render_markdown(audit_result))
    return audit_result


def main() -> int:
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0 if result["status"] == "PASS_INDEPENDENT_OUTPUT_RECOMPUTATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
