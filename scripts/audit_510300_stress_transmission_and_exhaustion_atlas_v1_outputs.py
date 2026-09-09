"""聚焦复核压力传导与耗竭图谱的时钟、事件、路径和裁决。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_stress_transmission_and_exhaustion_atlas_v1.yaml"
STRESS = "STRESS_DELEVERAGING"


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


def _same_number(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    left_missing = left is None or pd.isna(left)
    right_missing = right is None or pd.isna(right)
    if left_missing or right_missing:
        return bool(left_missing and right_missing)
    return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))


def _normalize_date(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result.sort_values(column, inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _verify_outcomes(
    config: dict[str, Any],
    mechanism_panel: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[bool, list[str]]:
    market = _normalize_date(
        pd.read_parquet(_project_path(config["inputs"]["etf_market"]["path"]))
    )
    market = market.loc[
        market["date"].le(pd.Timestamp(config["dates"]["outcome_market_cutoff"]))
    ].reset_index(drop=True)
    dividends = pd.read_csv(_project_path(config["inputs"]["dividends"]["path"]))
    for column in ["record_date", "ex_date"]:
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    dividends["cash_dividend_per_share"] = pd.to_numeric(
        dividends["cash_dividend_per_share"], errors="raise"
    )
    dividend_events = {
        date: group[["record_date", "cash_dividend_per_share"]].to_dict("records")
        for date, group in dividends.groupby("ex_date")
    }
    positions = {date: index for index, date in enumerate(market["date"])}
    stored = outcomes.set_index("date")
    mismatches: list[str] = []

    def add(message: str) -> None:
        if len(mismatches) < 20:
            mismatches.append(message)

    for signal_date in mechanism_panel["date"]:
        if signal_date not in stored.index:
            add(f"缺少信号日 {signal_date.date()}")
            continue
        row = stored.loc[signal_date]
        signal_position = positions.get(signal_date)
        if signal_position is None or signal_position + 1 >= len(market):
            for horizon in [10, 20]:
                if not _as_bool(row[f"future_censored_{horizon}d"]):
                    add(f"{signal_date.date()} {horizon}日应删失")
            continue
        entry_position = signal_position + 1
        entry_date = market.loc[entry_position, "date"]
        entry_open = float(market.loc[entry_position, "open"])
        if pd.Timestamp(row["entry_date"]) != entry_date:
            add(f"{signal_date.date()} T+1进入日不一致")
        if not _same_number(row["entry_open"], entry_open):
            add(f"{signal_date.date()} T+1开盘价不一致")
        for horizon in [10, 20]:
            exit_position = entry_position + horizon - 1
            prefix = f"{horizon}d"
            if exit_position >= len(market):
                if not _as_bool(row[f"future_censored_{prefix}"]):
                    add(f"{signal_date.date()} {prefix}应删失")
                for metric in ["return", "mae_magnitude", "mfe", "positive_wealth_fraction"]:
                    if not pd.isna(row[f"future_{metric}_{prefix}"]):
                        add(f"{signal_date.date()} {prefix}删失指标非空")
                continue
            cumulative_cash = 0.0
            path: list[float] = []
            for position in range(entry_position, exit_position + 1):
                date = market.loc[position, "date"]
                for dividend in dividend_events.get(date, []):
                    if entry_date <= dividend["record_date"]:
                        cumulative_cash += float(dividend["cash_dividend_per_share"])
                path.append(
                    (float(market.loc[position, "close"]) + cumulative_cash)
                    / entry_open
                    - 1.0
                )
            values = np.asarray(path, dtype=float)
            expected = {
                f"future_return_{prefix}": float(values[-1]),
                f"future_mae_magnitude_{prefix}": float(max(0.0, -values.min())),
                f"future_mfe_{prefix}": float(max(0.0, values.max())),
                f"future_positive_wealth_fraction_{prefix}": float(
                    values.astype(float).__gt__(0.0).mean()
                ),
            }
            if _as_bool(row[f"future_censored_{prefix}"]):
                add(f"{signal_date.date()} {prefix}不应删失")
            for key, expected_value in expected.items():
                if not _same_number(row[key], expected_value):
                    add(f"{signal_date.date()} {key}不一致")
    return not mismatches, mismatches


def _crossing_timing(
    values: pd.Series,
    anchor_position: int,
    lookback: int,
    followup: int,
    threshold: float,
) -> tuple[pd.Timestamp | pd.NaT, float | None, bool]:
    start = max(0, anchor_position - lookback)
    end = min(len(values) - 1, anchor_position + followup)
    current = values.iloc[start]
    if pd.notna(current) and float(current) >= threshold:
        if start > 0:
            previous = values.iloc[start - 1]
            if pd.notna(previous) and float(previous) < threshold:
                return values.index[start], float(anchor_position - start), False
        return pd.NaT, None, True
    for position in range(start + 1, end + 1):
        previous = values.iloc[position - 1]
        current = values.iloc[position]
        if pd.notna(previous) and pd.notna(current) and previous < threshold <= current:
            return values.index[position], float(anchor_position - position), False
    return pd.NaT, None, False


def _verify_price_events(
    config: dict[str, Any],
    panel: pd.DataFrame,
    stored: pd.DataFrame,
) -> bool:
    specification = config["event_definitions"]["price_damage_events"]
    peak_rows = int(specification["rolling_peak_rows"])
    cooldown = int(specification["cooldown_trading_rows"])
    future_rows = int(specification["minimum_complete_future_rows"])
    lookback = int(specification["leading_lookback_rows"])
    followup = int(specification["lagging_followup_rows"])
    threshold = float(specification["drawdown_threshold"])
    crossing_threshold = float(config["gates"]["leading_gate"]["channel_crossing_threshold"])
    drawdown = panel["total_return_index_rebuilt"] / panel[
        "total_return_index_rebuilt"
    ].rolling(peak_rows, min_periods=peak_rows).max() - 1.0
    trigger = drawdown.le(threshold) & drawdown.shift(1).gt(threshold)
    positions: list[int] = []
    last = -10_000
    for position in np.flatnonzero(trigger.to_numpy()):
        if position - last < cooldown:
            continue
        positions.append(int(position))
        last = int(position)
    if len(stored) != len(positions):
        return False
    channel_map = {
        "source": "pressure_source_score",
        "transmission": "transmission_score",
        "cross_index_etf": "cross_index_etf_pressure",
        "same_index_etf": "same_index_etf_pressure",
        "breadth": "breadth_transmission_score",
        "if_raw_basis": "if_raw_basis_pressure",
        "h3_downside": "downside_pressure_percentile",
    }
    index = pd.DatetimeIndex(panel["date"])
    for row_number, position in enumerate(positions):
        row = stored.iloc[row_number]
        if str(row["event_id"]) != f"PDE_{row_number + 1:03d}":
            return False
        if pd.Timestamp(row["trigger_date"]) != panel.loc[position, "date"]:
            return False
        if not _same_number(row["drawdown20_at_trigger"], drawdown.iloc[position]):
            return False
        if _as_bool(row["future_20d_complete"]) != bool(position + future_rows < len(panel)):
            return False
        for label, column in channel_map.items():
            values = pd.Series(panel[column].to_numpy(), index=index)
            crossing, timing, censored = _crossing_timing(
                values, position, lookback, followup, crossing_threshold
            )
            stored_date = row[f"{label}_cross_date"]
            if pd.isna(crossing) != pd.isna(stored_date):
                return False
            if pd.notna(crossing) and pd.Timestamp(stored_date) != crossing:
                return False
            if not _same_number(row[f"{label}_lead_rows"], timing):
                return False
            if _as_bool(row[f"{label}_left_censored"]) != censored:
                return False
    return True


def _verify_router_events(
    config: dict[str, Any],
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
    events: pd.DataFrame,
    question_a: pd.DataFrame,
    question_b: pd.DataFrame,
) -> bool:
    episodes = pd.read_csv(_project_path(config["inputs"]["router_episodes"]["path"]))
    episodes = episodes.loc[
        episodes["state"].eq(STRESS)
        & ~episodes["left_censored"].map(_as_bool)
        & ~episodes["right_censored"].map(_as_bool)
    ].copy()
    episodes["start_date"] = pd.to_datetime(episodes["start_date"]).dt.normalize()
    episodes["end_date"] = pd.to_datetime(episodes["end_date"]).dt.normalize()
    if len(events) != 61 or len(events) != len(episodes) or not events["event_id"].is_unique:
        return False
    if not question_a["event_id"].is_unique or not question_b["event_id"].is_unique:
        return False
    if not question_a["event_weight"].eq(1.0).all() or not question_b["event_weight"].eq(1.0).all():
        return False
    outcome_by_date = outcomes.set_index("date")
    positions = {date: index for index, date in enumerate(panel["date"])}
    a_by_event = question_a.set_index("event_id")
    b_by_event = question_b.set_index("event_id")
    for episode, event in zip(episodes.itertuples(index=False), events.itertuples(index=False)):
        expected_id = f"RSE_{int(episode.episode_id):03d}"
        if event.event_id != expected_id:
            return False
        if pd.Timestamp(event.start_date) != episode.start_date or pd.Timestamp(event.end_date) != episode.end_date:
            return False
        start = positions[episode.start_date]
        end = positions[episode.end_date]
        search_end = min(len(panel) - 1, end + 20)
        repair_condition = (
            panel["total_return_3d"].gt(0.0)
            & panel["breadth_change5_rebuilt"].gt(0.0)
            & panel["total_return_index_rebuilt"].gt(panel["price_ma20_rebuilt"])
        )
        repair_anchor: pd.Timestamp | pd.NaT = pd.NaT
        for position in range(start, search_end + 1):
            previous = False if position == 0 else bool(repair_condition.iloc[position - 1])
            if not previous and bool(repair_condition.iloc[position]):
                repair_anchor = panel.loc[position, "date"]
                break
        candidates = panel.loc[start:search_end]
        candidates = candidates.loc[candidates["exhaustion_candidate"]]
        timing_candidate = pd.NaT if candidates.empty else candidates["date"].iloc[0]
        if pd.isna(repair_anchor) != pd.isna(event.independent_repair_anchor_date):
            return False
        if pd.notna(repair_anchor) and pd.Timestamp(event.independent_repair_anchor_date) != repair_anchor:
            return False
        if pd.isna(timing_candidate) != pd.isna(event.exhaustion_timing_candidate_date):
            return False
        expected_lead = (
            None
            if pd.isna(repair_anchor) or pd.isna(timing_candidate)
            else float(positions[repair_anchor] - positions[timing_candidate])
        )
        if not _same_number(event.exhaustion_timing_lead_rows, expected_lead):
            return False
        if expected_id in a_by_event.index:
            outcome = outcome_by_date.loc[episode.start_date]
            expected_label = int(
                float(outcome["future_mae_magnitude_10d"])
                >= float(config["prediction_questions"]["question_a_left_tail"]["major_mae_threshold"])
            )
            if int(a_by_event.loc[expected_id, "major_left_tail_10d"]) != expected_label:
                return False
        if expected_id in b_by_event.index:
            row = b_by_event.loc[expected_id]
            candidate_date = pd.Timestamp(row["date"])
            if candidate_date != pd.Timestamp(event.exhaustion_candidate_date):
                return False
            outcome = outcome_by_date.loc[candidate_date]
            specification = config["prediction_questions"]["question_b_exhaustion"]
            expected_label = int(
                float(outcome["future_return_20d"])
                >= float(specification["terminal_return_threshold"])
                and float(outcome["future_positive_wealth_fraction_20d"])
                >= float(specification["positive_wealth_fraction_threshold"])
                and float(outcome["future_mae_magnitude_20d"])
                <= float(specification["maximum_mae_threshold"])
            )
            if int(row["sustained_repair_20d"]) != expected_label:
                return False
    return True


def _fit(frame: pd.DataFrame, predictors: list[str], target: str) -> LogisticRegression:
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    model.fit(frame[predictors].to_numpy(dtype=float), frame[target].to_numpy(dtype=int))
    return model


def _verify_model_core(
    config: dict[str, Any],
    frame: pd.DataFrame,
    result: dict[str, Any],
    predictors: list[str],
    target: str,
    primary: str,
) -> bool:
    if not result["sample_gate_passed"]:
        return result["model_gate_passed"] is False
    fitted = _fit(frame, predictors, target)
    for index, predictor in enumerate(predictors):
        if not _same_number(result["coefficients"][predictor], fitted.coef_[0][index]):
            return False
    aucs: list[float] = []
    for era in config["dates"]["eras"]:
        train = frame.loc[~frame["era"].eq(era)]
        test = frame.loc[frame["era"].eq(era)]
        if train[target].nunique() < 2 or test[target].nunique() < 2:
            continue
        model = _fit(train, predictors, target)
        probability = model.predict_proba(test[predictors].to_numpy(dtype=float))[:, 1]
        aucs.append(float(roc_auc_score(test[target], probability)))
    median_auc = None if not aucs else float(np.median(aucs))
    if len(aucs) != int(result["leave_one_era_out_evaluable_eras"]):
        return False
    if not _same_number(median_auc, result["leave_one_era_out_median_auc"]):
        return False
    primary_index = predictors.index(primary)
    deletion_pass = True
    for event_id in frame["event_id"]:
        sample = frame.loc[~frame["event_id"].eq(event_id)]
        if sample[target].nunique() < 2 or _fit(sample, predictors, target).coef_[0][primary_index] <= 0.0:
            deletion_pass = False
            break
    if deletion_pass != bool(result["delete_any_event_preserves_primary_direction"]):
        return False
    interval = result["primary_bootstrap_90pct_ci"]
    positive_ci = interval[0] is not None and float(interval[0]) > 0.0
    cv_pass = bool(
        len(aucs) >= int(config["prediction_questions"]["estimation"]["minimum_evaluable_eras"])
        and median_auc is not None
        and median_auc >= float(config["prediction_questions"]["estimation"]["auc_threshold"])
    )
    return bool(
        positive_ci == result["primary_positive_ci_passed"]
        and cv_pass == result["leave_one_era_out_auc_gate_passed"]
        and bool(result["model_gate_passed"]) == bool(positive_ci and cv_pass)
    )


def _direction(
    values: pd.Series,
    expected: str,
    minimum: int,
    fraction_gate: float,
    pvalue_gate: float,
    median_gate: float = 0.0,
) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    oriented = clean if expected == "positive" else -clean
    positive = int(oriented.gt(0.0).sum())
    fraction = None if len(clean) == 0 else positive / len(clean)
    pvalue = None if len(clean) == 0 else float(
        binomtest(positive, len(clean), 0.5, alternative="greater").pvalue
    )
    passed = bool(
        len(clean) >= minimum
        and float(oriented.median()) >= median_gate
        and fraction is not None
        and fraction >= fraction_gate
        and pvalue is not None
        and pvalue <= pvalue_gate
    )
    return {
        "observations": int(len(clean)),
        "median": None if clean.empty else float(clean.median()),
        "direction_fraction": fraction,
        "one_sided_sign_pvalue": pvalue,
        "passed": passed,
    }


def _verify_leading_results(
    config: dict[str, Any],
    price_events: pd.DataFrame,
    router_events: pd.DataFrame,
    stored: pd.DataFrame,
) -> bool:
    gate = config["gates"]["leading_gate"]
    complete = price_events.loc[price_events["future_20d_complete"].map(_as_bool)]
    channels = {
        "PRESSURE_SOURCE": "source_lead_rows",
        "TRANSMISSION": "transmission_lead_rows",
        "CROSS_INDEX_ETF": "cross_index_etf_lead_rows",
        "SAME_INDEX_ETF": "same_index_etf_lead_rows",
        "CSI300_COMPONENT_BREADTH": "breadth_lead_rows",
        "IF_RAW_BASIS": "if_raw_basis_lead_rows",
        "H3_DOWNSIDE": "h3_downside_lead_rows",
    }
    expected: dict[str, dict[str, Any]] = {}
    for channel, column in channels.items():
        expected[channel] = _direction(
            complete[column],
            "positive",
            int(gate["minimum_valid_price_damage_events"]),
            float(gate["minimum_positive_lead_fraction"]),
            float(gate["maximum_one_sided_sign_test_pvalue"]),
            float(gate["minimum_median_lead_trading_rows"]),
        )
    expected["EXHAUSTION_TO_REPAIR"] = _direction(
        router_events["exhaustion_timing_lead_rows"],
        "positive",
        int(gate["question_b_minimum_rows"]),
        float(gate["minimum_positive_lead_fraction"]),
        float(gate["maximum_one_sided_sign_test_pvalue"]),
        float(gate["minimum_median_lead_trading_rows"]),
    )
    stored_map = stored.set_index("channel")
    if set(stored_map.index) != set(expected):
        return False
    for channel, values in expected.items():
        row = stored_map.loc[channel]
        if int(row["observations"]) != values["observations"]:
            return False
        for column in ["median", "direction_fraction", "one_sided_sign_pvalue"]:
            if not _same_number(row[column], values[column]):
                return False
        if _as_bool(row["passed"]) != values["passed"]:
            return False
    return True


def _verify_mechanism_gate(
    config: dict[str, Any], trajectories: pd.DataFrame, report_gate: dict[str, Any]
) -> bool:
    gate = config["gates"]["mechanism_gate"]
    specifications = {
        "transmission_onset": ("transmission_onset_delta", "positive"),
        "exhaustion_late_vs_early": ("exhaustion_late_minus_early", "positive"),
        "marginal_price_impact_late_vs_early": (
            "marginal_price_impact_late_minus_early",
            "negative",
        ),
    }
    full_pass = True
    for key, (column, expected_direction) in specifications.items():
        expected = _direction(
            trajectories[column],
            expected_direction,
            int(gate["minimum_events_full_sample"]),
            float(gate["minimum_direction_fraction"]),
            float(gate["maximum_one_sided_sign_test_pvalue"]),
        )
        stored = report_gate["full_sample"][key]
        full_pass &= expected["passed"]
        if int(stored["observations"]) != expected["observations"]:
            return False
        for field in ["median", "direction_fraction", "one_sided_sign_pvalue"]:
            if not _same_number(stored[field], expected[field]):
                return False
        if bool(stored["passed"]) != expected["passed"]:
            return False
    era_passing = {"transmission": 0, "exhaustion": 0}
    for era in config["dates"]["eras"]:
        sample = trajectories.loc[trajectories["era"].eq(era)]
        for label, column in [
            ("transmission", "transmission_onset_delta"),
            ("exhaustion", "exhaustion_late_minus_early"),
        ]:
            clean = pd.to_numeric(sample[column], errors="coerce").dropna()
            passed = bool(
                len(clean) >= int(gate["minimum_events_per_era"])
                and float(clean.median()) > 0.0
                and float(clean.gt(0.0).mean()) >= float(gate["minimum_direction_fraction"])
            )
            era_passing[label] += int(passed)
    era_pass = bool(
        era_passing["transmission"] >= int(gate["minimum_eras_transmission_direction"])
        and era_passing["exhaustion"] >= int(gate["minimum_eras_exhaustion_direction"])
    )
    return bool(
        era_passing["transmission"] == report_gate["transmission_eras_passing"]
        and era_passing["exhaustion"] == report_gate["exhaustion_eras_passing"]
        and bool(report_gate["full_sample_passed"]) == bool(full_pass)
        and bool(report_gate["era_replication_passed"]) == era_pass
        and bool(report_gate["passed"]) == bool(full_pass and era_pass)
    )


def _render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# 510300 压力传导与耗竭机制图谱 V1 聚焦复核",
        "",
        f"- 复核状态：`{audit['status']}`",
        f"- 正式状态：`{audit['formal_result_status']}`",
        f"- 失败检查数：{len(audit['failed_checks'])}",
        "- 范围：只复核冻结、点时/未来隔离、T+1路径、独立事件、时序符号、模型核心量与最终裁决。",
        "- 未重复：1000次自助抽样全流程、全项目扫描、泛安全审计、无关旧产物审计。",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
    ]
    for name, passed in audit["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.append("")
    return "\n".join(lines)


def audit() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    paths = {key: _project_path(value) for key, value in config["paths"].items()}
    required_keys = [
        "manifest",
        "freeze_receipt",
        "input_audit",
        "mechanism_panel",
        "outcome_panel",
        "router_event_atlas",
        "price_damage_events",
        "event_trajectories",
        "leading_results",
        "question_a_rows",
        "question_b_rows",
        "result_json",
        "result_markdown",
    ]
    missing = [config["paths"][key] for key in required_keys if not paths[key].exists()]
    if missing:
        raise FileNotFoundError(f"缺少正式研究产物：{missing}")

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    result = json.loads(paths["result_json"].read_text(encoding="utf-8"))
    input_audit = json.loads(paths["input_audit"].read_text(encoding="utf-8"))
    panel = _normalize_date(pd.read_parquet(paths["mechanism_panel"]))
    outcomes = _normalize_date(pd.read_parquet(paths["outcome_panel"]))
    router_events = pd.read_csv(
        paths["router_event_atlas"], parse_dates=[
            "start_date",
            "end_date",
            "exhaustion_candidate_date",
            "exhaustion_timing_candidate_date",
            "independent_repair_anchor_date",
        ]
    )
    price_events = pd.read_csv(paths["price_damage_events"])
    for column in [item for item in price_events.columns if item.endswith("_date")]:
        price_events[column] = pd.to_datetime(price_events[column], errors="coerce")
    trajectories = pd.read_csv(paths["event_trajectories"])
    leading = pd.read_csv(paths["leading_results"])
    question_a = pd.read_csv(paths["question_a_rows"], parse_dates=["date"])
    question_b = pd.read_csv(paths["question_b_rows"], parse_dates=["date"])

    hash_mismatches: dict[str, dict[str, str]] = {}
    for group in ["tracked_files", "input_files"]:
        for relative, expected in manifest[group].items():
            path = _project_path(relative)
            actual = _sha256(path) if path.exists() else "MISSING"
            if actual != expected:
                hash_mismatches[relative] = {"expected": expected, "actual": actual}

    outcome_ok, outcome_mismatches = _verify_outcomes(config, panel, outcomes)
    price_ok = _verify_price_events(config, panel, price_events)
    router_ok = _verify_router_events(
        config, panel, outcomes, router_events, question_a, question_b
    )
    question_a_ok = _verify_model_core(
        config,
        question_a,
        result["prediction_questions"]["QUESTION_A_LEFT_TAIL"],
        ["pressure_source_score", "transmission_score"],
        "major_left_tail_10d",
        "transmission_score",
    )
    question_b_ok = _verify_model_core(
        config,
        question_b,
        result["prediction_questions"]["QUESTION_B_EXHAUSTION"],
        ["transmission_score", "exhaustion_score"],
        "sustained_repair_20d",
        "exhaustion_score",
    )
    leading_ok = _verify_leading_results(config, price_events, router_events, leading)
    mechanism_ok = _verify_mechanism_gate(
        config, trajectories, result["gates"]["mechanism_gate"]
    )

    leading_map = leading.set_index("channel")["passed"].map(_as_bool).to_dict()
    question_a_result = result["prediction_questions"]["QUESTION_A_LEFT_TAIL"]
    question_b_result = result["prediction_questions"]["QUESTION_B_EXHAUSTION"]
    leading_gate = bool(
        leading_map.get("TRANSMISSION", False)
        and leading_map.get("EXHAUSTION_TO_REPAIR", False)
        and question_a_result["model_gate_passed"]
        and question_b_result["model_gate_passed"]
    )
    event_gate = bool(
        len(router_events) >= int(config["gates"]["event_gate"]["minimum_complete_router_events"])
        and question_a_result["delete_any_event_preserves_primary_direction"]
        and question_b_result["delete_any_event_preserves_primary_direction"]
    )
    external_channels = config["gates"]["external_replication_gate"]["channels"]
    external_passes = {channel: bool(leading_map.get(channel, False)) for channel in external_channels}
    external_gate = bool(
        sum(external_passes.values())
        >= int(config["gates"]["external_replication_gate"]["minimum_channels_passing_leading_rule"])
        and external_passes["CROSS_INDEX_ETF"]
        and external_passes["CSI300_COMPONENT_BREADTH"]
    )
    all_pass = bool(
        leading_gate and event_gate and result["gates"]["mechanism_gate"]["passed"] and external_gate
    )
    expected_status = (
        config["adjudication"]["if_all_four_pass"]
        if all_pass
        else config["adjudication"]["if_leading_gate_fails"]
        if not leading_gate
        else config["adjudication"]["if_other_gate_fails"]
    )

    future_columns = [
        column
        for column in panel.columns
        if any(token in column.lower() for token in ["future", "forward", "target_return"])
    ]
    warmup = panel["mechanism_feature_availability"].eq("NO_VIEW_WARMUP")
    checks = {
        "freeze_and_bound_hashes": bool(
            manifest.get("frozen_before_new_atlas_outcome_read") is True
            and manifest.get("new_atlas_outcome_read_before_freeze") is False
            and not hash_mismatches
            and result.get("manifest_sha256") == _sha256(paths["manifest"])
        ),
        "input_data_contract_and_upstream_boundaries": bool(
            input_audit.get("passed") is True
            and input_audit.get("boundaries", {}).get("old_router_modified") is False
            and input_audit.get("boundaries", {}).get("systemic_holdout_opened") is False
        ),
        "mechanism_panel_clock_and_future_isolation": bool(
            len(panel) == int(config["data_gates"]["expected_signal_rows"])
            and not future_columns
            and warmup.sum() <= int(config["data_gates"]["maximum_leading_feature_warmup_rows"])
            and warmup.iloc[: int(warmup.sum())].all()
            and not warmup.iloc[int(warmup.sum()) :].any()
            and not panel.loc[~warmup, ["pressure_source_score", "transmission_score", "exhaustion_score"]].isna().any(axis=None)
        ),
        "all_t_plus_one_event_paths_recomputed": outcome_ok,
        "independent_price_damage_events_and_signed_timing_recomputed": price_ok,
        "router_events_equal_weight_and_labels_recomputed": router_ok,
        "question_a_full_model_cv_and_delete_event_recomputed": question_a_ok,
        "question_b_full_model_cv_and_delete_event_recomputed": question_b_ok,
        "leading_synchronous_lagging_summaries_recomputed": leading_ok,
        "mechanism_directions_and_era_replication_recomputed": mechanism_ok,
        "four_gates_and_formal_status_recomputed": bool(
            result["gates"]["leading_gate"]["passed"] == leading_gate
            and result["gates"]["event_gate"]["passed"] == event_gate
            and result["gates"]["external_replication_gate"]["passed"] == external_gate
            and result["gates"]["all_four_gates_passed"] == all_pass
            and result["status"] == expected_status
        ),
        "no_strategy_or_execution_promotion": bool(
            result["strategy_backtest"] == "NOT_RUN"
            and set(result["portfolio_metrics"].values()) == {"NOT_COMPUTED"}
            and result["boundaries"]["portfolio_mapping"] == "DISABLED"
            and result["boundaries"]["current_signal"] == "NOT_CREATED"
            and result["boundaries"]["order_generation"] == "DISABLED"
            and result["boundaries"]["live_trading_authorized"] is False
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    audit_result = {
        "project_id": config["protocol"]["project_id"],
        "status": (
            "PASS_FOCUSED_MECHANISM_OUTPUT_RECONCILIATION"
            if not failed
            else "FAIL_FOCUSED_MECHANISM_OUTPUT_RECONCILIATION"
        ),
        "formal_result_status": result["status"],
        "checks": checks,
        "failed_checks": failed,
        "hash_mismatches": hash_mismatches,
        "outcome_mismatches": outcome_mismatches,
        "audit_scope": {
            "bootstrap_1000_full_repetition_recomputed": False,
            "bootstrap_reported_interval_gate_consistency_checked": True,
            "full_project_scan": False,
            "generic_security_or_privacy_scan": False,
            "unrelated_legacy_artifact_audit": False,
        },
        "boundaries": {
            "formal_research_module_imported": False,
            "strategy_returns_recomputed": False,
            "systemic_holdout_opened": False,
            "position_mapping": "DISABLED",
            "current_signal": "NOT_CREATED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(paths["output_audit_json"], audit_result)
    _atomic_text(paths["output_audit_markdown"], _render_markdown(audit_result))
    return audit_result


def main() -> int:
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0 if result["status"] == "PASS_FOCUSED_MECHANISM_OUTPUT_RECONCILIATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
