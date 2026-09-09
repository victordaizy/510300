"""用趋势与全A卖压确认五日尾部风险，减少上涨阶段的误空仓。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from global_liquidity_regime_5d_discovery_v0 import (
    DEVELOPMENT_CUTOFF,
    HORIZON,
    PROJECT_ROOT,
    SELECTION_START,
    _aggregate_offsets,
    _atomic_json,
    _atomic_parquet,
    _evaluate_rule_offset,
    _load_dividends,
    _load_market,
    _safe_float,
    _sha256,
)


FIT_START = pd.Timestamp("2016-08-15")
FIT_END = pd.Timestamp("2018-12-31")

SEVERITY_FILE = (
    PROJECT_ROOT / "data" / "features" / "510300_multi_domain_severity_rank_5d_discovery_v0.parquet"
)
ALL_A_FILE = PROJECT_ROOT / "data" / "features" / "510300_all_a_fragility_shape_5d_discovery_v0.parquet"
OUTPUT_FEATURES = PROJECT_ROOT / "data" / "features" / "510300_conditional_tail_gate_5d_discovery_v0.parquet"
OUTPUT_OFFSETS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_conditional_tail_gate_5d_discovery_v0"
    / "offset_metrics.parquet"
)
OUTPUT_BLOCKS = (
    PROJECT_ROOT
    / "data"
    / "research"
    / "510300_conditional_tail_gate_5d_discovery_v0"
    / "block_predictions.parquet"
)
OUTPUT_REPORT = (
    PROJECT_ROOT / "reports" / "discovery" / "510300_conditional_tail_gate_5d_discovery_v0.json"
)


@dataclass(frozen=True)
class RuleSpec:
    family: str
    logic: str
    decision: Callable[[pd.Series, int], int]


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="raise").dt.normalize().astype("datetime64[ns]")
    output.sort_values("date", inplace=True)
    output.drop_duplicates("date", keep="last", inplace=True)
    return output.reset_index(drop=True)


def _build_frame() -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    dividends = _load_dividends()
    market = _load_market(dividends)
    market = _normalize_dates(market)

    severity = pd.read_parquet(
        SEVERITY_FILE,
        columns=[
            "date",
            "risk_rolling252_all_a_logistic",
            "risk_fit_ecdf_all_a_logistic",
            "risk_rolling252_mean3",
        ],
    )
    severity = _normalize_dates(severity)
    all_a = pd.read_parquet(
        ALL_A_FILE,
        columns=[
            "date",
            "etf_return_1d",
            "etf_return_5d",
            "etf_return_20d",
            "etf_range_position_60d",
            "etf_downside_share_20d",
            "down_amount_share",
            "share_down2",
            "share_close_near_low",
            "cs_mean_ret",
        ],
    )
    all_a = _normalize_dates(all_a)

    frame = market.merge(severity, on="date", how="inner", validate="one_to_one")
    frame = frame.merge(all_a, on="date", how="inner", validate="one_to_one")
    frame = frame[frame["date"] <= DEVELOPMENT_CUTOFF].sort_values("date").reset_index(drop=True)

    fit_mask = frame["date"].between(FIT_START, FIT_END)
    selection_mask = frame["date"].between(SELECTION_START, DEVELOPMENT_CUTOFF)
    if int(fit_mask.sum()) < 500 or int(selection_mask.sum()) < 450:
        raise ValueError("条件尾部闸门的拟合段或选择段覆盖不足")

    thresholds = {
        "down_amount_fit_median": float(frame.loc[fit_mask, "down_amount_share"].median()),
        "down_amount_fit_q80": float(frame.loc[fit_mask, "down_amount_share"].quantile(0.80)),
        "share_down2_fit_q60": float(frame.loc[fit_mask, "share_down2"].quantile(0.60)),
        "share_down2_fit_q80": float(frame.loc[fit_mask, "share_down2"].quantile(0.80)),
        "near_low_fit_q80": float(frame.loc[fit_mask, "share_close_near_low"].quantile(0.80)),
    }

    risk = pd.to_numeric(frame["risk_rolling252_all_a_logistic"], errors="coerce")
    diagnostics = {
        "common_rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "selection_rows": int(selection_mask.sum()),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "selection_risk_available_rows": int((selection_mask & risk.notna()).sum()),
        "selection_feature_missing_rows": {
            column: int(frame.loc[selection_mask, column].isna().sum())
            for column in [
                "risk_rolling252_all_a_logistic",
                "etf_return_1d",
                "etf_return_5d",
                "etf_return_20d",
                "etf_range_position_60d",
                "down_amount_share",
                "share_down2",
                "share_close_near_low",
            ]
        },
    }
    return frame, thresholds, diagnostics


def _tail_and(column: str, risk_cutoff: float, predicate: Callable[[pd.Series], bool]) -> Callable[[pd.Series, int], int]:
    def decide(row: pd.Series, previous: int) -> int:
        del previous
        risk = float(row[column])
        return int(not (risk >= risk_cutoff and predicate(row)))

    return decide


def _tail_baseline(column: str, risk_cutoff: float) -> Callable[[pd.Series, int], int]:
    return _tail_and(column, risk_cutoff, lambda row: True)


def _sticky_tail(
    column: str,
    trigger_cutoff: float,
    recovery_cutoff: float,
    trigger_predicate: Callable[[pd.Series], bool],
    persistence_predicate: Callable[[pd.Series], bool],
) -> Callable[[pd.Series, int], int]:
    def decide(row: pd.Series, previous: int) -> int:
        risk = float(row[column])
        if risk >= trigger_cutoff and trigger_predicate(row):
            return 0
        if previous == 0 and risk >= recovery_cutoff and persistence_predicate(row):
            return 0
        return 1

    return decide


def _make_rules(thresholds: dict[str, float]) -> dict[str, RuleSpec]:
    risk = "risk_rolling252_all_a_logistic"
    trend20 = lambda row: float(row["etf_return_20d"]) < 0.0
    trend5 = lambda row: float(row["etf_return_5d"]) < 0.0
    red_day = lambda row: float(row["etf_return_1d"]) < 0.0
    below_midrange = lambda row: float(row["etf_range_position_60d"]) < 0.50
    heavy_down_amount = lambda row: float(row["down_amount_share"]) >= thresholds["down_amount_fit_q80"]
    broad_down = lambda row: float(row["share_down2"]) >= thresholds["share_down2_fit_q60"]
    severe_broad_down = lambda row: float(row["share_down2"]) >= thresholds["share_down2_fit_q80"]
    near_low = lambda row: float(row["share_close_near_low"]) >= thresholds["near_low_fit_q80"]
    down_amount_above_median = (
        lambda row: float(row["down_amount_share"]) >= thresholds["down_amount_fit_median"]
    )

    rules: dict[str, RuleSpec] = {
        "BASELINE_TAIL05": RuleSpec(
            "BASELINE",
            "全A滚动严重度进入最高5%即空仓",
            _tail_baseline(risk, 0.95),
        ),
        "BASELINE_TAIL10": RuleSpec(
            "BASELINE",
            "全A滚动严重度进入最高10%即空仓",
            _tail_baseline(risk, 0.90),
        ),
        "BASELINE_TAIL20": RuleSpec(
            "BASELINE",
            "全A滚动严重度进入最高20%即空仓",
            _tail_baseline(risk, 0.80),
        ),
        "TAIL10_AND_ETF_TREND20_NEG": RuleSpec(
            "TREND_CONFIRMATION",
            "最高10%风险且510300二十日收益为负才空仓",
            _tail_and(risk, 0.90, trend20),
        ),
        "TAIL20_AND_ETF_TREND20_NEG": RuleSpec(
            "TREND_CONFIRMATION",
            "最高20%风险且510300二十日收益为负才空仓",
            _tail_and(risk, 0.80, trend20),
        ),
        "TAIL10_AND_ETF_TREND5_NEG": RuleSpec(
            "TREND_CONFIRMATION",
            "最高10%风险且510300五日收益为负才空仓",
            _tail_and(risk, 0.90, trend5),
        ),
        "TAIL20_AND_ETF_TREND5_NEG": RuleSpec(
            "TREND_CONFIRMATION",
            "最高20%风险且510300五日收益为负才空仓",
            _tail_and(risk, 0.80, trend5),
        ),
        "TAIL10_AND_ETF_RED_DAY": RuleSpec(
            "ONE_DAY_CONFIRMATION",
            "最高10%风险且510300当日下跌才空仓",
            _tail_and(risk, 0.90, red_day),
        ),
        "TAIL20_AND_ETF_RED_DAY": RuleSpec(
            "ONE_DAY_CONFIRMATION",
            "最高20%风险且510300当日下跌才空仓",
            _tail_and(risk, 0.80, red_day),
        ),
        "TAIL10_AND_RANGE60_BELOW_HALF": RuleSpec(
            "TREND_CONFIRMATION",
            "最高10%风险且处于六十日区间下半部才空仓",
            _tail_and(risk, 0.90, below_midrange),
        ),
        "TAIL20_AND_RANGE60_BELOW_HALF": RuleSpec(
            "TREND_CONFIRMATION",
            "最高20%风险且处于六十日区间下半部才空仓",
            _tail_and(risk, 0.80, below_midrange),
        ),
        "TAIL10_AND_DOWN_AMOUNT_Q80": RuleSpec(
            "BREADTH_CONFIRMATION",
            "最高10%风险且下跌成交额占比不低于拟合段80分位才空仓",
            _tail_and(risk, 0.90, heavy_down_amount),
        ),
        "TAIL20_AND_DOWN_AMOUNT_Q80": RuleSpec(
            "BREADTH_CONFIRMATION",
            "最高20%风险且下跌成交额占比不低于拟合段80分位才空仓",
            _tail_and(risk, 0.80, heavy_down_amount),
        ),
        "TAIL10_AND_SHARE_DOWN2_Q60": RuleSpec(
            "BREADTH_CONFIRMATION",
            "最高10%风险且跌超2%股票占比不低于拟合段60分位才空仓",
            _tail_and(risk, 0.90, broad_down),
        ),
        "TAIL20_AND_SHARE_DOWN2_Q60": RuleSpec(
            "BREADTH_CONFIRMATION",
            "最高20%风险且跌超2%股票占比不低于拟合段60分位才空仓",
            _tail_and(risk, 0.80, broad_down),
        ),
        "TAIL10_AND_NEAR_LOW_Q80": RuleSpec(
            "BREADTH_CONFIRMATION",
            "最高10%风险且收盘近低股票占比不低于拟合段80分位才空仓",
            _tail_and(risk, 0.90, near_low),
        ),
        "TAIL20_AND_NEAR_LOW_Q80": RuleSpec(
            "BREADTH_CONFIRMATION",
            "最高20%风险且收盘近低股票占比不低于拟合段80分位才空仓",
            _tail_and(risk, 0.80, near_low),
        ),
        "TAIL10_AND_TREND20_NEG_AND_DOWN_AMOUNT_MEDIAN": RuleSpec(
            "DOUBLE_CONFIRMATION",
            "最高10%风险、二十日趋势为负且下跌成交额超过拟合段中位数才空仓",
            _tail_and(risk, 0.90, lambda row: trend20(row) and down_amount_above_median(row)),
        ),
        "TAIL20_AND_TREND20_NEG_AND_DOWN_AMOUNT_MEDIAN": RuleSpec(
            "DOUBLE_CONFIRMATION",
            "最高20%风险、二十日趋势为负且下跌成交额超过拟合段中位数才空仓",
            _tail_and(risk, 0.80, lambda row: trend20(row) and down_amount_above_median(row)),
        ),
        "TAIL10_AND_TREND20_NEG_AND_SHARE_DOWN2_Q80": RuleSpec(
            "DOUBLE_CONFIRMATION",
            "最高10%风险、二十日趋势为负且跌超2%股票占比进入拟合段最高20%才空仓",
            _tail_and(risk, 0.90, lambda row: trend20(row) and severe_broad_down(row)),
        ),
        "TAIL20_AND_TREND20_NEG_AND_SHARE_DOWN2_Q80": RuleSpec(
            "DOUBLE_CONFIRMATION",
            "最高20%风险、二十日趋势为负且跌超2%股票占比进入拟合段最高20%才空仓",
            _tail_and(risk, 0.80, lambda row: trend20(row) and severe_broad_down(row)),
        ),
        "TAIL10_AND_TREND20_NEG_OR_DOWN_AMOUNT_Q80": RuleSpec(
            "OR_CONFIRMATION",
            "最高10%风险且二十日趋势为负或下跌成交额极端才空仓",
            _tail_and(risk, 0.90, lambda row: trend20(row) or heavy_down_amount(row)),
        ),
        "TAIL20_AND_TREND20_NEG_OR_DOWN_AMOUNT_Q80": RuleSpec(
            "OR_CONFIRMATION",
            "最高20%风险且二十日趋势为负或下跌成交额极端才空仓",
            _tail_and(risk, 0.80, lambda row: trend20(row) or heavy_down_amount(row)),
        ),
        "STICKY_TAIL10_TREND20_RECOVER50": RuleSpec(
            "HYSTERESIS",
            "最高10%风险且二十日趋势为负触发；风险低于50%或趋势转正后复位",
            _sticky_tail(risk, 0.90, 0.50, trend20, trend20),
        ),
        "STICKY_TAIL20_TREND20_RECOVER50": RuleSpec(
            "HYSTERESIS",
            "最高20%风险且二十日趋势为负触发；风险低于50%或趋势转正后复位",
            _sticky_tail(risk, 0.80, 0.50, trend20, trend20),
        ),
    }
    return rules


def main() -> int:
    required = [SEVERITY_FILE, ALL_A_FILE]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        payload = {"status": "BLOCKED_MISSING_INPUT", "missing": missing}
        _atomic_json(payload, OUTPUT_REPORT)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    dividends = _load_dividends()
    frame, thresholds, feature_audit = _build_frame()
    rules = _make_rules(thresholds)

    # 通用模拟器要求五个诊断列；这里将它们映射到本轮真实决策输入，不参与额外决策。
    frame["prob_logistic_good5"] = 1.0 - frame["risk_fit_ecdf_all_a_logistic"]
    frame["prob_hgb_good5"] = 1.0 - frame["risk_rolling252_mean3"]
    frame["prob_ensemble_good5"] = 1.0 - frame["risk_rolling252_all_a_logistic"]
    frame["prob_logistic_bad_tail5"] = frame["risk_rolling252_all_a_logistic"]
    frame["score_mechanism_risk"] = frame["risk_rolling252_all_a_logistic"]

    offset_rows: list[dict[str, Any]] = []
    block_frames: list[pd.DataFrame] = []
    for rule_name, spec in rules.items():
        for offset in range(HORIZON):
            metrics, blocks = _evaluate_rule_offset(
                frame,
                dividends,
                rule_name,
                spec.decision,
                offset,
            )
            metrics["rule_family"] = spec.family
            metrics["rule_logic"] = spec.logic
            blocks["rule_family"] = spec.family
            offset_rows.append(metrics)
            block_frames.append(blocks)

    offsets = pd.DataFrame(offset_rows)
    blocks = pd.concat(block_frames, ignore_index=True)
    aggregates = _aggregate_offsets(offsets)
    metadata = offsets.groupby("rule", sort=False).first()[["rule_family", "rule_logic"]]
    for item in aggregates:
        item["rule_family"] = str(metadata.loc[item["rule"], "rule_family"])
        item["rule_logic"] = str(metadata.loc[item["rule"], "rule_logic"])
    best = aggregates[0]
    passed = bool(best["development_gate"])

    output_columns = [
        "date",
        "etf_open",
        "etf_close",
        "benchmark_close",
        "future5_full_factor",
        "future5_cash_factor",
        "future5_excess_factor",
        "target_good5",
        "risk_rolling252_all_a_logistic",
        "risk_fit_ecdf_all_a_logistic",
        "risk_rolling252_mean3",
        "etf_return_1d",
        "etf_return_5d",
        "etf_return_20d",
        "etf_range_position_60d",
        "etf_downside_share_20d",
        "down_amount_share",
        "share_down2",
        "share_close_near_low",
        "cs_mean_ret",
    ]
    _atomic_parquet(frame[output_columns], OUTPUT_FEATURES)
    _atomic_parquet(offsets, OUTPUT_OFFSETS)
    _atomic_parquet(blocks, OUTPUT_BLOCKS)

    payload = {
        "status": (
            "DEVELOPMENT_CANDIDATE_FOUND_FREEZE_REQUIRED_NO_VALIDATION_READ"
            if passed
            else "DEVELOPMENT_REJECTED_CONTINUE_SEARCH"
        ),
        "project_id": "510300_CONDITIONAL_TAIL_GATE_5D_DISCOVERY_V0",
        "scope": {
            "execution_asset": "510300.SH",
            "allowed_states": [0, 1],
            "state_meanings": {"0": "CASH_CNY", "1": "FULL_510300"},
            "benchmark": "H00300_TOTAL_RETURN",
            "signal_time": "T日收盘后",
            "execution_time": "T+1交易日开盘",
            "holding_block_trading_days": HORIZON,
        },
        "development_contract": {
            "data_ceiling": DEVELOPMENT_CUTOFF.date().isoformat(),
            "fit_period": [FIT_START.date().isoformat(), FIT_END.date().isoformat()],
            "selection_period": [SELECTION_START.date().isoformat(), DEVELOPMENT_CUTOFF.date().isoformat()],
            "calendar_offsets": list(range(HORIZON)),
            "rule_count": int(len(rules)),
            "threshold_policy": "风险覆盖率固定为5%、10%、20%；收益零点和区间中点为经济阈值；卖压阈值仅由拟合段分位数确定",
            "validation_2021_plus_loaded": False,
            "gate": "压力成本下五个日历错位的总体年化超额和242日滚动超额中位数必须全部不低于20%",
        },
        "fit_only_thresholds": thresholds,
        "feature_audit": feature_audit,
        "best_development_rule": best,
        "all_rule_aggregates": aggregates,
        "all_offset_metrics": offsets.to_dict("records"),
        "development_gate_passed": passed,
        "validation_2021_plus_loaded": False,
        "input_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
            for path in required
        },
        "artifacts": {
            "features": str(OUTPUT_FEATURES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "offset_metrics": str(OUTPUT_OFFSETS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "block_predictions": str(OUTPUT_BLOCKS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "interpretation": (
            "条件尾部闸门通过开发门槛；先冻结规则和输入哈希，再读取2021年后的独立验证。"
            if passed
            else "趋势和卖压确认仍未达到20%开发门槛；保留完整前沿并继续搜索。"
        ),
        "is_trading_signal": False,
        "live_trading_authorized": False,
    }
    _atomic_json(payload, OUTPUT_REPORT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rule_count": len(rules),
                "best_development_rule": best,
                "validation_2021_plus_loaded": False,
                "report": str(OUTPUT_REPORT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
