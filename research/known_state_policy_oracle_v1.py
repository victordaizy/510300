"""510300状态已知条件策略Oracle V1。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from backtest.engine import (  # noqa: E402
    BacktestCosts,
    run_long_cash_backtest,
    summarize_backtest,
)
from three_state_trend_router_v1_0_1 import (  # noqa: E402
    ContractError,
    load_inputs as load_daily_inputs,
)


CONFIG_PATH = ROOT / "config" / "510300_known_state_policy_oracle_v1.yaml"
MANIFEST_PATH = (
    ROOT / "config" / "510300_known_state_policy_oracle_v1_0_1_correction_manifest.json"
)

STATE_BULL = "BULL_TREND"
STATE_BEAR = "BEAR_TREND"
STATE_RANGE = "RANGE"
STATE_CENSORED = "CENSORED"
COMPLETE_STATES = (STATE_BULL, STATE_BEAR, STATE_RANGE)
POLICY_PRIMARY = "STATE_ROUTER_RANGE_NO_T"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    checks = {
        "project_id": config["protocol"]["project_id"]
        == "510300_KNOWN_STATE_POLICY_ORACLE_V1",
        "oracle_future": config["protocol"]["oracle_future_information_used"] is True,
        "no_predictability": config["protocol"]["state_predictability_evaluated"] is False,
        "asset": config["scope"]["execution_asset"] == "510300.SH",
        "holdings": config["scope"]["allowed_holdings"]
        == ["510300.SH", "CASH_CNY"],
        "window": config["oracle_state_definition"]["decision_horizon_trading_days"]
        == 20,
        "primary_threshold": config["oracle_state_definition"][
            "primary_absolute_threshold"
        ]
        == 0.05,
        "robustness": config["oracle_state_definition"][
            "robustness_absolute_thresholds"
        ]
        == [0.03, 0.07],
        "primary_policy": config["policies"][POLICY_PRIMARY]
        == {STATE_BULL: 1.0, STATE_BEAR: 0.0, STATE_RANGE: 0.0},
        "capital": config["execution"]["initial_capital_cny"] == 20000.0,
        "lot": config["execution"]["lot_size_shares"] == 100,
        "base_slippage": config["execution"]["base_costs"][
            "slippage_bps_per_leg"
        ]
        == 5.0,
        "stress_slippage": config["execution"]["stress_costs"][
            "slippage_bps_per_leg"
        ]
        == 10.0,
        "target": config["research_question"]["target_net_sharpe"] == 1.2,
        "prior_manifests": config["selection_bias"]["prior_manifest_count"] == 350,
        "total_trials": config["selection_bias"][
            "expected_total_trial_count_including_current"
        ]
        == 352,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ContractError(f"状态已知Oracle冻结配置异常：{failed}")
    forbidden_scope = (
        "leverage_allowed",
        "short_selling_allowed",
        "derivatives_execution_allowed",
        "paper_or_live_signal_allowed",
    )
    if any(bool(config["scope"][key]) for key in forbidden_scope):
        raise ContractError("Oracle不得授权杠杆、卖空、衍生品或信号")
    rescue_fields = (
        "threshold_rescue_after_result",
        "horizon_rescue_after_result",
        "policy_mapping_rescue_after_result",
        "range_strategy_rescue_after_result",
    )
    if any(config["protocol"][key] != "forbidden" for key in rescue_fields):
        raise ContractError("Oracle结果后救援开关没有全部冻结")


def validate_manifest(config: dict[str, Any]) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise ContractError("Oracle冻结清单不存在，禁止计算未来状态结果")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("project_id") != config["protocol"]["project_id"]:
        raise ContractError("Oracle冻结清单项目编号不匹配")
    if manifest.get("state") != "FROZEN_BEFORE_ORACLE_OUTCOME_CALCULATION":
        raise ContractError("Oracle冻结清单状态不允许计算")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ContractError("Oracle冻结后配置发生漂移")
    mismatches: dict[str, dict[str, str]] = {}
    for section in ("tracked_files", "input_files"):
        for relative, expected in manifest.get(section, {}).items():
            path = _project_path(relative)
            actual = sha256_file(path) if path.exists() else "MISSING"
            if actual != expected:
                mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ContractError(f"Oracle冻结文件发生漂移：{mismatches}")
    if manifest.get("oracle_outcomes_read_before_freeze") is not False:
        raise ContractError("Oracle冻结前结果可见性证明失败")
    return manifest


def classify_oracle_state(future_return: float, threshold: float) -> str:
    if threshold <= 0:
        raise ValueError("状态阈值必须为正数")
    if not math.isfinite(future_return):
        return STATE_CENSORED
    if future_return >= threshold:
        return STATE_BULL
    if future_return <= -threshold:
        return STATE_BEAR
    return STATE_RANGE


def build_oracle_blocks(
    execution_dates: pd.Series,
    total_return_index: pd.DataFrame,
    threshold: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(execution_dates, errors="raise")).reset_index(drop=True)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ContractError("Oracle执行日历重复或未升序")
    expected_rows = int(config["data_contract"]["expected_evaluation_trading_days"])
    if len(dates) != expected_rows:
        raise ContractError(f"Oracle评价日历应为{expected_rows}日，实际{len(dates)}日")
    tri = total_return_index[["date", "close"]].copy()
    tri["date"] = pd.to_datetime(tri["date"], errors="raise").dt.normalize()
    tri["close"] = pd.to_numeric(tri["close"], errors="coerce")
    tri = tri.drop_duplicates("date").set_index("date")
    closes = tri.reindex(dates)["close"]
    if closes.isna().any() or (closes <= 0).any():
        raise ContractError("Oracle日历缺少有效H00300全收益点位")
    horizon = int(config["oracle_state_definition"]["decision_horizon_trading_days"])
    rows: list[dict[str, Any]] = []
    block_id = 0
    for anchor in range(0, len(dates), horizon):
        complete = anchor + horizon < len(dates)
        signal_date = pd.Timestamp(dates.iloc[anchor])
        if complete:
            end_position = anchor + horizon
            end_date = pd.Timestamp(dates.iloc[end_position])
            future_return = float(closes.iloc[end_position] / closes.iloc[anchor] - 1.0)
            state = classify_oracle_state(future_return, threshold)
            execution_start = pd.Timestamp(dates.iloc[anchor + 1])
            execution_days = horizon
        else:
            end_position = len(dates) - 1
            end_date = pd.Timestamp(dates.iloc[end_position])
            future_return = math.nan
            state = STATE_CENSORED
            execution_start = (
                pd.Timestamp(dates.iloc[anchor + 1]) if anchor + 1 < len(dates) else pd.NaT
            )
            execution_days = max(len(dates) - anchor - 1, 0)
        rows.append(
            {
                "threshold": float(threshold),
                "block_id": int(block_id),
                "anchor_position": int(anchor),
                "signal_date": signal_date,
                "execution_start": execution_start,
                "execution_end": end_date,
                "execution_trading_days": int(execution_days),
                "oracle_future_20d_total_return": future_return,
                "state": state,
                "complete": bool(complete),
                "uses_future_information": True,
            }
        )
        block_id += 1
    blocks = pd.DataFrame(rows)
    complete_count = int(blocks["complete"].sum())
    censored_count = int((~blocks["complete"]).sum())
    if complete_count != int(config["data_contract"]["expected_complete_oracle_blocks"]):
        raise ContractError("完整Oracle区间数量不符合冻结合同")
    if censored_count != int(config["data_contract"]["expected_censored_blocks"]):
        raise ContractError("删失Oracle区间数量不符合冻结合同")
    trailing = int(blocks.loc[~blocks["complete"], "execution_trading_days"].sum())
    if trailing != int(
        config["data_contract"]["expected_trailing_days_after_censored_anchor"]
    ):
        raise ContractError("末尾删失区间交易日数不符合冻结合同")
    return blocks


def build_targets(
    blocks: pd.DataFrame, policy_mapping: dict[str, float], policy_name: str
) -> pd.DataFrame:
    if set(policy_mapping) != set(COMPLETE_STATES):
        raise ContractError(f"策略{policy_name}没有完整三态映射")
    targets = blocks[["signal_date", "state", "complete"]].rename(
        columns={"signal_date": "date"}
    )
    targets = targets.copy()
    targets["target_position"] = targets["state"].map(policy_mapping).fillna(0.0)
    targets["risk_off_override"] = targets["target_position"].eq(0.0)
    targets["trade_allowed"] = True
    targets["signal_reason"] = (
        "已知状态Oracle::" + policy_name + "::" + targets["state"].astype(str)
    )
    return targets


def _costs(config: dict[str, Any], name: str) -> BacktestCosts:
    values = config["execution"][name]
    return BacktestCosts(
        commission_rate=float(values["commission_rate_per_leg"]),
        minimum_commission_cny=float(values["minimum_commission_cny_per_leg"]),
        stamp_duty_rate=float(values["stamp_duty_rate"]),
        slippage_bps=float(values["slippage_bps_per_leg"]),
        lot_size=int(config["execution"]["lot_size_shares"]),
        cash_annual_rate=float(config["execution"]["cash_annual_rate"]),
    )


def _summary(
    ledger: pd.DataFrame, trades: pd.DataFrame, initial_capital: float
) -> dict[str, Any]:
    result = dict(summarize_backtest(ledger, trades, initial_capital))
    result["total_slippage_cost_cny"] = float(ledger["daily_slippage_cost_cny"].sum())
    result["total_execution_cost_cny"] = float(
        ledger["daily_total_execution_cost_cny"].sum()
    )
    return result


def _period_metrics(
    ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    frame = ledger.loc[ledger["date"].between(start, end)]
    returns = frame["daily_return"].to_numpy(dtype=float)
    if len(returns) == 0:
        return {"observations": 0, "total_return": None, "net_sharpe": None}
    volatility = float(np.std(returns, ddof=1) * math.sqrt(242.0))
    annualized_mean = float(np.mean(returns) * 242.0)
    wealth = np.cumprod(1.0 + returns)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return {
        "observations": int(len(returns)),
        "total_return": float(np.prod(1.0 + returns) - 1.0),
        "annualized_mean_return": annualized_mean,
        "annualized_volatility": volatility,
        "net_sharpe": annualized_mean / volatility if volatility > 0 else None,
        "maximum_drawdown": float(drawdown.min()),
    }


def evaluate_policy(
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    initial = float(config["execution"]["initial_capital_cny"])
    start = config["dates"]["evaluation_start"]
    end = config["dates"]["evaluation_end"]
    base_ledger, base_trades = run_long_cash_backtest(
        etf_daily, dividends, targets, initial, _costs(config, "base_costs"), start, end
    )
    stress_ledger, stress_trades = run_long_cash_backtest(
        etf_daily,
        dividends,
        targets,
        initial,
        _costs(config, "stress_costs"),
        start,
        end,
    )
    recent_start, recent_end = (
        pd.Timestamp(value) for value in config["dates"]["recent_period"]
    )
    report = {
        "base": _summary(base_ledger, base_trades, initial),
        "stress": _summary(stress_ledger, stress_trades, initial),
        "recent_period": {
            "start": str(recent_start.date()),
            "end": str(recent_end.date()),
            "base": _period_metrics(base_ledger, recent_start, recent_end),
            "stress": _period_metrics(stress_ledger, recent_start, recent_end),
        },
    }
    frames = {
        "base_ledger": base_ledger,
        "base_trades": base_trades,
        "stress_ledger": stress_ledger,
        "stress_trades": stress_trades,
    }
    return report, frames


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    return "NA" if value is None else f"{float(value):.{digits}%}"


def render_markdown(report: dict[str, Any]) -> str:
    primary = report["threshold_results"]["0.05"]
    primary_policy = primary["policies"][POLICY_PRIMARY]
    lines = [
        "# 510300 状态已知条件策略 Oracle V1 结果",
        "",
        f"- 状态：`{report['status']}`",
        "- 问题：假设未来20日市场状态已被完美告知，分阶段动作能否达到净夏普1.2？",
        "- Oracle：明确使用未来H00300全收益；不检验状态预测，不是可交易策略。",
        "- 主动作：上涨满仓510300，下跌空仓，震荡不交易。",
        "",
        "## 主口径（未来20日涨跌阈值 ±5%）",
        "",
        "| 策略映射 | 基准成本净夏普 | 压力成本净夏普 | 总收益（压力） | 最大回撤（基准） |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy_name, values in primary["policies"].items():
        lines.append(
            f"| `{policy_name}` | {_fmt(values['base']['sharpe_zero_cash_rate'])} | "
            f"{_fmt(values['stress']['sharpe_zero_cash_rate'])} | "
            f"{_pct(values['stress']['total_return'])} | "
            f"{_pct(values['base']['max_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            "## 三种状态数量",
            "",
            "| 状态 | 完整20日区间数 |",
            "|---|---:|",
        ]
    )
    for state in COMPLETE_STATES:
        lines.append(f"| `{state}` | {primary['state_counts'][state]} |")
    lines.extend(
        [
            "",
            "## 稳健性",
            "",
            "| 绝对阈值 | 主映射基准净夏普 | 主映射压力净夏普 |",
            "|---:|---:|---:|",
        ]
    )
    for key, values in report["threshold_results"].items():
        policy = values["policies"][POLICY_PRIMARY]
        lines.append(
            f"| ±{float(key):.0%} | {_fmt(policy['base']['sharpe_zero_cash_rate'])} | "
            f"{_fmt(policy['stress']['sharpe_zero_cash_rate'])} |"
        )
    lines.extend(
        [
            "",
            f"- 条件命题：`{'PASS' if report['adjudication']['conditional_policy_hypothesis_supported'] else 'FAIL'}`。",
            "- 状态预测命题：`NOT_EVALUATED`。",
            "- 历史可交易策略目标：`NOT_ACHIEVED`。",
            "- 独立前向目标：`NOT_ACHIEVED`。",
            "",
            "## 正确解释",
            "",
            "若条件命题通过，只能说明高夏普主要取决于状态信息质量，且牛满仓、熊/震荡空仓的动作映射在经济上成立。",
            "它不能说明现实技术因子已经可以提前知道状态，也不生成Paper、Shadow、订单或实盘授权。",
            "",
        ]
    )
    return "\n".join(lines)


def run_study(write: bool = True) -> dict[str, Any]:
    config = load_config()
    manifest = validate_manifest(config)
    etf, _index, tri, dividends, _audit = load_daily_inputs(config)
    start = pd.Timestamp(config["dates"]["evaluation_start"])
    end = pd.Timestamp(config["dates"]["evaluation_end"])
    evaluation_dates = etf.loc[etf["date"].between(start, end), "date"].reset_index(
        drop=True
    )
    thresholds = sorted(
        [float(config["oracle_state_definition"]["primary_absolute_threshold"])]
        + [
            float(value)
            for value in config["oracle_state_definition"][
                "robustness_absolute_thresholds"
            ]
        ]
    )
    threshold_results: dict[str, Any] = {}
    all_blocks: list[pd.DataFrame] = []
    primary_targets: pd.DataFrame | None = None
    primary_frames: dict[str, pd.DataFrame] | None = None
    for threshold in thresholds:
        blocks = build_oracle_blocks(evaluation_dates, tri, threshold, config)
        all_blocks.append(blocks)
        state_counts = {
            state: int(((blocks["state"] == state) & blocks["complete"]).sum())
            for state in COMPLETE_STATES
        }
        policies: dict[str, Any] = {}
        for policy_name, policy_mapping in config["policies"].items():
            targets = build_targets(blocks, policy_mapping, policy_name)
            policy_report, frames = evaluate_policy(etf, dividends, targets, config)
            policies[policy_name] = policy_report
            if threshold == 0.05 and policy_name == POLICY_PRIMARY:
                primary_targets = targets
                primary_frames = frames
        threshold_results[_threshold_key(threshold)] = {
            "threshold": threshold,
            "complete_blocks": int(blocks["complete"].sum()),
            "censored_blocks": int((~blocks["complete"]).sum()),
            "state_counts": state_counts,
            "state_future_return_statistics": {
                state: {
                    "mean": float(
                        blocks.loc[
                            blocks["complete"] & blocks["state"].eq(state),
                            "oracle_future_20d_total_return",
                        ].mean()
                    ),
                    "minimum": float(
                        blocks.loc[
                            blocks["complete"] & blocks["state"].eq(state),
                            "oracle_future_20d_total_return",
                        ].min()
                    ),
                    "maximum": float(
                        blocks.loc[
                            blocks["complete"] & blocks["state"].eq(state),
                            "oracle_future_20d_total_return",
                        ].max()
                    ),
                }
                for state in COMPLETE_STATES
            },
            "policies": policies,
        }
    if primary_targets is None or primary_frames is None:
        raise ContractError("主Oracle口径没有生成执行结果")
    primary = threshold_results["0.05"]
    primary_policy = primary["policies"][POLICY_PRIMARY]
    all_long = primary["policies"]["ALL_LONG"]
    all_long_drawdown = abs(float(all_long["base"]["max_drawdown"]))
    drawdown_ratio = (
        abs(float(primary_policy["base"]["max_drawdown"])) / all_long_drawdown
        if all_long_drawdown > 0
        else None
    )
    gates_config = config["conditional_hypothesis_gates"]
    gates = {
        "primary_base_net_sharpe": float(
            primary_policy["base"]["sharpe_zero_cash_rate"]
        )
        >= float(gates_config["primary_base_net_sharpe_minimum"]),
        "primary_stress_net_sharpe": float(
            primary_policy["stress"]["sharpe_zero_cash_rate"]
        )
        >= float(gates_config["primary_stress_net_sharpe_minimum"]),
        "every_robustness_stress_net_sharpe": all(
            float(threshold_results[key]["policies"][POLICY_PRIMARY]["stress"][
                "sharpe_zero_cash_rate"
            ])
            >= float(gates_config["every_robustness_stress_net_sharpe_minimum"])
            for key in ("0.03", "0.07")
        ),
        "primary_stress_total_return_positive": float(
            primary_policy["stress"]["total_return"]
        )
        > 0,
        "primary_drawdown_ratio_vs_all_long": (
            drawdown_ratio is not None
            and drawdown_ratio
            <= float(
                gates_config["primary_maximum_drawdown_ratio_vs_all_long_maximum"]
            )
        ),
        "minimum_complete_blocks_per_state": all(
            primary["state_counts"][state]
            >= int(gates_config["minimum_complete_blocks_per_state"])
            for state in COMPLETE_STATES
        ),
    }
    conditional_pass = bool(all(gates.values()))
    status = (
        "PASS_CONDITIONAL_POLICY_VALUE_GIVEN_PERFECT_STATE_KNOWLEDGE_NOT_TRADABLE"
        if conditional_pass
        else "FAIL_CONDITIONAL_POLICY_VALUE_GIVEN_PERFECT_STATE_KNOWLEDGE"
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "status": status,
        "evidence_class": config["protocol"]["evidence_class"],
        "research_question": config["research_question"],
        "oracle_information_boundary": {
            "future_information_used": True,
            "state_predictability_evaluated": False,
            "state_available_at_anchor_close_by_assumption": True,
            "execution_at_next_open": True,
            "not_a_tradable_signal": True,
        },
        "evaluation_window": {
            "start": config["dates"]["evaluation_start"],
            "end": config["dates"]["evaluation_end"],
            "trading_days": int(len(evaluation_dates)),
        },
        "freeze": {
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "frozen_at_asia_shanghai": manifest["frozen_at_asia_shanghai"],
            "oracle_outcomes_read_before_freeze": False,
        },
        "threshold_results": threshold_results,
        "adjudication": {
            "conditional_policy_hypothesis_supported": conditional_pass,
            "conditional_target_net_sharpe": 1.2,
            "primary_base_net_sharpe": primary_policy["base"][
                "sharpe_zero_cash_rate"
            ],
            "primary_stress_net_sharpe": primary_policy["stress"][
                "sharpe_zero_cash_rate"
            ],
            "primary_drawdown_ratio_vs_all_long": drawdown_ratio,
            "gates": gates,
            "state_prediction_hypothesis": "NOT_EVALUATED",
            "historical_tradable_strategy_target_achieved": False,
            "verified_forward_target_achieved": False,
            "goal_achieved": False,
            "interpretation": (
                "状态完美已知时，牛满仓、熊空仓、震荡不交易的条件策略映射通过高夏普门；现实状态识别仍未解决"
                if conditional_pass
                else "即使状态完美已知，固定分阶段映射仍未通过高夏普门"
            ),
        },
        "governance": {
            "diagnostic_only": True,
            "paper_signal_allowed": False,
            "shadow_signal_allowed": False,
            "position_mapping_enabled": False,
            "order_generation": False,
            "broker_connection": False,
            "position_change": False,
            "live_trading_authorized": False,
        },
        "artifacts": {},
    }
    if write:
        paths = {key: _project_path(value) for key, value in config["paths"].items()}
        for key in (
            "block_table",
            "primary_targets",
            "primary_base_ledger",
            "primary_base_trades",
            "primary_stress_ledger",
            "primary_stress_trades",
            "result_json",
            "result_markdown",
        ):
            paths[key].parent.mkdir(parents=True, exist_ok=True)
        pd.concat(all_blocks, ignore_index=True).to_parquet(paths["block_table"], index=False)
        primary_targets.to_parquet(paths["primary_targets"], index=False)
        primary_frames["base_ledger"].to_parquet(paths["primary_base_ledger"], index=False)
        primary_frames["base_trades"].to_parquet(paths["primary_base_trades"], index=False)
        primary_frames["stress_ledger"].to_parquet(
            paths["primary_stress_ledger"], index=False
        )
        primary_frames["stress_trades"].to_parquet(
            paths["primary_stress_trades"], index=False
        )
        artifact_keys = (
            "block_table",
            "primary_targets",
            "primary_base_ledger",
            "primary_base_trades",
            "primary_stress_ledger",
            "primary_stress_trades",
        )
        report["artifacts"] = {
            key: _artifact_record(paths[key]) for key in artifact_keys
        }
        paths["result_json"].write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        paths["result_markdown"].write_text(render_markdown(report), encoding="utf-8")
    return report
