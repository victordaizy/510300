"""使用统一30/70滞回规则回测第二轮机制因子。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from backtest.engine import BacktestCosts
from backtest.run_registered_factor_backtests import _run_period, _without_frames
from research.registered_factors import build_hysteresis_target


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round2.yaml"
FEATURE_FILE = ROOT / "data" / "features" / "510300_round2_mechanism_dataset.parquet"
RESEARCH_FILE = ROOT / "reports" / "research" / "round2_mechanism_research.json"
ETF_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
JSON_FILE = ROOT / "reports" / "backtest" / "round2_mechanism_backtests.json"
MARKDOWN_FILE = ROOT / "reports" / "backtest" / "round2_mechanism_backtests.md"
LEDGER_ROOT = ROOT / "data" / "processed" / "round2_mechanism_backtests"


def _render_markdown(report: dict) -> str:
    lines = [
        "# 510300第二轮机制因子统一规则回测", "",
        "## 治理口径", "",
        "- 所有因子沿用第一轮同一套30%/70%点时滞回规则，没有逐因子调阈值。",
        "- 信号在收盘后形成，下一交易日开盘执行；执行T+1、100份整手、分红、佣金、最低佣金和滑点。",
        "- 第二轮假设由已观察过的历史结果派生，任何通过项最多标记为探索性前向候选，不能冻结。", "",
        "|因子|D20证据|开发超额|pseudo-OOS超额|pseudo-OOS回撤|15bp压力超额|交易次数|滚动12月超额>0|状态|",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["factors"]:
        development = item["periods"]["development"]
        pseudo = item["periods"]["pseudo_oos"]
        stress = item["pseudo_oos_stress_15bps"]
        rolling = item["periods"]["full_retrospective"]["rolling_12m_excess"]["positive_ratio"]
        rolling_text = "" if rolling is None else f"{rolling:.2%}"
        lines.append(
            f"|{item['factor_name_cn']}|{item['d20_evidence_rating']}（{item['d20_evidence_score']}/7）|"
            f"{development['excess_vs_realistic_buy_hold']:.2%}|{pseudo['excess_vs_realistic_buy_hold']:.2%}|"
            f"{pseudo['strategy']['max_drawdown']:.2%}|{stress['excess_vs_realistic_buy_hold']:.2%}|"
            f"{pseudo['strategy']['trade_count']}|{rolling_text}|{item['candidate_status']}|"
        )
    provisional = [item for item in report["factors"] if item["candidate_status"] == "ROUND2_PROVISIONAL_FOR_TRUE_FORWARD"]
    lines += ["", "## 结果", ""]
    if provisional:
        for item in provisional:
            lines.append(f"- `{item['factor_id']}`可进入真正前向候选讨论，但本轮不能冻结。")
    else:
        lines.append("- 没有第二轮因子同时通过证据、pseudo-OOS超额、风险改善、15bp压力和最低交易样本门槛。")
    lines += ["", "完整分期指标与交易台账保存在JSON和数据目录中。", ""]
    return "\n".join(lines)


def main() -> int:
    for path in (FEATURE_FILE, RESEARCH_FILE, ETF_FILE, DIVIDEND_FILE):
        if not path.exists():
            raise FileNotFoundError(f"第二轮回测输入缺失：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    research = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    d20_evidence = {
        item["factor_id"]: item["evidence"]
        for item in research["results"]
        if item["target"] == registry["targets"]["primary"]
    }
    features = pd.read_parquet(FEATURE_FILE)
    features["date"] = pd.to_datetime(features["date"])
    prices = pd.read_parquet(ETF_FILE)
    prices["date"] = pd.to_datetime(prices["date"])
    dividends = pd.read_csv(DIVIDEND_FILE, parse_dates=["record_date", "ex_date", "payment_date"])
    cost_settings = settings["backtest"]
    base_costs = BacktestCosts(
        commission_rate=float(cost_settings["commission_rate"]),
        minimum_commission_cny=float(cost_settings["minimum_commission_cny"]),
        stamp_duty_rate=float(cost_settings["stamp_duty_rate"]),
        slippage_bps=float(cost_settings["slippage_bps_base"]),
        lot_size=int(cost_settings["lot_size"]),
        cash_annual_rate=float(cost_settings["cash_annual_rate"]),
    )
    stress_costs = BacktestCosts(
        commission_rate=base_costs.commission_rate,
        minimum_commission_cny=base_costs.minimum_commission_cny,
        stamp_duty_rate=base_costs.stamp_duty_rate,
        slippage_bps=float(cost_settings["slippage_bps_stress"]),
        lot_size=base_costs.lot_size,
        cash_annual_rate=base_costs.cash_annual_rate,
    )
    initial_cash = float(cost_settings["initial_cash"])
    rule = registry["strategy_rule"]
    split = settings["research_split"]
    periods = {
        "development": (split["development_start"], split["development_end"]),
        "pseudo_oos": (split["validation_start"], split["validation_end"]),
        "contaminated_retrospective": (split["final_holdout_start"], split["final_holdout_end"]),
        "full_retrospective": (settings["project"]["start_date"], settings["project"]["end_date"]),
    }
    buy_hold_targets = features[["date"]].assign(target_position=1.0)
    factor_results: list[dict] = []
    LEDGER_ROOT.mkdir(parents=True, exist_ok=True)
    for definition in registry["factor_definitions"]:
        targets = build_hysteresis_target(
            features,
            factor_column=definition["column"],
            expected_sign=definition["expected_sign"],
            threshold_mode=definition["threshold_mode"],
            percentile_window=int(rule["percentile_window"]),
            minimum_history=int(rule["minimum_history"]),
            entry_percentile=float(rule["entry_percentile"]),
            exit_percentile=float(rule["exit_percentile"]),
            initial_position=float(rule["initial_position"]),
        )
        item_periods: dict[str, dict] = {}
        factor_directory = LEDGER_ROOT / definition["factor_id"]
        factor_directory.mkdir(parents=True, exist_ok=True)
        for name, (start, end) in periods.items():
            raw = _run_period(prices, dividends, targets, buy_hold_targets, base_costs, initial_cash, start, end)
            item_periods[name] = _without_frames(raw)
            raw["ledger"].to_parquet(factor_directory / f"{name}_ledger.parquet", index=False)
            raw["trades"].to_parquet(factor_directory / f"{name}_trades.parquet", index=False)
        stress_raw = _run_period(
            prices,
            dividends,
            targets,
            buy_hold_targets,
            stress_costs,
            initial_cash,
            split["validation_start"],
            split["validation_end"],
        )
        stress = _without_frames(stress_raw)
        evidence = d20_evidence[definition["factor_id"]]
        pseudo = item_periods["pseudo_oos"]
        sample_pass = pseudo["strategy"]["trade_count"] >= 6
        provisional = (
            evidence["rating"] in {"WEAK", "STRONG"}
            and pseudo["excess_vs_realistic_buy_hold"] > 0
            and pseudo["risk_adjusted_or_drawdown_improves"]
            and stress["excess_vs_realistic_buy_hold"] > 0
            and sample_pass
        )
        factor_results.append(
            {
                "factor_id": definition["factor_id"],
                "factor_name_cn": definition["name_cn"],
                "role": definition["role"],
                "expected_sign": definition["expected_sign"],
                "d20_evidence_rating": evidence["rating"],
                "d20_evidence_score": evidence["score"],
                "periods": item_periods,
                "pseudo_oos_stress_15bps": stress,
                "minimum_pseudo_oos_trade_count": 6,
                "sample_sufficiency_pass": sample_pass,
                "candidate_status": "ROUND2_PROVISIONAL_FOR_TRUE_FORWARD" if provisional else "NOT_ELIGIBLE",
            }
        )
    report = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "derivation_note": registry["derivation_note"],
        "rule": rule,
        "costs": base_costs.__dict__,
        "stress_slippage_bps": stress_costs.slippage_bps,
        "factor_count": len(factor_results),
        "factors": factor_results,
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text(_render_markdown(report), encoding="utf-8")
    print(f"第二轮回测报告：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
