"""使用统一30/70滞回规则回测第四轮流通市值近似加权因子。"""

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
REGISTRY_FILE = ROOT / "config" / "factor_registry_round4.yaml"
FEATURE_FILE = ROOT / "data" / "features" / "510300_round4_cap_weighted_dataset.parquet"
RESEARCH_FILE = ROOT / "reports" / "research" / "round4_cap_weighted_research.json"
ETF_FILE = ROOT / "data" / "raw" / "market" / "510300_daily_raw.parquet"
DIVIDEND_FILE = ROOT / "data" / "reference" / "510300_dividends.csv"
JSON_FILE = ROOT / "reports" / "backtest" / "round4_cap_weighted_backtests.json"
MARKDOWN_FILE = ROOT / "reports" / "backtest" / "round4_cap_weighted_backtests.md"
LEDGER_ROOT = ROOT / "data" / "processed" / "round4_cap_weighted_backtests"


def _render_markdown(report: dict) -> str:
    lines = [
        "# 第四轮流通市值近似加权统一规则回测",
        "",
        "## 治理口径",
        "",
        "- 六个因子全部沿用30%/70%点时滚动分位数滞回规则，没有逐因子调参或翻转方向。",
        "- 信号在收盘后形成，下一交易日开盘执行；含T+1、100份整手、分红、佣金、最低佣金和滑点。",
        "- 候选必须同时满足D20原始证据、D5审计证据、pseudo-OOS超额、风险改善、15bp压力和最低交易次数。",
        "- 流通市值权重是近似值；即使通过也只能进入真正前向观察，不能冻结。",
        "",
        "|因子|D20审计评级|D5评级|开发超额|pseudo-OOS超额|pseudo-OOS回撤|15bp压力超额|交易次数|状态|",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["factors"]:
        development = item["periods"]["development"]
        pseudo = item["periods"]["pseudo_oos"]
        stress = item["pseudo_oos_stress_15bps"]
        lines.append(
            f"|{item['factor_name_cn']}|{item['d20_evidence_rating']}（原始{item['d20_uncapped_evidence_rating']}）|"
            f"{item['d5_evidence_rating']}|{development['excess_vs_realistic_buy_hold']:.2%}|"
            f"{pseudo['excess_vs_realistic_buy_hold']:.2%}|{pseudo['strategy']['max_drawdown']:.2%}|"
            f"{stress['excess_vs_realistic_buy_hold']:.2%}|{pseudo['strategy']['trade_count']}|"
            f"{item['candidate_status']}|"
        )
    eligible = [item for item in report["factors"] if item["candidate_status"] != "NOT_ELIGIBLE"]
    lines += ["", "## 回测门槛结果", ""]
    if eligible:
        for item in eligible:
            lines.append(f"- `{item['factor_id']}`进入真正前向观察，但不得冻结或声称已验证。")
    else:
        lines.append("- 没有因子通过全部预登记门槛；不得从失败项中事后挑选回测赢家。")
    lines += ["", "完整分期指标、压力测试和逐日交易台账保存在JSON与数据目录。", ""]
    return "\n".join(lines)


def main() -> int:
    for path in (SETTINGS_FILE, REGISTRY_FILE, FEATURE_FILE, RESEARCH_FILE, ETF_FILE, DIVIDEND_FILE):
        if not path.exists():
            raise FileNotFoundError(f"第四轮回测输入缺失：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    research = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    evidence = {(item["factor_id"], item["target"]): item["evidence"] for item in research["results"]}
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
    gate = registry["candidate_gate"]
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
        factor_directory = LEDGER_ROOT / definition["factor_id"]
        factor_directory.mkdir(parents=True, exist_ok=True)
        item_periods: dict[str, dict] = {}
        for period_name, (start, end) in periods.items():
            raw = _run_period(prices, dividends, targets, buy_hold_targets, base_costs, initial_cash, start, end)
            item_periods[period_name] = _without_frames(raw)
            raw["ledger"].to_parquet(factor_directory / f"{period_name}_ledger.parquet", index=False)
            raw["trades"].to_parquet(factor_directory / f"{period_name}_trades.parquet", index=False)
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
        d20 = evidence[(definition["factor_id"], registry["targets"]["primary"])]
        d5 = evidence[(definition["factor_id"], registry["targets"]["sensitivity"])]
        pseudo = item_periods["pseudo_oos"]
        checks = {
            "d20_uncapped_evidence": d20["uncapped_rating"] in gate["primary_d20_uncapped_evidence_ratings"],
            "d5_audited_evidence": d5["rating"] in gate["sensitivity_d5_audited_evidence_ratings"],
            "minimum_trade_count": pseudo["strategy"]["trade_count"] >= int(gate["minimum_pseudo_oos_trade_count"]),
            "positive_pseudo_oos_excess": pseudo["excess_vs_realistic_buy_hold"] > 0,
            "risk_adjusted_or_drawdown_improves": bool(pseudo["risk_adjusted_or_drawdown_improves"]),
            "positive_15bps_stress_excess": stress["excess_vs_realistic_buy_hold"] > 0,
        }
        eligible = all(checks.values())
        factor_results.append(
            {
                "factor_id": definition["factor_id"],
                "factor_name_cn": definition["name_cn"],
                "role": definition["role"],
                "expected_sign": definition["expected_sign"],
                "d20_evidence_rating": d20["rating"],
                "d20_uncapped_evidence_rating": d20["uncapped_rating"],
                "d20_evidence_score": d20["score"],
                "d5_evidence_rating": d5["rating"],
                "d5_evidence_score": d5["score"],
                "periods": item_periods,
                "pseudo_oos_stress_15bps": stress,
                "candidate_gate_checks": checks,
                "candidate_status": "ROUND4_PROVISIONAL_FOR_TRUE_FORWARD" if eligible else "NOT_ELIGIBLE",
            }
        )
    report = {
        "status": "PASS_APPROXIMATION_NOT_OFFICIAL_WEIGHT",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "registration_timing": registry["registration_timing"],
        "weighting_semantics": "DAILY_CIRCULATING_MARKET_CAP_APPROXIMATION_NOT_OFFICIAL_CSI_WEIGHT",
        "rule": rule,
        "candidate_gate": gate,
        "costs": base_costs.__dict__,
        "stress_slippage_bps": stress_costs.slippage_bps,
        "factor_count": len(factor_results),
        "factors": factor_results,
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text(_render_markdown(report), encoding="utf-8")
    print(f"第四轮流通市值近似加权回测报告：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
