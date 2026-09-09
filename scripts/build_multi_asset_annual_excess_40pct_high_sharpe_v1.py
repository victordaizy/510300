"""生成多资产40%净超额与高夏普研究状态。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from research.multi_asset_annual_excess_40pct_high_sharpe_v1 import load_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v1.yaml"
OLD_REPORTS = {
    "ALL_ETF_MOMENTUM_V1R": ROOT / "reports" / "backtest" / "csi300_all_etf_momentum_alpha_v1r.json",
    "ETF_ROTATION_V1": ROOT / "reports" / "backtest" / "csi300_etf_rotation_alpha_v1.json",
    "DUAL_SLEEVE_V1": ROOT / "reports" / "backtest" / "csi300_dual_sleeve_projection_v1.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _old_result_summary(path: Path, *, dual_sleeve: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if dual_sleeve:
        base = value["recent"]["base_cost"]
        stress = value["recent"]["stress_cost"]
    else:
        base = value["base_cost"]
        stress = value["stress_cost"]
    return {
        "status": value["status"],
        "base_annualized_excess": float(base["annualized_excess"]),
        "stress_annualized_excess": float(stress["annualized_excess"]),
        "base_strategy_sharpe": float(base["strategy"]["sharpe_zero_cash_rate"]),
        "stress_strategy_sharpe": float(stress["strategy"]["sharpe_zero_cash_rate"]),
        "eligible_for_reuse": False,
    }


def build_status(contract: dict[str, Any], config_path: Path) -> dict[str, Any]:
    data_paths = {key: ROOT / value for key, value in contract["available_local_data"].items()}
    fund_master = pd.read_parquet(data_paths["broad_etf_master"])
    etf_panel = pd.read_parquet(
        data_paths["broad_etf_total_return_panel"], columns=["date", "con_code"]
    )
    a_share_panel = pd.read_parquet(
        data_paths["a_share_point_in_time_panel"], columns=["date", "con_code"]
    )
    future = pd.read_parquet(data_paths["index_future_daily"], columns=["date", "symbol"])
    old = {
        "ALL_ETF_MOMENTUM_V1R": _old_result_summary(OLD_REPORTS["ALL_ETF_MOMENTUM_V1R"]),
        "ETF_ROTATION_V1": _old_result_summary(OLD_REPORTS["ETF_ROTATION_V1"]),
        "DUAL_SLEEVE_V1": _old_result_summary(OLD_REPORTS["DUAL_SLEEVE_V1"], dual_sleeve=True),
    }
    return {
        "schema_version": "1.0.0",
        "report_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "TARGET_NOT_YET_MET_NO_FROZEN_CANDIDATE",
        "goal_achieved": False,
        "objective": {
            "initial_capital_cny": float(contract["scope"]["initial_capital_cny"]),
            "user_transaction_fee_rate_per_leg": float(
                contract["costs"]["user_transaction_fee_rate_per_leg"]
            ),
            "minimum_annualized_excess": float(contract["objective"]["minimum_annualized_excess"]),
            "minimum_strategy_net_sharpe": float(
                contract["objective"]["minimum_strategy_net_sharpe"]
            ),
            "benchmark": contract["benchmark"]["primary_id"],
        },
        "local_data_readiness": {
            "broad_etf": {
                "master_asset_count": int(fund_master["ts_code"].nunique()),
                "panel_asset_count": int(etf_panel["con_code"].nunique()),
                "date_min": pd.to_datetime(etf_panel["date"]).min().date().isoformat(),
                "date_max": pd.to_datetime(etf_panel["date"]).max().date().isoformat(),
                "status": "DATA_AVAILABLE_MECHANISM_NOT_FROZEN",
            },
            "a_share_point_in_time": {
                "security_count": int(a_share_panel["con_code"].nunique()),
                "row_count": int(len(a_share_panel)),
                "date_min": pd.to_datetime(a_share_panel["date"]).min().date().isoformat(),
                "date_max": pd.to_datetime(a_share_panel["date"]).max().date().isoformat(),
                "status": "DATA_AVAILABLE_MECHANISM_NOT_FROZEN",
            },
            "index_future": {
                "symbol_count": int(future["symbol"].nunique()),
                "date_min": pd.to_datetime(future["date"]).min().date().isoformat(),
                "date_max": pd.to_datetime(future["date"]).max().date().isoformat(),
                "status": "DATA_AVAILABLE_HEDGE_GRANULARITY_NOT_YET_QUALIFIED",
            },
        },
        "active_research_lanes": {
            "BROAD_LIQUID_ETF_CROSS_ASSET": {
                "status": "MECHANISM_SELECTION_PENDING",
                "prior_nine_factor_momentum_reuse_allowed": False,
            },
            "A_SHARE_CROSS_SECTIONAL_HEDGED": {
                "status": "MECHANISM_SELECTION_PENDING",
                "prior_failed_residual_trend_reuse_allowed": False,
            },
        },
        "prior_results": old,
        "decision": {
            "existing_candidate_meets_40pct_and_sharpe_1_5": False,
            "old_fee_reduction_can_rescue_low_sharpe_models": False,
            "next_step": "每条研究线只冻结一个与旧失败公式不同的经济机制，再做成本后可见期筛选。",
        },
        "inputs": {
            "contract": {
                "path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(config_path),
            },
            **{
                key: {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256(path),
                }
                for key, path in data_paths.items()
            },
        },
        "safety": contract["safety"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    readiness = report["local_data_readiness"]
    old = report["prior_results"]
    lines = [
        "# 多资产年化净超额40个百分点、高夏普：当前状态",
        "",
        f"- 状态：`{report['status']}`",
        "- 账户：500,000元；每条买入或卖出腿的用户交易费率为0.0001。",
        "- 硬门槛：基础与压力成本后的年化净超额均不低于40个百分点，净夏普均不低于1.50。",
        "- 当前结论：尚无冻结候选达到目标。",
        "",
        "## 可用研究数据",
        "",
        f"- 广泛ETF：主表 {readiness['broad_etf']['master_asset_count']} 只，收益面板 {readiness['broad_etf']['panel_asset_count']} 只，区间 {readiness['broad_etf']['date_min']} 至 {readiness['broad_etf']['date_max']}。",
        f"- 点时A股：{readiness['a_share_point_in_time']['security_count']}只、{readiness['a_share_point_in_time']['row_count']}行，区间 {readiness['a_share_point_in_time']['date_min']} 至 {readiness['a_share_point_in_time']['date_max']}。",
        f"- IF连续日线：区间 {readiness['index_future']['date_min']} 至 {readiness['index_future']['date_max']}；对冲手数与50万元账户容量尚待候选级检验。",
        "",
        "## 旧公式不能救回",
        "",
        f"- 全ETF九因子动量V1R：压力超额 {old['ALL_ETF_MOMENTUM_V1R']['stress_annualized_excess']:.2%}，压力夏普 {old['ALL_ETF_MOMENTUM_V1R']['stress_strategy_sharpe']:.2f}，已冻结拒绝。",
        f"- 八因子ETF轮动V1：压力超额 {old['ETF_ROTATION_V1']['stress_annualized_excess']:.2%}，压力夏普 {old['ETF_ROTATION_V1']['stress_strategy_sharpe']:.2f}，已冻结拒绝。",
        f"- 双袖投影V1近期段：压力超额 {old['DUAL_SLEEVE_V1']['stress_annualized_excess']:.2%}，压力夏普 {old['DUAL_SLEEVE_V1']['stress_strategy_sharpe']:.2f}，且外部段失败，已冻结拒绝。",
        "- 降低费用无法把0.3至0.6附近的夏普提升到1.50，也无法把个位数或十余个百分点超额提升到40个百分点。",
        "",
        "## 下一步",
        "",
        "1. 广泛ETF线：只选择一个不同于旧动量打分的经济机制。",
        "2. A股对冲线：只选择一个不同于旧残差趋势的横截面机制，并先验证50万元对冲粒度。",
        "3. 两条线在可见期任何一项低于40%超额或1.50夏普即拒绝，不调参救回。",
        "4. 历史通过仍不是完成；至少726个前瞻日与全部门槛通过后才能宣布达标。",
        "",
        "本文件不是收益承诺、仓位建议、订单或实盘授权。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成多资产40%净超额与高夏普研究状态")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    contract = load_contract(config_path)
    report = build_status(contract, config_path)
    if not args.no_write:
        json_path = ROOT / contract["outputs"]["current_status_json"]
        markdown_path = ROOT / contract["outputs"]["current_status_markdown"]
        atomic_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        atomic_text(markdown_path, render_markdown(report))
    print(
        json.dumps(
            {
                "状态": report["status"],
                "目标已达成": report["goal_achieved"],
                "目标年化净超额": report["objective"]["minimum_annualized_excess"],
                "最低净夏普": report["objective"]["minimum_strategy_net_sharpe"],
                "ETF资产数": report["local_data_readiness"]["broad_etf"]["panel_asset_count"],
                "A股证券数": report["local_data_readiness"]["a_share_point_in_time"]["security_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
