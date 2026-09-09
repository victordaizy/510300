"""汇总第四轮流通市值近似加权研究与统一回测的正式决定。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round4.yaml"
DATA_STATUS_FILE = ROOT / "reports" / "research" / "000300_circulating_cap_weighted_breadth_dataset.json"
RESEARCH_FILE = ROOT / "reports" / "research" / "round4_cap_weighted_research.json"
BACKTEST_FILE = ROOT / "reports" / "backtest" / "round4_cap_weighted_backtests.json"
JSON_FILE = ROOT / "reports" / "research" / "round4_factor_selection_decision.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "round4_factor_selection_decision.md"


def _percentage(value: float) -> str:
    return f"{value:.2%}"


def main() -> int:
    for path in (SETTINGS_FILE, REGISTRY_FILE, DATA_STATUS_FILE, RESEARCH_FILE, BACKTEST_FILE):
        if not path.exists():
            raise FileNotFoundError(f"第四轮选择汇总缺少输入：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    data_status = json.loads(DATA_STATUS_FILE.read_text(encoding="utf-8"))
    research = json.loads(RESEARCH_FILE.read_text(encoding="utf-8"))
    backtest = json.loads(BACKTEST_FILE.read_text(encoding="utf-8"))
    eligible = [item["factor_id"] for item in backtest["factors"] if item["candidate_status"] != "NOT_ELIGIBLE"]
    comparison = data_status["current_official_snapshot_comparison"]
    result = {
        "status": "PASS",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "round": 4,
        "governance": registry["research_status"],
        "registration_timing": registry["registration_timing"],
        "strategy_selection_status": "NO_ROUND4_STRATEGY_CANDIDATE",
        "overall_strategy_status": "NO_FACTOR_FROZEN",
        "strategy_frozen": False,
        "eligible_factor_ids": eligible,
        "tested_factor_ids": [item["factor_id"] for item in registry["factor_definitions"]],
        "studywide_hypothesis_count": research["studywide_multiple_testing_audit"]["studywide_hypothesis_count"],
        "weighting_semantics": data_status["weighting_semantics"],
        "current_official_snapshot_comparison": comparison,
        "pv_lc_status": "USER_DEFERRED_NOT_ACTIVE",
        "next_research_track": "CSI300_OFFICIAL_SECTOR_INDEX_BREADTH_IF_FIVE_YEAR_PRICE_COVERAGE_PASSES",
        "reason": "第四轮D20六因子全部REJECTED，且所有统一规则pseudo-OOS和15bp压力超额均为负。",
        "blocked_tracks": registry["blocked_tracks"],
    }
    if eligible:
        raise RuntimeError(f"汇总逻辑预期无第四轮候选，但发现：{eligible}")
    lines = [
        "# 第四轮策略因子选择决定",
        "",
        "## 最终状态",
        "",
        "- 第四轮状态：`NO_ROUND4_STRATEGY_CANDIDATE`。",
        "- 项目总体状态：`NO_FACTOR_FROZEN`。",
        "- 六个流通市值近似加权因子、十二个D20/D5假设和六组统一回测已完成。",
        "- D20六个因子全部`REJECTED`；所有pseudo-OOS和15bp压力超额均为负。",
        "",
        "## 近似权重的误差边界",
        "",
        f"- 每日有效权重成员数{data_status['minimum_cap_weight_component_count']}至{data_status['maximum_cap_weight_component_count']}只，最低成员覆盖{data_status['minimum_member_market_cap_coverage']:.2%}。",
        f"- 在{comparison['snapshot_date']}与当前官方快照比较：Spearman={comparison['spearman_weight_correlation']:.3f}，Pearson={comparison['pearson_weight_correlation']:.3f}。",
        f"- 前十大只重合{comparison['top10_overlap_count']}只，平均绝对权重差{comparison['mean_absolute_weight_difference_pct_points']:.3f}个百分点。",
        "- 因此本轮只能回答流通市值近似参与和集中度，不能替代中证自由流通调整权重。",
        "",
        "## 因子决定",
        "",
        "|因子|D20评级|D5评级|pseudo-OOS超额|15bp压力超额|决定|",
        "|---|---|---|---:|---:|---|",
    ]
    for item in backtest["factors"]:
        pseudo = item["periods"]["pseudo_oos"]
        stress = item["pseudo_oos_stress_15bps"]
        lines.append(
            f"|{item['factor_name_cn']}|{item['d20_evidence_rating']}|{item['d5_evidence_rating']}|"
            f"{_percentage(pseudo['excess_vs_realistic_buy_hold'])}|"
            f"{_percentage(stress['excess_vs_realistic_buy_hold'])}|不进入策略|"
        )
    lines += [
        "",
        "## 下一步边界",
        "",
        "1. PV/LC按用户要求暂不开展，不再列为当前优先项。",
        "2. 官方历史权重仍受权限阻塞；流通市值近似分支到此停止，不新增变换。",
        "3. 验证11只中证沪深300一级行业子指数的五年行情覆盖；若完整，再事前登记板块宽度和轮动因子。",
        "4. 点时行业成分和行业权重仍不可得，行业子指数研究不得称为板块贡献或行业权重回测。",
        "",
    ]
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"第四轮选择决定：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
