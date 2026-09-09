"""汇总已完成的持续研究，并保存供后续任务承接的导航。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("510300_adaptive_allocation_v1", "条件切换与滚动回归", 28),
    ("510300_return_classification_v1", "涨跌分类与幅度加权", 26),
    ("510300_overnight_global_information_v1", "免费全球盘后信息", 16),
    ("510300_policy_liquidity_quantity_v1", "央行操作量与资金压力", 16),
    ("510300_direct_policy_utility_v1", "直接学习风险收益与换仓成本", 12),
    ("510300_calendar_liquidity_timing_v1", "事先可知的日历条件", 14),
]


def update() -> None:
    completed, all_rows = [], []
    for stage, title, count in STAGES:
        path = ROOT / "reports/research" / stage
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        metrics = pd.read_csv(path / "metrics.csv")
        assert len(metrics) == (count + 1) * 2
        base = next(row for row in result["primary"] if row["cost"] == "BASE")
        stress = next(row for row in result["primary"] if row["cost"] == "STRESS")
        completed.append({"round": len(completed) + 1, "study": result["study_id"], "title": title,
                          "status": result["status"], "result": (path / "result.json").relative_to(ROOT).as_posix(),
                          "candidate_configurations": count, "evaluation_accounts": len(metrics),
                          "primary_base": base, "primary_stress": stress,
                          "post_selected_best_base": result["post_selected_best_base"]})
        all_rows.extend({"round": title, **row} for row in metrics.to_dict("records"))
    frame = pd.DataFrame(all_rows)
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    deliveries = [
        {"rounds": [1, 2, 3], "zip": "deliverables/510300夏普1.2持续研究_前三轮_GPT审阅_20260906.zip",
         "md": "deliverables/510300夏普1.2持续研究_前三轮_20260906/510300夏普1.2研究_三轮结果与完整中文规则.md"},
        {"rounds": [4], "zip": "deliverables/510300夏普1.2持续研究_第四轮央行操作量_GPT审阅_20260906.zip",
         "md": "deliverables/510300夏普1.2持续研究_第四轮央行操作量_20260906/第四轮结果与全部中文因子规则.md"},
        {"rounds": [5], "zip": "deliverables/510300夏普1.2持续研究_第五轮直接仓位学习_GPT审阅_20260906.zip",
         "md": "deliverables/510300夏普1.2持续研究_第五轮直接仓位学习_20260906/第五轮结果与全部中文因子规则.md"},
        {"rounds": [6], "zip": "deliverables/510300夏普1.2持续研究_第六轮日历条件_GPT审阅_20260906.zip",
         "md": "deliverables/510300夏普1.2持续研究_第六轮日历条件_20260906/第六轮结果与全部中文因子规则.md"},
    ]
    for item in deliveries:
        assert (ROOT / item["zip"]).is_file() and (ROOT / item["md"]).is_file()
        receipt = ROOT / item["zip"]
        if receipt.with_suffix(".delivery.json").exists():
            saved = json.loads(receipt.with_suffix(".delivery.json").read_text(encoding="utf-8"))
            item.update({"bytes": saved["bytes"], "sha256": saved["sha256"], "members": saved["members"]})
    latest = {
        "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "goal": "完整账户成本后夏普率至少 1.2", "goal_achieved": False,
        "status": "SIX_ROUNDS_COMPLETED_TARGET_NOT_MET_CONTINUATION_ACTIVE",
        "completed_rounds": completed, "running_studies": [],
        "registered_configurations_in_this_resumption": sum(row["candidate_configurations"] for row in completed),
        "evaluation_accounts_in_this_resumption": len(frame),
        "count_warning": "含重复对照，登记配置数量不是独立研究机制数量；还另存首轮50条影子账户、第五轮1条和第六轮2条零费用诊断账户。",
        "post_selected_best_base": best.to_dict(),
        "independent_high_sharpe_evidence": "NOT_ESTABLISHED",
        "assets": ["510300.SH", "CASH_CNY"], "other_etf_scope": "USER_QUESTION_PENDING",
        "source_budget": 0, "live_trading_authorized": False, "position_impact": 0,
        "automation_id": "510300-1-2", "automation_status": "ACTIVE",
        "continuation_thread": "01a07230-2e44-7753-bb97-2dc6fca3df71",
        "deliveries": deliveries,
        "new_evidence": [
            "第六轮14个候选均未达1.2；主方案月初三日基础夏普0.238860，压力夏普0.005231，零费用诊断0.509782。",
            "本次恢复研究的最高历史读数升至0.519648，来自第六轮事后最好K2_MONTH_EDGE；压力夏普0.374131，基础年化5.491051%，回撤21.240989%。",
            "K2_MONTH_EDGE的基础分期夏普依次为1.549233、0.093996、0.124593，收益优势集中在2020—2021；不能拿最好两年替换完整区间。",
            "日历加价格共识基础夏普-0.091824，相对价格共识的增量区间跨零，没有证明日历信息可稳定改善已训练模型。",
        ],
        "next_work": "保留第六轮日期窗口和失败结论，不因为2020—2021分期夏普超过1.2就删去后续时期，也不添加日历阈值扫描。下一轮优先排查ETF净值折溢价、净申赎或跨市场基差等有不同经济含义的信息，先查既有终结记录和历史可用时钟。只有新机制且资料足够时预登记有限候选并运行完整账户；若相关方向已终结或依赖未获得的资料，继续不依赖它的其他新机制。现有免费来源和510300与现金范围保持，其他ETF问题仍待用户回复。",
        "checks": "各包已经完成结构和保存账户数值核对；未做安全审计，未上传，未声称外部评审已完成。",
    }
    status_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    status_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 510300 夏普率 1.2 持续研究：截至第六轮", "",
             "**目前尚未实现目标。** 本次恢复研究后已完成六轮、一百一十二个登记配置和二百三十六个完整评价账户，含重复对照。另有第一轮五十条影子账户、第五轮一条和第六轮两条零费用诊断账户。", "",
             "所有普通评价账户统一采用二十万元起始资金，交易510300和人民币现金，现金利率为零，包含分红、整手、T+1、等待日、佣金、滑点和期末退出。区间为二〇二〇年一月二日至二〇二六年八月十四日开盘，共一千六百零四个交易日。", "",
             "|轮次与机制|主方案基础夏普|压力夏普|基础年化收益|基础最大回撤|本轮事后最好夏普|",
             "|---|---:|---:|---:|---:|---:|"]
    for row in completed:
        base, stress, selected = row["primary_base"], row["primary_stress"], row["post_selected_best_base"]
        lines.append(f"|第{row['round']}轮：{row['title']}|{base['net_sharpe']:.4f}|{stress['net_sharpe']:.4f}|{base['annualized_return']:.2%}|{-base['max_drawdown']:.2%}|{selected['net_sharpe']:.4f}|")
    lines += ["", f"六轮全部登记候选中的事后最高基础夏普升至 **{best.net_sharpe:.4f}**，距离一点二尚有明显差距。这个数值是事后挑选的历史结果，不能视为已经证明稳定有效。", "",
              "第四轮加入央行七天逆回购已披露操作量，相对已有资金利率信息没有显示稳定收益改善；该增量区间跨零。第五轮直接优化风险收益取舍，主方案夏普仍只有零点一七六，零费用诊断也只有零点二二一，说明只降低费用或改变训练目标不足以解决问题。", "",
              "第六轮的新最高值来自月末五个自然日与月初三个交易日组合：基础夏普零点五二〇、压力夏普零点三七四，基础年化收益百分之五点四九、最大回撤百分之二十一点二四。它在二〇二〇至二〇二一年的夏普为一点五四九，但后两个分期分别只有零点〇九四和零点一二五；完整区间仍未达标，不把最好两年当成全期结果。", "",
              "后续继续研究新的有效条件信息。每小时自动续行安排保持启用，实际运行依赖本机与应用保持运行、工作区可访问。目标未完成，不将自动任务启用、模型收敛或数据核对完成写成策略达标。", "",
              "## 全部中文规则与研究包", ""]
    for item in deliveries:
        name = "前三轮" if len(item["rounds"]) > 1 else f"第{item['rounds'][0]}轮"
        lines.append(f"- {name}：[中文因子、规则及完整结果]({Path(item['md']).relative_to('deliverables').as_posix()})；[GPT审阅包]({Path(item['zip']).relative_to('deliverables').as_posix()})。")
    lines += ["", "各审阅包均完整保留该轮原始资料、冻结协议、代码、训练记录和全部逐日账户。只做必要的结构与数值检查，没有安全审计、上传或外部审阅声明。"]
    overview = ROOT / "deliverables/510300夏普1.2持续研究_截至第六轮_20260906.md"
    overview.write_text("\n".join(lines) + "\n", encoding="utf-8")
    frame.to_csv(ROOT / "deliverables/510300夏普1.2持续研究_六轮完整指标_20260906.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"状态": latest["status"], "总览": str(overview), "最新研究索引": str(status_path),
                      "登记配置": latest["registered_configurations_in_this_resumption"],
                      "评价账户": latest["evaluation_accounts_in_this_resumption"],
                      "最高基础夏普": float(best.net_sharpe)}, ensure_ascii=False))


if __name__ == "__main__":
    update()
