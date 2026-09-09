"""交付抛物线转向的全部中文规则及已完成的历史账户。"""
from __future__ import annotations

import json
import shutil
import pandas as pd

from research.parabolic_reversal_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300抛物线转向_第74轮_20260908"
DOCUMENT = OUT / "抛物线转向_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_PARABOLIC_AFTERNOON_ENTRY_20260908.md"
CN = {"date": "交易日", "origin": "收盘决定日", "execution_date": "计划执行日", "origin_index": "历史原点序号",
    "simulation_only": "仅为研究模拟", "requested_quantity": "请求份额", "reference_weight": "策略股票目标",
    "action": "中文动作", "signal_state": "信号状态", "wealth_high": "含分红财富最高价", "wealth_low": "含分红财富最低价",
    "wealth_close": "含分红财富收盘", "input_valid": "输入完整有效", "source_state": "指标状态", "sar_input": "本日检查保护价",
    "sar_level": "本日展示保护价", "next_sar": "下一日保护价", "extreme_price": "趋势极值", "acceleration": "加速系数",
    "trend_state": "趋势方向", "target": "股票目标比例", "reversed": "本日是否转向", "initialized": "本日是否完成初始化"}


def chinese_states(frame):
    out = frame.copy()
    if "source_state" in out:
        out["source_state"] = out.source_state.replace({"VIEW_UP_STATE": "有效上行", "VIEW_DOWN_STATE": "有效下行",
            "NO_VIEW_FIRST_VALID_BAR": "无观点：只有第一根有效价格", "NO_VIEW_INPUT_RESET": "无观点：输入缺失重置"})
    if "signal_state" in out:
        out["signal_state"] = out.signal_state.replace({"MODEL_VIEW_AVAILABLE": "有效策略目标", "NO_VIEW": "无观点"})
    if "trend_state" in out:
        out["trend_state"] = out.trend_state.map({1: "上行", 0: "下行"}).fillna("无观点")
    return out.rename(columns=CN)


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require([r["round"] for r in index["completed_rounds"]] == list(range(1, 74)), "索引不是截至73轮，不能重复收尾")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "转向规则已固定内容改变")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((RESEARCH / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(receipt["reviewer_source_sha256"] == digest(ROOT / "scripts/review_round74_saved.py"), "保存核对入口改变")
    require(not result["historical_point_target_met"], "不能按失败关闭已出现1.2的候选")
    require(NEXT.exists() and not OUT.exists(), "后续说明缺失或交付目录已存在")
    status = "COMPLETED_REJECTED_PARABOLIC_REVERSAL_MAIN_LOSS"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False,
        "decision": "主两档费用夏普为负，较早也低于1.2；主基础价格加分红在显式费用前已亏损。结束本常规转向设置，不调加速、上限、确认或反向规则救回。",
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 510300抛物线转向：第74轮", "", "## 结果与决定", "",
        "**本轮没有达到净夏普1.2，结束这项方案。** 主评价基础费用净夏普为负0.099，压力费用为负0.217；较早历史分别为0.306和0.211。主账户亏损，较早虽赚钱，也没有形成两段历史共同改善。", "",
        "本轮只使用现有每日最高价、最低价、收盘价和已知现金分红，独立定义股票与现金切换。没有拟合模型，没有补EPS、公募或其他慢来源。一个常规设置，完成四个新账户，16个旧对照直接复用。", "",
        "## 策略怎样进入和退出", "",
        "价格上行阶段持续出现新高时，保护价逐渐加速靠近；最低价触及保护价后转为下行。下行阶段按最低价对称计算，最高价触及保护价后转为上行。每个收盘确认方向，上行对应股票目标100%，下行对应现金目标；下一开盘按真实现金、份额和费用成交。", "",
        "加速系数从0.02开始，每出现严格的新极值增加0.02，最高0.2；转向后重置。持仓与目标相差不到10个百分点时维持份额，达到或超过带宽才按整手调整；本次实际没有额外补仓。再次进入依照最新有效方向，不额外设置冷却。原学习模型的6%亏损、8%回撤及60日上限不加入本轮。", "",
        "保护价只用于收盘后的方向判断，实际成交不能假定在保护价发生。触线相等也算转向，每天最多检查一次，不猜测当日高低价出现的先后。数据缺失时保留无观点和实际持仓。", "",
        "## 两段完整历史结果", "", "### 主评价：2020年1月2日至2026年8月14日开盘", ""]
    lines += table(result["all_metrics"])
    lines += ["### 较早诊断：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["两段各从20万元开始，分别保留1604日和1219日完整账户及全部空仓日。主基础累计亏13.52%，压力亏22.06%；较早基础累计赚20.07%，压力赚11.10%。四个账户的复合年化均低于对应买入持有；较早基础夏普0.306略高于买入持有0.279，不能因此写成四项指标都优于或都劣于买入持有。", "",
        "## 失败主要发生在哪里", "",
        "主基础价格损益负14,619.70元、分红9,265.60元，相加已亏5,354.10元；再扣佣金5,330.72元和滑点16,352.00元，最终亏27,036.82元。因此问题包括进入、退出和持有路径，不能仅归因于费用。这里的费用前分解沿用已发生的实际份额，不是另跑一个免手续费策略。", "",
        "主基础75个完整周期中23个盈利、52个亏损，盈利周期合计赚181,727.09元，亏损周期合计亏208,763.91元。较早基础52周期中22盈30亏，总共赚40,132.51元，但最大的两段盈利合计91,733.88元，其他周期合计为负，收益存在集中性。", "",
        "主基础比原学习退出账户少110,170.40元，其中价格收益少98,280.10元、分红多228.00元、显式费用多12,118.30元。较早基础比原少63,430.38元。新方案增加了持仓覆盖和交易频率，但没有换来更好的完整账户结果。", "",
        "## 完整周期与账户经济分解", "", "|历史|费用|周期数|盈利／亏损周期|价格损益（元）|分红（元）|佣金（元）|滑点（元）|净损益（元）|持仓收盘数|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pd.read_csv(RESEARCH / "saved_cycle_profit_groups.csv").itertuples():
        ledger = pd.read_parquet(RESEARCH / row.period / row.cost / f"{PRIMARY}_ledger.parquet")
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早历史'}|{'基础' if row.cost=='BASE' else '压力'}|{row.cycles}|{row.positive_cycles}／{row.negative_cycles}|{ledger.price_pnl.sum():,.2f}|{ledger.dividend_recognized.sum():,.2f}|{ledger.commission.sum():,.2f}|{ledger.slippage_cost.sum():,.2f}|{row.net_profit:,.2f}|{int(ledger.shares.gt(0).sum())}|")
    lines += ["", "主账户每档150笔成交、75个周期；较早每档104笔、52个周期。四个账户均无整笔受阻未成交请求，评价原点指标均有效。两段开始时历史递推已经处于上行，因此申请进入，不代表评价首日才出现突破。终点开盘强制清仓，使上行目标原点数分别821和676，比持仓收盘数820和675各多1。", "",
        "## 基础费用逐年结果", "", "|历史|年份|该年净收益|净夏普|年内最大回撤|成交笔数|", "|---|---|---:|---:|---:|---:|"]
    years = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in years[years.model.eq(PRIMARY) & years.cost.eq("BASE")].itertuples():
        sharpe = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "未定义"
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早历史'}|{row.year}|{row.cumulative_return:.2%}|{sharpe}|{-row.max_drawdown:.2%}|{row.trade_count}|")
    lines += ["", "2026年只到固定终点。单年超过1.2不是整个账户达标；年内回撤与全期回撤使用的起算高点不同。本轮2023年存在交易，不能沿用其他策略全年空仓的描述。", "",
        "## 全部因子与完整中文规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()
    lines += [line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in protocol[1:]]
    lines += ["", "## 保存内容和核对", "",
        "同目录包含3456日全部指标中文表、四份逐日完整因子与进出场请求、四份完整账户、全部成交、周期分解、年度和分段指标、20项含旧对照的完整结果、冻结设置与参考算法许可。规则已经用中文逐项写明，原始记录保留便于追溯。", "",
        "11项必要测试一次通过。完成3456行因子、5646个收盘份额请求、254个含登记日分红权益完整周期、20项账户指标和12组经济差额的保存复算。递推状态最大复算误差为零，未另跑诊断账户或拟合模型。这是计算及经济记录核对，不是证明策略有效，也不是与外部编译库逐位比对。", "",
        "## 下一项：原策略下午提前进入", "",
        "下一项保留原学习退出和风险退出，只检验已满足条件的入场能否在下午提前执行。现有分钟数据有1211日，较早历史完全没有分钟覆盖；没有覆盖的日子继续原日线机制，因此只能验证局部改动，不能把较早回退结果当新时点的有效性证明。", "",
        "目前已核对来源身份及成交量单位，第75轮尚未登记或回测。继续暂停EPS和其他慢来源，不制作GPT数值包。目标仍未完成，研究继续。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "account_coverage.csv", "result.json",
        "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv", "saved_parabolic_state_replay.csv",
        "saved_state_and_request_replay.csv", "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json", "wealth_scale_receipt.json"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    shutil.copy2(NEXT, OUT / "后续下午进入方向及来源边界.md")
    shutil.copy2(ROOT / "docs/reference/ta_lib_sar_20260908/LICENSE.txt", OUT / "参考算法许可.txt")
    chinese_states(pd.read_parquet(RESEARCH / "factors.parquet")).to_csv(OUT / "全部历史指标中文表.csv", index=False, encoding="utf-8-sig")
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost, cost_label in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet").to_csv(OUT / f"{label}_{cost_label}_完整账户.csv", index=False, encoding="utf-8-sig")
            chinese_states(pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")).to_csv(OUT / f"{label}_{cost_label}_逐日全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(folder / f"{PRIMARY}_trades.csv", OUT / f"{label}_{cost_label}_全部成交.csv")
    record = {"round": 74, "study": result["study_id"], "title": "价格极值与加速保护价的独立进出场", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), status="ROUND74_COMPLETE_PARABOLIC_REVERSAL_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计74轮，348不同设置，362已评价来源版本，367登记含5旧未运行，1162主评价记录。第74轮一个转向设置，无新模型或参考账户。",
        checks="第74轮11测试，3456指标20账户指标12差额5646请求254含分红周期复算完成。",
        process_state_note="第74轮完整账户、失败归因和中文交付完成；第75轮仅下午进入方向及现有分钟源可行性完成，尚未登记或实现。",
        next_work={"status": "AFTERNOON_ENTRY_DIRECTION_NOT_REGISTERED", "focus": "保留原退出，只用现有分钟局部检验提前进入；较早无分钟不算新时点验证", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_parabolic_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "PARABOLIC_REVERSAL_ROUND74_CHINESE_RESULTS", "rounds": [74], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "PARABOLIC_RULES_FACTORS_AND_FULL_ACCOUNTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery, exclusive=True)
    print(json.dumps({"交付": delivery, "最新状态": index["status"], "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
