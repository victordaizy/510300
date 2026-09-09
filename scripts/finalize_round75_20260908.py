"""交付下午进入的实际结果、全部中文因子及原模型系数，不重跑研究。"""
from __future__ import annotations

import json
import shutil
import pandas as pd

from research.afternoon_entry_v1_output_fix import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from research.learned_cycle_exit_v1 import FEATURES, CN
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300下午提前进入_第75轮_20260908"
DOCUMENT = OUT / "下午提前进入_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_AFTERNOON_ENTRY_RISK_BUDGET_20260908.md"
FIELDS = {"date": "交易日", "origin": "收盘决定日", "execution_date": "计划执行日", "action": "中文动作", "requested_quantity": "请求份额",
    "filled_quantity": "实际成交份额", "entry_rearmed": "重新进入资格", "days_since_exit": "距实际退出交易日数", "cash_at_signal": "信号时已知现金",
    "entry_signal": "下午临时进入条件", "signal_time": "下午决定时刻", "observation_label": "观察分钟标签", "execution_label": "成交代理分钟标签",
    "factor_status": "因子状态", "execution_status": "成交状态", "signal_price": "信号价格", "execution_volume": "后续成交分钟总份额",
    "previous_d60": "昨日完整六十日强弱", "partial_d60": "下午临时六十日强弱", "partial_overnight_log": "今日已知隔夜对数收益",
    "partial_intraday_log": "今日下午临时日内对数收益", "partial_difference": "今日下午临时日内减隔夜差值", "exit_reasons": "退出原因",
    "continuation_prediction": "原八项继续持有预测", "learning_fit_origin": "模型成熟日", "negative_confirmation_count": "连续负值次数",
    "learned_exit_requested": "学习退出请求", "learning_status": "学习状态", "d60_factor": "完整六十日强弱", "entry_condition": "原完整进入条件",
    "original_price_exit": "原价格退出条件", **dict(zip(FEATURES, CN))}


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require([r["round"] for r in index["completed_rounds"]] == list(range(1, 75)), "索引不是截至74轮，不能重复收尾")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "下午进入固定内容改变")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((RESEARCH / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(digest(ROOT / "scripts/review_round75_saved.py") == receipt["reviewer_sha256"], "保存核对入口改变")
    require(not result["historical_point_target_met"] and NEXT.exists() and not OUT.exists(), "本轮状态或交付前提不符")
    status = "COMPLETED_REJECTED_AFTERNOON_ENTRY_NO_INCREMENT"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False,
        "decision": "主基础和压力均低于原策略；较早只有原日线回退，不能证明下午改动有效。结束固定下午进入，不移动分钟、阈值或容量救回。",
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True)
    lines = ["# 510300下午提前进入：第75轮", "", "## 结果与决定", "",
        "**下午提前进入没有带来改善，结束这项改动。** 主评价基础费用净夏普从原0.705降至0.661，压力费用从0.641降至0.465，没有达到1.2。基础净利润76,324.55元，比原少6,809.02元；压力净利润46,507.24元，比原少27,323.41元。", "",
        "本轮保留原六十日日内相对隔夜进入和原八项学习退出，只增加一次下午进入机会。使用已有分钟数据和保存模型，没有补EPS、财报、公募或其他行情，没有训练新模型。", "",
        "## 策略怎样进入和退出", "",
        "昨日日内相对隔夜六十日强弱超过1，今天下午按当时价格计算的临时值也超过1，且真实空仓、有重新进入资格、距上次退出至少两个交易日，则在14:31形成买入请求，以14:46标签分钟开盘价作为之后的执行代理。信号用14:29标签分钟收盘价，不能提前知道最终日线信号。", "",
        "请求先按已知现金、信号价格、费用和整手计算；未来执行分钟价格或成交量不足只能让请求未成交，不能删除该请求。请求超过执行分钟量10%时整笔不成交，之后仍可按原日终规则请求下一开盘进入。", "",
        "退出保留原条件：完整价格转弱、6%亏损、8%回撤、最长60交易日，或者原模型连续两个收盘预测继续持有收益为负。任一触发，下一开盘请求全部卖出，受阻继续请求。下午买入不能当天卖出。卖出后等待旧进入条件消失并满足原两日冷却，再等待新机会。", "",
        "## 完整历史结果", "", "### 主评价：2020年1月2日至2026年8月14日开盘", ""]
    lines += table(result["all_metrics"])
    lines += ["### 较早历史：2015年1月5日至2019年12月31日开盘，仅检验原规则回退", ""] + table(result["earlier_diagnostics"])
    lines += ["两段各从20万元开始，1604日和1219日完整账户、全部空仓日保留。主基础累计收益38.16%，压力23.25%。较早两档所有完整经济账户和保存日终决策均与原策略一致，但较早完全没有分钟数据，因此表中的0.748及0.725只是原日线结果，不能拿来证明新下午时点有效。", "",
        "## 下午发生了多少次交易", "", "|主评价费用|下午请求|实际成交|现金导致少于请求的成交|分钟容量不足未成交|完整周期|持仓收盘数|", "|---|---:|---:|---:|---:|---:|---:|",
        "|基础|19|17|8|2|25|259|", "|压力|19|17|6|2|25|258|", "",
        "整笔未成交发生在2021年11月24日及2024年6月17日，两档相同。现金不足导致部分成交指通过容量检查后，后续价格与费用改变可负担整手数，不是把固定10%容量上限改成部分成交规则。主完整账户各50笔成交，比原多2笔。", "",
        "17次下午实际买入中，16次到当天收盘仍满足完整进入条件，1次已经消失。2026年5月27日下午已买入，完整收盘条件却不再成立，不能事后撤销这笔交易；基础账户于6月9日退出，该周期亏11,122.79元。", "",
        "## 费用为什么会扩大结果差距", "",
        "基础比原少赚6,809.02元，其中价格收益少6,520.10元、分红多466.40元、显式佣金滑点多755.32元。压力比原少27,323.41元，其中价格收益少26,626.70元。费用影响了实际份额和自己的持仓盈亏，进而能改变原模型判断，不能把压力账户理解为基础路径机械多扣一点手续费。", "",
        "2024年9月24日进入的一段最能说明这种影响：9月25日同一成熟模型在基础账户预测略为正，在压力账户略为负。压力更早累计两次负值，于9月27日退出；基础9月30日退出。这段基础净赚40,752.14元、压力18,463.83元。差异来自各自完整路径，不能全部解释成手续费差。", "",
        "|费用|9月25日继续持有预测|当天连续负值数|实际退出日|该周期净利润（元）|", "|---|---:|---:|---|---:|",
        "|基础|0.00003747927|0|2024年9月30日|40,752.14|", "|压力|负0.00003429821|1|2024年9月27日|18,463.83|", "",
        "这些例子用于解释已经保存的结果，不用来挑日期或改预测阈值。没有证明所有下午提前进入都无效；本轮固定时点、条件与完整退出连接方式没有产生改善。", "",
        "## 完整周期经济分解", "", "|历史|费用|周期数|盈利／亏损周期|价格损益（元）|分红（元）|佣金（元）|滑点（元）|净损益（元）|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups = pd.read_csv(RESEARCH / "saved_cycle_profit_groups.csv")
    for row in groups.itertuples():
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早回退'}|{'基础' if row.cost=='BASE' else '压力'}|{row.cycles}|{row.positive_cycles}／{row.negative_cycles}|{row.price_pnl:,.2f}|{row.dividend:,.2f}|{row.commission:,.2f}|{row.slippage:,.2f}|{row.net_profit:,.2f}|")
    lines += ["", "## 基础费用逐年结果", "", "|历史|年份|该年净收益|净夏普|年内最大回撤|成交笔数|", "|---|---|---:|---:|---:|---:|"]
    years = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in years[years.model.eq(PRIMARY) & years.cost.eq("BASE")].itertuples():
        sharpe = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "未定义"
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早回退'}|{row.year}|{row.cumulative_return:.2%}|{sharpe}|{-row.max_drawdown:.2%}|{row.trade_count}|")
    lines += ["", "2023年全年空仓、波动为零，夏普未定义，仍保留全年账户。2026年只到固定终点。个别年份夏普超过1.2，不能代替完整账户目标，也不能剔除不利年份。", "",
        "## 全部因子与完整中文规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()
    lines += [line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in protocol[1:]]
    lines += ["", "## 计算完成情况与文件", "",
        "完成四个新账户，其中两个较早账户仅用于回退一致性检查；复用八个旧对照。12项策略测试及1项保存比较测试通过。已保存复算1211日临时因子、12项账户指标、8组经济差额、38个下午请求、707个有效日终预测和68个含分红完整周期，四账户合计5646日。因子最大差为浮点精度范围内的约0.000000000000072。复核没有另跑账户或拟合模型，也不代表有效性通过。", "",
        "首次运行在保存三个账户后，因内存空值NaN与保存空值None的形式差异停在较早回退检查。经济账户比较此前已通过；修正只统一读取保存后的格式，保留缺失状态及三个账户原字节，只补剩下一个账户。原失败、原来源和修正来源均保留，因此统计为一个策略设置、两个来源版本。", "",
        "同目录附全部141个月度时点的中文模型说明：114个成熟模型的八项均值、尺度、系数与截距，以及27个无成熟模型时点；另附完整日终因子、下午信号请求、真实账户、成交、周期和差额。", "",
        "## 下一项", "",
        "转向原急跌回升与原学习退出两条规则之间的风险预算分配：用各自过去完整账户已经实现的波动决定预算，保留各自进出场。当前只是下一方向，尚未实现、登记或回测。继续暂停EPS、公募和全部慢来源，目标仍未完成。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    for name in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "afternoon_statistics.csv", "result.json",
        "saved_factor_replay.csv", "saved_metrics_recomputation.csv", "saved_account_differences.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv",
        "saved_afternoon_requests.csv", "saved_account_checks.csv", "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json", "RUN_STOPPED.json",
        "沿用的每月八项模型中文规则.md"]:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置及只修正保存比较.json")
    shutil.copy2(ROOT / "config/510300_afternoon_entry_v1.json", OUT / "原始冻结设置.json")
    shutil.copy2(ROOT / "reports/research/510300_afternoon_entry_saved_comparison_test.json", OUT / "保存比较测试回执.json")
    shutil.copy2(NEXT, OUT / "下一风险预算方向.md")
    pd.read_csv(RESEARCH / "all_partial_entry_factors.csv").rename(columns=FIELDS).to_csv(OUT / "1211日全部下午因子中文表.csv", index=False, encoding="utf-8-sig")
    example = []
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早回退")]:
        factor = pd.read_csv(RESEARCH / f"{period}_daily_factors.csv", parse_dates=["date"])
        for cost, cost_label in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet").to_csv(OUT / f"{label}_{cost_label}_完整账户.csv", index=False, encoding="utf-8-sig")
            d = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            d.merge(factor.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one").rename(columns=FIELDS).to_csv(
                OUT / f"{label}_{cost_label}_日终全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            pd.read_parquet(folder / "afternoon_decisions.parquet").rename(columns=FIELDS).to_csv(OUT / f"{label}_{cost_label}_下午因子请求和成交.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(folder / f"{PRIMARY}_trades.csv", OUT / f"{label}_{cost_label}_全部成交.csv")
            if period == "evaluation":
                rows = d[d.origin.between("2024-09-24", "2024-09-27")].copy()
                rows["cost"] = cost
                example.append(rows)
    pd.concat(example).to_csv(OUT / "2024年九月费用影响学习退出的保存记录.csv", index=False, encoding="utf-8-sig")
    record = {"round": 75, "study": result["study_id"], "title": "原退出不变，增加一次下午进入", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 2,
        "source_correction": "原三账户已保存，统一保存空值比较后只补一个账户；一个设置两个来源",
        "earlier_evidence": "ORIGINAL_DAILY_FALLBACK_ONLY", "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption"]:
        index[key] += 1
    for key in ["evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), status="ROUND75_COMPLETE_AFTERNOON_ENTRY_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计75轮，349不同设置，364已评价来源版本，369登记含5旧未运行，1168主评价记录。第75轮一个设置两个来源，原三账户保留只补一个。",
        checks="第75轮12策略测试和1保存比较测试；1211因子12指标8差额38下午请求707预测68周期复算，四路径5646日。",
        process_state_note="第75轮完整账户、失败归因、保存核对和中文交付完成；第76轮仅两策略风险预算方向，尚未登记或实现。",
        next_work={"status": "TWO_POLICY_RISK_BUDGET_DIRECTION_NOT_REGISTERED", "focus": "原急跌与学习规则不变，用各自已实现完整账户波动分配预算，直接真实合并账户检验", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_afternoon_entry_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "AFTERNOON_ENTRY_ROUND75_CHINESE_RESULTS", "rounds": [75], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "AFTERNOON_RULES_MODELS_AND_FULL_ACCOUNTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery, exclusive=True)
    print(json.dumps({"交付": delivery, "最新状态": index["status"], "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
