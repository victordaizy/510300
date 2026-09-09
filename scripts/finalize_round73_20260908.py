"""交付三项持仓退出的完整历史结果和全部中文规则。"""
from __future__ import annotations

import json
import shutil
import pandas as pd

from research.position_state_exit_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.position_state_exit_inputs_v1 import FEATURES, CN
from research.intraday_overnight_increment_v1 import now, digest, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300三项持仓退出_第73轮_20260908"
DOCUMENT = OUT / "三项持仓退出_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_POSITION_STATE_PARABOLIC_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require([r["round"] for r in index["completed_rounds"]] == list(range(1, 73)), "索引不是截至72轮，不能重复收尾")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "三项模型已固定内容改变")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((RESEARCH / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(receipt["reviewer_source_sha256"] == digest(ROOT / "scripts/review_round73_saved.py"), "保存核对入口改变")
    require(not result["historical_point_target_met"], "不能按失败关闭已出现1.2的候选")
    status = "COMPLETED_REJECTED_POSITION_STATE_ONLY_MAIN_PROFIT_LOST"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False,
        "decision": "三项实际持仓退出主基础净夏普从原0.704868降至0.029075，压力为负；较早收益增加但两档夏普略低。结束本次特征组缩减，不扫描其他因子数量或组合救回。",
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300三项实际持仓退出：第73轮", "", "## 结果与决定", "",
        "**本轮没有达到净夏普1.2，结束这项简化。** 主评价基础净夏普从原八项模型的0.705降至0.029，压力费用下从0.641降至负0.045。较早基础从0.748降至0.734，压力从0.725降至0.713。主评价最终亏损；较早收益略增，仍不能称为跨时期改善。", "",
        "本轮只用已有数据，真正重新估计三项输入的平均继续收益模型：持有时间、实际含分红盈亏、实际含分红回撤。统一移除四项市场价格预测因子和固定进入类别。没有把旧模型系数置零，没有改变原训练目标和样本，没有补EPS或公募资料。", "",
        "141个原月度时点中114次真实拟合全部成功，27个较早时点仍因成熟样本不足没有学习模型。完成四个新账户，16个旧对照直接复用。", "",
        "## 策略怎样进入和退出", "",
        "连续两个收盘出现原日内相对隔夜强势条件，空仓且具有进入资格时，在下一开盘买入。持仓后，三项模型根据自己已持有多久、当前实际盈亏和回撤，判断继续持有相对提前卖出是否划算；预测连续两天为负，请求下一开盘退出。原价格转弱、6%亏损触发、8%回撤触发和最长60交易日退出仍然有效。卖出后先等旧进入条件消失，再等待新机会。", "",
        "三项模型重新拟合全部系数，只看已结束的历史参考交易。不同费用账户用各自实际成交和分红计算输入。没有成熟模型时，预测保留缺失，原价格和时间退出继续工作。", "",
        "## 两段完整历史结果", "", "### 主评价：2020年1月2日至2026年8月14日开盘", ""]
    lines += table(result["all_metrics"])
    lines += ["### 较早诊断：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["两段各从20万元开始，保留1604日和1219日完整账户路径及全部空仓日。主基础累计负0.63%、压力负4.54%；较早基础正53.77%、压力正51.63%。基础主年化为负0.09%，净夏普略为正，来自算术平均日收益与复合累计收益的区别，不能据此称主账户赚钱。", "",
        "## 这次失败说明什么", "",
        "主基础比原模型少赚84,387.40元：价格收益减少88,410.40元，分红增加3,870.50元，佣金与滑点反而节省152.50元。主要问题是持仓路径和价格收益。较早基础比原多赚3,984.89元，年化从8.64%升到8.92%，但波动也增大，净夏普略低。不能把较早结果写成收益也下降，也不能用这一局部收益增长掩盖主评价恶化。", "",
        "主基础29个完整周期中17个赚钱、12个亏钱，盈利周期净利润合计62,594.91元，亏损周期合计负63,848.73元，最终亏1,253.82元。盈利次数更多也可能亏钱，夏普目标需要考察完整收益大小和波动。", "",
        "主评价从原24周期、252个持仓收盘，变为29周期、207个收盘，平均每周期从10.5缩到7.14个收盘；新原共有22个进入原点，只新7个、只原2个，27周期出现学习退出。较早9个进入原点全部相同，持仓收盘却从299延长到337，学习退出从2次变为1次。简化在不同历史下可能提前或推迟卖出，不能只用一个方向解释。", "",
        "在新实际持仓状态上对比保存的原八项模型：主每档207次有效预测中，三项为负、原八项非负38次；三项非负、原八项为负22次。较早各133次有效预测中相应为1次和26次。这是相同真实输入上的解释性比较，不是额外交易策略。", "",
        "在本轮固定样本、原惩罚与原进入条件下，直接去掉整组市场状态输入没有取得改善；这也不证明原四项市场因子每一项都独立有效，或者会长期有效。", "",
        "|进入决定日|新退出日|原退出日|新周期净收益|原周期净收益|观察|", "|---|---|---|---:|---:|---|"]
    matches = pd.read_csv(RESEARCH / "saved_entry_matched_cycle_comparison.csv")
    main = matches[matches.period.eq("evaluation") & matches.cost.eq("BASE")]
    for origin, comment in [("2024-09-24", "提前卖出，少保留上涨"), ("2021-06-07", "推迟卖出，亏损扩大"), ("2024-08-27", "提前卖出，减少亏损")]:
        row = main[main.entry_origin.eq(origin)].iloc[0]
        lines.append(f"|{origin}|{row.exit_date_new}|{row.exit_date_old}|{row.cycle_net_return_new:.2%}|{row.cycle_net_return_old:.2%}|{comment}|")
    lines += ["", "正反例全部来自保存的匹配周期，完整表另附。周期收益按各自含买入佣金的实际成本计算，并包含归属持仓的分红权益。不能按这些日期挑选新规则，也不能把单周期收益率差当作整个账户的因果差额。", "",
        "## 完整周期经济结果", "", "|历史|费用|周期数|盈利／亏损周期|价格损益（元）|分红（元）|佣金（元）|滑点（元）|净损益（元）|持仓收盘数|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pd.read_csv(RESEARCH / "saved_cycle_profit_groups.csv").itertuples():
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早历史'}|{'基础' if row.cost=='BASE' else '压力'}|{row.cycles}|{row.positive_cycles}／{row.negative_cycles}|{row.gross_price_profit:,.2f}|{row.dividend_recognized:,.2f}|{row.commission:,.2f}|{row.slippage:,.2f}|{row.net_profit:,.2f}|{row.held_closes}|")
    lines += ["", "四个账户均没有整笔受阻未成交请求。较早两档费用各有204个持仓判断因缺成熟模型而没有学习预测，仍执行原价格退出；缺失没有被填成零。两档费用本轮的持仓日期相同，但实际份额、成本和净收益不同。", "",
        "## 基础费用逐年结果", "", "|历史|年份|该年净收益|净夏普|年内最大回撤|成交笔数|", "|---|---|---:|---:|---:|---:|"]
    years = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in years[years.model.eq(PRIMARY) & years.cost.eq("BASE")].itertuples():
        sharpe = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "未定义"
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早历史'}|{row.year}|{row.cumulative_return:.2%}|{sharpe}|{-row.max_drawdown:.2%}|{row.trade_count}|")
    lines += ["", "2023年全年空仓、波动为零，夏普未定义，不填零或删除该年。2026年只到固定终点。单年超过1.2不是整体达标。年内回撤与完整账户回撤的起算高点不同。", "",
        "## 全部因子与完整中文规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()
    lines += [line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in protocol[1:]]
    lines += ["", "## 保存内容和核对", "",
        "同目录“每月三项持仓退出模型中文规则.md”列出全部114个成熟模型的三个均值、标准差、系数与截距，并说明27个无模型月份。四份逐日因子和进出场文件列出原进入因子、自己的三个持仓变量、模型日期、预测、连续计数、请求及退出原因；另附完整账户、成交、周期归因和新旧差额。", "",
        "10项关键测试一次通过。已复算3456日原进入与价格因子、141个月度时点的90,217条训练归属记录、114个三项模型的标准化和加权岭目标条件、20项账户指标、16组经济差额、5646个份额请求、680个有效持仓预测及76个含分红完整周期。预测只存在浮点精度范围内差异，408个无模型持仓日保留缺失。复核没有重新拟合或模拟，也不表示有效性通过。", "",
        "## 下一项：独立价格转向规则", "",
        "下一项改用价格极值推动保护价的抛物线转向规则，独立定义进入与退出，不继续增删本轮学习因子。方向持续并出现新极值时，保护价逐渐加速靠近；触及保护价后改变方向，下一开盘执行，只做多或持现金。它也可能在震荡中反复亏损，因此只登记一个常规设置，直接检验完整账户。", "",
        "当前只完成方向查重和官方定义阅读，尚未登记、实现或回测下一轮。不调本轮因子数量救回，不补EPS及慢源，不制作GPT数值包。目标仍未完成，持续研究继续。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    files = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "model_coverage.csv", "result.json", "training_receipts.csv",
        "每月三项持仓退出模型中文规则.md", "saved_models.json", "saved_all_entry_factors.csv", "saved_training_support_replay.csv", "saved_account_metrics.csv",
        "saved_account_differences.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv", "saved_prediction_comparison.csv", "saved_entry_matched_cycle_comparison.csv",
        "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json"]
    for name in files:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    factors = pd.read_csv(RESEARCH / "saved_all_entry_factors.csv", parse_dates=["date"])
    factor_fields = factors[["date", "d60_factor", "entry_condition", "original_price_exit"]]
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost, cost_label in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet").to_csv(OUT / f"{label}_{cost_label}_完整账户.csv", index=False, encoding="utf-8-sig")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            decisions = decisions.merge(factor_fields.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            decisions.rename(columns={"origin": "收盘决定日", "execution_date": "实际计划执行日", "action": "中文动作", "requested_quantity": "请求份额",
                "exit_reasons": "退出原因", "entry_rearmed": "再次进入资格", "continuation_prediction": "三项持仓继续收益预测", "learning_fit_origin": "模型训练日",
                "negative_confirmation_count": "连续负值次数", "learned_exit_requested": "学习退出请求", "learning_status": "模型状态",
                "d60_factor": "日内相对隔夜六十日强弱", "entry_condition": "原进入条件", "original_price_exit": "原价格退出", **dict(zip(FEATURES, CN))}).to_csv(
                OUT / f"{label}_{cost_label}_逐日全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(folder / f"{PRIMARY}_trades.csv", OUT / f"{label}_{cost_label}_全部成交.csv")
    record = {"round": 73, "study": result["study_id"], "title": "只用持有时间实际盈亏与回撤学习退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), status="ROUND73_COMPLETE_POSITION_STATE_ONLY_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计73轮，347不同设置，361已评价来源版本，366登记含5旧未运行，1152主评价记录。第73轮114月度拟合与27无模型时点另计，一个三项设置。",
        checks="第73轮10测试、141时点90217训练成员114三项模型、20指标16差额5646请求680预测及76含分红周期复算完成。",
        process_state_note="第73轮训练、完整账户、失败归因和中文交付完成；第74轮仅独立抛物线转向方向，尚未登记或实现。",
        next_work={"status": "PARABOLIC_REVERSAL_DIRECTION_NOT_REGISTERED", "focus": "只用已有价格极值及递增保护价独立定义进出场，不继续修改学习因子", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_position_state_exit_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "POSITION_STATE_EXIT_ROUND73_CHINESE_RESULTS", "rounds": [73], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "POSITION_STATE_RULES_MODELS_AND_FULL_ACCOUNTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery, exclusive=True)
    print(json.dumps({"交付": delivery, "最新状态": index["status"], "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
