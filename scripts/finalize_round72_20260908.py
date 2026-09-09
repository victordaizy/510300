"""交付逐周期删除退出的完整中文规则、历史账户及失败归因。"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from research.deleted_cycle_stability_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import FEATURES, CN
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300逐周期删除退出_第72轮_20260908"
DOCUMENT = OUT / "逐周期删除退出_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_DELETED_CYCLE_POSITION_ONLY_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require([r["round"] for r in index["completed_rounds"]] == list(range(1, 72)), "索引不是截至71轮，不能重复收尾")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "已固定的本轮内容改变")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((RESEARCH / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(receipt["reviewer_source_sha256"] == digest(ROOT / "scripts/review_round72_saved.py"), "保存核对入口改变")
    require(not result["historical_point_target_met"], "不能按失败关闭已出现1.2的候选")
    require(all(metric(result, PRIMARY, p, c)["net_sharpe"] < metric(result, "REARM_RIDGE", p, c)["net_sharpe"] for p in ["evaluation", "earlier_diagnostic"] for c in cfg["costs"]), "两段两费用均变差结论不符")
    status = "COMPLETED_REJECTED_DELETED_CYCLE_UNANIMITY_DELAYED_EXITS"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False,
        "decision": "全部模型一致才退出使主净夏普由0.704868降为0.356497，主回撤由10.5912%扩大至25.2455%；较早和压力费用也更差。结束此一致退出规则，不调整投票比例、删除比例、阈值或确认天数救回。",
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300逐周期删除退出：第72轮", "", "## 结果与决定", "",
        "**本轮没有达到夏普1.2，结束这一改动。** 主评价基础净夏普从原平均收益模型的0.705降至0.356，压力从0.641降至0.289；较早基础从0.748降至0.700，压力从0.725降至0.677。两段两档费用的年化和最终资产也都下降，主基础最大回撤从10.59%扩大至25.25%。", "",
        "本轮直接使用现有日线、分红和已结束交易，不补EPS、财报、公募或其他新来源。原141个月度时点中114个具备成熟样本，各自依次删除一个完整旧周期并重新估计，共2055次删除模型拟合全部成功；27个早期时点没有成熟模型。这里有一个策略设置，2055次计算不是2055种被挑选的策略。", "",
        "## 策略怎样做", "",
        "进入继续使用日内相对隔夜强弱：近六十日日内表现持续强于隔夜，连续两个收盘满足条件，才在下一开盘买入。卖出后先等旧进入条件消失，再等待新的进入机会。", "",
        "持有后，先看原模型是否认为继续持有不划算；再依次拿掉一个已结束旧交易，检查重新估计的模型是否仍然这样判断。原模型和所有删除模型连续两天都给出负的继续收益，才增加下一开盘卖出请求。原价格转弱、亏损6%、回撤8%、最长60交易日退出一直有效。", "",
        "这个改动想减少依赖单次旧行情的脆弱退出。实际结果却是：模型意见只要有一个不同就延迟学习退出，更长的持仓扩大了一些亏损和盈利回吐。模型之间更一致，不能直接等同于投资结果更稳定。", "",
        "## 两段历史的完整表现", "", "### 主评价：2020年1月2日至2026年8月14日开盘", ""]
    lines += table(result["all_metrics"])
    lines += ["### 较早诊断：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["两段分别从20万元开始，包含1604日和1219日的完整账户路径及全部空仓日。新策略基础累计收益分别22.42%和47.85%，压力为17.09%和45.77%。这些历史已被多轮研究使用，不能视为新独立样本。", "",
        "## 为什么失败", "",
        "主评价基础相较原平均收益退出少赚38,285.79元：价格收益减少46,177.40元，分红增加5,904.10元，佣金和滑点反而节省1,987.51元。所以这次损失主要来自实际持仓路径，并不是交易费用太高。较早基础少赚7,870.37元，其中价格收益减少7,902.00元、费用节省31.63元。", "",
        "主基础从原24个周期、252个持仓收盘，变成21个周期、343个持仓收盘，平均每周期从10.5个收盘延长至16.33个。与原策略有19个相同进入原点，新增2个，缺少5个；退出时点改变后，下一次可以买入的机会也会改变。较早9个进入原点全部相同，持仓收盘从299增加到304。", "",
        "在新账户实际出现的同一持仓状态上，主基础343次有效判断中，有101次原模型预测继续收益为负，却被至少一个删除模型的非负预测阻止计入负值确认。压力为100次；较早两档费用各6次。主各16个周期出现学习退出，较早只1个、原模型为2个。这里比较的是同一实际输入上的模型意见，不是额外运行的可交易方案。", "",
        "延迟退出既有损失也有获益，下面同时列出。全部匹配周期另附，未按照这些日期调整规则。", "",
        "|进入决定日|新退出日|原退出日|新周期净收益|原周期净收益|观察|", "|---|---|---|---:|---:|---|"]
    matches = pd.read_csv(RESEARCH / "saved_entry_matched_cycle_comparison.csv")
    main = matches[matches.period.eq("evaluation") & matches.cost.eq("BASE")]
    for origin, comment in [("2021-06-07", "更晚退出，亏损扩大"), ("2020-06-30", "更晚退出，盈利回吐"), ("2024-09-24", "更晚退出，保留更多上涨")]:
        row = main[main.entry_origin.eq(origin)].iloc[0]
        lines.append(f"|{origin}|{row.exit_date_new}|{row.exit_date_old}|{row.cycle_net_return_new:.2%}|{row.cycle_net_return_old:.2%}|{comment}|")
    lines += ["", "周期净收益以各自实际买入金额加佣金为分母，包括归属于该持仓的分红权益。单个周期收益差不能直接当作整个账户的因果差额。收盘触发止损后在下一开盘执行，6%的止损触发值也不保证实际损失被限制在6%。", "",
        "## 全部周期经济结果", "", "|历史|费用|周期数|盈利／亏损周期|价格损益（元）|分红（元）|佣金（元）|滑点（元）|净损益（元）|持仓收盘数|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pd.read_csv(RESEARCH / "saved_cycle_profit_groups.csv").itertuples():
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早历史'}|{'基础' if row.cost=='BASE' else '压力'}|{row.cycles}|{row.positive_cycles}／{row.negative_cycles}|{row.gross_price_profit:,.2f}|{row.dividend_recognized:,.2f}|{row.commission:,.2f}|{row.slippage:,.2f}|{row.net_profit:,.2f}|{row.held_closes}|")
    lines += ["", "压力主评价持仓342个收盘，比基础少1个。两档费用共享同一组训练模型，但按自己的实际成本、份额、浮盈亏和回撤决定退出，因此允许出现不同轨迹。本轮四个账户均没有整笔受阻未成交请求。", "",
        "## 基础费用逐年表现", "", "|历史|年份|该年净收益|净夏普|年内最大回撤|成交笔数|", "|---|---|---:|---:|---:|---:|"]
    years = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for row in years[years.model.eq(PRIMARY) & years.cost.eq("BASE")].itertuples():
        sharpe = f"{row.net_sharpe:.3f}" if pd.notna(row.net_sharpe) else "未定义"
        lines.append(f"|{'主评价' if row.period=='evaluation' else '较早历史'}|{row.year}|{row.cumulative_return:.2%}|{sharpe}|{-row.max_drawdown:.2%}|{row.trade_count}|")
    lines += ["", "2023年全年空仓、波动为零，夏普未定义，不填零或删除该年。2026年只到固定终点。单年超过1.2不代表整体达标。年内回撤从该年起点计算，完整账户回撤包含此前完整高点。", "",
        "## 全部因子和完整中文进出场规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()
    lines += [line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in protocol[1:]]
    lines += ["", "## 交付与核对", "",
        "同目录“每月删除周期退出模型中文规则.md”和“全部月度模型系数.csv”列出114个原模型及2055个删除模型的完整八项均值、标准差、系数和截距。中文展示10位小数，保存模型使用完整精度。四份逐日因子文件列出真实持仓输入、原模型预测、删除模型预测范围、全部模型最大预测、非负模型数量、连续计数和进出场请求；另有完整账户、成交、全部周期及费用差额。", "",
        "10项测试一次通过。已独立核对3456日进入和价格因子、141个训练时点及90,217条月度成员、2055个删除模型的成员与标准化，并检查保存系数满足原加权岭回归条件。全部原完整模型保持原内容。20项账户指标、16组经济差额、5646个份额请求、885个持仓预测和60个含分红完整周期均能从保存记录复算；预测差异仅在浮点精度范围内。较早两档费用各204个无模型持仓日继续原价格退出，预测保留缺失。以上复核没有新拟合或模拟，不代表策略有效性通过。", "",
        "## 下一项：缩减退出模型", "",
        "下一项只保留持有时间、实际含分红盈亏、实际含分红回撤三个持仓变量，重新估计原平均继续收益模型。删除全部四项市场价格预测因子和固定进入类别，检验更少输入能否改善完整账户。进入、成熟样本、费用和原价格退出保持原义。只确定了这一方向，尚未登记或运行下一轮，不能称为已经改善。", "",
        "继续暂停EPS及慢源补齐。不把本轮全部同意改成多数投票、不调整删除比例或确认天数救回，也不恢复上轮中位数。目标仍未实现，继续检验有依据的简单改动。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    files = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "model_coverage.csv", "result.json", "training_receipts.csv",
        "deletion_training_receipts.csv", "每月删除周期退出模型中文规则.md", "全部月度模型系数.csv", "saved_models.json", "saved_all_entry_factors.csv",
        "saved_training_support_replay.csv", "saved_deletion_training_replay.csv", "saved_account_metrics.csv", "saved_account_differences.csv",
        "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv", "saved_prediction_comparison.csv", "saved_entry_matched_cycle_comparison.csv",
        "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json"]
    for name in files:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    factors = pd.read_csv(RESEARCH / "saved_all_entry_factors.csv", parse_dates=["date"])
    factor_fields = factors[["date", "d60_factor", "entry_condition", "original_price_exit"]]
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost, cost_label in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            ledger.to_csv(OUT / f"{label}_{cost_label}_完整账户.csv", index=False, encoding="utf-8-sig")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            decisions = decisions.merge(factor_fields.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            decisions.rename(columns={"origin": "收盘决定日", "execution_date": "实际计划执行日", "action": "中文动作", "requested_quantity": "请求份额",
                "exit_reasons": "退出原因", "entry_rearmed": "再次进入资格", "continuation_prediction": "全部模型预测最大值",
                "base_continuation_prediction": "原模型继续收益预测", "deleted_max_prediction": "删除模型预测最大值", "deleted_min_prediction": "删除模型预测最小值",
                "committee_size": "全部模型数量", "nonnegative_predictions": "非负预测模型数量", "learning_fit_origin": "模型训练日",
                "negative_confirmation_count": "连续全部负值次数", "learned_exit_requested": "学习退出请求", "learning_status": "模型状态",
                "d60_factor": "日内相对隔夜六十日强弱", "entry_condition": "原进入条件", "original_price_exit": "原价格退出", **dict(zip(FEATURES, CN))}).to_csv(
                OUT / f"{label}_{cost_label}_逐日全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(folder / f"{PRIMARY}_trades.csv", OUT / f"{label}_{cost_label}_全部成交.csv")
    record = {"round": 72, "study": result["study_id"], "title": "逐个删除旧周期且全部模型同意退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), status="ROUND72_COMPLETE_DELETED_CYCLE_UNANIMITY_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计72轮，346不同设置，360已评价来源版本，365登记含5旧未运行，1142主评价记录。第72轮2055子拟合组成一个策略，114成熟月份及27原样本不足月份另计。",
        checks="第72轮10测试、141时点90217成员2055删除模型、20指标16差额5646请求885预测及60含分红周期复算完成。",
        process_state_note="第72轮训练、完整账户、失败归因和中文交付完成；第73轮仅确定缩减为三项持仓输入的方向，尚未登记或拟合。",
        next_work={"status": "POSITION_STATE_ONLY_EXIT_DIRECTION_NOT_REGISTERED", "focus": "移除四项市场价格预测输入及固定类别，只用持有时长实际盈亏回撤重拟合继续收益，唯一设置", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_deleted_cycle_stability_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "DELETED_CYCLE_STABILITY_ROUND72_CHINESE_RESULTS", "rounds": [72], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "DELETED_CYCLE_RULES_MODELS_AND_FULL_ACCOUNTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery, exclusive=True)
    print(json.dumps({"交付": delivery, "最新状态": index["status"], "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
