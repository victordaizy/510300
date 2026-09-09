"""完成第37至40轮保存结果交付、失败归因和持续研究状态更新。"""
import json
from pathlib import Path
import pandas as pd
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.deliver_simple_strategy_rounds34_36_20260907 import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300状态切换与近期学习_第37至40轮_20260907"
DIAG = ROOT / "reports/research/510300_simple_adaptation_saved_diagnostic_v1"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
STUDIES = [
    (37, "volume_weighted_trend", "成交量加权均线与普通均线", "VWMA_20_60", "STOP_VOLUME_WEIGHTED_TREND_TUNING"),
    (38, "recency_weighted_exit", "旧交易随时间降低训练权重", "RECENCY_RIDGE", "STOP_CALENDAR_DECAY_TUNING"),
    (39, "market_state_signal_router", "每天按强趋势和其余状态选择策略", "STATE_ROUTER", "STOP_DAILY_REGIME_THRESHOLD_TUNING"),
    (40, "committed_state_router", "入场时选定策略，退出后再选择", "COMMITTED_ROUTER", "STOP_COMMITMENT_COOLDOWN_TUNING"),
]


def sources(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def saved_diagnostics(results):
    DIAG.mkdir(parents=True, exist_ok=True)
    delta_rows, switch_events = [], []
    for n, slug, title, primary, decision in STUDIES:
        if n == 37:
            continue
        folder = sources(slug)
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                current = pd.read_parquet(folder / period / cost / f"{primary}_ledger.parquet")
                baselines = ["REARM_RIDGE"] + (["STATE_ROUTER"] if n == 40 else [])
                for baseline in baselines:
                    reference = pd.read_parquet(folder / period / cost / f"{baseline}_ledger.parquet")
                    equity = float(current.equity.iloc[-1] - reference.equity.iloc[-1])
                    price = float(current.price_pnl.sum() - reference.price_pnl.sum())
                    dividend = float(current.dividend_recognized.sum() - reference.dividend_recognized.sum())
                    friction = float((current.commission + current.slippage_cost).sum() - (reference.commission + reference.slippage_cost).sum())
                    require(abs(equity - price - dividend + friction) < 1e-6, "保存账户增量不能与损益核对")
                    delta_rows.append({"round": n, "period": period, "cost": cost, "model": primary, "baseline": baseline,
                                       "terminal_equity_difference": equity, "price_pnl_difference": price, "dividend_difference": dividend,
                                       "extra_commission_and_slippage": friction})
                if n == 39:
                    states = pd.read_parquet(folder / f"{period}_states.parquet").set_index("date")
                    prior_selector = states.strong_market.shift(1, fill_value=False)
                    old_keeps_holding = states.learned_state.where(~prior_selector, states.trend_state).eq(1)
                    forced = states.strong_market.ne(prior_selector) & old_keeps_holding & states.target.eq(0) & states.target.shift(1).eq(1)
                    sales = current[current.filled_quantity < 0].copy()
                    sales["switch_only"] = sales.origin.map(forced).fillna(False)
                    for row in sales[sales.switch_only].itertuples():
                        switch_events.append({"period": period, "cost": cost, "decision_date": row.origin, "actual_sale_date": row.date,
                                              "actual_quantity": -row.filled_quantity, "old_source_still_wanted_stock": True})
    pd.DataFrame(delta_rows).to_csv(DIAG / "saved_increment_decomposition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(switch_events).to_csv(DIAG / "routing_forced_exit_events.csv", index=False, encoding="utf-8-sig")
    p38 = sources("recency_weighted_exit")
    cfg = json.loads((ROOT / "config/510300_recency_weighted_exit_v1.json").read_text(encoding="utf-8"))
    checks = []
    for period, key, expected_days in [("evaluation", "all_metrics", 1604), ("earlier_diagnostic", "earlier_diagnostics", 1219)]:
        for cost in ["BASE", "STRESS"]:
            saved = pd.read_parquet(p38 / period / cost / "RECENCY_RIDGE_ledger.parquet")
            original = next(m for m in results[38][key] if m["model"] == "RECENCY_RIDGE" and m["cost"] == cost)
            computed = summarize(saved, cfg)
            require(len(saved) == expected_days and not saved.terminal_unliquidated.iloc[-1], "控制台报错前保存的账户日期或清算不完整")
            require(saved.accounting_error.abs().max() < 1e-6, "加权退出保存账户财富恒等式失败")
            for field in ["net_sharpe", "annualized_return", "cumulative_return", "max_drawdown", "commission", "slippage_cost", "trade_count"]:
                require(abs(computed[field] - original[field]) < 1e-9, "加权退出保存结果与账户重算不同")
            cycles = pd.read_csv(p38 / period / cost / "RECENCY_RIDGE_cycles.csv")
            require((cycles.holding_intervals >= 1).all(), "加权退出保存周期违反次日可卖")
            checks.append({"period": period, "cost": cost, "saved_rows": len(saved), "metrics_recomputed_from_saved_account": True})
    memberships = pd.read_parquet(p38 / "training_memberships.parquet")
    require((memberships.exit_index <= memberships.fit_index).all(), "加权退出模型训练包含未来周期")
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "加权退出实际执行的冻结内容变化")
    completion = {"recorded_at": now(), "status": "COMPLETE_SAVED_NUMERICAL_RESULTS_AFTER_FINAL_CONSOLE_DISPLAY_ERROR",
                  "original_process_exit_code": 1, "error_stage": "FINAL_CONSOLE_JSON_SERIALIZATION_AFTER_RESULT_FILE_WRITE",
                  "error": "最终控制台显示包含日期对象，标准JSON打印报错；全部模型、账户和result.json此前均已保存。",
                  "resolution": "本脚本从保存账户独立重算指标并恢复中文结果显示，保留原冻结执行代码，不重跑模型或账户。",
                  "saved_account_checks": checks, "mature_training_memberships_checked": len(memberships),
                  "new_accounts_generated": 0, "new_model_fits": 0, "research_parameters_changed": False,
                  "source_result_sha256": digest(p38 / "result.json")}
    write_json(p38 / "completion_receipt.json", completion)
    write_json(DIAG / "result.json", {"study_id": "510300_SIMPLE_ADAPTATION_SAVED_DIAGNOSTIC_V1", "completed_at": now(),
               "status": "COMPLETE_SAVED_ACCOUNT_ATTRIBUTION_AND_REPORTING_RECOVERY", "account_comparisons": len(delta_rows),
               "forced_exit_events_including_two_costs": len(switch_events), "new_accounts_generated": 0, "new_model_fits": 0,
               "goal_achieved": False, "position_impact": 0})
    return delta_rows, switch_events, completion


def correct_old_training_count():
    result = json.loads((sources("learned_cycle_exit") / "result.json").read_text(encoding="utf-8"))
    receipts = pd.read_csv(sources("learned_cycle_exit") / "training_receipts.csv")
    count = int((receipts.status != "FIT_COMPLETE").sum())
    require(count == result["no_view_fit_origins"] == 140 and len(receipts) == 846, "第31轮训练计数不能与原记录核对")
    document = ROOT / "deliverables/510300学习退出与重新进入_20260907/学习退出与重新进入_全部因子规则及历史表现.md"
    old_content = document.read_text(encoding="utf-8")
    new_content = old_content.replace("134个模型更新时点", "140个模型更新时点")
    if new_content != old_content:
        document.write_text(new_content, encoding="utf-8")
    builder = ROOT / "scripts/deliver_learned_cycle_and_rearm_20260907.py"
    builder.write_text(builder.read_text(encoding="utf-8").replace("134个模型更新时点", "140个模型更新时点"), encoding="utf-8")
    receipt_file = document.parent / "delivery_receipt.json"
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    for row in receipt["files"]:
        if row["path"] == document.name:
            row["sha256"] = digest(document)
    receipt["training_count_text_correction"] = {"corrected_at": now(), "old_no_model_update_count": 134, "actual_no_model_update_count": 140,
                                                "successful_fits_unchanged": 706, "actual_update_records": 846,
                                                "account_results_changed": False}
    write_json(receipt_file, receipt)
    return receipt["training_count_text_correction"]


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed == set(range(1, 37)) or completed == set(range(1, 41)), "研究索引已出现其他新轮次，停止覆盖")
    OUT.mkdir(parents=True, exist_ok=True)
    results = {n: json.loads((sources(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, title, primary, decision in STUDIES}
    delta_rows, forced, completion = saved_diagnostics(results)
    count_correction = correct_old_training_count()
    copies, records = [], []
    for n, slug, title, primary, decision in STUDIES:
        result = results[n]
        record = {"round": n, "study": result["study_id"], "title": title, "status": "COMPLETED_TARGET_NOT_MET_NO_IMPROVEMENT",
                  "result": f"reports/research/510300_{slug}_v1/result.json", "candidate_configurations": result["candidate_configurations"],
                  "evaluated_candidate_source_runs": result["candidate_configurations"], "evaluation_accounts": result["evaluation_accounts"],
                  "new_accounts_generated": result["new_accounts_generated"], "reused_control_accounts": result["reused_control_accounts"],
                  "earlier_diagnostic_accounts": result["earlier_diagnostic_accounts"], "new_earlier_diagnostic_accounts": result["new_earlier_diagnostic_accounts"],
                  "new_model_fits": result.get("completed_fits", 0), "new_reference_accounts": 0,
                  "primary_base": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "BASE"),
                  "primary_stress": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "STRESS"),
                  "post_selected_best_base": result["post_selected_best_base"]}
        if n == 38:
            record["completion_receipt"] = str((sources(slug) / "completion_receipt.json").relative_to(ROOT))
        records.append(record)
        write_json(sources(slug) / "acceptance_outcome.json", {"recorded_at": now(), "status": record["status"], "decision": decision,
                   "goal_achieved": False, "position_impact": 0})
        for filename, label in [("metrics.csv", "全部主评价"), ("earlier_diagnostics.csv", "全部较早诊断"),
                                ("yearly_metrics.csv", "逐年结果"), ("era_metrics.csv", "分阶段结果")]:
            target = OUT / f"第{n}轮_{label}.csv"
            target.write_bytes((sources(slug) / filename).read_bytes())
            copies.append(target)
    for source, filename in [(sources("volume_weighted_trend") / "volume_increment.csv", "成交量权重_实际账户增量.csv"),
                             (sources("recency_weighted_exit") / "training_receipts.csv", "训练样本年龄_逐月完整记录.csv"),
                             (sources("recency_weighted_exit") / "cycle_weights_and_ages.csv", "每次训练_完整周期的年龄和实际权重.csv"),
                             (sources("recency_weighted_exit") / "model_coverage.csv", "加权退出_真实持仓中的模型覆盖.csv"),
                             (sources("recency_weighted_exit") / "每月加权退出模型中文规则.md", "按年龄加权退出_逐月实际中文规则.md"),
                             (sources("committed_state_router") / "mode_cycle_results.csv", "锁定策略_各模式实际周期收益.csv"),
                             (DIAG / "saved_increment_decomposition.csv", "全部改动_保存账户损益与费用归因.csv"),
                             (DIAG / "routing_forced_exit_events.csv", "每日选择_仅因改选而实际卖出的记录.csv")]:
        target = OUT / filename
        target.write_bytes(source.read_bytes())
        copies.append(target)
    content = ["# 510300：状态切换、近期学习和交易成本的检验", "", "第37至40轮，生成于2026年9月7日；行情终点2026年8月14日开盘。", "",
               "## 先看结论", "", "继续检验了四种改动及一项普通均线对照，净夏普1.2仍未达到。保留的比较基线仍是第32轮‘线性退出＋等待新机会’：主评价基础净夏普0.705、压力0.641，较早基础0.748、压力0.725。主评价年化5.38%、最大回撤10.59%，24次完整买卖。它仍是历史比较候选，没有确认独立的稳定超额。", "",
               "|新增设置|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|决定|", "|---|---:|---:|---:|---:|---|",
               "|成交量加权二十日及六十日均线|−0.530|−0.651|0.367|0.277|未改善，停止。|",
               "|同规则普通均线对照|−0.478|−0.598|0.239|0.143|未达标，保留比较记录。|",
               "|旧训练交易每年权重减半|0.464|0.411|0.612|0.588|两个时期均低于原候选，停止。|",
               "|每天按强趋势或其余状态选择策略|0.498|0.402|0.671|0.606|没有覆盖新增成本，停止。|",
               "|每次进入锁定策略，卖完后再选择|0.357|0.251|0.774|0.715|较早略好、近年更差，不采用。|", "",
               "## 这次更清楚的失败原因", "",
               "第一，成交量加权没有提供足够收益信息。相同进入退出规则下，加权均线主评价比普通均线少赚3834.23元，费用反而少216.77元，主要差异来自价格损益。较早时期加权较好，但不能覆盖近年的相反结果。", "",
               "第二，‘最近二十笔’确实可能很旧，但只提高近期样本权重也没有改善。2026年8月3日退出模型的二十个周期中，16个结束超过一年，11个超过三年，最早为2018年10月19日。本轮把每个周期的权重按242个交易日减半，并保持总训练权重与原模型相同；完成114次训练，27个时点样本不足。最新加权后的有效周期数只有8.09。主评价比原候选少赚23258.04元，额外费用为负524.96元，即费用还略省，亏差主要来自改变退出时点后的价格损益。旧样本的年龄事实与‘旧样本导致失败’是两个不同结论，后者未被本轮证实。", "",
               "第三，每天切换策略会产生真实交易成本。第39轮主评价相对原候选少赚9640.16元，其中额外佣金及滑点8374.66元、价格损益少532.70元、分红少732.80元。买卖笔数从48笔增加到96笔。较早时期价格及分红合计多赚3179.70元，但多付8632.49元费用，最终仍少赚5452.79元。", "",
               "第四，固定持仓期间采用的策略，也不能保证解决选择失误。第39轮有4次主评价实际卖出只因改选策略而发生，旧策略当时仍愿意持有；较早有7次。本轮第40轮允许实际入场后一直采用原选定策略的退出状态。主评价买卖仍为96笔，夏普进一步降至0.357；分给趋势策略的27个实际周期合计亏损10932.33元，分给日内强弱学习退出的21个周期盈利54294.39元。两个模式的损益都是本次选择账户中的实际结果，不能直接当作两个独立账户收益率。", "",
               "这些结果支持把研究重点放在是否有新增、可兑现的收益信息上。仅增加因子、偏重新数据或按状态换策略，都不足以自动提高夏普。既有强趋势标签描述当前价格路径，尚不能证明此时改用突破策略能获得更好的后续净收益。", "",
               "## 统一比较口径", "", "主评价为2020年1月2日至2026年8月14日开盘，1604个账户日；较早诊断为2015年1月5日至2019年12月31日开盘，1219日。各从20万元现金开始，只510300与现金，所有空仓日也计入评价。较早历史已被看过，不能称为新的独立验证。", "",
               "基础费用佣金万分之二、最低5元、滑点万分之五；压力费用佣金万分之四、最低5元、滑点千分之一。年化242日，现金与无风险收益零，完整分红、100份整手、0.001元价位、方向性涨跌停及买入次日可卖，终点统一开盘退出。买卖笔数分别计算买入与卖出，平均仓位包含空仓日，回撤负数表示从先前净值高点下降。", ""]
    for n, slug, title, primary, decision in STUDIES:
        content += [f"## 第{n}轮：{title}的全部结果", "", "### 主评价", ""] + table(results[n]["all_metrics"])
        content += ["### 较早诊断", ""] + table(results[n]["earlier_diagnostics"])
    content += ["## 全部中文因子和完整进入退出规则", "", "以下是收益计算前登记的规则，实际结果已在上文完整列出。相同基础策略不更改原参数，模型系数逐月另附中文文件。", ""]
    for n, slug, title, primary, decision in STUDIES:
        rule_text = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8")
        content += ["\n".join("#" + line if line.startswith("#") else line for line in rule_text.splitlines()), ""]
    content += ["## 完成数量和必要核对", "",
               "第37至40轮共五个新增设置，36个主评价记录，其中10个新账户和26个原样对照；另36个较早诊断记录，其中10个新账户和26个原样对照。新增114次按年龄加权的月度模型拟合，没有新建参考账户，没有获取新行情或补EPS。", "",
               "四项成交量因子测试、四项年龄权重与成熟边界测试、一项实际持仓策略锁定测试，共九项必要测试通过。全部新账户运行时完成日期、财富恒等式、实际持有和终点结算核对。已保存的16项账户差额全部能拆为价格损益、分红和佣金滑点。", "",
               "第38轮最后向控制台显示日期对象时发生JSON打印错误，原命令退出码为一；模型、账户和结果文件在报错之前已经完整写入。本次从四条已保存新账户独立重算夏普、年化、回撤、费用和交易笔数，确认与结果一致，并核对训练样本成熟时间。未重新训练、未重跑账户、未更改冻结规则；结果完成状态以这些直接证据确认。", "",
               "另修正上一份第31至33轮报告的一处文字计数：无模型的更新时点应为140个，原文误写134个。原记录共有846个更新时点，706次成功拟合、140次缺少成熟样本。原模型和账户结果均不变，旧中文报告与其文件记录已同步修正。", "",
               "累计完成40轮、307个不同配置或范围、319个已评价来源版本、836个主评价记录。登记来源版本324，包含5个旧未运行来源绑定。重复对照、较早诊断、参考账户和月度拟合不作为新的独立策略。", "",
               "## 后续研究范围", "",
               "停止本次成交量均线、时间半衰期、两个策略的强趋势切换阈值及锁定规则的细调，保留原第32轮候选作比较。下一项拟检验它开仓时的仓位控制：使用现成二十日波动率决定这一笔初始仓位，并以固定半仓及原满仓作为对照，进入和退出信号保持原样。先固定规则再比较完整账户，不以降低波动本身冒充提高夏普或稳定超额。该下一项尚未登记和运行。", "",
               "EPS及研报、股数、估值、财报和公募补齐保持暂停；其他ETF目标范围问题仍待用户回复，当前不扩展资产、不重复询问。持续研究任务保持活动，目标未完成。不制作GPT数值审阅包，不增加安全审计，不发订单。", "",
               "## 同目录的完整结果和逐月实际模型", ""]
    for path in copies:
        content += [f"- [{path.name}]({path.name})"]
    content += ["", "沿用的原线性退出模型：[每月实际中文规则](../510300简单策略改动_第34至36轮_20260907/已有线性退出模型_每月实际中文规则.md)。", "",
                "上一份结果：[第34至36轮完整说明](../510300简单策略改动_第34至36轮_20260907/全部因子_进出场规则与历史表现.md)。", ""]
    document = OUT / "策略切换为何没有改善_全部因子规则与历史表现.md"
    text = "\n".join(content)
    require("```" not in text and all(p.is_file() for p in copies), "中文规则或普通结果文件不完整")
    document.write_text(text, encoding="utf-8")
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS37_40_COMPLETE_ENTRY_SIZING_RESEARCH_NEXT", latest_completed_round=records[-1], running_studies=[], goal_achieved=False,
                 process_state_note="第37至40轮均完成且进程已终止。第38轮最终控制台显示报错，经保存模型及四新账户直接核对确认数值工作完整；不重跑。",
                 count_warning="完成40轮、307个不同配置或范围、319个已评价来源版本、836个主评价记录；登记来源版本324含5个旧未运行绑定。本次另36个较早记录不计入836。",
                 next_work=["保留第32轮原REARM_RIDGE作为比较基线，主评价0.705及较早0.748尚未确认达标。",
                            "第37轮成交量均线主评价-0.530，对照-0.478；停止相关窗口及过滤搜索。",
                            "第38轮242交易日半衰期加权后主评价0.464、较早0.612；保留已发现的样本年龄事实，但不称其为已证实失败原因，停止半衰期调整。",
                            "第39轮每日状态选择0.498，额外费用8374.66元；第40轮入场后锁定策略0.357，主评价趋势模式27周期亏10932.33元。停止这两个状态的切换和锁定细调。",
                            "下一项拟只在第32轮原候选开仓时按既有20日波动率控制初始仓位，以固定半仓及原满仓对照；入场与退出信号保持同一原模型。尚未登记和运行，先固定简单规则，不扫描风险目标。",
                            "EPS和来源补齐保持暂停，其他ETF范围问题待回复，不重复询问；不做GPT数值包。"],
                 checks="第37至40轮9项必要测试通过；全部新账户完成时钟、财富及结算核对；第38轮4新账户保存指标重新计算一致；16项损益增量已核对。",
                 latest_saved_adaptation_diagnostic=str((DIAG / "result.json").relative_to(ROOT)),
                 prior_training_count_text_correction=count_correction)
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [37, 38, 39, 40]]
    delivery = {"created_at": now(), "type": "STATE_ADAPTATION_ROUNDS37_40_CHINESE_RESULTS", "rounds": [37, 38, 39, 40],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 40 and index["evaluation_accounts_in_this_resumption"] == 836 and index["evaluated_configurations_in_this_resumption"] == 307 and
            index["evaluated_candidate_source_runs_including_corrected_replays"] == 319 and index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 324, "累计研究计数不符")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_SAVED_ACCOUNT_ATTRIBUTION", "rounds": [37, 38, 39, 40],
               "new_configurations": 5, "main_records": 36, "new_main_accounts": 10, "reused_main_accounts": 26,
               "earlier_records": 36, "new_earlier_accounts": 10, "new_model_fits": 114, "no_view_fit_origins": 27,
               "new_reference_accounts": 0, "necessary_tests_passed": 9, "new_gpt_review_archive_created": False, "security_audit_performed": False,
               "goal_achieved": False, "round38_console_error_recovery": completion["status"],
               "files": [{"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "交付文件": len(receipt["files"]), "文档字符": len(text),
                      "已完成轮数": 40, "主评价记录": 836, "第38轮保存数值完成已核对": True, "目标达到": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
