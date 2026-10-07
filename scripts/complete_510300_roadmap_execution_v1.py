"""汇总全部路线图的实际证据、已做工作和仍未满足的依赖。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.roadmap_execution_v1 import CONFIG, PRIOR_EVIDENCE, digest, now, write_json


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out = ROOT / cfg["output"]
    first = ROOT / cfg["first_batch"]
    tasks = json.loads((out / "original_48_tasks.json").read_text(encoding="utf-8"))
    collector = json.loads((out / "collector_current_status.json").read_text(encoding="utf-8"))
    schedule = json.loads((out / "scheduled_task_receipt.json").read_text(encoding="utf-8-sig"))
    trigger = json.loads((out / "scheduled_trigger_verification.json").read_text(encoding="utf-8-sig"))
    if trigger["last_task_result"] != 0:
        raise ValueError("调度任务的本轮实际触发未成功")
    intervals = pd.read_csv(out / "uncertainty_intervals.csv")
    payoffs = pd.read_csv(out / "payoff_distribution.csv")
    registry_count = sum(1 for _ in (out / "study_registry.jsonl").open(encoding="utf-8"))
    routes = [
        {"id": "R02", "name": "融资压力缓和", "hypothesis": "同等价格修复状态下，独立可见的融资行为构成是否增加次一开盘至再下一收盘的净收益信息。",
         "prior": "既有融资收缩/价格交互已含48个主期不重叠观察，均值−0.1087%、胜率45.83%；47/48买入与推算偿还同向。F03残差旧用途也存在，原旧零值问题及纠正版本分别保留。",
         "inputs": "纠正余额/买入2,674日止于2025-12-31；本次取得沪市2026-09-30直接rzche等一日原响应。偿还仍混合多种行为，不是强平识别。",
         "gap": "当前一日新回执不是历史首版；直接偿还汇总并不自动提供余额恒等式之外的新行为构成。深市本次TLS失败，不能只用沪市宣称两市。",
         "decision": "INTAKE_COMPLETE_RETURN_TEST_NOT_ADMITTED", "next": "只有新增、可核验且非旧余额代数重述的分项或独立新时点证据，才冻结一次D对照；不换融资窗口重跑旧用途。"},
        {"id": "R03", "name": "同指数ETF总需求与迁移", "hypothesis": "510300份额减少时，体系流出与产品迁移是否对相同合法跨日合同有不同增量。",
         "prior": "旧官方周度10产品2,870行/287快照的五规则冻结失败；旧三只沪市300产品迁移HIGH/LOW主压力夏普0.3703/0.1394、年化1.1912%/0.3126%，未过原深化门。",
         "inputs": "现有2021-01-08至2026-08-14周度份额；本次920只沪市ETF的2026-09-30份额，含510300/510310/510330。",
         "gap": "920不是920只沪深300产品；名单须按跟踪指数和上市史核对。完整同指数产品范围、深市份额、同日可比NAV和历史发布时间仍缺；TOT_VOL是份额，不是净资金或NAV。",
         "decision": "INTAKE_COMPLETE_FULL_BASKET_SOURCE_NOT_ADMITTED", "next": "补齐事前产品全集、NAV/份额调整及真实接收连续记录，再注册整体需求相对单产品的唯一增量；不直接重复旧三只迁移。"},
        {"id": "R04", "name": "公开事实后的持续调整", "hypothesis": "新增披露的实际回购金额，在扣除首轮价格反应及已知指数暴露后是否还有固定跨日增量。",
         "prior": "盈利广度旧主压力夏普0.0872、年化0.2961%；EPS差异和已知解禁另有冻结失败。回购事实账本57文档并非指数全集；后续911候选链/108公司有0交易字段准入。",
         "inputs": "已有原始公告、方案身份、累计额差分和旧点时成员资料；不得把累计金额多次相加或把无新增披露当零回购。",
         "gap": "完整事件全集、同根方案/金额单位、首次可知时间及对当时指数的可比暴露仍未完成联结；市场值分母缺失不能取未来值。",
         "decision": "INTAKE_COMPLETE_EVENT_UNIVERSE_AND_EXPOSURE_NOT_ADMITTED", "next": "先补一个可核对的完整事件队列和当时暴露，固定披露后第一完整反应日，再从其后合法开盘比较；不靠研报叙事生成历史标签。"},
        {"id": "R05", "name": "一个跨市场通道：真实IF持有成本", "hypothesis": "已知融资成本与已公告现金分红调整后的真实合约偏离，是否有额外跨日信息。",
         "prior": "旧八条真实期限结构规则冻结拒绝；总OI增量失败、旧全球隔夜主压力夏普0.1308、年化0.8417%。本批不同时搜索汇率/股债/海外风险。",
         "inputs": "旧真实IF合约/现货覆盖存在，但严格成本调整需要同合约日历、当时可知利率及预期分红映射。",
         "gap": "未建立区别于旧基差水平的完整点时持有成本输入；事后实现分红和总OI不能分别充当事前预期与净多头。",
         "decision": "INTAKE_COMPLETE_CARRY_SOURCE_NOT_ADMITTED", "next": "只有新增的可核验成本成分成立才登记单一残差，不换旧期限尾部分位救回。"},
        {"id": "R06", "name": "持有与退出价值", "hypothesis": "同一已通过初步经济门的固定入场，另一退出是否改善净收益而非仅推迟兑现亏损。",
         "prior": "D-native未建立优势；旧96因子49规则/245反事实表明降低费用或提高仓位没有完成目标。已有退出家族及原失败保留。",
         "inputs": "可读取D-native全部周期和旧入场，但当前没有本轮新准入的机会模块。",
         "gap": "缺少符合本路线图前提的固定入场；不能因新基准亏损就开始退出搜索。",
         "decision": "NOT_RUN_NO_ADMITTED_ENTRY_MODULE", "next": "先满足R02—R05之一的经济门，才冻结同入场持有政策对照。当前只完成原周期净分布。"},
        {"id": "R07", "name": "宏观来源及政策解释储备", "hypothesis": "保留风险状态及竞争解释，避免把风险持续性直接翻译成未来收益方向。",
         "prior": "旧M1/M2月度增量终态为旧定义无可靠增量、新定义不足；宏观波动两轮观察分别保留风险持续性与原因未定边界。",
         "inputs": "现有初值、分项、政策文本及两轮完成回执；不干预并行宏观分支。",
         "gap": "宏观原因不自动形成当前分支买卖总分，也不覆盖旧五日用途失败。",
         "decision": "CONTEXT_LEDGER_COMPLETE_NO_NEW_SCORE", "next": "只在出现新初值、明确原因链或独立证据时追加说明。"},
        {"id": "R08", "name": "真正盘口储备", "hypothesis": "只有新增时钟、估值或成交字段解决原用途缺口，才恢复严格盘口实验。",
         "prior": "已有181日/543文件，原M1/M2严格来源门未过；本批没有新增这类字段。",
         "inputs": "复用原来源摘要与第一批状态纠正，不再读整套三流。",
         "gap": "同步时钟、参考估值和可成交证据未改善。",
         "decision": "RESERVE_NO_NEW_FIELD_EVIDENCE", "next": "等待具体新字段或原始生成时钟证据；不重复同一来源导出。"},
    ]
    write_json(out / "mechanism_admission.json", {"recorded_at": now(), "routes": routes, "new_return_experiments_admitted": 0,
        "all_mechanisms_permanently_disproved": False, "data_observer_may_continue_independently": True})
    card_dir = out / "mechanism_cards"
    card_dir.mkdir(exist_ok=True)
    for row in routes:
        evidence = [str(Path("reports/research") / p) for p in PRIOR_EVIDENCE[row["id"]] if (ROOT / "reports/research" / p).exists()]
        text = [f"# {row['id']}｜{row['name']}", "", "假设：" + row["hypothesis"], "", "旧用途及结果：" + row["prior"],
                "", "现有字段：" + row["inputs"], "", "未解决：" + row["gap"], "", "本次处置：`" + row["decision"] + "`。", "", "下一最小步骤与停止线：" + row["next"],
                "", "若准入，主对照为同期限D日线字段；先固定信息可得原点，再比较共同日期预测及完整日历净账户。不得把D-native原二日模型直接移植为不同期限的已验证对照。模型和主目标见experiment_specs.json；当前没有新拟合、标签或账户。", "", "直接证据：", ""]
        text += ["- " + p for p in evidence]
        (card_dir / (row["id"] + "_" + row["name"].replace("：", "_") + ".md")).write_text("\n".join(text) + "\n", encoding="utf-8")
    specs = []
    for row in routes[:4]:
        specs.append({"route": row["id"], "primary_question": row["hypothesis"], "status": row["decision"],
            "primary_target": "信息已可得及规定首轮反应结束后的下一开盘至再下一收盘净持有收益",
            "price_control": "按同原点、同期限重建D八字段对照，D-common仅保留原研究角色",
            "comparison": "同输入完整日期的唯一新增观测增量；同时报告全部日历与未知日期",
            "first_model": "线性岭alpha=1，前序成熟样本标准化；不搜索模型/符号/期限/退出/仓位",
            "candidate_feature_frozen": False, "reason": "输入和旧用途门未通过，不能把未定义的新字段假装已经注册",
            "confirmation": "需要独立新日期及成本/执行证据；历史已暴露仅为探索",
            "stop": row["next"], "new_fits": 0})
    write_json(out / "experiment_specs.json", specs)
    write_json(out / "model_registry.json", {"one_model_family_per_admitted_increment": "RIDGE_LINEAR", "alpha": 1,
        "new_models_fitted": 0, "interaction_search": False, "parameter_search": False,
        "minimum_training_rule": "未来具体源准入后，按事件/日期有效样本预先冻结；不将分钟行或模型更新次数当样本",
        "native_36_updates_are_one_design": True})
    write_json(out / "baseline_registry.json", {"D_COMMON": "原869输入/364预测/471账户日；原16结果不变",
        "D_NATIVE": "原字段、自有资格；新4账户已完成，不能晋升赢家",
        "BUY_HOLD": "后续准入机制同日起止、相同费用/分红/期末合同；本轮未新跑",
        "SIMPLE_RISK_CONTROL": "须预先固定唯一风险参考；本轮未新跑，也不选择旧最优风控",
        "comparison_stage": "D配对统计完成；新机制四基准完整账户NOT_RUN_SOURCE_GATE",
        "metric_contract": "metric_contract.json"})
    write_json(out / "causal_clock_tests.json", {"new_tests_passed": 9, "elapsed_seconds": 1.89,
        "prefix_test": "追加未来记录及晚到回执不进入过去as-of选择", "unknown_receipt": "不从经济日期推测收到日期",
        "existing_native_tests": "第一批7项新测试及5项原测试保持",
        "new_real_event_dedup_and_nonevent_negative_control": "NOT_RUN_NO_ADMITTED_EVENT_UNIVERSE",
        "full_future_order_invariance": "NO_NEW_PORTFOLIO_ORDERS_TO_VALIDATE",
        "pre_freeze_adapter_fix": "工作簿空行的读取守卫；无标签/拟合/统计结果时已修正",
        "calendar_fix": "改用已核对的2026全年官方日历；原共享2010—2026文件实际仅到8月14日，未改原文件"})
    write_json(out / "execution_contract.json", {"version": "REUSE_NATIVE_ACCOUNT_CONTRACT_NO_NEW_POLICY",
        "reference": "config/510300_daily_native_baseline_v1.json", "decision": "盘后决定次日请求，不能见最终开盘价后再宣称按开盘成交",
        "holding": "T+1开盘至T+2收盘；新买份额不当日卖出", "lot": 100, "max_long_exposure": 1.0,
        "cash_short_borrow": False, "fill": "分钟价格及量额容量仅条件代理，真实排队未知",
        "dividend_cash_inventory": "复用冻结现金/库存/应收分红、部分/未成交和终点计价",
        "additional_delay_policy": "NOT_RUN_NO_ADMITTED_NEW_POLICY", "broker_connection": False})
    write_json(out / "risk_policy.json", {"research_only": True, "hard_exposure_ceiling": 1.0,
        "actual_capital": None, "maximum_actual_drawdown": None, "maximum_actual_event_loss": None,
        "status": "WAITING_USER_REAL_RISK_PARAMETERS", "position_selection_from_backtest": False,
        "pilot_allowed": False, "simulation_is_not_current_holding": True})
    write_json(out / "single_account_portfolio.json", {"status": "NOT_RUN_NO_ADMITTED_INCREMENTAL_MODULE", "admitted_modules": [],
        "orders": [], "actual_position": "UNKNOWN", "max_total_long_exposure": 1.0,
        "same_etf_inventory_count": 1, "double_count_cash_dividend_fee": False,
        "new_portfolio_stress_accounts": 0, "basis": "P04前提未满足，不把多个失败满仓回测相加"})
    forward = {"status": "NOT_FROZEN_STRATEGY_GATE_NOT_PASSED", "strategy_start": None,
        "independent_strategy_observations": 0, "simulated_strategy_orders": 0, "actual_fills": 0,
        "public_raw_observer_start": collector["completed_at"], "raw_observer_is_strategy_forward_test": False,
        "F03_execution_calibration": "NOT_RUN_NO_PLAN_OR_FILL_OBSERVATIONS",
        "F04_pilot": "NOT_AUTHORIZED_NO_CONCRETE_CAPITAL_RISK_OR_BROKER_CONTRACT",
        "F05_scaling": "NOT_RUN_NO_PILOT_EVIDENCE", "recovery": "先有新准入模块及统一账户/压力/集成通过，再冻结具体前瞻策略；未来样本不可补做"}
    write_json(out / "forward_manifest.json", forward)
    write_json(out / "monthly_scorecard.json", {"month": "2026-10", "population": "NOT_STARTED_STRATEGY_FORWARD",
        "forward_days": 0, "net_sharpe": None, "cagr": None, "excess_return": None, "max_drawdown": None,
        "raw_observation_days": 1, "public_sources_received": collector["sources_with_rows"],
        "missing_financial_metrics_mean_unknown_not_zero": True, "next_review": "有新策略观测后按完整自然月登记，固定主口径；不择最好最近窗口"})
    cycles = pd.read_parquet(first / "native_cycles.parquet")
    cycles["outcome"] = cycles.net_pnl.map(lambda value: "PROFIT" if value > 0 else "LOSS" if value < 0 else "FLAT")
    cycles["market_mechanism_cause"] = "UNKNOWN_NO_CAUSAL_IDENTIFICATION"
    cycles["execution_evidence"] = "CONDITIONAL_PROXY_NOT_ACTUAL_FILL"
    cycles[["capital", "cost", "entry_date", "exit_date", "outcome", "net_pnl", "commission", "slippage", "market_mechanism_cause", "execution_evidence"]].to_csv(out / "failure_and_success_ledger.csv", index=False, encoding="utf-8-sig")
    write_json(out / "research_budget_review.json", {"reviewed_at": now(), "new_source_requests": 3,
        "raw_response_bytes": collector["raw_bytes"], "fee_cny": 0, "new_fits": 0, "new_accounts": 0,
        "saved_comparisons": 5, "bootstrap_indices_preserved": True, "native_and_old_results_unchanged": True,
        "no_new_evidence_no_repeated_gate_export": True, "source_observer_can_run_without_strategy_admission": True,
        "source_storage_limit_bytes": cfg["raw_vintage_collector"]["max_total_storage_bytes"]})
    write_json(out / "promotion_decision.json", {"decision": "NO_PROMOTION_NO_VALIDATED_STRATEGY",
        "development_progress": True, "historical_joint_target_met_by_native": False,
        "independent_validation": "NOT_ESTABLISHED", "portable_live_performance": "NOT_ESTABLISHED",
        "do_not_change_cost_annualization_sample_or_assets_to_pass": True})

    # 从真实接收响应提取有限事实；绝不倒推历史首次公布或制造历史序列。
    received_rows = []
    day_dir = out / "raw_vintages" / collector["observation_day"]
    for source in collector["sources"]:
        raw = day_dir / (source["source_id"] + ".observed_rows.json")
        if not raw.exists():
            continue
        rows = json.loads(raw.read_text(encoding="utf-8"))
        if source["source_id"] == "SSE_MARGIN":
            for row in rows:
                if row.get("opDate") != collector["requested_stat_date"].replace("-", ""):
                    raise ValueError("沪市融资响应统计日与请求不符")
                for field in ["rzye", "rzmre", "rzche"]:
                    received_rows.append({"source": source["source_id"], "stat_date": collector["requested_stat_date"],
                        "symbol": "SSE_MARKET", "field": field, "raw_value": row[field], "unit": "CNY",
                        "received_at": source["received_at"], "first_publication_proven": False, "sha256": source["sha256"]})
        if source["source_id"] == "SSE_ETF_SHARES":
            for row in rows:
                if row.get("SEC_CODE") in ["510300", "510310", "510330"]:
                    if row["STAT_DATE"] != collector["requested_stat_date"]:
                        raise ValueError("ETF统计日与请求不符")
                    received_rows.append({"source": source["source_id"], "stat_date": row["STAT_DATE"], "symbol": row["SEC_CODE"],
                        "field": "TOT_VOL", "raw_value": row["TOT_VOL"], "unit": "TEN_THOUSAND_SHARES_SOURCE_CONTRACT",
                        "received_at": source["received_at"], "first_publication_proven": False, "sha256": source["sha256"]})
    pd.DataFrame(received_rows).to_csv(out / "received_public_facts.csv", index=False, encoding="utf-8-sig")
    permissions = """# 来源使用范围核对

本次只为个人非商业研究读取公开信息；免费可读、免费完整历史、批量接口许可与再分发权限分别记录。

|来源|本次核对及可用范围|尚未建立|
|---|---|---|
|上海证券交易所|[法律声明](https://www.sse.com.cn/home/legal/)允许遵守声明的非商业浏览、下载；已取得两份公开JSON响应，接口和原始字段留存|不由此承诺接口长期稳定、历史首版或无限制商业再分发|
|深圳证券交易所|[法律声明](https://www.szse.cn/application/laws/index.html)同样区分非商业浏览下载；本次融资请求因TLS失败保留回执|没有本次数据，不改成HTTP或关闭验证假装成功|
|巨潮公告/深证信API|公告正文是事实入口，[数据服务平台](https://webapi.cninfo.com.cn/)是另一个服务入口；现有原件按原来源保存|未核准整库自动批量接口/再分发，不把网页阅读当API套餐许可|
|已有第三方分钟、日线和研究缓存|本批复用用户已有文件，无新增费用；数据资产目录保留原元数据和限制|未取得来源合同不宣称可任意再分发，也不因文件名含商用数据商就要求购买|
|Hugging Face盘口储备|使用已存在181日来源记录，不重复申请或下载|未把研究许可扩大为严格队列/公允价值或公开再分发资格|

只取得字段事实，不向网站发送消息或申请账户。[融资字段官方说明](https://www.sse.com.cn/market/othersdata/margin/detail/)表明偿还额混合直接还款、卖券还款、强平和权益调整，不能命名为纯强平。当前公开响应的接收时刻不能补成过去首次可得时刻。

交易日历使用已保存2026全年官方版本，并与[2026休市公告](https://www.sse.com.cn/disclosure/dealinstruc/closed/c/c_20251222_10802510.shtml)核对。本地较短历史交易日文件不用于当前日常观察。
"""
    (out / "source_permissions.md").write_text(permissions, encoding="utf-8")
    macro_paths = ["reports/research/510300_m1_m2_monthly_increment_v1/status.json",
                   "reports/research/510300_macro_volatility_observation_v2_run1/completion_receipt.json",
                   "reports/research/510300_macro_volatility_mechanisms_v2_run2/completion_receipt.json"]
    write_json(out / "macro_context_ledger.json", {"status": "CONTEXT_ONLY_NO_TRADING_GATE",
        "evidence": [{"path": path, "sha256": digest(ROOT / path)} for path in macro_paths],
        "risk_direction_causality_separate": True, "unknown_and_mixed_causes_preserved": True,
        "old_fixed_m1m2_increment_rejection_kept": True, "new_score_or_fits": 0})

    completed = {
        "G02": "metric_contract.json", "G03": "study_registry.jsonl", "A01": "第一批封卷", "A02": "第一批108日归因",
        "A03": "第一批四层资格", "A04": "第一批纠错与设计分类", "A05": "第一批C−D差额", "A06": "第一批盘口状态纠正",
        "D02": "第一批字段分离", "D05": "第一批制度矩阵", "D06": "第一批D-native完整运行", "D07": "第一批状态与12项测试",
        "D08": "raw_vintages/；scheduled_task_receipt.json", "D09": "source_permissions.md",
        "R07": "macro_context_ledger.json", "T04": "model_registry.json", "T06": "bootstrap_indices.npz；uncertainty_intervals.csv",
        "P01": "payoff_distribution.csv", "P02": "execution_contract.json", "M03": "research_budget_review.json", "M04": "promotion_decision.json"}
    limited = {
        "D01": ("data_catalog_original_24.csv；data_catalog_additional.csv", "本轮29项直接资产与分支来源，非整个仓库所有原始文件逐项验证"),
        "D03": ("第一批amount_diagnostic.json", "有限单位/累计/浮点原因核对完成，原始精度/对齐根因未知"),
        "D04": ("causal_clock_tests.json；第一批asof面板", "D-native及实际接收时间选择已验证；新事件源完整管线尚未准入"),
        "R01": ("mechanism_registry.csv；selection_report.json", "96项映射与16家族完成；未找到可验证的单一99篇来源索引，不宣称逐篇复核"),
        "T01": ("experiment_specs.json", "主问题/期限/停止线已写；关键新字段未通过准入，不虚称全部实验规范已可运行"),
        "T02": ("baseline_registry.json", "D-common/native已公平对齐；新机制的买入持有/简单风险完整账户待源门"),
        "T03": ("causal_clock_tests.json", "9项必要测试通过；真实事件去重/非事件对照未在无合格全集时伪造"),
        "T05": ("selection_report.json；trial_log.jsonl", "787条直接摘要与本轮全部结果留存；完整跨历史选择次数未识别"),
        "T07": ("all_event_cluster_sensitivities.csv；new_and_removed_entry_dates.csv", "完整贡献与交易集合完成；延迟执行账户未运行，不把贡献置零当真实账户"),
        "P06": ("causal_clock_tests.json", "基准与时间/错误路径测试通过；无新组合可做完整集成"),
        "M02": ("failure_and_success_ledger.csv", "294个跨情景原周期盈亏对称记录；机制原因未知，尚无前瞻实际机会")}
    pending = {
        "G01": ("WAIT_USER_PARAMETERS", "objective_contract.json", "等待真实本金、用款期限、风险和券商费用；其他研究继续"),
        "R02": ("RETURN_TEST_NOT_ADMITTED", "mechanism_cards/R02_融资压力缓和.md", routes[0]["gap"]),
        "R03": ("RETURN_TEST_NOT_ADMITTED", "mechanism_cards/R03_同指数ETF总需求与迁移.md", routes[1]["gap"]),
        "R04": ("RETURN_TEST_NOT_ADMITTED", "mechanism_cards/R04_公开事实后的持续调整.md", routes[2]["gap"]),
        "R05": ("RETURN_TEST_NOT_ADMITTED", "mechanism_cards/R05_一个跨市场通道_真实IF持有成本.md", routes[3]["gap"]),
        "R06": ("NOT_RUN_DEPENDENCY", "mechanism_admission.json", "没有已过初步经济门的固定入场，不启动退出搜索"),
        "R08": ("RESERVE_UNCHANGED", "mechanism_admission.json", "原严格字段缺口未改善，不重复处理181日"),
        "P03": ("WAIT_USER_PARAMETERS", "risk_policy.json", "真实风险约束未确认；未按回测倒推风险承受"),
        "P04": ("NOT_RUN_DEPENDENCY", "single_account_portfolio.json", "0新准入模块，不强行组合"),
        "P05": ("NOT_RUN_DEPENDENCY", "single_account_portfolio.json", "基准4情景已完成；不存在可重跑的正式新组合"),
        "F01": ("NOT_RUN_DEPENDENCY", "forward_manifest.json", "没有合格策略；公开源观察冻结不能替代前瞻策略冻结"),
        "F02": ("WAIT_FUTURE_QUALIFIED_STRATEGY", "forward_manifest.json", "尚无合格策略信号或模拟订单；原始数据观察单列"),
        "F03": ("WAIT_FUTURE_EXECUTION_EVIDENCE", "forward_manifest.json", "尚无计划与真实可见执行对照，0真实成交"),
        "F04": ("SEPARATE_ACTUAL_TRADING_AUTHORITY_REQUIRED", "forward_manifest.json", "路线图指定另行授权；未指定真实资本/券商/风险，不下单"),
        "F05": ("NOT_RUN_DEPENDENCY", "forward_manifest.json", "未实测，不扩容"),
        "M01": ("WAIT_FUTURE_STRATEGY_SAMPLE", "monthly_scorecard.json", "0前瞻策略日；金融统计为空，不写零夏普")}
    assert set(completed) | set(limited) | set(pending) == {row["任务ID"] for row in tasks}
    task_rows = []
    for row in tasks:
        key = row["任务ID"]
        if key in completed:
            status, evidence, boundary = "COMPLETED_IN_STATED_SCOPE", completed[key], "研究登记/工程或所列统计的完成，不代表金融目标通过"
        elif key in limited:
            status = "PARTIAL_WITH_EXPLICIT_LIMIT"
            evidence, boundary = limited[key]
        else:
            status, evidence, boundary = pending[key]
        task_rows.append({**row, "本次状态": status, "证据入口": evidence, "未完成部分或恢复条件": boundary, "原附件状态保持": True})
    pd.DataFrame(task_rows).to_csv(out / "48项执行台账.csv", index=False, encoding="utf-8-sig")
    log = [{"at": now(), "type": "FIRST_BATCH_ALREADY_COMPLETED", "fits": 36, "accounts": 4, "path": cfg["first_batch"]},
           {"at": now(), "type": "POST_RESULT_SAVED_DIAGNOSTIC", "comparisons": 5, "fits": 0, "accounts": 0, "indices": "bootstrap_indices.npz"},
           {"at": now(), "type": "MECHANISM_ADMISSION", "routes": 7, "new_return_experiments_admitted": 0},
           {"at": now(), "type": "PUBLIC_SOURCE_CAPTURE", "requests": 3, "sources_with_rows": collector["sources_with_rows"], "new_strategy_observations": 0}]
    (out / "trial_log.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in log), encoding="utf-8")
    for item in json.loads((out / "freeze.json").read_text(encoding="utf-8"))["files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结文件变化：" + item["path"])
    result = {"at": now(), "status": "ALL_48_ROUTED_CURRENT_EXECUTABLE_WORK_DONE_CONDITIONAL_STAGES_NOT_COMPLETED",
        "user_authority": "其余的都进行", "task_count": 48, "completed_in_stated_scope": len(completed), "partial_with_limits": len(limited),
        "gated_or_waiting": len(pending), "factor_mapping_rows": 96, "source_study_summaries": registry_count,
        "new_fits_after_first_batch": 0, "new_accounts_after_first_batch": 0, "new_saved_comparisons": 5,
        "new_tests_passed": 9, "new_public_requests": 3, "public_sources_with_rows": collector["sources_with_rows"],
        "public_raw_bytes": collector["raw_bytes"], "source_observer_installed": True, "task_name": schedule["task_name"],
        "scheduled_holiday_branch_actually_triggered": True, "scheduled_last_task_result": trigger["last_task_result"],
        "new_strategy_forward_observations": 0, "actual_orders": 0, "financial_goal_achieved": False,
        "all_48_scientific_tasks_complete": False, "next_independent_strategy_experiment": "NOT_ADMITTED_SOURCE_OR_NOVELTY_GATES"}
    write_json(out / "result.json", result)
    base = intervals.loc[intervals.capital.eq(200000) & intervals.cost.eq("BASE")].iloc[0]
    mse = intervals.loc[intervals.comparison.eq("MSE_COMMON_MINUS_NATIVE")].iloc[0]
    payoff = payoffs.loc[payoffs.capital.eq(200000) & payoffs.cost.eq("BASE")].iloc[0]
    report = f"""# 全部路线图的执行状态与当前结论

已按用户“其余的都进行”扩大范围，48项全部进入有证据的执行台账。当前{len(completed)}项在所列范围内完成、{len(limited)}项部分完成且保留限制、{len(pending)}项因来源/机制/真实风险/未来样本或实际交易授权尚未完成。不是48项科学验证全部完成；没有找到已验证的高夏普策略。

## 本轮实际新增

- 96因子逐项映射到16个机制家族，保留定义、反例、旧重叠、额外观测及原始来源。787条当前直接研究摘要建立版本/终态快照；摘要数不是独立试验数。99篇资料的单一核准索引未找到，不声称逐篇读完。
- 原24项数据用途保留，新增5项跨融资/ETF/盈利分支的现有直接资产目录。字段/时钟/范围仍逐用途判断。
- 对原471账户日、364共同预测日各做5,000次20日区块配对抽样，实际索引已保存；只读取保存结果，新增拟合和账户均0。
- 所有月份、新增/删除入场日期、重叠持有事件簇、事件贡献置零的有限敏感性、全部周期净盈亏分布和盈亏对称账本已保存。延迟执行账户未运行，未把分解假装成可交易新政策。
- 9项时间隔离、区块、净期望、日历和失败回执测试通过。第一批12项和原账户结果保持。

## 金融解释

20万元BASE的native−common完整日历年化算术收益差为{base.point:.3%}，单项95%区间[{base.nominal_95_lower:.3%}, {base.nominal_95_upper:.3%}]；本次5比较校正区间[{base.local_five_comparison_lower:.3%}, {base.local_five_comparison_upper:.3%}]。它不是CAGR差的区间，且局部校正没有消除全研究库的选择偏差。

MSE改善点值{mse.point:.10f}，单项95%区间[{mse.nominal_95_lower:.10f}, {mse.nominal_95_upper:.10f}]跨0。较小误差未建立可靠增量；原D-native20万元BASE年化1.021%、夏普0.142、回撤16.978%，压力年化−0.603%继续保留。

同一BASE合同93周期，胜率{payoff.profit_probability:.2%}、条件净盈利{payoff.mean_gain_net:.3%}、条件净亏损{payoff.mean_loss_net:.3%}，pG−qL={payoff.pG_minus_qL_net:.4%}。最差周期{payoff.minimum_cycle_net:.3%}，最差5%周期均值{payoff.worst_5pct_cycle_mean_net:.3%}。周期等权净期望已经扣费，不再减费，也不能代替全账户复合收益。压力周期均值略正但账户亏损的事实未改。

## 三条主线及储备的处置

融资、ETF迁移、公开信息调整均已有直接旧用途。融资的48个旧主期观察未示优势，F03残差还存在首版及旧错误版本的边界；沪市一日直接偿还字段并不自动识别被迫卖压。ETF旧三产品迁移未过深化门，当前沪市全ETF列表不是完整同指数NAV/份额体系。公司回购911候选链/108公司仍0交易字段准入，不能以数目替代完整事件/指数暴露。

逐条一页卡见mechanism_cards/，包括最小对照、仍缺字段和停止条件。R05只核真实IF持有成本一个通道，旧基差/总OI/全球隔夜失败不重开。R06没有合格入场，不启动退出优化。R07保留宏观解释账本，R08没有新字段，不重复导出181日。当前0个新收益实验准入，不说明所有可能机制永久无效。

## 免费版本留存已经实际运行

2026-10-02实际请求三个官方公开源：沪市融资成功1行，沪市ETF份额成功920行，均为2026-09-30统计日；深市融资TLS失败，无数据。原响应合计{collector['raw_bytes']:,}字节，收到时间逐请求留存。本次沪市融资还有直接rzche字段；ETF保留份额原单位，不命名为净资金。数据是今天回取，不是证明9月30日当天已知。

已安装Windows任务`{schedule['task_name']}`，周一至周五09:15检查；非交易日只写跳过回执，使用pythonw后台运行。电脑可用且该用户登录时执行，最多3请求/次、0费用、250MiB总上限，2026日历过期即停止取数。当前已实测国庆非交易日分支0网络请求；调度器下次触发10月5日会跳过，按已核对日历首个交易日为10月8日。未来定时成功仍待实际运行回执，不能由安装成功预先宣称。

它是原始公开数据观察，不是策略前瞻试验。F01尚无可晋升的策略，F02/F03的策略信号、模拟订单及实际成交均0。保存原始版本不等于独立金融验证。

## 剩余条件与可继续动作

G01/P03的真实本金、用款期限、最大回撤、单事件亏损和券商收费已向用户询问，暂保持未知；按2万元/20万元研究情景可独立推进。P04没有合格模块，不强行组合；P05没有正式新组合可跑；P06保留已完成的基准和时间/错误路径测试，不声称新组合集成已通过。

F04按路线图另需具体实际交易授权，本次没有连接券商或下单。F05不能在没有实际证据时扩容。M01创建了空样本月度记录，金融统计为空；M02对全部历史盈亏对称记账，市场原因UNKNOWN；M03不重复无新证据的工程，M04明确不晋升。

接续应先检查新原始版本是否提供了实质新字段、完整产品/事件范围和可用时钟，再依一页卡准入。没有新证据时不重复这套核对或继续调参；公开源观察按已安装任务保留版本即可。未来有效样本只能随时间积累，不能在本轮补做。

完整入口：48项执行台账.csv、mechanism_admission.json、uncertainty_intervals.csv、received_public_facts.csv、scheduled_task_receipt.json、scheduled_trigger_verification.json、forward_manifest.json、result.json。定时任务已经实际触发，休市分支返回LastTaskResult=0；未来交易日联网结果另候真实回执。第一批报告与Pro包仍为独立历史交付，不覆盖。
"""
    (out / "全部路线图执行报告.md").write_text(report, encoding="utf-8")
    files = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "file_index.json"]
    write_json(out / "file_index.json", {"indexed_at": now(), "files": files, "total_bytes": sum(row["bytes"] for row in files),
        "mutable_observer_receipts": "未来新原始版本只追加；collector_current_status和scheduled_observer_last_receipt随实际运行更新，索引是本次快照"})
    print(json.dumps({**result, "report_files": len(files), "report_bytes": sum(row["bytes"] for row in files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
