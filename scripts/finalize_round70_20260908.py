"""交付第70轮完整中文规则和账户结果，更新继续研究的位置。"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from research.volatility_term_output_fix_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300海外波动期限_第70轮_20260908"
DOCUMENT = OUT / "海外波动期限_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_VOLATILITY_TERM_MEDIAN_CONTINUATION_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require([r["round"] for r in index["completed_rounds"]] == list(range(1, 70)), "索引不是截至69轮，不能重复收尾")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "波动期限原始或修正来源改变")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((RESEARCH / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"] and receipt["recomputed_metrics"] == 20, "结果尚不能按失败完成交付")
    cycles = pd.read_csv(RESEARCH / "saved_cycle_profit_groups.csv")
    status = "COMPLETED_REJECTED_VOLATILITY_TERM_HIGH_EXPOSURE_AND_COST_DRAG"
    decision = "主评价净夏普0.043504、压力负0.073106，较早0.439190、压力0.299596；均未达1.2。主评价约78%仓位仍有48.13%回撤，2022和2023连续亏损。结束同一期限关系，不翻向、平滑或改变阈值挽救。"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300海外波动期限：第70轮", "", "## 结果和处理", "",
        "**这一简单规则失败，夏普1.2目标仍未实现。** 主评价基础净夏普0.044、压力负0.073，基础复合年化负0.61%、最大回撤48.13%、累计负3.96%。较早基础净夏普0.439、压力0.300，基础复合年化6.72%、最大回撤30.31%、累计38.80%。", "",
        "本轮只用现成两个免费官方文件，建立一个持有或现金规则，四个完整账户已经完成。规则不需要EPS，也没有新的行情下载或模型拟合。后续不围绕这一个期限比值再反向、改阈值或加平滑。", "",
        "## 老板可以直接理解的规则", "",
        "每天上午九点看美国已经结束的最近交易日：如果市场对未来九天的预期波动低于未来30天，就准备持有510300；如果九天达到或超过30天，就准备退出。满足持有条件可以重新进入。每次用前一中国收盘的资金和价格先算份额，再在当前开盘按真实约束成交；缺失或过期资料不作新买卖，保留实际份额。", "",
        "经济直觉是避开海外短期风险突然高于月度风险的阶段；实际结果表明，它没有有效避开这两段历史中的主要风险。资料来自美国股票期权对应的预期波动，并不是中国股票的预测收益。", "",
        "## 主评价：2020年1月2日至2026年8月14日开盘", ""]
    lines += table([metric(result, model, cost=cost) for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"] for cost in cfg["costs"]])
    lines += ["## 较早诊断：2015年1月5日至2019年12月31日开盘", ""]
    lines += table([metric(result, model, period="earlier_diagnostic", cost=cost) for model in [PRIMARY, "REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"] for cost in cfg["costs"]])
    lines += ["两段各自以20万元开始。夏普使用全部交易日的净收益，包括空仓日，不能只算赚钱的持仓。现金和无风险收益均沿用零假设，年化使用242日。主评价基础夏普略正而复合年化为负并不矛盾：其算术日均收益年化约0.71%，波动导致复合增长低于算术平均。", "",
        "## 盈利、分红和费用是怎样形成的", "",
        "|历史|费用|完整持仓周期|盈利／亏损周期|价格损益（元）|分红权益（元）|佣金（元）|滑点（元）|最终净损益（元）|平均持仓收盘数|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cycles.itertuples():
        lines.append(f"|{'主评价' if r.period=='evaluation' else '较早诊断'}|{'基础' if r.cost=='BASE' else '压力'}|{r.cycles}|{r.positive_cycles}／{r.negative_cycles}|{r.gross_price_profit:,.2f}|{r.dividend_recognized:,.2f}|{r.commission:,.2f}|{r.slippage:,.2f}|{r.net_profit:,.2f}|{r.mean_held_closes:.2f}|")
    lines += ["", "主评价基础价格损益负2,098.20元，加分红21,101.10元，扣费前合计19,002.90元；佣金和滑点合计26,925.61元，最终亏损7,922.71元。分红对结果有实质影响，但不足以覆盖完整交易成本。较早基础扣费前113,505.90元，费用35,914.91元，剩余77,590.99元。", "",
        "主评价90个完整周期、180笔成交，较早89个完整周期、178笔成交；均无持仓中追加买入和整笔未成交请求。主评价1256个持仓收盘、较早917个，平均持仓接近十四日和十日。这些是真实账户记录，不是把比值变动次数当成交次数。", "",
        "## 基础费用逐年表现", "", "|历史|年份|该年累计净收益|该年净夏普|该年内最大回撤|成交笔数|", "|---|---|---:|---:|---:|---:|"]
    years = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for r in years[years.model.eq(PRIMARY) & years.cost.eq("BASE")].itertuples():
        lines.append(f"|{'主评价' if r.period=='evaluation' else '较早诊断'}|{r.year}|{r.cumulative_return:.2%}|{r.net_sharpe:.3f}|{-r.max_drawdown:.2%}|{r.trade_count}|")
    lines += ["", "2026年只到固定终点，并非全年。上述每年回撤从该年自身起点计算，不能替代完整账户回撤。2022年亏22.64%、2023年亏12.74%，这两年合在一起的基础年化负17.84%、净夏普负1.488，说明短期海外风险恢复正常并不足以确认中国股票已经脱离持续下跌。2017年和2025年单年夏普超过1.2，不代表完整研究期间达标。", "",
        "## 数据缺失和时间核对", "",
        "两个官方文件按源日期外连接后，在固定支持区间共有3958行，其中32行缺一项。不能删掉这些行后退回此前完整数据。中国主评价1604个决定原点中，1573个完整、31个资料不足；其中30天原本有持仓，按规则继续保留，1天空仓。两档费用的这些状态一致；较早1219个原点都完整，没有过期原点。", "",
        "主评价有1227个有效满仓目标原点，减去终点开盘强制清算覆盖的1个，再加30个资料不足却继续持仓的日子，等于1256个持仓收盘。较早918个满仓目标原点减终点1个，等于917个持仓收盘。资料不足不等于现金。", "",
        "用独立的源日期字典和时区计算重放两段合计5308行状态，目标最大差为零；另核对5646个实际份额请求、20项账户指标、12组账户经济差额、两费用合计358个含分红完整周期。未重训模型，也没有在核对时再模拟账户。", "",
        "第一次完整运行因毫秒与纳秒日期列的严格类型比较而停在保存前；1604个执行日期逐行完全一致、最大时间差为零。原版本保留，仅在另一个来源版本中改为比较实际时间值，交易规则和数据均未改变。原始运行曾在内存构造1个账户，未保存账本、未形成或查看收益指标；修正版本实际完成4个账户。登记统计为1个策略设置、2个执行来源版本，不隐去实现修正。关键测试最终10项通过；首轮测试中另有一项测试数据整数列不能注入无穷值，已在冻结前修正测试列类型。", "",
        "两个文件是当前取得的历史汇总，不能证明当年每一天的首次文件送达时刻。以下下午五点和上午九点为明确的历史可用性约定，本轮没有新建真实交易信号。", "",
        "## 全部因子与完整冻结规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()
    lines += [line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in protocol[1:]]
    lines += ["", "## 下一步", "",
        "回到现成的持仓周期学习资料，检验一个新的学习目标：预测继续持有收益的条件中位数，再按原来的连续两次负值条件退出。中位数表示分布的中间位置，可能减轻少数特别大行情对模型的牵引；它也可能过早丢掉大盈利，因此不能事先宣称比平均收益学习更好。其进场、原价格退出和费用继续固定，先登记一个模型再作完整账户。", "",
        "这一方向尚未登记、拟合或回测；不是第70轮失败后调整波动期限规则。现有国债和信用利差资料也做了小范围覆盖检查，较早时期不全，信用趋势另有旧失败研究，暂不追补或重做。EPS、财报、股数、公募申赎慢来源继续暂停。", "",
        "方法依据：[条件分位数回归官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)。官方方法只说明如何估计中位数，不能证明用于510300交易会盈利。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    filenames = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "account_coverage.csv", "source_receipt.json",
        "tests_receipt.json", "tests_receipt_before_fixture_type_fix.json", "output_correction_receipt.json", "saved_verification_receipt.json", "acceptance_outcome.json", "result.json",
        "saved_evaluation_source_replay.csv", "saved_earlier_diagnostic_source_replay.csv", "saved_source_clock_coverage.csv", "saved_metrics_recomputation.csv",
        "saved_account_differences.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv", "saved_no_view_requests.csv", "saved_no_view_holding_summary.csv"]
    for name in filenames:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置与输出修正.json")
    mapping = {"date": "预算原点日期", "execution_date": "实际执行日期", "decision_time": "北京时间决定时刻", "source_date": "美国源日期", "vix_close": "30天预期波动收盘",
        "vix9d_close": "九天预期波动收盘", "available_at": "假定可用北京时间", "source_age_days": "资料自然日年龄", "source_state": "资料状态", "term_ratio": "九天除以30天",
        "target": "有效股票预算比例", "policy_reason": "中文判断原因"}
    for period in ["evaluation", "earlier_diagnostic"]:
        factors = pd.read_parquet(RESEARCH / f"{period}_states.parquet")
        factors["source_state"] = factors.source_state.map({"VIEW": "完整有效", "MISSING_PAIR": "最新一对缺项", "STALE": "资料过期", "BEFORE_SOURCE": "尚无来源", "NOT_USED_NO_NEXT_EXECUTION": "终点后无执行日"})
        factors.rename(columns=mapping).to_csv(OUT / f"{'主评价' if period=='evaluation' else '较早诊断'}_全部逐日因子和状态.csv", index=False, encoding="utf-8-sig")
        for cost in cfg["costs"]:
            for kind in ["ledger", "decisions"]:
                pd.read_parquet(RESEARCH / period / cost / f"{PRIMARY}_{kind}.parquet").to_csv(OUT / f"{period}_{cost}_{kind}.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(RESEARCH / period / cost / f"{PRIMARY}_trades.csv", OUT / f"{period}_{cost}_trades.csv")
    NEXT.write_text("""# 第70轮后：条件中位数继续价值学习

第70轮已完成。实际结果位于 reports/research/510300_volatility_term_v1_output_fix，配置 config/510300_volatility_term_v1_output_fix.json，运行 research/volatility_term_output_fix_v1.py；原config和runner及首次失败输出均保留在没有output_fix的同名位置。原始1个内存账户在保存前因毫秒/纳秒Series.equals误报时间错位而停止，1604日期逐行一致；只改实际值比较，不动参数。来源版本计2、策略设置1、最终4新账户16旧对照，原尝试1个未保存未出收益指标账户单列。不能重跑原始或修正版，不为小输出核对改变历史结论。

主BASE/STRESS净夏普0.043503979630/负0.073105712136，基础年化负0.6079647%、回撤48.1347272%、累计负3.9613526%；早0.439189736643/0.299596430843，基础年化6.7246749%、回撤30.3109130%、累计38.7954961%。主90周期180成交1256持仓收盘，基础33盈57亏；早89周期178成交917持仓收盘，基础44盈45亏；均0追加0整笔未成交，初始末端目标均1。主1227目标1减终点1加30缺失保留持仓=1256；早918减1=917。主31 NO_VIEW中30持仓1空仓，两费用一致；早0 NO_VIEW。

主基础价格负2098.2、分红21101.1、佣金6667.70518、滑点20257.9，净负7922.70518；早价格101963.8、分红11542.1、佣金9122.4077、滑点26792.5，净77590.9923。主压力价格负2890、分红19643.9、佣金12570.38004、滑点34876.1，净负30692.58004。早压力价格97650.6、分红10722.8、佣金17045.51456、滑点48266.4，净43061.48544。2022/23基础各负22.6409%/负12.7379%，合段年化负17.8385%、夏普负1.487529；同来源期限风险不能有效避开中国持续下跌。结束该比值，别改方向、阈值、平滑和等待挽救。

10关键测试最终3.70秒通过，首个失败仅测试整数列注入无穷值、冻结前修好。源3958外连接日期32缺项；独立Python字典、bisect和ZoneInfo重放主3456加早1852=5308行，20指标12差额5646预算请求358周期、62费用计数NO_VIEW核对完成。scripts/review_round70_saved.py只读复算已完成，收尾脚本 finalize_round70_20260908.py 后勿重跑。交付 deliverables/510300海外波动期限_第70轮_20260908，为全中文规则MD和逐日因子/完整账户CSV，无GPT包。

索引截至70应为344不同设置、358已评价来源版本、363登记含5旧未运行、1122主评价记录。当前目标回合属于PROGRESS（70完整结果和独立核对），目标未实现，不标记complete或blocked。股票交易范围仍仅510300及现金，v6支持历史新方法，没有预测显著性前置门。

下一第71轮仅选定方向，尚未写协议、代码、登记、拟合或回测。回到现有第31/32轮原D60自然结束参考周期，检验条件中位数继续价值替代平方误差的平均继续价值估计。经济问题是原平均目标可能受少数大行情牵引；中位数也可能损害大赢家，因此只可检验，不能先宣称改进。不是第70轮阈值变种，也不是复活旧宏观来源。原第31轮停止的是邻近深度、惩罚和确认天数细调；本方向改变损失函数和所估分布位置，明确登记新方法，保留旧结论。

拟限定1模型：最小加权绝对误差的线性条件中位数，QuantileRegressor quantile=0.5、alpha=0、solver=highs，截距、8原持仓因子、按周期等权、训练权重均值和标准差、输入clip±5；不试其他分位点、正则强度、确认次数或模型。0.5是中位数定义，alpha0选择无惩罚绝对误差估计，不以收益选择。月度沿用原最后20个已完全自然退出周期，至少10周期100记录；始终只能用拟合时点已结束周期，不提前纳入部分已知日。资料、标签、原点、已冻模型切割支持均优先重用，不重建原参考账户，不重新获取任何行情。

保持原D60进入与再次进入资格、原固定止损/追踪/最长60日等退出；本次预测连续两个负值才增加退出，零、正值或无模型清除连续计数，无模型继续原规则，不填0预测。新账户必须自身实际持仓构造因子。两段两费用四新账户，与原R32/原R46和BH等保存对照比较。是否另保留原D60无模型旧账户由实际已有文件确定，不新增候选。先核对旧模型接口和原规则，写完整中文协议、必要中位数/时钟/缺失/成交测试，再登记拟合；若未形成跨时期改善就结束不扫分位数。

查重：rg QuantileRegressor|中位数退出|中位数继续|最小绝对|MEDIAN.*EXIT 在research/docs/config未命中。官方方法已查 https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html，说明0.5为条件中位数，支持sample_weight，以线性规划求解；本机版本及实际接口需开工时轻量确认，不把网页1.9.0当作本机版本。

本轮顺手检查现存data/raw/macro三文件只读字段/首尾，没有建立策略：china_credit_spread_3y_daily.parquet2504行2016-08-12至2026-08-18，旧MACRO02已有信用趋势研究；china_government_bond_yields_daily.parquet2509行2016-08-12至2026-08-25；china_government_bond_short_curve_v1.parquet1658行2019-12-23至2026-08-14。均不足2015完整早期覆盖，不展开慢补齐。未用数值选新利差方向。

EPS、财报、股数、公募慢来源继续暂停，现有免费来源足够先做单模型；不创建子任务，不做GPT数值包，不新增安全审计或真实交易。
""", encoding="utf-8")
    record = {"round": 70, "study": result["study_id"], "title": "海外九天与30天预期波动关系", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 2,
        "unsaved_in_memory_account_attempts": 1, "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption"]:
        index[key] += 1
    for key in ["evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 2
    index["evaluation_accounts_in_this_resumption"] += 10
    index.update(updated_at=now(), status="ROUND70_COMPLETE_VOLATILITY_TERM_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计70轮，344不同设置，358已评价来源版本，363登记含5旧未运行，1122主评价记录。第70轮另有1个保存前停止的内存账户尝试，不算新增设置或已交付评价记录。",
        checks="第70轮10测试，3958源日期及5308因子时钟、20指标12差额5646请求358周期和62费用计数NO_VIEW已核对。",
        process_state_note="第70轮失败归因及全中文交付完成；第71轮只选定条件中位数继续价值学习方向，尚未登记或拟合。",
        next_work={"status": "MEDIAN_CONTINUATION_DIRECTION_NOT_REGISTERED", "focus": "仅利用原已结束持仓周期，检验条件中位数继续价值；先固定完整进出场和单模型", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_volatility_term_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "VOLATILITY_TERM_ROUND70_CHINESE_RESULTS", "rounds": [70], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "VOLATILITY_TERM_RULES_AND_FULL_ACCOUNT_RESULTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery, exclusive=True)
    print(json.dumps({"交付": delivery, "研究状态": index["status"], "下一步": index["next_work"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
