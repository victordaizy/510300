"""交付放量组合、动态退出及同状态预测比较，保留完整失败结果。"""
from bisect import bisect_right
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES, predict
from scripts.deliver_simple_strategy_rounds34_36_20260907 import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300放量组合与动态退出_第51至52轮_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
STUDIES = [
    (51, "climax_learned_equal_blend", "放量收强与原学习退出各半", "CLIMAX_LEARNED_HALF",
     "COMPLETED_CLIMAX_BLEND_NO_ROBUST_IMPROVEMENT",
     "主评价和较早夏普均低于原急跌各半组合，较早原放量子策略亏损。停止这组权重和放量门槛调整，不采用。"),
    (52, "dynamic_continuation_exit", "学习含后续退出选择的继续价值", "DYNAMIC_CONTINUATION",
     "COMPLETED_DYNAMIC_CONTINUATION_NO_IMPROVEMENT",
     "动态继续价值在两段、两档费用下均低于同为单次确认的原标签对照。原标签单次确认仅较早略好、主评价明显变差。两种改动都不替换原模型，停止本方法的迭代次数、确认次数及惩罚强度调整。"),
]


def path(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def select(result, key, period="evaluation", cost="BASE"):
    rows = result["all_metrics" if period == "evaluation" else "earlier_diagnostics"]
    return next(m for m in rows if m["model"] == key and m["cost"] == cost)


def common_state_predictions():
    models = json.loads((path("dynamic_continuation_exit") / "saved_models.json").read_text(encoding="utf-8"))["models"]
    indexes = [r["fit_index"] for r in models]
    rows, summary = [], []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            source = P32 / period / cost / "REARM_RIDGE_decisions.parquet"
            decisions = pd.read_parquet(source)
            decisions = decisions[decisions.learning_status.eq("PREDICTION_AVAILABLE")]
            local = []
            for row in decisions.to_dict("records"):
                pos = bisect_right(indexes, int(row["origin_index"])) - 1
                require(pos >= 0, "共同持仓状态找不到当时模型")
                model = models[pos]
                require(model["status"] == "FIT_COMPLETE" and model["latest_exit_index"] <= model["fit_index"] <= row["origin_index"],
                        "共同状态比较使用未成熟动态模型")
                value = predict(model["model"], np.array([row[k] for k in FEATURES], dtype=float))
                local.append({"period": period, "cost": cost, "origin": row["origin"], "original_cycle_id": row["learning_cycle_id"],
                              "fit_origin": model["fit_origin"], "natural_prediction": float(row["continuation_prediction"]),
                              "dynamic_prediction": value, "difference": value - row["continuation_prediction"]})
            frame = pd.DataFrame(local)
            summary.append({"period": period, "cost": cost, "same_saved_states": len(frame),
                            "natural_negative_dynamic_nonnegative": int((frame.natural_prediction.lt(0) & frame.dynamic_prediction.ge(0)).sum()),
                            "natural_nonnegative_dynamic_negative": int((frame.natural_prediction.ge(0) & frame.dynamic_prediction.lt(0)).sum()),
                            "higher_dynamic_predictions": int(frame.difference.gt(0).sum()),
                            "mean_prediction_difference": float(frame.difference.mean()),
                            "source_sha256": digest(source)})
            rows.extend(local)
    return rows, summary


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 51)), set(range(1, 53))], "已有其他轮次，停止覆盖研究索引")
    results = {n: json.loads((path(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, *_ in STUDIES}
    new_keys = {51: {"CLIMAX_LEARNED_HALF"}, 52: {"DYNAMIC_CONTINUATION", "NATURAL_SINGLE_CONFIRM"}}
    require(not any(m["meets_point_target"] for n, r in results.items() for m in r["all_metrics"] if m["model"] in new_keys[n]),
            "新结果已达到点估计目标，需重新写验收说明")
    OUT.mkdir(parents=True, exist_ok=True)
    records, copies, differences, cycle_stats = [], [], [], []
    for n, slug, title, primary, status, decision in STUDIES:
        result = results[n]
        records.append({"round": n, "study": result["study_id"], "title": title, "status": status,
                        "result": str((path(slug) / "result.json").relative_to(ROOT)),
                        **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
                                                 "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
                        "evaluated_candidate_source_runs": result["candidate_configurations"],
                        "primary_base": select(result, primary), "primary_stress": select(result, primary, cost="STRESS"),
                        "post_selected_best_base": result["post_selected_best_base"]})
        write_json(path(slug) / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
                   "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0})
        for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "result.json"]:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((path(slug) / filename).read_bytes())
            copies.append(dest)
        comparisons = [("CLIMAX_LEARNED_HALF", "OLD_PANIC_BLEND")] if n == 51 else [
            ("DYNAMIC_CONTINUATION", "NATURAL_SINGLE_CONFIRM"), ("DYNAMIC_CONTINUATION", "REARM_RIDGE"), ("NATURAL_SINGLE_CONFIRM", "REARM_RIDGE")]
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                folder = path(slug) / period / cost
                for new, old in comparisons:
                    current = pd.read_parquet(folder / f"{new}_ledger.parquet")
                    prior = pd.read_parquet(folder / f"{old}_ledger.parquet")
                    require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(prior.date)), "新旧账户完整日期不一致")
                    d = {"round": n, "period": period, "cost": cost, "new_model": new, "control": old,
                         "final_equity_difference": float(current.equity.iloc[-1] - prior.equity.iloc[-1]),
                         "price_pnl_difference": float(current.price_pnl.sum() - prior.price_pnl.sum()),
                         "dividend_difference": float(current.dividend_recognized.sum() - prior.dividend_recognized.sum()),
                         "extra_commission_and_slippage": float(current.commission.sum() + current.slippage_cost.sum() - prior.commission.sum() - prior.slippage_cost.sum())}
                    require(abs(d["final_equity_difference"] - d["price_pnl_difference"] - d["dividend_difference"] + d["extra_commission_and_slippage"]) < 1e-6,
                            "新旧账户差额无法由价格、股息及费用核对")
                    differences.append(d)
                new_keys = ["CLIMAX_LEARNED_HALF"] if n == 51 else ["DYNAMIC_CONTINUATION", "NATURAL_SINGLE_CONFIRM"]
                for key in new_keys:
                    filename = f"{key}_trades.csv" if n == 51 else f"{key}_cycles.csv"
                    dest = OUT / f"第{n}轮_{period}_{cost}_{filename}"
                    dest.write_bytes((folder / filename).read_bytes())
                    copies.append(dest)
                    if n == 52:
                        c = pd.read_csv(folder / filename)
                        cycle_stats.append({"period": period, "cost": cost, "model": key, "cycles": len(c),
                                            "mean_holding_intervals": float(c.holding_intervals.mean()),
                                            "minimum_holding_intervals": int(c.holding_intervals.min()),
                                            "maximum_holding_intervals": int(c.holding_intervals.max()),
                                            "learned_exit_cycles": int(c.exit_reasons.str.contains("学习条件", regex=False).sum())})
    common, common_summary = common_state_predictions()
    for filename, rows in [("两轮新旧账户价格股息费用全部差额.csv", differences), ("第52轮_全部周期持有时间统计.csv", cycle_stats),
                           ("第52轮_原持仓相同状态的新旧模型预测.csv", common), ("第52轮_相同状态预测汇总.csv", common_summary)]:
        dest = OUT / filename
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
        copies.append(dest)
    extras = [
        (path("climax_learned_equal_blend") / "source_statistics.csv", "第51轮_原信号机会统计.csv"),
        (path("climax_learned_equal_blend") / "state_counts.csv", "第51轮_全部三信号交集.csv"),
        (path("dynamic_continuation_exit") / "training_receipts.csv", "第52轮_月度模型训练记录.csv"),
        (path("dynamic_continuation_exit") / "all_value_iterations.csv", "第52轮_全部价值迭代记录.csv"),
        (path("dynamic_continuation_exit") / "model_coverage.csv", "第52轮_模型实际使用与无模型统计.csv"),
        (path("dynamic_continuation_exit") / "每月动态退出模型中文规则.md", "第52轮_每月动态退出模型中文规则.md"),
        (ROOT / "deliverables/510300学习目标与互补组合_第44至46轮_20260907/沿用原模型_每月实际中文规则.md", "原模型_每月实际中文规则.md"),
    ]
    for source, name in extras:
        dest = OUT / name
        dest.write_bytes(source.read_bytes())
        copies.append(dest)
    summary_path = OUT / "第52轮_保存状态诊断说明.json"
    write_json(summary_path, {"created_at": now(), "status": "SAVED_COMMON_STATE_PREDICTION_COMPARISON_NOT_NEW_ACCOUNT", "summary": common_summary,
               "definition": "只在原第32轮已保存、原模型可用的实际持仓状态上，调用当时已训练的第52轮模型，与原预测比较。两模型输入状态完全相同。",
               "limits": "这是看过结果后的模型行为解释，不是新账户、独立验证或下一次可实现收益。重叠持仓日不是独立样本。",
               "new_accounts": 0, "new_fits": 0, "new_data": 0})
    copies.append(summary_path)
    iterations = pd.read_csv(path("dynamic_continuation_exit") / "all_value_iterations.csv")
    last_iterations = iterations[iterations.iteration.eq(60)]
    require(len(last_iterations) == 114, "实际月度模型数量与迭代记录不符")
    r51, r52 = results[51], results[52]
    text = ["# 510300：放量组合与动态退出的完整结果", "", "第51至52轮，2026年9月7日。规则、因子、进入和退出均用中文说明。", "",
            "## 先看结论", "",
            "仍未达到完整账户扣费净夏普1.2。这次完成一个放量信号组合，以及动态退出与其单次确认对照，共三个设置。放量组合低于此前急跌各半组合；把后续退出选择计入模型后，持有时间和风险增加，收益反而下降。这些改动都不采用，原基线和旧失败记录保留。", "",
            "EPS、前瞻盈利研报日期、股数、估值、财报和公募来源补齐继续暂停。研究直接使用现有价格、成交量及分红记录，没有下载新行情，没有制作GPT审阅数值包。", "",
            "|方案|主评价基础夏普|主评价压力夏普|主评价基础复合年化|主评价基础最大回撤幅度|较早基础夏普|较早压力夏普|",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for result, key, label in [(r52, "REARM_RIDGE", "原学习退出与等待新机会"), (r52, "PANIC_LEARNED_HALF", "原急跌与学习退出各半，仅局部线索"),
                               (r51, "CLIMAX_LEARNED_HALF", "第51轮：放量收强与学习退出各半"),
                               (r52, "DYNAMIC_CONTINUATION", "第52轮：计入后续退出选择"),
                               (r52, "NATURAL_SINGLE_CONFIRM", "第52轮对照：原标签改为单次确认")]:
        b, s = select(result, key), select(result, key, cost="STRESS")
        eb, es = select(result, key, "earlier_diagnostic"), select(result, key, "earlier_diagnostic", "STRESS")
        text.append(f"|{label}|{b['net_sharpe']:.3f}|{s['net_sharpe']:.3f}|{b['annualized_return']:.2%}|{-b['max_drawdown']:.2%}|{eb['net_sharpe']:.3f}|{es['net_sharpe']:.3f}|")
    text += ["", "主评价为2020年1月2日至2026年8月14日终点开盘，1604个账户日；较早诊断为2015年1月5日至2019年12月31日终点开盘，1219日。两段各从20万元开始，保留全部空仓日，扣除交易费用，现金及无风险收益均假设为零，按242日年化。最大回撤在上表按正幅度展示，后面的完整明细保留负号。", "",
             "两段历史均已被多轮观察，不能称为全新留出验证。原急跌组合的较早夏普仅0.604，其急跌子策略主评价只有三个不同持仓周期；主评价0.944不能当作已验证的稳定高夏普。", "",
             "## 放量组合为什么没有改善", "",
             "本轮放量回升与此前高波动急跌回升是不同信号。放量信号在原两个信号都未要求持有时，增加了主评价35个、较早9个持仓意向日；天数不是独立事件数。其原单策略主评价7个完整周期、较早6个完整周期。较早原放量单策略基础夏普−0.513、复合年化−2.84%，压力夏普−0.544。新增机会并没有在较早历史中形成正贡献。", "",
             "各半组合的主评价基础夏普0.891、较早0.493，均低于原急跌各半组合的0.944和0.604。主评价压力费用下复合年化3.55%，低于对应买入持有约3.61%。这组组合到此结束，不继续扫描比例、量比或跌幅门槛。", "",
             "## 动态退出为什么没有改善", "",
             "本轮对同一批34个原自然结束周期、1461条状态建立一步收益，再用上次迭代估计的后续退出选择更新继续持有价值。月度时点、成熟周期集合和八个因子标准化都与原模型一致。共114个月度最终模型，每个固定60次回归，实际6840次回归；另27个月度时点没有足够成熟周期。6840是算法内部拟合次数，不是6840个独立策略，也不是6840次独立验证。", "",
             "主评价动态模型只有15个持仓周期，其中4个由学习条件退出，平均每周期持有32.6个开盘间隔；同为单次确认的原标签对照有24个周期，其中23个包含学习退出，平均持有9.0个间隔。前者平均仓位30.39%，后者13.44%。动态模型年化收益3.17%，低于对照4.02%，最大回撤22.18%，明显高于对照10.07%。", "",
             "两种模型较早历史都只有9个周期、其中2个包含学习退出，均存在204个没有模型的持仓收盘。动态模型较早夏普0.594，原标签单次确认0.775；后者略高于原两次确认0.748，但其主评价从原0.705降至0.603，不能据此统一替换。", "",
             "为避免把不同持仓路径混成模型本身的差异，另把新旧模型放到原第32轮完全相同、原模型可用的持仓状态上比较。只调用已保存的当时模型，没有新训练、新标签或新账户。", "",
             "|原状态所属时期和费用|相同持仓状态数|原预测为负、新预测非负|原预测非负、新预测为负|新预测高于原预测的状态数|新减原平均预测差|",
             "|---|---:|---:|---:|---:|---:|"]
    for s in common_summary:
        label = ("主评价" if s["period"] == "evaluation" else "较早") + "／" + ("基础" if s["cost"] == "BASE" else "压力")
        text.append(f"|{label}|{s['same_saved_states']}|{s['natural_negative_dynamic_nonnegative']}|{s['natural_nonnegative_dynamic_negative']}|{s['higher_dynamic_predictions']}|{s['mean_prediction_difference']:.2%}|")
    text += ["", "这里的预测差是以模型标签收益为单位，百分数差对应百分点，不是实际账户年化收益。共同状态诊断用于说明模型行为，不能据此宣称找到了真实退出收益，也不能把重叠日期当成独立样本。", "",
             f"在114个月度训练中，第60次与第59次在训练状态上的最大预测变化，其跨月份中位数为{last_iterations.maximum_prediction_change.median():.8f}，最高为{last_iterations.maximum_prediction_change.max():.8f}。这只记录迭代变化，不能证明收敛，更不能证明盈利。训练状态中的负预测条数，跨月份中位数从首轮359.5降到末轮56.0；这些状态在月份间重复出现，不能作为独立统计检验。", "",
             "本轮结果支持停止这一动态价值近似，而不支持继续延长迭代次数来期待收益变好。只取消两天确认的对照也没有两段共同改善。", "",
             "## 全部账户差额", "", "以下为新方案减表中对照，直接由完整逐日账本核对。额外费用为负表示费用节省。", "",
             "|轮次及比较|时期及费用|终点资产差额（元）|价格损益差（元）|股息差（元）|额外佣金及滑点（元）|", "|---|---|---:|---:|---:|---:|"]
    names = {"CLIMAX_LEARNED_HALF": "放量各半", "OLD_PANIC_BLEND": "原急跌各半", "DYNAMIC_CONTINUATION": "动态退出",
             "NATURAL_SINGLE_CONFIRM": "原标签单次确认", "REARM_RIDGE": "原标签两次确认"}
    for d in differences:
        label = ("主评价" if d["period"] == "evaluation" else "较早") + "／" + ("基础" if d["cost"] == "BASE" else "压力")
        text.append(f"|第{d['round']}轮：{names[d['new_model']]}减{names[d['control']]}|{label}|{d['final_equity_difference']:.2f}|{d['price_pnl_difference']:.2f}|{d['dividend_difference']:.2f}|{d['extra_commission_and_slippage']:.2f}|")
    text.append("")
    for n, slug, title, primary, status, decision in STUDIES:
        text.extend([f"## 第{n}轮全部因子、进入、持有与退出规则", ""])
        lines = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8").splitlines()[1:]
        text.extend(["#" + line if line.startswith("##") else line for line in lines])
        text.extend(["", f"### 第{n}轮全部主评价", ""] + table(results[n]["all_metrics"]))
        text.extend([f"### 第{n}轮全部较早诊断", ""] + table(results[n]["earlier_diagnostics"]))
        text.extend(["本轮验收：" + decision, ""])
    next_work = [
        "第51轮CLIMAX_LEARNED_HALF主评价基础0.890953、压力0.817272，较早0.493114、压力0.463704，未改善原第46轮组合；原V1较早6周期亏损。停止这组比例及量价门槛。",
        "第52轮DYNAMIC_CONTINUATION主评价0.331979、压力0.304054，较早0.594319、压力0.572287；同次确认的原自然标签为0.603303、0.531011、0.774501、0.751135。动态近似两段两费用均更差，停止价值迭代次数和惩罚强度调整。原标签单次确认主评价低于原两次确认，也不采用。",
        "第52轮训练114个月度最终模型、27无模型更新、6840内部回归，原34成熟参考周期和1461状态不变。实际动态主评价15周期、4个学习退出，平均32.6持有间隔，对照24周期、23个学习退出、平均9.0间隔。迭代变化和同状态预测已保存，不重复拟合。",
        "停止继续围绕同一批34参考周期改退出标签或确认次数。下一步先简要核对现有简单策略库，优先选择能引入不同进入机会、无需补源且没有被完整重复检验的机制，再固定一个或少量清楚的进出场比较。现有日历、月末和趋势/反转方法有旧研究，不能改名重复。当前没有新研究登记或运行。",
        "原第32轮仍作比较基线，第46轮0.944主评价、0.604较早只作局部线索，完整净夏普1.2与稳定超额尚未实现。EPS、研报日期、股数、估值、财报及公募来源补齐保持暂停；其他ETF问题待原回复，不重复问，不做GPT数值包、不建子任务。",
    ]
    text += ["## 已完成工作及后续重点", "",
             "五项必要测试通过：一步收益与登记权益的金额单位、自然终点不继续估值、同周期连接、缺失状态拒绝及成熟时间。两轮16项新旧账户差额全部可由价格、分红和费用核对；模型实际使用、每次迭代、持有周期、交易和同状态预测全部保留。没有安全审计。", "",
             "这次3个设置，每段历史24个账户记录。主评价6个新账户、18个复用对照；较早6个新候选账户、1个补算的原放量压力费用对照、17个复用对照。新补算原对照不增加配置数，没有新参考账户。", "",
             "累计完成52轮、323个不同配置或范围、335个已评价来源版本、944个主评价记录；登记来源版本340个，含5个旧未运行绑定。复用账户、不同费用和重复训练都不能当作新的独立策略证据。", "",
             "下一步停止围绕同一批34个参考周期继续改退出目标和确认次数，把研究转回能带来不同进入机会的已有简单信号。先查已完成规则和结果，排除重复，再确定下一项完整进出场比较。当前没有下一项已登记或运行的回测。持续研究目标保持活动，完整净夏普1.2仍未完成。", "",
             "## 普通交付文件", ""]
    for p in copies:
        text.append(f"- [{p.name}]({p.name})")
    for n, slug, *_ in STUDIES:
        text.append(f"- 第{n}轮[原始结果索引](<{(path(slug)/'result.json').as_posix()}>)。")
    document = OUT / "放量组合与动态退出_全部因子规则和历史表现.md"
    content = "\n".join(text) + "\n"
    require("```" not in content, "中文规则被代码替代")
    document.write_text(content, encoding="utf-8")
    for match in re.finditer(r"\]\(([^)]+)\)", content):
        target = match.group(1).strip("<>")
        if not target.startswith("http"):
            require((OUT / target).is_file(), "交付文件链接不存在：" + target)
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                      "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[k] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS51_52_COMPLETE_EXISTING_SIMPLE_SIGNAL_RESEARCH_CONTINUES", latest_completed_round=records[-1],
                 running_studies=[], goal_achieved=False, process_state_note="第51至52轮及同状态保存预测比较完成，进程均已退出，下一项未登记未运行。",
                 checks="5项必要测试通过；16项新旧全账户差额、114月度最终模型及同状态预测已核对。",
                 count_warning="累计52轮、323不同配置或范围、335已评价来源版本、944主评价记录；登记340含5旧未运行绑定。较早、拟合、参考另计。",
                 next_work=next_work, latest_saved_dynamic_prediction_diagnostic=str(summary_path.relative_to(ROOT)))
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [51, 52]]
    delivery = {"created_at": now(), "type": "CLIMAX_BLEND_DYNAMIC_EXIT_ROUNDS51_52_CHINESE_RESULTS", "rounds": [51, 52],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 52 and index["evaluation_accounts_in_this_resumption"] == 944 and
            index["evaluated_configurations_in_this_resumption"] == 323 and index["evaluated_candidate_source_runs_including_corrected_replays"] == 335 and
            index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 340, "累计研究数量不符")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_ACCOUNTS_AND_SAME_STATE_PREDICTIONS", "rounds": [51, 52],
               "new_configurations": 3, "main_records": 24, "new_main_accounts": 6, "reused_main_accounts": 18,
               "earlier_records": 24, "new_earlier_candidate_accounts": 6, "new_earlier_control_accounts": 1, "reused_earlier_accounts": 17,
               "new_model_fits": 6840, "monthly_final_models": 114, "new_reference_accounts": 0,
               "necessary_tests_passed": 5, "saved_account_difference_checks": len(differences),
               "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "文档字符": len(content), "普通文件": len(receipt["files"]),
                      "累计完成": 52, "主评价记录": 944, "共同状态比较": common_summary, "目标完成": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
