"""保存日历固定组合与月度启停的结果、失败归因及中文交付。"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300日历组合与月度启停_第57至58轮_20260907"
STUDIES = [(57, "calendar_learned_equal_blend", "月内两端与原学习退出固定各半", "CALENDAR_LEARNED_HALF",
            "基础主评价只略高于原学习退出，压力费用与较早历史变差；结束固定日历组合，不调整权重或月初月末边界。"),
           (58, "calendar_monthly_activation", "用过去一年净收益每月启停日历预算", "CALENDAR_MONTHLY_ACTIVATION",
            "较早结果比固定组合改善，但主评价变差，两段两费用均低于原学习退出；结束该月度启停，不调整回看年限、收益阈值、月份、频率或预算比例。")]


def folder(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def metric(result, key, period="evaluation", cost="BASE"):
    return next(m for m in result["all_metrics" if period == "evaluation" else "earlier_diagnostics"] if m["model"] == key and m["cost"] == cost)


def table(rows):
    lines = ["|方案|费用|净夏普|复合年化|最大回撤幅度|较买入持有年化差|平均股票仓位|成交笔数|佣金（元）|滑点（元）|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        lines.append(f"|{m['name']}|{'基础' if m['cost']=='BASE' else '压力'}|{m['net_sharpe']:.3f}|{m['annualized_return']:.2%}|{-m['max_drawdown']:.2%}|{100*m['annualized_return_excess_vs_buy_hold']:.2f}个百分点|{m['mean_exposure']:.2%}|{m['trade_count']}|{m['commission']:.2f}|{m['slippage_cost']:.2f}|")
    return lines + [""]


def saved_month_diagnostic():
    """只按已保存的月初决定分组旧日历固定路径，不产生新策略账户。"""
    p = folder("calendar_monthly_activation")
    updates = pd.read_csv(p / "monthly_activation_updates.csv", parse_dates=["market_date", "execution_date", "last_valid_evidence_date"])
    updates["month"] = updates.execution_date.dt.to_period("M").astype(str)
    rows = []
    for period in ["evaluation", "earlier_diagnostic"]:
        local = updates[updates.period.eq(period)].set_index("month")
        require(local.index.is_unique, "一个月有多个启停决定")
        for cost in ["BASE", "STRESS"]:
            ledger = pd.read_parquet(p / period / cost / "K2_MONTH_EDGE_ledger.parquet")
            for month, g in ledger.groupby(ledger.date.dt.to_period("M").astype(str)):
                choice = local.loc[month]
                rows.append({"period": period, "cost": cost, "month": month, "gate_evidence_date": choice.last_valid_evidence_date,
                             "past_reference_year_return": choice.current_reference_year_return, "calendar_enabled": bool(choice.calendar_enabled),
                             "old_calendar_month_return": float(np.expm1(np.log1p(g.net_return).sum())),
                             "month_has_terminal_open": bool(g.mark_clock.ne("CLOSE").any()), "days": len(g)})
    months = pd.DataFrame(rows)
    groups = []
    for (period, cost, enabled), g in months[~months.month_has_terminal_open].groupby(["period", "cost", "calendar_enabled"]):
        groups.append({"period": period, "cost": cost, "calendar_enabled": bool(enabled), "complete_months": len(g),
                       "positive_months": int(g.old_calendar_month_return.gt(0).sum()), "mean_month_return": float(g.old_calendar_month_return.mean()),
                       "median_month_return": float(g.old_calendar_month_return.median())})
    months.to_csv(p / "saved_calendar_month_outcomes.csv", index=False, encoding="utf-8-sig")
    grouped = pd.DataFrame(groups)
    grouped.to_csv(p / "saved_calendar_month_groups.csv", index=False, encoding="utf-8-sig")
    write_json(p / "saved_diagnostic_receipt.json", {"created_at": now(), "status": "SAVED_CONTROL_PATH_GROUPING_ONLY", "month_cost_rows": len(months),
               "terminal_open_month_cost_rows": int(months.month_has_terminal_open.sum()), "new_accounts": 0, "new_models": 0,
               "post_result_diagnostic": True, "does_not_estimate_tradable_selected_month_portfolio": True,
               "code": "scripts/finalize_rounds57_58_20260907.py"})
    return grouped


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 57)), set(range(1, 59))], "已有其他轮次，不覆盖索引")
    results = {n: json.loads((folder(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, *_ in STUDIES}
    require(not any(r["historical_point_target_met"] for r in results.values()), "出现达到1.2的候选，不能使用未达标交付")
    OUT.mkdir(parents=True, exist_ok=True)
    groups = saved_month_diagnostic()
    copies, deltas, records = [], [], []
    for n, slug, title, primary, decision in STUDIES:
        p, result = folder(slug), results[n]
        status = "COMPLETED_NO_ACROSS_PERIOD_AND_COST_IMPROVEMENT"
        write_json(p / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
                   "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0})
        records.append({"round": n, "study": result["study_id"], "title": title, "status": status, "result": str((p / "result.json").relative_to(ROOT)),
                        **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
                                                 "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
                        "new_earlier_control_accounts": result.get("new_earlier_control_accounts", 0),
                        "evaluated_candidate_source_runs": result["candidate_configurations"], "primary_base": metric(result, primary),
                        "primary_stress": metric(result, primary, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]})
        filenames = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "state_counts.csv", "result.json", "acceptance_outcome.json"]
        filenames += ["execution_statistics.csv"] if n == 57 else ["monthly_activation_updates.csv", "saved_calendar_month_outcomes.csv", "saved_calendar_month_groups.csv", "saved_diagnostic_receipt.json"]
        for filename in filenames:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((p / filename).read_bytes())
            copies.append(dest)
        for period in ["evaluation", "earlier_diagnostic"]:
            states = pd.read_parquet(p / f"{period}_states.parquet")
            cn = {"market_date": "价格信息截止日", "execution_date": "计划成交日", "decision_time": "九点决策时间", "month_first3": "月初三交易日条件",
                  "month_last5_calendar": "月末五自然日条件", "calendar_state": "日历状态", "learned_state": "原学习状态", "target": "组合股票目标",
                  "inputs_complete": "两状态完整", "is_update": "本次月度更新", "reference_history_available": "参考过去一年收益完整",
                  "current_reference_year_return": "截至该收盘参考过去一年收益", "calendar_enabled": "当月启用日历", "calendar_budget": "日历预算",
                  "learned_budget": "学习预算", "has_valid_activation_decision": "已有有效启停决定", "last_valid_evidence_date": "上次有效启停观察截止日",
                  "last_valid_reference_year_return": "上次启停所用参考收益", "activation_status": "启停状态", "static_half_target": "固定各半对照目标"}
            dest = OUT / f"第{n}轮_{period}_全部因子与预算.csv"
            states.rename(columns=cn).to_csv(dest, index=False, encoding="utf-8-sig")
            copies.append(dest)
            for cost in ["BASE", "STRESS"]:
                q = p / period / cost
                a = pd.read_parquet(q / f"{primary}_ledger.parquet")
                controls = ["REARM_RIDGE", "PANIC_LEARNED_HALF"] if n == 57 else ["CALENDAR_LEARNED_HALF", "REARM_RIDGE"]
                for control in controls:
                    b = pd.read_parquet(q / f"{control}_ledger.parquet")
                    require(pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date)), "账户比较日期不一致")
                    d = {"round": n, "period": period, "cost": cost, "candidate": primary, "control": control,
                         "final_equity_difference": float(a.equity.iloc[-1] - b.equity.iloc[-1]),
                         "price_pnl_difference": float(a.price_pnl.sum() - b.price_pnl.sum()),
                         "dividend_difference": float(a.dividend_recognized.sum() - b.dividend_recognized.sum()),
                         "extra_fees": float(a.commission.sum() + a.slippage_cost.sum() - b.commission.sum() - b.slippage_cost.sum())}
                    require(abs(d["final_equity_difference"] - d["price_pnl_difference"] - d["dividend_difference"] + d["extra_fees"]) < 1e-6, "费用股息价格差未核对账户")
                    deltas.append(d)
                for suffix in ["trades.csv", "decisions.parquet"]:
                    dest = OUT / f"第{n}轮_{period}_{cost}_{suffix}"
                    dest.write_bytes((q / f"{primary}_{suffix}").read_bytes())
                    copies.append(dest)
    dest = OUT / "两轮完整账户_价格股息费用差额.csv"
    pd.DataFrame(deltas).to_csv(dest, index=False, encoding="utf-8-sig")
    copies.append(dest)
    r57, r58 = results[57], results[58]
    text = ["# 510300：固定组合与按过去表现切换策略的结果", "", "第57至58轮，2026年9月7日。所有因子、预算变化、进入、退出及历史表现用中文呈现。", "",
            "## 管理层先看结论", "",
            "净夏普1.2仍未实现。这次先组合两种不同机制，再尝试按当时过去表现启停其中一种，都没有跨时期和费用压力的一致改善。固定组合增加了交易费用；月度启停减少了交易，却没有选准下一阶段更有收益的月份。两项方法结束，不继续调权重、月末边界或过去收益回看长度。", "",
            "|方案|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|主评价基础年化|主评价基础最大回撤幅度|",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for result, key in [(r57, "CALENDAR_LEARNED_HALF"), (r58, "CALENDAR_MONTHLY_ACTIVATION"), (r57, "K2_MONTH_EDGE"),
                        (r57, "REARM_RIDGE"), (r57, "PANIC_LEARNED_HALF"), (r57, "BUY_HOLD")]:
        b, s = metric(result, key), metric(result, key, cost="STRESS")
        eb, es = metric(result, key, "earlier_diagnostic"), metric(result, key, "earlier_diagnostic", "STRESS")
        text.append(f"|{b['name']}|{b['net_sharpe']:.3f}|{s['net_sharpe']:.3f}|{eb['net_sharpe']:.3f}|{es['net_sharpe']:.3f}|{b['annualized_return']:.2%}|{-b['max_drawdown']:.2%}|")
    text += ["", "主评价2020年1月2日至2026年8月14日终点开盘1604日；较早2015年1月5日至2019年12月31日终点开盘1219日。两段分别20万元，空仓日、交易费用、分红与真实成交限制全部保留；242日年化，现金及无风险收益零。历史已被多轮观察，不能视为新的独立检验。", "",
             "## 固定各半为什么不够好", "",
             "第57轮把旧月内两端规则与旧学习退出状态各赋予一半预算。两者在同一天同时要求持有，组合才请求全仓；只有一个要求持有则半仓；两者都退出才归零。状态分别维护，实际资金合并在一个账户，不是把两个净值简单平均。", "",
             "主评价中，日历独自要求持有444个决策日、学习独自要求187日、两者同时66日、都不要求907日；较早分别293、206、93、627日。日历确实补充了学习策略的空仓日，但多占用市场时间不等于增加稳定收益。", "",
             "主评价基础费用下夏普0.712略高于原学习0.705，但最大回撤从10.59%扩大到13.88%；压力夏普0.575低于原学习0.641。较早夏普只有0.284，低于原学习0.748。旧日历策略较早本身净夏普负0.165、年化负3.76%，不能因为其主评价0.520，就预先把它当成稳定的补充来源。", "",
             "新组合主评价207笔买卖、较早139笔；原学习分别48、18笔。主评价新组合基础佣金和滑点合计22,012.29元，压力39,351.51元。下面的账户差额把交易成本与价格、股息的变化分开，不能把所有失败仅归因于成本。", "",
             "## 月度切换为什么仍没选准", "",
             "第58轮每月首个交易日九点，用旧日历连续基础账户此前242个完整交易日净收益作判断。赚钱则各半；未赚钱则全部预算交给学习状态；月中不变。主评价80次月度决定中45次启用日历，较早60次中27次启用，全部判断均有完整参考收益。", "",
             "主评价成交从固定组合207笔降至150笔，较早从139笔降至79笔；较早夏普从0.284升至0.370，但主评价从0.712降至0.633。两段两费用的夏普都低于原学习退出。历史表现驱动的切换真正发生了，但它没有提高完整目标表现。", "",
             "为解释原因，固定已保存的旧日历评价账户，不改其持仓、份额、退出或费用，只按每个月开头已经作出的启停决定分组，查看该月随后实际发生的原路径收益。启停输入来自2013年起连续参考账户，分组输出来自各评价期独立20万元旧日历账户，两个账户用途不同，不能混为一条路径。", "",
             "|时期及费用|月初决定|完整月份数|其中盈利月份|随后旧日历平均月收益|随后旧日历月收益中位数|",
             "|---|---|---:|---:|---:|---:|"]
    for r in groups.to_dict("records"):
        label = ("主评价" if r["period"] == "evaluation" else "较早") + "／" + ("基础" if r["cost"] == "BASE" else "压力")
        text.append(f"|{label}|{'启用日历' if r['calendar_enabled'] else '关闭日历'}|{r['complete_months']}|{r['positive_months']}|{r['mean_month_return']:.3%}|{r['median_month_return']:.3%}|")
    text += ["", "主评价基础费用中，决定启用的45个完整月份随后平均月收益仅0.158%，关闭的34个月份反而为0.979%；较早启用的26个月份为负0.738%，关闭的33个月份为正0.102%。这些结果与“过去赚钱就更适合下月继续配置”的假设不一致。它可能反映跟随表现出现滞后，但分组本身不能证明因果或预测能力。", "",
             "每段最后一个月份包含统一终点开盘，单列保留在逐月文件中，不混入完整月的平均值：主评价完整月份79个、较早59个，分别另有1个终点月，两档费用合计保存280条逐月记录。所有终点月仍完整计入正式账户净夏普，没有删去其收益。", "",
             "这是事后诊断，不是新可交易策略。不能把启用与关闭标签反过来再宣称成功，也不能将筛选月份收益直接相乘当成新账户，因为实际预算、交易费用、入场前持仓及后续资金都会变化。", "",
             "## 实际账户差额", "", "全部为候选减对照；额外费用为负表示节省。16条均能核对至实际终点资产。", "",
             "|轮次|时期及费用|对照|终点资产差（元）|价格损益差（元）|股息差（元）|额外费用（元）|",
             "|---|---|---|---:|---:|---:|---:|"]
    names = {m["model"]: m["name"] for result in results.values() for m in result["all_metrics"]}
    for d in deltas:
        label = ("主评价" if d["period"] == "evaluation" else "较早") + "／" + ("基础" if d["cost"] == "BASE" else "压力")
        text.append(f"|{d['round']}|{label}|{names[d['control']]}|{d['final_equity_difference']:.2f}|{d['price_pnl_difference']:.2f}|{d['dividend_difference']:.2f}|{d['extra_fees']:.2f}|")
    for n, slug, title, primary, decision in STUDIES:
        text += ["", f"## 第{n}轮完整中文因子及进出场规则", ""]
        lines = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8").splitlines()[1:]
        for line in lines:
            line = re.sub(r"\]\((510300_[^)]+\.md)\)", lambda m: "](<" + (ROOT / "docs" / m.group(1)).as_posix() + ">)", line)
            text.append("#" + line if line.startswith("##") else line)
        text += ["", f"### 第{n}轮全部主评价", ""] + table(results[n]["all_metrics"])
        text += [f"### 第{n}轮全部较早诊断", ""] + table(results[n]["earlier_diagnostics"])
        text += ["本轮验收：" + decision, ""]
    next_work = [
        "第57轮固定日历与R32各半失败。主评价0.711769、压力0.575499，较早0.283839、压力0.190408；主评价207成交、较早139。K2月内两端较早新补旧对照夏普负0.165207、压力负0.280625，年化负3.7621%，不是稳健补充。停止日历权重和月初月末边界调整。",
        "第58轮按过去242日K2连续基础净收益月初启停失败。主评价0.632578、压力0.507578，较早0.370298、压力0.311000；150/79成交。月初80/60次决定、45/27次启用，0无效更新。新生成1个2013-06-03开始连续K2参考账户，没有新模型。停止该方法窗口、门槛、启停频率、月份及预算比例。",
        "第58轮事后固定旧K2评价账户按月初决定分组完成。主评价基础启用45完整月随后均值0.1579%，关闭34月0.9793%；较早启用26月负0.7380%，关闭33月0.1017%。保存280条含两费用逐月记录、8组；两段终点开盘月份单列但正式账户保留。启停输入连续参考与分组输出评价账户用途不同，不拼接成新账户，不按结果反转开关或重复分组。",
        "下一方向检查原学习模型的训练路径覆盖，而非继续叠加进入筛选。已限定检索每个合格进入日的反事实自然退出训练，未找到同源完成研究；第33轮只在进入前问旧退出模型，第36轮只学原已发生完整交易。拟用现有日线，在每个原条件允许的进入日独立模拟一次原自然退出路径，拓展持仓状态覆盖；仍用一个简单线性模型与旧八因子，不新增外部资料。先核对实际成交、分红、终点删失及重叠事件权重再固定一个版本，不能把重叠假想路径计成独立新样本，也不能只改原34周期的标签营救。当前仅完成方向与重复研究检索，未登记、未生成新路径、未训练或回测。",
        "R32仍为比较基线，R46主评价0.944391/压力0.879003、较早0.604421/0.572113只作局部线索；完整1.2和稳定超额未实现。55/56回踩及收盘位置失败已交付，其他停止规则保留索引和旧说明。不重新展开全部历史。",
        "EPS、报告日期、股数、估值、财报、公募等慢来源继续暂停；仅510300及现金，其他ETF等原回复、不重复问。中文规则和普通结果，不做GPT数值包、无安全审计、无子任务或券商操作。",
    ]
    text += ["## 研究进度与下一步", "",
             "本次完成两个新设置。每段22条评价记录；主评价4个新候选账户和18个保存对照，较早4个新候选账户、2个新补旧日历控制和16个保存对照。另外生成1个连续日历参考账户，不计为新策略或新评价记录。没有新模型拟合或新行情下载。", "",
             "16项必要测试通过，其中第57轮5项新组合测试及5项旧日历账户与时钟测试，第58轮6项月度启停测试。第57轮合成测试的权重初始类型在冻结前由整数改为浮点数，正式逻辑未改变；通过后才冻结和运行。16项账户价格股息费用差额核对通过。", "",
             "累计完成58轮、331个不同配置或范围、343个已评价来源版本、1004个主评价记录；登记来源版本348个，含5个旧未运行绑定。上述数量不代表独立有效策略数量，也不证明目标已经完成。", "",
             "下一步检查原学习模型是否只覆盖了少量实际进入路径。拟用同一份日线，对每个当时满足原进入条件的日期模拟原自然退出，检查更完整的持仓状态能否改善同一个简单线性模型。必须处理重叠路径、成熟时间、未成交、分红和终点删失，不能把重复行情当成独立数据。目前只完成方法检索，未登记或运行，不预报其结果。", "",
             "夏普1.2和稳定超额仍为目标，持续研究保持开启；EPS和其他慢来源补齐继续暂停。", "",
             "## 普通结果文件", ""]
    for p in copies:
        text.append(f"- [{p.name}]({p.name})")
    for n, slug, *_ in STUDIES:
        text.append(f"- 第{n}轮[完整原始结果](<{(folder(slug)/'result.json').as_posix()}>)，同目录保存全部账户、状态和交易。")
    document = OUT / "日历组合与月度启停_全部因子规则和历史表现.md"
    content = "\n".join(text) + "\n"
    require("```" not in content, "中文规则被代码替代")
    document.write_text(content, encoding="utf-8")
    for match in re.finditer(r"\]\(([^)]+)\)", content):
        path = match.group(1).strip("<>")
        if not path.startswith("http"):
            require((OUT / path).is_file(), "交付链接不存在：" + path)
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS57_58_COMPLETE_CALENDAR_BLEND_AND_ACTIVATION_FAILED", latest_completed_round=records[-1],
                 running_studies=[], goal_achieved=False, process_state_note="第57至58轮和保存月份分组完成，进程正常退出；下一方向未登记未运行。",
                 count_warning="累计58轮、331不同配置或范围、343已评价来源版本、1004主评价记录；登记348含5旧未运行绑定。较早、参考另计。",
                 next_work=next_work, checks="16项必要测试及16项账户差额核对完成；保存280条月份费用记录与8个完整月分组，无新增诊断账户。")
    index["latest_saved_calendar_activation_diagnostic"] = {"study": r58["study_id"], "status": "SAVED_FIXED_CONTROL_MONTH_GROUPING_COMPLETE",
        "source": "reports/research/510300_calendar_monthly_activation_v1/saved_diagnostic_receipt.json", "new_accounts": 0, "new_models": 0}
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [57, 58]]
    delivery = {"created_at": now(), "type": "CALENDAR_BLEND_AND_ACTIVATION_ROUNDS57_58_CHINESE_RESULTS", "rounds": [57, 58],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 58 and index["evaluated_configurations_in_this_resumption"] == 331 and
            index["evaluation_accounts_in_this_resumption"] == 1004 and index["evaluated_candidate_source_runs_including_corrected_replays"] == 343 and
            index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 348, "累计计数不一致")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_CALENDAR_ACTIVATION_DIAGNOSTIC", "rounds": [57, 58],
               "new_configurations": 2, "main_records": 22, "new_main_accounts": 4, "reused_main_accounts": 18,
               "earlier_records": 22, "new_earlier_candidate_accounts": 4, "new_earlier_control_accounts": 2, "reused_earlier_accounts": 16,
               "new_reference_accounts": 1, "new_model_fits": 0, "necessary_tests_passed": 16, "account_difference_checks": len(deltas),
               "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "文档字符": len(content), "普通文件": len(receipt["files"]),
                      "累计轮次": 58, "累计主评价": 1004, "账户差额核对": len(deltas), "参考账户天数": r58['reference_days']}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
