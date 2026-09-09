"""交付状态筛选失败与压缩后突破局部改善的完整中文证据。"""
import json
import re
from pathlib import Path

import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.deliver_simple_strategy_rounds34_36_20260907 import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300状态选择与突破确认_第53至54轮_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
STUDIES = [
    (53, "variance_ratio_entry_router", "方差比选择突破或反弹进入", "VARIANCE_RATIO_ROUTER", "UNGATED_TWO_MODE",
     "COMPLETED_VARIANCE_ROUTING_NO_IMPROVEMENT",
     "方差状态筛选两段、两档费用均低于相同规则不加筛选的对照，停止这项方法及窗口、合计尺度、阈值和方向调整。"),
    (54, "compression_confirmed_entry", "压缩后等待实际突破进入", "AFTER_COMPRESSION_BREAKOUT", "ORDINARY_BREAKOUT",
     "COMPLETED_LOCAL_IMPROVEMENT_VS_ORDINARY_BREAKOUT_TARGET_NOT_MET",
     "相对普通突破，两段、两档费用的夏普、复合收益和回撤均改善；保留为弱基线的局部改善。它仍低于原学习退出，两段年化也低于买入持有，尚未达到1.2或证明稳定超额。"),
]


def folder(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def metric(r, key, period="evaluation", cost="BASE"):
    return next(m for m in r["all_metrics" if period == "evaluation" else "earlier_diagnostics"] if m["model"] == key and m["cost"] == cost)


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 53)), set(range(1, 55))], "有其他新轮次，停止覆盖当前研究索引")
    results = {n: json.loads((folder(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, *_ in STUDIES}
    require(not any(r["historical_point_target_met"] for r in results.values()), "出现达到1.2的结果，不能使用未完成说明")
    OUT.mkdir(parents=True, exist_ok=True)
    records, copies, differences, cycle_checks, fixed_groups, fixed_cycles = [], [], [], [], [], []
    for n, slug, title, primary, control, status, decision in STUDIES:
        result = results[n]
        write_json(folder(slug) / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
                   "goal_achieved": False, "position_impact": 0, "independent_validation": "NOT_ESTABLISHED"})
        records.append({"round": n, "study": result["study_id"], "title": title, "status": status,
                        "result": str((folder(slug) / "result.json").relative_to(ROOT)),
                        **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
                                                 "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
                        "evaluated_candidate_source_runs": result["candidate_configurations"], "primary_base": metric(result, primary),
                        "primary_stress": metric(result, primary, cost="STRESS"), "post_selected_best_base": result["post_selected_best_base"]})
        for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "result.json"]:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((folder(slug) / filename).read_bytes())
            copies.append(dest)
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                p = folder(slug) / period / cost
                current = pd.read_parquet(p / f"{primary}_ledger.parquet")
                prior = pd.read_parquet(p / f"{control}_ledger.parquet")
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(prior.date)), "直接对照的完整日期不同")
                d = {"round": n, "period": period, "cost": cost, "new_model": primary, "control": control,
                     "final_equity_difference": float(current.equity.iloc[-1] - prior.equity.iloc[-1]),
                     "price_pnl_difference": float(current.price_pnl.sum() - prior.price_pnl.sum()),
                     "dividend_difference": float(current.dividend_recognized.sum() - prior.dividend_recognized.sum()),
                     "extra_commission_and_slippage": float(current.commission.sum() + current.slippage_cost.sum() - prior.commission.sum() - prior.slippage_cost.sum())}
                require(abs(d["final_equity_difference"] - d["price_pnl_difference"] - d["dividend_difference"] + d["extra_commission_and_slippage"]) < 1e-6,
                        "完整账户差额无法用价格、股息和费用核对")
                differences.append(d)
                keys = [primary, control] if n == 53 or period == "earlier_diagnostic" else [primary]
                for key in keys:
                    c = pd.read_csv(p / f"{key}_cycles.csv")
                    l = pd.read_parquet(p / f"{key}_ledger.parquet")
                    residual = float(l.equity.iloc[-1] - 200000 - c.net_profit_cny.sum())
                    require(abs(residual) < 1e-6, "周期损益不能核对到账户，不能宣称模式贡献完整")
                    cycle_checks.append({"round": n, "period": period, "cost": cost, "model": key, "cycles": len(c), "cycle_profit_account_residual": residual})
                    dest = OUT / f"第{n}轮_{period}_{cost}_{key}_全部持仓周期.csv"
                    dest.write_bytes((p / f"{key}_cycles.csv").read_bytes())
                    copies.append(dest)
                if n == 53:
                    f = pd.read_parquet(folder(slug) / "all_daily_factors.parquet").set_index("date")
                    c = pd.read_csv(p / f"{control}_cycles.csv", parse_dates=["entry_origin"])
                    c["entry_variance_ratio"] = c.entry_origin.map(f.variance_ratio)
                    c["condition_would_admit"] = (c["mode"].eq(1) & c.entry_variance_ratio.gt(1)) | (c["mode"].eq(2) & c.entry_variance_ratio.lt(1))
                    fixed_cycles.extend({"period": period, "cost": cost, **row} for row in c.to_dict("records"))
                    for (mode, admitted), g in c.groupby(["mode", "condition_would_admit"]):
                        fixed_groups.append({"period": period, "cost": cost, "entry_mode": int(mode), "condition_would_admit": bool(admitted),
                                             "cycles": len(g), "positive_cycles": int(g.net_profit_cny.gt(0).sum()), "fixed_reference_cycle_profit_cny": float(g.net_profit_cny.sum())})
    for filename, rows in [("两轮相对直接对照_价格股息费用差额.csv", differences), ("全部周期损益到账户核对.csv", cycle_checks),
                           ("第53轮_无筛选原路径按方差条件逐次分组.csv", fixed_cycles), ("第53轮_无筛选原路径分组汇总.csv", fixed_groups)]:
        dest = OUT / filename
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
        copies.append(dest)
    for source, name in [(folder("variance_ratio_entry_router") / "state_counts.csv", "第53轮_每日状态与进入机会统计.csv"),
                         (folder("variance_ratio_entry_router") / "mode_statistics.csv", "第53轮_实际两种模式贡献.csv"),
                         (folder("compression_confirmed_entry") / "preparation_statistics.csv", "第54轮_压缩事件与准备窗口统计.csv"),
                         (folder("compression_confirmed_entry") / "cycle_statistics.csv", "第54轮_实际进入与不同准备事件.csv"),
                         (ROOT / "reports/discovery/volatility_compression_change_point_v1.md", "旧压缩事件研究_已有失败结论.md")]:
        dest = OUT / name
        dest.write_bytes(source.read_bytes())
        copies.append(dest)
    r53, r54 = results[53], results[54]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            a, b = metric(r54, "AFTER_COMPRESSION_BREAKOUT", period, cost), metric(r54, "ORDINARY_BREAKOUT", period, cost)
            require(a["net_sharpe"] > b["net_sharpe"] and a["annualized_return"] > b["annualized_return"] and a["max_drawdown"] > b["max_drawdown"],
                    "第54轮局部改善文字不符合实际三项指标")
    text = ["# 510300：选择市场状态与等待突破确认的结果", "", "第53至54轮，2026年9月7日。使用现有数据，完整中文因子、进出场规则及历史表现。", "",
            "## 管理层先看结论", "",
            "完整账户净夏普1.2仍未实现。本次三种设置中，用方差状态在突破和反弹之间选择没有改善；压缩之后等实际向上突破，比普通突破有两段共同改善，但总体仍弱于原学习退出基线。两种结论分别保留，不能把局部改善当成已实现高夏普。", "",
            "|方案|主评价基础夏普|主评价压力夏普|主评价基础复合年化|主评价基础最大回撤幅度|较早基础夏普|较早压力夏普|",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for r, key, name in [(r53, "VARIANCE_RATIO_ROUTER", "第53轮：按方差状态选模式"), (r53, "UNGATED_TWO_MODE", "第53轮直接对照：同规则不加状态"),
                         (r54, "AFTER_COMPRESSION_BREAKOUT", "第54轮：压缩后等实际突破"), (r54, "ORDINARY_BREAKOUT", "第54轮直接对照：普通突破"),
                         (r54, "REARM_RIDGE", "原学习退出比较基线"), (r54, "BUY_HOLD", "买入持有")]:
        b, s = metric(r, key), metric(r, key, cost="STRESS")
        eb, es = metric(r, key, "earlier_diagnostic"), metric(r, key, "earlier_diagnostic", "STRESS")
        text.append(f"|{name}|{b['net_sharpe']:.3f}|{s['net_sharpe']:.3f}|{b['annualized_return']:.2%}|{-b['max_drawdown']:.2%}|{eb['net_sharpe']:.3f}|{es['net_sharpe']:.3f}|")
    text += ["", "主评价2020年1月2日至2026年8月14日终点开盘，1604个账户日；较早诊断2015年1月5日至2019年12月31日终点开盘，1219日，两段各20万元。全部空仓日、交易费用和分红均计入，现金及无风险收益假设零，242日年化。主评价和较早历史都已被多轮观察，不能称为新的独立样本。", "",
             "## 为什么状态筛选失败", "",
             "主评价1604个决策日中，方差条件把750日分为较偏同向累积、854日分为较偏涨跌交替。它允许80个突破信号日、22个反弹信号日，排除了另外114个突破信号日和18个反弹信号日。这些都是可能进入的日期，不等于实际交易，更不等于独立事件。", "",
             "实际新账户完成39个主评价周期、78笔买卖，平均股票仓位29.45%；不加条件的对照有61个周期、122笔买卖，平均仓位47.44%。减少交易后费用下降，但年化收益从3.63%降至0.45%，最大回撤反而从26.90%扩大到30.53%。较早新账户22个周期，基础复合年化为−0.18%，也弱于对照。", "",
             "为解释筛选方向，固定不加条件对照已经发生的全部进入、份额和退出，再按进入时的方差条件分组。主评价原36次突破中，条件会排除的23次合计盈利71,330.83元，会允许的13次合计亏损53,519.28元；较早原26次突破中，会排除的17次合计盈利53,250.53元，会允许的9次合计亏损19,817.23元。", "",
             "这说明该条件在这批历史中没有挑出更有收益的突破机会，可能出现等到涨跌已经同向累积才追入的滞后。后半句是机制假设，不能由分组本身证明。分组是看过结果后的固定路径诊断；删去某些交易后，现金和后续机会会变化，不能把这几项金额直接相减当成可交易新策略，也不据此反向挑选条件。", "",
             "以下保留基础与压力费用的全部固定路径分组，不能只展示突破中的有利组：", "",
             "|时期与费用|进入模式|原条件会允许|原路径周期数|原固定周期净损益（元）|", "|---|---|---|---:|---:|"]
    for g in fixed_groups:
        label = ("主评价" if g["period"] == "evaluation" else "较早") + "／" + ("基础" if g["cost"] == "BASE" else "压力")
        text.append(f"|{label}|{'突破' if g['entry_mode']==1 else '反弹'}|{'是' if g['condition_would_admit'] else '否'}|{g['cycles']}|{g['fixed_reference_cycle_profit_cny']:.2f}|")
    text += ["", "实际新账户的突破贡献不等于上述会允许组。新主评价实际22次突破合计亏5,892.42元，17次反弹合计赚11,988.82元；较早16次突破赚698.86元，6次反弹亏2,523.94元。所有周期损益均核对至完整账户。这种差别来自资金释放、后续机会和份额路径，正是不能把事后分组直接当账户的原因。", "",
             "## 压缩后等突破，改善在哪里", "",
             "旧研究未支持“压缩本身直接预测方向”的解释。本轮保留旧波动周期、历史分位和事件去重条件，但在统一含分红数据上重构准备事件；先观察压缩，再等后来实际出现二十日新高才买入。旧报告的40个成熟事件没有直接当成本轮清单，也没有使用其中保存的未来突破或收益标签。", "",
             "新主评价有25个当期压缩事件，准备状态覆盖505个决策日，普通突破194日中60日同时满足准备条件。准备状态可由期前事件延续进入评价期，因此准备日数不必等于当期事件数乘二十。最终完成16次持仓，来自15个不同准备事件，5次盈利；较早有13个当期压缩事件，最终8次持仓来自8个准备事件，基础费用7次盈利、压力费用6次盈利。", "",
             "相对普通突破，主评价基础夏普从0.135升至0.273，年化从0.89%升至1.92%，最大回撤从23.41%降至19.49%；较早基础夏普从0.283升至0.687，年化从2.91%升至3.06%，回撤从27.01%降至5.01%。压力费用下也有同方向改善，符合本轮的局部改善判断。", "",
             "较早基础费用的终点资产只比普通突破多1,707.43元：价格损益多266.30元，股息少4,243.60元，佣金及滑点节省5,684.73元。较早收益提升主要依赖交易减少后的费用节省，不能把它全称为因子预测增益。主评价终点多14,706.12元，包括价格损益多13,957.70元、股息少5,176.70元、费用节省5,925.12元。", "",
             "它仍不是高夏普方案：主评价只有0.273，低于买入持有0.288和原学习退出0.705；较早0.687也低于原学习退出0.748。两段复合年化都低于买入持有。保留的是“等待准备后的价格确认可能有价值”这个有限结果，不是把弱普通突破的改善宣称为稳定超额。", "",
             "## 新旧账户差额", "", "全部差额为新候选减本轮直接对照。额外费用为负表示节省，八项均能与实际账户核对。", "",
             "|轮次|时期与费用|终点资产差额（元）|价格损益差（元）|股息差（元）|额外佣金及滑点（元）|", "|---|---|---:|---:|---:|---:|"]
    for d in differences:
        label = ("主评价" if d["period"] == "evaluation" else "较早") + "／" + ("基础" if d["cost"] == "BASE" else "压力")
        text.append(f"|{d['round']}|{label}|{d['final_equity_difference']:.2f}|{d['price_pnl_difference']:.2f}|{d['dividend_difference']:.2f}|{d['extra_commission_and_slippage']:.2f}|")
    text.append("")
    for n, slug, title, primary, control, status, decision in STUDIES:
        text.extend([f"## 第{n}轮全部因子和进入、持有、退出、重新进入规则", ""])
        lines = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8").splitlines()[1:]
        text.extend(["#" + line if line.startswith("##") else line for line in lines])
        text.extend(["", f"### 第{n}轮全部主评价", ""] + table(results[n]["all_metrics"]))
        text.extend([f"### 第{n}轮全部较早诊断", ""] + table(results[n]["earlier_diagnostics"]))
        text.extend(["本轮验收：" + decision, ""])
    next_work = [
        "第53轮方差状态选择失败：主评价基础0.096135、压力0.021597，较早0.050159、压力负0.001414，全部低于同规则无筛选对照0.334122/0.238243及0.172659/0.099782。停止窗口、尺度、阈值和方向调整。",
        "第53轮事后固定无筛选路径：主评价被条件排除的23个突破原赚71330.83元，允许13个原亏53519.28元；较早排除17个赚53250.53，允许9个亏19817.23。不能直接反向选择后当作新账户。实际新交易路径及份额不同，已保存全部模式和分组，不重复诊断。",
        "第54轮压缩后等突破局部改善：主评价0.273047、压力0.231954，较早0.686811、压力0.634285，均高于普通突破0.134545/0.068306及0.282672/0.225096；收益与回撤亦同向改善。主评价16周期来自15准备事件、较早8周期来自8事件。保留局部机制，但主评价低于买入持有和原学习退出，两段年化低于买入持有，不算稳定超额或1.2。",
        "旧压缩事件研究已有40个二十日成熟自适应事件、55%振幅扩张、不支持直接方向。第54轮保留定量条件在当前含分红数据重算，再等实际突破；不能再把旧事件研究当作未做过，不能直接用未来突破标签。",
        "下一项优先检验突破后回踩原突破位、收盘守住再进入是否能减少假突破。已对research/docs/config中的retest、回踩突破及突破回踩作限定检索，未发现510300同源完整策略；R23均线收复与R26跌破低点后收复不是突破高点回踩。继续先核对定义，再固定一个候选与普通突破直接对照，不绑定第54轮去追加参数。目前下一项未登记未运行。",
        "原第32轮仍作基线，第46轮0.944主评价、0.604较早仍只作局部线索。停止围绕原34参考周期改退出标签，EPS及所有慢来源补齐继续暂停；只510300和现金，其他ETF问题待原回复、不重复问，不做GPT数值包、不创建子任务。",
    ]
    text += ["## 完成情况和下一步", "",
             "本次两个研究、三个设置，每时期18个记录。主评价6个新候选账户、12个复用对照；较早6个新候选账户、2个新生成的旧普通突破对照、10个复用对照。没有新模型拟合、新参考账户或新行情下载。", "",
             "十项必要测试通过：第53轮六项验证方差单位、漂移与尺度、序列性质、缺失、因果窗口及模式选择；第54轮四项验证准备窗口、事件去重、未来隔离、对数波动与历史分位。第53轮曾修正一个合成反弹测试误设为新高的问题，正式数据、策略和参数没有改变；原登记与测试完整保留后才开始账户运行。八项账户差额和十四条周期到账户核对通过，没有进行安全审计。", "",
             "累计完成54轮、326个不同配置或范围、338个已评价来源版本、962个主评价记录；登记来源版本343个，含5个旧未运行绑定。复用对照、较早诊断和不同费用不能算新的独立策略。", "",
             "下一步准备检查“突破之后回踩原突破位，收盘守住再进入”的完整规则，用普通突破作直接对照，重点检验假突破造成的损失。当前限定检索没有找到同源510300完整策略，但尚未登记或运行新候选，也不把第53轮失败条件直接反向交易。第54轮保留作局部线索，继续追求完整净夏普1.2及稳定超额。EPS、研报日期、股数、估值、财报、公募等慢来源补齐继续暂停，不做GPT数值包。", "",
             "## 普通交付文件", ""]
    for p in copies:
        text.append(f"- [{p.name}]({p.name})")
    for n, slug, *_ in STUDIES:
        text.append(f"- 第{n}轮[原始结果索引](<{(folder(slug)/'result.json').as_posix()}>)；[全部逐日因子](<{(folder(slug)/'all_daily_factors.parquet').as_posix()}>)。")
    document = OUT / "状态选择与突破确认_全部因子规则和历史表现.md"
    content = "\n".join(text) + "\n"
    require("```" not in content, "策略规则被代码块替代")
    document.write_text(content, encoding="utf-8")
    for match in re.finditer(r"\]\(([^)]+)\)", content):
        target = match.group(1).strip("<>")
        if not target.startswith("http"):
            require((OUT / target).is_file(), "交付链接不存在：" + target)
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                      "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[k] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS53_54_COMPLETE_LOCAL_COMPRESSION_CONFIRMATION_IMPROVEMENT", latest_completed_round=records[-1],
                 running_studies=[], goal_achieved=False, process_state_note="第53至54轮及保存路径分组完成，进程全部正常退出；下一项未登记未运行。",
                 count_warning="累计54轮、326不同配置或范围、338已评价来源版本、962主评价记录；登记343含5旧未运行绑定。较早、参考另计。",
                 checks="10项必要测试通过；8项账户差额和14个周期到账户核对完成。第53轮测试合成价格修正在正式收益读取前完成。",
                 next_work=next_work)
    index["additional_filtered_breakout_candidate"] = {"study": r54["study_id"], "model": "AFTER_COMPRESSION_BREAKOUT",
        "status": "LOCAL_IMPROVEMENT_VS_WEAK_BREAKOUT_CONTROL_NOT_HIGH_SHARPE",
        "source_result": str((folder("compression_confirmed_entry") / "result.json").relative_to(ROOT)),
        "base_main_sharpe": metric(r54, "AFTER_COMPRESSION_BREAKOUT")["net_sharpe"],
        "stress_main_sharpe": metric(r54, "AFTER_COMPRESSION_BREAKOUT", cost="STRESS")["net_sharpe"],
        "base_earlier_sharpe": metric(r54, "AFTER_COMPRESSION_BREAKOUT", "earlier_diagnostic")["net_sharpe"],
        "stress_earlier_sharpe": metric(r54, "AFTER_COMPRESSION_BREAKOUT", "earlier_diagnostic", "STRESS")["net_sharpe"],
        "goal_achieved": False}
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [53, 54]]
    delivery = {"created_at": now(), "type": "STATE_ROUTING_COMPRESSION_CONFIRMATION_ROUNDS53_54_CHINESE_RESULTS", "rounds": [53, 54],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 54 and index["evaluation_accounts_in_this_resumption"] == 962 and
            index["evaluated_configurations_in_this_resumption"] == 326 and index["evaluated_candidate_source_runs_including_corrected_replays"] == 338 and
            index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 343, "累计计数不一致")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_ACCOUNTS_AND_FIXED_PATH_DIAGNOSTIC", "rounds": [53, 54],
               "new_configurations": 3, "main_records": 18, "new_main_accounts": 6, "reused_main_accounts": 12,
               "earlier_records": 18, "new_earlier_candidate_accounts": 6, "new_earlier_control_accounts": 2, "reused_earlier_accounts": 10,
               "new_model_fits": 0, "new_reference_accounts": 0, "necessary_tests_passed": 10, "account_difference_checks": len(differences),
               "cycle_account_checks": len(cycle_checks), "fixed_reference_group_rows": len(fixed_groups),
               "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "文档字符": len(content), "普通文件": len(receipt["files"]),
                      "累计轮次": 54, "累计主评价": 962, "局部改善": index["additional_filtered_breakout_candidate"], "账户差额": differences}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
