"""交付分批退出和进入价格上限两轮完整中文结果及保存路径归因。"""
import json
import re
from pathlib import Path
import pandas as pd
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.deliver_simple_strategy_rounds34_36_20260907 import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300分批退出与进入价格_第49至50轮_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
DIAG = ROOT / "reports/research/510300_partial_exit_saved_diagnostic_v1"
STUDIES = [
    (49, "partial_learned_exit", "学习退出先减半，余下按原规则退出", "LEARNED_HALF_THEN_NATURAL",
     "COMPLETED_PARTIAL_EXIT_NO_IMPROVEMENT", "两段夏普与收益均弱于原全部退出，主评价回撤也扩大；剩余份额继续持有多数带来负增量，不采用，不搜索减仓比例或次数。"),
    (50, "volatility_capped_entry", "进入价格不得超过信号收盘加一日波动幅度", "VOL_ONE_SIGMA_CAP",
     "COMPLETED_ENTRY_CAP_MISSED_MAJOR_GAIN_NO_IMPROVEMENT", "主评价漏掉2024年9月25日原盈利持仓，收益与夏普下降；较早账户完全不变，不采用，不扫描价格上限倍数、波动窗口或订单有效期。"),
]


def folder(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 49)), set(range(1, 51))], "研究索引出现其他新轮次，停止覆盖")
    results = {n: json.loads((folder(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, *_ in STUDIES}
    diag = json.loads((DIAG / "result.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    records, copies, account_differences, entry_changes, cap_attempts, cap_checks = [], [], [], [], [], []
    for n, slug, title, primary, status, decision in STUDIES:
        result = results[n]
        record = {"round": n, "study": result["study_id"], "title": title, "status": status,
                  "result": f"reports/research/510300_{slug}_v1/result.json",
                  **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
                                           "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
                  "evaluated_candidate_source_runs": result["candidate_configurations"],
                  "primary_base": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "BASE"),
                  "primary_stress": next(m for m in result["all_metrics"] if m["model"] == primary and m["cost"] == "STRESS"),
                  "post_selected_best_base": result["post_selected_best_base"]}
        records.append(record)
        write_json(folder(slug) / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
                   "goal_achieved": False, "position_impact": 0})
        for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "result.json"]:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((folder(slug) / filename).read_bytes())
            copies.append(dest)
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                current = pd.read_parquet(folder(slug) / period / cost / f"{primary}_ledger.parquet")
                old = pd.read_parquet(ROOT / "reports/research/510300_rearmed_session_exit_v1" / period / cost / "REARM_RIDGE_ledger.parquet")
                require(pd.DatetimeIndex(current.date).equals(pd.DatetimeIndex(old.date)), "原候选与新账户日历不同")
                diff = {"round": n, "period": period, "cost": cost,
                        "final_equity_difference": float(current.equity.iloc[-1] - old.equity.iloc[-1]),
                        "price_pnl_difference": float(current.price_pnl.sum() - old.price_pnl.sum()),
                        "dividend_difference": float(current.dividend_recognized.sum() - old.dividend_recognized.sum()),
                        "extra_commission_and_slippage": float(current.commission.sum() + current.slippage_cost.sum() - old.commission.sum() - old.slippage_cost.sum())}
                require(abs(diff["final_equity_difference"] - diff["price_pnl_difference"] - diff["dividend_difference"] + diff["extra_commission_and_slippage"]) < 1e-6,
                        "保存账户差额无法由价格、股息与费用核对")
                account_differences.append(diff)
                cycles_path = folder(slug) / period / cost / f"{primary}_cycles.csv"
                dest = OUT / f"第{n}轮_{period}_{cost}_全部持仓周期.csv"
                dest.write_bytes(cycles_path.read_bytes())
                copies.append(dest)
                if n == 50:
                    rejected = current[current.entry_price_cap_rejected]
                    fields = ["date", "open", "entry_price_ceiling_after_ex_adjustment", "requested_quantity", "filled_quantity", "status"]
                    cap_attempts.extend({"period": period, "cost": cost, **r} for r in rejected[fields].to_dict("records"))
                    if len(rejected):
                        cut = rejected.date.min()
                        pd.testing.assert_frame_equal(current.loc[current.date < cut, old.columns].reset_index(drop=True), old.loc[old.date < cut].reset_index(drop=True))
                        checked = int((current.date < cut).sum())
                    else:
                        pd.testing.assert_frame_equal(current[old.columns], old)
                        checked = len(current)
                    cap_checks.append({"period": period, "cost": cost, "checked_rows": checked,
                                       "all_old_ledger_fields_equal_until_first_cap_effect": True,
                                       "whole_period_identical": len(rejected) == 0})
                    c = pd.read_csv(cycles_path)
                    oc = pd.read_csv(ROOT / "reports/research/510300_rearmed_session_exit_v1" / period / cost / "REARM_RIDGE_cycles.csv")
                    for label, source, other in [("原进入日期在新账户中未出现", oc, c), ("新进入日期在原账户中未出现", c, oc)]:
                        for row in source[~source.entry_date.isin(other.entry_date)].to_dict("records"):
                            entry_changes.append({"period": period, "cost": cost, "difference": label,
                                                  **{k: row[k] for k in ["entry_date", "exit_date", "entry_quantity", "net_profit_cny"]}})
    extra = [(folder("partial_learned_exit") / "partial_exit_statistics.csv", "第49轮_全部减仓和模型调用统计.csv"),
             (folder("volatility_capped_entry") / "entry_cap_statistics.csv", "第50轮_全部进入请求统计.csv"),
             (DIAG / "per_partial_cycle_increment.csv", "第49轮_每次剩余份额继续持有差额.csv"),
             (DIAG / "entry_date_differences.csv", "第49轮_所有原进入与新进入日期差异.csv"),
             (DIAG / "prefix_checks.csv", "第49轮_首次变化前经济路径核对.csv"),
             (DIAG / "result.json", "第49轮_保存路径诊断.json"),
             (ROOT / "deliverables/510300学习目标与互补组合_第44至46轮_20260907/沿用原模型_每月实际中文规则.md", "两轮沿用的原模型_每月实际中文规则.md")]
    for original, name in extra:
        dest = OUT / name
        dest.write_bytes(original.read_bytes())
        copies.append(dest)
    for name, rows in [("两轮相对原候选_全部价格股息费用差额.csv", account_differences), ("第50轮_全部价格上限未成交请求.csv", cap_attempts),
                       ("第50轮_所有原进入与新进入日期差异.csv", entry_changes), ("第50轮_实际生效前及较早全期账户核对.csv", cap_checks)]:
        dest = OUT / name
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
        copies.append(dest)
    feasibility = []
    for period in ["evaluation", "earlier_diagnostic"]:
        path = ROOT / "reports/research/510300_rearmed_session_exit_v1" / period / "BASE/REARM_RIDGE_decisions.parquet"
        d = pd.read_parquet(path)
        eligible = d[d.cycle_return.ge(.06) & d.requested_quantity.eq(0) & d.entry_mode.eq(1)]
        feasibility.append({"period": period, "original_saved_source_sha256": digest(path), "eligible_closes": len(eligible),
                            "different_original_cycles": int(eligible.learning_cycle_id.nunique()),
                            "first_dates_by_cycle": {str(k): str(v) for k, v in eligible.groupby("learning_cycle_id").origin.min().items()}})
    feasibility_path = ROOT / "reports/research/510300_entry_pyramiding_scope_check_20260907.json"
    write_json(feasibility_path, {"checked_at": now(), "status": "DESCRIPTIVE_EXISTING_PATH_CHECK_NO_NEW_STRATEGY", "observations": feasibility,
               "definition": "原满仓路径已浮盈至少6%、未请求卖出且当前处于原模式的收盘；6%来自原固定止损幅度，未扫描阈值。",
               "decision": "主评价只涉及三个原周期，当前不另建确认加仓引擎；这不是对尚未实现的半仓加仓账户触发次数的精确证明。",
               "new_strategy_configurations": 0, "new_accounts": 0, "new_model_fits": 0})
    dest = OUT / "未启动加仓引擎_原路径机会观察.json"
    dest.write_bytes(feasibility_path.read_bytes())
    copies.append(dest)
    absent = pd.read_csv(DIAG / "entry_date_differences.csv")
    missing_main = absent[absent.period.eq("evaluation") & absent.cost.eq("BASE")]
    text = ["# 510300：分批退出与进入价格约束的实际结果", "", "第49至50轮，2026年9月7日。全部规则用中文说明，附两档费用、完整历史和逐次差额。", "",
            "## 管理层先看结论", "",
            "这两轮仍未达到完整账户扣费净夏普1.2。把原学习退出改成只卖一半，或在原买入请求上增加价格上限，都没有改善原候选。前者留下的份额多数继续走弱，后者漏掉了一次主要盈利行情。停止这两项改动及相邻参数搜索，保留原进入和全部退出作为比较基线。", "",
            "|方案|主评价基础夏普|主评价压力夏普|基础复合年化收益|基础最大回撤幅度|较早基础夏普|较早压力夏普|",
            "|---|---:|---:|---:|---:|---:|---:|",
            "|原进入与学习信号全部退出|0.705|0.641|5.38%|10.59%|0.748|0.725|",
            "|第49轮：学习退出先减半|0.622|0.588|5.11%|11.44%|0.703|0.681|",
            "|第50轮：增加一日波动幅度买价上限|0.482|0.414|3.23%|10.59%|0.748|0.725|",
            "|原急跌与学习信号各半组合，仅局部线索|0.944|0.879|4.01%|5.39%|0.604|0.572|", "",
            "主评价为2020年1月2日至2026年8月14日开盘，1604个账户日；较早诊断为2015年1月5日至2019年12月31日开盘，1219日，各从20万元开始。所有夏普包含空仓日和交易费用，现金与无风险收益零，242日年化。较早0.748没有改善，而是第50轮两档费用的全账户原字段与原候选逐日完全一致。", "",
            "原0.944组合的较早夏普仅0.604，急跌子策略主评价只有三个不同持仓周期，不能将其当成已验证的稳定高夏普策略。", "",
            "## 为什么分批退出失败", "",
            "主评价实际14个持仓周期全部发生一次减半，留下部分累计产生391个继续持有的收盘判断；完整买卖42笔。原全部退出有24个持仓周期、48笔买卖。主评价平均股票仓位从原15.68%升至23.08%，持仓更久并未带来更高收益。较早9个周期中只有2个真正发生减半，另有199次检查没有可用模型，不能把全部较早历史都描述成学习退出证据。", "",
            "固定每个新周期已经发生的进入日期、实际份额和退出终点，逐次比较“实际减半后继续持有”与“减半当天同时卖完余下份额”。主评价14次里只有5次继续持有更好，9次更差；基础费用下合计差额为−22,925.96元，其中成交价差−29,680.80元、新增股息权益6,748.90元、佣金节省5.94元。较早两次一正一负，合计−5,531.59元。全部32条两档费用逐次结果附后，不只展示坏例子。", "",
            "这项逐次比较是看过结果后的保存路径诊断。它固定实际份额，未模拟提前释放现金后怎样重新进入，因此不是另一套可交易账户，不能直接把−22,925.96元当作整套策略损失。", "",
            f"新账户14个主评价进入日期都是原24个日期中的一部分，另10个原日期未出现。原这10次持仓中，{int(missing_main.net_profit_cny.lt(0).sum())}次亏损、{int(missing_main.net_profit_cny.gt(0).sum())}次盈利，原路径净损益合计{missing_main.net_profit_cny.sum():,.2f}元。继续持仓既阻止了部分盈利进入，也避开了部分亏损进入，不能把缺少10次进入全部当作错失盈利。", "",
            "把上述变化全部放回真实整手账户后，主评价基础费用终点仅比原少4,919.39元：价格损益少9,799.60元，股息多836.10元，费用节省4,044.11元。较早终点少5,887.44元。原退出过早不是这组证据支持的统一失败原因，不应据此放宽学习退出。", "",
            "## 为什么买价上限失败", "",
            "原信号产生后，价格上限在下一开盘只拦住三次请求，全部发生在2024年9月至10月。较早历史没有任何一次被拦住。", "",
            "|被拦住的开盘日|开盘参考价（元）|当时已定买价上限（元）|",
            "|---|---:|---:|",
            "|2024年9月25日|3.489|3.469|",
            "|2024年9月30日|4.007|3.949|",
            "|2024年10月8日|4.656|4.352|", "",
            "是否超限比较的是开盘参考价加滑点并按价格跳动处理后的模拟买价，上表开盘价尚未加滑点。基础与压力费用的三个拒绝日期相同。", "",
            "原2024年9月25日至30日这一周期，基础费用净赚33,605.62元、压力费用净赚32,485.33元；新账户没有该次进入，也没有其他新进入日期。主评价周期从24个降为23个，基础年化从5.38%降为3.23%，低于买入持有3.63%。后续现金与整手数量也受影响，完整账户终点差额不能简单等同于这一笔旧利润。", "",
            "由此只能说，这条基于一日波动幅度的追价限制在本次历史中筛掉了重要收益，不能推导所有价格约束都无效，也不能把条件反过来只买这些跳涨日期来追求1.2。", "",
            "## 两轮全部账户差额", "",
            "以下均为新账户减原学习全部退出账户，直接由保存账本核对。额外费用为负表示费用节省。", "",
            "|轮次、时期及费用|终点资产差额（元）|价格损益差（元）|股息差（元）|额外佣金及滑点（元）|",
            "|---|---:|---:|---:|---:|"]
    for r in account_differences:
        label = f"第{r['round']}轮／{'主评价' if r['period']=='evaluation' else '较早'}／{'基础' if r['cost']=='BASE' else '压力'}"
        text.append(f"|{label}|{r['final_equity_difference']:.2f}|{r['price_pnl_difference']:.2f}|{r['dividend_difference']:.2f}|{r['extra_commission_and_slippage']:.2f}|")
    text.append("")
    for n, slug, title, primary, status, decision in STUDIES:
        text += [f"## 第{n}轮全部因子和进出场规则", ""]
        lines = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8").splitlines()[1:]
        text += ["#" + line if line.startswith("##") else line for line in lines]
        text += ["", f"### 第{n}轮全部主评价结果", ""] + table(results[n]["all_metrics"])
        text += [f"### 第{n}轮全部较早历史结果", ""] + table(results[n]["earlier_diagnostics"])
        text += ["本轮验收：" + decision, ""]
    text += ["## 已完成核对及下一步", "",
             "第49轮9项必要测试、第50轮5项必要测试全部通过。部分退出验证实际数量、未成交保留、保护优先、登记权益、剩余成本分摊、旧周期股息、最小一手、原全出经济一致性和未来价格隔离。进入上限验证事前波动、除息、拒绝后不按日低补成交及边界成交。两轮全部8项账户差额可由价格、股息与费用核对。", "",
             "部分退出首次改变前，主评价各36日、较早各920日的原经济字段逐日一致；第一次减半与原全部退出同日同参考价。进入上限首次生效前原账本字段一致，较早两档费用全期字段完全一致。这些检查证明本轮实际改动与记录相符，不证明未来收益。", "",
             "本次2个新配置，每个时期18个记录，包括4个新账户、14个复用对照；没有新模型拟合、没有新参考账户、没有新行情下载。累计完成50轮、320个不同配置或范围、332个已评价来源版本、920个主评价记录；登记来源版本337个，含5个旧未运行绑定。不同费用和复用账户不算新的独立策略。", "",
             "另仅在已保存原路径上观察“浮盈达到原6%风险幅度且尚未请求卖出”的可能加仓机会：主评价18个收盘，来自3个原周期；较早136个收盘，来自4个原周期。当前不为如此少的主评价周期新建加仓引擎。这只是原路径的机会观察，并未实现半仓后加仓的新策略，也不算一个新回测。", "",
             "下一步继续从已有可用信号中寻找当前组合尚未覆盖的不同机会，优先核对放量收强的回升信号与原学习退出是否已有同源组合结果；它与第46轮高波动急跌回升有不同成交量条件。只有确认并非重复实现、完整两段状态可得时才固定一个新的比较，不搜索权重，不反向利用本轮被拦住的三个跳涨日期。当前下一研究尚未登记或运行。", "",
             "完整扣费净夏普1.2和稳定超额目标仍未完成，持续研究保持活动。EPS、研报日期、股数、估值、财报和公募来源补齐继续暂停，不制作GPT数值包。仍仅510300和现金，其他ETF问题等待原回复，不重复询问。", "",
             "## 普通交付文件", ""]
    for p in copies:
        text.append(f"- [{p.name}]({p.name})")
    text += ["", "直接账户证据：", ""]
    for n, slug, title, primary, status, decision in STUDIES:
        text.append(f"- 第{n}轮：[结果索引](<{(folder(slug)/'result.json').as_posix()}>)；[基础费用逐日账户](<{(folder(slug)/'evaluation/BASE'/(primary+'_ledger.parquet')).as_posix()}>)。")
    document = OUT / "分批退出与进入价格_全部因子规则和历史表现.md"
    content = "\n".join(text) + "\n"
    require("```" not in content, "规则被代码块替代")
    document.write_text(content, encoding="utf-8")
    for match in re.finditer(r"\]\(([^)]+)\)", content):
        target = match.group(1).strip("<>")
        if not target.startswith("http"):
            require((OUT / target).is_file(), "交付链接缺失：" + target)
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for k in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                      "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[k] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS49_50_COMPLETE_EXISTING_SIGNAL_RESEARCH_CONTINUES", latest_completed_round=records[-1], running_studies=[], goal_achieved=False,
                 process_state_note="第49至50轮及分批退出保存路径诊断完成，全部进程退出；无下一项已登记研究。",
                 count_warning="累计50轮、320不同配置或范围、332已评价来源版本、920主评价记录；登记337含5旧未运行绑定。较早、拟合、参考另计。",
                 checks="14项必要测试通过；8项全账户差额、分批退出32项固定周期差额及价格上限实际变化前与较早全期原账本一致性已核对。",
                 latest_saved_partial_exit_diagnostic=str((DIAG/"result.json").relative_to(ROOT)),
                 entry_pyramiding_scope_check=str(feasibility_path.relative_to(ROOT)),
                 next_work=[
                     "第49轮先减半主评价基础0.621551、压力0.588282，较早0.703190、压力0.680536，均低于原全部退出。停止减仓比例、次数和负预测确认天数搜索。",
                     "分批退出主评价14周期中9次留下份额较差，固定实际份额合计少22925.96元；原另10个进入日期未出现，其中6亏4赚，因此不是全部错失盈利。完整账户实际少4919.39元。保留原学习全部退出。",
                     "第50轮买价上限主评价0.481778、压力0.414361；仅拦2024-09-25、09-30、10-08，漏掉原9月25日至30日净赚33605.62元的周期。较早两费用全期原字段完全相同，不算改善。停止价格上限倍数、波动窗口及有效期搜索，不反向只买被拦日期。",
                     "按原6%风险幅度观察确认加仓机会，主评价仅3个原周期、较早4个；不是新策略账户，当前不建加仓引擎。",
                     "下一项优先核对既有V1_CLIMAX_RECOVERY放量收强与REARM_RIDGE是否已有完全同源组合。V1不是第46轮V6_PANIC_RECOVERY；旧第28轮包含原D60、S1和V1，不等于当前学习退出与V1。先检查已有结果和两个时期来源，再仅在非重复且有明确机制时固定一个比较，禁止相邻权重搜索。当前未登记未运行。",
                     "原第32轮保持比较基线，第46轮主评价0.944391、较早0.604421只作局部线索，完整1.2仍未达到。EPS及所有来源补齐保持暂停，其他ETF问题待原回复，不重复询问，不做GPT数值包或新子任务。"])
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [49, 50]]
    delivery = {"created_at": now(), "type": "PARTIAL_EXIT_ENTRY_CAP_ROUNDS49_50_CHINESE_RESULTS", "rounds": [49, 50],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 50 and index["evaluation_accounts_in_this_resumption"] == 920 and index["evaluated_configurations_in_this_resumption"] == 320 and
            index["evaluated_candidate_source_runs_including_corrected_replays"] == 332 and index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 337,
            "累计研究数量不符")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_SAVED_ACCOUNT_ATTRIBUTION", "rounds": [49, 50],
               "new_configurations": 2, "main_records": 18, "new_main_accounts": 4, "reused_main_accounts": 14,
               "earlier_records": 18, "new_earlier_accounts": 4, "reused_earlier_accounts": 14,
               "new_model_fits": 0, "new_reference_accounts": 0, "necessary_tests_passed": 14, "saved_account_difference_checks": 8,
               "fixed_cycle_increment_rows": 32, "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "文档字符": len(content), "普通文件": len(receipt["files"]),
                      "累计完成": 50, "主评价记录": 920, "原缺少10次进入的净损益": float(missing_main.net_profit_cny.sum()),
                      "第50轮实际账户差额": [d for d in account_differences if d["round"] == 50], "目标完成": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
