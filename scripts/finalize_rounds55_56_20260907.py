"""交付第55至56轮中文因子、进出场及完整账户结果，更新持续研究索引。"""
from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300回踩与收盘位置_第55至56轮_20260907"
STUDIES = [
    (55, "breakout_retest_entry", "先突破后回踩确认进入", ["BREAKOUT_RETEST"], "ORDINARY_BREAKOUT",
     "突破后回踩的主评价净收益为负，较早历史改善没有延续；停止本回踩方法的时限、容忍区间、确认条件及退出参数细调。"),
    (56, "close_location_entry_gate", "成交量加权及等权收盘位置确认进入", ["VOLUME_LOCATION_ENTRY", "EQUAL_LOCATION_ENTRY"], "REARM_RIDGE",
     "两个收盘位置版本在两段、两档费用下的夏普均低于原学习退出；停止窗口、正值阈值、反向条件及确认次数细调。"),
]


def source(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def metric(result, key, period="evaluation", cost="BASE"):
    return next(m for m in result["all_metrics" if period == "evaluation" else "earlier_diagnostics"] if m["model"] == key and m["cost"] == cost)


def table(rows):
    text = ["|策略|费用|净夏普|复合年化|最大回撤幅度|较买入持有年化差|平均股票仓位|成交笔数|佣金（元）|滑点（元）|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        sharpe = f"{m['net_sharpe']:.3f}" if m["net_sharpe"] is not None else "无有效值"
        text.append(f"|{m['name']}|{'基础' if m['cost']=='BASE' else '压力'}|{sharpe}|{m['annualized_return']:.2%}|{-m['max_drawdown']:.2%}|{m['annualized_return_excess_vs_buy_hold']*100:.2f}个百分点|{m['mean_exposure']:.2%}|{m['trade_count']}|{m['commission']:.2f}|{m['slippage_cost']:.2f}|")
    return text + [""]


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 55)), set(range(1, 57))], "已经存在其他轮次，不覆盖研究索引")
    results = {n: json.loads((source(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, *_ in STUDIES}
    require(not any(r["historical_point_target_met"] for r in results.values()), "出现达到1.2的候选，不能套用失败交付")
    OUT.mkdir(parents=True, exist_ok=True)
    copies, records, deltas, cycle_checks, cycle_summaries = [], [], [], [], []
    for n, slug, title, candidates, control, decision in STUDIES:
        p, result = source(slug), results[n]
        outcome = {"recorded_at": now(), "status": "COMPLETED_TARGET_NOT_MET_NO_ACROSS_PERIOD_IMPROVEMENT", "decision": decision,
                   "goal_achieved": False, "position_impact": 0, "independent_validation": "NOT_ESTABLISHED"}
        write_json(p / "acceptance_outcome.json", outcome)
        primary = result["primary"][0]["model"]
        records.append({"round": n, "study": result["study_id"], "title": title, "status": outcome["status"],
                        "result": str((p / "result.json").relative_to(ROOT)),
                        **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
                                                 "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
                        "evaluated_candidate_source_runs": result["candidate_configurations"],
                        "primary_base": metric(result, primary), "primary_stress": metric(result, primary, cost="STRESS"),
                        "post_selected_best_base": result["post_selected_best_base"]})
        for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "result.json", "acceptance_outcome.json"]:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((p / filename).read_bytes())
            copies.append(dest)
        factors = pd.read_parquet(p / "all_daily_factors.parquet")
        factor_names = {"date": "日期", "setup_event_index": "原突破日序号", "setup_event_date": "原突破日期", "fixed_breakout_level": "固定突破位_含分红尺度",
                        "setup_age": "距原突破交易日数", "setup_created": "当日建立机会", "candidate_entry": "当日确认信号", "setup_status": "机会状态",
                        "ordinary_breakout": "普通突破条件", "first_breakout": "普通突破首日", "low_total_return_scale": "当日最低价_含分红尺度",
                        "prior20_high": "此前二十日最高收盘_含分红尺度", "inputs_available": "价格输入完整", "daily_location": "当日收盘位置_负一至正一",
                        "volume_sum21": "二十一日总成交量_原单位", "volume_location21": "二十一日成交量加权收盘位置",
                        "equal_location21": "二十一日等权收盘位置", "location_available": "二十一日因子完整"}
        dest = OUT / f"第{n}轮_全部逐日新因子与状态.csv"
        factors.rename(columns=factor_names).to_csv(dest, index=False, encoding="utf-8-sig")
        copies.append(dest)
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                folder = p / period / cost
                comparisons = [(key, control) for key in candidates]
                if n == 56:
                    comparisons.append(("VOLUME_LOCATION_ENTRY", "EQUAL_LOCATION_ENTRY"))
                for key, comparator in comparisons:
                    a = pd.read_parquet(folder / f"{key}_ledger.parquet")
                    b = pd.read_parquet(folder / f"{comparator}_ledger.parquet")
                    require(pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date)), "两账户没有相同完整日期")
                    d = {"round": n, "period": period, "cost": cost, "candidate": key, "control": comparator,
                         "final_equity_difference": float(a.equity.iloc[-1] - b.equity.iloc[-1]),
                         "price_pnl_difference": float(a.price_pnl.sum() - b.price_pnl.sum()),
                         "dividend_difference": float(a.dividend_recognized.sum() - b.dividend_recognized.sum()),
                         "extra_fees": float(a.commission.sum() + a.slippage_cost.sum() - b.commission.sum() - b.slippage_cost.sum())}
                    require(abs(d["final_equity_difference"] - d["price_pnl_difference"] - d["dividend_difference"] + d["extra_fees"]) < 1e-6,
                            "价格股息及费用差不能核对终点资产差")
                    deltas.append(d)
                for key in candidates:
                    a = pd.read_parquet(folder / f"{key}_ledger.parquet")
                    c = pd.read_csv(folder / f"{key}_cycles.csv")
                    residual = float(a.equity.iloc[-1] - 200000 - c.net_profit_cny.sum())
                    require(abs(residual) < 1e-6, "持仓周期损益与完整账户不符")
                    cycle_checks.append({"round": n, "period": period, "cost": cost, "model": key, "residual_cny": residual})
                    cycle_summaries.append({"round": n, "period": period, "cost": cost, "model": key, "cycles": len(c),
                                            "positive_cycles": int(c.net_profit_cny.gt(0).sum()), "positive_profit": float(c.loc[c.net_profit_cny.gt(0), "net_profit_cny"].sum()),
                                            "nonpositive_profit": float(c.loc[c.net_profit_cny.le(0), "net_profit_cny"].sum()), "total_profit": float(c.net_profit_cny.sum())})
                    dest = OUT / f"第{n}轮_{period}_{cost}_{key}_全部持仓周期.csv"
                    dest.write_bytes((folder / f"{key}_cycles.csv").read_bytes())
                    copies.append(dest)
                    if n == 56:
                        d = pd.read_parquet(folder / f"{key}_decisions.parquet")
                        d = d[d.location_status.notna()][["origin", "execution_date", "location_gate_value", "entry_allowed", "requested_quantity", "action"]]
                        dest = OUT / f"第56轮_{period}_{cost}_{key}_每次实际入场检查.csv"
                        d.rename(columns={"origin": "决策收盘日", "execution_date": "计划成交日", "location_gate_value": "位置因子", "entry_allowed": "通过检查",
                                          "requested_quantity": "请求买入份额", "action": "动作原因"}).to_csv(dest, index=False, encoding="utf-8-sig")
                        copies.append(dest)
    for n, slug, filename in [(55, "breakout_retest_entry", "state_statistics.csv"), (55, "breakout_retest_entry", "entry_statistics.csv"),
                              (56, "close_location_entry_gate", "entry_gate_statistics.csv")]:
        dest = OUT / f"第{n}轮_{filename}"
        dest.write_bytes((source(slug) / filename).read_bytes())
        copies.append(dest)
    for filename, rows in [("完整账户_价格股息费用差额.csv", deltas), ("全部周期到账户核对.csv", cycle_checks), ("全部候选_盈利与亏损周期汇总.csv", cycle_summaries)]:
        dest = OUT / filename
        pd.DataFrame(rows).to_csv(dest, index=False, encoding="utf-8-sig")
        copies.append(dest)
    r55, r56 = results[55], results[56]
    for key in ["VOLUME_LOCATION_ENTRY", "EQUAL_LOCATION_ENTRY"]:
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                require(metric(r56, key, period, cost)["net_sharpe"] < metric(r56, "REARM_RIDGE", period, cost)["net_sharpe"], "收盘位置失败文字不符合结果")
    text = ["# 510300：回踩进入与收盘位置筛选的完整结果", "", "第55至56轮，2026年9月7日。中文因子、进出场规则及全部历史表现。", "",
            "## 先看结论", "",
            "完整账户扣费净夏普1.2仍未实现。这次三个新设置都没有跨时期改善各自原方案，全部结束本轮细调。前瞻EPS及其他慢来源补齐继续暂停，未增加数据源或重新训练模型。", "",
            "|方案|主评价基础夏普|主评价压力夏普|较早基础夏普|较早压力夏普|主评价基础复合年化|主评价基础最大回撤幅度|",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for result, key in [(r55, "BREAKOUT_RETEST"), (r55, "ORDINARY_BREAKOUT"), (r56, "VOLUME_LOCATION_ENTRY"), (r56, "EQUAL_LOCATION_ENTRY"),
                        (r56, "REARM_RIDGE"), (r56, "PANIC_LEARNED_HALF"), (r56, "BUY_HOLD")]:
        b, s = metric(result, key), metric(result, key, cost="STRESS")
        eb, es = metric(result, key, "earlier_diagnostic"), metric(result, key, "earlier_diagnostic", "STRESS")
        text.append(f"|{b['name']}|{b['net_sharpe']:.3f}|{s['net_sharpe']:.3f}|{eb['net_sharpe']:.3f}|{es['net_sharpe']:.3f}|{b['annualized_return']:.2%}|{-b['max_drawdown']:.2%}|")
    text += ["", "主评价为2020年1月2日至2026年8月14日终点开盘，1604个账户日；较早为2015年1月5日至2019年12月31日终点开盘，1219日。每段各从20万元开始，所有空仓日、佣金、滑点、整手、分红和未成交约束均保留。242日年化，现金与无风险收益均假设零；没有缩短窗口、降低费用或只算持仓日。两段历史均已被多轮研究观察，不属于独立验证。", "",
             "## 第55轮：回踩减少了买入，也漏掉或改变了收益机会", "",
             "主评价65个突破首日建立待回踩机会，16日后来确认，31日收盘跌破固定突破位而取消，18日到期；最终14次实际持仓，来自14个不同突破事件。较早52个新建机会、10日确认、22日失败、21日到期，实际7次持仓。较早的终止状态合计与当期新建数量不同，是因为允许期前机会延续，不能硬把当期新建数当成所有状态的共同分母。", "",
             "主评价净夏普从普通突破0.135降至负0.383，复合年化从0.89%降至负2.26%。最大回撤虽从23.41%收窄至15.40%，但收益为负，不能因为少交易、回撤小而保留为高夏普策略。较早夏普从0.283升至0.652、年化从2.91%升至5.07%，没有延续到主评价。", ""]
    for s in cycle_summaries:
        if s["round"] == 55 and s["cost"] == "BASE":
            text.append(f"{'主评价' if s['period']=='evaluation' else '较早'}基础费用：{s['cycles']}次完整持仓，{s['positive_cycles']}次盈利；盈利周期合计{s['positive_profit']:,.2f}元，非盈利周期合计{s['nonpositive_profit']:,.2f}元，净损益{s['total_profit']:,.2f}元。")
    text += ["", "较早2019年1月25日至4月26日这一个周期盈利49,228.74元，而整个较早账户净赚56,524.28元，改善集中在少量行情。这里列出集中度事实，不把该事件单独删去后再宣称另一个策略，也不据此挑选年份。", "",
             "## 第56轮：成交量有局部差别，新增筛选没有改善原基线", "",
             "预定主方案要求最近二十一日成交量加权的收盘位置大于零；同时运行同窗口等权版本。两者都只额外决定是否接受原买入，不改原学习退出。成交量加权指标由日内高低收盘及成交量计算，不是真实公募申购、资金净流入或EPS。", "",
             "主评价加权版进行了59次符合原条件的入场检查，21次通过、38次暂缓，最终21个持仓周期；等权版70次检查、20次通过、50次暂缓、20个周期。较早分别12次检查、8次通过、8个周期，以及13次检查、7次通过、7个周期。所有实际检查均有完整因子；检查次数与独立事件、成交次数不是同一概念。", "",
             "加权版主评价夏普0.685优于等权版0.448，但较早0.636低于等权版0.688。更关键的是，两个版本在两段及压力费用下均低于原学习退出。不能只比较两个新增版本，再把其中相对较高者说成有效增量。", "",
             "基础费用下，加权版终点资产较原学习退出主评价少6,394.63元、较早少21,657.98元；等权版分别少41,608.14元、14,879.37元。减少交易没有抵消持仓时点及份额变化造成的收益损失。逐日检查、全部持仓周期和价格、股息、费用差额一并保存，不再针对这批结果改收盘位置窗口或零门槛。", "",
             "## 完整账户差额", "", "以下均为候选减对照；额外费用为负表示节省。每条差额均核对到实际终点资产。", "",
             "|轮次|时期及费用|候选|对照|终点资产差（元）|价格损益差（元）|股息差（元）|额外费用（元）|",
             "|---|---|---|---|---:|---:|---:|---:|"]
    name_map = {m["model"]: m["name"] for r in results.values() for m in r["all_metrics"]}
    for d in deltas:
        label = ("主评价" if d["period"] == "evaluation" else "较早") + "／" + ("基础" if d["cost"] == "BASE" else "压力")
        text.append(f"|{d['round']}|{label}|{name_map[d['candidate']]}|{name_map[d['control']]}|{d['final_equity_difference']:.2f}|{d['price_pnl_difference']:.2f}|{d['dividend_difference']:.2f}|{d['extra_fees']:.2f}|")
    text.append("")
    for n, slug, title, candidates, control, decision in STUDIES:
        text += [f"## 第{n}轮全部因子、进入、持有、退出和再进入规则", ""]
        for line in (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8").splitlines()[1:]:
            text.append("#" + line if line.startswith("##") else line)
        text += ["", f"### 第{n}轮全部主评价", ""] + table(results[n]["all_metrics"])
        text += [f"### 第{n}轮全部较早诊断", ""] + table(results[n]["earlier_diagnostics"])
        text += ["本轮验收：" + decision, ""]
    next_work = [
        "第55轮回踩进入失败：主评价夏普负0.382948、压力负0.435149、年化负2.26%；较早0.652144、压力0.626683。实际14及7周期。早期净利润56524.28元中2019一次49228.74元；不能据此挑年份。停止回踩时限、容忍区间、确认涨幅和退出参数调整。",
        "第56轮收盘位置进入失败。成交量加权21日主评价0.684714、压力0.627552，较早0.635891、压力0.615850；等权0.448200、压力0.387889，较早0.688272、压力0.670638。两段两费用均低于原REARM_RIDGE。主评价21/20周期、较早8/7周期；没有模型或参考重训。停止该因子窗口、阈值、反向条件与确认次数调整。",
        "转向已有不同机制之间的互补检验，少给同一学习进入添加筛选。已读第6轮日历协议与代码：K2_MONTH_EDGE固定月末五个自然日或月初三个交易日，09:00形成当日开盘计划，原主评价夏普0.519648。限定检索没有找到它与R32学习退出的已做组合。下一步只研究两者原规则各半的有限组合，不扫描权重；先核对9点日历与前一收盘价格时钟，以及较早旧日历账户是否存在，缺少则如实记新旧对照。当前这项未登记、未运行、没有新收益读取。",
        "现有R32仍为比较基线，R46主评价0.944391、较早0.604421仍是局部线索，未到1.2或独立验证。R54压缩后突破只有较弱普通突破的局部改善。停止围绕34原参考周期调整学习退出标签，不重做已保存的失败诊断。",
        "EPS、日期、股数、估值、财报、公募等慢来源补齐继续暂停，只510300与现金，其他ETF等待原回复且不重复问。所有进入、退出与历史表现用中文；无GPT数值包、无安全审计、无子任务或券商交易。",
    ]
    text += ["## 完成情况及继续方向", "",
             "本次两轮、三个新设置，每时期20条评价记录：6个新候选账户、14个复用对照。没有新模型、新参考账户或新数据下载。13项必要测试通过，16项账户差额与12项周期到账户核对通过；未开展安全审计，未制作GPT数值包。", "",
             "累计完成56轮、329个不同配置或范围、341个已评价来源版本、982个主评价记录；登记来源版本346个，含5个旧未运行绑定。这些是工作记录数量，不是329个独立有效策略。", "",
             "继续方向是检验已有不同机制的互补性。下一项拟用原月内两端规则和原学习退出各半，先核对日历在上午九点可知、价格只用此前收盘的时钟，以及较早对照账户的覆盖。两者原进入退出不调整，不根据收益选择权重。目前仅完成方法与重复研究检索，尚未登记、运行该组合，不能预报其夏普。", "",
             "目标仍为完整账户净夏普至少1.2及稳定超额。没有完成目标，持续研究保持开启。EPS和其他慢来源补齐维持暂停。", "",
             "## 普通结果文件", ""]
    for p in copies:
        text.append(f"- [{p.name}]({p.name})")
    for n, slug, *_ in STUDIES:
        text.append(f"- 第{n}轮[原始完整结果](<{(source(slug)/'result.json').as_posix()}>)，同目录保留全部逐日账户、决策及交易。")
    document = OUT / "回踩与收盘位置_全部因子规则和历史表现.md"
    content = "\n".join(text) + "\n"
    require("```" not in content, "中文策略规则被代码替代")
    document.write_text(content, encoding="utf-8")
    for match in re.finditer(r"\]\(([^)]+)\)", content):
        target = match.group(1).strip("<>")
        if not target.startswith("http"):
            require((OUT / target).is_file(), "交付链接不存在：" + target)
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS55_56_COMPLETE_NO_ACROSS_PERIOD_IMPROVEMENT", latest_completed_round=records[-1],
                 running_studies=[], goal_achieved=False, process_state_note="第55至56轮完整完成，进程正常退出；下一项日历组合未登记未运行。",
                 count_warning="累计56轮、329不同配置或范围、341已评价来源版本、982主评价记录；登记346含5旧未运行绑定。较早、参考另计。",
                 checks="13项必要测试通过，16项完整账户差额及12项周期到账户核对通过，无新模型、参考或数据。", next_work=next_work)
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [55, 56]]
    delivery = {"created_at": now(), "type": "RETEST_AND_LOCATION_ROUNDS55_56_CHINESE_RESULTS", "rounds": [55, 56],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 56 and index["evaluated_configurations_in_this_resumption"] == 329 and
            index["evaluation_accounts_in_this_resumption"] == 982 and index["evaluated_candidate_source_runs_including_corrected_replays"] == 341 and
            index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 346, "累计研究计数不一致")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "CHINESE_RULES_AND_COMPLETE_ACCOUNT_RESULTS_DELIVERED", "rounds": [55, 56],
               "new_configurations": 3, "main_records": 20, "new_main_accounts": 6, "reused_main_accounts": 14,
               "earlier_records": 20, "new_earlier_candidate_accounts": 6, "reused_earlier_accounts": 14,
               "new_model_fits": 0, "new_reference_accounts": 0, "necessary_tests_passed": 13,
               "account_difference_checks": len(deltas), "cycle_account_checks": len(cycle_checks),
               "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文说明": str(document), "文档字符": len(content), "普通文件": len(receipt["files"]),
                      "累计轮次": 56, "累计主评价": 982, "账户差额": len(deltas), "周期核对": len(cycle_checks)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
