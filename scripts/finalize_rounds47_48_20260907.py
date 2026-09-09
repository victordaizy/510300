"""交付第三信号组合与自适应速度两轮结果，保留未达标结论。"""
import json
import re
from pathlib import Path
import pandas as pd
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.deliver_simple_strategy_rounds34_36_20260907 import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/510300第三信号与自适应速度_第47至48轮_20260907"
INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
STUDIES = [
    (47, "broader_equal_blends", "增加更频繁的第三个信号", "ADD_VOLUME_BREAKOUT",
     "COMPLETED_NO_IMPROVEMENT_FROM_THIRD_SIGNALS", "两个三等份组合在两时期的夏普都低于原两信号组合，不采用，不扫描这些第三信号的相邻权重或阈值。"),
    (48, "adaptive_speed_trend", "方向效率自动调整均线速度", "ADAPTIVE_SPEED",
     "COMPLETED_ADAPTIVE_SPEED_TARGET_NOT_MET", "自适应速度在主评价好于固定速度，但较早未改善，复合收益弱且回撤大；不采用，不搜索相邻窗口、快慢周期或确认天数。"),
]


def folder(slug):
    return ROOT / f"reports/research/510300_{slug}_v1"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    completed = {r["round"] for r in index["completed_rounds"]}
    require(completed in [set(range(1, 47)), set(range(1, 49))], "研究索引出现其他新轮次，停止覆盖")
    results = {n: json.loads((folder(slug) / "result.json").read_text(encoding="utf-8")) for n, slug, *_ in STUDIES}
    OUT.mkdir(parents=True, exist_ok=True)
    copies, records, differences = [], [], []
    for n, slug, title, primary, status, decision in STUDIES:
        r = results[n]
        record = {"round": n, "study": r["study_id"], "title": title, "status": status,
                  "result": f"reports/research/510300_{slug}_v1/result.json",
                  **{k: r[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
                                      "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]},
                  "evaluated_candidate_source_runs": r["candidate_configurations"],
                  "primary_base": next(m for m in r["all_metrics"] if m["model"] == primary and m["cost"] == "BASE"),
                  "primary_stress": next(m for m in r["all_metrics"] if m["model"] == primary and m["cost"] == "STRESS"),
                  "post_selected_best_base": r["post_selected_best_base"]}
        if n == 47:
            record.update(new_earlier_candidate_accounts=4, new_earlier_control_accounts=2)
        records.append(record)
        write_json(folder(slug) / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "decision": decision,
                   "goal_achieved": False, "position_impact": 0})
        for filename in ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "result.json"]:
            dest = OUT / f"第{n}轮_{filename}"
            dest.write_bytes((folder(slug) / filename).read_bytes())
            copies.append(dest)
        keys = ["ADD_VOLUME_BREAKOUT", "ADD_REGULAR_REBOUND"] if n == 47 else ["ADAPTIVE_SPEED"]
        control = "ORIGINAL_TWO" if n == 47 else "FIXED_NEUTRAL_SPEED"
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                for key in keys:
                    a = pd.read_parquet(folder(slug) / period / cost / f"{key}_ledger.parquet")
                    b = pd.read_parquet(folder(slug) / period / cost / f"{control}_ledger.parquet")
                    require(pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date)), "保存账户日历不同")
                    d = {"round": n, "period": period, "cost": cost, "candidate": key, "control": control,
                         "final_equity_difference": float(a.equity.iloc[-1] - b.equity.iloc[-1]),
                         "price_pnl_difference": float(a.price_pnl.sum() - b.price_pnl.sum()),
                         "dividend_difference": float(a.dividend_recognized.sum() - b.dividend_recognized.sum()),
                         "extra_commission_and_slippage": float(a.commission.sum() + a.slippage_cost.sum() - b.commission.sum() - b.slippage_cost.sum())}
                    require(abs(d["final_equity_difference"] - d["price_pnl_difference"] - d["dividend_difference"] + d["extra_commission_and_slippage"]) < 1e-6,
                            "已保存账户差额无法核对")
                    differences.append(d)
                if n == 48:
                    for key in ["ADAPTIVE_SPEED", "FIXED_NEUTRAL_SPEED"]:
                        dest = OUT / f"第48轮_{period}_{cost}_{key}_全部持仓周期.csv"
                        dest.write_bytes((folder(slug) / period / cost / f"{key}_cycles.csv").read_bytes())
                        copies.append(dest)
    for original, name in [
        (folder("broader_equal_blends") / "state_counts.csv", "第47轮_全部组合状态.csv"),
        (folder("adaptive_speed_trend") / "factor_statistics.csv", "第48轮_速度和有效因子统计.csv"),
        (ROOT / "deliverables/510300学习目标与互补组合_第44至46轮_20260907/沿用原模型_每月实际中文规则.md", "第47轮沿用原模型_每月实际中文规则.md")]:
        dest = OUT / name
        dest.write_bytes(original.read_bytes())
        copies.append(dest)
    dest = OUT / "全部保存账户_价格分红费用差额.csv"
    pd.DataFrame(differences).to_csv(dest, index=False, encoding="utf-8-sig")
    copies.append(dest)
    text = ["# 510300：增加信号和自动调速的实际结果", "", "第47至48轮，2026年9月7日。供管理层阅读，全部因子和进出场规则均用中文。", "",
            "## 先看结论", "",
            "这两轮的四个设置均未达到完整扣费夏普1.2。增加交易机会没有提升原组合；均线自动调整速度也未形成可用策略。两条新方向均停止相邻参数搜索，前瞻EPS和其他原来源补齐继续暂停。", "",
            "|策略|主评价基础夏普|主评价压力夏普|基础复合年化收益|基础最大回撤幅度|较早基础夏普|较早压力夏普|",
            "|---|---:|---:|---:|---:|---:|---:|",
            "|原急跌与学习信号各半，比较线索|0.944|0.879|4.01%|5.39%|0.604|0.572|",
            "|第47轮：加入放量突破，三个各三分之一|0.841|0.780|4.35%|6.33%|0.456|0.412|",
            "|第47轮：加入普通均值回升，三个各三分之一|0.891|0.800|3.52%|5.54%|0.439|0.392|",
            "|第48轮：方向效率自动调速|0.006|−0.103|−0.75%|32.93%|0.068|−0.030|",
            "|第48轮：固定中性速度对照|−0.141|−0.257|−2.75%|34.18%|0.081|−0.023|", "",
            "主评价统一为2020年1月2日至2026年8月14日开盘，共1604个账户日；较早诊断为2015年1月5日至2019年12月31日开盘，共1219日。两段均从20万元开始。夏普使用完整账户的日收益，包含全部空仓日，扣除佣金与滑点，现金和无风险收益为零，242日年化。较早历史也已被研究过，不能称为独立验证。", "",
            "原0.944组合仍只是主评价中的局部线索：其年化收益4.01%，较早夏普0.604；急跌子策略主评价仅三次不同持仓，较早单策略亏损。没有因此升级为已验证策略。", "",
            "## 这些结果说明了什么", "",
            "第三信号提供了更多参与机会，但这些机会的收益与风险不够好。原放量突破单策略主评价有20次持仓，基础夏普0.509；本次补算的较早16次持仓夏普只有0.080，压力费用下复合年化收益已为负。普通均值回升单策略主评价25次持仓、夏普0.370，较早13次持仓、夏普−0.112。单策略发生次数增加本身不能证明组合获益。", "",
            "加入放量突破后，主评价买卖由原52笔增至89笔，年化收益由4.01%升至4.35%，最大回撤由5.39%扩大至6.33%，夏普反而下降。较早夏普降至0.456，复合年化收益2.98%。加入普通均值回升后，主评价94笔买卖，年化收益3.52%，低于买入持有的3.63%；较早年化收益2.51%。两种结果都不足以替换原组合。", "",
            "三等份组合也降低了原两个信号各自的预算，因此不能把差额全部理解为只增加第三个信号的纯贡献；这是按固定新预算运行一个统一账户的整体效果。状态表保留了全部仓位档位天数，实际账户含再平衡和费用。", "",
            "自动调速方案的主评价平均十日方向效率0.319，平均反应系数0.0836；较早分别0.336和0.0892。固定速度对照为0.1337。自适应反应系数每天改变，但改变本身不产生收益优势：主评价67个持仓周期、134笔买卖，较早52个周期、104笔买卖；两段复合年化收益均略为负，最大回撤分别32.93%和35.25%。", "",
            "主评价自适应夏普略高于零而复合年化为负并不矛盾：夏普分子为日收益的算术均值，复合年化受逐日波动损耗影响。本轮同时保留算术均值、年化波动率与复合年化，避免把不同指标混为一谈。", "",
            "## 保存账户差额", "",
            "以下直接从已完成的账户核对终点差额，无新回测；第一项减去对照。价格、分红与额外费用共同决定差额，不将单一因子直接认定为因果来源。", "",
            "|时期与费用|新设置相对对照|终点资产差额（元）|价格损益差（元）|分红差（元）|额外佣金及滑点（元）|",
            "|---|---|---:|---:|---:|---:|"]
    labels = {"ADD_VOLUME_BREAKOUT": "加入放量突破－原两信号组合", "ADD_REGULAR_REBOUND": "加入普通回升－原两信号组合",
              "ADAPTIVE_SPEED": "自动调速－固定速度"}
    for r in differences:
        if r["cost"] == "BASE":
            period = "主评价基础" if r["period"] == "evaluation" else "较早基础"
            text.append(f"|{period}|{labels[r['candidate']]}|{r['final_equity_difference']:.2f}|{r['price_pnl_difference']:.2f}|{r['dividend_difference']:.2f}|{r['extra_commission_and_slippage']:.2f}|")
    text += ["", "额外费用为负代表费用减少；完整两档费用的12项比较在附表。", ""]
    for n, slug, title, primary, status, decision in STUDIES:
        text += [f"## 第{n}轮全部因子与完整进出规则", ""]
        lines = (ROOT / f"docs/510300_{slug.upper()}_V1.md").read_text(encoding="utf-8").splitlines()[1:]
        text += ["#" + line if line.startswith("##") else line for line in lines]
        text += ["", f"### 第{n}轮全部主评价结果", ""] + table(results[n]["all_metrics"])
        text += [f"### 第{n}轮全部较早历史结果", ""] + table(results[n]["earlier_diagnostics"])
        text += ["验收结论：" + decision, ""]
    text += ["## 交付和继续方向", "",
             "本次4个新配置，共22个主评价记录，其中8个新账户、14个复用对照；较早也有22个记录，其中8个新配置账户、2个新生成的原放量突破对照、12个复用对照。没有新预测模型拟合、没有新连续参考账户、没有下载新行情。", "",
             "第48轮5项必要测试全部通过，验证未来数据隔离、单向路径的封闭解、平坦与往返路径、缺失中断及双收盘确认。第47轮复用已测试状态映射与事件账户，直接核对全日历、允许的仓位档位、财富恒等式和终点结算。上述12项保存账户差额全部核对成立。", "",
             "累计48轮、318个不同配置或范围、330个已评价来源版本、902个主评价记录；登记来源版本335个，其中5个旧绑定未运行。费用档、复用对照和更早诊断不等于新的独立策略数。", "",
             "下一项优先检查一个尚未证实的假设：原学习模型第一次要求退出时，是否可以先减一半，余下部分仍按原自然退出规则管理，减少一次性清仓错过延续行情的可能。需要把部分卖出后的分红权益、剩余成本、退出等待和再入场规则先写清楚，再固定一个版本直接计算；不能把该假设写成已经确认的失败原因。此下一项在本说明生成时尚未登记或运行。", "",
             "完整账户净夏普1.2及稳定超额仍未完成，研究继续。EPS、研报日期、股数、估值、财报和公募来源补齐保持暂停；不制作GPT审阅数值包。当前仍只研究510300和现金，其他ETF的范围问题等待原回复，不重复询问。", "",
             "普通结果文件：", ""]
    for p in copies:
        text.append(f"- [{p.name}]({p.name})")
    text += ["", "完整账户和因子直接来源：", ""]
    for n, slug, title, primary, status, decision in STUDIES:
        text.append(f"- 第{n}轮：[结果索引](<{(folder(slug) / 'result.json').as_posix()}>)；[主方案逐日账户](<{(folder(slug) / 'evaluation/BASE' / (primary + '_ledger.parquet')).as_posix()}>)。")
    text.append(f"- 第48轮：[逐日全部因子](<{(folder('adaptive_speed_trend') / 'factors.parquet').as_posix()}>)。")
    document = OUT / "第三信号与自动调速_全部因子规则和历史表现.md"
    content = "\n".join(text) + "\n"
    require("```" not in content, "中文规则文档出现代码替代")
    document.write_text(content, encoding="utf-8")
    for match in re.finditer(r"\]\(([^)]+)\)", content):
        target = match.group(1).strip("<>")
        if not target.startswith("http"):
            require((OUT / target).is_file(), "文档链接缺失：" + target)
    for record in records:
        if record["round"] not in completed:
            index["completed_rounds"].append(record)
            index["evaluation_accounts_in_this_resumption"] += record["evaluation_accounts"]
            for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption",
                        "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
                index[key] += record["candidate_configurations"]
    index.update(updated_at=now(), status="ROUNDS47_48_COMPLETE_EXIT_MECHANISM_RESEARCH_CONTINUES", latest_completed_round=records[-1], running_studies=[], goal_achieved=False,
                 process_state_note="第47至48轮已完成并退出，暂无下一项已登记研究；下一方向为部分退出机制，尚未运行。",
                 count_warning="累计48轮、318不同配置或范围、330已评价来源版本、902主评价记录；登记335含5旧未运行绑定。较早、拟合、参考另计。",
                 checks="第48轮5项必要测试通过；第47轮复用已测状态和事件账户并核对全日历和结算；12项保存账户差额核对完成。",
                 next_work=[
                     "第47轮两个固定三等份组合主评价0.841234和0.890635，较早0.455751和0.438951，均低于第46轮；停止这两个第三信号的邻近权重、阈值搜索。",
                     "第48轮自动调速主评价0.005932、压力负0.102940，较早0.067587、压力负0.029802，两个基础复合年化均略负；固定速度对照也失败，停止该自适应均线相邻参数。",
                     "第32轮仍为原比较基线，第46轮0.944391只保留局部线索，较早0.604421且急跌只有三次主评价周期，不能认定达标。",
                     "下一项优先检查原学习退出第一次触发时减一半、余下按原自然退出规则管理的机制。尚未证实原一次性退出过早；先检查是否已有相同实现，再明确部分成交、分红权益、剩余成本、卖出等待及再入场，固定一个候选和原全出对照直接算账户，不搜索减仓比例。",
                     "保留EPS及所有原来源补齐暂停；其他ETF目标问题仍待回复，不重复询问。仅现有免费来源、510300和现金，不做GPT数值包、不创建子任务。"])
    index["partial_rounds"] = [r for r in index.get("partial_rounds", []) if r["round"] not in [47, 48]]
    delivery = {"created_at": now(), "type": "THIRD_SIGNAL_ADAPTIVE_SPEED_ROUNDS47_48_CHINESE_RESULTS", "rounds": [47, 48],
                "directory": str(OUT), "main_document": str(document), "new_gpt_review_archive_created": False}
    index["deliveries"] = [d for d in index["deliveries"] if d.get("type") != delivery["type"]] + [delivery]
    require(len(index["completed_rounds"]) == 48 and index["evaluation_accounts_in_this_resumption"] == 902 and index["evaluated_configurations_in_this_resumption"] == 318 and
            index["evaluated_candidate_source_runs_including_corrected_replays"] == 330 and index["registered_candidate_source_runs_including_unrun_legacy_bindings"] == 335,
            "累计研究计数不符")
    write_json(INDEX, index)
    receipt = {"created_at": now(), "status": "COMPLETE_CHINESE_RULES_AND_ACCOUNT_DIFFERENCES", "rounds": [47, 48],
               "new_configurations": 4, "main_records": 22, "new_main_accounts": 8, "reused_main_accounts": 14,
               "earlier_records": 22, "new_earlier_candidate_accounts": 8, "new_earlier_control_accounts": 2, "reused_earlier_accounts": 12,
               "new_model_fits": 0, "new_reference_accounts": 0, "necessary_tests_passed": 5, "saved_account_difference_checks": 12,
               "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False,
               "files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in [document] + copies]}
    write_json(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({"中文文档": str(document), "文档字符": len(content), "普通交付文件": len(receipt["files"]),
                      "完成轮数": 48, "主评价记录": 902, "目标完成": False, "基础费用保存差额": [d for d in differences if d["cost"] == "BASE"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
