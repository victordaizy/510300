"""交付已有候选比较及明确的后续组合，不增加新策略计数。"""
import json
import shutil
from pathlib import Path
import pandas as pd
from scripts.saved_candidate_frontier_20260909 import ROOT, OUT
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    ranked = pd.read_csv(OUT / "standard_four_scenario_ranking.csv")
    primary = pd.read_csv(OUT / "main_base_only_ranking.csv")
    destination = ROOT / "deliverables/510300已有候选薄弱情景比较_截至第130轮_20260909"
    destination.mkdir()
    document = destination / "已有候选比较及下一步研究.md"
    chosen = pd.concat([ranked.head(5), primary.head(1)]).drop_duplicates("candidate_id")
    lines = ["# 510300已有候选比较：把改动集中在真正的短板", "",
        "截至第130轮，完整目标仍未达到。已保存结果中，第128轮的四项最低夏普为0.766，在具备两段历史、两档费用的标准开盘方案中最高。"
        "第91轮主历史基础费用曾达到1.233，但最弱情景只有0.395；因此只追最高单点容易选错下一步。", "",
        "本次只读134个结果文档，包含第22轮明确引用的来源重放子结果。852条成对费用记录按名称和实际保存表现合并为455个命名表现版本，包含对照和不同保存版本，"
        "不是455种新策略。162个标准开盘版本具备可比较的四情景指标，没有一项四个夏普全部达到1.2；其中36项四情景年化超额均为正。"
        "这不证明逐年超额稳定或独立验证通过。", "",
        "另有7个完整版本涉及盘中成交假设，未混入标准开盘主排序；283个版本缺少较早历史，3个夏普因零波动未定义，均单独保留，没有补零或重新回测。"
        "第22轮父汇总和两个因子文件本身无指标，但它引用的两组账户已纳入，没有遗漏对应重放账户。", "",
        "## 主要比较", "", "主历史为2020年1月2日至2026年8月14日开盘；较早历史为2015年1月5日至2019年12月31日开盘。以下为扣费夏普。", "",
        "|方案|主基础|主压力|较早基础|较早压力|四项最低|四情景年化超额均为正|", "|---|---:|---:|---:|---:|---:|---|"]
    for r in chosen.itertuples():
        lines.append(f"|{r.name}|{r.main_base_net_sharpe:.3f}|{r.main_stress_net_sharpe:.3f}|{r.earlier_base_net_sharpe:.3f}|{r.earlier_stress_net_sharpe:.3f}|{r.minimum_sharpe:.3f}|{'是' if r.all_four_positive_excess else '否'}|")
    lines.extend(["", "## 为什么还没完成目标", "",
        "第一，风险缩减容易同时压低收益。109普通波动对照主基础年化1.47%、较早2.00%，平均股票敞口约2.73%与5.87%，均低于买入持有收益。"
        "91主基础年化3.36%、较早2.93%，较早波动明显更高。它们的高主历史夏普不能直接解释为稳定超额。", "",
        "第二，相对更好的进入退出仍不够强。128主基础年化6.97%、较早8.74%，两段压力费用年化也高于买入持有，但年化波动约8%和12%，"
        "四情景夏普仍在0.766至0.884之间。它是目前更值得集中研究的比较对象，尚不是通过验收的策略。", "",
        "第三，增加方法名称不一定改变交易。114和117虽有不同的状态定义，四条完整保存账户逐列相同，应理解为同一已发生交易表现。"
        "新指标130还出现持有307个收盘后亏损退出的交易，说明简单反转计数不能保证及时进入或退出。", "",
        "下一131只测试一个事先明确的组合：用128保存研究参考的机会，配109已有二十日普通波动乘数和既定10个百分点调仓带宽。"
        "目标是检验能否保留机会收益并降低风险，不重训旧模型，不补慢来源。这个组合因本次历史诊断而提出，仍属事后研究选择；"
        "新目标和账户尚未计算，不预告一定改善。完整规则见下一项研究方向.md。", "",
        "## 本次核对的范围", "",
        f"已核四情景前五及主历史最高方案，共{verification['selected_named_candidates']}个命名候选、{verification['actual_saved_ledgers_checked']}条保存账户。"
        "日期、终点开盘、现金、份额、分红应收、财富变化、净收益、夏普、费用和相对买入持有年化收益均可复算。"
        "其中114与117四对账户逐列相同。其他保存指标按来源清单汇总，本次没有重验全部旧模型的独立性，也没有重跑任何账户。", "",
        "这项诊断不计作第131轮，不改变原411个不同设置、427个已评价来源版本或1742条主评价记录；旧失败及未定义状态均保留。无需GPT数值包或ZIP。", ""])
    document.write_text("\n".join(lines), encoding="utf-8")
    result.update(status="SAVED_FRONTIER_AND_TOP_ACCOUNTS_CHECKED_DIAGNOSTIC_ONLY", verified_at=verification["verified_at"],
        verified_saved_ledgers=verification["actual_saved_ledgers_checked"], best_four_scenario_comparison="ENTRY_VINTAGE_EXIT",
        minimum_sharpe_best_saved=.7660394266347471, next_method="ONE_FIXED_VINTAGE_REFERENCE_AND_ROLLING_RISK_COMBINATION_NOT_REGISTERED")
    write_json(OUT / "result.json", result)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in {".csv", ".json"}:
            shutil.copy2(p, destination / p.name)
    next_path = ROOT / "docs/510300_VINTAGE_REFERENCE_RISK_NEXT_20260909.md"
    shutil.copy2(next_path, destination / "下一项研究方向.md")
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 130, "候选诊断不能覆盖新的研究索引")
    index.update(updated_at=now(), latest_saved_candidate_frontier=str((OUT / "result.json").relative_to(ROOT)),
        next_work={"status": "VINTAGE_REFERENCE_ROLLING_RISK_SINGLE_COMBINATION_PLANNED", "focus": "第128轮已保存研究参考意图与第109轮已有普通波动乘数的单一组合",
            "source": str(next_path.relative_to(ROOT)), "candidate_round": 131, "registered": False, "new_models_or_accounts": 0})
    index["deliveries"].append({"created_at": now(), "type": "SAVED_FOUR_SCENARIO_DIAGNOSTIC_THROUGH130", "rounds": [], "directory": str(destination),
        "main_document": str(document), "new_gpt_review_archive_created": False})
    write_json(index_path, index)
    print(json.dumps({"document": str(document), "count_unchanged": index["count_warning"], "next": index["next_work"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
