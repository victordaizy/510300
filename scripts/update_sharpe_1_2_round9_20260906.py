"""承接第九轮实测、失败归因和金融及公募申购原始来源修复。"""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import now, require


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    current = read(path)
    study = "510300_CONTEXTUAL_EXPERT_TRACKING_V1"
    require(len(current["completed_rounds"]) == 8 and all(x["study"] != study for x in current["completed_rounds"]), "索引已经变化，不覆盖")
    folder = ROOT / "reports/research/510300_contextual_expert_tracking_v1"
    result = read(folder / "result.json")
    receipt = read(ROOT / "deliverables/510300夏普1.2持续研究_第九轮动态机制组合_GPT审阅_20260906.delivery.json")
    base, stress = [next(x for x in result["primary"] if x["cost"] == cost) for cost in ("BASE", "STRESS")]
    current["completed_rounds"].append({"round": 9, "study": study, "title": "按市场状态持续学习的机制组合",
        "status": result["status"], "result": "reports/research/510300_contextual_expert_tracking_v1/result.json",
        "candidate_configurations": 11, "evaluation_accounts": 24, "mechanism_shadow_accounts_excluded_from_evaluation_count": 6,
        "primary_base": base, "primary_stress": stress, "post_selected_best_base": result["post_selected_best_base"],
        "source_coverage_note": "使用原24个底层输出按机制分组，未重训；没有加入尚未补齐的完整指数盈利、公募申购等新资料。"})
    require(sum(x["candidate_configurations"] for x in current["completed_rounds"]) == 154, "登记配置合计不符")
    require(sum(x["evaluation_accounts"] for x in current["completed_rounds"]) == 326, "评价账户合计不符")
    current.update({"updated_at": now(), "goal_achieved": False,
                    "status": "NINE_ROUNDS_COMPLETED_SOURCE_CORRECTION_AND_ORIGINAL_SUBSCRIPTION_RESEARCH_CONTINUE_TARGET_NOT_MET",
                    "registered_configurations_in_this_resumption": 154, "evaluation_accounts_in_this_resumption": 326,
                    "independent_high_sharpe_evidence": "NOT_ESTABLISHED", "automation_status": "ACTIVE",
                    "failure_attribution": "reports/research/510300_eight_round_failure_attribution_20260906/result.json",
                    "latest_continuation_note": "docs/510300_ROUND9_CONTINUATION_20260906.md"})
    previous_flow = next(x for x in current["running_studies"] if x["study"] == "510300_FUNDAMENTAL_AND_FUND_FLOW_REBUILD_V1")
    bank = read(ROOT / "reports/research/510300_bank_financial_scope_adapter_v1/result.json")
    current["running_studies"] = [
        {"study": "510300_ORIGINAL_EARNINGS_SOURCE_COMPLETION_V1", "status": "PARTIAL_BANK_SCOPE_AND_PERIOD_ADAPTER_VERIFIED_SOURCE_COVERAGE_CONTINUES",
         "path": "reports/research/510300_original_earnings_source_completion_v1", "missing_original_documents_in_initial_queue": 2686,
         "affected_symbols_in_initial_queue": 193, "first_batch_archived": 24,
         "new_adapter": "reports/research/510300_bank_financial_scope_adapter_v1",
         "documents_with_supported_layout": bank["documents_with_supported_layout"], "new_adapter_fact_rows": bank["fact_rows"],
         "remaining_work": "已检查24份，2份符合明确合并累计版式、8项数值。其余22份扩展银行、券商、保险版式。原非金融解析器直接用于银行会混淆母公司/合并与单季/累计，该迁移错误未进入策略。核实原始去年同期、普通股EPS与其他权益工具扣除，继续补完整指数盈利。"},
        previous_flow,
        {"study": "510300_ORIGINAL_FUND_SUBSCRIPTION_REPORTS_V1", "status": "ONE_ORIGINAL_QUARTERLY_SHARE_FLOW_EXAMPLE_VERIFIED_HISTORY_COLLECTION_CONTINUES",
         "path": "reports/research/510300_original_fund_subscription_reports_v1", "unique_original_reports": 1,
         "byte_identical_official_archives": 2, "period": "2021Q3", "net_subscription_units": -1312200000.0,
         "original_publication_catalog_record_verified": False, "source_admitted_for_portfolio": False,
         "remaining_work": "扩展510300原始季报目录与份额申赎历史，核对首次公开、修订、基金身份、折算及期初期末衔接。份额不等于实际人民币现金，单只ETF不等于全市场公募。"}]
    current["deliveries"].append({"rounds": [9], "zip": "deliverables/510300夏普1.2持续研究_第九轮动态机制组合_GPT审阅_20260906.zip",
        "md": "deliverables/510300第九轮_动态机制组合与失败归因_20260906/第九轮结果与全部中文因子规则.md",
        "bytes": receipt["bytes"], "sha256": receipt["sha256"], "members": receipt["members"]})
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 夏普1.2持续研究：截至第九轮", "",
             "累计154个登记配置、326条完整评价账户，包含重复对照，不能解释为154种独立机制。没有候选完整区间达到夏普1.2；目标保持未完成，研究继续。第九轮另有6条机制学习用模拟账户，未计入326条。", "",
             "| 轮次 | 研究机制 | 预设主方案基础夏普 | 主方案压力夏普 | 本轮事后最高夏普 | 候选数 | 完整评价账户数 |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    frames = []
    for row in current["completed_rounds"]:
        lines.append(f"| {row['round']} | {row['title']} | {row['primary_base']['net_sharpe']:.4f} | {row['primary_stress']['net_sharpe']:.4f} | {row['post_selected_best_base']['net_sharpe']:.4f} | {row['candidate_configurations']} | {row['evaluation_accounts']} |")
        frame = pd.read_csv((ROOT / row["result"]).parent / "metrics.csv")
        frame.insert(0, "研究名称", row["title"])
        frame.insert(0, "轮次", row["round"])
        frames.append(frame)
    require(sum(len(f) for f in frames) == 326, "实际账户表行数不符")
    lines += ["", "目前事后最高仍为第六轮月末月初方案，完整账户夏普0.5196；只有某一分期超过1.2，后两期明显不足，不能宣布达标。所有历史已经被反复观察，独立高夏普证据未建立。", "",
              "第九轮将现金、买入持有、趋势、反转、趋势震荡混合和预测模型作为六种机制，按四种市场状态持续更新权重。主方案夏普0.1578、年化收益1.05%、最大回撤30.02%，低于固定等权0.2069和买入持有0.2879。复杂切换没有改善该批输入的表现。", "",
              "八轮失败归因显示：沿原份额路径加回费用后各轮主方案仍全部低于1.2；27个选择区间的过去排名与下一阶段排名平均相关负0.0398，选出的前三名11次超过全体均值；24个旧候选日收益两两相关中位数0.5221。它们是既有资料的诊断，不能直接解释成统计显著的普遍无效。", "",
              "原始数据修复继续。首批24份金融报告中，2份通过明确的合并累计表头和会计等式识别，形成8项数值。已拦住非金融解析器迁移到银行时把母公司单季利润误标合并累计的问题。这个错误没有进入新策略，旧非金融研究本来排除了金融公司。", "",
              "公募新增一条实际来源：510300原始2021三季报的申购、赎回、期初期末份额已经核对，上交所与巨潮存档完全一致；净赎回13.122亿份，不能称为13.122亿元。继续补季度历史和首次公开、修订记录；77份协会月报的原始时钟路线也继续。单只ETF的份额不能代替全市场申购。", "",
              "完整中文说明、逐日机制权重图、全部24条新账户、旧八轮失败归因输入和原始来源在第九轮GPT审阅ZIP。仅作结构和必要数值核对，未做额外安全审计，未上传或获得外部审阅。所有研究仍仅限510300与现金，免费来源；没有真实交易权限。"]
    (ROOT / "deliverables/510300夏普1.2持续研究_截至第九轮_20260906.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(ROOT / "deliverables/510300夏普1.2持续研究_九轮完整指标_20260906.csv", index=False, encoding="utf-8-sig")
    note = """# 第九轮后的继续执行记录

先读最新索引，不重复运行已完成冻结研究。第九轮510300_CONTEXTUAL_EXPERT_TRACKING_V1已完成11候选、24评价账户，另6条机制学习用账户。冻结清单SHA256为53f563d894128182cf5a938fb3fe4b952e13ba27d506beafa61f37b0b9208dc8。主方案基础夏普0.157758、压力0.121538；固定等权事后最好0.206900。全部9轮154配置326评价账户均未到1.2，总体事后最高仍第6轮0.519648。目标active，未完成，不能因本轮结束而停止实质研究。

第九轮结果、全部24账本、6学习账本、2823个更新时点、16938行权重均已保存并核验。五项策略针对性测试通过；本轮严格为11种候选加买入持有对照，两档费用24账户。不要把现金无定义夏普改成零。旧模型不重训。ZIP为deliverables/510300夏普1.2持续研究_第九轮动态机制组合_GPT审阅_20260906.zip，中文报告和两幅图在同名展开目录。已经交付，不重打包。

八轮失败归因在reports/research/510300_eight_round_failure_attribution_20260906。固定原成交份额加回费用，八轮主方案仍低于1.2，不是可执行零费用反事实。27个过去504日排名与下阶段日均收益排名平均相关为负0.0397946，选中前三11次胜全体均值，未证明排名延续。旧24策略日收益相关中位数0.522111、协方差参与率2.405726；后者不是字面上的独立策略数量。R8不重叠60日预测仅9个，不能过度解读。所有旧结果保留，不用新组合追认旧失败。

金融来源优先继续reports/research/510300_original_earnings_source_completion_v1。2686份初始缺口、193家公司；首批12家24份官方报告已归档。新research/bank_financial_scope_adapter_v1.py及其协议和测试已冻结、完成24份处理，2份CMB三季报匹配明确版式、8项数值；结果位于reports/research/510300_bank_financial_scope_adapter_v1。四项针对性测试通过。下一步扩展其余22份银行/证券/保险版式，随后补历史同公司同比与新调入成分。不要改已冻结适配器，可登记经来源证据支持的新版本。

关键范围错误：旧非金融extract_metrics_from_page_texts函数直接用于招商银行2021三季报1211361859，会将第20页母公司第三季度营业利润367.63亿元误标合并累计。第18页实际合并1至9月营业利润1165.81亿元，归母净利润936.15亿元，收入2514.10亿元，每股收益3.62元。新适配器按合并表标题、四个年份列、累计/单季期间和单位选列，并核对全部列收入加费用=营业利润、归母加少数=净利润。原非金融研究原本排除金融公司，错误迁移结果未入任何新策略。不能称旧非金融全部事实错误。普通股EPS还需扣优先股股息/永续债利息，不能等同未扣其他权益的归母利润除普通股数。上年比较列不能替代当年原始版本。

公募新路线已取得实质实例：reports/research/510300_original_fund_subscription_reports_v1。510300的2021三季报上交所与巨潮文件字节相同，SHA256为1f00478ba41787744757c3ace5cf322312d93a452013c1e4b55079b17f7030aa。封面送出日2021-10-27，档案地址同日，PDF创建/修改时间2021-10-26。第12页原始份额变动：期初9186787690，申购4397400000，赎回5709600000，期末7874587690，净额负1312200000，份额衔接精确。已查看封面和第12页，并从表格单元格读取。单位是份，不是人民币；拆分格横线原样保留。公募全市场与单只ETF不能替代。

继续收集510300原始季度公告目录及PDF历史，优先使用发行人、上交所或巨潮正式公告；身份须是510300本体，不能把联接基金或其他300ETF误认成本体。当前仅原始一份实例，尚未核实完整历史公告目录与修订，所以 source_admitted_for_portfolio=false。按报告送出日期保守计算的下个开盘2021-10-28只标为暂定，PDF元数据本身不是公布时钟。下一步补原始公告目录、当时公开证据、相邻期份额衔接与折算，形成冻结数据版本后才能研究申赎强度、净份额压力等。不要用季度末基金净值乘净份额称为精确现金流。数据脚本scripts/record_original_fund_subscription_example_20260906.py已完成，不重复覆盖结果。

77份协会月报来源修复仍在reports/research/510300_fundamental_and_fund_flow_rebuild_v1：44份早期现存PDF2023年11月21日生成，不能用这证明原内容变化或原先时钟。保留5处相邻同范围份额差异、2025年11月分类变化，未运行该申购策略。两条公募来源可并行推进，不因一条困难停止全部研究。

当前依据用户授权的research_authority_v6继续实质新机制/修正来源，不无限改同一轮阈值、年份、方向。EPS、PE、分红回购、股本和其他权益同口径，避免恒等变换与重复计回报；逆回购必须全期限实际公告与预告区别，不回到仅七天。成本、完整日期、分红和T+1等沿用原默认账户；其他ETF交易范围未得到新增授权。只用免费来源，不购买、不接券商、不下单，不新建任务或擅自派子代理。每轮中文说明与完整GPT审阅ZIP，只做必要结构与数值检查。

Windows数据真实路径E:\\ResearchData\\New project 8\\data，减少通过C盘联接重复读取。旧索引更新脚本不得覆盖第九轮，读取最新索引后承接下一轮。heartbeat保持ACTIVE并优先信最新索引；只在结果、实质来源修复、失败或确需用户输入时通知。每次续行做实事，目标尚未达不得标完成。
"""
    (ROOT / "docs/510300_ROUND9_CONTINUATION_20260906.md").write_text(note, encoding="utf-8")
    print(json.dumps({"状态": current["status"], "登记配置": 154, "完整评价账户": 326, "第九轮ZIP哈希": receipt["sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
