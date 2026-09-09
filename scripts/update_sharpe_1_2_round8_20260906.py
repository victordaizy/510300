"""将第八轮和下一阶段金融原始资料补齐状态写入持续研究索引。"""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import now, require


def main():
    index = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    current = json.loads(index.read_text(encoding="utf-8"))
    study = "510300_ORIGINAL_EARNINGS_BREADTH_V1"
    require(all(x["study"] != study for x in current["completed_rounds"]), "第八轮已经写入索引")
    require(len(current["completed_rounds"]) == 7, "最新索引轮次已变化，需要重新衔接")
    report = ROOT / "reports/research/510300_original_earnings_breadth_v1"
    result = json.loads((report / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((ROOT / "deliverables/510300夏普1.2持续研究_第八轮原始财报_GPT审阅_20260906.delivery.json").read_text(encoding="utf-8"))
    base, stress = [next(x for x in result["primary"] if x["cost"] == c) for c in ("BASE", "STRESS")]
    entry = {"round": 8, "study": study, "title": "原始财报盈利改善广度", "status": result["status"],
             "result": (report / "result.json").relative_to(ROOT).as_posix(), "candidate_configurations": 16, "evaluation_accounts": 34,
             "primary_base": base, "primary_stress": stress, "post_selected_best_base": result["post_selected_best_base"],
             "source_coverage_note": "旧库主要为非金融公司，2022年5月才首次有240家覆盖；主方案2024年1月才有成熟训练预测，2025年12月首次持有。"}
    current["completed_rounds"].append(entry)
    current.update({"updated_at": now(), "goal_achieved": False, "status": "EIGHT_ROUNDS_COMPLETED_SOURCE_CORRECTION_AND_NEW_MECHANISMS_CONTINUE_TARGET_NOT_MET",
                    "registered_configurations_in_this_resumption": 143, "evaluation_accounts_in_this_resumption": 302,
                    "independent_high_sharpe_evidence": "NOT_ESTABLISHED", "automation_status": "ACTIVE"})
    current["running_studies"] = [
        {"study": "510300_ORIGINAL_EARNINGS_SOURCE_COMPLETION_V1", "status": "SOURCE_COLLECTION_CONTINUES_ONE_BANK_EPS_EXAMPLE_VERIFIED",
         "path": "reports/research/510300_original_earnings_source_completion_v1", "missing_original_documents": 2686,
         "affected_symbols": 193, "first_batch_archived": 24, "individually_verified_bank_example_documents": 1,
         "individually_verified_bank_example_facts": 6,
         "remaining_work": "核实首批其余金融报告，扩展原始报表覆盖。银行、券商、保险分别定义普通股EPS、归属于普通股的利润、净资产及其他权益工具扣除；不能沿用工业现金流解释。处理新调入成分、原始去年同期、股本变化，再登记新的因子版本和完整账户。"},
        {"study": "510300_FUNDAMENTAL_AND_FUND_FLOW_REBUILD_V1", "status": "SOURCE_RECONSTRUCTION_CONTINUES_44_REGENERATED_PDF_VERSIONS_IDENTIFIED",
         "path": "reports/research/510300_fundamental_and_fund_flow_rebuild_v1", "source_reports_archived": 77, "structured_category_rows": 154,
         "early_reports_with_pdf_creation_2023_11_21": 44,
         "remaining_work": "恢复公募月报原始发布日期和数值版本；保留2025年11月分类变化及5处同范围相邻月报差异。510300份额需补首次公开时钟、拆分和修订。估值与股东回报需原始股本、分红和回购记录，避免普通股EPS口径与其他权益收益混淆。"}]
    current["deliveries"].append({"rounds": [8], "zip": "deliverables/510300夏普1.2持续研究_第八轮原始财报_GPT审阅_20260906.zip",
                                 "md": "deliverables/510300第八轮_原始财报盈利广度_20260906/第八轮结果与全部中文因子规则.md",
                                 "bytes": receipt["bytes"], "sha256": receipt["sha256"], "members": receipt["members"]})
    current["latest_continuation_note"] = "docs/510300_ROUND8_CONTINUATION_20260906.md"
    current["bank_eps_source_example"] = "reports/research/510300_original_earnings_source_completion_v1/金融公司每股盈利与股东回报_原始报告口径实例.md"
    index.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 夏普1.2持续研究：截至第八轮", "", "累计143个登记配置、302条完整评价账户，包含重复对照，不等于143种独立机制。没有候选完整区间达到夏普1.2，持续研究继续。", "",
             "| 轮次 | 研究机制 | 预设主方案基础夏普 | 主方案压力夏普 | 本轮事后最高夏普 | 候选数 | 完整账户数 |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    frames = []
    for item in current["completed_rounds"]:
        lines.append(f"| {item['round']} | {item['title']} | {item['primary_base']['net_sharpe']:.4f} | {item['primary_stress']['net_sharpe']:.4f} | {item['post_selected_best_base']['net_sharpe']:.4f} | {item['candidate_configurations']} | {item['evaluation_accounts']} |")
        frame = pd.read_csv((ROOT / item["result"]).parent / "metrics.csv")
        frame.insert(0, "研究名称", item["title"])
        frame.insert(0, "轮次", item["round"])
        frames.append(frame)
    lines += ["", "全部历史中事后最高仍为第六轮月末月初方案，完整账户夏普0.5196。它某一分期超过1.2，但后续两期明显不足，不能挑单段宣布达标。", "",
              "第七轮已经补上逆回购全期限公开量，保留常规实际公告、买断式月报和预告的区别；旧七天研究失败不能外推到总量。第八轮实际检验原始盈利改善广度，主方案夏普0.0913，资料覆盖限制较重。不同轮次信息覆盖和训练支持不同，不能仅凭跨轮排名解释因子增益。", "",
              "当前继续两条具体工作：补齐金融公司等2686份原始财报缺口，首批24份已归档；恢复77份公募月报原始时钟，44份早期PDF存在后续重新生成迹象。招商银行原始报告的普通股盈利、其他权益扣除与单季累计列已经核实一例，下一版将据此改进盈利因子定义。", "",
              "第八轮中文结果与规则：510300第八轮_原始财报盈利广度_20260906/第八轮结果与全部中文因子规则.md。相应GPT审阅ZIP保存全部34账户、冻结资料、原始财务事实、公募月报及首批金融PDF。", "",
              "仅研究与模拟，不连接券商或产生真实订单；免费来源、完整日期、真实分红和两档费用保持一致。"]
    (ROOT / "deliverables/510300夏普1.2持续研究_截至第八轮_20260906.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(ROOT / "deliverables/510300夏普1.2持续研究_八轮完整指标_20260906.csv", index=False, encoding="utf-8-sig")
    note = """# 第八轮后的继续执行记录

读取最新索引，承接本任务；第八轮已完成，不重训、不覆盖、不重新打包。第八轮冻结清单哈希3514436e4b162755dbb065f762f07682f4b53218ad9edfde7f140b9f7104e565，16候选34账户。主方案基础夏普0.091286，压力0.087205，本轮最好为价格岭回归六十日0.419925；全八轮最高仍为第六轮0.519648。独立高夏普证据未建立。

原始财报广度使用11891个公告的五项财务事实，精确对应主文件或已确认同日配套文件。五项选择59454条，59416条原文数字与单位换算通过，38条空白或不可直接解析缺失，3条未解决文件关联的总资产事实缺失。总体旧覆盖失败结论保留，不能说旧财报库全部错误。金融公司此前按原课题定义预先排除，覆盖到2022年5月才首次达240家，主方案2024年1月才首次有足够成熟训练，2025年12月才首次买入。不要通过把240改成220救回同一轮。

继续从 reports/research/510300_original_earnings_source_completion_v1 开始。缺口清单2686份、193家公司，已按受影响成分股日数选12家公司共24份，免费官方PDF全部归档于 data/raw/510300_original_earnings_source_completion_v1，前十二页文字在batch_01_records。首批选择与结果已保存，不能当成已全部核实的财务数值。

已核实招商银行2021三季报第2页一例，bank_eps_source_example_1211361859.json有6个字段及原始表列。普通股EPS3.62元是1至9月累计，同一行首列1.27元是单季度；归母利润936.15亿元与普通股盈利口径还差优先股股息及永续债利息扣除。经营现金流下降受存款变化影响，不能简单套工业企业质量因子。原始页面图和中文实例在同目录；本例不是通用银行解析器验证，也未进入第八轮冻结包中已完成的策略。

下一步核实首批其余报表，按银行、券商、保险不同财务含义处理普通股盈利、净资产、股本和其他权益扣除，补旧同比与新调入成分。EPS、PE、股东回报必须同口径并避免回购重复记收益。原始财报有实质补齐后可注册新数据版本研究，保持旧回测和失败记录；不要无限改窗口、符号和门槛。市场价与EPS乘PE为同口径关系，本身不是新增预测。

公募从 reports/research/510300_fundamental_and_fund_flow_rebuild_v1 继续。77份月报已结构化，44份早期PDF在元数据中显示2023年11月21日重新生成；元数据本身不能证明首发日期，也不能证明原表已经修改。继续寻找当时原始公告或逐字段吻合的同期公开证据。保留2025年11月分类切换与5处同范围相邻月报份额版本差异，不把资产规模变化或份额变化当作精确净现金申购。此线未运行策略。

每次续行要完成实质来源修复或可执行的新机制。沿用免费来源和510300与现金研究，真实交易未授权。每轮完整中文规则、全部账户和GPT审阅ZIP，只有必要结构和数值检查。data真实路径为 E:\\ResearchData\\New project 8\\data，避免C盘长路径大量重复小文件读取。最新索引优先；不要运行旧六轮更新脚本覆盖最新八轮。
"""
    (ROOT / "docs/510300_ROUND8_CONTINUATION_20260906.md").write_text(note, encoding="utf-8")
    print(json.dumps({"状态": current["status"], "登记配置": 143, "完整账户": 302, "第八轮ZIP": receipt["sha256"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
