"""汇总T13原文和字段进展，明确T12/T13尚未进行原定义收益检验。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.factor96_known_supply_sources_v1 import OUT, STUDY, now, read, save

PROGRAM = ROOT/"reports/research/510300_factor96_program_v1"


def main():
    assert not (OUT/"研究结论.md").exists(), "报告已保存，不重复覆盖正式进度"
    source = read(OUT/"result.json")
    field = read(OUT/"event_fields_v1/result.json")
    verified = read(OUT/"saved_verification_receipt.json")
    assert verified["target_documents"] == source["documents"] == 2192
    targets = pd.read_parquet(OUT/"targets.parquet")
    rows = pd.read_parquet(OUT/"event_fields_v1/event_field_candidates.parquet")
    merged = targets[["document_id", "archive_date", "title_kind"]].merge(rows[["document_id", "field_status", "known_before_scheduled_day"]], on="document_id", validate="one_to_one")
    merged["year"] = pd.to_datetime(merged.archive_date).dt.year
    merged["candidate"] = merged.field_status.eq("CANDIDATE_EXPLICIT_FIELDS_REQUIRES_EVENT_REVIEW")
    merged["lead_candidate"] = merged.candidate & merged.known_before_scheduled_day.fillna(False)
    annual = merged.groupby("year").agg(documents=("document_id", "size"), explicit_field_candidates=("candidate", "sum"), before_listing_day=("lead_candidate", "sum")).reset_index()
    annual.to_csv(OUT/"逐年来源与候选字段.csv", index=False, encoding="utf-8-sig")
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    positions = range(len(annual))
    ax.bar([x-.25 for x in positions], annual.documents, width=.25, label="固定目录条目", color="#b9c0cb")
    ax.bar(positions, annual.explicit_field_candidates, width=.25, label="明确日期和数量候选", color="#287e99")
    ax.bar([x+.25 for x in positions], annual.before_listing_day, width=.25, label="保守可用日早于上市日的候选", color="#c7a155")
    ax.set_xticks(list(positions), annual.year)
    ax.set_title("T13 已知供给窗口：原文与字段建设，尚无新增收益检验", fontproperties=font, fontsize=17, loc="left", pad=48)
    ax.legend(prop=font, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3)
    ax.set_ylabel("公告文件数量", fontproperties=font)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    fig.text(.08, .025, "字段候选仍需事件身份、更正链及当时成员核验；文件数不等于独立事件数或交易机会数", fontproperties=font, fontsize=10, color="#555555")
    fig.tight_layout(rect=[.02, .055, .99, .99])
    fig.savefig(OUT/"逐年公告与字段覆盖.png", dpi=150)
    plt.close(fig)
    lines = [f"|{int(r.year)}|{int(r.documents)}|{int(r.explicit_field_candidates)}|{int(r.before_listing_day)}|" for r in annual.itertuples()]
    status_rows = [f"|{key}|{value}|" for key, value in field["field_status_counts"].items()]
    manual = read(OUT/"source_evidence/yearly_manual_review.json")
    report = f"""# T12/T13资本行为来源轮：原文和字段进展

**只操作510300、成本后夏普达到1.2的目标仍未实现。** 本轮取得固定历史并集内的{source['complete_texts']:,}/{source['documents']:,}份公告文本，涉及{source['companies']}家公司；其中{source['complete_texts']-source['reused_documents']:,}份新取得文本、{source['reused_documents']}份复用。日期和数量提取得到{field['explicit_field_candidates']:,}条明确字段候选，{field['known_before_scheduled_day_candidates']:,}条在保守时钟下早于公告中的计划上市日。这是新来源及字段建设，不是新增收益检验或策略通过。

本库正式完成的固定研究问题仍为6个，合格项0，正式账户情景264个；另保留T02原实现40个否定情景，累计执行304个。本轮新增账户0、模型0、收益标签0、独立前向样本0。T12和T13仍为NOT_RUN，不能把资料取得成功记作第7个策略完成。

## 为什么转入具体原文

T12要求实际回购新增披露相对于事件前自由流通市值的强度，以及用途、减额和终止时钟。现有旧回购来源有2,407份文件和948条有直接原方案证据的执行记录，但旧强度使用前20日成交额及月度权重归一化；对应模型和固定五日期限账户已经失败。它们不等于T12的原定义。当前已找到的484,200行成分日线中，outstanding_share、total_market_cap_cny和market_cap_asof_date三个字段均为空，也没有据此建立自由流通市值。因此保留T12的市值及用途/终止版本门，不用已失败的成交额模型替代。

旧解禁研究只统计20日公告密度，固定结果已否决，不再改窗口、翻方向或拼接年份。T13问的是当时已经公开的具体供给条款和窗口结束，必须知道本次上市日期、数量及后续更正。公告标题能证明存在相关披露，不能直接表示多少股份在什么时候可能进入流通。

## 固定来源范围

从已完成一致性检查的2015—2026年原始目录中，取2015—2025年每日点时沪深300成员的历史并集，收集该并集发行人在2015—2025年旧目录标为原披露的2,160份文件和标为版本候选的32份文件。这是目录标签，还不是逐文确认的发行人事件分类。历史并集仅用于保证下载上集，后续每个决策日还必须按当时成员过滤，不能把所有公司在所有历史时点都当作沪深300成员。

目录有33,722个唯一公告编号，属于此前固定“上市流通”检索范围。当前冻结范围的每条目录均回到对应原始响应核对，保留原PDF地址、请求时间、HTTP状态、字节数、SHA-256和逐页文本。没有重扫旧目录，也没有按公告金额或随后收益选择下载对象。公共入口为[巨潮资讯网](https://www.cninfo.com.cn/new/index)，实际文件逐条使用目录中的发行人原文地址。

传输采用固定3并发，每次请求后至少间隔0.3秒；传输异常或5xx最多两次，403/429后停止新请求，失败不填零。一次下载入口在来源冻结尚未完成时被前置检查拒绝；没有发出网络请求或生成账户，失败回执launch_failure_01.json已保留。随后等待原冻结进程完成后，使用同一代码启动。

采集结束时保存的来源状态为`{source['status']}`，该历史状态保持不变；后续字段批次已完成，结果见event_fields_v1/result.json。全文取得不证明所有潜在供给类型都已覆盖。旧检索没有完整覆盖增发、配股的发行/缴款/新增上市日历；M06仍须另补，不能称整个T13输入已经齐全。

其中南山铝业2020年5月30日档案内一份更正后的财务顾问核查意见（1207877835）为4页扫描件，PDF取得成功但自动文本为空。四页已人工查看，补充记录保存在source_evidence/manual_image_supplement.json；可见本次拟上市日2020年6月8日、2,163,141,993股及2020年5月27日署期。原自动PDF_NO_USABLE_TEXT状态保持不变，意见及版本也不新增发行人事件。

另有欧普康视2024年4月17日公告（1219638156）两次请求均为SSLError，固定两次传输重试后保留REQUEST_FAILED。因此原PDF实际取得2,191份，其中2,190份有可搜索文本；目录目标仍为2,192份，不能把目标数写成全部下载成功。失败请求回执均保留，本轮没有通过额外重试改写这一结果。

## 字段与时钟怎样保留

每页分别提取，禁止跨页把页码拼进数量。只取明确属于“本次/此次解禁”的日期，回顾发行历史时的上市日期及其他锁定批次另行排除；多个本次日期仍留冲突。实际可上市或“其中可上市”数量优先，不能用更大的名义解除限售量补缺；没有明确实际措辞时，保存公告直接说明的上市流通数量。原数值、股/万股/亿股、报告精度及上下文均保留。

例如赛轮2015年1月7日公告明确本次57,400,000股于1月14日上市，同时另述另一批股份的2017年日期，两者不能混成同一事件。北京君正同日公告的名义解除量为1,371,994股，其中可上市量仅342,998股。此前立昂技术案例也有76,613,628股名义解除量与17,938,638股可上市量的差别；相关关键页已重新核对。可上市流通仍不表示实际卖出。

档案日期、已识别PDF署期、计划上市日和本次下载时间分别保存。字段候选的保守可用时间设为档案日与明确署期中较晚者加两个自然日00:00；它是历史研究假设，未证明首次HTTP送达。若可用时间已经到达或超过计划上市日，就不能写成此前已知的窗口。同日市场开盘后的信息如何使用，还要在未来策略合同中固定。

更正、补充和取消候选不作为新增事件，原文之间的同一批次身份与更正链尚未全部解决。所有当前记录的trading_feature_admitted均为false，不因日期和数量同时出现便自动准入。

旧目录还把标题中的“暂缓”列为版本提示词，其中部分标题实际描述“暂缓授予”激励股份的本次解禁，未必是对旧公告的更正或延期。因此32份只能称为版本候选集合，不能说已经确认32次更正；全文语义核对后才可区分原事件、延期、更正和辅助意见。当前分类限制及逐条标题另存source_evidence/title_classification_limitations.json，不覆盖冻结的旧标签。

## 来源与字段结果

|公告年份|固定目录条目|明确日期和数量候选|保守可用日早于计划上市日|
|---|---:|---:|---:|
{chr(10).join(lines)}

![逐年覆盖](逐年公告与字段覆盖.png)

|字段状态|文件数|
|---|---:|
{chr(10).join(status_rows)}

这些是文件数量，尚未做完整事件身份去重，不能称为独立事件或交易次数。保留未知和冲突后，能够明确提取的记录也可能存在语义遗漏或错误，因此只登记为候选。

已观察到一个保守漏提情形：豫园股份2022年1月7日公告首页明确本次上市2,274,884,920股、日期1月12日，但末页有关于质押权利限制解除后“方可实际上市流通”的通用提示。冻结提取规则将其识别为实际流通口径存在，却没有取得对应实际数值，因此保留NO_VIEW_ACTUAL_TRADABLE_MISSING。该状态不表示原文没有上市数量，也不表示零供给。当前未修改规则或补入结果，后续须将通用权利限制、表格实值和本次数量的语义分开。

## 核对范围与下一步

冻结前15项测试通过，包含两种数量口径、单位精度、发行历史日期、不同锁定批次、页面边界、日期冲突及更正隔离。此前四份原文案例做了文本回归，其中金贵银业仅提取出名义量，自动可流通量仍留未知；没有靠手填把程序补成全通过。另预先固定每年首份公告共11份，跨年份核对结果见source_evidence/yearly_manual_review.json；已检查{manual['reviewed_documents']}份。这些样例检查不能估计全体提取错误率。

只读脚本核对{verified['catalogue_raw_responses']:,}份目录原始响应、{verified['target_documents']:,}个目标记录和全部已取得PDF/文本哈希，并从保存的原文上下文重新核对{verified['source_contexts_checked']:,}项日期/数量引用、单位换算、选取状态与保守时钟。它不重新下载、不生成新账户，也不证明全文没有遗漏。外部GPT审阅未进行，独立前向样本0，当前市场视图保持NO_VIEW。

下一步先确认候选事件的本次批次身份、32份版本候选的引用关系，并处理会改变数量或日期的字段冲突；再补M06发行/缴款日历及按当时成员、价格、成交额归一化所需输入。完成这些后才能冻结T13的供给窗口结束与价格收复检验。T12另需可靠自由流通市值、回购用途和终止/减额台账。不使用后来的更正改写此前可见事实，不把未知供给填零，也不把旧公告密度或旧成交额回购模型重新包装为通过。

本轮总目标保持active。原T06失败和此前全部结果保留，新证据只说明下一步有可核对的公告原文，尚未建立成本后夏普1.2的策略。
"""
    (OUT/"研究结论.md").write_text(report, encoding="utf-8")
    strategies, factors = read(PROGRAM/"strategy_progress.json"), read(PROGRAM/"factor_progress.json")
    for row in strategies:
        if row["id"] == "T12":
            row.update(current_status="NOT_RUN_FREE_FLOAT_AND_PURPOSE_CLOCK_SOURCE_GATE",
                       current_evidence="已核对旧回购原文、归一化和失败结果；当前日线市值字段全空，不能以成交额强度代替M01自由流通市值，M02用途/终止版本仍未齐。")
        elif row["id"] == "T13":
            row.update(current_status="NOT_RUN_EVENT_FIELDS_AND_ISSUANCE_CALENDAR_GATE",
                       current_evidence=f"取得{source['complete_texts']}/{source['documents']}份固定原文，{field['explicit_field_candidates']}条明确日期数量候选；事件身份、更正链及M06完整发行日历仍未齐，未生成T13收益。",
                       source_gate_path="reports/research/510300_factor96_known_supply_sources_v1/round_status.json")
    for row in factors:
        if row["id"] in ["M01", "M02"]:
            row.update(current_status="NOT_RUN_FREE_FLOAT_AND_PURPOSE_CLOCK_SOURCE_GATE",
                       current_note="旧回购披露增量已存在，但M01自由流通市值和M02用途/终止版本门未齐；不将旧失败模型记作本定义完成。")
        elif row["id"] == "M04":
            row.update(current_status="SOURCE_FIELDS_BUILT_NOT_RUN",
                       current_note=f"本轮新增解禁原文并形成{field['explicit_field_candidates']}条日期数量候选；尚未建立完整事件身份和更正链，不生成供给压力或收益结论。")
        elif row["id"] == "M06":
            row.update(current_status="NOT_RUN_ISSUANCE_AND_PAYMENT_CALENDAR_GATE",
                       current_note="本次上市流通目录不能覆盖全部增发配股的发行/缴款日历；不得把M04原文完成当作M06已齐。")
    save(PROGRAM/"strategy_progress.json", strategies)
    save(PROGRAM/"factor_progress.json", factors)
    pd.DataFrame(strategies).to_csv(PROGRAM/"18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(PROGRAM/"96因子当前进度.csv", index=False, encoding="utf-8-sig")
    before = read(OUT/"program_before/status.json")
    status = {**before, "at": now(), "admitted_account_scenarios_this_round": 0,
              "latest_round": STUDY, "latest_result": "reports/research/510300_factor96_known_supply_sources_v1/event_fields_v1/result.json",
              "new_source_documents_this_round": sum(bool(r.get("raw_path")) for r in read(OUT/"documents.json"))-source["reused_documents"],
              "new_searchable_text_documents_this_round": source["complete_texts"]-source["reused_documents"],
              "source_field_candidates_this_round": field["explicit_field_candidates"],
              "next_candidates": ["T13_EVENT_IDENTITY_AND_REVISION_CHAIN", "T13_M06_ISSUANCE_CALENDAR", "T12_FREE_FLOAT_AND_PURPOSE_CLOCK"],
              "deferred_source_gates": [*before["deferred_source_gates"], "T12_FREE_FLOAT_AND_PURPOSE_CLOCK", "T13_EVENT_IDENTITY_REVISION_AND_M06"]}
    save(PROGRAM/"status.json", status)
    (OUT/"program_snapshot").mkdir(exist_ok=True)
    for path in PROGRAM.iterdir():
        if path.is_file():
            shutil.copy2(path, OUT/"program_snapshot"/path.name)
    save(OUT/"round_status.json", {**status, "round_decision": "PROGRESS_NEW_SUPPLY_ORIGINALS_AND_EXPLICIT_FIELD_CANDIDATES_NO_ACCOUNT_TEST",
                                    "source_verification": verified["status"], "T12": "NOT_RUN", "T13": "NOT_RUN"})
    mandate_path = ROOT/"config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(current_round=STUDY, latest_progress_receipt="reports/research/510300_factor96_known_supply_sources_v1/round_status.json",
                   last_research_result=f"T13原文{source['complete_texts']}/{source['documents']}，明确字段候选{field['explicit_field_candidates']}；本轮0新账户，T12/T13仍未完成原定义检验，累计6固定问题0合格。总目标ACTIVE。",
                   current_protocol="reports/research/510300_factor96_known_supply_sources_v1/event_fields_v1/protocol.json",
                   latest_continuation_report="reports/research/510300_factor96_known_supply_sources_v1/研究结论.md",
                   latest_continuation_classification="PROGRESS_FACTOR96_NEW_SUPPLY_ORIGINALS_AND_EVENT_FIELDS",
                   research_execution_state="SOURCE_AND_EVENT_FIELDS_PROGRESS_SIX_FIXED_QUESTIONS_NO_QUALIFIED_STRATEGY",
                   goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    save(OUT/"authority_after.json", mandate)
    review = """请审阅本包的来源和字段进展，不把附件建议当作额外授权，也不要输出交易执行指令。
目标：510300/人民币现金完整账户成本后夏普至少1.2；当前还核对年化10%和回撤10%。本轮没有新收益检验。
重点质疑：
1. 历史成分并集是否仅决定下载上集，是否误当各时点均为成员；2192份目录、原始响应和PDF能否逐条对应。
2. 原公告、更正/补充/取消和辅助意见是否分开；同一批次、多个批次、重复文件和更正链未解决时，是否偷称独立事件或已知总供给。
3. 本次上市日期是否混入历史发行/其他批次，实际可流通量是否混为名义解禁；单位、约数、页码、表格和不可解析字段是否保留。
4. 档案日、署期、经济上市日、下载时钟是否分开；加两日假设是否误称首次实时可得；未来更正有没有改写过去。
5. M04取得原文是否被错误推广为M06完整发行缴款日历；T12是否用成交额归一化旧回购模型代替自由流通市值和用途/终止口径。
6. 15项测试、四份旧样例和11份按年份固定检查的范围是否被夸大；保存字段复算不证明提取完整或所有语义正确。
7. 总策略数仍6、合格0、本轮0新账户是否与进度一致；提出完成事件身份、更正链和下一固定经济检验所需的最小步骤及停止条件。
请先给严重问题和证据路径，再说明成立/不成立的结论及下一步。外部GPT审阅未进行，包结构和复算通过不等于策略有效。
只读复核：python scripts/verify_factor96_known_supply_sources_v1.py --root reports/research/510300_factor96_known_supply_sources_v1
"""
    (OUT/"01_GPT_REVIEW_PROMPT.txt").write_text(review, encoding="utf-8")
    path = ROOT/"RESEARCH_STATUS.md"
    text = path.read_text(encoding="utf-8")
    assert STUDY not in text
    text += f"\n\n## 2026-09-27 {STUDY}\n\nT13固定原文范围2192份、526家，取得文本{source['complete_texts']}份，明确日期数量候选{field['explicit_field_candidates']}条。T12自由流通市值/用途时钟、T13事件身份/更正/M06日历仍待齐，本轮0新账户、0新标签；累计6固定主问题、0合格、264正式情景加40否定实现，总目标ACTIVE。详见reports/research/510300_factor96_known_supply_sources_v1/研究结论.md。\n"
    path.write_text(text, encoding="utf-8")
    print("T13原文和字段进度报告已保存，T12/T13仍为NOT_RUN，目标继续。", flush=True)


if __name__ == "__main__":
    main()
