"""汇总真实全目录来源与全日历覆盖，为下一行业结构观察提供事实。"""
from __future__ import annotations

import pandas as pd

from research import official_industry_full_sequence_intake_v1 as intake

ROOT, OUT, parent = intake.ROOT, intake.OUT, intake.parent
RESULTS = OUT / "implementation_v1_0_1"


def main():
    report = OUT / "行业主线_全发布来源与日历覆盖.md"
    if report.exists():
        raise RuntimeError("本全来源报告已存在，不重复生成。")
    received = parent.read(OUT / "summary.json")
    summary = parent.read(RESULTS / "summary.json")
    daily = pd.read_parquet(RESULTS / "results/全部3488日官方发布钟_成员覆盖_源龄与未知.parquet")
    yearly = pd.read_parquet(RESULTS / "results/全部年份源覆盖与完整日历分母.parquet")
    cases = pd.read_parquet(RESULTS / "results/原九固定案例全发布序列逐行覆盖.parquet")
    metadata = pd.read_parquet(RESULTS / "results/全部36发布节点_原完整门与逐行来源结构.parquet")
    evaluation = daily.loc[daily.evaluation_calendar]
    clocks = daily.available_at.notna() & daily.available_at.gt(daily.decision_at)
    if clocks.any() or len(daily) != 3488 or len(evaluation) != 2855 or len(cases) != 9:
        raise ValueError("保存日历、案例或公布钟不一致。")
    latest = daily.loc[daily.date.eq(pd.Timestamp("2026-09-30"))].iloc[0]
    if latest.snapshot_id != "2025H2" or cases.loc[cases.date.eq(pd.Timestamp("2024-09-30")), "snapshot_id"].iloc[0] != "2023H2":
        raise ValueError("同日15:00观察提前使用23:59新发布。")
    year_rows = []
    for row in yearly.itertuples():
        year_rows.append(f"| {row.year} | {row.calendar_days} | {row.membership_known_days} | {row.all_300_known_days} | {row.mean_member_coverage:.2%} | {int(row.source_age_max_days)} | {row.source_age_over_one_year_days} |")
    case_rows = []
    for row in cases.itertuples():
        coverage = f"{row.classified_members}/{row.members}" if row.membership_view_allowed else "分母未知"
        case_rows.append(f"| {row.date:%Y-%m-%d} | {row.snapshot_id} | {coverage} | {int(row.source_age_calendar_days)} |")
    blocked_rows = []
    for row in metadata.loc[~metadata.row_source_snapshot_eligible].itertuples():
        span = evaluation.loc[evaluation.snapshot_id.eq(row.id)]
        period = f"{span.date.min():%Y-%m-%d}至{span.date.max():%Y-%m-%d}" if len(span) else "不覆盖评价日历"
        reason = "PDF原表重复证券，不能确定唯一行" if row.pdf_received else "原件或公布页面请求失败"
        blocked_rows.append(f"| {row.id} | {row.published} | {reason} | {period} | {len(span)} |")
    unknown_by_state = evaluation.loc[evaluation.classified_members.eq(0)].groupby("view_state").size().to_dict()
    source_urls = {node["id"]: node["url"] for node in received["nodes"]}
    figures = [OUT / "figures" / f"{key}_原件第一页.png" for key in ("2014Q4", "2017Q1", "2021Q2", "2026H1")]
    figures.append(OUT / "figures/2017Q1_原件第14页重复证券.png")
    if not all(path.exists() for path in figures):
        raise ValueError("已查看的原件图缺失。")
    text = f"""# 行业主线研究：已保存官方目录的全发布来源与日历覆盖

2026-10-05，TECH.R219—R220。本轮完成不同信息来源的全日历基础，最新实际金融仍TECH.R216拒绝；没有新账户、拟合、未来标签或行情日线，收益/夏普NOT_COMPUTED，完整目标未达。下一步进入行业结构的数值观察，不能由来源成功直接推出收益提升。

## 这轮解决了什么

上一轮仅八个案例快照，三个旧整表因非明确行被完整性门挡住。新用途在取得其余来源前固定两个字段：完整快照仍要求全部证券分类明确；逐行用途要求原标题、证券集合和唯一身份结构合格，并只使用有明确分类与行出处的证券行。个别未知证券仍保留，不从未来或旧快照补分类。原R218三个整表失败结果完全保留。

这项区别产生实际可用性变化：2019-01-08、2019-06-19、2020-04-01三个原案例，在新的逐行用途下均能明确300个成员的分类；原缺码证券仍未知，没有补写它们的代码。允许这一用途不代表允许降低交易指标门槛或改动旧策略。

## 实际取得与解析

固定36目录发布节点，原8PDF及解析复用。新增54逻辑GET：51成功、3失败、0重定向/HTTP失败。新增25份PDF、2177页；合计33原件、2879页、124926个跨快照分类行。原件/公布页面失败节点为2019Q2、2020Q1、2021Q1，终态保留，不重试本用途。目录未列出的2021Q4/2022仍UNKNOWN，不能宣称不存在。

33原件中19份完整快照门通过；32份满足逐行结构资格。2017Q1原PDF第14页连续把石大胜华、科森科技记为同一603026，证券身份重复，因此该快照逐行结构也不合格。原PDF画面已查看，不根据证券名称推测改码。[该原公布页面]({source_urls['2017Q1']})。

该节点及三个未取得节点按最新已公布的目录位置保留为未知，不回退较老快照：

| 节点 | 目录/页面公布日 | 不合格原因 | 覆盖的评价收盘区间 | 原日历槽 |
|---|---|---|---|---:|
""" + "\n".join(blocked_rows) + f"""

这四个节点不是四个被删除的样本区间，全部日期仍在3488日表和2855评价分母中。已有成员的分类未知保留为未知，不能给它们默认为0收益、牛市、现金建议或有效空仓策略。

## 全日历覆盖与源龄

共3488原收盘槽，2015-01-05以后2855评价槽；原成员实际取得846900行，764101行分类已知、82799行未知。评价槽中1561个300成员全部已知、995个部分已知、299个0已知。0已知包括成员源缺失{unknown_by_state.get('NO_VIEW_MEMBERSHIP_SOURCE_MISSING', 0)}槽、最新节点结构失败/缺原件{unknown_by_state.get('NO_VIEW_LATEST_SOURCE_STRUCTURAL_FAILURE', 0)}槽。原2026-08-14之后成员缺口不补。

| 年份 | 全评价日历 | 成员源完整日 | 300分类全知日 | 成员源完整日的平均分类覆盖 | 最大源龄/自然日 | 源龄超过365日日历槽 |
|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(year_rows) + """

平均覆盖这一列只以成员源完整日为分母，包含当日原件结构失败产生的0覆盖；不能替代全日历可用性。2026为截至09-30的部分年度，不按全年外推。

最重要限制是306个评价槽分类源龄超过365日，最大820日。2022—2023使用的是当时已公布的旧分类记录；当前取得它也未认证历史首版。因此“表内有明确分类”不等于“已验证公司当时业务归属一直未变”。源龄逐日实报，365日仅诊断标记，没有按结果引入年龄过滤策略。

CSRC季度与CAPCO2023半年度保留版本。数量、行业等权结构和指数权重是不同对象；本轮121039行业数量行不是指数收益贡献、产业原因或资金流。指数权重和行业贡献仍未计算。

## 原九案例及公布钟

| 观察收盘 | 最新已公布节点 | 逐行成员分类覆盖 | 原件源龄/自然日 |
|---|---|---|---:|
""" + "\n".join(case_rows) + """

公布钟统一为实际公布日北京时间23:59，全部3488槽核对均无晚于15:00观察的使用。2024-09-30收盘仍用2023H2；2026H1虽于2026-09-30公布，历史同日终点仍用2025H2，不能将2026H1提前。[2026H1公布页面](""" + source_urls["2026H1"] + """）。2015首案例没有前一成员源，两个299/300案例中的未知证券仍不倒补。

## 实现失败与修复范围

原冻结解析全部25新PDF已经完成，但在存储元数据时混合旧JSON日期字符串和新Timestamp，ArrowTypeError终止。原代码、启动与失败回执、25解析全部保留。v1_0_1只统一available_at带北京时间时区的ns存储类型，从已保存25新/8旧解析一次完成全日历汇总；0重新GET/解析，公布时点、分类资格及完整门不改。179个冻结/复用文件哈希保持。

三个逐行必要测试和一个混合时间实际存储回归通过；两个登记前测试失败记录保留，分别为缺失值必须Python None的错误断言、时间单位未明确固定为ns。后者仅明确存储单位，前者只纠正测试的缺失判断，没有收益规则变化。4种原件首页及2017重复证券页共5幅已查看。实际全槽公布钟和两个同日终点核对已完成。

## 下一具体实验

下一用途为行业相对强弱、传播与轮动的数值描述，不是把行业数量作为新投票。先明确固定当时成员/分类的行业集合，使用已经存在、保守滞后一个完整交易日的成分含息收益，观察行业相对ETF的强弱、行业排名是否稳定、上涨是否扩大；未知成员、原件结构失败、旧源龄分别保留。

原四个上涨案例、全部原阶段事件和退出反例同时展示，包括2019恢复被过早截断、2025盈利被截断的案例。行业数据与价格是否提供不同信息需要先核实，避免重演R216四场景完全同义的失效。结构指标的窗口、最小集合和未知分支先登记，不能看新账户结果后选行业或窗口。

之后才登记一个完整金融用途，配原A、ETF价格和相同源覆盖归因对照，使用原资金/费用/风险/次开成交/T+1/股息，全日历联合报告净CAGR、净夏普、实际pB、回撤、次数、最坏周期与集中度。全部旧历史仍开发/校准，真正独立验证和去过拟合未建立。当前没有已准入待跑的新金融候选，也没有收益提升证明。

## 实际材料

- [固定来源协议](protocol.json)、[54新增请求与36节点](summary.json)、[用途卡](../../../docs/510300_OFFICIAL_INDUSTRY_FULL_SEQUENCE_V1.md)。
- [原解析协议](parser_protocol.json)、[原存储失败回执](initial_parser_failure_receipt.json)、[时间存储修复协议](implementation_v1_0_1/protocol.json)。
- [全部实际汇总](implementation_v1_0_1/summary.json)、[全日历覆盖CSV](implementation_v1_0_1/results/全部3488日官方发布钟_成员覆盖_源龄与未知.csv)、[原九案例CSV](implementation_v1_0_1/results/原九固定案例全发布序列逐行覆盖.csv)。
- 大成员表及全部行业数量为parquet，保留来源行出处，避免另复制大CSV。
- 上一金融：[R216完整结果](../510300_broker_cohort_failure_v1/真实进入后固定组失效_完整结果与反例.md)；上一来源：[R218案例报告](../510300_official_industry_publication_intake_v1/官方分类_案例覆盖与下一实验.md)。
"""
    text = text.replace("](" + source_urls["2026H1"] + "）。", "](" + source_urls["2026H1"] + ")。")
    with report.open("x", encoding="utf-8") as stream:
        stream.write(text)
    parent.write(OUT / "delivery_receipt.json", {"at": parent.original.now(), "report": str(report.absolute().relative_to(ROOT)),
        "report_sha256": parent.digest(report), "all_slots_read": len(daily), "evaluation_slots_read": len(evaluation),
        "saved_case_rows_read": len(cases), "all_36_metadata_rows_read": len(metadata),
        "clock_violations": int(clocks.sum()), "same_day_new_publication_excluded": ["2024-09-30", "2026-09-30"],
        "zero_known_by_state": unknown_by_state, "new_accounts": 0})
    parent.write(OUT / "source_figure_view_receipt.json", {"at": parent.original.now(), "viewed": 5,
        "figures": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": parent.digest(path)} for path in figures],
        "source_finding": "2017Q1原PDF中603026连续对应两个名称，保持唯一身份结构失败，不推测改码。"})
    print("全来源报告完成：33原件/3488日、全部缺口与源龄、9案例和后续结构观察问题已保存。", flush=True)


if __name__ == "__main__":
    main()
