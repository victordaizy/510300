"""将实际分类来源、全部案例与失败写成可阅读的研究报告。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from research import official_industry_publication_intake_v1 as intake

ROOT, OUT = intake.ROOT, intake.OUT
RESULTS = OUT / "implementation_v1_0_1"


def main():
    report = OUT / "官方分类_案例覆盖与下一实验.md"
    if report.exists():
        raise RuntimeError("本阶段报告已存在，不重复生成。")
    received = intake.parent.read(OUT / "summary.json")
    result = intake.parent.read(RESULTS / "summary.json")
    catalog = intake.parent.read(OUT / "complete_catalog_publication_index.json")
    service = intake.parent.read(OUT / "goal_service_status_after_result.json")
    if service["goal"]["status"] != "active" or result["new_accounts"] != 0 or result["pages_read"] != 702:
        raise ValueError("实际来源结果或目标状态不一致。")
    case = pd.read_parquet(RESULTS / "results/九固定案例公布钟与300成员分类覆盖.parquet")
    members = pd.read_parquet(RESULTS / "results/九案例全部已取得成员_已知与未知保留.parquet")
    unknown = pd.read_parquet(RESULTS / "results/原件全部未知分类_保持原五证券行.parquet")
    industry = pd.read_parquet(RESULTS / "results/九案例全部已知行业_成员数量非指数权重.parquet")
    if len(case) != 9 or len(members) != 2400 or len(unknown) != 5 or len(industry) != 221:
        raise ValueError("保存结果数量与实际完成汇总不一致。")
    snapshots = []
    source_urls = {node["id"]: node["url"] for node in received["nodes"]}
    for row in result["parsed"]:
        snapshots.append(f"| [{row['id']}]({source_urls[row['id']]}) | {row['published']} | {row['pages']} | {row['rows']} | {row['unknown_classification_rows']} | {'通过' if row['parse_passed'] else '失败，原缺码保留'} |")
    cases = []
    views = {"NO_VIEW_MEMBERSHIP_SOURCE_MISSING": "成员源缺失，NO_VIEW", "NO_VIEW_LATEST_PUBLISHED_SNAPSHOT_PARSE_FAILED": "最新快照解析失败，NO_VIEW",
        "PARTIAL_MEMBER_CLASSIFICATION_UNKNOWN": "有未知成员，逐行保留", "CASE_MEMBERS_CLASSIFIED": "本案例成员全部已知"}
    for row in case.to_dict("records"):
        coverage = f"{row['classified_members']}/{row['members']}" if row["membership_view_allowed"] else "未知分母，不计算覆盖"
        cases.append(f"| {row['decision_date']:%Y-%m-%d} | {row['membership_source_date']:%Y-%m-%d} | {row['latest_published_snapshot_id']} | {coverage} | {views.get(row['view_state'], row['view_state'])} |")
    failures = [f"| {row.snapshot_id} | {row.symbol} | {row.pdf_page} | {row.row_origin} | {row.section_name} | {row.major_code} |" for row in unknown.itertuples()]
    first_cases = industry.loc[industry.decision_date.eq(pd.Timestamp("2024-09-24"))].sort_values(["member_count", "industry_key"], ascending=[False, True]).head(8)
    count_rows = [f"| {row.industry_key} | {row.major_name} | {row.member_count} | {row.count_share_of_300:.2%} |" for row in first_cases.itertuples()]
    text = """# 官方行业分类：案例覆盖、源限制与下一实验

2026-10-05，TECH.R217—R218。目标是继续寻找能提高510300完整账户收益和夏普的事前信息，当前阶段只完成新来源和固定案例覆盖。最新实际金融仍是TECH.R216的拒绝结果，未产生新策略收益、胜率、盈亏比或夏普，均为NOT_COMPUTED；目标服务实际active、完整目标未达。

## 为什么研究行业主线

招商的主线、资金、博弈、周期，以及国金的主线与兴业的轮动研究，提出一个值得检验的问题：同样的ETF价格转强或转弱，内部行业是在少数方向集中、向更广范围传播，还是轮动失序。上一实验中，固定成分价格组失效与同覆盖ETF价格退出四场景完全相同，不能再用同义投票增加指标。

行业结构可以成为不同观察层，但行业主线的超额收益不能记作510300收益。小资金较低容量约束可以帮助选择机会，却不能由此推出正期望。本项目仍只研究510300.SH日线、上一完整周和CASH_CNY，行业只用于解释与信号，未取得其他资产交易权限。

分阶段策略可以改变入场、持有与退出：早期重新定价不必等待全部慢变量转好，延续阶段则需要持续证据。是否有用必须由完整账户检验，不能由四个上涨案例决定；已失败的R212分阶段政策与R216固定组失效保持冻结。

## 实际来源取得与全部解析

本机20次固定逻辑GET全部HTTP成功，0请求异常、0重定向：8带公布日期官方页面、8PDF、3目录、1方法页。浏览工具此前页面超时/附件访问限制与本机成功分开保存，不能将浏览工具失败写成官方原件缺失。

八份PDF共702页，全文证券代码集合与解析集合均一致、没有重复证券，共32,223个快照证券行。这些是跨快照分类记录，不是32,223家不同公司。严格的完整快照门要求每一证券分类均可明确，因此5份通过、3份因5个缺码行失败。

| 快照与公布页面 | 实际公布日 | 页数 | 证券行 | 分类未知行 | 原完整快照解析门 |
|---|---|---:|---:|---:|---|
""" + "\n".join(snapshots) + """

CSRC季度分类与CAPCO2023半年度分类保留各自版本。它们不是原申万31行业代理的修复，也不等于沪深300行业权重。公布时间只有日期时，统一用北京时间当天23:59；不得用报告覆盖期末提前赋值。CAPCO规则见[2023年行业统计分类指引通知](https://www.capco.org.cn/xhdt/tzgg/202305/20230521/j_2023052117544500016846630061707656.html)。

## 九个固定案例：缺失和失败全部保留

每个观察收盘只用ETF日历前一完整源日成员。该源日成员不足300时保持成员源未知，不将0行解释成覆盖率0或100%；已知最新快照失败时不回退旧快照。九日期在来源用途登记前固定。

| 观察日 | 成员前一源日 | 当时已公布的所选最新快照 | 分类覆盖 | 状态 |
|---|---|---|---|---|
""" + "\n".join(cases) + """

保存2400个实际取得成员行：1498个分类已知、902个未知。首个2015-01-05案例缺2014-12-31成员源，单列未知，不虚造300行。2015-06-29唯一未分类600958.SH，2025-07-03唯一未分类001391.SZ；不从下一快照或当前网页倒补，也不推测缺失原因。

2024-09-24与09-30收盘仍用2023H2；2024H1虽在09-30公布，23:59保守钟晚于当日15:00，至10-08可用。三个2024案例均能明确300成员分类。这只证明案例源覆盖，不证明分类变化预测收益。

2024-09-24前一源日的部分成员数量分布如下。它可帮助明确研究对象，**成员数量占比不是指数权重、收益贡献、资金净流入或产业原因**。

| 分类代码 | 官方大类名称 | 成员数 | 占300成员数量 |
|---|---|---:|---:|
""" + "\n".join(count_rows) + """

## 五个原件缺码与一次汇总实现失败

| 快照 | 证券 | PDF页 | 页:表:行 | 原可见门类文本 | 原大类代码 |
|---|---|---:|---|---|---|
""" + "\n".join(failures) + """

已查看原件首页8幅和缺码相关页3幅。旧单行门类名称不完整，画面中也没有用于行解析的括号门类代码；不依据证券名称、后来归属或收益推测补码。五行保留原始单元格、页面与行出处，原5通过/3失败不改变。这个门针对整份快照，即使缺码行不是本次成员，也不能改口径将原快照写成通过；未来若使用单行证据，应另立明确用途，保留原完整快照裁决。

原解析完成八PDF后，在首个0成员案例误对没有industry_key的空表归组，终止为KeyError。原代码、启动标记、八份解析和失败回执保留。隔离v1_0_1只修正空成员汇总，从已保存结果完成一次九案例汇总；0重新GET、0重新解析PDF、0改分类代码/钟/版本/质量门。原6必要测试已通过，另2必要回归通过；54个原冻结/复用文件哈希保持。此修复不是调整策略或补救历史收益。

## 还缺什么，下一步怎样做

当前三个官方保存目录列出36个2014Q3以来发布节点：29季度、7半年度。8份已收到、28份尚未收到，逐次页面链接和目录日期均已建立索引。目录未列出的2021Q4/2022来源仍未知，不宣称不存在，也不以未来分类填补。当前2023以后PDF链接含2026年迁移目录，因此页面的历史公布日期不认证当前PDF等于历史首版；first_vintage=NOT_CERTIFIED。

下一完整来源用途应先固定36节点、复用已收到8份、对28未收到页面及其明确附件各有限请求，全部失败与源钟保留。已收到的原件不再重复请求。随后按逐行来源版本和原成员生成全日历分类可知状态，先报告覆盖与陈旧程度，尚未构造或测试新交易收益。2026H1于2026-09-30公布，不用于现有同日收盘历史终点。

来源合格后，再登记一个完整的行业传播用途：先解释原上涨及失败，比较固定行业集合的相对价格强弱、参与范围及轮动变化，研究它们能否区分ETF同样走弱后的恢复和继续下跌。不得先看到收益再挑行业、窗口、阈值或只保留2024。指数权重或指数贡献没有合格源时保持NOT_COMPUTED；成员等权观察须明确标注。

完整金融检验应包含原A、相同进入/退出的价格对照、相同来源覆盖对照，采用冻结资金/费用/风险/次开成交/T+1/股息。联合报告净CAGR、净夏普、回撤、完整次数、pB、最坏周期与集中度。所有既有历史为开发材料，真正独立验证和去过拟合尚未建立。R216所发现的信息重复、早退截断恢复和费用事实是反例，继续保留。

## 可复核材料

- [有限来源协议](protocol.json)、[本机请求与原件哈希](summary.json)。
- [原隔离解析协议](parser_protocol.json)、[原部分失败](initial_parser_failure_receipt.json)、[空成员修复协议](implementation_v1_0_1/protocol.json)。
- [完整实际汇总](implementation_v1_0_1/summary.json)、[九案例源钟和覆盖CSV](implementation_v1_0_1/results/九固定案例公布钟与300成员分类覆盖.csv)。
- [所有保存目录发布节点](complete_catalog_publication_index.json)、[目标服务实际返回](goal_service_status_after_result.json)。
- 上一真实金融：[R216完整结果与反例](../510300_broker_cohort_failure_v1/真实进入后固定组失效_完整结果与反例.md)。
- 券商来源与机制：[4家7份阅读及四段量价解释](../510300_broker_cycle_inspiration_v1/券商周期框架_研究启发与具体上涨解释.md)。
"""
    with report.open("x", encoding="utf-8") as stream:
        stream.write(text)
    figures = [OUT / "figures" / f"{node['id']}_原件第一页.png" for node in received["nodes"]]
    figures += [OUT / "figures" / "2018Q3_原件第2页.png", OUT / "figures" / "2018Q3_原件第83页.png", OUT / "figures" / "2019Q1_原件第84页.png"]
    if not all(path.exists() for path in figures):
        raise ValueError("实际已查看原件图路径缺失。")
    intake.parent.write(OUT / "source_figure_view_receipt.json", {"at": intake.parent.original.now(),
        "viewed": len(figures), "first_pages": 8, "failure_diagnostic_pages": 3,
        "images": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": intake.parent.digest(path)} for path in figures],
        "quality_conclusion": "原件缺码不补，原5通过/3失败保留；不是策略效果检查。"})
    intake.parent.write(OUT / "delivery_receipt.json", {"at": intake.parent.original.now(),
        "report": str(report.absolute().relative_to(ROOT)), "report_sha256": intake.parent.digest(report),
        "nine_saved_case_rows_read": len(case), "all_saved_member_rows_read": len(members),
        "saved_unknown_classification_rows_read": len(unknown), "saved_industry_count_rows_read": len(industry),
        "catalog_nodes_indexed": len(catalog["rows"]), "new_accounts": 0, "goal_achieved": False})
    print("实际来源报告完成：九案例、五原缺码、36发布节点、0新金融全部保留。", flush=True)


if __name__ == "__main__":
    main()
