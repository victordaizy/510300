"""交付官方追溯目录、已记录宣布与阶段研究的来源边界。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from research import official_policy_catalog_format_repair_v1_0_1 as repair
from research import write_entry_information_sequence_report_v1 as render

ROOT, OUT, SOURCE, parent = repair.ROOT, repair.OUT, repair.SOURCE, repair.parent


def read_table(name):
    return pd.read_parquet(OUT / "results" / (name + ".parquet"))


def timing_figure(chain, logical):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    original = pd.read_parquet(repair.original.previous.OUT / "results/全部3488进入信息身份_公布钟与量价行业.parquet")
    window = original.loc[original.date.between("2024-09-23", "2024-09-30")]
    rate = window.loc[window.policy_rate_first_observed_here & window.rate.eq(1.5)]
    weekly = window.loc[window.weekly_hist.gt(0)]
    pmi = window.loc[window.orders_publication_since_previous_decision]
    if len(rate) != 1 or len(pmi) != 1 or weekly.empty:
        raise ValueError("原2024慢指标可观察时点改变。")
    nodes = chain.set_index("node_id")
    marks = [
        ("已核对宣布：降准降息/住房支持", nodes.loc["R01", "source_available_upper"], "R01/H02 09-24 09:19:36"),
        ("已核对工具：补充金额", nodes.loc["C01", "source_available_upper"], "C01 09-24 11:42:50"),
        ("原链登记的次开执行日", "2024-09-25T09:30:00+08:00", "09-25；只是原链研究口径"),
        ("实施确认：日期级来源日终上界", nodes.loc["R02", "source_available_upper"], "R02 09-27；首发时点未认证"),
        ("原PMI更新可观察", pmi.orders_available_at.iloc[0], "09-30 09:30"),
        ("原利率表首次观察1.5%", rate.decision_time.iloc[0], "09-30 16:00；首次观察不等于首次宣布"),
        ("上一完整周MACD正值可观察", weekly.decision_time.iloc[0], "09-30 16:00；沿用原周线供给"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
    colors = ["#126f69", "#126f69", "#777777", "#bf7632", "#35698c", "#35698c", "#35698c"]
    for y, ((label, at, note), color) in enumerate(zip(marks, colors)):
        clock = pd.Timestamp(at).tz_convert("Asia/Shanghai").tz_localize(None)
        axes[0].scatter(clock, y, color=color, s=45)
        axes[0].annotate(note, (clock, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=color)
    axes[0].set_yticks(range(len(marks)), [mark[0] for mark in marks]); axes[0].invert_yaxis()
    axes[0].set_xlim(pd.Timestamp("2024-09-23"), pd.Timestamp("2024-10-05"))
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=2)); axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].set_title("2024年：宣布、实施、资料首次观察和慢指标确认是不同的时点", fontsize=12)
    dates = logical.loc[logical.catalog_event_date.eq(pd.Timestamp("2019-01-04")) & logical.navigation_tool_tags.str.contains("准备金率")]
    if dates.empty:
        raise ValueError("2019降准追溯母集没有原记录。")
    first = dates.earliest_retrospective_publication.min()
    marks2 = [("原六次降准研究的日期级上界", pd.Timestamp("2019-01-04 23:59:59"), "01-04日终；不冒充认证首发"),
        ("原阶段修复信号", pd.Timestamp("2019-01-09 16:00:00"), "01-09；沿用原信号，不生成新策略"),
        ("本固定目录最早回顾记录公开", first.tz_convert("Asia/Shanghai").tz_localize(None), str(first.date())+"；不能倒填01-04")]
    for y, (label, at, note) in enumerate(marks2):
        axes[1].scatter(at, y, color="#35698c" if y < 2 else "#bf7632", s=45)
        axes[1].annotate(note, (at, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9)
    axes[1].set_yticks(range(3), [mark[0] for mark in marks2]); axes[1].invert_yaxis()
    axes[1].set_xlim(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-07-01"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator()); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].set_title("2019年：回顾材料能用于追溯，公布时间不能提前到事件日", fontsize=12)
    for ax in axes:
        ax.grid(axis="x", alpha=.18)
        ax.set_ylim(ax.get_ylim()[0]+.2, ax.get_ylim()[1]-.2)
    fig.suptitle("官方来源时钟与原点位：日周线研究，没有分钟行情\n24条人工核对链不代表全部新闻；原来源首版及最早公开仍未认证", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .92])
    path = OUT / "figures/2019与2024_宣布实施与慢确认时钟.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("官方目录已交付，不覆盖。")
    summary = parent.read(OUT / "summary.json")
    initial = parent.read(SOURCE / "summary.json")
    manifest = parent.read(SOURCE / "source_manifest.json")
    if summary["all_catalog_rows"] != 88 or summary["source_instance_records"] != 1814 or summary["new_requests"] != 0:
        raise ValueError("格式处理的实际完整范围不匹配。")
    logical = read_table("全部逻辑记录_所有源实例与日期边界")
    all_rows = read_table("全部日期或月份条目_精度未知保留")
    statuses = read_table("七原文格式处理_失败与月份未知保留")
    chain = pd.read_parquet(SOURCE / "results/原24人工核对链_原复核与执行时钟保持.parquet")
    key_rows = pd.read_parquet(SOURCE / "results/全部19关键点_原链可知钟与两假启动.parquet")
    events = pd.read_parquet(SOURCE / "results/全部143进入点_已记录宣布与资料覆盖区别.parquet")
    figure = timing_figure(chain, logical)
    source_years = []
    for year in range(2015, 2027):
        documents = [doc for doc in manifest["documents"] if doc["catalog_year"] == year]
        ids = {doc["catalog_id"] for doc in documents}
        rows = all_rows.loc[all_rows.catalog_id.isin(ids)]
        source_years.append({"年": year, "固定源文": len(documents), "原生或探测成功": sum(doc["status"].startswith("FETCHED") for doc in documents),
            "失败": sum(doc["status"] == "FETCH_FAILED" for doc in documents), "源条目": len(rows),
            "具体日未知条目": int(rows.catalog_event_date.isna().sum()), "完整公告全集": "未建立"})
    failures = [doc for doc in manifest["documents"] if doc["status"] == "FETCH_FAILED"]
    if len(failures) != 1 or initial["original_events_with_recorded_new_chain"] != 1:
        raise ValueError("原生失败或全事件原链覆盖改变。")
    source_case_rows = []
    for start, end, name in [("2015-06-20", "2015-07-01", "2015反弹"), ("2017-03-10", "2017-03-20", "2017利率背景"),
            ("2019-01-01", "2019-01-25", "2019修复"), ("2020-03-20", "2020-04-15", "2020修复"),
            ("2024-09-20", "2024-09-30", "2024重新定价"), ("2026-04-20", "2026-05-12", "2026假启动")]:
        selected = logical.loc[logical.catalog_event_date.between(start, end)]
        source_case_rows.extend({"固定案例": name, **row} for row in selected.to_dict("records"))
    repair.table("六原案例对应期间_全部目录条目包括其他", pd.DataFrame(source_case_rows))
    diagnostic = {"at": parent.original.now(), "decision": "TECH.R228", "implementation": "1.0.1",
        "retrospective_index_complete_88": True, "retrospective_source_documents_fixed": 46,
        "source_documents_received": 45, "one_original_tls_failure_kept": failures[0]["catalog_id"],
        "original_index_parse_failure_kept": "INDEX_4旧标题缺中国字样；原68母目录结果保持，新格式处理中补齐88。",
        "retrospective_logical_records_are_independent_policy_events": False,
        "recorded_chain_new_event_count": int(events.recorded_new_chain_nodes.gt(0).sum()),
        "recorded_chain_new_event_date": "2024-09-24", "recorded_chain_node_ids": "H02|R01|C01",
        "recorded_nodes_share_one_source_not_three_independent_news": True,
        "2024_09_24_chronology_entries": int(logical.catalog_event_date.eq(pd.Timestamp("2024-09-24")).sum()),
        "2020_03_30_chronology_entries": int(logical.catalog_event_date.eq(pd.Timestamp("2020-03-30")).sum()),
        "chronology_can_replace_original_announcements": False, "direct_same_day_policy_entry_gate_admitted": False,
        "new_accounts": 0, "new_fits": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "next_source_scope": "原25个利率记录身份、完整准备金率导航45逻辑记录、资本市场工具3目录记录与原24核对链；逐条追溯宣布/实施、同源合并、全部缺失保持，不能按原盈亏选源。",
        "next_scope_is_policy_catalog_only": False, "next_financial_run": "NOT_ADMITTED_NOT_RUN"}
    parent.write(OUT / "post_run_diagnosis.json", diagnostic)
    text = "# 510300 官方政策宣布来源：完整目录、时钟与真实缺口\n\n"
    text += "TECH.R227登记、R228结果及源文格式实现1.0.1，2026-10-06。结论：官方追溯母目录已补成完整88入口，固定46源文中45收到，整理1814源实例/1143规范全文逻辑记录。**回顾目录不能代替全部原宣布来源；现有24核对链也不能当全国新闻全集。**下一用途要把真实宣布/实施、工具类型与同源信息分别记录，未准入新金融规则。\n\n"
    text += "6原必要测试、19整段因果前缀、3格式必要测试通过。49登记后原生请求、两已保存结构原文复用，另有一次只读未留正文探测；含先前3次原生探测总52请求。格式处理0新请求/链重算/金融；原68目录、1790源记录及全部失败不覆盖。数据和机制推进是本轮成果，收益夏普仍未达，最新金融R224固定行业失效拒绝保持。\n\n"
    text += "## 来源范围与实际结果\n\n"
    text += "固定的是央行货币政策大事记栏目当前五页88入口，目标年份2015—2025全部目录及2026参考期至6月30日的两个目录。累计和季度版本都保留，不挑全年/季度更有利文本。2026第三季度没有已发布回顾目录时保持资料缺口。目录完整是固定栏目范围，不是全国全部政策的完整性。\n\n"
    text += render.markdown(pd.DataFrame(source_years)) + "\n\n"
    failure = failures[0]
    text += f"唯一原生失败是[{failure['title']}]({failure['url']})，TLS EOF，没有响应原文、不重试。2019全年回顾可作其他独立来源，但没有用它改写该失败。原生45成功中的字体拆日期在2016第三季度原文按完整段落重连，恢复18条日期实例。2022四版本‘3月下旬’、2017两版本‘1月’共6实例/3逻辑记录的具体日仍未知；仅保留期间边界，不补日。\n\n"
    text += "源文格式只处理旧题目少‘中国’字样和文本结构。原1464条未受影响源实例逐列精确保持，46源文集合/顺序不改；原INDEX_4失败和原解析状态仍保留。原源与程序校验保持，原24链、原19前缀不重算。\n\n"
    text += "## 导航记录不等于独立事件或利好评分\n\n"
    text += render.markdown(read_table("全部工具导航_非经济方向"), {"tool": "工具导航", "source_instances": "源实例", "logical_records": "规范全文逻辑记录"}) + "\n\n"
    text += "逻辑身份按事件日与规范全文建立，重复523逻辑记录、额外671源实例全保留。不同版本只写‘人民银行’或‘中国人民银行’仍可有不同全文身份，1143不是独立政策事件数；同一公告分拆的多个节点也不是多条独立确认。工具标签只是导航，持平、收紧、放松均包含；存贷款基准利率、住房条件等可能位于‘其他’，全部‘其他’也保留供追溯，不能只从七标签推全国政策强弱。\n\n"
    text += "## 对原具体上涨与反例的实质解释\n\n"
    text += "**2019年修复。**固定母目录确实有1月4日宣布降准1个百分点、1月15/25分别实施0.5个百分点并不续做一季度到期MLF的条目，但该条目在本固定目录最早回顾公开为2019-05-24 17:00。原六次降准研究已保存政府网1月4日报道及日终上界，应复用该原报道，不从5月材料把源钟倒填到1月4日。新目录建立追溯依据，没有重新运行六次降准直接买入表达。\n\n"
    text += "**2024年重新定价。**原直播链R01/H02在09-24 09:19:36上界记录了降准降息计划和住房资金支持比例调整，C01在11:42:50补充互换便利/回购增持再贷款额度。原利率背景仍为7月记录，完全可以与这些已公开宣布同时成立。三节点共享同一直播来源，不是三条独立新闻；5000亿/3000亿额度也不等于当天真实股票购买。实施、申请、中标和股票成交要分开。\n\n"
    text += "大事记09-24有住房首付款及政策期限通知，因全文‘人民银行/中国人民银行’差别保留4逻辑记录；这些不能代替上述三直播节点。降准在09-27实施才记入相应条目，回顾来源晚至11月/次年2月。换成整个大事记仍不足以覆盖当天全部宣布，不能将实施又计为一笔新的20基点意外。\n\n"
    text += "**2015、2017、2020和2026。**2015六月目录記6月28日定向降准实施，本固定最早回顾在8月；不能据实施日期解释6月27日最早宣布。2017-03-16该目录有的是委员会人员调整，不是完整操作利率上调原文；2020-03-30当前目录0条，不代表当天没有利率公告，04-03定向降准条目需追溯原宣布来源。2026-05-11原假启动前没有原24链新节点，只能说这24链未记录，不是没有政策。原首次突破/技术指标向好却亏损的反例不删除。\n\n"
    text += "## 全部进入点的已记录宣布覆盖\n\n"
    text += "全部143原进入事件中，仅2024-09-24与原24链的区间新公布相交，且是同一来源的3节点。原链集中2024—2025，其他年份及多数日期尚未核对。不能把无记录填成‘没有政策’，不能把只有一次交集当高胜率样本，也不能立即把‘当天有新政策’设成进入门。宣布之后的信息怎样延续、传播和失效，需要完整用途另行定义。\n\n"
    fields = ["date", "stage_entry_type", "recorded_new_chain_nodes", "recorded_new_chain_ids", "recorded_latest_chain_clock", "recorded_latest_chain_ids"]
    text += render.markdown(key_rows[fields], {"date": "原观察日", "stage_entry_type": "原路线", "recorded_new_chain_nodes": "区间新记录节点",
        "recorded_new_chain_ids": "节点身份", "recorded_latest_chain_clock": "原链最近保存公开上界", "recorded_latest_chain_ids": "最近节点"}) + "\n\n"
    text += "原3488/143/240和19关键点表中的参考范围覆盖字段沿原初始PARSED定义，仅是回顾范围诊断；v1.0.1精度未知另列，不能把该字段当历史同日公告全集。宏观16:00仅描述当时已保存上界可知，原链15:00复核、次开规则仍原样保留；15:00之后发布不能解释已经形成的当日收盘价格。\n\n"
    text += f"![宣布实施与慢确认时钟]({figure.absolute().as_posix()})\n\n"
    text += "## 下一项具体用途与停止条件\n\n"
    text += "下一步固定合并来源：原25个已观察利率记录身份、完整准备金率导航45逻辑记录、资本市场工具3目录记录和原24核对链，逐条追溯官方宣布、实施及更新节点。25不是25次政策变化、45不是45次降准，3也不是3个独立冲击；先消除工具/日期/同源重复歧义，再形成可核对母集。相同全文、同一发布会多节点按事件和共同来源保存关系，不简单相加评分。旧SCIO两年完整490入口及相关原文可复用；需补操作公告和发布会来源，不能只依赖大事记。所有缺失/未核对保持，不能按原赢亏只找成功来源。\n\n"
    text += "来源可知钟早于拟入场、宣布/实施分清后，才登记修复、重新定价、延续与失效的完整机制，在相同20万元/成本/风险预算下检验收益、夏普、p×B与标准净期望。若只有一两个已知成功政策阶段、覆盖无法核对或日期精度不能支持用途，不准入完整金融；不拿来源数/绿色测试当收益证据，不通过改天数、符号或费用救旧失败。新的历史原文仍不是独立未来样本，原13前瞻保持。\n\n"
    text += "直接资料：[央行货币政策大事记栏目](https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125963/index.html)、[2024年央行大事记](https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125963/2025091218344837008/index.html)、[2024-09-24官方直播](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)。全部源URL/原生原文/失败回执位于上级source_manifest与receipts，原政策490目录/24链研究及六降准金融终态不覆盖。\n"
    path = OUT / "官方政策目录_具体上涨时钟与来源缺口.md"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    parent.write(OUT / "delivery_receipt.json", {"at": parent.original.now(), "report": repair.original.relative(path),
        "report_sha256": parent.digest(path), "figure": repair.original.relative(figure), "figure_sha256": parent.digest(figure),
        "all_catalog_rows": 88, "selected_source_documents": 46, "source_instance_records": 1814, "logical_records": 1143,
        "one_tls_failure_retained": True, "six_exact_day_unknown_rows_retained": True, "new_accounts": 0, "goal_achieved": False})
    print("R228官方来源完整报告完成：88目录/46计划源文、全部时期及原19点位，宣布与回顾边界保留；零新金融。", flush=True)


if __name__ == "__main__":
    main()
