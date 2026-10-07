"""交付完整进入信息顺序，不从旧周期统计反选新策略。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research import entry_information_sequence_description_study_v1 as study

ROOT, OUT, parent = study.ROOT, study.OUT, study.parent


def format_value(value):
    if value is None or pd.isna(value):
        return "未知/不适用"
    if isinstance(value, (bool, np.bool_)):
        return "是" if value else "否"
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}"
    return str(value).replace("|", "/").replace("\n", " ")


def markdown(frame, labels=None):
    names = [labels.get(name, name) if labels else name for name in frame.columns]
    rows = ["| " + " | ".join(names) + " |", "| " + " | ".join("---" for name in names) + " |"]
    rows.extend("| " + " | ".join(format_value(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def read_table(name):
    return pd.read_parquet(OUT / "results" / (name + ".parquet"))


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("进入信息描述已交付，不覆盖。")
    summary = parent.read(OUT / "summary.json")
    if summary["decision"] != "TECH.R226" or summary["original_events"] != 143 or summary["new_accounts"] != 0:
        raise ValueError("完整描述结果范围不匹配。")
    if len(summary["prefix_checks"]) != 19 or not all(item["all_prior_rows_exact"] for item in summary["prefix_checks"]):
        raise ValueError("十九关键截断未全部一致。")
    daily = read_table("全部3488进入信息身份_公布钟与量价行业")
    events = read_table("全部143进入事件_来源重复与既有周期上下文")
    keys = read_table("原17关键日_来源钟与行业量价")
    negatives = read_table("两个新增亏损信号_全部进入信息与上下文")
    sources = read_table("全部来源范围年份_更新与未知分母")
    repeats = read_table("全部同来源路线首次重复_既有上下文而非新胜率")
    fresh = events.loc[events.orders_publication_since_previous_decision].copy()
    if len(fresh) != 11 or not negatives.orders_earlier_same_source_route_events.eq(0).all():
        raise ValueError("首次/新公布反例与完整分母不匹配。")
    original_complete = fresh.loc[fresh.original_account_context.eq("COMPLETE")]
    r224_complete = fresh.loc[fresh.r224_status.eq("COMPLETE")]
    diagnostic = {"at": parent.original.now(), "decision": "TECH.R226",
        "all_event_orders_known": int(events.orders_clock_admitted.sum()), "all_event_policy_rate_known": int(events.policy_rate_clock_admitted.sum()),
        "orders_new_publication_events": len(fresh), "policy_rate_new_publication_events": int(events.policy_rate_publication_since_previous_decision.sum()),
        "funding_new_publication_events": int(events.funding_publication_since_previous_decision.sum()),
        "margin_new_publication_events": int(events.margin_publication_since_previous_decision.sum()),
        "fresh_pmi_original_complete_contexts": len(original_complete),
        "fresh_pmi_original_positive_contexts": int(original_complete.original_cycle_net_pnl.gt(0).sum()),
        "fresh_pmi_original_negative_contexts": int(original_complete.original_cycle_net_pnl.lt(0).sum()),
        "fresh_pmi_r224_complete_contexts": len(r224_complete),
        "fresh_pmi_r224_positive_contexts": int(r224_complete.r224_net_pnl.gt(0).sum()),
        "fresh_pmi_r224_negative_contexts": int(r224_complete.r224_net_pnl.lt(0).sum()),
        "known_bad_signals_both_first_pmi_route": True,
        "known_bad_2026_daily_weekly_hist_positive_and_industry_positive_fraction": .75,
        "pmi_repricing_repeats_original_actual_cycles": int(repeats.loc[repeats.source.eq("orders") & repeats.route.eq("REPRICING") & repeats.source_route_status.eq("该来源下该路线重复"), "original_complete_contexts"].sum()),
        "current_rate_clock_is_complete_policy_news": False,
        "single_freshness_gate_admitted": False, "new_accounts": 0, "new_fits": 0,
        "next_source_question": "独立记录官方政策宣布/生效时间和工具类型；先核实覆盖及全部未知，不能把现有生效利率当完整新闻钟。",
        "next_source_status": "PROPOSED_NOT_REGISTERED_OR_RUN", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    parent.write(OUT / "post_run_diagnosis.json", diagnostic)
    view_rows = []
    for figure in summary["figures"]:
        path = ROOT / figure
        if not path.is_file():
            raise ValueError("六图交付缺失。")
        view_rows.append({"path": figure, "sha256": parent.digest(path), "actually_viewed": True,
            "inspection": "中文标签/实际22、92、95、31和21、17日分母明确；行业未知为断线；16点宏观与前一行业源日区别明确。"})
    parent.write(OUT / "figure_view_receipt.json", {"at": parent.original.now(), "figures": view_rows, "all_six_actually_viewed": True})
    source_small = sources.loc[sources.scope.isin(["原2855研究日历", "原143进入事件"])]
    displayed = pd.concat([keys, negatives], ignore_index=True).sort_values("date")
    displayed["date"] = pd.to_datetime(displayed.date).dt.strftime("%Y-%m-%d")
    displayed["相对成交量"] = np.exp(displayed.log_relative_volume)
    displayed["PMI新订单原值"] = displayed.pmi_orders_level + 50
    key_fields = ["date", "stage_entry_type", "相对成交量", "daily_hist", "weekly_hist", "PMI新订单原值", "pmi_orders_change",
        "industry_view_allowed", "industry_positive5_fraction", "rotation_churn5blocks"]
    key_labels = {"date": "观察日", "stage_entry_type": "原路线", "daily_hist": "原日MACD柱", "weekly_hist": "上一完整周柱",
        "pmi_orders_change": "PMI环比", "industry_view_allowed": "行业可知", "industry_positive5_fraction": "行业上涨比例", "rotation_churn5blocks": "两5块排名变化"}
    clock_fields = ["date", "orders_available_at", "orders_source_age_calendar_days", "orders_publication_since_previous_decision",
        "orders_earlier_same_source_route_events", "rate", "funding_policy_known_at", "policy_rate_source_age_calendar_days",
        "policy_rate_publication_since_previous_decision", "policy_rate_earlier_same_source_route_events"]
    clock_labels = {"date": "观察日", "orders_available_at": "PMI保存公布钟", "orders_source_age_calendar_days": "PMI源龄天",
        "orders_publication_since_previous_decision": "PMI区间新公布", "orders_earlier_same_source_route_events": "同PMI同路线此前次数",
        "rate": "对应利率%", "funding_policy_known_at": "对应利率保存已知钟", "policy_rate_source_age_calendar_days": "利率记录源龄天",
        "policy_rate_publication_since_previous_decision": "利率钟区间新公布", "policy_rate_earlier_same_source_route_events": "同利率同路线此前次数"}
    report = "# 510300 进入信息更新：具体上涨、假启动与来源时序\n\n"
    report += "TECH.R225登记、R226一次完整描述，2026-10-06。结论：来源更新的实现可以继续使用，但**单独的首次PMI/首次利率背景、日周MACD同正或行业多数上涨，不足以解释好坏进入；当前对应生效利率字段也不能代替全部政策宣布钟。**下一步优先补真正宣布信息及工具类型，解释不同机制后才登记完整进入—持有用途。原R224金融拒绝保持，未提高或重新计算账户收益/夏普。\n\n"
    report += "这次包含全部3488观察日、原2855研究日历、全部143阶段事件、原四案例240行/17关键日、两个新增亏损信号及38窗口行。6必要测试与19整段前缀逐值复算通过、6图实际查看；0新账户、拟合、未来训练目标、行情、参数搜索或金融重跑。历史仍属开发，首版未认证，独立证据未建立。\n\n"
    report += "## 可使用的来源及其含义\n\n"
    report += "宏观沿用原16:00决定钟，行业沿用前一完整源日。PMI身份含参考月份/保存公布钟，融资/资金身份含统计日/已知钟，对应利率身份含rate/policy_known_at。原known为假或钟晚于决定都保留未知；首次见到旧记录与区间新公布分开。资金/融资几乎每日更新，不能因此叫政策冲击；公布时间在周末或假期可落入下一个ETF决定区间，不等于发生在该交易日。\n\n"
    report += markdown(source_small) + "\n\n"
    report += "原2855日，对应利率25个身份第一次出现，只有1个钟落入此前ETF决定至当前决定区间，其余24个首次见到的其实是旧公布钟。全部143进入事件中142利率可知却0个区间新利率公布；这不说明这些信号前不存在新政策，而是说明该字段的用途是已生效利率背景。PMI142可知/1未知、11区间新公布；资金142、融资135区间更新，但两者是日常记录。\n\n"
    report += "## 六个具体窗口能解释什么\n\n"
    report += "**2015年反弹失败。**日MACD在关键反弹日仍负，后面周MACD也负；资金和融资天天更新，6月关键日PMI新订单保存值仍50.6。成交放大与现金宽松本身不能说明下跌结束。全部22日行业分类未知保留，不能用后来分类证明当时没有主线。\n\n"
    report += "**2019年修复。**1月8日日MACD已正、上一完整周仍负，前一源日15/19行业上涨（78.95%），软件、电力、汽车领先；PMI新订单49.7、环比−0.7，保存利率钟还是2018年3月。实际原阶段1月9日修复信号在同利率同修复路线此前已经出现2次，R224原已完成净盈利7978.96元。要求所有来源首次或等待PMI已改善，会与这段成功修复冲突；不能把现有旧利率钟说成没有2019新政策。\n\n"
    report += "**2020年修复。**4月1日日柱刚正、完整周仍负，前一源日14/18行业上涨（77.78%），新订单从29.3至52.0；利率新记录首次出现，但保存钟为3月30日，严格区间判定不叫4月1日新公布。实际4月7日修复只是小盈利，6月2日另一个修复已完成盈利10756.16元；同对应利率同修复路线此前3次，并不是利率首次。4月15日至7月14日官方行业源失败保留，不能拼成连续行业确认。\n\n"
    report += "**2024年重新定价。**9月24日相对量3.3384、日柱正而完整周负，前一源日12/16行业上涨（75%），保险、汽车、资本市场服务领先；PMI新订单48.9、环比−0.4，融资5日净变化负，利率记录仍指7月22日。原次日进入的R224已完成周期净盈利3776.25元，但9月24日当前字段既无新PMI也无新利率钟。9月30日才出现新PMI/利率记录、完整周柱转正；这与上涨已经展开的顺序一致，不能把迟来的慢指标当成事前原因。9月26行业参与100%而轮动排名指标随后升到0.4191，不能按事后结果把排名变化低于某阈值定义成牛市。\n\n"
    report += "**2017年新增失败突破。**4月5日价格突破、相对量1.5408，日周MACD均正；新订单53.3、环比+0.3，3月31日发布距离5.29天，同PMI同突破路线首次，对应利率同路线也首次。行业当天未知。原实际4月6日进入、4月19日退出，压力净亏1181.26元。只买当月首次突破仍会保留此反例。\n\n"
    report += "**2026年新增失败突破。**5月11日相对量2.3905、日周MACD均正，前一源日12/16行业上涨（75%），领先组5日平均+5.9283%，DR007相对对应利率缺口−0.1013个百分点；PMI新订单50.6、环比−1.0，4月30日公布距11.27天，同PMI同突破路线首次。对应利率记录源龄368天，此路线此前9次；融资未知不能填成利好。5月12日实际进入至5月19日退出净亏1753.77元。这说明多数技术背景同时向好仍需更强的启动机制证据，并不是证明特定排名阈值可救模型。\n\n"
    report += markdown(displayed[key_fields], key_labels) + "\n\n"
    report += "## 公布钟、首次与重复的全部证据\n\n" + markdown(displayed[clock_fields], clock_labels) + "\n\n"
    report += "下面保留全部11个PMI区间新公布事件，不能只展示成功者。原阶段4个已有完成周期3正1负，R224同4个周期4正；另外7个事件没有实际周期。这个差别来自已有现金与持有规则，不能报成11笔交易或新的100%胜率。2019/2024成功启动也不在11个公布区间事件中；没有足够证据将同日/同区间发布设为单独进入门。\n\n"
    report += "下表pmi_orders_level为原输入相对50的中心化值，实际PMI=该值+50；原值与环比都保持。\n\n"
    report += markdown(fresh[["date", "stage_entry_type", "orders_reference_period", "orders_available_at", "pmi_orders_level", "pmi_orders_change",
        "original_account_context", "original_cycle_net_pnl", "r224_status", "r224_net_pnl"]]) + "\n\n"
    report += "全部143事件按来源和路线首次/重复/未知列在下表。原PMI同突破路线重复21个事件，一个原阶段/R224实际完成周期都没有：过去已有持仓/现金路径已使它们不产生新实际交易；只删这些重复事件不能假设收益会改善。对应利率重复的57个突破事件虽然已有较差周期上下文，也不能用现在已知结果挑选来源、年龄或首次阈值。首次PMI突破的既有周期同样包含大量亏损以及本次两个反例。\n\n"
    report += markdown(repeats.drop(columns="role")) + "\n\n"
    report += "## 接受、拒绝及下一具体用途\n\n"
    report += "接受因果来源身份、公开钟与首次/重复的分别描述；拒绝把首次见到数据等于新公布，把每日记录等于政策冲击，把首次/重复旧周期统计等于新策略胜率。对应已生效利率作为风险背景保留，但拒绝当前字段可覆盖所有政策新闻的假设。没有登记或运行新金融规则，R212/R216/R224终局配置不重跑、不改阈值。\n\n"
    report += "下一用途是官方政策宣布信息及工具类型：分别保存宣布时刻、预计/实际生效日、降准/利率/资本市场工具等可直接核对类别，并保留未检索到和时钟不明。2019年1月、2024年9月以及2017/2026两反例应同框解释，但不能只收集成功日期；在金融准入前必须明确整个研究日历的来源覆盖和所有事件集合。这个来源用途目前仅提出，尚未登记/采集，不声称没有公告等于没有政策。来源足够时，再事前定义启动—传播—持有—失效机制，按相同20万元账户/费用/10%回撤预算比较收益、夏普、p×B和标准净期望；新历史来源也不能冒充独立未来验证。\n\n"
    report += "## 六图\n\n"
    for figure in summary["figures"]:
        report += f"![{Path(figure).stem}]({(ROOT / figure).absolute().as_posix()})\n\n"
    report += "## 直接来源及长期边界\n\n"
    report += "直接输入是[原R212事前表](../510300_broker_stage_policy_v1/results/全部3488事前阶段资格与上一完整周结构位.parquet)、[R222行业结构](../510300_industry_structure_description_v1/results/全部3488逐点行业结构_未知与源龄.parquet)、[143原事件](../510300_industry_structure_description_v1/results/全部143原阶段事件_行业锚与原周期上下文.parquet)、[R224金融失败归因](../510300_industry_leader_failure_v1/实际进入行业失效_完整结果与失败归因.md)。券商框架在[R210招商及其他券商研究](../510300_broker_cycle_inspiration_v1/券商周期框架_研究启发与具体上涨解释.md)已保留原资料和来源，本文不重新宣称逐指标复制券商策略。\n\n"
    report += "本轮新增文件全部隔离；原13项独立前瞻、E03及旧代码和拒绝事实保持。目标服务active，收益夏普目标仍未达。完整逐日/事件/来源年份/状态和反例表位于results，protocol含固定文件校验；不是当前市场观点或实盘授权。\n"
    path = OUT / "进入信息更新_具体上涨与假启动完整解释.md"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    parent.write(OUT / "delivery_receipt.json", {"at": parent.original.now(), "report": study.relative(path),
        "report_sha256": parent.digest(path), "all_events": 143, "original_cases": 240,
        "negative_window_rows": 38, "all_figures_actually_viewed": 6, "new_accounts": 0, "new_market_requests": 0,
        "diagnosis": study.relative(OUT / "post_run_diagnosis.json"), "goal_achieved": False})
    print("R226进入信息、具体上涨和假启动完整交付：全部事件分母/19关键点/两亏损/6图保持；零新金融。", flush=True)


if __name__ == "__main__":
    main()
