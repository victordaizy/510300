"""交付完整原公告时序、上涨与假启动解释及一个不同机制的下一用途。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import original_policy_trace_format_v1_0_1 as repair
from research import write_entry_information_sequence_report_v1 as render

ROOT, OUT, SOURCE, parent = repair.ROOT, repair.OUT, repair.SOURCE, repair.parent


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def read_table(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / "results" / (name + ".parquet"))


def local_clocks(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    d = frame.copy()
    for column in columns:
        d[column] = d[column].map(lambda value: "未知" if pd.isna(value) else str(repair.inputs.clock(value)))
    return d


def draw_cases(daily: pd.DataFrame, rates: pd.DataFrame) -> Path:
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    observed_rates = rates[["source_available_upper", "seven_day_rate_percent"]].copy()
    observed_rates["source_available_upper"] = observed_rates.source_available_upper.map(repair.inputs.clock).astype("datetime64[ns, Asia/Shanghai]")
    observed_rates = observed_rates.sort_values("source_available_upper")
    calendar = daily[["date", "decision_time"]].copy()
    calendar["decision_time"] = calendar.decision_time.map(repair.inputs.clock).astype("datetime64[ns, Asia/Shanghai]")
    rate_daily = pd.merge_asof(calendar, observed_rates, left_on="decision_time", right_on="source_available_upper", direction="backward")
    records = daily.merge(rate_daily[["date", "seven_day_rate_percent"]], on="date", validate="one_to_one")
    cases = [
        ("2015：操作利率下降，价格仍可能下跌", "2015-06-22", "2015-07-01", [("2015-06-25", "2.70%操作原文"), ("2015-06-30", "2.50%操作原文")]),
        ("2017：加息操作背景下的原假突破", "2017-03-13", "2017-04-20", [("2017-03-16", "2.35%→2.45%操作记录"), ("2017-04-05", "原突破，后完成亏损")]),
        ("2019：宣布、转载可知与修复不同日", "2019-01-02", "2019-01-18", [("2019-01-07", "01-05转载最早ETF可知"), ("2019-01-09", "原阶段修复信号")]),
        ("2020：新操作与收盘后宣布分开", "2020-03-25", "2020-04-10", [("2020-03-30", "2.20%操作原文"), ("2020-04-07", "04-03 16:57公告首次可知")]),
        ("2024：上午宣布，操作表随后确认", "2024-09-23", "2024-10-18", [("2024-09-24", "09:19/11:42宣布"), ("2024-09-30", "09-27实施/09-29操作可知")]),
        ("2026：旧操作背景与原假突破", "2026-04-27", "2026-05-20", [("2026-05-11", "旧1.40%记录；原突破后亏损")]),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(17, 13))
    for ax, (title, start, end, marks) in zip(axes.flat, cases):
        window = records.loc[records.date.between(start, end)]
        if window.empty:
            raise ValueError("固定六案例缺少原日线。")
        normalized = 100 * (window.ac / window.ac.iloc[0] - 1)
        ax.plot(window.date, normalized, color="#176c69", linewidth=1.8, label="现金平移价相对窗口首日")
        ax.axhline(0, color="#999999", linewidth=.6)
        ax.set_ylabel("价格变化（%）", fontsize=9)
        ax2 = ax.twinx()
        ax2.step(window.date, window.seven_day_rate_percent, where="post", color="#b77b37", linewidth=1.4, label="已记录7天操作利率")
        ax2.set_ylabel("操作记录利率（%）", fontsize=9, color="#90652f")
        values = window.seven_day_rate_percent.dropna()
        lo, hi = (values.min()-.1, values.max()+.1) if len(values) else (0, 1)
        ax2.set_ylim(lo, hi)
        for index, (at, text) in enumerate(marks):
            ax.axvline(pd.Timestamp(at), color="#4f6790", linestyle="--", alpha=.5, linewidth=.8)
            ax.text(.02, .94-index*.08, text, transform=ax.transAxes, fontsize=8, va="top", color="#405c83")
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.grid(alpha=.13)
        ax.tick_params(labelsize=8); ax2.tick_params(labelsize=8)
        if title.startswith("2015"):
            lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
            ax.legend(lines, [line.get_label() for line in lines], loc="lower left", fontsize=8)
    fig.suptitle("六个固定原案例：政策来源的可知顺序与510300日线价格\n操作原文是已记录利率，不是完整政策新闻；没有消息记录不代表没有政策，未计算新策略收益", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, .94])
    path = OUT / "figures/六固定案例_政策来源与价量启动顺序.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def next_proposal() -> dict:
    return {
        "status": "PROPOSED_NOT_REGISTERED_OR_RUN",
        "name": "RECORDED_SUPPORT_PRICE_ACCEPTANCE_AND_CARRY_STATE",
        "question": "已核对支持信息在随后价格被接受并继续传播时，是否应保持一个可失效的机会状态，而不是每个入场日重新要求全部宏观变好？",
        "mechanism": "支持信息激活观察→价格接受→确认后延续持有→结构失效或反向实际操作退出；同信息各节点不独立计分。",
        "source_role": "有限已记录政策存在作为正证据；空记录未知。实际操作、同期官方转载、宣布、实施分别保留；外汇/考核/迟发回顾不自动启动股票多头。",
        "price_anchor": "新支持信息第一次落入原ETF决定区间时，固定当时已知价格区域；后来信息不得改变原锚。原20日尺度/上一完整周可作结构描述，冻结前须选定唯一动作与失效定义。",
        "entry_role": "价格被接受时才进入，宏观允许落后；不直接看到降准就买。",
        "holding_role": "进入后只用新形成价格及上一完整周决定是否保留，确认后允许转换持有方法；不是原三个静态条件的重复上升沿。",
        "required_ablation_controls": ["同价格动作不使用支持信息", "同进入但固定持有退出", "原A保存账户", "原阶段保存账户"],
        "fixed_account_boundary": "同20万元/资金风险/基础和压力费用/2015—2019与2020—2026/整手T+1/股息和开放损益。",
        "acceptance": "同口径CAGR及净Sharpe增量、DD门、完成净pB>1且标准净期望>0、原稳定区间同时报告；次数软目标。",
        "source_collection_before_financial": "本固定来源用途已结束；下步优先完成唯一机制及必要对照准入，不以补齐全国新闻全集无限延后。",
        "financial_admission": "NOT_ADMITTED_NOT_RUN", "new_financial_runs": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
    }


def save_once_or_same(path: Path, value: dict) -> None:
    """仅接续未完成的文档绘图，不重复来源计算或覆盖已保存诊断。"""
    if path.exists():
        old = parent.read(path)
        left = {key: item for key, item in old.items() if key != "at"}
        right = {key: item for key, item in value.items() if key != "at"}
        if left != right:
            raise ValueError("已保存诊断或下一提案不同，禁止绘图接续覆盖。")
    else:
        parent.write(path, value)


def main() -> None:
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("原公告解释已经交付，不覆盖。")
    result = parent.read(OUT / "summary.json")
    initial = parent.read(SOURCE / "summary.json")
    if (result["all_key_rows"] != 19 or len(result["prefix_checks"]) != 19
            or result["all_daily_rows"] != 3488 or result["all_events"] != 143
            or initial["all_key_rows"] != 17 or result["new_requests"] != 0):
        raise ValueError("完整原案例与格式结果不匹配。")
    sources = read_table("全部22新来源_格式读取与原失败保留")
    nodes = read_table("全部来源节点_旧49与原六新源逐值保持")
    keys = read_table("全部19关键日_原17与两假启动")
    events = read_table("全部143原进入事件_真正可知来源")
    rate_records = pd.read_parquet(SOURCE / "results/全部25利率操作原文_改变与首发不混同.parquet")
    daily = pd.read_parquet(repair.original.DAILY)
    matched = sources.loc[sources.status.eq(repair.MATCHED)]
    conflicts = sources.loc[sources.status.eq("SOURCE_CONTENT_MATCHED_PUBLICATION_METADATA_DATE_CONFLICT")]
    financial = parent.read(ROOT / "reports/research/510300_industry_leader_failure_v1/summary.json")
    financial_rows = pd.DataFrame(financial["metrics"])
    financial_rows = financial_rows.loc[financial_rows.cost.eq("STRESS") & financial_rows.policy.isin(["A_SAVED_WEIGHT", "FIXED_INDUSTRY_LEADERS_FAILED_ETF_POSITIVE"])]
    proposal = next_proposal()
    save_once_or_same(OUT / "next_support_acceptance_state_proposal.json", proposal)
    ages = nodes.source_available_upper.map(repair.inputs.clock)
    first_known = []
    for node_id in ["LOCAL_GOV_20190105", "SOURCE_001", "CHAIN_R01", "CHAIN_C01", "RATE_24"]:
        node = nodes.loc[nodes.node_id.eq(node_id)].iloc[0]
        available = repair.inputs.clock(node.source_available_upper)
        eligible = daily.loc[daily.decision_time.map(repair.inputs.clock).ge(available)]
        first_known.append({"node_id": node_id, "source_available_upper": str(available),
            "first_original_etf_decision_date": str(pd.Timestamp(eligible.date.iloc[0]).date()),
            "information_role": node.information_role, "source_url": node.source_url})
    repair.table("五关键政策源_实际公开与首次ETF可知", pd.DataFrame(first_known))
    interval_events = events.loc[events.new_recorded_nodes.gt(0)]
    diagnostic = {
        "at": parent.original.now(), "all_fixed_reference_identities": 97,
        "saved_source_reused_identities": 49, "new_or_local_reference_matches": 22, "still_unknown_reference_identities": 26,
        "new_native_requests": 22, "received": 18, "fetch_failures": 4,
        "verified_new_content_and_clock": 15, "visible_metadata_date_conflicts": 2, "literal_match_unknown": 1,
        "verified_rate_operation_sources": 25, "rate_previous_record_unknown": int(rate_records.previous_observed_rate_percent.isna().sum()),
        "rate_observed_decreases": int(rate_records.change_vs_previous_observed_bp.lt(0).sum()),
        "rate_observed_increases": int(rate_records.change_vs_previous_observed_bp.gt(0).sum()),
        "rate_differences_are_immediate_policy_changes": False,
        "original_initial_description_had_17_keys": True, "final_original_and_negative_keys": 19,
        "all_original_daily_rows": 3488, "all_original_events": 143,
        "events_with_new_recorded_node": len(interval_events), "same_day_new_policy_hard_gate_admitted": False,
        "old_gov_announced_event_date": "2019-01-04", "old_gov_source_visible_date": "2019-01-05",
        "old_source_correction_requires_old_financial_rerun": False,
        "2020_april3_announcement_after_original_16_clock": True, "first_april3_announcement_etf_decision": "2020-04-07",
        "first_known_policy_sources": first_known,
        "next_support_state_proposal": relative(OUT / "next_support_acceptance_state_proposal.json"),
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
        "financial_metrics": "NOT_COMPUTED", "latest_actual_financial_decision": "TECH.R224",
        "latest_actual_financial_status": financial["status"], "goal_achieved": False,
        "complete_policy_announcement_coverage": "NOT_ESTABLISHED", "independent_validation": "NOT_ESTABLISHED",
    }
    save_once_or_same(OUT / "post_run_diagnosis.json", diagnostic)
    figure = draw_cases(daily, rate_records)
    evidence = matched[["source_id", "information_role", "source_available_upper", "title", "url"]]
    evidence = local_clocks(evidence, ["source_available_upper"])
    conflict_evidence = conflicts[["source_id", "visible_publication_upper", "metadata_publication_literal", "information_role", "url"]]
    conflict_evidence = local_clocks(conflict_evidence, ["visible_publication_upper"])
    displayed = keys[["date", "stage_entry_type", "daily_hist", "weekly_hist", "log_relative_volume", "pmi_orders_level", "pmi_orders_change",
                      "industry_view_allowed", "industry_positive5_fraction", "new_node_ids", "new_announcement_node_ids"]].copy()
    displayed["相对量"] = np.exp(displayed.pop("log_relative_volume"))
    displayed["PMI新订单原值"] = displayed.pop("pmi_orders_level") + 50
    displayed["date"] = pd.to_datetime(displayed.date).dt.strftime("%Y-%m-%d")
    rate_view = local_clocks(rate_records[["node_id", "observed_operation_date", "source_available_upper", "seven_day_rate_percent", "previous_observed_rate_percent", "change_vs_previous_observed_bp"]], ["source_available_upper"])
    rate_view["observed_operation_date"] = pd.to_datetime(rate_view.observed_operation_date).dt.strftime("%Y-%m-%d")
    text = "# 510300 原政策公告：上涨启动、传播与假启动\n\n"
    text += "2026-10-06，TECH.R229登记、R230一次来源追溯和格式实现1.0.1。**接受策略可以随已知阶段改变；这轮最有价值的发现是，政策宣布、实际操作、旧利率背景和价格接受是不同信息，不能要求它们在每一个进场日同时变好。**下一实验优先把信息持续作用和价格失效定义成一个完整机会状态。还未登记或运行新金融策略，收益和夏普没有新增改善结果，最新实际金融R224仍拒绝。\n\n"
    text += "## 固定范围、真实覆盖与技术缺口\n\n"
    text += "97原参考身份由25已保存利率操作、45准备金导航全文、3资本工具目录及24原核对节点组成；不是97独立政策。42固定查询组只是查源分组，共完成42查询、8次字面题目/索引跟进与6次官方入口打开，所有网页结果保存在evidence。22本机新原文请求收到18个、4个真实失败不重试。原实现6个新源内容/钟核对通过、17原关键日实际完成；这两个分母如实留档，不写成原实现已完成19日。\n\n"
    text += "单独格式实现只读取12个已保存页面及既有来源关联，0请求。最终15新源内容和公开钟可核对，2个可见日与元数据跨日冲突、1个原字面核对要求未通过仍保留未知；4取得失败保持。原55节点逐值不变，新增9格式可核对节点和1原政府转载，共65节点、60共同源身份。这仍不是65次独立冲击：共同文件、事件多工具、操作与实施会重复，economic_identity是描述身份，未经全国经济事件归一认证。\n\n"
    text += "97身份中49原源复用、22身份有新源或既有源关联、26仍未知。五组既有源关联只是复用，并不宣称新发现；没有关联的目录也可能在现有资料外有原公告。全3488日、143原进入事件和原17+两个假启动共19关键日已连接，19整段前缀精确；6原必要测试和3格式测试通过。原结果、原源码、所有未知与失败保留。当前页面首版和全国公告覆盖未认证，既有首发网页本身也不保证历史首版未修改。\n\n"
    text += "## 具体上涨与两次假启动\n\n"
    text += "**2015年反弹与下跌。**6月25日原7天操作利率2.70%、6月30日2.50%均在上午已知，价格仍在剧烈下跌背景。2.70%相对前一保存记录3.35%的差并不证明当日一次降息65基点，因为中间操作覆盖不完整。日MACD关键反弹日仍负，行业全部22案例日未知。说明观察到货币操作下降并不足以确认抛压已经消化。6月27日宣布/28日实施的政府线索取得失败保留，不因此声称这段没有政策。\n\n"
    text += "**2017年原假突破。**3月16日09:46:29操作原文7天利率2.45%，前一保存值为2月3日2.35%；这属于操作利率上调背景。4月5日相对量1.5408、日周MACD正、PMI新订单53.3且环比+0.3，仍在原压力周期亏1181.26元。加息操作可以描述相反背景，却不能由一个反例直接定义负号加分或硬门，更不能把这笔旧亏损从历史删除。\n\n"
    text += "**2019年修复。**正文宣布发生在1月4日，原gov_20190104.html页面实际标注1月5日，日期级上界23:59:59；本固定原日历最早在1月7日16:00可知。此前R228文档把该保存页称为1月4日来源不准确，本轮新增纠正，旧报告和旧六降准策略账户不覆盖。1月8日行业15/19上涨、日柱正而周柱负，PMI49.7且−0.7；1月9日原修复后来已完成净盈利7978.96元。‘同进场日刚发布政策’与‘已知支持之后价格修复’是不同用途，后者更符合这一信息顺序。当前转载不是全网最早宣布认证。\n\n"
    text += "**2020年修复。**3月30日09:45:05原操作2.20%，4月1日订单29.3→52.0、日柱正但周负、14/18行业上涨；4月1日不是新操作公布日。4月3日官方降准公告时刻16:57:32，晚于原16:00决定，首次可知ETF观察日为4月7日，不能提前到4月3日信号。公告同时写明4月15日/5月15日分步执行和超额准备金利率下调，实施日期不等于公告钟。后续6月2日原修复净盈利10756.16元，不是在当天才得到首次政策支持；行业源缺口仍保持。\n\n"
    text += "**2024年重新定价。**9月24日原直播R01/H02上界09:19:36、资本工具C01上界11:42:50，共同文件一源。相对量3.3384、日柱正/周负、12/16行业上涨；PMI48.9且−0.4、融资五日负、原利率背景仍7月22日。宣布能够先于慢统计与周线确认，支持研究不同阶段的信息角色。9月27日实施报道、9月29日保存操作1.50%原文，都在9月30日ETF决定区间已知；不是两个新的独立政策刺激。原次日进入周期净盈利3776.25元，5000亿元互换便利和3000亿元再贷款是工具额度而非实际股票购买。\n\n"
    text += "**2026年原假突破。**5月11日相对量2.3905、日周正、12/16行业上涨、领先五日+5.9283%，资金差−0.1013个百分点；PMI50.6但环比−1.0、融资未知。保存1.40%操作原文为2025年5月8日；这轮没有找到与该进入日同区间的新记录，含义仅是本目录没有节点，不能声称当天没有政策。原5月12日至19日完成净亏1753.77元。新的机制应检验原支持背景是否仍被价格接受及是否失效，而不是因已知这笔亏损临时删掉某条路线。\n\n"
    text += render.markdown(displayed) + "\n\n"
    text += "## 公告公布与原日历实际可知\n\n" + render.markdown(pd.DataFrame(first_known)) + "\n\n"
    text += "全143原进入事件只有两次与本固定目录新节点区间相交：2020-04-07修复及2024-09-24重新定价。2024的两个宣布节点和一个住房更新共用来源。这个分母说明单独新增‘当天必须有新政策’容易遗漏先政策后价格的修复；不代表两笔新策略交易、100%胜率或政策对股票的因果识别，也不代表其余141日没有新闻。\n\n"
    text += render.markdown(interval_events[["date", "stage_entry_type", "new_node_ids", "new_unique_common_sources", "new_announcement_node_ids"]]) + "\n\n"
    text += "## 全部核对新源与仍未知\n\n" + render.markdown(evidence) + "\n\n"
    text += "两源的页面可见日期与PubDate跨年不一致，技术格式读取后仍未取得可用于新节点的公开上界：\n\n" + render.markdown(conflict_evidence) + "\n\n"
    text += "SOURCE_018是已收到的外汇准备金上调转载，原登记字面核对词仍未全部匹配，保留完整原文与未知，不将此情况写成无政策或HTTP失败。SOURCE_003/006两政府页404、SOURCE_009湖南源TLS错误、SOURCE_013南昌源TLS EOF均保留请求回执；搜索片段不代替原文钟。外汇存款/远期售汇准备金、年度考核和平均法是不同工具，不能统一当人民币降准正向信号。\n\n"
    text += "## 25原利率操作的逐文事实\n\n"
    text += "25原文利率和秒级钟均与冻结原记录一致；1个前值未知、20个相对前一保存值下降、4个上升。差值仅相对前一已观察操作；未认证完整中间操作、最早政策宣布或瞬时加减幅度。原表和原背景时间不覆盖，不能把首次见到原数据叫首次政策发布。\n\n" + render.markdown(rate_view) + "\n\n"
    text += "## 金融结果仍然是什么\n\n"
    fields = ["period", "policy", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles", "win_rate", "payoff", "p_times_b", "standard_expectancy_loss_units", "average_full_year_cycles", "open_pnl_cny"]
    text += "以下仅复用R224已保存压力费用账户，不是本次新策略：\n\n" + render.markdown(financial_rows[fields]) + "\n\n"
    text += "R224四经济门0/4、稳定门失败。全账户净收益/Sharpe尚未同时提高；p×实际净盈亏比也没有在所有场景超过1。20万元资金、手续费、滑点、股息、风险减仓和所有现金日保持；原A开放损益未知显示未知。小资金的容量和选择灵活性可以支持设计，但旧高换手路径已经显示，次数增加不能代替净优势。本次金融指标NOT_COMPUTED，目标active而未达成。\n\n"
    text += "## 下一完整实验：支持信息的持续作用、价格接受与延续\n\n"
    text += "优先研究一个不同于三个静态进入条件的状态机制：①有已核对支持信息时只激活观察；②在信息已知后固定当时价格区域，随后价格被接受才进入；③确认后切换到允许趋势延续的持有动作；④跌破原结构或出现已知反向操作时失效。宏观数据可以落后，全部工具和共同源各按角色使用，未知不直接作空头证据。持有期由可观察失效决定，不按事后最高点决定。\n\n"
    text += "这不是单日新闻硬门、同源多数投票、日周MACD统一加权分或旧阶段配置调参数。正式准入前还要确定唯一价格接受/失效动作，并配同价格无政策、同进入固定退出、原A及原阶段对照；没有这些具体合同不能报告收益。来源固定用途本轮已完成，下一优先是机制与完整账户，不以追求全国新闻全集无限延后实验。当前提案PROPOSED_NOT_REGISTERED_OR_RUN，0已准入待跑金融。\n\n"
    text += "按同资金风险/两个时期/基础压力费用、实际p×B>1及标准净期望正、CAGR和净Sharpe增量、原稳定区间共同判断。所有历史均开发用途，独立验证仍未建立。一次配置失败冻结，后续只有不同机制/信息或真正新样本才进入新用途；原R212/R216/R224、旧六降准拒绝和原E03/13前瞻逐值保持。\n\n"
    text += "## 固定六案例图\n\n" + f"![六固定案例政策与价格顺序]({figure.absolute().as_posix()})\n\n"
    text += "## 直接来源与保存证据\n\n"
    text += "框架来源与7份券商资料在[招商、华泰、国金及兴业研究](../../510300_broker_cycle_inspiration_v1/券商周期框架_研究启发与具体上涨解释.md)已记录，并不等于复制其策略。[央行2020-04-03公告](https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212550691528/index.html)支持16:57:32与分步实施事实；[2019-01-05政府转载](https://app.www.gov.cn/govdata/gov/201901/05/433817/article.html)支持宣布与转载日不同；[2024-09-24发布会原文](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)与原人工核对链记录同源及工具。当前取得网页，不认证原历史首版。\n\n"
    text += "原冻结[用途卡](../../../../docs/510300_ORIGINAL_POLICY_ANNOUNCEMENT_TRACE_V1.md)、原protocol/summary/raw/receipts及格式protocol/summary/results都保留；本报告、post_run_diagnosis和下一状态提案同目录。完整97/22/3488/143/19表及共同源关系可独立查看。研究只涉及510300.SH日线/上一完整周和现金、多头优先，其他观察数据不形成交易资产或实盘委托。\n"
    report = OUT / "原政策公告_上涨启动与假启动完整解释.md"
    with report.open("x", encoding="utf-8") as stream:
        stream.write(text)
    parent.write(OUT / "delivery_receipt.json", {
        "at": parent.original.now(), "report": relative(report), "report_sha256": parent.digest(report),
        "figure": relative(figure), "figure_sha256": parent.digest(figure), "figure_view_required": True,
        "all_original_key_points_in_report": 19, "new_accounts": 0,
        "new_actual_financial_metrics": "NOT_COMPUTED", "next_proposal": relative(OUT / "next_support_acceptance_state_proposal.json"),
    })
    print("R230原公告、六案例完整解释和下一状态提案已保存；一图待实际查看，旧金融未变。", flush=True)


if __name__ == "__main__":
    main()
