"""从实际保存的全部描述结果写研究报告，不重新计算信号或账户。"""
from __future__ import annotations

import pandas as pd

from research import industry_structure_description_study_v1 as study


def value(number, percent=False, digits=4):
    if pd.isna(number):
        return "未知"
    return f"{100 * float(number):.{digits}f}%" if percent else f"{float(number):.{digits}f}"


def markdown(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "／").replace("\n", " ") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    out, parent = study.OUT, study.parent
    path = out / "行业结构_具体上涨与失败反例.md"
    if path.exists():
        raise RuntimeError("行业结构报告已经保存，不覆盖。")
    summary = parent.read(out / "summary.json")
    diagnosis = parent.read(out / "post_run_diagnosis.json")
    keys = pd.read_parquet(out / "results/原17关键日行业结构_源龄与未知.parquet")
    counters = pd.read_parquet(out / "results/原R216退出反例日期_逐点行业结构.parquet")
    contexts = pd.read_parquet(out / "results/原周期1_5_20槽说明性上下文.parquet")
    joined = pd.read_parquet(out / "results/全部143事件三个固定时点_行业与原股票对照.parquet")
    key_table = markdown(["观察日／前一源日", "结构可知", "领先行业", "20日相对ETF", "领先5日／ETF5日", "行业上涨比例", "轮动", "代表成员／源龄天"], [
        [f"{r.date:%Y-%m-%d}／{r.industry_source_date:%m-%d}", "是" if r.daily_view_allowed else "未知",
         r.leader_names or "未知", value(r.leader_mean_relative20, True, 2),
         f"{value(r.leader_mean_return5, True, 2)}／{value(r.etf_return5_same_lagged_interval, True, 2)}",
         value(r.industry_positive5_fraction, True, 2), value(r.rotation_churn5blocks),
         f"{r.retained_member_count}/300／{value(r.industry_source_age_days, digits=0)}"] for r in keys.itertuples()])
    counter_table = markdown(["观察日", "结构可知", "领先5日／ETF5日", "行业上涨比例", "轮动", "分类及20历史可知／300"], [
        [f"{r.date:%Y-%m-%d}", "是" if r.view_allowed else "未知", f"{value(r.leader_mean_return5, True, 2)}／{value(r.etf_return5_same_lagged_interval, True, 2)}",
         value(r.industry_positive5_fraction, True, 2), value(r.rotation_churn5blocks), r.eligible20_member_count] for r in counters.itertuples()])
    case_table = markdown(["原案例", "原日期区间", "全部行", "逐点已知／未知", "固定锚已知／未知"], [
        [r["case_id"], f'{str(r["date_min"])[:10]}—{str(r["date_max"])[:10]}', r["rows"],
         f'{r["daily_known"]}／{r["daily_unknown"]}', f'{r["fixed_known"]}／{r["fixed_unknown"]}'] for r in diagnosis["case_counts"]])
    year_table = markdown(["年", "原日历", "已知", "未知", "已知比例", "已知且源龄超365日", "代表成员中位"], [
        [r["year"], r["calendar_rows"], r["known"], r["unknown"], value(r["known_fraction"], True, 2),
         r["known_source_age_over365"], value(r["retained_member_count_median"], digits=0)] for r in diagnosis["yearly_structure_coverage"]])
    context_table = markdown(["时期", "原信号后槽", "全部描述状态", "原完成周期", "原赢／输", "信息早于原退出"], [
        [r.period, r.relative_session, r.state, r.original_completed_cycles,
         f"{r.original_wins}／{r.original_losses}", r.information_before_original_exit] for r in contexts.itertuples()])
    discordant = joined.loc[joined.original_account_context.eq("COMPLETE") & joined.relative_session.eq(5)
        & joined.descriptive_fixed_state.eq("ETF_POSITIVE_FIXED_LEADERS_NONPOSITIVE")]
    discordance_table = markdown(["原信号日／观察日", "ETF锚后累计", "固定领先行业累计", "原股票投票", "原净损益元", "早于原退出"], [
        [f"{r.anchor_date:%Y-%m-%d}／{r.date:%Y-%m-%d}", value(r.etf_cumulative_same_lagged_interval, True, 3),
         value(r.leader_fixed_mean_cumulative_return, True, 3), r.descriptive_propagation_state,
         value(r.original_cycle_net_pnl, digits=2), "是" if r.information_before_original_exit_open else "否"] for r in discordant.itertuples()])
    report = f"""# 行业结构：具体上涨、失败反例与下一实验

2026-10-06；TECH.R221固定观察、TECH.R222保存结果。范围仍为510300.SH日线、上一完整周与现金；行业只作解释，不交易成分股。用户允许策略随状态变化，原冻结失败只约束原用途。本轮已完成数值研究，**没有新账户，因此尚未证明收益率或夏普提高**。

## 当前结论

行业结构可以增加ETF单一价格之外的描述。2024年9月24日观察，用前一源日9月23日数据，领先为保险、汽车、资本市场服务，保留行业过去5日上涨比例75%；9月26日观察升至100%，说明这段上涨有扩散证据。2019年1月和2020年4月早期也能观察到行业参与，日线已修复而上一完整周仍未确认。

同时，行业强弱和股票上涨比例不是同一件事。近期原完成周期在固定信号后第5槽有8次ETF锚后收益正、原领先行业收益非正，其中7次观察早于原退出；原周期8次均亏损。这给了“指数表面维持、原主线失去支撑”的下一假设。但是这些是已知历史周期的说明性上下文，**不是8笔新交易、不是新策略100%胜率，也不是独立验证**。计时起点为原信号，尚未改成实际进入，后续必须完整检验。

简单行业转弱退出仍不可接受：2019年6月19日的弱势次日观察快速恢复，原R216在这里提前退出后损失盈利；2024年轮动指标先降后升而上涨仍在扩散，也不支持统一“轮动高就卖”的规则。

## 券商报告如何转化为本地研究

招商目标文章把市场阶段放在主线、资金和博弈的共同背景中，提示不同时期的行为可以不同；兴业2024年报告以五日行业收益排名变化描述轮动。这支持研究阶段识别和主线持续性，未提供可直接复制的510300账户策略。我们采用的是自己的单一观察定义，不宣称复现券商收益、专有拥挤度或固定五年周期。

原文：[招商作者正文转载](https://finance.sina.cn/2026-06-03/detail-iniaecsk0758056.d.html)、[兴业作者正文转载](https://finance.sina.com.cn/stock/bxjj/2024-06-02/doc-inaxityx8553467.shtml)。此前已保存4家券商7份不同报告和可读原件，详见[券商框架报告](../510300_broker_cycle_inspiration_v1/券商周期框架_研究启发与具体上涨解释.md)。2026报告只作现时启发，没有倒填成2015等历史某日的已知消息。

## 固定方法与实际规模

全部3488观察槽、143原阶段事件、3003原0至20槽观察、240原案例行、17原关键日均保留。形成38004逐点行业记录、32949固定行业观察记录和43800锚点实际成员行。4项必要测试通过，17次输入截断的身份、数值与行业统计精确一致；18个冻结输入/代码文件保持。无重跑账户、拟合、训练目标或新行情请求。

观察日使用前一完整ETF日的实际成员和当时已公布分类。分类和20日价格历史同时明确至少294/300、原市场源VIEW_ALLOWED才形成逐点结构；单行业至少5成员、报价历史完整比例98%，至少4行业。按过去20日强弱固定前3领先，其余跟随。相同名单计算过去20日和两个不重叠5日回报，行业内股票等权、行业间等权。源龄、分类版本及未知保留，没有指数行业权重或资金净流入。

轮动为两个5日块的平均绝对排名变化除以行业数减1；自然0界用于ETF和领先行业绝对涨跌的四种描述，未搜索交易阈值、窗口或符号。固定锚后不重选赢家。当前名单的过去20日行业回报不是20日前可以交易得到的行业收益。

## 原四段的量价、行业和宏观顺序

**2015年6月下跌与短反弹。** 日周MACD转弱、波动20/60约1.284、融资五日约−6.22%，PMI订单水平仍50.6并不足以否定下跌。行业归属原件已有，但分类与20日完整报价只有约275—277/300，低于原门；四关键日及全部22案例行行业结构未知。这里不能写出“行业提前警告”的成功故事，亦不能用补0的行业指标解释止跌。

**2019年1月修复。** 1月8日观察，行业和20历史可知297/300，19个保留行业覆盖200成员。领先的软件服务、电力热力和汽车过去20日相对ETF约7.51个百分点，过去5日绝对回报3.64%，ETF同期1.80%，上涨行业比例78.95%。日线柱已正、上一完整周柱仍负，已知PMI订单49.7，资金价格利差下降。可研究价格和行业参与先于慢经济数据、周线确认的修复顺序。1月14、18日行业上涨比例均80%，轮动0.4526后0.3211，未形成持续单调下降。整个92行逐点均可知，但从案例起日固定行业跟踪只有1行满足全部组完整门，不能将当日重选领先的强度冒充最初主线始终延续。

**2020年3—7月修复与趋势。** 4月1日观察，18保留行业覆盖196成员，上涨比例77.78%，领先零售、电力热力和医药过去5日3.20%、ETF1.61%；日线柱刚正，上一完整周仍负。已公布PMI订单52.0、环比改善22.7个百分点，融资五日仍负1.40%。这属于疫情冲击后的修复背景，不能把不同指标统一加分。4月15日至7月14日当前最新2020Q1原件取得失败，行业状态保留未知；所以4月23日、5月28/29日和6月8日不能继续用旧2019Q4表补成已知。6月8日日周技术确认和融资五日+1.70%可观察，行业部分仍未知。95行逐点已知34、未知61；固定锚已知8、未知87。

**2024年9月重新定价与扩散。** 9月24日观察用9月23日行业源，16行业覆盖212成员，前3为保险、汽车、资本市场服务；过去20日相对ETF9.23个百分点、过去5日领先2.92%、ETF1.23%，上涨比例75%，轮动0.3417。当日成交份额约此前20日中位数3.3384倍、波动比1.3175；日线柱正而上一完整周负，已知PMI订单48.9、融资五日−0.87%。因此慢数据没有一致转好，也能描述量价重新定价和行业参与。

9月26日观察使用9月25日源，上涨行业16/16=100%，轮动0.2583；9月30日观察使用9月27日源，上涨行业17/17=100%，轮动却回升0.4191。集合数量也从16变17，两个点不是同一组行业的纯轮动加速度。上涨扩散与排名变动可并存，不能规定轮动下降才准买或轮动升就卖。9月30日仍用2023H2已公布分类，源龄180日，不提前使用当天晚间才可用的2024H1。31行逐点全部可知，原8月30固定锚只有12行满足完整传播门；9月24等关键日固定锚总体未知，保存的局部领先均值不能当成全体可知信号。

### 全部原17关键日

表中涨跌和相对强弱均为前一源日已形成区间；相对百分比为两个累计回报之差，解释为百分点。行业比例是保留行业，不是沪深300全部行业或指数权重贡献。

{key_table}

## 失败反例与原周期的完整对照

2019年6月12日行业上涨比例85%；6月19日仅1/19=5.26%，ETF及领先行业过去5日分别约−1.23%和−1.27%；6月20日观察恢复为13/19=68.42%，ETF及领先行业已正。原R216缩短2019-06-12周期，把原净盈利1989.10元变为−50.28元，并在重新释放现金后的7月2日产生另一笔亏损。7月2日观察又有82.35%行业上涨，所以“多数行业上涨”也不是足够的进场条件。6月20日收盘观察已晚于原6月20日开盘退出，不能事后称能提前避免该退出。

2025年6月26日、7月3/4日只有293/291/291成员分类与20报价历史同时明确，逐点行业结构未知；本次不能宣称行业信息已经解释或修复另一恢复被截断反例。

{counter_table}

### ETF正、原领先行业非正：近期第5槽全部8次

{discordance_table}

2025-03-21这次，原股票领先及跟随上涨比例约53.69%/78.67%，都过股票多数门；固定领先行业均值却为−0.437%、ETF锚后+1.773%。因此行业组与股票多数确实测量不同信息。其余7次原股票状态与行业状态的交集同样完整保存，没有只保留这次。一个不一致案例不能证明长期增量。

下面是两个时期、1/5/20三个固定槽、所有已知和未知状态。原周期上下文是R212早期27、近期44个完成周期；这些时点不是实际进入后5日，而且一个原周期可在三个时点重复出现。分组计数不能相加成独立样本数，也不能当作新策略进入胜率。近期第1槽ETF与行业均正22次，原赢7/输15；第5槽两者均正12次，原赢3/输9，说明“主线还正”不足以让旧进入规则盈利。

{context_table}

## 数据覆盖的实际限制

2015以后2855个评价槽，1956结构已知、899未知；143原阶段信号只有83行业锚可形成、60未知。原3003固定观察里934完整可知（含83个0槽没有后事件信息）、2069未知。固定行业传播要求各组完整，原锚成员的后续未知以及完整门使可用程度明显低于逐点重建，不能偷换为同一种“已知”。

{case_table}

{year_table}

已知结构实际只代表145—250成员，中位220；少于5成员的行业不进入观察。这不是指数行业权重。209个结构已知评价槽的分类源龄仍超过365日，尤其2023全部173个已知槽都用旧分类。CSRC/CAPCO分类不是申万，现代下载原件和供应商历史报价/股息实际首版均未认证。全历史为开发/校准，独立验证NOT_ESTABLISHED。

## 下一步具体完整实验

优先问题改为“ETF表面仍有支撑时，原领先行业失效是否应缩短持有”，与R216的“ETF自身和两组股票都弱就退出”不同。当前是新假设，尚未登记或运行金融。

1. 在实际买入时按前一源日固定行业和成员，以实际买入为时钟起点，禁止沿用本轮信号后的5槽来冒充买入后5槽。行业数据是否明确作为单独分支；未知保持原持有和失效流程。
2. 单一主政策考虑实际进入后首次5个已形成源日：ETF同期累计正、原领先行业均值非正才触发下一开盘退出；原风险止损优先，缺源不试错补齐。是否使用跟随行业完整门取决于该金融用途实际测量对象，需在读新收益前固定，不能为了账户结果调覆盖。
3. 完整对照保持原阶段账户、相同观察覆盖的ETF价格条件退出、全日历价格条件退出。逐笔真实现金重跑，并单列是否产生新进入、被截断原赢家、释放现金后的亏损和未平仓变化，不能只汇总原8亏损截短节约金额。
4. 使用原20万元、股息权益、T+1、100份、下一开盘、最低佣金、原风险预算与DD10%上限；两个原时期、基础及压力费用一起报告。接受门要求相对原A及阶段基线成本后CAGR/夏普增量、实际p×净盈亏比>1和标准净期望>0，稳定性和原反例；次数为软目标。保留年度/费用/最坏日笔/集中度/敞口/开放损益。
5. 该描述中的8个旧亏损仅帮助形成问题，不能作为独立留出或目标收益证据。下一用途一次登记一次完整运行；历史通过也只算开发结果，独立前瞻证据另按原冻结安排积累。

该方向可能只减少亏损，并不能保证提高交易次数或足以超过原A。原进入规则在近期大量假启动及摩擦问题仍然存在；本轮没有证明可以靠新退出完全修复。若新行业规则与价格对照相同、实际无增量、截断原赢家或经济/稳定门不通过，保留拒绝，不把阈值、窗口或正负改到通过。

## 小资金优势在本项目中的位置

小资金可降低容量约束并选择等待时机。原已计算的10万元订单占510300日成交额比例中位约0.00548%、最大0.06435%；不等于方向优势或开盘零冲击。原费用下10万元名义往返约基础0.14%、压力0.28%，较小拆单受最低佣金影响更大。因此应把灵活性用于有依据的进出场、完整风险控制和避免无优势交易，不以多交易本身代替收益提升。

## 保存结果与图

原程序一次完整观察通过；原图标题误写固定60日分母，四案例实际22/92/95/31行。原图/代码/结果保持，独立交付绘图只按实际行数纠正标题，未重复观察或改方法；4幅交付图逐幅查看。可观察性和曲线空白与保存表一致。

![2015反例](figures_delivery/原案例18_行业强弱扩散与轮动.png)

![2019修复](figures_delivery/原案例37_行业强弱扩散与轮动.png)

![2020修复](figures_delivery/原案例42_行业强弱扩散与轮动.png)

![2024扩散](figures_delivery/原案例55_行业强弱扩散与轮动.png)

依据：[固定用途卡](../../../docs/510300_INDUSTRY_STRUCTURE_DESCRIPTION_V1.md)、[实际摘要](summary.json)、[完整描述诊断](post_run_diagnosis.json)、[17关键日](results/原17关键日行业结构_源龄与未知.csv)、[完整逐点表](results/全部3488逐点行业结构_未知与源龄.parquet)、[全部429时点对照](results/全部143事件三个固定时点_行业与原股票对照.csv)、[所有周期状态](results/原周期1_5_20槽说明性上下文.csv)。旧R212/R216拒绝、R218整表失败、R220原件失败/未知及13项独立前瞻保持。
"""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    parent.write(out / "report_receipt.json", {"at": parent.original.now(), "report": study.relative(path),
        "sha256": parent.digest(path), "source_summary_sha256": parent.digest(out / "summary.json"),
        "new_accounts": 0, "financial_metrics": "NOT_COMPUTED", "goal_achieved": False})
    print("完整行业报告已保存：四原案例、17关键日、全部状态/未知和具体后续金融假设，未声明收益提升。", flush=True)


if __name__ == "__main__":
    main()
