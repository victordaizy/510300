"""连接公司驱动、事件时间与实际价格；只形成诊断，不生成交易收益。"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_current_price_transmission_v1"
STAMP = datetime.now(timezone(timedelta(hours=8))).isoformat()
BASE, END = "2026-08-31", "2026-09-28"
NAMES = {"sh510300": "沪深300ETF", "sh000300": "沪深300价格指数", "sz300308": "中际旭创",
         "sz300502": "新易盛", "sh688256": "寒武纪", "sh600036": "招商银行", "sh601318": "中国平安"}
STOCKS = ["sz300308", "sz300502", "sh688256", "sh600036", "sh601318"]
PRESS = "https://www.mccormick.senate.gov/news/press-releases/senators-mccormick-gallego-cornyn-fetterman-introduce-bill-to-keep-chinese-transceivers-out-of-u-s-national-security-systems/"
LEGISLATION = "https://www.mccormick.senate.gov/about/legislation/"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pct(after, before):
    return (after / before - 1) * 100


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮诊断已完成，不覆盖结果。")
    src = read(OUT / "source_result.json")
    event = read(OUT / "event_source_result.json")
    previous = read(ROOT / src["reused_raw_prices"])
    weights_source = read(ROOT / "reports/research/510300_current_driver_outlook_20260929/price_expectation_bridge/reviewed_facts.json")
    weights = {x["symbol"]: x["weight_pct"] for x in weights_source["top10"]}
    raw, adjusted = {}, {}
    for symbol in NAMES:
        if symbol in previous["raw_daily_rows"]:
            rows = previous["raw_daily_rows"][symbol]
        else:
            rows = src["responses"][symbol + "_raw"]["rows"]
        raw[symbol] = [x for x in rows if BASE <= x["date"] <= END]
        adjusted[symbol] = (src["responses"][symbol + "_qfq"]["rows"] if symbol != "sh000300" else raw[symbol])
    dates = [x["date"] for x in raw["sh000300"]]
    if dates[0] != BASE or dates[-1] != END or len(set(dates)) != len(dates):
        raise ValueError("基准窗口端点或日期唯一性不符")
    rows = []
    for symbol in NAMES:
        for series in [raw[symbol], adjusted[symbol]]:
            if [x["date"] for x in series] != dates or any(x["close"] <= 0 for x in series):
                raise ValueError(f"{symbol}价格日期未对齐或存在非正价格")
        start, prev, end = raw[symbol][0], raw[symbol][-2], raw[symbol][-1]
        quote = src["responses"]["sina_previous_close"][symbol]
        tolerance = 0.01 if symbol == "sh000300" else 0.00001
        if quote["quote_date"] != "2026-09-29" or abs(end["close"] - quote["previous_close"]) > tolerance:
            raise ValueError(f"{symbol}末日收盘未与次日昨收字段一致")
        weight = weights.get(symbol[2:])
        period = pct(end["close"], start["close"])
        rows.append({
            "symbol": symbol, "name": NAMES[symbol], "start_close": start["close"], "end_close": end["close"],
            "raw_price_change_pct": period,
            "vendor_adjusted_price_change_pct": None if symbol == "sh000300" else pct(adjusted[symbol][-1]["close"], adjusted[symbol][0]["close"]),
            "last_session_previous_date": prev["date"], "last_session_price_change_pct": pct(end["close"], prev["close"]),
            "last_session_open_gap_pct": pct(end["open"], prev["close"]),
            "last_session_open_to_close_pct": pct(end["close"], end["open"]),
            "weight_20260831_pct": weight,
            "fixed_start_weight_price_contribution_pp": None if weight is None else weight * period / 100,
            "end_close_secondary_vendor_agrees_within_quote_precision": True,
            "raw_adjusted_close_different_dates": [a["date"] for a, b in zip(raw[symbol], adjusted[symbol]) if abs(a["close"] - b["close"]) > 0.00001],
        })
    keyed = {x["symbol"]: x for x in rows}
    pa = keyed["sh601318"]
    pa["gross_wealth_change_with_dividend_held_as_cash_pct"] = pct(pa["end_close"] + 0.98, pa["start_close"])
    contribution = sum(keyed[s]["fixed_start_weight_price_contribution_pp"] for s in STOCKS)
    total_weight = sum(weights[s[2:]] for s in STOCKS)
    buyback = event["responses"].get("innolight_buyback")
    if buyback is None:
        raise ValueError("回购原件未取得，不能形成完成金额结论")
    buyback_text = "\n".join(p["text"] for p in buyback["pages"])
    if "499,724.68" not in buyback_text or "5,653,063" not in buyback_text or "实施完成" not in buyback_text:
        raise ValueError("回购原件核心字段不匹配")
    web_text = (OUT / "sources/web_reader_policy_and_buyback.txt").read_text(encoding="utf-8")
    if "S.5548" not in web_text or "five-year" not in web_text:
        raise ValueError("官方网页读取材料未包含必要状态与过渡期")
    facts = {
        "recorded_at": STAMP, "base_close": BASE, "end_close": END, "price_rows_per_symbol": len(dates),
        "prices": rows, "selected_weight_pct": total_weight, "fixed_weight_five_stock_contribution_pp": contribution,
        "contribution_limit": "8月末固定权重乘未复权价格变化的量级算例，不是全量指数正式归因，不与前复权回报混算。",
        "current_prices_are_executable": False,
        "dividend": {"symbol": "sh601318", "gross_cash_per_share": 0.98, "ex_date": "2026-09-10",
                     "wealth_method": "期末股价加税前现金股息，现金不复投；不含税费，不是策略账户。",
                     "etf_check": "供应商最新分红页仍列1月19日；本窗口前复权与未复权收盘一致。未把9月4日旧本地分红表冒充覆盖9月28日。"},
        "policy_event": {
            "bill": "S.5548", "official_release_date": "2026-09-25",
            "status_as_shown": "发起人官网列示已读两次并转国土安全与政府事务委员会，未显示已立法实施。",
            "named_companies": ["InnoLight", "Eoptolink"],
            "scope_from_sponsor_summary": "联邦国家安全系统的受覆盖光收发器采购；摘要还涉及关联企业及软硬件组件，不能仅按直接销售对象判定全部实际敞口。",
            "transition_and_supply": "五年过渡、美国及盟国产能评估、缺少可信替代时有限可续豁免。",
            "stated_policy_reason": "发起人称安全风险、供应链依赖和本土制造能力；这是政策方公开理由，不等于已证明产品存在后门。",
            "causal_inference": "法规覆盖与客户去风险可能影响订单及长期份额，尾部风险上升也可先改变估值；两者均未识别量化效应。",
            "bill_text_available": False, "source_urls": [PRESS, LEGISLATION],
            "direct_http_status": [r["http_status"] for r in event["receipts"] if r["key"].startswith("optical_bill")],
            "source_availability": "直接HTTP为403；官方网页经浏览工具读取成功，完整工具提取内容单独保存；不称取得法案正文。",
            "pre_event_consensus_probability": None, "company_affected_revenue_share": None,
            "price_causal_share": None,
        },
        "buyback": {"start": "2026-09-01", "completion": "2026-09-23", "document_date": "2026-09-25",
                    "prior_reprint_date": "2026-09-24晚", "date_note": "A股原件署9月25日，9月24日晚已有转载；二者均在9月28日前，不混为同一日期。",
                    "cash_cny": 4997246800, "shares": 5653063, "status": "本计划已实施完成",
                    "new_future_buy_flow_from_completed_plan_cny": 0,
                    "future_new_plan_possible": True, "purpose": "股权激励或员工持股；未用部分后续按条件注销，不等于已经注销。",
                    "source_url": buyback["url"]},
        "inference_limits": [
            "知情复盘不等于在9月28日前做出预判；政策风险也不全是本次首次出现。",
            "未被此提案摘要点名的寒武纪同样下跌，可能有共同市场或其他公司冲击；没有可识别因果的控制组。",
            "没有前置调查或隐含政策概率，不能把全部跌幅当预期差或宣称下跌已充分。",
            "金融两家公司不能代表全金融；五家公司之外的指数影响未计算，不把它们当零。",
        ],
    }
    save(OUT / "reviewed_facts.json", facts)
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    matplotlib.rcParams["font.family"] = font.get_name()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.7), sharey=True)
    groups = [STOCKS[:3], STOCKS[3:]]
    colors = {"sz300308": "#1864ab", "sz300502": "#009989", "sh688256": "#b05e13", "sh600036": "#1864ab", "sh601318": "#ad426b"}
    for ax, group, title in zip(axes, groups, ["科技相关公司：阶段变化与最后一日调整", "金融公司：表现与传导有分化"]):
        for symbol in group:
            vals = [r["close"] / adjusted[symbol][0]["close"] * 100 for r in adjusted[symbol]]
            ax.plot(range(len(dates)), vals, lw=2.0, color=colors[symbol], label=NAMES[symbol])
        ax.plot(range(len(dates)), [r["close"] / raw["sh000300"][0]["close"] * 100 for r in raw["sh000300"]], lw=2, ls="--", color="#41464d", label="沪深300价格指数")
        ax.axhline(100, lw=0.7, color="#9ba1a6")
        ax.axvspan(len(dates) - 2, len(dates) - 1, color="#f4d9ae", alpha=0.4)
        ax.set_xticks([0, 5, 10, 15, 19], [dates[i][5:] for i in [0, 5, 10, 15, 19]])
        ax.set_title(title, fontsize=12, pad=14)
        ax.grid(axis="y", color="#e4e7eb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    axes[0].set_ylabel("8月31日收盘 = 100")
    fig.suptitle("同一组公司的价格，不能用同一种因果解释", fontsize=16, x=0.065, ha="left", y=0.98)
    fig.text(0.065, 0.04, "个股使用供应商前复权价格；基准为价格指数。黄色区间为9月24日至28日。\n两者分红口径不同，图示不代表账户收益、超额收益或已识别的事件因果效应。", fontsize=10, color="#555d65")
    fig.subplots_adjust(top=0.85, bottom=0.22, left=0.065, right=0.98, wspace=0.10)
    fig.savefig(OUT / "阶段价格与最后一日调整.png", dpi=170, facecolor="white")
    plt.close(fig)
    table_lines = []
    for r in rows:
        adjusted_label = "—（价格指数）" if r["vendor_adjusted_price_change_pct"] is None else f"{r['vendor_adjusted_price_change_pct']:+.2f}%"
        table_lines.append(f"| {r['name']} | {r['raw_price_change_pct']:+.2f}% | {adjusted_label} | {r['last_session_price_change_pct']:+.2f}% |")
    table = "\n".join(table_lines)
    gaps = "\n".join(f"| {keyed[s]['name']} | {keyed[s]['last_session_open_gap_pct']:+.2f}% | {keyed[s]['last_session_open_to_close_pct']:+.2f}% |" for s in STOCKS[:3])
    report = f"""# 因子为什么变化：价格反应、政策约束与实际承接

记录时间：{STAMP}。价格窗口固定为2026年8月31日收盘至9月28日收盘；9月29日盘中报价仅用昨收字段交叉核对，不冒充当前成交价。

**本轮结论：旧的经营增长证据仍在，但新增政策约束可能提高风险补偿要求，而中际旭创本笔回购已结束。它们为科技相关股票最新调整提供了可追踪解释；实际订单损失、政策冲击占跌幅多少、下跌后剩余交易空间均未得到量化。五家公司也解释不了同期指数的大部分跌幅。**

**先把价格事实与解释分开。**

| 观察对象 | 全窗口未复权价格变化 | 全窗口供应商前复权变化 | 最后一交易日变化 |
| --- | ---: | ---: | ---: |
{table}

最后一日是9月24日收盘到9月28日收盘，不是把9月25日不存在的A股收盘补出来。7个对象各20日，日期对齐；末日收盘与新浪次日昨收在报价精度内一致。行情来自腾讯、新浪供应商，非交易所官方成交记录。固定五家公司沿用前轮研究对象，本轮开始前已看到部分价格端点，不冒充盲测。

平安9月10日每股派息0.98元含税，原价下跌含除息影响。若期初持股、现金不复投且忽略税费，期末股价加股息对应财富变化为{pa['gross_wealth_change_with_dividend_held_as_cash_pct']:.2f}%；前复权价格比为{pa['vendor_adjusted_price_change_pct']:.2f}%，两者不同。不能把前复权价格比当精确总回报。来源：[平安权益分派原件](sources/pingan_dividend.pdf)，第1页。ETF窗口内两种价格一致，最新供应商分红表仍列1月19日；旧本地分红表只覆盖到9月4日，未用它替代后续检查。

按8月末官方固定权重，五股合计{total_weight:.2f}%，原价变化的静态加权量级为{contribution:+.3f}个百分点；同期沪深300价格指数为{keyed['sh000300']['raw_price_change_pct']:+.2f}%。这是起始权重算例，不是完整指数贡献分解。剩余部分涉及其余成分、权重路径、指数维护等，不能直接归因于某一宏观因素。ETF比价格指数少跌也不能直接称为超额收益或错价；分红、净值与二级价格等口径尚未在本轮完整分解。

![阶段价格](<{(OUT / '阶段价格与最后一日调整.png').as_posix()}>)

**新政策变化：先追制定原因、实际覆盖及约束，再判断订单和估值。**

美国参议员9月25日官方说明点名InnoLight、Eoptolink，拟限制联邦国家安全系统使用受覆盖光收发器。官网议案表列示S.5548转交委员会；现有材料没有显示已立法实施。官方摘要列五年过渡、美国及盟国产能评估和有限豁免。这支持“替代供给与切换成本也是政策执行约束”的判断。提出者公开理由涉及安全、供应链依赖和本土制造；其风险指控不是本研究已验证的产品技术事实。[官方说明]({PRESS})、[议案状态]({LEGISLATION})。

正文未取得，不能据摘要精确测算政府采购、承包关系、关联企业及组件穿透后的实际收入覆盖。也不能由这份摘要推出所有美国商业数据中心立即停购，或保证商业订单完全不受影响。两条影响链应分别追踪：一条是客户去风险、认证与替代、订单和长期市场份额；另一条是扩围概率与损失尾部提高所需风险补偿，即使本季利润未变也可先压低估值。具有拟议五年过渡安排的政策，若推进，也可能通过现在的客户选择和价格起作用。

直接HTTP取得上述两页返回403；浏览工具成功读取官方页面，提取内容保存于[sources/web_reader_policy_and_buyback.txt](sources/web_reader_policy_and_buyback.txt)。这与取得法案逐条正文不同。本轮未把有关另一FCC程序的媒体描述并入此议案，也未假定9月25日之前完全没有此类政策担忧。

**实际买盘变化：回购完成意味着这笔承接已发生。**

原件确认：本次回购2026年9月1日至23日执行，买入5,653,063股、花费49.972468亿元，方案已完成。原计划40亿至80亿元，不能拿上限减已花金额，再把约30亿元当仍承诺的未来买盘。完成后的本计划新增买入流为零；不排除公司以后另启计划。股份用于激励或员工持股，不能按已注销处理。[回购结果原件]({buyback['url']})，第1—4页。

原件署9月25日，9月24日晚已有转载，两者都先于9月28日；保留日期差别。本轮没有完整逐日公司订单与市场成交分解，不能量化回购结束导致多少跌幅，也不能把回购价下限当未来托底价。

**价格已如何反应，以及还不知道什么。**

| 公司 | 9月28日开盘相对前收 | 当日开盘至收盘 |
| --- | ---: | ---: |
{gaps}

跌幅并非全部出现在开盘跳空，三家公司当天开盘后仍下跌。但这是事件发生后的观察，不是已登记的可交易预测，不能按开盘价补造一笔躲跌交易。此提案摘要未点名的寒武纪也显著下跌，提示共同风险承担、资金或其他公司信息也可能作用；它不是经过匹配的有效控制组。

缺少提案前的市场政策概率和受影响收入预期，也没有当日订单取消原件。因此“政策风险重新定价”是得到时间与覆盖支持的候选解释，并非全部跌幅的已识别原因。价格下跌并不自动说明风险已计足，更不构成买入赔率计算。

**把相同方法用于不同因子，得到下一阶段的条件判断。**

| 原因与约束 | 因子为何会变 | 对未来股市的传导 | 最能改变判断的后续信息 |
| --- | --- | --- | --- |
| 政策仍限当前摘要范围、替代供应受约束 | 近期直接订单冲击可能较有限，但远期份额风险存在 | 经营增长与估值折价可同时存在；没有证据直接给持续下跌或反弹幅度 | 正文覆盖、立法推进、客户认证和采购是否切换 |
| 扩围或客户提前去风险 | 未来订单/价格/利用率可能下修，备货回款可能受压 | 风险补偿变化进一步传入真实现金流；需要同时下调盈利路径 | 公司的订单与交付指引、客户采购、库存应收变化 |
| 已完成回购没有新计划接续 | 过去累计回购增加，但这笔未来买盘终止 | 同样抛压下承接可能减弱；能否被其他买盘补足未知 | 新承诺、实际执行和价格承接，不能仅看累计金额 |
| 跨季资金压力缓解但实体需求未变 | DR007可回落，贷款需求却未必增强 | 流动性改善与银行息差、企业收入改善不同步 | 原F3与F1分别观察，不把同一次资金改善重复算利好 |
| 需求真正恢复并超过既有预期 | 融资需求、销量和回款改善，成本传导能力可能增强 | 盈利支撑扩散到更多指数权重，但仍取决于定价和估值 | 新订单的行业分布、零售信贷、公司指引与既有预期差 |
| 股市上涨反过来改善金融盈利 | 投资收益、基金销售和交易收入增加 | 可以形成反馈，不能把反馈当独立新增的实体需求证据 | 营运与净利润、量价、市场收益分别核对 |

金融传导依据已读的[招行半年报](https://static.cninfo.com.cn/finalpage/2026-08-29/1225530237.PDF)和[上轮完整分析](../510300_financial_driver_bridge_v1/金融权重的盈利原因与指数传导.md)。招行规模贡献抵消利率拖累、零售需求仍弱；其管理层已预期下半年息差降幅收窄，重复这个预期不是新增惊喜。平安净利润与营运利润的差额不能当稳定经营增量。

当前可采纳的判断是：短阶段内，应把新政策范围与客户反应放在科技相关公司的旧同比增速之前，把未来实际承接放在累计回购额之前。对510300整体，保留制造业与非制造业、资金供给与有效融资需求、科技与金融的分化，尚不足以给出有净优势的方向及仓位。

轻量研究只保留三项必要约束：判断所用消息在当时可知；一个主解释有能推翻它的后续事实；股息、成交时点和成本不造假。无需先把每个因子拉去做五年十年统一检验，但挑选事后最顺的片段也不能证明有效。本轮没有新收益策略、调参或账户，原F1/F2/F3及FIN1原样保留，等待其各自约定信息。下一次判断重点是新信息是否改变本轮路径，而非重复已经发生的跌幅。

**夏普1.2目标仍未实现。** 本轮净夏普、年化收益均未计算，研究仍在因果路径与剩余预期差阶段；0个新账户、0个新收益测试，不把解释能力冒充盈利能力。
"""
    report_name = "价格反应与新增信息.md"
    (OUT / report_name).write_text(report, encoding="utf-8")
    result = {
        "study_id": "510300_CURRENT_PRICE_TRANSMISSION_V1", "recorded_at": STAMP,
        "previous_goal_turn_classification": "PROGRESS_FINANCIAL_EARNINGS_CAUSES_AND_PRE_RELEASE_JUDGMENT",
        "continuation_classification": "PROGRESS_PRICE_RESPONSE_POLICY_SCOPE_AND_COMPLETED_BUYBACK",
        "status": "CURRENT_DRIVER_PATH_REFINED_REMAINING_INDEX_EDGE_UNPROVEN",
        "concrete_progress": ["固定阶段价格与最后一日分开", "平安除息与财富变化分开", "核实光模块采购提案范围与供给约束", "核实中际回购已结束", "将经营冲击与风险补偿、实际承接分开"],
        "new_accounts": 0, "new_strategy_return_tests": 0, "strategy_rule_created": False,
        "existing_account_counts": {"admitted": 800, "executed": 952},
        "net_sharpe": None, "net_cagr": None, "performance_status": "NOT_COMPUTED",
        "current_index_return_forecast_made": False, "existing_forecasts_unchanged": True,
        "goal_achieved": False, "goal_status": "active", "not_a_blocked_turn": True, "orders_authorized": False,
        "report": report_name, "price_latest_complete_date": END,
        "next_question": "9月30日新订单信息是否支持原F1/F2；新政策或客户证据是否改变当前风险路径；五家公司之外的指数盈利与价格是否提供不同方向。",
    }
    save(OUT / "result.json", result)
    p = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read(p)
    protocol.update({
        "latest_concrete_diagnostic": "reports/research/510300_current_price_transmission_v1/result.json",
        "latest_price_and_event_transmission": "reports/research/510300_current_price_transmission_v1/reviewed_facts.json",
        "policy_scope_rule": "区分提案、已生效规则、覆盖范围、过渡与替代供给约束；政策风险影响估值不以当前订单已损失为必要条件。",
        "completed_buyback_rule": "方案已完成不再把上限差额当未来承诺买盘；累计金额、计划余额与实际执行分开。",
        "price_causality_rule": "阶段与最后一日反应分别解释；复盘后看到的开盘后跌幅不构成事前交易，单日共现不识别原因份额。",
        "updated_at": STAMP,
    })
    save(p, protocol)
    p = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(p)
    mandate.update({
        "current_round": result["study_id"], "latest_progress_receipt": "reports/research/510300_current_price_transmission_v1/result.json",
        "latest_continuation_report": "reports/research/510300_current_price_transmission_v1/" + report_name,
        "latest_continuation_classification": result["continuation_classification"],
        "last_research_result": "固定五股阶段价格影响约-0.393个百分点，不能解释指数大部分跌幅；核实新增采购提案及中际回购已结束，未识别剩余指数净优势。",
        "last_source_result": "价格快照与分红原件、回购原件已保存；政策官方网页浏览提取可读，直接HTTP403和法案正文缺失保留。",
        "latest_price_and_event_transmission": "reports/research/510300_current_price_transmission_v1/result.json",
        "latest_driver_diagnostic_at": STAMP, "next_research_question": result["next_question"],
    })
    save(p, mandate)
    p = ROOT / "RESEARCH_STATUS.md"
    head = f"""<!-- CURRENT_PRICE_TRANSMISSION_V1_20260929 -->

## 2026-09-29 固定阶段价格、采购提案与回购完成

沿用五家公司，8月31日至9月28日固定权重价格影响约{contribution:+.3f}个百分点，不能解释指数大部分跌幅；最新一日科技相关公司集中调整。官方摘要确认S.5548采购提案的有限范围、过渡与供给约束，正文未取得；中际回购49.97亿元已于9月23日完成，不能把上限差额视作未来承诺买盘。分开经营风险、风险补偿及实际承接。原宏观与FIN1判断不改，新增账户0、收益测试0，目标active且未实现。本轮PROGRESS。

详见[价格反应与新增信息](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_current_price_transmission_v1/价格反应与新增信息.md>)。

"""
    p.write_text(head + p.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"报告": str(OUT / report_name), "五股固定权重影响百分点": contribution,
                      "平安含现金股息财富变化百分比": pa["gross_wealth_change_with_dividend_held_as_cash_pct"],
                      "状态": result["status"], "新增账户": 0, "目标实现": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
