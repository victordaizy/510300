"""保存跨通道政策事实、可用时钟和当时可观察状态。"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from policy_information_clock_v1 import ROOT, OUT, now, save, sha, soup_for


def fact_rows() -> list[dict]:
    rows = []

    def add(identifier, chain, channel, stage, date, upper, source, title, incremental, prior, unresolved,
            checks, amount=None, unit=None, underlying_date=None, precision="DATE_UPPER_BOUND", url=None):
        receipt = json.loads((OUT / "receipts" / f"{source}.json").read_text(encoding="utf-8"))
        path = OUT / receipt["raw_path"]
        if sha(path) != receipt["sha256"]:
            raise ValueError("来源身份不一致")
        if path.suffix == ".html":
            text = re.sub(r"\s+", "", soup_for(source).get_text("", strip=True))
        else:
            text = re.sub(r"\s+", "", path.read_text(encoding="utf-8"))
        for needle in checks:
            if re.sub(r"\s+", "", needle) not in text:
                raise ValueError(f"来源未发现定位内容：{identifier}: {needle}")
        rows.append({"node_id": identifier, "chain": chain, "channel": channel, "stage": stage,
            "economic_event_date": date, "source_available_upper": upper + "+08:00", "time_precision": precision,
            "title": title, "new_or_confirmed_information": incremental, "previously_known": prior,
            "still_unknown_at_this_node": unresolved, "amount": amount, "amount_unit": unit,
            "underlying_execution_date": underlying_date, "expectation": None, "surprise": None,
            "expectation_status": "UNKNOWN_NO_AUTHENTICATED_PRIOR_CONSENSUS",
            "source_key": source, "source_url": url or receipt.get("url"), "source_path": receipt["raw_path"],
            "source_sha256": receipt["sha256"], "source_checks": checks,
            "historical_first_version_authenticated": False,
            "scope": "MANUALLY_VERIFIED_CHAIN_NODE_NOT_INDEPENDENT_EVENT_OR_PREDICTOR"})

    add("R01", "利率与流动性", "利率与流动性", "宣布", "2024-09-24", "2024-09-24T09:19:36", "joint_20240924_csrc",
        "宣布降准与降息", "宣布准备金率降0.5个百分点，7天政策利率拟由1.7%降至1.5%。", "原7天政策利率1.7%；这是政策计划，不是当日操作利率已变。",
        "具体实施时点与贷款、存款利率传导仍待落实；未取得事前共识。", ["2024-09-2409:10:58", "2024-09-2409:19:36", "从目前的1.7%调降至1.5%"], -0.2, "百分点：7天政策利率计划变化", precision="TRANSCRIPT_SEGMENT_END_UPPER_BOUND")
    add("R02", "利率与流动性", "利率与流动性", "实施", "2024-09-27", "2024-09-27T23:59:59", "rate_20240927_govwechat",
        "1.5%政策利率生效", "明确从9月27日起7天逆回购利率为1.5%。", "9月24日已经宣布1.5%目标，不能再次将全部20基点降息当作新意外。",
        "贷款需求和实体融资成本的最终变化尚未知。", ["2024-09-27", "从9月27日起", "1.70%调整为1.50%"], 1.5, "百分比：实施后操作利率", "2024-09-27")
    add("R03", "利率与流动性", "利率与流动性", "宣布", "2025-05-07", "2025-05-07T09:20:49", "joint_20250507_csrc",
        "新一轮降准降息", "宣布准备金率降0.5个百分点，政策利率拟由1.5%降至1.4%。", "此前政策利率1.5%；同时处在贸易摩擦与谈判变化期。",
        "总量工具与结构工具用途不同；政策影响和市场预期差未被量化。", ["2025-05-0709:20:49", "从目前的1.5%调降至1.4%"], -0.1, "百分点：7天政策利率计划变化", precision="TRANSCRIPT_SEGMENT_END_UPPER_BOUND")
    add("R04", "利率与流动性", "利率与流动性", "实施", "2025-05-08", "2025-05-08T09:20:30", "rate_20250508_inherited",
        "1.4%逆回购操作", "当日7天逆回购操作利率确认1.4%。", "前日已宣布降至1.4%，这条是执行确认。", "无法从本条推断新增净信用或股票购买规模。",
        ["2025-05-0809:20:30", "1.40%", "1586亿元"], 1.4, "百分比：当日操作利率", "2025-05-08", "SECOND")

    add("H01", "地产金融", "地产", "宣布", "2024-05-17", "2024-05-17T18:22:00", "housing_20240517_pboc",
        "住房再贷款3000亿", "拟设3000亿元再贷款，预计支持约5000亿元银行贷款；央行支持比例60%。", "政策用途是合格主体收购存量住房。",
        "收购数量、成交价格、贷款实际投放均未知；3000与5000不可相加。", ["2024-05-1718:22:00", "3000亿元", "5000亿元", "60%"], 3000, "亿元：再贷款额度", precision="SECOND")
    add("H02", "地产金融", "地产", "细则调整", "2024-09-24", "2024-09-24T09:19:36", "joint_20240924_csrc",
        "资金支持比例60%→100%", "央行对保障性住房再贷款的资金支持比例拟从60%提高到100%。", "3000亿元工具已于5月宣布；本次改变银行融资激励条件。",
        "不是再新增3000亿元额度，也不能确认全部额度已使用。", ["由原来的60%提高到100%", "2024-09-2409:19:36"], 40, "百分点：央行资金支持比例变化", precision="TRANSCRIPT_SEGMENT_END_UPPER_BOUND")

    add("C01", "资本市场工具", "资本市场", "宣布", "2024-09-24", "2024-09-24T11:42:50", "joint_20240924_csrc",
        "互换便利首期5000亿", "提出首期5000亿元互换便利及3000亿元股票回购增持再贷款。", "早间已经提出将创设两项工具；此段补充金额。",
        "额度不等于当日资金投入；具体操作和申请仍待公布。", ["2024-09-2411:42:50", "互换便利首期操作规模是5000亿元", "首期额度是3000亿元"], 5000, "亿元：互换便利计划额度", precision="TRANSCRIPT_SEGMENT_END_UPPER_BOUND")
    add("C02", "资本市场工具", "资本市场", "细则", "2024-10-18", "2024-10-18T23:59:59", "sfisf_20241018_csrc",
        "互换便利启动操作", "公布期限、质押品与用途；首批申请超2000亿元。", "工具额度已经宣布；申请金额不是实际中标或股票成交。",
        "何时中标、融资和投资规模仍待后续披露。", ["2024-10-18", "首批申请额度已超2000亿元", "互换期限1年"], 2000, "亿元以上：申请额度下界")
    add("C03", "资本市场工具", "资本市场", "细则", "2024-10-18", "2024-10-18T23:59:59", "repurchase_20241018_csrc",
        "回购增持再贷款落地", "首期3000亿元再贷款、利率1.75%，银行先发放合格贷款再按规定申请。", "9月24日已宣布工具；不是新增第二笔3000亿元。",
        "公司借款及回购进度需另查公告；额度不等于已成交。", ["2024-10-18", "3000亿元", "1.75%", "21家"], 3000, "亿元：回购增持再贷款首期额度")
    add("C04", "资本市场工具", "资本市场", "操作结果", "2024-10-21", "2024-10-21T17:00:30", "sfisf_20241021_pboc",
        "互换便利首笔500亿", "首次操作500亿元、中标费率20基点。", "首期工具总额度5000亿元；本次操作只是其一部分。",
        "操作金额不等于同日新增买入股票金额。", ["2024-10-2117:00:30", "500亿元", "中标费率为20bp"], 500, "亿元：当次互换操作", "2024-10-21", "SECOND")
    add("C05", "资本市场工具", "资本市场", "执行进度", "2024-12-31", "2024-12-31T23:59:59", "sfisf_20241231_csrc",
        "披露投放与扩围", "披露首批实际投放超过90%，备选机构从20家增至40家。", "所述90%的基数为首批500亿元操作，不是5000亿元总额度。",
        "尚不能据此获得每日净股票购买路径；该进度的最早披露尚未独立确认。", ["2024-12-31", "实际投放超过90%", "40家备选机构池"], 90, "百分比以上：首批操作投放比例下界")
    add("C06", "资本市场工具", "资本市场", "操作结果", "2025-01-02", "2025-01-02T17:00:30", "sfisf_20250102_pboc",
        "第二次互换550亿", "第二次操作550亿元、中标费率10基点。", "与首笔500亿元是不同次操作；不能把5000亿总额度重复相加。",
        "投资方向和逐日净买入仍需交易或机构资料。", ["2025-01-0217:00:30", "550亿元", "中标费率为10bp"], 550, "亿元：当次互换操作", "2025-01-02", "SECOND")

    add("F01", "财政化债", "财政与债务", "方向", "2024-10-12", "2024-10-12T23:59:59", "fiscal_20241012_mof",
        "预告加力化债", "预告较大规模增加债务限额、支持地方置换隐性债务。", "此前已有化债安排；本条未公布11月的6万亿额度。",
        "新增限额具体数额、年度安排和审批进度未知。", ["2024年10月12日", "较大规模增加债务额度", "隐性债务"])
    add("F02", "财政化债", "财政与债务", "额度与年度安排", "2024-11-08", "2024-11-09T23:59:59", "debt_20241109_mof",
        "6万亿限额分三年", "6万亿元置换限额，2024—2026年每年2万亿元。", "10月12日已预告加力化债；此处明确金额和时间安排。",
        "不是6万亿当期新增最终需求；本来源11月9日发表，不能冒充11月8日盘中可知。", ["2024年11月9日", "每年2万亿元", "6万亿元债务限额"], 60000, "亿元：三年存量债务置换限额", "2024-11-08")
    add("F03", "财政化债", "财政与债务", "执行进度", "2025-01-10", "2025-01-10T23:59:59", "debt_20250110_mof",
        "披露2万亿发行完成", "披露2024年2万亿元置换额度已于12月18日发行完毕。", "额度已于11月批准；这是执行进度，不是再新增2万亿元。",
        "12月18日是执行日期，本来源不能在该日提前入账；是否有更早进度披露尚待追溯。", ["2025年1月10日", "12月18日", "2万亿元置换额度"], 20000, "亿元：已发行的2024年置换额度", "2024-12-18")

    add("D01", "消费与设备更新", "增长与消费", "既有方案解读", "2024-03-14", "2024-03-14T23:59:59", "consumer_20240314_ndrc",
        "两新行动已部署", "说明设备更新、消费品以旧换新等行动与实施机制。", "2月23日已有部署，行动方案已公布；本条是可核对解读，不是最早政策宣布。",
        "本条不能当作7月3000亿元额度已知。", ["2024/03/14", "2月23日", "行动方案"])
    add("D02", "消费与设备更新", "增长与消费", "资金与补贴细则", "2024-07-25", "2024-07-25T23:59:59", "consumer_20240725_govwechat",
        "两新安排约3000亿", "统筹约3000亿元超长期特别国债用于两新；约1500亿元支持地方消费品以旧换新。", "已有两新框架；7月19日国常会已研究加力措施，文件落款7月24日不等于该日公开。",
        "总额度、消费部分和实际使用需分别记录，不与原特别国债总额重复相加。", ["2024-07-25", "2024年7月24日", "3000亿元", "1500亿元"], 3000, "亿元左右：两新合计资金安排")
    add("D03", "消费与设备更新", "增长与消费", "范围扩充", "2025-01-08", "2025-01-08T23:59:59", "consumer_20250108_ndrc",
        "补贴品类8→12", "家电支持由8类扩至12类，增加手机等数码产品购新补贴。", "两新支持已存在，新增内容是适用范围和补贴安排。",
        "不能仅凭扩围推断消费净增量，提前消费、替代及执行进度需另检验。", ["2025/01/08", "8类增加到2025年的12类", "手机等数码产品购新补贴"], 4, "类：家电补贴品类增加")

    add("T01", "中美贸易政策", "外部与贸易", "宣布", "2025-04-04", "2025-04-04T23:59:59", "trade_20250404_mof",
        "宣布反制加征34%", "宣布在现行适用税率基础上对美国商品加征34%，计划4月10日生效。", "针对美国4月2日措施作出回应。", "实际生效前政策仍可被后续公告替代。",
        ["2025年04月04日", "34%", "4月10日12时01分"], 34, "百分点：本轮加征部分", "2025-04-10T12:01:00")
    add("T02", "中美贸易政策", "外部与贸易", "替代先前安排", "2025-04-09", "2025-04-09T23:59:59", "trade_20250409_mof",
        "34%调整至84%", "在原定生效时点前，将本轮加征税率由34%改为84%。", "34%的计划尚未按原方案生效；不能累计成34%+84%。", "后续措施仍可能变化。",
        ["2025年04月09日", "由34%提高至84%", "4月10日12时01分"], 84, "百分点：替代后的本轮加征部分", "2025-04-10T12:01:00")
    web = "raw/trade_three_web_rendered.txt"
    add("T03", "中美贸易政策", "外部与贸易", "再调整", "2025-04-11", "2025-04-11T23:59:59", "trade_three_web_rendered",
        "84%调整至125%", "宣布4月12日起将本轮加征税率由84%改为125%。", "84%的安排已经实施；新公告取代这一税率。", "没有量化市场事前预计的税率。",
        ["2025年4月11日", "由84%提高至125%"], 125, "百分点：替代后的本轮加征部分", "2025-04-12",
        url="https://gss.mof.gov.cn/gzdt/zhengcejiedu/202504/t20250411_3961824.htm")
    add("T04", "中美贸易政策", "外部与贸易", "谈判确认", "2025-05-07", "2025-05-07T06:15:59", "trade_three_web_rendered",
        "盘前确认将举行会谈", "确认访问瑞士期间将举行中美经贸高层会谈。", "双方此前已释放接触信号；是否达成协议尚未知。", "关税降幅、暂停期限和细节均未公布。",
        ["2025-05-0706:15", "将于5月9日-12日访问瑞士"], precision="MINUTE_UPPER_BOUND",
        url="https://www.mofcom.gov.cn/xwfb/xwfyrth/art/2025/art_89f7cb660ffe49e6a70bfb778ede6e22.html")
    add("T05", "中美贸易政策", "外部与贸易", "谈判进展", "2025-05-12", "2025-05-12T05:30:59", "trade_three_web_rendered",
        "盘前披露实质性进展", "披露会谈取得实质性进展，并将公布联合声明。", "会谈已经预告；盘前已增加取得进展的信息。", "本条仍未给出具体关税调整数值，不能提前引用下午声明。",
        ["2025-05-1205:30", "实质性进展", "将于5月12日发布"], underlying_date="2025-05-11_GENEVA_EVENING", precision="MINUTE_UPPER_BOUND",
        url="https://www.mofcom.gov.cn/xwfb/ldrhd/art/2025/art_0adb7ac71bca40f2ab9af8e3d1eefcbb.html")
    add("T06", "中美贸易政策", "外部与贸易", "具体安排", "2025-05-12", "2025-05-12T15:00:59", "trade_20250512_inherited",
        "关税细节与90日期限", "公布取消91个百分点、暂停24个百分点90天并保留10个百分点相关加征安排。", "当日盘前已经公布取得实质性进展；下午才知道此处具体细节。",
        "10%不是所有商品总税率，90天暂停不是永久取消；剩余可交易反应需独立检验。", ["2025-05-1215:00", "90天", "剩余10%"], 90, "天：部分加征关税暂停期限", precision="MINUTE_UPPER_BOUND")
    return rows


def clock_fields(nodes: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(market.date)).tz_localize("Asia/Shanghai")
    closes, opens = dates + pd.Timedelta(hours=15), dates + pd.Timedelta(hours=9, minutes=30)
    fields = []
    for row in nodes.to_dict("records"):
        upper = pd.Timestamp(row["source_available_upper"])
        i = int(closes.searchsorted(upper, side="left"))
        oi = int(opens.searchsorted(upper, side="left"))
        fields.append({"node_id": row["node_id"], "review_close": dates[i].strftime("%Y-%m-%d"),
            "research_execution_open": dates[i + 1].strftime("%Y-%m-%d"),
            "first_daily_open_after_source_upper": dates[oi].strftime("%Y-%m-%d"),
            "execution_note": "研究统一等待首次可复核收盘再于下一开盘执行；不声称这是最早可交易时点。日期精度来源可能保守延迟。"})
    return nodes.merge(pd.DataFrame(fields), on="node_id", validate="one_to_one")


def state_frame(nodes: pd.DataFrame, market: pd.DataFrame, money: pd.DataFrame, pmi: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    market = market.copy()
    market["date"] = pd.to_datetime(market.date)
    market["price_return20"] = market.close.pct_change(20)
    market["price_vol20_annual242"] = market.close.pct_change().rolling(20).std(ddof=1) * np.sqrt(242)
    market["price_drawdown60"] = market.close / market.close.rolling(60).max() - 1
    money = money.copy()
    money["known"] = pd.to_datetime(money.available_at_upper_bound, utc=True).dt.tz_convert("Asia/Shanghai")
    pmi = pmi.copy()
    pmi["known"] = pd.to_datetime(pmi.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    rates = rates.copy()
    rates["known"] = pd.to_datetime(rates.published_at).dt.tz_localize("Asia/Shanghai")
    result = []
    for row in nodes.to_dict("records"):
        cutoff = pd.Timestamp(row["review_close"], tz="Asia/Shanghai") + pd.Timedelta(hours=15)
        mr = market.loc[market.date == cutoff.tz_localize(None).normalize()].iloc[0]
        mm = money[money.known <= cutoff].sort_values("known").iloc[-1]
        pp = pmi[pmi.known <= cutoff].sort_values("known").iloc[-1]
        rr = rates[rates.known <= cutoff].sort_values("known").iloc[-1]
        result.append({"node_id": row["node_id"], "review_close": row["review_close"], "close": float(mr.close),
            "price_return20": float(mr.price_return20), "price_vol20_annual242": float(mr.price_vol20_annual242),
            "price_drawdown60": float(mr.price_drawdown60), "money_month": mm.stat_month,
            "money_available": str(mm.known), "m1_yoy_pp": float(mm.m1_yoy_pp), "m2_yoy_pp": float(mm.m2_yoy_pp),
            "spread_m1_minus_m2_pp": float(mm.m1_yoy_pp - mm.m2_yoy_pp), "money_definition": mm.definition_version,
            "pmi_month": pp.reference_period, "pmi_available": str(pp.known), "pmi_new_orders": float(pp.first_release_value),
            "latest_operation_notice_date": str(rr.notice_date)[:10], "latest_operation_notice_available": str(rr.known),
            "latest_operation_notice_rate_percent": float(rr.seven_day_rate_percent),
            "rate_note": "原25条操作记录只作来源状态，可能晚于政策宣布或实施；不得当成完整首次宣布序列。",
            "state_use": "描述当时状态，无拟合系数、政策方向评分或买卖建议；过去60日回撤不是优化出的退出规则。"})
    return pd.DataFrame(result)


def main() -> None:
    if (OUT / "inputs/policy_nodes_freeze.json").exists():
        raise FileExistsError("政策事实已固定；不要覆盖，请用离线复核入口检查")
    sources = {
        "market.parquet": ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet",
        "money_104.csv": ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/money_104.csv",
        "pmi_new_orders.parquet": ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/pmi_new_orders.parquet",
        "operation_rate_records.parquet": ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/operation_rate_records.parquet",
    }
    identities = []
    for name, p in sources.items():
        dst = OUT / "inputs" / name
        shutil.copy2(p, dst)
        identities.append({"name": name, "sha256": sha(dst), "source": str(p.relative_to(ROOT))})
    rows = fact_rows()
    save(OUT / "inputs/policy_nodes.json", rows)
    shutil.copy2(__file__, OUT / "code/build_policy_chain_v1.py")
    save(OUT / "inputs/policy_nodes_freeze.json", {"at": now(), "node_count": len(rows), "chain_count": len(set(r["chain"] for r in rows)),
        "nodes_sha256": sha(OUT / "inputs/policy_nodes.json"), "code_sha256": sha(OUT / "code/build_policy_chain_v1.py"), "inputs": identities,
        "new_return_labels_or_models": 0, "information_role": "跨通道人工来源链示例；未按收益选取，但不是政策全集，也不是预注册策略。",
        "expectation_missing_is_zero": False, "clock_clarification": "另列可用上界之后的首个日频开盘，与固定收盘复核后下一开盘分开；新增列只说明时钟，不改变协议执行规则。"})
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    nodes = clock_fields(pd.DataFrame(rows), market)
    flat = nodes.copy()
    flat["source_checks"] = flat.source_checks.map(lambda x: json.dumps(x, ensure_ascii=False))
    flat.to_csv(OUT / "results/跨通道政策链_完整事实与时钟.csv", index=False, encoding="utf-8-sig")
    nodes.to_parquet(OUT / "results/跨通道政策链_完整事实与时钟.parquet", index=False)
    states = state_frame(nodes, market, pd.read_csv(OUT / "inputs/money_104.csv"), pd.read_parquet(OUT / "inputs/pmi_new_orders.parquet"), pd.read_parquet(OUT / "inputs/operation_rate_records.parquet"))
    states.to_csv(OUT / "results/政策复核时点_当时可观察状态.csv", index=False, encoding="utf-8-sig")
    states.to_parquet(OUT / "results/政策复核时点_当时可观察状态.parquet", index=False)
    summary = {"stage_status": "COMPLETED_FIXED_CATALOG_EXTRACTION_AND_SIX_POLICY_CHAINS",
        "nodes": len(nodes), "chains": int(nodes.chain.nunique()), "policy_consensus_qualified_nodes": 0,
        "global_first_publication_proven_nodes": 0, "historical_availability_bounds_with_source": len(nodes),
        "new_model_fits": 0, "new_accounts": 0, "new_exit_comparisons": 0, "independent_forward_events": 0,
        "return_prediction": "NOT_RUN_POLICY_MODULE_NO_FROZEN_MODEL", "risk_prediction": "NOT_RUN_POLICY_MODULE_NO_FROZEN_MODEL",
        "main_200k_account": "NOT_RUN_POLICY_INFORMATION_INCREMENT_NOT_ESTABLISHED", "cost_comparison_20k": "NOT_RUN_SAME_GATE",
        "whole_macro_objective_complete": False, "tradable_signal_status": "NO_VIEW_NO_VALIDATED_POLICY_RULE",
        "conclusion": "政策通道不能被剪刀差取代；有必要保留经济用途、已知计划、兑现进度、政策替代和公布时钟。新资料建立动态更新输入结构，尚未建立未来收益或下行风险正增量。",
        "historical_price_end": str(pd.to_datetime(market.date).max().date())}
    save(OUT / "results/result.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
