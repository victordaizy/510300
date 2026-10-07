"""比较已经发生的追保与质押纾困信息，保留同一政策链及亏损案例。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from zoneinfo import ZoneInfo
from zipfile import ZipFile
from xml.etree import ElementTree

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_collateral_relief_cases_v1"
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
MARGIN = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs/margin.parquet"
STUDY = "510300_HISTORICAL_COLLATERAL_RELIEF_CASES_V1"

SOURCES = [
    ("draft_20150612", "2015-06-12", "https://www.csrc.gov.cn/csrc/c100028/c1001933/content.shtml", "允许融资融券合约合理展期", "2015年融资融券规则征求意见"),
    ("offshore_20150612", "2015-06-12", "https://www.csrc.gov.cn/csrc/c100029/c1000237/content.shtml", "场外配资", "同日发布会及外部接入监管"),
    ("final_20150701", "2015-07-01", "https://www.csrc.gov.cn/csrc/c100028/c1001916/content.shtml", "2个交易日", "2015年融资融券办法正式发布"),
    ("sse_rule_20150701", "2015-07-01", "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988867.shtml", "上证发〔2015〕64号", "配套细则及附件入口"),
    ("szse_20180626", "2018-06-26", "https://www.szse.cn/aboutus/trends/news/t20180626_552064.html", "展期", "2018年6月股票质押风险及展期支持说明"),
    ("cbirc_20181019", "2018-10-19", "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20181019_4658804.shtml", "质权人", "上交所保存的银保监会负责人答记者问"),
    ("sse_20181021", "2018-10-21", "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20181021_4659845.shtml", "股票质押式回购", "上交所质押风险与资金支持工作部署"),
    ("pledge_rule_20190118", "2019-01-18", "https://www.sse.com.cn/lawandrules/sselawsrules/trade/specific/repo/c/c_20210128_5312116.shtml", "延期后累计的回购期限可以超过3年", "2018年纾困后续正式展期规则"),
    ("pledge_review_20190118", "2019-01-18", "https://www.szse.cn/aboutus/trends/news/t20190118_564239.html", "114亿元", "2018年度股票质押回购风险分析报告"),
    ("margin_rule_20190809", "2019-08-09", "https://www.csrc.gov.cn/csrc/c100028/c1000931/content.shtml", "130%", "130%统一最低比例取消的实际时点核对，非收益事件"),
    ("sse_rule_doc_20150701", "2015-07-01", "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/10058402/files/741bc6a39a244725be269865a9cef34c.docx", "130%", "2015年实施细则原附件，核对最低比例与追保安排"),
]

EVENTS = [
    {"id": "M15_DRAFT", "date": "2015-06-12", "available_date": "2015-06-12", "label": "两融展期及处置方式草案", "cluster": "2015年两融规则变化", "source_keys": ["draft_20150612", "offshore_20150612"], "stage": "征求意见，尚未实施", "changed_constraint": "拟允许合约展期并调整违约处置；同日加强场外配资外部接入监管", "not_established": "不是全渠道放松；不能认定场外杠杆全部清理或草案已经生效"},
    {"id": "M15_FINAL", "date": "2015-07-01", "available_date": "2015-07-01", "label": "两融展期及追保规则实施", "cluster": "2015年两融规则变化", "source_keys": ["final_20150701", "sse_rule_20150701"], "stage": "正式发布并实施", "changed_constraint": "允许协商展期，放宽补充担保物期限和比例安排，处置方式更灵活", "not_established": "6月12日已公开主要方向；不是取消债务、所有担保要求或全部强平"},
    {"id": "P18_JUNE", "date": "2018-06-26", "available_date": "2018-06-26", "label": "深市质押展期支持说明", "cluster": "2018年至2019年股票质押纾困", "source_keys": ["szse_20180626"], "stage": "存量处置说明及继续支持", "changed_constraint": "对经营正常但临时资金困难的融资人提供必要展期；披露协商及补担保做法", "not_established": "限深市口径；不是全国统一停平仓或新增现金已到账"},
    {"id": "P18_OCT", "date": "2018-10-19", "available_date": "2018-10-21", "label": "质押处置与纾困资金部署", "cluster": "2018年至2019年股票质押纾困", "source_keys": ["cbirc_20181019", "sse_20181021"], "stage": "监管要求及工作部署", "changed_constraint": "质权人综合评估后稳妥处置，允许保险专项产品，部署完善质押处置及融资工具", "not_established": "没有统一禁止平仓；同日多项政策及周末信息合并，不能单独归因"},
    {"id": "P19_FINAL", "date": "2019-01-18", "available_date": "2019-01-18", "label": "质押违约合约展期规则实施", "cluster": "2018年至2019年股票质押纾困", "source_keys": ["pledge_rule_20190118"], "stage": "正式发布并实施", "changed_constraint": "符合条件的违约合约经双方协商可超过3年；全部用于偿还旧违约债务的新融资适用部分规则豁免", "not_established": "借新还旧并非等额新增股票购买；不能提前写成2018年已实施规则"},
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(name, value):
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本轮设定已经保存，不覆盖；采集和分析分别执行。")
    parent = json.loads((ROOT / "reports/research/510300_historical_leverage_seller_constraints_v1/protocol.json").read_text(encoding="utf-8"))
    protocol = {
        "study_id": STUDY, "recorded_at": now(), "research_mode": "HISTORICAL_ONLY",
        "previous_goal_turn_classification": "PROGRESS_HISTORICAL_BORROWING_REPAYMENT_AND_SELLER_CONSTRAINTS",
        "question": "不同债务类型的展期、追保与处置规则改变后，融资活动如何变化，公开后可成交的510300收益能否重复？",
        "sample_selection": "从上一轮明确的2015和2018年问题出发，按原始规则内容确定5个节点。2018年纾困的超过3年展期细则实际于2019年1月发布，作为同一政策链后续节点纳入，先于本轮收益计算。不是全部纾困政策的穷尽事件库。",
        "events": EVENTS, "historical_market_previously_seen": True,
        "independent_validation": False, "before_this_study_returns": True,
        "observation_windows": [["2015-05-04", "2015-08-31"], ["2018-05-02", "2019-03-29"]],
        "holding_sessions": [5, 20], "primary_horizon": 5,
        "entry": "各节点可得日期日末后的下一实际交易日开盘；2018年10月19至21日合并后仍为10月22日。",
        "exit": "入场后第5或第20个交易日开盘，不搜索更有利期限。",
        "pre_post_financing": "以入场前5个和入场起5个完整交易日比较买入、反推偿还与净变化；后5日是事后机制观察，绝不作为同次入场信息。两融不代替质押或场外杠杆。",
        "drawdown_description": "期间收盘财富相对入场开盘的最低回报，含截至该收盘的除息现金；不是可保证成交止损或完整账户回撤。",
        "costs": parent["event_return"]["costs"],
        "execution_limit": "10万元参考往返按开盘预算换算数量，未模拟预提交订单、盘口、风险预算及全账户；不计算策略夏普。",
        "comparisons": "保留草案和实施、说明和实施各自结果；同一政策链不计为独立重复成功。复用2024年2月5日参考结果作已见对照，不重新筛选。",
        "source_role": "2019年1月年度质押报告仅供2018年事后解释，不能回填2018年信号；2019年8月两融说明只核对规则区别，不增加收益事件。",
        "old_studies": ["reports/research/510300_historical_policy_constraint_events_v1/result.json", "reports/research/510300_historical_leverage_seller_constraints_v1/result.json"],
        "new_parameters_fitted": 0, "new_accounts": 0, "new_prospective_forecasts": 0,
        "goal_achieved": False, "orders_authorized": False,
    }
    save("protocol.json", protocol)
    print("已固定5个历史信息节点、5日和20日期限；尚未计算本轮收益。", flush=True)


def fetch():
    if not (OUT / "protocol.json").exists():
        raise RuntimeError("先保存事件和期限。")
    raw = OUT / "raw"
    raw.mkdir(exist_ok=True)
    existing = {r["key"]: r for r in json.loads((OUT / "source_index.json").read_text(encoding="utf-8"))} if (OUT / "source_index.json").exists() else {}

    def one(spec):
        key, date, url, check, purpose = spec
        destination = raw / (key + (".docx" if url.endswith(".docx") else ".html"))
        if key in existing and existing[key]["status"] == "CONTENT_PRESENT" and destination.exists():
            return existing[key]
        row = {"key": key, "publication_date": date, "url": url, "purpose": purpose, "retrieved_at": now(), "historical_first_vintage_authenticated": False}
        if key in existing:
            row["previous_attempt"] = existing[key]
        try:
            if destination.exists():
                body = destination.read_bytes()
                row["mode"] = "REUSE_THIS_STUDY_RAW"
            else:
                response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
                row.update(http_status=response.status_code, final_url=response.url, mode="NEW_PUBLIC_SOURCE")
                response.raise_for_status()
                body = response.content
                destination.write_bytes(body)
            if url.endswith(".docx"):
                with ZipFile(BytesIO(body)) as archive:
                    document = ElementTree.fromstring(archive.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                plain = "\n".join("".join(p.itertext()) for p in document.findall(".//w:p", ns))
            else:
                soup = BeautifulSoup(body, "html.parser", from_encoding="utf-8")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                plain = soup.get_text("\n", strip=True)
            (raw / (key+".txt")).write_text(plain, encoding="utf-8")
            matches = re.sub(r"\s+", "", check) in re.sub(r"\s+", "", plain)
            row.update(status="CONTENT_PRESENT" if matches else "BODY_NOT_CONFIRMED", raw_path=destination.relative_to(ROOT).as_posix(), raw_sha256=hashlib.sha256(body).hexdigest(), required_text=check, required_text_present=matches)
            if key == "sse_rule_20150701":
                row["attachment_links"] = [{"text": a.get_text(strip=True), "url": urljoin(url, a["href"])} for a in soup.find_all("a", href=True) if "2015年修订" in a.get_text()]
        except requests.RequestException as exc:
            row.update(status="SOURCE_REQUEST_FAILED", error=str(exc))
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(one, SOURCES))
    save("source_index.json", sources)
    print(json.dumps([{"key": r["key"], "status": r["status"], "attachment_links": r.get("attachment_links", [])} for r in sources], ensure_ascii=False), flush=True)


def source_facts(sources):
    mapping = {r["key"]: r for r in sources}
    assert all(r["status"] == "CONTENT_PRESENT" for r in sources), "仍有原文未取得。"
    facts = [
        ("draft_20150612", "2015-06-12", "拟允许展期与优化担保物违约处置已在6月12日公开征求意见。", "草案尚未实施；7月1日不应被视为这些方向首次出现，但正式落地仍改变实施确定性。"),
        ("offshore_20150612", "2015-06-12", "同日发布会要求检查外部接入，重申不得为场外配资及非法证券业务提供便利。", "合法两融条款放宽与场外渠道监管并存；两融余额不覆盖全部场外杠杆。"),
        ("sse_rule_doc_20150701", "2015-07-01", "第十八条允许展期，每次不超过6个月；第四十三条仍保留130%最低比例，追保期限和追保后比例可协商。", "新增处置弹性不等于取消担保要求或债务。"),
        ("margin_rule_20190809", "2019-08-09", "这一日期公布取消130%统一最低限制，交由券商与客户自主约定。", "只用于核对制度沿革，不能前移至2015年，也不表示券商不设风险线。"),
        ("szse_20180626", "2018-06-26", "此前一周深市股票质押实际日均违约处置约3000万元；对经营正常但临时资金困难的融资人继续提供必要展期支持。", "只涵盖其统计对象和期限；没有全国不平仓承诺，也不测量防御性提前卖出。"),
        ("cbirc_20181019", "2018-10-19", "要求银行质权人综合评估后稳妥处置，允许保险专项产品参与质押流动性风险化解。", "区别银行质押债权、两融账户与保险专项产品；无全市场停平仓指令。"),
        ("pledge_rule_20190118", "2019-01-18", "符合条件且双方同意的违约合约可超过3年；新增融资全部还旧违约债务时，适用部分集中度和质押率等规则豁免。", "新贷款对应旧债偿还，不能等额算作股票净买入；各户执行及违约损失没有归零。"),
        ("pledge_review_20190118", "2018-12-31", "年末低于约定保障比例的质押市值2990亿元，全年控股股东及一致行动人申报违约金额482亿元，全年二级市场处置114亿元。", "存量市值、年度特定主体申报量与实际卖出处置不同；不相减当作被拯救规模，不计算强平贡献率。"),
        ("pledge_review_20190118", "2018-12-31", "高比例质押公司中，沪深300成员26家，占成员数8.7%；2017年减持约束后，处置流动性下降而平均质押率仅降3个百分点。", "26家是数量而非指数权重；这是2019年公布的事后解释，不能作为2018年已知输入。"),
    ]
    result = []
    for i, (key, cutoff, fact, limit) in enumerate(facts, start=1):
        src = mapping[key]
        result.append({"id": f"F{i:02d}", "fact": fact, "interpretation_limit": limit, "economic_date_or_cutoff": cutoff,
                       "publication_date": src["publication_date"], "source_key": key, "source_url": src["url"], "raw_path": src["raw_path"]})
    save("规则变化与上游约束.json", result)
    return mapping, result


def financing_block(frame):
    assert len(frame) == 5 and frame[["market_rzye", "market_rzmre", "balance_change", "implied_repayment"]].notna().all().all(), "融资五日窗口不完整。"
    return {"first": frame.date.iloc[0], "last": frame.date.iloc[-1], "days": len(frame),
            "buy_daily": float(frame.market_rzmre.mean()), "repay_daily": float(frame.implied_repayment.mean()),
            "net_daily": float(frame.balance_change.mean()), "balance_change": float(frame.balance_change.sum())}


def analyze():
    cfg = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert cfg["events"] == EVENTS and cfg["holding_sessions"] == [5, 20], "事件与期限不得临时调整。"
    sources = json.loads((OUT / "source_index.json").read_text(encoding="utf-8"))
    source_map, facts = source_facts(sources)
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date)
    margin = pd.read_parquet(MARGIN).sort_values("date").reset_index(drop=True)
    margin["date"] = pd.to_datetime(margin.date)
    assert market.date.is_unique and margin.date.is_unique
    margin["balance_change"] = margin.market_rzye.diff()
    margin["implied_repayment"] = margin.market_rzmre - margin.balance_change
    dividend = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividend = dividend[dividend.symbol.eq("510300.SH")].copy()
    dividend["record_date"] = pd.to_datetime(dividend.record_date)
    observations = []
    for start, stop in cfg["observation_windows"]:
        p = margin[margin.date.between(start, stop)]
        assert set(p.date) == set(market[market.date.between(start, stop)].date), "日历与融资覆盖不同。"
        assert p.implied_repayment.ge(0).all()
        observations.extend(p[["date", "market_rzye", "market_rzmre", "balance_change", "implied_repayment", "source"]].to_dict("records"))
    save("固定阶段融资日线.json", observations)

    rows = []
    for spec in cfg["events"]:
        i = int(market.date.searchsorted(pd.Timestamp(spec["available_date"]), side="right"))
        entry = market.iloc[i]
        assert entry.date > pd.Timestamp(spec["available_date"])
        j = int(margin.date.searchsorted(entry.date))
        assert margin.date.iloc[j] == entry.date
        before = financing_block(margin.iloc[j-5:j])
        after = financing_block(margin.iloc[j:j+5])
        buying_change = after["buy_daily"] - before["buy_daily"]
        repay_contribution = before["repay_daily"] - after["repay_daily"]
        net_shift = after["net_daily"] - before["net_daily"]
        assert abs(net_shift-buying_change-repay_contribution) < .01
        anchor = int(market.date.searchsorted(pd.Timestamp(spec["date"]), side="left"))-1
        initial_reflection = market.iloc[i-1].wealth/market.iloc[anchor].wealth * (entry.open+entry.dividend)/market.iloc[i-1].close-1
        row = {**spec, "source_urls": [source_map[k]["url"] for k in spec["source_keys"]],
               "entry_date": entry.date, "entry_open": float(entry.open),
               "already_reflected_from_pre_information_close": float(initial_reflection),
               "first_version_authenticated": False, "precise_surprise_measured": False,
               "financing_before": before, "financing_after": after,
               "buying_change_contribution": buying_change, "repayment_change_contribution": repay_contribution,
               "net_flow_change": net_shift, "post_five_day_activity_is_entry_information": False}
        for horizon in cfg["holding_sessions"]:
            end = market.iloc[i+horizon]
            entitlement = float(dividend.loc[dividend.record_date.ge(entry.date) & dividend.record_date.lt(end.date), "cash_dividend_per_share"].sum())
            trade = round_trip(entry.open, end.open, entitlement, cfg["costs"])
            held = market.iloc[i:i+horizon].copy()
            daily_div = held.dividend.astype(float).copy()
            daily_div.iloc[0] = 0.0
            close_return = (held.close.to_numpy()+daily_div.cumsum().to_numpy())/entry.open-1
            worst_i = int(np.argmin(close_return))
            row[f"holding_{horizon}"] = {"exit_date": end.date, "exit_open": float(end.open),
                                         "gross_return": float((end.open+entitlement)/entry.open-1),
                                         "dividend_per_share": entitlement,
                                         "lowest_close_return_from_entry": float(min(0.0, close_return.min())),
                                         "lowest_close_date": held.date.iloc[worst_i] if close_return.min() < 0 else entry.date,
                                         **trade}
            assert abs(trade["shares"]*(trade["sell_price"]-trade["buy_price"]+entitlement)-trade["commissions_cny"]-trade["net_pnl_cny"]) < 1e-7
        row["overlapping_earlier_20_day_windows"] = [r["id"] for r in rows if entry.date < r["holding_20"]["exit_date"]]
        rows.append(row)
    save("五个历史节点的收益与融资分解.json", rows)
    old_events = json.loads((ROOT / "reports/research/510300_historical_leverage_seller_constraints_v1/四个信息时点与费用后参考收益.json").read_text(encoding="utf-8"))
    control = next(r for r in old_events if r["source_date"] == "2024-02-05")
    control_j = int(margin.date.searchsorted(pd.Timestamp(control["entry_date"])))
    control["financing_before"] = financing_block(margin.iloc[control_j-5:control_j])
    control["financing_after"] = financing_block(margin.iloc[control_j:control_j+5])
    save("2024年既有对照.json", {"mode": "OLD_RETURNS_UNCHANGED_NEW_FIXED_WINDOW_ACTIVITY_NOT_INDEPENDENT", "original_path": "reports/research/510300_historical_leverage_seller_constraints_v1/四个信息时点与费用后参考收益.json", "event": control})
    chart(rows, control, market, margin)
    make_report(rows, control, source_map)
    result = {"study_id": STUDY, "recorded_at": now(), "status": "HISTORICAL_COLLATERAL_RELIEF_MECHANISMS_AND_RETURNS_COMPARED",
              "classification": "PROGRESS_HISTORICAL_CROSS_EPISODE_COLLATERAL_CONSTRAINT_COMPARISON",
              "research_mode": "HISTORICAL_ONLY", "historical_information_nodes": len(rows), "policy_chains": 2,
              "independent_event_count_established": False, "daily_margin_observations": len(observations),
              "official_source_documents": len(sources), "net_returns": [{"id": r["id"], "net5": r["holding_5"]["net_return"], "net20": r["holding_20"]["net_return"], "worst_close20": r["holding_20"]["lowest_close_return_from_entry"]} for r in rows],
              "new_full_accounts": 0, "new_fitted_parameters": 0, "new_forecast_cards": 0,
              "net_sharpe": None, "net_cagr": None, "goal_achieved": False, "orders_authorized": False,
              "causal_effect_identified": False, "report": (OUT / "历史发现_展期缓释与剩余收益.md").relative_to(ROOT).as_posix()}
    result["findings"] = [
        "2015年放宽追保期限和处置弹性仍保留130%最低比例；与2019年取消统一最低比例不同。",
        "2015年草案公开后五日融资余额仍净增，价格却明显下跌；融资余额增长不等于价格风险消失。",
        "2015年正式实施后，融资买入下降与偿还上升同时发生，支持公告不足以保证可交易修复。",
        "2018年10月融资买入恢复、净流出收窄与2024年有表面相似性，但其5日费用后收益为负；这一组合也不是充分条件。",
        "2018年质押纾困与2019年规则实施后回报不同，节点相关且同时存在其他政策，不能建立因果胜率或拼成账户。",
    ]
    result["next_research_question"] = "比较2018年10月与2024年2月这两个融资买入恢复却收益不同的历史节点：先检查已有研究，再查当时的510300净值与溢价、ETF份额、权重行业承接和已发生反应；观察确认后的剩余收益，不把事后五日资金变化提前用作入场。"
    save("result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def chart(rows, control, market, margin):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    cases = [{"date": r["entry_date"], "title": r["date"]+"\n"+r["label"], "net5": r["holding_5"]["net_return"], "net20": r["holding_20"]["net_return"]} for r in rows]
    cases.append({"date": pd.Timestamp(control["entry_date"]), "title": "2024-02-05\n两融缓释说明 既有对照", "net5": control["holding_5"]["net_return"], "net20": control["holding_20"]["net_return"]})
    for ax, case in zip(axes.flat, cases):
        i = int(market.date.searchsorted(case["date"]))
        p = market.iloc[i:i+20].copy()
        d = p.dividend.astype(float).copy()
        d.iloc[0] = 0.0
        price_path = (p.close.to_numpy()+d.cumsum().to_numpy())/p.open.iloc[0]-1
        j = int(margin.date.searchsorted(case["date"]))
        b = margin.iloc[j:j+20]
        assert np.array_equal(p.date.to_numpy(), b.date.to_numpy()), "图中交易日不一致。"
        debt_path = b.market_rzye.to_numpy()/margin.market_rzye.iloc[j-1]-1
        ax.plot(np.arange(21), np.r_[0, price_path*100], color="#007F77", label="510300相对入场开盘", linewidth=2)
        ax.plot(np.arange(21), np.r_[0, debt_path*100], color="#B96D43", label="融资余额相对前日", linewidth=1.6)
        ax.axhline(0, color="#84909D", linewidth=.7)
        ax.axvline(5, color="#B5BDC6", linestyle="--", linewidth=.7)
        ax.set_title(case["title"], loc="left", fontsize=10, pad=11)
        ax.set_xlabel("入场后第几个交易日的收盘")
        ax.set_ylabel("变动 %")
        ax.grid(axis="y", alpha=.18)
        ax.text(.03,.04,f"参考交易净收益 5日 {case['net5']:+.2%} / 20日 {case['net20']:+.2%}", transform=ax.transAxes, fontsize=8, bbox={"facecolor":"white", "edgecolor":"none", "alpha":.8})
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.95), ncol=2, frameon=False)
    fig.suptitle("展期和处置缓释之后 价格与融资余额并不总是同时修复", fontsize=15, y=.994, fontweight="bold")
    fig.text(.04,.016,"曲线为入场后20个交易日收盘路径，不含费用；标注交易收益按5日或20日后的下一开盘退出并扣费。两融余额不代表股票质押或全部场外杠杆。", fontsize=9, color="#526170")
    fig.tight_layout(rect=[0,.045,1,.91], h_pad=2.3, w_pad=2)
    fig.savefig(OUT / "展期信息后的价格与融资路径.png", dpi=160, facecolor="white")
    plt.close(fig)


def make_report(rows, control, sources):
    def link(key, title):
        return f"[{title}]({sources[key]['url']})"

    def date(value):
        return str(pd.Timestamp(value).date())

    returns_table = "\n".join(
        f"| {r['date']} {r['label']} | {date(r['entry_date'])} | {r['holding_5']['net_return']:+.2%} | {r['holding_20']['net_return']:+.2%} | {r['holding_20']['lowest_close_return_from_entry']:.2%} |"
        for r in rows)
    returns_table += f"\n| 2024-02-05 两融缓释说明 已有对照 | {date(control['entry_date'])} | {control['holding_5']['net_return']:+.2%} | {control['holding_20']['net_return']:+.2%} | 未在本轮重算 |"
    financing_table = "\n".join(
        f"| {r['date']} | {r['financing_before']['buy_daily']/1e8:.2f} → {r['financing_after']['buy_daily']/1e8:.2f} | {r['financing_before']['repay_daily']/1e8:.2f} → {r['financing_after']['repay_daily']/1e8:.2f} | {r['financing_before']['net_daily']/1e8:+.2f} → {r['financing_after']['net_daily']/1e8:+.2f} |"
        for r in rows)
    financing_table += f"\n| 2024-02-05 对照 | {control['financing_before']['buy_daily']/1e8:.2f} → {control['financing_after']['buy_daily']/1e8:.2f} | {control['financing_before']['repay_daily']/1e8:.2f} → {control['financing_after']['repay_daily']/1e8:.2f} | {control['financing_before']['net_daily']/1e8:+.2f} → {control['financing_after']['net_daily']/1e8:+.2f} |"
    draft, final, june, october, january = rows
    body = f"""# 历史展期与追保缓释后的价格和融资变化

本轮比较2015年两融规则变化、2018年股票质押纾困及2019年后续规则，检验约束缓释能否形成公开后可成交的修复收益。结论是：支持公告本身不足以形成买入依据；融资买入恢复、净流出缩小也不保证随后五日获利。2024年的成功案例需要进一步解释其条件，不能直接复制为普遍规则。

五个节点按原文和前轮研究问题确定，收益期限统一为5日、20日；保留全部节点和亏损，不选择最好的起点或期限。2015年的草案与实施、2018年至2019年的质押纾困属于两条政策链，五个节点不是五次独立试验。历史行情此前已被研究，这是一轮机制发现，不是独立验证。

**先看实际改变了什么。** 股票下跌使担保资产相对债务减少，账户可能需要追加担保或偿还债务。展期缓解到期压力，延长追保时间缓解处置紧迫性，扩大担保物则改变能够提供何种资产。这些安排可以改变交易约束，但没有自动消除债务、恢复借款人的偿付能力，也没有使所有参与者停止卖出。

2015年6月12日已经公开拟允许展期和优化违约处置，7月1日正式发布。7月1日早于原定7月11日征求意见截止，提前落地仍有信息增量；不能把整个方向当作首次出现，也不能认为实施完全没有新信息。同日6月12日发布会还加强场外配资外部接入监管，因此合法两融的灵活性与场外渠道约束可以同时存在。{link('draft_20150612','草案说明')}、{link('final_20150701','正式发布说明')}、{link('offshore_20150612','同日发布会')}。

已取得的2015年上交所细则原附件，第十八条允许每次不超过6个月的展期，第四十三条仍保留130%最低比例，允许协商追保期限和追加后的比例，并扩大补充担保方式。2019年8月才公布取消130%的统一最低限制，转由券商和客户自主约定。两次制度变化应分开记录。{link('sse_rule_doc_20150701','2015年原附件')}、{link('margin_rule_20190809','2019年规则说明')}。

2018年6月深交所的说明针对经营正常但暂时资金困难的融资人，要求继续提供必要展期支持，并披露此前一周深市质押实际日均违约处置约3000万元。10月19日银行业监管要求质权人综合评估后稳妥处置，并允许保险专项产品参与纾困；10月21日交易所部署完善处置及融资工具。它们没有宣布全市场禁止平仓，也不等于资金已经到达股票买方账户。{link('szse_20180626','6月质押说明')}、{link('cbirc_20181019','10月19日答记者问')}、{link('sse_20181021','10月21日工作部署')}。

2018年纾困的后续正式规则在2019年1月18日公布：符合条件的违约合约经双方同意，可延长至累计超过3年；新增融资全部用于偿还旧违约合约时，适用部分集中度和质押率等规则豁免。借新还旧首先改变债务期限和债权关系，不能将贷款金额直接计作股票净购买。这个节点按实际公布日期纳入，没有回填到2018年。{link('pledge_rule_20190118','正式通知')}。

**公开后的剩余收益存在明显差异。** 统一在来源日期结束后的下一交易日开盘进入，5个或20个完整交易日后开盘退出。10月19日至21日信息合并，入场为10月22日。参考预算10万元，单边佣金万分之四、最低5元，单边滑点千分之一，按0.001元报价不利取整，100份整手。本轮这些窗口没有登记分红权益。

| 公布信息 | 参考入场 | 5日费用后 | 20日费用后 | 20日持有途中最低收盘回报 |
| --- | --- | ---: | ---: | ---: |
{returns_table}

最低收盘回报以入场开盘为起点，不是完整账户最大回撤，也不包括盘中更低价格。价格曲线按每天收盘显示，交易收益按下一开盘退出，所以二者端点可能不同。参考交易按开盘预算换算份额，未模拟预提交订单、开盘盘口和完整风险预算，不是可直接执行的账户方案。

![公开信息后的价格与融资余额路径](展期信息后的价格与融资路径.png)

**融资变化进一步否定了几种简单解释。** 下表在每个参考入场日前后各取5个完整交易日，金额为日均亿元。偿还额按融资买入减余额变化反推，包含现金还款、卖券还款、强平及其他统计调整，不能命名为实际卖股金额。后五日是事后观察，不能作为同次入场信息。它也不覆盖股票质押和全部场外杠杆。

| 信息节点 | 日均买入 前 → 后 | 日均反推偿还 前 → 后 | 日均净变化 前 → 后 |
| --- | ---: | ---: | ---: |
{financing_table}

2015年6月草案后，首五日融资余额仍净增加{draft['financing_after']['balance_change']/1e8:.2f}亿元，而五日期限参考交易亏损{abs(draft['holding_5']['net_return']):.2%}。余额仍增长与价格下跌可以同时发生，不能等到融资余额收缩后才认定所有压力开始。

2015年7月正式实施后，日均买入较此前五日减少{abs(final['buying_change_contribution'])/1e8:.2f}亿元，偿还增加{abs(final['repayment_change_contribution'])/1e8:.2f}亿元，五日融资余额合计减少{abs(final['financing_after']['balance_change'])/1e8:.2f}亿元。正式放宽处置并未阻止这一窗口的融资收缩与亏损，不能据此计算政策的独立因果效果。

2018年10月的反例更接近上一轮2024年的表面特征：日均融资买入增加{october['buying_change_contribution']/1e8:.2f}亿元，偿还也增加{abs(october['repayment_change_contribution'])/1e8:.2f}亿元，净流出明显收窄，但五日参考收益仍为{october['holding_5']['net_return']:+.2%}。入场前，从10月18日收盘到10月22日开盘，价格已经变化{october['already_reflected_from_pre_information_close']:+.2%}；这段不是本次可交易收益，也不能全部归因于其中某项公告。由此不能把“买入恢复且净流出收窄”单独升级成入场条件。

2019年1月正式展期节点的二十日收益为{january['holding_20']['net_return']:+.2%}，同时首五日融资余额仍减少{abs(january['financing_after']['balance_change'])/1e8:.2f}亿元。持有期间存在其他市场信息，本轮没有把全部上涨归于展期通知，也没有据此改用二十日作为胜出的期限。

**再向上追查质押风险来源，约束不止平仓线。** 2019年1月公布的2018年度报告解释了另一条路径：高比例质押降低股东追加担保能力；减持限制降低债权人在违约后卖出股票变现的能力；如果贷款时给出的质押率未充分反映流动性下降，股价下跌更容易暴露信用缺口。因此，实际卖出少既可能包含风险得到协调，也可能包含资产无法及时处置，不能仅凭卖出少认定偿付风险消失。

报告中的年末低于保障比例的质押市值2990亿元、全年特定控股股东违约申报482亿元、二级市场处置114亿元，统计对象与时间口径不同，不能相减得到被拯救的金额。沪深300中高比例质押的26家仅为成员数量，不是8.7%的指数权重。该报告只用于事后解释，绝不成为2018年信号。{link('pledge_review_20190118','2018年度质押风险分析')}。

沪深融资数据复用既有本地数据：沪市官方，深市历史批量源为金十并有既有官方抽样及缺口修复，未声称全序列逐日官方核对。公告依据目前官网保存的历史原文重建，未认证最早网页版本。本轮按固定日期保留309条日线，完成日期覆盖、余额恒等式、源文条款与往返费用核算；没有扩展参数搜索或重新拟合旧失败。

本轮没有证明某项政策的因果收益，也未量化公告相对当时共识的精确预期差。实用结论是：公告应当说明谁的哪项约束被改变；随后还要区分真实承接、价格已经发生的反应和从可成交价格起的收益。单独使用公告或融资改善，当前这些案例不足以支持直接进入完整账户。

下一项历史问题聚焦2018年10月与2024年2月：同样出现买入恢复和净流出收窄，ETF净值与溢价、份额及权重行业承接是否不同，完成观察后还剩多少收益。先检查已有研究再补缺口，不按后来涨跌选择解释，不将事后观察前移。完整账户费用后夏普1.2仍未达到。
"""
    (OUT / "历史发现_展期缓释与剩余收益.md").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="历史担保约束案例研究")
    parser.add_argument("action", choices=["prepare", "fetch", "analyze"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "fetch":
        fetch()
    else:
        analyze()


if __name__ == "__main__":
    main()
