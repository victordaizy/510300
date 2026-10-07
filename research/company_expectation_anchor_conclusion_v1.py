"""将已取得的公司原件及预测快照整理为可用于后续比较的经营基准。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_company_expectation_anchor_v1"
PRIOR = ROOT / "reports/research/510300_weight_company_driver_bridge_20260929"
STAMP = datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate_table(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"], recursive=False)]
        if len(cells) == 6 and re.fullmatch(r"20\d\d", cells[0]):
            rows.append({
                "year": int(cells[0]), "institution_count": int(cells[1]),
                "minimum": float(cells[2]), "mean": float(cells[3]),
                "maximum": float(cells[4]), "industry_mean": float(cells[5]),
            })
    if [row["year"] for row in rows] != [2026, 2027, 2028]:
        raise ValueError("预测汇总表年份与已观察页面不一致，请查看原HTML。")
    return rows


def displayed_details(table):
    rows = []
    for tr in table.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"], recursive=False)]
        if len(cells) == 9 and re.fullmatch(r"20\d\d-\d\d-\d\d", cells[-1]):
            rows.append({
                "institution": cells[0], "analyst": cells[1],
                "eps_cny": dict(zip(["2026", "2027", "2028"], cells[2:5])),
                "net_profit_as_displayed": dict(zip(["2026", "2027", "2028"], cells[5:8])),
                "report_date": cells[8],
            })
    if not rows:
        raise ValueError("没有识别到页面机构明细，不能用空表冒充已保存基准。")
    return rows


def main():
    if (OUT / "result.json").exists():
        raise SystemExit("本轮结论已保存；后续信息另存新轮次，不覆盖当前预期基准。")
    old = read_json(PRIOR / "reviewed_facts.json")
    companies = {row["symbol"]: row for row in old["companies"]}
    snapshots = []
    for code in ["300308", "300502", "688256"]:
        receipt = read_json(OUT / f"receipts/ths_{code}.json")
        soup = BeautifulSoup((OUT / f"sources/ths_{code}.html").read_bytes(), "html.parser")
        text = soup.get_text(" ", strip=True)
        date_match = re.search(r"截至\s*(\d{4}-\d{2}-\d{2})", text)
        if date_match is None:
            raise ValueError(f"{code}未找到预测页面标示日期。")
        tables = soup.find_all("table")
        profit = aggregate_table(tables[1])
        eps = aggregate_table(tables[0])
        details = displayed_details(tables[2])
        actual = companies[code]["parent_profit"] / 100_000_000
        annual = next(row for row in profit if row["year"] == 2026)["mean"]
        snapshots.append({
            "symbol": code, "name": companies[code]["name"],
            "page_as_of": date_match.group(1), "retrieved_at": receipt["at"],
            "url": receipt["url"], "raw_sha256": receipt["sha256"],
            "source_kind": "CURRENT_VENDOR_FORECAST_PROXY_NOT_FULL_ORIGINAL_BROKER_CONSENSUS",
            "lookback_as_described_by_vendor": "六个月以内",
            "net_profit_unit": "亿元", "net_profit_aggregates": profit,
            "eps_unit": "元每股，股本口径未统一，不用于估值或分歧信号",
            "eps_aggregates_for_source_record_only": eps,
            "visible_detail_rows": details,
            "visible_detail_rows_are_complete_aggregate_membership": False,
            "h1_parent_profit_cny_100million": actual,
            "conditional_h2_profit_cny_100million": annual - actual,
            "conditional_h2_vs_h1_pct": ((annual - actual) / actual - 1) * 100,
            "calculation_assumption": "仅在供应商净利润字段与公司归母净利润可比时成立；原券商报告尚未逐一核实。",
            "calculation_is_forecast_or_price_target": False,
            "usable_as_historical_pre_h1_consensus": False,
        })
    snapshot = {
        "recorded_at": STAMP,
        "baseline_selection": "采用本轮各公司首次成功下载的直接HTML；其他展示版本作为差异记录，不择取更有利数值。",
        "companies": snapshots,
        "interpretation": "未来以同机构、同预测年度、同指标口径比较修正；均值的覆盖、版本或股本变化不直接解释成经营预期变化。",
    }
    write_json(OUT / "current_vendor_forecast_baseline.json", snapshot)

    eoptolink = companies["300502"]
    preview_actual = eoptolink["parent_profit"] / 100_000_000
    eop_rows = next(row for row in snapshots if row["symbol"] == "300502")["visible_detail_rows"]
    southwest = next(row for row in eop_rows if row["institution"] == "西南证券")
    sinolink = next(row for row in eop_rows if row["institution"] == "国金证券")
    sw_eps = float(southwest["eps_cny"]["2026"])
    sl_eps = float(sinolink["eps_cny"]["2026"])
    sw_profit = float(southwest["net_profit_as_displayed"]["2026"].removesuffix("亿"))
    sl_profit = float(sinolink["net_profit_as_displayed"]["2026"].removesuffix("亿"))
    cam_revenue = companies["688256"]["revenue"] / 100_000_000
    facts = {
        "recorded_at": STAMP,
        "eoptolink_pre_h1_management_anchor": {
            "document": "sources/eoptolink_h1_preview.pdf", "pdf_page": 1,
            "signature_date": "2026-07-19", "cninfo_catalogue_date": "2026-07-20",
            "precise_first_public_timestamp_established": False,
            "parent_profit_range_cny_100million": [70, 80],
            "actual_h1_parent_profit_cny_100million": preview_actual,
            "actual_within_range": 70 <= preview_actual <= 80,
            "difference_from_range_midpoint_pct": (preview_actual / 75 - 1) * 100,
            "is_market_consensus_surprise": False,
        },
        "cambricon_incentive_anchor": {
            "document": "sources/cambricon_incentive_summary.pdf", "pdf_pages": [13, 14, 15],
            "company_authored_mirror": True, "date": "2026-07-29",
            "2026_revenue_full_vesting_target_cny_100million": 135,
            "2026_revenue_trigger_cny_100million": 108,
            "h1_actual_revenue_cny_100million": cam_revenue,
            "h2_revenue_to_target_cny_100million": 135 - cam_revenue,
            "h2_revenue_to_trigger_cny_100million": 108 - cam_revenue,
            "target_reached_company_vesting_ratio": 1,
            "trigger_reached_below_target_company_vesting_ratio": 0.8,
            "scope": "首次授予第一类激励对象2026年公司层面收入考核；其他类型条款不合并简化。",
            "is_company_profit_forecast_or_commitment": False,
            "is_market_consensus": False,
        },
        "eoptolink_eps_denominator_diagnostic": {
            "vendor_rows": [southwest, sinolink],
            "displayed_eps_gap_pct": (sl_eps / sw_eps - 1) * 100,
            "displayed_profit_gap_pct": (sl_profit / sw_profit - 1) * 100,
            "southwest_implied_share_count_100million": sw_profit / sw_eps,
            "sinolink_implied_share_count_100million": sl_profit / sl_eps,
            "implied_share_count_ratio": (sw_profit / sw_eps) / (sl_profit / sl_eps),
            "capitalization_date": "2026-06-11", "capitalization_ratio": 0.4,
            "company_h1_pdf_page": 29, "shares_after_capitalization": 1394256684,
            "inference": "隐含股数约相差1.4倍，与10转4相容；未取得两原报告，不能断言差异发生在券商计算还是供应商摘录。",
            "new_eps_disagreement_model_created": False,
        },
        "innolight_prospectus_scope": {
            "document": "sources/innolight_prospectus.pdf", "total_pdf_pages": 476,
            "extracted_pages": [1, 60], "relevant_pdf_pages": [24, 25, 56],
            "finding": "已读概要未建立与半年报可比的定量H1预告；PDF第25页是公司截至当时的订单未大幅受影响陈述，不是未来需求保证。",
            "absence_of_forecast_anywhere_proved": False,
            "basis_for_quantitative_h1_surprise": False,
        },
        "broker_comparison_limit": {
            "innolight_broker": "群益证券(香港)",
            "earlier_report_date": "2026-04-17", "later_report_date": "2026-09-23",
            "2026_profit_forecast_same_cny_100million": 334.75,
            "2027_profit_forecasts_cny_100million": [590.85, 624.12],
            "source_quality": "有作者的券商报告摘要转载，尚非完整原PDF",
            "nearest_prior_report_established": False,
            "attribute_cumulative_revision_to_september_event": False,
        },
        "vendor_version_differences": [
            {"symbol": "300308", "direct_html_2026_mean": 327.36, "other_web_display_2026_mean": 326.66},
            {"symbol": "300502", "direct_html_count": 19, "direct_html_2026_mean": 196.71,
             "other_search_display_count": 20, "other_search_display_2026_mean": 197.17},
        ],
        "version_difference_is_true_revision_or_chronological_membership_change": False,
        "required_h2_profit_conditions": [{k: row[k] for k in [
            "symbol", "name", "conditional_h2_profit_cny_100million", "conditional_h2_vs_h1_pct", "calculation_assumption"
        ]} for row in snapshots],
    }
    write_json(OUT / "reviewed_facts.json", facts)

    table_rows = []
    for row in snapshots:
        a = next(value for value in row["net_profit_aggregates"] if value["year"] == 2026)
        table_rows.append(f"| {row['name']} | {a['institution_count']} | {a['mean']:.2f} | {row['h1_parent_profit_cny_100million']:.2f} | {row['conditional_h2_profit_cny_100million']:.2f} | {row['conditional_h2_vs_h1_pct']:+.1f}% |")
    table = "\n".join(table_rows)
    report = f"""# 事前预期与后续兑现基准

记录时间：{STAMP}。沿用中际旭创、新易盛、寒武纪三个观察对象。研究执行标的仍为510300与人民币现金。

本轮将“业绩增长”推进到“哪些增长已经公开、当前参考预期要求何种后续兑现、变化由什么造成”。最明确的新发现是：新易盛半年报的高同比增长落在此前公司预告之内；一部分每股盈利分歧与股本口径不统一有关；寒武纪激励考核有目标值和触发值，不能把目标当作订单或盈利承诺。这些发现有助于减少错误买入理由，但尚未给出510300成本后的持有优势。

**事前知道什么，决定后来的增长有多少新增信息。**

新易盛7月预告上半年归母净利润70—80亿元，实际75.29亿元，在区间内，只比75亿元中点高约{facts['eoptolink_pre_h1_management_anchor']['difference_from_range_midpoint_pct']:.2f}%。这仅是相对管理层区间中点的偏差，不是市场一致预期惊喜。利润同比增长90.98%不能单独证明8月公布时出现新的盈利利好；产品组合、季度拆分、毛利率或后续指引仍可能带来新增信息，不能据此断言市场必然完全预期到。

来源：[新易盛业绩预告，第1页](https://static.cninfo.com.cn/finalpage/2026-07-20/1225431444.pdf)、[半年报，第7页](https://static.cninfo.com.cn/finalpage/2026-08-25/1225499406.PDF)。预告PDF签署7月19日，巨潮目录日期7月20日；本轮未将目录日期冒充精确的首次公开时刻。

寒武纪7月激励摘要中，首次授予第一类对象2026年收入考核目标135亿元、触发108亿元；达到触发但未达目标，公司层面归属比例80%。H1收入约59.96亿元，H2分别还需约75.04亿元、48.04亿元才达到这两个收入节点。这是激励机制，不是公司预测。摘要还明确考核不构成对投资者的业绩预测或实质承诺。不同类别的归属期及利润条件另有规定，不能只拿一个135亿元讲成全公司的硬承诺。

来源：[公司激励摘要镜像，第13—15页](https://file.finance.sina.com.cn/211.154.219.97:9494/MRGG/CNSESH_STOCK/2026/2026-7/2026-07-29/12466849.PDF)、[半年报，第7页](https://static.cninfo.com.cn/finalpage/2026-08-08/1225464969.PDF)。巨潮目录亦定位到该摘要；没有为核对重复下载同一文件。

中际旭创7月港交所招股书已保存，读取前60页。概要中的近期发展及无重大不利变动陈述，没有建立可与H1实际归母利润直接相减的定量预告。该结果不证明全篇或其他公告不存在预告，也不把公司当时认为订单未大幅受影响扩大为未来供应、监管或客户行为保证。

来源：[中际旭创港交所招股书，PDF第24—25、56页](https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0722/2026072200014_c.pdf)。

**把未来增长放到已经存在的参考预期上。**

以下均为9月29日本次直接下载的同花顺预测摘要；不是半年报公布前的历史共识，也不是完整原券商报告逐一核实后的全市场共识。每家页面详细列表只提取到10条报告，不能用这10条复算或替代汇总机构全集。

| 公司 | 供应商汇总机构数 | 全年净利润预测均值* | H1实际归母净利润 | H2所需利润* | H2相对H1* |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

金额单位：亿元。*条件计算为“供应商全年净利润预测均值−公司H1归母净利润”，仅在两者口径可比时成立；尚未逐份核实预测报告。数值用于提出经营问题，不是本研究预测、承诺、买卖阈值或估值结论。H2高于H1本身也不证明难以兑现，仍要看季节性、订单及供给条件。汇总可能含六个月内较旧预测。

来源：[中际旭创预测页](https://basic.10jqka.com.cn/300308/worth.html)、[新易盛预测页](https://basic.10jqka.com.cn/300502/worth.html)、[寒武纪预测页](https://basic.10jqka.com.cn/688256/worth.html)。H1实际值来自已保存的三份半年报，数字和口径见[上轮经营事实](../510300_weight_company_driver_bridge_20260929/reviewed_facts.json)。

这份代理已经要求相当明显的后续增长。真正值得研究的是，哪些新证据会改变该盈利路径、实现概率或现金回收速度。新易盛“1.6T下半年加速”和“2027年NPO批量交付”已在9月交流中披露，后续重述同样内容不自动构成新信息。若交付提前、客户采用规模或良率超出原先依据，才可能改变预测。远期盈利消息也可能今天改变价格；区分信息到来时间与利润兑现时间，不能因利润在2027年才出现，就排除其对眼前价格的影响。

来源：[新易盛9月投资者关系记录](https://static.cninfo.com.cn/finalpage/2026-09-12/1225562013.PDF)。

**预测因子变化，也要追查变化本身的原因。**

供应商展示的两条新易盛2026年预测，西南EPS14.47元、净利润201.73亿元，国金EPS22.97元、净利润228.29亿元。EPS差约{facts['eoptolink_eps_denominator_diagnostic']['displayed_eps_gap_pct']:.1f}%，利润差约{facts['eoptolink_eps_denominator_diagnostic']['displayed_profit_gap_pct']:.1f}%；两者隐含股数约13.94亿与9.94亿，相差约1.4倍。公司6月11日实施10转4。这与分母口径差异相容；由于未取得两份原报告，仍不能断言问题发生在券商测算还是供应商摘录。故不能把全部EPS差异解释成需求分歧，也不据此恢复旧EPS分歧模型。转增的每股口径变化与新增融资的经济影响需要另行区分。

股本依据：[新易盛半年报，第29页](https://static.cninfo.com.cn/finalpage/2026-08-25/1225499406.PDF)。预测数据见上述供应商原页及保存的HTML。

供应商另外的网页展示与直接HTML存在均值、覆盖数差异：新易盛19家/196.71亿元与20家/197.17亿元，中际旭创327.36亿元与326.66亿元。不能把两个抓取版本差异当成真实时间序列中的上修，也不能断言一定是机构新增。后续盈利修正优先比较同一家机构、同一年、同一口径；机构进出与数据更新另列。

群益中际旭创4月17日与9月23日的作者报告摘要，2026年预测同为334.75亿元，2027年由590.85至624.12亿元。两日期之间最近前一版尚未确认，不能将数月累计的远期修正全部归给9月23日的技术消息。该例说明应追查改动的是哪一年、销量还是利润率，以及真正的改动时点。

摘要来源：[4月17日](https://m.hibor.com.cn/wap_detail.aspx?id=889fbfedf4f235fe37d97fc3a6bd0240)、[9月23日](https://stock.finance.sina.com.cn/stock/go.php/vReport_Show/kind/8/rptid/843485463759/index.phtml)。摘要有作者，但不是本轮取得的完整原PDF。

**下一阶段只追最能改变经营推演的事实。**

| 原因链 | 哪种新事实支持更强的盈利路径 | 哪种事实会推翻它 | 因子的作用 |
| --- | --- | --- | --- |
| 光模块需求转为可交付订单 | 客户实物部署、明确交付节奏和可用产能同步改善，且高于已公开节奏 | 部署推迟、取消订单，或只是支出单价/租赁分类变化 | 实物交付、订单时间和产能约束比资本开支总金额更接近收入 |
| 交付转为单位利润 | 高速产品占比、良率或单位成本改善，足以覆盖降价及汇率影响 | 量增被降价、良率问题、费用抵消 | 产品结构与单位利润共同解释盈利，不只看收入同比 |
| 备货转为现金 | 原材料与在制品转成交付，回款跟上，未依赖异常延长信用 | 成品积压、验收延误、应收增长快于可解释的交付 | 存货和现金流随阶段解释；扩产期现金减少不自动看空 |
| 寒武纪供给转为客户可用算力 | 芯片供给、软件适配、验收和采购回款形成连续链条 | 供货或适配瓶颈使预付/库存难以转为验收收入 | 国内计算供给约束单独判断，不能套用北美光模块订单 |

这些是待区分的机制假设，不是假定已经发生的新订单。对下一条信息只记录“变了什么、为什么变、改变哪段盈利路径、价格是否已反映”；没有另增因子库或长期网格回测。可取得的季度收入与利润不足以唯一识别销量、价格和良率，缺失拆分时保留多种解释。

三家公司在8月末指数权重合计7.08%。即使企业判断成立，传到510300还要考虑其余成分、贴现率和风险补偿；不将三家公司景气当成整个指数方向。若三家都涨10%、其他不动且权重固定，约贡献0.708个百分点，这只是说明量级的算例，未含ETF跟踪与交易成本。每次新信息还需看价格已经发生的反应，超出供应商参考值也不自动等于仍有剩余买入空间。

本轮新增3份PDF（2份官方原站、1份公司文件镜像）、3份预测HTML；必要核对集中于预告单位、激励条款、股本变化和算术。目录/报告接口的失败请求保留，不因失败无限补源。没有新增收益测试或完整账户；净夏普、净年化未计算。旧账户800个有效/952个已运行及冻结失败不变，整体夏普1.2目标仍未实现，继续研究。既有三项宏观预测不改写，待真实信息发布后再核对。

可复用基准：[current_vendor_forecast_baseline.json](current_vendor_forecast_baseline.json)；原因与算术：[reviewed_facts.json](reviewed_facts.json)；本轮状态：[result.json](result.json)。
"""
    (OUT / "事前预期与后续兑现基准.md").write_text(report, encoding="utf-8")
    result = {
        "study_id": "510300_COMPANY_EXPECTATION_ANCHOR_V1", "recorded_at": STAMP,
        "previous_goal_turn_classification": "PROGRESS_REAL_DRIVERS_COMPANY_TRANSMISSION_AND_CURRENT_PRICE_ANCHORS",
        "continuation_classification": "PROGRESS_PRIOR_EXPECTATION_AND_CONDITIONAL_EARNINGS_HURDLES",
        "status": "MANAGEMENT_ANCHOR_AND_CURRENT_VENDOR_PROXY_RECORDED_NO_TRADABLE_INDEX_EDGE",
        "concrete_progress": ["新易盛事前预告与实际可比", "寒武纪激励目标与触发门槛分清", "三家公司当前预测代理及H2条件数落盘", "EPS分歧追到股本或摘录口径", "短期价格与远期利润的时间区分"],
        "new_pdf_documents": 3, "official_host_pdfs": 2, "company_pdf_mirrors": 1,
        "new_vendor_html_documents": 3, "new_accounts": 0, "new_strategy_return_tests": 0,
        "strategy_rule_created": False, "net_sharpe": None, "net_cagr": None,
        "performance_status": "NOT_COMPUTED", "historical_market_consensus_obtained": False,
        "current_vendor_forecast_proxy_obtained": True, "nearest_prior_broker_versions_obtained": False,
        "current_index_return_forecast_made": False, "existing_macro_forecast_cards_unchanged": True,
        "goal_achieved": False, "goal_status": "active", "orders_authorized": False,
        "not_a_blocked_turn": True, "report": "事前预期与后续兑现基准.md",
        "next_question": "下一条真实交付、单位利润或验收信息是否改变既有盈利路径；再判断主要权重净影响及已发生的价格反应。",
    }
    write_json(OUT / "result.json", result)
    protocol_path = ROOT / "config/510300_driver_expectation_research_v1.json"
    protocol = read_json(protocol_path)
    protocol.update({
        "latest_concrete_diagnostic": "reports/research/510300_company_expectation_anchor_v1/result.json",
        "latest_company_expectation_anchor": "reports/research/510300_company_expectation_anchor_v1/current_vendor_forecast_baseline.json",
        "current_price_expectation_bridge": "CURRENT_VENDOR_PROXY_RECORDED_NOT_HISTORICAL_CONSENSUS_OR_IMPLIED_PRICE",
        "forecast_revision_rule": "同机构、同预测年度、同指标口径比较；覆盖、股本和版本变化与真实经营修正分开。",
        "long_horizon_news_rule": "信息发生时间与盈利兑现时间分开；远期利润变化可影响当前价格，不计入当期已实现利润。",
        "updated_at": STAMP,
    })
    write_json(protocol_path, protocol)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read_json(mandate_path)
    mandate.update({
        "current_round": result["study_id"],
        "latest_progress_receipt": "reports/research/510300_company_expectation_anchor_v1/result.json",
        "latest_continuation_report": "reports/research/510300_company_expectation_anchor_v1/事前预期与后续兑现基准.md",
        "latest_continuation_classification": result["continuation_classification"],
        "last_research_result": "取得新易盛事前预告、寒武纪激励原条款及三家公司当前预测代理；分清EPS股本口径，计算H2条件盈利基准。未识别510300可交易优势，新增账户0、收益测试0，目标未实现。",
        "last_source_result": "新增3份PDF及3份供应商预测HTML；当前代理不是历史市场共识，原券商最近前版仍不完整。",
        "latest_driver_diagnostic_at": STAMP,
        "latest_company_expectation_anchor": "reports/research/510300_company_expectation_anchor_v1/result.json",
        "next_research_question": result["next_question"],
    })
    write_json(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    prior_status = status_path.read_text(encoding="utf-8")
    heading = """<!-- COMPANY_EXPECTATION_ANCHOR_V1_20260929 -->

## 2026-09-29 事前预告、当期预测代理与后续经营基准

新易盛H1实际75.29亿元落在事前70至80亿元预告内，较中点约高0.39%；寒武纪135亿元激励收入目标与108亿元触发值分开。保存三家公司当期供应商盈利预测，条件计算H2分别需约191、121、32亿元净利润；未冒充历史共识。部分EPS分歧与股本分母不统一相容，原报告未全取；当前版本差异不当真实上修。新增3份PDF、3份预测HTML，无新收益测试或账户，旧800/952账户及失败不变。夏普1.2目标仍未实现，目标active，本轮PROGRESS。

详见[事前预期与后续兑现基准](<C:/Users/戴周阳/Documents/New project 8/reports/research/510300_company_expectation_anchor_v1/事前预期与后续兑现基准.md>)。接下来以真实交付、单位利润和验收更新推演盈利路径；已有宏观预测保持原样。

"""
    status_path.write_text(heading + prior_status, encoding="utf-8")
    print(json.dumps({"状态": result["status"], "报告": str(OUT / result["report"]), "新易盛相对预告中点百分比": facts["eoptolink_pre_h1_management_anchor"]["difference_from_range_midpoint_pct"], "H2条件基准": facts["required_h2_profit_conditions"], "新增账户": 0, "目标实现": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
