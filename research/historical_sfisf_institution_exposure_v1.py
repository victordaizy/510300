"""分解历史机构持仓账面变化；不将集团总量冒充工具专属买盘。"""

from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_sfisf_institution_exposure_v1"
REPORT = OUT / "历史发现_融资买股与净多头不是一个量.md"
NEXT = (
    "结束对这四份报告中SFISF专属成本和对冲的反复推测。转向2024—2025年明确披露ETF直接增持的历史事件，"
    "先比对现有汇金及基金份额研究，追查购买发生期、份额变化与市场压力的先后；"
    "只补已有研究未覆盖的因果环节，再评价公告后仍可成交的费用后收益。"
)


def relative(path):
    return path.relative_to(ROOT).as_posix()


def link(path, label):
    return f"[{label}](<{path.as_posix()}>)"


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def d(value):
    return Decimal(str(value))


def yi(value):
    return f"{d(value) / d(100000000):,.2f}"


def source_index():
    sources = {}
    specs = [
        ("中信证券_2024年报", "600030", "2024-12-31", "2025-03-26", 165, "2025-03-27"),
        ("中信证券_2025中报", "600030", "2025-06-30", "2025-08-28", 97, "2025-08-29"),
        ("中金公司_2024年报", "601995", "2024-12-31", "2025-03-28", 188, "2025-03-29"),
        ("中金公司_2025中报", "601995", "2025-06-30", "2025-08-29", 91, "2025-08-30"),
    ]
    for key, code, period, approved, approval_page, date in specs:
        receipt = json.loads((OUT / "sources" / (key + ".receipt.json")).read_text(encoding="utf-8"))
        pages = json.loads((OUT / "sources" / (key + ".pages.json")).read_text(encoding="utf-8"))
        sources[key] = {**receipt, "report_end": period, "board_approval_date": approved,
                        "board_approval_pdf_page": approval_page, "cninfo_catalogue_date": date,
                        "catalogue_path": relative(OUT / "sources" / (code + "_报告目录.json")),
                        "earliest_market_release_time_verified": False,
                        "pages": len(pages), "page_texts": pages}
        y, m, day = map(int, approved.split("-"))
        expected = f"{y}年{m}月{day}日"
        assert expected in "".join(pages[approval_page - 1]["text"].split())
    key = "中信银行_20241025融资成交"
    sources[key] = json.loads((OUT / "sources" / (key + ".receipt.json")).read_text(encoding="utf-8"))
    sources[key].update({"stated_transaction_date": "2024-10-25", "page_label_time": "2024-10-25 10:48",
                         "url_path_date": "2024-10-29", "earliest_market_release_time_verified": False})
    sources["证监会_首批名单"] = {
        "url": "https://www.csrc.gov.cn/csrc/c100028/c7513142/content.shtml",
        "publication_date": "2024-10-18", "acquisition": "网页工具直接读取官方正文；本地HTTPS请求失败",
        "selected_first_two_in_list": ["中信证券", "中金公司"], "raw_html_saved": False,
    }
    old = ROOT / "reports/research/510300_policy_information_clock_v1"
    for key in ["sfisf_20241018_csrc", "sfisf_20241021_pboc"]:
        receipt = json.loads((old / "receipts" / (key + ".json")).read_text(encoding="utf-8"))
        sources[key] = {"url": receipt["url"], "path": relative(old / "raw" / (key + ".html")), "reused": True}
    return sources


def verify_page(sources, key, page, terms):
    text = sources[key]["page_texts"][page - 1]["text"]
    compact = "".join(text.split())
    for term in terms:
        if term not in compact:
            raise ValueError(f"原报告定位不一致：{key} PDF第{page}页 {term}")
    return {"source": key, "pdf_page": page, "terms": terms}


def holdings(sources):
    """只使用相同集团口径，精确核对原数后分解两期末存量。"""
    inputs = [
        ("中信证券", "交易性金融资产中的股票", "中信证券_2024年报", 234, "中信证券_2025中报", 132,
         "200456231981.28", "202095054008.04", "190131001203.46", "180554925561.12"),
        ("中信证券", "其他权益工具投资，未单拆股票", "中信证券_2024年报", 237, "中信证券_2025中报", 135,
         "90667793757.38", "89280953970.13", "85230083148.69", "82931764272.91"),
        ("中金公司", "交易性金融资产中的股票／股权", "中金公司_2024年报", 274, "中金公司_2025中报", 137,
         "100146562754", "99297431784", "104704802725", "99167004128"),
        ("中金公司", "其他权益工具投资中的股票", "中金公司_2024年报", 276, "中金公司_2025中报", 139,
         "5289635124", "4999986698", "6342488252", "6199969452"),
        ("中金公司", "其他权益工具投资中的基金及其他", "中金公司_2024年报", 276, "中金公司_2025中报", 139,
         "2574298466", "2705439969", "3501124886", "3312497709"),
    ]
    rows, checks = [], []
    for firm, label, prior, pp, current, cp, fv0, cost0, fv1, cost1 in inputs:
        for source, page, values in [(prior, pp, [fv0, cost0]), (current, cp, [fv1, cost1])]:
            tokens = [f"{d(v):,.2f}" if "." in v else f"{d(v):,}" for v in values]
            checks.append(verify_page(sources, source, page, tokens))
        mark0, mark1 = d(fv0) - d(cost0), d(fv1) - d(cost1)
        delta_fv, delta_cost = d(fv1) - d(fv0), d(cost1) - d(cost0)
        delta_mark = mark1 - mark0
        assert delta_fv == delta_cost + delta_mark
        rows.append({
            "institution": firm, "category": label, "scope": "集团合并口径",
            "start": "2024-12-31", "end": "2025-06-30", "unit": "CNY",
            "start_fair_value": fv0, "start_initial_cost": cost0,
            "end_fair_value": fv1, "end_initial_cost": cost1,
            "change_fair_value": str(delta_fv), "change_initial_cost": str(delta_cost),
            "start_fair_value_minus_cost": str(mark0), "end_fair_value_minus_cost": str(mark1),
            "change_fair_value_minus_cost": str(delta_mark),
            "prior_source": prior, "prior_pdf_page": pp, "current_source": current, "current_pdf_page": cp,
            "cash_net_purchase_identified": False, "sfisf_attributed": False,
            "warning": "期末仍持有资产的成本变动不是期间现金净买入；市值减成本差额变动不是纯当期估值损益。",
        })
    return rows, checks


def instrument_facts(sources):
    return [
        {"fact": "中信证券2024年股票自营加大非方向性投资，目的包括降低组合波动及抵御宏观冲击。",
         "source": "中信证券_2024年报", "pdf_page": 33, "scope": "公司股票自营业务，不是SFISF专属仓位"},
        {"fact": "中信证券2025上半年权益及另类投资业务增加非方向性对冲等策略布局。",
         "source": "中信证券_2025中报", "pdf_page": 20, "scope": "公司经营举措及业绩段，不是下半年计划"},
        {"fact": "中金2024年报自述落地市场首笔互换便利。",
         "source": "中金公司_2024年报", "pdf_page": 11, "scope": "交易发生确认，未提供该交易金额、融资价或购股明细"},
        {"fact": "中金2025中报的权益衍生工具名义本金281602973071元；同报告说明名义本金不是承担风险数额。",
         "source": "中金公司_2025中报", "pdf_pages": [128, 129],
         "scope": "集团全业务名义金额，含不同方向和标的，不可减去股票市值计算净敞口"},
        {"fact": "中信2025中报权益衍生工具名义金额559094180022.06元，列于用于非套期栏。",
         "source": "中信证券_2025中报", "pdf_page": 124,
         "scope": "会计套期分类不是经济风险对冲识别；结合公司非方向性对冲业务，不能据该栏名称称没有经济对冲"},
        {"fact": "2024年10月25日，中信银行与中信证券达成SFISF项下国债质押式回购交易。",
         "source": "中信银行_20241025融资成交", "scope": "确认后续融资成交，不确认融资利率、金额或股票购买"},
        {"fact": "首笔500亿元互换操作中标费率20bp；属于互换环节，后续融资成本不在该费率内。",
         "source": "sfisf_20241021_pboc", "scope": "公开操作费率，不是两家公司已识别的融资全成本"},
        {"fact": "2024年10月18日规则公开说明资本市场资金可用于股票和股票ETF的投资与做市。",
         "source": "sfisf_20241018_csrc", "scope": "允许用途，不是实际使用比例，也不是保证每笔均为长期多头"},
    ]


def figure(rows):
    row = rows[2]
    values = [float(d(row[k]) / d(100000000)) for k in
              ["change_fair_value", "change_initial_cost", "change_fair_value_minus_cost"]]
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 11})
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    fig.patch.set_facecolor("#f6f3ec")
    ax.set_facecolor("#fffdf8")
    labels = ["账面市值变化", "期末持仓初始成本变化", "市值减成本的差额变化"]
    colors = ["#286f74", "#a76253", "#be8843"]
    ax.barh(range(3), values, color=colors, height=.52)
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.set_xlim(-8, 58)
    ax.axvline(0, color="#888", linewidth=.8)
    ax.set_xlabel("亿元人民币")
    for i, v in enumerate(values):
        ax.text(v + (1 if v >= 0 else -.6), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle("中金公司：持仓市值增长，不能直接当成新增买盘", fontsize=16, y=.97)
    ax.set_title("交易性金融资产中的股票／股权，2024年末至2025年6月末，集团合并口径", fontsize=10, pad=12)
    fig.text(.04, .045,
             "恒等分解：+45.58 = −1.30 + 46.89（亿元；分项四舍五入）。\n"
             "成本是期末仍持有资产的初始成本；差额含价格、持仓更替等影响，不等于纯估值损益。此图不归因SFISF。",
             fontsize=9, color="#545a59")
    fig.tight_layout(rect=[.02, .15, .99, .94])
    fig.savefig(OUT / "中金持仓账面变化分解.png", dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)


def report(rows, sources):
    citic_a = sources["中信证券_2024年报"]["url"]
    citic_h = sources["中信证券_2025中报"]["url"]
    cicc_a = sources["中金公司_2024年报"]["url"]
    cicc_h = sources["中金公司_2025中报"]["url"]
    lines = [
        "# 历史发现：融资买股与净多头不是一个量", "",
        "2026年9月30日。只研究2024—2025年已发生的交易、财报和制度，不作前瞻预测。", "",
        "**机构能够融资、股票账面规模增加、机构愿意承担更多股市上涨或下跌风险，分别是不同的信息。此次找到一组直接反例：中金2025年上半年交易性股票／股权市值增加45.58亿元，期末持仓成本却减少1.30亿元；中信同期两个所列权益资产科目的账面规模均下降。同时，经营原文明确存在非方向性策略。因而不能将融资规模或持仓市值直接做成510300的多头因子。**", "",
        "对象为首批名单中前两家证券公司——中信证券和中金公司，各读2024年报、2025中报。名单排序是取样方法，不是获批优先级或投资能力排名。两家公司不是全行业代表，不据此估计全部工具参与者的比例。已有部分成交新闻线索在登记研究范围前看过，本轮是知情后的历史原因分析。", "",
        "**1．钱确实融到了，但20bp不是融资全成本。** 中信银行原文确认，2024年10月25日与中信证券完成SFISF项下国债质押式回购。首笔央行互换操作的20bp属于互换费率，后面取得现金还涉及回购融资利息；期限、金额基数及其他直接成本也需对应，不能把20bp当全部年化借款成本。银行新闻没有给出这笔回购的金额或成交利率，本轮四份报告也未定位到可以逐笔对应的专属成本。这里记为未知，不能用全公司回购利息除期末余额代替。[中信银行成交说明](" + sources["中信银行_20241025融资成交"]["url"] + ")；[央行首笔操作](" + sources["sfisf_20241021_pboc"]["url"] + ")。", "",
        "中金2024年报确认其落地首笔互换便利，但这一表述没有同时提供现金融资价、所买股票、持有期限和专属对冲仓位。[中金2024年报，PDF第11页](" + cicc_a + ")。", "",
        "**2．先拆‘持仓增加’的原因。** 下表全部使用集团合并口径，同期比较2024年12月31日与2025年6月30日，单位亿元。各行是特定会计科目，不是完整股票风险敞口；中信的其他权益工具没有在该表中单拆股票，不能冒充纯股票。", "",
        "| 机构及原始科目 | 期初市值 | 期末市值 | 市值变动 | 期末持仓成本变动 | 市值减成本差额的变动 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + row["institution"] + "：" + row["category"] + " | " + " | ".join(
            yi(row[k]) for k in ["start_fair_value", "end_fair_value", "change_fair_value", "change_initial_cost", "change_fair_value_minus_cost"]) + " |")
    lines += [
        "", "计算只使用一个恒等式：**市值变化＝期末持仓初始成本的变化＋两期‘市值减成本’差额的变化。** 没有拟合参数。中金2024年报PDF第274、276页与2025中报第137、139—140页相互对应；中信2024年报PDF第234、237页与2025中报第132—135页对应。[中金2025中报](" + cicc_h + ")；[中信2025中报](" + citic_h + ")。", "",
        "中金交易性股票／股权的45.58亿元增量，由−1.30亿元成本存量变化和＋46.89亿元市值减成本差额变化组成。这足以否定‘市值增加多少就是新买入多少’。但成本存量变化也不等于现金净买入：卖出资产按原成本退出，期间还可能有持仓更替、汇率或分类变化；差额变化不能直接认定为当期价格上涨产生的纯收益。报告未提供本轮所需的逐笔桥接，不能把剩下的未知部分编成机构交易。", "",
        "中金其他权益工具中的股票成本同期增加12.00亿元，说明不同科目方向并不一致。也不能只摘出这一行，忽略交易性股票、基金、其他风险与衍生品，便称整个集团增持了同等沪深300多头。", "",
        "**3．机构的经营动机不只有等待指数上涨。** 中信2024年报的实际经营段说明，股票自营扩大非方向性布局，意在降低波动和应对宏观冲击；2025中报实际经营段又提到非方向性对冲策略。它们提供了公司经营原因的原始证据，但没有将这类策略逐笔归属到SFISF。[2024年报，PDF第33页](" + citic_a + ")；[2025中报，PDF第20页](" + citic_h + ")。", "",
        "制度层面，工具用途包括股票及股票ETF的投资和做市，因此‘使用工具’本来也不能自动等同长期持有一笔指数多头。进一步推论是：新增融资既可能支持增加股票风险，也可能服务库存、客户交易或缓解被迫卖出；这些是需要区分的竞争解释，本轮没有算出它们各占多少。[2024年10月18日官方规则说明](" + sources["sfisf_20241018_csrc"]["url"] + ")。", "",
        "**4．对冲必须看方向和敏感度，不能减名义本金。** 中金2025年6月末集团权益衍生工具名义本金为2816.03亿元，报告明确说明名义本金并不代表风险金额。这个总数没有给出多空方向、各标的、净Delta和工具专属归属，不能从股票市值中直接减去。[中金中报，PDF第128—129页](" + cicc_h + ")。", "",
        "中信同期期末权益衍生工具名义金额5590.94亿元，列在非套期栏。会计上的套期分类不能直接替代经济对冲判断；报告业务部分已经提到非方向性对冲，不能因附注的分类名称便判定没有风险对冲。[中信中报，PDF第124—125页及第20页](" + citic_h + ")。", "",
        "真正需要的是同一范围、同一时刻的现货仓位与带方向的衍生品风险敏感度，以及它们相对沪深300的对应关系。公开总额在本轮无法完成这一转换。股票现货订单可能产生需求，但交易对手的对冲、其他账户卖出及既有持仓替代也会影响市场净压力，不能从某一家名义规模推算全市场价格影响。", "",
        "**5．信息时点和用途。** 四份报告的财务报表分别获董事会于2025年3月26日、8月28日（中信），3月28日、8月29日（中金）批准；巨潮目录日期各晚一天。批准日期不是公开时刻，目录日期也没有证明全市场最早披露。6月末的持仓不能倒填为6月末即可用的交易信号。中信银行网页标注10月25日10:48，URL含10月29日，保留两者，不由URL猜测首次发布。", "",
        "就510300研究而言，本轮得到的是因子使用规则：融资费与融资总成本分开；期末账面量先拆成本及估值差额；机构总仓位与专属新增净风险分开。融资能力可以作为当时约束变化的背景信息。只有这些背景信息还能够连接到已发生的真实需求变化、指数覆盖及消息后剩余收益，才具备进一步测试多头账户的理由。机构通过融资、做市、对冲取得的收益来源，不能直接搬到只持有510300和现金的账户。", "",
        "四份报告没有支持一个可量化的SFISF专属净多头信号。本轮据此停止把该金额直接升级为买入条件；这不表示工具没有稳市作用，也不是做空理由。未新增收益回测或账户，不重复上一轮十个政策节点的结果。原政策账户压力费用夏普约0.690、年化4.61%的失败保留，夏普1.2和年化10%的目标仍未实现。", "",
        "只核对了五行原数、科目范围、四份批准日及恒等式；四张关键表格页已查看。公司官网访问失败后使用巨潮正式披露原报告，首次失败另存，不反复请求。当前取得文件未认证为当年首版。", "",
        "下一历史问题：" + NEXT, "",
        link(OUT / "两家机构账面变化.json", "完整原数及分解") + "；" + link(OUT / "source_index.json", "原报告来源和时点") + "；" + link(OUT / "工具与对冲事实.json", "交易与对冲事实") + "；" + link(OUT / "中金持仓账面变化分解.png", "账面变化图") + "。", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def update_state(result):
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({
        "current_round": result["study_id"], "latest_progress_receipt": relative(OUT / "result.json"),
        "last_research_result": "两家首批机构四份报告完成：中金2025上半年交易性股票／股权市值增45.58亿元，期末持仓成本减1.30亿元，市值减成本差额增46.89亿元；中信实际经营包含非方向性策略。确认融资成交，但专属总成本、购股与净对冲未识别，不把集团账面量升级为SFISF多头信号。",
        "last_source_result": "新增4份公司原报告、1份银行成交原文、2份巨潮目录；复用2份工具规则和操作记录；网页核对首批名单。",
        "latest_historical_diagnostic_at": result["recorded_at"], "latest_historical_report": relative(REPORT),
        "latest_historical_sfisf_institution_exposure": relative(OUT / "result.json"),
        "latest_continuation_report": relative(REPORT), "latest_continuation_classification": result["classification"],
        "current_driver_continuation_classification": result["classification"], "current_driver_consecutive_blocked_goal_turns": 0,
        "goal_status": "active", "goal_achieved": False, "next_research_question": NEXT,
        "latest_goal_service_status": "active", "latest_goal_service_status_observed_at": "2026-09-30T14:03:13+08:00",
    })
    save(path, data)
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({"latest_completed_study": relative(OUT / "result.json"), "latest_report": relative(REPORT), "updated_at": result["recorded_at"]})
    save(path, data)
    path = ROOT / "RESEARCH_STATUS.md"
    old = path.read_text(encoding="utf-8")
    marker = "<!-- HISTORICAL_SFISF_INSTITUTION_EXPOSURE_V1_20260930 -->"
    if marker not in old:
        entry = marker + "\n\n## 2026-09-30 历史机构融资、账面量与净多头已区分\n\n"
        entry += "首批名单前两家证券公司的2024年报和2025中报完成原文定位。中金交易性股票／股权市值半年增加45.58亿元，其中期末持仓成本变动−1.30亿元、市值减成本差额变动＋46.89亿元，不能将账面增量当现金净买入。中信实际经营原文包含非方向性投资和对冲，不能把集团总额当SFISF专属方向敞口。\n\n"
        entry += "银行原文确认中信证券融资成交；20bp是互换费率，专属融资全成本仍未知。四份报告未定位专属购股及对冲明细。新增收益测试、账户、参数和预测均0，旧失败不改，目标active且未达成。继续历史研究，不等待未来披露。\n\n"
        entry += "详见" + link(REPORT, "机构融资与净多头的历史发现") + "及" + link(OUT / "两家机构账面变化.json", "原数与精确分解") + "。\n\n"
        path.write_text(entry + old, encoding="utf-8")


def main():
    sources = source_index()
    rows, checks = holdings(sources)
    checks += [
        verify_page(sources, "中信证券_2024年报", 33, ["非方向性投资", "经营举措及业绩"]),
        verify_page(sources, "中信证券_2025中报", 20, ["非方向性对冲策略", "经营举措及业绩"]),
        verify_page(sources, "中金公司_2024年报", 11, ["首笔互换便利"]),
        verify_page(sources, "中金公司_2025中报", 128, ["281,602,973,071"]),
        verify_page(sources, "中金公司_2025中报", 129, ["并不代表本集团所承担的风险数额"]),
        verify_page(sources, "中信证券_2025中报", 124, ["559,094,180,022.06", "用于非套期"]),
    ]
    bank = (OUT / "sources/中信银行_20241025融资成交.txt").read_text(encoding="utf-8")
    assert "10月25日再次与中信证券达成" in bank
    policy = BeautifulSoup((ROOT / sources["sfisf_20241018_csrc"]["path"]).read_bytes(), "html.parser").get_text(" ", strip=True)
    assert "投资和做市" in policy
    save(OUT / "两家机构账面变化.json", rows)
    save(OUT / "工具与对冲事实.json", instrument_facts(sources))
    save(OUT / "关键原文定位.json", checks)
    figure(rows)
    report(rows, sources)
    for item in sources.values():
        item.pop("page_texts", None)
    save(OUT / "source_index.json", sources)
    result = {
        "study_id": "510300_HISTORICAL_SFISF_INSTITUTION_EXPOSURE_V1",
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "HISTORICAL_FINANCING_GROSS_HOLDINGS_AND_DIRECTIONAL_EXPOSURE_DISTINGUISHED",
        "classification": "PROGRESS_HISTORICAL_INSTITUTION_MOTIVE_AND_BOOK_VALUE_DECOMPOSITION",
        "institutions": 2, "company_reports": 4, "balance_categories": len(rows),
        "cicc_trading_equity_decomposition_cny": {k: rows[2][k] for k in
            ["change_fair_value", "change_initial_cost", "change_fair_value_minus_cost"]},
        "dedicated_sfisf_financing_rate_identified": False, "dedicated_sfisf_purchase_amount_identified": False,
        "dedicated_sfisf_delta_exposure_identified": False, "missing_means_zero": False,
        "new_return_tests": 0, "new_accounts": 0, "new_fitted_parameters": 0, "new_forecast_cards": 0,
        "old_accounts_unchanged": True, "independent_validation": False, "goal_achieved": False,
        "wait_for_future_data": False, "orders_authorized": False, "report": relative(REPORT),
        "next_research_question": NEXT,
    }
    save(OUT / "result.json", result)
    update_state(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
