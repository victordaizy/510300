"""整理已保存的银行经营传导结果，并生成可分享的中文研究图。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_bank_profit_credit_bridge_v13"
REPORT = "第十三轮_信贷扩张怎样传到银行普通股盈利.md"


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_link(path, label):
    return f"[{label}](<{path.as_posix()}>)"


def publish():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮完成记录已经保存，不覆盖。")
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "PASS_PRIMARY_STATEMENT_AND_PROFIT_IDENTITIES"
    facts = pd.read_csv(OUT / "results/原表金额与期间来源.csv")
    bridge = pd.read_csv(OUT / "results/平安银行_净利润增量金额桥.csv")
    detail = json.loads((OUT / "results/传导分解摘要.json").read_text(encoding="utf-8"))
    companies = pd.read_csv(OUT / "results/三家银行_当时已知年度业绩.csv")
    q4 = pd.read_csv(OUT / "results/快报减三季报_隐含第四季度.csv")
    cohort = pd.read_csv(OUT / "results/按参考权重固定三家公司.csv")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha256((OUT / "protocol.json").read_bytes()).hexdigest() == frozen["protocol_sha256"]
    assert sha256((OUT / "inputs/parent_stock_context.parquet").read_bytes()).hexdigest() == frozen["input_sha256"]
    np.testing.assert_allclose(bridge.profit_change_yi.sum(), detail["pab"]["net_profit_change_yi"])
    def pair(key):
        r = facts[facts.company.eq("平安银行") & facts.key.eq(key)].iloc[0]
        return np.array([r.previous, r.current], dtype=float)
    profit, ordinary, eps = pair("parent_profit") / 100, pair("ordinary_profit") / 100, pair("eps")
    np.testing.assert_allclose(pair("parent_profit") - pair("preferred_dividend") - pair("perpetual_interest"), pair("ordinary_profit"))
    np.testing.assert_allclose(np.round(pair("ordinary_profit") / pair("weighted_shares"), 2), eps)
    loan_growth = 100 * (pair("loan_avg")[1] / pair("loan_avg")[0] - 1)
    figure_path = OUT / "figures/信贷扩张到每股盈利_三个抵消环节.png"
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.8, 9.6), facecolor="#f8f9fb")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1], left=.19, right=.97, top=.80, bottom=.19, hspace=.53, wspace=.30)
    ax = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 1])
    fig.text(.045, .945, "信贷扩张到每股盈利：经营增长会经过多重抵消", fontsize=21, weight="bold", color="#17384d")
    fig.text(.045, .900, f"平安银行 2020 年：贷款日均余额 +{loan_growth:.1f}%  →  净利息收入 +{detail['pab']['nii_growth_percent']:.1f}%  →  净利润 +2.6%", fontsize=13, color="#38576c")
    fig.text(.045, .858, "年报于 2021 年 2 月 2 日公开；以下均为 2020 年与 2019 年对比。", fontsize=10.5, color="#586a79")
    labels = bridge.item.tolist() + ["最终净利润增量"]
    vals = bridge.profit_change_yi.tolist() + [detail["pab"]["net_profit_change_yi"]]
    colors = ["#237d82" if x >= 0 else "#b26750" for x in vals[:-1]] + ["#193c52"]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=.56)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#9caebb", lw=.9)
    ax.set_xlim(-139, 132)
    ax.set_xlabel("对净利润同比增量的贡献，亿元")
    ax.set_title("营业收入多了 155.84 亿元，净利润多了 7.33 亿元", fontsize=12, loc="left", pad=19, weight="bold")
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    for yy, value in zip(y, vals):
        ax.text(value + (3 if value >= 0 else -3), yy, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right", fontsize=10.5, color="#233e4c")
    xx = np.arange(2)
    width = .32
    for k, year in enumerate(["2019 年", "2020 年"]):
        bars = b.bar(xx + (k-.5)*width, [profit[k], ordinary[k]], width, label=year, color=["#b6c2ce", "#237d82"][k])
        for bar in bars:
            b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+6, f"{bar.get_height():.2f}", ha="center", fontsize=9.5)
    b.set_xticks(xx, ["归母净利润", "普通股股东利润"])
    b.set_ylim(0, 352)
    b.set_ylabel("亿元")
    b.set_title("总利润增加，普通股股东利润略降", loc="left", fontsize=11.5, pad=17, weight="bold")
    b.legend(frameon=False, ncol=2, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.19))
    b.grid(axis="y", alpha=.16)
    b.set_axisbelow(True)
    bars = c.bar([0, 1], eps, color=["#b6c2ce", "#237d82"], width=.52)
    for bar in bars:
        c.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.06, f"{bar.get_height():.2f}", ha="center", fontsize=12)
    c.set_xticks([0, 1], ["2019 年", "2020 年"])
    c.set_ylim(0, 1.95)
    c.set_ylabel("基本每股收益，元")
    c.set_title("还要除以加权平均普通股股数", loc="left", fontsize=11.5, pad=17, weight="bold")
    c.grid(axis="y", alpha=.16)
    c.set_axisbelow(True)
    fig.text(.045, .130, "股东分配：归母净利润 +7.33 亿元，永续债利息分配增加 8.20 亿元，普通股股东利润减少 0.87 亿元。", fontsize=10.5, color="#38576c")
    fig.text(.045, .098, f"每股分母：加权平均普通股股数增加 {detail['pab']['share_growth_percent']:.2f}%；这与普通股股东利润变化共同影响每股收益。", fontsize=10.5, color="#38576c")
    fig.text(.045, .063, "来源：平安银行 2020 年报，PDF 第 30、148、240 页。贷款日均余额不含贴现；金额按原表复算。", fontsize=9, color="#6b7c88")
    fig.text(.045, .035, "范围：单家公司经营传导案例，不代表全部银行、510300 每股盈利或未来收益；两年比率不等于因果估计。", fontsize=9, color="#6b7c88")
    for a in [ax, b, c]:
        a.set_facecolor("#f8f9fb")
    fig.savefig(figure_path, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(figure_path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    source_links = {
        "cmb": "[招商银行当时的业绩快报](https://static.cninfo.com.cn/finalpage/2021-01-15/1209106000.PDF)",
        "cib": "[兴业银行当时的业绩快报](https://static.cninfo.com.cn/finalpage/2021-01-15/1209105758.PDF)",
        "pab": "[平安银行2020年报](https://static.cninfo.com.cn/finalpage/2021-02-02/1209224370.PDF)",
        "cmb_q3": "[招商银行2020年三季报](https://static.cninfo.com.cn/finalpage/2020-10-31/1208662780.PDF)",
        "cib_q3": "[兴业银行2020年三季报](https://static.cninfo.com.cn/finalpage/2020-10-30/1208651184.PDF)"}
    company_table = "\n".join(f"| {r.company} | {r.revenue_growth_percent:+.2f}% | {r.profit_growth_percent:+.2f}% | {r.eps_previous:.2f} → {r.eps_current:.2f} |" for r in companies.itertuples())
    bridge_table = "\n".join(f"| {r.item} | {r.profit_change_yi:+.2f} |" for r in bridge.itertuples())
    q4_table = "\n".join(f"| {r.company} | {r.implied_2019_q4_yi:.2f} | {r.implied_2020_q4_yi:.2f} | {r.growth_percent:+.2f}% |" for r in q4.itertuples())
    symbols = {"600036.SH": "招商银行", "601166.SH": "兴业银行", "000001.SZ": "平安银行"}
    price_table = "\n".join(f"| {symbols[r.symbol]} | {r.weight*100:.3f}% | {r.past20_total_return*100:+.2f}% |" for r in cohort.itertuples())
    link_v10 = file_link(ROOT / "reports/research/510300_constituent_risk_transmission_v10/第十轮_指数降波与成分风险的区别.md", "成分风险与权重口径")
    link_v11 = file_link(ROOT / "reports/research/510300_holiday_money_credit_bridge_v11/第十一轮_春节存款持有人与信贷结构.md", "货币余额、春节与信贷结构")
    link_v12 = file_link(ROOT / "reports/research/510300_funding_quantity_price_bridge_v12/第十二轮_货币数量与融资价格为何可能背离.md", "融资价格与货币数量")
    report = f"""**新确认的结论：信贷扩张可以传成银行收入增长，同时被息差、减值和普通股分配稀释。把信贷增量直接当成510300盈利增量，会跳过决定结果的中间环节。** 2021年2月9日能够看到的三家银行资料已出现这种分化；这不是后来年报补出的解释。

本轮沿用2021年2月9日21:00观察点，在阅读三家年度原公告前，按上一轮银行成分参考权重固定招商、兴业、平安银行。三家合计占参考篮子 **{frozen['cohort_reference_weight']*100:.3f}%**，全部25家银行占 **{frozen['bank_reference_weight']*100:.3f}%**；三家约覆盖银行参考权重的 **{frozen['cohort_reference_weight']/frozen['bank_reference_weight']*100:.2f}%**。这些是保存的历史参考权重，不是ETF实际持仓，也没有历史首版认证。结果不能推广为全部银行或整个510300的确定方向。

**当时可以知道什么，先于解释结果。** 招商和兴业只使用2021年1月15日前后公布的2020年度业绩快报与此前三季报；平安银行使用2021年2月2日已公布的完整年报。为避免冒称日内精确时点，文件纳入时间保守取相应公布日23:59:59。三家资料都早于2月9日的1月金融数据公告。招商、兴业快报是未经审计的初步核算，未披露的全年息差和详细减值保留未知，不拿3月以后年报回填。

| 银行 | 2020年营业收入同比 | 归母净利润同比 | 基本每股收益，2019→2020，元 |
|---|---:|---:|---:|
{company_table}

来源：{source_links['cmb']} PDF第1页、{source_links['cib']} PDF第1页、{source_links['pab']} PDF第148和240页。这里三个层次不能混写：营业收入、归母净利润、普通股每股收益。兴业和平安已实际出现“总利润增长、普通股每股收益下降”，招商则仍有每股收益增长。

**平安银行提供了当时已经公开、可以用金额核对的完整路径。** 不含贴现的贷款日均余额增长 **{loan_growth:.1f}%**，贷款平均收益率从6.58%降到6.07%，下行51个基点。同期存款成本率也由2.46%降到2.23%，但没有抵消全部资产收益率变化：净息差由2.62%降到2.53%，下行9个基点，净利息收入仍因规模等变化增加 **96.89亿元、10.8%**。贷款平均收益率还含客户和业务结构变化，不等于同一借款人利率统一下降51个基点。来源：年报PDF第30—31页。

按原表金额与日均余额作对称恒等分解，净利息收入增加96.89亿元，可分成生息资产规模项+253.20亿元、资产综合收益率项−156.82亿元、计息负债规模项−112.97亿元、负债综合成本率项+113.48亿元。综合比率用原表利息金额除以日均余额构造，避免用已四舍五入利率造成差额；它包括结构变化，仅保证会计金额一致，不是四项经济因果效应。

时间方向也不能省略。2020年第四季度相对第三季度，平安存款成本由2.13%降至2.10%，但同业业务及其他负债成本从2.01%升至2.21%、发行债务证券成本从2.74%升至2.90%；综合计息负债成本由2.19%升至2.25%。净息差由2.48%降至2.44%，季度净利息收入由248.49亿元降至244.96亿元。**存款变便宜、批发融资变贵、整体息差收窄可以同时发生。** 这与上一轮“两个融资利率之差收窄，不等于绝对资金价格下降”的口径区别相接，但不是同一日期的因果回归。

**从收入到净利润，抵消发生在哪里。** 2020年平安银行营业收入同比增加155.84亿元，最终净利润只增加7.33亿元。下表以同一合并利润表为主、用减值分项表复核，全部是对同比利润增量的贡献。

| 环节 | 对净利润增量贡献，亿元 |
|---|---:|
{bridge_table}
| 合计：净利润增量 | +7.33 |

其中，贷款减值计提同比 **减少101.40亿元**，非信贷资产减值计提同比 **增加210.31亿元**，合计才是增加108.91亿元。年报把非信贷部分与理财回表等存量资产处置联系起来；“债权投资”减值还含按摊余成本计量的资管计划、信托计划及收益权，不能直接说是政府债券损失。这个结构不能写成“2020年新放的贷款变坏吞掉了增长”。来源：年报PDF第34、148页。所得税费用减少主要与国债利息等免税收入变化有关，也不等于法定税率下调。

**从归母净利润到普通股每股盈利，还有资本工具分配和股数。** 年报PDF第240页可以逐项闭合：

| 项目 | 2019年 | 2020年 | 变化 |
|---|---:|---:|---:|
| 归母净利润，亿元 | 281.95 | 289.28 | +7.33 |
| 减：优先股宣告股息，亿元 | 8.74 | 8.74 | 0.00 |
| 减：永续债利息，亿元 | 0.00 | 8.20 | +8.20 |
| 普通股股东利润，亿元 | 273.21 | 272.34 | −0.87 |
| 加权平均普通股股数，百万股 | 17,764 | 19,406 | +9.24% |
| 基本每股收益，元 | 1.54 | 1.40 | 按公告列值下降 |

永续债利息在这里是归母利润向普通股利润的分配扣减，没有作为营业费用重复扣一次。资本补充有助于扩展业务，但“银行做大”和“每一股普通股分得更多”仍是两个不同问题。公告EPS已四舍五入，直接用1.40/1.54得到的百分比不同于未四舍五入的每股金额变化，因此不混用两种EPS降幅。

![平安银行利润与每股收益传导]({figure_path.as_posix()})

**其他两家为什么不能硬套同一原因。** 招商快报可由营业收入减非利息净收入，算出全年净利息收入由1730.90亿元升至1850.15亿元，增长6.89%；快报没有足够全年息差、减值明细。其已公开三季报的集团前三季度净利息收益率2.51%、同比下降14个基点，第三季度为2.53%、环比回升8个基点；这不能冒充第四季度或全年息差。同页还有本行口径2.58%，不能与集团2.51%混接。来源：{source_links['cmb_q3']} PDF第7页。

兴业快报拨备前利润增加196.12亿元，税前利润只增加21.34亿元，两者差额扩大174.78亿元。它约束了收入到税前利润的传导，但快报不足以把这174.78亿元全归成信用减值，全年净息差同样未知。快报明确EPS计算考虑优先股股息，但没有足够资料完成与平安一样的逐项金额桥；不能拿平安的原因替兴业补答案。

**年度增速不高，也不表示当时没有盈利修复消息。** 用同一公司已公开的全年快报减前三季度归母利润，可以得到下列第四季度隐含金额；它仍受快报初步核算和跨公告版本差异约束，不冒充单季经审计结果。

| 公司 | 隐含2019年第四季度，亿元 | 隐含2020年第四季度，亿元 | 同比 |
|---|---:|---:|---:|
{q4_table}

来源：两家年度快报、{source_links['cmb_q3']}和{source_links['cib_q3']}的前三季度原表。这说明年度平均可能掩盖末季修复；相应信息在1月快报时已能计算，不能到2月9日又把它全部当成新的宏观利好。

截至2月9日收盘，既有20日含分红价格序列显示：

| 公司 | 参考篮子权重 | 此前20日收益 |
|---|---:|---:|
{price_table}

这些涨幅只说明价格先前已经发生明显变化，不能证明公司基本面已被完全定价，也不能据此算出“剩余合理空间”。没有同期可靠盈利预期与估值反事实，就不能把涨幅都归给年度业绩，更不能把2月9日收盘后发布的货币公告解释成这些此前涨幅的原因。行情及参考权重沿用{link_v10}。

**把各环节连在一起，2021年2月9日更准确的描述如下。**

| 环节 | 已核实的情况 | 仍不能跨越的推断 |
|---|---|---|
| 货币来源 | 1月M1同比14.7%，但公布余额与上月同为62.56万亿元；同比基数、春节跨月与持有人结构同时变化 | 不能改写成当月企业经营现金同幅度扩张 |
| 信贷结构 | 企业中长期同比多3800亿元，企事业贷款总额跨公告少3100亿元，短贷与票据分别少增 | 不能把期限变长等同所有企业融资和订单全面加速 |
| 融资价格 | 近20个定盘日FDR007相对7天政策利率的均值比前20日提高48.99个基点 | 不能因信用数量增加就认定资金持续变便宜；定盘口径不是银行净息差 |
| 银行经营 | 信贷规模、利息收入、减值、资本工具分配和普通股盈利分化；两家快报又显示末季利润修复 | 不能把收入、归母利润或累计年度增速任选一项作为全部盈利状态 |
| 订单与价格 | 制造业新订单52.3，环比回落1.3点；510300此前60日已涨15.95% | 不能把扩张水平等同边际加速，或把旧消息当全新的上涨理由 |
| 波动内部 | 300只股票中138只下行能量下降；参考权重平均个股下行能量反而升2.02% | 指数降波不等于多数股票风险同步减轻；相关性变化也会降波 |

上游金额和来源见{link_v11}，资金定盘和债券口径见{link_v12}，成分与价格口径见{link_v10}。这是混合变化背景，不能被一个“高位降波”或“牛市剪刀差改善”标签概括。上述几层也不是相互独立的票数：融资结构会影响利润，利润消息可能先进入价格，价格又进入历史波动。

按原固定观察口径，2月9日快照后的下一交易日开盘起20个交易日，E0毛收益为−11.09%，再延迟一天E1为−13.96%。这里仅保留既有结果，没有把后来的下跌反向指定为某个环节的因果结果；没有扣除交易费用、账户约束，不是可执行策略收益。这个已看过结果的病例能修正错误传导解释，不能建立独立预测成功。

**本轮的确定性止于金额、时序和逻辑约束。** 可以确认“信贷多增→银行普通股盈利加速→510300继续上涨”缺少必需的中间条件；也可以确认当时存在不同于年度均值的盈利修复消息。要回答还剩多少可获得收益，仍需要在多个历史公告中区分已公开消息、相对预期的新信息、事前价格和内部风险，保持缺失和混合原因，不把三家公司或一个病例扩成新的分组打分。

本轮完成5份当时已公开PDF、46项成对原表事实、利润金额恒等式、普通股分配及EPS复算，关键原表页已目视核对。官方旧公告当前可下载，但没有2021年首次抓取的不可变存档，仍不声称完整首版认证或独立验证。没有新增预测模型、策略账户或交易动作，目标仍为进行中。

复算文件：{file_link(OUT / 'results/原表金额与期间来源.csv', '逐项金额、页码与来源')}；{file_link(OUT / 'results/平安银行_净利润增量金额桥.csv', '利润增量明细')}；{file_link(OUT / 'results/平安银行_净利息收入规模与综合比率分解.csv', '净利息收入分解')}；{file_link(OUT / 'protocol.json', '固定范围与时点')}。
"""
    (OUT / REPORT).write_text(report, encoding="utf-8")
    result = {"at": datetime.now().astimezone().isoformat(), "study_id": "510300_BANK_PROFIT_CREDIT_BRIDGE_V13",
        "status": "BANK_PROFIT_CREDIT_TRANSMISSION_COMPLETED", "continuation_classification": "PROGRESS", "goal_achieved": False,
        "snapshot_at": "2021-02-09T21:00:00+08:00", "source_documents": 5, "paired_facts": len(facts),
        "cohort": frozen["cohort"], "reference_weight": frozen["cohort_reference_weight"],
        "findings": ["信贷规模扩张与银行净息差收窄可以并存", "平安收入增加155.84亿元而净利润增加7.33亿元", "平安减值增加由非信贷分项主导，不能归因于当年新贷款", "归母利润增加不保证普通股利润或EPS增加", "招商兴业的隐含末季利润修复在1月快报已可计算"],
        "no_view": ["招商兴业当时未披露的全年息差和减值分项", "完整历史首版认证", "相对同期预期的银行业绩新信息", "完整指数盈利传导", "独立剩余收益优势"],
        "new_models": 0, "new_accounts": 0, "orders_authorized": False, "independent_validation": False,
        "report": REPORT, "figure": str(figure_path.relative_to(OUT))}
    save("result.json", result)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    save("publication_receipt.json", {"at": result["at"], "status": "REPORT_AND_FIGURE_GENERATED_AWAITING_VISUAL_CHECK",
        "report_sha256": sha256((OUT / REPORT).read_bytes()).hexdigest(), "figure_sha256": sha256(figure_path.read_bytes()).hexdigest(),
        "saved_profit_bridge_recomputed": True, "saved_eps_recomputed": True, "goal_achieved": False})
    print(json.dumps({"报告": str(OUT / REPORT), "图表": str(figure_path), "状态": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    publish()
