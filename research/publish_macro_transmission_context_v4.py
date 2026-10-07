"""输出传导链解释、具体反例与同一背景下的更新图。"""
from pathlib import Path
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.macro_transmission_context_v4 import OUT, load, save, csv, now, digest


def pct(v):
    return "缺失" if pd.isna(v) else f"{v:.2%}"


def num(v, digits=2):
    return "缺失" if pd.isna(v) else f"{v:.{digits}f}"


def plot(weekly):
    m = load("market.csv")
    v = load("daily_vol.csv")
    d = m.merge(v, left_on="date", right_on="observation_date", validate="one_to_one")
    d = d[d.date.between("2024-08-01", "2024-11-05")].copy()
    d["date"] = pd.to_datetime(d.date)
    q = weekly[weekly.observation_date.between("2024-08-01", "2024-11-05")].copy()
    q["date"] = pd.to_datetime(q.observation_date)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1.1, 1.1]})
    fig.patch.set_facecolor("#fbfbf8")
    colors = {"price": "#173f5f", "up": "#d47d24", "down": "#a42d3c", "macro": "#527f73"}
    axes[0].plot(d.date, 100 * d.wealth / d.wealth.iloc[0], color=colors["price"], lw=2)
    axes[0].set_ylabel("510300含分红指数\n8月1日=100")
    axes[0].set_title("价格：政策公布前后的状态已经改变，旧月报仍可能保持同一个数值", loc="left", fontsize=12)
    for date, label, y in [("2024-09-20", "9/20\n政策事实尚未公布", .80), ("2024-09-24", "9/24\n新工具与降息方向公布", .98), ("2024-09-30", "9/30\n此前20日已涨26.47%", .66)]:
        t = pd.Timestamp(date)
        for ax in axes:
            ax.axvline(t, color="#9ba3a9", linestyle=":" if date != "2024-09-24" else "--", lw=1)
        axes[0].annotate(label, xy=(t, y), xycoords=("data", "axes fraction"), xytext=(-50 if date == "2024-09-20" else 12, 0), textcoords="offset points", va="top", fontsize=9, bbox={"fc": "#fbfbf8", "ec": "none", "alpha": .85})
    axes[1].step(q.date, q.spread_pp, where="post", color=colors["macro"], lw=2, label="当时已公布M1同比−M2同比")
    axes[1].axhline(0, color="#999999", lw=.6)
    axes[1].set_ylabel("剪刀差\n百分点")
    axes[1].set_title("货币读数：9/13至10/11的周度原点仍是8月数据，剪刀差−13.6个百分点", loc="left", fontsize=12)
    axes[1].legend(loc="upper left", frameon=False)
    axes[2].plot(d.date, d.v_rv20 * 100, color=colors["price"], lw=1.8, label="总波动RV20")
    axes[2].plot(d.date, d.v_upside20 * 100, color=colors["up"], lw=1.5, label="上行RMS20")
    axes[2].plot(d.date, d.v_downside20 * 100, color=colors["down"], lw=1.8, label="下行RMS20")
    axes[2].set_ylabel("年化幅度\n%")
    axes[2].set_title("风险路径：下行幅度下降可以与上涨冲击、总波动急升同时发生", loc="left", fontsize=12)
    axes[2].legend(loc="upper left", ncol=3, frameon=False)
    axes[3].plot(q.date, q.internal_breadth20 * 100, marker="o", markersize=4, color=colors["macro"], label="历史成分近20日上涨覆盖（滞后一交易日）")
    axes[3].set_ylim(0, 105)
    axes[3].set_ylabel("等权覆盖\n%")
    axes[3].set_title("内部参与：上涨覆盖显著扩展，但它不直接给出此后还能涨多少", loc="left", fontsize=12)
    axes[3].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.set_facecolor("#fbfbf8")
        ax.grid(axis="y", color="#dce0e2", lw=.6)
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.suptitle("同一个剪刀差背景下，价格、政策、波动来源与参与程度会持续变化", fontsize=16, x=.08, ha="left", y=.985)
    fig.text(.08, .014, "完整固定病例时序；不是政策收益归因或交易规则。周度宏观阶梯只表示本研究快照，不代表公告精确发生时刻。\n行情与全部未来标签沿用原冻结资料；历史首次版本未认证。", color="#5f6970", fontsize=9)
    fig.subplots_adjust(left=.10, right=.97, top=.935, bottom=.075, hspace=.37)
    fig.savefig(OUT / "figures" / "同一宏观背景下的多层变化.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def policy_clock_note():
    src = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/raw/gov_pboc_wechat_20240927.html"
    raw = src.read_bytes()
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    assert "从9月27日起" in text and "1.50%" in text
    shutil.copy2(src, OUT / "sources" / "policy_rate_effective_20240927.html")
    receipt = {"at": now(), "source_url": "https://app.www.gov.cn/govdata/gov/202409/27/519932/article.html", "local_source": str(src.relative_to(ROOT)), "sha256": digest(src), "announcement_date": "2024-09-24", "effective_date": "2024-09-27", "official_repost_publication_precision": "DAY_ONLY", "conservative_full_text_known_at": "2024-09-27T23:59:59+08:00", "inherited_operation_catalogue_first_1_5_date": "2024-09-29", "meaning": "完整底表的policy_rate字段只代表该继承目录最后记录到的操作利率，不能当首次宣布时钟或完整生效时钟。9/27那行继承值1.7%不代表政策当天没有下调。正文病例明确分开宣布、实施与目录首见。", "parent_records_changed": False}
    save(OUT / "results" / "政策利率目录时钟补充.json", receipt)


def publish():
    verify = json.loads((OUT / "verification_receipt.json").read_text(encoding="utf-8"))
    assert verify["status"].startswith("PASS_")
    x = pd.read_csv(OUT / "results" / "104个月_多层证据与原后续路径.csv")
    w = pd.read_csv(OUT / "results" / "445周_多层证据与原后续路径.csv")
    flow = pd.read_csv(OUT / "results" / "2025年6至8月_社融多增来源.csv")
    groups = pd.read_csv(OUT / "results" / "信贷订单联合场景_全部历史分布.csv")
    cases = x.set_index("stat_month")
    episode = w[w.observation_date.between("2024-09-13", "2024-10-18")].copy()
    csv(episode, "2024年9至10月_同一背景内连续更新.csv")
    policy_clock_note()
    plot(w)
    chain = [
        {"层次": "融资来源与用途", "证据": "企业/居民短期和中长期、票据；政府债占社融存量及增量构成", "已能区分": "政府融资支持、贷款期限结构变化、总量与分项分化", "不能直接推出": "企业主动资本开支；民营借贷；资金已买股票", "下一证据": "按实际投向与财政支出、置换债、行业资产负债表核对"},
        {"层次": "存款与货币结构", "证据": "M1/M2当期余额项和隐含基数项；企业/居民存款；既有活期与其他存款快照", "已能区分": "当期比率改善还是低基数；总存款增长是否等于活期增长", "不能直接推出": "同一笔钱从存款搬到理财或股市；资金活化唯一原因", "下一证据": "存款性公司资产负债两端与部门存款用途"},
        {"层次": "需求与回款", "证据": "新订单、购进和出厂价格、贷款需求调查、企业收款和周转、工业利润和应收账款、房价覆盖", "已能区分": "信贷支持是否得到部分实体活动印证；供给宽松和借款需求的差别", "不能直接推出": "贷款审批真实批准率；沪深300整体盈利预期", "下一证据": "对应沪深300行业及公司披露的收入、利润和回款"},
        {"层次": "折现与政策", "证据": "官方PE来源台账、国债利率、操作利率记录、已核实资本市场政策节点", "已能区分": "价格路径与估值不是一回事；宣布、额度、申请、操作各阶段", "不能直接推出": "市场真实预期差；政策造成涨幅的精确比例", "下一证据": "事件前已可得预期和实际落地，补完整事件目录"},
        {"层次": "已经发生的定价与参与", "证据": "过去20/60/252日收益、距一年高点、均线位置/方向、成分等权上涨覆盖", "已能区分": "下跌背景的初步修复、已走出较长涨幅、指数与多数成分不同步", "不能直接推出": "事后牛熊真值；价格低就是便宜；上涨已充分定价", "下一证据": "历史权重行业贡献与当时盈利/估值变化"},
        {"层次": "风险路径与更新", "证据": "上行/下行/均值方差项、旧波动退出、原20日后续路径", "已能区分": "下跌收敛与总波动下降；上涨冲击与卖压冲击；数据/政策发布后状态变化", "不能直接推出": "确定方向或保证回撤；足够独立的交易优势", "下一证据": "同一传导链在完整不同历史段和新资料中的复核"},
    ]
    csv(pd.DataFrame(chain), "六层传导_证据含义与缺口.csv")
    r21, rjul, raug = cases.loc["2021-01"], cases.loc["2025-07"], cases.loc["2025-08"]
    rows = []
    for month in ["2020-06", "2021-01", "2024-08", "2025-07", "2025-08"]:
        r = cases.loc[month]
        rows.append(f"| {month} / {r.observation_date} | {num(r.delta3_spread_pp,1)} | {num(r.loan_corporate_long_ytd_yoy_change_yi,0)} / {num(r.loan_household_long_ytd_yoy_change_yi,0)} | {num(r.orders_first_release_value,1)} | {pct(r.past_return60)} | {num(r.valuation_original_pe_official)} | {pct(r.internal_breadth20)} | {pct(r.E0_20_return)} / {pct(r.E1_20_return)} |")
    ep_rows = []
    for date in ["2024-09-20", "2024-09-27", "2024-09-30"]:
        r = episode[episode.observation_date.eq(date)].iloc[0]
        ep_rows.append(f"| {date} | {r.spread_pp:.1f} | {pct(r.past_return20)} | {pct(r.v_downside20)} | {pct(r.v_rv20)} | {pct(r.internal_breadth20)} | {pct(r.E0_20_return)} / {pct(r.E1_20_return)} |")
    f_rows = []
    for r in flow.itertuples():
        f_rows.append(f"| {r.stat_month} | {r.total_tsf_ytd_yoy_increase_yi/10000:.2f} | {r.government_bond_ytd_yoy_increase_yi/10000:.2f} | {r.government_share_of_tsf_yoy_increase:.1%} | {r.rmb_to_real_economy_ytd_yoy_increase_yi/10000:+.4f} |")
    joint = groups[(groups.panel == "月度") & (groups.period == "全期") & groups.joint_state.eq("剪刀差改善_信贷与订单共同支持")]
    j0 = joint[joint.variant.eq("E0")].iloc[0]
    j1 = joint[joint.variant.eq("E1")].iloc[0]
    report = f"""# 第四轮：从指标关系转向资金、实体活动与市场定价的传导链

本轮主线已经按用户最新纠正改变。M1/M2和波动率只占整条链的一部分，研究单位是同一时点的融资来源、存款结构、需求与回款、政策和估值、已经发生的价格路径、内部参与以及随后的更新。没有给这些指标打总分，也没有把牛熊标签贴回历史。

这轮取得的明确判断是：**剪刀差改善至少可能对应不同的资金与经济过程；信贷、订单甚至与剪刀差共同改善，也不足以跳过已经定价和内部参与的检查。当前资料支持机制分解与反例，仍未形成能够确定510300未来方向的完整因果链。**

## 1. 已实际接入哪些证据

在原104个月和445周快照上，连接了12类来源渠道，并保留原价格、波动分解和成分上涨覆盖。月度104行中，103行有截止日前的市场快照：贷款、存款、新订单、制造业价格各覆盖103行；社融存量79行；房价66行；银行家及企业家调查各89行；工业利润71行。季度调查在89个月中分别只使用了29个不同季度状态，重复使用不能增加独立样本。最后2026-08月报晚于2026-09-11行情截止，仍为NO_VIEW。

资料按各自公布时钟连接。金融统计中贷款与M1/M2来自同一原文，哈希一致，沿原公告上界；社融、房价、季度调查等沿各自保守可用时间。因此2025年7月金融统计的观察日，最新可用工业利润是6月、房价是6月、企业调查是二季度，不能统称为7月已经全面改善。

六层结构及每一层的“能说明什么/尚缺什么”保存在[六层证据表](results/六层传导_证据含义与缺口.csv)。价格位置、估值、实际盈利和盈利预期分别记录；PE低不能自动解释为便宜。估值和债券来源沿旧台账的次交易日可用假设，首次历史版本仍未认证。

## 2. 2025年：社融多增、剪刀差改善与私人需求全面恢复不能等同

新核对的央行同期原文给出以下**年初累计、相对上年同期多增**构成，单位万亿元：

| 统计期结束月 | 社融累计同比多增 | 政府债累计同比多增 | 政府债占社融多增 | 对实体人民币贷款累计同比多增 |
|---|---:|---:|---:|---:|
{chr(10).join(f_rows)}

91.1%、95.3%、99.4%说的是“累计同比多增”的算术贡献比例，绝不是政府融资占全部社融的比例。政府债以外还含国企等融资，也不是私人部门的严格口径。[7月14日央行实录](https://www.pbc.gov.cn/hanglingdao/128697/5580410/5580420/2025100917113619740/index.html)、[7月社融报告官方转载](https://app.www.gov.cn/govdata/gov/202508/13/534663/article.html)、[8月央行社融报告](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/523c260b344c4f1390664430295064a9/index.html)。

央行7月14日同时解释：政府债靠前发行和银行债券投资增加支持货币派生；上年整治资金空转压低了基数；企业存款恢复。同时，地方专项债置换平台贷款会压低表内贷款增速。因此不能把“贷款少增”全部解释成需求崩塌，也不能把“社融多增”全部解释成企业主动借钱扩产。该实录在2025-07-14晚间才可用，不回填到此前观察日。7月社融官方转载只有日期，按当日日末上界保留，没有强行放进8月13日21点快照。

把信贷与实体证据继续接上：2025年前七个月、前八个月企业中长期贷款分别比各自上年同期少增1.30、1.32万亿元；住户中长期分别少增0.13、0.23万亿元。对应制造业新订单49.4、49.5，仍低于50；当时最新规模以上工业利润累计同比−1.8%、−1.7%；二季度贷款需求调查同比低3.2个百分点，企业收款和周转调查同比也较弱。购进价格指数51.5/53.3，出厂价格48.3/49.1，是另一处价格传导分化，不能直接当作企业利润率变化的百分点。

这些观测**不支持把该段统一命名为“内生需求全面复苏”**。它更符合财政融资支撑、贷款置换、存款恢复与低基数并存，而部分需求和回款证据仍偏弱的混合状态。这里的“符合”是机制推断，不是已识别每一笔资金去向。

前轮已确认，2025年7/8月剪刀差三月改善分别为3.3/2.8个百分点；同一对数分解下，当期M1相对M2余额项只有+0.288/+0.223，隐含基数与修订项为+2.934/+2.500，单位均为对数百分点。两种单位不能直接相加比较，但分解说明了不能把同比改善幅度全部读成当期活化。新M1从2025年起包括个人活期等新增范围，不能和旧口径拼接。

市场已经走到哪里也不同：两次公告前20个交易日510300已分别上涨{pct(rjul.past_return20)}、{pct(raug.past_return20)}；前60日涨{pct(rjul.past_return60)}、{pct(raug.past_return60)}。原E0后20日为{pct(rjul.E0_20_return)}、{pct(raug.E0_20_return)}；延迟一交易日为{pct(rjul.E1_20_return)}、{pct(raug.E1_20_return)}。它们不能证明前述因素造成了涨幅，却已经否定“看到同样剪刀差改善，就认为处于同一行情阶段”的读法。弱信贷也不是自动看空信号。

## 3. 2021年：多项经济证据共同改善，仍可能遇到已涨较多和参与分化

2021-01数据在2021-02-09公开。剪刀差三月改善6.7个百分点；企业和住户中长期贷款当月分别同比多增3800、1957亿元；制造业新订单52.3。当时最新银行家需求判断同比上升6.3个百分点。若只是要求几项经济指标同时向好，这个时点会被算成支持案例。

但当时510300过去60日已涨{pct(r21.past_return60)}，收盘处于过去252日含分红高点，距200日均线{pct(r21.price_ma200_gap)}；前一日历史成分近20日上涨覆盖仅{pct(r21.internal_breadth20)}，说明指数新高与多数成分同周期上涨并不同步。官方PE来源台账最新可用值为{r21.valuation_original_pe_official:.2f}，这里仅作为可见估值坐标，不据单一数值认定高估。

同时下行RMS20下降、最近5日下行也同步缓和。这不是旧下跌移出窗口的唯一假象，但它仍未给未来路径提供保证。原E0后20日{pct(r21.E0_20_return)}，最差路径{pct(r21.E0_20_worst_path)}；E1为{pct(r21.E1_20_return)}。剪刀差改善本身也有强基数成分：对数相对余额当期项−0.246、隐含基数与修订项+6.252。春节错位是候选解释，未将其强行确认为唯一成因。

可以据此确定的是：“信贷/订单支持、下行收敛”这个描述与“未来20日确定向上”之间还缺定价和新增信息的证据。不能据这个已知亏损病例反向推出“高位一律卖出”，也不能用后来下跌证明当时上涨已充分定价。

## 4. 2024年：同一宏观读数中，政策、价格和波动结构会改变判断对象

以下三个固定周度原点，当时已公布的M1−M2都为−13.6个百分点，且下行RMS20下降、最近5日同步缓和：

| 观察日 | 剪刀差百分点 | 此前20日涨幅 | 下行RMS20年化 | 总RV20年化 | 成分上涨覆盖 | 后20日E0 / E1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(ep_rows)}

9月20日处于下行均线之下，9月24日的资本市场工具尚未公开。不能拿后来政策替9月20日的低位低波动补一个事前确定性理由。9月27日工具信息已出现，价格与上涨覆盖已经显著变化；9月30日价格已走出更大的上涨。后两点**下行幅度下降但总波动明显上升**，增长来自上涨冲击，不能笼统说成“降波”。[9月24日联合发布会原文](https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml)支持政策先后顺序，不证明涨幅全部由这项政策造成。

9月27日E0下一开盘是9月30日；延迟一交易日则是国庆后的10月8日，所以收益对更新和执行时刻特别敏感。9月27日E1与9月30日E0是同一个入场/退出窗口，不能当两个独立验证。表内后续收益仅用来揭示状态不同，不构成事后筛时策略。

利率也必须分清三个时间：9月24日宣布方向，9月27日正式下调至1.5%，继承操作目录直到9月29日才首次记录1.5%。底表policy_rate字段表示该目录最后已记录操作，不是完整政策生效时间；9月27日那一行的旧目录值1.7%不能解读为当天政策仍未降息。已附[时钟补充](results/政策利率目录时钟补充.json)和[官方9月27日公告转载](https://app.www.gov.cn/govdata/gov/202409/27/519932/article.html)，原父研究未修改。

![同一宏观背景中的多层变化](figures/同一宏观背景下的多层变化.png)

## 5. 全部病例与全样本约束

以下是预先固定的五个已知病例，不用它们代替全样本。中长期贷款列为同一累计区间的同比多增，单位亿元；2020-06住户+500亿元未超过披露舍入合计界550亿元，所以保留方向不确定。上涨覆盖为前一交易日可得历史成分的等权比例，并非按指数权重计算的贡献。

| 统计月 / 观察日 | 剪刀差三月变化pp | 企业 / 住户中长期同比多增 | 新订单 | 此前60日 | 可用官方PE | 上涨覆盖 | E0 / E1后20日 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

全104个月只设置了一项辅助联合核对：剪刀差改善、企业和住户中长期累计均同比多增、最新制造业新订单>=50。旧M1口径中共有{int(j0.rows)}个月符合，全部集中在2019—2021年，E0平均后20日{pct(j0.E0_20_return_mean)}，E1{pct(j1.E1_20_return_mean)}；此前60日已平均上涨{pct(j0.past_return60_mean)}。它没有提供跨时期覆盖，更不能把不同层面的“共同支持”直接当稳定收益。其余分化、缺失和未支持场景全部保存，没有删除负收益或另挑阈值。

这个三项核对只是在检查链中一个局部，主分析仍是完整证据过程。每个阶段的资金来源、估值、参与和新消息不同，不能靠继续增加筛选条件把11个月筛成几个好结果。也没有把此前冻结的贷款、社融、住户存贷等失败策略重新训练。

## 6. 哪些结论已更清楚，哪些连接还需要补

已能作出的判断：2025年这段金融改善的融资来源明显带有财政支撑，且同比基数很重要；当前所接需求与回款证据不足以证明全面内生恢复。2021年的经济支持与随后股价下跌可以并存，说明方向判断还要解释市场已经反映多少。2024年同一月报背景下，政策公布、价格跃迁、参与扩展和上/下行波动变化会令观察对象迅速改变。三者均不支持将两个指标的固定方向当完整市场判断。

尚未闭合的是“资金和政策如何传入沪深300各主要行业的盈利与估值，股价反映了其中多少”。本轮没有完整的财政实际支出对应、企业实际借款用途、历史加权行业贡献、当时盈利预期差和真实投资者买入链条。外部贸易、汇率和国际利率冲击也未完整纳入。不能凭已加入的十多项指标给剩余上涨空间写一个确定数字。

下一步沿这几个尚未闭合的连接继续核对直接来源，优先区分财政/置换资金、实体回款和指数权重行业盈利，再解释价格与估值变化；观察随新数据和政策更新。继续使用两个完整固定历史段作反证，保留同一时期的失败和分化，不扩参数网格寻找漂亮均值。

研究目标保持active，确定方向或稳定交易优势仍未证实。本轮新模型0、账户回测0、订单0、夏普NOT_COMPUTED；未制作ZIP。行情截止与原E0/E1标签不变。验证完成549个原点、两表共138列标签、6074条来源时钟及320项信贷算术；这些检查确认接表一致，不能把重建的历史资料升级成独立前向验证。

完整文件： [104个月完整链](results/104个月_多层证据与原后续路径.csv)、[445周完整链](results/445周_多层证据与原后续路径.csv)、[28个月连续历史段](results/两个固定历史段_完整28个月.csv)、[全部联合分布](results/信贷订单联合场景_全部历史分布.csv)、[来源与缺失](results/各层证据覆盖与缺失.csv)、[冻结方案](protocol.json)。
"""
    report_path = OUT / "第四轮_立体传导与明确结论.md"
    report_path.write_text(report, encoding="utf-8")
    (OUT / "00_阅读说明.md").write_text("# 阅读说明\n\n先读《第四轮_立体传导与明确结论.md》，再查看图和病例。results内两张完整表保留104个月和445周全部行；带“不含未来标签”的文件先保存输入链，再连接原冻结后续标签。未知值没有以零代填。\n\n贷款均是银行统计净增，不代表企业真实资本开支。政府社融多增比例只用于解释同期融资结构。政策利率目录不是完整宣布/生效时钟，见独立补充说明。PE、债券和宏观页面均为历史重建或既有来源快照，未认证首次版本。\n\n代码入口位于项目research目录，副本在code；按prepare、fetch、build顺序构建后运行verify_macro_transmission_context_v4.py和publish_macro_transmission_context_v4.py。build已有完成记录会阻止覆盖，复算核心可直接运行verify脚本。未建立交易系统或账户。\n", encoding="utf-8")
    save(OUT / "goal_progress.json", {"updated_at": now(), "goal_status": "active", "goal_achieved": False, "classification": "PROGRESS_INTEGRATED_CREDIT_CAUSES_REAL_ACTIVITY_PRICING_AND_DYNAMIC_CONTEXT", "user_correction_applied": True, "completed": "12类来源接入104月/445周；3份融资来源原文；五病例与两完整历史段；同一宏观背景内政策、价格、波动路径更新", "remaining": "资金用途至指数行业盈利及估值变化的连接、当时预期差、独立新验证；仍不能确定未来方向", "next_bounded_action": "沿财政和置换融资、实体回款、沪深300权重行业盈利及价格/估值变化核对传导；不把多指标打分或事后牛熊分箱当结论。"})
    for name in ["macro_transmission_context_v4.py", "verify_macro_transmission_context_v4.py", "publish_macro_transmission_context_v4.py"]:
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    save(OUT / "completion_receipt.json", {"at": now(), "round_status": "COMPLETED_MULTILAYER_CONTEXT_RESEARCH", "goal_status": "active", "goal_achieved": False, "report_sha256": digest(report_path), "verification": verify["status"], "new_models": 0, "new_accounts": 0, "orders": 0, "independent_validation": False})
    print("第四轮解释、连续病例和多层变化图已生成；总目标仍未完成。")


if __name__ == "__main__":
    publish()
