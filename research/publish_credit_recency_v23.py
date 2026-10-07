"""发布信贷区间与期限结构的完整研究记录；保持原研究结果不变。"""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import shutil
import sys
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_credit_recency_structure_v23"
BASE = OUT.as_posix()
REPORT = "第二十三轮_信用支持的近期性与期限用途.md"
FIGURE = "figures/累计信用_近期净增与原11个月路径.png"
SOURCE_2020 = "https://www.gov.cn/xinwen/2021-01/29/5583670/files/738465a2b90e4231b0e333e932215a9b.pdf"
SOURCE_2021 = "https://jrj.wuhan.gov.cn/ynzx_57/xwzx/202105/t20210506_1680086.shtml"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def link(label, path):
    return f"[{label}](<{BASE}/{path}>)"


def supplement():
    if (OUT / "supplement_receipt.json").exists():
        raise RuntimeError("投向补充已保存，不覆盖。")
    assert (OUT / "supplement_scope.json").exists()
    sources = {
        "2020FY": {"file": "2020FY_政府网原报告.txt", "raw": "2020FY_政府网原报告.pdf", "known": "2021-01-30T23:59:59+08:00", "url": SOURCE_2020,
                   "clock_evidence": "2020FY_政府网当时报道.html", "type": "政府网保存的央行原报告PDF；以1月30日官方转载确认此前已公布"},
        "2021Q1": {"file": "2021Q1_武汉金融局当时转载.txt", "raw": "2021Q1_武汉金融局当时转载.html", "known": "2021-05-06T23:59:59+08:00", "url": SOURCE_2021,
                   "clock_evidence": "2021Q1_武汉金融局当时转载.html", "type": "政府网站当时转载的央行报告数据；未取得央行原页面，不声称最早发布时间"},
    }
    facts = []
    specs = [
        ("2020FY", "本外币企事业单位总额", "本外币企事业单位贷款余额110.53万亿元,同比增长12.4%", 1105300, 12.4, 121600, 28000),
        ("2020FY", "本外币企事业中长期", "中长期贷款余额66.06万亿元,同比增长15.3%", 660600, 15.3, 87800, 29100),
        ("2020FY", "本外币企事业短期及票据", "短期贷款及票据融资余额41.57万亿元,同比增长8.2%", 415700, 8.2, 31400, -1419),
        ("2020FY", "本外币工业中长期", "本外币工业中长期贷款余额11.01万亿元,同比增长20%", 110100, 20.0, 18400, 12500),
        ("2020FY", "本外币服务业中长期", "本外币服务业中长期贷款余额45.04万亿元,同比增长14.3%", 450400, 14.3, 56200, 11700),
        ("2020FY", "本外币住户经营性", "本外币住户经营性贷款余额13.62万亿元,同比增长20%", 136200, 20.0, 22700, 10000),
        ("2020FY", "本外币住户消费性", "住户消费性贷款余额49.57万亿元,同比增长12.7%", 495700, 12.7, 55900, -5717),
        ("2021Q1", "本外币企事业单位总额", "本外币企事业单位贷款余额116.09万亿元,同比增长10.9%", 1160900, 10.9, 55600, -7170),
        ("2021Q1", "本外币工业中长期", "本外币工业中长期贷款余额11.92万亿元,同比增长24.2%", 119200, 24.2, 9160, 4873),
        ("2021Q1", "本外币服务业中长期", "本外币服务业中长期贷款余额47.83万亿元,同比增长14.9%", 478300, 14.9, 27700, 5461),
        ("2021Q1", "本外币住户经营性", "本外币住户经营性贷款余额14.75万亿元,同比增长24.6%", 147500, 24.6, 11300, 6437),
        ("2021Q1", "本外币住户消费性", "住户消费性贷款余额51万亿元,同比增长14.1%", 510000, 14.1, 14300, 7135),
    ]
    def normalize(s):
        return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s))
    for period, label, needle, stock, yoy, flow, change in specs:
        src = sources[period]
        text = normalize((OUT / "sources" / src["file"]).read_text("utf-8"))
        pos = text.find(needle)
        assert pos >= 0, (period, needle)
        evidence = text[pos:pos + 180]
        # 表示金额的原文数词可能以亿元或万亿元给出，均按同一片段复核。
        for value in [flow, abs(change)]:
            candidates = [f"{value:g}亿元", f"{value / 10000:g}万亿元"]
            assert any(v in evidence for v in candidates), (period, label, value, evidence)
        sign = "多增" if change > 0 else "少增"
        assert sign in evidence
        facts.append({"report_period": period, "category": label, "stock_yi": stock, "stock_yoy_pct": yoy,
                      "period_net_increase_yi": flow, "reported_yoy_net_increase_change_yi": change,
                      "evidence": evidence, "known_at_upper_bound": src["known"], "source_url": src["url"],
                      "source_type": src["type"], "source_sha256": sha(OUT / "sources" / src["raw"]),
                      "historical_first_vintage_verified": False})
    pd.DataFrame(facts).to_csv(OUT / "results/两个固定季度_贷款期限行业与住户用途.csv", index=False, encoding="utf-8-sig")
    origins = pd.read_csv(OUT / "results/原11个共同支持月份_全部背景与原路径.csv").set_index("stat_month")
    clocks = []
    for m in ["2021-01", "2021-05"]:
        for period, s in sources.items():
            clocks.append({"origin_stat_month": m, "snapshot_at": origins.loc[m, "snapshot_at"], "report_period": period,
                           "known_at_upper_bound": s["known"],
                           "admission": "AVAILABLE_RECONSTRUCTED" if pd.Timestamp(s["known"]) <= pd.Timestamp(origins.loc[m, "snapshot_at"]) else "NO_VIEW_PUBLISHED_AFTER_ORIGIN"})
    pd.DataFrame(clocks).to_csv(OUT / "results/两个季度报告_两个原观察点时钟.csv", index=False, encoding="utf-8-sig")
    save("supplement_receipt.json", {"at": now(), "status": "PASS_12_FACTS_WITH_SOURCE_TEXT_AND_CONSERVATIVE_CLOCKS", "facts": len(facts), "source_scope": sources,
                                  "pdf_visual_pages": [1, 4], "original_pbc_pages": "两条旧地址404，未将404页面当作原报告",
                                  "association_copy": "证书域名不匹配，未关闭验证；改用政府网站已核实的当时转载",
                                  "historical_first_vintage_verified": False, "selection_after_results": True, "independent_validation": False})
    print("已保存两个季度的12项投向事实；后发资料在早观察点保持NO_VIEW。")


def draw(wide, fixed):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(17, 13), facecolor="#fbfaf7")
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 1.1], left=.095, right=.97, top=.88, bottom=.10, hspace=.48)
    a, b, c = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax in [a, b, c]:
        ax.set_facecolor("#fbfaf7")
    history = wide[wide.stat_month.between("2019-01", "2021-12")]
    fieldnames = ["credit_YTD_corporate_long_direction", "credit_3_corporate_long_direction", "credit_YTD_household_long_direction", "credit_3_household_long_direction"]
    coding = {"多增": 0, "少增": 1, "显示舍入界内": 2, "NO_VIEW": 3}
    colors = ["#247984", "#b9673c", "#d8c58e", "#d8d8d8"]
    arr = np.array([[coding[v] for v in history[f]] for f in fieldnames])
    a.imshow(arr, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
    a.set(yticks=range(4), yticklabels=["企事业中长期 · 累计", "企事业中长期 · 近3月", "住户中长期 · 累计", "住户中长期 · 近3月"],
          xticks=np.arange(0, len(history), 3), xticklabels=history.stat_month.iloc[::3].tolist())
    a.tick_params(axis="x", labelsize=10)
    for x in [11.5, 23.5]:
        a.axvline(x, color="white", lw=3)
    a.set_title("A  累计仍多增，近期已经转弱：两个部门的中长期净增方向", loc="left", fontweight="bold", pad=17)
    a.legend(handles=[Patch(color=v, label=k) for k, v in zip(["同比多增", "同比少增", "显示舍入界内", "缺可比历史"], colors)],
             ncol=4, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01), fontsize=9)
    x = np.arange(len(fixed))
    long = fixed.credit_3_corporate_long_yoy_change_yi.to_numpy(float)
    short = fixed.credit_3_corporate_short_yoy_change_yi.to_numpy(float)
    bills = fixed.credit_3_bills_yoy_change_yi.to_numpy(float)
    total = fixed.credit_3_corporate_total_yoy_change_yi.to_numpy(float)
    residual = total - long - short - bills
    pos, neg = np.zeros(len(fixed)), np.zeros(len(fixed))
    for name, values, color in [("中长期", long, "#247984"), ("短期", short, "#b9673c"), ("票据", bills, "#c8a155"), ("未单列项及舍入差", residual, "#9ca1a7")]:
        values = values / 10000
        base = np.where(values >= 0, pos, neg)
        b.bar(x, values, bottom=base, color=color, label=name, width=.65)
        pos += np.maximum(values, 0)
        neg += np.minimum(values, 0)
    b.plot(x, total / 10000, "D", color="#202e3c", ms=6, label="企事业总额净差")
    b.axhline(0, color="#555555", lw=.8)
    b.set(xticks=x, xticklabels=fixed.stat_month, ylabel="最近3月净增的同比差／万亿元", ylim=(-1.7, 2.15))
    b.set_title("B  原11个月：中长期、短期与票据的增减可能相反", loc="left", fontweight="bold", pad=16)
    b.legend(ncol=5, frameon=False, loc="upper center", fontsize=9)
    for i, v in enumerate(total):
        if v < 0:
            b.annotate(f"{v:,.0f}亿", (i, v / 10000), xytext=(0, -17), textcoords="offset points", ha="center", fontsize=9,
                       bbox={"facecolor": "#fbfaf7", "edgecolor": "none", "pad": 1})
    bw = .32
    c.bar(x - bw / 2, fixed.E0_20_return * 100, bw, color="#247984", label="原E0")
    c.bar(x + bw / 2, fixed.E1_20_return * 100, bw, color="#9ca1a7", label="原E1再延一天")
    c.axhline(0, color="#555555", lw=.8)
    c.set(xticks=x, xticklabels=fixed.stat_month, ylabel="原后续20日毛收益／%", ylim=(-17, 15))
    c.set_title("C  全部原路径保留；不能依期限结构事后删除亏损月", loc="left", fontweight="bold", pad=16)
    c.legend(frameon=False, ncol=2, loc="upper left")
    for ax in [b, c]:
        ax.tick_params(axis="x", labelrotation=30, labelsize=10)
    fig.suptitle("信用支持发生在什么期限、什么部门、什么时间？", x=.095, y=.975, ha="left", fontsize=22, fontweight="bold")
    fig.text(.095, .933, "104个月统一核对；A展示2019—2021连续区间，B/C展示事先已有的11个共同支持月份。", fontsize=12, color="#555555")
    fig.text(.095, .035, "金额为跨公告净增差，不是贷款发放总额或资金用途因果贡献。A的累计与3月区间长度不同；新旧统计范围不拼接。", fontsize=10, color="#555555")
    fig.savefig(OUT / FIGURE, dpi=160, bbox_inches="tight")
    plt.close(fig)


def publish():
    if (OUT / "completion.json").exists():
        raise RuntimeError("本轮已完成，不覆盖。")
    assert json.loads((OUT / "verification.json").read_text("utf-8"))["status"].startswith("PASS_")
    assert (OUT / "supplement_receipt.json").exists()
    wide = pd.read_csv(OUT / "results/104个月_信用近期性与完整经营定价背景.csv")
    fixed = pd.read_csv(OUT / "results/原11个共同支持月份_全部背景与原路径.csv")
    both = wide[wide.credit_YTD_both_long.eq("两部门中长期共同多增")]
    assert len(both) == 23
    mixed = fixed[fixed.credit_3_corporate_total_direction.eq("少增")]
    assert len(mixed) == 5
    draw(wide, fixed)
    rows = []
    for r in fixed.itertuples():
        total = r.credit_3_corporate_total_yoy_change_yi
        other = total - r.credit_3_corporate_long_yoy_change_yi - r.credit_3_corporate_short_yoy_change_yi - r.credit_3_bills_yoy_change_yi
        rows.append([r.stat_month, f"{r.credit_3_corporate_long_yoy_change_yi:+,.0f}", f"{r.credit_3_corporate_short_yoy_change_yi:+,.0f}",
                     f"{r.credit_3_bills_yoy_change_yi:+,.0f}", f"{other:+,.0f}", f"{total:+,.0f}",
                     f"{r.credit_3_household_long_yoy_change_yi:+,.0f}" + ("（舍入界内）" if r.credit_3_household_long_direction == "显示舍入界内" else ""),
                     f"{r.E0_20_return * 100:+.2f}%", f"{r.E1_20_return * 100:+.2f}%"])
    context = []
    for r in fixed.itertuples():
        margin = "缺失" if pd.isna(r.industrial_0_margin_yoy_change) else f"{r.industrial_0_margin_yoy_change:+.2f}pp"
        context.append([r.stat_month, f"{r.delta3_spread_pp:+.1f}", f"{r.d3_relative_current_log_pp:+.3f}",
                        f"{r.d3_relative_base_revision_log_pp:+.3f}", f"{r.orders_first_release_value:.1f} / {r.orders_orders_change1_pp:+.1f}",
                        margin, f"{r.context_fdr_policy_gap_bp_mean_change:+.2f}", f"{r.past_return60 * 100:+.2f}%"])
    body = f"""**本轮把信用支持拆成“最近还是累计、总额还是中长期、企业还是住户、经营还是消费”以后，得到更具体的结论：原先看似同向的信贷支持中，既有真实的中长期扩张，也有明显的期限重排和近期放缓；不能把它们压成一个全面需求改善标签。**

这轮仍研究原104个月，9类贷款、单月/最近3月/年内累计三种固定区间，共2808项区间金额和2808项跨公告同比比较。原389列经营、货币、价格、波动、内部参与及5/20/60日结果完整保留。上一轮有实际进展，本轮继续推进，总目标尚未完成。

**累计仍然支持，并不保证最近三个月仍支持。**

原104个月中，有23个月企事业和住户中长期贷款年内累计均同比多增。按固定最近三个月重看，14个月仍共同多增，8个月至少一方未确认多增，1个月缺可比历史。这些23个月均在2019—2021年，不能解释成新旧M1两套口径都有同样规律。

8个月中的7个月是2021年6—12月：年初形成的累计多增仍在，而住户或企事业最近三月的净增已不再同比多增。2021年8—10月，两部门最近三月均明确少增。另一个是2020年12月，住户近三月按显示值只多400亿元，合计显示舍入界为±650亿元，不能确认其方向为正。这里的“未确认多增”同时包含明确少增与舍入界内，不能全部称作信用收缩。

最近三月采用和剪刀差变化同样的三个统计月，并与上年同期比较；它不是季节调整，也不能消除春节日期和疫情基数。累计与三月长度不同，不比较二者绝对量谁更大。

**原11个共同支持月份中，5个月的中长期增加与企事业总额少增并存。**

下表各金额是“最近三个统计月净增加额减去年同期净增加额”，单位亿元；不是贷款余额同比增速，也不是发放额。企业分项沿企事业口径，包括非金融企业及机关团体。未单列项及舍入差完整保留，前四个分项相加等于总额差。

{table(['货币月','企业中长期','企业短期','票据','未单列/舍入','企事业总额','住户中长期','原E0后20日','原E1后20日'], rows)}

五个月为2020年8、9、10月和2021年1、5月。以2020年8月为例，企事业中长期最近三月多增8852亿元，但票据少增9472亿元、短期少增181亿元，加上未单列项变化，总额少增872亿元。它表示结构偏向中长期，没有追踪同一笔贷款被置换。

2021年5月的最近三月对应3—5月。中长期多增5981亿元，短期少增8992亿元，票据少增4846亿元，未单列项及舍入差贡献944亿元，合计企事业总额少增6913亿元。同期净增本身仍为31609亿元，上年同期为38522亿元；**少增不等于余额净减少，更不等于没有实体融资。**

从全年连续观察，2021年12个统计月企业中长期累计都同比多增，但企事业总额11个月明确少增，10月在显示舍入界内。这个期限区别有连续证据，不再只依赖一个春节病例。

**投向资料确认了部分真实支持，也限制了“居民需求全面转强”的说法。**

在完成全样本拆分以后，本轮固定补核2020年全年和2021年一季度两份贷款投向资料，用于原2021年1月、5月两个病例；补充范围和结果后选择均已记录，不是新增独立检验。

2020年末本外币工业中长期贷款余额同比增长20%，全年净增比上年多1.25万亿元；服务业中长期余额同比增长14.3%。这些是具体部门的信贷扩张证据，不能把前述结构变化一概说成纯统计假象。同一报告中，住户经营性贷款全年同比多增1万亿元，住户消费性贷款反而同比少增5717亿元。**住户信用支持还需区分经营与消费，不能将住户贷款一律当作消费、住房购买或企业销售收入。**[央行原报告，政府网PDF第1、2、4页]({SOURCE_2020})。

到2021年一季度，本外币工业中长期余额同比增长24.2%，企事业全部贷款却同比少增7170亿元；住户经营性与消费性贷款当季则分别同比多增6437、7135亿元。支持真实存在，部门与期限也确实分化，不能套用一个全经济同步强弱结论。[武汉金融局2021年5月6日对央行报告数据的转载]({SOURCE_2021})。

两份报告的本外币统计、余额增速和季度/全年净增，与本轮人民币月度区间差额分别保存，未混算成同一数字。没有贷款发放和回收的逐笔流水，不能用净额唯一分离信贷供给、借款需求、偿还或核销，更不能确认资金已支付给哪些沪深300公司。

2020年报告的央行旧发布地址已404，但政府网原PDF取得成功；用2021年1月30日官方转载确认此前已公开，作为保守上界。2021年一季度央行旧地址也404，本轮仅将已保存的5月6日政府网站转载作为该资料可得上界，不声称拿到了4月30日的原始送达。它在2021年2月9日观察点不可用，在6月10日观察点已可用。全部来源仍未证明历史不可修订首版。

**把信用来源接回订单、利润、资金成本、既有价格和波动。**

下表保留原11个月，而非按新的信用解释重新筛选。普通剪刀差变化用百分点；当期与隐含基数项用可相加的对数百分点，二者不能与普通百分点混加。资金价是近20条与前20条FDR007相对政策利率均值之差，不是企业贷款利率。

{table(['货币月','剪刀差三月Δ','当期相对余额项','隐含基数/修订项','新订单/较前月Δ','已知工业利润率同比Δ','资金价相对变化bp','此前60日收益'], context)}

这里可以看清，订单仍扩张、边际订单放缓、不同期限融资分化、已知利润率恢复和金融资金成本变化可以同时发生。宏观工业利润率又不等于沪深300整体利润率，不能越过公司与指数行业权重的连接。

对波动的解释继续沿用原测量：下行缓和可能含旧损失退出，也可能与已发生反弹同步；它不能证明上述信用已经变成未被定价的新收益。所有原总波动、上下行尺度、近期下行变化与市场广度在完整表中保留。本轮不新增波动阈值或牛熊标签。

不能因为找出5个月融资结构分化，就把它们删除来改善收益。即使2019年3月最近三月两部门中长期和企事业总额均多增、订单扩张，公告时510300此前60日已经上涨29.50%，之后原E0/E1仍为−8.21%/−5.28%；它仍然保留。2019年12月也是中长期和总额多增，原E0/E1约−0.17%/−1.06%，E0途中最差收盘约−11.87%。信用解释更加准确，并不自动补上进入前的方向优势。

原11个月E0/E1平均20日毛收益仍为+0.15%/+0.48%。这些是原样本观察，未扣费、未模拟账户，不能称为策略收益；本轮没有计算新组收益来挑选赢家。

![累计与近期信用、原11个月的完整路径](<{BASE}/{FIGURE}>)

本轮已经把“信贷多增”的含义推进到可复查的部门、期限、区间和用途层面。下一段尚未闭合的是：这些融资变化如何形成指数主要公司的增量订单、现金与盈利预期，以及这些变化相对价格中的既有预期还剩什么。不能用贷款分类直接填写这两个答案。

金额与原文、区间覆盖、显示舍入、历史可见时钟和389个原列均已复算通过；961个非缺失原文金额重新解析，两个未披露分项保留。2023年扩围不跨范围比较；2026年8月原文结构保留，但公告晚于冻结行情截止日，观察仍为NO_VIEW。核对通过证明本轮测量一致，不证明独立预测有效。新增模型0、账户0、订单0，总目标保持进行中。

直接文件：

- {link('104个月完整背景与区间比较','results/104个月_信用近期性与完整经营定价背景.csv')}；{link('原11个月全表','results/原11个共同支持月份_全部背景与原路径.csv')}。
- {link('2808项区间金额及公式','results/2808项_当月三月累计净增与公式.csv')}；{link('2808项跨公告同比比较','results/2808项_同区间跨公告同比比较.csv')}。
- {link('两个季度的行业与用途事实','results/两个固定季度_贷款期限行业与住户用途.csv')}；{link('补充资料对原观察点的可得时钟','results/两个季度报告_两个原观察点时钟.csv')}。
- {link('原文与区间复算','verification.json')}；{link('冻结范围','protocol.json')}；{link('投向补充范围与来源状态','supplement_receipt.json')}。
"""
    (OUT / REPORT).write_text(body, encoding="utf-8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save("publication_draft.json", {"at": now(), "report": REPORT, "report_sha256": sha(OUT / REPORT), "figure": FIGURE,
                                   "figure_sha256": sha(OUT / FIGURE), "visual_review": "PENDING"})
    print("第二十三轮报告与三幅图已生成，待视觉检查。")


def complete():
    if (OUT / "completion.json").exists():
        raise RuntimeError("已存在本轮完成回执。")
    d = json.loads((OUT / "publication_draft.json").read_text("utf-8"))
    v = json.loads((OUT / "visual_review.json").read_text("utf-8"))
    assert v["status"] == "PASS_VISUAL_REVIEW" and v["figure_sha256"] == sha(OUT / FIGURE)
    assert d["report_sha256"] == sha(OUT / REPORT) and d["figure_sha256"] == sha(OUT / FIGURE)
    save("completion.json", {"at": now(), "status": "COMPLETED_FIXED_CREDIT_RECENCY_AND_STRUCTURE_COMPARISON", "previous_turn_classification": "PROGRESS",
         "current_turn_classification": "PROGRESS", "goal_status": "active", "goal_achieved": False,
         "report": REPORT, "report_sha256": sha(OUT / REPORT), "figure": FIGURE, "figure_sha256": sha(OUT / FIGURE),
         "verification_sha256": sha(OUT / "verification.json"), "new_models": 0, "new_accounts": 0, "orders_authorized": False,
         "new_evidence": "累计共同多增23个月中，最近三月仅14个月确认同向、8个月未确认、1个月缺历史；原11个月中5个月中长期多增而企事业总额少增。两个季度补证真实工业信贷及住户经营/消费分化。",
         "remaining": "融资到指数主要公司经营与尚未反映的盈利预期仍未闭合；未形成独立可重复的方向优势。", "independent_validation": False,
         "global_mandate_modified": False})
    print("第二十三轮完成，完整研究目标保持进行中。")


if __name__ == "__main__":
    {"supplement": supplement, "publish": publish, "complete": complete}[sys.argv[1]]()
