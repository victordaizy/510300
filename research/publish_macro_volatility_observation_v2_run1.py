"""把首轮宏观与波动观察整理为中文报告、静态图及可筛选本地底表。"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT if (ROOT / "inputs/market_daily.csv").exists() else ROOT / "reports/research/510300_macro_volatility_observation_v2_run1"
RESULT = STUDY / "results"
OLD = "M1_OLD_M2_MMF2018"
NEW = "M1_NEW2025"


def pct(value: float) -> str:
    return "缺失" if pd.isna(value) else f"{value * 100:+.2f}%"


def pp(value: float) -> str:
    return "缺失" if pd.isna(value) else f"{value * 100:+.2f}"


def table(headers, rows) -> str:
    return "|" + "|".join(headers) + "|\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("|" + "|".join(str(v) for v in row) + "|" for row in rows)


def figures(md, u) -> None:
    folder = STUDY / "figures"
    folder.mkdir(exist_ok=True)
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "#fbfaf7", "figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for r, (regime, title, color) in enumerate([(OLD, "旧M1口径", "#236777"), (NEW, "新M1口径", "#bb653b")]):
        for c, (y, ylabel) in enumerate([("pre_publication_return60", "公告前60日收益"), ("E0_20_return", "E0后20日毛收益")]):
            d = md.loc[md.training_regime == regime, ["delta3_spread_pp", y]].dropna()
            ax = axes[r, c]
            ax.scatter(d.delta3_spread_pp, 100 * d[y], color=color, s=34, alpha=.75, linewidth=.5, edgecolors="white")
            ax.axhline(0, color="#bfc5c8", lw=.8)
            ax.axvline(0, color="#bfc5c8", lw=.8)
            rho = d.delta3_spread_pp.corr(d[y], method="spearman")
            ax.set_title(f"{title} · n={len(d)} · 秩相关 {rho:.3f}", loc="left", fontsize=12)
            ax.set_xlabel("M1−M2剪刀差三个月变化 / 百分点")
            ax.set_ylabel(ylabel + " / %")
            ax.grid(alpha=.14)
    fig.suptitle("货币改善对应此前行情，还是仍有后续空间？", fontsize=17, weight="bold")
    fig.savefig(folder / "时序与后续分布.png", dpi=170)
    plt.close(fig)
    selection = [("H1", "return", "M1主导 − M2主导\n月度改善组"), ("H2", "worst_close", "下行缓和 − 相似历史对照\n途中最差收盘"), ("H4", "return", "剪刀差改善 − 相似历史对照\n未来20日收益")]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ytick, labels = [], []
    for i, (q, metric, label) in enumerate(selection):
        for k, delay in enumerate(["E0", "E1"]):
            row = u.loc[(u.question == q) & (u.training_regime == OLD) & (u.metric == metric) & (u.delay == delay)].iloc[0]
            y = -i * 2 + (0.2 if k == 0 else -0.2)
            ax.errorbar(row.mean_difference * 100, y, xerr=np.array([[row.mean_difference - row.ci_low], [row.ci_high - row.mean_difference]]) * 100,
                        fmt="o", color="#236777" if k == 0 else "#bb653b", capsize=4, lw=2, label=delay if i == 0 else None)
        ytick.append(-i * 2)
        labels.append(label)
    ax.set_yticks(ytick, labels)
    ax.axvline(0, color="#667278", lw=1)
    ax.set_xlabel("差异 / 百分点；横线为6个发布周期时间块的95%描述性区间")
    ax.set_title("旧口径的主比较均未显示稳定优势", loc="left", fontsize=17, weight="bold")
    ax.legend(title="E1比E0延后一交易日", frameon=False)
    ax.grid(axis="x", alpha=.14)
    fig.savefig(folder / "有限条件比较.png", dpi=170)
    plt.close(fig)


def write_report(md, wd, g, c, u, matched) -> None:
    def rho(regime, variable, outcome):
        return float(c.loc[(c.origin == "月度") & (c.training_regime == regime) & (c.period == "全期") & (c.x == variable) & (c.y == outcome), "spearman"].iloc[0])
    rows = []
    selection = [("H1", "return", "M1侧主导改善 − M2侧主导改善", "24 / 7个月"), ("H3", "return", "改善时此前已涨 − 此前未涨", "17 / 14个月"), ("H2", "worst_close", "下行波动缓和 − 相似价格/RV历史对照", "163周，72个目标发布周期"), ("H4", "return", "剪刀差改善 − 相似价格/波动历史对照", "106周，28个目标发布周期")]
    for q, metric, desc, n in selection:
        a = u.loc[(u.question == q) & (u.training_regime == OLD) & (u.metric == metric) & (u.delay == "E0")].iloc[0]
        b = u.loc[(u.question == q) & (u.training_regime == OLD) & (u.metric == metric) & (u.delay == "E1")].iloc[0]
        rows.append([desc, n, pp(a.mean_difference), f"[{pp(a.ci_low)}, {pp(a.ci_high)}]", pp(b.mean_difference)])
    main_table = table(["固定比较", "覆盖", "E0差异/百分点", "95%描述性区间", "E1差异/百分点"], rows)
    raw = g.loc[(g.origin == "月度") & (g.period == "全期") & (g.horizon == 20) & (g.delay == 0) & (g.group_variable == "arithmetic_group") & g.group.isin(["M1侧主导改善", "M2侧主导改善"])]
    raw_table = table(["口径", "算术来源", "成熟月数", "20日平均毛收益", "中位数", "去掉最好一例后均值"], [["旧" if r.training_regime == OLD else "新", r.group, r.rows, pct(r["mean"]), pct(r["median"]), pct(r.mean_without_best_one)] for _, r in raw.iterrows()])
    vr = g.loc[(g.origin == "周度") & (g.training_regime == OLD) & (g.period == "全期") & (g.horizon == 20) & (g.delay == 0) & (g.group_variable == "volatility_group")]
    vtable = table(["旧口径周度过程", "周数", "涉及发布周期", "20日平均毛收益", "中位数"], [[r.group, r.rows, r.release_cycles, pct(r["mean"]), pct(r["median"])] for _, r in vr.iterrows()])
    text = f"""# 510300宏观与波动规律观察 V2：首轮结果

方案冻结：2026-09-28；结果整理：2026-09-29。行情固定截至2026-09-11。性质：历史描述与有限条件比较。本轮没有仓位、账户回测或夏普计算。

**已经完成104个月复算、445个固定周度观察点和全104个月原文事实首遍提取。旧口径没有显示可直接转为交易条件的稳定宏观增量；新口径有正向线索，但样本及相似对照都很短。**

结果允许保留线索，也允许否定直觉。这里没有以“必须找到正规律”为完成条件，更没有证明夏普1.2目标已经实现。

## 1. 先回答最实际的问题：改善后是否仍有空间？

旧口径中，剪刀差三个月变化与**公告前60日收益**的Spearman秩相关为 **{rho(OLD, 'delta3_spread_pp', 'pre_publication_return60'):.3f}**，与**公告后E0的20日毛收益**为 **{rho(OLD, 'delta3_spread_pp', 'E0_20_return'):.3f}**；公告前20日为{rho(OLD, 'delta3_spread_pp', 'pre_publication_return20'):.3f}，E1后20日为{rho(OLD, 'delta3_spread_pp', 'E1_20_return'):.3f}。这更像一条与此前中期行情相关的环境信息，尚不能据此证明确定的领先或滞后因果。

控制相近的过去20/60日收益、总波动、下行尺度及其变化后，剪刀差改善周相对三个历史非改善对照的20日收益差为 **−1.15个百分点**。延迟一天为 **−0.73个百分点**，区间均跨零。控制只能使用当时已经完整成熟的历史标签，同一发布周期最多选一个对照，目标周期等权。

因此，这轮不支持“剪刀差改善就加仓”，也不支持反过来按剪刀差改善去做空。它只说明这次固定的改善表示没有得到稳定正增量证据。

{main_table}

区间按完整发布周期顺序以连续6周期时间块重采样2000次，固定种子。它们是描述性不确定性区间；相邻行情、20日窗口重叠、历史控制反复使用和未观察因素仍可能影响结论。没有把周数当作独立宏观事件数，也没有把区间解释为因果保证。H1的M2侧仅7个月，对比本身较弱。

![固定比较](figures/有限条件比较.png)

## 2. 剪刀差上升的算术来源不同，不能预设哪类一定更好

“M1侧主导”只是M1三个月加速对剪刀差改善的正向贡献较大；“M2侧主导”是M2减速的正向贡献较大。**二者不是“需求复苏”和“资金搬家”的已识别原因。**

{raw_table}

旧口径M1侧24个月的平均数略正，但去掉最好一例后变为负；M2侧只有7个月。两者都不足以形成稳定入场规律。新口径M1侧7个月优于M2侧唯一一个月，但一个对照无法代表一个完整分布；该小样本重采样出现的狭窄或退化区间不具备可靠推断含义。

反例不能删掉：2021年1月货币数据在2021-02-09公布，属于M1侧主导改善，E0后20日毛收益 **−11.09%**，途中最差收盘 **−12.74%**；延迟一天仍为 **−13.96%**。2019年3月同类公告后的20日毛收益为 **−8.21%**。这些是按事先固定的最差两例规则列出的反例，未被用来倒填经济原因。

## 3. 下行幅度缓和能描述已发生过程，但没有稳定预测出更轻的后续损失

{vtable}

直接看分组，“价格已经上涨且下行缓和”并没有稳定优于其他过程。相似过去价格和RV条件的历史近邻比较中，下行缓和对应的20日最差收盘改善只有 **0.19个百分点**，区间为 **[−0.73，+1.05]个百分点**；总收益差 **+0.53个百分点**，区间 **[−1.09，+2.40]**。分时期看，最差收盘差异由2018—2021年的+0.69个百分点转为2022—2024年的−0.35个百分点。

这并不否认总波动和下行波动需要分开：上涨会抬高总波动，下行占比下降也可能是正收益平方扩大分母。但“历史下行幅度开始下降”在这里尚不是稳定的未来风险改善信号。

## 4. 新口径保留正向线索，但不把短样本叫作独立验证

新M1有20个月公告，其中19个月E0的20日结果成熟；严格同口径三月变化与后20日收益可比较的只有 **16个月**。剪刀差三月变化与E0后20日毛收益的秩相关为 **{rho(NEW, 'delta3_spread_pp', 'E0_20_return'):.3f}**，与公告前20日为 **{rho(NEW, 'delta3_spread_pp', 'pre_publication_return20'):.3f}**，与此前60日为 **{rho(NEW, 'delta3_spread_pp', 'pre_publication_return60'):.3f}**。它同时关联此前和以后，不能据此断言全是滞后，也不能证明后续增量已经成立。

更关键的是，H4可匹配目标只有3周，且都属于**同一个发布周期**。因此价格/波动控制后的增量仍然没有可用的分布证据。原始相关为正应完整保留，后续需要新口径下新的发布周期和可比对照，不能用旧口径拼接扩样。

![先后顺序](figures/时序与后续分布.png)

## 5. 原因证据补到了哪一步？

全部104份官方原文已按统一模板提取M1/M2余额、住户/企业/财政/非银存款，以及住户、企业、企业中长期贷款等7项增量事实。7个增量字段均完成提取，原句、单位、原文路径、发布时间和统计注释随表保存。**59份是当月分项、45份是年初累计分项，分别标明，没有直接拼成一条同口径月增量。**

但企业总存款不等于企业活期存款，财政存款不等于实际支出，贷款增加也不能唯一说明支付或需求改善。所以104个月的唯一经济原因均保持“未知/可混合”。这是识别边界，不能把自动提取的事实改名成已经证实的原因。

补充核对了三份官方机制资料：

- 2024年8月5日转载的央行工作会议已确认手工补息及资金空转整治，可支持制度背景的存在，不能回填成4—7月已知的量化原因。[官方会议资料](https://jrj.sh.gov.cn/ZXYW178/20240805/e7a428bfdf2a4c0bb3e05437a5998e3f.html)
- 2024年第三季度执行报告解释了存款、资管、银行融资及证券保证金之间的迁移与反馈。因此货币变化可以受金融投资行为影响；这份11月材料只作此前月份的事后机制说明。[执行报告专栏4，印刷页28—29](https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/5347949/afbfa5df25ee45889d916a2819b60a43/2024110815410752868.pdf)
- 新M1纳入个人活期及支付机构备付金，旧的“居民活期转向企业活期必然增加M1”解释不能机械移植；2024年的回溯新口径没有提前进入2024年观察。[M1修订答记者问](https://jrj.sh.gov.cn/SCDT197/20241203/e2d0d7d348ab4aaaa2fde9a976d3ae17.html)

可比余额的基数分析也有边界：同一统计定义下可以复算相对余额的一年差分变化恒等式，但不同历史公告的余额并不自动具有同一发布版本可比性。本轮只保存算术诊断，未把它作为已经确认的“基数导致改善”标签。

## 6. 指数内部验证采用了现有等权参与代理

本轮复用了历史点时成分的等权20日上涨比例及5日变化，沿用原资料的成员覆盖条件，并整体滞后一交易日。445个周度点中361个有可用代理；2026年缺少该代理时保持缺失。**没有获得本轮全历史行业权重贡献，因此不能回答每月哪些行业贡献了多少指数收益。**

例如，2024-09-20观察使用9月19日的成员信息，过去20日上涨比例约27.7%；9月27日观察使用9月26日信息，该比例约81.8%。这说明内部参与确有扩散，但不能证明由M1经营传导造成，更不能据此倒选买点。

## 7. 复算范围、修正和缺口

- 3,476条日线；104个月公告：84个月旧口径、20个月新口径。E0后20日成熟103个月，E1成熟102个月。
- 445个固定周度点全部保留：357个使用旧口径、82个使用新口径、6个尚无本样本内已发布宏观。441个有20日行情标签，其中435个同时具有宏观背景。
- 原HTML的1,228个显示数值按其显示精度复算一致，观察/入场日期同步核对。原始输入与104份原文共107份文件与旧索引相符。
- 主分析严格要求`definition_version`相同，因此2022-12、2023-01、2023-02的三月货币变化缺失；附件的旧训练口径复刻字段同时保留，不覆盖原附件。
- 2026年8月公告于2026-09-14发布，晚于本轮行情截止；观察及后续标签均缺失，不能沿用9月11日收盘伪造9月14日的完整快照。
- 输入仍是重建的官方发布时间和今天保存的历史页面，未证明网页历史首版不可变；没有重新认证全交易日历或完整分红登记/到账。
- 毛收益使用固定份额的除息日权益近似，未扣费用、未模拟整手、订单、现金冲突、T+1或盘中最低价。它们不是策略收益。窗口未成熟保持缺失，未填零。

## 8. 本轮完成后的判断与下一步边界

本轮完成的是观察工作，整体夏普1.2账户目标仍未证明。旧口径这组固定表示不进入仓位设计；也不调整窗口、阈值或反转方向救援结果。新口径的正关联保留为待验证线索，不能用增加文章数量代替新的宏观发布周期。

后续最有价值的材料是**当时可得的活期/定期及资管迁移证据**，用于区分需求活动、制度迁移和股市反馈，再提出不同的后续可验证事实。若仍只有M1/M2和企业总存款，不把原因字段强行填满；若控制价格后仍无增量，就停留在环境观察。内部代理只承担传导核对，不另开行业轮动或择时系统。

按最新项目交付偏好直接保留可读结果，本轮未生成ZIP。原有失败、冻结研究和自动任务不作修改。

主要文件：[可筛选月周观察台](观察台.html)、[104个月完整观察](results/104个月_完整观察.csv)、[445周完整观察](results/固定周度_完整观察.csv)、[104个月原因证据](results/104个月_原因证据台账.csv)、[全部固定比较](results/主20日_固定比较与区间.csv)、[失败案例](results/典型与失败案例.csv)、[冻结方案](protocol.json)。
"""
    (STUDY / "首轮观察结论.md").write_text(text, encoding="utf-8")


def write_html(md, wd) -> None:
    records = []
    cause = pd.read_csv(RESULT / "104个月_原因证据台账.csv").set_index("stat_month")
    cols = ["origin", "origin_id", "stat_month", "training_regime", "observation_date", "available_at_upper_bound", "spread_pp", "delta3_m1_pp", "delta3_m2_pp", "delta3_spread_pp", "arithmetic_group", "volatility_group", "past_return20", "past_return60", "rv20", "downside20", "downside_change5", "internal_breadth20", "internal_breadth_change5", "macro_status", "source_url"]
    cols += [f"E{d}_{h}_{k}" for d in [0, 1] for h in [5, 20, 60] for k in ["return", "worst_close", "drawdown", "entry_date", "status"]]
    for frame in [md, wd]:
        for row in frame[cols].to_dict("records"):
            mon = row["stat_month"]
            if mon in cause.index:
                row["flow_period"] = cause.at[mon, "flow_period"]
                row["corporate_deposit_flow_100m"] = cause.at[mon, "corporate_deposit_flow_100m"]
                row["fiscal_deposit_flow_100m"] = cause.at[mon, "fiscal_deposit_flow_100m"]
                row["cause"] = cause.at[mon, "economic_cause"]
            records.append({k: None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()})
    payload = json.dumps(records, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>510300 · 宏观与波动观察 V2</title><style>
:root{color-scheme:light;--ink:#18343d;--muted:#596d72;--line:#d9e0df;--teal:#236777;--rust:#a85e38}*{box-sizing:border-box}body{margin:0;background:#f6f5f0;color:var(--ink);font:15px/1.7 "Microsoft YaHei",system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:42px 30px}header{border-bottom:2px solid var(--ink);padding-bottom:22px}.eyebrow{letter-spacing:.12em;color:var(--teal);font-size:12px;font-weight:700}h1{font-size:32px;line-height:1.4;margin:9px 0 12px}h2{font-size:22px;margin:0 0 12px}.lead{font-size:18px;max-width:1000px;margin:0}.meta{color:var(--muted);margin:14px 0 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:26px 0}.card{padding:21px;background:#fff;border:1px solid var(--line);border-top:4px solid var(--teal)}.card strong{display:block;font-size:30px;line-height:1.5}.card p{margin:4px 0}.card small{color:var(--muted)}.note{padding:17px 20px;border-left:4px solid var(--rust);background:#efe9dd;margin:20px 0}.links{display:flex;gap:16px;flex-wrap:wrap;margin:15px 0}a{color:var(--teal);text-underline-offset:3px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0;align-items:end}label{font-size:12px;color:var(--muted)}select,input{display:block;margin-top:3px;padding:9px;border:1px solid #bbc8c9;border-radius:3px;background:white;color:var(--ink);font:14px "Microsoft YaHei"}.count{margin-left:auto;color:var(--muted);align-self:center}.scroll{overflow:auto;max-height:620px;border:1px solid var(--line);background:white}table{border-collapse:collapse;width:max-content;min-width:100%;font-size:13px}th,td{padding:10px 13px;text-align:right;border-bottom:1px solid #e2e8e6;white-space:nowrap}th{position:sticky;top:0;background:var(--ink);color:white;font-weight:500;z-index:1}th:first-child,td:first-child{text-align:left}tr:hover td{background:#edf5f4}.neg{color:#a24438}.pos{color:#20664e}.missing{color:#859393}details{margin-top:28px;border-top:1px solid var(--line);padding-top:16px}summary{cursor:pointer;font-size:19px;font-weight:600}.figure{display:block;width:100%;max-width:1200px;margin:20px auto}footer{margin-top:32px;color:var(--muted);font-size:13px}@media(max-width:760px){main{padding:24px 15px}.cards{grid-template-columns:1fr}h1{font-size:26px}.count{margin-left:0}}</style></head>
<body><main><header><span class="eyebrow">510300 / 宏观与波动 / V2 首轮观察</span><h1>货币变化以后，还剩多少行情？</h1><p class="lead">旧口径未显示稳定的后续增量。新口径有正向线索，仍缺足够发布周期和相似对照。</p><p class="meta">行情截至 2026-09-11 · 104个月公告 · 445个固定周度观察点 · 全部为历史毛收益观察</p></header>
<div class="cards"><article class="card"><span>旧口径：剪刀差变化的时序关系</span><strong>0.332 → 0.091</strong><p>与公告前60日 / 公告后20日收益的秩相关</p><small>提示与已发生行情联系更强，尚不是因果判断。</small></article><article class="card"><span>控制已有价格与波动后</span><strong>−1.15 个百分点</strong><p>改善周的20日收益相对历史近邻差异</p><small>106周、28个目标发布周期；描述性区间跨零。</small></article><article class="card"><span>原因证据完成度</span><strong>104 / 104</strong><p>全部公告完成存贷款分项与原句提取</p><small>唯一经济原因仍未识别；当月与累计分开保留。</small></article></div>
<div class="note">E0：观察快照后的下一交易日开盘；E1：再延后一交易日。主期限固定20日，5日和60日为辅助。未成熟与资料不足显示“缺失”。本页不计算仓位、费用后账户收益或夏普。</div>
<nav class="links"><a href="首轮观察结论.md">阅读完整结论</a><a href="results/104个月_完整观察.csv">月度完整CSV</a><a href="results/固定周度_完整观察.csv">周度完整CSV</a><a href="results/104个月_原因证据台账.csv">原因证据CSV</a><a href="results/典型与失败案例.csv">失败与典型案例</a><a href="evidence/机制来源核对.md">官方机制证据</a></nav>
<section><h2>月度与周度观察底表</h2><div class="controls">
<label>观察原点<select id="origin"><option value="月度">月度公告</option><option value="周度">固定周度</option></select></label>
<label>M1口径<select id="regime"><option value="">全部</option value="M1_OLD_M2_MMF2018">旧口径</option><option value="M1_NEW2025">新口径</option></select></label>
<label>参考开盘<select id="delay"><option value="0">E0</option><option value="1">E1 延迟一天</option></select></label>
<label>后续交易日<select id="horizon"><option value="20">20日 / 主期限</option><option value="5">5日 / 辅助</option><option value="60">60日 / 辅助</option></select></label>
<label>筛选年份、日期或过程<input id="filter" placeholder="例如：2024、下行缓和、M1侧"></label><span class="count" id="count"></span></div>
<div class="scroll"><table><thead><tr><th>观察点</th><th>货币所属月</th><th>口径</th><th>观察收盘</th><th>剪刀差/点</th><th>三月变化/点</th><th>算术来源</th><th>此前20日</th><th>RV20</th><th>下行RMS20</th><th>下行5日变化</th><th>随后毛收益</th><th>途中最差收盘</th><th>路径回撤</th><th>内部上涨广度</th><th>企业存款增量/亿元</th><th>报告期</th><th>经济原因</th><th>来源</th></tr></thead><tbody id="body"></tbody></table></div></section>
<details open><summary>主比较：保留延迟一天及不确定性</summary><img class="figure" src="figures/有限条件比较.png" alt="旧口径下三项比较的E0、E1结果及时间块区间，均跨过零"></details>
<details><summary>先后顺序：全部月度散点</summary><img class="figure" src="figures/时序与后续分布.png" alt="旧、新口径剪刀差变化与公告前60日及随后20日收益散点图"></details>
<footer>严格同统计定义的三月变化在2022-12、2023-01、2023-02缺失；原附件的旧训练口径数值另存复刻字段。内部指标是滞后一交易日的历史成员等权代理，不是行业权重贡献。资料为历史重建版本，未证明首版不可变；收益使用除息权益近似，未模拟完整登记到账或交易约束。周度窗口和发布周期相互关联，不按独立周数推断显著性。</footer>
</main><script>const data=PAYLOAD;
const el=id=>document.getElementById(id);const escapeHtml=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function number(v,percentage=false){if(v===null||v===undefined)return '<span class="missing">缺失</span>';let x=percentage?v*100:v;return '<span class="'+(x<0?'neg':'pos')+'">'+x.toFixed(2)+(percentage?'%':'')+'</span>'}
function draw(){const origin=el('origin').value,regime=el('regime').value,q=el('filter').value.trim(),prefix='E'+el('delay').value+'_'+el('horizon').value;const rows=data.filter(r=>r.origin===origin&&(!regime||r.training_regime===regime)&&(!q||[r.origin_id,r.stat_month,r.arithmetic_group,r.volatility_group].join(' ').includes(q)));el('count').textContent=rows.length+' 个观察点 · '+rows.filter(r=>r[prefix+'_return']!==null).length+' 个成熟行情标签';el('body').innerHTML=rows.map(r=>'<tr>'+[escapeHtml(r.origin_id),escapeHtml(r.stat_month||'缺失'),r.training_regime==='M1_NEW2025'?'新':(r.training_regime?'旧':'缺失'),escapeHtml(r.observation_date||'缺失'),number(r.spread_pp),number(r.delta3_spread_pp),escapeHtml(r.arithmetic_group),number(r.past_return20,true),number(r.rv20,true),number(r.downside20,true),number(r.downside_change5,true),number(r[prefix+'_return'],true),number(r[prefix+'_worst_close'],true),number(r[prefix+'_drawdown'],true),number(r.internal_breadth20,true),number(r.corporate_deposit_flow_100m),r.flow_period==='YTD'?'年初累计':(r.flow_period==='MONTH'?'当月':'缺失'),escapeHtml(r.cause||'缺失'),r.source_url?'<a target="_blank" rel="noopener" href="'+escapeHtml(r.source_url)+'">官方</a>':'缺失'].map(v=>'<td>'+v+'</td>').join('')+'</tr>').join('')};['origin','regime','delay','horizon'].forEach(id=>el(id).addEventListener('change',draw));el('filter').addEventListener('input',draw);draw();</script></body></html>"""
    old = md.loc[md.training_regime == OLD]
    before = old.delta3_spread_pp.corr(old.pre_publication_return60, method="spearman")
    after = old.delta3_spread_pp.corr(old.E0_20_return, method="spearman")
    page = page.replace("0.332 → 0.091", f"{before:.3f} → {after:.3f}").replace("原因证据完成度", "公告事实覆盖")
    (STUDY / "观察台.html").write_text(page.replace("PAYLOAD", payload), encoding="utf-8")


def main() -> None:
    md = pd.read_csv(RESULT / "104个月_完整观察.csv")
    wd = pd.read_csv(RESULT / "固定周度_完整观察.csv")
    g = pd.read_csv(RESULT / "全部分组分布.csv")
    c = pd.read_csv(RESULT / "全部连续关联.csv")
    u = pd.read_csv(RESULT / "主20日_固定比较与区间.csv")
    matched = pd.read_csv(RESULT / "历史近邻_全部目标与缺口.csv")
    figures(md, u)
    write_report(md, wd, g, c, u, matched)
    write_html(md, wd)
    for script in ["macro_volatility_observation_v2_run1.py", "publish_macro_volatility_observation_v2_run1.py"]:
        source = ROOT / "research" / script
        if source.exists():
            shutil.copy2(source, STUDY / "code" / script)
    print("中文报告、两张研究图和可筛选观察台已生成。")


if __name__ == "__main__":
    main()
