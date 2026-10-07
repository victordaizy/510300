"""交付同月份可比性、条件关联及完整失败案例。"""
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
OUT = ROOT / "reports/research/510300_balance_source_comparability_v14"
REPORT = "第十四轮_余额来源的季节可比性与信贷结构.md"
OLD = "M1_OLD_M2_MMF2018"
GROUPS = ["改善_当期相对余额同向", "改善_当期相对余额未同向"]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def link(path, label):
    return f"[{label}](<{path.as_posix()}>)"


def publish():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮完成记录已保存，不覆盖。")
    build = json.loads((OUT / "results/build_receipt.json").read_text(encoding="utf-8"))
    assert build["status"] == "COMPLETED_CALENDAR_SUPPORT_AND_CONDITIONAL_ASSOCIATION"
    a = pd.read_csv(OUT / "results/104个月_完整输入与原结果.csv")
    groups = pd.read_csv(OUT / "results/原分组与同月可比结果.csv")
    coef = pd.read_csv(OUT / "results/全部条件投影系数_不筛选.csv")
    year = pd.read_csv(OUT / "results/逐年留出_系数敏感性.csv")
    monthly = pd.read_csv(OUT / "results/逐月份共同支持.csv")
    pairs = pd.read_csv(OUT / "results/全部同月跨年配对及背景差.csv")
    seasonality = pd.read_csv(OUT / "results/货币分解项的月份结构.csv")
    def difference(entry, scope):
        return float(groups.loc[groups.regime.eq(OLD) & groups.entry.eq(entry) & groups.scope.eq(scope) & groups.group.eq("两组均值差"), "return_mean_percent"].iloc[0])
    all0, same0 = [difference("E0", x) for x in ["ALL_ORIGINAL", "SAME_CALENDAR_MONTH_COMMON_SUPPORT"]]
    all1, same1 = [difference("E1", x) for x in ["ALL_ORIGINAL", "SAME_CALENDAR_MONTH_COMMON_SUPPORT"]]
    short = coef[coef.regime.eq(OLD) & coef.entry.eq("E0") & coef.feature.eq("current_component")].set_index("stage")
    stages = ["A_RAW", "B_CALENDAR", "C_CONTEXT"]
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11,
                        "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.9, 8.2), facecolor="#f8f9fb")
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=.08, right=.96, bottom=.29, top=.74, wspace=.51)
    fig.text(.045, .93, "余额来源的收益差，先过可比性这一关", fontsize=23, weight="bold", color="#193a50")
    fig.text(.045, .871, "同月对照只剩 9 个观察；同时考虑信贷、订单和事前价格后，正向关联并不稳定。", fontsize=13, color="#456276")
    x, width = np.arange(2), .32
    for k, (label, vals, color) in enumerate([("原31个月", [all0, all1], "#a8b9c8"), ("同月可比9个月", [same0, same1], "#237e83")]):
        bars = axes[0].bar(x+(k-.5)*width, vals, width, color=color, label=label)
        for bar in bars:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.07, f"{bar.get_height():+.2f}", ha="center", fontsize=12)
    axes[0].set_xticks(x, ["E0：次日开盘", "E1：再延迟一天"])
    axes[0].set_ylim(0, 2.7)
    axes[0].set_ylabel("两组后20日平均毛收益之差，百分点")
    axes[0].set_title("当期余额同向组 − 未同向组", loc="left", fontsize=13, weight="bold", pad=20)
    axes[0].legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1.0))
    axes[0].grid(axis="y", alpha=.18)
    axes[0].set_axisbelow(True)
    labels = ["两项货币分解", "再考虑统计月份", "再考虑价格、信贷等"]
    colors = ["#8d9dac", "#547d98", "#237e83"]
    for i, stage in enumerate(stages):
        r = short.loc[stage]
        estimate = r.coefficient_return_pp_per_feature_unit
        axes[1].errorbar(estimate, i, xerr=[[estimate-r.low95_descriptive], [r.high95_descriptive-estimate]], fmt="o", color=colors[i], markersize=7, capsize=5, elinewidth=2)
        axes[1].text(estimate, i-.16, f"{estimate:+.2f}", ha="center", fontsize=11, color="#294659")
    axes[1].axvline(0, color="#a5b2bd", ls="--", lw=1)
    axes[1].set_yticks(range(3), labels)
    axes[1].set_ylim(2.55, -.65)
    axes[1].set_xlim(-1.65, 1.45)
    axes[1].set_xlabel("当期余额项每增1个对数百分点\n对应的后20日收益差，百分点")
    axes[1].set_title("同一57个月中的条件关联", loc="left", fontsize=13, weight="bold", pad=20)
    axes[1].grid(axis="x", alpha=.18)
    for ax in axes:
        ax.set_facecolor("#f8f9fb")
    fig.text(.045, .163, "左图：同月对照仅覆盖 4、8、11 月；未同向组的3个月全部来自2019年。两组仍存在年份和经济背景差异。", fontsize=10.5, color="#456276")
    fig.text(.045, .121, "右图：线段为6个月相关性修正后的95%描述区间，全部跨零；完整设计含25个参数，不能将样本内拟合当预测成绩。", fontsize=10.5, color="#456276")
    fig.text(.045, .079, "两个面板的样本与问题不同：左图复核原来的改善分组，右图使用完整背景的所有旧口径月份。", fontsize=10, color="#687d8c")
    fig.text(.045, .044, "历史探索，未扣交易费用；不证明季节导致了差额变化，也不支持因系数转负而反向交易。", fontsize=10, color="#687d8c")
    figure = OUT / "figures/余额来源均值差与可比性.png"
    fig.savefig(figure, dpi=240, facecolor=fig.get_facecolor())
    fig.savefig(figure.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
    old_month = monthly[monthly.regime.eq(OLD)]
    calendar_table = []
    for month in range(1, 13):
        count = [int(old_month.loc[old_month.calendar_month.eq(month) & old_month.group.eq(g), "count"].iloc[0]) for g in GROUPS]
        calendar_table.append(f"| {month}月 | {count[0]} | {count[1]} | {'有' if all(count) else '无'} |")
    coefficient_table = []
    for stage, label in zip(stages, labels):
        row0 = short.loc[stage]
        row1 = coef[coef.entry.eq("E1") & coef.feature.eq("current_component") & coef.stage.eq(stage)].iloc[0]
        coefficient_table.append(f"| {label} | {row0.coefficient_return_pp_per_feature_unit:+.3f} | [{row0.low95_descriptive:+.3f}, {row0.high95_descriptive:+.3f}] | {row1.coefficient_return_pp_per_feature_unit:+.3f} |")
    pairs_table = "\n".join(f"| {r.current_supported_month} | {r.base_only_month} | {r.E0_return_difference_pp:+.2f} | {r.E1_return_difference_pp:+.2f} |" for r in pairs.itertuples())
    q = a[a.stat_month.isin(["2020-08", "2022-08"])].set_index("stat_month")
    case_rows = []
    for month in q.index:
        r = q.loc[month]
        credit_total = r.loan_corporate_total_ytd_yoy_change_yi
        credit_parts = sum(r[f"loan_{key}_ytd_yoy_change_yi"] for key in ["corporate_long", "corporate_short", "bills"])
        case_rows.append({"stat_month": month, "corporate_total_yoy_change_yi": credit_total,
            "corporate_long_yoy_change_yi": r.loan_corporate_long_ytd_yoy_change_yi,
            "corporate_short_yoy_change_yi": r.loan_corporate_short_ytd_yoy_change_yi,
            "bills_yoy_change_yi": r.loan_bills_ytd_yoy_change_yi, "unallocated_difference_yi": credit_total-credit_parts,
            "scope": "年初累计新增贷款同区间跨原公告差额；保留未分配残差，不等于原公告直接给出的同版本同比。"})
    pd.DataFrame(case_rows).to_csv(OUT / "results/两次八月_企业信贷增量来源.csv", index=False, encoding="utf-8-sig")
    cases = pd.DataFrame(case_rows).set_index("stat_month")
    def case_row(label, field, digits=2, multiplier=1):
        return f"| {label} | {q.loc['2020-08', field]*multiplier:+.{digits}f} | {q.loc['2022-08', field]*multiplier:+.{digits}f} |"
    case_table = "\n".join([
        case_row("剪刀差三个月变化，百分点", "delta3_spread_pp", 1),
        case_row("当期M1相对M2余额项，对数百分点", "current_component", 3),
        case_row("年初累计新增企事业贷款同比增幅，%", "corporate_total_yoy_percent"),
        case_row("其中中长期同比多增，亿元", "loan_corporate_long_ytd_yoy_change_yi", 0),
        case_row("其中短期同比多增，亿元", "loan_corporate_short_ytd_yoy_change_yi", 0),
        case_row("其中票据融资同比多增，亿元", "loan_bills_ytd_yoy_change_yi", 0),
        f"| 企事业合计与上述三项之差，亿元 | {cases.loc['2020-08','unallocated_difference_yi']:+.0f} | {cases.loc['2022-08','unallocated_difference_yi']:+.0f} |",
        case_row("住户中长期同比多增，亿元", "loan_household_long_ytd_yoy_change_yi", 0),
        case_row("制造业新订单指数", "orders_first_release_value", 1),
        case_row("FDR007相对7天政策利率近20日均值，基点", "funding_gap_bp"),
        case_row("此前20日510300收益，%", "pre_return20_pp"),
        case_row("此前60日510300收益，%", "pre_return60_pp"),
        case_row("下行波动尺度，年化%", "v_downside20", 2, 100),
        f"| 下行窗口变化 | {q.loc['2020-08','downside_window_state']} | {q.loc['2022-08','downside_window_state']} |",
        case_row("E0后20日毛收益，%", "E0_20_return", 2, 100),
        case_row("E1后20日毛收益，%", "E1_20_return", 2, 100)])
    s = seasonality[seasonality.scope.eq("ALL_VALID_BALANCE")].set_index("component")
    leave = year[year.entry.eq("E0") & year.feature.eq("current_component")]
    v2 = link(ROOT / "reports/research/510300_macro_volatility_mechanisms_v2_run2/第二轮机制观察结论.md", "V2余额和基数分解")
    v4 = link(ROOT / "reports/research/510300_macro_transmission_context_v4/第四轮_立体传导与明确结论.md", "V4信贷同区间及多层背景")
    v12 = link(ROOT / "reports/research/510300_funding_quantity_price_bridge_v12/第十二轮_货币数量与融资价格为何可能背离.md", "V12资金定盘口径")
    report = f"""**本轮更明确的结论：当期余额支持的剪刀差改善，尚未表现出稳定的后续收益优势。此前约2.09个百分点的两组收益差缺乏充分同月对照；把信贷、需求、资金价格和先前股价一起考虑后，当期项与基数项都没有稳定方向。** 这不是说货币来源没有意义，而是它还不能单独完成从货币变化到510300可获得收益的推断。

上一轮V13已完成银行利润传导。本轮回到原104个月，只检验先前结果的季节可比性和多层条件关联；沿用原数据、定义、固定20日E0/E1结果，不设仓位，不寻找更好的切点。所有既有结果此前已经看过，新分析也属于探索，不是独立验证。

**原来的31个改善月，实际上只有9个能够找到同月份的另一组。** 沿用{v2}原分组，旧M1口径中17个月的当期相对余额同向、14个月未同向。所谓同向，仍只是三个月M1余额增幅超过M2，不等于企业活期、订单或真实投资改善。

| 统计月份 | 当期相对余额同向，月数 | 未同向，月数 | 两组同月均有观察 |
|---|---:|---:|---|
{chr(10).join(calendar_table)}

只有4月、8月、11月满足两组共同覆盖。同向组6个月来自2020、2022、2024年；未同向组3个月全部来自2019年。三个月份等权、同组同月份内各年等权后：

| 比较范围 | E0组间差，百分点 | E1组间差，百分点 |
|---|---:|---:|
| 原31个月，同向减未同向 | {all0:+.2f} | {all1:+.2f} |
| 同月份可比9个月，同向减未同向 | {same0:+.2f} | {same1:+.2f} |

E0同月对比的两组均值为+1.57%与+1.02%。从2.09缩到0.55是换成可比月份之后的描述，不能声称差额减少的某个百分比“由季节造成”；样本同时发生变化，年份和其他冲击仍不同。E1差额仍为2.07，也不能省略。E1同时改变入场与20日窗口末端，差异不能全解释成第一天跳空。

全部6对跨年同月比较都保留，三个2019年对照各被使用两次，不能算6次独立试验：

| 当期余额同向月份 | 未同向对照月份 | E0收益差，百分点 | E1收益差，百分点 |
|---|---|---:|---:|
{pairs_table}

新口径9个改善月份中，同向的只有2025年7月和8月，在未同向组里没有相同统计月份。因此同月比较为NO_COMMON_SUPPORT，不用旧口径或别的月份补齐。

**货币分解项本身有很强的月份结构。** 在旧口径78个有效余额分解月中，仅按统计月份分别取平均，能解释当期项约{s.loc['current_component','calendar_month_r2_in_sample']*100:.2f}%、基数项约{s.loc['base_component','calendar_month_r2_in_sample']*100:.2f}%的样本内变异。这是描述性拟合比例，不是春节或季节的因果贡献，也不是未来预测成绩。同月份不能消除春节在1月或2月、疫情、政策和信用周期的差别。

**连续变量分析同时放入价格、风险、信贷、订单和资金价格，没有继续筛小格子。** 固定用相同57个完整旧口径月份，依次观察：两项货币分解；再加12个月份效应；再加11项事前背景。背景包括此前20/60日收益、总波动及下行占比，企事业贷款总量、企业中长期和住户中长期的年初累计同比变化，新订单水平及变化，FDR007相对政策利率、FR007相对FDR007的近20日均值。

贷款同比以相同年初累计区间比较，分母是对应上年累计值；不是贷款余额增速。2018年缺少此前同比基数、2023年信贷统计范围断点、2022年部分住户分项和货币口径衔接缺失仍保留，未做插补。57个完整月份集中在2019、2020、2021、2022、2024年。新口径完整数据仅15个月，无法估计同一个25参数设计，保持NOT_ESTIMABLE_FULL_DESIGN，没有删控制项换取可估计。

下表是当期相对余额项每增加1个对数百分点，对应的后20日收益差，单位百分点。它描述样本内条件关联；两个货币项始终同时进入。

| 固定比较层次 | E0系数 | E0的95%描述区间 | E1系数 |
|---|---:|---|---:|
{chr(10).join(coefficient_table)}

完整背景下，基数项的E0系数为+0.259、描述区间[−1.718,+2.236]，E1系数+0.112，同样没有明确方向。逐一留出每个完整样本年份后，当期项的E0系数范围为[{leave.coefficient_return_pp_per_feature_unit.min():+.3f},{leave.coefficient_return_pp_per_feature_unit.max():+.3f}]，去掉2021年会转正；这里保留2021年，没有因它改变符号就删除。所有年份留出与E1结果见完整表。

区间按实际统计月距离，采用固定6个月Bartlett核修正相关性；跨越缺失年份的相邻行不会被误当相邻月份。区间只作描述，未作多重比较校正。完整设计57个月、25参数，只有32个残差自由度，相关变量之间也会分享信息。价格、订单和信贷可能处在传导链中，因此控制后的系数不代表货币的总因果效应；区间跨零也不证明经济作用恰好为零。更不能因为系数转负就反向交易。

![均值差与条件关联]({figure.as_posix()})

**一个更贴近经济过程的同月对照：2020年8月与2022年8月。** 它们来自上述全部同月比较，选择这两个例子解释结构，不新增收益检验，也不据结果更换分组。前者的金融数据在2020年9月11日16:00:31公布，后者在2022年9月9日16:00:40公布；按原快照与下一交易日开盘规则连接行情。

下表所有贷款多增和贷款增幅，指年初累计新增贷款与上年相同累计区间比较，不是8月单月增速；金额来自既有原公告累计重建，保留版本及舍入差异。

| 当时可以看到的环节 | 2020年8月数据公布时 | 2022年8月数据公布时 |
|---|---:|---:|
{case_table}

金融数据原文：[2020年8月央行报告]({q.loc['2020-08','loan_source_url']})、[2022年8月央行报告]({q.loc['2022-08','loan_source_url']})。累计重建沿既有上半年累计加7、8月原公告及同期比较，完整来源见{v4}；新订单来自[2020年8月统计局公告]({q.loc['2020-08','orders_source_url']})和[2022年8月统计局公告]({q.loc['2022-08','orders_source_url']})。定盘指标沿{v12}，不是全天质押回购加权利率。

两次年初累计新增企事业贷款都增长约三成，但增长内容不同：2020年中长期同比多增19057亿元；2022年中长期少增3340亿元，短贷和票据多增更多，住户中长期也明显少增。合计与列出的三项之间尚有−168、−2721亿元差额，原样列出，不把它强行归入某个贷款用途或改数配平。仅凭净额，不能证明同一笔票据被长期贷款置换，也不能确定借款企业的最终支出。

2022年资金价格较低、下行尺度下降且最近5日也同步缓和，但订单仍低于50，长期及住户信贷没有共同改善；随后E0亏6.34%、E1亏5.78%。2020年当时下行尺度没有下降，后来E0却涨2.44%。这两个结果不证明信贷结构唯一导致涨跌，但足以说明“当期余额支持+资金便宜+降波”仍会遗漏实体需求、信用期限和已有价格路径。新增贷款总量相似，也可能对应不同的经济过程。

**这轮改变了什么判断。** 原来的两组收益差不能再当作已经找到的余额来源优势；同月对照稀疏、年份集中，多层条件关联也不稳定。波动和货币仍有用于解释环境的信息，但不能把它们与订单、信贷简单计票。下一步有价值的连接应当是历史上新的信用或政策变化何时成为新增信息、有没有改变具体盈利和市场内部路径，以及消息公开后还剩多少收益；不会为保住本轮均值再换月份、删年份、调窗口或增加阈值。

本轮保留104个月与所有缺失，完成原分组重算、全部同月对照、固定背景下16次描述拟合（6次三层E0/E1、10次逐年留出）。没有新增预测模型、策略账户或交易动作；现有历史首版认证与独立验证限制仍在。目标保持进行中。

可复算文件：{link(OUT / 'results/原分组与同月可比结果.csv', '完整分组结果')}；{link(OUT / 'results/全部同月跨年配对及背景差.csv', '全部配对与背景差')}；{link(OUT / 'results/全部条件投影系数_不筛选.csv', '全部系数与描述区间')}；{link(OUT / 'results/逐年留出_系数敏感性.csv', '逐年敏感性')}；{link(OUT / 'results/104个月_完整输入与原结果.csv', '104个月数据与缺失')}；{link(OUT / 'protocol.json', '固定分析范围')}。
"""
    (OUT / REPORT).write_text(report, encoding="utf-8")
    core = short.loc["C_CONTEXT"]
    result = {"at": datetime.now().astimezone().isoformat(), "study_id": "510300_BALANCE_SOURCE_COMPARABILITY_V14",
        "status": "CALENDAR_COMPARABILITY_AND_CONTEXT_ASSOCIATION_COMPLETED", "continuation_classification": "PROGRESS", "goal_achieved": False,
        "old_original_group_months": 31, "old_common_support_months": 9, "common_calendar_months": [4, 8, 11], "common_controls_years": [2019],
        "old_group_difference_pp": {"E0_original": all0, "E0_same_month": same0, "E1_original": all1, "E1_same_month": same1},
        "full_context_rows_old": build["full_context_rows_old"], "full_context_rows_new": build["full_context_rows_new"],
        "current_component_E0_full_context": {"coefficient": float(core.coefficient_return_pp_per_feature_unit), "low95": float(core.low95_descriptive), "high95": float(core.high95_descriptive)},
        "new_regime_status": "NO_COMMON_CALENDAR_SUPPORT_AND_NOT_ESTIMABLE_FULL_CONTEXT_DESIGN", "descriptive_fits": 16,
        "new_prediction_models": 0, "new_accounts": 0, "independent_validation": False,
        "conclusion": "既有余额来源收益差缺乏足够同月对照；在固定多层背景下未显示稳定方向，不能升级成入场优势。",
        "report": REPORT, "figure": str(figure.relative_to(OUT))}
    save("result.json", result)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    save("publication_receipt.json", {"at": result["at"], "report_sha256": digest(OUT / REPORT), "figure_sha256": digest(figure), "status": "AWAITING_VISUAL_AND_SAVED_CHECK"})
    print(json.dumps({"报告": str(OUT / REPORT), "图表": str(figure), "当期项系数": result["current_component_E0_full_context"]}, ensure_ascii=False))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    publish()
