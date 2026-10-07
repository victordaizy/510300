"""发布货币结构、信贷对应项及510300定价位置的研究结果。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.money_balance_sources_v8 import OUT, now, save, digest

plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def frame_md(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def fmt(value, digits=2):
    return f"{value:+,.{digits}f}"


def plot(intervals, a3):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.suptitle("2025年8月：剪刀差改善，货币结构发生了什么变化？", fontsize=21, fontweight="bold")
    month = intervals[(intervals.interval == "单月") & intervals.end_month_end.isin(["2025-06", "2025-07", "2025-08"])].sort_values("end_month_end")
    x = np.arange(3)
    ax = axes[0, 0]
    bars1 = ax.bar(x - .18, month.m1_new_delta / 10000, width=.34, color="#2563a8", label="M1当月余额变化")
    bars2 = ax.bar(x + .18, month.m2_delta / 10000, width=.34, color="#d29b32", label="M2当月余额变化")
    ax.bar_label(bars1, fmt="%+.2f", padding=4, fontsize=10)
    ax.bar_label(bars2, fmt="%+.2f", padding=4, fontsize=10)
    ax.set_xticks(x, ["6月", "7月", "8月"])
    ax.set_ylim(-4.8, 6.5)
    ax.axhline(0, color="#333333", lw=.7)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_ylabel("万亿元")
    ax.set_title("同比改善，可与当月相对活化减弱同时发生", loc="left", fontsize=13, pad=12)
    ax.text(.01, .015, "8月M1环比 +0.15%，M2环比 +0.62%\n未作季节调整；环比不能单独判定需求强弱", transform=ax.transAxes, fontsize=10, va="bottom")

    ax = axes[0, 1]
    labels = ["银行等对政府债权", "央行对政府债权", "减：政府在央行存款", "对政府净债权"]
    amounts = [a3.delta_odcs_gov_claims_derived / 10000, a3.delta_cb_gov_claims / 10000, -a3.delta_cb_gov_deposits / 10000, a3.delta_government_net / 10000]
    starts, cumulative = [], 0
    for i, value in enumerate(amounts):
        starts.append(0 if i == 3 else cumulative)
        if i < 3:
            cumulative += value
    for i, (v, b) in enumerate(zip(amounts, starts)):
        ax.bar(i, v, bottom=b, width=.65, color=("#1b6d67" if i == 3 else "#2563a8" if v > 0 else "#b05a43"))
        ax.text(i, max(b, b + v) + .13, f"{v:+.2f}", ha="center", fontsize=12)
    ax.set_xticks(range(4), ["银行等\n政府债权", "央行\n政府债权", "政府存款\n增加的抵减", "政府净债权\n变化合计"])
    ax.set_ylim(0, 4.8)
    ax.set_ylabel("万亿元")
    ax.set_title("6—8月政府净债权增加，并非政府存款净释放", loc="left", fontsize=13, pad=12)
    ax.text(.01, .95, "政府在央行存款增加0.53万亿元", transform=ax.transAxes, va="top", fontsize=10)

    ax = axes[1, 0]
    left_names = ["对政府净债权", "对非金融部门债权", "其他净项下降", "国外净资产", "对其他金融部门债权", "非M2存款、债券和资本"]
    left_values = [a3.delta_government_net, a3.delta_nonfinancial_claims, -a3.delta_other_net, a3.delta_nfa,
                   a3.delta_other_financial_claims, -a3.delta_excluded_deposits - a3.delta_bonds - a3.delta_capital]
    horizontal(ax, left_names, np.array(left_values) / 10000)
    ax.set_title("6—8月M2增加6.20万亿元：资产负债对应项", loc="left", fontsize=13, pad=12)
    ax.set_xlabel("对M2余额变化的算术贡献，万亿元")

    ax = axes[1, 1]
    corporate = a3.delta_corporate_demand + a3.delta_corporate_time
    other = a3.m2_delta - corporate - a3.delta_personal_total - a3.delta_odcs_financial_deposits_m2
    names = ["其他金融性公司存款", "个人存款", "单位存款", "现金、备付金及其余项"]
    vals = [a3.delta_odcs_financial_deposits_m2, a3.delta_personal_total, corporate, other]
    horizontal(ax, names, np.array(vals) / 10000)
    ax.set_title("同一笔6—8月M2增加：负债端的持有结构", loc="left", fontsize=13, pad=12)
    ax.set_xlabel("余额变化，万亿元；仅计入M2部分")
    fig.supxlabel("资料：央行原表的事后取得版本，2024回溯M1仅作可比分析。两种M2分解不能相加；净余额不等于账户迁移或股票净流入。", fontsize=10)
    target = OUT / "figures/货币结构与银行信用_2025年8月.png"
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)
    return target


def horizontal(ax, labels, values):
    y = np.arange(len(values))
    ax.barh(y, values, color=["#2563a8" if v >= 0 else "#b05a43" for v in values], height=.57)
    ax.set_yticks(y, labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", lw=.6)
    ax.set_xlim(min(-1.0, min(values) - .5), max(values) + .8)
    for i, v in enumerate(values):
        ax.text(v + (.06 if v >= 0 else -.06), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=11)


def run():
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "PASS_SAVED_BALANCE_IDENTITIES_AND_SOURCE_ANCHORS"
    data = pd.read_csv(OUT / "results/20个月_货币与银行资产负债原金额.csv").set_index("stat_month")
    intervals = pd.read_csv(OUT / "results/完整窗口_单月三个月年内余额增量及对应项.csv")
    a3 = intervals[(intervals.interval == "三个月") & (intervals.end_month_end == "2025-08")].iloc[0]
    a1 = intervals[(intervals.interval == "单月") & (intervals.end_month_end == "2025-08")].iloc[0]
    cases = pd.read_csv(OUT / "results/四个原病例_信息位置波动与原后续路径.csv")
    figure = plot(intervals, a3)
    month_rows = []
    for row in intervals[(intervals.interval == "单月") & intervals.end_month_end.isin(["2025-06", "2025-07", "2025-08"])].itertuples():
        month_rows.append([row.end_month_end, fmt(row.m1_new_delta), fmt(row.m2_delta), fmt(row.delta_corporate_demand),
                           fmt(row.delta_personal_demand_derived), fmt(row.delta_odcs_financial_deposits_m2)])
    m1_rows = []
    for name, field in [("流通中货币", "m0"), ("单位活期存款", "corporate_demand"), ("个人活期存款（跨表推导）", "personal_demand_derived"), ("支付机构客户备付金", "payment_reserves")]:
        now_delta = data.loc["2025-08", field] - data.loc["2025-05", field]
        previous_delta = data.loc["2024-08", field] - data.loc["2024-05", field]
        m1_rows.append([name, fmt(now_delta), fmt(previous_delta)])
    counterpart_rows = []
    for name, field, sign in [("国外净资产", "nfa", 1), ("对政府净债权", "government_net", 1), ("对非金融部门债权", "nonfinancial_claims", 1),
                              ("对其他金融部门债权", "other_financial_claims", 1), ("不纳入M2的存款", "excluded_deposits", -1),
                              ("债券", "bonds", -1), ("实收资本", "capital", -1), ("其他净项", "other_net", -1)]:
        counterpart_rows.append([name, fmt(sign * a3["delta_" + field]), fmt(sign * a3["previous_year_delta_" + field]),
                                 fmt(sign * a3["yoy_change_of_delta_" + field])])
    market_rows = []
    for row in cases.itertuples():
        market_rows.append([row.stat_month, row.published_at[:10], fmt(row.delta3_spread_pp, 1),
                            fmt(row.past_return60 * 100) + "%", f"{row.v_upside20*100:.2f}% / {row.v_downside20*100:.2f}%",
                            f"{row.orders_first_release_value:.1f}", fmt(row.E0_20_return * 100) + "%", fmt(row.E1_20_return * 100) + "%"])
    case_path = ROOT / "reports/research/510300_information_change_transmission_v6/results/五个固定病例_新信息与传导全表.csv"
    expectation = pd.read_csv(case_path).query("stat_month == '2025-08'").iloc[0]
    save(OUT / "inputs/既有预期比较的引用收据.json", {"at": now(), "source": case_path.relative_to(ROOT).as_posix(), "sha256": digest(case_path),
        "forecast_date": expectation.expectation_forecast_date, "survey_source": expectation.expectation_forecast_source,
        "spread_surprise_proxy_pp": float(expectation.expectation_spread_surprise_proxy_pp),
        "interpretation": "差值来自两项调查汇总预期；-0.1个百分点落在四项一位小数显示值合计±0.2的舍入范围内，不判为显著负面意外。"})
    bp = a3.m1_m2_base_relative_log_pp
    cp = a3.m1_m2_current_relative_log_pp
    result = {
        "at": now(), "study_id": "510300_MONEY_BALANCE_TRANSMISSION_V8", "status": "COMPLETED_CAUSAL_ACCOUNTING_DIAGNOSTIC_RETURN_EDGE_UNPROVEN",
        "m1_august_month_growth_percent": float(a1.m1_current_growth_pct), "m2_august_month_growth_percent": float(a1.m2_current_growth_pct),
        "august_m1_change_yi": float(a1.m1_new_delta), "august_m2_change_yi": float(a1.m2_delta),
        "august_other_financial_deposits_change_yi": float(a1.delta_odcs_financial_deposits_m2),
        "august_other_financial_deposits_share_of_m2_delta": float(a1.delta_odcs_financial_deposits_m2 / a1.m2_delta),
        "may_to_august_government_net_change_yi": float(a3.delta_government_net),
        "may_to_august_government_deposits_change_yi": float(a3.delta_cb_gov_deposits),
        "may_to_august_nonfinancial_claims_change_yi": float(a3.delta_nonfinancial_claims),
        "may_to_august_nonfinancial_claims_change_less_than_prior_year_yi": float(a3.yoy_change_of_delta_nonfinancial_claims),
        "may_to_august_other_net_signed_contribution_yi": float(-a3.delta_other_net),
        "may_to_august_m1_m2_current_relative_log_pp": float(cp), "may_to_august_m1_m2_base_relative_log_pp": float(bp),
        "may_to_august_base_share_in_log_change": float(bp / (bp + cp)),
        "balance_months": 20, "intervals": 35, "new_models": 0, "new_accounts": 0, "goal_achieved": False,
        "strongest_conclusion": "2025年8月剪刀差改善与政府、金融机构资金结构变化并存；有单位活期修复，但不能据此确认实体信用和经营需求同步加速。",
        "unproven": "没有证明这些结构变化能在当时价格、波动与预期之外，提高510300后续方向或成本后收益的可靠性。"
    }
    save(OUT / "result.json", result)
    report = f"""# 第八轮：剪刀差背后的银行信用、政府存款与金融机构资金

**本轮最明确的结论是：2025年8月的剪刀差改善，包含单位活期修复、去年同期低基数，以及政府和金融机构资产负债变化；它不能被直接解释成企业与居民同时扩大信用、经营需求全面加速。** 对510300，要继续区分盈利预期变化与金融条件引起的估值变化，并检查价格已经反映了多少。

本轮固定观察2024年1月至2025年8月全部20个月，保留19个可算单月、2025年8个三个月窗口和8个年内窗口，共35组余额差额。2025年9月之后数据留在原PDF，不进入本轮计算。所用年度表为2025年11—12月网址下、此次取得的历史版本；没有把它们视为2024—2025年原观察日已经取得的文件。以下金额如无特别注明，单位为亿元。

![货币结构与银行信用](figures/货币结构与银行信用_2025年8月.png)

**一、M1同比提高，不代表当月活期相对M2继续提高。**

2025年8月原公告M1同比6.0%、M2同比8.8%，剪刀差为-2.8个百分点，三个月改善2.8个百分点。但8月余额中，M1仅增加{a1.m1_new_delta:,.2f}，环比{a1.m1_current_growth_pct:.2f}%；M2增加{a1.m2_delta:,.2f}，环比{a1.m2_current_growth_pct:.2f}%。**当月M1/M2余额之比反而下降。** 这是同比、环比、结构比例三个不同问题，不表示同比数据失真。

{frame_md(["统计月", "M1当月变化", "M2当月变化", "单位活期变化", "个人活期变化（推导）", "其他金融性公司存款变化（计入M2部分）"], month_rows)}

8月单位活期增加2,923.10，个人活期减少2,432.88，现金增加557.49，备付金增加621.08，合计M1增加1,668.78。个人定期增加3,484.86，个人存款合计仍增加1,051.98。**可以说个人存款净变化更偏定期，不能说已观测到2,432.88亿元从活期逐笔转入定期。** 新存款、提款、企业和居民之间支付等都可能同时发生。

6月、7月与8月必须连着看，季末增加和随后回落很明显。本轮没有季节调整，也没有用原始环比下降判定需求下降。所有20个月和35个窗口已经保存，表内没有按后续涨跌删月份。

**二、三个月的改善确有当期修复，但同比基数仍占主要部分。**

2025年5月末至8月末，M1增加{a3.m1_new_delta:,.2f}，其中单位活期增加{a3.delta_corporate_demand:,.2f}、个人活期增加{a3.delta_personal_demand_derived:,.2f}。单位定期减少{abs(a3.delta_corporate_time):,.2f}，所以单位存款的活期结构确实改善；这一正面事实保留。

{frame_md(["新M1分项", "2025年5月末至8月末变化", "2024年同期变化（回溯口径）"], m1_rows)}

采用同一原表的可比余额精确分解，三个月M1相对M2的**对数同比变化**合计{cp+bp:.6f}个对数百分点，其中：当期相对余额变化{cp:+.6f}，去年同期基数项{bp:+.6f}。基数项占该对数变化的{100*bp/(bp+cp):.2f}%。去年同期单位活期减少19,014.23，是M1基数部分的重要金额来源；表格本身没有唯一识别这些存款下降的行为原因。

这里不能把{cp+bp:.6f}个对数百分点与原公告普通剪刀差改善2.8个百分点直接相加或混称。按本轮余额算出的同比与原公告一位小数同比，8个月均处在显示精度的舍入范围内；没有用更精细余额覆盖原发布数值。第二轮使用原公布余额与增速得到的“基数/修订残差”仍保留，本轮同表分解也不能把舍入差异都称作统计修订。

个人活期的推导有两种：2025年用总活期减单位活期，并用个人存款减个人定期交叉核对；2024年用2025表脚注的新M1回溯值减现金、单位活期和央行备付金。后者回溯总额只精确到1亿元，推导值不是高精度的原披露个人活期。2024旧M1单独保存，未替换当年历史信息集。

**三、政府融资、政府净债权、财政资金释放必须分开。**

央行存款性公司概览中：政府净债权＝其他存款性公司对政府债权＋央行对政府债权－政府在央行的存款。编制方法见[国家统计局说明](https://www.stats.gov.cn/zs/tjws/zytjzbqs/hbgyl/202410/t20241025_1957180.html)。这里的政府存款是货币当局表的特定项目，不冒充全部金融机构财政存款。

2025年6—8月的对应金额是：

| 项目 | 余额变化或净债权贡献 |
|---|---:|
| 其他存款性公司对政府债权增加 | {fmt(a3.delta_odcs_gov_claims_derived)} |
| 央行对政府债权减少 | {fmt(a3.delta_cb_gov_claims)} |
| 政府在央行存款增加，对净债权作抵减 | {fmt(-a3.delta_cb_gov_deposits)} |
| 政府净债权增加合计 | {fmt(a3.delta_government_net)} |

因此，约3.05万亿元政府净债权增加，主要伴随银行等机构对政府债权增加；同期政府在央行存款净增加0.53万亿元。**这不是该期间政府存款净释放的证据，也不否认期间内发生了财政支出。** 净余额不能还原税收入库、发债、支出和偿债各自的流水，更不能直接指定有多少进入某家企业、偿还了欠款或创造了新订单。

第四轮已核对的“2025年1—8月政府债券占社融累计同比多增的99.36%”，分母是社融同比多增4.66万亿元；本轮3.05万亿元是6—8月存款性公司政府净债权余额差。期间、统计范围、分母与是否扣除政府存款都不同，两者不能当作同一个比例，更不能把99.36%解释成M2或股市资金的政府贡献。

**四、M2扩大，对应实体债权增长、政府债权增长及其他项目，全部列出才看得清。**

M2＝国外净资产＋政府净债权＋非金融部门债权＋其他金融部门债权－不纳入M2的存款－债券－实收资本－其他净项。以下为按此恒等式计算的有符号贡献，正值支持M2余额增加、负值抵减；它们是账面对应关系，不是各变量独立造成M2的因果估计。

{frame_md(["对应项目", "2025年6—8月对M2变化贡献", "2024年同期贡献", "本期比去年同期变化"], counterpart_rows)}

各项含0.01亿元展示舍入差合计，2025年6—8月M2增加{a3.m2_delta:,.2f}，去年同期增加{a3.previous_year_m2_delta:,.2f}。政府净债权多增{a3.yoy_change_of_delta_government_net:,.2f}；非金融部门债权增加{a3.delta_nonfinancial_claims:,.2f}，**比去年同期少增{abs(a3.yoy_change_of_delta_nonfinancial_claims):,.2f}**。非金融部门包括非金融机构及其他居民部门，不能直接称为全部企业贷款；债权也不局限于贷款。

2025年这段非金融债权增加中，其他存款性公司对非金融机构债权增加20,375.40，对其他居民部门债权增加585.89。它没有显示居民信用与企业信用同强度扩张。与原发布报告中企业、居民中长期贷款同比少增的证据方向一致，但两表机构范围和统计对象不同，本轮没有把余额差替换成贷款净投放。

还必须保留“其他净项”：其下降对本期M2余额增加的算术贡献为{(-a3.delta_other_net):,.2f}，比去年同期的贡献多{(-a3.yoy_change_of_delta_other_net):,.2f}。该项包含合并冲销差和未单列项目，足以影响解释，却没有被本轮唯一拆成税款、清欠、股票投资等行为。**不能把未识别项悄悄归入实体需求，也不能因政府债权显著增加就省略它。**

**五、同一M2变化，负债端持有结构也发生了变化。**

2025年6—8月，其他金融性公司在银行等机构的、计入M2部分的存款增加{a3.delta_odcs_financial_deposits_m2:,.2f}，约占M2净增加的{100*a3.delta_odcs_financial_deposits_m2/a3.m2_delta:.2f}%；个人存款增加{a3.delta_personal_total:,.2f}；单位活期加定期合计增加{a3.delta_corporate_demand+a3.delta_corporate_time:,.2f}。8月单月前一项增加{a1.delta_odcs_financial_deposits_m2:,.2f}，约占当月M2净增加的{100*a1.delta_odcs_financial_deposits_m2/a1.m2_delta:.2f}%。

这些比例是净余额增量的算术比值，其他月份分母为负或接近零时不作为结构权重。这里不把“其他金融性公司存款”写成已经流入股票的资金，也不据此推断居民正在入市。存款持有人分类与资产端的政府/企业债权分类是同一张资产负债表的两面，**不能把两面的贡献再相加。**

**六、接回510300：货币来源、订单、位置和波动必须同时判断。**

下面沿用此前固定病例和原20交易日路径，展示统计月、实际公布日、公布时市场已经走过的幅度。原E0、E1都保留，不选择收益更好的起点。

{frame_md(["统计月", "公布日", "剪刀差三月变化/百分点", "当时此前60日收益", "上行/下行年化波动尺度", "已公布新订单PMI", "E0后二十日毛收益", "E1后二十日毛收益"], market_rows)}

2025年8月这一行最能说明为什么不能只看剪刀差：三个月改善2.8个百分点，同时单位活期确有修复，但非金融债权较上年同期少增，新订单49.5，资金增量相当一部分体现在其他金融性公司存款；数据公布时510300此前60日已经上涨19.16%，上行波动尺度18.83%、下行10.65%。这些是已经发生的路径，不等于未来波动上界。

第六轮已保存的9月11日调查中，M1预期6.0%、M2预期8.7%；实际为6.0%和8.8%。剪刀差相对这两项汇总预期之差为-0.1个百分点，落在四项一位小数显示数合计±0.2个百分点的舍入范围内，**没有清晰的额外正面意外**。不是把三个月改善2.8个百分点都视为公告当天的新信息。

该病例原E0从2025年9月15日开盘至10月20日收盘的20交易日毛收益约+0.39%，延后一天的E1约+1.68%。反过来，2024年8月的负面货币与订单背景之后，窗口内出现了新的政策信息和大幅上涨；新政策不能提前放入9月13日的判断，也不能用事后新M1回溯值改写当时背景。

这些病例及其重叠窗口不构成独立的预测样本。本轮没有据它们设阈值、选牛熊分组、修改旧模型或计算新的策略收益。第六轮既有预期增量检验的失败状态保持不变。当前更明确的是：**货币结构改善是环境证据，是否有尚未反映在价格中的盈利或政策变化，才是下一步需要验证的收益问题。**

**七、已收敛的判断和仍待区分的传导。**

| 候选传导 | 本轮确认的部分 | 仍需什么来区分 |
|---|---|---|
| 企业经营回升推动活期增加 | 三个月单位活期增加、单位定期减少；原现金流研究显示部分公司收款改善 | 信贷实际用途、新订单、盈利预期修订；少付款和税款时点也会改善现金 |
| 财政支持传向企业 | 银行等对政府债权上升，政府净债权增加 | 财政实际支付对象和用途；政府在央行存款同期净增加，不能用净释放解释 |
| 金融资产配置与风险偏好变化 | 其他金融性公司计入M2的存款增加；市场已有明显上涨 | 存款对应机构和资产交易、政策意外及首次价格反应；不能将存款直接指为股票流入 |
| 基数与季节因素 | 同表同比分解显示基数项主要；季度月度差异显著 | 基数对应的行为原因、季节调整与更长同口径样本；不把基数改善当作新增需求 |

这张表是可区分的机制清单，不是买卖评分，也不保证以上四类穷尽所有原因。本轮没有找出能确认510300未来方向的可重复关系，完整目标继续保持active。下一步应集中到机制发生变化的具体公告：发布前市场预期是什么、当天哪些权重行业重新定价、第一段反应后是否仍存在可获得收益；不能只继续增加宏观变量数量。

**数据与复算。** 20个月原金额、35组完整区间、M2全部对应项、M1基数分项、原市场路径并列表和原PDF均已保存。32处原表数值经页面核对，35个区间余额差与恒等式复算通过；这只验证金额和来源连接，不验证预测能力。主要文件：

- [20个月原金额](results/20个月_货币与银行资产负债原金额.csv)
- [35组完整区间](results/完整窗口_单月三个月年内余额增量及对应项.csv)
- [全部M2对应项](results/M2变化_全部资产负债对应项.csv)
- [同表M1基数分项](results/三个月剪刀差_当期与基数的分项对数贡献.csv)
- [原公布信息、原市场路径与事后结构并列](results/20个月_原公布信息市场路径与事后结构并列.csv)
- [复算结果](verification.json)

来源为[2025年存款性公司概览](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172611493.pdf)、[2024年存款性公司概览](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111416432184461.pdf)、[2025年其他存款性公司表](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172647559.pdf)、[2025年货币当局表](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172659799.pdf)、[2024年货币当局表](https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111416404019802.pdf)。[央行货币定义](https://www.pbc.gov.cn/rmyh/109339/2025080818580470423/index.html)说明新M1包括个人活期和支付备付金。来源原件、取得时间和哈希见source_receipt.json。
"""
    report_path = OUT / "第八轮_银行信用存款结构与510300定价.md"
    report_path.write_text(report, encoding="utf-8")
    save(OUT / "goal_progress.json", {"at": now(), "goal_status": "active", "turn_classification": "PROGRESS",
        "completed": ["20个月资产负债原表连接", "35组余额对应项复算", "财政存款与政府融资分开", "M1基数和主体结构分解", "原510300市场位置和波动时钟连接"],
        "not_achieved": "未来方向的可靠增量与成本后优势尚未证实。", "goal_achieved": False,
        "next_question": "在政策或盈利预期变化的具体公告中，区分已发生的估值反应与剩余可获得收益。"})
    save(OUT / "completion_receipt.json", {"at": now(), "round_status": "RESEARCH_COMPLETE_GOAL_ACTIVE",
        "report": report_path.name, "report_sha256": digest(report_path), "figure": figure.relative_to(OUT).as_posix(),
        "figure_sha256": digest(figure), "verification_sha256": digest(OUT / "verification.json"),
        "figure_visual_review_pending": True, "goal_achieved": False})
    for name in ["money_balance_sources_v8.py", "money_balance_transmission_v8.py", "verify_money_balance_transmission_v8.py", Path(__file__).name]:
        shutil.copy2(ROOT / "research" / name, OUT / "code" / name)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    run()
