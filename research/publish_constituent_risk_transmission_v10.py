"""发布成分风险、共同变动与货币信用背景的连接结果。"""
from __future__ import annotations

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
OUT = ROOT / "reports/research/510300_constituent_risk_transmission_v10"


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def pct(value):
    return f"{100 * value:.2f}%"


def publish():
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "PASS_SAVED_CONSTITUENT_RISK_IDENTITIES"
    data = pd.read_csv(OUT / "results/111个观察点_篮子风险与原后续路径.csv")
    rows = data.set_index(["origin_id", "basket"])
    january = rows.loc[("monthly_2021-01", "REFERENCE_WEIGHT")]
    equal = rows.loc[("monthly_2021-01", "EQUAL")]
    april = rows.loc[("monthly_2023-03", "REFERENCE_WEIGHT")]
    february = rows.loc[("lpr_2024-02", "REFERENCE_WEIGHT")]
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    money = monthly.loc["2021-01"]
    frac = january.variance_change_correlation_component / (january.current_variance - january.previous_variance)
    individual_change = january.current_weighted_individual_down2 / january.previous_weighted_individual_down2 - 1
    stocks = pd.read_parquet(OUT / "results/成分股风险与贡献.parquet")
    jstocks = stocks[(stocks.origin_id == "monthly_2021-01") & (stocks.basket == "REFERENCE_WEIGHT")].copy()
    jstocks["weighted_individual_down2_change"] = jstocks.weight * (jstocks.current_down2 - jstocks.previous_down2)
    industries = jstocks.groupby("industry", as_index=False).agg(weight=("weight", "sum"), stock_count=("symbol", "count"),
                    weighted_individual_down2_change=("weighted_individual_down2_change", "sum"))
    industries.sort_values("weighted_individual_down2_change", ascending=False).to_csv(OUT / "results/20210209_行业自身下行能量变化.csv", index=False, encoding="utf-8-sig")
    receipt = json.loads((OUT / "results/build_receipt.json").read_text(encoding="utf-8"))
    table = ["| 2021-02-09复核 | 五个交易日前的20日窗口 | 当日的20日窗口 |", "|---|---:|---:|"]
    for label, old, new in [
        ("510300实际总波动", january.etf_rv_previous, january.etf_rv_current),
        ("510300实际下行波动", january.etf_down_previous, january.etf_down_current),
        ("参考权重篮子总波动", january.previous_rv, january.current_rv),
        ("参考权重篮子下行波动", january.previous_net_down, january.current_net_down),
        ("300只等权篮子总波动", equal.previous_rv, equal.current_rv),
        ("300只等权篮子下行波动", equal.previous_net_down, equal.current_net_down),
    ]:
        table.append(f"| {label} | {pct(old)} | {pct(new)} |")
    coverage = ["| 类别 | 原观察点 | 完整等权篮子 | 完整参考权重篮子 |", "|---|---:|---:|---:|",
                "| 月度货币观察 | 104 | 26 | 24 |", "| 原先7个LPR病例（单列） | 7 | 1 | 1 |"]
    report = f"""# 第十轮：指数变平稳，未必意味着多数成分股风险减轻

这轮得到的明确结论是：**要区分指数层面的降波、成分股自身的风险变化，以及成分之间的相互抵消。三者可能朝不同方向变化。** 加上货币来源、信用用途和前期定价后，才知道“降波”处在什么过程中；仅在高位低位或牛熊市之间分组，仍会漏掉这层结构。

计算覆盖全部104个月观察点和原先七个LPR病例。来源不足的观察点原样保留未知，没有删去缺口股票后重新分配权重。完整结果只作结构解释，没有新增模型、账户或收益分组搜索。

**2021年2月9日：指数变稳，但多数成分没有同步减轻下行风险。**

{chr(10).join(table)}

上表六项均为20日年化尺度。两组股票篮子使用同一组观察日300只成分，固定权重回看前后窗口。参考权重篮子来自此前最近的完整快照；等权篮子则每只股票1/300。它们描述两个不同的聚合角度，不互相择优替换，也不冒充510300实际持仓归因。

300只股票中，只有138只、即46.0%的个股下行平方能量下降；其余162只没有下降。按参考权重平均的个股自身下行平方能量，反而增长{individual_change * 100:.2f}%。同一窗口，只有119只股票的20日总收益为正，即39.67%；这些上涨股票占参考权重52.34%。这些是按本轮固定300只股票与股东总收益重算的广度，和其他表中不同成员、价格或时钟定义的广度不强行混用。

进一步拆开参考权重篮子的总方差，成分之间的平均相关由{january.previous_weighted_mean_correlation:.3f}降至{january.current_weighted_mean_correlation:.3f}，解释了总方差下降的{frac * 100:.2f}%；余下部分来自个股波动结构变化。这里是固定权重下的对称代数分解，比例的分母是方差变化，不是波动率变化，也不是“政策造成下跌”的因果份额。

行业内部也不是同向缓和：电子、计算机、汽车等行业的个股下行能量加权项增加；食品饮料、电力设备、农林牧渔等减少。七只股票当期行业分类缺失，合计参考权重约0.60%，单独保留。行业分歧可由价格记录确认，实际资金迁移、持仓集中或投资者动机没有被这些价格数据单独证实。

**把这一天放回货币—信用—经营—定价的顺序，含义更清楚。**

| 当时已经公开或可计算的层面 | 2021年2月9日21点复核时的事实 | 可以支持的判断 |
|---|---|---|
| 剪刀差 | 1月M1−M2为+5.3个百分点，近3月改善+6.7个百分点 | 标题指标改善，但要继续拆来源 |
| 当期与基数 | 相对同比增长的对数变化+6.0060，其中当期相对余额项−0.2456，基数及版本残差+6.2516 | 改善不能全部解释成当期新增经营支付；普通百分点与对数百分点不混加 |
| 信贷 | 企业中长期同比多增3,800亿元，居民中长期同比多增1,957亿元 | 这两个融资分项较上年同期增强，但不能推出所有公司的盈利都提高 |
| 订单 | 新订单52.3，较上月下降1.3个百分点 | 仍处扩张区间，扩张广度较上月减弱 |
| 前期价格与估值 | 510300前60个交易日已上涨15.95%；此前同表记录的沪深300原始PE为18.32 | 应检查改善是否已经反映；PE本身没有给出高估或低估结论 |
| 内部风险 | 仅46%个股下行风险下降；等权下行尺度略升 | 指数降波不足以证明普遍风险收敛 |

1月金融数据在2月9日16:00:20公布，表中收盘价格在消息之前已经形成。21点的复核把新发布数据和最近可见的市场状态连接起来；**不能把此前的价格变动称为对这条收盘后公告的反应。**

按原月度观察口径，下一可执行开盘起20个交易日E0收益为{january.parent_E0_20_return * 100:+.2f}%，再延迟一天的E1为{january.parent_E1_20_return * 100:+.2f}%。它们均为原样保留的固定股数含分红毛收益，未扣成本，不是本轮新建策略。本例说明“剪刀差改善＋信用多增＋指数降波”仍未完成从宏观到剩余收益的证明链；不能用后来下跌倒推某个因素必然导致下跌。

**2023年4月11日的完整观察进一步说明：指数波动下降时，个股平均波动甚至可能上升。**

该例用来说明当期分解项方向相反，没有按后续股票收益选择。参考权重篮子的总波动从{pct(april.previous_rv)}降至{pct(april.current_rv)}；个股波动的加权平均却从{pct(april.previous_weighted_average_stock_vol)}升至{pct(april.current_weighted_average_stock_vol)}，平均相关从{april.previous_weighted_mean_correlation:.3f}降至{april.current_weighted_mean_correlation:.3f}。因此，本例的篮子降波来自共同涨跌程度下降压过了个股波动的增加，不能解读为所有组成部分都更稳定。

相反，原先固定的2024年2月20日LPR病例中，199/300、即66.33%的个股下行平方能量下降，参考权重平均的个股下行能量也下降；但总波动上升。更广泛的下行缓和，可以与上涨带来的总波动扩大同时发生。这里只比较当前机制，LPR与月度观察保持各自时钟，没有把两个病例拼成新的买卖阈值。

**本轮能确认“这些机制确实存在”，还不能确认它们的总体发生概率或预测胜率。**

{chr(10).join(coverage)}

月度观察中77个月存在未解决的成分收益缺口，1个月的复核时点超出原行情截止；26个完整等权观察中，另有2个月参考权重不完整或不满足原时间限制，因此完整参考权重观察为24个。七个LPR病例只有2024年2月20日这一例满足本轮完整300只、连续25日的数据要求。2024年7月LPR的两只股票各有一个未解决收益日，本轮不补零，也不据部分篮子解释它的成分风险。

在24个完整参考权重月度观察中，16次篮子下行风险下降，其中5次个股自身下行能量的加权平均上升，4次下降的股票不足一半。这些数字只描述可完成的样本。缺口覆盖并不随机，尤其2025年新M1口径只有2个完整参考权重月度观察，不能据此给总体比例、跨口径规律或预测优势下结论。

**计算含义和使用边界。**

对每一个固定权重篮子，日收益R由正收益贡献P减负收益贡献绝对值G得到。篮子的下行平方能量是252×20日平均[min(R,0)²]；各下跌贡献合计的平方能量为252×20日平均[G²]。两者之差衡量同日上涨抵消下跌的作用。比较前后窗口时，必须看负收益来源变化，也看抵消作用变化；不能把这个“平方能量差”当作实际资金流量。

总方差另拆成A+Bρ：A是各成分自身方差乘权重平方之和，B等于加权个股波动之和的平方减A，ρ为波动及权重加权的平均相关。固定同一组权重后，方差变化=ΔA+平均ρ×ΔB+平均B×Δρ。前两项归到个股波动结构，最后一项归到共同变动程度。这是恒等式分解，不是因果识别。

本轮观察篮子的固定每日权重有别于实际指数的调整股本、价格变化和维护处理。[中证指数公布的沪深300编制方案（2023年9月版）](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300_Index_Methodology_cn.pdf)列出了调整市值、除数和分红等处理；这里只用它说明口径差异，不把该版本当作2021年所有规则的历史认证。本轮历史参考权重也未取得全期首版认证。完整参考权重月度观察与510300最近20日收益的相关系数中位数约0.993，但高相关不等于精确复制，逐点误差均保留。

对研究方向的直接影响是：**以后“降波”需要同时回答三件事——旧损失是否退出、多数成分是否缓和、内部抵消是否暂时增强。然后再回到货币和信贷的原因，判断改善能否形成新的、尚未兑现的指数盈利或估值变化。** 这些事实可以提高解释的分辨率；它们尚未提供确定性的涨跌预测。

本轮重新计算52个完整篮子、1,300条日分项、15,600条成分风险记录，全部恒等式和原标签复核通过。既有横截面离散度预测研究的冻结拒绝状态保留，本轮没有将该失败候选改名重跑。总目标继续保持进行中，0个新模型、0个新账户。

直接查看完整数据：

- [111个观察点的完整状态及原路径]({(OUT / 'results/111个观察点_篮子风险与原后续路径.csv').as_posix()})
- [逐日上涨、下跌及抵消分项]({(OUT / 'results/固定篮子_25日上涨下跌分项.csv').as_posix()})
- [2021年2月9日行业自身下行能量变化]({(OUT / 'results/20210209_行业自身下行能量变化.csv').as_posix()})
- [未解决的成分收益缺口]({(OUT / 'results/未解决成分收益缺口.csv').as_posix()})
- [复核结果]({(OUT / 'verification.json').as_posix()})
"""
    report_path = OUT / "第十轮_指数降波与成分风险的区别.md"
    report_path.write_text(report, encoding="utf-8")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 11,
        "figure.facecolor": "#f5f7f9", "axes.facecolor": "#ffffff", "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(2, 2, hspace=.46, wspace=.30)
    fig.suptitle("2021年2月9日：指数降波，没有覆盖多数成分股", x=.055, ha="left", y=.97, fontsize=21, weight="bold", color="#16374a")
    fig.text(.058, .915, "在货币、信用改善的标题之下，进一步区分内部风险、相互抵消与已经发生的上涨。", fontsize=12, color="#536976")
    ax = fig.add_subplot(grid[0, 0])
    old = [january.etf_down_previous * 100, january.previous_net_down * 100, equal.previous_net_down * 100]
    current = [january.etf_down_current * 100, january.current_net_down * 100, equal.current_net_down * 100]
    x = np.arange(3)
    ax.bar(x - .18, old, .36, color="#b1c4cf", label="五个交易日前")
    ax.bar(x + .18, current, .36, color="#37788e", label="当日")
    for i in range(3):
        ax.text(i - .18, old[i] + .25, f"{old[i]:.2f}", ha="center", fontsize=10)
        ax.text(i + .18, current[i] + .25, f"{current[i]:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x, ["510300实际", "参考权重篮子", "300只等权篮子"])
    ax.set_ylim(0, 19); ax.set_ylabel("20日下行尺度（年化%）")
    ax.set_title("指数和等权篮子，方向已经不同", loc="left", fontsize=13, weight="bold", pad=16)
    ax.legend(frameon=False, ncol=2, fontsize=10, loc="upper center")
    ax = fig.add_subplot(grid[0, 1]); ax.axis("off")
    ax.set_title("看个股：只有138/300只下行风险下降", loc="left", fontsize=13, weight="bold", pad=16)
    ax.text(.02, .78, "46.0%", fontsize=35, weight="bold", color="#37788e")
    ax.text(.47, .79, "54.0%", fontsize=35, weight="bold", color="#b4774b")
    ax.text(.02, .63, "下行能量下降", fontsize=12)
    ax.text(.47, .63, "没有下降", fontsize=12)
    ax.text(.02, .39, "个股自身下行能量的权重平均：+2.02%", fontsize=12, color="#513e34")
    ax.text(.02, .22, "过去20日上涨：119/300只（39.67%）", fontsize=12)
    ax.text(.02, .08, "上涨股票的参考权重：52.34%", fontsize=12)
    ax = fig.add_subplot(grid[1, 0])
    values = [-january.variance_change_stock_vol_component * 10000, -january.variance_change_correlation_component * 10000]
    ax.barh(["个股波动变化", "共同变动降低"], values, height=.48, color=["#91a9b9", "#37788e"])
    for i, val in enumerate(values): ax.text(val + 1.5, i, f"{val / sum(values) * 100:.1f}%", va="center", fontsize=12)
    ax.set_xlim(0, max(values) * 1.3)
    ax.set_ylim(-.5, 1.65)
    ax.set_xlabel("对篮子年化方差下降的贡献（百分点²）")
    ax.set_title("总方差降低，约77%来自共同变动减弱", loc="left", fontsize=13, weight="bold", pad=16)
    ax.text(.01, .97, "参考权重平均相关：0.229 → 0.200", transform=ax.transAxes, va="top", fontsize=11)
    ax = fig.add_subplot(grid[1, 1]); ax.axis("off")
    ax.set_title("再接回当时的货币、信用与定价", loc="left", fontsize=13, weight="bold", pad=16)
    evidence = ["剪刀差近3月改善6.7个百分点，主要在基数项", "企业、居民中长期贷款同比均多增", "新订单52.3，较上月回落1.3个百分点", "510300前60个交易日已上涨15.95%", "收盘后发布的新数据，不能解释此前的收盘涨跌"]
    for i, line in enumerate(evidence): ax.text(0, .88 - i * .175, line, fontsize=11, color="#304b5e")
    fig.text(.06, .045, "固定300只观察篮子，用于解释聚合结构；参考权重未认证全期首版，不是ETF精确持仓归因。\n本轮104个月中仅26个等权、24个参考权重观察满足完整数据要求；缺失原样保留，不推断总体比例或预测胜率。", fontsize=10, color="#596e7c", linespacing=1.6)
    fig.subplots_adjust(left=.08, right=.955, top=.845, bottom=.15)
    figure = OUT / "figures/指数降波与内部风险_20210209.png"
    fig.savefig(figure, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=.2); plt.close(fig)
    result = {"at": datetime.now().astimezone().isoformat(), "study_id": "510300_CONSTITUENT_RISK_TRANSMISSION_V10",
        "status": "CONSTITUENT_MECHANISM_COMPLETED_PENDING_VISUAL_REVIEW", "continuation_classification": "PROGRESS",
        "main_conclusion": "指数降波可能与多数个股下行风险未缓和、权重平均个股下行能量上升同时存在；必须拆开内部抵消和普遍改善。",
        "january2021": {"observation_date": "2021-02-09", "stocks_with_lower_downside": 138, "all_stocks": 300,
            "stock_up20_count": 119, "individual_weighted_down_energy_change": individual_change,
            "correlation_component_share_variance_reduction": frac, "index_past60_return": float(january.past_return60)},
        "source_scope": "FROZEN_LOCAL_INPUTS_AND_CURRENT_OFFICIAL_METHODOLOGY_REFERENCE",
        "official_reference": "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300_Index_Methodology_cn.pdf",
        "coverage": receipt["status_counts"], "verification_status": verification["status"],
        "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False, "goal_status": "active",
        "report": report_path.relative_to(ROOT).as_posix(), "report_sha256": sha(report_path),
        "figure": figure.relative_to(ROOT).as_posix(), "figure_sha256": sha(figure), "visual_review_pending": True}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps({"status": result["status"], "january2021": result["january2021"], "report_sha256": result["report_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    publish()
