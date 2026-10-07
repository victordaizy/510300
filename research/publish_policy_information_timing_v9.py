"""发布政策、信用与价格路径的第九轮机制观察。"""
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
OUT = ROOT / "reports/research/510300_policy_information_timing_v9"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def pct(value):
    return f"{value * 100:+.2f}%"


def loan(value):
    return "未知（口径或分项缺失）" if pd.isna(value) else f"{value:+,.0f}"


def publish():
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "PASS_SAVED_LABELS_CLOCKS_AND_ROLLING_IDENTITIES"
    cases = pd.read_csv(OUT / "results/七个调查偏差病例_完整时序与传导.csv")
    ext = pd.read_csv(OUT / "results/56个月_货币基数与波动窗口构成.csv").set_index("month")
    july = cases.set_index("month").loc["2024-07"]
    j = ext.loc["2024-07"]
    facts = json.loads((OUT / "results/日期价格事实.json").read_text(encoding="utf-8"))
    sep = facts["september2024_path"]
    dates = {r["date"]: r for r in facts["dates"]}
    survey_labels = {
        "2022-05": "5年以上比调查多降10—15bp；1年期并非更宽松",
        "2022-08": "1年期比调查少降5bp；5年以上符合调查",
        "2023-06": "5年以上比调查少降至少2.5bp；1年期符合调查",
        "2023-08": "1年期比调查少降5bp；5年以上至少少降15bp",
        "2024-02": "5年以上比调查多降10—20bp；1年期调查数值未知",
        "2024-07": "两期限比旧调查多降10bp；中间已有政策降息",
        "2024-09": "多数预计降息，实际不降；缺少数值中位数",
    }
    table = ["| 公告月 | 相对已保存调查的偏差 | 公告前20日 | 第一完整反应日 | 原研究随后20日 |",
             "|---|---|---:|---:|---:|"]
    macro = ["| 公告月／当时最新货币月 | 剪刀差／近3月变化（百分点） | 企业中长期同比多增（亿元） | 居民中长期同比多增（亿元） | 最新新订单 |",
             "|---|---:|---:|---:|---:|"]
    for row in cases.itertuples():
        table.append(f"| {row.month} | {survey_labels[row.month]} | {pct(row.pre_return20)} | {pct(row.first_total_return)} | {pct(row.target20)} |")
        macro.append(f"| {row.month}／{row.known_money_stat_month} | {row.known_money_spread_pp:+.1f}／{row.known_money_delta3_spread_pp:+.1f} | {loan(row.known_credit_corporate_long_ytd_yoy_change_yi)} | {loan(row.known_credit_household_long_ytd_yoy_change_yi)} | {row.known_orders_first_release_value:.1f} |")
    text = f"""# 第九轮：政策宽松、信用背景与价格反应必须按先后顺序连接

已明确的结论是：**旧调查显示的“降息超预期”，和20日下行波动下降，都可能产生误读；把两者叠加，并不自动形成相互印证的看涨证据。** 本轮补上的是调查以后发生了什么、信贷和订单当时怎样、近期下跌是否真的缓和、第一段涨跌已经反映了什么。尚未证明一个能稳定预测510300未来收益的条件组合。

这不是把高低位、牛熊市再切几个格子。分析顺序是：货币变化从哪里来 → 哪些借款主体正在增加融资 → 订单和经营是否接住融资 → 何时有新增信息 → 市场已经怎样反应 → 此后是否又发生新事件。每一层都有独立含义，不能用一个总分代替。

**2024年7月22日提供了一个完整的反例。**

| 时点 | 已经公开的信息 | 仍然不知道的内容 |
|---|---|---|
| 7月19日15:43发布、17:00修改的调查 | 两个期限LPR调查中位数均为不变 | 周一新政策以及由此引起的预期调整 |
| 7月22日08:00 | 7天逆回购利率由1.80%降至1.70%，操作方式调整 | 08:00至09:00间更新后的LPR数值共识 |
| 7月22日09:00 | 两个期限LPR均下降10bp，分别至3.35%、3.85% | 单独LPR消息的即时价格贡献 |
| 7月22日09:30—15:00 | 510300开盘相对前收{pct(july.first_gap_return)}，当日含分红收益{pct(july.first_total_return)} | 下一段收益的方向 |
| 7月23日开盘—8月19日收盘 | 原研究20个交易日收益事后为{pct(july.target20)} | 此路径不能提前成为7月22日的特征 |

08:00的[央行政策利率及操作方式公告](https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125469/5409998/index.html)和09:00的[LPR公告](https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/5410019/index.html)是两条不同的来源。09:20:30的操作结果公告也另行保留，不能用它覆盖更早的08:00消息。相对7月19日调查的−10bp差额仍然保留；**没有取得08:00以后更新的数值预期，所以不能把它直接称为09:00时点的实时超预期，也不能武断地改成零。**

当时最新可见的是6月金融数据：M1−M2剪刀差为−11.2个百分点，较3个月前又下降4.0个百分点。既有余额分解中，同期相对增长的对数变化为−4.2652个对数百分点，其中当期余额相对变化−3.8159、基数及版本残差−0.4493。这里是对数分解，不能与普通剪刀差百分点混加，也不能把残差指认为某一种经济原因。

信用和订单没有提供普遍好转的验证：截至6月，企业中长期贷款年内累计同比少增16,300亿元，居民中长期同比少增2,800亿元；制造业新订单指数49.5。这些信息描述融资需求与经营承接偏弱，不能证明降息已经转化成新增订单、利润与股票上涨。此前20个交易日510300已经上涨{pct(july.pre_return20)}，因此也不能只凭一句“当时位置低”概括价格背景。

波动这一层还有独立的误读。7月22日收盘，20日下行波动从五个交易日前的{j.review_d20_five_days_ago_pct:.3f}%降至{j.review_d20_now_pct:.3f}%；但最近5日下跌幅度平方和为{j.review_recent5_down_energy_pct_squared:.5f}，前一段5日仅{j.review_previous5_down_energy_pct_squared:.5f}。它之所以下降，是因为退出20日窗口的旧5日平方和更大，为{j.review_exited5_down_energy_pct_squared:.5f}。后三个数字的单位都是“百分数的平方”，不是收益率。

对应恒等式是D²当期减D²五日前 = 252/20 ×（最近5日负收益平方和 − 退出窗口5日负收益平方和）。最近5日与前5日的比较，回答的是近期下跌是否减轻；最近5日与退出窗口5日的比较，解释的是滚动指标为何变化。**两者回答不同问题。** 同时，7月22日D20比7月19日的4.890%还高；“比前一天上升”与“比五日前下降”并不矛盾。

把以上各层接起来，本例应解释为：信用与订单偏弱的环境中，政策先降息，旧调查尚未更新；首日价格走弱，20日降波又有旧损失退出的机械作用。它不能被记成“新增利好＋抛压缓解”的双重确认。这个解释否定的是该推断链，**不是把一个亏损病例改造成必跌规则**。

**全部七个调查偏差病例均保留。**

{chr(10).join(table)}

七例按旧调查偏差或多数降息预期落空选出，不按股票收益挑选。某一个期限偏宽松不等于整个政策组合都偏宽松。第一完整反应日按父研究时钟确定；“原研究随后20日”从该日之后的下一交易日开盘开始，持有20个交易日，固定股数计分红、排除买入日除息权利，未扣成本。它不是最早可交易时点，不是本轮新账户，也不同于月度M1观察表的E0/E1时钟。

下面的货币、信贷、订单均按各次公告自己的时点连接，不能把月度发布日那一刻的价格、估值或广度误当成LPR公告时的状态。

{chr(10).join(macro)}

贷款同比多增均指当年年初至该统计月，和上年同区间相比；不是单月放贷金额。2023年跨口径比较、2022年4月居民中长期缺失均保留未知。新订单来自此前已公布的制造业PMI分项，50是扩张收缩的扩散指数界线，不代表订单金额增长率。完整56个月及原先28个月缺失调查均已保存，没有拿这七例计算一个可推广的胜率。

2024年2月尤其说明“剪刀差改善”和“实体修复”仍须分开：当时最新1月剪刀差近3月改善5.6个百分点，但相对增长的对数变化5.3115中，当期相对余额项为−0.3600，基数及版本残差为+5.6716；新订单仍是49.0，企业与居民中长期同比增量方向也不同。其后上涨不能反过来证明“剪刀差改善就是企业融资和经营都转强”。

**后来发生的新政策，也不能记到最初消息的账上。**

2024年9月20日LPR不降，原研究9月23日开盘至10月25日收盘的20日收益为{pct(sep['original_target20'])}。在9月24日第一条已核对政策节点之前，至9月23日收盘只实现{pct(sep['before_september24_node_return'])}；余下{sep['after_september23_close_contribution_same_initial_capital'] * 100:+.2f}个百分点发生在此后的多政策窗口。这里按同一初始资金拆时间段，不能把后段金额全当作9月24日单项政策的因果效果，更不能据此得出“不降息利好”。

2023年8月21日LPR之后，8月27日又公布[证券交易印花税减半公告](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm)。8月28日510300开盘相对前收{pct(dates['2023-08-28']['gap_return'])}，开盘至收盘{pct(dates['2023-08-28']['intraday_return'])}，全天相对前收{pct(dates['2023-08-28']['total_close_return'])}。这说明开盘已经反映的信息，与开盘之后剩下的收益，必须分开测量；整日变化也包含其他消息，不能单项归因。该公告仅作资本市场环境节点，不用来改写510300交易费率。旧目录中的“零条后续节点”只表示目录未收录，绝不等于后来没有事件；本轮以单独文件增加这一节点，没有改写旧目录。

**信贷传导还必须考虑借款人和银行的不同损益。**

[央行2023年第二季度货币政策执行报告](https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/4883187/e6dc129d48d9474cb6455691eee05766/2023081717054357626.pdf)在8月17日已经公布，早于8月18日调查与8月21日LPR。报告印刷第5—6页指出，银行净息差收窄、贷款扩张有资本约束，利润留存是内源补充资本的重要方式。因此，贷款利率下降对借款人的成本、银行的息差、信用损失及后续贷款供给，需要分别分析。它支持这条传导机制，**并不证明2023年8月五年期LPR不变就是由其中某一项单独造成的**。也不能据此跳过510300内部不同企业的盈利权重和估值变化。

上一轮已经量化的“货币来源与存款持有人结构”，本轮继续接到“预期是否更新、价格是否已反应、风险是否真的缓和”这一段。由此得到的可执行研究约束是：一笔候选机会必须逐项说明其新增信息、实体承接、指数盈利关联和尚未兑现的价格空间；没有这些证据时，M1−M2改善、降息或降波中的任何一个，甚至几个并列出现，都不足以确认方向。

本轮完成56个月的三类时点连接、112个波动窗口复算、1,176条逐日路径和旧收益标签对照；四个新增原文文件均留存来源与哈希。全体56个父级收益标签未变，最大重算误差小于1e−15。原LPR研究已经完成过实际探索训练，204次拟合、14个含基准账户场景，候选策略往返交易为0；未达到原研究两种成本下的目标，结果保留。**本轮0个新模型、0个新账户，不构成独立验证，总目标仍未完成。**

下一步的有效新增证据应来自提前固定的信息变化及其盈利传导，而不是继续看完这些历史收益再增减阈值。当前任务中另已记录的未来订单、成本与资金利率观察保持原状态，不重复新建、不提前填结果。

报告使用的历史行情、调查以及月度资料均为既有研究保存数据；本次下载的央行与财政部原文是当前可取得的页面或文件，不等于已认证的历史首版。全部来源时间、未知项、未扣成本及后续事件边界都随结果保存。

直接查看：

- `results/七个调查偏差病例_完整时序与传导.csv`：七个病例全部数值。
- `results/56个月_原调查新信息时序与价格波动.csv`：完整56个月，旧字段原样保留。
- `results/56个月_货币基数与波动窗口构成.csv`：货币当期与基数项、最新与退出窗口的平方和。
- `results/56个月_第一完整反应日至原终点的逐日路径.csv`：同一原终点下的逐日路径，起点为首反应日开盘，不与随后20日收益混用。
- `results/原观察窗口内_已记录的后续政策节点.csv`及`results/定向补充的窗口内后续政策.csv`：不完整政策目录的原记录与本轮追加。
- `verification.json`：上述保存数据的复核结果。
"""
    report = OUT / "第九轮_信息先后信用背景与波动窗口.md"
    report.write_text(text, encoding="utf-8")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 11,
        "figure.facecolor": "#f7f8fa", "axes.facecolor": "#ffffff", "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(14, 12.5))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 2.0, 2.7], hspace=.55, wspace=.3)
    fig.suptitle("同一次“宽松”，要看信息先后、信用背景与剩余行情", x=.055, ha="left", y=.97, fontsize=21, weight="bold", color="#132d46")
    top = fig.add_subplot(grid[0, :]); top.axis("off")
    blocks = [("7月19日调查", "LPR中位数：均不变"), ("7月22日 08:00", "政策利率先下降10bp"),
              ("7月22日 09:00", "LPR再公布下降10bp"), ("7月22日收盘", "510300当日 −0.81%")]
    for x, (title, body) in zip([.01, .265, .52, .775], blocks):
        top.text(x, .82, title + "\n" + body, va="top", fontsize=12, linespacing=1.9,
                 bbox={"boxstyle": "round,pad=.7", "facecolor": "#e7eff6", "edgecolor": "none"})
    top.text(.01, -.02, "旧调查到LPR公告之间已有新信息；08:00以后更新的数值预期仍未知。", fontsize=11, color="#664f32")
    left = fig.add_subplot(grid[1, 0])
    vals = [j.review_previous5_down_energy_pct_squared, j.review_recent5_down_energy_pct_squared, j.review_exited5_down_energy_pct_squared]
    left.bar(["前一段5日", "最近5日", "退出窗口的旧5日"], vals, color=["#bccbd5", "#c97942", "#427b95"], width=.55)
    for i, val in enumerate(vals): left.text(i, val + .03, f"{val:.3f}", ha="center", fontsize=12)
    left.set_ylim(0, 1.0); left.set_ylabel("负收益平方和（百分数²）")
    left.set_title("7月22日：降波中混有旧大跌退出", loc="left", fontsize=13, weight="bold", pad=17)
    left.text(.02, .93, f"D20：{j.review_d20_five_days_ago_pct:.2f}% → {j.review_d20_now_pct:.2f}%（相比5日前）", transform=left.transAxes, fontsize=11)
    right = fig.add_subplot(grid[1, 1]); right.axis("off")
    right.set_title("7月22日当时能看到的信用与订单", loc="left", fontsize=13, weight="bold", pad=17)
    evidence = ["最新货币月：2024年6月", "M1−M2：−11.2个百分点，近3月再降4.0", "企业中长期贷款：年内累计同比少增1.63万亿元",
                "居民中长期贷款：年内累计同比少增0.28万亿元", "制造业新订单：49.5", "公告前20个交易日510300：+2.39%"]
    for i, line in enumerate(evidence): right.text(0, .91 - i * .145, line, fontsize=11.2, color="#263d4e")
    bottom = fig.add_subplot(grid[2, :])
    x = np.arange(len(cases)); width = .33
    bottom.bar(x - width/2, cases.first_total_return * 100, width, label="第一完整反应日", color="#6195ab")
    bottom.bar(x + width/2, cases.target20 * 100, width, label="原研究随后20日", color="#d08a55")
    for i, row in enumerate(cases.itertuples()):
        for off, val in [(-width/2, row.first_total_return * 100), (width/2, row.target20 * 100)]:
            bottom.text(i + off, val + (.6 if val >= 0 else -.7), f"{val:+.2f}%", ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
    bottom.axhline(0, color="#9aa8b0", lw=.8)
    bottom.set_xticks(x, cases.month); bottom.set_ylim(-9, 29)
    bottom.set_ylabel("含分红毛收益（%）")
    bottom.set_title("七个调查偏差病例：后续收益还包含新的政策和经营信息", loc="left", fontsize=13, weight="bold", pad=16)
    bottom.legend(loc="upper left", frameon=False, ncol=2)
    bottom.grid(axis="y", alpha=.13)
    fig.text(.06, .035, "随后20日：首反应日之后的下一交易日开盘起算，固定股数计分红，未扣成本。七例按调查偏差选择，非独立验证。\n2024年9月窗口包含9月24日起的新政策；图中整段收益不等于最初LPR消息的因果效果。", fontsize=10, color="#536571", linespacing=1.65)
    fig.subplots_adjust(left=.075, right=.96, top=.89, bottom=.13)
    figure = OUT / "figures/信息先后与信用波动_七病例.png"
    fig.savefig(figure, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    result = {"at": datetime.now().astimezone().isoformat(), "study_id": "510300_POLICY_INFORMATION_TIMING_V9",
        "status": "MECHANISM_TIMING_DIAGNOSTIC_COMPLETED_PENDING_VISUAL_REVIEW",
        "conclusion": "旧调查偏差不等于更新后的预期差，滚动降波不等于近期下跌减轻，后续政策不能回填为原始信号。",
        "july2024": {"minutes_between_policy_cut_and_lpr": 60, "updated_consensus": "UNKNOWN",
            "first_reaction_return": float(july.first_total_return), "original_target20": float(july.target20),
            "d20_pct": float(j.review_d20_now_pct), "d20_five_days_ago_pct": float(j.review_d20_five_days_ago_pct),
            "recent_negative_return_energy_pct_squared": float(j.review_recent5_down_energy_pct_squared),
            "previous_negative_return_energy_pct_squared": float(j.review_previous5_down_energy_pct_squared)},
        "verified_events": 56, "cases": 7, "missing_surveys": 28, "new_models": 0, "new_accounts": 0,
        "old_result_unchanged": True, "goal_achieved": False, "goal_status": "active", "independent_validation": False,
        "report_path": report.relative_to(ROOT).as_posix(), "report_sha256": digest(report),
        "figure_path": figure.relative_to(ROOT).as_posix(), "figure_sha256": digest(figure),
        "verification_status": verification["status"], "visual_review_pending": True}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    publish()
