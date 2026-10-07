"""将货币持有人、春节时点和信贷结构连接为可复查的研究说明。"""
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
OUT = ROOT / "reports/research/510300_holiday_money_credit_bridge_v11"
REPORT = "第十一轮_春节存款持有人与信贷结构.md"
FIGURE = "figures/同比高增背后的余额与信贷结构.png"
DISPLAY = "C:/Users/戴周阳/Documents/New project 8/"


def stamp():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def local(relative, title):
    return f"[{title}](<{DISPLAY}{relative}>)"


def transfer_examples():
    # 所有金额均为假设的100元，不是对实际工资流或银行放贷用途的估计。
    vectors = [
        ("单位活期转个人活期", -100, 100, 0, 0),
        ("单位活期转个人定期", -100, 0, 100, 0),
        ("单位活期提取现金", -100, 0, 0, 100),
        ("新增银行贷款并留在单位活期", 100, 0, 0, 0),
    ]
    rows = []
    for label, unit, personal, term, cash in vectors:
        rows.append({"scenario": label, "unit_demand_change_cny": unit,
            "personal_demand_change_cny": personal, "personal_term_change_cny": term,
            "cash_change_cny": cash, "old_m1_change_cny": unit + cash,
            "new_m1_change_cny": unit + personal + cash,
            "m2_change_cny": unit + personal + term + cash,
            "scope": "假设其他条件不变；余额归属恒等式，不是观察到的资金流。",
            "available_at_20210209_21": False})
    pd.DataFrame(rows).to_csv(OUT / "results/100元划转_新旧口径示意.csv", index=False, encoding="utf-8-sig")


def make_figure(money, flows):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    fig.set_facecolor("#faf9f5")
    fig.subplots_adjust(left=.07, right=.975, top=.84, bottom=.15, wspace=.23, hspace=.58)
    fig.text(.07, .953, "2021年1月：M1同比高增，哪些事实同时发生？", fontsize=21, weight="bold", color="#183a47")
    fig.text(.07, .91, "把同比、余额、存款持有人和贷款期限拆开，才能判断指标背后的过程。", fontsize=12, color="#53626b")
    navy, orange = "#287487", "#c37442"
    selected = money.loc[["2020-12", "2021-01", "2021-02"]]
    x = np.arange(3)
    labels = ["2020年12月", "2021年1月", "2021年2月＊"]
    ax = axes[0, 0]
    values = selected.m1_yoy_pp.to_numpy()
    ax.plot(x, values, color=navy, marker="o", markersize=8, linewidth=2.5)
    for i, v in enumerate(values):
        ax.text(i, v + .7, f"{v:.1f}%", ha="center", fontsize=12, color=navy)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 18), xlim=(-.3, 2.3), ylabel="同比增速（%）")
    ax.set_title("A  同比增速先升后降", loc="left", fontsize=14, pad=14)
    ax = axes[0, 1]
    balances = selected.m1_value_yi.to_numpy() / 10000
    bars = ax.bar(x, balances, width=.56, color=[navy, navy, orange])
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in balances], padding=6, fontsize=12)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 77), ylabel="M1余额（万亿元）")
    ax.set_title("B  1月余额按公布精度与上月持平", loc="left", fontsize=14, pad=14)
    jan20 = flows.loc[("2020-01", "MONTH_REPORTED")]
    jan21 = flows.loc[("2021-01", "MONTH_REPORTED")]
    panels = [
        (axes[1, 0], ["deposit_corporate", "deposit_household"], ["非金融企业存款", "住户存款"],
         "C  1月存款增量在持有人之间呈相反变化", (-2.5, 5.6)),
        (axes[1, 1], ["loan_corporate", "loan_corporate_long", "loan_household"],
         ["企事业贷款总额", "其中：中长期", "住户贷款"],
         "D  中长期多增与企事业贷款总额少增并存", (0, 3.9)),
    ]
    for ax, keys, ticks, title, limits in panels:
        pos = np.arange(len(keys))
        for shift, record, color, label in [(-.19, jan20, navy, "2020年1月"), (.19, jan21, orange, "2021年1月")]:
            vals = [record[k + "_value_yi"] / 10000 for k in keys]
            bars = ax.bar(pos + shift, vals, width=.35, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], padding=5, fontsize=10)
        ax.axhline(0, color="#889298", linewidth=.7)
        ax.set(xticks=pos, xticklabels=ticks, ylim=limits, ylabel="当月新增（万亿元）")
        ax.set_title(title, loc="left", fontsize=14, pad=14)
        ax.legend(loc="upper left" if keys[0] == "deposit_corporate" else "upper right", frameon=False, fontsize=10, ncol=2)
    for ax in axes.flat:
        ax.set_facecolor("#faf9f5")
        ax.grid(axis="y", color="#deded8", linewidth=.65, alpha=.7)
        ax.set_axisbelow(True)
    fig.text(.07, .083, "＊2021年2月数据于3月10日公布，不能进入2月9日的判断。企业中长期贷款是企事业贷款总额的子项，不可相加。", fontsize=10, color="#53626b")
    fig.text(.07, .052, "来源：央行对应月份金融统计公告。同比为原报值；跨公告金额有精度及版本差异。存款净变动不证明同一笔资金的去向。", fontsize=10, color="#53626b")
    fig.savefig(OUT / FIGURE, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def publish():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("已完成的本轮说明不覆盖。")
    transfer_examples()
    money = pd.read_csv(OUT / "results/货币余额与发布时点.csv").set_index("month")
    flows = pd.read_csv(OUT / "results/原文贷款存款分项.csv").set_index(["month", "interval"])
    log = pd.read_csv(OUT / "results/一月与两月_当期基数对数分解.csv").set_index("end")
    previous_path = ROOT / "reports/research/510300_constituent_risk_transmission_v10/result.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    case = previous["january2021"]
    make_figure(money, flows)
    receipts = json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8"))["items"]
    urls = {r["month"]: r["url"] for r in receipts if "month" in r}
    relative = OUT.relative_to(ROOT).as_posix()
    prev_report = local(previous["report"], "上一轮价格与成分风险复核")
    report = f"""**本轮得到的更具体结论：2021年1月的“M1同比上升＋中长期信贷多增＋指数降波”，没有形成三项独立的需求改善确认。** M1还受到春节跨月、持有人和同比基数影响；中长期多增与企事业贷款总额少增并存；指数降波也没有对应多数成分风险缓和。把这些过程连接起来，比把指标同向简单计票更接近实际经济含义。

本轮沿用既定2021年1月病例，只核实金额、统计范围和时序。原来的股票收益已经看过，以下属于历史机制解释，不是独立样本检验。

**一、先把“同比上升”与“当期余额扩张”分开。**

| 月份 | M1同比 | M1余额，万亿元 | M2同比 | M1−M2，百分点 |
|---|---:|---:|---:|---:|
| 2020年12月 | 8.6% | 62.56 | 10.1% | −1.5 |
| 2021年1月 | 14.7% | 62.56 | 9.4% | +5.3 |
| 2021年2月 | 7.4% | 59.35 | 10.1% | −2.7 |

2021年1月M1同比提高6.1个百分点，但按公布精度，余额与上月持平。两个62.56万亿元各有四舍五入误差，不能写成真实余额一分钱未变。同期M0增加约5300亿元，M1减去M0的非现金部分反而减少约5300亿元。这里的“非现金M1”不能等同非金融企业全部存款。[央行2020年12月报告]({urls['2020-12']})、[2021年1月报告]({urls['2021-01']})。

从2020年12月至2021年1月，按公布余额计算M1相对M2的同比对数变化，当期相对变化贡献 **{log.loc['2021-01'].current_relative_log_pp:.3f}**，去年同期基数贡献 **+{log.loc['2021-01'].base_relative_log_pp:.3f}**，合计 **+{log.loc['2021-01'].total_relative_growth_log_pp:.3f}个对数百分点**。改善并不是当月M1跑赢M2造成的。普通剪刀差本月增加6.8个百分点是另一种度量，不能与这些对数项相加，也不能把基数项直接认定为春节的因果贡献。

**二、春节时点通过存款持有人影响旧M1，企业存款多增也需要拆开。**

2020年春节在1月，2021年在2月，这一日历差异在2021年2月9日观察时已经公开。[2020年原定安排](https://app.www.gov.cn/govdata/gov/201911/21/451111/article.html)、[2021年安排](https://app.www.gov.cn/govdata/gov/202011/25/465322/article.html)。2020年原安排后来因疫情延长，此处只用它确认春节所在月份。

| 1月新增存款 | 2020年，亿元 | 2021年，亿元 | 跨公告差额，亿元 |
|---|---:|---:|---:|
| 非金融企业 | −16,100 | +9,484 | +25,584 |
| 住户 | +42,400 | +14,800 | −27,600 |
| 两者合计 | +26,300 | +24,284 | −2,016 |

企业存款同比多出的2.5584万亿元，伴随着住户存款同比少增2.76万亿元；两个持有人合起来没有表现为同样规模的新增存款扩张。这个结构与春节支付落在不同月份相容，但净额不能追踪同一笔工资，更不能把全部差异都归给春节。[央行2020年1月原文]({urls['2020-01']})及上述2021年1月原文。

央行在2022年1月报告中明确解释，节前集中发薪会使单位活期转到个人存款并压低当时口径的M1；当月M1同比−1.9%，剔除春节错时后约+2%。这是另一年的官方机制证据，不能把约2%移植给2021年，也不能把2022年的解释回填进2021年的信息集。[央行2022年1月报告]({urls['2022-01']})。

**三、中长期贷款多增，不能直接改写成企业融资总量全面加速。**

| 1月新增人民币贷款 | 2020年，亿元 | 2021年，亿元 | 跨公告差额，亿元 |
|---|---:|---:|---:|
| 企事业单位总额 | 28,600 | 25,500 | −3,100 |
| 其中：中长期 | 16,600 | 20,400 | +3,800 |
| 其中：短期 | 7,699 | 5,755 | −1,944 |
| 其中：票据融资 | +3,596 | −1,405 | −5,001 |
| 住户总额 | 6,341 | 12,700 | +6,359 |

这支持“期限结构向中长期偏移、住户增量更强”的描述。它仍可能包含真实项目融资改善，但并未证明经营周转、订单、投资和偿债能力同时改善。分项总额未必精确相加，报告中企事业单位包含非金融企业及机关团体；也没有逐笔证据证明某一笔票据被中长期贷款置换。数据来源为上述两年1月央行公告。

**四、用两个月观察检验解释，但严格区分后来才知道的事实。**

2021年2月M1同比回落至7.4%，并不意味着同期所有信贷需求都逆转。合并2021年1—2月单月公告，与2020年2月公告直接给出的同期累计相比：企事业中长期贷款31,400亿元，对照20,800亿元，多10,600亿元；企事业贷款总额37,500亿元，对照39,900亿元，少2,400亿元；住户贷款14,121亿元，对照2,209亿元，多11,912亿元。

企业存款两月合计由2020年的−13,300亿元到2021年的−14,716亿元，1月看起来巨大的企业存款同比多增没有持续为两月累计多增。两个春节月份合看有助于检查跨月解释，但不是季节调整，2020年疫情、经营和支付冲击仍未排除。[央行2020年2月累计与单月原文]({urls['2020-02']})、[2021年2月原文]({urls['2021-02']})。

2021年2月数据在 **3月10日16:30** 才公布，只能用于事后检验。本轮观察点仍是 **2月9日21:00**，不会用上述两月结果替当时的判断补答案。

**五、2025年新M1口径改变了同一种支付行为的统计表现。**

下面只是假设100元发生指定划转、其他条件不变的余额归属演示，不是实际资金流估计：

| 行为 | 旧口径M1变化 | 新口径M1变化 | M2变化 |
|---|---:|---:|---:|
| 单位活期转个人活期 | −100元 | 0元 | 0元 |
| 单位活期转个人定期 | −100元 | −100元 | 0元 |
| 单位活期提取现金 | 0元 | 0元 | 0元 |
| 新增银行贷款并留在单位活期 | +100元 | +100元 | +100元 |

新版M1已经包括个人活期和非银行支付机构客户备付金。因此同样的企业发薪，不一定再表现为M1流出；它还取决于接收后留在活期、转定期或发生其他支付。**这不表示春节影响消失，而是旧经验需要重新核对具体资金归属。**[央行货币定义](https://www.pbc.gov.cn/rmyh/109339/2025080818580470423/index.html)。

**六、把上游原因接回510300，而不是停在宏观解释。**

2021年2月9日，已有价格上涨、订单及市场内部证据也需要同时保留：

| 环节 | 当时已公开或可计算的事实 | 对解释的约束 |
|---|---|---|
| 货币 | 同比高增，公布的M1余额环比基本持平；春节月份错位 | 不能把同比跳升直接当成当期需求激增 |
| 信贷 | 企业中长期多增，但企事业贷款总额少增 | 确认期限结构变化，尚未确认所有企业融资扩张 |
| 订单 | 制造业新订单52.3，较上月回落1.3点 | 高于50的扩张水平与边际放缓同时存在 |
| 已有定价 | 510300此前60日已上涨{case['index_past60_return'] * 100:.2f}% | 数据改善不等于仍有等量未兑现上涨空间 |
| 内部风险 | 300只中仅{case['stocks_with_lower_downside']}只下行能量下降；参考权重平均个股下行能量反而上升{case['individual_weighted_down_energy_change'] * 100:.2f}% | 指数变稳不代表多数股票同步转稳 |

行情、订单和成分数据出处及覆盖限制见{prev_report}。1月金融数据2月9日16:00:20公布，当日收盘在公告之前形成，所以此前价格与波动不能称为对本条公告的反应。参考权重来自保存的历史快照，不能冒充实际ETF持仓，也没有全历史首版认证。按原来固定的月度观察口径，E0后续20日含分红毛收益−11.09%，延迟一天E1为−13.96%，这里只保留原结果，不从后来下跌反向指定因果。

**这个病例已经能够排除一种具体误读：三个表面同向的指标，不等于三个独立、尚未被价格反映的利好。** 后续真正要连接的是融资由谁取得、是否进入新增订单和经营现金、如何影响指数主要行业的利润，以及这些变化在价格里已经反映多少。期限结构、支付跨月和股票之间共同变动，都可能让表面信号与经济含义不同。

计算保存了7个月货币余额、5条单月或累计信贷存款记录、28项同区间比较及新旧口径演示。金额差采用对应原公告，不能冒称后来同版本官方同比：例如2021年1月人民币贷款原报同比多增2252亿元，而两个历史公告相减得到2400亿元；存款对应为6245亿元与6900亿元。差异原样保留，不能只用四舍五入解释。

本轮形成可复查的机制进展，没有新增模型、账户或交易规则，尚未证明新的成本后方向优势。后来的官方机制说明和新口径演示也不构成历史独立验证。

{local(relative + '/figures/同比高增背后的余额与信贷结构.png', '查看四张对照图')} · {local(relative + '/results/一月与两月累计_同区间结构对照.csv', '同区间金额底表')} · {local(relative + '/results/一月与两月_当期基数对数分解.csv', '余额与基数分解')} · {local(relative + '/results/100元划转_新旧口径示意.csv', '新旧口径计算示意')}
"""
    (OUT / REPORT).write_text(report, encoding="utf-8")
    save("result.json", {"at": stamp(), "study_id": "510300_HOLIDAY_MONEY_CREDIT_BRIDGE_V11",
        "status": "PUBLICATION_PENDING_SAVED_RECOMPUTATION", "continuation_classification": "PROGRESS",
        "main_conclusion": "春节时点、存款持有人和贷款期限结构共同影响指标含义；M1高增和中长期多增不能合并视为广泛需求同步增强。",
        "previous_result_sha256": digest(previous_path), "report": relative + "/" + REPORT,
        "report_sha256": digest(OUT / REPORT), "figure": relative + "/" + FIGURE,
        "figure_sha256": digest(OUT / FIGURE), "visual_review_pending": True,
        "new_models": 0, "new_accounts": 0, "independent_validation": False,
        "goal_achieved": False, "goal_status": "active"})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps({"报告": str(OUT / REPORT), "图": str(OUT / FIGURE), "待完成": "保存值复算及图形检查"}, ensure_ascii=False))


if __name__ == "__main__":
    publish()
