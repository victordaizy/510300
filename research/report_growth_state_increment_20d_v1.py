"""展示完整冻结结果，记录误差集中与不同政策的时钟，不修改模型。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from growth_state_increment_20d_v1 import ROOT, OUT, REFRAME, csv, save, sha, now

BLUE, TEAL, RED, GREY, GOLD = "#225e91", "#147f77", "#b54d60", "#64768a", "#b37b29"


def policy_clock_cases() -> pd.DataFrame:
    receipts = json.loads((ROOT / "reports/research/510300_post_information_capital_adjustment_v1/evidence/policy_context_source_receipts.json").read_text(encoding="utf-8"))
    source_by_date = {r["date"]: r for r in receipts}
    facts = [
        {"event_date": "2024-05-17", "family": "地产金融", "event": "保障性住房再贷款", "source_available_upper": "2024-05-17T18:22:00+08:00",
         "precision": "SECOND", "announced_amount": 3000, "amount_unit": "亿元再贷款额度", "separate_expected_bank_lending": 5000,
         "economic_role": "支持银行向合格收购主体融资；央行额度和预计带动银行贷款不能相加算作购买股票资金。",
         "not_equal_to": "已执行贷款或已成交收储规模", "source_checks": ["3000亿元保障性住房再贷款", "2024-05-1718:22:00"]},
        {"event_date": "2024-11-08", "family": "财政化债", "event": "隐性债务置换限额", "source_available_upper": "2024-11-09T23:59:59+08:00",
         "precision": "DATE_OF_OFFICIAL_TRANSCRIPT", "announced_amount": 60000, "amount_unit": "亿元债务限额", "annual_2024_2026_amount": 20000,
         "economic_role": "置换存量债务、缓解利息与现金流约束；分三年安排。", "not_equal_to": "六万亿元当期新增最终需求",
         "source_checks": ["2024年11月9日", "每年2万亿元", "6万亿元债务限额"]},
        {"event_date": "2025-05-12", "family": "贸易政策", "event": "日内瓦联合声明相关安排", "source_available_upper": "2025-05-12T15:00:59+08:00",
         "precision": "MINUTE_UPPER_BOUND", "suspended_tariff_pp": 24, "suspension_days": 90, "remaining_reciprocal_tariff_pp": 10,
         "economic_role": "调整文件所列对等及反制关税安排，影响外需、成本与风险溢价。",
         "not_equal_to": "所有商品总税率均为10%或永久取消全部关税", "source_checks": ["2025-05-1215:00", "90天", "剩余10%"]},
    ]
    dates = pd.DatetimeIndex(pd.read_parquet(OUT / "inputs/market.parquet", columns=["date"]).date)
    verified = []
    for item in facts:
        receipt = source_by_date[item["event_date"]]
        raw = ROOT / receipt["path"]
        if sha(raw) != receipt["sha256"]:
            raise ValueError("政策归档原文身份不同")
        text = "".join(BeautifulSoup(raw.read_bytes(), "html.parser").get_text("", strip=True).split())
        if not all(s in text for s in item["source_checks"]):
            raise ValueError("政策原文关键事实未定位：" + item["event"])
        known = pd.Timestamp(item["source_available_upper"]).tz_localize(None)
        pos = int(np.searchsorted((dates + pd.Timedelta(hours=15)).to_numpy(), np.datetime64(known), side="left"))
        item.update(first_observation_close=str(dates[pos].date()), next_open_execution=str(dates[pos + 1].date()),
                    source_url=receipt["url"], source_sha256=receipt["sha256"], raw_path=receipt["path"],
                    expectation=None, surprise=None, scope="PREVIOUSLY_OBSERVED_CASE_NOT_POLICY_UNIVERSE_OR_RETURN_TEST")
        verified.append({**receipt, "identity_matched_now": True})
    save(OUT / "evidence/三类政策原文核对.json", {"checked_at": now(), "facts": facts, "source_receipts": verified,
        "clock_note": "按15时截止演示保守可用时钟；15:00分钟时间戳按15:00:59，不能冒充15:00:00前已完整取得。财政新闻次日发布的保守上界不代表真实首发是次日。",
        "not_used_in_growth_models": True, "expectation_status": "缺少有历史时间证据的预期；不编码利好强度或超预期程度。"})
    flat = pd.DataFrame([{k: v for k, v in x.items() if k != "source_checks"} for x in facts])
    csv(flat, OUT / "results/地产财政贸易政策_事实与时钟.csv")
    return flat


def diagnostics() -> dict:
    primary = pd.read_csv(OUT / "results/月度主要评价及损失.csv")
    train = pd.read_csv(OUT / "results/全部月度训练原点.csv").set_index("origin_id")
    models = {m["fit_id"]: m for m in json.loads((OUT / "results/固定模型与训练集合.json").read_text(encoding="utf-8"))}
    primary["return_excess_error"] = primary.loss_P1 - primary.loss_P0
    primary["risk_excess_loss"] = primary.loss_R1 - primary.loss_R0
    for feature in ("growth_level", "growth_change"):
        lows, highs = [], []
        for row in primary.itertuples():
            sample = train.loc[models[row.fit_P1]["training_ids"], feature]
            lows.append(float(sample.min()))
            highs.append(float(sample.max()))
        primary[feature + "_train_min"] = lows
        primary[feature + "_train_max"] = highs
        primary[feature + "_outside_train_range"] = (primary[feature] < lows) | (primary[feature] > highs)
    csv(primary, OUT / "results/事后误差集中与外推诊断_全部月份.csv")
    worst = primary.loc[primary.return_excess_error.idxmax()]
    result = {"status": "POST_HOC_EXPLANATION_NO_NEW_SELECTION", "worst_return_excess_error_origin": worst.date,
        "worst_reference_period": worst.reference_period, "worst_point_share_of_positive_excess_errors": float(worst.return_excess_error / primary.return_excess_error.clip(lower=0).sum()),
        "return_MSE_change_vs_price": float(primary.loss_P1.mean() / primary.loss_P0.mean() - 1),
        "extreme_case": {c: float(worst[c]) for c in ["growth_level", "growth_change", "growth_level_train_min", "growth_level_train_max", "growth_change_train_min", "growth_change_train_max", "return_20d", "prediction_P0", "prediction_P1", "downside_variance_20d", "prediction_R0", "prediction_R1"]},
        "year_definition": "按观察原点所在日历年，可能包含上一年所属月，所以个别年份可有13次发布。主要评价所属月份2018-03至2026-07连续101个月。",
        "new_fits": 0, "removed_events": 0, "new_thresholds": 0}
    save(OUT / "results/事后诊断摘要.json", result)
    return result


def draw() -> None:
    plt.rcParams.update({"font.family": FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(), "axes.unicode_minus": False,
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#bac3cc", "text.color": "#25364a",
        "axes.labelcolor": "#25364a", "xtick.color": GREY, "ytick.color": GREY, "figure.facecolor": "white", "svg.hashsalt": "growth_state20d_v1"})
    primary = pd.read_csv(OUT / "results/月度主要评价及损失.csv", parse_dates=["date"])
    metrics = pd.read_csv(OUT / "results/八项基准汇总.csv").set_index("model")
    yearly = pd.read_csv(OUT / "results/逐年预测贡献.csv")
    summary = json.loads((OUT / "results/summary.json").read_text(encoding="utf-8"))

    def finish(fig, name: str) -> None:
        fig.savefig(OUT / "figures" / (name + ".png"), dpi=150, facecolor="white")
        fig.savefig(OUT / "figures" / (name + ".svg"), metadata={"Date": None})
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(15.7, 6.8), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(left=.13, right=.95, top=.78, bottom=.23, wspace=.47)
    fig.suptitle("二十日增长状态实验：收益与风险分别检验，均未建立增量", x=.06, ha="left", y=.964, fontsize=20, fontweight="bold")
    fig.text(.06, .881, "101 个月主要评价；比较同一批时点。没有把周度填充变成更多独立宏观样本。", color=GREY)
    keys = ["P0", "P1", "MEAN", "ZERO"]
    labels = ["价格基准", "+ 新订单状态", "成熟训练均值", "零收益预测"]
    colors = [BLUE, RED, TEAL, GREY]
    values = metrics.loc[keys, "rmse_pp"].to_numpy()
    axs[0].barh(np.arange(4), values, color=colors, height=.55)
    axs[0].set_yticks(np.arange(4), labels)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0, 6.8)
    axs[0].set_xlabel("收益 RMSE（百分点，越低越好）")
    for i, value in enumerate(values):
        axs[0].text(value + .08, i, f"{value:.3f}", va="center", color=colors[i])
    gains = [r for r in summary["comparisons"] if r["candidate"] == "R1"]
    for i, r in enumerate(gains):
        value = r["mean_improvement"]
        axs[1].errorbar(value, i, xerr=[[value - r["two_sided_90pct_lower"]], [r["two_sided_90pct_upper"] - value]], fmt="o", color=[RED, GREY, TEAL][i], capsize=5, markersize=6)
        axs[1].annotate(f"均值 {value:+.3f}", xy=(.98, i), xycoords=("axes fraction", "data"), xytext=(0, -18), textcoords="offset points", ha="right", fontsize=10)
    axs[1].set_yticks([0, 1, 2], ["相对价格风险模型", "相对成熟训练均值", "相对过去20日风险"])
    axs[1].set_ylim(2.45, -.45)
    axs[1].axvline(0, color=GREY, ls="--", lw=1)
    axs[1].set_xlim(-.15, 1.52)
    axs[1].set_xlabel("风险损失改善：正值表示新增模型更好")
    axs[1].set_title("下行风险：平均 QLIKE 差及 90% 双侧区间", fontsize=12)
    for ax in axs:
        ax.grid(axis="x", color="#e8edf1")
        ax.set_axisbelow(True)
    fig.text(.06, .121, "收益模型 MSE 比价格基准增加 12.47%；风险模型虽优于简单历史风险，却未优于价格风险模型和全部预定基准。", fontsize=10, color=GREY)
    fig.text(.06, .062, "两个模块按各自冻结条件停止；20万元 / 2万元账户均未运行。该结论只适用于本次新订单水平与月度变化表达。", fontsize=10, color=GREY)
    finish(fig, "增长状态_收益与风险的全部基准")

    fig, axs = plt.subplots(3, 1, figsize=(15.6, 11.4), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.2, 1.]})
    fig.subplots_adjust(left=.08, right=.975, top=.86, bottom=.15, hspace=.26)
    fig.suptitle("把全部预测画出来：极端经济状态暴露了模型外推问题", x=.08, ha="left", y=.975, fontsize=20, fontweight="bold")
    fig.text(.08, .929, "每个点是一份月度新信息后的判断；所有收益均从下一开盘开始，不含此前行情。", color=GREY)
    axs[0].plot(primary.date, primary.return_20d * 100, color=GREY, lw=1, marker=".", ms=4, label="随后20日实际收益")
    axs[0].plot(primary.date, primary.prediction_P0 * 100, color=BLUE, lw=1.1, label="价格预测")
    axs[0].plot(primary.date, primary.prediction_P1 * 100, color=RED, lw=1.1, label="加入增长状态")
    axs[0].axhline(0, color=GREY, lw=.7)
    axs[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=10)
    axs[0].set_ylabel("二十日收益（%）")
    extreme = primary[primary.reference_period == "2020-02"].iloc[0]
    axs[0].set_ylim(-16, 30)
    axs[0].annotate("2020-02 新订单 29.3\n新增模型预测 +13.09%\n随后实际 −10.95%", xy=(extreme.date, extreme.prediction_P1 * 100), xytext=(pd.Timestamp("2018-06-01"), 23), fontsize=9, color=RED, arrowprops={"arrowstyle": "->", "color": RED})
    for column, label, color in [("downside_variance_20d", "实际下行方差", GREY), ("prediction_R0", "价格风险预测", BLUE), ("prediction_R1", "加入增长状态", RED)]:
        axs[1].plot(primary.date, primary[column], color=color, lw=1.1, label=label)
    axs[1].set_yscale("log")
    axs[1].set_ylabel("下行方差\n（对数刻度）")
    axs[1].legend(loc="upper right", frameon=False, ncol=3, fontsize=10)
    axs[2].step(primary.date, primary.growth_level + 50, where="post", color=TEAL, lw=1.2)
    axs[2].axhline(50, color=GREY, ls="--", lw=.8)
    axs[2].set_ylabel("当次制造业\n新订单指数")
    axs[2].xaxis.set_major_locator(mdates.YearLocator())
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axs:
        ax.grid(axis="y", color="#e8edf1")
    fig.text(.08, .084, "极端月是事后误差诊断：保留该月、全部训练集合及原判据；没有删除疫情月份，也没有重新缩尾或调参。", fontsize=10, color=GREY)
    fig.text(.08, .04, "历史已用于其他研究；本次不是独立未见样本。505次周度及公告复核预测、101个月主要评价均完整保存。", fontsize=10, color=GREY)
    finish(fig, "全部月度预测_极端状态与实际走势")

    fig, axs = plt.subplots(2, 1, figsize=(13.5, 7.8), sharex=True)
    fig.subplots_adjust(left=.10, right=.97, top=.80, bottom=.17, hspace=.22)
    fig.suptitle("逐年贡献：不能只看某几个月预测正确", x=.08, ha="left", y=.96, fontsize=20, fontweight="bold")
    fig.text(.08, .884, "柱子高于零表示加入增长状态有改善；年份按观察日归属。没有删除任何事件。", color=GREY)
    axs[0].bar(yearly.year, yearly.return_improvement * 10000, color=[TEAL if x > 0 else RED for x in yearly.return_improvement])
    axs[0].set_ylabel("收益MSE改善\n（百分点平方）")
    axs[1].bar(yearly.year, yearly.risk_improvement, color=[TEAL if x > 0 else RED for x in yearly.risk_improvement])
    axs[1].set_ylabel("下行风险\nQLIKE损失改善")
    axs[1].set_xticks(yearly.year)
    for ax in axs:
        ax.axhline(0, color=GREY, lw=.8)
        ax.grid(axis="y", color="#e8edf1")
        ax.set_axisbelow(True)
    fig.text(.08, .065, "收益误差受到2020年极端状态的明显影响；下行风险相对价格模型在各观察年份均未改善。", fontsize=10, color=GREY)
    finish(fig, "逐年增量_全部年份")


def report(diag: dict, policy: pd.DataFrame) -> None:
    summary = json.loads((OUT / "results/summary.json").read_text(encoding="utf-8"))
    metrics = {r["model"]: r for r in summary["metrics"]}
    for name in ("新研究立项.md", "用户需求与本轮边界.md"):
        shutil.copy2(REFRAME / name, OUT / "evidence" / ("范围重订阶段_" + name))
    text = f"""# 增长状态的二十日收益与下行风险增量

**本次具体实现没有通过：加入制造业新订单相对50的位置及月度变化，收益MSE相对价格基准增加{diag['return_MSE_change_vs_price']:.2%}；下行风险模型也未通过独立判据。两个模块分别冻结停止，不运行账户或调整卖点。这个结果不涵盖财政、地产、贸易等政策作用，也不是对全部宏观研究的否定。**

这是上一阶段新立项的第一项独立子实验。它使用已公布经济状态，不包含合格的PMI市场共识，不能解读为“PMI超预期失效”。历史数据此前用于其他宏观与价格研究，独立前向事件仍为0。

## 样本、模型与真正的更新

保留139个月制造业新订单，首月无月度变化，因此有138个月训练原点。36个已成熟月度训练样本后开始评价，主要评价为2018-03至2026-07所属月，共101个月；观察日从2018-04-02至2026-07-31。训练只纳入目标终点严格早于当前判断的月度标签；20日收益从次日开盘计算，并包含应得分红现金权利。

按周末交易日与新订单公告后的首个可用收盘形成共同复核日历，共505次可评价预测。新增691组标签包含训练期与评价期的周度/公告原点；不能将691或505当作独立宏观样本。主要统计始终按101个月，连续6个月区块、10000次固定抽样。

收益模型P0使用过去20日收益和波动，P1再加新订单水平与月度变化。风险模型R0使用过去20日收益和下行方差，R1再加相同增长信息。全部采用固定惩罚1的训练内标准化岭回归，共408次唯一训练集合拟合，没有参数搜索。风险预测在对数目标上训练并作训练内均值修正；其判据是预先指定的QLIKE，而非收益MSE。

## 全部基准

| 收益预测 | 二十日收益RMSE（百分点） |
|---|---:|
| P0：价格基准 | {metrics['P0']['rmse_pp']:.3f} |
| P1：加增长状态 | {metrics['P1']['rmse_pp']:.3f} |
| 成熟训练均值 | {metrics['MEAN']['rmse_pp']:.3f} |
| 零收益预测 | {metrics['ZERO']['rmse_pp']:.3f} |

收益增量P0损失减P1损失为-0.00038949，90%双侧区间约[-0.00122393, 0.00007976]，仍跨零。前后两半均未改善。不能据此断言任何增长变量都有稳定负作用。

| 下行风险预测 | 平均QLIKE（越低越好） |
|---|---:|
| R0：价格风险模型 | {metrics['R0']['mean_loss']:.6f} |
| R1：加增长状态 | {metrics['R1']['mean_loss']:.6f} |
| 成熟训练下行方差均值 | {metrics['RMEAN']['mean_loss']:.6f} |
| 过去20日下行方差 | {metrics['DOWN20']['mean_loss']:.6f} |

QLIKE可以为负，其绝对数不是收益；应比较相同样本的损失差。R1相对R0的损失改善为-0.031985，90%双侧区间[-0.058393,-0.013050]。R1优于简单历史下行方差，但没有优于全部预定基准；不能只公布这一项胜出对照。两个研究目标分别裁决，没有用风险结果救收益模型。

![全部基准](figures/增长状态_收益与风险的全部基准.png)

## 为什么不能只数预测方向

公告当天不是常规周度复核日的75次事件，固定相同训练系数、当前价格输入、起点与20日终点，只比较采用最新增长值或上次周度复核时的增长值。收益预测有44次更新后更准，但平均平方误差反而增加0.00037712；风险有40次更准，平均QLIKE改善0.019255。这是预先指定的刷新诊断，没有构成独立账户或证明稳定可交易价值。

最新与旧增长输入使20万元的做多门槛判断改变12次。加入增长信息相对价格基准，在101个月主要评价中使20万元规则改变13次，2万元规则改变12次；这些只是根据预测和估计压力成本推导的规则差异，账户没有运行，不能填零收益或零夏普。

## 极端状态与误差集中

2020-02所属月新订单为29.3、较上月下降22.1，明显超出当时59个成熟训练月份的水平范围48.6—54.8、月度变化范围-1.9—2.8。新增收益模型在2020-03-02观察后预测+13.09%，随后20日实际为-10.95%；价格基准预测为+1.43%。风险模型同时把下行方差预测放大到约0.00746，实际约0.000280。

该月占全部正的新增收益误差约{diag['worst_point_share_of_positive_excess_errors']:.2%}，说明线性/对数线性表达在极端增长状态下外推不可靠。这是看到结果后的解释，已经单列“事后诊断”。没有删掉该月、改变样本起点、缩尾或改惩罚。单一失败模型无法证明所有经济信息无用；它也不能因为找到解释而改判通过。

![全部月度路径](figures/全部月度预测_极端状态与实际走势.png)

## 政策主线继续按不同机制记录

同步核对了三类政策的官方原文，并保存金额用途、时间精度、保守可用时点与下一可执行日：地产再贷款是融资额度与预计银行贷款的关系，财政化债是置换与利息/现金流约束，贸易声明是部分关税安排及有效期。它们不按统一“利好分”相加。

- [人民银行2024-05-17发布会](https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212554091417/index.html)：3000亿元再贷款与预计带动5000亿元银行贷款分列，不能相加成为流入股票的资金。
- [财政部2024-11-09发布的11月8日实录](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202411/t20241109_3947230.htm)：6万亿元限额分三年安排，不等于当期新增最终需求。
- [商务部2025-05-12说明](https://www.mofcom.gov.cn/syxwfb/art/2025/art_9748270381e54001bc291213ec6ee778.html)：文件所列部分关税暂停90天，与全部关税永久取消不同。15:00分钟时间戳按15:00:59取上界，在15时截止规则下不能假设已提前读完。

这三条仍是历史案例，尚未构成完整政策母集，也没有准入的事前共识。因此未进入增长模型，未计算政策收益贡献，更不表示所有政策信息已补齐。原货币五日失败、NBS固定五分钟失败和85/15终止均保持。

## 实现衔接、交付和下一步

初次执行在读取新标签前发现旧日历只到2026-08-14，行情到2026-09-11。用既有、已核对的上交所2026年度日历补足20个日期，重叠区间一致；原冻结文件完整保留，补充日历与修正代码另有回执。统计协议、模型和判据未改。恢复后只完成一轮真实标签和模型检验。

账户状态：P1与R1均为NOT_RUN_OWN_PREDICTION_GATE，20万元主账户与2万元成本对照、固定/回撤退出比较均未运行。年化10%和净夏普1.2未建立。下一步应继续独立政策链及其条件作用，不能将本次PMI失败改为别的窗口或反向交易来追求过线。
"""
    (OUT / "研究报告.md").write_text(text, encoding="utf-8")
    save(OUT / "evidence/研究与诊断计数.json", {"one_successful_label_and_model_run": True, "initial_calendar_admission_failure": True,
        "new_20d_labeled_origins": 691, "new_model_fits": 408, "new_accounts": 0, "diagnostic_new_fits": 0,
        "price_data_end": "2026-09-11", "decision_data_end": "2026-07-31", "independent_forward_events": 0,
        "policy_cases": len(policy), "whole_macro_program_complete": False})


if __name__ == "__main__":
    policy = policy_clock_cases()
    diagnostic = diagnostics()
    draw()
    report(diagnostic, policy)
    print("完整基准、全部月份图、误差诊断及三类政策时钟已整理；未改动冻结模型或结果。")
