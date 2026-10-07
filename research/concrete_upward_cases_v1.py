"""用已保存上涨图谱解释具体价量和指标顺序；不重算图谱、拟合或运行策略。"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.point_first_passage_study_v1 import read, write_json, now

SOURCE = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
OUT = ROOT / "reports/research/510300_concrete_upward_cases_v1"
CASES = [(37, "2019春季持续上涨"), (42, "2020修复后加速上涨"),
         (55, "2024九月快速上涨"), (18, "2015六月短反弹失败对照")]
FEATURE_NAMES = {"daily_hist_rising": "日MACD柱首次回升", "daily_hist_positive": "日MACD首次转正",
                 "above_ema20": "价格首次站上EMA20", "up_volume_dominant": "五日涨跌量差首次转正",
                 "volume_expansion": "首次相对量≥1.5", "weekly_hist_positive": "前完整周MACD首次转正"}


def main():
    if (OUT / "summary.json").exists():
        raise ValueError("具体案例解释已经保存，不覆盖或作为新图谱重复计数。")
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(SOURCE / "results/features.parquet")
    episodes = pd.read_parquet(SOURCE / "results/上涨段全集.parquet")
    timing = pd.read_parquet(SOURCE / "results/首次确认时差.parquet")
    labels = pd.read_parquet(SOURCE / "results/全部原点结果标签.parquet")
    old_result = read(SOURCE / "result.json")
    chosen = episodes.loc[episodes.episode_id.isin([i for i, _ in CASES])].copy()
    if len(chosen) != 4 or not chosen.admitted.all():
        raise ValueError("指定的原图谱案例不存在或原状态不合格。")
    source_paths = [SOURCE / "result.json", SOURCE / "protocol.json", SOURCE / "results/features.parquet",
                    SOURCE / "results/上涨段全集.parquet", SOURCE / "results/首次确认时差.parquet",
                    SOURCE / "results/全部原点结果标签.parquet", SOURCE / "results/各阶段特征出现率.parquet",
                    SOURCE / "results/反推规则证据表.parquet", SOURCE / "results/价格相同条件下的增量.parquet",
                    Path(__file__)]
    manifest = {"at": now(), "user_direction": "应先解释具体上涨段的量价和指标，再反推能提前识别的点位",
                "role": "DESCRIPTIVE_REUSE_OF_SAVED_ATLAS_NOT_NEW_BACKTEST",
                "case_selection": "预先指定37/42/55三已知长期或爆发上涨及18已知短反弹，用于说明路径差异；不是随机抽样、样本胜率或优选规则。",
                "actual_old_atlas": old_result, "new_fits": 0, "new_accounts": 0, "new_outcome_labels": 0,
                "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]}
    write_json(OUT / "source_and_scope.json", manifest, exclusive=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    snapshots, summaries = [], []
    for episode_id, title in CASES:
        e = chosen.loc[chosen.episode_id.eq(episode_id)].iloc[0]
        events = timing.loc[timing.episode_id.eq(episode_id)].set_index("feature")
        bottom, peak = int(e.bottom_idx), int(e.peak_idx)
        left, right = max(0, bottom-20), min(len(d), int(e.confirm_down_idx)+16)
        part = d.iloc[left:right].copy()
        # EMA与收盘的前向经济平移口径一致；展示时同时减去当日平移量。
        raw_ema = part.ema20-part.cash_shift
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True,
                                 gridspec_kw={"height_ratios": [2.3, 1.3, 1.7, 1.2]})
        ax = axes[0]
        ax.plot(part.date, part.close, color="#273e51", label="原始日收盘", linewidth=1.5)
        ax.plot(part.date, raw_ema, color="#d1832e", label="EMA20（换回当日原价单位）", linewidth=1.25)
        ax.scatter([e.bottom_date, e.peak_date], d.close.iloc[[bottom, peak]], color="#737b83", s=35, marker="x", label="事后低/高点，非当时信号")
        ema_event = events.loc["above_ema20"]
        if ema_event.present_in_wave:
            i = int(ema_event.first_idx)
            ax.scatter([d.date.iloc[i]], [d.close.iloc[i]], color="#b93235", s=42, label="首次站上EMA20（当日可观察）", zorder=4)
        ax.set_title(f"{title}｜原图谱段{episode_id}｜事后低→高含息涨幅{e.gross_rise:.2%}")
        ax.set_ylabel("价格 / 元")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5)
        colours = np.where(part.return1.gt(0), "#bc4a43", "#338a79")
        axes[1].bar(part.date, part.relative_volume, width=.85, color=colours, alpha=.8, label="当日量 / 此前20日中位量")
        axes[1].axhline(1., color="#777", linewidth=.8)
        axes[1].set_ylabel("相对成交量")
        volume_axis = axes[1].twinx()
        volume_axis.plot(part.date, part.up_volume_balance5, color="#34495e", linewidth=1., label="五日涨跌量差 / 总量")
        volume_axis.set_ylim(-1.1, 1.1)
        volume_axis.set_ylabel("量的方向代理")
        handles, names = axes[1].get_legend_handles_labels()
        h2, n2 = volume_axis.get_legend_handles_labels()
        axes[1].legend(handles+h2, names+n2, loc="upper left", frameon=False, fontsize=8.5)
        axes[2].bar(part.date, part.daily_hist, width=.85, color=np.where(part.daily_hist.ge(0), "#bc4a43", "#338a79"), alpha=.65, label="日MACD柱：2×(DIF−DEA)")
        axes[2].plot(part.date, part.daily_dif, color="#2e6a9e", linewidth=1., label="日DIF")
        axes[2].plot(part.date, part.daily_dea, color="#bd872c", linewidth=1., label="日DEA")
        axes[2].step(part.date, part.weekly_hist, color="#6f657c", linewidth=1.1, where="post", label="此前完整周MACD柱")
        axes[2].axhline(0, color="#777", linewidth=.8)
        axes[2].set_ylabel("MACD / 元")
        axes[2].legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
        axes[3].plot(part.date, part.rv_ratio, color="#7a577d", linewidth=1.3, label="RV20 / 此前252日RV20中位数")
        axes[3].axhline(1., color="#777", linestyle="--", linewidth=.8)
        axes[3].set_ylabel("相对波动率")
        axes[3].legend(loc="upper left", frameon=False, fontsize=8.5)
        for axis in axes:
            axis.axvspan(e.bottom_date, e.peak_date, alpha=.07, color="#e0b64e")
            axis.axvline(e.confirm_up_date, color="#b18332", alpha=.6, linestyle="--", linewidth=.8)
            axis.grid(alpha=.18)
        axes[3].set_xlabel("真实交易日；黄色区域为事后分段，虚线为旧5%上涨确认日")
        fig.savefig(OUT / f"{episode_id}_{title}.png", dpi=160)
        plt.close(fig)

        dates = {}
        def add(date, name):
            if pd.notna(date):
                dates.setdefault(pd.Timestamp(date), []).append(name)
        add(e.bottom_date, "事后低点（只作对齐）")
        for feature, name in FEATURE_NAMES.items():
            row = events.loc[feature]
            if row.present_in_wave:
                add(row.first_date, name)
        add(e.confirm_up_date, "旧5%上涨确认")
        add(e.peak_date, "事后高点（只作对齐）")
        add(e.confirm_down_date, "旧5%回落确认")
        for date, names in sorted(dates.items()):
            x = d.loc[d.date.eq(date)].iloc[0]
            label = labels.loc[labels.date.eq(date)]
            snapshots.append({"episode_id": episode_id, "case": title, "date": date, "milestones": "；".join(names),
                              "raw_close": x.close, "ema20_in_raw_units": x.ema20-x.cash_shift,
                              "daily_macd_hist": x.daily_hist, "daily_dif": x.daily_dif, "daily_dea": x.daily_dea,
                              "prior_completed_week_hist": x.weekly_hist, "prior_completed_week_date": x.weekly_last_date,
                              "relative_volume": x.relative_volume, "up_volume_balance5": x.up_volume_balance5,
                              "rv_ratio": x.rv_ratio, "price_above_ema20": bool(x.above_ema20),
                              "breakout_previous_twenty_closes": bool(x.breakout20),
                              "old_twenty_day_net_outcome_label": float(label.net_reference_return.iloc[0]) if len(label) and label.status.iloc[0] == "MATURE" else np.nan,
                              "label_is_executable_account_profit": False})
        item = {"episode_id": episode_id, "case": title, "bottom": e.bottom_date, "peak": e.peak_date,
                "retrospective_gross_rise": e.gross_rise, "original_up_confirmation": e.confirm_up_date,
                "first_macd_rising": events.loc["daily_hist_rising", "first_date"],
                "first_price_above_ema20": events.loc["above_ema20", "first_date"],
                "first_week_macd_positive": events.loc["weekly_hist_positive", "first_date"]}
        summaries.append(item)
    pd.DataFrame(snapshots).to_csv(OUT / "关键日事前量价指标与原结果标签.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT / "四段路径与指标顺序.csv", index=False, encoding="utf-8-sig")
    lines = ["# 先解释具体上涨，再反推进场点位", "",
             "用户明确要求先解释具体上涨段的量价与指标，再反推当时能识别的点位。本报告复用旧49段图谱，形成三个上涨路径与一个短反弹的直观解释；不是新图谱、新策略回测或独立验证。", "",
             "旧图谱数据为2012-05-28至2026-09-16，合格49段、2826成熟原点。低/高点用于事后对齐，不能把最低点已知、最高点能卖当成策略。日MACD12/26/9柱为2×(DIF−DEA)，周线只取此前完整周；相对量对前20日中位量，RV20对此前252日中位数，五日涨跌量差是量价代理，不是主动买卖差或资金身份。", "",
             "## 1. 2019：先止跌修复，随后价格确认", "",
             "原段2019-01-02至04-19，事后含息涨幅39.13%。01-03日MACD负柱首次收敛，01-04相对量1.62倍且五日涨跌量差转正，01-08日柱转正，01-09价格站上EMA20，01-14前完整周MACD转正。", "",
             "01-09可以观察到价格确认与量的方向改善，当时相对量1.40倍、波动比0.82、周MACD仍负。价格较事后低点已经涨3.75%，因此不是‘最低点买入’。01-03只是下跌速度减缓；真正持续上涨不能由一个负柱变短推出。", "",
             "![2019上涨路径](37_2019春季持续上涨.png)", "",
             "## 2. 2020：高波动修复与后段加速", "",
             "原段2020-03-23至07-13，事后含息涨幅38.54%。03-24负柱开始收敛，03-25相对量1.67倍、五日涨跌量差转正，但波动比2.13、价格尚未站上EMA20、前完整周MACD仍负。04-01日柱转正，04-07才首次站上EMA20；前完整周MACD到06-08才转正。", "",
             "这段不能套用‘上涨前必须低波动’或‘必须先周线金叉’：修复早段仍高波动，周线转正时距事后低点已经涨14.07%。早段修复与后段加速是不同阶段，风险距离和可成交价格需要分别判断。这里仅解释原路径，不将04-07自动认定为高胜率入场。", "",
             "![2020上涨路径](42_2020修复后加速上涨.png)", "",
             "## 3. 2024：早段确认与晚段拥挤看起来都很强", "",
             "原段2024-09-13至10-08，事后含息涨幅36.68%。09-18日负柱开始收敛，09-19五日涨跌量差转正，09-20相对量1.97倍，09-23日柱转正；09-24价格站上EMA20且突破此前20个收盘，相对量3.34倍、五日涨跌量差1.00，波动比1.57，而前完整周MACD仍负。", "",
             "09-24是当时已经可观察的价格和量确认，较事后低点已涨6.16%。09-30周MACD才转正，价格距低点已经涨31.13%。10-08日/周MACD、量和涨幅更强，相对量6.85倍、波动比3.63，却处于这段事后高点。最新固定事件模型在10-08决定、10-09进场、10-10退出，实际扣费−5.32%。这个真实反例说明指标强度不能直接等同于进场质量，也不能仅用‘放量且MACD好’表达整个路径。", "",
             "![2024上涨路径](55_2024九月快速上涨.png)", "",
             "## 4. 2015：放量反弹和MACD回升也会失败", "",
             "原段2015-06-29至06-30，事后只持续一个上涨区间，但涨幅7.22%。06-30日MACD负柱收敛、相对量2.27倍、五日涨跌量差转正；价格仍在EMA20下，日DIF和此前完整周MACD仍负，波动比2.63。随后07-01再次下跌并触发旧5%回落确认。", "",
             "最新固定事件模型在06-30决定、07-01进场、07-03退出，实际扣费−7.23%。06-30的量与日柱改善确实存在，但没有建立价格持续修复。2020早段也曾价格在均线下且高波动，所以不能据这一个反例后验写成‘均线下永不买’或一个波动阈值。", "",
             "![2015短反弹](18_2015六月短反弹失败对照.png)", "",
             "## 这些案例支持怎样的研究顺序", "",
             "先分清三层事实：日MACD负柱收敛描述下跌减速；量与价格确认描述修复是否得到支持；距离前期价格和风险边界、次日开盘跳空及后续维持情况决定这一点位能否形成好的实际盈亏。周MACD适合提供此前阶段背景，在这些案例中等待转正会不同程度延后入场。后续维持只有逐日发生后才可用，不能回填到首次信号日。", "",
             "更值得先研究的不是指标越多越好，而是相同初始改善之后的顺序：价格是否随后确认、确认是否很快失败、以及到实际能进场时是否已过度延伸。具体日期可作为解释和候选研究原点，不能仅凭三个成功例和一个失败例得出胜率或盈亏比。", "",
             "旧完整图谱提供必要约束：49段中日MACD柱回升很常见，但在186个相同上穿EMA20价格事件上，单项日/周MACD、量、波动等九个附加状态没有一个通过原固定增量门。该失败保持；本报告没有重跑门或把案例改称新策略。旧顺序模型和已试过滤也需要按具体完整定义判断，不能用改名或阈值搜索恢复。", "",
             "下一步具体工作是把这些‘修复→确认→延伸/失败’的解释映射到旧186个价格事件和已有反例，先提出一个有明确当时可知证据、失效条件与可成交时点的完整机制。相同旧表达直接保留原裁决；实质不同的机制才单独固定，再评价完整账户。现在没有已冻结的新规则，不先选止盈止损、窗口或指标组合使案例看起来获利。", "",
             "[逐关键日原值和原结果标签](关键日事前量价指标与原结果标签.csv) · [四段顺序表](四段路径与指标顺序.csv) · [复用来源与范围](source_and_scope.json) · [原49段全集](../510300_upward_episode_anatomy_v1/results/上涨段全集.csv) · [原九项增量失败](../510300_upward_episode_anatomy_v1/results/反推规则证据表.csv) · [最近模型全部真实点位](../510300_point_first_passage_study_v1/results/实际进出点位与事前指标.csv)", "",
             "当前全部历史仍为开发资料，供应首版未认证；没有新的独立验证、收益/夏普提升或实盘授权。原实际模型金融裁决TECH.R158保持，TECH.R159只接受研究顺序与这些有限描述事实。", ""]
    (OUT / "具体上涨段解释与点位反推.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(OUT / "summary.json", {"at": now(), "technical_decision": "TECH.R159",
                                      "status": "ACCEPTED_DIRECTION_AND_SAVED_CASE_EXPLANATION_NO_NEW_STRATEGY",
                                      "cases": [i for i, _ in CASES], "key_date_snapshots": len(snapshots),
                                      "old_atlas_reused": True, "new_fits": 0, "new_accounts": 0, "new_labels": 0,
                                      "accepted_strategy": False, "returns_or_sharpe_improved": False,
                                      "next_experiment": "PROPOSED_MECHANISM_QUESTION_NOT_REGISTERED",
                                      "latest_actual_financial_decision": "TECH.R158", "goal_achieved": False}, exclusive=True)
    print("已复用原图谱生成四个具体路径图与关键日表；没有新图谱、拟合或策略回测。", flush=True)


if __name__ == "__main__":
    main()
