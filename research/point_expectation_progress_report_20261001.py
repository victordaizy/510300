"""汇总已有点位诊断、旧周期映射及入场前技术状态，不生成新交易规则。"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research import point_payoff_bottleneck_v1 as bottleneck
from research import historic_cycles_point_translation_v1 as old

OUT = ROOT / "reports/research/510300_point_expectation_progress_20261001"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"


def save_table(name, frame):
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT / f"{name}.parquet", index=False)


def state_table(points, features):
    columns = ["date", "ac", "ema20", "daily_hist", "daily_hist_positive", "daily_hist_rising", "above_ema20",
               "weekly_last_date", "weekly_hist", "weekly_hist_positive", "weekly_hist_rising", "weekly_available",
               "relative_volume", "up_volume_balance5", "rv20", "rv_ratio", "atr20", "past20_return"]
    result = points.merge(features[columns].rename(columns={"date": "feature_date"}),
                          left_on="entry_origin", right_on="feature_date", how="left", validate="many_to_one")
    assert len(result) == len(points) and result.feature_date.notna().all()
    assert result.feature_date.lt(result.entry_date).all()
    known_week = result.weekly_last_date.notna()
    assert result.loc[known_week, "weekly_last_date"].lt(result.loc[known_week, "feature_date"].dt.to_period("W-FRI").dt.start_time).all()
    result["future_outcome_is_label_only"] = True
    result["post_result_descriptive_analysis_only"] = True
    return result


def plot(comp, measurements):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1., 1.25]})
    fig.patch.set_facecolor("#f7f8fa")
    ax = axes[0]
    rules = list(bottleneck.NAMES)
    for i, rule in enumerate(rules):
        g = comp.loc[comp.rule.eq(rule)].set_index("scenario")
        ax.scatter(g.loc["ACTUAL_SAVED", "product"], i + .09, color="#163f69", s=68,
                   label="扣参考摩擦" if i == 0 else None, zorder=3)
        ax.scatter(g.loc["GROSS_SAME_POINTS", "product"], i - .09, color="#bd6e29", s=56,
                   marker="s", label="同点位无摩擦归因" if i == 0 else None, zorder=3)
        ax.text(g.loc["ACTUAL_SAVED", "product"] - .018, i + .3,
                f"{g.loc['ACTUAL_SAVED', 'product']:.3f}", ha="right", color="#163f69", fontsize=10)
    ax.set_yticks(range(len(rules)), [bottleneck.NAMES[x] for x in rules])
    ax.invert_yaxis()
    ax.set_xlim(.2, 1.15)
    ax.set_title("当前六组点位：去掉摩擦仍未达到门槛", loc="left", fontsize=14, pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.16), ncol=2, frameon=False, fontsize=10)
    ax = axes[1]
    selected = measurements.loc[measurements.basis.eq("FIXED_POINT")]
    pivot = selected.pivot(index="model", columns="period", values="product")
    for i, model in enumerate(old.NAMES):
        early, recent = pivot.loc[model, list(old.PERIODS)]
        ax.plot([early, recent], [i, i], color="#d3dce3", linewidth=2, zorder=1)
        ax.scatter(early, i, color="#163f69", s=48, label="2015—2019" if i == 0 else None, zorder=3)
        ax.scatter(recent, i, color="#bd6e29", marker="s", s=44,
                   label="2020—2026年8月14日" if i == 0 else None, zorder=3)
    ax.set_yticks(range(len(old.NAMES)), list(old.NAMES.values()))
    ax.invert_yaxis()
    ax.set_xlim(.72, 1.6)
    ax.set_title("旧日期固定份额映射：时期差异明显", loc="left", fontsize=14, pad=20)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.16), ncol=2, frameon=False, fontsize=10)
    for ax in axes:
        ax.axvline(1, color="#b72c35", linestyle="--", linewidth=1.5)
        ax.set_xlabel("胜率 p × 实际盈亏比 B；红线右侧才超过 1")
        ax.grid(axis="x", color="#e4e7eb", linewidth=.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_facecolor("#f7f8fa")
    fig.suptitle("510300 点位质量：样本数值与可用证据要分别判断", x=.03, y=.98, ha="left", fontsize=21, weight="bold")
    fig.text(.03, .035, "左：固定原入场与退出，仅比较成本口径。右：全部14套旧方案，625条完成记录只有79组不同日期组合，仍有持仓重叠。\n"
             "右图是旧日期的解释性映射，原退出可能依赖原账户路径；没有重新生成可执行规则，也没有独立验证。", fontsize=10, color="#566371")
    fig.subplots_adjust(left=.16, right=.975, bottom=.22, top=.86, wspace=.73)
    fig.savefig(OUT / "点位期望与跨期差异.png", dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(OUT / "点位期望与跨期差异.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    assert not OUT.exists(), "已有本轮交付，不能静默覆盖。"
    OUT.mkdir(parents=True)
    comp = pd.read_parquet(bottleneck.OUT / "results/固定归因比较.parquet")
    attribution = pd.read_parquet(bottleneck.OUT / "results/亏损路径与空间.parquet")
    geometry = pd.read_parquet(bottleneck.OUT / "results/逐笔收益来源.parquet")
    measurements = pd.read_parquet(old.OUT / "results/指标对照.parquet")
    sensitivity = pd.read_parquet(old.OUT / "results/集中度与描述区间.parquet")
    points = pd.read_parquet(old.OUT / "results/原周期与固定点位逐笔.parquet")
    annual = pd.read_parquet(old.OUT / "results/逐年次数.parquet")
    features = pd.read_parquet(old.OUT / "inputs/features.parquet")
    completed = points.loc[~points.terminal].copy()
    state = state_table(completed, features)
    save_table("全部旧点位的事前技术状态", state)
    save_table("全部十四方案点位指标", measurements)
    save_table("全部十四方案逐年次数", annual)
    save_table("六组点位固定归因", comp)
    qualified = measurements.loc[measurements.basis.eq("FIXED_POINT")].groupby("model").point_estimate_pass.all()
    leads = list(qualified.index[qualified])
    save_table("两条历史线索逐笔点位", state.loc[state.model.isin(leads)])
    pivot = comp.pivot(index="rule", columns="scenario", values="product")
    current = attribution.set_index("rule").loc["BOTH"]
    sub = geometry.loc[geometry.rule.eq("BOTH") & geometry.geometry_bin.eq("RR_1_TO_2")]
    share = sub.actual_recomputed_net.nlargest(2).sum() / sub.actual_recomputed_net.sum()
    assert len(sub) == 9
    lead_metrics = measurements.loc[measurements.basis.eq("FIXED_POINT") & measurements.model.isin(leads)].merge(sensitivity, on=["model", "period"])
    lines = ["# 510300点位研究：当前数学门槛与新增成果", "",
             "**当前结论：六组直接点位规则尚未达到p×实际盈亏比>1；旧十四方案的进出日期按固定份额映射后，有两条线索在两个历史时期的样本数值都超过1，但早期结果高度依赖同一笔2015年盈利，尚未形成已验证的可用策略。**", "",
             "## 当前要求与数学定义", "",
             "只研究510300标的的日线及已完成周线多空点位。交易次数是希望提高的软目标，不再要求每个自然年五笔；当前不做期权收益和完整账户夏普验收。", "",
             "p=实际盈利笔数/全部完成笔数；q=实际亏损笔数/全部完成笔数；B=平均正净收益率/平均负净收益率绝对值。先统一每笔回报分母并扣参考摩擦，再计算。用户要求pB>1，另报告平均净回报>0。标准亏损单位期望是pB−q，无平手时q=1−p；利润因子=pB/q。数学定义也可见[CME关于交易数学期望的课程](https://www.cmegroup.com/education/courses/trading-psychology/the-mathematics-of-trading-success)。", "",
             "例如60%胜率、实际B=1.5，pB=0.9，标准期望为0.5个平均亏损单位；它有正的样本期望，但仍不符合用户指定的更严格门槛。实际B不能由计划止盈2R代替。", "",
             "## 六组固定点位为何没有达标", "",
             "固定原入场、方向、份额、目标、失效和退出，逐笔重算475条含比较重复的记录、6426条持有期间收盘路径；回报复算最大误差为零。", "",
             "| 规则 | 实际pB | 同日期无摩擦pB | 原退出决定日收盘标记pB |", "|---|---:|---:|---:|"]
    for rule, name in bottleneck.NAMES.items():
        r = pivot.loc[rule]
        lines.append(f"| {name} | {r.ACTUAL_SAVED:.3f} | {r.GROSS_SAME_POINTS:.3f} | {r.DECISION_CLOSE_MARK:.3f} |")
    lines += ["", "无摩擦和决定日收盘均为解释性归因，不是可执行替代策略；也不把非线性的pB当成一定随成本单调变化的指标。六组在这两种归因下仍未超过1。", "",
              f"周线双向37笔中，{int(current.losses)}笔亏损，{int(current.losses_ever_positive)}笔曾有扣摩擦正浮盈，只有{int(current.losses_ever_1r)}笔曾在收盘达到计划1R。固定点位中{int(current.plan_rr_le1)}笔入场时计划净目标/计划净失效亏损不超过1，计划比中位数为{current.plan_rr_median:.3f}。这说明目标空间偏小和方向持续性不足都有迹象，不能把问题简单归结为退出慢。", "",
              f"次日开盘退出相对原决定日收盘标记，平均变化为{current.mean_exit_delay_effect:+.3%}，本组反而略有利。风险R单位下周线双向pB约为0.395，价格回报口径约为0.550，两者都失败，不能切换分母宣布达标。", "",
              f"事后计划比(1,2]子组虽然9笔的pB为1.433，但两笔最大盈利贡献净回报加总的{share:.1%}；该分组只作诊断，没有升级成过滤条件。其他分箱结果均保存在原诊断报告。", "",
              "## 旧日期重算得到的两条线索", "",
              "原十四方案全部纳入，对两个保存时期分别计算。原周期收益包含加减仓，分母为累计买入支出；本次另按首笔入场开盘到最终退出开盘，以约十万元名义的固定份额持有，净收益除初始原价名义金额。双边参考佣金各万四、至少5元，双边滑点各千一、0.001元不利取整，计股息。", "",
              "原十四方案共625条完成记录，另有28条期末强平记录单列排除；625条只有79组不同进出日期组合，不同组合仍会重叠。它们不是625个独立观察或十四个独立策略验证。", "",
              "全部十四套在2020年以后固定点位pB为1.142—1.423。以下两套在2015—2019也略大于1，其余较早时期均低于1。", "",
              "| 历史线索 | 时期 | 完成笔数 | 胜率 | 实际B | pB | 平均净回报/笔 | 完整年均次数 | 去最大盈利后pB |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in lead_metrics.itertuples():
        lines.append(f"| {r.name} | {old.PERIOD_NAMES[r.period]} | {r.n} | {r.p:.2%} | {r.b:.3f} | {r.product:.3f} | {r.mean:.2%} | {r.full_year_average:.2f} | {r.product_without_largest_winner:.3f} |")
    lines += ["", "两者共享2015-02-11至2015-05-15这一笔，固定份额净回报36.81%，分别贡献较早时期净回报加总的85.83%和74.35%。这里的‘加总’是各笔参考收益率之和，不是资金账户累计收益。删除最大盈利只是依赖程度诊断，不是事后排除交易。", "",
              "核心辅助回撤门槛的完整年度次数为：2015至2025依次5、2、8、3、4、4、5、2、2、7、5；2026截至8月14日4笔。相关方向确认辅助依次4、2、3、2、4、4、5、2、0、8、5；2026同期4笔。频率已经完整列出，但不会为了增加次数降低质量门槛。", "",
              "这些线索来自日线趋势与波动率、收益连续成段、市场回撤或相关方向确认，以及已保存的原退出模型。原核心还包含旧月度训练退出与持仓路径；本次只读取保存账簿。旧末次退出可能依赖原账户路径，固定份额映射并没有证明能在新持仓系统中逐日复现相同退出。原方案的关闭结论保留。", "",
              "年份区块重采样仅在有完成点位的年份之间重采样，零笔年份仍在年度频率表完整保留。区间只描述历史样本，不检验零交易年对使用体验的影响，也未校正长期反复研究、十四方案选择或期限结束强平。早期两条线索的描述区间均跨1，尚无跨时期稳定性的充分证据。", "",
              "## 已保存的事前技术状态", "",
              "本轮保存了全部625条完成记录的入场决定日状态，包括收盘相对EMA20、日MACD柱方向、此前完整周的MACD、相对成交量、五日涨跌方向量平衡、20日波动率及相对过去一年波动位置。所有特征日均早于实际入场，周特征严格来自此前完整自然周。未来点位回报另列为结果标签，未用于拟合、选择指标阈值或新建过滤。", "",
              "这一步为用户提出的‘从涨跌解释量价，再反推策略’提供完整对照。能解释一笔大行情并不证明当时能选中它；下一步首先需要将已知入场条件与成功、失败的完整样本逐项对照，并把原依赖持仓路径的退出讲清楚。只有固定事前规则并验证其增量后，才考虑新候选，不能按本次最漂亮子组回填规则。", "",
              "## 可复查文件", "",
              "- `六组点位固定归因.csv`：原成本、无摩擦、决定收盘、风险单位的完整比较。",
              "- `全部十四方案点位指标.csv`：全部十四方案、两个时期、两种口径。",
              "- `全部十四方案逐年次数.csv`：保留零交易年及2026部分年份。",
              "- `全部旧点位的事前技术状态.csv`：625条含跨模型重复的完整特征与结果标签。",
              "- `两条历史线索逐笔点位.csv`：两条数值达标线索的全部完成记录，含同源重复。",
              "- `点位期望与跨期差异.png`及SVG：完整比较图。", "",
              "六组新点位数据截至2026-09-16，旧十四方案账簿截至2026-08-14。本轮没有新增历史策略回测、委托或期权收益；六项必要数学与股息边界测试通过，原周期收益独立复算最大误差约2.1e-16，原账簿开盘价与日线快照最大差异为零。整体目标仍为进行中，未宣布实现。", ""]
    (OUT / "本轮新增研究结论.md").write_text("\n".join(lines), encoding="utf-8")
    plot(comp, measurements)
    sources = [bottleneck.OUT / "result.json", bottleneck.OUT / "verification.json", old.OUT / "summary.json",
               old.OUT / "verification.json", old.OUT / "protocol.json", CONTEXT / "active_goal_effective_requirements.json",
               ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md", ROOT / "docs/510300_RETURN_RUNS_STATE_V1.md",
               ROOT / "docs/510300_CORE_AUXILIARY_DRAWNDOWN_GATE_NEXT_20260912.md",
               ROOT / "research/return_confirmation_auxiliary_batch_inputs_v1.py", Path(__file__)]
    common.save_json(OUT / "sources.json", {"at": common.now(), "files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": common.digest(p)} for p in sources],
                     "external_math_reference": "https://www.cmegroup.com/education/courses/trading-psychology/the-mathematics-of-trading-success"})
    summary = {"at": common.now(), "study": "510300_POINT_EXPECTATION_PROGRESS_20261001", "status": "POINT_RESEARCH_PROGRESS_WITH_TWO_HISTORICAL_NUMERIC_LEADS",
               "current_six_rules_passing": 0, "historic_fixed_point_numeric_leads_both_periods": leads,
               "historical_point_mapping_is_new_executable_rule": False, "entry_state_rows_with_comparison_duplicates": len(state),
               "new_strategy_accounts": 0, "tests_passed_this_round": 6, "goal_achieved": False, "orders_authorized": False,
               "next_question": "将旧状态组合在入场前的已知条件和原路径退出讲清楚，并比较全体成功失败点位；不按单一2015盈利或事后分箱拟合过滤。"}
    common.save_json(OUT / "summary.json", summary)
    common.save_json(OUT / "delivery_receipt.json", {"at": common.now(), "status": "SAVED_RESULTS_AND_PRE_ENTRY_FEATURE_CLOCKS_CHECKED",
                     "artifacts": [{"path": p.name, "sha256": common.digest(p)} for p in sorted(OUT.iterdir()) if p.is_file()],
                     "independent_strategy_validation": False, "goal_achieved": False})
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
