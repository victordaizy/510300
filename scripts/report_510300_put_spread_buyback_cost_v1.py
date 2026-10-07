"""只从保存预测及账本解释十日买回成本实验。"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_put_spread_buyback_cost_daily_v1"


def main():
    r = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    m = {x["policy"]: x for x in json.loads((OUT / "results/account_metrics.json").read_text(encoding="utf-8")) if x["scenario"] == "STRESS"}
    pm = {x["model"]: x for x in r["prediction_metrics"]["models"]}
    w = pd.read_parquet(OUT / "results/all_rolling_two_years.parquet")
    p = r["primary"]
    names = {"MATCHED_VARIANCE": "同覆盖原方差定价", "EMPIRICAL": "历史平均买回成本", "PRICE": "价格条件买回成本", "MACRO": "价格加宏观买回成本"}
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for policy, color in zip(names, ["#88919B", "#298B7E", "#2874B8", "#CC6235"]):
        d = pd.read_parquet(OUT / "accounts/STRESS" / policy / "ledger.parquet")
        axes[0].plot(d.date, d.equity / 10000, label=names[policy], color=color, lw=1.5)
        axes[1].plot(d.date, d.drawdown * 100, color=color, lw=1.1)
    axes[0].set_title("固定十日买回成本预测：压力成本下完整账户")
    axes[0].set_ylabel("账户权益（万元）")
    axes[1].set_ylabel("回撤（%）")
    axes[0].axhline(20, ls="--", color="#555555", lw=.7)
    axes[0].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.1, .01, "选腿、持有期、成本与风险规则保持；两年每日更新。开发历史，尚无独立验证。", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(OUT / "压力账户净值与回撤.png", dpi=160)
    plt.close(fig)
    price_improvement = 1 - pm["PRICE"]["MSE"] / pm["EMPIRICAL"]["MSE"]
    macro_worse = pm["MACRO"]["MSE"] / pm["PRICE"]["MSE"] - 1
    lines = [
        f"本轮直接预测同一认沽价差十日后的压力买回成本，完成{r['new_daily_estimates']:,}个逐日估计，其中{r['new_ridge_fits']:,}次岭回归，以及8条完整账户路径。没有找到符合净夏普1.2、年化10%、最大回撤10%的策略。",
        "",
        f"主压力账户期末权益{p['ending_equity']:,.2f}元，净夏普{p['net_sharpe']:.3f}，年化{p['annualized_return']:.2%}，最大回撤{p['max_drawdown']:.2%}。57个完成周期的胜率{p['win_rate']:.2%}，平均盈利{p['average_win']:.2f}元、平均亏损{p['average_loss']:.2f}元；末端仍有持仓，计入{p['terminal_reserve_cny']:.2f}元压力清算储备，未假装已经完成。",
        "", "四组压力账户：", "",
    ]
    for policy, value in m.items():
        lines.append(f"- {names[policy]}：夏普{value['net_sharpe']:.3f}，年化{value['annualized_return']:.2%}，最大回撤{value['max_drawdown']:.2%}。")
    lines.extend([
        "", f"三模型各有1,262个成熟预测。价格模型的归一化买回成本MSE相对历史均值改善{price_improvement:.2%}；宏观模型较价格模型恶化{macro_worse:.2%}。宏观减价格的MSE增量区间为{r['prediction_metrics']['MSE_increment_ci95']}，跨零；分为2021—2023和2024年以后时，宏观MSE也均高于价格对照。预测误差点值改善没有转化为合格的成本后账户。",
        "", f"主账户相对价格模型的年化算术收益增量为{r['paired_macro_minus_price_annual_mean'] * 100:.3f}个百分点，固定20日区块区间[{r['paired_increment_ci95'][0] * 100:.3f}, {r['paired_increment_ci95'][1] * 100:.3f}]个百分点，未确认正增量。四组压力账户的滚动两年窗口中，同时达到全部目标的窗口共{int(w.loc[w.scenario.eq('STRESS'), 'joint_targets'].sum())}个。",
        "",
        "预测目标为第10个后续交易日买回价差的实际压力价格代理，按原价差宽度金额归一化。价格变量为信用额占宽度、隐波/实现波动率、报价观察日之后已发生的含分红标的收益及当前20日趋势。宏观再增加资金政策利差、社融同比三月变化、美股和VIX表征。四项宏观的可用时间均逐项检查早于当日09:00；没有把资金政策利差称为名义利率与中性利率差。",
        "",
        "三个模型使用同一过去两年、已成熟的历史池。历史均值作基线，价格与宏观使用固定岭惩罚10，日更；到期日尚未成熟的标签不能训练。预测买回金额加全部固定手续费形成入场信用下限，开盘压力信用额不足则不成交。原Delta选腿、期限、十日持有、费用、保护、保证金和尾部预算均保持，不因本轮结果更改。",
        "",
        "已复算3,816条保存预测、两个历史截断及未来标签扰动，并核对8个账户的现金恒等式和入场尾部预算。各实际账户无缺价或价格边界异常，但日线多腿报价仍不构成同步成交证据。原父研究失败保留，本研究也是在已见开发历史上的后续检验，没有新增独立验证或行情采集。",
    ])
    (OUT / "研究结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "saved_interpretation.json").write_text(json.dumps({"price_MSE_improvement_vs_empirical": price_improvement,
          "macro_MSE_worsening_vs_price": macro_worse, "all_stress_rolling_joint_pass": int(w.loc[w.scenario.eq("STRESS"), "joint_targets"].sum()),
          "new_models_or_accounts": 0, "goal_achieved": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("买回成本实验中文结论与净值图已生成；没有重跑账户。")


if __name__ == "__main__":
    main()
