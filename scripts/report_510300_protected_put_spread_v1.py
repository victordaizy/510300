"""从已保存期权账户生成中文结果说明及净值图，不重新训练或运行账户。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_protected_put_spread_daily_v1"


def main():
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUT / "results/account_metrics.json").read_text(encoding="utf-8"))
    verify = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    rolling = pd.read_parquet(OUT / "results/all_rolling_two_years.parquet")
    primary = result["primary"]
    stress = {m["policy"]: m for m in metrics if m["scenario"] == "STRESS"}
    names = {"UNCONDITIONAL": "持续卖出保险", "RV": "过去波动率筛选", "PRICE": "价格方差预测筛选", "MACRO": "加入宏观的方差筛选"}
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"UNCONDITIONAL": "#88919B", "RV": "#298B7E", "PRICE": "#2874B8", "MACRO": "#CC6235"}
    for policy in names:
        ledger = pd.read_parquet(OUT / "accounts/STRESS" / policy / "ledger.parquet")
        axes[0].plot(ledger.date, ledger.equity / 10000, label=names[policy], color=colors[policy], lw=1.6)
        axes[1].plot(ledger.date, ledger.drawdown * 100, color=colors[policy], lw=1.2)
    axes[0].axhline(20, color="#555555", ls="--", lw=.7)
    axes[0].set_ylabel("账户权益（万元）")
    axes[1].set_ylabel("回撤（%）")
    axes[0].set_title("510300保护性认沽价差：压力成本下完整账户")
    axes[0].legend(loc="best", fontsize=9)
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.text(.1, .01, "最近两年训练、每日更新；含空仓日、逐腿费用与滑点。历史日线价格代理，尚无独立验证。", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(OUT / "压力账户净值与回撤.png", dpi=160)
    plt.close(fig)
    main_roll = rolling[rolling.policy.eq("MACRO") & rolling.scenario.eq("STRESS")]
    summary = {"new_models_or_accounts": 0, "parameter_changes": 0,
               "primary_rolling_windows": len(main_roll), "primary_joint_target_windows": int(main_roll.joint_targets.sum()),
               "primary_best_rolling_sharpe": float(main_roll.net_sharpe.max()),
               "all_stress_joint_target_windows": int(rolling.loc[rolling.scenario.eq("STRESS"), "joint_targets"].sum()),
               "source_status": result["status"], "goal_achieved": False}
    (OUT / "saved_interpretation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "本轮已按用户明确授权加入510300 ETF期权，完成1,277次每日尾部分布更新和8条完整账户路径，没有找到同时满足净夏普1.2、年化10%、最大回撤10%的策略。两年训练、每日更新和20万元本金均保留，未继承其他任务的本金或收益目标。",
        "",
        f"主压力账户期末权益{primary['ending_equity']:,.2f}元，净夏普{primary['net_sharpe']:.3f}，年化{primary['annualized_return']:.2%}，最大回撤{primary['max_drawdown']:.2%}。20个完整周期中10盈10亏，平均盈利{primary['average_win']:.2f}元、平均亏损{primary['average_loss']:.2f}元，实际盈亏比{primary['cash_payoff_ratio']:.3f}。日收益偏度{primary['daily_skew']:.3f}，最差一天{primary['worst_day']:.2%}。接受负偏性并没有在本轮换来正的净收益。",
        "",
        "四种政策共用当时可知的合约、五日尾部预测和账户约束。压力结果如下：", "",
    ]
    for policy, m in stress.items():
        lines.append(f"- {names[policy]}：净夏普{m['net_sharpe']:.3f}，年化{m['annualized_return']:.2%}，最大回撤{m['max_drawdown']:.2%}，完整周期{m['completed_cycles']}个。")
    lines.extend([
        "",
        f"宏观主账户相对同覆盖价格模型的年化算术收益增量为{result['paired_macro_minus_price_annual_mean'] * 100:.3f}个百分点；固定20日区块抽样区间为[{result['paired_increment_ci95'][0] * 100:.3f}, {result['paired_increment_ci95'][1] * 100:.3f}]个百分点。该区间全部低于零。它限制本轮宏观方差预测的具体用途，不能扩展成对全部宏观信息的否定。",
        "",
        f"四组压力账户各有{len(main_roll)}个重叠滚动两年窗口，三项目标同时达标均为零。主账户最高窗口夏普{main_roll.net_sharpe.max():.3f}；价格对照最高窗口夏普{rolling.loc[rolling.policy.eq('PRICE') & rolling.scenario.eq('STRESS'), 'net_sharpe'].max():.3f}。没有按最好窗口替换完整区间。",
        "",
        "结构固定为卖出Delta最接近-0.30的认沽、买入同到期更低行权价且Delta最接近-0.10的认沽。观察日期为执行日前两个交易日，到期30—75个自然日、优先45日；持有10个交易日。费用每腿每侧5元，压力滑点为每份至少0.0005元或报价1%，再作不利报价单位取整。单组初始理论最坏损失含费用不超过权益5%，条件五日ES95不超过权益2.5%，并保留回撤余量和保证金。持有中可按剩余最大损失缩减，不补仓。",
        "",
        "实际收取的权利金伴随空头期权负债入账。卖认沽价差仍有方向敞口，不能把全部损益解释成纯波动溢价。保证金模拟保留1.2倍价差与单腿要求的较大值，不提前享受组合申报减免；保证金与交易步骤的依据见[上交所组合策略业务指引](https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250610_10781452.shtml)。",
        "",
        "1,609个候选原点中1,534个选腿合格，75个当时无合格配对。五日标签1,529个已知、5个右删失；十日标签1,524个已知、10个右删失。未知未来标签没有用于入场筛选。各实际账户没有缺价标记或价差价格边界异常；持续卖出对照末端仍持有，计入清算成本储备，未假装已平仓。",
        "",
        f"五日尾部分布的1,272个成熟预测中，低于5%预测分位的比例为{result['monitor']['tail_q05_breach_fraction']:.2%}。固定五日相位共253个成熟原点、194个滚动告警窗口，没有触发预定告警；这只说明该告警未触发，不证明收益规律有效。告警不曾事后改变账户启停。",
        "",
        f"已复算全部{verify['saved_tail_updates_recomputed']}次保存尾部更新，检查两个历史截断、标签成熟时间和未来标签扰动；8个账户逐日现金、期权市值、费用及权益均平衡，最坏损失与入场ES预算逐笔符合。全部方差预测复用上一冻结研究，未将其重复计作新拟合。",
        "",
        "当前定价门采用未来20日方差去近似30—75日到期赔付，再用于10日持有交易，存在明确期限差异。后续若检验第10日买回价差的实际成本，应另立有限预测问题，保持本轮合约、期限、成本及失败结果。",
        "",
        "全部行情截至2026年8月14日，属于反复研究过的开发历史。不同腿日线价格缺少同步成交证明，不能认定为实盘可成交策略。没有新增行情采集、审核包或用户表格。",
    ])
    (OUT / "研究结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("保护性认沽价差结果说明与净值图已从保存账本生成；未重跑账户。")


if __name__ == "__main__":
    main()
