"""用真实八账户结果解释PE处理、买卖时点与未达标原因。"""
from pathlib import Path
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity

SOURCE = ROOT / "reports/research/510300_forward_eps_guosen_pe_condition_policy_v1"
USAGE = ROOT / "reports/research/510300_forward_eps_saved_pe_usage_diagnostic_v1"
OUT = ROOT / "deliverables/510300前瞻EPS市盈率对照结果与进出场_20260907"
NAMES = {"G1_GUOSEN_RELATION_QUALIFIED_PE": "关系相容PE", "G2_GUOSEN_EPS_PROFIT_NO_PE": "完全不使用PE",
         "C0_ORIGINAL_GUOSEN_EPS": "原EPS三因子", "BUY_HOLD": "买入持有"}
COLORS = {"G1_GUOSEN_RELATION_QUALIFIED_PE": "#166b9a", "G2_GUOSEN_EPS_PROFIT_NO_PE": "#d18426",
          "C0_ORIGINAL_GUOSEN_EPS": "#328266", "BUY_HOLD": "#868b93"}


def render_chart(base):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.4), constrained_layout=True)
    for model, ledger in base.items():
        axes[0].plot(ledger.date, ledger.equity / 10000, label=NAMES[model], color=COLORS[model], linewidth=1.7)
    axes[0].set(title="完整账户净值：2020年初至2026年评价终点", ylabel="账户净值（万元）")
    axes[0].legend(ncol=4, loc="upper left", frameon=False)
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for model, ledger in base.items():
        if model == "BUY_HOLD":
            continue
        year = ledger.loc[ledger.date.dt.year.eq(2024)]
        exposure = 100 * year.shares * year.mark / year.equity
        axes[1].step(year.date, exposure, where="post", label=NAMES[model], color=COLORS[model], linewidth=1.7)
    axes[1].set(title="2024年的持仓差异：9月初减仓或退出，改变了后续收益", ylabel="股票占账户净值（%）", ylim=(-4, 110))
    axes[1].legend(ncol=3, loc="upper left", frameon=False)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    axes[1].set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    for axis in axes:
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    chart = OUT / "完整净值与2024年进出场影响.png"
    fig.savefig(chart, dpi=160, facecolor="white")
    plt.close(fig)
    return chart


def main():
    result = read(SOURCE / "result.json")
    checked = read(SOURCE / "saved_numerical_verification.json")
    assert checked["status"] == "PASS_SAVED_GUOSEN_PE_CONDITION_SAME_TRAINING_AND_COMPLETE_ACCOUNTS"
    assert checked["original_E3_all_training_calendars_and_labels_identical"]
    assert result["evaluation_accounts"] == 8 and result["new_accounts_generated"] == 4
    OUT.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(result["all_metrics"])
    years = pd.read_csv(SOURCE / "yearly_metrics.csv")
    config = read(ROOT / "config/510300_forward_eps_guosen_pe_condition_policy_v1.json")
    assert config["horizon"] == 60
    signals = pd.read_parquet(SOURCE / "signals.parquet")
    events = set(np.flatnonzero(signals.event_mask))
    ledgers, fills, monthly, decisions = {}, [], [], []
    for metric in result["all_metrics"]:
        cost, model = metric["cost"], metric["model"]
        ledger = pd.read_parquet(SOURCE / "evaluation" / cost / f"{model}_ledger.parquet")
        chosen = pd.read_parquet(SOURCE / "evaluation" / cost / f"{model}_decisions.parquet")
        ledgers[(cost, model)] = ledger
        trades = ledger.loc[ledger.filled_quantity.ne(0)].copy()
        trades.insert(0, "费用情景", cost); trades.insert(1, "策略", NAMES[model])
        fills.append(trades)
        decision = chosen.loc[chosen.origin_index.isin(events)].copy()
        decision.insert(0, "费用情景", cost); decision.insert(1, "策略", NAMES[model])
        decision.rename(columns={"mu5": "未来60交易日预测收益", "variance5": "未来60交易日估计方差"}, inplace=True)
        decisions.append(decision)
        for period, group in ledger.groupby(ledger.date.dt.to_period("M")):
            monthly.append({"费用情景": cost, "策略": NAMES[model], "月份": str(period),
                "月度净损益元": float(group.pnl.sum()), "费用元": float(group.commission.sum() + group.slippage_cost.sum()),
                "月末净值元": float(group.equity.iloc[-1]), "平均持有份额": float(group.shares.mean()),
                "平均股票资金占比": float((group.shares * group.mark / group.equity).mean())})
    fills = pd.concat(fills, ignore_index=True)
    decisions = pd.concat(decisions, ignore_index=True)
    monthly = pd.DataFrame(monthly)
    fills.to_csv(OUT / "全部真实成交_含终点清算.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(OUT / "全部预定月末判断_含无观点.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "逐月实际持仓与净损益.csv", index=False, encoding="utf-8-sig")
    for src, target in [("metrics.csv", "完整账户表现.csv"), ("yearly_metrics.csv", "逐年账户表现.csv"),
                        ("era_metrics.csv", "三个既定阶段表现.csv")]:
        shutil.copyfile(SOURCE / src, OUT / target)
    summary_table = []
    for metric in result["all_metrics"]:
        summary_table.append(f'| {NAMES[metric["model"]]} | {"基础" if metric["cost"] == "BASE" else "压力"} | '
            f'{metric["net_sharpe"]:.3f} | {metric["annualized_return"]:.2%} | '
            f'{metric["annualized_return_excess_vs_buy_hold"] * 100:+.2f}个百分点 | '
            f'{metric["max_drawdown"]:.2%} | {metric["trade_count"]} |')
    year_table = []
    for model, title in NAMES.items():
        selected = years.loc[years.cost.eq("BASE") & years.model.eq(model)].set_index("year")
        year_table.append(f'| {title} | ' + ' | '.join(f'{selected.loc[year, "cumulative_return"]:.2%}' for year in [2024, 2025, 2026]) + ' |')
    base = {model: ledger for (cost, model), ledger in ledgers.items() if cost == "BASE"}
    ref = base["C0_ORIGINAL_GUOSEN_EPS"]
    gap = {model: float(ledger.equity.iloc[-1] - ref.equity.iloc[-1]) for model, ledger in base.items()}
    cost_gap = {model: float((ledger.commission + ledger.slippage_cost).sum() - (ref.commission + ref.slippage_cost).sum())
                for model, ledger in base.items()}
    august = []
    for model in ["C0_ORIGINAL_GUOSEN_EPS", "G1_GUOSEN_RELATION_QUALIFIED_PE", "G2_GUOSEN_EPS_PROFIT_NO_PE"]:
        row = decisions.loc[decisions.费用情景.eq("BASE") & decisions.策略.eq(NAMES[model]) & decisions.origin.eq(pd.Timestamp("2024-08-30"))].iloc[0]
        actual = base[model].loc[base[model].date.eq(pd.Timestamp("2024-09-02"))].iloc[0]
        august.append(f'| {NAMES[model]} | {row["未来60交易日预测收益"]:.3%} | {row.reference_weight:.1%} | '
                      f'{actual.filled_quantity:+,.0f} | {actual.shares:,.0f} |')
    chart = render_chart(base)
    content = f"""# 前瞻EPS市盈率对照结果与完整进出场

2026年9月7日更新。第21轮已经实际完成并核对：使用原国信来源，把市盈率加上数值关系条件，或完全去掉市盈率，均没有改善原EPS策略的完整夏普率。主方案从原来的0.619降为0.380；去PE方案为0.304。目标1.2未达到，稳定超额没有独立证据。第19、20轮仍按已启动的原采集及接续执行，不因本轮结果改变已登记规则。

本轮的作用是把已发现的来源问题落实到完整交易账户。三种EPS方法使用同一50个月末来源范围，每次成熟训练月份、47个已兑现标签及36次实际预测时点完全相同。两种新方法合计保存72个拟合模型，基础与压力各两条新账户，加四条复用对照，共八条评价账户。

## 完整历史表现

统一20万元，2020年1月2日至2026年8月14日开盘终点，全部1604日及早期现金时间计入。表中年化超额是各策略年化收益减同费用买入持有年化收益，以百分点表示。交易次数包含统一评价终点清算，不能全部叫作模型自主退出。

| 策略 | 费用 | 完整夏普率 | 年化收益 | 年化超额 | 最大回撤 | 成交次数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(summary_table)}

基础费用下，关系相容PE方案比原EPS少赚{-gap['G1_GUOSEN_RELATION_QUALIFIED_PE']:,.2f}元，去PE方案少赚{-gap['G2_GUOSEN_EPS_PROFIT_NO_PE']:,.2f}元。它们的费用分别比原EPS少{-cost_gap['G1_GUOSEN_RELATION_QUALIFIED_PE']:.2f}元和{-cost_gap['G2_GUOSEN_EPS_PROFIT_NO_PE']:.2f}元，所以主要差距不能归因于多交费用。

主方案相对原EPS的年化平均日收益差，20日区块95%区间约为负10.93至正0.73个百分点，60日区块约负10.17至正0.37个百分点；均跨零。主方案相对去PE的两种区块区间也均跨零。区间直接来自本轮保存的2000次配对抽样，没有重抽挑结果；它们与表中几何年化收益差不是同一统计量。

## 差距主要怎样形成

三个EPS方案2020至2023年实际均未持仓。2024年收益差距很大，而2025年收益接近：

| 策略，基础费用 | 2024年收益 | 2025年收益 | 2026年至评价终点收益 |
| --- | ---: | ---: | ---: |
{chr(10).join(year_table)}

2024年8月30日，同样的判断日与训练日历产生了不同预测和目标份额。9月2日实际交易如下，负数表示卖出：

| 策略 | 预测未来60交易日收益 | 判断时选择的股票占比 | 下一开盘实际买卖份额 | 成交后持仓份额 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(august)}

原EPS保留54900份，关系相容PE方案减到13700份，去PE方案全部退出。2024年9月的净损益分别为46170.90元、10958.26元、负188.73元。关系相容PE方案在10月8日卖出剩余13700份，12月2日重新买入25300份；去PE方案也在12月2日重新进入，买入12000份。所有日期均为账户真实保存的下一开盘成交，没有把月末信号当同日成交。

这说明因子口径的改变会通过预测影响仓位，少数关键月末可以大幅改变整段结果。原方案在这段历史里得分较好，并不能反过来证明其口径问题可以忽略；相容条件使结果变差，也不能证明错误数字具有稳定预测价值。它暴露的是当前模型对因子处理和特定时点的敏感性。后续仍须看已固定的多机构、较长历史对照，不能看完2024年后临时规定那一年用旧策略。

![完整净值与2024年进出场影响](<{chart.as_posix()}>)

## 来源问题确实进入了旧模型

对原27084条公司月末记录追溯：原EPS36个实际预测月中，有4879次公司PE输入，其中1621次未通过新关系条件，包括1615次数值不相容和6次缺参考价。每个预测月都涉及这些报告，33个月的PE倒数中位数会改变。

原50个来源合格月应用条件后仍都满足覆盖门槛，最少55家公司；47个曾进入训练的月份中44个月中位数会改变。因此本轮区别来自同一时期的输入及训练结果变化，不能归因于挑选了新的评价年份。未通过条件的次数不等于已经确认的错误率，因子差值也不等于收益贡献。

## 因子与全部进出场规则

"""
    rules = (ROOT / "docs/510300_FORWARD_EPS_GUOSEN_PE_CONDITION_POLICY_V1.md").read_text(encoding="utf-8")
    rules = rules[rules.index("## 两种新策略与原策略对照"):]
    rules = rules.replace("此为两个新方法与来源范围、四条新账户，登记时尚未读取它们的成绩。", "以上规则均在本轮新账户运行前固定，现已按规则完成两种方法及四条新账户。")
    content += rules.replace("## ", "### ")
    content += f'\n\n[完整账户数值](<{(OUT / "完整账户表现.csv").as_posix()}>)；[全部实际进出场](<{(OUT / "全部真实成交_含终点清算.csv").as_posix()}>)；[全部预定月末判断，包括没有观点](<{(OUT / "全部预定月末判断_含无观点.csv").as_posix()}>)。\n'
    content += f'\n[原月末PE使用诊断](<{(USAGE / "result.json").as_posix()}>)；[本轮保存模型与八账户核对](<{(SOURCE / "saved_numerical_verification.json").as_posix()}>)。\n'
    doc = OUT / "前瞻EPS市盈率对照_结果原因与进出场.md"
    assert "```" not in content
    doc.write_text(content, encoding="utf-8")
    save(OUT / "delivery_receipt.json", {"created_at": now(), "document": identity(doc), "chart": identity(chart),
        "source_result": identity(SOURCE / "result.json"), "saved_verification": identity(SOURCE / "saved_numerical_verification.json"),
        "csv_files": 6, "all_filled_trade_rows": len(fills), "scheduled_decision_rows": len(decisions),
        "monthly_account_rows": len(monthly), "visual_check_pending": True,
        "new_gpt_review_archive_created": False, "new_account_evaluations_in_delivery": 0}, exclusive=True)
    print(json.dumps({"document": str(doc), "chart": str(chart), "trade_rows": len(fills), "month_end_decisions": len(decisions)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
