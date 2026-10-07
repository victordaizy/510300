"""整理两年每日迁移与共享训练的固定结果，生成同风险预算比较图。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, digest
from research.strategy_review_diagnostics_v1 import metrics
from research.intraday_overnight_increment_v1 import fill_price, commission

PARENT = ROOT / "reports/research/510300_selected_mix_daily_two_year_v1"
POOLED = ROOT / "reports/research/510300_selected_mix_pooled_daily_v1"
EXECUTION = ROOT / "reports/research/510300_selected_mix_risk_before_band_v1"
BOUNDED = ROOT / "reports/research/510300_selected_mix_bounded_value_daily_v1"
NAMES = {"ORIGINAL_TAIL": "旧模型＋同一尾部预算", "DAILY_TWO_YEAR_MIN10_TAIL": "两年日更，10周期门槛",
         "DAILY_TWO_YEAR_MIN5_TAIL": "两年日更，5周期门槛", "POOLED_DAILY_TWO_YEAR_TAIL": "两年日更，三类共享训练",
         "POOLED_RISK_BEFORE_BAND10": "共享训练，风险目标先行", "POOLED_BOUNDED5_RISK_BAND10": "共享训练，固定五日价值"}


def describe(folder):
    result = read(folder / "result.json")
    shared = folder == POOLED
    p = result["primary"]
    receipts = pd.read_parquet(folder / "results/training_receipts.parquet")
    primary_name = p["policy"]
    stress = pd.read_parquet(folder / "accounts/STRESS" / primary_name / "ledger.parquet")
    cost = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")["costs"]["STRESS"]
    last = stress.iloc[-1]
    q = int(last.shares)
    price = fill_price(float(last.mark), -1, cost, .001)
    exit_reserve = commission(q, price, cost) + q * (float(last.mark) - price) if q else 0.
    rows = [f"本轮{'三类过程共享训练' if shared else '原信号图两年每日迁移'}已经完成，主候选未达到高夏普目标。",
            "", f"20万元完整压力账户截至2026年9月16日变为{p['end_equity']:,.2f}元，净夏普{p['sharpe']:.3f}、复合年化{p['annual_return']:.2%}、最大回撤{abs(p['max_drawdown']):.2%}。目标仍为1.2、10%、10%三项同时满足。",
            "", "相同日历、相同末端风险预算下的本轮压力结果：", ""]
    for r in result["all_accounts"]:
        if r["cost"] == "STRESS":
            rows.append(f"- {NAMES[r['policy']]}：夏普{r['sharpe']:.3f}，年化{r['annual_return']:.2%}，回撤{abs(r['max_drawdown']):.2%}。")
    rows += ["", "评价为2020年1月2日至2026年9月16日全部1,627个交易日，包括空仓与失败。年化242日，现金及无风险收益按零；压力佣金每侧万分之四、最低5元，滑点每侧千分之一并按不利价格单位取整。次开盘模拟执行、100份、T+1、现金及分红等式沿用原账户引擎。",
             "", "信号与风险预算逐日变化，但没有根据后来绩效选择当日模型。每次训练只使用最近两个日历年内入场、且在决策日之前已经自然结束的周期；不使用人工终点退出产生标签。参考账户现金与持仓连续保留，训练窗口限制不等于每两年重置账户。"]
    checks = result["checks"]
    if shared:
        samples = pd.read_parquet(folder / "training_reference/pooled_samples.parquet")
        rows += ["", f"实际完成{checks['daily_records']:,}条逐日模型资格记录、{checks['actual_model_fits']:,}次模型拟合、22条内部依赖账户和2个末端完整账户；额外重建趋势、反弹两个参考，D60参考直接复用。",
                 "", "三类过程各占相同总权重，类内每周期等权、周期内每状态等权。普通模型共用斜率但保留三类截距；周期内模型共用斜率但保留各周期截距。实际D60预测采用D60自己的截距，不把其他过程的较高均值当成D60收益。原八特征、岭惩罚1和标准化裁剪5不变。",
                 "", "固定最低总周期10、每类至少3、总状态行至少100；每天重新判定，资格失败时不延续过期模型。各类的保存有标签周期为：" + "、".join(f"{k} {g.cycle_id.nunique()}个/{len(g)}行" for k, g in samples.groupby("signal")) + "。",
                 "", "三类过程共享同一指数，持有区间可能重叠。新增状态不是新增独立市场路径，training_receipts.parquet单独保留重叠组计数；周期数量只作探索资格，不当作统计独立样本量。"]
    else:
        coverage = receipts.groupby("profile").agg(records=("fit_index", "size"), eligible=("eligible_for_fit", "sum"), maximum_cycles=("training_cycle_count", "max"))
        rows += ["", f"实际完成{checks['daily_training_records']:,}条模型资格记录、{checks['fitted_models']:,}次拟合、44条内部依赖账户、6个末端完整账户及1条原价格参考重建。资格记录包含两种模型和两种门槛，不是同等数量独立预测或独立交易。",
                 "", "门槛覆盖：" + "；".join(f"{k}：{int(v.eligible)}/{int(v.records)}日具备拟合资格，窗口最多{int(v.maximum_cycles)}个周期" for k, v in coverage.iterrows()) + "。",
                 "", "两年日更版将原入场日锁住的系数改为当日模型，将月度协方差和联合下行预算改为每天估计最近两年。5周期100行主候选在新结果计算前固定；10周期对照保留，不再据结果继续放宽。",
                 "", "已存在的内部账户只作结构诊断，未接末端尾部预算，不能晋升为当前候选："]
        for profile in ["STRICT10", "EXPLORATORY5"]:
            ledger = pd.read_parquet(folder / "internal_graphs" / profile / "reproduction/accounts/STRESS/SELECTED_MIX_BAND10_SIMPLE2/ledger.parquet")
            m = metrics(ledger)
            rows.append(f"{profile}内部压力夏普{m['sharpe']:.3f}、年化{m['annual_return']:.2%}、回撤{abs(m['max_drawdown']):.2%}。")
        rows += ["这说明本轮未达标不能全部归因于50%仓位限制；训练与参考预算迁移后的内部行为本身也没有保持旧绩效。"]
    comparisons = result.get("comparisons", result.get("paired_comparisons", []))
    rows += ["", "固定20日区块、2,000次的压力账户算术年化收益差：", ""]
    for row in comparisons:
        if row["cost"] == "STRESS":
            rows.append(f"- 相对{NAMES[row['right']]}：{row['annual_arithmetic_difference'] * 100:+.3f}个百分点，95%区间[{row['lower_95'] * 100:+.3f}, {row['upper_95'] * 100:+.3f}]个百分点。")
    rows += ["", "这些区间未经完整研究家族选择校正，不能换算成策略没有过拟合的概率。滚动两年结果完整保留；窗口高度重叠，不是独立试验。",
             "", "末端账户目标仓位至多50%，五日ES95事前预算为权益2.5%，10%标的跳空情景的预算不超过权益5%且不超过距90%历史峰值余量的一半；调仓和未来退出费用一并预留。风险约束优先于调仓带。预算在决策收盘检查，价格变化及次开盘成交可能令实际暴露偏离目标，不能解释为回撤保证。",
             "", "同一尾部预测使用两年内已经成熟的五日收益，先匹配20日动量正负状态，至少60条，否则退到同窗全部成熟资料。尾部监控按固定五日相位，结果成熟后才更新；本轮记录报警，不依据报警结果再调整策略参数。",
             "", f"主压力期末仍持有{q:,}份，按收盘计入权益，没有提前知道终点并强制卖出。若仅按末日收盘估算全部卖出的额外费用储备约{exit_reserve:.2f}元；这是费用敏感性，未改写原账本或模拟成交。",
             "", "必要验证已完成：训练窗口与标签成熟顺序、保存系数复算、未来与过期标签扰动、现金与分红等式以及计划仓位／尾部预算。验证没有带来新独立行情证据。",
             "", "这是在已反复使用的历史上进行的一次新固定研究。原85/15结果保留，本轮失败也保留，不把两者混同。行情采集仍暂停，当前市场观点NO_VIEW；只有510300和现金，未执行期权、自动交易或订单。没有制作审核ZIP或用户汇总表格。",
             "", "源码、冻结协议、保存模型、每日决定、连续账户与逐年／滚动结果均已落盘。高夏普目标尚未完成。"]
    (folder / "研究结论.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    save(folder / "saved_interpretation.json", {"primary_final_shares": q, "estimated_terminal_exit_reserve": exit_reserve,
                                                "report": "研究结论.md", "goal_achieved": False})


def describe_extension(folder):
    result = read(folder / "result.json")
    p = result["primary"]
    passed = p["historical_point_targets_met"]
    rows = [f"{NAMES[p['policy']]}已完成，{'历史点值达到三项目标，仍属开发证据' if passed else '尚未达到三项目标'}。", "",
            f"20万元完整压力账户变为{p['end_equity']:,.2f}元；净夏普{p['sharpe']:.3f}、复合年化{p['annual_return']:.2%}、最大回撤{abs(p['max_drawdown']):.2%}。",
            "", "所有账户均覆盖2020年1月2日至2026年9月16日的1,627个交易日，包含空仓和失败，按242日年化，现金及无风险收益为零。两年指每个决策日训练观察只向前取两个日历年，不是挑选表现最好的最近两年。"]
    if folder == EXECUTION:
        d = next(r for r in result["checks"]["details"] if r["cost"] == "STRESS")
        rows += ["", "本轮只改变执行顺序：先计算原信号的风险可行目标，再应用固定10个百分点调仓带。如果原持仓已经超限，必须减仓；原信号归零与回撤停止也优先退出。空仓初次进入不被调仓带阻拦。",
                 "", f"压力成交次数由{d['old_fills']}降至{d['new_fills']}，佣金与滑点合计由{d['old_fee']:,.2f}元降至{d['new_fee']:,.2f}元，减少{d['old_fee']-d['new_fee']:,.2f}元。新路径有{d['band_held_decisions']}次可选变动因带宽保持，{d['risk_reductions']}次超限减险没有被阻拦。",
                 "", "减少费用不是同额增加净利润。减少调仓也改变仓位和之后损益，本轮结果从完整实际成交路径重新计算。新增两档末端账户，没有重新拟合模型或重建内部信号图。"]
    else:
        checks = result["checks"]
        rows += ["", "本轮将学习标签从持有到自然退出的价值，改为从下个开盘继续持有最多五个交易日的清算价值差；原价格规则更早退出时，以较早退出为边界。入场沉没费用不重复扣除，仅比较未来卖出所得和继续持有所新增的股息权益。",
                 "", "每行五日标签成熟即可用于随后训练，不必等待整个原持仓周期结束。两年窗口、三类共享斜率、类别截距、八项状态、岭惩罚1、两收盘负预测退出、风险目标先行和原账户约束均保持。五日是固定期限，与原尾部预算的期限一致，没有搜索最佳天数。",
                 "", f"保存成熟标签{result['mature_label_rows']:,}行、期末待成熟{result['pending_label_rows']}行；完成{checks['actual_model_fits']:,}次逐日模型拟合、22条内部依赖账户和2条末端账户，没有新增参考账户。四个历史截断检查确认，在隐藏随后自然退出后，已经成熟的五日标签与完整保存结果一致。",
                 "", "周期资格指已有成熟状态的参考周期，其中可有尚未自然结束的持仓。样本仍来自同一指数，状态重叠、参考周期也可重叠，不能把新增行数或重复日更训练当成新增独立市场证据。"]
    comparison = next(row for row in result["comparisons"] if row["cost"] == "STRESS")
    rows += ["", f"相对{NAMES[comparison['right']]}，完整压力账户算术年化收益差为{comparison['annual_arithmetic_difference']*100:+.3f}个百分点；固定20日区块2,000次的95%区间为[{comparison['lower_95']*100:+.3f}, {comparison['upper_95']*100:+.3f}]个百分点。没有进行整个研究家族的选择校正。",
             "", f"压力账户全部滚动两年中，三项目标同时通过的窗口为{result['rolling_two_year_joint_passes']}个；重叠窗口不是独立试验。期末仍持有{p['terminal_shares']:,}份，按收盘计价，未利用终点日期提前卖出。",
             "", "目标仓位上限50%、五日ES95预算为权益2.5%、标的10%跳空预算为权益5%且不超过距90%历史峰值余量的一半；调仓及未来退出成本计入。预算在决策收盘计算，实际开盘和收盘价格变化仍可令实际暴露偏离目标，不能保证最大回撤。",
             "", "费用沿用原基础与压力档，100份、T+1、现金、分红和下一开盘执行均保留。保存账本权益、日收益、决定时钟及风险预算检查通过。",
             "", "原版及先前失败结果保留。所有结果仍属于反复使用历史后的开发研究；未采集新行情，当前观点NO_VIEW，没有期权或交易订单。本轮没有审核ZIP或用户汇总表格。"]
    (folder / "研究结论.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def chart():
    roots = [folder for folder in [PARENT, POOLED, EXECUTION, BOUNDED] if (folder / "result.json").exists()]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.3), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = {"ORIGINAL_TAIL": "#697B8C", "DAILY_TWO_YEAR_MIN10_TAIL": "#CC7D55", "DAILY_TWO_YEAR_MIN5_TAIL": "#1D7394", "POOLED_DAILY_TWO_YEAR_TAIL": "#437C60",
              "POOLED_RISK_BEFORE_BAND10": "#A3663C", "POOLED_BOUNDED5_RISK_BAND10": "#694799"}
    sources = []
    for root in roots:
        r = read(root / "result.json")
        for m in r["all_accounts"]:
            if m["cost"] != "STRESS":
                continue
            model = m["policy"]
            path = root / "accounts/STRESS" / model / "ledger.parquet"
            ledger = pd.read_parquet(path)
            values = np.r_[200000., ledger.equity.to_numpy(float)]
            dd = (values / np.maximum.accumulate(values) - 1)[1:]
            label = f"{NAMES[model]}｜夏普 {m['sharpe']:.3f}，年化 {m['annual_return']:.2%}"
            axes[0].plot(ledger.date, ledger.equity / 10000, label=label, color=colors[model], lw=1.5)
            axes[1].plot(ledger.date, dd, color=colors[model], lw=1.0)
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("完整账户权益（万元）")
    axes[1].set_ylabel("从历史峰值回撤")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].axhline(20, color="#999999", linestyle=":", lw=.8)
    axes[0].margins(y=.45)
    fig.suptitle("原信号图迁移：相同尾部预算下的压力账户", x=.085, ha="left", y=.965, fontsize=17, weight="bold")
    fig.text(.085, .921, "2020年1月2日至2026年9月16日；全部账户日保留，规则固定后完整运行。", fontsize=11, color="#56616C")
    fig.text(.085, .045, "初始20万元；年化242日，现金与无风险收益0。图中旧模型对照也已接入相同尾部预算。", fontsize=9.5, color="#56616C")
    fig.text(.085, .02, "目标仓位最多50%，含交易费用和T+1。历史已经参与研究，不是独立前向验证。", fontsize=9.5, color="#56616C")
    fig.subplots_adjust(left=.085, right=.97, top=.88, bottom=.11, hspace=.1)
    destination = roots[-1] / "两年日更压力账户比较.png"
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    save(roots[-1] / "figure_sources.json", {"figure": destination.name, "sources": sources})
    print("研究结论与相同尾部预算的压力净值图已生成。", flush=True)


if __name__ == "__main__":
    describe(PARENT)
    if (POOLED / "result.json").exists():
        describe(POOLED)
    for folder in [EXECUTION, BOUNDED]:
        if (folder / "result.json").exists():
            describe_extension(folder)
    chart()
