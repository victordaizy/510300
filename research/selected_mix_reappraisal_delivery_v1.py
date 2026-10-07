"""展示已经完成的原版策略复核；不改变信号、模型或账户。"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import OUT, OLD, MODEL, digest, read, save, verify_sources
from research.strategy_review_diagnostics_v1 import cycles


def describe_saved_outer_family():
    sources = [OLD / "result.json", OLD / "joint_target_metrics.csv", OLD / "candidate_outcomes.json"]
    models = read(sources[0])["candidate_models"]
    frame = pd.read_csv(sources[1])
    frame = frame.loc[frame.model.isin(models)].copy()
    assert len(models) == 8 and len(frame) == 32
    assert not frame.duplicated(["model", "period", "cost"]).any()
    expected = {("evaluation", "BASE"), ("evaluation", "STRESS"),
                ("earlier_diagnostic", "BASE"), ("earlier_diagnostic", "STRESS")}
    for _, group in frame.groupby("model"):
        assert set(zip(group.period, group.cost)) == expected
    frame["all_three_point_targets"] = (frame.net_sharpe.ge(1.2) & frame.annualized_return.ge(.1)
                                        & frame.max_drawdown.ge(-.1))
    passed = frame.groupby("model").all_three_point_targets.all().to_dict()
    main = frame.loc[frame.period.eq("evaluation") & frame.cost.eq("STRESS")]
    outcome = read(sources[2])
    assessment = {"status": "EXISTING_EIGHT_SETTING_FAMILY_DESCRIBED_NO_NEW_SELECTION",
                  "timing": "主复核完成后对原第209轮保存结果作补充描述，不是预注册的新稳健性实验。",
                  "main_history_end": "2026-08-14", "early_history_end": "2019-12-31",
                  "original_primary": outcome["primary_preserved"],
                  "original_primary_four_scenario_point_pass": passed[outcome["primary_preserved"]],
                  "configurations": len(models), "accounts": len(frame),
                  "four_scenario_joint_point_passes": sum(passed.values()),
                  "main_stress_sharpe_range": [main.net_sharpe.min(), main.net_sharpe.max()],
                  "main_stress_cagr_range": [main.annualized_return.min(), main.annualized_return.max()],
                  "largest_drawdown_magnitude": -frame.max_drawdown.min(),
                  "per_model_joint_pass": passed, "all_saved_metrics": frame.to_dict("records"),
                  "interpretation": "最后外层配比有局部一致性；八套共享已筛选底层，不是八个独立发现，不能消除上游选择偏差。",
                  "new_accounts": 0, "new_model_fits": 0, "new_independent_evidence": 0,
                  "source_files": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}}
    save(OUT / "results/saved_outer_family_assessment.json", assessment)
    return assessment


def make_chart(result):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei"],
                         "axes.unicode_minus": False, "font.size": 10.5})
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.8), sharex="col",
                             gridspec_kw={"height_ratios": [2.1, 1]})
    fig.patch.set_facecolor("#F7F8FA")
    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CFD5DB")
        ax.grid(axis="y", color="#E6EAF0", linewidth=.7)
        ax.set_axisbelow(True)
    colors = {"BASE": "#7C99B3", "STRESS": "#155A80"}
    names = {"BASE": "基础成本", "STRESS": "压力成本"}
    used, profiles = [], []
    for col, period in enumerate(["earlier", "main"]):
        for cost in ["BASE", "STRESS"]:
            source = (OLD / "earlier_diagnostic" / cost / f"{MODEL}_ledger.parquet" if period == "earlier"
                      else OUT / "reproduction/accounts" / cost / MODEL / "ledger.parquet")
            ledger = pd.read_parquet(source)
            dates = pd.to_datetime(ledger.date)
            values = ledger.equity.to_numpy(float)
            nav = np.r_[200000., values]
            drawdown = (nav / np.maximum.accumulate(nav) - 1)[1:]
            completed, unfinished = cycles(ledger)
            positive = completed.loc[completed.profit.gt(0), "total_return"]
            negative = completed.loc[completed.profit.lt(0), "total_return"]
            profiles.append({"period": period, "cost": cost,
                             "daily_skew_descriptive": float(ledger.net_return.skew()),
                             "worst_day": float(ledger.net_return.min()),
                             "best_day": float(ledger.net_return.max()),
                             "max_exposure": float(ledger.exposure.max()),
                             "holding_days": int(ledger.shares.gt(0).sum()),
                             "complete_cycles": len(completed), "positive_cycles": len(positive),
                             "negative_cycles": len(negative),
                             "realized_mean_cycle_return_payoff": float(positive.mean() / -negative.mean()),
                             "unfinished_cycle": unfinished,
                             "ex_ante_reward_risk_ratio": "NOT_ESTABLISHED_BY_REALIZED_PAYOFF",
                             "independent_evidence": False})
            m = next(row for row in result["historical_metrics"] if row["period"] == period and row["cost"] == cost)
            label = f"{names[cost]}：夏普 {m['sharpe']:.3f}｜年化 {m['annual_return']:.2%}"
            axes[0, col].plot(dates, values / 10000, color=colors[cost], lw=1.7, label=label,
                              linestyle="--" if cost == "BASE" else "-")
            axes[1, col].plot(dates, drawdown, color=colors[cost], lw=1.0)
            if cost == "STRESS":
                axes[1, col].fill_between(dates, drawdown, 0, color=colors[cost], alpha=.09)
                axes[0, col].text(.97, .32, f"压力末值 {values[-1] / 10000:.2f} 万元",
                                  transform=axes[0, col].transAxes, ha="right", color=colors[cost], fontsize=11)
            used.append({"period": period, "cost": cost, "path": source.relative_to(ROOT).as_posix(),
                         "sha256": digest(source), "days": len(ledger)})
        title = "较早历史｜2015—2019年" if period == "earlier" else "主历史｜2020年至2026年9月16日"
        axes[0, col].set_title(title, loc="left", fontsize=12, pad=12, weight="bold")
        axes[0, col].axhline(20, color="#8C9299", lw=.8, linestyle=":")
        axes[0, col].legend(loc="upper left", frameon=False, fontsize=9.5)
        axes[0, col].set_ylim(18, 43)
        axes[1, col].set_ylim(-.078, .006)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[1, col].xaxis.set_major_locator(mdates.YearLocator())
        axes[1, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1, col].tick_params(axis="x", labelsize=9)
    axes[0, 0].set_ylabel("完整账户权益（万元）")
    axes[1, 0].set_ylabel("从此前权益高点回撤")
    axes[0, 1].annotate("最大盈利周期约占总净利润42%", xy=(pd.Timestamp("2024-10-08"), 34.006),
                        xytext=(pd.Timestamp("2021-03-01"), 30.5), fontsize=9.5,
                        arrowprops={"arrowstyle": "->", "color": "#9A5B32"}, color="#9A5B32")
    fig.suptitle("原85/15指数策略｜固定原版复核", x=.07, y=.965, ha="left", fontsize=20, weight="bold")
    fig.text(.07, .916, "四个历史场景达到点值门槛；两段历史都参与过筛选，尚无独立交易验证。",
             fontsize=12, color="#4C5663")
    fig.text(.07, .058, "两段分别从20万元开始，未拼接。只持有510300与现金；费用、分红、空仓日均计入。年化242日，现金与无风险收益按0。",
             fontsize=9.5, color="#4C5663")
    fig.text(.07, .031, "较早历史末日按原规则开盘清算；主历史按收盘估值。主历史22条依赖账户从头复现，较早两账户复算原保存账本。",
             fontsize=9.5, color="#4C5663")
    fig.subplots_adjust(left=.07, right=.98, top=.855, bottom=.135, wspace=.19, hspace=.12)
    path = OUT / "原版历史账户与回撤.png"
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    save(OUT / "figure_sources.json", {"figure": path.name, "sha256": digest(path), "inputs": used})
    save(OUT / "results/realized_return_profiles.json", profiles)


def main():
    verify_sources(OUT)
    result = read(OUT / "result.json")
    receipt = read(OUT / "reproduction_receipt.json")
    assert receipt["status"] == "PASS_FROZEN_ORIGINAL_FULL_GRAPH_REPRODUCTION"
    assert all(r["all_ledger_columns_exact"] and r["common_decision_columns_exact"]
               and not r["noncommon_decision_columns"] for r in receipt["results"])
    assert result["historical_point_passes"] == 4
    family = describe_saved_outer_family()
    make_chart(result)
    save(OUT / "delivery_status.json", {
        "status": "COMPLETED_FIXED_ORIGINAL_REPRODUCTION_AND_EVIDENCE_REAPPRAISAL",
        "strategy": MODEL, "reproduced_accounts": receipt["accounts"],
        "reproduced_ledger_rows": receipt["total_ledger_rows"],
        "implementation": "research/selected_mix_reappraisal_v1.py",
        "launcher": "scripts/run_510300_selected_mix_reappraisal.ps1",
        "assessment": result["verdict"], "original_rules_changed": False,
        "existing_outer_configurations_reviewed": family["configurations"],
        "existing_outer_family_four_scenario_point_passes": family["four_scenario_joint_point_passes"],
        "complete_assessment_files": ["result.json", "results/saved_outer_family_assessment.json",
                                      "results/realized_return_profiles.json", "原策略重新评估.md"],
        "model_refits": 0, "new_independent_observations": 0,
        "real_time_data_ingestion_implemented_in_this_entry": False,
        "forward_execution_enabled": False, "current_market_view": "NO_VIEW",
        "goal_achieved": False, "review_zip_created": False,
        "report": "原策略重新评估.md", "figure": "原版历史账户与回撤.png"})
    print("原版账本和结果已核对；历史净值图与交付状态已生成。没有新增交易模拟或模型拟合。", flush=True)


if __name__ == "__main__":
    main()
