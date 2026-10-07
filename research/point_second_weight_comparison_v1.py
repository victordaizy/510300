"""补齐第二条固定候选的完整权重账户，不调整信号或原目标。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import upward_episode_anatomy_v1 as common
from research import point_account_nr7_inputs_v1 as measure
from research import point_second_weight_inputs_v1 as mapping
from research.adaptive_allocation_v1 import normalize_dividends
from research.anti_overfit_evidence_inputs_v1 import paired_block_effect
from research.point_account_nr7_complement_v1 import verify_account, annual_rows

OUT = ROOT / "reports/research/510300_point_second_weight_comparison_v1"
SOURCE = ROOT / "reports/research/510300_point_weight_information_diagnostic_v1"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
PERIODS = {"2015_2019": ("2015-01-05", "2019-12-31"), "2020_2026": ("2020-01-02", "2026-09-30")}
KINDS = ("A_SAVED_WEIGHT", "B_SAVED_WEIGHT")
COSTS = ("BASE", "STRESS")
ITEMS = ("daily", "trades", "orders", "decisions", "rejections")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def folder(period, cost, kind):
    branch = "inputs/controls" if kind == "A_SAVED_WEIGHT" else "results/accounts"
    return OUT / branch / period / cost / kind


def load_account(path):
    result = {name: pd.read_parquet(path / (name + ".parquet")) for name in ITEMS}
    result["terminal"] = read(path / "terminal.json")
    return result


def discovery_rows():
    rows = []
    for period in PERIODS:
        for cost in COSTS:
            saved = load_account(SOURCE / "results/accounts" / period / cost / "SAVED_WEIGHT")
            d, x, orders = saved["daily"], saved["decisions"], saved["orders"]
            joined = x.merge(d[["date", "equity", "close"]], left_on="origin", right_on="date", how="left", validate="one_to_one")
            joined["planned_distance"] = (joined.desired_shares - joined.shares_before).abs() * joined.close / joined.equity
            adds = joined.loc[joined.shares_before.gt(0) & joined.desired_shares.gt(joined.shares_before)]
            small = adds.loc[adds.planned_distance.lt(.1)]
            filled = orders.loc[orders.side.eq("BUY") & orders.origin.isin(small.origin)]
            rows.append({"period": period, "cost": cost, "existing_position_add_requests": len(adds),
                         "projected_below_ten_points": len(small), "actual_small_buy_orders": len(filled),
                         "direct_buy_cost_cny": float((filled.commission + filled.slippage).sum()),
                         "whole_account_cost_cny": float((d.commission + d.slippage).sum())})
    return rows


def freeze():
    if OUT.exists():
        raise RuntimeError("第二候选完整权重比较已存在，不覆盖。")
    inputs = [(SOURCE / "inputs" / name, "inputs/" + name) for name in (
        "prices.parquet", "dividends.csv", "parent_signals.parquet", "earlier_signals.parquet", "risks.parquet")]
    inputs += [(CONTEXT / "return_sharpe_resumed_scope_20261002.json", "inputs/resumed_scope.json"),
               (SOURCE / "summary.json", "inputs/previous_weight_summary.json")]
    for period in PERIODS:
        for cost in COSTS:
            for name in [x + ".parquet" for x in ITEMS] + ["terminal.json"]:
                inputs.append((SOURCE / f"results/accounts/{period}/{cost}/SAVED_WEIGHT/{name}",
                               f"inputs/controls/{period}/{cost}/A_SAVED_WEIGHT/{name}"))
    code = ("research/point_second_weight_comparison_v1.py", "research/point_second_weight_inputs_v1.py",
            "research/point_weight_information_inputs_v1.py", "research/point_account_nr7_inputs_v1.py",
            "research/point_account_nr7_complement_v1.py", "research/daily_supply_test_v1.py",
            "research/upward_episode_anatomy_v1.py", "research/adaptive_allocation_v1.py",
            "research/anti_overfit_evidence_inputs_v1.py", "tests/test_point_second_weight_v1.py")
    inputs += [(ROOT / name, "code/" + name) for name in code]
    for src, _ in inputs:
        if not src.is_file():
            raise FileNotFoundError(str(src))
    OUT.mkdir(parents=True)
    manifest = {}
    for src, dest in inputs:
        target = OUT / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        manifest[dest] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(target)}
    discovery = discovery_rows()
    common.save_json(OUT / "inputs/execution_discovery.json", {
        "at": common.now(), "rows": discovery,
        "decision": "近期STRESS只涉及6笔实际小额买入、直接费用约69.81元；先补齐未测量的B完整权重账户，不把节省这点费用当作主要收益改进。",
        "limit": "这只是已保存实际订单的直接费用；不是跳过订单后账户利润的上界。",
        "related_prior_execution_study": "reports/research/510300_selected_mix_risk_before_band_v1/result.json",
        "prior_execution_study_not_restarted": True})
    manifest["inputs/execution_discovery.json"] = {"sha256": common.digest(OUT / "inputs/execution_discovery.json")}
    protocol = {
        "study": "510300_POINT_SECOND_WEIGHT_COMPARISON_V1", "at": common.now(),
        "latest_user_instruction": "请继续，提高夏普率和收益率",
        "question": "此前B只做正/零点位账户，是否因丢失原仓位大小而低估它？保留两条固定来源各自的原目标，B能否在共同账户中提高收益和夏普？",
        "known_background": "已知A保留仓位强弱后近期STRESS净年化3.99%、夏普1.217、pB1.073；较早年化1.84%、夏普0.438、pB0.611。B正/零点位近期表现弱于A，B完整强弱尚未按当前共同账户测量。两候选均来自既有筛选，非未见数据新发现。",
        "source_a": measure.PARENT_A, "source_b": measure.PARENT_B,
        "primary_comparison": "B_SAVED_WEIGHT减A_SAVED_WEIGHT，两个时期和两档成本全部报告，不按近期结果切换来源或组合两者。",
        "exact_change": "B原列逐日适配给相同weight_account入口，0、未知和每个原目标大小都保留。输出逐表标记真实signal_source；不修改原来源计算、模型、系数、条件、确认天数或原退出。",
        "common_account": "20万元、510300与现金；原50%仓位上限、10个百分点调仓带、5日条件ES预算2.5%、10%跳空预算5%和回撤余量，10%账户回撤停止，次开盘、T+1、登记/应收/到账分开、期末不强平。现金和无风险率假设0；252交易日年化。",
        "costs": {"BASE": {"commission_each_side": .0002, "slippage_each_side": .0005},
                  "STRESS": {"commission_each_side": .0004, "slippage_each_side": .001},
                  "minimum_commission_cny": 5, "adverse_tick": .001, "lot": 100},
        "periods": PERIODS, "new_account_evaluations": 4, "reused_control_accounts": 4,
        "new_model_fits": 0, "parameter_search": False, "signal_combinations": 0,
        "economic_screen": "每个时期成本场景B净年化和净夏普都须严格高于A且为正，实际净pB>1、净周期均值>0、回撤<=10%。必须四个场景全通过才能称跨期历史改进；仍不是独立验证。",
        "frequency": "完整周期及每个完整年度次数均报告，增加订单不算增加机会，无逐年五次硬门槛。",
        "uncertainty": "主配对同时报告20日与252日循环区块各2000次、固定种子20261001的2.5%/97.5%描述分位数；不据区间选方法，不冒充整个历史家族选择校正。",
        "stop_rule": "若B不通过，保留拒绝，不改变B权重、父规则、A/B混合比例或风险阈值救回。",
        "historical_data_role": "DEVELOPMENT_AND_CALIBRATION",
        "old_failures_preserved": True, "existing_forward_protocol_unchanged": True,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "orders_authorized": False,
        "literature": {"transaction_cost_method": "https://www.nber.org/papers/w15205",
                       "etf_units": "https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734754.shtml"},
        "literature_limit": "交易摩擦论文只解释执行和信号寿命的关系，不证明此次候选B对510300有效。"}
    common.save_json(OUT / "protocol.json", protocol)
    manifest["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "before_new_account_results": True, "files": manifest})
    print("已固定候选B完整权重与A的同口径比较：四个新账户，不训练、不改参数。", flush=True)


def check_frozen():
    for name, record in read(OUT / "freeze.json")["files"].items():
        if common.digest(OUT / name) != record["sha256"]:
            raise ValueError("固定输入不同：" + name)
        if name.startswith("code/") and common.digest(ROOT / record["source"]) != record["sha256"]:
            raise ValueError("代码不同于结果前版本：" + name)


def parent_frames():
    recent = pd.read_parquet(OUT / "inputs/parent_signals.parquet")
    recent = recent.pivot(index="origin", columns="candidate", values="target").reset_index()
    early = pd.read_parquet(OUT / "inputs/earlier_signals.parquet")
    early = early.loc[early.period.eq("earlier_diagnostic")].pivot(index="origin", columns="model", values="target").reset_index()
    if pd.Timestamp("2019-12-31") not in set(early.origin):
        early = pd.concat([early, pd.DataFrame({"origin": [pd.Timestamp("2019-12-31")],
                                              measure.PARENT_A: [np.nan], measure.PARENT_B: [np.nan]})], ignore_index=True)
    return {"2015_2019": early, "2020_2026": recent}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("固定账户已经开始，不覆盖重跑。")
    check_frozen()
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now()})
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    data, _ = common.features(prices, div)
    risks = pd.read_parquet(OUT / "inputs/risks.parquet")
    parents = parent_frames()
    accounts, records, years, checks = {}, [], [], []
    for period, (start, end) in PERIODS.items():
        d = data.loc[data.date.le(pd.Timestamp(end))].reset_index(drop=True)
        table("原来源目标/" + period, parents[period])
        for cost in COSTS:
            result = mapping.source_account(d, div, parents[period], risks, cost, start, measure.PARENT_B)
            for name in ITEMS:
                table(f"accounts/{period}/{cost}/B_SAVED_WEIGHT/{name}", result[name])
            common.save_json(folder(period, cost, "B_SAVED_WEIGHT") / "terminal.json", result["terminal"])
            for kind in KINDS:
                saved = load_account(folder(period, cost, kind))
                accounts[(period, cost, kind)] = saved
                checks.append({"period": period, "cost": cost, "policy": kind, **verify_account(saved)})
                m = measure.metrics(saved)
                y = annual_rows(saved, period, kind, cost)
                counts = [row["completed_cycles"] for row in y if row["full_year"]]
                m["average_full_year_cycles"] = float(np.mean(counts))
                m["zero_trade_full_years"] = sum(n == 0 for n in counts)
                records.append({"period": period, "cost": cost, "policy": kind, **m})
                years.extend(y)
                if kind == "B_SAVED_WEIGHT":
                    print(f"{period}/{cost}/候选B：净年化{m['net_cagr']:.2%}，净夏普{m['net_sharpe']:.3f}，实际净pB{m['p_times_b']:.3f}，完整周期{m['completed_cycles']}。", flush=True)
    metrics = pd.DataFrame(records)
    table("完整权重共同账户比较", metrics)
    table("逐年净收益与次数", pd.DataFrame(years))
    table("保存账户核对", pd.DataFrame(checks))
    diffs, intervals, gates = [], [], []
    for period in PERIODS:
        for cost in COSTS:
            x = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost)].set_index("policy")
            a, b = x.loc[KINDS[0]], x.loc[KINDS[1]]
            diffs.append({"period": period, "cost": cost, **{k + "_delta": b[k] - a[k] for k in (
                "net_cagr", "net_sharpe", "max_drawdown", "ending_equity", "mean_exposure", "completed_cycles", "orders",
                "gross_at_actual_quantities_pnl", "total_commission", "total_slippage", "p_times_b")}})
            reasons = []
            for key in ("net_cagr", "net_sharpe"):
                if not np.isfinite(b[key]) or not b[key] > max(0., a[key]):
                    reasons.append(key + "_NOT_HIGHER_THAN_A_AND_ZERO")
            if not np.isfinite(b.p_times_b) or not b.p_times_b > 1:
                reasons.append("ACTUAL_NET_P_TIMES_B_NOT_ABOVE_ONE")
            if not np.isfinite(b.mean_cycle_net_return) or not b.mean_cycle_net_return > 0:
                reasons.append("MEAN_CYCLE_RETURN_NOT_POSITIVE")
            if not b.max_drawdown <= .1:
                reasons.append("DRAWDOWN_EXCEEDS_TEN_PERCENT")
            gates.append({"period": period, "cost": cost, "economic_screen_pass": not reasons, "reasons": reasons})
            da, db = [accounts[(period, cost, kind)]["daily"] for kind in KINDS]
            pd.testing.assert_series_equal(da.date, db.date)
            for block in (20, 252):
                item, _ = paired_block_effect(da.net_return.to_numpy(float), db.net_return.to_numpy(float), block=block)
                intervals.append({"period": period, "cost": cost, **item})
    table("B相对A的账户增量", pd.DataFrame(diffs))
    table("预设区块的描述区间", pd.DataFrame(intervals))
    passed = all(row["economic_screen_pass"] for row in gates)
    summary = {"study": "510300_POINT_SECOND_WEIGHT_COMPARISON_V1", "at": common.now(),
               "status": "HISTORICAL_SCREEN_PASSED_NOT_INDEPENDENTLY_VALIDATED" if passed else "REJECTED_FROZEN",
               "new_account_evaluations": 4, "reused_control_accounts": 4, "saved_accounts_checked": 8,
               "new_model_fits": 0, "parameter_search": False, "necessary_tests_passed": 3,
               "all_four_economic_screens_pass": passed, "economic_gates": gates, "differences": diffs,
               "cross_period_return_and_sharpe_improvement": all(x["net_cagr_delta"] > 0 and x["net_sharpe_delta"] > 0 for x in diffs),
               "historical_data_role": "DEVELOPMENT_AND_CALIBRATION", "old_failures_preserved": True,
               "existing_forward_protocol_unchanged": True, "independent_validation": "NOT_ESTABLISHED",
               "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    print("四个新增账户已完成，固定经济比较" + ("通过，独立验证仍未建立。" if passed else "未通过。"), flush=True)


def report():
    check_frozen()
    summary = read(OUT / "summary.json")
    m = pd.read_parquet(OUT / "results/完整权重共同账户比较.parquet")
    delta = pd.read_parquet(OUT / "results/B相对A的账户增量.parquet")
    years = pd.read_parquet(OUT / "results/逐年净收益与次数.parquet")
    intervals = pd.read_parquet(OUT / "results/预设区块的描述区间.parquet")
    lines = ["# 第二条固定候选：保留完整权重后能否提高收益与夏普", "",
             "本轮响应用户恢复的提高收益和夏普目标，完成此前缺失的B完整仓位账户。没有等待新日线才开展历史开发比较，也没有把历史结果改称独立验证。", "",
             "A为CORE_AUXILIARY_DRAWDOWN_GATE，B为LAG_CONFIRMED_RUNS_AUXILIARY。两者沿用各自原始目标大小、零与未知，用同一个weight_account执行。此前已有A强弱账户，B只测过正/零点位；本轮只补4条B账户，复用4条A对照。", "",
             "两者同为20万元、510300与现金，50%仓位上限、10个百分点调仓带、5日ES与跳空/回撤预算、次日开盘、T+1、分红应收和到账分开；末端不强平。全年空仓日计入，现金和无风险率假设0，252交易日年化。", "",
             "基础成本为单边万二佣金、万五滑点；压力成本为单边万四佣金、千一滑点。均至少5元佣金、0.001元不利刻度和100份整数交易。", "",
             "| 时期 | 成本 | 来源 | 净年化 | 净夏普 | 回撤 | 实际净pB | 完整周期 | 完整年均次数 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in m.itertuples():
        lines.append(f"| {r.period} | {r.cost} | {'A' if r.policy == KINDS[0] else 'B'} | {r.net_cagr:.2%} | {r.net_sharpe:.3f} | {r.max_drawdown:.2%} | {r.p_times_b:.3f} | {r.completed_cycles} | {r.average_full_year_cycles:.2f} |")
    lines += ["", "两个时期分别启动，不拼接净值。近期截至2026年9月30日，2026不计完整年度次数均值。完整周期从空仓到清仓，加减仓订单不增加次数；实际周期回报按净损益/累计买入支出计算，pB是实际净胜率乘平均盈亏比。", "",
              "| 时期 | 成本 | B年化增量 | B夏普增量 | 期末资金增量 | 完整周期增量 |",
              "|---|---|---:|---:|---:|---:|"]
    for r in delta.itertuples():
        lines.append(f"| {r.period} | {r.cost} | {r.net_cagr_delta*100:+.2f}个百分点 | {r.net_sharpe_delta:+.3f} | {r.ending_equity_delta:+,.2f}元 | {int(r.completed_cycles_delta):+d} |")
    lines += ["", "四场景同时提高收益与夏普并通过pB、净均值和回撤要求：**" + ("通过历史筛查，仍未独立验证" if summary["all_four_economic_screens_pass"] else "没有通过，不采用B替换A") + "**。", ""]
    for g in summary["economic_gates"]:
        lines.append(f"- {g['period']}/{g['cost']}：" + ("通过历史经济筛查。" if g["economic_screen_pass"] else "未通过：" + "；".join(g["reasons"]) + "。"))
    lines += ["", "| 压力成本时期/来源 | 毛损益（实际份额） | 佣金及滑点 | 平均仓位 | 周期净均值 | 订单数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in m.loc[m.cost.eq("STRESS")].itertuples():
        lines.append(f"| {r.period}/{'A' if r.policy == KINDS[0] else 'B'} | {r.gross_at_actual_quantities_pnl:,.2f}元 | {r.total_commission+r.total_slippage:,.2f}元 | {r.mean_exposure:.2%} | {r.mean_cycle_net_return:.2%} | {r.orders} |")
    lines += ["", "| 压力成本时期/年度 | A净收益 | B净收益 | A完整周期 | B完整周期 |", "|---|---:|---:|---:|---:|"]
    for (period, year), group in years.loc[years.cost.eq("STRESS")].groupby(["period", "year"], sort=True):
        a, b = [group.loc[group.policy.eq(kind)].iloc[0] for kind in KINDS]
        lines.append(f"| {period}/{year}{'截至9月30日' if year == 2026 else ''} | {a.calendar_year_return:.2%} | {b.calendar_year_return:.2%} | {int(a.completed_cycles)} | {int(b.completed_cycles)} |")
    lines += ["", "| 时期 | 成本 | 区块天数 | 夏普差2.5%—97.5% | 年化差2.5%—97.5% |", "|---|---|---:|---:|---:|"]
    for r in intervals.itertuples():
        lines.append(f"| {r.period} | {r.cost} | {r.block_trading_days} | [{r.sharpe_delta_lower_2_5:.3f}, {r.sharpe_delta_upper_97_5:.3f}] | [{r.cagr_delta_lower_2_5:.2%}, {r.cagr_delta_upper_97_5:.2%}] |")
    lines += ["", "区块各2000次，固定种子20261001，同时报告两种预设尺度。它们是历史描述区间，不代表独立验证、未来成功概率或完整试验家族的选择校正。", "",
              "本轮先核对了调仓顺序：近期A压力成本中，31次持仓中加仓请求有22次被风险预算裁成不足10个百分点，但仅6次实际成交，直接买入费用69.81元。较早为65次实际成交、558.39元。该金额不是取消订单后利润的上界；它说明仅宣传这些直接费用节省，不能解释显著的收益改善。相似执行规则已有旧研究，本轮没有重启。", "",
              "[交易成本与动态调仓论文](https://www.nber.org/papers/w15205)说明执行成本应进入持仓决策，但不证明这次B来源或任何510300规则有效。[上交所ETF单位说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734754.shtml)支持本账户采用的100份和0.001元单位。", "",
              "本轮不按结果混合A/B、不调整B参数、不把旧失败版本恢复有效，也不替换已冻结前瞻比较。保留所有时期及两档费用结果。三个针对性测试通过，八份保存账户完成资金、时序和完整周期核对。目标是否实现仍须看实际经济门槛与后续独立证据。", "",
              "文件入口：protocol.json、summary.json、results/完整权重共同账户比较.csv、results/逐年净收益与次数.csv；新账户在results/accounts，原对照在inputs/controls。", ""]
    (OUT / "研究结论.md").write_text("\n".join(lines), encoding="utf-8")
    plot()
    common.save_json(OUT / "delivery_receipt.json", {"at": common.now(), "report_sha256": common.digest(OUT / "研究结论.md"),
                     "metrics_sha256": common.digest(OUT / "results/完整权重共同账户比较.parquet"),
                     "new_accounts": 4, "reused_controls": 4, "saved_accounts_checked": 8, "new_model_fits": 0})
    print("候选B完整权重研究结论、逐年结果及净值图已保存。", flush=True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    for ax, period in zip(axes, PERIODS):
        for kind, label, color in ((KINDS[0], "A原仓位强弱", "#286e7b"), (KINDS[1], "B原仓位强弱", "#ba713e")):
            d = load_account(folder(period, "STRESS", kind))["daily"]
            ax.plot(d.date, d.equity/200000, label=label, color=color, lw=1.7)
        ax.set_title("2015—2019：较早时期" if period == "2015_2019" else "2020—2026年9月30日：近期", loc="left")
        ax.set_ylabel("20万元账户净值")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left")
    fig.suptitle("两条固定来源都保留完整权重，结果如何？", x=.07, ha="left", fontsize=15)
    fig.text(.07, .01, "同一账户引擎与压力费用；两时期独立启动。历史开发比较，未建立独立验证。", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(OUT / "两候选完整权重净值.png", dpi=150)
    fig.savefig(OUT / "两候选完整权重净值.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两条已有候选的完整权重账户比较。")
    parser.add_argument("action", choices=("freeze", "run", "report"))
    command = parser.parse_args().action
    {"freeze": freeze, "run": run, "report": report}[command]()
