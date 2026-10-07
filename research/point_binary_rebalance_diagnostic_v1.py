"""固定一个简化版本，辨别仓位强弱与持仓中再平衡的不同作用。"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import anti_overfit_evidence_inputs_v1 as evidence
from research import point_account_nr7_inputs_v1 as measurement
from research import point_binary_rebalance_inputs_v1 as projection
from research import point_weight_information_inputs_v1 as weighted
from research import upward_episode_anatomy_v1 as common
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_account_nr7_complement_v1 import annual_rows, verify_account

OUT = ROOT / "reports/research/510300_point_binary_rebalance_diagnostic_v1"
SOURCE = ROOT / "reports/research/510300_point_weight_information_diagnostic_v1"
POLICIES = ("POINT_BINARY", "SAVED_WEIGHT", "BINARY_REBALANCE")
PERIODS = {"2015_2019": ("2015-01-05", "2019-12-31"),
           "2020_2026": ("2020-01-02", "2026-09-30")}
COSTS = ("BASE", "STRESS")
LABELS = {"POINT_BINARY": "旧二值点位账户", "SAVED_WEIGHT": "原仓位强弱与再平衡",
          "BINARY_REBALANCE": "统一正目标与同样再平衡"}
ACCOUNT_ITEMS = ("daily", "trades", "orders", "decisions", "rejections")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load_account(folder):
    result = {name: pd.read_parquet(folder / (name + ".parquet")) for name in ACCOUNT_ITEMS}
    result["terminal"] = read_json(folder / "terminal.json")
    return result


def account_folder(period, cost, policy):
    branch = "results/accounts" if policy == "BINARY_REBALANCE" else "inputs/controls"
    return OUT / branch / period / cost / policy


def freeze():
    if OUT.exists():
        raise RuntimeError("固定再平衡诊断已存在，不覆盖或重新冻结。")
    sources = []
    for name in ("prices.parquet", "dividends.csv", "parent_signals.parquet",
                 "earlier_signals.parquet", "risks.parquet", "old_terminal_rejection.json"):
        sources.append((SOURCE / "inputs" / name, "inputs/" + name))
    sources += [(SOURCE / "summary.json", "inputs/previous_summary.json"),
                (SOURCE / "results/同风险预算的仓位信息比较.parquet", "inputs/previous_metrics.parquet")]
    for period in PERIODS:
        for cost in COSTS:
            for policy in POLICIES[:2]:
                for name in [x + ".parquet" for x in ACCOUNT_ITEMS] + ["terminal.json"]:
                    relative = f"{period}/{cost}/{policy}/{name}"
                    sources.append((SOURCE / "results/accounts" / relative, "inputs/controls/" + relative))
    for name in ("research/point_binary_rebalance_diagnostic_v1.py",
                 "research/point_binary_rebalance_inputs_v1.py",
                 "research/point_weight_information_inputs_v1.py",
                 "research/point_account_nr7_inputs_v1.py",
                 "research/point_account_nr7_complement_v1.py",
                 "research/daily_supply_test_v1.py", "research/upward_episode_anatomy_v1.py",
                 "research/adaptive_allocation_v1.py", "research/anti_overfit_evidence_inputs_v1.py",
                 "tests/test_point_binary_rebalance_v1.py"):
        sources.append((ROOT / name, "code/" + name))
    for src, _ in sources:
        if not src.is_file():
            raise FileNotFoundError(str(src))
    OUT.mkdir(parents=True)
    files = {}
    for src, name in sources:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        files[name] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(target)}
    protocol = {
        "study": "510300_POINT_BINARY_REBALANCE_DIAGNOSTIC_V1", "at": common.now(),
        "latest_user_instruction": "去除过拟合，并把我们的模型完善做得更好，夏普率更高，收益率更高",
        "question": "原收益改善有多少依赖来源仓位大小？在完全相同再平衡引擎内，删除连续目标大小能否更简单且提高收益和夏普？",
        "known_before_run": "原SAVED_WEIGHT对比旧POINT_BINARY同时改变目标大小和持仓中加减仓，存在归因混杂。近期压力年化3.99%、夏普1.217，较早年化1.84%、夏普0.438且pB0.611。既有反复选择和失败结论保留。",
        "single_new_variant": "BINARY_REBALANCE：每个已知正目标统一为0.5，零仍为零，未知仍未知；0.5继承现有预算上限，没有拟合。实际份额继续接受原风险上限。",
        "primary_comparison": "BINARY_REBALANCE减SAVED_WEIGHT；两者调用完全相同的weight_account。POINT_BINARY只作已有背景对照，不把它与新版本的差异全归因于一个因素。",
        "fixed_common_rules": "20万元；50%仓位上限；原10个百分点调整带；5日条件ES预算2.5%；10%跳空预算5%及原回撤余量；次开盘且买入只能缩减前夜请求；T+1；股息应收和到账分开；回撤10%停止新风险；末端不强平；现金及无风险率0；252交易日年化。",
        "costs": {"BASE": {"commission_each_side": .0002, "slippage_each_side": .0005},
                  "STRESS": {"commission_each_side": .0004, "slippage_each_side": .001},
                  "minimum_commission_cny": 5, "adverse_tick": .001, "lot": 100},
        "periods": PERIODS, "new_account_evaluations": 4, "reused_accounts": 8,
        "new_signal_fits": 0, "new_point_replays": 0, "parameter_grid": False,
        "economic_screen": "四个时期成本场景逐一要求新版本净年化和净夏普均严格高于SAVED_WEIGHT，均为正，实际净pB>1，周期均值>0，最大回撤<=10%；不接受只赢近期。",
        "frequency": "只报告自然完成周期和完整年度均值；加减仓订单不计新周期，无逐年次数硬下限。",
        "uncertainty": "对主要比较同时报告20和252交易日循环配对区块各2000次的2.5%/97.5%描述分位数，随机种子20261001；不挑区块、不把描述区间当独立有效性证据。",
        "attribution": "报告实际成交量毛损益、摩擦、平均仓位、完整周期和共同日历；相同风险上限不等于实际风险暴露完全一致。",
        "historical_data_role": "DEVELOPMENT_AND_CALIBRATION_NOT_UNTOUCHED",
        "complexity_limit": "只简化资金分配层；原复杂点位来源、机械状态和月度退出模型仍存在，不能宣称整条模型已经简单或去除过拟合。",
        "stop_rule": "该固定版本不通过时记录拒绝，不追加目标仓位、调整带或风险参数搜索。",
        "forward_protocol": "原已冻结前瞻比较不改、不根据本轮结果替换候选；本轮未创建新前瞻候选。",
        "old_terminal_rejection_preserved": True, "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False, "orders_authorized": False,
        "method_source": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf",
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "before_new_financial_results": True, "files": files})
    print("已固定一个简化版本、四个新账户比较及全部失败条件，尚未运行新账户。", flush=True)


def check_frozen():
    for name, record in read_json(OUT / "freeze.json")["files"].items():
        if common.digest(OUT / name) != record["sha256"]:
            raise ValueError("固定输入改变：" + name)
        if name.startswith("code/") and common.digest(ROOT / record["source"]) != record["sha256"]:
            raise ValueError("当前代码不同于固定版本：" + name)


def load_parents():
    recent = pd.read_parquet(OUT / "inputs/parent_signals.parquet")
    recent = recent.pivot(index="origin", columns="candidate", values="target").reset_index()
    early = pd.read_parquet(OUT / "inputs/earlier_signals.parquet")
    early = early.loc[early.period.eq("earlier_diagnostic")].pivot(index="origin", columns="model", values="target").reset_index()
    if pd.Timestamp("2019-12-31") not in set(early.origin):
        early = pd.concat([early, pd.DataFrame({"origin": [pd.Timestamp("2019-12-31")],
                          measurement.PARENT_A: [np.nan], measurement.PARENT_B: [np.nan]})], ignore_index=True)
    return {"2015_2019": early, "2020_2026": recent}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("固定新账户已经开始，不重新计算或覆盖。")
    check_frozen()
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now(), "orders_authorized": False})
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    data, _ = common.features(prices, div)
    risks = pd.read_parquet(OUT / "inputs/risks.parquet")
    parents = load_parents()
    accounts, records, yearly, checks, source_checks = {}, [], [], [], []
    for period, (start, end) in PERIODS.items():
        d = data.loc[data.date.le(pd.Timestamp(end))].reset_index(drop=True)
        original = parents[period]
        projected = projection.binary_rebalance_parents(original)
        np.testing.assert_array_equal(original[measurement.PARENT_A].gt(0), projected[measurement.PARENT_A].gt(0))
        np.testing.assert_array_equal(original[measurement.PARENT_A].isna(), projected[measurement.PARENT_A].isna())
        source_checks.append({"period": period, "rows": len(original), "positive_and_unknown_flags_identical": True,
                              "original_positive_target_levels": int(original.loc[original[measurement.PARENT_A].gt(0), measurement.PARENT_A].nunique()),
                              "projected_positive_target_levels": int(projected.loc[projected[measurement.PARENT_A].gt(0), measurement.PARENT_A].nunique())})
        table("signals/" + period, projected)
        for cost in COSTS:
            result = weighted.weight_account(d, div, projected, risks, cost, start)
            for name in ACCOUNT_ITEMS:
                table(f"accounts/{period}/{cost}/BINARY_REBALANCE/{name}", result[name])
            common.save_json(account_folder(period, cost, "BINARY_REBALANCE") / "terminal.json", result["terminal"])
            for policy in POLICIES:
                saved = load_account(account_folder(period, cost, policy))
                accounts[(period, cost, policy)] = saved
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(saved)})
                m = measurement.metrics(saved)
                years = annual_rows(saved, period, policy, cost)
                counts = [x["completed_cycles"] for x in years if x["full_year"]]
                m["average_full_year_cycles"] = float(np.mean(counts))
                m["zero_trade_full_years"] = sum(n == 0 for n in counts)
                records.append({"period": period, "cost": cost, "policy": policy, **m})
                yearly.extend(years)
                if policy == "BINARY_REBALANCE":
                    print(f"{period}/{cost}：净夏普{m['net_sharpe']:.4f}，净年化{m['net_cagr']:.2%}，实际净pB{m['p_times_b']:.4f}，完整周期{m['completed_cycles']}。", flush=True)
    metrics, years = pd.DataFrame(records), pd.DataFrame(yearly)
    table("统一账户比较", metrics)
    table("逐年收益与完整周期", years)
    table("保存账户复算", pd.DataFrame(checks))
    table("来源信息删减检查", pd.DataFrame(source_checks))
    differences, intervals, gates = [], [], []
    for period in PERIODS:
        for cost in COSTS:
            subset = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost)].set_index("policy")
            a, b = subset.loc["SAVED_WEIGHT"], subset.loc["BINARY_REBALANCE"]
            differences.append({"period": period, "cost": cost,
                                **{key + "_delta": b[key] - a[key] for key in (
                                    "net_cagr", "net_sharpe", "max_drawdown", "ending_equity", "mean_exposure",
                                    "gross_at_actual_quantities_pnl", "total_commission", "total_slippage",
                                    "completed_cycles", "orders", "p_times_b")}})
            reasons = []
            for key in ("net_cagr", "net_sharpe"):
                if not np.isfinite(b[key]) or not b[key] > max(0., a[key]):
                    reasons.append(key + "_NOT_HIGHER_THAN_CONTROL_AND_ZERO")
            if not np.isfinite(b.p_times_b) or not b.p_times_b > 1:
                reasons.append("ACTUAL_NET_P_TIMES_B_NOT_ABOVE_ONE")
            if not np.isfinite(b.mean_cycle_net_return) or not b.mean_cycle_net_return > 0:
                reasons.append("MEAN_CYCLE_RETURN_NOT_POSITIVE")
            if not b.max_drawdown <= .1:
                reasons.append("DRAWDOWN_ABOVE_TEN_PERCENT")
            gates.append({"period": period, "cost": cost, "economic_screen_pass": not reasons, "reasons": reasons})
            da = accounts[(period, cost, "SAVED_WEIGHT")]["daily"]
            db = accounts[(period, cost, "BINARY_REBALANCE")]["daily"]
            pd.testing.assert_series_equal(da.date, db.date)
            for block in (20, 252):
                item, _ = evidence.paired_block_effect(da.net_return.to_numpy(float), db.net_return.to_numpy(float), block=block)
                intervals.append({"period": period, "cost": cost, **item})
    table("同再平衡引擎的增量", pd.DataFrame(differences))
    table("固定两种区块描述区间", pd.DataFrame(intervals))
    all_economic = all(row["economic_screen_pass"] for row in gates)
    summary = {"study": "510300_POINT_BINARY_REBALANCE_DIAGNOSTIC_V1", "at": common.now(),
               "status": "HISTORICAL_SIMPLIFICATION_SCREEN_PASSED_NOT_VALIDATED" if all_economic else "REJECTED_FROZEN",
               "new_account_evaluations": 4, "reused_accounts": 8, "saved_accounts_recomputed": len(checks),
               "new_signal_fits": 0, "new_point_replays": 0, "parameter_search": False,
               "necessary_tests_passed": 3, "economic_gates": gates,
               "all_four_economic_screens_pass": all_economic,
               "cross_period_return_and_sharpe_improvement": all(x["net_cagr_delta"] > 0 and x["net_sharpe_delta"] > 0 for x in differences),
               "differences": differences, "information_projection_checks": source_checks,
               "historical_data_role": "DEVELOPMENT_AND_CALIBRATION",
               "independent_validation": "NOT_ESTABLISHED", "whole_model_overfitting_removed": False,
               "original_forward_comparison_unchanged": True, "old_terminal_rejection_preserved": True,
               "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    print("四个新账户已完成并落盘；固定跨期经济门槛" + ("通过，仍未独立验证。" if all_economic else "未通过，版本拒绝。"), flush=True)


def report():
    check_frozen()
    summary = read_json(OUT / "summary.json")
    metrics = pd.read_parquet(OUT / "results/统一账户比较.parquet")
    delta = pd.read_parquet(OUT / "results/同再平衡引擎的增量.parquet")
    yearly = pd.read_parquet(OUT / "results/逐年收益与完整周期.parquet")
    intervals = pd.read_parquet(OUT / "results/固定两种区块描述区间.parquet")
    rows = ["# 510300：删去仓位强弱，模型是否更简单且更赚钱", "",
            "目标同时保留：减少过拟合风险，提高扣费后全账户收益和夏普，并满足实际净p×B>1、净期望为正。只让回测好看、只做程序检查或只安排未来观察，都不算实现。", "",
            "本轮发现并纠正一处归因混杂：先前SAVED_WEIGHT与POINT_BINARY的比较，同时改变了目标仓位大小和持仓中的再平衡。原数字没有算错，但不能把全部收益改善归因于来源仓位强弱。", "",
            "唯一新版本BINARY_REBALANCE把已知正目标统一为原预算上限50%，保留零与未知；与SAVED_WEIGHT共享完全相同的账户引擎、10个百分点调整带、风险约束、成交和费用。50%只是请求上限，实际成交可能更少。没有新训练、参数网格或增加技术指标。", "",
            "只有四个新账户：两个时期、两档成本。另八个对照直接复用；全部十二个保存账户核对财富恒等式、前收盘到次开盘、自然完整周期。两个时期独立启动，不拼接成连续净值。", "",
            "## 扣费后完整账户结果", "",
            "| 时期 | 成本 | 版本 | 净年化 | 净夏普 | 最大回撤 | 实际净pB | 完整周期 | 完整年均次数 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for item in metrics.itertuples():
        rows.append(f"| {item.period} | {item.cost} | {LABELS[item.policy]} | {item.net_cagr:.2%} | {item.net_sharpe:.3f} | {item.max_drawdown:.2%} | {item.p_times_b:.3f} | {item.completed_cycles} | {item.average_full_year_cycles:.2f} |")
    rows += ["", "20万元、510300与现金，无杠杆。原50%仓位上限、5日条件ES预算2.5%、10%跳空预算5%及回撤余量均保留。所有空仓日计入，252交易日年化，现金及无风险率假设0。2026截至9月30日，不计入完整年度次数均值。", "",
             "基础成本：单边万二佣金、万五滑点；压力成本：单边万四佣金、千一滑点。两者均至少5元佣金、价格按0.001不利取整、100份整数交易。账户保留应收股息；期末持仓不虚构强平。", "",
             "完整周期从空仓到清仓，加减仓不增加次数；p为净盈利周期比例，B为平均净盈利回报除以平均净亏损回报绝对值。周期回报分母为累计买入支出。标准亏损单位期望为pB-q，用户pB>1作为额外更严格门槛保留。", "",
             "## 控制再平衡规则后的改动效果", "",
             "以下均为新统一目标版本减原仓位强弱版本。相同上限不等于相同实际暴露；原始改动可能改变平均仓位和风险占用。", "",
             "| 时期 | 成本 | 年化变化 | 夏普变化 | 期末权益变化 | 平均仓位变化 | 完整周期变化 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for item in delta.itertuples():
        rows.append(f"| {item.period} | {item.cost} | {item.net_cagr_delta*100:+.2f}个百分点 | {item.net_sharpe_delta:+.3f} | {item.ending_equity_delta:+,.2f}元 | {item.mean_exposure_delta*100:+.2f}个百分点 | {int(item.completed_cycles_delta):+d} |")
    rows += ["", "| 压力成本时期/版本 | 实际份额毛损益 | 佣金及滑点 | 平均仓位 | 成交订单 | 周期净均值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for item in metrics.loc[metrics.cost.eq("STRESS")].itertuples():
        rows.append(f"| {item.period}/{LABELS[item.policy]} | {item.gross_at_actual_quantities_pnl:+,.2f}元 | {item.total_commission+item.total_slippage:,.2f}元 | {item.mean_exposure:.2%} | {item.orders} | {item.mean_cycle_net_return:.2%} |")
    rows += ["", "毛损益只加回本账户实际成交份额的成本，用来解释收益差异；不是无成本可执行账户。", "",
             "## 预先固定的取舍", "",
             f"四个时期成本场景是否全部同时提高净年化、净夏普并通过实际净pB、净均值与回撤要求：**{'是，但仍只有历史支持' if summary['all_four_economic_screens_pass'] else '否，新固定版本拒绝'}**。", ""]
    for gate in summary["economic_gates"]:
        rows.append(f"- {gate['period']}/{gate['cost']}：{'通过历史经济筛查' if gate['economic_screen_pass'] else '未通过：' + '；'.join(gate['reasons'])}。")
    rows += ["", "不会按这个结果追加45%、55%仓位，或把10个百分点变化带换成其他值来救回版本。只删除仓位大小并不能删除原点位层的机械状态与月度学习退出；整条模型复杂度和历史选择问题仍存在。", "",
             "## 同时披露两种预设描述区间", "",
             "| 时期 | 成本 | 区块交易日 | 夏普差2.5%—97.5% | 年化差2.5%—97.5% |",
             "|---|---|---:|---:|---:|"]
    for item in intervals.itertuples():
        rows.append(f"| {item.period} | {item.cost} | {item.block_trading_days} | [{item.sharpe_delta_lower_2_5:.3f}, {item.sharpe_delta_upper_97_5:.3f}] | [{item.cagr_delta_lower_2_5:.2%}, {item.cagr_delta_upper_97_5:.2%}] |")
    rows += ["", "两种配对循环区块各2000次，固定种子20261001，均为历史描述性稳健性筛查，不是未来成功概率，也不校正整个项目的历史选择。反复切分已经看过的数据不能把它变成新检验；该问题与[修正夏普研究](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)讨论的多次试验选择偏差一致。本轮未计算缺少完整可比试验范围的正式DSR/PBO。", "",
             "## 每年真实完成次数与收益", "",
             "| 时期/年度 | 原强弱收益 | 新统一目标收益 | 原完整周期 | 新完整周期 |",
             "|---|---:|---:|---:|---:|"]
    stress_years = yearly.loc[yearly.cost.eq("STRESS")]
    for (period, year), group in stress_years.groupby(["period", "year"], sort=True):
        a = group.loc[group.policy.eq("SAVED_WEIGHT")].iloc[0]
        b = group.loc[group.policy.eq("BINARY_REBALANCE")].iloc[0]
        rows.append(f"| {period}/{year}{'截至9月30日' if year == 2026 else ''} | {a.calendar_year_return:.2%} | {b.calendar_year_return:.2%} | {int(a.completed_cycles)} | {int(b.completed_cycles)} |")
    rows += ["", "## 结论边界", "",
             "本轮输出的是一次受控的模型简化试验结果。数据截至2026年9月30日，全部已经用于研究；正结果也只能属于开发证据，负结果不能被隐藏。没有宣布去除过拟合，没有新增实盘授权，没有改变此前冻结的唯一前瞻比较。", "",
             ("本次简化通过预设历史经济筛查，只保留为待验证的开发线索，不替换已冻结的前瞻候选。" if summary["all_four_economic_screens_pass"] else "本次简化未通过预设历史经济筛查，不保留为改进模型。") + "下一步若要改进点位或持有退出，必须有与旧失败规则不同、可以事前说明的新机制，而不是继续按历史结果搜索阈值。", "",
             "文件入口：protocol.json为结果前规则，summary.json为固定结论，results/统一账户比较.csv为12条比较，results/accounts为4条新账户，inputs/controls为8条旧对照。三项针对性测试使用合成数据，没有加入市场收益结果。", ""]
    (OUT / "研究结论.md").write_text("\n".join(rows), encoding="utf-8")
    plot()
    common.save_json(OUT / "delivery_receipt.json", {"at": common.now(),
                     "report_sha256": common.digest(OUT / "研究结论.md"),
                     "metrics_sha256": common.digest(OUT / "results/统一账户比较.parquet"),
                     "saved_accounts_recomputed": 12, "new_accounts": 4, "model_fits": 0,
                     "independent_validation": "NOT_ESTABLISHED"})
    print("研究结论、完整比较和两时期净值图已保存。", flush=True)


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
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8))
    colors = {"POINT_BINARY": "#9a958f", "SAVED_WEIGHT": "#246f7c", "BINARY_REBALANCE": "#bd673d"}
    for ax, period in zip(axes, PERIODS):
        for policy in POLICIES:
            d = load_account(account_folder(period, "STRESS", policy))["daily"]
            ax.plot(d.date, d.equity / 200000, lw=1.7, color=colors[policy], label=LABELS[policy])
        ax.set_title("2015—2019：较早时期" if period == "2015_2019" else "2020—2026年9月30日：近期", loc="left")
        ax.set_ylabel("20万元账户净值")
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("同样再平衡时，去掉目标仓位强弱是否更好？", x=.07, ha="left", fontsize=16)
    fig.text(.07, .012, "压力成本；两段账户独立启动。历史机制诊断，未建立独立验证；全部空仓日计入。", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(OUT / "仓位简化与再平衡.png", dpi=150)
    fig.savefig(OUT / "仓位简化与再平衡.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定单一仓位简化试验。")
    parser.add_argument("action", choices=["freeze", "run", "report"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "report": report}[args.action]()
