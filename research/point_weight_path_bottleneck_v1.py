"""解释主候选真实仓位的收益来源，固定路径而不重新选择交易。"""
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
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_weight_path_inputs_v1 import describe_account

OUT = ROOT / "reports/research/510300_point_weight_path_bottleneck_v1"
SOURCE = ROOT / "reports/research/510300_point_second_weight_comparison_v1"
PERIODS = ("2015_2019", "2020_2026")
COSTS = ("BASE", "STRESS")
ITEMS = ("daily", "trades", "orders", "decisions")
PATH_LABELS = {"WIN": "盈利", "FLAT": "平手", "LOSS_AFTER_POSITIVE_CLOSE_MARK": "曾有净浮盈后亏损", "LOSS_WITHOUT_POSITIVE_CLOSE_MARK": "未见净浮盈而亏损", "UNFINISHED": "未完成"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(name, data):
    target = OUT / "results" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(target.with_suffix(".parquet"), index=False)
    data.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("实际仓位路径诊断已存在，不覆盖。")
    sources = [(SOURCE / "inputs/prices.parquet", "inputs/prices.parquet"),
               (SOURCE / "inputs/dividends.csv", "inputs/dividends.csv"),
               (SOURCE / "summary.json", "inputs/previous_comparison_summary.json")]
    for period in PERIODS:
        for cost in COSTS:
            for name in ITEMS:
                rel = f"{period}/{cost}/A_SAVED_WEIGHT/{name}.parquet"
                sources.append((SOURCE / "inputs/controls" / rel, "inputs/accounts/" + rel))
    for name in ("research/point_weight_path_bottleneck_v1.py", "research/point_weight_path_inputs_v1.py",
                 "research/point_account_nr7_inputs_v1.py", "research/daily_supply_test_v1.py",
                 "tests/test_point_weight_path_v1.py"):
        sources.append((ROOT / name, "code/" + name))
    for path, _ in sources:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    OUT.mkdir(parents=True)
    files = {}
    for src, name in sources:
        dst = OUT / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[name] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(dst)}
    protocol = {
        "study": "510300_POINT_WEIGHT_PATH_BOTTLENECK_V1", "at": common.now(),
        "objective": "为提高净收益和夏普定位现有主候选实际仓位的损益来源，不新增参数搜索。",
        "question": "在实际加减仓、分红和费用下，损失主要来自入场未形成浮盈、已得浮盈回吐、持有中的日内/隔夜收益，还是退出决定至成交？",
        "known_prior": "A保留仓位强弱后的近期压力净年化3.99%、夏普1.217，较早1.84%、0.438；B替换A未跨期改善。旧固定份额点位路径已有诊断，但未还原当前四个A权重账户的实际现金流路径。",
        "accounts": "保存的A两个时期、两档费用，共四个账户；只读实际订单，不修改任何进入、退出、加減仓、风险或时间。",
        "full_calendar_identity": "真实旧库存乘隔夜原价差，加真实开盘后库存乘日内原价差，加按登记资格确认的股息，减佣金和滑点，逐日等于权益变动。",
        "cycle_path": "逐周期真实买卖现金流累积加收盘库存价值及已除息股息，形成当时的净损益；支付不重复计算收益。",
        "close_mark": "观察日收盘再扣一次全部剩余库存的假设卖出摩擦，只作标记，不假定能见到收盘后按收盘成交，不改变T+1规则。",
        "path_categories": "每个完整周期固定分盈利、平手、最终亏损但持有期间曾有正净清算标记、最终亏损且未见正净清算标记；全部纳入，无利润大小筛选。",
        "phases": "入场日、入场与退出之间、实际退出日、必要时退出后股息确认。阶段损益可加总，不能只取盈利阶段作为策略。",
        "execution_bridge": "最终实际净损益减最后持有收盘假设清算标记；另列首次清仓决定至成交的标记差及交易日数。这是描述，不是执行改善收益的保证。",
        "peak_limit": "峰值和峰值至最终差只在结果后计算；不可用为信号、可执行策略、可得利润上界或候选验收。",
        "right_censoring": "末端开放周期只报告已观察路径和市值损益，胜率与完整周期分组排除，账户权益仍包含。",
        "sources_not_reopened": ["510300_POINT_SECOND_WEIGHT_COMPARISON_V1", "510300_POINT_BINARY_REBALANCE_DIAGNOSTIC_V1", "SELECTED_MIX_BAND10_SIMPLE2"],
        "new_accounts": 0, "new_model_fits": 0, "new_parameters": 0,
        "historical_data_role": "DEVELOPMENT_DIAGNOSTIC_NOT_INDEPENDENT_VALIDATION",
        "data_cutoff": "2026-09-30", "selection_after_result": "不按亏损路径标签新增过滤、追踪止盈阈值或持有天数，不把峰值归因变成事后最优账户。",
        "literature": {"trading_costs": "https://www.nber.org/papers/w15205", "backtest_overfitting": "https://www.davidhbailey.com/dhbpapers/overfitting.pdf"},
        "literature_role": "只支持研究方法及样本选择风险，不证明510300存在新增优势。",
        "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "before_path_results": True, "files": files})
    print("已固定真实仓位路径诊断：四份已存账户，不新增回测或参数。", flush=True)


def check_frozen():
    for name, record in read(OUT / "freeze.json")["files"].items():
        if common.digest(OUT / name) != record["sha256"]:
            raise ValueError("固定资料发生改变：" + name)
        if name.startswith("code/") and common.digest(ROOT / record["source"]) != record["sha256"]:
            raise ValueError("运行代码不等于结果前版本：" + name)


def run():
    check_frozen()
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("路径诊断已经启动，不覆盖结果。")
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now()})
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    records, categories, phases, exits, checks = [], [], [], [], []
    for period in PERIODS:
        for cost in COSTS:
            folder = OUT / f"inputs/accounts/{period}/{cost}/A_SAVED_WEIGHT"
            account = {name: pd.read_parquet(folder / (name + ".parquet")) for name in ITEMS}
            cycles, paths, daily = describe_account(prices, div, account, cost)
            prefix = {"period": period, "cost": cost}
            for name, frame in (("逐周期", cycles), ("逐日周期路径", paths), ("账户日损益分解", daily)):
                table(f"{period}/{cost}/{name}", frame)
            complete = cycles.loc[cycles.status.eq("COMPLETE")]
            losses = complete.loc[complete.net_pnl.lt(0)]
            record = {**prefix, "completed_cycles": len(complete), "unfinished_cycles": int(cycles.status.ne("COMPLETE").sum()),
                      "wins": int(complete.net_pnl.gt(0).sum()), "losses": len(losses),
                      "losses_after_positive_close_mark": int(losses.peak_close_liquidation_pnl.gt(0).sum()),
                      "losses_without_positive_close_mark": int(losses.peak_close_liquidation_pnl.le(0).sum()),
                      "completed_cycle_net_pnl": float(complete.net_pnl.sum()),
                      "account_total_net_pnl": float(daily.saved_equity.iloc[-1] - 200000.),
                      "completed_exit_mark_difference": float(complete.final_open_vs_prior_close_mark.sum()),
                      "completed_first_intent_mark_difference": float(complete.first_exit_intent_to_actual_pnl.sum()),
                      "delayed_exit_cycles": int(complete.exit_intent_to_execution_sessions.gt(1).sum()),
                      "loss_peak_to_exit_sum": float(losses.peak_to_exit_difference.sum()),
                      "loss_peak_close_sum": float(losses.peak_close_liquidation_pnl.clip(lower=0).sum()),
                      "mean_holding_closes": float(complete.held_closes.mean())}
            for column in ("overnight_price_pnl", "intraday_price_pnl", "dividend_accrual", "commission", "slippage"):
                record[column] = float(daily[column].sum())
            records.append(record)
            for category in PATH_LABELS:
                group = cycles.loc[cycles.outcome_path.eq(category)]
                categories.append({**prefix, "category": category, "count": len(group),
                                   "net_pnl": float(group.net_pnl.sum()) if category != "UNFINISHED" else np.nan,
                                   "mean_net_return": float(group.net_return.mean()),
                                   "peak_to_exit_sum": float(group.peak_to_exit_difference.sum()) if category != "UNFINISHED" else np.nan,
                                   "positive_peak_sum": float(group.peak_close_liquidation_pnl.clip(lower=0).sum()),
                                   "observed_terminal_net_mark": float(group.terminal_net_mark.sum())})
            for phase in ("entry_day_net_pnl", "holding_interior_net_pnl", "exit_day_net_pnl", "post_exit_dividend_net_pnl"):
                phases.append({**prefix, "phase": phase, "all_complete_net_pnl": float(complete[phase].sum()),
                               "winning_cycles_net_pnl": float(complete.loc[complete.net_pnl.gt(0), phase].sum()),
                               "losing_cycles_net_pnl": float(losses[phase].sum())})
            for reason, group in complete.groupby("exit_reason", dropna=False):
                exits.append({**prefix, "exit_reason": reason, "count": len(group), "net_pnl": float(group.net_pnl.sum()),
                              "last_close_mark_difference": float(group.final_open_vs_prior_close_mark.sum())})
            err = float((daily.saved_equity - daily.reconstructed_equity).abs().max())
            checks.append({**prefix, "days": len(daily), "cycles": len(cycles), "path_rows": len(paths),
                           "maximum_daily_equity_error": err, "actual_orders_unchanged": True})
            print(f"{period}/{cost}：完成{len(complete)}笔、亏损{len(losses)}笔，其中曾有净浮盈{record['losses_after_positive_close_mark']}笔；退出开盘相对收盘标记合计{record['completed_exit_mark_difference']:+,.2f}元。", flush=True)
    for name, rows in (("账户损益来源", records), ("亏损路径分类", categories), ("完整周期阶段贡献", phases), ("真实退出原因", exits), ("路径还原检查", checks)):
        table(name, pd.DataFrame(rows))
    common.save_json(OUT / "summary.json", {"study": "510300_POINT_WEIGHT_PATH_BOTTLENECK_V1", "at": common.now(),
        "status": "COMPLETED_FIXED_ACCOUNT_PATH_DIAGNOSTIC", "accounts_described": 4,
        "new_account_evaluations": 0, "new_model_fits": 0, "new_parameters": 0,
        "necessary_tests_passed": 3, "results": records, "checks": checks,
        "independent_validation": "NOT_ESTABLISHED", "new_strategy_validated": False,
        "goal_achieved": False, "orders_authorized": False})


def report():
    check_frozen()
    s = read(OUT / "summary.json")
    rows = pd.read_parquet(OUT / "results/账户损益来源.parquet")
    categories = pd.read_parquet(OUT / "results/亏损路径分类.parquet")
    phases = pd.read_parquet(OUT / "results/完整周期阶段贡献.parquet")
    exits = pd.read_parquet(OUT / "results/真实退出原因.parquet")
    lines = ["# 当前主候选：实际仓位的入场、持有和退出损益", "",
             "本轮固定A的四份已有账户，补上真实加减仓后的路径解释。不新增策略、账户回测、拟合或参数；所有历史继续作为开发资料。", "",
             "逐日用真实旧库存计算隔夜原价差、真实开盘后库存计算日内原价差，加按登记资格确认的股息，减佣金及滑点。四份账户的权益逐日完整还原。", "",
             "下表含末端未完成周期的市值；完整周期统计另列。两个时期分别20万元启动，不能把表中损益直接拼成一个账户。", "",
             "| 时期 | 成本 | 隔夜价格损益 | 日内价格损益 | 股息 | 佣金与滑点 | 账户净损益 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows.itertuples():
        lines.append(f"| {r.period} | {r.cost} | {r.overnight_price_pnl:,.2f} | {r.intraday_price_pnl:,.2f} | {r.dividend_accrual:,.2f} | {r.commission+r.slippage:,.2f} | {r.account_total_net_pnl:,.2f} |")
    lines += ["", "金额单位为元。隔夜与日内是损益发生的时间段，不是两条可以独立执行的策略；尤其不能假定T+1的ETF能在买入当日收盘退出。", "",
              "| 时期 | 成本 | 完整周期 | 亏损周期 | 曾有净浮盈后亏损 | 未见净浮盈而亏损 | 未完成周期 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows.itertuples():
        lines.append(f"| {r.period} | {r.cost} | {r.completed_cycles} | {r.losses} | {r.losses_after_positive_close_mark} | {r.losses_without_positive_close_mark} | {r.unfinished_cycles} |")
    lines += ["", "净浮盈按每个已观察持有收盘的真实累计现金流、剩余库存价值和已确认股息，再扣一次假设清仓摩擦计算。它不表示能在看到收盘后按收盘价成交。最高值及其日期是事后标签，绝不作为当时的输入。", "",
              "| 压力成本时期 | 路径类别 | 笔数 | 实际净损益 | 正浮盈峰值合计 | 峰值至最终差合计 |",
              "|---|---|---:|---:|---:|---:|"]
    for r in categories.loc[categories.cost.eq("STRESS") & categories.category.ne("UNFINISHED")].itertuples():
        lines.append(f"| {r.period} | {PATH_LABELS[r.category]} | {r.count} | {r.net_pnl:,.2f} | {r.positive_peak_sum:,.2f} | {r.peak_to_exit_sum:,.2f} |")
    lines += ["", "峰值至最终差并非可得利润，也不是某个止盈方法能提高的收益；加总峰值同样不能形成真实账户。", "",
              "| 时期 | 成本 | 最后持有收盘标记至真实清仓差 | 首次清仓决定标记至真实清仓差 | 超过下一交易日才清仓的周期 |",
              "|---|---|---:|---:|---:|"]
    for r in rows.itertuples():
        lines.append(f"| {r.period} | {r.cost} | {r.completed_exit_mark_difference:+,.2f} | {r.completed_first_intent_mark_difference:+,.2f} | {r.delayed_exit_cycles} |")
    lines += ["", "正值表示实际退出比相关收盘清算标记更有利，负值表示更不利。这个差同时含价格变化、股息和费用差，不能直接归因于可消除的滑点。", "",
              "| 压力成本时期 | 阶段 | 全部完整周期净损益 | 盈利周期贡献 | 亏损周期贡献 |",
              "|---|---|---:|---:|---:|"]
    phase_names = {"entry_day_net_pnl": "入场日", "holding_interior_net_pnl": "持有中间阶段", "exit_day_net_pnl": "退出日", "post_exit_dividend_net_pnl": "退出后股息确认"}
    for r in phases.loc[phases.cost.eq("STRESS")].itertuples():
        lines.append(f"| {r.period} | {phase_names[r.phase]} | {r.all_complete_net_pnl:,.2f} | {r.winning_cycles_net_pnl:,.2f} | {r.losing_cycles_net_pnl:,.2f} |")
    lines += ["", "盈利和亏损分类使用最终结果，属于解释分组，不能据此筛掉某一类未来交易。阶段净损益相加等于完整周期净损益；它们不是不同持有期限的回测。", "",
              "| 时期 | 成本 | 真实清仓原因 | 完整周期 | 净损益 |",
              "|---|---|---|---:|---:|"]
    for r in exits.itertuples():
        lines.append(f"| {r.period} | {r.cost} | {r.exit_reason} | {r.count} | {r.net_pnl:,.2f} |")
    lines += ["", "本轮不把浮盈回吐自动解释为止盈太迟，也不把隔夜、入场日或亏损组的事后结果用作新过滤。调整任何退出规则仍需要事前可用的辨别信息，以及保留全部结果的同口径账户比较。", "",
              "[动态交易与成本研究](https://www.nber.org/papers/w15205)讨论了预测收益、信号衰减与交易摩擦的共同作用；[回测过拟合研究](https://www.davidhbailey.com/dhbpapers/overfitting.pdf)说明重复选择历史方案需要额外处理选择偏差。两篇论文均不证明本模型对510300有效。", "",
              "三项针对性测试通过，覆盖实际现金流和股息、未来变化不改前段路径、错误库存与损益拒绝。样本截至2026年9月30日；未完成周期未强平，四个场景不是四组独立市场证据。", "",
              "结果入口：results/账户损益来源.csv、results/亏损路径分类.csv、results/完整周期阶段贡献.csv，各时期成本子目录保存完整周期和逐日路径。"]
    text = "\n".join(lines) + "\n"
    (OUT / "研究结论.md").write_text(text, encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = ["隔夜价差", "日内价差", "股息", "佣金滑点", "账户净损益"]
    for ax, r in zip(axes, rows.loc[rows.cost.eq("STRESS")].itertuples()):
        values = np.array([r.overnight_price_pnl, r.intraday_price_pnl, r.dividend_accrual, -r.commission-r.slippage, r.account_total_net_pnl]) / 10000
        colors = ["#4278a8" if v >= 0 else "#b45353" for v in values]
        colors[-1] = "#28766c"
        ax.bar(labels, values, color=colors, width=.65)
        ax.axhline(0, color="#666666", linewidth=.7)
        ax.set_title("2015—2019年" if r.period == "2015_2019" else "2020—2026年9月30日")
        ax.set_ylabel("损益（万元）")
        ax.tick_params(axis="x", labelrotation=12)
        ax.margins(y=.2)
        for i, value in enumerate(values):
            ax.annotate(f"{value:+.2f}", (i, value), xytext=(0, 4 if value >= 0 else -4), textcoords="offset points", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    fig.suptitle("主候选真实持仓损益来源｜20万元账户，压力成本", fontsize=14)
    fig.text(.5, .018, "两个时期分别启动；分项是实际损益归因，不是可独立执行的策略。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(OUT / "实际仓位损益来源.png", dpi=160)
    fig.savefig(OUT / "实际仓位损益来源.svg")
    plt.close(fig)
    common.save_json(OUT / "delivery_receipt.json", {"at": common.now(), "report_sha256": common.digest(OUT / "研究结论.md"),
        "summary_sha256": common.digest(OUT / "summary.json"), "new_account_evaluations": 0,
        "saved_accounts_described": 4, "status": s["status"]})
    print("已生成真实仓位路径说明及损益图；没有把峰值当作可执行收益。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="当前主候选的实际仓位损益分解")
    parser.add_argument("action", choices=("freeze", "run", "report"))
    globals()[parser.parse_args().action]()
