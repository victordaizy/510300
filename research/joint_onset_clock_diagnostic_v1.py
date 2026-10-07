"""全体联合出生触发时钟的固定解释；不运行新账户或选择过滤条件。"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import joint_onset_clock_inputs_v1 as clock_rules
from research.trend_expansion_phase_study_v1 import load, OUT as PRIOR, EXPLANATION, saved_account
from research.point_first_passage_study_v1 import read, write_json, require, digest, now
from research.point_account_nr7_inputs_v1 import trade_statistics, PARENT_A

OUT = ROOT / "reports/research/510300_joint_onset_clock_diagnostic_v1"
TEST = ROOT / "tests/test_joint_onset_clock_v1.py"
ERAS = {"ALL": ("2015-01-01", "2026-09-30"), "2015_2019": ("2015-01-01", "2019-12-31"),
        "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-30")}
NAMES = {"PRICE_LAST": "价格最后到达", "DAILY_DIF_LAST": "日DIF最后到达",
         "PREVIOUS_COMPLETE_WEEK_LAST": "上一完整周柱最后到达", "MULTIPLE_SAME_ORIGIN": "同日多项到达"}


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def tests():
    require(not (OUT / "tests_receipt.json").exists(), "必要测试已有记录，不自动重试。")
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", str(TEST)], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8")
    output = result.stdout + result.stderr
    write_json(OUT / "tests_receipt.json", {
        "at": now(), "exit_code": result.returncode, "passed": 3 if result.returncode == 0 and "3 passed" in output else 0,
        "output": output, "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                                    for p in [Path(clock_rules.__file__), TEST]],
    }, exclusive=True)
    print(output, end="", flush=True)
    require(result.returncode == 0 and "3 passed" in output, "三必要时钟测试未通过，保存首次结果。")


def freeze():
    require(not (OUT / "protocol.json").exists(), "触发时钟解释已经固定。")
    receipt = read(OUT / "tests_receipt.json")
    require(receipt["passed"] == 3 and receipt["exit_code"] == 0, "必要测试未通过。")
    for s in receipt["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "归因版本不对应测试。")
    sources = {}
    for folder in [EXPLANATION, PRIOR]:
        for s in read(folder / "protocol.json")["sources"]:
            require(digest(ROOT / s["path"]) == s["sha256"], "原已完成研究来源改变。")
            sources[s["path"]] = s["sha256"]
    data, _, _, _ = load()
    clocks = clock_rules.clocks(data)
    phases = pd.read_parquet(EXPLANATION / "results/全部原点_当时日周展开阶段.parquet")
    pd.testing.assert_series_equal(clocks.joint_phase_onset, phases.joint_phase_onset, check_exact=True)
    require(int(clocks.loc[clocks.date.ge("2015-01-01")].joint_phase_onset.sum()) == 85, "原85出生母集改变。")
    prefixes = []
    for end in ["2019-01-18", "2020-06-08", "2024-09-30"]:
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        pd.testing.assert_frame_equal(clock_rules.clocks(local), clocks.iloc[:len(local)].reset_index(drop=True), check_exact=True)
        prefixes.append({"through": end, "rows": len(local), "status": "PASS_CLOCK_PREFIX"})
    write_json(OUT / "preflight.json", {"at": now(), "status": "PASS_3488_ORIGINAL_BIRTH_FLAGS_AND_THREE_PREFIXES",
                                      "prefixes": prefixes, "new_accounts": 0}, exclusive=True)
    paths = [Path(__file__), Path(clock_rules.__file__), TEST, OUT / "tests_receipt.json", OUT / "preflight.json",
             EXPLANATION / "protocol.json", EXPLANATION / "summary.json",
             EXPLANATION / "results/全部原点_当时日周展开阶段.parquet", PRIOR / "protocol.json", PRIOR / "summary.json",
             PRIOR / "saved_result_verification.json", PRIOR / "actual_point_detail_summary.json",
             PRIOR / "next_all_event_clock_diagnostic_proposal.json",
             PRIOR / "results/全部合格出生信号_真实执行与未成交.parquet",
             PRIOR / "results/全部实际进出点位_当时指标与A覆盖.parquet"]
    for period in ("2015_2019", "2020_2026"):
        for cost in ("BASE", "STRESS"):
            paths.extend(PRIOR / f"results/accounts/{period}/{cost}/JOINT_PHASE_START/{name}.parquet"
                         for name in ("daily", "orders", "trades", "decisions", "rejections"))
            paths.append(PRIOR / f"results/accounts/{period}/{cost}/JOINT_PHASE_START/terminal.json")
    for p in paths:
        sources[p.relative_to(ROOT).as_posix()] = digest(p)
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_ALL_JOINT_ONSET_CLOCK_DIAGNOSTIC_V1", "technical_decision": "TECH.R166",
        "hypothesis": "周线确认迟到与均线附近反复重新成立可能是不同错误；按原85出生的最后到达时钟解释实际进出，不能以分组反选过滤营救R165。",
        "classification": "当前与前一已知原点比较价/EMA20、日DIF零轴、上一完整周柱零轴；一个新增项按该项，多项同日独立MULTIPLE；未知单列，不伪造出生。",
        "population": "3488全日先标，原85事件、84原实际周期（83完成/1开放）及1未成交全保留。原两费用、ALL与三个原时期全部32单元，不挑赢家/费用/时期。",
        "metrics": "仅已保存原周期净回报/毛损益/费用及A当时库存目标；开放/未成交未知保留。分组是原账户子集，未独立重配现金，不计算分组策略收益或Sharpe。",
        "gross_decomposition": "完成周期实际原始开盘现金流+既有股息，减原实际佣金及滑点，逐周期复算净损益；不是零费用反事实。",
        "old_label_gate": "不读取原20日标签作为实际胜率，不生成训练标签；出生分类不读取任何未来结果或事后波段。",
        "no_rescue": "不把较好触发类变成R165过滤、不改MACD/EMA/周线延迟/窗口/退出/费用或时期；旧固定政策保持关闭。",
        "necessary_tests": 3, "actual_prefix_checks": 3, "new_accounts": 0, "new_model_fits": 0,
        "new_training_labels": 0, "new_market_requests": 0, "independent_validation": "NOT_ESTABLISHED",
        "source_first_vintage": "NOT_CERTIFIED", "goal_achieved": False,
        "sources": [{"path": k, "sha256": v} for k, v in sorted(sources.items())],
    }, exclusive=True)
    print("三时钟及同日多项解释已冻结，原3488出生标记与三个前缀一致；尚未按结果统计分组。", flush=True)


def plot(events):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    local = events.loc[events.cost.eq("STRESS")]
    closed = local.loc[local.actual_status.eq("COMPLETE")]
    colors = ["#26638e", "#ba782c", "#398071", "#775694"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [2, 1]})
    for kind, color in zip(clock_rules.CLOCKS, colors):
        rows = closed.loc[closed.birth_clock.eq(kind)]
        axes[0].scatter(rows.origin, rows.actual_net_return*100, color=color, s=30, label=NAMES[kind], alpha=.8)
    axes[0].axhline(0, color="#777777", linewidth=.8)
    axes[0].axvline(pd.Timestamp("2020-01-01"), color="#999999", linestyle="--", linewidth=.7)
    axes[0].set_ylabel("原完成周期净回报 / %")
    axes[0].set_title("全体已知出生：83个原完成周期；1开放与1未成交保留在下图")
    axes[0].legend(loc="upper right", fontsize=9)
    x = np.arange(len(clock_rules.CLOCKS))
    bottom = np.zeros(len(x))
    for name, label, color in [("WIN", "完成盈利", "#398071"), ("LOSS", "完成亏损", "#b45b55"),
                               ("OPEN", "开放未知", "#aab2ba"), ("BLOCKED", "次开未成交", "#343f4a")]:
        counts = []
        for kind in clock_rules.CLOCKS:
            group = local.loc[local.birth_clock.eq(kind)]
            if name == "WIN":
                n = int((group.actual_status.eq("COMPLETE") & group.actual_net_return.gt(0)).sum())
            elif name == "LOSS":
                n = int((group.actual_status.eq("COMPLETE") & group.actual_net_return.lt(0)).sum())
            elif name == "OPEN":
                n = int(group.execution_status.eq("ACTUAL_OPEN").sum())
            else:
                n = int(group.execution_status.eq("REJECTED_AT_ACTUAL_OPEN").sum())
            counts.append(n)
        axes[1].bar(x, counts, bottom=bottom, color=color, label=label, width=.55)
        bottom += counts
    for i, value in enumerate(bottom):
        axes[1].text(i, value+.5, str(int(value)), ha="center")
    axes[1].set_xticks(x, [NAMES[k] for k in clock_rules.CLOCKS])
    axes[1].set_ylabel("全体资格数")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.suptitle("触发时钟解释：原固定账户子集，不是四条新策略", fontsize=15)
    fig.tight_layout()
    path = OUT / "全部85出生_时钟与实际结果.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "本固定解释已经开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "冻结来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_accounts": 0}, exclusive=True)
    data, _, _, parents = load()
    known = clock_rules.clocks(data)
    table("全部原点_事前最后到达时钟", known)
    statuses = pd.read_parquet(PRIOR / "results/全部合格出生信号_真实执行与未成交.parquet")
    points = pd.read_parquet(PRIOR / "results/全部实际进出点位_当时指标与A覆盖.parquet")
    selected = ["period", "cost", "cycle_id", "status", "net_return", "net_pnl", "dividend_cny", "buy_debit",
                "daily_hist", "relative_volume", "up_volume_balance5", "rv_ratio"]
    actual = points[selected].rename(columns={c: "actual_"+c for c in selected if c not in ("period", "cost", "cycle_id")})
    events = statuses.merge(known.loc[known.joint_phase_onset].rename(columns={"date": "origin"}),
                            on="origin", how="left", validate="many_to_one")
    require(len(events) == 170 and events.birth_clock.isin(clock_rules.CLOCKS).all(), "全部85原资格或时钟匹配不完整。")
    events = events.merge(actual, on=["period", "cost", "cycle_id"], how="left", validate="one_to_one")
    matched = events.execution_status.isin(["ACTUAL_COMPLETE", "ACTUAL_OPEN"])
    np.testing.assert_allclose(events.loc[matched, "net_return"], events.loc[matched, "actual_net_return"],
                               rtol=0, atol=0, equal_nan=True)
    for period in ("2015_2019", "2020_2026"):
        targets = parents[period].set_index("origin")[PARENT_A]
        for cost in ("BASE", "STRESS"):
            mask = events.period.eq(period) & events.cost.eq(cost)
            baseline, _ = saved_account(period, cost, "A_SAVED_WEIGHT")
            inv = baseline["daily"].set_index("date")
            events.loc[mask, "A_origin_inventory_known"] = events.loc[mask, "origin"].isin(inv.index)
            events.loc[mask, "A_origin_shares"] = events.loc[mask, "origin"].map(inv.shares)
            events.loc[mask, "A_origin_target"] = events.loc[mask, "origin"].map(targets)
            orders = pd.read_parquet(PRIOR / f"results/accounts/{period}/{cost}/JOINT_PHASE_START/orders.parquet")
            orders["raw_cash_flow"] = np.where(orders.side.eq("SELL"), 1, -1)*orders.quantity*orders.raw_open
            amounts = orders.groupby("cycle_id")[["raw_cash_flow", "commission", "slippage"]].sum()
            for i in events.index[mask & events.actual_status.eq("COMPLETE")]:
                row = events.loc[i]
                value = amounts.loc[int(row.cycle_id)]
                gross = float(value.raw_cash_flow + row.actual_dividend_cny)
                require(abs(gross-value.commission-value.slippage-row.actual_net_pnl) < 1e-7, "原周期毛净费用恒等式失败。")
                events.loc[i, ["actual_gross_pnl", "actual_commission", "actual_slippage"]] = [gross, value.commission, value.slippage]
    require(events.A_origin_target.notna().all(), "当时原A目标缺失，不可填零。")
    table("全部85资格及原实际周期_触发时钟与A覆盖", events)
    stats = []
    for cost in ("BASE", "STRESS"):
        for era, (start, end) in ERAS.items():
            sample = events.loc[events.cost.eq(cost) & events.origin.between(start, end)]
            for kind in clock_rules.CLOCKS:
                group = sample.loc[sample.birth_clock.eq(kind)]
                complete = group.loc[group.actual_status.eq("COMPLETE")]
                trade = complete[["actual_status", "actual_net_return", "actual_net_pnl"]].rename(
                    columns={"actual_status": "status", "actual_net_return": "net_return", "actual_net_pnl": "net_pnl"})
                stats.append({
                    "era": era, "cost": cost, "birth_clock": kind, "known_events": len(group),
                    "open_cycles": int(group.execution_status.eq("ACTUAL_OPEN").sum()),
                    "unexecuted_origins": int(group.execution_status.eq("REJECTED_AT_ACTUAL_OPEN").sum()),
                    **trade_statistics(trade),
                    "mean_actual_gross_pnl": float(complete.actual_gross_pnl.mean()) if len(complete) else np.nan,
                    "completed_actual_gross_pnl": float(complete.actual_gross_pnl.sum()),
                    "completed_actual_commission": float(complete.actual_commission.sum()),
                    "completed_actual_slippage": float(complete.actual_slippage.sum()),
                    "A_positive_inventory": int(group.A_origin_shares.gt(0).sum()),
                    "A_positive_known_target": int(group.A_origin_target.gt(0).sum()),
                    "standalone_policy_or_sharpe": False,
                })
    result = pd.DataFrame(stats)
    require(len(result) == 32, "原四类四时期两费用的全32单元缺失。")
    table("四时钟原实际分组_全部32单元", result)
    cases = events.loc[events.origin.isin(pd.to_datetime(["2015-06-17", "2019-01-18", "2019-03-27",
                                                        "2020-06-08", "2020-06-16", "2024-09-30", "2024-11-20"]))].copy()
    table("四原案例关键出生_事前时钟与真实结果", cases)
    chart = plot(events)
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "解释执行期间来源改变。")
    write_json(OUT / "summary.json", {
        "at": now(), "study": protocol["study"], "technical_decision": "TECH.R166",
        "status": "COMPLETED_FIXED_ALL_EVENT_CLOCK_EXPLANATION_NO_NEW_POLICY",
        "all_known_daily_rows": len(known), "unique_known_births": int(events.origin.nunique()),
        "actual_cycles_ignoring_cost_replicates": int(events.loc[matched, ["period", "cycle_id"]].drop_duplicates().shape[0]),
        "original_completed_cycles_per_cost": int(events.loc[events.cost.eq("STRESS") & events.actual_status.eq("COMPLETE")].shape[0]),
        "original_open_cycles_per_cost": int(events.loc[events.cost.eq("STRESS") & events.execution_status.eq("ACTUAL_OPEN")].shape[0]),
        "unexecuted_origins_per_cost": int(events.loc[events.cost.eq("STRESS") & events.execution_status.eq("REJECTED_AT_ACTUAL_OPEN")].shape[0]),
        "clock_counts": events.loc[events.cost.eq("STRESS")].birth_clock.value_counts().to_dict(),
        "all_group_cells": len(result), "necessary_tests": 3, "actual_prefix_checks": 3,
        "frozen_sources": len(protocol["sources"]), "chart": chart,
        "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "latest_financial_strategy_decision_preserved": "TECH.R165", "next_financial_candidate": "NOT_REGISTERED_NO_READY_CANDIDATE",
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False,
    }, exclusive=True)
    print(result.loc[result.cost.eq("STRESS"), ["era", "birth_clock", "known_events", "completed_cycles", "wins", "win_rate",
                                                 "payoff", "p_times_b", "mean_cycle_net_return", "completed_cycle_net_pnl"]].to_string(index=False), flush=True)
    print("全体原实际时钟归因一次完成；未运行新策略账户或选择过滤。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="全体85联合出生的事前时钟与原实际结果解释。")
    parser.add_argument("command", choices=("tests", "freeze", "run"))
    args = parser.parse_args()
    {"tests": tests, "freeze": freeze, "run": run}[args.command]()
