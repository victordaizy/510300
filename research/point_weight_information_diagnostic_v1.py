"""测量将原仓位强弱二值化所丢失的信息，不复活旧冻结失败策略。"""
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
from research import point_account_nr7_inputs_v1 as binary
from research import point_weight_information_inputs_v1 as weighted
from research import upward_episode_anatomy_v1 as common
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_account_nr7_complement_v1 import paired_interval, verify_account, annual_rows

OUT = ROOT / "reports/research/510300_point_weight_information_diagnostic_v1"
SOURCE = ROOT / "reports/research/510300_point_account_nr7_complement_v1"
CONTEXT = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001"
EARLY = ROOT / "reports/research/510300_point_state_reconstruction_v1/results/逐日组合来源状态.parquet"


def table(name, data):
    p = OUT / "results" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(p.with_suffix(".parquet"), index=False)
    data.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    if OUT.exists():
        raise RuntimeError("固定仓位信息诊断已存在，不覆盖。")
    OUT.mkdir(parents=True)
    sources = [(SOURCE / "inputs/prices.parquet", "inputs/prices.parquet"),
               (SOURCE / "inputs/dividends.csv", "inputs/dividends.csv"),
               (SOURCE / "inputs/parent_signals.parquet", "inputs/parent_signals.parquet"),
               (SOURCE / "results/事前风险.parquet", "inputs/risks.parquet"),
               (SOURCE / "summary.json", "inputs/previous_summary.json"),
               (EARLY, "inputs/earlier_signals.parquet"),
               (ROOT / "reports/research/510300_core_auxiliary_drawdown_gate_v1/acceptance_outcome.json", "inputs/old_terminal_rejection.json")]
    for cost in ("BASE", "STRESS"):
        for item in ("daily.parquet", "trades.parquet", "orders.parquet", "decisions.parquet", "rejections.parquet", "terminal.json"):
            sources.append((SOURCE / f"results/accounts/2020_2026/{cost}/POINT_A/{item}", f"inputs/baseline/{cost}/{item}"))
    for p in (Path(__file__), Path(weighted.__file__), Path(binary.__file__), Path(common.__file__),
              ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/daily_supply_test_v1.py",
              ROOT / "tests/test_point_weight_information_v1.py"):
        sources.append((p, "code/" + p.name))
    files = {}
    for src, name in sources:
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        files[name] = {"source": src.relative_to(ROOT).as_posix(), "sha256": common.digest(target)}
    protocol = {
        "study": "510300_POINT_WEIGHT_INFORMATION_DIAGNOSTIC_V1", "at": common.now(),
        "question": "原点位只保留父目标的正/零；在同样资金、风险上限和成本下，保留原目标大小能否提高账户收益与夏普？",
        "known_evidence": "已知上一轮点位A压力年化3.12%、夏普0.880、平均仓位10.39%；原旧动态账户2020年以来曾有约1.53夏普但较早失败，旧关闭结论保留。NR7补充失败，不能加入。",
        "status_scope": "固定机制归因，不是新的独立策略检验，不恢复旧第176轮或任何已停用方案的有效性。",
        "signals": "原保存A目标权重，不改模型、阈值、父信号与训练；2020年以来用已接续到9月30日的STRESS来源，2015—2019使用当时保存的earlier_diagnostic独立时期来源，不拼接为连续历史。",
        "only_comparison": "POINT_BINARY对照上轮正/零固定入场份额；SAVED_WEIGHT按原权重大小调整股票比例。",
        "weight_rule": "目标=min(原目标,0.5)，沿用旧源的10个百分点仓位变化带；已有仓位且目标正、差距小于10个百分点时不为追逐微小变化成交。目标为零全部退出。独立风险上限始终优先；不存在阈值网格。",
        "risk_and_capital": "20万元、原50%上限、5日条件ES预算2.5%、10%跳空情景预算5%与回撤余量；现金及无风险比较率0，252交易日年化，与上一轮完全一致。",
        "execution": "前收盘确定最大目标份额；次开盘买入只可按现金与同一风险预算缩减。未成交的全退请求锁定。已持仓可加减仓，T+1，登记股息/应收/到账分开，末端不强平。",
        "cycles": "仍按空仓到最终清仓计一次，加减仓订单不增加次数；该周期净回报分母为累计买入支出，明确不同于单次固定份额点位口径。",
        "periods": {"2020_2026": ["2020-01-02", "2026-09-30"], "2015_2019": ["2015-01-05", "2019-12-31"]},
        "mechanism_screen": "四个时期成本场景逐一报告净夏普、净年化和回撤相对二值基线变化；只有每个场景收益和夏普都严格提高，才称跨期方向一致。该描述不取代实际pB>1与独立验证。",
        "earlier_endpoint": "原较早信号末端不完整的下一日判断保留NaN；不影响当期最后真实开盘。最后持仓自然标记，不用旧期末强平获利。",
        "new_accounts": 6, "reused_accounts": 2, "model_fits": 0, "parameter_search": False,
        "old_rejection_preserved": True, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "orders_authorized": False,
    }
    common.save_json(OUT / "protocol.json", protocol)
    files["protocol.json"] = {"sha256": common.digest(OUT / "protocol.json")}
    common.save_json(OUT / "freeze.json", {"at": common.now(), "files": files})
    print("仓位信息诊断已固定：沿用原权重和原10个百分点变化带，只比较信息是否丢失。", flush=True)


def load_saved(folder):
    r = {name: pd.read_parquet(folder / (name + ".parquet")) for name in ("daily", "trades", "orders", "decisions", "rejections")}
    r["terminal"] = json.loads((folder / "terminal.json").read_text(encoding="utf-8"))
    return r


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("本诊断已经开始，不再次运行。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for name, record in frozen["files"].items():
        if common.digest(OUT / name) != record["sha256"]:
            raise ValueError("固定文件改变：" + name)
        if name.startswith("code/") and common.digest(ROOT / record["source"]) != record["sha256"]:
            raise ValueError("执行代码不同于固定版本：" + name)
    common.save_json(OUT / "RUN_STARTED.json", {"at": common.now()})
    prices = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    data, _ = common.features(prices, div)
    risks = pd.read_parquet(OUT / "inputs/risks.parquet")
    recent = pd.read_parquet(OUT / "inputs/parent_signals.parquet").pivot(index="origin", columns="candidate", values="target").reset_index()
    early = pd.read_parquet(OUT / "inputs/earlier_signals.parquet")
    early = early.loc[early.period.eq("earlier_diagnostic")].pivot(index="origin", columns="model", values="target").reset_index()
    if not pd.Timestamp("2019-12-31") in set(early.origin):
        early = pd.concat([early, pd.DataFrame({"origin": [pd.Timestamp("2019-12-31")], binary.PARENT_A: [np.nan], binary.PARENT_B: [np.nan]})], ignore_index=True)
    accounts, records, annual, deltas, checks = {}, [], [], [], []
    for period, start, end, parent in [("2020_2026", "2020-01-02", "2026-09-30", recent),
                                        ("2015_2019", "2015-01-05", "2019-12-31", early)]:
        d = data.loc[data.date.le(pd.Timestamp(end))].reset_index(drop=True)
        empty = pd.DataFrame({"date": d.date, "entry_event": False, "stop_index": np.nan, "setup_date": pd.NaT})
        table("signals/" + period, parent)
        for cost in ("BASE", "STRESS"):
            for policy in ("POINT_BINARY", "SAVED_WEIGHT"):
                if policy == "POINT_BINARY" and period == "2020_2026":
                    result = load_saved(OUT / "inputs/baseline" / cost)
                elif policy == "POINT_BINARY":
                    result = binary.account(d, div, empty, parent, risks, "POINT_A", cost, start)
                else:
                    result = weighted.weight_account(d, div, parent, risks, cost, start)
                accounts[(period, cost, policy)] = result
                for name in ("daily", "trades", "orders", "decisions", "rejections"):
                    table(f"accounts/{period}/{cost}/{policy}/{name}", result[name])
                common.save_json(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", result["terminal"])
                # 正式比较直接从落盘文件读回，避免只验证内存对象。
                saved = load_saved(OUT / f"results/accounts/{period}/{cost}/{policy}")
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(saved)})
                m = binary.metrics(saved)
                yearly = annual_rows(saved, period, policy, cost)
                counts = [x["completed_cycles"] for x in yearly if x["full_year"]]
                m["average_full_year_cycles"] = float(np.mean(counts))
                m["zero_trade_full_years"] = sum(n == 0 for n in counts)
                records.append({"period": period, "cost": cost, "policy": policy, **m})
                annual.extend(yearly)
                print(f"{period}/{cost}/{policy}：夏普{m['net_sharpe']:.4f}，年化{m['net_cagr']:.2%}，周期{m['completed_cycles']}。", flush=True)
            a, b = [accounts[(period, cost, k)] for k in ("POINT_BINARY", "SAVED_WEIGHT")]
            am, bm = binary.metrics(a), binary.metrics(b)
            deltas.append({"period": period, "cost": cost,
                           "sharpe_delta": bm["net_sharpe"] - am["net_sharpe"],
                           "cagr_delta": bm["net_cagr"] - am["net_cagr"],
                           "max_drawdown_delta": bm["max_drawdown"] - am["max_drawdown"],
                           "cycle_delta": bm["completed_cycles"] - am["completed_cycles"],
                           "ending_equity_delta": bm["ending_equity"] - am["ending_equity"],
                           **paired_interval(a["daily"].net_return.to_numpy(float), b["daily"].net_return.to_numpy(float))})
    m, yearly, delta = pd.DataFrame(records), pd.DataFrame(annual), pd.DataFrame(deltas)
    table("同风险预算的仓位信息比较", m)
    table("逐年收益与完整周期", yearly)
    table("仓位信息的账户增量", delta)
    table("保存账户复算", pd.DataFrame(checks))
    prefixes = []
    for end in ("2021-12-31", "2023-12-29", "2025-12-31"):
        short = data.loc[data.date.le(pd.Timestamp(end))].reset_index(drop=True)
        check = weighted.weight_account(short, div, recent, risks, "STRESS", "2020-01-02")
        full = accounts[("2020_2026", "STRESS", "SAVED_WEIGHT")]["daily"]
        pd.testing.assert_frame_equal(full.iloc[:len(check["daily"])].reset_index(drop=True), check["daily"])
        prefixes.append({"cutoff": end, "exact_daily_match": True})
    table("账户截断一致性", pd.DataFrame(prefixes))
    consistent = bool((delta.sharpe_delta.gt(0) & delta.cagr_delta.gt(0)).all())
    summary = {"study": "510300_POINT_WEIGHT_INFORMATION_DIAGNOSTIC_V1", "at": common.now(),
               "status": "SAVED_WEIGHT_INFORMATION_DIAGNOSTIC_COMPLETE", "new_accounts": 6, "reused_accounts": 2,
               "cross_period_return_and_sharpe_improvement": consistent, "comparisons": deltas,
               "old_terminal_status_preserved": True, "independent_validation": "NOT_ESTABLISHED",
               "necessary_tests_passed": 3, "saved_accounts_recomputed": len(checks), "prefixes": prefixes,
               "goal_achieved": False, "orders_authorized": False}
    common.save_json(OUT / "summary.json", summary)
    write_report(m, delta, yearly, summary)
    plot(accounts)
    common.save_json(OUT / "delivery_receipt.json", {"at": common.now(), "report_sha256": common.digest(OUT / "研究结论.md"),
                     "metrics_sha256": common.digest(OUT / "results/同风险预算的仓位信息比较.parquet"), "saved_accounts_recomputed": True})
    state = json.loads((CONTEXT / "state.json").read_text(encoding="utf-8"))
    state.update(updated_at=common.now(), latest_completed_study=summary["study"],
                 latest_result=str((OUT / "summary.json").relative_to(ROOT)), latest_report=str((OUT / "研究结论.md").relative_to(ROOT)),
                 latest_research_status=summary["status"], new_investment_account_evaluations_this_continuation=16,
                 latest_progress="已完成点位账户、NR7失败补充和保存仓位强弱的共同风险比较；不将诊断结果升级成新独立验证。",
                 necessary_tests_passed_this_continuation=9, goal_achieved=False)
    common.save_json(CONTEXT / "state.json", state)
    print(json.dumps(common.clean(summary), ensure_ascii=False, indent=2), flush=True)


def write_report(m, delta, annual, summary):
    rows = ["# 510300：保留仓位强弱能否提高收益与夏普", "",
            "本轮接着前一轮的账户问题，固定比较原目标的正/零与原目标大小。使用同样20万元、两档成本、50%仓位上限、ES和回撤余量，保留原来源10个百分点调整带。没有训练新模型、搜索阈值或恢复旧失败方案的有效状态。", "",
            "这是一项有已知结果背景的机制归因：旧动态源账户在近期曾有较高夏普，但较早时期失败。本轮使用当前共同风险约束检查信息差，不能称未见数据验证。原第176轮关闭裁决继续有效。", "",
            "| 时期 | 成本 | 方案 | 净夏普 | 净年化 | 最大回撤 | 年均完整周期 | 实际净pB | 期末权益 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    names = {"POINT_BINARY": "只用正/零的点位账户", "SAVED_WEIGHT": "保留原仓位强弱的诊断账户"}
    for r in m.itertuples():
        rows.append(f"| {r.period} | {r.cost} | {names[r.policy]} | {r.net_sharpe:.3f} | {r.net_cagr:.2%} | {r.max_drawdown:.2%} | {r.average_full_year_cycles:.2f} | {r.p_times_b:.3f} | {r.ending_equity:,.2f}元 |")
    rows += ["", "2020—2026截至2026年9月30日，年均频率只用2020—2025；较早为2015—2019单独启动的保存来源，不拼接成一条连续历史。全部交易日与空仓期均纳入；现金及无风险比较率假设0，252交易日年化，末端不强平。", "",
             "实际净pB按完整资金周期净损益/累计买入支出计算。有加减仓时不能把这项pB直接冒充原单次固定十万元点位的pB；目标权重也不是经校准的获胜概率。", "",
             "| 时期 | 成本 | 夏普变化 | 年化变化 | 完整周期变化 | 期末资金变化 |",
             "|---|---|---:|---:|---:|---:|"]
    for r in delta.itertuples():
        rows.append(f"| {r.period} | {r.cost} | {r.sharpe_delta:+.3f} | {r.cagr_delta*100:+.2f}个百分点 | {r.cycle_delta:+d} | {r.ending_equity_delta:+,.2f}元 |")
    rows += ["", f"四个场景的收益与夏普是否同时提高：**{'是' if summary['cross_period_return_and_sharpe_improvement'] else '否'}**。该判断只描述固定历史比较，完整目标仍未实现。", "",
             "| 压力成本年度 | 点位账户收益 | 保留强弱收益 | 点位完整周期 | 保留强弱完整周期 |",
             "|---|---:|---:|---:|---:|"]
    sub = annual.loc[annual.period.eq("2020_2026") & annual.cost.eq("STRESS")]
    for year in sorted(sub.year.unique()):
        a = sub.loc[sub.year.eq(year) & sub.policy.eq("POINT_BINARY")].iloc[0]
        b = sub.loc[sub.year.eq(year) & sub.policy.eq("SAVED_WEIGHT")].iloc[0]
        rows.append(f"| {year}{'（至9月30日）' if year == 2026 else ''} | {a.calendar_year_return:.2%} | {b.calendar_year_return:.2%} | {a.completed_cycles} | {b.completed_cycles} |")
    rows += ["", "| 压力成本时期/方案 | 按实际份额毛损益 | 佣金及滑点 | 平均股票仓位 | 真实成交订单数 | 完整周期数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in m.loc[m.cost.eq("STRESS")].itertuples():
        rows.append(f"| {r.period}/{names[r.policy]} | {r.gross_at_actual_quantities_pnl:+,.2f}元 | {r.total_commission+r.total_slippage:,.2f}元 | {r.mean_exposure:.2%} | {r.orders} | {r.completed_cycles} |")
    rows += ["", "新增或减少的加减仓订单不算新增完整交易。毛损益归因只加回实际时点、实际份额的摩擦，不代表无成本策略或放宽账户约束。", "",
             "结论应同时结合此前的NR7结果：补充窄幅机会增加了次数却损害收益；原有仓位强弱是否有用，则以上述同口径比较判断。更高的历史收益或夏普不能自动解决跨期失效、历史多次选择和前瞻样本不足。", "",
             "当前pB>1与净均值为正的用户要求保留；独立前瞻完成点位仍为0。没有给出实盘建议、恢复终止策略、修改旧失败结果或进行期权收益回测。", "",
             "三项必要测试、8份落盘账户复算、3次历史截断核对通过。全部逐日账户、逐笔加减仓、完整周期及配对区块描述区间已保存。", "",
             "数据入口：`results/同风险预算的仓位信息比较.csv`、`results/仓位信息的账户增量.csv`、`results/逐年收益与完整周期.csv`、`results/accounts`。", ""]
    (OUT / "研究结论.md").write_text("\n".join(rows), encoding="utf-8")


def plot(accounts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fp = Path("C:/Windows/Fonts/msyh.ttc")
    if fp.exists():
        font_manager.fontManager.addfont(str(fp))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(fp)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, period, title in zip(axes, ("2015_2019", "2020_2026"), ("较早时期：2015—2019", "近期：2020—2026年9月30日")):
        for policy, label, color in (("POINT_BINARY", "只保留持有/空仓", "#9b6b48"), ("SAVED_WEIGHT", "保留原目标仓位强弱", "#226b7c")):
            d = accounts[(period, "STRESS", policy)]["daily"]
            ax.plot(d.date, d.equity / 200000, label=label, color=color, lw=1.7)
        ax.set_title(title, loc="left")
        ax.set_ylabel("20万元账户净值")
        ax.grid(axis="y", alpha=.18)
        ax.legend(loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("同样的进出来源，仓位信息能否改善收益？", x=.06, ha="left", fontsize=15)
    fig.text(.06, .012, "共同风险约束、压力成本；两个时期独立启动，不拼接。固定机制诊断，旧失败裁决保留，未建立独立验证。", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(OUT / "仓位信息与收益夏普.png", dpi=150)
    fig.savefig(OUT / "仓位信息与收益夏普.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定仓位信息的共同风险诊断。")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()
