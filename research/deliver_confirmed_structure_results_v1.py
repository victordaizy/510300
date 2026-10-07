"""从一次运行的保存账本提取全部实际点位，并交付具体上涨解释。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.confirmed_structure_study_v1 import OUT, EXPLANATION, PRIMARY, PERIODS, COSTS, table
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def number(value, digits=4):
    return "未知" if pd.isna(value) else f"{float(value):.{digits}f}"


def percent(value, digits=2):
    return "未知" if pd.isna(value) else f"{float(value):.{digits}%}"


def case_windows(actual, cases):
    observed_end = actual.period.map({k: pd.Timestamp(v[1]) for k, v in PERIODS.items()})
    until = actual.exit_date.fillna(observed_end)
    frames = []
    for episode_id, case in cases.groupby("original_episode_id", sort=True):
        lo, hi = case.date.min(), case.date.max()
        x = actual.loc[actual.entry_date.le(hi) & until.ge(lo)].copy()
        x["original_episode_id"] = int(episode_id)
        x["retrospective_window_for_explanation_only"] = True
        x["cycle_observed_until"] = until.loc[x.index]
        frames.append(x)
    return pd.concat(frames, ignore_index=True)


def run():
    require(not (OUT / "delivery_receipt.json").exists(), "本次保存点位和报告已经交付，不重写。")
    financial, explanation = read(OUT / "summary.json"), read(EXPLANATION / "summary.json")
    require(read(OUT / "saved_result_verification.json")["status"] == "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES", "实际账户保存核对未完成。")
    known = pd.read_parquet(EXPLANATION / "results/2015起全部已知结构与量价A覆盖.parquet")
    events = known.loc[known.structure_onset].copy()
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    yearly = pd.read_parquet(OUT / "results/逐年净收益与实际周期次数.parquet")
    qualifications, actual, scenario_counts = [], [], []
    for period, (start, end) in PERIODS.items():
        population = events.loc[events.date.between(start, end)]
        for cost in COSTS:
            folder = OUT / f"results/accounts/{period}/{cost}/{PRIMARY}"
            trades = pd.read_parquet(folder / "trades.parquet")
            decisions = pd.read_parquet(folder / "decisions.parquet")
            orders = pd.read_parquet(folder / "orders.parquet")
            rejects = pd.read_parquet(folder / "rejections.parquet")
            require(trades.entry_origin.is_unique, "一个出生原点存在多个账户周期。")
            for event in population.itertuples(index=False):
                decision = decisions.loc[decisions.origin.eq(event.date)]
                require(len(decision) == 1, "当时出生缺少唯一账户决定。")
                trade = trades.loc[trades.entry_origin.eq(event.date)]
                row = event._asdict()
                row.update(period=period, cost=cost, desired_shares=int(decision.desired_shares.iloc[0]),
                           decision_reason=decision.reason.iloc[0], shares_before=int(decision.shares_before.iloc[0]))
                if len(trade):
                    t = trade.iloc[0]
                    row.update(execution_status=t.status, cycle_id=int(t.cycle_id), entry_date=t.entry_date,
                               exit_date=t.exit_date, actual_net_return=t.net_return, actual_net_pnl=t.net_pnl,
                               entry_quantity=int(t.entry_quantity), actual_entry_raw=float(t.entry_raw),
                               actual_entry_fill=float(t.entry_price), actual_buy_debit=float(t.buy_debit))
                else:
                    row.update(execution_status="NO_ACTUAL_CYCLE_NO_RETURN", cycle_id=-1, entry_date=pd.NaT,
                               exit_date=pd.NaT, actual_net_return=np.nan, actual_net_pnl=np.nan,
                               entry_quantity=0, actual_entry_raw=np.nan, actual_entry_fill=np.nan, actual_buy_debit=np.nan)
                for name in ("H1", "H2", "L1", "L2"):
                    require(getattr(event, name+"_confirmation_date") <= event.date, "出生使用了未来枢轴确认。")
                qualifications.append(row)
            for t in trades.itertuples(index=False):
                before = known.loc[known.date.eq(t.entry_origin)]
                require(len(before) == 1 and bool(before.structure_onset.iloc[0]), "实际进场没有已知结构出生。")
                row = t._asdict()
                row.update(period=period, cost=cost)
                for key, value in before.iloc[0].to_dict().items():
                    row["entry_known_"+key] = value
                sold = orders.loc[orders.cycle_id.eq(t.cycle_id) & orders.side.eq("SELL")]
                final = sold.iloc[-1] if t.status == "COMPLETE" and len(sold) else None
                row.update(final_exit_raw=final.raw_open if final is not None else np.nan,
                           final_exit_fill=final.fill_price if final is not None else np.nan)
                actual.append(row)
            scenario_counts.append({"period": period, "cost": cost, "births": len(population),
                                    "actual_cycles": len(trades), "completed": int(trades.status.eq("COMPLETE").sum()),
                                    "open": int(trades.status.ne("COMPLETE").sum()), "rejections": len(rejects)})
    qualifications, actual = pd.DataFrame(qualifications), pd.DataFrame(actual)
    require(len(qualifications) == 2*len(events), "两费用全体出生没有完整保留。")
    table("全部合格结构出生_当时指标与实际成交", qualifications)
    table("全部实际进出点位_已知高低结构与A覆盖", actual)
    cases = pd.read_parquet(EXPLANATION / "results/四案例逐日完整量价与确认结构.parquet")
    table("四案例相交的全部实际周期_不筛收益", case_windows(actual, cases))
    mapped = pd.read_parquet(EXPLANATION / "results/原61分段及49正式波段_完整结构覆盖.parquet")
    admitted = mapped.loc[mapped.admitted]
    proposal = {
        "at": now(), "status": "PROPOSED_DIFFERENT_STRUCTURE_BREAK_EXPLANATION_NOT_ADMITTED_FINANCIAL_POLICY",
        "after_decisions": ["TECH.R168", "TECH.R169", "TECH.R170"],
        "research_question": "在完整双高双低抬升尚未成立时，下降结构首次被破坏能否更早标识修复与急涨，同时区分2015失败反弹？",
        "source_clock_intent": "继续使用原严格2/2日收盘确认组件；先考察上一原点已确认两高、两低下降结构中的最后高点被当日收盘首次突破。不得回填中心日期、不缩短确认、不改枢轴窗口。",
        "necessary_old_review": "核旧图谱/海龟、日线供给测试、快速结构与简单量价反转的完整触发、失效和终态；相同完整用途已经终态则停止，不换阈值重跑。",
        "explanation_population": "原四例、61分段/49正式波段、全部突破及其失败；日周MACD、量方向和RV只记录当时原值，不能在看过收益后选择过滤。",
        "financial_policy": "NOT_DEFINED_OR_REGISTERED；完整退出、参考标签和准入须在未来一次连接结果前确定，不继承R170收益证据。",
        "old_failures": "R170固定双抬升政策关闭，R165及所有旧终态保持；不缩确认、不加量/MACD阈值、不重组A、不放宽原风险预算。",
        "new_admitted_unrun_candidates": 0, "new_accounts_now": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    }
    write_json(OUT / "next_structure_break_explanation_proposal.json", proposal, exclusive=True)
    lines = [
        "# 具体上涨的量价、确认结构与完整账户结果", "",
        "本次按用户要求，先解释具体上涨/失败段，再将事前固定的点位定义放进共同账户检验。结论：高低点逐次抬升能解释部分慢涨行情，但确认太晚会错过急涨；这套完整规则交易次数提高，净收益和净夏普未提高，固定政策拒绝并关闭。", "",
        "## 四段的当时信息", "",
        "事后低点、高点和原5%确认线只用于回顾对齐，均不是入场或退出依据。价格枢轴中心必须再等两个交易日才可确认，图上台阶从确认日显示。所有数值取当时日线及上个完整周。", "",
        "| 具体路径 | 当时量价和指标 | 能提前识别的结构及局限 |", "|---|---|---|",
        "| 2019-01-02至04-19，原上涨39.13% | 01-15日MACD柱已正、量方向+0.727；01-23日DIF+0.0109、上周柱+0.0209，但相对量仅0.863，量方向−0.031 | 01-23首次双抬升，03-11/03-20再次成立。慢涨可以逐步确认；不是每个上涨确认日都放量，量方向也不始终为正。 |",
        "| 2020-03-23至07-13，原上涨38.54% | 03-25相对量1.666但日/周柱仍负；04-01日柱转正而日DIF−0.0925、上周柱−0.0951；06-17三项动量正，相对量仅0.987 | 03-09旧结构先成立后失败；04-01、04-27、06-17先后重建。早期修复与后段加速不能混成一笔交易。 |",
        "| 2024-09-13至10-08，原上涨36.68% | 09-24相对量3.338、量方向+1、日柱+0.0266，但上一完整周柱−0.0540；至10-08相对量6.852、动量和波动显著扩张 | 主升前至主升高点均没有完整双抬升，10-21才成立，确认明显滞后。不能把10-08高点回填成此前已知高点。 |",
        "| 2015-06-29至06-30，一日反弹7.22% | 06-30相对量2.271、量方向+0.094，但日柱−0.2068、上周柱−0.0663，价格在EMA下且波动约基准的2.7倍 | 当时双抬升未成立，随后继续下跌。放量反弹并不自动代表持续上涨。此前已有结构的失效也保留。 |", "",
        "MACD柱改善描述短期动量的变化，DIF正负和完整周柱有不同到达时钟；相对量描述交易活跃程度，量方向描述最近上涨/下跌日的量差，RV描述价格波动。它们可以帮助解释过程，本轮结果不能证明它们是上涨原因或可单独预测收益。", "",
    ]
    for i, name in zip((18, 37, 42, 55), explanation["charts"]):
        lines.extend([f"![原案例{i}的量价、当时已确认结构与A覆盖](../510300_confirmed_structure_explanation_v1/{name})", ""])
    lines.extend([
        "## 反推的唯一完整规则", "",
        "最近四个已确认交替收盘高低点中，两高点抬升、两低点抬升，当前收盘仍高于最新已确认低点。只在上一原点已知未成立、当前已知成立时出生；空仓次真实开盘尝试。已知结构不再成立或收盘跌到最新已确认低点，次合法开退出。未知本身不退出，风险只减，无加仓、固定2R或固定20日到期。", "",
        "原20万元、252日年化、现金收益0、50%仓位上限、原ES/跳空/回撤预算、T+1、100份、.001刻度及股息资格保持。两时期、两费用全报告；不以计划盈亏比代替实际周期盈亏比。", "",
        "| 时期 | 费用 | 净年化 | 全日历净夏普 | 最大回撤 | 完成/开放 | 胜率 | 实际盈亏比B | p×B | 完整年均次数 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in metrics.loc[metrics.policy.eq(PRIMARY)].itertuples(index=False):
        lines.append(f"| {r.period} | {r.cost} | {percent(r.net_cagr,4)} | {number(r.net_sharpe)} | {percent(r.max_drawdown)} | {r.completed_cycles}/{r.unfinished_cycles} | {percent(r.win_rate)} | {number(r.payoff)} | {number(r.p_times_b)} | {number(r.average_full_year_cycles,2)} |")
    lines.extend(["", "| 时期（STRESS） | 原A净年化/夏普 | 纯价格净年化/夏普 | 新结构相对原A |", "|---|---:|---:|---|"])
    for period in PERIODS:
        a = metrics.loc[metrics.period.eq(period) & metrics.cost.eq("STRESS") & metrics.policy.eq("A_SAVED_WEIGHT")].iloc[0]
        price = metrics.loc[metrics.period.eq(period) & metrics.cost.eq("STRESS") & metrics.policy.eq("PRICE_CONFIRMATION")].iloc[0]
        lines.append(f"| {period} | {percent(a.net_cagr)}/{number(a.net_sharpe)} | {percent(price.net_cagr)}/{number(price.net_sharpe)} | 净收益与夏普均较低 |")
    early = metrics.loc[metrics.period.eq("2015_2019") & metrics.cost.eq("STRESS") & metrics.policy.eq(PRIMARY)].iloc[0]
    recent = metrics.loc[metrics.period.eq("2020_2026") & metrics.cost.eq("STRESS") & metrics.policy.eq(PRIMARY)].iloc[0]
    lines.extend([
        "", f"较早期实际份额毛损益{early.gross_at_actual_quantities_pnl:.2f}元，佣金和滑点合计{early.total_commission+early.total_slippage:.2f}元，摩擦足以耗掉小幅毛盈利。近期毛损益{recent.gross_at_actual_quantities_pnl:.2f}元已经为负，不能仅归因费用。两时期账户均未触发停机。2019期末开放周期保留，不因全体后来发生失效而回填2019终点。", "",
        f"原111出生全部进入全体记录；两费用共{len(qualifications)}资格行/{len(actual)}实际周期行。压力实际111周期，其中110完成、1在较早期末开放。全部61原分段及49正式波段保留；正式波段确认后至高点有{int(admitted.structure_active_sessions.eq(0).sum())}段未出现结构成立、有{int(admitted.structure_births.eq(0).sum())}段没有新的结构出生，二者不能混作盈利机会。", "",
        "四场景的经济门全部失败，20/252日两尺度、对原A/纯价格两个比较的固定配对区块区间全部保存，历史稳定门也失败。没有只选择胜率更高的时期或个别赢家。", "",
        "## 全部实际点位和验证记录", "",
        "- [全部实际进出及进场前结构、量价和A覆盖](results/全部实际进出点位_已知高低结构与A覆盖.csv)",
        "- [全部结构出生、当时决定与实际成交](results/全部合格结构出生_当时指标与实际成交.csv)",
        "- [四案例相交的所有真实周期](results/四案例相交的全部实际周期_不筛收益.csv)",
        "- [十二场景共同口径对比](results/完整账户共同口径比较.csv)",
        "- [逐年实际收益与周期次数](results/逐年净收益与实际周期次数.csv)",
        "- [冻结金融定义](protocol.json)、[实际结果及全部统计区间](summary.json)、[保存核对](saved_result_verification.json)", "",
        "四输入测试及三账户行为测试最终通过，四实际行情前缀过去状态精确一致，八原A/纯价格对照精确复现。初次日期精度表示失败及未知报价测试错误预期均保存：前者仅统一datetime64[ns]，后者补明确的后续已知跌破路径，输入/账户逻辑未因金融结果变化。32解释来源、71金融来源保持；四新账户只运行一次，0拟合/新训练标签/新行情采集。", "",
        "## 接受与拒绝，以及下一步", "",
        "接受确认时钟、具体路径和漏掉急涨的解释；拒绝这套固定完整政策作为收益夏普改进。不能从拒绝推成所有高低结构或所有技术分析都无效，也不能给本政策加指标或换确认窗口救回。原A、全部旧失败及真实前瞻登记保持。", "",
        "下一具体工作：核旧完整定义后，先解释下降结构最后一个已确认高点首次被收盘突破的时钟；看能否更早区分2019/2020修复、2024急涨和2015失败。仍保留四例、原61/49段及全部失败，量价指标只记录原值。完整金融入出政策尚未定义或登记，已准入待跑候选为0；不假定新方向可以盈利。", "",
        "全部历史仍是DEVELOPMENT_CALIBRATION，first-vintage未认证，独立验证未成立，全局DSR/PBO未计算，去过拟合与提高完整收益夏普的目标均未达到。目标继续active；不把此次历史运行称独立验证或实盘授权。", "",
        "[下一解释提案](next_structure_break_explanation_proposal.json)", "",
    ])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_SAVED_ACTUAL_POINTS_AND_FOUR_CASE_EXPLANATION_DELIVERED",
        "technical_decision": "TECH.R170", "scenario_counts": scenario_counts,
        "all_qualification_rows": len(qualifications), "all_actual_cycle_rows": len(actual),
        "original_episodes_preserved": len(mapped), "original_admitted_waves": len(admitted),
        "admitted_no_structure_active": int(admitted.structure_active_sessions.eq(0).sum()),
        "admitted_no_new_structure_birth": int(admitted.structure_births.eq(0).sum()),
        "new_accounts_in_delivery": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in
                    [Path(__file__), OUT / "summary.json", OUT / "saved_result_verification.json", EXPLANATION / "summary.json", report]],
        "goal_achieved": False,
        "open_case_windows_clipped_at_own_account_end": True,
    }, exclusive=True)
    print(f"全部{len(qualifications)}出生行/{len(actual)}实际周期行、四例解释和共同账户报告已保存，没有重跑。", flush=True)


def correct_case_window():
    correction = OUT / "delivery_correction_01"
    require((correction / "reason.json").exists(), "原派生表与原因必须先保留。")
    require(not (correction / "correction_receipt.json").exists(), "案例观察范围已经修正。")
    actual = pd.read_parquet(OUT / "results/全部实际进出点位_已知高低结构与A覆盖.parquet")
    cases = pd.read_parquet(EXPLANATION / "results/四案例逐日完整量价与确认结构.parquet")
    corrected = case_windows(actual, cases)
    table("四案例相交的全部实际周期_不筛收益", corrected)
    for r in corrected.itertuples(index=False):
        require(r.cycle_observed_until <= pd.Timestamp(PERIODS[r.period][1]), "案例仍越过自己账户观察末日。")
    report = OUT / "研究结果与下一步.md"
    with report.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n案例派生表修正：开放周期只观察至自己账户末日，2019末开放周期不能被当作2020或2024案例中的持仓。原错误派生表与原因保存在delivery_correction_01；222资格/周期、原账本和收益指标均保持，没有金融重跑。\n")
    receipt = read(OUT / "delivery_receipt.json")
    receipt["open_case_windows_clipped_at_own_account_end"] = True
    receipt["case_window_correction"] = "delivery_correction_01/correction_receipt.json"
    for source in receipt["sources"]:
        source["sha256"] = digest(ROOT / source["path"])
    write_json(OUT / "delivery_receipt.json", receipt)
    write_json(correction / "correction_receipt.json", {
        "at": now(), "status": "PASS_CASE_WINDOWS_CLIPPED_AT_EACH_ORIGINAL_ACCOUNT_END",
        "saved_actual_cycle_rows_unchanged": len(actual), "corrected_case_rows": len(corrected),
        "new_accounts": 0, "financial_source_changes": 0,
        "corrected_table_sha256": digest(OUT / "results/四案例相交的全部实际周期_不筛收益.parquet"),
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in
                    [Path(__file__), correction / "reason.json", OUT / "delivery_receipt.json", report]],
    }, exclusive=True)
    print("案例开放周期已截断到自身观察末日；原金融账本、222全体记录及指标未变。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="保存点位报告及有原失败证据的案例观察范围修正。")
    parser.add_argument("action", nargs="?", default="deliver", choices=("deliver", "correct-case-window"))
    args = parser.parse_args()
    run() if args.action == "deliver" else correct_case_window()
