"""只读一次保存账户，交付全部点位、四例解释及全体资金恒等式。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.downtrend_break_study_v1 import OUT, EXPLANATION, PRIMARY, PERIODS, COSTS, table
from research.point_first_passage_study_v1 import read, write_json, digest, now, require


def percent(value, digits=2):
    return "未知" if pd.isna(value) else f"{float(value):.{digits}%}"


def number(value, digits=4):
    return "未知" if pd.isna(value) else f"{float(value):.{digits}f}"


def case_windows(actual, cases):
    """开放周期只观察到它所属账户末日，不延伸进入后一个时期。"""
    observed_end = actual.period.map({k: pd.Timestamp(v[1]) for k, v in PERIODS.items()})
    until = actual.exit_date.fillna(observed_end)
    frames = []
    for episode_id, case in cases.groupby("original_episode_id", sort=True):
        lo, hi = case.date.min(), case.date.max()
        selected = actual.loc[actual.entry_date.le(hi) & until.ge(lo)].copy()
        selected["original_episode_id"] = int(episode_id)
        selected["retrospective_window_for_explanation_only"] = True
        selected["cycle_observed_until"] = until.loc[selected.index]
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    require(result.cycle_observed_until.le(result.period.map({k: pd.Timestamp(v[1]) for k, v in PERIODS.items()})).all(),
            "案例派生表超过了自身账户观察末日。")
    return result


def capital_identity(trades, period, cost):
    """拆解已有净利润，不生成等额策略或收益反事实。"""
    complete = trades.loc[trades.status.eq("COMPLETE")]
    q, r = complete.buy_debit.astype(float), complete.net_return.astype(float)
    require(len(complete) > 0 and q.gt(0).all() and r.notna().all(), "资金恒等式缺少完整实际周期。")
    require(np.allclose(q * r, complete.net_pnl, atol=1e-8, rtol=0), "实际周期金额与净收益率不一致。")
    mean_component = float(q.sum() * r.mean())
    allocation_component = float(((q - q.mean()) * (r - r.mean())).sum())
    actual_pnl = float(complete.net_pnl.sum())
    error = actual_pnl - mean_component - allocation_component
    require(abs(error) < 1e-6, "资金恒等式没有还原保存净利润。")
    largest = complete.loc[complete.net_pnl.idxmax()]
    return {
        "period": period, "cost": cost, "complete_cycles": len(complete),
        "open_cycles_preserved": int(trades.status.ne("COMPLETE").sum()),
        "mean_actual_buy_debit": float(q.mean()), "mean_cycle_net_return": float(r.mean()),
        "sum_actual_buy_debit": float(q.sum()), "mean_return_component_cny": mean_component,
        "allocation_covariance_component_cny": allocation_component, "actual_completed_pnl_cny": actual_pnl,
        "identity_error_cny": error, "positive_cycle_pnl_cny": float(complete.loc[complete.net_pnl.gt(0), "net_pnl"].sum()),
        "negative_cycle_pnl_cny": float(complete.loc[complete.net_pnl.lt(0), "net_pnl"].sum()),
        "largest_actual_winner_origin": largest.entry_origin,
        "largest_actual_winner_net_pnl": float(largest.net_pnl),
        "role": "POST_RESULT_ALL_POPULATION_SAVED_PNL_IDENTITY_NOT_A_NEW_POLICY",
    }


def run():
    require(not (OUT / "delivery_receipt.json").exists(), "本轮点位和报告已经交付，不重复写入。")
    financial = read(OUT / "summary.json")
    explanation = read(EXPLANATION / "summary.json")
    require(financial["technical_decision"] == "TECH.R173" and financial["new_primary_accounts"] == 4,
            "首次突破完整账户尚未完成。")
    require(read(OUT / "saved_result_verification.json")["status"] == "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES",
            "四保存账户的指标与资金核对尚未通过。")
    known = pd.read_parquet(EXPLANATION / "results/2015起全部已知突破与量价A覆盖.parquet")
    events = known.loc[known.break_event].copy()
    metrics = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    qualifications, actual, scenario_counts, identities = [], [], [], []
    for period, (start, end) in PERIODS.items():
        population = events.loc[events.date.between(start, end)]
        for cost in COSTS:
            folder = OUT / f"results/accounts/{period}/{cost}/{PRIMARY}"
            trades = pd.read_parquet(folder / "trades.parquet")
            decisions = pd.read_parquet(folder / "decisions.parquet")
            orders = pd.read_parquet(folder / "orders.parquet")
            rejects = pd.read_parquet(folder / "rejections.parquet")
            require(trades.entry_origin.is_unique, "同一个首次突破原点对应多个账户周期。")
            cancelled, occupied = 0, 0
            for event in population.itertuples(index=False):
                decision = decisions.loc[decisions.origin.eq(event.date)]
                require(len(decision) == 1, "首次突破缺少唯一当时决定。")
                d = decision.iloc[0]
                trade = trades.loc[trades.entry_origin.eq(event.date)]
                rejected = rejects.loc[rejects.origin.eq(event.date)]
                row = event._asdict()
                row.update(period=period, cost=cost, desired_shares=int(d.desired_shares),
                           decision_reason=d.reason, source_weight=d.source_weight, known_es95=d.known_es95,
                           shares_before=int(d.shares_before), planned_execution_date=d.execution_date,
                           rejection_reason=None)
                if len(trade):
                    t = trade.iloc[0]
                    row.update(execution_status=t.status, cycle_id=int(t.cycle_id), entry_date=t.entry_date,
                               exit_date=t.exit_date, actual_net_return=t.net_return, actual_net_pnl=t.net_pnl,
                               entry_quantity=int(t.entry_quantity), actual_entry_raw=float(t.entry_raw),
                               actual_entry_fill=float(t.entry_price), actual_buy_debit=float(t.buy_debit))
                else:
                    if len(rejected):
                        require(len(rejected) == 1 and rejected.reason.iloc[0] == "OPEN_AT_OR_BELOW_KNOWN_BROKEN_HIGH_CANCELLED",
                                "未成交原点出现未解释的拒绝原因。")
                        status = "CANCELLED_NEXT_OPEN_AT_OR_BELOW_INITIAL_STOP"
                        row["rejection_reason"] = rejected.reason.iloc[0]
                        cancelled += 1
                    else:
                        require(int(d.shares_before) > 0 and int(d.desired_shares) == int(d.shares_before),
                                "未成交原点既非开盘取消，也非已有持仓不新建周期。")
                        status = "ALREADY_HOLDING_NO_NEW_CYCLE"
                        occupied += 1
                    row.update(execution_status=status, cycle_id=-1, entry_date=pd.NaT, exit_date=pd.NaT,
                               actual_net_return=np.nan, actual_net_pnl=np.nan, entry_quantity=0,
                               actual_entry_raw=np.nan, actual_entry_fill=np.nan, actual_buy_debit=np.nan)
                for name in ("H1", "H2", "L1", "L2"):
                    require(getattr(event, "prior_" + name + "_confirmation_index") <= event.origin_index - 1,
                            "入场所引用高低点没有在上一原点确认。")
                qualifications.append(row)
            for t in trades.itertuples(index=False):
                before = known.loc[known.date.eq(t.entry_origin)]
                require(len(before) == 1 and bool(before.break_event.iloc[0]), "实际进场没有已知首次突破。")
                row = t._asdict()
                row.update(period=period, cost=cost)
                row.update({"entry_known_" + key: value for key, value in before.iloc[0].to_dict().items()})
                sold = orders.loc[orders.cycle_id.eq(t.cycle_id) & orders.side.eq("SELL")]
                final = sold.iloc[-1] if t.status == "COMPLETE" and len(sold) else None
                row.update(final_exit_raw=final.raw_open if final is not None else np.nan,
                           final_exit_fill=final.fill_price if final is not None else np.nan)
                actual.append(row)
            require(len(trades) + cancelled + occupied == len(population), "全体首次突破数量未还原。")
            scenario_counts.append({"period": period, "cost": cost, "first_break_events": len(population),
                                    "actual_cycles": len(trades), "completed": int(trades.status.eq("COMPLETE").sum()),
                                    "open": int(trades.status.ne("COMPLETE").sum()),
                                    "next_open_cancelled": cancelled, "already_holding_no_new_cycle": occupied})
            identities.append(capital_identity(trades, period, cost))
    qualifications, actual = pd.DataFrame(qualifications), pd.DataFrame(actual)
    require(len(qualifications) == len(events) * len(COSTS), "没有保留两费用全部合格原点。")
    table("全部首次突破资格_当时指标成交取消及A覆盖", qualifications)
    table("全部实际进出点位_已知下降结构量价及资金", actual)
    table("全部四场景完成周期_实际资金恒等式", pd.DataFrame(identities))
    cases = pd.read_parquet(EXPLANATION / "results/四案例逐日完整量价与上一确认下降结构.parquet")
    table("四案例相交的全部实际周期_不筛收益", case_windows(actual, cases))
    mapped = pd.read_parquet(EXPLANATION / "results/原61分段及49正式波段_完整首次突破覆盖.parquet")
    admitted = mapped.loc[mapped.admitted]
    primary = metrics.loc[metrics.policy.eq(PRIMARY)]
    point_quality_passed = bool(primary.p_times_b.gt(1).all() & primary.standard_expectancy_loss_units.gt(0).all())
    require(point_quality_passed and primary.mean_cycle_net_return.gt(0).all(), "交付与实际四场景交易质量不一致。")
    economic_passes = sum(bool(x["economic_gate_passed"]) for x in financial["comparisons"])
    proposal = {
        "at": now(), "status": "PROPOSED_ALL_SIGNAL_ORIGINAL_BUDGET_ATTRIBUTION_NO_NEW_POLICY_ADMITTED",
        "after_decisions": ["TECH.R171", "TECH.R172", "TECH.R173"],
        "research_question": "为何四场景已完成交易质量均为正，近期完整账户仍显著低于A？全部66原点的已知预算、真实资金覆盖、持有风险减仓及摩擦各贡献多少？",
        "fixed_population": "66合格原点×两费用全部保留；53实际周期/11次开取消/2已持仓分别解释，104完成/2开放不混合，四场景与全部失败同时报告。",
        "method": "只复算原计划份额、开盘可买上限、原ES/跳空/DD限制及实际减仓；连接全部保存订单与决定。原A库存和当时目标分别列出，不能用A库存0代替A没有信号。",
        "known_diagnostic": "四场景实际资金恒等式已经归档；近期STRESS均值项6647.31元加资金收益对应项−5181.32元=实际1465.99元。这是账本代数，不是等額或放大仓位策略。",
        "scope_limit": "R166/R167预算诊断和原波动/配仓/收益过滤终态保持；本次只针对R173新事实解释，不能当作重新准入旧配仓机制。",
        "no_rescue": "R173固定完整政策关闭；不改确认窗口、退出、费用、时期、风险预算，不加MACD/量/RV门，不混A，不挑赢家或赋未成交信号虚构收益。",
        "new_admitted_unrun_candidates": 0, "new_policy_accounts_planned": 0, "new_model_fits_planned": 0,
        "new_training_labels_planned": 0, "new_market_requests_planned": 0,
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    }
    write_json(OUT / "next_all_signal_budget_attribution_proposal.json", proposal, exclusive=True)
    lines = [
        "# 具体上涨解释、首次下降结构突破与完整账户结果", "",
        "本轮按用户要求，先解释具体上涨段及失败反弹的量价、日/周MACD和波动率，再用当时已确认的下降结构反推出完整入出政策。实际结果有一项进展：两个时期、两档成本下，完成交易的胜率×实际盈亏比均>1且平均净收益为正；完整账户只有较早期STRESS一个场景通过经济门，近期明显不及原A，历史稳定门也未通过。这套固定政策拒绝并关闭，点位质量保留为开发事实。", "",
        "## 四段具体路径与提前到达的信息", "",
        "当时日线和上一完整周才可使用；图上的高低点从确认日开始显示，中心日期不回填。原事后低点/高点只供解释对齐，不能当作可执行进出。价格结构用已发生现金分红平移坐标，实际成交使用原始开盘。", "",
        "| 原案例 | 当时量价和指标 | 可识别点位与真实压力成本结果 |", "|---|---|---|",
        "| 2019-01-02至04-19上涨39.13% | 01-18已突破上一确认下降高点；相对量1.391、量方向+0.218、日DIF+0.00088/柱+0.03881，上一完整周柱+0.00306，RV比0.804 | 01-18收盘形成信号，01-21开盘3.166入、03-11出，实际净+14.79%。03-18另一个突破同样合格，03-19入/03-26出却净−2.94%，完整保留。早于旧双抬升01-23的确认。 |",
        "| 2020-03-23至07-13上涨38.54% | 05-29突破上一确认下降高点，相对量0.833、量方向+0.690、日DIF+0.00378但日柱−0.02134/上一完整周柱−0.01229，RV比0.910 | 05-29信号，06-01开盘3.898入、07-27出，实际净+17.36%。信号并非日周MACD全正，也不是统一早于旧04-01/04-27双抬升。 |",
        "| 2024-09-13至10-08上涨36.68% | 09-24相对量3.338、量方向+1、日柱+0.02665但DIF−0.02602/上一完整周柱−0.05396，RV比1.574；突破08-26高点，该高点早在08-28确认 | 09-24信号，09-25开盘3.489入、10-31出，实际净+13.25%；旧双抬升到10-21才成立。A库存当时0，但A已知目标31.92%，所以不是原A完全未覆盖的信号。 |",
        "| 2015-06-29至06-30反弹7.22%后失败 | 06-30相对量2.271，但日周MACD柱负、波动约基准2.7倍；反弹仍未突破当时已确认下降高点 | 整个固定案例窗没有新的合格首次突破，不能把一日放量反弹或事后低点称为本策略入场。 |", "",
        "量价和MACD可以描述修复/加速的到达顺序，RV描述波动，不构成上涨因果证明。四例显示突破可以早于完整双抬升，但并非每段都更早，且突破自身也会失败。没有从赢家反选指标阈值。", "",
    ]
    for episode, chart in zip((18, 37, 42, 55), explanation["charts"]):
        lines.extend([f"![原案例{episode}的当时下降结构及量价](../510300_downtrend_break_explanation_v1/{chart})", ""])
    lines.extend([
        "## 反推出的唯一完整政策", "",
        "上一原点已确认的两高、两低都下降，上一收盘<=同一个最后高点H2，当前收盘>H2；每个高点锚只承认第一次已知跨越。空仓次真实开尝试，若开盘现金平移价<=旧H2即取消，不延后追入。初始失效线就是被突破旧H2；持有后用新已确认低点只向上移动失效线，收盘<=线则次合法开退出。未知不强制退出，原风险只减，无加仓、A混合、固定2R或20日到期。", "",
        "该完整用途在连接旧结果标签前固定。原严格日收盘2/2确认组件、20万元、252日年化、现金收益0、50%上限、ES/跳空/回撤预算、T+1、100份、.001价格刻度和原股息账户口径保持。BASE佣金/滑点为0.02%/0.05%，STRESS为0.04%/0.10%，最低佣金5元。", "",
        "| 时期 | 费用 | 净年化 | 全日历净夏普 | 最大回撤 | 完成/开放 | 胜率 | 实际盈亏比B | p×B | pB−q | 完整年均次数 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in primary.itertuples(index=False):
        lines.append(f"| {r.period} | {r.cost} | {percent(r.net_cagr,4)} | {number(r.net_sharpe)} | {percent(r.max_drawdown)} | {r.completed_cycles}/{r.unfinished_cycles} | {percent(r.win_rate)} | {number(r.payoff)} | {number(r.p_times_b)} | {number(r.standard_expectancy_loss_units)} | {number(r.average_full_year_cycles,2)} |")
    lines.extend(["", "B=实际盈利周期平均净收益率/实际亏损周期平均净损失率，p为完成周期胜率，q为实际亏损概率；以上均是交易完成后的点估计，不是预先目标止盈/止损比。较早仅5个赢家、近期6个赢家，优势的稳定性尚未成立。次数为软目标，2026不完整年度不纳入完整年均次数。", "",
                  "| 时期 | 费用 | 新规则净年化/夏普 | 原A净年化/夏普 | 纯价格净年化/夏普 | 四条件经济门 |", "|---|---|---:|---:|---:|---|"])
    for comparison in financial["comparisons"]:
        period, cost = comparison["period"], comparison["cost"]
        rows = metrics.loc[metrics.period.eq(period) & metrics.cost.eq(cost)].set_index("policy")
        p, a, price = (rows.loc[key] for key in (PRIMARY, "A_SAVED_WEIGHT", "PRICE_CONFIRMATION"))
        values = [f"{percent(r.net_cagr,4)}/{number(r.net_sharpe)}" for r in (p, a, price)]
        lines.append(f"| {period} | {cost} | {values[0]} | {values[1]} | {values[2]} | {'通过' if comparison['economic_gate_passed'] else '未通过'} |")
    lines.extend([
        "", "只有2015_2019/STRESS通过单场景经济门；早期BASE夏普改善但年化略低于A，近期两档费用均远低于A。因此不能说四个场景都各自失败，也不能把1/4场景通过说成完整准入。两比较、20/252日固定配对区块各2000次的全部区间保存，历史稳定门未通过；近期相对A的收益和夏普差值两尺度95%区间均为负。它们仍是开发描述，不是独立验证。", "",
        "## 全部信号为什么没有转化为更高账户收益", "",
        "66个合格原点：较早28、近期38。每档费用53个实际周期（52完成、1较早期末开放），11次开跌回初始失效线取消（早期6/近期5），另2个已有持仓不新建周期（2019-08-23、2023-04-17）。取消和已有持仓事件没有虚构成交收益；同锚重复跨越另有2次不产生新资格。两费用共132资格行、106真实周期行，104完成/2开放。", "",
        "实际已完成净利润可以恒等拆成：Σ(Q×r)=ΣQ×平均r + Σ[(Q−平均Q)(r−平均r)]，Q是每周期实际买入支出，r是该周期实际净收益率。右边两个量共同还原已有账本，不能把均值项称为可执行等额账户。", "",
        "| 时期 | 费用 | 完成周期均值项/元 | 资金与收益对应项/元 | 实际完成净利润/元 | 最大实际赢家原点及净利润 |", "|---|---|---:|---:|---:|---|",
    ])
    for r in identities:
        lines.append(f"| {r['period']} | {r['cost']} | {r['mean_return_component_cny']:.2f} | {r['allocation_covariance_component_cny']:.2f} | {r['actual_completed_pnl_cny']:.2f} | {r['largest_actual_winner_origin']:%Y-%m-%d} / {r['largest_actual_winner_net_pnl']:.2f}元 |")
    earlier = primary.loc[primary.period.eq("2015_2019") & primary.cost.eq("STRESS")].iloc[0]
    recent = primary.loc[primary.period.eq("2020_2026") & primary.cost.eq("STRESS")].iloc[0]
    lines.extend([
        "", f"近期压力成本，实际份额毛损益{recent.gross_at_actual_quantities_pnl:.2f}元，佣金{recent.total_commission:.2f}元/滑点{recent.total_slippage:.2f}元，完成净利润{recent.completed_cycle_net_pnl:.2f}元。账户平均投入约{percent(recent.mean_exposure)}，所有现金日纳入夏普。盈利来自6周期、亏损来自26周期，最大的2020-05-29赢家10502.89元超过最终全部净利润，多数失败必须一起计入。", "",
        f"较早压力实际完成净利润{earlier.completed_cycle_net_pnl:.2f}元，账户净增{earlier.ending_equity-200000:.2f}元；差额来自期末开放周期，不能把开放利润提前填入完成交易胜率。较早最大赢家2015-02-12贡献21495.81元，超过全体完成净利润，单例盈利不等于稳定优势。两个账户均未触发回撤停机。", "",
        f"原61分段/49正式波段完整保留，49段中{int(admitted.first_break_events.eq(0).sum())}段确认后至高点没有新首次突破。旧2846标签（2826成熟/20删失）与9尾部无新标签保持；66事件的旧20日描述pB约0.494不能替代本次动态失效的完成交易结果。", "",
        "## 验证、数据限制和处置", "",
        "四输入/四账户行为测试最终通过，四真实行情前缀一致，八原A/纯价格日账、订单和周期精确复现；四新主账户一次运行，保存指标与资金/库存/T+1核对通过。41解释来源、81金融来源冻结；初次两失败原源码、输出和原因全部保留：仅固定nullable字符串dtype及现金平移的浮点运算表示，未改变断言、触发/退出逻辑或根据金融结果调参，修复时新主账户结果尚未读取。", "",
        "原量价、复权所需已发生分红及共同账户来源只用本地冻结文件；MACD/RV为过去行情派生，周线只到上一完整周。回执/文件哈希说明本次来源，不能认证历史first-vintage或真实当时取得全部来源。全段历史已经反复使用，角色DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED，global DSR/PBO NOT_COMPUTED；去过拟合与完整收益夏普改进目标均未实现。", "",
        "接受：本确认时钟、四例的早晚区别、全部实际完成交易质量为正的开发事实。拒绝：R173固定完整政策作为收益/夏普改进，关闭并保留；R170双抬升及所有旧终态保持。不得据结果增加MACD/量/RV过滤、调整窗口/失效、混A或放宽风险预算营救。源/实现真实错误、实质新信息或真正新样本才可能另行独立重验，不把全项目停止等同本政策关闭。", "",
        "下一具体实验是全部66信号的原资金预算与实际订单归因：先解释已知ES/跳空/DD约束、真实投入与持有期间减仓，区分原A库存和当时目标。四场景、所有取消/已有持仓/失败及开放周期保留。暂不登记另一账户策略，不把旧R166/R167预算机制重新说成未试；已准入待跑新候选0。真实原前瞻两登记与十二状态值保持，最早资格新收盘仍2026-10-08 15:05，按其原1008实际交易日协议执行。", "",
        "## 可追溯文件", "",
        "- [全部真实进出、当时量价/下降结构/资金](results/全部实际进出点位_已知下降结构量价及资金.csv)",
        "- [全部合格点及成交、取消、已有持仓](results/全部首次突破资格_当时指标成交取消及A覆盖.csv)",
        "- [四原案例相交的全部真实周期](results/四案例相交的全部实际周期_不筛收益.csv)",
        "- [十二场景共同口径](results/完整账户共同口径比较.csv)、[逐年收益与完成次数](results/逐年净收益与实际周期次数.csv)",
        "- [四场景资金恒等式](results/全部四场景完成周期_实际资金恒等式.csv)",
        "- [R171解释](../510300_downtrend_break_explanation_v1/summary.json)、[R172冻结政策](protocol.json)、[R173结果与全部区间](summary.json)",
        "- [八必要测试](tests_receipt.json)、[八原控制复现](control_preflight.json)、[保存账本核对](saved_result_verification.json)",
        "- [下一全体资金归因提案](next_all_signal_budget_attribution_proposal.json)", "",
    ])
    report = OUT / "研究结果与下一步.md"
    with report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    write_json(OUT / "delivery_receipt.json", {
        "at": now(), "status": "PASS_ALL_BREAK_POINTS_CASES_AND_SAVED_CAPITAL_IDENTITIES_DELIVERED",
        "technical_decision": "TECH.R173", "scenario_counts": scenario_counts,
        "all_qualification_rows": len(qualifications), "all_actual_cycle_rows": len(actual),
        "all_four_actual_point_quality_passed": point_quality_passed,
        "individual_economic_gate_passes": economic_passes, "overall_economic_gate_passed": financial["all_four_economic_gates_passed"],
        "historical_stability_gate_passed": financial["historical_stability_gate_passed"],
        "capital_identity_cells": len(identities), "original_episodes_preserved": len(mapped),
        "original_admitted_waves": len(admitted), "admitted_no_new_first_break": int(admitted.first_break_events.eq(0).sum()),
        "open_case_windows_clipped_at_own_account_end": True,
        "new_accounts_in_delivery": 0, "new_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in
                    [Path(__file__), OUT / "summary.json", OUT / "saved_result_verification.json", EXPLANATION / "summary.json", report]],
        "goal_achieved": False,
    }, exclusive=True)
    print(f"已交付全部{len(qualifications)}资格行/{len(actual)}实际周期行、四案例及四资金恒等式；没有新账户重跑。", flush=True)


if __name__ == "__main__":
    run()
