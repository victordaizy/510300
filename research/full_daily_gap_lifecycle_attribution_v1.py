"""只读R177保存账本，解释全部缺口的生命周期、真实资金及退出时钟。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.full_daily_gap_study_v1 import OUT as FINANCIAL, EXPLANATION, PRIMARY, PERIODS, COSTS
from research.point_first_passage_study_v1 import read, write_json, digest, now, require, WEIGHT
from research.adaptive_allocation_v1 import normalize_dividends

OUT = ROOT / "reports/research/510300_full_daily_gap_lifecycle_attribution_v1"
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
CASE_IDS = (18, 37, 42, 55)
LEDGERS = ("daily", "orders", "trades", "decisions", "rejections")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def freeze():
    require(not (OUT / "protocol.json").exists(), "固定缺口生命周期归因已登记。")
    prior = read(FINANCIAL / "summary.json")
    require(prior["technical_decision"] == "TECH.R177" and prior["new_primary_accounts"] == 4,
            "四保存账户尚未完成。")
    require(read(FINANCIAL / "delivery_receipt.json")["all_actual_cycle_rows"] == 130,
            "全体真实周期尚未交付。")
    require(read(STATE)["latest_technical_decision"] == "TECH.R177", "当前阶段不再是R177。")
    for parent in (FINANCIAL, EXPLANATION):
        for source in read(parent / "protocol.json")["sources"]:
            require(digest(ROOT / source["path"]) == source["sha256"], "原冻结来源改变。")
    paths = [Path(__file__), ROOT / "research/full_daily_gap_account_v1.py",
             ROOT / "research/full_daily_gap_inputs_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             Path(sys.modules[normalize_dividends.__module__].__file__).absolute(), WEIGHT / "inputs/dividends.csv",
             FINANCIAL / "protocol.json", FINANCIAL / "summary.json", FINANCIAL / "delivery_receipt.json",
             FINANCIAL / "saved_result_verification.json", FINANCIAL / "next_all_gap_lifecycle_diagnostic_proposal.json",
             FINANCIAL / "results/完整账户共同口径比较.parquet",
             FINANCIAL / "results/全部完整缺口资格_原点量价A覆盖及真实执行状态.parquet",
             FINANCIAL / "results/全部实际进出点位_已知缺口量价及真实资金.parquet",
             EXPLANATION / "protocol.json", EXPLANATION / "summary.json",
             EXPLANATION / "results/全部3488已知缺口与量价_前2015A未知保留.parquet",
             EXPLANATION / "results/全部向上缺口至只上移回补_仅路径.parquet",
             EXPLANATION / "results/原61分段及49正式波段_完整日线缺口覆盖.parquet",
             EXPLANATION / "results/四案例逐日完整量价及整日缺口.parquet"]
    for period in PERIODS:
        for cost in COSTS:
            folder = FINANCIAL / f"results/accounts/{period}/{cost}/{PRIMARY}"
            paths.extend(folder / f"{name}.parquet" for name in LEDGERS)
            paths.append(folder / "terminal.json")
    paths = list(dict.fromkeys(paths))
    write_json(OUT / "protocol.json", {
        "at":now(),"study":"510300_FULL_DAILY_GAP_SAVED_LIFECYCLE_ATTRIBUTION_V1","technical_decision":"TECH.R178",
        "hypothesis":"具体上涨可能包含较早修复、反复回补和较晚追入；全部保存缺口及实际库存的出生/抬线/回补/次开时钟可解释R177为何次数增加却完整收益夏普不足。",
        "population":"97出生、194资格、四保存主账户和130真实周期（128完成/2开放），所有取消/已有持仓/失败/开放及原61/49段保留。",
        "known_clock":"按真实周期逐个已知原点还原初始下沿、新完整缺口只抬线和日低点回补，逐项匹配原decisions的下沿/库存/失效及原orders；不是新账户或替代退出。",
        "retrospective_alignment":"每个出生与所有原bottom_idx至peak_idx相交的分段逐条对齐，无相交保留未知；只报告距底/峰的日数和位置比例，不按后验阶段形成过滤，不选最有利分段。",
        "cycle_wealth":"真实BUY支出、逐次SELL净现金、登记日真实份额的已发生除息应收和当前真实剩余份额乘日收盘共同还原周期账面损益。持有期间最高收盘账面损益是事后诊断，不是可执行卖价/收益反事实或决策字段。",
        "exit_identity":"实际完成净利润减最终卖出前原点的周期账面损益=原点剩余份额乘(真实卖出日原开盘−原点收盘)+其后该周期股息应收−最终卖出佣金−滑点。逐笔身份核对，不赋未成交收益。",
        "all_outcome_groups":"固定完整周期组：持有期间最高收盘账面损益>0或<=0，交叉最终实际净利润正/零/负，六组×两时期×两费用全报告；空组质量未知，开放另列。分组全部为事后解释，不能做交易门。",
        "charts":"原四案例全部相交STRESS周期，图窗包含原案例窗及这些周期自身入出/观察末日；显示真实持仓/下沿、日低点回补及实际买卖、当时量/日MACD/上一完整周，不修改原案例或经济样本。",
        "no_rescue":"R177及所有旧终态保持，不改缺口大小/回补/退出/量/MACD/RV过滤/费用/预算/时期、不混A或拼接R173；新金融候选0。",
        "necessary_new_tests":0,"new_accounts":0,"new_fits":0,"new_training_labels":0,"new_market_requests":0,
        "historical_sample_role":"DEVELOPMENT_CALIBRATION","source_first_vintage":"NOT_CERTIFIED",
        "independent_validation":"NOT_ESTABLISHED","global_DSR_PBO":"NOT_COMPUTED","overfitting_removed":False,"goal_achieved":False,
        "sources":[{"path":p.relative_to(ROOT).as_posix(),"sha256":digest(p)} for p in paths],
    }, exclusive=True)
    print("R178全体缺口生命周期与真实资金身份固定；0新账户，不营救原政策。", flush=True)


def episode_alignments(known, episodes):
    rows = []
    for event in known.loc[known.date.ge("2015-01-01") & known.gap_event].itertuples(index=False):
        matched = episodes.loc[episodes.bottom_idx.le(event.origin_index) & episodes.peak_idx.ge(event.origin_index)]
        base = {"birth_date":event.date,"birth_index":int(event.origin_index),"event_anchor":event.gap_anchor_id,
                "retrospective_only_not_a_decision_field":True}
        if not len(matched):
            rows.append({**base,"episode_id":-1,"episode_alignment_status":"NO_RETROSPECTIVE_ASCENT_AT_ORIGIN",
                         "sessions_since_bottom":np.nan,"sessions_until_peak":np.nan,"fraction_ascent_elapsed":np.nan,
                         "episode_admitted":None,"after_original_confirmation":None})
        for episode in matched.itertuples(index=False):
            length = int(episode.peak_idx-episode.bottom_idx)
            rows.append({**base,"episode_id":int(episode.episode_id),"episode_alignment_status":"ALL_MATCHING_ORIGINAL_ASCENTS",
                         "sessions_since_bottom":int(event.origin_index-episode.bottom_idx),
                         "sessions_until_peak":int(episode.peak_idx-event.origin_index),
                         "fraction_ascent_elapsed":(event.origin_index-episode.bottom_idx)/length if length else np.nan,
                         "episode_admitted":bool(episode.admitted),"after_original_confirmation":bool(event.date>=episode.confirm_up_date)})
    return pd.DataFrame(rows)


def dividend_entitlements(cycle_orders, dividends, last_date):
    rows = []
    for event in dividends.itertuples(index=False):
        if event.ex_date > last_date:
            continue
        record = cycle_orders.loc[cycle_orders.date.le(event.record_date)]
        quantity = int(record.loc[record.side.eq("BUY"),"quantity"].sum()-record.loc[record.side.eq("SELL"),"quantity"].sum())
        require(quantity >= 0, "登记日周期库存为负。")
        if quantity:
            rows.append({"record_date":event.record_date,"ex_date":event.ex_date,
                         "record_quantity":quantity,"dividend_cny":quantity*float(event.cash_dividend_per_share)})
    return rows


def classify(peak, pnl):
    return ("POSITIVE_HOLDING_CLOSE" if peak>0 else "NONPOSITIVE_HOLDING_CLOSE")+"_FINAL_"+("WIN" if pnl>0 else "LOSS" if pnl<0 else "ZERO")


def saved_cycle(cycle, account, quotes, dividends, period, cost):
    orders, decisions, daily = account["orders"], account["decisions"], account["daily"]
    local_orders = orders.loc[orders.cycle_id.eq(cycle.cycle_id)].sort_values("date")
    last_date = daily.date.max()
    observed_end = cycle.exit_date if cycle.status=="COMPLETE" else last_date
    selected = daily.loc[daily.date.between(cycle.entry_date,observed_end)]
    entitlements = dividend_entitlements(local_orders, dividends, last_date)
    require(abs(sum(row["dividend_cny"] for row in entitlements)-float(cycle.dividend_cny))<=1e-6,
            "原登记库存没有还原周期股息。")
    require(int(local_orders.loc[local_orders.side.eq("BUY"),"quantity"].sum())==int(cycle.entry_quantity),
            "周期出现加仓或入场数量不符。")
    floor = int(cycle.entry_gap_lower_ticks)
    point_rows, cash_rows, changes, first_failure = [], [], [], None
    for day in selected.itertuples(index=False):
        until = local_orders.loc[local_orders.date.le(day.date)]
        bought, sold = until.loc[until.side.eq("BUY")], until.loc[until.side.eq("SELL")]
        quantity = int(bought.quantity.sum()-sold.quantity.sum())
        cash_delta = float((sold.quantity*sold.fill_price-sold.commission).sum()-(bought.quantity*bought.fill_price+bought.commission).sum())
        accrued = sum(row["dividend_cny"] for row in entitlements if row["ex_date"]<=day.date)
        require(quantity==int(day.shares), "真实周期库存没有还原保存日账。")
        marked = cash_delta+accrued+quantity*float(day.close)
        cash_rows.append({"period":period,"cost":cost,"cycle_id":int(cycle.cycle_id),"date":day.date,
                          "actual_shares":quantity,"actual_close":float(day.close),"cash_flow_cny":cash_delta,
                          "accrued_cycle_dividend_cny":accrued,"cycle_marked_pnl_cny":marked,
                          "holding_at_close":quantity>0,"retrospective_path_only_not_an_executable_exit":True})
        if quantity==0:
            continue
        event = quotes.loc[day.date]
        original = decisions.loc[decisions.origin.eq(day.date)]
        require(len(original)==1, "持有原点没有唯一保存决定。")
        decision = original.iloc[0]
        require(int(decision.shares_before)==quantity,"原点份额与日收盘库存不同。")
        previous = floor
        if decision.reason not in ("ACCOUNT_DRAWDOWN_STOP","LOCKED_EXIT") and bool(event.current_quote_known) and bool(event.gap_event):
            floor = max(floor,int(event.gap_lower_ticks))
        require(floor==int(decision.known_current_gap_floor_ticks),"原完整缺口抬线与保存决定不同。")
        raised = floor>previous
        if raised:
            changes.append(day.date)
        touched = bool(event.current_quote_known and int(event.known_cash_low_ticks)<=floor)
        if decision.reason=="FULL_DAILY_UP_GAP_FILLED_DAILY_LOW":
            require(touched and int(decision.desired_shares)==0,"回补退出缺少已知日低点。")
            first_failure = day.date if first_failure is None else first_failure
        point_rows.append({"period":period,"cost":cost,"cycle_id":int(cycle.cycle_id),"origin":day.date,
                           "entry_origin":cycle.entry_origin,"shares_before":quantity,"initial_floor_ticks":int(cycle.entry_gap_lower_ticks),
                           "floor_before_ticks":previous,"floor_after_ticks":floor,"floor_raised":raised,
                           "new_full_gap_while_holding":bool(event.gap_event),"quote_known":bool(event.current_quote_known),
                           "known_low_ticks":int(event.known_cash_low_ticks),"known_close_ticks":int(event.known_cash_close_ticks),
                           "low_touched_active_floor":touched,"low_touched_initial_floor":bool(event.current_quote_known and int(event.known_cash_low_ticks)<=int(cycle.entry_gap_lower_ticks)),
                           "close_above_active_floor":bool(event.current_quote_known and int(event.known_cash_close_ticks)>floor),
                           "decision_reason":decision.reason,"planned_next_open":decision.execution_date,
                           "desired_shares":int(decision.desired_shares),"actual_cycle_marked_pnl_cny":marked})
    require(floor==int(cycle.trailing_gap_floor_ticks),"最终下沿没有还原周期记录。")
    marks = pd.DataFrame(cash_rows)
    held = marks.loc[marks.holding_at_close]
    require(len(held)>0,"真实周期没有持有收盘。")
    peak = held.loc[held.cycle_marked_pnl_cny.idxmax()]
    row = {"period":period,"cost":cost,"cycle_id":int(cycle.cycle_id),"entry_origin":cycle.entry_origin,
           "entry_date":cycle.entry_date,"exit_date":cycle.exit_date,"status":cycle.status,
           "cycle_observed_until":observed_end,"entry_raw":float(cycle.entry_raw),"entry_fill":float(cycle.entry_price),
           "entry_quantity":int(cycle.entry_quantity),"actual_buy_debit":float(cycle.buy_debit),
           "initial_floor_ticks":int(cycle.entry_gap_lower_ticks),"final_floor_ticks":floor,"floor_raise_count":len(changes),
           "holding_close_rows":len(held),"first_known_low_fill_origin":first_failure,
           "best_observed_holding_close_date":peak.date,"best_observed_holding_close_pnl_cny":float(peak.cycle_marked_pnl_cny),
           "best_observed_holding_close_shares":int(peak.actual_shares),"actual_net_pnl":cycle.net_pnl,
           "actual_net_return":cycle.net_return,"final_result_group":"RIGHT_CENSORED_NOT_A_COMPLETED_GROUP",
           "final_exit_origin":pd.NaT,"exit_origin_marked_pnl_cny":np.nan,"final_open_gap_gross_cny":np.nan,
           "dividend_accrual_after_final_origin_cny":np.nan,"final_sell_commission":np.nan,"final_sell_slippage":np.nan,
           "exit_clock_identity_error_cny":np.nan,"decline_from_best_holding_close_to_actual_net_pnl_cny":np.nan,
           "loss_with_previously_positive_holding_close":None,"fill_on_raised_floor_only":None,
           "best_close_is_retrospective_not_a_decision_field":True}
    if cycle.status=="COMPLETE":
        require(pd.notna(first_failure),"完成周期没有原回补失效时钟。")
        final = local_orders.loc[local_orders.side.eq("SELL")].iloc[-1]
        before = marks.loc[marks.date.eq(final.origin)]
        require(len(before)==1,"最终退出原点缺少实际库存财富。")
        before = before.iloc[0]
        require(int(before.actual_shares)==int(final.quantity),"最终卖出数量不等于原点库存。")
        opening_delta = int(final.quantity)*(float(final.raw_open)-float(before.actual_close))
        later_dividend = float(cycle.dividend_cny)-float(before.accrued_cycle_dividend_cny)
        residual = float(cycle.net_pnl)-float(before.cycle_marked_pnl_cny)-opening_delta-later_dividend+float(final.commission)+float(final.slippage)
        require(abs(residual)<=1e-6,"最终退出时钟没有还原真实净利润。")
        final_point = next(p for p in reversed(point_rows) if p["origin"]==final.origin)
        row.update(final_result_group=classify(float(peak.cycle_marked_pnl_cny),float(cycle.net_pnl)),
                   final_exit_origin=final.origin,exit_origin_marked_pnl_cny=float(before.cycle_marked_pnl_cny),
                   final_open_gap_gross_cny=opening_delta,dividend_accrual_after_final_origin_cny=later_dividend,
                   final_sell_commission=float(final.commission),final_sell_slippage=float(final.slippage),
                   exit_clock_identity_error_cny=residual,
                   decline_from_best_holding_close_to_actual_net_pnl_cny=float(peak.cycle_marked_pnl_cny)-float(cycle.net_pnl),
                   loss_with_previously_positive_holding_close=bool(peak.cycle_marked_pnl_cny>0 and cycle.net_pnl<0),
                   fill_on_raised_floor_only=bool(not final_point["low_touched_initial_floor"] and final_point["low_touched_active_floor"]))
    return row, point_rows, cash_rows


def groups(cycles):
    rows = []
    for period in PERIODS:
        for cost in COSTS:
            for phase in ("POSITIVE_HOLDING_CLOSE","NONPOSITIVE_HOLDING_CLOSE"):
                for outcome in ("WIN","ZERO","LOSS"):
                    group = phase+"_FINAL_"+outcome
                    selected = cycles.loc[cycles.period.eq(period) & cycles.cost.eq(cost) & cycles.final_result_group.eq(group)]
                    rows.append({"period":period,"cost":cost,"retrospective_group":group,"completed_cycles":len(selected),
                                 "actual_net_pnl_cny":float(selected.actual_net_pnl.sum()),
                                 "mean_actual_net_return":float(selected.actual_net_return.mean()),
                                 "mean_floor_raise_count":float(selected.floor_raise_count.mean()),
                                 "final_open_gap_gross_cny":float(selected.final_open_gap_gross_cny.sum()),
                                 "dividend_accrual_after_final_origin_cny":float(selected.dividend_accrual_after_final_origin_cny.sum()),
                                 "final_sell_friction_cny":float((selected.final_sell_commission+selected.final_sell_slippage).sum()),
                                 "not_an_entry_filter_or_a_new_strategy":True})
    return pd.DataFrame(rows)


def charts(known, cases, cycles, holding, wealth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = next((p for p in ("C:/Windows/Fonts/msyh.ttc","C:/Windows/Fonts/simhei.ttf") if Path(p).exists()),None)
    if font:
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    paths, case_rows = [], []
    for case_id in CASE_IDS:
        case = cases.loc[cases.original_episode_id.eq(case_id)]
        lo, hi = case.date.min(),case.date.max()
        stress = cycles.loc[cycles.cost.eq("STRESS") & cycles.entry_date.le(hi) & cycles.cycle_observed_until.ge(lo)]
        lower = min(lo,stress.entry_date.min()) if len(stress) else lo
        upper = max(hi,stress.cycle_observed_until.max()) if len(stress) else hi
        frame = known.loc[known.date.between(lower,upper)]
        fig, axes = plt.subplots(4,1,figsize=(15,11),sharex=True,gridspec_kw={"height_ratios":[3,2,1,1]})
        price, pnl, macd, volume = axes
        price.plot(frame.date,frame.close,color="#29475c",label="原报价收盘")
        price.fill_between(frame.date,frame.low,frame.high,color="#b0c1ce",alpha=.3,label="整日高低区间")
        price.axvspan(lo,hi,color="#cfd5d9",alpha=.15,label="原固定案例窗")
        for cycle in stress.itertuples(index=False):
            h=holding.loc[holding.period.eq(cycle.period)&holding.cost.eq("STRESS")&holding.cycle_id.eq(cycle.cycle_id)]
            w=wealth.loc[wealth.period.eq(cycle.period)&wealth.cost.eq("STRESS")&wealth.cycle_id.eq(cycle.cycle_id)]
            shift=known.set_index("date").cash_shift.reindex(h.origin).to_numpy(float)
            price.step(h.origin,h.floor_after_ticks*.001-shift,where="post",color="#be6b38",label="当时真实持仓缺口下沿")
            price.scatter([cycle.entry_date],[cycle.entry_raw],marker="^",color="#1f8b78",s=55)
            if cycle.status=="COMPLETE":
                exit_quote=known.loc[known.date.eq(cycle.exit_date)].iloc[0]
                price.scatter([cycle.exit_date],[exit_quote.open],marker="v",color="#af3555",s=55)
                failure=known.loc[known.date.eq(cycle.first_known_low_fill_origin)].iloc[0]
                price.scatter([failure.date],[failure.low],marker="x",color="#8e3f6d",s=55)
            pnl.plot(w.date,w.cycle_marked_pnl_cny,label=f"{cycle.entry_origin:%m-%d}实际周期")
            case_rows.append({"original_episode_id":case_id,**cycle._asdict(),"original_case_window_start":lo,
                              "original_case_window_end":hi,"figure_observed_start":lower,"figure_observed_end":upper,
                              "retrospective_intersection_only":True})
        for ax in (price,pnl):
            handles,labels=ax.get_legend_handles_labels()
            unique=dict(zip(labels,handles))
            if unique:
                ax.legend(unique.values(),unique.keys(),fontsize=8,loc="best")
        pnl.axhline(0,color="#666",linewidth=.7)
        macd.plot(frame.date,frame.daily_dif,label="日DIF",color="#bf8629")
        macd.bar(frame.date,frame.daily_hist,color=np.where(frame.daily_hist.ge(0),"#ae5471","#488f7f"),alpha=.35,label="日MACD柱")
        macd.plot(frame.date,frame.weekly_hist,label="上一完整周柱",color="#29475c")
        macd.axhline(0,color="#777",linewidth=.6)
        macd.legend(fontsize=8)
        volume.bar(frame.date,frame.relative_volume,color="#7894a5",alpha=.65,label="相对前20日成交量中位数")
        volume.plot(frame.date,frame.rv_ratio,color="#bd8129",label="RV20/前252日中位数")
        volume.legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.16)
        price.set_ylabel("价格/元")
        pnl.set_ylabel("实际周期账面损益/元")
        macd.set_ylabel("动量原值")
        volume.set_ylabel("量与波动率比值")
        fig.suptitle(f"原案例{case_id}：完整缺口、真实下沿及退出时钟\n三角为真实开盘买卖，叉为已知回补；最高账面值仅事后观察",fontsize=15)
        fig.autofmt_xdate()
        fig.tight_layout(rect=(0,0,1,.94))
        path=OUT/f"案例{case_id}_真实缺口生命周期与资金.png"
        fig.savefig(path,dpi=140)
        plt.close(fig)
        paths.append(path.relative_to(ROOT).as_posix())
    table("四原案例全部相交压力周期_真实时钟与资金",pd.DataFrame(case_rows))
    return paths


def run():
    require(not (OUT/"RUN_STARTED.json").exists(),"固定生命周期诊断已开始，不重启。")
    protocol=read(OUT/"protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT/source["path"])==source["sha256"],"诊断登记来源改变。")
    write_json(OUT/"RUN_STARTED.json",{"at":now(),"new_accounts":0},exclusive=True)
    known=pd.read_parquet(EXPLANATION/"results/全部3488已知缺口与量价_前2015A未知保留.parquet")
    quotes=known.set_index("date")
    require(quotes.index.is_unique,"保存行情日期不唯一。")
    episodes=pd.read_parquet(EXPLANATION/"results/原61分段及49正式波段_完整日线缺口覆盖.parquet")
    cases=pd.read_parquet(EXPLANATION/"results/四案例逐日完整量价及整日缺口.parquet")
    dividends=normalize_dividends(pd.read_csv(WEIGHT/"inputs/dividends.csv"))
    alignments=episode_alignments(known,episodes)
    require(alignments.birth_index.nunique()==97,"全部出生没有保留。")
    cycle_rows,holding_rows,wealth_rows,scenario_rows=[],[],[],[]
    for period in PERIODS:
        for cost in COSTS:
            folder=FINANCIAL/f"results/accounts/{period}/{cost}/{PRIMARY}"
            account={name:pd.read_parquet(folder/f"{name}.parquet") for name in LEDGERS}
            for cycle in account["trades"].itertuples(index=False):
                row,holding,wealth=saved_cycle(cycle,account,quotes,dividends,period,cost)
                cycle_rows.append(row)
                holding_rows.extend(holding)
                wealth_rows.extend(wealth)
            scenario_rows.append({"period":period,"cost":cost,"cycles":len(account["trades"]),
                                  "completed":int(account["trades"].status.eq("COMPLETE").sum()),
                                  "open":int(account["trades"].status.ne("COMPLETE").sum()),
                                  "original_orders":len(account["orders"]),"original_decisions":len(account["decisions"])})
    cycles,holding,wealth=pd.DataFrame(cycle_rows),pd.DataFrame(holding_rows),pd.DataFrame(wealth_rows)
    require(len(cycles)==130 and int(cycles.status.eq("COMPLETE").sum())==128,"全体周期数量不同。")
    require(cycles.loc[cycles.status.ne("COMPLETE"),"actual_net_return"].isna().all(),"开放被填入完成收益。")
    table("全部97出生_原上涨段位置仅事后对齐",alignments)
    table("全部130周期_已知抬线回补及真实资金时钟",cycles)
    table("全部持有原点_下沿更新回补与原决定精确核对",holding)
    table("全部周期逐日真实库存现金应收及收盘财富",wealth)
    table("全部六组两时期两费用_最高账面与最终结果仅解释",groups(cycles))
    figures=charts(known,cases,cycles,holding,wealth)
    complete=cycles.loc[cycles.status.eq("COMPLETE")]
    pressure=complete.loc[complete.cost.eq("STRESS")]
    scenario_summary=[]
    for scenario in scenario_rows:
        selected=complete.loc[complete.period.eq(scenario["period"])&complete.cost.eq(scenario["cost"])]
        scenario_summary.append({**scenario,"completed_positive_close_then_loss":int(selected.loss_with_previously_positive_holding_close.astype(bool).sum()),
                                 "exit_on_raised_floor_only":int(selected.fill_on_raised_floor_only.astype(bool).sum()),
                                 "cycles_with_floor_raise":int(selected.floor_raise_count.gt(0).sum()),
                                 "final_open_gap_gross_cny":float(selected.final_open_gap_gross_cny.sum()),
                                 "post_final_origin_dividend_cny":float(selected.dividend_accrual_after_final_origin_cny.sum()),
                                 "final_exit_friction_cny":float((selected.final_sell_commission+selected.final_sell_slippage).sum()),
                                 "completed_pnl_cny":float(selected.actual_net_pnl.sum()),
                                 "exit_identity_max_abs_error":float(selected.exit_clock_identity_error_cny.abs().max())})
    for source in protocol["sources"]:
        require(digest(ROOT/source["path"])==source["sha256"],"运行期间来源改变。")
    write_json(OUT/"summary.json",{
        "at":now(),"technical_decision":"TECH.R178","status":"COMPLETED_ALL_SAVED_GAP_LIFECYCLE_AND_EXIT_WEALTH_IDENTITIES",
        "all_gap_births":97,"episode_alignment_rows":len(alignments),"unmatched_births_preserved":int(alignments.episode_id.eq(-1).sum()),
        "all_actual_cycle_rows":len(cycles),"completed_cycles":len(complete),"open_cycles":int(cycles.status.ne("COMPLETE").sum()),
        "holding_origins_verified":len(holding),"cycle_close_wealth_rows":len(wealth),"complete_exit_identities_verified":len(complete),
        "retrospective_group_cells":24,"all_scenario_diagnostics":scenario_summary,"charts":figures,
        "case_cycle_rows":sum(1 for case_id in CASE_IDS for cycle in cycles.loc[cycles.cost.eq("STRESS")].itertuples(index=False)
                              if cycle.entry_date<=cases.loc[cases.original_episode_id.eq(case_id)].date.max()
                              and cycle.cycle_observed_until>=cases.loc[cases.original_episode_id.eq(case_id)].date.min()),
        "pressure_complete_cycles":len(pressure),"pressure_positive_holding_close_then_loss":int(pressure.loss_with_previously_positive_holding_close.astype(bool).sum()),
        "new_accounts":0,"new_fits":0,"new_training_labels":0,"new_market_requests":0,"necessary_new_tests":0,
        "frozen_sources":len(protocol["sources"]),"latest_financial_strategy_decision_preserved":"TECH.R177",
        "independent_validation":"NOT_ESTABLISHED","overfitting_removed":False,"global_DSR_PBO":"NOT_COMPUTED","goal_achieved":False,
    },exclusive=True)
    print(f"R178全97出生、130周期、{len(holding)}持有决定和128退出资金身份完成；0新账户。",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="全部保存缺口生命周期与真实资金时钟。")
    parser.add_argument("command",choices=("freeze","run"))
    args=parser.parse_args()
    {"freeze":freeze,"run":run}[args.command]()
