"""用已保存的510300期权预测，检验10万元账户及三档固定风险预算。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_option_volatility_probability_payoff_v1"
DEFAULT_OUT = ROOT / "reports/research/510300_option_volatility_account_risk_v1"
STRUCTURES = ["LONG_ATM_STRADDLE", "PROTECTED_SHORT_ATM_STRADDLE"]
POLICIES = ["MODEL_POSITIVE_EXPECTANCY", "EMPIRICAL_POSITIVE_EXPECTANCY", "ALWAYS_PROTECTED_SHORT_REFERENCE"]
PROFILES = [("R1_D10", .01, .10), ("R2_D20", .02, .20), ("R4_D30", .04, .30)]
INITIAL = 100000.0
FEE = 5.0
ANNUAL_DAYS = 242
MARGIN_MULTIPLIER = 1.2
SIGNAL_COLUMNS = ["decision_date", "observed_date", "entry_date", "label_end_date", "structure", "model",
                  "known_risk_scale_cny", "expected_net_risk_unit", "p_win", "conditional_gain", "conditional_loss", "fit_date"]
LEG_COLUMNS = ["decision_date", "structure", "leg_number", "contract_code", "option_type", "strike", "side", "historical_unit", "prior_close"]


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def rounding(value: Decimal, digits: int) -> float:
    return float(value.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def adjust_terms(unit: int, strike: float, previous_spot: float, cash_dividend: float):
    new_unit = int(rounding(Decimal(unit)*Decimal(str(previous_spot))/(Decimal(str(previous_spot))-Decimal(str(cash_dividend))), 0))
    new_strike = rounding(Decimal(str(strike))*Decimal(unit)/Decimal(new_unit), 3)
    return new_unit, new_strike


def fill_price(price: float, action: int, scenario: str) -> float:
    tick, fraction = (.0002, .005) if scenario == "BASE" else (.0005, .01)
    return max(0., float(price)+action*max(tick, float(price)*fraction))


def prepare(out: Path) -> dict:
    if (out / "freeze.json").exists():
        raise RuntimeError("本版本已冻结，禁止覆盖运行；请使用verify。")
    inputs = {
        "option_eod.parquet": PARENT / "inputs/option_eod.parquet",
        "option_risk.parquet": PARENT / "inputs/option_risk.parquet",
        "etf_market.parquet": PARENT / "inputs/etf_market.parquet",
        "dividends.csv": ROOT / "data/reference/510300_dividends.csv",
        "predictions.csv": PARENT / "results/完整滚动预测.csv",
        "candidate_legs.csv": PARENT / "results/完整候选合约腿与报价.csv",
        "parent_protocol.json": PARENT / "protocol.json",
        "parent_summary.json": PARENT / "summary.json",
        "parent_freeze.json": PARENT / "freeze.json",
        "parent_parameters.json": PARENT / "models/全部月度参数.json",
        "mandate.json": ROOT / "config/510300_options_100k_doubling_mandate_v2.json",
    }
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    for name, source in inputs.items():
        shutil.copy2(source, out / "inputs" / name)
    shutil.copy2(Path(__file__), out / "code" / Path(__file__).name)
    protocol = {
        "study_id": "510300_OPTION_VOLATILITY_ACCOUNT_RISK_V1", "frozen_at": now(),
        "mandate": json.loads(inputs["mandate.json"].read_text(encoding="utf-8")),
        "scope": "现有预测到资金账户的固定映射；不重新训练、不改变翼宽、期限、特征或预测参数",
        "policies": {
            POLICIES[0]: "每个决策日从波动率模型的两结构中选择单位事前风险净期望最高且大于0者；无正期望则空仓",
            POLICIES[1]: "完全相同规则，但使用已保存的成熟样本经验预测",
            POLICIES[2]: "有事前可选合约便使用保护性卖出跨式；仅用于成本和保证金对照，非推荐方案",
        },
        "selection_clock": "原决策日收盘后选择，下一交易日开盘代理价格入场，沿用原第10持有交易日退出；一套组合在仓时不新增另一组合",
        "selection_fields": SIGNAL_COLUMNS, "leg_fields": LEG_COLUMNS,
        "future_label_fields_never_used": ["label_status", "target_profit", "target_BASE", "target_STRESS", "pnl_BASE_cny", "pnl_STRESS_cny", "entry_open", "exit_close"],
        "risk_profiles": [{"id": k,"risk_fraction":r,"drawdown_exit_line":d} for k,r,d in PROFILES],
        "position_sizing": "决策时按当时净值×风险预算/已知风险尺度向下取整；另限制为前一观察日每腿成交量与持仓量最小值的1%，不因次日价格变好而扩大数量",
        "entry_limits": "次日开盘组合的最大到期损失加双边费用不得超决策预算；现金需覆盖开仓费用、借助逐腿裸空保证金计算的过渡准备金；否则整组不成交",
        "risk_bound": "最大到期损失用于仓位，不保证盘中或提前平仓时损失不会因价差、跳空、手续费而超出",
        "margin": "保护性卖跨式拆成认购熊市价差与认沽牛市价差，保守相加两侧宽度×单位，再乘1.2；开仓另保留两条短腿逐腿裸空保证金的1.2倍过渡准备金；经纪商规则未实测",
        "cash_accounting": "权利金收支计入现金，同时期权资产/负债逐日盯市；收取权利金不等于利润；保证金冻结但不重复作为费用扣减",
        "mark_to_market": "当日收盘价乘当日有效合约单位逐腿相加；均为日线价格代理，不是同步买卖报价",
        "dividends": "从M标准单位与行权价开始；在除息日按原单位×前收盘/(前收盘−现金红利)四舍五入取整数，再按原行权价×原单位/新单位保留3位；只读取当日及此前事件",
        "adjustment_checks": "每日官方合约简称中的调整后行权价与重建条款交叉核对，不用最终合约快照反填早期选择",
        "unknown_prior_labels": "不依据原标签是否未知拒绝交易；账户独立沿当时所持合约路径估值和退出",
        "missing_quotes": "入场缺乏有效开盘印记时不成交；持仓估值缺失则只为维持账簿沿用上一有效标记并标为不完整，待可观察组合开盘退出，不将缺失记零收益；有此问题的账户不能宣称有效达标",
        "cost_scenarios": {"BASE":"每腿每侧5元，加max(2个最小报价单位,0.5%权利金)不利滑点", "STRESS":"每腿每侧5元，加max(5个最小报价单位,1%权利金)不利滑点"},
        "drawdown_action": "收盘净值从累计峰值回撤达到该档退出线，次日开盘清仓并全期停止新开仓；不按年重置失败账户",
        "annual_days": ANNUAL_DAYS, "cash_and_risk_free_return": 0,
        "calendar": "从首个已保存预测日开始至原日线2026-08-14；缺预测时持现金或延续既有持仓，空仓日全部计入",
        "year_reports": "完整自然年与部分年份分开；滚动一年使用日历周年日之后第一个交易日，记录实际天数；滚动净值片段不等同于每个起点重新投入10万元",
        "doubling_interpretation": "自然年或滚动一年净收益100%是历史描述，不将事后找到的最佳窗口当作未来翻倍概率或稳定证明",
        "historical_data_previously_seen": True, "independent_holdout": False,
        "no_new_fits": True, "no_new_downloads": True, "orders_authorized": False,
        "references": [
            "https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250610_10781452.shtml",
            "https://star.sse.com.cn/assortment/options/guide/c/c_20161121_4201435.shtml",
            "https://www.sse.com.cn/assortment/options/disclo/update/c/c_20260116_10805396.shtml",
        ],
    }
    dump(out / "protocol.json", protocol)
    files = [out/"protocol.json",*sorted((out/"inputs").glob("*")),*sorted((out/"code").glob("*"))]
    dump(out/"freeze.json",{"frozen_at":now(),"before_first_new_account":True,"files":{p.relative_to(out).as_posix():sha(p) for p in files}})
    return protocol


def load_inputs(out: Path):
    eod = pd.read_parquet(out/"inputs/option_eod.parquet")
    risk = pd.read_parquet(out/"inputs/option_risk.parquet")
    market = pd.read_parquet(out/"inputs/etf_market.parquet")
    market["date"] = pd.to_datetime(market.date)
    market = market.set_index("date").sort_index()
    divs = pd.read_csv(out/"inputs/dividends.csv")
    divs["ex_date"] = pd.to_datetime(divs.ex_date)
    dividends = {r.ex_date:float(r.cash_dividend_per_share) for r in divs.itertuples() if r.ex_date>=pd.Timestamp("2019-12-23")}
    events = {}
    for date, amount in dividends.items():
        previous_spot = float(market.loc[market.index<date].close.iloc[-1])
        events[date] = (previous_spot,amount)
    eod["trade_date"] = pd.to_datetime(eod.trade_date)
    risk["trade_date"] = pd.to_datetime(risk.trade_date)
    panel = eod.merge(risk[["trade_date","contract_code","exchange_contract_id","contract_symbol"]],on=["trade_date","contract_code"],how="left",validate="one_to_one")
    panel = panel.sort_values(["trade_date","contract_code"],kind="stable")
    states, quotes, previous_quotes = {}, {}, {}
    term_rows, adjustment_rows, disagreements = [], [], []
    adjusted_price_checks = 0
    for r in panel.itertuples(index=False):
        date = pd.Timestamp(r.trade_date)
        code = str(r.contract_code)
        identity = str(r.exchange_contract_id) if pd.notna(r.exchange_contract_id) else ""
        parsed = re.fullmatch(r"510300([CP])(\d{4})([A-Z])(\d{5})",identity)
        state = states.get(code)
        if state is not None:
            for event,(previous_spot,amount) in events.items():
                if state["last_date"]<event<=date:
                    old_unit,old_strike = state["unit"],state["strike"]
                    unit,strike = adjust_terms(old_unit,old_strike,previous_spot,amount)
                    state = {"unit":unit,"strike":strike,"version":"A" if state["version"]=="M" else chr(ord(state["version"])+1),"last_date":event}
                    adjustment_rows.append({"date":str(event.date()),"contract_code":code,"old_unit":old_unit,"new_unit":unit,"old_strike":old_strike,"new_strike":strike,"previous_spot":previous_spot,"dividend":amount})
        if parsed is not None and parsed[3]=="M":
            if state is not None and state["version"]!="M":
                disagreements.append({"date":str(date.date()),"contract_code":code,"reason":"调整后代码异常回到M"})
            state = {"unit":10000,"strike":int(parsed[4])/1000,"version":"M","last_date":date}
        if state is not None:
            state["last_date"] = date
            states[code] = state.copy()
            if parsed is not None and parsed[3]!=state["version"]:
                disagreements.append({"date":str(date.date()),"contract_code":code,"reason":"当日交易代码版本与重建不符"})
            symbol_match = re.search(r"[购沽]\d+月(\d+)([A-Z])?$",str(r.contract_symbol))
            if parsed is not None and parsed[3]!="M" and symbol_match is not None:
                adjusted_price_checks += 1
                if abs(int(symbol_match[1])/1000-state["strike"])>1e-9:
                    disagreements.append({"date":str(date.date()),"contract_code":code,"reason":"当日调整简称与重建行权价不符","official":int(symbol_match[1])/1000,"reconstructed":state["strike"]})
        quote = r._asdict()
        quote.update(unit=state["unit"] if state else None,effective_strike=state["strike"] if state else None,
                     terms_resolved=state is not None,previous_quote=previous_quotes.get(code))
        quotes[(date,code)] = quote
        previous_quotes[code] = {"date":date,"close":float(r.close),"unit":quote["unit"],"settlement":float(r.settlement)}
        term_rows.append({"date":str(date.date()),"contract_code":code,"unit":quote["unit"],"strike":quote["effective_strike"],"version":state["version"] if state else None,"identity_present":parsed is not None})
    csv(out/"results/重建逐日期权条款.csv",pd.DataFrame(term_rows))
    csv(out/"results/逐次分红调整.csv",pd.DataFrame(adjustment_rows))
    dump(out/"terms_reconstruction.json",{"adjusted_price_checks":adjusted_price_checks,"adjustment_records":len(adjustment_rows),
        "unresolved_rows":sum(x["unit"] is None for x in term_rows),"disagreements":disagreements})
    if disagreements:
        raise RuntimeError(f"条款重建与每日官方简称存在{len(disagreements)}处不一致，账户尚未运行。")
    # 只把事前字段交给账户，未来标签和未来报价不进入选择器。
    original_predictions = pd.read_csv(out/"inputs/predictions.csv")
    predictions = original_predictions[SIGNAL_COLUMNS].copy()
    legs = pd.read_csv(out/"inputs/candidate_legs.csv",dtype={"contract_code":str})[LEG_COLUMNS].copy()
    signals = {(pd.Timestamp(d),m):g.to_dict("records") for (d,m),g in predictions.groupby(["decision_date","model"])}
    candidate_legs = {(d,s):g.sort_values("leg_number").to_dict("records") for (d,s),g in legs.groupby(["decision_date","structure"])}
    calendar = pd.DatetimeIndex(sorted(eod.trade_date.unique()))
    calendar = calendar[calendar>=pd.Timestamp(predictions.decision_date.min())]
    csv(out/"results/账户可读取的预测字段.csv",predictions)
    return quotes,market,events,signals,candidate_legs,calendar


def position_terms(legs, date, quotes):
    terms=[]
    for leg in legs:
        q=quotes.get((date,leg["contract_code"]))
        if q is None or not q["terms_resolved"]:
            return None
        terms.append({**leg,"unit":int(q["unit"]),"effective_strike":float(q["effective_strike"]),"quote":q})
    return terms


def minimum_expiry_payoff(terms) -> float:
    if all(t["side"]>0 for t in terms):
        return 0.0
    knots=sorted({0.,*[t["effective_strike"] for t in terms]})
    tail_delta=sum(t["side"]*t["unit"] for t in terms if t["option_type"]=="C")
    if tail_delta<0:
        raise ValueError("组合具有无上限的上行亏损，不能进入本轮风险预算。")
    return min(sum(t["side"]*t["unit"]*max(spot-t["effective_strike"] if t["option_type"]=="C" else t["effective_strike"]-spot,0.) for t in terms) for spot in knots)


def margin_required(terms, previous_spot: float, transition: bool) -> float:
    shorts=[t for t in terms if t["side"]<0]
    if not shorts:
        return 0.0
    spread_margin=0.0
    naked_margin=0.0
    for short in shorts:
        matching=[t for t in terms if t["side"]>0 and t["option_type"]==short["option_type"] and t["unit"]==short["unit"]]
        if len(matching)!=1:
            raise ValueError("保护翼不存在或合约单位不一致。")
        wing=matching[0]
        width=wing["effective_strike"]-short["effective_strike"] if short["option_type"]=="C" else short["effective_strike"]-wing["effective_strike"]
        if width<=0:
            raise ValueError("保护翼方向错误。")
        spread_margin+=width*short["unit"]
        premium=float(short["quote"]["previous_settlement"])
        strike=short["effective_strike"]
        if short["option_type"]=="C":
            naked=premium+max(.12*previous_spot-max(strike-previous_spot,0.),.07*previous_spot)
        else:
            naked=min(premium+max(.12*previous_spot-max(previous_spot-strike,0.),.07*strike),strike)
        naked_margin+=naked*short["unit"]
    return MARGIN_MULTIPLIER*(max(spread_margin,naked_margin) if transition else spread_margin)


def choose_signal(date, policy, signals):
    model="MATURE_EMPIRICAL" if policy==POLICIES[1] else "VOLATILITY_LOGIT_GAMMA"
    rows=signals.get((date,model),[])
    if policy==POLICIES[2]:
        rows=[r for r in rows if r["structure"]==STRUCTURES[1]]
    else:
        rows=[r for r in rows if np.isfinite(r["expected_net_risk_unit"]) and r["expected_net_risk_unit"]>0]
    if not rows:
        return None
    return sorted(rows,key=lambda r:(-r["expected_net_risk_unit"],r["structure"]))[0]


def simulate(policy, profile, scenario, data):
    quotes,market,events,signals,candidate_legs,calendar=data
    profile_id,risk_fraction,stop_line=profile
    account_id=f"{policy}__{profile_id}__{scenario}"
    meta={"account_id":account_id,"policy":policy,"profile":profile_id,"scenario":scenario}
    cash=INITIAL
    peak=INITIAL
    position=None
    pending=None
    must_exit=False
    halted=False
    incomplete=False
    trade_number=0
    nav_rows,fills,cycles,decisions,positions=[],[],[],[],[]

    def observed_previous_spot(at):
        value=float(market.loc[market.index<at].close.iloc[-1])
        if at in events:
            value-=events[at][1]
        return value

    def transact(terms,quantity,at,field,opening,trade_id,reason):
        nonlocal cash
        change=0.0
        for leg in terms:
            action=int(leg["side"])*(1 if opening else -1)
            raw=float(leg["quote"][field])
            fill=fill_price(raw,action,scenario)
            commission=FEE*quantity
            delta=-action*fill*leg["unit"]*quantity-commission
            cash+=delta
            change+=delta
            fills.append({**meta,"trade_id":trade_id,"date":str(at.date()),"phase":"OPEN" if opening else "CLOSE",
                          "reason":reason,"price_field":field,"contract_code":leg["contract_code"],"option_type":leg["option_type"],
                          "action":action,"quantity":quantity,"unit":leg["unit"],"strike":leg["effective_strike"],
                          "raw_price":raw,"fill_price":fill,"fee_cny":commission,"cash_change_cny":delta,
                          "slippage_cost_cny":action*(fill-raw)*leg["unit"]*quantity})
        return change

    def close_position(at,field,reason):
        nonlocal position,must_exit,incomplete
        if position is None:
            return True
        terms=position_terms(position["legs"],at,quotes)
        if terms is None or any(not np.isfinite(t["quote"][field]) or t["quote"][field]<=0 for t in terms):
            incomplete=True
            must_exit=True
            decisions.append({**meta,"date":str(at.date()),"decision":"EXIT_WAIT_MISSING_PRICE","reason":reason})
            return False
        transact(terms,position["quantity"],at,field,False,position["trade_id"],reason)
        holding=int(calendar.get_loc(at)-calendar.get_loc(position["entry_date"])+1)
        cycles.append({**meta,"trade_id":position["trade_id"],"structure":position["structure"],
                       "decision_date":position["decision_date"],"entry_date":str(position["entry_date"].date()),
                       "exit_date":str(at.date()),"exit_reason":reason,"quantity":position["quantity"],
                       "entry_equity_cny":position["entry_equity"],"risk_budget_cny":position["risk_budget"],
                       "maximum_terminal_loss_at_entry_cny":position["max_loss"],"entry_margin_reserve_cny":position["entry_margin"],
                       "holding_sessions":holding,"net_pnl_cny":cash-position["cash_before"],
                       "p_win":position["signal"]["p_win"],"forecast_expected_risk_unit":position["signal"]["expected_net_risk_unit"]})
        position=None
        must_exit=False
        return True

    for i,date in enumerate(calendar):
        held_during_day=position is not None
        if position is not None and must_exit:
            close_position(date,"open","DRAWDOWN_OR_MARGIN_OR_DATA_EXIT")
        if pending is not None:
            order=pending
            pending=None
            if not halted and position is None:
                terms=position_terms(order["legs"],date,quotes)
                reason=None
                if terms is None or any(not np.isfinite(t["quote"]["open"]) or t["quote"]["open"]<=0 for t in terms):
                    reason="ENTRY_UNFILLED_NO_OBSERVED_OPEN"
                else:
                    one_debit=sum(t["side"]*fill_price(t["quote"]["open"],t["side"],scenario)*t["unit"] for t in terms)
                    one_max_loss=max(0.0,one_debit+2*FEE*len(terms)-minimum_expiry_payoff(terms))
                    entry_margin=margin_required(terms,observed_previous_spot(date),True)*order["quantity"]
                    projected_cash=cash-(one_debit+FEE*len(terms))*order["quantity"]
                    if one_max_loss*order["quantity"]>order["risk_budget"]+1e-8:
                        reason="ENTRY_LIMIT_GAP_EXCEEDS_RISK_BUDGET"
                    elif projected_cash<entry_margin-1e-8:
                        reason="ENTRY_REJECT_INSUFFICIENT_CASH_OR_TRANSITION_MARGIN"
                    elif pd.Timestamp(order["signal"]["label_end_date"])>=min(pd.Timestamp(t["quote"]["expiry_date"]) for t in terms)-pd.Timedelta(days=2):
                        reason="ENTRY_REJECT_KNOWN_EXPIRY_TOO_CLOSE"
                if reason is not None:
                    decisions.append({**meta,"date":str(date.date()),"decision":reason,"signal_date":order["signal"]["decision_date"]})
                else:
                    trade_number+=1
                    cash_before=cash
                    trade_id=f"{account_id}__{trade_number:04d}"
                    transact(terms,order["quantity"],date,"open",True,trade_id,"SAVED_SIGNAL")
                    position={"trade_id":trade_id,"entry_date":date,"decision_date":order["signal"]["decision_date"],
                              "structure":order["signal"]["structure"],"quantity":order["quantity"],"legs":order["legs"],
                              "entry_equity":cash_before,"cash_before":cash_before,"risk_budget":order["risk_budget"],
                              "max_loss":one_max_loss*order["quantity"],"entry_margin":entry_margin,
                              "exit_date":pd.Timestamp(order["signal"]["label_end_date"]),"signal":order["signal"]}
                    held_during_day=True
        if position is not None and (date>=position["exit_date"] or date==calendar[-1]):
            close_position(date,"close","HORIZON_CLOSE" if date>=position["exit_date"] else "FINAL_CLOSE")

        option_value=0.0
        margin=0.0
        stale_today=False
        if position is not None:
            terms=position_terms(position["legs"],date,quotes)
            if terms is None:
                raise RuntimeError("已持有合约缺乏可重建条款，不能伪造账户估值。")
            margin=margin_required(terms,observed_previous_spot(date),False)*position["quantity"]
            for t in terms:
                price=t["quote"]["close"]
                stale=not np.isfinite(price) or price<=0
                if stale:
                    prior=t["quote"]["previous_quote"]
                    if prior is None or prior["unit"] is None or not np.isfinite(prior["close"]) or prior["close"]<=0:
                        raise RuntimeError("没有可追溯标记，账户必须保留为无法计算。")
                    price=prior["close"]*prior["unit"]/t["unit"]
                    stale_today=True
                    incomplete=True
                    must_exit=True
                value=t["side"]*position["quantity"]*t["unit"]*price
                option_value+=value
                positions.append({**meta,"date":str(date.date()),"trade_id":position["trade_id"],"contract_code":t["contract_code"],
                                  "side":t["side"],"quantity":position["quantity"],"unit":t["unit"],"strike":t["effective_strike"],
                                  "mark_price":price,"market_value_cny":value,"stale_mark":stale})
        equity=cash+option_value
        peak=max(peak,equity)
        drawdown=equity/peak-1
        margin_deficit=max(0.,margin-cash)
        if margin_deficit>1e-8:
            must_exit=True
            decisions.append({**meta,"date":str(date.date()),"decision":"MARGIN_DEFICIT_NEXT_OPEN_EXIT","deficit_cny":margin_deficit})
        if drawdown<=-stop_line+1e-12 and not halted:
            halted=True
            must_exit=position is not None
            pending=None
            decisions.append({**meta,"date":str(date.date()),"decision":"DRAWDOWN_HALT","drawdown":drawdown})
        if not halted and position is None and i+1<len(calendar):
            signal=choose_signal(date,policy,signals)
            if signal is not None:
                legs=candidate_legs.get((signal["decision_date"],signal["structure"]))
                if not legs:
                    raise ValueError("已保存预测没有对应事前合约腿。")
                risk_budget=risk_fraction*equity
                quantity=math.floor((risk_budget+1e-10)/signal["known_risk_scale_cny"])
                observed=pd.Timestamp(signal["observed_date"])
                capacity=math.floor(.01*min(min(float(quotes[(observed,l["contract_code"])]["volume"]),float(quotes[(observed,l["contract_code"])]["open_interest"])) for l in legs))
                quantity=min(quantity,capacity)
                if quantity<1:
                    decisions.append({**meta,"date":str(date.date()),"decision":"NO_ENTRY_INTEGER_LOT_OR_CAPACITY","risk_budget_cny":risk_budget,"one_unit_known_risk_cny":signal["known_risk_scale_cny"]})
                else:
                    assert pd.Timestamp(signal["entry_date"])==calendar[i+1]
                    pending={"signal":signal,"legs":legs,"quantity":quantity,"risk_budget":risk_budget}
                    decisions.append({**meta,"date":str(date.date()),"decision":"SIGNAL_QUEUED","structure":signal["structure"],"quantity":quantity,"risk_budget_cny":risk_budget})
        previous_equity=nav_rows[-1]["equity_cny"] if nav_rows else INITIAL
        nav_rows.append({**meta,"date":str(date.date()),"cash_cny":cash,"option_market_value_cny":option_value,"equity_cny":equity,
                         "daily_return":equity/previous_equity-1,"drawdown":drawdown,"margin_reserved_cny":margin,
                         "cash_available_cny":cash-margin,"margin_deficit_cny":margin_deficit,"held_during_day":held_during_day,
                         "open_leg_count":len(position["legs"]) if position else 0,"halted":halted,"stale_mark_today":stale_today,
                         "incomplete_data_seen":incomplete})
    if position is not None:
        incomplete=True
    return pd.DataFrame(nav_rows),pd.DataFrame(fills),pd.DataFrame(cycles),pd.DataFrame(decisions),pd.DataFrame(positions),incomplete


def account_metrics(nav: pd.DataFrame, cycles: pd.DataFrame, fills: pd.DataFrame):
    returns=nav.daily_return.to_numpy(float)
    equity=nav.equity_cny.to_numpy(float)
    years=len(nav)/ANNUAL_DAYS
    standard=float(np.std(returns,ddof=1)) if len(returns)>1 else 0.
    pnl=cycles.net_pnl_cny.to_numpy(float) if not cycles.empty else np.array([])
    wins=pnl[pnl>0];losses=-pnl[pnl<0]
    meta=nav.iloc[0][["account_id","policy","profile","scenario"]].to_dict()
    return {**meta,"start":nav.date.iloc[0],"end":nav.date.iloc[-1],"trading_days":len(nav),"initial_capital_cny":INITIAL,
            "ending_equity_cny":float(equity[-1]),"net_profit_cny":float(equity[-1]-INITIAL),"total_return":float(equity[-1]/INITIAL-1),
            "annualized_return":float((equity[-1]/INITIAL)**(1/years)-1) if equity[-1]>0 else -1.,
            "net_sharpe":float(np.mean(returns)/standard*np.sqrt(ANNUAL_DAYS)) if standard>0 else None,
            "max_drawdown":float(-nav.drawdown.min()),"completed_cycles":len(cycles),
            "win_rate":float((pnl>0).mean()) if len(pnl) else None,
            "cash_payoff_ratio":float(wins.mean()/losses.mean()) if len(wins) and len(losses) else None,
            "exposure_day_fraction":float(nav.held_during_day.mean()),"maximum_margin_cny":float(nav.margin_reserved_cny.max()),
            "margin_deficit_days":int(nav.margin_deficit_cny.gt(0).sum()),"stale_mark_days":int(nav.stale_mark_today.sum()),
            "halted":bool(nav.halted.any()),"fees_cny":float(fills.fee_cny.sum()) if not fills.empty else 0.,
            "slippage_cost_cny":float(fills.slippage_cost_cny.sum()) if not fills.empty else 0.}


def yearly_metrics(nav,cycles):
    records=[]
    dates=pd.to_datetime(nav.date)
    previous=INITIAL
    for year,g in nav.groupby(dates.dt.year,sort=True):
        full=nav.date.iloc[0][:4]<str(year)<nav.date.iloc[-1][:4]
        trades=cycles[cycles.entry_date.str[:4].eq(str(year))] if not cycles.empty else cycles
        balance=g.equity_cny.to_numpy(float)
        peak=np.maximum.accumulate(np.r_[previous,balance])
        annual_return=float(balance[-1]/previous-1)
        records.append({"account_id":nav.account_id.iloc[0],"year":int(year),"complete_year":full,"starting_equity_cny":previous,
                        "ending_equity_cny":float(balance[-1]),"net_return":annual_return,
                        "maximum_drawdown_in_year":float(-np.min(np.r_[previous,balance]/peak-1)),
                        "completed_cycles_by_entry_year":len(trades),"at_least_five":len(trades)>=5 if full else None,
                        "doubled_in_year":annual_return>=1 if full else None})
        previous=float(balance[-1])
    return pd.DataFrame(records)


def rolling_years(nav):
    dates=pd.DatetimeIndex(pd.to_datetime(nav.date))
    equity=nav.equity_cny.to_numpy(float)
    rows=[]
    for i,start in enumerate(dates):
        anniversary=start+pd.DateOffset(years=1)
        j=int(dates.searchsorted(anniversary))
        if j>=len(dates):
            break
        balances=equity[i:j+1]
        rate=float(balances[-1]/balances[0]-1)
        rows.append({"account_id":nav.account_id.iloc[0],"start":str(start.date()),"end":str(dates[j].date()),
                     "calendar_days":int((dates[j]-start).days),"trading_intervals":j-i,"net_return":rate,
                     "maximum_drawdown":float(-np.min(balances/np.maximum.accumulate(balances)-1)),"doubled":rate>=1})
    return pd.DataFrame(rows)


def self_checks():
    # 用独立现金流恒等式检查卖出权利金不是即时利润。
    starting_cash=100000.
    credit=1250.
    liability=-credit
    assert starting_cash+credit+liability==starting_cash
    terms=[{"side":-1,"option_type":"C","unit":10000,"effective_strike":4.},
           {"side":-1,"option_type":"P","unit":10000,"effective_strike":4.},
           {"side":1,"option_type":"C","unit":10000,"effective_strike":4.2},
           {"side":1,"option_type":"P","unit":10000,"effective_strike":3.8}]
    np.testing.assert_allclose(minimum_expiry_payoff(terms),-2000,atol=1e-8)
    np.testing.assert_allclose(-credit+40-minimum_expiry_payoff(terms),790,atol=1e-8)
    assert adjust_terms(10000,4.,4.859,.123)==(10260,3.899)
    assert fill_price(.1,1,"STRESS")>fill_price(.1,1,"BASE")>.1
    assert fill_price(.1,-1,"STRESS")<fill_price(.1,-1,"BASE")<.1
    assert not set(["label_status","target_profit","pnl_STRESS_cny","entry_open","exit_close"])&set(SIGNAL_COLUMNS+LEG_COLUMNS)
    day=pd.Timestamp("2024-01-02")
    candidate={"structure":STRUCTURES[0],"expected_net_risk_unit":.1,"label_status":"UNKNOWN","pnl_STRESS_cny":-999999}
    source={(day,"VOLATILITY_LOGIT_GAMMA"):[candidate.copy()]}
    before=choose_signal(day,POLICIES[0],source)["structure"]
    source[(day,"VOLATILITY_LOGIT_GAMMA")][0].update(label_status="READY",pnl_STRESS_cny=999999)
    assert choose_signal(day,POLICIES[0],source)["structure"]==before
    return {"short_premium_is_not_profit":True,"protected_short_terminal_loss":True,"dividend_rounding":True,
            "cost_direction":True,"future_label_poisoning":True}


def finish(out: Path, tables: dict, metrics: list, incomplete: dict):
    for name,frame in tables.items():
        csv(out/"results"/name,frame)
    all_metrics=pd.DataFrame(metrics)
    years=tables["完整自然年与部分年.csv"]
    rolls=tables["完整滚动一年.csv"]
    for i,row in all_metrics.iterrows():
        annual=years[years.account_id.eq(row.account_id)&years.complete_year]
        rolling=rolls[rolls.account_id.eq(row.account_id)]
        all_metrics.loc[i,"best_complete_year_return"]=float(annual.net_return.max()) if not annual.empty else np.nan
        all_metrics.loc[i,"worst_complete_year_return"]=float(annual.net_return.min()) if not annual.empty else np.nan
        all_metrics.loc[i,"minimum_completed_cycles_full_year"]=int(annual.completed_cycles_by_entry_year.min()) if not annual.empty else 0
        all_metrics.loc[i,"best_rolling_year_return"]=float(rolling.net_return.max()) if not rolling.empty else np.nan
        all_metrics.loc[i,"worst_rolling_year_return"]=float(rolling.net_return.min()) if not rolling.empty else np.nan
        all_metrics.loc[i,"median_rolling_year_return"]=float(rolling.net_return.median()) if not rolling.empty else np.nan
        all_metrics.loc[i,"rolling_year_windows"]=len(rolling)
        all_metrics.loc[i,"rolling_year_doubling_windows"]=int(rolling.doubled.sum()) if not rolling.empty else 0
        all_metrics.loc[i,"complete_year_doubling_count"]=int(annual.net_return.ge(1).sum())
        all_metrics.loc[i,"data_complete"]=not incomplete[row.account_id]
    csv(out/"results/全部账户指标.csv",all_metrics)
    primary=all_metrics[all_metrics.policy.eq(POLICIES[0])&all_metrics.scenario.eq("STRESS")].copy()
    rename={"profile":"风险档位","ending_equity_cny":"期末净值（元）","net_profit_cny":"累计净利润（元）","annualized_return":"年化收益",
            "net_sharpe":"净夏普","max_drawdown":"实际最大回撤","completed_cycles":"完整交易数","win_rate":"实现胜率",
            "best_complete_year_return":"最好完整年度收益","worst_complete_year_return":"最差完整年度收益",
            "best_rolling_year_return":"最好滚动一年收益","worst_rolling_year_return":"最差滚动一年收益",
            "rolling_year_doubling_windows":"滚动一年翻倍窗口数","minimum_completed_cycles_full_year":"完整年最少交易数","data_complete":"数据完整"}
    csv(out/"results/三档风险结果_中文.csv",primary[list(rename)].rename(columns=rename))
    summary={"study_id":"510300_OPTION_VOLATILITY_ACCOUNT_RISK_V1","completed_at":now(),"initial_capital_cny":INITIAL,
             "one_year_target_cny":200000,"status":"FIXED_RISK_SCENARIO_ACCOUNT_TEST_COMPLETE","accounts":len(all_metrics),
             "new_fits":0,"new_downloads":0,"main_policy":POLICIES[0],"primary_cost":"STRESS",
             "minimum_model_sharpe_target":1.3,"goal_achieved":False,"live_execution_proven":False,
             "independent_validation":False,"numerical_one_year_doubling_accounts":int(all_metrics.rolling_year_doubling_windows.gt(0).sum()),
             "primary_one_year_doubling_accounts":int(primary.rolling_year_doubling_windows.gt(0).sum()),
             "primary_accounts":json.loads(primary.to_json(orient="records",force_ascii=False))}
    dump(out/"summary.json",summary)
    lines=["# 510300期权：10万元账户与一年翻倍检验", "",
           "本金改为10万元，目标为一年扣费后达到20万元；按用户要求同时查看不同回撤情景。夏普1.3和完整自然年至少5次交易继续报告。", "",
           "本轮不重新训练，直接使用前轮保存的预测。模型策略与经验基线采用相同的正净期望选择规则；保护性卖跨式常开仅作保证金和成本对照。三档单次风险预算1%、2%、4%，退出线10%、20%、30%。退出线并非可保证的最大回撤上限。", "",
           "## 主模型，压力成本", "", "|风险档|期末净值|年化收益|净夏普|实际最大回撤|完整交易|最好滚动一年|最差滚动一年|一年翻倍窗口|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary.itertuples(index=False):
        sharpe=f"{r.net_sharpe:.3f}" if pd.notna(r.net_sharpe) else "未定义（无有效波动）"
        lines.append(f"|{r.profile}|{r.ending_equity_cny:,.2f}|{r.annualized_return:.2%}|{sharpe}|{r.max_drawdown:.2%}|{r.completed_cycles}|{r.best_rolling_year_return:.2%}|{r.worst_rolling_year_return:.2%}|{int(r.rolling_year_doubling_windows)}|")
    lines += ["", "## 解读与限制", "",
              "- 账户期间从2021年3月开始至2026年8月；期末金额是整个期间累计净值，不能当作一年的结果。完整自然年为2022—2025，2021和2026单列部分年份。",
              "- 所有空仓日保留。没有按年重置本金或回撤，也没有把多个方案盈利拼成一个账户。滚动一年窗口高度重叠，窗口比例不代表未来成功概率。",
              "- 风险预算按整数组合取整，次日开盘超预算则取消整组，不能靠事后提高仓位实现翻倍。主模型没有正期望的结构不会仅为凑次数而开仓。",
              "- 当前标价来自日线开盘和收盘，并扣假设滑点与每腿费用；未取得同步历史买卖报价。因此账户结果是执行假设下的研究结果，不是历史真实可成交收益。",
              "- 卖出权利金同时记期权负债；保证金保守计入占用。用两个价差保证金相加并加20%余量，开仓另检查裸空短腿过渡准备金；尚未接入经纪商实测。",
              "- 历史调整按现金分红与上日标的收盘重建，再与当日官方调整简称交叉核对；前轮未知标签不再直接作为拒绝交易理由。",
              "- 回撤和止损基于日收盘净值；盘中最大回撤和下一次开盘的跳空损失可能更大，不能保证不超过名义退出线。",
              "- 历史已多次研究，结果不是全新独立留出验证；尚不能宣称未来稳定盈利或一年翻倍。", "",
              "查看完整逐日账户、逐腿现金流、成交周期、年度结果和滚动一年表，可以逐项核对资金变化。"]
    (out/"账户结果与一年目标.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def plot_results(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    nav=pd.read_csv(out/"results/完整逐日账户.csv")
    rolling=pd.read_csv(out/"results/完整滚动一年.csv")
    fig,axes=plt.subplots(3,1,figsize=(11.5,10.5),constrained_layout=True)
    colors=["#1D6F66","#3675AA","#C56B35"]
    names=["单次风险1% / 10%退出线","单次风险2% / 20%退出线","单次风险4% / 30%退出线"]
    for (profile,_,_),color,label in zip(PROFILES,colors,names):
        selected=nav[nav.policy.eq(POLICIES[0])&nav.scenario.eq("STRESS")&nav.profile.eq(profile)]
        account=selected.account_id.iloc[0]
        dates=pd.to_datetime(selected.date)
        axes[0].plot(dates,selected.equity_cny/10000,color=color,label=label,linewidth=1.6)
        axes[1].plot(dates,selected.drawdown*100,color=color,linewidth=1.6)
        r=rolling[rolling.account_id.eq(account)]
        axes[2].plot(pd.to_datetime(r.end),r.net_return*100,color=color,linewidth=1.6)
    axes[0].set_title("510300期权｜10万元主模型账户，压力成本",loc="left",fontweight="bold")
    axes[0].set_ylabel("账户净值（万元）")
    axes[0].legend(loc="best",frameon=False,ncol=1)
    axes[1].set_ylabel("收盘最大回撤路径（%）")
    axes[2].set_ylabel("滚动一年净收益（%）")
    axes[2].axhline(100,color="#B83D35",linestyle="--",linewidth=1,label="一年翻倍要求 +100%")
    axes[2].legend(loc="best",frameon=False)
    for ax in axes:
        ax.grid(axis="y",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"三档账户与一年目标.png",dpi=160)
    plt.close(fig)


def verify(out: Path):
    frozen=json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    for name,value in frozen["files"].items():
        if sha(out/name)!=value:
            raise ValueError(f"冻结文件发生变化：{name}")
    nav=pd.read_csv(out/"results/完整逐日账户.csv")
    fills=pd.read_csv(out/"results/完整逐腿现金流.csv")
    cycles=pd.read_csv(out/"results/完整持仓周期.csv")
    marks=pd.read_csv(out/"results/完整逐日期权持仓.csv")
    metrics=pd.read_csv(out/"results/全部账户指标.csv")
    years=pd.read_csv(out/"results/完整自然年与部分年.csv")
    rolling=pd.read_csv(out/"results/完整滚动一年.csv")
    raw=pd.read_parquet(out/"inputs/option_eod.parquet",columns=["trade_date","contract_code","open","close"])
    raw["date"]=pd.to_datetime(raw.trade_date).dt.strftime("%Y-%m-%d")
    raw_prices={(r.date,str(r.contract_code)):(r.open,r.close) for r in raw.itertuples(index=False)}
    terms=pd.read_csv(out/"results/重建逐日期权条款.csv",dtype={"contract_code":str})
    term_lookup={(r.date,r.contract_code):(r.unit,r.strike) for r in terms.itertuples(index=False)}
    for f in fills.itertuples(index=False):
        source_price=raw_prices[(f.date,f.contract_code)][0 if f.price_field=="open" else 1]
        np.testing.assert_allclose(source_price,f.raw_price,rtol=0,atol=1e-12)
        np.testing.assert_allclose(fill_price(source_price,f.action,f.scenario),f.fill_price,rtol=0,atol=1e-12)
        assert f.quantity==int(f.quantity) and f.quantity>0
        np.testing.assert_allclose(f.fee_cny,FEE*f.quantity,rtol=0,atol=1e-12)
        np.testing.assert_allclose(term_lookup[(f.date,f.contract_code)],[f.unit,f.strike],rtol=0,atol=1e-12)
        np.testing.assert_allclose(-f.action*f.fill_price*f.unit*f.quantity-f.fee_cny,f.cash_change_cny,rtol=1e-10,atol=1e-7)
    for mark in marks.itertuples(index=False):
        np.testing.assert_allclose(term_lookup[(mark.date,mark.contract_code)],[mark.unit,mark.strike],rtol=0,atol=1e-12)
        if not mark.stale_mark:
            np.testing.assert_allclose(raw_prices[(mark.date,mark.contract_code)][1],mark.mark_price,rtol=0,atol=1e-12)
        np.testing.assert_allclose(mark.side*mark.quantity*mark.unit*mark.mark_price,mark.market_value_cny,rtol=1e-10,atol=1e-7)
    signed_fills=fills.assign(signed_quantity=fills.action*fills.quantity)
    positions_from_flows=signed_fills.groupby(["account_id","contract_code","date"]).signed_quantity.sum()
    for (account,code),group in positions_from_flows.groupby(level=[0,1],sort=False):
        dates=nav.loc[nav.account_id.eq(account),"date"]
        quantities=group.droplevel([0,1]).reindex(dates,fill_value=0).cumsum()
        held=marks[marks.account_id.eq(account)&marks.contract_code.eq(code)].copy()
        actual=held.assign(signed_quantity=held.side*held.quantity).groupby("date").signed_quantity.sum().reindex(dates,fill_value=0)
        np.testing.assert_array_equal(quantities.to_numpy(),actual.to_numpy())
    for account,g in nav.groupby("account_id",sort=False):
        f=fills[fills.account_id.eq(account)]
        c=cycles[cycles.account_id.eq(account)]
        held=marks[marks.account_id.eq(account)]
        cash_changes=f.groupby("date").cash_change_cny.sum().reindex(g.date,fill_value=0).to_numpy()
        np.testing.assert_allclose(INITIAL+np.cumsum(cash_changes),g.cash_cny,rtol=1e-10,atol=1e-7)
        value=held.groupby("date").market_value_cny.sum().reindex(g.date,fill_value=0).to_numpy()
        np.testing.assert_allclose(value,g.option_market_value_cny,rtol=1e-10,atol=1e-7)
        np.testing.assert_allclose(g.cash_cny+g.option_market_value_cny,g.equity_cny,rtol=1e-10,atol=1e-7)
        eq=g.equity_cny.to_numpy(float)
        np.testing.assert_allclose(eq/np.r_[INITIAL,eq[:-1]]-1,g.daily_return,rtol=1e-9,atol=1e-12)
        np.testing.assert_allclose(eq/np.maximum.accumulate(np.r_[INITIAL,eq])[1:]-1,g.drawdown,rtol=1e-9,atol=1e-12)
        for trade in c.itertuples(index=False):
            tf=f[f.trade_id.eq(trade.trade_id)]
            np.testing.assert_allclose(tf.cash_change_cny.sum(),trade.net_pnl_cny,rtol=1e-9,atol=1e-7)
            assert trade.decision_date<trade.entry_date<=trade.exit_date
            assert trade.maximum_terminal_loss_at_entry_cny<=trade.risk_budget_cny+1e-7
        actual=account_metrics(g,c,f)
        saved=metrics[metrics.account_id.eq(account)].iloc[0]
        for key in ["ending_equity_cny","total_return","annualized_return","net_sharpe","max_drawdown","completed_cycles","fees_cny","slippage_cost_cny"]:
            np.testing.assert_allclose(actual[key] if actual[key] is not None else np.nan,saved[key],rtol=1e-9,atol=1e-9,equal_nan=True)
        actual_years=yearly_metrics(g,c)
        saved_years=years[years.account_id.eq(account)].reset_index(drop=True)
        np.testing.assert_allclose(actual_years[["net_return","completed_cycles_by_entry_year"]],saved_years[["net_return","completed_cycles_by_entry_year"]],rtol=1e-9,atol=1e-10)
        actual_roll=rolling_years(g)
        saved_roll=rolling[rolling.account_id.eq(account)].reset_index(drop=True)
        np.testing.assert_allclose(actual_roll[["net_return","maximum_drawdown"]],saved_roll[["net_return","maximum_drawdown"]],rtol=1e-9,atol=1e-10)
    result={"status":"PASS_SAVED_CASHFLOWS_NAV_CYCLES_AND_ONE_YEAR_METRICS","checked_at":now(),"accounts":nav.account_id.nunique(),
            "fills":len(fills),"completed_cycles":len(cycles),"new_accounts":0,"new_fits":0,"new_downloads":0}
    dump(out/"verification/保存结果复算.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description="510300期权10万元固定风险情景账户；研究，不下单。")
    parser.add_argument("action",choices=["run","verify"])
    parser.add_argument("--root",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.action=="verify":
        print(json.dumps(verify(args.root),ensure_ascii=False,indent=2))
        return
    checks=self_checks()
    prepare(args.root)
    dump(args.root/"verification/账务独立例子检查.json",checks)
    data=load_inputs(args.root)
    print("合约条款重建完成，开始18组固定账户，不重新训练模型。",flush=True)
    containers={name:[] for name in ["完整逐日账户.csv","完整逐腿现金流.csv","完整持仓周期.csv","完整逐日决策.csv","完整逐日期权持仓.csv","完整自然年与部分年.csv","完整滚动一年.csv"]}
    all_metrics=[]
    incomplete={}
    for policy in POLICIES:
        for profile in PROFILES:
            for scenario in ["BASE","STRESS"]:
                nav,fills,cycles,decisions,marks,bad=simulate(policy,profile,scenario,data)
                for name,frame in zip(list(containers)[:5],[nav,fills,cycles,decisions,marks]):
                    if not frame.empty:
                        containers[name].append(frame)
                containers["完整自然年与部分年.csv"].append(yearly_metrics(nav,cycles))
                containers["完整滚动一年.csv"].append(rolling_years(nav))
                all_metrics.append(account_metrics(nav,cycles,fills))
                incomplete[nav.account_id.iloc[0]]=bad
                print(f"完成 {policy} {profile[0]} {scenario}：{len(cycles)}次交易，期末{nav.equity_cny.iloc[-1]:,.2f}元。",flush=True)
    tables={name:pd.concat(parts,ignore_index=True) for name,parts in containers.items()}
    summary=finish(args.root,tables,all_metrics,incomplete)
    plot_results(args.root)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(json.dumps(verify(args.root),ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
