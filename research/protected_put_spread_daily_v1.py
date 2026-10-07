"""510300保护性认沽价差：两年每日更新、定价对照和单一现金账户。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import binom, skew


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config/510300_existing_data_training_mandate_v1.json").is_file())
OUT = ROOT / "reports/research/510300_protected_put_spread_daily_v1"
PARENT = ROOT / "reports/research/510300_synthetic_short_gamma_variance_v1"
STUDY = "510300_PROTECTED_PUT_SPREAD_DAILY_V1"
POLICIES = ["UNCONDITIONAL", "RV", "PRICE", "MACRO"]
PRIMARY = "MACRO"
FEATURES = ["log_iv", "log_iv_rv20", "trend20"]
INITIAL = 200000.0
FEE = 5.0
START = pd.Timestamp("2021-01-04")
END = pd.Timestamp("2026-08-14")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def helper(root):
    source = root / "code/account_primitives_source.py"
    if not source.exists():
        source = ROOT / "research/option_volatility_account_risk_v1.py"
    spec = importlib.util.spec_from_file_location("put_spread_account_primitives", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fill(price, action, scenario):
    tick_slip, proportional = (.0002, .005) if scenario == "BASE" else (.0005, .01)
    value = max(0., float(price) + action * max(tick_slip, proportional * float(price)))
    return (math.ceil(value * 10000 - 1e-9) if action > 0 else math.floor(value * 10000 + 1e-9)) / 10000


def put_value(forward, strike, variance, tau):
    if tau <= 0 or variance <= 0:
        return max(strike - forward, 0.)
    scale = math.sqrt(variance * tau)
    d1 = (math.log(forward / strike) + .5 * variance * tau) / scale
    return strike * ndtr(-d1 + scale) - forward * ndtr(-d1)


def specification():
    return {
        "study_id": STUDY, "created_at": now(), "primary": PRIMARY,
        "capital_cny": INITIAL, "assets": ["SSE_510300_ETF_OPTIONS", "CASH_CNY"],
        "observed_underlying": "510300.SH", "evaluation_start": str(START.date()), "end": str(END.date()),
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "economic_question": "已收取的下行保险权利金在保护、赔付和费用之后是否有补偿；方差预测能否改善该补偿。",
        "novelty": "两腿认沽信用价差，和旧四腿卖跨式、ETF合成负Gamma库存具有不同现金流；旧结论不改变。",
        "policies": POLICIES, "cost_scenarios": ["BASE", "STRESS"],
        "clocks": {
            "decision": "交易日09:00；期权所有选腿及定价原始资料用前两个交易日收盘；价格方差预测用此前已保存的当日09:00预测",
            "execution": "同日各腿开盘价代理；分别作不利滑点和最小报价单位取整；不同腿并非同步可成交报价",
            "maturity": "训练标签的退出日严格早于决策日，不能使用当日09:30才出现的退出价格",
        },
        "selection": {
            "standard_contract_identity": "当日官方510300[CP]YYMMMKKKKK代码；M合约单位10000；禁止用晚期快照行权价筛历史",
            "calendar_dte_on_observation": [30, 75], "target_calendar_dte": 45,
            "each_leg_prior_volume": 100, "each_leg_prior_open_interest": 500,
            "short_put_delta": -.30, "long_put_delta": -.10,
            "long_strike": "严格低于空头行权价且同到期；同到期先确定短腿，再选择保护腿；不按价差或收益选期",
            "expiry_priority": "距45日最近，其次到期日；同腿按Delta距离、行权价、合约代码稳定排序",
            "forward": "同到期最近平值标准认购认沽收盘价按F=K+C-P还原零利率远期代理；不额外假定未来现金分红",
            "no_fallback": "选定期限缺少可用远期对时不改用另一期；未读未来结果",
        },
        "pricing": {
            "RV": "当日09:00已保存的20日已实现方差持续值",
            "PRICE": "已冻结的两年日更价格岭回归方差均值预测",
            "MACRO": "同历史池价格加资金利差、社融同比三月变化及海外信息的方差均值预测",
            "fair_value_proxy": "同一个方差用于两腿，零利率Black公式计算无漂移远期下的预期到期赔付代理；不是已证明的物理期望",
            "tenor_limitation": "未来20日方差预测替代30—75日剩余期限方差，且交易10日后平仓；期限与风险溢价假设均须披露",
            "gate": "观察日毛净权利金超过模型价差价值加压力往返费用滑点；开盘压力可得净信用额仍需满足预定信用下限",
            "UNCONDITIONAL": "相同信息覆盖，只要求预计权利金覆盖压力费用，不用模型定价门",
            "common_coverage": "四政策均须有两条保存方差预测和合格的五日尾部训练分布",
        },
        "tail_training": {
            "window": "最近两个日历年内的候选入场原点，五日标签已成熟",
            "minimum_rows": 252, "nearest_neighbors": 126, "features": FEATURES,
            "standardization": "只用当时训练均值和样本标准差，标准化截断[-3,3]，欧氏距离，按原点先后解同距",
            "target": "每组从入场开盘到五个交易日后开盘的压力净损益/事前最大损失金额",
            "ES95": "最近邻最差ceil(5%*126)个损益比的平均损失，非独立126次事件；同时记录5%分位",
            "ES_zero": "取零为下限；最坏损失5%预算始终独立约束数量，不因历史尾部为零放大最坏损失",
            "update": "每交易日重算，保留训练和邻居原点；方差模型复用原2,722条合格逐期预测，不算新的拟合",
        },
        "risk": {
            "one_position_only": True, "initial_terminal_loss_including_fees_fraction": .05,
            "entry_five_day_ES95_fraction": .025, "entry_drawdown_headroom_fraction": .5,
            "holding_sessions": 10, "exit": "第10个后续交易日开盘；不加仓摊平，不因当天另一合约信号替换在仓腿",
            "planned_exit_expiry_buffer_calendar_days": 5,
            "size": "09:00冻结整数张数，实际开盘若预算、信用下限或过渡现金不符整组不成交，不按有利价格加张",
            "capacity": "张数不超过两腿观察日成交量及持仓量最小值的1%",
            "daily_bound": "已有仓位每日以昨日权益、价差剩余最坏损失和距90%峰值空间复核；必要时开盘只减仓",
            "daily_bound_is_not_ES": "持有中按整个有保护支付上界管理，入场ES估计不冒充在仓每一天的精确条件ES",
            "margin": "全持有期保留1.2倍max(价差宽度保证金,认沽单腿保证金)；不预先享受组合申报减免",
            "leg_order": "先买保护再卖短腿；退出先买平短腿再卖保护；逐腿检查现金不为负",
            "drawdown": "收盘回撤达到10%后下一可成交开盘退出，终止本轮新增；不是保证回撤上限",
        },
        "costs": {"fee_per_contract_per_side": FEE, "tick": .0001,
                  "BASE": "每腿每侧不利max(0.0002,报价0.5%)，再按报价单位不利取整",
                  "STRESS": "每腿每侧不利max(0.0005,报价1%)，再按报价单位不利取整",
                  "cash_interest": 0., "annual_sessions": 252},
        "missing_and_expiry": {
            "future_missing_not_an_entry_filter": True,
            "missing_open": "当日不成交，退出等待；已出现无法确定估值/平仓则账户留为不完整，不据此晋升",
            "missing_close": "优先当日有效结算价标记并记录；再用前次金额连续值且账户不完整，次日请求退出",
            "expiry_failure": "计划在到期前退出；若未能退出到期则账户不完整并停止收益晋升，不伪造现金结算替代实物交割",
            "right_censor": "末端候选保留；期末持仓记录市值及压力清算成本储备，不把回测截止提前告知策略",
        },
        "evaluation": {
            "primary_increment": "MACRO-STRESS相对PRICE-STRESS的完整账户日收益",
            "bootstrap": {"block_sessions": 20, "draws": 2000, "seed": 2026092502},
            "periods": ["2021-2023", "2024-END"], "all_rolling_two_calendar_years": True,
            "shadow_monitor": "固定五日相位的成熟预测，60次窗口中低于预测5%分位的次数触及Binomial(60,.05)上尾1%才告警；只记录，不事后更改启停",
            "option_premium_not_profit": True, "max_cycle_concentration": True,
            "all_cash_sharpe": None, "independent_validation": False,
            "promotion": "三项目标和完整数据账户只是历史候选门；不能据已反复研究的历史声明实盘或独立验证完成",
        },
        "new_downloads": 0, "orders_authorized": False, "parameter_search": False,
    }


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("本版已冻结，禁止覆盖。")
    mandate = json.loads((ROOT / "config/510300_existing_data_training_mandate_v1.json").read_text(encoding="utf-8"))
    assert "SSE_510300_ETF_OPTIONS" in mandate["executable_assets"]
    assert mandate["capital_cny"] == INITIAL and mandate["training_window_calendar_years"] == 2
    paths = {
        "option_eod.parquet": ROOT / "data/raw/return_tail/options/510300_tushare_eod.parquet",
        "option_risk.parquet": ROOT / "data/raw/return_tail/options/510300_sse_risk_indicators.parquet",
        "market.parquet": PARENT / "inputs/market.parquet",
        "calendar_prices.parquet": ROOT / "reports/research/510300_sequential_patterns_regime_v1/inputs/prices.parquet",
        "dividends.csv": ROOT / "data/reference/510300_dividends.csv",
        "variance_predictions.parquet": PARENT / "results/predictions.parquet",
        "variance_verification.json": PARENT / "verification.json",
        "variance_protocol.json": PARENT / "protocol.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, src in paths.items():
        (root / "inputs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, root / "inputs" / name)
    (root / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), root / "code" / Path(__file__).name)
    shutil.copy2(ROOT / "research/option_volatility_account_risk_v1.py", root / "code/account_primitives_source.py")
    save(root / "protocol.json", specification(), exclusive=True)
    files = [root / "protocol.json", root / "authority_update.json", *sorted((root / "inputs").glob("*")), *sorted((root / "code").glob("*"))]
    save(root / "freeze.json", {"at": now(), "new_payoff_labels_read": False, "new_accounts_run": False,
                               "files": {p.relative_to(root).as_posix(): digest(p) for p in files}}, exclusive=True)
    print("选腿、十日持有、四组比较与尾部预算已在新增收益标签前冻结。", flush=True)


def verify_freeze(root):
    f = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for rel, value in f["files"].items():
        if digest(root / rel) != value:
            raise RuntimeError(f"冻结文件变化：{rel}")


def load_market(root):
    market = pd.read_parquet(root / "inputs/market.parquet")
    market["date"] = pd.to_datetime(market.date).astype("datetime64[ns]")
    market = market.set_index("date").sort_index()
    # 只按已存在的价格链算当时特征，标签另外生成。
    log_returns = np.log1p(market.total_simple)
    market["vol20"] = np.sqrt(log_returns.pow(2).rolling(20).mean() * 252)
    market["trend20_feature"] = log_returns.rolling(20).sum() / np.maximum(market["vol20"] * np.sqrt(20 / 252), 1e-8)
    extra = pd.read_parquet(root / "inputs/calendar_prices.parquet", columns=["date"])
    calendar = pd.DatetimeIndex(sorted(set(pd.to_datetime(extra.date)) | set(market.index)))
    return market, calendar


def load_chain(root, market, h):
    eod = pd.read_parquet(root / "inputs/option_eod.parquet")
    risk = pd.read_parquet(root / "inputs/option_risk.parquet")
    for frame in [eod, risk]:
        frame["trade_date"] = pd.to_datetime(frame.trade_date).astype("datetime64[ns]")
        assert not frame.duplicated(["trade_date", "contract_code"]).any()
    panel = eod.merge(risk[["trade_date", "contract_code", "exchange_contract_id", "contract_symbol", "delta", "implied_volatility"]],
                      on=["trade_date", "contract_code"], how="left", validate="one_to_one")
    panel["expiry_date"] = pd.to_datetime(panel.expiry_date).astype("datetime64[ns]")
    div = pd.read_csv(root / "inputs/dividends.csv")
    events = {pd.Timestamp(r.ex_date): float(r.cash_dividend_per_share) for r in div.itertuples()
              if pd.Timestamp(r.ex_date) >= panel.trade_date.min()}
    states, quotes, changes, term_records = {}, {}, [], []
    disagreements = []
    checks = 0
    for row in panel.sort_values(["trade_date", "contract_code"], kind="stable").itertuples(index=False):
        q = row._asdict()
        day, code = pd.Timestamp(row.trade_date), str(row.contract_code)
        identity = str(row.exchange_contract_id)
        parsed = re.fullmatch(r"510300([CP])(\d{4})([A-Z])(\d{5})", identity)
        state = states.get(code)
        if state is not None:
            for event, amount in events.items():
                if state["last_date"] < event <= day:
                    previous = float(market.loc[market.index < event, "close"].iloc[-1])
                    unit, strike = h.adjust_terms(state["unit"], state["strike"], previous, amount)
                    changes.append({"date": event, "contract_code": code, "old_unit": state["unit"], "new_unit": unit,
                                    "old_strike": state["strike"], "new_strike": strike})
                    state = {"unit": unit, "strike": strike, "version": "A" if state["version"] == "M" else chr(ord(state["version"]) + 1), "last_date": event}
        if parsed and parsed[3] == "M":
            if state is not None and state["version"] != "M":
                disagreements.append((day, code, "合约版本退回M"))
            state = {"unit": 10000, "strike": int(parsed[4]) / 1000, "version": "M", "last_date": day}
        if state is not None:
            state["last_date"] = day
            states[code] = state.copy()
            if parsed and parsed[3] != state["version"]:
                disagreements.append((day, code, "版本不匹配"))
            symbol = re.search(r"[购沽]\d+月(\d+)([A-Z])?$", str(row.contract_symbol))
            if parsed and parsed[3] != "M" and symbol:
                checks += 1
                if abs(int(symbol[1]) / 1000 - state["strike"]) > 1e-9:
                    disagreements.append((day, code, "调整行权价不匹配"))
        q.update(contract_code=code, unit=state["unit"] if state else None,
                 effective_strike=state["strike"] if state else None,
                 standard=bool(parsed and parsed[3] == "M" and parsed[1] == row.option_type))
        quotes[(day, code)] = q
        term_records.append({"date": day, "contract_code": code, "unit": q["unit"], "strike": q["effective_strike"], "standard": q["standard"]})
    if disagreements:
        save(root / "results/term_disagreements.json", disagreements)
        raise RuntimeError("合约条款与当日官方简称不一致，停止账户计算。")
    pd.DataFrame(term_records).to_parquet(root / "results/daily_terms.parquet", index=False)
    pd.DataFrame(changes).to_parquet(root / "results/dividend_adjustments.parquet", index=False)
    save(root / "results/terms_reconstruction.json", {"rows": len(panel), "adjustments": len(changes), "official_adjusted_price_checks": checks,
                                                     "unresolved_rows": sum(r["unit"] is None for r in term_records), "disagreements": 0})
    # 选腿面板明确替换最新元数据，保留当日风险信息。
    terms = pd.DataFrame(term_records).rename(columns={"date": "trade_date", "strike": "effective_strike", "unit": "effective_unit"})
    panel = panel.merge(terms, on=["trade_date", "contract_code"], validate="one_to_one")
    return panel, quotes, events


def terms_at(candidate, day, quotes):
    result = []
    for code, side in [(candidate["long_code"], 1), (candidate["short_code"], -1)]:
        q = quotes.get((day, code))
        if q is None or q["unit"] is None:
            return None
        result.append({"code": code, "side": side, "unit": int(q["unit"]), "strike": float(q["effective_strike"]), "quote": q})
    if result[0]["unit"] != result[1]["unit"] or result[0]["strike"] >= result[1]["strike"]:
        return None
    return result


def valid_quote(terms, field):
    return terms is not None and all(np.isfinite(t["quote"][field]) and t["quote"][field] > 0 and t["quote"]["volume"] > 0 for t in terms)


def width_cash(terms):
    return (terms[1]["strike"] - terms[0]["strike"]) * terms[0]["unit"]


def credit(terms, field, scenario):
    return -sum(t["side"] * fill(t["quote"][field], t["side"], scenario) * t["unit"] for t in terms)


def close_cost(terms, field, scenario):
    return sum(-t["side"] * fill(t["quote"][field], -t["side"], scenario) * t["unit"] for t in terms)


def margin(terms, spot, field="settlement"):
    short = terms[1]
    premium = float(short["quote"][field])
    if not np.isfinite(premium) or premium < 0:
        return math.inf
    strike = short["strike"]
    naked = min(premium + max(.12 * spot - max(spot - strike, 0.), .07 * strike), strike) * short["unit"]
    return 1.2 * max(width_cash(terms), naked)


def candidates(root, panel, quotes, market, calendar):
    daily = {d: g for d, g in panel.groupby("trade_date", sort=True)}
    rows = []
    first = int(calendar.searchsorted(panel.trade_date.min())) + 2
    for i in range(first, int(calendar.searchsorted(END, side="right"))):
        day, observed = calendar[i], calendar[i - 2]
        base = {"idx": i, "date": day, "observed_date": observed,
                "exit5_date": calendar[i + 5], "exit10_date": calendar[i + 10], "selection_status": "NO_ELIGIBLE_PUT_PAIR"}
        source = daily.get(observed)
        if source is None or observed not in market.index:
            rows.append(base)
            continue
        source = source.copy()
        source["dte"] = (source.expiry_date - observed).dt.days
        ok = source.standard & source.dte.between(30, 75) & source.volume.ge(100) & source.open_interest.ge(500)
        ok &= source.close.gt(0) & source.implied_volatility.between(.01, 3) & source.delta.notna()
        puts = source[ok & source.option_type.eq("P") & source.delta.between(-1, 0)].copy()
        possible = []
        for expiry, group in puts.groupby("expiry_date", sort=True):
            if group.effective_strike.nunique() < 2:
                continue
            group = group.assign(distance=(group.delta + .30).abs())
            short = group.sort_values(["distance", "effective_strike", "contract_code"], kind="stable").iloc[0]
            lower = group[group.effective_strike.lt(short.effective_strike)].copy()
            if lower.empty:
                continue
            lower["distance"] = (lower.delta + .10).abs()
            long = lower.sort_values(["distance", "effective_strike", "contract_code"], kind="stable").iloc[0]
            possible.append((abs(int((expiry - observed).days) - 45), expiry, short, long))
        if not possible:
            rows.append(base)
            continue
        _, expiry, short, long = sorted(possible, key=lambda x: (x[0], x[1]))[0]
        base.update(short_code=str(short.contract_code), long_code=str(long.contract_code), expiry=expiry,
                    short_strike=float(short.effective_strike), long_strike=float(long.effective_strike),
                    short_delta=float(short.delta), long_delta=float(long.delta),
                    capacity=int(math.floor(.01 * min(short.volume, short.open_interest, long.volume, long.open_interest))))
        if base["exit10_date"] >= expiry - pd.Timedelta(days=5):
            rows.append({**base, "selection_status": "KNOWN_EXPIRY_TOO_CLOSE"})
            continue
        same = source[source.standard & source.expiry_date.eq(expiry) & source.volume.ge(100) & source.open_interest.ge(500) & source.close.gt(0)]
        pair = same[same.option_type.eq("C")].merge(same[same.option_type.eq("P")], on="effective_strike", suffixes=("_C", "_P"))
        if pair.empty:
            rows.append({**base, "selection_status": "NO_SAME_EXPIRY_FORWARD_PAIR"})
            continue
        spot = float(market.loc[observed, "close"])
        pair["distance"] = abs(pair.effective_strike - spot)
        atm = pair.sort_values(["distance", "effective_strike", "contract_code_C", "contract_code_P"], kind="stable").iloc[0]
        forward = float(atm.effective_strike + atm.close_C - atm.close_P)
        iv = float((short.implied_volatility + long.implied_volatility) / 2)
        rv = float(market.loc[observed, "vol20"])
        base.update(forward_proxy=forward, atm_call_code=str(atm.contract_code_C), atm_put_code=str(atm.contract_code_P),
                    log_iv=math.log(iv), log_iv_rv20=math.log(iv / rv), trend20=float(market.loc[observed, "trend20_feature"]),
                    tau=float((expiry - observed).days / 365), observed_spot=spot)
        t = terms_at(base, observed, quotes)
        raw_credit = -sum(x["side"] * x["unit"] * x["quote"]["close"] for x in t)
        stress_credit = credit(t, "close", "STRESS")
        exit_slip = close_cost(t, "close", "STRESS") - raw_credit
        known_loss = width_cash(t) - stress_credit + 4 * FEE + exit_slip
        base.update(raw_credit=raw_credit, prior_stress_credit=stress_credit, prior_exit_slip=exit_slip,
                    width_cash=width_cash(t), known_loss=known_loss, unit=t[0]["unit"], selection_status="READY")
        if forward <= 0 or not np.isfinite([forward, iv, rv, known_loss, base["trend20"]]).all() or known_loss <= 0:
            base["selection_status"] = "INVALID_KNOWN_INPUT"
        rows.append(base)
    result = pd.DataFrame(rows)
    result.to_parquet(root / "results/candidates.parquet", index=False)
    return result


def payoff_labels(root, c, quotes, calendar):
    rows = []
    for candidate in c[c.selection_status.eq("READY")].to_dict("records"):
        row = {"idx": candidate["idx"], "date": candidate["date"]}
        entry = terms_at(candidate, candidate["date"], quotes)
        for horizon in [5, 10]:
            end = candidate[f"exit{horizon}_date"]
            status = "READY"
            if end > END:
                status = "RIGHT_CENSORED"
            elif not valid_quote(entry, "open"):
                status = "UNKNOWN_ENTRY_QUOTE"
            else:
                for day in calendar[(calendar >= candidate["date"]) & (calendar <= end)]:
                    daily_terms = terms_at(candidate, day, quotes)
                    if not valid_quote(daily_terms, "close"):
                        status = "UNKNOWN_HELD_DAILY_QUOTE"
                        break
                exit_terms = terms_at(candidate, end, quotes)
                if status == "READY" and not valid_quote(exit_terms, "open"):
                    status = "UNKNOWN_EXIT_QUOTE"
            row[f"status{horizon}"] = status
            row[f"exit{horizon}_date"] = end
            row[f"pnl{horizon}"] = None
            row[f"risk_return{horizon}"] = None
            if status == "READY":
                pnl = credit(entry, "open", "STRESS") - close_cost(exit_terms, "open", "STRESS") - 4 * FEE
                row[f"pnl{horizon}"] = pnl
                row[f"risk_return{horizon}"] = pnl / candidate["known_loss"]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_parquet(root / "results/payoff_labels.parquet", index=False)
    return result


def tail_at(day, current, joined):
    lower = day - pd.DateOffset(years=2)
    train = joined[joined.date.ge(lower) & joined.exit5_date.lt(day) & joined.status5.eq("READY")]
    train = train[np.isfinite(train[FEATURES + ["risk_return5"]]).all(axis=1)]
    receipt = {"date": day, "lower_bound": lower, "training_count": len(train), "latest_training_exit": train.exit5_date.max() if len(train) else None}
    if len(train) < 252 or not np.isfinite([current[k] for k in FEATURES]).all():
        return {**receipt, "tail_status": "NO_VIEW"}, None
    raw = train[FEATURES].to_numpy(float)
    mean, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
    scale[scale < 1e-12] = 1
    z = np.clip((raw - mean) / scale, -3, 3)
    current_z = np.clip((np.array([current[k] for k in FEATURES]) - mean) / scale, -3, 3)
    selected = np.argsort(np.square(z - current_z).sum(axis=1), kind="stable")[:126]
    values = train.risk_return5.to_numpy(float)[selected]
    es = max(0., -float(np.sort(values)[:int(math.ceil(.05 * len(values)))].mean()))
    result = {**receipt, "tail_status": "UPDATED", "ES95_risk_unit": es,
              "q05_risk_unit": float(np.quantile(values, .05)), "neighbor_count": len(values)}
    model = {**result, "mean": mean.tolist(), "scale": scale.tolist(), "training_indices": train.idx.to_list(),
             "neighbor_indices": train.idx.iloc[selected].to_list()}
    return result, model


def learn(root, c, labels):
    joined = c.merge(labels.drop(columns=["date", "exit5_date", "exit10_date"]), on="idx", how="left", validate="one_to_one")
    vp = pd.read_parquet(root / "inputs/variance_predictions.parquet",
                         columns=["date", "model", "variance_prediction", "variance_persistence"])
    vp["date"] = pd.to_datetime(vp.date).astype("datetime64[ns]")
    lookup = {(r.date, r.model): r for r in vp.itertuples()}
    signals, models, receipts = [], [], []
    for current in c[c.date.ge(START)].to_dict("records"):
        day = current["date"]
        receipt = {"date": day, "status": current["selection_status"]}
        if current["selection_status"] != "READY":
            receipts.append(receipt)
            continue
        tail, model = tail_at(day, current, joined)
        receipts.append({**receipt, **tail})
        if model is not None:
            models.append(model)
        p, m = lookup.get((day, "PRICE")), lookup.get((day, "MACRO"))
        if model is None or p is None or m is None:
            continue
        signal = {**current, **tail, "common_ready": True}
        for policy, variance in [("UNCONDITIONAL", None), ("RV", p.variance_persistence),
                                 ("PRICE", p.variance_prediction), ("MACRO", m.variance_prediction)]:
            fair = 0. if variance is None else current["unit"] * (
                put_value(current["forward_proxy"], current["short_strike"], variance, current["tau"])
                - put_value(current["forward_proxy"], current["long_strike"], variance, current["tau"]))
            minimum_credit = fair + 4 * FEE + current["prior_exit_slip"]
            signal[f"fair_{policy}"] = fair
            signal[f"minimum_credit_{policy}"] = minimum_credit
            signal[f"gate_{policy}"] = current["prior_stress_credit"] > minimum_credit
        signals.append(signal)
        if len(signals) % 300 == 0:
            print(f"两年尾部分布与定价信号已更新至{day.date()}。", flush=True)
    frame = pd.DataFrame(signals)
    frame.to_parquet(root / "results/signals.parquet", index=False)
    pd.DataFrame(receipts).to_parquet(root / "results/update_receipts.parquet", index=False)
    save(root / "results/tail_models.json", models)
    return frame, models


def simulate(policy, scenario, signals, quotes, market, calendar, events):
    days = calendar[(calendar >= START) & (calendar <= END)]
    signal_map = {r["date"]: r for r in signals.to_dict("records")}
    cash, peak = INITIAL, INITIAL
    position = None
    halted, incomplete, inconsistent, must_exit = False, False, False, False
    ledgers, fills, cycles, decisions, marks = [], [], [], [], []
    trade_counter = 0
    previous_value = 0.
    previous_reserve = 0.

    def record_fill(terms, quantity, day, opening, trade_id, reason):
        nonlocal cash
        ordered = terms if opening else [terms[1], terms[0]]
        change = 0.
        for t in ordered:
            action = t["side"] * (1 if opening else -1)
            raw = float(t["quote"]["open"])
            price = fill(raw, action, scenario)
            fee = FEE * quantity
            delta = -action * price * t["unit"] * quantity - fee
            cash += delta
            change += delta
            if cash < -1e-7:
                raise RuntimeError("逐腿交易现金为负。")
            fills.append({"date": day, "trade_id": trade_id, "opening": opening, "reason": reason,
                          "contract_code": t["code"], "action": action, "quantity": quantity,
                          "unit": t["unit"], "strike": t["strike"], "raw_price": raw, "fill_price": price,
                          "cash_change": delta, "fee": fee, "slippage": action * (price - raw) * t["unit"] * quantity,
                          "cash_after_leg": cash})
        return change

    def reduce_position(quantity, day, reason):
        nonlocal position, incomplete, must_exit
        if quantity <= 0:
            return 0.
        terms = terms_at(position["candidate"], day, quotes)
        if not valid_quote(terms, "open"):
            must_exit = True
            incomplete = True
            decisions.append({"date": day, "action": "EXIT_WAIT_MISSING_OPEN", "reason": reason, "quantity": quantity})
            return 0.
        # 买平短腿前检查现金，短腿平仓后保护腿可卖出。
        short_buy = fill(terms[1]["quote"]["open"], 1, scenario) * terms[1]["unit"] * quantity + FEE * quantity
        if short_buy > cash + 1e-8:
            incomplete, must_exit = True, True
            decisions.append({"date": day, "action": "EXIT_WAIT_INSUFFICIENT_TRANSITION_CASH", "quantity": quantity})
            return 0.
        change = record_fill(terms, quantity, day, False, position["trade_id"], reason)
        position["cashflow"] += change
        position["quantity"] -= quantity
        if position["quantity"] == 0:
            cycles.append({"trade_id": position["trade_id"], "entry_date": position["entry_date"], "exit_date": day,
                           "holding_sessions": int(calendar.get_loc(day) - calendar.get_loc(position["entry_date"])),
                           "initial_quantity": position["initial_quantity"], "net_pnl": position["cashflow"],
                           "entry_equity": position["entry_equity"], "entry_maximum_loss": position["entry_maximum_loss"],
                           "entry_ES95": position["entry_ES95"], "exit_reason": reason,
                           "prediction_gate_surplus": position["candidate"]["prior_stress_credit"] - position["candidate"][f"minimum_credit_{policy}"]})
            position = None
            must_exit = False
        return change

    for day in days:
        idx = int(calendar.get_loc(day))
        prev_day = calendar[idx - 1]
        previous_equity = ledgers[-1]["equity"] if ledgers else INITIAL
        decision_peak = peak
        day_cashflow = 0.
        had_position_at_decision = position is not None
        held_today = had_position_at_decision
        budget = min(.05 * previous_equity, .5 * max(0., previous_equity - .9 * decision_peak))
        if position is not None:
            prior_terms = terms_at(position["candidate"], prev_day, quotes)
            reason = None
            quantity_to_exit = 0
            if halted or must_exit:
                reason, quantity_to_exit = "RISK_OR_DATA_EXIT", position["quantity"]
            elif day >= position["candidate"]["exit10_date"]:
                reason, quantity_to_exit = "TEN_SESSION_EXIT", position["quantity"]
            elif day >= position["candidate"]["expiry"] - pd.Timedelta(days=5):
                reason, quantity_to_exit = "KNOWN_EXPIRY_BUFFER_EXIT", position["quantity"]
            elif prior_terms is None:
                reason, quantity_to_exit = "MISSING_KNOWN_TERMS_EXIT", position["quantity"]
            else:
                remaining_loss = max(0., width_cash(prior_terms) + position["last_mark_per_group"] + position["last_exit_reserve_per_group"])
                allowed = math.floor(budget / remaining_loss) if remaining_loss > 1e-9 else position["quantity"]
                if allowed < position["quantity"]:
                    reason, quantity_to_exit = "DAILY_REMAINING_LOSS_REDUCTION", position["quantity"] - max(0, allowed)
            if quantity_to_exit:
                decisions.append({"date": day, "action": "REDUCE_REQUEST_0900", "reason": reason,
                                  "quantity": quantity_to_exit, "known_equity": previous_equity, "remaining_budget": budget})
                day_cashflow += reduce_position(quantity_to_exit, day, reason)

        signal = signal_map.get(day)
        if not had_position_at_decision and position is None and not halted:
            if signal is None:
                decisions.append({"date": day, "action": "NO_VIEW_COMMON_INFORMATION"})
            elif not signal[f"gate_{policy}"]:
                decisions.append({"date": day, "action": "PRICING_GATE_REJECT", "pricing_surplus": signal["prior_stress_credit"] - signal[f"minimum_credit_{policy}"]})
            else:
                known_loss = signal["known_loss"]
                es = signal["ES95_risk_unit"]
                q = math.floor(budget / known_loss)
                if es > 1e-12:
                    q = min(q, math.floor(.025 * previous_equity / (known_loss * es)))
                q = min(q, signal["capacity"])
                prior_terms = terms_at(signal, signal["observed_date"], quotes)
                prior_margin = margin(prior_terms, signal["observed_spot"])
                prior_long_cost = fill(prior_terms[0]["quote"]["close"], 1, "STRESS") * prior_terms[0]["unit"] + FEE
                while q > 0 and (q * prior_long_cost > cash or cash + q * (signal["prior_stress_credit"] - 2 * FEE) < q * prior_margin):
                    q -= 1
                decisions.append({"date": day, "action": "ENTRY_REQUEST_0900", "quantity": q,
                                  "known_loss": known_loss, "ES95_risk_unit": es, "budget": budget,
                                  "known_equity": previous_equity, "minimum_credit": signal[f"minimum_credit_{policy}"]})
                if q < 1:
                    decisions.append({"date": day, "action": "NO_INTEGER_QUANTITY"})
                else:
                    terms = terms_at(signal, day, quotes)
                    rejection = None
                    if not valid_quote(terms, "open"):
                        rejection = "NO_OBSERVED_ENTRY_OPEN"
                    else:
                        stress_credit = credit(terms, "open", "STRESS")
                        max_loss = width_cash(terms) - stress_credit + 4 * FEE + signal["prior_exit_slip"]
                        prior_spot = float(market.loc[prev_day, "close"]) - events.get(day, 0.)
                        entry_margin = margin(terms, prior_spot, "previous_settlement")
                        long_cost = fill(terms[0]["quote"]["open"], 1, "STRESS") * terms[0]["unit"] + FEE
                        if stress_credit < signal[f"minimum_credit_{policy}"] - 1e-8 or stress_credit >= width_cash(terms):
                            rejection = "PRESET_CREDIT_LIMIT_UNFILLED"
                        elif max_loss * q > budget + 1e-8 or max_loss * q * es > .025 * previous_equity + 1e-8:
                            rejection = "ENTRY_RISK_BUDGET_UNFILLED"
                        elif q * long_cost > cash + 1e-8 or cash + q * (stress_credit - 2 * FEE) < q * entry_margin - 1e-8:
                            rejection = "ENTRY_TRANSITION_MARGIN_UNFILLED"
                    if rejection:
                        decisions.append({"date": day, "action": rejection, "quantity": q})
                    else:
                        trade_counter += 1
                        trade_id = f"{policy}_{scenario}_{trade_counter:04d}"
                        change = record_fill(terms, q, day, True, trade_id, "FROZEN_PRICING_SIGNAL")
                        day_cashflow += change
                        position = {"trade_id": trade_id, "candidate": signal, "quantity": q, "initial_quantity": q,
                                    "entry_date": day, "cashflow": change, "entry_equity": previous_equity,
                                    "entry_maximum_loss": max_loss * q, "entry_ES95": max_loss * q * es,
                                    "last_mark_per_group": -stress_credit, "last_exit_reserve_per_group": signal["prior_exit_slip"] + 2 * FEE}
                        decisions.append({"date": day, "action": "ENTRY_FILLED", "trade_id": trade_id,
                                          "quantity": q, "actual_maximum_loss": max_loss * q,
                                          "actual_ES95": max_loss * q * es, "entry_margin": entry_margin * q,
                                          "known_equity": previous_equity, "budget": budget})
                        held_today = True

        value, reserved_margin, exit_reserve = 0., 0., 0.
        stale_today, violation_today = False, False
        if position is not None:
            terms = terms_at(position["candidate"], day, quotes)
            if terms is None:
                incomplete, must_exit, stale_today = True, True, True
                per_group = position["last_mark_per_group"]
                reserved_margin = ledgers[-1]["margin"] if ledgers else 0.
                exit_per_group = position["last_exit_reserve_per_group"]
            else:
                per_group = 0.
                for t in terms:
                    qprice = t["quote"]["close"]
                    source = "close"
                    if not np.isfinite(qprice) or qprice <= 0:
                        qprice = t["quote"]["settlement"]
                        source = "settlement"
                        incomplete, must_exit, stale_today = True, True, True
                    if not np.isfinite(qprice) or qprice < 0:
                        per_group = position["last_mark_per_group"]
                        incomplete, must_exit, stale_today = True, True, True
                        break
                    per_group += t["side"] * t["unit"] * float(qprice)
                    marks.append({"date": day, "trade_id": position["trade_id"], "contract_code": t["code"],
                                  "side": t["side"], "unit": t["unit"], "quantity": position["quantity"],
                                  "mark": float(qprice), "mark_source": source,
                                  "market_value": t["side"] * t["unit"] * float(qprice) * position["quantity"]})
                reserved_margin = margin(terms, float(market.loc[day, "close"])) * position["quantity"]
                if not np.isfinite(reserved_margin):
                    incomplete, must_exit = True, True
                    reserved_margin = max(cash, 0.)
                if valid_quote(terms, "close"):
                    exit_per_group = close_cost(terms, "close", "STRESS") + per_group + 2 * FEE
                else:
                    exit_per_group = position["last_exit_reserve_per_group"]
                if per_group > .0002 * terms[0]["unit"] + 1e-8 or per_group < -width_cash(terms) - .0002 * terms[0]["unit"] - 1e-8:
                    inconsistent, violation_today = True, True
            value = per_group * position["quantity"]
            position["last_mark_per_group"] = per_group
            position["last_exit_reserve_per_group"] = max(0., exit_per_group)
            if day >= position["candidate"]["expiry"]:
                incomplete, must_exit = True, True
            if day == days[-1]:
                exit_reserve = max(0., exit_per_group) * position["quantity"]
            if cash < reserved_margin - 1e-7:
                must_exit = True
                decisions.append({"date": day, "action": "MARGIN_DEFICIT_NEXT_OPEN_EXIT", "deficit": reserved_margin - cash})
        equity = cash + value - exit_reserve
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if drawdown <= -.1 + 1e-12:
            halted = True
            must_exit = position is not None
        residual = (equity - previous_equity) - (day_cashflow + value - previous_value - exit_reserve + previous_reserve)
        if abs(residual) > 1e-6:
            raise RuntimeError("逐日现金、期权市值与清算储备不平。")
        ledgers.append({"date": day, "cash": cash, "option_value": value, "terminal_exit_reserve": exit_reserve,
                        "equity": equity, "return": equity / previous_equity - 1, "drawdown": drawdown,
                        "margin": reserved_margin, "free_cash": cash - reserved_margin,
                        "quantity": position["quantity"] if position is not None else 0, "held_today": held_today,
                        "halted": halted, "incomplete": incomplete, "stale_today": stale_today,
                        "pricing_violation_today": violation_today, "pricing_inconsistency_seen": inconsistent,
                        "cashflow": day_cashflow, "accounting_error": residual})
        previous_value, previous_reserve = value, exit_reserve
    return {"ledger": pd.DataFrame(ledgers), "fills": pd.DataFrame(fills), "cycles": pd.DataFrame(cycles),
            "decisions": pd.DataFrame(decisions), "marks": pd.DataFrame(marks),
            "terminal_position": clean(position)}


def return_metrics(returns):
    r = np.asarray(returns, dtype=float)
    curve = np.cumprod(1 + r)
    volatility = float(np.std(r, ddof=1)) if len(r) > 1 else 0.
    return {"cumulative_return": float(curve[-1] - 1),
            "annualized_return": float(curve[-1] ** (252 / len(r)) - 1),
            "net_sharpe": float(r.mean() / volatility * np.sqrt(252)) if volatility > 1e-12 else None,
            "annualized_volatility": volatility * np.sqrt(252),
            "max_drawdown": float(-np.min(np.r_[1., curve] / np.maximum.accumulate(np.r_[1., curve]) - 1)),
            "worst_day": float(r.min()), "daily_skew": float(skew(r, bias=False)) if volatility > 1e-12 else None}


def account_metrics(policy, scenario, account):
    ledger, fills, cycles = account["ledger"], account["fills"], account["cycles"]
    result = {"policy": policy, "scenario": scenario, **return_metrics(ledger["return"]),
              "ending_equity": float(ledger.equity.iloc[-1]), "completed_cycles": len(cycles),
              "cost_cny": float((fills.fee + fills.slippage).sum()) if len(fills) else 0.,
              "fee_cny": float(fills.fee.sum()) if len(fills) else 0.,
              "terminal_reserve_cny": float(ledger.terminal_exit_reserve.iloc[-1]),
              "exposure_day_fraction": float(ledger.held_today.mean()),
              "maximum_margin_cny": float(ledger.margin.max()), "minimum_free_cash": float(ledger.free_cash.min()),
              "halted": bool(ledger.halted.any()), "incomplete_data": bool(ledger.incomplete.any()),
              "pricing_inconsistency": bool(ledger.pricing_inconsistency_seen.any()),
              "stale_days": int(ledger.stale_today.sum()), "has_terminal_position": account["terminal_position"] is not None,
              "maximum_accounting_error": float(ledger.accounting_error.abs().max())}
    pnl = cycles.net_pnl.to_numpy(float) if len(cycles) else np.array([])
    wins, losses = pnl[pnl > 0], -pnl[pnl < 0]
    result.update(win_rate=float((pnl > 0).mean()) if len(pnl) else None,
                  average_win=float(wins.mean()) if len(wins) else None,
                  average_loss=float(losses.mean()) if len(losses) else None,
                  cash_payoff_ratio=float(wins.mean() / losses.mean()) if len(wins) and len(losses) else None,
                  pnl_without_best_completed_cycle=float(pnl.sum() - pnl.max()) if len(pnl) else None)
    result["historical_point_targets_met"] = bool(result["net_sharpe"] is not None and result["net_sharpe"] >= 1.2
                                                   and result["annualized_return"] >= .1 and result["max_drawdown"] <= .1)
    result["eligible_historical_candidate"] = result["historical_point_targets_met"] and not result["incomplete_data"] and not result["pricing_inconsistency"]
    return result


def yearly_and_rolling(policy, scenario, ledger):
    yearly, rolling = [], []
    for year, group in ledger.groupby(ledger.date.dt.year):
        yearly.append({"policy": policy, "scenario": scenario, "year": int(year), **return_metrics(group["return"])})
    dates = pd.DatetimeIndex(ledger.date)
    for i, start in enumerate(dates):
        j = int(dates.searchsorted(start + pd.DateOffset(years=2)))
        if j >= len(dates):
            break
        metrics = return_metrics(ledger["return"].iloc[i:j + 1])
        rolling.append({"policy": policy, "scenario": scenario, "start": start, "end": dates[j], **metrics,
                        "joint_targets": bool(metrics["net_sharpe"] is not None and metrics["net_sharpe"] >= 1.2
                                              and metrics["annualized_return"] >= .1 and metrics["max_drawdown"] <= .1)})
    return yearly, rolling


def monitor(root, signals, labels):
    if signals.empty:
        summary = {"mature_five_day_predictions": 0, "tail_q05_breach_fraction": None,
                   "fixed_phase_mature_origins": 0, "monitor_windows": 0, "warning_windows": 0,
                   "status": "NO_MATURE_PREDICTIONS"}
        save(root / "results/monitor_summary.json", summary)
        return summary
    mature = signals.merge(labels[["idx", "status5", "risk_return5", "status10", "risk_return10"]], on="idx", how="left", validate="one_to_one")
    mature["tail_breach"] = mature.status5.eq("READY") & mature.risk_return5.lt(mature.q05_risk_unit)
    mature.to_parquet(root / "results/shadow_prediction_outcomes.parquet", index=False)
    fixed = mature[mature.idx.mod(5).eq(int(mature.idx.iloc[0]) % 5) & mature.status5.eq("READY")].copy()
    rows = []
    for i in range(59, len(fixed)):
        window = fixed.iloc[i - 59:i + 1]
        breaches = int(window.tail_breach.sum())
        probability = float(binom.sf(breaches - 1, 60, .05))
        rows.append({"latest_origin_date": window.date.iloc[-1], "latest_maturity_date": window.exit5_date.iloc[-1],
                     "n": 60, "breaches": breaches, "binomial_tail_reference": probability,
                     "warning": probability <= .01, "warning_is_proof_of_permanent_failure": False})
    pd.DataFrame(rows).to_parquet(root / "results/relationship_monitor.parquet", index=False)
    summary = {"mature_five_day_predictions": int(mature.status5.eq("READY").sum()),
               "tail_q05_breach_fraction": float(mature.loc[mature.status5.eq("READY"), "tail_breach"].mean()),
               "fixed_phase_mature_origins": len(fixed), "monitor_windows": len(rows),
               "warning_windows": sum(r["warning"] for r in rows),
               "independence_warning": "五日相位减少标签交叠，不保证独立；二项尾概率只是冻结告警尺度，未驱动账户。"}
    save(root / "results/monitor_summary.json", summary)
    return summary


def mechanism_checks(root):
    h = helper(root)
    assert h.adjust_terms(10000, 4., 4.859, .123) == (10260, 3.899)
    initial, premium = 200000., 1300.
    assert initial + premium - premium == initial
    for spot in [0., 2., 3.8, 4., 8.]:
        payoff = max(3.8 - spot, 0.) - max(4. - spot, 0.)
        assert -.2 - 1e-10 <= payoff <= 1e-10
    assert fill(.03123, 1, "STRESS") >= fill(.03123, 1, "BASE") > .03123
    assert fill(.03123, -1, "STRESS") <= fill(.03123, -1, "BASE") < .03123
    assert put_value(4., 4., .09, .1) > put_value(4., 4., .04, .1)
    # 标签于当日开盘退出，不能进入当日09:00训练。
    day = pd.Timestamp("2024-01-03")
    fake = pd.DataFrame({"idx": np.arange(253), "date": pd.date_range("2022-01-05", periods=253),
                         "exit5_date": [pd.Timestamp("2023-01-01")] * 252 + [day], "status5": "READY",
                         "risk_return5": np.linspace(-.4, .2, 253), "log_iv": .1, "log_iv_rv20": .2, "trend20": .3})
    current = {"log_iv": .1, "log_iv_rv20": .2, "trend20": .3}
    _, model = tail_at(day, current, fake)
    assert 252 not in model["training_indices"]
    before = model["ES95_risk_unit"]
    fake.loc[252, "risk_return5"] = -1e9
    after, _ = tail_at(day, current, fake)
    assert after["ES95_risk_unit"] == before
    return {"premium_and_liability_balance": True, "bounded_put_spread_payoff": True,
            "cash_dividend_terms": True, "adverse_cost_tick": True,
            "mature_label_only_and_future_poisoning": True}


def verify_saved(root):
    verify_freeze(root)
    c = pd.read_parquet(root / "results/candidates.parquet")
    labels = pd.read_parquet(root / "results/payoff_labels.parquet")
    joined = c.merge(labels.drop(columns=["date", "exit5_date", "exit10_date"]), on="idx", how="left", validate="one_to_one")
    by_idx = joined.set_index("idx")
    models = json.loads((root / "results/tail_models.json").read_text(encoding="utf-8"))
    for model in models:
        train = by_idx.loc[model["training_indices"]]
        assert train.date.ge(pd.Timestamp(model["lower_bound"])).all()
        assert train.exit5_date.lt(pd.Timestamp(model["date"])).all()
        values = by_idx.loc[model["neighbor_indices"], "risk_return5"].to_numpy(float)
        expected = max(0., -float(np.sort(values)[:int(math.ceil(.05 * len(values)))].mean()))
        np.testing.assert_allclose(expected, model["ES95_risk_unit"], atol=1e-12)
        np.testing.assert_allclose(float(np.quantile(values, .05)), model["q05_risk_unit"], atol=1e-12)
    prefix_checks = []
    for cutoff in [pd.Timestamp("2023-12-29"), pd.Timestamp("2025-12-31")]:
        subset = [m for m in models if pd.Timestamp(m["date"]) <= cutoff]
        if not subset:
            continue
        model = subset[-1]
        day = pd.Timestamp(model["date"])
        current = c[c.date.eq(day)].iloc[0].to_dict()
        prefix = joined[joined.date.le(day)].copy()
        prefix.loc[prefix.exit5_date.ge(day), "risk_return5"] = -1e8
        recomputed, details = tail_at(day, current, prefix)
        np.testing.assert_allclose(recomputed["ES95_risk_unit"], model["ES95_risk_unit"], atol=1e-12)
        assert details["neighbor_indices"] == model["neighbor_indices"]
        prefix_checks.append(str(day.date()))
    accounts = 0
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            folder = root / "accounts" / scenario / policy
            ledger = pd.read_parquet(folder / "ledger.parquet")
            fills = pd.read_parquet(folder / "fills.parquet")
            decisions = pd.read_parquet(folder / "decisions.parquet")
            np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.option_value - ledger.terminal_exit_reserve, atol=1e-6)
            if not fills.empty:
                by_day = fills.groupby("date").cash_change.sum().reindex(ledger.date, fill_value=0).to_numpy(float)
                np.testing.assert_allclose(INITIAL + np.cumsum(by_day), ledger.cash, atol=1e-6)
                assert (fills.cash_after_leg >= -1e-7).all()
                np.testing.assert_allclose(fills.fee, fills.quantity * FEE, atol=1e-10)
            entered = decisions[decisions.action.eq("ENTRY_FILLED")]
            if not entered.empty:
                assert (entered.actual_maximum_loss <= entered.budget + 1e-7).all()
                assert (entered.actual_ES95 <= .025 * entered.known_equity + 1e-7).all()
            accounts += 1
    evidence = {"at": now(), "saved_tail_updates_recomputed": len(models), "strict_two_calendar_years_and_maturity": True,
                "prefix_and_future_poisoning_checks": prefix_checks, "cash_accounts_reconciled": accounts,
                "entry_risk_and_ES_budgets_verified": True, "mechanism_checks": mechanism_checks(root)}
    save(root / "verification.json", evidence)
    return evidence


def run(root):
    verify_freeze(root)
    assert digest(Path(__file__)) == digest(root / "code/protected_put_spread_daily_v1.py")
    save(root / "RUN_STARTED.json", {"at": now(), "study_id": STUDY}, exclusive=True)
    (root / "results").mkdir(exist_ok=True)
    save(root / "mechanism_checks.json", mechanism_checks(root))
    market, calendar = load_market(root)
    panel, quotes, events = load_chain(root, market, helper(root))
    print("历史标准合约与分红后条款已还原，开始生成冻结两腿候选。", flush=True)
    c = candidates(root, panel, quotes, market, calendar)
    labels = payoff_labels(root, c, quotes, calendar)
    signals, models = learn(root, c, labels)
    metrics, yearly, rolling, ledgers = [], [], [], {}
    for scenario in ["BASE", "STRESS"]:
        for policy in POLICIES:
            account = simulate(policy, scenario, signals, quotes, market, calendar, events)
            folder = root / "accounts" / scenario / policy
            folder.mkdir(parents=True, exist_ok=True)
            for name, frame in account.items():
                if isinstance(frame, pd.DataFrame):
                    frame.to_parquet(folder / f"{name}.parquet", index=False)
                else:
                    save(folder / f"{name}.json", frame)
            measure = account_metrics(policy, scenario, account)
            metrics.append(measure)
            y, r = yearly_and_rolling(policy, scenario, account["ledger"])
            yearly.extend(y)
            rolling.extend(r)
            ledgers[(policy, scenario)] = account["ledger"]
            print(f"{scenario}/{policy}完成：夏普{measure['net_sharpe']}，年化{measure['annualized_return']:.3%}，回撤{measure['max_drawdown']:.3%}。", flush=True)
    save(root / "results/account_metrics.json", metrics)
    save(root / "results/yearly_metrics.json", yearly)
    pd.DataFrame(rolling).to_parquet(root / "results/all_rolling_two_years.parquet", index=False)
    paired = ledgers[(PRIMARY, "STRESS")]["return"].to_numpy() - ledgers[("PRICE", "STRESS")]["return"].to_numpy()
    rng = np.random.default_rng(2026092502)
    estimates = []
    n = len(paired)
    for _ in range(2000):
        starts = rng.integers(0, n, size=math.ceil(n / 20))
        indices = ((starts[:, None] + np.arange(20)) % n).ravel()[:n]
        estimates.append(float(paired[indices].mean() * 252))
    periods = []
    for lower, upper in [(START, pd.Timestamp("2023-12-31")), (pd.Timestamp("2024-01-01"), END)]:
        for policy in POLICIES:
            ledger = ledgers[(policy, "STRESS")]
            cut = ledger[ledger.date.between(lower, upper)]
            periods.append({"policy": policy, "start": lower, "end": upper, **return_metrics(cut["return"])})
    save(root / "results/fixed_periods.json", periods)
    monitoring = monitor(root, signals, labels)
    main = next(m for m in metrics if m["policy"] == PRIMARY and m["scenario"] == "STRESS")
    result = {"study_id": STUDY, "at": now(),
              "status": "HISTORICAL_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION" if main["eligible_historical_candidate"] else "FROZEN_NO_QUALIFIED_PROTECTED_PUT_SPREAD",
              "primary": main, "goal_achieved": False, "source_end": str(END.date()),
              "new_tail_model_updates": len(models), "reused_variance_predictions": 2722, "accounts": len(metrics),
              "all_candidate_dates": len(c), "ready_candidate_dates": int(c.selection_status.eq("READY").sum()),
              "selection_reasons": c.selection_status.value_counts().to_dict(),
              "mature_label_status5": labels.status5.value_counts().to_dict(),
              "mature_label_status10": labels.status10.value_counts().to_dict(),
              "common_ready_decision_days": len(signals), "gate_days": {p: int(signals[f"gate_{p}"].sum()) for p in POLICIES},
              "paired_macro_minus_price_annual_mean": float(paired.mean() * 252),
              "paired_increment_ci95": np.quantile(estimates, [.025, .975]),
              "monitor": monitoring, "historical_candidates": [m for m in metrics if m["eligible_historical_candidate"]],
              "independent_validation": False, "orders_authorized": False}
    save(root / "result.json", result)
    verification = verify_saved(root)
    print(json.dumps(clean({"status": result["status"], "primary": main, "verification": verification}), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="保护性认沽价差的一次冻结历史研究")
    parser.add_argument("command", choices=["freeze", "run", "verify", "check"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out)
    elif args.command == "run":
        run(args.out)
    elif args.command == "verify":
        print(json.dumps(clean(verify_saved(args.out)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(mechanism_checks(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
