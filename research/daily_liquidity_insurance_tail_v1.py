"""两年滚动、逐日更新的流动性承接分布研究；仅510300现货研究模拟。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
OUT = WORKSPACE / "reports/research/510300_daily_liquidity_insurance_tail_v1"
STUDY = "510300_DAILY_LIQUIDITY_INSURANCE_TAIL_V1"
PRICE = ["pressure5", "trend20", "log_rv5_rv60"]
MACRO = PRICE + ["credit_acceleration3_pp", "funding_gap_pp", "log_iv_rv20"]
MODELS = {"HISTORY": [], "PRICE": PRICE, "MACRO": MACRO}
COSTS = {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005},
         "STRESS": {"commission": .0004, "minimum": 5., "slippage": .001}}
START, END = "2021-01-04", "2026-08-14"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def engine(root):
    name = "frozen_liquidity_insurance_account"
    spec = importlib.util.spec_from_file_location(name, root / "code/account_engine.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def freeze(root):
    if (root / "freeze.json").exists():
        raise RuntimeError("本轮已经冻结，禁止重复登记或覆盖。")
    for folder in ["inputs", "code", "results", "accounts"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    sources = {
        "prices.parquet": "reports/research/510300_integrated_macro_micro_prediction_v1/inputs/features.parquet",
        "capital_views.parquet": "reports/research/510300_integrated_macro_micro_prediction_v1/inputs/capital_views.parquet",
        "dividends.csv": "reports/research/510300_integrated_macro_micro_prediction_v1/inputs/dividends.csv",
        "credit.parquet": "data/raw/macro/510300_macro_stress_2015_v2/tsf_stock_yoy_release_vintage_2015_2026.parquet",
        "credit_receipt.json": "reports/data_quality/510300_macro_stress_inputs_2015_v2.json",
        "option_eod.parquet": "data/raw/return_tail/options/510300_tushare_eod.parquet",
        "option_risk.parquet": "data/raw/return_tail/options/510300_sse_risk_indicators.parquet",
        "previous_mandate.json": "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, source in sources.items():
        shutil.copy2(WORKSPACE / source, root / "inputs" / name)
    shutil.copy2(__file__, root / "code/daily_liquidity_insurance_tail_v1.py")
    shutil.copy2(WORKSPACE / "research/intraday_overnight_increment_v1.py", root / "code/account_engine.py")
    protocol = {
        "study_id": STUDY, "frozen_at": now(), "period": [START, END],
        "primary": "MACRO", "primary_comparison": "MACRO_MINUS_PRICE_ON_COMMON_INFORMATION",
        "user_instruction": "研究可放宽单笔3:1，改为账户尾部风险约束；只用最近两年的数据训练，每天滚动更新。",
        "mechanism": "只在最近五日含分红收益为负时考虑增加现货库存；检验信用变化、融资压力、保险定价代理是否区分随后价格修复与持续恶化。现货承接并不直接收取期权权利金。",
        "assets": ["510300.SH", "CASH_CNY"], "capital_cny": 200000,
        "targets": {"net_sharpe": 1.2, "net_cagr": .1, "max_drawdown": .1},
        "price_features": PRICE, "additional_features": MACRO[len(PRICE):],
        "feature_definitions": {
            "pressure5": "负的前一收盘五日含分红对数收益 / (rv20日波动率乘sqrt(5))；有符号，不截去上涨历史。",
            "trend20": "前一收盘二十日含分红对数收益 / (rv20日波动率乘sqrt(20))。",
            "log_rv5_rv60": "前一收盘5日与60日均方收益波动率之比的对数。",
            "credit_acceleration3_pp": "最后已公开社融存量同比减三个月前已公开同比；月序连续，不能称信用脉冲或银行贷款增速。",
            "funding_gap_pp": "复用已冻结09:00可知的DR007减七日逆回购政策率；不称实际利率减中性利率。",
            "log_iv_rv20": "两交易日前、约45日期限平值认购认沽隐含波动率均值 / 同日20日已实现年化波动率，取对数；不是已识别的方差风险溢价。",
        },
        "option_selection": "仅用历史官方交易代码M标准行还原行权价，30至75日到期，成交量>=100、持仓>=500，IV在0.01至3；先到期距45日、再行权价距当日ETF收盘；同到期同价认购认沽对。无结果后备选择。",
        "availability": "执行日09:00；价格至前日收盘；期权指标额外滞后一交易日即源日i-2；社融按available_at并额外推到发布日次日00:00；资金沿用已有时钟。所有来源均为事后收集，期权首次发布时间未认证，故仅开发回放。",
        "source_missing": {"NFCI": "NO_ADMITTED_REAL_TIME_VINTAGES", "CHINA_R_STAR_AND_EXPECTED_INFLATION": "NO_ADMITTED_PIT_SERIES"},
        "distribution": {
            "training_window": "严格使用执行日向前两个日历年内的预测原点；不使用更早历史或全样本标准化",
            "maturity": "五日持有标签的退出开盘日期严格小于本次执行日期；当天09:30退出的标签不能在09:00训练",
            "minimum_common_mature_rows": 252, "update": "每个交易日重建历史池和条件分布，包含所有可用历史原点，不依据后来是否实际交易筛选",
            "method": "HISTORY使用全部共同成熟样本；PRICE/MACRO按仅历史均值标准差标准化、截断[-5,5]，欧氏距离最近126行的经验分布；距离并列先历史原点较早者。固定一次，不搜索k或带宽。",
            "overlap": "标签为入场开盘至第五个后续交易日开盘加实际分红登记权益的每份收益。相邻五日标签重叠，126近邻不是126次独立机会；记录非重叠区间数，置信区间用日历区块。",
            "risk_statistics": "条件均值、方差、5%分位数和下侧5%期望损失；尾部按最差ceil(n*0.05)个完整观察平均，包含分位点边界。",
        },
        "decision": "同一账户逐日枚举合法100份目标数量，最大化五日预期收益减gamma=4的半方差惩罚、这次调整成本和预留平仓成本。资金不足不融资；未来开盘只用于成交与原预算复核，不用于改变预测。",
        "inventory": "五日收益非负时禁止增加份额；允许根据新分布减持或持有。预测缺失时最多保留旧份额五交易日，期间不得增加；随后请求退出。没有强制五日平仓或下跌加倍规则。",
        "risk": {"planned_net_rr_minimum": None, "conditional_five_day_ES95_fraction": .025,
                 "single_gap_stress_return": -.1, "gap_stress_loss_budget_fraction": .05,
                 "maximum_position_fraction": .5, "drawdown_trigger": .1,
                 "drawdown_reserve": "以已有收盘权益峰值计算距90%峰值剩余空间，单次10%假设冲击至多消耗其一半；平仓成本计入。",
                 "stop": "收盘回撤达到10%后下一可卖开盘清仓且本轮不恢复；这不是保证最大回撤，跌停与T+1可延迟。"},
        "costs": COSTS, "annual_days": 252, "cash_return": 0,
        "execution": "09:00按前收盘调整已知除息额确定固定数量与风险预算；开盘上跳使原风险预算超限则缩小买入数量；100份、tick0.001、最低佣金5；T+1；涨跌停拒绝相关方向；登记、除息应收、支付分账。",
        "controls": ["HISTORY", "PRICE", "BUY_HOLD"],
        "monitor": "预测原点五日结果成熟后跟踪最近126个预测的误差、5%分位覆盖与ES突破。至少63个成熟预测且下侧突破率>10%才记录观察报警；不将重叠观察当独立检验，不在本轮据报警增加新的启停参数。",
        "comparison": "一次主要配对比较；20日循环区块2000次、seed=202609246。全期、固定2021-2023与2024-终点及全部滚动两年分别披露；不挑最好窗口。",
        "acceptance": "压力成本全账户同时达夏普1.2、年化10%、回撤10%，宏观相对价格年化算术收益增量95%下界>0且两固定子期均正，才保留开发候选；历史已反复研究，仍不能认定独立验证。",
        "duplicate_review": [
            "旧ABCD检验形态后内部参与和日内隔夜；本轮不复活这些失败过滤。",
            "旧联合14特征504日月度Ridge预测均值；本轮测试信用变化和期权保险定价信息的条件完整分布，按两日历年每日更新并限制尾部库存。",
            "旧第83轮尾部预算对两个旧策略分配资金；本轮直接预测510300条件分布，无旧85/15及旧策略拼接。",
            "旧第126轮逐日专家后悔组合未稳定改善；本轮没有选择历史最优专家，也不将每日更新本身当收益证据。",
            "已有期权卖跨式研究为不同资产与目标范围；这里只复用原始期权观察数据，不导入其订单或账户。",
        ],
        "parameter_search": False, "new_collection": False, "orders_authorized": False,
        "independent_validation": False, "review_package": False, "user_spreadsheet": False,
    }
    save(root / "protocol.json", protocol, True)
    prior = json.loads((root / "inputs/previous_mandate.json").read_text(encoding="utf-8"))
    update = {"at": now(), "user_answers": [protocol["user_instruction"]],
              "supersedes_single_trade_rr3_for_this_research": True, "old_frozen_studies_unchanged": True,
              "scope_retained": protocol["assets"], "training_window_calendar_years": 2,
              "daily_update_required": True, "risk_parameters_are_fixed_research_assumptions": protocol["risk"],
              "new_data_collection_remains_paused": True, "orders_authorized": False}
    save(root / "authority_update.json", update, True)
    prior.update(minimum_planned_net_reward_risk_ratio=None,
                 planned_reward_risk_requirement_stage="本轮按用户新指令放宽单笔3:1，改为账户尾部风险；旧冻结研究不变",
                 latest_user_instruction=protocol["user_instruction"], training_window_calendar_years=2,
                 model_update_frequency="EVERY_TRADING_DAY", account_tail_risk_contract=protocol["risk"],
                 latest_state_mechanism_instruction_at=now(),
                 latest_state_mechanism_instruction_receipt=str((root / "authority_update.json").relative_to(WORKSPACE)),
                 latest_integrated_experiment=STUDY)
    save(WORKSPACE / "config/510300_existing_data_training_mandate_v1.json", prior)
    files = [root / "protocol.json", root / "authority_update.json", *sorted((root / "inputs").glob("*")), *sorted((root / "code").glob("*"))]
    save(root / "freeze.json", {"frozen_at": now(), "before_new_labels_and_accounts": True,
                               "files": {p.relative_to(root).as_posix(): digest(p) for p in files}, "sources": sources}, True)
    print("已冻结一个条件分布增量实验、两年每日更新及账户尾部预算，尚未计算新收益。", flush=True)


def verify_freeze(root):
    frozen = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
    for rel, expected in frozen["files"].items():
        if digest(root / rel) != expected:
            raise AssertionError(f"冻结来源改变：{rel}")
    correction = root / "implementation_fix_before_labels.json"
    if correction.exists():
        receipt = json.loads(correction.read_text(encoding="utf-8"))
        assert digest(root / receipt["corrected_code"]) == receipt["corrected_sha256"]
        assert digest(Path(__file__)) == receipt["corrected_sha256"]
    return True


def option_features(root, d):
    eod = pd.read_parquet(root / "inputs/option_eod.parquet")
    risk = pd.read_parquet(root / "inputs/option_risk.parquet")
    keep = ["trade_date", "contract_code", "exchange_contract_id", "implied_volatility"]
    panel = eod.merge(risk[keep], on=["trade_date", "contract_code"], how="left", validate="one_to_one")
    panel["trade_date"] = pd.to_datetime(panel.trade_date)
    panel["expiry_date"] = pd.to_datetime(panel.expiry_date)
    parsed = panel.exchange_contract_id.fillna("").str.extract(r"^510300([CP])(\d{4})M(\d{5})$")
    panel["historical_strike"] = pd.to_numeric(parsed[2], errors="coerce") / 1000
    panel["dte"] = (panel.expiry_date - panel.trade_date).dt.days
    mask = parsed[0].eq(panel.option_type) & panel.historical_strike.notna()
    mask &= panel.dte.between(30, 75) & panel.volume.ge(100) & panel.open_interest.ge(500)
    mask &= panel.implied_volatility.between(.01, 3) & panel.close.gt(0)
    px = d.set_index("date")
    rows = []
    for date, part in panel[mask].groupby("trade_date", sort=True):
        if date not in px.index:
            continue
        pair = part[part.option_type.eq("C")].merge(part[part.option_type.eq("P")],
            on=["expiry_date", "historical_strike"], suffixes=("_C", "_P"))
        if pair.empty:
            continue
        pair["expiry_distance"] = abs(pair.dte_C - 45)
        pair["strike_distance"] = abs(pair.historical_strike - px.loc[date, "close"])
        row = pair.sort_values(["expiry_distance", "strike_distance", "expiry_date", "historical_strike", "contract_code_C", "contract_code_P"], kind="stable").iloc[0]
        iv = (float(row.implied_volatility_C) + float(row.implied_volatility_P)) / 2
        rv = float(px.loc[date, "rv20_day"]) * np.sqrt(252)
        rows.append({"option_source_date": date, "iv_atm": iv, "rv20_annual": rv,
                     "log_iv_rv20": np.log(iv / rv) if rv > 0 else np.nan,
                     "call_id": row.contract_code_C, "put_id": row.contract_code_P,
                     "expiry_date": row.expiry_date, "historical_strike": row.historical_strike})
    return pd.DataFrame(rows)


def design(root, e):
    d = pd.read_parquet(root / "inputs/prices.parquet")
    d["date"] = pd.to_datetime(d.date)
    d = d[d.date.le(END)].reset_index(drop=True)
    dividends = e.normalize_dividends(pd.read_csv(root / "inputs/dividends.csv"))
    d = e.build_features(e.normalize_prices(d), dividends)
    for window in [5, 20, 60]:
        d[f"rv{window}_day"] = np.sqrt(d.total_log.pow(2).rolling(window).mean())
    d["pressure5_close"] = -d.total_log.rolling(5).sum() / (d.rv20_day * np.sqrt(5))
    d["trend20_close"] = d.total_log.rolling(20).sum() / (d.rv20_day * np.sqrt(20))
    d["log_rv5_rv60_close"] = np.log(d.rv5_day / d.rv60_day)
    views = pd.read_parquet(root / "inputs/capital_views.parquet")
    views["date"] = pd.to_datetime(views.date)
    cols = ["date", "decision_time", "funding_gap_pp", "funding_known", "dr_available_at", "policy_available_at", "preholiday3", "holiday_gap_days", "global_shock"]
    x = d[["date"]].merge(views[cols], on="date", how="left", validate="one_to_one")
    x["idx"] = np.arange(len(x))
    for column in PRICE:
        x[column] = d[column + "_close"].shift(1)
    credit = pd.read_parquet(root / "inputs/credit.parquet").sort_values("reference_period").reset_index(drop=True)
    period = pd.PeriodIndex(credit.reference_period, freq="M")
    value = dict(zip(period, credit.first_release_value))
    credit["credit_acceleration3_pp"] = [float(v) - float(value.get(p - 3, np.nan)) for p, v in zip(period, credit.first_release_value)]
    credit["credit_source_available_at"] = pd.to_datetime(credit.available_at, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    credit["credit_available_at"] = (credit.credit_source_available_at.dt.normalize() + pd.Timedelta(days=1)).astype("datetime64[ns]")
    credit = credit.rename(columns={"reference_period": "credit_reference_period", "first_release_value": "credit_yoy"})
    x["decision_time"] = pd.to_datetime(x.decision_time).astype("datetime64[ns]")
    x = pd.merge_asof(x.sort_values("decision_time"), credit[["credit_available_at", "credit_source_available_at", "credit_reference_period", "credit_yoy", "credit_acceleration3_pp"]].sort_values("credit_available_at"),
                      left_on="decision_time", right_on="credit_available_at", direction="backward")
    x["credit_known"] = x.credit_acceleration3_pp.notna() & ((x.decision_time - x.credit_available_at).dt.days <= 60)
    options = option_features(root, d)
    x["option_source_date"] = d.date.shift(2)
    x = x.merge(options, on="option_source_date", how="left", validate="many_to_one")
    x["option_known"] = x.log_iv_rv20.notna()
    x["common_known"] = x.funding_known.fillna(False).astype(bool) & x.credit_known & x.option_known
    x["common_known"] &= np.isfinite(x[MACRO].to_numpy(float)).all(axis=1)
    for source in ["dr_available_at", "policy_available_at", "credit_available_at"]:
        known = x.common_known
        assert (pd.to_datetime(x.loc[known, source]) <= x.loc[known, "decision_time"]).all()
    assert (x.loc[x.option_known, "option_source_date"] < x.loc[x.option_known, "date"]).all()
    return d, dividends, x, options


def labels(d, dividends):
    result = []
    for i in range(len(d) - 5):
        entry, end = d.date.iloc[i], d.date.iloc[i + 5]
        amount = dividends.loc[dividends.record_date.ge(entry) & dividends.record_date.lt(end), "cash_dividend_per_share"].sum()
        result.append({"idx": i, "date": entry, "exit_idx": i + 5, "exit_date": end,
                       "gross_return5": (d.open.iloc[i + 5] + amount) / d.open.iloc[i] - 1,
                       "dividend_per_share": amount})
    return pd.DataFrame(result)


def empirical_statistics(values):
    y = np.asarray(values, float)
    tail = np.sort(y)[:max(1, int(np.ceil(len(y) * .05)))]
    return {"mu5": float(y.mean()), "variance5": float(y.var(ddof=1)),
            "q05": float(np.quantile(y, .05)), "es95": max(0., -float(tail.mean())),
            "win_probability": float((y > 0).mean()), "selected_tail_count": len(tail)}


def nonoverlap_count(indices):
    next_allowed, count = -1, 0
    for i in sorted(indices):
        if i >= next_allowed:
            count += 1
            next_allowed = i + 5
    return count


def one_day_distribution(i, x, lab):
    today = x.date.iloc[i]
    eligible = lab.date.ge(today - pd.DateOffset(years=2)) & lab.exit_idx.lt(i)
    pool = lab.loc[eligible].copy()
    pool = pool[x.common_known.iloc[pool.idx.to_numpy(int)].to_numpy(bool)]
    receipt = {"idx": i, "date": today, "training_lower_bound": today - pd.DateOffset(years=2),
               "n_train": len(pool), "latest_exit_idx": int(pool.exit_idx.max()) if len(pool) else None,
               "common_known_today": bool(x.common_known.iloc[i])}
    if not receipt["common_known_today"] or len(pool) < 252:
        receipt["status"] = "NO_VIEW_CURRENT" if not receipt["common_known_today"] else "NO_VIEW_TRAINING"
        return [], receipt, []
    receipt["status"] = "UPDATED"
    indices = pool.idx.to_numpy(int)
    predictions, records = [], []
    for model, columns in MODELS.items():
        if columns:
            values = x.loc[indices, columns].to_numpy(float)
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            sd[sd < 1e-12] = 1.
            z = np.clip((values - mean) / sd, -5, 5)
            current = np.clip((x.loc[i, columns].to_numpy(float) - mean) / sd, -5, 5)
            distance = ((z - current) ** 2).sum(axis=1)
            selection = np.lexsort((indices, distance))[:126]
        else:
            selection = np.arange(len(indices))
            mean, sd = np.array([]), np.array([])
        chosen = pool.iloc[selection]
        summary = empirical_statistics(chosen.gross_return5)
        predictions.append({**receipt, "model": model, "n_selected": len(chosen),
                            "nonoverlap_intervals_selected": nonoverlap_count(chosen.idx), **summary})
        records.append({"idx": i, "model": model, "columns": columns, "mean": mean.tolist(),
                        "scale": sd.tolist(), "training_indices": indices.tolist(),
                        "selected_indices": chosen.idx.tolist()})
    return predictions, receipt, records


def learn(x, lab, root):
    predictions, schedules, records = [], [], []
    for i in np.flatnonzero(x.date.ge(START)):
        p, s, r = one_day_distribution(int(i), x, lab)
        predictions.extend(p)
        schedules.append(s)
        records.extend(r)
        if len(schedules) % 300 == 0:
            print(f"逐日条件分布已更新至{x.date.iloc[i].date()}，已记录{len(predictions)}个模型判断。", flush=True)
    pred = pd.DataFrame(predictions)
    pd.DataFrame(schedules).to_parquet(root / "results/update_schedule.parquet", index=False)
    save(root / "results/saved_distribution_models.json", records)
    pred.to_parquet(root / "results/predictions.parquet", index=False)
    return pred, records


def risk_limits(nav, peak, es95):
    remaining = max(0., nav - .9 * peak)
    gap_budget = min(.05 * nav, .5 * remaining)
    return {"es_budget_cny": .025 * nav, "gap_budget_cny": gap_budget,
            "position_budget_cny": .5 * nav, "es95": max(float(es95), 1e-8)}


def risk_valid(e, quantity, ref, limits, cost):
    if quantity == 0:
        return True
    notional = quantity * ref
    exit_price = e.fill_price(ref, -1, cost, .001)
    exit_cost = quantity * (ref - exit_price) + e.commission(quantity, exit_price, cost)
    return (notional <= limits["position_budget_cny"] + 1e-8 and
            notional * limits["es95"] + exit_cost <= limits["es_budget_cny"] + 1e-8 and
            .1 * notional + exit_cost <= limits["gap_budget_cny"] + 1e-8)


def plan(e, account, ref, peak, prediction, pressure):
    nav = account.value(ref)
    limits = risk_limits(nav, peak, prediction.es95)
    maximum = int(limits["position_budget_cny"] / ref / 100) * 100
    if pressure <= 0:
        maximum = min(maximum, account.shares)
    maximum = min(maximum, account.shares + e.affordable_quantity(account.cash,
                  e.fill_price(ref, 1, COSTS["STRESS"], .001), COSTS["STRESS"], 100))
    best = None
    # 两费用账户均使用压力成本做事前决策，避免低成本条件改变经济门槛。
    for target in range(0, maximum + 1, 100):
        if not risk_valid(e, target, ref, limits, COSTS["STRESS"]):
            continue
        change = target - account.shares
        qpx = e.fill_price(ref, 1 if change > 0 else -1, COSTS["STRESS"], .001)
        adjust_cost = abs(change) * abs(qpx - ref) + e.commission(change, qpx, COSTS["STRESS"])
        exit_px = e.fill_price(ref, -1, COSTS["STRESS"], .001)
        exit_cost = target * (ref - exit_px) + e.commission(target, exit_px, COSTS["STRESS"])
        weight = target * ref / nav
        score = weight * prediction.mu5 - 2. * weight * weight * prediction.variance5 - (adjust_cost + exit_cost) / nav
        row = {"target_shares": target, "requested_quantity": change, "planned_weight": weight,
               "score": score, "estimated_adjustment_cost": adjust_cost, "reserved_exit_cost": exit_cost,
               "reference_price": ref, "decision_equity": nav, "mu5": prediction.mu5,
               "variance5": prediction.variance5, **limits}
        if best is None or score > best["score"] + 1e-12:
            best = row
    assert best is not None
    return best


def simulate(root, e, d, dividends, x, predictions, policy, cost_name):
    first = int(np.flatnonzero(d.date.ge(START))[0])
    last = len(d) - 1
    pred = {int(r.idx): r for r in predictions[predictions.model.eq(policy)].itertuples()}
    account = e.Account(200000.)
    cfg = {"lot": 100, "tick": .001, "limit_fraction": .1}
    cost = COSTS[cost_name]
    previous_nav, previous_mark, peak = 200000., float(d.close.iloc[first - 1]), 200000.
    stopped, last_view = False, first - 6
    records, decisions = [], []
    events = dividends.to_dict("records")
    for i in range(first, last + 1):
        row = d.iloc[i]
        day, op, close = row.date, float(row.open), float(row.close)
        old_shares, recognized, paid = account.shares, 0., 0.
        for k, event in enumerate(events):
            if event["ex_date"] == day:
                value = account.entitlements.get(k, 0) * event["cash_dividend_per_share"]
                account.receivables[k] = value
                recognized += value
            if event["payment_date"] < day and k in account.receivables:
                value = account.receivables.pop(k)
                account.cash += value
                paid += value
        # 除息转换不产生新买入者的股息权；原持仓的应收款已单独入账。
        ref = float(row.previous_close - row.dividend)
        prediction = pred.get(i)
        p = {"requested_quantity": 0, "target_shares": account.shares, "reason": "NO_VIEW_HOLD_OLD_INVENTORY"}
        if i == last:
            p.update(requested_quantity=-account.shares, target_shares=0, reason="TERMINAL_EXIT")
        elif policy == "BUY_HOLD":
            if i == first:
                quantity = e.affordable_quantity(account.cash, e.fill_price(ref, 1, cost, .001), cost, 100)
                p.update(requested_quantity=quantity, target_shares=quantity, reason="BUY_HOLD_INITIAL")
        elif stopped:
            p.update(requested_quantity=-account.shares, target_shares=0, reason="DRAWDOWN_STOP")
        elif prediction is not None:
            last_view = i
            p = {**plan(e, account, ref, peak, prediction, x.pressure5.iloc[i]), "reason": "DAILY_DISTRIBUTION"}
        elif i - last_view >= 5:
            p.update(requested_quantity=-account.shares, target_shares=0, reason="NO_VIEW_INVENTORY_EXPIRED")
        requested = int(p["requested_quantity"])
        pre_gap_request = requested
        # 开盘只复核09:00已确定的预算；无法靠事后涨价扩大原风险额度。
        if requested > 0 and policy != "BUY_HOLD":
            while requested > 0 and not risk_valid(e, account.shares + requested, op, p, COSTS["STRESS"]):
                requested -= 100
        execution = e.execute_order(account, requested, op, float(row.previous_close), float(row.dividend), i, cost, cfg)
        mark = op if i == last else close
        if i != last:
            for k, event in enumerate(events):
                if event["payment_date"] == day and k in account.receivables:
                    value = account.receivables.pop(k)
                    account.cash += value
                    paid += value
                if event["record_date"] == day:
                    account.entitlements[k] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        assert abs(error) < 1e-6
        account.assert_valid()
        peak = max(peak, nav)
        dd = 1 - nav / peak
        if policy != "BUY_HOLD" and dd >= .1:
            stopped = True
        records.append({"date": day, "idx": i, "policy": policy, "cost": cost_name,
                        "open": op, "mark": mark, "mark_clock": "OPEN_TERMINAL" if i == last else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * mark / nav, "accounting_error": error,
                        "drawdown": dd, "risk_stopped": stopped, "terminal_unliquidated": bool(i == last and account.shares),
                        **execution})
        decisions.append({"date": day, "idx": i, "policy": policy, "cost": cost_name,
                          "prediction_available": prediction is not None, "pressure5": x.pressure5.iloc[i],
                          "pre_open_requested_quantity": pre_gap_request, "gap_checked_request": requested,
                          "filled_quantity": execution["filled_quantity"], "actual_open": op, **p})
        previous_nav, previous_mark = nav, mark
    return pd.DataFrame(records), pd.DataFrame(decisions)


def summary(e, ledger):
    r = ledger.net_return.to_numpy(float)
    stats = e.return_metrics(r, 252)
    centered = r - r.mean()
    sd = r.std(ddof=0)
    sorted_r = np.sort(r)
    stats.update(ending_equity=float(ledger.equity.iloc[-1]),
                 mean_exposure=float(ledger.exposure.mean()), cash_days=int(ledger.shares.eq(0).sum()),
                 buy_orders=int(ledger.filled_quantity.gt(0).sum()), sell_orders=int(ledger.filled_quantity.lt(0).sum()),
                 cost_cny=float((ledger.commission + ledger.slippage_cost).sum()),
                 worst_day=float(r.min()), es95_daily=max(0., -float(sorted_r[:int(np.ceil(len(r) * .05))].mean())),
                 daily_skew=float((centered ** 3).mean() / sd ** 3) if sd > 0 else None,
                 excess_kurtosis=float((centered ** 4).mean() / sd ** 4 - 3) if sd > 0 else None,
                 stopped=bool(ledger.risk_stopped.any()),
                 maximum_accounting_error=float(ledger.accounting_error.abs().max()))
    return stats


def mature_monitor(pred, lab):
    scored = pred.merge(lab[["idx", "exit_idx", "exit_date", "gross_return5"]], on="idx", how="left", validate="many_to_one")
    scored["prediction_error"] = scored.gross_return5 - scored.mu5
    scored["q05_breach"] = (scored.gross_return5 < scored.q05).where(scored.gross_return5.notna())
    rows = []
    for model, group in scored.groupby("model"):
        for row in group.itertuples():
            past = group[group.exit_idx.lt(row.idx)].tail(126)
            eligible = len(past) >= 63
            rate = float(past.q05_breach.astype(float).mean()) if len(past) else None
            rows.append({"idx": row.idx, "date": row.date, "model": model, "mature_prior_predictions": len(past),
                         "latest_mature_exit_idx": float(past.exit_idx.max()) if len(past) else None,
                         "mean_prior_error": float(past.prediction_error.mean()) if len(past) else None,
                         "prior_q05_breach_rate": rate,
                         "observation_alert": bool(eligible and rate > .1),
                         "used_to_change_trades": False})
    return scored, pd.DataFrame(rows)


def comparisons(e, accounts, scored):
    left = accounts[("STRESS", "MACRO")]
    right = accounts[("STRESS", "PRICE")]
    a, b = left.net_return.to_numpy(float), right.net_return.to_numpy(float)
    rng = np.random.default_rng(202609246)
    differences, sharpe = [], []
    for _ in range(2000):
        ix = e.block_indices(rng, len(a), 20)
        differences.append(float((a[ix] - b[ix]).mean() * 252))
        sharpe.append(e.return_metrics(a[ix], 252)["net_sharpe"])
    pair = scored[scored.model.isin(["MACRO", "PRICE"])].pivot(index="idx", columns="model", values="mu5")
    actual = scored.drop_duplicates("idx").set_index("idx").gross_return5
    pair["actual"] = actual
    pair = pair.dropna()
    losses = {m: float(((pair[m] - pair.actual) ** 2).mean()) for m in ["MACRO", "PRICE"]}
    halves = {}
    for name, lo, hi in [("2021_2023", START, "2023-12-31"), ("2024_END", "2024-01-01", END)]:
        ix = left.date.between(lo, hi).to_numpy()
        halves[name] = {"annualized_arithmetic_increment": float((a[ix] - b[ix]).mean() * 252),
                        "MACRO": e.return_metrics(a[ix], 252), "PRICE": e.return_metrics(b[ix], 252)}
    return {"primary_annualized_arithmetic_increment": float((a - b).mean() * 252),
            "increment_ci95": np.quantile(differences, [.025, .975]).tolist(),
            "primary_sharpe_ci95": e.interval(sharpe), "paired_mature_predictions": len(pair),
            "prediction_MSE": losses, "prediction_MSE_improvement": 1 - losses["MACRO"] / losses["PRICE"],
            "fixed_eras": halves, "interval_status": "DEVELOPMENT_BLOCK_INTERVAL_NOT_PROJECT_SELECTION_ADJUSTED"}


def rolling_windows(e, accounts):
    rows = []
    for (cost, model), ledger in accounts.items():
        if cost != "STRESS":
            continue
        dates = pd.DatetimeIndex(ledger.date)
        for end in range(len(ledger)):
            lower = dates[end] - pd.DateOffset(years=2)
            if lower < dates[0]:
                continue
            start = dates.searchsorted(lower, side="left")
            sample = ledger.iloc[start:end + 1]
            rows.append({"cost": cost, "model": model, "start": sample.date.iloc[0], "end": sample.date.iloc[-1],
                         "days": len(sample), **e.return_metrics(sample.net_return.to_numpy(float), 252)})
    return pd.DataFrame(rows)


def exposure_diagnostic(ledger, d):
    frame = ledger.merge(d[["date", "total_simple"]], on="date", how="left", validate="one_to_one")
    r = frame.total_simple.to_numpy(float)
    y = frame.net_return.to_numpy(float)
    z = np.column_stack([np.ones(len(r)), r, np.minimum(r, 0.) ** 2])
    beta = np.linalg.lstsq(z, y, rcond=None)[0]
    stress = r <= np.quantile(r, .05)
    return {"descriptive_linear_beta": float(beta[1]), "descriptive_negative_move_squared_coefficient": float(beta[2]),
            "account_mean_return_worst5pct_underlying_days": float(y[stress].mean()),
            "underlying_worst5pct_days": int(stress.sum()),
            "inference": "事后暴露描述，平方项为负可提示下跌非线性损失；不能据此证明卖出方差风险溢价或真实Gamma。"}


def checks(e):
    account = e.Account(200000.)
    cfg = {"lot": 100, "tick": .001, "limit_fraction": .1}
    e.execute_order(account, 1000, 4., 4., 0., 10, COSTS["STRESS"], cfg)
    same_day = e.execute_order(account, -1000, 4., 4., 0., 10, COSTS["STRESS"], cfg)
    assert same_day["filled_quantity"] == 0
    next_day = e.execute_order(account, -1000, 3.7, 4., 0., 11, COSTS["STRESS"], cfg)
    assert next_day["filled_quantity"] == -1000
    limits = risk_limits(200000., 200000., .05)
    assert risk_valid(e, 24000, 4., limits, COSTS["STRESS"])
    assert not risk_valid(e, 50000, 4., limits, COSTS["STRESS"])
    reduced = risk_limits(185000., 200000., .05)
    assert reduced["gap_budget_cny"] < limits["gap_budget_cny"]
    assert empirical_statistics([-.1, -.02, .03, .04])["es95"] == .1
    return {"T_plus_one": True, "gap_loss_realized_at_price": True,
            "tail_and_gap_budget": True, "drawdown_reserve_contracts": True, "tail_keeps_worst_loss": True}


def verify_saved(root):
    e = engine(root)
    verify_freeze(root)
    x = pd.read_parquet(root / "results/decision_information.parquet")
    lab = pd.read_parquet(root / "results/mature_labels.parquet")
    pred = pd.read_parquet(root / "results/predictions.parquet")
    selected = json.loads((root / "results/saved_distribution_models.json").read_text(encoding="utf-8"))
    index = {(int(row.idx), row.model): row for row in pred.itertuples()}
    reproduced = 0
    for saved in selected:
        i, model = saved["idx"], saved["model"]
        pool = lab.set_index("idx").loc[saved["training_indices"]]
        assert pool.exit_idx.max() < i
        assert (pool.date >= x.date.iloc[i] - pd.DateOffset(years=2)).all()
        vals = lab.set_index("idx").loc[saved["selected_indices"], "gross_return5"]
        stats = empirical_statistics(vals)
        for k, value in stats.items():
            assert abs(value - getattr(index[(i, model)], k)) < 1e-12
        reproduced += 1
    # 历史裁切只复核两个原点，不重新拟合或重复账户收益。
    prefix = []
    for cutoff in ["2023-12-29", "2025-12-31"]:
        eligible = x[x.date.le(cutoff) & x.idx.isin(pred.idx)]
        if eligible.empty:
            continue
        i = int(eligible.idx.iloc[-1])
        p, _, _ = one_day_distribution(i, x.iloc[:i + 1].copy(), lab[lab.exit_idx.lt(i)].copy())
        for row in p:
            np.testing.assert_allclose([row[k] for k in ["mu5", "es95", "q05"]],
                                      [getattr(index[(i, row["model"])], k) for k in ["mu5", "es95", "q05"]], atol=1e-14, rtol=0)
        prefix.append(str(x.date.iloc[i].date()))
    maximum_error, accounts = 0., 0
    for path in (root / "accounts").glob("*/*_ledger.parquet"):
        frame = pd.read_parquet(path)
        cash = 200000. - (frame.filled_quantity * frame.fill_price.fillna(0)).cumsum() - frame.commission.cumsum() + frame.dividend_paid.cumsum()
        np.testing.assert_allclose(cash, frame.cash, atol=1e-7, rtol=0)
        np.testing.assert_allclose(frame.filled_quantity.cumsum(), frame.shares, atol=0, rtol=0)
        np.testing.assert_allclose(frame.cash + frame.shares * frame.mark + frame.dividend_receivable, frame.equity, atol=1e-7, rtol=0)
        assert not frame.terminal_unliquidated.iloc[-1]
        maximum_error = max(maximum_error, float(frame.accounting_error.abs().max()))
        accounts += 1
    receipt = {"verified_at": now(), "frozen_files_unchanged": True, "saved_distributions_recomputed": reproduced,
               "strict_two_calendar_years_and_exit_maturity": True, "prefix_cutoffs": prefix,
               "accounts_cash_shares_equity_reconciled": accounts, "maximum_accounting_error": maximum_error,
               "mechanism_checks": checks(e), "new_parameter_search": False}
    save(root / "verification.json", receipt)
    return receipt


def run(root, resume=False):
    verify_freeze(root)
    if resume:
        assert (root / "RUN_STARTED.json").exists()
        assert (root / "implementation_fix_before_labels.json").exists()
        assert not (root / "results/mature_labels.parquet").exists()
        assert not (root / "result.json").exists()
        save(root / "RUN_RESUMED_BEFORE_LABELS.json", {"resumed_at": now(), "reason": "仅修复pandas日期精度拼接错误，协议和经济参数保持"}, True)
    else:
        save(root / "RUN_STARTED.json", {"started_at": now()}, True)
    e = engine(root)
    save(root / "mechanism_checks.json", checks(e))
    d, dividends, x, options = design(root, e)
    d.to_parquet(root / "results/market.parquet", index=False)
    x.to_parquet(root / "results/decision_information.parquet", index=False)
    options.to_parquet(root / "results/option_observations.parquet", index=False)
    primary = x[x.date.ge(START)]
    save(root / "data_admission.json", {"days": len(primary), "common_known_days": int(primary.common_known.sum()),
         "funding_known_days": int(primary.funding_known.sum()), "credit_known_days": int(primary.credit_known.sum()),
         "option_known_days": int(primary.option_known.sum()), "NFCI": "NO_ADMITTED_REAL_TIME_VINTAGES",
         "CHINA_R_STAR": "NO_ADMITTED_PIT_SERIES", "first_public_option_clock_authenticated": False,
         "new_market_data_downloads": 0, "historical_development_only": True})
    lab = labels(d, dividends)
    lab.to_parquet(root / "results/mature_labels.parquet", index=False)
    pred, records = learn(x, lab, root)
    if pred.empty:
        save(root / "result.json", {"status": "NO_VIEW_INSUFFICIENT_COMMON_TRAINING", "goal_achieved": False})
        return
    scored, monitor = mature_monitor(pred, lab)
    scored.to_parquet(root / "results/prediction_evaluation.parquet", index=False)
    monitor.to_parquet(root / "results/mature_monitor.parquet", index=False)
    accounts, metrics, annual = {}, [], []
    for cost_name in COSTS:
        folder = root / "accounts" / cost_name
        folder.mkdir(exist_ok=True)
        for policy in [*MODELS, "BUY_HOLD"]:
            ledger, decisions = simulate(root, e, d, dividends, x, pred, policy, cost_name)
            ledger.to_parquet(folder / f"{policy}_ledger.parquet", index=False)
            decisions.to_parquet(folder / f"{policy}_decisions.parquet", index=False)
            accounts[(cost_name, policy)] = ledger
            metrics.append({"cost": cost_name, "policy": policy, **summary(e, ledger)})
            for year, group in ledger.groupby(ledger.date.dt.year):
                annual.append({"cost": cost_name, "policy": policy, "year": int(year),
                               "days": len(group), **summary(e, group)})
            print(f"已完成{cost_name}／{policy}完整账户。", flush=True)
    compare = comparisons(e, accounts, scored)
    rolling = rolling_windows(e, accounts)
    rolling.to_parquet(root / "results/all_rolling_two_year_windows.parquet", index=False)
    save(root / "results/account_metrics.json", metrics)
    save(root / "results/calendar_year_metrics.json", annual)
    save(root / "results/increment_comparison.json", compare)
    primary_metrics = next(m for m in metrics if m["cost"] == "STRESS" and m["policy"] == "MACRO")
    point_pass = primary_metrics["net_sharpe"] is not None and primary_metrics["net_sharpe"] >= 1.2 and primary_metrics["annualized_return"] >= .1 and primary_metrics["max_drawdown"] >= -.1
    increment_pass = compare["increment_ci95"][0] > 0 and all(v["annualized_arithmetic_increment"] > 0 for v in compare["fixed_eras"].values())
    window = rolling[rolling.model.eq("MACRO")]
    result = {"study_id": STUDY, "completed_at": now(),
              "status": "DEVELOPMENT_CANDIDATE_REQUIRES_FORWARD_VALIDATION" if point_pass and increment_pass else "FROZEN_NO_QUALIFIED_HIGH_SHARPE_LIQUIDITY_INSURANCE",
              "primary": primary_metrics, "all_account_metrics": metrics, "comparison": compare,
              "daily_model_updates": len(records), "unique_updated_days": int(pred.idx.nunique()),
              "first_prediction_date": pred.date.min(), "last_prediction_date": pred.date.max(),
              "tail_observation_alerts": int(monitor.loc[monitor.model.eq("MACRO"), "observation_alert"].sum()),
              "all_rolling_two_years": {"windows": len(window), "median_sharpe": window.net_sharpe.median(),
                   "min_sharpe": window.net_sharpe.min(), "max_sharpe": window.net_sharpe.max(),
                   "min_cagr": window.annualized_return.min(), "max_cagr": window.annualized_return.max(),
                   "simultaneously_met_point_targets": int((window.net_sharpe.ge(1.2) & window.annualized_return.ge(.1) & window.max_drawdown.ge(-.1)).sum())},
              "exposure_diagnostic": exposure_diagnostic(accounts[("STRESS", "MACRO")], d),
              "point_target_met": point_pass, "macro_increment_gate_met": increment_pass,
              "goal_achieved": False, "independent_validation": False,
              "market_data_collection": "REMAINS_PAUSED", "orders_authorized": False}
    save(root / "result.json", result, True)
    verification = verify_saved(root)
    print(json.dumps(clean({"状态": result["status"], "主账户": primary_metrics,
                           "配对增量": compare, "验证": verification}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "run", "resume", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "resume": lambda root: run(root, resume=True), "verify": verify_saved}[args.command](args.root)
