"""510300期权波动率组合的第一轮概率与盈亏幅度训练。仅用既有数据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import GammaRegressor, LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "reports/research/510300_option_volatility_probability_payoff_v1"
STUDY = "510300_OPTION_VOLATILITY_PROBABILITY_PAYOFF_V1"
FEATURES = ["log_atm_iv", "log_iv_over_rv20", "log_rv5_over_rv20", "log_rv20_over_rv60", "put_call_iv_difference_scaled"]
STRUCTURES = ["LONG_ATM_STRADDLE", "PROTECTED_SHORT_ATM_STRADDLE"]
MODELS = ["MATURE_EMPIRICAL", "VOLATILITY_LOGIT_GAMMA"]


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def prepare(out: Path) -> dict:
    if (out / "freeze.json").exists():
        raise RuntimeError("该版本已经冻结；请使用verify，不重复训练或覆盖。")
    inputs = {
        "option_eod.parquet": ROOT / "data/raw/return_tail/options/510300_tushare_eod.parquet",
        "option_risk.parquet": ROOT / "data/raw/return_tail/options/510300_sse_risk_indicators.parquet",
        "etf_market.parquet": ROOT / "reports/research/510300_dense_probability_payoff_nodes_v1/inputs/market.parquet",
        "mandate.json": ROOT / "config/510300_options_only_volatility_mandate_v1.json",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "inputs").mkdir(exist_ok=True)
    for name, src in inputs.items():
        shutil.copy2(src, out / "inputs" / name)
    (out / "code").mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), out / "code" / Path(__file__).name)
    protocol = {
        "study_id": STUDY, "frozen_at": now(), "new_downloads": 0,
        "user_request": json.loads(inputs["mandate.json"].read_text(encoding="utf-8")),
        "stage": "训练概率与盈亏幅度；本轮不生成资金账户、交易指令或夏普估计",
        "historical_prices_previously_seen": True, "independent_holdout": False,
        "structures": STRUCTURES, "models": MODELS, "features": FEATURES,
        "decision_clock": "决策日在收盘后，全部行情和风险特征统一使用前一交易日；下一交易日入场",
        "holding_sessions": 10,
        "contract_selection": {
            "dte_calendar_days": [30, 75], "target_dte": 45,
            "atm": "同到期、同行权价的标准认购认沽对；先按到期距45日、再按行权价距观察日ETF收盘价排序",
            "minimum_prior_volume_each_leg": 100, "minimum_prior_open_interest_each_leg": 500,
            "historical_identity": "只使用当日官方exchange_contract_id中的M标准标识；行权价取最后5位/1000，每张10000份",
            "snapshot_metadata_not_used_for_selection": ["is_adjusted", "strike", "contract_unit"],
            "protected_short_wings": "同到期，买入最接近中心行权价下方5%的认沽与上方5%的认购；两翼分别至少一档价外",
            "no_contract_fallback_after_outcomes": True,
            "same_underlying_only": "510300；无ETF或期货Delta对冲，组合仍可能存在方向敞口",
        },
        "labels": {
            "unit": "一套组合的净损益 / 观察日可知的组合风险金额",
            "long_risk_scale": "观察日认购加认沽总权利金加双边固定费用",
            "short_risk_scale": "较大保护翼宽度乘10000，减观察日净权利金收入，加双边固定费用",
            "fee_cny_per_contract_per_side": 5.0,
            "BASE": "入场各腿日开盘价、退出各腿日收盘价；买加卖减max(0.0002,报价*0.005)",
            "STRESS": "同上，买加卖减max(0.0005,报价*0.01)；这是主训练标签",
            "DAY_EXTREME": "逐腿买价取当日高价、卖价取当日低价，另扣固定费用；仅作价格包络诊断",
            "execution_evidence": "日线价格代理，不是历史同步可成交组合报价，也不是下单模拟成交证明",
            "unscored": "到期或历史调整、缺失关键日行情/风险标识、零成交、报价无效时标记未知；不填零盈亏",
            "unscored_not_a_trade_filter": "未来路径导致的未知标签不可用于事前拒单，本轮因此不生成账户",
        },
        "training": {
            "minimum_mature_samples_per_structure": 252,
            "refit": "每自然月首个事前合约可选且成熟样本足够的决策日；不依据该日未来标签是否完整安排拟合；当月参数固定",
            "maturity": "标签退出日期严格早于模型时点",
            "weights": "各十日持有区间平均逆重叠数；不把重叠标签算作独立证据",
            "probability": "带截距L2逻辑回归；平均加权损失的惩罚强度1；不做类别再平衡",
            "amplitudes": "盈利/亏损分别为Gamma均值回归，正幅度，alpha=1",
            "scaling": "仅用当时成熟样本均值和标准差；标准化特征截断[-3,3]",
            "fallback": "若成熟样本只有一个类别或条件样本不足20，使用当时经验均值并明确标记",
            "baseline": "相同成熟样本与重叠权重的经验胜率、盈利与亏损幅度",
            "parameter_search": False,
        },
        "evaluation": "概率Brier与AUC、盈利/亏损条件均方误差、净期望均方误差；全段、2023年前段、2024年后段分别报告",
        "no_profitability_claim_from_predictions": True,
        "next_account_requirements": "组合统一现金流、保证金、逐日估值与空仓日；调整及缺失行情处理完整后才能评价20万元全账户目标",
    }
    dump(out / "protocol.json", protocol)
    frozen_paths = [out / "protocol.json", *sorted((out / "inputs").glob("*")), *sorted((out / "code").glob("*"))]
    dump(out / "freeze.json", {"frozen_at": now(), "before_new_payoff_labels": True,
                              "files": {p.relative_to(out).as_posix(): digest(p) for p in frozen_paths}})
    return protocol


def load_data(out: Path):
    eod = pd.read_parquet(out / "inputs/option_eod.parquet")
    risk = pd.read_parquet(out / "inputs/option_risk.parquet")
    market = pd.read_parquet(out / "inputs/etf_market.parquet")
    for d, column in [(eod, "trade_date"), (risk, "trade_date"), (market, "date")]:
        d[column] = pd.to_datetime(d[column]).dt.normalize()
    if eod.duplicated(["trade_date", "contract_code"]).any() or risk.duplicated(["trade_date", "contract_code"]).any():
        raise ValueError("合约日期键重复。")
    panel = eod.merge(risk[["trade_date", "contract_code", "exchange_contract_id", "delta", "implied_volatility"]],
                      on=["trade_date", "contract_code"], how="left", validate="one_to_one")
    panel["expiry_date"] = pd.to_datetime(panel.expiry_date).dt.normalize()
    parsed = panel.exchange_contract_id.fillna("").str.extract(r"^510300([CP])(\d{4})M(\d{5})$")
    panel["historical_standard"] = parsed[0].notna() & parsed[0].eq(panel.option_type)
    panel["historical_strike"] = pd.to_numeric(parsed[2], errors="coerce") / 1000
    mismatch = panel.historical_standard & (abs(panel.historical_strike-panel.strike) > 1e-8)
    returns = np.log1p(pd.to_numeric(market.total_simple, errors="coerce"))
    for window in [5, 20, 60]:
        market[f"rv{window}"] = np.sqrt(returns.pow(2).rolling(window).mean()*242)
    market = market.set_index("date").sort_index()
    calendar = pd.DatetimeIndex(sorted(panel.trade_date.unique()))
    daily = {date: part.copy() for date, part in panel.groupby("trade_date", sort=True)}
    lookup = {(pd.Timestamp(row.trade_date), str(row.contract_code)): row for row in panel.itertuples(index=False)}
    dump(out / "data_summary.json", {
        "option_rows": len(eod), "risk_rows": len(risk), "calendar_days": len(calendar),
        "start": str(calendar.min().date()), "end": str(calendar.max().date()),
        "snapshot_adjusted_but_historical_standard_rows": int((panel.historical_standard & panel.is_adjusted).sum()),
        "snapshot_strike_disagrees_with_historical_standard_rows": int(mismatch.sum()),
        "history_reconstruction": "仅标准M行按当日官方交易代码还原；调整A行不推断单位",
    })
    return panel, market, calendar, daily, lookup


def select_pair(rows: pd.DataFrame, market_row: pd.Series):
    rows = rows.copy()
    rows["dte"] = (rows.expiry_date-rows.trade_date).dt.days
    valid = rows.historical_standard & rows.dte.between(30,75) & rows.volume.ge(100) & rows.open_interest.ge(500)
    valid &= rows.implied_volatility.between(.01,3) & rows.close.gt(0) & rows.delta.notna()
    rows = rows[valid].copy()
    calls = rows[rows.option_type.eq("C")]
    puts = rows[rows.option_type.eq("P")]
    pairs = calls.merge(puts, on=["expiry_date","historical_strike"], suffixes=("_C","_P"))
    if pairs.empty:
        return None, rows
    pairs["expiry_distance"] = abs(pairs.dte_C-45)
    pairs["strike_distance"] = abs(pairs.historical_strike-float(market_row.close))
    pairs = pairs.sort_values(["expiry_distance","strike_distance","expiry_date","historical_strike","contract_code_C","contract_code_P"], kind="stable")
    return pairs.iloc[0], rows


def price_fill(price: float, action: int, scenario: str) -> float:
    if scenario == "DAY_EXTREME":
        return price
    tick, fraction = (.0002,.005) if scenario == "BASE" else (.0005,.01)
    return max(0.0, price + action*max(tick, fraction*price))


def build_samples(out: Path):
    panel, market, calendar, daily, lookup = load_data(out)
    sample_rows, legs_rows = [], []
    for i in range(1, len(calendar)-10):
        decision, observed, entry, exit_date = calendar[i], calendar[i-1], calendar[i+1], calendar[i+10]
        if observed not in market.index:
            continue
        m = market.loc[observed]
        if not np.isfinite(m[["rv5","rv20","rv60"]].to_numpy(float)).all() or min(m.rv5,m.rv20,m.rv60)<=0:
            continue
        pair, eligible = select_pair(daily[observed], m)
        common = {"decision_date": str(decision.date()), "observed_date": str(observed.date()), "entry_date": str(entry.date()),
                  "label_end_date": str(exit_date.date()), "entry_i": i+1, "exit_i": i+10}
        if pair is None:
            for structure in STRUCTURES:
                sample_rows.append({**common,"structure":structure,"label_status":"NO_PRIOR_ELIGIBLE_ATM_PAIR"})
            continue
        center = float(pair.historical_strike)
        expiry = pair.expiry_date
        atm_iv = (float(pair.implied_volatility_C)+float(pair.implied_volatility_P))/2
        features = dict(zip(FEATURES, [np.log(atm_iv), np.log(atm_iv/m.rv20), np.log(m.rv5/m.rv20), np.log(m.rv20/m.rv60),
                                       (float(pair.implied_volatility_P)-float(pair.implied_volatility_C))/atm_iv]))
        central = [{"contract_code":str(pair.contract_code_C),"option_type":"C","strike":center,"side":1},
                   {"contract_code":str(pair.contract_code_P),"option_type":"P","strike":center,"side":1}]
        same = eligible[eligible.expiry_date.eq(expiry)]
        lower = same[same.option_type.eq("P") & same.historical_strike.lt(center)].copy()
        upper = same[same.option_type.eq("C") & same.historical_strike.gt(center)].copy()
        wings = None
        if not lower.empty and not upper.empty:
            lower["distance"] = abs(lower.historical_strike-center*.95)
            upper["distance"] = abs(upper.historical_strike-center*1.05)
            lo = lower.sort_values(["distance","historical_strike","contract_code"],kind="stable").iloc[0]
            hi = upper.sort_values(["distance","historical_strike","contract_code"],kind="stable").iloc[0]
            wings = [{"contract_code":str(lo.contract_code),"option_type":"P","strike":float(lo.historical_strike),"side":1},
                     {"contract_code":str(hi.contract_code),"option_type":"C","strike":float(hi.historical_strike),"side":1}]
        for structure in STRUCTURES:
            row = {**common,**features,"structure":structure,"expiry_date":str(expiry.date()),"center_strike":center,
                   "atm_iv":atm_iv,"rv20":float(m.rv20),"label_status":"READY"}
            legs = [dict(x) for x in central]
            if structure == "PROTECTED_SHORT_ATM_STRADDLE":
                if wings is None:
                    row["label_status"]="NO_PRIOR_ELIGIBLE_PROTECTION_WINGS"
                    sample_rows.append(row)
                    continue
                for leg in legs:
                    leg["side"]=-1
                legs += [dict(x) for x in wings]
            prior_debit = sum(x["side"]*float(lookup[(observed,x["contract_code"])].close)*10000 for x in legs)
            round_fees = 10*len(legs)
            if structure == "LONG_ATM_STRADDLE":
                scale = prior_debit+round_fees
            else:
                scale = max(center-wings[0]["strike"],wings[1]["strike"]-center)*10000+prior_debit+round_fees
            row["known_risk_scale_cny"] = scale
            row["prior_net_delta"] = sum(x["side"]*float(lookup[(observed,x["contract_code"])].delta) for x in legs)
            if scale<=0 or not np.isfinite(scale):
                row["label_status"]="INVALID_PRIOR_RISK_SCALE"
            for leg_number,leg in enumerate(legs):
                leg_row = {**common,"structure":structure,"leg_number":leg_number,**leg,"historical_unit":10000,
                           "prior_close":float(lookup[(observed,leg["contract_code"])].close)}
                for label,date in [("entry",entry),("exit",exit_date)]:
                    quote = lookup.get((date,leg["contract_code"]))
                    if quote is not None:
                        for field in ["open","close","high","low","volume","exchange_contract_id"]:
                            leg_row[f"{label}_{field}"] = getattr(quote,field)
                legs_rows.append(leg_row)
                for date in calendar[i+1:i+11]:
                    q = lookup.get((date,leg["contract_code"]))
                    if q is None or not q.historical_standard or abs(q.historical_strike-leg["strike"])>1e-8:
                        row["label_status"]="UNKNOWN_ADJUSTMENT_OR_MISSING_DAILY_ID"
                        break
                for date in [entry,exit_date]:
                    q = lookup.get((date,leg["contract_code"]))
                    if q is None:
                        row["label_status"]="UNKNOWN_MISSING_ENTRY_OR_EXIT"
                    elif q.volume<=0 or not np.isfinite([q.open,q.close,q.high,q.low]).all() or min(q.open,q.close,q.high,q.low)<=0:
                        row["label_status"]="UNKNOWN_NO_TRADE_OR_INVALID_PRICE"
                if expiry<=exit_date:
                    row["label_status"]="UNKNOWN_EXPIRY_WITHIN_HORIZON"
            if row["label_status"]=="READY":
                for scenario in ["BASE","STRESS","DAY_EXTREME"]:
                    pnl = -round_fees
                    for leg in legs:
                        q0,q1 = lookup[(entry,leg["contract_code"])],lookup[(exit_date,leg["contract_code"])]
                        side=leg["side"]
                        if scenario=="DAY_EXTREME":
                            opening = q0.high if side>0 else q0.low
                            closing = q1.low if side>0 else q1.high
                        else:
                            opening=price_fill(q0.open,side,scenario)
                            closing=price_fill(q1.close,-side,scenario)
                        pnl += side*(closing-opening)*10000
                    row[f"pnl_{scenario}_cny"] = float(pnl)
                    row[f"target_{scenario}"] = float(pnl/scale)
                row["target_profit"]=int(row["pnl_STRESS_cny"]>0)
            sample_rows.append(row)
    samples=pd.DataFrame(sample_rows)
    legs=pd.DataFrame(legs_rows)
    csv(out/"results/全部组合样本与不可评价原因.csv",samples)
    csv(out/"results/完整候选合约腿与报价.csv",legs)
    csv(out/"results/样本覆盖计数.csv",samples.groupby(["structure","label_status"]).size().rename("count").reset_index())
    print(f"组合样本构建完成：{len(samples)}条，标签可评价{int(samples.label_status.eq('READY').sum())}条。",flush=True)
    return samples


def uniqueness(sample: pd.DataFrame) -> np.ndarray:
    start,end=int(sample.entry_i.min()),int(sample.exit_i.max())
    changes=np.zeros(end-start+2)
    np.add.at(changes,sample.entry_i.to_numpy(int)-start,1)
    np.add.at(changes,sample.exit_i.to_numpy(int)-start+1,-1)
    active=np.cumsum(changes[:-1])
    inverse=np.divide(1.,active,out=np.zeros_like(active),where=active>0)
    sums=np.r_[0,np.cumsum(inverse)]
    return (sums[sample.exit_i.to_numpy(int)-start+1]-sums[sample.entry_i.to_numpy(int)-start])/(sample.exit_i-sample.entry_i+1).to_numpy()


def fit_bundle(sample: pd.DataFrame, fit_date: str) -> dict:
    x=sample[FEATURES].to_numpy(float)
    y=sample.target_STRESS.to_numpy(float)
    w=uniqueness(sample)
    mean=np.average(x,axis=0,weights=w)
    std=np.sqrt(np.average((x-mean)**2,axis=0,weights=w))
    std=np.maximum(std,1e-8)
    z=np.clip((x-mean)/std,-3,3)
    p=float(np.average(y>0,weights=w))
    empirical={"p":p}
    for name,mask in [("gain",y>0),("loss",y<0)]:
        empirical[name]=float(np.average(abs(y[mask]),weights=w[mask])) if mask.any() else 0.0
    model={}
    convergences=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always",ConvergenceWarning)
        if 0<p<1:
            fit=LogisticRegression(C=1/w.sum(),solver="lbfgs",max_iter=2000,tol=1e-8)
            fit.fit(z,y>0,sample_weight=w)
            model["p"]={"kind":"LOGIT","coef":fit.coef_[0].tolist(),"intercept":float(fit.intercept_[0])}
        else:
            model["p"]={"kind":"EMPIRICAL","value":p}
        for name,mask in [("gain",y>0),("loss",y<0)]:
            if mask.sum()>=20:
                fit=GammaRegressor(alpha=1,fit_intercept=True,max_iter=2000,tol=1e-8)
                fit.fit(z[mask],abs(y[mask]),sample_weight=w[mask])
                model[name]={"kind":"GAMMA","coef":fit.coef_.tolist(),"intercept":float(fit.intercept_)}
            else:
                model[name]={"kind":"EMPIRICAL","value":empirical[name]}
        convergences=[str(item.message) for item in caught]
    if convergences:
        raise RuntimeError("数值优化未收敛："+"；".join(convergences))
    return {"fit_date":fit_date,"n_train":len(sample),"effective_weight_sum":float(w.sum()),
            "train_dates":sample.decision_date.tolist(),"latest_label_end":sample.label_end_date.max(),
            "feature_mean":mean.tolist(),"feature_std":std.tolist(),"empirical":empirical,"model":model}


def predict(bundle: dict, features: np.ndarray, model: str):
    if model=="MATURE_EMPIRICAL":
        return [bundle["empirical"][k] for k in ["p","gain","loss"]]
    z=np.clip((features-np.array(bundle["feature_mean"]))/np.array(bundle["feature_std"]),-3,3)
    output=[]
    for key in ["p","gain","loss"]:
        head=bundle["model"][key]
        if head["kind"]=="EMPIRICAL":
            value=head["value"]
        else:
            score=float(z@np.array(head["coef"])+head["intercept"])
            value=float(expit(score) if key=="p" else np.exp(score))
        output.append(value)
    return output


def train(out: Path,samples: pd.DataFrame):
    ready=samples[samples.label_status.eq("READY")].copy()
    predictions=[]
    bundles=[]
    for structure in STRUCTURES:
        part=samples[samples.structure.eq(structure) & samples.known_risk_scale_cny.gt(0) & samples[FEATURES].notna().all(axis=1)].sort_values("decision_date")
        training_pool=ready[ready.structure.eq(structure)].sort_values("decision_date")
        active_bundle=None
        active_month=None
        for row in part.itertuples(index=False):
            month=row.decision_date[:7]
            if month!=active_month:
                mature=training_pool[training_pool.label_end_date.lt(row.decision_date)]
                if len(mature)<252:
                    continue
                active_bundle=fit_bundle(mature,row.decision_date)
                active_bundle["structure"]=structure
                active_bundle["bundle_id"]=len(bundles)
                bundles.append(active_bundle)
                active_month=month
                if len(bundles)%20==0:
                    print(f"已训练{len(bundles)}组月度模型，当前{structure} {month}。",flush=True)
            if active_bundle is None:
                continue
            for model in MODELS:
                p,w,l=predict(active_bundle,np.array([getattr(row,c) for c in FEATURES]),model)
                record=row._asdict()
                record.update(model=model,bundle_id=active_bundle["bundle_id"],fit_date=active_bundle["fit_date"],
                              n_train=active_bundle["n_train"],effective_weight_sum=active_bundle["effective_weight_sum"],
                              p_win=p,conditional_gain=w,conditional_loss=l,expected_net_risk_unit=p*w-(1-p)*l,
                              predicted_payoff_ratio=w/l if l>0 else None)
                predictions.append(record)
    frame=pd.DataFrame(predictions)
    dump(out/"models/全部月度参数.json",bundles)
    csv(out/"results/完整滚动预测.csv",frame)
    return frame,bundles


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions=predictions[predictions.label_status.eq("READY")].copy()
    records=[]
    for (structure,model),all_rows in predictions.groupby(["structure","model"]):
        masks={"ALL":np.ones(len(all_rows),bool),"THROUGH_2023":all_rows.decision_date.lt("2024-01-01"),"FROM_2024":all_rows.decision_date.ge("2024-01-01")}
        for period,mask in masks.items():
            d=all_rows[mask]
            if d.empty:
                continue
            y=d.target_STRESS.to_numpy(float)
            p=d.p_win.to_numpy(float)
            gain,loss=y>0,y<0
            records.append({"structure":structure,"model":model,"period":period,"n":len(d),
                            "start":d.decision_date.min(),"end":d.decision_date.max(),
                            "observed_profit_fraction":float(np.mean(gain)),
                            "mean_predicted_profit_probability":float(p.mean()),
                            "brier":float(np.mean((p-gain)**2)),
                            "auc":float(roc_auc_score(gain,p)) if gain.any() and (~gain).any() else None,
                            "gain_mse":float(np.mean((d.conditional_gain.to_numpy()[gain]-y[gain])**2)) if gain.any() else None,
                            "loss_mse":float(np.mean((d.conditional_loss.to_numpy()[loss]+y[loss])**2)) if loss.any() else None,
                            "expected_mse":float(np.mean((d.expected_net_risk_unit.to_numpy()-y)**2)),
                            "mean_net_pnl_per_candidate_cny":float(d.pnl_STRESS_cny.mean()),
                            "mean_net_risk_unit":float(np.mean(y)),
                            "mean_positive_risk_unit":float(np.mean(y[gain])) if gain.any() else None,
                            "mean_negative_risk_unit_magnitude":float(np.mean(-y[loss])) if loss.any() else None})
    return pd.DataFrame(records)


def finish(out: Path,samples: pd.DataFrame,predictions: pd.DataFrame,bundles: list):
    metrics=evaluate(predictions)
    csv(out/"results/概率与盈亏幅度分项评价.csv",metrics)
    base=metrics[metrics.model.eq("MATURE_EMPIRICAL")]
    trained=metrics[metrics.model.eq("VOLATILITY_LOGIT_GAMMA")]
    compare=trained.merge(base,on=["structure","period"],suffixes=("_trained","_baseline"),validate="one_to_one")
    for metric in ["brier","gain_mse","loss_mse","expected_mse"]:
        compare[f"{metric}_improvement_fraction"]=1-compare[f"{metric}_trained"]/compare[f"{metric}_baseline"]
    csv(out/"results/固定经验基线对照.csv",compare)
    bins=[]
    for (structure,model),d in predictions[predictions.label_status.eq("READY")].groupby(["structure","model"]):
        groups=pd.cut(d.p_win,bins=[0,.4,.5,.6,.7,.8,1],include_lowest=True)
        for label,g in d.groupby(groups,observed=True):
            bins.append({"structure":structure,"model":model,"probability_bin":str(label),"n":len(g),
                         "predicted_p":float(g.p_win.mean()),"observed_p":float(g.target_profit.mean()),
                         "net_pnl_mean_cny":float(g.pnl_STRESS_cny.mean())})
    csv(out/"results/固定概率分箱.csv",pd.DataFrame(bins))
    summary={"study_id":STUDY,"completed_at":now(),"status":"FIRST_OPTION_VOLATILITY_PAYOFF_TRAINING_COMPLETE",
             "candidate_rows":len(samples),"scored_rows":int(samples.label_status.eq("READY").sum()),
             "prediction_rows":len(predictions),"monthly_bundles":len(bundles),
             "new_probability_fits":sum(b["model"]["p"]["kind"]=="LOGIT" for b in bundles),
             "new_amplitude_fits":sum(b["model"][k]["kind"]=="GAMMA" for b in bundles for k in ["gain","loss"]),
             "new_account_backtests":0,"new_downloads":0,"goal_achieved":False,
             "account_net_sharpe":"NOT_COMPUTED","account_drawdown":"NOT_COMPUTED",
             "annual_completed_trades":"NOT_COMPUTED","all_data_is_retrospective":True,
             "no_historical_bid_ask_execution_proof":True,
             "unknown_labels_not_counted_as_zero_pnl":True}
    dump(out/"summary.json",summary)
    lines=["# 510300期权：波动率组合第一轮训练", "", "用户已将研究范围改为只交易510300 ETF期权与现金；20万元、净夏普1.3、回撤10%、每完整年至少5次的既有目标继续保留。", "",
           "本轮完成买入跨式和带保护翼卖出跨式的盈利概率、盈利幅度、亏损幅度训练。没有生成资金账户，因此没有计算或宣称新的夏普、回撤和年度交易次数。",
           "", f"共{len(samples)}条候选结构记录，{summary['scored_rows']}条可计算日线代理标签；{len(predictions)}条滚动预测，{len(bundles)}组月度参数。两个模型为同成熟样本经验基线与固定正则化的波动率模型；没有参数网格搜索。", "",
           "## 固定对照", "", "下表为相对经验基线的误差减少比例，正数更好，负数更差。", "",
           "|结构|时段|样本数|概率Brier改善|盈利幅度MSE改善|亏损幅度MSE改善|净期望MSE改善|", "|---|---|---:|---:|---:|---:|---:|"]
    for row in compare.itertuples(index=False):
        lines.append(f"|{row.structure}|{row.period}|{row.n_trained}|{row.brier_improvement_fraction:.2%}|{row.gain_mse_improvement_fraction:.2%}|{row.loss_mse_improvement_fraction:.2%}|{row.expected_mse_improvement_fraction:.2%}|")
    lines += ["", "## 口径与下一步", "",
              "1. 价格来自已有日线，基础和压力成本属于假设。逐腿日内极端价是包络，不是同步组合成交价。不能用本轮候选标签均值直接当作可实现账户收益。",
              "2. 旧合约主表的调整标识和行权价存在后来快照回填。本轮只按当日官方交易代码识别M标准合约，恢复当日行权价和10000份单位；不使用未来是否调整来筛选。",
              "3. 跨调整或缺失关键日行情的标签保留未知。它们不能在未来资金回测里被事后剔除来避免亏损；需要先解决退出与估值，再评价完整账户。",
              "4. 全部特征滞后一交易日，标签成熟后才参与后续月度训练；十日标签重叠，采用逆重叠权重，但不因此宣称样本独立。2024年后的结果也是历史后段诊断，不是重新获得的全新留出集。",
              "5. 买入跨式承担时间损耗，保护性卖出跨式承担波动扩大和尾部价格变化。都保留净Delta方向敞口；没有现货或期货对冲，不宣称纯Gamma套利。",
              "6. 后续应在完整组合现金流与保证金框架下，检验预测是否改善真实可执行的净损益；不能把同时持有两边等同于稳定获利。",
              "", "官方规则参考：", "- https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250610_10781452.shtml", "- https://www.szse.cn/www/investor/institute/rules/t20230203_598565.html"]
    (out/"第一轮训练结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def verify(out: Path) -> dict:
    frozen=json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    for name,expected in frozen["files"].items():
        if digest(out/name)!=expected:
            raise ValueError(f"冻结文件发生变化：{name}")
    samples=pd.read_csv(out/"results/全部组合样本与不可评价原因.csv")
    legs=pd.read_csv(out/"results/完整候选合约腿与报价.csv")
    pred=pd.read_csv(out/"results/完整滚动预测.csv")
    bundles=json.loads((out/"models/全部月度参数.json").read_text(encoding="utf-8"))
    ready=samples[samples.label_status.eq("READY")]
    assert (samples.observed_date<samples.decision_date).all()
    assert (samples.entry_date>samples.decision_date).all()
    assert (samples.exit_i-samples.entry_i+1).eq(10).all()
    groups={(d,s):g for (d,s),g in legs.groupby(["decision_date","structure"])}
    for row in ready.itertuples(index=False):
        block=groups[(row.decision_date,row.structure)]
        assert len(block)==(2 if row.structure=="LONG_ATM_STRADDLE" else 4)
        for scenario in ["BASE","STRESS","DAY_EXTREME"]:
            pnl=-10*len(block)
            for leg in block.itertuples(index=False):
                if scenario=="DAY_EXTREME":
                    opening=leg.entry_high if leg.side>0 else leg.entry_low
                    closing=leg.exit_low if leg.side>0 else leg.exit_high
                else:
                    opening=price_fill(leg.entry_open,leg.side,scenario)
                    closing=price_fill(leg.exit_close,-leg.side,scenario)
                pnl+=leg.side*(closing-opening)*10000
            np.testing.assert_allclose(pnl,getattr(row,f"pnl_{scenario}_cny"),atol=1e-7,rtol=1e-9)
            np.testing.assert_allclose(pnl/row.known_risk_scale_cny,getattr(row,f"target_{scenario}"),atol=1e-10,rtol=1e-9)
    for b in bundles:
        assert b["latest_label_end"]<b["fit_date"]
        mature=ready[ready.structure.eq(b["structure"]) & ready.label_end_date.lt(b["fit_date"])]
        assert len(mature)==b["n_train"] and mature.decision_date.tolist()==b["train_dates"]
    for row in pred.itertuples(index=False):
        b=bundles[int(row.bundle_id)]
        p,w,l=predict(b,np.array([getattr(row,c) for c in FEATURES]),row.model)
        np.testing.assert_allclose([p,w,l,p*w-(1-p)*l],[row.p_win,row.conditional_gain,row.conditional_loss,row.expected_net_risk_unit],rtol=1e-9,atol=1e-10)
    saved=pd.read_csv(out/"results/概率与盈亏幅度分项评价.csv")
    actual=evaluate(pred)
    for c in actual.select_dtypes(include=[np.number]).columns:
        np.testing.assert_allclose(saved[c],actual[c],rtol=1e-9,atol=1e-10,equal_nan=True)
    result={"status":"PASS_SAVED_LABELS_PREDICTIONS_AND_METRICS","checked_at":now(),"scored_rows":len(ready),
            "prediction_rows":len(pred),"new_fits":0,"new_accounts":0,"new_downloads":0}
    dump(out/"verification/验证结果.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description="仅用既有数据训练510300期权波动率组合，或只读复核已有结果。")
    parser.add_argument("action",choices=["run","verify"])
    parser.add_argument("--root",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.action=="verify":
        print(json.dumps(verify(args.root),ensure_ascii=False,indent=2))
        return
    prepare(args.root)
    samples=build_samples(args.root)
    predictions,bundles=train(args.root,samples)
    if predictions.empty:
        raise RuntimeError("成熟样本不足，没有可比较预测；不得报告通过。")
    result=finish(args.root,samples,predictions,bundles)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    print(json.dumps(verify(args.root),ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
