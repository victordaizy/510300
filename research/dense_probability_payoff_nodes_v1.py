"""已有行情的成熟日样本训练及单类节点扩展；预测、节点和账户分阶段保存。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import sparse_event_training_core_v3 as core
import if_node_joint_payoff_training_v1 as joint
import probability_payoff_prediction_stage_v1 as stage


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_dense_probability_payoff_nodes_v1"
PRIMARY = "DENSE_IF_VOL_GAMMA"
NAMES = ["DENSE_EMPIRICAL", "DENSE_IF_EMPIRICAL", "DENSE_IF_GAMMA",
         "DENSE_IF_VOL_MEAN", PRIMARY, "DENSE_PRICE_VOL_GAMMA", "MONTHLY_IF_GAMMA"]


def freeze(out):
    core.require(not out.exists(), "本轮已存在，不覆盖。")
    prior = ROOT / "reports/research/510300_probability_payoff_prediction_stage_v1"
    original = ROOT / "reports/research/510300_if_exhaustion_existing_training_v1"
    cfg = core.load(original / "protocol.json")
    cfg.update(study_id="510300_DENSE_PROBABILITY_PAYOFF_NODES_V1", target_net_sharpe=1.3,
        user_instruction="请继续，达到目标再停止；先训练胜率与盈亏幅度，再逐步增加节点；不用采集。",
        hypothesis="月末抽样损失训练信息；已知波动水平应解释条件盈亏的尺度。两项增量分别用固定对照识别。",
        training_sample="全部data_eligible日；只纳入标签结束时点严格早于模型时点的十日标签。",
        overlap_weight="在当次成熟训练集内，计算每段十日标签的平均逆重叠数，作为样本权重；不声称标签独立。",
        model_clock="沿用100个月末预测原点，每月末更新一次，随后逐日固定参数；末次期末模型只保存不参与评价。",
        model_family="无加权类别再平衡的L2逻辑回归；条件Gamma均值回归；波动尺度为事前vol20*sqrt(10)。",
        loss_penalty=1.0, standardized_clip=3.0, net_label_cost_proxy=.0024,
        amplitude_variants="同一IF概率分别配原幅度经验均值、Gamma、波动缩放经验均值、带波动offset的Gamma；价格版作为辅助信息对照。",
        primary_model=PRIMARY, comparison_models=NAMES,
        fixed_comparisons=[(PRIMARY, "DENSE_IF_GAMMA"), (PRIMARY, "DENSE_IF_VOL_MEAN"),
            (PRIMARY, "DENSE_PRICE_VOL_GAMMA"), ("DENSE_IF_GAMMA", "MONTHLY_IF_GAMMA"),
            ("DENSE_IF_EMPIRICAL", "DENSE_EMPIRICAL")],
        node_stages={"S0_IF_ONLY": "原10个IF耗竭节点，日期不变。",
            "S1_PLUS_FORECAST_ONSET": "加入主模型的p>0.5、W>L、E>0联合状态由假转真的首日；不调阈值，不在持仓中叠加。"},
        node_interpretation="预测状态转正不等于已证明市场变盘；三个门槛并非三份独立信息，p>0.5且W>L已蕴含E>0。",
        node_timing="每日15:01使用当时特征和最近一次月末模型；下一交易日开盘，固定第10日收盘；模型更新触发也明确标记。",
        prior_node_state="首个有模型日前及数据缺口前状态未知，只有相邻完整交易日可触发转正，不把第一次得到模型自动算新增节点。",
        evaluation="月末、原IF节点和全部日预测分开；新节点只在完成预测保存后产生并独立报告。日预测的十日标签相关。",
        annual_frequency_is_training_admission_gate=False, minimum_complete_cycles_each_full_calendar_year=5,
        no_parameter_grid=True, no_feature_direction_reversal=True,
        historical_results_already_seen=True, independent_holdout=False,
        branch_budget="本协议两项固定训练增量和一类节点扩展各做一次；不能因为未达1.3继续在本分支调整窗口或门槛。",
        orders_authorized=False, new_downloads=0, goal_achieved=False)
    cfg["account"].update(prediction_error_multiplier=0, min_mature_prequential_errors=0,
                          scope="与原100个月末评价的首日一致，截止最后可用IF信号十日标签结束；全程包括空仓。")
    out.mkdir(parents=True)
    sources = {}

    def take(source, name):
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        sources[name] = {"source": str(source.relative_to(ROOT)), "sha256": core.digest(dest)}

    for name in ["market.parquet", "monthly.csv", "events.csv", "previous_models.json"]:
        take(prior / "inputs" / name, "inputs/" + name)
    take(prior / "models/逐期直接均值模型.json", "inputs/previous_gamma_models.json")
    for name in ["if_saved_features.parquet", "dividends.csv"]:
        source = original / "inputs" / name
        core.require(core.digest(source) == core.load(original / "freeze.json")["identities"]["inputs/" + name], "原冻结来源改变。")
        take(source, "inputs/" + name)
    for name in ["protocol.json", "summary.json", "stage_decision.json", "delivery_receipt.json"]:
        take(prior / name, "inputs/previous_" + name)
    for name in [Path(__file__).name, "sparse_event_training_core_v3.py", "if_node_joint_payoff_training_v1.py",
                 "if_exhaustion_existing_training_v1.py", "if_exhaustion_local_features_v1.py", "probability_payoff_prediction_stage_v1.py"]:
        take(ROOT / "research" / name, "code/" + name)
    for name in ["510300_existing_data_training_mandate_v1.json", "510300_sparse_node_training_contract_v1.json"]:
        source = ROOT / "config" / name
        take(source, "inputs/" + name)
        value = core.load(source)
        value.update(latest_user_instruction=cfg["user_instruction"], continuation_requested_at=core.now(),
                     current_round=cfg["study_id"], goal_achieved=False)
        core.save(source, value)
        core.save(out / name, value)
    core.save(out / "protocol.json", cfg)
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_new_fits_and_node_outcomes": True,
        "historical_prices_already_seen": True, "independent_holdout": False, "sources": sources,
        "identities": {"protocol.json": core.digest(out / "protocol.json"), **{k:v["sha256"] for k,v in sources.items()}}})
    print("已固定成熟日样本、波动尺度和一类预测状态节点；开始训练。", flush=True)


def check_inputs(out):
    amendment_path = out / "implementation_amendment.json"
    amendment = core.load(amendment_path) if amendment_path.exists() else None
    for name, identity in core.load(out / "freeze.json")["identities"].items():
        if amendment is not None and name == amendment["replaced_path"]:
            core.require(core.digest(out / amendment["preserved_original_path"]) == identity == amendment["old_sha256"], "原冻结代码副本不同。")
            identity = amendment["new_sha256"]
        core.require(core.digest(out / name) == identity, "冻结文件改变：" + name)


def rows(out, cfg):
    market = core.read_market(out)
    features = pd.read_parquet(out / "inputs/if_saved_features.parquet").sort_values("date")
    features["basis_oi_interaction"] = features.basis_residual_percentile * features.open_interest_shock_percentile
    columns = sum(cfg["features"].values(), [])
    calendar = pd.DatetimeIndex(market.date)
    data = []
    for f in features.loc[features.data_eligible & (features.date <= cfg["signal_end"])].itertuples():
        origin = int(calendar.get_loc(f.date))
        entry, end = origin+1, origin+cfg["horizon_sessions"]
        core.require(end < len(market), "已有行情不足以成熟本轮日样本。")
        block = market.iloc[entry:end+1]
        gross = (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0])-1
        y = gross-cfg["net_label_cost_proxy"]
        record = {"month": str(f.date.date()), "decision_at": (f.date.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15, minutes=1)).isoformat(),
            "entry_i": entry, "exit_i": end, "entry_date": str(calendar[entry].date()), "exit_date": str(calendar[end].date()),
            "label_end_at": (calendar[end].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15)).isoformat(),
            "reference_close": float(market.close.iloc[origin]), "vol20": float(market.vol20.iloc[origin]),
            "target10": gross, "target_net_proxy": y, "target_profit": int(y > 0),
            "if_event": bool(f.pressure_exhaustion_event), "feature_source_date": str(f.feature_source_date.date())}
        record["known_scale"] = record["vol20"]*math.sqrt(cfg["horizon_sessions"])
        record.update({col: float(getattr(f, col)) for col in columns})
        core.require(record["feature_source_date"] < record["month"], "IF特征未保持前一交易日可用。")
        data.append(record)
    frame = pd.DataFrame(data)
    core.require(np.isfinite(frame[columns+["known_scale", "target_net_proxy"]].to_numpy()).all(), "日样本含缺失或非有限值。")
    core.require((frame.known_scale > 0).all(), "波动尺度非正。")
    old = pd.read_csv(out / "inputs/monthly.csv")
    matched = frame.set_index("month").loc[old.month]
    np.testing.assert_allclose(matched[columns+["target_net_proxy"]], old[columns+["target_net_proxy"]], atol=1e-12, rtol=0)
    original_events = pd.read_csv(out / "inputs/events.csv")
    core.require(frame.loc[frame.if_event, "month"].tolist() == original_events.month.tolist(), "原IF节点日期改变。")
    return frame, market.iloc[:int(frame.exit_i.max())+1].copy()


def weights(sample):
    left = int(sample.entry_i.min())
    right = int(sample.exit_i.max())
    diff = np.zeros(right-left+2)
    np.add.at(diff, sample.entry_i.to_numpy(int)-left, 1)
    np.add.at(diff, sample.exit_i.to_numpy(int)-left+1, -1)
    concurrent = np.cumsum(diff)[:-1]
    inverse = np.divide(1., concurrent, out=np.zeros_like(concurrent), where=concurrent > 0)
    integral = np.r_[0., np.cumsum(inverse)]
    result = (integral[sample.exit_i.to_numpy(int)-left+1]-integral[sample.entry_i.to_numpy(int)-left])/(sample.exit_i-sample.entry_i+1).to_numpy()
    core.require(np.isfinite(result).all() and (result > 0).all(), "重叠权重无效。")
    return result


def design(sample, columns, weight=None, transform=None):
    raw = sample[columns].to_numpy(float)
    if transform is None:
        center = np.average(raw, axis=0, weights=weight)
        scale = np.sqrt(np.average((raw-center)**2, axis=0, weights=weight))
        active = scale >= 1e-12
        transform = {"columns": columns, "center": center.tolist(), "scale": np.where(active, scale, 1).tolist(), "active": active.tolist()}
    z = np.clip((raw-np.array(transform["center"]))/np.array(transform["scale"]), -3, 3)
    z[:, ~np.array(transform["active"], bool)] = 0.
    return np.column_stack([np.ones(len(sample)), z]), transform


def fit_head(sample, columns, weight, kind, positive=None, scaled=False):
    selected = np.ones(len(sample), bool) if kind == "LOGISTIC" else (sample.target_net_proxy > 0 if positive else sample.target_net_proxy < 0).to_numpy()
    part, w = sample.loc[selected], weight[selected]
    core.require(len(part) > 0, "缺少条件方向，不能拟合。")
    w = w/w.sum()
    x, transform = design(part, columns, w)
    y = part.target_profit.to_numpy(float) if kind == "LOGISTIC" else np.abs(part.target_net_proxy.to_numpy(float))
    if scaled:
        y = y/part.known_scale.to_numpy(float)
    penalty = np.r_[0., np.ones(len(columns))]
    mean = float(w@y)
    initial = np.r_[math.log(mean/(1-mean)) if kind == "LOGISTIC" else math.log(mean), np.zeros(len(columns))]

    def objective(beta):
        eta = x@beta
        if kind == "LOGISTIC":
            loss = w@(np.logaddexp(0, eta)-y*eta)
            gradient = x.T@(w*(expit(eta)-y))
        else:
            ratio = y*np.exp(-eta)
            loss = w@(ratio+eta)
            gradient = x.T@(w*(1-ratio))
        return float(loss+.5*np.sum(penalty*beta**2)), gradient+penalty*beta

    fit = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter":1000, "gtol":1e-10, "ftol":1e-14})
    error = float(np.max(abs(objective(fit.x)[1])))
    # 两种目标均凸；保持原1e-6方程精度，避免以线搜索停止标志否定已收敛的解。
    core.require(np.isfinite(fit.x).all() and np.isfinite(fit.fun) and error < 1e-6, "固定模型未通过原定梯度收敛标准。")
    return {**transform, "kind": kind, "positive": positive, "scaled": scaled, "n": len(part), "beta":fit.x.tolist(),
            "gradient_max":error,"optimizer_success_flag":bool(fit.success),"optimizer_message":str(fit.message)}


def fit_bundle(sample, cfg, at):
    w = weights(sample)
    price = cfg["features"]["price"]
    both = price+cfg["features"]["if_joint"]
    head = {"p_if": fit_head(sample, both, w, "LOGISTIC"), "p_price": fit_head(sample, price, w, "LOGISTIC")}
    for prefix, columns, scaled in [("if_raw", both, False), ("if_scaled", both, True), ("price_scaled", price, True)]:
        for label, positive in [("gain", True), ("loss", False)]:
            head[prefix+"_"+label] = fit_head(sample, columns, w, "GAMMA", positive, scaled)
    empirical = {"p": float(np.average(sample.target_profit, weights=w))}
    for label, positive in [("gain", True), ("loss", False)]:
        mask = (sample.target_net_proxy > 0 if positive else sample.target_net_proxy < 0).to_numpy()
        y = abs(sample.target_net_proxy.to_numpy(float)[mask])
        empirical[label] = float(np.average(y, weights=w[mask]))
        empirical[label+"_scaled"] = float(np.average(y/sample.known_scale.to_numpy(float)[mask], weights=w[mask]))
    return {"as_of": at, "n_train":len(sample), "train_dates":sample.month.tolist(), "train_max_label_end_at":sample.label_end_at.max(),
        "sum_uniqueness_weights":float(w.sum()), "weight_sha256":hashlib.sha256(w.astype("<f8").tobytes()).hexdigest(),
        "heads":head, "empirical":empirical}


def predict_bundle(bundle, part):
    result = {}
    for key, h in bundle["heads"].items():
        x, _ = design(part, h["columns"], transform=h)
        raw = x@h["beta"]
        result[key] = expit(raw) if h["kind"] == "LOGISTIC" else np.exp(raw)*(part.known_scale.to_numpy(float) if h["scaled"] else 1.)
    e = bundle["empirical"]
    definitions = {"DENSE_EMPIRICAL":(e["p"], e["gain"], e["loss"]),
        "DENSE_IF_EMPIRICAL":(result["p_if"],e["gain"],e["loss"]),
        "DENSE_IF_GAMMA":(result["p_if"],result["if_raw_gain"],result["if_raw_loss"]),
        "DENSE_IF_VOL_MEAN":(result["p_if"],e["gain_scaled"]*part.known_scale.to_numpy(float),e["loss_scaled"]*part.known_scale.to_numpy(float)),
        PRIMARY:(result["p_if"],result["if_scaled_gain"],result["if_scaled_loss"]),
        "DENSE_PRICE_VOL_GAMMA":(result["p_price"],result["price_scaled_gain"],result["price_scaled_loss"])}
    outputs = []
    for name, (p,gain,loss) in definitions.items():
        result_frame = part.copy()
        result_frame["model"], result_frame["fit_as_of"] = name, bundle["as_of"]
        result_frame["n_train"], result_frame["sum_uniqueness_weights"] = bundle["n_train"], bundle["sum_uniqueness_weights"]
        result_frame["p_win"],result_frame["conditional_gain"],result_frame["conditional_loss"] = p,gain,loss
        result_frame["expected_net_proxy"] = p*gain-(1-p)*loss
        result_frame["predicted_payoff_ratio"] = gain/loss
        result_frame["joint_gate_pass"] = (result_frame.p_win > .5)&(result_frame.conditional_gain > result_frame.conditional_loss)&(result_frame.expected_net_proxy > 0)
        outputs.append(result_frame)
    return outputs


def prediction_metrics(predictions, monthly_dates):
    scoped = []
    for scope, mask in [("MONTHLY", predictions.month.isin(monthly_dates)), ("IF_NODE", predictions.if_event), ("DAILY_CORRELATED", np.ones(len(predictions), bool))]:
        part = predictions.loc[mask].copy()
        part["scope"] = scope
        scoped.append(part)
    metrics, _ = stage.score_predictions(pd.concat(scoped, ignore_index=True))
    return metrics


def comparisons(metrics, cfg):
    result = []
    for left, right in cfg["fixed_comparisons"]:
        joined = metrics.loc[metrics.model == left].merge(metrics.loc[metrics.model == right], on=["scope", "segment", "head"], suffixes=("", "_reference"), validate="one_to_one")
        for r in joined.itertuples():
            result.append({"model":left,"reference":right,"scope":r.scope,"segment":r.segment,"head":r.head,"n":r.n,
                           "mse_improvement":1-r.mse/r.mse_reference if r.mse_reference > 0 else None,"mae_difference":r.mae-r.mae_reference})
    return pd.DataFrame(result)


def reconstruct_predictions(out, frame, bundles):
    old = {m["prediction_month"]:m for m in core.load(out / "inputs/previous_models.json")
           if m["model"] == "B_IF_JOINT" and m["prediction_role"] == "MONTHLY_CALIBRATION"}
    gamma = {m["prediction_month"]:m for m in core.load(out / "inputs/previous_gamma_models.json")
             if m["model"] == "GAMMA_IF" and m["prediction_role"] == "MONTHLY_CALIBRATION"}
    outputs = []
    for i, bundle in enumerate(bundles):
        right = bundles[i+1]["as_of"] if i+1 < len(bundles) else "9999"
        part = frame.loc[(frame.decision_at >= bundle["as_of"]) & (frame.decision_at < right)].copy()
        outputs.extend(predict_bundle(bundle, part))
        month = bundle["as_of"][:10]
        control = part.copy()
        control["model"], control["fit_as_of"] = "MONTHLY_IF_GAMMA", bundle["as_of"]
        control["n_train"], control["sum_uniqueness_weights"] = old[month]["n_train"], old[month]["n_train"]
        phead = old[month]["probability"]
        x,_ = joint.design(part, phead["columns"], phead)
        p = expit(x@phead["beta"])
        values = []
        for key in ["gain", "loss"]:
            h = gamma[month][key]
            x,_ = joint.design(part,h["columns"],h)
            values.append(np.exp(x@h["beta"]))
        gain, loss = values
        control["p_win"],control["conditional_gain"],control["conditional_loss"] = p,gain,loss
        control["expected_net_proxy"],control["predicted_payoff_ratio"] = p*gain-(1-p)*loss,gain/loss
        control["joint_gate_pass"] = (p > .5)&(gain > loss)&(control.expected_net_proxy > 0)
        outputs.append(control)
    return pd.concat(outputs, ignore_index=True).sort_values(["model", "decision_at"], ignore_index=True)


def train(out):
    core.require(not (out / "prediction_summary.json").exists(), "训练已完成，不重复拟合。")
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    frame, market = rows(out, cfg)
    old = [m for m in core.load(out / "inputs/previous_models.json") if m["model"] == "B_IF_JOINT" and m["prediction_role"] == "MONTHLY_CALIBRATION"]
    bundles = []
    for i,m in enumerate(old):
        sample = frame.loc[frame.label_end_at < m["decision_at"]]
        bundles.append(fit_bundle(sample,cfg,m["decision_at"]))
        if (i+1)%20 == 0:
            print(f"已完成{i+1}/{len(old)}个月末：概率和正负幅度共{8*(i+1)}次新拟合。", flush=True)
    final_at = frame.decision_at.max()
    final = fit_bundle(frame.loc[frame.label_end_at < final_at],cfg,final_at)
    core.save(out / "models/月末模型.json",bundles)
    core.save(out / "models/期末参考模型.json",final)
    core.export(out / "results/全部成熟日样本.csv",frame)
    predictions = reconstruct_predictions(out,frame,bundles)
    core.export(out / "results/完整逐日预测.csv",predictions)
    metrics = prediction_metrics(predictions,{m["prediction_month"] for m in old})
    core.export(out / "results/预测分项评价.csv",metrics)
    core.export(out / "results/固定预测对照.csv",comparisons(metrics,cfg))
    summary = {"status":"DENSE_PREDICTION_TRAINING_COMPLETE_BEFORE_NODE_EXPANSION","completed_at":core.now(),
        "training_rows":len(frame),"first_training_date":frame.month.min(),"last_training_date":frame.month.max(),
        "monthly_fits":len(bundles),"new_probability_fits":2*(len(bundles)+1),"new_conditional_mean_fits":6*(len(bundles)+1),
        "prediction_rows":len(predictions),"evaluation_days":predictions.month.nunique(),"monthly_evaluation_origins":len(old),
        "last_monthly_train_rows":bundles[-1]["n_train"],"last_monthly_uniqueness_weight_sum":bundles[-1]["sum_uniqueness_weights"],
        "label_overlap_is_not_independence":True,"new_node_types":0,"new_accounts":0,"new_downloads":0,"goal_achieved":False}
    core.save(out / "prediction_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def build_nodes(predictions):
    primary = predictions.loc[predictions.model == PRIMARY].sort_values("decision_at").copy()
    adjacent = primary.entry_i.diff().eq(1)
    onset = primary.joint_gate_pass & ~primary.joint_gate_pass.shift(1,fill_value=False) & adjacent
    primary["forecast_onset"] = onset
    primary["onset_at_model_refresh"] = onset & primary.fit_as_of.ne(primary.fit_as_of.shift(1))
    primary["new_only"] = onset & ~primary.if_event
    primary["S0_IF_ONLY"] = primary.if_event
    primary["S1_PLUS_FORECAST_ONSET"] = primary.if_event | onset
    return primary


def frequency_table(nav, trades, cfg):
    rows, result = [], []
    for key, part in nav.groupby(["node_stage","model","scenario"],sort=True):
        cycle = trades.loc[(trades.node_stage == key[0])&(trades.model == key[1])&(trades.scenario == key[2])]
        years = pd.to_datetime(part.date).dt.year
        first,last = int(years.min()),int(years.max())
        for year in range(first,last+1):
            n = int(cycle.entry_date.str[:4].eq(str(year)).sum())
            full = first < year < last
            rows.append({"node_stage":key[0],"model":key[1],"scenario":key[2],"year":year,"complete_calendar_year":full,
                         "completed_cycles_by_entry_year":n,"at_least_five":n >= 5 if full else None})
        complete = [r["completed_cycles_by_entry_year"] for r in rows if (r["node_stage"],r["model"],r["scenario"]) == key and r["complete_calendar_year"]]
        result.append({"node_stage":key[0],"model":key[1],"scenario":key[2],"complete_years":len(complete),
                       "minimum_cycles_in_full_year":min(complete) if complete else None,
                       "annual_frequency_pass":bool(complete and min(complete) >= cfg["minimum_complete_cycles_each_full_calendar_year"])})
    return pd.DataFrame(rows),pd.DataFrame(result)


def expand(out):
    core.require((out / "prediction_summary.json").exists(),"先完成概率与盈亏预测，再增加节点。")
    core.require(not (out / "summary.json").exists(),"扩点及账户已完成，不重复运行。")
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    _,market = rows(out,cfg)
    predictions = pd.read_csv(out / "results/完整逐日预测.csv")
    nodes = build_nodes(predictions)
    core.export(out / "results/逐日节点触发表.csv",nodes)
    scoped = []
    for label,mask in [("ORIGINAL_IF",nodes.if_event),("ADDED_FORECAST_ONLY",nodes.new_only),("ALL_FORECAST_ONSETS",nodes.forecast_onset)]:
        part = predictions.loc[predictions.month.isin(nodes.loc[mask,"month"])].copy()
        part["scope"] = label
        scoped.append(part)
    node_metric,_ = stage.score_predictions(pd.concat(scoped,ignore_index=True))
    core.export(out / "results/原节点与新增节点预测评价.csv",node_metric)
    chosen = [PRIMARY,"DENSE_IF_VOL_MEAN","DENSE_PRICE_VOL_GAMMA","MONTHLY_IF_GAMMA"]
    effective = copy.deepcopy(cfg)
    effective["models"] = {name:{} for name in chosen+["EVENT_ONLY"]}
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends = dividends.loc[dividends.symbol == "510300.SH"]
    start_i = int(predictions.entry_i.min())
    navs,cycles,choices,metrics = [],[],[],[]

    def account(node_stage,name,cost,source,event_only=False):
        selected = source.loc[source.model == name].copy()
        allowed = selected.joint_gate_pass if name in chosen else np.ones(len(selected),bool)
        admitted = selected.loc[allowed].copy()
        admitted["prediction"] = admitted.expected_net_proxy+cfg["net_label_cost_proxy"]
        admitted["prior_prediction_rmse"],admitted["mature_prequential_error_count"] = 0.,0
        nav,ledger,decisions,metric = core.account_run(name,cost,admitted,market,start_i,effective,dividends,event_only=event_only)
        nav["node_stage"] = node_stage
        navs.append(nav)
        for record in ledger:
            record["node_stage"] = node_stage
            cycles.append(record)
        decision_map = {r["month"]:r for r in decisions}
        for row in selected.itertuples():
            record = {"node_stage":node_stage,"model":name,"scenario":cost,"month":row.month,
                "entry_date":row.entry_date,"p_win":row.p_win,"conditional_gain":row.conditional_gain,
                "conditional_loss":row.conditional_loss,"expected_net_proxy":row.expected_net_proxy,"joint_gate_pass":row.joint_gate_pass,
                **decision_map.get(row.month,{"decision":"JOINT_GATE_REJECTED"})}
            choices.append(record)
        metric.update(node_stage=node_stage)
        metrics.append(joint.joint_metrics(metric,ledger))

    for node_stage in cfg["node_stages"]:
        selected = predictions.loc[predictions.month.isin(nodes.loc[nodes[node_stage],"month"])]
        for name in chosen:
            for cost in ["BASE","STRESS"]:
                account(node_stage,name,cost,selected)
        event_only = selected.loc[selected.model == PRIMARY].copy()
        event_only["model"] = "EVENT_ONLY"
        for cost in ["BASE","STRESS"]:
            account(node_stage,"EVENT_ONLY",cost,event_only,True)
        print(f"已完成{node_stage}的固定对照账户，原节点与新增节点分别保留。",flush=True)
    for name in ["CASH","BUY_AND_HOLD"]:
        for cost in ["BASE","STRESS"]:
            account("BENCHMARK",name,cost,predictions.iloc[:0])
    nav,trade,decision,metric = pd.concat(navs,ignore_index=True),pd.DataFrame(cycles),pd.DataFrame(choices),pd.DataFrame(metrics)
    annual,checks = frequency_table(nav,trade,cfg)
    metric = metric.merge(checks,on=["node_stage","model","scenario"],validate="one_to_one")
    metric["all_numeric_targets_met"] = metric.meets_numeric_targets & metric.annual_frequency_pass
    for name,frame in [("完整逐日账户",nav),("完整持仓周期",trade),("逐节点交易决定",decision),("账户联合评价",metric),("完整自然年交易次数",annual)]:
        core.export(out / "results" / (name+".csv"),frame)
    summary = {**core.load(out / "prediction_summary.json"),"status":"DENSE_TRAINING_AND_ONE_NODE_EXPANSION_COMPLETE",
        "completed_at":core.now(),"new_node_types":1,"original_if_nodes":int(nodes.if_event.sum()),
        "forecast_onsets":int(nodes.forecast_onset.sum()),"onsets_at_monthly_refresh":int(nodes.onset_at_model_refresh.sum()),
        "added_unique_nodes":int(nodes.new_only.sum()),"combined_nodes":int(nodes.S1_PLUS_FORECAST_ONSET.sum()),
        "new_accounts":len(metric),"primary_accounts":metric.loc[metric.model == PRIMARY].to_dict("records"),
        "numeric_passed_accounts":int(metric.all_numeric_targets_met.sum()),"model_selected_after_account_results":False,
        "new_downloads":0,"independent_validation_established":False,"goal_achieved":False,"orders_authorized":False}
    core.save(out / "summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def verify_bundle(bundle,frame):
    sample = frame.loc[frame.label_end_at < bundle["as_of"]]
    core.require(sample.month.tolist() == bundle["train_dates"] and len(sample) == bundle["n_train"],"成熟训练成员不同。")
    core.require(sample.label_end_at.max() == bundle["train_max_label_end_at"],"成熟标签最晚时点不同。")
    w = weights(sample)
    core.require(hashlib.sha256(w.astype("<f8").tobytes()).hexdigest() == bundle["weight_sha256"],"重叠权重不同。")
    core.require(abs(w.sum()-bundle["sum_uniqueness_weights"]) < 1e-10,"重叠权重合计不同。")
    for h in bundle["heads"].values():
        mask = np.ones(len(sample),bool) if h["kind"] == "LOGISTIC" else (sample.target_net_proxy > 0 if h["positive"] else sample.target_net_proxy < 0).to_numpy()
        part,weight = sample.loc[mask],w[mask]/w[mask].sum()
        core.require(h["n"] == len(part),"正负幅度训练样本不同。")
        x,transform = design(part,h["columns"],weight)
        for key in ["center","scale","active"]:
            np.testing.assert_allclose(h[key],transform[key],atol=1e-12,rtol=0)
        beta = np.array(h["beta"])
        if h["kind"] == "LOGISTIC":
            residual = expit(x@beta)-part.target_profit.to_numpy(float)
        else:
            y = abs(part.target_net_proxy.to_numpy(float))
            if h["scaled"]:
                y = y/part.known_scale.to_numpy(float)
            residual = 1-y*np.exp(-x@beta)
        gradient = x.T@(weight*residual)+np.r_[0.,np.ones(len(beta)-1)]*beta
        core.require(np.max(abs(gradient)) < 1e-6,"保存模型方程不能通过。")
    e = bundle["empirical"]
    core.require(abs(e["p"]-np.average(sample.target_profit,weights=w)) < 1e-12,"经验概率不同。")
    for label,positive in [("gain",True),("loss",False)]:
        mask = (sample.target_net_proxy > 0 if positive else sample.target_net_proxy < 0).to_numpy()
        y = abs(sample.target_net_proxy.to_numpy(float)[mask])
        for key,target in [(label,y),(label+"_scaled",y/sample.known_scale.to_numpy(float)[mask])]:
            core.require(abs(e[key]-np.average(target,weights=w[mask])) < 1e-12,"经验条件幅度不同。")


def verify_accounts(out,market,cfg,predictions):
    nodes = build_nodes(predictions)
    pd.testing.assert_frame_equal(nodes.reset_index(drop=True),pd.read_csv(out / "results/逐日节点触发表.csv"),check_dtype=False,rtol=1e-10,atol=1e-12)
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/完整持仓周期.csv")
    metrics = pd.read_csv(out / "results/账户联合评价.csv")
    calendar = market.iloc[int(predictions.entry_i.min()):].date.dt.strftime("%Y-%m-%d").tolist()
    price = market.set_index(market.date.dt.strftime("%Y-%m-%d")).close
    for key,part in nav.groupby(["node_stage","model","scenario"],sort=True):
        core.require(part.date.tolist() == calendar,"逐日账户漏掉空仓交易日。")
        local = trades.loc[(trades.node_stage == key[0])&(trades.model == key[1])&(trades.scenario == key[2])]
        row = metrics.loc[(metrics.node_stage == key[0])&(metrics.model == key[1])&(metrics.scenario == key[2])].iloc[0]
        equity = part.equity_cny.to_numpy(float)
        previous_equity = np.r_[cfg["capital_cny"],equity[:-1]]
        returns = equity/previous_equity-1
        peak = np.maximum.accumulate(np.r_[cfg["capital_cny"],equity])[1:]
        np.testing.assert_allclose(returns,part.daily_return,atol=1e-12,rtol=0)
        np.testing.assert_allclose(equity/peak-1,part.drawdown,atol=1e-12,rtol=0)
        np.testing.assert_allclose(part.cash_cny+part.shares*price.loc[part.date].to_numpy()+part.dividend_receivable_cny,equity,atol=1e-7,rtol=0)
        core.require((local.exit_i > local.entry_i).all(),"成交周期违反T+1。")
        core.require(len(local) == row.opportunities and abs(local.net_pnl_cny.sum()-(equity[-1]-cfg["capital_cny"])) < 1e-6,"完整周期与账户总利润不同。")
        position,cash = np.zeros(len(part)),np.full(len(part),cfg["capital_cny"],float)
        cash_changes = part.dividend_paid_cny.to_numpy(float).copy()
        calendar_index = {day:i for i,day in enumerate(calendar)}
        for r in local.itertuples():
            begin,end = calendar_index[r.entry_date],calendar_index[r.exit_date]
            position[begin:end] += r.shares
            cash_changes[begin] -= r.entry_cash_debit
            cash_changes[end] += r.shares*r.exit_fill_price-r.exit_commission_cny
            expected_profit = r.shares*r.exit_fill_price-r.exit_commission_cny+r.dividends_cny-r.entry_cash_debit
            core.require(abs(expected_profit-r.net_pnl_cny) < 1e-7,"周期逐笔现金利润不同。")
        np.testing.assert_allclose(position,part.shares,atol=0,rtol=0)
        cash += np.cumsum(cash_changes)
        np.testing.assert_allclose(cash,part.cash_cny,atol=1e-6,rtol=0)
        sharpe = returns.mean()/returns.std(ddof=1)*math.sqrt(cfg["annual_days"]) if returns.std(ddof=1) > 1e-14 else None
        core.require((sharpe is None and pd.isna(row.net_sharpe)) or (sharpe is not None and abs(sharpe-row.net_sharpe) < 1e-9),"全日历夏普不同。")
        core.require(abs(float(-part.drawdown.min())-row.max_drawdown) < 1e-12,"最大回撤不同。")
        recomputed = joint.joint_metrics({},local.to_dict("records"))
        for name,value in recomputed.items():
            if isinstance(value,(float,int)):
                core.require(abs(float(row[name])-value) < 1e-7,"胜率或盈亏指标不同："+name)
            elif value is None:
                core.require(pd.isna(row[name]),"缺少方向时不应有比率。")
    annual,checks = frequency_table(nav,trades,cfg)
    saved = pd.read_csv(out / "results/完整自然年交易次数.csv")
    for frame in [annual,saved]:
        frame["at_least_five"] = frame.at_least_five.astype("boolean")
    pd.testing.assert_frame_equal(annual,saved,check_dtype=False)
    keys = ["node_stage","model","scenario"]
    pd.testing.assert_frame_equal(checks.sort_values(keys,ignore_index=True),metrics[keys+["complete_years","minimum_cycles_in_full_year","annual_frequency_pass"]].sort_values(keys,ignore_index=True),check_dtype=False)
    all_numeric = (metrics.net_sharpe >= 1.3)&(metrics.max_drawdown <= .1)&metrics.annual_frequency_pass
    core.require(all_numeric.equals(metrics.all_numeric_targets_met),"完整目标联合评价不同。")
    return {"saved_accounts":len(metrics),"daily_rows":len(nav),"complete_cycles":len(trades),"frequency_rows":len(annual)}


def verify(out):
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    frame,market = rows(out,cfg)
    pd.testing.assert_frame_equal(frame,pd.read_csv(out / "results/全部成熟日样本.csv"),check_dtype=False,rtol=1e-10,atol=1e-12)
    bundles = core.load(out / "models/月末模型.json")
    final = core.load(out / "models/期末参考模型.json")
    for bundle in [*bundles,final]:
        verify_bundle(bundle,frame)
    reconstructed = reconstruct_predictions(out,frame,bundles)
    saved = pd.read_csv(out / "results/完整逐日预测.csv")
    pd.testing.assert_frame_equal(reconstructed,saved,check_dtype=False,rtol=1e-10,atol=1e-12)
    metrics = prediction_metrics(reconstructed,{m["as_of"][:10] for m in bundles})
    for rebuilt,name in [(metrics,"预测分项评价.csv"),(comparisons(metrics,cfg),"固定预测对照.csv")]:
        pd.testing.assert_frame_equal(rebuilt,pd.read_csv(out / "results" / name),check_dtype=False,rtol=1e-9,atol=1e-12)
    account = verify_accounts(out,market,cfg,saved) if (out / "summary.json").exists() else {"saved_accounts":0}
    return {"status":"PASS_SAVED_DENSE_TRAINING_AND_NODE_RECOMPUTATION","checked_at":core.now(),
        "saved_probability_models":2*(len(bundles)+1),"saved_amplitude_models":6*(len(bundles)+1),"prediction_rows":len(saved),
        **account,"new_fits":0,"new_accounts":0,"new_downloads":0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="现有日样本的概率幅度训练及逐步扩点。")
    parser.add_argument("command",choices=["freeze","train","expand","verify"])
    parser.add_argument("--root",type=Path,default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root),ensure_ascii=False,indent=2))
    else:
        {"freeze":freeze,"train":train,"expand":expand}[args.command](args.root)
