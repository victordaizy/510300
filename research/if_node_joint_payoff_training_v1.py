"""已有IF节点的概率及条件盈亏联合训练；固定样本、期限和账户规则。"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import sparse_event_training_core_v3 as core
import if_exhaustion_existing_training_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
STUDY = "510300_if_node_joint_payoff_training_v1"
OUT = ROOT / "reports/research" / STUDY


def freeze(out):
    core.require(not out.exists(), "本轮目录已存在，不覆盖。")
    previous = ROOT / "reports/research/510300_if_exhaustion_existing_training_v1"
    cfg = core.load(previous / "protocol.json")
    cfg.update({
        "study_id": STUDY.upper(), "user_instruction": "少数节点、高盈亏比、变盘信号、高胜率、高夏普、拒绝过拟合；请继续；不用采集。",
        "target_net_sharpe": 1.3, "model_change": "固定原节点、特征和10日期限，将单一收益回归改为盈利概率和正负条件幅度三个部分。",
        "models": {"A_PRICE_JOINT": {"groups": ["price"]}, "B_IF_JOINT": {"groups": ["price", "if_joint"]},
                   "MATURE_EMPIRICAL": {"groups": []}},
        "policies": {"JOINT_PRIMARY": {"rule": "预计净收益为正，获利概率>0.5，预计盈利幅度>预计亏损幅度。"},
                     "EXPECTANCY_CONTROL": {"rule": "仅要求预计收益覆盖账户估计压力费用，检验联合约束的影响。"}},
        "joint_model": {"mean_loss_penalty": 1.0, "standardized_feature_clip": 3.0,
                        "net_label_cost_proxy": 0.0024,
                        "probability": "无类别加权的逻辑回归，平均对数损失加L2惩罚，截距不惩罚。",
                        "magnitudes": "按正负净代理收益分组，对绝对幅度的对数做岭回归，再以训练残差指数均值校正。",
                        "missing_class": "若训练集缺少任一方向，条件幅度未知，保留预测记录但不入场。",
                        "empirical": "成熟训练样本的实际获利频率、条件盈利均值和条件亏损均值，不拟合特征。"},
        "probability_target": "十日毛收益减24bp名义往返压力成本是否大于0；是固定期限的净收益代理，不等于含最低佣金和账户提前退出的实际周期胜率。",
        "entry_threshold_meaning": "0.5和盈亏幅度比1为本轮未经搜索的方向性分界，不是用户指定的高胜率/高盈亏比验收数字。",
        "account_prediction": "预计净代理收益加回24bp后传入原引擎，与当时仓位估计的压力费用比较；成交仍逐笔实际扣费。",
        "training_distribution_limit": "124个月末样本用于估计条件关系，节点本身只有10个；分别报告两类预测，不声称事件概率已校准。",
        "selection_history": "IF节点在上一轮已看见历史表现；本轮是同一家族、改变预测表示的探索，不重置此前试验次数。",
        "no_parameter_grid": True, "historical_results_already_seen": True,
        "score_segments": ["ALL", "FIRST_HALF", "SECOND_HALF"],
        "stop_condition": "一次固定训练和账户后如无增量或未达目标，关闭这一表示，不在本批结果上换窗口/阈值/止盈止损。",
    })
    cfg["account"].update({"prediction_error_multiplier": 0, "min_mature_prequential_errors": 0})
    out.mkdir(parents=True)
    core.save(out / "protocol.json", cfg)
    identities = core.load(previous / "freeze.json")["identities"]
    sources = []

    def copy_file(source, relative):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        core.require(core.digest(source) == core.digest(target), "来源复制错误。")
        sources.append({"source": str(source.relative_to(ROOT)), "copy": relative, "sha256": core.digest(target)})

    for relative, expected in identities.items():
        source = previous / relative
        core.require(core.digest(source) == expected, f"父级冻结文件变化：{relative}")
        if relative.startswith("inputs/"):
            copy_file(source, "inputs/parent_mandate.json" if relative == "inputs/mandate.json" else relative)
    for original, copied in [("protocol.json", "previous_training_protocol.json"), ("freeze.json", "previous_freeze.json"),
                             ("summary.json", "previous_summary.json"), ("branch_decision.json", "previous_branch_decision.json"),
                             ("delivery_receipt.json", "previous_delivery_receipt.json"),
                             ("results/月末训练特征与十日标签.csv", "previous_monthly.csv"),
                             ("results/候选事件与十日标签.csv", "previous_events.csv"),
                             ("results/完整逐日账户.csv", "previous_daily_accounts.csv"),
                             ("results/账户指标.csv", "previous_account_metrics.csv")]:
        copy_file(previous / original, "inputs/" + copied)
    for name in ["510300_existing_data_training_mandate_v1.json", "510300_sparse_node_training_contract_v1.json"]:
        copy_file(ROOT / "config" / name, "inputs/" + name)
    for name in ["if_exhaustion_existing_training_v1.py", "sparse_event_training_core_v3.py", "if_exhaustion_local_features_v1.py"]:
        source = previous / "code" / name
        core.require(core.digest(ROOT / "research" / name) == core.digest(source), "当前计算依赖与父级冻结代码不同。")
        copy_file(source, "code/" + name)
    copy_file(Path(__file__).resolve(), "code/" + Path(__file__).name)
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_this_round_fits": True,
        "historical_results_already_seen": True, "independent_holdout": False, "sources": sources,
        "identities": {"protocol.json": core.digest(out / "protocol.json"), **{r["copy"]: r["sha256"] for r in sources}}})
    print("联合模型、两项固定比较规则、原节点和成本已保存；未调整历史失败记录。", flush=True)


def check_inputs(out):
    for relative, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / relative) == expected, f"本轮输入或代码变化：{relative}")


def make_rows(out, cfg):
    monthly, events, market = parent.make_rows(out, cfg)
    for frame, name in [(monthly, "previous_monthly.csv"), (events, "previous_events.csv")]:
        pd.testing.assert_frame_equal(frame, pd.read_csv(out / "inputs" / name), check_dtype=False, rtol=1e-10, atol=1e-12)
        frame["target_net_proxy"] = frame.target10 - cfg["joint_model"]["net_label_cost_proxy"]
        frame["target_profit"] = (frame.target_net_proxy > 0).astype(int)
    core.require(len(monthly) == 124 and len(events) == 10, "完整原节点或训练样本范围变化。")
    return monthly, events, market


def design(frame, columns, transform=None):
    x = frame[columns].to_numpy(float)
    if transform is None:
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        active = scale >= 1e-12
        transform = {"columns": columns, "center": center.tolist(), "scale": np.where(active, scale, 1).tolist(), "active": active.tolist()}
    z = np.clip((x - np.array(transform["center"])) / np.array(transform["scale"]), -3, 3)
    z[:, ~np.array(transform["active"], bool)] = 0
    return np.column_stack([np.ones(len(frame)), z]), transform


def fit_probability(frame, columns):
    x, transform = design(frame, columns)
    y = frame.target_profit.to_numpy(float)
    penalty = np.r_[0.0, np.ones(len(columns))]
    initial = np.zeros(x.shape[1])
    initial[0] = math.log((y.sum() + 0.5) / (len(y) - y.sum() + 0.5))
    if y.min() == y.max():
        return {**transform, "kind": "SINGLE_CLASS_SMOOTHED_REFERENCE", "beta": initial.tolist(), "n": len(y), "gradient_max": None}

    def objective(beta):
        score = x @ beta
        loss = np.mean(np.logaddexp(0, score) - y * score) + 0.5 * np.sum(penalty * beta ** 2)
        gradient = x.T @ (expit(score) - y) / len(y) + penalty * beta
        return loss, gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-10, "ftol": 1e-14})
    gradient_max = float(np.max(np.abs(objective(result.x)[1])))
    core.require(result.success and gradient_max < 1e-6, "概率模型未收敛，不使用失败参数。")
    return {**transform, "kind": "UNWEIGHTED_LOGISTIC", "beta": result.x.tolist(), "n": len(y), "gradient_max": gradient_max}


def fit_magnitude(frame, columns, positive):
    selected = frame.loc[frame.target_net_proxy > 0] if positive else frame.loc[frame.target_net_proxy < 0]
    if selected.empty:
        return {"kind": "MISSING_CLASS", "n": 0}
    x, transform = design(selected, columns)
    y = np.log(np.abs(selected.target_net_proxy.to_numpy(float)))
    penalty = np.diag(np.r_[0.0, np.ones(len(columns))])
    beta = np.linalg.solve(x.T @ x / len(y) + penalty, x.T @ y / len(y))
    smearing = float(np.exp(y - x @ beta).mean())
    core.require(np.isfinite(beta).all() and np.isfinite(smearing), "条件幅度模型出现非有限值。")
    return {**transform, "kind": "LOG_MAGNITUDE_RIDGE", "beta": beta.tolist(), "smearing": smearing,
            "n": len(y), "train_months": selected.month.tolist()}


def fit_joint(sample, columns, name, role, row=None, as_of=None):
    metadata = {"model": name, "prediction_role": role, "train_months": sample.month.tolist(),
                "n_train": len(sample), "train_max_label_end_at": sample.label_end_at.max(),
                "decision_at": row.decision_at if row is not None else as_of,
                "prediction_month": row.month if row is not None else None}
    if columns:
        return {**metadata, "kind": "THREE_PART_MODEL", "probability": fit_probability(sample, columns),
                "gain": fit_magnitude(sample, columns, True), "loss": fit_magnitude(sample, columns, False)}
    gain = sample.loc[sample.target_net_proxy > 0, "target_net_proxy"]
    loss = -sample.loc[sample.target_net_proxy < 0, "target_net_proxy"]
    return {**metadata, "kind": "EMPIRICAL_REFERENCE", "p_win": float(sample.target_profit.mean()),
            "conditional_gain": float(gain.mean()) if len(gain) else None,
            "conditional_loss": float(loss.mean()) if len(loss) else None,
            "gain_n": len(gain), "loss_n": len(loss)}


def predict_joint(model, row):
    if model["kind"] == "EMPIRICAL_REFERENCE":
        p, gain, loss = model["p_win"], model["conditional_gain"], model["conditional_loss"]
    else:
        frame = pd.DataFrame([row])
        head = model["probability"]
        x, _ = design(frame, head["columns"], head)
        p = float(expit((x @ head["beta"])[0]))
        values = []
        for head in [model["gain"], model["loss"]]:
            if head["n"] == 0:
                values.append(None)
            else:
                x, _ = design(frame, head["columns"], head)
                values.append(float(np.exp((x @ head["beta"])[0]) * head["smearing"]))
        gain, loss = values
    available = gain is not None and loss is not None
    expectation = p * gain - (1 - p) * loss if available else None
    ratio = gain / loss if available and loss > 0 else None
    return {"p_win": p, "conditional_gain": gain, "conditional_loss": loss,
            "predicted_payoff_ratio": ratio, "expected_net_proxy": expectation,
            "conditional_heads_available": available,
            "joint_gate_pass": bool(available and p > 0.5 and gain > loss and expectation > 0)}


def train(monthly, events, cfg):
    snapshots, monthly_predictions, event_predictions, final = [], [], [], {}
    cost = cfg["joint_model"]["net_label_cost_proxy"]
    for name, spec in cfg["models"].items():
        columns = sum([cfg["features"][g] for g in spec["groups"]], [])
        for role, rows, destination in [("MONTHLY_CALIBRATION", monthly, monthly_predictions), ("ENTRY_EVENT", events, event_predictions)]:
            for _, row in rows.iterrows():
                sample = monthly.loc[monthly.label_end_at < row.decision_at]
                if len(sample) < cfg["min_train"]:
                    continue
                model = fit_joint(sample, columns, name, role, row)
                output = predict_joint(model, row)
                record = {k: row[k] for k in ["month", "decision_at", "entry_date", "exit_date", "entry_i", "exit_i",
                    "label_end_at", "reference_close", "vol20", "target10", "target_net_proxy", "target_profit"]}
                record.update({"model": name, "prediction_role": role, "n_train": len(sample),
                    "train_max_label_end_at": sample.label_end_at.max(), **output,
                    "prediction": output["expected_net_proxy"] + cost if output["expected_net_proxy"] is not None else None,
                    "prior_prediction_rmse": 0.0, "mature_prequential_error_count": 0})
                snapshots.append(model)
                destination.append(record)
        cutoff = pd.Timestamp(cfg["signal_end"], tz="Asia/Shanghai").replace(hour=15, minute=2).isoformat()
        final[name] = fit_joint(monthly.loc[monthly.label_end_at < cutoff], columns, name, "FINAL_SNAPSHOT", as_of=cutoff)
        print(f"已完成{name}的逐期联合模型和最终参数。", flush=True)
    return pd.DataFrame(monthly_predictions), pd.DataFrame(event_predictions), snapshots, final


def scores(predictions):
    result, bins = [], []
    for name, full in predictions.groupby("model", sort=True):
        full = full.sort_values("decision_at")
        for segment, sample in [("ALL", full), ("FIRST_HALF", full.iloc[:len(full)//2]), ("SECOND_HALF", full.iloc[len(full)//2:])]:
            p = np.clip(sample.p_win.to_numpy(float), 1e-12, 1 - 1e-12)
            y = sample.target_profit.to_numpy(float)
            result.append({"model": name, "segment": segment, "n": len(sample), "mean_predicted_win_probability": float(p.mean()),
                "observed_proxy_win_rate": float(y.mean()), "brier_score": float(np.mean((p-y)**2)),
                "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
                "net_proxy_rmse": float(np.sqrt(np.mean((sample.target_net_proxy-sample.expected_net_proxy)**2))),
                "probability_direction_hit": float(((p > 0.5) == y).mean())})
        category = pd.cut(full.p_win, [0, .2, .4, .6, .8, 1], include_lowest=True, right=True)
        for label, part in full.groupby(category, observed=False):
            bins.append({"model": name, "probability_bin": str(label), "n": len(part),
                         "mean_predicted_probability": float(part.p_win.mean()), "observed_proxy_win_rate": float(part.target_profit.mean())})
    return pd.DataFrame(result), pd.DataFrame(bins)


def joint_metrics(metric, trades):
    pnl = np.array([t["net_pnl_cny"] for t in trades], float)
    debit = np.array([t["entry_cash_debit"] for t in trades], float)
    win, loss = pnl > 1e-6, pnl < -1e-6
    nw, nl, n = int(win.sum()), int(loss.sum()), len(pnl)
    total = float(pnl.sum())
    metric.update({"win_rate": nw/n if n else None, "losing_opportunities": nl,
                   "average_win_cny": float(pnl[win].mean()) if nw else None,
                   "average_loss_cny": float(-pnl[loss].mean()) if nl else None,
                   "realized_cash_payoff_ratio": float(pnl[win].mean() / -pnl[loss].mean()) if nw and nl else None,
                   "realized_return_payoff_ratio": float((pnl[win]/debit[win]).mean() / -(pnl[loss]/debit[loss]).mean()) if nw and nl else None,
                   "average_net_profit_cny": total/n if n else None,
                   "net_profit_cny": total,
                   "largest_winner_share_of_net_profit": float(pnl[win].max()/total) if nw and total > 0 else None,
                   "payoff_state": "DEFINED" if nw and nl else "NO_TRADES" if not n else "NO_OBSERVED_LOSS" if not nl else "NO_OBSERVED_WIN"})
    return metric


def accounts(predictions, events, monthly_predictions, market, dividends, cfg):
    navs, trades, decisions, metrics = [], [], [], []
    start_i = int(monthly_predictions.entry_i.min())

    def run_one(name, scenario, policy, forecast, event_only=False):
        effective = copy.deepcopy(cfg)
        if event_only:
            effective["models"][name] = {"groups": []}
        selected = forecast.loc[forecast.model == name].copy()
        selected["joint_gate_required"] = policy == "JOINT_PRIMARY"
        if name in cfg["models"]:
            allowed = selected.conditional_heads_available.astype(bool)
            if policy == "JOINT_PRIMARY":
                allowed &= selected.joint_gate_pass.astype(bool)
            admitted = selected.loc[allowed]
        else:
            admitted = selected
        nav, cycles, choices, metric = core.account_run(name, scenario, admitted, market, start_i, effective, dividends, event_only=event_only)
        nav["policy"] = policy
        navs.append(nav)
        for record in cycles:
            record["policy"] = policy
            trades.append(record)
        choices_by_date = {r["month"]: r for r in choices}
        for row in selected.to_dict("records"):
            decision = choices_by_date.get(row["month"], {"decision": "JOINT_FILTER_REJECTED" if row.get("conditional_heads_available", True) else "MISSING_CONDITIONAL_CLASS"})
            record = {k: row.get(k) for k in ["month", "entry_date", "p_win", "conditional_gain", "conditional_loss",
                       "predicted_payoff_ratio", "expected_net_proxy", "joint_gate_pass", "prediction"]}
            record.update(decision)
            record.update({"model": name, "scenario": scenario, "policy": policy})
            decisions.append(record)
        metric["policy"] = policy
        metrics.append(joint_metrics(metric, cycles))

    for policy in cfg["policies"]:
        for name in cfg["models"]:
            for scenario in cfg["account"]["slippage_per_side"]:
                run_one(name, scenario, policy, predictions)
    unfiltered = events.assign(model="EVENT_ONLY", prediction=0.0, prior_prediction_rmse=0.0, mature_prequential_error_count=0)
    for scenario in cfg["account"]["slippage_per_side"]:
        run_one("EVENT_ONLY", scenario, "EVENT_RULE_DIAGNOSTIC", unfiltered, event_only=True)
        for name in ["CASH", "BUY_AND_HOLD"]:
            run_one(name, scenario, "BENCHMARK", predictions)
    return pd.concat(navs, ignore_index=True), pd.DataFrame(trades), pd.DataFrame(decisions), pd.DataFrame(metrics)


def compare_saved_baselines(out, nav):
    old = pd.read_csv(out / "inputs/previous_daily_accounts.csv")
    columns = ["date", "cash_cny", "shares", "equity_cny", "daily_return", "drawdown", "dividend_receivable_cny"]
    compared = 0
    for name in ["EVENT_ONLY", "CASH", "BUY_AND_HOLD"]:
        for scenario in ["BASE", "STRESS"]:
            a = old.loc[(old.model == name) & (old.scenario == scenario), columns].reset_index(drop=True)
            b = nav.loc[(nav.model == name) & (nav.scenario == scenario), columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-10, atol=1e-7)
            compared += len(a)
    return compared


def run(out):
    core.require(not (out / "summary.json").exists(), "训练已完成，不重跑已有结果。")
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    monthly, events, market = make_rows(out, cfg)
    month_p, event_p, snapshots, final = train(monthly, events, cfg)
    core.save(out / "models/逐期联合模型.json", snapshots)
    core.save(out / "models/期末联合模型.json", final)
    for name, frame in [("月末训练样本", monthly), ("全部节点样本", events), ("月末逐期预测", month_p), ("节点逐期预测", event_p)]:
        core.export(out / "results" / f"{name}.csv", frame)
    for prefix, frame in [("月末", month_p), ("节点", event_p)]:
        metric, calibration = scores(frame)
        core.export(out / "results" / f"{prefix}预测指标.csv", metric)
        core.export(out / "results" / f"{prefix}概率分组.csv", calibration)
    dividends = pd.read_csv(out / "inputs/dividends.csv")
    dividends = dividends.loc[dividends.symbol == "510300.SH"]
    print(f"已保存{len(snapshots)}份逐期联合快照，开始原节点的完整账户计算。", flush=True)
    nav, trades, choices, metrics = accounts(event_p, events, month_p, market, dividends, cfg)
    baseline_rows = compare_saved_baselines(out, nav)
    for name, frame in [("完整逐日账户", nav), ("机会成交账簿", trades), ("逐节点交易判断", choices), ("账户联合指标", metrics)]:
        core.export(out / "results" / f"{name}.csv", frame)
    model_snapshots = [m for m in [*snapshots, *final.values()] if m["kind"] == "THREE_PART_MODEL"]
    passing = []
    for (name, policy), group in metrics.loc[metrics.model.isin(cfg["models"])].groupby(["model", "policy"]):
        if len(group) == 2 and group.meets_numeric_targets.all():
            passing.append({"model": name, "policy": policy})
    summary = {"study_id": cfg["study_id"], "completed_at": core.now(), "status": "FIXED_JOINT_PAYOFF_TRAINING_COMPLETED",
               "candidate_events": len(events), "monthly_training_rows": len(monthly),
               "monthly_prediction_rows": len(month_p), "event_prediction_rows": len(event_p),
               "prequential_joint_snapshots": len(snapshots), "final_joint_snapshots": len(final),
               "logistic_fits_including_final": sum(m["probability"]["kind"] == "UNWEIGHTED_LOGISTIC" for m in model_snapshots),
               "conditional_magnitude_fits_including_final": sum(m[head]["n"] > 0 for m in model_snapshots for head in ["gain", "loss"]),
               "empirical_reference_snapshots_including_final": sum(m["kind"] == "EMPIRICAL_REFERENCE" for m in [*snapshots, *final.values()]),
               "account_scenarios": len(metrics), "baseline_daily_rows_matching_original": baseline_rows,
               "numeric_targets_pass_both_costs": passing, "target_net_sharpe": 1.3,
               "maximum_drawdown_target": .1, "account_start": metrics.start.iloc[0], "account_end": metrics.end.iloc[0],
               "calendar_trading_days_per_account": int(metrics.n_trading_days.iloc[0]),
               "new_downloads": 0, "independent_forward_events": 0, "goal_achieved": False, "orders_authorized": False}
    core.save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(metrics.loc[metrics.scenario == "STRESS", ["model", "policy", "opportunities", "win_rate", "realized_cash_payoff_ratio", "net_sharpe", "max_drawdown"]].to_string(index=False), flush=True)


def verify_head(model, sample):
    if model["kind"] == "EMPIRICAL_REFERENCE":
        core.require(abs(model["p_win"] - sample.target_profit.mean()) < 1e-12, "经验概率不同。")
        for field, selected in [("conditional_gain", sample.loc[sample.target_net_proxy > 0, "target_net_proxy"]),
                                ("conditional_loss", -sample.loc[sample.target_net_proxy < 0, "target_net_proxy"])]:
            if len(selected):
                core.require(abs(model[field] - selected.mean()) < 1e-12, "经验条件幅度不同。")
        return
    head = model["probability"]
    x, computed_transform = design(sample, head["columns"])
    for key in ["center", "scale", "active"]:
        core.require(np.allclose(head[key], computed_transform[key], atol=1e-12, rtol=0), "概率标准化不是训练期数值。")
    beta = np.array(head["beta"])
    if head["kind"] == "UNWEIGHTED_LOGISTIC":
        gradient = x.T @ (expit(x @ beta)-sample.target_profit.to_numpy(float))/len(sample) + np.r_[0., np.ones(len(beta)-1)]*beta
        core.require(np.max(abs(gradient)) < 1e-6, "保存概率参数不满足收敛条件。")
    for key, positive in [("gain", True), ("loss", False)]:
        head = model[key]
        selected = sample.loc[sample.target_net_proxy > 0] if positive else sample.loc[sample.target_net_proxy < 0]
        core.require(len(selected) == head["n"], "条件幅度样本数不同。")
        if not len(selected):
            continue
        x, transform = design(selected, head["columns"])
        for field in ["center", "scale", "active"]:
            core.require(np.allclose(head[field], transform[field], atol=1e-12, rtol=0), "幅度标准化不是训练期数值。")
        y, beta = np.log(abs(selected.target_net_proxy.to_numpy(float))), np.array(head["beta"])
        residual = (x.T@x/len(y)+np.diag(np.r_[0., np.ones(len(beta)-1)]))@beta - x.T@y/len(y)
        core.require(np.max(abs(residual)) < 1e-10, "条件幅度参数不满足保存方程。")
        core.require(abs(head["smearing"]-np.exp(y-x@beta).mean()) < 1e-12, "条件均值还原不同。")


def verify(out):
    check_inputs(out)
    cfg = core.load(out / "protocol.json")
    monthly, events, market = make_rows(out, cfg)
    snapshots = core.load(out / "models/逐期联合模型.json")
    final = core.load(out / "models/期末联合模型.json")
    indexed = monthly.set_index("month", drop=False)
    rows = {"MONTHLY_CALIBRATION": monthly.set_index("month"), "ENTRY_EVENT": events.set_index("month")}
    predictions = pd.concat([pd.read_csv(out / "results/月末逐期预测.csv"), pd.read_csv(out / "results/节点逐期预测.csv")])
    maximum_prediction_error = 0.0
    for model in [*snapshots, *final.values()]:
        sample = indexed.loc[model["train_months"]]
        eligible = monthly.loc[monthly.label_end_at < model["decision_at"]]
        core.require(sample.month.tolist() == eligible.month.tolist(), "训练样本包含未来标签或漏掉成熟月份。")
        core.require(sample.label_end_at.max() == model["train_max_label_end_at"], "训练标签时钟不同。")
        verify_head(model, sample)
        if model["prediction_month"] is None:
            continue
        row = rows[model["prediction_role"]].loc[model["prediction_month"]]
        saved = predictions.loc[(predictions.model == model["model"]) & (predictions.prediction_role == model["prediction_role"]) & (predictions.month == model["prediction_month"])].iloc[0]
        output = predict_joint(model, row)
        for field in ["p_win", "conditional_gain", "conditional_loss", "predicted_payoff_ratio", "expected_net_proxy"]:
            if output[field] is None:
                core.require(pd.isna(saved[field]), "缺失条件类别被填值。")
            else:
                error = abs(saved[field]-output[field])
                maximum_prediction_error = max(maximum_prediction_error, error)
                core.require(error < 1e-10, "保存参数不能复现联合预测。")
    for prefix, role in [("月末", "MONTHLY_CALIBRATION"), ("节点", "ENTRY_EVENT")]:
        expected, _ = scores(predictions.loc[predictions.prediction_role == role])
        pd.testing.assert_frame_equal(expected, pd.read_csv(out / "results" / f"{prefix}预测指标.csv"), check_dtype=False, rtol=1e-10, atol=1e-12)
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/机会成交账簿.csv")
    metrics = pd.read_csv(out / "results/账户联合指标.csv")
    choices = pd.read_csv(out / "results/逐节点交易判断.csv")
    close = market.assign(day=market.date.dt.strftime("%Y-%m-%d")).set_index("day").close
    for (name, policy, scenario), daily in nav.groupby(["model", "policy", "scenario"]):
        selected = trades.loc[(trades.model == name) & (trades.policy == policy) & (trades.scenario == scenario)]
        metric = metrics.loc[(metrics.model == name) & (metrics.policy == policy) & (metrics.scenario == scenario)].iloc[0]
        equity = daily.equity_cny.to_numpy(float)
        returns = equity/np.r_[200000., equity[:-1]]-1
        core.require(len(daily) == 2016 and not daily.date.duplicated().any(), "完整账户日历不一致。")
        core.require(np.allclose(daily.cash_cny + daily.shares*daily.date.map(close) + daily.dividend_receivable_cny, equity, rtol=0, atol=1e-7), "账户现金和权益不平。")
        core.require(np.allclose(returns, daily.daily_return, rtol=0, atol=1e-12), "连续收益不同。")
        if returns.std(ddof=1) > 1e-14:
            core.require(abs(returns.mean()/returns.std(ddof=1)*np.sqrt(242)-metric.net_sharpe) < 1e-10, "完整账户夏普不同。")
        else:
            core.require(pd.isna(metric.net_sharpe), "全现金夏普不能赋值。")
        dd = -(equity/np.maximum.accumulate(np.r_[200000., equity])[1:]-1).min()
        core.require(abs(dd-metric.max_drawdown) < 1e-12, "账户回撤不同。")
        core.require(abs(selected.net_pnl_cny.sum()-(equity[-1]-200000)) < 1e-7, "全部周期损益不能对账。")
        measured = joint_metrics({}, selected.to_dict("records"))
        for field, value in measured.items():
            if value is None:
                core.require(pd.isna(metric[field]), "无交易或无亏损指标误填。")
            elif isinstance(value, str):
                core.require(value == metric[field], "盈亏比状态不同。")
            else:
                core.require(abs(value-metric[field]) < 1e-7, "联合账户指标不同。")
        for trade in selected.itertuples():
            core.require(trade.exit_i > trade.entry_i, "交易违反T+1。")
            dividend = market.iloc[int(trade.entry_i)+1:int(trade.exit_i)+1].dividend.sum()*trade.shares
            pnl = trade.shares*trade.exit_fill_price-trade.exit_commission_cny+dividend-trade.entry_cash_debit
            core.require(abs(pnl-trade.net_pnl_cny) < 1e-7, "单周期费用分红损益不同。")
    chosen = choices.loc[(choices.policy == "JOINT_PRIMARY") & (choices.decision == "BUY")]
    core.require(((chosen.p_win > .5) & (chosen.conditional_gain > chosen.conditional_loss) & (chosen.expected_net_proxy > 0)).all(), "联合规则入场条件失效。")
    core.require(len(choices.loc[choices.model.isin(cfg["models"])]) == 120, "未完整保留节点入场和拒绝决定。")
    baseline_rows = compare_saved_baselines(out, nav)
    return {"status": "PASS_SAVED_JOINT_MODELS_PREDICTIONS_AND_ACCOUNTS", "model_snapshots": len(snapshots)+len(final),
            "prediction_rows": len(predictions), "maximum_prediction_error": maximum_prediction_error,
            "account_scenarios": len(metrics), "baseline_daily_rows_matching_original": baseline_rows,
            "new_fits": 0, "new_accounts": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="本地固定IF节点的胜率与条件盈亏训练。")
    parser.add_argument("command", choices=["freeze", "train", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "train": run}[args.command](args.root)


if __name__ == "__main__":
    main()
