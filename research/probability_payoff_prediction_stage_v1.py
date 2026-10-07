"""先评价概率和条件盈亏预测，再讨论节点扩展；本轮不运行交易账户。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sparse_event_training_core_v3 as core
import if_node_joint_payoff_training_v1 as previous


ROOT = Path(__file__).resolve().parents[1]
STUDY = "510300_probability_payoff_prediction_stage_v1"
OUT = ROOT / "reports/research" / STUDY
REQUEST = "先训练胜率与盈亏幅度，在逐步增加节点"


def freeze(out):
    core.require(not out.exists(), "本轮目录已存在，不能覆盖。")
    parent = ROOT / "reports/research/510300_if_node_joint_payoff_training_v1"
    out.mkdir(parents=True)
    sources = {}

    def copy_file(source, target):
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        sources[target] = {"source": str(source.relative_to(ROOT)), "sha256": core.digest(destination)}

    for original, target in [
        ("results/月末训练样本.csv", "monthly.csv"), ("results/全部节点样本.csv", "events.csv"),
        ("results/月末逐期预测.csv", "previous_monthly_predictions.csv"), ("results/节点逐期预测.csv", "previous_event_predictions.csv"),
        ("models/逐期联合模型.json", "previous_models.json"), ("models/期末联合模型.json", "previous_final_models.json"),
        ("inputs/market.parquet", "market.parquet"), ("protocol.json", "previous_protocol.json"),
        ("freeze.json", "previous_freeze.json"), ("summary.json", "previous_summary.json"),
        ("branch_decision.json", "previous_branch_decision.json"), ("delivery_receipt.json", "previous_delivery_receipt.json")]:
        copy_file(parent / original, "inputs/" + target)
    for name in ["if_node_joint_payoff_training_v1.py", "sparse_event_training_core_v3.py", "if_exhaustion_existing_training_v1.py", "if_exhaustion_local_features_v1.py"]:
        source = parent / "code" / name
        core.require(core.digest(source) == core.digest(ROOT / "research" / name), "计算依赖与上一轮冻结副本不同。")
        copy_file(source, "code/" + name)
    copy_file(Path(__file__).resolve(), "code/" + Path(__file__).name)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    contract_path = ROOT / "config/510300_sparse_node_training_contract_v1.json"
    copy_file(mandate_path, "inputs/mandate_before_priority_change.json")
    copy_file(contract_path, "inputs/contract_before_priority_change.json")
    changed_at = core.now()
    mandate, contract = core.load(mandate_path), core.load(contract_path)
    update = {"latest_user_instruction": REQUEST,
              "research_sequence": ["训练并评价盈利概率", "训练并评价盈利与亏损条件幅度", "逐步增加节点", "完整账户及年度频率验收"],
              "annual_frequency_is_training_admission_gate": False,
              "minimum_complete_cycles_each_full_calendar_year": 5,
              "annual_frequency_requirement_stage": "最终完整策略验收，不作为单个节点或预测模型开始训练的门槛。",
              "prediction_priority_revision_at": changed_at,
              "prediction_priority_receipt": "reports/research/510300_probability_payoff_prediction_stage_v1/authority_update.json"}
    mandate.update(update)
    mandate["as_of_date"] = changed_at[:10]
    contract.update(update)
    contract["current_round"] = "先进行概率和盈亏幅度的预测训练评价，当前不增加节点，不据年度次数阻止训练。"
    contract["frequency_evaluation"] = "最终完整策略按每完整自然年至少5个持仓周期评价；预测训练阶段不设此准入门槛。"
    core.save(mandate_path, mandate)
    core.save(contract_path, contract)
    core.save(out / "mandate_after_priority_change.json", mandate)
    core.save(out / "node_contract_after_priority_change.json", contract)
    core.save(out / "authority_update.json", {"recorded_at": changed_at, "user_instruction": REQUEST,
              "supersedes": "先检查年度节点数量再开始训练。", "kept": {"capital_cny": 200000,
              "target_net_sharpe": 1.3, "max_drawdown": .1, "minimum_annual_cycles_final_strategy": 5,
              "assets": ["510300.SH", "CASH_CNY"], "new_collection_enabled": False}})
    protocol = {"study_id": STUDY.upper(), "user_instruction": REQUEST, "stage": "PREDICTION_ONLY_BEFORE_NODE_EXPANSION",
                "previous_results_already_seen": True, "previous_failures_preserved": True,
                "sample": "完全沿用124个月末训练样本与10个固定IF节点；不增加或删除节点。",
                "label": "原10日毛收益减24bp的净收益代理；不是实际账户周期损益。",
                "probability": "保留上一轮逐期训练的无类别加权逻辑概率和成熟经验概率，单独检查Brier及对数损失，不用本次结果回填概率。",
                "amplitude_change": "新Gamma均值回归，直接估计正/负条件收益幅度；不使用对数残差的统一还原系数。",
                "gamma_objective": "平均[y*exp(-eta)+eta]+0.5*sum(beta_without_intercept**2)，mu=exp(eta)。",
                "penalty": 1.0, "feature_clip_standard_deviations": 3,
                "features": "原价格4项、价格加IF共7项；每个条件组仅用成熟训练样本标准化。",
                "controls": ["ORIGINAL_LOG_PRICE", "ORIGINAL_LOG_IF", "EMPIRICAL_ALL", "GAMMA_PRICE", "GAMMA_IF", "IF_PROB_EMPIRICAL_AMPLITUDES"],
                "ablation_meaning": "IF概率搭配无特征历史条件均值是预先声明的幅度去特征对照，不拼接交易或账户。",
                "evaluation": "月末100次与节点10次分开，分别统计概率、实际盈利组、实际亏损组、整体预期收益误差；原顺序前半/后半并列。",
                "conditional_group_meaning": "按已实现方向分组仅用于事后评价W/L，预测时不使用未来方向。",
                "clock": "原模型每个预测时点之前标签已结束的全部月末样本；保存训练日期及每个条件子样本。",
                "no_parameter_grid": True, "no_new_nodes": True, "new_accounts": 0,
                "frequency_gate_at_this_stage": False, "final_annual_frequency_requirement": 5,
                "new_downloads": 0, "orders_authorized": False, "goal_achieved": False}
    core.save(out / "protocol.json", protocol)
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_new_amplitude_fits": True,
              "prior_head_diagnostics_already_seen": True, "independent_holdout": False,
              "sources": sources, "identities": {"protocol.json": core.digest(out / "protocol.json"), **{k:v["sha256"] for k,v in sources.items()}}})
    print("已更新研究顺序并固定本轮幅度训练；年度5次只用于最终策略验收。", flush=True)


def check_inputs(out):
    for path, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / path) == expected, f"冻结文件变化：{path}")


def read_rows(out):
    monthly = pd.read_csv(out / "inputs/monthly.csv")
    events = pd.read_csv(out / "inputs/events.csv")
    market = pd.read_parquet(out / "inputs/market.parquet")
    for frame in [monthly, events]:
        for r in frame.itertuples():
            block = market.iloc[int(r.entry_i):int(r.exit_i)+1]
            gross = (block.close.iloc[-1] + block.dividend.iloc[1:].sum()) / block.open.iloc[0] - 1
            core.require(abs(gross-r.target10) < 1e-12 and abs(gross-.0024-r.target_net_proxy) < 1e-12, "原十日标签不能复现。")
            core.require(pd.Timestamp(r.label_end_at) > pd.Timestamp(r.decision_at), "标签与决策先后不同。")
    return monthly, events


def fit_gamma(sample, columns, positive):
    selected = sample.loc[sample.target_net_proxy > 0] if positive else sample.loc[sample.target_net_proxy < 0]
    if selected.empty:
        return {"kind": "MISSING_CLASS", "n": 0}
    x, transform = previous.design(selected, columns)
    y = np.abs(selected.target_net_proxy.to_numpy(float))
    penalty = np.r_[0., np.ones(len(columns))]
    initial = np.r_[math.log(float(y.mean())), np.zeros(len(columns))]

    def objective(beta):
        eta = x @ beta
        ratio = y * np.exp(-eta)
        loss = float(np.mean(ratio + eta) + .5*np.sum(penalty*beta**2))
        gradient = x.T @ (1-ratio)/len(y) + penalty*beta
        return loss, gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-10, "ftol": 1e-14})
    gradient_max = float(np.max(np.abs(objective(result.x)[1])))
    core.require(result.success and gradient_max < 1e-6, "直接条件均值模型没有收敛。")
    return {**transform, "kind": "GAMMA_CONDITIONAL_MEAN", "beta": result.x.tolist(), "n": len(y),
            "train_months": selected.month.tolist(), "gradient_max": gradient_max}


def predict_gamma(model, row):
    if model["n"] == 0:
        return None
    x, _ = previous.design(pd.DataFrame([row]), model["columns"], model)
    return float(np.exp((x @ model["beta"])[0]))


def make_output(row, model, p, gain, loss, role):
    available = gain is not None and loss is not None
    return {"model": model, "scope": "MONTHLY" if role == "MONTHLY_CALIBRATION" else "NODE",
            "month": row.month, "decision_at": row.decision_at, "label_end_at": row.label_end_at,
            "target_net_proxy": row.target_net_proxy, "target_profit": int(row.target_profit),
            "p_win": p, "conditional_gain": gain, "conditional_loss": loss,
            "expected_net_proxy": p*gain-(1-p)*loss if available else None,
            "predicted_payoff_ratio": gain/loss if available and loss > 0 else None}


def score_predictions(predictions):
    records = []
    for (scope, name), full in predictions.groupby(["scope", "model"], sort=True):
        full = full.sort_values("decision_at")
        for segment, sample in [("ALL", full), ("FIRST_HALF", full.iloc[:len(full)//2]), ("SECOND_HALF", full.iloc[len(full)//2:])]:
            p = np.clip(sample.p_win.to_numpy(float), 1e-12, 1-1e-12)
            y = sample.target_profit.to_numpy(float)
            records.append({"scope": scope, "model": name, "segment": segment, "head": "PROBABILITY", "n": len(sample),
                            "mse": float(np.mean((p-y)**2)), "mae": float(np.mean(abs(p-y))),
                            "rmse": float(np.sqrt(np.mean((p-y)**2))), "mean_prediction": float(p.mean()), "mean_actual": float(y.mean()),
                            "log_loss": float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))})
            for head, part, pred_col, sign in [
                ("GAIN", sample.loc[sample.target_net_proxy > 0], "conditional_gain", 1),
                ("LOSS", sample.loc[sample.target_net_proxy < 0], "conditional_loss", -1),
                ("EXPECTATION", sample, "expected_net_proxy", 1)]:
                actual = part.target_net_proxy.to_numpy(float)*sign
                prediction = part[pred_col].to_numpy(float)
                mse = float(np.mean((actual-prediction)**2)) if len(part) else None
                records.append({"scope": scope, "model": name, "segment": segment, "head": head, "n": len(part),
                                "mse": mse, "mae": float(np.mean(abs(actual-prediction))) if len(part) else None,
                                "rmse": math.sqrt(mse) if mse is not None else None,
                                "mean_prediction": float(prediction.mean()) if len(part) else None,
                                "mean_actual": float(actual.mean()) if len(part) else None, "log_loss": None})
    metrics = pd.DataFrame(records)
    comparisons = []
    for new, reference in [("GAMMA_IF", "ORIGINAL_LOG_IF"), ("GAMMA_PRICE", "ORIGINAL_LOG_PRICE"),
                           ("GAMMA_IF", "EMPIRICAL_ALL"), ("IF_PROB_EMPIRICAL_AMPLITUDES", "ORIGINAL_LOG_IF"),
                           ("GAMMA_IF", "IF_PROB_EMPIRICAL_AMPLITUDES")]:
        a = metrics.loc[metrics.model == new]
        b = metrics.loc[metrics.model == reference]
        joined = a.merge(b, on=["scope", "segment", "head"], suffixes=("", "_reference"), validate="one_to_one")
        for r in joined.itertuples():
            comparisons.append({"scope": r.scope, "segment": r.segment, "head": r.head, "model": new, "reference": reference,
                                "n": r.n, "mse_improvement": 1-r.mse/r.mse_reference if pd.notna(r.mse_reference) and r.mse_reference > 0 else None,
                                "mae_difference": r.mae-r.mae_reference})
    return metrics, pd.DataFrame(comparisons)


def train(out):
    core.require(not (out / "summary.json").exists(), "本轮已训练，不覆盖或重试选优。")
    check_inputs(out)
    monthly, events = read_rows(out)
    saved = core.load(out / "inputs/previous_models.json")
    final_saved = core.load(out / "inputs/previous_final_models.json")
    row_lookup = {"MONTHLY_CALIBRATION": monthly.set_index("month", drop=False), "ENTRY_EVENT": events.set_index("month", drop=False)}
    names = {"A_PRICE_JOINT": "ORIGINAL_LOG_PRICE", "B_IF_JOINT": "ORIGINAL_LOG_IF", "MATURE_EMPIRICAL": "EMPIRICAL_ALL"}
    old_by_key = {(m["model"], m["prediction_role"], m["prediction_month"]): m for m in saved}
    snapshots, predictions, final = [], [], []
    for old in saved:
        role, month = old["prediction_role"], old["prediction_month"]
        row = row_lookup[role].loc[month]
        sample = monthly.loc[monthly.label_end_at < old["decision_at"]]
        core.require(sample.month.tolist() == old["train_months"], "历史概率训练样本与当前成熟样本不同。")
        old_prediction = previous.predict_joint(old, row)
        predictions.append(make_output(row, names[old["model"]], old_prediction["p_win"], old_prediction["conditional_gain"], old_prediction["conditional_loss"], role))
        if old["kind"] == "EMPIRICAL_REFERENCE":
            continue
        columns = old["probability"]["columns"]
        new_name = "GAMMA_PRICE" if old["model"] == "A_PRICE_JOINT" else "GAMMA_IF"
        model = {"model": new_name, "previous_probability_model": old["model"], "prediction_role": role,
                 "prediction_month": month, "decision_at": old["decision_at"], "train_months": sample.month.tolist(),
                 "train_max_label_end_at": sample.label_end_at.max(), "n_train": len(sample),
                 "gain": fit_gamma(sample, columns, True), "loss": fit_gamma(sample, columns, False)}
        gain, loss = predict_gamma(model["gain"], row), predict_gamma(model["loss"], row)
        snapshots.append(model)
        predictions.append(make_output(row, new_name, old_prediction["p_win"], gain, loss, role))
        if old["model"] == "B_IF_JOINT":
            empirical = old_by_key[("MATURE_EMPIRICAL", role, month)]
            predictions.append(make_output(row, "IF_PROB_EMPIRICAL_AMPLITUDES", old_prediction["p_win"], empirical["conditional_gain"], empirical["conditional_loss"], role))
    for name in ["A_PRICE_JOINT", "B_IF_JOINT"]:
        old = final_saved[name]
        sample = monthly.loc[monthly.label_end_at < old["decision_at"]]
        columns = old["probability"]["columns"]
        final.append({"model": "GAMMA_PRICE" if name == "A_PRICE_JOINT" else "GAMMA_IF", "decision_at": old["decision_at"],
                      "train_months": sample.month.tolist(), "train_max_label_end_at": sample.label_end_at.max(), "n_train": len(sample),
                      "gain": fit_gamma(sample, columns, True), "loss": fit_gamma(sample, columns, False)})
    predictions = pd.DataFrame(predictions).sort_values(["scope", "model", "decision_at"], ignore_index=True)
    metrics, comparisons = score_predictions(predictions)
    core.save(out / "models/逐期直接均值模型.json", snapshots)
    core.save(out / "models/期末直接均值模型.json", final)
    core.export(out / "results/完整概率与幅度预测.csv", predictions)
    core.export(out / "results/分项预测误差.csv", metrics)
    core.export(out / "results/固定对照比较.csv", comparisons)
    summary = {"study_id": STUDY.upper(), "completed_at": core.now(), "status": "PREDICTION_STAGE_TRAINED_NO_NODE_EXPANSION",
               "monthly_training_rows": len(monthly), "node_rows": len(events), "monthly_evaluation_origins": 100,
               "node_evaluation_origins": 10, "prediction_rows": len(predictions), "new_gamma_model_bundles": len(snapshots)+len(final),
               "new_conditional_mean_fits": sum(m[h]["n"] > 0 for m in [*snapshots, *final] for h in ["gain", "loss"]),
               "new_probability_fits": 0, "saved_probability_models_reused": 222,
               "new_nodes": 0, "new_accounts": 0, "new_downloads": 0, "frequency_used_as_training_gate": False,
               "final_strategy_minimum_annual_cycles": 5, "target_net_sharpe": 1.3, "max_drawdown": .1,
               "independent_validation_established": False, "goal_achieved": False, "orders_authorized": False}
    core.save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(metrics.loc[(metrics.segment == "ALL") & metrics.model.isin(["ORIGINAL_LOG_IF", "GAMMA_IF", "IF_PROB_EMPIRICAL_AMPLITUDES"]),
                      ["scope", "model", "head", "n", "rmse", "mean_prediction", "mean_actual"]].to_string(index=False), flush=True)


def verify_gamma(head, sample, positive):
    selected = sample.loc[sample.target_net_proxy > 0] if positive else sample.loc[sample.target_net_proxy < 0]
    core.require(head["n"] == len(selected), "条件分组样本数不同。")
    if not len(selected):
        return
    core.require(head["train_months"] == selected.month.tolist(), "条件组使用了其它样本。")
    x, transform = previous.design(selected, head["columns"])
    for key in ["center", "scale", "active"]:
        core.require(np.allclose(head[key], transform[key], atol=1e-12, rtol=0), "训练标准化不同。")
    y, beta = np.abs(selected.target_net_proxy.to_numpy(float)), np.array(head["beta"])
    gradient = x.T@(1-y*np.exp(-x@beta))/len(y) + np.r_[0., np.ones(len(beta)-1)]*beta
    core.require(np.max(abs(gradient)) < 1e-6, "保存条件均值参数不满足收敛条件。")


def verify(out):
    check_inputs(out)
    monthly, events = read_rows(out)
    saved = core.load(out / "inputs/previous_models.json")
    old_by_key = {(m["model"], m["prediction_role"], m["prediction_month"]): m for m in saved}
    old_final = core.load(out / "inputs/previous_final_models.json")
    snapshots = core.load(out / "models/逐期直接均值模型.json")
    finals = core.load(out / "models/期末直接均值模型.json")
    predictions = pd.read_csv(out / "results/完整概率与幅度预测.csv")
    lookup = {"MONTHLY_CALIBRATION": monthly.set_index("month", drop=False), "ENTRY_EVENT": events.set_index("month", drop=False)}
    reconstructed = []
    names = {"A_PRICE_JOINT": "ORIGINAL_LOG_PRICE", "B_IF_JOINT": "ORIGINAL_LOG_IF", "MATURE_EMPIRICAL": "EMPIRICAL_ALL"}
    for old in saved:
        row = lookup[old["prediction_role"]].loc[old["prediction_month"]]
        output = previous.predict_joint(old, row)
        sample = monthly.loc[monthly.label_end_at < old["decision_at"]]
        core.require(sample.month.tolist() == old["train_months"], "保存概率时钟不同。")
        previous.verify_head(old, sample)
        reconstructed.append(make_output(row, names[old["model"]], output["p_win"], output["conditional_gain"], output["conditional_loss"], old["prediction_role"]))
        if old["model"] == "B_IF_JOINT":
            empirical = old_by_key[("MATURE_EMPIRICAL", old["prediction_role"], old["prediction_month"])]
            reconstructed.append(make_output(row, "IF_PROB_EMPIRICAL_AMPLITUDES", output["p_win"], empirical["conditional_gain"], empirical["conditional_loss"], old["prediction_role"]))
    for model in [*snapshots, *finals]:
        sample = monthly.loc[monthly.label_end_at < model["decision_at"]]
        core.require(sample.month.tolist() == model["train_months"], "新幅度训练包含未成熟或遗漏的标签。")
        core.require(sample.label_end_at.max() == model["train_max_label_end_at"], "最晚训练标签时钟不同。")
        for head, positive in [("gain", True), ("loss", False)]:
            verify_gamma(model[head], sample, positive)
        if "prediction_month" not in model:
            continue
        row = lookup[model["prediction_role"]].loc[model["prediction_month"]]
        old = old_by_key[(model["previous_probability_model"], model["prediction_role"], model["prediction_month"])]
        p = previous.predict_joint(old, row)["p_win"]
        reconstructed.append(make_output(row, model["model"], p, predict_gamma(model["gain"], row), predict_gamma(model["loss"], row), model["prediction_role"]))
    reconstructed = pd.DataFrame(reconstructed).sort_values(["scope", "model", "decision_at"], ignore_index=True)
    pd.testing.assert_frame_equal(reconstructed, predictions, check_dtype=False, rtol=1e-10, atol=1e-12)
    metrics, comparisons = score_predictions(reconstructed)
    for frame, name in [(metrics, "分项预测误差.csv"), (comparisons, "固定对照比较.csv")]:
        pd.testing.assert_frame_equal(frame, pd.read_csv(out / "results" / name), check_dtype=False, rtol=1e-10, atol=1e-12)
    for scope, group in predictions.groupby("scope"):
        pivot = group.pivot(index="month", columns="model", values="p_win")
        core.require(np.allclose(pivot.GAMMA_IF, pivot.ORIGINAL_LOG_IF) and np.allclose(pivot.GAMMA_PRICE, pivot.ORIGINAL_LOG_PRICE), "幅度训练改变了概率。")
    return {"status": "PASS_SAVED_PREDICTION_STAGE_RECOMPUTATION", "saved_new_conditional_heads": 2*(len(snapshots)+len(finals)),
            "prediction_rows": len(predictions), "mature_label_clock": "PASS", "unchanged_probability": "PASS",
            "new_fits": 0, "new_nodes": 0, "new_accounts": 0, "new_downloads": 0}


def main():
    parser = argparse.ArgumentParser(description="先训练概率与盈亏幅度的本地预测阶段。")
    parser.add_argument("command", choices=["freeze", "train", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
    else:
        {"freeze": freeze, "train": train}[args.command](args.root)


if __name__ == "__main__":
    main()
