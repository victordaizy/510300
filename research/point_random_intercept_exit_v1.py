"""隔离方法：成熟周期随机截距方差、原八项岭预测和原共同账户。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import copy
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController, prediction_identity
from research.learned_cycle_exit_v1 import FEATURES, state_values, training_rows
from research.point_account_cashflow_state_v1 import digest, now, require, write_json
from research import point_volatility_unit_exit_v1 as source_contract
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_random_intercept_exit_v1"
CURRENT = source_contract.CURRENT
WEIGHT = source_contract.WEIGHT
CONFIG_NAMES = source_contract.CONFIG_NAMES
STUDY = "510300_POINT_RANDOM_INTERCEPT_EXIT_V1"
KIND = "CYCLE_EQUAL_RANDOM_INTERCEPT_REML_THEN_GLS_RIDGE"


def components(rows, cfg):
    """把原周期等权二次型精确拆成周期内和周期均值部分。"""
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "随机截距输入缺失，不删行补救。")
    require(np.isfinite(weights).all() and (weights > 0).all(), "原周期权重无效。")
    require(not rows.duplicated(["cycle_id", "origin_index"]).any(), "训练周期原点重复。")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    dx, dy = np.empty_like(z), np.empty_like(y)
    group_z, group_y, counts, ids = [], [], [], []
    for cycle_id, positions in rows.groupby("cycle_id", sort=True).indices.items():
        n = len(positions)
        require(np.allclose(weights[positions], 1/n, atol=1e-14, rtol=0), "原周期总权重一、周期内等权改变。")
        require(np.all(np.diff(rows.origin_index.to_numpy()[positions]) > 0), "训练原点没有按周期递增。")
        mz, my = z[positions].mean(axis=0), float(y[positions].mean())
        dx[positions], dy[positions] = z[positions]-mz, y[positions]-my
        group_z.append(mz); group_y.append(my); counts.append(n); ids.append(int(cycle_id))
    group_z, group_y = np.asarray(group_z), np.asarray(group_y)
    centered_z, centered_y = group_z-group_z.mean(axis=0), group_y-group_y.mean()
    _, singular, vt = np.linalg.svd((z-np.average(z, axis=0, weights=weights))*np.sqrt(weights[:, None]), full_matrices=False)
    tolerance = max(z.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
    rank = int(np.sum(singular > tolerance))
    basis = vt[:rank].T
    return {"mean": mean, "scale": scale, "z": z, "y": y, "weights": weights, "cycle_ids": ids,
            "counts": np.asarray(counts, int), "group_z": group_z, "group_y": group_y,
            "within_x": dx.T@(weights[:, None]*dx), "within_xy": dx.T@(weights*dy),
            "within_yy": float(dy@(weights*dy)), "between_x": centered_z.T@centered_z,
            "between_xy": centered_z.T@centered_y, "between_yy": float(centered_y@centered_y),
            "rank": rank, "basis": basis, "rows": len(rows), "cycles": len(ids)}


def reml_profile(comp, mean_weight):
    """返回去除固定n_j log(n_j)常数后的两倍负剖面REML及残差尺度。"""
    k = float(mean_weight)
    if not 0 < k <= 1:
        return np.inf, np.nan
    basis = comp["basis"]
    information = basis.T@(comp["within_x"]+k*comp["between_x"])@basis
    rhs = basis.T@(comp["within_xy"]+k*comp["between_xy"])
    if comp["rank"]:
        factor = cho_factor(information, lower=True, check_finite=True)
        solution = cho_solve(factor, rhs, check_finite=True)
        logdet = float(2*np.log(np.diag(factor[0])).sum())
        rss = comp["within_yy"]+k*comp["between_yy"]-float(rhs@solution)
    else:
        logdet = 0.
        rss = comp["within_yy"]+k*comp["between_yy"]
    degrees = comp["rows"]-comp["rank"]-1
    if degrees <= 0 or rss <= 0 or not np.isfinite(rss):
        return np.inf, np.nan
    objective = -(comp["cycles"]-1)*np.log(k)+np.log(comp["cycles"])+logdet+degrees*np.log(rss/degrees)
    return float(objective), float(rss/degrees)


def solve_pooling(comp, cfg, mean_weight):
    """k=0退化为原周期内岭，k=1为原周期等权的普通岭；候选k只由REML决定。"""
    k = float(mean_weight)
    require(0 <= k <= 1 and cfg["ridge_alpha"] == 1., "信息权重或原岭惩罚改变。")
    information = comp["within_x"]+k*comp["between_x"]+cfg["ridge_alpha"]*np.eye(len(FEATURES))
    rhs = comp["within_xy"]+k*comp["between_xy"]
    beta = cho_solve(cho_factor(information, lower=True, check_finite=True), rhs, check_finite=True)
    intercept = float(comp["group_y"].mean()-comp["group_z"].mean(axis=0)@beta)
    require(np.isfinite(beta).all() and np.isfinite(intercept), "周期随机截距系数非有限。")
    return {"kind": KIND, "features": FEATURES.copy(), "mean": comp["mean"].tolist(), "scale": comp["scale"].tolist(),
            "feature_clip": cfg["feature_clip"], "coefficients": beta.tolist(), "intercept": intercept, "between_cycle_mean_weight": k,
            "new_cycle_random_intercept": 0., "new_cycle_intercept_rule": "POPULATION_INTERCEPT_WITH_ZERO_UNKNOWN_RANDOM_EFFECT"}


def fit_random_intercept(comp, cfg):
    require(comp["cycles"] > 1, "周期共同方差至少需要两个训练周期。")
    fit = minimize_scalar(lambda k: reml_profile(comp, k)[0], bounds=(0., 1.), method="bounded",
                          options={"xatol": 1e-8, "maxiter": 500, "disp": 0})
    require(fit.success and np.isfinite(fit.fun), "REML未收敛或方差不识别，不重启或换求解器。")
    boundary, _ = reml_profile(comp, 1.)
    k = 1. if boundary <= fit.fun else float(fit.x)
    objective, noise_scale = reml_profile(comp, k)
    require(np.isfinite(objective) and noise_scale > 0, "随机截距方差不识别。")
    model = solve_pooling(comp, cfg, k)
    ratio = (1-k)/k
    model.update(variance_ratio=ratio, residual_variance_scale=noise_scale,
                 random_intercept_variance=ratio*noise_scale, reml_objective=objective,
                 reml_evaluations=int(fit.nfev)+2, zero_random_variance_boundary=k == 1.,
                 reml_fixed_effect_rank=comp["rank"]+1,
                 residual_covariance_definition="sigma_e_squared_times_cycle_size_identity_plus_sigma_b_squared_ones",
                 variance_fit="UNPENALIZED_REML_THEN_ORIGINAL_ALPHA_ONE_GLS_RIDGE")
    model["training_cycle_effects"] = [
        {"cycle_id": cid, "rows": int(n), "conditional_training_effect": float((1-k)*(my-model["intercept"]-mz@np.asarray(model["coefficients"])))}
        for cid, n, mz, my in zip(comp["cycle_ids"], comp["counts"], comp["group_z"], comp["group_y"])]
    return model


def random_intercept_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "随机截距预测模型身份错误。")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "预测必须具备原八项输入。")
    require(model["new_cycle_random_intercept"] == 0., "不能把未知新周期的未来随机效应加入预测。")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


class RandomInterceptController(EntryVintageExitController):
    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "固定版本必须在真实买入首次收盘选择。")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t)-1
            self.record = self.models[k] if k >= 0 else None
            self.identity = prediction_identity(self.record)
            self.negative_count = 0
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index, "本笔固定版本时钟变化。")
        x = state_values(self.data, t, cycle, current_value, peak_value)
        record, prediction, status = self.record, None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index, "使用未来周期或未来模型。")
            if np.isfinite(x).all():
                prediction, status = random_intercept_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"].startswith("NO_VIEW_MODEL"):
            status = record["status"]
        self.negative_count = self.negative_count+1 if prediction is not None and prediction < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": prediction,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "model_selection_index": self.selection_index, "model_selection_origin": self.data.date.iloc[self.selection_index],
                "fixed_prediction_identity": self.identity, "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days, **dict(zip(FEATURES, x))}


def source_paths():
    paths = source_contract.source_paths()
    paths.extend(Path(p) for p in ["research/point_random_intercept_exit_v1.py", "tests/test_point_random_intercept_exit_v1.py",
        "research/adaptive_allocation_v1.py", "research/simple_intraday_protection_v1.py", "research/simple_session_divergence_v1.py",
        "reports/research/510300_point_prequential_bias_admission_v1/summary.json",
        "reports/research/510300_point_exit_sign_calibration_v1/error_components/summary.json",
        "reports/research/510300_point_random_intercept_exit_v1/prior_method_review.json",
        "reports/research/510300_point_random_intercept_exit_v1/tests_receipt.json"])
    return sorted(set(paths))


def freeze():
    tests = json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 7, "七项必要方法测试未通过。")
    protocol = {"study": STUDY, "frozen_at": now(), "linked_question": "E08_周期共同误差的方差分量方法，非E02新增市场字段",
        "hypothesis": "固定周期截距训练只保留周期内信息；用成熟样本估计周期共同方差后，有限保留周期均值信息可能改善未知新周期的原收益预测。",
        "candidate_configurations": 1, "economic_cases_predeclared": 4,
        "original_target": "原BASE自然退出相对下一开盘退出的净继续收益，分红/费用/标签终点不改。",
        "original_training": {"recent_cycles":20,"minimum_cycles":10,"minimum_rows":100,"ridge_alpha":1.,"feature_clip":5.,"features":FEATURES},
        "variance_model": "同周期协方差sigma_e^2*n_j*I+sigma_b^2*11'；n_j为该成熟训练周期保存状态数，沿用周期总权重一；不同周期独立。",
        "variance_estimation": "无惩罚固定效应的剖面REML估计k=1/(1+sigma_b^2/sigma_e^2)，单次bounded，物理域0<k<=1，xatol1e-8/maxiter500，比较合法零方差端点k1；不多起点、不收益选k。",
        "final_fit": "固定所估k，原alpha1岭拟合；周期内二次型+k乘周期均值二次型+beta平方；总体截距不罚。两阶段方差估计与岭预测明确分开。",
        "rank": "原标准化/clip5后加权中心设计SVD识别固定效应秩，阈值max(shape)*machine_eps*largest_singular；仅用于REML去掉精确冗余方向。",
        "new_cycle_prediction": "只用总体截距和原八项系数；未知新周期随机效应严格0，不把训练周期条件效应或未来标签加入当前预测。",
        "clock": "原142个月首收盘、原成熟成员；入场首次真实收盘固定版本、原两天严格负预测确认、无模型不补换。",
        "cache": "相同有序训练成员/值/权重/固定设置只在首次月份估计，成功或失败均复用；每月时点记录保留。",
        "control": "k0精确退化原周期内岭，所有115个成熟原记录参数误差<=1e-12；k1普通周期等权岭仅合成测试，不作收益候选。",
        "prediction_gate": "原同自然周期同原点配对且原可用预测覆盖完整；两时期原收益MSE严格下降，周期等权整入场年度块改善95%下界>0，5000次seed51030073。",
        "economic_contract": "原A共同账户，两段独立20万元，252日全日历，50%上限/10百分点调仓带/原ES及跳空回撤预算，T+1/100份/0.001/分红，原BASE/STRESS。BASE账户也使用STRESS来源目标。",
        "economic_gate": "四场景CAGR和全账户Sharpe均严格提高，实际净pB>1、平均自然完成周期净收益>0、最大回撤<=10%，全部逐年次数/费用报告。",
        "failure_exit": "预测门失败不运行经济账户，任何门失败拒绝固定方法；不换方差结构、优化器、正则、窗口、阈值或未来周期效应营救。",
        "limits": "不是直接预测未来周期截距，不假定共同误差可交易；误差独立/高斯及n_j异方差结构是方法假设，原已用历史不能验证其真实成立或消除选优偏差。",
        "method_reference": "https://www.statsmodels.org/stable/mixed_linear.html；随机截距、零均值群组效应与ML/REML定义，非盈利证据。",
        "independent_validation": "NOT_ESTABLISHED", "history_role": "DEVELOPMENT_CALIBRATION", "goal_achieved":False,
        "sources": [{"path":str(p),"sha256":digest(ROOT/p)} for p in source_paths()]}
    write_json(OUT/"protocol.json", protocol, exclusive=True)
    write_json(OUT/"freeze.json", {"at":now(),"protocol_sha256":digest(OUT/"protocol.json")},exclusive=True)
    print(f"E08随机截距方差方法已冻结：{len(protocol['sources'])}来源，一方法；尚未估计历史候选或读取其效果。",flush=True)


def check():
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    frozen=json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT/"protocol.json")==frozen["protocol_sha256"],"E08冻结协议改变。")
    for source in protocol["sources"]:
        require(digest(ROOT/source["path"])==source["sha256"],"E08来源改变："+source["path"])
    return len(protocol["sources"])


def save_table(name, frame):
    folder=OUT/"results";folder.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(folder/f"{name}.parquet",index=False)
    frame.to_csv(folder/f"{name}.csv",index=False,encoding="utf-8-sig")


def input_identity(rows, cfg):
    matrix=rows[["cycle_id","origin_index","target","sample_weight",*FEATURES]]
    header=json.dumps({"alpha":cfg["ridge_alpha"],"clip":cfg["feature_clip"]},sort_keys=True).encode()
    return hashlib.sha256(header+pd.util.hash_pandas_object(matrix,index=False).to_numpy().tobytes()).hexdigest()


def model_at_entry(records, entry_index):
    return source_contract.model_at_entry(records, entry_index)


def mean_improvement_interval(cycle_table):
    years=sorted(cycle_table.entry_year.unique())
    if len(years)<2:
        return {"status":"NOT_COMPUTED_INSUFFICIENT_YEARS","low":None,"high":None}
    blocks=[cycle_table.loc[cycle_table.entry_year.eq(y),"improvement"].to_numpy(float) for y in years]
    rng=np.random.default_rng(51030073); values=np.empty(5000)
    for i in range(5000):
        selected=rng.integers(0,len(blocks),len(blocks))
        values[i]=np.concatenate([blocks[k] for k in selected]).mean()
    low,high=np.quantile(values,[.025,.975])
    return {"status":"DESCRIPTIVE_REUSED_DEVELOPMENT_HISTORY","low":float(low),"high":float(high)}


def fit_and_evaluate():
    check()
    write_json(OUT/"PREDICTION_STARTED.json",{"at":now()},exclusive=True)
    samples=pd.read_parquet(ROOT/CURRENT/"results/training_reference/samples.parquet")
    cycles=pd.read_parquet(ROOT/CURRENT/"results/training_reference/cycles.parquet").set_index("cycle_id")
    originals=json.loads((ROOT/CURRENT/"inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    cfg=json.loads((ROOT/CURRENT/"inputs/config/within_cycle_exit.json").read_text(encoding="utf-8"))
    cache, candidates, controls, fit_rows={},[],[],[]
    for original in originals:
        record={k:copy.deepcopy(v) for k,v in original.items() if k!="model"};record["model"]=None
        if original["status"]=="FIT_COMPLETE":
            rows, ids=training_rows(samples,original["fit_index"],cfg)
            require(ids==original["training_cycles"] and len(rows)==original["training_rows"],"原成熟训练成员改变。")
            require(len(ids)>=10 and len(rows)>=100,"原成熟模型不符合样本门。")
            key=input_identity(rows,cfg); first=key not in cache
            if first:
                comp=components(rows,cfg)
                control=solve_pooling(comp,cfg,0.)
                try:
                    model=fit_random_intercept(comp,cfg); failure=None
                except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as error:
                    model=None;failure=str(error)
                cache[key]={"first_fit_origin":original["fit_origin"],"control":control,"model":model,"failure":failure}
            stored=cache[key]
            control=stored["control"]
            errors=[abs(control["intercept"]-original["model"]["intercept"])]
            errors.extend(float(np.max(np.abs(np.asarray(control[k])-np.asarray(original["model"][k])))) for k in ["mean","scale","coefficients"])
            require(max(errors)<=1e-12,"k0未还原原保存模型参数。")
            controls.append({"fit_origin":original["fit_origin"],"max_parameter_error":max(errors),"input_identity":key,"control_first_estimation":first})
            record.update(model=copy.deepcopy(stored["model"]),status="FIT_COMPLETE" if stored["model"] else "NO_VIEW_MODEL_VARIANCE_FIT_FAILED",
                          failure=stored["failure"],input_identity=key,first_estimation_origin=stored["first_fit_origin"],reused=not first)
            fit_rows.append({"fit_origin":original["fit_origin"],"first_estimation":first,"status":record["status"],
                             "between_cycle_mean_weight":stored["model"]["between_cycle_mean_weight"] if stored["model"] else None,
                             "variance_ratio":stored["model"]["variance_ratio"] if stored["model"] else None})
        candidates.append(record)
    write_json(OUT/"candidate_models.json",{"at":now(),"models":candidates},exclusive=True)
    paired=[]
    for row in samples.itertuples():
        cycle=cycles.loc[row.cycle_id]; entry_index=int(cycle.entry_index)
        original=model_at_entry(originals,entry_index);candidate=model_at_entry(candidates,entry_index)
        old,new,status=np.nan,np.nan,"NO_VIEW_NO_ORIGINAL_MODEL_AT_ENTRY"
        x=[getattr(row,k) for k in FEATURES]
        if original and original["status"]=="FIT_COMPLETE":
            require(original["latest_exit_index"]<=original["fit_index"]<=entry_index<=row.origin_index<row.early_exit_index<row.exit_index,"预测或成熟时钟非法。")
            old=within_cycle_prediction(original["model"],x)
            if candidate and candidate["status"]=="FIT_COMPLETE":
                require(candidate["fit_index"]==original["fit_index"],"配对模型月份不同。")
                new=random_intercept_prediction(candidate["model"],x);status="PAIRED_PREDICTION_AVAILABLE"
            else:
                status="NO_VIEW_CANDIDATE_VARIANCE_MODEL_AT_ENTRY"
        paired.append({"cycle_id":int(row.cycle_id),"origin":row.origin,"origin_index":int(row.origin_index),
                       "entry_index":entry_index,"entry_year":int(cycle.entry_date.year),"mature_date":row.mature_date,
                       "target":float(row.target),"baseline_prediction":old,"candidate_prediction":new,"status":status})
    paired=pd.DataFrame(paired); diagnostics=[]; cycle_rows=[]
    for period,start,end in [("2015_2019","2015-01-05","2019-12-31"),("2020_2026","2020-01-02","2026-09-30")]:
        group=paired.loc[paired.origin.between(pd.Timestamp(start),pd.Timestamp(end))].copy()
        old_available=group.baseline_prediction.notna(); available=old_available&group.candidate_prediction.notna()
        matched=group.loc[available].copy()
        matched["old_error"]=(matched.baseline_prediction-matched.target)**2
        matched["new_error"]=(matched.candidate_prediction-matched.target)**2
        values=matched.groupby("cycle_id").agg(baseline_mse=("old_error","mean"),candidate_mse=("new_error","mean"),entry_year=("entry_year","first"),rows=("origin_index","size")).reset_index()
        values["improvement"]=values.baseline_mse-values.candidate_mse;values["period"]=period;cycle_rows.append(values)
        interval=mean_improvement_interval(values) if len(values) else {"status":"NOT_COMPUTED_NO_PAIRS","low":None,"high":None}
        baseline=float(values.baseline_mse.mean()) if len(values) else None
        candidate=float(values.candidate_mse.mean()) if len(values) else None
        complete=int(available.sum())==int(old_available.sum()) and old_available.any()
        passed=bool(complete and candidate<baseline and interval["low"] is not None and interval["low"]>0)
        diagnostics.append({"period":period,"original_available_rows":int(old_available.sum()),"paired_rows":int(available.sum()),"cycles":len(values),
                            "complete_pair_coverage":bool(complete),"baseline_raw_return_mse":baseline,"candidate_raw_return_mse":candidate,
                            "relative_mse_change":candidate/baseline-1 if baseline else None,"improvement_interval":interval,"prediction_gate_passed":passed})
    passed=all(x["prediction_gate_passed"] for x in diagnostics)
    distinct=len(cache);failed=sum(v["model"] is None for v in cache.values())
    accounting={"candidate_configurations":1,"monthly_records":len(candidates),"mature_monthly_records":len(controls),
        "distinct_training_inputs":distinct,"new_variance_optimizations":distinct,"new_candidate_coefficient_estimations":distinct-failed,
        "new_combined_candidate_fits":distinct,"failed_distinct_candidate_fits":failed,"reused_monthly_records":len(controls)-distinct,
        "control_coefficient_reestimations":distinct,"control_monthly_parameter_checks":len(controls),
        "estimation_calls_by_stage":distinct+(distinct-failed)+distinct,"independent_trials":"NOT_ESTABLISHED","global_DSR_PBO":"NOT_COMPUTED"}
    write_json(OUT/"trial_accounting.json",accounting,exclusive=True)
    save_table("同自然原点配对预测",paired)
    save_table("逐周期预测误差",pd.concat(cycle_rows,ignore_index=True))
    save_table("原参数退化复算",pd.DataFrame(controls))
    save_table("逐月方差信息权重",pd.DataFrame(fit_rows))
    write_json(OUT/"prediction_summary.json",{"study":STUDY,"completed_at":now(),"status":"PREDICTION_GATE_PASS_ECONOMIC_STAGE_PENDING" if passed else "REJECTED_FIXED_RANDOM_INTERCEPT_METHOD_PREDICTION_GATE_FAILED",
        "prediction_gate_passed":passed,"periods":diagnostics,"trial_accounting":accounting,"original_parameter_max_error":max(x["max_parameter_error"] for x in controls),
        "candidate_no_view_rows":int(paired.candidate_prediction.isna().sum()),"new_strategy_accounts":0,"account_return_sharpe":"NOT_COMPUTED",
        "independent_validation":"NOT_ESTABLISHED","goal_achieved":False},exclusive=True)
    if not passed:
        write_json(OUT/"economic_stage_status.json",{"at":now(),"status":"SKIPPED_PREDICTION_GATE_FAILED","new_strategy_accounts":0,"account_return_sharpe":"NOT_COMPUTED"},exclusive=True)
    check()
    files=sorted((OUT/"results").glob("*"))+[OUT/"candidate_models.json",OUT/"prediction_summary.json",OUT/"trial_accounting.json"]
    write_json(OUT/"prediction_verification_receipt.json",{"at":now(),"status":"PASS_SAVED_ORIGINAL_PARAMETER_DEGENERATION_AND_FROZEN_SOURCES","sources_unchanged":check(),
        "artifacts":[{"path":str(p.relative_to(ROOT)),"sha256":digest(p)} for p in files]},exclusive=True)
    print(json.dumps({"periods":diagnostics,"trial_accounting":accounting},ensure_ascii=False),flush=True)


def economic_stage():
    require(not (OUT/"ECONOMIC_STARTED.json").exists(),"经济阶段已启动，不重复运行。")
    check()
    pred=json.loads((OUT/"prediction_summary.json").read_text(encoding="utf-8"))
    require(pred["prediction_gate_passed"],"预测门失败，禁止补跑经济账户。")
    write_json(OUT/"ECONOMIC_STARTED.json",{"at":now()},exclusive=True)
    from research import point_core_observation_inputs_v1 as engine
    from research.point_weight_information_inputs_v1 import weight_account
    from research.point_account_nr7_inputs_v1 import PARENT_A, metrics as account_metrics
    from research.point_second_weight_comparison_v1 import annual_rows
    from research import upward_episode_anatomy_v1 as common
    from research.adaptive_allocation_v1 import normalize_dividends
    data=pd.read_parquet(ROOT/CURRENT/"inputs/candidate_features.parquet")
    prices=pd.read_parquet(ROOT/WEIGHT/"prices.parquet");risks=pd.read_parquet(ROOT/WEIGHT/"risks.parquet")
    dividends=normalize_dividends(pd.read_csv(ROOT/CURRENT/"inputs/dividends.csv"))
    execution_data,_=common.features(prices,dividends)
    configs={name:json.loads((ROOT/CURRENT/f"inputs/config/{name}.json").read_text(encoding="utf-8")) for name in CONFIG_NAMES}
    originals=json.loads((ROOT/CURRENT/"inputs/within_models.json").read_text(encoding="utf-8"))["models"]
    candidate=json.loads((OUT/"candidate_models.json").read_text(encoding="utf-8"))["models"]
    globals_copy={**engine.assemble_chain.__globals__,"EntryVintageExitController":RandomInterceptController}
    modified=types.FunctionType(engine.assemble_chain.__code__,globals_copy,"assemble_random_intercept_chain",engine.assemble_chain.__defaults__,engine.assemble_chain.__closure__)
    metrics,checks,annual=[],[],[]
    for period,start,end,next_date in [("2015_2019","2015-01-05","2019-12-31","2020-01-02"),("2020_2026","2020-01-02","2026-09-30","2026-10-08")]:
        frame=data.loc[data.date.le(pd.Timestamp(end))].copy().reset_index(drop=True)
        continuous={}
        for name in ["PANIC_ONLY","REARM_RIDGE"]:
            ledger,decisions,cycles=[pd.read_parquet(ROOT/CURRENT/f"results/continuous/{name}_{kind}.parquet") for kind in ["ledger","decisions","cycles"]]
            continuous[name]=(ledger.loc[ledger.date.le(pd.Timestamp(end))],decisions.loc[decisions.origin.le(pd.Timestamp(end))],cycles)
        baseline=engine.assemble_chain(frame,dividends,configs,[r for r in originals if r["fit_index"]<len(frame)],start,pd.Timestamp(next_date),continuous)
        changed=modified(frame,dividends,configs,[r for r in candidate if r["fit_index"]<len(frame)],start,pd.Timestamp(next_date),continuous)
        saved_path=Path("reports/research/510300_point_core_observation_v1/results/earlier_diagnostic/STRESS/full_factors.parquet") if period=="2015_2019" else CURRENT/"results/STRESS/full_factors.parquet"
        saved=pd.read_parquet(ROOT/saved_path)
        np.testing.assert_allclose(baseline["factors"]["STRESS"][PARENT_A],saved[PARENT_A],atol=1e-11,rtol=0,equal_nan=True)
        for name,chain in [("CONTROL",baseline),("RANDOM_INTERCEPT",changed)]:
            parents=chain["factors"]["STRESS"][["date",PARENT_A]].rename(columns={"date":"origin"})
            for cost in ["BASE","STRESS"]:
                account=weight_account(execution_data.loc[execution_data.date.le(pd.Timestamp(end))].reset_index(drop=True),dividends,parents,risks,cost,start)
                if name=="CONTROL":
                    folder=ROOT/WEIGHT/f"controls/{period}/{cost}/A_SAVED_WEIGHT"
                    for table in ["daily","orders","trades","decisions"]:
                        pd.testing.assert_frame_equal(account[table],pd.read_parquet(folder/f"{table}.parquet"),check_exact=False,atol=1e-9,rtol=0)
                    checks.append({"period":period,"cost":cost,"saved_account_match":True})
                folder=OUT/f"accounts/{period}/{cost}/{name}";folder.mkdir(parents=True)
                for table,value in account.items():
                    if isinstance(value,pd.DataFrame):value.to_parquet(folder/f"{table}.parquet",index=False)
                    else:write_json(folder/f"{table}.json",value)
                year_rows=annual_rows(account,period,name,cost);m=account_metrics(account)
                counts=[r["completed_cycles"] for r in year_rows if r["full_year"]]
                m.update(average_full_year_cycles=float(np.mean(counts)),zero_trade_full_years=sum(n==0 for n in counts))
                metrics.append({"period":period,"cost":cost,"candidate":name,**m});annual.extend(year_rows)
        save_table("候选完整来源_"+period,changed["factors"]["STRESS"])
    comparisons=[]
    for period in ["2015_2019","2020_2026"]:
        for cost in ["BASE","STRESS"]:
            pair={r["candidate"]:r for r in metrics if r["period"]==period and r["cost"]==cost}
            old,new=pair["CONTROL"],pair["RANDOM_INTERCEPT"]
            cagr,sharpe=new["net_cagr"]-old["net_cagr"],new["net_sharpe"]-old["net_sharpe"]
            quality=new["p_times_b"]>1 and new["mean_cycle_net_return"]>0
            comparisons.append({"period":period,"cost":cost,"net_cagr_delta":cagr,"net_sharpe_delta":sharpe,
                                "quality_pass":bool(quality),"joint_pass":bool(cagr>0 and sharpe>0 and quality and new["max_drawdown"]<=.1)})
    passed=all(x["joint_pass"] for x in comparisons)
    save_table("共同账户比较",pd.DataFrame(metrics));save_table("逐年净收益与次数",pd.DataFrame(annual))
    write_json(OUT/"economic_summary.json",{"study":STUDY,"at":now(),"status":"HISTORICAL_JOINT_GATE_PASS_INDEPENDENT_VALIDATION_PENDING" if passed else "REJECTED_FIXED_RANDOM_INTERCEPT_METHOD_ECONOMIC_GATE_FAILED",
        "metrics":metrics,"control_checks":checks,"comparisons":comparisons,"historical_joint_gate_pass":passed,
        "new_strategy_accounts":4,"new_control_replays":4,"new_internal_reference_replays":16,"sources_unchanged":check(),"independent_validation":"NOT_ESTABLISHED","goal_achieved":False},exclusive=True)
    print("四个共同候选账户完成，依冻结经济门报告，独立验证仍未建立。",flush=True)


def main():
    parser=argparse.ArgumentParser(description="隔离周期随机截距方差分量方法")
    parser.add_argument("action",choices=["freeze","predict","account","check"])
    args=parser.parse_args()
    if args.action=="freeze":freeze()
    elif args.action=="predict":fit_and_evaluate()
    elif args.action=="account":economic_stage()
    else:print(f"冻结来源未变：{check()}份。",flush=True)


if __name__=="__main__":
    main()
