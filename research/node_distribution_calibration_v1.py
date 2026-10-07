"""固定已有节点，以此前成熟节点校准三项预测；不改节点日期和交易门槛。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit

import sparse_event_training_core_v3 as core
import if_node_joint_payoff_training_v1 as joint
import probability_payoff_prediction_stage_v1 as scoring
import dense_probability_payoff_nodes_v1 as dense


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_node_distribution_calibration_v1"
PRIMARY = "TYPE_OFFSET_CALIBRATION"
MODELS = ["UNCHANGED_PREDICTIONS","SHARED_OFFSET_CALIBRATION",PRIMARY]


def freeze(out):
    core.require(not out.exists(),"本轮目录已经存在。")
    parent = dense.OUT
    cfg = core.load(parent / "protocol.json")
    summary = core.load(parent / "summary.json")
    cfg.update(study_id="510300_NODE_DISTRIBUTION_CALIBRATION_V1",primary_model=PRIMARY,
        parent_study=summary["status"],account_end=summary["primary_accounts"][0]["end"],
        hypothesis="日样本训练后，选中节点的条件分布可能改变；以当时成熟节点估计预测校正量。",
        known_parent_result="新增138节点的代理胜率51.45%，平均收益-0.1939%；主预测高估收益、低估亏损。此信息已查看，本轮是同一家族探索。",
        models_description={"UNCHANGED_PREDICTIONS":"直接复用原预测和原账户作为对照。",
            "SHARED_OFFSET_CALIBRATION":"原logit概率及log幅度作offset，每个头仅学习一个公共校正量。",
            PRIMARY:"相同公共校正量，另加原IF节点指示项；校正系数全部以L2惩罚收缩向0。"},
        target="原10日净收益代理，压力成本24bp；账户仍逐笔计实际费用。",
        fitting="每个候选节点决策前，仅用此前已结束标签的完整候选节点；重叠权重继承父研究，条件组只用于拟合相应幅度。",
        objective="加权平均逻辑损失或Gamma均值损失，加0.5*sum(beta**2)；lambda=1，不搜索。原预测固定为offset。",
        prior_when_no_mature_sample="无成熟节点时校正量0；某方向没有成熟样本时该头校正量0。不会删去这些早期预测或更换评价起点。",
        node_dates="完全固定父研究148节点；本轮没有新的触发类型、阈值或窗口。",
        comparison="原预测、公共校正、按节点类型校正三者并列；原IF/新增节点分别报告，全部及前后半段并列。",
        account_rules="原联合门槛、10日退出、8%账户回撤永久停止、全部空仓日期和2018-05-02起点保持。",
        final_requirements={"capital_cny":200000,"net_sharpe":1.3,"max_drawdown":.1,"cycles_each_complete_year":5},
        branch_budget="只做本次预先固定的分布校准；失败不得改方向、门槛、窗口或重置账户继续拼接。",
        selected_after_results=False,new_node_types=0,new_downloads=0,independent_holdout=False,goal_achieved=False)
    out.mkdir(parents=True)
    sources={}

    def take(source,name):
        dest=out/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
        sources[name]={"source":str(source.relative_to(ROOT)),"sha256":core.digest(dest)}

    for name in ["market.parquet","dividends.csv"]:
        take(parent/"inputs"/name,"inputs/"+name)
    for name,target in [("逐日节点触发表.csv","node_source.csv"),("完整逐日账户.csv","previous_accounts.csv"),
                        ("完整持仓周期.csv","previous_cycles.csv"),("账户联合评价.csv","previous_metrics.csv")]:
        take(parent/"results"/name,"inputs/"+target)
    for name in ["protocol.json","freeze.json","summary.json","prediction_summary.json","implementation_amendment.json"]:
        take(parent/name,"inputs/parent_"+name)
    for name in [Path(__file__).name,"dense_probability_payoff_nodes_v1.py","sparse_event_training_core_v3.py",
        "if_node_joint_payoff_training_v1.py","if_exhaustion_existing_training_v1.py","if_exhaustion_local_features_v1.py","probability_payoff_prediction_stage_v1.py"]:
        take(ROOT/"research"/name,"code/"+name)
    for name in ["510300_existing_data_training_mandate_v1.json","510300_sparse_node_training_contract_v1.json"]:
        take(ROOT/"config"/name,"inputs/"+name)
    core.save(out/"protocol.json",cfg)
    core.save(out/"freeze.json",{"frozen_at":core.now(),"before_calibration_fits":True,"parent_outcomes_already_seen":True,
        "sources":sources,"identities":{"protocol.json":core.digest(out/"protocol.json"),**{k:v["sha256"] for k,v in sources.items()}}})
    print("节点分布校准已固定；保留全部148节点与原账户时段。",flush=True)


def check_inputs(out):
    dense.check_inputs(out)


def candidates(out):
    source=pd.read_csv(out/"inputs/node_source.csv")
    return source.loc[source.S1_PLUS_FORECAST_ONSET].sort_values("decision_at").reset_index(drop=True)


def design(frame,typed):
    return np.column_stack([np.ones(len(frame)),frame.if_event.astype(float)]) if typed else np.ones((len(frame),1))


def fit_head(sample,weight,head,typed):
    mask=np.ones(len(sample),bool) if head=="probability" else (sample.target_net_proxy>0 if head=="gain" else sample.target_net_proxy<0).to_numpy()
    part=sample.loc[mask]
    beta0=np.zeros(2 if typed else 1)
    if not len(part):
        return {"head":head,"typed":typed,"n":0,"beta":beta0.tolist(),"status":"UNCHANGED_PRIOR_NO_MATURE_GROUP"}
    w=weight[mask]/weight[mask].sum()
    x=design(part,typed)
    if head=="probability":
        offset=logit(part.p_win.to_numpy(float))
        y=part.target_profit.to_numpy(float)
    else:
        offset=np.log(part["conditional_"+head].to_numpy(float))
        y=abs(part.target_net_proxy.to_numpy(float))

    def objective(beta):
        eta=offset+x@beta
        if head=="probability":
            value=w@(np.logaddexp(0,eta)-y*eta)
            gradient=x.T@(w*(expit(eta)-y))
        else:
            ratio=y*np.exp(-eta)
            value=w@(ratio+eta)
            gradient=x.T@(w*(1-ratio))
        return float(value+.5*np.sum(beta**2)),gradient+beta

    result=minimize(objective,beta0,jac=True,method="L-BFGS-B",options={"maxiter":1000,"gtol":1e-10,"ftol":1e-14})
    error=float(np.max(abs(objective(result.x)[1])))
    core.require(np.isfinite(result.fun) and np.isfinite(result.x).all() and error<1e-6,"节点校准未满足固定凸目标收敛标准。")
    return {"head":head,"typed":typed,"n":len(part),"beta":result.x.tolist(),"status":"FITTED",
            "gradient_max":error,"optimizer_success_flag":bool(result.success),"optimizer_message":str(result.message)}


def prediction(row,model,name):
    x=design(pd.DataFrame([row]),model["typed"])
    p=float(expit(logit(row.p_win)+(x@model["heads"]["probability"]["beta"])[0]))
    gain=float(row.conditional_gain*np.exp((x@model["heads"]["gain"]["beta"])[0]))
    loss=float(row.conditional_loss*np.exp((x@model["heads"]["loss"]["beta"])[0]))
    result=row.to_dict()
    result.update(model=name,p_win=p,conditional_gain=gain,conditional_loss=loss,expected_net_proxy=p*gain-(1-p)*loss,
                  predicted_payoff_ratio=gain/loss,joint_gate_pass=bool(p>.5 and gain>loss and p*gain-(1-p)*loss>0),n_mature_nodes=model["n_train"])
    return result


def score(predictions):
    groups=[]
    for scope,mask in [("ALL_NODES",np.ones(len(predictions),bool)),("ORIGINAL_IF",predictions.if_event),("ADDED_FORECAST",~predictions.if_event)]:
        part=predictions.loc[mask].copy()
        part["scope"]=scope
        groups.append(part)
    metric,_=scoring.score_predictions(pd.concat(groups,ignore_index=True))
    compare=[]
    for name in MODELS[1:]:
        merged=metric.loc[metric.model==name].merge(metric.loc[metric.model==MODELS[0]],on=["scope","segment","head"],suffixes=("","_reference"),validate="one_to_one")
        for r in merged.itertuples():
            compare.append({"model":name,"reference":MODELS[0],"scope":r.scope,"segment":r.segment,"head":r.head,"n":r.n,
                            "mse_improvement":1-r.mse/r.mse_reference if r.mse_reference>0 else None,"mae_difference":r.mae-r.mae_reference})
    return metric,pd.DataFrame(compare)


def train(out):
    core.require(not (out/"prediction_summary.json").exists(),"校准已训练，不重跑。")
    check_inputs(out)
    source=candidates(out)
    models,predictions=[],[]
    baseline=source.copy()
    baseline["model"],baseline["n_mature_nodes"]=MODELS[0],0
    predictions.extend(baseline.to_dict("records"))
    for name,typed in [(MODELS[1],False),(PRIMARY,True)]:
        for _,row in source.iterrows():
            sample=source.loc[source.label_end_at<row.decision_at]
            w=dense.weights(sample) if len(sample) else np.array([])
            model={"model":name,"typed":typed,"decision_at":row.decision_at,"month":row.month,"n_train":len(sample),
                "train_dates":sample.month.tolist(),"train_max_label_end_at":sample.label_end_at.max() if len(sample) else None,
                "heads":{h:fit_head(sample,w,h,typed) for h in ["probability","gain","loss"]}}
            models.append(model)
            predictions.append(prediction(row,model,name))
        print(f"已完成{name}的全部逐节点校准，未改节点母集。",flush=True)
    result=pd.DataFrame(predictions).sort_values(["model","decision_at"],ignore_index=True)
    core.save(out/"models/逐节点校准参数.json",models)
    core.export(out/"results/完整节点预测.csv",result)
    metric,comparison=score(result)
    core.export(out/"results/预测分项评价.csv",metric)
    core.export(out/"results/固定预测对照.csv",comparison)
    summary={"status":"NODE_CALIBRATION_TRAINING_COMPLETE","completed_at":core.now(),"candidate_nodes":len(source),
        "saved_bundles":len(models),"new_probability_fits":sum(m["heads"]["probability"]["status"]=="FITTED" for m in models),
        "new_amplitude_fits":sum(m["heads"][h]["status"]=="FITTED" for m in models for h in ["gain","loss"]),
        "unfitted_prior_heads":sum(v["status"]!="FITTED" for m in models for v in m["heads"].values()),
        "prediction_rows":len(result),"new_nodes":0,"new_accounts":0,"new_downloads":0,"goal_achieved":False}
    core.save(out/"prediction_summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def accounts(out):
    core.require((out/"prediction_summary.json").exists(),"先训练再运行账户。")
    core.require(not (out/"summary.json").exists(),"账户已经完成。")
    check_inputs(out)
    cfg=core.load(out/"protocol.json")
    source=pd.read_csv(out/"inputs/node_source.csv")
    core.export(out/"results/逐日节点触发表.csv",source)
    predictions=pd.read_csv(out/"results/完整节点预测.csv")
    market=core.read_market(out)
    market=market.loc[market.date<=cfg["account_end"]].copy()
    start_i=int(source.entry_i.min())
    dividend=pd.read_csv(out/"inputs/dividends.csv")
    dividend=dividend.loc[dividend.symbol=="510300.SH"]
    old_nav=pd.read_csv(out/"inputs/previous_accounts.csv")
    old_cycles=pd.read_csv(out/"inputs/previous_cycles.csv")
    old_metrics=pd.read_csv(out/"inputs/previous_metrics.csv")
    navs,cycles,choices,metrics=[],[],[],[]
    for name in cfg["node_stages"]:
        mask=predictions.if_event if name=="S0_IF_ONLY" else np.ones(len(predictions),bool)
        selected=predictions.loc[mask]
        for cost in ["BASE","STRESS"]:
            previous=old_nav.loc[(old_nav.node_stage==name)&(old_nav.model==dense.PRIMARY)&(old_nav.scenario==cost)].copy()
            previous["model"]=MODELS[0]
            navs.append(previous)
            previous_cycles=old_cycles.loc[(old_cycles.node_stage==name)&(old_cycles.model==dense.PRIMARY)&(old_cycles.scenario==cost)].copy()
            previous_cycles["model"]=MODELS[0]
            cycles.extend(previous_cycles.to_dict("records"))
            metric=old_metrics.loc[(old_metrics.node_stage==name)&(old_metrics.model==dense.PRIMARY)&(old_metrics.scenario==cost)].iloc[0].to_dict()
            metric["model"]=MODELS[0]
            for k in ["complete_years","minimum_cycles_in_full_year","annual_frequency_pass","all_numeric_targets_met"]:
                metric.pop(k)
            metrics.append(metric)
            for model in MODELS[1:]:
                local=selected.loc[selected.model==model].copy()
                allowed=local.loc[local.joint_gate_pass].copy()
                allowed["prediction"]=allowed.expected_net_proxy+cfg["net_label_cost_proxy"]
                allowed["prior_prediction_rmse"],allowed["mature_prequential_error_count"]=0.,0
                effective=copy.deepcopy(cfg)
                effective["models"]={model:{}}
                nav,ledger,decision,metric=core.account_run(model,cost,allowed,market,start_i,effective,dividend)
                nav["node_stage"]=name
                navs.append(nav)
                for r in ledger:
                    r["node_stage"]=name
                    cycles.append(r)
                by_date={r["month"]:r for r in decision}
                for r in local.to_dict("records"):
                    choices.append({"node_stage":name,"model":model,"scenario":cost,"month":r["month"],"entry_date":r["entry_date"],
                        "p_win":r["p_win"],"conditional_gain":r["conditional_gain"],"conditional_loss":r["conditional_loss"],
                        "expected_net_proxy":r["expected_net_proxy"],"joint_gate_pass":r["joint_gate_pass"],
                        **by_date.get(r["month"],{"decision":"JOINT_GATE_REJECTED"})})
                metric["node_stage"]=name
                metrics.append(joint.joint_metrics(metric,ledger))
    nav,trade,metric=pd.concat(navs,ignore_index=True),pd.DataFrame(cycles),pd.DataFrame(metrics)
    annual,frequency=dense.frequency_table(nav,trade,cfg)
    metric=metric.merge(frequency,on=["node_stage","model","scenario"],validate="one_to_one")
    metric["all_numeric_targets_met"]=metric.meets_numeric_targets&metric.annual_frequency_pass
    for name,frame in [("完整逐日账户",nav),("完整持仓周期",trade),("逐节点交易决定",pd.DataFrame(choices)),("账户联合评价",metric),("完整自然年交易次数",annual)]:
        core.export(out/"results"/(name+".csv"),frame)
    summary={**core.load(out/"prediction_summary.json"),"status":"NODE_CALIBRATION_AND_ACCOUNT_COMPARISON_COMPLETE","completed_at":core.now(),
        "new_accounts":8,"reused_parent_accounts":4,"total_comparison_accounts":12,"primary_accounts":metric.loc[metric.model==PRIMARY].to_dict("records"),
        "numeric_passed_accounts":int(metric.all_numeric_targets_met.sum()),"independent_validation_established":False,"goal_achieved":False}
    core.save(out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def verify_model(model,source):
    sample=source.loc[source.label_end_at<model["decision_at"]]
    core.require(sample.month.tolist()==model["train_dates"] and len(sample)==model["n_train"],"校准纳入了未成熟标签或遗漏节点。")
    w=dense.weights(sample) if len(sample) else np.array([])
    for h in model["heads"].values():
        head=h["head"]
        mask=np.ones(len(sample),bool) if head=="probability" else (sample.target_net_proxy>0 if head=="gain" else sample.target_net_proxy<0).to_numpy()
        part=sample.loc[mask]
        core.require(len(part)==h["n"],"校准方向样本数不同。")
        beta=np.array(h["beta"])
        if not len(part):
            core.require(np.all(beta==0),"无成熟样本时更改了原预测。")
            continue
        weight=w[mask]/w[mask].sum()
        x=design(part,h["typed"])
        if head=="probability":
            residual=expit(logit(part.p_win.to_numpy(float))+x@beta)-part.target_profit.to_numpy(float)
        else:
            residual=1-abs(part.target_net_proxy.to_numpy(float))/part["conditional_"+head].to_numpy(float)*np.exp(-x@beta)
        gradient=x.T@(weight*residual)+beta
        core.require(np.max(abs(gradient))<1e-6,"保存校准参数未满足固定目标方程。")


def verify(out):
    check_inputs(out)
    source=candidates(out)
    models=core.load(out/"models/逐节点校准参数.json")
    baseline=source.copy()
    baseline["model"],baseline["n_mature_nodes"]=MODELS[0],0
    records=baseline.to_dict("records")
    lookup=source.set_index("month",drop=False)
    for model in models:
        verify_model(model,source)
        records.append(prediction(lookup.loc[model["month"]],model,model["model"]))
    rebuilt=pd.DataFrame(records).sort_values(["model","decision_at"],ignore_index=True)
    pd.testing.assert_frame_equal(rebuilt,pd.read_csv(out/"results/完整节点预测.csv"),check_dtype=False,rtol=1e-10,atol=1e-12)
    metric,comparison=score(rebuilt)
    for frame,name in [(metric,"预测分项评价.csv"),(comparison,"固定预测对照.csv")]:
        pd.testing.assert_frame_equal(frame,pd.read_csv(out/"results"/name),check_dtype=False,rtol=1e-9,atol=1e-12)
    verification={"saved_accounts":0}
    if (out/"summary.json").exists():
        cfg=core.load(out/"protocol.json")
        market=core.read_market(out)
        market=market.loc[market.date<=cfg["account_end"]].copy()
        # 借用父研究的完整账户复算，节点表保持直接输入，不重新运行资金账户。
        daily_source=pd.read_csv(out/"inputs/node_source.csv")
        node_check=dense.build_nodes(daily_source)
        pd.testing.assert_frame_equal(node_check.reset_index(drop=True),daily_source,check_dtype=False,rtol=1e-10,atol=1e-12)
        verification=dense.verify_accounts(out,market,cfg,daily_source)
    return {"status":"PASS_SAVED_NODE_CALIBRATION_RECOMPUTATION","checked_at":core.now(),"saved_bundles":len(models),
            "prediction_rows":len(rebuilt),**verification,"new_fits":0,"new_accounts":0,"new_downloads":0}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="固定节点的概率和幅度分布校准。")
    parser.add_argument("command",choices=["freeze","train","accounts","verify"])
    parser.add_argument("--root",type=Path,default=OUT)
    args=parser.parse_args()
    if args.command=="verify":
        print(json.dumps(verify(args.root),ensure_ascii=False,indent=2))
    else:
        {"freeze":freeze,"train":train,"accounts":accounts}[args.command](args.root)
