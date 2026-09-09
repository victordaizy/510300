"""用三个已知市场状态学习两父组合未来相对净收益，严格等待目标成熟。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.intraday_overnight_increment_v1 import require

FEATURES=["mom20","sma120","vol20"]
CN=["二十日含分红对数涨跌","一百二十日财富均线偏离","二十日含分红日波动"]
PARENTS=["TWO_POLICY_MIN_VARIANCE","THREE_POLICY_MIN_VARIANCE"]


def target_frame(data,reference_returns,first,horizon=20):
    returns=np.asarray(reference_returns,dtype=float)
    require(returns.shape==(len(data),2) and 1<=first<len(data)-1,"父组合标签输入维度或起点不符")
    records=[]
    for t in range(first-1,len(data)-1):
        end=t+horizon;values=np.full(2,np.nan);status="NO_VIEW_TERMINAL_OR_UNFINISHED_TARGET"
        if end<len(data)-1:
            window=returns[t+1:end+1]
            if np.isfinite(window).all() and (window>-1).all():
                values=np.prod(1+window,axis=0)-1;status="COMPLETE_HISTORICAL_TARGET"
            else:status="NO_VIEW_INCOMPLETE_TARGET_INPUT"
        records.append({"origin_index":t,"origin":data.date.iloc[t],"maturity_index":end,
            "maturity_date":data.date.iloc[end] if end<len(data)-1 else pd.NaT,"target_status":status,
            "two_future_return":values[0],"three_future_return":values[1],"target":values[1]-values[0],
            **{name:float(data[name].iloc[t]) for name in FEATURES}})
    return pd.DataFrame(records)


def mature_rows(labels,t,config):
    return labels[labels.maturity_index.le(t)].tail(config["training_window"]).copy()


def fit_advantage(rows,config):
    x=rows[FEATURES].to_numpy(float);y=rows.target.to_numpy(float)
    require(len(rows)>0 and np.isfinite(x).all() and np.isfinite(y).all(),"条件相对收益训练输入缺失")
    mean=x.mean(axis=0);scale=x.std(axis=0,ddof=0);scale=np.where(scale>1e-12,scale,1.)
    z=np.clip((x-mean)/scale,-config["feature_clip"],config["feature_clip"])
    fitted=Ridge(alpha=config["ridge_alpha"],solver="svd",fit_intercept=True).fit(z,y)
    if not np.isfinite(fitted.coef_).all() or not np.isfinite(fitted.intercept_):raise FloatingPointError("条件收益模型系数非有限")
    return {"kind":"CONTEXTUAL_PARENT_RELATIVE_RIDGE","features":FEATURES.copy(),"mean":mean.tolist(),"scale":scale.tolist(),
        "coefficients":fitted.coef_.tolist(),"intercept":float(fitted.intercept_),"feature_clip":config["feature_clip"]}


def predict_advantage(model,values):
    require(model["kind"]=="CONTEXTUAL_PARENT_RELATIVE_RIDGE" and model["features"]==FEATURES,"条件相对收益模型身份错误")
    z=np.clip((np.asarray(values)-model["mean"])/model["scale"],-model["feature_clip"],model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


def build_policy(data,reference_returns,parent_targets,first,config):
    dates=pd.DatetimeIndex(data.date);targets=np.asarray(parent_targets,dtype=float)
    require(data.index.equals(pd.RangeIndex(len(data))) and dates.is_monotonic_increasing and not dates.has_duplicates,"条件模型原点必须完整递增")
    require(targets.shape==(len(data),2),"父组合股票目标维度不符")
    require((np.isnan(targets)|(np.isfinite(targets)&(targets>=0)&(targets<=1))).all(),"父组合目标超出无杠杆范围")
    labels=target_frame(data,reference_returns,first,config["target_horizon"])
    selected=0;status="INITIAL_TWO_POLICY_PARENT";estimate=np.nan;prediction_origin=pd.NaT;prediction_values=np.full(3,np.nan)
    rows_out=[];models=[]
    for t in range(len(data)):
        outside=t<first-1 or t==len(data)-1;scheduled=False;changed=False
        if not outside and t>=first and dates[t].to_period("M")!=dates[t-1].to_period("M"):
            scheduled=True;prior=selected;training=mature_rows(labels,t,config);stored=None;failure=None;attempted=False
            eligible=len(training)>=config["minimum_training_rows"]
            training_valid=np.isfinite(training[FEATURES+["target"]].to_numpy(float)).all()
            model_status="NO_VIEW_INSUFFICIENT_MATURE_ROWS"
            if eligible and not training_valid:model_status="NO_VIEW_INCOMPLETE_TRAINING_INPUT"
            elif eligible:
                attempted=True
                try:stored=fit_advantage(training,config);model_status="FIT_COMPLETE"
                except (RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:model_status="NO_VIEW_MODEL_FIT_FAILED";failure=str(error)
            require(not len(training) or int(training.maturity_index.max())<=t,"条件训练使用未成熟标签")
            prediction_values=data[FEATURES].iloc[t].to_numpy(float);prediction_origin=dates[t];estimate=np.nan;selected=0
            status=model_status
            if stored is not None and np.isfinite(prediction_values).all():
                estimate=predict_advantage(stored,prediction_values);require(np.isfinite(estimate),"条件收益预测非有限")
                selected=int(estimate>0);status="PREDICTION_AVAILABLE"
            elif stored is not None:status="NO_VIEW_INCOMPLETE_CURRENT_FEATURES"
            changed=selected!=prior
            models.append({"fit_index":t,"fit_origin":dates[t],"fit_time":dates[t]+pd.Timedelta(hours=15,minutes=5),
                "status":model_status,"prediction_status":status,"eligible_by_count":eligible,"fit_attempted":attempted,
                "training_rows":len(training),"training_start_index":int(training.origin_index.iloc[0]) if len(training) else None,
                "training_end_index":int(training.origin_index.iloc[-1]) if len(training) else None,
                "latest_maturity_index":int(training.maturity_index.max()) if len(training) else None,
                "missing_training_rows":int((~np.isfinite(training[FEATURES+["target"]].to_numpy(float)).all(axis=1)).sum()),
                "prediction":estimate,"selected_parent":PARENTS[selected],"model":stored,"failure":failure})
        target=np.nan if outside else float(targets[t,selected])
        rows_out.append({"date":dates[t],"selected_parent":None if outside else PARENTS[selected],"target":target,
            "two_parent_target":targets[t,0],"three_parent_target":targets[t,1],"relative_prediction":np.nan if outside else estimate,
            "prediction_origin":pd.NaT if outside else prediction_origin,"selection_status":"NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status,
            "selection_update_scheduled":scheduled,"selection_changed":changed,
            **{"prediction_"+name:np.nan if outside else float(value) for name,value in zip(FEATURES,prediction_values)}})
    return pd.DataFrame(rows_out),labels,models
