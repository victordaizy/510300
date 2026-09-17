"""在原174完整账户风险上同时计算12%和15%两档预算。"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.account_volatility_exposure_inputs_v1 import risk_budget
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1'
PARENT='EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS=[PARENT]
PRIMARY='ACCOUNT_RISK_12'
RISK_TARGETS={'ACCOUNT_RISK_12':.12,'ACCOUNT_RISK_15':.15}
CANDIDATES={'ACCOUNT_RISK_12':'12%账户风险预算','ACCOUNT_RISK_15':'15%账户风险预算'}


def calibrated_paths(parent,realized,first,cfg):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['risk_targets']==RISK_TARGETS
        and cfg['account_risk_window']==60,'固定两档风险预算或窗口不同')
    return {model:risk_budget(parent,realized,first,cfg['annual_days'],60,budget) for model,budget in RISK_TARGETS.items()}


def calibration_frames(data,parents_by_cost,cfg,start):
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    period='evaluation' if pd.Timestamp(start)==pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    origins=np.arange(first-1,len(data)-1)
    rows=[]
    for cost,frame in frames.items():
        ledger=pd.read_parquet(SOURCE/period/cost/f'{PARENT}_ledger.parquet')
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first:])),'来源收益日历错位')
        realized=np.full(len(data),np.nan)
        realized[first:]=ledger.net_return.to_numpy(float)
        parent=frame[PARENT+'_parent_target'].to_numpy(float)
        paths=calibrated_paths(parent,realized,first,cfg)
        frame['source_realized_net_return']=realized
        frame['source_account_volatility60']=paths[PRIMARY][1]
        for model,(target,risk,multiplier) in paths.items():
            np.testing.assert_allclose(risk,paths[PRIMARY][1],atol=0,rtol=0,equal_nan=True)
            frame[model+'_target']=target
            frame[model+'_multiplier']=multiplier
            values=target[origins]
            rows.append({'model':model,'cost':cost,'risk_budget':RISK_TARGETS[model],'decision_origins':len(origins),
                'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                'unknown_target_origins':int(np.isnan(values).sum()),'full_target_origins':int((values==1).sum()),
                'mean_target':float(np.nanmean(values))})
    return frames,rows
