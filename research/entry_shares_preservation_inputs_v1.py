"""逐条沿用同费用第181轮目标，执行层单独决定保持实际份额。"""
import numpy as np
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT]
PRIMARY='ENTRY_SHARES_PRESERVATION'
CANDIDATES={PRIMARY:'正目标期间保持初次实际买入份额'}


def preservation_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['holding_positive_policy']=='KEEP_ACTUAL_SHARES', '固定执行规则不同')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    summaries=[]
    for cost,frame in frames.items():
        source=frame[PARENT+'_parent_target'].to_numpy(float)
        frame[PRIMARY+'_target']=source.copy()
        values=source[origins]
        summaries.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),
            'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
            'unknown_target_origins':int(np.isnan(values).sum()),'source_target_changes':0})
    return frames,summaries
