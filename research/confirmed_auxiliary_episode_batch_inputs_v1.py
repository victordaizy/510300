"""用两种固定资格状态控制辅助策略的进入，核心目标始终单独保留。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY='EPISODE_START_CONFIRMED_AUXILIARY'
MODELS=['TREND_NOISE_REFERENCE_BLEND','RETURN_RUNS_STATE','RETURN_LAG_STATE','RETURN_SIGN_BALANCE']
CANDIDATES={PRIMARY:'辅助段开始时任一方向确认','EPISODE_WAIT_CONFIRMED_AUXILIARY':'辅助段内等待任一方向确认'}

def confirmed_auxiliary_episode_batch_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['parent_models']==MODELS and cfg['combination']=='FIXED_EPISODE_START_AND_WAIT_DIRECTION_QUALIFICATIONS','辅助资格规则改变')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1); summaries=[]
    for cost,frame in frames.items():
        core=frame[MODELS[0]+'_parent_target'].to_numpy(float); aux=frame[MODELS[1]+'_parent_target'].to_numpy(float)
        dirs=[]
        for model,col in zip(MODELS[2:],['lag_direction','sign_direction']):
            v=parents_by_cost[cost][model].positive_direction.to_numpy(float)
            require((np.isnan(v)|(v==0)|(v==1)).all(),'保存方向值不同')
            full=np.full(len(data),np.nan); full[origins]=v; frame[col]=full; dirs.append(full)
        allowed=(dirs[0]==1)|(dirs[1]==1)
        for model,mode in [(PRIMARY,'START'),('EPISODE_WAIT_CONFIRMED_AUXILIARY','WAIT')]:
            qual=np.full(len(data),np.nan); target=np.full(len(data),np.nan); effective=np.full(len(data),np.nan)
            state=0 # 0: no episode; 1: qualified; 2: failed start; 3: waiting; 4: unknown start
            for t in origins:
                if not np.isfinite(aux[t]):
                    state=0; continue
                if aux[t]==0:
                    state=0
                    if np.isfinite(core[t]): qual[t]=0; effective[t]=0; target[t]=core[t]
                    continue
                direction_known=np.isfinite(dirs[0][t]) and np.isfinite(dirs[1][t])
                if state==0:
                    if not direction_known: state=4
                    elif allowed[t]: state=1
                    else: state=2 if mode=='START' else 3
                elif state==3 and direction_known and allowed[t]: state=1
                if state==1: qual[t]=1
                elif state in [2,3]: qual[t]=0
                if np.isfinite(core[t]) and np.isfinite(qual[t]):
                    effective[t]=aux[t] if qual[t]==1 else 0
                    target[t]=min(1.,core[t]+effective[t])
            frame[model+'_qualified']=qual; frame[model+'_effective_auxiliary_target']=effective; frame[model+'_target']=target
            cur=target[origins]; q=qual[origins]; known=np.isfinite(cur)
            summaries.append({'model':model,'cost':cost,'decision_origins':len(origins),'positive_target_origins':int((cur>0).sum()),'zero_target_origins':int((cur==0).sum()),'unknown_target_origins':int(np.isnan(cur).sum()),'qualified_origins':int((q==1).sum()),'unqualified_origins':int((q==0).sum()),'full_weight_origins':int((cur==1).sum()),'mean_target':float(np.nanmean(cur)) if known.any() else None})
    return frames,summaries
