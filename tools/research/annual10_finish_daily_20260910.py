"""恢复被工具时限中断的季度训练，复用校验通过的已完成模型；交易规则不变。"""
from __future__ import annotations
import sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research.annual10_daily_opportunity_20260910 import *

def main():
    cfg,data,div=setup();require(load_json(OUT/'protocol.json')['code_sha256']==digest(ROOT/'tools/research/annual10_daily_opportunity_20260910.py'),'原冻结代码变化')
    require(not (OUT/'predictions.npz').exists(),'预测已完成，不覆盖')
    labels=read_frame(OUT/'labels.parquet');labels=labels[labels.status.eq('MATURE')].copy()
    x,features=inputs(data);prob=np.full((len(data),4),np.nan);pred=prob.copy();records=[];reused=0;new=0
    cuts=[i for i in range(1,len(data)-1) if data.date.iloc[i].to_period('Q')!=data.date.iloc[i-1].to_period('Q') and data.date.iloc[i]>=pd.Timestamp('2014-01-01')]+[len(data)-1]
    with threadpool_limits(limits=1):
        for k,(t,end) in enumerate(zip(cuts[:-1],cuts[1:])):
            train=labels[labels.exit_index<=t-5];y=train.roi.to_numpy(float);cls=y>0
            record={'fit_index':t,'fit_origin':data.date.iloc[t],'prediction_end_index':end-1,'n':len(train),'status':'NO_VIEW_SUPPORT'}
            if len(train)<400 or min(cls.sum(),(~cls).sum())<30:records.append(record);continue
            weight=concurrency_weights(train,len(data));xt=x[train.origin_index.to_numpy(int)]
            require(np.isfinite(xt).all() and np.isfinite(x[t:end]).all(),'特征缺失')
            train_hash=hashlib.sha256(xt.tobytes()+y.tobytes()+weight.tobytes()).hexdigest()
            gain=np.average(y[cls],weights=weight[cls]);loss=np.average(y[~cls],weights=weight[~cls])
            path=OUT/'models'/f'{t}.joblib'
            if path.exists():
                saved=joblib.load(path);require(saved['record']['training_hash']==train_hash,'恢复模型训练样本改变')
                require(saved['record']['latest_exit_index']==int(train.exit_index.max()),'恢复模型成熟时钟改变')
                mods=saved['models'];reused+=3
            else:
                mods=classifiers()
                for name,m in mods.items():
                    if name=='LOGISTIC':m.fit(xt,cls,standardscaler__sample_weight=weight,logisticregression__sample_weight=weight)
                    else:m.fit(xt,cls,sample_weight=weight)
                    new+=1
            for j,(name,m) in enumerate(mods.items()):prob[t:end,j]=m.predict_proba(x[t:end])[:,1]
            prob[t:end,3]=prob[t:end,:3].mean(axis=1);pred[t:end]=prob[t:end]*gain+(1-prob[t:end])*loss
            record.update(status='FIT_COMPLETE',latest_exit_index=int(train.exit_index.max()),gain_mean=float(gain),loss_mean=float(loss),training_hash=train_hash)
            if not path.exists():joblib.dump({'record':record,'models':mods,'features':features},path,compress=3)
            records.append(record)
    np.savez_compressed(OUT/'predictions.npz',probability=prob,predictions=pred)
    write_json(OUT/'training_records.json',{'records':records,'completed_model_fits':new+reused,'new_in_resume':new,'reused_in_resume':reused,'completed_at':now(),
      'interruption':'Original tool execution timed out after 37 complete quarterly bundles. The first incomplete bundle, if partly fitted, was rerun with unchanged parameters; unpersisted partial fit call count is not available.'})
    print('COMPLETE',new+reused,'REUSED',reused,'NEW',new,flush=True)

if __name__=='__main__':main()
