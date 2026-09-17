"""一次批量重放固定两来源方案的必要节点，核对原区间并连续至八月末。"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends,target_request,summarize
from research.intraday_overnight_increment_v1 import now,digest,require,write_json
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account,simulate_policy,simulate_rearmed_exit,unpack
from research.post_selection_continuous_factors_v1 import align_decisions,align_returns,planned_weights,ordinary_multiplier,direction_targets,support_choice,joint_downside_budgets,continuous_minimum_variance_budget
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
from research.learned_cycle_exit_v1 import ExitController
from research.simple_intraday_protection_v1 import make_rules as session_rules
from research.simple_volume_reversal_v1 import make_rules as panic_rules
from research.simple_price_entry_exit_v1 import signals
from research.return_runs_state_inputs_v1 import return_runs_state_factors
from research.return_lag_state_inputs_v1 import return_lag_state_factors
from research.return_sign_balance_inputs_v1 import return_sign_balance_factors
from research.downside_reference_risk_inputs_v1 import moment_risk_frame
from research.trend_noise_reference_blend_inputs_v1 import market_amplitudes
from research.account_volatility_exposure_inputs_v1 import risk_budget
from research.episode_account_risk_budget_inputs_v1 import fixed_episode_budget
from research.two_close_zero_exit_inputs_v1 import confirmation_request,zero_streak
from research.addition_gate_exposure_batch_inputs_v1 import gate_request as addition_request,market_factors
from research.incremental_selected_intent_mix_inputs_v1 import gate_request as master_request

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_post_selection_continuous_replay_v1'
CONFIG=ROOT/'config/510300_post_selection_continuous_replay_v1.json'
INPUTS=ROOT/'reports/research/510300_post_selection_extension_inputs_v1'
PRIMARY='SELECTED_MIX_BAND10_SIMPLE2'
MAIN='2020-01-02'
CUTOFF='2026-08-31'


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def old_configuration(name):
    return read('config/510300_'+name+'_v1.json')


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(),'本轮已冻结或开始，请接续保存输出')
    index=read('reports/research/510300_sharpe_1_2_latest_research.json')
    require(index['latest_completed_round']['round']==212 and not index['running_studies'],'前序状态不同')
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    tested=subprocess.run([sys.executable,'-m','pytest','tests/test_post_selection_continuous_accounts_v1.py',
        'tests/test_post_selection_continuous_factors_v1.py','-q','-p','no:cacheprovider'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr,encoding='utf-8')
    print(tested.stdout,flush=True)
    require(tested.returncode==0 and '10 passed' in tested.stdout,'连续引擎及目标必要验证未通过')
    write_json(OUT/'tests_receipt.json',{'recorded_at':now(),'exit_code':0,'passed':10,'seconds':time.perf_counter()-started},exclusive=True)
    old=old_configuration('incremental_selected_intent_mix')
    keys=['evaluation_start','initial_capital','lot','tick','limit_fraction','annual_days','cash_annual_rate_assumption',
        'high_sharpe_target','annual_return_target','costs','dividends','earlier_start','earlier_terminal','weight_band']
    cfg={k:old[k] for k in keys}
    cfg.update(study_id='510300_POST_SELECTION_CONTINUOUS_REPLAY_V1',round=213,registered_at=now(),primary=PRIMARY,
        candidate_models=[PRIMARY],candidate_configurations=0,data_cutoff=CUTOFF,
        features=str((INPUTS/'candidate_features.parquet').relative_to(ROOT)),
        rules='docs/510300_POST_SELECTION_CONTINUOUS_REPLAY_V1.md',planned_account_replays=22,
        planned_existing_reference_replays=2,new_model_fits=0,new_reference_accounts=0,goal_achieved=False,
        evidence_class='FIXED_STRATEGY_PRE_SELECTION_HISTORICAL_GAP_CONTINUOUS_REPLAY',
        current_period_mark='LAST_COMPLETE_CLOSE_WITHOUT_ARTIFICIAL_LIQUIDATION',position_impact=0,
        independent_validation='NOT_ESTABLISHED',source_budget_cny=0,strict_forward_evidence_days=0)
    rules=(ROOT/'docs/510300_MINIMAL_CONTINUOUS_REPLAY_NEXT_20260913.md').read_text(encoding='utf-8')
    (ROOT/cfg['rules']).write_text(rules.replace('本项尚未实现、登记或运行。','本项已完成实现和必要测试，现于首次完整来源重放之前冻结。'),encoding='utf-8')
    graph=read(INPUTS/'dependency_graph.json')
    paths=[Path(__file__),ROOT/'research/post_selection_continuous_accounts_v1.py',ROOT/'research/post_selection_continuous_factors_v1.py',
        ROOT/'tests/test_post_selection_continuous_accounts_v1.py',ROOT/'tests/test_post_selection_continuous_factors_v1.py',
        ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/cfg['rules'],OUT/'tests_receipt.json',
        INPUTS/'dependency_graph.json',INPUTS/'required_account_state_inventory.csv',INPUTS/'saved_verification_receipt.json',
        ROOT/'data/reference/sse_trade_calendar_2026.csv',ROOT/'research/event_account_indexed_request_v1.py',
        ROOT/'research/simple_price_entry_exit_v1.py',ROOT/'research/rearmed_cycle_exit_account_v1.py',
        ROOT/'research/two_policy_min_variance_inputs_v1.py',ROOT/'research/three_source_order_intent_mix_inputs_v1.py']
    paths.extend(ROOT/n[k] for n in graph['nodes'] for k in ['configuration','program'])
    paths.extend([ROOT/'reports/research/510300_within_cycle_exit_v1/saved_models.json',ROOT/'reports/research/510300_learned_cycle_exit_v1/saved_models.json'])
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    index['running_studies']=[{'round':213,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True,status='MINIMAL_CONTINUOUS_REPLAY_FROZEN',source=cfg['rules'])
    write_json(ROOT/'reports/research/510300_sharpe_1_2_latest_research.json',index)
    print('第213轮固定方案的22条必要账户已冻结，直接开始连续重放。',flush=True)


class Pipeline:
    def __init__(self,cfg):
        self.cfg=cfg
        full=pd.read_parquet(ROOT/cfg['features'])
        self.data=full[full.date.le(CUTOFF)].reset_index(drop=True)
        require(str(self.data.date.iloc[-1].date())==CUTOFF,'连续末日不同')
        self.div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
        self.first=int(np.flatnonzero(self.data.date.ge(MAIN))[0])
        calendar=pd.read_csv(ROOT/'data/reference/sse_trade_calendar_2026.csv')
        dates=pd.to_datetime(calendar.trade_date)
        self.next_date=str(dates[dates>pd.Timestamp(CUTOFF)].iloc[0].date())
        self.graph={n['node']:n for n in read(INPUTS/'dependency_graph.json')['nodes']}
        self.inventory={(r['node'],r['cost']):r for r in pd.read_csv(INPUTS/'required_account_state_inventory.csv').to_dict('records')}
        self.accounts={};self.targets={};self.checks=[];self.target_checks=[];self.settings={};self.calls={}
        self.within=read('reports/research/510300_within_cycle_exit_v1/saved_models.json')['models']
        self.ridge=read('reports/research/510300_learned_cycle_exit_v1/saved_models.json')['models']['D60_INTRA__RIDGE']
        self.session=session_rules(self.data)['D60_INTRA']

    def setting(self,key):
        if key not in self.settings:
            self.settings[key]=read(self.graph[key]['configuration'])
        return self.settings[key]

    def target(self,key,cost,values,check=True):
        values=np.asarray(values,float)
        require(len(values)==len(self.data) and (np.isnan(values)|(np.isfinite(values)&(values>=0)&(values<=1))).all(),'来源目标无效：'+key)
        self.targets[(key,cost)]=values
        if check and key in self.graph:
            folder=ROOT/self.graph[key]['saved_folder']/'evaluation'/cost
            path=folder/(key+'_decisions.parquet')
            if path.is_file():
                saved=pd.read_parquet(path)
                ids=saved.origin_index.to_numpy(int)
                expected=saved.reference_weight.to_numpy(float)
                np.testing.assert_allclose(values[ids],expected,atol=1e-12,rtol=1e-12,equal_nan=True,err_msg='旧目标前缀改变：'+key+'/'+cost)
                error=np.abs(values[ids]-expected)
                self.target_checks.append({'node':key,'cost':cost,'rows':len(ids),'maximum_absolute_difference':float(np.nanmax(error))})
        return values

    def get(self,key,cost):
        return self.targets[(key,cost)]

    def save_factors(self,name,frame):
        folder=OUT/'factors';folder.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(folder/(name+'.parquet'),index=False)

    def verify_prefix(self,key,cost,ledger,decisions):
        source=self.inventory[(key,cost)]
        original=pd.read_parquet(ROOT/source['ledger'])
        old_decisions=pd.read_parquet(ROOT/source['decisions'])
        old=original[original.date.le('2026-08-13')].reset_index(drop=True)
        new=ledger[ledger.date.le('2026-08-13')].reset_index(drop=True)
        require(pd.DatetimeIndex(old.date).equals(pd.DatetimeIndex(new.date)),'旧账本日期不同：'+key)
        common=[c for c in old.select_dtypes(include=['number','bool']).columns if c in new]
        for c in common:
            np.testing.assert_allclose(new[c].to_numpy(float),old[c].to_numpy(float),rtol=0,atol=1e-6,equal_nan=True,err_msg='旧账本改变：'+key+'/'+cost+'/'+c)
        for c in ['shares','requested_quantity','filled_quantity','status','mark_clock']:
            require(new[c].tolist()==old[c].tolist(),'旧账户整数或状态改变：'+key+'/'+c)
        chosen=decisions[decisions.origin.le('2026-08-13')].reset_index(drop=True)
        require(pd.DatetimeIndex(chosen.origin).equals(pd.DatetimeIndex(old_decisions.origin)) and
            pd.DatetimeIndex(chosen.execution_date).equals(pd.DatetimeIndex(old_decisions.execution_date)),'旧决定时钟改变：'+key)
        for c in ['requested_quantity','reference_weight','negative_confirmation_count','entry_mode','entry_rearmed']:
            if c in old_decisions and c in chosen:
                np.testing.assert_allclose(chosen[c].to_numpy(float),old_decisions[c].to_numpy(float),atol=1e-12,rtol=1e-12,equal_nan=True,err_msg='旧申请改变：'+key+'/'+c)
        for c in ['action','exit_reasons','fixed_prediction_identity']:
            if c in old_decisions and c in chosen:
                require(chosen[c].fillna('').tolist()==old_decisions[c].fillna('').tolist(),'旧决定状态改变：'+key+'/'+c)
        self.checks.append({'node':key,'cost':cost,'old_ledger_rows':len(old),'old_decision_rows':len(chosen),
            'numeric_ledger_columns':len(common),'maximum_equity_difference':float(np.max(np.abs(new.equity-old.equity))),
            'new_continuous_rows':len(ledger),'status':'PASS_OLD_NORMAL_PREFIX'})

    def account(self,key,cost,kind='target',values=None,rule=None,spec=None,controller_factory=None,request=None):
        require((key,cost) not in self.accounts,'共享账户被重复计算：'+key)
        cfg=self.setting(key)
        start=self.graph[key]['replay_start']
        args=(self.data,self.div,cfg,cfg['costs'][cost],start)
        kwargs={'next_execution_date':self.next_date}
        if kind=='target':
            fn=simulate_indexed_request_account
            policy=request or (lambda a,p,v,c,m,t:target_request(a,p,v,c))
            args+= (key,)
            kwargs.update(targets=values,event_mask=np.ones(len(self.data),bool),request_policy=policy)
        else:
            fn=simulate_rearmed_exit if kind=='rearmed' else simulate_policy
            args+=(rule,spec)
            if kind=='rearmed':
                kwargs['controller']=controller_factory()
        self.calls[(key,cost)]=(fn,args,kwargs,controller_factory)
        folder=OUT/'accounts'/cost/key;folder.mkdir(parents=True,exist_ok=True)
        run=fn(*args,**kwargs)
        ledger,decisions,state=run[0],run[1],run[-1]
        decisions['decision_time']=pd.to_datetime(decisions.origin)+pd.Timedelta(hours=15,minutes=5)
        decisions['source_model']=key;decisions['source_cost']=cost
        ledger.to_parquet(folder/'ledger.parquet',index=False);decisions.to_parquet(folder/'decisions.parquet',index=False)
        if kind!='target':
            run[2].to_csv(folder/'cycles.csv',index=False,encoding='utf-8-sig')
        write_json(folder/'checkpoint.json',state,exclusive=True)
        self.verify_prefix(key,cost,ledger,decisions)
        self.accounts[(key,cost)]=(ledger,decisions,state)
        self.target(key,cost,align_decisions(self.data,decisions),check=False)
        print(f'连续账户 {len(self.accounts)}/22：{key}／{cost}；原正常区间一致。',flush=True)
        return ledger,decisions,state

    def run(self):
        data=self.data;first=self.first;costs=list(self.cfg['costs'])
        cfg128=self.setting('ENTRY_VINTAGE_EXIT')
        for cost in costs:
            self.account('ENTRY_VINTAGE_EXIT',cost,'rearmed',rule=self.session,spec=cfg128['specification'],
                controller_factory=lambda:EntryVintageExitController(data,self.within,2))
        ordinary=ordinary_multiplier(data)
        downside=moment_risk_frame(data,self.setting('DOWNSIDE_REFERENCE_RISK'))
        self.save_factors('ordinary_and_downside',pd.DataFrame({'date':data.date,'ordinary_multiplier':ordinary,
            'downside_multiplier':downside.DOWNSIDE_REFERENCE_RISK_multiplier}))
        for cost in costs:
            vintage=self.get('ENTRY_VINTAGE_EXIT',cost)
            self.target('VINTAGE_REFERENCE_RISK',cost,vintage*ordinary)
            self.target('DOWNSIDE_REFERENCE_RISK',cost,vintage*downside.DOWNSIDE_REFERENCE_RISK_multiplier.to_numpy(float))
        self.account('DOWNSIDE_REFERENCE_RISK','BASE',values=self.get('DOWNSIDE_REFERENCE_RISK','BASE'))
        cfg91=self.setting('CONTINUOUS_REFERENCE_MIN_VARIANCE')
        self.account('PANIC_ONLY','BASE','price',rule=panic_rules(data)[0]['V6_PANIC_RECOVERY'],spec=cfg91['panic_spec'])
        self.account('REARM_RIDGE_CONTINUOUS','BASE','rearmed',rule=self.session,spec=cfg91['learned_spec'],
            controller_factory=lambda:ExitController(data,self.ridge,2))
        reference_first=int(np.flatnonzero(data.date.ge(cfg91['reference_start']))[0])
        rets=np.column_stack([align_returns(data,self.accounts[(k,'BASE')][0]) for k in ['PANIC_ONLY','REARM_RIDGE_CONTINUOUS']])
        states=np.column_stack([self.get(k,'BASE') for k in ['PANIC_ONLY','REARM_RIDGE_CONTINUOUS']])
        budget=continuous_minimum_variance_budget(data.date,rets,states,reference_first,242)
        self.save_factors('continuous_reference_budget',budget)
        for cost in costs:
            self.target('CONTINUOUS_REFERENCE_MIN_VARIANCE',cost,budget.target)
        self.account('CONTINUOUS_REFERENCE_MIN_VARIANCE','BASE',values=budget.target.to_numpy(float))
        pair=['DOWNSIDE_REFERENCE_RISK','CONTINUOUS_REFERENCE_MIN_VARIANCE']
        pair_returns=np.column_stack([align_returns(data,self.accounts[(k,'BASE')][0]) for k in pair])
        weights,joint=joint_downside_budgets(data,pair_returns,first)
        self.save_factors('joint_downside_budget',joint)
        choice=support_choice(data,self.within,first)
        positive,noise=market_amplitudes(data,self.setting('TREND_NOISE_REFERENCE_BLEND'))
        known=np.isfinite(positive)&np.isfinite(noise)
        trend_budget=np.full(len(data),np.nan)
        denominator=positive+noise
        trend_budget[known]=np.divide(positive[known],denominator[known],out=np.zeros(known.sum()),where=denominator[known]>0)
        for cost in costs:
            joint_target=self.target('JOINT_DOWNSIDE_REFERENCE_PAIR',cost,np.sum(weights*np.column_stack([self.get(k,cost) for k in pair]),axis=1))
            routed=self.target('MODEL_SUPPORT_REFERENCE_ROUTER',cost,np.where(choice,joint_target,self.get('VINTAGE_REFERENCE_RISK',cost)))
            self.target('TREND_NOISE_REFERENCE_BLEND',cost,np.clip(trend_budget*self.get('VINTAGE_REFERENCE_RISK',cost)+(1-trend_budget)*routed,0,1))
        directional={}
        for key,func in [('RETURN_RUNS_STATE',return_runs_state_factors),('RETURN_LAG_STATE',return_lag_state_factors),('RETURN_SIGN_BALANCE',return_sign_balance_factors)]:
            f=func(data);directional[key]=f;self.save_factors(key,f)
            values=direction_targets(f,first)
            for cost in costs:self.target(key,cost,values)
        lag=directional['RETURN_LAG_STATE'].positive_direction.to_numpy(float)
        sign=directional['RETURN_SIGN_BALANCE'].positive_direction.to_numpy(float)
        for cost in costs:
            core=self.get('TREND_NOISE_REFERENCE_BLEND',cost);aux=self.get('RETURN_RUNS_STATE',cost)
            known=np.isfinite(core)&np.isfinite(aux)&np.isfinite(lag)&np.isfinite(sign)
            allowed=(lag==1.)|(sign==1.)
            effective=np.full(len(data),np.nan);effective[known]=0.;effective[known&allowed]=aux[known&allowed]
            target=np.full(len(data),np.nan);target[known]=np.minimum(1.,core[known]+effective[known])
            self.target('EITHER_CONFIRMED_RUNS_AUXILIARY',cost,target)
            self.account('EITHER_CONFIRMED_RUNS_AUXILIARY',cost,values=target)
            returns=align_returns(data,self.accounts[('EITHER_CONFIRMED_RUNS_AUXILIARY',cost)][0])
            daily,risk,mult=risk_budget(target,returns,first,242,60,.1)
            self.target('ACCOUNT_VOLATILITY_EXPOSURE',cost,daily)
            unused,risk30,current=risk_budget(target,returns,first,242,30,.1)
            episode,fixed,starts=fixed_episode_budget(target,current)
            self.target('RISK_EPISODE_30_10',cost,episode)
            self.save_factors('account_risk_'+cost,pd.DataFrame({'date':data.date,'source_return':returns,'source_target':target,
                'risk60':risk,'multiplier60':mult,'risk30':risk30,'multiplier30':current,'episode_multiplier':fixed,'episode_start':starts}))
            for key,parent in [('TWO_CLOSE_ZERO_EXIT','ACCOUNT_VOLATILITY_EXPOSURE'),('CONFIRMED_RISK_EPISODE_30_10','RISK_EPISODE_30_10')]:
                values=self.get(parent,cost);counts=zero_streak(values)
                def request(a,p,v,c,m,t,counts=counts,key=key,parent=parent):
                    result=confirmation_request(a,p,v,c,'TWO_CLOSE_ZERO_EXIT',int(counts[t]))
                    if key!='TWO_CLOSE_ZERO_EXIT':result.update(output_candidate=key,budget_source=parent,exit_policy='TWO_CLOSE_ZERO_EXIT')
                    return result
                self.account(key,cost,values=values,request=request)
            cfg23=self.setting('R2_Z_CONFIRM')
            self.account('R2_Z_CONFIRM',cost,'price',rule=signals(data)['R2_Z_CONFIRM'],spec=cfg23['candidate_specs']['R2_Z_CONFIRM'])
            rebound=np.minimum(1.,daily+.5*self.get('R2_Z_CONFIRM',cost))
            self.target('MEAN_REBOUND_AUX_50',cost,rebound)
            self.account('MEAN_REBOUND_AUX_50',cost,values=rebound)
            sources=['TWO_CLOSE_ZERO_EXIT','CONFIRMED_RISK_EPISODE_30_10','MEAN_REBOUND_AUX_50']
            combined=sum(weight*planned_weights(data,self.accounts[(k,cost)][1],self.accounts[(k,cost)][0],self.cfg,first) for k,weight in zip(sources,[.8,.15,.05]))
            self.target('INTENT_MIX_80_15_05',cost,combined)
            self.target('INTENT_MIX_BAND_20',cost,combined)
            exposure=np.minimum(1.,combined*1.15)
            self.target('ADD_GATE_EXPOSURE_115',cost,exposure)
            market=market_factors(data);allowed=market.buy_allowed_lower_r01.to_numpy(bool);daily_return=market.daily_total_simple.to_numpy(float)
            self.account('ADD_GATE_EXPOSURE_115',cost,values=exposure,
                request=lambda a,p,v,c,m,t: addition_request(a,p,v,c,m,bool(allowed[t]),daily_return[t]))
            runs=np.minimum(1.,core+aux)
            self.target('RUNS_OPPORTUNITY_CAPPED_SUM',cost,runs)
            self.account('RUNS_OPPORTUNITY_CAPPED_SUM',cost,values=runs)
            plans=[planned_weights(data,self.accounts[(k,cost)][1],self.accounts[(k,cost)][0],self.cfg,first) for k in ['ADD_GATE_EXPOSURE_115','RUNS_OPPORTUNITY_CAPPED_SUM']]
            master=np.minimum(1.,.85*plans[0]+.15*plans[1])
            self.target(PRIMARY,cost,master)
            self.account(PRIMARY,cost,values=master,
                request=lambda a,p,v,c,m,t:master_request(a,p,v,c,m,True,daily_return[t]))
        require(len(self.accounts)==22,'实际重放账户数量与最小依赖不同')
        self.save_factors('all_required_targets',pd.DataFrame({'date':data.date,**{key+'__'+cost:value for (key,cost),value in self.targets.items()}}))
        return self

    def verify_actual_resume(self):
        key,cost='ENTRY_VINTAGE_EXIT','STRESS'
        ledger,decisions,final=self.accounts[(key,cost)]
        eligible=ledger[ledger.shares.gt(0)&ledger.date.le('2026-08-13')]
        require(len(eligible)>0,'必要持仓中恢复验证缺少实际周期')
        split_date=eligible.date.iloc[-1]
        split=int(np.flatnonzero(self.data.date.eq(split_date))[0])
        fn,args,kwargs,factory=self.calls[(key,cost)]
        first=fn(*args,**{**kwargs,'controller':factory(),'stop_index':split})
        snapshot_path=OUT/'actual_held_split_checkpoint.json'
        write_json(snapshot_path,first[-1],exclusive=True)
        resumed=fn(*args,**{**kwargs,'controller':factory(),'resume':read(snapshot_path)})
        pd.testing.assert_frame_equal(pd.concat([first[0],resumed[0]],ignore_index=True),ledger,check_exact=True)
        joined=pd.concat([first[1],resumed[1]],ignore_index=True)
        pd.testing.assert_frame_equal(joined,decisions[joined.columns],check_exact=True)
        require(resumed[-1]==final,'实际恢复的最终内部状态不同')
        write_json(OUT/'actual_resume_verification.json',{'verified_at':now(),'status':'PASS_ACTUAL_HELD_CYCLE_JSON_RESUME_EXACT',
            'source':key,'cost':cost,'split_close':str(split_date.date()),'split_shares':unpack(first[-1]['state'])['account']['shares'],
            'joined_ledger_rows':len(ledger),'joined_decision_rows':len(joined),'account_and_controller_state_exact':True,
            'validation_replays':2,'new_candidates':0,'full_pipeline_incremental_resume_verified':False},exclusive=True)


def run():
    cfg=read(CONFIG)
    require(not (OUT/'RUN_STARTED.json').exists(),'完整续算已开始，请读取保存进展并定位故障，不盲目重跑')
    for item in cfg['frozen_files']:require(digest(ROOT/item['path'])==item['sha256'],'冻结输入或程序改变')
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG)},exclusive=True)
    started=time.perf_counter()
    pipeline=Pipeline(cfg).run()
    pipeline.verify_actual_resume()
    pd.DataFrame(pipeline.checks).to_csv(OUT/'old_account_prefix_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pipeline.target_checks).to_csv(OUT/'old_target_prefix_comparison.csv',index=False,encoding='utf-8-sig')
    metrics=[];extension=[];positions=[]
    for (key,cost),(ledger,decisions,state) in pipeline.accounts.items():
        last=ledger.iloc[-1];pending=decisions.iloc[-1]
        positions.append({'model':key,'cost':cost,'date':str(last.date.date()),'cash':last.cash,'shares':int(last.shares),
            'equity':last.equity,'dividend_receivable':last.dividend_receivable,'next_execution_date':str(pending.execution_date.date()),
            'next_request':int(pending.requested_quantity),'next_target':pending.reference_weight})
        if key==PRIMARY:
            metrics.append({'model':key,'cost':cost,'name':'固定85%与15%方案连续至8月31日',**summarize(ledger,cfg),
                'mark_scope':'CONTINUOUS_CLOSE_NO_TERMINAL_LIQUIDATION','ending_shares':int(last.shares)})
            start=ledger[ledger.date.eq('2026-08-13')].iloc[0]
            gap=ledger[ledger.date.ge('2026-08-14')]
            extension.append({'model':key,'cost':cost,'start_close':'2026-08-13','end_close':CUTOFF,'completed_days':len(gap),
                'start_equity':start.equity,'end_equity':last.equity,'net_profit_cny':last.equity-start.equity,
                'net_return':last.equity/start.equity-1,'trades':int(gap.filled_quantity.ne(0).sum()),
                'commission':float(gap.commission.sum()),'slippage_cost':float(gap.slippage_cost.sum()),'ending_shares':int(last.shares),
                'strict_forward':False,'annualized_short_period_metrics_computed':False})
    pd.DataFrame(metrics).to_csv(OUT/'continuous_account_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(extension).to_csv(OUT/'extension_segment_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(positions).to_csv(OUT/'ending_positions_and_next_requests.csv',index=False,encoding='utf-8-sig')
    original=read('reports/research/510300_incremental_selected_intent_mix_v1/result.json')
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'COMPLETED_OLD_PREFIX_EXACT_CONTINUOUS_ACCOUNTS_TO_AUGUST31',
        'candidate_configurations':0,'candidate_models':[PRIMARY],'evaluation_accounts':2,'new_accounts_generated':22,
        'reused_control_accounts':0,'earlier_diagnostic_accounts':0,'new_earlier_diagnostic_accounts':0,'new_model_fits':0,'new_reference_accounts':0,
        'existing_source_account_replays':20,'existing_continuous_reference_replays_included':2,'resume_validation_replays':2,
        'all_metrics':metrics,'earlier_diagnostics':[r for r in original['earlier_diagnostics'] if r['model']==PRIMARY],
        'earlier_metrics_reused_from_round209':True,'post_selected_best_base':metrics[0], 'primary':PRIMARY,
        'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED','position_impact':0,'strict_forward_evidence_days':0,
        'old_normal_account_prefixes_verified':len(pipeline.checks),'old_target_prefixes_verified':len(pipeline.target_checks),
        'extension_segment_results':extension,'continuous_cutoff':CUTOFF,'terminal_liquidation':False,
        'monthly_model_updates_due_in_extension':0,'old_model_last_fit_origin':'2026-08-03',
        'new_strategy_performance_computed':True,'full_pipeline_incremental_resume_verified':False,
        'run_seconds':time.perf_counter()-started,'source_cost_cny':0,'network_requests':0}
    write_json(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'status':result['status'],'seconds':result['run_seconds'],'extension':extension,'metrics':metrics},ensure_ascii=False),flush=True)


def verify():
    cfg=read(CONFIG);result=read(OUT/'result.json')
    inventory=pd.read_csv(INPUTS/'required_account_state_inventory.csv')
    data=pd.read_parquet(ROOT/cfg['features']);data=data[data.date.le(CUTOFF)]
    ledger_rows=decision_rows=0
    for item in inventory.to_dict('records'):
        folder=OUT/'accounts'/item['cost']/item['node']
        ledger=pd.read_parquet(folder/'ledger.parquet');decisions=pd.read_parquet(folder/'decisions.parquet')
        state=unpack(read(folder/'checkpoint.json')['state'])
        require(ledger.mark_clock.eq('CLOSE').all() and str(ledger.date.iloc[-1].date())==CUTOFF,'保存账本结算不是连续收盘')
        require(int(ledger.shares.iloc[-1])==state['account']['shares'] and abs(ledger.cash.iloc[-1]-state['account']['cash'])<1e-9,'保存状态与末日账本不同')
        require(len(decisions)==len(ledger)+1 and decisions.execution_date.iloc[-1]==pd.Timestamp('2026-09-01'),'保存连续决定缺少最后真实收盘')
        require(ledger.accounting_error.abs().max()<1e-6,'保存财富恒等式不成立')
        original=pd.read_parquet(ROOT/item['ledger']);old=original.iloc[:-1];matched=ledger.iloc[:len(old)]
        for c in ['cash','shares','dividend_receivable','equity','net_return','requested_quantity','filled_quantity']:
            np.testing.assert_allclose(matched[c],old[c],atol=1e-6,rtol=0,equal_nan=True)
        if item['node']==PRIMARY:
            stored=next(r for r in result['all_metrics'] if r['cost']==item['cost'])
            measured=summarize(ledger,cfg)
            for k in ['net_sharpe','annualized_return','max_drawdown','trade_count']:
                require(abs(measured[k]-stored[k])<1e-12,'保存主账户绩效不同')
        ledger_rows+=len(ledger);decision_rows+=len(decisions)
    require(read(OUT/'actual_resume_verification.json')['status']=='PASS_ACTUAL_HELD_CYCLE_JSON_RESUME_EXACT','实际持仓恢复未通过')
    for item in cfg['frozen_files']:require(digest(ROOT/item['path'])==item['sha256'],'冻结程序或输入发生改变')
    write_json(OUT/'saved_verification_receipt.json',{'verified_at':now(),'status':'PASS_22_CONTINUOUS_ACCOUNTS_OLD_PREFIX_AND_ACTUAL_CYCLE_RESUME',
        'accounts':22,'ledger_rows':ledger_rows,'decision_rows':decision_rows,'new_model_fits':0,
        'terminal_liquidation':False,'old_normal_prefix_pass':True,'actual_cycle_resume_exact':True,
        'full_pipeline_incremental_resume_verified':False,'independent_performance_validation':False},exclusive=True)
    print('22条连续账户保存核对完成，原正常区间及实际持仓恢复一致。',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='固定来源一次批量连续续算。')
    parser.add_argument('action',choices=['prepare','run','verify'])
    globals()[parser.parse_args().action]()
