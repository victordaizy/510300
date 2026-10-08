"""本次研究的时点、规模、账务、原输入与有限确定性复算检查。"""
from __future__ import annotations
import sys,json,unittest,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research import annual10_capital_20260910 as cap
from tools.research import annual10_residual_20260910 as res
from tools.research import annual10_episode_value_20260910 as ep
from tools.research import annual10_daily_opportunity_20260910 as daily
from tools.research import annual10_etf_flow_20260910 as flow
from tools.research import annual10_winning_budget_20260910 as win
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import Account,execute_order,write_json,now,digest
from research.simple_price_entry_exit_v1 import simulate_policy
OUT=ROOT/'research_runs/annual10_sharpe12_20260910'

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg,cls.data,cls.div=daily.setup()
    def test_capital_zero_missing_and_bounds(self):
        p=np.array([np.nan,0.,.1,.4,.9,1.]);data=pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=6)})
        for name in cap.POLICIES:
            v=cap.capital_target(data,p,np.zeros(6),name)
            self.assertTrue(np.isnan(v[0]));self.assertEqual(v[1],0.)
            self.assertTrue(((v[np.isfinite(v)]>=0)&(v[np.isfinite(v)]<=1)).all())
    def test_capital_risk_future_invariance(self):
        n=650;d=pd.DataFrame({'date':pd.bdate_range('2015-01-01',periods=n)});rng=np.random.default_rng(24);p=rng.uniform(0,1,n);r=rng.normal(0,.005,n)
        a=cap.capital_target(d,p,r,'RISK_CONTRACT');r[401:]*=30;p[401:]=1
        b=cap.capital_target(d,p,r,'RISK_CONTRACT');np.testing.assert_equal(a[:401],b[:401])
    def test_alignment_rejects_same_day(self):
        d=self.data.iloc[:4];q=pd.DataFrame({'origin':d.date.iloc[:2].to_numpy(),'execution_date':d.date.iloc[:2].to_numpy(),'reference_weight':[0.,1.]})
        with self.assertRaises(ValueError):cap.aligned(d,q)
    def test_alignment_unknown_remains_unknown(self):
        d=self.data.iloc[:4];q=pd.DataFrame({'origin':[d.date.iloc[1]],'execution_date':[d.date.iloc[2]],'reference_weight':[1.]})
        a=cap.aligned(d,q);self.assertTrue(np.isnan(a[0]));self.assertEqual(a[1],1.)
    def test_residual_arithmetic(self):
        c=np.array([0.,.5,1.,np.nan]);s=np.array([1.,1.,1.,1.])
        np.testing.assert_equal(res.residual_target(c,s,'REMAINING',.5),[.5,.75,1.,np.nan])
        np.testing.assert_equal(res.residual_target(c,s,'IDLE_ONLY',1.),[1.,.5,1.,np.nan])
    def test_calendar_prefix(self):
        a=res.calendar_targets(self.data);b=res.calendar_targets(self.data.iloc[:2400])
        for k in a:np.testing.assert_equal(a[k][:2399],b[k][:2399])
    def test_episode_target_future_invariance(self):
        n=130;data=pd.DataFrame({'vol20':np.repeat(.2,n)});core=np.zeros(n);s=np.ones((n,len(ep.SOURCES)));q=np.zeros_like(s,dtype=bool);q[10::20,0]=True;s[25::20,0]=0
        p=np.zeros_like(s);p[:,0]=.02;days=np.full_like(s,20)
        a=ep.learned_targets(data,core,s,q,days,p,0.,'VOL10','REMAINING',1)[0]
        p[81:]=-1.;q[81:]=True;s[81:]=0
        b=ep.learned_targets(data,core,s,q,days,p,0.,'VOL10','REMAINING',1)[0]
        np.testing.assert_equal(a[:81],b[:81])
    def test_concurrency_weights(self):
        d=pd.DataFrame({'entry_index':[1,2],'exit_index':[3,4]})
        np.testing.assert_allclose(daily.concurrency_weights(d,8),[2/3,2/3],atol=1e-15)
    def test_fixed_size_no_future_volatility(self):
        a=daily.scaled_sleeve(np.array([0.,1.,1.,np.nan,1.,0.,1.]),np.array([.2,.2,.5,.8,.1,.4,.4]),'VOL10')
        np.testing.assert_equal(a,[0.,.5,.5,np.nan,.5,0.,.25])
    def test_daily_label_reproduction(self):
        labels=cap.read_frame(daily.OUT/'labels.parquet');labels=labels[labels.status.eq('MATURE')]
        for idx in [0,500,1000,2000,3000]:
            row=labels.iloc[idx];t=int(row.origin_index);f=self.data.iloc[t:t+23].copy().reset_index(drop=True);e=np.zeros(len(f),int);e[0]=1
            led,dec,cy=simulate_policy(f,self.div,self.cfg,self.cfg['costs']['BASE'],str(f.date.iloc[1].date()),{'entry':e,'exit':{1:np.zeros(len(f),bool)}},daily.SPEC)
            self.assertAlmostEqual(float(cy.iloc[0].net_profit_cny/cy.iloc[0].entry_cost_cny),float(row.roi),places=13)
    def test_model_maturity_records(self):
        for phase,name,lag in [('episode_value','training_records.json',5),('daily_opportunity','training_records.json',5),('etf_flow','training_H5.json',2),('etf_flow','training_H20.json',2)]:
            records=cap.load_json(OUT/phase/name)['records']
            for r in records:
                if r['status']=='FIT_COMPLETE':self.assertLessEqual(r['latest_exit_index'],r['fit_index']-lag)
    def test_flow_sources_delayed_two_days(self):
        sources={k:flow.read_source(ROOT/v) for k,v in flow.PATHS.items()};before,_=flow.flow_features(self.data,sources);cut=2800;when=self.data.date.iloc[cut]
        for f in sources.values():
            for c in f.select_dtypes(include='number').columns:f.loc[f.date.ge(when),c]*=7
        after,_=flow.flow_features(self.data,sources);pd.testing.assert_frame_equal(before.iloc[:cut+2],after.iloc[:cut+2])
    def test_flow_unknown_does_not_add_position(self):
        p=np.array([np.nan,-1.,1.]);c=np.array([.4,.4,.4]);cost={'commission':.0002,'slippage':.0005}
        np.testing.assert_equal(flow.policy_target(p,c,cost,0.,'REMAINING_CORE150X2'),[.4,.4,1.])
        self.assertTrue(np.isnan(flow.policy_target(p,c,cost,0.,'STANDALONE')[0]))
    def test_flow_no_prediction_before_source(self):
        f=cap.read_frame(flow.OUT/'features.parquet');valid=np.isfinite(f.drop(columns='date')).all(axis=1).to_numpy()
        for h in flow.HORIZONS:
            p=np.load(flow.OUT/f'predictions_H{h}.npz')['predictions'];self.assertTrue(np.isnan(p[~valid]).all())
    def test_t_plus_one_and_cash(self):
        a=Account(200000.);cost=self.cfg['costs']['BASE'];b=execute_order(a,100,4.,4.,0.,0,cost,self.cfg);self.assertEqual(b['filled_quantity'],100)
        s=execute_order(a,-100,4.,4.,0.,0,cost,self.cfg);self.assertEqual(s['filled_quantity'],0)
        s=execute_order(a,-100,4.,4.,0.,1,cost,self.cfg);self.assertEqual(s['filled_quantity'],-100);self.assertGreaterEqual(a.cash,0)
    def test_directional_limit(self):
        a=Account(200000.);x=execute_order(a,100,4.4,4.,0.,0,self.cfg['costs']['BASE'],self.cfg);self.assertEqual(x['filled_quantity'],0)
    def test_all_saved_input_hashes(self):
        for phase in ['capital','residual','episode_value','daily_opportunity','etf_flow','winning_budget']:
            pro=cap.load_json(OUT/phase/'protocol.json')
            for r in pro['files']:self.assertEqual(digest(ROOT/r['path']),r['sha256'])
    def test_winning_current_and_latch(self):
        p=np.array([.2,.2,0.,.2,.2,np.nan]);y=np.array([-.01,.02,.5,.02,-.02,100.]);v=np.repeat(.01,6)
        np.testing.assert_equal(win.winning_target(p,y,v,2,'ZERO','CURRENT'),[.4,1.,0.,1.,.4,np.nan])
        np.testing.assert_equal(win.winning_target(p,y,v,2,'ZERO','LATCHED'),[.4,1.,0.,1.,1.,np.nan])
    def test_winning_future_invariance(self):
        p=np.repeat(.2,100);y=np.linspace(-.03,.05,100);v=np.repeat(.01,100)
        a=win.winning_target(p,y,v,2,'ENTRY_SIGMA','LATCHED');y[75:]=-100.;p[75:]=0.
        b=win.winning_target(p,y,v,2,'ENTRY_SIGMA','LATCHED');np.testing.assert_equal(a[:75],b[:75])
    def test_winning_reference_before_entry_unknown(self):
        led=cap.read_frame(OUT/'capital/evaluation/BASE/R150__FULL_SUPPORT_ledger.parquet')
        profit,sigma=win.episode_state(self.data,led)
        first=led.loc[led.shares.gt(0),'date'].iloc[0];idx=int(pd.DatetimeIndex(self.data.date).get_loc(first))
        self.assertTrue(np.isnan(profit[:idx]).all());self.assertTrue(np.isfinite(profit[idx]));self.assertTrue(np.isfinite(sigma[idx]))
    def test_reproduce_three_residual_accounts(self):
        f=self.data;first=self.cfg['evaluation_start'];src=cap.read_frame(res.OUT/'evaluation/BASE/source_targets.parquet')
        for key,k,a in [('V1_CLIMAX_RECOVERY',2,1.),('V1_CLIMAX_RECOVERY',2,.5),('CAL_MONTH_EDGE',3,1.)]:
            model=f'CORE{k}__{key}__IDLE_ONLY__A{int(a*100)}';target=res.residual_target(np.minimum(1.,k*src.parent150.to_numpy()),src[key].to_numpy(),'IDLE_ONLY',a)
            led,_=simulate_event_account(f,self.div,self.cfg,self.cfg['costs']['BASE'],first,model,targets=target,event_mask=np.ones(len(f),bool))
            old=cap.read_frame(res.OUT/'evaluation/BASE'/f'{model}_ledger.parquet')
            for c in ['shares','cash','equity','net_return','commission','slippage_cost']:np.testing.assert_allclose(led[c],old[c],atol=1e-9,rtol=0)

def verify_phase(phase):
    cfg,data,div=daily.setup();folder=OUT/phase;rows=[]
    for path in sorted(folder.rglob('*_ledger.parquet')):
        led=cap.read_frame(path);checks=cap.accounting(led,cfg)
        prev=np.r_[0,led.shares.to_numpy()[:-1]]
        np.testing.assert_array_equal(led.shares,prev+led.filled_quantity)
        # 同一开盘的卖出不超过前一日结转的份额。
        assert ((-led.filled_quantity.clip(upper=0))<=prev).all()
        rows.append({'path':str(path.relative_to(ROOT)),**checks,'sha256':digest(path),
                     'nav_path_sha256':hashlib.sha256(led[['equity','net_return','shares']].to_numpy(dtype=np.float64).tobytes()).hexdigest()})
    pd.DataFrame(rows).to_csv(folder/'independent_account_checks.csv',index=False)
    print(phase,len(rows),'独立账务检查通过',flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='accounts':verify_phase(sys.argv[2])
    else:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResearchTests))
        write_json(OUT/'new_tests_receipt.json',{'completed_at':now(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'code_sha256':digest(Path(__file__))})
        raise SystemExit(0 if result.wasSuccessful() else 1)
