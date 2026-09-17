"""冻结两种辅助资格方案，读取保存来源并生成八个完整账户。"""
import json
from pathlib import Path
from research.confirmed_auxiliary_episode_batch_inputs_v1 import PRIMARY,MODELS,CANDIDATES,confirmed_auxiliary_episode_batch_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now,digest,require,write_json
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/research/510300_confirmed_auxiliary_episode_batch_v1'; CONFIG=ROOT/'config/510300_confirmed_auxiliary_episode_batch_v1.json'
P143=ROOT/'reports/research/510300_trend_noise_reference_blend_v1';P165=ROOT/'reports/research/510300_return_runs_state_v1';P171=ROOT/'reports/research/510300_return_lag_state_v1';P172=ROOT/'reports/research/510300_return_sign_balance_v1';P174=ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1';P173=ROOT/'reports/research/510300_sign_confirmed_runs_auxiliary_v1';P168=ROOT/'reports/research/510300_runs_opportunity_union_v1';P32=ROOT/'reports/research/510300_rearmed_session_exit_v1'
PARENTS={MODELS[0]:P143,MODELS[1]:P165,MODELS[2]:P171,MODELS[3]:P172}
CONTROLS={'EITHER_CONFIRMED_RUNS_AUXILIARY':(P174,'第174轮任一方向确认'),'SIGN_CONFIRMED_RUNS_AUXILIARY':(P173,'第173轮上涨日确认'),'RUNS_OPPORTUNITY_CAPPED_SUM':(P168,'第168轮相加封顶'),MODELS[0]:(P143,'第143轮核心'),MODELS[1]:(P165,'第165轮辅助'),'BUY_HOLD':(P32,'买入持有')}
def freeze():
 require(not CONFIG.exists(),'第175轮已冻结'); prior=json.loads((ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json').read_text(encoding='utf-8')); tests=json.loads((OUT/'tests_receipt.json').read_text(encoding='utf-8')); require(tests['exit_code']==0 and tests['passed']==5,'必要测试未通过')
 keys=['evaluation_start','data_cutoff','initial_capital','lot','tick','limit_fraction','annual_days','cash_annual_rate_assumption','high_sharpe_target','costs','features','dividends','earlier_start','earlier_terminal','weight_band']; cfg={k:prior[k] for k in keys}; cfg.update(study_id='510300_CONFIRMED_AUXILIARY_EPISODE_BATCH_V1',round=175,primary=PRIMARY,candidate_configurations=2,candidate_models=list(CANDIDATES),registered_at=now(),decision_clock='15:05:00',parent_models=MODELS,combination='FIXED_EPISODE_START_AND_WAIT_DIRECTION_QUALIFICATIONS',new_model_fits=0,new_reference_accounts=0,source_budget_cny=0,goal_achieved=False,position_impact=0,rules='docs/510300_CONFIRMED_AUXILIARY_EPISODE_BATCH_V1.md',independent_validation='NOT_ESTABLISHED',outer_exit_retry='RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE',reentry='ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT')
 paths=[Path(__file__),ROOT/'research/confirmed_auxiliary_episode_batch_inputs_v1.py',ROOT/'research/saved_target_batch_runner_v1.py',ROOT/'research/saved_parent_target_alignment_v1.py',ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',ROOT/'tests/test_confirmed_auxiliary_episode_batch_v1.py',OUT/'tests_receipt.json',ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/cfg['rules'],ROOT/'docs/510300_CONFIRMED_AUXILIARY_EPISODE_BATCH_NEXT_20260911.md',ROOT/'config/510300_research_authority_v6.json']
 for model,folder in PARENTS.items():
  pc=ROOT/'config'/f'{folder.name}.json'; paths.extend([pc,ROOT/json.loads(pc.read_text(encoding='utf-8'))['rules'],folder/'saved_verification_receipt.json'])
  for period in ['evaluation','earlier_diagnostic']:
   for cost in cfg['costs']: paths.append(folder/period/cost/f'{model}_decisions.parquet')
 for period in ['evaluation','earlier_diagnostic']:
  for cost in cfg['costs']:
   for model,(folder,_) in CONTROLS.items(): paths.append(folder/period/cost/f'{model}_ledger.parquet')
 cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]; write_json(CONFIG,cfg,exclusive=True); print('第175轮已冻结。')
def run(): return run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,confirmed_auxiliary_episode_batch_frames)
if __name__=='__main__':
 import sys; {'freeze':freeze,'run':run}[sys.argv[1]]()
