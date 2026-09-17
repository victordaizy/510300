import json
from pathlib import Path
from research.drawdown_gate_recovery_inputs_v1 import PRIMARY,MODELS,CANDIDATES,drawdown_gate_recovery_frames
from research.saved_target_batch_runner_v1 import run_saved_target_batch
from research.intraday_overnight_increment_v1 import now,digest,require,write_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/research/510300_drawdown_gate_recovery_v1';CONFIG=ROOT/'config/510300_drawdown_gate_recovery_v1.json'
P143=ROOT/'reports/research/510300_trend_noise_reference_blend_v1';P165=ROOT/'reports/research/510300_return_runs_state_v1';P176=ROOT/'reports/research/510300_core_auxiliary_drawdown_gate_v1';P174=ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1';P168=ROOT/'reports/research/510300_runs_opportunity_union_v1';P32=ROOT/'reports/research/510300_rearmed_session_exit_v1'
PARENTS={MODELS[0]:P143,MODELS[1]:P165};CONTROLS={'CORE_AUXILIARY_DRAWDOWN_GATE':(P176,'第176轮即时恢复'),'EITHER_CONFIRMED_RUNS_AUXILIARY':(P174,'第174轮任一方向确认'),'RUNS_OPPORTUNITY_CAPPED_SUM':(P168,'第168轮相加封顶'),MODELS[0]:(P143,'第143轮核心'),MODELS[1]:(P165,'第165轮辅助'),'BUY_HOLD':(P32,'买入持有')}
def freeze():
 require(not CONFIG.exists(),'第177轮已冻结');prior=json.loads((ROOT/'config/510300_core_auxiliary_drawdown_gate_v1.json').read_text(encoding='utf-8'));tests=json.loads((OUT/'tests_receipt.json').read_text(encoding='utf-8'));require(tests['exit_code']==0 and tests['passed']==2,'必要测试未通过');keys=['evaluation_start','data_cutoff','initial_capital','lot','tick','limit_fraction','annual_days','cash_annual_rate_assumption','high_sharpe_target','costs','features','dividends','earlier_start','earlier_terminal','weight_band'];cfg={k:prior[k] for k in keys};cfg.update(study_id='510300_DRAWDOWN_GATE_RECOVERY_V1',round=177,primary=PRIMARY,candidate_configurations=1,candidate_models=list(CANDIDATES),registered_at=now(),decision_clock='15:05:00',parent_models=MODELS,drawdown_window=60,drawdown_gate=-.05,recovery_closes=2,new_model_fits=0,new_reference_accounts=0,source_budget_cny=0,goal_achieved=False,position_impact=0,rules='docs/510300_DRAWDOWN_GATE_RECOVERY_NEXT_20260912.md',independent_validation='NOT_ESTABLISHED')
 paths=[Path(__file__),ROOT/'research/drawdown_gate_recovery_inputs_v1.py',ROOT/'research/saved_target_batch_runner_v1.py',ROOT/'research/saved_parent_target_alignment_v1.py',ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',ROOT/'tests/test_drawdown_gate_recovery_v1.py',OUT/'tests_receipt.json',ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/cfg['rules'],ROOT/'config/510300_research_authority_v6.json']
 for m,f in PARENTS.items():paths += [f/per/cost/f'{m}_decisions.parquet' for per in ['evaluation','earlier_diagnostic'] for cost in cfg['costs']]
 for m,(f,_) in CONTROLS.items():paths += [f/per/cost/f'{m}_ledger.parquet' for per in ['evaluation','earlier_diagnostic'] for cost in cfg['costs']]
 cfg['frozen_files']=[{'path':str(x.relative_to(ROOT)),'sha256':digest(x)} for x in sorted(set(paths))];write_json(CONFIG,cfg,exclusive=True);print('第177轮已冻结')
def run():return run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,drawdown_gate_recovery_frames)
if __name__=='__main__':
 import sys;{'freeze':freeze,'run':run}[sys.argv[1]]()
