"""第195轮一次生成两条缺失参考并评价八条预先列明的组合账户。"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends,save_account,summarize
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.mean_rebound_auxiliary_inputs_v1 import CORE,AUX,MODELS,PRIMARY,WEIGHTS,CANDIDATES,rebound_frames
from research.saved_target_referenced_batch_v1 import run_referenced_target_batch
from research.simple_price_entry_exit_v1 import signals,simulate_policy

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_mean_rebound_auxiliary_v1'
CONFIG=ROOT/'config/510300_mean_rebound_auxiliary_v1.json'
CORE_SOURCE=ROOT/'reports/research/510300_account_volatility_exposure_v1'
AUX_SOURCE=ROOT/'reports/research/510300_simple_price_entry_exit_v1'
REFERENCES=OUT/'reference_sources'
PARENTS={CORE:CORE_SOURCE,AUX:REFERENCES}
CONTROLS={CORE:(CORE_SOURCE,'第181轮原账户风险预算'),AUX:(REFERENCES,'原均值反弹参考'),
    'BUY_HOLD':(ROOT/'reports/research/510300_rearmed_session_exit_v1','买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'REFERENCE_RUN_STARTED.json').exists(),'本轮已冻结或开始参考计算')
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'tests_receipt.json').exists(),'本轮测试已有回执')
    began=time.perf_counter()
    tested=subprocess.run([sys.executable,'-m','pytest','tests/test_mean_rebound_auxiliary_v1.py','-q','-p','no:cacheprovider'],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr,encoding='utf-8')
    print(tested.stdout,flush=True)
    require(tested.returncode==0 and '6 passed' in tested.stdout,'本轮六项必要测试未通过')
    write_json(OUT/'tests_receipt.json',{'recorded_at':now(),'exit_code':tested.returncode,'passed':6,
        'seconds':time.perf_counter()-began,'timing_scope':'完整测试进程墙钟'},exclusive=True)
    core_config=ROOT/'config/510300_account_volatility_exposure_v1.json'
    aux_config=ROOT/'config/510300_simple_price_entry_exit_v1.json'
    old=json.loads(core_config.read_text(encoding='utf-8'))
    old_aux=json.loads(aux_config.read_text(encoding='utf-8'))
    keys=['evaluation_start','data_cutoff','initial_capital','lot','tick','limit_fraction','annual_days',
        'cash_annual_rate_assumption','high_sharpe_target','costs','features','dividends','earlier_start','earlier_terminal','weight_band']
    cfg={k:old[k] for k in keys}
    cfg.update(study_id='510300_MEAN_REBOUND_AUXILIARY_V1',round=195,registered_at=now(),primary=PRIMARY,
        candidate_models=list(CANDIDATES),candidate_configurations=2,parent_models=MODELS,auxiliary_weights=WEIGHTS.copy(),
        auxiliary_specification=old_aux['candidate_specs'][AUX],decision_clock='15:05:00',annual_return_target=.10,
        planned_new_reference_accounts=2,new_model_fits=0,new_reference_accounts=2,source_budget_cny=0,
        goal_achieved=False,position_impact=0,independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_FINITE_DIVERSIFICATION_COMBINATION',
        rules='docs/510300_MEAN_REBOUND_AUXILIARY_V1.md',
        source_preflight='reports/research/510300_saved_rebound_diversification_preflight_20260913/result.json',
        previous_goal_turn_classification='PROGRESS_ROUND194_EIGHT_ACCOUNTS_VERIFIED_DELIVERED')
    require(old_aux['costs']==cfg['costs'] and old_aux['initial_capital']==cfg['initial_capital'],'原辅助费用或本金不同')
    with (ROOT/cfg['rules']).open('x',encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_MEAN_REBOUND_AUXILIARY_NEXT_20260913.md').read_text(encoding='utf-8').replace(
            '尚未登记、实现、测试或计算新组合。','本方案在必要测试通过后、首次参考及组合计算前冻结。'))
        stream.write('\n\n## 原181来源的全部中文因素及规则\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths=[Path(__file__),ROOT/'research/mean_rebound_auxiliary_inputs_v1.py',ROOT/'research/saved_target_referenced_batch_v1.py',
        ROOT/'research/saved_parent_target_alignment_v1.py',ROOT/'research/event_clock_account_v1.py',
        ROOT/'research/saved_target_account_checks_v1.py',ROOT/'research/simple_price_entry_exit_v1.py',
        ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'research/joint_account_acceptance_v1.py',ROOT/'tests/test_mean_rebound_auxiliary_v1.py',
        core_config,aux_config,ROOT/old_aux['rules'],ROOT/old['rules'],OUT/'tests_receipt.json',OUT/'tests_output.txt',
        ROOT/'config/510300_research_authority_v6.json',CORE_SOURCE/'saved_verification_receipt.json']
    paths.extend(ROOT/cfg[k] for k in ['rules','features','dividends','source_preflight'])
    for period in ['evaluation','earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [CORE_SOURCE/period/cost/f'{CORE}_decisions.parquet',CORE_SOURCE/period/cost/f'{CORE}_ledger.parquet',
                CONTROLS['BUY_HOLD'][0]/period/cost/'BUY_HOLD_ledger.parquet']
            if period=='evaluation':
                paths += [AUX_SOURCE/period/cost/f'{AUX}_{suffix}' for suffix in ['ledger.parquet','decisions.parquet','cycles.csv']]
            else:
                require(not (AUX_SOURCE/period/cost/f'{AUX}_ledger.parquet').exists(),'较早辅助已存在，应先核实来源变化')
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round']==194,'前序完成状态不同')
    index['running_studies']=[{'round':195,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True,status='MEAN_REBOUND_AUXILIARY_FROZEN',source=cfg['rules'])
    write_json(path,index)
    print('第195轮两套组合及两条新增较早参考已冻结。',flush=True)


def create_references(cfg):
    require(not (OUT/'REFERENCE_RUN_STARTED.json').exists(),'参考计算已开始，请接续保存状态')
    write_json(OUT/'REFERENCE_RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG)},exclusive=True)
    began=time.perf_counter()
    data=pd.read_parquet(ROOT/cfg['features'])
    frame=data[data.date.le(cfg['earlier_terminal'])].copy()
    dividends=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    rule=signals(frame)[AUX]
    metrics,files=[],[]
    for cost_id,cost in cfg['costs'].items():
        main=REFERENCES/'evaluation'/cost_id
        main.mkdir(parents=True)
        for suffix in ['ledger.parquet','decisions.parquet','cycles.csv']:
            original=AUX_SOURCE/'evaluation'/cost_id/f'{AUX}_{suffix}'
            dest=main/original.name
            shutil.copy2(original,dest)
            require(digest(dest)==digest(original),'主历史旧辅助复制改变内容')
            files.append(dest)
        ledger,decisions,cycles=simulate_policy(frame,dividends,cfg,cost,cfg['earlier_start'],rule,cfg['auxiliary_specification'])
        folder=REFERENCES/'earlier_diagnostic'/cost_id
        save_account(folder,AUX,ledger,decisions)
        cycles.to_csv(folder/f'{AUX}_cycles.csv',index=False,encoding='utf-8-sig')
        files += [folder/f'{AUX}_{suffix}' for suffix in ['ledger.parquet','decisions.parquet','cycles.csv']]
        require(ledger.accounting_error.abs().max()<1e-6 and not ledger.terminal_unliquidated.iloc[-1],'新增辅助未完整结算')
        metrics.append({'period':'earlier_diagnostic','cost':cost_id,'model':AUX,**summarize(ledger,cfg)})
        print(f'{cost_id}：一条缺失较早辅助已生成，主历史来源直接复用。',flush=True)
    receipt={'completed_at':now(),'new_reference_accounts':2,'reused_main_reference_accounts':2,'new_model_fits':0,
        'run_seconds':time.perf_counter()-began,'metrics':metrics,'config_sha256':digest(CONFIG),
        'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in files]}
    write_json(OUT/'reference_receipt.json',receipt,exclusive=True)
    return receipt


def run():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path'])==item['sha256'],'冻结来源改变')
    require(not (OUT/'RUN_STARTED.json').exists(),'本轮组合已经开始')
    receipt_path=OUT/'reference_receipt.json'
    receipt=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else create_references(cfg)
    require(receipt['config_sha256']==digest(CONFIG),'参考账户登记版本不同')
    result=run_referenced_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,rebound_frames,receipt)
    print(json.dumps(save_joint_assessment(OUT,cfg,result),ensure_ascii=False),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'run':run}[sys.argv[1]]()
