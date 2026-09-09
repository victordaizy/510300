"""在保留原方法文件的条件下，明确绑定修正后的EPS事实与独立输出目录。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity
from research import forward_eps_monthly_policy_v1 as monthly
from research import forward_eps_residual_policy_v1 as residual
from research import forward_eps_explicit_entry_exit_v1 as entry_exit
from research.explicit_entry_exit_account_v1 import simulate_entry_exit_account

MANIFEST=ROOT/'config/510300_forward_eps_corrected_source_pipeline_v2_manifest.json'
CONFIGS={'monthly':'510300_forward_eps_monthly_policy','residual':'510300_forward_eps_residual_policy','entry_exit':'510300_forward_eps_explicit_entry_exit'}


def empty_cycle_schema(*args,**kwargs):
    ledger,decisions,cycles=simulate_entry_exit_account(*args,**kwargs)
    if cycles.empty:
        cycles=pd.DataFrame(columns=['cycle_id','entry_date','exit_date','cycle_net_profit_cny','cycle_net_return','exit_reasons'])
    return ledger,decisions,cycles


def corrected_paths(scope):
    if scope!='csi':raise ValueError('第二版接续只包含已登记沪深300主范围')
    return (ROOT/'reports/research/510300_forward_eps_csi_facts_v2',
            ROOT/'reports/research/510300_forward_eps_csi_originals_v1',
            ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi')


def freeze():
    if MANIFEST.exists():raise FileExistsError('修正来源接续已登记')
    paths=[Path(__file__),ROOT/'docs/510300_FORWARD_EPS_CORRECTED_SOURCE_EXECUTION_V2.md',
           ROOT/'research/forward_eps_monthly_policy_v1.py',ROOT/'research/forward_eps_residual_policy_v1.py',
           ROOT/'research/forward_eps_explicit_entry_exit_v1.py',ROOT/'research/explicit_entry_exit_account_v1.py',
           ROOT/'config/510300_forward_eps_monthly_policy_v1_manifest.json',ROOT/'config/510300_forward_eps_residual_policy_v1_manifest.json',
           ROOT/'config/510300_forward_eps_explicit_entry_exit_v1_manifest.json',ROOT/'config/510300_forward_eps_csi_facts_v2_manifest.json',
           ROOT/'research/event_clock_account_v1.py',ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
           ROOT/'research/financial_annual_components_v1.py',ROOT/'research/forward_eps_guosen_history_v1.py',
           ROOT/'research/forward_eps_guosen_layout_v2.py',ROOT/'research/forward_eps_guosen_history_v2.py',
           ROOT/'tests/test_forward_eps_corrected_source_pipeline_v2.py',
           ROOT/'config/510300_research_authority_v6.json',ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet',
           ROOT/'reports/research/510300_forward_eps_csi_directory_v1/historical_membership.parquet']
    for name in CONFIGS.values():
        old=ROOT/'config'/(name+'_v1.json');config=read(old)
        config.update({'study_id':name.upper()+'_V2','version':'2.0.0','source_fact_version':'v2','source_layout_correction_only':True})
        new=ROOT/'config'/(name+'_v2.json');save(new,config,exclusive=True);paths.extend([old,new])
        assert {k:v for k,v in config.items() if k not in ['study_id','version','source_fact_version','source_layout_correction_only']}=={
            k:v for k,v in read(old).items() if k not in ['study_id','version']}
    first=read(ROOT/'config/510300_forward_eps_monthly_policy_v1.json')
    paths.extend(ROOT/first['inputs'][key] for key in ['dividends','calendar'])
    save(MANIFEST,{'registered_at':now(),'source_layout_repair_only':True,'method_thresholds_costs_and_windows_unchanged':True,
                   'old_csi_zero_view_accounts_observed':True,'old_residual_price_baseline_observed':True,
                   'new_candidate_parameter_combinations':0,'planned_corrected_source_accounts':22,
                   'all_three_method_source_bindings_registered_together':True,'files':[identity(path) for path in paths]},exclusive=True)
    print('三组既定方法的修正来源接续已登记，阈值及费用均未改变。',flush=True)


def run(stage):
    for item in read(MANIFEST)['files']:
        if identity(ROOT/item['path'])['sha256']!=item['sha256']:raise ValueError('来源接续冻结输入变化')
    if not (ROOT/'reports/research/510300_forward_eps_csi_facts_v2/result.json').exists():
        raise RuntimeError('原文识别修正仍在运行，承接原进程等待')
    if stage=='monthly_prepare' or stage=='monthly_run':
        monthly.CONFIG=ROOT/'config/510300_forward_eps_monthly_policy_v2.json';monthly.MANIFEST=MANIFEST;monthly.scope_paths=corrected_paths
        monthly.prepare('csi') if stage=='monthly_prepare' else monthly.run('csi')
    elif stage=='residual':
        residual.CONFIG=ROOT/'config/510300_forward_eps_residual_policy_v2.json';residual.MANIFEST=MANIFEST
        residual.OUT=ROOT/'reports/research/510300_forward_eps_residual_policy_v2'
        residual.SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi';residual.run()
    elif stage=='entry_exit':
        entry_exit.CONFIG=ROOT/'config/510300_forward_eps_explicit_entry_exit_v2.json';entry_exit.MANIFEST=MANIFEST
        entry_exit.OUT=ROOT/'reports/research/510300_forward_eps_explicit_entry_exit_v2'
        entry_exit.SOURCE=ROOT/'reports/research/510300_forward_eps_monthly_policy_v2_csi'
        entry_exit.simulate_entry_exit_account=empty_cycle_schema;entry_exit.run()


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true');group.add_argument('--stage',choices=['monthly_prepare','monthly_run','residual','entry_exit'])
    args=parser.parse_args();freeze() if args.freeze else run(args.stage)
