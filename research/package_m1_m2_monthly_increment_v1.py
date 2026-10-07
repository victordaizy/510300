"""将本轮冻结结果与直接依赖打包，执行CRC、索引、哈希和解压离线核对。"""
from __future__ import annotations

import ast
import csv
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from research.m1_m2_release_sources_v1 import ROOT, OUT, now, sha, write_json
from research.m1_m2_monthly_increment_v1 import read, check_protocol


def import_closure(paths):
    """包含本轮Python模块的本地导入闭包，不执行任何研究入口。"""
    paths=set(paths)
    queue=[p for p in paths if p.suffix=='.py']
    visited=set()
    while queue:
        path=queue.pop()
        if path in visited:
            continue
        visited.add(path)
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module]
            elif isinstance(node,ast.Import):
                names=[n.name for n in node.names]
            for name in names:
                if name.split('.')[0] not in ['research','src','backtest']:
                    continue
                candidate=ROOT/Path(*name.split('.')).with_suffix('.py')
                if candidate.is_file() and candidate not in paths:
                    paths.add(candidate)
                    queue.append(candidate)
                directory=candidate.parent
                while directory!=ROOT and ROOT in directory.parents:
                    init=directory/'__init__.py'
                    if init.is_file():
                        paths.add(init)
                    directory=directory.parent
    return paths


def prepare_context():
    index_path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=read(index_path)
    assert index['status']=='TERMINATED_BY_USER_HIGH_OVERFITTING_RISK'
    rows=[]
    configurations=sorted(set((ROOT/'config').glob('510300_*.json')) | set((ROOT/'config').glob('510300_*.yaml')))
    for path in configurations:
        record={'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}
        if path.suffix=='.json':
            value=read(path)
            if isinstance(value,dict):
                for key in ['study_id','round','candidate_configurations','registered_at','primary','status']:
                    v=value.get(key)
                    if isinstance(v,(str,int,float,bool)) or v is None:
                        record[key]=v
        rows.append(record)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT/'registered_configuration_inventory.csv',index=False,encoding='utf-8-sig')
    trial={'snapshot_at':now(),'source':index_path.relative_to(ROOT).as_posix(),'source_sha256':sha(index_path),
           'old_completed_round_records':len(index['completed_rounds']),
           'registered_configuration_file_count':len(configurations),
           **{k:index[k] for k in ['registered_configurations_in_this_resumption','evaluated_configurations_in_this_resumption',
                'evaluated_candidate_source_runs_including_corrected_replays','evaluation_accounts_in_this_resumption','count_warning'] if k in index},
           'counts_are_not_independent_trials':True,'full_effective_trial_universe_known':False,
           'DSR':'NOT_COMPUTED_MISSING_EFFECTIVE_TRIAL_UNIVERSE',
           'this_round_new_macro_features':1,'this_round_sequential_fits':106,'this_round_new_accounts':0}
    write_json(OUT/'selection_history_snapshot.json',trial)
    status={**read(OUT/'prediction_result.json'),'risk_module_review':'COMPLETED_SAVED_OUTPUT_VERIFICATION',
            'risk_prediction_status':'PRIOR_FROZEN_GATE_PASSED_REVERIFIED',
            'risk_account_status':'PRIOR_ACCOUNTS_UNDERPERFORM_SIMPLE_VOL10_NO_PROMOTION',
            'research_round_completed':True,'performance_target_achieved':False,
            'terminated_strategy_status_preserved':index['status'],'external_gpt_review':'NOT_PERFORMED',
            'automation_created':False,'collection_daemon_started':False}
    write_json(OUT/'status.json',status)
    return configurations


def collect_paths():
    protocol=check_protocol()
    paths={ROOT/r['path'] for r in protocol['frozen_files']}
    paths|={p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    paths|={Path(__file__),ROOT/'research/verify_m1_m2_monthly_increment_v1.py'}
    risk=ROOT/'reports/research/510300_monthly_downside_forecast_v1'
    paths|={p for p in risk.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for protocol_name in ['protocol.json','account_protocol.json']:
        for item in read(risk/protocol_name)['frozen_files']:
            p=ROOT/item['path']
            assert sha(p)==item['sha256']
            paths.add(p)
    for item in read(risk/'result.json')['metrics']:
        if item['reused']:
            paths|={p for p in (ROOT/item['source']).parent.iterdir() if p.is_file()}
    extra=[
        'reports/research/510300_sharpe_1_2_latest_research.json',
        'reports/research/510300_post_selection_extension_inputs_v1/candidate_prices.parquet',
        'reports/research/510300_post_selection_extension_inputs_v1/candidate_dividend_coverage.json',
        'reports/research/510300_post_selection_extension_inputs_v1/result.json',
        'reports/research/510300_post_selection_extension_inputs_v1/saved_verification_receipt.json',
        'reports/research/510300_post_selection_extension_inputs_v1/dependency_graph.json',
        'reports/research/510300_post_selection_extension_inputs_v1/source_precision_comparison.csv',
        'reports/research/510300_monthly_single_factor_walkforward_v1/result.json',
        'reports/research/510300_monthly_single_factor_walkforward_v1/protocol.json',
        'reports/research/510300_monthly_single_factor_walkforward_v1/account_metrics.csv',
        'reports/research/510300_conditional_variance_budget_v1/acceptance_outcome.json',
        'reports/research/510300_conditional_variance_budget_v1/result.json',
        'reports/backtest/macro_01_m2_accel_trend_v1.json',
        'config/macro_01_m2_accel_trend_v1.yaml',
        'config/har_volatility_challenger.yaml',
        'deliverables/510300策略终止_20260917/策略终止说明.md',
        'data/reference/510300_dividends_coverage.json']
    paths|={ROOT/p for p in extra}
    assert all(p.is_file() for p in paths)
    return import_closure(paths)


def check_zip(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len(names)==len(set(names))
        rows=list(csv.DictReader(StringIO(z.read('FILE_INDEX.csv').decode('utf-8-sig'))))
        assert len(rows)==len(names)-1
        assert {r['path'] for r in rows}==set(names)-{'FILE_INDEX.csv'}
        import hashlib
        for row in rows:
            raw=z.read(row['path'])
            assert len(raw)==int(row['bytes']) and hashlib.sha256(raw).hexdigest()==row['sha256']
    return {'status':'PASS_CRC_DUPLICATE_INDEX_SIZE_AND_SHA256','members':len(names),'indexed_files':len(rows),
            'bytes':path.stat().st_size,'sha256':sha(path)}


def main():
    configurations=prepare_context()
    files=collect_paths()
    build_time=now()
    stem='510300_M1M2月频增量与风险复核_V1_GPT审阅_20260917'
    target=ROOT/'deliverables'/f'{stem}.zip'
    building=target.with_suffix('.building.zip')
    assert not target.exists() and not building.exists(), '目标包已存在，禁止覆盖'
    entries={p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(files)}
    for path in configurations:
        entries['history/configuration_snapshot/'+path.name]=path.read_bytes()
    report='reports/research/510300_m1_m2_monthly_increment_v1/'
    entries['00_README_FIRST.md']=f'''# 510300 M1/M2月频增量与风险复核

截至2026-09-11的有限历史研究；打包时间{build_time}。

本轮旧口径宏观增量未通过，新口径样本不足；B/D账户未运行。已有风险预测门通过，但C账户收益和夏普低于A。85/15保持终止。没有独立达标证据，没有订单。

建议阅读顺序：

1. {report}研究报告.md
2. {report}USER_REQUEST.md及GPT审阅提示词.md
3. {report}protocol.json、source_admission.json、prediction_result.json、status.json
4. {report}证据地图与排除项.md及去重与选择历史.md
5. {report}verification中的复核回执与risk_reused_account_comparison.csv
6. FILE_INDEX.csv及history/configuration_snapshot中的上游登记配置

直接来源、全部预测、保存系数、区块索引、已复核风险研究的全部20份账户及本轮代码均包含。原始PBC目录及失败回执保留。完整旧实验配置是上下文，不代表附带整个旧项目的运行数据。

离线复核使用Python、numpy、pandas、scipy、requests、beautifulsoup4等现有运行环境。在本包根目录运行 `python -m research.verify_m1_m2_monthly_increment_v1 --output 本次离线复核`，会写入新的核对回执，不重新拟合、不重新抽样、不下载、不生成账户。输出目录必须没有同名回执；不需运行admit/freeze/predict。

读取HTML和CSV即可人工审阅。全部新统计结果属于已用历史的顺序回放；原官网页面为现时重建，严格前向月份为0。只有本地算术核对及包结构校验，不代表外部GPT已经评审。包不扩展研究或交易权限。
'''.encode('utf-8')
    entries['01_GPT_REVIEW_PROMPT.md']=(OUT/'GPT审阅提示词.md').read_bytes()
    import importlib.metadata
    versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','requests','beautifulsoup4','scikit-learn','pyarrow']}
    entries['environment.json']=json.dumps({'python':sys.version,'packages':versions,'platform':'Windows',
                                           'full_environment_not_bundled':True},ensure_ascii=False,indent=2).encode('utf-8')
    index=StringIO(newline='')
    writer=csv.DictWriter(index,fieldnames=['path','bytes','sha256'])
    writer.writeheader()
    import hashlib
    for path,raw in sorted(entries.items()):
        writer.writerow({'path':path,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    entries['FILE_INDEX.csv']=index.getvalue().encode('utf-8-sig')
    with zipfile.ZipFile(building,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,raw in sorted(entries.items()):
            z.writestr(name,raw)
    structure=check_zip(building)
    restored=ROOT/'reports/research/510300_m1_m2_delivery_verification_20260917/restored_packet'
    restored.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(building) as z:
        for name in z.namelist():
            destination=(restored/name).resolve()
            assert destination.is_relative_to(restored.resolve())
        z.extractall(restored)
    verify_out=restored.parent/'offline_verification'
    env=os.environ.copy()
    env['PYTHONIOENCODING']='utf-8'
    run=subprocess.run([sys.executable,'-m','research.verify_m1_m2_monthly_increment_v1','--output',str(verify_out)],
                       cwd=restored,env=env,capture_output=True,text=True,encoding='utf-8')
    write_json(restored.parent/'replay_process_receipt.json',{'completed_at':now(),'exit_code':run.returncode,
               'stdout':run.stdout,'stderr':run.stderr,'new_fits':0,'new_accounts':0,'network_requests':0})
    assert run.returncode==0, '新解压目录离线复核失败，保留building包及回执：'+run.stderr
    building.rename(target)
    receipt={'completed_at':now(),'status':'PASS_STRUCTURAL_AND_FRESH_EXTRACTION_SAVED_RECOMPUTATION',
             'zip':target.relative_to(ROOT).as_posix(),**structure,
             'delivery_status':'PASS_STRUCTURAL_AND_FRESH_EXTRACTION_SAVED_RECOMPUTATION',
             'fresh_extraction':restored.relative_to(ROOT).as_posix(),
             'fresh_replay_receipt':(restored.parent/'replay_process_receipt.json').relative_to(ROOT).as_posix(),
             'source_report_months':104,'risk_account_ledgers':20,'registered_configuration_files':len(configurations),
             'external_review_performed':False,'new_fits_in_packaging':0,'new_accounts_in_packaging':0,
             'strict_forward_evidence_months':0,'performance_target_achieved':False}
    write_json(OUT/'delivery_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
