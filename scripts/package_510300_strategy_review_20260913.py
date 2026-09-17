"""构建单一策略诊断复核ZIP，并核验成员与解压后的只读经济复算。"""
from __future__ import annotations
import ast
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_strategy_review_diagnostics_v1'
DEST=ROOT/'deliverables/510300最新策略评审诊断_20260913'
ZIP=DEST/'510300最新策略评审诊断_GPT复核包.zip'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    DEST.mkdir(exist_ok=True)
    if ZIP.exists():
        raise RuntimeError('交付ZIP已经存在；应核验现有ZIP，不能静默覆盖。')
    files={}
    def add(path,member=None):
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        member=member or 'project/'+path.relative_to(ROOT).as_posix()
        if member in files and files[member]!=path:
            raise RuntimeError('成员冲突：'+member)
        files[member]=path
    def tree(path):
        for p in sorted(path.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                add(p)
    # 先保存用户可直接打开的根文档。
    for path in sorted((OUT/'delivery').iterdir()):
        if path.is_file():
            shutil.copy2(path,DEST/path.name)
            add(path,path.name)
    add(ROOT/'deliverables/510300完整策略与逐笔买卖说明_20260913/510300策略完整说明与全部历史买卖点.docx','09_原策略说明书.docx')
    add(OUT/'saved_economic_verification.json','10_保存结果经济复算.json')
    direct_trees=['510300_september_monthly_continuation_v1','510300_incremental_selected_intent_mix_v1',
                  '510300_point_pass_fixed_diagnostic_v1','510300_post_selection_extension_inputs_v1',
                  '510300_post_selection_data_feasibility_v1','510300_fixed_research_origin_v1']
    for name in direct_trees:
        tree(ROOT/'reports/research'/name)
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and 'delivery' not in path.relative_to(OUT).parts:
            add(path)
    graph=read(ROOT/'reports/research/510300_post_selection_extension_inputs_v1/dependency_graph.json')['nodes']
    source_seeds=[ROOT/p for p in read(OUT/'code_dependency_files.json')]
    source_seeds += [ROOT/n['program'] for n in graph]
    source_seeds += [ROOT/'scripts/verify_510300_strategy_review_saved.py',ROOT/'scripts/write_510300_strategy_review_report.py',Path(__file__)]
    seen=set();pending=list(source_seeds);literal_files=set()
    while pending:
        p=pending.pop()
        if p in seen:
            continue
        if not p.exists():
            raise FileNotFoundError(p)
        seen.add(p);add(p)
        parsed=ast.parse(p.read_text(encoding='utf-8-sig'))
        for node in ast.walk(parsed):
            modules=[]
            if isinstance(node,ast.ImportFrom) and node.module:
                modules=[node.module]
            elif isinstance(node,ast.Import):
                modules=[x.name for x in node.names]
            for module in modules:
                if module.startswith(('research.','scripts.')):
                    dependency=ROOT/Path(*module.split('.')).with_suffix('.py')
                    if dependency.exists():pending.append(dependency)
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith(('config/','reports/','data/')):
                f=ROOT/node.value
                if f.is_file():literal_files.add(f)
    for f in literal_files:add(f)
    for name in ['research/__init__.py','scripts/__init__.py','tests/test_strategy_review_diagnostics_v1.py',
                 'docs/510300_STRATEGY_REVIEW_DIAGNOSTICS_V1.md','docs/510300_RESEARCH_HEARTBEAT_THROUGH216_20260913.md',
                 'docs/510300_POINT_PASS_FIXED_DIAGNOSTIC_V1.md','docs/510300_POINT_PASS_FIXED_DIAGNOSTIC_NEXT_20260913.md',
                 'docs/510300_NEW_DAILY_INPUT_ADAPTER_V1.md','docs/510300_WAIT_NEW_COMPLETE_DAILY_DATA_20260913.md',
                 'config/510300_new_daily_input_adapter_runtime_v1.json','config/510300_fixed_date_continuation_validation_v1.json',
                 'config/510300_point_pass_fixed_diagnostic_v1.json','config/510300_september_monthly_continuation_v1.json']:
        if (ROOT/name).exists():add(ROOT/name)
    for n in graph:
        config=ROOT/n['configuration'];add(config)
        c=read(config)
        for key in ['rules']:
            if key in c and (ROOT/c[key]).is_file():add(ROOT/c[key])
        # 补充早期追踪直接读取过的原节点决定与账本。
        for cost in ['BASE','STRESS']:
            folder=ROOT/n['saved_folder']/'earlier_diagnostic'/cost
            for suffix in ['decisions.parquet','ledger.parquet']:
                p=folder/(n['node']+'_'+suffix)
                if p.exists():add(p)
    for name in ['510300_fixed_date_continuation_validation_v1','510300_new_daily_input_adapter_v1']:
        folder=ROOT/'reports/research'/name
        for path in folder.iterdir():
            if path.is_file():add(path)
    original_cfg=read(ROOT/'config/510300_adaptive_allocation_v1.json')
    for p in original_cfg['inputs'].values():add(ROOT/p)
    raw_lineage=read(OUT/'sources/20241008_open_price_lineage.json')
    for item in raw_lineage:add(ROOT/item['path'])
    coverage={'status':'DIRECT_DIAGNOSTIC_SOURCE_CLOSURE_PACKAGED','declared_current_dependency_nodes':len(graph),
              'current_original_account_paths':22,'python_source_modules':len(seen),
              'source_modules':[p.relative_to(ROOT).as_posix() for p in sorted(seen)],
              'declared_complete_result_trees':direct_trees+['510300_strategy_review_diagnostics_v1'],
              'direct_literal_files':[p.relative_to(ROOT).as_posix() for p in sorted(literal_files)],
              'exclusions':['虚拟环境和已安装依赖','其他资产与未使用的历史研究目录','整个历史搜索全部原始尝试','重复Word渲染文件'],
              'missing_evidence':['历史分钟VWAP','真实开盘委托、队列与可成交量','新的独立完整交易周期','有效独立策略试验数'],
              'security_audit':False,'external_review_performed':False,'upload_performed':False}
    save(DEST/'packaged_source_coverage.json',coverage);add(DEST/'packaged_source_coverage.json','packaged_source_coverage.json')
    total=sum(p.stat().st_size for p in files.values())
    assert total<500*1024*1024,'未压缩输入超过本轮500MiB预算'
    index=io.StringIO(newline='')
    writer=csv.DictWriter(index,fieldnames=['member','bytes','sha256','source_path'])
    writer.writeheader()
    for member,path in sorted(files.items()):
        writer.writerow({'member':member,'bytes':path.stat().st_size,'sha256':digest(path),
                         'source_path':str(path.relative_to(ROOT))})
    index_bytes=index.getvalue().encode('utf-8-sig')
    (DEST/'FILE_INDEX.csv').write_bytes(index_bytes)
    temporary=ZIP.with_suffix('.building.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for member,path in sorted(files.items()):z.write(path,member)
        z.writestr('FILE_INDEX.csv',index_bytes)
    with zipfile.ZipFile(temporary) as z:
        members=z.namelist();assert len(members)==len(set(members))
        assert z.testzip() is None
        assert set(members)==set(files)|{'FILE_INDEX.csv'}
        rows=list(csv.DictReader(io.StringIO(z.read('FILE_INDEX.csv').decode('utf-8-sig'))))
        for row in rows:
            content=z.read(row['member'])
            assert len(content)==int(row['bytes']) and hashlib.sha256(content).hexdigest()==row['sha256']
    temporary.replace(ZIP)
    sha=digest(ZIP)
    (DEST/(ZIP.name+'.sha256')).write_text(sha+'  '+ZIP.name+'\n',encoding='utf8')
    receipt={'status':'PASS_ZIP_CRC_DUPLICATES_INDEX_SIZE_HASH_COVERAGE','path':str(ZIP),'bytes':ZIP.stat().st_size,
             'sha256':sha,'members':len(members),'indexed_files':len(files),'uncompressed_source_bytes':total,
             'created_at':datetime.now().astimezone().isoformat(),'security_audit':False,'upload_performed':False,
             'external_gpt_review':False,'independent_performance_validation':False}
    save(DEST/'ZIP结构核验回执.json',receipt)
    # 在本轮明确的临时目录核验解压后的直接入口，原项目数据不作为回退路径。
    extraction=(DEST/'离线核验解压').resolve()
    assert extraction.is_relative_to(DEST.resolve()) and extraction.name=='离线核验解压'
    if extraction.exists():raise RuntimeError('离线核验目录已存在，不覆盖未知内容。')
    extraction.mkdir()
    with zipfile.ZipFile(ZIP) as z:z.extractall(extraction)
    project=extraction/'project'
    verified=subprocess.run([sys.executable,str(project/'scripts/verify_510300_strategy_review_saved.py'),
                             '--project',str(project),'--receipt',str(DEST/'ZIP解压后经济复算.json')],
                            cwd=project,capture_output=True,text=True,encoding='utf8')
    (DEST/'ZIP解压后经济复算日志.txt').write_text(verified.stdout+verified.stderr,encoding='utf8')
    assert verified.returncode==0,verified.stderr
    smoke=subprocess.run([sys.executable,'-c',
        "from research.strategy_review_full_graph_v1 import DiagnosticPipeline; p=DiagnosticPipeline('CAPITAL_100K'); [p.setting(k) for k in p.graph]; print('解压包独立加载27节点配置、原模型和完整输入成功。')"],
        cwd=project,capture_output=True,text=True,encoding='utf8')
    (DEST/'ZIP解压后代码加载日志.txt').write_text(smoke.stdout+smoke.stderr,encoding='utf8')
    assert smoke.returncode==0,smoke.stderr
    save(DEST/'交付完成回执.json',{'status':'PASS_STRUCTURAL_AND_EXTRACTED_SAVED_ECONOMIC_RECOMPUTATION',
         'zip':receipt,'extracted_saved_verification':read(DEST/'ZIP解压后经济复算.json'),
         'extracted_code_direct_inputs_load':'PASS_27_NODE_CONFIG_AND_MODELS',
         'fresh_full_diagnostic_replay_from_zip':'NOT_RUN_SAVED_ECONOMICS_RECOMPUTED',
         'report':'01_诊断报告.md','prompt':'02_GPT复核提示词.txt'})
    print(json.dumps({'交付完成':True,**receipt},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
