"""在新目录完整重做本轮；验证输入摘要及全部资金/决策，不覆盖研究原件。"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REL=Path('research_runs/conclusion_beta_iteration_20260910')
SCRIPT=Path('tools/research/conclusion_beta_iteration_20260910.py')

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True,type=Path);args=parser.parse_args()
    dst=args.destination.resolve()
    if dst.exists():raise FileExistsError('必须使用尚不存在的新目录。')
    protocol=json.loads((ROOT/REL/'protocol.json').read_text())
    paths={x['path'] for x in protocol['files']}
    for period in ('evaluation','earlier_diagnostic'):
        for cost in ('BASE','STRESS'):
            paths.add(f'reports/research/510300_continuous_reference_min_variance_v1/{period}/{cost}/BUY_HOLD_ledger.parquet')
    for item in protocol['files']:
        assert digest(ROOT/item['path'])==item['sha256'],item['path']
    dst.mkdir(parents=True)
    for rel in sorted(paths):
        source=ROOT/rel;target=dst/rel
        assert target.resolve().is_relative_to(dst) and source.is_file()
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    env=dict(os.environ,PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for cmd in ('freeze','run'):
        subprocess.run([sys.executable,str(dst/SCRIPT),cmd],cwd=dst,env=env,check=True)
    import pandas as pd
    files=list((ROOT/REL).glob('*/*/*_ledger.parquet'))+list((ROOT/REL).glob('*/*/*_decisions.parquet'))
    for p in files:
        a=pd.read_parquet(p);b=pd.read_parquet(dst/p.relative_to(ROOT))
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    for name in ('factors.parquet','groups.parquet'):
        pd.testing.assert_frame_equal(pd.read_parquet(ROOT/REL/name),pd.read_parquet(dst/REL/name),check_exact=True)
    assert digest(ROOT/REL/'metrics.csv')==digest(dst/REL/'metrics.csv')
    receipt={'status':'PASS','daily_tables_exact':len(files),'factor_and_group_tables_exact':2,'metrics_csv_bytes_identical':True,
             'original_inputs':len(protocol['files']),'additional_saved_control_inputs':4,'strategy_parameter_changes':0,'new_independent_market_samples':0,
             'source_code_sha256':digest(ROOT/SCRIPT),'metrics_sha256':digest(ROOT/REL/'metrics.csv')}
    (ROOT/REL/'fresh_reproduction_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
