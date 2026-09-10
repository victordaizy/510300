"""在新目录复算固定研究并比较全部逐日账本；不覆盖源目录，不下载行情。"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
RUN=Path('research_runs/annual10_sequential_20260910')
MANIFEST=RUN/'reproduction_inputs.json'

def verify(root):
    rows=json.loads((root/MANIFEST).read_text(encoding='utf-8'))['files']
    for row in rows:
        p=root/row['path']
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():raise ValueError('输入路径非法')
        if not p.is_file() or p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('输入摘要不一致：'+row['path'])
    print('INPUTS_VERIFIED',len(rows),flush=True)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','rerun'])
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args();rows=verify(ROOT)
    if args.action=='verify':return
    if args.destination is None:parser.error('rerun需要指定尚不存在的--destination目录')
    dest=args.destination.resolve()
    if dest.exists():raise FileExistsError('禁止覆盖已有目录')
    dest.mkdir(parents=True)
    for rel in [r['path'] for r in rows]+[MANIFEST.as_posix()]:
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p)
    verify(dest)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    commands=[['annual10_sequential_20260910.py','freeze']]
    commands += [['annual10_sequential_20260910.py',period,cost] for period in ['evaluation','earlier_diagnostic'] for cost in ['BASE','STRESS']]
    commands += [['annual10_sequential_20260910.py','finish'],['annual10_sequential_audit_20260910.py']]
    for script,*extra in commands:
        subprocess.run([sys.executable,str(dest/'tools/research'/script),*extra],cwd=dest,env=env,check=True)
    import pandas as pd
    left=pd.read_csv(ROOT/RUN/'all_metrics.csv');right=pd.read_csv(dest/RUN/'all_metrics.csv')
    pd.testing.assert_frame_equal(left,right,check_exact=True)
    paths=list((ROOT/RUN).glob('*/*/*_ledger.parquet'))+list((ROOT/RUN).glob('*/*/sources/*_ledger.parquet'))
    for p in paths:
        pd.testing.assert_frame_equal(pd.read_parquet(p),pd.read_parquet(dest/p.relative_to(ROOT)),check_exact=True)
    result={'status':'PASS','metrics_rows':len(right),'daily_ledgers_exactly_equal':len(paths),'independent_market_evidence_added':False}
    (dest/RUN/'fresh_workspace_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
