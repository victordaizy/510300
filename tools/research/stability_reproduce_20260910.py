"""在新目录复现本轮，不覆盖原文件；不连接券商或启动定时任务。"""
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
RUN=Path('research_runs/stability_distribution_20260910')

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(destination):
    destination=destination.resolve()
    if destination.exists():raise FileExistsError('仅允许尚不存在的新复现目录')
    rows=json.loads((ROOT/RUN/'protocol.json').read_text(encoding='utf-8'))['files']
    for r in rows:
        src=ROOT/r['path']
        if not src.resolve().is_relative_to(ROOT.resolve()) or src.is_symlink():raise ValueError('输入路径越界')
        if src.stat().st_size!=r['bytes'] or digest(src)!=r['sha256']:raise ValueError('输入摘要不符：'+r['path'])
    destination.mkdir(parents=True)
    for r in rows:
        dst=destination/r['path'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/r['path'],dst)
    return destination,len(rows)

def rerun(destination):
    dst,count=prepare(destination)
    env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for action in ['freeze','run','verify']:
        with (dst/f'{action}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(dst/'tools/research/stability_distribution_20260910.py'),action],cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    import pandas as pd
    import numpy as np
    files=sorted((ROOT/RUN).glob('*/*/*_ledger.parquet'))+sorted((ROOT/RUN).glob('*/*/*_decisions.parquet'))
    exact=True
    for p in files:
        a,b=pd.read_parquet(p),pd.read_parquet(dst/p.relative_to(ROOT))
        pd.testing.assert_frame_equal(a,b,check_exact=True)
    a=pd.read_csv(ROOT/RUN/'metrics.csv').sort_values(['model','period','cost']).reset_index(drop=True)
    b=pd.read_csv(dst/RUN/'metrics.csv').sort_values(['model','period','cost']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    receipt=dict(status='PASS',verified_input_files=count,ledger_and_decision_tables_compared=len(files),
                 all_tables_cell_equal=exact,metrics_csv_byte_equal=digest(ROOT/RUN/'metrics.csv')==digest(dst/RUN/'metrics.csv'),
                 new_independent_market_evidence=False,source_code_sha256=digest(ROOT/'tools/research/stability_distribution_20260910.py'))
    (dst/'REPRODUCTION_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    rerun(p.parse_args().destination)
