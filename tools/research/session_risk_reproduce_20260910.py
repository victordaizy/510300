"""在新目录复现固定时段研究，或只核验现有输入和滚动结果。"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.research.session_risk_transfer_20260910 import OUT, validate, digest, write_json


def audit():
    validate();rows=[]
    for p in sorted(OUT.glob('*/*/*_ledger.parquet')):
        l=pd.read_parquet(p);r=l.net_return.to_numpy(float);window=726
        if len(r)<window:continue
        w=np.lib.stride_tricks.sliding_window_view(r,window)
        c=np.expm1(np.log1p(w).sum(axis=1)*242/window)
        sd=w.std(axis=1,ddof=1)*np.sqrt(242)
        sharpe=np.divide(w.mean(axis=1)*242,sd,out=np.full(len(w),np.nan),where=sd>1e-15)
        period,cost=p.parent.parent.name,p.parent.name;model=p.stem.removesuffix('_ledger')
        rows.append({'period':period,'cost':cost,'model':model,'windows':len(c),
            'cagr_min':float(np.min(c)),'cagr_median':float(np.median(c)),
            'sharpe_min':float(np.nanmin(sharpe)),'sharpe_median':float(np.nanmedian(sharpe)),
            'joint_pass_fraction':float(np.mean((c>=.1)&(sharpe>=1.2))),
            'independent_windows':False})
    pd.DataFrame(rows).to_csv(OUT/'rolling_726_summary.csv',index=False)
    write_json(OUT/'audit_receipt.json',{'rolling_paths':len(rows),'input_hashes':'PASS',
        'code_sha256':digest(ROOT/'tools/research/session_risk_transfer_20260910.py'),
        'research_only':True,'live_ready':False})
    print('ROLLING_AUDIT_COMPLETE',len(rows))


def prepare(destination):
    protocol=validate();dest=Path(destination).resolve()
    if dest.exists():raise FileExistsError('复现目录必须不存在，不覆盖旧结果。')
    dest.mkdir(parents=True)
    for item in protocol['files']:
        p=item['path'];target=dest/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
        if digest(target)!=item['sha256']:raise ValueError('复制后的输入摘要不符')
    return dest


def stage(destination,action):
    dest=Path(destination).resolve()
    if not (dest/'tools/research/session_risk_transfer_20260910.py').exists():raise ValueError('先prepare建立复现目录')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    subprocess.run([sys.executable,str(dest/'tools/research/session_risk_transfer_20260910.py'),action],cwd=dest,env=env,check=True)


def compare(destination):
    dest=Path(destination).resolve();other=dest/OUT.relative_to(ROOT)
    left=pd.read_csv(OUT/'metrics.csv').sort_values(['period','cost','model']).reset_index(drop=True)
    right=pd.read_csv(other/'metrics.csv').sort_values(['period','cost','model']).reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right,check_exact=True)
    count=0
    for p in OUT.glob('*/*/*.parquet'):
        target=other/p.relative_to(OUT)
        pd.testing.assert_frame_equal(pd.read_parquet(p),pd.read_parquet(target),check_exact=True)
        count+=1
    receipt={'status':'PASS','policy_and_control_metrics':len(left),'parquet_frames_exact':count,
        'metrics_csv_byte_identical':digest(OUT/'metrics.csv')==digest(other/'metrics.csv'),
        'independent_market_evidence_added':False}
    write_json(OUT/'fresh_workspace_reproduction.json',receipt);print(json.dumps(receipt,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','audit','prepare','freeze','earlier','main','finish','compare'])
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    if args.action=='verify':validate();print('INPUT_HASHES_PASS')
    elif args.action=='audit':audit()
    elif args.destination is None:parser.error('需要--destination新目录路径')
    elif args.action=='prepare':print(prepare(args.destination))
    elif args.action=='compare':compare(args.destination)
    else:stage(args.destination,args.action)
if __name__=='__main__':main()
