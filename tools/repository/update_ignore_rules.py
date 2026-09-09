"""从发布索引生成紧凑忽略规则，并用Git验证全部还原路径。"""
from __future__ import annotations

import bisect
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]
MARKER='# 以下规则由本次发布文件索引生成：还原材料继续通过 Releases 管理。'


def escape(path: str) -> str:
    for char in ('[',']','*','?'):
        path=path.replace(char,'\\'+char)
    return path


def main() -> None:
    summary=json.loads((ROOT/'catalog/snapshot.json').read_text(encoding='utf-8'))
    rows=[]
    for index in summary['indexes']:
        with (ROOT/index['path']).open(encoding='utf-8',newline='') as handle:
            rows.extend({'path':r['path'],'storage':r['storage']} for r in csv.DictReader(handle))
    direct=defaultdict(list)
    children=defaultdict(set)
    counts=defaultdict(lambda:[0,0])
    git_paths=sorted(r['path'] for r in rows if r['storage']=='git')
    for row in rows:
        parts=row['path'].split('/')
        parent='/'.join(parts[:-1])
        direct[parent].append(row)
        index=0 if row['storage']=='git' else 1
        for depth in range(len(parts)):
            directory='/'.join(parts[:depth])
            counts[directory][index]+=1
            if depth:
                children['/'.join(parts[:depth-1])].add(directory)
    patterns=[]

    def visit(directory: str) -> None:
        git_count,release_count=counts[directory]
        if not release_count:
            return
        if directory and not git_count:
            patterns.append('/'+escape(directory)+'/')
            return
        if directory.count('/')>=1 and release_count>=64 and git_count<=8:
            prefix=directory+'/'
            patterns.append('/'+escape(directory)+'/**')
            selected=[]
            position=bisect.bisect_left(git_paths,prefix)
            while position<len(git_paths) and git_paths[position].startswith(prefix):
                selected.append(git_paths[position]); position+=1
            parents=set()
            for path in selected:
                parts=path.split('/')
                for depth in range(len(directory.split('/'))+1,len(parts)):
                    parents.add('/'.join(parts[:depth]))
            patterns.extend('!/'+escape(p)+'/' for p in sorted(parents,key=lambda p:(p.count('/'),p)))
            patterns.extend('!/'+escape(p) for p in selected)
            return
        patterns.extend('/'+escape(r['path']) for r in direct[directory] if r['storage']=='release')
        for child in sorted(children[directory]):
            visit(child)

    visit('')
    path=ROOT/'.gitignore'
    original=path.read_text(encoding='utf-8').split(MARKER)[0].rstrip()
    path.write_text(original+'\n\n'+MARKER+'\n'+'\n'.join(patterns)+'\n',encoding='utf-8',newline='\n')
    release_paths={r['path'] for r in rows if r['storage']=='release'}
    command=['git','-c','core.quotepath=false','check-ignore','--no-index','--stdin','-z']
    checked=subprocess.run(command,cwd=ROOT,input=('\0'.join(sorted(release_paths))+'\0').encode('utf-8'),capture_output=True,check=False)
    if checked.returncode not in {0,1}:
        raise RuntimeError(checked.stderr.decode('utf-8',errors='replace'))
    matched=set(checked.stdout.decode('utf-8').rstrip('\0').split('\0'))
    missing=sorted(release_paths-matched)
    git_checked=subprocess.run(command,cwd=ROOT,input=('\0'.join(git_paths)+'\0').encode('utf-8'),capture_output=True,check=False)
    if git_checked.returncode not in {0,1}:
        raise RuntimeError(git_checked.stderr.decode('utf-8',errors='replace'))
    accidentally_ignored=[p for p in git_checked.stdout.decode('utf-8').split('\0') if p]
    result={'status':'PASS_RELEASE_RESTORE_IGNORE_COVERAGE' if not missing and not accidentally_ignored else 'FAILED','generated_patterns':len(patterns),'release_paths_checked':len(release_paths),'git_paths_checked':len(git_paths),'release_paths_not_ignored':missing,'git_paths_accidentally_ignored':accidentally_ignored}
    (ROOT/'catalog/IGNORE_RULES_VERIFICATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if missing or accidentally_ignored:
        raise SystemExit(1)


if __name__=='__main__':
    main()
