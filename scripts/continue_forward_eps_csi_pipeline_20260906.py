"""等待已确认的Windows采集进程，再按冻结方法完成前瞻主研究。"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now
from research.forward_eps_guosen_history_v1 import identity

OUT=ROOT/'reports/research/510300_forward_eps_csi_pipeline_v1'


def wait_collector(pid,expected_time):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME)]
    kernel.GetProcessTimes.restype=wintypes.BOOL
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000|0x100000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error==87:
            return {'collector_pid':pid,'status':'REGISTERED_COLLECTOR_PID_NO_LONGER_EXISTS','checked_at':now()}
        raise OSError(error,'无法查询已登记采集进程，不能据此宣称采集终止')
    try:
        creation,exit_time,kernel_time,user_time=(wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel_time),ctypes.byref(user_time)):
            raise OSError(ctypes.get_last_error(),'无法确认采集进程创建时刻')
        ticks=(creation.dwHighDateTime<<32)|creation.dwLowDateTime
        seconds=ticks/10000000-11644473600
        expected=datetime.fromisoformat(expected_time).timestamp()
        if abs(seconds-expected)>1:
            return {'collector_pid':pid,'status':'REGISTERED_COLLECTOR_TERMINATED_PID_REUSED','checked_at':now(),
                    'observed_creation_unix':seconds,'expected_creation_unix':expected}
        print('已绑定现有采集进程句柄，等待3352份登记原件处理结束。',flush=True)
        while True:
            code=kernel.WaitForSingleObject(handle,10000)
            if code==0:return {'collector_pid':pid,'creation_unix':seconds,'status':'COLLECTOR_PROCESS_TERMINATED','checked_at':now()}
            if code!=258:raise OSError(ctypes.get_last_error(),'等待采集句柄失败，未将观察超时视作终止')
            save(OUT/'status.json',{'status':'WAITING_CONFIRMED_COLLECTOR_HANDLE','collector_pid':pid,'collector_creation_unix':seconds,'checked_at':now(),'new_research_accounts_from_this_pipeline':0})
    finally:kernel.CloseHandle(handle)


def main(pid,created):
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),ROOT/'research/forward_eps_guosen_history_v1_1.py',ROOT/'research/forward_eps_monthly_policy_v1.py',
           ROOT/'config/510300_forward_eps_csi_facts_v1_1_manifest.json',ROOT/'config/510300_forward_eps_monthly_policy_v1_manifest.json']
    save(OUT/'RUN_STARTED.json',{'started_at':now(),'collector_pid':pid,'collector_creation_date':created,
         'continuation_steps_frozen_before_execution':True,'files':[identity(p) for p in paths]},exclusive=True)
    try:
        terminal=wait_collector(pid,created);save(OUT/'collector_terminal_receipt.json',terminal,exclusive=True)
        collection=ROOT/'reports/research/510300_forward_eps_csi_originals_v1/result.json'
        if not collection.exists():raise RuntimeError('原采集进程已终止但没有最终来源结果；保留已有缓存，需检查错误后再决定恢复')
        source=read(collection)
        if source.get('selected_reports')!=3352:raise ValueError('最终原件队列与登记数量不符')
        steps=[('absolute_year_eps',[ '-m','research.forward_eps_guosen_history_v1_1','--scope','csi','--run'],ROOT/'reports/research/510300_forward_eps_csi_facts_v1_1/result.json'),
               ('monthly_features',['-m','research.forward_eps_monthly_policy_v1','--scope','csi','--prepare'],ROOT/'reports/research/510300_forward_eps_monthly_policy_v1_csi/source_receipt.json'),
               ('complete_accounts',['-m','research.forward_eps_monthly_policy_v1','--scope','csi','--run'],ROOT/'reports/research/510300_forward_eps_monthly_policy_v1_csi/result.json')]
        records=[]
        for name,args,expected in steps:
            if expected.exists():
                records.append({'step':name,'status':'ALREADY_COMPLETED_SAVED_RESULT_REUSED','result':identity(expected)})
                continue
            save(OUT/'status.json',{'status':'RUNNING_FROZEN_CONTINUATION_STEP','step':name,'started_at':now()})
            print('开始冻结的后续阶段',name,flush=True)
            with (OUT/(name+'.log')).open('x',encoding='utf-8') as log:
                result=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode!=0 or not expected.exists():raise RuntimeError('后续阶段未完成：'+name+'，退出码'+str(result.returncode))
            records.append({'step':name,'status':'COMPLETED_WITH_SAVED_RESULT','result':identity(expected)})
        save(OUT/'result.json',{'completed_at':now(),'status':'FROZEN_CSI_FORWARD_EPS_RESEARCH_COMPLETED_REVIEW_REQUIRED',
             'steps':records,'account_result':'reports/research/510300_forward_eps_monthly_policy_v1_csi/result.json','goal_achieved':False},exclusive=True)
        save(OUT/'status.json',{'completed_at':now(),'status':'COMPLETED_RESULTS_READY_FOR_REVIEW','goal_achieved':False})
        print('历史成分前瞻EPS研究已按冻结规则完成，完整结果等待数值复核和交付。',flush=True)
    except Exception as exc:
        save(OUT/'failure.json',{'failed_at':now(),'status':'CONTINUATION_STEP_FAILED','error_type':type(exc).__name__,'error':str(exc)},exclusive=True)
        save(OUT/'status.json',{'failed_at':now(),'status':'CONTINUATION_STEP_FAILED','error_type':type(exc).__name__,'error':str(exc)})
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--collector-pid',type=int,required=True);ap.add_argument('--collector-created',required=True);a=ap.parse_args()
    main(a.collector_pid,a.collector_created)
