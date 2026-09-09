"""将已完成金融诊断和真实进行中的历史成分主研究写入当前索引。"""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now


def main():
    b=ROOT/'reports/research';path=b/'510300_sharpe_1_2_latest_research.json'
    delivery=ROOT/'deliverables/510300夏普1.2持续研究_前瞻EPS金融诊断与沪深300扩展_GPT审阅_20260906.delivery.json'
    if not delivery.exists():raise RuntimeError('新交付包尚未完成，不提前登记成功交付')
    old=read(path);save(b/'510300_pre_forward_eps_history_progress_snapshot_20260906.json',old,exclusive=True)
    result=read(b/'510300_forward_eps_monthly_policy_v1_financial/result.json')
    evidence=read(b/'510300_forward_eps_monthly_policy_v1_financial/saved_numerical_verification.json')
    pipeline=read(b/'510300_forward_eps_csi_pipeline_v1/status.json')
    archived=0
    for p in (b/'510300_forward_eps_csi_originals_v1/document_records').glob('*.json'):
        try:d=read(p)
        except (json.JSONDecodeError,UnicodeError):continue
        archived+=bool(d.get('source'))
    old.update({'updated_at':now(),'goal_achieved':False,
                'status':'TEN_ROUNDS_PLUS_FORWARD_EPS_FINANCIAL_DIAGNOSTIC_COMPLETE_CSI_PRIMARY_PIPELINE_RUNNING',
                'registered_configurations_in_this_resumption':169,'evaluated_configurations_in_this_resumption':165,
                'evaluation_accounts_in_this_resumption':352,
                'count_warning':'十轮161个已完成配置、342账户；第十一轮两范围各4配置已登记，金融4配置10账户完成，历史成分4配置10账户待完成。不同范围分别计数，含重复买入持有对照。',
                'latest_continuation_note':'docs/510300_FORWARD_EPS_HISTORY_CONTINUATION_20260906.md'})
    old['partial_rounds']=[{'round':11,'study':'510300_FORWARD_EPS_MONTHLY_POLICY_V1','status':'FINANCIAL_DIAGNOSTIC_COMPLETE_CSI_PRIMARY_RUNNING',
                           'completed_scope':'financial','primary_scope':'csi','registered_scope_configurations':8,
                           'completed_scope_configurations':4,'evaluation_accounts_completed':10,
                           'financial_result':'reports/research/510300_forward_eps_monthly_policy_v1_financial/result.json',
                           'primary_financial_base':result['primary'][0],'primary_financial_stress':result['primary'][1],
                           'csi_result_when_ready':'reports/research/510300_forward_eps_monthly_policy_v1_csi/result.json',
                           'methods_for_both_scopes_frozen_before_financial_returns':True}]
    old['running_studies']=[x for x in old.get('running_studies',[]) if x.get('study') not in ['510300_FORWARD_EPS_SOURCE_PILOT_V2','510300_FORWARD_EPS_MONTHLY_POLICY_V1']]
    old['running_studies'].append({'study':'510300_FORWARD_EPS_MONTHLY_POLICY_V1','status':pipeline['status'],
         'financial_diagnostic_complete':True,'historical_csi_directory_reports':3352,'historical_csi_directory_companies':369,
         'original_reports_archived_at_snapshot':archived,'snapshot_at':now(),'collector_exec_session_id':13017,
         'collector_pid':21760,'collector_creation_date':'2026-09-06T21:26:37.403124+08:00','pipeline_exec_session_id':41475,
         'pipeline_state':'reports/research/510300_forward_eps_csi_pipeline_v1/status.json',
         'collector_state':'reports/research/510300_forward_eps_csi_originals_v1/result.json',
         'remaining_work':'沿用现有采集和接续进程，按已冻结方法提取CSI年度EPS、构建月度因子、训练与完整账户；完成后只读数值复核及新交付。'})
    old['forward_eps_history_evidence']={'public_financial_directory_reports':1694,'institutions':37,'company_year_queries_with_recovery':144,
        'financial_originals':154,'financial_eps_reports':139,'financial_eps_facts':417,'financial_unresolved_reports':15,
        'financial_scope_companies_with_eps':11,'all_guosen_stock_directory_reports':7726,'historical_csi_union':668,
        'historical_csi_selected_reports':3352,'historical_csi_companies_with_reports':369,
        'source_facts_version':'research/forward_eps_guosen_history_v1_1.py','monthly_valid_origins':38,
        'first_model_fit':'2024-06-28','new_saved_numeric_check':evidence,'main_study_not_complete':True}
    old['next_work']=['确认既有3352原件采集进程和接续进程，沿用现有句柄，不因超时重启。',
        '等待冻结CSI年度EPS提取、月度特征和十条账户实际完成；检查日志和结果，不依据金融结果改方法。',
        '只读核对主研究全部来源和账户并交付完整包，再更新第十一轮整体状态。',
        '未达目标则继续多机构前瞻盈利、股本修正及公募、股东回报、全部期限流动性的有限新研究。']
    old.setdefault('deliveries',[]).append(read(delivery))
    save(path,old)
    receipt={'updated_at':now(),'status':old['status'],'registered_configurations':169,'evaluated_configurations':165,
             'evaluation_accounts':352,'collector_archive_snapshot':archived,'pipeline_status':pipeline['status'],'goal_achieved':False}
    save(b/'510300_forward_eps_history_progress_update_20260906.json',receipt,exclusive=True)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
