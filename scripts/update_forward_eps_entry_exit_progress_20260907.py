"""保存旧索引，再登记十五轮的真实进度和用户取消数值审阅包的要求。"""
from pathlib import Path
import sys
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now

TARGET = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
SNAPSHOT = ROOT / 'reports/research/510300_sharpe_1_2_before_entry_exit_update_20260907.json'
latest = read(TARGET)
assert [r['round'] for r in latest['completed_rounds']] == list(range(1,11)), '先核对已有新进度，禁止重复覆盖'
new_studies = [
    (11,'510300_forward_eps_monthly_policy_v2_csi','前瞻EPS月末收益风险仓位',8,12,30),
    (12,'510300_forward_eps_residual_policy_v2','全部成熟价格基线与EPS误差修正',2,4,12),
    (13,'510300_forward_eps_explicit_entry_exit_v2','盈利三条件进入与明确退出',2,2,6),
    (14,'510300_forward_eps_forecast_entry_exit_v1','盈利预测进入与每日或月末退出',4,4,10),
    (15,'510300_forward_eps_utility_exit_v1','原EPS仓位加预测期限与趋势退出',2,2,8)]
results = {}
for round_number, name, title, configs, runs, accounts in new_studies:
    path = ROOT / 'reports/research' / name / 'result.json'
    result = read(path)
    assert read(path.parent / 'saved_numerical_verification.json')['status'].startswith('PASS')
    results[round_number] = result
    primary = {r['cost']:r for r in result['primary']}
    valid = [r for r in result['all_metrics'] if r['cost']=='BASE' and r['model']!='BUY_HOLD'
             and r['net_sharpe'] is not None and math.isfinite(float(r['net_sharpe']))]
    entry = {'round':round_number,'study':result['study_id'],'title':title,'status':'COMPLETED_TARGET_NOT_MET',
             'result':str(path.relative_to(ROOT)).replace('\\','/'),'candidate_configurations':configs,
             'evaluated_candidate_source_runs':runs,'evaluation_accounts':accounts,
             'primary_base':primary['BASE'],'primary_stress':primary['STRESS'],
             'post_selected_best_base':max(valid,key=lambda r:r['net_sharpe']) if valid else None,
             'corrected_source_version':'v2','goal_achieved':False}
    if round_number==11:
        entry.update({'financial_diagnostic_result':'reports/research/510300_forward_eps_monthly_policy_v1_financial/result.json',
                      'legacy_low_coverage_source_result':'reports/research/510300_forward_eps_monthly_policy_v1_csi/result.json',
                      'account_count_note':'金融十条、旧CSI源十条、修正CSI源十条；8种方法范围，含4种源修正重放。'})
    elif round_number==12:
        entry.update({'legacy_source_result':'reports/research/510300_forward_eps_residual_policy_v1/result.json',
                      'account_count_note':'旧源六条与正确源六条；两种方法各有两次来源版本运行。'})
    elif round_number==13:
        entry.update({'registered_but_not_run_legacy_source_configurations':2,
                      'account_count_note':'旧V1只登记未跑，V2首次实际评价六条；不把零成交夏普写成零。'})
    elif round_number==15:
        entry['account_count_note']='两个新增退出候选，加原EPS仓位与买入持有重复对照，各双费用，共八条。'
    latest['completed_rounds'].append(entry)

best = next(r for r in results[11]['all_metrics'] if r['cost']=='BASE' and r['model']=='E3_FORWARD_EPS')
latest['post_selected_best_base'] = {'round':11,'title':'仅前瞻EPS月末收益风险仓位','source_result':latest['completed_rounds'][10]['result'],**best}
latest['updated_at'] = now()
latest['status'] = 'CONTINUING_FORWARD_EPS_AND_COHERENT_ENTRY_EXIT_TARGET_NOT_MET'
latest['goal_achieved'] = False
latest['independent_high_sharpe_evidence'] = 'NOT_ESTABLISHED'
latest['registered_configurations_in_this_resumption'] = 179
latest['evaluated_configurations_in_this_resumption'] = 179
latest['evaluated_candidate_source_runs_including_corrected_replays'] = 185
latest['registered_candidate_source_runs_including_unrun_legacy_bindings'] = 187
latest['evaluation_accounts_in_this_resumption'] = 408
assert sum(x['candidate_configurations'] for x in latest['completed_rounds']) == 179
assert sum(x['evaluation_accounts'] for x in latest['completed_rounds']) == 408
latest['count_warning'] = '179个不同方法或范围配置；185次已评价候选与来源版本；另两个旧绑定仅登记未运行，登记版本187。408条账户含重复对照及源更正重放，不是独立样本。最新正确源五轮40条账户。'
latest['pending_source_work'] = [r for r in latest['running_studies'] if r['study']!='510300_FORWARD_EPS_MONTHLY_POLICY_V1']
latest['running_studies'] = []
latest['process_state_note'] = '原3352原件采集、旧接续及五轮正确源计算均已结束，当前无这些模型或采集子进程在运行。研究目标继续。'
latest['partial_rounds'] = []
latest['prepare_gpt_numerical_review_package'] = False
latest['delivery_preference'] = '明确中文进出场规则与普通结果文件，不准备GPT审阅数值包。'
latest['latest_continuation_note'] = 'docs/510300_FORWARD_EPS_ENTRY_EXIT_CONTINUATION_20260907.md'
latest['next_work'] = [
    '按保存公司月度记录区分真正盈利上修下修、同报告延用、跨年度切换与缺失，解释43零7负的修正中位数。',
    '预先登记有限的新前瞻EPS修正广度及幅度因子，规则同时包含协调一致的入场、持有、退出及再入场，然后实际完成账户。',
    '继续免费多机构预测、参考价格和股数口径对齐、公募净申购、股东回报与全部期限逆回购资料，不把尚未形成的输入写成已验证。',
    '不重启完成的采集，不重跑旧失败账户，不再运行数值审阅包脚本；保留完整历史和费用，目标保持active。']
latest['checks'] = '最新五轮40账户保存数值核对通过；CSI来源采用2791份保存原文、8376EPS、27084公司月度记录、139月末、47标签、108模型收据。第十五轮五项关键测试、八账户与十八次退出触发只读核对通过。无重新拟合或随机抽样用于核对，无全PDF字节重验、无额外安全审计、无新GPT包或外部审阅。'
latest['new_evidence'].extend([
    '3352固定CSI原件全部归档，30062页、2943290971字节，采集与接续正常结束。',
    '合法EPS行名和独立年度表头修正，成功503份提升至2791份，8376条年度EPS、361公司；561份缺口保留。',
    '正确源月末前瞻EPS模型基础夏普0.6189727421，原价格加EPS主方案0.5222882061，均未达到1.2。',
    '利润修正中位数50有效月末43次零、7次负，三条件同时为正方案零交易且夏普不可算。',
    '固定预测入场每日退出仅EPS夏普0.0597001951，对照月末退出0.1547615359；七轮完整进出场已保存。',
    '保留原入场与仓位的预测期限退出没有触发；加趋势退出九次，夏普0.1112598218，出现多次重入后次日再退出。',
    '已按用户要求交付中文因子与进出场说明及普通CSV，取消本轮GPT数值审阅包。'])
e = latest['forward_eps_history_evidence']
e['financial_monthly_valid_origins'] = e.pop('monthly_valid_origins',38)
e['financial_first_model_fit'] = e.pop('first_model_fit','2024-06-28')
e['financial_saved_numeric_check'] = e.pop('new_saved_numeric_check',None)
e.update({'historical_csi_archived_reports':3352,'historical_csi_parsed_reports':2791,'historical_csi_eps_facts':8376,
          'historical_csi_eps_companies':361,'historical_csi_unresolved_reports':561,'historical_csi_valid_monthly_origins':50,
          'historical_csi_first_valid_monthly_origin':'2022-06-30','source_facts_version':'research/forward_eps_guosen_history_v2.py',
          'corrected_source_result':'reports/research/510300_forward_eps_csi_facts_v2/result.json','main_study_not_complete':False,
          'market_consensus':False,'exact_next_twelve_month_eps_constructed':False,
          'latest_saved_numeric_check':read(ROOT / 'reports/research/510300_forward_eps_monthly_policy_v2_csi/saved_numerical_verification.json')})
latest['completed_source_rebuilds'].append({'study':'510300_FORWARD_EPS_CSI_ORIGINALS_AND_FACTS_V2','originals':3352,
                                         'parsed_reports':2791,'forecast_eps_facts':8376,'unresolved_reports':561,
                                         'result':'reports/research/510300_forward_eps_csi_facts_v2/result.json'})
latest['forward_eps_financial_attribution_evidence'] = {'result':'reports/research/510300_forward_eps_exposure_attribution_v1_2_financial/result.json',
                                                    'chinese_explanation':'docs/510300_FORWARD_EPS_FINANCIAL_FAILURE_ATTRIBUTION_20260906.md',
                                                    'conclusion':'共同模型期间平均持仓规模贡献较大，时变项为负且区间跨零；不是已确认稳定择时。'}
delivery = ROOT / 'deliverables/510300前瞻EPS进出场研究_20260906'
assert (delivery / '前瞻EPS策略_因子进出场与历史表现.md').exists()
latest['deliveries'].append({'created_at':now(),'type':'CHINESE_MD_AND_ORDINARY_CSV_NO_GPT_PACKAGE',
                             'directory':str(delivery),'main_document':str(delivery / '前瞻EPS策略_因子进出场与历史表现.md'),
                             'latest_complete_account_rows':40,'new_gpt_review_archive_created':False})


def normalize(value):
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,list):
        return [normalize(x) for x in value]
    if isinstance(value,dict):
        return {k:normalize(v) for k,v in value.items()}
    return value


save(SNAPSHOT,read(TARGET),exclusive=True)
save(TARGET,normalize(latest))
print('最新索引已更新：十五轮、179方法范围、408账户；目标未达，停止GPT数值审阅包。',flush=True)
