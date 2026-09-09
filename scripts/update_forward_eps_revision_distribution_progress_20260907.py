"""接续十五轮索引，保存旧快照后登记第十六轮和下一项来源工作。"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now

TARGET=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
SNAPSHOT=ROOT/'reports/research/510300_sharpe_1_2_before_revision_distribution_update_20260907.json'
latest=read(TARGET)
assert [x['round'] for x in latest['completed_rounds']]==list(range(1,16)), '已有其他进度，禁止覆盖'
out=ROOT/'reports/research/510300_forward_eps_revision_distribution_policy_v1'
r=read(out/'result.json');verification=read(out/'saved_numerical_verification.json')
assert verification['status'].startswith('PASS') and r['evaluation_accounts']==14
primary={x['cost']:x for x in r['primary']}
best=max([x for x in r['all_metrics'] if x['cost']=='BASE' and x['model'].startswith('D')],key=lambda x:x['net_sharpe'])
entry={'round':16,'study':r['study_id'],'title':'前瞻盈利修正分布与一致进出场','status':'COMPLETED_TARGET_NOT_MET',
       'result':'reports/research/510300_forward_eps_revision_distribution_policy_v1/result.json',
       'candidate_configurations':3,'evaluated_candidate_source_runs':3,'evaluation_accounts':14,
       'new_accounts_generated':6,'reused_control_accounts':8,'trained_models':72,
       'primary_base':primary['BASE'],'primary_stress':primary['STRESS'],'post_selected_best_base':best,
       'account_count_note':'三候选各双费用生成六账户，四种旧对照双费用八账户原值复用；评价十四账户。',
       'saved_numerical_verification':verification,'goal_achieved':False}
latest['completed_rounds'].append(entry)
latest['updated_at']=now();latest['goal_achieved']=False
latest['status']='CONTINUING_FORWARD_EPS_VALUATION_SHARE_ALIGNMENT_TARGET_NOT_MET'
latest['registered_configurations_in_this_resumption']=182
latest['evaluated_configurations_in_this_resumption']=182
latest['evaluated_candidate_source_runs_including_corrected_replays']=188
latest['registered_candidate_source_runs_including_unrun_legacy_bindings']=190
latest['evaluation_accounts_in_this_resumption']=422
assert sum(x['candidate_configurations'] for x in latest['completed_rounds'])==182
assert sum(x['evaluation_accounts'] for x in latest['completed_rounds'])==422
assert latest['post_selected_best_base']['net_sharpe']>best['net_sharpe']
latest['count_warning']='182不同方法或范围，188次已评价候选来源版本，另两个旧绑定未运行，登记版本190。422评价账户含复用和重复对照及更正来源重放，不是独立样本。本轮十四账户中六条新生成、八条复用。'
latest['latest_continuation_note']='docs/510300_FORWARD_EPS_REVISION_DISTRIBUTION_CONTINUATION_20260907.md'
latest['running_studies']=[];latest['partial_rounds']=[]
latest['process_state_note']='第十六轮、保存数值核对与价格股数资料检查均已完成，无这些OS计算进程仍在执行；目标继续。'
latest['checks']='本轮六项关键测试、5500公司月份原预测配对、72个保存模型的训练时钟与标准化、十四账户与八次双费用趋势退出及保存区间通过。没有在复核中重新拟合或生成账户、下载、随机抽样，没有额外安全审计或GPT数值包。'
latest['next_work']=[
    '核对现有39158行财报面板的总股本、公告时点和单位，先与研报明确股本快照对齐；检查公司行为资料以连接研报基准日与交易日。',
    '利用已匹配7092条EPS公司月份的未复权价格，完成有真实股数依据的前瞻估值子集，保留83条缺价及其余股本缺口；不伪造精确股本或把复权因子当股数。',
    '核对真实历史权重与盈利预测代表性；继续免费多机构前瞻、公募净申购、股东回报、全部期限逆回购，形成有限新候选并写明协调进出场后完整检验。',
    '不重复旧源采集、旧账户或GPT包，不改本轮因子及退出阈值补成绩；目标1.2和稳定超额继续保持active。']
latest['new_evidence'].extend([
    '前瞻修正中位数原因已核实：有效月5383公司比较中同报告2304、新报告未变768、上修842、下修1469，57.07%为零；原计算可复现。',
    '第十六轮新增三修正分布因子，两种模型72拟合、十四账户（六新八复用），仅盈利分布夏普0.4143262605，价格加分布0.5233363162，一致进出场0.1529927426，均未达标。',
    '新主方案2023年10月至2024年2月首轮持仓亏26196.30816元，旧EPS同期现金；新模型全期比旧EPS少35904.65208元。',
    '现有月末未复权价格匹配7092条前瞻EPS公司月份，83条缺失；对应股数和总市值字段全部为空，39158行旧财报总股本只是待对齐候选。',
    '已经交付本轮中文失败原因、因子和完整进出场以及普通结果CSV，没有生成GPT数值审阅包。'])
latest['forward_eps_revision_diagnostic']={'result':'reports/research/510300_forward_eps_revision_diagnostic_v1/result.json',
                                         'all_comparable_company_months':5500,'valid_month_comparable_rows':5383,
                                         'same_report_rows':2304,'new_report_unchanged_rows':768,'up_rows':842,'down_rows':1469}
inventory=read(ROOT/'reports/research/510300_forward_eps_price_share_alignment_inventory_v1/result.json')
latest['pending_source_work'].append({'study':inventory['study_id'],'status':inventory['status'],
                                    'result':'reports/research/510300_forward_eps_price_share_alignment_inventory_v1/result.json',
                                    'unadjusted_month_end_close_matches':7092,'missing_month_end_close_rows':83,
                                    'historical_share_field_nonnull_rows':0,'historical_financial_panel_candidate_rows':39158,
                                    'remaining_work':'核对研报、财报与公司行为的当时股数口径，形成可比前瞻估值；未构造或回测此新因子。'})
latest['prepare_gpt_numerical_review_package']=False
folder=ROOT/'deliverables/510300前瞻盈利修正分布与进出场_20260907'
assert (folder/'前瞻盈利修正_失败归因与完整策略说明.md').exists()
latest['deliveries'].append({'created_at':now(),'type':'CHINESE_MD_AND_ORDINARY_CSV_NO_GPT_PACKAGE',
                             'directory':str(folder),'main_document':str(folder/'前瞻盈利修正_失败归因与完整策略说明.md'),
                             'evaluation_accounts':14,'new_accounts_generated':6,'reused_control_accounts':8,
                             'new_gpt_review_archive_created':False})
save(SNAPSHOT,read(TARGET),exclusive=True)
save(TARGET,latest)
print('索引更新为十六轮、182配置、422评价账户，当前最好夏普仍0.619；目标继续。',flush=True)
