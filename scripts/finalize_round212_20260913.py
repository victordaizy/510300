"""交付已接纳的补充历史输入及最小续算依赖，直接转入批量连续重放。"""
import json
import shutil

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json
from research.post_selection_extension_inputs_v1_2 import ROOT,OUT,CONFIG,PRIMARY


def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    graph=json.loads((OUT/'dependency_graph.json').read_text(encoding='utf-8'))
    decision=('补充行情和因子接纳准备已完成，旧3456日55列数值及标记精确不变。固定两来源方案已裁剪为27个必要节点，'
        '其中仅22条来源及主账户需要实际重放；下一项直接批量恢复到8月31日，不重跑全部历史策略、不补慢因素。')
    detail=['上面四档指标仅复用209已经核对的历史简单方案，不是使用新增日期计算的新成绩。本轮零新候选、零新拟合、零新账户、零网络请求。',
        '', '## 已完成的输入接纳', '',
        '|检查|实际结果|','|---|---|',
        '|候选价格与因素|原3456日加20日，共3476日，延长到2026年9月11日|',
        '|旧价格前缀|价格、成交量额、日期类型及原修正标记全部保持|',
        '|旧因子前缀|55列、190080个数值及标记逐格精确相同|',
        '|新增因子|20日全部通过原有完整性标记|',
        '|新旧新浪重叠|10日的开高低收及量额全部精确相同|',
        '|两来源价格|共同30日开高低收全部精确相同|',
        '|两来源尾数|最大相差49份、48元；腾讯量和金额均可由新浪值按百单位舍入复现|',
        '|分红|引用211官方候选覆盖至9月11日，14事件，无新事件；原账本未修改|',
        '', '成交量以份、金额以人民币元保存。腾讯原响应的手数和万元金额显示精度较低，因此保留与原历史同来源的新浪未舍入值，不平均两源数值。成交量会影响通用量比因素，金额目前不进入原34个新增特征；源差异完整列在 source_precision_comparison.csv。',
        '开发中修复了两个输入检查问题：万元换算产生不足百万分之一元的浮点尾数，以及拼接后日期从毫秒变成微秒类型。保存了失败原因与修复记录；没有放宽行情50份／50元精度界限，没有改变旧价格或策略。最终保存文件重新读取后核对通过。',
        '', '## 为什么后面可以更快', '',
        '27个必要节点中，只有22条主历史和连续参考账户需要实际模拟，其余只生成目标、方向或读取已保存模型。共享节点只算一次；201调仓账户、181风险账户等没有被下游读取的实际成交不重复模拟。109只需要普通二十日波动乘数，整套条件方差拟合及它的历史父账户可以省去。',
        '原两份必要模型文件各有141条记录，最后模型都在2026年8月3日。下一项先直接接到8月31日并恢复连续状态，这一段没有新的月首拟合时点；九月再按原规则补月度模型，不让训练工程拖住前面的可用进度。',
        '', '## 必要节点及中文规则', '',
        '|来源|需要什么|直接父来源|固定规则与恢复状态|','|---|---|---|---|']
    kinds={'ACCOUNT_BOTH_COSTS':'两费用实际账户','ACCOUNT_BASE_ONLY':'仅基础实际账户',
        'ACCOUNT_BASE_TARGET_STRESS':'基础实际账户；压力只取目标','TARGET_ONLY':'只取目标',
        'FACTOR_TARGET_ONLY':'因素及目标','FACTOR_DIRECTION_ONLY':'只取方向','FACTOR_ONLY':'只取因素','SAVED_MODEL_ARTIFACT':'已保存模型'}
    for n in graph['nodes']:
        detail.append(f"|{n['node']}|{kinds[n['output_kind']]}|{'、'.join(n['direct_parents']) or '原价格因素或模型资料'}|{n['rule_cn']} 状态：{n['required_state_cn']}|")
    detail+=['','上表为依赖和执行摘要。每个节点的原配置、实际程序及纯计算入口均列于 minimal_dependencies.csv 和 dependency_graph.json，可直接用于接续；原完整策略中文说明仍以209交付为准。',
        '', '## 下一项直接做什么', '',
        '实现一个统一连续重放入口，按依赖顺序生成固定来源。主账户仍从原2020年起点开始，两条连续参考仍从2013年起点开始；仅在新目录输出。对原8月13日及之前正常区间核对持仓、现金、申请、成交和净值，再让8月14日按正常申请继续。',
        '8月14日的研究终点强制清仓被识别为结算动作，因此不能直接用该日清仓现金续算。下一项会在8月31日保存账户、当前持仓周期、退出等待、计数、分红权利和模型版本，并核对分段恢复与一次连续运行的一致性。当前尚未实现完整恢复状态，不能把这轮输入核对说成策略已经续算成功。',
        '所有补充日期发生在209正式冻结之前；严格前瞻证据仍为零。现有历史净夏普1.2及年化10%门槛通过的结论保持，独立稳定表现仍待证据。']
    write_json(OUT/'candidate_outcomes.json',{'recorded_at':now(),'new_candidates':0,'new_accounts':0,'goal_achieved':False,
        'status':'INPUTS_READY_MINIMAL_REPLAY_NEXT','old_numeric_feature_prefix_exact':True,
        'complete_checkpoint_verified':False,'dependency_nodes':27,'required_main_source_account_replays':22},exclusive=True)
    nxt=ROOT/'docs/510300_MINIMAL_CONTINUOUS_REPLAY_NEXT_20260913.md'
    doc=ROOT/'deliverables/510300策略续算提速_第212轮_20260913/补充因素及最小计算范围_中文说明.md'
    delivered=deliver_round(ROOT,OUT,CONFIG,doc,'补充因素接纳与最小计算范围',
        'COMPLETED_EXACT_INPUT_PREFIX_MINIMAL_REPLAY_READY',decision,'\n'.join(detail),nxt,
        nxt.read_text(encoding='utf-8').splitlines(),'MINIMAL_CONTINUOUS_REPLAY_PREPARED',
        '直接按27节点和22条必要账户批量恢复到8月31日；原正常区间核对后保存连续状态，不补慢因子或重扫旧研究')
    for name in ['candidate_prices.parquet','candidate_features.parquet']:
        shutil.copy2(OUT/name,doc.parent/name)
    for name in ['510300_round212_numeric_representation_fix.json','510300_round212_datetime_dtype_fix.json']:
        shutil.copy2(ROOT/'reports/research'/name,doc.parent/name)
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(path.read_text(encoding='utf-8'))
    configuration=str(CONFIG.relative_to(ROOT))
    index['latest_completed_round']['configuration']=configuration
    index['completed_rounds'][-1]['configuration']=configuration
    index['next_work'].update(candidate_round=213,registered=False,planned_settings=0,planned_new_accounts=22,
        planned_new_model_fits=0,planned_new_reference_accounts=0,planned_existing_reference_replays=2,
        external_data_required=False,research_class='FIXED_STRATEGY_MINIMAL_CONTINUOUS_ENGINEERING_REPLAY',
        saved_input_result=str((OUT/'result.json').relative_to(ROOT)),continuation_cutoff='2026-08-31')
    index['status']='ROUND212_HISTORICAL_POINT_PASS_INPUTS_EXACT_MINIMAL_CONTINUOUS_REPLAY_NEXT'
    index['latest_post_selection_extension_inputs']=str((OUT/'result.json').relative_to(ROOT))
    index['supplemental_inputs_ready_for_engineering']=True
    index['supplemental_data_accepted_into_strategy']=False
    index['strict_forward_evidence_days']=0
    index['independent_validation']='NOT_ESTABLISHED'
    write_json(path,index)
    print(json.dumps(delivered,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
