"""直接从最新完成结果和下一项规则生成接续，避免每轮复制过时模板再修正。"""
import argparse
import json
import re
import tomllib
from pathlib import Path
import pandas as pd

from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description='仅生成原持续任务更新参数，不直接修改自动任务。')
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--stamp', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'\d{8}', args.stamp) is not None, '日期标识必须是八位数字')
    index = read(ROOT/'reports/research/510300_sharpe_1_2_latest_research.json')
    latest, nxt = index['latest_completed_round'], index['next_work']
    require(latest['round']==args.round and not index['goal_achieved'] and not index['running_studies'], '当前完成状态不同')
    require(nxt['candidate_round']==args.round+1 and not nxt['registered'], '下一项准备状态不同')
    cfg_path = ROOT/latest.get('configuration', f"config/{latest['study'].lower()}.json")
    cfg = read(cfg_path)
    result_path = ROOT/latest['result']
    out = result_path.parent
    result, checked = read(result_path), read(out/'saved_verification_receipt.json')
    feature_dates = pd.read_parquet(ROOT/cfg['features'], columns=['date']).date
    tested, accepted = read(out/'tests_receipt.json'), read(out/'acceptance_outcome.json')
    require(result['study_id']==cfg['study_id']==latest['study'] and cfg['round']==args.round, '配置与结果身份不同')
    require(tested['exit_code']==0 and not accepted['goal_achieved'], '必要测试或结论状态不同')
    next_path = ROOT/nxt['source']
    next_rules = next_path.read_text(encoding='utf-8')
    delivery = [d for d in index['deliveries'] if args.round in d.get('rounds', [])]
    require(len(delivery)==1 and Path(delivery[0]['main_document']).is_file(), '最新中文交付缺失或不唯一')
    lines = [
        f"继续用户目标：{index['goal']}。每段历史、每档费用分别报告；既有历史已经反复研究，回测点值不等于新的独立验证。目标未完成时不标complete；有具体可推进工作时不标blocked。",
        '用户要求快、简单、灵活。现有config/510300_research_authority_v6.json允许新的有限候选、策略调整与历史滚动训练。保留所有旧冻结文件与失败结果；新参数试验应明确另行登记为使用过历史的参数校准，不改写旧成绩，也不能宣称独立验证。仅510300.SH和人民币现金，不增加其他ETF、券商、实盘、订单、Paper、Shadow、新任务、子代理或memory写入。继续暂停EPS、公募、估值等慢来源，无GPT数值包、ZIP、重复Word或额外安全审计，全部因子和进出场用中文。',
        f"上回合为PROGRESS：第{args.round}轮完成其预定工作、必要验证、冻结、保存核对、结论和中文交付；是否生成账户以以下实际计数为准。权威索引reports/research/510300_sharpe_1_2_latest_research.json；latest_completed_round={args.round}、running_studies为空、goal_achieved=false，无运行会话。禁止重跑已完成研究的prepare/run/verify/finalize及旧诊断。开始先查当前具体文件状态，接续未完成步骤，不宽泛重扫全部历史。",
        f"最新研究{cfg['study_id']}，候选{cfg['candidate_models']}。设置{cfg_path.relative_to(ROOT)}，结果{result_path.relative_to(ROOT)}，完整规则{cfg['rules']}。必要测试{tested['passed']}项；保存耗时{result['run_seconds']}秒，口径：{result.get('run_seconds_scope','核心计算，不包括开发、测试、核对与交付')}。实际新拟合{result['new_model_fits']}，新主及来源账户{result['new_accounts_generated']}，新较早账户{result['new_earlier_diagnostic_accounts']}，参考新增{result['new_reference_accounts']}。",
        accepted['decision'],
    ]
    for period, key in [('主历史', 'all_metrics'), ('较早历史', 'earlier_diagnostics')]:
        for row in result[key]:
            if row['model'] in cfg['candidate_models'] or row['model']=='ACCOUNT_VOLATILITY_EXPOSURE':
                lines.append(f"{period}／{row['model']}／{row['cost']}：净夏普{row['net_sharpe']}，复合年化{row['annualized_return']:.10%}，最大回撤{row['max_drawdown']:.10%}，成交{row['trade_count']}次。")
    lines += [
        f"保存核对已完成：{out.relative_to(ROOT)}/saved_verification_receipt.json，时间{checked['verified_at']}，状态{checked['status']}。核对不是独立绩效验证。最新中文交付{Path(delivery[0]['main_document']).relative_to(ROOT)}。费用和价格利润差可直接读当前目录saved_comparison_differences.csv或saved_profit_differences.csv，实际周期与账户核对读取同目录保存表，不重跑旧账户。",
        f"下一第{nxt['candidate_round']}轮状态{nxt['status']}，只准备完整事前方案；尚未实现、测试、冻结或回测。重点{nxt['focus']}。计划{nxt['planned_settings']}套候选、{nxt['planned_new_accounts']}新账户、{nxt['planned_new_model_fits']}新拟合、{nxt['planned_new_reference_accounts']}新参考；外部来源需求{nxt['external_data_required']}。具体方案来源{next_path.relative_to(ROOT)}，完整内容附后。已有后续进展时以实际文件为准，不覆盖重建。",
        f"共有因素{cfg['features']}，{len(feature_dates)}行至{feature_dates.max().date()}；本轮使用截止{cfg['data_cutoff']}。旧主first1852/anchor1851，较早前缀1852/first633/anchor632；原3456行基准终点8月14日开盘清仓，新连续账户以最后真实收盘结算，具体时钟按本轮结果。分红{cfg['dividends']}，14事件。每账户20万元、242年化、现金及无风险0；基础佣金.0002/min5/slip.0005，压力.0004/min5/slip.001。整百份、.001价位、T+1、方向涨跌停、分红登记/除息/支付分别记账。原冻结文件保持；Windows中文，PYTHONIOENCODING=utf-8，.venv\\Scripts\\python.exe，以模块运行。",
        '已完成的诊断不要重跑：reports/research/510300_joint_saved_frontier_through186/result.json为标准四情景联合比较；后续新轮次依各自结果，不把174旧夏普重点候选当作满足年化目标。reports/research/510300_saved_session_attribution_through188/result.json复核181/182的8账本11292行，隔夜归因不能用于删除隔夜、假设T+0或保证延后买入有效。',
        '完成必要测试和一次冻结后直接运行预定账户；保存核对、candidate_outcomes，用fast_round_delivery_v1交付并更新索引和下一具体有限方案。接续提示使用scripts/save_current_research_heartbeat_v2.py --round 当前完成轮次 --stamp 当天八位日期，一次生成reports/research/510300_heartbeat_update_through当前轮次_args.json；它直接读取最新结果与下一方案，不再复制旧THROUGH164模板后反复修正文案。随后通过automation_update更新原510300-1-2完整字段，回读TOML核对；不直接编辑自动任务文件、不新建重复任务。',
        '原持续任务保持ACTIVE及通知偏好。无变化安静，仅在实质进展、完成、失败或需用户行动时通知。不购买或消耗额度重置。',
        '以下为下一项完整事前规则：\n\n'+next_rules,
    ]
    if index.get('latest_saved_combination_screen'):
        screen = read(ROOT/index['latest_saved_combination_screen'])
        lines.insert(-1, f"批量提速已经落实，已完成的保存收益组合诊断不要重复运行：{index['latest_saved_combination_screen']}。"
            f"纳入{screen['distinct_eligible_paths']}条不同完整路径、读取{screen['ledger_files_read']}份账本，"
            f"两个预定起点共{screen['optimizer_objective_evaluations']}次目标函数评价，"
            f"最佳虚拟最弱门槛比值{screen['best_virtual_minimum_joint_ratio']}。虚拟日收益加权不能冒充完整组合账户，也不是不可行性证明。"
            '已结束第196轮十四套窗口与时点参数网格；其完整十八套比较保留在reports/research/510300_risk_window_clock_batch_v1/all_eighteen_setting_comparison.csv。'
            '优先复用通用程序和已保存来源，将同一明确机制的有限候选一次批量计算、一次集中核对和中文交付，避免每个小参数单独开一轮或反复编写同类审计。')
    if index.get('latest_incremental_saved_mix'):
        mix = read(ROOT/index['latest_incremental_saved_mix'])
        lines.insert(-1, f"增量收益组合诊断也已经完成，不要重跑：{index['latest_incremental_saved_mix']}，"
            f"复用{mix['reused_paths']}条旧矩阵，仅增读{mix['new_ledger_files_read']}份账本，共{mix['total_paths']}条完整路径，"
            f"两起点{mix['optimizer_objective_evaluations']}次评价，虚拟最弱门槛比值{mix['best_virtual_minimum_joint_ratio']}。"
            '第200轮只处理过截至199轮筛选的三个来源，不能把它当作后来增量筛选的实际账户验证；不能把原目标为零的等待持仓误算为空仓。'
            '已保存筛选权重仍非独立证据，不再重复优化同一矩阵。')
    if index.get('current_best_joint_comparison_candidate'):
        best_joint = index['current_best_joint_comparison_candidate']
        lines.insert(-1, f"当前夏普与年化联合比较优先使用索引current_best_joint_comparison_candidate："
            f"{best_joint['model']}，第{best_joint['round']}轮，来源{best_joint['source_result']}，"
            f"最弱门槛比值{best_joint['minimum_joint_ratio']}。旧174、181、182等比较字段保留的是不同历史比较口径，不能覆盖当前联合候选。历史点值是否通过以最新实际账户联合结果为准，不能把历史通过写成已经取得独立验证。")
    if index.get('historical_joint_point_target_met'):
        lines.insert(-1, f"历史联合点值已经通过：第{index['historical_point_pass_round']}轮，候选{index['historical_point_pass_candidates']}。"
            f"实际比较表{index['latest_actual_selected_intent_mix_comparison']}；简单方案说明{index['latest_simpler_strategy_chinese_explanation']}。"
            '第209轮把截至208轮的新308路径筛选转成八套实际组合、三十二条完整账户，全部四场景净夏普与年化门槛通过且保存核对已完成。'
            '禁止把这些真实账户结果降格为只有虚拟组合，但也不能把它们改称独立验证。保持设置并严格按最新next_work接续；已完成的210诊断不得重做，不重跑308路径优化或209账户。')
    if index.get('latest_fixed_point_pass_diagnostic'):
        lines.insert(-1, f"固定稳定性诊断已完成：{index['latest_fixed_point_pass_diagnostic']}。"
            '210读取三个固定方案十二份账本，三项测试、72000条重采样指标独立核对通过，零新候选与账户。'
            '简单两来源主压力20日区块夏普2.5%至97.5%分位数0.5458至1.7773，年化2.9375%至20.9359%；'
            '最大盈利周期约占总净利润42.97%、前五占86.83%。历史点值正确，稳定及独立证据未建立。'
            '不能把重采样比例当未来成功概率，也不能删除年份或周期重新拼出策略；不重复这些诊断。')
    if index.get('latest_post_selection_data_feasibility'):
        lines.insert(-1, f"免费补充资料检查已完成：{index['latest_post_selection_data_feasibility']}。"
            '211对新浪、腾讯、基金官网和上交所各一次正常请求；两行情均覆盖2026-09-11，共同30日开高低收完全一致，新增20日日期齐全，金额量字段非缺失。'
            '官方分红候选覆盖至9月11日，14事件、16公告、无新增事件，未修改全局分红或任何旧输入。'
            '209正式冻结2026-09-13T09:03:57.947046+08:00，新增二十日都在冻结以前，不是严格前瞻；当前零前瞻日、零事前新决定，9月14日才是下一官方交易日。'
            '简单方案主压力8月13日持有61800份，正常下一开盘申请零，但8月14日终点强制卖出61800份；恢复时不能直接从终点清算现金接续。'
            '后续只用保存原始响应和表，不重复请求相同来源；212完成状态以最新索引与下面说明为准。')
    if index.get('latest_post_selection_extension_inputs'):
        lines.insert(-1, f"输入接纳及最小依赖已经完成，不重跑212：{index['latest_post_selection_extension_inputs']}。"
            '最终设置为config/510300_post_selection_extension_inputs_v1_2.json，原v1和v1_1保留了浮点表示与日期类型检查失败；最终无策略或数值改动，55列190080格旧因子精确相同。'
            '候选价格与因素均3476行、至2026-09-11，保存在reports/research/510300_post_selection_extension_inputs_v1；新20日全有效。'
            'dependency_graph.json有27节点，required_account_state_inventory.csv明确22条必要实际账户；纯目标层不要重放账户，109不要拟合GARCH。'
            '两模型文件各141记录、最后2026-08-03。下一项直接按新规则恢复至8月31日，22条固定来源重放、零新拟合；之后才处理9月月首训练。'
            '212没有生成新账户、收益或完整恢复快照。原终点强制清仓不能续用；无需再安排一轮依赖盘点或行情核对。')
    if index.get('latest_continuous_fixed_strategy_result') and not index.get('latest_monthly_incremental_continuation_result'):
        lines.insert(-1,f"213连续账户已经完成，不重跑：{index['latest_continuous_fixed_strategy_result']}。"
            '22条必要账户一次批量重放至2026-08-31，38742账户行、38764决定，旧8月13日及之前正常区间一致。'
            '10项测试通过。真实持有64200份的学习退出来源经JSON状态恢复，账户、决定和最终控制器状态逐值一致；无成交空字段类型已在保存接续汇总中规范化，不改交易。'
            '原整批run在恢复验证类型比较处中止过，最终通过scripts/complete_round213_saved_20260913.py读取22条已保存账户完成汇总，原verify已经通过，禁止因此再重跑整批。'
            '主连续全期基础夏普1.2883747177、年化10.97366829%；压力1.2239092504、10.37370548%，末日均空仓。'
            '8月13日收盘至8月31日收盘12日压力净利润2962.06704元、收益0.77239675%，8月18日卖57500份、8月21日卖余下4300份，是自然退出。'
            '当前已有真实连续结果，不能继续写成数据尚未用于策略。整图增量恢复尚未验证；下一项从保存8月31日checkpoint增量接九月，并仅补原定9月1日的两条月首模型。'
            '九月训练参考需用31原simulate_learned_exit基准，不能拿91带再入场锁定的REARM_RIDGE替代。历史点值通过保持，独立稳定表现未建立。')
    if index.get('latest_monthly_incremental_continuation_result'):
        lines.insert(-1, f"214月首与全图增量已完成，不重跑：{index['latest_monthly_incremental_continuation_result']}。"
            '原定九月两条模型实际拟合完成，20个成熟周期790行、最晚退出7月20日；零新成熟样本，所以分别与八月模型系数相同。旧282条模型记录完整保留，各序列现在142条。'
            '模型fit_origin为模拟9月1日15:05，实际文件生成于9月13日，不是实时九月一日记录。'
            '22条账户均从213八月末checkpoint实际恢复至9月11日，复用38742旧行，只新增198行，最终38940账户行、38962决定。'
            '全图增量已验证：所有旧账本及有效目标前缀一致，主压力同一完整目标单次运行精确相同；没有把全部22条来源重新整段回放。'
            '主全期基础夏普1.2847775261、年化10.90965107%；压力1.2204943135、年化10.31334841%，1624日。九月9日主策略均现金，零收益零新交易，现金日全部计入。'
            '初次比较器在未使用准备期的目标缓存空值上中止，2019年12月31日后有效区间零差异；research/september_saved_continuation_v1.py复用2模型和5完成增量，继续17条完成。原verify已经通过，不重复。'
            '最后所有22来源下一日期9月14日、净申请均零；主账户均无持仓，但两个ENTRY_VINTAGE_EXIT及BASE下行参考仍有持仓，不能全部按现金初始化。'
            '原研究scope、费用、因素与进出场不变，历史点值继续通过，严格前瞻零日、独立稳定证据仍未建立。')
    if index.get('latest_saved_entry_age_attribution'):
        lines.insert(-1, f"入场日龄诊断已完成，不要重跑：{index['latest_saved_entry_age_attribution']}。"
            '只读取201零门槛简单合并的四份保存账本，134个完整周期，未新增账户。入场首日、第2至5日、第6至10日和第11日以后在四场景的合计净利润都为正，'
            '不能据此删除不利日或只挑最终亏损周期，也不能把阶段归因当作延后入场的策略收益。')
    if index.get('latest_saved_zero_target_exit_preflight'):
        lines.insert(-1, f"明确零退出诊断已完成，不要重跑：{index['latest_saved_zero_target_exit_preflight']}。"
            '其中下一开盘价格差只是研究线索，不是账户增量利润；第198轮已实际计算并核对两次零确认账户。'
            '可复用research/event_account_indexed_request_v1.py、research/saved_requested_account_checks_v1.py和198的固定确认逻辑。'
            '保持旧冻结代码完整，新批次用明确包装适配候选身份，不重跑已完成198，也不重新生成已存在的共用入口。')
    saved = tomllib.loads(Path('E:/CodexData/.codex/automations/510300-1-2/automation.toml').read_text(encoding='utf-8'))
    require(saved['id']=='510300-1-2' and saved['kind']=='heartbeat', '原持续任务身份不同')
    if index.get('latest_continuous_fixed_strategy_result'):
        obsolete=('已完成的诊断不要重跑：','批量提速已经落实，','增量收益组合诊断也已经完成，',
            '免费补充资料检查已完成：','输入接纳及最小依赖已经完成，','入场日龄诊断已完成，','明确零退出诊断已完成，')
        lines=[line for line in lines if not line.startswith(obsolete)]
        lines.insert(-1,'旧研究、数据接纳及诊断的定位以最新索引字段为准，不重新读取所有历史。当前输入已核对、连续账户已生成；直接执行下面尚未完成的具体步骤。')
    if index.get('latest_fixed_research_origin'):
        origin = read(ROOT/index['latest_fixed_research_origin'])
        lines.insert(-1, f"研究延续起点已经本地保存：{index['latest_fixed_research_origin']}，实际记录时间{origin['recorded_at']}。"
            '22个完整状态、对应账本与决定、两模型及输入身份已绑定，下一日期9月14日。不要重复创建，后续是否已完成以最新索引为准。'
            '该文件固定研究实验，既不代表实际账户也不启用Paper/Shadow；未来收益尚未观察。')
    if index.get('latest_date_parameterized_continuation_validation'):
        lines.insert(-1, f"215日期入口已完成并核对，不重跑：{index['latest_date_parameterized_continuation_validation']}。"
            'research/fixed_date_continuation_v1.py已支持显式上一目录、输入、新截止日、输出和接纳回执；非月首零拟合，月首缺记录拒绝。'
            '六项新测试通过；22账户先1天再8天，38940账本行、38962决定及所有完整状态与214逐值精确一致。'
            '重复同日直接复用且保存文件未变。两个实际验证段核心6.8608及8.3606秒，包含核对总21.4985秒。'
            '44段198行仅是工程检验，零新候选、零新主绩效记录、零独立日期；当前2720主评价记录保持。'
            '正式下一研究源是214原目录，215/first_day和remaining_days不得替代研究起点；不要重写两个原月模型或重新完整回放22账户。'
            '免费新日线输入接纳的完成状态以最新索引为准；周日没有新完整收盘时不请求网络。')
    if index.get('latest_daily_input_adapter_result'):
        lines.insert(-1, f"216免费输入适配已完成，不重跑：{index['latest_daily_input_adapter_result']}。"
            '十项测试9.29秒通过，两原始响应解析相同，3476行15列价格及63列因素精确复现212；没有重新跑212或215。'
            '实际检查2026-09-13T11:18:37.265062+08:00确认最新完整日与已接纳日均9月11日，网络零、新账户零、严格前瞻零日。'
            '入口research/new_daily_input_adapter_v1.py，设置config/510300_new_daily_input_adapter_runtime_v1.json，输出reports/research/510300_fixed_daily_continuation_v1。'
            '未来有完整新日线会直接获取接纳并续算；当前只保存latest_check.json，无新日线目录或latest_completed.json。成功来源复用，失败保留且至少五分钟后新尝试编号只重试未通过来源。'
            '下一有意义时间9月14日15:05之后。当前工程已完成，切勿再拆新一轮依赖、日期、缓存或时钟审计以维持进展。'
            '216是实际进展，连续外部阻塞计数为零。后续若仍无新完整收盘且没有具体必要工作，按同一外部阻塞逐回合计数；达到三次真正僵局后按工具要求标blocked，保留原任务到时检查。状态文件不是活进程句柄。')
        lines = [line for line in lines if not line.startswith(f"下一第{nxt['candidate_round']}轮状态")]
        lines.insert(-1, f"下一研究步骤已实现，等待外部新日期：{nxt['focus']}。"
            f"现成入口设置{nxt['runtime_settings']}；规则{nxt['source']}。不到新完整收盘时不发网络请求、不重复旧计算、不新建研究轮次。")
    prompt = '\n\n'.join(lines).strip()
    update = {'id':saved['id'], 'mode':'update', 'kind':saved['kind'], 'name':saved['name'], 'prompt':prompt,
        'status':saved['status'], 'rrule':saved['rrule'], 'targetThreadId':saved['target_thread_id']}
    if 'notification_policy' in saved:
        update['notificationPolicy'] = saved['notification_policy']
    prompt_path = ROOT/'docs'/f'510300_RESEARCH_HEARTBEAT_THROUGH{args.round}_{args.stamp}.md'
    args_path = ROOT/'reports/research'/f'510300_heartbeat_update_through{args.round}_args.json'
    require(not prompt_path.exists() and not args_path.exists(), '本轮参数已生成，请直接使用已有参数')
    with prompt_path.open('x', encoding='utf-8') as stream:
        stream.write(prompt)
    write_json(args_path, update, exclusive=True)
    print(json.dumps({'完成轮次':args.round, '下一轮':nxt['candidate_round'], '参数':str(args_path.relative_to(ROOT)), '字数':len(prompt)}, ensure_ascii=False))


if __name__=='__main__':
    main()
