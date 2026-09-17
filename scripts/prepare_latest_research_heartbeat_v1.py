"""从已完成索引生成简短接续提示，减少每轮重复编写和过时信息。"""
import argparse
import json
import re
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='保存当前完整研究进度的原持续任务更新参数。')
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--stamp', required=True)
    args = parser.parse_args()
    require(re.fullmatch(r'\d{8}', args.stamp) is not None, '日期标识必须为八位数字')
    index = json.loads((ROOT/'reports/research/510300_sharpe_1_2_latest_research.json').read_text(encoding='utf-8'))
    latest = index['latest_completed_round']
    require(latest['round'] == args.round and not index['goal_achieved'] and not index['running_studies'], '最新完成研究或目标状态不同')
    result_path = ROOT/latest['result']
    out = result_path.parent
    result = json.loads(result_path.read_text(encoding='utf-8'))
    cfg_path = ROOT/'config'/f"{latest['study'].lower()}.json"
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    verification = json.loads((out/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    acceptance = json.loads((out/'acceptance_outcome.json').read_text(encoding='utf-8'))
    tests = json.loads((out/'tests_receipt.json').read_text(encoding='utf-8'))
    require(cfg['study_id'] == result['study_id'] == latest['study'] and cfg['round'] == args.round, '研究设置与结果身份不同')
    require(not acceptance['goal_achieved'] and tests['exit_code'] == 0, '失败关闭或必要测试状态不同')
    next_work = index['next_work']
    require(next_work['candidate_round'] == args.round+1 and not next_work['registered'], '下一项准备状态不同')
    next_path = ROOT/next_work['source']
    require(next_path.is_file(), '下一项完整事前规则缺失')
    template = (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH164_20260910.md').read_text(encoding='utf-8')
    authority = template.split('\n\n本回合为PROGRESS：', 1)[0]
    authority = f"当前用户目标：{index['goal']}。只按最新双门槛验收；夏普单独通过不能标记目标完成。\n\n" + authority
    common = template.split('\n\n共有行情', 1)[1].split('\n\n交付使用', 1)[0]
    best = index['current_best_four_scenario_comparison_candidate']
    deliveries = [delivery for delivery in index['deliveries'] if args.round in delivery.get('rounds', [])]
    require(len(deliveries) == 1, '当前研究交付不是唯一')
    main_document = Path(deliveries[0]['main_document'])
    require(main_document.is_file(), '当前研究中文交付缺失')
    measured = []
    for key, period in [('all_metrics', '主历史'), ('earlier_diagnostics', '较早历史')]:
        for row in result[key]:
            if row['model'] in cfg['candidate_models']:
                measured.append(f"{period}、{row['cost']}、{row['model']}：净夏普{row['net_sharpe']}，年化{row['annualized_return']}，最大回撤{row['max_drawdown']}。")
    lines = [authority, '',
        f"本回合为PROGRESS：第{args.round}轮已经完成必要测试、冻结、完整账户、独立保存结果核对、失败关闭和中文交付。权威索引reports/research/510300_sharpe_1_2_latest_research.json，goal_achieved=false、running_studies为空，没有运行进程。THROUGH{args.round-1}及更早进度过时，禁止重跑已完成研究的prepare/freeze/run/verify/finalize或旧诊断。只读最新索引及具体当前文件，接续未完成处，不重复遍历长历史。Windows中文，PYTHONIOENCODING=utf-8，.venv\\Scripts\\python.exe，用模块入口。", '',
        f"当前第{args.round}轮：{latest['title']}。研究{latest['study']}，主模型{cfg['primary']}。设置{cfg_path.relative_to(ROOT)}，结果{result_path.relative_to(ROOT)}，完整规则{cfg['rules']}。必要测试{tests['passed']}项、{tests['seconds']}秒；核心计算{result['run_seconds']}秒，不含开发、测试、结果核对和交付。",
        *measured, acceptance['decision'], '',
        f"独立核对已经完成，回执{(out/'saved_verification_receipt.json').relative_to(ROOT)}，核对时间{verification['verified_at']}，状态{verification['status']}。具体实际周期、收益归因、费用、覆盖和未知状态读取同目录saved_account_checks.csv、saved_actual_cycles.csv、saved_comparison_differences.csv、account_coverage.csv、target_coverage.csv；无需重新运行账户或再次核对。当前交付{main_document.relative_to(ROOT)}，包含全部中文因子与进出场规则。", '',
        f"均衡比较候选{best['study']}、{best['model']}，最后比较{best['last_compared_completed_round']}。主基础/压力夏普{best['base_main_sharpe']}/{best['stress_main_sharpe']}，较早{best['base_earlier_sharpe']}/{best['stress_earlier_sharpe']}。四整段复合超额为正仍不等于逐年稳定超额；较早不足1.2，独立验证{best['independent_validation']}，目标尚未完成。原143的20/60日区块比较区间跨零、较早前三盈利周期贡献集中的原诊断保持，不重跑、不称为已证明高夏普。来源{best['source_result']}。", '',
        f"下一第{next_work['candidate_round']}轮完整事前规则：{next_path.relative_to(ROOT)}。状态{next_work['status']}，重点：{next_work['focus']}。当前只准备事前方案，尚未登记、实现、测试、冻结或计算新账户。开始前先查对应具体研究文件是否有后续进展，接续而非覆盖。计划{next_work['planned_settings']}套设置、{next_work['planned_new_accounts']}新账户、{next_work['planned_new_model_fits']}新模型、{next_work['planned_new_reference_accounts']}新参考，外部来源需求{next_work['external_data_required']}。完整参数和处理边界以该事前文档为准，不自行变更。", '',
        '下一166为第143与165保存收盘目标各半：组合账户独立，不能平均夏普或净值代替实际成交。父模型TREND_NOISE_REFERENCE_BLEND与RETURN_RUNS_STATE，目录trend_noise_reference_blend_v1及return_runs_state_v1；建议新模型RUNS_REFERENCE_BLEND、研究510300_RUNS_REFERENCE_BLEND_V1，文件runs_reference_blend_inputs_v1.py及runs_reference_blend_v1.py。仅在两父目标已知时取算术平均，任何未知保留未知，不将空闲一半转给另一来源。保存对照143/165/BH，四新账户、零重训练和新增参考。两费用使用各自父目标，原143内部模型及参考独立连续保留，不接收新账户反馈。第166完整中文V1文档需纳入143与165全部原规则。' if next_work['candidate_round'] == 166 else '新方法不得修改已经关闭方法的窗口、方向、费用、阈值或父来源绑定救回；新组合必须作为单独固定规则和真实账户评价，原结果仍保留。', '',
        '框架可复用research/saved_target_batch_runner_v1.py、saved_parent_target_alignment_v1.py、event_clock_account_v1.py。前两者可读已保存父收盘决定并核对身份日期时钟费用；不会重新生成父账户。已有持仓的正目标偏差不足.1保持，否则调中心；空仓正目标直接尝试整百份进入；已知零及终点开盘优先清仓。未知保持实有份额，不能当现金。受阻下一收盘按最新目标决定。独立账户核对可用saved_target_account_checks_v1.py并写本轮源因素/父目标的独立检查，不重复旧核对。', '',
        '共有行情'+common, '',
        '通过必要测试、完整中文规则和一次冻结后，实际跑预定新账户。完成后先candidate_outcomes及saved_verification_receipt，再fast_round_delivery_v1交付和更新总索引、当前比较候选的最后比较轮次，并准备一个确有区别的下一有限方法。可运行scripts/prepare_latest_research_heartbeat_v1.py --round 当前完成轮次 --stamp 当天八位日期，保存简短进度及原任务更新参数；每轮只生成一次，已有参数就直接使用，不重复生成。然后通过automation_update更新原510300-1-2完整字段并回读TOML，不直接编辑、不创建重复任务。', '',
        '自动目标此前实际为usageLimited，继续时按实际工具状态判断；不达标不标complete，有明确工作不标blocked，不消耗重置或购买额度。保持原ACTIVE与通知偏好，无变化安静，只在实质进展、完成、失败或需行动时通知。继续推进。']
    prompt = '\n'.join(lines).strip()
    prompt_path = ROOT/'docs'/f'510300_RESEARCH_HEARTBEAT_THROUGH{args.round}_{args.stamp}.md'
    with prompt_path.open('x', encoding='utf-8') as stream:
        stream.write(prompt)
    saved = tomllib.loads(Path('E:/CodexData/.codex/automations/510300-1-2/automation.toml').read_text(encoding='utf-8'))
    require(saved['id'] == '510300-1-2' and saved['kind'] == 'heartbeat', '原持续任务身份改变')
    update = {'id': saved['id'], 'mode': 'update', 'kind': saved['kind'], 'name': saved['name'], 'prompt': prompt,
        'status': saved['status'], 'rrule': saved['rrule'], 'targetThreadId': saved['target_thread_id']}
    if 'notification_policy' in saved:
        update['notificationPolicy'] = saved['notification_policy']
    args_path = ROOT/'reports/research'/f'510300_heartbeat_update_through{args.round}_args.json'
    write_json(args_path, update, exclusive=True)
    print(json.dumps({'完成轮次': args.round, '字数': len(prompt), '更新参数': str(args_path.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
