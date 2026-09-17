"""精简继续提示，直接链接下一项完整规则，保留必要身份和进度。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT/'reports/research/510300_sharpe_1_2_latest_research.json').read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 161 and not index['goal_achieved'] and not index['running_studies'], '161完成状态不同')
    previous = (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH160_20260910.md').read_text(encoding='utf-8')
    authority = previous.split('\n\n本回合为PROGRESS：', 1)[0]
    best = previous.split('\n\n均衡比较候选仍143', 1)[1].split('\n\n160 study', 1)[0].replace('最后比较160', '最后比较161')
    common = previous.split('\n\n共有行情', 1)[1].split('\n\n交付复用fast_round_delivery', 1)[0]
    out = ROOT/'reports/research/510300_drawdown_depth_risk_v1'
    result = json.loads((out/'result.json').read_text(encoding='utf-8'))
    verification = json.loads((out/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    measured = []
    for period, chinese in [('all_metrics', '主'), ('earlier_diagnostics', '较早')]:
        for row in result[period]:
            if row['model'] == 'DRAWDOWN_DEPTH_RISK':
                measured.append(f"{chinese}{'基础' if row['cost']=='BASE' else '压力'}：净夏普{row['net_sharpe']:.15g}、年化{row['annualized_return']:.15g}、回撤{row['max_drawdown']:.15g}。")
    prompt = authority+'\n\n'+'''本回合为PROGRESS：161已完整实现、六测试5.79秒通过、一次冻结、零模型四新账户、独立逐窗口和完整资金核对、失败关闭及中文交付。没有运行进程。权威reports/research/510300_sharpe_1_2_latest_research.json现在161轮、445设置、461已评价来源版本、466登记含5旧未运行、2042主指标，goal_achieved=false、running_studies空。禁止重跑161及更早prepare/freeze/run/verify/finalize或旧诊断；THROUGH160及更早进度过时。下一162有完整可实施方法，不是阻塞。Windows中文，PYTHONIOENCODING=utf-8，.venv\\Scripts\\python.exe，用模块入口。为提速只读当前具体文件，不重复遍历旧长报告。

均衡比较候选仍143'''+best+'\n\n'+'''161 study510300_DRAWDOWN_DEPTH_RISK_V1，PRIMARY=DRAWDOWN_DEPTH_RISK，research/drawdown_depth_risk_inputs_v1.py、drawdown_depth_risk_v1.py，config/510300_drawdown_depth_risk_v1.json，reports/research/510300_drawdown_depth_risk_v1。完整规则docs/510300_DRAWDOWN_DEPTH_RISK_V1.md。wealth/含今天120均值>1准入；cap1=min(1,.1/vol20)，cap2=min(1,.04/UI60)，UI0cap2=1；正趋势取min，已知非正0，其他不完整NO_VIEW。每60完整wealth窗口独立prefixmax，dd=wealth/prefixmax−1，UI=sqrt平均60个dd平方（第一天0），不是每日日前60高点再滚60均方，也非actualaccount最高点。自主每日目标与.1调仓带真实nextOPEN。

161四场景结果：'''+''.join(measured)+f'''核心计算{result['run_seconds']:.15g}秒，不含开发测试核对交付。四S低1.2/143，主净亏，较早更高CAGR不抵消主失败。主各38cycles862持仓135成交，15cycles只持有至多2CLOSE；较早20cycles730持仓92成交，9cycles≤2CLOSE。UI在主54/较早58判断日更严格，仅说明改变target，不是新增成交数或单独获利因果。本轮未另建去掉UI的匹配账户，不能把整个损益差归因UI。无未知目标/受阻请求，全部terminalOPEN清仓。

161主gross价格+div7176.6/5704元，fee8630.86646/15811.6344，net−1454.26646/−10107.6344；较早gross96382.9/94527.1，fee6098.05118/11535.9402，net90284.84882/82991.1598。主BASE正微小日算术均值及Sharpe而复利净亏并不矛盾。相对143主终值少67856.3391/72857.4686、较早多31470.37596/21631.2988，但四S更低。关闭CLOSED_DRAWDOWN_DEPTH_RISK_FULL_GOAL_NOT_MET。

161独立scripts/verify_round161_20260910.py已经完成，原始价格分红重建财富/vol/trend，再3397完整窗口逐日prefixmax与UI，两上限target、5646收盘决定116完整cycles454真实fill及完整权益分红费用通过；数值误差UI2.78e−17、trend4.45e−16。回执{verification['verified_at']}，reviewer SHA {verification['reviewer_source_sha256']}。共享saved_target_account_checks_v1.py SHA2602e9268fc97f31f5af332ae45816947a10614b852e0d51c5900367db20dffc。finalize_round161_20260910完成，deliverables/510300持续回撤双上限_第161轮_20260910/持续回撤双上限_结果及全部中文规则.md，saved_short_cycle_summary.json仅保存诊断。

159/160也已完整关闭交付。159八曲线主S−.05677/−.13049，较早.64201/.62057，25模型200曲线3158断点全中文已存。160双向市场累计收益状态主S.57898/.56357，较早.76991/.74819，0.978856秒零训练4账户；各场景仅3完整cycles。文件在各自marginal_monotone_exit_v1、sequential_return_state_v1目录，禁止重跑。

下一162直接读docs/510300_SESSION_SIGNED_RANK_NEXT_20260910.md并实施，目前只有事前MD，没有实现测试config/run。建议study510300_SESSION_SIGNED_RANK_V1、PRIMARY=SESSION_SIGNED_RANK，research/session_signed_rank_inputs_v1.py、session_signed_rank_v1.py，reports/research/510300_session_signed_rank_v1；开始前查这些具体文件是否有新进展，接续而非覆盖。1setting4newaccounts、0训练/参考/外部data，3保存controls143/131/BH，每段2new+6saved=8metrics。

162以每日intraday_log−overnight_log完整60日差，去掉精确0参与排序（0是已知平局不是missing）；按绝对值rank，精确ties平均名次，不舍入近似ties，z=sum(sign(d)*rank)/sqrt(sum(rank²))；全0→knownz0。完整窗口非零数、正负rank和、rank平方和都存。z连续2CLOSE>1.96→方向1；连续2<0→方向0；其他已知维持方向并各自清不符连续计数。初始完整窗口从方向0及两计数0当天更新。窗口缺失当日NO_VIEW，内部方向计数清0，完整恢复重启，不跨缺失拼接。方向1且vol20>0target=min(1,.1/vol20)，方向0target0，风险缺失positive→NO_VIEW。完整市场日历递推不在实际入场/评价起点reset。已有160非训练递推框架可静态复用，但不是旧CUSUM同一公式/阈值变体。

162官方SciPy Wilcoxon说明已查https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html，仅借绝对差带符号rank与ties定义。本score与无连续修正、去零、tie平均的单侧标准化秩一致，可用scipy单侧asymptotic或独立rankdata核对；不能用双侧返回statistic或pvalue代替score。1.96是固定门槛，不宣称金融iid、对称或95%交易置信。旧D60使用幅度和波动，此处改用排序，旧crosssection多数广度对象不同；不改161UI预算或160触发参数救回。必要合成测试手算tie/zero/allzero/missing、极端值rank影响、2确认与恢复、futureprefix、风控目标、真实nextOPEN与分红。

162共用saved_target_batch_runner_v1、event_clock_account_v1，自主目标无父目标，空仓positive直接算整百份，持仓偏差<.1带保持否则调center，known0/terminalOPEN优先全退。无原D60学习/固定止盈止损期限/重入等待/每笔退出锁。受阻nextclose按最新known决定，unknown保留现有份，不能变cash。原价、ownNAV、fee、lot、T1、div登记除息到账各自核对。先tests及完整中文V1规则，再freeze一次run4。参考161入口与saved_target_account_checks_v1，并独立由raw开收盘分红计算session差和rank/方向/target，不重复run旧账户。

共有行情'''+common+'''

交付使用fast_round_delivery_v1，candidate_outcomes与saved_verification_receipt完成后deliver，162完成预计446设置462已评价467含旧未运行2050主指标，更新143最后比较。沿用原510300-1-2，通过automation_update更新完整字段并回读原TOML，不直接编辑不重复建任务。最新goal实际usageLimited，不达标不标complete，有明确新动作不标blocked，不消耗重置或购买额度。保持原ACTIVE及通知偏好，无变化安静，只在实质进展/完成/失败/需要行动时通知。继续推进。'''
    with (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH161_20260910.md').open('x', encoding='utf-8') as stream:
        stream.write(prompt+'\n')
    saved = tomllib.loads(Path('E:/CodexData/.codex/automations/510300-1-2/automation.toml').read_text(encoding='utf-8'))
    require(saved['id'] == '510300-1-2' and saved['kind'] == 'heartbeat', '原持续任务身份改变')
    args = {'id': saved['id'], 'mode': 'update', 'kind': saved['kind'], 'name': saved['name'], 'prompt': prompt,
        'status': saved['status'], 'rrule': saved['rrule'], 'targetThreadId': saved['target_thread_id']}
    if 'notification_policy' in saved:
        args['notificationPolicy'] = saved['notification_policy']
    write_json(ROOT/'reports/research/510300_heartbeat_update_through161_args.json', args, exclusive=True)
    print(json.dumps({'字数': len(prompt), '原状态': saved['status'], '完成轮次': 161}, ensure_ascii=False))


if __name__ == '__main__':
    main()
