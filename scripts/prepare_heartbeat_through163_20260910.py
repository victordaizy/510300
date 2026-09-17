"""保存价量规则完成状态和下一项完整方法，更新原任务参数。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT/'reports/research/510300_sharpe_1_2_latest_research.json').read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 163 and not index['goal_achieved'] and not index['running_studies'], '163完成状态不同')
    previous = (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH162_20260910.md').read_text(encoding='utf-8')
    authority = previous.split('\n\n本回合为PROGRESS：', 1)[0]
    best = previous.split('\n\n均衡比较候选仍143', 1)[1].split('\n\n162 study', 1)[0].replace('最后比较162', '最后比较163')
    common = previous.split('\n\n共有行情', 1)[1].split('\n\n交付复用', 1)[0]
    prompt = authority+'\n\n'+'''本回合为PROGRESS：163已完整实现、六测试3.82秒通过、一次冻结、零训练四新账户、独立原始价量因素及完整资金核对、失败关闭及中文交付。没有运行进程。权威reports/research/510300_sharpe_1_2_latest_research.json为163轮、447设置、463已评价来源版本、468登记含5旧未运行、2058主指标，goal_achieved=false、running_studies空。禁止重跑163及更早prepare/freeze/run/verify/finalize或旧诊断。THROUGH162及更早进度过时。Windows中文，PYTHONIOENCODING=utf-8，.venv\\Scripts\\python.exe，用模块入口；只读具体当前文件，避免重复长历史。

均衡比较候选仍143'''+best+'''

163 study510300_PRICE_VOLUME_COHERENCE_V1，PRIMARY PRICE_VOLUME_COHERENCE，research/price_volume_coherence_inputs_v1.py、price_volume_coherence_v1.py，config/510300_price_volume_coherence_v1.json，reports/research/510300_price_volume_coherence_v1，完整规则docs/510300_PRICE_VOLUME_COHERENCE_V1.md。每日r=log((close+div)/previousclose)、q=log(volume/previousvolume)，完整20对Pearson同期correlation、20日r和momentum；corr>.2且momentum>0连续2准入，corr<−.2连续2或momentum<=0连续2退出，两个退出分别计数，其他完整延续方向。常数/缺失相关unknown并重启内部计数，实际未知保留份额。允许方向target min(1,.1/vol20)，不允许0。完整市场日历状态、.1已有持仓调仓带、nextOPEN，无父目标或训练。

163净夏普主.16737675716228376/.1028149811703744，较早.571839999796138/.4760513958629185；CAGR主.009071657722498393/.004688815092744065，较早.03560896666185109/.029193136124872687；MDD主−.15672219947240776/−.17273847434417597，较早−.09940565647602505/−.1051837320169606。四S低1.2/143，四CAGR低BH；较早S高BH不可混说全部指标低BH。四账户核心1.4713710999931209秒，不含开发测试核对交付。主各30cycles519持仓99fills，早各28cycles471持仓BASE85/STRESS86fills，费用/ownNAV/整手使实际成交数可不同，目标与因素相同。无未知或受阻请求，终点已清。主519正1085零，早472正747零；早多一个正目标来自terminalOPEN。主cycles中位13closes、早14.5；主仅2cycle、早仅1cycle≤2，不能把全部失败说成两日反复。

163四gross价格+div19784.4/19942.6/45490.6/44539.8元，fees7447.50892/13644.40604/6943.5012/13344.41456，net12336.89108/6298.19396/38547.0988/31195.38544。较143终值少54065.18156/56451.64024/20267.37406/30164.47556；四价格损益也低，不能把不足全归因费用。163独立scripts/verify_round163_20260910.py通过3436完整相关窗口（误差4.44e−16）、5646收盘决定、116完整cycles、369真实原始开盘fills与所有资金分红费用。回执2026-09-10T09:26:57.071416+08:00，reviewerSHAe63ac0dd613f36012f9f062ad03968175b5e87fc7daec084eff9930f996e77aa，共享saved_target_account_checks_v1.py SHA2602e9268fc97f31f5af332ae45816947a10614b852e0d51c5900367db20dffc。finalize_round163_20260910完成，关闭CLOSED_PRICE_VOLUME_COHERENCE_FULL_GOAL_NOT_MET，交付deliverables/510300价量相关_第163轮_20260910/价量相关_结果及全部中文规则.md。全部cycles在同目录saved_actual_cycles.csv。

162也已关闭交付，净夏普主.25256/.23184、较早.44409/.42170，四CAGR低BH，独立3396排序窗口误差0及4账户通过；路径session_signed_rank_v1。160双向累计收益和161回撤双上限及更早全已关闭，不重跑。

下一164读docs/510300_ORDINAL_ENTROPY_NEXT_20260910.md完整事前规则。目前只有MD，没有实现测试config或run。开始先查research/ordinal_entropy_inputs_v1.py、ordinal_entropy_v1.py、config/510300_ordinal_entropy_v1.json、reports/research/510300_ordinal_entropy_v1以及scripts/prepare_round164_20260910.py是否有新进展，接续未完处不覆盖。建议study510300_ORDINAL_ENTROPY_V1，PRIMARY ORDINAL_ENTROPY，1setting4newaccounts，0训练/新增参考/外部data；3保存对照143/131/BH，每段8metrics。

164含今天60完整wealth，每连续3值按低到高排列时间位置，精确ties较早日期排先，共58三日模式。六列频数对应012/021/102/120/201/210，除58得p，entropy=−sum(p*logp)/log6，p0项0，误差裁[0,1]不round，全部常数模式012且entropy0。momentum60=log(wealth_t/wealth_t−60)，两端共61财富值必须完整。entropy<.9且momentum>0联合连续2→允许；entropy>.95连续2或momentum<=0连续2→不允许，三个条件分别计数，不符清零。完整市场日历从0起步不在入场/评价边界reset；熵或动量未知则当日方向targetunknown，内部方向及计数清0，完整恢复从0重启。允许且vol20>0target=min(1,.1/vol20)，不允许target0；允许但风险unknown→targetunknown不改方向。未知实际保留shares，不能当cash。

164公式已查Bandt/Pompe原论文第二节及三阶示例：https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevLett.88.174102/fulltext；相邻排序頻率熵定义，归一化除log6。论文没证明ETF策略获利，时间先后tie处理及.9/.95交易门槛是本轮固定选择。已查目标research/config/docs没有permutation entropy/排列熵/ordinal pattern旧策略命中；旧123bayesian_run_length有命中但非此方法且已关闭。164不是162日内隔夜差值幅度rank，也不改163价量阈值救回。三日順序看時間关系与分布，不设新来源。必要6类测试手算模式/熵、递增递减/平局/常数、60与61完整缺失边界、退出分别确认、futureprefix、普通风险与真实进出分红；独立排序用另一枚举实现及全资金核对。

164复用saved_target_batch_runner_v1、event_clock_account_v1，空仓positive直接买整百份，已有positive偏差<.1带保持，否则调center，known0及terminalOPEN优先全部退。无学习/止盈止损/持有期限/入场锁/额外重入等待/每笔退出锁；受阻nextclose最新目标，unknown保留实股。冻结一次run4，保存对照不重算。163入口/测试/verify可静态复用简短框架，独立检查原价/ownNAV/费用/分红/完整模式，不重跑旧账户。

共有行情'''+common+'''

交付复用fast_round_delivery_v1，先candidate_outcomes及saved_verification_receipt，再中文结果及全部规则。164完成预计448设置464已评价469登记含旧未运行2066主指标；更新143最后比较并准备下一个确有区别的有限候选。通过automation_update更新原510300-1-2完整字段并回读TOML，不直接改不重复建任务。工具可能去掉prompt尾换行；首尾空白外全文一致及其他字段一致应单独说明，勿重复更新。goal在本轮实际再次读取仍usageLimited、tokensUsed17764782；不达标不标complete，有可做工作不标blocked，不消耗重置或购买额度。保持原ACTIVE和通知偏好，无变化安静，仅实质进展、完成、失败或需行动时通知。继续推进。
'''
    prompt = prompt.replace('排序頻率', '排序频率').replace('三日順序看時間', '三日顺序看时间').strip()
    with (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH163_20260910.md').open('x', encoding='utf-8') as stream:
        stream.write(prompt)
    saved = tomllib.loads(Path('E:/CodexData/.codex/automations/510300-1-2/automation.toml').read_text(encoding='utf-8'))
    require(saved['id'] == '510300-1-2' and saved['kind'] == 'heartbeat', '原持续任务身份改变')
    args = {'id': saved['id'], 'mode': 'update', 'kind': saved['kind'], 'name': saved['name'], 'prompt': prompt,
        'status': saved['status'], 'rrule': saved['rrule'], 'targetThreadId': saved['target_thread_id']}
    if 'notification_policy' in saved:
        args['notificationPolicy'] = saved['notification_policy']
    write_json(ROOT/'reports/research/510300_heartbeat_update_through163_args.json', args, exclusive=True)
    print(json.dumps({'字数': len(prompt), '原状态': saved['status'], '完成轮次': 163}, ensure_ascii=False))


if __name__ == '__main__':
    main()
