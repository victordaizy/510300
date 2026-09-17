"""保持原持续任务，将下一步切换至无需训练的持续回撤风险方法。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT/'reports/research/510300_sharpe_1_2_latest_research.json').read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 160 and not index['goal_achieved'] and not index['running_studies'], '160完成状态不同')
    previous = (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH159_20260910.md').read_text(encoding='utf-8')
    authority = previous.split('\n\n本回合为PROGRESS：', 1)[0]
    best = previous.split('\n\n均衡比较候选仍143', 1)[1].split('\n\n159 study=', 1)[0].replace('最后比较159', '最后比较160')
    common = previous.split('\n\n共有行情', 1)[1].split('\n\n交付复用fast_round_delivery', 1)[0]
    prompt = authority+'\n\n'+'''本回合为PROGRESS：159和160都完成必要测试、冻结、真实账户、独立数值与净值核对、失败关闭和中文交付。权威reports/research/510300_sharpe_1_2_latest_research.json现在160轮、444设置、460已评价来源版本、465登记含5旧未运行、2034主绩效记录。goal_achieved=false，running_studies空，没有未取回进程。禁止重跑160及更早prepare/freeze/run/verify/finalize。THROUGH159及更早进度过时。下一161已有完整可实施方案，不是阻塞。Windows中文，PYTHONIOENCODING=utf-8，.venv\\Scripts\\python.exe，用模块入口，只读具体必要文件。

均衡比较候选仍143'''+best+'\n\n'+'''160 study510300_SEQUENTIAL_RETURN_STATE_V1，PRIMARY=SEQUENTIAL_RETURN_STATE；research/sequential_return_state_inputs_v1.py、sequential_return_state_v1.py、config/510300_sequential_return_state_v1.json、reports/research/510300_sequential_return_state_v1。六测试2.88秒、零新模型、4新账户、零参考。核心0.9788561000023037秒，不含开发测试核对交付。完整日历r简单含分红收益除前20日sampleSD（不含今天，>1e−12），不减均值不截断；上max(0,前上+z−.5)，下max(0,前下−z−.5)，任一≥5方向up/down，两累积同时0，重复同方向也reset，没触发延续。初始和缺失后从非上涨与0开始，缺失当天NO_VIEW。up target=min(1,.1/当期20日年化SD)，down0；每CLOSE目标nextOPEN按ownNAV交易，.1绝对调仓带，明确0/终点优先，无每笔锁定。不同于旧60持仓内固定60基准单侧附加退出。

160主基础/压力S.5789753002187783/.5635748531025716，CAGR.04253692392288474/.04130442745431376，MDD−.09122572587439268/−.09350630564439123；较早S.7699075491915827/.7481863903757212，CAGR.06030684294991814/.05845681637601774，MDD−.1358859379802058/−.13697632639419313。四CAGR超BH0.627945/0.521948/2.134561/1.975654个百分点，但四S<1.2/143。相对143主CAGR更低，较早更高，风险回撤更大，不能称稳定或独立通过。

160每账户只有3实际完整持仓周期，主各664持仓CLOSE、28买30卖58成交，较早642持仓21买20卖41成交。主665positive/939zero目标、8up/5down触发；较早642positive/577zero、7up/4down，重复同向触发不是新交易。无未知目标、无受阻请求、终点全清；主正目标比持仓多1是terminalOPEN优先。主净63597.86632/61539.22564元，费用2245.13368/4126.97436元；较早净68617.07694/66264.52748、费用1635.32306/3122.87252元。对143主终值少2804.20632/1210.60856、较早多9802.60408/4904.66648。主3段2020-07-03→2021-07-28 261closes、2024-09-25→2025-04-08 126、2025-06-26→2026-08-14terminal 277；较早3段159/363/120closes，结果来自少量长区间。

160 scripts/verify_round160_20260910.py已完成：原始价格分红还原total_simple和current/lagged20SD，独立循环3435日双证据，4accounts5646决定12完整周期198真实开盘fill通过。回执2026-09-10T02:00:26.629248+08:00，reviewer SHA a6859ddc0a7976af74fafbf1619a0efcc8ed0173347eea7c6c0367ba7dc2c5ec，shared checker2602e9268fc97f31f5af332ae45816947a10614b852e0d51c5900367db20dffc。finalize_round160_20260910完成，CLOSED_SEQUENTIAL_RETURN_STATE_FULL_GOAL_NOT_MET，交付deliverables/510300双向收益状态_第160轮_20260910/双向收益状态_结果及全部中文规则.md，包含全部12实际周期及全中文因素进出场；全部方向触发记录.csv已保存。不要再运行这些脚本。

159也完整关闭，P=reports/research/510300_marginal_monotone_exit_v1，主S−.056771413310628197/−.13048959053466713，较早.6420133294714869/.6205724010735365，4.2304263sec。25models200curves3158knots全中文已交付deliverables/510300八曲线退出_第159轮_20260910/八曲线退出_结果及全部中文规则.md。主29cycles其中22只持有2日，不能继续换同退出拟合器；所有≤159旧失败不重复。

下一161直接实施docs/510300_DRAWDOWN_DEPTH_RISK_NEXT_20260910.md。建议study510300_DRAWDOWN_DEPTH_RISK_V1，PRIMARY=DRAWDOWN_DEPTH_RISK，research/drawdown_depth_risk_inputs_v1.py、drawdown_depth_risk_v1.py，P=reports/research/510300_drawdown_depth_risk_v1。目前只有事前MD，没有实现/测试/config/run；接续前查这些具体文件有无新进展。1设置4新真实账户，0模型/参考/外部data，3保存controls143/131/BH，每段2新+6保存=8指标。可静态复用160或155入口及saved_target_batch_runner_v1，不要重跑它们。

161全规则：财富wealth按rawclose+div/previousclose连乘，初始1；当前wealth/含今天120均值−1严格正允许持有，明确≤0target0。风险1=current20simple sampleSD*sqrt242，普通cap=min(1,.1/vol20)，positive且vol未知或≤0→NO_VIEW。风险2=每个完整60收盘财富窗口内从第一天顺序维护prefixmax，每日dd=wealth/prefixmax−1，UI60=sqrt(mean(60个dd^2))，第一天dd0也计入。UI用小数不年化，UI0→drawdowncap1，否则min(1,.04/UI)。positive且两风险完整→target=min(普通cap,drawdowncap)；趋势未知target未知；已知nonpositive0不依赖其他缺失。窗口缺失不跳过、不用未来峰值或滚动60高点再滚动60均方（后者需要119天且是不同算法），不用无限历史峰值。每60窗口重启prefixmax，高点出窗允许UI减少属既定定义。不要把Martin Ratio替代原净Sharpe目标。

161出处提出者PeterG.Martin https://www.tangotools.com/ui/ui.htm已查，定义跌幅平方平均根与先后峰值；60window/.04riskbudget/120trend和普通cap取小值是本轮事先固定设计，不是来源证实有效。旧94是actual账户最高权益和floor保险、两个父策略；旧downside是日负收益波动。该局部市场回撤度量不改它们也不改160阈值救回。有界检索未见现成Ulcer本交易结构，不宣称全历史数学唯一证明。

161执行复用event_clock_account_v1。每天CLOSE ownNAV和rawclose算目标整百份、nextOPEN实际成；空仓正target直接算份不受.1带限制，持仓positive且偏差<.1保持，≥.1调至center，明确0或terminalOPEN优先全退。正小target经整手可能0。无旧D60/学习/stop/take/maxdays/重入等待或每笔意图锁。受阻nextclose按最新known重新请求，unknown保留份，不能unknown→cash。保存所有因素、cap与target，每个fees完全同target但ownaccountfills不同。

161必要测试：手算UI与prefixmax顺序、回撤恢复和窗口边界、财富同比例缩放、分红校正、缺失zero、futureprefix、两个cap取min、空仓和带、真实nextOPEN进出/分红。先test/full中文V1规则再freeze一次run4accounts。独立verification参考verify_round160_20260910和saved_target_account_checks_v1，从raw+div重建财富/vol120trend，用独立逐window逐day循环算UI和target，再校验account。不必额外安全审计/昂贵重复测试/ZIP，不训练无用模型。

共有行情'''+common+'\n\n'+'''交付复用fast_round_delivery_v1，candidate_outcomes+saved_verification_receipt完成再deliver；161完成预期445设置461已评价466含旧未运行2042主指标。保持143比较身份并更新最后比较，下一方案不能随便救回固定失败方法。通过automation_update更新原510300-1-2并读TOML确认，保留name/kind/status/rrule/target与通知偏好，不直接写TOML不重复创建。最新get_goal实际status usageLimited，tokensUsed17764782；未达目标不标complete，有下一方法不标blocked，不消耗重置信用或购买额度。持续任务ACTIVE。无变化安静，实质进展/完成/失败/需行动才通知。继续推进。'''
    path = ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH160_20260910.md'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(prompt+'\n')
    saved = tomllib.loads(Path('E:/CodexData/.codex/automations/510300-1-2/automation.toml').read_text(encoding='utf-8'))
    require(saved['id'] == '510300-1-2' and saved['kind'] == 'heartbeat', '原持续任务身份改变')
    args = {'id': saved['id'], 'mode': 'update', 'kind': saved['kind'], 'name': saved['name'], 'prompt': prompt,
        'status': saved['status'], 'rrule': saved['rrule'], 'targetThreadId': saved['target_thread_id']}
    if 'notification_policy' in saved:
        args['notificationPolicy'] = saved['notification_policy']
    write_json(ROOT/'reports/research/510300_heartbeat_update_through160_args.json', args, exclusive=True)
    print(json.dumps({'字数': len(prompt), '状态': saved['status'], '完成轮次': 160}, ensure_ascii=False))


if __name__ == '__main__':
    main()
