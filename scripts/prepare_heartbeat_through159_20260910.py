"""将继续任务指向已交付159和无需新训练的160。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json').read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 159 and not index['goal_achieved'], '159交付状态不同')
    previous = (ROOT / 'docs/510300_RESEARCH_HEARTBEAT_THROUGH158_20260910.md').read_text(encoding='utf-8')
    authority = previous.split('\n\n本回合为PROGRESS：', 1)[0]
    best = previous.split('\n\n均衡比较候选仍143', 1)[1].split('\n\n158 study=', 1)[0].replace('最后比较158', '最后比较159')
    common = previous.split('\n\n共有行情', 1)[1].split('\n\n统一fast_round_delivery', 1)[0]
    prompt = authority + '\n\n' + '''本回合为PROGRESS：159已实现、9测试5.48秒通过、一次冻结、25复合模型和四账户、独立保序及全账户核对、失败关闭和中文全曲线交付。权威reports/research/510300_sharpe_1_2_latest_research.json为159轮、443设置、459已评价来源版本、464登记含5旧未运行、2026主绩效记录，goal_achieved=false，running_studies空。不得重跑159及更早prepare/freeze/run/verify/finalize，THROUGH158及更早进度过时。下一160有完整可实施方案，不是阻塞。Windows中文，PYTHONIOENCODING=utf-8，.venv\\Scripts\\python.exe，用模块入口，只读必要当前文件。

均衡比较候选仍143''' + best + '\n\n' + '''159 study=510300_MARGINAL_MONOTONE_EXIT_V1，PRIMARY=MARGINAL_MONOTONE_EXIT；research/marginal_monotone_exit_inputs_v1.py、marginal_monotone_exit_v1.py、config/510300_marginal_monotone_exit_v1.json、reports/research/510300_marginal_monotone_exit_v1。141月度114可用、25实际输入各拟合一次89后来复用，全成功；200曲线记录=175加权保序求解+25恒定进入类别。原114完整8因素绝对状态全局加权标准化clip5，成熟协方差符号固定单调方向，单因素weighted PAVA后线性插值、两端常数、8曲线各1/8，入场首CLOSE固定全模型直到实际退出。核心训练+四账户4.230426300026011秒，不含开发测试核对交付。

159主基础/压力S−.056771413310628197/−.13048959053466713，CAGR−.007870813417944655/−.0138067764954516，MDD−.18994193608892496/−.2167933996287477；较早S.6420133294714869/.6205724010735365，CAGR.0778325843687236/.07481455243709655，MDD−.1379164435416316/−.1391665613986051。四S<1.2/128/143，四S与CAGR都低于158；相对143主CAGR低、较早高。主各29周期190状态58成交，26学习退出、22只持有2日；较早9周期340状态18成交，136可预测204入场没成熟模型、零学习退出。四账户完整清仓无受阻请求。

159主价格−9066/−8769元、分红7978.2/7803.9元、费用9117.59776/16641.28724元，净−10205.39776/−17606.38724元；较早净91740.77038/87649.07524元。对128主终值少122758.67482/121573.67156、较早少13326.40704/20607.57336，主费用差只481.07482/869.67156元，主要退步来自路径。scripts/verify_round159_20260910.py用独立SciPy保序解和手工插值，25模型最大误差3.7789216200678766e−14，200曲线3158断点，652实际预测76周期1060状态5646决定152真实成交及权益费用分红通过。回执2026-09-10T01:49:22.585186+08:00，reviewer SHA b8bb6227ed407dbfd56f9ab805e1a31908efb9ff64480bb33da28a887a018446。saved_all_factor_curve_parameters.csv、saved_all_curve_knots.csv、全部已拟合模型参数.md全部已保存。finalize_round159_20260910完成，CLOSED_MARGINAL_MONOTONE_EXIT_FULL_GOAL_NOT_MET，deliverables/510300八曲线退出_第159轮_20260910/八曲线退出_结果及全部中文规则.md包括25组完整参数，不重跑。

下一160直接实施docs/510300_SEQUENTIAL_RETURN_STATE_NEXT_20260910.md。新结构为完整市场日历上双向累计每天含分红简单收益的标准化变化，决定进入和退出，配10%年化波动预算；零模型训练、4新账户、零新参考和外部来源。建议research/sequential_return_state_inputs_v1.py、sequential_return_state_v1.py，study510300_SEQUENTIAL_RETURN_STATE_V1，PRIMARY=SEQUENTIAL_RETURN_STATE。先查这些具体文件和tests/config/RUN_STARTED/result是否已存在，接续未完成动作，避免重写或重跑。

160全部定义见事前MD：当天含分红简单收益，除以仅昨天及以前20日样本SD；不减均值、不截断，sd须>1e−12。上证据=max(0,前上+z−.5)，下证据=max(0,前下−z−.5)。任一≥5令方向为上涨/非上涨，并把两个证据同时归0，从下一天重新累计；同方向重复触发也归0。无触发延续状态，首次完整输入从非上涨和双零起当天更新。缺失或无效尺度当天NO_VIEW，内部状态清零，待完整连续窗口恢复重启，不能跨缺失拼接。完整历史起点顺序计算，不能在评价起点选择状态。已知非上涨target0；上涨target=min(1,.1/当前含今天20日样本SD*sqrt242)，此处分母为整个年化SD，缺失或<=0则NO_VIEW。保存触发前/后双证据、方向、两个波动、目标。

160双向市场状态不同于第60轮实际持仓内固定入场前60日均值SD的单侧累积转弱附加退出，也不同于episodic_alpha_library_v1.py监测完成事件收益而停新进入。旧两方法及失败保持。NIST https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm已查，给上下双递推和常用k.5/h5，金融获利属于研究假设。本轮不试邻近阈值/窗口/基准/重置/预算。

160复用saved_target_batch_runner_v1与event_clock_account_v1，独立目标，无父目标。空仓正目标直接尝试买入，已有持仓才用绝对.1调仓带，小于带保持份额，否则调至中心；明确0优先全退，整手目标可能为0。每close据本账户NAV、rawclose估计整百份，nextOPEN实际成交。无D60/学习/期限/止损止盈、无重入额外等待、退出未成nextclose依最新目标重决策、不锁旧意图。未知保留现有份额，终点OPEN优先。3保存对照143 TREND_NOISE_REFERENCE_BLEND/131 VINTAGE_REFERENCE_RISK/BH，每段2新+6保存=8指标。

160可静态复用median_slope_risk_v1.freeze结构和saved_target_batch_runner_v1；无需拟合、不读旧模型/样本。独立核对参考scripts/verify_round155_20260909.py与saved_target_account_checks_v1，原始价格分红重建simple及20波动、独立循环双证据全部目标，然后真实账户。必要测试手算、阈值、重置、缺失、前20日SD不含当天、futureprefix、目标调仓带、真实次日交易分红。先测试完整中文规则再冻结一次，四账户，失败关闭，无新网格。方向目标方法不能把恒定参考进入/学习旧因素带入。

共有行情''' + common + '\n\n' + '''交付复用fast_round_delivery_v1，核对回执与candidate_outcomes完成后索引加1设置、1来源、8主指标；159以后160预计444设置460已评价465含旧未运行2034主指标。保留当前最佳143并更新最后比较。完成后通过automation_update更新原510300-1-2，保留名称kind状态排程target和通知偏好，读TOML确认，不直接编辑不新建。目标工具最新实际状态usageLimited，不能把进展标complete或消耗重置信用；任务保持ACTIVE。无变化安静，实质进展/完成/失败/需要行动才通知。继续推进。'''
    path = ROOT / 'docs/510300_RESEARCH_HEARTBEAT_THROUGH159_20260910.md'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(prompt+'\n')
    automation = tomllib.loads(Path('E:/CodexData/.codex/automations/510300-1-2/automation.toml').read_text(encoding='utf-8'))
    require(automation['id'] == '510300-1-2' and automation['kind'] == 'heartbeat', '原持续任务身份不同')
    args = {'id': automation['id'], 'mode': 'update', 'kind': automation['kind'], 'name': automation['name'],
        'prompt': prompt, 'status': automation['status'], 'rrule': automation['rrule'], 'targetThreadId': automation['target_thread_id']}
    if 'notification_policy' in automation:
        args['notificationPolicy'] = automation['notification_policy']
    write_json(ROOT / 'reports/research/510300_heartbeat_update_through159_args.json', args, exclusive=True)
    print(json.dumps({'字数': len(prompt), '状态': automation['status'], '轮次': 159}, ensure_ascii=False))


if __name__ == '__main__':
    main()
