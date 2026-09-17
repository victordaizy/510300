"""保存第162轮完成和第163轮已明确规则的原任务继续提示。"""
import json
import tomllib
from pathlib import Path
from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT/'reports/research/510300_sharpe_1_2_latest_research.json').read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round'] == 162 and not index['goal_achieved'] and not index['running_studies'], '162完成状态不同')
    previous = (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH161_20260910.md').read_text(encoding='utf-8')
    authority = previous.split('\n\n本回合为PROGRESS：', 1)[0]
    best = previous.split('\n\n均衡比较候选仍143', 1)[1].split('\n\n161 study', 1)[0].replace('最后比较161', '最后比较162')
    common = previous.split('\n\n共有行情', 1)[1].split('\n\n交付使用', 1)[0]
    prompt = authority+'\n\n'+'''本回合为PROGRESS：162已完整实现、六测试4.26秒通过、一次冻结、零训练四新账户、独立排序与资金核对、失败关闭及完整中文交付。权威reports/research/510300_sharpe_1_2_latest_research.json为162轮、446设置、462已评价来源版本、467登记含5旧未运行、2050主指标，goal_achieved=false、running_studies空。禁止重跑162及更早prepare/freeze/run/verify/finalize或旧诊断。THROUGH161及更早进度已过时。Windows中文，PYTHONIOENCODING=utf-8，.venv\\Scripts\\python.exe，模块入口；只读当前具体文件以提速。

均衡比较候选仍143'''+best+'''

162 study510300_SESSION_SIGNED_RANK_V1，PRIMARY SESSION_SIGNED_RANK，research/session_signed_rank_inputs_v1.py和session_signed_rank_v1.py，config/510300_session_signed_rank_v1.json，reports/research/510300_session_signed_rank_v1，完整中文规则docs/510300_SESSION_SIGNED_RANK_V1.md。完整60日日内log减隔夜log，精确非零绝对差平均rank，正rank和减负rank和除rank平方和根号；全零已知0，缺失unknown重启。score连续2>1.96准入，连续2<0退出，其余延续；允许持有时min(1,.1/vol20)，方向0target0。完整市场日历状态，.1已有仓位调仓带，nextOPEN。

162四场景净夏普主.2525559913930417/.23183820631107022、较早.4440917686276874/.4216966230196898；CAGR主.01405018763713327/.012724397551757683、较早.031723496396441994/.029931648919706726。最大回撤主−.1490121762573676/−.152511492083735、较早−.16100835832064408/−.16246659060789384。四S低1.2/143，四CAGR低BH；较早S高BH不可混说所有指标低BH。核心四账户1.4649602000135928秒，不含开发测试核对交付。主各5cycles498持仓51fills，较早4cycles605持仓41fills，无未知目标/受阻请求，终点全清。四净利润19377.77606/17483.69136/34072.83918/32032.25488，较143少47024.29658/45266.14284/24741.63368/29327.60612。

162 scripts/verify_round162_20260910.py已核对3396完整rank窗口误差0、5646收盘决定、18完整cycles、184原价真实fills及全部资金分红费用。saved_verification_receipt.json时间2026-09-10T09:08:33.199590+08:00，reviewerSHA99761a79f526df88013be47664f2498be39040b148394cd110412aa7e12338ed，共享saved_target_account_checks_v1.py SHA2602e9268fc97f31f5af332ae45816947a10614b852e0d51c5900367db20dffc。finalize_round162_20260910完成，关闭CLOSED_SESSION_SIGNED_RANK_FULL_GOAL_NOT_MET，交付deliverables/510300日内隔夜排序_第162轮_20260910/日内隔夜排序_结果及全部中文规则.md含18cycles。

下一163完整事前方案docs/510300_PRICE_VOLUME_COHERENCE_NEXT_20260910.md已保存。开始先查research/price_volume_coherence_inputs_v1.py、price_volume_coherence_v1.py、config/510300_price_volume_coherence_v1.json、reports/research/510300_price_volume_coherence_v1及scripts/prepare_round163_20260910.py是否已有后续进展，接续未完工作，不覆盖或重复。建议study510300_PRICE_VOLUME_COHERENCE_V1，PRIMARY PRICE_VOLUME_COHERENCE，一设置4新账户、0训练/参考/新data；3保存对照143/131/BH，每段8metrics。当前只有事前方案，尚无实现测试冻结计算。

163每日含分红对数收益r、成交量对数变化q=log(volume_t/volume_t−1)，含今天完整20对的Pearson同期corr20、r的20日和momentum20。corr>.2且momentum>0联合连续2CLOSE→允许持有；corr<−.2连续2或momentum<=0连续2→不允许持有；三个条件分别计数、不符即清，其他已知延续。相关常数/缺失则unknown，内部方向及计数清零，恢复完整重启0；不要把不同退出条件交替两天视为连续2。wholewindow不可删missing拼接。初次按完整市场日历0方向更新，不在entry/评价边界reset。允许且vol20>0target min(1,.1/vol20)，方向0target0，positive风险unknown→NO_VIEW。相关仅数值误差裁±1，门槛不round，不使用pvalue或证明预测力。官方https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html已查，手算corr及独立scipy核对。成交量是实际份额，不能当公募申购或净流入。3456行volume全正完整share、amount全正CNY已read-only预检；此前无价量相关策略命中。旧EOM/OBV/CMF/MFI/VWMA/NVI/EFI都已闭，不要重复；Klinger只检索公式未采用，勿花时间。

163直接复用saved_target_batch_runner_v1和event_clock_account_v1，自主target无父目标。空仓positive越过.1带买整百份；已有positive权重偏差<.1保持否则调center，known0及terminalOPEN优先退出。未知目标保持实际份额不变cash；受阻nextclose最新目标决定。无D60学习/止盈止损/持仓天数/重入等待/每笔退出锁。必要6类测试后写完整中文V1规则再一次freeze/run4，独立原始价格分红成交量+scipy相关与市场状态+四账户核对。162入口/测试/verify可静态复用框架；不重跑旧账户。

共有行情'''+common+'''

交付复用fast_round_delivery_v1，先candidate_outcomes和saved_verification_receipt，再完整中文结果及所有进出规则；163完成预计447设置463已评价468含旧未运行2058主指标，更新143最后比较并准备一个确有区别的下一有限方案。通过automation_update更新原510300-1-2完整字段并回读TOML，不直接改TOML不建重复任务。最新goal实际usageLimited，不达标不标complete，有可做新动作不标blocked，不消耗重置或购买额度。保留原ACTIVE及通知偏好，没变化安静，只有实质进展、完成、失败或需要行动才通知。继续推进。
'''
    with (ROOT/'docs/510300_RESEARCH_HEARTBEAT_THROUGH162_20260910.md').open('x', encoding='utf-8') as stream:
        stream.write(prompt)
    saved = tomllib.loads(Path('E:/CodexData/.codex/automations/510300-1-2/automation.toml').read_text(encoding='utf-8'))
    require(saved['id'] == '510300-1-2' and saved['kind'] == 'heartbeat', '原持续任务身份改变')
    args = {'id': saved['id'], 'mode': 'update', 'kind': saved['kind'], 'name': saved['name'], 'prompt': prompt,
        'status': saved['status'], 'rrule': saved['rrule'], 'targetThreadId': saved['target_thread_id']}
    if 'notification_policy' in saved:
        args['notificationPolicy'] = saved['notification_policy']
    write_json(ROOT/'reports/research/510300_heartbeat_update_through162_args.json', args, exclusive=True)
    print(json.dumps({'字数': len(prompt), '原状态': saved['status'], '完成轮次': 162}, ensure_ascii=False))


if __name__ == '__main__':
    main()
