"""将持续任务的默认组合账户说明替换为下一项学习退出的准确接续信息。"""
import json
from pathlib import Path

from research.intraday_overnight_increment_v1 import require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'reports/research/510300_heartbeat_update_through191_args.json'
    args = json.loads(path.read_text(encoding='utf-8'))
    text = args['prompt']
    old = next(p for p in text.split('\n\n') if p.startswith('框架可复用research/'))
    new = ('第192轮使用research/rearmed_cycle_exit_account_v1.py的simulate_rearmed_exit，'
        '以及simple_intraday_protection_v1.py的make_rules(frame)[D60_INTRA]。'
        '这是一次买入、全部退出、锁定退出意图及等待旧条件消失的独立持仓账户；'
        '本轮不使用saved_target账户的.1调仓带，不直接跟随181目标。'
        '原128仅作为固定版本原模型对照，181为当前联合目标比较，32为买入持有，共12条旧对照直接复用。'
        '样本reports/research/510300_within_cycle_exit_v1/extended_reference_samples.parquet，'
        '时点同目录saved_models.json。单成分研究single_component_exit_inputs_v1.py的缓存与固定模型结构可参考，'
        '但须用新研究自己的十二项定义、二十项岭回归和模型类型，不改变旧文件。'
        '必要结果核对可参考scripts/verify_round157_20260909.py的实际持仓状态及账本部分；'
        '新增数学核对只验证二十项正规方程、两级尺度、成熟成员和向前缓存，避免重训父策略或额外数值包。')
    text = text.replace(old, new)
    text = text.replace('真实账户', '完整模拟账户').replace('真实nextOPEN', '按下一开盘模拟')
    text = text.replace('自动目标此前实际为usageLimited，继续时按实际工具状态判断；',
                        '本次已读目标工具为active，目标“目标夏普1.2，年化10”未完成；继续时按实际工具状态判断；')
    extra = ('已完成且不要重跑的诊断：reports/research/510300_joint_saved_frontier_through186/result.json'
        '复核至186轮标准四情景候选，181是八个指标相对联合门槛的较佳比较，未有四情景全部过线；'
        '后续187—191均失败。181主基础／压力净夏普1.3064368／1.2226828，年化9.663436%／8.993719%；'
        '较早净夏普1.068767／1.055728，年化9.884874%／9.836233%，仍未达标。'
        '174是旧夏普重点的均衡比较，不能把174当成已满足新年化目标。'
        'reports/research/510300_saved_session_attribution_through188/result.json已复核181与182共8条旧账本、11292行，'
        '181主要损失来自持仓隔夜，新增买入当天日内贡献合计为正，不能删除隔夜、假设T+0或由此保证延后买入有利。'
        '189四账户、190八账户、191四账户已完成核对与交付，日历补充暂停，不调日历天数救回。')
    require(extra not in text, '本轮持续任务参数已完善')
    text += '\n\n' + extra
    args['prompt'] = text
    write_json(path, args)
    (ROOT / 'docs/510300_RESEARCH_HEARTBEAT_THROUGH191_20260913.md').write_text(text, encoding='utf-8')
    print(json.dumps(args, ensure_ascii=False))


if __name__ == '__main__':
    main()
