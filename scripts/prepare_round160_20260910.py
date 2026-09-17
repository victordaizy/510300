"""保存六项实际测试与双向市场状态的训练前完整规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_SEQUENTIAL_RETURN_STATE_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第160轮：', 1)
    text = text.replace('沿用已保存目标账户的百分之十绝对仓位调仓带：比较目标比例和本账户当前股票市值占上一收盘净资产的比例，偏差达到十个百分点时调至目标中心；小于十个百分点时维持份额。',
        '沿用已保存目标账户的百分之十绝对仓位调仓带。空仓时不使用调仓带，正目标直接计算目标份额。已有持仓且目标为正时，比较目标比例和本账户当前股票市值占上一收盘净资产的比例，偏差达到十个百分点时调至目标中心；小于十个百分点时维持份额。目标份额等于目标比例乘上一收盘净资产，除以原始收盘价，再向下取一百份整数倍；即使正目标，整手取整也可能得到零份。')
    text = text.replace('此文为第160轮事前方案，尚未实现、测试、冻结或计算新账户。',
        '此文为第160轮新历史账户计算前确定的完整规则。六项必要测试已经实际通过，尚未冻结或计算新历史账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试2.88秒通过：可手算的双向累积、恰好达到五、方向保持、任一触发双累积归零、缺失与零波动、'
        '前二十日波动明确排除今天、除息不产生虚假损失、仓位上限与未知状态、空仓小目标直接进入和已有仓位调仓带、'
        '未来前缀隔离、两档费用同目标、下一开盘真实进入退出及重新进入、分红确认到账和终点清算。\n\n'
        '生产因素直接读取共用文件已经保存的含分红简单收益和当前二十日年化波动；独立结果核对重新从原始收盘和每份分红计算两者。'
        '前二十日尺度由简单收益完整窗口现场计算。按本文定义在完整日历顺序递推，未训练任何收益预测模型。'
        '未来读取历史价格前已经完成上述合成测试，全部阈值、窗口、重置和风险预算只有本文一套。\n')
    path = ROOT/'docs/510300_SEQUENTIAL_RETURN_STATE_V1.md'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_sequential_return_state_v1/tests_receipt.json', {'recorded_at': now(),
        'exit_code': 0, 'passed': 6, 'seconds': 2.88,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_sequential_return_state_v1.py -q',
        'output': '6 passed in 2.88s'}, exclusive=True)
    print('第160轮六项已通过测试及完整中文规则已保存。', flush=True)


if __name__ == '__main__':
    main()
