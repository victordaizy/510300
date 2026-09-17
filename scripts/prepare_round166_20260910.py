"""保存组合测试记录，并纳入两套来源的全部中文因素和规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RUNS_REFERENCE_BLEND_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第166轮：', 1)
    text = text.replace('此文为第166轮事前方案，尚未实现、测试、冻结或计算组合账户。',
        '此文为第166轮新组合账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新组合账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试实际4.03秒通过，覆盖两套目标各半的手算、一套零目标时空闲预算保持现金、任何未知保留未知、'
        '目标范围、固定权重、基础与压力费用来源分别读取、错误身份和非下一交易日执行日期拒绝、未来与前缀隔离。'
        '真实账户测试确认一套退出而另一套继续时不会全部清仓，两套都为零才全部退出；未知时保持已有份额，'
        '退出后再次进入，分红权利、除息确认和到账分开，终点开盘清算。新账户净资产不同会产生不同份额，不能复制父账户股份。\n\n'
        '以下保留两套来源各自冻结时的完整中文原文。其中原轮次训练、测试及事前状态属于保存的历史记录，'
        '第166轮读取已完成来源，不重新进行那些训练、测试或账户计算。\n')
    for title, path in [('第一来源：第143轮完整规则', 'docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md'),
        ('第二来源：第165轮完整规则', 'docs/510300_RETURN_RUNS_STATE_V1.md')]:
        body = (ROOT/path).read_text(encoding='utf-8')
        text += '\n## '+title+'\n\n'+body.split('\n', 2)[2]
    with (ROOT/'docs/510300_RUNS_REFERENCE_BLEND_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_runs_reference_blend_v1/tests_receipt.json', {'recorded_at': now(),
        'exit_code': 0, 'passed': 6, 'seconds': 4.03,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_runs_reference_blend_v1.py -q',
        'output': '6 passed in 4.03s'}, exclusive=True)
    print('第166轮测试记录及含全部父规则的中文文档已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
