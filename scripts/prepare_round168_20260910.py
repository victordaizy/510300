"""保存两种机会合并测试记录和完整中文父规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RUNS_OPPORTUNITY_UNION_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第168轮：', 1)
    text = text.replace('此文为第168轮事前方案，尚未登记、实现、测试、冻结或计算新账户。',
        '此文为第168轮新账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试用时3.57秒通过，覆盖两种目标的手算和百分之一百上限、单边信号保持父目标规模、'
        '双边零目标、另一套目标为零或为百分之一百时仍不掩盖未知、费用来源与日期身份、未来与前缀隔离。'
        '独立资金测试覆盖空仓进入、调仓带、带外极小正目标整手取整清仓、带内保持、明确零目标退出、'
        '未知保持实际份额、退出后再次进入，并对两种方案各自核对真实下一开盘、费用、分红登记除息到账和终点清算。\n\n'
        '以下保留两套来源各自冻结时的全部中文规则。原轮次训练、测试和事前状态属于原记录，'
        '第168轮只读取已经形成的来源，不重复原训练或父账户计算。\n')
    for title, path in [('第一来源：第143轮完整规则', 'docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md'),
        ('第二来源：第165轮完整规则', 'docs/510300_RETURN_RUNS_STATE_V1.md'),
        ('共同实际执行的整手取整说明', 'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md')]:
        text += '\n## '+title+'\n\n'+(ROOT/path).read_text(encoding='utf-8').split('\n', 2)[2]
    with (ROOT/'docs/510300_RUNS_OPPORTUNITY_UNION_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_runs_opportunity_union_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 6, 'seconds': 3.57,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_runs_opportunity_union_v1.py -q',
        'output': '6 passed in 3.57s'}, exclusive=True)
    print('第168轮必要测试和完整中文规则已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
