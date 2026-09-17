"""记录必要测试，组装本轮和三套原来源的全部中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_SIGN_CONFIRMED_RUNS_AUXILIARY_NEXT_20260911.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第173轮：', 1)
    text = text.replace('此文为第173轮事前方案，尚未登记、实现、测试、冻结或计算新账户。',
        '此文为第173轮账户计算前确定的完整规则，七项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '七项必要测试用时7.02秒通过，覆盖条件组合手算、相加封顶、辅助独立进入、'
        '方向不允许时核心保留、未使用辅助目标未知仍传播、方向未知传播、非法方向拒绝，'
        '以及方向与第172轮独立波动率目标的区别。'
        '费用来源、来源身份、下一开盘日期、未来隔离、前缀一致和决定时钟也已核对。'
        '实际资金测试覆盖整手与调仓带、带外极小正目标清仓、带内保持、核心保留、明确零目标退出、'
        '未知保持份额、再次进入、分红登记除息到账及固定终点开盘清算。\n\n'
        '下面保留三套来源冻结时的全部中文规则。原轮次的测试、训练、历史结论和事前状态属于原记录；'
        '本轮只读取已经形成的来源，不重复模型训练或父账户回测。第172轮的独立策略仍按失败关闭，'
        '本轮只使用它原有的允许方向，不能把它的波动率限制再套到组合上。\n')
    for title, path in [
        ('核心来源：第143轮全部中文规则', 'docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md'),
        ('辅助来源：第165轮全部中文规则', 'docs/510300_RETURN_RUNS_STATE_V1.md'),
        ('方向来源：第172轮全部中文规则', 'docs/510300_RETURN_SIGN_BALANCE_V1.md'),
        ('共同实际执行的整手取整说明', 'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md')]:
        text += '\n## '+title+'\n\n'+(ROOT/path).read_text(encoding='utf-8').split('\n', 2)[2]
    with (ROOT/'docs/510300_SIGN_CONFIRMED_RUNS_AUXILIARY_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_sign_confirmed_runs_auxiliary_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 7, 'seconds': 7.02,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_sign_confirmed_runs_auxiliary_v1.py -q',
        'output': '7 passed in 7.02s'}, exclusive=True)
    print('第173轮必要测试和全部中文来源规则已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
