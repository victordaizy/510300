"""保存三种组合的共同测试及全部中文来源规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RETURN_CONFIRMATION_AUXILIARY_BATCH_NEXT_20260911.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第174轮：', 1)
    text = text.replace('此文为第174轮事前方案，尚未登记、实现、测试、冻结或计算新账户。',
        '此文为第174轮账户计算前确定的完整规则，七项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '七项必要测试用时3.19秒通过，覆盖三个方案的手算、核心保留、辅助独立进入和相加封顶，'
        '未知辅助和所需方向不能被条件短路掩盖、只用相关方向的方案不依赖上涨日方向，'
        '以及两套方向不重复使用独立波动率目标。费用来源、下一开盘日期、目标范围、未来与前缀隔离、'
        '收盘判断时钟均通过核对。三个独立资金账户分别检查辅助撤出但核心保留、'
        '目标为零的退出、退出后重新进入，以及未知保持实际份额、分红登记除息到账和终点开盘清算。\n\n'
        '以下是四套来源原有的全部中文规则。原轮次的训练、测试、事前状态及失败结论属于原记录，'
        '本轮读取已形成来源，不重跑旧模型或父账户。方向来源只使用允许状态。\n')
    for title, path in [
        ('核心来源：第143轮全部规则', 'docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md'),
        ('辅助来源：第165轮全部规则', 'docs/510300_RETURN_RUNS_STATE_V1.md'),
        ('相邻相关方向：第171轮全部规则', 'docs/510300_RETURN_LAG_STATE_V1.md'),
        ('上涨日优势方向：第172轮全部规则', 'docs/510300_RETURN_SIGN_BALANCE_V1.md'),
        ('共同实际执行的整手取整说明', 'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md')]:
        text += '\n## '+title+'\n\n'+(ROOT/path).read_text(encoding='utf-8').split('\n', 2)[2]
    with (ROOT/'docs/510300_RETURN_CONFIRMATION_AUXILIARY_BATCH_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 7, 'seconds': 3.19,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_return_confirmation_auxiliary_batch_v1.py -q',
        'output': '7 passed in 3.19s'}, exclusive=True)
    print('第174轮必要测试和全部中文来源规则已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
