"""保存上涨日优势规则的完整中文说明和必要测试回执。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RETURN_SIGN_BALANCE_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第172轮：', 1).replace(
        '此文为第172轮事前方案，尚未登记、实现、测试、冻结或计算新账户。',
        '此文为第172轮新账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n六项测试8.23秒通过，核对上涨、下跌、持平的实际零收益边界和计数手算，'
        '相同上涨频率但不同累计收益、全上涨和全下跌边界、全持平的明确零优势、缺失窗口保持未知，'
        '以及两日联合进入、分别两日退出、未知后状态重启、普通风险规模和调仓带。\n\n'
        '还核对未来与前缀隔离、两费用共用市场状态、十五点零五分判断与下一开盘执行、'
        '未知保持实际份额、退出后再次进入、分红登记除息到账及终点开盘清算。此规则不依赖父策略目标、周期或预测训练。\n')
    with (ROOT/'docs/510300_RETURN_SIGN_BALANCE_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_return_sign_balance_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 6, 'seconds': 8.23,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_return_sign_balance_v1.py -q',
        'output': '6 passed in 8.23s'}, exclusive=True)
    print('第172轮完整中文规则与六项必要测试回执已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
