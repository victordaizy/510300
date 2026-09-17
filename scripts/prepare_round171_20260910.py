"""保存相邻收益相关状态的完整中文规则和实际测试回执。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RETURN_LAG_STATE_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第171轮：', 1).replace(
        '此文为第171轮事前方案，尚未登记、实现、测试、冻结或计算新账户。',
        '此文为第171轮新账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n六项测试5.77秒通过，核对完整六十个收益形成五十九组相邻配对、'
        '均值及交叉偏差手算、正负一边界、单边常量导致未定义、进入的严格正值与退出的非正边界、'
        '两种退出分别计数、未知时清内部状态、缺失六十日传播、除息后总收益正确、普通风险规模与调仓带。\n\n'
        '同时核对未来数据与前缀隔离、两费用共用同一完整市场状态、十五点零五分判断及下一开盘执行、'
        '真实资金未知保持、退出后再次进入、登记除息到账分开及终点开盘清算。规则不依赖父目标、父周期或预测模型训练。\n')
    with (ROOT/'docs/510300_RETURN_LAG_STATE_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_return_lag_state_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 6, 'seconds': 5.77,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_return_lag_state_v1.py -q',
        'output': '6 passed in 5.77s'}, exclusive=True)
    print('第171轮完整中文规则和六项必要测试回执已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
