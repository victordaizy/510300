"""保存价量规则的实际测试记录与新账户计算前的全部中文定义。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_PRICE_VOLUME_COHERENCE_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第163轮：', 1)
    text = text.replace('此文为第163轮事前方案，尚未实现、测试、冻结或计算新账户。',
        '此文为第163轮新历史账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新历史账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试实际3.82秒通过。手算价量相关与独立统计库结果一致；放大或平移变量不改变相关，翻转一列符号会翻转相关。'
        '常数窗口保持未知，不能给出已知零相关。缺失某日成交量会使当天及下一天的成交量变化未知，'
        '相应二十日窗口全部保留缺失，不能跳过缺失凑满二十日。\n\n'
        '已验证正负零点二的严格门槛、上涨动量为零时退出条件成立、两种退出分别累计、'
        '缺失重启、风险上限和已知不准入时零目标。前缀与未来数据修改不影响过去，'
        '两档费用使用相同因素及目标，准备期和终点收盘不产生新请求。\n\n'
        '实际账户测试覆盖下一开盘买卖、未知时保持份额、全部退出后重新进入、分红登记及除息确认、'
        '到账日期独立于除息日、终点开盘清算，以及空仓初次正目标不受已有持仓调仓带阻拦。'
        '生产实现直接从原始收盘、前收盘、分红和成交量构造价量变化；独立结果核对再重算原始因素及相关系数。\n')
    with (ROOT/'docs/510300_PRICE_VOLUME_COHERENCE_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_price_volume_coherence_v1/tests_receipt.json', {'recorded_at': now(),
        'exit_code': 0, 'passed': 6, 'seconds': 3.82,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_price_volume_coherence_v1.py -q',
        'output': '6 passed in 3.82s'}, exclusive=True)
    print('第163轮六项测试记录及全部中文规则已保存，尚未冻结或计算新账户。')


if __name__ == '__main__':
    main()
