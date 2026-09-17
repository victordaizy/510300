"""保存价格顺序分布的实际测试记录与事前中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_ORDINAL_ENTROPY_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第164轮：', 1)
    text = text.replace('此文为第164轮事前方案，尚未实现、测试、冻结或计算新账户。',
        '此文为第164轮新历史账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新历史账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试实际11.23秒通过。使用原论文七个数的例子手算五个三日模式，三类分别出现两次、一次、两次，'
        '频数及标准化熵与独立熵函数一致；六类平均出现时熵为一，完全递增、递减和常数序列的熵为零。'
        '精确平局按时间先后处理，并已覆盖三种含平局的不同排列。\n\n'
        '已测试六十日顺序窗口与六十一日动量窗口的不同缺失恢复边界、严格零点九及零点九五门槛、'
        '两个退出条件分别连续确认、未知状态重启、普通风险上限、已知空仓目标不依赖风险资料。'
        '除息后的含分红财富保持不变时，不会错误制造下跌动量。未来资料或截断前缀不改变过去目标，两档费用因素相同。\n\n'
        '真实账户测试覆盖下一开盘进出、退出后重入、未知目标保持份额、登记日持仓产生分红权利、'
        '除息确认与之后到账分别核算、终点开盘清算和完整资金恒等关系。独立结果核对将从原始价格与分红还原财富，'
        '用逐窗口的另一种排序枚举方式核对全部六类频数、熵、动量、连续方向及真实账户。\n')
    with (ROOT/'docs/510300_ORDINAL_ENTROPY_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_ordinal_entropy_v1/tests_receipt.json', {'recorded_at': now(),
        'exit_code': 0, 'passed': 6, 'seconds': 11.23,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_ordinal_entropy_v1.py -q',
        'output': '6 passed in 11.23s'}, exclusive=True)
    print('第164轮六项测试记录及完整中文规则已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
