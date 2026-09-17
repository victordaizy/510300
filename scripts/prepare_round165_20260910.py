"""保存收益强弱连续段的实际测试记录与事前中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RETURN_RUNS_STATE_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第165轮：', 1)
    text = text.replace('此文为第165轮事前方案，尚未实现、测试、冻结或计算新账户。',
        '此文为第165轮新历史账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新历史账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试实际7.45秒通过。三强三弱集中排列时为两段，交替排列时为六段，两种排列的预期段数同为四、方差同为一点二，'
        '标准化分数符号相反。精确中位数平局按规则移出段数序列；全部收益相同或只有一类非平局时，统计分数保持未知。'
        '全部日收益都为正的测试窗口仍可区分较强与较弱，因此强弱分类并不等于涨跌分类。\n\n'
        '已核对负一进入边界不计入、零分数退出边界计入、两个退出条件分别连续确认、未知后重启、'
        '完整六十日缺失边界、普通风险目标、已知不允许持有时不依赖风险资料。除息对应经济收益为零时，'
        '不会错误制造下跌收益或人为有效的连续段分数。未来修改、前缀截断和两档费用都不改变过去已知的共同市场目标。\n\n'
        '真实账户测试覆盖下一开盘进出及退出后重入、未知时维持份额、分红登记与除息确认和到账分开、'
        '终点开盘清算、空仓初次正目标越过已有持仓调仓带。独立结果核对将从原始收盘及分红重建收益，'
        '通过另行排序求中位数并按连续同类分组复算全部窗口，再核对完整资金和实际成交。\n')
    with (ROOT/'docs/510300_RETURN_RUNS_STATE_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_return_runs_state_v1/tests_receipt.json', {'recorded_at': now(),
        'exit_code': 0, 'passed': 6, 'seconds': 7.45,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_return_runs_state_v1.py -q',
        'output': '6 passed in 7.45s'}, exclusive=True)
    print('第165轮六项测试记录与全部中文规则已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
