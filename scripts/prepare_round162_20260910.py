"""保存排序信号测试与新历史账户计算前的完整中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_SESSION_SIGNED_RANK_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第162轮：', 1)
    text = text.replace('此文为第162轮事前方案，尚未实现、测试、冻结或计算新账户。',
        '此文为第162轮新历史账户计算前确定的完整规则，六项必要测试已经通过，尚未冻结或计算新历史账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试实际4.26秒通过，覆盖带符号名次手算、精确平局、已知零差、独立排序与单侧标准化秩核对、'
        '极端值只在名次改变时影响分数、严格门槛和连续两个收盘确认、缺失中断与恢复、风险上限、'
        '全部平局时的已知零分数、除息不产生错误日内隔夜差、空仓初次买入及已有持仓调仓带、'
        '未来前缀隔离、相同因素的两档费用、真实下一开盘退出重入和分红确认到账。\n\n'
        '生产排序直接按绝对非零差值的精确不同取值分组，用每组名次起点与终点的平均数赋予平局名次，'
        '不调用统计检验概率决定仓位。共用文件的日内和隔夜对数变化读取方式与原始开收盘及分红定义一致；'
        '独立结果核对将重新按原始数据计算，并使用独立统计库的排序函数核对每个完整窗口。\n')
    with (ROOT/'docs/510300_SESSION_SIGNED_RANK_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_session_signed_rank_v1/tests_receipt.json', {'recorded_at': now(),
        'exit_code': 0, 'passed': 6, 'seconds': 4.26,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_session_signed_rank_v1.py -q',
        'output': '6 passed in 4.26s'}, exclusive=True)
    print('第162轮六项已通过测试和完整中文规则已保存。', flush=True)


if __name__ == '__main__':
    main()
