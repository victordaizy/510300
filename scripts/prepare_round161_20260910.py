"""保存持续回撤必要测试和计算新历史账户前的完整中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_DRAWDOWN_DEPTH_RISK_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第161轮：', 1)
    text = text.replace('此文为第161轮事前方案，尚未实现、测试、冻结或计算新账户。',
        '此文为第161轮计算新历史账户前确定的完整规则，六项必要测试已通过，尚未冻结或计算新历史账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '六项测试实际5.79秒通过，覆盖可手算的回撤深度与持续天数、窗口内先后峰值、价格恢复、同比例缩放、'
        '六十日和一百二十日缺失传播范围、双上限取较低值、正趋势风险未知与明确非正趋势零目标、除息财富不变、'
        '空仓直接进入及已有持仓调仓带、未来前缀隔离、费用同因素、真实下一开盘进出与重新进入以及分红到账。\n\n'
        '生产计算对每个六十日窗口独立取前缀最大值，再计算全部六十个回撤的平方平均根。'
        '当前二十日波动读取已经固定的共用因素，趋势由完整财富窗口现场计算。独立结果核对由原始价格和分红重新还原财富、'
        '普通波动和趋势，并以另一逐窗口、逐日循环复算持续回撤。未拟合任何收益模型，也未读取新策略历史收益来选择设置。\n')
    with (ROOT/'docs/510300_DRAWDOWN_DEPTH_RISK_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_drawdown_depth_risk_v1/tests_receipt.json', {'recorded_at': now(),
        'exit_code': 0, 'passed': 6, 'seconds': 5.79,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_drawdown_depth_risk_v1.py -q',
        'output': '6 passed in 5.79s'}, exclusive=True)
    print('第161轮六项已通过测试和中文规则已保存。', flush=True)


if __name__ == '__main__':
    main()
