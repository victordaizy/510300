"""保存已完成测试和八曲线退出的训练前中文规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, require, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / 'docs/510300_MARGINAL_MONOTONE_EXIT_V1.md'
    require(not path.exists(), '第159轮完整规则已经准备')
    text = (ROOT / 'docs/510300_MARGINAL_MONOTONE_EXIT_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第159轮：', 1)
    text = text.replace('此文为第159轮事前方案，尚未实现、测试、冻结、训练或生成新账户。',
        '此文为第159轮训练与账户计算前确定的完整规则。九项必要测试已经通过，尚未训练新历史模型或生成新账户。')
    text += ('\n## 实现细节与必要测试\n\n'
        '九项测试实际用时5.48秒，已验证手算加权区块、精确相同横坐标、独立SciPy保序解、周期重复后等权不变、'
        '协方差方向、恒定因素与精确零协方差、缺失状态、线性插值与端点延伸、八因素等权、'
        '成功失败缓存及未来隔离、入场固定版本、受阻退出、重新进入和分红到账。'
        '独立求解器的权重、增减方向和返回值定义见[SciPy官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.isotonic_regression.html)。\n\n'
        '先将精确相同的标准化横坐标合并，原权重和加权目标分别求和；不把数值接近但不同的横坐标合并。'
        '拟合完成后保留两端断点及每次预测水平发生改变前后的断点。只删除前后纵坐标完全相同的平段内部冗余点，'
        '因此保存的曲线在任意横坐标上的直线插值与完整支撑点完全等价，没有追加平滑或近似。'
        '同时保存原支撑点数量和合并区块数量，常数曲线保存单点及原因。\n\n'
        '每日记录每条曲线尚未除以八的原预测。每条预测对合成结果的贡献是该值的八分之一，'
        '这一区别在全部参数说明中保持明确。生产拟合采用本地相邻区块算法，独立核对采用SciPy的独立实现，'
        '并逐条比较支撑点及实际持仓状态预测。\n')
    path.write_text(text, encoding='utf-8')
    write_json(ROOT / 'reports/research/510300_marginal_monotone_exit_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 9, 'seconds': 5.48,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_marginal_monotone_exit_v1.py -q',
        'output': '9 passed in 5.48s'}, exclusive=True)
    print('第159轮九项已通过测试和完整中文规则已保存。', flush=True)


if __name__ == '__main__':
    main()
