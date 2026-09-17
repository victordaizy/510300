"""保存实际测试回执和含全部父因素的中文事前规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RUNS_COVARIANCE_BUDGET_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第167轮：', 1)
    text = text.replace('此文为第167轮事前方案，尚未实现、测试、冻结或计算新账户。',
        '此文为第167轮新账户计算前确定的完整规则，七项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n'
        '七项最终测试用时3.91秒，核对样本方差、协方差、收益差方差及解析预算的手算，'
        '预算零和一的上下界，完整常量收益与相同序列的风险退化、不足二百四十二日和缺失收益保留上次预算。'
        '还核对每月首个交易日更新、月中保持、两段账户分别各半初始化、两费用共用预算且各自使用对应目标、'
        '未来收益与终点开盘收益不能改变此前预算、截短历史时此前结果保持一致。\n\n'
        '实际账户测试覆盖空仓正目标进入、十个百分点调仓带、带外极小正目标整手取整清仓、'
        '带内极小正目标保持实际份额、明确零目标优先清仓、未知保持、退出后再次进入，'
        '并核对下一开盘执行、真实费用、分红登记除息到账、终点开盘清算。\n\n'
        '下列两套来源规则为各自冻结时的完整中文原文。其中原轮次训练及事前状态属于历史记录，'
        '第167轮仅读取已经形成的来源，不重新训练或计算父账户。\n')
    sources = [('第一来源：第143轮完整规则', 'docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md'),
        ('第二来源：第165轮完整规则', 'docs/510300_RETURN_RUNS_STATE_V1.md'),
        ('共同实际执行的整手取整说明', 'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md')]
    for title, path in sources:
        text += '\n## '+title+'\n\n'+(ROOT/path).read_text(encoding='utf-8').split('\n', 2)[2]
    with (ROOT/'docs/510300_RUNS_COVARIANCE_BUDGET_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_runs_covariance_budget_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 7, 'seconds': 3.91,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_runs_covariance_budget_v1.py -q',
        'output': '7 passed in 3.91s',
        'prior_test_run': {'passed': 7, 'seconds': 16.02, 'subsequent_change': '增加非零常量收益的精确退化处理与测试'}}, exclusive=True)
    print('第167轮测试回执及全部中文因素和进出规则已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
