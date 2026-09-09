"""输出本轮失败归因、完整中文策略规则及普通结果表，不生成审阅包。"""
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,now

RESULT=ROOT/'reports/research/510300_forward_eps_revision_distribution_policy_v1'
DIAG=ROOT/'reports/research/510300_forward_eps_revision_diagnostic_v1'
OUT=ROOT/'deliverables/510300前瞻盈利修正分布与进出场_20260907'
NAMES={'D1_EPS_DISTRIBUTION':'仅前瞻盈利分布，新主方案',
       'D2_PRICE_EPS_DISTRIBUTION':'价格加前瞻盈利分布',
       'D3_EPS_DISTRIBUTION_COHERENT_TREND':'前瞻盈利分布加一致趋势进出场',
       'C0_ORIGINAL_EPS':'原仅EPS方案，复用对照',
       'C1_ORIGINAL_PRICE_EPS':'原价格加EPS方案，复用对照',
       'C2_MATCHED_PRICE':'相同有效月份价格模型，复用对照','BUY_HOLD':'买入持有510300，复用对照'}


def run():
    result=read(RESULT/'result.json');diag=read(DIAG/'result.json')
    assert read(RESULT/'saved_numerical_verification.json')['status'].startswith('PASS')
    inventory=read(ROOT/'reports/research/510300_forward_eps_price_share_alignment_inventory_v1/result.json')
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for row in result['all_metrics']:
        rows.append({'策略':NAMES[row['model']],'记录代号':row['model'],'费用情景':'基础' if row['cost']=='BASE' else '压力',
                     '完整账户净夏普':row['net_sharpe'],'年化收益率':row['annualized_return'],
                     '累计收益率':row['cumulative_return'],'最大回撤':row['max_drawdown'],
                     '实际成交笔数':row['trade_count'],'佣金元':row['commission'],'滑点成本元':row['slippage_cost'],
                     '新生成账户':row['model'].startswith('D')})
    pd.DataFrame(rows).to_csv(OUT/'第十六轮十四条账户结果.csv',index=False,encoding='utf-8-sig')
    accounts={model:pd.read_parquet(RESULT/'evaluation/BASE'/(model+'_ledger.parquet')) for model in NAMES}
    daily=pd.DataFrame({'日期':accounts['BUY_HOLD'].date,
                        '新盈利分布损益元':accounts['D1_EPS_DISTRIBUTION'].pnl,
                        '旧EPS损益元':accounts['C0_ORIGINAL_EPS'].pnl,
                        '一致趋势损益元':accounts['D3_EPS_DISTRIBUTION_COHERENT_TREND'].pnl})
    daily['新因子对旧EPS损益差元']=daily['新盈利分布损益元']-daily['旧EPS损益元']
    daily['一致趋势对新因子损益差元']=daily['一致趋势损益元']-daily['新盈利分布损益元']
    years=daily.groupby(daily['日期'].dt.year).sum(numeric_only=True)
    years.to_csv(OUT/'逐年实际损益与差额.csv',encoding='utf-8-sig')
    cycle=accounts['D1_EPS_DISTRIBUTION'].loc[lambda x:x.date.between('2023-10-09','2024-02-01')]
    first_loss=float(cycle.pnl.sum())
    trades=[]
    for model in ['D1_EPS_DISTRIBUTION','D2_PRICE_EPS_DISTRIBUTION','D3_EPS_DISTRIBUTION_COHERENT_TREND']:
        for row in accounts[model].loc[lambda x:x.filled_quantity.ne(0)].itertuples():
            action='进入或重新进入' if row.filled_quantity>0 and row.shares_before==0 else '增加' if row.filled_quantity>0 else '全部退出' if row.shares==0 else '减少'
            reason=getattr(row,'execution_exit_reasons','')
            if not reason:reason='研究终点统一开盘退出' if row.mark_clock=='OPEN_TERMINAL' else '月末收益、风险和成本比较'
            trades.append({'策略':NAMES[model],'成交日期':str(row.date.date()),'操作':action,
                           '成交份额':int(row.filled_quantity),'成交后份额':int(row.shares),'原因':reason})
    pd.DataFrame(trades).to_csv(OUT/'三个新方案全部进出场成交.csv',index=False,encoding='utf-8-sig')
    counts=diag['valid_model_month_comparable_company_months']
    lines=[f'''# 第十六轮：前瞻盈利修正的失败原因、因子和完整进出场

生成时间：{now()}。本轮完成两种新模型72次逐月拟合、六条新账户，并复用八条原对照评价，共十四条。未准备GPT数值审阅包。

## 一、结果先说清楚

夏普1.2仍未达到。新主方案“仅前瞻盈利分布”的完整账户净夏普0.414，低于原仅EPS方案的0.619。加价格后为0.523，与原价格加EPS的0.522接近；一致趋势进出场为0.153。不能把分布更有变化就说成预测更准，也不能把小幅点估计改善当稳定超额。

## 二、为何原修正因子经常为零

50个有效月末共有5383条可比公司月份：同一份报告延用2304条；换了报告但原文利润预测未变768条；真正上修842条；真正下修1469条。前两类共3072条，占{counts['unchanged_count']/counts['comparable_company_count']:.2%}。当大部分值为零或零处在分布中间时，中位数就无法反映其余公司的变化。

这不是证明原数学公式算错。保存中位数可从公司记录重现，同报告同年度修正确实为零。问题是“是否有新报告”和“新报告是否改变盈利预期”被汇总方式掩盖。全部5500条可比公司月份已核对同证券、同绝对年度、原利润标签、百万元单位和公开日期；其中2351条与前面月份重复同一报告对，不能当作独立盈利事件。

新修正广度在50个月中有49个不同值，平均幅度和报告更新比例各有50个不同值；广度有11个月为正、平均幅度10个月为正。它们恢复了一部分变化信息，但回测不支持因此就能获得更高夏普。

## 三、各方案的历史表现

评价为20万元、2020年1月2日至2026年8月14日开盘退出的完整账户，包含早期现金月份，1604个交易日。基础费用：佣金万分之二、最低5元、单边滑点万分之五；压力费用：佣金万分之四、最低5元、单边滑点千分之一。100份整手、T+1、下一开盘、涨跌停限制、分红与应收均计入；现金和夏普参考收益为零，每年242个交易日。

| 方案 | 费用 | 年化收益 | 净夏普 | 最大回撤 | 成交笔数 |
|---|---|---:|---:|---:|---:|''']
    for row in rows:
        lines.append(f"| {row['策略']} | {row['费用情景']} | {row['年化收益率']:.2%} | {row['完整账户净夏普']:.3f} | {row['最大回撤']:.2%} | {row['实际成交笔数']} |")
    lines.append(f'''
两种区块长度计算的配对增量区间均跨零，尚无稳定增量证据。历史已经多轮研究，这些区间不是消除了筛选偏差的独立验证。

## 四、这次具体亏在哪里

新主方案2023年10月9日进入，2023年11月1日又增加少量份额，2024年2月1日全部退出。首轮实际账户损失{-first_loss:,.2f}元；同一时期原仅EPS方案保持现金。新因子让模型更早给出了买入判断，但这次较早进入落在下跌阶段。

最终新主方案净值比原仅EPS方案少35,904.65元。逐年差额如下，它们是保存账户的真实损益差，不是重新跑不同时间段挑结果；各年会受到前期净值与后续可买份额影响，不能解释为相互独立的因果贡献。

| 年度 | 新盈利分布损益 | 原EPS损益 | 新因子减原EPS | 一致趋势损益 |
|---|---:|---:|---:|---:|''')
    for year,row in years.iterrows():
        lines.append(f"| {year} | {row['新盈利分布损益元']:,.2f} | {row['旧EPS损益元']:,.2f} | {row['新因子对旧EPS损益差元']:,.2f} | {row['一致趋势损益元']:,.2f} |")
    lines.append('''
一致趋势方案阻止了2023年弱趋势中的进入，但后续上涨收益也受到影响。相对同一盈利预测的原仓位方案，它的全期净值少47,890.30元。它修正了已知弱趋势下反复进入的问题，基础费用实际发生4次趋势清仓，但仍未形成更好的收益风险表现。进出场逻辑一致只是必要条件，本身不是盈利证明。

## 五、下一项来源问题

已经定位到可复用的历史未复权价格：7175条有前瞻EPS的公司月份中，7092条能匹配当月价格，83条缺失。但这批对应行情记录的历史股数与总市值字段全部为空。

已有39158行旧财报面板含总股本字段，可以继续核对公告和股本变化；财报期末股本不能直接冒充研报基准日或交易日股本。当前没有把未经股本对齐的EPS除以市场价格，写成“已完成校正的前瞻市盈率”。下一步围绕前瞻EPS与当时价格、股数和股东回报的匹配继续推进，保留免费来源限制。

这项覆盖检查未下载新数据、未计算新的未来收益、未增加策略账户，也不替代继续寻找稳定超额与夏普1.2的目标。

## 六、全部中文规则

以下是运行前登记的完整规则，含因子、训练、进入、增加、减少、清仓、缺失处理和再次进入，没有用程序代码替代。

''')
    protocol=(ROOT/'docs/510300_FORWARD_EPS_REVISION_DISTRIBUTION_POLICY_V1.md').read_text(encoding='utf-8')
    lines.append(protocol)
    lines.append('''

## 七、保存记录

本目录包含普通CSV：十四条账户结果、三个新方案全部成交、逐年实际损益差额。完整模型与账户在 reports/research/510300_forward_eps_revision_distribution_policy_v1；修正原因在 reports/research/510300_forward_eps_revision_diagnostic_v1；价格和股数覆盖在 reports/research/510300_forward_eps_price_share_alignment_inventory_v1。

六项必要测试通过；保存因子、72个模型预测、训练日期、十四条完整账户、八次双费用趋势退出及四份保存区块文件核对通过。复核没有重训模型、生成新账户、重新抽样或下载资料，没有进行额外安全审计或外部GPT审阅。目标继续保持未完成。
''')
    (OUT/'前瞻盈利修正_失败归因与完整策略说明.md').write_text('\n'.join(lines),encoding='utf-8')
    print('第十六轮中文规则、失败归因和普通结果表已保存。',str(OUT),flush=True)


if __name__=='__main__':
    run()
