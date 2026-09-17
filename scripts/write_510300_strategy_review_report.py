"""依据已验证诊断结果生成中文报告和可分享图表。"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_strategy_review_diagnostics_v1'
DELIVERY = OUT/'delivery'


def load(name):
    return pd.read_csv(OUT/(name+'.csv'))


def md_table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(str(x).replace('|','／') for x in row)+'|' for row in rows])


def write(name, text):
    (DELIVERY/name).write_text(text, encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    DELIVERY.mkdir(exist_ok=True)
    accounts, probes = load('account_metrics'), load('counterfactual_account_metrics')
    pair = load('paired_exit_summary')
    case = pair[(pair.cost=='STRESS') & (pair.scope=='main_mature')]
    block = pd.read_csv(ROOT/'reports/research/510300_point_pass_fixed_diagnostic_v1/bootstrap_intervals.csv')
    old = block[(block.model=='SELECTED_MIX_BAND10_SIMPLE2') & (block.period=='evaluation') & (block.cost=='STRESS') & (block.block_length==20)].iloc[0]
    model_names = {'PRICE_ORIGINAL':'原价格、止损、追踪及60日退出','RIDGE_MONTHLY':'原滚动线性退出',
                   'WITHIN_MONTHLY':'滚动周期内退出','WITHIN_ENTRY_FIXED':'当前入场固定周期内退出',
                   'TIME20':'原价格规则加20日上限','PROFIT08':'原价格规则加8%止盈'}
    scenario_names = {'NO_INTERNAL_UPSCALE':'内部风险倍率只降不升','NO_ZERO_WAIT':'取消上层两次归零等待',
                      'NO_115_MULTIPLIER':'取消1.15倍固定放大','FINAL_RISK_CAP10':'最终目标按ETF波动代理封顶10%',
                      'NO_FINAL_BAND':'取消最终10个百分点带宽','SOURCE_A_ONLY':'只用甲的计划，同一最终账户规则',
                      'SOURCE_B_ONLY':'只用乙的计划，同一最终账户规则','BUY_HOLD':'买入持有，分红留现金',
                      'CONSTANT20':'固定20%目标、10个百分点带宽','ETF_VOL10':'ETF自身20日波动目标10%'}
    main_stress = accounts[(accounts.period=='main') & (accounts.cost=='STRESS')].iloc[0]
    main_base = accounts[(accounts.period=='main') & (accounts.cost=='BASE')].iloc[0]
    metric_table = md_table(['区间／成本','交易日','年化','净夏普','最大回撤','模拟成交','完整周期'],
        [[('2020—2026/9/11' if r.period=='main' else '2015—2019')+'／'+('基础' if r.cost=='BASE' else '压力'),
          r.days,f'{r.annual_return:.4%}',f'{r.sharpe:.4f}',f'{r.max_drawdown:.4%}',r.fills,r.cycles] for r in accounts.itertuples()])
    concentration = md_table(['主历史成本','总净利润','最大周期净利润','最大周期占比','前五周期占比'],
        [[label,f'{r.profit:,.2f}元',f'{r.profit*r.top1_fraction:,.2f}元',f'{r.top1_fraction:.2%}',f'{r.top5_fraction:.2%}']
         for label,r in [('基础',main_base),('压力',main_stress)]])
    exit_table = md_table(['固定退出对照','成熟完整周期','相对原价格退出的平均增量','改善／恶化／相同','配对区间2.5%—97.5%'],
        [[model_names[r.variant],r.complete_cycles,f'{r.mean_increment*100:+.3f}个百分点',
          f'{r.better_cycles}／{r.worse_cycles}／{r.same_cycles}',f'[{r.delta_q025*100:+.3f}, {r.delta_q975*100:+.3f}]个百分点']
         for r in case.itertuples() if r.variant!='PRICE_ORIGINAL'])
    chosen = probes[(probes.cost=='STRESS') & (probes.period=='main') & probes.scenario.isin(scenario_names)]
    module_table = md_table(['主历史压力成本情景','年化','净夏普','实际年化波动','最大回撤'],
        [['当前固定85%／15%方案',f'{main_stress.annual_return:.2%}',f'{main_stress.sharpe:.3f}',f'{main_stress.volatility:.2%}',f'{main_stress.max_drawdown:.2%}']]+
        [[scenario_names[r.scenario],f'{r.annual_return:.2%}',f'{r.sharpe:.3f}',f'{r.volatility:.2%}',f'{r.max_drawdown:.2%}'] for r in chosen.itertuples()])
    capital_rows = []
    for variant in ['CAPITAL_100K','CAPITAL_1M']:
        result = json.loads((OUT/'full_graph'/variant/'result.json').read_text(encoding='utf8'))
        for r in result['metrics']:
            d=next(x for x in result['differences'] if x['cost']==r['cost'])
            capital_rows.append([f"{r['capital']/10000:.0f}万元／{r['cost']}",f"{r['annual_return']:.4%}",f"{r['sharpe']:.4f}",
                                f"{d['max_target_difference']*100:.2f}个百分点",d['zero_positive_disagreement_days']])
    capital_table=md_table(['完整依赖图初始资金／成本','年化','净夏普','最大单日目标差','零／正目标不同日'],capital_rows)
    label=load('label_within_cycle_relationship')
    risk=load('holding_risk'); r=risk[(risk.period=='main')&(risk.cost=='BASE')].iloc[0]
    costs=load('cost_path_constraints')
    cost_table=md_table(['区间／成本实验','年化','净夏普','申请与成交数量不符日'],
        [[('主历史' if x.period=='main' else '较早历史')+'／'+('固定基础请求，压力执行' if x.scenario=='BASE_REQUESTS_STRESS_COST' else '固定基础目标，压力自有账户'),
          f'{x.annual_return:.4%}',f'{x.sharpe:.4f}',x.constrained_execution_rows] for x in costs.itertuples()])
    early=load('all_complete_cycles');early=early[(early.period=='earlier')&early.entry.str.startswith('2019-05-31')]
    early_text='；'.join(f"{x.cost}在{x.exit}清仓，净利润{x.profit:,.2f}元" for x in early.itertuples())
    report=f'''# 510300 最新策略：外部评审诊断结果

报告日期：2026年9月13日。对象：SELECTED_MIX_BAND10_SIMPLE2，第209轮事后选出的85%／15%两来源方案；当前主历史使用第214轮连续至9月11日收盘的账户。

**结论：原说明书的核心回测数字成立，策略在已观察历史中明显优于本次固定简单对照；但“稳定年化10%、夏普1.2”的证据尚未建立。达到年化10%较依赖内部风险放大、退出等待和少数行情。学习退出的独立经济增量仍无法确认。保留固定研究版本，停止以这些历史继续追逐阈值，是目前有证据支持的处理。**

这轮已执行完整诊断。四原账户372笔成交复算通过；原27节点依赖图对应的22账户重建通过，逐日净值最大差为零；另计算56条最终或下游诊断账户、66条资金／起点敏感性账户、396条固定入场的单周期账户。22条重建与两条最终账户复核属于验证，不能计为新策略。独立只读核验覆盖255,191条账本行和36行配对区间结果。零新模型拟合，零新增严格前瞻日。

## 1. 回测数字和利润集中度

{metric_table}

所有区间使用20万元初始资金、242日年化、现金及无风险收益0，包含现金日。主历史1624日，较早历史1219日。较早历史终点开盘清算，主历史末日按收盘连续结算；两者不能随意合并成一个账户。372笔是两区间两费用合计，不是372个独立交易机会。

{concentration}

最大周期均为2024年9月25日—10月8日。外部评审的40.36%、42.30%、82.36%、85.48%得到原账本复算支持。第210轮压力成本42.97%／86.83%采用8月14日旧终点，分母不同，不能与当前数值混用。

第210轮既有20日区块重采样，选中简单方案主压力净夏普区间为[{old.sharpe_lower_2_5:.4f}, {old.sharpe_upper_97_5:.4f}]，年化区间为[{old.annual_lower_2_5:.4%}, {old.annual_upper_97_5:.4%}]。本轮直接保留原结果，没有重抽样。该诊断针对8月14日原终点，未校正策略选择偏差，不能当成未来达标概率。

## 2. 最大盈利退出的具体原因已经查明

2024年9月25日首次收盘，底层周期内退出模型锁定9月2日的已成熟版本。9月26日预测继续收益为−1.60046%，9月27日为−6.35428%，连续两次为负，底层在9月27日收盘提出退出并于9月30日开盘卖出。

同日滚动线性参考也连续两次为负。第143轮核心目标归零，第165轮连续段辅助目标本来为零。15%的乙来源在9月27日收盘目标归零、9月30日开盘退出；85%的甲来源内含两次归零确认，9月27日第一次零仍保留份额，9月30日第二次零才请求卖出。最终账户9月27日目标仍约84.95%，直到9月30日合成目标为零，10月8日开盘清仓。

因此不能把整个9月30日—10月8日的延迟收益归因于模型准确预测节后顶部。主压力原最大周期全账户收益30.196%；取消上层零确认后的同一固定窗口为13.235%，末次卖出提前到9月30日。该差异同时包含整段历史重演后的仓位与资金状态，不能把两条账户利润元直接相减当成纯延迟因果效应。

9月27日周期内预测中，浮盈标准化项贡献−4.27819个百分点；把该贡献仅作代数扣除后，预测仍为−2.07609%。这支持“浮盈是重要驱动”，同时反对“只靠浮盈一项就能解释全部退出”的过强说法。扣除一项贡献不是重新拟合模型。

完整前五周期的所有节点目标、逐日决定、成交、风险倍率和模型贡献已提供；逐条可从 top5_all_node_decisions.csv、top5_all_node_ledgers.csv、top5_all_targets.csv 和 within_model_prediction_contributions.csv 复核。

## 3. 固定入场后的学习退出增量

以原自然价格退出参考的33个已结束周期固定入场日期；其中10个入场时模型未成熟，23个已成熟（较早6个、主历史17个），另保留1个未结束参考周期。每个周期独立20万元，六方案同一入场日、同一费用下首次份额完全相同；提前退出后保持现金至原自然周期结束，不允许重新进入。这是配对事件诊断，不是把396个局部账户连成可交易组合，也不是把自然周期终点当成事前退出信息。

主历史压力成本结果如下，单位是单周期全账户收益差，不是年化贡献：

{exit_table}

当前入场固定模型为8个改善、5个恶化、4个相同，平均约+0.556个百分点，但区间跨零。全部23个成熟周期合计平均+0.530个百分点，配对区间约[−1.567,+2.456]个百分点，同样不能确认稳定正增量。简单8%止盈的正点值也未形成可靠优势，不据此另选新策略。

这里每次只读当时可用的原模型；不拟合新参数，不把同周期相邻状态随机拆到训练与验证。配对区间使用固定5000次周期级重采样，保存抽样索引；历史模型／策略设计仍使用过这些时期，所以该区间不消除选优偏差。

标签与浮盈在各完整自然周期内去均值后，相关系数中位数为{label.correlation.median():.4f}。实际标签还受下一开盘价格、费用、分红及分母影响，不等于严格的“最终浮盈减当前浮盈”。这一机械关系确实存在于标签结构中，但是否能预测未来退出收益仍必须由上面的后续完整周期验证。

2022年10月至2024年5月共20条月度模型记录，最近成熟退出日一直为2022年9月8日；9月2026年的模型也没有增加新成熟周期。月度拟合记录不等于每月有新信息。2017年3月前无成熟学习模型，不能拿那段表现证明学习退出有效。

## 4. 风险放大、等待与最终账户结果

主历史基础账户有{r.cash_close_days}个收盘空仓日，占{r.cash_close_fraction:.2%}；{r.held_close_days}个收盘持仓日，持仓日平均仓位{r.mean_exposure_held:.2%}，最高{r.maximum_exposure:.2%}，80%以上{r.above80_days}日。全期平均17.74%不能替代持仓时的风险说明。

2025年4月10日基础最终目标99.9055%，当时已知ETF20日年化波动27.4899%，目标乘波动约27.4639%。这是一项加仓决定，不是该周期首次入场。它说明10%不是整个最终账户统一的风险上限；该代理也不是实际未来波动预测。

{module_table}

只降不升、取消退出等待、取消1.15倍放大均使压力年化降至10%以下，而夏普仍高于原方案。原组合通过年化门槛的部分来源可具体归到仓位与等待规则。这个结果支持风险／收益取舍的解释，不证明这些模块没有价值，也不授权采用历史上某项更高夏普对照。

“最终目标波动代理封顶10%”仍保留原带宽和下一开盘执行；它并非逐刻实际风险硬上限。简单对照实际波动也不完全相同，表内明确展示波动，不能仅凭回撤大小比较。来源甲与乙主历史日净收益相关约0.742，并共享核心与连续段机会；85%／15%不能表述为两个独立Alpha来源。

## 5. 成本、资金规模、冷启动及恢复

{cost_table}

基础请求份额原封不动改用压力成本，主历史6行、较早5行申请受实际约束；纯粹固定成交份额的费用恒等式会使假设现金最低分别达到−6,446.82元、−2,607.73元，因此不能把那条不受资金约束的现金流当可执行回测。

较早历史原基础年化10.8809%、夏普1.2287；固定基础目标改压力成本后为10.4134%、1.1805；原整个系统压力路径则为11.0310%、1.2395。故“高成本较早表现更好”来自状态／目标路径变化足以抵消费用损失，不能解释为提高成本本身有利。2019年5月31日同一入场机会：{early_text}。各可用早期节点直接决定和原路径缺项已分别保存。

完整27节点、22账户的资金敏感性如下，历史训练模型保持原样，各内部账户的资金一同改变：

{capital_table}

总体绩效在这三个资金点上接近，但10万元情景最大单日目标差超过83个百分点，说明“总收益相近”不能替代日常状态分叉核验。原模型训练样本仍按原已冻结研究保留，本项没有重新按资金规模训练标签或选择策略。

2021年1月4日重新以现金启动的压力结果为年化7.9957%、夏普0.9515。公平比较同一2021—2026区间，保留原历史状态的账户为年化8.5485%、夏普1.0521；两者本来都低于完整主历史目标。差异来自旧持仓／风险窗口／账户状态与起点，不把缺少2020年盈利误称为恢复程序错误。

原第215轮22账户分段恢复精确一致回执直接保留。本轮另在2024年9月27日真实持仓状态序列化最终账户并恢复，份额、现金、净值及成交与原完整路径一致。冷启动和从完整checkpoint恢复是不同操作。

## 6. 执行证据与故障处理缺口

10月8日4.656元开盘价在原日线文件、整理后的行情和新延伸价格表一致；原来源为akshare.fund_etf_hist_sina，实际获取时间2026年8月16日。这三份是同一数据来源链，不能当三份独立行情证明。日成交量无法证明当时开盘能成交68,300份或69,800份。

仅把最大周期最后一笔卖出的价格从原开盘换成同日收盘，固定份额压力净利润减少16,658.53元；换成下一交易日开盘减少25,329.16元。这两项只有日线支持，属于价格敏感性；历史分钟VWAP、开盘集合竞价可成交量、委托方式和队列证据均为 BLOCKED_NO_HISTORICAL_MINUTE_AUCTION_ORDER_EVIDENCE。

底层硬止损与最终风险的差别也有真实例子：2024年9月11日底层触发6%止损、计划9月12日退出，最终账户9月11日收盘仍有风险，首次归零到9月20日，中间多经历4个交易日。其他来源可能同时持仓，不能把4天全部归因于两次零确认。

当前未知目标会保持份额并重置零确认计数；确定性序列“正、零、未知、零、零”的计数是0、1、0、1、2。最终规则没有单独的持续故障时间上限，也没有把底层硬止损与普通零目标设置为不同优先级。这些是尚需另行设计并验证的运营规则，本轮没有改变已冻结策略。

上交所官方原文已核实基金收盘集合竞价于2026年7月6日实施：[上交所修订发布交易规则](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml)。包内保存原网页及获取回执。制度变化作为预先已知断点有必要记录；本轮前后统计只作描述，短后段不能识别因果影响。评审文字中的“上海电力公司”是链接标签错误，实际来源是上交所。

## 7. 下一步和停止条件

优先保留当前固定版本和已经存在的9月13日研究起点。第一次有新完整日线后，用事前留存的决定时钟继续，现金日、亏损日、迟到日全部保留并标记。现有入口与自动任务没有被本轮更改；严格前瞻仍为0日。

学习退出的下一项证据应是新完整周期的配对增量，比较原价格退出、原线性模型、当前周期内模型和本轮已固定的简单退出。已有主历史6年多只有17个成熟可比自然周期，不能把几十个交易日当足量独立验证。建议20个新的完整自然周期作为首个正式复核点，同时承認它仍未必足以证明长期稳定。

20个新周期后若学习模型相对价格退出的压力平均增量不为正，应停止继续增加该学习退出的复杂度；若点值为正但区间跨零，只能保留“未确认”，不能升格。年化10%及夏普1.2仍保留为完整账户目标；时间／风险口径、执行缺项和独立证据必须同时交代，不能换窗口救结果。出现未来模型、未成熟训练周期、账户恢复不一致或无法解释的状态分叉时停止该条绩效解释，先定位原因。

执行方面优先取得未来真实可验证的开盘观测以及研究层逐笔申请／模拟成交回执，历史关键开盘证据取不到就保留阻塞状态。订单／券商／实盘不在本轮授权范围。硬退出与缺失数据的运营方案应在应用前单独冻结、验证，不作为调高旧回测的新因子。

## 8. 本轮实现错误、验证和边界

本轮发现并修正两处诊断实现错误：风险倍率限制首次把明确零目标乘空倍率变成未知，影响12条诊断账户；附加纯费用现金流表首次把份额数量误传为方向，影响两张附加表。两次错误输出及当时代码已保留在 failed_attempts，并从最终统计排除。它们不是原策略的错误，也未改变原账本。风险语义增加两项回归测试；费用表用独立方向／价位／最低佣金公式逐笔核验。

全部原输入身份保持不变。原策略22账户重建、两原最终账户复算、持仓恢复、全部有效反事实账本、配对索引和纯费用表均已通过保存结果核对。这证明所列数字可由所列代码和输入算出；不证明真实成交、未校正选择偏差消失或未来稳定达标。

本次不计算折减夏普或有效独立试验次数：第209轮前读过308条不同账户路径及多轮候选，但308条路径也不能直接当308个独立统计检验。缺少完整相关性与淘汰过程的有效试验模型，保留 NOT_COMPUTED，不把第209轮编号当独立尝试次数。
'''
    write('01_诊断报告.md', report.replace('承認','承认'))
    write('00_README_FIRST.md', '''# 510300 最新策略评审诊断复核包

先读 **01_诊断报告.md**，再将 **02_GPT复核提示词.txt** 连同本ZIP提交给你选择的审阅者。

本包包含用户原评审、当前85%／15%策略及全部直接输入、原账户／模型、完整本轮诊断代码和结果、未通过的诊断尝试、证据缺口与下一步。

阅读顺序：

1. 01_诊断报告.md：本轮实际发现和结论。
2. 03_评审问题证据矩阵.csv：每项问题的证据位置、状态和限制。
3. 04_下一轮固定验证安排.md：未来留存、验证口径和停止条件。
4. 05_复现与包边界.md：运行入口及本包实际核验层级。
5. 06_用户请求.txt、07_用户提供的外部评审.txt：原始任务和评审文字。
6. 08_字段与口径说明.md：周期、年化、成本、反事实和错误尝试的含义。
7. project/reports/research/510300_strategy_review_diagnostics_v1/：全部诊断数据。
8. project/reports/research/510300_september_monthly_continuation_v1/：当前原策略22账户及模型。
9. project/reports/research/510300_incremental_selected_intent_mix_v1/：第209轮来源、候选及原早期账户。
10. FILE_INDEX.csv：全包成员大小及SHA-256；此表本身不自我哈希。

有效历史诊断已完成；严格前瞻0日；关键开盘盘口／队列／可成交量未验证。本包没有被自动上传或交给外部模型。
''')
    write('02_GPT复核提示词.txt', '''请以独立、严格但建设性的量化研究审阅者身份审阅本ZIP。对象是510300的SELECTED_MIX_BAND10_SIMPLE2：85%与15%来源合成，属于观察过历史后选出的研究方案。请先读00_README_FIRST.md和01_诊断报告.md，再按FILE_INDEX.csv定位原账户、模型、协议和本轮实际诊断。

请逐项验证：
1. 四个原账户、372笔模拟成交、全日历收益及前五盈利周期贡献是否成立；2024年9月27日模型退出、上层两次零确认、9月30日与10月8日成交的原因是否被正确区分。
2. 固定入场的396个局部账户是否构成公平配对；33个已结束自然周期、23个成熟周期、主历史17个成熟周期是否分类正确。入场固定周期内模型约+0.556个百分点的主压力单周期平均增量及跨零区间，能否支持任何强于“未确认”的结论。不要把396个账户、790个状态或372笔成交当独立样本。
3. 标签的周期内机械关系、各特征预测贡献与实际退出增量是否被混淆；当前预测剔除浮盈贡献仍为负，是否足以推翻“纯止盈”的强说法。
4. 风险只降不升、取消零等待、取消1.15倍的反事实是否只改变指定机制，是否正确传播明确零和未知；来源高度相关是否削弱独立Alpha解释。请检查failed_attempts中保留的两次诊断错误是否已隔离，修复是否充分。
5. 固定请求压力执行、固定目标压力自有账户、原完整压力路径，以及纯摩擦现金流不可执行状态，是否被正确区分。资金大小和冷启动诊断是否使用公平区间；恢复核验能证明什么。
6. 买入持有、20%配置、ETF波动目标10%是否采用一致分红／费用／时钟；实测风险不同的比较应如何解释。
7. 关键开盘4.656元的原始来源链、替代价格敏感性、缺少分钟／委托／队列证据、2026年7月6日制度变更和未知持续风险，哪些已核实、哪些仍被阻塞。
8. 本包是否夸大了重建通过、bootstrap、历史点值、独立验证或严格前瞻。第209轮编号和308条收益路径不等于独立尝试次数；请不要制造DSR数值。

请输出：A. 按严重程度排序的具体错误或遗漏，逐条给出文件／字段／计算依据；B. 对每项外部批评判定支持、部分支持、反证或证据不足；C. 哪些简单、可重复的决策值得保留研究；D. 下一轮最少且有区分力的工作，明确优先级、独立样本单位、通过标准与停止条件；E. 应停止投入的方向。若认为应修改研究策略，明确新的事前协议，不能从本包既有历史挑最优结果并称独立验证。

请用中文。本包只授权研究复核，不授权真实仓位、订单、Paper／Shadow、券商或实盘。
''')
    evidence=[
        ['收益集中','CONFIRMED','account_metrics.csv；all_complete_cycles.csv','最大周期与前五比例已由原完整账本复算'],
        ['学习退出稳定增量','NOT_ESTABLISHED','paired_exit_summary.csv；paired_exit_cycles.csv','主历史成熟自然周期17个，当前模型平均增量的区间跨零'],
        ['模型就是单一止盈规则','NOT_PROVEN','within_model_prediction_contributions.csv；label_within_cycle_relationship.csv','浮盈贡献重要，但剔除该项后关键日预测仍为负'],
        ['风险预算多次放大','CONFIRMED','counterfactual_account_metrics.csv；all_decision_risk_proxies.csv','固定模块移除与目标风险代理；没有证明未来波动'],
        ['成本改变退出路径','CONFIRMED','cost_path_constraints.csv；early_cost_divergence_all_available_nodes.csv','严格区分固定请求、固定目标及全系统压力'],
        ['资金规模决定一切','NOT_SUPPORTED_IN_TESTED_RANGE','full_graph/CAPITAL_100K/result.json；full_graph/CAPITAL_1M/result.json','总体指标接近但个别目标分叉显著；未重训模型'],
        ['断点恢复错误','NOT_REPRODUCED','held_resume_receipt.json；full_graph_reconstruction_checks.json','持仓恢复与22原账户重建通过；冷启动不是恢复'],
        ['最终层硬退出／未知上限','GAP_CONFIRMED','hard_exit_to_final_flat_delays.csv；unknown_state_semantics.json','尚未实施最终故障上限或退出原因优先级'],
        ['开盘可成交数量和VWAP','BLOCKED','sources/20241008_open_price_lineage.json；opening_price_sensitivity.csv','三份同源日线不能证明独立报价、队列或分钟成交'],
        ['收盘制度变化','OFFICIAL_SOURCE_CONFIRMED','sources/sse_rule_receipt.json；close_rule_regime_descriptive.csv','正式实施日期已核实，策略影响只作描述'],
        ['历史选优／折减夏普','SELECTION_CONFIRMED_DSR_NOT_COMPUTED','第209轮配置和结果；本报告第8节','完整有效独立试验数未建立'],
        ['严格前瞻','ZERO_NEW_DAYS','原fixed_research_origin_v1/origin.json','9月13日研究起点已存在，旧历史没有改记成实时样本']]
    pd.DataFrame(evidence,columns=['问题','判定','本轮目录内证据路径','边界']).to_csv(DELIVERY/'03_评审问题证据矩阵.csv',index=False,encoding='utf-8-sig')
    write('04_下一轮固定验证安排.md', '''# 下一轮固定验证安排

优先级一：保持原85%／15%方案、研究起点和费用协议，继续新完整日的真实事前留存。原起点位于project/reports/research/510300_fixed_research_origin_v1/origin.json。9月13日没有新增完整交易日，本轮没有新建或修改自动任务。

每条新增观察绑定：实际保存时间、经济信号日、决定截止时间、下一开盘日期、来源首次可得时间、原响应路径／SHA-256、27节点目标与未知状态、模型身份／训练成熟日、22账户checkpoint前后身份、模拟请求与实际模拟成交、整手／资金受阻原因、费用与分红。历史事后重建必须单列，不能替代事前文件。

优先级二：以原自然价格退出周期为唯一比较单位，保留原价格、原线性、入场固定周期内，以及本轮固定20日／8%退出对照。全部使用同一入场日与经济起点；模型不可含当前尚未结束周期。当前报告完成了历史诊断，未来对照还没有产生新完整周期，不能把设置文件当未来结果。

首个正式复核点建议为20个全新、入场时模型成熟且已完整结束的自然周期；同时列全部未成熟和未结束周期。此建议尚未替换原运行入口，也不代表20个周期足以证明稳定。以原价格退出为唯一主增量比较，其他对照只用于解释。压力平均增量不为正则停止继续增加学习退出复杂度；点值为正但区间跨零，保持未确认。不得按盈利周期子集或更换终点救结果。

优先级三：以未来可验证的开盘观测补执行证据。保留预定委托方式、截止时间、可成交量／未成交处理及完整账户影响；历史2024年10月8日分钟、队列证据缺失继续标BLOCKED。日线低点、收盘或下一开盘仅作敏感性，不能叫真实VWAP或实际成交。

优先级四：另外冻结硬退出与持续资料缺失的运营规则，明确优先级、有效期、人工处置条件及其研究验证。当前策略尚无最终故障上限，不能宣称底层6%止损等于最终账户6%止损。没有运营验证前不写入当前策略。

所有新历史诊断更优结果均不自动升级。完整账户年化10%、净夏普1.2的目标保留；独立验证要另外成立。发现未来数据、成熟周期不合法、checkpoint恢复不一致、来源身份改变或不可解释的资本／费用分叉时，暂停受影响绩效解释并定位原因。当前无真实订单授权。
''')
    write('05_复现与包边界.md', '''# 复现与包边界

代码和数据按原项目相对路径放在project/下。Python依赖为numpy、pandas、pyarrow、scipy、scikit-learn；图表生成额外需要matplotlib。包内有环境版本回执，不包含Windows虚拟环境。

优先使用只读核验：在解压后的project目录运行 python scripts/verify_510300_strategy_review_saved.py 。该入口复算255,191行有效账本、保存配对抽样索引和纯摩擦现金流；不生成新账户、不训练、不下载、不抽新随机样本。不要运行旧研究的prepare／run／verify／finalize。

本轮代码入口为research/strategy_review_diagnostics_v1.py（freeze、basic、exits、accounts）、research/strategy_review_full_graph_v1.py（两个资金情景、一个冷启动和22原账户重建），以及research/strategy_review_followup_evidence_v1.py。应在独立拷贝中重演，避免覆盖本包已保存结果。原冻结文件不能修改；首次无效诊断及修复说明保留在failed_attempts。

包根FILE_INDEX.csv为本包唯一成员索引，列出除其自身外全部文件的字节数及SHA-256。ZIP进行CRC、重复成员、索引覆盖、大小与哈希核验；另有保存结果经济复算。这些不是外部GPT结论，也不是数据安全审计或真实市场成交验证。

包含：本轮全部有效／无效诊断输出；当前27节点配置、22原账户及模型；第209轮候选和第210轮既有稳定性诊断；日线、分红与相关获取／更正回执；前瞻起点／恢复回执；原始评审与可复制提示词。历史288等其他未用研究、虚拟环境、其他资产与整库缓存不在本包范围。配置中的旧冻结清单可能指向其他历史研究，本包未宣称能重跑全部历史搜索；代码入口的直接读取覆盖以packaged_source_coverage.json为准。

缺项：历史分钟VWAP、开盘队列／委托／可成交数量；未选择过的新完整绩效周期；有效独立策略试验数。因此全选择流程独立复现、策略选择偏差校正及实盘稳定达标均未建立。
''')
    write('06_用户请求.txt','用户最初请求：关于我们最新的策略，510300。\n用户提供完整外部策略评审。\n用户明确选择：直接开展评审提出的诊断，并打包结果。\n语言：简体中文；Windows；减少不必要的安全性审计。\n')
    shutil.copy2(Path('E:/CodexData/.codex/attachments/3ab94d9c-1070-4a05-ab27-d0ae23351915/pasted-text.txt'), DELIVERY/'07_用户提供的外部评审.txt')
    write('08_字段与口径说明.md','''# 字段与口径

- annual_return：完整日历按242日年化的复合账户收益，原始数值0.10代表10%。sharpe：现金／无风险0，日净收益均值除标准差再乘根号242。
- complete cycle：从实际首次买入到全部退出的完整账户周期；final账户主历史43个，和固定入场参考的17个成熟自然周期属于不同定义。
- paired mean_increment：每个完整自然周期等权的全账户收益差，0.005代表0.5个百分点；不是每笔交易胜率、年化贡献或未来概率。
- target_vol_proxy：决定时点目标仓位乘已知ETF20日波动；不是实现波动，也不是未来波动或硬上限。
- BASE_REQUESTS_STRESS_COST：固定基础请求数量，以压力费用运行真实资金约束；BASE_TARGETS_STRESS_COST：固定基础目标比例，以压力账户重算自身份额。
- pure_friction_fixed_fills_cashflow：成交数量假定不变时的费用现金流恒等式。若假设现金变负，该路径不可执行，不作收益推荐。
- NO_INTERNAL_UPSCALE：来源完整账户风险倍率只降不升，上游信息固定；明确零仍为零，未知仍为未知。首次错误版本保留但无效。
- top5_fixed_window_counterfactuals：在原前五盈利周期相同起止窗口度量各反事实账户收益，资金起点不同；元差不能直接作纯单模块因果贡献。
- frozen／reconstructed：规则或保存状态可定位、原数值可重建；与事前真实可得、独立验证、实际成交是不同证据。
- failed_attempts：本轮诊断实现错误及当时输出，全部排除最终有效统计；两个错误都不是原策略错误。
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    fig, ax=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1.05,1]},layout='constrained')
    names=['当前策略','倍率只降不升','取消归零等待','取消1.15倍']
    values=[main_stress.annual_return]+[chosen[chosen.scenario.eq(k)].annual_return.iloc[0] for k in ['NO_INTERNAL_UPSCALE','NO_ZERO_WAIT','NO_115_MULTIPLIER']]
    bars=ax[0].barh(names[::-1],np.array(values[::-1])*100,color=['#7c93aa','#497894','#276576','#ba7228'])
    ax[0].axvline(10,color='#943c36',ls='--',lw=1.4,label='原年化目标10%')
    for b,v in zip(bars,values[::-1]):ax[0].text(v*100+.08,b.get_y()+b.get_height()/2,f'{v:.2%}',va='center',fontsize=11)
    ax[0].set_xlim(0,12);ax[0].set_xlabel('完整主历史压力成本年化收益（%）');ax[0].set_title('达到年化目标依赖风险与等待规则',fontweight='bold',pad=15)
    selected=case[case.variant.isin(['RIDGE_MONTHLY','WITHIN_ENTRY_FIXED','TIME20','PROFIT08'])]
    labels=['线性退出','当前周期内退出','20日时间上限','8%止盈']
    for i,(r,label) in enumerate(zip(selected.itertuples(),labels)):
        ax[1].plot([r.delta_q025*100,r.delta_q975*100],[i,i],color='#497894',lw=3)
        ax[1].scatter(r.mean_increment*100,i,color='#ba7228',s=65,zorder=3)
    ax[1].axvline(0,color='#943c36',ls='--',lw=1.4);ax[1].set_yticks(range(4),labels);ax[1].invert_yaxis()
    ax[1].set_xlabel('相对原价格退出的单周期收益差（百分点）');ax[1].set_title('17个成熟自然周期：增量区间均跨零',fontweight='bold',pad=15)
    for a in ax:
        a.spines[['top','right']].set_visible(False);a.grid(axis='x',alpha=.15);a.set_axisbelow(True)
    fig.suptitle('510300 固定策略诊断｜历史反事实，不是独立前瞻',fontsize=17,fontweight='bold')
    fig.savefig(DELIVERY/'诊断图表.png',dpi=160)
    plt.close(fig)
    print('中文报告、复核提示、证据矩阵与诊断图表已生成。')


if __name__=='__main__':
    main()
