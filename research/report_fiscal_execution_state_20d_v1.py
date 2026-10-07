"""用已保存结果生成财政、融资状态和方向对照图及中文报告。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_fiscal_execution_state_20d_v1'
REV=ROOT/'reports/research/510300_reversal_monthly_diagnostic_v1'


def main():
    s=json.loads((OUT/'results/summary.json').read_text(encoding='utf-8'))
    rs=json.loads((REV/'summary.json').read_text(encoding='utf-8'))
    d=pd.read_parquet(OUT/'results/每日已知财政融资与价格.parquet');d['date']=pd.to_datetime(d.date)
    plotted=d[(d.date>='2024-01-01')&(d.date<='2026-07-31')].copy()
    money=pd.read_csv(ROOT/'reports/research/510300_macro_dynamic_reframe_v1/inputs/money_104.csv')
    money['known_at']=pd.to_datetime(money.available_at_upper_bound,utc=True).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None).astype('datetime64[ns]')
    money=money.sort_values('known_at');plotted['decision_at']=pd.to_datetime(plotted.decision_at).astype('datetime64[ns]')
    plotted=pd.merge_asof(plotted,money[['known_at','m1_yoy_pp','m2_yoy_pp','training_regime']],left_on='decision_at',right_on='known_at',direction='backward')
    plotted.to_csv(OUT/'results/宏观走势对比图全部日频点.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(4,1,figsize=(15,12),sharex=True,gridspec_kw={'height_ratios':[1.7,1.1,1.2,1.0]},facecolor='#f6f7fb')
    a=axs[0];a.plot(plotted.date,plotted.close,color='#163d60',lw=1.55,label='510300 未复权日收盘价');a.set_ylabel('元');a.legend(loc='upper left',frameon=False);a.set_title('价格：信息到达时已经发生的上涨，不属于下一开盘之后的收益',loc='left',fontsize=12,pad=9)
    for word,idx in [('区间低点',plotted.close.idxmin()),('区间高点',plotted.close.idxmax())]:
        r=plotted.loc[idx];a.scatter([r.date],[r.close],s=22,color='#163d60');a.annotate(f'{word} {r.close:.3f}\n{r.date:%Y-%m-%d}',(r.date,r.close),xytext=(30,-22) if word=='区间低点' else (-90,20),textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':'#8e99a9','lw':.7},bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    a.margins(y=.19)
    a=axs[1]
    for j,(regime,g) in enumerate(plotted.groupby('training_regime',sort=False)):
        a.step(g.date,g.m1_yoy_pp,where='post',color='#276eaa',lw=1.4,label='M1 同比（新旧口径断开）' if j==0 else None)
    a.step(plotted.date,plotted.m2_yoy_pp,where='post',color='#d4883c',lw=1.3,label='M2 同比');a.axhline(0,color='#9ca3af',lw=.8);a.set_ylabel('同比 %');a.legend(loc='lower left',bbox_to_anchor=(0,1.0),ncol=2,frameon=False,fontsize=10);a.set_title('货币：按公开时点更新；未知的财政共识不填成零预期差',loc='left',fontsize=12,pad=31)
    a=axs[2];a.step(plotted.date,plotted.general_model_yoy_pp,where='post',color='#247e6d',lw=1.5,label='一般公共预算累计支出同比');a.step(plotted.date,plotted.fund_model_yoy_pp,where='post',color='#b85e53',lw=1.5,label='政府性基金累计支出同比');a.axhline(0,color='#9ca3af',lw=.8);a.set_ylabel('同比 %');a.legend(loc='lower left',bbox_to_anchor=(0,1.0),ncol=2,frameon=False,fontsize=10);a.set_title('财政执行：两类支出分开；这是累计执行数，不是当月新增刺激规模',loc='left',fontsize=12,pad=31)
    a=axs[3];a.plot(plotted.date,plotted.funding_spread20_pp,color='#836398',lw=1.35,label='已知 DR007 − 操作利率，过去20日平均');a.axhline(0,color='#9ca3af',lw=.8);a.set_ylabel('百分点');a.legend(loc='upper left',frameon=False);a.set_title('融资条件：每天只使用此前已经公布的信息；不把央行操作量当净流入股市',loc='left',fontsize=12,pad=9)
    for a in axs:
        a.grid(alpha=.14);a.set_xlim(plotted.date.min(),plotted.date.max())
        for dt in ['2024-09-24','2025-05-07']:a.axvline(pd.Timestamp(dt),color='#8e99a9',lw=.8,alpha=.7,ls='--')
    axs[0].text(pd.Timestamp('2024-09-24'),axs[0].get_ylim()[1],' 9·24政策发布',fontsize=9,color='#626b7a',va='top')
    axs[0].text(pd.Timestamp('2025-05-07'),axs[0].get_ylim()[1],' 5·7政策发布',fontsize=9,color='#626b7a',va='top')
    axs[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3));axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle('510300 × 货币、财政执行与融资条件',x=.07,y=.975,ha='left',fontsize=21,fontweight='bold',color='#16314a')
    fig.text(.07,.94,f'2024-01—2026-07 · {len(plotted)} 个交易日全部保留 · 上图描述同步状态，不把走势相似直接当成预测证据',color='#52606d')
    fig.text(.07,.025,'虚线为上一阶段已核对的政策发布日期，仅作背景。财政数据按本页及目录的保守公开日入场；转载日期不等于市场最早获知时间。\n本图价格不是账户净值。新旧M1不拼接共用阈值；预测检验仍使用次日开盘后的固定二十日标签。',fontsize=10,color='#52606d',linespacing=1.6)
    fig.subplots_adjust(top=.89,bottom=.10,left=.07,right=.96,hspace=.55)
    for ext in ['png','svg']:fig.savefig(OUT/f'figures/510300_货币财政融资_走势对比.{ext}',dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)
    metrics=pd.DataFrame(s['metrics']).set_index('model');comps=pd.DataFrame(s['comparisons'])
    fig,axes=plt.subplots(1,2,figsize=(15,7),facecolor='#f6f7fb',gridspec_kw={'width_ratios':[1.15,1]})
    keys=['P0','P1','P2','P_REV','P_MOM','MEAN','ZERO'];labels=['价格＋融资基准','＋两类财政执行','＋财政×融资交互','只允许均值回归方向','只允许动量方向*','成熟样本均值','零收益预测']
    a=axes[0];values=metrics.loc[keys,'rmse_pp'];bars=a.barh(labels,values,color=['#7790aa','#257e6d','#439a89','#bf674f','#918298','#aaa8af','#c8c9cd']);a.invert_yaxis();a.set_xlim(0,6.5);a.set_xlabel('二十日收益预测 RMSE（百分点，越小越好）');a.set_title('88 次共同财政评价原点，全部方法都报告',loc='left',fontsize=12,pad=16)
    for bar,v in zip(bars,values):a.text(v+.05,bar.get_y()+bar.get_height()/2,f'{v:.4f}',va='center',fontsize=10)
    a=axes[1];subset=comps.set_index(['candidate','reference']).loc[[('R1','R0'),('R2','R0'),('R2','R1')]].copy();names=['财政执行 vs 风险基准','财政＋交互 vs 风险基准','增加交互 vs 仅财政']
    for i,(_,row) in enumerate(subset.iterrows()):
        mean=row.mean_improvement;lo=row.one_sided_98_75pct_lower;hi=row.two_sided_97_5pct_upper;a.errorbar(mean,i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color='#836398',capsize=5)
    a.set_yticks(range(len(names)),names);a.invert_yaxis();a.axvline(0,color='#777',ls='--',lw=1);a.set_xlabel('QLIKE 损失改善（右侧为改善）');a.set_title('下行风险：区间跨零，未建立可靠增量',loc='left',fontsize=12,pad=16);a.grid(axis='x',alpha=.15)
    fig.suptitle('财政信息有轻微点值改善，尚未通过预定验收',x=.07,y=.965,ha='left',fontsize=20,fontweight='bold',color='#16314a')
    fig.text(.07,.885,'2018-06-15—2026-07-23 · 固定参数 · 四个候选分别判定 · 每项比较使用连续六次财政更新区块',color='#52606d')
    fig.text(.07,.04,'* 动量对照在这些评价时点的斜率均为零，预测等于成熟均值，不能称为“动量获胜”。\n区间展示97.5%双侧范围；验收使用98.75%单侧下界，保守分配四候选的错误额度。账户和卖点比较未运行。',fontsize=10,color='#52606d',linespacing=1.6)
    fig.subplots_adjust(top=.77,bottom=.22,left=.16,right=.96,wspace=.70)
    for ext in ['png','svg']:fig.savefig(OUT/f'figures/财政执行_收益风险与均值回归对照.{ext}',dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)
    report=f'''# 本轮结论

新增的财政执行和融资状态关系，尚未建立可靠的510300收益或下行风险增量。两类财政执行使收益MSE相对价格加融资基准下降0.2714%，加一个交互后合计下降0.4246%，幅度很小、区块下界未通过，且收益预测仍落后于成熟训练样本均值。本轮固定实现终止，不能据此关闭财政或全部宏观研究。

用户提出“A股反转多于动量”。独立的固定月末诊断得到170原点：81反转、88延续、1后续零收益；反转比例47.92899%，连续六个月区块95%区间41.17647%—55.02959%，过去和未来二十日回报相关系数−0.006870，区间−0.213475—0.196660。它不支持在本期限上直接预设反转明显占优。个股横截面反转研究不等于510300时间序列择时证据；反转频数也不是均值平稳性、合理价值或净账户盈利的证明。

## 信息与检验范围

财政部月度原文从2015年2月到2026年7月，共127所属期，一二月合并、不拆分。11份2015年数据按当次明确公布的同口径增速处理，避免11项政府性基金转入一般公共预算造成机械增速。来源中119份为HTTP原文，8份为网页工具完整正文，失败尝试均保存；没有逐份证明网页从历史首发以来从未修改，不能称严格历史版本认证。

字段分别保留一般公共预算和政府性基金累计支出金额、表面同比、同口径同比、公开日、来源和摘录。没有获得财政调查共识，预期及预期差为空，不能称“财政超预期”。累计执行包含工资、民生、付息等结构，不是纯新增需求、企业到账利润或净股票资金。

融资状态为决策以前已知的DR007与已公布7天逆回购操作利率之差的20交易日均值。DR007继承供应商历史数据，按下一交易日使用；实际操作记录不等于最早政策宣布全集。财政增速与融资状态的一项交互为本轮独立的经济条件假设，不复试已失败的操作数量模型。

财政本页、正文、目录的明确公开日取较晚值，统一在当日23:59:59后可用。2019年7/8月及2024年6/7月两组转载日期映射共同收盘，原文全部保留，按协议同日复核一次、状态取最新所属月。7月2026原文在8月21日公布，晚于本实验7月31日判断截止，不进入评价。最终124个财政训练原点、701个周度/公告判断原点，497个产生预测，88个主要评价，704次保存模型估计。

首次执行在时间列精度不同处失败；修复后又在共同收盘原点处拦截。两次均在任何收益标签和拟合之前停止。后续仅统一时间精度、实现协议原有同日去重；原协议、代码、启动和更正记录全部保留。实际只完成一套固定研究结果。

## 完整收益对照

| 方法 | RMSE，百分点 | 含义 |
|---|---:|---|
| 价格＋融资 P0 | {metrics.loc['P0','rmse_pp']:.6f} | 固定控制基准 |
| 加两类财政 P1 | {metrics.loc['P1','rmse_pp']:.6f} | 未通过预定门 |
| 再加交互 P2 | {metrics.loc['P2','rmse_pp']:.6f} | 未通过预定门 |
| 只允许反转 P_REV | {metrics.loc['P_REV','rmse_pp']:.6f} | 成熟训练、斜率非正 |
| 只允许动量 P_MOM | {metrics.loc['P_MOM','rmse_pp']:.6f} | 这些评价时点斜率均零 |
| 成熟样本均值 | {metrics.loc['MEAN','rmse_pp']:.6f} | 简单基准 |
| 恒定零收益 | {metrics.loc['ZERO','rmse_pp']:.6f} | 简单基准 |

反转对照相对成熟均值没有更好的点值；不能因为动量对照与均值一样，就称动量策略成功。所有模型只用已成熟财政标签，未来二十日从下一可成交开盘起算，分红作为现金权益、不再投资。

## 风险与动态更新

R1相对R0的QLIKE平均改善为−0.017840，R2相对R0为−0.014572；均未建立正增量。风险目标独立于收益MSE判定，不用改变指标救回收益失败。

81个非周末公告复核原点进行了共同起点诊断：最新财政/融资输入与上个周末状态使用同一当时价格、同一模型、同一后续二十日标签。P1有45次更接近结果，但平均MSE反而增加；P2为44次，平均也恶化。R1有46次、R2有50次风险误差改善，平均QLIKE各改善0.005285/0.008943；这是已预定的诊断，不能替代未通过的主要风险门。20万元入场成本阈值的推导中，P1仅1次、P2仅2次改变是否入场；2万元结果相同。没有连续账户成交。

## 当前能做与不能做的结论

新信息可能导致延续，也可能诱发暂时注意力买盘后的回落；目前证据还不能可靠识别哪一种将在510300出现。政策时间轴、货币预期差、财政执行、市场状态须分别保留其含义。下一项独立假设应围绕真实资金调整或政策消息的持有价值传导，不继续修改这次财政窗口、方向、模型或均值回归阈值。

完整20万元账户、2万元成本对照和“上涨较多后再根据回撤退出”比较，均为NOT_RUN_OWN_PREDICTION_GATE，不写为零收益。既无买卖建议，也没有年化10%、夏普1.2或独立验证成立的结论。主研究仍有政策预期、实时更新账户及真实资金行为等未完成要求。

## 图表

![货币财政融资走势](figures/510300_货币财政融资_走势对比.png)

![财政模型对照](figures/财政执行_收益风险与均值回归对照.png)

## 文献仅提供机制动机

- [Li等：A股动量与反转随时期变化，大学存档及论文](https://centaur.reading.ac.uk/109131/)。
- [Yu等：周频个股赢家输家组合中的反转](https://www.sciencedirect.com/science/article/pii/S1059056018301928)。
- [时间序列动量与反转：指数研究发现结果取决于期限等条件](https://arxiv.org/abs/1702.07374)。
- [Dissecting Momentum in China工作论文：新闻日与非新闻日回报的抵消、注意力价格压力和新闻信息反应不足](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5130681)。其个股、工作论文证据不能替代本ETF验证。

## 阅读与复核

原文及失败尝试在raw与evidence/acquisition；protocol.json和freeze.json给出固定设定；results保留全部原点、完整日频、参数、训练集合、逐事件损失、按年贡献和固定抽样索引。verify_fiscal_execution_state_20d_v1.py仅复算保存证据，不下载、不拟合、不生成账户或新随机样本。压缩包结构核对与保存结果复算不等于外部GPT已经审阅，也不证明完整策略有效。
'''
    (OUT/'研究结论.md').write_text(report,encoding='utf-8')
    print(json.dumps({'图表日频点':len(plotted),'模型对照原点':s['evaluation_events'],'状态':'已保存图表与结论，无新增研究拟合'},ensure_ascii=False))


if __name__=='__main__':main()
