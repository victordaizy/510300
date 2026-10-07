"""保存解释、复核已保存数值并打包；不新增拟合或账户。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
REL=Path('reports/research/510300_private_manager_intent_v1')
OUT=ROOT/REL
ZIP_NAME='510300_私募操作意向与可加仓空间_V1_GPT审阅_20260922.zip'
PREVIOUS='510300_一月资金行为与持有验证_V1_GPT审阅_20260922.zip'
PREVIOUS_SHA='fe987757d7baab36a9c4656e0c5700079154e1f96431eaf7f0fdba3c92da6fea'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')


def prepare():
    s=read(OUT/'summary.json');d=pd.read_parquet(OUT/'逐事件特征与随后收益.parquet');p=pd.read_csv(OUT/'固定逐时点预测.csv')
    z=d.dropna(subset=['intent_headroom'])
    fair={
        'role':'结果解释所需的同样本描述和预测符号复核；不是新增候选或通过门槛',
        'matched_months':z.month.tolist(),'n':len(z),
        'intent_spearman_same_sample':float(spearmanr(z.intent,z.forward_return20).statistic),
        'intent_headroom_spearman_same_sample':float(spearmanr(z.intent_headroom,z.forward_return20).statistic),
        'direction_changes':{},'new_fits':0,'new_accounts':0,
    }
    for name,base in [('B','A'),('C','AC')]:
        j=p[p.model.eq(name)].merge(p[p.model.eq(base)],on='month',suffixes=('_candidate','_base'))
        fair['direction_changes'][name]=dict(n=len(j),changes=int((np.sign(j.prediction_candidate)!=np.sign(j.prediction_base)).sum()),
                                            positive_candidate=int(j.prediction_candidate.gt(0).sum()))
    dump(OUT/'同样本与预测符号解释.json',fair)
    pd.read_parquet(OUT/'inputs/market.parquet').to_csv(OUT/'inputs/510300完整日线与分红.csv',index=False,encoding='utf-8-sig')
    previous=ROOT/'deliverables'/PREVIOUS
    assert digest(previous)==PREVIOUS_SHA
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    prefix='reports/research/510300_participant_identity_clock_v1/'
    keep={
        'january_case_summary.json':'一月案例摘要.json',
        'facts/2026年1月行为回放全日线.csv':'一月完整57日.csv',
        'figures/2026年1月_价格成交份额与政策对照.png':'一月资金与价格对照.png',
        '研究结论.md':'前轮一月案例结论.md',
        '资金行为预测任务书.md':'前轮预测任务书.md',
        'evidence/日频份额方向旧失败.json':'日频份额方向旧失败.json',
        'evidence/季度申赎旧失败.json':'季度申赎旧失败.json',
        'evidence/融资压力旧失败.json':'融资压力旧失败.json',
        'evidence/反转固定诊断.json':'反转固定诊断.json',
    }
    with zipfile.ZipFile(previous) as archive:
        for src,dst in keep.items():(evidence/dst).write_bytes(archive.read(prefix+src))
    dump(evidence/'前轮包身份.json',{'archive':PREVIOUS,'sha256':PREVIOUS_SHA,'scope':'只摘取案例和既有裁决作为背景；前轮原始持有人PDF未在本包重放',
                                   'members':[{ 'source':prefix+k,'current':'evidence/'+v,'sha256':digest(evidence/v)} for k,v in keep.items()]})
    report=f'''# 私募操作意向与可加仓空间：固定历史诊断

本轮把“根据当时的信息预测大资金下一步操作”落实为一项事前调查检验。得到一个有限线索：在相同27个月内，增减仓意向与随后20日收益的秩相关为0.298，意向乘可加仓空间为0.353。但逐时点预测只有8次和3次评价，且没有改变预测涨跌方向；15组相邻调查仓位变化也没有验证意愿会变成持续买入。结论是保留机制线索、尚未建立可交易增量；不能宣布无效，也不能直接跟随。

## 用户的一月判断如何纳入

无需等年报确认身份才研究供给压力。前轮已经完整回放2025年12月至2026年2月57个交易日：1月15日和16日，510300成交额约254亿元、259亿元，均约此前20日中位数的6.46倍；1月14日至28日，份额统计值下降373.15亿份，同段含分红回报约-0.99%。这些数据与已公开持有结构一起，支持研究大型持有者释放供给。份额统计日不自动等于当天可用；已知持有量只能提供跨期界限，不能据此给每笔交易实名。

1月14日上交所公布融资保证金比例不得低于100%，1月19日实施，既有合约及其展期沿用原规定。这是当时可纳入判断的融资约束变化。把这类政策、异常成交和公开份额变化放在一起推断，比只等待持有人表更贴近预测。政策目标、具体账户卖出动机和接下来股价仍分别是需要验证的推断。

关键区别在于：大量供给出现以后，价格为什么只小幅调整，承接能否持续。市场中不同资金可能同时反向操作，不能把“大资金”写成一个统一意志。1月19日除息0.123元已经校正；当日原价-2.49%，含分红约+0.04%，不能把除息误认成继续抛售。

来源：[上交所1月14日通知](https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/specific/margin/c/c_20260114_10805174.shtml)、[1月26日当时报导](https://paper.cnstock.com/html/2026-01/26/content_2174084.htm)。前轮案例是用户指定的已知历史，用于说明预测对象，不是策略证据。

## 本轮新增了什么

固定公开资料检索取得113个来源线索、110个下载成功记录。去重后36个月有计划指数，其中33个月按所保留公开日期参与历史诊断，23个月来自带日期的公开正文，10个月采用原发布方目录日期加当前报告。27个月同时有同一调查的平均仓位，44个月有调查仓位记录。2014年1月至2026年8月152个月的全表保留缺失，不插值、不把重复报道算新事件。

本轮使用两个已在收益读取前固定的量：增减仓意向=(计划指数-100)/100；可加仓空间=1-调查仓位。两者相乘只是愿望和空间的粗略代理，不是资金金额。33个月中31个月指数高于100、1个月等于100、1个月低于100，因此“高于100就买”几乎无法区分时机。

三份有数值但未用于主诊断的记录：2018年8月缺可核对公开日期；2024年7月目录上架日已到8月；2025年8月报告的计划段写成7月，未把月份矛盾默默纠正。所有其他候选和抓取失败链接均保存。周频净值测算仓位没有混入月度调查仓位。

## 固定检验结果

每次从公开日之后的第一个交易日开盘开始，持有20个交易日至收盘；计入持有期间应得分红，入场日除息不享有。先固定期限、字段和模型，再读收益。标签为每份持有毛收益，不是账户收益。

| 比较 | 样本数 | 秩相关 | 含义 |
|---|---:|---:|---|
| 单独意向与随后收益，所有可用月份 | 33 | +0.148 | 弱的描述性正相关 |
| 单独意向与随后收益，与下一行同样本 | 27 | +0.298 | 避免把33与27样本变化当增量 |
| 意向乘空间与随后收益 | 27 | +0.353 | 机制线索，不能当独立因果或预测证据 |
| 意向与下一期调查仓位变化 | 15 | -0.182 | 没有验证意向会稳定兑现为增仓 |
| 意向乘空间与下一期调查仓位变化 | 15 | -0.119 | 同样没有建立行为预测链条 |

只看带日期公开正文的子样本，意向与收益为+0.130（23次），意向乘空间为+0.483（17次）；样本不同，不能直接比较增幅。与下期仓位变化分别为+0.017和+0.109（各9次），又与全样本符号不同，进一步说明行为证据不足。下期平均仓位可能受价格涨跌、调查样本变化影响，即便相关显著也不能直接叫真实资金流预测。

固定模型使用价格的此前20日收益和波动率为基准，加意向为B，加意向、空间及交互为C。标准化和Ridge惩罚10固定；至少24个已成熟标签训练。A/B、AC/C各自使用完全相同的训练和评价月份。

| 固定比较 | 评价次数 | 价格基准RMSE | 新增后RMSE | 成熟均值RMSE | 恒零RMSE |
|---|---:|---:|---:|---:|---:|
| 加意向B | 8 | 2.806 | 2.731 | 2.747 | 3.038 |
| 加意向、空间及交互C | 3 | 3.179 | 2.900 | 3.121 | 1.795 |

单位均为百分点，两个比较的样本不同。B的MSE较价格基准下降5.28%，较成熟均值仅下降1.17%；C的MSE较价格基准下降16.75%，但仍不如恒零预测。两组全部预测为正，方向改变都是0次。因此误差下降尚未体现择时决策差异。未达到预定24次评价及至少3年跨度，不运行显著性重采样、不宣称稳定正增量或稳定负作用。

22条保存模型记录已经逐项复核输入时钟、训练集合、标准化、岭回归方程及预测。保存数据的复核不新增拟合、不新增回测。

## 对后续预测的具体含义

应将资金目标、可操作空间和交易后的价格反应连起来。以下是待检验的条件关系，尚无冻结阈值，不是当前买卖指令：

| 当时可观察的组合 | 可以提出的预测 | 后续推翻条件 |
|---|---|---|
| 融资约束收紧、异常大成交、已公开的份额继续减少、价格走弱 | 卖出供给仍占优势，继续上行更困难 | 供给持续但价格开始稳定或回升 |
| 同样有大额供给，价格却稳定 | 存在较强承接；下一步需预测承接是否持续 | 相近供给下价格明显转弱 |
| 卖出压力减弱，价格保持稳定并转强 | 有条件研究供给衰减后的跟随机会 | 卖压重现或价格跌回已有支撑 |
| 管理人愿意加仓且尚有空间，但宽基供给更强 | 私募意向未必主导指数，不机械看多 | 公开后续行为与盘面共同转为需求占优 |

这些关系可以在实名确认前预测，但应该保存当时判断和失效条件。上面表格不宣称国家队减持结束，也不把卖压被承接等同于接下来必涨。涨多后结合回撤退出仍须在有效入场和持有条件之上检验。

目前最新合格意向只到2026年3月，行情输入到2026年9月11日；本报告没有9月22日的实时资金判断。历史来源缺失不随机、原始调查样本及规模权重未知、历史首版未认证、意向针对全A股主观私募而非专门针对510300。不能据此给出精确身份概率或当前仓位。

本轮账户为NOT_RUN_PREDICTIVE_GATE，严格前向天数0。20万元主账户、2万元成本对照、242日口径保留，夏普1.2且年化10%或夏普1.5目标尚未达成。旧日频流量、季度流量、融资及宏观具体失败和已终止85/15均不重调或复活。本轮资料与数值到此冻结，不围绕正相关换窗口挑结果。
'''
    (OUT/'研究结论.md').write_text(report,encoding='utf-8')
    (OUT/'用户要求.md').write_text('用户要求重点预测大资金会怎么想、怎么操作，然后跟随；很多延后披露只用于验证。用户指出2026年1月国家队放量减仓，要求灵活使用当时信息。\n\n沿用：510300与宏观/资金图形，宏观预期和政策持续更新，涨幅较大后结合回撤退出，20万元主账户与2万元成本对照。最终目标为净夏普1.2且年化10%或净夏普1.5。\n\n本轮是私募独立意向通道的有限检验，不代替国家队、公募、社保和个人全部资金研究，也未完成宏观与持续账户目标。\n',encoding='utf-8')
    (OUT/'审阅提示词.md').write_text('请先读研究结论和protocol，再看逐事件数据、模型记录、同样本解释及三张图。审查是否把管理人调查意向误写为真实买入，是否把全A股意向当510300专属信号，是否误用来源时点、除息和已成熟训练标签。尤其检查33与27样本比较、仅8/3次预测的证据边界，以及C仍输给恒零预测、预测方向未改变。\n\n请给出明确批评，并提出下一项不同机制的优先级、具体预测对象、对照、验证及停止条件。不要建议围绕现有正相关换窗口、方向或权重。怎样利用政策和资金约束提前判断供给持续性及承接，而不是等待持有人表；怎样区分愿意买、能够买、真的买及对价格的影响。\n\n本包含冻结历史实验，不是未见样本，也未通过外部审阅。前轮一月案例只作解释背景，其完整原始材料在前轮包。保存数值验证不认证商业来源历史首版。没有账户结果、没有当前交易信号。不要宣称达成夏普目标或授权实盘。\n',encoding='utf-8')
    (OUT/'00_README_FIRST.md').write_text('阅读顺序：研究结论.md → 同样本散点图及走势对照 → protocol.json → 逐事件特征与随后收益.csv → 固定逐时点预测.csv及固定模型记录.json → 月度意向人工准入.csv、全部月份覆盖与缺口.csv和来源链接与下载记录.json → 审阅提示词.md。\n\n本包数值验证命令：python research/private_manager_intent_delivery_v1.py verify --root .\n绘图：python research/private_manager_intent_diagnostic_v1.py plots\n固定历史重算：python research/private_manager_intent_diagnostic_v1.py analyze（会重新计算模型；不是只读验证）。\n\n原始抓取全文和PDF仅本地保存，商业全文及其长摘录不进入审阅包。提供原链接、哈希、日期和抽取事实；离线可复核从已保存字段到标签、预测和误差的计算，不能离线逐份认证原网页历史版本。代码的fetch/extract/curate属于源资料准备阶段，需要包外原文，不是完整离线步骤。\n\n早期案例仅附摘录，包内未重新重建持有人研究。源资料缺口和当前研究未完成均明确保留。\n',encoding='utf-8')
    exclusions={'excluded':['raw_local_only全部商业正文、PDF、网页脚本、搜索返回及来源页截图','receipts含网页长段落的内部抓取记录','field_candidates.json及字段候选.csv长原文片段','付费或需登录的全文，未尝试访问','其他研究的完整模型与持有人原始PDF，前轮只摘取背景','任何凭证、订单和实盘服务'],
                'included':'独立事实、完整来源链接与字节哈希、人工准入表、冻结协议、全部本轮标签和模型、图和完整价格输入',
                'scope':'尚无历史首版认证；结构通过不等于信息有效或外部审阅'}
    dump(OUT/'排除项与复核边界.json',exclusions)
    packages=['numpy','pandas','scipy','scikit-learn','pyarrow','matplotlib','requests','beautifulsoup4','pypdfium2']
    from importlib.metadata import version
    (OUT/'requirements.txt').write_text('\n'.join(f'{name}=={version(name)}' for name in packages)+'\n',encoding='utf-8')
    print('本轮结论、同样本解释和交付导航已保存',flush=True)


def fair_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,
                         'figure.facecolor':'#f7f9fc','axes.spines.top':False,'axes.spines.right':False})
    d=pd.read_parquet(OUT/'逐事件特征与随后收益.parquet').dropna(subset=['intent_headroom'])
    fig,axs=plt.subplots(1,2,figsize=(14,6.3),sharey=True)
    fig.subplots_adjust(top=.79,bottom=.22,left=.08,right=.97,wspace=.17)
    fig.suptitle('相同27个月：私募意向与随后20日收益',x=.08,ha='left',y=.97,fontsize=20,fontweight='bold')
    fig.text(.08,.9,'公开之后的下一开盘开始计算｜每个点为一个月｜历史描述性线索，尚未建立预测增量',color='#536176')
    for ax,col,title,color in [(axs[0],'intent','单独增减仓意向','#2563b8'),(axs[1],'intent_headroom','意向 × 可加仓空间','#0e897b')]:
        corr=spearmanr(d[col],d.forward_return20).statistic
        ax.scatter(d[col]*100,d.forward_return20*100,s=49,color=color,alpha=.82)
        ax.axhline(0,color='#929ba9',ls='--',lw=1);ax.axvline(0,color='#929ba9',ls='--',lw=1)
        ax.set_title(f'{title}，秩相关 {corr:+.3f}',loc='left',fontsize=12)
        ax.set_xlabel('计划指数 − 100' if col=='intent' else '(计划指数 − 100) × (1 − 调查仓位)')
        ax.grid(alpha=.16);ax.set_axisbelow(True)
        for idx in [d.forward_return20.idxmax(),d.forward_return20.idxmin()]:
            r=d.loc[idx];ax.annotate(r.month,xy=(r[col]*100,r.forward_return20*100),xytext=(6,6),textcoords='offset points',fontsize=9,color='#536176')
    axs[0].set_ylabel('随后20日毛收益（%）')
    fig.text(.08,.07,'相同月份比较，避免把样本变化当信息增量。标注为全样本收益最高与最低月份，未据此剔除或选策略。\n15组相邻调查仓位变化没有验证持续买入；固定模型评价仅8次/3次，新增信息未改变预测涨跌方向。',fontsize=10,color='#536176')
    path=OUT/'figures/同27个月_意向与可加仓空间.png';fig.savefig(path,dpi=170);plt.close(fig)
    print('同样本对比图已保存',flush=True)


def verify(root):
    out=Path(root).resolve()/REL;checks=0
    def check(x,msg):
        nonlocal checks
        checks+=1
        if not bool(x):raise AssertionError(msg)
    def near(a,b,msg,tol=1e-9):check(np.allclose(a,b,rtol=0,atol=tol,equal_nan=True),msg)
    spec=read(out/'protocol.json');freeze=read(out/'freeze_receipt.json')
    check(digest(out/'protocol.json')==freeze['protocol_sha256'],'协议字节保持冻结')
    check(digest(Path(root)/'research/private_manager_intent_diagnostic_v1.py')==freeze['analysis_code_sha256'],'分析代码保持冻结')
    for name,h in spec['frozen_source_files'].items():check(digest(out/name)==h,'冻结输入 '+name)
    d=pd.read_parquet(out/'逐事件特征与随后收益.parquet');m=pd.read_parquet(out/'inputs/market.parquet').sort_values('date').reset_index(drop=True)
    m.date=pd.to_datetime(m.date);check(m.date.is_unique,'市场日期唯一');check(d.month.is_unique,'月度信息不重复')
    e=pd.read_csv(out/'月度调查仓位.csv').set_index('month')
    for r in d.itertuples():
        pub=pd.Timestamp(r.available_date);entry=int(np.flatnonzero(m.date.gt(pub))[0]);end=entry+19
        check(str(m.iloc[entry].date.date())==r.entry_date,'下一开盘时钟')
        check(str(m.iloc[end].date.date())==r.exit_date,'固定20日')
        div=float(m.iloc[entry+1:end+1].dividend.sum())
        near(div,r.holding_dividend,'持有分红不含入场日')
        near((m.iloc[end].close+div)/m.iloc[entry].open-1,r.forward_return20,'独立收益复算')
        near(m.iloc[entry-1].wealth/m.iloc[entry-21].wealth-1,r.prior_return20,'过去收益时钟')
        vol=np.std(m.iloc[entry-20:entry].total_simple,ddof=1)*np.sqrt(242)
        near(np.log(vol),r.logvol20,'风险输入时钟')
        near((r.plan_index-100)/100,r.intent,'意向数值')
        if pd.notna(r.headroom):near((1-r.survey_exposure_pct/100)*r.intent,r.intent_headroom,'意向空间乘积')
        if pd.notna(r.next_survey_exposure_change_pp):
            nr=e.loc[str(pd.Period(r.month,freq='M')+1)]
            near(nr.survey_exposure_pct-r.survey_exposure_pct,r.next_survey_exposure_change_pp,'相邻仓位验证')
    forecasts=pd.read_csv(out/'固定逐时点预测.csv');records=read(out/'固定模型记录.json')
    check(len(records)==len(forecasts)==22,'模型保存数')
    for r in records:
        target=d[d.month.eq(r['month'])].iloc[0]; required=spec['fixed_models']['C'] if r['model'] in ['AC','C'] else spec['fixed_models']['B']
        hist=d[(d.exit_date<target.decision_date)&(d.available_date<target.available_date)].dropna(subset=required+['forward_return20'])
        check(hist.month.tolist()==r['train_months'],'成熟训练样本完整匹配')
        check(len(hist)>=24,'训练门槛')
        v=hist[r['features']].to_numpy();mu=np.array(r['means']);sd=np.array(r['scales']);beta=np.array(r['coefficients'])
        near(v.mean(axis=0),mu,'训练均值');near(np.where(v.std(axis=0)==0,1,v.std(axis=0)),sd,'训练标准差')
        x=(v-mu)/sd;y=hist.forward_return20.to_numpy()
        near(x.T@(y-r['intercept']-x@beta),10*beta,'保存系数满足岭回归方程',1e-8)
        near(r['intercept'],y.mean(),'截距')
        predicted=r['intercept']+((target[r['features']].to_numpy(dtype=float)-mu)/sd)@beta
        f=forecasts[(forecasts.month==r['month'])&(forecasts.model==r['model'])].iloc[0]
        near(predicted,f.prediction,'保存系数复算预测');near(f.training_mean,y.mean(),'成熟均值基准');near(f.actual,target.forward_return20,'预测目标')
    s=read(out/'summary.json')
    for ev in s['predictive_comparisons']:
        name=ev['model'];base='A' if name=='B' else 'AC'
        j=forecasts[forecasts.model.eq(name)].merge(forecasts[forecasts.model.eq(base)],on='month',suffixes=('_added','_base'))
        check(len(j)==ev['evaluation_events']<24,'评价样本门槛不足')
        near(j.actual_added,j.actual_base,'两组目标相同');near(j.n_train_added,j.n_train_base,'两组训练数量相同')
        y=j.actual_added.to_numpy();ea=(y-j.prediction_added.to_numpy())**2;eb=(y-j.prediction_base.to_numpy())**2;em=(y-j.training_mean_added.to_numpy())**2
        near(np.sqrt(ea.mean())*100,ev['rmse_added_pp'],'预测误差');near(1-ea.mean()/eb.mean(),ev['mse_skill_vs_base'],'基准改善');near(1-ea.mean()/em.mean(),ev['mse_skill_vs_mean'],'均值改善')
    fair=read(out/'同样本与预测符号解释.json');z=d.dropna(subset=['intent_headroom'])
    check(z.month.tolist()==fair['matched_months'],'同样本月份')
    near(spearmanr(z.intent,z.forward_return20).statistic,fair['intent_spearman_same_sample'],'同样本意向相关')
    near(spearmanr(z.intent_headroom,z.forward_return20).statistic,fair['intent_headroom_spearman_same_sample'],'同样本交互相关')
    for r in s['descriptive_results']:
        cols={'意向与此前收益':('intent','prior_return20'),'意向与随后收益':('intent','forward_return20'),
              '意向乘空间与随后收益':('intent_headroom','forward_return20'),'意向与下期调查仓位变化':('intent','next_survey_exposure_change_pp'),
              '意向乘空间与下期调查仓位变化':('intent_headroom','next_survey_exposure_change_pp')}
        x,y=cols[r['comparison']];a=d if r['subset']=='全部历史重建' else d[d.clock.eq('DATED_ARTICLE')];a=a.dropna(subset=[x,y])
        check(len(a)==r['n'],'描述性样本数');near(spearmanr(a[x],a[y]).statistic,r['spearman'],'描述性相关')
    check(s['accounts_run']==0 and s['strict_forward_days']==0,'保持未运行账户与前向')
    result={'status':'PASS_SAVED_VALUES_CLOCKS_AND_MODEL_EQUATIONS','checks':checks,'new_fits':0,'new_accounts':0,'new_downloads':0,'new_resamples':0,'verified_at':now()}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def build_zip():
    check=verify(ROOT);dump(OUT/'保存数值复核.json',check)
    names=[
        '00_README_FIRST.md','研究结论.md','用户要求.md','审阅提示词.md','requirements.txt','排除项与复核边界.json',
        'source_plan.json','protocol.json','freeze_receipt.json','summary.json','admitted_sources.json','月度意向人工准入.csv','月度调查仓位.csv',
        '全部月份覆盖与缺口.csv','来源链接与下载记录.json','原发布方公开目录.csv','逐事件特征与随后收益.csv','逐事件特征与随后收益.parquet',
        '固定相关诊断.csv','固定逐时点预测.csv','固定模型记录.json','同样本与预测符号解释.json','保存数值复核.json','图形核对.json',
    ]
    files={}
    for n in names:files[(REL/n).as_posix()]=OUT/n
    for folder in ['inputs','figures','evidence']:
        for p in (OUT/folder).rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=p
    for p in OUT.glob('*candidate_urls.json'):files[p.relative_to(ROOT).as_posix()]=p
    for n in ['private_manager_intent_v1.py','private_manager_intent_diagnostic_v1.py','private_manager_intent_delivery_v1.py']:
        files['research/'+n]=ROOT/'research'/n
    extras={'00_README_FIRST.md':(OUT/'00_README_FIRST.md').read_bytes(),'审阅提示词.md':(OUT/'审阅提示词.md').read_bytes(),'requirements.txt':(OUT/'requirements.txt').read_bytes()}
    members={n:p.read_bytes() for n,p in files.items()};members.update(extras)
    buf=io.StringIO(newline='');w=csv.writer(buf);w.writerow(['path','bytes','sha256'])
    for n,b in sorted(members.items()):w.writerow([n,len(b),hashlib.sha256(b).hexdigest()])
    members['FILE_INDEX.csv']=buf.getvalue().encode('utf-8-sig')
    destination=ROOT/'deliverables'/ZIP_NAME;temp=destination.with_suffix('.building.zip')
    if destination.exists():raise ValueError('交付包已经存在，禁止覆盖')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,b in members.items():z.writestr(n,b)
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        index=list(csv.DictReader(io.StringIO(z.read('FILE_INDEX.csv').decode('utf-8-sig'))))
        assert set(r['path'] for r in index)==set(z.namelist())-{'FILE_INDEX.csv'}
        for r in index:
            b=z.read(r['path']);assert len(b)==int(r['bytes']) and hashlib.sha256(b).hexdigest()==r['sha256']
        extracted=ROOT/'artifacts'/'private_intent_review_verify_20260922'
        if extracted.exists():raise ValueError('复核解压路径已存在，请使用新目录')
        extracted.mkdir(parents=True)
        for n in z.namelist():
            target=(extracted/n).resolve();assert target.is_relative_to(extracted.resolve())
        z.extractall(extracted)
    proc=subprocess.run([sys.executable,str(extracted/'research/private_manager_intent_delivery_v1.py'),'verify','--root',str(extracted)],capture_output=True,text=True,encoding='utf-8')
    if proc.returncode:raise RuntimeError(proc.stdout+'\n'+proc.stderr)
    temp.replace(destination)
    receipt={'path':str(destination),'bytes':destination.stat().st_size,'sha256':digest(destination),'members':len(members),'indexed_members':len(index),
             'crc_duplicates_hashes':'PASS','fresh_extraction_saved_verification':json.loads(proc.stdout.strip()),'external_review':'NOT_PERFORMED','created_at':now()}
    dump(OUT/'delivery_receipt.json',receipt)
    status_path=ROOT/'reports/research/510300_macro_research_program_status_v1.json';status=read(status_path)
    study='510300_PRIVATE_MANAGER_INTENT_V1';status['completed_substudies']=[x for x in status['completed_substudies'] if x['id']!=study]
    status['completed_substudies'].append({'id':study,'status':read(OUT/'summary.json')['status'],'scope':'36意向月份、33历史事件、27仓位完整月份、15相邻仓位验证、8/3预测评价，22保存模型；不足预测门槛，不反号或改窗口。',
                                         'accounts':'NOT_RUN_PREDICTIVE_GATE','whole_objective_complete':False})
    status.update(latest_review_zip='deliverables/'+ZIP_NAME,latest_review_zip_sha256=receipt['sha256'],whole_objective_complete=False,
                  current_view='NO_VIEW_NO_VALIDATED_CAPITAL_BEHAVIOR_RULE',
                  next_priority='保留私募意向与空间线索，固定当前不足评价结论；研究独立政策或资金约束对供给持续性和承接的条件预测，不将延迟身份确认设为前提，不把旧份额/融资信号改名或围绕本轮正相关调参。')
    if status.get('remaining_requirements'):
        status['remaining_requirements'][0]='私募意向与可加仓空间已完成固定历史诊断，评价不足且未建立真实行为链条；继续检验独立政策或资金约束与供给承接，不以新的参数搜索救援本轮结果。'
    dump(status_path,status)
    log=ROOT/'RESEARCH_STATUS.md';old=log.read_text(encoding='utf-8')
    text='> 2026-09-22 私募操作意向固定历史诊断完成：`510300_PRIVATE_MANAGER_INTENT_V1`。113来源线索，36个独立意向月份，33个可用事件；同27月意向/意向乘空间与随后20日收益秩相关0.298/0.353。15组相邻调查仓位未建立持续买入证据。新增意向/加入空间模型仅8/3次评价，22条保存模型，预测方向改变0次；样本门槛不足，账户NOT_RUN，前向0。不得把相关性或点值MSE下降当可交易增量。旧失败和85/15终止保留。见[研究结论](reports/research/510300_private_manager_intent_v1/研究结论.md)及[审阅包](deliverables/'+ZIP_NAME+')。\n\n'
    if not old.startswith(text):log.write_text(text+old,encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description='私募意向研究交付与保存数值复核');p.add_argument('action',choices=['prepare','fair_plot','verify','build_zip']);p.add_argument('--root',default=str(ROOT));a=p.parse_args()
    if a.action=='verify':verify(a.root)
    else:globals()[a.action]()
