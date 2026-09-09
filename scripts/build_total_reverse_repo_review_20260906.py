"""打包本轮全部结果与因子修正证据，只做必要的结构和数值检查。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/research/510300_total_reverse_repo_v2'
FACTOR=ROOT/'reports/research/510300_factor_definition_review_20260906'
FUND=ROOT/'reports/research/510300_fundamental_and_fund_flow_rebuild_v1'
DELIVERY=ROOT/'deliverables/510300第七轮_因子修正_20260906'
ZIP=ROOT/'deliverables/510300夏普1.2持续研究_第七轮因子修正_GPT审阅_20260906.zip'

def physical(path:Path,root:Path=ROOT)->Path:
    rel=path.relative_to(root).as_posix()
    return Path(r'E:\ResearchData\New project 8')/rel if root==ROOT and rel.startswith('data/') else path

def label(key):
    from research.total_reverse_repo_v2 import NAMES
    if key in NAMES:return NAMES[key]
    group,kind,h=key.split('_')
    return {'PRICE':'价格','FUNDING':'价格与资金利率','SEVEN':'七天公告流量','REGULAR':'常规全期限公告量','ALL':'全期限与买断式公开量'}[group]+'／'+{'RIDGE':'岭回归','ET':'极随机树'}[kind]+'／二十日'

def verify_saved(root):
    from research.adaptive_allocation_v1 import summarize
    report=root/'reports/research/510300_total_reverse_repo_v2'
    config=json.loads((root/'config/510300_total_reverse_repo_v2.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(report/'metrics.csv')
    maximum=0.
    for row in metrics.to_dict('records'):
        ledger=pd.read_parquet(report/'evaluation'/row['cost']/(row['model']+'_ledger.parquet'))
        actual=summarize(ledger,config)
        for key,value in actual.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                if value is None or not np.isfinite(value):continue
                error=abs(float(row[key])-float(value))
                maximum=max(maximum,error)
                assert error<1e-7, (row['model'],key,error)
        assert len(ledger)==1604
        assert ledger.date.min()==pd.Timestamp('2020-01-02') and ledger.date.max()==pd.Timestamp('2026-08-14')
        assert ledger.accounting_error.abs().max()<1e-6
        if row['model']=='BUY_HOLD':
            old=pd.read_parquet(root/'reports/research/510300_adaptive_allocation_v1/evaluation'/row['cost']/'BUY_HOLD_ledger.parquet')
            for c in ['equity','shares','cash','net_return','commission','slippage_cost']:
                np.testing.assert_array_equal(ledger[c].to_numpy(),old[c].to_numpy())
    train=json.loads((report/'training_receipts.json').read_text(encoding='utf-8'))
    assert len(train)==280
    assert all(pd.Timestamp(t['last_label_exit'])<=pd.Timestamp(t['fit_origin']) for t in train)
    assert all(t['train_rows']>=300 for t in train)
    for t in set(z['fit_origin'] for z in train):
        rows=[z for z in train if z['fit_origin']==t]
        for col in ['train_rows','first_train_origin','last_train_origin','last_label_exit']:
            assert len(set(z[col] for z in rows))==1
    return {'status':'PASS_SAVED_ACCOUNT_METRICS_AND_TRAINING_CLOCKS','evaluation_accounts':len(metrics),
            'quarterly_fits':len(train),'maximum_metric_error':maximum,'benchmark_daily_parity':'EXACT',
            'scope':'保存账户指标、训练成熟时刻和交付结构核对；不重训，不进行安全审计'}

def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(2,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for key,name,color in [('T1_PRIMARY_ALL','全期限与买断式主方案','#187d91'),('T2_REGULAR_ALL_TENOR','常规全期限对照','#a57536'),('BUY_HOLD','买入持有','#62666d')]:
        d=pd.read_parquet(REPORT/'evaluation/BASE'/(key+'_ledger.parquet'))
        wealth=d.equity.to_numpy()/200000
        peaks=np.maximum.accumulate(np.r_[1.,wealth])[1:]
        ax[0].plot(d.date,wealth,label=name,color=color,linewidth=1.5)
        ax[1].plot(d.date,100*(wealth/peaks-1),color=color,linewidth=1.2)
    ax[0].set_title('第七轮：补全逆回购口径后的完整账户',loc='left',pad=44,fontsize=15)
    ax[0].legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False)
    ax[0].set_ylabel('账户净值（初始为1）')
    ax[1].set_ylabel('回撤（%）')
    for a in ax:a.grid(alpha=.15)
    fig.text(.08,.02,'2020-01-02 至 2026-08-14 开盘；基础费用、真实分红、现金等待与期末退出均计入。\n金额按公告首次公开时点使用；买断式预告与月报分别记录，不能称为每日实际净投放。',fontsize=9,color='#555555')
    fig.tight_layout(rect=(0,.065,1,1))
    fig.savefig(DELIVERY/'第七轮_完整账户净值与回撤.png',dpi=150)
    plt.close(fig)

def build():
    assert not DELIVERY.exists() and not ZIP.exists(),'本轮交付已存在，禁止覆盖'
    checked=verify_saved(ROOT)
    result=json.loads((REPORT/'result.json').read_text(encoding='utf-8'))
    source=json.loads((REPORT/'source_reconstruction_receipt.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(REPORT/'metrics.csv')
    eras=pd.read_csv(REPORT/'era_metrics.csv')
    base=next(r for r in result['primary'] if r['cost']=='BASE')
    stress=next(r for r in result['primary'] if r['cost']=='STRESS')
    DELIVERY.mkdir(parents=True)
    lines=['# 第七轮结果：逆回购口径修正与因子重建','',
           f"主方案基础夏普为 **{base['net_sharpe']:.4f}**，压力夏普为 **{stress['net_sharpe']:.4f}**；基础年化收益 **{base['annualized_return']:.2%}**，最大回撤 **{-base['max_drawdown']:.2%}**。尚未达到夏普1.2，持续研究继续。",'',
           '本轮完成十五个候选、三十二个完整评价账户、二百八十次季度训练。所有候选和失败结果保留，没有挑选最好年份、改费用或倒转方向补救。','',
           f"本轮事后最好为“{label(result['post_selected_best_base']['model'])}”，基础夏普 **{result['post_selected_best_base']['net_sharpe']:.4f}**，属于价格对照，并非补全逆回购带来的成功。",'',
           '## 具体修正了什么','',
           f"重新解析{source['ordinary_notice_count']}篇普通公告，拆出七天、十四天、二十一天、二十八天和六十三天逆回购。二〇一五年至截止日，普通公告内非七天期限累计金额为{source['other_tenor_total_100m']:,.0f}亿元；这是多年累计投放流量，不能当作未到期余额。",'',
           '仅二〇一六年，非七天期限就有六点九万亿元，占当年常规逆回购累计投放约百分之二十七点八。重新汇总与央行年报在其四舍五入精度内一致。',
           '同一公告中还可能同时列出中期借贷便利。新解析按表标题区分，央票、特别国债和中期借贷便利不加入逆回购总量；同时存在投标量和中标量时只取中标量。',
           '买断式单独栏目新增三十七篇截止日内公告，拆成四十四条分期限记录，其中十四条来自八篇已执行月报，三十条来自二十九篇操作预告。分别保留公开日、操作月份、已知实际操作日与明确到期日。',
           '本轮研究的是按公开时点汇总的新披露金额，包含单列的月报及预告信息。它不是每日实际净投放。月报没有逐笔实际日期、部分到期证据也不全，因此没有虚构每日净投放。',
           '旧第四轮的结论追加适用范围：它只检验七天最新公告规模，不能否证总逆回购。原协议、代码和结果未覆盖。','',
           '## 全部候选结果','',
           '| 策略 | 基础夏普 | 压力夏普 | 基础年化收益 | 相对买入持有年化差 | 最大回撤 | 成交笔数 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    order=[*__import__('research.total_reverse_repo_v2',fromlist=['ENSEMBLES']).ENSEMBLES,'BUY_HOLD']+[x for x in metrics.model.unique() if not x.startswith('T') and x!='BUY_HOLD']
    for key in order:
        a=metrics.loc[(metrics.model==key)&(metrics.cost=='BASE')].iloc[0]
        b=metrics.loc[(metrics.model==key)&(metrics.cost=='STRESS')].iloc[0]
        lines.append(f'| {label(key)} | {a.net_sharpe:.4f} | {b.net_sharpe:.4f} | {a.annualized_return:.2%} | {a.annualized_return_excess_vs_buy_hold:.2%} | {-a.max_drawdown:.2%} | {a.trade_count} |')
    lines += ['', '## 主方案分期与不确定性','', '| 阶段 | 基础夏普 | 年化收益 | 最大回撤 |', '| --- | ---: | ---: | ---: |']
    for r in eras.loc[(eras.model=='T1_PRIMARY_ALL')&(eras.cost=='BASE')].itertuples():lines.append(f'| {r.era} | {r.net_sharpe:.4f} | {r.annualized_return:.2%} | {-r.max_drawdown:.2%} |')
    u=result['uncertainty']['BASE']
    lines += ['',f"主方案夏普百分之九十五区间为{u['primary_sharpe_95_interval']}。相对常规全期限组的年化日均增量区间为{u['increment_vs_regular_95_interval']}；相对七天流量组为{u['increment_vs_seven_95_interval']}，均跨过零。区间未校正历次重复研究，不构成稳定超额证据。",'',
              '费用升高还会改变考虑成本后的仓位选择，因此压力账户未必简单等于基础账户减去额外费用。本轮压力主方案亏损略少，不表示成本增加能提升策略。', '',
              '## 其他因子与用户新增方向','',
              '附件《六轮95项因子_完整中文说明与修正》逐项列出定义、单位、可用时点和检查深度。三十四项价格因子独立复算通过；六个旧操作量因子确认存在总量范围不足。日内财富分解、简单两日涨跌比例、成交量与申赎的语义也已经澄清。',
              '盈利、估值、股东回报和公募申赎已经加入持续研究范围。本次归档基金业协会七十七份月报，形成股票基金与混合基金一百五十四条结构化记录；二十五份旧报告存在疑似迁移或较晚重发日期，分类范围在二〇二五年十一月变化，另有五条同口径相邻月份份额修订差异。它们均作为实际来源问题记录，不能回填或忽略。',
              '公募规模不是净申购，基金份额变化也不能精确等同于沪深300现金流入。盈利乘市盈率是同口径价格关系，不能把恒等变换当成预测；分红与回购对每股盈利的影响还要避免重复计收益。', '',
              '## 全部中文规则','', (ROOT/'docs/510300_TOTAL_REVERSE_REPO_V2_PROTOCOL.md').read_text(encoding='utf-8'),'',
              '## 验证与范围','',
              '五项针对性测试覆盖多期限、金额单位、投标与中标去重、中期借贷便利排除、盘后月报与预告不可提前使用、未来新增公告不改变过去特征。保存的三十二个账户指标复算通过，买入持有逐日与原账户一致，季度训练标签均已到期。',
              '只做必要结构和数值核对，未做安全审计、未上传、未声称外部审阅。研究不触发真实订单。']
    (DELIVERY/'第七轮结果与全部中文规则.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (DELIVERY/'00_先读说明.md').write_text('# 从这里开始\n\n先读《第七轮结果与全部中文规则》，再读附件内的95项因子说明和盈利、公募资金来源说明。\n\n本包含第七轮完整冻结输入、原始公告、源代码、全部账户、训练记录，和本次因子检查及公募官方月报。原相对路径保留。FILE_INDEX.csv覆盖本包文件（索引自身除外）。\n\n运行打包脚本的--verify-only可重新计算保存账户指标，不重训、不下载。配置保留历史时间和全部费用。历史已被多次观察；目标未达到，持续研究保持运行。\n',encoding='utf-8')
    (DELIVERY/'01_GPT审阅提示词.md').write_text('# 请审阅第七轮因子口径修正\n\n目标是510300与现金的完整账户成本后夏普1.2，使用免费来源。请优先检查逆回购期限是否完整、同公告的MLF是否排除、投标量与中标量是否重复、月报与预告是否错时或冒充每日实际净投放，及常规操作量与资金利率是否混淆。对来源覆盖与净投放缺口给出实质修正方法。\n\n核对全部15个候选与32个完整账户、训练标签到期时刻、无判断日期、分红、整手、费用与期末退出。不要把点估计、最好分期或假设实时接口可用当成已验证结果。\n\n继续检查附件95项因子的经济含义。围绕盈利增长、估值变化、股东回报及公募申赎，指出独立信息与恒等变换、实际流量与价格影响、基金份额变化与净现金申购的差别。审查77份公募月报的发布时间迁移、分类改变和历史版本差异。\n\n提出下一轮一个能实际执行的研究方案，写完整中文因子、免费来源、可用时钟、有限候选、账户和验收标准，以及无效时的下一方向。不得以挑选年份、无限扫描参数、删费用或未来财报回填来实现1.2。任何建议只是待研究，不授权真实交易。\n',encoding='utf-8')
    (DELIVERY/'保存结果核对.json').write_text(json.dumps(checked,ensure_ascii=False,indent=2),encoding='utf-8')
    manifestpath=ROOT/'config/510300_total_reverse_repo_v2_manifest.json'
    expected={r['path']:r for r in json.loads(manifestpath.read_text(encoding='utf-8'))['files']}
    paths={ROOT/x for x in expected}
    paths.update({manifestpath,Path(__file__)})
    for folder in [REPORT,FACTOR,FUND]:paths.update(p for p in folder.rglob('*') if p.is_file() and not p.name.endswith('_preview.png') and 'probe' not in p.name)
    parentmf=ROOT/'config/510300_adaptive_allocation_v1_manifest.json'
    paths.add(parentmf)
    paths.update(ROOT/r['path'] for r in json.loads(parentmf.read_text(encoding='utf-8'))['files'])
    from research.total_reverse_repo_v2 import NOTICE,DR007
    paths.update(ROOT/x for x in pd.read_parquet(physical(NOTICE)).raw_path)
    paths.add(NOTICE.parent/'pboc_source_acquisition_manifest.json')
    drreceipt=DR007.parent/'dr007_tushare_source_acquisition_manifest.json'
    paths.add(drreceipt)
    paths.update(ROOT/x['response_artifact']['path'] for x in json.loads(physical(drreceipt).read_text(encoding='utf-8'))['chunk_receipts'])
    rawroot=Path(r'E:\ResearchData\New project 8\data\raw')
    for child in ['510300_total_reverse_repo_v2','510300_fundamental_and_fund_flow_rebuild_v1/amac']:
        paths.update(ROOT/p.relative_to(Path(r'E:\ResearchData\New project 8')) for p in (rawroot/child).rglob('*') if p.is_file())
    source3=json.loads((ROOT/'config/510300_overnight_global_information_v1_manifest.json').read_text(encoding='utf-8'))
    paths.update(ROOT/x['path'] for x in source3['files'] if 'data/raw/' in x['path'])
    for stage,files in [('510300_overnight_global_information_v1',['features.parquet','source_receipt.json','source_clock_alignment.parquet']),('510300_policy_liquidity_quantity_v1',['features.parquet','feature_receipt.json']),('510300_calendar_liquidity_timing_v1',['features.parquet','feature_receipt.json'])]:
        paths.update(ROOT/'reports/research'/stage/x for x in files)
    extra=['research/asymmetric_stress_hazard_source_remediation_v1_0_1.py','research/calendar_liquidity_timing_v1.py','research/overnight_global_information_v1.py',
           'scripts/inspect_repo_sources_20260906.py','scripts/acquire_buyout_repo_20260906.py','scripts/acquire_amac_monthly_funds_20260906.py',
           'scripts/structure_amac_monthly_funds_20260906.py','scripts/review_factor_definitions_20260906.py',
           'data/raw/flow/510300_etf_share_premium_level_full_v1.parquet','data/raw/valuation/000300_valuation_daily_raw.parquet',
           'docs/000300_POINT_IN_TIME_VALUATION_V2_RECONSTRUCTIBILITY_SPEC.md','docs/510300_CREATION_DATA_READINESS_LEDGER_V1_SPEC.md']
    paths.update(ROOT/x for x in extra)
    for cost in ['BASE','STRESS']:paths.add(ROOT/'reports/research/510300_adaptive_allocation_v1/evaluation'/cost/'BUY_HOLD_ledger.parquet')
    for x in ['research/__init__.py','scripts/__init__.py','tests/__init__.py']:
        if (ROOT/x).is_file():paths.add(ROOT/x)
    draw()
    rows=[]
    with zipfile.ZipFile(ZIP,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        # 单次读源文件，同时写交付目录和压缩包，减少大量小文件的重复读取。
        for n,p in enumerate(sorted(paths),1):
            relative=p.relative_to(ROOT).as_posix()
            payload=physical(p).read_bytes()
            digest=hashlib.sha256(payload).hexdigest()
            if relative in expected:assert digest==expected[relative]['sha256'] and len(payload)==expected[relative]['bytes'],relative
            target=DELIVERY/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(payload)
            archive.writestr(relative,payload)
            rows.append({'path':relative,'bytes':len(payload),'sha256':digest})
            if n%400==0:print(f'交付文件已写入 {n}/{len(paths)}',flush=True)
        assert verify_saved(DELIVERY)==checked
        for p in sorted(DELIVERY.iterdir()):
            if not p.is_file():continue
            payload=p.read_bytes()
            archive.writestr(p.name,payload)
            rows.append({'path':p.name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=['path','bytes','sha256'])
        writer.writeheader()
        writer.writerows(sorted(rows,key=lambda x:x['path']))
        contents=stream.getvalue().encode('utf-8-sig')
        (DELIVERY/'FILE_INDEX.csv').write_bytes(contents)
        archive.writestr('FILE_INDEX.csv',contents)
    with zipfile.ZipFile(ZIP) as archive:
        names=archive.namelist()
        assert archive.testzip() is None
        assert len(names)==len(set(names)) and set(names)=={r['path'] for r in rows}|{'FILE_INDEX.csv'}
        for row in rows:
            b=archive.read(row['path'])
            assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
    receipt={'zip':str(ZIP),'bytes':ZIP.stat().st_size,'sha256':hashlib.sha256(ZIP.read_bytes()).hexdigest(),'members':len(names),
             'crc':'PASS','index':'PASS','hashes':'PASS','recomputation':checked,'security_audit_performed':False,'external_review_received':False}
    ZIP.with_suffix('.delivery.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='第七轮结果复算与中文交付')
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:print(json.dumps(verify_saved(ROOT),ensure_ascii=False),flush=True)
    else:build()
