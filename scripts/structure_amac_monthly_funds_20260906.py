"""按官方报告内的同口径两个月份数据整理公募资金线索。"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_fundamental_and_fund_flow_rebuild_v1'

def main():
    records=json.loads((OUT/'amac_source_records.json').read_text(encoding='utf-8'))
    rows=[]
    for r in records:
        body=r['extracted_text']
        for cat in ['股票基金','混合基金']:
            matching=[line.strip() for line in body.splitlines() if re.match(r'^(?:其中[：:]\s*)?'+cat+r'\s',line.strip())]
            if len(matching)!=1:raise ValueError(f"{r['report_month']} {cat} 行不能唯一定位")
            values=re.findall(r'-?\d[\d,]*(?:\.\d+)?',matching[0])
            if len(values)!=6:raise ValueError(f"{r['report_month']} {cat} 数字列数不是六：{matching[0]}")
            n,shares,nav,pn,ps,pnav=map(lambda x:float(x.replace(',','')),values)
            month=pd.Period(r['report_month'],freq='M')
            date=pd.Timestamp(r['publication_date'])
            lag=(date-month.end_time.normalize()).days
            state='LIST_DATE_PLAUSIBLE_ORIGINAL_TIME_NOT_INDEPENDENTLY_PROVEN'
            if lag<=0:state='INVALID_LIST_DATE_NOT_AFTER_OBSERVATION_MONTH'
            elif lag>62:state='SUSPECT_MIGRATION_OR_LATE_REPUBLICATION_DATE'
            scope='OPEN_ENDED_SUBCATEGORY' if '开放式基金' in body else 'ALL_FUNDS_BY_ASSET_CATEGORY'
            rows.append({'report_month':r['report_month'],'category':cat,'listed_publication_date':r['publication_date'],
                         'publication_date_status':state,'days_after_report_month_end':lag,'category_scope':scope,
                         'current_fund_count':int(n),'previous_fund_count_same_report':int(pn),
                         'current_shares_100m':shares,'previous_shares_100m_same_report':ps,'current_nav_100m':nav,'previous_nav_100m_same_report':pnav,
                         'share_change_100m':shares-ps,'share_log_growth_same_report':float(np.log(shares/ps)),
                         'nav_change_100m':nav-pnav,'nav_log_growth_same_report':float(np.log(nav/pnav)),
                         'count_change':int(n-pn),'source_url':r['source_url'],'raw_path':r['raw_path'],'source_sha256':r['sha256'],
                         'cash_net_subscription_amount':'NOT_IDENTIFIED_BY_AGGREGATE_SHARES',
                         'original_first_publication_proven':False})
    frame=pd.DataFrame(rows)
    revisions=[]
    for cat,part in frame.groupby('category'):
        lookup={r.report_month:r for r in part.itertuples()}
        for r in part.itertuples():
            previous=str(pd.Period(r.report_month,freq='M')-1)
            if previous not in lookup:continue
            old=lookup[previous]
            revisions.append({'report_month':r.report_month,'category':cat,'same_scope':r.category_scope==old.category_scope,
                              'previous_month_shares_difference_100m':r.previous_shares_100m_same_report-old.current_shares_100m,
                              'previous_month_nav_difference_100m':r.previous_nav_100m_same_report-old.current_nav_100m,
                              'source_url':r.source_url,'previous_source_url':old.source_url})
    comp=pd.DataFrame(revisions)
    frame.to_parquet(OUT/'amac_equity_mixed_monthly_as_reported.parquet',index=False)
    frame.to_csv(OUT/'公募股票与混合基金_官方月报结构化.csv',index=False,encoding='utf-8-sig')
    comp.to_csv(OUT/'相邻月报版本与口径差异.csv',index=False,encoding='utf-8-sig')
    summary={'source_reports':len(records),'category_rows':len(frame),'first_month':frame.report_month.min(),'last_month':frame.report_month.max(),
             'date_status_report_counts':frame.drop_duplicates('report_month').publication_date_status.value_counts().to_dict(),
             'scope_first_months':frame.groupby('category_scope').report_month.min().to_dict(),
             'adjacent_same_scope_share_revision_rows':int((comp.same_scope&comp.previous_month_shares_difference_100m.abs().gt(.02)).sum()),
             'all_original_first_publication_times_proven':False,'return_research_run':False,
             'next_work':'先恢复疑似迁移的原始发布日期；份额月变动优先使用同一份月报内的当月与上月，类别变更不跨口径求差。另核510300份额来源时钟与拆分，再做新版本账户。'}
    (OUT/'amac_structure_receipt.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 公募申赎因子的官方资料与可比口径','',
           f"已归档{len(records)}份基金业协会官方月报，覆盖2020年2月至2026年6月，整理股票基金与混合基金共{len(frame)}条记录。原文、网址、哈希及目录记录均保存。",'',
           '每条记录分别保留本期与前一期的基金数量、份额和资产净值；优先使用同一份月报内的两个期间计算变化，以免混入后续修订或分类切换。份额以亿份计，资产净值以亿元计。', '',
           '它们提供申赎线索，但不能精确识别现金净申购：份额变化还可能包含新成立、清盘、份额折算与类别变化；股票基金覆盖的投资方向也不全是沪深300。资产净值变化同时包含价格涨跌，因此绝不能直接把规模变化当净申购。','',
           '## 本次发现的实质问题','',
           '目录日期按原样保存，不能直接宣称就是首次公开日期。部分旧报告的日期成批相同，明显滞后于统计月份；这些记录还需核对原始发布，而不能改成推测的次月日期进入回测。报告上传目录也有2023年的迁移痕迹。本次没有发现目录日期早于统计月末的记录。',
           '后期报告从开放式基金下的股票、混合子类别转成按资产类型列示的全基金分类，不能把分类切换时的份额差额当作净申购。每期同时提供的前月数据可用于同口径比较，但仍需保留类别标记。','',
           '| 发布日检查 | 报告数 |','| --- | ---: |']
    translations={'LIST_DATE_PLAUSIBLE_ORIGINAL_TIME_NOT_INDEPENDENTLY_PROVEN':'日期顺序合理，但未独立证明首次发布', 'INVALID_LIST_DATE_NOT_AFTER_OBSERVATION_MONTH':'日期不晚于统计月末，不能使用', 'SUSPECT_MIGRATION_OR_LATE_REPUBLICATION_DATE':'滞后超过六十二日，需查迁移或重发'}
    for k,v in summary['date_status_report_counts'].items():lines.append(f'| {translations[k]} | {v} |')
    lines += ['',f"在统计范围相同的相邻月报中，有{summary['adjacent_same_scope_share_revision_rows']}条类别记录的前月份额与上份报告不同，已经逐条保留差额。不能用最新修订值覆盖历史记录。",'',
              '## 盈利、估值和股东回报的接入方法','',
              '盈利与估值需要相同的公司、财报期间和股本口径。企业盈利增长、市盈率变化与现金分红对持有回报的贡献分别估计；不把点位除以市盈率得到的隐含盈利当作独立预测。',
              '股份回购保留授权计划、已执行金额、注销与股本变化。只有已执行的行为才能进入历史事实；股份回购金额不直接当成股东收到的现金，并避免与每股盈利提升重复计量。',
              '已有510300账户包含真实分红登记与到账。增加股息率解释变量时，不把同一分红再次加到账户收益。现有份额与估值文件仍需要核对原始发布时间、历史财报版本和拆分；本文件没有声称这些字段已全部可用于有效策略。','',
              '下一研究先解决这些来源问题，同时推进不依赖它们的完整账户。结果未达到夏普1.2，持续研究保持运行。']
    (OUT/'盈利估值股东回报与公募申赎_来源进展及口径.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
