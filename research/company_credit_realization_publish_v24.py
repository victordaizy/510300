"""保存经营机制的原表证据、图表与本轮结论，不生成策略或账户。"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from company_credit_realization_v24 import OUT, NAMES, now, sha, save


def drivers():
    facts=[]
    def f(symbol,year,page,label,unit,values,interpretation):
        name=f'{symbol}_{year}Q1'
        pages=json.loads((OUT/'sources'/(name+'.pages.json')).read_text('utf-8'))
        text=pages[page-1]['text']
        for v in values:
            assert v in text,(name,page,label,v)
        facts.append(dict(symbol=symbol,company=NAMES[symbol],pdf=name+'.pdf',pdf_page=page,metric=label,unit=unit,raw_values=' | '.join(values),interpretation=interpretation,source_sha256=sha(OUT/'sources'/(name+'.pdf'))))
    f('601318.SH',2021,5,'归母营运利润：2021、2020','百万元',['39,120','35,914'],'管理层调整口径，与法定归母净利润分列。')
    f('601318.SH',2020,5,'归母营运利润：2020、2019','百万元',['35,914','34,119'],'用于相同披露口径的两年金额对照，不称现金利润。')
    f('601318.SH',2021,9,'寿险及健康险新业务价值：2021、2020','百万元',['18,980','16,453'],'不是集团归母净利润，也不是当期收到的保费现金。')
    f('601318.SH',2020,9,'寿险及健康险新业务价值：2020、2019','百万元',['16,453','21,642'],'2020共同比较值一致；两份报告均采用11.0%风险贴现率。')
    f('601318.SH',2021,9,'新业务价值率：2021、2020、原报同比降幅','百分比；百分点',['31.4','33.4','2.1'],'原表降幅2.1个百分点有未四舍五入因素，不用显示值相减覆盖原文。')
    f('601318.SH',2021,6,'华夏幸福相关投资：减值及估值调整、税后归母净利影响、税后归母营运利润影响','亿元',['182','100','29'],'4月23日原报已披露；三个数为不同口径，不能相加。不能解释全部股价下跌。')
    f('601318.SH',2021,16,'信用减值损失：2021、2020','百万元',['22,628','16,705'],'公司说明包含投资资产减值计提；不与华夏幸福182亿元再次加总。')
    f('601318.SH',2021,16,'其他资产减值损失：2021、2020','百万元',['7,237','706'],'资产质量与贷款投放量需分别看待。')
    f('600036.SH',2021,7,'集团贷款、存款余额及各自较年初增长','亿元；百分比',['53,125.29','5.64','58,272.14','3.53'],'较年初的存量变化，不能改写成同期贷款同比增速。')
    f('600036.SH',2021,7,'集团净息差水平、同比下降、环比上升','百分比；基点',['2.52','4','11'],'同一个息差可以同比下降而环比改善。')
    f('600036.SH',2021,17,'集团净利息收入：2021、2020','百万元',['49,524','45,756'],'利息收入与利息支出相减的净收入。')
    f('600036.SH',2021,17,'集团净手续费佣金收入：2021、2020','百万元',['27,202','22,061'],'新增盈利也来自手续费业务，不可全部归给信贷规模。')
    f('600036.SH',2021,7,'本行日均活期存款占比及相对上年全年提升','百分比；百分点',['65.13','4.27'],'本行口径；不是集团数字，也不是全国M1组成。')
    f('600036.SH',2021,8,'本行净利息收益率及同比、环比变化','百分比；基点',['2.58','3','10'],'本行解释涉及活期存款成本下降与较早投放贷款；不能替代集团净息差2.52%。')
    f('000333.SZ',2021,6,'营业收入：2021、2020','千元',['82,504,017','58,013,031'],'销售规模增长，保持与营业总收入科目的区别。')
    f('000333.SZ',2021,6,'营业成本：2021、2020','千元',['63,526,113','43,428,300'],'营业成本增长快于营业收入；据原科目计算毛利率变化，不宣称全部由原料涨价造成。')
    f('600519.SH',2021,6,'税金及附加：2021、2020','元',['3,827,695,095.72','2,447,742,851.57'],'原报解释涉及生产公司向销售子公司的销量与消费税附加；不同于现金流量表缴纳的各项税费。')
    f('600519.SH',2021,6,'合同负债：季末、年初','元',['5,340,569,221.03','13,321,549,147.69'],'原报解释为经销商预付货款减少；不能只凭余额变化量推出终端需求下降。')
    f('000858.SZ',2021,6,'合同负债：季末、年初','元',['4,985,609,933.78','8,618,543,467.25'],'7页解释为此前经销商打款与本季使用预收款共同作用。')
    f('600276.SH',2021,5,'研发费用：2021、2020','元',['1,316,228,078.71','811,204,994.62'],'研发投入增加；费用不等于全部当期研发现金付款。')
    f('600276.SH',2021,3,'本期计提股权激励费用','万元',['18,603.03'],'原报另有管理层剔除口径；本轮坚持法定归母利润，不以剔除后增速替换。')
    df=pd.DataFrame(facts)
    df.to_csv(OUT/'results/21项经营机制与原表说明.csv',index=False,encoding='utf-8-sig')
    computed={
       'midea_operating_revenue_cost_margin_2021':1-63526113/82504017,
       'midea_operating_revenue_cost_margin_2020':1-43428300/58013031,
       'midea_margin_yoy_pp':100*((1-63526113/82504017)-(1-43428300/58013031)),
       'pingan_nb_value_yoy':18980/16453-1,
       'pingan_nb_value_vs_2019':18980/21642-1,
       'pingan_operating_profit_yoy':39120/35914-1,
       'pingan_operating_profit_vs_2019':39120/34119-1,
       'cmb_group_net_interest_yoy':49524/45756-1,
       'cmb_group_fee_yoy':27202/22061-1,
       'hengrui_rd_expense_yoy':1316228078.71/811204994.62-1,
       'moutai_taxes_and_surcharges_yoy':3827695095.72/2447742851.57-1,
    }
    save('driver_calculations.json',computed)
    save('source_version_note.json',{'at':now(),'hengrui_correction_published_date':'2021-06-04','corrected_area':'股东总数及前十名股东持股表','earnings_and_cashflow_replaced':False,'basis':'更正说明第3页明确其他内容不变；本轮使用4月20日文件中的金额。','financial_report_count':12,'supplemental_correction_notices':1,'official_catalog_query_failures':2,'mirror_sources':['贵州茅台2020Q1发行人报告PDF的东方财富镜像','贵州茅台2021Q1发行人报告PDF的东方财富镜像','恒瑞更正说明的新浪发行人公告镜像'],'first_vintage_verified':False,'midea_pdf_note':'财务报表为图像页，原第13、14页已渲染人工读取，现金流两年度分项均与小计相符。'})
    print('已保存21项经营机制证据及计算结果。')


def chart():
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    f=pd.read_csv(OUT/'results/24项同比与2019基数比较.csv')
    p=pd.read_csv(OUT/'results/六家公司_公告前后与原20日收盘路径.csv')
    symbols=list(NAMES); names=[NAMES[s] for s in symbols]
    profit=f[f.metric.eq('归母净利润')].set_index('symbol').loc[symbols]
    fig,ax=plt.subplots(1,2,figsize=(16.8,7.9),gridspec_kw={'width_ratios':[1.05,1.1]})
    fig.patch.set_facecolor('#f6f5f0')
    for a in ax:a.set_facecolor('#f6f5f0');a.axvline(0,color='#888880',linewidth=.8);a.grid(axis='x',alpha=.18);a.set_axisbelow(True)
    y=np.arange(6)
    ax[0].barh(y-.16,profit.yoy_2021*100,height=.28,color='#1b706c',label='2021Q1 相对 2020Q1')
    ax[0].barh(y+.16,profit.total_change_vs_2019*100,height=.28,color='#bb9457',label='2021Q1 相对 2019Q1（两年累计）')
    for i,(_,r) in enumerate(profit.iterrows()):
        for v,off in [(r.yoy_2021*100,-.16),(r.total_change_vs_2019*100,.16)]:
            ax[0].text(v+(1 if v>=0 else -1),i+off,f'{v:+.2f}%',va='center',ha='left' if v>=0 else 'right',fontsize=10)
    ax[0].set(yticks=y,yticklabels=names,xlim=(-54,57),xlabel='归母净利润变动（%）',title='利润同比都增长，离2019年的距离却不同')
    ax[0].invert_yaxis();ax[0].legend(loc='upper left',bbox_to_anchor=(0,-.15),frameon=False,fontsize=10)
    window_names=['季报公开后首个收盘至货币观察日','E0原20日的成分收盘参考','E1原20日的成分收盘参考']
    colors=['#bb9457','#28516c','#88a8b7']
    labels=['季报后首个收盘 → 6月10日（各公司起点不同）','原 E0 对应20日的成分收盘参考','原 E1 对应20日的成分收盘参考']
    for j,w in enumerate(window_names):
        v=p[p.window.eq(w)].set_index('symbol').loc[symbols]
        for i,(_,r) in enumerate(v.iterrows()):
            offset=(j-1)*.23
            if r.status!='OK':
                ax[1].text(.35,i+offset,'缺失：6月2日公司行动未对清',va='center',fontsize=8.5,color='#8e6255');continue
            value=r.return_value*100
            ax[1].barh(i+offset,value,height=.2,color=colors[j],label=labels[j] if i==0 else None)
            ax[1].text(value+(.35 if value>=0 else -.35),i+offset,f'{value:+.2f}%',va='center',ha='left' if value>=0 else 'right',fontsize=9)
    ax[1].set(yticks=y,yticklabels=names,xlim=(-19,16),xlabel='固定股数、现金分红留存的收盘参考收益（%）',title='已公布盈利增长，也没有保证后面20日上涨')
    ax[1].invert_yaxis();ax[1].legend(loc='upper left',bbox_to_anchor=(0,-.15),frameon=False,fontsize=9)
    fig.suptitle('2021年5月货币病例：把信用结构、盈利兑现和价格先后连接起来',x=.05,y=.97,ha='left',fontsize=19,fontweight='bold',color='#203436')
    fig.text(.05,.9,'观察时点 2021-06-10 21:00　｜　原前五权重 + 既定医药病例　｜　2021-05-31参考权重合计19.626%',fontsize=11,color='#536364')
    fig.text(.05,.035,'左图均为已公开季度金额；右图是不同起点的历史路径，不能据此认定盈利或信贷造成股价变化。\n成分收盘参考不是ETF开盘可成交收益；原510300的20日毛收益仍为 E0 −3.42%、E1 −1.15%。',fontsize=10,color='#536364')
    fig.subplots_adjust(left=.1,right=.97,top=.81,bottom=.25,wspace=.32)
    fig.savefig(OUT/'figures/公司盈利基数与原价格路径.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)
    print('已生成公司经营与价格时序图。')


def complete():
    # 仅核对本轮真实依赖，不以文件存在代替研究有效性。
    f=pd.read_csv(OUT/'results/24项同比与2019基数比较.csv')
    a=pd.read_csv(OUT/'results/96个主要财务原表金额.csv')
    cash=pd.read_csv(OUT/'results/经营净现金同比变化_金额桥接.csv')
    p=pd.read_csv(OUT/'results/六家公司_公告前后与原20日收盘路径.csv')
    for (_,metric),g in a.groupby(['symbol','metric']):
        common=g[g.value_year.eq(2020)]
        assert len(common)==2 and common.value_cny.iloc[0]==common.value_cny.iloc[1]
    for symbol,g in cash.groupby('symbol'):
        expected=f[(f.symbol.eq(symbol))&f.metric.eq('经营活动净现金')].iloc[0].change_2021_vs_2020_cny
        assert abs(g.contribution_cny.sum()-expected)<.001
    receipts=json.loads((OUT/'input_receipts.json').read_text('utf-8'))
    for r in receipts:assert sha(OUT/'inputs'/r['name'])==r['sha256']
    old=json.loads((OUT/'freeze.json').read_text('utf-8'))
    assert old['protocol_sha256']==sha(OUT/'protocol.json')
    assert old['selection_sha256']==sha(OUT/'inputs/固定六家公司.csv')
    src=json.loads((OUT/'source_receipts.json').read_text('utf-8'))
    assert len(src)==12
    for r in src:assert sha(OUT/'sources'/r['name'])==r['sha256']
    correction=json.loads((OUT/'supplement_receipt.json').read_text('utf-8'))
    assert sha(OUT/'sources'/correction['name'])==correction['sha256']
    assert p.status.eq('OK').sum()==17 and p.status.ne('OK').sum()==1
    assert (pd.to_datetime(a.published_date)<pd.Timestamp('2021-06-10T21:00:00')).all()
    assert (OUT/'第二十四轮_信用结构如何接到公司盈利与价格.md').exists()
    save('verification.json',{'at':now(),'status':'PASS_FIXED_FACTS_AND_PATH_IDENTITIES','headline_atoms':len(a),'cross_report_2020_comparisons':24,'cashflow_bridges':4,'price_windows_total':18,'price_windows_admitted':17,'preserved_missing_window':'美的2021-05-06至2021-06-10，公司行动未对清','frozen_input_hashes_unchanged':True,'reports_before_money_observation':True,'source_first_vintage_verified':False,'independent_financial_validation':False,'visual_checks':'美的图像页现金流原值及最终图已经目视核对；其他数字的文本提取与原页定位保留。'})
    for name in ['company_credit_realization_facts_v24.py','company_credit_realization_paths_v24.py','company_credit_realization_publish_v24.py','discover_company_reports_v24.py']:
        shutil.copy2(Path(__file__).parent/name,OUT/'code'/name)
    save('completion.json',{'at':now(),'status':'COMPLETED_FIXED_COMPANY_REALIZATION_AND_BASE_COMPARISON','previous_goal_turn_classification':'PROGRESS','this_goal_turn_classification':'PROGRESS','goal_status':'active','goal_achieved':False,'new_evidence':['六家公司利润均同比增长，但美的有低基数、平安仍有信用损失与新业务价值未回到2019的问题','四家非金融权重公司合并经营现金均低于2019Q1，原因不同，且茅台与美的包含金融业务流量','一季报已公开41至51天，五条合格事前路径四涨一跌；原后20日六家成分收盘路径E0/E1均为负','缺失美的公告到货币起点路径未填补；原ETF历史结果未改'],'certainty_supported':'经营恢复、现金回款、信用质量和新增可获价格空间不是同一个事实；原单病例已否定这些条件足以保证20日上涨。','certainty_not_supported':'未找到可独立验证的未来方向确定性；不能把这一个已知结果病例提升为通用做空或择时规律。','new_models':0,'new_accounts':0,'orders_authorized':False,'global_mandate_modified':False})
    files=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name!='file_index.csv':files.append({'path':path.relative_to(OUT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    pd.DataFrame(files).to_csv(OUT/'file_index.csv',index=False,encoding='utf-8-sig')
    print('本轮证据、报告与图表已完成；未来方向确定性未达到，目标保持进行中。')


if __name__=='__main__':
    {'drivers':drivers,'chart':chart,'complete':complete}[sys.argv[1]]()
