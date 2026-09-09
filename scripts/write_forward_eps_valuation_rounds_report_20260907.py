"""输出估值来源、股数差异、两轮完整规则与实际账户解释，不生成审阅包。"""
from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,now

R17=ROOT/'reports/research/510300_forward_eps_report_valuation_policy_v1'
R18=ROOT/'reports/research/510300_forward_eps_optional_valuation_v1'
SOURCE=ROOT/'reports/research/510300_forward_eps_report_valuation_source_v1'
SHARES=ROOT/'reports/research/510300_forward_eps_share_snapshot_reconciliation_v1'
OUT=ROOT/'deliverables/510300前瞻EPS估值与持续进出场_20260907'
NAMES={'V1_EPS_REPORTED_VALUATION':'EPS加研报估值空间',
       'V2_MATCHED_EPS':'相同估值有效月份的仅EPS对照',
       'V3_EPS_VALUATION_COHERENT_TREND':'EPS加估值空间并采用一致趋势进出场',
       'O1_EPS_OPTIONAL_VALUATION':'EPS持续更新，估值按成熟误差补充',
       'C0_ORIGINAL_EPS':'原完整EPS模型，复用对照','BUY_HOLD':'买入持有510300，复用对照'}


def run():
    a,b=read(R17/'result.json'),read(R18/'result.json')
    for directory in [R17,R18]:
        assert read(directory/'saved_numerical_verification.json')['status'].startswith('PASS')
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for round_number,result in [(17,a),(18,b)]:
        for row in result['all_metrics']:
            rows.append({'轮次':round_number,'策略':NAMES[row['model']],'记录代号':row['model'],
                         '费用情景':'基础' if row['cost']=='BASE' else '压力','年化收益率':row['annualized_return'],
                         '完整账户净夏普':row['net_sharpe'],'累计收益率':row['cumulative_return'],
                         '最大回撤':row['max_drawdown'],'实际成交笔数':row['trade_count'],
                         '佣金元':row['commission'],'滑点成本元':row['slippage_cost']})
    assert len(rows)==16
    pd.DataFrame(rows).to_csv(OUT/'第十七十八轮十六条账户结果.csv',index=False,encoding='utf-8-sig')
    paths={'V1_EPS_REPORTED_VALUATION':R17,'V2_MATCHED_EPS':R17,'V3_EPS_VALUATION_COHERENT_TREND':R17,
           'O1_EPS_OPTIONAL_VALUATION':R18,'C0_ORIGINAL_EPS':R18}
    ledgers={name:pd.read_parquet(path/'evaluation/BASE'/(name+'_ledger.parquet')) for name,path in paths.items()}
    trade_rows=[]
    for name,path in paths.items():
        if name=='C0_ORIGINAL_EPS':continue
        for row in ledgers[name].loc[lambda x:x.filled_quantity.ne(0)].itertuples():
            action='进入或重新进入' if row.filled_quantity>0 and row.shares_before==0 else '增加' if row.filled_quantity>0 else '全部退出' if row.shares==0 else '减少'
            reason=getattr(row,'execution_exit_reasons','')
            if not reason:reason='研究终点统一开盘退出' if row.mark_clock=='OPEN_TERMINAL' else '月末预测收益、风险和交易成本比较'
            trade_rows.append({'策略':NAMES[name],'成交日期':str(row.date.date()),'判断日期':str(row.origin.date()),
                               '操作':action,'成交份额':int(row.filled_quantity),'成交后份额':int(row.shares),
                               '成交价格元':row.fill_price,'原因':reason})
    pd.DataFrame(trade_rows).to_csv(OUT/'四个新候选基础费用全部进出场.csv',index=False,encoding='utf-8-sig')
    coverage=pd.read_parquet(SOURCE/'monthly_valuation_features.parquet')
    coverage[['origin','valuation_company_count','all_valuation_features_valid']].to_csv(OUT/'估值观点月度覆盖.csv',index=False,encoding='utf-8-sig')
    pd.read_parquet(SHARES/'snapshot_pairs.parquet').to_csv(OUT/'七十二份研报股数配对.csv',index=False,encoding='utf-8-sig')
    font=FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
    fig,axes=plt.subplots(2,1,figsize=(12,6.8),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    colours=['#1867A2','#CE6A26','#777777']
    for name,colour,label in zip(['V1_EPS_REPORTED_VALUATION','O1_EPS_OPTIONAL_VALUATION','C0_ORIGINAL_EPS'],colours,
                                 ['EPS与估值同时可用才更新','EPS持续更新，估值只作补充','原EPS对照']):
        ledger=ledgers[name]
        axes[0].plot(ledger.date,ledger.equity/200000,color=colour,lw=1.5,label=label)
    axes[0].set_title('净值改善需要与信息是否持续更新一起看',fontproperties=font,fontsize=15,loc='left')
    axes[0].set_ylabel('基础费用后净值',fontproperties=font)
    axes[0].legend(prop=font,loc='upper left',frameon=False)
    visible=coverage.loc[coverage.origin.ge('2020-01-01')]
    axes[1].step(visible.origin,visible.valuation_company_count,where='post',color='#1867A2',lw=1.5,label='有明确估值观点的成分公司数')
    axes[1].axhline(10,color='#AE3636',ls='--',lw=1,label='固定门槛：十家公司')
    axes[1].set_ylabel('公司数量',fontproperties=font)
    axes[1].legend(prop=font,loc='upper right',frameon=False)
    for ax in axes:
        ax.axvspan(pd.Timestamp('2025-04-01'),pd.Timestamp('2026-08-14'),color='#DDB05C',alpha=.15)
        ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[1].set_xlim(pd.Timestamp('2020-01-02'),pd.Timestamp('2026-08-14'))
    fig.text(.075,.015,'浅色区域：2025年4月后估值覆盖不足；蓝线继续持有，不代表仍有新的估值预测。',fontproperties=font,fontsize=10)
    fig.tight_layout(rect=[0,.04,1,1])
    figure=OUT/'净值与估值覆盖.png';fig.savefig(figure,dpi=180,facecolor='white');plt.close(fig)
    statements=[f'''# 前瞻EPS估值与持续进出场：第十七、十八轮结果

生成时间：{now()}。目标夏普1.2尚未达到，本轮继续保持未完成。以下全部为研究账户，不是实际持仓指令；未生成GPT数值审阅包。

## 一、结果与关键区别

第十七轮“EPS加研报估值空间”基础净夏普0.687，是当前历史观察的较高点；但相同月份、没有新估值因子的EPS对照也达到0.679。2025年4月后估值覆盖一直没有达到十家公司门槛，普通月末方案保持旧份额，不能把后半程收益解释为持续估值判断成功。

因此独立登记第十八轮：EPS正常更新，只有估值资料和成熟误差样本充分时才补充修正。其基础净夏普0.643，压力费用0.633，略高于原EPS的0.619和0.609；但估值只实际参与两个历史月末，增量区间仍跨零，不能称为稳定新优势。

两轮共四个新增候选，实际生成八条新账户、复用八条对照，共十六条评价账户，不是十六个独立实验。

## 二、相同完整账户口径

初始20万元，2020年1月2日至2026年8月14日开盘退出，1604个交易日，保留早期无预测的现金期。现金和夏普参考收益按零，每年242个交易日年化。100份整手、T+1、下一交易日开盘执行，计入涨跌停成交限制、分红和应收、实际现金以及佣金滑点。

基础费用为佣金万分之二、最低5元、单边滑点万分之五；压力费用为佣金万分之四、最低5元、单边滑点千分之一。最大回撤负号代表下跌。最后统一退出是评价终点规定，不能冒充模型自主卖点。

| 轮次与方案 | 费用 | 年化收益 | 净夏普 | 最大回撤 | 成交笔数 |
|---|---|---:|---:|---:|---:|''']
    for row in rows:
        if row['轮次']==18 and row['记录代号'] in ['C0_ORIGINAL_EPS','BUY_HOLD']:continue
        statements.append(f"| 第{row['轮次']}轮：{row['策略']} | {row['费用情景']} | {row['年化收益率']:.2%} | {row['完整账户净夏普']:.3f} | {row['最大回撤']:.2%} | {row['实际成交笔数']} |")
    statements.append(f'''
![完整净值和每月估值信息覆盖](<{figure.as_posix()}>)

## 三、估值因子实际增加了多少

第十七轮主方案基础终值305,939.84元，共同月份EPS对照304,738.56元，差额仅1,201.28元。主方案相对共同月份对照的20日区块年化算术收益增量区间约为负0.043至正0.223个百分点，跨越零。不能把从原0.619到0.687的整段变化归功于新增估值因子。

第十八轮EPS始终保留原36个可预测月末，估值修正只在2025年2月28日和3月31日进入判断。前者把60日收益预测从约3.01%修正到负0.11%，后者从约4.18%修正到1.81%；对应下一交易日按同一收益风险仓位规则调整。估值缺失时使用独立有效的EPS基线，缺失修正仍记录为缺失。

该方案基础终值比原EPS多2,108.69元，年化收益提高约0.11个百分点。20日区块的年化算术增量区间约负0.650至正0.945个百分点，60日区块约负0.416至正0.611个百分点，均跨零。两次历史修正不足以证明可以持续改善。

## 四、这轮核对了哪些估值和股数资料

固定2791份已有EPS原件中，2714份有明确参考收盘价，2718份有原文总市值，402份同时有合理估值及报价。新因子只用同份报告的合理价区间中点相对参考报价的比例，不从正文猜目标，不把缺失填零。402份的中点空间全部为正，来源具有明显偏多特征，不把目标价当一定能实现的回报。

按历史真实成分和最新报告选择，增长与估值各至少30家公司、利润修正至少15家，另有至少十家公司提供明确目标空间，得到31个共同有效月末。模型最终仅有17次可用月末预测，最后一次为2025年3月31日。2025年4月至2026年7月连续16个月目标空间覆盖不足；EPS本身仍可能有效，这正是第十八轮需要保持EPS基线的原因。

72份明确总股本快照来自2017年3月至2021年8月，覆盖38家公司；55份与更早披露财报的差异在原百万股显示精度内，17份超出该范围，全部保留。相符不证明期间没有公司行为，差异也不能未经核对就都归因于增发。旧财报是回取后按公开日期重建的资料，不是逐日不可变历史快照。

这些快照不能直接填补2022年以后交易日的股数。已匹配7175条有EPS公司月份中的7092条月末价格，83条缺价；对应行情里的历史股数和总市值字段为空。因此尚未完成“当前价格下、已校正股数的前瞻市盈率”，也没有把研报市值除价格的近似数量冒充精确原始股数。

## 五、目前可明确归纳的失败原因

1. 因子来源覆盖和信息更新时间会改变交易日历。减少更新后偶然避开一些交易，可能提高历史分数，必须与相同日历对照比较。
2. 新估值信息本身增量有限；研报目标偏多、覆盖下降、真正可用修正月份很少，难以支撑持续稳定超额。
3. 原EPS模型仍有买入时点误判；此前盈利分布更丰富也未自然改善收益。
4. 明确退出规则不等于更好的整体表现。第十七轮一致趋势方案发生两次趋势退出和一次预测过期退出，之后仍可能因资料不足而空仓，基础夏普为负0.332。

下一步继续扩大和校正前瞻盈利信息的可比性，核对历史权重代表性、更多免费机构预测、公司行为与股数时钟，并保留公募净申购、股东回报和全部期限逆回购方向。未达到真实验收前不宣布夏普1.2实现，也不改旧参数或删除失败账户。

## 六、第十八轮的实际进入与退出

| 成交日期 | 操作 | 成交份额 | 成交后份额 |
|---|---|---:|---:|''')
    for row in trade_rows:
        if row['策略']==NAMES['O1_EPS_OPTIONAL_VALUATION']:
            statements.append(f"| {row['成交日期']} | {row['操作']} | {row['成交份额']:,} | {row['成交后份额']:,} |")
    statements.append('\n## 七、运行前写定的全部中文规则\n')
    for filename in ['510300_FORWARD_EPS_REPORT_VALUATION_POLICY_V1.md','510300_FORWARD_EPS_OPTIONAL_VALUATION_V1.md']:
        content=(ROOT/'docs'/filename).read_text(encoding='utf-8')
        lines=[]
        for line in content.splitlines():
            lines.append('##'+line if line.startswith('#') else line)
        statements.append('\n'.join(lines))
    statements.append('''

## 八、保存与核对

本目录附十六账户结果、四个新增候选基础费用完整成交、估值月度覆盖、72份研报股数配对四份普通CSV及净值图。全部规则为中文，没有用代码替代。

九项必要测试通过；402份目标估值对照保存原文、共同训练时钟、34个第十七轮模型、两个第十八轮修正模型、十六条账户及保存区间核对通过。核对未重训、生成新账户、重新抽样或下载，不宣称重新核验全部原PDF字节；没有额外安全审计、外部GPT审阅或新数值审阅包。所有旧结果保留，研究目标继续。
''')
    (OUT/'前瞻EPS估值_两轮结果与完整进出场.md').write_text('\n'.join(statements),encoding='utf-8')
    print('两轮中文结果、规则、普通CSV与净值覆盖图已保存。',str(OUT),flush=True)


if __name__=='__main__':
    run()
