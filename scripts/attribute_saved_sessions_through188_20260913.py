"""按保存份额拆分日内隔夜及新建仓首日，不生成假想账户。"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest,now,require,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_saved_session_attribution_through188'
SOURCES={'ACCOUNT_VOLATILITY_EXPOSURE':'account_volatility_exposure','EPISODE_ACCOUNT_RISK_BUDGET':'episode_account_risk_budget'}
RULE_URL='https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20180806_4607055.shtml'


def main():
    began=time.perf_counter()
    require(not OUT.exists(),'本次保存时段归因已经存在')
    OUT.mkdir()
    feature_path=ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet'
    market=pd.read_parquet(feature_path).set_index('date')
    files=[Path(__file__),feature_path]
    summaries,daily=[],[]
    for model,study in SOURCES.items():
        for period in ['evaluation','earlier_diagnostic']:
            for cost in ['BASE','STRESS']:
                path=ROOT/f'reports/research/510300_{study}_v1'/period/cost/f'{model}_ledger.parquet'
                files.append(path)
                x=pd.read_parquet(path)
                old=x.shares.shift(fill_value=0)
                prior_mark=x.mark.shift()
                prior_mark.iloc[0]=market.loc[market.index<x.date.iloc[0],'close'].iloc[-1]
                overnight=old*(x.open-prior_mark)
                intraday=x.shares*(x.mark-x.open)
                dividend=x.dividend_recognized
                fees=x.commission+x.slippage_cost
                np.testing.assert_allclose(overnight+intraday,x.price_pnl,atol=1e-7,rtol=0)
                np.testing.assert_allclose(overnight+dividend+intraday-fees,x.pnl,atol=1e-7,rtol=0)
                fresh=(old==0)&(x.filled_quantity>0)
                row={'model':model,'period':period,'cost':cost,'trading_days':len(x),'net_profit':float(x.pnl.sum()),
                    'overnight_price_pnl':float(overnight.sum()),'dividend_recognized':float(dividend.sum()),
                    'overnight_plus_dividend':float((overnight+dividend).sum()),'intraday_price_pnl':float(intraday.sum()),
                    'commission_and_slippage':float(fees.sum()),'fresh_entries':int(fresh.sum()),
                    'fresh_entry_intraday_pnl':float(intraday[fresh].sum()),'fresh_entry_intraday_positive_count':int((intraday[fresh]>0).sum())}
                summaries.append(row)
                daily.append(pd.DataFrame({'model':model,'period':period,'cost':cost,'date':x.date,
                    'prior_shares':old,'after_open_shares':x.shares,'overnight_price_pnl':overnight,
                    'dividend_recognized':dividend,'intraday_price_pnl':intraday,'fees':fees,'net_profit':x.pnl,
                    'fresh_entry':fresh}))
    pd.DataFrame(summaries).to_csv(OUT/'session_attribution.csv',index=False,encoding='utf-8-sig')
    pd.concat(daily,ignore_index=True).to_parquet(OUT/'daily_components.parquet',index=False)
    write_json(OUT/'source_files.json',[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in files],exclusive=True)
    result={'completed_at':now(),'status':'SAVED_EIGHT_ACCOUNTS_SESSION_PNL_RECONCILED','saved_accounts':8,
        'daily_rows':sum(x['trading_days'] for x in summaries),'new_accounts':0,'new_models':0,
        'not_new_strategy_round':True,'goal_achieved':False,'run_seconds':time.perf_counter()-began,
        'sources':{'historical_sse_closing_mechanism_notice':RULE_URL},'summaries':summaries}
    write_json(OUT/'result.json',result,exclusive=True)
    lines=['# 两套保留方案的日内与隔夜损益归因','',
        '本诊断只读取第181、182轮的八条已保存模拟账户，逐日拆分份额损益，未删除隔夜收益、未创建新账户、未计算可交易的“只留日内”夏普。', '',
        '隔夜价格损益按上一收盘持有份额乘以开盘与上一收盘标记价的差计算；日内按开盘成交后的份额乘以收盘与开盘价差计算。终点开盘清仓日没有日内持仓收益。股息权利确认单独计入，避免把除息价格下降误判成额外交易损失。', '',
        '|方案|历史|费用|隔夜价格加股息|日内价格损益|佣金加滑点|全账户净利润|',
        '|---|---|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{'第181轮' if r['model']=='ACCOUNT_VOLATILITY_EXPOSURE' else '第182轮'}|{'主' if r['period']=='evaluation' else '较早'}|{'基础' if r['cost']=='BASE' else '压力'}|{r['overnight_plus_dividend']:,.2f}|{r['intraday_price_pnl']:,.2f}|{r['commission_and_slippage']:,.2f}|{r['net_profit']:,.2f}|")
    lines += ['', '金额单位为元，各账户初始二十万元。不同账户净值、仓位、股息权利与费用路径不同，不能把这些金额比直接解释为分量夏普，也不能把它们线性拼成另一账户。',
        '第181轮主历史新建仓30次，首日日内价格损益基础约8702元、压力约8558元；较早16次相应约12396元和12463元。因而本次保存账簿不支持“入场首日日内整体亏钱，所以应统一推迟买入”的解释。',
        '主历史两套方案的含股息隔夜贡献为负；较早第181轮也为负，但第182轮为正。不能据此直接删掉所有隔夜：本项目保留T+1约束，当日新买入不能通过同日普通卖出消除第一晚持仓；改变退出时点还会改变后续份额、现金和费用。',
        f'核对历史执行机制时，上交所2018年公告明确此次股票收盘集合竞价调整不改变基金等品种的收盘价产生方式。不能把股票收盘集合竞价假设套到全部ETF历史，再把日线收盘价视为保证成交价。[上交所2018年公告]({RULE_URL})。此处仅引用该历史公告，不据此声称已确认2026年所有品种的最新细则。',
        '下一项只检验使用已知星期和原下一开盘执行的固定周期减仓，不引入收盘成交或分钟数据。周末风险减少能否覆盖失去的日内收益与新增交易费用，需要另行完整账户计算；本归因并不证明其有效。']
    (OUT/'日内隔夜损益说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
