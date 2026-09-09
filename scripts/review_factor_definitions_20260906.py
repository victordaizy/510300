"""实质检查既有因子定义与来源，生成逐项中文说明；不改旧冻结结果。"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_factor_definition_review_20260906"
PARENT=ROOT/"reports/research/510300_adaptive_allocation_v1"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def independent_price(data):
    """从未复权开高低收和分红事件独立重建三十四列。"""
    p=pd.read_parquet(Path(r"E:\ResearchData\New project 8\data\raw\market\510300_daily_downside_risk_v1.parquet"))
    p["date"]=pd.to_datetime(p.date)
    p=p.loc[p.date<=data.date.max()].sort_values("date").reset_index(drop=True)
    div=pd.read_csv(Path(r"E:\ResearchData\New project 8\data\reference\510300_dividends.csv"))
    dv=pd.Series(div.cash_dividend_per_share.to_numpy(),index=pd.to_datetime(div.ex_date)).groupby(level=0).sum()
    cash=p.date.map(dv).fillna(0)
    previous=p.close.shift(1)
    rr=(p.close+cash)/previous-1
    lr=np.log1p(rr)
    wealth=(1+rr.fillna(0)).cumprod()
    result={}
    for w in [1,2,5,10,20,60,120,252]:result[f"mom{w}"]=lr.rolling(w).sum()
    for w in [5,20,60,120,200]:result[f"sma{w}"]=wealth/wealth.rolling(w).mean()-1
    for w in [5,20,60]:result[f"vol{w}"]=rr.rolling(w).std(ddof=1)*np.sqrt(242)
    for w in [20,60]:result[f"downvol{w}"]=np.sqrt(rr.clip(upper=0).pow(2).rolling(w).mean()*242)
    for w in [20,60,120]:result[f"dd{w}"]=wealth/wealth.rolling(w).max()-1
    for part,series in {"overnight_log":np.log((p.open+cash)/previous),"intraday_log":np.log((p.close+cash)/(p.open+cash))}.items():
        for w in [1,5,20]:result[f"{part}_{w}"]=series.rolling(w).sum()
    result['range']=(p.high-p.low)/previous
    result['close_location']=((p.close-p.low)/(p.high-p.low).replace(0,np.nan)).fillna(.5)
    result['volume_ratio']=np.log(p.volume/p.volume.rolling(20).mean())
    dw=wealth.diff()
    result['efficiency20']=(dw.rolling(20).sum().abs()/dw.abs().rolling(20).sum().replace(0,np.nan)).fillna(0)
    up=dw.clip(lower=0).rolling(2).sum()
    down=-dw.clip(upper=0).rolling(2).sum()
    result['rsi2']=(100*up/(up+down).replace(0,np.nan)).fillna(50)
    result['z20']=((wealth-wealth.rolling(20).mean())/wealth.rolling(20).std(ddof=1).replace(0,np.nan)).fillna(0)
    result['vol_ratio']=result['vol20']/result['vol60'].clip(lower=1e-12)
    checks={}
    for c,series in result.items():
        a,b=series.to_numpy(),data[c].to_numpy()
        missing_equal=np.array_equal(np.isnan(a),np.isnan(b))
        error=float(np.nanmax(np.abs(a-b)))
        checks[c]={"missing_pattern_equal":missing_equal,"maximum_absolute_error":error,"passed":bool(missing_equal and error<1e-9)}
    pure=np.log(p.close/p.open)
    mask=cash>0
    diagnostics={"dividend_day_count":int(mask.sum()),"intraday_wealth_vs_pure_price_max_difference":float((result['intraday_log_1']-pure).loc[mask].abs().max()),
                 "dividend_decomposition":"带现金权益的财富分解满足恒等式，但日内项不是纯开盘到收盘价格收益"}
    return checks,diagnostics

def price_description(c):
    if re.fullmatch(r"mom\d+",c):return f"最近{c[3:]}个交易日含分红日收益加一后的自然对数之和", "含分红对数回报"
    if re.fullmatch(r"sma\d+",c):return f"累计含分红财富除以最近{c[3:]}日财富均值，再减一", "比例"
    if re.fullmatch(r"vol\d+",c):return f"最近{c[3:]}日含分红收益样本标准差，乘二百四十二的平方根", "年化比例"
    if c.startswith('downvol'):return f"最近{c[7:]}日将正收益记零，负收益平方的日均值乘二百四十二后开方", "年化下行波动"
    if re.fullmatch(r"dd\d+",c):return f"当前累计含分红财富除以最近{c[2:]}日最高财富，再减一", "比例"
    if c.startswith('overnight_log_'):return f"最近{c.split('_')[-1]}日隔夜财富变化之和；开盘价加当日每份分红，除以前收盘价后取自然对数", "含现金权益的财富分解"
    if c.startswith('intraday_log_'):return f"最近{c.split('_')[-1]}日日内财富变化之和；收盘价加分红，除以开盘价加分红，再取自然对数", "含现金权益的财富分解"
    values={"range":("最高价减最低价，再除以前收盘价","比例"),"close_location":("收盘价减最低价，除以当日最高最低价差；无振幅时取二分之一","零至一"),
            "volume_ratio":("当日成交量除以含当日的二十日平均成交量，再取自然对数","无量纲，不是净申购"),
            "efficiency20":("二十日财富净变化的绝对值，除以逐日财富变化绝对值之和；无变化取零","零至一"),
            "rsi2":("最近两日正财富变化之和除以两日涨跌幅绝对值之和，再乘一百；分母为零取五十","零至一百，简单两日比例"),
            "z20":("当前财富减二十日均值，再除以二十日财富样本标准差；标准差为零取零","标准差倍数"),
            "vol_ratio":("二十日年化波动率除以六十日年化波动率，分母设极小正数下限","倍数")}
    return values[c]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(PARENT/"features.parquet")
    pc=read(PARENT/"input_receipt.json")["features"]
    checks,diag=independent_price(data)
    rows=[]
    def add(c,group,meaning,unit,clock,issue,status,frame):
        x=pd.to_numeric(frame[c],errors='coerce')
        rows.append({"因子字段":c,"因子组":group,"中文定义":meaning,"单位或含义":unit,"可用时点":clock,"问题或使用边界":issue,"本次检查结论":status,
                     "非缺失行数":int(x.notna().sum()),"缺失行数":int(x.isna().sum()),"最小值":float(x.min()),"最大值":float(x.max())})
    for c in pc:
        desc,unit=price_description(c)
        issue="使用历史供应商行情；没有逐日首次交付快照证明。"
        if 'intraday_log' in c:issue+="日内项含现金权益，不能解释为纯开盘买入、收盘卖出的回报。"
        if c=='rsi2':issue+="是简单两日涨跌比例，不是使用递归平滑的标准相对强弱指标。"
        if c=='volume_ratio':issue+="二级市场成交量不是基金一级市场净申赎。"
        add(c,'价格与财富',desc,unit,'当日收盘后，次日开盘执行',issue,'逐值独立复算通过' if checks[c]['passed'] else '复算不一致',data)
    extroot=ROOT/'reports/research/510300_overnight_global_information_v1'
    ext=pd.read_parquet(extroot/'features.parquet')
    extcols=read(extroot/'source_receipt.json')['external_features']
    alignment=pd.read_parquet(extroot/'source_clock_alignment.parquet')
    valid=alignment.available_at.notna() & alignment.decision_time.notna()
    clockerrors=int((alignment.loc[valid,'available_at']>alignment.loc[valid,'decision_time']).sum())
    for c in extcols:
        sym,suffix=c.split('_',1)
        label={'GSPC':'标普五百指数','IXIC':'纳斯达克综合指数','HSI':'恒生指数','N225':'日经二二五指数','VIX':'波动率指数'}[sym]
        if suffix.startswith('mom'):desc=f"{label}最近{suffix[3:]}个来源交易日的收盘对数收益之和"
        elif suffix=='vol20':desc=f"{label}最近二十个来源交易日的收盘对数收益样本标准差，按二百四十二日年化"
        elif suffix=='dd60':desc=f"{label}收盘值除以最近六十个来源交易日最高收盘值，再减一"
        elif suffix=='age_days':desc=f"{label}来源预定可用时刻距中国开盘前判断时刻的自然日数"
        elif suffix=='log_level':desc="波动率指数收盘点位的自然对数"
        else:raise ValueError(c)
        issue="来源是指数价格或波动率指数，不是指数含分红总收益。市场收盘时钟不等于免费接口实时交付证明。"
        if sym=='VIX' and suffix=='vol20':issue+="这里衡量波动率指数自身的变化波动，不是股票实现波动率。"
        add(c,'海外观察',desc,'自然日' if suffix=='age_days' else '指数变化或对数刻度','按各市场既定收盘后时钟，下一中国开盘前九点取已知值',issue,
            '对齐记录未发现时钟穿越；本次未逐值重算该列' if clockerrors==0 else '存在时钟问题',ext)
    qroot=ROOT/'reports/research/510300_policy_liquidity_quantity_v1'
    q=pd.read_parquet(qroot/'features.parquet')
    fund={"dr007_known":"最新已知存款类机构七天质押式回购加权利率，百分数除以一百", "policy_rate_known":"最新明确公布的央行七天逆回购操作利率，百分数除以一百",
          "funding_gap":"DR007减央行七天操作利率", "dr007_change5":"DR007相对于五个股票交易日前的变化", "dr007_change20":"DR007相对于二十个股票交易日前的变化", "funding_gap_mean20":"最近二十个股票交易日资金利差均值"}
    for c,desc in fund.items():add(c,'资金利率',desc,'小数利率或利差','DR007在利率日期后的下一股票开盘可用；政策利率按公告时刻','七天利率与各期限投放总量是不同变量；不应该加总各期限利率。','已核身份、单位与时钟规则；未发现此次指出的期限漏量问题',q)
    quantities={"quantity_log":"最新一篇明确七天公告的亿元金额加一后取对数", "quantity_surprise20":"当前七天公告对数金额，相对于此前二十篇明确七天公告金额的标准分",
                "quantity_change5_20":"最近五篇明确七天公告对数规模均值减最近二十篇均值", "quantity_age_days":"最近明确七天公告距当日十五点的自然日数", "quantity_fresh_today":"最近明确七天规模公告是否属于当前日期", "quantity_funding_interaction":"七天公告规模标准分乘资金利差"}
    for c,desc in quantities.items():add(c,'旧七天操作量',desc,'公告金额对数、自然日或交互项','公告公开后使用，最多延用十个自然日','漏掉其他逆回购期限及买断式；按公告次数和最近值刻画，不能解释为每日实际总投放或净投放。','范围不足，已新建全期限版本；原结果只对七天定义有效',q)
    calroot=ROOT/'reports/research/510300_calendar_liquidity_timing_v1'
    cal=pd.read_parquet(calroot/'features.parquet')
    calcols=read(calroot/'feature_receipt.json')['calendar_features']
    cd={"month_phase_sin":"当月日期序号减一，除以当月自然日总数，再乘两倍圆周率后取正弦", "month_phase_cos":"当月日期序号减一，除以当月自然日总数，再乘两倍圆周率后取余弦", "year_phase_sin":"月份序号减一，除以十二，再乘两倍圆周率后取正弦", "year_phase_cos":"月份序号减一，除以十二，再乘两倍圆周率后取余弦",
        "month_ordinal_scaled":"当前开市日在本月已发生股票交易日中的序号除以二十二，最大取一", "remaining_calendar_days_scaled":"本月剩余自然日数除以三十一；剩余天数不含当前日",
        "month_first3":"当前是否本月前三个股票交易日", "month_last5_calendar":"当前是否本月最后五个自然日", "quarter_first3":"当前是否季初月份前三个股票交易日",
        "quarter_last5_calendar":"当前是否季末月份最后五个自然日", "reopen_gap_log":"当前开市日距此前开市日自然日间隔加一后取自然对数", "post_break_first3":"开市日与此前开市日相隔至少四个自然日时，从当前起三个开市日记一，其余记零", "month_edge":"月初前三个交易日或月末最后五个自然日"}
    for c in calcols:
        desc='当前开市日是否星期'+{'mon':'一','tue':'二','wed':'三','thu':'四','fri':'五'}[c.split('_')[-1]] if c.startswith('weekday_') else cd[c]
        add(c,'日历',desc,'标志或日期刻度','开盘前由日期及此前已发生开市日确定','只是日期条件，未包含当日实际公募申购、缴税或逆回购量；不能用日历名称冒充资金证据。','已核日期定义，沿用第六轮已完成的前缀不变验证',cal)
    require_count=len(rows)==95
    if not require_count:raise ValueError(f'因子数应为95，实际{len(rows)}')
    pd.DataFrame(rows).to_csv(OUT/'95项因子逐项中文核对.csv',index=False,encoding='utf-8-sig')
    (OUT/'price_recomputation.json').write_text(json.dumps({'columns':checks,'all_34_passed':all(x['passed'] for x in checks.values()),'diagnostics':diag,'overseas_alignment_clock_errors':clockerrors},ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 六轮95项因子的定义、问题与修正','',
           '范围是2026年9月6日恢复研究的六轮，共95个不同输入因子。重复使用同一价格因子的模型不重复计数。逐项列出实际计算含义和本次核对深度，不能把局部核对称为全部因子已证明有效。','',
           '三十四项价格因子从未复权行情和分红事件独立重建，逐值比较与缺失位置检查均通过。海外因子核对了保存的时钟对齐记录，本次没有逐值独立重算全部三十一列。利率、旧逆回购和日历逐项检查了来源与定义。','',
           '确认需要处理的问题：旧七天规模覆盖不足；旧规模是最近公告与按公告次数计算，不能等同每日总流量；日内财富分解与纯价格日内收益含义不同；两日简单涨跌比例不是递归平滑的相对强弱指标；成交量不是净申购；市场收盘时刻不自动证明接口当时可用。日历条件也不能代替实际资金观测。','']
    for group in pd.DataFrame(rows)['因子组'].drop_duplicates():
        lines += ['## '+group,'','| 因子 | 完整中文定义 | 单位 | 可用时点 | 问题或边界 | 本次结论 |','| --- | --- | --- | --- | --- | --- |']
        for r in rows:
            if r['因子组']==group:lines.append('| '+' | '.join(str(r[k]).replace('|','／') for k in ['因子字段','中文定义','单位或含义','可用时点','问题或使用边界','本次检查结论'])+' |')
        lines += ['']
    lines += ['## 新增经济因子的落实','',
              '盈利：区分实际公布的归母利润、每股盈利与从指数点位反推的隐含盈利；指数聚合使用当时成分、权重、股本和财报版本。指数点位除以市盈率属于恒等变换，不能当成独立盈利预测。',
              '估值：同一对象、同一盈利口径与同一股本下比较市盈率；亏损企业、缺失权重和财报修订必须显式处理。旧日频估值文件缺逐日历史版本证明，不能直接当作已通过点时验证的输入。',
              '股东回报：持有回报计入现金分红；股份回购区分计划、实际执行、注销、再发行和股份激励，不把回购金额直接当作股东收到的现金，不与每股盈利变化重复相加。510300账户已经计入自身分红，成分股分红收益率作为解释变量时不能再加到账户收益里。',
              '公募申赎：510300实际份额变化与全市场股票、混合基金的份额变化分别记录；基金规模变化还受净值影响，份额变化还可能包括新发、清盘、拆分、类别变更，均不等于沪深300即时现金净流入。旧份额文件的当日十五点时钟是程序设定，未附逐日原始发布时间。',
              '基金业协会官方月报已归档77份，覆盖2020年2月至2026年6月。旧目录部分发布日期呈批量迁移特征，且分类表后来发生变化；须先解决时钟与类别可比性，再按实际公开信息验证。',
              '', '本文件是研究口径检查，不是安全审计；没有覆盖或修改任何旧冻结绩效。']
    (OUT/'六轮95项因子_完整中文说明与修正.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'因子数':len(rows),'价格独立复算全部通过':all(x['passed'] for x in checks.values()),'海外已存时钟穿越':clockerrors,'分红财富分解':diag},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
