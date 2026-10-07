"""用固定月末原点检查510300二十日延续/反转，描述统计不冒充交易收益。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_reversal_monthly_diagnostic_v1"
SOURCE = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
CALENDAR = ROOT / "reports/research/510300_growth_state_increment_20d_v1/inputs/calendar_extended.parquet"


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    import sys
    recovery = "--recover-date-parse" in sys.argv
    if OUT.exists() and not recovery:
        raise FileExistsError("诊断已建立，不能覆盖或更换窗口")
    if recovery and ((OUT/'summary.json').exists() or (OUT/'全部月末原点与二十日收益.csv').exists()):
        raise ValueError("已经产生标签或结果，不能使用无标签恢复路径")
    OUT.mkdir(parents=True,exist_ok=recovery)
    for folder in ["inputs", "figures", "code"]:
        (OUT / folder).mkdir(exist_ok=recovery)
    import shutil
    shutil.copy2(SOURCE, OUT / "inputs/market.parquet")
    shutil.copy2(CALENDAR, OUT / "inputs/calendar.parquet")
    shutil.copy2(Path(__file__), OUT / "code" / (("corrected_" if recovery else "")+Path(__file__).name))
    plan = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "request": "用户提示A股反转多于动量，应检验而非先验断言", "class": "POST_REQUEST_FIXED_DESCRIPTIVE_DIAGNOSTIC_ALREADY_OBSERVED_HISTORY", "asset": "510300.SH", "origin": "每个完整自然月最后交易日收盘；过去20个交易日含分红回报；次日开盘持有20个交易日至收盘的现金权益回报", "start": "首个有完整20日过去回报的月末", "end": "2026-07-31，与宏观实验截止一致", "horizon": 20, "bins_pp": [-100,-10,-5,0,5,10,100], "bootstrap": {"draws": 10000,"block_months":6,"seed":20260922}, "counterexample": "过去涨随后跌与过去跌随后涨定义为反转；不是价格序列平稳性检验，也不是估计合理价值，不能把频数直接称为反转策略盈利。", "account_runs":0,"new_model_fits":0,"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"selection": "不搜索期限、阈值、样本起点或最好年份；本轮已看历史，不是未见样本。"}
    if recovery:
        plan=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        save(OUT/'date_parse_correction.json',dict(before_any_label=True,reason="原代码把截止日期后中文说明一起传入时间解析；第一次循环即失败，未产生标签或结果。仅解析字符串前10位YYYY-MM-DD，不改变截止日、窗口或统计设定。",corrected_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    else:
        save(OUT / "protocol.json", plan)
        save(OUT / "freeze.json", {"before_labels":True,"protocol_sha256":hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),"code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    m = pd.read_parquet(SOURCE).sort_values("date").reset_index(drop=True)
    m["date"] = pd.to_datetime(m.date)
    m["past20"] = np.expm1(np.log1p(m.total_simple).rolling(20).sum())
    c = pd.read_parquet(CALENDAR)
    c = c[c.is_open].copy();c["date"] = pd.to_datetime(c.trade_date)
    month_end = set(c.groupby(c.date.dt.to_period("M")).date.max())
    rows=[]
    for i,r in m.iterrows():
        if r.date not in month_end or r.date > pd.Timestamp(plan['end'][:10]) or not np.isfinite(r.past20):continue
        if i+20 >= len(m):raise ValueError("固定标签未成熟")
        b=m.iloc[i+1:i+21]
        div=b.dividend.to_numpy(float).copy();div[0]=0
        future=(float(b.close.iloc[-1])+float(div.sum()))/float(b.open.iloc[0])-1
        rows.append(dict(date=str(r.date.date()),origin_index=int(i),past20=float(r.past20),future20=future,entry_date=str(b.date.iloc[0].date()),exit_date=str(b.date.iloc[-1].date()),entry_open=float(b.open.iloc[0]),exit_close=float(b.close.iloc[-1]),dividend=float(div.sum()),pattern="上涨后回落" if r.past20>0 and future<0 else "下跌后回升" if r.past20<0 and future>0 else "上涨延续" if r.past20>0 and future>0 else "下跌延续" if r.past20<0 and future<0 else "零收益"))
    d=pd.DataFrame(rows);d.to_csv(OUT/'全部月末原点与二十日收益.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    rev=(d.past20*d.future20<0).to_numpy();valid=(d.past20*d.future20!=0).to_numpy()
    b=plan['bootstrap'];n=len(d);rng=np.random.default_rng(b['seed'])
    starts=rng.integers(0,n-5,size=(b['draws'],int(np.ceil(n/6))));idx=(starts[:,:,None]+np.arange(6)).reshape(b['draws'],-1)[:,:n]
    proportions=rev[idx].sum(1)/valid[idx].sum(1)
    np.savez_compressed(OUT/'固定区块索引.npz',indices=idx.astype(np.int16))
    x=d.past20.to_numpy();y=d.future20.to_numpy();correlations=np.asarray([np.corrcoef(x[ix],y[ix])[0,1] for ix in idx])
    pairs=[]
    for sign in [-1,1]:
        subset=d[d.past20*sign>0]
        pairs.append(dict(prior='此前下跌' if sign<0 else '此前上涨',months=len(subset),reversal_months=int((subset.future20*sign<0).sum()),continuation_months=int((subset.future20*sign>0).sum()),mean_future_return_pp=float(subset.future20.mean()*100),median_future_return_pp=float(subset.future20.median()*100)))
    chance=(x>0).mean()*(y<0).mean()+(x<0).mean()*(y>0).mean()
    summary=dict(status='COMPLETED_FIXED_DESCRIPTIVE_DIAGNOSTIC_NO_ACCOUNT',months=n,first=d.date.iloc[0],last=d.date.iloc[-1],reversal_months=int(rev.sum()),continuation_months=int(((x*y)>0).sum()),zero_months=int((~valid).sum()),reversal_fraction=float(rev.sum()/valid.sum()),reversal_fraction_block95=np.quantile(proportions,[.025,.975]).tolist(),marginal_sign_independence_reference=float(chance),correlation=float(np.corrcoef(x,y)[0,1]),correlation_block95=np.quantile(correlations,[.025,.975]).tolist(),by_prior_sign=pairs,new_models=0,account_runs=0,not_full_A_share_conclusion=True,not_stationarity_test=True)
    save(OUT/'summary.json',summary)
    bins=np.array(plan['bins_pp'])/100
    d['bin']=pd.cut(d.past20,bins=bins,include_lowest=True,right=True)
    grouped=d.groupby('bin',observed=True).agg(months=('date','size'),mean_future=('future20','mean'),median_future=('future20','median'),mean_past=('past20','mean')).reset_index()
    grouped.to_csv(OUT/'固定涨跌幅区间对照.csv',index=False,encoding='utf-8-sig')
    render(d,summary,grouped)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def render(d,summary,grouped):
    n=summary['months']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,9),facecolor='#f6f7fb');gs=fig.add_gridspec(2,2,height_ratios=[1.6,1],hspace=.40,wspace=.30)
    ax=fig.add_subplot(gs[0,:]);colors={'上涨后回落':'#d95f45','下跌后回升':'#246ea3','上涨延续':'#1b8e77','下跌延续':'#9074ad','零收益':'#888888'}
    for name,color in colors.items():
        s=d[d.pattern==name]
        if len(s):ax.scatter(s.past20*100,s.future20*100,c=color,s=36,alpha=.8,label=f'{name}  {len(s)} 次',edgecolors='white',linewidths=.3)
    ax.axvline(0,color='#6b7280',lw=.8);ax.axhline(0,color='#6b7280',lw=.8);ax.set_xlabel('此前 20 个交易日含分红回报（%）');ax.set_ylabel('次日开盘后 20 日现金权益回报（%）');ax.grid(alpha=.14);ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=5,fontsize=9,frameon=False)
    a=fig.add_subplot(gs[1,0]);counts=[summary['reversal_months'],summary['continuation_months']];bars=a.barh(['反转','延续'],counts,color=['#d95f45','#1b8e77'],height=.5)
    for bar,num in zip(bars,counts):a.text(num+1,bar.get_y()+bar.get_height()/2,str(num),va='center')
    a.set_xlim(0,max(counts)*1.3);a.set_xlabel('月末原点次数');a.set_title(f"反转占比 {summary['reversal_fraction']:.1%}；区块区间 {summary['reversal_fraction_block95'][0]:.1%}—{summary['reversal_fraction_block95'][1]:.1%}",loc='left',fontsize=11)
    a=fig.add_subplot(gs[1,1]);names=['≤−10%','−10%～−5%','−5%～0%','0%～5%','5%～10%','>10%']
    a.bar(range(len(grouped)),grouped.mean_future*100,color='#246ea3',width=.65);a.axhline(0,color='#6b7280',lw=.8);a.set_xticks(range(len(grouped)),names[:len(grouped)],rotation=18,fontsize=9);a.set_ylabel('随后平均回报（%）');a.set_title('预先固定涨跌幅区间；保留极端月份',loc='left',fontsize=11)
    for i,r in grouped.iterrows():a.text(i,float(r.mean_future)*100+(0.22 if r.mean_future>=0 else -.35),f'n={r.months}',ha='center',fontsize=9)
    a.set_ylim(min(grouped.mean_future.min()*100,0)-.85,max(grouped.mean_future.max()*100,0)+1.0)
    fig.suptitle('510300：二十日尺度上，反转是否比延续更多？',x=.07,y=.975,ha='left',fontsize=19,fontweight='bold',color='#16314a')
    fig.text(.07,.925,f"{summary['first']}—{summary['last']} · {n} 个完整月末原点 · 数据截止 2026-09-11 · 保留所有原点",color='#52606d')
    fig.text(.07,.035,'此图描述过去涨跌与未来回报的关系；反转频数不等于策略盈利，也不证明价格围绕固定均值回归。\n后续收益从下一开盘起计算；月末之间仍可能存在少量标签重叠，区间使用连续六个月区块。',fontsize=10,color='#52606d',linespacing=1.6)
    fig.subplots_adjust(top=.84,bottom=.17,left=.07,right=.95)
    fig.savefig(OUT/'figures/510300_二十日反转与延续对比.png',dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(OUT/'figures/510300_二十日反转与延续对比.svg',facecolor=fig.get_facecolor());plt.close(fig)


if __name__=='__main__':
    import sys
    if '--render-only' in sys.argv:
        render(pd.read_csv(OUT/'全部月末原点与二十日收益.csv'),json.loads((OUT/'summary.json').read_text(encoding='utf-8')),pd.read_csv(OUT/'固定涨跌幅区间对照.csv'))
        print('只重新排版保存图表；未重新计算标签、统计或模型。')
    else:main()
