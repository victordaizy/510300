"""用真实每日价格展示增长、利率与政策更新；不画虚构策略净值。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_macro_dynamic_reframe_v1"
BLUE,TEAL,GOLD,RED,GREY="#225e91","#147f77","#b37b29","#b54d60","#64768a"


def style() -> None:
    plt.rcParams.update({"font.family":FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),"axes.unicode_minus":False,"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#bac3cc","text.color":"#25364a","axes.labelcolor":"#25364a","xtick.color":GREY,"ytick.color":GREY,"figure.facecolor":"white","svg.hashsalt":"macro_dynamic_reframe_v1"})


def save(fig, name: str) -> None:
    fig.savefig(OUT/"figures"/(name+".png"),dpi=155,facecolor="white")
    fig.savefig(OUT/"figures"/(name+".svg"),metadata={"Date":None})
    plt.close(fig)


def draw() -> None:
    style()
    d=pd.read_parquet(OUT/"results/全部日线_增长与操作利率已知记录.parquet")
    d["date"]=pd.to_datetime(d.date)
    v=d[d.date>=pd.Timestamp("2015-01-01")]
    policies=json.loads((OUT/"inputs/policy_context.json").read_text(encoding="utf-8"))
    fig,axs=plt.subplots(4,1,figsize=(15.6,11.6),sharex=True,gridspec_kw={"height_ratios":[1.7,1.2,1.15,.9]})
    fig.subplots_adjust(left=.08,right=.975,top=.87,bottom=.16,hspace=.19)
    fig.suptitle("510300 放回宏观背景：增长、利率与政策是不同的信息",x=.08,ha="left",y=.976,fontsize=21,fontweight="bold")
    fig.text(.08,.928,"价格保留每个交易日；宏观值按已公布时点进入。此图描述历史共变，不代表因果或买卖信号。",color=GREY)
    axs[0].plot(v.date,v.close,color=BLUE,lw=1.2)
    axs[0].set_ylabel("510300 收盘价（元）")
    axs[1].step(v.date,v.first_release_value,where="post",color=TEAL,lw=1.3)
    axs[1].axhline(50,color=GREY,lw=.8,ls="--")
    axs[1].set_ylabel("PMI 新订单\n（扩散指数）")
    axs[1].text(.012,.84,"实际经济状态；未取得共识时不称为“超预期”",transform=axs[1].transAxes,fontsize=10,color=GREY,bbox={"facecolor":"white","edgecolor":"none","alpha":.9})
    axs[2].step(v.date,v.seven_day_rate_percent,where="post",color=GOLD,lw=1.4)
    axs[2].set_ylabel("7 天操作利率\n本地已记录值（%）")
    axs[2].text(.012,.82,"操作公告记录，尚不是完整的政策宣布／实施时间序列",transform=axs[2].transAxes,color=RED,fontsize=10)
    axs[3].fill_between(v.date,v.total_return_drawdown*100,0,color=RED,alpha=.20)
    axs[3].plot(v.date,v.total_return_drawdown*100,color=RED,lw=.8)
    axs[3].set_ylabel("含分红回撤（%）")
    for ax in axs:
        ax.grid(axis="y",color="#e8edf1",lw=.8)
        ax.set_axisbelow(True)
    # 关键政策仅作案例标记，不按上涨与否选择训练样本。
    for row in policies:
        at=pd.Timestamp(row["date"])
        axs[0].axvline(at,color=GREY,alpha=.20,lw=.8)
    axs[0].annotate("2024-09 政策密集更新\n详见下一张时序图",xy=(pd.Timestamp("2024-09-24"),3.65),xytext=(pd.Timestamp("2023-01-01"),5.35),fontsize=10,color=GREY,arrowprops={"arrowstyle":"->","color":GREY})
    axs[-1].xaxis.set_major_locator(mdates.YearLocator())
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.08,.092,"增长数据：139 个月，2015-01 至 2026-07；利率：25 条含起始锚点的变动记录，不能据此声称政策事件完整。",fontsize=10,color=GREY)
    fig.text(.08,.062,"价格截至 2026-09-11。增长、利率各自截止以后留空；少量政策竖线仅为背景案例，未计算政策贡献率。",fontsize=10,color=GREY)
    fig.text(.08,.032,"来源：国家统计局当次发布、人民银行操作公告及项目已核对的 510300 日线。全部绘图日线另附 CSV / Parquet。",fontsize=10,color=GREY)
    save(fig,"510300_增长利率政策_宏观全景")

    case=pd.read_csv(OUT/"results/九月案例_全部交易日.csv",parse_dates=["date"])
    c=json.loads((OUT/"results/九月案例_时钟与价格核对.json").read_text(encoding="utf-8"))
    fig,axs=plt.subplots(3,1,figsize=(15.6,11.3),sharex=True,gridspec_kw={"height_ratios":[2,1,1.2]})
    fig.subplots_adjust(left=.08,right=.975,top=.85,bottom=.16,hspace=.29)
    fig.suptitle("同一段持有期内，新的政策会改变已知信息",x=.08,ha="left",y=.974,fontsize=21,fontweight="bold")
    fig.text(.08,.928,"2024 年 9 月案例：观察后出现新政策，应重新判断；先前发生的上涨不能算成更新后赚到的收益。",color=GREY)
    axs[0].plot(case.date,case.close,color=BLUE,marker="o",markersize=4,lw=1.6,label="每日收盘")
    axs[0].plot(case.date,case.open,color=GREY,marker="_",markersize=10,lw=0,label="每日开盘")
    axs[0].axvspan(pd.Timestamp("2024-09-19"),pd.Timestamp("2024-09-25"),color=GOLD,alpha=.10)
    axs[0].set_ylabel("510300 价格（元）")
    axs[0].legend(loc="upper left",frameon=False,ncol=2)
    axs[0].text(.018,.65,f"旧五日标签：9/19 开盘 → 9/25 收盘\n毛收益 {c['old_window_return']:+.2%}，包含 9/24 新政策出现后的行情",transform=axs[0].transAxes,fontsize=11,color=GOLD)
    marks=[("2024-09-18","旧判断\n观察收盘",RED), ("2024-09-24","降准降息等\n新政策宣布",TEAL), ("2024-09-26","政治局\n经济部署",TEAL), ("2024-09-27","降息\n正式实施",GOLD)]
    for date,label,color in marks:
        at=pd.Timestamp(date)
        for ax in axs:
            ax.axvline(at,color=color,lw=.9,ls=":",alpha=.6)
        axs[0].text(at,1.02,label,transform=axs[0].get_xaxis_transform(),ha="center",fontsize=9,color=color)
    calendar=pd.date_range(case.date.min(),case.date.max(),freq="D")
    announced=np.where(calendar>=pd.Timestamp("2024-09-24"),1.5,1.7)
    observed=np.where(calendar>=pd.Timestamp("2024-09-29"),1.5,1.7)
    axs[1].step(calendar,announced,where="post",color=TEAL,lw=2,label="已经宣布的拟调至水平")
    axs[1].step(calendar,observed,where="post",color=GOLD,lw=1.7,ls="--",label="旧操作利率表已记录水平")
    axs[1].set_ylim(1.44,1.78)
    axs[1].set_ylabel("7 天利率（%）")
    axs[1].legend(loc="upper left",frameon=False,ncol=2,fontsize=10)
    axs[1].text(pd.Timestamp("2024-09-24"),1.455,"当日上午政策内容已公布",fontsize=9,color=TEAL)
    axs[2].set_ylim(-.6,2.4)
    tracks=[(2,"原五日实验", "2024-09-18", "2024-09-25",RED), (1,"动态复核示意", "2024-09-24", "2024-09-25",TEAL), (0,"实施事实", "2024-09-27", "2024-09-30",GOLD)]
    for y,label,start,end,color in tracks:
        axs[2].plot([pd.Timestamp(start),pd.Timestamp(end)],[y,y],color=color,lw=5,solid_capstyle="round")
    axs[2].set_yticks([2,1,0],["原五日\n固定判断","动态复核\n示意","9/27\n正式生效"])
    axs[2].text(pd.Timestamp("2024-09-19"),2.15,"期间没有中途更新",fontsize=9,color=RED)
    axs[2].text(pd.Timestamp("2024-09-24"),.68,"更新判断后才可执行；此处未假设买入",fontsize=9,color=TEAL)
    axs[2].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for ax in axs[:2]:
        ax.grid(axis="y",color="#e8edf1")
    fig.text(.08,.092,f"若按 9/24 收盘复核、9/25 开盘执行，至旧标签终点的价格变化为 {c['illustrative_update_open_to_old_exit']:+.2%}；这只是成交时钟演示。",fontsize=10,color=GREY)
    fig.text(.08,.060,"图中未定义新策略买卖方向；上述变化没有扣费，不能当作新策略收益，更不能归因为单项政策。",fontsize=10,color=GREY)
    fig.text(.08,.028,"政策来源：证监会保存的 9/24 实时逐字稿；中国政府网转载央行微信的 9/27 实施公告。案例由用户事后指出，非独立验证。",fontsize=10,color=GREY)
    save(fig,"政策更新_宣布实施与可成交时点")

    diag=pd.read_csv(OUT/"results/旧五日实验_补充基准汇总.csv")
    fig,ax=plt.subplots(figsize=(12,5.2))
    fig.subplots_adjust(left=.26,right=.92,top=.78,bottom=.19)
    labels=["A：三项价格变量","B：价格 + 剪刀差预期差","已成熟训练事件均值","恒定预测零收益"]
    colors=[BLUE,RED,TEAL,GREY]
    y=np.arange(4)
    ax.barh(y,diag.rmse_pp,color=colors,height=.56)
    for i,r in diag.iterrows():
        ax.text(r.rmse_pp+.025,i,f"{r.rmse_pp:.3f}",va="center",fontsize=12,color=colors[i])
    ax.set_yticks(y,labels)
    ax.invert_yaxis()
    ax.set_xlim(0,2.8)
    ax.set_xlabel("五日收益 RMSE（百分点，越低越好）")
    ax.grid(axis="x",color="#e8edf1");ax.set_axisbelow(True)
    fig.suptitle("旧实验的价格基准，也尚未证明预测价值",x=.06,ha="left",y=.965,fontsize=19,fontweight="bold")
    fig.text(.06,.862,"同一批 24 个评价事件；A、B 的 24 次预测全部为负。补充基准是事后诊断，不重选策略。",fontsize=10,color=GREY)
    fig.text(.06,.045,"原结果继续保留：未建立可靠正增量。误差差值的双侧区间跨零，不应解释成宏观存在稳定负作用。",fontsize=10,color=GREY)
    save(fig,"旧子实验_四个预测基准")
    print("三张说明图及对应 SVG 已生成。")


if __name__=="__main__":
    draw()
