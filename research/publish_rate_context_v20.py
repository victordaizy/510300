"""发布全体月份的利率来源外推结果，保留不能推广的证据。"""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
import numpy as np
import pandas as pd

from rate_context_generalization_v20 import OUT, digest, now, save

VISIBLE=Path(r"C:\Users\戴周阳\Documents\New project 8\reports\research\510300_rate_context_generalization_v20")
LABELS={"N_UP_R_UP":"名义升、实际升","N_UP_R_NONUP":"名义升、实际未升","N_NONUP_R_UP":"名义未升、实际升","N_NONUP_R_NONUP":"名义未升、实际未升"}
SUPPORTED="剪刀差改善_信贷与订单共同支持"


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(map(str,r))+" |" for r in rows])


def link(label,name):
    return f"[{label}](<{(VISIBLE/name).as_posix()}>)"


def main():
    if (OUT/"completion.json").exists():
        raise RuntimeError("本轮已完成，不覆盖。")
    v=json.loads((OUT/"verification.json").read_text("utf-8"))
    assert v["status"]=="PASS_RAW_CLOCK_205_OUTCOMES_AND_COMPLETE_GROUP_RECOMPUTATION"
    read=lambda name:pd.read_csv(OUT/"results"/name)
    m=read("104个月_原宏观价格背景与事前事后利率.csv")
    groups=read("全部四种利率组合_原E0E1与事后对照.csv")
    comparison=read("唯一主比较与固定延迟对照.csv")
    corr=read("事前事后连续关联_全部保留.csv")
    context=read("各组起点宏观价格与波动背景.csv")
    old=m[m.training_regime.eq("M1_OLD_M2_MMF2018")]
    rare=old[old.main_before_state.eq("N_UP_R_NONUP")].sort_values("stat_month")
    joint=old[old.joint_credit_orders_state.eq(SUPPORTED)].sort_values("stat_month")
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#a5afbb"})
    fig,axs=plt.subplots(1,3,figsize=(19,7.7),gridspec_kw={"width_ratios":[1,1.25,1.45]})
    fig.patch.set_facecolor("#f7f8fa")
    for ax in axs:
        ax.set_facecolor("white");ax.grid(alpha=.14);ax.set_axisbelow(True)
    ax=axs[0]
    for i,(regime,label) in enumerate([("M1_OLD_M2_MMF2018","旧口径84个月"),("M1_NEW2025","新口径19个月")]):
        a=corr[corr.regime.eq(regime)&corr.clock.eq("main")&corr.entry.eq("E0")&corr.field.eq("real")&corr.outcome.eq("E0_20_return")].set_index("timing")
        for off,key,name,color in [(-.17,"BEFORE_ORIGIN","起点已知变化","#218787"),(.17,"DURING_WINDOW_NOT_INPUT","持有期间变化","#b36559")]:
            value=a.loc[key,"spearman"]
            ax.barh(i+off,value,height=.27,color=color,label=name if i==0 else None)
            ax.text(value-.025 if value<0 else value+.025,i+off,f"{value:+.3f}",ha="right" if value<0 else "left",va="center",fontsize=11)
    ax.set_xlim(-1,.15);ax.set_yticks([0,1],["旧口径\n84个月","新口径\n19个月"]);ax.invert_yaxis()
    ax.axvline(0,color="#7d8793",lw=.8);ax.set_xlabel("与原后20日收益的Spearman关联")
    ax.set_title("新口径同期关联较强\n起点关联仍弱",loc="left",fontweight="bold",pad=18)
    ax.legend(loc="upper left",frameon=False,fontsize=10)
    ax=axs[1]
    y=np.arange(len(rare))
    for shift,entry,color in [(-.16,"E0","#218787"),(.16,"E1","#b36559")]:
        vals=rare[entry+"_20_return"].to_numpy()*100
        ax.barh(y+shift,vals,height=.27,color=color,label=entry)
        for yy,value in zip(y+shift,vals):
            ax.text(value+(.45 if value>=0 else -.45),yy,f"{value:+.2f}%",ha="left" if value>=0 else "right",va="center",fontsize=9)
    ax.set_yticks(y,rare.stat_month);ax.invert_yaxis();ax.set_xlim(-20,31);ax.axvline(0,color="#7d8793",lw=.8)
    ax.set_xlabel("原20日固定股数、现金分红毛收益／%")
    ax.set_title("名义升、实际未升的6个月\n上涨与失败全部保留",loc="left",fontweight="bold",pad=18)
    ax.legend(frameon=False,loc="lower right",fontsize=10)
    ax=axs[2]
    norm=colors.TwoSlopeNorm(vmin=-25,vcenter=0,vmax=25)
    sc=ax.scatter(joint.pre_return60_pp,joint.E0_20_return*100,s=joint.valuation_original_pe_official*12,
                  c=joint.main_origin_real_delta20_bp,cmap="coolwarm",norm=norm,edgecolor="white",linewidth=1)
    offsets={"2019-03":(-55,-17),"2019-11":(6,6),"2019-12":(5,-18),"2020-07":(5,-8),"2020-08":(-40,14),
             "2020-09":(-10,11),"2020-10":(-18,-21),"2020-11":(6,4),"2020-12":(12,0),"2021-01":(6,-8),"2021-05":(6,-5)}
    for r in joint.itertuples():
        ax.annotate(r.stat_month,(r.pre_return60_pp,r.E0_20_return*100),xytext=offsets[r.stat_month],textcoords="offset points",fontsize=9)
    ax.axhline(0,color="#8a94a1",lw=.8);ax.axvline(0,color="#8a94a1",lw=.8)
    ax.set_xlim(-6,35);ax.set_ylim(-15,15);ax.set_xlabel("起点之前60日价格涨幅／%");ax.set_ylabel("原E0后20日毛收益／%")
    ax.set_title("同样信贷与订单共同支持的11个月\n起点价格和利率仍有差别",loc="left",fontweight="bold",pad=18)
    cb=fig.colorbar(sc,ax=ax,orientation="horizontal",fraction=.07,pad=.17)
    cb.set_label("颜色：起点TIPS近20记录变化／基点",fontsize=10)
    fig.suptitle("把三个解释案例放回全部月份后，单向利率规则没有站稳",x=.06,ha="left",y=.98,fontsize=21,fontweight="bold")
    fig.text(.06,.897,"保留原104月、旧新M1口径、E0/E1及来源延迟对照；没有新增模型或仓位。",color="#5a6776",fontsize=12)
    fig.text(.06,.07,"右图：仅原11个共同支持月，全体保留；气泡面积随原CSI300市盈率变化。三图均为历史描述，未作独立验证。",color="#5a6776",fontsize=10)
    fig.text(.06,.036,"数据截止2026-09-11。起点美国利率按数据日美东日末可见假定；期间变化不是起点输入。E0/E1是原固定20日毛收益。",color="#5a6776",fontsize=10)
    fig.subplots_adjust(left=.07,right=.965,top=.77,bottom=.24,wspace=.38)
    figure="figures/全部月份_利率解释与起点判断的区别.png"
    fig.savefig(OUT/figure,dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
    lines=["**第二十轮：把利率来源的三个案例放回全部104个月**", "",
      "本轮更明确的结论：上一轮的三段行情能够展示不同传导渠道，但不能推广为‘实际利率升就看空、实际利率降就看多’。加入完整历史、原信贷订单状态和来源时钟后，事前关系很弱，少数月份的高均值也缺乏稳定成员与相近背景。三例解释保留，起点交易判断尚未得到支持。", "",
      "上一轮分类为PROGRESS，本轮在其证据上增加全期检验：新增13份财政部原XML，复用5份，覆盖2018—2026年同日名义和TIPS数据。原104个月全部保留，84个旧M1月份、20个新口径月份分别处理。只有103个起点在原行情截止日前可见，E0完整103个，E1完整102个；2026年8月的原起点在9月14日，继续NO_VIEW，2026年7月E1未成熟继续缺失。", "",
      "**先固定比较，再读取结果。** 起点仍是原金融数据发布后第一个21点快照。对美国10年名义、TIPS和差额，计算当时已知值相对前20条美国记录的变化；美国记录跨年连续，不混成20个A股交易日。美国日末可见只是原研究的保守时钟假设，另保留整条来源再迟一记录。变化边界固定为零，不搜索高低阈值；‘未升’包括恰为零。主比较预先固定为旧M1下名义上升时，实际上升组减未上升组的后20日E0收益差，所有其余组合和E1一起保留。", "",
      "**旧口径的完整四组结果并没有给出确定方向。**", ""]
    subset=groups[groups.regime.eq("M1_OLD_M2_MMF2018")&groups.clock.eq("main")&groups.scope.eq("ALL")&groups.timing.eq("BEFORE_ORIGIN")]
    rows=[]
    for key,label in LABELS.items():
        a=subset[subset.group.eq(key)].set_index("entry");r=a.loc["E0"]
        rows.append([label,int(r.n),f"{r.mean_percent:+.2f}%",f"{r.median_percent:+.2f}%",f"{int(r.positive_n)}/{int(r.n)}",f"{r.worst_return_percent:+.2f}%",f"{a.loc['E1','mean_percent']:+.2f}%"])
    lines += [table(["起点利率近20记录变化","月数","E0均值","E0中位数","E0正收益月","E0最差单窗","E1均值"],rows), "",
      "原起点名义升、实际未升组只有6个月，平均+3.37%，中位数+1.39%，同时包含−11.09%的亏损窗口；该窗口E1为−13.96%。实际与名义同升的39个月，E0平均−0.35%，仍有18个月正收益。两组均值差看起来有方向，但它不是确定性分类，也不能自动认定是实际利率的增量作用。", "",
      "**这六个月是完整列表，尤其保留亏损和均值集中来源。**", ""]
    rows=[]
    for r in rare.itertuples():
        rows.append([r.stat_month,r.snapshot_at[:10],f"{r.main_origin_real_delta20_bp:+.0f}",f"{r.E0_20_return*100:+.2f}%",f"{r.E1_20_return*100:+.2f}%",r.joint_credit_orders_state])
    lines += [table(["统计月","原判断日","起点实际变化/bp","E0收益","E1收益","原信贷订单与剪刀差状态"],rows), "",
      "2020年的两个月E0平均+15.20%，2021年的两个月平均−4.99%；2018年一个月+1.66%，2022年一个月−1.86%，2019、2023、2024年没有该组月份。这些都参与原结果，没有为保住均值删年。", "",
      "**同样的6个月数量，并不意味着相同的6个月。** 来源滞后一记录之后，原组只有2020年11月、2021年1月仍在；2018年3月、2020年5月、2021年3月、2022年2月退出，2019年11月、2020年8月、2022年5月、2023年1月进入。分类依赖差值是否跨零，边界附近的一条记录会改变标签。不能将这种敏感性解释为当日新增利率消息的因果效果。", ""]
    a=comparison[comparison.regime.eq("M1_OLD_M2_MMF2018")&comparison.scope.eq("ALL")]
    rows=[]
    for r in a.itertuples():
        rows.append(["原时钟" if r.clock=="main" else "来源迟一记录",r.entry,f"{r.up_n}/{r.nonup_n}",f"{r.up_mean_percent:+.2f}%",f"{r.nonup_mean_percent:+.2f}%",f"{r.difference_pp:+.2f}"])
    lines += [table(["来源时钟","收益版本","实际升/未升月数","实际升均值","实际未升均值","升减未升/百分点"],rows), "",
      "冻结方案要求两组各至少8个月且覆盖至少4个自然年才计算区块描述区间。本轮所有主比较和固定对照都未满足，保留未计算，没有降低要求或改用日频重复样本制造数量。没有区间也不能把点估计当成统计确认。", "",
      "**背景并不相近，不能把收益差全归给利率来源。**", ""]
    a=context[context.regime.eq("M1_OLD_M2_MMF2018")&context.clock.eq("main")&context.scope.eq("ALL")]
    rows=[]
    for field,label,scale in [("orders_first_release_value","制造业新订单指数",1),("valuation_original_pe_official","原CSI300市盈率",1),
       ("corporate_long_yoy_percent","年初累计新增企业中长期贷款同比/%",1),("household_long_yoy_percent","年初累计新增住户中长期贷款同比/%",1),
       ("pre_return60_pp","此前60日价格涨幅/%",1),("internal_breadth20","内部上涨参与代理/%",100),("v_downside20","年化下行尺度/%",100),
       ("funding_gap_bp","FDR007减政策利率近20日均值/bp",1)]:
        values=[]
        for group in ["N_UP_R_UP","N_UP_R_NONUP"]:
            r=a[a.group.eq(group)&a.field.eq(field)].iloc[0]
            values.append(f"{r['mean']*scale:.2f}（n={int(r.n)}）")
        rows.append([label]+values)
    lines += [table(["起点背景，组内均值","名义升、实际升","名义升、实际未升"],rows), "",
      "高均值组同时有更高的订单、较强的中长期信用背景和更高的估值，但内部参与范围反而较窄；下行尺度和国内定盘资金价差均值又比较接近。信贷分项有原口径断点和缺失，表中逐项报告可用数量。这里的贷款增幅是年初累计新增量同比，不是贷款余额同比。上述变量共同变化，不能从边际分组均值中识别单个渠道。", "",
      "在原11个剪刀差改善、信贷与订单共同支持月中，原时钟实际与名义同升只有3个月，E0平均+1.42%；名义升而实际未升只有2个月，平均+0.33%。方向与全体比较不同。来源延迟后变成3个月对4个月，均值+1.42%对+2.13%。它既不能证明实际升更好，也不能证明实际降更好；同一联合经营标签仍不足以固定后续方向。11个月完整散点和逐月底表保留了此前价格、估值与利率差异。", "",
      f"![全样本对照](<{(VISIBLE/figure).as_posix()}>)", "",
      "**起点知道的变化，与以后才发生的变化，必须分开。** 以下Spearman只描述单调关联，范围−1至+1，不是胜率，也不是解释比例；全部名义、实际和差额均列出，不选择最好的一列。", ""]
    rows=[]
    for regime,label in [("M1_OLD_M2_MMF2018","旧口径84月"),("M1_NEW2025","新口径19月")]:
        for field,name in [("nominal","名义10年"),("real","TIPS实际10年"),("compensation","通胀补偿差额")]:
            z=corr[corr.regime.eq(regime)&corr.clock.eq("main")&corr.entry.eq("E0")&corr.field.eq(field)&corr.outcome.eq("E0_20_return")].set_index("timing")
            rows.append([label,name,f"{z.loc['BEFORE_ORIGIN','spearman']:+.3f}",f"{z.loc['DURING_WINDOW_NOT_INPUT','spearman']:+.3f}"])
    lines += [table(["制度与有效月数","利率成分","起点前变化—后收益","持有期间变化—同期收益"],rows), "",
      "新口径19个完整E0月份中，窗口内实际利率变化与同期收益的关联为−0.786，看起来很强；但起点前已经知道的实际利率变化与后收益只有−0.076。E1对应−0.695与−0.036。能较好描述同期变动，不等于提前知道下一段利率或股价变化；新口径样本也更短。旧口径实际利率两项均约−0.10，三个历史案例呈现的强烈印象并未在全部旧口径月份复现。", "",
      "旧口径事前通胀补偿与后收益的关联约+0.322，E1约+0.326，但新口径对应−0.099、−0.019。它只是完整诊断表中的一个结果，通胀补偿还包含风险和流动性因素，不能从相关系数挑出另一条单指标规则。此处没有拟合新模型，也未恢复V14或旧冻结失败分支。", "",
      "**本轮给下一步判断留下的约束。** 不能把‘经营支持+实际利率未升’当作已经确认的上行状态；不能用后来发生的利率变化补足起点缺失的理由；不能把多种宏观条件简单计票。若要进一步解释和利用差异，需要直接识别当时新出现的经营预期、政策约束或资金行为变化，再核对价格已经反映多少，以及下行波动和内部参与是否同步改变。应以具体的新信息和传导为对象，继续追踪完整历史事件，不能为了获得更好的均值调整这次的零边界或期限。", "",
      "本轮完成18份原XML、2186条同日记录（其中2173条在原截止前可见）、1030条起点/开盘/终点时钟记录和205条原20日现金收益路径的复算。最差路径按原定义含入场瞬间零和其后收盘权益，不是盘中最大跌幅或峰值回撤。当前历史数据的首发不可变版本仍未认证；美国日末可见为假定，本轮也不是独立验证。没有新增账户或可执行买卖规则，总目标保持未完成。", "",
      "直接文件："+"；".join([link("104个月完整连接","results/104个月_原宏观价格背景与事前事后利率.csv"),
       link("固定主比较","results/唯一主比较与固定延迟对照.csv"),link("全部四种组合","results/全部四种利率组合_原E0E1与事后对照.csv"),
       link("六个月组成员变化","results/两套六个月的全部成员变化.csv"),link("全部背景比较","results/各组起点宏观价格与波动背景.csv"),link("复算结果","verification.json")])+"。", "",
      "美国原值来自财政部固定年份XML：[名义曲线示例](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value=2024)、[TIPS实际曲线示例](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_real_yield_curve&field_tdr_date_value=2024)。所有年份原文件和来源回执均保存在本轮sources目录。中国货币、信贷、订单、估值、资金和市场数据沿冻结输入，完整原出处随底表保留。"]
    report="第二十轮_全体月份的利率来源与联合背景.md"
    (OUT/report).write_text("\n".join(lines),encoding="utf-8")
    save("publication_pending_visual.json",{"at":now(),"report":report,"report_sha256":digest(OUT/report),"figure":figure,
         "figure_sha256":digest(OUT/figure),"goal_achieved":False})
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    print(json.dumps({"报告":str(VISIBLE/report),"图":str(VISIBLE/figure),"字符数":len('\n'.join(lines))},ensure_ascii=False))


if __name__=="__main__":
    main()
