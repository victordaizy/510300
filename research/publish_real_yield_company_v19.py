"""发布原三个窗口的利率来源与公司传导说明，不产生预测或仓位。"""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from real_yield_company_sources_v19 import OUT, CASES, COMPANIES, digest, now, save

VISIBLE_ROOT=Path(r"C:\Users\戴周阳\Documents\New project 8")
VISIBLE_OUT=VISIBLE_ROOT/"reports/research/510300_real_yield_company_exposure_v19"


def link(label,relative):
    return f"[{label}](<{(VISIBLE_OUT/relative).as_posix()}>)"


def md_table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+
                     ["| "+" | ".join(map(str,row))+" |" for row in rows])


def main():
    report_name="第十九轮_利率来源与权重公司经营传导.md"
    if (OUT/"completion.json").exists():
        raise RuntimeError("本轮已经交付，不覆盖。")
    verification=json.loads((OUT/"verification.json").read_text("utf-8"))
    assert verification["status"]=="PASS_SAVED_RATE_COMPANY_CLOCK_AND_CASH_RECOMPUTATION"
    m=pd.read_csv(OUT/"inputs/monthly.csv").set_index("stat_month")
    snap=pd.read_csv(OUT/"results/69个原时点_实际利率与通胀补偿.csv")
    windows=pd.read_csv(OUT/"results/三窗口_同日利率变化精确分解.csv")
    stocks=pd.read_csv(OUT/"results/18个原公司窗口_经营背景与事后收益.csv")
    daily=pd.read_csv(OUT/"inputs/daily_paths.csv")
    facts=pd.read_csv(OUT/"results/公司原表事实_金额期间范围与发布时间.csv")
    derived=json.loads((OUT/"results/公司派生事实.json").read_text("utf-8"))
    model=pd.read_csv(OUT/"results/三窗口_当前模型五项变化_仅回顾.csv")
    company=facts.groupby("symbol").company.first().to_dict()
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#a6acb5"})
    fig,axs=plt.subplots(2,3,figsize=(18,10),gridspec_kw={"height_ratios":[1,1.12]})
    fig.patch.set_facecolor("#f7f8fa")
    palette={"nominal_10y_percent":("名义收益率","#273b56"),"tips_10y_percent":("TIPS实际收益率","#b25148"),
             "inflation_compensation_10y_percent":("通胀补偿差额","#158b87")}
    for j,case in enumerate(CASES):
        a=m.loc[case];s=snap[snap.stat_month.eq(case)]
        origin=s[s.role.eq("origin")].iloc[0]
        d=s[s.role.eq("daily_close")].sort_values("cutoff_at")
        ax=axs[0,j];ax.set_facecolor("white")
        for field,(label,color) in palette.items():
            ys=np.r_[0,100*(d[field].to_numpy()-origin[field])]
            ax.plot(np.arange(21),ys,label=label,color=color,lw=2.2)
        ax.axhline(0,color="#a6acb5",lw=.7)
        ax.set_ylim(-25,65);ax.set_xlim(0,20)
        ax.set_xticks([0,5,10,15,20]);ax.set_xlabel("原窗口交易日序号（0为发布日夜间）")
        ax.set_ylabel("相对起点变化／基点")
        ax.grid(axis="y",alpha=.16)
        ax.set_title(f"统计月 {case}\n{a.E0_20_entry_date} → {a.E0_20_exit_date}",loc="left",fontweight="bold",pad=12)
        e=windows[windows.stat_month.eq(case)&windows.clock.eq("main")].iloc[0]
        ax.text(.03,.92,f"期末：{e.nominal_10y_window_change_bp:+.0f} = {e.tips_10y_window_change_bp:+.0f} + {e.inflation_compensation_10y_window_change_bp:+.0f}",
                transform=ax.transAxes,fontsize=11,color="#273b56")
        b=stocks[stocks.stat_month.eq(case)].set_index("symbol").loc[COMPANIES]
        ay=axs[1,j];ay.set_facecolor("white")
        v=b.stock_return20_cc_percent.to_numpy()
        ay.barh(np.arange(6),v,color=["#ab6258" if y>=0 else "#479391" for y in v],height=.64)
        ay.axvline(0,color="#8d97a2",lw=.8)
        ay.set_xlim(-28,38);ay.set_yticks(range(6),[company[z] for z in COMPANIES]);ay.invert_yaxis()
        ay.grid(axis="x",alpha=.16);ay.set_axisbelow(True)
        ay.set_xlabel("原20日收盘至收盘总收益／%（固定股数、现金分红）")
        for y,val in enumerate(v):
            inside=val < -8
            ay.text(val+(1 if val>=0 or inside else -1),y,f"{val:+.2f}%",va="center",
                    ha="left" if val>=0 or inside else "right",fontsize=10,color="white" if inside else "#242d38")
        ay.set_title(f"510300原开盘入场收益：{a.E0_20_return*100:+.2f}%",loc="left",fontweight="bold",pad=12)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.51,.895),ncol=3,frameon=False)
    fig.suptitle("同样是名义利率上升，来源和公司结果可以不同",x=.06,ha="left",fontsize=21,fontweight="bold",y=.985)
    fig.text(.06,.925,"固定三个连续窗口与六家原权重公司；完整保留上涨、下跌和中间情形。",color="#596474",fontsize=12)
    fig.text(.06,.053,"上排：按每个A股观察时点可见的同日10年名义/TIPS曲线；日内来源按美东日末可见假定，另有滞后一条记录对照。",fontsize=10,color="#596474")
    fig.text(.06,.028,"下排公司收益与标题ETF开盘入场收益口径不同。图中是窗口内已实现路径，不能提前作为起点信号，也不是利率的因果贡献。",fontsize=10,color="#596474")
    fig.subplots_adjust(left=.075,right=.98,top=.79,bottom=.115,wspace=.30,hspace=.42)
    figure="figures/三窗口_实际利率路径与原权重公司收益.png"
    fig.savefig(OUT/figure,dpi=170,facecolor=fig.get_facecolor())
    plt.close(fig)
    lines=["**第十九轮：名义利率的来源、公司经营渠道与510300的剩余收益**", "",
      "本轮结论是：三个窗口都出现剪刀差改善、信贷和订单共同支持，但利率变化的来源、公司现金流与金融资产负债渠道并不相同。把这些渠道放到当时的价格位置和估值旁边，能够具体解释为什么相似的宏观标签对应不同的收益；现在还不能据此给出可重复的涨跌确定性。",
      "", "本轮沿用V18全部三个连续窗口、69个观察时点、60个交易日，以及当时权重前五家公司与医药行业权重第一的恒瑞医药。三次前五恰为同五家公司。选择时已看过历史收益，因此本轮是机制解释；完整保留六家公司，不能称作独立验证。", "",
      "**先看已知的起点，随后才看窗口里发生的变化。**", ""]
    rows=[]
    for case in CASES:
        a=m.loc[case];s=snap[snap.stat_month.eq(case)&snap.role.eq("origin")].iloc[0]
        rows.append([case,a.snapshot_at[:10],f"{a.delta3_spread_pp:+.2f}",f"{a.d3_relative_current_log_pp:+.3f}",
                     f"{a.d3_relative_base_revision_log_pp:+.3f}",f"{a.pre_return60_pp:+.2f}%",f"{a.valuation_original_pe_official:.2f}",
                     f"{s.tips_10y_change20_bp:+.0f}"])
    lines += [md_table(["统计月","原判断日","剪刀差近3月变化/百分点","当期余额相对项/对数百分点","基数及版本项/对数百分点","此前60日涨幅","原CSI300市盈率","起点TIPS近20记录变化/bp"],rows), "",
      "对数口径的两项相加等于对数增速差的变化，不能拿来直接分摊普通剪刀差的6.70个百分点。2021年1月的当期相对余额项为负，正向改善主要在基数及版本项；仅凭表面剪刀差变好，不能说经营资金正在同比例加速活化。信贷各分项、PMI订单和后续经营兑现仍需分别核对。原六家三季报只提供经营背景，不代表每个判断日全部最新公司信息，更不是市场盈利预期。", "",
      "**窗口内同日利率拆解：前两次实际利率下降，第三次实际利率上升。**", ""]
    rows=[]
    for case in CASES:
        a=m.loc[case];e=windows[windows.stat_month.eq(case)&windows.clock.eq("main")].iloc[0]
        lag=windows[windows.stat_month.eq(case)&windows.clock.eq("delay1")].iloc[0]
        rows.append([case,f"{a.E0_20_entry_date}—{a.E0_20_exit_date}",f"{e.nominal_10y_window_change_bp:+.0f}",f"{e.tips_10y_window_change_bp:+.0f}",
                     f"{e.inflation_compensation_10y_window_change_bp:+.0f}",f"{lag.nominal_10y_window_change_bp:+.0f} = {lag.tips_10y_window_change_bp:+.0f} + {lag.inflation_compensation_10y_window_change_bp:+.0f}",
                     f"{a.E0_20_return*100:+.2f}%",f"{a.E1_20_return*100:+.2f}%"])
    lines += [md_table(["统计月","原20日窗口","名义10年/bp","TIPS10年/bp","通胀补偿/bp","来源额外迟一记录/bp","E0收益","E1收益"],rows), "",
      "原式为同一美国日期、同10年期限的名义CMT = TIPS实际CMT + 两者差额。它是收益率恒等式，不是股票收益因果分摊。第三窗的43基点中38基点在TIPS这一项，不能解释成‘510300跌幅的88%由实际利率导致’。E0从原发布后下一A股开盘持有20个交易日；E1再延迟一个A股交易日，仍各持有20日。两者固定股数、分红留现金，未扣成本，也不是完整账户结果。", "",
      "这一区别在发布时间上尤其关键：2月9日21点，当时已知的TIPS近20记录变化仍是−10基点；来源再迟一条时为−1基点。2月18日开盘的对应变化为+5基点，额外迟一记录则为−7基点。因此事后完整窗口的+38/+40基点很清楚，起点预警和春节后即时判断却没有同等强度的证据，不能倒推为‘2月9日已经知道要跌’。", "",
      f"![三窗口利率与原权重公司收益](<{(VISIBLE_OUT/figure).as_posix()}>)", "",
      "**同一利率环境，先区分公司收入、现金、融资、再投资和估值五条渠道。**", ""]
    rows=[]
    evidence={
      "600519.SH":"合并财务费用为负；其利息收入与营业利息收入是两栏，集团还含财务公司。",
      "601318.SH":"新业务价值同比−27.1%；投资收益、再投资风险与存量债券估值同时存在。",
      "000858.SZ":"利润增长同时出现现金流下降；回款跨年、票据托收和税款影响需拆开。",
      "600036.SH":"贷款余额较年初+11.48%，但集团净息差同比−14bp；规模与价差两条线。",
      "000333.SZ":"第三季度收入+15.71%；借款与货币性投资并存，财务费用变化主要因利息收入减少。",
      "600276.SH":"利润与经营现金同增；合并利息收入1.98亿元，不能把市场利率压力全叫借款成本。"}
    for s in COMPANIES:
        d=facts[(facts.symbol.eq(s))&facts.metric.isin(["revenue_9m","profit_9m","cfo_9m"])].set_index("metric")
        cash="金融口径，另列" if s in ["600036.SH","601318.SH"] else f"{d.loc['cfo_9m','reported_change']:+.2f}%"
        rows.append([company[s],f"{d.loc['revenue_9m','reported_change']:+.2f}%",f"{d.loc['profit_9m','reported_change']:+.2f}%",cash,evidence[s]])
    lines += [md_table(["公司","前三季收入同比","前三季归母利润同比","前三季经营现金同比","需要继续追踪的渠道"],rows), "",
      "这些是各公司原前三季度报告中的不同经营背景，不能用利润增速直接排列未来股价，也不能把金融公司的经营现金流与制造业回款混比。茅台的原合并报表也包含财务公司经营。恒瑞三季报利息费用栏为空，保持空白，不编成零；本轮未把上半年‘无借款’表述外推到年末。", "",
      f"五粮液经营现金流同比减少{abs(derived['wuliangye_cfo_decline_yi']):.2f}亿元，其中现金流入少34.36亿元，支付税费多{derived['wuliangye_tax_delta_yi']:.2f}亿元，其余经营现金支出多11.78亿元，三项精确相加。多缴税款对应现金净减少额的{derived['wuliangye_tax_accounting_share_percent']:.2f}%，这是会计分解比例。公司原文还把流入减少与一季度回款提前到上年末、上年票据到期托收额较高联系起来。其作用是限制‘现金降了就一定是销量或需求塌了’的解释，不能由此断言全国M1变动由五粮液或同一因素主导。", "",
      "招商银行集团前三季净息差2.51%，同比下降14基点，但第三季2.53%，环比上升8基点；同比与环比不能互换。本行活期存款占比61.33%、日均活期存款占比59.74%又是本公司范围，不能直接与集团净息差拼成严密金额桥。需要继续看贷款收益、负债成本、减值和普通股分配，才能判断信贷规模对股东收益的传导。", "",
      "**公告内容本身也在窗口中更新。** 招商银行2021年1月15日的原快报，全年归母利润增长4.82%；原快报减前三季报得到隐含第四季度利润增长32.70%。它在1月12日起点尚不可用，在第二个窗口中才出现。中国平安2月3日公司通稿同时给出全年归母营运利润+4.9%、归母净利润−4.2%、寿险及健康险新业务价值−34.7%，三个口径表达的是不同层面的业务结果；这份通稿在第三个起点已知。", "",
      "平安全年报告的完整发布目录未获原始证实，所以其中敏感性表只用于回顾说明：投资收益率和风险贴现率**同时**增加50基点，集团内含价值从1,328,112百万元升到1,381,825百万元；另一张表对指定公允价值债券施加政府收益率曲线平移+50基点，给出利润减少4,377百万元、权益减少14,384百万元。这两张表的对象、假设和计量不同，可以同时成立，不能相加抵销，也不能直接代入美国TIPS的38基点。", "",
      "**将六家全部收益保留，防止只讲符合故事的公司。**", ""]
    rows=[]
    for s in COMPANIES:
        d=stocks[stocks.symbol.eq(s)].set_index("stat_month")
        rows.append([company[s]]+[f"{d.loc[c,'stock_return20_cc_percent']:+.2f}%" for c in CASES]+[f"{d.loc['2021-01','original_weight_percent']:.3f}%",f"{d.loc['2021-01','original_weight_contribution_pp']:+.3f}"])
    lines += [md_table(["公司","第一窗收益","第二窗收益","第三窗收益","第三窗原权重","第三窗原权重贡献/百分点"],rows), "",
      "公司收益沿用V18的20日收盘至收盘固定股数、现金分红口径，与ETF开盘入场收益并不相同。原权重贡献只描述所选公司的算术贡献，不是ETF真实持仓归因；其余成员和原四个缺失公司窗口保持原状态。第三窗平安上涨、茅台等下跌，并不足以证明利率是唯一原因；第二窗平安也显著下跌，全部列出以限制单一叙事。", "",
      "**实际利率和通胀补偿还不是最终经济原因。** 同日差额同时含预期通胀、通胀风险和TIPS流动性因素；TIPS收益率也不等于纯预期实际短利率。可参考[美联储DKW方法说明](https://www.federalreserve.gov/econres/notes/feds-notes/tips-from-tips-update-and-discussions-20190521.html)。本轮另存当前模型的五项分解作为机制诊断，绝不放进2020/2021起点输入。", ""]
    q=model[model.stat_month.eq("2021-01")&model.clock.eq("main")].iloc[0]
    lines += [f"当前DKW版本在第三窗模型曲线中给出：预期实际短率{q['exp.real.short.rate.10_change_bp']:+.2f}基点、实际期限溢价{q['real.term.prem.10_change_bp']:+.2f}基点、预期通胀{q['exp.inflation.10_change_bp']:+.2f}基点、通胀风险溢价{q['inflation.risk.prem.10_change_bp']:+.2f}基点。四项加总约{q['nominal.yield.fitted.10_change_bp']:+.2f}基点，模型原始名义曲线变动{q['nominal.yield.raw.10_change_bp']:+.2f}基点，二者变化残差{q['nominal_fit_residual_pp_change_bp']:+.2f}基点。其期限曲线与财政部票息CMT口径不同，不能用这四项硬分摊43基点。更关键的是当前文件明确用截至2025年11月12日数据重新估计，属于事后模型，不能说2021年市场当时已经观测到这些数值。", "",
      "国内资金也不能被美国收益率覆盖：第三窗原中国10年收益率仅增加3.70基点，FDR007最近20日均值从2.4569%降到2.1172%。FDR007是定盘值，不是加权DR007；这些变化不支持把这段下跌笼统解释为‘国内所有资金价格都在收紧’。从利率到公司融资、资产估值和资金流还需要币种、期限及持仓对应证据。", "",
      "**此次立体观察得到的可用结论。**", "",
      "一，M1/M2要先拆余额、基数和版本，再核对信贷究竟流向哪里、订单和企业利润是否兑现。二，已兑现的经营改善仍可能在股价中提前体现，必须与起点估值、此前涨幅和市场内部参与范围并看。三，同一种名义利率变化可以来自不同来源，公司受影响的渠道也不同。四，波动率是结果路径的压缩表达，仍需区分上涨波动、下跌波动和旧样本滚出，不能仅贴‘高位降波/低位降波’标签。", "",
      "这些结果已经排除了‘剪刀差改善且信贷、订单支持就足以确定上涨’的说法，也把更值得检查的经营与估值冲突具体化了。但三个已知结果的连续窗口不能建立稳定的条件收益规律，更不能保证未来涨跌。本轮保持总目标未完成；若继续检验，下一项应回到既有全部合格历史起点，事先固定仅使用起点已知量的比较，保留各周期和失败例，不能把窗口内TIPS变化用作起点筛选条件，也不能从这三例倒调阈值。", "",
      "复算已经通过：502条名义/TIPS同日记录、69个原时点、138条仅回顾模型记录、54项公司原表事实、18个公司窗口及27条公告可见性。原表相关页已经目视核对；复算不等于历史首发版本认证、独立验证或因果识别。原文件首发不可变版本仍未证实，利率采用原有日末可见假设并保留延迟一记录对照。", "",
      "直接证据与结果："+"；".join([link("同日利率分解","results/三窗口_同日利率变化精确分解.csv"),
       link("公司原表事实","results/公司原表事实_金额期间范围与发布时间.csv"),link("18个公司窗口","results/18个原公司窗口_经营背景与事后收益.csv"),
       link("公告可见性","results/三窗口公司公告可见性.csv"),link("复算结果","verification.json")])+"。", ""]
    docs=facts.drop_duplicates("source_file")
    for _,row in docs.iterrows():
        lines.append(f"{row.company}原材料：{link(row.source_file,'sources/'+row.source_file)}；[官方来源]({row.source_url})。")
        lines.append("")
    (OUT/report_name).write_text("\n".join(lines),encoding="utf-8")
    # 保存展示用的三窗核心量，免去下游从474列原底表再次手工筛选。
    overview=m[["snapshot_at","delta3_spread_pp","d3_relative_current_log_pp","d3_relative_base_revision_log_pp",
                "pre_return60_pp","valuation_original_pe_official","internal_breadth20","E0_20_entry_date","E0_20_exit_date",
                "E0_20_return","E1_20_return","origin_fdr007_percent_mean20","exit_fdr007_percent_mean20","cgb10_window_change_bp"]].reset_index()
    overview=overview.merge(windows[windows.clock.eq("main")],on="stat_month",validate="one_to_one")
    overview.to_csv(OUT/"results/三个连续窗口_宏观起点与利率来源.csv",index=False,encoding="utf-8-sig",float_format="%.15g")
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    save("publication_pending_visual.json",{"at":now(),"report":report_name,"figure":figure,
         "report_sha256":digest(OUT/report_name),"figure_sha256":digest(OUT/figure),"goal_achieved":False})
    print(json.dumps({"报告":str(VISIBLE_OUT/report_name),"图":str(VISIBLE_OUT/figure),"字符数":len('\n'.join(lines))},ensure_ascii=False))


if __name__=="__main__":
    main()
