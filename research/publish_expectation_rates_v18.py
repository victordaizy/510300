"""生成连续三期传导图和便于阅读的研究结论。"""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from expectation_rates_sources_v18 import OUT, CASES, now, save, digest


def link(relative, label):
    return f"[{label}](<{(OUT / relative).as_posix()}>)"


def main():
    report_path=OUT / "第十八轮_经营扩张预期变化与权重股分化.md"
    if report_path.exists():
        raise RuntimeError("报告已生成，不重复覆盖。")
    verification=json.loads((OUT / "verification.json").read_text("utf-8"))
    assert verification["status"]=="PASS_SAVED_CLOCK_CASH_CONTRIBUTION_RECOMPUTATION"
    m=pd.read_csv(OUT / "results/三个连续月份_经营预期利率与内部传导.csv").set_index("stat_month")
    p=pd.read_csv(OUT / "results/原60个交易日_ETF波动与利率时序.csv")
    s=pd.read_csv(OUT / "results/所有行业_原权重已覆盖贡献.csv")
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei"],"axes.unicode_minus":False,
                         "font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(3,3,figsize=(18,11.6),sharex=True,sharey="row")
    fig.patch.set_facecolor("#faf9f4")
    for j,c in enumerate(CASES):
        a=m.loc[c];d=p[p.stat_month==c].sort_values("day_number")
        x=np.arange(21)
        axes[0,j].plot(x,np.r_[0,d.path_return.to_numpy()*100],color="#164c63",lw=2.4,label="510300原E0累计毛收益")
        axes[0,j].annotate(f"{a.E0_20_return*100:+.2f}%",(20,a.E0_20_return*100),xytext=(-2,9),
                           textcoords="offset points",ha="right",fontsize=12,fontweight="bold")
        axes[0,j].set_title(f"{c}统计月\n{a.E0_20_entry_date} 至 {a.E0_20_exit_date}",fontsize=12,pad=12)
        lines=[("ust10_percent","美国10年国债","#9b4f32"),
               ("cgb10_percent","中国10年国债","#164c63")]
        for col,label,color in lines:
            y=np.r_[0,(d[col].to_numpy()-a["origin_"+col])*100]
            axes[1,j].plot(x,y,label=label,color=color,lw=1.9)
        y=np.r_[0,d.fdr_policy_gap_bp_mean20.to_numpy()-a.origin_fdr_policy_gap_bp_mean20]
        axes[1,j].plot(x,y,label="FDR007－政策利率近20条均差",color="#7a782e",lw=1.9,ls="--")
        for col,origin,label,color in [("v_rv20","rv20","总波动RV20","#164c63"),
                                        ("v_downside20","downside20","下行尺度D20","#9b4f32")]:
            axes[2,j].plot(x,np.r_[a[origin],d[col].to_numpy()]*100,label=label,color=color,lw=2)
        for i in range(3):
            ax=axes[i,j];ax.set_facecolor("#faf9f4");ax.grid(axis="y",alpha=.18)
            ax.set_xlim(0,20.7);ax.set_xticks([0,5,10,15,20]);ax.axhline(0,color="#777777",lw=.7,alpha=.5)
        axes[2,j].set_xlabel("原窗口交易日（0为观察起点）")
    axes[0,0].set_ylabel("相对原开盘本金，%")
    axes[1,0].set_ylabel("相对观察起点变化，基点")
    axes[2,0].set_ylabel("最近20日年化历史尺度，%")
    axes[0,0].legend(loc="upper left",frameon=False,fontsize=9)
    axes[1,0].legend(loc="upper left",frameon=False,fontsize=8.5)
    axes[2,0].legend(loc="upper left",frameon=False,fontsize=9)
    fig.suptitle("经营共同支持状态延续，价格与利率仍走出不同路径",x=.07,ha="left",fontsize=19,fontweight="bold")
    fig.text(.07,.939,"固定三个连续月份、原20日窗口；同一行使用相同纵轴。图示为历史关联，未识别唯一因果。",fontsize=11,color="#555555")
    fig.text(.07,.02,"美债使用A股当时已经可见的美国上一记录，国内国债使用下一A股开盘可见规则；FDR007为上午定盘。\n"
                       "第0点的波动按观察日起算；收益从下一交易日开盘起算。底部两条线不能相加，总波动包含均值修正。",fontsize=9.5,color="#555555")
    fig.subplots_adjust(left=.07,right=.97,top=.89,bottom=.11,hspace=.30,wspace=.12)
    figure=OUT / "figures/三个连续月份_预期利率与波动路径.png"
    fig.savefig(figure,dpi=165,facecolor=fig.get_facecolor());plt.close(fig)
    lines=[
        "**第十八轮：经营扩张延续时，510300的新增信息、利率期限和权重贡献如何分开**",
        "本轮沿第十七轮已提出的2020年11月至2021年1月连续三期追查。更具体的发现是：**最大的剪刀差改善主要落在同比基数项；后续大跌时，美国长端利率明显上升，国内短期资金价格却已缓和，食品饮料和医药的权重贡献转为显著负值，银行与部分保险股表现不同。** 这些差异使“货币改善—流动性更松—指数上涨”无法作为一条直接推论。",
        "这三个病例的ETF结果此前已知，本轮用于解释传导和排除不符合事实的说法。实际收益、成本后完整账户和独立样本优势仍是不同问题，长期目标保持进行中。",
        "三期都满足此前固定的“剪刀差三个月改善、企业及住户中长期贷款累计同比多增、新订单至少50”状态。期间与时钟保持原样：",
        "| 统计月 | 2020-11 | 2020-12 | 2021-01 |",
        "|---|---:|---:|---:|",
    ]
    def triplet(label,values):
        lines.append("| "+label+" | "+" | ".join(values)+" |")
    triplet("实际观察日",[m.loc[c,"observation_date"] for c in CASES])
    triplet("M1同比／M2同比",[f"{m.loc[c,'m1_yoy_pp']:.1f}%／{m.loc[c,'m2_yoy_pp']:.1f}%" for c in CASES])
    triplet("剪刀差三个月改善，百分点",[f"{m.loc[c,'delta3_spread_pp']:+.1f}" for c in CASES])
    triplet("同口径对数当期相对余额项",[f"{m.loc[c,'d3_relative_current_log_pp']:+.3f}" for c in CASES])
    triplet("同口径对数基数及修订项",[f"{m.loc[c,'d3_relative_base_revision_log_pp']:+.3f}" for c in CASES])
    triplet("当时新订单指数",[f"{m.loc[c,'orders_first_release_value']:.1f}" for c in CASES])
    triplet("剪刀差相对事前调查代理，百分点",["缺失","−1.0","＋5.0"])
    triplet("观察前60日ETF收益",[f"{m.loc[c,'past_return60']*100:+.2f}%" for c in CASES])
    triplet("当时官方PE",[f"{m.loc[c,'valuation_original_pe_official']:.2f}" for c in CASES])
    triplet("原E0后20日毛收益",[f"{m.loc[c,'E0_20_return']*100:+.2f}%" for c in CASES])
    triplet("再等一天的原E1后20日毛收益",[f"{m.loc[c,'E1_20_return']*100:+.2f}%" for c in CASES])
    lines += ["", "这里的对数当期项和基数项沿用第二轮的精确余额分解，两者相加的是对数增速差变化，不能与普通百分点6.7直接混加；基数项还保留隐含基数及修订影响。2021年1月当期相对项−0.246、基数项＋6.252，已经不支持把14.7%的M1同比直接解释为当期交易性资金全面加速。此前第十一轮已核对春节错位、现金与企业活期的影响，本轮没有将其重新命名为新发现。",
        "企业、住户中长期贷款的累计同比多增分别为：2020年前11个月27,678和5,432亿元，2020全年29,200和5,000亿元，2021年1月3,800和1,957亿元。期间长度不同，不以三个金额的大小排列需求强弱。新订单从53.9到53.6再到52.3，仍在扩张区间而速度有所变化；第十七轮确认的后续经营数据也继续保留。",
        "**新增信息也存在分化。** 2020年12月M1、M2分别比事前调查低1.6、0.6个百分点；2021年1月则分别高4.4、低0.6个百分点。前者相对剪刀差代理为−1.0，后者为＋5.0。两份调查来自原先固定的Bualuang报告转载Bloomberg Survey，不能当作临公布最后一刻全部市场预期，也不能把两个汇总数之差称为逐个分析师剪刀差预测的汇总。2020年11月没有合格配对预期，仍为NO_VIEW。",
        "同页贷款调查进一步说明，信息包内部不必同向。2020年12月新增人民币贷款调查为12,500亿元，央行明确的当月值为12,600亿元，差＋100亿元；不能用旧表按累计数差分得到的12,502亿元来比月度调查。2021年1月调查35,000亿元、公布35,800亿元，差＋800亿元。社融调查两期原数同时保存；本轮未补2020年12月官方社融实际数，1月较晚官方转载的51,700亿元仍按第六轮回顾证据处理，未倒填原起点。",
        f"调查原页见{link('sources/Bls210111.pdf','2021年1月11日报告，第12页')}、{link('sources/Bls210208.pdf','2021年2月8日报告，第12页')}；全部对应项见{link('results/两个事前调查_中美全部对应数字.csv','中美调查及实际数值表')}。",
        "利率要分清期限和市场。下表比较原观察时点与原E0退出收盘时已经可见的数值；1个基点等于0.01个百分点。",
        "| 变化项 | 2020-11病例 | 2020-12病例 | 2021-01病例 |",
        "|---|---:|---:|---:|",
    ]
    for label,col in [("美国2年债，基点","ust2_window_change_bp"),("美国10年债，基点","ust10_window_change_bp"),
                      ("美国10年债再滞后1条记录，基点","ust10_delay1_window_change_bp"),("中国10年债，基点","cgb10_window_change_bp"),
                      ("FDR007减政策利率近20条均差，基点","fdr_policy_mean20_change_bp")]:
        triplet(label,[f"{m.loc[c,col]:+.2f}" for c in CASES])
    triplet("美元兑人民币中间价变化",[f"{m.loc[c,'midpoint_window_change_percent']:+.2f}%" for c in CASES])
    lines += ["", "第一段ETF上涨11.76%时，美国长债上升12个基点，国内国债与短期资金价格下降；美债再滞后一条记录后涨幅为2个基点，提示第一段端点依赖较明显。第二段国内短期资金均差上升48.54个基点，ETF仍小幅上涨。第三段ETF下跌11.09%时，美国长债上升43个基点，延后记录仍为45个基点；国内资金均差下降33.97个基点。**同样是“利率变化”，期限、市场与发生时间不同，不能互相替代。**",
        "第三段美国10年债从1.19%到1.62%，2年债从0.11%到0.14%；国内10年债从3.2370%到3.2740%。国内FDR007近20条平均从2.45688%降到2.117195%，同期已知7天逆回购政策利率均为2.20%。这是定盘指标的缓和，不足以代表所有机构的融资条件；中间价也不是离岸人民币成交价。",
        "2020年11月5日、12月16日和2021年1月27日三次FOMC决定均维持联邦基金目标0—0.25%，后两次维持每月至少800亿美元国债、400亿美元机构住房抵押贷款支持证券购买。最后窗口到3月16日结束，3月17日决定发生在退出以后。因而这里没有“窗口内已经加息”的证据。[12月16日声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20201216a.htm)、[1月27日声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20210127a.htm)。",
        "美国CPI的内容也不支持笼统的“全面超预期”。同一2月8日报告记录1月核心CPI环比调查0.2%，对应官方档案为0.0%；总体环比0.3%与调查相同，总体及核心同比均比调查低0.1个百分点，后两个差值处在展示精度边界附近。12月的总体、核心环比分别0.4%、0.1%，均等于同一1月11日报告调查值；总体同比仅高0.1个百分点。更准确的问题是市场如何重估未来增长、通胀和风险补偿，而这些未来预期尚未由当前CPI直接识别。[2021年1月13日CPI档案](https://www.bls.gov/news.release/archives/cpi_01132021.htm)、[2月10日CPI档案](https://www.bls.gov/news.release/archives/cpi_02102021.htm)。",
        "上述两个BLS档案分别注明1月19日、2月11日重发，更正表2、表6的特殊季调序列。本轮保守保留较晚版本可用时刻，不宣称四个摘录值已获不可变首版认证。2月10日美国CPI在北京时间21:30发布，A股已经收盘且随后休春节；原公告与重发版本的下一A股开盘均为2月18日。1月档案两种时钟对应1月14日与1月21日，未混用。全部5份BLS PDF直连403，保留失败记录；数字由本轮可读官方HTML档案摘录，没有伪造完整本地原件。",
        "3月10日晚公布的2月CPI总体同比1.7%、核心同比1.3%，分别比1月1.4%、1.4%一升一降。在它第一次能反映到A股的3月11日之前，原入场本金已经累计−12.37个百分点，此后到原退出日贡献＋1.28个百分点，合计−11.09%。这个先后顺序排除了把此前全部下跌算到这次已公布数字上的说法；它仍不能测出市场此前怎样预期该数据。[3月10日CPI档案](https://www.bls.gov/news.release/archives/cpi_03102021.htm)。",
        f"![三个连续月份的完整路径](<{figure.as_posix()}>)",
        "回到510300内部，下面列出的都是原观察时点指数权重乘以相应股票的收盘至收盘固定股数收益，分红留现金。成分资料只有合格收盘路径，因此另外保留同口径ETF和19日入场收盘比较，不直接把这些贡献等同ETF原开盘收益。",
        "| 行业已覆盖贡献，百分点 | 2020-11病例 | 2020-12病例 | 2021-01病例 |",
        "|---|---:|---:|---:|",
    ]
    for ind in ["食品饮料","医药生物","电子","电力设备","家用电器","银行","非银金融"]:
        triplet(ind,[f"{s[(s.stat_month==c)&(s.industry_reference==ind)].iloc[0].known_contribution_pp:+.3f}" for c in CASES])
    lines += ["", "第三段食品饮料、医药合计贡献约−4.824个百分点，银行约＋0.026；行业参考中18类为负、9类为正，另有少量行业名称缺失。按各起点权重选择的前五恰好是同一组公司，仍逐期使用各自原权重：2021年1月病例贵州茅台−18.15%、五粮液−20.36%、美的集团−18.96%、招商银行−1.70%，中国平安＋9.31%。把所有成分写成一个“牛市／熊市”标签，会丢掉这些具体差异。",
        "可计算成分的上涨家数由第一段164只、第二段119只、第三段114只变化，对应上涨股票的原始权重分别65.31%、48.90%、27.16%。第三段114／299约38%的股票仍上涨，但上涨权重只有27%左右。ETF的20日总波动从观察起点21.15%升至退出30.97%，下行尺度从14.44%升至27.32%；变化与高权重板块的下跌相伴。两类波动是已实现路径的度量，不能由这里反推出观察日就确定会跌。",
        "覆盖边界需要和数字一起读。三期完整股票路径分别297／300、300／300、299／300，覆盖原权重99.431%、100%、99.927%；缺失的4个股票病例继续留空，不补零、不把它们的权重分给其他股票。全部可计算部分按原权重贡献之和为＋11.325、＋1.507、−10.153个百分点；最后一项未覆盖0.073%权重，也存在原权重静态篮子与真实ETF持仓、再平衡及跟踪差异，不能称ETF完整归因。缺失股票都不在三期前五名中。",
        "**可以明确收敛的认识是：货币统计改善、超预期、融资价格和股票现金流定价必须逐环对应。** 2021年初这个历史段中，剪刀差的大幅改善包含基数跃升，信贷和订单仍有支持；后续长端利率变化与高权重行业回撤同在，短端资金价格、银行及部分保险股又走出不同方向。这比“宏观好所以应该涨”更接近实际传导，也比给波动贴统一多空标签更有解释力。",
        "仍未识别的环节也具体了：美国名义长债的变化中，真实利率、预期通胀、期限溢价各占多少；境内公司预期现金流与要求回报率如何同步调整；这组变化能否在原起点之前被识别并在其他独立阶段留下可成交优势。三个已知结果的病例不支持因果系数、胜率门槛或确定的交易方向，本轮没有建立新策略。",
        f"已经从保存原字段复算896个完整股票病例、17,920个成分日、69个利率时点以及原E0/E1六个ETF窗口，最大数值误差{verification['maximum_numeric_error']:.3g}；3条公司行动已进入固定股份和现金账本。核对属于计算和时钟一致性，不是独立预测验证。",
        "",
        "- "+link("results/三个连续月份_经营预期利率与内部传导.csv","三个月完整连接表")+"；"+link("results/原60个交易日_ETF波动与利率时序.csv","全部60个交易日")+"。",
        "- "+link("results/900个原成员_收益贡献与缺失.csv","900个原成员及全部缺失")+"；"+link("results/原权重前五_全部15个公司病例.csv","原权重前五15个病例")+"；"+link("results/所有行业_原权重已覆盖贡献.csv","全部行业贡献")+"。",
        "- "+link("results/八次官方公告_事件时钟与档案版本时钟.csv","官方公告双时钟")+"；"+link("results/窗口内全部公告_同一原本金的前后贡献.csv","消息前后原本金贡献")+"；"+link("verification.json","计算核对记录")+"；"+link("protocol.json","原固定范围")+"。",
        "",
        "下一步围绕尚未识别的长端利率来源继续做有界历史核对：先拆2021年这一段的名义与实际利率、通胀补偿及其公布时序，再检查这些变化与银行、消费和医药的现金流／折现暴露是否吻合；不把本轮行业涨跌反过来用作当时的选择条件。"
    ]
    report_path.write_text("\n\n".join(lines).replace("|\n\n|","|\n|"),encoding="utf-8")
    shutil.copy2(Path(__file__),OUT / "code" / Path(__file__).name)
    save("publication_receipt.json",{"at":now(),"report":report_path.name,"report_sha256":digest(report_path),
         "figure":str(figure.relative_to(OUT)),"figure_sha256":digest(figure),"goal_achieved":False})
    print(json.dumps({"报告":str(report_path),"图":str(figure)},ensure_ascii=False))


if __name__=="__main__":
    main()
