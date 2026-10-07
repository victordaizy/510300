"""交付同一货币发布周期内的完整过程比较。"""
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from within_release_volatility_v22 import OUT,PHASES,RELIEF,digest,save,now

REPORT="第二十二轮_降波已实现反弹与剩余空间.md"
FIGURE="figures/同一货币周期_已实现与剩余收益.png"
BASE="C:/Users/戴周阳/Documents/New project 8/reports/research/510300_within_release_volatility_v22"


def link(label,path):
    return f"[{label}](<{BASE}/{path}>)"


def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |",*["| "+" | ".join(map(str,r))+" |" for r in rows]])


def draw(distributions,correlations,cases):
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei"],"axes.unicode_minus":False,"font.size":10,
       "figure.facecolor":"#fbfaf7","axes.facecolor":"#fbfaf7","axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(2,2,figsize=(14.8,10.6))
    x=np.arange(3)
    colors=["#237781","#b26835","#69717b"]
    a=axs[0,0]
    for k,(target,label) in enumerate([("observed_close_return","观察期间已经发生的收益"),("late20_return","较晚起点之后20日收益")]):
        frame=correlations[correlations.y.eq(target)].set_index("analysis_period").loc[PHASES]
        pos=x+(k-.5)*.30
        a.bar(pos,frame.weighted_spearman,.27,color=colors[k],label=label)
        for i,r in enumerate(frame.itertuples()):
            v=r.weighted_spearman
            if pd.notna(r.low):
                a.errorbar(pos[i],v,yerr=[[v-r.low],[r.high-v]],fmt="none",ecolor="#333333",capsize=4,lw=1)
            a.text(pos[i],v+(.045 if v>=0 else -.045),f"{v:+.3f}",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
    a.axhline(0,color="#777777",lw=.8)
    a.set(xticks=x,xticklabels=PHASES,ylabel="D20变化与收益的周期加权秩相关",ylim=(-.85,.52))
    a.legend(frameon=False,fontsize=9,loc="upper left")
    a.set_title("A  降波与已经上涨联系更强",loc="left",fontweight="bold",pad=12)
    a.text(.98,.03,"新口径18个成熟周期，区间未计算",transform=a.transAxes,ha="right",fontsize=9,color="#555555")
    b=axs[0,1]
    for k,(clock,col,label) in enumerate([("E0","observed_close_return_mean","两观察点之间已发生"),("E0","late20_return_mean","较晚E0随后20日"),("E1","late20_return_mean","较晚E1再延迟一天")]):
        z=distributions[(distributions.state==RELIEF)&(distributions.clock==clock)].set_index("analysis_period").loc[PHASES]
        pos=x+(k-1)*.23;vals=z[col]*100
        b.bar(pos,vals,.21,color=colors[k],label=label)
        for i,v in enumerate(vals):
            b.text(pos[i],v+(.07 if v>=0 else -.07),f"{v:+.2f}",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
    b.axhline(0,color="#777777",lw=.8)
    b.set(xticks=x,xticklabels=PHASES,ylabel="周期等权平均收益／%",ylim=(-1.5,3.25))
    b.legend(frameon=False,fontsize=9,loc="upper right")
    b.set_title("B  D下降且近期同步缓和的全部观察",loc="left",fontweight="bold",pad=12)
    b.text(.02,.04,"E0：52对/39周期；41对/27周期；22对/14周期",transform=b.transAxes,fontsize=9,color="#555555")
    for ax,pair,title in [(axs[1,0],"2024-09-20__2024-09-27","C  政策后反弹：等待段占主要部分"),
                          (axs[1,1],"2021-01-15__2021-01-22","D  经营支持期：延长尾段占主要部分")]:
        r=cases.loc[pair]
        vals=np.array([r.removed_wait_component,r.denominator_component,r.new_tail_contribution])*100
        base=np.r_[0,np.cumsum(vals)[:-1]]
        ax.bar(range(3),vals,bottom=base,width=.62,color=["#237781","#a1a1a1","#b26835"])
        ax.bar(3,vals.sum(),width=.62,color="#2d4052")
        for i,v in enumerate(vals):
            # 小分量在柱外标注，避免字被窄色块遮盖。
            ax.text(i,base[i]+v/2,f"{v:+.2f}",ha="center",va="center",fontsize=11,color="white" if abs(v)>1 else "#222222",
                    bbox={"facecolor":"#fbfaf7","edgecolor":"none","pad":1} if abs(v)<=1 else None)
        ax.text(3,vals.sum()-.7,f"{vals.sum():+.2f}",ha="center",va="top",fontsize=12,fontweight="bold")
        ax.axhline(0,color="#777777",lw=.8)
        ax.set(xticks=range(4),xticklabels=["减去等待段","入场分母变化","新增尾段","前后收益差"],ylim=(-28,4),ylabel="较晚20日减较早20日／百分点")
        ax.set_title(title,loc="left",fontweight="bold",pad=13)
        ax.text(.02,.03,f"较早20日 {r.early20_return*100:+.2f}% → 较晚20日 {r.late20_return*100:+.2f}%",transform=ax.transAxes,fontsize=10,color="#555555")
    fig.suptitle("同一份货币数据下，降波之后还剩什么？",x=.06,y=.98,ha="left",fontsize=20,fontweight="bold")
    fig.text(.06,.94,"336对相邻周全部保留；风险变化、经营更新与价格分段分别记录。算术分解不等于经济因果。",fontsize=11,color="#555555")
    fig.text(.06,.028,"A：旧口径区间为6周期区块描述区间。B：前段与后段长度不同。C/D：负值为收益差，非因果损失贡献。均为未扣费历史观察。",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.065,right=.97,top=.865,bottom=.10,wspace=.24,hspace=.47)
    fig.savefig(OUT/FIGURE,dpi=165,bbox_inches="tight")
    plt.close(fig)


def publish():
    if (OUT/"completion.json").exists():
        raise RuntimeError("已完成本轮，不覆盖。")
    verification=json.loads((OUT/"verification.json").read_text("utf-8"))
    assert verification["status"].startswith("PASS_")
    d=pd.read_csv(OUT/"results/全部时期和风险状态_周期等权分布.csv")
    c=pd.read_csv(OUT/"results/D变化_已发生价格与后续结果关联.csv")
    cases=pd.read_csv(OUT/"results/原四个病例_全部相邻周.csv").set_index("pair_id")
    draw(d,c,cases)
    co=[]
    for phase in PHASES:
        z=c[c.analysis_period.eq(phase)].set_index("y")
        row=[phase,f"{int(z.iloc[0]['pairs'])} / {int(z.iloc[0]['cycles'])}"]
        for key in ["observed_close_return","late20_return","common_return"]:
            r=z.loc[key];s=f"{r.weighted_spearman:+.3f}"
            s+=f" [{r.low:+.3f}, {r.high:+.3f}]" if pd.notna(r.low) else "（区间未计算）"
            row.append(s)
        co.append(row)
    sr=[]
    for phase in PHASES:
        z=d[(d.analysis_period==phase)&(d.state==RELIEF)].set_index("clock")
        a,b=z.loc["E0"],z.loc["E1"]
        sr.append([phase,f"{int(a.pairs)} / {int(a.cycles)}",f"{a.observed_close_return_mean*100:+.2f}%",f"{a.late20_return_mean*100:+.2f}%",
                   f"{a.late20_return_median*100:+.2f}%",f"{a.late20_return_positive*100:.2f}%",f"{b.late20_return_mean*100:+.2f}%",f"{a.late20_worst_min*100:+.2f}%"])
    cr=[]
    for r in cases.itertuples():
        cr.append([r.stat_month,f"{r.early_origin_id} → {r.late_origin_id}",r.state.replace("两点间", ""),
          f"{r.observed_close_return*100:+.2f}%",f"{r.early20_return*100:+.2f}%",f"{r.common_return*100:+.2f}%",
          f"{r.new_tail_contribution*100:+.2f}pp",f"{r.late20_return*100:+.2f}%"])
    body=f"""**本轮把同一份货币数据下的过程比较做完整后，结论更清楚：下行波动减轻与已经发生的反弹联系较强，随后20日收益却没有同样稳定的改善。较晚起点与较早起点的收益差，还需要分开等待期间的涨跌和向后延长终点新增的风险。**

这不是将宏观目标缩减成波动恒等式。它检查的是原研究尚未闭合的“看到改善以后，还剩多少空间”。完整方向优势仍未证明，本轮有实际进展，总目标保持进行中。

**一、同一货币公告是共同背景，其他信息仍在变。**

从原445周中保留全部观察，6周尚无可用货币公告；其余439周归于103个发布周期，形成336对同周期相邻周。每对按日期生成，未挑最低波动、最强反弹或每月最好的周。E0/E1共672条路径，其中664条成熟、8条原样待成熟。成熟E0为332对：旧前期156对/48周期、旧后期117对/36周期、新口径59对/18周期。

每对的M1/M2原公告与读数不变，贷款来源也未更换；但101对之间更新了订单资料，99对更新制造业价格，65对更新工业经营，30对更新社融，59对更新住房资料，企业家与银行家调查各25对更新。这些是已有来源记录的变化次数，不能相加当独立消息数。

因此，“同一份M1/M2”不能被写成“整个宏观环境完全不变”。两端各自的信贷、订单、成本、利润率、估值、广度与国内外利率完整保留；旧政策利率目录存在首次记录滞后，不能据目录值认定政策当日未变。已有24节点政策目录只在8对之间记有事件，零记录不表示没有其他消息。

**二、波动缓和更明显地伴随已经发生的价格恢复。**

下表是两周观察点之间D20变化与三种收益的周期加权秩相关。负相关表示D下降时，该段价格往往上涨；它不是胜率或因果份额。

{table(['货币统计期','成熟周对 / 周期','与两观察收盘间已发生收益','与后点新20日收益','与后点至旧终点剩余收益'],co)}

旧口径两段，降波与已发生收益的相关为−0.469、−0.431；与随后20日收益仅+0.023、+0.119，两个后续收益区间均跨零。新口径前后者分别−0.645、+0.111，只有18个成熟周期，未达到事先固定24周期的区间计算门槛。

这里不能把负相关全解释为独立经济信号：D20和这段价格变化使用了部分相同的日收益。它支持的判断是，看到降波时应先核对已经完成的价格路径；不能再把它与已经上涨当成两份独立看涨证据。这个结果不否定第三轮发现的风险**水平**惯性，也没有证明风险的短期**变化**能稳定预测未来方向。

**三、真正的近期下跌缓和，也不等于剩余收益确定为正。**

336对中178对D20下降，其中116对最近5日下行平方和也不高于紧邻前5日，62对反而增强。配对跨度受假期影响，实际相隔1至多个交易日；最近5日比较仍严格沿原5日定义。全部状态均已披露，下表单列同时缓和者。

{table(['货币统计期','成熟周对 / 周期','两观察间已发生均值','后点E0新20日均值','后点E0中位数','周期等权正收益比例','后点E1均值','最差一次途中收盘'],sr)}

均值、中位数和比例均先确保每个货币发布周期权重相等，周期内有效周对等权；最差值为实际最差记录。它们不是未来上涨概率。前段通常约一周，后段20日，表格只说明价格先后关系，不声称两段收益率可以作相同期限的收益能力比较。“途中收盘”是相对入场本金的最差收盘，不是日内最低或峰值最大回撤。

即使在原来的“剪刀差改善、企业和住户中长期信贷均同比多增、订单不低于50”背景中，仍有12对同时下行缓和的观察，来自2018—2021段的9个周期。其后点E0平均−0.97%、中位数−0.33%，E1平均−2.22%；E0最差途中收盘−12.48%。这是固定旧条件的完整小样本结果，不是新的独立验证，也不能据此反向做空或继续加条件删除失败对。其他全部信用订单背景同时保留。

按日期固定抽取不重叠周对后，旧口径64对、新口径14对。同步缓和组各时期只剩10、11、6对，后点E0均值依次+0.95%、−2.77%、−1.68%；稀少样本不能估算稳定优势。它们仍来自同一段已见历史。

**四、两种不同的“后来收益更差”，应当拆开。**

以下两对从原四个固定周期中选作解释示例；选择用于展示算术来源，发生在看到本轮结果之后。四个周期全部14对列在后表，未据示例建立规则。

2024年9月20日至27日，最新货币统计月始终是8月；企业、住户中长期累计同比分别少增19,256、2,430亿元，新订单48.9未更新。期间已有降准降息、资金支持比例及互换便利的政策节点。两观察收盘之间价格上涨18.17%，成分上涨覆盖从27.70%扩至81.76%，D20从9.33%降至8.86%，总RV却从11.14%升至32.97%。这是政策信息、价格和内部参与已经改变的过程，不能把D下降全部归因于旧信贷转好。[已保存节点与时钟](<{BASE}/results/周对之间_已有政策目录节点.csv>)。

较早E0从9月23日开盘到10月25日收盘，收益+23.59%；较晚E0从9月30日开盘到11月1日收盘，收益−0.70%。若较晚起点仍观察到较早的10月25日终点，剩余收益是+0.82%；新延长尾段再贡献−1.52个百分点，两者合为−0.70%。两个20日收益相差−24.28个百分点，其中减去等待段贡献−22.58、入场分母变化−0.19、新尾段−1.52。等待段包含较晚实际开盘跳空，不能将22.58全说成9月27日收盘时已知收益，也不能将其全部归因于单项政策。

2021年1月15日至22日则不同。最新12月货币背景中，企业和住户中长期累计同比多增29,200、5,000亿元，订单53.6未更新；已公布工业利润率同比高0.14个百分点。两观察收盘间上涨2.01%，D20从9.91%降至9.42%，广度69.70%升至76.17%；按保守美国时钟，已知10年实际收益率下降4基点。多层背景并没有形成之后确定上涨的保证。

较早E0从1月18日至2月19日，收益+6.33%；较晚E0从1月25日至2月26日，收益−4.22%。较晚进入但仍在原2月19日结束，剩余收益为+4.17%；新加入尾段贡献−8.39个百分点，合计−4.22%。这里收益差的大部分来自延长终点，而非观察期间已经上涨太多。若实际计划在较晚起点再持有20日，这段亏损必须承担，不能拿缩短到旧终点的正收益替代原20日结果。

上述两例只是对价格路径和当时信息的复原。没有建立利率、政策、经营预期各自的因果反事实，也没有以看到尾段亏损为理由另选持有期限。

**五、原四个周期的全部相邻周。**

{table(['货币月','较早 → 较晚观察','波动过程','已发生收益','较早20日','较晚至旧终点','延长尾段贡献','较晚新20日'],cr)}

较晚至旧终点的持有长度较短；与延长尾段同以较晚入场本金为分母，二者相加等于较晚新20日。较早20日的本金不同，不能把这些列直接混加。E1、分红权益、各周经营和利率背景另在完整表中保留。2024年9月30日新订单已经从48.9更新至49.9，不继续声称所有周都有相同需求信息。

![同一货币周期的已实现和剩余路径](<{BASE}/{FIGURE}>)

**六、已经排除的跳步与仍缺的证据。**

本轮使“等到下行缓和以后再确认”这句话具体化了：首先核对D下降是否只是旧损失退出；其次核对价格和参与范围已经变化多少；再核对期间新的经营与政策信息；最后明确要承担的是原终点剩余路径还是新的完整20日风险。任何一个环节都不能由其他指标代替。

按这条顺序，现有资料仍未形成可重复的进入前收益优势。波动缓和较多地记录了正在发生或已经发生的恢复，但未来经营预期、折现和新消息怎样改变下一段价格仍未闭合。不能因需要明确结论，就将这些混合过程压成一个买卖分数。

全部384个原列保留；664条成熟路径以独立逐日现金递推核对，最大误差约5.3×10⁻¹⁶；890个美国利率快照时钟、原周对成员、加权结果与固定区块区间均一致。这些核对证明测量正确，不是独立预测或账户验证。所有收益为固定股数含应得分红毛观察，未扣费用；当前资料版本与历史首次送达限制保持原样。新增模型0、账户0、订单0。

直接文件：

- {link('336对完整价格与背景','results/全部周对_E0完整连接.csv')}；{link('672条E0/E1金额桥','results/672条_原终点剩余与延长尾段金额桥.csv')}。
- {link('全部风险状态与三个时期','results/全部时期和风险状态_周期等权分布.csv')}；{link('全部原信用订单背景','results/全部既有信用订单背景_不挑选最佳组.csv')}。
- {link('已发生与后续关联及区间','results/D变化_已发生价格与后续结果关联.csv')}；{link('不重叠周对的全部成员','results/按日期固定的不重叠周对_全部成员.csv')}。
- {link('完整445周背景','results/445周_原多层背景与利率利润补充.csv')}；{link('冻结范围','protocol.json')}；{link('复算结果','verification.json')}。
"""
    (OUT/REPORT).write_text(body,encoding="utf-8")
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    save("publication_draft.json",{"at":now(),"report":REPORT,"report_sha256":digest(OUT/REPORT),"figure":FIGURE,"figure_sha256":digest(OUT/FIGURE),"visual_review":"PENDING"})
    print("第二十二轮报告与图已生成，待视觉检查。")


def complete():
    if (OUT/"completion.json").exists():
        raise RuntimeError("完成回执已经存在。")
    draft=json.loads((OUT/"publication_draft.json").read_text("utf-8"))
    visual=json.loads((OUT/"visual_review.json").read_text("utf-8"))
    assert visual["status"]=="PASS_VISUAL_REVIEW" and visual["figure_sha256"]==digest(OUT/FIGURE)
    assert draft["report_sha256"]==digest(OUT/REPORT) and draft["figure_sha256"]==digest(OUT/FIGURE)
    save("completion.json",{"at":now(),"status":"COMPLETED_FIXED_WITHIN_RELEASE_PROCESS_COMPARISON",
      "previous_turn_classification":"PROGRESS","current_turn_classification":"PROGRESS","goal_status":"active","goal_achieved":False,
      "report":REPORT,"report_sha256":digest(OUT/REPORT),"figure":FIGURE,"figure_sha256":digest(OUT/FIGURE),
      "verification_sha256":digest(OUT/"verification.json"),"independent_validation":False,"new_models":0,"new_accounts":0,"orders_authorized":False,
      "global_mandate_modified":False,
      "new_evidence":"同一货币公告的336对相邻周中，D变化与已发生收益关联更强，后续关联弱且旧时期区间跨零；区分等待期间、旧终点剩余及延长尾段。原共同支持且近期缓和12对/9周期后20日仍未提供稳定正收益。",
      "remaining":"未形成已知来源变化、尚未定价的公司盈利或风险补偿以及独立可重复方向优势的完整证据；不把新的观察当旧失败模型重启依据。"})
    print("第二十二轮已完成；完整研究目标继续保持进行中。")


if __name__=="__main__":
    {"publish":publish,"complete":complete}[sys.argv[1]]()
