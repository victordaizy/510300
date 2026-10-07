"""交付现金流的期间、收付与货币统计边界结论。"""
from pathlib import Path
import json
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.operating_cashflow_sources_v7 import OUT, now, save, sha


def location(path):
    return "C:/Users/戴周阳/Documents/New project 8/" + Path(path).relative_to(ROOT).as_posix()


def link(path,label):
    return f"[{label}](<{location(path)}>)"


def figure():
    x=pd.read_csv(OUT / "results/三家公司_原公告现金流改善来源.csv")
    totals=pd.read_csv(OUT / "results/三家公司_原公告半年经营现金流.csv").set_index("company_name")
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False,"font.size":11})
    fig,axes=plt.subplots(1,3,figsize=(15.6,7.5),sharey=True)
    fig.patch.set_facecolor("#f7f8fa")
    fig.subplots_adjust(left=.07,right=.97,top=.76,bottom=.25,wspace=.21)
    fig.text(.07,.94,"经营净现金流都在改善，实际来源不同",fontsize=22,weight="bold",color="#183b50")
    fig.text(.07,.884,"2025年上半年公司原公告｜下图为同比增减的金额贡献，单位：亿元；不是现金余额，也不是股价贡献",fontsize=11.5,color="#5a6b78")
    labels={"中国建筑":["经营收款\n增加","经营付款\n增加","净现金流\n改善"],"保利发展":["经营收款\n略增","经营付款\n减少","净现金流\n改善"],"中国石油":["缴所得税前\n经营现金减少","支付所得税\n减少","净现金流\n改善"]}
    for ax,name in zip(axes,["中国建筑","保利发展","中国石油"]):
        parts=x[x.company.eq(name)].contribution_cny.to_numpy()/1e8
        value=totals.loc[name]
        ax.set_facecolor("white")
        ax.set_title(name+"\n"+f"经营净额：{value.prior_net_operating_cny/1e8:+.2f} → {value.current_net_operating_cny/1e8:+.2f}",fontsize=12,pad=18,color="#203e50")
        running=0.
        for i,amount in enumerate(parts):
            after=running+amount
            ax.bar(i,abs(amount),bottom=min(running,after),width=.58,color="#178879" if amount>=0 else "#c36a4a",zorder=3)
            if abs(amount)<1:ax.plot([i-.29,i+.29],[after,after],color="#178879",linewidth=3,zorder=4)
            ax.text(i,max(running,after)+13,f"{amount:+.2f}",ha="center",va="bottom",fontsize=12,color="#203e50")
            ax.plot([i+.29,i+1-.29],[after,after],color="#b1bcc4",linestyle="--",linewidth=1)
            running=after
        ax.bar(2,running,width=.58,color="#285879",zorder=3)
        ax.text(2,running+13,f"{running:+.2f}",ha="center",va="bottom",fontsize=12,weight="bold",color="#203e50")
        ax.axhline(0,color="#8293a0",linewidth=.9)
        ax.grid(axis="y",color="#e1e6eb",linewidth=.8)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#c0ccd5")
        ax.set_xticks([0,1,2],labels[name],fontsize=10.8)
        ax.set_ylim(-165,460)
        ax.set_xlim(-.6,2.6)
    axes[0].set_ylabel("对经营净现金流同比变化的贡献（亿元）")
    fig.text(.07,.137,"中国建筑仍为经营净流出；保利改善主要来自少付款；中国石油改善包含少付所得税的影响。",fontsize=11.3,color="#203e50")
    fig.text(.07,.089,"数据来自三家公司2025年半年报同表两期。中国石油使用报告内国际准则现金流表，未混加不同准则的利润调整项。",fontsize=9.8,color="#617583")
    fig.text(.07,.05,"不能将这三种过程统一称为新订单扩张，也不能把公司经营净现金流直接当作国内M1增量。",fontsize=10.5,color="#617583")
    dest=OUT / "figures/三家公司_同样现金流改善的不同来源.png"
    fig.savefig(dest,dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)
    return dest


def run():
    check=json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    assert check["status"]=="PASS_FIXED_COHORT_PERIOD_AND_CASHFLOW_RECOMPUTATION"
    agg=pd.read_csv(OUT / "results/247家公司汇总_现金流期间及收付原金额.csv").set_index("field")
    concentration=json.loads((OUT / "results/concentration_and_period.json").read_text(encoding="utf-8"))
    primary=json.loads((OUT / "results/primary_review.json").read_text(encoding="utf-8"))
    net=agg.loc["net_operating"]
    primary_fig=figure()
    bridge=[]
    for field in ["group_net_profit","operating_receivables_reduction","operating_payables_increase","inventory_noncash_other_adjustments","net_operating"]:
        r=agg.loc[field]
        bridge.append(f"| {r.label} | {r.latest_half_prior_cny/1e8:,.2f} | {r.latest_half_current_cny/1e8:,.2f} | {r.latest_half_change_cny/1e8:+,.2f} |")
    sector=pd.read_csv(OUT / "results/全部非金融行业_现金流增量与期间.csv")
    sector_lines=[]
    for r in sector.head(6).itertuples():
        sector_lines.append(f"| {r.industry_name} | {r.companies} | {r.net_operating_ttm_change_cny/1e8:+,.2f} | {r.net_operating_latest_half_change_cny/1e8:+,.2f} | {r.net_operating_earlier_half_change_cny/1e8:+,.2f} |")
    report=f"""**第七轮：现金流改善来自什么，以及它如何影响M1/M2的解释**

本轮沿用2025年8月固定病例，即2025年9月12日观察时的247家现金流配对非金融公司，取得五个既定报告期共1,235条完整现金流明细，再用公司原公告核对预定的两家最大正贡献公司和一家最大负贡献公司。上一轮已完成预期与信息时序连接，本轮属于继续取得可区分解释的证据，未运行新收益模型或账户。

现在能更明确地说：**经营现金流变好，可能是收款增加、付款减少、税费结算变化，或这些因素的组合；它并不必然伴随订单与利润增长。** 对2025年夏季的M1/M2解释，应保留经营现金流改善这一正面事实，同时继续区分改善发生的期间、付款的对手方、资金最终用途，以及市场此前已知什么。不能把它自动接成“剪刀差改善→企业扩产→利润上行→指数继续上涨”。

**先分清最新半年与更早半年的变化。**

固定247家公司TTM经营净现金流同比增加{net.ttm_change_cny/1e8:,.2f}亿元、增长{net.ttm_yoy*100:.2f}%。其期间来源如下，全部沿用同一组公司，不按结果改变成员：

| 期间 | 本期经营净现金流／亿元 | 可比上期／亿元 | 同比增加／亿元 | 同比增速 | 占TTM增量 |
|---|---:|---:|---:|---:|---:|
| 最新2025年上半年对2024年上半年 | {net.latest_half_current_cny/1e8:,.2f} | {net.latest_half_prior_cny/1e8:,.2f} | {net.latest_half_change_cny/1e8:,.2f} | {net.latest_half_yoy*100:.2f}% | {concentration['latest_half_share_of_ttm_increment']*100:.2f}% |
| 更早2024年下半年对2023年下半年 | {net.earlier_half_current_cny/1e8:,.2f} | {net.earlier_half_prior_cny/1e8:,.2f} | {net.earlier_half_change_cny/1e8:,.2f} | {net.earlier_half_yoy*100:.2f}% | {concentration['earlier_half_share_of_ttm_increment']*100:.2f}% |

最新半年确实改善了14.12%，这个正面变化必须保留。68.87%来自更早半年说的是金额增量的期间占比，也受两半年不同现金流规模影响，不能由此声称最新半年没有改善，或认定更早部分全部不可持续。TTM有平滑作用，也会混入已经发生的经营和付款过程。

**最新半年的改善，不能简单叫作“利润增长带来更多现金”。**

下表从同一批现金流补充资料的当前汇编值构造主金额桥，单位亿元。净利润含少数股东口径，与前轮归母利润不是同一项。存货、非现金及其他调整保留为合并余额，没有将缺失的存货明细或税费返还填为已知零。

| 调整项目 | 2024年上半年 | 2025年上半年 | 对净现金流同比变化的贡献 |
|---|---:|---:|---:|
{chr(10).join(bridge)}

最需要避免的误读是：经营性应收项目这一项仍为−9,020.72亿元，意味着在该现金流调整口径下仍占用现金，只是较上年−10,878.82亿元少占1,858.10亿元。**少占现金，不等于应收余额已经全面下降，也不等于回收的都是政府旧欠款。** 经营性应收项目比单独的应收账款广，不能用资产负债表单一科目变化机械替代。同期应付项目提供的现金支持反而少了478.73亿元，不能统一描述成靠扩大应付占款改善现金流。

直接法提供另一种核对：最新半年经营现金流入增加3,397.15亿元，经营现金流出增加1,882.22亿元，净增加1,514.93亿元；其中销售和劳务收款增加2,564.57亿元，采购付款也增加829.83亿元。直接收付与间接利润调整是同一个现金流的两种表达，不能当作两份独立利好再叠加计分。

**同样是现金流改善，公司原文显示三种不同过程。**

这三家公司按完整固定同组公司TTM现金流增量的前两名与末一名选取，未按股票收益挑选。以下全部使用2025年半年报原公告的同表两期，和上面的当前汇编回顾数据分别保存。

中国建筑原公告显示，上半年经营现金流入同比增加370.71亿元、现金流出增加111.32亿元，净现金流因此改善259.39亿元。公司把主要原因归于销售及劳务收款增加；当期营业收入同比下降3.2%，经营活动仍净流出828.31亿元。收款改善与收入扩张、现金净流入是不同事实，尚不能从原文量化其中有多少来自财政清欠。[中国建筑2025年半年报](https://static.cninfo.com.cn/finalpage/2025-08-29/1224610866.PDF)，PDF第32、101、237页。

保利发展上半年经营净现金流从−171.48亿元变为+160.17亿元，改善331.65亿元。但经营收款合计只增加约0.20亿元，经营付款减少331.45亿元。公司说明工程款支付减少；现金表还显示采购付款减少181.98亿元、其他经营付款减少141.38亿元，后者主要对应与有关单位往来付款减少。这里不能把“与有关单位往来”自动解释为关联交易，也不能把总额全部叫作工程款。原公告同时显示收入同比下降16.08%。[保利发展2025年半年报](https://static.cninfo.com.cn/finalpage/2025-08-26/1224571945.PDF)，PDF第13、80、181、184页。

中国石油在当前版本的TTM排名中是最大负贡献公司，但最新半年原公告经营净现金流同比增加86.44亿元。原报告国际准则现金流表显示：缴所得税前经营现金减少117.48亿元，支付所得税减少203.92亿元，合起来净现金流增加86.44亿元。公司现金改善可以与税前经营现金下降并存，少付所得税也不能直接解释为税率下降。本例只用同一套原表解释，未混加两套会计准则的利润调整项。[中国石油2025年半年报](https://static.cninfo.com.cn/finalpage/2025-08-27/1224585332.PDF)，PDF第23、140页。

![同样现金流改善的不同来源](<{location(primary_fig)}>)

中国建筑的2024年年报还提供了较早期间的核对：全年经营净现金流157.74亿元、上年110.30亿元；全文说明加强现金流管理。这两个年度数不能直接当作下半年值，必须分别扣除同年上半年。[中国建筑2024年年报](https://static.cninfo.com.cn/finalpage/2025-04-16/1223102350.PDF)，PDF第30页。本轮不据笼统管理层解释推算具体清欠金额。

**现金流改善具有一定广度，同时存在明显的行业和公司金额差异。**

247家公司中，{concentration['companies_with_ttm_cashflow_increase']}家TTM经营净现金流增加、{concentration['companies_with_ttm_cashflow_decrease']}家减少；最新半年有{concentration['companies_with_latest_half_cashflow_increase']}家增加。前三家正贡献公司为中国建筑、保利发展、比亚迪，合计占全部正增量的{concentration['top_three_share_of_positive_pool']*100:.2f}%，占正负抵消后净增量的{concentration['top_three_share_of_net_increment']*100:.2f}%。两个比例的分母不同，不混用。

按行业列出TTM金额增量最大的六组，完整行业表另附。公司集团金额汇总不等于指数权重贡献，上市母子公司范围重叠、A/H分配等仍有解释边界。

| 行业 | 公司数 | TTM现金流同比增量／亿元 | 最新半年贡献／亿元 | 更早半年贡献／亿元 |
|---|---:|---:|---:|---:|
{chr(10).join(sector_lines)}

**从公司现金流到M1，还缺哪一段证据。**

首先是期间：上述最新财报截止2025年6月30日及更早，8月货币数据在9月12日公布。财报可以作为当时已知背景，不能用来证明8月新发生的货币变化已经造成此前半年的企业现金改善。

其次是口径和用途。2025年新M1包含流通中货币、单位及个人活期存款、非银行支付机构客户备付金；公司现金流报表则记录期间流量，现金及现金等价物还可能包含外币等不同范围。[人民银行关于狭义与广义货币的说明](https://www.pbc.gov.cn/rmyh/109339/2025080818580470423/index.html)。中国石油原报告披露现金及现金等价物中人民币约47.3%、美元约47.2%，已足以说明财报折合人民币的现金不是同口径国内M1。

保利原公告可以把资金用途再走一步：经营净流入160.17亿元，扣除投资净流出13.70亿元、筹资净流出105.29亿元及汇兑影响约0.08亿元，最终现金及现金等价物净增加41.10亿元。经营净现金流增加不代表全部留在期末活期存款，融资偿付和投资也会改变最终留存。

销售收款本身也不等于新销售。[财政部对现金流项目的说明](https://www.mof.gov.cn/bd/mhwzzxly/hjs/202212/t20221229_3861394.htm)明确区分当前销售收款、以前销售回款和预收款，并涉及增值税及退款。因而，中国建筑收款增加还需进一步区分这些来源，不能从该总额直接认定新订单加速。

作为限定条件下的账户示例：若付款方和收款方都使用计入M1的人民币活期存款，一方付100、另一方收100，只改变双方存款分布，本身不增加这两方合计的M1。实际宏观变化还需分别识别新贷款与归还、财政收支、不同存款类别转换等流量；本轮没有取得把247家公司逐笔现金交易连接到这些宏观账户的证据，不能声称已经识别剪刀差的全部资金来源。

**对波动和510300判断的具体影响。**

第六轮已经显示，在同一份M1/M2数据仍为最新信息时，新增政策和价格变化会使上行、下行波动项分别放大。本轮又说明，即使企业现金流同时改善，背后的经营过程也不同。后续判断应分别追问：收款改善是否缓解融资约束，减少付款是否对应采购与项目节奏变化，税费变化是否持续，新增订单与利润是否接力，以及这些信息是否已经反映到权重公司的价格。

这些是需要验证的传导命题，并非已经证明的5—20日收益方向。现金流改善可能影响融资约束或风险补偿，但不能仅凭这几项会计事实计算未来上涨概率，也不能给少付款的企业统一贴利好或利空标签。

**来源边界及本轮交付。**

1,235条新取得的净经营现金流与V5保存值完全一致，因而本轮确实解释了此前+12.87%的同组总数。不过它们仍是当前版本回顾。三家公司原公告的50个字段比较中，14个与当前汇编值存在超过1元差异。例如保利2025年半年净利润原文约65.68亿元，当前字段约72.06亿元，现金流净额却相同；中国石油经营净现金流原文2,270.63亿元，当前字段2,272.40亿元。这说明净额对得上，不代表每个分项都已经认证为历史首发值。本轮保留原公告与当前汇编两个版本，没有只替换有利记录，也没有将全体汇总升级为严格历史预测底表。

已复算固定成员、五期净额、247个公司期间恒等式、全体直接法与间接法金额桥，以及三家公司原公告现金流桥和保利期末现金恒等式；原表单位与关键页已目视核对。这些核对证明所述金额关系，未建立独立的未来收益验证。

- {link(OUT/'results/247家公司_经营现金流增量完整排名.csv','全部247家公司现金流期间与增量')}。
- {link(OUT/'results/现金流增量_直接法与间接法完整金额桥.csv','全体现金流收付与营运资金金额桥')}；{link(OUT/'results/全部非金融行业_现金流增量与期间.csv','完整行业表')}。
- {link(OUT/'results/三家公司_原公告现金流改善来源.csv','三家公司原公告的改善来源')}；{link(OUT/'results/三家公司_原公告与当前汇编字段对照.csv','原公告与当前版本差异')}。
- {link(OUT/'results/保利_经营现金到期末现金的用途恒等式.json','经营现金至期末现金用途')}；{link(OUT/'verification.json','必要计算核对记录')}。

总目标保持active。本轮明确推进了现金流来源、期间与货币统计的传导解释，尚未证明510300可重复的未来方向或成本后账户优势；旧模型的冻结失败和缺失状态继续保留。
"""
    target=OUT / "第七轮_现金流改善的来源与货币传导.md"
    target.write_text(report,encoding="utf-8")
    next_cards=[
        {"mechanism":"收款增加与占款缓和","known":"中国建筑收款增加但收入下降，整体应收现金占用仍为负。","discriminating_next_fact":"后续同口径披露区分当期销售、旧款、预收；观察订单执行、应收和合同资产占款是否继续改善。","unidentified":"本轮无法量化财政清欠的独立贡献或持续付款时间表。"},
        {"mechanism":"付款减少","known":"保利现金改善主要来自较少经营付款，包含采购及与有关单位往来。","discriminating_next_fact":"对照施工和采购数量、项目交付及支付安排，区分规模调整、付款时点与效率变化。","unidentified":"缺少逐笔对手方和合同时间表，不推断全部是拖欠或全部是效率提高。"},
        {"mechanism":"税费结算影响","known":"中国石油原表税前经营现金下降，较少所得税支付使净现金流增加。","discriminating_next_fact":"区分税负变化、汇算预缴及应付税款变化，判断后续支付能否重复。","unidentified":"较少支付本身未证明法定税率下降或主营扩张。"},
    ]
    save(OUT / "results/后续可区分的经营传导命题.json",{"at":now(),"cards":next_cards,"not_return_forecasts":True,"not_first_time_unseen_predictions":True})
    save(OUT / "goal_progress.json",{"at":now(),"goal_status":"active","turn_classification":"PROGRESS","completed_this_round":["固定247家公司五期明细取得并连接原现金流总量","最新半年与更早半年分解","经营收付和营运资金主桥及完整公司行业分布","原公告识别收款、少付款、税款支付三种不同过程","资金最终用途和M1统计边界明确"],"not_achieved":"尚未证明510300可重复的未来方向或成本后账户优势。","new_models":0,"new_accounts":0,"goal_achieved":False})
    shutil.copy2(__file__,OUT / "code" / Path(__file__).name)
    save(OUT / "completion_receipt.json",{"at":now(),"round_status":"RESEARCH_DELIVERABLE_COMPLETE_GOAL_STILL_ACTIVE","report":target.name,"report_sha256":sha(target),"figure":str(primary_fig.relative_to(OUT)),"figure_sha256":sha(primary_fig),"verification_sha256":sha(OUT / "verification.json"),"round_achieved":True,"goal_achieved":False})
    print("第七轮报告、原公告对照与现金流构成图已完成，总目标仍在进行。")
    print(location(target))


if __name__ == "__main__":
    run()
