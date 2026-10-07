"""形成经营传导报告与可导出的研究图，不把解释提升为预测结论。"""
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orders_cost_profit_bridge_v21 import OUT, now, digest, save

REPORT = "第二十一轮_订单原料库存与实际利润率.md"
FIGURE = "figures/订单扩张_价格扩散与利润兑现.png"
BASE = "C:/Users/戴周阳/Documents/New project 8/reports/research/510300_orders_cost_profit_bridge_v21"


def link(label, suffix):
    return f"[{label}](<{BASE}/{suffix}>)"


def value(v, places=2, sign=False):
    if pd.isna(v):
        return "未覆盖"
    return f"{v:+.{places}f}" if sign else f"{v:.{places}f}"


def figure(x, totals):
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei"],"axes.unicode_minus":False,
                         "font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#fafaf7","axes.facecolor":"#fafaf7"})
    fig=plt.figure(figsize=(14,10.4))
    gs=fig.add_gridspec(2,2,height_ratios=[1.13,1],hspace=.54,wspace=.24)
    ax=fig.add_subplot(gs[0,0]);bx=fig.add_subplot(gs[0,1]);cx=fig.add_subplot(gs[1,:])
    ix=np.arange(len(x))
    ax.plot(ix,x.pmi_0_input_price_diffusion,"o-",color="#b56530",label="主要原材料购进价格")
    ax.plot(ix,x.pmi_0_output_price_diffusion,"s-",color="#21646d",label="出厂价格")
    ax.axhline(50,color="#777777",lw=.8,ls="--")
    ax.set(xticks=ix,xticklabels=x.stat_month,ylabel="制造业扩散指数",ylim=(40,79))
    ax.tick_params(axis="x",rotation=55)
    ax.set_title("A  原11个联合支持月：购进指数均高于出厂",loc="left",pad=15,fontweight="bold")
    ax.legend(loc="upper left",frameon=False,fontsize=9)
    vals=x.industrial_0_margin_yoy_change
    colors=["#21646d" if pd.notna(v) and v>=0 else "#b56530" for v in vals]
    bx.bar(ix,vals,color=colors,width=.65)
    for i,v in enumerate(vals):
        if pd.isna(v):
            bx.text(i,-.17,"缺失",ha="center",va="top",fontsize=9,color="#777777",rotation=90)
        else:
            bx.text(i,v+(.08 if v>=0 else -.07),f"{v:+.2f}",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
    bx.axhline(0,color="#777777",lw=.8)
    bx.set(xticks=ix,xticklabels=x.stat_month,ylabel="已公布工业累计利润率较上年同期／百分点",ylim=(-.88,2.9))
    bx.set_xlim(-.65,len(x)-.35)
    bx.tick_params(axis="x",rotation=55)
    bx.set_title("B  同一起点的实际利润率：4升、4降、3缺失",loc="left",pad=15,fontweight="bold")
    bx.text(.02,.93,"工业统计期通常早于PMI月份\n未把二者当成同期、同企业数据",transform=bx.transAxes,va="top",fontsize=9,color="#555555")
    t=totals.loc["2021-02"]
    contributions=[t.cost_contribution_pp,t.fee_contribution_pp,t.other_net_change_pp]
    bottoms=[0,contributions[0],sum(contributions[:2])]
    cx.bar(range(3),contributions,bottom=bottoms,color=["#21646d","#4f9098","#b56530"],width=.6)
    cx.bar(3,t.margin_yoy_change,color="#273d52",width=.6)
    for i,v in enumerate(contributions):
        cx.text(i,bottoms[i]+v/2,f"+{v:.2f}",color="white",ha="center",va="center",fontsize=13,fontweight="bold")
    cx.text(3,t.margin_yoy_change+.10,f"+{t.margin_yoy_change:.2f}",ha="center",fontsize=13,fontweight="bold")
    for i in range(2):
        cx.plot([i+.3,i+.7],[bottoms[i]+contributions[i]]*2,color="#999999",ls="--",lw=.8)
    cx.set(xticks=range(4),xticklabels=["每百元收入成本下降","每百元收入费用下降","其余净项及舍入残差","利润率同比提高"],ylabel="相同收入分母的算术贡献／百分点",ylim=(0,3.8))
    cx.set_title("C  2021年1—2月：成本涨价普遍，与实际成本占比下降并存",loc="left",pad=16,fontweight="bold")
    cx.text(.015,.93,"公布：2021-03-27\n晚于原E0终点2021-03-16",transform=cx.transAxes,ha="left",va="top",fontsize=10,color="#555555")
    fig.suptitle("从订单到盈利，中间还有售价、库存结转和费用",x=.075,y=.975,ha="left",fontsize=19,fontweight="bold")
    fig.text(.075,.935,"全部原联合支持月份保留；A、B为起点背景，C为后来披露的经营结果，均非新交易规则。",fontsize=11,color="#555555")
    fig.text(.075,.035,"来源：原月度观察、当期国家统计局原表。PMI差不是利润率；同比按原文可比口径；工业企业不等于沪深300。",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.075,right=.97,top=.86,bottom=.095)
    fig.savefig(OUT/FIGURE,dpi=170,bbox_inches="tight")
    plt.close(fig)


def publish():
    if (OUT/"completion.json").exists():
        raise RuntimeError("本轮已完成，不覆盖。")
    verification=json.loads((OUT/"verification.json").read_text("utf-8"))
    assert verification["status"].startswith("PASS_")
    supplementary=json.loads((OUT/"sources/20210327_官方解读.receipt.json").read_text("utf-8"))
    assert supplementary["status"]=="SAVED_OFFICIAL_TEXT"
    x=pd.read_csv(OUT/"results/原11个月_完整联合支持背景.csv")
    totals=pd.read_csv(OUT/"results/65期_成本费用与利润率同比分解.csv").set_index("stat_month")
    assert len(x)==11 and x.pmi_0_output_minus_input_diffusion_pp.lt(0).all()
    assert x.industrial_0_margin_yoy_change.gt(0).sum()==4
    assert x.industrial_0_margin_yoy_change.lt(0).sum()==4
    assert x.industrial_0_margin_yoy_change.isna().sum()==3
    figure(x,totals)
    table=["| 原货币月 | 新订单 | 购进价格 | 出厂价格 | 出厂减购进 | 当时已知工业期 | 利润率同比变化 | 原E0收益 |",
           "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
    for r in x.itertuples(index=False):
        table.append(f"| {r.stat_month} | {r.orders_first_release_value:.1f} | {r.pmi_0_input_price_diffusion:.1f} | {r.pmi_0_output_price_diffusion:.1f} | {r.pmi_0_output_minus_input_diffusion_pp:+.1f} | {r.industrial_0_period if pd.notna(r.industrial_0_period) else '未覆盖'} | {value(r.industrial_0_margin_yoy_change,sign=True)} | {r.E0_return_percent:+.2f}% |")
    body=f"""**本轮推进到一个具体机制：当前采购涨价、库存成本结转、实际利润率和股票剩余收益，处在不同环节。2021年初，原料涨价较普遍与工业利润率改善同时发生，原510300观察窗口仍然下跌。因此，既不能把采购价格差直接当利润率，也不能把已实现的经营改善直接当尚未兑现的股票收益。**

本轮保留原104个月和原11个联合支持月，把当期制造业订单及购进、出厂价格接到实际工业收入、成本、费用与利润。没有新增高低位阈值、牛熊划分或预测拟合。原收益已被研究过，本轮属于机制解释；完整目标仍未达成。

**一、11个月的完整对照：类似的价格差，伴随不同的利润率状态。**

{chr(10).join(table)}

订单、购进与出厂三列均为当时已经公布的制造业扩散指数；出厂减购进是指数点差，利润率同比变化是百分点，不能混为同一种量。原E0仍是月度快照后下一开盘起20个交易日、固定股数含分红的毛收益，不含费用。

11个月的购进指数均高于出厂指数；其中有工业起点资料的8个月，利润率同比上升、下降各4个月，另外3个月保持缺失。2019年11月的49.0与47.3均低于50，2021年1月的67.1与57.2均高于50，不能把负的指数差全部称为原料涨价。这个并列**没有建立价格差与利润率之间的同期因果比较**：PMI是月度季调调查，工业是不同企业范围的年内累计财务指标，且起点可见工业统计期通常更早。它明确说明，两类指标不能互相替代或简单重复计票。

2020年7至10月的起点背景中，工业利润率同比差从−0.48逐步收窄至−0.06个百分点；到11月观察时变为+0.08。与此同时，购进与出厂扩散差一直为负。这是改善过程与采购环境并存的记录，不是按日历月消除季节性后的同企业试验。2019年三个月没有合格工业起点，未用未来报告补齐。

**二、2021年病例：价格压力并没有自动变成当期整体利润率下降。**

2021年2月9日原观察起点，最新1月新订单52.3、主要原材料购进价格67.1、出厂价格57.2。差值−9.9描述两种价格涨跌普遍程度的差异，绝不代表毛利率下降9.9个百分点，也不是原料上涨67.1%。[1月PMI原报告](https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202101/t20210131_1812932.html)同时显示订单扩张较上月减慢。

当时已经公布的2020年工业全年收入同比+0.8%、成本+0.6%，利润率6.08%、同比提高0.20个百分点。这是1月27日已公布的上一年累计结果，不是当时已经知道1至2月的经营表现。[2020年工业原报告](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900985.html)。

到了3月27日才公布的2021年1至2月结果，营业收入同比+45.5%、营业成本+43.5%，每百元收入成本82.92元、同比减少1.16元，费用8.79元、减少1.38元，利润率6.60%、提高3.15个百分点。按同一收入分母作算术拆解：

**利润率提高3.15 = 成本占比下降贡献1.16 + 费用占比下降贡献1.38 + 其余净项及舍入残差0.61。**

这不是三个可独立交易的因素，也不是因果贡献。利润总额还包含营业成本和四项费用以外的项目，0.61不被指认为补贴、投资收益或单一原因。同批官方表显示41个工业大类中38个利润同比增加；低基数明显影响同比，但官方同时公布相对2019年同期增长72.1%、两年平均增长31.2%，不能把全部改善说成只有2020年低基数。[1至2月工业原报告](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901035.html)。

**三、为什么会这样：补上库存及不同企业的位置。**

同日[国家统计局原因解读](https://www.stats.gov.cn/sj/sjjd/202302/t20230202_1896463.html)指出，部分原材料行业当期产品售价上升，而结转成本仍受此前低价库存影响；生产销售恢复、疫情比较基数和春节生产安排也共同影响经营。这里有官方原文支持的库存机制，不能简化为“买贵了，所以所有企业利润马上下降”。

由这段证据可以提出一个有条件的传导解释：对于先入库再投入生产、随销售结转成本的原料，当前补库价格与本期结转价格可能不同。若销售提价先发生、低价库存尚未消耗完，利润可以阶段性改善；之后库存替换、销量、合同提价和费用摊薄的变化，才决定这种改善能否延续。这是机制推断，不是本轮已经测出的库存周期，也没有取得全部公司的采购批次、成本流转及订单合同。

更不能把工业的总数直接套到510300。工业统计涵盖规模以上工业法人，既有非上市企业，也不包含银行、保险等全部指数行业；工业利润总额与归母净利润不同。第十九轮六家原权重公司中，两家金融企业尤其需要贷款定价、存款成本、信用损失和保险新业务的另一套传导。行业范围相近的公司也有不同产品与库存结构，不能由行业盈利同比直接推断某一家公司盈利意外。

**四、把经营事实放回货币、定价与波动的先后次序。**

| 日期 | 当时已经知道的事 | 与原观察窗口的关系 |
| --- | --- | --- |
| 2021-01-27 | 2020年工业全年利润率同比提高0.20个百分点 | 2月9日起点已知的较早经营背景 |
| 2021-01-31 | 1月订单仍扩张；购进价格67.1、出厂57.2 | 起点已知，采购环境与前期财务不属同一期 |
| 2021-02-09收盘后 | 1月剪刀差及信贷资料公开；此前60日ETF已涨15.95% | 原21点快照，消息前收盘不能叫公告反应 |
| 2021-02-28 | 2月订单51.5，购进66.7、出厂58.5 | 新经营调查在原持有窗口内公布 |
| 2021-03-16 | 原E0窗口结束，毛收益−11.09% | 原E1至3月17日为−13.96% |
| 2021-03-27 | 1至2月实际利润率提高3.15个百分点，官方说明库存因素 | 晚于两种原窗口终点，不能回填2月判断 |

起点的M1同比还含春节与基数影响，信贷数量又有借款主体和期限差异；前几轮已分别核对。现在增加了经营这一层：订单扩张、原料涨价与实际盈利恢复可以同时出现。随后股价下跌不能据此简单归因于“订单没有兑现”或“工业利润普遍恶化”，也不能反向得出利润增长时应该卖出。

第十九至二十轮还显示，后续实际利率变化、不同公司风险敞口及市场内部路径必须保留；事后相关较强不等于起点可判断。完整的未识别部分仍是：这些经营事实相对当时市场预期新增了多少，如何进入指数权重公司的未来盈利，以及股票折现与风险补偿同时怎样变化。没有这些证据，不能把任一条经营会计关系提升为确定性方向。

![订单价格与实际利润率](<{BASE}/{FIGURE}>)

**五、完成范围与剩余限制。**

本轮保留104个月，连接832条起点／后续经营位置；复用116份当期PMI记录和65份工业原报告，保存每份全部41行业及分组，共3,250条原表记录。工业起点71个月有原范围内资料，其余33个月保留无可用起点；65期利润率同比都完成成本、费用和剩余净项的同分母拆解。不存在把重复月度背景当作832次独立试验的问题。

65期原金额、报告内比例、全部信息时钟及原{verification['inherited_columns_unchanged']}列继承结果已核对。核对中的2018年、2022年网页表头布局差异只修正读取检查，没有改经营数字、样本或原收益。同比直接沿官方可比口径；不同年份公开总额有企业范围等差异，不跨版本相除重造同比。报告内行业加总仅允许显示舍入误差，负基期脚注保持原文和数值缺失。

3月27日原因解读是看过本轮经营数据后定向补充的一个来源，范围单独记录；未重定义样本，也没有回填起点。当前下载页面不是不可修订历史首版认证。旧制造业成本模型、账户与失败状态保留；新增模型0、账户0，尚未取得独立的未来方向证据。总目标继续进行中。

直接文件：

- {link('104个月完整连接表','results/104个月_订单价格成本与原后续路径.csv')}，{link('原11个月全部背景','results/原11个月_完整联合支持背景.csv')}。
- {link('65期成本费用与利润率分解','results/65期_成本费用与利润率同比分解.csv')}，{link('全部行业原金额','results/65期_全部原表行业及经营金额.csv')}。
- {link('832条信息角色及时间','results/832条_经营信息角色与可见性.csv')}，{link('2021病例完整先后关系','results/2021年1月病例_全部起点与后续经营信息.csv')}。
- {link('固定范围','protocol.json')}，{link('原金额与时钟核对','verification.json')}，{link('库存机制官方原文','sources/20210327_官方解读.html')}。
"""
    (OUT/REPORT).write_text(body,encoding="utf-8")
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    save("publication_draft.json",{"at":now(),"report":REPORT,"report_sha256":digest(OUT/REPORT),"figure":FIGURE,"figure_sha256":digest(OUT/FIGURE),"visual_review":"PENDING","goal_achieved":False})
    print("研究报告与图已生成，等待图中文字和布局检查。")


def complete():
    if (OUT/"completion.json").exists():
        raise RuntimeError("完成回执已存在。")
    d=json.loads((OUT/"publication_draft.json").read_text("utf-8"))
    assert digest(OUT/REPORT)==d["report_sha256"] and digest(OUT/FIGURE)==d["figure_sha256"]
    review=json.loads((OUT/"visual_review.json").read_text("utf-8"))
    assert review["status"]=="PASS_VISUAL_REVIEW" and review["figure_sha256"]==digest(OUT/FIGURE)
    save("completion.json",{"at":now(),"status":"COMPLETED_FIXED_OPERATING_TRANSMISSION_DIAGNOSTIC",
         "previous_turn_classification":"PROGRESS","current_turn_classification":"PROGRESS",
         "report":REPORT,"report_sha256":digest(OUT/REPORT),"figure":FIGURE,"figure_sha256":digest(OUT/FIGURE),
         "verification_sha256":digest(OUT/"verification.json"),"goal_status":"active","goal_achieved":False,
         "independent_validation":False,"new_models":0,"new_accounts":0,"orders_authorized":False,"global_mandate_modified":False,
         "new_finding":"原11个月同为购进扩散高于出厂，起点工业利润率却4升4降3缺失；2021初采购环境与库存结转、收入分母不同，实际利润率改善在原价格窗口后才披露。",
         "remaining":"经营现实相对事前预期的变化、各类成分的未来盈利与折现，以及可重复的进入前证据尚未贯通；不继续改阈值挽救旧模型。"})
    print("第二十一轮已完成并保存回执；总体研究目标保持进行中。")


if __name__=="__main__":
    {"publish":publish,"complete":complete}[sys.argv[1]]()
