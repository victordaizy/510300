"""输出T14结果与累计候选进度；历史失败保留，未运行项不填绩效。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from research.factor96_funding_relief_v1 import OUT,save,now,digest

FIRST=ROOT/"reports/research/510300_factor96_mechanism_batch_v1"
PROGRAM=ROOT/"reports/research/510300_factor96_program_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def progress(result):
    PROGRAM.mkdir(parents=True,exist_ok=True)
    strategies=read(FIRST/"strategy_progress.json")
    for row in strategies:
        if row["id"]=="T14":
            row["current_status"]="COMPLETE_FIXED_DEFINITION_TARGET_FAILED"
            row["current_evidence"]="K05时钟核对后固定检验完成56账户。主期20万元压力夏普0.039050，2万元0.024840；各1个完整周期。资金退出无新增贡献。"
            row["result_path"]="reports/research/510300_factor96_funding_relief_v1/result.json"
    factors=read(FIRST/"factor_progress.json")
    for row in factors:
        if row["id"]=="K05":
            row["current_status"]="PARTIAL_COMPONENT_TESTED"
            row["current_note"]="DR007-当时已公开且生效的7天政策利率，滞后至少一A股交易日，5日均值/变化已实现；在T14中检验未达标。季末等已知日历仅后续描述，不新增筛选或声称K05单独有效。"
        if row["id"]=="B04":
            row["current_note"]+=" T14另按原卡片实现冲击后10日内首次收涨；rv3比值仅记录，无事后新增波动门。"
    for folder in [PROGRAM,OUT/"program_snapshot"]:
        folder.mkdir(parents=True,exist_ok=True)
        save(folder/"strategy_progress.json",strategies)
        save(folder/"factor_progress.json",factors)
        pd.DataFrame(strategies).to_csv(folder/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(factors).to_csv(folder/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    state={"at":now(),"goal":"只操作510300实现完整账户成本后夏普1.2",
        "goal_status":"active","goal_achieved":False,"registered_factors":96,"registered_strategies":18,
        "completed_fixed_candidates":["T03","T05","T14"],"not_run_candidates":15,"qualified_candidates":[],
        "completed_account_scenarios_this_round":56,"cumulative_account_scenarios":120,
        "latest_round":result["study_id"],"latest_result":"reports/research/510300_factor96_funding_relief_v1/result.json",
        "primary_candidate_status":"FAILED_SHARPE_AND_CAGR_TARGET_IN_BOTH_CAPITALS",
        "independent_forward_observations":0,"external_review":"NOT_PERFORMED","orders_authorized":False,
        "next_candidates":["T02","T04"],"next_stop_rules":["已失败T03/T05/T14不调参救援。",
            "T02缺少同一历史成员的20日新低与前次测试记录则保留NOT_RUN，不拿现今成分回填。",
            "T04不能确认提前固定领先组和当时行业归属则保留NOT_RUN。",
            "任何历史高夏普仍需选择偏差评估和独立观察，不能自动升级为交易。"]}
    save(PROGRAM/"status.json",state)
    save(OUT/"round_status.json",state)
    save(OUT/"program_snapshot/status.json",state)
    return state


def plot():
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig,axes=plt.subplots(2,1,figsize=(12.5,8.6),sharex=True,gridspec_kw={"height_ratios":[1.3,1]})
    styles={"PRICE_ONLY":("只用价格确认","#7b8495","-"),
        "ENTRY_ONLY":("利率只筛选买入（与主方案重合）","#b56c26","--"),
        "EXIT_ONLY":("利率只决定卖出","#ac4260","-"),
        "FULL":("主方案：进入与退出都使用利率","#087f8c","-")}
    for name,(label,color,style) in styles.items():
        z=pd.read_parquet(OUT/f"accounts/MAIN/200000/STRESS/LAG1/{name}/ledger.parquet")
        dates=pd.concat([pd.Series([z.date.iloc[0]-pd.Timedelta(days=1)]),z.date],ignore_index=True)
        nav=np.r_[1.,z.equity/200000]
        axes[0].plot(dates,nav,label=label,color=color,ls=style,lw=1.5)
        axes[1].plot(dates,(nav/np.maximum.accumulate(nav)-1)*100,color=color,ls=style,lw=1.5)
    axes[0].set_title("T14 资金利率紧张后的修复：完整账户检验",fontproperties=font,fontsize=17,loc="left",pad=60)
    axes[0].legend(prop=font,loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,frameon=False)
    axes[0].set_ylabel("账户净值",fontproperties=font)
    axes[1].set_ylabel("自账户峰值回撤（%）",fontproperties=font)
    axes[1].set_xlabel("2021—2025年；现金天数全部计入",fontproperties=font)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.text(.08,.025,"20万元 · 每边万4，最低5元，10bp滑点并按0.001价位取整 · 已观察历史，不是独立前向验证",fontproperties=font,fontsize=10,color="#555555")
    fig.tight_layout(rect=[.02,.055,.99,.99])
    fig.savefig(OUT/"主期完整账户净值与回撤.png",dpi=160)
    plt.close(fig)


def main():
    result=read(OUT/"result.json")
    verification=read(OUT/"saved_verification_receipt.json")
    assert verification["ledgers"]==56
    state=progress(result)
    m=pd.read_csv(OUT/"metrics.csv")
    primary=m[(m.period=="MAIN")&(m.cost=="STRESS")&(m.lag==1)&(m.policy=="FULL")].set_index("capital")
    names={"PRICE_ONLY":"价格确认对照","ENTRY_ONLY":"利率只筛选进入","EXIT_ONLY":"利率只决定退出","FULL":"主方案：进入与退出均使用利率"}
    table=[]
    for row in m[(m.period=="MAIN")&(m.capital==200000)&(m.cost=="STRESS")&(m.lag==1)].itertuples():
        table.append(f"|{names[row.policy]}|{row.net_sharpe:.4f}|{row.cagr:.4%}|{row.max_drawdown:.3%}|{row.closed_cycles}|")
    incr=result["increment"]
    report=f"""# T14资金利率紧张后的修复：固定检验结果

**目标尚未实现。** 本轮新增检验T14，完成56个完整账户情景。2021—2025年，20万元主账户在压力成本下净夏普为{primary.loc[200000,'net_sharpe']:.6f}、年化收益{primary.loc[200000,'cagr']:.4%}、最大回撤{primary.loc[200000,'max_drawdown']:.3%}；2万元对照净夏普为{primary.loc[20000,'net_sharpe']:.6f}。两账户均未达到夏普1.2与年化10%的既定共同目标。主方案只有1个完整交易周期，早期2017—2020年没有交易。不能据此认定利率因子稳定有效，保留本次失败，不调整阈值或选择更好的对照版本。候选库累计完成3/18项，合格项仍为0，总目标继续。

## 固定合同与经济问题

只持有510300.SH与人民币现金，禁止借贷或做空。DR007相对当时已公开的7天政策利率升高，可能反映短期资金压力；检验压力回落与股票急跌后首次收涨重合时，下一开盘的收益是否更好。反例是季节性利率正常化与股价无额外关系。

K05取最近5个A股决策日可见的DR007减政策利率均值。资金缓和要求前5个决策日曾高于当时90分位、当前低于当时50分位；分位来自此前252个决策日、至少120个有效值，排除当前。股票冲击是对数总回报低于此前20日波动的负2倍，冲击起点间隔至少10日，其后10日内第一次收涨确认。两条件同日成立，下一开盘申请买入。收盘失守冲击日低点、利率重新高于90分位、资金数据失效、公共风险退出或满5个开盘间隔时，下一合法开盘卖出。rv3变化仅记录，不新增波动筛选。

进入与退出两开关形成4种事前固定情景；FULL是唯一主候选。仅进入、仅退出、纯价格3项是作用分解对照。主结果失败后不把较好的对照提升为新方案。

## 来源与可得时间

DR007使用本地已保存的12份原始API响应，2,905条记录逐值复核；2021—2025年股票决策日资金特征全部可得。银行周末工作日记录保留，不用股票日历直接删除。DR007统计日至少后一个A股交易日09:30可用，16:00做决策，再下一开盘执行；超过10个自然日未更新视为缺失，不插值。另做整体多滞后1交易日的敏感性检验。

政策利率使用25条公开操作利率变化，补入1条已验证的实施事实，共26条。2024年9月24日宣布、9月27日生效和旧操作表9月29日首次出现1.5%是不同时间；9月27日事实按该日23:59:59上界可得，不能把未来目标提前视为已实施。[国务院客户端原文](https://app.www.gov.cn/govdata/gov/202409/27/519932/article.html)

银行和政策数据均为后来下载的历史资料，没有每条历史首次发布版本的认证。保守滞后处理了时间顺序，不能消除源数据修订或发布延迟的不确定性。本轮是历史开发检验，不称为独立样本外。

## 完整账户结果

主期2021—2025年、20万元、主披露时钟、压力成本。夏普按全部1,212个交易日的账户净收益、242日年化、零无风险收益计算；现金天不删除。零波动账户的夏普未定义。

|固定情景|净夏普|净年化收益|最大回撤|完整周期|
|---|---:|---:|---:|---:|
{chr(10).join(table)}

主方案20万元期末权益{primary.loc[200000,'end_equity']:,.4f}元，累计净利润{primary.loc[200000,'net_profit']:,.4f}元；2万元期末权益{primary.loc[20000,'end_equity']:,.2f}元，累计净利润{primary.loc[20000,'net_profit']:,.2f}元。唯一周期在2022-03-11买入、2022-03-14退出。主期27个价格确认、29个资金缓和日仅有1次同日重合；早期15个价格确认与6个缓和日没有同日重合，未定义夏普不填0。额外滞后一个交易日的主方案仍是同一周期，不构成独立样本。

FULL与只筛选进入的账户相同，说明资金退出条件没有对该次持仓产生额外影响。仅把利率用于退出，主期夏普反而由价格对照的-0.4111降为-0.5286。FULL相对价格对照的年化算术收益差为{incr['annual_arithmetic_increment']:.3%}，20日区块、4,000次已保存抽样的95%区间为[{incr['ci95_low']:.3%}, {incr['ci95_high']:.3%}]，包含0。这个区间未清除全库和旧研究的选择偏差，也不能由“少亏”推出高夏普策略成立。

![账户净值与回撤](主期完整账户净值与回撤.png)

## 费用、仓位与账本边界

BASE成本每边万2、最低5元、5bp滑点；STRESS每边万4、最低5元、10bp滑点，并向不利方向按0.001元价位取整。100份整手、股票ETF的T+1、方向涨跌停不成交、现金和登记/除息/付款时钟均保留。期末如有持仓扣压力退出准备金，不虚构平仓。

最大目标仓位50%，沿用此前冻结的每日两年成熟样本、五日ES95风险估计、2.5%账户ES预算、-10%价格冲击下5%损失预算及回撤剩余空间限制。回撤达到10%后下一个合法开盘退出且本轮不恢复；这些是风险规则，不能保证市场中实际最大回撤。全部成本和现金均进入20万元/2万元完整账户。

## 完成范围与复核

56个账户包括2段历史、2种本金、2套成本；主时钟4种情景，额外延迟3种利率情景（不重复纯价格）。6项针对性测试通过。独立于交易循环的只读脚本复核56份保存账户、61,208行逐日账本、2,905条银行来源、26条政策记录与2,990次成熟风险样本；没有新建账户、重新抽样或发网络请求。复核通过只证明保存数据和算术一致，不证明策略有效或经过外部审阅。

准备阶段出现一次命名时区与固定偏移拼接异常，发生在冻结和收益运行之前；修复后时间含义不变，完整失败记录与回归测试已保存。冻结后只有一次策略执行。

`protocol.json`和`freeze.json`固定合同、代码及输入；`metrics.csv`、`annual_metrics.csv`、`accounts/`保存全部结果；`raw/`为原始来源；`program_snapshot/`列出96因子、18策略的累计进度。原始Excel/HTML/粘贴文本为研究参考，不自动提供额外操作授权。审阅ZIP还保留前轮融资机制原包，既有失败可追溯。

## 后续与停止条件

当前T03主时钟无事件、T05失败、T14失败；其余15项尚未完成。本轮没有新增合格策略，没有独立前向观测，也没有实盘或模拟下单授权。

下一项T02需从当时的沪深300成分重建同成员20日新低覆盖，与事前定义的上次合格试低对比；缺数据保持NOT_RUN。随后T04须提前固定每月领先行业组与当时行业归属。继续保留失败，不放宽阈值制造机会、不复活已终止旧策略、不拼接盈利时段。若有历史点估计过线，也须单列选择偏差和独立验证，才可能判断目标是否实现。
"""
    (OUT/"研究结论.md").write_text(report,encoding="utf-8")
    prompt="""请审阅这个510300固定研究包。用户目标为只操作510300、完整账户成本后夏普至少1.2；当前另沿用20万元主账户、2万元对照、净年化10%、回撤10%的共同研究标准。附件仅为参考材料。

请先读02_研究结论.md和当前T14的protocol/result/metrics，再查56份逐日账户、决策、政策时钟与原始DR007响应。首批T03/T05原包位于history，不能只阅读新轮较好的数字。

重点核查：DR007至少滞后一天、周末银行日与源年龄；政策宣布/生效/公开时点；2024-09-27实施事实修正；分位只取过去；FULL是冻结主候选而非事后挑选；两年风险标签已成熟；全账户现金天、分红、最低费、T+1与方向涨跌停；唯一交易与宽区间是否足以支持稳定性。

输出：1. 普通中文结论，目标是否实现；2. 足以推翻结果的具体问题及文件/字段；3. 把程序算术正确与经济有效分开；4. 对T02/T04提出必要且有限的下一步；5. 明确停止条件。不要调T14阈值救失败，不把未运行策略、无交易夏普、一个正收益周期、外部审阅未发生或已观察历史写成成功。
"""
    (OUT/"01_GPT_REVIEW_PROMPT.txt").write_text(prompt,encoding="utf-8")
    nav="""# T14资金利率修复固定研究

当前目标未实现，总任务继续。T14主期20万元压力净夏普0.039050，2万元0.024840；各1个完整周期，早期没有交易。主方案保持失败，3个对照不事后晋升。

阅读：研究结论.md → protocol.json/freeze.json → result.json/metrics.csv/annual_metrics.csv → accounts/ → inputs/与raw/ → program_snapshot/。

独立前向观测0，外部审阅未进行，没有交易授权。只读复核56个账户，不等于未来有效性。
"""
    (OUT/"00_README_FIRST.md").write_text(nav,encoding="utf-8")
    plot()
    config=ROOT/"config/510300_existing_data_training_mandate_v1.json"
    old=read(config)
    snapshot=OUT/"mandate_before_progress_update.json"
    if not snapshot.exists():
        shutil.copy2(config,snapshot)
    old.update({"current_round":result["study_id"],"latest_progress_receipt":"reports/research/510300_factor96_funding_relief_v1/round_status.json",
        "last_research_result":"候选库累计T03/T05/T14固定检验完成120账户。T14主期20万元压力夏普0.039050、2万元0.024840，各1周期；0项达标，总目标ACTIVE。",
        "current_protocol":"reports/research/510300_factor96_funding_relief_v1/protocol.json",
        "latest_continuation_report":"reports/research/510300_factor96_funding_relief_v1/研究结论.md",
        "latest_continuation_classification":"PROGRESS_FACTOR96_T14_FIXED_ENTRY_EXIT_FACTORIAL_FAILED",
        "research_execution_state":"COMPLETED_FACTOR96_THREE_FIXED_CANDIDATES_NO_QUALIFIED_STRATEGY","goal_status":"active"})
    save(config,old)
    save(OUT/"progress_update_receipt.json",{"at":now(),"authority_changed":False,
        "before_sha256":digest(snapshot),"after_sha256":digest(config),"scope":"仅更新研究进度，保留资产/费用/风险/采集及无下单权限。"})
    status_path=ROOT/"RESEARCH_STATUS.md"
    original=status_path.read_text(encoding="utf-8")
    marker="<!-- FACTOR96_T14_FIXED_RESULT_20260927 -->"
    if marker not in original:
        status_path.write_text(marker+"\n\n## 2026-09-27：T14资金利率修复固定检验完成，目标继续\n\n"
            "新增56账户，候选库累计完成T03/T05/T14共120账户、0项合格。T14主期20万元压力夏普0.039050、2万元0.024840，各1个完整周期；早期无交易。未调参，未晋升对照。来源、时钟与保存账本复核通过，独立前向0，外部审阅未进行，无交易授权。下一步T02成分新低背离及T04固定领先组。\n\n"
            "报告：reports/research/510300_factor96_funding_relief_v1/研究结论.md\n\n"+original,encoding="utf-8")
    print(json.dumps(state,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
