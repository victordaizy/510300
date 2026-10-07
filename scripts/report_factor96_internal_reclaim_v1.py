"""明确区分T02正式结果与停牌字段缺陷运行，保留完整候选进度。"""
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
from research.factor96_internal_reclaim_v1_0_1 import OUT,save,now,digest

ORIGINAL=ROOT/"reports/research/510300_factor96_internal_reclaim_v1"
PROGRAM=ROOT/"reports/research/510300_factor96_program_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def figure():
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig,axes=plt.subplots(2,1,figsize=(12.5,8.6),sharex=True,gridspec_kw={"height_ratios":[1.3,1]})
    styles=[("PRICE_ALL",0,"全部价格事件对照","#c89137","-"),
        ("PRICE_COMMON",0,"同覆盖价格对照","#758196","--"),
        ("FULL",0,"主方案：同成员新低收缩","#b43950","-"),
        ("FULL",1,"敏感性：额外等待一日","#198695",":")]
    for name,lag,label,color,style in styles:
        z=pd.read_parquet(OUT/f"accounts/MAIN/200000/STRESS/LAG{lag}/{name}/ledger.parquet")
        dates=pd.concat([pd.Series([z.date.iloc[0]-pd.Timedelta(days=1)]),z.date],ignore_index=True)
        nav=np.r_[1.,z.equity/200000]
        axes[0].plot(dates,nav,label=label,color=color,ls=style,lw=1.6)
        axes[1].plot(dates,(nav/np.maximum.accumulate(nav)-1)*100,color=color,ls=style,lw=1.6)
    axes[0].set_title("T02 二次试低与内部背离：修正停牌字段后的正式结果",fontproperties=font,fontsize=17,loc="left",pad=58)
    axes[0].legend(prop=font,loc="lower left",bbox_to_anchor=(0,1.01),ncol=2,frameon=False)
    axes[0].set_ylabel("完整账户净值",fontproperties=font)
    axes[1].set_ylabel("自账户峰值回撤（%）",fontproperties=font)
    axes[1].set_xlabel("2021—2025年；包含所有现金日、分红与费用",fontproperties=font)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.text(.08,.025,"20万元 · 每边万4，最低5元，10bp滑点并向不利方向取整 · 延迟版本不晋升为主方案",fontproperties=font,fontsize=10,color="#555555")
    fig.tight_layout(rect=[.02,.055,.99,.99])
    fig.savefig(OUT/"主期完整账户净值与回撤.png",dpi=160)
    plt.close(fig)


def main():
    result=read(OUT/"result.json")
    assert read(OUT/"saved_verification_receipt.json")["ledgers"]==40
    m=pd.read_csv(OUT/"metrics.csv")
    main=m[(m.period=="MAIN")&(m.cost=="STRESS")&(m.lag==0)]
    primary=main[main.policy.eq("FULL")].set_index("capital")
    inc=result["increment"]
    labels={"PRICE_ALL":"全部价格事件","PRICE_COMMON":"同覆盖价格对照","FULL":"主方案：内部新低收缩"}
    table=[]
    for r in main[main.capital.eq(200000)].itertuples():
        table.append(f"|{labels[r.policy]}|{r.net_sharpe:.4f}|{r.cagr:.3%}|{r.max_drawdown:.3%}|{r.closed_cycles}|")
    for r in m[(m.period=="MAIN")&(m.cost=="STRESS")&(m.capital==200000)&(m.lag==1)].itertuples():
        table.append(f"|额外一日：{labels[r.policy]}|{r.net_sharpe:.4f}|{r.cagr:.3%}|{r.max_drawdown:.3%}|{r.closed_cycles}|")
    report=f"""# T02二次试低与内部背离：正式修正结果

**只操作510300、完整账户成本后夏普1.2的目标仍未实现。** T02固定主方案在2021—2025年、20万元压力账户中的夏普为{primary.loc[200000,'net_sharpe']:.6f}，净年化{primary.loc[200000,'cagr']:.3%}、最大回撤{primary.loc[200000,'max_drawdown']:.3%}，11个完整周期；2万元账户夏普{primary.loc[20000,'net_sharpe']:.6f}，也未达标。内部背离相对同覆盖价格对照有小幅改善，但收益增益区间包含0，不能证明稳定有效。额外延迟一日的主方案夏普0.540980仍不达标，不因数字更好替换主时钟。候选库累计完成T03、T05、T14、T02共4项，0项合格。

## 实现修正与哪些结果可以使用

正式结果位于`510300_factor96_internal_reclaim_v1_0_1`。原V1已有40个账户，在独立复算阶段发现：14,771条官方全日停牌记录的当日未复权收盘为空，已知前收盘有效；V1错误排除了这些记录的低价窗口。V1账户和冻结文件全部保留，但不作为经济结论。原V1主期20万元主方案夏普-1.358776已被否定为不符合原定停牌处理规则的结果。

修正只对“官方确认全日停牌、无公司行为、前收盘有效、原四态可用”的记录，以前收盘作为当日平值；普通缺失不填零。两版本的全部直接来源哈希完全相同，策略阈值、事件、持有期、费用、覆盖门与风控不变。8项测试通过，新版本另行冻结。`source_evidence/implementation_diff.patch`给出完整代码差异，修复回执与旧失败同包保留。修复发生在已经看到旧结果之后，必须保留这项开发历史，不能当作新的独立验证。

本轮实际计算80个账户：40个已否定实现账户和40个正式修正账户；全候选库累计实际计算200个，其中160个作为当前固定候选结果，40个保留为实现失败。40个情景不是40个独立策略。

## 固定问题和规则

问题是：指数再次接近旧低点时，跌破各自20日低点的成分股减少，是否表明内部卖压衰减、随后修复更可持续。反例是少数权重股继续恶化，等权新低减少也挡不住指数下跌。

首次低点a必须严格跌破此前20日财富低点；后两天低价都高于a，等a+2收盘才确认。二次试低发生在a+3至a+10，a仍为前20日最低，二测低价距a严格小于0.5ATR[a-1]。每个首次低点只认领一次二测，不重叠追认。二测b冻结前20日低点L0；当天或随后10日内第一次财富收盘>L0时完成收复。未收复和到期事件保留，不从母集删除。

成分新低使用原始最低价、可解释的现金分红与股数转换重建财富低价。连续21日低价与收益必须可用，当前最低价与前20日比较。a、b两日取共同属于指数且两边价格窗口都完整的同一批股票；至少294/300名可比，两日自身覆盖也至少98%。同分母下二测的新低数量严格低于首次测试，才称内部背离。内部值在b固定，收复时不重新挑有利日期。B02成交额、回报和收盘位置只记录，不追加过滤。

主方案同时要求收复、覆盖合格、内部新低减少。PRICE_COMMON删除背离条件但保留同覆盖，PRICE_ALL只用价格事件。主要增量为FULL减PRICE_COMMON，从而把信息作用与缺数据筛选分开。收盘失守L0-0.5ATR[b-1]、五个开盘间隔到期或公共风险触发，则下一合法开盘卖出。

## 可得时间和来源

当日成分日线统一按18:00可得，ETF买单从下一开盘开始；股票日线原始接口说明其不复权且停牌期间无数据，因此不能凭日线缺行就认定零收益。[Tushare日线字段说明](https://tushare.pro/document/2?doc_id=27)

成员使用官方调样重放，明确“收盘后生效”的调整在下一交易日切换。包内补充65份成员调整原文、附件和2015历史锚点；没有用当前成员回填。股价原始表、公司行为四态记录和官方停牌证据分开保存，没有把供应商复权因子当成原始价格。总计668个历史成员身份参与数据准备。

2021—2025年新低覆盖可用1,200/1,212日，2017—2020年808/974日；早期覆盖较差，不把缺数据少交易当成稳健。18个主期价格确认中17个满足同覆盖、11个存在背离。早期7个价格确认中3个同覆盖、2个背离。全价格历史记录42次二测，其中40次收复、2次过期未收复；这一全历史计数不等于评价区间交易次数。

所有来源均为后来重建的历史记录，不证明当年存在可持续、准时的实时采集。原始版本认证和独立前向样本仍不足。

## 账户与对照

下表为2021—2025年、20万元、压力成本；年化242日，现金与无风险收益为零，完整保留全部1,212个交易日。

|固定情景|净夏普|净年化|最大回撤|完整周期|
|---|---:|---:|---:|---:|
{chr(10).join(table)}

20万元主方案期末{primary.loc[200000,'end_equity']:,.2f}元，净损益{primary.loc[200000,'net_profit']:,.2f}元；2万元期末{primary.loc[20000,'end_equity']:,.2f}元，净损益{primary.loc[20000,'net_profit']:,.2f}元。早期20万元主方案夏普-0.279878、年化-0.120%、2个完整周期。不能用某段亏损较小或延迟版本转正替换整体结论。

主方案相对同覆盖价格对照的年化算术收益差{inc['annual_arithmetic_increment']:.3%}；20日区块、4,000次已保存抽样的95%区间为[{inc['ci95_low']:.3%}, {inc['ci95_high']:.3%}]，包括0。没有进行全候选库选择偏差调整，已观察历史也不算独立样本外。

![正式账户净值与回撤](主期完整账户净值与回撤.png)

BASE每边万2、最低5元、5bp滑点；STRESS每边万4、最低5元、10bp滑点并按0.001元向不利方向取整。100份整手、股票ETF T+1、方向涨跌停不成交、登记/除息/付款和现金约束均进入账本。[上交所股票ETF交易制度说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)

最大目标仓位50%，复用此前冻结的两年成熟样本、每日五日ES95风险估计，沿用2.5%ES预算、-10%冲击下5%损失预算及回撤剩余空间限制。10%回撤触发后下一个可成交开盘清仓且不重启；不是实际最大回撤保证。

## 验证与下一步

8项针对性测试覆盖除权、真实停牌字段、缺失窗口重建、两日确认、未来数据不改变过去、失败事件留存、同成员分母和交易时钟。只读复核从每个21日窗口重新起算，不沿用研究代码跨缺口的计算坐标；并复算40份保存账户、全部事件、费用分红与抽样增益。通过只表示保存结果一致，外部审阅未进行，独立前向0，没有交易授权。

保持T02固定失败，不改0.5ATR、10日窗口、98%覆盖、5日退出或延迟时钟以救结果。T04需要当时行业与可核验历史权重；当前成员准入不等于权重准入，后者已有来源门缺口，应先解决再检验固定领先组。并行推进不依赖该缺口的其他独立机制，避免反复改造这些失败组合。
"""
    (OUT/"研究结论.md").write_text(report,encoding="utf-8")
    figure()
    strategies=read(PROGRAM/"strategy_progress.json")
    factors=read(PROGRAM/"factor_progress.json")
    for row in strategies:
        if row["id"]=="T02":
            row.update(current_status="COMPLETE_FIXED_DEFINITION_TARGET_FAILED",current_evidence="V1停牌字段缺陷已保留并否定；V1.0.1正式40账户。主期20万元压力夏普-0.616675，2万元-0.704129，主方案均未达标。",
                result_path="reports/research/510300_factor96_internal_reclaim_v1_0_1/result.json")
        if row["id"]=="T04":
            row.update(current_status="NOT_RUN_PIT_WEIGHT_SOURCE_GATE",current_evidence="官方点时成员可用；历史指数权重版本证据仍未准入。须先核验固定领先组贡献口径与权重，不能改用等权声称完成原策略。")
    for row in factors:
        if row["id"] in ["B02","B03","E02"]:
            row["current_status"]="PARTIAL_COMPONENT_TESTED"
            row["current_note"]={"B02":"两日确认首次低点、十日内二测；成交额/回报/CLV只记录。T02完整账户未达标。",
                "B03":"二测日冻结前20日低点，收复后才请求下一开盘；2个未收复事件保留。",
                "E02":"由原始低价及可解释公司行为构成21日完整窗口；两次同成员同分母新低收缩已检验，T02失败，无单因子有效性结论。"}[row["id"]]
    state={"at":now(),"goal_status":"active","goal_achieved":False,"registered_factors":96,"registered_strategies":18,
        "completed_fixed_candidates":["T03","T05","T14","T02"],"not_run_candidates":14,"qualified_candidates":[],
        "admitted_account_scenarios_this_round":40,"invalid_implementation_accounts_this_round":40,
        "cumulative_admitted_account_scenarios":160,"cumulative_executed_account_scenarios":200,
        "latest_round":result["study_id"],"latest_result":"reports/research/510300_factor96_internal_reclaim_v1_0_1/result.json",
        "next_candidates":["T04_SOURCE_GATE","T01_OVERLAP_CONTRACT","T09_SOURCE_GATE"],
        "independent_forward_observations":0,"external_review":"NOT_PERFORMED","orders_authorized":False}
    for folder in [PROGRAM,OUT/"program_snapshot"]:
        folder.mkdir(parents=True,exist_ok=True)
        save(folder/"strategy_progress.json",strategies)
        save(folder/"factor_progress.json",factors)
        save(folder/"status.json",state)
        pd.DataFrame(strategies).to_csv(folder/"18策略当前进度.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(factors).to_csv(folder/"96因子当前进度.csv",index=False,encoding="utf-8-sig")
    save(OUT/"round_status.json",state)
    (OUT/"01_GPT_REVIEW_PROMPT.txt").write_text("""请审阅T02固定候选及其实现修正。用户只允许交易510300与人民币现金，目标为完整账户成本后夏普1.2；当前协议并列年化10%与回撤10%。

请先读02_研究结论.md、V1.0.1协议/结果/指标，再读原V1 verification_failure_01.json和实施差异。不能引用旧V1的-1.358776作为正式结果，也不能把额外延迟的0.540980提升为新主方案。

重点：两日低点确认是否看未来；所有不收复事件是否保留；20日新低是否由可解释公司行为原始价重建；官方停牌与普通缺失是否分开；两个测试日期是否为同一批历史成分同分母；294覆盖是否真正满足；资金、T+1、分红、费用、风险样本成熟及完整现金天是否一致。

输出普通中文结论、会推翻结果的问题和文件证据、下一步需要的新证据及停止条件。分开程序正确与策略有效。不要通过调T02阈值、改变失败分支或拼接盈利年份制造成功。T04的成员准入不代表权重准入。独立前向与外部审阅的真实状态均要保留。
""",encoding="utf-8")
    (OUT/"00_README_FIRST.md").write_text("T02正式结果在本目录V1.0.1，主期20万元压力夏普-0.616675，目标未达。原V1停牌字段处理有缺陷，40个账户保留但不采信。先读研究结论.md，再读协议、结果、事件、成员窗口与账户。总目标继续；无独立前向或交易授权。",encoding="utf-8")
    config=ROOT/"config/510300_existing_data_training_mandate_v1.json"
    snapshot=OUT/"mandate_before_progress_update.json"
    if not snapshot.exists():
        shutil.copy2(config,snapshot)
    mandate=read(config)
    mandate.update(current_round=result["study_id"],latest_progress_receipt="reports/research/510300_factor96_internal_reclaim_v1_0_1/round_status.json",
        last_research_result="累计4个固定候选均未达标；T02停牌字段修正后40账户，主期20万元压力夏普-0.616675。另保留40份否定实现账户，总目标ACTIVE。",
        current_protocol="reports/research/510300_factor96_internal_reclaim_v1_0_1/protocol.json",
        latest_continuation_report="reports/research/510300_factor96_internal_reclaim_v1_0_1/研究结论.md",
        latest_continuation_classification="PROGRESS_FACTOR96_T02_REPAIRED_IMPLEMENTATION_FIXED_TARGET_FAILED",
        research_execution_state="COMPLETED_FACTOR96_FOUR_FIXED_CANDIDATES_NO_QUALIFIED_STRATEGY",goal_status="active")
    save(config,mandate)
    save(OUT/"progress_update_receipt.json",{"at":now(),"before_sha256":digest(snapshot),"after_sha256":digest(config),"authority_changed":False})
    status=ROOT/"RESEARCH_STATUS.md"
    text=status.read_text(encoding="utf-8")
    marker="<!-- FACTOR96_T02_FIXED_CORRECTION_20260927 -->"
    if marker not in text:
        status.write_text(marker+"\n\n## 2026-09-27：T02二次试低内部背离完成，目标继续\n\n"
            "正式V1.0.1主期20万元压力夏普-0.616675、2万元-0.704129；40个正式账户均保留。旧V1官方停牌字段缺陷导致的40账户另行否定保存，未改策略参数。累计4项固定候选、160个当前正式账户、200个实际执行账户，0项合格。T04权重版本证据仍未准入。报告：reports/research/510300_factor96_internal_reclaim_v1_0_1/研究结论.md\n\n"+text,encoding="utf-8")
    print(json.dumps(state,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
