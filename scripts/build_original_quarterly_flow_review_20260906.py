"""生成第十轮中文解释、图表及含原始申赎报告的自包含审阅包。"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.original_quarterly_flow_policy_v1 import OUT,SOURCE,PARENT,CONFIG,MANIFEST,NAMES,physical,sha
from research.intraday_overnight_increment_v1 import now
from scripts.verify_original_quarterly_flow_saved_20260906 import verify_saved

DELIVERY=ROOT/"deliverables/510300第十轮_原始季度申购赎回_20260906"
ZIP=ROOT/"deliverables/510300夏普1.2持续研究_第十轮原始申赎_GPT审阅_20260906.zip"
REPORT_NAME="第十轮结果与全部中文因子规则.md"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return f"{value*100:.2f}%"


def md_table(frame,columns,labels,formats):
    lines=["| "+" | ".join(labels)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for row in frame.to_dict("records"):
        lines.append("| "+" | ".join(formats.get(col,str)(row[col]) for col in columns)+" |")
    return "\n".join(lines)


def prepare():
    DELIVERY.mkdir(parents=True,exist_ok=True)
    assert not (DELIVERY/REPORT_NAME).exists(),"第十轮中文交付已经准备，避免覆盖"
    result=read(OUT/"result.json")
    receipt=verify_saved()
    (DELIVERY/"保存结果核对.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    config=read(CONFIG)
    metrics=pd.read_csv(OUT/"metrics.csv")
    eras=pd.read_csv(OUT/"era_metrics.csv")
    base=metrics.loc[metrics.cost=="BASE"].copy()
    stress=metrics.loc[metrics.cost=="STRESS"].set_index("model")
    base["压力夏普"]=base.model.map(stress.net_sharpe)
    base["策略"]=base.model.map(NAMES)
    table=md_table(base,["策略","net_sharpe","压力夏普","annualized_return","max_drawdown","trade_count"],
                   ["方案","基础夏普","压力夏普","基础年化收益","基础最大回撤","基础成交次数"],
                   {"net_sharpe":lambda x:f"{x:.4f}","压力夏普":lambda x:f"{x:.4f}","annualized_return":percent,"max_drawdown":lambda x:percent(-x)})
    e=eras.loc[eras.cost=="BASE"].pivot(index="model",columns="era",values="net_sharpe").reindex(base.model)
    e.insert(0,"策略",e.index.map(NAMES))
    eras_table=md_table(e.reset_index(),["策略","2020—2021","2022—2023","2024—终点"],
                       ["方案","2020至2021夏普","2022至2023夏普","2024至终点夏普"],
                       {k:lambda x:f"{x:.4f}" for k in ["2020—2021","2022—2023","2024—终点"]})
    protocol=(ROOT/"docs/510300_ORIGINAL_QUARTERLY_FLOW_POLICY_V1_PROTOCOL.md").read_text(encoding="utf-8")
    report="""# 第十轮：公募申赎信息的原始数据检验

**完整账户成本后夏普1.2尚未实现，研究继续。** 本轮新增7个候选、16条完整评价账户。主方案“价格加季度申赎”夏普负0.0135，年化收益负1.51%，最大回撤48.11%。同样节奏的仅价格模型夏普0.0107；加入本轮三项申赎指标没有获得可确认的改善。

本轮最高为“净赎回后持有”的反向规则：夏普0.3114、年化2.63%、最大回撤11.55%。它的回撤较小，但收益低于买入持有的年化3.63%，不能当作稳定超额或1.2已经达标。

## 老板先看：为什么要研究，究竟验证了什么

盈利告诉我们公司赚多少钱；估值倍数反映市场愿意为盈利支付多少价格；分红和股份变化影响股东获得的回报；公募申购赎回反映一部分持有需求。本轮先把其中的“申赎”落实到可核对的原始报告，再检验它是否真的对510300有用。

本轮使用510300本体季报中的申购份额和赎回份额。它只代表这只ETF，不能替代全市场公募申购。ETF申赎对价可能是证券组合、现金或其他约定对价，因此基金份额增加不是同金额的现金流入。[上交所关于ETF申赎的说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734753.shtml)

数据不能在季度末提前使用。以2021年第三季度为例，季报10月27日公布，本轮10月28日收盘首次使用，10月29日开盘才能执行。原始报告期初91.86787690亿份，申购43.974亿份、赎回57.096亿份、期末78.74587690亿份，净赎回13.122亿份，四项衔接一致。[510300原始2021年三季报](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2021-10-27/510300_20211027_1_INbeqLrb.pdf)

## 历史表现：完整区间与两档费用

模拟本金20万元；2020年1月2日至2026年8月14日开盘退出，共1604个交易日。结果包含佣金、最低收费、滑点、分红权益与到账、整手、下一开盘执行及T+1限制。夏普参考利率与现金收益假设均为零。最大回撤列用正数表示亏损幅度。

"""+table+"""

每一条都是完整账户结果，初期未持有、后期退出和所有中间日期均保留。基础佣金为万分之二、最低5元、单边滑点万分之五；压力佣金万分之四、最低5元、单边滑点千分之一。压力费用会影响模型的效用选择，成交次数因此可能变化。

![完整账户净值与回撤](第十轮_完整账户净值与回撤.png)

## 每个时期是否都有效

"""+eras_table+"""

主方案在2022至2023年夏普为负1.2995，在2024至终点恢复至0.8790，不能只挑后一段说明策略有效。净赎回反向规则三个分期夏普都为正，但均低于0.4，完整目标仍有明显距离。

用同样的20日连续区块共同抽样2000次，主方案基础夏普95%区间为负0.7521至0.6822；相对仅价格模型的年化日均收益增量区间为负2.79至正2.29个百分点，相对同公告趋势规则为负8.74至正4.73个百分点。区间都未支持稳定改善。这些区间没有完成所有历史尝试的多重选择校正；历史已被反复观察，独立验证尚未建立。

本轮每次训练只有25至51份已经到期的季度观察，合计28个拟合时点、三组模型84次拟合。全部评价时点均有合格输入，但真正的新信息仍很稀疏。净申购可能与机构配置、套利、逆势买入或既有行情同时发生，本轮没有识别这些行为各自的因果效应。

## 原始资料与本轮修正

上交所2012年至2026年的逐年公告查询共保存83条定期报告目录，各年返回总数与唯一记录数一致。最终定位2012年三季度至2026年二季度56份季报，55份有上交所公告日期。2013年二季报在巨潮取得原始PDF，数值完整，但官方公布目录证据未找到，保留为不可用于策略。2013年三季度因为缺乏合格上一季来源，亦不能计算本轮的净申赎变化；第一份报告缺上一季度，不使用。

第一版单页识别有9份跨页申赎表未读完整。修正版本只拼接同一报告中标题所在页与紧邻下一页，要求五个明确份额项目顺序、单位和数值等式正确。修正后56份份额等式与55次相邻期衔接全部吻合，原版失败记录保留。修正发生在第十轮标签、拟合及账户收益读取之前。

公告目录日期与报告送出日取较晚者，再延至下一交易日。PDF创建时间只用于排查晚生成版本，不能证明首次公开时点；未声称持有历史当时就保存的密码学收据。早期PDF属性标题沿用其他基金模板，身份以正文封面与510300代码核实。

![原始季度净份额变化](第十轮_原始季度净份额需求.png)

## 每个因子与每个策略的全部中文规则

以下为收益计算前已冻结的完整协议。三组模型和四条规则均使用同一批六因子有效的季度来源；仅价格对照也遵守相同事件节奏与样本范围。

"""+protocol.replace("# 第十轮：原始季度申购赎回是否改善510300判断","### 收益读取前的完整研究协议",1)+"""

## 接下来：把盈利、估值与股东回报补成独立信息

本轮7条候选结果完整保留。下一阶段优先补金融成分公司的原始合并财报、普通股每股盈利以及原始股本、分红、回购执行资料，形成同口径盈利和估值输入；同时继续解决全市场公募月报的原始发布日期与分类变化。不得把这个单只ETF的季度申赎试验解释为全部公募需求已经研究完。

第九轮及更早的结果作为历史背景列入包内摘要，不在本轮重训或复算。当前十轮合计161个登记配置、342条完整评价账户，含重复对照；总体事后最高仍为第六轮月末月初规则夏普0.5196，未达到1.2。详细经济定义另见《盈利估值股东回报与公募需求_中文框架.md》。本轮仅研究510300与现金，没有真实下单。
"""
    (DELIVERY/REPORT_NAME).write_text(report,encoding="utf-8")
    framework="""# 盈利、估值、股东回报与公募需求：统一中文框架

研究目标继续保持完整账户成本后夏普至少1.2，并检验稳定超额。用户提出的四个方面都纳入研究，但必须各自有独立来源、明确单位和当时可用日期。

## 盈利与估值如何连起来

同一每股盈利口径下，每股价格可写成“每股盈利乘以市盈率”。它把价格分解为企业赚钱能力与市场定价倍数，提供解释框架。用当前价格除以当前市盈率再得出每股盈利，随后乘回市盈率，仍然只是恢复原价格，没有增加预测信息。

真正的研究要分别判断未来盈利变化、估值倍数可能如何变化，并检验能否优于当时已知的价格信息。指数层面还需使用当时成分和一致的加权口径，不能把不同公司的每股盈利简单相加，也不能把成分股市盈率的简单平均当成指数估值。盈利为负或口径不可比时，保留无法计算的状态。

普通股每股盈利的分子应对应普通股股东可享有盈利，分母对应同期间加权普通股数；优先股及其他权益工具可能影响分子。[IFRS每股收益准则说明](https://www.ifrs.org/issued-standards/list-of-standards/ias-33-earnings-per-share.html/) A股各公司仍按其原始报告具体披露核实。招商银行原始实例已经表明，归母利润、普通股盈利、母公司利润和合并利润不能直接互换；单季与年初累计也必须分开。

## 股东回报如何避免重复计算

持有期总回报包括价格变化和期间有权获得的现金分红，均除以期初投入价格进行比较。股价水平与股息率单位不同，不能直接把两者相加。

原有510300完整账户已经按分红登记、除息形成应收、支付到账计入回报。把股息率作为预测因子时，只使用当时已公布的分红信息，不能把同一笔分红再加一次到账户收益。成分公司分红还要区分公司层面现金与ETF层面实际分配。

股份回购分别记录计划授权、实际执行、注销和净股本变化。计划金额不能冒充已发生回购；向其他股东回购股票，不等于所有持有人都收到同样现金。回购注销可能通过股数变化影响每股盈利，若该影响已经计入每股盈利，就不能再次把同一回购金额当成独立收益叠加。增发和股权激励等增加股数的行为同时记录。

## 公募申赎如何形成需求因子

全市场股票及混合基金规模变化同时包含净值涨跌和申赎等效应，不能直接称为现金净流入。份额变化也需排除新成立、清盘、份额折算和分类切换。主动权益基金、被动ETF以及新基金募集分别观察，再按有证据的共同口径汇总。

第十轮已经把510300本体季报的申购、赎回、期初、期末和拆分份额接入。三项因子是净份额变化占期初份额、该比例较上一季的变化、双向申赎活动占期初份额。它们的单位都是份额比例，不能称为精确人民币现金流。ETF申赎可以使用证券组合或现金，且可能服务于套利；正净申购与未来上涨的关系必须验证。[上交所ETF申赎说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734753.shtml)

第十轮所有7种方案均未达夏普1.2；这说明本轮单只ETF季度信息与固定用法尚不足够，不能推出公募需求整体无用。全市场月报另有77份来源正在恢复原始公布时钟，当前不能把它们全部当作已准入策略。

## 如何组合成下一项可检验研究

先从原始财报形成盈利增长与变化方向，从当时价格及同口径盈利形成估值，从已宣布和已执行的分红回购形成股东回报信息，再加入独立可用的申赎需求。各组先与价格对照做增量比较，缺失输入明确保留无判断。通过后才检验合成仓位，并完整计入交易成本、分红和现金。研究前固定方法和比较对象，不以某个漂亮分期代替完整区间。

当前下一项研究尚未运行，具体有限方案将在新来源达到可用条件后另行登记。已完成十轮的结果不因换解释框架而改变。
"""
    (DELIVERY/"盈利估值股东回报与公募需求_中文框架.md").write_text(framework,encoding="utf-8")
    state=read(ROOT/"reports/research/510300_sharpe_1_2_latest_research.json")
    (DELIVERY/"历史九轮结果摘要.json").write_text(json.dumps(state["completed_rounds"][:9],ensure_ascii=False,indent=2),encoding="utf-8")
    (DELIVERY/"00_README_FIRST.md").write_text("# 第十轮阅读顺序\n\n1. `第十轮结果与全部中文因子规则.md`：老板摘要、全部方案、完整中文因子和规则。\n2. `盈利估值股东回报与公募需求_中文框架.md`：响应用户的经济框架与防止重复计算。\n3. `01_GPT_REVIEW_PROMPT.md`：可直接交给GPT的评议要求。\n4. `reports/research/510300_original_quarterly_flow_policy_v1`：全部16账户、28事件、84模型、标签、拟合收据、全部分年分期结果与保存的区块统计。\n5. `reports/research/510300_original_fund_subscription_reports_v1`及`data/raw/510300_original_fund_subscription_reports_v1`：56份原始PDF、83条目录、逐年查询、公布时钟、跨页修正前后记录和图像。\n6. `config`、`docs`、`research`、`tests`及`scripts`：冻结规则、完整实现与只读复算入口。\n7. `FILE_INDEX.csv`是本包文件、字节数和哈希的权威索引，不包含它自身的哈希。\n\n只读复算脚本是`scripts/verify_original_quarterly_flow_saved_20260906.py`，从包的根目录运行即可。它不下载、不拟合、不新建账户或随机抽样，读取本地保存模型进行预测一致性检查。依赖版本见`离线复算依赖版本.txt`。不要重新运行已完成的冻结研究。\n\n本包包含当前轮直接输入和全部结果；既往九轮仅作为摘要及保存指标背景，不包含其全部原始数据、模型与账本，也没有重新复算它们。前九轮已有各自完整审阅包。本包不声称独立高夏普证据成立；未上传，未收到外部审阅；仅做结构与必要数值检查。\n",encoding="utf-8")
    (DELIVERY/"01_GPT_REVIEW_PROMPT.md").write_text("# 请评议第十轮原始季度申赎研究\n\n用户要求使用免费来源，寻找510300与现金的完整账户成本后夏普至少1.2及稳定超额；持续研究，不因单轮结束停止。用户还要求全部期限逆回购、盈利乘估值、股东回报与公募申购均正确处理；中文写出每个因子和规则。\n\n请先检查原始报告的基金身份、单位、跨页表、公布日期、份额恒等式与55次相邻期衔接；2013年二季报没有官方目录证据，不能准入。检查修正是否确有原始证据，且在新策略收益前冻结。\n\n再检查季度只出现一次训练起点、成熟标签次序、每组匹配训练集、初始化日期、事件外保持份额、未来信息、费用、分红、现金、整手和T+1。主方案与对照是否足以回答增量问题？60日标签和实际季度公告间隔的差异是否影响解释？区块抽样和反复历史选择还有哪些局限？\n\n全部7候选均未达到1.2。请评议失败原因，区分已经验证的缺陷与合理但未证明的假设。不能因最高规则表现相对较好就宣布有效，也不能从一个ETF申赎试验推断全部公募需求无用。对盈利、估值、普通股EPS、分红和回购的定义给出反证性检查，避免循环恒等式、修订版本和重复计回报。\n\n最后提出下一轮有限、可执行的新机制、所需免费原始来源、优先顺序、最小有效样本、独立验证方式和明确停止条件。不要推荐持续改变同一批阈值直到凑到1.2。请给出至少一条不依赖最难恢复资料的独立推进路径。旧九轮摘要仅背景，本包没有重跑旧研究。没有外部审阅结果，不能自称已被验证；不授权真实订单。\n",encoding="utf-8")
    packages=["numpy","pandas","pyarrow","scikit-learn","joblib","pdfplumber","pypdfium2","matplotlib","scipy","threadpoolctl"]
    (DELIVERY/"离线复算依赖版本.txt").write_text("\n".join(f"{p}=={importlib.metadata.version(p)}" for p in packages)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei"],"axes.unicode_minus":False,"font.size":10})
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,height_ratios=[2,1])
    for key,color in [("M1_PRICE_FLOW","#b6453f"),("M2_PRICE_ONLY","#8068a6"),("R2_NEGATIVE_FLOW","#238975"),("BUY_HOLD","#45556c")]:
        ledger=pd.read_parquet(OUT/"evaluation/BASE"/(key+"_ledger.parquet"))
        nav=ledger.equity/200000
        axes[0].plot(ledger.date,nav,label=NAMES[key],color=color,lw=1.5)
        dd=nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1
        axes[1].plot(ledger.date,dd*100,color=color,lw=1.1)
    axes[0].set_title("510300 第十轮：加入原始季度申赎后，尚未达到夏普1.2",loc="left",pad=14,fontweight="bold")
    axes[0].set_ylabel("完整账户净值（初始1）")
    axes[0].legend(ncol=2,loc="upper left",frameon=False)
    axes[1].set_ylabel("回撤（%）")
    for ax in axes:
        ax.grid(alpha=.16)
        ax.spines[["top","right"]].set_visible(False)
    fig.text(.08,.012,"2020-01-02至2026-08-14开盘退出；20万元；基础费用；含分红、现金及交易约束。历史已多次观察。",fontsize=9,color="#555555")
    fig.tight_layout(rect=[0,.025,1,1])
    fig.savefig(DELIVERY/"第十轮_完整账户净值与回撤.png",dpi=145)
    plt.close(fig)
    facts=pd.read_parquet(SOURCE/"quarterly_share_flow_facts_v1_1.parquet")
    values=facts.net_units_divided_by_beginning_units*100
    fig,ax=plt.subplots(figsize=(12,4.7))
    ax.bar(np.arange(len(facts)),values,color=np.where(values>=0,"#bd6653","#278778"),width=.75)
    ax.axhline(0,color="#555555",lw=.6)
    ax.set_xticks(np.arange(0,len(facts),4),facts.period.iloc[::4],rotation=30)
    ax.set_ylabel("本季净份额变化 / 期初份额（%）")
    ax.set_title("原始季报份额需求：56个季度数值完整，55份有可用官方公布时钟",loc="left",pad=13,fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.16)
    fig.text(.085,.01,"按报告季度展示，仅用于数据说明；策略按公布后交易日使用。2013Q2数值展示、时钟不准入。份额比例不等于人民币资金流。",fontsize=9,color="#555555")
    fig.tight_layout(rect=[0,.035,1,1])
    fig.savefig(DELIVERY/"第十轮_原始季度净份额需求.png",dpi=145)
    plt.close(fig)
    print(json.dumps({"状态":"第十轮中文规则、经济框架、两张图和数值核对已保存","目录":str(DELIVERY)},ensure_ascii=False),flush=True)


def dependencies(paths):
    pending=list(paths)
    while pending:
        path=pending.pop()
        if path.suffix!=".py":
            continue
        tree=ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for name in names:
                if name and name.startswith(("research.","scripts.")):
                    dependency=ROOT/(name.replace(".","/")+".py")
                    if dependency.exists() and dependency not in paths:
                        paths.add(dependency)
                        pending.append(dependency)
    return paths


def package():
    assert not ZIP.exists(),"第十轮ZIP已存在，禁止覆盖"
    expected={r["path"]:r for r in read(MANIFEST)["files"]}
    paths={ROOT/name for name in expected}|{MANIFEST,Path(__file__),ROOT/"scripts/verify_original_quarterly_flow_saved_20260906.py"}
    for folder in [OUT,SOURCE]:
        paths.update(p for p in folder.rglob("*") if p.is_file())
    raw=physical(ROOT/"data/raw/510300_original_fund_subscription_reports_v1")
    paths.update(ROOT/"data/raw/510300_original_fund_subscription_reports_v1"/p.relative_to(raw) for p in raw.rglob("*") if p.is_file())
    config=read(CONFIG)
    paths.update(ROOT/name for name in config["inputs"].values())
    for cost in ["BASE","STRESS"]:
        paths.add(PARENT/"evaluation"/cost/"BUY_HOLD_ledger.parquet")
    for row in read(DELIVERY/"历史九轮结果摘要.json"):
        folder=(ROOT/row["result"]).parent
        paths.update(folder/name for name in ["metrics.csv","result.json"])
    for name in ["research","scripts"]:
        init=ROOT/name/"__init__.py"
        if init.exists():
            paths.add(init)
    paths=dependencies(paths)
    root_files={p.name:p for p in DELIVERY.iterdir() if p.is_file()}
    assert "FILE_INDEX.csv" not in root_files
    members={p.relative_to(ROOT).as_posix():p for p in paths}
    assert not set(members)&set(root_files)
    members.update(root_files)
    index=[]
    temporary=ZIP.with_suffix(".building.zip")
    assert not temporary.exists(),"临时ZIP已存在，先查明现有运行状态"
    with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(members.items()):
            content=physical(path).read_bytes() if path.is_relative_to(ROOT) and not path.is_relative_to(DELIVERY) else path.read_bytes()
            digest=hashlib.sha256(content).hexdigest()
            if name in expected:
                assert digest==expected[name]["sha256"],name
            archive.writestr(name,content)
            if name not in root_files:
                copied=DELIVERY/name
                copied.parent.mkdir(parents=True,exist_ok=True)
                copied.write_bytes(content)
            index.append({"path":name,"bytes":len(content),"sha256":digest})
        stream=io.StringIO(newline="")
        writer=csv.DictWriter(stream,fieldnames=["path","bytes","sha256"])
        writer.writeheader()
        writer.writerows(index)
        content=stream.getvalue().encode("utf-8-sig")
        archive.writestr("FILE_INDEX.csv",content)
        (DELIVERY/"FILE_INDEX.csv").write_bytes(content)
    with zipfile.ZipFile(temporary) as archive:
        names=archive.namelist()
        assert archive.testzip() is None and len(names)==len(set(names))
        assert set(names)=={r["path"] for r in index}|{"FILE_INDEX.csv"}
        for row in index:
            content=archive.read(row["path"])
            assert len(content)==row["bytes"] and hashlib.sha256(content).hexdigest()==row["sha256"]
    assert temporary.stat().st_size<512_000_000,"文件超过交付上限"
    temporary.replace(ZIP)
    receipt={"completed_at":now(),"zip":str(ZIP),"bytes":ZIP.stat().st_size,"sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
             "members":len(names),"index_members":len(index),"crc":"PASS","duplicate_members":"NONE","index_sizes_hashes":"PASS",
             "frozen_file_coverage":len(expected),"recomputation":read(DELIVERY/"保存结果核对.json"),
             "exclusions":"前九轮只包含历史摘要和保存指标，本轮不重算其原始模型和账户；完整旧证据见已交付各轮ZIP。",
             "security_audit_performed":False,"external_review_received":False,"uploaded":False}
    ZIP.with_suffix(".delivery.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="第十轮中文规则、经济框架和原始申赎审阅包")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--package",action="store_true")
    args=parser.parse_args()
    prepare() if args.prepare else package()
