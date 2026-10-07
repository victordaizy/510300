"""生成全研究综合报告、科学图表和有索引的单一GPT审阅包。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_all_research_abcd_increment_v1"
ZIP = ROOT / "deliverables/510300_全研究综合与ABCD增量实验_V1_GPT审阅_20260924.zip"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def put(name, content):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content.rstrip() + "\n", encoding="utf-8")


def js(name, obj):
    put(name, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def link(path, title="原始证据"):
    return f"[{title}](legacy_evidence/{path})"


def table(frame, labels, percent=(), decimals=()):
    lines = ["|" + "|".join(labels.values()) + "|", "|" + "|".join("---" for _ in labels) + "|"]
    for r in frame.to_dict("records"):
        cells = []
        for k in labels:
            v = r.get(k)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("未定义")
            elif k in percent:
                cells.append(f"{100 * v:.2f}%")
            elif k in decimals:
                cells.append(f"{v:.3f}")
            elif isinstance(v, float):
                cells.append(f"{v:,.2f}")
            else:
                cells.append(str(v).replace("|", "／"))
        lines.append("|" + "|".join(cells) + "|")
    return "\n".join(lines)


def figures():
    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    colors = {"A": "#1B4F72", "B": "#B27319", "C": "#177E70", "D": "#8C4C86", "BUY_HOLD": "#A8ABB0"}
    labels = {"A": "A 原形态", "B": "B 内部参与变化", "C": "C 日内隔夜补偿变化", "D": "D 联合", "BUY_HOLD": "510300买入持有"}
    target = OUT / "figures"
    target.mkdir(exist_ok=True)
    metrics = pd.read_csv(OUT / "results/account_comparison.csv")
    for period, title, filename in [("PRIMARY", "主区间：2021-01-04 至 2026-08-14", "主区间净值与回撤"), ("RECENT_DIAGNOSTIC", "近期诊断：2024-08-20 至 2026-08-14", "近期净值与回撤")]:
        fig, axes = plt.subplots(2, 1, figsize=(13.2, 8.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        fig.set_facecolor("#FAFBFC")
        for policy in colors:
            p = OUT / f"results/accounts/{period}/STRESS/{policy}"
            d = pd.read_parquet(p / "ledger.parquet")
            terminal = json.loads((p / "terminal.json").read_text(encoding="utf-8"))
            equity = d.equity_cny.to_numpy(float).copy()
            equity[-1] -= terminal["terminal_haircut_cny"]
            nav = equity / 200000
            dd = nav / np.maximum.accumulate(np.r_[1, nav])[1:] - 1
            dates = pd.to_datetime(d.date)
            width = 1.3 if policy == "BUY_HOLD" else 1.8
            axes[0].plot(dates, nav, color=colors[policy], lw=width, label=labels[policy], alpha=.8 if policy == "BUY_HOLD" else 1)
            axes[1].plot(dates, dd * 100, color=colors[policy], lw=width)
        for ax in axes:
            ax.grid(True, color="#DDE1E6", alpha=.6, linewidth=.5)
            ax.set_facecolor("white")
            ax.axvline(pd.Timestamp("2024-09-24"), color="#9AA1AA", ls=":", lw=1)
        axes[0].axhline(1, color="#596273", lw=.6)
        axes[0].set_ylabel("完整账户净值（初始=1）")
        axes[1].set_ylabel("回撤（%）")
        axes[0].legend(loc="upper left", ncol=2, frameon=False)
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=9))
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.suptitle("510300 形态与信息增量｜" + title, x=.08, ha="left", y=.985, fontsize=17, fontweight="bold")
        selected = metrics.query("period == @period and cost == 'STRESS' and policy in ['A','B','C','D']")
        summary = "　".join(f"{r.policy}：夏普 {r.net_sharpe:.3f}" for r in selected.itertuples())
        fig.text(.08, .934, summary, fontsize=11, color="#384655")
        fig.text(.08, .02, "压力成本；20万元；全部交易日与空仓日保留。B/C/D入场采用共同资料覆盖，缺失对照另列报告。\n虚线为2024-09-24。已见历史研究，独立前向样本为0；买入持有期末含卖出费用储备。", fontsize=9, color="#566373")
        fig.subplots_adjust(left=.08, right=.97, top=.90, bottom=.11, hspace=.13)
        fig.savefig(target / f"{filename}.png", dpi=180)
        fig.savefig(target / f"{filename}.svg")
        plt.close(fig)


def reports():
    m = pd.read_csv(OUT / "results/account_comparison.csv")
    coverage = json.loads((OUT / "inventory/coverage.json").read_text(encoding="utf-8"))
    supplementary = json.loads((OUT / "inventory/supplement_coverage.json").read_text(encoding="utf-8"))
    metrics = {"policy": "组别", "net_cagr": "净复合年化", "net_sharpe": "净夏普", "max_drawdown": "最大回撤", "completed_cycles": "完整周期", "average_exposure": "平均暴露", "net_profit_cny": "净利润（元）"}
    primary = table(m.query("period=='PRIMARY' and cost=='STRESS'"), metrics, ("net_cagr", "max_drawdown", "average_exposure"), ("net_sharpe",))
    base = table(m.query("period=='PRIMARY' and cost=='BASE'"), metrics, ("net_cagr", "max_drawdown", "average_exposure"), ("net_sharpe",))
    recent = table(m.query("period=='RECENT_DIAGNOSTIC' and cost=='STRESS'"), metrics, ("net_cagr", "max_drawdown", "average_exposure"), ("net_sharpe",))
    families = [
        ("趋势、反转与价格规则", "旧第23轮已检验12种明确进出场，包含趋势突破／反弹切换；主方案基础／压力夏普0.481／0.378。", "固定价格过程和可执行时钟可复用；双策略切换本身不再是新发现。", "reports/research/510300_sharpe_1_2_latest_research.json"),
        ("连续评分与自动仓位", "完整评分压力夏普0.351；静态对照归因的收益增量区间跨零。合法评分范围22.705—33.112，使20分退出不可达。", "先核对规则能产生什么行为，再谈优势；原连续评分模型拒绝保留。", "reports/research/510300_csp_v1_static_vs_timing_attribution_v1/研究报告.md"),
        ("旧85/15组合及风险预算", "第209轮有8个实际组合历史点值通过；50／60／70日局部稳健性失败，2026-09-17用户正式终止固定组合。", "保存真实历史过线事实；不把历史筛选结果当独立验证，不恢复SELECTED_MIX_BAND10_SIMPLE2。", "reports/research/510300_sharpe_1_2_latest_research.json"),
        ("稀疏机会与1.3档案重评", "214张指标表、2710行保存记录；14个原模型主时期两档成本达到旧1.3／10%回撤要求，较早压力夏普0.823—1.089。", "这些数值通过不包含现行新增年化10%条件；目标变更记录与原裁决分别保留。", "reports/research/510300_saved_candidates_sharpe13_v1/研究结论.md"),
        ("日内、隔夜与继续持有", "条件增量M1基础／压力夏普0.018／0.167，训练与评价方向反转；第107轮隔夜下行平方占比退出也未改善。", "损益归因可以描述承担何种风险；不能直接删掉隔夜或复活旧方向。", "reports/research/510300_intraday_overnight_increment_v1/REPORT.md"),
        ("固定核心、传播与衰竭", "旧驱动周期62个；传播M1有36成熟事件，Lift0.949；衰竭T1有56成熟事件，尾部Lift1.049；均冻结失败。", "这项已做正式检验，附件应补充；本轮B只检验形态起点后的固定群体变化，旧失败不变。", "reports/discovery/index_driver_episode_v1_result.md"),
        ("成分广度与内部风险", "多数成分上涨＋20日趋势基础夏普-0.100，相同价格对照-0.263；有相对改善但整体无可用证据。", "内部参与面不是自动买入门；点时成员和四态收益处理仍是可复用资料。", "docs/510300_BREADTH_MAJORITY_TREND_V1.md"),
        ("盈利、EPS与增长分歧", "增长分歧33成熟评价原点，MSE恶化5.97%，账户NOT_RUN；原财报／研报口径、覆盖、股本和报告年龄修正保留。", "保留来源和经济事实；拒绝此60日预测用途，不能借综合工作更改裁决。", "reports/research/510300_eps_growth_disagreement_increment_v1/研究结论与下一步.md"),
        ("估值与会计测量", "慢估值先验、PB—ROE、前瞻盈利收益率、CF／DR／RC分别有数据或预测阶段记录；口径不同，不能把便宜当即刻买点。", "查看对应终态及测量定义；归因恒等式成立只证明账目可解释。", "CONTEXT.md"),
        ("逆回购与公募季度申赎", "四项保存预测配对没有确认一致增量；20日和60日标签与披露日不同。", "公开操作量不等于每日实际净投放；披露份额申赎不等于全市场公募净现金流。", "docs/510300_SAVED_RETURN_INFORMATION_REVIEW_V1.md"),
        ("M1／M2状态和预期差", "旧口径月频Δ3剪刀差MSE恶化1.0173%；旧口径五日预期差恶化4.983%；新版后续已实际训练，10次评价B对A再恶化4.99%。", "不能停留在旧的样本门不足叙述；最新训练失败和口径断点一并保留。", "reports/research/510300_new_m1_consensus_training_v1/训练结果.md"),
        ("增长、财政与利率传导", "新订单20日收益MSE恶化12.47%；财政收益MSE点值改善0.27%仍未通过；LPR点值改善0.74%但区间跨零，股债增量未建立。", "各机制分开；预测未通过的账户记NOT_RUN，不补成0收益。", "reports/research/510300_growth_state_increment_20d_v1/研究报告.md"),
        ("宏观消息与政策时钟", "NBS固定五分钟B1严格前序MSE比B0高7.82%；政策目录、宣布／细则／实施和用途已有独立整理。", "新消息是重新评估的触发因素，政策目录数量不是可交易事件数或入市金额。", "docs/510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md"),
        ("大资金身份、意向与承接", "一月份额下降与持有人资料是联合历史证据；私募意向加空间曾有MSE点值改善，但实际压力账户夏普-0.329。", "实名持仓报告放验证层；意向、调查仓位、份额、成交额和真实订单流分开。", "reports/research/510300_private_manager_exploratory_training_v1/训练结果.md"),
        ("IF与少数高胜率节点", "旧总持仓增量拒绝；新耗竭既有数据训练压力夏普约0.348；后续概率幅度扩点30笔夏普0.371，节点校准后26笔夏普0.304。", "高胜率、模型MSE改善和高盈亏比不能代替全账户夏普；旧频率配额现已取消。", "reports/research/510300_probability_payoff_progress_20260924/训练进展与结果.md"),
        ("AH、EPU与解禁", "三个明确增量研究均有冻结拒绝；部分已运行完整账户，不能统称资料未齐或未测。", "保留各自失败证据，不能仅因名称属于估值／宏观／供给就自动加权。", "reports/research/510300_unlock_announcement_increment_v1/研究结论与下一步.md"),
        ("下行风险、压力传递与波动预算", "风险预测门通过、账户未改善的情况存在；成分脆弱性DSV5增量G3未通过，压力危险率后续实际训练近常数。", "风险预测、收益预测和仓位效用各自评价；减小波动不自动提高净收益质量。", "docs/510300_CONSTITUENT_FRAGILITY_DSV5_INCREMENT_V1_APPEND_ONLY_CLOSURE_20260905.md"),
        ("PCF、IOPV与价格让步", "存在来源一致性和可用性成果，也存在同日回执／估值／成交缺口；目前不用新增市场采集。", "月度份额、PCF、盘后信息、期货持仓和订单流不是同一资金变量；无合格时点继续NO_VIEW。", "reports/research/510300_close_concession_source_feasibility_v1"),
        ("期权与跨资产相关研究", "旧期权信息可作为510300辅助研究；近期另有期权独立账户授权文件，且资金和收益目标不同。", "纳入相关研究索引，但本次A/B/C/D只执行510300与现金，不拼接期权、其他ETF或多资产收益。", "config/510300_options_100k_doubling_mandate_v2.json"),
        ("最新连续形态V1/V2与后续V3", "V1近期压力夏普1.185但单笔占75.51%；V2较长压力夏普0.491，启停放宽仍未建立。V3只有协议、输入和代码，无完成结果。", "保留三形态作为固定观察基准；V3标未完成，本次没有替另一个版本补跑或假定成功。", "reports/research/510300_sequential_patterns_2021_2026_v2/研究结论.md"),
    ]
    family_lines = ["|研究家族|已经知道什么|本次综合后的用途|证据|", "|---|---|---|---|"]
    for title, result, use, source in families:
        if (OUT / "legacy_evidence" / source).is_dir():
            files = sorted((OUT / "legacy_evidence" / source).glob("*.md"))
            source = files[0].relative_to(OUT / "legacy_evidence").as_posix() if files else source
        family_lines.append(f"|{title}|{result}|{use}|{link(source)}|")
    put("02_全研究综合报告.md", f"""# 510300全研究综合与A/B/C/D增量实验

**综合现有研究并完成本轮实验后，尚没有证据支持把“形态＋内部持续参与＋日内／隔夜补偿”升级为有效交易系统。** 主区间2021-01-04至2026-08-14，压力夏普A/B/C/D为0.495/0.343/0.432/0.420，年化分别3.72%/1.93%/2.66%/2.40%，均未达到现行年化10%、夏普1.2目标。C有近期和相同资料覆盖下的局部改善，但不确定性区间跨零，收益仍集中于同一轮2024年行情；B和D没有证明稳定优势。本轮完成的是全研究综合及固定增量实验，长期收益目标仍未实现。

## 1. 本次“全部研究”的覆盖范围

本地原始编号研究索引完整包含216轮；另扫描{coverage['study_directories']}个相关一级报告目录，并对文件名及正文作补充定位，形成{supplementary['unique_indexed_files']}条唯一文件索引。包内保存{coverage['copied_evidence_files'] + supplementary['additional_files']}份原文／配置／结果快照，可按路径、家族和原始状态检索。**目录、文件、参数、账户、成本情景和独立实验不是同一个计数。** 75个目录没有在固定顶层字段中提取到终态，不能直接解释为75个未完成研究或75个失败实验。

核心机制、相关代码和本次直接输入已经逐项阅读和核对；其余材料保留完整索引与文本快照，不宣称逐行人工复审或全仓库重跑。旧账户的全部逐日路径、原始PDF与海量原始行情没有为本次综合重复打包；本次实验的直接输入、代码、全部逐日账户、冻结记录和统计抽样则完整保留。范围见[inventory/coverage.json](inventory/coverage.json)、[补充覆盖](inventory/supplement_coverage.json)、[216轮索引](inventory/legacy_216_rounds.csv)和[文件索引](inventory/artifact_inventory.csv)。

## 2. 当前目标与历史口径

2026-09-24最新本地权威配置及本次附件一致：20万元，只使用510300.SH和CASH_CNY，成本后复合年化至少10%、净夏普至少1.2、最大回撤风险目标10%，年度交易次数没有下限；不用新增市场采集。本轮沿用最新形态的252交易日年化。较早许多研究使用242日，并有不同期末估值、现金收益和费用设置；本报告保留其原值，不能将表格拼成同口径排行榜。

1.5→1.3→1.2是不同时间的要求演变；“逐年最少5笔”已取消。旧14模型在1.3／回撤10%下的过线事实仍保留，但不能自动满足现行新增年化10%条件。SELECTED_MIX_BAND10_SIMPLE2在2026-09-17被用户终止，其历史过线和终止同时成立。{link('config/510300_existing_data_training_mandate_v1.json', '本轮保存的当前约束')}；{link('reports/research/510300_sharpe_1_2_latest_research.json', '旧研究与终止原始索引')}。

## 3. 对用户附件的两项实质修正

**第一，固定核心驱动、身份持续、传播和衰竭早已做过正式检验。** 旧驱动周期M1“传播后延续”36个成熟事件，MIDDLE20 Lift=0.949，95%区间0.558—1.396；中位相对收益为-0.366%，尾部发生率13.89%，高于基准10.21%。T1“衰竭后尾部”56个成熟事件，TAIL20 Lift=1.049，区间0—2.099。两项均HISTORICAL_REJECTED_FROZEN。因此不能只拿CONTEXT.md的定义，把这套机制说成主要未探索方向。{link('reports/discovery/index_driver_episode_v1_result.md')}。

**第二，部分“数据不足”的分支后来已经得到用户放宽样本要求并实际训练。** 新版M1、LPR预期、私募意向、IF耗竭、压力危险率和概率幅度节点都有更晚记录。综合结论必须同时保留早期数据门与后续实际结果，不能用较早NO_VIEW抹去后来训练，也不能用后来的训练抹去首版来源和小样本限制。

## 4. 全研究家族的综合裁决

{chr(10).join(family_lines)}

这里的“保留”主要指保留事实、代码、来源、失败边界或冻结观察基准，不代表保留一个可交易买点。对数据阻断、预测失败、账户失败、历史点值通过但稳健性失败、未完成方案分别记录；NO_VIEW、NOT_RUN、NOT_COMPUTED和零交易导致的夏普未定义保持原义。

## 5. 本轮实际执行的四组

|组别|固定定义|允许改变的决策|
|---|---|---|
|A|原V1/V2的PATTERN_ONLY，三类形态、优先顺序和账户规则不变|原有确认入场、形态失效和五日退出|
|B|形态准备日按此前5日正贡献固定至多3个行业，并固定该日成员与权重；比较同一群体3日上涨参与率相对准备日的变化|变化非负时准入；持有后已知变负，下一可卖开盘退出|
|C|I=(收盘−开盘)/前收盘；O=(开盘−前收盘＋应得分红)/前收盘；K=最近3日平均(I−max(−O,0))，比较K相对形态准备日变化|变化非负时准入；持有后已知变负，下一可卖开盘退出|
|D|B和C同时非负，只有这一项预定逻辑交互|任一已知变负则退出|

B的权重用于描述准备日固定群体的参与率，不是中证官方每日指数贡献。没有用当天冠军更换群体，没有剔除消失证券后重新归一化。C的I+O严格等于含分红的当日简单总收益，但没有把隔夜利润从账户里删除。股票ETF保留T+1，当天新买份额即使收盘提示退出也必须承担第一晚；[上交所说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)已核对。

本次新增用途与旧研究有明确边界：锚点是已冻结的价格形态准备日，判断的是后续同群体参与和补偿的变化，目标为原短持有账户的增量；不是重跑原20日传播/衰竭、静态广度20日门、D20/D60方向或隔夜平方风险占比。3/5日窗沿用既有短周期量级并在本轮结果前固定，没有搜索。完整协议和哈希见[protocol.json](protocol.json)及[freeze.json](freeze.json)。

## 6. 数据覆盖对比较的实质影响

共同截止日为2026-08-14，因为内部归因源只到该日；价格源到9月16日的尾段没有单边混入四组。本轮主要区间42次确认中，两变量共同可用只有23次。19次缺失包括准备日前五个连续归因日不足、无正贡献核心、或确认时固定成员覆盖不足等原因。原形态A完成40周期；资料可用性对照A_COVERAGE完成22周期。

B/C/D采用同一个入场资料范围，A保留原基准；A_COVERAGE只加资料可用性，不使用变量符号，是判断纯新增信息不可少的比较。**C在本轮也受共同内部资料范围约束，不能把其结果称为“仅凭全部价格日期的C策略”。** 持有期间缺值本身不触发卖出。输入均为现有历史重建；月度snapshot_date和行业有效区间不等于已认证的历史首次公布时钟。同日月末快照已排除，但后续首次真正可得时间仍未被认证。因此本轮结果只支持探索性历史比较。

## 7. 主区间完整账户结果

主区间2021-01-04至2026-08-14，共1361个交易日，20万元本金，包含所有空仓日。基础佣金万二、滑点5基点；压力万四、10基点，均最低5元，按0.001元不利取整、100份整手、次日可卖。期末另计卖出费用储备。每个方案都是单一资金账户，三个形态不分别满仓相加。

压力成本：

{primary}

基础成本：

{base}

BUY_HOLD的完整周期为0是因为期末仍持有，不等于没有投资；CASH的夏普未定义。A与母版同区间逐日账本完全一致。28个账户包括四组、资料可用性对照、买入持有和现金，两个成本、两个固定时期；另有4次A母版等价重放用于核对，不算新策略。没有策略训练或参数网格。

![主区间完整逐日净值与回撤](figures/主区间净值与回撤.png)

## 8. 进入与继续持有，实际各改变了什么

在主时期压力成本下，A接受40次入场，2个确认因资金已占用或优先级未进入。共同可用性先排除19个确认。B再按参与变化排除7个、实际16次买入，持有中8次因新增信息提前退出；C仅再排除2个，另1个资金冲突，实际20次买入，2次新增信息退出；D再排除8个，实际15次买入，8次新增信息退出。完整原因见[决策归因](results/decision_attribution.csv)。

因此不能把B的改善或恶化只归因于“止盈更早”，也不能把C的结果只归因于选入场。这里是逐项决策变化记录，不是新做一批进入／退出消融，更不把改变后的利润差分解成可相加的独立因果效应。

## 9. 近期线索与最大的脆弱性

近期诊断固定2024-08-20至2026-08-14，共481个交易日：

{recent}

C近期两档成本的年化、夏普和回撤点值达到目标，但压力账户只有7个完整周期，最大一笔占净利润84.34%，前三笔占113.48%。主时期C最大一笔占134.00%；除该笔外的其余完整周期净利润合计为-10,362.75元。这是利润归因，不是把该笔删掉后制造一条新策略。

主时期最大一笔占比：A95.15%、B194.52%、C134.00%、D155.37%。超过100%表示其余周期合计亏损。因此B/D虽然降低回撤和暴露，也未建立分散的利润来源。C的局部改进值得如实记录，但不能从这条已经看过的2024年路径中选择新参数。

本轮近期A压力夏普1.213，高于原截至9月16日的1.185，**只是统一截止日后少了23个空仓交易日，净利润同为58,239.35元**。这不构成新收益或新独立验证。四组使用共同截止日是数据约束，不能拿这次跨过1.2来宣布原形态已被证明成功。

![近期完整逐日净值与回撤](figures/近期净值与回撤.png)

## 10. 增量、不确定性和价格控制

主时期压力B−A、C−A、D−A的年化算术日均收益差分别为-1.874、-1.135、-1.424个百分点。相同20日移动区块、2000次联合抽样，对3个主要候选使用单侧0.05/3校正，收益差和夏普差区间均跨零，没有候选通过预定历史增量判据。抽样索引完整保留；这是有限已见历史的不确定性描述，不校正全部既往搜索，也不改变独立前向为0的事实。

相对A_COVERAGE，C的年化算术日均增量为+0.434个百分点，但90%双侧区间约[-0.934,+2.069]个百分点；D为+0.145个百分点，区间亦跨零。不能将覆盖减少产生的变化记作新信息的功劳。

按全期平均暴露缩放A后，三组有正的点值差；按同波动缩放A，主区间三组均没有正的点值差。二者都是事后线性归因，未重新模拟整手、最低佣金和资金路径，不是可执行策略。详见[暴露归因](results/exposure_attribution.csv)。

在22个共同可用且已成熟的形态标签上，固定控制trend_z、log(rv20)和形态类型后，B与C的样本内残差相关分别0.096和0.635。这个小样本相关性受共同大行情、事件选择和重叠影响，未作为交易门或预测资格，也没有训练成新账户。它与完整账户没有稳定增量可以同时成立。对价格变化和波动的控制在这里是诊断性投影，不声称因果识别。这里实际作了3个样本内最小二乘投影，策略拟合仍为0。

大量已见历史试验会使最佳夏普偏乐观，这是[Bailey与López de Prado的DSR研究](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)讨论的问题。本次没有以8433个文件数、216个轮次数替代有效独立试验数，也未伪造一个全项目DSR值。

## 11. 综合后的研究取舍

1. **固定形态保留为共同观察和对照入口。** 其近期收益路径与长期局限都已量化，原三形态、失败／超时案例和账户冲突应继续完整保留；不把近期点值过线当通过。
2. **内部结构回到解释与待验证信息层。** 原传播／衰竭和本轮B均未支持自动准入或退出，停止本版阈值优化。现有点时成员、贡献事实和固定群体记录可复用。
3. **日内／隔夜补偿保留为局部线索，但本版C冻结不晋升。** 它相对共同覆盖对照略有改善，主时期证据仍不足且集中于一笔。后续若有新的独立样本，应评估同一固定定义；不能立即改窗、换符号、删大赚周期或搜更漂亮止盈。
4. **宏观、盈利、政策和资金按已经完成的具体用途分别裁决。** 有来源不等于有预测增量；某个用途失败也不等于该经济资料毫无价值。尚未完成的V3、政策完整母集和真实资金行为链按原状态保留。
5. **下一轮优先得到新增独立信息，而非增加历史自由度。** 当前不用采集，本轮已经把现有材料能支持的固定四组比较做完；本地新增合格资料出现后，可以沿用冻结版本记录。若要提出完全新机制，应先说明与此次及旧实验的实质差异、预先规定对照和停止条件。本报告没有自动启动下一套回测或等待任务。

## 12. 验证与最终状态

冻结前6项人工机制检查通过；4次真实A母版账本等价；28个账户共25,788行保存账目复算通过；3个历史截断点的价格特征与形态确认一致。另以独立代码复算130,195行信号上下文、1,157行固定成员、590个入场决策、112个内部特征前缀点，最大误差6.67e-16。对应[saved_verification.json](saved_verification.json)与[context_verification.json](context_verification.json)。这些验证证明实现与保存资料相符，不建立来源首发认证、外部审阅或真实交易成交。

本轮B/C/D固定用途终态：FROZEN_NO_RELIABLE_ABCD_INCREMENT。当前共同历史截止2026-08-14，价格资料另到2026-09-16，均不是9月24日当天判断。当前NO_VIEW；长期收益目标未达；本轮综合与有限实验完成。没有新增市场行情下载、策略训练、独立前向样本或订单。
""")
    # 用实际保存的天数覆盖说明，避免手写日期行数误差。
    report = OUT / "02_全研究综合报告.md"
    content = report.read_text(encoding="utf-8")
    primary_days = int(m.query("period=='PRIMARY' and policy=='A'").days.iloc[0])
    recent_days = int(m.query("period=='RECENT_DIAGNOSTIC' and policy=='A'").days.iloc[0])
    content = content.replace("共1361个交易日", f"共{primary_days}个交易日").replace("共481个交易日", f"共{recent_days}个交易日")
    content = content.replace("少了23个空仓交易日", f"少了{504-recent_days}个空仓交易日")
    report.write_text(content, encoding="utf-8")
    put("03_与旧研究的逐项去重.md", """# 本轮定义与旧研究的实质差异

|旧研究|已做用途与原裁决|本轮差异|保留边界|
|---|---|---|---|
|INDEX_DRIVER_EPISODE_DISCOVERY_V1|独立驱动周期确认后20日传播延续、衰竭尾部；两项失败|用既有价格形态setup固定群体，比较确认及短持有期间相对setup的参与变化|不重跑原M1/T1，不宣称该机制未做过|
|BREADTH_MAJORITY_TREND_V1|整个当日成员20日上涨家数超过一半＋均线；总体失败|固定setup成员权重，3日参与变化相对固定起点，而非整体多数水平|月度快照归因不是官方逐日精确贡献|
|INTRADAY_OVERNIGHT_INCREMENT_V1、SIMPLE_SESSION_DIVERGENCE_V1|D20/D60水平／方向、训练预测或信号仓位；既有失败保留|C是短形态内“日内补偿减隔夜损失”的相对起点变化，无训练参数|不反向救D20，不删除所有隔夜|
|OVERNIGHT_DOWNSIDE_EXIT_V1，第107轮|20日隔夜负收益平方占全部日内隔夜平方，学习退出；未改善|本轮不用平方风险占比或学习退出，用预定0变化阈值|旧模型与窗口失败仍成立|
|RANGE_OVERNIGHT_RISK_V1，第146轮|区间＋隔夜风险乘数缩仓；回撤降但夏普退步|本轮不改连续风险预算，只有准入和下一开盘退出|不把缩小风险自动叫信息增量|
|SEQUENTIAL_PATTERNS_REGIME_V1／V2|三形态与6/4、3/2启停；失败|A只复用已冻结PATTERN_ONLY；B/C/D是有限信息比较|不修改形态窗、五日持有、失效和成本|
|PATTERN_DAILY_STATE_LEARNING_V3|普通日状态学习方案，只有冻结文件，暂无结果|本轮无该训练，没有宣称其已失败或成功|保持V3未完成，不替原版本续跑|

本次查重覆盖旧编号索引、上述直接代码、CONTEXT定义和本轮文档快照。新组合不等于新经济规律；本轮未发现完全相同已完成表示，但不宣称所有别名已被形式化证明不同。
""")
    put("04_当前裁决与后续优先级.md", """# 当前裁决与后续优先级

本轮全研究综合和有限A/B/C/D实验已完成；20万元、净年化10%、净夏普1.2的长期策略目标仍未实现。

|对象|当前裁决|允许复用的内容|停止条件|
|---|---|---|---|
|A原形态|固定观察对照；主时期未达标|原过程、确认、失败案例与账本|不得仅依赖改截止日后的近期1.213晋升|
|B内部参与|冻结：无可靠增量|固定群体描述和成员数据|不改核心数量、3/5日窗、覆盖门或方向救回|
|C补偿变化|冻结不晋升；保留局部线索|同定义未来观察与独立检验所需代码|不以22个样本相关0.635或近期7笔代替验证|
|D联合|冻结：无可靠增量|共同交互及失败记录|不改权重、加第三因素或更换成功窗口|
|既有85/15|用户终止继续有效|历史结果及稳健性反证|不恢复按日续算、D60或调参|
|其他宏观／盈利／资金研究|逐项继承原裁决及后续修订|来源、数据时钟、已保存预测和失败原因|禁止以本轮综合自动晋升|

优先级1：保留本轮全部对照和不利结果，完成外部可读交付。优先级2：后续本地出现实际新资料时，使用事前冻结版本形成新增独立记录；现阶段保持不用采集。优先级3：仅对确有新增信息或清晰新决策问题的研究另立协议，先说明旧用途为什么不同；未运行项目仍记NOT_RUN。

没有为“完成本轮”要求策略必须成功；也没有把完成报告当作已实现收益目标。下一研究不在本轮自动启动。
""")
    put("00_README_FIRST.md", f"""# 先读这里：510300全研究综合与A/B/C/D增量实验

本轮已完成综合与实验。主要结论：旧固定核心传播／衰竭已失败，本轮新锚定定义也未建立稳定增量；四组主时期压力夏普0.495/0.343/0.432/0.420。长期年化10%／夏普1.2目标未实现。

建议阅读顺序：

1. [全研究综合报告](02_全研究综合报告.md)：全部家族、关键遗漏修正、账户和局限。
2. [用户请求](用户请求.md)与[附件原文](用户附件原文.txt)：本次工作来源。
3. [逐项去重](03_与旧研究的逐项去重.md)、[协议](protocol.json)、[冻结](freeze.json)。
4. [全部账户指标](results/account_comparison.csv)、[增量区间](results/paired_increment.csv)、[逐年收益](results/annual.csv)、[决策归因](results/decision_attribution.csv)。
5. [旧216轮索引](inventory/legacy_216_rounds.csv)、[报告目录](inventory/study_directory_inventory.csv)、[8433项原文件定位](inventory/artifact_inventory.csv)及[正文补充定位](inventory/related_text_supplement.csv)。
6. [下一步与停止条件](04_当前裁决与后续优先级.md)、[审阅提示词](01_GPT_REVIEW_PROMPT.md)。

legacy_evidence保存{coverage['copied_evidence_files'] + supplementary['additional_files']}份原研究文本、配置和结果，不等于完整复制旧原始行情及全部旧模型。inputs与results是本次直接研究快照；FILE_INDEX.csv记录包内全部文件的大小及SHA-256，索引自身除外。当前来源首发局限、旧失败和未完成V3均保留。

本轮没有上传给GPT或完成外部审阅。包内材料用于研究复核，不能当作当前仓位信号。
""")
    put("01_GPT_REVIEW_PROMPT.md", """# 可直接复制的GPT审阅提示词

请审阅本包的510300全研究综合及已完成A/B/C/D增量实验。先阅读00_README_FIRST.md和02_全研究综合报告.md，再核对protocol.json、freeze.json、results及legacy_evidence。用户仅允许510300.SH与现金，20万元，成本后年化10%、夏普1.2、回撤风险目标10%，取消年度最低次数，不新增市场采集。

请特别核对：

1. 是否准确纳入原216轮及后续盈利、宏观、资金、期权信息和形态研究；是否遗漏了已失败的固定核心传播／衰竭；是否把较早数据不足误当成后来仍未训练。
2. B的固定群体、月度权重和首发时间局限是否足以限制结论；42确认只有23共同可用时，A与A_COVERAGE对照是否被正确解释；C也受共同覆盖约束是否清楚。
3. 进入与持有两种决策、T+1、第一晚风险、分红权利及到账、整手、费用、全部空仓日、期末储备是否一致。
4. 原形态近期1.185变成本轮1.213是否只是少了23个空仓日；是否把C近期7笔、最大一笔84.34%当成稳定证据。
5. 2000次20日联合区块、三候选校正、同暴露／同波动理想归因和22个样本残差相关的用途有没有被夸大；请不要拿文件数当有效独立试验数。
6. 当前“无可靠增量”的裁决是否与保存数据一致；指出可验证的实现错误、证据不足或替代解释，并给出严重程度及文件路径。

最后请给出下一步优先级、最小必要验证、停止条件和应当放弃的重复路线。保留原失败、NO_VIEW、NOT_RUN、未定义夏普与已终止85/15；不通过换名称、调窗口、删大行情或挑最好区间救回。未完成V3不可当作有结果。若建议新实验，请逐项说明相对旧研究的新信息或新用途。本包不是外部审阅已通过，也不授权订单、券商或实盘。
""")
    put("EXCLUSIONS.md", """# 范围与排除

包内全量保存本次直接输入、代码、28个账户逐日账本／交易／决策、统计抽样、验证记录与冻结文件。旧研究保存可检索的原始文档／配置／结果快照和文件索引。旧包解压副本不重复计数；旧全量模型、大体积原始行情、完整研报PDF库、环境、缓存和日志没有重复搬入，因此本包不声称完整重跑全部历史项目。

本地来源首发和不可变事前采集未获新增认证；数据截止不同分别标明。本轮市场下载为0。为核对规则与方法，仅浏览上交所ETF交易说明及DSR论文，不属于新增市场数据。没有外部上传或外部GPT审阅。

相关期权和其他资产研究仅进入索引和背景，其账户不进入本次510300单账户收益。75个目录缺顶层可提取终态，只表示未提取，不自动裁决失败或完成。

ZIP仅作CRC、重复路径、索引／大小／哈希和解压保存结果复核；未开展额外安全审计。
""")
    put("复算说明.md", """# 解压后的复核

使用Python及requirements.txt所列依赖。在解压目录以PowerShell运行本包code/all_research_abcd_increment_v1.py的verify子命令，并显式传入--root为解压目录；再运行code/verify_510300_abcd_saved_context_v1.py，同样传入--root。

verify只复算保存账目、指标、冻结哈希及形态截断，不新建账户或拟合。第二个验证器独立复算固定成员、变量和已保存入场决策。需要完全重建实验时，应另复制一份独立工作目录并保留原冻结和运行记录，不能删除RUN_STARTED.json在原目录反复筛选。

code/parent_engine.py为母版引擎快照。protocol.json与freeze.json是本轮运行前的协议和身份。机器可读结果的完整定义见协议；历史说明中的242日与本轮252日年化不混同。
""")
    disposition = {"study_id": "510300_ALL_RESEARCH_ABCD_INCREMENT_V1", "status": "FROZEN_NO_RELIABLE_ABCD_INCREMENT", "research_synthesis_and_abcd_completed": True, "strategy_goal_achieved": False,
        "primary_two_cost_numerical_pass": [], "historical_increment_supported": [], "C_recent_point_pass": True,
        "C_recent_point_pass_is_independent_validation": False, "strict_forward_observations": 0,
        "current_market_view": "NO_VIEW", "selected_strategy_revived": False, "next_experiment_started": False,
        "frozen_after_result_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    js("research_disposition.json", disposition)
    js("external_references.json", {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "sources": [
        {"title": "上交所ETF是否实施T+0交易", "url": "https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml", "finding": "股票ETF采用T+1。"},
        {"title": "Bailey and Lopez de Prado 2014 The Deflated Sharpe Ratio", "url": "https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf", "finding": "多重试验、选择偏差与非正态会影响报告夏普；本轮未计算全项目DSR。"}]})
    put("requirements.txt", "numpy\npandas\npyarrow\nmatplotlib")
    js("runtime.json", {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__})
    for filename in ["verify_510300_abcd_saved_context_v1.py", Path(__file__).name]:
        shutil.copy2(ROOT / "scripts" / filename, OUT / "code" / filename)
    # 本次输入源补充说明不更改已冻结输入。
    js("source_supplement.json", {"parent_event_labels": {"source": "reports/research/510300_sequential_patterns_regime_v1/results/event_labels.parquet", "snapshot": "inputs/parent_event_labels.parquet", "sha256": sha(OUT / "inputs/parent_event_labels.parquet")}, "code_snapshot": "freeze.json记录运行前代码；交付和独立复算代码另由FILE_INDEX覆盖。"})


def package():
    if ZIP.exists():
        raise RuntimeError("最终ZIP已存在，禁止覆盖已交付快照。")
    members = [p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "FILE_INDEX.csv"]
    records = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(members)]
    with (OUT / "FILE_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(records)
    temporary = ZIP.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for r in records:
            z.write(OUT / r["path"], r["path"])
        z.write(OUT / "FILE_INDEX.csv", "FILE_INDEX.csv")
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(records) + 1
        assert set(z.namelist()) == {r["path"] for r in records} | {"FILE_INDEX.csv"}
        for r in records:
            b = z.read(r["path"])
            assert len(b) == r["bytes"] and hashlib.sha256(b).hexdigest() == r["sha256"]
    assert temporary.stat().st_size < 512_000_000
    temporary.replace(ZIP)
    receipt = {"status": "PASS_ZIP_CRC_UNIQUE_PATHS_INDEX_SIZES_HASHES", "zip": str(ZIP), "bytes": ZIP.stat().st_size,
               "members": len(records) + 1, "indexed_files": len(records), "sha256": sha(ZIP), "external_review": False,
               "saved_account_verification": json.loads((OUT / "saved_verification.json").read_text(encoding="utf-8")),
               "saved_context_verification": json.loads((OUT / "context_verification.json").read_text(encoding="utf-8"))}
    ZIP.with_suffix(".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["saved_account_verification", "saved_context_verification"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    reports()
    figures()
    if "--package" in sys.argv:
        package()
    else:
        print("报告与图表已生成，等待实际查看后封装。", flush=True)
