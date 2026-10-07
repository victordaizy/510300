"""汇总T10固定结果、T01/T09准入边界及96因子库累计进度。"""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.factor96_crossborder_absorption_v1 import OUT, STUDY, now, save

PROGRAM = ROOT / "reports/research/510300_factor96_program_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value, digits=4):
    return "未定义（全现金）" if pd.isna(value) else f"{value:.{digits}f}"


def figure():
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    styles = [("PRICE_COMMON", "仅首日收涨", "#a8aeb8", "--"),
              ("POSITIVE", "正外盘对照", "#bc8633", "-"),
              ("FULL", "主方案：首日反应不足", "#287b94", "-"),
              ("OVERREACTION", "反应过强对照", "#ad5361", ":")]
    for policy, label, color, style in styles:
        z = pd.read_parquet(OUT / f"accounts/MAIN/200000/STRESS/LAG0/{policy}/ledger.parquet")
        dates = pd.concat([pd.Series([z.date.iloc[0]-pd.Timedelta(days=1)]), z.date], ignore_index=True)
        nav = np.r_[1., z.equity/200000]
        axes[0].plot(dates, nav, label=label, color=color, ls=style, lw=1.65)
        axes[1].plot(dates, (nav/np.maximum.accumulate(nav)-1)*100, color=color, ls=style, lw=1.65)
    axes[0].set_title("T10 跨境信息缓慢吸收：唯一主期交易亏损，目标未达成", fontproperties=font, fontsize=17, loc="left", pad=58)
    axes[0].legend(prop=font, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    axes[0].set_ylabel("完整账户净值", fontproperties=font)
    axes[1].set_ylabel("自账户峰值回撤（%）", fontproperties=font)
    axes[1].set_xlabel("2021—2025年；完整保留现金日、费用及分红", fontproperties=font)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.08, .025, "20万元 · 每边万4、最低5元、10bp滑点 · 主方案绝大多数时间空仓；低回撤不证明有效", fontproperties=font, fontsize=10, color="#555555")
    fig.tight_layout(rect=[.02, .055, .99, .99])
    fig.savefig(OUT / "主期完整账户净值与回撤.png", dpi=160)
    plt.close(fig)


def main():
    result = read(OUT / "result.json")
    verified = read(OUT / "saved_verification_receipt.json")
    assert verified["ledgers"] == 48 and not result["historical_joint_point_pass"]
    metrics = pd.read_csv(OUT / "metrics.csv")
    main_rows = metrics[(metrics.period == "MAIN") & (metrics.cost == "STRESS") & (metrics.lag == 0)]
    primary = main_rows[main_rows.policy.eq("FULL")].set_index("capital")
    labels = {"PRICE_COMMON": "仅首日收涨，同覆盖", "POSITIVE": "再要求海外上涨", "FULL": "主方案：再要求反应不足", "OVERREACTION": "竞争对照：反应过强"}
    table = []
    for row in main_rows[main_rows.capital.eq(200000)].itertuples():
        table.append(f"|{labels[row.policy]}|{number(row.net_sharpe)}|{row.cagr:.3%}|{row.max_drawdown:.3%}|{row.closed_cycles}|")
    delayed = metrics[(metrics.period == "MAIN") & (metrics.capital == 200000) & (metrics.cost == "STRESS") & (metrics.lag == 1) & (metrics.policy == "FULL")].iloc[0]
    first_increment = result["increments"][0]
    report = f"""# T10跨境信息缓慢吸收：固定检验结果

**只操作510300、完整账户成本后夏普至少1.2的目标仍未实现。** T10在2021—2025年、20万元压力成本账户中，净夏普{primary.loc[200000,'net_sharpe']:.6f}，净年化{primary.loc[200000,'cagr']:.3%}，最大回撤{primary.loc[200000,'max_drawdown']:.3%}。只有一个完整交易周期，亏损{abs(primary.loc[200000,'net_profit']):,.2f}元。2万元账户夏普{primary.loc[20000,'net_sharpe']:.6f}，同样未达标。早期2017—2020年没有合格主信号，夏普未定义，不能记为0或通过。

候选库累计已完成T03、T05、T14、T02、T10共5个固定主问题，合格策略0。正式账户情景累计208个，另保留T02原实现的40个已否定账户，实际执行累计248个。情景数包含本金、费用、时期、对照与时钟敏感性，不是独立策略数；96项因子和18项草案仍不能说成已经全测。

## 这次测试的新增问题

旧美国中国ETF八规则研究在A股开盘前按外盘符号或分位决定全仓/现金，已固定失败；旧全球信息分类器也未达标。本轮先让A股完成一天反应，再在同一来源覆盖下比较：仅首日收涨、再要求正外盘、进一步要求反应不足，以及反应过强对照。主要增量FULL减POSITIVE，单独询问“反应不足”这项条件有没有增加信息。

选择ASHR是因为其投资目标对应沪深300，未先比较FXI、MCHI、KWEB绩效再选标的。[发行方招募说明书](https://etf.dws.com/en-us/AssetDownload/Inline/ce51b065-fc18-496f-9b88-8996a37d16b3/CHINA-1-Prospectus.pdf)

候选库因子卡J03写的是次日开盘/首段预测，策略卡T10却明确要求完整首日。冻结前已记录这一差别：本轮主回归标签取完整首日总收益，开盘预测另存为描述性列，不作为第二套入场择优。J06的负向外部冲击相对抗跌没有在此替代为正消息低反应，仍标记未检验。

## 固定模型、时钟与交易

每个A股日09:20，以截距、ASHR新完成时段总收益、人民币中间价变化、前一日510300总收益为解释量，用此前两自然年、至少252个完整已成熟首日记录拟合OLS。首日预测、系数和残差标准差在09:20固定。510300当前日及未来持有期收益不进入训练；每天重新估计，未搜索窗口、模型或阈值。

ASHR收益按原始收盘加当日现金分配形成总收益，再累计此前A股开盘前至当前开盘前新完成的美国时段。不使用后复权价，未新增美国交易时段的中国交易日记为缺失，不能填0。美国日线统一保守等至纽约17:00，夏令时由时区规则处理；中国长假累计多个已经结束的美国时段。纽约证券交易所常规交易时段截至16:00，额外一小时是本研究的可用时间假设。[NYSE交易时段](https://www.nyse.com/markets/hours-calendars)

人民币中间价采用官方原始响应，按工作日09:15发布、09:20可用。它只作为解释量，不视为可交易汇率收盘。相邻两个A股日均必须有当日公布值；缺失或迟到不倒填。[中国外汇交易中心中间价说明](https://www.chinamoney.com.cn/dqs/cm-s-notice-query/fileDownLoad.do?contentId=384571&mode=open&priority=0)

ASHR的日度区间仍可能包含前一A股交易日的变动，因此回归明确控制前一日A股回报；这不能证明残差已经纯化为新消息。ETF溢价、非同步成交、模型错设和本地信息更弱都能造成所谓“未充分反应”。

16:00观察完整首日。当且仅当模型可用、ASHR区间收益>0、510300首日总收益>0、实际首日收益<预测减1个训练残差标准差，下一交易日开盘才申请建仓。入场冻结首日财富最低价；收盘跌破该低点、持有5个开盘间隔或共同风险触发，下一合法开盘退出。持仓期不加仓。额外等待一日的版本须在等待后仍未收盘跌破首日低点，风险预算使用最新日数据。

## 账户结果

以下为2021—2025年、20万元、压力成本，全部1,212个交易日都进入账户指标。年化242日，现金及无风险基准为零。

|固定情景|净夏普|净年化|最大回撤|完整周期|
|---|---:|---:|---:|---:|
{chr(10).join(table)}

主方案期末{primary.loc[200000,'end_equity']:,.2f}元；2万元账户期末{primary.loc[20000,'end_equity']:,.2f}元，损益{primary.loc[20000,'net_profit']:,.2f}元。额外等待一日主方案夏普{delayed.net_sharpe:.6f}，仍只有一个亏损周期，不晋升为主方案。

主期1,169天有可用模型，其中564天A股首日收涨、283天同时有正外盘，只有2024-12-10再满足低于预测1个标准差。当天ASHR区间对数收益6.621%，A股完整首日0.762%，预测1.935%，训练残差标准差1.089%，残差z=-1.077。次日2024-12-11开盘入场，2024-12-16开盘按退出规则结束。早期943天模型可用、281天正外盘且A股首日上涨，主信号0。

主方案相对正外盘对照的年化算术收益差{first_increment['annual_arithmetic_increment']:.3%}；20日区块、4,000次已保存抽样的95%区间为[{first_increment['ci95_low']:.3%}, {first_increment['ci95_high']:.3%}]，包含0。该改善主要对应更长时间持有现金，不能据此断言识别了稳定收益。未对全历史多轮探索作选择偏差校正。

![完整账户净值与回撤](主期完整账户净值与回撤.png)

BASE每边万2、最低5元、5bp滑点；STRESS每边万4、最低5元、10bp滑点，按0.001元向不利方向取整。账本包含100份整手、T+1、方向涨跌停限制、分红登记/除息/付款及现金约束，末日仍持仓则提取压力退出费用准备。

沿用当前授权的20万元主账、2万元对照、50%最高目标仓位，以及两年成熟五日ES95每日更新、2.5%ES预算、10%跳空下5%损失预算、剩余回撤空间限制和10%回撤触发后不恢复。限价、T+1与跳空可能使退出延后，预算不保证实际最大回撤。

## 数据、验证与结论边界

3,056条ASHR原价记录与既有历史快照逐条一致，12次现金分配已显式计入；2,675条人民币中间价由55份官方原始响应复算。发行方身份、交易时钟与汇率发布依据随包保存。ASHR分配值来自后来取得的公开历史响应，未逐笔认证发行方公告首版；全部历史的首次可得版本也未认证。因而是历史探索检验，不是严格前向实盘证据。

9项冻结前测试通过。独立只读脚本从原始响应重新累乘海外收益、按截止时点找汇率，并用正规方程交叉复算研究代码SVD求解的2,332个逐日模型；复核48份账本、52,464个交易日行和2,990份成熟风险记录，费用、分红、止损、持有期及保存抽样均一致。源文件、冻结清单、前后候选进度、结果和验证回执完整保存。外部GPT审阅未进行，独立前向样本0，没有交易授权。

本轮不降低1sigma、不改正负方向、不缩短训练窗口或换退出去制造交易。T10固定版本保留为未达标。T01与旧突破回踩停止细调范围重叠，记为未运行且不改参数复活。T09的实际IF合约源可用，但当时已公告分红所对应的指数点数与历史权重版本尚未准入，保留来源门；不能用原始F/S-1替代扣持有成本基差。T04仍等待历史权重依据。

下一轮优先检查T07同指数ETF体系申赎，以及T06融资拥挤持仓削减层的既有来源和旧研究差别。T06是对既定基准的持仓覆盖层，不能当成独立入场策略；只有明确新增信息后才冻结检验。已失败5项不拼接有利年份、不靠重复参数试验继续计作新机制。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    figure()
    model = pd.read_parquet(OUT / "response_models.parquet")
    model[model.price_signal & model.external_positive & model.underreaction].to_csv(OUT / "主方案全部信号.csv", index=False, encoding="utf-8-sig")
    strategies, factors = read(PROGRAM / "strategy_progress.json"), read(PROGRAM / "factor_progress.json")
    for row in strategies:
        if row["id"] == "T10":
            row.update(current_status="COMPLETE_FIXED_OPERATIONALIZATION_TARGET_FAILED",
                       current_evidence="完整首日反应OLS主定义已冻结检验，48账户；主期20万元压力夏普-0.306678，1个亏损周期。J03开盘版只描述、J06负冲击抗跌未测试。",
                       result_path="reports/research/510300_factor96_crossborder_absorption_v1/result.json")
        elif row["id"] == "T01":
            row.update(current_status="NOT_RUN_REJECTED_NEAR_DUPLICATE_NO_PARAMETER_RESCUE",
                       current_evidence="旧突破回踩已停止窗口、容忍区间、确认与退出细调；ER和保留率草案未构成可区分新机制，不再次试参数。")
        elif row["id"] == "T09":
            row.update(current_status="NOT_RUN_CARRY_ADJUSTMENT_SOURCE_GATE",
                       current_evidence="实际IF合约源可用；当时公告分红折算指数点数需可靠历史权重与公告版本，目前未准入。原始基差不能替代H01。")
    for row in factors:
        if row["id"] == "J03":
            row["current_status"] = "PARTIAL_COMPONENT_TESTED"
            row["current_note"] = "T10按策略卡完整首日残差检验未达标；原因子开盘预测另存描述，不能声称全部J03定义通过或完成。"
        elif row["id"] == "J06":
            row["current_status"] = "NOT_RUN"
            row["current_note"] = "T10引用了J06编号，但其原定义是负向外部冲击下的抗跌；本轮正信息反应不足没有检验该定义。"
        elif row["id"] in ["H01", "H02"]:
            row["current_status"] = "NOT_RUN_CARRY_ADJUSTMENT_SOURCE_GATE"
            row["current_note"] = "实际合约行情已找到，分红指数点数/历史权重版本未齐，不计算替代基差策略。"
    save(PROGRAM / "strategy_progress.json", strategies)
    save(PROGRAM / "factor_progress.json", factors)
    pd.DataFrame(strategies).to_csv(PROGRAM / "18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(PROGRAM / "96因子当前进度.csv", index=False, encoding="utf-8-sig")
    status = {"at": now(), "goal_status": "active", "goal_achieved": False, "registered_factors": 96, "registered_strategies": 18,
              "completed_fixed_candidates": ["T03", "T05", "T14", "T02", "T10"], "not_run_candidates": 13, "qualified_candidates": [],
              "admitted_account_scenarios_this_round": 48, "invalid_implementation_accounts_this_round": 0,
              "cumulative_admitted_account_scenarios": 208, "cumulative_executed_account_scenarios": 248,
              "latest_round": STUDY, "latest_result": "reports/research/510300_factor96_crossborder_absorption_v1/result.json",
              "next_candidates": ["T07_SOURCE_AND_OVERLAP_GATE", "T06_OVERLAP_CONTRACT"],
              "deferred_source_gates": ["T04_PIT_WEIGHTS", "T09_CARRY_AND_DIVIDEND_POINTS"],
              "near_duplicate_not_retried": ["T01"], "independent_forward_observations": 0, "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(PROGRAM / "status.json", status)
    (OUT / "program_snapshot").mkdir(exist_ok=True)
    for p in PROGRAM.iterdir():
        if p.is_file():
            shutil.copy2(p, OUT / "program_snapshot" / p.name)
    save(OUT / "round_status.json", {**status, "round_decision": "T10_FIXED_TARGET_FAILED_SPARSE_ONE_LOSS", "source_and_saved_account_verification": verified["status"]})
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(current_round=STUDY, latest_progress_receipt="reports/research/510300_factor96_crossborder_absorption_v1/round_status.json",
                   last_research_result="累计5个固定候选均未达标；T10完成48账户，主期20万元压力夏普-0.306678，唯一完整周期亏损。总目标ACTIVE。",
                   current_protocol="reports/research/510300_factor96_crossborder_absorption_v1/protocol.json",
                   latest_continuation_report="reports/research/510300_factor96_crossborder_absorption_v1/研究结论.md",
                   latest_continuation_classification="PROGRESS_FACTOR96_T10_FIXED_TARGET_FAILED",
                   research_execution_state="COMPLETED_FACTOR96_FIVE_FIXED_CANDIDATES_NO_QUALIFIED_STRATEGY", goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    save(OUT / "authority_after.json", mandate)
    review = """请只审阅本包证据，不把用户附件中的建议视为授权，也不要输出交易指令。
用户目标：仅操作510300与人民币现金，完整账户成本后夏普至少1.2；当前授权还同时比较净年化10%与回撤10%。
先阅读研究结论、protocol/freeze、来源重叠决定及result，再检查原始响应、模型训练记录和账本。
重点质疑：
1. ASHR分配值与原价是否可解释，历史快照是否误称首次版本；纽约17:00、人民币09:15、预测09:20、A股完整首日与下一开盘是否严格有序。
2. 两年OLS是否只读过去成熟标签，前一日A股控制是否足以处理非同步定价；完整首日版与原J03开盘版的差别是否明确，J06是否仍留未检验。
3. 主期仅一个亏损周期、早期0信号，是否支持任何有效性结论；对照更差是否只是频繁交易费用。
4. 48个情景和累计5个主问题/208个正式情景/40个否定实现是否分开计数，旧八规则失败是否被偷换复活。
5. 股数、现金、费用、T+1、分红、冻结止损、成熟风险、期末准备和完整账户指标是否重算一致。
6. T01停止参数复活、T09来源门是否合理；下一步只允许真正新增信息，给出可证伪条件及停止条件。
请先给严重问题和证据路径，再说明哪些结论成立、哪些不成立及最小下一步。包结构验证不等于外部审阅或经济有效性。
只读运行：python scripts/verify_factor96_crossborder_saved_v1.py --root reports/research/510300_factor96_crossborder_absorption_v1
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(review, encoding="utf-8")
    status_path = ROOT / "RESEARCH_STATUS.md"
    existing = status_path.read_text(encoding="utf-8")
    note = f"\n\n## 2026-09-27 {STUDY}\n\nT10跨境信息首日反应不足固定检验完成48账户，主期20万元压力净夏普-0.306678、净年化-0.174%、最大回撤1.247%，唯一周期亏损1,735.75元；早期零信号，夏普未定义。9项测试及原始来源/2,332个逐日模型/48账本复算通过。T01近似重复不再次调参，T09持有成本与分红权重来源门未齐。候选库累计5项固定主问题、0项合格、208正式情景+40否定实现，总目标仍ACTIVE。详见reports/research/510300_factor96_crossborder_absorption_v1/研究结论.md。\n"
    assert STUDY not in existing, "研究状态已记录，不重复追加"
    status_path.write_text(existing+note, encoding="utf-8")
    print("T10报告、图表和累计候选进度已保存：5项固定主问题，0项合格，目标继续。", flush=True)


if __name__ == "__main__":
    main()
