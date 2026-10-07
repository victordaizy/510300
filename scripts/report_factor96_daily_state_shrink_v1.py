"""写入日更校准固定检验的结论、对照图和研究进度。"""
from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"reports/research/510300_factor96_daily_state_shrink_v1"
PROGRAM = ROOT/"reports/research/510300_factor96_program_v1"
AUTHORITY = ROOT/"config/510300_existing_data_training_mandate_v1.json"
STUDY = "510300_FACTOR96_DAILY_STATE_SHRINK_V1"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main():
    assert not (OUT/"round_status.json").exists(), "本轮进度已写入，不重复"
    result = read(OUT/"result.json")
    verify = read(OUT/"saved_verification_receipt.json")
    assert verify["status"] == "PASS_SAVED_DAILY_SHRINK_CLOCK_COEFFICIENTS_AND_ACCOUNTS"
    assert result["new_accounts"] == 112
    table = pd.read_csv(OUT/"metrics.csv")
    main_rows = table[(table.period == "MAIN") & (table.cost == "STRESS") & table.lag.eq(1)]
    status, strategies, factors = [read(PROGRAM/name) for name in ["status.json", "strategy_progress.json", "factor_progress.json"]]
    assert status["cumulative_admitted_account_scenarios"] == 320
    assert status["cumulative_executed_account_scenarios"] == 472
    stamp = datetime.now().astimezone().isoformat()
    status.update(at=stamp, latest_round=STUDY, latest_result=OUT.relative_to(ROOT).as_posix()+"/round_status.json",
        admitted_account_scenarios_this_round=112, invalid_implementation_accounts_this_round=0,
        cumulative_admitted_account_scenarios=432, cumulative_executed_account_scenarios=584,
        new_source_documents_this_round=0, new_searchable_text_documents_this_round=0,
        source_field_candidates_this_round=0, new_source_occurrences_this_round=0,
        new_version_roles_this_round=0, new_explicit_version_links_this_round=0, completed_calendar_examples_this_round=0,
        additional_fixed_research_questions=[{"id": "T18_DAILY_AUTHORITY_VARIANT", "status": result["status"], "result": OUT.relative_to(ROOT).as_posix()+"/result.json"}],
        completed_total_fixed_questions=8, original_library_completed_questions=7,
        original_T18_quarterly_seed="NOT_RUN", goal_status="active", goal_achieved=False,
        next_candidates=["T13_EVENT_IDENTITIES_AND_FREE_FLOAT", "T12_PURPOSE_ROOTS_AND_FREE_FLOAT", "T04_HISTORICAL_WEIGHTS"])
    summary = "日更变体历史点估计通过但未独立验证" if result["primary_historical_point_pass"] else "日更变体固定检验未达到目标"
    for row in strategies:
        if row["id"] == "T18":
            row["current_status"] = "DAILY_AUTHORITY_VARIANT_COMPLETE_ORIGINAL_QUARTERLY_NOT_RUN"
            row["current_evidence"] = summary+"；112新账户；三家族固定等权、成熟内层校准、分歧与CUSUM消融。遵守用户两年日更要求；原季度草案仍NOT_RUN，不改旧失败规则。"
            row["daily_authority_variant_result_path"] = OUT.relative_to(ROOT).as_posix()+"/result.json"
            row["original_seed_status"] = "NOT_RUN"
    for row in factors:
        if row.get("id") in ["P05", "P06"]:
            row["current_evidence"] = summary+"；仅本轮固定三家族预测分歧及有界成熟误差CUSUM用途，不能泛化为全部因子用途有效。"
            row["current_note"] = row["current_evidence"]
            row["current_status"] = "TESTED_DAILY_CALIBRATION_VARIANT_NO_STANDALONE_CLAIM"
            row["daily_authority_variant_result_path"] = OUT.relative_to(ROOT).as_posix()+"/result.json"
        elif row.get("id") in ["A01", "E01", "F01"]:
            row["additional_daily_calibration_use"] = summary+"；原有定义及旧结果不变。"
    authority = read(AUTHORITY)
    save(OUT/"authority_before.json", authority)
    authority.update(current_round=STUDY, current_protocol=OUT.relative_to(ROOT).as_posix()+"/protocol.json",
        latest_progress_receipt=OUT.relative_to(ROOT).as_posix()+"/round_status.json",
        latest_continuation_report=OUT.relative_to(ROOT).as_posix()+"/研究结论.md",
        latest_continuation_classification="PROGRESS_FIXED_DAILY_SHRINK_VARIANT_AND_112_ACCOUNTS",
        last_research_result=summary+"；累计432正式+152否定实现=584账户。原库7个固定问题加1个授权日更变体，目标未完成。",
        research_execution_state="FIXED_DAILY_ADAPTATION_COMPLETE_NO_INDEPENDENTLY_QUALIFIED_STRATEGY",
        goal_achieved=False, goal_status="active")
    assert authority["executable_assets"] == ["510300.SH", "CASH_CNY"] and not authority["orders_authorized"]
    save(AUTHORITY, authority)
    save(OUT/"authority_after.json", authority)
    save(PROGRAM/"status.json", status)
    save(PROGRAM/"strategy_progress.json", strategies)
    save(PROGRAM/"factor_progress.json", factors)
    pd.DataFrame(strategies).to_csv(PROGRAM/"18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(PROGRAM/"96因子当前进度.csv", index=False, encoding="utf-8-sig")
    for p in PROGRAM.iterdir():
        if p.is_file():
            target = OUT/"program_snapshot"/p.name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(p, target)
    save(OUT/"round_status.json", status)
    names = {"NO_STATE": "无状态均值", "STATIC_EQUAL": "三家族固定等权", "CALIBRATED": "成熟误差校准",
             "DISAGREEMENT": "校准加分歧控制", "MONITORED": "校准加失效监测", "FULL": "完整日更变体"}
    lines = ["| 方案 | 净夏普 | 净年化 | 最大回撤 | 完整交易周期 |", "|---|---:|---:|---:|---:|"]
    for policy in names:
        row = main_rows[(main_rows.capital == 200000) & (main_rows.policy == policy)].iloc[0]
        sharpe = f"{row.net_sharpe:.6f}" if pd.notna(row.net_sharpe) else "未定义"
        lines.append(f"| {names[policy]} | {sharpe} | {row.cagr:.3%} | {row.max_drawdown:.3%} | {int(row.completed_cycles)} |")
    small = main_rows[(main_rows.capital == 20000) & (main_rows.policy == "FULL")].iloc[0]
    early = table[(table.period == "EARLY") & (table.cost == "STRESS") & table.lag.eq(1) & table.policy.eq("FULL") & table.capital.eq(200000)].iloc[0]
    delay = table[(table.period == "MAIN") & (table.cost == "STRESS") & table.lag.eq(2) & table.policy.eq("FULL") & table.capital.eq(200000)].iloc[0]
    pairs = "\n".join(f"- 相对{names[p['control']]}：年化日均收益差{p['annual_mean_return_difference']:.3%}，95%区间[{p['interval95'][0]:.3%}, {p['interval95'][1]:.3%}]。" for p in result["primary_increment"])
    text = f"""# 510300三家族日更成熟校准固定检验

**夏普1.2目标尚未实现。{summary}。** 本轮完成112个完整账户情景，0次参数搜索。累计432正式账户情景、152否定实现情景，共584；原候选库7个问题及本轮1个授权日更变体，均无独立验证合格策略。原T18季度草案仍NOT_RUN。

## 为什么转向这项检验

发行目录与配股原文上一轮已完成交付，但完整事件链和历史自由流通分母仍未满足T13。本轮核查官方规则及接口说明，没有取得可准入的完整历史权重或分母序列；不声称这些数据绝对不存在，也不继续把更多公告数量当作已能回测。

价格路径效率、历史成员20日总回报广度和融资净变化已有来源。附件T18建议季度更新；用户2026-09-24明确要求“只用最近两年的数据训练，每天滚动更新”。本轮遵守现行授权，事前记录日更变体，与未运行的原季度草案分别登记。没有复活旧SELECTED_MIX、价格优化或后悔加权失败账户。

## 固定方法与账户

每日仅用当前之前两日历年、五日开盘收益已经成熟的样本，三个信息家族分别进行固定强度岭回归，再固定等权。内层校准使用固定五日相位的成熟预测；内层训练同样不能越过当前两年下限。平均预测误差按n/(n+60)收缩，分歧和有界CUSUM监测各自保留消融对照。

主方案把外部广度和融资统计多等一完整交易日，另有再等一日对照；价格特征取当日收盘，次开盘执行。净期望扣0.28%双边比例成本代理，随后映射0、12.5%、25%、37.5%、50%五档目标。实际账户另计最小佣金、滑点和价位，20万元与2万元分别模拟，没有把代理成本当作实际账户已付成本。

原有ES、缺口、回撤预算和50%上限继续；本周期价格下跌时不追加份额，开盘跳空跌破首次进入财富价也撤去新增请求。最长20日，无有效模型、目标为零或风险退出则下个合法开盘退出，T+1及涨跌停保守约束保留。回撤停机不因模型恢复而重启。所有空仓日、应收分红、未平仓损益及末日退出成本准备计入。

## 2021—2025年主期结果

以下均为20万元、压力成本、主时钟，未从对照中另挑胜者：

{chr(10).join(lines)}

主方案2万元对照：净夏普{small.net_sharpe:.6f}，净年化{small.cagr:.3%}，最大回撤{small.max_drawdown:.3%}。20万元主方案在2017—2020年前期的净夏普{early.net_sharpe:.6f}；主期再多等一日的净夏普{delay.net_sharpe:.6f}。全体112行结果在metrics.csv，逐年结果在annual_metrics.csv，不拼接有利时期。

主期20万元压力净收益的配对区块复算：

{pairs}

这些差值是日均收益差乘242，不是复合年化差。4000次20日循环区块的抽样索引已保存。普通95%区间不代表已消除本项目长期反复研究的选择偏差。

![完整主期净值与回撤](主期20万元压力净值与回撤.png)

## 验证与边界

20项冻结前测试通过。只读核验从保存输入核对特征、每一内外层系数的训练时钟和充分统计量、五档目标及112套账本，逐日检查请求、成交、现金、份额、分红、退出和指标。归档后另作全新解压核验。图中半仓持有基准没有候选的风险退出；现金零息且夏普未定义。

主时钟有{result['model_available_days']['1']}个有效模型判断日，延迟时钟有{result['model_available_days']['2']}个。其余时点保持NO_VIEW，已有持仓依事前规则退出，账户现金日仍计入评价；NO_VIEW不被解释成当前市场看空。来源属于2026年回取历史，没有逐日首版发布快照。两段历史均已被此前研究观察，本轮不是独立样本外或前向验证。

固定规则结果保留，不按本轮收益修改窗口、岭强度、监测阈值、仓位或退出条件。下一步回到未满足的数据门及不同机制；不把失败账户择优拼接。独立前向样本0，external_review=NOT_PERFORMED，目标active。研究资产仅510300.SH与CASH_CNY，无下单授权，原PCF／IOPV任务暂停不变。
"""
    (OUT/"研究结论.md").write_text(text, encoding="utf-8")
    prompt = """请审阅当前ZIP。先读00_README_FIRST.md、研究结论、protocol.json及source_evidence/daily_training_authority.json。
用户目标仅交易510300达到成本后夏普1.2，同时检验年化10%和回撤10%。本轮是服从现行两年日更要求的T18参考变体；原季度草案未运行，不能混称全部原草案已实现。
请重点核对：外层及内层是否都受当前两年下限约束；标签j+6是否严格早于判断；E01历史成员与缺失及统计时钟；模型未知与实际现金是否分开；三个家族是否固定等权且没有挑专家；有界CUSUM是否按固定20个成熟非重叠原点处理；0.28%净期望代理与实际佣金滑点是否分开；五档是否符合50%上限；跌价和跳空时是否追加，回撤停机是否擅自恢复；112账户及前期/主期/延迟/资金/成本是否完整披露；历史开发结果是否被当作独立样本外。
请给出严重程度、具体文件和证据、经济反例、下一步策略或数据工作、验证条件及停止条件。保存结果核验和结构检查不是策略认证。不要建议以修改阈值或拼接盈利区间挽救本轮失败。只读命令见REPRODUCE_SAVED_RESULTS.txt，不要重跑prepare/freeze/run入口。外部审阅未执行，本提示供用户使用。
"""
    (OUT/"01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for policy, label, color in [("FULL", "完整日更变体", "#bb3e31"), ("STATIC_EQUAL", "三家族固定等权", "#245b83"), ("NO_STATE", "无状态均值", "#53806a"), ("BUY_HOLD_50", "半仓买入持有基准", "#858585")]:
        key = "MAIN_200000_STRESS_"+("benchmark_" if policy == "BUY_HOLD_50" else "lag1_")+policy
        ledger = pd.read_parquet(OUT/"accounts"/key/"ledger.parquet")
        axes[0].plot(ledger.date, ledger.equity/200000, label=label, color=color, lw=1.8 if policy == "FULL" else 1.2)
        axes[1].plot(ledger.date, -ledger.drawdown, color=color, lw=1.2)
    axes[0].set_title("510300｜2021—2025年 · 20万元 · 压力成本 · 主时钟", loc="left", fontsize=15, pad=15)
    axes[0].set_ylabel("完整账户净值")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)
    axes[1].set_ylabel("账户回撤")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.085, .015, "保留全部现金日与未平仓损益；半仓持有基准未应用候选风险退出。历史开发证据，不构成独立验证。", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(OUT/"主期20万元压力净值与回撤.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"round": STUDY, "new_accounts": 112, "formal_accounts": 432, "executed_accounts": 584, "goal_status": "active", "goal_achieved": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
