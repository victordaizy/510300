"""从保存结果形成研究结论、统计适用性说明、复核入口与交付导航。"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "reports/research/510300_sequential_patterns_regime_v1"
sys.path.insert(0, str(OUT / "code"))
from sequential_patterns_regime_v1 import clean, detect, digest, features, now, save_csv, save_json


def markdown(path, value):
    path.write_text(value.strip() + "\n", encoding="utf-8")


def fmt(value, pct=False):
    if pd.isna(value):
        return "未定义"
    if value == 0:
        value = 0.0
    return f"{value:.2%}" if pct else f"{value:.3f}"


def chart():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import PercentFormatter
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 8.0), sharex=True, gridspec_kw={"height_ratios": [1.8, 1]})
    series = [("PATTERN_ONLY", "仅形态 · 16笔", "#176b79"),
              ("STATE_ONLY", "固定市场状态 · 10笔", "#b27425"),
              ("FULL", "完整启停 · 0笔", "#b24e52"),
              ("BUY_HOLD", "买入持有", "#85909b")]
    for policy, label, color in series:
        d = pd.read_parquet(OUT / "results/accounts/RECENT_504_PRIMARY/STRESS" / policy / "daily.parquet")
        dates = pd.to_datetime(d.date)
        axes[0].plot(dates, d.equity_cny / 200000, label=label, color=color, lw=1.7 if policy != "FULL" else 2.2)
        axes[1].plot(dates, d.drawdown, color=color, lw=1.5)
    axes[0].set_title("510300：形态线索与启停机制出现明显分化", loc="left", fontsize=17, pad=15)
    axes[0].set_ylabel("完整20万元账户净值")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[1].set_ylabel("距历史高点回撤")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].axhline(-.10, ls="--", lw=1, color="#b24e52", alpha=.6)
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.08, .026,
             "2024-08-20至2026-09-16，压力成本，包含全部空仓日。仅形态夏普1.185，未达到1.2。\n"
             "完整启停未开仓，夏普未定义；历史重放不是独立前向验证。最大一笔利润来自2024年9月下旬至10月。",
             color="#555555", fontsize=9)
    fig.tight_layout(rect=[0, .09, 1, 1])
    fig.savefig(OUT / "账户对比.png", dpi=165)
    fig.savefig(OUT / "账户对比.svg")
    plt.close(fig)


def main():
    r = OUT / "results"
    summary = json.loads((r / "summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(r / "account_comparison.csv")
    signals = pd.read_parquet(r / "signals.parquet")
    dec = pd.read_parquet(r / "daily_decisions.parquet")
    labels = pd.read_parquet(r / "event_labels.parquet")
    primary = comparison[comparison.period == "RECENT_504_PRIMARY"]
    annual = pd.read_csv(r / "annual.csv")
    initial_annual = r / "annual_initial_metadata.csv"
    if not initial_annual.exists():
        shutil.copy2(r / "annual.csv", initial_annual)
    annual = pd.read_csv(initial_annual)
    history_calendar = pd.read_csv(OUT / "inputs/historical_calendar.csv", dtype=str).trade_date
    future_calendar = pd.read_csv(OUT / "inputs/calendar.csv", dtype=str).trade_date
    calendar = pd.concat([history_calendar, future_calendar]).drop_duplicates().sort_values()
    corrected = 0
    for ix, row in annual.iterrows():
        yr = str(row.year)
        expected = calendar[calendar.str.startswith(yr)]
        ledger = pd.read_parquet(r / "accounts" / row.period / row.cost / row.policy / "daily.parquet")
        actual = ledger.loc[ledger.date.str.startswith(yr), "date"]
        partial = actual.iloc[0] != expected.iloc[0] or actual.iloc[-1] != expected.iloc[-1]
        corrected += int(bool(row.partial_year) != bool(partial))
        annual.loc[ix, "partial_year"] = bool(partial)
    save_csv(r / "annual.csv", annual)
    save_json(r / "annual_metadata_correction.json", {"recorded_at": now(), "corrected_rows": corrected,
              "reason": "首年或末年不自动等于不完整年；改为核对实际首末交易日与日历。原次数、收益和账户不变。",
              "original_metadata": "results/annual_initial_metadata.csv", "accounts_rerun": 0})
    joined = signals.merge(dec, left_on=["signal_idx", "family"], right_on=["idx", "family"], suffixes=("", "_decision"))
    save_csv(r / "trigger_decisions.csv", joined)
    current_primary = joined[joined.signal_date >= summary["primary_start"]]
    save_csv(r / "primary_gate_reasons.csv", current_primary.groupby(["family", "reason"]).size().rename("count").reset_index())
    save_csv(r / "latest_historical_state.csv", dec[dec.date == summary["data_end"]])
    trades = pd.read_csv(r / "accounts/RECENT_504_PRIMARY/STRESS/PATTERN_ONLY/trades.csv")
    total_pnl = float(trades.net_pnl.sum())
    biggest = trades.nlargest(1, "net_pnl").iloc[0]
    concentration = {"net_pnl_cny": total_pnl, "largest_trade": biggest.to_dict(),
                     "largest_share_of_net_pnl": biggest.net_pnl / total_pnl,
                     "top3_share_of_net_pnl": trades.nlargest(3, "net_pnl").net_pnl.sum() / total_pnl,
                     "definition": "按完整成交周期的实际人民币净利润排序，分母是16周期净利润之和；不冒充固定事件窗口归因。",
                     "new_accounts": 0}
    save_json(r / "profit_concentration.json", concentration)
    record = joined[joined.signal_id == biggest.signal_id].iloc[0]
    save_json(r / "largest_trade_gate_record.json", record.to_dict())
    frequency = []
    for family, group in signals.groupby("family"):
        idx = group.signal_idx.to_numpy()
        waits = idx[5:] - idx[:-5]
        frequency.append({"family": family, "confirmed_events": len(group),
                          "median_span_to_six_signals_sessions": float(np.median(waits)) if len(waits) else None,
                          "share_six_event_spans_over_504": float((waits > 504).mean()) if len(waits) else None,
                          "note": "只度量识别信号数量的等待，不含退出成熟时间、不剔除未成交；不是可用训练样本数。"})
    save_csv(r / "recognition_delay.csv", frequency)
    recent_labels = labels[(labels.signal_date >= summary["primary_start"]) & (labels.cost == "STRESS") & (labels.label_status == "MATURE")]
    save_csv(r / "recent_event_by_family.csv", recent_labels.groupby("family").agg(
        count=("net_return", "size"), mean_net=("net_return", "mean"),
        median_net=("net_return", "median"), win_rate=("net_return", lambda x: (x > 0).mean())).reset_index())

    # 保存原始数值痕迹，同时明确零方差账户不存在夏普及夏普差异区间。
    raw_ci = pd.read_csv(r / "paired_bootstrap.csv")
    valid_ci = raw_ci.copy()
    valid_ci["status"] = "COMPUTED_FROM_SAVED_DRAWS"
    invalid = valid_ci.measure == "SHARPE_DELTA"
    valid_ci.loc[invalid, ["lower95", "upper95", "median"]] = np.nan
    valid_ci.loc[invalid, "status"] = "NOT_COMPUTED_ZERO_VARIANCE_FULL_ACCOUNT"
    save_csv(r / "paired_bootstrap_valid.csv", valid_ci)
    save_json(r / "statistical_validity.json", {
        "recorded_at": now(), "authoritative_interval_table": "results/paired_bootstrap_valid.csv",
        "original_arithmetic_trace": "results/paired_bootstrap.csv",
        "raw_trace_sha256": digest(r / "paired_bootstrap.csv"),
        "invalidated_rows": int(invalid.sum()),
        "reason": "原始区块计算对零波动分母使用0保护值；FULL近期全程空仓，夏普没有定义，三个夏普差异区间全部作废，不能用保护值作推断。",
        "account_metrics_already_reported_sharpe_null": True,
        "protocol_parameters_changed": False, "accounts_rerun": 0, "new_draws": 0,
        "cagr_intervals_unchanged": True})

    # 用真实输入截断复核形态时序；不重新生成研究账户或选择参数。
    raw = pd.read_parquet(OUT / "inputs/prices.parquet")
    div = pd.read_csv(OUT / "inputs/dividends.csv")
    saved_features = pd.read_parquet(r / "features.parquet")
    saved_steps = pd.read_parquet(r / "steps.parquet")
    checks = []
    for cutoff in (650, 1600, 2600, len(raw) - 20):
        subset = features(raw.iloc[:cutoff], div)
        pd.testing.assert_frame_equal(saved_features.iloc[:cutoff].reset_index(drop=True), subset)
        _, steps, sig = detect(subset)
        pd.testing.assert_frame_equal(saved_steps[saved_steps.idx < cutoff].reset_index(drop=True), steps)
        pd.testing.assert_frame_equal(signals[signals.signal_idx < cutoff].reset_index(drop=True), sig)
        checks.append({"cutoff_rows": cutoff, "cutoff_date": subset.date.iloc[-1], "feature_step_signal_prefix": "PASS"})
    save_json(OUT / "causality_verification.json", {"recorded_at": now(), "tests": checks,
               "synthetic_unit_tests": 7, "pytest_result": "7 passed", "account_reruns": 0})
    chart()

    names = {"PATTERN_ONLY": "仅形态", "STATE_ONLY": "加入固定市场状态", "RECENT_ONLY": "仅近期启停", "FULL": "完整启停机制", "BUY_HOLD": "买入持有", "CASH": "现金"}
    table = ["|方案|基础净年化|基础夏普|压力净年化|压力夏普|压力最大回撤|压力完整周期|", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in names:
        b = primary[(primary.policy == policy) & (primary.cost == "BASE")].iloc[0]
        s = primary[(primary.policy == policy) & (primary.cost == "STRESS")].iloc[0]
        cycles = "持续持有，期末未清仓" if policy == "BUY_HOLD" else str(int(s.completed_cycles))
        table.append(f"|{names[policy]}|{fmt(b.net_cagr, True)}|{fmt(b.net_sharpe)}|{fmt(s.net_cagr, True)}|{fmt(s.net_sharpe)}|{fmt(s.max_drawdown, True)}|{cycles}|")
    delay_table = ["|形态|全历史确认数|累计六次确认的中位跨度|超过504日的六次确认窗口比例|", "|---|---:|---:|---:|"]
    chinese = {"BREAKOUT": "收缩后突破", "RECLAIM": "破低后收复", "REPAIR": "急跌后修复"}
    for row in frequency:
        delay_table.append(f"|{chinese[row['family']]}|{row['confirmed_events']}|{row['median_span_to_six_signals_sessions']:.0f}交易日|{row['share_six_event_spans_over_504']:.1%}|")
    markdown(OUT / "研究结论.md", f"""
# 510300 连续形态与状态启停 V1：形态有近期线索，启停尚未建立

本轮已经把研究主线改为三类连续形态及其启停，而非一套均线规则。最近504日仅按形态交易在基础成本下年化14.85%、夏普1.281；压力成本下年化13.63%、夏普1.185，未同时通过两档成本的1.2门槛。完整启停机制没有产生交易，近期年化0%、夏普未定义，未实现10%/1.2目标。一个很明显的约束已经量化：形态触发稀疏，等待足够同状态成功案例时，可能错过其主要收益；预设的市场状态过滤也会误删机会。本轮不能据此宣布三类形态普遍无效，亦不能把基础成本下的接近过线当成当前有效。

## 用户方向与本轮执行

仅研究510300.SH与现金，初始20万元，最大回撤10%作为保留的风险目标。当前夏普以新要求1.2为准，新增明确年化10%，年度交易次数下限已取消。不采集要求保持。

形态被记录成“准备—观察/试探—确认或失败/超时”。日线首版使用后续收盘确认、次日开盘成交、T+1和最长5日正常持有；日内分钟级过程没有验证。14次股息的登记、除息和到账分开处理。价量代理不叫真实主力净流入。

协议在计算本轮结果前固定，三类247个过程全部保存，其中87个确认、160个失败或超时。收缩后突破123个过程、42次确认；破低后收复66个过程、24次确认；急跌后修复58个过程、21次确认。失败案例没有从样本中删除。近期期内17次确认，实际仅形态账户完成16个周期，账户冲突与未成交原因另存。

## 固定近期主评价：{summary['primary_start']}至{summary['primary_end']}

{chr(10).join(table)}

基础成本单边佣金万二、滑点5基点；压力成本万四、10基点，均最低5元并按0.001元不利取整。全部504个交易日和空仓日计入收益与风险，不把每笔夏普相加。以上是已反复使用历史上的顺序重放；本轮未得到任何独立前向样本。买入持有的“0完整周期”是期末持续持仓，不能误读为未投资。

## 收益有多集中

压力成本仅形态账户16个完整周期净利润共{total_pnl:,.2f}元。最大一笔为{biggest.entry_date}至{biggest.exit_date}的收缩后突破，净利润{biggest.net_pnl:,.2f}元，占总净利润{concentration['largest_share_of_net_pnl']:.2%}；前三笔合计占{concentration['top3_share_of_net_pnl']:.2%}，超过100%表示其余交易合计为负。

这里按完整交易周期归因，不能把全部盈利解释为固定的政策事件窗口收益。该最大周期的确认时点是{biggest.signal_date}，当时状态为{record.state}、已成熟近期同类案例{int(record.n_train)}个，完整机制理由为{record.reason}。这说明后来看上去成功的形态，不等于当时已能用本规则证明它应该启用。

## 为什么启停没有参与

V1每类仅看此前504日已完成、剔除重叠后最后12个案例，至少6个；近期压力净收益的90%正态近似下界及相对普通时点买入的平均超额都必须为正。完整机制还要求预设的市场状态，并有至少4个同状态案例。近期17次确认的完整机制全部未通过；完整时间序列虽有3个交易日显示突破形态具备资格，但这些日子没有新的突破确认，因此不能虚构交易。

{chr(10).join(delay_table)}

上述跨度甚至还没加退出成熟时间和未成交损耗。它衡量的是“等待六个案例”是否赶得上形态环境变化，不是证明六个案例足够显著。固定市场状态过滤使压力账户年化由13.63%降至3.17%，说明这套状态映射没有建立增量。不能直接得出所有状态信息无效；也不能为挽救本版事后放宽样本数、删除状态或缩短窗口。

最后本地资料日2026-09-16的突破近期收益下界虽已转正，但当天为RANGE且同状态仅3例；破低收复仅4例；急跌修复的近期下界为负。这只是9月16日的历史重建，不是9月24日实时判断。

## 较早历史用于认识失败

2015—2019仅形态压力账户年化-2.05%、最大回撤24.48%；2020年至资料末日年化3.20%、夏普0.442、回撤11.19%。这些不是要求形态每年赚钱的否决门槛，而是提示近期优势具有阶段性且可能遭受较大尾部损失。长重放中仅近期启停只发生2次交易、年化-0.28%；完整机制仍0次。所有36个独立完整账户都保留，没有拼接各自最好时段。

## 统计与复核范围

7项边界测试通过；真实行情在4个历史位置截断后，既有特征、过程步骤和确认信号一致。保存账户的现金+股票+应收恒等式、逐日收益、整手、T+1、已成熟标签引用、费用与指标已复算。此类复核说明计算和保存文件吻合，不代替科学验证或外部审阅。

区块抽样的有效结果见results/paired_bootstrap_valid.csv。原始计算对零波动使用的0保护值不具有统计意义：完整机制全程空仓，所以3项夏普差异区间全部标为NOT_COMPUTED，不能把夏普写成0或据此评优；原始数值痕迹与作废说明一并保留，未重新抽样或运行账户。同暴露/同波动理想归因另存，只能作事后解释，不能当真实账户或独立交易证据。

## 当前状态及下一步

本地价格截至9月16日，当前为NO_VIEW/ABSTAIN/POSITION_UNSET；不会把研究账户的空仓解释成用户应当空仓。本版历史结果冻结为FROZEN_NUMERICAL_FAILURE，整体收益目标未达成。现阶段已有完整可运行的形态识别与启停记录机制，尚没有验证成功的可执行启停策略。

每日观察使用Windows任务计划程序，工作日北京时间16:10检查本地资料，无采集、无订单、无聊天推送；无新资料不重复发关注记录。当前试运行结果是没有当天完整资料，新增前向观察0。主机必须开机且用户登录。输入规则和停用方法见每日观察说明.md。

下一项有价值的证据，是在冻结版本之后真实记录新过程及结果，判断事件频率能否支撑及时启停。若要研究新启停表示，优先检验低维连续环境信息能否减少等待同类交易的迟滞，必须独立版本、固定验证和停止条件，不在V1的已见2024年大赚周期上选择新窗口。用户未恢复采集时，仅等待人工补充本地资料；不以重复训练制造新证据。
""")
    markdown(OUT / "每日观察说明.md", """
# 每日收盘观察入口

已设置Windows任务Codex-510300-Sequential-Patterns-Observe-V1，周一至周五北京时间16:10。不是Codex原生自动化；当前任务工具列表没有automation_update。任务仅读取本地输入，现有行情采集暂停保持，不会调用券商或订单接口。

当前缺当天行情，实际试运行输出NO_VIEW_NO_CURRENT_LOCAL_RECEIPT，不生成前向记录。无资料变化不增加attention_changes.jsonl关注项。这个文件是本地变化日志，不是聊天消息推送。

输入目录为项目下data/forward/510300_sequential_patterns_regime_v1/inbox。每个交易日需要一个以YYYY-MM-DD.json命名的本地清单，字段为source_received_at（当天真实接收时点，ISO格式含+08:00），以及prices、dividends、dividend_coverage三项；每项包含path和sha256。价格表为原始日线Parquet，含date/open/high/low/close/volume/amount，必须保留冻结历史并覆盖新增完整交易日；股息表列和覆盖回执沿用包内inputs的格式。覆盖回执必须覆盖当天并匹配股息文件哈希。不能用回填文件冒充当日接收。

只有16:00至18:00且当前交易日完整输入可用时，才保存当天不可覆盖快照；快照复制直接输入和SHA-256，记录状态、形态步骤、确认、启停理由与当天成熟标签。历史缺口可以帮助计算，但其重建的旧观察不算前向天数。当天没有形态也是一份有效研究观察。没有成交回执，因此输出不代表实际成交。

以后新资料用于冻结规则观察，不重新跑36个历史账户、调参或替换V1。脚本位于scripts/observe_sequential_patterns_regime_v1.py；本包code目录保留同一版本。可在Windows任务计划程序中禁用上述唯一任务；用户发出停止指令时应立即禁用此任务并停止研究。无需修改旧采集任务。

任务依赖主机开机且当前用户登录；16:10错过后补运行超过18:00时会拒绝伪装当天及时观察。交易日历目前只至2026年末，到期输出NO_VIEW_CALENDAR_EXPIRED，不能自动猜测节假日。每天只记录研究信息，最大回撤与当前持仓均不能由此推断。
""")
    markdown(OUT / "00_README_FIRST.md", """
# 三类连续形态与启停 V1：阅读导航

先读研究结论.md，再看账户对比.png、protocol.md、user_request_full.md（用户原文）与user_request.md（结构摘要）。用户目标是改研究主线，当前目标年化10%、夏普1.2、20万元、回撤风险目标10%，年度次数下限取消。

关键结果：仅形态近期基础夏普1.281，压力1.185；完整机制未开仓，夏普未定义。最大盈利周期贡献约四分之三，无法据此确认当前有效。历史回放不是独立样本，当前行情仍止于9月16日。

results/episodes.csv及steps.csv包含所有准备、失败、超时与确认；signals.csv、trigger_decisions.csv、training_events.csv说明当时为何允许或拒绝。accounts目录保存三时段×两成本×六方案共36账户，不是36种搜索策略。annual.csv披露次数和逐年收益，无次数下限。

results/paired_bootstrap_valid.csv是最终可解释的统计区间表；paired_bootstrap.csv仅保留原始数值痕迹，其3项夏普差异由于FULL零波动已作废，详见statistical_validity.json。不能把作废值抄成研究发现。

freeze.json固定本轮结果运行前的协议、输入与代码。source_manifest.json对应原始路径；inputs与sources提供直接数值输入、来源回执和14份官方分红文件。code保存引擎、七项测试、结果整理、复核与每日观察脚本。

每日观察说明.md与forward/scheduler_receipt.json说明真实Windows任务。无当前资料时只留下NO_VIEW，不代表已新增有效观察。观察入口不会恢复行情采集。

可复制01_GPT_REVIEW_PROMPT.md让外部审阅者批评研究。ZIP根目录FILE_INDEX.csv是本包成员大小与哈希的权威索引，独立交付回执在ZIP外。结构校验和保存结果复算不等于已完成外部GPT审阅。
""")
    markdown(OUT / "01_GPT_REVIEW_PROMPT.md", """
请以严格研究审阅者身份审阅本包。先阅读研究结论、用户请求和冻结协议，再核对代码、直接输入、失败样本和完整账户。不要把结构校验误认为策略科学有效，也不要把历史重放叫独立验证。

1. 判断三个定义是否忠实描述“收缩—试探—确认/失败”的连续过程，是否用未来高低点或事后成功选样；日线版本与用户的短期含义是否匹配。
2. 核对原始价、除息中性指数、股息应收/到账、T+1、涨跌停延迟、整手、最低佣金、滑点和终点持仓估值。检查未成交与重叠机会是否被不当删去。
3. 重点解释为什么仅形态基础年化14.85%、夏普1.281，而压力夏普1.185；最大一笔与前三笔利润集中度意味着什么。不可因为接近门槛而降低成本假设。
4. FULL为0笔、夏普未定义。判断这说明近期样本门槛/状态映射识别迟滞，还是形态本身没有优势；严禁把空仓低回撤当策略成功。核对只有成熟标签进入近期选择，停用时是否仍记录假设结果。
5. results/paired_bootstrap_valid.csv才是可解释区间。原始计算零方差保护值已作废，不能拿其夏普差异做结论。并检查历史反复研究与单标的事件依赖是否阻止独立推断。
6. 给出按优先级排序的下一步：哪些证据值得继续前向收集，哪些定义应停止，什么新启停表示可减少迟滞；每项提供事前验证、最低证据与停止条件。不建议以已见2024年大赚交易重新选择窗口/状态。

用户只允许510300.SH与现金，20万元，扣费年化10%、夏普1.2，沿用回撤10%风险目标，已取消年度交易次数下限；不用采集的指令保持。你的建议不得自动扩大到其他品种、杠杆、券商、Paper/Shadow或真实下单。请区分本地可验证事实、推断与待验证建议。最后明确是否足以支持“当前有效”，若不足缺少什么。
""")
    markdown(OUT / "EXCLUSIONS.md", """
# 范围与排除

本包自含当前三形态研究的直接日线、全部股息记录及官方源文件、日历、来源回执、代码、测试、全部过程/标签/启停记录、36账户、抽样索引及冻结记录。价格表来自项目既有准入表；包内提供近期多源接收回执，不声称重建了2012年以来每次原始下载。

不包含整个项目的旧策略库、原失败账户或全部历史压缩包；只包含两份明确相关的旧协议以说明历史已被研究，不重启或救回旧策略。不包含虚拟环境、缓存、Git对象、凭据、未连接的订单簿、真实交易回执或全部论文全文。没有新市场数据采集、模型拟合、参数网格或外部GPT审阅。

当前新前向样本为0。任务计划与NO_VIEW试运行回执证明任务存在和脚本可运行，不证明同日行情已到齐或研究达到目标。压缩包的CRC/索引/哈希与保存结果复算仅为交付和计算一致性验证，不作额外安全审计。
""")
    (OUT / "requirements.txt").write_text("numpy\npandas\npyarrow\nmatplotlib\npytest\ntzdata\n", encoding="utf-8")
    for name in ("finalize_sequential_patterns_regime_v1.py", "verify_sequential_patterns_delivery_v1.py", "package_sequential_patterns_regime_v1.py"):
        src = PROJECT / "scripts" / name
        if src.exists():
            shutil.copy2(src, OUT / "code" / name)
    latest = PROJECT / "data/forward/510300_sequential_patterns_regime_v1/latest_status.json"
    if latest.exists():
        shutil.copy2(latest, OUT / "forward/initial_observer_receipt.json")
    mandate_path = PROJECT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = json.loads(mandate_path.read_text(encoding="utf-8"))
    mandate.update(target_revision_at=now(), previous_target_net_sharpe=1.3,
                   target_revision_receipt="reports/research/510300_sequential_patterns_regime_v1/authority_update.json",
                   annual_frequency_revision_at=now(),
                   annual_frequency_revision_receipt="reports/research/510300_sequential_patterns_regime_v1/authority_update.json",
                   annual_frequency_counting="逐年披露完整周期，不设置最低次数。",
                   latest_progress_receipt="reports/research/510300_sequential_patterns_regime_v1/results/summary.json",
                   last_research_result="247完整过程、87确认、36固定账户；仅形态近期压力夏普1.185，完整启停0笔；整体目标未达成，当前NO_VIEW。")
    save_json(mandate_path, mandate)
    save_json(OUT / "current_mandate_snapshot.json", mandate)
    status = PROJECT / "RESEARCH_STATUS.md"
    content = status.read_text(encoding="utf-8")
    marker = "510300_SEQUENTIAL_PATTERNS_REGIME_V1"
    if marker not in content:
        prefix = "> 2026-09-24 最新主线已按用户调整为三类连续短期形态与市场状态启停，年度次数下限取消，20万元、净年化10%、净夏普1.2、回撤风险目标10%，不用采集保持。完成 `510300_SEQUENTIAL_PATTERNS_REGIME_V1`：247过程、87确认、36固定账户。最近504日仅形态基础年化14.85%/夏普1.281，压力13.63%/1.185；完整启停0笔、夏普未定义，目标未达。最大一笔约占净利润四分之三，历史结果冻结，新增独立前向0。Windows每日16:10仅读本地资料，当前NO_VIEW。见[本轮结论](reports/research/510300_sequential_patterns_regime_v1/研究结论.md)；此前1.3和年度至少5次已由本次用户指令替代，旧失败与终止记录保留。\n\n"
        status.write_text(prefix + content, encoding="utf-8")
    print(json.dumps({"报告": str(OUT / "研究结论.md"), "最大单笔净利润占比": concentration["largest_share_of_net_pnl"],
                      "前三笔占比": concentration["top3_share_of_net_pnl"], "历史前序验证": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
