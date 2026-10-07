"""交付融资构成比较及已核实源数据纠正的影响。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multidim_financing_composition_v1 as study


ROOT, OUT = study.ROOT, study.OUT


def pct(x):
    return "未定义" if not np.isfinite(x) else f"{x:.2%}"


def main(figure_only=False):
    if (OUT / "交付结果.json").exists() and not figure_only:
        raise RuntimeError("融资构成结果已经交付，不覆盖。")
    d = pd.read_parquet(OUT / "全部日期的融资构成与价格状态.parquet")
    obs = pd.read_csv(OUT / "全部统一去重观察.csv")
    all_obs = pd.read_csv(OUT / "全部收缩观察.csv")
    grid = pd.read_csv(OUT / "全部八组合结果.csv")
    cases = pd.read_csv(OUT / "原五次高分的收缩构成.csv")
    periods = pd.read_csv(OUT / "总体收缩对照.csv")
    result = study.read(OUT / "result.json")
    checks = study.read(OUT / "必要计算核对.json")
    corr_dir = OUT / "data_correction"
    correction = study.read(corr_dir / "correction_receipt.json")
    recalculation = study.read(corr_dir / "recalculation_result.json")
    for row in grid.itertuples():
        source = obs if row.selection == "统一不重叠观察" else all_obs
        block = source[source.era.eq(row.era)] if "—" in str(row.era) else source[pd.to_datetime(source.entry_date).dt.year.eq(int(row.era))]
        block = block[block.composition.eq(row.composition) & block.price_condition.eq(row.price_condition)]
        assert len(block) == row.n
        if len(block):
            assert abs(block.net_label.mean() - row.mean_net5) < 1e-12
    assert (obs.entry_idx.to_numpy()[1:] > obs.exit_idx.to_numpy()[:-1]).all()
    old_monthly = ROOT / "reports/research/510300_multidim_money_surprise_score_v1"
    corrected_monthly = corr_dir / "monthly_score"
    before = pd.read_csv(old_monthly / "全部固定高分机会.csv")
    after = pd.read_csv(corrected_monthly / "全部固定高分机会.csv")
    pd.testing.assert_frame_equal(before, after)
    old_models = study.read(old_monthly / "saved_models.json")
    new_models = study.read(corrected_monthly / "saved_models.json")
    assert all(a[k] == b[k] for a,b in zip(old_models,new_models) for k in ["state_tree", "joint_tree"])
    old_daily = pd.read_parquet(study.DAILY)
    corrected_daily = pd.read_parquet(ROOT / correction["corrected_daily_path"])
    excluded = ["融资五日净变化", "融资买入活跃度", "valid_features"]
    pd.testing.assert_frame_equal(old_daily.drop(columns=excluded), corrected_daily.drop(columns=excluded))
    account_protocol = study.read(old_monthly / "account_protocol.json")
    files_checked = []
    for source in account_protocol["source_receipts"]:
        path = ROOT / source["path"]
        if path.name in ["price_features.parquet", "510300_dividends.csv", "account_engine.py"]:
            assert study.sha(path) == source["sha256"]
            files_checked.append(source["path"])
    assert len(files_checked) == 3
    account_equivalence = {
        "at": study.common.now(), "status": "UNCHANGED_EFFECTIVE_ACCOUNT_INPUTS",
        "reason": "原状态树及联合树完全一致，全部固定高分机会文件逐字段一致；账户实际读取的市场价格、分红、风险ES和引擎文件不变。纠正的两个融资特征不被账户模拟直接读取。",
        "signal_files_equal": True, "all_94_trees_equal": True,
        "market_columns_except_two_financing_features_and_feature_validity_equal": True,
        "risk_dividend_engine_files_hash_matched": files_checked,
        "extra_account_runs": 0,
        "primary_200k_net_sharpe": 1.2415538369005386,
        "account_result_reference": (old_monthly / "account_result.json").relative_to(ROOT).as_posix(),
        "not_a_new_independent_validation": True,
    }
    study.save("原账户有效输入一致性.json", account_equivalence)
    correction["old_monthly_and_ordinary_findings"] = "SAME_PARAMETER_RECOMPUTED; ORIGINAL_ACCOUNT_EFFECTIVE_INPUTS_UNCHANGED"
    (corr_dir / "correction_receipt.json").write_text(json.dumps(correction, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    main_grid = grid[grid.selection.eq("统一不重叠观察") & grid.era.eq("2024—2025")]
    primary_obs = obs[obs.era.eq("2024—2025")]
    primary_period = periods[periods.selection.eq("统一不重叠观察") & periods.era.eq("2024—2025")].iloc[0]
    same_direction = primary_obs.composition.isin([study.CAUSES[0], study.CAUSES[2]])
    stats_correction = pd.read_csv(corr_dir / "普通日期纠正前后比较.csv")
    ordinary = stats_correction[stats_correction.era.eq("2024—2025") & stats_correction.group.eq("普通日期") & stats_correction.selection.eq("不重叠高分机会")].iloc[0]
    corrected_matches = pd.read_csv(corr_dir / "event_state_control/相似日期汇总.csv")
    matched_high = corrected_matches[corrected_matches.era.eq("2024—2025") & corrected_matches.selection.eq("原固定高分事件")].iloc[0]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), gridspec_kw={"width_ratios": [1.05, 1.35]})
    values = np.full((4,2), np.nan)
    a = axes[0]
    for i, cause in enumerate(study.CAUSES):
        for j, price in enumerate(study.PRICES):
            row = main_grid[main_grid.composition.eq(cause) & main_grid.price_condition.eq(price)].iloc[0]
            values[i,j] = row.mean_net5 * 100
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e9ecef")
    a.imshow(values, cmap=cmap, vmin=-1.2, vmax=1.2, aspect="auto")
    a.set_xticks([0,1], ["价格变化改善", "价格变化未改善"])
    a.set_yticks(range(4), ["买入未增\n偿还未增", "买入未增\n偿还增加", "买入增加\n偿还增加", "买入增加\n偿还未增"])
    for i,cause in enumerate(study.CAUSES):
        for j,price in enumerate(study.PRICES):
            row = main_grid[main_grid.composition.eq(cause) & main_grid.price_condition.eq(price)].iloc[0]
            label = "无观察" if row.n == 0 else f"均值 {row.mean_net5:+.2%}\n{int(row.n)}次｜胜率 {row.win_rate:.0%}" + ("\n仅一个案例" if row.n == 1 else "")
            a.text(j,i,label,ha="center",va="center",fontsize=11,color="white" if row.n == 1 else "black")
    a.set_title("八个固定组合：净收益与样本数", loc="left", fontsize=13, pad=14)
    a.spines[:].set_visible(False)
    a = axes[1]
    y = np.arange(len(cases))
    buy = cases.buy_change5 / 5e8
    repay_contribution = -cases.repay_change5 / 5e8
    net = cases.net_change5 / 5e8
    a.barh(y-.18, buy, .32, color="#285f78", label="买入变化对净融资的贡献")
    a.barh(y+.18, repay_contribution, .32, color="#c48248", label="偿还变化对净融资的贡献")
    a.scatter(net, y, color="#202b33", s=38, zorder=4, label="净融资变化（两项合计）")
    a.set_yticks(y, pd.to_datetime(cases.entry_date).dt.strftime("%Y-%m-%d"))
    a.invert_yaxis()
    a.axvline(0, linewidth=.8, color="#666666")
    a.set_xlim(-540,410)
    a.set_xlabel("日均金额变化（亿元，相对前五个交易日）")
    a.set_title("原五次高分：融资收缩的构成不同", loc="left", fontsize=13, pad=14)
    a.spines[["top","right"]].set_visible(False)
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0,-.20), fontsize=10)
    fig.suptitle("融资净收缩为何不同｜2024—2025年历史发现",x=.04,ha="left",fontsize=18,fontweight="bold")
    fig.text(.04,.91,"48次统一去重观察，覆盖28段收缩；其中47次买入和偿还同向变化。净余额无法单独说明压力原因。",fontsize=12,color="#444444")
    fig.text(.04,.026,"左图为固定10000份五日成本后收益，不是账户。价格改善指近五日涨跌幅高于此前五日。隐含偿还不等于强平或股票卖出。",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.105,right=.97,top=.79,bottom=.30,wspace=.48)
    figure = OUT / "融资构成与价格变化_历史组合.png"
    fig.savefig(figure,dpi=155,facecolor="white")
    plt.close(fig)
    if figure_only:
        print("已更新图表布局，收益计算及规则未改变。", flush=True)
        return

    rows=[]
    for row in main_grid.itertuples():
        rows.append(f"| {row.composition} | {row.price_condition} | {row.n} | {pct(row.mean_net5)} | {pct(row.median_net5)} | {pct(row.win_rate)} |")
    case_rows=[]
    for row in cases.itertuples():
        case_rows.append(f"| {row.entry_date} | {row.buy_change5/5e8:+.2f} | {row.repay_change5/5e8:+.2f} | {row.net_change5/5e8:+.2f} | {pct(row.net_label)} |")
    year_rows=[]
    for row in periods[periods.selection.eq("统一不重叠观察")].itertuples():
        year_rows.append(f"| {row.era} | {row.n} | {row.episodes} | {pct(row.mean_net5)} | {pct(row.win_rate)} |")
    report_path = OUT / "融资收缩构成与价格变化_历史发现.md"
    report = f"""# 融资收缩构成与价格变化：历史发现

**主要发现：相同的融资余额下降可以来自不同的买入与偿还变化，但本轮固定组合尚未形成实用的净收益优势。** 2024—2025年48次统一去重观察覆盖28段融资收缩，平均五日净收益{pct(primary_period.mean_net5)}、胜率{pct(primary_period.win_rate)}。其中47次买入与隐含偿还同向变化，单看余额无法判断究竟是买入减得更多，还是偿还增得更多。

本轮还纠正了一条已确认的历史源数据错误，并用原参数复算受影响的此前评分。原月度树、五次主要期高分机会及账户实际输入不变；普通日期仍缺少净优势。

![融资构成与价格变化](<{figure.as_posix()}>)

## 八种组合的结果

先用当时已知的融资余额最近五日下降确定观察总体，按统一时序选择五日收益不重叠的日期，再分组。分组不看未来收益，也不为各组合重新挑一套更有利的入场时钟。

比较两个相邻五日窗口：融资买入增加或未增加 × 隐含偿还增加或未增加 × 价格变化改善或未改善。价格改善表示近五日财富收益高于此前五日；即使仍在下跌，也可能属于改善，不能直接命名为买盘承接。

| 融资构成变化 | 指数价格条件 | 观察数 | 平均五日净收益 | 中位数 | 胜率 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

样本较多的四格，均值介于-0.48%与+0.10%。另有一格仅一次盈利，三格没有观察；不能把一次成功写成可用的100%胜率。全部八格保留，没有根据结果倒置“价格改善”的方向。

阶段差异如下。2025年整体接近零，不能依靠55%的观察胜率认定收益足够；该阶段盈亏比约0.83。

| 阶段 | 不重叠观察数 | 不同收缩段数 | 平均净收益 | 胜率 |
|---|---:|---:|---:|---:|
{chr(10).join(year_rows)}

所有收益均为固定10000份、判断后次开盘进入、入场后第5个交易日开盘退出，沿用单边4bp佣金、最低5元、10bp滑点及价位取整，包含原权益登记口径。它们是机会观察，不是完整账户净值。48次观察只有28段收缩背景，不能视为48个独立冲击。

## 原五次高分为何不是同一种压力缓解

以下金额为日均亿元，比较判断时已知的最近五个统计交易日与此前五日。最后一列沿用原事件固定股数收益。

| 原入场日期 | 买入变化 | 隐含偿还变化 | 净融资变化：买入减偿还 | 后来五日净收益 |
|---|---:|---:|---:|---:|
{chr(10).join(case_rows)}

- 2024年7月的净收缩缓和来自买入增加得比偿还更多，两者都在上升。
- 2024年8月买入与偿还同时回落，买入回落略多，净收缩没有改善。
- 2024年9月的大盈利之前，买入日均减少103.72亿元，隐含偿还日均减少87.97亿元。净融资从此前日均小幅净增0.39亿元变为净减15.36亿元。这里并不存在“融资收缩已经缓和”的事前证据，后来的大涨不能倒推成当时已出现该机制。
- 2025年1月净收缩缓和，来自偿还下降得比买入更多。
- 2025年4月买入和偿还同时增加，但偿还增加更快，净收缩加深。

净余额下降、下降速度变化、买入与偿还的构成，是不同问题。上述分解说明当时的会计来源；资金持有人的动机、强制平仓占比、其他投资者承接等尚未识别。

上交所公式为融资余额变化等于融资买入减融资偿还；偿还还包含直接还款、卖券还款、强制平仓及权益调整。现有沪深两市汇总也不等于沪深300成分专属需求。[官方统计定义](https://www.sse.com.cn/market/othersdata/margin/detail/index.shtml)

## 本轮纠正的数据错误及影响

旧历史缓存把2024年8月8日深市融资余额、买入等金额记为0，导致合并余额仅余沪市、次日反推偿还为巨额负值。旧深市官方抽样没有覆盖这一日。新分解在生成组合收益结果以前遇到该异常并停止，随后定点取得深交所原始响应。

官方当天融资余额为6,649.54亿元、融资买入额为220.88亿元。原响应保存于`sources/szse_20240808.json`，新数据副本仅修正该日字段和由其产生的余额差分；官网以亿元保留两位小数，仍是历史回取而非当年首次版本。[深交所官方当日接口]({correction['official_url']})

纠正影响60个原评分日期：融资五日净变化两处，融资买入活跃度60处。其余价格、收益标签、宏观和分红不变。为此完成原47个事件同参数重算，共94棵树和47次线性基准重估，没有新增候选家族或参数搜索。

- 原94棵状态树与联合树逐项一致，全部固定高分机会文件逐字段一致。
- 原账户的行情、风险ES、分红、引擎及信号有效输入相同，原2024—2025年20万元账户夏普1.242保持；本轮通过有效输入一致性确认，未新增账户运行。
- 线性参考模型有13次预测变化，修正后的主要期MSE另存，不保留旧数值作为当前依据。
- 普通日期仍为35次，胜率37.14%；两次入场时点前移一个交易日，平均净收益由{ordinary.mean_net5_old:.4%}修正为{ordinary.mean_net5_corrected:.4%}，仍为负。
- 原五次事件的15个相似日期对照平均收益由约-0.97%修正为{matched_high.control_mean_net5:.2%}。匹配距离和所选日期有所变化，旧数值已由纠正副本替代；没有因此识别公告因果效果。

源数据纠正与收益调参是两件事。旧文件保留，不把旧融资策略失败重启为候选；未对其余全部历史实验逐项重算，后续研究须使用新来源指针。

## 与旧研究的区别及当前决定

旧T03“融资收缩减速＋急跌修复”和T05“高偿还＋价格韧性”已有未达标结果，本轮没有改窗口或退出重跑它们。已有2024年1—2月和2015/2018等政策案例的融资分解也保留。新增的是全部合格收缩日的三维固定组合比较，以及原五次高分的具体构成。

本轮没有选出可直接进入账户的组合，也没有把样本较好的格子改造成入场规则。现有证据支持先把融资收缩留在状态解释层；仅靠买入与偿还总量，尚未找到足以支持夏普和收益目标的条件优势。

进一步研究应先区分共同成交规模变化与融资资金参与占比，核对既有同类试验，避免把高度同向的买入与偿还当作两份独立证据。没有新信息增量时，不继续增加同一家族的筛选条件。

目标仍为完整账户成本后夏普至少1.2、年化至少10%、既定回撤与逐年次数要求。原局部账户点值不代表全部目标已实现。当前仍为历史研究，无前瞻观点、无交易授权。

## 结果文件

- `全部八组合结果.csv`：重叠和统一去重口径的所有八格、阶段及年份。
- `单维主效应对照.csv`：融资构成和价格变化各自的对照。
- `全部统一去重观察.csv`：121次完整记录，含主要期48次。
- `原五次高分的收缩构成.csv`：对应旧事件的金额、变化、时钟及原收益。
- `data_correction/correction_receipt.json`：官方纠正、原值、新值和影响范围。
- `data_correction/原月度评分纠正差异.csv`及`普通日期纠正前后比较.csv`：既有结论的纠正对照。
- `原账户有效输入一致性.json`：旧账户有效输入为何保持一致。

必要核对限于融资恒等式、输入时钟、统一去重、保存汇总和纠正影响；没有新参数搜索。
"""
    report_path.write_text(report, encoding="utf-8")
    checks.update(saved_group_rows_recomputed=len(grid), original_94_tree_structures_unchanged=True,
                  original_monthly_opportunity_files_equal=True, original_account_effective_inputs_equal=True)
    study.save("必要计算核对.json", checks)
    result.update(status="COMPLETED_FINANCING_COMPOSITION_NO_DEMONSTRATED_EDGE", finalized_at=study.common.now(),
                  report=report_path.name, figure=figure.name,
                  primary_observations=int(primary_period.n), primary_episodes=int(primary_period.episodes),
                  primary_mean_net5=float(primary_period.mean_net5), primary_win_rate=float(primary_period.win_rate),
                  primary_same_direction_flow_observations=int(same_direction.sum()),
                  new_fits=141, source_correction_refits=141, new_hypothesis_fits=0, new_candidate_families=0,
                  official_source_correction=(corr_dir/"correction_receipt.json").relative_to(ROOT).as_posix(),
                  original_account_metrics_unchanged=True, candidate_promoted=False)
    study.save("result.json", result)
    source_config = ROOT / "config/510300_financing_source_correction_20240808_v1.json"
    source_config.write_text(json.dumps({"updated_at":study.common.now(),"status":"USE_OFFICIAL_CORRECTED_COPY_FOR_NEW_RESEARCH",
        "corrected_margin_path":correction["corrected_margin_path"],"corrected_margin_sha256":correction["corrected_margin_sha256"],
        "corrected_daily_path":correction["corrected_daily_path"],"corrected_daily_sha256":correction["corrected_daily_sha256"],
        "receipt":result["official_source_correction"],"legacy_files":"保留原文件；未经纠正的旧融资输入不能作为新研究默认来源。",
        "scope":"仅一日官方确认的缓存零值纠正，非统计口径修订或策略调参。"},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    relative_result = (OUT / "result.json").relative_to(ROOT).as_posix()
    relative_report = report_path.relative_to(ROOT).as_posix()
    config_path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    config = study.read(config_path)
    config.update(latest_completed_study=relative_result, latest_report=relative_report,
                  current_study=(OUT/"protocol.json").relative_to(ROOT).as_posix(), updated_at=study.common.now(),
                  latest_result_summary="融资构成×价格变化八格：主要期48次、28段收缩，均值-0.11%、胜率45.83%；47次买入与偿还同向，常见组合未显净优势。修复2024-08-08深市零行，原94棵树及账户信号不变，普通日期纠正后仍负。",
                  latest_completed_branch_boundary="没有新增可用组合，不因个别格子或一次盈利推广；原账户夏普点值不变，完整目标未完成。",
                  next_historical_question="先去重既有融资参与占比研究，区分共同成交规模与杠杆资金参与份额；若没有新信息增量，停止追加融资收缩过滤器。新研究使用官方纠正副本。",
                  financing_source_override=source_config.relative_to(ROOT).as_posix(), goal_achieved=False)
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = study.read(mandate_path)
    mandate.update(latest_multidim_financing_composition=relative_result,latest_historical_report=relative_report,
                   latest_historical_diagnostic_at=study.common.now(),financing_source_override=source_config.relative_to(ROOT).as_posix(),
                   current_driver_continuation_classification="PROGRESS_FINANCING_COMPONENT_DISCOVERY_AND_SOURCE_CORRECTION",
                   current_driver_consecutive_blocked_goal_turns=0,
                   latest_completed_historical_branch_boundary=config["latest_completed_branch_boundary"])
    mandate_path.write_text(json.dumps(mandate,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    study.save("交付结果.json",{"at":study.common.now(),"status":"COMPLETED_PENDING_FIGURE_REVIEW","report":report_path.name,
                                "report_sha256":study.sha(report_path),"necessary_checks":checks,"goal_achieved":False,
                                "orders_authorized":False,"figure_visual_review":"PENDING"})
    print("融资构成报告与官方纠正影响已保存。",flush=True)
    print("主要期：",int(primary_period.n),"次，平均",pct(primary_period.mean_net5),"，胜率",pct(primary_period.win_rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交付融资构成结果或更新图表布局")
    parser.add_argument("--figure-only", action="store_true", help="只更新图表，不修改分组与收益")
    main(parser.parse_args().figure_only)
