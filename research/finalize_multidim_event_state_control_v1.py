"""保存公布时点对照的实际决策路径、简明图表及完整历史结论。"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multidim_event_state_control_v1 as study


ROOT, OUT = study.ROOT, study.OUT


def pct(x, digits=2):
    return "未定义" if not np.isfinite(x) else f"{x:.{digits}%}"


def decision_path(tree: dict, vector: np.ndarray):
    node, route, names = 0, [], []
    vector = vector.astype(np.float32)
    while tree["children_left"][node] != -1:
        index = tree["feature"][node]
        left = float(vector[index]) <= tree["threshold"][node]
        name = tree["features"][index]
        route.append(f"{name} {'<=' if left else '>'} {tree['threshold'][node]:.8f}")
        names.append(name)
        node = tree["children_left"][node] if left else tree["children_right"][node]
    return node, route, names


def main():
    if (OUT / "交付结果.json").exists():
        raise RuntimeError("公布时点对照已经交付，不覆盖。")
    protocol = study.read(OUT / "protocol.json")
    for source in protocol["source_receipts"]:
        assert study.sha(ROOT / source["path"]) == source["sha256"]
    daily, events, predictions, models, calendar = study.load_inputs()
    score = pd.read_csv(OUT / "逐日对照评分.csv")
    opportunities = pd.read_csv(OUT / "全部不重叠高分机会.csv")
    comparisons = pd.read_csv(OUT / "公布与普通日期比较.csv")
    matched = pd.read_csv(OUT / "逐公布事件的相似日期比较.csv")
    pairs = pd.read_csv(OUT / "全部历史相似日期.csv")
    result = study.read(OUT / "result.json")
    checks = study.read(OUT / "必要计算核对.json")

    # 直接核对保存的机会时序和汇总，避免用重叠标签冒充连续可参与次数。
    for _, block in opportunities.groupby(["era", "group"]):
        ordered = block.sort_values("entry_idx")
        assert (ordered.entry_idx.to_numpy()[1:] > ordered.exit_idx.to_numpy()[:-1]).all()
    for row in comparisons.itertuples():
        block = score[score.era.eq(row.era) & score.group.eq(row.group)]
        if row.selection == "全部高分标签":
            block = block[block.high_score]
        elif row.selection == "不重叠高分机会":
            block = opportunities[opportunities.era.eq(row.era) & opportunities.group.eq(row.group)]
        assert len(block) == row.n
        if len(block):
            assert abs(block.actual_net5.mean() - row.mean_net5) < 1e-12
            assert abs(block.actual_net5.median() - row.median_net5) < 1e-12
    for row in matched.itertuples():
        controls = pairs[pairs.stat_month.eq(row.stat_month)]
        assert len(controls) == 3
        assert abs(controls.control_net5.mean() - row.control_mean_net5) < 1e-12
        assert (pd.to_datetime(controls.control_exit_date) < pd.Timestamp(row.entry_date)).all()

    paths, leaf_records = [], []
    for model in models:
        event = predictions[predictions.stat_month.eq(model["stat_month"])].iloc[0]
        tree = model["state_tree"]
        leaf, route, used = decision_path(tree, event[study.FEATURES].to_numpy(float))
        pool = events[events.stat_month.isin(model["training_months"])].copy()
        pool["leaf"] = [study.evaluate(tree, v)[1] for v in pool[study.FEATURES].to_numpy(float)]
        same = pool[pool.leaf.eq(leaf)]
        assert len(same) == tree["samples"][leaf]
        assert abs(same.actual_net5.mean() - event.state_prediction) < 1e-12
        paths.append({"stat_month": event.stat_month, "entry_date": event.entry_date,
                      "era": event.era, "score": event.state_score,
                      "high_score": bool(event.state_score >= 80 and event.state_prediction > 0),
                      "path_feature_count": len(set(used)), "path_features": ";".join(used),
                      "path": " 且 ".join(route), "leaf_training_events": len(same),
                      "leaf_training_mean": same.actual_net5.mean(),
                      "leaf_training_median": same.actual_net5.median(),
                      "leaf_training_win_rate": same.actual_net5.gt(0).mean(),
                      "realized_net5": event.actual_net5})
        for item in same.itertuples():
            leaf_records.append({"model_month": event.stat_month, "training_month": item.stat_month,
                                 "training_entry_date": item.entry_date, "training_exit_date": item.exit_date,
                                 "training_net5": item.actual_net5,
                                 "prediction_is_training_leaf_mean": event.state_prediction})
    path_table = pd.DataFrame(paths)
    primary_paths = path_table[path_table.era.eq("2024—2025") & path_table.high_score]
    assert len(primary_paths) == 5
    single_margin = primary_paths.path_features.eq("融资五日净变化") & primary_paths.path_feature_count.eq(1)
    path_table.to_csv(OUT / "全部公布日实际决策路径.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(leaf_records).to_csv(OUT / "全部叶节点训练事件.csv", index=False, encoding="utf-8-sig")
    annual_rows = []
    for group in [study.EVENT, study.ORDINARY, study.OTHER]:
        for year in range(2021, 2026):
            block = opportunities[opportunities.group.eq(group) & pd.to_datetime(opportunities.entry_date).dt.year.eq(year)]
            annual_rows.append({"group": group, "entry_year": year, **study.summary(block.actual_net5)})
    annual = pd.DataFrame(annual_rows)
    annual.to_csv(OUT / "逐年不重叠机会.csv", index=False, encoding="utf-8-sig")
    checks.update(saved_summary_rows_checked=len(comparisons), matched_event_means_checked=len(matched),
                  saved_decision_paths_checked=len(path_table), high_primary_paths_using_only_margin=int(single_margin.sum()),
                  old_source_files_unchanged=True, opportunity_intervals_nonoverlapping_within_period_and_group=True)
    study.save("必要计算核对.json", checks)

    primary = comparisons[comparisons.era.eq("2024—2025") & comparisons.selection.eq("不重叠高分机会")].set_index("group")
    ordinary, event_row = primary.loc[study.ORDINARY], primary.loc[study.EVENT]
    matched_high = matched[matched.era.eq("2024—2025") & matched.high_score]
    raw_high = comparisons[comparisons.era.eq("2024—2025") & comparisons.group.eq(study.ORDINARY) & comparisons.selection.eq("全部高分标签")].iloc[0]

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1, 1.35]})
    colors = ["#205d7a", "#bd6c36"]
    x = np.arange(2)
    a = axes[0]
    a.bar(x - .18, [event_row.mean_net5 * 100, ordinary.mean_net5 * 100], .34, color=colors[0], label="平均净收益")
    a.bar(x + .18, [event_row.median_net5 * 100, ordinary.median_net5 * 100], .34, color=colors[1], label="收益中位数")
    for pos, mean, median in zip(x, [event_row.mean_net5, ordinary.mean_net5], [event_row.median_net5, ordinary.median_net5]):
        for dx, value in [(-.18, mean), (.18, median)]:
            a.text(pos + dx, value * 100 + (.07 if value >= 0 else -.07), f"{value:.2%}", ha="center", va="bottom" if value >= 0 else "top", fontsize=11)
    a.set_xticks(x, ["公布日\n5次高分机会", "普通日期\n35次高分机会"])
    a.set_ylabel("五日成本后收益（%）")
    a.set_ylim(-1.1, 2.5)
    a.set_title("同一批已保存模型：普通日期未延续优势", loc="left", fontsize=13)
    a.axhline(0, color="#777777", linewidth=.8)
    a.legend(frameon=False, loc="upper right")
    a.spines[["top", "right"]].set_visible(False)
    a = axes[1]
    y = np.arange(len(matched_high))
    a.barh(y - .17, matched_high.event_net5 * 100, .32, color=colors[0], label="公布日实际收益")
    a.barh(y + .17, matched_high.control_mean_net5 * 100, .32, color=colors[1], label="此前3个相似日期平均")
    a.set_yticks(y, pd.to_datetime(matched_high.entry_date).dt.strftime("%Y-%m-%d"))
    a.invert_yaxis()
    for pos, value in zip(y, matched_high.event_net5):
        a.text(value * 100 + (.07 if value >= 0 else -.07), pos - .17, f"{value:.2%}", ha="left" if value >= 0 else "right", va="center", fontsize=10)
    a.axvline(0, color="#777777", linewidth=.8)
    a.set_xlim(-2.6, 9.1)
    a.set_xlabel("五日成本后收益（%）；按原入场日期排列")
    a.set_title("相似状态仍不足以解释公布日收益", loc="left", fontsize=13)
    a.legend(frameon=False, loc="lower right")
    a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("2024—2025｜公布时点与指数状态的历史对照", x=.055, ha="left", fontsize=18, fontweight="bold")
    fig.text(.055, .91, "五次公布日高分的实际路径都只使用融资收缩条件；八项输入尚未形成共同判断。", fontsize=12, color="#444444")
    fig.text(.055, .035, "固定10000份、持有5个交易日、计入原压力成本。各组单独去重；不是完整账户。匹配差异不等于公告因果效应。", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.07, right=.97, bottom=.18, top=.8, wspace=.45)
    figure = OUT / "公布时点与指数状态_历史对照.png"
    fig.savefig(figure, dpi=160, facecolor="white")
    plt.close(fig)

    compare_lines = []
    for era in ["2021—2023", "2024—2025"]:
        for group in [study.EVENT, study.ORDINARY]:
            row = comparisons[comparisons.era.eq(era) & comparisons.group.eq(group) & comparisons.selection.eq("不重叠高分机会")].iloc[0]
            compare_lines.append(f"| {era} | {group} | {int(row.n)} | {pct(row.mean_net5)} | {pct(row.median_net5)} | {pct(row.win_rate)} | {row.return_payoff:.2f} |")
    match_lines = [f"| {r.entry_date} | {pct(r.event_net5)} | {pct(r.control_mean_net5)} | {r.mean_distance:.3f} | {r.controls} |" for r in matched_high.itertuples()]
    path_lines = [f"| {r.entry_date} | {r.path} | {r.leaf_training_events} | {pct(r.leaf_training_mean)} | {pct(r.realized_net5)} |" for r in primary_paths.itertuples()]
    annual_lines = [f"| {r.entry_year} | {r.n} | {pct(r.mean_net5)} | {pct(r.win_rate)} |" for r in annual[annual.group.eq(study.ORDINARY)].itertuples()]
    report = f"""# 公布时点与指数状态：原高分组合能否扩展到普通交易日

本轮结论：2024—2025年，原月度状态模型用于普通日期后的35次不重叠高分机会，平均五日净收益{pct(ordinary.mean_net5)}、胜率{pct(ordinary.win_rate)}。原公布日5次平均{pct(event_row.mean_net5)}，但上一轮2024—2025年账户利润的82.26%来自一笔。现有证据不足以通过扩大交易日期来满足收益与次数目标。

另一个关键发现是：虽然原模型输入八项指数状态，这五次实际高分路径都只使用了“融资五日净变化”。其他变量在模型其他分支可能有作用，但没有参与这五次入选。不能把输入维度多直接解释为这些机会已获得多因素共同支持。

![历史对照](<{figure.as_posix()}>)

## 计算范围与结果

本轮复用原47棵状态树，不拟合、不调参数；按当时已知模型完成{result['scored_dates']}个日期评分，其中{result['ordinary_dates']}个普通日期、47个原可评分公布日、7个其他公布日。原47个公布事件的评分与标签全部复现。

| 阶段 | 日期类别 | 不重叠高分次数 | 平均净收益 | 中位数 | 胜率 | 收益盈亏比 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(compare_lines)}

所有表中收益均为固定10000份、下一开盘买入、入场后第5个交易日开盘退出的成本后标签，沿用单边4bp佣金、最低5元、单边10bp滑点及整价位取整。它们不是账户收益，也不能年化成账户夏普。

普通日期在2024—2025年原有{int(raw_high.n)}条高分日标签，平均{pct(raw_high.mean_net5)}，包含同一轮走势的重叠窗口。按判断时序保留可先后参与的35次后，均值变为{pct(ordinary.mean_net5)}。所有重叠标签仍保留，但不当作127次独立机会。

按入场年份观察普通日期，有阶段差异；2025年小幅转正也不足以证明完整目标。

| 入场年份 | 不重叠高分次数 | 平均净收益 | 胜率 |
|---|---:|---:|---:|
{chr(10).join(annual_lines)}

2021年尚无达到80分的普通日机会，且第一棵原模型直到2021年7月才生成，不能把该年视作完整覆盖的零机会全年。

## 五次高分实际用了什么

| 原入场日期 | 实际经过的条件 | 对应成熟月度训练事件数 | 叶节点训练平均 | 后来实际净收益 |
|---|---|---:|---:|---:|
{chr(10).join(path_lines)}

这些融资下降阈值来自已保存的历史模型，本轮没有重新选择。五次路径均是一条融资收缩判断，不足以证明宏观、资金和量价存在有效交互。相同高分也不是校准后的上涨概率。

融资余额的下降只说明净融资变小，可能对应买入减少、偿还增加，或两者同时变化。上交所统计公式将余额变动分为融资买入与融资偿还，偿还又包含直接还款、卖券还款、强制平仓及权益调整。因此，本轮不能把融资收缩命名为强平结束或卖压已经出清。[上交所融资融券统计说明](https://www.sse.com.cn/market/othersdata/margin/detail/index.shtml)

## 在公布以前，类似状态的普通日期怎样

对每个公布事件，仅从此前504个交易日内已经完成五日收益观察的普通日期中，按八项状态的标准化距离取三个互不重叠日期。排名不读取收益。主要期五次高分共匹配15个唯一日期；对照平均{pct(matched_high.control_mean_net5.mean())}，公布事件平均{pct(matched_high.event_net5.mean())}，其中四次高于各自对照。

| 原事件入场 | 事件净收益 | 三个历史相似日期平均 | 八维平均距离 | 对照入场日期 |
|---|---:|---:|---:|---|
{chr(10).join(match_lines)}

距离是八维标准化差异的均方根，越低越接近；不是概率。全47事件共141次匹配使用、132个唯一日期，重复日期不能增加独立样本数。所有逐维差异保存在“八项状态匹配差异.csv”。

这仍不能证明公布本身产生了额外收益：对照来自之前的时间，匹配只控制八个可观察维度；普通日期还沿用最近模型，模型年龄最高62天，而原公布时点更新了模型。这些差异以及持有期间新信息都未被完全控制。

2024年8月货币报告实际于9月13日19:26公布，本轮按真实公布日和交易日历归属到9月18日开盘；不能把它归属为8月底已知信息。[央行原始公布页](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/41425ee3179143a29c4656e31b6df3d8/index.html)

后续9月政策信息和价格变化仍保留在原真实收益中。本轮没有删除大盈利、剔除新消息影响的日期或把后来政策搬回原判断时点。

## 当前结论与下一步

上一轮20万元账户在2024—2025年的净夏普1.242、年化2.00%、最大回撤0.52%保持不变；它仍未满足年化至少10%、每个完整年份至少5次自然交易的要求。主要期M2预期偏差没有改变任何原评分，原全部账户结论也不改写。

本轮新增证据反对直接将同一月度模型扩展到普通日期以凑足次数。普通日固定表达缺少正的平均净收益，因此本轮不新增账户，不把它包装成新可执行策略。

后续历史发现应先去重既有融资研究，区分融资买入萎缩、偿还增强及价格承接的组合是否真有不同收益。只能观察到净余额时保留原因未识别；不能为了形成“多维”外观，强制树模型选更多变量或加深树。

本轮是已研究历史上的归因和对照，不是独立验证或市场前瞻。目标继续保持未完成，当前观点NO_VIEW、仓位UNSET，无交易授权。

## 可复核文件

- “逐日对照评分.csv”：所有1082个日期、模型版本、模型年龄、评分及结果。
- “全部不重叠高分机会.csv”：全部时序机会，不隐藏亏损。
- “全部公布日实际决策路径.csv”：所有47次公布的真实路径。
- “全部叶节点训练事件.csv”：每个路径对应的成熟训练事件。
- “全部历史相似日期.csv”：141次对照使用及收益。
- “protocol.json”：计算前保存的范围、时钟、固定比较和局限。
- 代码：`{(ROOT/'research/multidim_event_state_control_v1.py').as_posix()}`。

仅做必要计算核对：原评分复现、模型可得时间、机会不重叠、保存汇总和训练叶均值一致。未新增参数搜索、拟合或完整账户。
"""
    report_path = OUT / "公布时点与指数状态_历史对照结果.md"
    report_path.write_text(report, encoding="utf-8")
    result.update(status="COMPLETED_STATE_TRANSFER_NO_NET_EDGE", finalized_at=study.common.now(),
                  report=report_path.name, figure=figure.name,
                  primary_ordinary_nonoverlap_count=int(ordinary.n), primary_ordinary_mean_net5=float(ordinary.mean_net5),
                  primary_ordinary_win_rate=float(ordinary.win_rate),
                  primary_high_paths_using_only_margin=int(single_margin.sum()),
                  ordinary_date_expansion_supported=False, causal_announcement_increment_established=False,
                  previous_account_results_unchanged=True)
    study.save("result.json", result)
    relative_result = (OUT / "result.json").relative_to(ROOT).as_posix()
    relative_report = report_path.relative_to(ROOT).as_posix()
    next_question = "先去重既有融资原因研究，辨别买入萎缩、偿还增强和指数价格承接的组合；原五次高分实际只有融资收缩单条件，不能直接扩展到普通日期或强制增加树深。"
    config_path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    config = study.read(config_path)
    config.update(latest_completed_study=relative_result, latest_report=relative_report,
                  current_study=(OUT / "protocol.json").relative_to(ROOT).as_posix(), updated_at=study.common.now(),
                  latest_result_summary="复用47个原月度状态模型，共1082个日期评分。主要期普通日35次不重叠高分平均净收益-0.23%、胜率37.14%；原公布日5次+1.90%。五次入选实际只用融资收缩单条件，尚未证明多因素交互；不扩频、不新增账户。",
                  latest_completed_branch_boundary="普通日期固定表达没有正净收益，不能据原事件期夏普扩频；公布时点的因果增量未识别。",
                  next_historical_question=next_question, goal_achieved=False)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = study.read(mandate_path)
    mandate.update(latest_multidim_event_state_control=relative_result, latest_historical_report=relative_report,
                   latest_historical_diagnostic_at=study.common.now(),
                   current_driver_continuation_classification="PROGRESS_EVENT_CLOCK_AND_STATE_SEPARATION",
                   current_driver_consecutive_blocked_goal_turns=0,
                   latest_completed_historical_branch_boundary="原月度状态树用于普通日期未呈现净优势；五个主要期高分路径均只用融资变化，不能宣称已发现有效多维交互。旧账户与目标未完成状态保持。")
    mandate_path.write_text(json.dumps(mandate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    study.save("交付结果.json", {"at": study.common.now(), "status": "COMPLETED_PENDING_FIGURE_REVIEW",
                               "report": report_path.name, "report_sha256": study.sha(report_path),
                               "necessary_checks": checks, "goal_achieved": False,
                               "orders_authorized": False, "figure_visual_review": "PENDING"})
    print("公布时点对照报告、真实路径和图表已保存。", flush=True)
    print(primary_paths[["entry_date", "path", "leaf_training_events"]].to_string(index=False))
    print("主要期普通日期：", int(ordinary.n), "次，平均", pct(ordinary.mean_net5), "，胜率", pct(ordinary.win_rate))


if __name__ == "__main__":
    main()
