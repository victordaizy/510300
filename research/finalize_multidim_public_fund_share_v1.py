"""公募份额联合评分的结果解释、模型路径及固定标签复核。"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multidim_public_fund_share_score_v1 as study


ROOT, OUT = study.ROOT, study.OUT
FIGURE = "基金份额规模与联合高分结果.png"


def path(tree, row):
    node, conditions = 0, []
    while tree["children_left"][node] != -1:
        feature = tree["features"][tree["feature"][node]]
        threshold = tree["threshold"][node]
        left = float(np.float32(row[feature])) <= threshold
        conditions.append({"feature": feature, "value": float(row[feature]), "threshold": threshold, "branch": "<=" if left else ">"})
        node = tree["children_left"][node] if left else tree["children_right"][node]
    return {"leaf": node, "conditions": conditions, "prediction": tree["value"][node], "training_n": tree["samples"][node]}


def interpret(predictions, events, sources):
    by_month = events.set_index("stat_month")
    models = {r["stat_month"]: r for r in study.read(OUT / "saved_models.json")}
    errors, all_paths, high = [], [], []
    for _, row in predictions.iterrows():
        model = models[row.stat_month]
        pool = by_month.loc[model["training_months"]]
        if not pool.exit_at.le(pd.Timestamp(row.decision_at)).all() or not pool.state_idx.ge(row.state_idx - 504).all():
            raise ValueError("保存的成熟训练池不符。")
        paths = {name: path(model[name+"_tree"], row) for name in ["state", "joint"]}
        for name in paths:
            errors.append(abs(paths[name]["prediction"] - row[name+"_prediction"]))
        all_paths.append({"stat_month": row.stat_month, **paths})
        if row.joint_score >= 80 and row.joint_prediction > 0:
            current_path = paths["joint"]
            members = pool.loc[[path(model["joint_tree"], r)["leaf"] == current_path["leaf"] for _, r in pool.iterrows()]]
            biggest = members.actual_net5.idxmax()
            remainder = members.drop(index=biggest)
            if abs(members.actual_net5.mean() - row.joint_prediction) > 1e-12:
                raise ValueError("训练叶均值与预测不一致。")
            high.append({"stat_month": row.stat_month, "entry_date": row.entry_date, "exit_date": row.exit_date,
                         "score": row.joint_score, "prediction": row.joint_prediction, "actual_net5": row.actual_net5,
                         "path": current_path, "training_members": members[["entry_date", "exit_date", "actual_net5", *study.NEW]].reset_index().to_dict("records"),
                         "largest_label_stat_month": biggest, "largest_label_net_return": float(members.loc[biggest, "actual_net5"]),
                         "largest_label_share_of_leaf_sum": float(members.loc[biggest, "actual_net5"] / members.actual_net5.sum()),
                         "remaining_saved_members_mean": float(remainder.actual_net5.mean()),
                         "diagnostic_scope": "固定保存叶子的均值贡献分解，不删除样本重训，不把剩余均值变成新预测。"})
    if max(errors) > 1e-12:
        raise ValueError("保存树与保存预测不一致。")
    summary = {}
    first, last = sources.iloc[0], sources.iloc[-1]
    for key in ["stock", "mixed"]:
        summary[key] = {"start_month": first.stat_month, "end_month": last.stat_month,
                        "start_shares_yi": first[key+"_shares_yi"], "end_shares_yi": last[key+"_shares_yi"],
                        "start_assets_yi": first[key+"_assets_yi"], "end_assets_yi": last[key+"_assets_yi"],
                        "shares_change": last[key+"_shares_yi"] / first[key+"_shares_yi"] - 1,
                        "assets_change": last[key+"_assets_yi"] / first[key+"_assets_yi"] - 1,
                        "unit_assets_change": last[key+"_unit_asset"] / first[key+"_unit_asset"] - 1,
                        "positive_months": int(sources[key+"_shares_yi"].gt(sources[key+"_previous_shares_yi"]).sum()),
                        "negative_months": int(sources[key+"_shares_yi"].lt(sources[key+"_previous_shares_yi"]).sum())}
    result = {"completed_at": study.common.now(), "source_stock_mixed_endpoints": summary,
              "all_high_paths": high, "largest_training_label_is_same_event_for_all_high_paths": len(set(r["largest_label_stat_month"] for r in high)) == 1,
              "saved_tree_prediction_max_error": max(errors), "new_fit_or_label_change": False}
    study.save("逐事件模型实际路径.json", all_paths)
    study.save("份额规模与高分来源解释.json", result)
    return result


def verify_labels(events):
    prices = pd.read_parquet(study.DAILY, columns=["date", "open"])
    prices["date"] = pd.to_datetime(prices.date).dt.tz_localize(None)
    prices = prices.set_index("date")
    div = pd.read_csv(study.DIV)
    div["record_date"] = pd.to_datetime(div.record_date)
    q, tick, fee = Decimal(10000), Decimal("0.001"), Decimal("0.0004")
    checks = []
    for _, r in events.iterrows():
        begin, end = pd.Timestamp(r.entry_date), pd.Timestamp(r.exit_date)
        buy = (Decimal(str(prices.at[begin, "open"])) * Decimal("1.001") / tick).to_integral_value(rounding=ROUND_CEILING) * tick
        sell = (Decimal(str(prices.at[end, "open"])) * Decimal("0.999") / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
        dividend = sum((Decimal(str(v)) for v in div.loc[div.record_date.ge(begin) & div.record_date.lt(end), "cash_dividend_per_share"]), Decimal(0))
        paid = q * buy + max(Decimal(5), q * buy * fee)
        received = q * (sell + dividend) - max(Decimal(5), q * sell * fee)
        value = float(received / paid - 1)
        error = abs(value - r.actual_net5)
        if error > 1e-12 or len(prices.loc[begin:end])-1 != 5:
            raise ValueError("固定五日标签取价不符。")
        checks.append({"stat_month": r.stat_month, "entry": r.entry_date, "exit": r.exit_date, "recomputed_net5": value, "saved_error": error})
    pd.DataFrame(checks).to_csv(OUT / "保存标签取价费用复核.csv", index=False, encoding="utf-8-sig")
    return max(r["saved_error"] for r in checks)


def draw(sources, opportunities):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.9), gridspec_kw={"width_ratios": [1.08, 1]})
    fig.patch.set_facecolor("#f6f8fa")
    dates = pd.to_datetime(sources.stat_month)
    for name, label, color, style in [("mixed_shares_yi", "混合基金份额", "#bf5338", "-"), ("mixed_assets_yi", "混合基金资产净值", "#bf5338", "--"),
                                       ("stock_shares_yi", "股票基金份额", "#167789", "-"), ("stock_assets_yi", "股票基金资产净值", "#167789", "--")]:
        values = sources[name] / sources[name].iloc[0] * 100
        axes[0].plot(dates, values, label=label, color=color, linestyle=style, linewidth=2)
        axes[0].annotate(f"{values.iloc[-1]:.1f}", (dates.iloc[-1], values.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=color)
    axes[0].set_title("份额与资产规模给出不同的信息", loc="left", fontsize=15, weight="bold", pad=15)
    axes[0].set_ylabel("2023年9月 = 100")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_ylim(60, 235)
    axes[0].margins(x=.09)
    axes[0].tick_params(axis="x", rotation=25)
    selected = opportunities[opportunities.model.eq("joint")]
    x = np.arange(len(selected))
    first = axes[1].bar(x-.17, selected.prediction*100, width=.32, label="当时模型预测", color="#738aab")
    second = axes[1].bar(x+.17, selected.net_return*100, width=.32, label="实际五日净收益", color="#bf5338")
    axes[1].bar_label(first, labels=[f"{v:+.2f}%" for v in selected.prediction*100], padding=4)
    axes[1].bar_label(second, labels=[f"{v:+.2f}%" for v in selected.net_return*100], padding=4)
    axes[1].axhline(0, color="#607080", linewidth=.8)
    axes[1].set_ylim(-3.6, 5.8)
    axes[1].set_xticks(x, selected.entry_date.to_list())
    axes[1].set_title("三个联合高分机会均亏损", loc="left", fontsize=15, weight="bold", pad=15)
    axes[1].set_ylabel("五日收益 / %")
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for ax in axes:
        ax.set_facecolor("#f6f8fa")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.15)
    fig.text(.065, .03, "左图是官方存量端点，份额变化不等于纯现金申赎；右图为14次顺序评分中的全部三个高分机会。\n固定10000份、次开盘入场、五个开盘间隔退出；佣金万四/边、最低5元，滑点千一/边。非完整账户收益。", fontsize=9, color="#4b5d70")
    fig.subplots_adjust(left=.065, right=.98, top=.86, bottom=.19, wspace=.23)
    fig.savefig(OUT / FIGURE, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(result, review, sources, predictions, opportunities):
    info = review["source_stock_mixed_endpoints"]
    values = []
    for key, name in [("stock", "开放式股票基金"), ("mixed", "开放式混合基金")]:
        r = info[key]
        values.append(f"| {name} | {r['start_shares_yi']:,.2f}→{r['end_shares_yi']:,.2f} | {r['shares_change']:+.2%} | {r['assets_change']:+.2%} | {r['unit_assets_change']:+.2%} |")
    scores = []
    for row in result["opportunity_summary"]:
        label = "八项原状态" if row["model"] == "state" else "增加两类基金份额的十变量联合树"
        scores.append(f"| {label} | {row['n']} | {row['win_rate']:.0%} | {row['mean_net5']:+.2%} |")
    paths = []
    for r in review["all_high_paths"]:
        c = r["path"]["conditions"][0]
        paths.append(f"| {r['entry_date']} | {r['exit_date']} | {r['score']:.2f} | {r['prediction']:+.2%} | {r['actual_net5']:+.2%} | 混合份额月变≤{c['threshold']:.2%} | {r['largest_label_share_of_leaf_sum']:.2%} |")
    source_rows = []
    for r in sources.to_dict("records"):
        source_rows.append(f"| {r['stat_month']} | {r['archive_date']} | {r['pdf_filename_date']} | {r['available_day']} | [官方原表]({r['url']}) |")
    mse = {r["model"]: r["mse"] for r in result["model_errors"]}
    sep = sources[sources.stat_month.eq("2024-08")].iloc[0]
    text = f"""# 公募份额变化与指数联合评分：历史结果

新增26份基金业协会原始PDF，八项原状态与股票、混合基金份额月变一起进行浅树评分。12次预热后有14次严格按时间生成的预测，其中13次受新字段影响。但联合树MSE比相同事件、相同训练池的八变量对照增加 **{result['mse_relative_change']:.2%}**，三个联合高分机会全部亏损，平均五日净收益 **−1.99%**。当前表达拒绝封存，未进入完整账户。

这次存在有解释价值的新信息，但没有得到新的交易优势。输入十个维度不等于高分路径使用了十个维度：本轮三个联合高分实际都只沿“混合基金份额下降”的一个条件进入高收益叶子。

## 存量为什么要拆开

固定的同口径观察为2023年9月至2025年10月。股票、混合均取官方表内开放式基金部分，排除另列的封闭运作和定期开放基金。

| 类别 | 份额端点/亿份 | 份额累计变化 | 资产净值累计变化 | 平均每份资产累计变化 |
|---|---:|---:|---:|---:|
{chr(10).join(values)}

混合基金资产净值几乎持平，份额却下降23.01%。仅凭资产规模容易遗漏这一方向差异。[2023年9月原表]({sources.iloc[0].url})、[2025年10月原表]({sources.iloc[-1].url})。

进一步拆开：资产净值A=份额Q×平均每份资产v。每份报告内部的当月和上月，可精确写成“平均v×份额变化 + 平均Q×v变化”。全26个月的两项贡献保存在来源明细中。这个分解是存量恒等式，平均每份资产仍混有基金组合构成变化，不能视为纯投资收益；份额变化也含发行、申赎、再投、转换、拆并或分类因素，不能直接命名为现金净流入。

股票基金份额扩张、混合基金份额收缩，不能据此确定同一批资金从主动产品转到ETF。股票基金含多种产品，混合基金可持有股票和债券，基金经理也可先使用现金或调整其他资产，缺少逐基金流向和持仓不能识别对沪深300的被迫交易量。

## 多维联合评分的固定比较

控制组使用订单水平、订单月变、资金利率政策差、融资五日变化、融资买入活跃度、指数趋势强度、短长波动比和成交活跃度。联合组只增加两项当月份额变化，未增加其他条件。

沿用深度2、每叶至少6条观察的决策树；每次只训练此前504个交易日内已经完成五日退出的月度事件，至少12次才评分。另列十变量岭回归alpha10和同池均值，未搜索参数。26次公布保留全部，12次预热，14次评分；第一笔评分入场日为{result['first_scored_entry']}，最后一笔为{result['last_scored_entry']}。

各模型在共同14次观察上的MSE：八变量树{mse['state']:.6f}，十变量树{mse['joint']:.6f}，十变量线性{mse['linear']:.6f}，同池均值{mse['mean']:.6f}。联合树优于本次线性比较，却劣于八变量树和简单均值；不能据此称复杂关系已经学对。

分数是预测在当次训练池拟合值里的中位百分位，不是上涨概率。五个固定20分档均已保存，空档保留为空；高分机会固定为≥80分且预测净收益为正，前一事件退出前不重复。

| 模型 | 高分机会 | 胜率 | 平均五日净收益 |
|---|---:|---:|---:|
{chr(10).join(scores)}

控制组三笔入场为2025-01-21、2025-02-27、2025-03-28。联合组三笔为下表，只有一笔重合。虽然它避开了控制组2025年3月底那笔较大亏损，但新增两笔仍亏损，因此平均亏损变小不能当作正收益优势。

| 联合入场日 | 退出日 | 分数 | 当时预测净收益 | 实际净收益 | 高分的实际路径 | 最大训练样本占该叶收益合计 |
|---|---|---:|---:|---:|---|---:|
{chr(10).join(paths)}

![存量变化与高分兑现](<{str(OUT / FIGURE)}>)

## 高分为什么偏乐观

三个高分叶子最主要的正收益标签，全部是 **2024-09-25开盘至2024-10-09开盘的+22.43%**。这条标签来自2024年8月公募统计，[官方月报]({sep.url})在2024-09-24公布，和当时的政策行情处于同一时间段。

该单条标签占三个高分叶子训练收益合计的94.18%、91.10%、99.07%；它把每次叶均值抬到约3.2%—4.1%。高分路径分别只有6、6、7条训练观察，其余保存成员的净收益平均仅约0.28%、0.44%、0.04%。这是已保存叶子的均值贡献拆解，没有删样本重新拟合，也没有把剩余均值当作另一套预测。

这表明当前高分非常依赖一段集中上涨，尚不能把它归因于混合基金份额下降。月报描述的份额变化发生在此前统计月；实际卖出何时发生、是否持续、指数盈利预期和政策变化如何共同作用，不能由月末份额总量直接确定。后续三笔实际亏损进一步反对将这组高分直接用于账户。

## 时钟、费用与本轮决定

官方目录日期与PDF文件名日期取较晚者，在当日23:59:59后使用；其中2024年1月报告目录为2月23日、文件名为2月26日，本轮使用2月26日。文件名是保守版本线索，仍不等于有当年首次发布或接收回执。

2023年9月之前多个文件集中在2023-11-26生成，避免把它们拆成早期独立信号，本轮从该批次最新统计月开始。2025年11月原表注明取消封闭/开放分类，因此当前同口径研究截至2025年10月，不把新表接到旧表上。

公布后下一A股开盘入场、五个开盘间隔后退出；固定10000份、每边万四佣金（最低5元）、每边千一滑点、0.001元报价步长，包含取得的股息权益。长假会延长日历持有时间，表内日期如实保留。这里是每笔投入的标签收益，不是20万元完整账户。

本轮预先要求“共同期MSE不变差、联合高分净均值为正、至少新增一笔高分入场”同时成立，才做完整账户；前两项失败。状态为`REJECTED_FIXED_INCREMENT_NO_PARAMETER_RESCUE`，不调整阈值、变量、树深或持有期营救。原有失败家族、既有账户和全年至少5次的最终要求均保持不变；本轮夏普`NOT_COMPUTED`，总体目标未达到。

必要复核覆盖26个标签的取价、五日长度、费用和股息，以及28棵保存树的预测和成熟训练池。原表首期、日期差异期和末期已逐页查看。这些检查保证本轮计算与固定设定相符，不构成独立策略验证。

## 全部来源与文件

| 统计月 | 官方目录日期 | 文件名日期 | 本轮较晚可用日 | 原始PDF |
|---|---|---|---|---|
{chr(10).join(source_rows)}

原始PDF及文本位于`raw/`，来源计划和SHA256见`source_plan.json`、`parsed_sources.json`。`全部26个月的官方份额与存量分解.csv`保存六列原始数值和原文行；`全部事件状态及标签.csv`保存26次事件；`逐事件联合评分.csv`保存14次预测；`固定五档比较.csv`、`四项模型误差.csv`、`全部固定高分机会.csv`保存全部比较。

高分路径及其完整训练成员见`份额规模与高分来源解释.json`。`saved_models.json`保留所有树和线性系数。来源抓取曾遇到一次TLS连接中断，已确认进程结束后复用已下载文件补齐，未关闭证书校验；没有因此改动事件选择或收益窗口。
"""
    (OUT / study.REPORT_NAME).write_text(text, encoding="utf-8")


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("本轮已完成交付，不覆盖。")
    result = study.read(OUT / "result.json")
    if result["account_admitted"]:
        raise RuntimeError("已有账户准入，需先完成固定账户，不把中间结果作为终结。")
    predictions = pd.read_csv(OUT / "逐事件联合评分.csv")
    events = pd.read_parquet(OUT / "月度事件联合状态及标签.parquet")
    sources = pd.read_csv(OUT / "全部26个月的官方份额与存量分解.csv")
    opportunities = pd.read_csv(OUT / "全部固定高分机会.csv")
    review = interpret(predictions, events, sources)
    label_error = verify_labels(events)
    study.save("verification.json", {"completed_at": study.common.now(), "saved_label_windows": len(events), "max_label_error": label_error,
                                     "saved_tree_predictions": 2*len(predictions), "max_saved_tree_error": review["saved_tree_prediction_max_error"],
                                     "mature_training_pools_checked": len(predictions), "new_fits": 0, "new_accounts": 0,
                                     "pdf_pages_viewed": ["2023-09.pdf第1页", "2024-01.pdf第1页", "2025-10.pdf第1页"]})
    draw(sources, opportunities)
    write_report(result, review, sources, predictions, opportunities)
    summary = "新增26份公募官方原始月报、12次预热后14次十变量非线性评分。新字段改变13次预测，但MSE增加1.71%；三个联合高分全部亏损、平均-1.99%。高分叶收益合计的91%—99%来自同一2024年9月标签，表达拒绝，不进入账户。"
    next_question = "先核对已经保存的几项联合评分中，高分训练样本是否共享2024年9月同一政策行情标签；只做来源及贡献归因，不删样本重训、不拼账户。由此区分独立机制证据和同一涨幅被多个字段重复解释，再选择新信息来源。"
    result_path = str((OUT / "result.json").relative_to(ROOT)).replace("\\", "/")
    report_path = str((OUT / study.REPORT_NAME).relative_to(ROOT)).replace("\\", "/")
    prior = OUT / "prior_status"
    prior.mkdir(exist_ok=True)
    for file in ["510300_historical_cause_discovery_v1.json", "510300_existing_data_training_mandate_v1.json"]:
        p = ROOT / "config" / file
        (prior / file).write_bytes(p.read_bytes())
        obj = study.read(p)
        if "historical_cause" in file:
            obj.update(latest_completed_study=result_path, latest_report=report_path, updated_at=study.common.now(),
                       current_study=str((OUT / "protocol.json").relative_to(ROOT)).replace("\\", "/"), latest_result_summary=summary,
                       next_historical_question=next_question, latest_completed_branch_boundary="公募份额十变量表达拒绝封存，没有新账户或前瞻观点。")
        else:
            obj.update(current_round=study.STUDY, latest_progress_receipt=result_path, last_research_result=summary,
                       latest_continuation_report=report_path, latest_continuation_classification="PROGRESS_PUBLIC_FUND_SHARE_JOINT_SCORE_AND_CONCENTRATION",
                       current_driver_continuation_classification="PROGRESS_PUBLIC_FUND_SHARE_JOINT_SCORE_AND_CONCENTRATION",
                       current_driver_consecutive_blocked_goal_turns=0, next_research_question=next_question,
                       latest_historical_diagnostic_at=study.common.now(), latest_historical_report=report_path,
                       latest_historical_public_fund_share_score=result_path, goal_status="active", goal_achieved=False,
                       last_source_result="取得26份2023年9月至2025年10月AMAC官方同口径PDF，核对份额及资产净值当前和上月数，按目录与文件名较晚日期使用。")
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    study.save("delivery_receipt.json", {"completed_at": study.common.now(), "report": report_path, "result": result_path,
                                        "new_official_pdf_files": 26, "new_scored_events": 14, "original_model_fits": 42,
                                        "new_fits_in_interpretation": 0, "new_accounts": 0, "goal_achieved": False,
                                        "files": {p.name: study.sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in [".json", ".csv", ".parquet", ".md", ".png"]}})
    print("公募份额结果、集中度解释及图表已保存；原表达封存，目标继续。", flush=True)


if __name__ == "__main__":
    main()
