"""比较同一方法修正来源前后真实预测、进出场和完整账户，生成中文交付。"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity

OUT = ROOT / "deliverables/510300前瞻EPS来源修正与六策略进出场_20260907"
RESEARCH = ROOT / "reports/research/510300_forward_eps_source_v4_replay_comparison_v1"
STUDIES = {
    ("两机构组", "修正前"): "510300_forward_eps_two_institution_policy_v2",
    ("两机构组", "修正后"): "510300_forward_eps_two_institution_policy_v3",
    ("市盈率组", "修正前"): "510300_forward_eps_valuation_consistency_policy_v1",
    ("市盈率组", "修正后"): "510300_forward_eps_valuation_consistency_policy_v2",
}
NAMES = {
    "S1_POOLED_EPS": "两机构前瞻EPS三因子", "S2_MATCHED_SOOCHOW_EPS": "东吴前瞻EPS三因子",
    "S3_POOLED_EPS_INFORMATION_QUALITY": "两机构EPS加信息质量", "Q1_RELATION_QUALIFIED_PE": "关系相容市盈率",
    "Q2_MATCHED_RAW_REPORTED_PE": "共同月份原市盈率", "Q3_MATCHED_EPS_PROFIT_NO_PE": "前瞻EPS与利润修正，不使用PE",
    "C0_ORIGINAL_GUOSEN_EPS": "原国信EPS对照", "BUY_HOLD": "买入持有对照",
}
MODELS = list(NAMES)[:6]


def path(study):
    return ROOT / "reports/research" / study


def tagged(frame, family, version):
    frame = frame.copy()
    frame.insert(0, "组别", family)
    frame.insert(1, "来源版本", version)
    return frame


def render_chart(ledgers):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    for axis, model in zip(axes.flat, MODELS):
        family = "两机构组" if model.startswith("S") else "市盈率组"
        for version, color in [("修正前", "#9A650B"), ("修正后", "#176A90")]:
            frame = ledgers[(family, version, "BASE", model)]
            axis.plot(frame.date, frame.equity / 10000, label=version, color=color, linewidth=1.55)
        bh = ledgers[(family, "修正后", "BASE", "BUY_HOLD")]
        axis.plot(bh.date, bh.equity / 10000, color="#979DA4", alpha=.55, linewidth=.85, label="买入持有")
        axis.set(title=NAMES[model], ylabel="账户净值（万元）")
        axis.xaxis.set_major_locator(mdates.YearLocator(2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
        axis.grid(axis="y", alpha=.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("六个相同方法：只修正EPS来源，完整账户仍按原进出场规则运行", fontsize=14)
    target = OUT / "六方法来源修正前后完整净值.png"
    fig.savefig(target, dpi=150, facecolor="white")
    plt.close(fig)
    return target


def main():
    results, checks = {}, {}
    for key, study in STUDIES.items():
        results[key] = read(path(study) / "result.json")
        checks[key] = read(path(study) / "saved_numerical_verification.json")
        assert checks[key]["complete_accounts_checked"] == 10
    OUT.mkdir(parents=True, exist_ok=False)
    RESEARCH.mkdir(parents=True, exist_ok=False)
    save(RESEARCH / "protocol.json", {"registered_at": now(), "purpose": "保存的同方法修正前后账户归因，不新拟合或重抽",
         "full_account_and_all_years_retained": True, "month_extremes_are_post_result_description_only": True,
         "files": [identity(path(s) / "result.json") for s in STUDIES.values()] + [identity(Path(__file__))]}, exclusive=True)
    metrics, years, eras, trades, decisions, month_rows, forecasts, coverage_rows = [], [], [], [], [], [], [], []
    ledgers, chosen_map, feature_map = {}, {}, {}
    for (family, version), study in STUDIES.items():
        folder = path(study)
        result = results[(family, version)]
        metric_frame = pd.DataFrame(result["all_metrics"])
        metric_frame["策略"] = metric_frame.model.map(NAMES)
        metrics.append(tagged(metric_frame, family, version))
        years.append(tagged(pd.read_csv(folder / "yearly_metrics.csv"), family, version))
        eras.append(tagged(pd.read_csv(folder / "era_metrics.csv"), family, version))
        monthly_features = pd.read_parquet(folder / "monthly_features_and_mature_labels.parquet")
        feature_map[(family, version)] = monthly_features
        event_indices = set(np.flatnonzero(pd.read_parquet(folder / "signals.parquet").event_mask))
        receipts = read(folder / "training_receipts.json")["rows"]
        for model in [m for m in MODELS if m.startswith("S" if family == "两机构组" else "Q")]:
            selected = [r for r in receipts if r["model"] == model]
            valid = [r for r in selected if r["status"] == "TRAINED_MONTHLY_MODEL"]
            coverage_rows.append({"组别": family, "来源版本": version, "model": model, "策略": NAMES[model],
                "合格因子月数": result["valid_monthly_feature_origins"], "实际预测数": len(valid),
                "最早预测日": min((r["origin"] for r in valid), default=None), "最后预测日": max((r["origin"] for r in valid), default=None),
                "成熟标签数": int(monthly_features.Y60.notna().sum())})
            for record in selected:
                current = monthly_features.loc[monthly_features.origin.eq(pd.Timestamp(record["origin"]))].iloc[0]
                train = monthly_features.loc[monthly_features.origin.isin(pd.to_datetime(record["training_months"]))]
                forecasts.append({"组别": family, "来源版本": version, "model": model, "策略": NAMES[model],
                    "origin": record["origin"], "status": record["status"], "prediction": record["prediction"],
                    "saved_actual_return": current.Y60, "training_samples": record["training_samples"],
                    "training_mean_return": float(train.Y60.mean()) if len(train) else np.nan,
                    "training_latest_exit": record["latest_label_exit_date"]})
        for row in result["all_metrics"]:
            cost, model = row["cost"], row["model"]
            ledger = pd.read_parquet(folder / "evaluation" / cost / (model + "_ledger.parquet"))
            chosen = pd.read_parquet(folder / "evaluation" / cost / (model + "_decisions.parquet"))
            assert len(ledger) == len(chosen) == 1604
            np.testing.assert_allclose(ledger.pnl, ledger.price_pnl + ledger.dividend_recognized - ledger.commission - ledger.slippage_cost, atol=1e-6, rtol=0)
            ledgers[(family, version, cost, model)] = ledger
            chosen_map[(family, version, cost, model)] = chosen
            filled = ledger.loc[ledger.filled_quantity.ne(0)].copy()
            filled["策略"] = NAMES[model]; filled["model"] = model; filled["费用情景"] = cost
            filled["交易类别"] = np.select([filled.mark_clock.eq("OPEN_TERMINAL"), filled.filled_quantity.gt(0) & filled.shares_before.eq(0),
                filled.filled_quantity.gt(0), filled.shares.eq(0)], ["评价终点清算", "首次或重新进入", "加仓", "自主全部退出"], default="减仓")
            trades.append(tagged(filled, family, version))
            month_decisions = chosen.loc[chosen.origin_index.isin(event_indices)].copy()
            month_decisions["策略"] = NAMES[model]; month_decisions["model"] = model; month_decisions["费用情景"] = cost
            month_decisions.rename(columns={"mu5": "预测未来60交易日收益", "variance5": "估计60日收益方差"}, inplace=True)
            decisions.append(tagged(month_decisions, family, version))
            for month, group in ledger.groupby(ledger.date.dt.to_period("M")):
                month_rows.append({"组别": family, "来源版本": version, "费用情景": cost, "model": model, "策略": NAMES[model],
                    "月份": str(month), "净损益元": float(group.pnl.sum()), "价格持仓损益元": float(group.price_pnl.sum()),
                    "分红确认元": float(group.dividend_recognized.sum()), "费用元": float(group.commission.sum()+group.slippage_cost.sum()),
                    "月末净值元": float(group.equity.iloc[-1]), "平均持有份额": float(group.shares.mean()),
                    "平均股票占比": float(group.exposure.mean()), "成交次数": int(group.filled_quantity.ne(0).sum())})
    metric_frame = pd.concat(metrics, ignore_index=True)
    year_frame = pd.concat(years, ignore_index=True)
    era_frame = pd.concat(eras, ignore_index=True)
    trade_frame = pd.concat(trades, ignore_index=True)
    decision_frame = pd.concat(decisions, ignore_index=True)
    monthly = pd.DataFrame(month_rows)
    forecast = pd.DataFrame(forecasts)
    coverage = pd.DataFrame(coverage_rows)
    pairs = []
    for family, cost, model in [(f, c, m) for f in ["两机构组", "市盈率组"] for c in ["BASE", "STRESS"]
                                for m in MODELS if m.startswith("S" if f == "两机构组" else "Q")]:
        mask = monthly.组别.eq(family) & monthly.费用情景.eq(cost) & monthly.model.eq(model)
        before = monthly.loc[mask & monthly.来源版本.eq("修正前")].set_index("月份")
        after = monthly.loc[mask & monthly.来源版本.eq("修正后")].set_index("月份")
        assert before.index.equals(after.index)
        for month in before.index:
            b, a = before.loc[month], after.loc[month]
            gap = float(a.净损益元 - b.净损益元)
            price, dividend, fee_effect = float(a.价格持仓损益元-b.价格持仓损益元), float(a.分红确认元-b.分红确认元), float(b.费用元-a.费用元)
            np.testing.assert_allclose(gap, price+dividend+fee_effect, atol=1e-6, rtol=0)
            pairs.append({"组别": family, "费用情景": cost, "model": model, "策略": NAMES[model], "月份": month,
                "修正前月净损益元": b.净损益元, "修正后月净损益元": a.净损益元, "净损益差元": gap,
                "持仓价格损益差元": price, "分红确认差元": dividend, "费用变化影响元": fee_effect,
                "修正前平均股票占比": b.平均股票占比, "修正后平均股票占比": a.平均股票占比})
    pair_frame = pd.DataFrame(pairs)
    prediction_summaries = []
    for family, model in [(f, m) for f in ["两机构组", "市盈率组"] for m in MODELS if m.startswith("S" if f == "两机构组" else "Q")]:
        group = forecast.loc[forecast.组别.eq(family) & forecast.model.eq(model)]
        valid_sets = {v: set(g.loc[g.prediction.notna() & g.saved_actual_return.notna(), "origin"])
                      for v, g in group.groupby("来源版本")}
        common = valid_sets["修正前"] & valid_sets["修正后"]
        for version in ["修正前", "修正后"]:
            for scope, dates in [("各版本全部成熟预测", valid_sets[version]), ("两版本相同预测月", common)]:
                selected = group.loc[group.来源版本.eq(version) & group.origin.isin(dates)]
                error = selected.prediction - selected.saved_actual_return
                baseline_error = selected.training_mean_return - selected.saved_actual_return
                prediction_summaries.append({"组别": family, "model": model, "策略": NAMES[model], "来源版本": version, "范围": scope,
                    "成熟预测次数": len(selected), "方向正确次数": int(((selected.prediction>0)==(selected.saved_actual_return>0)).sum()),
                    "实际上涨次数": int(selected.saved_actual_return.gt(0).sum()), "预测均值": float(selected.prediction.mean()),
                    "实际收益均值": float(selected.saved_actual_return.mean()), "预测均方误差": float((error**2).mean()),
                    "当时训练均值基线误差": float((baseline_error**2).mean())})
    error_frame = pd.DataFrame(prediction_summaries)
    uncertainty_rows = []
    for family in ["两机构组", "市盈率组"]:
        old_folder = path(STUDIES[(family, "修正前")])
        new_folder = path(STUDIES[(family, "修正后")])
        old_config = read(ROOT / "config" / (STUDIES[(family, "修正前")] + ".json"))
        new_config = read(ROOT / "config" / (STUDIES[(family, "修正后")] + ".json"))
        for key in ["random_seed", "bootstrap_repetitions", "bootstrap_day_blocks", "annual_days"]:
            assert old_config[key] == new_config[key]
        for cost in ["BASE", "STRESS"]:
            old_returns = pd.read_parquet(old_folder / (cost + "_all_evaluation_returns.parquet"))
            new_returns = pd.read_parquet(new_folder / (cost + "_all_evaluation_returns.parquet"))
            pd.testing.assert_series_equal(old_returns.BUY_HOLD, new_returns.BUY_HOLD)
            pd.testing.assert_index_equal(old_returns.index, new_returns.index)
            assert len(old_returns) == len(new_returns) == 1604
            for block in [20, 60]:
                name = f"{cost}_block{block}_saved_bootstrap_statistics.parquet"
                old_draws, new_draws = pd.read_parquet(old_folder / name), pd.read_parquet(new_folder / name)
                assert len(old_draws) == len(new_draws) == old_config["bootstrap_repetitions"]
                derived = {}
                for model in [m for m in MODELS if m.startswith("S" if family == "两机构组" else "Q")]:
                    for statistic, column in [("年化平均收益差", "_minus_buy_hold"), ("夏普差", "_sharpe")]:
                        values = new_draws[model + column] - old_draws[model + column]
                        derived[model + "_" + statistic] = values
                        lower, upper = np.quantile(values.dropna(), [.025, .975])
                        uncertainty_rows.append({"组别": family, "model": model, "策略": NAMES[model], "费用情景": cost,
                            "区块交易日数": block, "指标": statistic, "下限": float(lower), "上限": float(upper),
                            "包含零": bool(lower <= 0 <= upper), "使用相同已保存区块次序": True})
                pd.DataFrame(derived).to_parquet(RESEARCH / f'{"S" if family == "两机构组" else "Q"}_{cost}_block{block}_paired_saved_differences.parquet', index=False)
    uncertainty_frame = pd.DataFrame(uncertainty_rows)
    frames = {"完整账户对照.csv": metric_frame, "逐年账户对照.csv": year_frame, "三个阶段账户对照.csv": era_frame,
              "全部实际进出场.csv": trade_frame, "全部预定月末判断.csv": decision_frame, "每月真实持仓与净损益.csv": monthly,
              "同方法逐月损益差归因.csv": pair_frame, "预测误差与共同月份对照.csv": error_frame,
              "覆盖训练与预测次数.csv": coverage, "全部保存预测与成熟实际收益.csv": forecast,
              "相同方法来源修正增量区间.csv": uncertainty_frame}
    for filename, frame in frames.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    table, attribution, examples, coverage_table, error_table, interval_table = [], [], [], [], [], []
    new_base_sharpes = []
    for model in MODELS:
        family = "两机构组" if model.startswith("S") else "市盈率组"
        selected = metric_frame.loc[metric_frame.组别.eq(family) & metric_frame.model.eq(model)]
        old = selected.loc[selected.来源版本.eq("修正前") & selected.cost.eq("BASE")].iloc[0]
        base = selected.loc[selected.来源版本.eq("修正后") & selected.cost.eq("BASE")].iloc[0]
        stress = selected.loc[selected.来源版本.eq("修正后") & selected.cost.eq("STRESS")].iloc[0]
        new_base_sharpes.append(float(base.net_sharpe))
        table.append(f'| {NAMES[model]} | {old.net_sharpe:.3f} | {base.net_sharpe:.3f} | {stress.net_sharpe:.3f} | {base.annualized_return:.2%} | {base.max_drawdown:.2%} | {base.trade_count} |')
        ci = uncertainty_frame.loc[uncertainty_frame.model.eq(model) & uncertainty_frame.费用情景.eq("BASE") & uncertainty_frame.指标.eq("年化平均收益差")].set_index("区块交易日数")
        interval_table.append(f'| {NAMES[model]} | {100*ci.loc[20,"下限"]:+.2f} 至 {100*ci.loc[20,"上限"]:+.2f} | {100*ci.loc[60,"下限"]:+.2f} 至 {100*ci.loc[60,"上限"]:+.2f} |')
        g = pair_frame.loc[pair_frame.model.eq(model) & pair_frame.费用情景.eq("BASE")]
        totals = g[["净损益差元", "持仓价格损益差元", "分红确认差元", "费用变化影响元"]].sum()
        attribution.append(f'| {NAMES[model]} | {totals.净损益差元:+,.2f} | {totals.持仓价格损益差元:+,.2f} | {totals.分红确认差元:+,.2f} | {totals.费用变化影响元:+,.2f} |')
        for extreme in [g.loc[g.净损益差元.idxmin()], g.loc[g.净损益差元.idxmax()]]:
            examples.append(f'| {NAMES[model]} | {extreme.月份} | {extreme.净损益差元:+,.2f} | {extreme.修正前平均股票占比:.1%} | {extreme.修正后平均股票占比:.1%} |')
        for version in ["修正前", "修正后"]:
            c = coverage.loc[coverage.model.eq(model) & coverage.来源版本.eq(version)].iloc[0]
            coverage_table.append(f'| {NAMES[model]} | {version} | {c.合格因子月数} | {c.成熟标签数} | {c.实际预测数} | {c.最早预测日} | {c.最后预测日} |')
            e = error_frame.loc[error_frame.model.eq(model) & error_frame.来源版本.eq(version) & error_frame.范围.eq("两版本相同预测月")].iloc[0]
            error_table.append(f'| {NAMES[model]} | {version} | {e.成熟预测次数} | {e.方向正确次数} | {e.预测均方误差:.6f} | {e.当时训练均值基线误差:.6f} |')
    chart = render_chart(ledgers)
    point_met = any(x >= 1.2 for x in new_base_sharpes)
    target_text = "出现基础费用下夏普达到1.2的历史点估计，仍未取得稳定超额的独立证据。" if point_met else "六种方法在修正来源后，完整账户夏普仍未达到1.2。"
    text = f"""# 510300前瞻EPS：来源修正、六种策略及完整进出场

2026年9月7日更新。{target_text}这是第22轮对六个已有方法的来源重算，原模型和交易规则保持不变。两组共十二条新账户，加八条复用对照，二十条评价记录；复用账户和来源重放不能当成独立实验。

## 本次发现并修正了什么

原PDF中“每股收益-最新股本摊薄（元/股）”这个明确字段没有被程序识别，使大量2022、2023年的前瞻EPS缺失。修正后，原4490份有效报告全部原样保留，新增1245份，合计5735份、17194条年度EPS。只修正字段名称识别，没有根据回测好坏改原件数值。

按目录实际发布日期，2022年有效报告由111增至606，2023年由零增至698，2024年由673增至725。2023年实际目录有863份；此前按编号前缀计为864份，两种年份口径已区分。第四版仍有999份保留缺失，其中577份被日期规则拒绝；随后发现部分是标题与日期抽在同一行，这一新问题另行记录，不偷偷混入本轮。

## 六个策略完整历史表现

统一20万元，2020年1月2日至2026年8月14日开盘终点，全部1604日和早期现金时间计入；242交易日年化。基础费用：佣金万分之二、最低5元，单边滑点万分之五；压力费用：佣金万分之四、最低5元，单边滑点千分之一。现金收益和夏普参考收益均为零。表中年化、回撤和成交次数为修正后基础费用；终点统一清算计入成交次数，不能称为自主卖点。

| 策略 | 修正前基础夏普 | 修正后基础夏普 | 修正后压力夏普 | 修正后年化收益 | 修正后最大回撤 | 成交次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(table)}

![六种方法的完整净值](<{chart.as_posix()}>)

## 数据补齐如何影响交易结果

下表把“修正后减修正前”的完整现金损益差分成三部分：不同持仓所产生的价格损益、确认的分红、费用变化的影响。三者相加等于净损益差，逐日和逐月都已核对。这能定位账户变化，不能把全部持仓差解释为某一个因子的独立因果贡献。

| 策略，基础费用 | 净损益差（元） | 持仓价格损益差（元） | 分红差（元） | 费用变化影响（元） |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(attribution)}

下表是来源修正前后年化平均日收益差的95%区间，单位为百分点；它与几何年化收益差不是同一个统计量。两版本的日期、区块算法、种子、次数及买入持有收益相同，按同一保存抽样次序将两版本“相对买入持有”统计相减，得到配对变化；没有重抽。区间包含零时，不能确认修正带来的收益增量稳定为正。

| 策略，基础费用 | 20日区块增量区间（百分点） | 60日区块增量区间（百分点） |
| --- | ---: | ---: |
{chr(10).join(interval_table)}

每种方法下列两行分别是事后定位出的最不利和最有利月份，便于查阅相应买卖记录；不是预设的市场阶段，也不用这些月份改交易规则。仓位是该月实际股票占净值的平均比例。

| 策略 | 月份 | 修正后减修正前净损益（元） | 原平均仓位 | 修正后平均仓位 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(examples)}

## 覆盖、训练和预测有没有变化

两个策略组各自共用月份。两机构组要求合并与东吴单机构均达到门槛；市盈率组只要求合并EPS、利润修正和相容PE达到门槛，所以两组不能直接视作完全同月对照。补齐来源自然改变训练月份及预测，模型参数保持不变。

| 策略 | 来源 | 合格因子月数 | 已成熟标签数 | 实际预测数 | 最早预测日 | 最后预测日 |
| --- | --- | ---: | ---: | ---: | --- | --- |
{chr(10).join(coverage_table)}

在修正前后都有预测且60日收益已经兑现的相同月末，下表比较误差。均方误差越低，预测数值越接近实际；它不是盈利、夏普或稳定性的替代指标。60日标签相互重叠，不能把每次都当独立样本。基线是该次预测时已经成熟训练样本的平均收益。

| 策略 | 来源 | 共同成熟预测数 | 方向正确数 | 预测均方误差 | 当时训练均值基线误差 |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(error_table)}

## 每个因子与进入、退出规则

"""
    rules = (ROOT / "docs/510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY.md").read_text(encoding="utf-8")
    rules = rules[rules.index("## 来源及三种固定方案"):]
    rules = rules.replace("原版本结果已经保存；本次来源重放登记时尚无新结果，目标仍未完成。", "原版本及本次来源重放均已保存，完整目标仍未被独立验证。")
    text += rules.replace("## ", "### ")
    text += "\n\n## 可逐笔检查的明细\n\n"
    for filename in frames:
        text += f'- [{filename}](<{(OUT / filename).as_posix()}>)\n'
    text += "\n本轮逐年和三个既定阶段结果全部保留在明细中。历史点估计、保存区块区间与独立验证是不同证据；本轮没有创建新独立留出期。来源和账户修正不会消除历史已经反复观察的问题。后续根据实际失效原因研究不同市场状态及多因子判断，目标继续保持夏普1.2及稳定超额。\n"
    document = OUT / "前瞻EPS六策略_来源修正结果与完整进出场.md"
    assert "```" not in text
    document.write_text(text, encoding="utf-8")
    save(RESEARCH / "result.json", {"completed_at": now(), "status": "SAVED_SOURCE_REPLAY_FORECAST_AND_ACCOUNT_COMPARISON_COMPLETE",
        "source_method_versions": 12, "full_evaluation_records": len(metric_frame), "new_accounts_generated": 0,
        "actual_trade_rows": len(trade_frame), "monthly_decision_rows": len(decision_frame),
        "paired_month_decompositions": len(pair_frame), "forecast_summaries": prediction_summaries,
        "same_seed_algorithm_return_dates_and_benchmark_paired_saved_draws": True,
        "source_change_intervals": uncertainty_rows,
        "new_base_sharpes": dict(zip(MODELS, new_base_sharpes)), "historical_point_target_met": point_met,
        "goal_achieved": False, "new_models_fit": 0, "new_random_draws": 0,
        "all_daily_and_monthly_cash_decompositions_verified": True,
        "files": [identity(OUT / name) for name in frames]}, exclusive=True)
    save(OUT / "delivery_receipt.json", {"created_at": now(), "document": identity(document), "chart": identity(chart),
        "csv_files": len(frames), "csv_row_counts": {k: len(v) for k, v in frames.items()},
        "visual_check_pending": True, "gpt_review_package_created": False,
        "comparison_result": identity(RESEARCH / "result.json")}, exclusive=True)
    print(json.dumps({"document": str(document), "chart": str(chart), "base_sharpes": dict(zip(MODELS, new_base_sharpes)),
        "point_target_met": point_met, "goal_achieved": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
