"""复算冻结的六例延迟收益，报告数据缺口；数据门失败时不拟合或生成账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def analyse(study: Path, out: Path) -> None:
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((study / "freeze_receipt.json").read_text(encoding="utf-8"))
    for item in frozen["frozen_files"]:
        assert digest(study / item["path"]) == item["sha256"], item["path"]
    out.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(study / "inputs/event_information.csv")
    official = pd.read_csv(study / "inputs/official_releases.csv")
    market = pd.read_parquet(study / "inputs/market_daily.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.normalize()
    dates = pd.DatetimeIndex(market.date)
    assert dates.is_unique and dates.is_monotonic_increasing
    assert (market[["open", "high", "low", "close"]] > 0).all().all()
    assert ((market.low <= market.open) & (market.open <= market.high)).all()
    assert ((market.low <= market.close) & (market.close <= market.high)).all()
    # 新收益的选择时钟仅由发布时间和日历决定，不能用后续涨跌移动日期。
    records, paths = [], []
    for item in events.to_dict("records"):
        published = pd.Timestamp(item["published_at"]).tz_convert("Asia/Shanghai")
        opening = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        obs = int(opening.searchsorted(published, side="right"))
        entry, end = obs + 1, obs + protocol["primary_horizon_trading_days"]
        assert obs >= 21 and end < len(market)
        pre = int((dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).searchsorted(published, side="left")) - 1
        assert pre < obs
        p, o, e, z = market.iloc[pre], market.iloc[obs], market.iloc[entry], market.iloc[end]
        dividend = float(market.iloc[entry + 1:end + 1].dividend.sum())
        assert float(market.iloc[pre + 1:end + 1].dividend.abs().sum()) == 0, "本六例预期无分红，发现分红应人工核对登记/到账权利。"
        initial = float(o.close / p.close - 1)
        unavailable = float(e.open / p.close - 1)
        residual = float((z.close + dividend) / e.open - 1)
        total = float((z.close + dividend) / p.close - 1)
        assert np.isclose((1 + unavailable) * (1 + residual) - 1, total)
        old_end = pre + 5
        old_return = float(market.iloc[old_end].wealth / p.wealth - 1)
        sigma = float(market.iloc[obs - 19:obs + 1].total_simple.std(ddof=1))
        pre_sigma = float(market.iloc[pre - 19:pre + 1].total_simple.std(ddof=1))
        support = float(market.iloc[pre - 19:pre + 1].low.min())
        surprise = float(item["spread_surprise_proxy_pp"])
        record = dict(item, event_id="MONEY_" + item["stat_month"],
                      pre_date=str(p.date.date()), observation_date=str(o.date.date()),
                      observation_at=str(o.date.date()) + "T15:00:00+08:00",
                      entry_date=str(e.date.date()), entry_at=str(e.date.date()) + "T09:30:00+08:00",
                      exit_date=str(z.date.date()), exit_at=str(z.date.date()) + "T15:00:00+08:00",
                      old_end_date=str(market.iloc[old_end].date.date()),
                      pre_close=float(p.close), observation_close=float(o.close), entry_open=float(e.open), exit_close=float(z.close),
                      old_first_5d_return=old_return, initial_response=initial,
                      unavailable_to_entry_return=unavailable, residual_5d_gross_return=residual,
                      total_same_endpoint_return=total, earned_dividend_per_share=dividend,
                      price_pre20_return=float(p.wealth / market.iloc[pre - 20].wealth - 1),
                      price_known_sigma20=sigma, price_log_sigma20=math.log(sigma),
                      support_pre20_low=support, support_distance_known_sigma=(o.close / support - 1) / pre_sigma,
                      price_pre20_positive=bool(p.wealth > market.iloc[pre - 20].wealth),
                      first_reaction_same_sign=bool(surprise * initial > 0),
                      residual_same_sign=bool(surprise * residual > 0),
                      funds_status="NO_HISTORICAL_PUBLICATION_TIME_PROOF", case_role="HISTORICAL_DESCRIPTION_ONLY")
        records.append(record)
        path_points = [(pre, -1., float(p.close), "公告前收盘"),
                       (obs, 0., float(o.close), "观察日收盘"),
                       (entry, .65, float(e.open), "可执行开盘")]
        path_points.extend((k, float(k - entry + 1), float(market.iloc[k].close), "持有日收盘") for k in range(entry, end + 1))
        for k, x, price, role in path_points:
            paths.append({"event_id": record["event_id"], "stat_month": item["stat_month"], "date": str(market.iloc[k].date.date()),
                          "point_role": role, "plot_x": x, "price": price,
                          "return_from_pre_close": price / p.close - 1,
                          "return_from_entry_open": price / e.open - 1 if x >= .65 else None})
    results = pd.DataFrame(records)
    # 重叠区间合并，只记录聚类，不能制造额外独立样本。
    cluster, previous_end = 0, pd.Timestamp.min
    for i, row in results.sort_values("entry_date").iterrows():
        start, finish = pd.Timestamp(row.entry_date), pd.Timestamp(row.exit_date)
        if start > previous_end:
            cluster += 1
        previous_end = max(previous_end, finish)
        results.loc[i, "event_cluster"] = cluster
    results.event_cluster = results.event_cluster.astype(int)
    save_csv(results, out / "六次事件_观察后五日完整结果.csv")
    save_csv(pd.DataFrame(paths), out / "六次事件_全部路径点.csv")
    months = official[["stat_month", "published_at", "definition_version", "m1_yoy_pp", "m2_yoy_pp"]].copy()
    months["case_admitted"] = months.stat_month.isin(results.stat_month)
    months["case_status"] = np.where(months.case_admitted, "RECONSTRUCTED_CONSENSUS_CASE_ONLY", "NO_ADMITTED_PAIRED_PRERELEASE_CONSENSUS")
    months["funds_admitted"] = False
    months["strict_forward_immutable"] = False
    save_csv(months, out / "104个月完整准入清单.csv")
    # 仅展示固定价格下的整手费用。没有信号、行情路径或逐日账户收益。
    cost_rows = []
    for capital in (protocol["main_capital_cny"], protocol["cost_comparison_capital_cny"]):
        for fraction in (1., .25):
            for name, slip in (("BASE", .0005), ("STRESS", .001)):
                budget = capital * fraction
                buy_price, sell_price = 4 * (1 + slip), 4 * (1 - slip)
                quantity = int(budget // (buy_price * 100)) * 100
                while quantity * buy_price + max(5, .0002 * quantity * buy_price) > budget:
                    quantity -= 100
                buy_notional, sell_notional = quantity * buy_price, quantity * sell_price
                buy_fee, sell_fee = max(5, .0002 * buy_notional), max(5, .0002 * sell_notional)
                total_cost = quantity * (buy_price - sell_price) + buy_fee + sell_fee
                # 求该笔固定份额交易覆盖双边费用所需的无滑点卖出参考价。
                low, high = 4., 4.2
                for _ in range(60):
                    mid = (low + high) / 2
                    sale = quantity * mid * (1 - slip)
                    if sale - max(5, .0002 * sale) >= buy_notional + buy_fee:
                        high = mid
                    else:
                        low = mid
                cost_rows.append({"capital_cny": capital, "budget_fraction": fraction, "budget_cny": budget,
                                  "cost": name, "reference_price": 4., "quantity_shares": quantity,
                                  "buy_cash_used": buy_notional + buy_fee, "budget_cash_left": budget - buy_notional - buy_fee,
                                  "buy_commission": buy_fee, "sell_commission": sell_fee,
                                  "slippage_cost": quantity * (buy_price - sell_price), "roundtrip_cost": total_cost,
                                  "cost_per_budget": total_cost / budget, "breakeven_reference_return": high / 4 - 1,
                                  "role": "FIXED_PRICE_COST_EXAMPLE_NOT_ACCOUNT"})
    save_csv(pd.DataFrame(cost_rows), out / "20万与2万_整手费用示例.csv")
    fund = json.loads((study / "evidence/fund_share_clock_closure.json").read_text(encoding="utf-8"))
    gates = []
    for arm in ("A", "B", "C", "D"):
        reason = "新口径配对预期案例只有6个，未达到36训练+24评价的预定最低门槛"
        if arm in ("C", "D"):
            reason += "；真实份额变量缺少历史首次披露时间证明"
        gates.append({"arm": arm, "status": "NOT_RUN", "reason": reason,
                      "model_fits": 0, "evaluation_events": 0, "account_runs": 0,
                      "cagr": None, "net_sharpe": None})
    save_csv(pd.DataFrame(gates), out / "四组检验状态.csv")
    nbs = json.loads((study / "evidence/NBS_G2_adjudication.json").read_text(encoding="utf-8"))
    attr = json.loads((study / "evidence/prior_failure_attribution.json").read_text(encoding="utf-8"))
    corr = pd.read_csv(study / "evidence/prior_24_correlation.csv", index_col=0).to_numpy()
    correlation_median = float(np.median(corr[np.triu_indices_from(corr, 1)]))
    assert np.isclose(correlation_median, attr["median_pairwise_return_correlation"])
    ranks = pd.read_csv(study / "evidence/prior_rank_continuation.csv")
    rank_mean = float(ranks.past_sharpe_future_mean_rank_correlation.mean())
    assert len(ranks) == attr["ranking_comparison_periods"]
    assert np.isclose(rank_mean, attr["mean_rank_correlation"])
    summary = {
        "status": "CASE_CLOCK_REVIEW_COMPLETE_MODEL_AND_ACCOUNT_NOT_RUN_DATA_GATES",
        "case_count": len(results), "monthly_universe_count": len(months), "missing_prerelease_pairs": int((~months.case_admitted).sum()),
        "market_start": str(dates.min().date()), "market_end": str(dates.max().date()), "market_rows": len(market),
        "first_reaction_matches_surprise": int(results.first_reaction_same_sign.sum()),
        "residual_matches_surprise": int(results.residual_same_sign.sum()),
        "positive_surprise_cases": int(results.spread_surprise_proxy_pp.gt(0).sum()),
        "positive_surprise_positive_residual": int((results.spread_surprise_proxy_pp.gt(0) & results.residual_5d_gross_return.gt(0)).sum()),
        "negative_surprise_cases": int(results.spread_surprise_proxy_pp.lt(0).sum()),
        "negative_surprise_negative_residual": int((results.spread_surprise_proxy_pp.lt(0) & results.residual_5d_gross_return.lt(0)).sum()),
        "independent_event_clusters": cluster,
        "fund_source_rows_previously_checked": fund["source_rows"], "fund_raw_responses_previously_checked": fund["historical_raw_response_count"],
        "NBS_actual_state": "G2_RAN_AND_FAILED", "NBS_MSE_relative_change": -nbs["arms"]["MAIN"]["relative_mse_improvement"],
        "prior_correlation_median_recomputed": correlation_median, "prior_rank_mean_recomputed": rank_mean,
        "prior_covariance_participation_ratio_from_saved_result_not_rerun": attr["covariance_participation_ratio"],
        "new_model_fits": 0, "new_account_runs": 0, "strict_forward_events": 0,
        "C_D_state": "NOT_RUN_FUND_CLOCK_AND_SAMPLE_GATES", "exit_rule_comparison": "NOT_RUN_ENTRY_NOT_VALIDATED",
        "risk_prediction_increment": "NOT_TESTED_BY_SIX_CASE_CLOCK_REVIEW", "DSR": "NOT_COMPUTED_NO_NEW_ACCOUNT_AND_INCOMPLETE_UPSTREAM_TRIAL_DISTRIBUTION",
        "account_target_achieved": False, "all_information_complete": False, "position_impact": 0,
        "statistical_boundary": "没有显著性结论，没有独立预测胜率估计，没有政策因果归因；同号计数是六例描述。",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def plot(study: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": "#182f43", "axes.labelcolor": "#344e61"})
    data = pd.read_csv(study / "results/六次事件_观察后五日完整结果.csv")
    paths = pd.read_csv(study / "results/六次事件_全部路径点.csv")
    output.mkdir(parents=True, exist_ok=True)
    blue, orange, grey = "#146b9c", "#c27925", "#8d9caa"
    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    fig.subplots_adjust(left=.085, right=.97, bottom=.23, top=.81)
    x = np.arange(len(data))
    bars_a = ax.bar(x - .2, data.old_first_5d_return, .37, color=grey, label="原口径：公告后前 5 个交易日（含首轮反应）")
    bars_b = ax.bar(x + .2, data.residual_5d_gross_return, .37, color=blue, label="新口径：观察首日后，次日开盘持有 5 日")
    for bars in (bars_a, bars_b):
        for bar in bars:
            v = bar.get_height()
            ax.annotate(f"{v:+.2%}", (bar.get_x() + bar.get_width() / 2, v),
                        xytext=(0, 6 if v >= 0 else -7), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=11)
    ax.set_xticks(x, [f"{r.stat_month} 数据\n预期差 {r.spread_surprise_proxy_pp:+.1f} 个百分点" for r in data.itertuples()])
    ax.axhline(0, color="#324e62", lw=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("510300 区间毛收益")
    ax.set_ylim(min(data.old_first_5d_return.min(), data.residual_5d_gross_return.min()) - .006,
                max(data.old_first_5d_return.max(), data.residual_5d_gross_return.max()) + .006)
    ax.grid(axis="y", alpha=.17)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=1, frameon=False, fontsize=10)
    fig.suptitle("消息之后，还剩多少能参与的行情？", x=.085, y=.955, ha="left", fontsize=24, fontweight="bold")
    fig.text(.085, .9, "六次有事前预期快照的 M1/M2 公布事件｜固定新时钟：观察一完整日 → 次日开盘 → 第五日收盘", fontsize=12)
    fig.text(.085, .13, "两个柱子的起点和终点均不同，不可相减解释为“损失的收益”。同一终点的严格分解另存完整结果表。", fontsize=10)
    fig.text(.085, .08, "保留所有六例；没有根据结果挑事件。预期来自早于公布 4—9 天的周报快照，并非即时完整共识。\n收益未扣费，未生成策略账户；样本少、历史已被看过，不能把同号次数当作可复制胜率。", fontsize=10, color="#5a6b78")
    for ext in ("png", "svg"):
        fig.savefig(output / f"公告后反应与延迟五日收益.{ext}", dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.975, top=.84, bottom=.18, hspace=.44, wspace=.13)
    lo, hi = paths.return_from_pre_close.min() - .007, paths.return_from_pre_close.max() + .007
    for ax, row in zip(axes.flat, data.itertuples()):
        p = paths[paths.event_id == row.event_id]
        ax.axvspan(-1, .65, color="#e8edf1", alpha=.85)
        ax.axvspan(.65, 5, color="#e6f3f8", alpha=.9)
        ax.plot(p.plot_x, p.return_from_pre_close, "o-", color=blue, lw=1.8, ms=3.8)
        entry = p[p.point_role == "可执行开盘"].iloc[0]
        ax.scatter([.65], [entry.return_from_pre_close], color=orange, s=50, marker="^", zorder=4)
        ax.axvline(.65, color=orange, ls="--", lw=.9)
        ax.axhline(0, color="#8796a2", lw=.7)
        ax.set_title(f"{row.stat_month} 数据｜预期差 {row.spread_surprise_proxy_pp:+.1f} pp", loc="left", fontsize=12, pad=12)
        ax.text(.02, .95, f"可参与五日 {row.residual_5d_gross_return:+.2%}", transform=ax.transAxes, va="top", fontsize=11,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        ax.text(0, -.24, f"观察 {row.observation_date}\n入场 {row.entry_date} → 退出 {row.exit_date}", transform=ax.transAxes, fontsize=9, color="#526b7d")
        ax.set_ylim(lo, hi)
        ax.set_xticks([-1, 0, 1, 3, 5], ["前收", "观察", "D1", "D3", "D5"])
        ax.tick_params(axis="x", labelbottom=True)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.grid(axis="y", alpha=.15)
    for ax in axes[:, 0]:
        ax.set_ylabel("相对公告前收盘的累计变化")
    fig.suptitle("首轮反应属于过去，橙色入场点之后才是本次待赚的收益", x=.07, y=.955, ha="left", fontsize=22, fontweight="bold")
    fig.text(.07, .9, "灰色：观察与等待成交期间｜蓝色：固定五个交易日｜六个面板使用同一纵轴，保留所有交易日收盘", fontsize=12)
    fig.text(.07, .075, "橙色三角为入场开盘，位于观察收盘和 D1 收盘之间；横轴是交易步骤，节假日不计为交易日。\n曲线按公告前收盘归一化，面板数字按真实入场开盘重新计算。毛收益、案例描述，无收益归因或账户验证。", fontsize=10, color="#526b7d")
    for ext in ("png", "svg"):
        fig.savefig(output / f"六次事件_初始反应与可参与路径.{ext}", dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结事件的延迟收益复核与绘图")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    study = args.study_dir.resolve()
    if args.plot:
        plot(study, args.output_dir or study / "figures")
    else:
        analyse(study, args.output_dir or study / "results")


if __name__ == "__main__":
    main()
