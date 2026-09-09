"""从既有完整账户分析费用、信号重复和策略排名延续性，不改变原策略。"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import now, require, return_metrics, write_json

OUT = ROOT / "reports/research/510300_eight_round_failure_attribution_20260906"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT / "result.json").exists(), "八轮失败归因已保存")
    state = json.loads((ROOT / "reports/research/510300_sharpe_1_2_latest_research.json").read_text(encoding="utf-8"))
    rows = []
    for study in state["completed_rounds"][:8]:
        folder = (ROOT / study["result"]).parent
        ids = [study["primary_base"]["model"]]
        if study["round"] == 1:
            ids += ["P2_TOP1_504", "P3_TOP3_252", "P4_EQUAL_ALL"]
        for key in ids:
            ledger = pd.read_parquet(folder / "evaluation/BASE" / (key + "_ledger.parquet"))
            net = return_metrics(ledger.net_return.to_numpy(), 242)
            friction = ledger.commission + ledger.slippage_cost
            diagnostic_equity = ledger.equity.to_numpy() + friction.cumsum().to_numpy()
            diagnostic_return = diagnostic_equity / np.r_[200000.0, diagnostic_equity[:-1]] - 1
            gross = return_metrics(diagnostic_return, 242)
            rows.append({"round": study["round"], "study": study["title"], "model": key,
                         "net_sharpe": net["net_sharpe"], "net_cagr": net["annualized_return"],
                         "same_shares_friction_added_back_sharpe": gross["net_sharpe"],
                         "same_shares_friction_added_back_cagr": gross["annualized_return"],
                         "commission": float(ledger.commission.sum()), "slippage": float(ledger.slippage_cost.sum()),
                         "total_friction_fraction_initial_capital": float(friction.sum() / 200000),
                         "mean_exposure": float(ledger.exposure.mean()), "held_days": int(ledger.shares.gt(0).sum()),
                         "complete_account_days": len(ledger), "under_1_2_even_when_same_friction_removed": gross["net_sharpe"] is None or gross["net_sharpe"] < 1.2})
    costs = pd.DataFrame(rows)
    costs.to_csv(OUT / "固定成交份额的摩擦影响.csv", index=False, encoding="utf-8-sig")
    config = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    ids = list(config["rule_names"]) + [m["id"] for m in config["models"]]
    returns = pd.concat([pd.read_parquet(PARENT / "shadow/BASE" / (key + "_ledger.parquet")).set_index("date").net_return.rename(key) for key in ids], axis=1)
    eval_returns = returns.loc[(returns.index >= "2020-01-02") & (returns.index <= "2026-08-14")]
    covariance = eval_returns.cov().to_numpy()
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    effective_rank = float(eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum())
    correlation = eval_returns.corr()
    upper = correlation.to_numpy()[np.triu_indices(len(ids), k=1)]
    correlation.to_csv(OUT / "二十四个既有候选的日收益相关性.csv", encoding="utf-8-sig")
    selections = pd.read_csv(PARENT / "BASE_P1_PRIMARY_TOP3_504_selection.csv")
    selections["selection_origin"] = pd.to_datetime(selections.selection_origin)
    quarter_rows, candidate_rows = [], []
    for k, row in enumerate(selections.itertuples()):
        end = selections.selection_origin.iloc[k + 1] if k + 1 < len(selections) else returns.index.max()
        past = returns.loc[returns.index <= row.selection_origin].tail(504)
        future = returns.loc[(returns.index > row.selection_origin) & (returns.index <= end)]
        if len(past) < 504 or len(future) < 20:
            continue
        historical_score = past.mean() / past.std(ddof=1) * np.sqrt(242)
        future_mean = future.mean() * 242
        selected = str(row.selected).split("|") if pd.notna(row.selected) else []
        valid = np.isfinite(historical_score) & np.isfinite(future_mean)
        rank_ic = float(spearmanr(historical_score[valid], future_mean[valid]).statistic)
        selected_mean = float(future_mean[selected].mean()) if selected else np.nan
        all_mean = float(future_mean.mean())
        quarter_rows.append({"selection_origin": row.selection_origin, "evaluation_end": end, "next_period_days": len(future),
                             "selected": "|".join(selected), "past_sharpe_future_mean_rank_correlation": rank_ic,
                             "selected_next_period_mean_ann": selected_mean, "all_candidates_next_period_mean_ann": all_mean,
                             "selected_minus_all_mean_ann": selected_mean - all_mean})
        for model in ids:
            candidate_rows.append({"selection_origin": row.selection_origin, "model": model, "selected": model in selected,
                                   "past_504_sharpe": float(historical_score[model]), "next_period_mean_ann": float(future_mean[model])})
    quarters = pd.DataFrame(quarter_rows)
    quarters.to_csv(OUT / "过去排名与下一阶段表现.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(OUT / "每次选择的全部候选诊断.csv", index=False, encoding="utf-8-sig")
    forecasts = []
    for folder_name, candidates in [
        ("510300_total_reverse_repo_v2", [("T1_PRIMARY_ALL", 20), ("T5_PRICE", 20)]),
        ("510300_original_earnings_breadth_v1", [("F1_PRIMARY_FULL_H60", 60), ("F2_EARNINGS_H60", 60), ("F3_PRICE_H60", 60)]),
    ]:
        folder = ROOT / "reports/research" / folder_name
        p = pd.read_parquet(folder / "predictions.parquet")
        labels = pd.read_parquet(folder / "labels.parquet")
        anchor = int(np.flatnonzero(p.date >= "2020-01-02")[0]) - 1
        for model, horizon in candidates:
            take = np.arange(anchor, len(p) - 1, horizon)
            x, y = p[model].to_numpy()[take], labels[f"Y{horizon}"].to_numpy()[take]
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            forecasts.append({"study": folder_name, "model": model, "horizon": horizon, "mature_nonoverlap_forecasts": len(x),
                              "first_mature_prediction_origin": str(p.date.iloc[take[valid][0]]) if len(x) else None,
                              "mean_prediction": float(x.mean()) if len(x) else None, "mean_realization": float(y.mean()) if len(y) else None,
                              "correlation": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 else None,
                              "sign_hit_rate": float(np.mean((x > 0) == (y > 0))) if len(x) else None,
                              "mse_ratio_to_zero_return_forecast": float(np.mean((x-y)**2) / np.mean(y**2)) if len(y) and np.mean(y**2) > 0 else None})
    pd.DataFrame(forecasts).to_csv(OUT / "不重叠成熟预测的校准诊断.csv", index=False, encoding="utf-8-sig")
    result = {"recorded_at": now(), "status": "COMPLETED_DIAGNOSTIC_EXISTING_EIGHT_ROUNDS_NO_PROMOTION", "cost_attribution": rows,
              "old_candidate_count": len(ids), "covariance_participation_ratio": effective_rank,
              "median_pairwise_return_correlation": float(np.nanmedian(upper)), "pairwise_return_correlation_over_0_8_fraction": float(np.mean(upper > .8)),
              "ranking_comparison_periods": len(quarters), "mean_rank_correlation": float(quarters.past_sharpe_future_mean_rank_correlation.mean()),
              "median_rank_correlation": float(quarters.past_sharpe_future_mean_rank_correlation.median()),
              "selected_beats_all_mean_periods": int(quarters.selected_minus_all_mean_ann.gt(0).sum()),
              "mean_selected_minus_all_ann": float(quarters.selected_minus_all_mean_ann.mean()), "forecast_diagnostics": forecasts,
              "diagnostic_is_executable_new_strategy": False, "all_historical_trials_adjusted": False,
              "limits": ["加回摩擦保持历史成交份额，不会重新利用多出的现金，不能作为真实零费用策略收益。",
                         "排名延续性只描述已保存候选之间的相对表现；日均平均是诊断，不是新组合账户。",
                         "收益协方差参与率描述样本内线性有效维度，不等于已证明的独立机制数量。",
                         "不重叠预测数量少，尤其财报60日仅少数样本，不能用相关系数点估计证明普适失败或成功。"]}
    write_json(OUT / "result.json", result, exclusive=True)
    lines = ["# 八轮失败原因：从已有账户取得的证据", "",
             "策略的有效性可能随市场改变，但按过去收益选强者是否有效，是另一个需要证据的问题。已完成的第一轮按过去两年选三种，基础夏普为负0.2486；同轮固定等权组合为0.0551。两者都未达标，切换方式没有改善结果。", "",
             "## 一、费用是不是唯一原因", "", "下面固定历史成交份额，把已经支付的佣金与滑点金额加回现金，检查同一持仓路径的摩擦影响。此曲线不重新买入多出的现金，是归因用反事实，不是可执行零费用策略。", "",
             "| 轮次 | 研究或方案 | 原账户夏普 | 固定份额加回摩擦后的夏普 | 原账户年化收益 | 显性摩擦占初始资金 | 实际持有日数 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['round']} | {row['study']}：{row['model']} | {row['net_sharpe']:.4f} | {row['same_shares_friction_added_back_sharpe']:.4f} | {row['net_cagr']:.2%} | {row['total_friction_fraction_initial_capital']:.2%} | {row['held_days']}／{row['complete_account_days']} |")
    lines += ["", "## 二、过去强者有没有延续", "",
              f"对{len(quarters)}个至少有20个后续交易日的选择区间，过去504日夏普排名与下一阶段年化日均收益排名的平均相关系数为{result['mean_rank_correlation']:.4f}，中位数为{result['median_rank_correlation']:.4f}。被选中的三种策略，只有{result['selected_beats_all_mean_periods']}个区间的日均收益均值超过全体候选均值。", "",
              "这项诊断没有用未来排名替换实际选择，也没有制作可以交易的事后最优组合。它检验的是原切换规则所依赖的延续性。", "",
              "## 三、多种策略是否提供多种独立信息", "",
              f"原24个底层候选的日收益两两相关系数中位数为{result['median_pairwise_return_correlation']:.4f}，其中{result['pairwise_return_correlation_over_0_8_fraction']:.2%}超过0.8。收益协方差的参与率约为{effective_rank:.2f}。该指标是样本内线性维度诊断，不能直接叫作独立策略数量，但说明增加候选名称不一定增加分散来源。", "",
              "所有策略都交易同一只ETF。不同均线周期和同一价格数据训练的模型仍可能持有相近仓位，不能按参数版本数量重复增加某一类机制的话语权。", "",
              "## 四、信息质量与持仓决策要分开", "",
              "原七天逆回购范围不足已由第七轮新版本修正，修正后主方案仍未达到目标。第八轮因原始金融公司财报缺口，2024年才形成主方案预测，2025年底才首次持有；这种长等待说明资料不足与模型输出不足，不能仅凭小回撤评价策略稳定。", "",
              "附件把第七轮和第八轮的实际决策节奏下已到期、不重叠预测单列，检验预测偏差和方向。样本不足的行原样报告，不以看过的好时期证明有效。", "",
              "## 本轮证据对下一步的约束", "",
              "需要同时改进可用信息、机制分组及切换方法。新研究先让趋势、反转、混合规则、预测模型分别形成一份机制判断，再由当时已知的趋势和波动状态决定如何更新权重；以过去已经结算的账户表现逐步更新，保留静态、无状态和硬切换对照。它是新的待验证方法，并不因会自动切换而预先合格。", "",
              "原始盈利、普通股股本与回报、公募原始公布时钟的补齐继续。不能用尚未核实的数据为新组合背书。所有旧账户和结论保持原状。"]
    (OUT / "八轮失败归因_中文说明.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ('cost_attribution','forecast_diagnostics','limits')}, ensure_ascii=False), flush=True)
    print(costs[['round','model','net_sharpe','same_shares_friction_added_back_sharpe']].to_string(index=False), flush=True)
    print(pd.DataFrame(forecasts).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
