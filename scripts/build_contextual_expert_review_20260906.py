"""形成第九轮全部中文规则、失败归因、原始来源与离线复算审阅包。"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPORT = ROOT / "reports/research/510300_contextual_expert_tracking_v1"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
DIAG = ROOT / "reports/research/510300_eight_round_failure_attribution_20260906"
BANK = ROOT / "reports/research/510300_bank_financial_scope_adapter_v1"
COMPLETION = ROOT / "reports/research/510300_original_earnings_source_completion_v1"
FUND = ROOT / "reports/research/510300_fundamental_and_fund_flow_rebuild_v1"
SUBSCRIPTION = ROOT / "reports/research/510300_original_fund_subscription_reports_v1"
FACTOR = ROOT / "reports/research/510300_factor_definition_review_20260906"
DELIVERY = ROOT / "deliverables/510300第九轮_动态机制组合与失败归因_20260906"
ZIP = ROOT / "deliverables/510300夏普1.2持续研究_第九轮动态机制组合_GPT审阅_20260906.zip"
MANIFESTS = ["config/510300_contextual_expert_tracking_v1_manifest.json",
             "config/510300_adaptive_allocation_v1_manifest.json",
             "config/510300_bank_financial_scope_adapter_v1_manifest.json"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def physical(path):
    relative = path.relative_to(ROOT)
    outside = Path(r"E:\ResearchData\New project 8") / relative
    # 解压包内优先使用本地文件，工作区的 data 联接可直接读取其实际路径。
    if relative.parts[0] == "data" and path.exists():
        return path.resolve()
    return outside if relative.parts[0] == "data" and outside.exists() else path


def number(value):
    return "无定义" if pd.isna(value) else f"{value:.4f}"


def frozen_files():
    expected = {}
    for name in MANIFESTS:
        for row in read(ROOT / name)["files"]:
            if row["path"] in expected:
                assert expected[row["path"]]["sha256"] == row["sha256"]
            expected[row["path"]] = row
    return expected


def verify_diagnostics():
    from research.intraday_overnight_increment_v1 import return_metrics
    state = read(ROOT / "reports/research/510300_sharpe_1_2_latest_research.json")
    result = read(DIAG / "result.json")
    rounds = {x["round"]: x for x in state["completed_rounds"][:8]}
    for row in result["cost_attribution"]:
        folder = (ROOT / rounds[row["round"]]["result"]).parent
        ledger = pd.read_parquet(folder / "evaluation/BASE" / (row["model"] + "_ledger.parquet"))
        equity = ledger.equity.to_numpy() + (ledger.commission + ledger.slippage_cost).cumsum().to_numpy()
        returns = equity / np.r_[200000.0, equity[:-1]] - 1
        value = return_metrics(returns, 242)["net_sharpe"]
        assert abs(value - row["same_shares_friction_added_back_sharpe"]) < 1e-10
    config = read(ROOT / "config/510300_adaptive_allocation_v1.json")
    models = list(config["rule_names"]) + [x["id"] for x in config["models"]]
    returns = pd.concat([pd.read_parquet(PARENT / "shadow/BASE" / (k + "_ledger.parquet")).set_index("date").net_return.rename(k) for k in models], axis=1)
    evaluation = returns.loc["2020-01-02":"2026-08-14"]
    eigenvalues = np.maximum(np.linalg.eigvalsh(evaluation.cov()), 0)
    participation = eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()
    assert abs(participation - result["covariance_participation_ratio"]) < 1e-10
    selections = pd.read_csv(PARENT / "BASE_P1_PRIMARY_TOP3_504_selection.csv", parse_dates=["selection_origin"])
    observed = []
    for i, row in enumerate(selections.itertuples()):
        end = selections.selection_origin.iloc[i+1] if i+1 < len(selections) else returns.index.max()
        past = returns.loc[returns.index <= row.selection_origin].tail(504)
        future = returns.loc[(returns.index > row.selection_origin) & (returns.index <= end)]
        if len(past) < 504 or len(future) < 20:
            continue
        historical = past.mean() / past.std(ddof=1) * np.sqrt(242)
        next_mean = future.mean() * 242
        valid = np.isfinite(historical) & np.isfinite(next_mean)
        observed.append(spearmanr(historical[valid], next_mean[valid]).statistic)
    assert len(observed) == result["ranking_comparison_periods"] == 27
    assert abs(np.mean(observed) - result["mean_rank_correlation"]) < 1e-10
    for row in result["forecast_diagnostics"]:
        folder = ROOT / "reports/research" / row["study"]
        predictions, labels = [pd.read_parquet(folder / f) for f in ("predictions.parquet", "labels.parquet")]
        start = int(np.flatnonzero(predictions.date >= "2020-01-02")[0])-1
        take = np.arange(start, len(predictions)-1, row["horizon"])
        x, y = predictions[row["model"]].to_numpy()[take], labels[f"Y{row['horizon']}"].to_numpy()[take]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        assert len(x) == row["mature_nonoverlap_forecasts"]
        assert abs(np.mean((x-y)**2) / np.mean(y**2) - row["mse_ratio_to_zero_return_forecast"]) < 1e-10
    return {"fixed_share_friction_paths": len(result["cost_attribution"]), "old_shadow_accounts": len(models),
            "selection_periods": len(observed), "forecast_comparisons": len(result["forecast_diagnostics"])}


def verify_saved():
    from research.adaptive_allocation_v1 import summarize
    from research.contextual_expert_tracking_v1 import FAMILIES, META
    config = read(ROOT / "config/510300_contextual_expert_tracking_v1.json")
    metrics = pd.read_csv(REPORT / "metrics.csv")
    maximum = 0.0
    assert len(metrics) == 24
    for row in metrics.to_dict("records"):
        ledger = pd.read_parquet(REPORT / "evaluation" / row["cost"] / (row["model"] + "_ledger.parquet"))
        actual = summarize(ledger, config)
        for key, value in actual.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value):
                error = abs(value - float(row[key]))
                assert error < 1e-7, (row["model"], key, error)
                maximum = max(maximum, error)
        assert len(ledger) == 1604 and str(ledger.date.min().date()) == "2020-01-02" and str(ledger.date.max().date()) == "2026-08-14"
        assert ledger.accounting_error.abs().max() < 1e-6
        np.testing.assert_allclose(ledger.equity, ledger.cash + ledger.shares * ledger.mark + ledger.dividend_receivable, rtol=0, atol=1e-6)
        np.testing.assert_allclose(ledger.net_return, ledger.equity.to_numpy()/np.r_[200000.0, ledger.equity.to_numpy()[:-1]]-1, rtol=0, atol=1e-12)
        assert ledger.shares.iloc[-1] == 0 and not ledger.terminal_unliquidated.iloc[-1]
        if row["model"] == "CASH":
            assert ledger.net_return.eq(0).all() and pd.isna(row["net_sharpe"])
        if row["model"] == "BUY_HOLD":
            old = pd.read_parquet(PARENT / "evaluation" / row["cost"] / "BUY_HOLD_ledger.parquet")
            for col in ["equity", "shares", "cash", "net_return", "commission", "slippage_cost"]:
                np.testing.assert_array_equal(ledger[col], old[col])
    data = pd.read_parquet(PARENT / "features.parquet").set_index("date")
    weights = pd.read_parquet(REPORT / "daily_mechanism_weights.parquet")
    updates = pd.read_parquet(REPORT / "online_update_receipts.parquet")
    targets = pd.read_parquet(REPORT / "teacher_targets.parquet").set_index("date")
    rewards = pd.read_parquet(REPORT / "teacher_returns.parquet").set_index("date")
    assert len(updates) == 2823 and len(weights) == 6*len(updates)
    np.testing.assert_allclose(weights[FAMILIES].sum(axis=1), 1, rtol=0, atol=1e-12)
    assert weights[FAMILIES].ge(0).all().all() and weights.target.between(0, 1).all()
    dot = np.sum(weights[FAMILIES].to_numpy() * targets.loc[weights.date, FAMILIES].to_numpy(), axis=1)
    np.testing.assert_allclose(dot, weights.target, rtol=0, atol=1e-12)
    state = 2*data.sma120.gt(0).astype(int) + data.vol_ratio.gt(1).astype(int)
    np.testing.assert_array_equal(updates.current_state4, state.loc[updates.origin])
    np.testing.assert_array_equal(updates.updated_previous_state4.iloc[1:], state.shift(1).loc[updates.origin.iloc[1:]])
    valid_updates = updates.loc[updates.latest_observed_return_date.notna()]
    assert (valid_updates.latest_observed_return_date <= valid_updates.origin).all()
    risk = np.maximum(np.sqrt(data.variance60).shift(1).loc[valid_updates.origin], .005)
    np.testing.assert_allclose(risk, valid_updates.previous_day_risk_scale, rtol=0, atol=1e-12)
    for key in FAMILIES:
        ledger = pd.read_parquet(REPORT / "mechanism_shadow/BASE" / (key + "_ledger.parquet"))
        decisions = pd.read_parquet(REPORT / "mechanism_shadow/BASE" / (key + "_decisions.parquet"))
        assert ledger.accounting_error.abs().max() < 1e-6 and len(ledger) == 2823
        np.testing.assert_array_equal(ledger.net_return, rewards.loc[ledger.date, key])
        np.testing.assert_array_equal(decisions.reference_weight, targets.loc[decisions.origin, key])
        assert (decisions.execution_date > decisions.origin).all()
    for key in META:
        a, b = [pd.read_parquet(REPORT / "evaluation" / cost / (key + "_decisions.parquet")) for cost in ("BASE", "STRESS")]
        np.testing.assert_array_equal(a.reference_weight, b.reference_weight)
        assert (a.execution_date > a.origin).all() and (b.execution_date > b.origin).all()
    expected = frozen_files()
    for name, row in expected.items():
        assert hashlib.sha256(physical(ROOT / name).read_bytes()).hexdigest() == row["sha256"], name
    bank = read(BANK / "result.json")
    assert bank["documents_examined"] == 24 and bank["documents_with_supported_layout"] == 2 and bank["fact_rows"] == 8
    facts = pd.read_parquet(BANK / "explicit_consolidated_q3_facts.parquet")
    assert len(facts) == 8 and facts.statement_scope.eq("CONSOLIDATED_ONLY").all()
    assert not facts.previous_year_comparative_is_original_prior_vintage.any()
    for row in read(COMPLETION / "batch_01_result.json")["rows"]:
        assert hashlib.sha256(physical(ROOT / row["raw_path"]).read_bytes()).hexdigest() == row["sha256"]
    subscription = read(SUBSCRIPTION / "example_2021Q3_result.json")
    for row in subscription["sources"]:
        assert hashlib.sha256(physical(ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    assert len({row["sha256"] for row in subscription["sources"]}) == 1
    flow = subscription["facts"]
    assert flow["beginning_units"] + flow["gross_subscription_units"] - flow["gross_redemption_units"] == flow["ending_units"]
    assert flow["gross_subscription_units"] - flow["gross_redemption_units"] == flow["net_subscription_units"]
    assert flow["exact_cash_flow_amount"] is None and not subscription["source_admitted_for_portfolio"]
    return {"status": "PASS_SAVED_ACCOUNTS_WEIGHT_CLOCKS_FROZEN_INPUTS_AND_FAILURE_DIAGNOSTICS", "evaluation_accounts": 24,
            "mechanism_shadow_accounts": 6, "learning_origins": len(updates), "mechanism_weight_rows": len(weights),
            "maximum_metric_error": maximum, "benchmark_daily_parity": "EXACT", "frozen_files_verified": len(expected),
            "financial_documents_hash_verified": 24, "financial_documents_matching_explicit_layout": 2,
            "original_fund_subscription_reports": 1, "byte_identical_official_fund_archives": 2,
            "failure_diagnostics": verify_diagnostics(), "underlying_models_retrained": False, "security_audit_performed": False}


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from research.contextual_expert_tracking_v1 import FAMILIES, NAMES
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2,1]})
    for key, name, color in [("A1_PRIMARY_CONTEXT4_BLEND", "四状态与总体混合", "#187f86"), ("A6_EQUAL_FAMILIES", "六机制固定等权", "#b47b3b"), ("BUY_HOLD", "买入持有", "#777b83")]:
        d = pd.read_parquet(REPORT / "evaluation/BASE" / (key + "_ledger.parquet"))
        wealth = d.equity.to_numpy()/200000
        high = np.maximum.accumulate(np.r_[1., wealth])[1:]
        axes[0].plot(d.date, wealth, color=color, label=name, linewidth=1.5)
        axes[1].plot(d.date, 100*(wealth/high-1), color=color, linewidth=1.2)
    axes[0].set_title("第九轮：市场状态与机制权重的持续学习", loc="left", fontsize=16, pad=42)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0,1.01), ncol=3, frameon=False)
    axes[0].set_ylabel("完整账户净值（初始为1）")
    axes[1].set_ylabel("回撤（%）")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.text(.08,.02,"2020-01-02 至 2026-08-14 开盘；基础费用、现金等待、分红及期末退出均计入。\n主方案夏普0.1578，固定等权0.2069；状态切换未带来已确认的收益改善。",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.065,1,1))
    fig.savefig(DELIVERY / "第九轮_完整账户净值与回撤.png",dpi=150)
    plt.close(fig)
    d = pd.read_parquet(REPORT / "daily_mechanism_weights.parquet")
    d = d.loc[d.model.eq("A1_PRIMARY_CONTEXT4_BLEND") & d.date.ge("2020-01-01")]
    fig, ax = plt.subplots(figsize=(11,5.4))
    ax.stackplot(d.date, *(d[k]*100 for k in FAMILIES), labels=[NAMES[k] for k in FAMILIES],
                 colors=["#d6d9dd","#668ca7","#238c8f","#cdac64","#955978","#5767a6"],linewidth=0)
    ax.set_title("主方案每天给予六种机制的权重",loc="left",fontsize=16,pad=45)
    ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=6,frameon=False,fontsize=9)
    ax.set_ylim(0,100)
    ax.set_ylabel("机制权重（%）")
    ax.margins(x=0)
    fig.text(.08,.02,"权重是各机制对次日目标仓位的影响程度，不是六只证券的实际持仓。\n账户最终只持有510300与现金；图中变化全部依据当时已结算资料。",fontsize=9,color="#555555")
    fig.tight_layout(rect=(0,.075,1,1))
    fig.savefig(DELIVERY / "第九轮_主方案逐日机制权重.png",dpi=150)
    plt.close(fig)


def prepare():
    assert not DELIVERY.exists() and not ZIP.exists(), "交付已经存在，不覆盖"
    check = verify_saved()
    result = read(REPORT / "result.json")
    metrics, eras = [pd.read_csv(REPORT / f) for f in ("metrics.csv","era_metrics.csv")]
    from research.contextual_expert_tracking_v1 import META, FAMILIES, NAMES
    base, stress = [next(x for x in result["primary"] if x["cost"] == cost) for cost in ("BASE","STRESS")]
    DELIVERY.mkdir(parents=True)
    lines = ["# 第九轮：动态机制组合与失败归因", "",
             f"**主方案成本后夏普{base['net_sharpe']:.4f}，未达到1.2。** 完整账户年化收益{base['annualized_return']:.2%}、最大回撤{-base['max_drawdown']:.2%}；压力费用夏普{stress['net_sharpe']:.4f}。", "",
             "本轮完成十一种候选、两档费用共二十四条完整评价账户，另有六条学习用机制模拟账户。主方案事先确定，底层预测没有重新训练或按本轮结果筛选。", "",
             "## 给老板看的解释", "",
             "这次检验的是：先把相似策略合成一类，再根据当时处于上涨或下跌趋势、波动上升或未上升，逐日调整各类策略的重要程度。六种判断来源是现金、买入持有、趋势、反转、趋势震荡混合、价格预测模型。最终账户只持有510300和现金。", "",
             "策略组合确实会随时间改变，但这次没有改善结果。六种机制固定等权的夏普为0.2069，高于主方案0.1578；同期买入持有为0.2879。不能从“市场会变”直接推出“自动切换一定更准确”。", "",
             "此次并未加入尚待补齐的完整指数盈利、公募申购等新信息。增加组合的复杂程度，仍可能只是重新分配同一批价格信息。盈利、估值、股东回报、基金申购的原始资料修正继续。", "",
             "## 全部候选与对照表现", "",
             "| 方案 | 基础夏普 | 压力夏普 | 基础年化收益 | 与买入持有年化差 | 基础最大回撤 | 成交笔数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key in META + FAMILIES:
        a,b = [metrics.loc[metrics.model.eq(key) & metrics.cost.eq(cost)].iloc[0] for cost in ("BASE","STRESS")]
        lines.append(f"| {NAMES[key]} | {number(a.net_sharpe)} | {number(b.net_sharpe)} | {a.annualized_return:.2%} | {a.annualized_return_excess_vs_buy_hold:.2%} | {-a.max_drawdown:.2%} | {int(a.trade_count)} |")
    lines += ["", "评价区间统一为2020年1月2日至2026年8月14日开盘，共1604个交易日，初始20万元。现金利息与夏普参考利率均为零；现金账户没有波动时夏普不定义。表内年化差是两条账户复合年化收益的差值，不是信息比率。压力账户使用同一条目标信号，提高佣金和滑点。", "",
              "## 主方案分期与不确定性", "", "| 阶段 | 基础夏普 | 年化收益 | 最大回撤 |", "|---|---:|---:|---:|"]
    for row in eras.loc[eras.model.eq(META[0]) & eras.cost.eq("BASE")].itertuples():
        lines.append(f"| {row.era} | {number(row.net_sharpe)} | {row.annualized_return:.2%} | {-row.max_drawdown:.2%} |")
    u=result["uncertainty"]["BASE"]
    lines += ["",f"二十日连续区块抽样得到主方案夏普95%区间{u['primary_sharpe_95_interval'][0]:.4f}至{u['primary_sharpe_95_interval'][1]:.4f}。相对固定等权的年化日均收益增量95%区间为{u['increment_vs_equal_95_interval'][0]:.2%}至{u['increment_vs_equal_95_interval'][1]:.2%}，跨过零。相对仅总体学习和硬切换的增量区间也都跨过零。", "",
              "已反复观察的历史不构成新的独立验证，这些区间也没有完成所有历史尝试的选择校正。本轮没有任何完整区间候选达到夏普1.2。", "",
              "## 已有八轮为什么失败", "",
              "第一，费用拖累明显，但不是唯一问题。保持历史实际成交份额，把已付佣金和滑点加回现金后，八轮主方案的夏普仍全部低于1.2。这是费用归因，额外现金没有重新投资，不能当作可交易的零费用账户。", "",
              "第二，过去强者缺少稳定延续。第一轮27个选择区间中，过去504日夏普排名与下一阶段日均收益排名的平均相关系数为负0.0398；选中的前三名只有11个区间超过全部候选的平均表现。这没有证明显著的负向规律，只说明该切换规则依赖的正向延续没有得到支持。", "",
              "第三，策略名称多，信息未必多。24个旧底层策略日收益的两两相关中位数为0.5221；收益协方差参与率为2.41，显示明显线性重叠。这个统计量不能直接叫作独立策略数量。", "",
              "第四，部分经济因子资料不完整。只统计七天逆回购的范围问题已在第七轮纠正；原始金融财报缺口使第八轮较晚才达到覆盖和训练要求。小回撤若来自长时间未持有，不能解释为已形成稳定超额能力。", "",
              "完整费用路径、排名区间和预测偏差表在“八轮失败归因_中文说明.md”及附表中，原始计算输入随包保留。", "",
              "## 盈利、估值、股东回报和公募申购怎样落地", "",
              "在每股盈利与市盈率口径相同的条件下，价格等于每股盈利乘市盈率；持有总回报再考虑期间现金分红及股份变化。现金分红需要按实际权益计入；回购既会影响股本和每股盈利，也可能影响现金资产，不能再把全部回购金额重复加给每一名持有人。研究要解释未来盈利、未来定价倍数和实际股东回报的变化，而不是把恒等关系当作新预测。", "",
              "首批二十四份金融公司原始报告已全部归档。新解析方式先核实合并报表标题、累计期间、年份列和金额单位，再检查会计等式。当前二份原始报告支持这一明确版式，合计得到八项可核对数值；其余二十二份仍需对应版式处理，不记成零，也未进入第九轮策略。", "",
              "招商银行2021三季报例子：合并前三季营业利润为1165.81亿元；旧非金融解析器直接用于该银行报告，会误抓母公司第三季度的367.63亿元并误标累计合并。新方式拒绝母公司页，并按实际期间表头取列。旧非金融研究原本排除了金融公司，这个迁移失败不等于旧研究所有已保存财务事实都错误。", "",
              "普通股前三季每股盈利为3.62元，第三季度为1.27元。银行归母净利润与普通股盈利分子之间还涉及优先股股息、永续债利息等扣除。后续要核对原始上年同期、普通股股数与其他权益工具，才能构造同口径的指数盈利。", "",
              "公募申购需要区分基金份额增减、基金净值涨跌和现金申购赎回。已经归档的77份协会月报仍存在原始公布时钟与部分数值版本问题，当前还没有将它们作为已合格申购因子。基金规模增长不等于资金净流入，基金份额增加也不等于这些资金全部买入沪深300。", "",
              "## 第九轮全部中文因子和执行规则", "", (ROOT / "docs/510300_CONTEXTUAL_EXPERT_TRACKING_V1_PROTOCOL.md").read_text(encoding="utf-8"), "",
              "## 三十四项既有价格输入的逐项中文定义", "",
              "下表是底层预测模型复用的输入。三项环境因子与其中部分数据重叠，不能当作三份新增独立信息。", "",
              "| 序号 | 中文计算规则 | 单位或含义 | 使用边界 |", "|---:|---|---|---|"]
    prices = pd.read_csv(FACTOR / "95项因子逐项中文核对.csv").query('因子组 == "价格与财富"')
    assert len(prices)==34
    for i,row in enumerate(prices.to_dict("records"),1):
        cells=[str(row[c]).replace("|","／") for c in ("中文定义","单位或含义","问题或使用边界")]
        lines.append(f"| {i} | {' | '.join(cells)} |")
    old_protocol = (ROOT / "docs/510300_ADAPTIVE_ALLOCATION_V1_PROTOCOL.md").read_text(encoding="utf-8")
    model_rules = old_protocol.split("## 12 个滚动训练模型",1)[1].split("## 四个组合方案",1)[0]
    lines += ["", "## 十二个底层预测模型的完整中文参数", "", model_rules.strip(), "",
              "## 数值核对与交付边界", "",
              "第九轮五项针对性测试、银行范围与期间四项测试均通过。二十四条评价账户的保存指标复算通过，买入持有与原基准逐日相同；六条机制账本、2823个学习时点及16938条机制权重记录核对通过。奖励归入前一日状态，风险尺度取前一日已知值，基础与压力组合目标相同。", "",
              "金融原始文件二十四份、公募月报七十七份、银行误读与修正实例，以及冻结协议、全部新账户、学习记录、失败归因输入均随包保留。旧预测模型只使用已保存输出，本包不会为交付重新训练或改变旧结果。", "",
              "压缩包只进行CRC、成员重复、索引覆盖、文件哈希和保存结果数值检查，未做额外安全审计，未上传，也未收到外部审阅。只研究510300与现金，未授权真实交易。夏普1.2与稳定超额目标未完成，继续实质资料修复和新机制研究。"]
    (DELIVERY / "第九轮结果与全部中文因子规则.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (DELIVERY / "八轮失败归因_中文说明.md").write_bytes((DIAG / "八轮失败归因_中文说明.md").read_bytes())
    bank_text = "# 银行原始报表修正结果\n\n首批24份报告逐份识别，2份匹配已核实版式、产生8项数值，其余22份保持待补齐状态。没有新财务因子进入第九轮账户。\n\n" + (ROOT / "docs/510300_BANK_FINANCIAL_SCOPE_ADAPTER_V1.md").read_text(encoding="utf-8")
    (DELIVERY / "银行原始报表修正结果.md").write_text(bank_text,encoding="utf-8")
    (DELIVERY / "保存结果核对.json").write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding="utf-8")
    (DELIVERY / "00_先读说明.md").write_text("# 阅读顺序\n\n1. 第九轮结果与全部中文因子规则.md：结论、每个方案、所有因子和中文执行规则。\n2. 八轮失败归因_中文说明.md：费用、排名延续、信息重复及资料缺口。\n3. 银行原始报表修正结果.md：范围与期间误读实例、当前已识别范围。\n4. 两张图：完整账户净值回撤、主方案每天采用各机制的比例。\n5. 01_GPT审阅提示.md：请外部模型检验本轮结论并提出下一研究优先级。\n\n离线复核：解压后在目录根运行 scripts/build_contextual_expert_review_20260906.py，传入 --verify-only；需要Python、numpy、pandas、pyarrow、scipy、scikit-learn。它复算保存的账户、学习时钟和失败诊断，不重新训练模型或生成新投资结果。PDF原文处理另需pdfplumber和pypdfium2。\n\nreports/research/510300_sharpe_1_2_latest_research.json 在包内是八轮失败归因输入的当次索引快照，第九轮结果以自身result.json和本说明为准；工作区持续索引会在本包完成后更新。\n\n结构核对不属于安全审计，也不表示已得到外部审阅。\n",encoding="utf-8")
    (DELIVERY / "01_GPT审阅提示.md").write_text("# 请独立审阅并给出下一研究方向\n\n目标为510300与现金的完整账户成本后夏普至少1.2，并取得可解释的稳定超额。当前第九轮主方案0.1578、静态等权0.2069，目标未达。请先核对结果，不把组合中旧规则的使用解释为旧研究获准。\n\n请优先回答：过去排名无延续、策略相关和费用归因能支持多强的失败解释；市场状态权重是否严格使用当时已结算信息，奖励状态和前一日风险尺度是否正确；六种机制回流与状态置信度是否有经济依据；学习用账户到实际完整账户是否存在隐藏错配。检查二十四条全部账户、三分期、压力费用和增量区间，不能只看事后最优。\n\n再核对银行母公司/合并与单季/累计误读例子。当前只有2份对应版式的8个数值，不能视为整个金融行业覆盖。请提出补齐原始普通股EPS、股本与权益扣除、分红回购以及公募实际申赎的最小可行路线，区分来源时钟和当前修订值。不要把价格恒等式、基金规模涨幅或基金份额变化等同额外预测或现金流。\n\n最后给出按优先级排序的至多三个下一研究机制，每个说明经济假设、必需免费来源、中文因子定义、事先固定的有限对照、可用时间、完整账户规则、验证与停止条件。不能要求扩大真实交易权限、购买数据，或无限调整年份、窗口、方向以凑出1.2。若现有资料不支持机制，请指出具体缺口和可以独立推进的修复工作。\n",encoding="utf-8")
    draw()
    print(json.dumps({"状态":"第九轮说明与图已生成，待视觉核对后打包",**check},ensure_ascii=False),flush=True)


def research_imports(paths):
    found=set(paths)
    todo=[p for p in found if p.suffix==".py"]
    while todo:
        path=todo.pop()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names=[]
            if isinstance(node,ast.Import):
                names=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                names=[node.module] if node.module!="research" else ["research."+x.name for x in node.names]
            for name in names:
                if name.startswith("research."):
                    child=ROOT/(name.replace(".","/")+".py")
                    if child.exists() and child not in found:
                        found.add(child)
                        todo.append(child)
    init=ROOT/"research/__init__.py"
    if init.exists():
        found.add(init)
    return found


def package():
    assert DELIVERY.exists() and not ZIP.exists(), "先形成并检查交付，不覆盖压缩包"
    checked = verify_saved()
    (DELIVERY / "保存结果核对.json").write_text(json.dumps(checked,ensure_ascii=False,indent=2),encoding="utf-8")
    flow_note = SUBSCRIPTION / "公募申购_原始基金报告路线与已核实实例.md"
    (DELIVERY / flow_note.name).write_bytes(flow_note.read_bytes())
    main_report = DELIVERY / "第九轮结果与全部中文因子规则.md"
    main_text = main_report.read_text(encoding="utf-8")
    heading = "## 交付时新增的公募申购原始来源实例"
    if heading not in main_text:
        main_report.write_text(main_text + "\n" + heading + "\n\n510300的2021年三季报直接披露本季申购43.974亿份、赎回57.096亿份，净赎回13.122亿份；期初加申购减赎回与期末精确相等。上交所与巨潮文件内容完全一致，送出日期为2021年10月27日。这是新的原始基金报告来源实例，尚未进入本轮策略，后续继续补齐季度历史与首次公布、修订记录。基金份额不等于人民币现金，也不能代替全市场公募申购。原始PDF、封面和份额表图片、中文说明随包交付。\n",encoding="utf-8")
    expected=frozen_files()
    paths={ROOT/name for name in expected} | {ROOT/name for name in MANIFESTS}
    paths.update({Path(__file__),ROOT/"scripts/diagnose_eight_round_failures_20260906.py",
                  ROOT/"scripts/verify_bank_eps_source_example_20260906.py",ROOT/"scripts/collect_original_earnings_gaps_20260906.py",
                  ROOT/"scripts/record_amac_pdf_version_evidence_20260906.py",ROOT/"reports/research/510300_sharpe_1_2_latest_research.json",
                  ROOT/"scripts/record_original_fund_subscription_example_20260906.py",
                  FACTOR/"95项因子逐项中文核对.csv",FACTOR/"price_recomputation.json",PARENT/"BASE_P1_PRIMARY_TOP3_504_selection.csv"})
    for folder in (REPORT,DIAG,BANK,COMPLETION,FUND,SUBSCRIPTION):
        paths.update(p for p in folder.rglob("*") if p.is_file())
    state=read(ROOT/"reports/research/510300_sharpe_1_2_latest_research.json")
    for item in state["completed_rounds"][:8]:
        folder=(ROOT/item["result"]).parent
        paths.update(folder/f for f in ("result.json","metrics.csv","yearly_metrics.csv","era_metrics.csv") if (folder/f).exists())
        models=[item["primary_base"]["model"]]
        if item["round"]==1:
            models += ["P2_TOP1_504","P3_TOP3_252","P4_EQUAL_ALL"]
        paths.update(folder/"evaluation/BASE"/(key+"_ledger.parquet") for key in models)
    old=read(ROOT/"config/510300_adaptive_allocation_v1.json")
    keys=list(old["rule_names"])+[x["id"] for x in old["models"]]
    paths.update(PARENT/"shadow/BASE"/(key+"_ledger.parquet") for key in keys)
    for cost in ("BASE","STRESS"):
        paths.add(PARENT/"evaluation"/cost/"BUY_HOLD_ledger.parquet")
    for folder in ("510300_total_reverse_repo_v2","510300_original_earnings_breadth_v1"):
        paths.update(ROOT/"reports/research"/folder/f for f in ("predictions.parquet","labels.parquet"))
    actual=Path(r"E:\ResearchData\New project 8")
    for child in ("data/raw/510300_original_earnings_source_completion_v1","data/raw/510300_fundamental_and_fund_flow_rebuild_v1/amac","data/raw/510300_original_fund_subscription_reports_v1"):
        paths.update(ROOT/p.relative_to(actual) for p in (actual/child).rglob("*") if p.is_file())
    paths=research_imports(paths)
    index=[]
    with zipfile.ZipFile(ZIP,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(paths):
            relative=path.relative_to(ROOT).as_posix()
            content=physical(path).read_bytes()
            digest=hashlib.sha256(content).hexdigest()
            if relative in expected:
                assert digest==expected[relative]["sha256"], "冻结文件变化："+relative
            copied=DELIVERY/relative
            copied.parent.mkdir(parents=True,exist_ok=True)
            copied.write_bytes(content)
            archive.writestr(relative,content)
            index.append({"path":relative,"bytes":len(content),"sha256":digest})
        for path in sorted(p for p in DELIVERY.iterdir() if p.is_file()):
            content=path.read_bytes()
            archive.writestr(path.name,content)
            index.append({"path":path.name,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()})
        stream=io.StringIO(newline="")
        writer=csv.DictWriter(stream,fieldnames=["path","bytes","sha256"])
        writer.writeheader()
        writer.writerows(index)
        content=stream.getvalue().encode("utf-8-sig")
        (DELIVERY/"FILE_INDEX.csv").write_bytes(content)
        archive.writestr("FILE_INDEX.csv",content)
    with zipfile.ZipFile(ZIP) as archive:
        names=archive.namelist()
        assert archive.testzip() is None
        assert len(names)==len(set(names)) and set(names)=={r["path"] for r in index}|{"FILE_INDEX.csv"}
        for row in index:
            content=archive.read(row["path"])
            assert len(content)==row["bytes"] and hashlib.sha256(content).hexdigest()==row["sha256"]
    receipt={"zip":str(ZIP),"bytes":ZIP.stat().st_size,"sha256":hashlib.sha256(ZIP.read_bytes()).hexdigest(),
             "members":len(names),"crc":"PASS","index":"PASS","hashes":"PASS", "recomputation":read(DELIVERY/"保存结果核对.json"),
             "security_audit_performed":False,"external_review_received":False}
    ZIP.with_suffix(".delivery.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="第九轮中文说明、保存结果复核与审阅交付")
    options=parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--prepare",action="store_true")
    options.add_argument("--package",action="store_true")
    options.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.package:
        package()
    else:
        print(json.dumps(verify_saved(),ensure_ascii=False),flush=True)
