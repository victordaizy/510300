"""解释已保存的负Gamma库存实验；不重训、不回测、不调整门槛。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_synthetic_short_gamma_variance_v1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    assert not (OUT / "saved_interpretation.json").exists(), "本轮解释已经完成，不重复追加。"
    r = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    v = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    p, cmp = r["primary"], r["comparison"]
    scores = r["prediction_metrics"]
    price = next(m for m in r["all_metrics"] if m["cost"] == "STRESS" and m["policy"] == "PRICE")
    uncon = next(m for m in r["all_metrics"] if m["cost"] == "STRESS" and m["policy"] == "UNCONDITIONAL")
    fixed = next(m for m in r["all_metrics"] if m["cost"] == "STRESS" and m["policy"] == "FIXED25")
    decision = pd.read_parquet(OUT / "accounts/STRESS/MACRO_decisions.parquet")
    phase = pd.read_parquet(OUT / "results/nonoverlap_prediction_phases.parquet")
    phase_p = phase[phase.model.eq("PRICE")].set_index("phase")
    phase_m = phase[phase.model.eq("MACRO")].set_index("phase")
    window = pd.read_parquet(OUT / "results/all_rolling_two_years.parquet")
    eligible = window[window.policy.eq("MACRO") & window.net_sharpe.ge(1.2)]
    price_gain = 1 - scores["PRICE"]["qlike"] / scores["PRICE"]["qlike_persistence"]
    macro_gain = 1 - scores["MACRO"]["qlike"] / scores["MACRO"]["qlike_persistence"]
    macro_worse = scores["MACRO"]["qlike"] / scores["PRICE"]["qlike"] - 1
    diagnostic = {
        "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "study_status": r["status"],
        "PRICE_QLIKE_improvement_vs_persistence": price_gain,
        "MACRO_QLIKE_improvement_vs_persistence": macro_gain,
        "MACRO_QLIKE_worsening_vs_PRICE": macro_worse,
        "MACRO_nonoverlap_phases_better_than_PRICE": int((phase_m.qlike < phase_p.qlike).sum()),
        "nonoverlap_phases_are_alternative_overlapping_cohorts": True,
        "known_decision_days": int(decision.known.sum()),
        "pricing_proxy_gate_days": int(decision.iv_gate.fillna(False).sum()),
        "positive_target_inventory_days": int(decision.target_shares.gt(0).sum()),
        "sharpe_only_two_year_windows": len(eligible),
        "max_cagr_among_sharpe_only_passers": eligible.annualized_return.max() if len(eligible) else None,
        "same_saved_share_path_gross_pnl": p["same_share_path_gross_pnl"], "actual_friction": p["cost_cny"],
        "net_pnl": p["ending_equity"] - 200000,
        "new_models_or_accounts": 0, "parameter_changes": 0,
        "finding": "方差预测改善没有转化为足以覆盖成本的库存收益；宏观方差增量未成立。现货库存未收到期权权利金。",
        "current_asset_scope_question": "PENDING_USER_REPLY_ETF_CASH_OR_PROTECTED_510300_OPTIONS",
    }
    save(OUT / "saved_interpretation.json", diagnostic)
    reference = {
        "url": "https://www.nber.org/papers/w31833", "title": "Risk Preferences Implied by Synthetic Options",
        "authors": "Ian Dew-Becker; Stefano Giglio", "year": 2023,
        "use": "区分动态复制期权形状与真实期权市场收益。仅方法论来源，不把美国样本当510300有效性证据。",
        "market_data_downloads": 0,
    }
    save(OUT / "external_reference.json", reference)
    text = f"""本轮已经完成2,722次逐日方差模型更新和10个完整ETF现金账户。没有找到满足净夏普1.2、年化10%、最大回撤10%三项目标的策略。主宏观方案压力净夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、最大回撤{abs(p['max_drawdown']):.2%}，20万元期末为{p['ending_equity']:,.2f}元。没有将局部高夏普或更低回撤解释为目标完成。

这是对“结构化空波动暴露”的进一步检验。原上一轮预测反弹方向，本轮改为预测未来20日实际方差，并构造一条价格下降时增加、价格上升时减少的库存形状。宏观信息进入方差预测，检验的是潜在赔付风险，不要求慢宏观变量直接预测明日涨跌。旧研究的拒绝结果保持。

账户只持有510300.SH及现金，没有卖出真实期权，也没有收到任何权利金。每20个交易日按事前固定时点重置价格锚，期间根据价格、剩余期限和已知隐含波动率计算虚拟看跌卖方Delta。该局部仓位形状具有负Gamma特征；实际收益的偏度还受择时、成本、跳空和风险减仓影响，不能预先指定。主账户实现日收益偏度约{p['daily_skew']:.2f}，为正偏；它并非已经实现的负偏收益产品。

每天使用当时向前两个日历年内、退出开盘严格早于本次09:00决策的成熟样本。价格模型使用1/5/22日已实现方差，宏观模型再加入DR007相对政策利率的缺口、社融同比三个月变化、已知美国市场收益与VIX变化。两个模型使用相同训练样本，标准化和残差分布仅由该样本决定。对数方差预测通过训练残差指数均值修正回均值尺度，并另估计上侧风险尺度。不搜索特征、窗口或惩罚强度。

隐含与预测波动统一放入固定20日等效期限的虚拟看跌模型，再以模型价格差是否覆盖预设压力成本作为资格条件。这只是风险定价代理，未把约45日的观察期权直接称为20日可成交套利。资料仍为事后采集，期权额外滞后一天，首次发布时钟未认证，属于开发回放。

风险规则沿用上一轮：五日ES95预算权益2.5%、10%标的压力情景预算权益5%、持仓上限50%，随距10%回撤线的空间缩小而降风险。权益回撤触及10%时下一可卖开盘退出。本轮的风险减仓和退出不受调仓带宽阻挡；普通细碎变化沿用10个百分点带宽。资金和T+1、整手、最低佣金、滑点、登记及分红支付均入账。降低仓位只能改变风险承担量，不能替代正收益来源。

1,341个成熟预测中，价格模型QLIKE为{scores['PRICE']['qlike']:.4f}，相比持续使用过去20日方差改善{price_gain:.2%}；宏观模型QLIKE为{scores['MACRO']['qlike']:.4f}，相比同一基线改善{macro_gain:.2%}，但比价格模型恶化{macro_worse:.2%}。20个固定非重叠相位中，仅{diagnostic['MACRO_nonoverlap_phases_better_than_PRICE']}个显示宏观优于价格。不同相位本身也不是20组独立验证。宏观预测上侧风险尺度被突破约{scores['MACRO']['upper90_breach_rate']:.2%}，校准程度不能直接转化为收益优势。

压力账户中，固定25%库存对照净夏普{fixed['net_sharpe']:.3f}、年化{fixed['annualized_return']:.2%}；无择时负Gamma形状净夏普{uncon['net_sharpe']:.3f}、年化{uncon['annualized_return']:.2%}；价格方差择时净夏普{price['net_sharpe']:.3f}、年化{price['annualized_return']:.2%}；宏观方差择时净夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}。前两对照同样受风险预算与信息覆盖限制，固定25%不是全期实际保持25%。

主宏观账户相对价格账户年化算术收益增量{cmp['annual_arithmetic_increment']:.2%}，开发区块95%区间为[{cmp['ci95'][0]:.2%}, {cmp['ci95'][1]:.2%}]。2021—2023年增量为负，2024年至终点为正。后段主夏普约1.018，但年化仅约1.06%，不能只报告夏普而漏掉收益容量。

主账户共有{diagnostic['known_decision_days']}个资料合格决策日，{diagnostic['pricing_proxy_gate_days']}天通过模型价格差门槛，{diagnostic['positive_target_inventory_days']}天形成正库存目标，最终26次买入调整、29次卖出调整。平均账户暴露仅{p['mean_exposure']:.2%}。同一份额路径的成本前利润{p['same_share_path_gross_pnl']:,.2f}元，费用及滑点{p['cost_cny']:,.2f}元，净亏损{200000-p['ending_equity']:,.2f}元。成本前归因没有重新优化仓位，也不能当成另一个零成本策略。

全部875个滚动两年窗口中，最高夏普约{r['rolling_two_years']['max_sharpe']:.3f}，最高年化约{r['rolling_two_years']['max_cagr']:.2%}。仅夏普达到1.2的窗口有{diagnostic['sharpe_only_two_year_windows']}个，但没有任何窗口同时满足三项目标。历史窗口高度重叠，本研究也已使用反复查看的市场历史，不具备独立前向证据。

机制上，需要把“承担期权形状的风险”和“在期权市场收到风险溢价”分开。现货买卖本身没有产生一笔外部保费。基于离散Delta形状的库存策略，收益来自实际买卖路径；即使看到隐含波动率高于预测方差，也不能在现货账上自动计入两者差值。[Dew-Becker与Giglio的原研究](https://www.nber.org/papers/w31833)专门比较合成期权与真实期权的收益，两者不能互相替代。这也解释了为什么预测方差优于持续基线，仍可能无法形成成本后高夏普账户。

本轮已逐项复算全部{v['saved_predictions_recomputed']}个保存预测，检查两年窗口和标签成熟时钟；两个历史截断原点一致；10个账户的现金、份额、股息应收与权益核对通过，所有虚拟期权现金流均为零。没有新采集、没有实盘或订单。

用户要求持续推进目标，本轮作为有意义新进展保留，目标仍未完成。下一条具体待明确的方向是允许真实期权现金流的有保护看跌价差研究：它需要本任务资产范围明确扩展，不能直接继承仓库另一个目标、本金不同的期权任务合同。研究范围说明已写好；当前尚未新增该期权实验的标签、模型或账户。
"""
    (OUT / "研究结论.md").write_text(text, encoding="utf-8")
    state_path = MAIN / "current_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(latest_goal_turn_classification="PROGRESS_SYNTHETIC_GAMMA_VARIANCE_TEN_ACCOUNTS_COMPLETED",
                 same_condition_consecutive_no_progress_goal_turns=0, goal_achieved=False,
                 latest_user_requested_study=str(OUT.relative_to(ROOT)), latest_user_requested_study_status=r["status"],
                 latest_research_workflow="CURRENT_SCOPE_COMPLETED_ASSET_SCOPE_CLARIFICATION_PENDING",
                 active_blocker_id=None, active_blocker_description=None,
                 latest_synthetic_gamma_primary=p, latest_synthetic_gamma_model_updates=r["model_updates"],
                 remaining_research_question="有保护510300期权可直接检验权利金补偿；等待本任务资产范围澄清，不能把合成形状当真实期权收益。")
    save(state_path, state)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = json.loads(mandate_path.read_text(encoding="utf-8"))
    mandate.update(as_of_date="2026-09-25", continuation_requested_at=now(), current_round=r["study_id"],
                   current_protocol=str((OUT / "protocol.json").relative_to(ROOT)),
                   latest_progress_receipt=str((OUT / "result.json").relative_to(ROOT)),
                   last_research_result="合成负Gamma与方差预测10账户完成：主压力夏普-0.077、年化-0.10%、回撤3.26%；目标未达，宏观未改善价格方差模型。",
                   latest_integrated_experiment=r["study_id"], goal_achieved=False)
    save(mandate_path, mandate)
    report_path = MAIN / "最新研究结论.md"
    previous = report_path.read_text(encoding="utf-8")
    addition = "2026-09-25继续研究：已完成两年每日更新的前瞻方差、合成负Gamma库存实验，共2,722模型更新和10个账户。主压力净夏普-0.077、年化-0.10%、回撤3.26%；宏观没有改善价格方差预测或完整账户。875个滚动两年窗口无三项目标同时达标。所有虚拟权利金现金流为零，未将现货仓位形状称为真实卖期权收益。\n\n详见[本轮研究结论](../510300_synthetic_short_gamma_variance_v1/研究结论.md)。有保护看跌价差的具体研究说明已准备，当前资产范围问题等待用户澄清。\n\n---\n\n"
    report_path.write_text(addition + previous, encoding="utf-8")
    print(json.dumps(clean(diagnostic), ensure_ascii=False))


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


if __name__ == "__main__":
    main()
