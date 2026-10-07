"""从保存结果说明覆盖差异和决策后果，不训练或重跑冻结实验。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "reports/research/510300_reclaim_path_discrimination_v1"
MAIN = BASE / "reports/research/510300_integrated_research_continuation_20260924"


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    destination = ROOT / "saved_interpretation.json"
    if destination.exists():
        raise RuntimeError("保存结果说明已经完成，不能改口径再次执行。")
    result = json.loads((ROOT / "result.json").read_text(encoding="utf-8"))
    event = pd.read_parquet(ROOT / "results/confirmed_event_outcomes.parquet")
    primary = event[event.primary & event.cost.eq("STRESS")].copy()
    common = primary[primary.internal_known]
    missing = primary[~primary.internal_known]
    excluded = common[~common.contraction_confirmed]
    predictions = pd.read_parquet(ROOT / "results/prequential_predictions.parquet")
    scored = predictions[predictions.primary & predictions.prediction_M1.notna() & predictions.actual_net_return.notna()]
    facts = pd.read_parquet(ROOT / "results/confirmation_facts.parquet").set_index("signal_id")
    fits = json.loads((ROOT / "results/saved_models.json").read_text(encoding="utf-8"))
    matched = 0
    for fit in fits:
        row = facts.loc[fit["signal_id"]]
        z = np.clip((row[fit["fields"]].to_numpy(float) - np.asarray(fit["mean"])) / np.asarray(fit["scale"]), -5, 5)
        prediction = fit["intercept"] + (z - np.asarray(fit["centered_mean"])) @ np.asarray(fit["beta"])
        saved = predictions[predictions.signal_id.eq(fit["signal_id"])][f"prediction_{fit['model']}"].iloc[0]
        assert abs(float(prediction) - float(saved)) < 1e-12
        assert fit["maximum_train_exit_idx"] < fit["decision_entry_idx"]
        matched += 1
    direction_changes = int(((scored.prediction_M0 > 0) != (scored.prediction_M1 > 0)).sum())
    detail = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "保存结果说明，不新增拟合或回测。",
        "primary_all_confirmed": len(primary),
        "primary_no_view": len(missing),
        "no_view_losing_events": int(missing.net_return.lt(0).sum()),
        "no_view_mean_net_event_return": float(missing.net_return.mean()),
        "common_mean_net_event_return": float(common.net_return.mean()),
        "all_confirmed_mean_net_event_return": float(primary.net_return.mean()),
        "no_view_losses_count_as_avoided_by_feature": False,
        "filtered_out_known_events": [{"signal_id": row.signal_id, "entry_date": row.entry_date,
                                       "contraction_percentage_points": float(row.contraction_point * 100),
                                       "net_return": float(row.net_return)} for row in excluded.itertuples()],
        "prediction_direction_changes": direction_changes,
        "M0_positive_predictions": int(scored.prediction_M0.gt(0).sum()),
        "M1_positive_predictions": int(scored.prediction_M1.gt(0).sum()),
        "saved_predictions_recomputed": matched,
        "worst_diagnostic_R_all_confirmed": float(primary.realized_R_at_original_full_quantity.min()),
        "R_definition": "原事件数量下，以实际开盘压力成本至固定失效价的估算损失为分母；并非新执行的1%风险预算账户。",
        "parameter_changes": 0, "new_fits": 0, "new_accounts": 0,
    }
    save(destination, detail)
    s = result["selection"]["STRESS"]
    prediction = result["prediction"]
    ci = s["paired_event_increment"]["ci95"]
    reference = primary[primary.signal_id.eq("RECLAIM_2023-09-21")].iloc[0]
    report = f"""已按本次用户指令完成相似破低形态的分叉检验：新增的固定低点参与信息没有改善入场取舍，不能晋升为交易条件。正式结果为FROZEN_NO_RELIABLE_RECLAIM_INTERNAL_INCREMENT。高夏普目标仍未完成。

2021-01-04至2026-08-14的完整母集有32次破低事件：11次收复确认，13次三日观察到期未确认，8次继续下跌失败。全部使用原冻结识别逻辑、原低点、原失效和冷却期；没有只挑成功图形。11次确认后事件均已成熟，另保留完整历史的事件与逐日过程。

唯一新增变量是：在破低当天固定全部成分股及权重，每只股票的参照低点锁定为事件之前20个交易日含分红收盘的最低值；确认时，仍在各自固定低点下方的权重是否减少。旧A/B/C/D的B检查固定核心行业的三日上涨参与，本轮检查全体固定成员相对自身事前低点的位置。二者不同，旧裁决保持原样。

数据口径要求准备与确认两时点的固定成员权重覆盖至少98%，并保留旧快照时点条件；缺失不填零、不重归一化，使用占比上下界，只有收缩下界大于零才算有收缩证据。11次确认中7次符合资料条件，4次保留NO_VIEW。宏观、资金、节日与计划事件只是当时已知背景，没有再叠加成入场门槛。

主比较在相同的7次合格事件中进行。压力成本下，新增条件保留6次，其中2盈4亏；排除1次，这1次盈利，没有避开亏损。被排除的是2022-09-07入场的事件：内部破低参与扩大约2.20个百分点，但原五日规则最后净收益约+0.58%。

另一个相反实例是2023-09-25入场：内部破低参与减少约{reference.contraction_point * 100:.2f}个百分点，条件明确通过，但确认后按原规则净收益为{reference.net_return * 100:.2f}%。从准备日到实际入场之前，含分红价格已经上行约{reference.confirmation_wait_return_not_earned * 100:.2f}%，这段确认前的反弹没有算成可获得收益。案例用于解释保存结果，正式比较仍包含全部合格事件。

相同覆盖下，保留全部事件的平均净事件收益为{s['baseline_mean_net_event_return'] * 100:.4f}%；加入过滤后，按原全部机会计量为{s['filtered_mean_per_original_event'] * 100:.4f}%，每机会减少{-s['paired_event_increment']['mean'] * 100:.4f}个百分点。三个相邻事件循环块重抽的95%开发区间为[{ci[0] * 100:.4f}, {ci[1] * 100:.4f}]个百分点。基础成本方向相同。这里没有把等额事件收益相加或复利成完整账户，也没有把未参与机会的比较值误称为已经成交收益。

覆盖差异尤其重要：4次NO_VIEW恰好全部亏损，平均净事件收益为{missing.net_return.mean() * 100:.2f}%。如果先排掉这些日期，再和全部11次比较，会制造出明显改善；但它来自资料覆盖，不能算成内部指标提前识别风险。本轮已经单独保留这4次，主要比较始终限制在同覆盖的7次。

预测比较同样只增加这一项信息。基础模型使用准备日趋势、收复幅度/ATR与确认耗时；增量模型加上固定低点参与变化。扩展训练、岭惩罚10及训练期标准化在看结果前固定，最少3条成熟历史沿用当前V2探索口径。本轮8次拟合形成4个成熟逐期预测，加入变量后MSE下降{prediction['relative_mse_improvement'] * 100:.2f}%，但前两个预测整体变差、后两个改善，配对误差改善区间跨零，未满足冻结门槛。两种模型对这4次预测都为负，方向变化{direction_changes}次；误差点值改善没有带来正负判断变化。这个样本量不足以证明预测优势。

已保存24条逐事件的成熟误差观察记录，每次最多使用此前12个已结束的预测，记录偏差、净成本补偿及最重亏损。大多数时点仍处于预热，能够观察的资料很少；本轮没有从这些记录建立或验证降权、暂停、恢复规则。一次亏损与规律退化仍不能混为一谈。

风险诊断中，全部11次确认事件的最差实现结果约{detail['worst_diagnostic_R_all_confirmed']:.2f}R。这里R以实际开盘压力成本至原冻结失效价的估算损失为分母，只用于解释退出误差，未构成新执行的1%风险预算账户。原规则仍保留T+1及收盘触发后下一可卖开盘，止损线不能作为保证成交价。股票ETF的T+1规则见[上交所说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)。

原破低收复协议没有事前盈利目标，所以3:1资格为NOT_DEFINED_REWARD_TARGET。至少3:1的要求保持；完整账户比较为NOT_RUN_NO_FROZEN_REWARD_TARGET，启停与恢复的账户效果也未运行。本轮没有额外设计策略或目标来填补这一缺项。事件过滤本身已经未通过，不能凭误差下降点值转入仓位映射。

这轮完成了母集、单项内部信息、真实确认后收益、过滤得失、逐期预测以及只使用成熟信息的误差记录。尚未证明事前选择优于简单基准，也没有证明规律退化应对能改善完整账户。该变量的这一固定用途保留为未获支持，不改阈值、低点窗口或只挑后半段救回。

原48个两成本历史事件标签与保存结果一致，两次历史截断的内部特征一致，8个保存预测从系数重新计算一致，训练结果时钟均早于决策。上述实现检查不证明策略有效。没有新增市场数据采集、审核ZIP或用户汇总表。

文献仅支持提出问题和保留风险边界：[Lo等的形态条件分布研究](https://www.nber.org/papers/w7613)不能替代510300证据；[Liquidity and Volatility](https://www.nber.org/papers/w27959)说明流动性供给可能承担意外信息与波动风险；[DSR研究](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)提示反复试验会抬高历史绩效。本轮未凭这些文献宣称存在交易优势，也未在缺少完整试验分布时补算DSR。
"""
    (ROOT / "研究结论.md").write_text(report, encoding="utf-8")
    status_path = MAIN / "current_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update({"latest_reclaim_primary_episodes": 32, "latest_reclaim_common_events": 7,
                   "latest_reclaim_avoided_losses": 0, "latest_reclaim_missed_winners": 1,
                   "latest_reclaim_prediction_scored_events": 4, "latest_reclaim_prediction_direction_changes": 0})
    save(status_path, status)
    main_report = MAIN / "最新研究结论.md"
    body = main_report.read_text(encoding="utf-8")
    marker = "正式状态FROZEN_NO_RELIABLE_RECLAIM_INTERNAL_INCREMENT，高夏普目标仍未完成。"
    addition = "同覆盖7次事件中保留6次（2盈4亏），排除的1次盈利，未避开亏损。另4次缺资料事件全亏，未将其计作指标预测成功。4个成熟逐期预测的MSE下降8.07%，但区间跨零、方向判断未变，不能晋升。"
    assert marker in body
    main_report.write_text(body.replace(marker, marker + addition, 1), encoding="utf-8")
    print(json.dumps(detail, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
