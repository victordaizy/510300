"""保存账户空仓原因与预测校准诊断；不修改训练、预算或生成新账户。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.manufacturing_price_transmission_daily_v1 as study
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_manufacturing_price_transmission_calibration_v1"
STUDY = "510300_MANUFACTURING_PRICE_TRANSMISSION_CALIBRATION_V1"


def conditional_bias_interval(frame, selection, rng):
    """在完整交易日日历上抽取季度长度区块，不把筛出的原点当独立样本。"""
    n, block, repetitions = len(frame), 63, 2000
    mask = selection & frame.gross_return20.notna()
    weights = mask.to_numpy(float)
    errors = np.where(mask, frame.gross_return20 - frame.forecast_mean, 0.)
    values = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        count = weights[indices].sum()
        if count:
            values.append(float(errors[indices].sum() / count))
    return {"lower_95": float(np.quantile(values, .025)) if values else None,
        "upper_95": float(np.quantile(values, .975)) if values else None,
        "block_trading_days": block, "replications": repetitions, "valid_replications": len(values),
        "selection_adjusted": False, "primary_meaning": "条件20日毛收益减当时预测均值，不是可交易账户收益差。"}


def run():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("预测校准诊断已固定，不覆盖。")
    OUT.mkdir(parents=True, exist_ok=True)
    account = study.OUT / "accounts/STRESS/JOINT__FIXED20_NO_ADD"
    files = [account / "ledger.parquet", account / "decisions.parquet", study.OUT / "inputs/mature_labels.parquet",
        study.OUT / "inputs/decision_information.parquet", study.OUT / "results/zero_target_budget_diagnostic.json", study.OUT / "result.json"]
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "motivation": "已保存340次零仓目标均允许按原风险预算买首100份，其中322次预测均值非正；下一步区分预算约束与预测校准。",
        "status": "POST_RESULT_DIAGNOSTIC_NOT_CONFIRMATORY",
        "groups": ["全部空仓且价格弱势决策", "零仓且预测均值非正", "零仓但预测均值为正", "计划正仓位"],
        "selection": "只按原始决策时已经知道的字段分组；所有成熟正负结果保留，未成熟标签保留缺失。",
        "target": "原执行开盘至第20个后续开盘含股息毛收益，分别比较预测均值与实现；不是账户收益，不把后知成功案例直接变交易。",
        "uncertainty": "完整交易日日历63日循环区块2000次，seed2026092615；另固定每20日相位，报告非重叠样本数及偏差；固定2020-2023/2024-末端。",
        "restrictions": "不重训、不改阈值/方向/退出/风险预算，不运行任何新账户，也不把正平均实现收益视为可交易策略。",
        "new_accounts": 0, "new_model_fits": 0, "goal_achieved": False}, True)
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(OUT / "protocol.json"),
        "input_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in files}, "before_group_realized_returns_read": True}, True)
    decisions = pd.read_parquet(account / "decisions.parquet")
    labels = pd.read_parquet(study.OUT / "inputs/mature_labels.parquet")[["idx", "gross_return20", "exit_date"]]
    x = pd.read_parquet(study.OUT / "inputs/decision_information.parquet")[["idx", "pricing_stat_month"]]
    frame = decisions.merge(labels, on="idx", how="left", validate="one_to_one").merge(x, on="idx", how="left", validate="one_to_one")
    eligible = frame.shares_before_decision.eq(0) & frame.pressure5.gt(0) & frame.reference_price.notna()
    zero = eligible & frame.target_shares.eq(0)
    group_masks = {"ALL_FLAT_WEAK": eligible, "ZERO_NONPOSITIVE_MEAN": zero & frame.forecast_mean.le(0),
                   "ZERO_POSITIVE_MEAN": zero & frame.forecast_mean.gt(0), "POSITIVE_TARGET": eligible & frame.target_shares.gt(0)}
    assert int(zero.sum()) == 340 and int(group_masks["ZERO_NONPOSITIVE_MEAN"].sum()) == 322
    allowed = sum(study.parent.risk_valid(study.engine, 100, float(r["reference_price"]), r, study.distribution.COSTS["STRESS"])
                  for r in frame[zero].to_dict("records"))
    assert allowed == 340
    rng, results = np.random.default_rng(2026092615), []
    phase = (frame.idx - int(frame.idx.iloc[0])) % 20 == 0
    for name, mask in group_masks.items():
        mature = frame[mask & frame.gross_return20.notna()]
        nonoverlap = mature[phase.loc[mature.index]]
        periods = []
        for lo, hi in [(study.START, "2023-12-31"), ("2024-01-01", study.END)]:
            part = mature[mature.date.between(lo, hi)]
            periods.append({"start": lo, "end": hi, "n": len(part), "mean_prediction": float(part.forecast_mean.mean()),
                "mean_gross_realization": float(part.gross_return20.mean()), "mean_bias": float((part.gross_return20 - part.forecast_mean).mean())})
        results.append({"group": name, "decisions": int(mask.sum()), "mature_results": len(mature),
            "unmatured_results": int(mask.sum()) - len(mature), "distinct_release_months": int(mature.pricing_stat_month.nunique()),
            "mean_prediction20": float(mature.forecast_mean.mean()), "mean_gross_realization20": float(mature.gross_return20.mean()),
            "mean_bias": float((mature.gross_return20 - mature.forecast_mean).mean()),
            "realized_positive_fraction": float(mature.gross_return20.gt(0).mean()),
            "median_gross_realization20": float(mature.gross_return20.median()),
            "nonoverlap_rows": len(nonoverlap), "nonoverlap_mean_bias": float((nonoverlap.gross_return20 - nonoverlap.forecast_mean).mean()),
            "bias_interval": conditional_bias_interval(frame, mask, rng), "fixed_periods": periods})
    frame.to_parquet(OUT / "saved_decision_outcomes.parquet", index=False)
    result = {"at": now(), "study_id": STUDY, "status": "SAVED_ZERO_TARGET_CALIBRATION_DIAGNOSTIC_COMPLETE",
        "groups": results, "zero_targets_first_lot_risk_feasible": allowed,
        "new_accounts": 0, "new_model_fits": 0, "independent_validation": False,
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    skipped = next(r for r in results if r["group"] == "ZERO_NONPOSITIVE_MEAN")
    summary = f"""对已保存主压力账户的事后诊断已完成，没有生成新策略或新账户。340次零仓目标均允许在原风险预算下买入首100份，其中322次预测20日均值非正。风险预算不是这些零仓目标的直接拦截原因；这不证明实际存在可交易机会。

在预测均值非正而空仓的322次决策中，{skipped['mature_results']}个20日结果成熟，{skipped['unmatured_results']}个尚未成熟。成熟原点当时平均预测{skipped['mean_prediction20']:.4%}，后来每份含息毛收益平均{skipped['mean_gross_realization20']:.4%}，平均偏差（实现减预测）{skipped['mean_bias']:.4%}。63交易日循环区块的95%偏差区间为[{skipped['bias_interval']['lower_95']:.4%}, {skipped['bias_interval']['upper_95']:.4%}]。固定每20日相位只有{skipped['nonoverlap_rows']}个非重叠原点，不能把322天视为322次独立机会。

两个固定时期的偏差分别为{skipped['fixed_periods'][0]['mean_bias']:.4%}、{skipped['fixed_periods'][1]['mean_bias']:.4%}。这个诊断衡量已保存预测的校准，不是事前可用的纠偏规则；平均毛收益没有计入真实持仓路径、风险缩量、重叠信号争用和成本，不能直接转成账户收益或据此反向交易。下一步必须先判断是否存在当时可观察、可重复的偏差来源，不能根据这组事后均值修改旧模型。

主账户末端为按市价计量的模拟权益，仍有9,500份未平仓，已经计提压力退出储备；不是全部交易已经实现的利润，也不是实际账户持仓。高夏普目标尚未完成。
"""
    (OUT / "诊断结论.md").write_text(summary, encoding="utf-8")
    save(OUT / "completion.json", {"at": now(), "result_sha256": digest(OUT / "result.json"), "report_sha256": digest(OUT / "诊断结论.md"),
        "parent_result_unchanged": digest(study.OUT / "result.json") == read(OUT / "freeze.json")["input_hashes"][(study.OUT / "result.json").relative_to(ROOT).as_posix()],
        "new_accounts": 0, "new_model_fits": 0, "goal_status": "active", "goal_achieved": False}, True)
    print("空仓与预测校准诊断已完成，未修改原模型或账户。", flush=True)


if __name__ == "__main__":
    run()
