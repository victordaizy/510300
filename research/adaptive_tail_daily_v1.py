"""冻结并执行只用当前两年数据的五日尾部风险在线组合实验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest, local_import_closure, LATEST
from research.index_volatility_scaled_tail_v1 import prepare as old_prepare, OUT as FILTERED_OUT, PREVIOUS
from research.nfci_tail_forecast_increment_v1 import fz0, interval
from research.adaptive_tail_daily_core_v1 import inputs_from, predict, validate

OUT = ROOT / "reports/research/510300_adaptive_tail_daily_v1"
STUDY = "510300_ADAPTIVE_TAIL_DAILY_V1"
PRIMARY = "ADAPTIVE_TWO_YEAR"
POLICIES = ["BASELINE", "VOL20_FILTERED", "FIXED_HALF", PRIMARY]
CORE = ROOT / "research/adaptive_tail_daily_core_v1.py"


def prepare():
    data, labels, baseline = old_prepare()
    filtered = read(FILTERED_OUT / "forecasts.json")
    return inputs_from(data, labels), baseline, filtered


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮已冻结，不覆盖或按结果改门槛")
    inputs, baseline, filtered = prepare()
    checks = validate(inputs, baseline, filtered)
    protocol = {
        "at": now(), "study_id": STUDY, "primary": PRIMARY,
        "question": "专家的风险估计优劣会变时，只根据当前两年内成熟预测误差调整权重，是否比原模型及固定各半更准确。",
        "economic_mechanism": "历史经验分布保留长期尾部场景，尺度调整可响应当前波动；预测错误决定风险模型权重，不用随后账户利润反选。",
        "experts": {"BASELINE": "同原两年动量条件经验五日含分红收益分布，至少60行，否则同窗全部样本。",
                    "VOL20_FILTERED": "同样本以历史原点与当前原点已知20日波动之比调整log(1+r5)，沿用原5%分位数与尾部均值。"},
        "strict_two_year_training": "每日T取[T-2日历年,T)。内层验证原点S只取最近一年固定五交易日相位，S+6<T；在S的每个专家重新仅用原点>=T-2年且退出索引<S的标签。没有引用当年历史模型的更早训练资料。",
        "feature_warmup": "沿用各原点已收盘的20日动量和波动特征；短特征预热不当作额外风险标签。真实账户峰值不因训练窗口移动而重置。",
        "daily_weight": "内层至少40个已成熟非重叠五日验证观察。w_FHS=exp(-mean_FZ0_FHS)/(exp(-mean_FZ0_HS)+exp(-mean_FZ0_FHS))。温度固定为1，不扫描。",
        "combination": "对两专家的VaR与ES预测分别做同一凸组合。它是预测组合，不能称为两完整分布的精确混合分位数。",
        "fallback": "内层不足40个成熟有效观察，或任何内层预测不满足FZ0负ES定义，当天权重固定各半；保存全部内层记录，不删除预测失败日。当前专家资料非法则整次运行报错，不能跳日。",
        "controls": ["BASELINE", "VOL20_FILTERED", "FIXED_HALF"],
        "primary_comparators": ["BASELINE", "FIXED_HALF"],
        "periods": {"earlier": ["2015-01-05", "2019-12-31"], "main": ["2020-01-02", "2026-09-16"]},
        "evaluation": "复用原评分日期、标签、固定五日相位与完整区间；当期之外才成熟的标签不计入前区间。逐日预测全部保存。",
        "continuation_gate": "主候选对BASELINE及FIXED_HALF在两个评价区间共四项FZ0均值差95%区间上界均<0，四项平均分位数损失差均<=0，主候选各区间VaR实际跌破率<=10%。全部通过才另立账户协议，失败新账户NOT_RUN。",
        "statistics": "配对循环区块自助法，每块4个非重叠五日观察，2000次；种子20260925。差值为候选减对照，越低越好；未校正项目级反复研究。",
        "mechanism_source": "https://public.econ.duke.edu/~ap172/Patton_Ziegel_Chen_JoE_2019.pdf",
        "source_scope": "论文支持VaR与ES联合评分；本轮软权重、嵌套窗口和门槛是事前固定的研究设计，不是论文已证明适用于510300的结论。",
        "deduplication": "旧prequential_one_day_tail_discovery_v1是单日持有优势预测的百分位、均值和投票；本轮是两个五日VaR/ES专家按成熟联合评分调整权重。旧downside_reference_pair组合策略仓位，本轮不改信号或仓位。",
        "selection_history": "本轮在已经看到固定尺度法前期改善而近期无确认后提出。历史区间已反复研究，本结果属于开发证据，不能改称独立验证。",
        "unchanged": ["全部原价格与分红资料", "两年训练合同", "20日特征窗口", "5%经验尾部估计", "账户尾部预算", "交易成本与T+1规则"],
        "parameter_search": 0, "new_regression_fits": 0, "planned_new_accounts_before_gate": 0,
        "checks": checks, "goal_achieved": False, "orders_authorized": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir()
    for path in [Path(__file__), CORE]:
        shutil.copy2(path, OUT / "code" / path.name)
    save(OUT / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__), CORE})
    sources.update([LATEST / "candidate_features.parquet", ROOT / "data/reference/510300_dividends.csv",
                    PREVIOUS / "forecasts.json", PREVIOUS / "mature_forecast_scores.parquet",
                    FILTERED_OUT / "forecasts.json"])
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
         "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    print("日更风险组合已固定：两年窗口内部顺序验证，对照原模型及固定各半；不按账户收益挑权重。", flush=True)


def verify():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, expected in frozen["sources"].items():
        assert digest(ROOT / name) == expected, "冻结来源发生变化：" + name


def run():
    verify()
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    inputs, baseline, filtered = prepare()
    base_map = {r["origin_index"]: r for r in baseline}
    filtered_map = {r["origin_index"]: r for r in filtered}
    receipts, all_predictions, inner_rows = [], [], []
    pairs = {}
    for number, saved in enumerate(baseline, 1):
        t = saved["origin_index"]
        metadata, predictions, inner = predict(inputs, t)
        np.testing.assert_array_equal(metadata["label_indices"], base_map[t]["label_indices"])
        for name, old in [("BASELINE", base_map), ("VOL20_FILTERED", filtered_map)]:
            np.testing.assert_allclose(predictions[name], [old[t]["q05"], old[t]["es95"]], atol=1e-14, rtol=0)
        assert all(row["validation_maturity_index"] < t for row in inner)
        assert all(row["latest_training_exit_index"] < row["validation_origin_index"]
                   for row in inner if "latest_training_exit_index" in row)
        receipts.append(metadata)
        inner_rows.extend(inner)
        pairs[t] = predictions
        for policy, (q, es) in predictions.items():
            all_predictions.append({"origin_index": t, "origin": inputs.dates[t], "policy": policy,
                                    "q05": q, "es95": es, "filtered_weight": metadata["filtered_weight"] if policy == PRIMARY else
                                    {"BASELINE": 0., "VOL20_FILTERED": 1., "FIXED_HALF": .5}[policy],
                                    "available": True})
        if number % 400 == 0:
            print(f"两年嵌套预测完成{number}/{len(baseline)}日，全部成熟时序检查通过。", flush=True)
    save(OUT / "forecasts.json", all_predictions, True)
    save(OUT / "daily_weight_receipts.json", receipts, True)
    pd.DataFrame(inner_rows).to_parquet(OUT / "nested_validation.parquet", index=False)
    saved_scores = pd.read_parquet(PREVIOUS / "mature_forecast_scores.parquet")
    saved_scores = saved_scores[saved_scores.policy.eq("BASELINE")].sort_values("origin_index").reset_index(drop=True)
    parts = []
    for policy in POLICIES:
        part = saved_scores.copy()
        part["policy"] = policy
        part["q05"] = [pairs[t][policy][0] for t in part.origin_index]
        part["es95"] = [pairs[t][policy][1] for t in part.origin_index]
        part["fz0"] = fz0(part.return5, part.q05, -part.es95)
        part["pinball"] = (.05 - part.return5.lt(part.q05).astype(float)) * (part.return5 - part.q05)
        part["breach"] = part.return5.lt(part.q05)
        part["used_macro_condition"] = False
        parts.append(part)
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(OUT / "mature_forecast_scores.parquet", index=False)
    phase = scores[scores.fixed_nonoverlap_phase]
    statistics, comparisons, annual = [], [], []
    rng = np.random.default_rng(20260925)
    for period in ["earlier", "main"]:
        by_policy = {name: phase[phase.period.eq(period) & phase.policy.eq(name)].sort_values("origin_index") for name in POLICIES}
        for name, part in by_policy.items():
            statistics.append({"period": period, "policy": name, "observations": len(part),
                               "mean_fz0": float(part.fz0.mean()), "mean_pinball": float(part.pinball.mean()),
                               "breaches": int(part.breach.sum()), "breach_rate": float(part.breach.mean())})
        for comparator in ["BASELINE", "FIXED_HALF"]:
            new, control = by_policy[PRIMARY], by_policy[comparator]
            np.testing.assert_array_equal(new.origin_index, control.origin_index)
            comparisons.append({"period": period, "policy": PRIMARY, "comparator": comparator,
                                "pinball_mean_difference": float(new.pinball.mean() - control.pinball.mean()),
                                "breach_rate": float(new.breach.mean()),
                                **interval(new.fz0.to_numpy() - control.fz0.to_numpy(), rng)})
    for (year, policy), part in phase.groupby([phase.origin.dt.year, "policy"], sort=True):
        annual.append({"year": int(year), "policy": policy, "observations": len(part),
                       "mean_fz0": float(part.fz0.mean()), "mean_pinball": float(part.pinball.mean()),
                       "breach_rate": float(part.breach.mean())})
    save(OUT / "yearly_risk_scores.json", annual, True)
    gate = len(comparisons) == 4 and all(r["upper_95"] < 0 and r["pinball_mean_difference"] <= 0
                                       and r["breach_rate"] <= .1 for r in comparisons)
    weights = pd.DataFrame(receipts)
    result = {"at": now(), "study_id": STUDY,
              "status": "RISK_FORECAST_INCREMENT_SUPPORTED_ACCOUNT_PENDING" if gate else "FROZEN_NO_RELIABLE_ADAPTIVE_TAIL_INCREMENT",
              "continuation_gate": gate, "period_scores": statistics, "comparisons": comparisons,
              "daily_origins": len(baseline), "new_combined_risk_estimates": 2 * len(baseline),
              "reproduced_saved_expert_predictions": 2 * len(baseline),
              "nested_validation_predictions": 2 * sum(r["status"] == "SCORED" for r in inner_rows),
              "daily_adaptive_weight_updates": int(weights.weight_status.eq("DAILY_TWO_YEAR_NESTED_SCORE_WEIGHT").sum()),
              "weight_status_counts": weights.weight_status.value_counts().to_dict(),
              "filtered_weight_minimum": float(weights.filtered_weight.min()),
              "filtered_weight_median": float(weights.filtered_weight.median()),
              "filtered_weight_maximum": float(weights.filtered_weight.max()),
              "new_parameter_searches": 0, "new_regression_fits": 0, "new_accounts": 0,
              "account_status": "NOT_RUN_PENDING_SEPARATE_ACCOUNT_PROTOCOL" if gate else "NOT_RUN_FAILED_RISK_GATE",
              "new_independent_market_observations": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="五日VaR/ES的两年嵌套日更权重实验。")
    parser.add_argument("command", choices=["freeze", "run"])
    options = parser.parse_args()
    {"freeze": freeze, "run": run}[options.command]()
