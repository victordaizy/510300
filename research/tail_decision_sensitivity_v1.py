"""在保存的真实研究账户状态上，诊断尾部预测能否影响决策；不生成新收益账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest, local_import_closure, LATEST
from research.selected_mix_risk_before_band_v1 import fixed_choice
from research.adaptive_tail_daily_v1 import OUT as FORECASTS

OUT = ROOT / "reports/research/510300_tail_decision_sensitivity_v1"
BASE = ROOT / "reports/research/510300_selected_mix_nfci_increment_daily_v1/accounts"
STUDY = "510300_TAIL_DECISION_SENSITIVITY_V1"


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("决策敏感度诊断已固定，不覆盖")
    protocol = {"at": now(), "study_id": STUDY,
        "question": "当前账户中五日ES预测是否经常决定仓位，还是其他不变风险预算限制了预测改善的账户传导。",
        "method": "逐个保存决策日固定当时现金权益、持仓、峰值、原信号、成本、调仓带，仅替换五日ES输入。基准逐值复现。",
        "cases": ["BASELINE", "VOL20_FILTERED", "FIXED_HALF", "ADAPTIVE_TWO_YEAR", "ZERO_ES_DIAGNOSTIC_ONLY"],
        "zero_es_scope": "ES=0只用于删除市场尾部损失项后的同状态决策敏感度，仍保留调仓平仓费用、50%上限及10%跳空和回撤余量。它不是合法风险预测，不是候选，也不是可执行收益上界。",
        "path_limit": "每一天都使用原保存账户状态，不让反事实份额传播到后一天。不得把这些局部选择连起来计算收益或夏普。",
        "failure_preserved": "上一项自适应风险预测未通过门槛。本诊断不解除失败，不计算被门槛禁止的新账户。",
        "comparison": "两档既定费用全部决策日，另计原目标非零日，逐项披露份额与下单申请是否改变。",
        "new_accounts": 0, "new_model_fits": 0, "orders_authorized": False, "goal_achieved": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir()
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update([LATEST / "candidate_features.parquet", FORECASTS / "forecasts.json", FORECASTS / "result.json",
                    ROOT / "config/510300_incremental_selected_intent_mix_v1.json"])
    sources.update(BASE / cost / "BASELINE/decisions.parquet" for cost in ["BASE", "STRESS"])
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    print("同状态决策敏感度已固定；不生成反事实账户收益。", flush=True)


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, "来源改变：" + name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    data = pd.read_parquet(LATEST / "candidate_features.parquet")
    config = read(ROOT / "config/510300_incremental_selected_intent_mix_v1.json")
    predictions = {(r["origin_index"], r["policy"]): r for r in read(FORECASTS / "forecasts.json")}
    records, summaries, reproduced = [], [], 0
    cases = read(OUT / "protocol.json")["cases"]
    for cost_name in ["BASE", "STRESS"]:
        decisions = pd.read_parquet(BASE / cost_name / "BASELINE/decisions.parquet")
        for row in decisions.itertuples():
            t = row.origin_index
            price = float(data.close.iloc[t])
            for case in cases:
                # 原账户首个2019年末决策也在预测保存范围内。
                es = 0. if case == "ZERO_ES_DIAGNOSTIC_ONLY" else float(predictions[t, case]["es95"])
                choice = fixed_choice(row.shares_before_decision, row.decision_nav, row.decision_peak,
                                      price, row.source_target, row.risk_stopped, es, config["costs"][cost_name],
                                      config["tick"], config["lot"])
                if case == "BASELINE":
                    assert choice["target_shares"] == row.target_shares
                    assert choice["projected_target_shares"] == row.projected_target_shares
                    assert choice["band_suppressed"] == row.band_suppressed
                    np.testing.assert_allclose(choice["planned_tail_loss"], row.planned_tail_loss, atol=1e-8, rtol=0)
                    reproduced += 1
                records.append({"cost": cost_name, "case": case, "origin": row.origin, "origin_index": t,
                                "source_positive": bool(row.source_target > 0), "current_es95": es,
                                "base_target_shares": row.target_shares, "counterfactual_target_shares": choice["target_shares"],
                                "base_projected_shares": row.projected_target_shares,
                                "counterfactual_projected_shares": choice["projected_target_shares"],
                                "decision_changed": choice["target_shares"] != row.target_shares,
                                "pre_band_target_changed": choice["projected_target_shares"] != row.projected_target_shares,
                                "target_exposure_difference": (choice["target_shares"] - row.target_shares) * price / row.decision_nav,
                                "risk_required_reduction": choice["risk_required_reduction"]})
        print(f"{cost_name}：已核对{len(decisions)}个保存决策时点，未构造新净值。", flush=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(OUT / "same_state_decision_sensitivity.parquet", index=False)
    for (cost, case), part in frame.groupby(["cost", "case"], sort=True):
        positive = part[part.source_positive]
        summaries.append({"cost": cost, "case": case, "all_decisions": len(part),
                          "source_positive_decisions": len(positive), "changed_decisions": int(part.decision_changed.sum()),
                          "pre_band_target_changed_decisions": int(part.pre_band_target_changed.sum()),
                          "changed_fraction_of_all": float(part.decision_changed.mean()),
                          "changed_fraction_of_source_positive": float(positive.decision_changed.mean()),
                          "mean_exposure_difference_all": float(part.target_exposure_difference.mean()),
                          "maximum_exposure_increase": float(part.target_exposure_difference.max()),
                          "maximum_exposure_decrease": float(part.target_exposure_difference.min())})
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
         "status": "COMPLETED_LOCAL_DECISION_SENSITIVITY_NOT_A_STRATEGY",
         "results": summaries, "saved_baseline_decisions_reproduced": reproduced,
         "new_accounts": 0, "new_model_fits": 0, "new_independent_observations": 0,
         "goal_achieved": False, "orders_authorized": False}, True)
    for row in summaries:
        if row["cost"] == "STRESS":
            print(f"压力费用 {row['case']}：{row['changed_decisions']}/{row['all_decisions']}日局部决策改变。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="保存账户状态上的风险预测决策敏感度。")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
