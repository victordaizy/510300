"""用已经取得的新价格结算旧M1待评价事件，不重拟合或重启已冻结策略。"""
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
from research.selected_mix_reappraisal_v1 import read, save, now, digest, local_import_closure
from research.local_pending_m1_completion_v1 import features_for, predict_saved

OUT = ROOT / "reports/research/510300_m1_pending_label_settlement_v1"
OLD = ROOT / "reports/research/510300_local_pending_m1_completion_v1"
PARENT = ROOT / "reports/research/510300_new_m1_consensus_training_v1"
PRICE = ROOT / "reports/research/510300_original_frozen_sse_completion_20260925"
STUDY = "510300_M1_PENDING_LABEL_SETTLEMENT_V1"


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("待评价事件结算已经固定，不重复覆盖")
    protocol = {"at": now(), "study_id": STUDY,
        "purpose": "原2026年8月M1事件缺9月17、18、21、22日行情；既有新增行情已补齐，只结算三份保存预测的固定五日实际结果。",
        "event": "2026-08", "historical_decision": "2026-09-15T15:00:00+08:00",
        "entry": "2026-09-16开盘", "exit": "2026-09-22收盘",
        "label": "原下一交易日开盘买入至第五个交易日收盘卖出的毛收益，入场日不计除息权益；不是五个开盘至开盘区间。",
        "frozen_models": "逐值复算原三条保存预测、原16个成熟事件特征；不重新训练，不使用新增标签调整系数或门槛。",
        "source_contract": "继承9月25日双行情源逐字段一致、旧价格前缀一致及上交所完整公告覆盖的独立来源协议；不冒充旧双官网来源接纳通过。",
        "evaluation": "单事件误差和方向如实保存；另合并原10个顺序评价事件形成11事件描述，不改原训练样本或历史失败。",
        "timing": "模型实际创建于9月22日收盘以后，历史重建输出不是事前承诺预测，新标签不计独立前向验证。",
        "limits": "原宏观表示STOP_CURRENT_REPRESENTATION_NO_PARAMETER_RESCUE继续保留；单个事件不构成重开依据。旧账户风险合同不同，不迁移其夏普到当前目标。",
        "new_fits": 0, "new_accounts": 0, "new_downloads": 0, "orders_authorized": False, "goal_achieved": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "code").mkdir()
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "protocol.json", protocol, True)
    sources = local_import_closure({Path(__file__)})
    sources.update([OLD / "results/保存模型的历史重建输出.csv", OLD / "results/补齐的历史事件特征.csv",
                    OLD / "inputs/common_market.parquet", OLD / "inputs/latest_prices.parquet",
                    OLD / "inputs/pending.csv", OLD / "inputs/mature_event_features.csv",
                    OLD / "inputs/saved_final_models.json", OLD / "inputs/dividends.csv",
                    OLD / "inputs/calendar.csv", OLD / "data_boundary.json",
                    PARENT / "results/逐期预测.csv", PARENT / "branch_decision.json",
                    PRICE / "candidate_features.parquet", PRICE / "protocol.json", PRICE / "result.json"])
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(sources)}}, True)
    print("M1旧待评价事件结算已固定：不拟合、不生成新账户、不改原失败。", flush=True)


def market_input():
    common = pd.read_parquet(OLD / "inputs/common_market.parquet")
    prices = pd.read_parquet(PRICE / "candidate_features.parquet")
    previous = pd.read_parquet(OLD / "inputs/latest_prices.parquet")
    columns = ["date", "open", "high", "low", "close"]
    pd.testing.assert_frame_equal(prices[columns].iloc[:len(previous)].reset_index(drop=True),
                                  previous[columns].reset_index(drop=True), check_exact=True, check_dtype=False)
    pd.testing.assert_frame_equal(prices[columns].iloc[:len(common)].reset_index(drop=True),
                                  common[columns].reset_index(drop=True), check_exact=True, check_dtype=False)
    dividend = pd.read_csv(OLD / "inputs/dividends.csv")
    by_date = dividend.assign(ex_date=pd.to_datetime(dividend.ex_date)).groupby("ex_date").cash_dividend_per_share.sum()
    tail = prices.loc[prices.date.gt(common.date.max()), columns].copy()
    tail["dividend"] = tail.date.map(by_date).fillna(0.)
    prior_close, prior_wealth = float(common.close.iloc[-1]), float(common.wealth.iloc[-1])
    total, wealth = [], []
    for row in tail.itertuples():
        simple = (row.close + row.dividend) / prior_close - 1.
        prior_wealth *= 1. + simple
        total.append(simple)
        wealth.append(prior_wealth)
        prior_close = row.close
    tail["total_simple"], tail["wealth"] = total, wealth
    result = pd.concat([common[[*columns, "dividend", "total_simple", "wealth"]], tail], ignore_index=True)
    assert not result.date.duplicated().any() and result.date.is_monotonic_increasing
    return result


def run():
    frozen = read(OUT / "freeze.json")
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, "来源改变：" + name
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    source = read(PRICE / "protocol.json")
    assert source["source_checks"]["new_unreconciled_dividend_announcements"] == 0
    market = market_input()
    raw = pd.read_csv(OLD / "inputs/pending.csv").iloc[0].to_dict()
    features = features_for(raw, market)
    assert features["stat_month"] == "2026-08" and features["observation_date"] == "2026-09-15"
    old_features = pd.read_csv(OLD / "results/补齐的历史事件特征.csv").iloc[0]
    fields = ["reference_close", "vol20", "price_pre20_return", "initial_response", "price_log_sigma20", "spread_surprise_proxy_pp"]
    np.testing.assert_allclose([features[k] for k in fields], old_features[fields].to_numpy(float), atol=1e-12, rtol=0)
    prior_events = pd.read_csv(OLD / "inputs/mature_event_features.csv")
    for row in prior_events.to_dict("records"):
        reproduced = features_for(row, market)
        np.testing.assert_allclose([reproduced[k] for k in fields], [row[k] for k in fields], atol=1e-12, rtol=0)
    required = read(OLD / "data_boundary.json")["required_horizon_dates"]
    block = market[market.date.isin(pd.to_datetime(required))].copy()
    assert block.date.dt.strftime("%Y-%m-%d").tolist() == required and len(block) == 5
    target = float((block.close.iloc[-1] + block.dividend.iloc[1:].sum()) / block.open.iloc[0] - 1.)
    models = read(OLD / "inputs/saved_final_models.json")
    predictions = pd.read_csv(OLD / "results/保存模型的历史重建输出.csv")
    np.testing.assert_allclose([predict_saved(models[r.model], features) for r in predictions.itertuples()],
                               predictions.prediction_gross_5d_return, atol=1e-14, rtol=0)
    prefix = features_for(raw, market[market.date.le(features["observation_date"])])
    for model in models.values():
        assert predict_saved(model, prefix) == predict_saved(model, features)
    predictions["actual_return"] = target
    predictions["label_status"] = "MATURED_FIXED_FIVE_SESSION_LABEL"
    predictions["prediction_error"] = predictions.prediction_gross_5d_return - target
    predictions["squared_error"] = predictions.prediction_error.pow(2)
    predictions["direction_correct"] = predictions.prediction_gross_5d_return.gt(0).eq(target > 0)
    predictions.to_parquet(OUT / "settled_predictions.parquet", index=False)
    block.to_parquet(OUT / "fixed_horizon_prices.parquet", index=False)
    old_predictions = pd.read_csv(PARENT / "results/逐期预测.csv")
    combined = []
    for row in predictions.itertuples():
        previous = old_predictions[old_predictions.model.eq(row.model)]
        errors = np.r_[(previous.prediction - previous.target5).to_numpy(), row.prediction_error]
        combined.append({"model": row.model, "previous_evaluation_events": len(previous),
                         "completed_evaluation_events": len(errors), "mse": float(np.square(errors).mean()),
                         "rmse": float(np.sqrt(np.square(errors).mean())),
                         "direction_accuracy": float((previous.prediction.gt(0).eq(previous.target5.gt(0)).sum() + int(row.direction_correct)) / len(errors))})
    values = {r["model"]: r for r in combined}
    improvement = 1. - values["B_PRICE_SURPRISE"]["mse"] / values["A_PRICE"]["mse"]
    result = {"at": now(), "study_id": STUDY, "status": "COMPLETED_PREVIOUSLY_PENDING_M1_LABEL_ORIGINAL_REJECTION_PRESERVED",
              "stat_month": "2026-08", "entry_date": required[0], "exit_date": required[-1],
              "actual_gross_return": target, "settled_predictions": predictions.to_dict("records"),
              "combined_descriptive_scores": combined, "surprise_mse_improvement_vs_price": improvement,
              "newly_completed_event_labels": 1, "old_mature_features_reproduced": len(prior_events),
              "same_negative_prediction_no_new_long_entry": bool(predictions.prediction_gross_5d_return.lt(0).all()),
              "new_fits": 0, "new_accounts": 0, "new_downloads": 0, "new_independent_forward_events": 0,
              "old_terminal_decision": read(PARENT / "branch_decision.json")["branch_decision"],
              "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print(f"M1固定五日毛收益{target:.4%}；已结算3份保存预测，合计11事件中预期差相对价格MSE改善{improvement:.4%}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="旧M1五日事件的保存预测结算。")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
