"""核对固定版与最近可用版模型的时钟和八因素，不生成新预测。"""
import json
from bisect import bisect_right
from pathlib import Path
import numpy as np
import pandas as pd
from research.continuation_strength_blend_inputs_v1 import current_close_signals
from research.entry_vintage_exit_inputs_v1 import prediction_identity
from research.within_cycle_exit_inputs_v1 import FEATURES
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_model_update_veto_preflight_20260909"
P128 = ROOT / "reports/research/510300_entry_vintage_exit_v1"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"


def main():
    prior_path = ROOT / "reports/research/510300_continuation_strength_blend_preflight_20260909/result.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    require(prior["status"] == "SAVED_CONTINUATION_PREDICTIONS_CLOCKS_STATES_AND_PARENT_TARGETS_READY", "原128固定模型输入确认缺失")
    for item in prior["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "原固定模型来源不再等于144输入确认")
    cfg_path = ROOT / "config/510300_episode_trend_admission_v1.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bound = {str(Path(item["path"])): item["sha256"] for item in cfg["frozen_files"]}
    original_path = ROOT / "config/510300_entry_vintage_exit_v1.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    model_path = ROOT / original["saved_models"]
    records = json.loads(model_path.read_text(encoding="utf-8"))["models"]
    fit_indices = [record["fit_index"] for record in records]
    require(fit_indices == sorted(set(fit_indices)), "原模型索引不唯一递增")
    by_date = {pd.Timestamp(record["fit_origin"]): record for record in records}
    identities = {record["fit_index"]: prediction_identity(record) for record in records}
    data = pd.read_parquet(ROOT / cfg["features"])
    require(digest(ROOT / cfg["features"]) == bound[str(Path(cfg["features"]))], "新分歧研究行情不再等于147绑定来源")
    sources = [ROOT / item["path"] for item in prior["sources"]]
    sources.extend([prior_path, cfg_path, original_path, model_path, ROOT / "docs/510300_MODEL_UPDATE_VETO_NEXT_20260909.md",
        ROOT / "research/continuation_strength_blend_inputs_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py",
        ROOT / "research/learned_cycle_exit_v1.py", ROOT / "reports/research/510300_episode_trend_admission_v1/saved_verification_receipt.json"])
    checks = []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, cfg["evaluation_start"]) if period == "evaluation" else (data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"])
        for cost in cfg["costs"]:
            decision_path = P128 / period / cost / "ENTRY_VINTAGE_EXIT_decisions.parquet"
            ledger_path = P128 / period / cost / "ENTRY_VINTAGE_EXIT_ledger.parquet"
            decisions = pd.read_parquet(decision_path)
            ledger = pd.read_parquet(ledger_path, columns=["date", "shares", "mark_clock"])
            signals = current_close_signals(decisions, ledger, frame, start, cost)
            available = signals.learning_status.eq("PREDICTION_AVAILABLE")
            require(np.isfinite(signals.loc[available, FEATURES].to_numpy(float)).all(), "已有可用固定预测缺少原八因素")
            require(np.isfinite(signals.loc[available, "continuation_prediction"]).all(), "已有固定预测数值缺失")
            different, newer, latest_status_counts = 0, 0, {}
            earliest_pair, latest_pair = None, None
            for row in signals.loc[available].itertuples():
                t = int(row.origin_index)
                fixed = by_date[pd.Timestamp(row.learning_fit_origin)]
                pos = bisect_right(fit_indices, t)-1
                require(pos >= 0, "固定预测可用但没有最近模型")
                latest = records[pos]
                require(latest["latest_exit_index"] <= latest["fit_index"] <= t and fixed["fit_index"] <= latest["fit_index"], "最近模型使用未来训练或早于原固定版本")
                require(pd.Timestamp(latest["fit_time"]) <= pd.Timestamp(row.origin)+pd.Timedelta(hours=15, minutes=5), "最近模型时点晚于当前收盘判断")
                require(pd.Timestamp(latest["fit_origin"]) == frame.date.iloc[latest["fit_index"]], "最近模型索引日期不匹配")
                require(row.fixed_prediction_identity == identities[fixed["fit_index"]], "原固定身份改变")
                latest_status_counts[latest["status"]] = latest_status_counts.get(latest["status"], 0)+1
                if latest["status"] == "FIT_COMPLETE":
                    model = latest["model"]
                    require(model["features"] == FEATURES and model["kind"] == "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "最近模型不是原八因素同类模型")
                    require(len(model["coefficients"]) == len(model["mean"]) == len(model["scale"]) == 8 and np.isfinite(model["coefficients"]).all() and
                        np.isfinite(model["mean"]).all() and np.isfinite(model["scale"]).all() and (np.asarray(model["scale"]) > 0).all() and
                        np.isfinite(model["intercept"]) and np.isfinite(model["feature_clip"]) and model["feature_clip"] > 0, "最近保存模型参数非法")
                    newer += int(latest["fit_index"] > fixed["fit_index"])
                    if identities[latest["fit_index"]] != row.fixed_prediction_identity:
                        different += 1
                        earliest_pair = earliest_pair or str(pd.Timestamp(row.origin).date())
                        latest_pair = str(pd.Timestamp(row.origin).date())
            parent_path = P143 / period / cost / "TREND_NOISE_REFERENCE_BLEND_decisions.parquet"
            require(digest(parent_path) == bound[str(parent_path.relative_to(ROOT))], "143父目标不再等于147绑定来源")
            parent = pd.read_parquet(parent_path, columns=["origin", "origin_index", "execution_date", "reference_weight"])
            require(np.array_equal(parent.origin_index, signals.origin_index) and pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(signals.origin)) and
                pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(signals.execution_date)), "原128状态与143父目标时钟不同")
            require(parent.reference_weight.between(0, 1).all(), "143父目标缺失或越界")
            sources.extend([decision_path, ledger_path, parent_path])
            checks.append({"period": period, "cost": cost, "decision_origins": len(signals), "fixed_available_origins": int(available.sum()),
                "explicit_no_mature_fixed_origins": int(signals.learning_status.eq("NO_VIEW_NO_MATURE_MODEL").sum()),
                "latest_record_status_counts_on_fixed_available": latest_status_counts, "newer_mature_version_origins": newer,
                "different_mature_prediction_identity_origins": different, "first_different_version_origin": earliest_pair, "last_different_version_origin": latest_pair,
                "complete_same_reference_eight_features": True, "new_predictions_generated": 0})
    receipt = {"checked_at": now(), "status": "FIXED_AND_LATEST_SAVED_MODEL_CLOCKS_AND_SAME_STATE_EIGHT_FEATURES_READY", "checks": checks,
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(sources))],
        "new_model_fits_or_predictions_or_targets_or_accounts": 0, "candidate_registered": False, "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", receipt, exclusive=True)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 147, "模型更新分歧的前序轮次不同")
    index["next_work"].update(status="MODEL_UPDATE_VETO_INPUT_CLOCKS_READY", input_preflight=str((OUT / "result.json").relative_to(ROOT)), registered=False)
    index["updated_at"] = now()
    write_json(index_path, index)
    print(json.dumps({"输入确认": checks, "新增模型预测目标账户": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
