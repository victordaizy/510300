"""确认已有预测的当前持仓、模型时钟及身份，不生成新预算或账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.entry_vintage_exit_inputs_v1 import prediction_identity
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_continuation_strength_blend_preflight_20260909"
P128 = ROOT / "reports/research/510300_entry_vintage_exit_v1"


def main():
    c128 = ROOT / "config/510300_entry_vintage_exit_v1.json"
    c131 = ROOT / "config/510300_vintage_reference_risk_v1.json"
    c143 = ROOT / "config/510300_trend_noise_reference_blend_v1.json"
    original, reference, latest = [json.loads(path.read_text(encoding="utf-8")) for path in [c128, c131, c143]]
    require(original["specification"]["modes"]["1"]["days"] == 60, "原退出最长持仓不再是60日")
    old_bound = {str(Path(item["path"])): item["sha256"] for item in reference["frozen_files"]}
    model_path = ROOT / original["saved_models"]
    model_receipt = json.loads((P128 / "reused_models_receipt.json").read_text(encoding="utf-8"))
    require(digest(model_path) == model_receipt["source_sha256"], "保存模型不再等于128复用的来源")
    records = json.loads(model_path.read_text(encoding="utf-8"))["models"]
    by_day = {pd.Timestamp(record["fit_origin"]): record for record in records}
    identities = {date: prediction_identity(record) for date, record in by_day.items()}
    data = pd.read_parquet(ROOT / latest["features"])
    sources = [c128, c131, c143, model_path, P128 / "reused_models_receipt.json", P128 / "saved_verification_receipt.json",
        ROOT / "docs/510300_CONTINUATION_STRENGTH_BLEND_NEXT_20260909.md",
        ROOT / latest["features"], ROOT / latest["dividends"], ROOT / "research/entry_vintage_exit_inputs_v1.py",
        ROOT / "reports/research/510300_trend_noise_reference_blend_v1/saved_verification_receipt.json"]
    checks = []
    for period in ["evaluation", "earlier_diagnostic"]:
        frame, start = (data, latest["evaluation_start"]) if period == "evaluation" else (data[data.date.le(latest["earlier_terminal"])], latest["earlier_start"])
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        indices = np.arange(first-1, len(frame)-1)
        for cost in latest["costs"]:
            path = P128 / period / cost / "ENTRY_VINTAGE_EXIT_decisions.parquet"
            ledger_path = P128 / period / cost / "ENTRY_VINTAGE_EXIT_ledger.parquet"
            require(digest(path) == old_bound[str(path.relative_to(ROOT))], "原预测决策不再等于131绑定的来源")
            require(digest(ledger_path) == old_bound[str(ledger_path.relative_to(ROOT))], "原参考当前持仓不再等于131绑定来源")
            decisions = pd.read_parquet(path)
            ledger = pd.read_parquet(ledger_path, columns=["date", "shares", "mark_clock"])
            require(np.array_equal(decisions.origin_index, indices), "原预测判断索引不同")
            require(pd.DatetimeIndex(decisions.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])) and
                pd.DatetimeIndex(decisions.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), "原预测不是当前收盘到下一开盘")
            current_shares = decisions.origin.map(ledger.set_index("date").shares)
            require(current_shares.isna().sum() == 1 and pd.isna(current_shares.iloc[0]), "原参考准备收盘以外缺少持仓行")
            current_shares.iloc[0] = 0
            held = current_shares.gt(0)
            require(decisions.loc[~held, "learning_status"].isna().all() and decisions.loc[~held, "continuation_prediction"].isna().all(), "空仓行混入持仓预测")
            require(decisions.loc[held, "learning_status"].isin(["PREDICTION_AVAILABLE", "NO_VIEW_NO_MATURE_MODEL"]).all(), "原参考出现未处理的学习状态")
            available = decisions.learning_status.eq("PREDICTION_AVAILABLE")
            unavailable = decisions.learning_status.eq("NO_VIEW_NO_MATURE_MODEL")
            require(np.isfinite(decisions.loc[available, "continuation_prediction"]).all() and decisions.loc[unavailable, "continuation_prediction"].isna().all(), "可用预测或无模型状态的数值不一致")
            selected = decisions.loc[held, "model_selection_index"].to_numpy(float)
            require(np.isfinite(selected).all() and np.equal(selected, np.floor(selected)).all(), "原模型选择索引缺失或不是整数")
            ages = decisions.loc[held, "origin_index"].to_numpy(int)-selected.astype(int)+1
            require((ages >= 1).all(), "原参考预测提前到实际进入之前")
            np.testing.assert_allclose(decisions.loc[held, "log_holding_days"], np.log1p(ages), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(decisions.loc[held, "vol20"], frame.vol20.iloc[decisions.loc[held, "origin_index"].to_numpy(int)], atol=1e-12, rtol=1e-12)
            for row in decisions.loc[held].itertuples():
                record = by_day[pd.Timestamp(row.learning_fit_origin)]
                selection_time = pd.Timestamp(row.model_selection_origin)+pd.Timedelta(hours=15, minutes=5)
                decision_time = pd.Timestamp(row.origin)+pd.Timedelta(hours=15, minutes=5)
                require(pd.Timestamp(record["fit_time"]) <= selection_time <= decision_time, "保存预测的模型时点晚于当时选择或判断")
                require(record["latest_exit_index"] <= record["fit_index"] <= row.model_selection_index <= row.origin_index, "保存预测使用未来模型或未结束训练周期")
                require(row.fixed_prediction_identity == identities[pd.Timestamp(row.learning_fit_origin)], "保存预测模型身份与固定版本不同")
                require((record["status"] == "FIT_COMPLETE") == (row.learning_status == "PREDICTION_AVAILABLE"), "保存预测状态与训练支持不同")
            for model, parent_folder in [("VINTAGE_REFERENCE_RISK", "510300_vintage_reference_risk_v1"), ("TREND_NOISE_REFERENCE_BLEND", "510300_trend_noise_reference_blend_v1")]:
                parent_path = ROOT / "reports/research" / parent_folder / period / cost / f"{model}_decisions.parquet"
                parent = pd.read_parquet(parent_path, columns=["origin", "origin_index", "execution_date", "reference_weight"])
                require(np.array_equal(parent.origin_index, indices) and pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(decisions.origin)), "下一项父目标日历不同")
                require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(decisions.execution_date)), "下一项父目标执行时钟不同")
                require(parent.reference_weight.between(0, 1).all(), "下一项父目标缺失或越界")
                sources.append(parent_path)
            checks.append({"period": period, "cost": cost, "decision_origins": len(decisions), "reference_holding_origins": int(held.sum()),
                "reference_flat_origins": int((~held).sum()), "available_prediction_origins": int(available.sum()), "no_mature_model_origins": int(unavailable.sum()),
                "positive_prediction_origins": int(decisions.continuation_prediction.gt(0).sum()), "nonpositive_prediction_origins": int(decisions.continuation_prediction.le(0).sum()),
                "maximum_observed_holding_days": int(ages.max()), "model_clocks_and_fixed_identities_checked": True})
            sources.extend([path, ledger_path])
    receipt = {"checked_at": now(), "status": "SAVED_CONTINUATION_PREDICTIONS_CLOCKS_STATES_AND_PARENT_TARGETS_READY",
        "sources": [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(sources))],
        "checks": checks, "new_predictions_or_budgets_or_accounts": 0, "candidate_registered": False,
        "historical_clock_note": "原保存时点是历史模拟可用性规则，不是2017年实际运行的独立证明。", "source_sha256": digest(Path(__file__))}
    write_json(OUT / "result.json", receipt, exclusive=True)
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    require(index["latest_completed_round"]["round"] == 143, "下一项预测幅度研究的前序轮次不同")
    index["next_work"].update(status="CONTINUATION_STRENGTH_INPUT_CLOCKS_READY", input_preflight=str((OUT / "result.json").relative_to(ROOT)), registered=False)
    index["updated_at"] = now()
    write_json(index_path, index)
    print(json.dumps({"已确认输入": checks, "新预测预算账户": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
