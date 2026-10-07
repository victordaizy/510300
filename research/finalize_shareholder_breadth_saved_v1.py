"""从已保存的单次初筛收尾；不重新筛选信号或运行账户。"""
import math

import pandas as pd

from research.shareholder_disclosure_breadth_v1 import OUT, PERIODS, build_daily, collection_receipt
from research.mechanism_odds_open_contract_v1 import now, read, save


def main():
    assert not (OUT / "result.json").exists()
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market = market.loc[market.date.le("2025-12-31")].reset_index(drop=True)
    daily = pd.read_parquet(OUT / "daily_features.parquet")
    labels = pd.read_parquet(OUT / "origins.parquet")
    events = pd.read_parquet(OUT / "inputs/events.parquet")
    uncertain = pd.read_parquet(OUT / "inputs/uncertain_events.parquet")
    members = pd.read_parquet(OUT / "inputs/membership.parquet")
    cutoff = pd.Timestamp("2021-12-31")
    past = build_daily(market.loc[market.date.le(cutoff), "date"], members,
        events.loc[events.notice_date.le(cutoff)], uncertain=uncertain.loc[uncertain.notice_date.le(cutoff)])
    original = daily.loc[daily.date.le(cutoff)].reset_index(drop=True)
    dtype_changes = {c: [str(original[c].dtype), str(past[c].dtype)] for c in original if original[c].dtype != past[c].dtype}
    # 完整样本在2023年出现缺失使计数列升为浮点；数值、日期、信号逐项仍必须一致。
    pd.testing.assert_frame_equal(original, past.reset_index(drop=True), check_dtype=False)
    errors = []
    for row in labels.loc[labels.status.eq("MATURE")].itertuples():
        a, b = int(row.entry_idx), int(row.exit_idx)
        value = (float(market.open.iloc[b]) + math.fsum(market.dividend.iloc[a + 1:b + 1])) / float(market.open.iloc[a]) - 1
        errors.append(abs(value - row.gross_return))
    assert max(errors, default=0) < 1e-12
    primary = {r["period"]: r for r in read(OUT / "summary.json") if r["group"] == "PRIMARY_NONOVERLAP"}
    full = primary["FULL"]
    gates = {"complete_provider_queries": collection_receipt()["status"] == "COMPLETE",
        "enough_nonoverlap": full["evaluated"] >= 20 and all(primary[p]["evaluated"] >= 5 for p in ["EARLY", "LATE"]),
        "mean_above_cost_proxy": full["mean"] > .0028, "increment_above_matched_month": full["increment"] > 0,
        "both_subperiods_positive": all(primary[p]["mean"] > 0 for p in ["EARLY", "LATE"])}
    receipt = {"at": now(), "status": "PASS_SAVED_NUMERIC_PREFIX_AND_LABELS", "future_prefix_checks": 1,
        "saved_labels_recomputed": len(errors), "max_label_error": max(errors, default=0),
        "original_run_terminal": "VERIFIER_DTYPE_MISMATCH_AFTER_SAVED_SCREEN",
        "dtype_only_difference": dtype_changes, "numeric_or_signal_differences": 0,
        "frozen_code_changed": False, "new_return_tests": 0, "new_accounts": 0,
        "note": "修正核对方法的整型/浮点表示差异；未修改信号、参数、来源或已保存收益。"}
    save(OUT / "verification_receipt.json", receipt)
    result = {"at": now(), "study_id": "510300_SHAREHOLDER_DISCLOSURE_BREADTH_V1",
        "status": "PASS_GROSS_SCREEN_ACCOUNT_REQUIRED" if all(gates.values()) else "REJECTED_FIXED_GROSS_SCREEN_NO_PARAMETER_RESCUE",
        "gates": gates, "weekly_origins": len(labels), "status_counts": labels.status.value_counts().to_dict(),
        "primary": primary, "concentration": read(OUT / "concentration.json"), "uncertainty": read(OUT / "uncertainty.json"),
        "account_status": "NOT_RUN_USER_REORIENTED_TO_STATE_MECHANISM",
        "disposition": "保留毛优势线索；按用户最新要求先解释在哪类阶段和为何有效，不按全区间均值自动扩展账户。",
        "net_sharpe": None, "net_cagr": None, "max_drawdown": None,
        "new_accounts": 0, "new_fixed_proxy_questions": 1, "strict_M03_run": False,
        "goal_achieved": False, "historical_first_vintage_verified": False, "orders_authorized": False}
    save(OUT / "result.json", result)
    print(f"旧初筛已收尾：32次无重叠机会，毛均值{full['mean']:.4%}；按新指令保留线索，先研究阶段与机制。", flush=True)


if __name__ == "__main__":
    main()
