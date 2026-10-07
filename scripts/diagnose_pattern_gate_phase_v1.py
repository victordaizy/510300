"""利用保存案例分解启停门槛与形态阶段，不生成新的策略账户。"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_sequential_patterns_2021_2026_v2"
OUT = ROOT / "reports/research/510300_pattern_gate_phase_diagnostic_v1"
sys.path.insert(0, str(PARENT / "code"))
from sequential_patterns_regime_v1 import digest, now, save_csv, save_json


def main():
    if (OUT / "summary.json").exists():
        raise RuntimeError("归因结果已保存，不能重复覆盖。")
    archive = ROOT / "deliverables/510300_连续形态启停_V2_2021至2026_GPT审阅_20260924.zip"
    assert digest(archive) == "7e26cd2e3258082bec9ad5a2dc3fdef538f66aaa25eeedd3c052194bac4675df"
    sources = {
        "inputs/features.parquet": PARENT / "inputs/features.parquet",
        "inputs/signals.parquet": PARENT / "inputs/signals.parquet",
        "inputs/labels.parquet": PARENT / "inputs/labels.parquet",
        "inputs/training_events.parquet": PARENT / "inputs/training_events.parquet",
        "inputs/decisions.parquet": PARENT / "relaxed/daily_decisions.parquet",
        "inputs/pattern_trades.csv": PARENT / "baseline/accounts/STRESS/PATTERN_ONLY/trades.csv",
        "inputs/parent_summary.json": PARENT / "summary.json",
        "inputs/parent_protocol.md": PARENT / "protocol.md",
        "code/diagnose_pattern_gate_phase_v1.py": Path(__file__),
    }
    for relative, path in sources.items():
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    protocol = {"frozen_at": now(), "purpose": "分解固定状态映射、稀少样本、近期均值与置信下界、同状态证据的具体阻断；仅归因，不改规则。",
                "sample": "2021-01-04至2026-09-16的全部42个确认形态",
                "gates": ["固定状态", "至少3成熟案例且对照完整", "90%收益下界>0", "平均超额>0", "至少2同状态案例", "同状态净均值>0", "同状态超额>0"],
                "phase_features": ["准备日状态到确认日状态", "准备到确认的趋势Z变化", "确认前3日趋势Z变化", "确认日成交量比"],
                "profit_attribution": "复用实际仅形态账户已实现人民币利润，不构造删除案例后的新账户。",
                "influence_diagnostic": "对最后资料日近期池计算去除最大已知盈利案例后的均值与下界，仅检查单个历史案例影响，不交易或挑选策略。",
                "future_window": "未来20日首次启用仅用于事后迟滞说明，不进入原预测、资格或交易。",
                "new_accounts": 0, "new_fits": 0, "market_downloads": 0,
                "parent_zip_sha256": digest(archive)}
    save_json(OUT / "protocol.json", protocol)
    save_json(OUT / "freeze.json", {"frozen_at": now(), "hashes": {name: digest(OUT / name) for name in sources} | {"protocol.json": digest(OUT / "protocol.json")}})
    d = pd.read_parquet(OUT / "inputs/features.parquet")
    sig = pd.read_parquet(OUT / "inputs/signals.parquet")
    sig = sig[sig.signal_date >= "2021-01-04"]
    dec = pd.read_parquet(OUT / "inputs/decisions.parquet")
    train = pd.read_parquet(OUT / "inputs/training_events.parquet").set_index("signal_id")
    labels = pd.read_parquet(OUT / "inputs/labels.parquet")
    labels = labels[(labels.cost == "STRESS") & (labels.label_status == "MATURE")].set_index("signal_id")
    trades = pd.read_csv(OUT / "inputs/pattern_trades.csv").set_index("signal_id")
    merged = sig.merge(dec, left_on=["signal_idx", "family"], right_on=["idx", "family"], suffixes=("", "_decision"))
    rows = []
    for row in merged.to_dict("records"):
        i, setup = int(row["signal_idx"]), int(row["setup_idx"])
        ids = row["train_signal_ids"].split("|") if row["train_signal_ids"] else []
        prior = train.loc[ids] if ids else train.iloc[:0]
        enough = len(prior) >= 3 and prior.excess_return.notna().all()
        conditions = {
            "g1_fixed_state": bool(row["STATE_ONLY"]), "g2_min_count": bool(enough),
            "g3_positive_lower": bool(row["normal90_lower"] > 0), "g4_positive_excess": bool(row["mean_excess"] > 0),
            "g5_state_count": bool(row["state_n"] >= 2), "g6_state_mean": bool(row["state_mean_net"] > 0),
            "g7_state_excess": bool(row["state_mean_excess"] > 0)}
        assert all(conditions.values()) == bool(row["FULL"])
        available = dec[(dec.family == row["family"]) & (dec.idx <= i)]
        enabled = available[available.FULL]
        after = dec[(dec.family == row["family"]) & (dec.idx > i) & (dec.idx <= i + 20) & dec.FULL]
        label = labels.loc[row["signal_id"]] if row["signal_id"] in labels.index else None
        trade = trades.loc[row["signal_id"]] if row["signal_id"] in trades.index else None
        rows.append({"signal_id": row["signal_id"], "family": row["family"], "signal_idx": i,
                     "setup_date": row["setup_date"], "signal_date": row["signal_date"],
                     "setup_state": d.state.iloc[setup], "confirm_state": row["state"],
                     "phase": d.state.iloc[setup] + "→" + row["state"],
                     "setup_to_confirm_z_change": d.trend_z.iloc[i] - d.trend_z.iloc[setup],
                     "prior_3d_z_change": d.trend_z.iloc[i] - d.trend_z.iloc[i - 3],
                     "volume_ratio": row["volume_ratio"], "n_train": row["n_train"],
                     "mean_net": row["mean_net"], "normal90_lower": row["normal90_lower"],
                     "mean_excess": row["mean_excess"], "state_n": row["state_n"],
                     "failed_gate_names": "|".join(k for k, v in conditions.items() if not v),
                     "failed_gate_count": sum(not v for v in conditions.values()),
                     "last_full_enabled_age": i - int(enabled.idx.iloc[-1]) if len(enabled) else None,
                     "hindsight_first_enabled_next20_delay": int(after.idx.iloc[0]) - i if len(after) else None,
                     "event_net_return": label.net_return if label is not None else None,
                     "actual_account_net_pnl": trade.net_pnl if trade is not None else None,
                     "account_execution_status": "EXECUTED" if trade is not None else "NOT_EXECUTED",
                     **conditions})
    frame = pd.DataFrame(rows)
    save_csv(OUT / "all_gates_and_phases.csv", frame)
    gate_summary = []
    for g in conditions:
        fail = ~frame[g]
        gate_summary.append({"gate": g, "failed_signals": int(fail.sum()),
                             "sole_failed_gate_signals": int((fail & frame.failed_gate_count.eq(1)).sum()),
                             "passed_signals": int((~fail).sum()),
                             "actual_pnl_in_failed_signals": frame.loc[fail, "actual_account_net_pnl"].sum(),
                             "warning": "门之间重叠；利润是已有账户周期归因，不是移除该门后的账户收益。"})
    save_csv(OUT / "gate_overlaps.csv", gate_summary)
    phase = frame.groupby(["family", "phase"]).agg(signals=("signal_id", "size"),
             mean_event_net=("event_net_return", "mean"), median_event_net=("event_net_return", "median"),
             actual_pnl=("actual_account_net_pnl", "sum"), volume_ratio_median=("volume_ratio", "median")).reset_index()
    save_csv(OUT / "phase_description.csv", phase)
    influence = []
    for row in dec[dec.date == dec.date.max()].to_dict("records"):
        ids = row["train_signal_ids"].split("|") if row["train_signal_ids"] else []
        pool = train.loc[ids] if ids else train.iloc[:0]
        if len(pool) < 3:
            continue
        without = pool.drop(pool.net_return.idxmax())
        lower = without.net_return.mean() - 1.2815515655446004 * without.net_return.std(ddof=1) / np.sqrt(len(without))
        influence.append({"family": row["family"], "historical_date": row["date"], "n": len(pool),
                          "largest_prior_event": pool.net_return.idxmax(), "original_mean": row["mean_net"],
                          "original_lower90": row["normal90_lower"], "without_largest_mean": without.net_return.mean(),
                          "without_largest_lower90": lower, "purpose": "历史案例影响诊断，非改标签或新策略。"})
    save_csv(OUT / "largest_mature_event_influence.csv", influence)
    summary = {"completed_at": now(), "previous_goal_turn_classification": "PROGRESS_V2_COMPLETED_AND_DELIVERED",
               "current_goal_turn_classification": "PROGRESS_CAUSAL_GATE_AND_PHASE_ATTRIBUTION",
               "confirmed_signals": len(frame), "gate_failures": gate_summary,
               "phase_groups": phase.to_dict("records"), "last_date_influence": influence,
               "all_gates_reproduced": True, "new_accounts": 0, "new_fits": 0,
               "new_market_downloads": 0, "strict_forward_observations": 0, "goal_achieved": False}
    save_json(OUT / "summary.json", summary)
    print(pd.DataFrame(gate_summary).to_string(index=False))
    print(phase.to_string(index=False))
    print(pd.DataFrame(influence).to_string(index=False))


if __name__ == "__main__":
    main()
