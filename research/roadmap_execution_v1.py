"""执行路线图登记及保存结果的有限统计，保持原研究不可变。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_roadmap_execution_v1.json"
PRIOR_EVIDENCE = {
    "R02": ["510300_multidim_financing_composition_v1/result.json", "510300_multidim_financing_composition_v1/data_correction/event_state_control/result.json", "510300_point_f03_information_intake_v1/summary.json", "510300_daily_score_financing_input_correction_v1/result.json"],
    "R03": ["510300_factor96_rapid_structure_v1/protocol.json", "510300_factor96_rapid_structure_v1/result.json", "510300_cross_etf_forced_flow_binary_screen_v1.json", "510300_fund_share_publication_receipts_closure_v1/result.json"],
    "R04": ["510300_original_earnings_breadth_v1/result.json", "510300_eps_growth_disagreement_increment_v1/result.json", "510300_unlock_announcement_increment_v1/result.json", "510300_corporate_repurchase_fact_ledger_v1/result.json", "510300_corporate_repurchase_chain_diagnostic_v1/result.json"],
    "R05": ["510300_if_true_term_structure_binary_screen_v1_0_1.json", "510300_if_open_interest_increment_v1/result.json", "510300_overnight_global_information_v1/result.json"],
    "R06": ["510300_factor96_bottleneck_diagnostic_v1/result.json", "510300_probability_payoff_exit_v1/result.json", "510300_daily_native_baseline_v1/summary.json"],
    "R07": ["510300_m1_m2_monthly_increment_v1/result.json"],
    "R08": ["510300_pressure_recovery_v1/source_admission_20261002/summary.json"],
}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def bootstrap_indices(n: int, draws: int, length: int, seed: int) -> np.ndarray:
    require(n > 1 and draws > 0 and 0 < length <= n, "区块抽样参数无效")
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, n, size=(draws, math.ceil(n / length)))
    return ((starts[:, :, None] + np.arange(length)) % n).reshape(draws, -1)[:, :n].astype(np.int32)


def net_payoff(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=float)
    require(len(x) > 0 and np.isfinite(x).all(), "周期净收益为空或有非有限值")
    positive, negative, flat = x > 0, x < 0, x == 0
    p, q = float(positive.mean()), float(negative.mean())
    gain = float(x[positive].mean()) if positive.any() else 0.0
    loss = float(-x[negative].mean()) if negative.any() else 0.0
    lower_count = max(1, math.ceil(0.05 * len(x)))
    return {"cycles": len(x), "profit_probability": p, "loss_probability": q,
            "flat_probability": float(flat.mean()), "mean_gain_net": gain, "mean_loss_net": loss,
            "pG_minus_qL_net": p * gain - q * loss, "equal_weight_cycle_mean_net": float(x.mean()),
            "minimum_cycle_net": float(x.min()), "worst_5pct_cycle_mean_net": float(np.sort(x)[:lower_count].mean()),
            "cost_deducted_again": False}


def admission_state(material_novelty: bool, source_qualified: bool, economic_gate: bool) -> str:
    if not material_novelty:
        return "NOT_ADMITTED_OLD_USE_OR_NO_MATERIAL_NEW_INFORMATION"
    if not source_qualified:
        return "NOT_ADMITTED_SOURCE_SCOPE_OR_CLOCK"
    if not economic_gate:
        return "NOT_ADMITTED_ECONOMIC_GATE"
    return "RESEARCH_CANDIDATE_ONLY"


def available_asof(records: pd.DataFrame, decision_at: pd.Timestamp) -> pd.DataFrame:
    required = ["economic_at", "received_at"]
    require(all(k in records for k in required), "缺少经济时间或实际收到时间")
    frame = records.copy()
    for column in required:
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    cutoff = pd.Timestamp(decision_at)
    require(cutoff.tzinfo is not None, "决策时刻必须包含时区")
    return frame.loc[frame.economic_at.le(cutoff) & frame.received_at.le(cutoff)].copy()


def prepare(cfg: dict, output: Path) -> None:
    require(not (output / "freeze.json").exists(), "已冻结，不重复准备")
    output.mkdir(parents=True, exist_ok=True)
    first = ROOT / cfg["first_batch"]
    parent = ROOT / cfg["parent"]
    write_json(output / "authority.json", {"recorded_at": now(), "user_text": cfg["user_authority"],
        "scope": "全部48任务按其依赖推进；不将研究授权改写为实盘权限", "old_first_batch_scope_superseded_for_future_work": True,
        "no_paid_data": True, "actual_orders_authorized": False})
    write_json(output / "objective_contract.json", {"assets": cfg["assets"], "research_capitals_cny": cfg["research_capitals_cny"],
        "net_sharpe_goal": cfg["goal_net_sharpe"], "cagr_goal": cfg["goal_cagr"], "primary_annual_days": cfg["primary_annual_days"],
        "real_capital": None, "real_drawdown_limit": None, "real_event_loss_limit": None, "cash_use_horizon": None,
        "actual_broker_fees": None, "user_parameter_question_pending": True, "trading_authority": False})
    (output / "metric_contract.json").write_bytes((first / "metric_contract.json").read_bytes())

    workbook = next((first / "received").glob("*.xlsx"))
    book = load_workbook(workbook, read_only=True, data_only=True)
    rows = list(book["48项任务"].values)
    header_index = next(i for i, row in enumerate(rows) if row and row[0] == "任务ID")
    headers = list(rows[header_index])
    tasks = [dict(zip(headers, row)) for row in rows[header_index + 1:] if row and row[0]]
    require(len(tasks) == 48, "任务清单不是48项")
    book.close()
    write_json(output / "original_48_tasks.json", tasks)

    factor_root = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
    factors = json.loads((factor_root / "factor_progress.json").read_text(encoding="utf-8"))
    require(len(factors) == 96 and len({row["id"] for row in factors}) == 96, "96因子映射不完整")
    routes = {"A": "价格基准/旧趋势", "B": "价格修复/旧过程", "C": "价格与活动代理", "D": "日内隔夜旧用途",
              "E": "指数内部观测", "F": "R02", "G": "R03", "H": "R05", "I": "只观察期权储备", "J": "R05",
              "K": "R07", "L": "R04", "M": "R04", "N": "R04/R03", "O": "R04", "P": "R06"}
    mappings = []
    for row in factors:
        mappings.append({"factor_id": row["id"], "name": row["name"], "family": row["family"],
            "roadmap_route": routes[row["id"][0]], "mechanism": row["mechanism"], "definition": row["definition"],
            "extra_observation_required": row["data_path_hint"], "counterexample": row["counterexample"],
            "known_prior_overlap": row["source_overlap_note"], "prior_evidence_hint": row["repository_hint"],
            "source_clock_requirement": row["source_clock_rule"], "prior_card_status": row["current_status"],
            "status_scope": "原卡进度不是全项目最新终态，须结合本批机制裁决及试验索引",
            "attachment_grade_only": row["source_grade"], "new_alpha_admitted": False,
            "source_ids": row["source_ids"], "source_urls": row["source_urls"]})
    pd.DataFrame(mappings).to_csv(output / "mechanism_registry.csv", index=False, encoding="utf-8-sig")

    studies = []
    research_root = ROOT / "reports/research"
    candidate_paths = set(research_root.glob("*/result.json")) | set(research_root.glob("*_v*.json"))
    candidate_paths |= {research_root / item for values in PRIOR_EVIDENCE.values() for item in values}
    evidence_rows = []
    for path in sorted(candidate_paths):
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            continue
        status = value.get("status", value.get("classification", "STATUS_NOT_DECLARED"))
        if not any(k in value for k in ["study_id", "study", "status", "classification"]):
            continue
        studies.append({"source_path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "study_id": value.get("study_id", value.get("study", path.parent.name)), "source_status": status,
            "source_classification": value.get("classification", value.get("evidence_class", "NOT_DECLARED")),
            "goal_achieved_declared": value.get("goal_achieved"), "independent_validation_declared": value.get("independent_validation"),
            "new_validation_this_batch": False, "historical_samples_unseen": False})
    (output / "study_registry.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in studies), encoding="utf-8")
    write_json(output / "selection_report.json", {"snapshot_at": now(), "source_summary_records": len(studies),
        "scope": "研究根目录的直接result及根层版本JSON，含指定直接依赖；不是完整Git历史，也不等于独立候选数",
        "factor_cards": 96, "families": len(set(row["family"] for row in factors)), "strategy_cards": 18,
        "selection_count_fully_identified": False, "all_history_already_exposed": True,
        "local_bootstrap_family_size": 5, "local_adjustment_does_not_correct_cross_round_selection": True,
        "99_paper_source_list_status": "NOT_LOCATED_AS_A_SINGLE_VERIFIED_99_ITEM_INDEX; 原96卡来源链接保留，不声称逐篇复核99文献"})
    for route, paths in PRIOR_EVIDENCE.items():
        for relative in paths:
            path = research_root / relative
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                evidence_rows.append({"route": route, "path": str(path.relative_to(ROOT)), "sha256": digest(path),
                    "status": value.get("status", value.get("classification", "STATUS_NOT_DECLARED"))})
    pd.DataFrame(evidence_rows).to_csv(output / "mechanism_prior_evidence.csv", index=False, encoding="utf-8-sig")

    # 原分钟24项用途原样保留；另登记本批新检查的现有分支输入，不读取整套盘口。
    original_catalog = pd.read_csv(parent / "01_实际数据清单.csv")
    original_catalog.to_csv(output / "data_catalog_original_24.csv", index=False, encoding="utf-8-sig")
    correction = json.loads((ROOT / "config/510300_financing_source_correction_20240808_v1.json").read_text(encoding="utf-8"))
    extra_inputs = [
        (correction["corrected_margin_path"], "R02", "融资余额/买入；偿还为会计推算，首版UNKNOWN"),
        (correction["corrected_daily_path"], "R02", "已有价格/融资条件；仅沿用2024-08-08纠正副本"),
        ("data/raw/flow/510300_cross_etf_forced_flow_sse_weekly_v1.parquet", "R03", "沪市固定产品周度份额；非完整同指数日频NAV体系"),
        ("reports/research/510300_original_earnings_breadth_v1/selected_verified_facts.parquet", "R04", "已核验累计财报事实；旧盈利广度表达目标失败"),
        ("reports/research/510300_original_earnings_breadth_v1/daily_member_facts.parquet", "R04", "旧成员映射，非新公司回购指数暴露证明"),
    ]
    assets = []
    for relative, route, scope in extra_inputs:
        path = ROOT / relative
        item = {"path": relative, "route": route, "scope": scope, "exists": path.exists(), "admission": "CONDITIONAL_REUSE_ONLY"}
        if path.exists():
            meta = pq.ParquetFile(path)
            item.update(bytes=path.stat().st_size, sha256=digest(path), rows=meta.metadata.num_rows,
                        fields="|".join(meta.schema_arrow.names), first_vintage="NOT_ESTABLISHED")
        assets.append(item)
    pd.DataFrame(assets).to_csv(output / "data_catalog_additional.csv", index=False, encoding="utf-8-sig")

    frozen_paths = [CONFIG, Path(__file__), ROOT / "docs/510300_ROADMAP_EXECUTION_V1.md",
        ROOT / "tests/test_roadmap_execution_v1.py", ROOT / "research/roadmap_public_vintages_v1.py",
        ROOT / "scripts/run_510300_roadmap_execution_v1.py", ROOT / "scripts/collect_510300_roadmap_public_vintages_v1.py",
        first / "native_accounts.parquet", first / "native_cycles.parquet", first / "native_orders.parquet",
        first / "native_forecasts.parquet", first / "summary.json", first / "freeze.json",
        parent / "08_完整账户逐日账本.parquet", parent / "11_完成持有周期.csv", parent / "09_全部模拟请求.csv",
        parent / "06_滚动预测.parquet", factor_root / "factor_progress.json", factor_root / "strategy_progress.json",
        output / "mechanism_registry.csv", output / "mechanism_prior_evidence.csv", output / "original_48_tasks.json",
        ROOT / "data/reference/sse_trade_calendar_2026.csv", ROOT / "data/reference/sse_trade_calendar_2026.metadata.json"]
    frozen_paths += [research_root / row for values in PRIOR_EVIDENCE.values() for row in values if (research_root / row).exists()]
    unique = sorted(set(frozen_paths))
    write_json(output / "freeze.json", {"frozen_at": now(), "new_bootstrap_draws_at_freeze": 0,
        "prior_baseline_results_already_seen": True, "new_fits": 0, "new_accounts": 0,
        "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": digest(p)} for p in unique]})
    print(json.dumps({"状态": "路线图登记及有限诊断已冻结", "任务": 48, "因子": 96, "研究摘要": len(studies), "冻结文件": len(unique)}, ensure_ascii=False))


def interval_row(values: np.ndarray, indices: np.ndarray, multiplier: float, comparison: str, **labels: object) -> dict:
    samples = values[indices].mean(axis=1) * multiplier
    nominal = np.quantile(samples, [0.025, 0.975])
    adjusted = np.quantile(samples, [0.005, 0.995])
    return {**labels, "comparison": comparison, "dates": len(values), "point": float(values.mean() * multiplier),
            "nominal_95_lower": float(nominal[0]), "nominal_95_upper": float(nominal[1]),
            "local_five_comparison_lower": float(adjusted[0]), "local_five_comparison_upper": float(adjusted[1]),
            "post_result_diagnostic": True, "independent_confirmation": False}


def run_saved_diagnostic(cfg: dict, output: Path) -> None:
    freeze = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    for item in freeze["files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "冻结文件变化：" + item["path"])
    claim = output / "run_claim.json"
    with claim.open("x", encoding="utf-8") as stream:
        json.dump({"started_at": now(), "single_saved_diagnostic": True}, stream, ensure_ascii=False)
    first, parent = ROOT / cfg["first_batch"], ROOT / cfg["parent"]
    native = pd.read_parquet(first / "native_accounts.parquet")
    common = pd.read_parquet(parent / "08_完整账户逐日账本.parquet")
    common = common.loc[common.model.eq("D")]
    native_cycles = pd.read_parquet(first / "native_cycles.parquet")
    common_cycles = pd.read_csv(parent / "11_完成持有周期.csv", parse_dates=["entry_date", "exit_date"])
    common_cycles = common_cycles.loc[common_cycles.model.eq("D")]
    orders = pd.read_parquet(first / "native_orders.parquet")
    settings = cfg["bootstrap"]
    account_index = bootstrap_indices(471, settings["draws"], settings["block_length"], settings["seed"])
    prediction_index = bootstrap_indices(364, settings["draws"], settings["block_length"], settings["seed"] + 1)
    np.savez_compressed(output / "bootstrap_indices.npz", account=account_index, prediction=prediction_index)
    results, payoffs, monthly, requests, contributions, sensitivities = [], [], [], [], [], []
    paired_rows = []
    for (capital, cost), part in native.groupby(["capital", "cost"], sort=True):
        original = common.loc[common.capital.eq(capital) & common.cost.eq(cost)]
        pair = part[["date", "net_return", "equity"]].merge(original[["date", "net_return", "equity"]], on="date", suffixes=("_native", "_common"), validate="one_to_one").sort_values("date")
        require(len(pair) == 471, "账户日历未完整对齐")
        difference = (pair.net_return_native - pair.net_return_common).to_numpy()
        require(np.isfinite(difference).all(), "配对收益非有限")
        results.append(interval_row(difference, account_index, cfg["primary_annual_days"], "ANNUALIZED_ARITHMETIC_RETURN_DIFFERENCE_NATIVE_MINUS_COMMON", capital=int(capital), cost=cost))
        pair["capital"], pair["cost"] = int(capital), cost
        pair["daily_return_difference"] = difference
        equity_difference = pair.equity_native - pair.equity_common
        pair["daily_cny_difference"] = equity_difference.diff().fillna(equity_difference.iloc[0])
        paired_rows.append(pair)
        for month, values in pair.groupby(pair.date.dt.to_period("M")):
            monthly.append({"capital": capital, "cost": cost, "month": str(month), "delta_cny": float(values.daily_cny_difference.sum()), "delta_arithmetic_return": float(values.daily_return_difference.sum())})
        cycles = native_cycles.loc[native_cycles.capital.eq(capital) & native_cycles.cost.eq(cost)]
        payoff = net_payoff(cycles.net_return.to_numpy())
        payoff.update(capital=int(capital), cost=cost, total_net_pnl=float(cycles.net_pnl.sum()),
                      total_commission=float(cycles.commission.sum()), total_slippage=float(cycles.slippage.sum()),
                      mean_holding_days=float(cycles.holding_trading_days.mean()), maximum_holding_days=int(cycles.holding_trading_days.max()))
        payoffs.append(payoff)
        order = orders.loc[orders.capital.eq(capital) & orders.cost.eq(cost)]
        requests.append({"capital": capital, "cost": cost, "requests": len(order), "unfilled": int(order.filled_quantity.eq(0).sum()),
                         "partial": int(order.status.str.contains("PARTIALLY").sum()), "actual_fills_verified": 0})
        prior_cycles = common_cycles.loc[common_cycles.capital.eq(capital) & common_cycles.cost.eq(cost)]
        new_entries, old_entries = set(pd.to_datetime(cycles.entry_date)), set(prior_cycles.entry_date)
        for date in sorted(new_entries | old_entries):
            contributions.append({"capital": capital, "cost": cost, "entry_date": date,
                "membership": "BOTH_ENTRY_DATES" if date in new_entries & old_entries else "NATIVE_ONLY" if date in new_entries else "COMMON_ONLY",
                "same_entry_date_does_not_guarantee_same_quantity_or_path": True})
        intervals = sorted([(pd.Timestamp(row.entry_date), pd.Timestamp(row.exit_date)) for frame in (cycles, prior_cycles) for row in frame.itertuples()])
        clusters: list[list[pd.Timestamp]] = []
        for start, end in intervals:
            if clusters and start <= clusters[-1][1]:
                clusters[-1][1] = max(clusters[-1][1], end)
            else:
                clusters.append([start, end])
        total = float(difference.sum())
        for cluster_id, (start, end) in enumerate(clusters, 1):
            selected = pair.loc[pair.date.between(start, end)]
            contribution = float(selected.daily_return_difference.sum())
            sensitivities.append({"capital": capital, "cost": cost, "cluster_id": cluster_id, "entry": start, "exit": end,
                "cluster_delta_cny": float(selected.daily_cny_difference.sum()), "cluster_delta_return": contribution,
                "zero_one_cluster_fixed_calendar_annual_arithmetic_delta": (total - contribution) / 471 * cfg["primary_annual_days"],
                "classification": "SAVED_CONTRIBUTION_BOUND_NOT_ACCOUNT_SIMULATION"})
    predictions = pd.read_parquet(parent / "06_滚动预测.parquet")
    prediction_native = pd.read_parquet(first / "native_forecasts.parquet")
    paired = predictions.loc[predictions.prediction_D.notna() & predictions.return_h2.notna(), ["date", "prediction_D", "return_h2"]].merge(prediction_native[["date", "prediction_D"]], on="date", suffixes=("_common", "_native"), validate="one_to_one").sort_values("date")
    mse_difference = ((paired.prediction_D_common - paired.return_h2) ** 2 - (paired.prediction_D_native - paired.return_h2) ** 2).to_numpy()
    results.append(interval_row(mse_difference, prediction_index, 1.0, "MSE_COMMON_MINUS_NATIVE", capital=None, cost="NOT_APPLICABLE"))
    pd.DataFrame(results).to_csv(output / "uncertainty_intervals.csv", index=False, encoding="utf-8-sig")
    write_json(output / "uncertainty_intervals.json", results)
    pd.concat(paired_rows, ignore_index=True).to_parquet(output / "paired_account_returns.parquet", index=False)
    paired.to_parquet(output / "paired_predictions.parquet", index=False)
    pd.DataFrame({"account_date": paired_rows[0].date.to_numpy()}).to_csv(output / "bootstrap_account_dates.csv", index=False)
    pd.DataFrame({"prediction_date": paired.date}).to_csv(output / "bootstrap_prediction_dates.csv", index=False)
    for filename, values in [("payoff_distribution.csv", payoffs), ("all_monthly_marginal_pnl.csv", monthly),
                             ("execution_request_counts.csv", requests), ("new_and_removed_entry_dates.csv", contributions),
                             ("all_event_cluster_sensitivities.csv", sensitivities)]:
        pd.DataFrame(values).to_csv(output / filename, index=False, encoding="utf-8-sig")
    for row in payoffs:
        require(abs(row["pG_minus_qL_net"] - row["equal_weight_cycle_mean_net"]) < 1e-12, "净期望恒等式不一致")
    saved_index = np.load(output / "bootstrap_indices.npz")
    require(np.array_equal(saved_index["account"], account_index) and np.array_equal(saved_index["prediction"], prediction_index), "保存抽样索引未一致")
    write_json(output / "saved_statistics_receipt.json", {"completed_at": now(), "status": "PASS_SAVED_DIAGNOSTIC_IDENTITIES",
        "account_scenarios": 4, "account_days_each": 471, "prediction_days": 364,
        "block_length": settings["block_length"], "draws_per_statistic": settings["draws"], "intervals": 5,
        "indices_saved": True, "indices_sha256": digest(output / "bootstrap_indices.npz"),
        "new_fits": 0, "new_accounts": 0, "post_result_diagnostic": True, "independent_confirmation": False,
        "delay_account_replay": "NOT_RUN_NO_NEW_ADMITTED_MECHANISM"})
    print(json.dumps({"状态": "保存结果区块诊断完成", "比较": len(results), "区块索引已保存": True, "新拟合": 0, "新账户": 0}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="48任务路线图登记与有限保存结果诊断")
    parser.add_argument("--stage", choices=["prepare", "run"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["output"]
    if args.stage == "prepare":
        prepare(cfg, output)
    else:
        run_saved_diagnostic(cfg, output)


if __name__ == "__main__":
    main()
