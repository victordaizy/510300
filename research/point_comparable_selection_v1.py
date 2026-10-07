"""已保存可比账户的有限选择稳定性诊断，不拟合或运行新交易策略。"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_comparable_selection_v1"
PERIODS = ("2015_2019", "2020_2026")
COSTS = ("BASE", "STRESS")
BLOCKS = 16
TOLERANCE = 1e-12
CANDIDATES = {
    "A_SAVED_WEIGHT": ("510300_point_weight_information_diagnostic_v1", "SAVED_WEIGHT"),
    "A_POINT_BINARY": ("510300_point_weight_information_diagnostic_v1", "POINT_BINARY"),
    "B_SAVED_WEIGHT": ("510300_point_second_weight_comparison_v1", "B_SAVED_WEIGHT"),
    "A_BINARY_REBALANCE": ("510300_point_binary_rebalance_diagnostic_v1", "BINARY_REBALANCE"),
    "B_POINT_BINARY": ("510300_point_account_nr7_complement_v1", "POINT_B"),
    "NR7_ONLY": ("510300_point_account_nr7_complement_v1", "NR7_ONLY"),
    "A_BINARY_PLUS_NR7": ("510300_point_account_nr7_complement_v1", "POINT_A_PLUS_NR7"),
    "PULLBACK_ONLY": ("510300_point_confirmed_pullback_study_v1", "PULLBACK_ONLY"),
    "A_PLUS_PULLBACK": ("510300_point_confirmed_pullback_study_v1", "A_PLUS_PULLBACK"),
    "DIRECTIONAL_ONLY": ("510300_point_directional_confirmation_study_v1", "DIRECTIONAL_ONLY"),
    "A_PLUS_DIRECTIONAL": ("510300_point_directional_confirmation_study_v1", "A_PLUS_DIRECTIONAL"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def align_returns(frames):
    """完全相同的真实日历才准入；缺日、未知或差异不补齐。"""
    require(bool(frames), "没有真实账户。")
    reference = next(iter(frames.values())).date.reset_index(drop=True)
    require(reference.is_monotonic_increasing and reference.is_unique, "账户日历不是严格递增唯一。")
    columns = {}
    for name, frame in frames.items():
        require(frame.date.reset_index(drop=True).equals(reference), "账户日历不一致：" + name)
        values = frame.net_return.to_numpy(float)
        require(np.isfinite(values).all() and (values > -1).all(), "账户收益未知或越界：" + name)
        columns[name] = values
    return pd.DataFrame({"date": reference, **columns})


def unique_candidates(panels):
    """两成本都完全相同才归并；无活动账户单列，不按盈利好坏删列。"""
    names = list(next(iter(panels.values())).columns.drop("date"))
    require(all(list(p.columns.drop("date")) == names for p in panels.values()), "费用档候选身份不一致。")
    kept, records = [], []
    for name in names:
        inactive = all(np.array_equal(p[name].to_numpy(), np.zeros(len(p))) for p in panels.values())
        duplicate = next((old for old in kept if all(np.array_equal(p[name].to_numpy(), p[old].to_numpy())
                                                   for p in panels.values())), None)
        status = "INACTIVE_RETAINED_RAW_EXCLUDED_RANKING" if inactive else "EXACT_DUPLICATE" if duplicate else "ADMITTED"
        records.append({"candidate": name, "status": status, "same_as": duplicate})
        if status == "ADMITTED":
            kept.append(name)
    require(len(kept) >= 2, "可比非重复候选少于两个。")
    return kept, records


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all(), "排序分数不是有限值。")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and abs(values[order[end]] - values[order[start]]) <= TOLERANCE:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def scores(count, sums, squares):
    require(count >= 2, "分割样本太少。")
    numerator = squares - sums * sums / count
    require((numerator >= -1e-15).all(), "分割方差出现不可解释的负值。")
    std = np.sqrt(np.maximum(numerator, 0.) / (count - 1))
    zero = std <= 1e-15
    require(not np.any(zero & (np.abs(sums) > 1e-13)), "出现非零恒定收益，排序夏普不可定义。")
    result = np.zeros(len(sums))
    np.divide(sums / count * np.sqrt(252), std, out=result, where=~zero)
    return result, zero


def cscv(panel, blocks=BLOCKS):
    """枚举全部对称分割；并列最优平分权重，未成交全零半样本排序分数为0。"""
    values = panel.drop(columns="date").to_numpy(float)
    names = list(panel.columns.drop("date"))
    require(blocks >= 2 and blocks % 2 == 0 and len(panel) >= blocks * 2, "时间块数量或样本不足。")
    require(len(names) >= 2 and np.isfinite(values).all(), "候选或有限收益不足。")
    groups = np.array_split(np.arange(len(panel)), blocks)
    counts = np.array([len(g) for g in groups])
    sums = np.array([values[g].sum(axis=0) for g in groups])
    squares = np.array([(values[g] ** 2).sum(axis=0) for g in groups])
    records, blocks_rows = [], []
    for block, indices in enumerate(groups):
        blocks_rows.append({"block": block, "rows": len(indices), "first_date": panel.date.iloc[indices[0]],
                            "last_date": panel.date.iloc[indices[-1]]})
    for split, selected in enumerate(itertools.combinations(range(blocks), blocks // 2)):
        inside = np.asarray(selected)
        outside = np.asarray([i for i in range(blocks) if i not in selected])
        is_score, is_zero = scores(int(counts[inside].sum()), sums[inside].sum(axis=0), squares[inside].sum(axis=0))
        oos_score, oos_zero = scores(int(counts[outside].sum()), sums[outside].sum(axis=0), squares[outside].sum(axis=0))
        winners = np.flatnonzero(np.abs(is_score - is_score.max()) <= TOLERANCE)
        ranks = average_ranks(oos_score)
        for winner in winners:
            rank = float(ranks[winner])
            relative = rank / (len(names) + 1)
            logit = float(np.log(relative / (1 - relative)))
            records.append({"split": split, "is_blocks": ",".join(map(str, selected)), "winner": names[winner],
                            "selection_weight": 1 / len(winners), "is_best_ties": len(winners),
                            "is_rows": int(counts[inside].sum()), "oos_rows": int(counts[outside].sum()),
                            "is_score": float(is_score[winner]), "oos_score": float(oos_score[winner]),
                            "oos_rank": rank, "oos_relative_rank": relative, "oos_logit": logit,
                            "strictly_below_median": relative < .5 - TOLERANCE,
                            "at_median": abs(relative - .5) <= TOLERANCE,
                            "pbo_le_zero": relative <= .5 + TOLERANCE,
                            "oos_negative_score": oos_score[winner] < -TOLERANCE,
                            "is_zero_score_candidates": int(is_zero.sum()), "oos_zero_score_candidates": int(oos_zero.sum())})
    rows = pd.DataFrame(records)
    expected = math.comb(blocks, blocks // 2)
    weights = rows.groupby("split").selection_weight.sum()
    np.testing.assert_allclose(weights, 1., atol=1e-14, rtol=0)
    require(len(weights) == expected, "没有枚举全部对称分割。")
    summary = {"blocks": blocks, "combinations": expected, "candidates": names,
               "pbo_le_zero": float(np.sum(rows.selection_weight * rows.pbo_le_zero) / expected),
               "strictly_below_median_fraction": float(np.sum(rows.selection_weight * rows.strictly_below_median) / expected),
               "median_mass_fraction": float(np.sum(rows.selection_weight * rows.at_median) / expected),
               "selected_oos_negative_fraction": float(np.sum(rows.selection_weight * rows.oos_negative_score) / expected),
               "weighted_oos_score": float(np.sum(rows.selection_weight * rows.oos_score) / expected),
               "splits_with_is_best_ties": int(rows.loc[rows.is_best_ties.gt(1), "split"].nunique()),
               "splits_with_oos_zero_candidates": int(rows.loc[rows.oos_zero_score_candidates.gt(0), "split"].nunique())}
    selection = rows.groupby("winner").apply(lambda g: pd.Series({
        "selection_fraction": g.selection_weight.sum() / expected,
        "selected_oos_score_mean": np.average(g.oos_score, weights=g.selection_weight),
        "selected_oos_below_median": np.average(g.strictly_below_median, weights=g.selection_weight),
        "selected_oos_negative": np.average(g.oos_negative_score, weights=g.selection_weight),
    }), include_groups=False).reset_index()
    return summary, rows, pd.DataFrame(blocks_rows), selection


def source_directory(candidate, period, cost):
    study, mode = CANDIDATES[candidate]
    return ROOT / "reports/research" / study / "results/accounts" / period / cost / mode


def prepare():
    require(not (OUT / "protocol.json").exists(), "固定诊断已经登记，不能覆盖。")
    receipt = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(receipt["passed"] == 6 and receipt["exit_code"] == 0, "必要统计测试尚未全部通过。")
    require(receipt["module_sha256"] == digest(Path(__file__)), "统计实现与测试版本不一致。")
    sources = {Path(__file__), ROOT / "tests/test_point_comparable_selection_v1.py", OUT / "tests_receipt.json"}
    inventory, identities = [], []
    for period in PERIODS:
        frames = {cost: {} for cost in COSTS}
        for candidate in CANDIDATES:
            directories = {cost: source_directory(candidate, period, cost) for cost in COSTS}
            availability = {cost: (directory / "daily.parquet").exists() for cost, directory in directories.items()}
            if not all(availability.values()):
                inventory.append({"period": period, "candidate": candidate, "status": "NOT_AVAILABLE_NOT_BACKFILLED", "same_as": None})
                continue
            for cost, directory in directories.items():
                daily = pd.read_parquet(directory / "daily.parquet")
                expected = daily.equity.to_numpy() / np.r_[200000., daily.equity.to_numpy()[:-1]] - 1
                np.testing.assert_allclose(daily.net_return, expected, atol=1e-13, rtol=0)
                frames[cost][candidate] = daily
                sources.update([directory / "daily.parquet", directory / "trades.parquet"])
            sources.add(ROOT / "reports/research" / CANDIDATES[candidate][0] / "protocol.json")
        panels = {cost: align_returns(frames[cost]) for cost in COSTS}
        require(panels["BASE"].date.equals(panels["STRESS"].date), "基础与压力账户日历不同。")
        kept, records = unique_candidates(panels)
        inventory.extend({"period": period, **row} for row in records)
        for cost, panel in panels.items():
            path = OUT / "inputs" / f"{period}_{cost}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(path, index=False)
            sources.add(path)
            identities.append({"period": period, "cost": cost, "rows": len(panel), "raw_candidates": len(panel.columns) - 1,
                               "ranking_candidates": kept, "first_date": str(panel.date.min().date()), "last_date": str(panel.date.max().date())})
    table("候选准入与完整重复身份", pd.DataFrame(inventory))
    protocol = {"study": "510300_POINT_COMPARABLE_SELECTION_V1", "registered_at": now(),
                "registration_decision": "TECH.R155", "result_decision": "TECH.R156", "blocks": BLOCKS,
                "question": "有限技术线可比账户的历史最优选择，在其余历史时间块是否仍有相对优势？",
                "known_before_registration": "已知全部账户结果及多轮历史选择；本次派生选择统计尚未计算。原A近期压力夏普1.217、较早0.438；V3已完成失败，不重新运行。",
                "hypothesis": "若局部最优依赖特定行情或选择，则对称分割的其余半样本排名和收益会不稳定。",
                "cohort": "仅六项本技术线研究、11个既存身份；20万元、50%上限、原ES/跳空/回撤预算、现金0、252日、两成本。不是全项目搜索宇宙。",
                "calendar": "两个独立启动时期分别计算；16连续近等长块，保留每个真实日；不拼接账户。",
                "ranking": "半样本真实净日收益均值除样本标准差乘sqrt252；精确全零半样本排序分数0。非零恒定收益拒绝。",
                "ties": "分数绝对差<=1e-12视为并列；IS最优平分选择权重，OOS使用平均名次。PBO为相对名次rank/(N+1)<=0.5，另报严格低于中位和中位质量。",
                "eligibility": "费用档都存在且日历完全一致；缺账户不补跑。两成本全零候选不参与排序；两成本全路径完全相同归并，原路径和失败保留。",
                "candidate_inventory": identities, "all_candidates": CANDIDATES,
                "no_promotion_gate": True, "parameter_search": False, "new_accounts": 0, "new_model_fits": 0,
                "limits": ["已知历史，全项目选择未纳入；低有限PBO也不能证明全项目没有过拟合。",
                           "时间块重组只是统计诊断，非按时间前行的训练或真实样本外账户。",
                           "长期持仓会跨块；模型原训练路径不重新切断，分割之间共享同一历史，组合数不是独立样本数。",
                           "两个时期可用候选集合不同，PBO不能直接当时期质量的同口径因果比较。",
                           "原策略裁决不变，不按新排名复活失败策略或新建调参路线。"],
                "global_dsr": "NOT_COMPUTED", "global_pbo": "NOT_COMPUTED_INCOMPLETE_GLOBAL_TRIAL_UNIVERSE",
                "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
                "source": "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf",
                "sources": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(sources)]}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("有限可比账户矩阵和16块选择诊断已冻结，尚未计算选择统计。", flush=True)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "固定统计已经开始，不重算覆盖。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for record in protocol["sources"]:
        require(digest(ROOT / record["path"]) == record["sha256"], "冻结来源改变：" + record["path"])
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "new_accounts": 0}, exclusive=True)
    summaries, selections, crossings = [], [], []
    for identity in protocol["candidate_inventory"]:
        period, cost = identity["period"], identity["cost"]
        raw = pd.read_parquet(OUT / "inputs" / f"{period}_{cost}.parquet")
        panel = raw[["date", *identity["ranking_candidates"]]]
        summary, rows, block_rows, selection = cscv(panel, protocol["blocks"])
        summary.update(period=period, cost=cost, rows=len(panel))
        summaries.append(summary)
        selection.insert(0, "cost", cost)
        selection.insert(0, "period", period)
        selections.append(selection)
        table(f"{period}_{cost}_全部选择分割", rows)
        table(f"{period}_{cost}_固定日历块", block_rows)
        dates = pd.DatetimeIndex(panel.date)
        block_index = np.repeat(np.arange(BLOCKS), block_rows.rows.to_numpy(int))
        for candidate in identity["ranking_candidates"]:
            trades = pd.read_parquet(source_directory(candidate, period, cost) / "trades.parquet")
            crossing_cycles, affected = 0, np.zeros(len(panel), dtype=bool)
            for trade in trades.itertuples(index=False):
                end = trade.exit_date if pd.notna(trade.exit_date) else dates[-1]
                included = (dates >= trade.entry_date) & (dates <= end)
                if len(np.unique(block_index[included])) > 1:
                    crossing_cycles += 1
                    affected |= included
            crossings.append({"period": period, "cost": cost, "candidate": candidate, "actual_cycles": len(trades),
                              "cross_block_cycles": crossing_cycles, "return_days_in_cross_block_cycles": int(affected.sum()),
                              "days": len(panel), "fraction_of_calendar_in_cross_block_cycles": float(affected.mean())})
    table("逐候选入选与其余半样本表现", pd.concat(selections, ignore_index=True))
    table("实际周期跨块依赖", pd.DataFrame(crossings))
    table("四场景有限选择稳定性", pd.DataFrame({k: v for k, v in r.items() if k != "candidates"} for r in summaries))
    result = {"study": protocol["study"], "completed_at": now(), "status": "COMPLETED_RESTRICTED_SELECTION_DIAGNOSTIC_NOT_STRATEGY_VALIDATION",
              "decision": "TECH.R156", "scenarios": summaries, "total_combinations": sum(s["combinations"] for s in summaries),
              "new_accounts": 0, "new_model_fits": 0, "new_training_labels": 0, "new_strategy_parameters": 0,
              "account_return_sharpe_improvement": "NOT_GENERATED_BY_STATISTICAL_DIAGNOSTIC",
              "latest_financial_strategy_decision": "TECH.R154", "global_dsr": protocol["global_dsr"], "global_pbo": protocol["global_pbo"],
              "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "overfitting_removed": False,
              "limits": protocol["limits"], "source": protocol["source"]}
    write_json(OUT / "summary.json", result, exclusive=True)
    print(pd.DataFrame({k: v for k, v in r.items() if k != "candidates"} for r in summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="有限可比账户的固定选择稳定性诊断。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    prepare() if args.command == "freeze" else run()
