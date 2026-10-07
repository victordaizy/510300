"""固定拆分融资收缩的买入、隐含偿还构成，并联合观察指数价格变化。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import multidim_nonlinear_score_v1 as common


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_financing_composition_v1"
STUDY = "510300_MULTIDIM_FINANCING_COMPOSITION_V1"
MARGIN = ROOT / "reports/research/510300_factor96_mechanism_batch_v1/data_repair/margin_complete.parquet"
DAILY = ROOT / "reports/research/510300_multidim_nonlinear_score_v1/historical_inputs_and_labels.parquet"
OLD_EVENTS = ROOT / "reports/research/510300_multidim_money_surprise_score_v1/逐事件联合评分.csv"
CAUSES = ["买入未增、偿还未增", "买入未增、偿还增加", "买入增加、偿还增加", "买入增加、偿还未增"]
PRICES = ["价格变化改善", "价格变化未改善"]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(common.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values: pd.Series):
    a = values.dropna().astype(float)
    p, n = a[a > 0], a[a < 0]
    return {"n": len(a), "mean_net5": float(a.mean()), "median_net5": float(a.median()),
            "win_rate": float(a.gt(0).mean()) if len(a) else np.nan,
            "return_payoff": float(p.mean() / -n.mean()) if len(p) and len(n) else np.nan,
            "minimum_net5": float(a.min()), "maximum_net5": float(a.max())}


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("融资构成比较已经固定，不覆盖。")
    save("protocol.json", {
        "study_id": STUDY, "frozen_at": common.now(),
        "previous_goal_turn_classification": "PROGRESS_EVENT_CLOCK_AND_STATE_SEPARATION",
        "question": "相同五日融资净收缩，借入买入和隐含偿还的不同变化，与指数价格变化共同出现时，其后净收益是否不同？",
        "scope": "仅沪深两市汇总融资环境与510300指数ETF；两市融资不是沪深300成分专属需求，也不包含逐户动机。",
        "primary_period": ["2024-01-01", "2025-12-31"], "earlier_context": ["2021-01-01", "2023-12-31"],
        "source_clock": "对齐完整ETF交易日后，将所有融资字段整体滞后一个交易日；价格和宏观背景只取判断日收盘前已知快照。保留历史回取和未逐日认证首次版本的限制。",
        "identity": "当日隐含偿还=融资买入额-(当日余额-前日余额)。当期五日净融资=五日买入-五日隐含偿还；相对前一不重叠五日的净融资变化=买入变化-隐含偿还变化。",
        "cause_boundary": "只识别两段等长期间的会计构成，不把隐含偿还称为卖压、强平或现金还款，也不声称知道客户动机。",
        "cohort": "判断日已知最近五个统计交易日余额净变化<0且当前和此前五日买入、隐含偿还完整；标签必须在2025年末以前成熟。",
        "composition": "比较最近五日与此前五日的总融资买入和总隐含偿还。严格增加为>0，未增加为<=0，形成固定四类，零值不删除。",
        "price_condition": "510300截至判断收盘的近五日财富收益，高于此前不重叠五日财富收益，记价格变化改善；否则未改善。它是价格描述，不能直接证明新买盘承接。",
        "groups": {"composition": CAUSES, "price": PRICES, "total": 8},
        "nonoverlap": "从2021年开始统一按时序选择首个合格收缩观察，原五日持有退出开盘之前不再选；从统一的非重叠样本中再分八组，不在每个组内另起优选时钟。",
        "episode": "连续的负五日融资状态属于同一收缩段；即使五日标签不重叠，也报告收缩段数量，不把每条观察当独立市场冲击。",
        "label": "复用原固定10000份、判断后次交易日开盘买入、入场后第5个交易日开盘卖出标签，单边佣金4bp最低5元、滑点10bp及价位取整，分红按登记日权益归属。",
        "comparisons": "全部八个组合、四种融资构成主效应、价格改善主效应、总体收缩对照；主要期和较早期分别报告，2024和2025另列。",
        "old_five_events": "原主要期五个高分公布事件按同一构成式解释，保留原日期和收益，不当作新策略样本重新扩大次数。",
        "closest_old_studies": [
            {"path": "reports/research/510300_factor96_mechanism_batch_v1/protocol.json", "boundary": "旧T03急跌修复加净收缩减速、T05高偿还加价格韧性均失败；不改其阈值、退出或时间。"},
            {"path": "reports/research/510300_historical_leverage_seller_constraints_v1/protocol.json", "boundary": "已有2024年1—2月固定案例分解，本轮不重复当作新发现。"},
            {"path": "reports/research/510300_historical_collateral_relief_cases_v1/protocol.json", "boundary": "已有2015/2018等放宽约束案例，不把放宽等同于止跌。"}
        ],
        "substantive_difference": "此前是价格事件加融资水平或减速过滤，以及少数政策窗口。本轮对全部合格收缩日期作买入变化、偿还变化、价格变化三维固定分组，先检验构成与条件差异，不训练新高分策略。",
        "selection_history": "既有融资及价格失败、原月度五次高分和普通日期失败已经看过，本轮是历史发现，不恢复独立性。",
        "new_fits": 0, "parameter_search": False, "new_accounts": 0,
        "promotion": "看到某格均值较高不直接选为交易规则；保留每格次数、反例、阶段差异和当前单笔集中度。",
        "source_receipts": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in [MARGIN, DAILY, OLD_EVENTS]],
        "official_definition": "https://www.sse.com.cn/market/othersdata/margin/detail/index.shtml",
        "orders_authorized": False, "goal_achieved": False, "new_prospective_forecasts_enabled": False,
    })
    print("已固定融资构成×价格变化的八种组合，先统一时序去重，零新增拟合。", flush=True)


def effective_sources():
    receipt_path = OUT / "data_correction/correction_receipt.json"
    if not receipt_path.exists():
        return MARGIN, DAILY
    receipt = read(receipt_path)
    margin_path, daily_path = ROOT / receipt["corrected_margin_path"], ROOT / receipt["corrected_daily_path"]
    assert sha(margin_path) == receipt["corrected_margin_sha256"]
    assert sha(daily_path) == receipt["corrected_daily_sha256"]
    return margin_path, daily_path


def build_inputs():
    margin_path, daily_path = effective_sources()
    d = pd.read_parquet(daily_path).sort_values("idx").reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date)
    margin = pd.read_parquet(margin_path).sort_values("date")
    margin["date"] = pd.to_datetime(margin.date)
    m = d[["date"]].merge(margin[["date", "market_rzye", "market_rzmre"]], on="date", how="left", validate="one_to_one")
    m["balance_change"] = m.market_rzye.diff()
    m["repayment_implied"] = m.market_rzmre - m.balance_change
    m["buy5"] = m.market_rzmre.rolling(5, min_periods=5).sum()
    m["repay5"] = m.repayment_implied.rolling(5, min_periods=5).sum()
    m["previous_buy5"], m["previous_repay5"] = m.buy5.shift(5), m.repay5.shift(5)
    m["net5"] = m.market_rzye - m.market_rzye.shift(5)
    m["previous_net5"] = m.net5.shift(5)
    m["buy_change5"], m["repay_change5"] = m.buy5 - m.previous_buy5, m.repay5 - m.previous_repay5
    m["net_change5"] = m.net5 - m.previous_net5
    m["balance_start5"] = m.market_rzye.shift(5)
    m["margin_stat_date"] = m.date.where(m.market_rzye.notna())
    cols = ["market_rzye", "market_rzmre", "repayment_implied", "buy5", "repay5", "previous_buy5", "previous_repay5",
            "net5", "previous_net5", "buy_change5", "repay_change5", "net_change5", "balance_start5", "margin_stat_date"]
    d[cols] = m[cols].shift(1)
    d["price_recent5"] = d.wealth / d.wealth.shift(5) - 1
    d["price_previous5"] = d.wealth.shift(5) / d.wealth.shift(10) - 1
    d["price_change5"] = d.price_recent5 - d.price_previous5
    d["composition_valid"] = np.isfinite(d[[c for c in cols if c != "margin_stat_date"] + ["price_recent5", "price_previous5"]].to_numpy(float)).all(axis=1)
    valid = d.composition_valid
    assert d.loc[valid, "margin_stat_date"].lt(d.loc[valid, "date"]).all()
    assert d.loc[valid, "repayment_implied"].ge(0).all()
    identity = (d.buy5 - d.repay5 - d.net5).abs()
    delta_identity = (d.buy_change5 - d.repay_change5 - d.net_change5).abs()
    assert identity[valid].max() < .01 and delta_identity[valid].max() < .01
    d["composition"] = np.select([
        d.buy_change5.le(0) & d.repay_change5.le(0),
        d.buy_change5.le(0) & d.repay_change5.gt(0),
        d.buy_change5.gt(0) & d.repay_change5.gt(0),
        d.buy_change5.gt(0) & d.repay_change5.le(0)], CAUSES, default="缺失")
    d["price_condition"] = np.where(d.price_change5.gt(0), PRICES[0], PRICES[1])
    d["net_five_return"] = d.net5 / d.balance_start5
    d["is_contraction"] = valid & d.net5.lt(0)
    d["episode_id"] = (d.is_contraction & ~d.is_contraction.shift(fill_value=False)).cumsum().where(d.is_contraction)
    d["entry_idx"] = d.idx + 1
    d["entry_date"] = d.date.shift(-1)
    d["exit_date"] = d.date.shift(-6)
    d["era"] = np.where(d.date.ge("2024-01-01"), "2024—2025", "2021—2023")
    d["eligible"] = d.date.between("2021-01-01", "2025-12-31") & d.is_contraction & d.exit_date.le("2025-12-31") & d.net_label.notna()
    d["selected_nonoverlap"] = False
    last_exit = -1
    for index, row in d[d.eligible].iterrows():
        if int(row.idx) >= last_exit:
            d.at[index, "selected_nonoverlap"] = True
            last_exit = int(row.exit_idx)
    return d, {"max_five_day_balance_identity_error_cny": float(identity[valid].max()),
               "max_between_window_identity_error_cny": float(delta_identity[valid].max()),
               "margin_source_days": len(margin), "all_margin_statistics_before_decision": True,
               "negative_implied_daily_repayments": int((d.loc[valid, "repayment_implied"] < 0).sum()),
               "effective_margin_source": margin_path.relative_to(ROOT).as_posix(),
               "effective_daily_source": daily_path.relative_to(ROOT).as_posix(),
               "single_official_data_correction_before_results": margin_path != MARGIN}


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("已有融资构成结果，不重跑。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["source_receipts"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    d, checks = build_inputs()
    inputs = d[d.date.between("2021-01-01", "2025-12-31")].copy()
    inputs.to_parquet(OUT / "全部日期的融资构成与价格状态.parquet", index=False)
    cohort = d[d.eligible].copy()
    chosen = cohort[cohort.selected_nonoverlap].copy()
    assert (chosen.entry_idx.to_numpy()[1:] > chosen.exit_idx.to_numpy()[:-1]).all()
    cols = ["date", "entry_date", "exit_date", "idx", "entry_idx", "exit_idx", "margin_stat_date", "era", "episode_id",
            "composition", "price_condition", "buy5", "repay5", "previous_buy5", "previous_repay5", "net5", "previous_net5",
            "buy_change5", "repay_change5", "net_change5", "net_five_return", "price_recent5", "price_previous5", "price_change5",
            "订单水平", "订单月度变化", "资金利率与政策利率差", "net_label", "selected_nonoverlap"]
    cohort[cols].to_csv(OUT / "全部收缩观察.csv", index=False, encoding="utf-8-sig")
    chosen[cols].to_csv(OUT / "全部统一去重观察.csv", index=False, encoding="utf-8-sig")
    grids, margins, periods = [], [], []
    for selection, frame in [("全部重叠日观察", cohort), ("统一不重叠观察", chosen)]:
        blocks = [(era, frame[frame.era.eq(era)]) for era in ["2021—2023", "2024—2025"]]
        blocks += [(str(year), frame[pd.to_datetime(frame.entry_date).dt.year.eq(year)]) for year in [2024, 2025]]
        for era, block in blocks:
            periods.append({"selection": selection, "era": era, "episodes": block.episode_id.nunique(), **stats(block.net_label)})
            for cause in CAUSES:
                c = block[block.composition.eq(cause)]
                margins.append({"selection": selection, "era": era, "dimension": "融资构成", "state": cause, **stats(c.net_label)})
                for price in PRICES:
                    cell = c[c.price_condition.eq(price)]
                    grids.append({"selection": selection, "era": era, "composition": cause, "price_condition": price,
                                  "episodes": cell.episode_id.nunique(), **stats(cell.net_label)})
            for price in PRICES:
                cell = block[block.price_condition.eq(price)]
                margins.append({"selection": selection, "era": era, "dimension": "价格变化", "state": price, **stats(cell.net_label)})
    grid = pd.DataFrame(grids)
    grid.to_csv(OUT / "全部八组合结果.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(margins).to_csv(OUT / "单维主效应对照.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(periods).to_csv(OUT / "总体收缩对照.csv", index=False, encoding="utf-8-sig")
    old = pd.read_csv(OLD_EVENTS)
    old = old[old.era.eq("2024—2025") & old.state_score.ge(80) & old.state_prediction.gt(0)]
    cases = old[["stat_month", "state_idx", "state_score", "actual_net5"]].merge(d[cols], left_on="state_idx", right_on="idx", how="left", validate="one_to_one")
    assert len(cases) == 5 and np.allclose(cases.actual_net5, cases.net_label, atol=1e-12, rtol=0)
    assert cases.net5.lt(0).all()
    cases.to_csv(OUT / "原五次高分的收缩构成.csv", index=False, encoding="utf-8-sig")
    checks.update(total_contraction_days=len(cohort), total_nonoverlap_observations=len(chosen),
                  original_five_event_labels_unchanged=True, unified_nonoverlap_clock_before_grouping=True,
                  old_sources_unchanged=True)
    save("必要计算核对.json", checks)
    main = grid[grid.selection.eq("统一不重叠观察") & grid.era.eq("2024—2025")]
    save("result.json", {"study_id": STUDY, "completed_at": common.now(),
                         "status": "COMPLETED_COMPOSITION_PRICE_GROUPS_PENDING_INTERPRETATION",
                         "classification": "PROGRESS_FINANCING_COMPONENT_AND_PRICE_INTERACTION_DISCOVERY",
                         "primary_cells": main.to_dict("records"), "periods": periods,
                         "total_contraction_days": len(cohort), "total_nonoverlap_observations": len(chosen),
                         "new_fits": 0, "parameter_grids": 0, "new_accounts": 0,
                         "account_net_sharpe": None, "goal_achieved": False, "independent_validation": False,
                         "current_view": "NO_VIEW", "position": "UNSET", "orders_authorized": False,
                         "new_prospective_forecasts_enabled": False})
    print("融资构成与价格变化的八种组合已计算完成。", flush=True)
    print(main.to_string(index=False))
    print(cases[["stat_month", "entry_date", "composition", "price_condition", "buy_change5", "repay_change5", "net_change5", "net_label"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="融资构成与指数价格变化的固定历史分组")
    parser.add_argument("stage", choices=["prepare", "run"])
    {"prepare": prepare, "run": run}[parser.parse_args().stage]()
