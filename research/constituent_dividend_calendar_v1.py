"""历史成员已公告派息覆盖面的单规则检验；明确供应商日期代理边界。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research.mechanism_odds_open_contract_v1 import save, read, now, digest
from research.lpr_joint_response_v1 import directional_limit, gross_return

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_constituent_dividend_calendar_v1"
SOURCE = ROOT / "data/raw/510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1/dividend_checkpoints"
MEMBERS = ROOT / "reports/research/510300_factor96_internal_reclaim_v1_0_1/inputs/membership.parquet"
MARKET = ROOT / "reports/research/510300_mechanism_odds_open_contract_v1/inputs/market.parquet"
PRIMARY = "ABOVE_SAME_MONTH_PRIOR_MEDIAN"
HOLD = 20
PERIODS = {"FULL": ("2017-01-01", "2025-12-31"),
           "EARLY": ("2017-01-01", "2020-12-31"),
           "LATE": ("2021-01-01", "2025-12-31")}


def normalize_schedules(raw):
    frame = raw.copy()
    frame["cash_div_tax"] = pd.to_numeric(frame.cash_div_tax, errors="coerce")
    frame = frame.loc[frame.div_proc.str.strip().eq("实施") & frame.cash_div_tax.gt(0)].copy()
    for col in ["imp_ann_date", "record_date", "ex_date", "pay_date"]:
        frame[col] = pd.to_datetime(frame[col], format="%Y%m%d", errors="coerce")
    # 已在2014年末公告、于2015年支付的事件也应进入最早日历。
    relevant = frame.imp_ann_date.le("2025-12-31") & (frame.pay_date.ge("2015-01-01") | frame.imp_ann_date.ge("2014-12-01"))
    frame = frame.loc[relevant].copy()
    required = ["imp_ann_date", "record_date", "ex_date", "pay_date"]
    assert not frame[required].isna().any().any(), "相关现金实施记录缺少必要日期。"
    assert (frame.imp_ann_date.lt(frame.record_date) & frame.record_date.lt(frame.ex_date) & frame.ex_date.le(frame.pay_date)).all(), "分红日期次序矛盾。"
    keys = ["ts_code", "imp_ann_date", "record_date", "ex_date", "pay_date"]
    schedules = frame[keys].drop_duplicates().sort_values(["imp_ann_date", "ts_code", "pay_date"]).reset_index(drop=True)
    # 不解释同日多条金额相加规则，只核对日历；发行人在窗口内最多计一次。
    by_notice = schedules.groupby(["ts_code", "imp_ann_date", "record_date"])[["ex_date", "pay_date"]].nunique()
    assert by_notice.le(1).all().all(), "同一实施公告与登记日存在互相冲突的派息日。"
    schedules["available_at"] = schedules.imp_ann_date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return schedules, {"positive_implemented_relevant_rows": len(frame), "distinct_payment_schedules": len(schedules),
                       "collapsed_same_schedule_rows": len(frame) - len(schedules), "cash_amounts_summed": False}


def prepare():
    assert not (OUT / "freeze.json").exists(), "本版本已冻结。"
    files = sorted(SOURCE.glob("*.parquet"))
    frames, manifests = [], []
    for path in files:
        frame = pd.read_parquet(path)
        symbol = path.stem.replace("_", ".")
        assert frame.empty or set(frame.ts_code) == {symbol}, "缓存文件与证券代码不一致。"
        frames.append(frame)
        manifests.append({"symbol": symbol, "path": str(path.relative_to(ROOT)), "sha256": digest(path), "rows": len(frame)})
    raw = pd.concat(frames, ignore_index=True)
    schedules, facts = normalize_schedules(raw)
    (OUT / "inputs").mkdir(exist_ok=True)
    schedules.to_parquet(OUT / "inputs/payment_schedules.parquet", index=False)
    shutil.copy2(MEMBERS, OUT / "inputs/membership.parquet")
    shutil.copy2(MARKET, OUT / "inputs/market.parquet")
    shutil.copy2(ROOT / "config/510300_existing_data_training_mandate_v1.json", OUT / "inputs/authority_snapshot.json")
    save(OUT / "source_files.json", manifests)
    save(OUT / "source_receipt.json", {"at": now(), "status": "PASS_LOCAL_PAYMENT_CALENDAR_PROXY_FIELDS",
         "source_files": len(files), "raw_rows": len(raw), **facts,
         "membership_rows": len(pd.read_parquet(MEMBERS)), "source_documentation": "https://tushare.pro/document/2?doc_id=103",
         "retrieved_at_range": [str(raw.retrieved_at.min()), str(raw.retrieved_at.max())],
         "historical_first_vintage_verified": False, "missing_files_are_not_zero_payments": True,
         "boundary": "实施日期是供应商历史字段；无完整原公告版本和股本金额，不等于严格点时M05现金流。",
         "new_data_api_requests": 0, "new_return_reads": 0})
    print(f"派息日历代理准备完成：{len(files)}份缓存、{len(schedules)}个去重日历；尚未计算新收益。", flush=True)


def seasonal_threshold(daily, minimum=20):
    result = daily.copy()
    result["seasonal_median"] = np.nan
    result["prior_same_month_days"] = 0
    result["selected"] = False
    for i, row in result.iterrows():
        if row.status != "SOURCE_OK":
            continue
        prior = result.loc[result.date.lt(row.date) & result.date.ge(row.date - pd.DateOffset(years=2))
                           & result.date.dt.month.eq(row.date.month) & result.breadth.notna()]
        result.at[i, "prior_same_month_days"] = len(prior)
        if len(prior) < minimum:
            result.at[i, "status"] = "NO_VIEW_SAME_MONTH_HISTORY"
            continue
        median = float(prior.breadth.median())
        result.at[i, "seasonal_median"] = median
        result.at[i, "selected"] = row.breadth > median
        result.at[i, "status"] = "READY"
    return result


def build_daily(calendar, membership, schedules, source_symbols):
    calendar = pd.DatetimeIndex(calendar)
    members = membership.copy()
    members["membership_date"] = pd.to_datetime(members.membership_date)
    groups = {day: set(part.symbol) for day, part in members.groupby("membership_date")}
    imp = schedules.imp_ann_date.to_numpy(dtype="datetime64[ns]")
    payments = schedules.pay_date.to_numpy(dtype="datetime64[ns]")
    symbols = schedules.ts_code.to_numpy(str)
    rows = []
    source_symbols = set(source_symbols)
    for i, day in enumerate(calendar):
        if day < pd.Timestamp("2015-01-01") or day > pd.Timestamp("2025-12-31"):
            continue
        current = groups.get(day, set())
        covered = current & source_symbols
        row = {"date": day, "idx": i, "members": len(current), "covered_members": len(covered),
               "coverage": len(covered) / 300., "missing_symbols": ";".join(sorted(current - covered)),
               "breadth": np.nan, "scheduled_issuer_count": np.nan, "scheduled_symbols": "",
               "status": "NO_VIEW_MEMBER_COVERAGE"}
        if len(current) != 300 or len(covered) < 285 or i + HOLD >= len(calendar):
            rows.append(row)
            continue
        end = calendar[i + HOLD]
        # 当天实施公告按日末可用，不能进入16时决策；实际支付日当天已经不算未来支付。
        known = (imp < day.to_datetime64()) & (payments > day.to_datetime64()) & (payments <= end.to_datetime64())
        scheduled = set(symbols[known]) & covered
        row.update(status="SOURCE_OK", horizon_end=end, scheduled_issuer_count=len(scheduled),
                   breadth=len(scheduled) / len(covered), scheduled_symbols=";".join(sorted(scheduled)))
        rows.append(row)
    return seasonal_threshold(pd.DataFrame(rows))


def weekly_origins(daily):
    result = daily.copy()
    iso = result.date.dt.isocalendar()
    result["iso_week"] = iso.year.astype(str) + "-" + iso.week.astype(str)
    result = result.drop_duplicates("iso_week", keep="first")
    return result.loc[result.date.between(*PERIODS["FULL"])].reset_index(drop=True)


def label_origins(origins, market):
    result = origins.copy()
    rows = []
    for origin in result.to_dict("records"):
        i = int(origin["idx"])
        row = {**origin, "information_status": origin["status"], "gross_return": np.nan,
               "stress_proportional_proxy_return": np.nan, "entry_idx": i + 1, "planned_exit_idx": i + 1 + HOLD}
        if row["status"] != "READY":
            rows.append(row)
            continue
        entry = i + 1
        if entry >= len(market):
            row["status"] = "CENSORED_NO_ENTRY_PRICE"
            rows.append(row)
            continue
        row["entry_date"] = market.date.iloc[entry]
        if directional_limit(market, entry, 1):
            row.update(status="UNFILLED_UPPER_LIMIT", gross_return=0., stress_proportional_proxy_return=0.,
                       exit_idx=entry, exit_date=market.date.iloc[entry])
            rows.append(row)
            continue
        end = entry + HOLD
        while end < len(market) and directional_limit(market, end, -1):
            end += 1
        if end >= len(market):
            row["status"] = "CENSORED_EXIT_AFTER_SAMPLE"
        else:
            value = gross_return(market, entry, end)
            row.update(status="MATURE", exit_idx=end, exit_date=market.date.iloc[end],
                       gross_return=value, stress_proportional_proxy_return=value - .0028,
                       exit_delay_sessions=end - entry - HOLD)
        rows.append(row)
    return pd.DataFrame(rows)


def matched_values(events):
    valid = events.loc[events.gross_return.notna()].copy()
    valid["year_month"] = valid.date.dt.strftime("%Y-%m")
    valid["matched_month_mean"] = valid.groupby("year_month").gross_return.transform("mean")
    valid["matched_difference"] = valid.gross_return - valid.matched_month_mean
    return valid


def statistics(events):
    valid = matched_values(events)
    rows = []
    periods = {**PERIODS, **{str(y): (f"{y}-01-01", f"{y}-12-31") for y in range(2017, 2026)}}
    for period, dates in periods.items():
        part = valid.loc[valid.date.between(*dates)]
        for group in [PRIMARY, "ALL_AVAILABLE_WEEKS"]:
            sample = part.loc[part.selected] if group == PRIMARY else part
            values = sample.gross_return
            rows.append({"period": period, "group": group, "n": len(sample), "filled": int(sample.status.eq("MATURE").sum()),
                         "mean_gross_return": values.mean(), "median_gross_return": values.median(),
                         "mean_stress_proportional_proxy_return": sample.stress_proportional_proxy_return.mean(),
                         "matched_month_mean": sample.matched_month_mean.mean(),
                         "matched_increment": sample.matched_difference.mean(),
                         "positive_fraction": values.gt(0).mean() if len(sample) else np.nan,
                         "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame(rows)


def uncertainty(events):
    n = len(events)
    rng = np.random.default_rng(20260928)
    indices = ((rng.integers(0, n, size=(2000, math.ceil(n / 8)))[:, :, None] + np.arange(8)) % n).reshape(2000, -1)[:, :n]
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=indices.astype(np.int16))
    values = events.gross_return.to_numpy(float)
    selected = events.selected.to_numpy(bool)
    month_codes, _ = pd.factorize(events.date.dt.strftime("%Y-%m"))
    draws = []
    for index in indices:
        v, s, codes = values[index], selected[index], month_codes[index]
        finite = np.isfinite(v)
        s = s & finite
        if not s.any():
            draws.append([np.nan, np.nan])
            continue
        totals = np.bincount(codes[finite], weights=v[finite], minlength=month_codes.max() + 1)
        counts = np.bincount(codes[finite], minlength=len(totals))
        means = np.divide(totals, counts, out=np.full(len(totals), np.nan), where=counts > 0)
        draws.append([v[s].mean(), (v[s] - means[codes[s]]).mean()])
    bounds = np.nanquantile(np.array(draws), [.025, .975], axis=0)
    return {"primary_mean_95_interval": bounds[:, 0].tolist(), "matched_increment_95_interval": bounds[:, 1].tolist(),
            "method": "完整周原点8周循环区块2000次；每次重新计算同年同月加权对照",
            "selection_bias_adjusted": False, "historical_independent_validation": False}


def freeze():
    assert not (OUT / "freeze.json").exists(), "本版本已冻结。"
    assert read(OUT / "prefreeze_tests.json")["exit_code"] == 0
    protocol = {**read(OUT / "design_registration.json"), "frozen_at": now(), "source_receipt": read(OUT / "source_receipt.json")}
    save(OUT / "protocol.json", protocol)
    paths = [Path(__file__), ROOT / "tests/test_constituent_dividend_calendar_v1.py",
             ROOT / "research/mechanism_odds_open_contract_v1.py", ROOT / "research/lpr_joint_response_v1.py",
             OUT / "design_registration.json", OUT / "protocol.json", OUT / "source_receipt.json",
             OUT / "source_files.json", OUT / "prefreeze_tests.json", *list((OUT / "inputs").glob("*"))]
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]})
    print("分红覆盖面一条规则、同月对照、20日期限及来源代理限制已冻结。", flush=True)


def run():
    for row in read(OUT / "freeze.json")["files"]:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]
    save(OUT / "run_started.json", {"at": now(), "stage": "FIXED_GROSS_SCREEN"})
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    membership = pd.read_parquet(OUT / "inputs/membership.parquet")
    schedules = pd.read_parquet(OUT / "inputs/payment_schedules.parquet")
    source_symbols = [r["symbol"] for r in read(OUT / "source_files.json")]
    daily = build_daily(market.date, membership, schedules, source_symbols)
    daily.to_parquet(OUT / "daily_information.parquet", index=False)
    origins = weekly_origins(daily)
    historical = market.loc[market.date.le("2025-12-31")].reset_index(drop=True)
    events = label_origins(origins, historical)
    events.to_parquet(OUT / "events.parquet", index=False)
    events.to_csv(OUT / "全部周原点与收益.csv", index=False, encoding="utf-8-sig")
    stats = statistics(events)
    stats.to_csv(OUT / "group_statistics.csv", index=False, encoding="utf-8-sig")
    parts = stats.loc[stats.group.eq(PRIMARY)].set_index("period")
    yearly = events.assign(year=events.date.dt.year).groupby("year").agg(origins=("date", "size"), ready=("information_status", lambda s: s.eq("READY").sum()))
    yearly["coverage"] = yearly.ready / yearly.origins
    yearly.to_csv(OUT / "年度来源覆盖.csv", encoding="utf-8-sig")
    source_gates = {"all_origins_coverage_at_least_90pct": events.information_status.eq("READY").mean() >= .9,
                    "each_year_coverage_at_least_85pct": yearly.coverage.ge(.85).all()}
    numeric_gates = {"mean_above_cost_proxy": parts.loc["FULL", "mean_gross_return"] > .0028,
                     "mean_above_matched_calendar_control": parts.loc["FULL", "matched_increment"] > 0,
                     "early_positive": parts.loc["EARLY", "mean_gross_return"] > 0,
                     "late_positive": parts.loc["LATE", "mean_gross_return"] > 0}
    source_pass, numeric_pass = all(source_gates.values()), all(numeric_gates.values())
    save(OUT / "uncertainty.json", uncertainty(events))
    selected = matched_values(events).loc[lambda x: x.selected]
    concentration = {"n": len(selected), "best_event": None, "mean_excluding_best": None}
    if len(selected):
        best = selected.gross_return.idxmax()
        total = selected.gross_return.sum()
        concentration.update(best_event=selected.loc[best].to_dict(), mean_excluding_best=selected.drop(index=best).gross_return.mean(),
                             best_share_of_arithmetic_sum=selected.loc[best, "gross_return"] / total if total else None)
    save(OUT / "concentration.json", concentration)
    # 后来公布的记录即使改动，也不应改变之前的日历与选择阈值。
    altered = schedules.copy()
    altered.loc[altered.imp_ann_date.ge("2022-01-01"), "pay_date"] += pd.Timedelta(days=90)
    rebuilt = build_daily(market.date, membership, altered, source_symbols)
    prior = daily.date.lt("2022-01-01")
    pd.testing.assert_frame_equal(daily.loc[prior], rebuilt.loc[prior])
    errors = []
    for row in events.loc[events.status.eq("MATURE")].itertuples():
        cash = math.fsum(historical.dividend.iloc[int(row.entry_idx) + 1:int(row.exit_idx) + 1])
        independent = (float(historical.open.iloc[int(row.exit_idx)]) + cash) / float(historical.open.iloc[int(row.entry_idx)]) - 1
        errors.append(abs(row.gross_return - independent))
    assert max(errors, default=0.) < 1e-12
    passed = source_pass and numeric_pass
    status = "PASS_GROSS_SCREEN_ACCOUNT_REQUIRED" if passed else ("REJECTED_FIXED_GROSS_SCREEN_NO_PARAMETER_RESCUE" if source_pass else "SOURCE_GATE_NOT_PASSED")
    result = {"at": now(), "study_id": "510300_CONSTITUENT_DIVIDEND_CALENDAR_V1", "status": status,
              "weekly_origins": len(events), "information_statuses": events.information_status.value_counts().to_dict(),
              "label_statuses": events.status.value_counts().to_dict(), "minimum_member_coverage": daily.coverage.min(),
              "source_gate": source_gates, "numeric_gate": numeric_gates,
              "primary": parts.loc["FULL"].to_dict(), "fixed_subperiods": parts.loc[["EARLY", "LATE"]].to_dict("index"),
              "account_status": "REQUIRED_NEXT_STAGE" if passed else "NOT_RUN_SCREEN_NOT_PASSED",
              "net_sharpe": None, "net_cagr": None, "maximum_drawdown": None,
              "strict_M05_status": "NOT_RUN_CASH_AMOUNT_AND_VINTAGE_SOURCE_GAPS", "historical_first_vintage_verified": False,
              "new_accounts": 0, "parameter_searches": 0, "independent_forward_observations": 0,
              "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result)
    save(OUT / "verification_receipt.json", {"at": now(), "status": "PASS_IMPLEMENTATION_CHECKS",
         "future_publication_prefix_checks": 1, "recomputed_mature_labels": len(errors),
         "maximum_label_difference": max(errors, default=0.), "new_accounts": 0,
         "boundary": "字段及实现核对不证明历史首版、独立验证或交易优势。"})
    print(f"成员派息覆盖面初筛完成：主组{len(selected)}次，毛均值{parts.loc['FULL', 'mean_gross_return']:.4%}，状态{status}。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="成员分红支付日历固定初筛")
    parser.add_argument("action", choices=["prepare", "freeze", "run"])
    {"prepare": prepare, "freeze": freeze, "run": run}[parser.parse_args().action]()
