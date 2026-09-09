"""央行实际操作规模与银行间资金压力的条件账户研究，仅历史模拟。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.adaptive_allocation_v1 import ROOT, identity, model_for, summarize, save_account, eligible_training
from research.intraday_overnight_increment_v1 import (
    Account, now, write_json, digest, require, choose_order, execute_order,
    fill_price, affordable_quantity, holding_total_return, normalize_dividends,
    block_indices, return_metrics, interval,
)
from research.asymmetric_stress_hazard_source_remediation_v1_0_1 import parse_pboc_open_market_notice

STUDY = "510300_POLICY_LIQUIDITY_QUANTITY_V1"
OUT = ROOT / "reports/research/510300_policy_liquidity_quantity_v1"
CONFIG = ROOT / "config/510300_policy_liquidity_quantity_v1.json"
MANIFEST = ROOT / "config/510300_policy_liquidity_quantity_v1_manifest.json"
PARENT = ROOT / "reports/research/510300_adaptive_allocation_v1"
NOTICE = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc_open_market_notice_ledger_20150105_20260814.parquet"
DR007 = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet"
FUNDING = ["dr007_known", "policy_rate_known", "funding_gap", "dr007_change5", "dr007_change20", "funding_gap_mean20"]
QUANTITY = ["quantity_log", "quantity_surprise20", "quantity_change5_20", "quantity_age_days", "quantity_fresh_today", "quantity_funding_interaction"]
ENSEMBLES = {"Q1_PRIMARY_FULL_H20": ("FULL", 20), "Q2_FUNDING_H20": ("FUNDING", 20),
             "Q3_PRICE_H20": ("PRICE", 20), "Q4_FULL_H60": ("FULL", 60)}
NAMES = {"Q1_PRIMARY_FULL_H20": "主方案：操作量与资金压力二十日共识",
         "Q2_FUNDING_H20": "资金信息二十日共识", "Q3_PRICE_H20": "价格信息二十日共识",
         "Q4_FULL_H60": "操作量与资金压力六十日共识", "BUY_HOLD": "买入持有"}


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "本轮已经冻结，禁止覆盖")
    config = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    config.update({"study_id": STUDY, "primary": "Q1_PRIMARY_FULL_H20", "model_start": "2019-12-31",
                   "funding_columns": FUNDING, "quantity_columns": QUANTITY,
                   "maximum_quantity_age_days": 10, "maximum_dr007_age_days": 7,
                   "common_feature_support_across_groups": True,
                   "historical_vendor_first_delivery_proven": False})
    config["models"] = [{"id": f"{group}_{kind}_H{h}", "group": group, "kind": kind, "horizon": h}
                         for group in ("PRICE", "FUNDING", "FULL") for kind in ("RIDGE", "ET") for h in (20, 60)]
    config["source_inputs"] = {"notice_ledger": str(NOTICE.relative_to(ROOT).as_posix()),
                                "dr007": str(DR007.relative_to(ROOT).as_posix())}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, config, exclusive=True)
    paths = [CONFIG, Path(__file__), ROOT / "docs/510300_POLICY_LIQUIDITY_QUANTITY_V1_PROTOCOL.md",
             ROOT / "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
             ROOT / "docs/510300_ADAPTIVE_ALLOCATION_V1_PROTOCOL.md", ROOT / "config/510300_research_authority_v6.json",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "research/asymmetric_stress_hazard_source_remediation_v1_0_1.py",
             ROOT / "tests/test_policy_liquidity_quantity_v1.py", PARENT / "features.parquet", PARENT / "input_receipt.json",
             NOTICE, DR007, ROOT / config["inputs"]["dividends"],
             NOTICE.parent / "pboc_source_acquisition_manifest.json", DR007.parent / "dr007_tushare_source_acquisition_manifest.json"]
    paths.extend(ROOT / p for p in pd.read_parquet(NOTICE).raw_path)
    parent_manifest_path = ROOT / "config/510300_adaptive_allocation_v1_manifest.json"
    paths.append(parent_manifest_path)
    paths.extend(ROOT / item["path"] for item in json.loads(parent_manifest_path.read_text(encoding="utf-8"))["files"])
    dr_receipt = json.loads((DR007.parent / "dr007_tushare_source_acquisition_manifest.json").read_text(encoding="utf-8"))
    require(dr_receipt["credential_archived"] is False, "DR007 来源记录不应归档凭证")
    paths.extend(ROOT / item["response_artifact"]["path"] for item in dr_receipt["chunk_receipts"])
    unique = sorted(set(paths))
    write_json(MANIFEST, {"study_id": STUDY, "frozen_at": now(), "files": [identity(p) for p in unique],
               "own_candidate_returns_read_before_freeze": False, "underlying_history_previously_observed": True,
               "prior_rounds_known": 3, "previous_valuation_and_macro_failures_preserved": True}, exclusive=True)
    print(json.dumps({"状态": "第四轮已冻结", "清单哈希": digest(MANIFEST)}, ensure_ascii=False), flush=True)


def verify_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    notices = pd.read_parquet(NOTICE).sort_values("published_at").reset_index(drop=True)
    checked = []
    for number, row in enumerate(notices.itertuples(), 1):
        path = ROOT / row.raw_path
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == row.raw_sha256, "央行原始公告身份不符")
        parsed = parse_pboc_open_market_notice(payload, source_url=row.source_url)
        for field in ("seven_day_operation_amount_100m", "seven_day_rate_percent"):
            left, right = getattr(parsed, field), getattr(row, field)
            require((left is None and pd.isna(right)) or (left is not None and not pd.isna(right) and abs(left - right) < 1e-12), f"央行规模或利率重解析不一致：{row.source_url}")
        require(pd.Timestamp(parsed.published_at) == row.published_at and parsed.parse_status == row.parse_status,
                "央行公告时间或状态重解析不一致")
        checked.append({"raw_path": row.raw_path, "source_url": row.source_url, "raw_sha256": row.raw_sha256,
                        "amount": parsed.seven_day_operation_amount_100m, "rate": parsed.seven_day_rate_percent,
                        "status": parsed.parse_status, "published_at": parsed.published_at})
        if number % 700 == 0:
            print(f"央行公告规模与时钟已重解析 {number}/{len(notices)} 条", flush=True)
    dr = pd.read_parquet(DR007).sort_values("date").reset_index(drop=True)
    require(set(dr.ts_code) == {"DR007.IB"} and set(dr.repo_maturity) == {"DR007"} and set(dr.provider_value_field) == {"weight"}, "实际资金利率身份或字段不符")
    require(not dr.date.duplicated().any() and np.isfinite(dr.dr007).all(), "DR007 日期重复或缺失")
    require(set(dr.availability_rule) == {"NEXT_TRADING_DAY_OPEN_AFTER_RATE_DATE"}, "DR007 可用时钟变化")
    pd.DataFrame(checked).to_parquet(OUT / "notice_reparse_receipts.parquet", index=False)
    write_json(OUT / "quantity_source_admission.json", {"status": "PASS_ARCHIVED_AMOUNT_RATE_AND_PUBLICATION_CLOCK_REPARSE",
               "notice_count": len(notices), "positive_amount_count": int((notices.seven_day_operation_amount_100m > 0).sum()),
               "explicit_zero_count": int((notices.seven_day_operation_amount_100m == 0).sum()),
               "absent_7d_row_count": int((~notices.has_seven_day_row).sum()), "dr007_rows": len(dr),
               "absent_row_filled_with_zero": False, "gross_seven_day_only_not_net_injection": True,
               "historical_real_time_delivery_proven": False})
    return notices, dr


def quantity_events(notices: pd.DataFrame) -> pd.DataFrame:
    selected = notices.loc[notices.has_seven_day_row & notices.seven_day_operation_amount_100m.notna()].copy()
    selected = selected.sort_values("published_at").reset_index(drop=True)
    require(not selected.notice_date.duplicated().any(), "同日明确七天规模记录重复，需独立裁定")
    require((selected.seven_day_operation_amount_100m >= 0).all(), "七天操作量为负")
    amount = np.log1p(selected.seven_day_operation_amount_100m)
    historical_mean = amount.shift(1).rolling(20).mean()
    historical_sd = amount.shift(1).rolling(20).std(ddof=1)
    selected["quantity_log"] = amount
    surprise = (amount - historical_mean) / historical_sd.replace(0, np.nan)
    selected["quantity_surprise20"] = surprise.where(historical_sd != 0, 0)
    selected["quantity_change5_20"] = amount.rolling(5).mean() - amount.rolling(20).mean()
    return selected


def asof_join(left: pd.DataFrame, right: pd.DataFrame, time: str) -> pd.DataFrame:
    a, b = left.copy(), right.copy()
    a["origin_time"] = pd.to_datetime(a.origin_time).astype("datetime64[ns]")
    b[time] = pd.to_datetime(b[time]).astype("datetime64[ns]")
    return pd.merge_asof(a.sort_values("origin_time"), b.sort_values(time), left_on="origin_time", right_on=time, direction="backward")


def prepare_features(data: pd.DataFrame, notices: pd.DataFrame, dr: pd.DataFrame, config: dict) -> pd.DataFrame:
    result = data.copy()
    result["origin_time"] = result.date + pd.Timedelta(hours=15)
    dates = pd.DatetimeIndex(result.date)
    observations = dr.copy()
    ix = dates.searchsorted(pd.to_datetime(observations.date), side="right")
    observations = observations.loc[ix < len(dates)].copy()
    ix = ix[ix < len(dates)]
    observations["dr_available_at"] = dates[ix].to_numpy() + pd.Timedelta(hours=9, minutes=30)
    observations = observations.rename(columns={"date": "dr_date"})
    # 非股票交易日的多条利率记录可在同一开盘可用；按原利率日期保留最新已知一条。
    observations = observations.sort_values(["dr_available_at", "dr_date"]).drop_duplicates("dr_available_at", keep="last")
    result = asof_join(result, observations[["dr_date", "dr_available_at", "dr007"]], "dr_available_at")
    rates = notices.loc[notices.seven_day_rate_percent.notna(), ["published_at", "seven_day_rate_percent"]].rename(columns={"published_at": "rate_published_at"})
    result = asof_join(result, rates, "rate_published_at")
    events = quantity_events(notices)[["notice_date", "published_at", "quantity_log", "quantity_surprise20", "quantity_change5_20", "source_url", "raw_path"]].rename(columns={"published_at": "quantity_published_at", "notice_date": "quantity_notice_date"})
    result = asof_join(result, events, "quantity_published_at")
    result["dr007_known"] = result.dr007 / 100
    result["policy_rate_known"] = result.seven_day_rate_percent / 100
    result["funding_gap"] = result.dr007_known - result.policy_rate_known
    result["dr007_change5"] = result.dr007_known.diff(5)
    result["dr007_change20"] = result.dr007_known.diff(20)
    result["funding_gap_mean20"] = result.funding_gap.rolling(20).mean()
    result["quantity_age_days"] = (result.origin_time - result.quantity_published_at).dt.total_seconds() / 86400
    result["quantity_fresh_today"] = (result.quantity_notice_date == result.date).astype(float)
    result["quantity_funding_interaction"] = result.quantity_surprise20 * result.funding_gap
    result["dr_age_days"] = (result.date - result.dr_date).dt.total_seconds() / 86400
    result["price_feature_valid"] = result.feature_valid
    result["feature_valid"] &= np.isfinite(result[FUNDING + QUANTITY]).all(axis=1)
    result["feature_valid"] &= result.quantity_age_days <= config["maximum_quantity_age_days"]
    result["feature_valid"] &= result.dr_age_days <= config["maximum_dr007_age_days"]
    for time in ("dr_available_at", "rate_published_at", "quantity_published_at"):
        valid = result[time].notna()
        require((result.loc[valid, time] <= result.loc[valid, "origin_time"]).all(), "宏观资料穿越决策时点")
    return result


def train_all(data: pd.DataFrame, dividends: pd.DataFrame, price_columns: list[str], config: dict) -> dict[str, np.ndarray]:
    first = int(np.flatnonzero(data.date >= config["evaluation_start"])[0]) - 1
    quarters = data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts = [first] + [t for t in range(first + 1, len(data) - 1) if quarters[t] != quarters[t - 1]] + [len(data) - 1]
    labels = {}
    for h in (20, 60):
        y = np.full(len(data), np.nan)
        for t in range(len(data) - h - 1):
            y[t] = holding_total_return(data, dividends, t + 1, t + h + 1)[0]
        labels[h] = y
    receipts, predictions = [], {}
    (OUT / "final_models").mkdir(exist_ok=True)
    for number, item in enumerate(config["models"], 1):
        columns = price_columns + (FUNDING if item["group"] in ("FUNDING", "FULL") else []) + (QUANTITY if item["group"] == "FULL" else [])
        x = data[columns].to_numpy(float)
        y = labels[item["horizon"]]
        prediction = np.full(len(data), np.nan)
        for t, end in zip(cuts[:-1], cuts[1:]):
            train = eligible_training(data, t, item["horizon"], None)
            require(len(train) >= config["minimum_train_samples"], "本轮共同有效成熟样本不足")
            require(np.isfinite(x[train]).all() and np.isfinite(y[train]).all(), "训练输入不完整")
            pred_ix = np.arange(t, end)[data.feature_valid.iloc[t:end].to_numpy()]
            seed = config["random_seed"] + int(data.date.iloc[t].strftime("%Y%m%d"))
            model = model_for(item["kind"], seed)
            with threadpool_limits(limits=1):
                model.fit(x[train], y[train])
                if len(pred_ix):
                    prediction[pred_ix] = model.predict(x[pred_ix])
            receipts.append({"model": item["id"], "group": item["group"], "horizon": item["horizon"],
                             "fit_origin": data.date.iloc[t], "train_rows": len(train),
                             "first_train_origin": data.date.iloc[train[0]], "last_train_origin": data.date.iloc[train[-1]],
                             "last_label_exit": data.date.iloc[train[-1] + item["horizon"] + 1],
                             "training_origin_sha256": hashlib.sha256(train.astype(np.int64).tobytes()).hexdigest(),
                             "training_array_sha256": hashlib.sha256(x[train].tobytes() + y[train].tobytes()).hexdigest(),
                             "last_prediction_origin": data.date.iloc[end - 1], "random_seed": seed})
        joblib.dump({"model": model, "features": columns, "specification": item, "last_fit_receipt": receipts[-1]},
                     OUT / "final_models" / f"{item['id']}.joblib", compress=3)
        predictions[item["id"]] = prediction
        print(f"资金与操作量模型已完成 {number}/{len(config['models'])}：{item['id']}", flush=True)
    for key, (group, h) in ENSEMBLES.items():
        predictions[key] = np.mean(np.column_stack([predictions[f"{group}_{kind}_H{h}"] for kind in ("RIDGE", "ET")]), axis=1)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "predictions.parquet", index=False)
    pd.DataFrame({"date": data.date, **{f"Y{h}": y for h, y in labels.items()}}).to_parquet(OUT / "labels.parquet", index=False)
    return predictions


def simulate_policy(data: pd.DataFrame, dividends: pd.DataFrame, config: dict, cost: dict, model_id: str,
                    prediction: np.ndarray | None, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = int(np.flatnonzero(data.date >= config["evaluation_start"])[0])
    last, anchor = len(data) - 1, first - 1
    account = Account(config["initial_capital"])
    dates = pd.DatetimeIndex(data.date)
    values = {name: data[name].to_numpy() for name in ("open", "close", "previous_close", "dividend", "variance60", "feature_valid")}
    events = dividends.to_dict("records")
    previous_nav, previous_mark = config["initial_capital"], float(values["close"][anchor])
    records, decisions = [], []
    reference = None

    def decide(t: int) -> dict:
        nonlocal reference
        close = float(values["close"][t])
        if model_id == "BUY_HOLD":
            quantity = affordable_quantity(account.cash, fill_price(close, 1, cost, config["tick"]), cost, config["lot"]) if t == anchor else 0
            decision = {"requested_quantity": quantity, "reference_weight": 1.0, "action": "买入持有", "view": "有效"}
        elif (t - anchor) % horizon:
            decision = {"requested_quantity": 0, "reference_weight": reference, "action": "非调整日保持份额", "view": "未重新决策"}
        elif not values["feature_valid"][t] or prediction is None or not np.isfinite(prediction[t]):
            decision = {"requested_quantity": 0, "reference_weight": None, "action": "无有效判断，保持已有份额", "view": "NO_VIEW"}
        else:
            decision = choose_order(account, close, float(prediction[t]), horizon * float(values["variance60"][t]), cost, config)
            decision["view"] = "有效"
        reference = decision["reference_weight"]
        decisions.append({"origin": dates[t], "decision_time": dates[t] + pd.Timedelta(hours=15),
                          "execution_date": dates[t + 1], "simulation_only": True, **decision})
        return decision

    pending = decide(anchor)
    for day in range(first, last + 1):
        date, op, cl = dates[day], float(values["open"][day]), float(values["close"][day])
        old_shares, recognized, paid = account.shares, 0.0, 0.0
        for key, event in enumerate(events):
            if event["ex_date"] == date:
                value = account.entitlements.get(key, 0) * event["cash_dividend_per_share"]
                account.receivables[key] = value
                recognized += value
            if event["payment_date"] < date and key in account.receivables:
                value = account.receivables.pop(key)
                account.cash += value
                paid += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        pretrade = account.value(op)
        execution = execute_order(account, quantity, op, float(values["previous_close"][day]), float(values["dividend"][day]), day, cost, config)
        mark = op if terminal else cl
        if not terminal:
            for key, event in enumerate(events):
                if event["payment_date"] == date and key in account.receivables:
                    value = account.receivables.pop(key)
                    account.cash += value
                    paid += value
                if event["record_date"] == date:
                    account.entitlements[key] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"资金操作量账户财富不守恒：{model_id}，{date}")
        account.assert_valid()
        records.append({"date": date, "open": op, "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * mark / nav, "accounting_error": error,
                        "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / pretrade, "view_at_order_origin": pending["view"], **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    return pd.DataFrame(records), pd.DataFrame(decisions)


def label(key: str) -> str:
    if key in NAMES:
        return NAMES[key]
    group, kind, horizon = key.split("_")
    return {"PRICE": "价格组", "FUNDING": "价格与资金组", "FULL": "价格、资金与操作量组"}[group] + "／" + {"RIDGE": "岭回归", "ET": "极随机树"}[kind] + f"／{horizon[1:]}日"


def run(expected: str) -> None:
    require(digest(MANIFEST) == expected, "第四轮冻结清单哈希不符")
    for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        path = ROOT / item["path"]
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], f"冻结文件已变化：{item['path']}")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected}, exclusive=True)
    notices, dr = verify_sources()
    base_data = pd.read_parquet(PARENT / "features.parquet")
    data = prepare_features(base_data, notices, dr, config)
    data.to_parquet(OUT / "features.parquet", index=False)
    price_columns = json.loads((PARENT / "input_receipt.json").read_text(encoding="utf-8"))["features"]
    dividends = normalize_dividends(pd.read_csv(ROOT / config["inputs"]["dividends"]))
    write_json(OUT / "feature_receipt.json", {"status": "PASS_COMMON_POINT_IN_TIME_FEATURE_SUPPORT",
               "price_columns": price_columns, "funding_columns": FUNDING, "quantity_columns": QUANTITY,
               "first_common_valid_origin": data.loc[data.feature_valid, "date"].iloc[0],
               "common_valid_origins": int(data.feature_valid.sum()),
               "no_view_evaluation_origins": data.loc[(data.date >= config["model_start"]) & (data.date < config["data_cutoff"]) & ~data.feature_valid, "date"].tolist()})
    predictions = train_all(data, dividends, price_columns, config)
    horizons = {item["id"]: item["horizon"] for item in config["models"]}
    horizons.update({key: h for key, (_, h) in ENSEMBLES.items()})
    metrics, yearly, eras, uncertainty = [], [], [], {}
    for cost_id, cost in config["costs"].items():
        ledgers = {}
        for key in list(predictions) + ["BUY_HOLD"]:
            ledger, decisions = simulate_policy(data, dividends, config, cost, key, predictions.get(key), horizons.get(key, 20))
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            ledgers[key] = ledger
        baseline = summarize(ledgers["BUY_HOLD"], config)
        for key, ledger in ledgers.items():
            item = {"cost": cost_id, "model": key, **summarize(ledger, config)}
            item["annualized_return_excess_vs_buy_hold"] = item["annualized_return"] - baseline["annualized_return"]
            item["meets_point_target"] = item["net_sharpe"] is not None and item["net_sharpe"] >= 1.2
            metrics.append(item)
            for year, part in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(part, config)})
            for name, start, end in (("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"),
                                     ("2024—终点", "2024-01-01", config["data_cutoff"])):
                part = ledger.loc[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": key, "era": name, **summarize(part, config)})
        returns = pd.DataFrame({"date": ledgers["BUY_HOLD"].date, **{k: v.net_return.to_numpy() for k, v in ledgers.items()}})
        returns.to_parquet(OUT / f"{cost_id}_all_evaluation_returns.parquet", index=False)
        primary = ledgers[config["primary"]].net_return.to_numpy()
        funding = ledgers["Q2_FUNDING_H20"].net_return.to_numpy()
        buy_hold = ledgers["BUY_HOLD"].net_return.to_numpy()
        rng = np.random.default_rng(config["random_seed"])
        samples = {"primary_sharpe": [], "annualized_arithmetic_excess_vs_buy_hold": [], "annualized_arithmetic_increment_vs_funding": []}
        for _ in range(config["bootstrap_repetitions"]):
            ix = block_indices(rng, len(primary), config["bootstrap_day_block"])
            sh = return_metrics(primary[ix], config["annual_days"])["net_sharpe"]
            samples["primary_sharpe"].append(np.nan if sh is None else sh)
            samples["annualized_arithmetic_excess_vs_buy_hold"].append(float((primary[ix] - buy_hold[ix]).mean() * 242))
            samples["annualized_arithmetic_increment_vs_funding"].append(float((primary[ix] - funding[ix]).mean() * 242))
        uncertainty[cost_id] = {**{key + "_95_interval": interval(value) for key, value in samples.items()},
                               "multiple_search_adjusted": False}
        print(f"第四轮 {cost_id} 的 17 个完整评价账户已完成", flush=True)
    frame = pd.DataFrame(metrics)
    frame.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    primary = frame.loc[frame.model == config["primary"]]
    best = frame.loc[(frame.cost == "BASE") & (frame.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    reached = bool(primary.loc[primary.cost == "BASE", "meets_point_target"].iloc[0])
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if reached else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary.to_dict("records"), "post_selected_best_base": best.to_dict(),
              "number_of_candidates": 16, "benchmark_count": 1, "trained_model_count": 12,
              "historical_vendor_first_delivery_proven": False, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY",
              "uncertainty": uncertainty, "position_impact": 0, "validated_live_strategy": "NONE"}
    write_json(OUT / "result.json", result, exclusive=True)
    write_json(OUT / "uncertainty.json", uncertainty)
    write_report(frame, pd.DataFrame(eras), result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


def write_report(frame: pd.DataFrame, eras: pd.DataFrame, result: dict) -> None:
    primary = result["primary"][0]
    best = result["post_selected_best_base"]
    lines = ["# 510300 央行操作量与资金压力：第四轮实际结果", "",
             f"预先指定主方案基础夏普为 **{primary['net_sharpe']:.4f}**，目标为 1.2。本轮事后最好候选为“{label(best['model'])}”，基础夏普 **{best['net_sharpe']:.4f}**。", "",
             "本轮新增央行七天逆回购已披露规模及其与 DR007 资金压力的交互；16 个候选与一个基准、两种费用，共实际计算 34 个完整评价账户。训练与预测采用三组共同有效时点，以便检验操作量是否带来额外信息。", "",
             "| 策略 | 基础夏普 | 压力夏普 | 基础年化收益 | 年化超额 | 最大回撤 | 平均仓位 | 成交笔数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    order = list(ENSEMBLES) + ["BUY_HOLD"] + [key for key in frame.model.unique() if key not in ENSEMBLES and key != "BUY_HOLD"]
    for key in order:
        a = frame.loc[(frame.cost == "BASE") & (frame.model == key)].iloc[0]
        b = frame.loc[(frame.cost == "STRESS") & (frame.model == key)].iloc[0]
        fmt = lambda value: "无定义" if pd.isna(value) else f"{value:.4f}"
        lines.append(f"| {label(key)} | {fmt(a.net_sharpe)} | {fmt(b.net_sharpe)} | {a.annualized_return:.2%} | {a.annualized_return_excess_vs_buy_hold:.2%} | {a.max_drawdown:.2%} | {a.mean_exposure:.2%} | {a.trade_count} |")
    lines.extend(["", "## 主方案分阶段", "", "| 阶段 | 基础夏普 | 年化收益 | 最大回撤 |", "| --- | ---: | ---: | ---: |"])
    for row in eras.loc[(eras.cost == "BASE") & (eras.model == "Q1_PRIMARY_FULL_H20")].itertuples():
        lines.append(f"| {row.era} | {row.net_sharpe:.4f} | {row.annualized_return:.2%} | {row.max_drawdown:.2%} |")
    for cost, item in result["uncertainty"].items():
        lines.extend(["", f"{cost}：主方案夏普的 95% 区间为 {item['primary_sharpe_95_interval']}；相对不含操作量的资金组共识，年化日均增量的 95% 区间为 {item['annualized_arithmetic_increment_vs_funding_95_interval']}。"])
    lines.extend(["", "这些区间来自二十日区块重抽样，没有宣称覆盖历次研究的多重选择影响。来源规模是七天品种的已披露操作量，不是净投放；没有七天操作行的公告不补零。历史页面时间也不是实时接口交付证明。", "",
                  "全部因子、信息时钟、训练、仓位与成本规则在 `docs/510300_POLICY_LIQUIDITY_QUANTITY_V1_PROTOCOL.md` 中用中文写明。现金等待日和无有效判断日都保留在净值中。", "",
                  f"结论状态：{result['status']}。原先失败的估值和宏观规则保持原结论，本轮未触发实际持仓或下单。"])
    (OUT / "510300央行操作量与资金压力_实际结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="冻结或执行央行操作量条件研究")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    require(args.freeze != args.run, "必须选择冻结或运行之一")
    if args.freeze:
        freeze()
    else:
        require(bool(args.expected_manifest_sha256), "缺少冻结清单哈希")
        run(args.expected_manifest_sha256)
