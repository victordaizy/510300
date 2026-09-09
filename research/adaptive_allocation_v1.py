"""510300 条件切换、滚动模型与连续账户研究；仅研究模拟。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.intraday_overnight_increment_v1 import (
    Account, affordable_quantity, block_indices, build_features, choose_order,
    digest, execute_order, fill_price, holding_total_return, interval,
    normalize_dividends, normalize_prices, now, require, return_metrics, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = "510300_ADAPTIVE_ALLOCATION_V1"
OUT = ROOT / "reports/research/510300_adaptive_allocation_v1"
CONFIG = ROOT / "config/510300_adaptive_allocation_v1.json"
MANIFEST = ROOT / "config/510300_adaptive_allocation_v1_manifest.json"
RULE_NAMES = {
    "R01_SMA60": "六十日趋势", "R02_SMA120": "一百二十日趋势", "R03_SMA200": "二百日趋势",
    "R04_MOM20": "二十日动量", "R05_MOM60": "六十日动量", "R06_MOM120": "一百二十日动量",
    "R07_TREND_VOTE": "三周期趋势投票", "R08_BREAKOUT": "六十日突破",
    "R09_RSI2": "两日超跌反弹", "R10_RSI2_TREND": "上涨趋势中的超跌反弹",
    "R11_REGIME": "趋势与震荡切换", "R12_RISK_TREND": "风险约束趋势投票",
}
META_NAMES = {
    "P1_PRIMARY_TOP3_504": "主方案：过去两年选三种", "P2_TOP1_504": "过去两年选一种",
    "P3_TOP3_252": "过去一年选三种", "P4_EQUAL_ALL": "全部底层策略固定等权",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def identity(path: Path) -> dict:
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def spec() -> dict:
    models = [{"id": f"M_{kind}_H{horizon}_{window}", "kind": kind, "horizon": horizon,
               "train_window": None if window == "EXPANDING" else 756}
              for kind in ("RIDGE", "HGB", "ET") for horizon in (5, 20)
              for window in ("EXPANDING", "ROLL756")]
    return {
        "study_id": STUDY, "version": "1.0.0", "evidence_class": "PREQUENTIAL_REPLAY_OF_ALREADY_OBSERVED_HISTORY",
        "shadow_start": "2015-01-05", "evaluation_start": "2020-01-02", "data_cutoff": "2026-08-14",
        "initial_capital": 200000.0, "lot": 100, "tick": 0.001, "limit_fraction": 0.1,
        "annual_days": 242, "cash_annual_rate_assumption": 0.0, "high_sharpe_target": 1.2,
        "gamma": 4.0, "weights": [0.0, 0.25, 0.5, 0.75, 1.0], "weight_band": 0.1,
        "minimum_train_samples": 300, "random_seed": 20260906, "models": models,
        "rule_names": RULE_NAMES, "meta_names": META_NAMES, "primary": "P1_PRIMARY_TOP3_504",
        "costs": {"BASE": {"commission": 0.0002, "minimum": 5.0, "slippage": 0.0005},
                  "STRESS": {"commission": 0.0004, "minimum": 5.0, "slippage": 0.001}},
        "bootstrap_repetitions": 2000, "bootstrap_day_block": 20,
        "inputs": {
            "prices": "data/raw/market/510300_daily_downside_risk_v1.parquet",
            "original_prices": "data/raw/r6/510300_daily.parquet",
            "dividends": "data/reference/510300_dividends.csv",
            "calendar": "data/reference/a_share_hs_trading_calendar_2010_2026_v1.parquet",
            "price_receipt": "reports/data_quality/510300_downside_risk_inputs_v1.json",
            "dividend_coverage": "reports/research/510300_adaptive_allocation_v1/frozen_inputs/dividend_coverage.json",
            "corrections": "data/reference/510300_downside_risk_price_corrections_v1.csv",
        },
        "scope": ["510300.SH", "CASH_CNY"], "live_trading_authorized": False,
        "position_impact": 0, "future_service_started": False, "one_historical_run": True,
    }


def freeze() -> None:
    require(not CONFIG.exists() and not MANIFEST.exists(), "研究已经冻结，禁止覆盖")
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / "frozen_inputs"
    frozen.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data/reference/510300_dividends_coverage.json", frozen / "dividend_coverage.json")
    authorization = ROOT / "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md"
    old = ROOT / "config/510300_research_authority_v5.json"
    authority = json.loads(old.read_text(encoding="utf-8"))
    authority.update({
        "schema_version": "6.0.0", "authority_id": "510300_RESEARCH_AUTHORITY_V6", "as_of_date": "2026-09-06",
        "generated_at": now(), "research_state": "USER_RESUMED_METHOD_RESEARCH",
        "current_user_authorization": identity(authorization), "supersedes_authority": relative(old),
        "superseded_authority_identity": identity(old),
        "historical_scope_note": "本次用户要求不限方法继续夏普1.2研究，允许新的有限候选集、条件组合和历史滚动训练。旧失败记录保留，不把旧单项范围限制用于本轮。",
        "scope_note": "实际完成训练与完整账户；无预测显著性前置门。研究范围先保留510300及现金，其他ETF等待用户回复。",
        "authorized_historical_studies": [*authority["authorized_historical_studies"], STUDY],
        "historical_walk_forward_training_authorized": True,
        "new_method_research_authorized": True, "additional_etf_scope": "USER_QUESTION_PENDING",
    })
    authority_path = ROOT / "config/510300_research_authority_v6.json"
    write_json(authority_path, authority, exclusive=True)
    config = spec()
    config["authorization"] = relative(authority_path)
    write_json(CONFIG, config, exclusive=True)
    paths = [CONFIG, authority_path, authorization, ROOT / "docs/510300_ADAPTIVE_ALLOCATION_V1_PROTOCOL.md",
             Path(__file__), ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "tests/test_adaptive_allocation_v1.py"]
    paths.extend(ROOT / value for value in config["inputs"].values())
    manifest = {"study_id": STUDY, "frozen_at": now(), "files": [identity(path) for path in paths],
                "candidate_returns_read_before_freeze": False, "underlying_history_previously_observed": True,
                "environment": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    write_json(MANIFEST, manifest, exclusive=True)
    print(json.dumps({"状态": "方案已冻结", "清单哈希": digest(MANIFEST)}, ensure_ascii=False), flush=True)


def verify_freeze(expected: str) -> dict:
    require(digest(MANIFEST) == expected, "冻结清单哈希不符")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], f"冻结文件变化：{path}")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def factors(prices: pd.DataFrame, dividends: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = build_features(normalize_prices(prices), normalize_dividends(dividends), 20)
    r = data.total_simple
    lr = np.log1p(r)
    wealth = (1 + r.fillna(0)).cumprod()
    data["wealth"] = wealth
    columns = []

    def add(name: str, value) -> None:
        data[name] = value
        columns.append(name)

    for w in (1, 2, 5, 10, 20, 60, 120, 252):
        add(f"mom{w}", lr.rolling(w).sum())
    for w in (5, 20, 60, 120, 200):
        add(f"sma{w}", wealth / wealth.rolling(w).mean() - 1)
    for w in (5, 20, 60):
        add(f"vol{w}", r.rolling(w).std(ddof=1) * np.sqrt(242))
    for w in (20, 60):
        add(f"downvol{w}", np.sqrt(r.clip(upper=0).pow(2).rolling(w).mean() * 242))
    for w in (20, 60, 120):
        add(f"dd{w}", wealth / wealth.rolling(w).max() - 1)
    for component in ("overnight_log", "intraday_log"):
        for w in (1, 5, 20):
            add(f"{component}_{w}", data[component].rolling(w).sum())
    add("range", (data.high - data.low) / data.previous_close)
    add("close_location", ((data.close - data.low) / (data.high - data.low).replace(0, np.nan)).fillna(0.5))
    add("volume_ratio", np.log(data.volume / data.volume.rolling(20).mean()))
    changes = wealth.diff()
    add("efficiency20", changes.rolling(20).sum().abs() / changes.abs().rolling(20).sum().replace(0, np.nan))
    up = changes.clip(lower=0).rolling(2).sum()
    down = -changes.clip(upper=0).rolling(2).sum()
    add("rsi2", (100 * up / (up + down).replace(0, np.nan)).fillna(50))
    add("z20", ((wealth - wealth.rolling(20).mean()) / wealth.rolling(20).std(ddof=1).replace(0, np.nan)).fillna(0))
    add("vol_ratio", data.vol20 / data.vol60.clip(lower=1e-12))
    data["efficiency20"] = data.efficiency20.fillna(0)
    data["feature_valid"] = np.isfinite(data[columns]).all(axis=1)
    data["variance60"] = r.rolling(60).var(ddof=1)
    return data, columns


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    paths = {key: ROOT / value for key, value in config["inputs"].items()}
    receipt = json.loads(paths["price_receipt"].read_text(encoding="utf-8"))
    coverage = json.loads(paths["dividend_coverage"].read_text(encoding="utf-8"))
    require(receipt["status"] == "PASS", "行情核验未通过")
    require(coverage["complete_history_confirmed"] and coverage["coverage_end"] >= config["data_cutoff"], "分红覆盖不足")
    require(coverage["distribution_file_sha256"] == digest(paths["dividends"]), "分红文件与覆盖回执不符")
    dividends = normalize_dividends(pd.read_csv(paths["dividends"]))
    require(len(dividends) == coverage["event_count"], "分红事件数量不符")
    prices = normalize_prices(pd.read_parquet(paths["prices"]))
    prices = prices.loc[prices.date <= pd.Timestamp(config["data_cutoff"])].reset_index(drop=True)
    calendar = pd.read_parquet(paths["calendar"])
    dates = pd.to_datetime(calendar.loc[calendar.is_open == 1, "trade_date"])
    dates = dates.loc[(dates >= prices.date.iloc[0]) & (dates <= prices.date.iloc[-1])]
    require(pd.DatetimeIndex(dates).equals(pd.DatetimeIndex(prices.date)), "行情与完整交易日历不一致")
    data, columns = factors(prices, dividends)
    first = int(np.flatnonzero(data.date >= pd.Timestamp(config["shadow_start"]))[0])
    require(data.feature_valid.iloc[first - 1:].all(), "影子研究开始后的因子不完整")
    require(float(data.identity_error.iloc[1:].abs().max()) < 1e-12, "日内隔夜财富分解不成立")
    write_json(OUT / "input_receipt.json", {"status": "PASS", "rows": len(data), "features": columns,
               "first": data.date.iloc[0], "last": data.date.iloc[-1], "dividend_events": len(dividends),
               "first_valid_feature_date": data.loc[data.feature_valid, "date"].iloc[0]})
    return data, dividends, columns


def rule_targets(data: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(data)
    result = {}
    for key, w in (("R01_SMA60", 60), ("R02_SMA120", 120), ("R03_SMA200", 200)):
        result[key] = (data[f"sma{w}"] > 0).to_numpy(float)
    for key, w in (("R04_MOM20", 20), ("R05_MOM60", 60), ("R06_MOM120", 120)):
        result[key] = (data[f"mom{w}"] > 0).to_numpy(float)
    vote = (data[["sma20", "sma60", "sma120"]] > 0).mean(axis=1).to_numpy(float)
    result["R07_TREND_VOTE"] = vote
    high = data.wealth.shift().rolling(60).max().to_numpy()
    low = data.wealth.shift().rolling(20).min().to_numpy()
    for key in ("R08_BREAKOUT", "R09_RSI2", "R10_RSI2_TREND", "R11_REGIME"):
        target = np.zeros(n)
        held = 0.0
        for t in range(n):
            if not data.feature_valid.iloc[t]:
                continue
            row = data.iloc[t]
            if key == "R08_BREAKOUT":
                if row.wealth > high[t]:
                    held = 1.0
                elif row.wealth < low[t]:
                    held = 0.0
            elif key == "R09_RSI2":
                if row.rsi2 < 10:
                    held = 1.0
                elif row.rsi2 > 70:
                    held = 0.0
            elif key == "R10_RSI2_TREND":
                if row.sma200 <= 0 or row.rsi2 > 70:
                    held = 0.0
                elif row.rsi2 < 20:
                    held = 1.0
            elif row.efficiency20 > 0.3:
                held = float(row.sma20 > 0)
            elif row.z20 < -1:
                held = 1.0
            elif row.z20 > 0:
                held = 0.0
            target[t] = held
        result[key] = target
    result["R12_RISK_TREND"] = vote * np.minimum(1, 0.15 / data.vol60.clip(lower=1e-12).fillna(1).to_numpy())
    return result


def model_for(kind: str, seed: int):
    if kind == "RIDGE":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if kind == "HGB":
        return HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, max_leaf_nodes=7,
                    min_samples_leaf=60, l2_regularization=10.0, early_stopping=False, random_state=seed)
    if kind == "ET":
        return ExtraTreesRegressor(n_estimators=96, max_depth=3, min_samples_leaf=40,
                    max_features=0.75, bootstrap=False, random_state=seed, n_jobs=1)
    raise ValueError("未知模型")


def eligible_training(data: pd.DataFrame, origin: int, horizon: int, window: int | None) -> np.ndarray:
    origins = np.arange(len(data))
    mask = data.feature_valid.to_numpy() & (origins + 1 + horizon <= origin)
    if window is not None:
        mask &= origins >= origin - window
    return origins[mask]


def train_predict(data: pd.DataFrame, dividends: pd.DataFrame, columns: list[str], config: dict) -> dict[str, np.ndarray]:
    x = data[columns].to_numpy(float)
    start = int(np.flatnonzero(data.date >= pd.Timestamp(config["shadow_start"]))[0]) - 1
    quarters = data.date.dt.to_period("Q").astype(str).to_numpy()
    cuts = [start] + [t for t in range(start + 1, len(data) - 1) if quarters[t] != quarters[t - 1]] + [len(data) - 1]
    labels = {}
    for horizon in (5, 20):
        y = np.full(len(data), np.nan)
        for t in range(len(data) - horizon - 1):
            y[t] = holding_total_return(data, dividends, t + 1, t + 1 + horizon)[0]
        labels[horizon] = y
    records, predictions = [], {}
    (OUT / "final_models").mkdir(exist_ok=True)
    for number, model_spec in enumerate(config["models"], 1):
        model_id, horizon = model_spec["id"], model_spec["horizon"]
        y = labels[horizon]
        prediction = np.full(len(data), np.nan)
        for t, end in zip(cuts[:-1], cuts[1:]):
            train = eligible_training(data, t, horizon, model_spec["train_window"])
            require(len(train) >= config["minimum_train_samples"], f"{model_id} 在 {data.date.iloc[t]} 训练样本不足")
            require(np.isfinite(x[train]).all() and np.isfinite(y[train]).all(), "训练输入非有限")
            seed = config["random_seed"] + int(data.date.iloc[t].strftime("%Y%m%d"))
            model = model_for(model_spec["kind"], seed)
            with threadpool_limits(limits=1):
                model.fit(x[train], y[train])
                prediction[t:end] = model.predict(x[t:end])
            records.append({"model": model_id, "fit_origin": data.date.iloc[t], "train_rows": len(train),
                            "first_train_origin": data.date.iloc[train[0]], "last_train_origin": data.date.iloc[train[-1]],
                            "last_label_exit": data.date.iloc[train[-1] + 1 + horizon],
                            "last_prediction_origin": data.date.iloc[end - 1], "random_seed": seed,
                            "train_array_sha256": hashlib.sha256(x[train].tobytes() + y[train].tobytes()).hexdigest()})
        joblib.dump({"model": model, "features": columns, "specification": model_spec,
                     "last_fit_receipt": records[-1]}, OUT / "final_models" / f"{model_id}.joblib", compress=3)
        predictions[model_id] = prediction
        print(f"滚动模型已完成 {number}/{len(config['models'])}：{model_id}，季度拟合 {len(cuts) - 1} 次", flush=True)
    pd.DataFrame(records).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": data.date, **predictions}).to_parquet(OUT / "predictions.parquet", index=False)
    pd.DataFrame({"date": data.date, **{f"Y{h}": y for h, y in labels.items()}}).to_parquet(OUT / "labels.parquet", index=False)
    return predictions


def target_request(account: Account, close: float, target: float, config: dict) -> dict:
    require(np.isfinite(target) and 0 <= target <= 1, "目标仓位超出范围")
    nav = account.value(close)
    weight = account.shares * close / nav
    target_shares = int(math.floor(target * nav / close / config["lot"])) * config["lot"]
    if 0 < target < 1 and account.shares > 0 and abs(target - weight) < config["weight_band"]:
        target_shares = account.shares
    elif target == 1 and account.shares > 0 and abs(target - weight) < config["weight_band"]:
        target_shares = account.shares
    quantity = target_shares - account.shares
    return {"requested_quantity": quantity, "reference_weight": target,
            "action": "调整到中文方案所定仓位" if quantity else "维持当前份额"}


def simulate(data: pd.DataFrame, dividends: pd.DataFrame, config: dict, cost: dict, start: str,
             model_id: str, targets: np.ndarray | None = None, prediction: np.ndarray | None = None,
             horizon: int = 1, rebalance: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    open_prices, close_prices = data.open.to_numpy(float), data.close.to_numpy(float)
    previous_close, ex_amount = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
    variance = data.variance60.to_numpy(float)
    record_events, ex_events, pay_events = {}, {}, {}
    for k, event in enumerate(dividends.itertuples()):
        for field, mapping in (("record_date", record_events), ("ex_date", ex_events)):
            event_date = getattr(event, field)
            if event_date in dates:
                mapping.setdefault(dates.get_loc(event_date), []).append((k, event.cash_dividend_per_share))
        pay_index = dates.searchsorted(event.payment_date)
        if pay_index < len(dates):
            pay_events.setdefault(pay_index, []).append((k, event.payment_date == dates[pay_index]))
    account = Account(config["initial_capital"])
    records, decisions = [], []
    previous_nav = config["initial_capital"]
    previous_mark = close_prices[anchor]
    reference_target = 0.0

    def decision_at(t: int) -> dict:
        nonlocal reference_target
        price = close_prices[t]
        scheduled = (t - anchor) % rebalance == 0
        if model_id == "BUY_HOLD":
            if t == anchor:
                request = affordable_quantity(account.cash, fill_price(price, 1, cost, config["tick"]), cost, config["lot"])
                result = {"requested_quantity": request, "reference_weight": 1.0, "action": "初始买入持有"}
            else:
                result = {"requested_quantity": 0, "reference_weight": 1.0, "action": "长期持有"}
        elif scheduled and prediction is not None:
            result = choose_order(account, price, float(prediction[t]), horizon * variance[t], cost, config)
        elif scheduled:
            require(targets is not None, "缺少目标")
            result = target_request(account, price, float(targets[t]), config)
        else:
            result = {"requested_quantity": 0, "reference_weight": reference_target, "action": "非调整日维持份额"}
        reference_target = float(result["reference_weight"])
        decisions.append({"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.NaT,
                          "origin_index": t, "simulation_only": True, **result})
        return result

    pending = decision_at(anchor)
    for day in range(first, last + 1):
        date, op, cl = dates[day], open_prices[day], close_prices[day]
        old_shares, recognized, paid = account.shares, 0.0, 0.0
        for k, amount in ex_events.get(day, []):
            value = account.entitlements.get(k, 0) * amount
            account.receivables[k] = value
            recognized += value
        for k, same_day in pay_events.get(day, []):
            if not same_day:
                value = account.receivables.pop(k, 0.0)
                paid += value
                account.cash += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        pretrade_nav = account.value(op)
        execution = execute_order(account, quantity, op, previous_close[day], ex_amount[day], day, cost, config)
        mark = op if terminal else cl
        if not terminal:
            for k, same_day in pay_events.get(day, []):
                if same_day:
                    value = account.receivables.pop(k, 0.0)
                    paid += value
                    account.cash += value
            for k, amount in record_events.get(day, []):
                account.entitlements[k] = account.shares
        nav = account.value(mark)
        price_pnl = old_shares * (op - previous_mark) + account.shares * (mark - op)
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, f"{model_id} 在 {date} 财富恒等式不成立：{error}")
        account.assert_valid()
        records.append({"date": date, "open": op, "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(),
                        "equity": nav, "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav,
                        "price_pnl": price_pnl, "dividend_recognized": recognized, "dividend_paid": paid,
                        "exposure": account.shares * mark / nav, "accounting_error": error,
                        "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / pretrade_nav, **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decision_at(day)
    return pd.DataFrame(records), pd.DataFrame(decisions)


def adaptive_targets(returns: np.ndarray, component_targets: np.ndarray, dates: pd.DatetimeIndex,
                     origin_indices: np.ndarray, model_ids: list[str], window: int | None, top: int,
                     annual_days: int = 242) -> tuple[np.ndarray, pd.DataFrame]:
    target = np.zeros(len(dates))
    selected: list[int] = []
    selections = []
    quarter = None
    for t in origin_indices:
        key = (dates[t].year, (dates[t].month - 1) // 3)
        if window is None:
            selected = list(range(len(model_ids)))
        elif key != quarter:
            selected = []
            if t - window + 1 >= 0:
                history = returns[t - window + 1:t + 1]
                valid = np.isfinite(history).all(axis=0)
                mean, std = np.mean(history, axis=0), np.std(history, axis=0, ddof=1)
                valid &= (mean > 0) & (std > 1e-15)
                score = np.divide(mean * np.sqrt(annual_days), std, out=np.full(len(model_ids), -np.inf), where=valid)
                selected = sorted(np.flatnonzero(valid).tolist(), key=lambda j: (-score[j], model_ids[j]))[:top]
            selections.append({"selection_origin": dates[t], "window": window,
                               "selected": "|".join(model_ids[j] for j in selected), "selected_count": len(selected),
                               "information_cutoff": dates[t]})
        quarter = key
        if selected:
            values = component_targets[t, selected]
            require(np.isfinite(values).all(), "组合引用的当日目标不完整")
            target[t] = float(np.mean(values))
    return target, pd.DataFrame(selections)


def summarize(ledger: pd.DataFrame, config: dict) -> dict:
    metrics = return_metrics(ledger.net_return.to_numpy(float), config["annual_days"])
    metrics.update({"trading_days": len(ledger), "mean_exposure": float(ledger.exposure.mean()),
                    "trade_count": int((ledger.filled_quantity != 0).sum()),
                    "commission": float(ledger.commission.sum()), "slippage_cost": float(ledger.slippage_cost.sum()),
                    "terminal_unliquidated": bool(ledger.terminal_unliquidated.iloc[-1]),
                    "max_accounting_error": float(ledger.accounting_error.abs().max())})
    return metrics


def save_account(folder: Path, model_id: str, ledger: pd.DataFrame, decisions: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(folder / f"{model_id}_ledger.parquet", index=False)
    decisions.to_parquet(folder / f"{model_id}_decisions.parquet", index=False)
    ledger.loc[ledger.requested_quantity != 0].to_csv(folder / f"{model_id}_trades.csv", index=False, encoding="utf-8-sig")


def run(expected: str) -> None:
    config = verify_freeze(expected)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "manifest_sha256": expected,
               "status": "RUNNING", "position_impact": 0}, exclusive=True)
    data, dividends, columns = load_data(config)
    data.to_parquet(OUT / "features.parquet", index=False)
    targets = rule_targets(data)
    pd.DataFrame({"date": data.date, **targets}).to_parquet(OUT / "rule_targets.parquet", index=False)
    predictions = train_predict(data, dividends, columns, config)
    model_specs = {item["id"]: item for item in config["models"]}
    ids = list(RULE_NAMES) + list(model_specs)
    dates = pd.DatetimeIndex(data.date)
    evaluation_anchor = int(np.flatnonzero(data.date >= pd.Timestamp(config["evaluation_start"]))[0]) - 1
    all_metrics, yearly, eras, uncertainty, eval_arrays = [], [], [], {}, {}
    for cost_id, cost in config["costs"].items():
        shadow_returns = np.full((len(data), len(ids)), np.nan)
        shadow_targets = np.full((len(data), len(ids)), np.nan)
        candidate_ledgers = {}
        for j, model_id in enumerate(ids + ["BUY_HOLD"]):
            h = model_specs.get(model_id, {}).get("horizon", 1)
            for kind, start in (("shadow", config["shadow_start"]), ("evaluation", config["evaluation_start"])):
                ledger, decisions = simulate(data, dividends, config, cost, start, model_id,
                                targets=targets.get(model_id), prediction=predictions.get(model_id), horizon=h, rebalance=h)
                save_account(OUT / kind / cost_id, model_id, ledger, decisions)
                if kind == "shadow" and model_id != "BUY_HOLD":
                    row_indices = dates.get_indexer(pd.to_datetime(ledger.date))
                    shadow_returns[row_indices, j] = ledger.net_return.to_numpy()
                    shadow_targets[decisions.origin_index.to_numpy(int), j] = decisions.reference_weight.to_numpy()
                elif kind == "evaluation":
                    candidate_ledgers[model_id] = ledger
            if (j + 1) % 6 == 0:
                print(f"{cost_id} 已完成底层账户 {j + 1}/{len(ids) + 1}", flush=True)
        for model_id, window, top in (("P1_PRIMARY_TOP3_504", 504, 3), ("P2_TOP1_504", 504, 1),
                                      ("P3_TOP3_252", 252, 3), ("P4_EQUAL_ALL", None, 24)):
            composite, selection = adaptive_targets(shadow_returns, shadow_targets, dates,
                          np.arange(evaluation_anchor, len(data) - 1), ids, window, top, config["annual_days"])
            selection.to_csv(OUT / f"{cost_id}_{model_id}_selection.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"date": dates, "target": composite}).to_parquet(OUT / f"{cost_id}_{model_id}_targets.parquet", index=False)
            ledger, decisions = simulate(data, dividends, config, cost, config["evaluation_start"], model_id,
                                          targets=composite, rebalance=5)
            save_account(OUT / "evaluation" / cost_id, model_id, ledger, decisions)
            candidate_ledgers[model_id] = ledger
        benchmark = candidate_ledgers["BUY_HOLD"]
        baseline = summarize(benchmark, config)
        for model_id, ledger in candidate_ledgers.items():
            metric = {"cost": cost_id, "model": model_id, **summarize(ledger, config)}
            metric["annualized_return_excess_vs_buy_hold"] = metric["annualized_return"] - baseline["annualized_return"]
            metric["meets_point_target"] = metric["net_sharpe"] is not None and metric["net_sharpe"] >= config["high_sharpe_target"]
            all_metrics.append(metric)
            for year, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": model_id, "year": int(year), **summarize(group, config)})
            for name, start, end in (("2020—2021", "2020-01-01", "2021-12-31"),
                                     ("2022—2023", "2022-01-01", "2023-12-31"),
                                     ("2024—终点", "2024-01-01", config["data_cutoff"])):
                group = ledger.loc[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": model_id, "era": name, **summarize(group, config)})
        primary = candidate_ledgers[config["primary"]]
        a, b = primary.net_return.to_numpy(), benchmark.net_return.to_numpy()
        rng = np.random.default_rng(config["random_seed"])
        sharpe, excess = [], []
        for _ in range(config["bootstrap_repetitions"]):
            ix = block_indices(rng, len(a), config["bootstrap_day_block"])
            value = return_metrics(a[ix], config["annual_days"])["net_sharpe"]
            sharpe.append(np.nan if value is None else value)
            excess.append(float((a[ix] - b[ix]).mean() * config["annual_days"]))
        uncertainty[cost_id] = {"primary_sharpe_95_interval": interval(sharpe),
                               "primary_annualized_arithmetic_excess_95_interval": interval(excess),
                               "multiple_search_adjusted": False, "previous_total_trial_count": "NOT_KNOWN"}
        eval_arrays[cost_id] = pd.DataFrame({"date": benchmark.date,
                                           **{k: v.net_return.to_numpy() for k, v in candidate_ledgers.items()}})
        eval_arrays[cost_id].to_parquet(OUT / f"{cost_id}_all_evaluation_returns.parquet", index=False)
        print(f"{cost_id} 的 29 个完整评价账户及分年、分段统计已完成", flush=True)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    write_json(OUT / "uncertainty.json", uncertainty)
    primary = metrics.loc[metrics.model == config["primary"]]
    best = metrics.loc[(metrics.cost == "BASE") & (metrics.model != "BUY_HOLD")].sort_values("net_sharpe", ascending=False).iloc[0]
    point_pass = bool(primary.loc[primary.cost == "BASE", "meets_point_target"].iloc[0])
    result = {"study_id": STUDY, "completed_at": now(), "manifest_sha256": expected,
              "status": "HISTORICAL_PRIMARY_POINT_TARGET_MET_VALIDATION_PENDING" if point_pass else "COMPLETED_PRIMARY_TARGET_NOT_MET",
              "primary": primary.to_dict("records"), "post_selected_best_base": best.to_dict(),
              "number_of_candidates": 28, "benchmark_count": 1, "trained_model_count": 12,
              "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY",
              "uncertainty": uncertainty, "position_impact": 0, "validated_live_strategy": "NONE"}
    write_json(OUT / "result.json", result, exclusive=True)
    write_report(metrics, pd.DataFrame(eras), uncertainty, config, result)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


def display_name(model_id: str) -> str:
    if model_id in RULE_NAMES:
        return RULE_NAMES[model_id]
    if model_id in META_NAMES:
        return META_NAMES[model_id]
    if model_id == "BUY_HOLD":
        return "买入持有"
    pieces = model_id.split("_")
    kind = {"RIDGE": "岭回归", "HGB": "浅层梯度提升", "ET": "极随机树"}[pieces[1]]
    return f"{kind}／{pieces[2][1:]}日／{'全部过去样本' if pieces[3] == 'EXPANDING' else '最近756日'}"


def write_report(metrics: pd.DataFrame, eras: pd.DataFrame, uncertainty: dict, config: dict, result: dict) -> None:
    primary = metrics.loc[(metrics.model == config["primary"]) & (metrics.cost == "BASE")].iloc[0]
    best = result["post_selected_best_base"]
    lines = ["# 510300 条件切换与滚动模型：第一轮实际结果", "",
             f"本轮完成 24 个底层策略、4 个组合与买入持有基准；基础和压力费用分别计算，共 58 个评价账户。统一评价为 2020 年 1 月 2 日至 2026 年 8 月 14 日开盘。",
             "", f"预先指定主方案夏普为 **{primary.net_sharpe:.4f}**，目标为 1.2；事后最好候选是“{display_name(best['model'])}”，夏普 **{best['net_sharpe']:.4f}**。事后最好者不替代主方案，也不构成独立验证。", "",
             "## 主方案及全部候选", "", "所有数字来自同一连续账户。年化超额为策略复合年化收益减去同期买入持有复合年化收益；负回撤表示从此前最高点的跌幅。2026 年仅计算至固定终点。", "",
             "| 策略 | 基础夏普 | 压力夏普 | 年化收益 | 年化超额 | 最大回撤 | 平均仓位 | 成交笔数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    order = [config["primary"], "BUY_HOLD"] + [x for x in metrics.model.unique() if x not in (config["primary"], "BUY_HOLD")]
    for model_id in order:
        row = metrics.loc[(metrics.model == model_id) & (metrics.cost == "BASE")].iloc[0]
        stress = metrics.loc[(metrics.model == model_id) & (metrics.cost == "STRESS")].iloc[0]
        fmt = lambda x: "无定义" if pd.isna(x) else f"{x:.4f}"
        lines.append(f"| {display_name(model_id)} | {fmt(row.net_sharpe)} | {fmt(stress.net_sharpe)} | {row.annualized_return:.2%} | {row.annualized_return_excess_vs_buy_hold:.2%} | {row.max_drawdown:.2%} | {row.mean_exposure:.2%} | {row.trade_count} |")
    lines.extend(["", "## 主方案的分阶段表现", "", "| 阶段 | 基础夏普 | 年化收益 | 最大回撤 |", "| --- | ---: | ---: | ---: |"])
    for row in eras.loc[(eras.model == config["primary"]) & (eras.cost == "BASE")].itertuples():
        lines.append(f"| {row.era} | {row.net_sharpe:.4f} | {row.annualized_return:.2%} | {row.max_drawdown:.2%} |")
    for cost_id, item in uncertainty.items():
        lines.extend(["", f"{cost_id}：主方案夏普的 20 日区块重抽样 95% 区间为 {item['primary_sharpe_95_interval']}；年化日均超额区间为 {item['primary_annualized_arithmetic_excess_95_interval']}。区间没有宣称覆盖历次研究的多重选择影响。"])
    lines.extend(["", "## 规则与材料", "", "所有因子、24 个底层方案、4 个组合的完整中文规则见 `docs/510300_ADAPTIVE_ALLOCATION_V1_PROTOCOL.md`。", "",
                  "每次季度训练均保存训练起点、标签实际退出日和训练数据哈希；评价账户逐日保存现金、持仓、应收分红、费用与财富恒等式误差。所有账户均保留现金等待日和期末退出成本。",
                  "", "已有历史被多轮研究，本轮只能证明规则在该段历史按时间推进后的实际读数，不能证明未来夏普稳定达到 1.2。是否达到数值目标与是否具有独立证据分别列示。",
                  "", f"本轮结论状态：{result['status']}。真实持仓和订单不受本研究影响。"])
    (OUT / "510300条件切换与滚动模型_实际结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行已冻结的中文条件策略研究")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    require(args.freeze != args.run, "必须选择冻结或运行之一")
    if args.freeze:
        freeze()
    else:
        require(bool(args.expected_manifest_sha256), "运行需要明确冻结清单哈希")
        run(args.expected_manifest_sha256)
