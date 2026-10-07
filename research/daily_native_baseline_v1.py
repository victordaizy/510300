"""第一批封卷修正与日线自有资格基准；旧结果只读，研究无实盘权限。"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import intraday_process_increment_v1 as old

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_daily_native_baseline_v1.json"


def configs() -> tuple[dict, dict, Path, Path]:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads((ROOT / cfg["parent_config"]).read_text(encoding="utf-8"))
    return cfg, parent, ROOT / cfg["output"], ROOT / cfg["parent_output"]


def daily_features(raw: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """只访问当日及之前日线；价格、金额资格互不冒充。"""
    p = raw.copy().sort_values("date").reset_index(drop=True)
    p["date"] = pd.to_datetime(p.date).dt.normalize()
    old.require(not p.date.duplicated().any(), "日线日期重复")
    cols = ["open", "high", "low", "close"]
    price_valid = (np.isfinite(p[cols]).all(axis=1) & p.low.gt(0)
                   & p.high.ge(p[cols].max(axis=1)) & p.low.le(p[cols].min(axis=1)))
    p["daily_price_valid"] = price_valid
    p.loc[~price_valid, cols] = np.nan
    p["daily_amount_valid"] = np.isfinite(p.amount) & p.amount.gt(0)
    p["dividend"] = p.date.map(dividends.set_index("ex_date").cash_dividend_per_share).fillna(0.0)
    p["previous_close"] = p.close.shift(1)
    p["daily_total_return"] = (p.close + p.dividend) / p.previous_close - 1
    p["daily_intraday_return"] = np.log(p.close / p.open)
    p["daily_range"] = (p.high - p.low) / p.open
    p["daily_close_location"] = ((p.close - p.low) / (p.high - p.low).replace(0, np.nan)).where(p.high.gt(p.low), .5)
    amount = p.amount.where(p.daily_amount_valid)
    p["log_amount"] = np.log(amount)
    p["log_amount_relative20"] = np.log(amount / amount.shift().rolling(20, min_periods=20).median())
    p["rv20"] = p.daily_total_return.rolling(20, min_periods=20).std(ddof=1)
    p["log_rv20"] = np.log(p.rv20.where(p.rv20.gt(0)))
    p["momentum20"] = np.log1p(p.daily_total_return).rolling(20, min_periods=20).sum()
    return p


def minute_fields(minutes: pd.DataFrame) -> pd.DataFrame:
    m = minutes.copy()
    m["trade_time"] = pd.to_datetime(m.trade_time)
    old.require(not m.trade_time.duplicated().any(), "分钟时标重复")
    m["date"] = m.trade_time.dt.normalize()
    m["clock"] = m.trade_time.dt.strftime("%H:%M")
    p = ["open", "high", "low", "close"]
    m["price_valid"] = np.isfinite(m[p]).all(axis=1) & m.low.gt(0) & m.high.ge(m[p].max(axis=1)) & m.low.le(m[p].min(axis=1))
    m["volume_valid"] = np.isfinite(m.vol) & m.vol.ge(0)
    m["amount_value_valid"] = np.isfinite(m.amount) & m.amount.ge(0)
    paired = m.volume_valid & m.amount_value_valid & m.vol.eq(0).eq(m.amount.eq(0))
    vwap = m.amount / m.vol.replace(0, np.nan)
    m["amount_bad"] = ~paired | (m.vol.gt(0) & ((vwap < m.low - .001 - 1e-12) | (vwap > m.high + .001 + 1e-12)))
    rounding = 2.0 / m.vol.replace(0, np.nan)
    m["strict_amount_bad"] = ~paired | (m.vol.gt(0) & ((vwap < m.low - rounding) | (vwap > m.high + rounding)))
    m["vwap"] = vwap
    m["execution_data_valid"] = m.price_valid & m.volume_valid & ~m.amount_bad
    return m


def input_qualification(panel: pd.DataFrame, process: pd.DataFrame, parent: dict) -> pd.DataFrame:
    f = panel[["date", "daily_price_valid", "daily_amount_valid"] + parent["daily_features"]].copy()
    own = f.daily_price_valid & f.daily_amount_valid & np.isfinite(f[parent["daily_features"]]).all(axis=1)
    f = f[["date"]]
    f["D_input_valid"] = own.to_numpy()
    context = process.set_index("date").reindex(f.date)
    for key in ("A", "B", "C"):
        ready = np.isfinite(context[parent["feature_groups"][key]]).all(axis=1).to_numpy(copy=True)
        if key in ("A", "B"):
            ready &= context.process_available.fillna(False).to_numpy(bool)
        if key == "B":
            ready &= context.B_amount_valid.fillna(False).to_numpy(bool)
        if key == "C":
            ready &= context.C_amount_valid.fillna(False).to_numpy(bool)
        f[key + "_input_valid"] = own.to_numpy() & ready
    f["D_common_input_valid"] = context.common_valid.fillna(False).to_numpy(bool)
    f["old_quality_state"] = context.quality_state.to_numpy()
    f["economic_asof"] = f.date + pd.Timedelta(hours=15, minutes=5)
    f["historical_received_at"] = pd.NaT
    f["receipt_state"] = "HISTORICAL_RECEIVE_TIME_NOT_PROVEN"
    return f


def reconcile_external(parent_dir: Path, output: Path, parent: dict) -> dict:
    process = pd.read_parquet(parent_dir / "04_逐日盘中过程.parquet")
    predictions = pd.read_parquet(parent_dir / "06_滚动预测.parquet")
    ledger = pd.read_parquet(parent_dir / "08_完整账户逐日账本.parquet")
    base = ledger.loc[ledger.capital.eq(200000) & ledger.cost.eq("BASE")]
    absent = base.loc[base.model.eq("D") & base.prediction_state.eq("NO_VIEW"), ["date"]].merge(process, on="date")
    absent["D_all_fields_finite"] = np.isfinite(absent[parent["daily_features"]]).all(axis=1)
    absent[["date", "quality_state", "regime", "D_all_fields_finite", "C_amount_valid", "process_available"]].to_csv(output / "01_原108日原因表.csv", index=False, encoding="utf-8-sig")
    wide = base.pivot(index="date", columns="model", values="equity")
    difference = wide.C - wide.D
    daily = pd.DataFrame({"date": wide.index, "D_equity": wide.D, "C_equity": wide.C, "cumulative_C_minus_D_cny": difference,
                          "daily_C_minus_D_pnl_cny": difference.diff().fillna(difference.iloc[0])}).reset_index(drop=True)
    daily.to_csv(output / "02_C相对D逐日人民币增量.csv", index=False, encoding="utf-8-sig")
    monthly = daily.groupby(daily.date.dt.to_period("M")).daily_C_minus_D_pnl_cny.sum().rename("C_minus_D_cny").reset_index()
    monthly.date = monthly.date.astype(str)
    monthly.to_csv(output / "03_C相对D逐月人民币增量.csv", index=False, encoding="utf-8-sig")
    fits = json.loads((parent_dir / "07_全部模型参数.json").read_text(encoding="utf-8"))
    models = {(pd.Timestamp(r["fit_date"]), r["model"]): r for r in fits}
    panel = process.set_index("date")
    maximum, count = 0.0, 0
    for row in predictions.loc[predictions.prediction_D.notna()].itertuples():
        for key in old.MODELS:
            actual = old.predict(models[(row.fit_date, key)], panel.loc[row.date], parent)
            maximum = max(maximum, abs(actual - getattr(row, "prediction_" + key)))
            count += 1
    old.require(maximum < 1e-12 and count == 1820, "原保存参数预测复算不一致")
    source = json.loads((ROOT / "reports/research/510300_pressure_recovery_v1/source_admission_20261002/summary.json").read_text(encoding="utf-8"))
    result = {"classification": "EXTERNAL_REVIEW_CLAIMS_LOCALLY_RECONCILED", "original_no_view_days": len(absent), "original_quality_causes": absent.quality_state.value_counts().to_dict(), "D_fields_finite_on_no_view_days": int(absent.D_all_fields_finite.sum()), "first_signal": predictions.loc[predictions.prediction_D.notna(), "date"].min(), "first_account_day": ledger.date.min(), "saved_prediction_values_recomputed": count, "maximum_prediction_error": maximum, "C_minus_D_final_cny": float(difference.iloc[-1]), "C_minus_D_202409_cny": float(monthly.loc[monthly.date.eq("2024-09"), "C_minus_D_cny"].iloc[0]), "C_minus_D_monthly": monthly.to_dict("records"), "orderbook_source_summary_path": "reports/research/510300_pressure_recovery_v1/source_admission_20261002/summary.json", "orderbook_source_summary_sha256": old.digest(ROOT / "reports/research/510300_pressure_recovery_v1/source_admission_20261002/summary.json"), "orderbook_full_raw_recheck": False, "orderbook_receipt_exists": bool(source), "original_results_changed": False, "new_fits": 0, "new_accounts": 0, "new_random_draws": 0}
    old.write_json(output / "review_reconciliation.json", result)
    return result


def amount_diagnostic(m: pd.DataFrame, prices: pd.DataFrame, output: Path) -> None:
    daily = m.groupby("date").agg(amount=("amount", "sum"), vol=("vol", "sum"))
    reference = prices.set_index("date").reindex(daily.index)
    distance = np.maximum(m.low * m.vol - m.amount, m.amount - m.high * m.vol).clip(lower=0)
    ratio = (m.vwap / ((m.high + m.low) / 2)).replace([np.inf, -np.inf], np.nan).dropna()
    result = {"unit_evidence": "元/份，分钟量额日聚合与原日线相符；没有全局100或1000倍误差的证据", "max_daily_amount_relative_difference": float(((daily.amount - reference.amount).abs() / reference.amount).max()), "max_daily_volume_relative_difference": float(((daily.vol - reference.volume).abs() / reference.volume).max()), "amount_integer_fraction": float(m.amount.mod(1).eq(0).mean()), "within_day_amount_decreases": int(m.groupby("date").amount.diff().lt(0).sum()), "vwap_to_mid_quantiles": {str(k): float(v) for k, v in ratio.quantile([.01, .5, .99]).items()}, "outside_ohlc_amount_exceeds_two_cny": int(distance.gt(2.0 + 1e-8).sum()), "strict_vwap_bad": int(m.strict_amount_bad.sum()), "more_than_one_tick_bad": int(m.amount_bad.sum()), "single_record_cause_identified": False, "causes_not_resolved": ["成交量与金额的细粒度时点对齐", "提供方逐字段舍入精度与原始计数映射"], "price_adjustment_evidence": "原未复权OHLC日聚合吻合，不支持全局复权混用；不证明逐分钟正确", "change_to_tolerance": False, "imputed_amount_rows": 0, "inference": "大部分微小越界与精度问题相容但不是原因证明；不采用成交量乘收盘价替代原金额"}
    old.write_json(output / "amount_diagnostic.json", result)


def freeze_inputs(cfg: dict, parent: dict, output: Path, parent_dir: Path) -> None:
    paths = [CONFIG, ROOT / cfg["parent_config"], ROOT / "docs/510300_DAILY_NATIVE_BASELINE_V1.md", ROOT / "research/daily_native_baseline_v1.py", ROOT / "scripts/run_510300_daily_native_baseline_v1.py", ROOT / "tests/test_daily_native_baseline_v1.py", ROOT / "research/intraday_process_increment_v1.py", ROOT / "research/intraday_overnight_increment_v1.py"]
    paths += [ROOT / v for v in parent["inputs"].values()]
    paths += list(parent_dir.glob("*"))
    paths += [output / name for name in ("daily_asof_panel.parquet", "input_qualification.parquet", "minute_execution_fields.parquet", "daily_prices_with_features.parquet", "dividends.parquet")]
    paths += list((output / "received").glob("*"))
    unique = list(dict.fromkeys(p for p in paths if p.is_file()))
    old.write_json(output / "freeze.json", {"frozen_at": old.now(), "new_native_labels_created": False, "new_native_fits": 0, "new_native_accounts": 0, "files": [{"path": str(p), "sha256": old.digest(p), "bytes": p.stat().st_size} for p in unique], "parent_result_tree_immutable": True})


def prepare(cfg: dict, parent: dict, output: Path, parent_dir: Path) -> None:
    original_freeze = json.loads((parent_dir / "freeze.json").read_text(encoding="utf-8"))
    for row in original_freeze["files"]:
        old.require(old.digest(Path(row["path"])) == row["sha256"], "父实验冻结身份变化：" + row["path"])
    output.mkdir(parents=True, exist_ok=False)
    received = output / "received"
    received.mkdir()
    for original in [Path(r"C:\Users\戴周阳\Downloads\510300_高夏普高收益_全部后续工作路线图.html"), Path(r"C:\Users\戴周阳\Downloads\510300_48项后续工作清单.xlsx"), Path(r"E:\CodexData\.codex\attachments\80925032-6345-47ab-8a89-eba58ccb9537\已粘贴的文本.txt")]:
        (received / original.name).write_bytes(original.read_bytes())
    old.write_json(output / "authority.json", {"user_reply": cfg["user_scope"], "scope": "第一批；未授权扩展为全部48项或新机制/采集/实盘", "received_sources_role": "外部复核与路线图，四项事实本地核实后采纳", "new_model_design": True, "research_only": True})
    reconcile_external(parent_dir, output, parent)
    raw = pd.read_parquet(ROOT / parent["inputs"]["prices"])
    dividends = old.normalize_dividends(pd.read_csv(ROOT / parent["inputs"]["dividends"]))
    prices = daily_features(raw, dividends)
    prices = prices.loc[prices.date.le(pd.Timestamp(parent["data_cutoff"]))].reset_index(drop=True)
    process = pd.read_parquet(parent_dir / "04_逐日盘中过程.parquet")
    panel = prices.loc[prices.date.between(cfg["training_origin_start"], cfg["origin_end"])].copy().reset_index(drop=True)
    old.require(pd.DatetimeIndex(panel.date).equals(pd.DatetimeIndex(process.date)), "原点日历变化")
    old.require(np.allclose(panel[cfg["daily_features"]], process[cfg["daily_features"]], equal_nan=True, atol=1e-12, rtol=0), "新日线特征没有逐值复现原定义")
    qualification = input_qualification(panel, process, parent)
    panel = panel.merge(qualification[["date", "D_input_valid", "economic_asof", "historical_received_at", "receipt_state"]], on="date", validate="one_to_one")
    panel["regime"] = np.where(panel.date.lt(parent["regime_change"]), "PRE_20260706", "POST_20260706")
    panel.to_parquet(output / "daily_asof_panel.parquet", index=False)
    panel.to_csv(output / "04_日线原点与时钟.csv", index=False, encoding="utf-8-sig")
    qualification.to_parquet(output / "input_qualification.parquet", index=False)
    qualification.to_csv(output / "05_各模型输入资格.csv", index=False, encoding="utf-8-sig")
    m = minute_fields(pd.read_parquet(ROOT / parent["inputs"]["minutes"]))
    m.loc[m.clock.isin(["09:30", "15:00"])].to_parquet(output / "minute_execution_fields.parquet", index=False)
    by_day = m.groupby("date").agg(rows=("clock", "size"), price_rows_valid=("price_valid", "sum"), volume_rows_valid=("volume_valid", "sum"), amount_value_rows_valid=("amount_value_valid", "sum"), amount_one_tick_bad=("amount_bad", "sum"), amount_strict_bad=("strict_amount_bad", "sum")).reset_index()
    by_day.to_csv(output / "06_分钟字段质量分离.csv", index=False, encoding="utf-8-sig")
    prices.to_parquet(output / "daily_prices_with_features.parquet", index=False)
    dividends.to_parquet(output / "dividends.parquet", index=False)
    amount_diagnostic(m, prices, output)
    catalog = pd.read_csv(parent_dir / "01_实际数据清单.csv")
    catalog["本批分级"] = "限定用途代理或历史参考"
    catalog.loc[catalog.用途.isin(["prices", "dividends", "calendar", "price_receipt", "dividend_coverage"]), "本批分级"] = "已核验的对应历史日线用途"
    catalog.loc[catalog.用途.str.contains("nbs"), "本批分级"] = "限定八窗口用途，不用于D-native"
    catalog["历史实时可得性"] = "不因文件存在补造历史接收时间"
    catalog.to_csv(output / "07_数据资产与用途.csv", index=False, encoding="utf-8-sig")
    old.write_json(output / "availability_contract_v2.json", {"layers": ["INPUT", "MODEL", "PREDICTION", "EXECUTION"], "D": {"fields": cfg["daily_features"], "depends_on_A_B_C": False, "current_label_required": False}, "A_B_C": "只拆输入资格，继续保留原过程定义与预热；无新独立模型拟合或账户", "execution": "依原09:30/15:00价格/量与量额一致性，现金和T+1；预测不因随后未成交而撤销", "states": ["INPUT_MISSING_OR_BAD", "TRAINING_NOT_MATURE", "PREDICTION_AVAILABLE", "NO_ENTRY_EDGE_BELOW_COST", "HOLD_TO_FIXED_EXIT", "REQUEST_NEXT_OPEN", "UNFILLED_EXECUTION_DATA_MISSING", "UNFILLED_NO_VOLUME", "UNFILLED_CAPACITY", "LOCAL_DATE_NOT_COVERED"], "historical_receive_proven": False})
    freeze_inputs(cfg, parent, output, parent_dir)
    print(json.dumps({"阶段": "第一批准备与新设计冻结完成", "原点": len(panel), "D自有输入合格": int(qualification.D_input_valid.sum()), "新拟合": 0}, ensure_ascii=False))


def native_forecasts(data: pd.DataFrame, cfg: dict, parent: dict) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """只用D字段和已成熟历史标签；当前标签未知也允许当前预测。"""
    rows, fits, member_frames = [], [], []
    model, last_fit_index, fit_date = None, None, pd.NaT
    model_cfg = copy.deepcopy(parent)
    model_cfg["model"].update(cfg["model"])
    for i, row in data.reset_index(drop=True).iterrows():
        train = data.loc[data.D_input_valid & data.date.lt(row.date) & data.exit_h2.le(row.date) & np.isfinite(data.return_h2)]
        result = {"date": row.date, "regime": row.regime, "D_input_valid": bool(row.D_input_valid), "input_state": "INPUT_AVAILABLE" if row.D_input_valid else "INPUT_MISSING_OR_BAD", "mature_training_rows": len(train), "model_state": "TRAINING_NOT_MATURE", "prediction_state": "NO_PREDICTION", "prediction_D": np.nan, "fit_date": pd.NaT, "last_training_label_exit": pd.NaT, "current_label_mature_at_decision": bool(pd.notna(row.exit_h2) and row.exit_h2 <= row.date), "economic_asof": row.date + pd.Timedelta(hours=15, minutes=5), "historical_received_at": pd.NaT, "receipt_state": "HISTORICAL_RECEIVE_TIME_NOT_PROVEN"}
        if not row.D_input_valid:
            result["model_state"] = "NOT_APPLIED_INPUT_INVALID"
            result["prediction_state"] = "INPUT_MISSING_OR_BAD"
        elif len(train) >= cfg["model"]["minimum_training_days"]:
            if model is None or i - last_fit_index >= cfg["model"]["refit_interval_trading_days"]:
                model = old.fit_ridge(train, cfg["daily_features"], model_cfg)
                old.require(model["last_training_label_exit"] <= row.date, "D-native训练用了未成熟标签")
                last_fit_index, fit_date = i, row.date
                fits.append({"fit_date": fit_date, "model": "D_NATIVE", "regime_policy": cfg["model"]["regime_policy"], **model})
                membership = train[["date", "exit_h2"]].copy()
                membership.insert(0, "fit_date", fit_date)
                member_frames.append(membership)
            result.update(model_state="MODEL_AVAILABLE", prediction_state="PREDICTION_AVAILABLE", prediction_D=old.predict(model, row, model_cfg), fit_date=fit_date, last_training_label_exit=model["last_training_label_exit"])
        rows.append(result)
    members = pd.concat(member_frames, ignore_index=True) if member_frames else pd.DataFrame(columns=["fit_date", "date", "exit_h2"])
    return pd.DataFrame(rows), fits, members


def execution_state(row: pd.Series | None) -> str:
    if row is None:
        return "EXECUTION_RECORD_MISSING"
    if not bool(row.price_valid):
        return "EXECUTION_PRICE_INVALID"
    if not bool(row.volume_valid):
        return "EXECUTION_VOLUME_INVALID"
    if bool(row.amount_bad):
        return "EXECUTION_AMOUNT_CONSISTENCY_FAILED"
    if row.vol <= 0:
        return "EXECUTION_NO_VOLUME"
    return "CONDITIONAL_EXECUTION_PROXY_AVAILABLE"


def runtime_layers(qualification: pd.DataFrame, forecasts: pd.DataFrame, minutes: pd.DataFrame,
                   parent_forecasts: pd.DataFrame) -> pd.DataFrame:
    output = qualification.merge(forecasts.drop(columns=["D_input_valid", "economic_asof", "historical_received_at", "receipt_state"]), on="date", validate="one_to_one")
    records = minutes.set_index(["date", "clock"])
    for clock, name in (("09:30", "open"), ("15:00", "close")):
        output[name + "_execution_data_state"] = [execution_state(records.loc[(date, clock)] if (date, clock) in records.index else None) for date in output.date]
    output["next_open_execution_state_at_decision"] = "UNKNOWN_UNTIL_NEXT_SESSION"
    output["execution_regime"] = np.where(output.date.lt("2026-07-06"), "PRE_CONTINUOUS_CLOSE_PROXY", "POST_CLOSING_AUCTION_PROXY")
    old_map = parent_forecasts.set_index("date").prediction_D.notna()
    output["D_common_prediction_available"] = output.date.map(old_map).fillna(False)
    for key in ("A", "B", "C"):
        output[key + "_native_model_state"] = "NOT_RUN_FROZEN_NEGATIVE_EXPERIMENT"
        output[key + "_native_prediction_state"] = "NOT_RUN"
    return output


def status_for_date(runtime: pd.DataFrame, date: pd.Timestamp) -> dict:
    date = pd.Timestamp(date).normalize()
    rows = runtime.loc[runtime.date.eq(date)]
    if rows.empty:
        return {"requested_date": date, "data_last_date": runtime.date.max(), "input_state": "LOCAL_DATE_NOT_COVERED", "model_state": "NOT_APPLIED_NO_LOCAL_DATE", "prediction_state": "NO_VIEW_NO_CURRENT_LOCAL_RECEIPT", "prediction": None, "execution_state": "NOT_ASSESSED", "actual_position": "UNKNOWN", "new_collection": False, "historical_signal_carried_as_current": False}
    row = rows.iloc[0]
    return {"requested_date": date, "data_last_date": runtime.date.max(), "input_state": row.input_state, "model_state": row.model_state, "prediction_state": row.prediction_state, "prediction": row.prediction_D, "fit_date": row.fit_date, "economic_asof": row.economic_asof, "receipt_state": row.receipt_state, "execution_open_state_observed_on_date": row.open_execution_data_state, "execution_close_state_observed_on_date": row.close_execution_data_state, "next_open_execution_state_at_decision": row.next_open_execution_state_at_decision, "actual_position": "UNKNOWN", "scope": "SAVED_HISTORICAL_REPLAY_NOT_CURRENT_ORDER"}


def verify_accounts(ledger: pd.DataFrame, metrics: pd.DataFrame, cycles: pd.DataFrame, cfg: dict) -> dict:
    error = 0.0
    for key, part in ledger.groupby(["capital", "cost", "model"]):
        row = metrics.loc[metrics.capital.eq(key[0]) & metrics.cost.eq(key[1]) & metrics.model.eq(key[2])].iloc[0]
        nav = part.cash + part.shares * part.close + part.receivable
        old.require(np.allclose(nav, part.equity, atol=1e-7, rtol=0), "新账户现金库存应收不守恒")
        returns = nav.to_numpy() / np.r_[key[0], nav.to_numpy()[:-1]] - 1
        old.require(np.allclose(returns, part.net_return, atol=1e-12, rtol=0), "新账户收益复算不一致")
        actual = old.return_metrics(returns, cfg["metric_contract"]["primary_annual_days"])
        for field in ("annualized_return", "net_sharpe", "max_drawdown"):
            difference = abs(float(actual[field]) - float(row[field]))
            old.require(difference < 1e-10, "新账户保存指标不一致")
            error = max(error, difference)
    if len(cycles):
        old.require((pd.to_datetime(cycles.exit_date) > pd.to_datetime(cycles.entry_date)).all(), "新周期违反T+1")
        old.require(np.allclose(cycles.gross_quote_pnl - cycles.commission - cycles.slippage, cycles.net_pnl, atol=1e-6), "新周期净费用不守恒")
    return {"status": "PASS_NATIVE_SAVED_ACCOUNT_RECOMPUTATION", "accounts": len(metrics), "ledger_rows": len(ledger), "cycles": len(cycles), "maximum_metric_error": error, "maximum_accounting_error": float(ledger.accounting_error.abs().max()), "actual_fills_verified": 0}


def run_native(cfg: dict, parent: dict, output: Path, parent_dir: Path) -> dict:
    old.require(not (output / "run_claim.json").exists(), "本版已经执行认领，禁止重复运行或覆盖")
    frozen = json.loads((output / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        old.require(old.digest(Path(item["path"])) == item["sha256"], "冻结文件变化：" + item["path"])
    old.write_json(output / "run_claim.json", {"claimed_at": old.now(), "status": "RUNNING_ONE_FIXED_NATIVE_DESIGN"})
    panel = pd.read_parquet(output / "daily_asof_panel.parquet")
    prices = pd.read_parquet(output / "daily_prices_with_features.parquet")
    dividends = pd.read_parquet(output / "dividends.parquet")
    minutes = pd.read_parquet(output / "minute_execution_fields.parquet")
    qualified = pd.read_parquet(output / "input_qualification.parquet")
    labeled = old.add_labels(panel, prices, dividends)
    forecasts, fits, members = native_forecasts(labeled, cfg, parent)
    forecasts.to_parquet(output / "native_forecasts.parquet", index=False)
    forecasts.to_csv(output / "08_D_native全部原点.csv", index=False, encoding="utf-8-sig")
    old.write_json(output / "native_saved_models.json", fits)
    members.to_csv(output / "09_D_native全部训练成员.csv", index=False, encoding="utf-8-sig")
    native_eval = forecasts.loc[forecasts.date.between(cfg["account_first_signal"], cfg["account_end"])].copy()
    old.require(native_eval.date.iloc[0] == pd.Timestamp(cfg["account_first_signal"]) and pd.notna(native_eval.prediction_D.iloc[0]), "指定首信号不能形成D-native预测，禁止改变评价起点")
    ledgers, orders, decisions, cycles, metric_rows = [], [], [], [], []
    for capital in cfg["account"]["capitals"]:
        for cost in cfg["account"]["costs"]:
            ledger, requests, decision, completed = old.simulate_account(prices, dividends, minutes, native_eval, capital, cost, "D", parent)
            old.require(ledger.date.min() == pd.Timestamp(cfg["account_start"]) and ledger.date.max() == pd.Timestamp(cfg["account_end"]), "账户完整评价日历变化")
            for frame in (ledger, requests, decision, completed):
                frame["capital"], frame["cost"], frame["model"] = capital, cost, "D_NATIVE"
            requests["execution_regime"] = np.where(requests.date.lt("2026-07-06"), "PRE_CONTINUOUS_CLOSE_PROXY", "POST_CLOSING_AUCTION_PROXY")
            metric_rows.append(old.account_metrics(ledger, requests, completed, parent))
            ledgers.append(ledger); orders.append(requests); decisions.append(decision); cycles.append(completed)
    ledger = pd.concat(ledgers, ignore_index=True)
    orders_all = pd.concat(orders, ignore_index=True)
    decisions_all = pd.concat(decisions, ignore_index=True)
    cycles_all = pd.concat(cycles, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    for frame, name in ((ledger, "native_accounts"), (orders_all, "native_orders"), (decisions_all, "native_decisions"), (cycles_all, "native_cycles")):
        frame.to_parquet(output / (name + ".parquet"), index=False)
        frame.to_csv(output / (name + ".csv"), index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "10_D_native完整账户指标.csv", index=False, encoding="utf-8-sig")
    original = pd.read_csv(parent_dir / "12_全部账户指标.csv")
    common = original.loc[original.model.eq("D")].copy()
    common["model"] = "D_COMMON"
    pd.concat([common, metrics], ignore_index=True).to_csv(output / "11_D_common与native完整账户.csv", index=False, encoding="utf-8-sig")
    parent_predictions = pd.read_parquet(parent_dir / "06_滚动预测.parquet")
    runtime = runtime_layers(qualified, forecasts, minutes, parent_predictions)
    runtime.to_parquet(output / "runtime_states.parquet", index=False)
    runtime.to_csv(output / "12_每日四层运行状态.csv", index=False, encoding="utf-8-sig")
    states = decisions_all.merge(runtime[["date", "input_state", "model_state", "prediction_state", "open_execution_data_state", "close_execution_data_state", "next_open_execution_state_at_decision"]], on="date", how="left", validate="many_to_one")
    states.to_csv(output / "13_完整账户行为与资格.csv", index=False, encoding="utf-8-sig")
    comparison = parent_predictions.loc[parent_predictions.prediction_D.notna(), ["date", "prediction_D", "return_h2"]].merge(forecasts[["date", "prediction_D"]], on="date", suffixes=("_common", "_native"), validate="one_to_one")
    old.require(len(comparison) == 364 and comparison.prediction_D_native.notna().all(), "原共同预测成员有变化")
    comparison["squared_error_common"] = (comparison.prediction_D_common - comparison.return_h2) ** 2
    comparison["squared_error_native"] = (comparison.prediction_D_native - comparison.return_h2) ** 2
    comparison.to_csv(output / "14_原共同预测日期对照.csv", index=False, encoding="utf-8-sig")
    periods = []
    for name, start, end in (("完整原评价", cfg["account_start"], cfg["account_end"]), ("旧C连带缺口", "2026-03-09", "2026-07-03"), ("新收盘制度", "2026-07-06", cfg["account_end"])):
        for (capital, cost), group in ledger.groupby(["capital", "cost"]):
            part = group.loc[group.date.between(start, end)]
            periods.append({"period": name, "capital": capital, "cost": cost, "days": len(part), "cumulative_return": float(np.prod(1 + part.net_return) - 1), "net_pnl_cny": float((part.price_pnl + part.dividend_recognized - part.commission - part.slippage).sum()), "mean_exposure": float(part.exposure.mean()), "interpretation": "预登记区段描述，不用于挑时期或晋升"})
    pd.DataFrame(periods).to_csv(output / "15_完整期与缺口区段.csv", index=False, encoding="utf-8-sig")
    bridge = []
    old_ledger = pd.read_parquet(parent_dir / "08_完整账户逐日账本.parquet")
    old_ledger = old_ledger.loc[old_ledger.model.eq("D")].copy()
    old_ledger["model"] = "D_COMMON"
    for key, group in pd.concat([old_ledger, ledger]).groupby(["capital", "cost", "model"]):
        for days in (242, 252):
            bridge.append({"capital": key[0], "cost": key[1], "model": key[2], "annual_days": days, "role": "PRIMARY" if days == 242 else "BRIDGE_ONLY_NOT_SELECTION", **old.return_metrics(group.net_return.to_numpy(), days)})
    pd.DataFrame(bridge).to_csv(output / "16_242与252口径桥接.csv", index=False, encoding="utf-8-sig")
    verification = verify_accounts(ledger, metrics, cycles_all, cfg)
    old.write_json(output / "saved_recomputation.json", verification)
    summary = {"study_id": cfg["study_id"], "status": "COMPLETED_FIRST_BATCH_NATIVE_BASELINE_DIAGNOSTIC", "original_results_changed": False, "original_origin_days": len(panel), "native_input_days": int(qualified.D_input_valid.sum()), "common_input_days": int(qualified.D_common_input_valid.sum()), "native_forecast_days_all_origins": int(forecasts.prediction_D.notna().sum()), "native_eval_forecasts_including_first_signal": int(native_eval.prediction_D.notna().sum()), "native_model_fits": len(fits), "native_account_scenarios": len(metrics), "account_days": int(metrics.account_days.iloc[0]), "native_no_view_account_days": metrics.no_view_days.tolist(), "paired_original_forecast_days": len(comparison), "paired_mse_common": float(comparison.squared_error_common.mean()), "paired_mse_native": float(comparison.squared_error_native.mean()), "paired_mse_reduction": float((comparison.squared_error_common - comparison.squared_error_native).mean()), "primary_annual_days": 242, "new_bootstrap_draws": 0, "prediction_alpha_confirmed": False, "independent_validation": "NOT_ESTABLISHED_PREVIOUSLY_EXPOSED_HISTORY", "strategy_promoted": False, "actual_trading_authorized": False, "actual_fills_verified": 0, "new_market_downloads": 0, "next_batch_started": False, "completed_at": old.now()}
    old.write_json(output / "summary.json", summary)
    old.write_json(output / "local_current_status_example.json", status_for_date(runtime, pd.Timestamp("2026-10-02")))
    old.write_json(output / "run_claim.json", {"status": summary["status"], "completed_at": old.now(), "new_fits": len(fits), "new_accounts": len(metrics), "single_configuration": True})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="第一批：封卷修正与D-native运行资格诊断")
    parser.add_argument("--stage", required=True, choices=["prepare", "run", "status"])
    parser.add_argument("--date", help="status必须明确请求日期，禁止默认使用旧日期冒充当前")
    args = parser.parse_args()
    cfg, parent, output, parent_dir = configs()
    if args.stage == "prepare":
        prepare(cfg, parent, output, parent_dir)
    elif args.stage == "run":
        print(json.dumps(old.clean(run_native(cfg, parent, output, parent_dir)), ensure_ascii=False))
    else:
        old.require(bool(args.date), "status需要明确--date")
        result = status_for_date(pd.read_parquet(output / "runtime_states.parquet"), pd.Timestamp(args.date))
        print(json.dumps(old.clean(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
