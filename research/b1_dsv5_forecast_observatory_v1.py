"""B0/B1 严格前向风险预测观察器，输出合同不含仓位或账户字段。"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from research import constituent_fragility_dsv5_increment_v1 as parent
from research import stk_mins_source_admission_v1 as source
from research.nbs_v2_common import (ContractError, TZ, clean, csv_bytes, encoded, identity,
                                   load_json, now, sha, verify, write_once)

CONFIG = "config/510300_b1_dsv5_forecast_observatory_v1.yaml"
MANIFEST = "config/510300_b1_dsv5_forecast_observatory_v1_manifest.json"
BASE = "data/forward/510300_b1_dsv5_forecast_observatory_v1"
FIELDS = ["origin_date", "model_vintage_id", "training_end_date", "b0_predicted_dsv5",
          "b1_predicted_dsv5", "q_b1_div_b0", "input_sha256", "prediction_generated_at",
          "label_maturity_date", "matured_actual_dsv5", "b0_qlike", "b1_qlike",
          "b1_minus_b0_loss_improvement"]
FORBIDDEN = {"target_weight", "target_shares", "buy_or_sell", "order_intent", "portfolio_return"}


def cfg(root: Path) -> dict:
    return yaml.safe_load((root / CONFIG).read_text("utf-8"))


def calendar(root: Path, config: dict) -> pd.DatetimeIndex:
    a = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(root / config["inputs"]["calendar_historical"],
                                                   usecols=["trade_date"]).trade_date))
    b = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(root / config["inputs"]["calendar_2026"],
                                                   usecols=["trade_date"]).trade_date))
    if a.has_duplicates or b.has_duplicates or not a.is_monotonic_increasing or not b.is_monotonic_increasing:
        raise ContractError("交易日历重复或不递增")
    if not a[(a >= b.min()) & (a <= b.max())].equals(b[(b >= a.min()) & (b <= a.max())]):
        raise ContractError("历史及官方日历重叠不一致")
    return a.union(b).sort_values()


def origins(cal: pd.DatetimeIndex, config: dict) -> pd.DatetimeIndex:
    anchor = pd.Timestamp(config["schedule"]["anchor_date"])
    if anchor not in cal:
        raise ContractError("冻结网格锚点不在日历中")
    result = cal[cal.get_loc(anchor)::5]
    return result[result >= pd.Timestamp(config["schedule"]["first_new_origin_date"])]


def register(root: Path) -> dict:
    config = cfg(root)
    prior_manifest = load_json(root / config["inputs"]["parent_manifest"])
    prior_receipt = load_json(root / config["inputs"]["parent_receipt"])
    prior_result = load_json(root / config["inputs"]["parent_result"])
    selected = {config["inputs"][k] for k in ["seed_daily", "seed_dividends", "seed_labels",
                                             "parent_protocol", "parent_code", "parent_result"]}
    frozen_ids = [entry for entry in prior_manifest["implementation_files"] + prior_manifest["input_metadata"]
                  + prior_receipt["outputs"] if entry["path"] in selected]
    if {entry["path"] for entry in frozen_ids} != selected:
        raise ContractError("父B1冻结身份不完整")
    verify(root, frozen_ids)
    if not prior_result["gates"]["G2_B1_VS_B0"]["passed"]:
        raise ContractError("父B1未通过预测门，不能登记为保留基准")
    paths = [CONFIG, "research/b1_dsv5_forecast_observatory_v1.py", "research/nbs_v2_common.py",
             "scripts/run_510300_b1_dsv5_forecast_observatory_v1.py",
             "tests/test_510300_b1_dsv5_forecast_observatory_v1.py"]
    # 父输入留存独立字节副本，今后的官方增量更新不会覆盖观察器训练种子。
    seed_mapping = {}
    for name, path in config["inputs"].items():
        if name in {"seed_daily", "seed_dividends", "seed_dividend_coverage", "seed_labels"}:
            destination = f"{BASE}/seed/{Path(path).name}"
            write_once(root / destination, (root / path).read_bytes())
            seed_mapping[name] = destination
            paths.append(destination)
        else:
            seed_mapping[name] = path
            paths.append(path)
    cal = calendar(root, config)
    grid = origins(cal, config)
    if str(grid[0].date()) != config["schedule"]["first_new_origin_date"]:
        raise ContractError("首个前向原点与冻结网格不符")
    report = {"model_id": config["model_id"], "registered_at": now(),
              "state": "REGISTERED_FORECAST_ONLY_WAITING_FIRST_ORIGIN", "seed_mapping": seed_mapping,
              "identities": [identity(root, path) for path in sorted(set(paths))],
              "first_new_origin_date": str(grid[0].date()), "calendar_coverage_end": str(cal.max().date()),
              "initial_forward_predictions": 0, "position_impact": 0, "live_trading_authorized": False}
    write_once(root / MANIFEST, report)
    write_once(root / BASE / "registration.json", {k: v for k, v in report.items() if k != "identities"})
    return {k: v for k, v in report.items() if k not in {"identities", "seed_mapping"}}


def check_prediction_record(record: dict) -> None:
    if set(record) != set(FIELDS) or FORBIDDEN.intersection(record):
        raise ContractError("观察器记录列不符合仅预测合同")
    for name in ["b0_predicted_dsv5", "b1_predicted_dsv5", "q_b1_div_b0"]:
        if not np.isfinite(record[name]) or record[name] <= 0:
            raise ContractError("预测必须是有限正数")
    generated = datetime.fromisoformat(record["prediction_generated_at"]).astimezone(TZ)
    if generated.date().isoformat() != record["origin_date"] or generated.time() < time(19, 30):
        raise ContractError("预测生成时钟不属于原点日19:30后的合法窗口")
    if pd.Timestamp(record["training_end_date"]) > pd.Timestamp(record["origin_date"]):
        raise ContractError("训练标签尚未成熟")
    if pd.Timestamp(record["label_maturity_date"]) <= pd.Timestamp(record["origin_date"]):
        raise ContractError("新预测标签成熟日必须晚于原点")


def compare_history(old: pd.DataFrame, new: pd.DataFrame) -> None:
    joined = old.set_index("date")[["open", "close"]].join(
        new.set_index("date")[["open", "close"]], lsuffix="_old", rsuffix="_new", how="inner")
    if joined.empty:
        raise ContractError("NO_VIEW_NO_OVERLAPPING_DAILY_SOURCE")
    for col in ["open", "close"]:
        if not np.allclose(joined[f"{col}_old"], joined[f"{col}_new"], atol=1e-12, rtol=0):
            raise ContractError("NO_VIEW_INPUT_HISTORY_DRIFT")


def compare_dividends(old: pd.DataFrame, new: pd.DataFrame, covered_through: pd.Timestamp) -> None:
    names = ["record_date", "ex_date", "payment_date", "cash_dividend_per_share"]
    def canonical(frame):
        part = frame.loc[pd.to_datetime(frame.ex_date) <= covered_through, names].copy()
        for column in names[:3]:
            part[column] = pd.to_datetime(part[column]).dt.strftime("%Y-%m-%d")
        part["cash_dividend_per_share"] = part.cash_dividend_per_share.astype(float)
        return part.sort_values("ex_date").reset_index(drop=True)
    if not canonical(old).equals(canonical(new)):
        raise ContractError("NO_VIEW_DIVIDEND_HISTORY_DRIFT")


def official_dividends(root: Path, config: dict, day: pd.Timestamp) -> tuple[pd.DataFrame, list[dict]]:
    receipt_path = config["live_input"]["dividend_coverage_receipt"]
    receipt = load_json(root / receipt_path)
    if not receipt.get("complete_history_confirmed") or pd.Timestamp(receipt["coverage_end"]) < day:
        raise ContractError("NO_VIEW_STALE_OFFICIAL_DIVIDEND_COVERAGE")
    data_path = config["live_input"]["dividend_ledger"]
    if sha(root / data_path) != receipt["distribution_file_sha256"]:
        raise ContractError("NO_VIEW_DIVIDEND_RECEIPT_HASH_MISMATCH")
    provenance = [identity(root, data_path), identity(root, receipt_path)]
    for item in receipt["official_source_snapshots"]:
        if sha(root / item["saved_file"]) != item["sha256"]:
            raise ContractError("NO_VIEW_OFFICIAL_DIVIDEND_SOURCE_HASH_MISMATCH")
        provenance.append(identity(root, item["saved_file"]))
    if not receipt["official_source_snapshots"]:
        raise ContractError("NO_VIEW_MISSING_OFFICIAL_DIVIDEND_SOURCES")
    result = parent.prepare_dividends(pd.read_csv(root / data_path), symbol="510300.SH")
    return result, provenance


def acquire_daily(root: Path, config: dict, day: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """只在合法原点调用冻结 fund_daily 来源，保存到观察器独立目录。"""
    from dotenv import load_dotenv
    source_config = yaml.safe_load((root / config["inputs"]["source_config"]).read_text("utf-8"))
    base = root / BASE / "raw_daily" / str(day.date())
    for receipt_path in sorted(base.glob("retrieved_at=*/receipt.json"), reverse=True):
        receipt = load_json(receipt_path)
        retrieved = datetime.fromisoformat(receipt["retrieved_at"]).astimezone(TZ)
        if receipt["success"] and retrieved.date() == day.date() and retrieved.time() >= time(15):
            verify(root, [{"path": receipt["normalized_relative_path"], "sha256": receipt["normalized_sha256"]},
                          {"path": receipt["raw_relative_path"], "sha256": receipt["response_sha256"]}])
            return pd.read_parquet(root / receipt["normalized_relative_path"]), receipt
    load_dotenv(root / ".env", override=False)
    try:
        token = source.resolve_ephemeral_token(source_config["source_contract"]["token_environment_name"])
    except Exception as error:
        raise ContractError("NO_VIEW_MISSING_EPHEMERAL_DAILY_SOURCE_CREDENTIAL") from error
    client = source.build_client(source_config, token)
    # 含全部前向历史和固定重叠区间，既不覆盖父日线，也不生成遗漏原点预测。
    result = client.call("fund_daily", {"ts_code": "510300.SH", "start_date": "20260801",
                                        "end_date": day.strftime("%Y%m%d")})
    if not result.success:
        raise ContractError(f"NO_VIEW_DAILY_PROVIDER_FAILED_{result.http_status}_{result.business_code}")
    receipt = source.save_immutable_capture(root, Path(BASE) / "raw_daily" / str(day.date()), result,
                                           source_config["source_contract"]["endpoint"])
    return result.frame, receipt


def risk_input(root: Path, config: dict, manifest: dict, day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    snapshot_dir = root / BASE / "inputs" / str(day.date())
    if (snapshot_dir / "receipt.json").exists():
        existing = load_json(snapshot_dir / "receipt.json")
        snapshot_ids = [p for p in existing["provenance"] if p["path"] in {
            (snapshot_dir / "daily.csv").relative_to(root).as_posix(),
            (snapshot_dir / "dividends.csv").relative_to(root).as_posix()}]
        verify(root, snapshot_ids)
        if len(snapshot_ids) != 2:
            raise ContractError("NO_VIEW_INCOMPLETE_ORIGIN_INPUT_RECEIPT")
        daily = parent.prepare_etf_daily(pd.read_csv(snapshot_dir / "daily.csv"), symbol="510300.SH")
        div = parent.prepare_dividends(pd.read_csv(snapshot_dir / "dividends.csv"), symbol="510300.SH")
        return parent.build_b1_daily_features(daily, div, annualization_days=252), div, identity(root, snapshot_dir / "receipt.json")
    # 先检查官方分红时点覆盖；缺失时不给过期总收益序列生成预测。
    div, provenance = official_dividends(root, config, day)
    seed_div = parent.prepare_dividends(pd.read_csv(root / manifest["seed_mapping"]["seed_dividends"]), symbol="510300.SH")
    seed_coverage = load_json(root / manifest["seed_mapping"]["seed_dividend_coverage"])
    compare_dividends(seed_div, div, pd.Timestamp(seed_coverage["coverage_end"]))
    raw, receipt = acquire_daily(root, config, day)
    if {"trade_date", "ts_code"}.issubset(raw.columns):
        raw = raw.rename(columns={"trade_date": "date", "ts_code": "symbol"})
        raw["date"] = pd.to_datetime(raw.date.astype(str))
    recent = parent.prepare_etf_daily(raw, symbol="510300.SH")
    old = parent.prepare_etf_daily(pd.read_parquet(root / manifest["seed_mapping"]["seed_daily"]), symbol="510300.SH")
    compare_history(old, recent)
    if recent.date.max() != day or recent.date.duplicated().any():
        raise ContractError("NO_VIEW_MISSING_CURRENT_DAILY_CAPTURE")
    for snapshot in (root / BASE / "inputs").glob("*/daily.csv"):
        compare_history(parent.prepare_etf_daily(pd.read_csv(snapshot), symbol="510300.SH"), recent)
        previous_div = parent.prepare_dividends(pd.read_csv(snapshot.parent / "dividends.csv"), symbol="510300.SH")
        compare_dividends(previous_div, div, pd.Timestamp(snapshot.parent.name))
    daily = pd.concat([old, recent]).drop_duplicates("date", keep="first").sort_values("date").reset_index(drop=True)
    daily = daily.loc[daily.date <= day].copy()
    cal = calendar(root, config)
    if not pd.DatetimeIndex(daily.date).equals(cal[(cal >= daily.date.min()) & (cal <= day)]):
        raise ContractError("NO_VIEW_DAILY_CALENDAR_GAP")
    write_once(snapshot_dir / "daily.csv", csv_bytes(daily))
    write_once(snapshot_dir / "dividends.csv", csv_bytes(div))
    provenance += [identity(root, receipt["raw_relative_path"]), identity(root, receipt["normalized_relative_path"]),
                   identity(root, snapshot_dir / "daily.csv"), identity(root, snapshot_dir / "dividends.csv")]
    payload = {"origin_date": str(day.date()), "captured_at": now(), "provenance": provenance}
    write_once(snapshot_dir / "receipt.json", payload)
    return parent.build_b1_daily_features(daily, div, annualization_days=252), div, identity(root, snapshot_dir / "receipt.json")


def matured_records(root: Path) -> list[dict]:
    result = []
    for path in sorted((root / BASE / "matured").glob("*.json")):
        receipt = load_json(root / BASE / "maturity_receipts" / path.name)
        verify(root, [receipt["matured_record"]])
        result.append(load_json(path))
    return result


def mature(root: Path, config: dict, daily: pd.DataFrame, dividends: pd.DataFrame, day: pd.Timestamp, input_id: dict) -> None:
    parent_cfg = yaml.safe_load((root / config["inputs"]["parent_protocol"]).read_text("utf-8"))
    for path in sorted((root / BASE / "predictions").glob("*.json")):
        prediction = load_json(path)
        output = root / BASE / "matured" / path.name
        if output.exists() or pd.Timestamp(prediction["label_maturity_date"] ) > day:
            continue
        prediction_receipt = load_json(root / BASE / "prediction_receipts" / path.name)
        verify(root, [prediction_receipt["prediction"], prediction_receipt["input"], prediction_receipt["model"]])
        origin = pd.Timestamp(prediction["origin_date"])
        schedule = pd.DataFrame([{"eligible": True, "origin_date": origin, "offset": 0,
                                  "sample_role": "STRICT_FORWARD", "era_id": "FORWARD"}])
        labels = parent.build_dsv5_label_ledger(schedule=schedule, b1_daily=daily,
                                               dividends=dividends, protocol=parent_cfg)
        if labels.iloc[0].horizon_end_date != pd.Timestamp(prediction["label_maturity_date"]):
            raise ContractError("NO_VIEW_FORWARD_LABEL_CLOCK_DRIFT")
        actual = float(labels.iloc[0].DSV5)
        a = parent.qlike_deviance(pd.Series([actual]), pd.Series([prediction["b0_predicted_dsv5"]]), epsilon=1e-8)[0]
        b = parent.qlike_deviance(pd.Series([actual]), pd.Series([prediction["b1_predicted_dsv5"]]), epsilon=1e-8)[0]
        record = {**prediction, "matured_actual_dsv5": actual, "b0_qlike": float(a), "b1_qlike": float(b),
                  "b1_minus_b0_loss_improvement": float(a-b)}
        check_prediction_record(record)
        write_once(output, record)
        write_once(root / BASE / "maturity_receipts" / path.name,
                   {"origin_date": str(origin.date()), "matured_record": identity(root, output),
                    "input": input_id, "scored_at": now()})


def fit_vintage(root: Path, config: dict, manifest: dict, daily: pd.DataFrame, slot: int, day: pd.Timestamp) -> dict:
    labels = pd.read_parquet(root / manifest["seed_mapping"]["seed_labels"],
                             columns=["offset", "origin_date", "horizon_end_date", "DSV5"])
    labels = labels.loc[labels.offset.eq(0), ["origin_date", "horizon_end_date", "DSV5"]].copy().reset_index(drop=True)
    labels["origin_date"] = pd.to_datetime(labels.origin_date)
    labels["horizon_end_date"] = pd.to_datetime(labels.horizon_end_date)
    for record in matured_records(root):
        labels.loc[len(labels)] = [pd.Timestamp(record["origin_date"]), pd.Timestamp(record["label_maturity_date"]),
                                   record["matured_actual_dsv5"]]
    labels = labels.loc[labels.horizon_end_date <= day].sort_values("origin_date")
    if labels.origin_date.duplicated().any():
        raise ContractError("NO_VIEW_DUPLICATED_FORWARD_TRAINING_ORIGIN")
    train = labels.merge(daily[["date", *parent.B1_FEATURES]], left_on="origin_date", right_on="date", validate="one_to_one")
    if len(train) != len(labels) or len(train) < 80:
        raise ContractError("NO_VIEW_INCOMPLETE_FROZEN_B1_TRAINING")
    model = parent.fit_nonnegative_qlike_ridge(train, feature_names=parent.B1_FEATURES, target_name="DSV5",
                                               l2_lambda=1.0, epsilon=1e-8)
    return {"model_vintage_id": f"B1_DSV5_FORWARD_REFIT_SLOT_{slot:04d}_{day:%Y%m%d}",
            "created_at": now(), "training_end_date": str(train.horizon_end_date.max().date()),
            "training_count": len(train), "training_sha256": hashlib.sha256(csv_bytes(train)).hexdigest(),
            "intercept": model.intercept, "coefficients": model.coefficients.tolist(),
            "means": model.means.tolist(), "scales": model.scales.tolist(),
            "b0": max(float(train.DSV5.mean()), 1e-8), "features": list(parent.B1_FEATURES),
            "objective_value": model.objective_value, "iterations": model.iterations}


def bounds(values: np.ndarray, repetitions: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260905)
    block = 13
    starts = rng.integers(0, len(values), size=(repetitions, int(np.ceil(len(values)/block))))
    indices = ((starts[:, :, None] + np.arange(block)) % len(values)).reshape(repetitions, -1)[:, :len(values)]
    sample = values[indices].mean(axis=1)
    return float(np.quantile(sample, .1)), float(np.quantile(sample, .9))


def review_metrics(values: np.ndarray, milestone: int) -> dict:
    values = np.asarray(values, dtype=float)
    if milestone not in {52, 104} or len(values) != milestone or not np.isfinite(values).all():
        return {"passed": False, "reason": "INSUFFICIENT_COMPLETE_FROZEN_ORIGINS"}
    if milestone == 52:
        upper = bounds(values[-26:])[1]
        return {"passed": bool(values.mean() > 0 and upper >= 0), "overall_mean_improvement": values.mean(),
                "recent_26_mean_improvement": values[-26:].mean(), "recent_26_upper_90pct": upper}
    lower = bounds(values)[0]
    return {"passed": bool(values[:52].mean() > 0 and values[52:].mean() > 0 and lower > 0),
            "first_52_mean_improvement": values[:52].mean(), "second_52_mean_improvement": values[52:].mean(),
            "overall_lower_90pct": lower}


def review_if_due(root: Path, grid: pd.DatetimeIndex, cal: pd.DatetimeIndex, day: pd.Timestamp) -> dict | None:
    records = {r["origin_date"]: r for r in matured_records(root)}
    latest = None
    for count in [52, 104]:
        path = root / BASE / "reviews" / f"review_{count}.json"
        if len(grid) < count or cal.get_loc(grid[count-1]) + 5 >= len(cal):
            continue
        if cal[cal.get_loc(grid[count-1])+5] > day or path.exists():
            continue
        keys = [str(d.date()) for d in grid[:count]]
        values = np.array([records[k]["b1_minus_b0_loss_improvement"] for k in keys if k in records])
        result = review_metrics(values, count)
        drift_receipts = [load_json(p) for p in (root / BASE / "ticks").glob("*.json")]
        drift = any("DRIFT" in r.get("state", "") for r in drift_receipts)
        result.update({"milestone": count, "reviewed_at": now(), "input_model_drift": drift})
        result["passed"] = result["passed"] and not drift
        write_once(path, result)
        latest = result
        if not result["passed"]:
            write_once(root / BASE / "archive.json", {"state": "ARCHIVED_FORECAST_BENCHMARK_FAILED",
                       "closed_at": now(), "review": identity(root, path), "position_impact": 0})
            break
    return latest


def export_ledger(root: Path) -> None:
    rows = {p.stem: load_json(p) for p in (root / BASE / "predictions").glob("*.json")}
    rows.update({p.stem: load_json(p) for p in (root / BASE / "matured").glob("*.json")})
    frame = pd.DataFrame([rows[k] for k in sorted(rows)], columns=FIELDS)
    path = root / BASE / "forecast_ledger.csv"
    temporary = path.with_suffix(".csv.tmp")
    temporary.write_bytes(csv_bytes(frame))
    temporary.replace(path)


def tick(root: Path, moment: datetime | None = None, allow_fetch: bool = True) -> dict:
    actual = datetime.now(TZ)
    if moment is not None and moment != actual:
        raise ContractError("生产入口不接受回拨时钟或历史日期参数")
    day = pd.Timestamp(actual.date())
    config = cfg(root)
    manifest = load_json(root / MANIFEST)
    result = {"model_id": config["model_id"], "checked_at": actual.isoformat(), "position_impact": 0,
              "live_trading_authorized": False}
    try:
        verify(root, manifest["identities"])
        if (root / BASE / "archive.json").exists():
            return {**result, "state": "ARCHIVED_FORECAST_BENCHMARK_FAILED"}
        cal = calendar(root, config)
        if day > cal.max():
            raise ContractError("NO_VIEW_OFFICIAL_CALENDAR_EXTENSION_REQUIRED")
        grid = origins(cal, config)
        if day not in grid:
            result["state"] = "NOT_ORIGIN_NOOP"
        elif actual.time() < time(19, 30):
            result["state"] = "WAIT_ORIGIN_19_30"
        else:
            slot = int(grid.get_loc(day)) + 1
            result["scheduled_origin_number"] = slot
            output = root / BASE / "predictions" / f"{day:%Y-%m-%d}.json"
            if output.exists():
                receipt = load_json(root / BASE / "prediction_receipts" / output.name)
                verify(root, [receipt["prediction"], receipt["input"], receipt["model"]])
                result["state"] = "ALREADY_RECORDED_IMMUTABLE_FORECAST"
            elif not allow_fetch:
                result["state"] = "ORIGIN_INPUT_FETCH_DISABLED_STATUS_ONLY"
            else:
                for missed in grid[grid < day]:
                    if not (root / BASE / "predictions" / f"{missed:%Y-%m-%d}.json").exists():
                        path = root / BASE / "missed_origins" / f"{missed:%Y-%m-%d}.json"
                        if not path.exists():
                            write_once(path, {"origin_date": str(missed.date()), "observed_at": now(),
                                       "state": "NO_VIEW_MISSED_ORIGIN_NO_BACKFILL"})
                daily, div, input_id = risk_input(root, config, manifest, day)
                mature(root, config, daily, div, day, input_id)
                review = review_if_due(root, grid, cal, day)
                if (root / BASE / "archive.json").exists():
                    result.update({"state": "ARCHIVED_FORECAST_BENCHMARK_FAILED", "review": review})
                else:
                    vintage_slot = 1 + ((slot - 1) // 13) * 13
                    vintage_path = root / BASE / "models" / f"slot_{vintage_slot:04d}.json"
                    if slot == vintage_slot and not vintage_path.exists():
                        write_once(vintage_path, fit_vintage(root, config, manifest, daily, slot, day))
                        write_once(root / BASE / "model_receipts" / vintage_path.name, identity(root, vintage_path))
                    if not vintage_path.exists():
                        raise ContractError("NO_VIEW_MISSING_FROZEN_REFIT_VINTAGE")
                    verify(root, [load_json(root / BASE / "model_receipts" / vintage_path.name)])
                    vintage = load_json(vintage_path)
                    current = daily.loc[daily.date.eq(day), list(parent.B1_FEATURES)].to_numpy(float)
                    value = float(np.exp(np.clip(vintage["intercept"] +
                                  ((current[0] - vintage["means"]) / vintage["scales"]) @ vintage["coefficients"], -60, 60)))
                    position = cal.get_loc(day)
                    if position + 5 >= len(cal):
                        raise ContractError("NO_VIEW_LABEL_MATURITY_CALENDAR_MISSING")
                    record = {"origin_date": str(day.date()), "model_vintage_id": vintage["model_vintage_id"],
                              "training_end_date": vintage["training_end_date"], "b0_predicted_dsv5": vintage["b0"],
                              "b1_predicted_dsv5": max(value, 1e-8), "q_b1_div_b0": max(value, 1e-8)/vintage["b0"],
                              "input_sha256": input_id["sha256"], "prediction_generated_at": now(),
                              "label_maturity_date": str(cal[position+5].date()), "matured_actual_dsv5": None,
                              "b0_qlike": None, "b1_qlike": None, "b1_minus_b0_loss_improvement": None}
                    check_prediction_record(record)
                    write_once(output, record)
                    write_once(root / BASE / "prediction_receipts" / output.name,
                               {"prediction": identity(root, output), "input": input_id,
                                "model": identity(root, vintage_path), "manifest_sha256": sha(root / MANIFEST)})
                    result.update({"state": "RECORDED_FORECAST_ONLY", "prediction": identity(root, output)})
                export_ledger(root)
    except ContractError as error:
        result["state"] = str(error)
        if "grid" in locals() and day in grid and actual.time() >= time(19, 30):
            review = review_if_due(root, grid, cal, day)
            if (root / BASE / "archive.json").exists():
                result.update({"state": "ARCHIVED_FORECAST_BENCHMARK_FAILED", "review": review,
                               "origin_input_failure": str(error)})
    except Exception as error:
        result.update({"state": "PROGRAM_FAILED_NO_FORECAST", "error_type": type(error).__name__})
    path = root / BASE / "ticks" / f"{actual:%Y%m%dT%H%M%S%f}_{uuid.uuid4().hex[:12]}.json"
    write_once(path, result)
    return clean(result)
