"""海外宽基训练、留出闸门与510300一次揭盲的十因子非线性模型。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
import yaml
from zoneinfo import ZoneInfo

from research.graph_regime_martin_turtle_v2 import (
    CostModel,
    _events_by_date,
    add_point_in_time_adjusted_ohlc,
    performance_metrics,
    simulate_static_exposure,
)
from research.weekly_daily_technical_v1 import _target_with_minimum_gate


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "510300_cross_market_chart_ml_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "510300_cross_market_chart_ml_v1_manifest.json"
FACTOR_COLUMNS = [
    "MOM5_VOL",
    "MOM20_VOL",
    "MOM60_VOL",
    "MOM252_VOL",
    "EMA20_120_GAP",
    "RANGE_POSITION_252",
    "VOL_RATIO_5_60",
    "DOWNSIDE_SHARE_20",
    "DRAWDOWN60_VOL",
    "EFFICIENCY_RATIO_20",
]


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    """读取并验证冻结配置。"""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("跨市场配置必须是YAML对象")
    ids = [item["id"] for item in config["factors"]["definitions"]]
    if ids != FACTOR_COLUMNS or len(ids) != 10:
        raise ValueError("因子集合必须严格等于冻结的10个因子")
    if config["protocol"]["parameter_rescue_after_results"] is not False:
        raise ValueError("协议禁止结果后参数救援")
    if config["governance"]["target_labels_used_for_training"] is not False:
        raise ValueError("510300标签不得参与训练")
    if config["governance"]["target_features_used_for_training"] is not False:
        raise ValueError("510300特征不得参与训练")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_implementation() -> dict[str, Any]:
    """要求运行文件与揭盲前冻结清单逐字节一致。"""

    if not MANIFEST_FILE.exists():
        raise RuntimeError("缺少实现冻结清单，禁止训练或揭盲")
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"冻结文件哈希变化：{relative}")
    return manifest


def build_chart_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """从单一市场的调整价格构建10个因果、无量纲图形因子。"""

    eps = float(config["factors"]["epsilon"])
    data = frame.copy().sort_values("date").drop_duplicates("date", keep="last")
    data = data.reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.normalize()
    close = pd.to_numeric(data["adj_close"], errors="raise").astype(float)
    raw_close = pd.to_numeric(data["close"], errors="raise").astype(float)
    raw_open = pd.to_numeric(data["open"], errors="raise").astype(float)
    adjustment = close / raw_close
    data["adj_open"] = raw_open * adjustment
    log_return = np.log(close / close.shift(1))
    vol5 = log_return.rolling(5).std(ddof=0)
    vol60 = log_return.rolling(60).std(ddof=0)
    vol252 = log_return.rolling(252).std(ddof=0)
    data["MOM5_VOL"] = log_return.rolling(5).sum() / (vol60 * np.sqrt(5.0) + eps)
    data["MOM20_VOL"] = log_return.rolling(20).sum() / (vol60 * np.sqrt(20.0) + eps)
    data["MOM60_VOL"] = log_return.rolling(60).sum() / (vol60 * np.sqrt(60.0) + eps)
    data["MOM252_VOL"] = log_return.rolling(252).sum() / (vol252 * np.sqrt(252.0) + eps)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema120 = close.ewm(span=120, adjust=False).mean()
    data["EMA20_120_GAP"] = np.log(ema20 / ema120) / (vol60 * np.sqrt(100.0) + eps)
    low252 = close.rolling(252).min()
    high252 = close.rolling(252).max()
    data["RANGE_POSITION_252"] = 2.0 * (close - low252) / (high252 - low252 + eps) - 1.0
    data["VOL_RATIO_5_60"] = np.log((vol5 + eps) / (vol60 + eps))
    downside_square = log_return.clip(upper=0.0).pow(2).rolling(20).sum()
    total_square = log_return.pow(2).rolling(20).sum()
    data["DOWNSIDE_SHARE_20"] = 2.0 * downside_square / (total_square + eps) - 1.0
    data["DRAWDOWN60_VOL"] = (close / close.rolling(60).max() - 1.0) / (
        vol60 * np.sqrt(60.0) + eps
    )
    path20 = close.diff().abs().rolling(20).sum()
    data["EFFICIENCY_RATIO_20"] = (close - close.shift(20)).abs() / (path20 + eps)
    data["bar_number"] = np.arange(len(data), dtype=int)
    return data


def add_forward_label(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """添加下一开盘进入、20个交易日后开盘退出的成本后方向标签。"""

    horizon = int(config["sample_design"]["prediction_horizon_bars"])
    round_trip = float(config["sample_design"]["label_round_trip_cost_bps"]) / 10_000.0
    data = features.copy()
    data["entry_date"] = data["date"].shift(-1)
    data["exit_date"] = data["date"].shift(-(horizon + 1))
    data["entry_open"] = data["adj_open"].shift(-1)
    data["exit_open"] = data["adj_open"].shift(-(horizon + 1))
    data["forward_log_return_net"] = np.log(data["exit_open"] / data["entry_open"]) - round_trip
    data["label_positive"] = data["forward_log_return_net"].gt(0.0).astype("Int64")
    invalid = data[["entry_open", "exit_open"]].isna().any(axis=1)
    data.loc[invalid, "label_positive"] = pd.NA
    return data


def _raw_path(config: dict[str, Any], symbol: str) -> Path:
    name = f"{symbol.replace('^', '')}_daily.parquet"
    return ROOT / config["data_contracts"]["raw_directory"] / name


def load_international_market(config: dict[str, Any], symbol: str) -> pd.DataFrame:
    path = _raw_path(config, symbol)
    if not path.exists():
        raise FileNotFoundError(f"缺少海外数据：{path}")
    frame = pd.read_parquet(path)
    if set(frame["symbol"].astype(str).unique()) != {symbol}:
        raise ValueError(f"海外文件标的错误：{symbol}")
    return frame


def _complete_rows(data: pd.DataFrame) -> pd.Series:
    return np.isfinite(data[FACTOR_COLUMNS].to_numpy(float)).all(axis=1)


def make_training_samples(config: dict[str, Any]) -> pd.DataFrame:
    """只从九个冻结海外训练市场生成样本。"""

    design = config["sample_design"]
    rows: list[pd.DataFrame] = []
    for item in config["data_contracts"]["training_symbols"]:
        symbol = item["symbol"]
        featured = add_forward_label(
            build_chart_features(load_international_market(config, symbol), config), config
        )
        mask = (
            _complete_rows(featured)
            & featured["date"].ge(pd.Timestamp(design["training_start"]))
            & featured["exit_date"].le(pd.Timestamp(design["training_label_exit_end"]))
            & featured["label_positive"].notna()
            & featured["bar_number"].mod(int(design["feature_sampling_step_bars"])).eq(0)
        )
        sample = featured.loc[
            mask,
            ["date", "entry_date", "exit_date", "forward_log_return_net", "label_positive"]
            + FACTOR_COLUMNS,
        ].copy()
        sample["symbol"] = symbol
        rows.append(sample)
    result = pd.concat(rows, ignore_index=True)
    if len(result) < int(design["minimum_training_samples"]):
        raise RuntimeError(f"训练样本不足：{len(result)}")
    return result


def fit_model(samples: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """按冻结参数训练一次模型。"""

    model_config = config["model"]
    model = HistGradientBoostingClassifier(
        loss=model_config["loss"],
        learning_rate=float(model_config["learning_rate"]),
        max_iter=int(model_config["max_iter"]),
        max_leaf_nodes=int(model_config["max_leaf_nodes"]),
        max_depth=int(model_config["max_depth"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        l2_regularization=float(model_config["l2_regularization"]),
        early_stopping=bool(model_config["early_stopping"]),
        random_state=int(model_config["random_state"]),
    )
    x = samples[FACTOR_COLUMNS].to_numpy(float)
    y = samples["label_positive"].astype(int).to_numpy()
    model.fit(x, y)
    return {
        "model": model,
        "base_rate": float(y.mean()),
        "factor_columns": FACTOR_COLUMNS,
        "training_rows": int(len(samples)),
        "training_symbols": sorted(samples["symbol"].unique().tolist()),
    }


def probability_to_exposure(probability: np.ndarray, base_rate: float, config: dict[str, Any]) -> np.ndarray:
    """按冻结概率边际阈值映射四档仓位。"""

    thresholds = np.asarray(config["model"]["probability_edge_thresholds"], dtype=float)
    exposures = np.asarray(config["model"]["target_exposures"], dtype=float)
    if len(thresholds) != 3 or len(exposures) != 4:
        raise ValueError("仓位映射必须是三个阈值和四档仓位")
    edge = np.round(np.asarray(probability, dtype=float) - float(base_rate), 12)
    bucket = np.digitize(edge, thresholds, right=False)
    return exposures[bucket]


def make_external_predictions(bundle: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """只对三个从未训练的海外市场产生固定间隔预测。"""

    design = config["sample_design"]
    rows: list[pd.DataFrame] = []
    for item in config["data_contracts"]["external_gate_symbols"]:
        symbol = item["symbol"]
        featured = add_forward_label(
            build_chart_features(load_international_market(config, symbol), config), config
        )
        mask = (
            _complete_rows(featured)
            & featured["date"].between(
                pd.Timestamp(design["external_gate_start"]),
                pd.Timestamp(design["external_gate_end"]),
            )
            & featured["label_positive"].notna()
            & featured["bar_number"].mod(int(design["target_rebalance_step_bars"])).eq(0)
        )
        sample = featured.loc[
            mask,
            ["date", "entry_date", "exit_date", "entry_open", "exit_open", "forward_log_return_net", "label_positive"]
            + FACTOR_COLUMNS,
        ].copy()
        probability = bundle["model"].predict_proba(sample[FACTOR_COLUMNS].to_numpy(float))[:, 1]
        sample["probability_positive"] = probability
        sample["probability_edge"] = probability - float(bundle["base_rate"])
        sample["target_exposure"] = probability_to_exposure(probability, bundle["base_rate"], config)
        sample["symbol"] = symbol
        rows.append(sample)
    return pd.concat(rows, ignore_index=True)


def evaluate_external_gate(predictions: pd.DataFrame, bundle: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """评估三个完全留出市场，不进行参数选择。"""

    gate = config["external_gate"]
    cost = float(gate["one_way_cost_bps_per_full_exposure"]) / 10_000.0
    cash_rate = float(gate["cash_annual_rate"])
    horizon = int(config["sample_design"]["prediction_horizon_bars"])
    cash_period = (1.0 + cash_rate) ** (horizon / 242.0) - 1.0
    market_results: dict[str, Any] = {}
    for symbol, group in predictions.groupby("symbol", sort=True):
        group = group.sort_values("entry_date").reset_index(drop=True)
        market_return = group["exit_open"].to_numpy(float) / group["entry_open"].to_numpy(float) - 1.0
        exposure = group["target_exposure"].to_numpy(float)
        previous = np.r_[0.0, exposure[:-1]]
        strategy_return = exposure * market_return + (1.0 - exposure) * cash_period - np.abs(exposure - previous) * cost
        strategy_return[-1] -= exposure[-1] * cost
        elapsed_years = max((group["exit_date"].iloc[-1] - group["entry_date"].iloc[0]).days / 365.2425, 1 / 365.2425)
        strategy_growth = float(np.prod(1.0 + strategy_return))
        benchmark_growth = float(np.prod(1.0 + market_return))
        strategy_cagr = strategy_growth ** (1.0 / elapsed_years) - 1.0
        benchmark_cagr = benchmark_growth ** (1.0 / elapsed_years) - 1.0
        market_results[symbol] = {
            "predictions": int(len(group)),
            "strategy_cagr": strategy_cagr,
            "benchmark_cagr": benchmark_cagr,
            "annualized_excess": strategy_cagr - benchmark_cagr,
            "average_exposure": float(exposure.mean()),
        }
    predicted_class = predictions["probability_positive"].gt(float(bundle["base_rate"])).astype(int)
    balanced_accuracy = float(
        balanced_accuracy_score(predictions["label_positive"].astype(int), predicted_class)
    )
    excesses = [item["annualized_excess"] for item in market_results.values()]
    enough_predictions = all(
        item["predictions"] >= int(config["sample_design"]["minimum_external_predictions_per_market"])
        for item in market_results.values()
    )
    checks = {
        "enough_predictions_each_market": enough_predictions,
        "median_annualized_excess_positive": bool(np.median(excesses) > 0.0),
        "minimum_positive_markets": bool(
            sum(value > 0.0 for value in excesses) >= int(gate["minimum_positive_markets"])
        ),
        "minimum_balanced_accuracy": bool(
            balanced_accuracy >= float(gate["minimum_balanced_accuracy"])
        ),
    }
    return {
        "status": "PASS_EXTERNAL_GATE" if all(checks.values()) else "REJECT_EXTERNAL_GATE_FROZEN",
        "passed": bool(all(checks.values())),
        "training_base_rate": float(bundle["base_rate"]),
        "pooled_balanced_accuracy": balanced_accuracy,
        "median_annualized_excess": float(np.median(excesses)),
        "positive_markets": int(sum(value > 0.0 for value in excesses)),
        "checks": checks,
        "markets": market_results,
    }


def run_external_stage(config: dict[str, Any]) -> dict[str, Any]:
    """训练并运行一次海外留出闸门。"""

    assert_frozen_implementation()
    data_audit = json.loads((ROOT / config["data_contracts"]["audit_file"]).read_text(encoding="utf-8"))
    if not data_audit.get("overall_pass"):
        raise RuntimeError("海外数据闸门未通过，禁止训练")
    samples = make_training_samples(config)
    bundle = fit_model(samples, config)
    predictions = make_external_predictions(bundle, config)
    report = evaluate_external_gate(predictions, bundle, config)
    report.update(
        {
            "project_id": config["protocol"]["project_id"],
            "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "training_rows": int(len(samples)),
            "training_symbols": bundle["training_symbols"],
            "factor_count": len(FACTOR_COLUMNS),
        }
    )
    for key, frame in (
        ("training_samples_file", samples),
        ("external_predictions_file", predictions),
    ):
        path = ROOT / config["outputs"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    model_path = ROOT / config["outputs"]["model_file"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report_path = ROOT / config["outputs"]["external_gate_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_target_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """合并冻结底座与只追加的新日线，不覆盖历史记录。"""

    contract = config["data_contracts"]
    market_parts = [pd.read_parquet(ROOT / path) for path in contract["target_market_files"]]
    benchmark_parts = [pd.read_parquet(ROOT / path) for path in contract["target_benchmark_files"]]
    market = pd.concat(market_parts, ignore_index=True)
    benchmark = pd.concat(benchmark_parts, ignore_index=True)
    for frame in (market, benchmark):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    market = market.sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    dividends = pd.read_csv(ROOT / contract["target_distributions_file"])
    for column in ("record_date", "ex_date", "payment_date"):
        dividends[column] = pd.to_datetime(dividends[column], errors="raise").dt.normalize()
    coverage = json.loads((ROOT / contract["target_distribution_coverage_file"]).read_text(encoding="utf-8"))
    if not coverage.get("complete_history_confirmed") or int(coverage.get("event_count", 0)) != len(dividends):
        raise RuntimeError("510300分红完整性闸门未通过")
    return market, benchmark, dividends


def make_target_features_and_predictions(
    market: pd.DataFrame,
    dividends: pd.DataFrame,
    bundle: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在冻结模型下产生一次510300特征和目标仓位。"""

    adjusted = add_point_in_time_adjusted_ohlc(market, dividends)
    adjusted = adjusted.rename(columns={"adjusted_close": "adj_close"})
    featured = build_chart_features(adjusted, config)
    mask = _complete_rows(featured) & featured["bar_number"].mod(
        int(config["sample_design"]["target_rebalance_step_bars"])
    ).eq(0)
    predictions = featured.loc[mask, ["date"] + FACTOR_COLUMNS].copy()
    probability = bundle["model"].predict_proba(predictions[FACTOR_COLUMNS].to_numpy(float))[:, 1]
    predictions["probability_positive"] = probability
    predictions["probability_edge"] = probability - float(bundle["base_rate"])
    predictions["target_exposure"] = probability_to_exposure(probability, bundle["base_rate"], config)
    return featured, predictions


def simulate_target(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    dividends: pd.DataFrame,
    config: dict[str, Any],
    *,
    slippage_bps: float,
) -> dict[str, pd.DataFrame]:
    """按下一开盘、整手、最低佣金和5千元闸门模拟510300。"""

    execution = config["target_execution"]
    costs = CostModel(
        commission_rate=float(execution["commission_rate_per_leg"]),
        minimum_commission=float(execution["minimum_commission_cny_per_leg"]),
        slippage_rate=float(slippage_bps) / 10_000.0,
        multiplier=1.0,
    )
    prediction_map = predictions.set_index("date").to_dict("index")
    first_signal = predictions["date"].min()
    data = features.loc[features["date"].ge(first_signal)].copy().reset_index(drop=True)
    record_events = _events_by_date(dividends, "record_date")
    ex_events = _events_by_date(dividends, "ex_date")
    payment_events = _events_by_date(dividends, "payment_date")
    entitlements: dict[pd.Timestamp, float] = {}
    receivables_by_payment: dict[pd.Timestamp, float] = defaultdict(float)
    cash = float(execution["initial_capital_cny"])
    shares = 0
    receivable = 0.0
    pending: dict[str, Any] | None = None
    daily_cash_rate = (1.0 + float(execution["cash_annual_rate"])) ** (
        1.0 / int(execution["trading_days_per_year"])
    ) - 1.0
    ledgers: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for row in data.itertuples(index=False):
        date = row.date
        for event in ex_events.get(date, []):
            amount = entitlements.get(event.ex_date, 0.0)
            receivable += amount
            receivables_by_payment[event.payment_date] += amount
        for _event in payment_events.get(date, []):
            amount = receivables_by_payment.pop(date, 0.0)
            cash += amount
            receivable -= amount

        commission = slippage = notional = 0.0
        if pending is not None:
            next_cash, next_shares, trade, rejection, preview = _target_with_minimum_gate(
                target_exposure=float(pending["target_exposure"]),
                raw_open=float(row.open),
                cash=cash,
                shares=shares,
                receivable=receivable,
                lot_size=int(execution["lot_size_shares"]),
                costs=costs,
                minimum_notional=float(execution["minimum_normal_trade_notional_cny"]),
                allow_small_liquidation=bool(execution["risk_liquidation_exempt_from_minimum_notional"]),
            )
            if trade is not None:
                cash, shares = next_cash, next_shares
                commission = float(trade["commission"])
                slippage = float(trade["slippage_cost"])
                notional = float(trade["raw_notional"])
                trades.append(
                    {
                        "signal_date": pending["signal_date"],
                        "execution_date": date,
                        "probability_positive": pending["probability_positive"],
                        "target_exposure": pending["target_exposure"],
                        **trade,
                    }
                )
            elif rejection == "BELOW_MINIMUM_NOTIONAL":
                blocked.append(
                    {
                        "signal_date": pending["signal_date"],
                        "execution_date": date,
                        "target_exposure": pending["target_exposure"],
                        "preview_raw_notional": preview,
                        "rejection": rejection,
                    }
                )
            pending = None

        cash *= 1.0 + daily_cash_rate
        equity = cash + shares * float(row.close) + receivable
        for event in record_events.get(date, []):
            entitlements[event.ex_date] = shares * float(event.cash_dividend_per_share)
        if date in prediction_map:
            signal = prediction_map[date]
            pending = {
                "signal_date": date,
                "probability_positive": float(signal["probability_positive"]),
                "target_exposure": float(signal["target_exposure"]),
            }
        ledgers.append(
            {
                "date": date,
                "equity": equity,
                "shares": shares,
                "cash": cash,
                "receivable": receivable,
                "exposure": shares * float(row.close) / equity,
                "day_commission": commission,
                "day_slippage": slippage,
                "day_raw_notional": notional,
            }
        )
    return {
        "ledger": pd.DataFrame(ledgers),
        "trades": pd.DataFrame(trades),
        "blocked": pd.DataFrame(blocked),
    }


def _h00300_ledger(benchmark: pd.DataFrame, dates: pd.Series, initial: float) -> pd.DataFrame:
    aligned = pd.DataFrame({"date": pd.to_datetime(dates)}).merge(
        benchmark[["date", "close"]], on="date", how="left"
    )
    aligned["close"] = pd.to_numeric(aligned["close"], errors="coerce").ffill().bfill()
    aligned["equity"] = initial * aligned["close"] / aligned["close"].iloc[0]
    aligned["exposure"] = 1.0
    aligned["day_raw_notional"] = 0.0
    aligned["day_commission"] = 0.0
    aligned["day_slippage"] = 0.0
    return aligned


def _bootstrap_excess_lower(
    strategy: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: dict[str, Any],
) -> float:
    merged = strategy[["date", "equity"]].merge(
        benchmark[["date", "equity"]], on="date", suffixes=("_strategy", "_benchmark")
    )
    excess_log = (
        np.log(merged["equity_strategy"]).diff() - np.log(merged["equity_benchmark"]).diff()
    ).dropna().to_numpy(float)
    block = int(config["target_acceptance"]["bootstrap_block_days"])
    repetitions = int(config["target_acceptance"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(config["target_acceptance"]["bootstrap_seed"]))
    starts = np.arange(0, max(len(excess_log) - block + 1, 1))
    estimates = np.empty(repetitions, dtype=float)
    blocks_needed = int(np.ceil(len(excess_log) / block))
    for index in range(repetitions):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([excess_log[start : start + block] for start in chosen])[: len(excess_log)]
        estimates[index] = np.exp(sample.mean() * 242.0) - 1.0
    return float(np.quantile(estimates, 0.025))


def reveal_target_once(config: dict[str, Any]) -> dict[str, Any]:
    """海外闸门通过后，执行唯一一次510300历史揭盲。"""

    assert_frozen_implementation()
    external_path = ROOT / config["outputs"]["external_gate_report"]
    external = json.loads(external_path.read_text(encoding="utf-8"))
    if not external.get("passed"):
        raise RuntimeError("海外留出闸门失败，禁止查看510300结果")
    report_path = ROOT / config["outputs"]["target_report_json"]
    if report_path.exists():
        raise RuntimeError("510300已经揭盲一次，禁止重复运行")
    bundle = joblib.load(ROOT / config["outputs"]["model_file"])
    market, benchmark, dividends = load_target_inputs(config)
    features, predictions = make_target_features_and_predictions(market, dividends, bundle, config)
    base = simulate_target(
        features, predictions, dividends, config,
        slippage_bps=float(config["target_execution"]["slippage_bps_per_leg"]),
    )
    stress = simulate_target(
        features, predictions, dividends, config,
        slippage_bps=float(config["target_execution"]["stress_slippage_bps_per_leg"]),
    )
    buyhold_config = {"price_and_execution": dict(config["target_execution"])}
    buyhold_config["price_and_execution"]["slippage_bps_per_leg"] = float(
        config["target_execution"]["slippage_bps_per_leg"]
    )
    target_start = base["ledger"]["date"].min()
    target_features = features.loc[features["date"].ge(target_start)].copy().reset_index(drop=True)
    buyhold = simulate_static_exposure(target_features, dividends, buyhold_config, 1.0)
    h00300 = _h00300_ledger(
        benchmark, base["ledger"]["date"], float(config["target_execution"]["initial_capital_cny"])
    )
    base_metrics = performance_metrics(base["ledger"])
    stress_metrics = performance_metrics(stress["ledger"])
    buyhold_metrics = performance_metrics(buyhold)
    h00300_metrics = performance_metrics(h00300)
    base_alpha_h = base_metrics["cagr"] - h00300_metrics["cagr"]
    base_alpha_b = base_metrics["cagr"] - buyhold_metrics["cagr"]
    stress_alpha_h = stress_metrics["cagr"] - h00300_metrics["cagr"]
    bootstrap_lower = _bootstrap_excess_lower(base["ledger"], h00300, config)
    trades = base["trades"]
    normal = trades.loc[~((trades.get("side") == "SELL") & (trades.get("target_exposure") == 0.0))] if not trades.empty else trades
    minimum_normal = float(normal["raw_notional"].min()) if not normal.empty else np.nan
    acceptance = config["target_acceptance"]
    checks = {
        "base_alpha_vs_h00300_20pp": bool(base_alpha_h >= float(acceptance["minimum_annualized_excess_vs_h00300"])),
        "base_alpha_vs_buyhold_20pp": bool(base_alpha_b >= float(acceptance["minimum_annualized_excess_vs_510300_buy_hold"])),
        "stress_alpha_vs_h00300_20pp": bool(stress_alpha_h >= float(acceptance["minimum_stress_annualized_excess_vs_h00300"])),
        "minimum_executed_trades": bool(len(trades) >= int(acceptance["minimum_executed_trades"])),
        "minimum_normal_trade_notional": bool(
            np.isfinite(minimum_normal)
            and minimum_normal >= float(acceptance["require_all_normal_trades_at_least_cny"])
        ),
        "bootstrap_lower_bound_positive": bool(bootstrap_lower > 0.0),
    }
    report = {
        "project_id": config["protocol"]["project_id"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS_TARGET_ALPHA_GATE" if all(checks.values()) else "REJECT_TARGET_FROZEN",
        "passed": bool(all(checks.values())),
        "evidence_label": config["protocol"]["target_history_label"],
        "factor_count": len(FACTOR_COLUMNS),
        "target_predictions": int(len(predictions)),
        "executed_trades": int(len(trades)),
        "blocked_small_orders": int(len(base["blocked"])),
        "minimum_normal_trade_notional_cny": minimum_normal,
        "base": base_metrics,
        "stress": stress_metrics,
        "buyhold_510300": buyhold_metrics,
        "h00300": h00300_metrics,
        "base_annualized_excess_vs_h00300": base_alpha_h,
        "base_annualized_excess_vs_510300_buyhold": base_alpha_b,
        "stress_annualized_excess_vs_h00300": stress_alpha_h,
        "bootstrap_95pct_lower_annualized_excess_vs_h00300": bootstrap_lower,
        "checks": checks,
    }
    prediction_path = ROOT / config["outputs"]["target_predictions_file"]
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)
    output_dir = ROOT / config["outputs"]["target_ledger_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for prefix, result in (("base", base), ("stress", stress)):
        result["ledger"].to_parquet(output_dir / f"{prefix}_ledger.parquet", index=False)
        result["trades"].to_csv(output_dir / f"{prefix}_trades.csv", index=False, encoding="utf-8-sig")
        result["blocked"].to_csv(output_dir / f"{prefix}_blocked_orders.csv", index=False, encoding="utf-8-sig")
    buyhold.to_parquet(output_dir / "buyhold_510300_ledger.parquet", index=False)
    h00300.to_parquet(output_dir / "h00300_ledger.parquet", index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = _target_markdown(report)
    markdown_path = ROOT / config["outputs"]["target_report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")
    return report


def _target_markdown(report: dict[str, Any]) -> str:
    def pct(value: float) -> str:
        return f"{value:.2%}"

    return "\n".join(
        [
            "# 510300跨市场图形机器学习V1一次性揭盲",
            "",
            f"- 状态：`{report['status']}`",
            f"- 证据标签：`{report['evidence_label']}`",
            f"- 因子：{report['factor_count']}个；目标预测：{report['target_predictions']}次；实际成交：{report['executed_trades']}笔。",
            "",
            "|口径|策略年化|H00300年化|对H00300超额|510300买入持有年化|对买入持有超额|最大回撤|",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"|基础5bp|{pct(report['base']['cagr'])}|{pct(report['h00300']['cagr'])}|{pct(report['base_annualized_excess_vs_h00300'])}|{pct(report['buyhold_510300']['cagr'])}|{pct(report['base_annualized_excess_vs_510300_buyhold'])}|{pct(report['base']['maximum_drawdown'])}|",
            f"|压力15bp|{pct(report['stress']['cagr'])}|{pct(report['h00300']['cagr'])}|{pct(report['stress_annualized_excess_vs_h00300'])}|{pct(report['buyhold_510300']['cagr'])}|{pct(report['stress']['cagr']-report['buyhold_510300']['cagr'])}|{pct(report['stress']['maximum_drawdown'])}|",
            "",
            f"- 20日区块Bootstrap年化超额95%下界：{pct(report['bootstrap_95pct_lower_annualized_excess_vs_h00300'])}。",
            f"- 最小普通成交：{report['minimum_normal_trade_notional_cny']:.2f}元；被5,000元闸门拦截：{report['blocked_small_orders']}次。",
            "- 该历史目标已揭盲，失败后禁止修改本协议参数；本报告不生成真实仓位或订单。",
        ]
    ) + "\n"
