from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
STUDY_ID = "CSI300_PIT_FUNDAMENTAL_UNDERREACTION_INCOMPLETE_COVERAGE_ANALYSIS_V1"
SCRIPT_VERSION = "1.0.1"

CONFIG_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_incomplete_coverage_analysis_v1.json"
MAIN_MANIFEST_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_incomplete_coverage_analysis_v1_manifest.json"
ADDENDUM_1_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_incomplete_coverage_analysis_v1_0_1_schema_addendum.json"
ADDENDUM_1_MANIFEST_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_incomplete_coverage_analysis_v1_0_1_schema_addendum_manifest.json"
ADDENDUM_2_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_incomplete_coverage_analysis_v1_0_2_input_completeness_addendum.json"
ADDENDUM_2_MANIFEST_PATH = ROOT / "config/csi300_pit_fundamental_underreaction_incomplete_coverage_analysis_v1_0_2_input_completeness_addendum_manifest.json"

REQUIRED_METRICS = (
    "OPERATING_REVENUE_YTD",
    "OPERATING_PROFIT_YTD",
    "PARENT_NET_PROFIT_YTD",
    "CORE_PARENT_NET_PROFIT_YTD",
    "OPERATING_CASH_FLOW_YTD",
    "ACCOUNTS_RECEIVABLE_END",
    "INVENTORY_END",
    "TOTAL_ASSETS_END",
    "TOTAL_LIABILITIES_END",
)

FLOW_METRICS = (
    "OPERATING_REVENUE_YTD",
    "OPERATING_PROFIT_YTD",
    "PARENT_NET_PROFIT_YTD",
    "CORE_PARENT_NET_PROFIT_YTD",
    "OPERATING_CASH_FLOW_YTD",
)

OPERATING_COLUMNS = (
    "revenue_growth_acceleration",
    "operating_margin_change",
    "operating_roa_change",
)

OUTCOME_CUTOFF = pd.Timestamp("2026-08-14")
MINIMUM_REFERENCE_COUNT = 30
TRAILING_DAYS = 730
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260902


@dataclass(frozen=True)
class ArtifactPaths:
    event_panel: Path
    result_json: Path
    result_markdown: Path
    run_receipt: Path


def project_path(relative_path: str) -> Path:
    return ROOT / Path(relative_path.replace("/", "\\"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def finite_all(values: Iterable[Any]) -> bool:
    return all(finite(value) for value in values)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if value is pd.NA:
        return None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json_strict(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        json_safe(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    for relative_path, metadata in manifest.get("files", {}).items():
        target = project_path(relative_path)
        if not target.exists():
            raise FileNotFoundError(f"冻结文件不存在：{relative_path}")
        expected_size = int(metadata["bytes"])
        actual_size = target.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"冻结文件大小漂移：{relative_path}，实际 {actual_size}，期望 {expected_size}"
            )
        actual_hash = sha256_file(target)
        if actual_hash != metadata["sha256"]:
            raise RuntimeError(
                f"冻结文件哈希漂移：{relative_path}，实际 {actual_hash}，期望 {metadata['sha256']}"
            )
    return manifest


def verify_protocol_chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    main_manifest = verify_manifest(MAIN_MANIFEST_PATH)
    addendum_1_manifest = verify_manifest(ADDENDUM_1_MANIFEST_PATH)
    addendum_2_manifest = verify_manifest(ADDENDUM_2_MANIFEST_PATH)

    parent_1 = addendum_1_manifest["parent_manifest"]
    if sha256_file(project_path(parent_1["path"])) != parent_1["sha256"]:
        raise RuntimeError("V1.0.1 的父冻结清单哈希不匹配")
    parent_2 = addendum_2_manifest["parent_addendum_manifest"]
    if sha256_file(project_path(parent_2["path"])) != parent_2["sha256"]:
        raise RuntimeError("V1.0.2 的父补充清单哈希不匹配")

    config = read_json(CONFIG_PATH)
    addendum_1 = read_json(ADDENDUM_1_PATH)
    addendum_2 = read_json(ADDENDUM_2_PATH)
    if config["study_id"] != STUDY_ID:
        raise RuntimeError("分析配置 study_id 不匹配")
    if addendum_2["information_seen_before_addendum"]["future_returns_computed"]:
        raise RuntimeError("输入补充不是在未来收益计算前冻结")
    return config, addendum_1, addendum_2


def quarter_index(report_period: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(report_period)
    quarter = (timestamp.month - 1) // 3 + 1
    expected = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}[quarter]
    if (timestamp.month, timestamp.day) != expected:
        raise ValueError(f"报告期不是标准季度末：{timestamp.date().isoformat()}")
    return timestamp.year * 4 + quarter - 1


def load_financial_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dependency = pd.read_parquet(project_path(config["frozen_sample"]["source_ledger_path"]))
    facts = pd.read_parquet(project_path(config["inputs"]["official_financial_facts"]["path"]))
    requirements = pd.read_parquet(
        project_path(config["inputs"]["official_fact_requirement_ledger"]["path"])
    )

    if int(dependency["target_event_ready"].sum()) != int(
        config["frozen_sample"]["expected_event_count"]
    ):
        raise RuntimeError("冻结就绪事件数漂移")
    if dependency.duplicated("announcement_id").any():
        raise RuntimeError("事件依赖账本公告 ID 不唯一")
    if facts.duplicated(["announcement_id", "metric_id"]).any():
        raise RuntimeError("官方事实公告—指标键不唯一")
    if not set(facts["metric_id"].unique()).issubset(set(REQUIRED_METRICS)):
        unexpected = sorted(set(facts["metric_id"].unique()) - set(REQUIRED_METRICS))
        raise RuntimeError(f"官方事实出现非冻结指标：{unexpected}")
    if bool(facts["market_price_read"].any()) or bool(facts["future_return_read"].any()):
        raise RuntimeError("官方事实源文件的原始无收益标志被改写")
    if bool(dependency["market_price_read"].any()) or bool(
        dependency["future_return_read"].any()
    ):
        raise RuntimeError("依赖账本的原始无收益标志被改写")
    if bool(requirements["market_price_read"].any()) or bool(
        requirements["future_return_read"].any()
    ):
        raise RuntimeError("需求账本的原始无收益标志被改写")
    return dependency, facts, requirements


def build_quarter_records(facts: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
    metadata_columns = [
        "announcement_id",
        "ts_code",
        "report_period",
        "period_type",
        "event_publication_date",
    ]
    metadata = facts[metadata_columns].drop_duplicates("announcement_id")
    wide = facts.pivot(
        index="announcement_id", columns="metric_id", values="metric_value_cny"
    ).reset_index()
    reports = metadata.merge(wide, on="announcement_id", how="left", validate="one_to_one")
    reports["report_period"] = pd.to_datetime(reports["report_period"], errors="raise")
    reports["quarter_index"] = reports["report_period"].map(quarter_index)
    if reports.duplicated(["ts_code", "quarter_index"]).any():
        duplicates = reports.loc[
            reports.duplicated(["ts_code", "quarter_index"], keep=False),
            ["ts_code", "report_period", "announcement_id"],
        ]
        raise RuntimeError(
            "官方事实存在同证券同报告期多文档："
            + duplicates.head(20).to_json(orient="records", force_ascii=False)
        )

    base: dict[tuple[str, int], dict[str, float]] = {}
    for row in reports.to_dict("records"):
        key = (str(row["ts_code"]), int(row["quarter_index"]))
        base[key] = {
            metric: float(row[metric]) if finite(row.get(metric)) else float("nan")
            for metric in REQUIRED_METRICS
        }

    for (symbol, index), record in base.items():
        quarter = index % 4 + 1
        previous = base.get((symbol, index - 1))
        for metric in FLOW_METRICS:
            output = metric.replace("_YTD", "_SQ")
            current_value = record[metric]
            if quarter == 1:
                record[output] = current_value
            elif previous is not None and finite_all((current_value, previous[metric])):
                record[output] = current_value - previous[metric]
            else:
                record[output] = float("nan")
    return base


class FeatureEngine:
    def __init__(self, quarters: dict[tuple[str, int], dict[str, float]]) -> None:
        self.quarters = quarters

    def value(self, symbol: str, index: int, column: str) -> float:
        row = self.quarters.get((symbol, index))
        if row is None:
            return float("nan")
        value = row.get(column, float("nan"))
        return float(value) if finite(value) else float("nan")

    @lru_cache(maxsize=None)
    def ttm(self, symbol: str, index: int, ytd_metric: str) -> float:
        single_metric = ytd_metric.replace("_YTD", "_SQ")
        values = [self.value(symbol, index - lag, single_metric) for lag in range(4)]
        return float(sum(values)) if finite_all(values) else float("nan")

    @lru_cache(maxsize=None)
    def operating(self, symbol: str, index: int) -> dict[str, Any]:
        revenue_now = self.value(symbol, index, "OPERATING_REVENUE_SQ")
        revenue_year_ago = self.value(symbol, index - 4, "OPERATING_REVENUE_SQ")
        revenue_previous = self.value(symbol, index - 1, "OPERATING_REVENUE_SQ")
        revenue_previous_year_ago = self.value(
            symbol, index - 5, "OPERATING_REVENUE_SQ"
        )
        if finite_all(
            (
                revenue_now,
                revenue_year_ago,
                revenue_previous,
                revenue_previous_year_ago,
            )
        ) and revenue_year_ago > 0 and revenue_previous_year_ago > 0:
            revenue_growth_acceleration = (
                revenue_now / revenue_year_ago
                - revenue_previous / revenue_previous_year_ago
            )
        else:
            revenue_growth_acceleration = float("nan")

        operating_profit_now = self.value(symbol, index, "OPERATING_PROFIT_SQ")
        operating_profit_year_ago = self.value(
            symbol, index - 4, "OPERATING_PROFIT_SQ"
        )
        if (
            finite_all(
                (
                    operating_profit_now,
                    operating_profit_year_ago,
                    revenue_now,
                    revenue_year_ago,
                )
            )
            and revenue_now > 0
            and revenue_year_ago > 0
        ):
            operating_margin_change = (
                operating_profit_now / revenue_now
                - operating_profit_year_ago / revenue_year_ago
            )
        else:
            operating_margin_change = float("nan")

        ttm_operating_now = self.ttm(symbol, index, "OPERATING_PROFIT_YTD")
        ttm_operating_year_ago = self.ttm(
            symbol, index - 4, "OPERATING_PROFIT_YTD"
        )
        assets_now = self.value(symbol, index, "TOTAL_ASSETS_END")
        assets_year_ago = self.value(symbol, index - 4, "TOTAL_ASSETS_END")
        assets_two_year_ago = self.value(symbol, index - 8, "TOTAL_ASSETS_END")
        if (
            finite_all(
                (
                    ttm_operating_now,
                    ttm_operating_year_ago,
                    assets_now,
                    assets_year_ago,
                    assets_two_year_ago,
                )
            )
            and assets_now > 0
            and assets_year_ago > 0
            and assets_two_year_ago > 0
        ):
            operating_roa_change = (
                ttm_operating_now / ((assets_now + assets_year_ago) / 2.0)
                - ttm_operating_year_ago
                / ((assets_year_ago + assets_two_year_ago) / 2.0)
            )
        else:
            operating_roa_change = float("nan")

        values = (
            revenue_growth_acceleration,
            operating_margin_change,
            operating_roa_change,
        )
        return {
            "revenue_growth_acceleration": revenue_growth_acceleration,
            "operating_margin_change": operating_margin_change,
            "operating_roa_change": operating_roa_change,
            "operating_metric_median": float(np.median(values))
            if finite_all(values)
            else float("nan"),
            "operating_metrics_complete": finite_all(values),
        }

    @lru_cache(maxsize=None)
    def continuity(self, symbol: str, index: int) -> dict[str, Any]:
        medians = [
            self.operating(symbol, index - lag)["operating_metric_median"]
            for lag in range(3)
        ]
        evaluable = finite_all(medians)
        nonnegative_count = int(sum(value >= 0 for value in medians)) if evaluable else 0
        return {
            "continuity_evaluable": evaluable,
            "continuity_nonnegative_count": nonnegative_count,
            "continuity_pass": bool(evaluable and nonnegative_count >= 2),
        }

    @lru_cache(maxsize=None)
    def quality(self, symbol: str, index: int) -> dict[str, Any]:
        ttm_operating_now = self.ttm(symbol, index, "OPERATING_PROFIT_YTD")
        ttm_operating_prior = self.ttm(symbol, index - 4, "OPERATING_PROFIT_YTD")
        ttm_cash_now = self.ttm(symbol, index, "OPERATING_CASH_FLOW_YTD")
        ttm_cash_prior = self.ttm(symbol, index - 4, "OPERATING_CASH_FLOW_YTD")
        ttm_revenue_now = self.ttm(symbol, index, "OPERATING_REVENUE_YTD")
        ttm_revenue_prior = self.ttm(symbol, index - 4, "OPERATING_REVENUE_YTD")
        ttm_core_now = self.ttm(symbol, index, "CORE_PARENT_NET_PROFIT_YTD")
        ttm_core_prior = self.ttm(symbol, index - 4, "CORE_PARENT_NET_PROFIT_YTD")
        ttm_parent_now = self.ttm(symbol, index, "PARENT_NET_PROFIT_YTD")
        ttm_parent_prior = self.ttm(symbol, index - 4, "PARENT_NET_PROFIT_YTD")

        receivable_now = self.value(symbol, index, "ACCOUNTS_RECEIVABLE_END")
        receivable_prior = self.value(symbol, index - 4, "ACCOUNTS_RECEIVABLE_END")
        inventory_now = self.value(symbol, index, "INVENTORY_END")
        inventory_prior = self.value(symbol, index - 4, "INVENTORY_END")
        liabilities_now = self.value(symbol, index, "TOTAL_LIABILITIES_END")
        liabilities_prior = self.value(symbol, index - 4, "TOTAL_LIABILITIES_END")
        assets_now = self.value(symbol, index, "TOTAL_ASSETS_END")
        assets_prior = self.value(symbol, index - 4, "TOTAL_ASSETS_END")

        all_values = (
            ttm_operating_now,
            ttm_operating_prior,
            ttm_cash_now,
            ttm_cash_prior,
            ttm_revenue_now,
            ttm_revenue_prior,
            ttm_core_now,
            ttm_core_prior,
            ttm_parent_now,
            ttm_parent_prior,
            receivable_now,
            receivable_prior,
            inventory_now,
            inventory_prior,
            liabilities_now,
            liabilities_prior,
            assets_now,
            assets_prior,
        )
        denominators = (
            ttm_operating_now,
            ttm_operating_prior,
            ttm_revenue_now,
            ttm_revenue_prior,
            ttm_parent_now,
            ttm_parent_prior,
            assets_now,
            assets_prior,
        )
        evaluable = finite_all(all_values) and all(value > 0 for value in denominators)
        if not evaluable:
            return {
                "quality_evaluable": False,
                "cash_conversion_change": float("nan"),
                "working_capital_intensity_change": float("nan"),
                "core_profit_share_change": float("nan"),
                "liability_ratio_change": float("nan"),
                "quality_nondeteriorating_count": 0,
                "quality_pass": False,
            }

        cash_conversion_change = (
            ttm_cash_now / ttm_operating_now
            - ttm_cash_prior / ttm_operating_prior
        )
        working_capital_intensity_change = (
            (receivable_now + inventory_now) / ttm_revenue_now
            - (receivable_prior + inventory_prior) / ttm_revenue_prior
        )
        core_profit_share_change = (
            ttm_core_now / ttm_parent_now - ttm_core_prior / ttm_parent_prior
        )
        liability_ratio_change = (
            liabilities_now / assets_now - liabilities_prior / assets_prior
        )
        changes = (
            cash_conversion_change,
            working_capital_intensity_change,
            core_profit_share_change,
            liability_ratio_change,
        )
        evaluable = finite_all(changes)
        nondeteriorating_count = (
            int(cash_conversion_change >= 0)
            + int(working_capital_intensity_change <= 0)
            + int(core_profit_share_change >= 0)
            + int(liability_ratio_change <= 0)
            if evaluable
            else 0
        )
        return {
            "quality_evaluable": evaluable,
            "cash_conversion_change": cash_conversion_change,
            "working_capital_intensity_change": working_capital_intensity_change,
            "core_profit_share_change": core_profit_share_change,
            "liability_ratio_change": liability_ratio_change,
            "quality_nondeteriorating_count": nondeteriorating_count,
            "quality_pass": bool(evaluable and nondeteriorating_count >= 3),
        }


def build_financial_event_panel(
    dependency: pd.DataFrame, facts: pd.DataFrame
) -> pd.DataFrame:
    ready = dependency.loc[dependency["target_event_ready"]].copy()
    ready["event_publication_date"] = pd.to_datetime(
        ready["event_publication_date"], errors="raise"
    ).dt.normalize()
    ready["report_period"] = pd.to_datetime(ready["report_period"], errors="raise")
    ready["quarter_index"] = ready["report_period"].map(quarter_index)
    engine = FeatureEngine(build_quarter_records(facts))

    records: list[dict[str, Any]] = []
    for event in ready.to_dict("records"):
        symbol = str(event["ts_code"])
        index = int(event["quarter_index"])
        operating = engine.operating(symbol, index)
        continuity = engine.continuity(symbol, index)
        quality = engine.quality(symbol, index)
        records.append(
            {
                "announcement_id": str(event["announcement_id"]),
                "ts_code": symbol,
                "report_period": pd.Timestamp(event["report_period"]),
                "period_type": str(event["period_type"]),
                "event_publication_date": pd.Timestamp(event["event_publication_date"]),
                "publication_year": int(event["publication_year"]),
                "industry_l1": str(event["industry_l1"]),
                "industry_l1_code": str(event["industry_l1_code"]),
                "quarter_index": index,
                **operating,
                **continuity,
                **quality,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["event_publication_date", "ts_code", "announcement_id"], kind="stable"
    ).reset_index(drop=True)


def attach_market_clocks(events: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    output = events.copy()
    calendar = pd.DatetimeIndex(
        benchmark.loc[benchmark["date"].le(OUTCOME_CUTOFF), "date"]
        .drop_duplicates()
        .sort_values()
    )
    first_dates: list[pd.Timestamp | pd.NaT] = []
    formation_dates: list[pd.Timestamp | pd.NaT] = []
    formation_positions: list[int | None] = []
    for publication_date in output["event_publication_date"]:
        first_position = int(calendar.searchsorted(publication_date, side="right"))
        formation_position = first_position + 4
        if formation_position >= len(calendar):
            first_dates.append(pd.NaT)
            formation_dates.append(pd.NaT)
            formation_positions.append(None)
        else:
            first_dates.append(calendar[first_position])
            formation_dates.append(calendar[formation_position])
            formation_positions.append(formation_position)
    output["first_available_session"] = first_dates
    output["formation_session"] = formation_dates
    output["formation_calendar_position"] = pd.array(formation_positions, dtype="Int64")
    return output


def load_market_inputs(
    config: dict[str, Any], addendum_1: dict[str, Any], addendum_2: dict[str, Any], ready_symbols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benchmark = pd.read_parquet(project_path(addendum_1["replacement"]["path"]))
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="raise").dt.normalize()
    benchmark = benchmark.loc[
        benchmark["date"].between(pd.Timestamp("2016-08-12"), OUTCOME_CUTOFF)
    ].copy()
    benchmark = benchmark.sort_values("date", kind="stable").drop_duplicates("date")
    benchmark = benchmark.rename(columns={"close": "benchmark_total_return_close"})
    if benchmark["benchmark_total_return_close"].isna().any() or (
        benchmark["benchmark_total_return_close"] <= 0
    ).any():
        raise RuntimeError("H00300 全收益收盘序列存在缺失或非正值")

    first_source = addendum_2["benchmark_reaction_implementation"][
        "first_session_intraday_source"
    ]
    price_index = pd.read_parquet(project_path(first_source["path"]))
    price_index["date"] = pd.to_datetime(price_index["date"], errors="raise").dt.normalize()
    price_index = price_index.loc[
        price_index["date"].between(pd.Timestamp("2016-08-12"), OUTCOME_CUTOFF),
        ["date", "open", "close"],
    ].copy()
    price_index = price_index.rename(
        columns={"open": "price_index_open", "close": "price_index_close"}
    )
    if price_index.duplicated("date").any():
        raise RuntimeError("000300 价格指数日期不唯一")
    if price_index[["price_index_open", "price_index_close"]].isna().any().any():
        raise RuntimeError("000300 价格指数开盘或收盘缺失")
    if (
        price_index[["price_index_open", "price_index_close"]] <= 0
    ).any().any():
        raise RuntimeError("000300 价格指数开盘或收盘非正")

    training_path = project_path(config["inputs"]["stock_training"]["path"]).as_posix()
    holdout_path = project_path(config["inputs"]["stock_holdout"]["path"]).as_posix()
    connection = duckdb.connect()
    connection.execute("CREATE TEMP TABLE target_symbols(ts_code VARCHAR)")
    connection.executemany(
        "INSERT INTO target_symbols VALUES (?)", [(symbol,) for symbol in ready_symbols]
    )
    stock_query = f"""
        SELECT
            p.con_code,
            CAST(p.date AS DATE) AS date,
            p.total_return_open,
            p.total_return_close
        FROM read_parquet(
            ['{training_path}', '{holdout_path}'],
            union_by_name = true
        ) AS p
        INNER JOIN target_symbols AS t
            ON p.con_code = t.ts_code
        WHERE p.date >= DATE '2018-01-01'
          AND p.date <= DATE '2026-08-14'
    """
    stock = connection.execute(stock_query).df()
    connection.close()
    stock["date"] = pd.to_datetime(stock["date"], errors="raise").dt.normalize()
    if stock.duplicated(["con_code", "date"]).any():
        raise RuntimeError("股票总收益面板证券—日期键不唯一")
    invalid_stock = (
        stock[["total_return_open", "total_return_close"]].isna().any(axis=1)
        | stock["total_return_open"].le(0)
        | stock["total_return_close"].le(0)
    )
    if invalid_stock.any():
        raise RuntimeError(f"股票总收益价格存在无效行：{int(invalid_stock.sum())}")
    return benchmark, price_index, stock


def merge_stock_price(
    events: pd.DataFrame,
    stock: pd.DataFrame,
    event_date_column: str,
    stock_price_column: str,
    output_column: str,
) -> pd.DataFrame:
    prices = stock[["con_code", "date", stock_price_column]].rename(
        columns={
            "con_code": "ts_code",
            "date": event_date_column,
            stock_price_column: output_column,
        }
    )
    return events.merge(
        prices,
        on=["ts_code", event_date_column],
        how="left",
        validate="many_to_one",
    )


def attach_initial_reaction(
    events: pd.DataFrame,
    benchmark: pd.DataFrame,
    price_index: pd.DataFrame,
    stock: pd.DataFrame,
) -> pd.DataFrame:
    output = merge_stock_price(
        events,
        stock,
        "first_available_session",
        "total_return_open",
        "stock_total_return_open_first",
    )
    output = merge_stock_price(
        output,
        stock,
        "formation_session",
        "total_return_close",
        "stock_total_return_close_formation",
    )

    first_benchmark = benchmark[["date", "benchmark_total_return_close"]].rename(
        columns={
            "date": "first_available_session",
            "benchmark_total_return_close": "benchmark_close_first",
        }
    )
    formation_benchmark = benchmark[["date", "benchmark_total_return_close"]].rename(
        columns={
            "date": "formation_session",
            "benchmark_total_return_close": "benchmark_close_formation",
        }
    )
    first_price_index = price_index.rename(
        columns={
            "date": "first_available_session",
            "price_index_open": "benchmark_price_open_first",
            "price_index_close": "benchmark_price_close_first",
        }
    )
    output = output.merge(
        first_benchmark, on="first_available_session", how="left", validate="many_to_one"
    )
    output = output.merge(
        formation_benchmark, on="formation_session", how="left", validate="many_to_one"
    )
    output = output.merge(
        first_price_index,
        on="first_available_session",
        how="left",
        validate="many_to_one",
    )

    required = [
        "stock_total_return_open_first",
        "stock_total_return_close_formation",
        "benchmark_close_first",
        "benchmark_close_formation",
        "benchmark_price_open_first",
        "benchmark_price_close_first",
    ]
    valid = output[required].notna().all(axis=1) & output[required].gt(0).all(axis=1)
    output["initial_reaction_price_available"] = valid
    output["stock_initial_reaction"] = np.where(
        valid,
        output["stock_total_return_close_formation"]
        / output["stock_total_return_open_first"]
        - 1.0,
        np.nan,
    )
    output["benchmark_initial_reaction"] = np.where(
        valid,
        (
            output["benchmark_price_close_first"]
            / output["benchmark_price_open_first"]
        )
        * (
            output["benchmark_close_formation"] / output["benchmark_close_first"]
        )
        - 1.0,
        np.nan,
    )
    output["initial_reaction_excess"] = (
        output["stock_initial_reaction"] - output["benchmark_initial_reaction"]
    )
    return output


def trailing_midrank_percentiles(
    events: pd.DataFrame,
    value_columns: list[str],
    output_prefix: str,
) -> pd.DataFrame:
    output = events.copy()
    percentile_columns = [f"{output_prefix}_{column}_percentile" for column in value_columns]
    count_columns = [f"{output_prefix}_{column}_reference_count" for column in value_columns]
    for column in percentile_columns:
        output[column] = np.nan
    for column in count_columns:
        output[column] = 0

    grouped = output.groupby(["industry_l1_code", "period_type"], sort=False)
    for _, indices in grouped.groups.items():
        group = output.loc[list(indices)].sort_values(
            ["formation_session", "announcement_id"], kind="stable"
        )
        dates = group["formation_session"].to_numpy(dtype="datetime64[ns]")
        for row_position, row_index in enumerate(group.index):
            current_date = dates[row_position]
            if np.isnat(current_date):
                continue
            lower_date = current_date - np.timedelta64(TRAILING_DAYS, "D")
            time_mask = (dates < current_date) & (dates >= lower_date)
            for value_column, percentile_column, count_column in zip(
                value_columns, percentile_columns, count_columns, strict=True
            ):
                current_value = output.at[row_index, value_column]
                if not finite(current_value):
                    continue
                reference_values = group.loc[time_mask, value_column].to_numpy(dtype=float)
                reference_values = reference_values[np.isfinite(reference_values)]
                reference_count = int(reference_values.size)
                output.at[row_index, count_column] = reference_count
                if reference_count < MINIMUM_REFERENCE_COUNT:
                    continue
                less_count = int(np.sum(reference_values < float(current_value)))
                equal_count = int(np.sum(reference_values == float(current_value)))
                output.at[row_index, percentile_column] = (
                    less_count + 0.5 * equal_count
                ) / reference_count
    return output


def attach_score(events: pd.DataFrame) -> pd.DataFrame:
    output = trailing_midrank_percentiles(
        events, list(OPERATING_COLUMNS), "operating"
    )
    operating_percentile_columns = [
        f"operating_{column}_percentile" for column in OPERATING_COLUMNS
    ]
    operating_references_ready = output[
        [f"operating_{column}_reference_count" for column in OPERATING_COLUMNS]
    ].ge(MINIMUM_REFERENCE_COUNT).all(axis=1)
    operating_values_ready = output[operating_percentile_columns].notna().all(axis=1)
    output["operating_reference_gate_pass"] = (
        operating_references_ready & operating_values_ready
    )
    output["operating_improvement_strength_percentile"] = np.where(
        output["continuity_pass"] & output["operating_reference_gate_pass"],
        output[operating_percentile_columns].median(axis=1),
        np.nan,
    )

    output = trailing_midrank_percentiles(
        output, ["initial_reaction_excess"], "reaction"
    )
    output["reaction_reference_gate_pass"] = (
        output["reaction_initial_reaction_excess_reference_count"].ge(
            MINIMUM_REFERENCE_COUNT
        )
        & output["reaction_initial_reaction_excess_percentile"].notna()
    )
    score_valid = (
        output["continuity_pass"]
        & output["quality_pass"]
        & output["operating_reference_gate_pass"]
        & output["reaction_reference_gate_pass"]
        & output["initial_reaction_price_available"]
    )
    output["score"] = np.where(
        score_valid,
        output["operating_improvement_strength_percentile"]
        - output["reaction_initial_reaction_excess_percentile"],
        np.nan,
    )
    output["score_available"] = output["score"].notna()
    output["score_group"] = pd.array([pd.NA] * len(output), dtype="Int64")
    for _, indices in output.loc[output["score_available"]].groupby(
        ["publication_year", "period_type"], sort=True
    ).groups.items():
        ordered = output.loc[list(indices)].sort_values(
            ["score", "announcement_id"], kind="stable"
        )
        count = len(ordered)
        # 五个等数量档至少需要五个事件；不足五个时不能把单个事件误标为 Q1。
        if count < 5:
            continue
        groups = np.floor(np.arange(count) * 5 / count).astype(int) + 1
        output.loc[ordered.index, "score_group"] = groups
    output["score_group_available"] = output["score_group"].notna()
    return output


def attach_outcomes(
    events: pd.DataFrame, benchmark: pd.DataFrame, stock: pd.DataFrame
) -> pd.DataFrame:
    output = events.copy()
    calendar = pd.DatetimeIndex(
        benchmark.loc[benchmark["date"].le(OUTCOME_CUTOFF), "date"].sort_values()
    )
    benchmark_close = benchmark.set_index("date")["benchmark_total_return_close"]
    for horizon in (60, 120):
        endpoint_column = f"outcome_{horizon}d_session"
        endpoint_dates: list[pd.Timestamp | pd.NaT] = []
        for formation_position, score_available in zip(
            output["formation_calendar_position"], output["score_available"], strict=True
        ):
            if not score_available or pd.isna(formation_position):
                endpoint_dates.append(pd.NaT)
                continue
            endpoint_position = int(formation_position) + horizon
            endpoint_dates.append(
                calendar[endpoint_position]
                if endpoint_position < len(calendar)
                else pd.NaT
            )
        output[endpoint_column] = endpoint_dates
        output[f"outcome_{horizon}d_mature"] = output[endpoint_column].notna()
        output = merge_stock_price(
            output,
            stock,
            endpoint_column,
            "total_return_close",
            f"stock_total_return_close_{horizon}d",
        )
        output[f"benchmark_total_return_close_{horizon}d"] = output[
            endpoint_column
        ].map(benchmark_close)
        price_valid = (
            output[f"outcome_{horizon}d_mature"]
            & output["stock_total_return_close_formation"].notna()
            & output[f"stock_total_return_close_{horizon}d"].notna()
            & output["benchmark_close_formation"].notna()
            & output[f"benchmark_total_return_close_{horizon}d"].notna()
            & output["stock_total_return_close_formation"].gt(0)
            & output[f"stock_total_return_close_{horizon}d"].gt(0)
            & output["benchmark_close_formation"].gt(0)
            & output[f"benchmark_total_return_close_{horizon}d"].gt(0)
        )
        output[f"outcome_{horizon}d_price_available"] = price_valid
        output[f"stock_total_return_{horizon}d"] = np.where(
            price_valid,
            output[f"stock_total_return_close_{horizon}d"]
            / output["stock_total_return_close_formation"]
            - 1.0,
            np.nan,
        )
        output[f"benchmark_total_return_{horizon}d"] = np.where(
            price_valid,
            output[f"benchmark_total_return_close_{horizon}d"]
            / output["benchmark_close_formation"]
            - 1.0,
            np.nan,
        )
        output[f"relative_return_{horizon}d"] = (
            output[f"stock_total_return_{horizon}d"]
            - output[f"benchmark_total_return_{horizon}d"]
        )
    return output


def q5_minus_q1(frame: pd.DataFrame, outcome_column: str) -> float:
    q1 = frame.loc[frame["score_group"].eq(1), outcome_column].dropna()
    q5 = frame.loc[frame["score_group"].eq(5), outcome_column].dropna()
    if q1.empty or q5.empty:
        return float("nan")
    return float(q5.mean() - q1.mean())


def issuer_cluster_bootstrap(
    frame: pd.DataFrame, outcome_column: str
) -> dict[str, Any]:
    tails = frame.loc[frame["score_group"].isin([1, 5])].copy()
    issuers = sorted(tails["ts_code"].unique())
    if not issuers:
        return {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "valid_repetitions": 0,
            "lower_95": None,
            "median": None,
            "upper_95": None,
            "positive_probability": None,
        }
    aggregates = tails.groupby(["ts_code", "score_group"])[outcome_column].agg(
        ["sum", "count"]
    )
    q1_sum = np.array(
        [aggregates.loc[(issuer, 1), "sum"] if (issuer, 1) in aggregates.index else 0.0 for issuer in issuers],
        dtype=float,
    )
    q1_count = np.array(
        [aggregates.loc[(issuer, 1), "count"] if (issuer, 1) in aggregates.index else 0.0 for issuer in issuers],
        dtype=float,
    )
    q5_sum = np.array(
        [aggregates.loc[(issuer, 5), "sum"] if (issuer, 5) in aggregates.index else 0.0 for issuer in issuers],
        dtype=float,
    )
    q5_count = np.array(
        [aggregates.loc[(issuer, 5), "count"] if (issuer, 5) in aggregates.index else 0.0 for issuer in issuers],
        dtype=float,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences: list[float] = []
    issuer_count = len(issuers)
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.integers(0, issuer_count, size=issuer_count)
        sampled_q1_count = float(q1_count[sampled].sum())
        sampled_q5_count = float(q5_count[sampled].sum())
        if sampled_q1_count <= 0 or sampled_q5_count <= 0:
            continue
        differences.append(
            float(q5_sum[sampled].sum() / sampled_q5_count)
            - float(q1_sum[sampled].sum() / sampled_q1_count)
        )
    if not differences:
        return {
            "repetitions": BOOTSTRAP_REPETITIONS,
            "valid_repetitions": 0,
            "lower_95": None,
            "median": None,
            "upper_95": None,
            "positive_probability": None,
        }
    array = np.asarray(differences)
    return {
        "repetitions": BOOTSTRAP_REPETITIONS,
        "valid_repetitions": int(array.size),
        "lower_95": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "upper_95": float(np.quantile(array, 0.975)),
        "positive_probability": float(np.mean(array > 0)),
    }


def evaluate_horizon(events: pd.DataFrame, horizon: int) -> dict[str, Any]:
    outcome_column = f"relative_return_{horizon}d"
    frame = events.loc[
        events["score_group_available"] & events[outcome_column].notna()
    ].copy()
    group_summary = (
        frame.groupby("score_group", observed=True)[outcome_column]
        .agg(["count", "mean", "median"])
        .reindex([1, 2, 3, 4, 5])
    )
    group_records = [
        {
            "score_group": int(group),
            "count": int(row["count"]) if finite(row["count"]) else 0,
            "mean": float(row["mean"]) if finite(row["mean"]) else None,
            "median": float(row["median"]) if finite(row["median"]) else None,
        }
        for group, row in group_summary.iterrows()
    ]
    means = group_summary["mean"].to_numpy(dtype=float)
    adjacent_differences = np.diff(means) if np.isfinite(means).all() else np.array([])
    adjacent_nonnegative_count = int(np.sum(adjacent_differences >= 0))
    adjacent_gate = bool(
        adjacent_differences.size == 4 and adjacent_nonnegative_count >= 3
    )
    overall_q5_q1 = q5_minus_q1(frame, outcome_column)
    q5_q1_gate = bool(finite(overall_q5_q1) and overall_q5_q1 > 0)

    if len(frame) >= 3 and frame["score"].nunique() > 1:
        spearman_result = spearmanr(frame["score"], frame[outcome_column], nan_policy="omit")
        spearman_correlation = float(spearman_result.statistic)
        spearman_p_value = float(spearman_result.pvalue)
    else:
        spearman_correlation = float("nan")
        spearman_p_value = float("nan")
    spearman_gate = bool(finite(spearman_correlation) and spearman_correlation > 0)

    early_mask = frame["event_publication_date"].between(
        pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")
    )
    late_mask = frame["event_publication_date"].between(
        pd.Timestamp("2022-01-01"), pd.Timestamp("2026-08-14")
    )
    early_q5_q1 = q5_minus_q1(frame.loc[early_mask], outcome_column)
    late_q5_q1 = q5_minus_q1(frame.loc[late_mask], outcome_column)
    chronological_gate = bool(
        finite_all((early_q5_q1, late_q5_q1))
        and early_q5_q1 > 0
        and late_q5_q1 > 0
    )

    industry_rows: list[dict[str, Any]] = []
    for (industry_code, industry_name), industry_frame in frame.groupby(
        ["industry_l1_code", "industry_l1"], sort=True
    ):
        if len(industry_frame) < 30:
            continue
        difference = q5_minus_q1(industry_frame, outcome_column)
        industry_rows.append(
            {
                "industry_l1_code": str(industry_code),
                "industry_l1": str(industry_name),
                "event_count": int(len(industry_frame)),
                "q5_minus_q1": difference,
                "positive": bool(finite(difference) and difference > 0),
            }
        )
    positive_industry_count = int(sum(row["positive"] for row in industry_rows))
    eligible_industry_count = len(industry_rows)
    industry_gate = bool(
        eligible_industry_count > 0
        and positive_industry_count > eligible_industry_count / 2
    )

    leave_one_year_rows: list[dict[str, Any]] = []
    for year in range(2018, 2027):
        difference = q5_minus_q1(
            frame.loc[frame["publication_year"].ne(year)], outcome_column
        )
        leave_one_year_rows.append(
            {
                "deleted_publication_year": year,
                "q5_minus_q1": difference,
                "positive": bool(finite(difference) and difference > 0),
            }
        )
    leave_one_year_gate = bool(
        leave_one_year_rows and all(row["positive"] for row in leave_one_year_rows)
    )

    leave_one_industry_rows: list[dict[str, Any]] = []
    for industry in industry_rows:
        difference = q5_minus_q1(
            frame.loc[
                frame["industry_l1_code"].ne(industry["industry_l1_code"])
            ],
            outcome_column,
        )
        leave_one_industry_rows.append(
            {
                "deleted_industry_l1_code": industry["industry_l1_code"],
                "deleted_industry_l1": industry["industry_l1"],
                "q5_minus_q1": difference,
                "positive": bool(finite(difference) and difference > 0),
            }
        )
    leave_one_industry_gate = bool(
        leave_one_industry_rows
        and all(row["positive"] for row in leave_one_industry_rows)
    )

    ordered = frame.sort_values([outcome_column, "announcement_id"], kind="stable")
    # 小样本若向下取整会删除零行，使“极端值依赖”测试名存实亡；
    # 只要两端各可保留至少一行，就固定至少删除一行。
    tail_count = (
        max(1, int(math.floor(len(ordered) * 0.01))) if len(ordered) >= 3 else 0
    )
    trimmed = (
        ordered.iloc[tail_count : len(ordered) - tail_count]
        if tail_count > 0
        else ordered
    )
    trimmed_q5_q1 = q5_minus_q1(trimmed, outcome_column)
    trimmed_gate = bool(finite(trimmed_q5_q1) and trimmed_q5_q1 > 0)
    q5_median_series = frame.loc[frame["score_group"].eq(5), outcome_column]
    q5_median = float(q5_median_series.median()) if not q5_median_series.empty else float("nan")
    q5_median_gate = bool(finite(q5_median) and q5_median > 0)

    gates = {
        "adjacent_monotonicity": adjacent_gate,
        "q5_minus_q1_positive": q5_q1_gate,
        "spearman_positive": spearman_gate,
        "early_and_late_positive": chronological_gate,
        "strict_majority_industries_positive": industry_gate,
        "leave_one_year_out_all_positive": leave_one_year_gate,
        "leave_one_industry_out_all_positive": leave_one_industry_gate,
        "trimmed_q5_minus_q1_positive": trimmed_gate,
        "q5_median_positive": q5_median_gate,
    }
    return {
        "horizon_market_sessions": horizon,
        "analysis_event_count": int(len(frame)),
        "group_summary": group_records,
        "adjacent_differences": adjacent_differences.tolist(),
        "adjacent_nonnegative_count": adjacent_nonnegative_count,
        "q5_minus_q1": overall_q5_q1,
        "spearman_correlation": spearman_correlation,
        "spearman_p_value": spearman_p_value,
        "early_q5_minus_q1": early_q5_q1,
        "late_q5_minus_q1": late_q5_q1,
        "eligible_industry_count": eligible_industry_count,
        "positive_industry_count": positive_industry_count,
        "industry_results": industry_rows,
        "leave_one_year_out": leave_one_year_rows,
        "leave_one_industry_out": leave_one_industry_rows,
        "trim_tail_event_count_each_side": tail_count,
        "trimmed_q5_minus_q1": trimmed_q5_q1,
        "q5_median": q5_median,
        "issuer_cluster_bootstrap": issuer_cluster_bootstrap(frame, outcome_column),
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }


def coverage_table(
    dependency: pd.DataFrame, group_columns: list[str]
) -> list[dict[str, Any]]:
    grouped = (
        dependency.groupby(group_columns, dropna=False)["target_event_ready"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "target_event_count", "sum": "ready_event_count"})
    )
    grouped["unready_event_count"] = (
        grouped["target_event_count"] - grouped["ready_event_count"]
    )
    grouped["ready_ratio"] = (
        grouped["ready_event_count"] / grouped["target_event_count"]
    )
    return json_safe(grouped.to_dict("records"))


def attrition_waterfall(events: pd.DataFrame) -> list[dict[str, Any]]:
    masks: list[tuple[str, pd.Series]] = []
    current = pd.Series(True, index=events.index)
    masks.append(("冻结完整依赖事件", current.copy()))
    current &= events["operating_metrics_complete"]
    masks.append(("当前三项经营改善可计算", current.copy()))
    current &= events["continuity_evaluable"]
    masks.append(("三次连续报告可评价", current.copy()))
    current &= events["continuity_pass"]
    masks.append(("连续性门通过", current.copy()))
    current &= events["quality_evaluable"]
    masks.append(("四项质量变化可评价", current.copy()))
    current &= events["quality_pass"]
    masks.append(("质量门通过", current.copy()))
    current &= events["initial_reaction_price_available"]
    masks.append(("初始价格反应可计算", current.copy()))
    current &= events["operating_reference_gate_pass"]
    masks.append(("三项经营改善历史参照均不少于30", current.copy()))
    current &= events["reaction_reference_gate_pass"]
    masks.append(("初始反应历史参照不少于30", current.copy()))
    current &= events["score_available"]
    masks.append(("唯一分数可形成", current.copy()))
    current &= events["score_group_available"]
    masks.append(("年度×报告类型单元可形成完整五档", current.copy()))
    total = len(events)
    return [
        {
            "stage": stage,
            "event_count": int(mask.sum()),
            "share_of_frozen_ready_sample": float(mask.mean()) if total else None,
        }
        for stage, mask in masks
    ]


def fmt_percent(value: Any, digits: int = 2) -> str:
    return "NA" if not finite(value) else f"{float(value) * 100:.{digits}f}%"


def fmt_number(value: Any, digits: int = 4) -> str:
    return "NA" if not finite(value) else f"{float(value):.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return lines


def render_report(result: dict[str, Any]) -> str:
    status = result["status"]
    counts = result["counts"]
    lines = [
        "# 沪深300点时基本面反应不足：不完整覆盖条件样本分析 V1",
        "",
        f"- 条件分析状态：`{status}`",
        "- 原始 V1 状态：`BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE`（保持不变）",
        "- 研究边界：当前完整官方事实事件的第一阶段机制检验；不构造组合、不计算策略夏普、不授权交易",
        "- 仓位影响：`0`；交易状态：`NO_TRADE`",
        "",
        "## 结论摘要",
        "",
        result["conclusion"],
        "",
        "## 样本与流失",
        "",
        f"冻结目标事件共 8,223 个；当前完整依赖事件 {counts['frozen_ready_events']:,} 个，覆盖率 {fmt_percent(counts['frozen_ready_ratio'], 3)}；未完整事件 971 个未填补、未读取其结果。",
        "",
    ]
    lines.extend(
        markdown_table(
            ["阶段", "事件数", "占7,252比重"],
            [
                [row["stage"], f"{row['event_count']:,}", fmt_percent(row["share_of_frozen_ready_sample"], 2)]
                for row in result["attrition_waterfall"]
            ],
        )
    )
    lines.extend(["", "### 初始价格反应覆盖", ""])
    lines.extend(
        markdown_table(
            ["公告年度", "完整事实事件", "初始反应可计算", "覆盖率"],
            [
                [
                    row["publication_year"],
                    f"{row['ready_event_count']:,}",
                    f"{row['reaction_available_event_count']:,}",
                    fmt_percent(row["reaction_available_ratio"], 2),
                ]
                for row in result["market_reaction_coverage_by_year"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "冻结股票面板在时间×证券维度不是完整矩形：2018—2022 年主要覆盖训练证券，2024—2026 年主要覆盖留出证券。缺失价格严格记为 `NO_VIEW`，没有另换数据源；因此最终机制样本还存在明显的价格选择边界。",
            "",
            "### 可形成完整五档的年度×报告类型单元",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["公告年度", "报告类型", "可入组事件", "60日结果", "120日结果"],
            [
                [
                    row["publication_year"],
                    row["period_type"],
                    row["group_available_event_count"],
                    row["outcome_60d_event_count"],
                    row["outcome_120d_event_count"],
                ]
                for row in result["groupable_cells"]
            ],
        )
    )
    lines.extend(
        [
            "",
            "完整五档只来自上述单元；其他年度×报告类型因可评分事件少于 5 个而严格 `NO_VIEW`。因此早晚期方向实际上主要比较 2021Q1 与 2023 年事件，不能解释为完整的多周期复制。",
            "",
            "## 固定机制检验",
            "",
        ]
    )
    for horizon_result in result["horizons"]:
        horizon = horizon_result["horizon_market_sessions"]
        bootstrap = horizon_result["issuer_cluster_bootstrap"]
        lines.extend(
            [
                f"### {horizon} 个共同交易日",
                "",
                f"可评价事件：{horizon_result['analysis_event_count']:,}；Q5-Q1 均值差：{fmt_percent(horizon_result['q5_minus_q1'])}；Spearman：{fmt_number(horizon_result['spearman_correlation'])}（p={fmt_number(horizon_result['spearman_p_value'])}）。",
                "",
            ]
        )
        lines.extend(
            markdown_table(
                ["分组", "事件数", "平均相对收益", "中位相对收益"],
                [
                    [
                        f"Q{row['score_group']}",
                        f"{row['count']:,}",
                        fmt_percent(row["mean"]),
                        fmt_percent(row["median"]),
                    ]
                    for row in horizon_result["group_summary"]
                ],
            )
        )
        lines.extend(
            [
                "",
                f"早期 Q5-Q1：{fmt_percent(horizon_result['early_q5_minus_q1'])}；晚期：{fmt_percent(horizon_result['late_q5_minus_q1'])}。合格行业 {horizon_result['eligible_industry_count']} 个，其中正向 {horizon_result['positive_industry_count']} 个。",
                "",
                f"发行人分组自助法 95% 区间：[{fmt_percent(bootstrap['lower_95'])}, {fmt_percent(bootstrap['upper_95'])}]；正值比例 {fmt_percent(bootstrap['positive_probability'])}。该区间不是通过门。",
                "",
            ]
        )
        lines.extend(
            markdown_table(
                ["冻结门", "结果"],
                [
                    [gate, "PASS" if passed else "FAIL"]
                    for gate, passed in horizon_result["gates"].items()
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 覆盖偏差边界",
            "",
            "当前结论只对 7,252 个官方事实依赖完整事件中的可评分子样本成立。原整体覆盖率和 2019 年覆盖率仍未达到冻结门；因此即使条件样本全部门通过，也不能推断 971 个缺失事件方向相同，不能进入第二阶段组合。",
            "",
            "## 状态边界",
            "",
            "- 原始数据准入：`BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE`",
            f"- 本次条件分析：`{status}`",
            "- 第二阶段：`NOT_AUTHORIZED`",
            "- `POSITION_IMPACT=0`",
            "- `NO_TRADE`",
            "",
        ]
    )
    return "\n".join(lines)


def selected_event_panel(events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "announcement_id",
        "ts_code",
        "report_period",
        "period_type",
        "event_publication_date",
        "publication_year",
        "industry_l1",
        "industry_l1_code",
        "first_available_session",
        "formation_session",
        "revenue_growth_acceleration",
        "operating_margin_change",
        "operating_roa_change",
        "continuity_evaluable",
        "continuity_nonnegative_count",
        "continuity_pass",
        "cash_conversion_change",
        "working_capital_intensity_change",
        "core_profit_share_change",
        "liability_ratio_change",
        "quality_evaluable",
        "quality_nondeteriorating_count",
        "quality_pass",
        "initial_reaction_price_available",
        "stock_initial_reaction",
        "benchmark_initial_reaction",
        "initial_reaction_excess",
        "operating_revenue_growth_acceleration_percentile",
        "operating_operating_margin_change_percentile",
        "operating_operating_roa_change_percentile",
        "operating_improvement_strength_percentile",
        "reaction_initial_reaction_excess_percentile",
        "score",
        "score_available",
        "score_group",
        "score_group_available",
        "outcome_60d_session",
        "outcome_60d_mature",
        "outcome_60d_price_available",
        "stock_total_return_60d",
        "benchmark_total_return_60d",
        "relative_return_60d",
        "outcome_120d_session",
        "outcome_120d_mature",
        "outcome_120d_price_available",
        "stock_total_return_120d",
        "benchmark_total_return_120d",
        "relative_return_120d",
    ]
    return events[columns].copy()


def main() -> int:
    started_at = datetime.now(TIMEZONE)
    started_monotonic = time.monotonic()
    print("[1/7] 核验冻结协议、补充清单和全部输入哈希……", flush=True)
    config, addendum_1, addendum_2 = verify_protocol_chain()
    artifacts = ArtifactPaths(
        event_panel=project_path(config["artifacts"]["event_panel"]),
        result_json=project_path(config["artifacts"]["result_json"]),
        result_markdown=project_path(config["artifacts"]["result_markdown"]),
        run_receipt=project_path(config["artifacts"]["run_receipt"]),
    )

    print("[2/7] 读取 7,252 个冻结事件及官方 PDF 财务事实……", flush=True)
    dependency, facts, _requirements = load_financial_inputs(config)
    events = build_financial_event_panel(dependency, facts)

    print("[3/7] 读取冻结股票总收益面板和沪深300基准……", flush=True)
    ready_symbols = sorted(events["ts_code"].unique())
    benchmark, price_index, stock = load_market_inputs(
        config, addendum_1, addendum_2, ready_symbols
    )
    events = attach_market_clocks(events, benchmark)
    events = attach_initial_reaction(events, benchmark, price_index, stock)

    print("[4/7] 计算连续经营改善、质量确认、历史百分位和唯一分数……", flush=True)
    events = attach_score(events)

    print("[5/7] 计算形成后 60/120 个共同交易日相对全收益……", flush=True)
    events = attach_outcomes(events, benchmark, stock)

    print("[6/7] 执行冻结门、逐年/逐行业删除和发行人分组自助法……", flush=True)
    horizons = [evaluate_horizon(events, horizon) for horizon in (60, 120)]
    all_horizons_pass = all(item["all_gates_pass"] for item in horizons)
    status = (
        config["terminal_statuses"]["conditional_pass"]
        if all_horizons_pass
        else config["terminal_statuses"]["conditional_fail"]
    )
    score_count = int(events["score_available"].sum())
    group_count = int(events["score_group_available"].sum())
    count_60 = int(events["relative_return_60d"].notna().sum())
    count_120 = int(events["relative_return_120d"].notna().sum())
    if all_horizons_pass:
        conclusion = (
            "在当前不完整覆盖的条件样本中，冻结的基本面改善—初始反应不足机制同时通过 60 日和 120 日全部方向门；"
            "但原覆盖门仍失败，因此该结果不授权第二阶段，只能作为条件证据保留。"
        )
    else:
        failed_horizons = [
            str(item["horizon_market_sessions"])
            for item in horizons
            if not item["all_gates_pass"]
        ]
        conclusion = (
            "在当前不完整覆盖的条件样本中，冻结机制没有通过全部验收门；"
            f"失败期限为 {'、'.join(failed_horizons)} 个共同交易日。"
            "按冻结规则不得改指标、窗口、分组、年份或行业营救。"
        )

    result = {
        "study_id": STUDY_ID,
        "script_version": SCRIPT_VERSION,
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "status": status,
        "conclusion": conclusion,
        "original_v1_status": "BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE",
        "original_v1_status_rewritten": False,
        "authorization": "USER_AUTHORIZED_INCOMPLETE_COVERAGE_CONDITIONAL_ANALYSIS",
        "counts": {
            "target_events": int(len(dependency)),
            "frozen_ready_events": int(len(events)),
            "frozen_unready_events": int((~dependency["target_event_ready"]).sum()),
            "frozen_ready_ratio": float(dependency["target_event_ready"].mean()),
            "ready_symbols": int(events["ts_code"].nunique()),
            "stock_price_symbols_available": int(stock["con_code"].nunique()),
            "score_available_events": score_count,
            "score_group_available_events": group_count,
            "outcome_60d_available_events": count_60,
            "outcome_120d_available_events": count_120,
        },
        "coverage": {
            "by_publication_year": coverage_table(dependency, ["publication_year"]),
            "by_period_type": coverage_table(dependency, ["period_type"]),
            "by_industry": coverage_table(
                dependency, ["industry_l1_code", "industry_l1"]
            ),
        },
        "attrition_waterfall": attrition_waterfall(events),
        "feature_summary": {
            "continuity_evaluable": int(events["continuity_evaluable"].sum()),
            "continuity_pass": int(events["continuity_pass"].sum()),
            "quality_evaluable": int(events["quality_evaluable"].sum()),
            "quality_pass": int(events["quality_pass"].sum()),
            "initial_reaction_available": int(
                events["initial_reaction_price_available"].sum()
            ),
            "operating_reference_gate_pass": int(
                events["operating_reference_gate_pass"].sum()
            ),
            "reaction_reference_gate_pass": int(
                events["reaction_reference_gate_pass"].sum()
            ),
            "score_median": float(events["score"].median())
            if score_count
            else None,
        },
        "market_reaction_coverage_by_year": [
            {
                "publication_year": int(year),
                "ready_event_count": int(len(frame)),
                "reaction_available_event_count": int(
                    frame["initial_reaction_price_available"].sum()
                ),
                "reaction_available_ratio": float(
                    frame["initial_reaction_price_available"].mean()
                ),
            }
            for year, frame in events.groupby("publication_year", sort=True)
        ],
        "groupable_cells": [
            {
                "publication_year": int(year),
                "period_type": str(period_type),
                "group_available_event_count": int(len(frame)),
                "outcome_60d_event_count": int(frame["relative_return_60d"].notna().sum()),
                "outcome_120d_event_count": int(frame["relative_return_120d"].notna().sum()),
            }
            for (year, period_type), frame in events.loc[
                events["score_group_available"]
            ].groupby(["publication_year", "period_type"], sort=True)
        ],
        "horizons": horizons,
        "selection_bias_boundary": {
            "unready_event_outcomes_read": False,
            "missing_fact_imputation": False,
            "inverse_probability_weighting": False,
            "extrapolation_to_unready_events": False,
        },
        "governance": {
            "phase_2_authorized": False,
            "portfolio_constructed": False,
            "strategy_sharpe_calculated": False,
            "position_impact": 0,
            "trade_status": "NO_TRADE",
            "parameter_search_performed": False,
            "post_result_rescue_performed": False,
        },
    }

    print("[7/7] 写入事件面板、严格 JSON、中文报告和运行收据……", flush=True)
    artifacts.event_panel.parent.mkdir(parents=True, exist_ok=True)
    panel = selected_event_panel(events)
    panel.to_parquet(artifacts.event_panel, index=False)
    write_json_strict(artifacts.result_json, result)
    artifacts.result_markdown.parent.mkdir(parents=True, exist_ok=True)
    artifacts.result_markdown.write_text(render_report(result), encoding="utf-8")

    completed_at = datetime.now(TIMEZONE)
    output_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (
            artifacts.event_panel,
            artifacts.result_json,
            artifacts.result_markdown,
        )
    }
    receipt = {
        "study_id": STUDY_ID,
        "script_version": SCRIPT_VERSION,
        "script_path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": time.monotonic() - started_monotonic,
        "status": status,
        "protocol_chain": {
            "main_manifest": {
                "path": str(MAIN_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(MAIN_MANIFEST_PATH),
            },
            "schema_addendum_manifest": {
                "path": str(ADDENDUM_1_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(ADDENDUM_1_MANIFEST_PATH),
            },
            "input_completeness_addendum_manifest": {
                "path": str(ADDENDUM_2_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(ADDENDUM_2_MANIFEST_PATH),
            },
        },
        "market_price_read": True,
        "future_return_read": True,
        "return_evaluation": "USER_AUTHORIZED_CONDITIONAL_INCOMPLETE_COVERAGE_SAMPLE",
        "unready_event_outcomes_read": False,
        "original_v1_receipt_rewritten": False,
        "original_v1_status": "BLOCKED_OFFICIAL_PDF_FACT_COVERAGE_BELOW_FROZEN_GATE",
        "frozen_ready_event_count": int(len(events)),
        "score_available_event_count": score_count,
        "score_group_available_event_count": group_count,
        "implementation_correction": "YEAR_PERIOD_CELLS_WITH_FEWER_THAN_FIVE_SCORED_EVENTS_ARE_NO_VIEW_FOR_FIVE_EQUAL_COUNT_GROUPS",
        "outputs": output_hashes,
        "phase_2_authorized": False,
        "position_impact": 0,
        "trade_status": "NO_TRADE",
    }
    write_json_strict(artifacts.run_receipt, receipt)
    print(
        f"完成：状态={status}，可评分事件={score_count}，60日={count_60}，120日={count_120}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
