"""运行并冻结 ORJ V2 非盲历史执行验证。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_hs_official_report_orj_net_excess_execution_v2 import (
    apply_same_security_overlap,
    attach_benchmark_returns,
    attach_execution_economics,
    build_annual_summary,
    build_decision_date_metrics,
    canonical_hash,
    classify_execution_frame,
    descriptive_trade_statistics,
    evaluate_gate,
    json_ready,
    load_config,
    select_high_orj,
    sha256_file,
)


EXPECTED_MANIFEST_STATUS = "FROZEN_ORJ_NET_EXCESS_EXECUTION_V2_BEFORE_REAL_RUN"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def project_path(relative: str) -> Path:
    """把配置中的正斜杠相对路径映射到 Windows 项目路径。"""

    return ROOT / Path(relative.replace("/", os.sep))


def sql_path(path: Path) -> str:
    """生成 DuckDB 可安全读取的绝对路径。"""

    return path.resolve().as_posix().replace("'", "''")


def file_record(path: Path) -> dict[str, Any]:
    """记录文件大小和 SHA-256。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    """以 UTF-8 严格 JSON 原子写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text_atomic(path: Path, text: str) -> None:
    """原子写入 UTF-8 文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子写入 CSV。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    """原子写入 ZSTD Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def protocol_file_groups(config: Mapping[str, Any]) -> dict[str, list[str]]:
    """返回协议冻结对象、父研究引用和真实输入。"""

    return {
        "frozen_files": [
            "config/a_share_hs_official_report_orj_net_excess_execution_v2.yaml",
            "research/a_share_hs_official_report_orj_net_excess_execution_v2.py",
            "scripts/run_a_share_hs_official_report_orj_net_excess_execution_v2.py",
            "tests/test_a_share_hs_official_report_orj_net_excess_execution_v2.py",
        ],
        "reference_files": [
            str(config["parent"]["config"]),
            str(config["parent"]["correction_manifest"]),
        ],
        "input_files": [
            str(config["parent"]["signal_ledger"]),
            str(config["inputs"]["unified_daily_market"]),
            str(config["inputs"]["industry_peer_return_panel"]),
            str(config["inputs"]["security_status_intervals"]),
            str(config["inputs"]["csi300_price_index"]),
            str(config["inputs"]["csi300_total_return_index"]),
        ],
    }


def freeze_protocol(config: Mapping[str, Any]) -> Path:
    """在真实执行结果计算前一次性冻结协议、代码、测试和输入。"""

    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise FileExistsError(f"协议清单已存在，拒绝覆盖：{manifest_path}")
    groups = protocol_file_groups(config)
    body: dict[str, Any] = {
        "status": EXPECTED_MANIFEST_STATUS,
        "study_id": config["study"]["study_id"],
        "version": config["study"]["version"],
        "frozen_at": datetime.now(TIMEZONE).isoformat(),
        "research_scope": config["study"]["research_scope"],
        "parent_result_was_reviewed": True,
        "real_v2_execution_outcomes_calculated_before_freeze": False,
        "future_label_selection_forbidden": "analysis_eligible",
        "result_rescue_by_parameter_change_forbidden": True,
    }
    for group, relatives in groups.items():
        body[group] = {
            relative: file_record(project_path(relative)) for relative in relatives
        }
    manifest = dict(body)
    manifest["manifest_content_sha256"] = canonical_hash(body)
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def verify_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """拒绝清单、代码、引用或输入发生任何漂移。"""

    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ORJ V2 协议尚未冻结；先运行 --freeze-protocol：{manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != EXPECTED_MANIFEST_STATUS:
        raise RuntimeError("ORJ V2 协议清单状态不允许运行")
    body = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    if canonical_hash(body) != manifest.get("manifest_content_sha256"):
        raise RuntimeError("ORJ V2 协议清单内容哈希不匹配")
    expected_groups = protocol_file_groups(config)
    checks: list[dict[str, Any]] = []
    for group, relatives in expected_groups.items():
        declared = manifest.get(group, {})
        if set(declared) != set(relatives):
            raise RuntimeError(f"ORJ V2 清单 {group} 文件集合漂移")
        for relative in relatives:
            actual = file_record(project_path(relative))
            expected = declared[relative]
            matches = actual == expected
            checks.append(
                {
                    "group": group,
                    "path": relative,
                    "actual": actual,
                    "expected": expected,
                    "matches": matches,
                }
            )
            if not matches:
                raise RuntimeError(f"ORJ V2 冻结文件漂移：{relative}")
    return {
        "status": "PASS_FROZEN_ORJ_V2_PROTOCOL_AND_INPUT_HASHES",
        "manifest_path": config["artifacts"]["protocol_manifest"],
        "manifest_content_sha256": manifest["manifest_content_sha256"],
        "checks": checks,
        "manifest": manifest,
    }


def configure_duckdb(output_root: Path) -> duckdb.DuckDBPyConnection:
    """创建只读输入、允许本地临时溢写的 DuckDB 会话。"""

    output_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute(
        f"PRAGMA temp_directory='{sql_path(output_root / 'duckdb_tmp')}'"
    )
    return connection


def load_selected_signals(
    connection: duckdb.DuckDBPyConnection,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """读取冻结父账本并仅用起点可知字段选股。"""

    signal_path = sql_path(project_path(config["parent"]["signal_ledger"]))
    source = connection.execute(
        f"""
        SELECT
            event_cluster_id,
            ts_code,
            CAST(event_publication_date AS DATE) AS event_publication_date,
            CAST(event_market_date AS DATE) AS event_market_date,
            event_market_day_index,
            orj,
            signal_eligible,
            analysis_eligible,
            event_cluster_type,
            component_report_types,
            exchange,
            industry_l1,
            industry_l1_code,
            origin_industry_peer_count,
            security_status_on_origin,
            event_raw_close,
            event_total_return_close
        FROM read_parquet('{signal_path}')
        WHERE signal_eligible
          AND CAST(event_market_date AS DATE) BETWEEN
              DATE '{config['periods']['descriptive_start']}'
              AND DATE '{config['periods']['primary_end']}'
        ORDER BY event_market_date, ts_code
        """
    ).fetchdf()
    selected = select_high_orj(source, config)
    if selected.empty:
        raise RuntimeError("冻结规则未选出任何 ORJ 信号")
    if selected.duplicated("execution_event_id").any():
        raise RuntimeError("ORJ V2 执行事件 ID 不唯一")
    return selected


def attach_planned_market_dates(
    connection: duckdb.DuckDBPyConnection,
    selected: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """把事件日指数机械映射到 t+1 开盘和 t+20 收盘日期。"""

    market_path = sql_path(project_path(config["inputs"]["unified_daily_market"]))
    minimum = int(selected["event_market_day_index"].min()) + 1
    maximum = int(selected["event_market_day_index"].max()) + 20
    calendar = connection.execute(
        f"""
        SELECT
            market_day_index,
            MIN(CAST(date AS DATE)) AS market_date,
            COUNT(DISTINCT CAST(date AS DATE)) AS distinct_date_count
        FROM read_parquet('{market_path}')
        WHERE market_day_index BETWEEN {minimum} AND {maximum}
        GROUP BY market_day_index
        ORDER BY market_day_index
        """
    ).fetchdf()
    if not calendar["distinct_date_count"].eq(1).all():
        raise RuntimeError("统一市场面板的 market_day_index 映射到多个日期")
    mapping = calendar.set_index("market_day_index")["market_date"]
    work = selected.copy()
    work["entry_market_day_index"] = (
        pd.to_numeric(work["event_market_day_index"], errors="raise").astype(int) + 1
    )
    work["exit_market_day_index"] = (
        pd.to_numeric(work["event_market_day_index"], errors="raise").astype(int) + 20
    )
    work["entry_date"] = pd.to_datetime(
        work["entry_market_day_index"].map(mapping), errors="coerce"
    ).dt.normalize()
    work["exit_date"] = pd.to_datetime(
        work["exit_market_day_index"].map(mapping), errors="coerce"
    ).dt.normalize()
    if work[["entry_date", "exit_date"]].isna().any().any():
        raise RuntimeError("部分 ORJ 信号无法映射到精确 t+1/t+20 市场日")
    return work


def attach_market_rows(
    connection: duckdb.DuckDBPyConnection,
    selected: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """一次扫描连接精确入场和退出市场行。"""

    market_path = sql_path(project_path(config["inputs"]["unified_daily_market"]))
    connection.register("selected_planned", selected)
    result = connection.execute(
        f"""
        SELECT
            s.*,
            e.con_code IS NOT NULL AS entry_row_present,
            e.pre_close AS entry_pre_close,
            e.raw_open AS entry_raw_open,
            e.raw_high AS entry_raw_high,
            e.raw_low AS entry_raw_low,
            e.raw_close AS entry_raw_close,
            e.total_return_open AS entry_total_return_open,
            e.total_return_close AS entry_total_return_close,
            e.is_suspended AS entry_is_suspended,
            e.observed_traded_row AS entry_observed_traded_row,
            e.observation_status AS entry_observation_status,
            x.con_code IS NOT NULL AS exit_row_present,
            x.pre_close AS exit_pre_close,
            x.raw_open AS exit_raw_open,
            x.raw_high AS exit_raw_high,
            x.raw_low AS exit_raw_low,
            x.raw_close AS exit_raw_close,
            x.total_return_open AS exit_total_return_open,
            x.total_return_close AS exit_total_return_close,
            x.is_suspended AS exit_is_suspended,
            x.observed_traded_row AS exit_observed_traded_row,
            x.observation_status AS exit_observation_status
        FROM selected_planned s
        LEFT JOIN read_parquet('{market_path}') e
          ON e.con_code = s.ts_code
         AND e.market_day_index = s.entry_market_day_index
        LEFT JOIN read_parquet('{market_path}') x
          ON x.con_code = s.ts_code
         AND x.market_day_index = s.exit_market_day_index
        ORDER BY s.event_market_date, s.high_orj_rank, s.ts_code
        """
    ).fetchdf()
    connection.unregister("selected_planned")
    if len(result) != len(selected):
        raise RuntimeError("统一市场入场/退出连接改变了事件行数")
    return result


def attach_security_statuses(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """按交易日 15:00 前可得的有效区间连接入场和退出状态。"""

    status_path = sql_path(project_path(config["inputs"]["security_status_intervals"]))
    requests = pd.concat(
        [
            frame[["execution_event_id", "ts_code", "entry_date"]]
            .rename(columns={"entry_date": "trade_date"})
            .assign(clock="entry"),
            frame[["execution_event_id", "ts_code", "exit_date"]]
            .rename(columns={"exit_date": "trade_date"})
            .assign(clock="exit"),
        ],
        ignore_index=True,
    )
    connection.register("status_requests", requests)
    statuses = connection.execute(
        f"""
        WITH ranked AS (
            SELECT
                r.execution_event_id,
                r.clock,
                s.status,
                s.available_at,
                ROW_NUMBER() OVER (
                    PARTITION BY r.execution_event_id, r.clock
                    ORDER BY s.available_at DESC, s.valid_from DESC
                ) AS choice
            FROM status_requests r
            JOIN read_parquet('{status_path}') s
              ON s.ts_code = r.ts_code
             AND CAST(s.valid_from AS DATE) <= CAST(r.trade_date AS DATE)
             AND (s.valid_to IS NULL OR CAST(s.valid_to AS DATE) >= CAST(r.trade_date AS DATE))
             AND s.available_at <= CAST(r.trade_date AS TIMESTAMP) + INTERVAL 15 HOUR
        )
        SELECT execution_event_id, clock, status, available_at
        FROM ranked
        WHERE choice = 1
        """
    ).fetchdf()
    connection.unregister("status_requests")
    status_wide = statuses.pivot(
        index="execution_event_id", columns="clock", values="status"
    ).rename(
        columns={
            "entry": "entry_security_status",
            "exit": "exit_security_status",
        }
    )
    available_wide = statuses.pivot(
        index="execution_event_id", columns="clock", values="available_at"
    ).rename(
        columns={
            "entry": "entry_status_available_at",
            "exit": "exit_status_available_at",
        }
    )
    joined = frame.merge(
        status_wide.reset_index(), on="execution_event_id", how="left", validate="one_to_one"
    ).merge(
        available_wide.reset_index(),
        on="execution_event_id",
        how="left",
        validate="one_to_one",
    )
    return joined


def attach_industry_path(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """连接 t+1 至 t+20 行业路径并重建 t+1 同业开收盘收益。"""

    peer_path = sql_path(project_path(config["inputs"]["industry_peer_return_panel"]))
    market_path = sql_path(project_path(config["inputs"]["unified_daily_market"]))
    keys = frame[
        [
            "execution_event_id",
            "ts_code",
            "entry_date",
            "entry_market_day_index",
            "exit_market_day_index",
        ]
    ].copy()
    connection.register("industry_event_keys", keys)
    path_stats = connection.execute(
        f"""
        SELECT
            s.execution_event_id,
            COUNT(p.market_day_index) FILTER (
                WHERE p.industry_peer_count >= 10
                  AND p.industry_peer_log_return IS NOT NULL
            ) AS industry_valid_path_rows,
            MIN(p.industry_peer_count) FILTER (
                WHERE p.industry_peer_log_return IS NOT NULL
            ) AS industry_minimum_peer_count,
            SUM(p.industry_peer_log_return) FILTER (
                WHERE p.industry_peer_count >= 10
                  AND p.industry_peer_log_return IS NOT NULL
            ) AS industry_log_return_sum_t1_t20,
            MAX(CASE
                WHEN p.market_day_index = s.entry_market_day_index
                THEN p.industry_peer_log_return ELSE NULL
            END) AS industry_peer_log_return_t1_close_to_close,
            MAX(CASE
                WHEN p.market_day_index = s.entry_market_day_index
                THEN p.industry_l1_code ELSE NULL
            END) AS entry_industry_l1_code
        FROM industry_event_keys s
        LEFT JOIN read_parquet('{peer_path}') p
          ON p.con_code = s.ts_code
         AND p.market_day_index BETWEEN
             s.entry_market_day_index AND s.exit_market_day_index
        GROUP BY s.execution_event_id
        """
    ).fetchdf()
    enriched = frame.merge(
        path_stats, on="execution_event_id", how="left", validate="one_to_one"
    )

    subject_keys = enriched[
        [
            "execution_event_id",
            "ts_code",
            "entry_date",
            "entry_market_day_index",
            "entry_industry_l1_code",
        ]
    ].copy()
    needed_dates = subject_keys[["entry_date"]].drop_duplicates().reset_index(drop=True)
    connection.register("industry_subject_keys", subject_keys)
    connection.register("industry_needed_dates", needed_dates)
    intraday = connection.execute(
        f"""
        WITH valid_intraday AS (
            SELECT
                p.con_code,
                CAST(p.date AS DATE) AS date,
                p.market_day_index,
                p.industry_l1_code,
                LN(m.total_return_close / m.total_return_open) AS intraday_log_return
            FROM read_parquet('{peer_path}') p
            JOIN industry_needed_dates d
              ON CAST(p.date AS DATE) = CAST(d.entry_date AS DATE)
            JOIN read_parquet('{market_path}') m
              ON m.con_code = p.con_code
             AND m.market_day_index = p.market_day_index
            WHERE COALESCE(m.observed_traded_row, FALSE)
              AND NOT COALESCE(m.is_suspended, FALSE)
              AND m.total_return_open > 0
              AND m.total_return_close > 0
        ), pools AS (
            SELECT
                date,
                industry_l1_code,
                COUNT(*) AS pool_count,
                SUM(intraday_log_return) AS pool_sum
            FROM valid_intraday
            GROUP BY date, industry_l1_code
        ), subject AS (
            SELECT
                s.execution_event_id,
                s.entry_date,
                s.entry_industry_l1_code,
                v.intraday_log_return AS subject_intraday_log_return
            FROM industry_subject_keys s
            LEFT JOIN valid_intraday v
              ON v.con_code = s.ts_code
             AND v.market_day_index = s.entry_market_day_index
             AND v.industry_l1_code = s.entry_industry_l1_code
        )
        SELECT
            s.execution_event_id,
            (
                p.pool_count
                - CASE WHEN s.subject_intraday_log_return IS NULL THEN 0 ELSE 1 END
            ) AS industry_intraday_peer_count,
            CASE
                WHEN p.pool_count
                     - CASE WHEN s.subject_intraday_log_return IS NULL THEN 0 ELSE 1 END
                     > 0
                THEN (
                    p.pool_sum - COALESCE(s.subject_intraday_log_return, 0.0)
                ) / (
                    p.pool_count
                    - CASE WHEN s.subject_intraday_log_return IS NULL THEN 0 ELSE 1 END
                )
                ELSE NULL
            END AS industry_peer_log_return_t1_open_to_close
        FROM subject s
        LEFT JOIN pools p
          ON p.date = CAST(s.entry_date AS DATE)
         AND p.industry_l1_code = s.entry_industry_l1_code
        """
    ).fetchdf()
    connection.unregister("industry_event_keys")
    connection.unregister("industry_subject_keys")
    connection.unregister("industry_needed_dates")
    return enriched.merge(
        intraday, on="execution_event_id", how="left", validate="one_to_one"
    )


def attach_broad_market_inputs(
    connection: duckdb.DuckDBPyConnection,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """连接 H00300 全收益收盘路径和 000300 次日开收盘。"""

    h00300_path = sql_path(project_path(config["inputs"]["csi300_total_return_index"]))
    price_path = sql_path(project_path(config["inputs"]["csi300_price_index"]))
    keys = frame[["execution_event_id", "entry_date", "exit_date"]].copy()
    connection.register("broad_market_keys", keys)
    index_inputs = connection.execute(
        f"""
        SELECT
            s.execution_event_id,
            h1.close AS h00300_entry_close,
            h20.close AS h00300_exit_close,
            p1.open AS csi300_entry_open,
            p1.close AS csi300_entry_close
        FROM broad_market_keys s
        LEFT JOIN read_parquet('{h00300_path}') h1
          ON CAST(h1.date AS DATE) = CAST(s.entry_date AS DATE)
        LEFT JOIN read_parquet('{h00300_path}') h20
          ON CAST(h20.date AS DATE) = CAST(s.exit_date AS DATE)
        LEFT JOIN read_parquet('{price_path}') p1
          ON CAST(p1.date AS DATE) = CAST(s.entry_date AS DATE)
        """
    ).fetchdf()
    connection.unregister("broad_market_keys")
    if index_inputs.duplicated("execution_event_id").any():
        raise RuntimeError("沪深300基准日期存在重复行")
    return frame.merge(
        index_inputs, on="execution_event_id", how="left", validate="one_to_one"
    )


def calculate_event_outcomes(
    connection: duckdb.DuckDBPyConnection,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """依次完成无未来筛选、成交、重叠、成本和基准计算。"""

    selected = load_selected_signals(connection, config)
    selection_summary = {
        "selected_events": int(len(selected)),
        "selected_decision_dates": int(selected["event_market_date"].nunique()),
        "selected_symbols": int(selected["ts_code"].nunique()),
        "selected_analysis_eligible_false": int(
            (~selected["analysis_eligible"].fillna(False).astype(bool)).sum()
        ),
        "primary_events": int(
            selected["evaluation_period_v2"].eq("PRIMARY_2021_2025").sum()
        ),
        "descriptive_events": int(
            selected["evaluation_period_v2"].eq("DESCRIPTIVE_2016_2020").sum()
        ),
        "minimum_candidates_per_selected_date": int(
            selected["event_date_candidate_count"].min()
        ),
        "selection_uses_signal_eligible_not_analysis_eligible": True,
    }
    frame = attach_planned_market_dates(connection, selected, config)
    frame = attach_market_rows(connection, frame, config)
    frame = attach_security_statuses(connection, frame, config)
    frame = attach_industry_path(connection, frame, config)
    frame = attach_broad_market_inputs(connection, frame, config)
    frame = classify_execution_frame(frame, config)
    frame = apply_same_security_overlap(frame)
    frame = attach_execution_economics(frame, config)
    frame = attach_benchmark_returns(frame, config)
    return frame, selection_summary


def top_symbol_capital_share(event_outcomes: pd.DataFrame) -> dict[str, Any]:
    """计算压力成本下前十大股票已承诺本金集中度，仅作描述。"""

    primary = event_outcomes.loc[
        event_outcomes["evaluation_period_v2"].eq("PRIMARY_2021_2025")
        & event_outcomes["allocation_included"].fillna(False)
        & pd.to_numeric(
            event_outcomes["stress_committed_capital"], errors="coerce"
        ).notna()
    ].copy()
    capital = primary.groupby("ts_code", observed=True)[
        "stress_committed_capital"
    ].sum().sort_values(ascending=False)
    total = float(capital.sum())
    top = capital.head(10)
    return {
        "resolved_symbols": int(len(capital)),
        "total_committed_capital_across_event_allocations_cny": total,
        "top_10_symbol_capital_share": float(top.sum() / total) if total > 0 else np.nan,
        "top_10_symbols": [
            {"ts_code": str(code), "committed_capital_cny": float(value)}
            for code, value in top.items()
        ],
        "explicitly_not_an_acceptance_gate": True,
    }


def build_descriptive_summary(
    event_outcomes: pd.DataFrame,
    decision_metrics: pd.DataFrame,
    selection_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """汇总样本、执行状态、频率和非门控交易统计。"""

    primary = event_outcomes.loc[
        event_outcomes["evaluation_period_v2"].eq("PRIMARY_2021_2025")
    ]
    status_counts = (
        event_outcomes["execution_status"].value_counts(dropna=False).to_dict()
    )
    primary_status_counts = primary["execution_status"].value_counts(dropna=False).to_dict()
    per_date = primary.groupby("event_market_date", observed=True).size()
    report_types = primary["event_cluster_type"].value_counts(dropna=False).to_dict()
    filled = primary.loc[primary["execution_status"].eq("FILLED_RESOLVED")]
    return {
        "selection": dict(selection_summary),
        "event_date_range": {
            "minimum": event_outcomes["event_market_date"].min(),
            "maximum": event_outcomes["event_market_date"].max(),
        },
        "execution_status_counts_all_periods": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "execution_status_counts_primary": {
            str(key): int(value) for key, value in primary_status_counts.items()
        },
        "primary_frequency": {
            "selected_events": int(len(primary)),
            "decision_dates": int(primary["event_market_date"].nunique()),
            "symbols": int(primary["ts_code"].nunique()),
            "mean_selected_signals_per_decision_date": float(per_date.mean()),
            "median_selected_signals_per_decision_date": float(per_date.median()),
            "maximum_selected_signals_per_decision_date": int(per_date.max()),
            "explicitly_not_an_acceptance_gate": True,
        },
        "primary_report_type_counts": {
            str(key): int(value) for key, value in report_types.items()
        },
        "primary_filled_costs": {
            "filled_resolved_events": int(len(filled)),
            "base_total_cost_cny": float(
                pd.to_numeric(
                    filled["base_total_explicit_and_slippage_cost"], errors="coerce"
                ).sum()
            ),
            "stress_total_cost_cny": float(
                pd.to_numeric(
                    filled["stress_total_explicit_and_slippage_cost"], errors="coerce"
                ).sum()
            ),
            "stress_mean_cost_per_filled_event_cny": float(
                pd.to_numeric(
                    filled["stress_total_explicit_and_slippage_cost"], errors="coerce"
                ).mean()
            ),
        },
        "stress_industry_trade_statistics": descriptive_trade_statistics(
            event_outcomes,
            scenario="stress",
            benchmark_column="industry_return",
        ),
        "stress_broad_market_trade_statistics": descriptive_trade_statistics(
            event_outcomes,
            scenario="stress",
            benchmark_column="broad_market_return",
        ),
        "capital_concentration": top_symbol_capital_share(event_outcomes),
        "decision_metric_rows": int(len(decision_metrics)),
    }


def percent(value: Any, digits: int = 3) -> str:
    """把小数格式化为百分比；缺失时保留 NO_VIEW。"""

    if value is None:
        return "NO_VIEW"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NO_VIEW"
    if not np.isfinite(number):
        return "NO_VIEW"
    return f"{number * 100:.{digits}f}%"


def path_table_row(label: str, path: Mapping[str, Any]) -> str:
    """生成结论表的一行。"""

    bootstrap = path["bootstrap"]
    halves = path["chronological_halves"]
    return (
        f"| {label} | {path['status']} | {path['eligible_primary_decision_dates']} | "
        f"{percent(path['overall_resolution_fraction'], 2)} | "
        f"{percent(bootstrap['point_estimate'])} | "
        f"[{percent(bootstrap['ci_lower'])}, {percent(bootstrap['ci_upper'])}] | "
        f"{percent(halves['earlier_mean_net_excess_return'])} / "
        f"{percent(halves['later_mean_net_excess_return'])} | "
        f"{path['positive_complete_years']}/5 |\n"
    )


def build_report_markdown(report: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """生成详细但不附加额外审查流程的中文研究报告。"""

    gate = report["gate_results"]
    descriptive = report["descriptive_summary"]
    stress_industry = gate["stress_industry_path"]
    stress_broad = gate["stress_broad_market_path"]
    base_industry = gate["all_cost_benchmark_paths"]["base__industry"]
    base_broad = gate["all_cost_benchmark_paths"]["base__broad_market"]
    status_counts = descriptive["execution_status_counts_primary"]
    execution_lines = "\n".join(
        f"- `{status}`：{count:,}" for status, count in status_counts.items()
    )
    industry_trade = descriptive["stress_industry_trade_statistics"]
    broad_trade = descriptive["stress_broad_market_trade_statistics"]
    annual_industry = stress_industry["annual_mean_net_excess_returns"]
    annual_broad = stress_broad["annual_mean_net_excess_returns"]
    annual_years = [str(year) for year in config["periods"]["primary_complete_years"]]
    annual_rows = "".join(
        f"| {year} | {percent(annual_industry.get(year))} | "
        f"{percent(annual_broad.get(year))} |\n"
        for year in annual_years
    )
    artifacts = config["artifacts"]
    return f"""# ORJ V2：真实成本后净超额历史执行验证

## 最终判断

**历史状态：`{gate['historical_status']}`。**

- 历史 Alpha 候选（相对点时行业全收益）：**{'是' if gate['historical_alpha_candidate'] else '否'}**。
- 历史强 Beta 候选（相对 H00300 沪深300全收益）：**{'是' if gate['historical_strong_beta_candidate'] else '否'}**。
- 这是看过父研究结果后的 **非盲历史执行测验**。即使通过，也只允许称为“历史候选”，不等于前瞻样本外确认、Shadow 授权或实盘 Alpha。
- 胜率、盈亏比、利润因子、利润幅度、频率、流动性和容量均只报告，不是通过条件。

## 测验的策略

1. 每个官方定期报告事件的市场日收盘后，使用父账本已冻结的 ORJ（公告后首个可交易日开盘相对前收的总回报跳空）。
2. 每个事件日只在 `signal_eligible=True` 的股票内排序；**不使用依赖未来 20 日标签是否完整的 `analysis_eligible`**。
3. 当日候选不少于 50 只，按 ORJ 从高到低、代码从小到大打破并列，选择严格前 20%。
4. 每个信号计划买 100 股：事件日后第 1 个市场日开盘买入，第 20 个市场日收盘卖出；不延后、不换日、不优化持有期。
5. 次日停牌或开盘涨停视为未成交并持有现金；缺失市场行、成交后退出停牌/跌停/缺失、退市状态均保留 `NO_VIEW`。
6. 同一股票已有一手持仓时跳过新信号；前一持仓状态无法解析时，后续同股信号全部保留为跳过状态。
7. 基础成本为单边 10bp 滑点，压力成本为单边 50bp；两者都加双边 3bp 佣金且每单最低 5 元、历史双向过户费、历史卖出印花税。
8. 股票公司行动用总回报价格处理；行业基准把 t+1 收收到收盘收益替换为点时同业开收到收盘收益；宽基准用 H00300 全收益路径，并用 000300 只校正 t+1 开盘起点。

## 主检验结果（2021—2025，压力成本）

| 路径 | 状态 | 有效决策日 | 事件解析率 | 日均净超额 | 95% 移动块区间 | 前半 / 后半 | 正收益完整年 |
|---|---:|---:|---:|---:|---:|---:|---:|
{path_table_row('行业 Alpha', stress_industry)}{path_table_row('沪深300强 Beta', stress_broad)}
通过规则很窄：事件解析率至少 90%、至少 100 个有效决策日、压力成本日均净超额及其 95% 移动块下界都严格大于 0、前后两半都严格为正、五个完整年度至少 3 年为正。没有胜率或盈亏比门槛。

## 基础成本对照

| 路径 | 日均净超额 | 95% 移动块区间 | 有效决策日 |
|---|---:|---:|---:|
| 行业 Alpha | {percent(base_industry['bootstrap']['point_estimate'])} | [{percent(base_industry['bootstrap']['ci_lower'])}, {percent(base_industry['bootstrap']['ci_upper'])}] | {base_industry['eligible_primary_decision_dates']} |
| 沪深300强 Beta | {percent(base_broad['bootstrap']['point_estimate'])} | [{percent(base_broad['bootstrap']['ci_lower'])}, {percent(base_broad['bootstrap']['ci_upper'])}] | {base_broad['eligible_primary_decision_dates']} |

## 年度稳定性（压力成本、决策日等权）

| 年度 | 行业净超额 | 沪深300全收益净超额 |
|---|---:|---:|
{annual_rows}
## 样本和成交状态

- 全阶段选中事件：{descriptive['selection']['selected_events']:,}；其中主检验事件：{descriptive['selection']['primary_events']:,}。
- 选中但 `analysis_eligible=False` 的事件：{descriptive['selection']['selected_analysis_eligible_false']:,}。这些记录证明本测验没有用未来标签完整性决定是否选股。
- 主检验决策日：{descriptive['primary_frequency']['decision_dates']:,}；股票数：{descriptive['primary_frequency']['symbols']:,}；每个决策日信号中位数：{descriptive['primary_frequency']['median_selected_signals_per_decision_date']:.1f}。
- 压力成本下，已解析成交事件总显性成本与滑点：{descriptive['primary_filled_costs']['stress_total_cost_cny']:,.2f} 元；每个已成交事件平均 {descriptive['primary_filled_costs']['stress_mean_cost_per_filled_event_cny']:,.2f} 元。

主检验执行状态：

{execution_lines}

## 用户明确不要求的指标（仅描述）

| 事件口径、压力成本 | 行业超额 | 沪深300全收益超额 |
|---|---:|---:|
| 解析事件数 | {industry_trade['resolved_events']:,} | {broad_trade['resolved_events']:,} |
| 超额胜率 | {percent(industry_trade['event_excess_win_rate'])} | {percent(broad_trade['event_excess_win_rate'])} |
| 平均正超额 / 平均负超额绝对值 | {percent(industry_trade['mean_positive_event_excess'])} / {percent(industry_trade['mean_negative_event_excess_abs'])} | {percent(broad_trade['mean_positive_event_excess'])} / {percent(broad_trade['mean_negative_event_excess_abs'])} |
| 盈亏比 | {industry_trade['payoff_ratio'] if industry_trade['payoff_ratio'] is not None else 'NO_VIEW'} | {broad_trade['payoff_ratio'] if broad_trade['payoff_ratio'] is not None else 'NO_VIEW'} |
| 利润因子 | {industry_trade['profit_factor'] if industry_trade['profit_factor'] is not None else 'NO_VIEW'} | {broad_trade['profit_factor'] if broad_trade['profit_factor'] is not None else 'NO_VIEW'} |

前十大股票占已解析事件承诺本金的 {percent(descriptive['capital_concentration']['top_10_symbol_capital_share'], 2)}；该集中度也不参与门控。

## 该结论能与不能说明什么

- 能说明：在冻结的历史样本、固定下一开盘/第 20 日收盘、100 股一手、明确停牌涨跌停与重叠规则下，是否观察到成本后为正且时间上稳定的行业或沪深300全收益超额。
- 不能说明：未来仍会盈利、真实开盘竞价一定能按日线开盘价成交、容量足够、所有退市终值已恢复、该结果是盲 OOS，或已经可以自动下单。
- 父 V1.0.1 仍保持 `REJECTED_FROZEN_OFFICIAL_REPORT_ORJ_PREDICTIVE_GATE_FAILED`，本研究没有重写旧门槛或旧结论。

## 可复核文件

- 事件级选择、成交、成本、基准和状态：`{artifacts['event_outcomes']}`
- 决策日组合指标：`{artifacts['decision_date_metrics']}`
- 年度摘要：`{artifacts['annual_summary']}`
- 机械门控：`{artifacts['gate_results']}`
- 运行与哈希回执：`{artifacts['run_receipt']}`
- 结构化报告：`{artifacts['report_json']}`
"""


def run_study(config: Mapping[str, Any]) -> dict[str, Any]:
    """运行真实历史执行验证并写出最小可复核产物。"""

    started_clock = time.perf_counter()
    started_at = datetime.now(TIMEZONE)
    protocol = verify_protocol(config)
    output_root = project_path(config["artifacts"]["root"])
    connection = configure_duckdb(output_root)
    try:
        event_outcomes, selection_summary = calculate_event_outcomes(
            connection, config
        )
    finally:
        connection.close()
    decision_metrics = build_decision_date_metrics(event_outcomes, config)
    annual_summary = build_annual_summary(decision_metrics)
    gate_results = evaluate_gate(decision_metrics, config)
    descriptive_summary = build_descriptive_summary(
        event_outcomes, decision_metrics, selection_summary
    )

    event_path = project_path(config["artifacts"]["event_outcomes"])
    decision_path = project_path(config["artifacts"]["decision_date_metrics"])
    annual_path = project_path(config["artifacts"]["annual_summary"])
    gate_path = project_path(config["artifacts"]["gate_results"])
    receipt_path = project_path(config["artifacts"]["run_receipt"])
    report_json_path = project_path(config["artifacts"]["report_json"])
    report_markdown_path = project_path(config["artifacts"]["report_markdown"])

    write_parquet_atomic(event_path, event_outcomes)
    write_csv_atomic(decision_path, decision_metrics)
    write_csv_atomic(annual_path, annual_summary)
    write_json_atomic(gate_path, gate_results)

    report = {
        "study": dict(config["study"]),
        "evidence_period": dict(config["periods"]),
        "generated_at": datetime.now(TIMEZONE).isoformat(),
        "parent_boundary": dict(config["parent"]),
        "strategy_contract": {
            "selection": dict(config["selection"]),
            "execution": dict(config["execution"]),
            "costs": dict(config["costs"]),
            "benchmarks": dict(config["benchmarks"]),
            "inference": dict(config["inference"]),
            "not_acceptance_gates": list(config["not_acceptance_gates"]),
        },
        "gate_results": gate_results,
        "descriptive_summary": descriptive_summary,
        "protocol_manifest_content_sha256": protocol[
            "manifest_content_sha256"
        ],
        "artifacts": dict(config["artifacts"]),
    }
    write_json_atomic(report_json_path, report)
    write_text_atomic(report_markdown_path, build_report_markdown(report, config))

    output_paths = [
        event_path,
        decision_path,
        annual_path,
        gate_path,
        report_json_path,
        report_markdown_path,
    ]
    receipt = {
        "study_id": config["study"]["study_id"],
        "historical_status": gate_results["historical_status"],
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(TIMEZONE).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started_clock),
        "protocol_verification_status": protocol["status"],
        "protocol_manifest_content_sha256": protocol[
            "manifest_content_sha256"
        ],
        "selection_summary": selection_summary,
        "invariants": {
            "event_id_unique": bool(
                not event_outcomes["execution_event_id"].duplicated().any()
            ),
            "selection_never_filters_on_analysis_eligible": True,
            "exact_entry_index_offset": bool(
                (
                    event_outcomes["entry_market_day_index"]
                    - event_outcomes["event_market_day_index"]
                ).eq(1).all()
            ),
            "exact_exit_index_offset": bool(
                (
                    event_outcomes["exit_market_day_index"]
                    - event_outcomes["event_market_day_index"]
                ).eq(20).all()
            ),
            "no_live_or_shadow_authorization": True,
        },
        "input_files": protocol["manifest"]["input_files"],
        "output_files": {
            path.relative_to(ROOT).as_posix(): file_record(path)
            for path in output_paths
        },
        "run_receipt_self_hash_intentionally_omitted": True,
    }
    if not all(receipt["invariants"].values()):
        raise RuntimeError(f"ORJ V2 输出不变量失败：{receipt['invariants']}")
    write_json_atomic(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    """解析命令行。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-protocol",
        action="store_true",
        help="在真实 V2 执行结果计算前一次性冻结协议、代码、测试和输入",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    config = load_config()
    if args.freeze_protocol:
        manifest_path = freeze_protocol(config)
        print(f"ORJ V2 协议已冻结：{manifest_path}")
        return
    receipt = run_study(config)
    print(
        "ORJ V2 历史执行验证完成："
        f"{receipt['historical_status']}；"
        f"事件 {receipt['selection_summary']['selected_events']:,} 条；"
        f"耗时 {receipt['elapsed_seconds']:.1f} 秒。"
    )


if __name__ == "__main__":
    main()
