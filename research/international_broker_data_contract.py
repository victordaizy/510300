"""中国上市券商国际业务研究的数据合同与底座审计。

本模块只检查研究边界、母样本、证据登记、表结构、覆盖和时点语义。
它不读取股票未来收益，不计算候选排名，不生成仓位或订单。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "international_broker_research.yaml"
REPORT_JSON = (
    ROOT / "reports" / "data_quality" / "international_broker_foundation.json"
)
REPORT_MD = (
    ROOT / "reports" / "data_quality" / "international_broker_foundation.md"
)

EXPECTED_TABLES = (
    "broker_parent_quarterly",
    "international_subsidiary_annual",
    "crossborder_demand_monthly",
    "broker_valuation_daily",
    "international_broker_events",
)
DISABLED_GOVERNANCE_FLAGS = (
    "allow_as_510300_alpha_input",
    "position_mapping_enabled",
    "order_generation_enabled",
    "broker_connection_enabled",
    "live_trading_authorized",
    "return_test_allowed",
)


class ContractError(ValueError):
    """机器可读研究合同违反冻结边界。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_parquet_write(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _file_evidence(root: Path, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "file": path.relative_to(root).as_posix(),
            "exists": False,
            "bytes": 0,
            "sha256": None,
        }
    return {
        "file": path.relative_to(root).as_posix(),
        "exists": True,
        "bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    raise ContractError(f"不支持的数据文件格式：{path}")


def _required_columns(data: pd.DataFrame, required: list[str]) -> list[str]:
    return sorted(set(required).difference(data.columns))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _nonempty_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    governance = config.get("governance", {})
    if governance.get("trading_authorization") != "RESEARCH_ONLY_NO_POSITION_CHANGE":
        errors.append("trading_authorization必须保持RESEARCH_ONLY_NO_POSITION_CHANGE")
    if governance.get("isolated_from_510300") is not True:
        errors.append("项目必须与510300系统隔离")
    for flag in DISABLED_GOVERNANCE_FLAGS:
        if governance.get(flag) is not False:
            errors.append(f"治理开关{flag}必须为false")
    if governance.get("return_test_requires_explicit_next_protocol") is not True:
        errors.append("收益检验必须要求显式冻结下一版协议")

    table_names = tuple(config.get("tables", {}).keys())
    if table_names != EXPECTED_TABLES:
        errors.append(f"分析表必须严格为五张且顺序固定：{EXPECTED_TABLES}")
    for table_name, table_config in config.get("tables", {}).items():
        required = table_config.get("required_columns", [])
        duplicates = sorted({column for column in required if required.count(column) > 1})
        if duplicates:
            errors.append(f"{table_name}存在重复字段：{duplicates}")
        for key in table_config.get("primary_key", []):
            if key not in required:
                errors.append(f"{table_name}主键字段未列入required_columns：{key}")
        for audit_field in ("evidence_ids", "retrieved_at", "data_status", "notes"):
            if audit_field not in required:
                errors.append(f"{table_name}缺少审计字段：{audit_field}")

    thresholds = config.get("frozen_thresholds", {})
    expected_thresholds = {
        "minimum_comparable_years": 3,
        "target_comparable_years": 5,
        "minimum_intl_profit_share": 0.10,
        "minimum_two_year_earnings_materiality": 0.05,
        "minimum_group_roe_uplift_percentage_points": 0.5,
        "intl_roe_above_cost_of_equity_consecutive_years": 2,
    }
    for name, expected in expected_thresholds.items():
        if thresholds.get(name) != expected:
            errors.append(f"冻结门槛{name}必须为{expected}")

    ordered_gates = config.get("gates", {}).get("ordered", [])
    if ordered_gates != [
        "G1_UNIVERSE",
        "G2_STRUCTURE",
        "G3_PROFIT_RECONSTRUCTION",
        "G4_PROFITABILITY",
        "G5_DEMAND",
        "G6_PRICING",
    ]:
        errors.append("研究门必须保持预设顺序")

    if errors:
        raise ContractError("；".join(errors))
    return {
        "status": "PASS",
        "table_count": len(table_names),
        "table_names": list(table_names),
        "governance_flags_disabled": list(DISABLED_GOVERNANCE_FLAGS),
    }


def audit_evidence_registry(
    root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], set[str]]:
    registry_config = config["evidence_registry"]
    path = root / registry_config["file"]
    result: dict[str, Any] = {
        "status": "BLOCKED_MISSING_EVIDENCE_REGISTRY",
        "file": _file_evidence(root, path),
        "row_count": 0,
        "primary_source_count": 0,
        "unhashed_primary_source_count": 0,
        "unknown_official_domains": [],
        "duplicate_source_ids": [],
        "errors": [],
    }
    if not path.exists():
        return result, set()

    data = _read_table(path)
    result["row_count"] = int(len(data))
    missing = _required_columns(data, registry_config["required_columns"])
    if missing:
        result["status"] = "BLOCKED_INVALID_EVIDENCE_SCHEMA"
        result["errors"].append(f"证据登记缺少字段：{missing}")
        return result, set()

    source_ids = data["source_id"].astype(str).str.strip()
    duplicates = sorted(source_ids[source_ids.duplicated(keep=False)].unique())
    result["duplicate_source_ids"] = duplicates
    if duplicates:
        result["errors"].append(f"source_id重复：{duplicates}")

    primary = data["authority_level"].astype(str).str.upper().eq("PRIMARY")
    result["primary_source_count"] = int(primary.sum())
    unhashed_primary = primary & ~_nonempty_string(data["snapshot_sha256"])
    result["unhashed_primary_source_count"] = int(unhashed_primary.sum())
    accepted = set(config["source_policy"]["accepted_official_domains"])
    unknown_domains: set[str] = set()
    for url in data.loc[primary, "url"].astype(str):
        host = (urlparse(url).hostname or "").lower()
        if not host or not any(host == domain or host.endswith(f".{domain}") for domain in accepted):
            unknown_domains.add(host or "<EMPTY>")
    result["unknown_official_domains"] = sorted(unknown_domains)
    if unknown_domains:
        result["errors"].append(
            f"PRIMARY证据存在未批准或空域名：{sorted(unknown_domains)}"
        )

    if source_ids.eq("").any():
        result["errors"].append("source_id不能为空")
    if result["errors"]:
        result["status"] = "BLOCKED_INVALID_EVIDENCE"
    elif result["unhashed_primary_source_count"]:
        result["status"] = "PARTIAL_UNHASHED_PRIMARY_SOURCES"
    else:
        result["status"] = "PASS"
    return result, set(source_ids[source_ids.ne("")])


def audit_universe(
    root: Path, config: dict[str, Any], known_source_ids: set[str]
) -> dict[str, Any]:
    universe_config = config["universe"]
    path = root / universe_config["file"]
    result: dict[str, Any] = {
        "status": "BLOCKED_MISSING_UNIVERSE",
        "file": _file_evidence(root, path),
        "row_count": 0,
        "core_a_issuer_count": 0,
        "dual_ah_count": 0,
        "scope_review_count": 0,
        "official_verified_core_count": 0,
        "unverified_core_count": 0,
        "errors": [],
        "warnings": [],
    }
    if not path.exists():
        return result

    data = _read_table(path)
    result["row_count"] = int(len(data))
    missing = _required_columns(data, universe_config["required_columns"])
    if missing:
        result["status"] = "BLOCKED_INVALID_UNIVERSE_SCHEMA"
        result["errors"].append(f"母样本缺少字段：{missing}")
        return result

    for column in universe_config["required_columns"]:
        data[column] = data[column].fillna("").astype(str).str.strip()
    core = data.loc[data["scope_status"].eq("CORE_A_LISTED")].copy()
    review = data.loc[data["scope_status"].str.startswith("SCOPE_REVIEW")].copy()
    dual = core.loc[core["h_ticker"].ne("")]
    verified = core["official_listing_verified"].map(_as_bool)
    result["core_a_issuer_count"] = int(len(core))
    result["dual_ah_count"] = int(len(dual))
    result["scope_review_count"] = int(len(review))
    result["official_verified_core_count"] = int(verified.sum())
    result["unverified_core_count"] = int((~verified).sum())

    if len(core) != int(universe_config["expected_core_a_issuer_count"]):
        result["errors"].append(
            "核心A股发行人数量不等于冻结值"
            f"{universe_config['expected_core_a_issuer_count']}"
        )
    if len(dual) != int(universe_config["expected_dual_ah_count"]):
        result["errors"].append(
            "A+H发行人数量不等于冻结值"
            f"{universe_config['expected_dual_ah_count']}"
        )

    invalid_a = sorted(
        core.loc[
            ~core["a_ticker"].str.match(r"^\d{6}\.(SH|SZ)$", na=False),
            "a_ticker",
        ].unique()
    )
    invalid_h = sorted(
        dual.loc[
            ~dual["h_ticker"].str.match(r"^\d{5}\.HK$", na=False),
            "h_ticker",
        ].unique()
    )
    if invalid_a:
        result["errors"].append(f"A股代码格式错误：{invalid_a}")
    if invalid_h:
        result["errors"].append(f"H股代码格式错误：{invalid_h}")

    for column in ("parent_id", "a_ticker"):
        duplicates = sorted(core.loc[core[column].duplicated(keep=False), column].unique())
        if duplicates:
            result["errors"].append(f"核心母样本{column}重复：{duplicates}")

    current_tickers = set(core["a_ticker"]) | set(core["h_ticker"])
    stale = sorted(
        current_tickers.intersection(universe_config["merger_bridge_required"])
    )
    if stale:
        result["errors"].append(f"已终止上市的海通代码仍在当前核心母样本：{stale}")

    unknown_sources = sorted(
        set(data.loc[data["listing_source_id"].ne(""), "listing_source_id"])
        .difference(known_source_ids)
    )
    if unknown_sources:
        result["errors"].append(f"母样本引用未知证据ID：{unknown_sources}")
    if result["unverified_core_count"]:
        result["warnings"].append(
            f"仍有{result['unverified_core_count']}个核心发行人未逐项取得官方上市证据"
        )

    if result["errors"]:
        result["status"] = "BLOCKED_INVALID_UNIVERSE"
    elif result["unverified_core_count"]:
        result["status"] = "BLOCKED_UNVERIFIED_OFFICIAL_LISTINGS"
    else:
        result["status"] = "PASS"
    return result


def _split_evidence_ids(values: pd.Series) -> set[str]:
    identifiers: set[str] = set()
    for value in values.fillna("").astype(str):
        identifiers.update(
            item.strip() for item in re.split(r"[;,|]", value) if item.strip()
        )
    return identifiers


def audit_table(
    root: Path,
    table_name: str,
    table_config: dict[str, Any],
    known_source_ids: set[str],
    *,
    override_path: Path | None = None,
    expected_core_tickers: set[str] | None = None,
) -> dict[str, Any]:
    path = override_path or (root / table_config["path"])
    result: dict[str, Any] = {
        "table": table_name,
        "status": "BLOCKED_MISSING_TABLE",
        "file": _file_evidence(root, path),
        "row_count": 0,
        "duplicate_primary_keys": 0,
        "invalid_date_counts": {},
        "unknown_evidence_ids": [],
        "coverage": {},
        "errors": [],
        "warnings": [],
    }
    if not path.exists():
        return result

    try:
        data = _read_table(path)
    except Exception as error:  # pragma: no cover - 依赖具体损坏文件格式
        result["status"] = "BLOCKED_UNREADABLE_TABLE"
        result["errors"].append(f"读取失败：{type(error).__name__}: {error}")
        return result

    result["row_count"] = int(len(data))
    missing = _required_columns(data, table_config["required_columns"])
    if missing:
        result["status"] = "BLOCKED_INVALID_SCHEMA"
        result["errors"].append(f"缺少字段：{missing}")
        return result
    if data.empty:
        result["status"] = "BLOCKED_EMPTY_TABLE"
        result["errors"].append("表存在但没有记录")
        return result

    key = table_config["primary_key"]
    duplicate_count = int(data.duplicated(subset=key, keep=False).sum())
    result["duplicate_primary_keys"] = duplicate_count
    if duplicate_count:
        result["errors"].append(f"主键重复记录数：{duplicate_count}")

    parsed_dates: dict[str, pd.Series] = {}
    for column in table_config.get("date_columns", []):
        raw = data[column]
        nonempty = _nonempty_string(raw)
        parsed = pd.to_datetime(raw.where(nonempty), errors="coerce")
        invalid_count = int((nonempty & parsed.isna()).sum())
        result["invalid_date_counts"][column] = invalid_count
        parsed_dates[column] = parsed
        if invalid_count:
            result["errors"].append(f"{column}存在{invalid_count}个无效日期")

    if table_name == "broker_parent_quarterly":
        period = parsed_dates["report_period"]
        publish = parsed_dates["report_publish_date"]
        leakage = period.notna() & publish.notna() & publish.lt(period)
        if leakage.any():
            result["errors"].append(
                f"report_publish_date早于report_period的记录数：{int(leakage.sum())}"
            )

    if table_name == "international_broker_events":
        announcement = parsed_dates["announcement_date"]
        approval = parsed_dates["approval_date"]
        effective = parsed_dates["effective_date"]
        revenue = parsed_dates["first_revenue_date"]
        approval_before_announcement = (
            announcement.notna() & approval.notna() & approval.lt(announcement)
        )
        effective_before_announcement = (
            announcement.notna() & effective.notna() & effective.lt(announcement)
        )
        revenue_before_effective = (
            effective.notna() & revenue.notna() & revenue.lt(effective)
        )
        for label, mask in (
            ("批准日早于公告日", approval_before_announcement),
            ("生效日早于公告日", effective_before_announcement),
            ("首次收入日早于生效日", revenue_before_effective),
        ):
            if mask.any():
                result["errors"].append(f"{label}的记录数：{int(mask.sum())}")
        conditional = data["implementation_status"].astype(str).str.contains(
            "EXPECTED_CONDITIONAL", na=False
        )
        if (conditional & revenue.notna()).any():
            result["errors"].append("条件性预计事件不得预填首次收入日期")

    used_ids = _split_evidence_ids(data["evidence_ids"])
    unknown = sorted(used_ids.difference(known_source_ids))
    result["unknown_evidence_ids"] = unknown
    if unknown:
        result["errors"].append(f"引用未知证据ID：{unknown}")
    missing_evidence = ~_nonempty_string(data["evidence_ids"])
    if missing_evidence.any():
        result["errors"].append(
            f"evidence_ids为空的记录数：{int(missing_evidence.sum())}"
        )
    missing_status = ~_nonempty_string(data["data_status"])
    if missing_status.any():
        result["errors"].append(
            f"data_status为空的记录数：{int(missing_status.sum())}"
        )

    coverage = table_config.get("coverage_requirements", {})
    critical_columns = coverage.get("critical_non_null_columns", [])
    missing_critical: dict[str, int] = {}
    for column in critical_columns:
        missing_count = int((~_nonempty_string(data[column])).sum())
        missing_critical[column] = missing_count
        if missing_count:
            result["errors"].append(f"关键字段{column}缺失记录数：{missing_count}")
    if critical_columns:
        result["coverage"]["missing_critical_counts"] = missing_critical

    expected = expected_core_tickers or set()
    if table_name == "broker_parent_quarterly" and expected:
        periods = parsed_dates["report_period"]
        year_end = periods.dt.month.eq(12) & periods.dt.day.eq(31)
        usable = data.loc[year_end].copy()
        usable["fiscal_year"] = periods.loc[year_end].dt.year
        counts = usable.groupby("ticker")["fiscal_year"].nunique()
        minimum_years = int(coverage["minimum_year_end_periods_per_issuer"])
        covered = set(counts.loc[counts.ge(minimum_years)].index.astype(str))
        ratio = len(covered & expected) / max(len(expected), 1)
        result["coverage"].update(
            {
                "issuer_ratio_with_minimum_years": ratio,
                "minimum_year_end_periods_per_issuer": minimum_years,
                "missing_or_short_history_tickers": sorted(expected.difference(covered)),
            }
        )
        if ratio < float(coverage["minimum_core_issuer_ratio"]):
            result["errors"].append(
                f"具有至少{minimum_years}个年末期的核心发行人覆盖率仅{ratio:.2%}"
            )
    elif table_name == "international_subsidiary_annual" and expected:
        years = pd.to_numeric(data["fiscal_year"], errors="coerce")
        usable = data.assign(_fiscal_year=years).dropna(subset=["_fiscal_year"])
        counts = usable.groupby("parent_ticker")["_fiscal_year"].nunique()
        minimum_years = int(coverage["minimum_fiscal_years_per_parent"])
        covered = set(counts.loc[counts.ge(minimum_years)].index.astype(str))
        ratio = len(covered & expected) / max(len(expected), 1)
        result["coverage"].update(
            {
                "parent_ratio_with_minimum_years": ratio,
                "minimum_fiscal_years_per_parent": minimum_years,
                "missing_or_short_history_tickers": sorted(expected.difference(covered)),
            }
        )
        if ratio < float(coverage["minimum_core_issuer_ratio"]):
            result["errors"].append(
                f"具有至少{minimum_years}个子公司审计年度的核心发行人覆盖率仅{ratio:.2%}"
            )
    elif table_name == "crossborder_demand_monthly":
        months = parsed_dates["month"].dt.to_period("M").nunique()
        result["coverage"]["distinct_months"] = int(months)
        if months < int(coverage["minimum_months"]):
            result["errors"].append(
                f"跨境需求月度覆盖仅{months}个月，低于{coverage['minimum_months']}个月"
            )
    elif table_name == "broker_valuation_daily" and expected:
        counts = data.assign(_date=parsed_dates["date"]).groupby("ticker")["_date"].nunique()
        minimum_days = int(coverage["minimum_trading_days_per_ticker"])
        covered = set(counts.loc[counts.ge(minimum_days)].index.astype(str))
        ratio = len(covered & expected) / max(len(expected), 1)
        result["coverage"].update(
            {
                "issuer_ratio_with_minimum_days": ratio,
                "minimum_trading_days_per_ticker": minimum_days,
                "missing_or_short_history_tickers": sorted(expected.difference(covered)),
            }
        )
        if ratio < float(coverage["minimum_core_issuer_ratio"]):
            result["errors"].append(
                f"具有至少{minimum_days}个交易日的核心发行人覆盖率仅{ratio:.2%}"
            )
    elif table_name == "international_broker_events":
        verified = data["data_status"].astype(str).str.startswith("VERIFIED_")
        result["coverage"]["verified_rows"] = int(verified.sum())
        if verified.sum() < int(coverage["minimum_verified_rows"]):
            result["errors"].append("正式事件表没有已核验官方事件")

    result["status"] = "PASS" if not result["errors"] else "BLOCKED_INVALID_DATA"
    return result


def audit_event_seed(
    root: Path, config: dict[str, Any], known_source_ids: set[str]
) -> dict[str, Any]:
    table_config = config["tables"]["international_broker_events"]
    seed_path = root / table_config["seed_file"]
    seed_config = dict(table_config)
    seed_config["path"] = table_config["seed_file"]
    return audit_table(
        root,
        "international_broker_events",
        seed_config,
        known_source_ids,
        override_path=seed_path,
    )


def materialize_event_seed(root: Path = ROOT) -> Path:
    """将已校验的官方事件种子原样固化为正式事件表。"""

    config = load_config(root / "config" / "international_broker_research.yaml")
    validate_contract(config)
    evidence, known_source_ids = audit_evidence_registry(root, config)
    if evidence["status"] == "BLOCKED_INVALID_EVIDENCE":
        raise ContractError("证据登记无效，禁止生成事件表")
    seed_audit = audit_event_seed(root, config, known_source_ids)
    if seed_audit["status"] != "PASS":
        raise ContractError(f"事件种子未通过校验：{seed_audit['errors']}")
    table_config = config["tables"]["international_broker_events"]
    seed = _read_table(root / table_config["seed_file"])
    output = root / table_config["path"]
    _atomic_parquet_write(output, seed)
    return output


def _gate_status(
    evidence: dict[str, Any],
    universe: dict[str, Any],
    tables: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    g1_pass = universe["status"] == "PASS" and evidence["status"] == "PASS"
    g2_pass = g1_pass and tables["international_subsidiary_annual"]["status"] == "PASS"
    g3_pass = (
        g2_pass
        and tables["broker_parent_quarterly"]["status"] == "PASS"
        and tables["international_subsidiary_annual"]["status"] == "PASS"
    )
    return {
        "G1_UNIVERSE": {
            "status": "PASS" if g1_pass else "BLOCKED",
            "reason": (
                f"universe={universe['status']}; "
                f"evidence={evidence['status']}"
            ),
        },
        "G2_STRUCTURE": {
            "status": "PASS" if g2_pass else "BLOCKED",
            "reason": tables["international_subsidiary_annual"]["status"],
        },
        "G3_PROFIT_RECONSTRUCTION": {
            "status": "PASS" if g3_pass else "BLOCKED",
            "reason": tables["broker_parent_quarterly"]["status"],
        },
        "G4_PROFITABILITY": {
            "status": "BLOCKED",
            "reason": "需要G3通过后另行计算经调整利润和资本回报",
        },
        "G5_DEMAND": {
            "status": "BLOCKED",
            "reason": tables["crossborder_demand_monthly"]["status"],
        },
        "G6_PRICING": {
            "status": "BLOCKED_EXPLICIT_APPROVAL_REQUIRED",
            "reason": "当前协议禁止收益检验；需用户批准并冻结下一版协议",
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    universe = report["universe"]
    lines = [
        "# 国际化券商研究数据底座审计",
        "",
        f"- 审计时间：{report['generated_at']}",
        f"- 项目状态：`{report['project_state']}`",
        f"- 安全状态：`{report['safety_state']}`",
        f"- 收益检验：`{'允许' if report['return_test_allowed'] else '禁止'}`",
        "",
        "## 母样本",
        "",
        f"- 核心A股发行人：{universe['core_a_issuer_count']} / {universe['expected_core_a_issuer_count']}",
        f"- A+H发行人：{universe['dual_ah_count']} / {universe['expected_dual_ah_count']}",
        f"- 已获官方上市证据：{universe['official_verified_core_count']}",
        f"- 尚待逐项官方核验：{universe['unverified_core_count']}",
        f"- 母样本状态：`{universe['status']}`",
        "",
        "## 五张表",
        "",
        "| 表 | 状态 | 记录数 |",
        "|---|---:|---:|",
    ]
    for table_name in EXPECTED_TABLES:
        table = report["tables"][table_name]
        lines.append(f"| `{table_name}` | `{table['status']}` | {table['row_count']} |")
    lines.extend(["", "## 研究门", ""])
    for gate_name, gate in report["gates"].items():
        lines.append(f"- `{gate_name}`：`{gate['status']}`；{gate['reason']}")
    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "逐项补齐42家核心发行人的交易所上市证据，然后从头部券商开始建立海外子公司持股、并表、少数股东和牌照审计。未完成以前保持`RESEARCH_ONLY / NO_POSITION_CHANGE`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    root: Path = ROOT,
    config_path: Path | None = None,
    *,
    write_reports: bool = True,
) -> dict[str, Any]:
    selected_config = config_path or (root / "config" / "international_broker_research.yaml")
    config = load_config(selected_config)
    contract = validate_contract(config)
    evidence, known_source_ids = audit_evidence_registry(root, config)
    universe = audit_universe(root, config, known_source_ids)
    universe["expected_core_a_issuer_count"] = int(
        config["universe"]["expected_core_a_issuer_count"]
    )
    universe["expected_dual_ah_count"] = int(
        config["universe"]["expected_dual_ah_count"]
    )
    tables = {
        table_name: audit_table(
            root,
            table_name,
            config["tables"][table_name],
            known_source_ids,
            expected_core_tickers={
                row["a_ticker"].strip()
                for row in _read_table(root / config["universe"]["file"]).to_dict("records")
                if row["scope_status"].strip() == "CORE_A_LISTED"
            },
        )
        for table_name in EXPECTED_TABLES
    }
    event_seed = audit_event_seed(root, config, known_source_ids)
    gates = _gate_status(evidence, universe, tables)
    report = {
        "project_id": config["protocol"]["project_id"],
        "project_state": config["protocol"]["state"],
        "as_of_date": config["protocol"]["as_of_date"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "safety_state": "RESEARCH_ONLY_NO_POSITION_CHANGE",
        "isolated_from_510300": True,
        "return_test_allowed": False,
        "return_test_block_reason": (
            "G1至G5尚未通过，且当前协议明确要求用户批准并冻结下一版收益检验协议。"
        ),
        "contract": contract,
        "evidence_registry": evidence,
        "universe": universe,
        "event_seed": event_seed,
        "tables": tables,
        "gates": gates,
    }
    if write_reports:
        report_json = root / "reports" / "data_quality" / "international_broker_foundation.json"
        report_md = root / "reports" / "data_quality" / "international_broker_foundation.md"
        _atomic_text_write(
            report_json, json.dumps(report, ensure_ascii=False, indent=2)
        )
        _atomic_text_write(report_md, _render_markdown(report))
    return report


def main() -> int:
    report = run_audit()
    print(
        json.dumps(
            {
                "项目状态": report["project_state"],
                "安全状态": report["safety_state"],
                "核心发行人": report["universe"]["core_a_issuer_count"],
                "A+H发行人": report["universe"]["dual_ah_count"],
                "待官方核验": report["universe"]["unverified_core_count"],
                "当前研究门": report["gates"]["G1_UNIVERSE"],
                "收益检验允许": report["return_test_allowed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
