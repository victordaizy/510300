"""构建国际券商G2实体边界与G3利润桥的严格证据状态矩阵。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.international_broker_evidence_acquisition import (
    ROOT,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
)


CONFIG_FILE = ROOT / "config" / "international_broker_g2_g3_v1_3.yaml"


def load_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    """读取冻结的V1.3输入定义。"""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """流式计算证据文件哈希。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(ticker: str, year: int) -> str:
    return f"SRC_CNINFO_ANNUAL_{ticker.split('.')[0]}_{year}"


def _expected_pairs(config: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (ticker, int(year))
        for ticker in config["protocol"]["matrix_tickers"]
        for year in config["protocol"]["matrix_years"]
    }


def verify_official_sources(
    root: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    """复核30份官方年报的身份、权威级别、路径与SHA256。"""

    manifest = pd.read_csv(
        root / config["paths"]["annual_report_manifest"],
        dtype=str,
        keep_default_na=False,
    )
    registry = pd.read_csv(
        root / config["paths"]["source_registry"],
        dtype=str,
        keep_default_na=False,
    )
    manifest["report_year"] = manifest["report_year"].astype(int)
    registry["report_year"] = registry["report_year"].astype(int)

    expected = _expected_pairs(config)
    manifest_pairs = set(zip(manifest["a_ticker"], manifest["report_year"]))
    if manifest_pairs != expected or len(manifest) != len(expected):
        missing = sorted(expected - manifest_pairs)
        unexpected = sorted(manifest_pairs - expected)
        raise RuntimeError(
            f"年报清单不是严格6×5矩阵：missing={missing}, unexpected={unexpected}, "
            f"rows={len(manifest)}"
        )

    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for item in manifest.sort_values(["a_ticker", "report_year"]).to_dict("records"):
        ticker = str(item["a_ticker"])
        year = int(item["report_year"])
        source_id = _source_id(ticker, year)
        matched = registry.loc[registry["source_id"].eq(source_id)]
        errors: list[str] = []
        if len(matched) != 1:
            errors.append(f"来源注册表匹配数量为{len(matched)}，应为1")
            authority = ""
            registry_sha = ""
            registry_url = ""
        else:
            registered = matched.iloc[0]
            authority = str(registered["authority_level"])
            registry_sha = str(registered["sha256"])
            registry_url = str(registered["url"])
            if authority != "PRIMARY":
                errors.append("来源权威级别不是PRIMARY")
            if str(registered["validation_status"]) != "PASS":
                errors.append("来源注册状态不是PASS")
            if (str(registered["a_ticker"]), int(registered["report_year"])) != (
                ticker,
                year,
            ):
                errors.append("来源注册表公司年度错配")

        source_url = str(item["source_url"])
        host = (urlparse(source_url).hostname or "").lower()
        if host != "static.cninfo.com.cn":
            errors.append(f"来源域名不是巨潮资讯官方静态域名：{host}")
        if registry_url and registry_url != source_url:
            errors.append("清单与来源注册表URL不一致")
        if str(item["validation_status"]) != "PASS":
            errors.append("年报清单身份验证不是PASS")

        local_file = str(item["local_file"])
        file_path = root / local_file
        expected_sha = str(item["sha256"]).lower()
        actual_sha = sha256(file_path) if file_path.exists() else ""
        if not file_path.exists():
            errors.append("年报本地文件不存在")
        elif actual_sha != expected_sha:
            errors.append("年报文件SHA256与清单不一致")
        if registry_sha and registry_sha.lower() != expected_sha:
            errors.append("清单与来源注册表SHA256不一致")

        row = {
            "ticker": ticker,
            "broker_name": config["companies"][ticker]["broker_name"],
            "year": year,
            "source_id": source_id,
            "authority_level": authority,
            "source_url": source_url,
            "local_file": local_file,
            "file_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "page_count": int(item["page_count"]),
            "source_status": "PASS" if not errors else "BLOCKED_SOURCE_EVIDENCE",
            "source_errors": "；".join(errors),
        }
        rows.append(row)
        lookup[(ticker, year)] = row
    return pd.DataFrame(rows), lookup


def _validate_disclosures(
    config: dict[str, Any], sources: dict[tuple[str, int], dict[str, Any]]
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """验证人工逐页摘录只引用矩阵内官方原文及有效页码。"""

    expected = _expected_pairs(config)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    evidence_ids: set[str] = set()
    for raw in config.get("verified_disclosures", []):
        item = dict(raw)
        pair = (str(item["ticker"]), int(item["year"]))
        if pair not in expected:
            raise RuntimeError(f"逐页披露超出6×5矩阵：{pair}")
        evidence_id = str(item["evidence_id"])
        if evidence_id in evidence_ids:
            raise RuntimeError(f"evidence_id重复：{evidence_id}")
        evidence_ids.add(evidence_id)

        evidence_year = int(item.get("evidence_report_year", pair[1]))
        source_pair = (pair[0], evidence_year)
        if source_pair not in sources:
            raise RuntimeError(f"披露证据未对应官方来源：{source_pair}")
        source = sources[source_pair]
        if source["source_status"] != "PASS":
            raise RuntimeError(f"披露证据来源未通过：{source_pair}")
        pages = [int(page) for page in item.get("pdf_pages", [])]
        if not pages:
            raise RuntimeError(f"逐页披露缺少PDF页码：{evidence_id}")
        invalid = [page for page in pages if page < 1 or page > source["page_count"]]
        if invalid:
            raise RuntimeError(f"逐页披露页码越界：{evidence_id} {invalid}")
        item["evidence_report_year"] = evidence_year
        grouped.setdefault(pair, []).append(item)
    return grouped


def _common_row(
    ticker: str,
    year: int,
    broker_name: str,
    entity_name: str,
    source: dict[str, Any],
    governance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "broker_name": broker_name,
        "year": year,
        "entity_name": entity_name,
        "direct_ownership_pct": None,
        "indirect_ownership_pct": None,
        "consolidated_flag": None,
        "minority_interest_pct": None,
        "shareholder_not_subsidiary_pct": None,
        "group_total_net_profit": None,
        "group_parent_attributable_net_profit": None,
        "international_entity_profit_total": None,
        "international_entity_net_profit": None,
        "group_profit_currency": None,
        "group_profit_unit": None,
        "international_profit_currency": None,
        "international_profit_unit": None,
        "fx_basis": "NATIVE_CURRENCY_NO_CONVERSION",
        "fx_rate": None,
        "recurring_client_driven_profit": None,
        "market_sensitive_profit": None,
        "one_off_profit": None,
        "evidence_id": None,
        "source_id": source["source_id"],
        "evidence_report_year": None,
        "pdf_pages": None,
        "source_local_file": source["local_file"],
        "file_sha256": source["file_sha256"],
        "disclosure_gap": None,
        "g2_status": "G2_BLOCKED_UNVERIFIED_ENTITY_SCOPE",
        "g3_status": "G3_BLOCKED_MISSING_PROFIT_EVIDENCE",
        "stop_status": "BLOCKED_G2_ENTITY_SCOPE_UNVERIFIED",
        "view_status": governance["view_status"],
        "authorization": governance["authorization"],
        "partial_consolidation_period": False,
    }


def build_entity_boundary(
    config: dict[str, Any],
    sources: dict[tuple[str, int], dict[str, Any]],
    disclosures: dict[tuple[str, int], list[dict[str, Any]]],
) -> pd.DataFrame:
    """建立实体级边界表；601211在2025年保留两条实体记录。"""

    rows: list[dict[str, Any]] = []
    governance = config["governance"]
    for ticker, year in sorted(_expected_pairs(config)):
        source = sources[(ticker, year)]
        broker_name = config["companies"][ticker]["broker_name"]
        items = disclosures.get((ticker, year), [])
        if not items:
            candidates = (
                config.get("unverified_entity_candidates", {})
                .get(ticker, {})
                .get(year, [])
            )
            entity_names = candidates or ["UNVERIFIED_OVERSEAS_ENTITY_SCOPE"]
            for entity_name in entity_names:
                row = _common_row(
                    ticker,
                    year,
                    broker_name,
                    str(entity_name),
                    source,
                    governance,
                )
                row["disclosure_gap"] = (
                    "该名称仅为待核候选实体；G2实体、直接/间接持股、并表及少数股东"
                    "边界尚未逐页核验；G3利润桥字段未核验。"
                )
                rows.append(row)
            continue

        for item in items:
            evidence_source = sources[(ticker, int(item["evidence_report_year"]))]
            row = _common_row(
                ticker,
                year,
                broker_name,
                str(item["entity_name"]),
                evidence_source,
                governance,
            )
            for field in (
                "direct_ownership_pct",
                "indirect_ownership_pct",
                "consolidated_flag",
                "minority_interest_pct",
                "shareholder_not_subsidiary_pct",
                "international_entity_profit_total",
                "international_entity_net_profit",
                "international_profit_currency",
                "international_profit_unit",
                "partial_consolidation_period",
            ):
                if field in item:
                    row[field] = item[field]
            row["evidence_id"] = item["evidence_id"]
            row["evidence_report_year"] = item["evidence_report_year"]
            row["pdf_pages"] = ",".join(str(value) for value in item["pdf_pages"])

            if item.get("consolidated_flag") is False:
                row["g2_status"] = "G2_PASS_NEGATIVE_CONTROL"
                row["g3_status"] = "G3_NOT_APPLICABLE_NO_CONSOLIDATED_PLATFORM"
                row["stop_status"] = "G3_BLOCKED_DISCLOSURE_INSUFFICIENT"
                row["disclosure_gap"] = (
                    "已核验33.42%持股主体是股东而非上市公司子公司，且所选合并范围页"
                    "未见并表境外平台；国际子公司利润桥不适用。"
                )
            elif item.get("boundary_verification_complete", True) is False:
                row["g2_status"] = "G2_BLOCKED_INCOMPLETE_CONSOLIDATION_BOUNDARY"
                row["g3_status"] = "G3_BLOCKED_DISCLOSURE_INSUFFICIENT"
                row["stop_status"] = "G3_BLOCKED_DISCLOSURE_INSUFFICIENT"
                row["disclosure_gap"] = (
                    "文本页仅支持直接全资关系及业务功能；并表显式语句、平台净利润、"
                    "币种及视觉复核未完成，不能升级G2通过。"
                )
            else:
                row["g2_status"] = "G2_PASS_ENTITY_CONSOLIDATION_BOUNDARY"
                row["g3_status"] = "G3_BLOCKED_UNRECONCILED_DISCLOSED_SUB_PROFIT"
                row["stop_status"] = "BLOCKED_G3_INCOMPLETE_PROFIT_BRIDGE"
                row["disclosure_gap"] = (
                    "缺集团总净利润、集团归母净利润与国际实体利润的同口径桥接；"
                    "缺经常性客户驱动、市场敏感及一次性利润拆分。"
                )
                if item.get("partial_consolidation_period") is True:
                    row["disclosure_gap"] += (
                        "该实体仅覆盖并购后纳入合并报表期间，不得视为完整年度。"
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def build_profit_bridge(
    entity_table: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """建立严格30行公司年度利润桥；多实体不做未经披露的加总。"""

    rows: list[dict[str, Any]] = []
    for ticker, year in sorted(_expected_pairs(config)):
        subset = entity_table.loc[
            entity_table["ticker"].eq(ticker) & entity_table["year"].eq(year)
        ]
        first = subset.iloc[0].to_dict()
        if len(subset) == 1:
            row = first
        elif subset["g2_status"].str.startswith("G2_BLOCKED").all():
            row = first
            row["entity_name"] = "MULTIPLE_UNVERIFIED_ENTITY_CANDIDATES"
            row["international_entity_profit_total"] = None
            row["international_entity_net_profit"] = None
            row["international_profit_currency"] = None
            row["international_profit_unit"] = None
            row["evidence_id"] = None
            row["pdf_pages"] = None
            row["g3_status"] = "G3_BLOCKED_DISCLOSURE_INSUFFICIENT"
            row["stop_status"] = "BLOCKED_G2_DUAL_PLATFORM_BOUNDARY_UNVERIFIED"
            row["disclosure_gap"] = (
                "保留两个独立候选平台，但持股、并表、少数股东及利润边界均未核实；"
                "禁止合并候选实体或推断利润。"
            )
        else:
            row = first
            row["entity_name"] = "MULTIPLE_ENTITIES_SEE_ENTITY_BOUNDARY_TABLE"
            row["international_entity_profit_total"] = None
            row["international_entity_net_profit"] = None
            row["international_profit_currency"] = None
            row["international_profit_unit"] = None
            row["evidence_id"] = "|".join(subset["evidence_id"].astype(str))
            row["pdf_pages"] = "|".join(subset["pdf_pages"].astype(str))
            row["g3_status"] = "G3_BLOCKED_DUAL_PLATFORM_PARTIAL_PERIOD"
            row["stop_status"] = "BLOCKED_G3_NO_UNDISCLOSED_ENTITY_AGGREGATION"
            row["disclosure_gap"] = (
                "同一公司年度存在两个境外平台，且海通国际仅为并购后部分期间；"
                "未获得同口径抵销与期间数据，因此禁止加总。"
            )
        row["g3_complete"] = False
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix_markdown(bridge: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    years = [int(year) for year in config["protocol"]["matrix_years"]]
    lines = [
        "| 券商 | " + " | ".join(str(year) for year in years) + " |",
        "|---|" + "---|" * len(years),
    ]
    for ticker in config["protocol"]["matrix_tickers"]:
        cells: list[str] = []
        for year in years:
            row = bridge.loc[
                bridge["ticker"].eq(ticker) & bridge["year"].eq(year)
            ].iloc[0]
            if str(row["g2_status"]).startswith("G2_PASS_NEGATIVE"):
                cells.append("G2负对照通过 / G3不适用")
            elif str(row["g2_status"]).startswith("G2_PASS"):
                cells.append("G2通过 / G3阻断")
            else:
                cells.append("G2阻断 / G3阻断")
        name = config["companies"][ticker]["broker_name"]
        lines.append(f"| {name}（{ticker}） | " + " | ".join(cells) + " |")
    return lines


def render_markdown(report: dict[str, Any], bridge: pd.DataFrame, config: dict[str, Any]) -> str:
    """渲染人可读状态，不产生估值、收益或排序结论。"""

    gaps = report["coverage"]["stop_status_counts"]
    gap_lines = [f"- `{key}`：{value}个公司年度" for key, value in gaps.items()]
    return "\n".join(
        [
            "# 国际券商G2→G3证据状态 V1.3",
            "",
            f"- 生成时间：{report['generated_at']}",
            f"- 官方年报文件：{report['sources']['pass_count']} / 30 通过SHA256复核",
            f"- G2公司年度通过：{report['coverage']['g2_pass_company_years']} / 30",
            f"- G2公司年度阻断：{report['coverage']['g2_blocked_company_years']} / 30",
            f"- G3完整利润桥：{report['coverage']['g3_complete_company_years']} / 30",
            f"- 研究视图：`{report['view_status']}`",
            f"- 授权：`{report['authorization']}`",
            "- G6估值、收益结论、510300输入、持仓与下单：`全部禁止`",
            "",
            "## 6家×2021—2025覆盖矩阵",
            "",
            *_matrix_markdown(bridge, config),
            "",
            "## 停止状态",
            "",
            *gap_lines,
            "",
            "## 关键口径",
            "",
            "- 所有金额保持年报原币种、原单位；`fx_basis=NATIVE_CURRENCY_NO_CONVERSION`，未做汇率换算。",
            "- 集团总净利润、集团归母净利润及国际实体净利润只有在同口径证据齐备后才能桥接。当前不得用营业收入、境外资产或营业利润代替利润桥字段。",
            "- 经常性客户驱动、市场敏感及一次性拆分未披露时保持空值；不采用推断填空。",
            "- 国泰海通2025年两个境外平台保持实体级分列；因海通国际是部分并表期间，利润桥不加总。",
            "- 中银证券的中银国际控股为33.42%股东而非其子公司；只在逐年核验通过的年度记录负对照。",
            "",
        ]
    )


def run(root: Path = ROOT) -> dict[str, Any]:
    """运行证据复核、矩阵生成与状态渲染。"""

    config = load_config(root / "config" / "international_broker_g2_g3_v1_3.yaml")
    source_table, sources = verify_official_sources(root, config)
    disclosures = _validate_disclosures(config, sources)
    entity_table = build_entity_boundary(config, sources, disclosures)
    bridge = build_profit_bridge(entity_table, config)

    if len(bridge) != 30 or bridge[["ticker", "year"]].duplicated().any():
        raise RuntimeError("利润桥不是严格30行唯一公司年度矩阵")

    source_pass = source_table["source_status"].eq("PASS")
    g2_pass = bridge["g2_status"].str.startswith("G2_PASS")
    g3_complete = bridge["g3_complete"].eq(True)
    report = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": config["protocol"]["scope"],
        "source_policy": config["protocol"]["source_policy"],
        "view_status": config["governance"]["view_status"],
        "authorization": config["governance"]["authorization"],
        "allow_g6_valuation": False,
        "allow_return_conclusion": False,
        "allow_510300_input": False,
        "allow_position_change": False,
        "sources": {
            "expected_count": 30,
            "pass_count": int(source_pass.sum()),
            "blocked_count": int((~source_pass).sum()),
            "combined_sha256": hashlib.sha256(
                "".join(source_table["file_sha256"].tolist()).encode("ascii")
            ).hexdigest(),
        },
        "coverage": {
            "company_year_count": int(len(bridge)),
            "entity_row_count": int(len(entity_table)),
            "g2_pass_company_years": int(g2_pass.sum()),
            "g2_blocked_company_years": int((~g2_pass).sum()),
            "g3_complete_company_years": int(g3_complete.sum()),
            "g3_blocked_or_not_applicable_company_years": int((~g3_complete).sum()),
            "stop_status_counts": {
                str(key): int(value)
                for key, value in bridge["stop_status"].value_counts().sort_index().items()
            },
        },
        "overall_status": "NO_VIEW_G3_INCOMPLETE",
    }

    paths = config["paths"]
    _atomic_csv(root / paths["source_status_table"], source_table)
    _atomic_csv(root / paths["entity_boundary_table"], entity_table)
    _atomic_csv(root / paths["profit_bridge_table"], bridge)
    _atomic_json(root / paths["audit_json"], report)
    _atomic_text(
        root / paths["audit_markdown"], render_markdown(report, bridge, config)
    )
    return report


def main() -> int:
    report = run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
