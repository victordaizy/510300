"""核对主线行业资料的既有时钟和覆盖；不构造信号或重跑账户。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from research import broker_stage_policy_study_v1 as study
from research import point_first_passage_study_v1 as original

ROOT = study.ROOT
OUT = ROOT / "reports/research/510300_broker_mainline_source_preflight_v1"
SOURCES = (
    ("industry_contribution", "data/features/000300_industry_contribution_daily.parquet", None),
    ("weighted_breadth", "data/features/000300_weighted_breadth_daily.parquet", "date"),
    ("raw_industry", "data/raw/reference/a_share_sw_industry_static.parquet", None),
    ("industry_intervals", "data/staging/a_share_hs_concentrated_low_risk_trend_v1_1/industry_l1_intervals.parquet", None),
    ("monthly_components", "data/curated/510300_structural_equity_risk_premium_engine_v1/csi300_component_monthly_state.parquet", "origin"),
    ("weights", "data/raw/constituents/000300_historical_weights.parquet", "trade_date"),
    ("raw_constituent_daily", "data/raw/constituents/000300_constituent_daily.parquet", "date"),
    ("daily_pit_membership", "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet", "membership_date"),
    ("remediated_constituent_returns", "data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/classified_constituent_returns_remediated.parquet", "date"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if (OUT / "summary.json").exists():
        raise RuntimeError("本来源检查已归档，不重复运行。")
    OUT.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_parquet(study.OUT / "results/全部3488事前阶段资格与上一完整周结构位.parquet", columns=["date"])
    eligible = pd.to_datetime(calendar["date"]).dt.normalize()
    eligible = pd.DatetimeIndex(eligible.loc[eligible.between("2015-01-05", "2026-09-30")].unique()).sort_values()
    if len(eligible) != 2855:
        raise ValueError("原两个完整评价时期日历不一致。")
    inventory = []
    for role, relative, date_column in SOURCES:
        path = ROOT / relative
        item = {"role": role, "path": relative, "exists": path.exists(), "financial_admission": "NOT_ESTABLISHED"}
        if path.exists():
            parquet = pq.ParquetFile(path)
            names = parquet.schema_arrow.names
            item.update({"rows": parquet.metadata.num_rows, "bytes": path.stat().st_size, "columns": names,
                         "metadata_only_is_not_source_certification": True})
            if date_column is not None:
                if date_column in names:
                    values = pc.unique(parquet.read(columns=[date_column]).column(date_column)).to_pandas()
                    parsed = pd.to_datetime(values, errors="coerce")
                    dates = pd.DatetimeIndex(parsed.dropna()).normalize().unique().sort_values()
                    overlap = eligible.intersection(dates)
                    missing = eligible.difference(dates)
                    item["calendar_presence"] = {"date_column": date_column, "min": str(dates.min().date()) if len(dates) else None,
                        "max": str(dates.max().date()) if len(dates) else None, "distinct_dates": len(dates),
                        "evaluation_calendar_dates_present": len(overlap), "evaluation_calendar_dates_without_rows": len(missing),
                        "missing_dates": [value.date().isoformat() for value in missing],
                        "meaning": "仅该表直接日期存在性；月度权重不能用此项判为逐日缺失；有日期不等于全部成分/行业齐全。"}
                else:
                    item["calendar_presence"] = {"status": "DATE_COLUMN_ABSENT", "requested_column": date_column}
            if role in {"industry_intervals", "raw_industry", "monthly_components"}:
                item["sha256"] = sha256(path)
        inventory.append(item)
    intervals_path = ROOT / dict((role, path) for role, path, _ in SOURCES)["industry_intervals"]
    intervals = pd.read_parquet(intervals_path)
    valid_from = pd.to_datetime(intervals.valid_from, errors="coerce")
    available = pd.to_datetime(intervals.available_at, errors="coerce")
    clock = {"rows": len(intervals), "symbols": int(intervals.ts_code.nunique()),
        "industries": int(intervals.industry_l1.nunique()),
        "valid_from_unknown": int(valid_from.isna().sum()), "available_at_unknown": int(available.isna().sum()),
        "generated_effective_close_clock_rows": int(available.eq(valid_from.dt.normalize() + pd.Timedelta(hours=15)).sum()),
        "evidence": "scripts/collect_a_share_hs_concentrated_low_risk_trend_v1_1_inputs.py::build_industry 显式调用 at_close(valid_from)",
        "historical_first_release_certified": False,
        "interpretation": "有效区间可供解释；available_at为程序根据生效日生成，不能据列名当成真实历史首次公布时刻。"}
    collector = ROOT / "scripts/collect_a_share_hs_concentrated_low_risk_trend_v1_1_inputs.py"
    collector_text = collector.read_text(encoding="utf-8-sig")
    needle = 'frame["available_at"] = at_close(frame["valid_from"])'
    if needle not in collector_text:
        raise ValueError("已核对的行业时钟构建逻辑发生变化，需重读来源。")
    code_lines = collector_text.splitlines()
    evidence = [{"line": i + 1, "text": line} for i, line in enumerate(code_lines) if needle in line]
    builder = ROOT / "research/build_weighted_breadth_dataset.py"
    monthly = next(row for row in inventory if row["role"] == "monthly_components")
    monthly_clock_fields = [name for name in monthly.get("columns", []) if "industry" in name and ("available" in name or "source" in name)]
    summary = {"at": original.now(), "status": "NOT_ADMITTED_INDUSTRY_MAINLINE_FINANCIAL_INPUTS_REQUIRE_CLOCK_AND_COVERAGE",
        "scope": "LOCAL_SAVED_TABLES_AND_BUILDER_PROVENANCE_ONLY_NO_NEW_MARKET_REQUESTS",
        "financial_parent_decision": "TECH.R212", "financial_parent_result": str(study.OUT.absolute().relative_to(ROOT) / "summary.json"),
        "evaluation_calendar_days": len(eligible), "inventory": inventory, "industry_clock": clock,
        "industry_clock_code_evidence": {"path": str(collector.relative_to(ROOT)), "sha256": sha256(collector), "lines": evidence},
        "industry_builder_evidence": {"path": str(builder.relative_to(ROOT)), "sha256": sha256(builder),
            "requires": ["industry_l1", "industry_asof_date", "industry_source"],
            "existing_function_does_not_prove_saved_industry_dataset_exists": True},
        "monthly_industry_specific_clock_or_source_fields": monthly_clock_fields,
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_requests": 0,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
        "next_experiment": "先明确行业分类生效与首发时钟、版本、当日成员/权重及日线覆盖；缺口保持未知。随后固定宽基匹配/扩散用途及对照，不按R212的亏损来源删除路线。",
        "not_a_family_rejection": "当前表未准入该金融用途，不等于行业/主线机制不可行；可另立不依赖行业分类的成分扩散观察用途。"}
    study.write(OUT / "summary.json", summary)
    lines = ["# 主线与510300匹配：现有来源检查", "", "2026-10-05。本检查只读本地保存表及构建程序，不构造信号、不改TECH.R212策略、不运行新账户。", "",
        "结论：现有行业资料尚不能直接取得本轮金融用途准入。行业区间表7536行的available_at由生效日当天15:00生成；这是一种日期代理，不是真实历史首次公布认证。月度成分表里的财务披露钟不能自动证明行业字段的可知性。", "",
        "| 来源 | 实际存在 | 行数 | 日期覆盖说明 |", "|---|---|---:|---|"]
    for item in inventory:
        coverage = item.get("calendar_presence", {})
        if "distinct_dates" in coverage:
            detail = f"{coverage['min']}至{coverage['max']}，直接覆盖评价日{coverage['evaluation_calendar_dates_present']}/2855"
        else:
            detail = coverage.get("status", "未按日历列检查")
        lines.append(f"| {item['role']} | {'是' if item['exists'] else '否'} | {item.get('rows', 'UNKNOWN')} | {detail} |")
    lines += ["", "上表只核对日期出现，尚未证明全部证券/行业齐全。月度权重需要按已知时刻承接，不能把无逐日快照直接当作缺失；成员和股票回报存在日期也不等于复权、停牌、首版和版本已认证。", "",
        f"行业钟实际扫描：{clock['generated_effective_close_clock_rows']}/{clock['rows']}行与生效日15:00精确相同；构建代码及哈希见summary.json。原行业贡献日表实际不存在，不因仓库有构建函数而声称已具备该输入。", "",
        "下一步先核对分类公告/版本与行业日线可用性，再预登记主线对宽基的传导和归因对照。也可另立不依赖行业分类的成分扩散观察，所有缺口保留未知，不事后选赢家或改变失败政策。本次未形成已准入待跑金融候选，不宣称新的胜率、收益或夏普。", "",
        "[完整来源、日历缺口和构建证据](summary.json)"]
    (OUT / "主线匹配_来源覆盖与时钟检查.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"主线来源检查完成：{len(inventory)}项本地文件，行业日期代理{clock['generated_effective_close_clock_rows']}/{clock['rows']}；未准入金融用途，0新账户。", flush=True)


def amend_membership_date_column() -> None:
    """更正本检查误请求的日期列；保留原检查，不改变任何源表。"""
    receipt = OUT / "membership_calendar_schema_amendment.json"
    if receipt.exists():
        raise RuntimeError("成员日期列说明已更正，不重复执行。")
    summary_path = OUT / "summary.json"
    old = summary_path.read_bytes()
    summary = json.loads(old.decode("utf-8-sig"))
    item = next(row for row in summary["inventory"] if row["role"] == "daily_pit_membership")
    if item["calendar_presence"] != {"status": "DATE_COLUMN_ABSENT", "requested_column": "date"}:
        raise ValueError("本次只更正已保存的误请求日期列，不扩展其他来源。")
    dates = pc.unique(pq.ParquetFile(ROOT / item["path"]).read(columns=["membership_date"]).column("membership_date")).to_pandas()
    dates = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna()).normalize().unique().sort_values()
    eligible = pd.read_parquet(study.OUT / "results/全部3488事前阶段资格与上一完整周结构位.parquet", columns=["date"])
    calendar = pd.DatetimeIndex(pd.to_datetime(eligible.loc[eligible.date.between("2015-01-05", "2026-09-30"), "date"])).normalize().unique().sort_values()
    missing = calendar.difference(dates)
    item["calendar_presence"] = {"date_column": "membership_date", "min": dates.min().date().isoformat(),
        "max": dates.max().date().isoformat(), "distinct_dates": len(dates),
        "evaluation_calendar_dates_present": len(calendar.intersection(dates)),
        "evaluation_calendar_dates_without_rows": len(missing), "missing_dates": [value.date().isoformat() for value in missing],
        "meaning": "成员表真实日期列为membership_date；日期存在不等于来源首版已认证或回报齐全。"}
    summary["membership_calendar_schema_amendment"] = receipt.name
    with (OUT / "summary_before_membership_calendar_schema_amendment.json").open("xb") as stream:
        stream.write(old)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path = OUT / "主线匹配_来源覆盖与时钟检查.md"
    report_old = report_path.read_bytes()
    with (OUT / "report_before_membership_calendar_schema_amendment.md").open("xb") as stream:
        stream.write(report_old)
    old_line = f"| daily_pit_membership | 是 | {item['rows']} | DATE_COLUMN_ABSENT |"
    coverage = item["calendar_presence"]
    new_line = f"| daily_pit_membership | 是 | {item['rows']} | {coverage['min']}至{coverage['max']}，直接覆盖评价日{coverage['evaluation_calendar_dates_present']}/2855 |"
    report_text = report_old.decode("utf-8")
    if report_text.count(old_line) != 1:
        raise ValueError("旧来源报告成员行不唯一。")
    report_path.write_text(report_text.replace(old_line, new_line), encoding="utf-8")
    study.write(receipt, {"at": original.now(), "status": "CORRECTED_REQUESTED_DATE_COLUMN_ONLY_SOURCE_UNCHANGED",
        "old_summary_sha256": hashlib.sha256(old).hexdigest(), "new_summary_sha256": sha256(summary_path),
        "old_requested_column": "date", "actual_column": "membership_date", "calendar_presence": coverage,
        "financial_admission_unchanged": summary["status"], "new_accounts": 0, "new_fits": 0, "new_market_requests": 0})
    print(f"成员日期列已更正为membership_date；实际覆盖{coverage['evaluation_calendar_dates_present']}/2855日，保留原说明与源表。", flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--amend-membership-date-column"]:
        amend_membership_date_column()
    elif not sys.argv[1:]:
        main()
    else:
        raise ValueError("只支持首次检查或已保存成员日期列的精确更正。")
