"""逐条核对参与研究的报告增长事实，固定有限原件样本。"""
from __future__ import annotations
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_eps_growth_disagreement_increment_v1"
CONFIG = ROOT / "config/510300_eps_growth_disagreement_increment_v1.json"
SOURCES = {"guosen": ("510300_forward_eps_csi_facts_v2", "510300_forward_eps_csi_originals_v1", "80000007"),
           "soochow": ("510300_forward_eps_soochow_facts_v4", "510300_forward_eps_soochow_originals_v1", "80000031")}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    config = read(CONFIG)
    source = ROOT / config["source_feasibility"]
    pair_path, monthly_path = source / "company_pairs.parquet", source / "monthly_source_feasibility.parquet"
    save(OUT / "source_check_claim.json", {"started_at": now(), "new_return_labels_read": False,
        "files": [identity(p) for p in [CONFIG, Path(__file__), pair_path, monthly_path]]})
    pairs = pd.read_parquet(pair_path)
    months = pd.read_parquet(monthly_path)
    eligible = set(months.loc[months.eligible_for_next_protocol, "origin"])
    used = pairs.loc[pairs.origin.isin(eligible) & pairs.paired_growth_available].sort_values(["origin", "ts_code"]).reset_index(drop=True)
    cache, files, checks, definitions = {}, {}, [], Counter()
    for i, row in enumerate(used.itertuples(index=False)):
        for institution, (fact_root, original_root, institution_code) in SOURCES.items():
            aid = getattr(row, "report_id_" + institution)
            key = (institution, aid)
            if key not in cache:
                fact_path = ROOT / "reports/research" / fact_root / "document_facts" / (aid + ".json")
                metadata_path = ROOT / "reports/research" / original_root / "document_records" / (aid + ".json")
                facts, record = read(fact_path), read(metadata_path)
                cache[key] = (facts, record)
                for path in (fact_path, metadata_path):
                    files[path.relative_to(ROOT).as_posix()] = identity(path)
            facts, record = cache[key]
            if facts["ts_code"] != row.ts_code or record["ts_code"] != row.ts_code or str(record["provider_metadata"]["company_code"]) != institution_code:
                raise ValueError("公司或机构身份不一致")
            yearly = {int(f["target_fiscal_year"]): f for f in facts["facts"]}
            first, second = yearly[row.origin.year], yearly[row.origin.year + 1]
            if first["eps_unit_as_reported"] != second["eps_unit_as_reported"] or first["source_eps_label"] != second["source_eps_label"]:
                raise ValueError("同报告两年度的EPS单位或行定义改变")
            if first["source_eps_row"] != second["source_eps_row"]:
                raise ValueError("两年度并非同一份原表EPS行")
            a, b = float(first["eps_value_exact"]), float(second["eps_value_exact"])
            denominator = abs(a) + abs(b)
            if not np.isfinite([a, b]).all() or denominator == 0:
                raise ValueError("源EPS数值无法产生有限对称增长")
            expected = 2 * (b - a) / denominator
            saved = float(getattr(row, "eps_growth_" + institution))
            if abs(expected - saved) > 1e-14:
                raise ValueError("原事实的年度EPS不能重现保存增长值")
            meta = record["provider_metadata"]
            available = max(record["directory_record"]["publishDate"][:10], meta["notice_date"][:10], meta["eitime"][:10], facts["report_internal_date"], facts["conservative_information_date"])
            if available != str(getattr(row, "information_date_" + institution))[:10] or not pd.Timestamp(available) < row.origin:
                raise ValueError("保存可用日期与原来源不一致")
            if not 1 <= (row.origin - pd.Timestamp(available)).days <= 180:
                raise ValueError("参与预测来源超出报告年龄")
            if first.get("actual_future_eps_used_as_predictor") or second.get("actual_future_eps_used_as_predictor"):
                raise ValueError("来源标记为实际未来EPS")
            definitions[institution + "|" + first.get("eps_definition", "未记载")] += 1
            checks.append({"origin": row.origin, "ts_code": row.ts_code, "institution": institution, "report_id": aid,
                           "current_fiscal_year": row.origin.year, "next_fiscal_year": row.origin.year + 1,
                           "current_eps_exact": first["eps_value_exact"], "next_eps_exact": second["eps_value_exact"],
                           "source_unit": first["eps_unit_as_reported"], "source_label": first["source_eps_label"],
                           "source_definition": first.get("eps_definition", "未记载"),
                           "information_date": available, "recomputed_growth": expected, "saved_growth": saved,
                           "absolute_error": abs(expected - saved)})
        if (i + 1) % 500 == 0:
            print(f"已核对 {i + 1}/{len(used)} 个公司月份的两机构原始预测事实。", flush=True)
    choices = [used.iloc[0], used.iloc[-1], used.sort_values(["absolute_growth_disagreement", "origin", "ts_code"], ascending=[False, True, True]).iloc[0]]
    samples = {}
    for reason, row in zip(["FIRST", "LAST", "MAX_DISAGREEMENT"], choices):
        for institution in SOURCES:
            aid = row["report_id_" + institution]
            facts, record = cache[(institution, aid)]
            fact = next(f for f in facts["facts"] if f["target_fiscal_year"] == row.origin.year + 1)
            pdf = ROOT / record["source"]["raw_pdf_path"]
            pdf_identity = identity(pdf)
            if pdf_identity["sha256"] != record["source"]["pdf_sha256"] or pdf_identity["sha256"] != fact["pdf_sha256"]:
                raise ValueError("固定样本原PDF哈希不符")
            sample = samples.setdefault((institution, aid), {"report_id": aid, "institution": institution, "symbol": row.ts_code,
                "pdf": pdf_identity, "page": fact["source_page"], "source_eps_row": fact["source_eps_row"], "header_raw": fact["header_raw"],
                "source_bbox_pdf_points": fact.get("source_bbox_pdf_points"), "selection_reasons": []})
            sample["selection_reasons"].append(reason)
            files[pdf_identity["path"]] = pdf_identity
            for path in [ROOT / record["source"]["raw_html_path"], ROOT / record["directory_record"]["source_raw_path"]]:
                files[path.relative_to(ROOT).as_posix()] = identity(path)
    frame = pd.DataFrame(checks)
    frame.to_parquet(OUT / "source_fact_checks.parquet", index=False)
    save(OUT / "source_validation.json", {"status": "PASS_USED_EPS_ROWS_RECOMPUTED_CLOCKS_AND_WITHIN_REPORT_UNITS_CHECKED",
        "completed_at": now(), "eligible_source_months": len(eligible), "paired_company_months": len(used),
        "institution_company_month_checks": len(checks), "unique_reports": len(cache),
        "maximum_absolute_growth_error": float(frame.absolute_error.max()), "definition_counts": dict(definitions),
        "raw_pdf_samples": list(samples.values()), "files": sorted(files.values(), key=lambda x: x["path"]),
        "all_raw_pdfs_visually_rechecked": False, "cross_institution_basic_diluted_definition_equivalence_proven": False,
        "predictor_interpretation": "各报告口径的年度增长差异，非完全统一会计口径的纯盈利分歧",
        "new_return_labels": 0, "new_model_fits": 0, "new_accounts": 0})
    print(json.dumps({"status": "来源事实与时钟核对完成", "company_months": len(used), "unique_reports": len(cache), "raw_pdf_samples": len(samples)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
