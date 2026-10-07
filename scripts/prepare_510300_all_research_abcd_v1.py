"""汇总本地510300研究原始索引，并为有限A/B/C/D实验保存输入快照。"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_all_research_abcd_increment_v1"
MATCH = re.compile(r"510300|000300|csi300|index_driver|index_structure|sector_forward_return|r6_|round5_|return_tail|research_registry|research_authority", re.I)
STATUS_KEYS = {"status", "research_status", "decision", "disposition", "goal_achieved", "independent_validation", "evidence_class", "study_id", "project_id"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8-sig")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def family(name):
    groups = [
        ("资料、权限与交付", r"source|data_|admission|delivery|verification|preflight|authority|readiness|governance|forward_host|publication|adapter"),
        ("形态与市场状态", r"pattern|regime|state|breakout|reclaim|compression|trend|reversal|rebound|macd|donchian"),
        ("盈利、估值与财报", r"eps|earnings|financial|valuation|fundamental|profit_attribution|pb_roe|normalized"),
        ("宏观、政策与资金", r"macro|policy|lpr|m1_m2|money|reverse_repo|fiscal|growth_state|fund_|subscription|private|participant|capital_adjustment|equity_support"),
        ("期权、期货与跨市场", r"option|futures|_if_|offshore|global|ah_premium|cross_etf|cross_market"),
        ("指数内部与驱动", r"driver|breadth|constituent|absorption|dispersion|sector|index_structure|component|internal"),
        ("退出、仓位与风险", r"exit|risk|budget|allocation|exposure|blend|mix|reference|volatility|downside|account|drawdown|continuation|holding|rebalance|session|overnight|intraday"),
        ("预测与收益诊断", r"forecast|prediction|probability|payoff|return|node|sharpe|support|diagnostic"),
    ]
    for label, pat in groups:
        if re.search(pat, name, re.I):
            return label
    return "其他相关研究与基础资料"


def statuses(value, prefix="", depth=0):
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else k
            if k in STATUS_KEYS and isinstance(v, (str, bool, int, float)):
                found.append(f"{key}={v}")
            elif depth < 2 and isinstance(v, dict):
                found.extend(statuses(v, key, depth + 1))
    return found


def main():
    if (OUT / "inventory/coverage.json").exists():
        raise RuntimeError("本次资料快照已经建立，不覆盖原快照。")
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {ROOT / "RESEARCH_STATUS.md", ROOT / "CONTEXT.md"}
    for folder in ["docs", "config", "research", "scripts"]:
        for p in (ROOT / folder).iterdir():
            if p.is_file() and MATCH.search(p.name) and "all_research_abcd" not in p.name:
                paths.add(p)
    study_rows = []
    for section in (ROOT / "reports").iterdir():
        if not section.is_dir():
            continue
        for p in section.iterdir():
            if p == OUT:
                continue
            if p.is_file() and MATCH.search(p.name):
                paths.add(p)
            elif p.is_dir() and MATCH.search(p.name):
                immediate = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".md", ".json", ".csv", ".txt", ".yaml", ".yml"}]
                st = []
                primary = []
                for f in immediate:
                    paths.add(f)
                    if f.name in {"result.json", "summary.json", "acceptance_outcome.json", "research_disposition.json", "continuation_status.json", "execution_receipt.json"}:
                        try:
                            st.extend(statuses(json.loads(f.read_text(encoding="utf-8-sig"))))
                            primary.append(f.relative_to(ROOT).as_posix())
                        except (ValueError, UnicodeError):
                            st.append(f"PARSE_ERROR:{f.name}")
                for sub in ["results", "evidence"]:
                    q = p / sub
                    if q.is_dir():
                        for f in q.iterdir():
                            if f.is_file() and f.suffix.lower() in {".md", ".json", ".csv"} and f.stat().st_size <= 2_000_000:
                                paths.add(f)
                                if f.name in {"summary.json", "result.json"}:
                                    try:
                                        st.extend(statuses(json.loads(f.read_text(encoding="utf-8-sig")), sub))
                                        primary.append(f.relative_to(ROOT).as_posix())
                                    except (ValueError, UnicodeError):
                                        st.append(f"PARSE_ERROR:{sub}/{f.name}")
                study_rows.append({"study_directory": p.relative_to(ROOT).as_posix(), "family": family(p.name), "direct_document_count": len(immediate), "saved_statuses": " | ".join(st), "status_evidence": " | ".join(primary), "status_note": "原始状态原样保存；目录不等于独立实验。" if st else "未提取到终态，不能视为完成或失败。"})
    records = []
    evidence_bytes = 0
    for p in sorted(paths):
        rel = p.relative_to(ROOT).as_posix()
        b = p.read_bytes()
        rec = {"path": rel, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(), "family": family(rel), "copy_in_packet": False, "saved_statuses": "", "read_scope": "SOURCE_CODE_INDEX_ONLY"}
        if p.suffix.lower() in {".json", ".md", ".yaml", ".yml", ".csv", ".jsonl", ".txt"} and len(b) <= 2_000_000:
            text = b.decode("utf-8-sig", errors="replace")
            rec["read_scope"] = "DOCUMENT_SNAPSHOT_AND_MACHINE_EXTRACTION_NOT_MANUAL_LINE_BY_LINE_REVIEW"
            if p.suffix.lower() == ".json":
                try:
                    rec["saved_statuses"] = " | ".join(statuses(json.loads(text)))
                except ValueError:
                    rec["saved_statuses"] = "PARSE_ERROR_PRESERVED"
            dst = OUT / "legacy_evidence" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b)
            rec["copy_in_packet"] = True
            evidence_bytes += len(b)
        records.append(rec)
    old_index = json.loads((ROOT / "reports/research/510300_sharpe_1_2_latest_research.json").read_text(encoding="utf-8-sig"))
    rounds = old_index["completed_rounds"]
    round_rows = []
    for r in rounds:
        rr = {k: r.get(k) for k in ["round", "study", "title", "status", "result", "candidate_configurations", "new_accounts_generated", "evaluation_accounts", "new_model_fits"]}
        rr["family"] = family(r.get("study", ""))
        for cost, key in [("BASE", "primary_base"), ("STRESS", "primary_stress")]:
            m = r.get(key) or {}
            if isinstance(m, dict):
                for field in ["model", "annualized_return", "net_sharpe", "max_drawdown", "trading_days"]:
                    rr[f"{cost}_{field}"] = m.get(field)
        round_rows.append(rr)
    write_csv(OUT / "inventory/artifact_inventory.csv", records)
    write_csv(OUT / "inventory/study_directory_inventory.csv", study_rows)
    write_csv(OUT / "inventory/legacy_216_rounds.csv", round_rows)
    write_json(OUT / "inventory/coverage.json", {"created_at": now(), "workspace": str(ROOT), "artifacts": len(records), "study_directories": len(study_rows), "legacy_numbered_round_records": len(rounds), "copied_evidence_files": sum(r["copy_in_packet"] for r in records), "copied_evidence_bytes": evidence_bytes, "scope": "本地原始研究索引216轮、报告一级研究目录、相关docs/config以及代码文件索引；不把目录数或重复指标数当独立试验数。", "excluded": ["嵌套交付包和验证解压副本不重复计数", "大体积原始资料、模型和旧账户仅索引其原研究报告与结果，不称全仓库重跑", "本次实验直接输入及完整账户另行全量保存", "无原始终态的V3等项目保留未完成"], "manual_review": "核心机制和本次输入逐项阅读，其余完整索引及原文快照可供检索，不宣称逐行人工复审。"})
    source_map = {
        "features.parquet": "reports/research/510300_sequential_patterns_2021_2026_v2/inputs/features.parquet",
        "signals.parquet": "reports/research/510300_sequential_patterns_2021_2026_v2/inputs/signals.parquet",
        "dividends.csv": "reports/research/510300_sequential_patterns_2021_2026_v2/inputs/dividends.csv",
        "prices.parquet": "reports/research/510300_sequential_patterns_2021_2026_v2/inputs/prices.parquet",
        "calendar.csv": "reports/research/510300_sequential_patterns_2021_2026_v2/inputs/calendar.csv",
        "historical_calendar.csv": "reports/research/510300_sequential_patterns_2021_2026_v2/inputs/historical_calendar.csv",
        "cross_section.parquet": "data/features/000300_point_in_time_index_cross_section_daily_v1_3.parquet",
        "attribution_quality.parquet": "data/features/000300_index_driver_summary_daily_v1_3.parquet",
        "industry_attribution.parquet": "data/features/000300_industry_driver_attribution_daily_v1_3.parquet",
        "attribution_quality.json": "reports/data_quality/000300_index_driver_attribution_v1_3_status.json",
        "current_mandate.json": "config/510300_existing_data_training_mandate_v1.json",
    }
    copied = []
    for name, rel in source_map.items():
        source, dest = ROOT / rel, OUT / "inputs" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        copied.append({"source": rel, "snapshot": dest.relative_to(OUT).as_posix(), "bytes": dest.stat().st_size, "sha256": sha(dest)})
    code = OUT / "code"
    code.mkdir(exist_ok=True)
    for src, dst in [(ROOT / "research/sequential_patterns_regime_v1.py", code / "parent_engine.py"), (Path(__file__), code / Path(__file__).name)]:
        shutil.copy2(src, dst)
    attach = Path(r"E:\CodexData\.codex\attachments\018ff8b7-a313-4e9a-934d-f55cd139ee9e\pasted-text-1.txt")
    shutil.copy2(attach, OUT / "用户附件原文.txt")
    (OUT / "用户请求.md").write_text("# 本轮用户请求\n\n/goal 综合510300所有研究\n\n范围澄清：综合后继续执行附件中的 A/B/C/D 增量实验。\n\n附带原文完整保存于用户附件原文.txt。\n", encoding="utf-8")
    write_json(OUT / "source_manifest.json", {"created_at": now(), "inputs": copied})
    print(json.dumps({"输出目录": str(OUT), "原轮次": len(rounds), "研究目录": len(study_rows), "索引文件": len(records), "原文快照": sum(r["copy_in_packet"] for r in records), "快照字节": evidence_bytes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
