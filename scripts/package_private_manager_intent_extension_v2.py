"""封存资料扩样，并从独立解压目录复算样本边界；不拟合模型。"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STUDY = "510300_private_manager_intent_source_extension_v2"
ZIP_NAME = "510300_私募意向资料扩样_V2_GPT审阅_20260922.zip"
STATUS = "ADMITTED_HISTORICAL_RECONSTRUCTION"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(root: Path) -> dict:
    root = root.resolve()
    result = load(root / "result.json")
    old = [r for r in load(root / "parent_snapshot/admitted_sources.json") if r["status"] == STATUS]
    added = load(root / "new_admitted_sources.json")
    combined = load(root / "combined_admitted_sources.json")
    protocol = load(root / "parent_snapshot/protocol.json")
    plan = load(root / "source_plan.json")
    mandate = load(root / "mandate.json")
    freeze = load(root / "source_freeze_receipt.json")
    receipts = load(root / "来源下载回执_无正文.json")
    receipt_by_key = {r["key"]: r for r in receipts}
    require(len(receipt_by_key) == len(receipts), "下载回执重复。")
    require(combined == sorted(old + added, key=lambda r: r["month"]), "旧记录或合并内容改变。")
    require(not ({r["month"] for r in old} & {r["month"] for r in added}), "新增与旧月份重叠。")
    by_month = {r["month"]: r for r in combined}
    require(len(by_month) == len(combined), "合并月份重复。")
    for row in combined:
        date.fromisoformat(row["available_date"])
        require(row["available_date"][:7] <= row["month"], "出现迟到来源。")
        require(row["status"] == STATUS and row["archived_first_version"] is False, "历史资料身份不一致。")
    for row in added:
        receipt = receipt_by_key[row["key"]]
        require(receipt["status"] == "SAVED", "准入来源没有保存成功。")
        require(row["source_sha256"] == receipt["sha256"] and row["url"] == receipt["url"], "来源身份不一致。")
    for name, expected_hash in freeze["frozen_files"].items():
        require(digest((root / name).read_bytes()) == expected_hash, f"冻结文件改变：{name}")
    require(freeze["source_set"] == [{"key": r["key"], "source_sha256": r["source_sha256"]} for r in added], "冻结来源集合不同。")

    expected_months = [f"{y:04d}-{m:02d}" for y in range(2014, 2027) for m in range(1, 13) if f"{y:04d}-{m:02d}" <= "2026-08"]
    coverage = read_csv(root / "全部月份覆盖与缺口.csv")
    require([r["month"] for r in coverage] == expected_months, "完整月份母集不一致。")
    for row in coverage:
        source = by_month.get(row["month"])
        if source is None:
            require(row["status"] == "MISSING_OR_ORIGINAL_EXCLUSION_PRESERVED", "缺失月份状态改变。")
            require(all(row[k] == "" for k in ("available_date", "plan_index", "survey_exposure_pct")), "缺失月被填值。")
        else:
            require(row["status"] == STATUS and row["available_date"] == source["available_date"], "月份状态或时钟不一致。")
            require(float(row["plan_index"]) == source["plan_index"], "计划指数不一致。")
            exposure = None if row["survey_exposure_pct"] == "" else float(row["survey_exposure_pct"])
            require(exposure == source.get("survey_exposure_pct"), "调查仓位不一致。")
    added_csv = read_csv(root / "新增月份与来源.csv")
    require([r["month"] for r in added_csv] == [r["month"] for r in added], "新增表月份不同。")
    for displayed, saved in zip(added_csv, added):
        for key, value in saved.items():
            require(displayed[key] == ("" if value is None else str(value)), f"新增表内容不同：{saved['month']} {key}")

    train = int(protocol["estimator"]["minimum_mature_training_events"])
    evaluation = int(protocol["promotion_gate"]["minimum_evaluation_events"])
    complete = [r for r in combined if r.get("survey_exposure_pct") is not None]
    recomputed = {
        "parent_admitted_events": len(old), "new_admitted_events": len(added),
        "combined_admitted_events": len(combined), "combined_intent_and_exposure_events": len(complete),
        "full_month_universe": len(expected_months), "missing_or_original_excluded_months": len(expected_months) - len(combined),
        "new_source_urls_attempted": len(receipts), "new_sources_saved": sum(r["status"] == "SAVED" for r in receipts),
        "new_sources_failed": sum(r["status"] != "SAVED" for r in receipts),
        "minimum_mature_training_events": train, "minimum_evaluation_events": evaluation,
        "upper_bound_evaluation_B": max(0, len(combined) - train), "upper_bound_evaluation_C": max(0, len(complete) - train),
        "sample_gate_B": len(combined) - train >= evaluation, "sample_gate_C": len(complete) - train >= evaluation,
        "plan_above100": sum(r["plan_index"] > 100 for r in combined),
        "plan_equal100": sum(r["plan_index"] == 100 for r in combined), "plan_below100": sum(r["plan_index"] < 100 for r in combined),
        "questioned_internal_components_months": [r["month"] for r in added if r["survey_components_internally_questioned"]],
        "parent_records_unchanged": all(r in combined for r in old),
    }
    for key, value in recomputed.items():
        require(result[key] == value, f"结果计数不一致：{key}")
    require(train == 24 and evaluation == 24 and not result["sample_gate_B"] and not result["sample_gate_C"], "样本边界与本轮状态不符。")
    require(len(receipts) <= plan["download_limit_new_unique_urls"], "来源数量超过预定预算。")
    for key in ("new_market_outcome_reads", "new_model_fits", "new_bootstraps", "new_accounts", "strict_forward_events", "position_impact"):
        require(result[key] == 0, f"活动计数不符：{key}")
    for key in ("net_sharpe", "annualized_return", "max_drawdown"):
        require(result[key] == "NOT_COMPUTED", "未运行指标不得填零。")
    require(result["account_stage"] == "NOT_RUN_PARENT_PREDICTIVE_SAMPLE_GATE", "账户门状态不同。")
    require(result["whole_goal_achieved"] is False and result["goal_status"] == "ACTIVE", "整体目标状态不同。")
    require(mandate["initial_capital_cny"] == 200000 and mandate["net_sharpe_minimum"] == 1.5 and mandate["maximum_drawdown_magnitude"] == 0.1, "用户约束不同。")
    require(mandate["executable_assets"] == ["510300.SH", "CASH_CNY"], "交易标的不同。")

    index = read_csv(root / "FILE_INDEX.csv")
    indexed = {r["path"]: r for r in index}
    require(len(indexed) == len(index), "索引路径重复。")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == set(indexed) | {"FILE_INDEX.csv"}, "文件与索引集合不相等。")
    for name, row in indexed.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "索引路径超出解压目录。")
        data = path.read_bytes()
        require(len(data) == int(row["bytes"]) and digest(data) == row["sha256"], f"索引内容不一致：{name}")
    clean = [r for r in combined if not r.get("survey_components_internally_questioned", False)]
    return {
        "status": "PASS_SAVED_SOURCE_COVERAGE_RECOMPUTATION", "indexed_files": len(index), "recomputed": recomputed,
        "excluding_questioned_upper_B": max(0, len(clean) - train),
        "excluding_questioned_upper_C": max(0, sum(r.get("survey_exposure_pct") is not None for r in clean) - train),
        "scope": "离线保存列表、月份母集、冻结身份、计数上限与索引；未复现网页原文或账户收益",
    }


def package() -> None:
    report = ROOT / "reports/research" / STUDY
    target = ROOT / "deliverables" / ZIP_NAME
    building = target.with_suffix(".building.zip")
    extract = report / "verification/fresh_extract"
    require(not target.exists() and not building.exists() and not extract.exists(), "交付或复核目录已存在，请另立版本。")
    members = {p.name: p.read_bytes() for p in report.iterdir() if p.is_file() and p.name not in {"download_summary.json", "delivery_receipt.json", "FILE_INDEX.csv"}}
    for path in (report / "parent_snapshot").iterdir():
        if path.is_file():
            members[path.relative_to(report).as_posix()] = path.read_bytes()
    members["code/private_manager_intent_source_extension_v2.py"] = (ROOT / "research/private_manager_intent_source_extension_v2.py").read_bytes()
    members["code/package_private_manager_intent_extension_v2.py"] = Path(__file__).read_bytes()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, ["path", "bytes", "sha256"])
    writer.writeheader()
    for name, data in sorted(members.items()):
        writer.writerow({"path": name, "bytes": len(data), "sha256": digest(data)})
    members["FILE_INDEX.csv"] = stream.getvalue().encode("utf-8-sig")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    extract.mkdir(parents=True)
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and archive.testzip() is None, "ZIP成员或CRC检查失败。")
        require(set(names) == set(members), "ZIP成员集合不同。")
        for name in names:
            destination = (extract / name).resolve()
            require(destination.is_relative_to(extract.resolve()), "ZIP路径超出目标目录。")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
    completed = subprocess.run([sys.executable, "-X", "utf8", str(extract / "code/package_private_manager_intent_extension_v2.py"), "--verify-root", str(extract)], check=True, capture_output=True, text=True, encoding="utf-8")
    validation = json.loads(completed.stdout)
    write_json(report / "verification/recomputation.json", validation)
    building.replace(target)
    receipt = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "archive": str(target),
        "bytes": target.stat().st_size, "sha256": digest(target.read_bytes()),
        "members": len(members), "indexed_files": len(members) - 1,
        "zip_crc_duplicates_and_index": "PASS", "fresh_extraction_verification": validation,
        "new_market_outcome_reads": 0, "new_model_fits": 0, "new_accounts": 0,
        "external_review_completed": False, "whole_goal_achieved": False,
    }
    write_json(report / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="封存私募意向资料扩样；离线复算不调用网络和模型")
    parser.add_argument("--verify-root", type=Path)
    args = parser.parse_args()
    if args.verify_root is not None:
        print(json.dumps(verify(args.verify_root), ensure_ascii=False, indent=2))
    else:
        package()
