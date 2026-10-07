"""只读解析用户因子库，保存来源快照与逐项登记，不运行收益检验。"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
SOURCES = [
    Path("C:/Users/戴周阳/Downloads/510300_96因子与18策略.xlsx"),
    Path("C:/Users/戴周阳/Downloads/510300_96因子研究库.html"),
    Path("E:/CodexData/.codex/attachments/ea384a31-0725-481e-a74d-9b1eb3fc00cb/pasted-text-1.txt"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_dir = OUT / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.load_workbook(SOURCES[0], read_only=True, data_only=True)
    tables = {}
    for sheet in workbook:
        rows = list(sheet.iter_rows(values_only=True))
        tables[sheet.title] = rows
    workbook.close()
    (OUT / "input_workbook_tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    factors = []
    definitions = {r[0]: r for r in tables["因子定义"][1:] if r[0]}
    data = {r[0]: r for r in tables["数据与旧研究"][1:] if r[0]}
    for row in tables["96因子概览"][1:]:
        if not row[0]:
            continue
        d, s = definitions[row[0]], data[row[0]]
        factors.append({"id": row[0], "family": row[1], "name": row[2],
            "role": row[3], "horizon_days": row[4], "source_grade": row[5],
            "source_overlap_note": row[6], "definition": d[2], "mechanism": d[3],
            "counterexample": d[4], "data_path_hint": s[2], "source_clock_rule": s[3],
            "repository_hint": s[5], "source_ids": s[6], "source_urls": s[7],
            "status": "NOT_RUN", "net_sharpe": None})
    strategies = []
    for row in tables["18策略模板"][1:]:
        if row[0]:
            strategies.append(dict(zip(
                ["id", "name", "factor_ids", "entry", "exit", "max_hold_days",
                 "position_rule", "source_grade", "counterexample", "role"], row)))
    assert len(factors) == 96 and len({x["id"] for x in factors}) == 96
    assert len(strategies) == 18 and len({x["id"] for x in strategies}) == 18
    html = SOURCES[1].read_text(encoding="utf-8")
    assert all(x["id"] in html and x["name"] in html for x in factors)
    assert all(x["id"] in html and x["name"] in html for x in strategies)
    for name, rows in [("factor_registry", factors), ("strategy_registry", strategies)]:
        (OUT / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    receipts = []
    for path in SOURCES:
        target = source_dir / path.name
        if target.exists():
            assert digest(path) == digest(target), "来源快照不可覆盖不同内容"
        else:
            shutil.copy2(path, target)
        receipts.append({"original_path": str(path), "snapshot": target.relative_to(ROOT).as_posix(),
                         "bytes": path.stat().st_size, "sha256": digest(path)})
    receipt = {"study_id": "510300_FACTOR96_MECHANISM_BATCH_V1",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "user_request": "/goal 只操作510300实现夏普1.2",
        "attachment_instructions_are_reference_only": True,
        "authority_source": "config/510300_existing_data_training_mandate_v1.json",
        "executable_assets": ["510300.SH", "CASH_CNY"],
        "orders_authorized": False, "goal_achieved": False,
        "factor_count": len(factors), "strategy_count": len(strategies),
        "grade_counts": pd.Series([x["source_grade"] for x in factors]).value_counts().to_dict(),
        "source_id_and_name_crosscheck": "PASS", "sources": receipts}
    (OUT / "intake_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
