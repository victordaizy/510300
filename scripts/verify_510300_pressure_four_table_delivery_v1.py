"""逐行核对保存的CSV、XLSX与上游矩阵；不把交付核对称为金融验证。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import posixpath
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = read(ROOT / "config/510300_pressure_four_table_delivery_v1.json")
    report, artifact = ROOT / cfg["report_directory"], ROOT / cfg["artifact_directory"]
    workbook = artifact / cfg["workbook_filename"]
    if not workbook.is_file():
        raise SystemExit("最终工作簿尚未保存，不能核对。")
    output = report / "verification.json"
    if output.exists():
        raise SystemExit("四表核对回执已存在，拒绝覆盖。")
    registration = read(report / "registration.json")
    checks = []
    for item in registration["inputs"]:
        p = Path(item["path"])
        checks.append({"check": "上游注册输入未变", "path": str(p), "passed": sha(p) == item["sha256"]})
    matrix = pq.read_table(ROOT / cfg["inputs"][2]).to_pylist()
    manifest = read(ROOT / cfg["inputs"][7])
    source_map = {(x["date"], x["stream"]): x for x in manifest["files"]}
    with (report / "01_同步行情表.csv").open(encoding="utf-8-sig", newline="") as handle:
        sync = list(csv.DictReader(handle))
    if len(sync) != len(matrix):
        raise SystemExit("同步行情CSV没有保留全部网格。")
    numeric_map = {"源名义时间原值": "source_time", "子集源行号（0起）": "source_quote_row", "最新价原值": "source_price_raw",
                   "IOPV原值（未认证）": "source_iopv_raw", "名义中间价（元）": "source_mid_cny", "名义价差（bp）": "source_spread_bps",
                   "10bp可见买盘（元）": "source_band_bid_depth_cny", "累计量原值": "source_cum_volume",
                   "过去本地配对日数": "prior_source_m2_pair_days", "过去严格配对日数": "prior_strict_m2_pair_days"}
    numeric_checks = 0
    for index, (saved, original) in enumerate(zip(sync, matrix, strict=True)):
        day = datetime.strptime(original["date"], "%Y%m%d").date().isoformat()
        if saved["日期"] != day or saved["缺失原因"] != original["strict_reasons"] or saved["时点资格"] != "NO_VIEW":
            raise SystemExit(f"第{index}行日期/资格/缺失原因不一致。")
        source = source_map[(original["date"], "行情")]
        if saved["来源ID"] != f"行情_{original['date']}" or source["sha256"] != original["source_quote_sha256"]:
            raise SystemExit(f"第{index}行源身份不一致。")
        for saved_key, original_key in numeric_map.items():
            wanted = original[original_key]
            if wanted is None or isinstance(wanted, float) and not math.isfinite(wanted):
                equal = saved[saved_key] == ""
            else:
                equal = math.isclose(float(saved[saved_key]), wanted, rel_tol=1e-12, abs_tol=1e-12)
            if not equal:
                raise SystemExit(f"第{index}行{saved_key}与上游矩阵不一致。")
            numeric_checks += 1
        for key in ["价格经济时间", "参考经济时间", "历史接收时间", "参考价值下界（元）", "参考价值上界（元）", "对应PCF已验证"]:
            if saved[key] != "":
                raise SystemExit(f"第{index}行将未识别的{key}填成已知值。")
    checks.append({"check": "CSV全部网格、原始身份、缺失值和数值与矩阵一致", "passed": True,
                   "rows": len(sync), "numeric_comparisons": numeric_checks})
    prepared = read(report / "prepared_summary.json")
    csv_names = {"同步行情": "01_同步行情表.csv", "事件": "02_事件表.csv", "订单": "03_订单表.csv", "结果": "04_结果表.csv",
                 "日期覆盖": "05_日期覆盖.csv", "数据来源": "06_原值来源.csv", "字段说明": "07_字段说明.csv"}
    tables = {}
    for name, filename in csv_names.items():
        with (report / filename).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            headers = next(reader)
            rows = list(reader)
        tables[name] = (headers, rows)
        checks.append({"check": "CSV表行数", "sheet": name, "rows": len(rows),
                       "passed": len(rows) == prepared["table_row_counts"][name]})
    with zipfile.ZipFile(workbook) as archive:
        if archive.testzip() is not None:
            raise SystemExit("工作簿ZIP校验失败。")
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(x.itertext()) for x in shared_root.findall(f"{M}si")]
        def value(cell: ET.Element) -> object:
            kind = cell.get("t")
            content = cell.find(f"{M}v")
            if kind == "inlineStr":
                return "".join(t.text or "" for t in cell.iter(f"{M}t"))
            text = content.text if content is not None else None
            if text is None:
                return None
            if kind == "s":
                return shared[int(text)]
            if kind == "b":
                return bool(int(text))
            if kind in {"str", "d", "e"}:
                return text
            return float(text)
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationships = {x.get("Id"): x.get("Target") for x in relationships}
        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_paths = {}
        for sheet in workbook_root.find(f"{M}sheets"):
            target = relationships[sheet.get(f"{R}id")]
            sheet_paths[sheet.get("name")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
        checks.append({"check": "九个工作表齐全", "passed": set(sheet_paths) == {"研究状态", *prepared["table_row_counts"]}})
        summary_values = {}
        formula_count = 0
        with archive.open(sheet_paths["研究状态"]) as handle:
            for _, element in ET.iterparse(handle, events=["end"]):
                if element.tag != f"{M}row":
                    continue
                for cell in element.findall(f"{M}c"):
                    summary_values[cell.get("r")] = value(cell)
                    formula_count += cell.find(f"{M}f") is not None
                element.clear()
        expected_summary = {"B6": 181, "B7": 543, "B8": 79288590, "B9": 47784, "B10": 0}
        checks.append({"check": "保存XLSX汇总公式缓存与来源一致", "passed": all(summary_values.get(k) == v for k, v in expected_summary.items()) and formula_count == 5,
                       "formula_count": formula_count})
        checks.append({"check": "保存XLSX正式事件/期望/夏普保持空白", "passed": all(summary_values.get(k) is None for k in ["B12", "B15", "B16"])})
        comparisons = 0
        for name, (headers, rows) in tables.items():
            actual_rows = 0
            with archive.open(sheet_paths[name]) as handle:
                for _, element in ET.iterparse(handle, events=["end"]):
                    if element.tag != f"{M}row":
                        continue
                    row_number = int(element.get("r"))
                    cells = {re_column(c.get("r")): value(c) for c in element.findall(f"{M}c")}
                    if row_number > 5 and any(v is not None and v != "" for v in cells.values()):
                        if actual_rows >= len(rows):
                            raise SystemExit(f"{name}出现CSV之外的非空行。")
                        wanted = rows[actual_rows]
                        for index, text in enumerate(wanted):
                            found = cells.get(column_name(index + 1))
                            if text == "":
                                okay = found is None or found == ""
                            elif text in {"True", "False"}:
                                okay = isinstance(found, bool) and found == (text == "True")
                            elif headers[index] == "日期":
                                serial = (datetime.fromisoformat(text) - datetime(1899, 12, 30)).days
                                okay = found == serial or found == text or found == text + "T00:00:00.000Z"
                            elif isinstance(found, (int, float)):
                                okay = math.isclose(found, float(text), rel_tol=1e-12, abs_tol=1e-12)
                            else:
                                okay = found == text
                            if not okay:
                                raise SystemExit(f"{name}!{column_name(index+1)}{row_number}与CSV不一致：{found!r} / {text!r}")
                            comparisons += 1
                        actual_rows += 1
                    element.clear()
            checks.append({"check": "保存XLSX全部数据单元格与CSV一致", "sheet": name, "rows": actual_rows, "passed": actual_rows == len(rows)})
        checks.append({"check": "XLSX逐单元格核对范围", "passed": True, "cell_comparisons": comparisons})
    preservation = read(report / "scope_preservation_registration.json")
    old = preservation["original_workbook"]
    checks.append({"check": "原早期工作簿保留", "passed": sha(Path(old["path"])) == old["sha256"]})
    scans = (artifact / "公式错误扫描.ndjson").read_text(encoding="utf-8")
    checks.append({"check": "作者工具公式错误扫描为空", "passed": "Cell search matched 0 entries" in scans})
    checks.append({"check": "九页预览已生成", "passed": len(list((artifact / "previews").glob("*.png"))) == 9})
    total_bytes = sum(p.stat().st_size for base in [report, artifact] for p in base.rglob("*") if p.is_file())
    checks.append({"check": "交付资料小于64MiB", "passed": total_bytes < cfg["scope"]["maximum_new_output_bytes"], "bytes_at_verification": total_bytes})
    result = {"verified_at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_FOUR_TABLE_DELIVERY" if all(x["passed"] for x in checks) else "FAIL_REQUIRES_REVIEW",
              "checks": checks, "check_count": len(checks), "passed_count": sum(x["passed"] for x in checks),
              "xlsx_cell_comparisons": comparisons, "source_matrix_numeric_comparisons": numeric_checks,
              "workbook": str(workbook), "workbook_bytes": workbook.stat().st_size, "workbook_sha256": sha(workbook),
              "source_grid_rows": len(matrix), "formal_event_count": None, "strategy_returns": "NOT_COMPUTED",
              "goal_achieved": False, "financial_validation": "NOT_RUN_SOURCE_GATE_FAILED",
              "scope": "保存四表与上游矩阵一致；不重验市场源真值、接收时钟、估值或执行，不是金融验证。"}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, ensure_ascii=False, indent=2))
    if not all(x["passed"] for x in checks):
        raise SystemExit("保存文件核对存在失败项，不能交付。")


def re_column(reference: str) -> str:
    return "".join(c for c in reference if c.isalpha())


def column_name(number: int) -> str:
    text = ""
    while number:
        number, remainder = divmod(number - 1, 26)
        text = chr(65 + remainder) + text
    return text


if __name__ == "__main__":
    main()
