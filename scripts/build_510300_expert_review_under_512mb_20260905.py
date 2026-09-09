"""从既有审阅包生成严格小于 512,000,000 字节的单个 ZIP。"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import struct
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "deliverables"
PARENT = "510300_ALL_WORK_EXPERT_REVIEW_20260905"
NAME = "510300_EXPERT_REVIEW_UNDER_512MB_20260905"
LIMIT = 512_000_000
CONTENT_BUDGET = 450_000_000
TZ = timezone(timedelta(hours=8))
FORMAL = {"config", "docs", "scripts", "tests", "research", "reports", "src", "backtest", "market_data", "logs", "output", "artifacts", "tools"}
STRUCTURED = {".md", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".sha256"}
DIRECT = ("510300", "csi300", "000300", "h00300", "priority_forward", "t_only", "nbs_", "dsv5", "v3_forward")


def rank(path: str, info: zipfile.ZipInfo) -> tuple[int, str]:
    parts = path.split("/")
    lower = path.lower()
    suffix = Path(path).suffix.lower()
    filename = parts[-1].lower()
    if len(parts) == 1 or parts[0] in FORMAL:
        return 0, "全部代码、协议、测试、正式报告、结果或项目入口"
    if parts[0] == "deliverables":
        return 0, "历史交付说明及回执"
    if parts[0] == "paper":
        if suffix in STRUCTURED or info.file_size <= 2 * 1024 * 1024:
            return 0, "前向状态、账本和运行结果"
        return 90, "大型延展输入面板，保留状态和结果即可审阅"
    if parts[0] == "outputs":
        return 90, "其他研究的大型原始数据导出"
    if parts[0] != "data" or len(parts) < 3:
        return 90, "辅助原始材料"
    family = parts[1]
    if family in {"reference", "models", "labels", "backtests", "forward", "governance", "validation"}:
        return 1, "基准、模型、标签、前向和验证资料"
    if family in {"curated", "research", "processed", "frozen"}:
        if any(t in filename for t in ("minute_features", "random_joint_draws", "extended_panel", "warmup_panel")):
            return 4, "可按代码重建的大型输入或模拟面板"
        return 1, "整理数据、研究输出和处理结果"
    if family in {"features", "audit", "checkpoints", "external_validation", "staging"}:
        if suffix in STRUCTURED and info.file_size <= 2 * 1024 * 1024:
            return 2, "结构化来源、核对、状态和中间证据"
        if suffix == ".parquet" and info.file_size <= 2 * 1024 * 1024:
            return 3, "小型特征或证据表"
        return 90, "大型特征、抓取检查点、原文截图或中间数据"
    if family == "raw":
        if path in {
            "data/raw/constituents/000300_constituent_daily.parquet",
            "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/official_financial_facts_v1_21_document_by_document.parquet",
        }:
            return 1, "主线整理数据及最新版官方财务事实"
        if parts[2] == "market" and not any(part.startswith(".") for part in parts[3:-1]):
            if info.file_size <= 2 * 1024 * 1024 or filename.startswith("510300_1m"):
                return 2, "510300与沪深300基础市场输入"
        if suffix in STRUCTURED and info.file_size <= 2 * 1024 * 1024 and any(word in filename for word in ("manifest", "receipt", "metadata", "status", "summary", "coverage", "ledger", "catalog")):
            return 2, "原始来源索引、覆盖、采集回执与状态"
        if any(t in lower for t in ("nbs_fixed_5min", "dr007_tushare")) and info.file_size <= 2 * 1024 * 1024:
            return 3, "最新主线的小型直接来源证据"
        return 90, "批量原始行情、公告原文、逐请求响应或重复缓存"
    return 90, "大型辅助数据，不影响代码和结论审阅"


def csv_data(rows: list[dict], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def copy_member(source: zipfile.ZipFile, target: zipfile.ZipFile, old: zipfile.ZipInfo, new_name: str) -> None:
    """直接复制经过上一轮 CRC 验证的压缩数据，避免解压再压缩。"""
    source.fp.seek(old.header_offset)
    header = source.fp.read(30)
    if len(header) != 30 or header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"原 ZIP 本地文件头无效：{old.filename}")
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    source.fp.seek(name_size + extra_size, 1)
    info = copy.copy(old)
    info.filename = new_name
    info.orig_filename = new_name
    info.flag_bits &= ~8
    target.fp.seek(target.start_dir)
    info.header_offset = target.fp.tell()
    target._writecheck(info)
    target._didModify = True
    target.fp.write(info.FileHeader(info.file_size > zipfile.ZIP64_LIMIT or info.compress_size > zipfile.ZIP64_LIMIT))
    remaining = old.compress_size
    while remaining:
        block = source.fp.read(min(remaining, 8 * 1024 * 1024))
        if not block:
            raise RuntimeError(f"原 ZIP 压缩内容提前结束：{old.filename}")
        target.fp.write(block)
        remaining -= len(block)
    target.filelist.append(info)
    target.NameToInfo[info.filename] = info
    target.start_dir = target.fp.tell()


def main() -> None:
    started = time.monotonic()
    destination = OUT / f"{NAME}.zip"
    partial = OUT / f"{NAME}.building.zip"
    if destination.exists() or partial.exists():
        raise FileExistsError("同名交付物已存在，不覆盖旧版本。")
    parent_receipt = json.loads((OUT / f"{PARENT}_BUILD_RECEIPT.json").read_text(encoding="utf-8"))
    selected, excluded = [], []
    with zipfile.ZipFile(OUT / f"{PARENT}.zip") as source:
        entries = []
        prefix = f"{PARENT}/project/"
        for info in source.infolist():
            if info.filename.startswith(prefix):
                relative = info.filename[len(prefix):]
                priority, reason = rank(relative, info)
                estimated = info.compress_size + 2 * len(f"{NAME}/project/{relative}".encode("utf-8")) + 200
                entries.append((priority, relative, info, reason, estimated))
        used = 0
        for priority, relative, info, reason, estimated in sorted(entries, key=lambda r: (r[0], r[1])):
            row = {"path": relative, "bytes": info.file_size, "compressed_bytes": info.compress_size, "reason": reason}
            if priority == 0 or (priority < 90 and used + estimated <= CONTENT_BUDGET):
                selected.append((info, row))
                used += estimated
            else:
                row["reason"] = reason if priority >= 90 else "512 MB 容量约束：优先保留正式研究和关键证据；本文件保留于原大包"
                excluded.append(row)
        if used >= CONTENT_BUDGET:
            raise RuntimeError("正式材料本身超出预留容量，需重新明确压缩范围。")
        selected.sort(key=lambda pair: pair[0].header_offset)
        paths = {r["path"] for _, r in selected}
        protected_count = sum(1 for priority, _, _, _, _ in entries if priority == 0)
        missing_protected = [relative for priority, relative, _, _, _ in entries if priority == 0 and relative not in paths]
        if missing_protected:
            raise RuntimeError(f"正式材料缺失：{missing_protected[:5]}")
        now = datetime.now(TZ).isoformat()
        print(json.dumps({"阶段": "512 MB 版材料已确定", "项目文件": len(selected), "正式材料全部保留": protected_count, "预计材料字节": used, "容量上限": LIMIT}, ensure_ascii=False), flush=True)
        rows = [r for _, r in selected]
        coverage = defaultdict(lambda: {"files": 0, "bytes": 0, "compressed_bytes": 0})
        for row in rows:
            parts = row["path"].split("/")
            group = "/".join(parts[:2] if parts[0] == "data" else parts[:1])
            for key in ("bytes", "compressed_bytes"):
                coverage[group][key] += row[key]
            coverage[group]["files"] += 1
        report_rows = [r for r in rows if r["path"].startswith(("docs/", "reports/")) and Path(r["path"]).suffix.lower() in {".md", ".json", ".pdf"}]
        recent_rows = [r for info, r in selected if info.date_time >= (2026, 9, 1, 0, 0, 0)]
        metadata = {"generated_at": now, "size_limit_bytes": LIMIT, "single_zip": True, "project_files": len(rows), "project_bytes": sum(r["bytes"] for r in rows), "all_formal_materials_preserved": True, "formal_material_count": protected_count, "excluded_from_parent": len(excluded), "parent_package": parent_receipt["zip_path"], "parent_sha256": parent_receipt["sha256"], "snapshot_date": "2026-09-05", "SECURITY_AUDIT": False, "PRIVACY_SCAN": False, "SECRET_SCAN": False, "MALWARE_SCAN": False, "REDACTION": False, "research_rerun_performed": False}
        readme = f"""# 510300 专家审阅包：单文件小于 512 MB

本包由截至 2026-09-05 的完整研究材料大包精简生成。采用更严格的十进制上限：ZIP 文件必须小于 **512,000,000 字节**，便于上传到限制为 512 MB 的平台。

保留 {len(rows):,} 个项目文件，其中 {protected_count:,} 个正式材料文件全部保留，包括代码、配置、协议、测试、报告、研究结果、最新状态和运行记录。最新 NBS G2 最终结果、DSV5 归档和 V4 权威状态均保留。另保留关键整理数据、来源回执、前向账本、小型特征及证据表；没有抽样改写原文件内容。

## 专家阅读顺序

1. [审阅任务书](01_EXPERT_REVIEW_PROMPT.md)。
2. [最新状态](02_CURRENT_STATUS.md) 与 [V4 权威入口](project/config/510300_research_authority_v4.json)。
3. [报告和文档索引](04_RESEARCH_REPORT_INDEX.csv)、[近期材料索引](05_RECENT_UPDATES.csv)。
4. 根据报告定位 `project/` 下的协议、代码、结果和数据，原相对路径保持不变。
5. `historical_review_context/` 保留历史问题、用户想法和既有专家反馈，仅作为历史上下文。

## 精简范围

精简批量原始行情、公告原文、逐请求响应、重复缓存、大型输入面板及部分可重建中间数据；保留研究方向、结论和失败证据。受容量约束，本包不承诺包含完整复现所有研究所需的全部原始数据。

- [本包文件清单](03_INCLUDED_FILES.csv)：实际装入的项目文件。
- [本次精简清单](07_EXCLUDED_FROM_LARGE_PACKAGE.csv)：原 3.96 GB 大包中存在、本包未装入的文件及原因。
- [前次原始资料排除清单](11_PARENT_PACKAGE_EXCLUDED_FILES.csv)：上次构建大包时已未装入的原始资料。
- [覆盖统计](06_COVERAGE.csv)：各资料目录的纳入数量和体量。

只交付这个 ZIP 即可开始专家审阅。需重跑某项研究时，按排除清单补充对应输入。此包的当前状态来自原大包快照，未启动新的研究或更新任何策略结论。

按用户要求，未做任何安全审查、内容扫描或脱敏；只检查压缩包结构、CRC、文件大小上限和整包哈希。
"""
        fields = ["path", "bytes", "compressed_bytes", "reason"]
        nav = {
            "00_README_FIRST.md": readme.encode("utf-8"),
            "01_EXPERT_REVIEW_PROMPT.md": source.read(f"{PARENT}/01_EXPERT_REVIEW_PROMPT.md"),
            "02_CURRENT_STATUS.md": source.read(f"{PARENT}/02_CURRENT_STATUS.md"),
            "04_RESEARCH_REPORT_INDEX.csv": csv_data(report_rows, fields),
            "05_RECENT_UPDATES.csv": csv_data(recent_rows, fields),
            "06_COVERAGE.csv": csv_data([{"directory": k, **v} for k, v in sorted(coverage.items())], ["directory", "files", "bytes", "compressed_bytes"]),
            "07_EXCLUDED_FROM_LARGE_PACKAGE.csv": csv_data(excluded, fields),
            "08_PACKAGE_METADATA.json": json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            "09_RECENT_GIT_COMMITS.txt": source.read(f"{PARENT}/09_RECENT_GIT_COMMITS.txt"),
            "10_PARENT_PACKAGE_RECORD.json": json.dumps(parent_receipt, ensure_ascii=False, indent=2).encode("utf-8"),
        }
        with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as target:
            for name, content in nav.items():
                target.writestr(f"{NAME}/{name}", content)
            previous_excluded = source.getinfo(f"{PARENT}/07_EXCLUDED_FILES.csv")
            copy_member(source, target, previous_excluded, f"{NAME}/11_PARENT_PACKAGE_EXCLUDED_FILES.csv")
            for info in source.infolist():
                if info.filename.startswith(f"{PARENT}/historical_review_context/"):
                    copy_member(source, target, info, NAME + info.filename[len(PARENT):])
            last = time.monotonic()
            for number, (info, row) in enumerate(selected, 1):
                copy_member(source, target, info, f"{NAME}/project/{row['path']}")
                if time.monotonic() - last > 15:
                    print(f"快速重打包：{number:,}/{len(selected):,} 个文件", flush=True)
                    last = time.monotonic()
            builder_relative = "scripts/" + Path(__file__).name
            target.write(Path(__file__), f"{NAME}/project/{builder_relative}")
            builder_info = target.getinfo(f"{NAME}/project/{builder_relative}")
            rows.append({"path": builder_relative, "bytes": builder_info.file_size, "compressed_bytes": builder_info.compress_size, "reason": "本次 512 MB 构包脚本"})
            target.writestr(f"{NAME}/03_INCLUDED_FILES.csv", csv_data(rows, fields))
        size = partial.stat().st_size
        if size >= LIMIT:
            raise RuntimeError(f"压缩包超过 512 MB：{size} 字节")
        print(f"文件已生成：{size:,} 字节，低于 512,000,000 字节；检查 ZIP 可读性。", flush=True)
        with zipfile.ZipFile(partial) as checked:
            members = checked.infolist()
            duplicate = [name for name, count in Counter(i.filename for i in members).items() if count > 1]
            if duplicate:
                raise RuntimeError(f"重复成员：{duplicate[:5]}")
            indexed = list(csv.DictReader(io.StringIO(checked.read(f"{NAME}/03_INCLUDED_FILES.csv").decode("utf-8-sig"))))
            for row in indexed:
                info = checked.getinfo(f"{NAME}/project/{row['path']}")
                if info.file_size != int(row["bytes"]):
                    raise RuntimeError(f"文件清单大小不一致：{row['path']}")
            actual_project_members = {i.filename for i in members if i.filename.startswith(f"{NAME}/project/")}
            if actual_project_members != {f"{NAME}/project/{row['path']}" for row in indexed}:
                raise RuntimeError("项目文件清单存在遗漏或多余项")
            bad = checked.testzip()
            if bad:
                raise RuntimeError(f"ZIP CRC 检查失败：{bad}")
        partial.rename(destination)
        digest = hashlib.sha256()
        with destination.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        receipt = {**metadata, "status": "PASS_SINGLE_ZIP_UNDER_512000000_BYTES_CRC_AND_INDEX", "zip_path": str(destination), "bytes": size, "MB_decimal": round(size / 1_000_000, 2), "MiB_binary": round(size / 2**20, 2), "remaining_bytes_to_limit": LIMIT - size, "project_files_including_new_builder": len(rows), "zip_members": len(members), "report_index_entries": len(report_rows), "sha256": digest.hexdigest(), "elapsed_seconds": round(time.monotonic() - started, 2)}
        (OUT / f"{NAME}_BUILD_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUT / f"{NAME}.sha256").write_text(f"{digest.hexdigest()}  {destination.name}\n", encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
