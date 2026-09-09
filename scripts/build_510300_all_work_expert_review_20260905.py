"""快速构建 510300 全研究专家审阅包，只检查交付结构。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import time
import zipfile
import zlib
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
NAME = "510300_ALL_WORK_EXPERT_REVIEW_20260905"
OUT = ROOT / "deliverables"
TZ = timezone(timedelta(hours=8))
FORMAL = {"config", "docs", "research", "scripts", "tests", "src", "backtest", "market_data", "reports", "paper", "logs", "output", "outputs", "artifacts", "tools"}
SKIP_ROOTS = {".git", ".venv", ".agents", ".codex", ".codex-remote-attachments", "__pycache__", ".pytest_cache", "tmp", "review_packages"}
DIRECT = ("510300", "csi300", "000300", "h00300", "b1_dsv5", "nbs_", "priority_forward", "t_only", "v3_forward")
SHARED_RAW = {"constituents", "cross_etf_forced_flow_v1", "flow", "forward", "fund", "fundamentals", "futures", "global_liquidity_shock_v0", "industry", "macro", "market", "market_margin_leverage_v0", "market_margin_validation_v1", "market_margin_validation_v2", "official", "primary_market", "r6", "reference", "regime_transition_router_v1", "remediation", "return_tail", "sina", "t_only_forward_v1", "tushare", "us_china_overnight_v1", "valuation"}
ARCHIVE_SUFFIXES = {".zip", ".7z", ".rar"}
CRITICAL = [
    "RESEARCH_STATUS.md",
    "config/510300_research_authority_v4.json",
    "docs/510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md",
    "docs/510300_NBS_FIXED_5MIN_USAGE_AUTHORIZATION_20260905.md",
    "reports/research/510300_nbs_fixed_5min_usage_v2_1/status.json",
    "reports/research/510300_nbs_fixed_5min_usage_v2_1/return_read_claim.json",
    "reports/research/510300_price_path_dsv5_risk_budget_policy_v1/ARCHIVE_DECISION_20260905.json",
    "reports/research/510300_stress_transmission_hazard_v2_g1b_historical_final_status_v1.json",
    "docs/DECISIONS.md",
    "scripts/build_510300_all_work_expert_review_20260905.py",
]


def log(message: str) -> None:
    print(message, flush=True)


def selection(relative: str, size: int) -> tuple[bool, str]:
    parts = relative.split("/")
    lower = relative.lower()
    filename = parts[-1].lower()
    suffix = Path(filename).suffix
    if NAME.lower() in lower:
        return (relative == "scripts/build_510300_all_work_expert_review_20260905.py", "本次构包脚本或本次交付物，避免自包含")
    if filename == ".env" or (filename.startswith(".env.") and filename != ".env.example"):
        return False, "本机环境输入，按文件名不纳入"
    if suffix in ARCHIVE_SUFFIXES or suffix in {".pyc", ".pyo"}:
        return False, "旧压缩包或解释器缓存"
    if any(p.startswith(("pytest", ".pytest", "__pycache__")) for p in parts[:-1]):
        return False, "测试运行临时目录"
    if parts[0] in SKIP_ROOTS or parts[0].startswith("data.c-to-e-residual"):
        return False, "本机运行目录、临时文件或迁移残留"
    if parts[0] in FORMAL:
        return True, "完整研究正文、代码、协议、结果及运行证据"
    if parts[0] == "deliverables":
        return len(parts) == 2, "历史交付回执和说明；旧展开副本不重复装入"
    if len(parts) == 1:
        return True, "项目根文件"
    if parts[0] != "data":
        return False, "非研究资料目录"
    if len(parts) > 1 and parts[1] == "tmp":
        return False, "数据临时缓存"
    if any(token in lower for token in DIRECT):
        return True, "510300及沪深300直接研究数据，完整保留"
    if len(parts) > 2 and parts[1] == "raw":
        if parts[2] in SHARED_RAW:
            return True, "510300共用行情、成分、宏观、衍生品及来源数据"
        if parts[2] == "cninfo" and suffix in {".json", ".csv", ".tsv", ".jsonl"} and size <= 16 * 1024 * 1024 and any(s in filename for s in ("manifest", "metadata", "ledger", "receipt", "status", "summary", "index", "catalog", "coverage", "facts")):
            return True, "公告来源索引、事实或采集回执"
        return False, "快速模式：全市场公告原文或其他研究原始批量数据仅列清单"
    if len(parts) > 1 and parts[1] == "staging":
        if size <= 1024 * 1024 and suffix in {".json", ".csv", ".tsv", ".md", ".txt", ".yaml", ".yml"}:
            return True, "其他研究的小型中间证据，保留上下文"
        return False, "快速模式：其他研究的大型中间面板仅列清单"
    return True, "完整整理数据、特征、账本、结果或来源核对证据"


def scan() -> dict:
    files, skipped, errors = [], [], []
    stack = [str(ROOT)]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = os.path.relpath(entry.path, ROOT).replace("\\", "/")
                    if entry.is_dir():
                        parts = relative.split("/")
                        excluded = (len(parts) == 1 and (entry.name in SKIP_ROOTS or entry.name.startswith((".pytest", "data.c-to-e-residual")))) or entry.name in {"__pycache__", ".git", ".venv"} or relative.startswith(("data/tmp", "deliverables/"))
                        if excluded or entry.is_symlink():
                            skipped.append(relative)
                        else:
                            stack.append(entry.path)
                    else:
                        st = entry.stat()
                        files.append([relative, st.st_size, st.st_mtime_ns])
        except OSError as exc:
            errors.append({"path": os.path.relpath(directory, ROOT), "error": str(exc)})
    return {"root": str(ROOT), "created_at": time.time(), "files": files, "skipped_directories": skipped, "errors": errors}


def csv_bytes(rows: list[dict], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def compress_source(row: dict) -> tuple[zipfile.ZipInfo, bytes, dict | None]:
    """并行读取单个源文件并生成标准 ZIP 原始 Deflate 数据。"""
    path = ROOT / row["path"]
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        content = handle.read()
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(content) != before.st_size:
        raise RuntimeError(f"文件在读取中发生变化，请重建快照：{row['path']}")
    stamp = datetime.fromtimestamp(before.st_mtime, TZ)
    zip_date = (max(1980, min(2107, stamp.year)), stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second)
    info = zipfile.ZipInfo(f"{NAME}/project/{row['path']}", zip_date)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 0
    info.external_attr = 0o600 << 16
    info.file_size = len(content)
    info.CRC = zlib.crc32(content) & 0xFFFFFFFF
    compressor = zlib.compressobj(1, zlib.DEFLATED, -15)
    compressed = compressor.compress(content) + compressor.flush()
    info.compress_size = len(compressed)
    changed = None
    if before.st_size != row["bytes"] or abs(before.st_mtime - datetime.fromisoformat(row["modified_at"]).timestamp()) > 0.001:
        changed = {"path": row["path"], "indexed_bytes": row["bytes"], "packed_bytes": len(content)}
    return info, compressed, changed


def write_compressed(archive: zipfile.ZipFile, info: zipfile.ZipInfo, compressed: bytes) -> None:
    """单线程写入已压缩数据，由标准库生成本地头和最终中央目录。"""
    archive.fp.seek(archive.start_dir)
    info.header_offset = archive.fp.tell()
    archive._writecheck(info)
    archive._didModify = True
    archive.fp.write(info.FileHeader(info.file_size > zipfile.ZIP64_LIMIT or info.compress_size > zipfile.ZIP64_LIMIT))
    archive.fp.write(compressed)
    archive.filelist.append(info)
    archive.NameToInfo[info.filename] = info
    archive.start_dir = archive.fp.tell()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    OUT.mkdir(exist_ok=True)
    inventory = json.loads(args.inventory.read_text(encoding="utf-8")) if args.inventory else scan()
    source_map = {r[0]: r for r in inventory["files"]}
    for relative in CRITICAL:
        st = (ROOT / relative).stat()
        source_map[relative] = [relative, st.st_size, st.st_mtime_ns]
    selected, excluded = [], []
    for relative, size, modified in sorted(source_map.values()):
        include, reason = selection(relative, size)
        row = {"path": relative, "bytes": size, "modified_at": datetime.fromtimestamp(modified / 1e9, TZ).isoformat(), "reason": reason}
        (selected if include else excluded).append(row)
    total = sum(r["bytes"] for r in selected)
    counts = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in selected:
        key = "/".join(row["path"].split("/")[:3] if row["path"].startswith("data/") else row["path"].split("/")[:1])
        counts[key]["files"] += 1
        counts[key]["bytes"] += row["bytes"]
    log(json.dumps({"阶段": "材料已选定", "文件数": len(selected), "原始GiB": round(total / 2**30, 3), "排除文件数": len(excluded), "最大目录": sorted(counts.items(), key=lambda x: x[1]["bytes"], reverse=True)[:18]}, ensure_ascii=False))
    if args.inventory_only:
        return
    zip_path = OUT / f"{NAME}.zip"
    partial = OUT / f"{NAME}.building.zip"
    if zip_path.exists() or partial.exists():
        raise FileExistsError("同名交付物已存在，请保留旧包并使用新的版本名。")
    now = datetime.now(TZ).isoformat()
    authority = json.loads((ROOT / CRITICAL[1]).read_text(encoding="utf-8"))
    included_paths = {r["path"] for r in selected}
    missing = sorted(set(CRITICAL) - included_paths)
    if missing:
        raise RuntimeError(f"缺少当前研究入口：{missing}")
    summary = {
        "generated_at": now, "source_root": str(ROOT), "data_root": str((ROOT / "data").resolve()),
        "source_file_count": len(selected), "source_bytes": total, "excluded_file_count": len(excluded),
        "excluded_bytes": sum(r["bytes"] for r in excluded), "compression": "ZIP_DEFLATED_LEVEL_1",
        "snapshot_semantics": "FILESYSTEM_SNAPSHOT_WITH_PER_FILE_CHANGE_RECORDS",
        "current_authority": authority, "skipped_directories": inventory.get("skipped_directories", []),
        "inventory_errors": inventory.get("errors", []),
        "SECURITY_AUDIT": False, "PRIVACY_SCAN": False, "SECRET_SCAN": False, "MALWARE_SCAN": False,
        "REDACTION": False, "FRESH_EXTRACTION_REPLAY": False, "research_rerun_performed": False,
    }
    readme = f"""# 510300 全研究专家审阅包

制作时间：{now}（北京时间）。共纳入 {len(selected):,} 个项目文件，原始 {total / 2**30:.2f} GiB。

本包覆盖当前项目中 510300 历史研究、严格前向工作、源代码、测试、冻结协议、报告、结果、运行回执及相关数据，并保留共用研究的上下文。原项目目录在 `project/` 下保持相对路径；没有重新运行研究。

## 建议阅读顺序

1. [专家审阅任务书](01_EXPERT_REVIEW_PROMPT.md)。
2. [最新状态与版本解释](02_CURRENT_STATUS.md)。
3. [当前 V4 权威状态](project/config/510300_research_authority_v4.json) 与 [NBS 最新完整报告](project/docs/510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md)。
4. [研究报告索引](04_RESEARCH_REPORT_INDEX.csv)，按路径进入报告及对应代码、数据。
5. [近期更新索引](05_RECENT_UPDATES.csv)，查看 9 月 1 日以来的增量。
6. [资料覆盖统计](06_COVERAGE.csv)、[包内文件清单](03_INCLUDED_FILES.csv) 与 [未装入文件清单](07_EXCLUDED_FILES.csv)。
7. `historical_review_context/` 保存旧审阅包独有的阅读说明、原始问题和既有专家反馈。它们属于历史上下文，不能覆盖最新权威文件。

## 完整性口径

完整收录正式代码、配置、协议、文档、报告、结果和运行证据；直接标识 510300/沪深300 的研究数据及共用行情、成分、宏观、期货、一级市场等数据保留。为快速交付，其他研究的大型中间面板、全市场公告原文和其他批量原始数据只登记文件路径、大小和原因。本包是全研究材料审阅快照，不是磁盘逐字节备份；缺少的原始文件不得解释为已在包内，完整复现个别研究仍需按排除清单补充。

不重复装入旧 ZIP、展开副本、虚拟环境、Git 内部对象、缓存、迁移残留和本机 `.env` 输入。只按目录或文件名选取，不做安全审查、内容扫描或脱敏。

## 交付检查

只检查 ZIP 可读、CRC、重复成员、清单与成员大小一致，并提供整包 SHA-256。未进行安全审查、研究重跑或外部专家审阅。本机绝对路径仍按原文保留；遇到原文中的本机路径，请从 `project/` 按对应相对路径定位。
"""
    prompt = """# 专家独立审阅任务书

请对本包中的 510300 全部研究工作进行研究质量审阅。先判断证据是否足够，再判断哪些方向值得继续。请直说哪里做得不够深入、假设过宽、测量不成立、样本不足、实现可能错误，或停止条件不合理；无需进行安全审查。

首先读取 02_CURRENT_STATUS.md 和 V4 权威状态，避免将旧快照当作当前结论。按 04_RESEARCH_REPORT_INDEX.csv 遍历研究谱系，并引用具体文件路径、字段或表格支持每一项判断。分清冻结协议、实际运行回执、历史探索、独立验证、前向观测和用户想法。

请重点回答：

1. 各研究家族的机制假设、测量和可交易含义是否相互吻合？覆盖估值/宏观、趋势/反转/MACD、日内、ETF 微观结构、资金流、期权期货、成分宽度、PIT 盈利与公告、风险预测、压力传导、DSV5、NBS 和严格前向工作。
2. 是否存在未来信息、历史版本替代、时间戳错误、幸存者偏差、样本筛选、多个试验共用样本、有效样本不足或未经计入的试验次数？
3. 数据准入和预测门、预测门和收益门之间是否被错误等同？收益、成本、基准、T+1、执行时点及压力情景是否一致？
4. 当前 NBS 五分钟用途准入及 G2 失败、DSV5 政策归档与 B1 观察、压力传导 V2 的 G1B NO_VIEW，应如何解读？哪些是真实负证据，哪些只是数据不足或程序失败？
5. 哪些方向停止，哪些只补来源/实现证据，哪些继续已有严格前向观测？最多列出三个优先方向，给出具体证据缺口、成本、可交付结果和停止线。

请交付：总体判断；按研究家族列出的继续/停止/补证据表；P0/P1/P2 问题及文件证据；重要结论的可信度与局限；未来 30/90 天工作优先次序。无法从包内证据确认的事项明确写“未证实”，并指出需要补充的文件。

所有冻结拒绝版本保留原结果，不能用改参数、窗口、方向、成本或资产池来重新包装。当前 V4 状态为 STRICT_FORWARD_ONLY；本包不启动新的历史回测。510300.SH 与 CASH_CNY 是执行范围；成分股、指数、衍生品及其他材料属于研究输入。审阅输出不包含仓位、订单或实际交易指令。
"""
    current = """# 当前状态与版本解释

快照截止 2026-09-05，当前权威入口是 `project/config/510300_research_authority_v4.json`。以下摘录来自本次实际读取的当前文件。

- 项目研究状态：`STRICT_FORWARD_ONLY`；当前已验证高夏普策略：`NONE`。
- NBS：固定八个五分钟窗口的用途合同获得明确授权，G0/G1 通过，首次真实 G2 已完成并失败。B1 严格前序 MSE 比 B0 高约 7.82%，三个时代均无改善；家族冻结，G3/G4 未运行。详见 `project/docs/510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md`。
- DSV5：旧政策归档、B1 仓位映射关闭，仅保留独立风险预测观察。NBS 的 B1 是另一个模型，不应与 DSV5 B1 混用。
- 压力传导 V2：历史 G1B 终局为 `G1B_NO_VIEW_INSUFFICIENT_EVALUABLE_ERAS_G2_NOT_RUN`；详见 `project/reports/research/510300_stress_transmission_hazard_v2_g1b_historical_final_status_v1.json`。
- V4 列出的严格前向研究范围为 PCF/IOPV、A50 盘后信息时差、宏观首次发布 vintage、B1 DSV5 预测观察、实际执行成本观察。此为研究范围，不能据此认定某项任务正在运行或已有新观测。
- 模型动作 `ABSTAIN`，模型仓位 `UNSET`，仓位影响 0；券商连接、订单及实盘未授权。

`project/RESEARCH_STATUS.md` 和 `project/docs/DECISIONS.md` 保留追加历史，旧段落与新版状态并列存在。以具体版本、日期、授权文件、冻结协议及运行回执辨别有效范围，不能把历次最有利段落拼成一个未曾运行的策略。本文仅提供导航，专家应复核原始证据。
"""
    reports = [r for r in selected if r["path"].startswith(("reports/", "docs/")) and Path(r["path"]).suffix.lower() in {".md", ".json", ".pdf"}]
    report_rows = [{"path": r["path"], "bytes": r["bytes"], "modified_at": r["modified_at"], "scope": "510300及相关主线" if any(t in r["path"].lower() for t in DIRECT) else "共用框架或其他研究上下文"} for r in reports]
    recent = [r for r in selected if r["modified_at"] >= "2026-09-01"]
    coverage = [{"directory": k, **v} for k, v in sorted(counts.items())]
    nav = {
        "00_README_FIRST.md": readme.encode("utf-8"),
        "01_EXPERT_REVIEW_PROMPT.md": prompt.encode("utf-8"),
        "02_CURRENT_STATUS.md": current.encode("utf-8"),
        "04_RESEARCH_REPORT_INDEX.csv": csv_bytes(report_rows, ["path", "bytes", "modified_at", "scope"]),
        "05_RECENT_UPDATES.csv": csv_bytes(recent, ["path", "bytes", "modified_at", "reason"]),
        "06_COVERAGE.csv": csv_bytes(coverage, ["directory", "files", "bytes"]),
        "07_EXCLUDED_FILES.csv": csv_bytes(excluded, ["path", "bytes", "modified_at", "reason"]),
        "08_PACKAGE_METADATA.json": json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
    }
    git = subprocess.run(["git", "log", "-15", "--date=iso-strict", "--format=%H %ad %s"], cwd=ROOT, capture_output=True, timeout=20)
    nav["09_RECENT_GIT_COMMITS.txt"] = git.stdout
    legacy_count = 0
    for old in sorted(OUT.glob("*.zip")):
        if not (old.name.startswith("510300_") or old.name == "QUANT_RESEARCH_ALL_WORK_GPT_PRO_REVIEW_20260828.zip") or old.name.startswith(NAME):
            continue
        with zipfile.ZipFile(old) as archive:
            for member in archive.infolist():
                parts = member.filename.strip("/").split("/")
                if len(parts) <= 2 and Path(parts[-1]).suffix.lower() in {".md", ".txt"} and member.file_size <= 2 * 1024 * 1024:
                    nav[f"historical_review_context/{old.stem}/{parts[-1]}"] = archive.read(member)
                    legacy_count += 1
    actual, changed = [], []
    last = time.monotonic()
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True, strict_timestamps=False) as archive:
        for name, content in nav.items():
            archive.writestr(f"{NAME}/{name}", content)
        processed = 0
        with ThreadPoolExecutor(max_workers=12) as executor:
            pending = deque()
            next_index, queued_bytes, number = 0, 0, 0
            while next_index < len(selected) or pending:
                while next_index < len(selected) and len(pending) < 48:
                    row = selected[next_index]
                    if pending and queued_bytes + row["bytes"] > 256 * 1024 * 1024:
                        break
                    pending.append((row, executor.submit(compress_source, row)))
                    queued_bytes += row["bytes"]
                    next_index += 1
                row, future = pending.popleft()
                info, compressed, change = future.result()
                write_compressed(archive, info, compressed)
                queued_bytes -= row["bytes"]
                actual.append({"path": row["path"], "zip_member": info.filename, "bytes": info.file_size, "modified_at": row["modified_at"], "crc32": f"{info.CRC:08x}", "reason": row["reason"]})
                if change:
                    changed.append(change)
                number += 1
                processed += info.file_size
                if time.monotonic() - last >= 15:
                    log(f"并行压缩进度：{number:,}/{len(selected):,} 个文件，已读取 {processed / 2**30:.2f}/{total / 2**30:.2f} GiB")
                    last = time.monotonic()
        archive.writestr(f"{NAME}/03_INCLUDED_FILES.csv", csv_bytes(actual, ["path", "zip_member", "bytes", "modified_at", "crc32", "reason"]))
        archive.writestr(f"{NAME}/10_SNAPSHOT_CHANGES.json", json.dumps(changed, ensure_ascii=False, indent=2).encode("utf-8"))
    log("压缩已完成，检查 ZIP 文件清单与 CRC。")
    with zipfile.ZipFile(partial) as archive:
        members = archive.infolist()
        duplicates = [name for name, count in Counter(i.filename for i in members).items() if count > 1]
        if duplicates:
            raise RuntimeError(f"存在重复成员：{duplicates[:5]}")
        for row in actual:
            if archive.getinfo(row["zip_member"]).file_size != row["bytes"]:
                raise RuntimeError(f"清单大小不一致：{row['path']}")
        for relative in CRITICAL:
            archive.getinfo(f"{NAME}/project/{relative}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC 失败：{bad}")
    partial.rename(zip_path)
    digest = hashlib.sha256()
    with zip_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    receipt = {"status": "PASS_ZIP_CRC_DUPLICATES_INDEX_SIZE_AND_CURRENT_ENTRYPOINTS", "zip_path": str(zip_path), "bytes": zip_path.stat().st_size, "sha256": digest.hexdigest(), "zip_members": len(members), "project_files": len(actual), "project_bytes": sum(r["bytes"] for r in actual), "report_index_entries": len(report_rows), "recent_files": len(recent), "historical_context_files": legacy_count, "excluded_files": len(excluded), "excluded_bytes": sum(r["bytes"] for r in excluded), "snapshot_changes": changed, "inventory_errors": inventory.get("errors", []), "elapsed_seconds": round(time.monotonic() - start, 2), "SECURITY_AUDIT": False, "PRIVACY_SCAN": False, "SECRET_SCAN": False, "MALWARE_SCAN": False, "REDACTION": False, "FRESH_EXTRACTION_REPLAY": False, "research_rerun_performed": False}
    (OUT / f"{NAME}_BUILD_RECEIPT.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"{NAME}.sha256").write_text(f"{digest.hexdigest()}  {zip_path.name}\n", encoding="utf-8")
    (OUT / f"{NAME}_专家阅读入口.md").write_text(readme + "\n\n" + current, encoding="utf-8")
    log(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
