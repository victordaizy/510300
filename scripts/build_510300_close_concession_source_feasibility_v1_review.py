"""打包本轮来源可行性结果；核对前包身份、ZIP结构及包内数据的离线复算。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
REPORT_REL = Path("reports/research/510300_close_concession_source_feasibility_v1")
REPORT = ROOT / REPORT_REL
OUT = ROOT / "deliverables"
NAME = "510300_CLOSE_CONCESSION_SOURCE_FEASIBILITY_V1_20260907_GPT_REVIEW"
PRIOR = OUT / "510300_PRICE_CONCESSION_REVIEW_20260907_GPT_REVIEW.zip"
PRIOR_SHA = "35592abaa22366b9dbf96aa7cd3c399f4a40a54e6dc6800d433d4533702ebde4"
CN = timezone(timedelta(hours=8))
SCRIPTS = [
    "scripts/probe_510300_close_concession_sources_v1.py",
    "scripts/analyze_510300_close_concession_sources_v1.py",
    "scripts/validate_510300_close_concession_sources_v1.py",
    "scripts/build_510300_close_concession_source_feasibility_v1_review.py",
    "tests/test_510300_close_concession_sources_v1.py",
]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def structural_check(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(i.is_dir() for i in archive.infolist()):
            raise ValueError("ZIP成员重复或存在未索引目录项")
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts or "\\" in n or ":" in n for n in names):
            raise ValueError("ZIP成员路径不符合本轮相对路径约定")
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"ZIP的CRC检查失败：{bad_member}")
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if len({r["path"] for r in rows}) != len(rows) or {r["path"] for r in rows} != set(names) - {"FILE_INDEX.csv"}:
            raise ValueError("根索引未覆盖全部非索引成员")
        for row in rows:
            raw = archive.read(row["path"])
            if len(raw) != int(row["bytes"]) or sha(raw) != row["sha256"]:
                raise ValueError(f"ZIP大小或哈希与索引不符：{row['path']}")
        return {"status": "PASS_ZIP_CRC_DUPLICATES_INDEX_SIZE_HASH", "member_count": len(names),
                "indexed_member_count": len(rows), "index_self_excluded": True}


def prepare_report_metadata(prior_check: dict) -> None:
    result = json.loads((REPORT / "analysis_result.json").read_text(encoding="utf-8"))
    verification = json.loads((REPORT / "validation/verification.json").read_text(encoding="utf-8"))
    clock = json.loads((REPORT / "validation/clock_receipt.json").read_text(encoding="utf-8"))
    if verification["status"] != "PASS_TARGETED_TESTS_AND_SAVED_RECOMPUTATION":
        raise ValueError("本轮针对性验证未通过，不能称为通过的交付")
    code_hashes = {sha((ROOT / SCRIPTS[0]).read_bytes()), sha((REPORT / "code_history/probe_before_batch_10.py").read_bytes())}
    if any(batch["script_sha256_before_request"] not in code_hashes for batch in result["receipts"]["batches"]):
        raise ValueError("某次请求清单引用的采集代码原文没有纳入本轮")
    source_documents = REPORT / "source_documents"
    source_documents.mkdir(exist_ok=True)
    copies = [
        ("source_batches/09_fund_valuation_method/sse_510300_prospectus_2026_1_pdf.raw", "510300_招募说明书_2026年第1号.pdf", [33, 34, 38, 76]),
        ("source_batches/01_initial/sse_step_spec_202508_pdf.raw", "IS120_STEP_0_58_202508.pdf", [20, 24]),
        ("source_batches/01_initial/sse_rule_tech_2026_pdf.raw", "2026交易规则技术实施指南.pdf", []),
    ]
    pdf_files = []
    for source, name, reviewed_pages in copies:
        raw = (REPORT / source).read_bytes()
        if not raw.startswith(b"%PDF-"):
            raise ValueError("保存原件缺少PDF文件标识")
        target = source_documents / name
        target.write_bytes(raw)
        pdf_files.append({"raw_path": source, "readable_copy": target.relative_to(REPORT).as_posix(),
                          "sha256": sha(raw), "bytes": len(raw), "reviewed_physical_pages": reviewed_pages})
    images = [{"path": p.relative_to(REPORT).as_posix(), "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
              for p in sorted((REPORT / "pdf_pages").glob("*.png"))]
    if len(images) != 6:
        raise ValueError("本轮声明的6个已查看渲染页面不齐")
    (REPORT / "pdf_visual_review.json").write_bytes(json_bytes({
        "status": "SPECIFIC_SOURCE_PAGES_RENDERED_AND_VIEWED_BY_CODEX", "files": pdf_files,
        "page_images": images, "renderer": "pdfplumber", "renderer_version": version("pdfplumber"),
        "review_note": "本轮逐张查看指定6页，核对公式、表格归属、单位和现金替代文字；未声称整本119页已完成视觉审阅。",
        "original_pdfs_modified": False, "human_external_review": False,
        "is118_pdf_available": False, "is118_web_extract_only": True,
    }))
    (REPORT / "05_执行范围与状态.json").write_bytes(json_bytes({
        "study_id": result["study_id"], "recorded_at_local_clock": datetime.now(CN).isoformat(),
        "status": result["status"], "user_followup": "继续", "expected_observation_date": "20260907",
        "scope": "执行上一轮最小来源可行性核查，完成输入与字段解释；不是新策略试验",
        "source_selection": "按前一批已见网页逐步发现下一批公开入口；各批请求清单先于请求保存",
        "parse_contract_timing": "看见来源结构后制定，不宣称策略事前冻结",
        "historical_walk_forward_authorized_by_v6": True, "historical_training_performed": False,
        "strategy_registered": False, "automatic_collection_scheduled": False,
        "executable_assets": ["510300.SH", "CASH_CNY"], "other_etf_market_data_requested": False,
        "underlying_close_prices_requested": False, "pcf_components_only_as_510300_input_definition": True,
        "current_authority": "510300_RESEARCH_AUTHORITY_V6", "broker_connections": 0,
        "model_action": "ABSTAIN", "model_position_target": "UNSET", "position_impact": 0,
        "old_forward_ledger_writes": 0, "qualified_close_observations": 0,
        "qualified_post_close_observations": 0, "new_forward_quality_days": 0,
        "reference_value": "NOT_COMPUTED", "event_returns": "NOT_COMPUTED", "net_sharpe": "NOT_COMPUTED",
        "local_clock_state": clock["state"], "source_update_time_contract": "NOT_IDENTIFIED",
        "actual_fill_fraction": "UNKNOWN_NO_ORDER_EVIDENCE", "security_audit": False,
        "external_gpt_review_received": False,
    }))
    (REPORT / "08_包含与排除说明.json").write_bytes(json_bytes({
        "included": ["本轮全部报告、输出、33次GET请求清单与回执、26份原始正文、失败记录", "本轮采集原文两个版本、离线解析、验证、8项测试与打包代码", "3份PDF原件的相同字节可读副本、6个已查看页面和提取文本", "本轮校时状态、当前V6权限和续行依据", "上一轮完整50成员快照，保留原索引在prior_review下"],
        "excluded": ["未取得的IS118原始PDF；仅纳入收到的网页提取文本与失败回执", "没有授权和运行的券商、Paper/Shadow、订单、实盘", "未获取的279只成分收盘行情、未运行的事件收益和账户路径", "项目其他策略的完整历史数据与重跑环境；只保留上一轮已声明的上下文快照", "Python环境、安装缓存、临时目录和本轮输出之外的工作区文件"],
        "prior_archive_identity": {"file_name": PRIOR.name, "bytes": PRIOR.stat().st_size, "sha256": PRIOR_SHA, "checks": prior_check},
        "prior_snapshot_status": "REVIEW_COMPLETE_PLAN_NOT_RUN", "current_snapshot_status": result["status"],
        "root_index_scope": "根FILE_INDEX.csv覆盖新包所有其他成员；prior_review/FILE_INDEX.csv仅解释旧包",
        "self_contained_claim_scope": "可离线复算本轮已保存来源与核对表；不声称能重跑所有旧策略或在线源采集",
        "validation_scope": "解析语义测试、保存输出复算、ZIP结构；不作安全性审计或外部GPT审阅",
    }))


def payloads() -> dict[str, bytes]:
    prior_raw = PRIOR.read_bytes()
    if sha(prior_raw) != PRIOR_SHA:
        raise ValueError("上一轮审阅包身份不符，停止复用")
    prior_check = structural_check(PRIOR)
    prepare_report_metadata(prior_check)
    files = {}
    for path in sorted(REPORT.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for relative in SCRIPTS + ["config/510300_research_authority_v6.json", "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md"]:
        files[relative] = (ROOT / relative).read_bytes()
    with zipfile.ZipFile(io.BytesIO(prior_raw)) as archive:
        for name in archive.namelist():
            files["prior_review/" + name] = archive.read(name)
    report_prefix = REPORT_REL.as_posix()
    root_readme = f"""# 510300 收盘价格让步：来源可行性 V1 审阅包

本轮已完成来源核查：PCF的14个基本字段与2400项成分字段比较一致，公开官网有IOPV和盘后量额字段；同步收盘估值、盘后未成交申报量与合格时钟仍不足。未计算价差、事件收益、账户或净夏普。

阅读顺序：

1. [{report_prefix}/02_来源可行性结论.md]({report_prefix}/02_来源可行性结论.md)。
2. [{report_prefix}/03_下一项验证设计_未运行.md]({report_prefix}/03_下一项验证设计_未运行.md)。
3. [{report_prefix}/04_来源字段与证据映射.md]({report_prefix}/04_来源字段与证据映射.md)。
4. 同目录analysis_result.json、PCF两张核对表、source_batches原件与回执、validation验证记录。
5. [01_GPT_REVIEW_PROMPT.md](01_GPT_REVIEW_PROMPT.md)，含批判与下一步优先级/验证/停止条件要求。

当前权限见config/510300_research_authority_v6.json。prior_review是上一轮完整快照，原状态仅属于上一轮；本包根FILE_INDEX.csv是新包的权威索引。本轮只完成来源可行性，没有策略登记、前向质量日或自动采集调度。

离线复核：在本包解压根目录，用Python 3.11或以上运行 `python -X utf8 scripts/analyze_510300_close_concession_sources_v1.py --verify-saved`；运行8项测试用 `python -X utf8 -m unittest discover -s tests -p test_510300_close_concession_sources_v1.py -v`。均仅依赖标准库，不联网。Windows当前校时状态保存在validation中；不要把原来未同步状态解释为新机器当前状态。

本包仅为本轮提供可离线复核的直接来源，不包含在线数据保证或所有旧研究的重跑环境。只进行必要的语义验证、保存输出复算与ZIP结构检查，不进行安全性审计，也未送交外部GPT。
"""
    files["00_README_FIRST.md"] = root_readme.encode("utf-8")
    files["01_GPT_REVIEW_PROMPT.md"] = (REPORT / "01_GPT_REVIEW_PROMPT.md").read_bytes()
    files["user/本轮续行请求.txt"] = "用户本轮原文：继续\n含义：继续上一轮第一类价格让步的来源可行性核查。原始机制长文随prior_review/user保留。\n".encode("utf-8")
    frozen = []
    for name, raw in sorted(files.items()):
        if "/source_batches/" in name or "/code_history/" in name or name == SCRIPTS[0]:
            frozen.append({"path": name, "bytes": len(raw), "sha256": sha(raw)})
    files["FROZEN_SOURCE_FILES.json"] = json_bytes({"scope": "来源请求前清单、全部回执、原始正文与所用采集脚本历史版本", "files": frozen})
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows({"path": name, "bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(files.items()))
    files["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    return files


def verify_packaged_recomputation(path: Path) -> dict:
    # 临时目录仅位于明确的deliverables目录下，进入后先核对最终绝对路径，再允许自动清理。
    with tempfile.TemporaryDirectory(prefix="510300_close_source_verify_", dir=OUT) as temporary:
        target_root = Path(temporary).resolve()
        if not target_root.is_relative_to(OUT.resolve()):
            raise ValueError("复算临时目录超出明确的交付目录")
        with zipfile.ZipFile(path) as archive:
            expected_prefix = REPORT_REL.as_posix() + "/"
            for name in archive.namelist():
                if name.startswith(expected_prefix) or name == SCRIPTS[1]:
                    target = (target_root / name).resolve()
                    if not target.is_relative_to(target_root):
                        raise ValueError("复算成员路径超出临时目录")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(name))
        command = [sys.executable, "-B", "-X", "utf8", str(target_root / SCRIPTS[1]), "--report", str(target_root / REPORT_REL), "--verify-saved"]
        run = subprocess.run(command, cwd=target_root, capture_output=True, timeout=60, check=False)
        output = (run.stdout + run.stderr).decode("utf-8")
        if run.returncode:
            raise ValueError("包内原始输入离线复算失败：" + output)
        return {"status": "PASS_PACKAGED_SAVED_SOURCE_RECOMPUTATION", "exit_code": run.returncode,
                "output": output, "network_requests": 0, "model_fits": 0, "future_return_reads": 0, "account_generation": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="构建510300本轮GPT审阅包，只做必要的结构与保存输出复核")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    target = OUT / (NAME + ".zip")
    if args.verify_only:
        check = structural_check(target)
        recomputation = verify_packaged_recomputation(target)
    else:
        if target.exists():
            raise ValueError("既有本轮ZIP已经存在；只读核对请使用--verify-only")
        files = payloads()
        building = OUT / (NAME + ".building.zip")
        with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, raw in sorted(files.items()):
                archive.writestr(name, raw)
        check = structural_check(building)
        recomputation = verify_packaged_recomputation(building)
        os.replace(building, target)
    receipt = {"status": "PASS_STRUCTURAL_AND_PACKAGED_OFFLINE_RECOMPUTATION", "created_at_local_clock": datetime.now(CN).isoformat(),
               "archive": str(target), "bytes": target.stat().st_size, "sha256": sha(target.read_bytes()),
               "structure": check, "packaged_recomputation": recomputation,
               "security_audit": False, "external_gpt_review": False, "upload_performed": False,
               "scope": "核对本轮来源与交付；不是策略收益或成交有效性证明"}
    receipt_path = OUT / (NAME + "_delivery_receipt.json")
    receipt_path.write_bytes(json_bytes(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
