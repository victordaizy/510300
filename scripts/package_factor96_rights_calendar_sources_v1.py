"""归档发行检索与配股原文时序，在全新解压目录只读复核。"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_rights_calendar_sources_v1"
CATALOGUE = ROOT / "reports/research/510300_factor96_issuance_remainder_v1"
DOCUMENTS = ROOT / "reports/research/510300_factor96_rights_issue_documents_v1"
FIELDS = ROOT / "reports/research/510300_factor96_rights_calendar_fields_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_issuance_repurchase_sources_v1"
ZIP = ROOT / "deliverables/510300_96候选库_发行检索完成与配股时序_GPT审阅_20260927.zip"
PRIOR_IDENTITY = (1277443370, "00dfd5df9f5c55df22843bd266526438656385c96d30db5eabe773b8f5a56230")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream_identity(stream):
    size, checksum = 0, hashlib.sha256()
    while block := stream.read(1024 * 1024):
        size += len(block)
        checksum.update(block)
    return size, checksum.hexdigest()


def identity(path):
    with path.open("rb") as stream:
        return stream_identity(stream)


def main():
    assert not ZIP.exists(), "最终文件已存在，不覆盖"
    assert not (OUT / "delivery_receipt.json").exists(), "交付回执已存在，不覆盖"
    status = read(OUT / "round_status.json")
    assert status["cumulative_admitted_account_scenarios"] == 320
    assert status["cumulative_invalid_implementation_account_scenarios"] == 152
    assert status["cumulative_executed_account_scenarios"] == 472
    assert status["admitted_account_scenarios_this_round"] == 0 and not status["goal_achieved"]
    assert status["goal_status"] == "active" and not status["orders_authorized"]
    catalogue, fields = read(CATALOGUE / "result.json"), read(FIELDS / "result.json")
    checks = [
        ("目录继承与剩余日期窗口", CATALOGUE, "scripts/verify_factor96_issuance_remainder_v1.py",
         "PASS_SAVED_REMAINDER_RESPONSES_INHERITANCE_AND_DATE_CLOSURE"),
        ("配股原文与条款版本时序", FIELDS, "scripts/verify_factor96_rights_calendar_fields_v1.py",
         "PASS_SAVED_RIGHTS_ORIGINALS_CLAUSE_ANCHORS_AND_VERSION_CLOCK"),
    ]
    for name, folder, script, expected in checks:
        assert read(folder / "saved_verification_receipt.json")["status"] == expected, name
    previous = read(PREVIOUS / "delivery_receipt.json")
    previous_zip = Path(previous["zip_path"])
    assert (previous["bytes"], previous["sha256"]) == PRIOR_IDENTITY
    assert identity(previous_zip) == PRIOR_IDENTITY

    files = {}
    for folder in [OUT, CATALOGUE, DOCUMENTS, FIELDS]:
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.name not in ["delivery_receipt.json", "fresh_extraction_verification.log"]:
                files[path.relative_to(ROOT).as_posix()] = path
    for name in [
        "research/factor96_issuance_remainder_v1.py", "research/factor96_issuance_catalogue_v1.py",
        "research/factor96_issuance_catalogue_completion_v1.py", "research/factor96_rights_issue_documents_v1.py",
        "research/factor96_rights_document_transport_completion_v1.py", "research/factor96_rights_calendar_fields_v1.py",
        "tests/test_factor96_issuance_remainder_v1.py", "tests/test_factor96_rights_calendar_fields_v1.py",
        "scripts/verify_factor96_issuance_remainder_v1.py", "scripts/verify_factor96_rights_calendar_fields_v1.py",
        "scripts/report_factor96_rights_calendar_sources_v1.py", "scripts/package_factor96_rights_calendar_sources_v1.py",
    ]:
        files[name] = ROOT / name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for path in (FIRST / "sources").iterdir():
        if path.is_file():
            files["sources/" + path.name] = path
    files["history/" + previous_zip.name] = previous_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS / "delivery_receipt.json"
    for folder in [CATALOGUE, DOCUMENTS, FIELDS]:
        for item in read(folder / "freeze.json")["files"]:
            location = DOCUMENTS if item.get("scope") == "source" else folder
            path = location / item["path"]
            assert path.relative_to(ROOT).as_posix() in files, item
            assert identity(path)[1] == item["sha256"], item

    nav = f"""# 510300候选库：发行检索完成与配股原文时序

**夏普1.2目标尚未实现。** 本轮0新账户、0新收益模型。累计7个固定问题仍0合格，320正式账户情景+152否定实现情景=472历史执行情景。T13仍NOT_RUN。

固定发行检索从971／987补至{catalogue['effective_complete_jobs']}／987完整，共{catalogue['unique_issuer_documents']:,}份去重发行人目录文档、94,441条来源记录；新增3,398条来自353次HTTP请求和21个原未请求日期窗口。检索器完成不等于全部经济发行事件完整，目录文档数也不是已取得全文的数量。

本阶段固定取得491份配股原始PDF与文本，涉及44个发行人。初次490成功、1份两次TLS失败；随后单独补取这一失败文档成功，原失败和原结果保持不变。直接来源总计694,511,769字节、22,733页。全体18,179个条款定位候选均未准入为交易字段，覆盖452份文档；其余39份不当作没有供给。另717份配股标题未纳入此阶段全文，排除清单保留。

目前明确核对的是1个兴业证券2015—2016年四文档示例：19个证据锚点、8个时点。缴款、清算、除权复牌和新增股份上市分别记录；更正值不会回填到更正公告之前；原文年份冲突保留未知。目录日结束时钟只是保守示例，不能证明历史首次HTTP可得。

阅读顺序：
1. 02_研究结论.md：本轮结果、原文案例、未知、验证条件和下一步。
2. reports/research/510300_factor96_issuance_remainder_v1/：冻结前版表、21个剩余窗口、353个新请求回执及原响应、当前987查询状态和目录。
3. reports/research/510300_factor96_rights_issue_documents_v1/：491目标与选择依据、原批次490份PDF／文本、原失败；transport_completion_v1/含独立补取、旧失败哈希和有效491文档清单。
4. reports/research/510300_factor96_rights_calendar_fields_v1/：全部定位候选、四份方便打开的case_pdfs/、19个锚点、版本与时点表、四页原图和冻结协议。
5. reports/research/510300_factor96_rights_calendar_sources_v1/：来源汇总、研究进度与权限前后快照，没有改写历史收益账本。
6. history/：上一轮完整来源与历史研究包原样保留，1,277,443,370字节；SHA-256 00dfd5df9f5c55df22843bd266526438656385c96d30db5eabe773b8f5a56230。其内部继续保留更早的完整包、负结果及失败回执。

01_GPT_REVIEW_PROMPT.txt供复制审阅。sources/为用户HTML、Excel及粘贴文本的原样副本；reference_library/为提取的96因子与18策略登记表。参考文件里的建议不是额外执行授权。

本轮两个只读核验的直接依赖均已包含，不必展开历史嵌套ZIP，命令见REPRODUCE_SAVED_RESULTS.txt。目录核验复算前版表继承及本轮新响应；前版原响应的完整复核证据在历史包及其回执，本轮没有重复递归执行旧轮核验。不要运行freeze/run采集器或报告更新入口来代替只读核验。

FILE_INDEX.csv记录除自身外每个成员大小与SHA-256。包外delivery_receipt.json记录最终ZIP身份和全新解压结果以避免自引用。未知保持未知。external_review=NOT_PERFORMED，独立前向0，当前市场NO_VIEW，无下单授权，原PCF／IOPV计划任务保持暂停。本包本地交付，没有上传。
"""
    request = """用户请求：/goal 只操作510300实现夏普1.2
用户要求先读pasted-text-1.txt；该粘贴文本、HTML及Excel原样副本已放入sources/。
原始位置：
C:\\Users\\戴周阳\\Downloads\\510300_96因子研究库.html
C:\\Users\\戴周阳\\Downloads\\510300_96因子与18策略.xlsx
E:\\CodexData\\.codex\\attachments\\ea384a31-0725-481e-a74d-9b1eb3fc00cb\\pasted-text-1.txt

当前研究合同仅使用510300.SH和CASH_CNY，20万元主账户、2万元对照，成本后夏普至少1.2、净年化至少10%、最大回撤目标10%。既有名义仓位上限50%、成熟两年5日ES95预算0.025、负10%缺口预算0.05、剩余回撤空间一半及回撤10%下一合法开盘退出不重启的约束继续。
成分公司原文只作指数研究信息来源，不授权交易其他股票。当前允许定向公开新证据采集；已暂停PCF／IOPV任务不恢复，无订单、期权、Paper／Shadow或实盘授权。目标未完成，不因归档或源码测试把目标标为达成。
"""
    commands = [f"python {script} --root {folder.relative_to(ROOT).as_posix()}" for _, folder, script, _ in checks]
    reproduce = "在全新解压目录执行以下只读命令，可用已安装依赖的Python替换python：\n\n" + "\n".join(commands)
    reproduce += "\n\n依赖Python、pandas、pyarrow，实际版本见REPRODUCTION_ENVIRONMENT.json。默认只输出JSON，不写研究状态、不联网、不生成账户。第二项从同级rights_issue_documents目录读取491份保存原文与文本，检查条款定位、整数金额、版本和查询时钟；不重新访问外部服务，也不证明历史首次HTTP可得或策略有效。已有6项日期窗口测试、10项字段时钟测试的原始回执在对应研究目录，无需重新运行采集器。\n"
    exclusions = "范围,原因\nT13及T12收益结果,数据门尚未满足且本轮未运行\n全部经济发行事件,987固定查询完整不等于经济事件全集\n717份其他角色配股标题全文,不属于本阶段冻结全文选择范围且排除表保留\n全体事件字段及更正终止链,18179条只作原文定位候选\n普通流通总股本或靠档权重替代自由流通分母,不能满足当前历史口径\n首次历史HTTP可得,当前下载和目录日不能追认\n历史全部源码结果重复平铺,前轮完整ZIP原样嵌套保留\n无关工作区及凭证环境,不是复核输入\n外部GPT审阅及独立前向观察,未执行\n真实交易或恢复暂停计划任务,无此授权\n"
    environment = {
        "python": platform.python_version(), "platform": platform.system(),
        "packages": {name: version(name) for name in ["pandas", "numpy", "pyarrow", "requests", "pypdfium2", "pytest"]},
        "verification_dependencies": ["pandas", "pyarrow"],
    }
    extras = {
        "00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (OUT / "研究结论.md").read_bytes(),
        "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"), "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"),
        "REPRODUCTION_ENVIRONMENT.json": json.dumps(environment, ensure_ascii=False, indent=2).encode("utf-8"),
    }
    assert set(files).isdisjoint(extras)
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extras)):
        size, checksum = identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = ZIP.with_suffix(".building.zip")
    print(f"开始打包{len(files) + len(extras)}个成员，上一轮完整包原字节保留。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files):
            archive.write(files[name], name, compress_type=zipfile.ZIP_STORED if files[name].suffix == ".zip" else zipfile.ZIP_DEFLATED)
        for name, content in sorted(extras.items()):
            archive.writestr(name, content)
    print("文件写入结束，核对ZIP结构与逐文件大小和哈希。", flush=True)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(files) + len(extras)
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert len(rows) == len(archive.namelist()) - 1
        assert {r["path"] for r in rows} == set(archive.namelist()) - {"FILE_INDEX.csv"}
        for row in rows:
            with archive.open(row["path"]) as stream:
                assert stream_identity(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(ZIP)
    destination = ROOT / "outputs" / ("factor96_rights_calendar_zip_verify_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            assert (destination / name).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    print("结构与逐文件索引通过，开始从全新解压目录只读复算。", flush=True)
    verified, logs = {}, []
    for name, folder, script, expected in checks:
        command = [sys.executable, "-X", "utf8", str(destination / script), "--root", str(destination / folder.relative_to(ROOT))]
        print(f"开始核对：{name}。", flush=True)
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
        logs.append(name + "\n" + run.stdout + run.stderr)
        (OUT / "fresh_extraction_verification.log").write_text("\n".join(logs), encoding="utf-8")
        assert run.returncode == 0, run.stderr[-4000:]
        verified[name] = json.loads(run.stdout.strip().splitlines()[-1])
        assert verified[name]["status"] == expected and verified[name]["new_accounts"] == 0
        assert verified[name]["network_requests"] == 0
        print(f"通过：{name}。", flush=True)
    size, checksum = identity(ZIP)
    receipt = {
        "at": datetime.now().astimezone().isoformat(), "zip_path": str(ZIP), "bytes": size, "sha256": checksum,
        "members": len(files) + len(extras), "indexed_members": len(files) + len(extras) - 1,
        "crc_duplicate_index_size_sha256": "PASS", "frozen_input_coverage": "PASS", "fresh_extraction": str(destination),
        "saved_output_recomputations": verified, "prior_zip_bytes": PRIOR_IDENTITY[0], "prior_zip_sha256": PRIOR_IDENTITY[1],
        "complete_fixed_queries": catalogue["effective_complete_jobs"], "unique_catalogue_documents": catalogue["unique_issuer_documents"],
        "rights_source_documents": fields["source_documents"], "clause_candidates": fields["clause_candidates"],
        "new_accounts": 0, "formal_accounts": 320, "invalid_implementation_accounts": 152, "executed_accounts": 472,
        "external_review": "NOT_PERFORMED", "independent_forward_observations": 0, "orders_authorized": False,
        "goal_achieved": False, "goal_status": "active",
    }
    (OUT / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
