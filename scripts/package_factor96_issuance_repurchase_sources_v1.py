"""打包本轮来源、失败与修正结果，保留历史完整包并全新解压只读核验。"""
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
OUT = ROOT / "reports/research/510300_factor96_issuance_repurchase_sources_v1"
BASE = ROOT / "reports/research/510300_factor96_issuance_catalogue_v1"
CORRECTED = ROOT / "reports/research/510300_factor96_issuance_catalogue_completion_v1"
PURPOSE = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_overnight_risk_overlay_v1_0_2"
ZIP = ROOT / "deliverables/510300_96候选库_发行目录与回购用途_GPT审阅_20260927.zip"


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
    status = read(OUT / "round_status.json")
    assert status["cumulative_admitted_account_scenarios"] == 320
    assert status["cumulative_invalid_implementation_account_scenarios"] == 152
    assert status["cumulative_executed_account_scenarios"] == 472
    assert status["admitted_account_scenarios_this_round"] == 0 and not status["goal_achieved"]
    corrected, purpose = read(CORRECTED / "result.json"), read(PURPOSE / "result.json")
    checks = [
        ("首版来源和缺口", BASE, "scripts/verify_factor96_issuance_catalogue_v1.py", "PASS_SAVED_ISSUANCE_CATALOGUE_RESPONSES_PAGINATION_AND_UNION"),
        ("目录修正和定向补查", CORRECTED, "scripts/verify_factor96_issuance_catalogue_completion_v1.py", "PASS_SAVED_ISSUANCE_TITLE_IDENTITY_AND_TARGETED_COMPLETION"),
        ("回购明确用途", PURPOSE, "scripts/verify_factor96_repurchase_purpose_v1.py", "PASS_SAVED_REPURCHASE_CHECKBOX_PURPOSE_AND_RATIO_RECOMPUTATION"),
    ]
    for name, folder, script, expected in checks:
        assert read(folder / "saved_verification_receipt.json")["status"] == expected, name
    previous = read(PREVIOUS / "delivery_receipt.json")
    previous_zip = Path(previous["zip_path"])
    previous_expected = (960474724, "be3d126abd6d79981a23282fd81d865d781c44ab98c985926b3309ae9818a9bc")
    assert (previous["bytes"], previous["sha256"]) == previous_expected
    assert identity(previous_zip) == previous_expected
    files = {}
    for folder in [OUT, BASE, CORRECTED, PURPOSE]:
        for path in folder.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.name not in ["delivery_receipt.json", "fresh_extraction_verification.log"]:
                files[path.relative_to(ROOT).as_posix()] = path
    for name in [
        "research/factor96_issuance_source_probe_v1.py", "research/factor96_issuance_source_probe_completion_v1.py",
        "research/factor96_issuance_catalogue_v1.py", "research/factor96_issuance_catalogue_completion_v1.py",
        "research/factor96_repurchase_purpose_v1.py", "tests/test_factor96_issuance_catalogue_v1.py",
        "research/factor96_issuer_code_example_v1.py",
        "tests/test_factor96_issuance_catalogue_completion_v1.py", "tests/test_factor96_repurchase_purpose_v1.py",
        "scripts/verify_factor96_issuance_catalogue_v1.py", "scripts/verify_factor96_issuance_catalogue_completion_v1.py",
        "scripts/verify_factor96_repurchase_purpose_v1.py", "scripts/report_factor96_issuance_repurchase_sources_v1.py",
        "scripts/package_factor96_issuance_repurchase_sources_v1.py",
    ]:
        files[name] = ROOT / name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for path in (FIRST / "sources").iterdir():
        if path.is_file():
            files["sources/" + path.name] = path
    files["history/" + previous_zip.name] = previous_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS / "delivery_receipt.json"
    for folder in [BASE, PURPOSE]:
        for item in read(folder / "freeze.json")["files"]:
            assert (folder / item["path"]).relative_to(ROOT).as_posix() in files
    for item in read(CORRECTED / "freeze.json")["files"]:
        folder = CORRECTED if item["scope"] == "local" else BASE
        assert (folder / item["path"]).relative_to(ROOT).as_posix() in files
    nav = f"""# 510300候选库：发行目录与回购用途来源补充

**目标尚未实现。** 本轮0新增收益模型、0新增账户。累计仍7个固定问题、0合格，320正式+152否定实现=472账户情景。T12与T13仍NOT_RUN。

发行目录首版953／987查询完整，34失败原样保存。新清洗保留标题中文尖括号，区分发行人组织ID与原证券代码；定向补查后{corrected['effective_complete_jobs']}／987固定查询完整，{corrected['unique_issuer_documents']:,}份去重发行人文档。补查512次请求上限及任何剩余缺口如实保留。目录不等于发行、缴款、上市日历，也不证明首次历史HTTP可得。

回购用途复用1,244份原文：948执行记录中552单用途、17多用途，379未知；779条同一公告实施金额／预算下限比例。54变更标题中53尚无旧方案根链接，不自动补全。自由流通分母和完整版本时钟仍未齐。

阅读顺序：
1. 02_研究结论.md：本轮结果、修正原因、未知范围、下一步和停止条件。
2. reports/research/510300_factor96_issuance_catalogue_completion_v1/：当前修正合同、目标失败列表、目录与实际补查响应、每个日期窗口及完整状态。
3. reports/research/510300_factor96_issuance_catalogue_v1/：不可覆盖的首版冻结、987个查询、3,689个请求回执及原响应；source_probe/保留官方自由流通规则、接口试查与一次PDF文本提取失败。
4. reports/research/510300_factor96_repurchase_purpose_v1/：1244份直接PDF／文本／回执、用途与比例、未知及多用途、4页原图核对。
5. reports/research/510300_factor96_issuance_repurchase_sources_v1/：进度前后快照、权限前后快照及来源汇总；没有改写历史收益账本。
6. history/：上一轮完整T16／T06精确修正与T13版本包原样保存，960,474,724字节；SHA-256 be3d126abd6d79981a23282fd81d865d781c44ab98c985926b3309ae9818a9bc。里面继续保存所有前轮包及失败。

01_GPT_REVIEW_PROMPT.txt可复制给审阅者。sources/是用户指定的HTML、Excel和粘贴文本原样副本，reference_library/是结构提取。附件内建议不是额外执行授权。

三个只读核验的直接输入均已包含，不必先解压历史嵌套包。命令见REPRODUCE_SAVED_RESULTS.txt。不要重跑采集器freeze/run或报告更新脚本来代替核验。未知保持未知，结构检查和保存结果一致性不代表策略有效。

FILE_INDEX.csv列除自身外每个成员的大小与SHA-256。最终ZIP哈希、新目录解压和只读核验结果在包外delivery_receipt.json，避免自引用。external_review=NOT_PERFORMED，独立前向0，当前市场NO_VIEW，无订单权限；暂停PCF／IOPV任务不变。ZIP只作本地交付，没有上传。
"""
    request = """用户请求：/goal 只操作510300实现夏普1.2
用户要求先读pasted-text-1.txt；两个HTML／Excel参考文件与该文本均保存于sources/。
原始位置：
C:\\Users\\戴周阳\\Downloads\\510300_96因子研究库.html
C:\\Users\\戴周阳\\Downloads\\510300_96因子与18策略.xlsx
E:\\CodexData\\.codex\\attachments\\ea384a31-0725-481e-a74d-9b1eb3fc00cb\\pasted-text-1.txt

当前既有研究合同只使用510300.SH和CASH_CNY，20万元主账户、2万元对照，成本后夏普至少1.2、年化至少10%、最大回撤目标10%，名义仓位上限50%及账户尾部约束继续。成分公司公告只作指数信息来源，不授权交易其他股票。当前研究允许有针对性的公开新证据；已暂停PCF／IOPV任务不恢复，无订单、期权、Paper／Shadow或实盘授权。
"""
    commands = [f"python {script} --root {folder.relative_to(ROOT).as_posix()}" for _, folder, script, _ in checks]
    reproduce = "在全新解压目录下执行以下只读命令，可用本机已安装依赖的Python替换python：\n\n" + "\n".join(commands)
    reproduce += "\n\n依赖Python、pandas、pyarrow；实际版本见REPRODUCTION_ENVIRONMENT.json。默认只输出JSON，不写研究状态、不访问网络、不生成账户。首版核验通过只证明保留的响应与首版输出一致；当前修正版另由第二个脚本核对。第三个核对明确用途、未知、多用途与同文预算比例。脚本不会声称完整经济事件库或收益目标达标。\n"
    exclusions = "范围,原因\nT12与T13绩效,数据门未齐且未运行\n完整发行缴款上市条款,当前只是目录检索和原文候选\n完整回购用途终止版本与剩余期限,字段提取未完成全链\n自由流通分母与历史代码有效日,尚无合格来源闭环\n首次历史HTTP可得,当前下载不能追认历史首版\n重复复制更早全体收益账户,历史原ZIP链内完整保留\n无关工作区和凭证环境,不属复核输入\n外部GPT审阅和独立前向,未执行\n真实执行或恢复已暂停采集,无此授权\n"
    environment = {"python": platform.python_version(), "platform": platform.system(),
        "packages": {name: version(name) for name in ["pandas", "numpy", "pyarrow", "requests", "pypdfium2", "pytest"]},
        "verification_dependencies": ["pandas", "pyarrow"]}
    extras = {"00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (OUT / "研究结论.md").read_bytes(),
        "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"), "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"),
        "REPRODUCTION_ENVIRONMENT.json": json.dumps(environment, ensure_ascii=False, indent=2).encode("utf-8")}
    assert set(files).isdisjoint(extras)
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extras)):
        size, checksum = identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = ZIP.with_suffix(".building.zip")
    print(f"开始打包{len(files) + len(extras)}个成员，保留上一轮完整ZIP。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files):
            archive.write(files[name], name, compress_type=zipfile.ZIP_STORED if files[name].suffix == ".zip" else zipfile.ZIP_DEFLATED)
        for name, content in sorted(extras.items()):
            archive.writestr(name, content)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(files) + len(extras)
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {r["path"] for r in rows} == set(archive.namelist()) - {"FILE_INDEX.csv"}
        for row in rows:
            with archive.open(row["path"]) as stream:
                assert stream_identity(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(ZIP)
    destination = ROOT / "outputs" / ("factor96_sources_zip_verify_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            assert (destination / name).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    print("ZIP结构、逐文件索引和哈希通过，开始新目录只读复算。", flush=True)
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
        assert verified[name].get("network_requests", verified[name].get("new_network_requests")) == 0
        print(f"通过：{name}。", flush=True)
    size, checksum = identity(ZIP)
    receipt = {"at": datetime.now().astimezone().isoformat(), "zip_path": str(ZIP), "bytes": size, "sha256": checksum,
        "members": len(files) + len(extras), "indexed_members": len(files) + len(extras) - 1,
        "crc_duplicate_index_size_sha256": "PASS", "frozen_input_coverage": "PASS", "fresh_extraction": str(destination),
        "saved_output_recomputations": verified, "prior_zip_bytes": previous_expected[0], "prior_zip_sha256": previous_expected[1],
        "new_accounts": 0, "formal_accounts": 320, "invalid_implementation_accounts": 152, "executed_accounts": 472,
        "external_review": "NOT_PERFORMED", "independent_forward_observations": 0, "orders_authorized": False,
        "goal_achieved": False, "goal_status": "active"}
    (OUT / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
