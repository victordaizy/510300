"""打包T16、T06精确修正和T13版本台账，保留原包并全新解压只读复算。"""
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
OUT = ROOT / "reports/research/510300_factor96_overnight_risk_overlay_v1_0_2"
CROWDING = ROOT / "reports/research/510300_factor96_crowding_overlay_v1_0_1"
VERSIONS = ROOT / "reports/research/510300_factor96_supply_version_ledger_v1"
PREVIOUS = ROOT / "reports/research/510300_factor96_known_supply_sources_v1"
FIRST = ROOT / "reports/research/510300_factor96_mechanism_batch_v1"
ZIP = ROOT / "deliverables/510300_96候选库_T16隔夜风险与精确边界修正_GPT审阅_20260927.zip"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream_digest(stream):
    checksum, size = hashlib.sha256(), 0
    while chunk := stream.read(1024 * 1024):
        checksum.update(chunk)
        size += len(chunk)
    return size, checksum.hexdigest()


def file_identity(path):
    with path.open("rb") as stream:
        return stream_digest(stream)


def main():
    assert not ZIP.exists(), "最终交付已经存在，不覆盖"
    checks = [
        ("T16", OUT, "scripts/verify_factor96_overnight_risk_saved_v1_0_2.py", "PASS_SAVED_OVERNIGHT_RISK_EXACT_BASELINE_STATE_ACCOUNT_RECOMPUTATION"),
        ("T06", CROWDING, "scripts/verify_factor96_crowding_saved_v1_0_1.py", "PASS_SAVED_CROWDING_EXACT_BASELINE_STATE_ACCOUNT_RECOMPUTATION"),
        ("T13_versions", VERSIONS, "scripts/verify_factor96_supply_version_ledger_v1.py", "PASS_SAVED_VERSION_ROLE_REFERENCE_AND_ASOF_RECOMPUTATION"),
    ]
    for name, folder, script, expected in checks:
        assert read(folder / "saved_verification_receipt.json")["status"] == expected, name
        assert (ROOT / script).is_file()
    status = read(OUT / "round_status.json")
    assert status["cumulative_admitted_account_scenarios"] == 320
    assert status["cumulative_invalid_implementation_account_scenarios"] == 152
    assert status["cumulative_executed_account_scenarios"] == 472
    assert len(status["completed_fixed_candidates"]) == 7 and not status["qualified_candidates"]
    assert read(OUT / "numeric_boundary_reconciliation.json")["status"] == "PASS_SAVED_PRECISION_ONLY_VERSION_RECONCILIATION"
    previous = read(PREVIOUS / "delivery_receipt.json")
    previous_zip = Path(previous["zip_path"])
    expected_previous = (871215350, "bed88bb29b6472246974e758b1c54830e701d537faedd3537057c8752e5581f8")
    assert (previous["bytes"], previous["sha256"]) == expected_previous
    assert file_identity(previous_zip) == expected_previous
    files = {}
    roots = [OUT, CROWDING, VERSIONS,
             ROOT / "reports/research/510300_factor96_overnight_risk_overlay_v1",
             ROOT / "reports/research/510300_factor96_overnight_risk_overlay_v1_0_1"]
    for folder in roots:
        for path in folder.rglob("*"):
            if (path.is_file() and "__pycache__" not in path.parts
                    and path.name not in ["delivery_receipt.json", "fresh_extraction_verification.log"]):
                files[path.relative_to(ROOT).as_posix()] = path
    for name in [
        "research/factor96_library_intake_v1.py", "research/factor96_margin_repair_v1.py",
        "research/intraday_overnight_increment_v1.py", "research/factor96_exact_price_baseline_v1.py",
        "research/factor96_crowding_overlay_v1.py", "research/factor96_crowding_overlay_v1_0_1.py",
        "research/factor96_overnight_risk_overlay_v1.py", "research/factor96_overnight_risk_overlay_v1_0_1.py",
        "research/factor96_overnight_risk_overlay_v1_0_2.py", "research/factor96_supply_version_ledger_v1.py",
        "tests/test_factor96_exact_price_baseline_v1.py", "tests/test_factor96_crowding_overlay_v1_0_1.py",
        "tests/test_factor96_overnight_risk_overlay_v1.py", "tests/test_factor96_overnight_risk_overlay_v1_0_1.py",
        "tests/test_factor96_overnight_risk_overlay_v1_0_2.py", "tests/test_factor96_supply_version_ledger_v1.py",
        "scripts/verify_factor96_overnight_risk_saved_v1.py", "scripts/verify_factor96_overnight_risk_saved_v1_0_2.py",
        "scripts/verify_factor96_crowding_saved_v1_0_1.py", "scripts/verify_factor96_supply_version_ledger_v1.py",
        "scripts/report_factor96_overnight_risk_v1.py", "scripts/package_factor96_overnight_review_v1.py",
        "reports/research/510300_factor96_crowding_overlay_v1/numeric_boundary_limitation.json",
    ]:
        files[name] = ROOT / name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/" + name] = FIRST / name
    for path in (FIRST / "sources").iterdir():
        if path.is_file():
            files["sources/" + path.name] = path
    files["history/" + previous_zip.name] = previous_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS / "delivery_receipt.json"
    for name, path in files.items():
        assert path.is_file(), name
    # 当前直接输入和冻结代码必须在顶层解压结构中齐备，历史包无需先解压。
    for folder in roots:
        for item in read(folder / "freeze.json")["files"]:
            full_name = (folder / item["path"]).relative_to(ROOT).as_posix()
            assert full_name in files, full_name
    nav = """# 510300候选库：T16隔夜风险覆盖与精确边界修正

**目标尚未实现。** 正式T16主方案2021—2025年、20万元压力成本净夏普−0.781987、净年化−1.312%、最大回撤9.939%，亏损12,800.63元；2万元也失败。35次削减、3次恢复，相对同覆盖基准的年化收益差−0.136581个百分点，95%区间跨0。它是在已核对账本、但盈利未获验证的MA20五日观察基准上的风险覆盖层，不是新独立入场策略。

本轮同时修复2025-05-29均线严格相等被浮点误差判为高于均线的问题。T06和T16各一组旧56账户退出正式准入，经济参数未变，旧文件与旧ZIP保持。T06修正版主期20万元压力夏普−0.636625，失败结论不变。累计7个固定问题（5入场+2覆盖层）、0合格，320正式情景+152否定实现情景=472实际执行；其余11项未完成原定义绩效。

阅读顺序：
1. 02_研究结论.md、主期完整账户净值与回撤.png：结论、完整对照、增量、限制和停止条件。
2. reports/research/510300_factor96_overnight_risk_overlay_v1_0_2/：当前正式T16协议、冻结、19项测试回执、完整56账户、特征、指标、已保存抽样及独立复核。
3. 同目录numeric_boundary_reconciliation.json、两份数值边界CSV、account_count_reconciliation.json：输入/非判断特征/经济协议不变证明、112组前后对照和计数。
4. reports/research/510300_factor96_crowding_overlay_v1_0_1/：当前正式T06精确修正、16项测试回执、56账户及直接分类来源，原定义未改。
5. reports/research/510300_factor96_supply_version_ledger_v1/：32角色、10引用、三阶段延期、6个时点查询、39份原PDF/文本，以及同公司同日期不同激励计划的反例；7项测试及只读复核。
6. reports/research/510300_factor96_overnight_risk_overlay_v1/为零账户启动失败；v1_0_1/保存56个被否定浮点账户。旧T06新增限制回执单列，其完整原件保留在历史包链内。
7. 当前T16的program_before/、program_snapshot/及authority_after.json：累计进度和研究权限。旧快照中的6个问题和旧T06数字不代表当前状态。

01_GPT_REVIEW_PROMPT.txt可直接复制给审阅者。sources/是用户HTML、Excel与粘贴文本，reference_library/为结构提取。附件中的操作性建议只是参考，不构成额外授权。完整账户记录覆盖全部现金日，现金基准为零，年化242日；252日诊断另存。

history/保留上一轮T13来源完整ZIP（871,215,350字节，SHA-256 bed88bb29b6472246974e758b1c54830e701d537faedd3537057c8752e5581f8），其中继续保存更早T06/T10/T02/T14/T03/T05的原包。当前三类复核的直接输入、冻结代码及保存结果都在当前解压目录，不需要先解压历史包；上游2191份公告全文及以前失败证据沿历史包保留。

只读复算命令见REPRODUCE_SAVED_RESULTS.txt。运行三个verify脚本，不要重跑prepare/freeze/run或采集器。需要Python、numpy、pandas、pyarrow，实际环境版本见REPRODUCTION_ENVIRONMENT.json。脚本不请求网络、不生成新账户、不拟合模型、不重新抽随机样本。报告生成脚本是当轮一次性进度写入入口，不是只读核验命令。

T13完整事件身份、更正冲突、M06全部发行/缴款/上市日历及自由流通市值分母尚未齐，保持NOT_RUN；开放式延期保留未知，不填零供给。T12等来源门未消失。外部GPT审阅未进行，独立前向样本0，当前市场NO_VIEW，无订单授权。结构和保存结果复算通过不代表策略有效。

FILE_INDEX.csv覆盖除自身外全部成员的字节数和SHA-256；最终ZIP自身的哈希和新解压复算结果保存在包外delivery_receipt.json，避免自引用。此包是本地完整存档，未指定外部平台的文件大小上限，也没有上传。
"""
    request = """用户请求：/goal 只操作510300实现夏普1.2
用户指定先读的粘贴文本和两个参考文件均有原样副本，见sources/。
原始路径：
- C:\\Users\\戴周阳\\Downloads\\510300_96因子研究库.html
- C:\\Users\\戴周阳\\Downloads\\510300_96因子与18策略.xlsx
- E:\\CodexData\\.codex\\attachments\\ea384a31-0725-481e-a74d-9b1eb3fc00cb\\pasted-text-1.txt
执行资产仅510300.SH/CASH_CNY，20万元主账户、2万元对照；当前研究合同另核对净年化10%和最大回撤10%。
本轮为历史研究模拟、已有原文的版本整理和实现修正，不改变已暂停PCF/IOPV任务或真实交易权限。当前研究授权原快照和更新进度见T16目录的source_evidence/current_mandate.json与authority_after.json。
"""
    reproduce = """在全新解压目录中执行以下只读命令，可用本地已安装依赖的Python替代python：

python scripts/verify_factor96_overnight_risk_saved_v1_0_2.py --root reports/research/510300_factor96_overnight_risk_overlay_v1_0_2
python scripts/verify_factor96_crowding_saved_v1_0_1.py --root reports/research/510300_factor96_crowding_overlay_v1_0_1
python scripts/verify_factor96_supply_version_ledger_v1.py --root reports/research/510300_factor96_supply_version_ledger_v1

依赖：Python、numpy、pandas、pyarrow；版本见REPRODUCTION_ENVIRONMENT.json。三个脚本默认只向终端输出回执，只有显式传入--receipt时才写回执，不更新研究账本。
预期状态依次为：
PASS_SAVED_OVERNIGHT_RISK_EXACT_BASELINE_STATE_ACCOUNT_RECOMPUTATION
PASS_SAVED_CROWDING_EXACT_BASELINE_STATE_ACCOUNT_RECOMPUTATION
PASS_SAVED_VERSION_ROLE_REFERENCE_AND_ASOF_RECOMPUTATION

复核分别覆盖T16的56账户61,208行、T06的56账户61,208行、T13的39原文32角色10引用3版本6时点查询。不会重建交易账户、拟合新模型、访问网络或重新抽取bootstrap样本。
不要运行研究入口的prepare/freeze/run来代替保存结果复核；它们有不可覆盖保护并涉及研究状态。报告脚本也不是只读入口。
原T16浮点脚本verify_factor96_overnight_risk_saved_v1.py作为发现实现问题的历史证据保留，不是当前通过命令。
"""
    exclusions = "范围,原因\n其余11项草案的原定义绩效,尚未完成固定检验\nT16在合格盈利基准上的部署验证,当前只是已核对账本的观察基准且结果失败\nT13完整事件去重和M06全部发行缴款日历,32角色及10引用不代表完整事件库\nT12及M06自由流通市值分母,来源门未齐\n无关工作区和Python缓存,不属于直接复算输入\n完整原始公告的顶层重复副本,原始完整集在历史原包内本轮39份直接原文已顶层包含\n否定版本作为正式绩效,历史保留但不再准入\n原始历史首版和独立前向验证,尚未建立\n外部审阅及真实执行,未进行且无订单授权\n"
    environment = {"python": platform.python_version(), "platform": platform.system(),
        "packages": {name: version(name) for name in ["numpy", "pandas", "pyarrow", "matplotlib", "pytest", "requests", "pypdfium2"]},
        "verification_dependencies": ["numpy", "pandas", "pyarrow"],
        "scope": "记录本次工具环境；无虚拟环境二进制或凭证打包。"}
    extras = {
        "00_README_FIRST.md": nav.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (OUT / "研究结论.md").read_bytes(),
        "USER_REQUEST.md": request.encode("utf-8"),
        "主期完整账户净值与回撤.png": (OUT / "主期完整账户净值与回撤.png").read_bytes(),
        "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"),
        "REPRODUCTION_ENVIRONMENT.json": json.dumps(environment, ensure_ascii=False, indent=2).encode("utf-8"),
        "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"),
    }
    assert set(files).isdisjoint(extras)
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extras)):
        size, checksum = file_identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = ZIP.with_suffix(".building.zip")
    print(f"写入{len(files) + len(extras)}个成员，原样保留上一轮T13完整ZIP。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(files):
            mode = zipfile.ZIP_STORED if files[name].suffix == ".zip" else zipfile.ZIP_DEFLATED
            archive.write(files[name], name, compress_type=mode)
        for name, data in sorted(extras.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(files) + len(extras)
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {r["path"] for r in rows} == set(archive.namelist()) - {"FILE_INDEX.csv"}
        for row in rows:
            with archive.open(row["path"]) as stream:
                assert stream_digest(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(ZIP)
    destination = ROOT / "outputs" / ("factor96_overnight_zip_verify_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ZIP) as archive:
        for name in archive.namelist():
            assert (destination / name).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    print("ZIP结构、索引和逐文件哈希通过，开始全新解压只读复算。", flush=True)
    verified, logs = {}, []
    for name, folder, script, expected in checks:
        command = [sys.executable, "-X", "utf8", str(destination / script), "--root", str(destination / folder.relative_to(ROOT))]
        print(f"开始只读核对{name}。", flush=True)
        run = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
        logs.append(f"来源：{name}\n{run.stdout}{run.stderr}")
        (OUT / "fresh_extraction_verification.log").write_text("\n".join(logs), encoding="utf-8")
        assert run.returncode == 0, f"{name}: {run.stderr[-3500:]}"
        verified[name] = json.loads(run.stdout.strip().splitlines()[-1])
        assert verified[name]["status"] == expected
        assert verified[name]["new_accounts"] == verified[name]["network_requests"] == 0
        print(f"{name}已通过全新解压保存结果复算。", flush=True)
    size, checksum = file_identity(ZIP)
    receipt = {"at": datetime.now().astimezone().isoformat(), "zip_path": str(ZIP), "bytes": size, "sha256": checksum,
        "members": len(files) + len(extras), "indexed_members": len(files) + len(extras) - 1,
        "crc_duplicate_index_size_sha256": "PASS", "frozen_input_coverage": "PASS",
        "fresh_extraction": str(destination), "saved_output_recomputations": verified,
        "prior_zip_bytes": previous["bytes"], "prior_zip_sha256": previous["sha256"],
        "formal_accounts": 320, "invalid_implementation_accounts": 152, "executed_accounts": 472,
        "external_review": "NOT_PERFORMED", "independent_forward_observations": 0,
        "orders_authorized": False, "goal_achieved": False, "goal_status": "active"}
    (OUT / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
