"""归档固定日更检验和全部112账本，保留历史包并全新解压验证。"""
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
OUT = ROOT/"reports/research/510300_factor96_daily_state_shrink_v1"
FIRST = ROOT/"reports/research/510300_factor96_mechanism_batch_v1"
PREVIOUS = ROOT/"reports/research/510300_factor96_rights_calendar_sources_v1"
ZIP = ROOT/"deliverables/510300_96候选库_日更成熟校准固定检验_GPT审阅_20260927.zip"
PRIOR = (1841355190, "2296e5269a445ed6f216e54c10bfe753b9f4253fc70814e9b3a80377f727dca0")
EXPECTED = "PASS_SAVED_DAILY_SHRINK_CLOCK_COEFFICIENTS_AND_ACCOUNTS"
VERIFY = "scripts/verify_factor96_daily_state_shrink_v1.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream_identity(stream):
    size, checksum = 0, hashlib.sha256()
    while chunk := stream.read(1024*1024):
        size += len(chunk)
        checksum.update(chunk)
    return size, checksum.hexdigest()


def identity(path):
    with path.open("rb") as stream:
        return stream_identity(stream)


def main():
    assert not ZIP.exists() and not (OUT/"delivery_receipt.json").exists()
    result, status = read(OUT/"result.json"), read(OUT/"round_status.json")
    assert result["new_accounts"] == 112 and not result["goal_achieved"]
    assert status["cumulative_admitted_account_scenarios"] == 432 and status["cumulative_executed_account_scenarios"] == 584
    assert read(OUT/"saved_verification_receipt.json")["status"] == EXPECTED
    previous = read(PREVIOUS/"delivery_receipt.json")
    prior_zip = Path(previous["zip_path"])
    assert (previous["bytes"], previous["sha256"]) == PRIOR and identity(prior_zip) == PRIOR
    files = {}
    for path in OUT.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.name not in ["delivery_receipt.json", "fresh_extraction_verification.log"]:
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ["research/factor96_daily_state_shrink_v1.py", "research/factor96_margin_repair_v1.py",
                 "research/factor96_library_intake_v1.py", "tests/test_factor96_daily_state_shrink_v1.py",
                 VERIFY, "scripts/report_factor96_daily_state_shrink_v1.py", "scripts/package_factor96_daily_state_shrink_v1.py",
                 "reports/research/510300_factor96_crowding_overlay_v1_0_1/code/account_engine.py"]:
        files[name] = ROOT/name
    for name in ["factor_registry.json", "strategy_registry.json", "input_workbook_tables.json", "intake_receipt.json"]:
        files["reference_library/"+name] = FIRST/name
    for path in (FIRST/"sources").iterdir():
        if path.is_file():
            files["sources/"+path.name] = path
    files["history/"+prior_zip.name] = prior_zip
    files["history/prior_delivery_receipt.json"] = PREVIOUS/"delivery_receipt.json"
    for item in read(OUT/"freeze.json")["files"]:
        path = OUT/item["path"]
        assert path.relative_to(ROOT).as_posix() in files and identity(path)[1] == item["sha256"]
    with zipfile.ZipFile(prior_zip) as archive:
        request = archive.read("USER_REQUEST.md").decode("utf-8")
    request += "\n本轮时钟选择：附件T18季度更新只作参考；现行用户已明确两年训练、每天滚动更新，原始授权回执已纳入source_evidence/daily_training_authority.json。原季度草案保持NOT_RUN，本轮是单独登记的日更变体。\n"
    nav = f"""# 510300：三家族日更成熟校准固定检验

**目标尚未实现。** 本轮状态：{result['status']}。112新账户完整保留，不扫描参数。累计432正式账户情景+152否定实现情景=584；原库7个固定问题加1个本轮授权日更变体，没有独立验证合格策略。原T18季度草案仍NOT_RUN。

阅读顺序：
1. 02_研究结论.md：主方案、全部对照、资金与时钟比较、增量区间及边界。
2. reports/research/510300_factor96_daily_state_shrink_v1/protocol.json、freeze.json、run_started.json：参数、先冻结后计算和固定112账户范围。
3. 同目录daily_forecasts_lag1/2.parquet、outer_models_lag1/2.json、inner_models_lag1/2.json：逐日三家族预测、成熟校准、当前两年窗口及五档输出。
4. accounts/下112套账本、决策和周期；metrics.csv、annual_metrics.csv、paired_increment.json及bootstrap_indices.npz：全部保存结果，不择优删改。
5. inputs/、source_evidence/和code/：价格、分红、融资、点时成员及总回报、风险标签、冻结源码、现行用户日更授权。旧配置仅作定义查重材料，不作为本轮计算依赖或重新执行入口。
6. history/：上一轮1,841,355,190字节完整发行检索与配股时序包原样保留，SHA-256 2296e5269a445ed6f216e54c10bfe753b9f4253fc70814e9b3a80377f727dca0；内含更早原包及失败证据。

用户原始HTML、Excel和粘贴文本在sources/，登记表在reference_library/。附件建议不是额外执行授权。

本轮只读核验的直接依赖均已包含，不必展开历史嵌套包。命令见REPRODUCE_SAVED_RESULTS.txt。20项冻结前测试原回执已保留；新目录核验重新核对保存系数充分统计量及112账本，不新增实验拟合、随机抽样、账户或网络请求。当前来源派生矩阵按冻结来源核对，更早原始来源完整证据沿历史ZIP链保存。

FILE_INDEX.csv覆盖除自身外全部成员。包外delivery_receipt.json包含最终ZIP身份和新目录核验，以避免自引用。结构和数值一致不代表策略有效。external_review=NOT_PERFORMED，独立前向0，当前市场NO_VIEW，研究范围仍510300.SH与CASH_CNY；无订单权限，原PCF／IOPV任务暂停。仅本地交付，没有上传。
"""
    reproduce = "在全新解压目录执行只读核验：\n\npython "+VERIFY+" --root "+OUT.relative_to(ROOT).as_posix()+"\n\n依赖numpy、pandas、pyarrow。系数充分统计量复算及112保存账本重放可能需要数分钟；不重跑prepare/freeze/run，不会追加研究账户或访问网络。此核验不能证明独立预测力。\n"
    exclusions = "范围,原因\nT18原季度草案,与现行用户每天训练要求不同且未执行\n原库其他未运行策略,对应数据门及状态保留\n历史权重与自由流通全序列,本轮没有取得准入序列\n20万元和2万元外其他资金规模,未注册不追加搜索\n修改窗口阈值或拼接有利阶段,固定失败不救援\n旧查重配置引用的非本轮输入平铺,仅作定义比较且不是当前复算依赖\n首次历史发布版本及独立前向,未建立\n外部GPT审阅与上传,未执行\n真实交易及恢复暂停任务,无此授权\n"
    extras = {"00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request.encode("utf-8"),
        "01_GPT_REVIEW_PROMPT.txt": (OUT/"01_GPT_REVIEW_PROMPT.txt").read_bytes(), "02_研究结论.md": (OUT/"研究结论.md").read_bytes(),
        "主期20万元压力净值与回撤.png": (OUT/"主期20万元压力净值与回撤.png").read_bytes(),
        "REPRODUCE_SAVED_RESULTS.txt": reproduce.encode("utf-8"), "03_EXCLUSIONS.csv": exclusions.encode("utf-8-sig"),
        "REPRODUCTION_ENVIRONMENT.json": json.dumps({"python": platform.python_version(), "platform": platform.system(),
            "packages": {p: version(p) for p in ["numpy", "pandas", "pyarrow", "matplotlib", "pytest"]}}, ensure_ascii=False, indent=2).encode("utf-8")}
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    assert set(files).isdisjoint(extras)
    for name in sorted(set(files) | set(extras)):
        size, checksum = identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    building = ZIP.with_suffix(".building.zip")
    print(f"开始写入{len(files)+len(extras)}个成员。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if path.suffix == ".zip" else zipfile.ZIP_DEFLATED)
        for name, content in sorted(extras.items()):
            archive.writestr(name, content)
    print("写入完成，核对结构及逐文件索引。", flush=True)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist())) == len(files)+len(extras)
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert len(rows) == len(archive.namelist())-1
        assert {r["path"] for r in rows} == set(archive.namelist())-{"FILE_INDEX.csv"}
        for row in rows:
            with archive.open(row["path"]) as stream:
                assert stream_identity(stream) == (int(row["bytes"]), row["sha256"]), row["path"]
    building.replace(ZIP)
    destination = ROOT/"outputs"/("factor96_daily_shrink_zip_verify_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ZIP) as archive:
        assert all((destination/name).resolve().is_relative_to(destination.resolve()) for name in archive.namelist())
        archive.extractall(destination)
    print("结构及索引通过，开始全新解压目录保存结果复算。", flush=True)
    lines = []
    command = [sys.executable, "-X", "utf8", str(destination/VERIFY), "--root", str(destination/OUT.relative_to(ROOT))]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW) as process:
        with (OUT/"fresh_extraction_verification.log").open("x", encoding="utf-8") as log:
            for line in process.stdout:
                lines.append(line)
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    assert returncode == 0, "".join(lines)[-5000:]
    verified = json.loads("".join(lines).strip().splitlines()[-1])
    assert verified["status"] == EXPECTED and verified["new_accounts"] == verified["network_requests"] == 0
    size, checksum = identity(ZIP)
    receipt = {"at": datetime.now().astimezone().isoformat(), "zip_path": str(ZIP), "bytes": size, "sha256": checksum,
        "members": len(files)+len(extras), "indexed_members": len(files)+len(extras)-1,
        "crc_duplicate_index_size_sha256": "PASS", "frozen_input_coverage": "PASS", "fresh_extraction": str(destination),
        "saved_output_recomputation": verified, "prior_zip_bytes": PRIOR[0], "prior_zip_sha256": PRIOR[1],
        "research_accounts_this_round": 112, "new_accounts_during_packaging": 0,
        "formal_accounts": 432, "invalid_implementation_accounts": 152, "executed_accounts": 584,
        "goal_achieved": False, "goal_status": "active", "external_review": "NOT_PERFORMED", "orders_authorized": False}
    (OUT/"delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
