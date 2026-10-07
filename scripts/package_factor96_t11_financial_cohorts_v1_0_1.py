"""将披露篮子直接证据及完整上游财务包合成一个可复算档案。"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_t11_financial_cohorts_v1_0_1"
O02 = ROOT / "reports/research/510300_factor96_t11_o02_date_proxy_definition_v1"
FINANCIAL = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_2"
FINAL = ROOT / "deliverables/510300_96候选库_T11披露篮子与完整财报证据_GPT审阅_20260927.zip"
VERIFY = "scripts/verify_factor96_t11_financial_cohorts_v1_0_1.py"


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def identity(path):
    with path.open("rb") as stream:
        return path.stat().st_size, hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value, exclusive=True):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def report():
    assert not (OUT / "result_freeze.json").exists()
    result = read(OUT / "result.json")
    daily = pd.read_parquet(OUT / "daily_financial_cohorts.parquet")
    positive = daily.loc[daily.positive_financial_measurement]
    diagnostic = {"at": now(), "positive_days": len(positive),
        "single_company_positive_days": int(positive.nonfinancial_reports.eq(1).sum()),
        "positive_day_company_count_median": float(positive.nonfinancial_reports.median()),
        "positive_day_company_count_min": int(positive.nonfinancial_reports.min()),
        "positive_day_company_count_max": int(positive.nonfinancial_reports.max()),
        "interpretation": "多数正向披露日仅涉及单一成员，不能称沪深300整体盈利改善；不事后增加公司数门槛改变冻结定义。"}
    save(OUT / "cohort_concentration_diagnostic.json", diagnostic)
    positive.to_csv(OUT / "positive_financial_days.csv", index=False, encoding="utf-8-sig")
    annual = "\n".join(f"| {r['year']} | {r['disclosure_days']} | {r['known_cohort_days']} | {r['positive_financial_days']} |" for r in result["annual_measurement"])
    text = f"""只操作510300实现完整账户成本后夏普1.2的目标仍未实现。本轮将已修复的财务来源接入事先冻结的行业披露篮子：775个披露日中207日财务测量完整，23日满足正向条件。23日中15日只有一家新披露非金融公司，公司数中位为1，不能解释为全指数盈利改善。没有读取市场价格或未来收益，没有新增策略账户；这些日期不是可执行交易次数。

财务篮子算法与原V1保持字节一致，仅把来源接入由失败的V2改为已通过全新解压重算的V2.0.2。只纳入披露日当时的成员最新报告；同日留最新报告期，迟到旧报告不重开事件；排除金融行业，报告年龄0至200日。每家公司现金质量以前两日历年严格较早可用日、较早报告期、同季度同业的历史报告归一化，零波动或少于两条参考未知。行业内取公司中位数，再对行业取中位数，各行业等票。任一当前非金融公司缺项使整篮子未知，不能退回旧报告或把剩余公司冒充完整篮子。

正向财务条件为L02篮子>1、行业标准化现金质量>=0、原始现金质量同比变化>=0。共有9,335条公司测量及91,411条历史行业参考依赖。775个日期中，482日因财务或参考缺项保持NO_VIEW，58日因行业身份不明保持NO_VIEW，28日没有非金融新披露，207日可算。没有因为样本少而降低完整性要求，也没有增加事后公司数筛选。

| 年份 | 披露日 | 完整财务篮子日 | 正向财务条件日 |
|---|---:|---:|---:|
{annual}

2026年仅截至资料覆盖的8月14日，不能把0个正向财务日当作全年失败。上述数量没有年度配额，也不代表交易频率：尚未计算首轮反应、仓位重叠、风险预算或可成交性。

上游财报包完整嵌套于history/financial_source_v2_0_2.zip。其2,535个目标字段中1,364项与旧值在容差内一致、896项金额不同、275项未知；全部13项具体来源检查通过，范围外33,138项旧字段未全部重验。更早的合并口径错误、遗漏字段失败、三项残余数字错误及旧测量均保留。本包直接输入的财务表与该已核验来源版本绑定，不以旧错误数据填缺。

O02日期代理定义及实现也随包保存，但真实价格输入和测量尚未运行。它预先规定首个完整日反应、严格前序两日历年同类中位和下一开盘才可能执行，7项合成时序/分红/缺失测试通过。原始严格T11要求实际公开时刻；first_seen、标题状态和修订链仍不完整，因此后续只能明确称日期代理探索，不能称独立前向验证。20日持有、失效、风险、成本和完整账户合同还须在读取T11未来收益前固定。

此次财务条件结果可能反映少数个股公告，对510300的收益贡献仍待检验。不能根据这23日反向修改行业聚合、公司数、方向或时间窗来制造漂亮结果。累计账户仍为432个正式加152个否定实现，共584个；合格候选为空，独立前向观测0。外部审阅NOT_PERFORMED，无订单权限。下一步先完成已冻结O02条件测量，再按事先固定的账户合同检验剩余收益。
"""
    (OUT / "研究结论.md").write_text(text, encoding="utf-8")
    prompt = """请评审本包T11披露日财务篮子测量及上游来源边界。目标仍为510300与现金、完整成本后夏普至少1.2、年化至少10%、目标回撤不超过10%；本包没有新账户或收益检验，目标未实现。

核对冻结V1算法与V1.0.1接入是否字节一致；检查最新报告前沿、行业与成员、两年严格历史参考、完整篮子缺失处理、两层中位和23日正财务条件。重点批评23日中15日仅一家公司的ETF传导问题，不能把个股改善说成全指数改善。不要事后加公司数门槛救援。

直接财务输入绑定完整嵌套的V2.0.2来源包，其中全部旧失败与未知保留。源修复和机械重算不证明全部财务字段正确或历史首次公开时钟。O02代码仅7项合成测试通过，实际测量未运行；原严格T11与日期代理探索应区分。

请给出最小必要的下一步、准入和停止条件：首轮价格反应、20日账户合同、风险成本、对照和独立验证。分别考虑信息未充分吸收与市场不认可两种机制。不得挑盈利年份、改方向或更改已失败定义。外部审阅尚未进行，本包不授权订单。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    program = ROOT / "reports/research/510300_factor96_program_v1"
    current = read(program / "status.json")
    result_path = (OUT / "result.json").relative_to(ROOT).as_posix()
    current.update({"at": now(), "latest_round": "510300_FACTOR96_T11_FINANCIAL_COHORTS_V1_0_1", "latest_result": result_path,
        "latest_progress_receipt": result_path, "latest_continuation_classification": "PROGRESS_FINANCIAL_COHORTS_23_POSITIVE_DAYS_NO_RETURNS",
        "latest_t11_financial_cohorts": result_path, "t11_positive_financial_days": 23,
        "t11_single_company_positive_days": 15, "source_collection_status": "COMPLETE_2117_TARGETS_2090_SAME_HASH_PDFS_27_UNKNOWNS",
        "last_source_result": "披露篮子775日中207日完整、23日财务条件正向，其中15日仅单公司；O02及T11收益账户未运行。"})
    assert current["cumulative_executed_account_scenarios"] == 584 and not current["qualified_candidates"]
    save(program / "status.json", current, exclusive=False)
    for filename, ids in [("factor_progress.json", {"L02", "L04"}), ("strategy_progress.json", {"T11"})]:
        items = read(program / filename)
        for item in items:
            if item["id"] in ids:
                item.update({"current_status": "FINANCIAL_COHORT_MEASURED_O02_AND_T11_RETURNS_NOT_RUN",
                    "current_evidence": current["last_source_result"], "current_evidence_path": result_path, "individual_performance": None})
        save(program / filename, items, exclusive=False)
        for csv_name in [filename.replace(".json", ".csv"), "96因子当前进度.csv" if filename.startswith("factor") else "18策略当前进度.csv"]:
            pd.DataFrame(items).to_csv(program / csv_name, index=False, encoding="utf-8-sig")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update({"current_round": current["latest_round"], "latest_progress_receipt": result_path,
        "current_protocol": (OUT / "protocol.json").relative_to(ROOT).as_posix(),
        "latest_continuation_report": (OUT / "研究结论.md").relative_to(ROOT).as_posix(),
        "latest_continuation_classification": current["latest_continuation_classification"],
        "research_execution_state": "FINANCIAL_COHORTS_COMPLETE_O02_REAL_MEASUREMENT_NOT_COMPUTED",
        "last_source_result": current["last_source_result"], "latest_financial_delivery": (FINANCIAL / "delivery_receipt.json").relative_to(ROOT).as_posix()})
    save(mandate_path, mandate, exclusive=False)
    (OUT / "program_after").mkdir()
    for path in program.iterdir():
        if path.is_file(): shutil.copy2(path, OUT / "program_after" / path.name)
    shutil.copy2(mandate_path, OUT / "program_after/mandate.json")
    for path in [Path(__file__), ROOT / VERIFY]:
        shutil.copy2(path, OUT / "code" / path.name)
    paths = [p for p in sorted(OUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    save(OUT / "result_freeze.json", {"at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": identity(p)[1]} for p in paths]})
    print("披露篮子结果及单公司集中度已封存，原算法与未知状态保持。", flush=True)


def package():
    assert not FINAL.exists() and not (OUT / "delivery_receipt.json").exists()
    prior_receipt = read(FINANCIAL / "delivery_receipt.json")
    prior = Path(prior_receipt["zip_path"])
    assert identity(prior) == (prior_receipt["bytes"], prior_receipt["sha256"])
    files = {}
    for base in [OUT, O02]:
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                files[p.relative_to(ROOT).as_posix()] = p
    for name in [VERIFY, "scripts/package_factor96_t11_financial_cohorts_v1_0_1.py",
                 "scripts/run_factor96_t11_financial_cohorts_v1_0_1.py", "research/factor96_t11_financial_cohorts_v1.py",
                 "research/factor96_t11_o02_date_proxy_v1.py", "tests/test_factor96_t11_financial_cohorts_v1.py",
                 "tests/test_factor96_t11_o02_date_proxy_v1.py"]:
        files[name] = ROOT / name
    files["history/financial_source_v2_0_2.zip"] = prior
    files["history/financial_source_delivery_receipt.json"] = FINANCIAL / "delivery_receipt.json"
    with zipfile.ZipFile(prior) as archive:
        request = archive.read("USER_REQUEST.md")
        environment = archive.read("REPRODUCTION_ENVIRONMENT.json")
    nav = """目标尚未实现。本包主结果是775个披露日中207日完整、23日财务条件正向，其中15日仅单公司。没有T11收益或新账户。

阅读顺序：02_研究结论.md、01_GPT_REVIEW_PROMPT.txt、本轮protocol/definition_freeze/result/result_freeze、三个完整测量表、positive_financial_days.csv和单公司集中度诊断。original_definition保留事先冻结的算法；inputs直接财务表绑定已通过核验的来源版本。

history/financial_source_v2_0_2.zip完整保留上游原文、全部旧反证、规范时钟与更正候选包及财务重算；根索引记录嵌套ZIP的精确身份。原文巨大导致本包较大，不表示外部平台能够接受上传。本包未上传或外部审阅。

reports/research/510300_factor96_t11_o02_date_proxy_definition_v1只包含下一步日期代理定义与7项测试通过的代码。实际市场输入未封存、O02未计算，完整账户尚未运行。不能把定义或测试当实际收益。

在解压根目录运行python scripts/verify_factor96_t11_financial_cohorts_v1_0_1.py --root reports/research/510300_factor96_t11_financial_cohorts_v1_0_1，可只读重算本轮直接测量。该命令核对嵌套来源包身份及输入绑定，不重复上游全部PDF解析，不联网、不创建账户。FILE_INDEX.csv是除自身外全部成员的权威索引。
"""
    extras = {"00_README_FIRST.md": nav.encode("utf-8"), "USER_REQUEST.md": request,
        "01_GPT_REVIEW_PROMPT.txt": (OUT / "01_GPT_REVIEW_PROMPT.txt").read_bytes(),
        "02_研究结论.md": (OUT / "研究结论.md").read_bytes(), "REPRODUCTION_ENVIRONMENT.json": environment,
        "03_EXCLUSIONS.csv": "范围,原因\nO02真实价格测量,尚未实施仅冻结定义与代码\nT11未来收益和账户,合同尚未固定与执行\n独立前向验证,没有新前向观测\n全旧字段人工核查,上游33138项范围外字段仍沿用旧档案\n外部上传审阅与订单,未执行\n".encode("utf-8-sig")}
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(set(files) | set(extras)):
        size, checksum = identity(files[name]) if name in files else (len(extras[name]), hashlib.sha256(extras[name]).hexdigest())
        writer.writerow({"path": name, "bytes": size, "sha256": checksum})
    extras["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    required = sum(p.stat().st_size for p in files.values()) + 1024**3
    assert shutil.disk_usage(FINAL.parent).free > required
    extraction = Path("E:/CodexData/verification/factor96_t11_financial_cohorts_v1_0_1") / datetime.now().strftime("%Y%m%d_%H%M%S")
    extraction.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(extraction).free > required
    building = FINAL.with_suffix(".building.zip")
    print("合并披露篮子直接证据与完整上游财报包，保留全部已知限制。", flush=True)
    with zipfile.ZipFile(building, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            method = zipfile.ZIP_STORED if path.suffix in {".zip", ".parquet", ".pdf"} else zipfile.ZIP_DEFLATED
            archive.write(path, name, compress_type=method)
        for name, value in sorted(extras.items()): archive.writestr(name, value)
    with zipfile.ZipFile(building) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        assert {r["path"] for r in rows} == set(names) - {"FILE_INDEX.csv"}
        for row in rows:
            assert archive.getinfo(row["path"]).file_size == int(row["bytes"])
            with archive.open(row["path"]) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
        assert all((extraction / name).resolve().is_relative_to(extraction.resolve()) for name in names)
        archive.extractall(extraction)
    building.replace(FINAL)
    print("主包结构与原文包身份核对通过，开始披露篮子只读重算。", flush=True)
    command = [sys.executable, "-X", "utf8", str(extraction / VERIFY), "--root", str(extraction / OUT.relative_to(ROOT))]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
    (OUT / "fresh_extraction_verification.log").write_text(process.stdout + process.stderr, encoding="utf-8")
    assert process.returncode == 0, process.stdout + process.stderr
    verified = json.loads(process.stdout.strip().splitlines()[-1])
    assert verified["status"] == "PASS_SAVED_T11_FINANCIAL_COHORTS_AND_PAST_ONLY_REFERENCES"
    size, checksum = identity(FINAL)
    receipt = {"at": now(), "zip_path": str(FINAL), "bytes": size, "sha256": checksum,
        "members": len(names), "indexed_members": len(rows), "crc_duplicate_index_size_sha256": "PASS",
        "fresh_extraction": str(extraction), "saved_output_recomputation": verified,
        "upstream_financial_zip_bytes": prior_receipt["bytes"], "upstream_financial_zip_sha256": prior_receipt["sha256"],
        "new_accounts": 0, "T11": "NOT_RUN", "O02": "NOT_COMPUTED", "goal_achieved": False,
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["report", "package"])
    options = parser.parse_args()
    report() if options.action == "report" else package()
