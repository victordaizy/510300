"""打包原样results_run1、两个实际输入及解释文件；不再次运行研究。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STUDY = Path(__file__).resolve().parent
# reports是跨盘目录链接；项目文档从调用脚本的逻辑项目路径定位。
PROJECT = Path(__file__).absolute().parents[3]
DELIVERY = PROJECT / "deliverables"
NAME = "510300_纯日线低频政策_results_run1_20261002"


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def main() -> None:
    archive = DELIVERY / (NAME + ".zip")
    temporary = DELIVERY / (NAME + ".building.zip")
    if archive.exists() or temporary.exists():
        raise FileExistsError("结果交付已存在，不覆盖")
    if not archive.resolve().is_relative_to(DELIVERY.resolve()) or not temporary.resolve().is_relative_to(DELIVERY.resolve()):
        raise ValueError("交付路径超出指定目录")
    authority = json.loads((STUDY / "authority_and_input_freeze.json").read_text(encoding="utf-8"))
    for name, field in (("run_offline.py", "script_sha256"), ("rules.json", "rules_sha256")):
        if sha((STUDY / "launch" / name).read_bytes()) != authority[field]:
            raise ValueError("收到的执行代码或参数已变化")
    original_index = json.loads((STUDY / "analysis/original_results_file_index.json").read_text(encoding="utf-8"))
    for row in original_index:
        body = (STUDY / "results_run1" / row["path"]).read_bytes()
        if sha(body) != row["sha256"] or len(body) != row["bytes"]:
            raise ValueError("原始结果自只读分析后改变：" + row["path"])
    if len(original_index) != 103:
        raise ValueError("原样结果文件数不是103")
    source = Path(authority["source_zip"])
    if sha(source.read_bytes()) != authority["source_zip_sha256"]:
        raise ValueError("原审阅ZIP自冻结后变化")
    members = {}
    for path in sorted(STUDY.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            members[path.relative_to(STUDY).as_posix()] = path.read_bytes()
    with zipfile.ZipFile(source) as prior:
        receipt = json.loads((STUDY / "results_run1/input_receipt.json").read_text(encoding="utf-8"))
        for name, key in (("readable/normalized_prices.csv", "prices_sha256"), ("readable/normalized_dividends.csv", "dividends_sha256")):
            body = prior.read(name)
            if sha(body) != receipt[key]:
                raise ValueError("两CSV与本轮实际输入不符")
            members[name] = body
    for name in ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md"):
        members["docs_snapshot/" + name] = (PROJECT / "docs" / name).read_bytes()
    readme = """# 已运行的纯日线低频政策：results_run1完整交付

本包含用户提供代码的第一次真实行情运行：3候选+3基准，2万/20万×BASE/STRESS，共24账户。2015-01-05—2026-08-14，每账户2823交易日；全部属于历史开发，旧结果未修改。

结论：12候选情景及24全情景均未同时达到夏普1.2与年化10%。三个候选四情景年化均低于三个基准，夏普均低于VOL10。本批结束、0候选晋升、0参数改动、0网络请求、0实际订单。

阅读顺序：

1. analysis/首轮研究结果与结论.md：共同口径结果、有限解释和停止条件。
2. results_run1/summary.csv、annual.csv、segments.csv：原样24份总表、全部288年度行及72固定分段行。
3. analysis/20万元六政策净值与回撤.png：全部6政策BASE/STRESS净值及回撤。
4. results_run1/accounts/：每个账户的account/orders/decisions/cycles四文件，共96文件；零闭合周期文件可能只有BOM，原样保留。
5. analysis/36项候选相对基准差异.csv、候选周期贡献集中.csv、持有天数与实际仓位.csv、固定组合与单策略差异.csv。
6. launch/：用户启动包全部9成员原样解压；readable/：本轮实际使用的两份CSV字节原样保存。
7. authority_and_input_freeze.json、self_test_local_receipt.json、local_execution_receipt.json、analysis/saved_result_review_receipt.json、FILE_INDEX.csv。
8. 01_给Pro的审阅提示词.md；docs_snapshot/为项目文档打包时快照，其他分支链接不递归包含。

results_run1全部103文件均为原程序原始输出，没有用中文别名覆盖summary.csv等。完整结果保留24账户、包括收益低及全现金年份。额外analysis目录仅从保存结果计算差异和作图，没有重跑策略。

本次执行沿用用户提供的标准库脚本，不需要pandas、pyarrow或新安装。净值图由本机已有Matplotlib绘制，不影响原程序运行。启动包合成测试及本机实际测试分别保存；前者不是本轮真实行情结果。

只读复算核对了24账户67752行、指标和净值恒等式，未新抽样或重跑账户。统计区间、多重历史选择纠偏、真实排队成交和独立验证未完成。份额及分红历史版本限制、开盘限价/滑点/容量代理、期末未强平合同继续保留。

原审阅ZIP47.17MB没有重复打包；仅纳入所用两CSV，降低交付体积。当前ZIP也包含程序要求的readable成员，但本批已运行完成，不建议无理由再次回放。后续新版本必须独立登记，不能覆盖results_run1。

本地压缩交付完成，尚未上传或获本轮Pro审核。结构与保存结果一致不等于策略有效。
"""
    prompt = """# 请审阅实际运行后的固定低频政策结果

请先读00_README_FIRST.md和analysis/首轮研究结果与结论.md，再核对results_run1三张总表及24套账户。这次已经在真实历史上跑完，不能再把启动包的synthetic_only回执误读成本轮没有运行。

用户明确允许现有两份CSV的有界历史开发，且入场/持有/退出可以一起研究。不要重做D-native、108日原因或48任务盘点，也不要恢复必须新字段/先有盈利入场的通用门槛。旧失败保留，新结果不称未见样本。此次原脚本和rules.json未改，所有参数固定，真实行情只运行一批24账户。

请重点回答：

1. 6政策的实现、成本、信号时钟、分红现金、容量/限价和期末估值是否与启动包合同一致？若有足以改变结论的缺陷，指出函数/行号、可复核反例，区分纠错与新政策。
2. 为什么TREND四情景年化低于3基准、夏普低于VOL10？34个周期和年度/分段损益表明哪些具体持有或退出问题，哪些解释只是猜测？不将最大周期贡献扣除冒充新的可执行账户。
3. REPAIR只有4周期、10持仓日、低于0.25%的平均账户暴露。3胜1负和低回撤能够支持什么，不能支持什么？请保留全现金年份夏普未定义和未成交记录。
4. MIX50与TREND日收益相关约0.991，趋势/修复持仓无重叠而修复只有10日。较小回撤和轻微夏普变化有多少可能来自风险暴露？当前未做暴露匹配，不将相关性或净值差额当互补的因果估计。
5. 24账户都未达到联合目标，本批设置应结束。若有值得继续的具体新假设，最多提出2项：写出由哪一笔/哪一类失败支持、唯一主要改动、固定基准、预先评价、停止条件，允许继续使用现有数据。不要直接提出均线/阈值网格或为了1.2无限搜索。

输出：本批总体裁决；必要实现问题表；三候选及三基准的经济解释；当前结论的边界；最多2项有具体诊断依据的下一实验；实际执行和未执行的核验。完整账户优先，禁止只摘录最好的期间或赢家。

本包未做置信区间/历史选择纠偏，也未获得真实成交或前瞻证据；不应把算术复算当这些验证。本轮数据止于2026-08-14，不据此生成当天交易建议。
"""
    members["00_README_FIRST.md"] = readme.encode("utf-8")
    members["01_给Pro的审阅提示词.md"] = prompt.encode("utf-8")
    indexes = [{"path": name, "bytes": len(body), "sha256": sha(body)} for name, body in sorted(members.items())]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
    writer.writeheader()
    writer.writerows(indexes)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name, body in sorted(members.items()):
            package.writestr(name, body)
        package.writestr("FILE_INDEX.csv", stream.getvalue().encode("utf-8-sig"))
    with zipfile.ZipFile(temporary) as package:
        names = package.namelist()
        if len(names) != len(set(names)) or package.testzip() is not None:
            raise ValueError("压缩包唯一性或CRC核对失败")
        if set(names) != set(members) | {"FILE_INDEX.csv"}:
            raise ValueError("压缩包文件覆盖不符")
        for row in indexes:
            body = package.read(row["path"])
            if len(body) != row["bytes"] or sha(body) != row["sha256"]:
                raise ValueError("压缩副本身份不符")
    os.replace(temporary, archive)
    receipt = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS_COMPLETE_RESULTS_PACKAGE", "zip_path": str(archive), "zip_bytes": archive.stat().st_size,
        "zip_sha256": sha(archive.read_bytes()), "members": len(names), "indexed_members": len(indexes),
        "original_result_files": len(original_index), "original_results_unchanged": True,
        "received_script_unchanged": True, "received_rules_unchanged": True,
        "source_input_csv_files": 2, "new_policy_account_runs_during_packaging": 0,
        "source_review_zip_unchanged": True, "new_random_draws": 0, "network_requests": 0,
        "uploaded": False, "external_review_performed": False, "independent_validation": False}
    (DELIVERY / (NAME + "_交付校验.json")).write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DELIVERY / (NAME + ".sha256")).write_text(receipt["zip_sha256"] + "  " + archive.name + "\n", encoding="utf-8")
    (DELIVERY / (NAME + "_给Pro的审阅提示词.md")).write_text(prompt, encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
