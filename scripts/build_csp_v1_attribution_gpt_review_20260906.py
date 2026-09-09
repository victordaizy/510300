"""将本轮归因和上一轮完整上下文合为一个可导航GPT审阅包。"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/research/510300_csp_v1_static_vs_timing_attribution_v1"
STEM = "510300_CSP_V1_STATIC_VS_TIMING_ATTRIBUTION_V1_GPT_REVIEW_20260906"
PRIOR = "deliverables/510300_CONDITIONAL_SCORE_POLICY_V1_GPT_REVIEW_20260905.zip"
PRIOR_SHA = "b358b34c29d4ec3d034ffb5112702441283e754ddc73de3d80c852cc04fbf43e"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def encoded(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    created = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    destination = ROOT / "deliverables"
    archive_path = destination / f"{STEM}.zip"
    require(not archive_path.exists(), "同名审阅包已存在，禁止覆盖")
    output = ROOT / REPORT
    manifest = json.loads((output / "freeze_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "execution_receipt.json").read_text(encoding="utf-8"))
    require(receipt["status"] == "COMPLETED" and result["state"] == "COMPLETED_DIAGNOSTIC_ONLY_NO_PROMOTION", "归因尚未完成")
    require(sha((output / "result.json").read_bytes()) == receipt["result"]["sha256"], "完成结果身份不符")
    payload, roles = {}, {}

    def add(name: str, content: bytes, role: str) -> None:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "包内路径非法")
        if name in payload:
            require(payload[name] == content, f"同一路径出现不同内容：{name}")
            if role not in roles[name]:
                roles[name] += "；" + role
        else:
            payload[name], roles[name] = content, role

    def note(name: str, content: str) -> None:
        add(name, (content.rstrip() + "\n").encode("utf-8"), "本次归因阅读与审阅材料")

    def project(relative: str, role: str) -> None:
        path = ROOT / relative
        require(path.is_file(), f"必需交付文件不存在：{relative}")
        add("project/" + relative, path.read_bytes(), role)

    prior_bytes = (ROOT / PRIOR).read_bytes()
    require(sha(prior_bytes) == PRIOR_SHA, "上一轮完整审阅包身份改变")
    with zipfile.ZipFile(io.BytesIO(prior_bytes)) as prior:
        require(prior.testzip() is None, "上一轮ZIP结构错误")
        require(len(prior.namelist()) == len(set(prior.namelist())), "上一轮ZIP重名")
        old_index = list(csv.DictReader(io.StringIO(prior.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(row["path"] for row in old_index) == set(prior.namelist()) - {"FILE_INDEX.csv"}, "原ZIP索引范围不符")
        for item in old_index:
            content = prior.read(item["path"])
            require(len(content) == int(item["bytes"]) and sha(content) == item["sha256"], "原ZIP索引身份不符")
        for name in prior.namelist():
            content = prior.read(name)
            if name == "project/RESEARCH_STATUS.md":
                add("prior_review/RESEARCH_STATUS_AT_PRIOR_DELIVERY.md", content, "上一轮交付时状态快照")
            elif name == "review_tools/recompute_saved_metrics.py":
                add("review_tools/recompute_parent_saved_metrics.py", content, "原八账户只读复算工具，--zip指向本包亦可使用")
            elif name.startswith("project/"):
                add(name, content, "上一轮完整项目内容：原V1全部239份结果、实现、直接输入及D20/R6背景")
            else:
                add("prior_review/" + name, content, "上一轮交付材料原样归档；内部文字和索引指上一轮包的日期与路径")
    for relative, item in manifest["files"].items():
        content = (ROOT / relative).read_bytes()
        require(sha(content) == item["sha256"] and len(content) == item["bytes"], f"冻结依赖变化：{relative}")
        add("project/" + relative, content, "本轮运行前冻结的实施或原研究依赖")
    new_files = sorted(path for path in output.rglob("*") if path.is_file())
    for path in new_files:
        project(path.relative_to(ROOT).as_posix(), "本轮成果目录完整收录")
    for relative in ["RESEARCH_STATUS.md", "docs/RESEARCH_REVIEW_ZIP_DELIVERY_RULE.md",
                     "scripts/verify_csp_v1_attribution_saved_delivery.py", Path(__file__).relative_to(ROOT).as_posix()]:
        project(relative, "本轮交付实现、项目状态或持续ZIP交付要求")
    add("review_tools/verify_saved_diagnostic.py", (ROOT / "scripts/verify_csp_v1_attribution_saved_delivery.py").read_bytes(), "只读复核本包12份真实规则账本和4条理想路径")
    note("00_README_FIRST.md", f'''# 510300静态底仓与状态择时归因：完整GPT审阅包

本包于{created}生成。本轮状态 **COMPLETED_DIAGNOSTIC_ONLY_NO_PROMOTION**，原CSP V1保持拒绝，净夏普目标1.2不变。

结果：C0与C1在基础和压力成本下逐日相同，只有首次建仓。FULL相对C1基础净年化差+0.4538个百分点，95%区间[-0.2162,1.1690]；同波动理想对照差+0.2088个百分点，区间[-0.4397,0.8470]。本次未确认稳定状态择时优势。

## 先读这些

1. `01_GPT_REVIEW_PROMPT.md`：本轮复核重点与下一步方向要求。
2. `02_封卷结论与审阅重点.md`：结论、限制和研究方向决定。
3. `03_完整归因报告.md`：四新增账户、原参照、四理想路径及全部区间。
4. `04_本轮冻结协议.md`：事后诊断身份、补充定义、预算与停止规则。
5. `05_收到的GPT审阅正文.txt`：用户提供的完整文本；没有收到其中sandbox链接的外部ZIP/证据。
6. `06_证据与复核地图.md`：原始数据、源码、账本、统计、回执的具体位置。

`project/`同时包含本轮全部成果与上一轮完整研究上下文，不需要另找上一轮ZIP。`prior_review/`是上一轮阅读材料快照，内含“尚未收到审阅”等当时状态，不代表本轮状态；其中旧索引仍指原包，不是本包索引。本包以根目录FILE_INDEX.csv为准。

本轮仅新增4条规则模拟账户、4条理想路径、1次20日×2000次联合区块抽样；新拟合/网格/优化器/下载均0。原结果、系数、冻结数据未改写。Paper/Shadow与实盘不在授权范围，真实持仓未知、仓位影响0。
''')
    prompt = '''# 给GPT的本轮独立审阅任务

请审阅这次已完成的独立归因，并据证据修订后续研究方向。先读00、02、03、04、05、06，再核对project中的原始账本、代码、冻结清单和统计。不要只阅读摘要，也不要默认上一轮审阅正文中所有复算声明已经由本轮重新验证。

本轮身份：510300_CSP_V1_STATIC_VS_TIMING_ATTRIBUTION_V1，DIAGNOSTIC_ONLY。原V1拒绝、目标净夏普1.2、已关闭分支均不变。新拟合0，真实规则模拟账户4，理想统计路径4，20日/2000次/20260906联合抽样1次。完成后封卷、不晋升对照、不自动启动优化。

请重点判断：

1. 同截距C0、恒评分C1是否忠实实现；全精度截距、起始锚点、首次部分成交、分红留现金、NO_VIEW余单、收盘冻结/次开盘/T+1/整手/费用是否一致。四个新账户是否完整覆盖1604日。C0/C1逐日完全相同和机械增量0是否成立。
2. FULL−C1的正点估计是否足以支持任何择时结论。请分别读净年化、夏普、效用和终值差区间；当前主要区间跨零。不要把“未确认稳定增量”写成“所有择时或修复机制无效”。
3. 四条理想路径是本地运行前补充冻结的定义：用原BUY_HOLD净日收益固定比例缩放，分别匹配FULL平均暴露/波动。请评价参照、全历史匹配、隐含每日调整、费用比例化和bootstrap固定k这些限制，说明它们到底能排除什么、不能排除什么。它们不是满足真实交易合同的策略。
4. 人民币分解是逐日会计恒等式，FULL−C1仍包含初始份额、暴露与时序变化，不是已识别的纯预测alpha。43.79%的净利润分量不能变成未来可复制alpha占比。
5. 新开盘合法性判断是否只使用开盘时可知字段，离线成交量质检是否与开盘执行分开；本样本旧新判断差异0是否充分支持沿用原FULL保存账本。不要把日线模拟写成每笔真实开盘成交的证明。
6. 保留原FULL−SIMPLE有限正向统计：原净年化差区间跨零，原夏普差和效用差区间不跨零。外部审阅者报告34项通过+1缺依赖，原本地回执35项通过，本轮新增合成测试9项通过；这是不同环境/轮次。

请输出具体的P0/P1/P2问题、证据文件和最小修正，明确是否足以推翻本轮结论。若仅是解释边界，请不要要求改原冻结系数或补额外账户来追求正结果。可用review_tools只读复算保存账本；--check-saved-samples仅复算已有索引，不生成新随机样本。不要运行原训练入口或重放新历史账户。

最后给出“继续/暂停/停止”方向判断。若现有证据仍没有合格达标策略候选，请直接写“没有合格优先候选或备选”。若建议独立新方向，必须说明新的经济机制或信息证据、实际可得数据、可用时钟、最小可证伪实验、有限预算和硬停止条件，区分尚未验证的想法与可立刻实施的候选。不得把同日线权重/窗口/带宽/种子变体或已关闭期权、PIT、IF等分支包装成新授权。建议本身不自动启动研究，不涉及真实持仓处置。
'''
    note("01_GPT_REVIEW_PROMPT.md", prompt)
    add("02_封卷结论与审阅重点.md", (output / "封卷结论与审阅重点.md").read_bytes(), "本轮简明结论")
    add("03_完整归因报告.md", (output / "研究报告.md").read_bytes(), "本轮完整报告")
    add("04_本轮冻结协议.md", (ROOT / "docs/510300_CSP_V1_STATIC_VS_TIMING_ATTRIBUTION_V1_PROTOCOL.md").read_bytes(), "本轮运行前冻结协议")
    add("05_收到的GPT审阅正文.txt", (ROOT / "docs/510300_CSP_V1_EXTERNAL_REVIEW_USER_TEXT_20260906.txt").read_bytes(), "用户提供的外部审阅正文")
    note("06_证据与复核地图.md", f'''# 证据地图与复核方法

## 当前成果

本轮根目录：`project/{REPORT}/`，完整收录{len(new_files)}个文件。

|内容|位置|
|---|---|
|研究身份和预算|project/config/510300_csp_v1_static_vs_timing_attribution_v1.json|
|冻结协议和源码|本轮freeze_manifest.json与frozen_sources；project/research/csp_v1_static_vs_timing_attribution_v1.py|
|一次性入口|project/scripts/run_510300_csp_v1_static_vs_timing_attribution_v1.py；已有run_claim，禁止重跑|
|新四账户全部日账/决策/成交|本轮accounts/*.csv；日账另有Parquet镜像|
|四条理想路径及匹配|本轮ideal/*.csv、ideal_matching.json|
|区块统计和全部抽样|本轮statistics.json、bootstrap_indices.npz、bootstrap_samples.npz|
|人民币归因恒等式|本轮wealth_attribution_BASE.csv与STRESS.csv|
|解析目标上下界|本轮analytic_bounds.json|
|输入与会计门|本轮data_contract.json、parent_account_checks.json、new_account_checks.json|
|验证和预算回执|本轮pre_freeze_validation.json、execution_receipt.json、delivery_verification.json|
|审阅解释差异|本轮review_reconciliation.json|

## 原研究上下文

原研究完整239份产物：`project/reports/research/510300_conditional_score_policy_v1/`。原8条主账户在evaluation，54次内层拟合、108条验证路径及两次最终拟合结果在development；本轮没有重新训练或复放这些验证账户。

原协议/候选原文/配置/全部实现/单元测试，以及原价格、分红、日历、来源凭证已随project路径收录。上一轮D20完整45份成果及冻结依赖、R6协议与拒绝证据、14份分红公告原件也保留，范围与原审阅包相同。原包SHA-256为 `{PRIOR_SHA}`，原包本身没有改动，本包将其内容展开收录而不嵌套ZIP。

## 只读复算

`review_tools/verify_saved_diagnostic.py`可只用Python标准库核对12份保存的规则账户及4条理想路径。传入--zip和本包绝对路径即可；或解压后从该脚本运行，默认根目录为脚本上一级。加--check-saved-samples时需要NumPy，按已经保存的索引复算2,000组指标和区间，没有随机数发生器、新账户或训练调用。

`review_tools/recompute_parent_saved_metrics.py`是上一轮原八账户指标复核器，--zip也可直接读取本包。

完整源代码测试环境依赖见07_环境记录.json。外部环境若没有Parquet依赖，可先使用逐日CSV及标准库只读工具，不能把依赖缺失测试记为通过。

FILE_INDEX.csv覆盖其自身之外全部成员，逐项列出字节数、SHA-256及作用。打包检查仅覆盖ZIP结构、重名、索引、哈希和声明范围。自动上传、外部GPT的新一轮审阅均未发生。
''')
    packages = {}
    for name in ["numpy", "pandas", "scipy", "pyarrow", "matplotlib", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    add("07_环境记录.json", encoded({"created_at": created, "python": platform.python_version(),
                                      "platform": platform.platform(), "packages": packages,
                                      "new_synthetic_tests": 9, "original_local_tests": 35,
                                      "external_environment_reported_tests": "34通过+1缺依赖，外部证据未收到"}), "运行与复核环境")
    note("08_上传GPT时粘贴这段.txt", "请完整阅读压缩包中的00_README_FIRST.md和01_GPT_REVIEW_PROMPT.md，独立复核本轮静态底仓—机械再平衡—状态择时归因，并修订下一步研究方向。不要只看摘要。请重点区分正向点估计与跨零区间、理想风险匹配与真实可交易账户、会计分解与预测alpha。如果没有合格新候选，请明确写没有；不要再用同日线参数变体营救原V1，也不要自动扩展已关闭分支。")
    add("09_PACKAGE_METADATA.json", encoded({"created_at": created, "study_id": result["study_id"],
         "state": result["state"], "budget_used": result["budget_used"], "frozen_files_verified": len(manifest["files"]),
         "current_result_tree_files": len(new_files), "prior_archive_indexed_members_verified": len(old_index),
         "prior_archive_sha256": PRIOR_SHA, "source_originals_unchanged": True,
         "external_review_material_received": "用户粘贴正文；外部ZIP/证据未收到",
         "new_external_gpt_review_performed": False, "automatically_uploaded": False,
         "position_impact": 0}), "本包范围和状态元数据")
    index_stream = io.StringIO(newline="")
    writer = csv.DictWriter(index_stream, fieldnames=["path", "role", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(payload):
        writer.writerow({"path": name, "role": roles[name], "bytes": len(payload[name]), "sha256": sha(payload[name])})
    add("FILE_INDEX.csv", index_stream.getvalue().encode("utf-8-sig"), "全包成员索引（不索引自身）")
    stage = destination / STEM
    stage.mkdir(exist_ok=False)
    for name, content in payload.items():
        if "/" not in name or name.startswith("review_tools/"):
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    temporary = destination / f"{STEM}.building.zip"
    require(not temporary.exists(), "已有未完成打包文件，不能覆盖")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(payload):
            archive.writestr(name, payload[name])
    with zipfile.ZipFile(temporary) as archive:
        require(archive.testzip() is None, "ZIP CRC检查不通过")
        require(not any(count > 1 for count in Counter(archive.namelist()).values()), "ZIP成员重名")
        index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(item["path"] for item in index) == set(archive.namelist()) - {"FILE_INDEX.csv"}, "新包索引范围不符")
        for item in index:
            content = archive.read(item["path"])
            require(len(content) == int(item["bytes"]) and sha(content) == item["sha256"], f"新包索引核对失败：{item['path']}")
        require(all("project/" + p.relative_to(ROOT).as_posix() in archive.namelist() for p in new_files), "本轮成果目录未完整收录")
    temporary.rename(archive_path)
    package_sha = sha(archive_path.read_bytes())
    package_receipt = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                       "status": "PASS_ZIP_STRUCTURAL_DELIVERY_CHECKS", "path": str(archive_path),
                       "bytes": archive_path.stat().st_size, "sha256": package_sha,
                       "members": len(payload), "indexed_members": len(payload) - 1,
                       "frozen_files_verified": len(manifest["files"]), "current_result_tree_files": len(new_files),
                       "prior_archive_indexed_members_verified": len(old_index),
                       "crc_passed": True, "duplicate_members": 0, "index_hash_size_scope_passed": True,
                       "original_archive_unchanged": sha((ROOT / PRIOR).read_bytes()) == PRIOR_SHA,
                       "new_external_review_performed": False, "position_impact": 0}
    (destination / f"{STEM}_receipt.json").write_bytes(encoded(package_receipt))
    (destination / f"{STEM}.sha256.txt").write_text(f"{package_sha}  {archive_path.name}\n", encoding="utf-8")
    print(json.dumps(package_receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
