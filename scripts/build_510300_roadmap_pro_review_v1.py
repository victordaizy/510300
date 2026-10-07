"""保留第一批快照，打包48任务执行结果及尚未完成的条件。"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_510300_intraday_process_pro_review_v1 import csv_bytes, encoded, sha

DELIVERY = ROOT / "deliverables"
REPORT = ROOT / "reports/research/510300_roadmap_execution_v1"
NAME = "510300_48项路线图_执行与未完成条件_Pro审阅包_20261002"
PRIOR_NAME = "510300_第一批封卷修正与D_native基准_Pro审阅包_20261002.zip"
PRIOR_SHA = "8b818b8ffbe9c99c1dd1c9674893a495d427cd150da81a2b250483a69ab958fb"


def build() -> None:
    archive = DELIVERY / (NAME + ".zip")
    building = DELIVERY / (NAME + ".building.zip")
    if archive.exists() or building.exists():
        raise FileExistsError("目标交付已存在，不覆盖既有版本")
    prior = DELIVERY / PRIOR_NAME
    if sha(prior.read_bytes()) != PRIOR_SHA:
        raise ValueError("第一批审阅包身份不符")
    members: dict[str, bytes] = {}
    purposes: dict[str, str] = {}
    mapping: dict[str, dict] = {}

    def add(name: str, value: bytes, purpose: str) -> None:
        members[name], purposes[name] = value, purpose

    with zipfile.ZipFile(prior) as pack:
        rows = list(csv.DictReader(io.StringIO(pack.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if len(pack.namelist()) != len(set(pack.namelist())) or pack.testzip() is not None:
            raise ValueError("第一批包结构异常")
        if {row["path"] for row in rows} != set(pack.namelist()) - {"FILE_INDEX.csv"}:
            raise ValueError("第一批包索引覆盖不符")
        for row in rows:
            value = pack.read(row["path"])
            if len(value) != int(row["bytes"]) or sha(value) != row["sha256"]:
                raise ValueError("第一批包成员不符：" + row["path"])
        for name in pack.namelist():
            destination = name if "/" in name else "prior_first_batch/" + name
            add(destination, pack.read(name), "已核对身份的第一批快照")
        for row in csv.DictReader(io.StringIO(pack.read("06_SOURCE_PATH_MAP.csv").decode("utf-8-sig"))):
            mapping[row["original_path"]] = row
        for relative in ("docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md"):
            add("prior_first_batch/" + Path(relative).name, members["project/" + relative], "第一批共享文档历史快照")

    def source(path: Path, purpose: str, expected: str | None = None) -> None:
        path = path.absolute()
        if not path.is_relative_to(ROOT):
            raise ValueError("本轮附加证据不在项目内：" + str(path))
        name = "project/" + path.relative_to(ROOT).as_posix()
        value = path.read_bytes()
        digest = sha(value)
        if expected is not None and digest != expected:
            raise ValueError("冻结或目录摘要不符：" + str(path))
        if name in members and members[name] != value and path.name not in ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md"):
            raise ValueError("与第一批历史成员发生不允许的变更：" + name)
        add(name, value, purpose)
        mapping[str(path)] = {"original_path": str(path), "package_path": name, "bytes": len(value), "sha256": digest, "purpose": purpose}

    frozen = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        source(ROOT / row["path"], "全部路线图冻结依赖", row["sha256"])
    index = json.loads((REPORT / "file_index.json").read_text(encoding="utf-8"))
    for row in index["files"]:
        source(REPORT / row["path"], "全部路线图结果、台账和实际回执", row["sha256"])
    source(REPORT / "file_index.json", "报告快照的逐文件索引")
    with (REPORT / "data_catalog_additional.csv").open(encoding="utf-8-sig", newline="") as stream:
        additional = list(csv.DictReader(stream))
    for row in additional:
        source(ROOT / row["path"], "附加五项直接资产，保留各自准入限制", row["sha256"])
    for row in json.loads((REPORT / "macro_context_ledger.json").read_text(encoding="utf-8"))["evidence"]:
        source(ROOT / row["path"], "宏观终态上下文的直接回执", row["sha256"])
    for relative in (
        "docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
        "scripts/complete_510300_roadmap_execution_v1.py",
        "scripts/run_510300_roadmap_public_observer_scheduled_v1.py",
        "scripts/install_510300_roadmap_public_observer_v1.ps1",
        "scripts/build_510300_roadmap_pro_review_v1.py",
        "scripts/verify_510300_roadmap_review_package_v1.py",
    ):
        source(ROOT / relative, "当前事实入口或本轮工具；打包不执行调度安装")
    add("06_SOURCE_PATH_MAP.csv", csv_bytes(list(mapping.values())), "原路径到包内路径映射")
    add("verify_review_package.py", (ROOT / "scripts/verify_510300_roadmap_review_package_v1.py").read_bytes(), "保存统计复算入口")
    add("requirements_review.txt", members["prior_first_batch/requirements_review.txt"], "复算直接依赖")

    prompt = """# 请审阅48项路线图的执行结果及未完成条件

先读00_README_FIRST.md，再读全部路线图执行报告、48项台账和一页机制卡。用户在第一批后明确追加“其余的都进行”，此前提示词的仅第一批权限是历史快照。请区分附件建议、实际执行和未完成前提，不由文档授权实盘。

本批目前21项限定范围完成、11项部分完成、16项条件未满足；未声称48项科学研究全部完成，也没有完成高夏普目标。第一批之后无新模型或账户，只新增保存结果区块诊断、来源/机制梳理及零费用原始版本观察。请审查这些计数和停止条件是否合理，不把挂起本身当研究成功。

请重点裁决：

1. 第一批四项封卷修正及D-native诊断是否准确？原A/B/C和D-common仍保留，是否存在真正需要纠正的实现问题？D-native移除共同门也改变训练成员及更新时钟，不能把差异全归于一个门。
2. 20万元BASE D-native年化1.021%、夏普0.142、回撤16.978%，压力年化−0.603%；MSE点改善6.357%但新增区间跨0。这些是否足以否定当前设计的晋升？请区分全账户收益、周期净期望、预测误差，不把周期均值当复利账户。
3. 新统计明确为看到第一批结果后的诊断：471日收益和364日预测分别使用保存的5000次20日循环区块索引，五比较校正仅限本轮。区块单位、重叠标签、事件稀疏性和历史多轮选择会怎样限制解释？不要把局部99%区间当全历史纠偏，也不要替换原A/B/C区间。
4. 96因子/16家族映射、787条直接摘要是否正确保留旧失败和未知？未找到99篇的单一核准索引，未声称99篇已逐一读完。哪些映射或旧用途认定需要具体修正？摘要数不是独立试验次数。
5. R02融资、R03完整同指数ETF需求、R04披露后调整、R05真实IF持有成本的准入暂停是否有证据？请指出是否存在被遗漏的、现在即可用且确实不同于已拒绝用途的最小实验；若没有，也允许维持暂停。请勿靠换窗口、费用、分位、符号或退出救回旧负结果。
6. 最新官方回执只有2026-09-30沪市融资1行及ETF份额920行，深市TLS失败。接收时刻是2026-10-02，不能倒填历史首次可得。920不是920只同指数ETF、份额不是净资金、偿还不是强平。零费用定时任务只存原响应；安装及休市分支已实测，未来交易日成功仍未知。这样的源观察应完成哪些任务、不能完成哪些任务？
7. 21/11/16是否如实对应附件验收标准？真实账户参数、完整新输入、合格入场、组合、独立未来样本与实盘授权各自缺什么？有无将可以继续的普通工作误设成等待用户或不可达门槛？请给出具体证据及最小纠正。

请输出：总体裁决；有路径/字段/反例的P0/P1/P2问题；48项完成认定的必要调整；值得继续的最多三个实验，逐个写假设、旧研究差异、必要输入、固定主对照、统计/经济验收、停止条件；已执行和未执行的核验。只审研究有效性及其实现，不做无关安全清单或泛化重构。

本包未上传，未获本轮外部审核，保存结果复算不等于独立金融验证。不要运行prepare/run、采集器或安装脚本来开展新实验；允许只运行根verify_review_package.py读取保存结果。完整复算沿用Windows路径环境，不能执行请明确静态审阅，不虚称已运行。
"""
    readme = """# 全部48项路线图：执行与未完成条件

当前结论：21项在所列范围完成、11项部分完成、16项等待具体前提；目标未达。最新阅读入口是本目录；prior_first_batch/及prior_review/均为历史快照。第一批“其余未启动”的表述由本轮新授权与执行台账更新，原冻结结果不改。

阅读顺序：

1. 01_PRO_REVIEW_PROMPT.md、02_USER_REQUEST_AND_SCOPE.md。
2. project/reports/research/510300_roadmap_execution_v1/全部路线图执行报告.md、48项执行台账.csv。
3. 同目录mechanism_cards/、mechanism_admission.json、uncertainty_intervals.csv、payoff_distribution.csv。
4. 同目录received_public_facts.csv、raw_vintages/、scheduled_trigger_verification.json、forward_manifest.json。
5. project/docs/510300_ROADMAP_EXECUTION_V1.md、相应config/research/scripts/tests；项目状态第15C节和决策PR30—PR34。
6. project/reports/research/510300_daily_native_baseline_v1/第一批研究结论与接续.md及其原A/B/C直接证据。
7. 03_EVIDENCE_MAP.md、04_SCOPE_AND_LIMITS.md、05_VERIFICATION.json、FILE_INDEX.csv。

收到的原路线图HTML、48项XLSX和用户粘贴文本完整保留在project/reports/research/510300_daily_native_baseline_v1/received/。

复算入口只读取保存文件，复算原16和第一批4账户、706个保存预测、共同364日误差及C−D边际，再核对新5组区间、4组周期分布与所有月份贡献。不训练、不回放账户、不重抽索引、不联网。Windows Python依赖见requirements_review.txt。

📁 解压目录/verify_review_package.py（在解压根目录的PowerShell中执行）

```powershell
python .\\verify_review_package.py --root .
```

这是共享给Pro的本地材料；没有公开上传或外部审核。来源时钟与完整金融有效性不由ZIP正确性保证。
"""
    scope = """# 请求、授权与实际范围

用户先选择“先完成第一批：封卷修正、依赖拆分与可运行基准”，随后明确“其余的都进行”。本轮因此推进全部48任务，并延续用户要求的Pro审阅交付。零费用、原失败不救回、只研究510300.SH/CASH_CNY等约束保持。期货、融资、ETF其他产品和公司公告仅作为观测来源，不授权交易这些资产。

本轮21项所列范围完成、11项部分完成、16项未满足前提。已有第一批模型36次顺序更新、4账户；其后新拟合0、新账户0。新5项区块统计是事后诊断，不是独立确认。新公开源3请求2成功，原始响应228762字节；本地09:15工作日任务已装，休市分支实测通过。未来联网成功和策略前瞻均未建立。

真实本金、用款期限、风险和券商费用尚未确认；继续用2万/20万元作为研究情景，不把这些设定写成实际账户事实。没有合格新入场模块；组合、实际成交、扩容及未来月度策略统计不能宣称完成。F04按路线图需另外具体授权，本轮未连接券商或下单。
"""
    evidence = """# 直接证据导航

R = project/reports/research/510300_roadmap_execution_v1/。

|审核问题|直接材料|
|---|---|
|48项原要求与新处置|R/original_48_tasks.json、48项执行台账.csv、result.json|
|96项/旧研究重叠|R/mechanism_registry.csv、study_registry.jsonl、mechanism_prior_evidence.csv；43项冻结中的直接结果|
|新机制为何未跑|R/mechanism_cards/、experiment_specs.json、mechanism_admission.json|
|原账户差/预测差|R/paired_account_returns.parquet、paired_predictions.parquet；第一批和原A/B/C完整结果|
|保存区块区间|R/bootstrap_indices.npz、两个日期表、uncertainty_intervals.json、saved_statistics_receipt.json|
|全部周期/月份/事件|R/payoff_distribution.csv、all_monthly_marginal_pnl.csv、all_event_cluster_sensitivities.csv、failure_and_success_ledger.csv|
|29项资产用途|R/data_catalog_original_24.csv、data_catalog_additional.csv及对应包内文件|
|官方原始版本|R/raw_vintages/、collector_current_status.json、received_public_facts.csv、source_permissions.md|
|调度实际验证|R/scheduled_task_receipt.json、scheduled_trigger_verification.json、scheduled_observer_last_receipt.json及包装/安装脚本|
|组合/策略前瞻未启动|R/single_account_portfolio.json、risk_policy.json、forward_manifest.json、monthly_scorecard.json|
|保存文件复算|根05_VERIFICATION.json、verify_review_package.py、FILE_INDEX.csv|

787条摘要是在当时读取直接结果后保存的注册快照，未递归纳入各分支全部原始附件；它们的路径/摘要/状态留在登记内。重点准入判定依赖的43项冻结和另列宏观终态直接回执包含在包内。
"""
    limits = """# 包含内容和范围限制

- 从核对SHA256的第一批ZIP复用全部直接证据，无嵌套ZIP；原A/B/C16账户、第一批4账户与收到的附件完整保留。历史根导航分别位于prior_first_batch/和prior_review/。
- 增加本轮完整报告、43项冻结依赖、5项新增目录资产、宏观直接终态回执、新代码/配置/协议/测试、当前共享文档快照。5项资产共约30.14MB未压缩，不是新采集交易信号。
- 181日完整Level-2三流、787个分支全部原件、整仓Git/虚拟环境和浏览器数据未递归打包。不能把本包称作全项目全部历史材料或99篇完整文献集。
- 原A/B/C的旧逐次抽样索引不存在，未复算旧区间；本轮保存的5组区间可复算，但仅是看过结果后的诊断。五比较局部校正不覆盖所有历史选择。
- 份额/融资最新回取不建立历史首版，失败回执保留。定时任务是原始数据观察，没有产生策略前瞻样本。未来成功需要未来回执，任务安装不能替代。
- 仅本地生成，未上传、未外部审核、未建立独立金融验证或真实成交；真实风险参数仍未知。保存结果一致不证明研究已经达到收益目标。
- 本地包预算50MiB，不是对任何平台上传限制的承诺。新共享文档只代表打包瞬间；后续交付记录不会反向修改包。
"""
    for name, body in (("00_README_FIRST.md", readme), ("01_PRO_REVIEW_PROMPT.md", prompt), ("02_USER_REQUEST_AND_SCOPE.md", scope),
                       ("03_EVIDENCE_MAP.md", evidence), ("04_SCOPE_AND_LIMITS.md", limits)):
        add(name, body.encode("utf-8"), "当前范围、结论和审阅导航")
    add("context/roadmap_package.json", encoded({"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "prior_zip_name": PRIOR_NAME, "prior_zip_sha256": PRIOR_SHA, "prior_identity_verified": True,
        "authority": "其余的都进行", "all_48_scientific_tasks_complete": False, "uploaded": False}), "交付谱系与新授权")

    def write_archive() -> None:
        records = [{"path": name, "bytes": len(value), "sha256": sha(value), "purpose": purposes[name]} for name, value in sorted(members.items())]
        with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as pack:
            for name, value in sorted(members.items()):
                pack.writestr(name, value)
            pack.writestr("FILE_INDEX.csv", csv_bytes(records))

    write_archive()
    print("已生成临时包，开始复算解压副本的保存统计。", flush=True)
    with tempfile.TemporaryDirectory(prefix="路线图Pro复算_", dir=DELIVERY) as temp:
        temporary = Path(temp)
        if not temporary.resolve().is_relative_to(DELIVERY.resolve()):
            raise ValueError("临时目录不在交付范围")
        extracted = temporary / "package"
        extracted.mkdir()
        with zipfile.ZipFile(building) as pack:
            for name in pack.namelist():
                part = PurePosixPath(name)
                if part.is_absolute() or ".." in part.parts:
                    raise ValueError("成员路径越界")
            pack.extractall(extracted)
        result_path = temporary / "recomputation.json"
        process = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "verify_review_package.py"), "--root", str(extracted), "--output", str(result_path)],
                                 cwd=extracted, capture_output=True, encoding="utf-8")
        if process.returncode:
            raise RuntimeError("解压副本复算失败：" + process.stdout + process.stderr)
        recomputation = json.loads(result_path.read_text(encoding="utf-8"))
    add("05_VERIFICATION.json", encoded(recomputation), "解压副本已实际执行的保存结果复算")
    write_archive()
    with zipfile.ZipFile(building) as pack:
        names = pack.namelist()
        if len(names) != len(set(names)) or pack.testzip() is not None:
            raise ValueError("交付包CRC或成员唯一性失败")
        rows = list(csv.DictReader(io.StringIO(pack.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if {row["path"] for row in rows} != set(names) - {"FILE_INDEX.csv"}:
            raise ValueError("交付索引覆盖失败")
        for row in rows:
            value = pack.read(row["path"])
            if len(value) != int(row["bytes"]) or sha(value) != row["sha256"]:
                raise ValueError("交付成员身份失败：" + row["path"])
    if building.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("交付包超过本地50MiB预算")
    os.replace(building, archive)
    receipt = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS_ROADMAP_PACKAGE_STRUCTURE_AND_SAVED_RECOMPUTATION", "zip_path": str(archive),
        "zip_bytes": archive.stat().st_size, "zip_sha256": sha(archive.read_bytes()),
        "members": len(names), "indexed_members": len(rows), "prior_zip_sha256": PRIOR_SHA,
        "roadmap_frozen_files": len(frozen["files"]), "roadmap_report_files": len(index["files"]) + 1,
        "additional_direct_assets": len(additional), "saved_recomputation": recomputation,
        "new_fits": 0, "new_account_simulations": 0, "new_random_draws": 0, "new_network_requests": 0,
        "uploaded": False, "external_review_performed": False, "all_48_scientific_tasks_complete": False}
    (DELIVERY / (NAME + "_交付校验.json")).write_bytes(encoded(receipt))
    (DELIVERY / (NAME + ".sha256")).write_text(receipt["zip_sha256"] + "  " + archive.name + "\n", encoding="utf-8")
    (DELIVERY / (NAME + "_给Pro的审阅提示词.md")).write_text(prompt, encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "zip_path", "zip_bytes", "zip_sha256", "members", "indexed_members")}, ensure_ascii=False))


if __name__ == "__main__":
    build()
