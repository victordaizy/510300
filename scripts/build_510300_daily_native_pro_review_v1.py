"""以已验证的原审阅快照为底本，制作第一批新结果审阅包。"""
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
REPORT = ROOT / "reports/research/510300_daily_native_baseline_v1"
NAME = "510300_第一批封卷修正与D_native基准_Pro审阅包_20261002"
PRIOR_NAME = "510300_分钟过程增量_V1_Pro审阅包_20261002.zip"
PRIOR_SHA = "cf905b3cba0d24ed2683ef02f79d9b950a8d98a06b9713fc3cdedd356e1abcef"


def build() -> None:
    archive = DELIVERY / (NAME + ".zip")
    temporary_archive = DELIVERY / (NAME + ".building.zip")
    if archive.exists() or temporary_archive.exists():
        raise FileExistsError("目标文件已存在；不覆盖原交付")
    prior = DELIVERY / PRIOR_NAME
    if sha(prior.read_bytes()) != PRIOR_SHA:
        raise ValueError("原审阅ZIP身份不符，停止复用")
    members: dict[str, bytes] = {}
    purposes: dict[str, str] = {}
    mapping: dict[str, dict] = {}

    def add(name: str, body: bytes, purpose: str) -> None:
        members[name] = body
        purposes[name] = purpose

    with zipfile.ZipFile(prior) as pack:
        old_index = list(csv.DictReader(io.StringIO(pack.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if pack.testzip() is not None or len(pack.namelist()) != len(set(pack.namelist())):
            raise ValueError("原包结构不符")
        if {row["path"] for row in old_index} != set(pack.namelist()) - {"FILE_INDEX.csv"}:
            raise ValueError("原包索引覆盖不符")
        for row in old_index:
            value = pack.read(row["path"])
            if sha(value) != row["sha256"] or len(value) != int(row["bytes"]):
                raise ValueError("原包成员身份不符：" + row["path"])
        for name in pack.namelist():
            # 原根说明和旧共享状态保留快照，当前导航另建。
            destination = name if "/" in name else "prior_review/" + name
            add(destination, pack.read(name), "已核对原ZIP的历史快照")
        for row in csv.DictReader(io.StringIO(pack.read("06_SOURCE_PATH_MAP.csv").decode("utf-8-sig"))):
            mapping[row["original_path"]] = row
        for relative in ("docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md"):
            original = "project/" + relative
            add("prior_review/" + Path(relative).name, members[original], "原共享文档打包时快照")

    def source(path: Path, purpose: str) -> None:
        path = path.absolute()
        if path.is_relative_to(ROOT):
            name = "project/" + path.relative_to(ROOT).as_posix()
        elif str(path) in mapping:
            name = mapping[str(path)]["package_path"]
        else:
            raise ValueError("未映射的工作区外来源：" + str(path))
        value = path.read_bytes()
        add(name, value, purpose)
        mapping[str(path)] = {"original_path": str(path), "package_path": name,
                              "bytes": len(value), "sha256": sha(value), "purpose": purpose}

    frozen = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        path = Path(row["path"])
        if sha(path.read_bytes()) != row["sha256"]:
            raise ValueError("新冻结文件已改变：" + str(path))
        source(path, "第一批冻结依赖")
    current_index = json.loads((REPORT / "file_index.json").read_text(encoding="utf-8"))
    for row in current_index["files"]:
        if sha((REPORT / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("第一批索引与保存文件不符")
    for path in sorted(REPORT.rglob("*")):
        if path.is_file():
            source(path, "第一批完整结果、附件与回执")
    for relative in (
        "docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
        "scripts/complete_510300_daily_native_first_batch_v1.py",
        "scripts/build_510300_daily_native_pro_review_v1.py",
        "scripts/verify_510300_daily_native_review_package_v1.py",
    ):
        source(ROOT / relative, "当前事实入口或第一批交付工具")
    add("06_SOURCE_PATH_MAP.csv", csv_bytes(list(mapping.values())), "当前原路径至包内映射，兼容原复算入口")
    add("verify_review_package.py", (ROOT / "scripts/verify_510300_daily_native_review_package_v1.py").read_bytes(), "保存结果复算入口")
    add("requirements_review.txt", members["prior_review/requirements_review.txt"], "同一验证环境的直接依赖")

    prompt = """# 请审阅第一批：封卷修正、依赖拆分与D-native基准

请解压后先读00_README_FIRST.md，再读取第一批报告、协议、四层资格、全部账户和完整源码。用户只选择本批，不代表授权执行48项。

这是你此前对分钟过程A/B/C提出四项修正后的本地落实。原实验保留不变，四项意见已按保存证据核对；本批新增一个固定D-native设计、36次顺序更新、4个账户。请独立裁决，不把本地核验回执当金融验证。

1. 四项修正是否准确？108日是否为80日C金额历史资格、28日制度预热；原预测是否真的要求当日未来标签成熟；C−D人民币边际差额是否可由2024年9月解释；181日来源文件存在是否等于严格用途准入。C−D累计3729.8021元、2024年9月0元请亲自核对。
2. D-native是否是公平的可运行日线参考？检查日线八字段重建、训练起点不扩样、504成熟日、20日更新、标准化及目标成熟时钟。原共同门移除也改变训练成员和首次拟合时钟，不能把全部差异当单一布尔门的因果效应。
3. 日频特征跨2026-07-06沿用是否可接受为条件诊断？评价制度影响矩阵是否把日线定义、盘中尾段、竞价容量与真实成交分开；保留未知，不因结果不利增加事后过滤。
4. 在完整471日、同一费用和账户合同下，20万元BASE的D-native年化1.021%、夏普0.142、回撤16.978%，STRESS年化−0.603%。共同364预测日MSE点值改善6.357%却没有账户优势，应该怎样解释？核对毛报价损益少38828.40元、额外摩擦6328.96元和净差45157.36元，别只归咎手续费或只比较周期等权均值。
5. 价格、量、金额质量与执行资格是否拆分充分？有限单位/累计/浮点边界诊断是否合理，哪些根因仍未知？不要把一档容许当全面修复，或以量乘价填缺值。当前日期缺失是否正确返回未知，而非现金或旧信号？
6. 检查足以改变结论的实现问题，列路径、函数/行号、失败例和最小修正。区分真正纠错与新设计探索。未执行的检查明确写未执行。
7. 判断第一批哪些应验收、哪些需最小纠正，按优先级提出最多三个后续任务。每个写假设、与旧研究的实质差异、所需现有数据或新证据、单一实验、事前评价和停止条件。允许结论是暂不再跑收益实验。不要通过窗口、月份、费用、方向或大规模搜索营救负结果。

输出：总体裁决；P0/P1/P2证据问题表；四项修正判定；D-native工程/预测/账户结论分开；可验收项与必要修正；最多三项接续任务及停止条件；实际执行的核验和缺失证据。只讨论相关研究质量，不做泛化安全或架构审查。

无新增bootstrap，旧逐次抽样索引未保存；本次未重算区间。原历史已暴露，新增诊断没有建立独立验证。本包不会授权采集、模拟盘、券商或实际下单。
"""
    readme = """# 第一批审阅包阅读入口

第一批已完成，覆盖恢复与收益优势分开判断。D-native覆盖原471账户日，但20万元BASE年化1.021%、夏普0.142、回撤16.978%；压力年化−0.603%，未达到金融目标。

建议阅读顺序：

1. 01_PRO_REVIEW_PROMPT.md：本次审阅问题。
2. 02_USER_REQUEST_AND_SCOPE.md：用户选择范围；完整收到的附件在project/reports/research/510300_daily_native_baseline_v1/received/。
3. project/reports/research/510300_daily_native_baseline_v1/第一批研究结论与接续.md及D_common与native净值.png。
4. 同目录封卷修正与设计分类.md、依赖与运行合同.md、21_48项清单本批落实对照.csv。
5. project/docs/510300_DAILY_NATIVE_BASELINE_V1.md、project/config/510300_daily_native_baseline_v1.json、project/research/daily_native_baseline_v1.py及测试。
6. 03_EVIDENCE_MAP.md、04_SCOPE_AND_LIMITS.md、05_VERIFICATION.json、FILE_INDEX.csv。
7. project/docs/PROJECT_STATE.md第15B节和RESEARCH_DECISIONS.md的PR26—PR29。

本包同时保留原A/B/C完整结果、实际清单24项来源、原23项及新67项冻结文件。原包根导航、提示词、验证回执和共享状态放在prior_review/，仅代表原打包时快照；当前以本根导航及索引为准。原研究目录没有被修改，原C边际月份解释以本批纠正说明为准。

project/维持相对项目布局，原绝对路径未改写，见06_SOURCE_PATH_MAP.csv。报告与账本CSV可直接阅读，Parquet保留精确数据。不要重复执行prepare/run；单次运行声明仍生效。只有下列入口用于复算保存文件。

📁 解压目录/verify_review_package.py（PowerShell，从解压根目录执行）

```powershell
python .\\verify_review_package.py --root .
```

依赖版本列于requirements_review.txt。该入口复算原16账户、原364日预测点统计及集中度、新4账户、706个保存预测、共同日MSE与C−D月份差；不会重新训练、回放账户或抽样。不能执行时请作静态审阅，并写明未执行。
"""
    scope = """# 用户请求及本次范围

用户在收到八工作包/48任务路线图后明确答复：“先完成第一批：封卷修正、依赖拆分与可运行基准”。本包供用户延续原Pro审阅流程；未上传。

11项本批明确任务完成：A01—A06、D02、D04—D07；G02/G03/D01/D03四项保留本分支口径、家族范围、24项资产与金额根因未知的边界；其他33项未启动。详细原状态与落实范围见逐项CSV，不把附件建议当全部授权。

本批为一个设计诊断，不是新alpha候选搜索。原A/B/C与旧来源失败不复活，原审阅ZIP保持原SHA256。只交易研究对象510300.SH/CASH_CNY；没有市场下载、自动任务、前瞻试验或实际订单。研究参考夏普1.2、年化10%，主242年化；真实本金、风险限制与券商费用未确认，独立验证未建立。
"""
    evidence = """# 证据导航

N = project/reports/research/510300_daily_native_baseline_v1/；O = project/reports/research/510300_intraday_process_increment_v1/。

|问题|直接证据|
|---|---|
|收到的审阅与48任务|N/received/；N/authority.json；N/21_48项清单本批落实对照.csv|
|108日原因、标签成熟、C−D边际|N/review_reconciliation.json；N/01_原108日原因表.csv；N/02_C相对D逐日人民币增量.csv；N/03_C相对D逐月人民币增量.csv|
|依赖与制度拆分|N/availability_contract_v2.json；N/05_各模型输入资格.csv；N/19_制度影响矩阵.csv；N/official_rule_reference.json|
|日线字段、历史成熟成员与706预测|N/daily_asof_panel.parquet；N/09_D_native全部训练成员.csv；N/native_saved_models.json；N/08_D_native全部原点.csv|
|4个完整账户与4原账户对照|N/11_D_common与native完整账户.csv；N/native_accounts.csv；N/native_orders.csv；N/native_decisions.csv；N/native_cycles.csv|
|预测改善与毛收益/摩擦|N/14_原共同预测日期对照.csv；N/17_native相对common人民币归因.csv；N/15_完整期与缺口区段.csv|
|金额、字段、源覆盖与未知|N/amount_diagnostic.json；N/06_分钟字段质量分离.csv；N/07_数据资产与用途.csv；N/20_两元严格容差浮点边界.csv|
|当地日期运行与验收|N/local_current_status_example.json；N/tests_receipt.json；N/completion_receipt.json；N/freeze.json；N/run_claim.json|
|原A/B/C完整证据|O全部目录；prior_review/03_EVIDENCE_MAP.md；原始分钟、日线、分红等仍在project/data/|
|解压副本复算|05_VERIFICATION.json；verify_review_package.py；根FILE_INDEX.csv|

共享状态文档包含其他分支背景，其全部链接没有递归打包。当前根索引列明实际范围。
"""
    limits = """# 纳入与边界

- 原包先核对SHA256及其成员索引，再复用完整直接证据；原包不变。本包不嵌套复制整个ZIP，保留其历史根说明在prior_review/。
- 本批报告目录全部文件、收到的三份附件、新67项冻结依赖、代码/配置/协议/测试、CSV/Parquet账户、运行与冻结回执全部包含。原A/B/C16账户和原清单24项来源保留。
- 181日Level-2整套三流未打包、未重读，保留来源摘要及资格限制。宏观/技术其他分支的中间件、全仓库Git/虚拟环境、浏览器资料不在范围。
- 原bootstrap逐次索引未保存，本包未新抽样；新基准没有新增置信区间。保存值复算与结构正确不是完整训练复现、真实成交或独立金融验证。
- 日线跨2026-07-06沿用只是一项条件诊断；历史首次送达时钟和金额对齐根因未建立。当前2026-10-02无本地数据，不能据此获得当天观点或实际持仓。
- 原20万元/2万元和费用为研究情景，未确认真实账户；更好覆盖和更小MSE不足以晋升策略。其余33任务未启动。
- 未指定平台上传硬上限；本地使用50MiB包预算，不能据此断言外部平台上限。
"""
    for name, body in [("00_README_FIRST.md", readme), ("01_PRO_REVIEW_PROMPT.md", prompt),
                       ("02_USER_REQUEST_AND_SCOPE.md", scope), ("03_EVIDENCE_MAP.md", evidence),
                       ("04_SCOPE_AND_LIMITS.md", limits)]:
        add(name, body.encode("utf-8"), "本批阅读入口与审阅范围")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    add("context/first_batch_package.json", encoded({"created_at": stamp, "prior_zip_name": PRIOR_NAME,
        "prior_zip_sha256": PRIOR_SHA, "prior_identity_verified_before_reuse": True,
        "source_external_review_received": True, "new_native_result_externally_reviewed": False,
        "new_market_downloads": 0, "uploaded": False}), "新旧快照谱系")

    def write_archive() -> None:
        index = [{"path": name, "bytes": len(body), "sha256": sha(body), "purpose": purposes[name]}
                 for name, body in sorted(members.items())]
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as pack:
            for name, body in sorted(members.items()):
                pack.writestr(name, body)
            pack.writestr("FILE_INDEX.csv", csv_bytes(index))

    write_archive()
    print("第一批临时包已写入，开始解压副本的保存结果复算。", flush=True)
    with tempfile.TemporaryDirectory(prefix="第一批Pro核验_", dir=DELIVERY) as temp:
        temp_root = Path(temp)
        if not temp_root.resolve().is_relative_to(DELIVERY.resolve()):
            raise ValueError("临时目录超出交付范围")
        extracted = temp_root / "package"
        extracted.mkdir()
        with zipfile.ZipFile(temporary_archive) as pack:
            for name in pack.namelist():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("压缩成员路径越界")
            pack.extractall(extracted)
        result_path = temp_root / "recomputation.json"
        proc = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "verify_review_package.py"),
                               "--root", str(extracted), "--output", str(result_path)],
                              cwd=extracted, capture_output=True, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError("副本保存结果复算失败：" + proc.stdout + proc.stderr)
        recomputation = json.loads(result_path.read_text(encoding="utf-8"))
    add("05_VERIFICATION.json", encoded(recomputation), "新旧保存结果在解压副本的实际复算")
    write_archive()
    with zipfile.ZipFile(temporary_archive) as pack:
        names = pack.namelist()
        if len(names) != len(set(names)) or pack.testzip() is not None:
            raise ValueError("正式包重名或CRC失败")
        rows = list(csv.DictReader(io.StringIO(pack.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if {row["path"] for row in rows} != set(names) - {"FILE_INDEX.csv"}:
            raise ValueError("正式包索引覆盖不全")
        for row in rows:
            value = pack.read(row["path"])
            if len(value) != int(row["bytes"]) or sha(value) != row["sha256"]:
                raise ValueError("正式包成员身份失败")
    if temporary_archive.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("包超过本地50MiB预算")
    os.replace(temporary_archive, archive)
    receipt = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
               "status": "PASS_FIRST_BATCH_PACKAGE_STRUCTURE_AND_SAVED_RECOMPUTATION",
               "zip_path": str(archive), "zip_bytes": archive.stat().st_size, "zip_sha256": sha(archive.read_bytes()),
               "members": len(names), "indexed_members": len(rows), "prior_zip_sha256": PRIOR_SHA,
               "native_frozen_files": len(frozen["files"]), "native_report_files": len(current_index["files"]) + 1,
               "crc_pass": True, "index_size_sha256_coverage_pass": True,
               "saved_result_recomputation": recomputation, "new_fits": 0, "new_account_simulations": 0,
               "new_random_draws": 0, "new_market_downloads": 0, "external_native_review_performed": False,
               "uploaded": False}
    (DELIVERY / (NAME + "_交付校验.json")).write_bytes(encoded(receipt))
    (DELIVERY / (NAME + ".sha256")).write_text(receipt["zip_sha256"] + "  " + archive.name + "\n", encoding="utf-8")
    (DELIVERY / (NAME + "_给Pro的审阅提示词.md")).write_text(prompt, encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "zip_path", "zip_bytes", "zip_sha256", "members", "indexed_members")}, ensure_ascii=False))


if __name__ == "__main__":
    build()
