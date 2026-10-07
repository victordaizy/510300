"""把已完成的分钟过程研究制作为聚焦、可核对的Pro审阅包。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT_REL = Path("reports/research/510300_intraday_process_increment_v1")
REPORT = ROOT / REPORT_REL
DELIVERY = ROOT / "deliverables"
NAME = "510300_分钟过程增量_V1_Pro审阅包_20261002"
BYTE_LIMIT = 50 * 1024 * 1024


def encoded(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8-sig")


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, encoding="utf-8", check=True)
    return result.stdout.strip()


def build() -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    archive = DELIVERY / (NAME + ".zip")
    temporary_archive = DELIVERY / (NAME + ".building.zip")
    if archive.exists() or temporary_archive.exists():
        raise FileExistsError("目标包或临时包已存在，请保留旧交付后另用版本名")
    members: dict[str, bytes] = {}
    purposes: dict[str, str] = {}
    source_map: dict[str, dict] = {}

    def add(name: str, body: bytes, purpose: str) -> None:
        if name in members and members[name] != body:
            raise ValueError("包内路径冲突：" + name)
        members[name] = body
        purposes[name] = purpose

    def source(path: Path, purpose: str) -> str:
        original = str(path)
        if path.is_relative_to(ROOT):
            target = "project/" + path.relative_to(ROOT).as_posix()
        else:
            target = "context/user_goal_objective.md"
        body = path.read_bytes()
        add(target, body, purpose)
        source_map[original] = {"original_path": original, "package_path": target, "bytes": len(body), "sha256": sha(body), "purpose": purpose}
        return target

    frozen = json.loads((REPORT / "freeze.json").read_text(encoding="utf-8"))
    for row in frozen["files"]:
        target = source(Path(row["path"]), "本轮冻结文件")
        if sha(members[target]) != row["sha256"]:
            raise ValueError("冻结身份已变化：" + row["path"])
    current_index = json.loads((REPORT / "file_index.json").read_text(encoding="utf-8"))
    for row in current_index["files"]:
        path = REPORT / row["path"]
        if sha(path.read_bytes()) != row["sha256"]:
            raise ValueError("研究当前索引不符：" + str(path))
    for path in sorted(REPORT.iterdir()):
        if path.is_file():
            source(path, "本轮完整保存结果与回执")
    inventory = pd.read_csv(REPORT / "01_实际数据清单.csv")
    for row in inventory.to_dict("records"):
        target = source(Path(row["路径"]), "实际清单来源：" + row["用途"])
        if sha(members[target]) != row["SHA256"]:
            raise ValueError("实际数据清单身份不符：" + row["用途"])

    related = [
        "docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
        "scripts/complete_510300_intraday_process_delivery_v1.py",
        "scripts/verify_510300_intraday_process_review_package_v1.py",
        "scripts/build_510300_intraday_process_pro_review_v1.py",
        "docs/510300_AFternoon_ENTRY_V1.md",
        "config/510300_afternoon_entry_v1.json",
        "research/ease_of_movement_inputs_v1.py",
        "config/510300_stk_mins_source_admission_v2.yaml",
        "docs/510300_NBS_FIXED_5MIN_USAGE_AUTHORIZATION_20260905.md",
        "docs/510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md",
        "reports/research/510300_pressure_recovery_v1/source_admission_20261002/summary.json",
        "reports/research/510300_pressure_recovery_v1/blocked_gate_recheck_20261002/summary.json",
        "reports/research/510300_pressure_recovery_v1/blocked_gate_recheck_20261002/当前阻塞与恢复条件.md",
    ]
    for relative in related:
        # Windows文件名大小写不敏感，统一以真实文件名写入，兼容外部Linux解压。
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError("直接相关证据不存在：" + relative)
        actual = next(x for x in path.parent.iterdir() if x.name.casefold() == path.name.casefold())
        source(actual, "当前状态、交付工具或旧研究直接背景")

    export_rows = []
    for filename in ("08_完整账户逐日账本.parquet", "10_逐日决策.parquet", "fixed_5min_process_windows.parquet", "normalized_prices.parquet", "normalized_dividends.parquet"):
        frame = pd.read_parquet(REPORT / filename)
        destination = "readable/" + Path(filename).with_suffix(".csv").name
        add(destination, frame.to_csv(index=False).encode("utf-8-sig"), "由原Parquet逐行导出，方便Pro读取")
        export_rows.append({"source": "project/" + (REPORT_REL / filename).as_posix(), "csv": destination, "rows": len(frame), "columns": list(frame.columns), "new_research": False})
    add("readable/CSV_EXPORTS.json", encoded(export_rows), "可读导出的来源与形状")
    add("06_SOURCE_PATH_MAP.csv", csv_bytes(list(source_map.values())), "原绝对路径至包内相对路径；不改原冻结文件")
    requirements = "\n".join(f"{name}=={version(name)}" for name in ("numpy", "pandas", "pyarrow", "matplotlib", "pytest", "tzdata")) + "\n"
    add("requirements_review.txt", requirements.encode(), "当前验证环境的直接依赖版本，不包含虚拟环境")
    add("verify_review_package.py", (ROOT / "scripts/verify_510300_intraday_process_review_package_v1.py").read_bytes(), "便携的保存结果复算入口")

    prompt = """# 给Pro的审阅任务：510300分钟过程是否增加可兑现的日线外信息

请先解压本包，按00_README_FIRST.md阅读。请独立检查证据，不要直接复述现有结论，也不要为凑出夏普1.2而优化参数。

用户希望利用已有、零新增费用的510300分钟数据，检验冲击后修复持续性A、相近活动下的价格推进变化B、固定尾盘变化C是否提供日线之外、可到达合法跨日交易时点的信息。执行资产仅510300.SH/CASH_CNY，不要求每天交易。盘口、队列、同步IOPV属于储备，没有这些数据不应再次阻塞本轮；分钟量价也不能冒充订单流。

本轮固定结果：291851根/1211日；1574冲击；869共同完整输入日；364共同预测日；95次保存拟合；D/A/B/C×2万元/20万元×基础/压力费用共16账户。A/B/C的MSE分别比日线增加1.717%/1.668%/0.107%；20万元BASE净夏普D/A/B/C约0.975/0.770/0.654/1.039。C的正账户差区间跨0。请验证这些数字和它们能支持的结论，而不是把它们当作前提。

请重点完成以下工作：

1. **先给独立裁决。** 本版“没有确认可靠增量”是否合理？分别评判A/B/C属于实现错误、测量不足、证据不足、固定表达无效，还是已有可以成立的有限事实。不要把“未显著”直接写成机制永远不存在；也不要把正点值写成稳定alpha。
2. **检查时钟与泄漏。** 逐分钟标签、盘后15:05、次日开盘/再下一日收盘、成熟训练标签、过去60日阈值/归一化、标准化/裁剪、20日重拟合是否在代码中一致？检查事件触发、30分钟后观察、事件合并和按日汇总有没有事后挑事件。预测表包含诊断未来标签，请核对模型实际选列。
3. **检查对照是否公平。** D日线变量是否充分；A_CONTROL是否真正分离“已反弹”和持续性；C是否有独立的新条件用途，还是重复旧失败？由B/C质量共同筛选D/A/B/C的日期是否造成选择偏差、样本损失或改变研究对象？同一制度训练、504成熟日和岭惩罚的选择会否限制结论？请具体引用代码和行号。
4. **检查测量与执行。** 严格VWAP越界5489根、超过一档42根、一档容许假设、仅45严格完整日是否足够支撑B/C？09:30/15:00量和开收盘作为容量/价格代理是否高估可执行性？100份、最小佣金、跳空、T+1、分红应收和付现、部分及未成交、延迟退出、终点库存是否完整？区别同一规则下的净期望与毛期望扣费，避免重复扣费。
5. **检查统计与利润来源。** 重复冲击是否已正确合并，364日是否足够？20交易日分块及三组校正是否合理、是否仍忽略跨研究选择？C的CAGR差与年化算术收益差区间不能混用。108个NO_VIEW账户日、低暴露和2024年9月单笔利润集中应如何影响判断？95次重拟合不等于95个独立候选；多个账户中的同一市场机会也不是独立样本。
6. **区分纠错与调参。** 冻结不应阻止发现真正的实现/口径错误。若有错误，指出文件、函数/行号、具体失败例和最小纠错方案，说明影响哪些输出、是否需要正式修订后重算。若只是窗口、符号、仓位或退出优化，请明确标为新的探索，不自动营救本版负结果。未亲自执行的验证必须写“未执行”。
7. **给下一步方向。** 排序最多三个有实质新信息或不同经济用途的候选，优先使用包内现有免费数据；每个写清假设、旧研究重叠、必要输入、最小单一实验、事前评价标准、继续/停止条件和能回答的问题。当前没有完整新独立样本，请诚实说明可做诊断与不能做确认的边界。不要给海量指标菜单或大规模参数搜索。允许结论是暂不值得追加实验。

请按以下格式输出：总体裁决；P0/P1/P2问题表（证据路径与行号、影响、最小修正）；A/B/C逐组判定；哪些结论保留/降级/撤回；下一步最多三项优先级与停止条件；尚缺证据与实际执行的核验。优先讨论足以改变结论的问题，不做泛化安全审查或无关架构建议。

包内校验仅确认文件结构、冻结身份、原保存账户/预测点统计和集中度；不构成独立金融验证。原bootstrap索引未保存，本次未重新抽样，不能称区间已从保存抽样重现。请检查相关实现但不要宣称已执行未执行的计算。你收到本包不意味着任何实盘、模拟盘或券商下单授权。
"""
    request = """# 用户请求与范围

本次用户原话：“把报告做成审阅包，我要交给pro审核一下”。

本包对应刚完成的510300_INTRADAY_PROCESS_INCREMENT_V1，供用户自行提交Pro。本地打包不代表已提交或收到Pro反馈。

研究目标完整原文：context/user_goal_objective.md。用户此前已将主线由缺少合格时钟/估值/队列证据的盘口研究调整为现有分钟过程相对日线的跨日增量。零数据费用、有限存储；不强制每天交易。原M1/M2及其他研究分支的拒绝和权限分别保留。

本包不运行新实验，不修改原冻结配置/代码/结果。当前研究完成，可靠增量、夏普1.2和独立验证未建立。本包中提出的未来研究均待审阅、未执行。
"""
    readme = r"""# 先读这里：510300分钟过程增量V1审阅包

这是2026-10-02已完成研究的聚焦快照。请独立审阅“未确认可靠增量”的判断；阅读顺序如下。

1. 01_PRO_REVIEW_PROMPT.md：完整审阅问题及期望输出。
2. 02_USER_REQUEST_AND_SCOPE.md与context/user_goal_objective.md：本次请求和原研究目标。
3. project/reports/research/510300_intraday_process_increment_v1/研究结论与下一步.md：主报告、全部16账户、成本/集中度解释。图片为同目录账户净值对比.png。
4. project/docs/510300_INTRADAY_PROCESS_INCREMENT_V1.md及project/config/510300_intraday_process_increment_v1.json：冻结口径。
5. 03_EVIDENCE_MAP.md：从主要结论定位数据、代码和账本；CSV优先阅读，Parquet保留精确原表。
6. project/docs/PROJECT_STATE.md第15A节及project/docs/RESEARCH_DECISIONS.md的PR21—PR25：长期状态和逐方向处置。
7. 04_SCOPE_AND_LIMITS.md、05_VERIFICATION.json和FILE_INDEX.csv：范围、核验边界和文件身份。

主分钟原始Parquet、日线、分红、原清单24项来源/相关缓存、23项冻结文件、当前研究目录全部文件均已纳入。旧长分钟及广度只为核对用途限制，不是本轮新增输入。包含核心研究模块及其账户依赖、原测试和只读复算入口。不会复制181日完整Level-2或整个仓库。

`project/`保留仓库相对布局；原文件中的Windows绝对路径不改写，通过06_SOURCE_PATH_MAP.csv映射到包内。原freeze.json保持字节不变。原运行入口受已执行标记保护，不应为审阅再次运行prepare/run。共享状态文档保留其他分支背景，未递归打包那些分支的所有链接；本包的直接证据范围以03及FILE_INDEX为准。

便携核验需要Python及requirements_review.txt所列依赖。只读核验会在独立临时目录调用原账户保存复算，不生成新信号、拟合、回测或抽样。它不修改解压后的原研究账本。

📁 解压目录/verify_review_package.py（PowerShell，从解压根目录执行）

```powershell
python .\verify_review_package.py --root .
```

若Pro运行环境不能执行代码，请依据CSV和源文件作静态审阅，并明确实际读取范围；不能把未运行的检查写成通过。已有历史被多次观察，结构检查与保存统计一致均不是独立验证或真实成交证明。
"""
    evidence = """# 结论与直接证据对应表

以下R代表project/reports/research/510300_intraday_process_increment_v1/，P代表project/。这些缩写仅用于阅读，实际文件名不变。

|问题|直接证据|
|---|---|
|用户新目标、盘口储备边界|context/user_goal_objective.md；02_USER_REQUEST_AND_SCOPE.md|
|来源24项、主分钟291851根/1211日|R/01_实际数据清单.csv；P/data/raw/market/510300_1m_tushare_raw.parquet；元数据/原质量回执|
|金额异常与45严格完整日|R/source_quality.json；R/02_逐日质量.csv；R/03_异常分钟原值.parquet|
|旧NBS用途只有八窗口|P/config/510300_nbs_fixed_5min_usage_v2_1.json；P/reports/data_quality/510300_nbs_fixed_5min_usage_v2_1/source_adjudication.json；原始长分钟纳入|
|广度2350有值、473缺失并非整表准入|R/18_相关缓存覆盖与异常.json；R/19_旧广度逐日字段可用性.csv；原缓存纳入|
|全体过程、1574事件及未修复状态|R/04_逐日盘中过程.csv；R/05_全部冲击事件.csv；readable/fixed_5min_process_windows.csv|
|共同日期、标签成熟与95次拟合|R/06_滚动预测.csv；R/07_全部模型参数.json；P/research/intraday_process_increment_v1.py的forecasts/add_labels/fit_ridge|
|D/A/B/C误差与A_CONTROL|R/predictive_comparison.json；R/06_滚动预测.csv；P/config/510300_intraday_process_increment_v1.json|
|全部16完整账户|readable/08_完整账户逐日账本.csv；R/12_全部账户指标.csv；原Parquet保留|
|费用、跳空、T+1、容量、未成交、分红|R/09_全部模拟请求.csv；readable/10_逐日决策.csv；R/11_完成持有周期.csv；P/research/intraday_process_increment_v1.py的simulate_account/constrained_execution；P/research/intraday_overnight_increment_v1.py|
|C增量区间、年份和期限|R/13_账户相对日线增量.csv；R/14_逐年账户归因.csv；R/15_持续时间诊断.csv|
|2024年9月集中度与全部月贡献|R/16_全账户收益集中度.csv；R/17_逐月相对日线归因.csv|
|旧尾盘及RV20拒绝仍在|P/reports/research/510300_intraday_close_pressure_2d_report.json；P/reports/research/510300_pressure_recovery_v1/volatility_conditioned_proxy_20261001/summary.json|
|原冻结/单次执行与保存复算|R/freeze.json；R/run_claim.json；R/saved_output_verification.json；R/post_result_interpretation_receipt.json|
|包内副本实际核验|05_VERIFICATION.json；verify_review_package.py；FILE_INDEX.csv|

研究模块build_process定义事件与三组字段，source_inventory定义质量，account_metrics/run_accounts定义完整账户统计；全源码及原测试已提供，可直接引用行号。固定模型参数均保存。诊断未来收益列仅用于评价，不能当作信号输入。
"""
    limits = """# 包范围、排除项与证据边界

本包内容预算为50MiB，这是本地打包控制值，不是对外部平台上传上限的断言。只为本轮研究打包，原研究文件保持不变。

|项目|包含/排除及原因|
|---|---|
|本轮原数据、代码/配置/测试、23项冻结文件|全部包含；原Windows路径另附映射|
|本轮当前研究目录|全部包含，原file_index.json与包根FILE_INDEX.csv分别保留；根索引对整个包有效|
|实际清单24项|全部包含；旧NBS/广度仍保持原窄用途，不纳入本轮模型|
|旧拒绝/盘口储备|包含直接报告、来源裁决及背景协议；不包含181日543文件等大体积Level-2历史及所有旧分支中间产物|
|其他技术、宏观及全部仓库历史|共享状态文档仅作快照；其他课题的链接不承诺全部随包，不构成那些课题的完整复现包|
|原bootstrap逐次索引|原实验未保存；包内有算法、种子、配置和区间结果。本次不新抽样，区间未逐次复现|
|环境及外部资源|记录直接依赖版本，不打包.venv、.git、浏览器/账号状态或凭据。官方网页引用原样保留，没有本轮重抓完整网页|
|完整重新训练/账户回放|未执行；仅复算保存账本、预测点统计、集中度并检查CSV导出一致性|
|外部审阅、独立金融验证、真实成交|均未建立；交付由用户提交Pro后才会发生外部审阅|

原分钟经过第三方代理，历史送达、再分发或当前免费重下载许可未被证明；包内源资料不产生新增数据许可。金额一档容许为测量假设，未知不补中性。低仓位/缺失造成平稳以及少数大涨贡献均不是稳定alpha证据。

本次核验范围为ZIP结构、哈希/索引/覆盖及保存结果复算；未进行安全审计。未联网、未训练、未回测、未抽样、未提交真实订单。
"""
    for name, content, purpose in [("00_README_FIRST.md", readme, "阅读导航"), ("01_PRO_REVIEW_PROMPT.md", prompt, "可复制的独立审阅任务"), ("02_USER_REQUEST_AND_SCOPE.md", request, "用户要求与范围"), ("03_EVIDENCE_MAP.md", evidence, "问题至直接证据映射"), ("04_SCOPE_AND_LIMITS.md", limits, "纳入、排除和验证边界")]:
        add(name, content.encode("utf-8"), purpose)
    add("context/git_snapshot.json", encoded({"captured_at": stamp, "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "recent_commits": git("log", "-3", "--format=%h %s").splitlines(), "tracked_worktree_status": git("status", "--short", "--untracked-files=no").splitlines(), "note": "当前工作树研究文件以包内哈希为准；存在既有未提交变更，未暂存或提交。不是完整Git仓库快照。"}), "Git来源上下文")
    add("context/runtime.json", encoded({"python": platform.python_version(), "platform": platform.platform(), "captured_at": stamp, "versions": {n: version(n) for n in ("numpy", "pandas", "pyarrow", "matplotlib", "pytest", "tzdata")}}), "已使用环境版本")

    def write_archive() -> None:
        index = [{"path": name, "bytes": len(body), "sha256": sha(body), "purpose": purposes[name]} for name, body in sorted(members.items())]
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as pack:
            for name, body in sorted(members.items()):
                pack.writestr(name, body)
            pack.writestr("FILE_INDEX.csv", csv_bytes(index))

    write_archive()
    print("已构建临时审阅包，正在从独立解压副本复算原保存结果。", flush=True)
    with tempfile.TemporaryDirectory(prefix="分钟Pro包核验_", dir=DELIVERY) as temp:
        temp_root = Path(temp)
        if not temp_root.resolve().is_relative_to(DELIVERY.resolve()):
            raise ValueError("临时解压目录越界")
        extracted = temp_root / "package"
        extracted.mkdir()
        with zipfile.ZipFile(temporary_archive) as pack:
            for name in pack.namelist():
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("包内路径越界")
            pack.extractall(extracted)
        output_receipt = temp_root / "recomputation.json"
        proc = subprocess.run([sys.executable, "-X", "utf8", str(extracted / "verify_review_package.py"), "--root", str(extracted), "--output", str(output_receipt)], cwd=extracted, capture_output=True, encoding="utf-8")
        if proc.returncode:
            raise RuntimeError("副本复算失败：" + proc.stdout + proc.stderr)
        recomputation = json.loads(output_receipt.read_text(encoding="utf-8"))
        print("解压副本的16账户、364共同预测日和16账户集中度复算通过。", flush=True)
    verification = {"prepared_at": stamp, "status": "PREPARED_WITH_EXTRACTED_SAVED_RESULT_RECOMPUTATION", "snapshot_study": "510300_INTRADAY_PROCESS_INCREMENT_V1", "expected_frozen_files": len(frozen["files"]), "current_tree_files": len(list(REPORT.iterdir())), "inventory_sources": len(inventory), "recomputation": recomputation, "structural_validation": "正式ZIP在加入本回执及根索引后重新进行CRC/重名/逐文件哈希/覆盖检查，结果在ZIP旁交付回执。", "external_review_performed": False, "new_fits": 0, "new_accounts": 0, "new_random_draws": 0, "new_market_downloads": 0}
    add("05_VERIFICATION.json", encoded(verification), "解压副本保存结果复算回执")
    write_archive()

    with zipfile.ZipFile(temporary_archive) as pack:
        names = pack.namelist()
        if len(names) != len(set(names)) or pack.testzip() is not None:
            raise ValueError("ZIP重名或CRC失败")
        rows = list(csv.DictReader(io.StringIO(pack.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if {x["path"] for x in rows} != set(names) - {"FILE_INDEX.csv"}:
            raise ValueError("根索引覆盖失败")
        for row in rows:
            body = pack.read(row["path"])
            if len(body) != int(row["bytes"]) or sha(body) != row["sha256"]:
                raise ValueError("索引字节或哈希不符：" + row["path"])
        for original, item in source_map.items():
            if sha(pack.read(item["package_path"])) != item["sha256"]:
                raise ValueError("来源覆盖身份不符：" + original)
    if temporary_archive.stat().st_size > BYTE_LIMIT:
        raise ValueError("超出本地50MiB包预算")
    os.replace(temporary_archive, archive)
    receipt = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": "PASS_ZIP_STRUCTURE_AND_EXTRACTED_SAVED_RECOMPUTATION", "zip_path": str(archive), "zip_bytes": archive.stat().st_size, "zip_sha256": sha(archive.read_bytes()), "members": len(names), "indexed_members": len(rows), "index_excludes_itself": True, "frozen_files": len(frozen["files"]), "current_tree_files": len(list(REPORT.iterdir())), "inventory_sources": len(inventory), "source_path_mappings": len(source_map), "crc_pass": True, "duplicate_members": 0, "index_size_sha256_coverage_pass": True, "fresh_extracted_saved_result_recomputation": recomputation, "local_package_byte_budget": BYTE_LIMIT, "security_audit_performed": False, "external_pro_review_performed": False, "uploaded": False}
    (DELIVERY / (NAME + "_交付校验.json")).write_bytes(encoded(receipt))
    (DELIVERY / (NAME + ".sha256")).write_text(receipt["zip_sha256"] + "  " + archive.name + "\n", encoding="utf-8")
    (DELIVERY / (NAME + "_给Pro的审阅提示词.md")).write_text(prompt, encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "zip_path", "zip_bytes", "zip_sha256", "members", "indexed_members", "frozen_files", "inventory_sources")}, ensure_ascii=False))


if __name__ == "__main__":
    build()
