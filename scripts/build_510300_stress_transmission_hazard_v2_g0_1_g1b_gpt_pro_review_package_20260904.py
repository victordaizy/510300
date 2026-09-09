"""构建 510300 压力传导危险率 V2 的 G0.1/G1B 终局 GPT Pro 审阅包。

本脚本复用 2026-09-03 已验证的 ZIP 构建与结构检查实现，但更新权威状态为：
G0.1 通过、G1A 通过、G1B 因仅有两个合格时代而 NO_VIEW、正式 G2 未运行、
历史分支归档且禁止结果性救援。脚本只打包和检查交付结构，不进行任何模型拟合、
收益评价、安全审计、隐私扫描、秘密扫描、恶意文件扫描或脱敏。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).absolute().parents[1]
BASE_BUILDER_PATH = (
    ROOT
    / "scripts/build_510300_stress_transmission_hazard_v2_gpt_pro_full_review_package_20260903.py"
)
BASE_SPEC = importlib.util.spec_from_file_location(
    "stress_transmission_hazard_v2_package_base",
    BASE_BUILDER_PATH,
)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"无法加载基础构包器：{BASE_BUILDER_PATH}")
base = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = base
BASE_SPEC.loader.exec_module(base)


PACKAGE_BASENAME = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_G0_1_G1B_FINAL_"
    "GPT_PRO_REVIEW_20260904"
)
DELIVERABLES = ROOT / "deliverables"
PRIMARY_ATTACHMENT = Path(
    r"E:\CodexData\.codex\attachments"
    r"\8712825e-d589-4c9f-90dd-dcac55d9f808\pasted-text-1.txt"
)
SUPPLEMENTAL_ATTACHMENTS = {
    "02A_EXTERNAL_REVIEW_STOP_AND_ARCHIVE_V2.txt": Path(
        r"E:\CodexData\.codex\attachments"
        r"\d922d27c-1a10-4ba5-916d-c73f7bdd7167\pasted-text.txt"
    ),
    "02B_EXTERNAL_REVIEW_G1B_REQUIREMENTS.txt": Path(
        r"E:\CodexData\.codex\attachments"
        r"\2fa92235-a606-4816-a1a3-31cf66e7bd1e\pasted-text.txt"
    ),
}

AUTHORITATIVE_STATE = "G1B_NO_VIEW_INSUFFICIENT_EVALUABLE_ERAS_G2_NOT_RUN"
FINAL_STATUS_PATH = (
    "reports/research/"
    "510300_stress_transmission_hazard_v2_g1b_historical_final_status_v1.json"
)
G0_1_STATUS_PATH = (
    "reports/research/"
    "510300_stress_transmission_hazard_v2_g0_1_post_remediation_status_v1.json"
)
G0_1_MANIFEST_PATH = (
    "config/"
    "510300_stress_transmission_hazard_v2_g0_1_post_remediation_version_lock_v1_manifest.json"
)
G1B_EXECUTION_RECEIPT_PATH = (
    "reports/audit/"
    "510300_stress_transmission_hazard_v2_g1b_prequential_era_identifiability_v1_execution_receipt.json"
)

REQUIRED_NAVIGATION_FILES = (
    "00_README_FIRST.md",
    "01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md",
    "02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt",
    "02A_EXTERNAL_REVIEW_STOP_AND_ARCHIVE_V2.txt",
    "02B_EXTERNAL_REVIEW_G1B_REQUIREMENTS.txt",
    "03_SESSION_STOP_POINT_AND_AUTHORITY.md",
    "04_GATE_TIMELINE_AND_CURRENT_STATUS.md",
    "05_DATA_SCOPE_AND_COMPLETENESS.md",
    "06_DATA_FILE_INDEX.csv",
    "07_INCLUDED_FILE_INDEX.csv",
    "08_REPRODUCTION_AND_VERSION_MAP.md",
    "09_VERIFICATION_BOUNDARY.md",
    "10_MISSING_OR_UNRESOLVED_REFERENCE_INDEX.csv",
    "11_EXPLICIT_DATA_ROOT_INDEX.csv",
    "12_GIT_SNAPSHOT.md",
    "13_PACKAGE_METADATA.json",
)

NOT_PERFORMED = tuple(
    dict.fromkeys(
        (
            *base.NOT_PERFORMED,
            "FORMAL_G2_EXECUTION",
            "MODEL_TRAINING",
            "RETURN_EVALUATION",
            "PORTFOLIO_EVALUATION",
            "PAPER_OR_SHADOW_EXECUTION",
            "BROKER_CONNECTION",
            "ORDER_OR_LIVE_EXECUTION",
        )
    )
)


def configure_base() -> None:
    """把基础构包器定向到新的、不可覆盖的交付物。"""

    base.PACKAGE_BASENAME = PACKAGE_BASENAME
    base.INTERNAL_ROOT = PACKAGE_BASENAME
    base.ZIP_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
    base.SHA256_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
    base.RECEIPT_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
    base.UPLOAD_MESSAGE_PATH = (
        DELIVERABLES / f"{PACKAGE_BASENAME}_UPLOAD_MESSAGE.txt"
    )
    base.ORIGINAL_ATTACHMENT = PRIMARY_ATTACHMENT
    base.SNAPSHOT_AT = datetime.now(base.TIME_ZONE)
    base.REQUIRED_NAVIGATION_FILES = REQUIRED_NAVIGATION_FILES
    base.NOT_PERFORMED = NOT_PERFORMED
    base.EXPLICIT_DATA_ROOTS = {
        **base.EXPLICIT_DATA_ROOTS,
        "data/raw/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1": (
            "G1 历史修复使用的上交所、深交所停复牌原始资料及采集回执"
        ),
        "data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1": (
            "G1 历史修复的规范化停复牌区间、逐日证据、四态收益和重建特征"
        ),
    }


def load_json(relative: str) -> dict[str, Any]:
    return base.load_json(relative)


def validate_state() -> dict[str, Any]:
    """读取并严格核对终局状态；任何漂移都拒绝构包。"""

    g0_1 = load_json(G0_1_STATUS_PATH)
    final = load_json(FINAL_STATUS_PATH)
    execution = load_json(G1B_EXECUTION_RECEIPT_PATH)
    manifest = load_json(G0_1_MANIFEST_PATH)

    expected = {
        "G0_1_POST_REMEDIATION_VERSION_LOCK": "PASS",
        "G1A_COMMON_SAMPLE_EVENT_IDENTIFIABILITY": "PASS_B2_ONLY",
        "G1A_B2_IDENTIFIABLE_EVENTS": 31,
        "G1A_B2_NON_EVENT_RISK_DAYS": 1102,
        "G1A_B2_COMMON_SAMPLE_DAYS": 1279,
        "G1B_PREQUENTIAL_ERA_IDENTIFIABILITY": (
            "NO_VIEW_INSUFFICIENT_EVALUABLE_ERAS"
        ),
        "G1B_QUALIFIED_ERA_COUNT": 2,
        "G1B_REQUIRED_QUALIFIED_ERA_COUNT": 3,
        "G2_STRUCTURAL_INCREMENT": "NOT_RUN_BLOCKED_BY_G1B",
        "B3_FULL_MODEL": "NO_VIEW_31_LT_40",
        "HISTORICAL_BRANCH": "ARCHIVED_NO_RESCUE",
        "MODEL_TRAINED": False,
        "RETURN_EVALUATION": "NOT_ALLOWED",
        "PORTFOLIO_EVALUATION": "NOT_ALLOWED",
        "POSITION_IMPACT": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": final.get(key)}
        for key, value in expected.items()
        if final.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "G1B 终局状态发生漂移，拒绝构包："
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    expected_eras = {
        "ERA_1_2015_2017": 0,
        "ERA_2_2018_2020": 3,
        "ERA_3_2021_2023": 19,
        "ERA_4_2024_CUTOFF": 7,
    }
    if final.get("G1B_PREQUENTIAL_EVENTS_BY_ERA") != expected_eras:
        raise RuntimeError("G1B 逐时代事件分布发生漂移，拒绝构包")
    if g0_1.get("G0_1_POST_REMEDIATION_VERSION_LOCK") != "PASS":
        raise RuntimeError("G0.1 版本锁不再是 PASS，拒绝构包")
    if g0_1.get("locked_commit_sha") != (
        "723aa67b67a85bc60c7f808f00d3294edf114836"
    ):
        raise RuntimeError("G0.1 锁定提交发生漂移，拒绝构包")
    if execution.get("model_trained") is not False:
        raise RuntimeError("G1B 执行回执显示发生模型训练，拒绝构包")
    if manifest.get("formal_g2_run") is not False:
        raise RuntimeError("G0.1 manifest 显示正式 G2 已运行，拒绝构包")

    forbidden_outputs = (
        ROOT
        / "config/510300_stress_transmission_hazard_v2_g2_counterfactual_v1_manifest.json",
        ROOT
        / "reports/research/510300_stress_transmission_hazard_v2_g2_counterfactual_status_v1.json",
    )
    existing_forbidden = [str(path) for path in forbidden_outputs if path.exists()]
    if existing_forbidden:
        raise RuntimeError(
            f"检测到冻结或执行的反事实 G2 输出，拒绝构包：{existing_forbidden}"
        )

    return {
        "g0_1": g0_1,
        "g1b_final": final,
        "g1b_execution": execution,
        "g0_1_manifest": manifest,
    }


def collect_git_info() -> dict[str, Any]:
    branch = str(base.run_git(["branch", "--show-current"])).strip()
    head = str(base.run_git(["rev-parse", "HEAD"])).strip()
    log = str(
        base.run_git(
            [
                "log",
                "-15",
                "--format=%H%x09%ad%x09%s",
                "--date=iso-strict",
            ]
        )
    ).strip()
    return {
        "branch": branch,
        "head": head,
        "recent_commit_log": log,
        "note": (
            "V2 研究终局已在本地提交；包内索引另行记录每个实际文件在构包时的 "
            "Git 状态。工作区中与本任务无关的既有改动不属于本包范围。"
        ),
    }


def collect_sources(
    builder: Any,
) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    """收集目标文件，并用 G0.1 manifest 强制补齐锁定闭包。"""

    git_info, primary_bytes = base.collect_sources(builder)
    manifest = load_json(G0_1_MANIFEST_PATH)
    for item in manifest.get("required_git_files", []):
        relative = item.get("path") if isinstance(item, dict) else item
        if not relative:
            raise RuntimeError("G0.1 manifest 中存在无路径的 Git 文件记录")
        builder.add(relative, "G0.1 manifest 锁定的 Git 文件", required=True)
    for item in manifest.get("data_files", []):
        relative = item.get("path") if isinstance(item, dict) else None
        if not relative:
            raise RuntimeError("G0.1 manifest 中存在无路径的数据文件记录")
        builder.add(relative, "G0.1 manifest 锁定的数据文件", required=True)

    supplemental: dict[str, bytes] = {}
    for archive_name, path in SUPPLEMENTAL_ATTACHMENTS.items():
        if not path.is_file():
            raise RuntimeError(f"缺少补充专家附件：{path}")
        supplemental[archive_name] = path.read_bytes()
    return collect_git_info(), primary_bytes, supplemental


def manifest_coverage(builder: Any, state: dict[str, Any]) -> dict[str, Any]:
    manifest = state["g0_1_manifest"]
    required_git = {
        str(item["path"] if isinstance(item, dict) else item).replace("\\", "/")
        for item in manifest.get("required_git_files", [])
    }
    required_data = {
        str(item["path"]).replace("\\", "/")
        for item in manifest.get("data_files", [])
    }
    included = set(builder.sources)
    missing_git = sorted(required_git - included)
    missing_data = sorted(required_data - included)
    if missing_git or missing_data:
        raise RuntimeError(
            "G0.1 manifest 覆盖不完整："
            f"missing_git={len(missing_git)}, missing_data={len(missing_data)}"
        )
    return {
        "required_git_file_count": len(required_git),
        "required_data_file_count": len(required_data),
        "missing_git_file_count": 0,
        "missing_data_file_count": 0,
        "coverage": "COMPLETE",
    }


def readme_text(
    source_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
    state: dict[str, Any],
) -> str:
    final = state["g1b_final"]
    eras = final["G1B_PREQUENTIAL_EVENTS_BY_ERA"]
    return f"""# 510300 压力传导危险率 V2：G0.1/G1B 终局审阅包

这是 `510300_STRESS_TRANSMISSION_HAZARD_V2` 在历史分支关闭时的单一、自包含
GPT Pro 审阅包。它纳入协议、代码、配置、测试、状态、回执、审计表、目标数据、
G0.1 独立净工作树复跑证据，以及三份外部专家原文。

## 当前权威结论

- G0.1 版本锁：`{final['G0_1_POST_REMEDIATION_VERSION_LOCK']}`。
- G1A：`{final['G1A_COMMON_SAMPLE_EVENT_IDENTIFIABILITY']}`；B2 事件
  {final['G1A_B2_IDENTIFIABLE_EVENTS']}，非事件风险日
  {final['G1A_B2_NON_EVENT_RISK_DAYS']}，共同样本日
  {final['G1A_B2_COMMON_SAMPLE_DAYS']}。
- G1B：`{final['G1B_PREQUENTIAL_ERA_IDENTIFIABILITY']}`。
- G1B 合格时代：{final['G1B_QUALIFIED_ERA_COUNT']} / 
  {final['G1B_REQUIRED_QUALIFIED_ERA_COUNT']}。
- 逐时代可识别事件：{eras}。
- G2：`{final['G2_STRUCTURAL_INCREMENT']}`；未拟合模型。
- B3：`{final['B3_FULL_MODEL']}`。
- 历史分支：`{final['HISTORICAL_BRANCH']}`。
- 收益与组合评价：`NOT_ALLOWED`；仓位影响：0。

## 建议阅读顺序

1. `01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md`
2. `02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt`
3. `02A_EXTERNAL_REVIEW_STOP_AND_ARCHIVE_V2.txt`
4. `02B_EXTERNAL_REVIEW_G1B_REQUIREMENTS.txt`
5. `03_SESSION_STOP_POINT_AND_AUTHORITY.md`
6. `04_GATE_TIMELINE_AND_CURRENT_STATUS.md`
7. `project/docs/510300_STRESS_TRANSMISSION_HAZARD_V2_G0_1_POST_REMEDIATION_VERSION_LOCK_V1_20260904.md`
8. `project/reports/research/510300_stress_transmission_hazard_v2_g0_1_post_remediation_status_v1.json`
9. `project/docs/510300_STRESS_TRANSMISSION_HAZARD_V2_G1B_PREQUENTIAL_ERA_IDENTIFIABILITY_V1_20260904.md`
10. `project/reports/data_quality/510300_STRESS_TRANSMISSION_HAZARD_V2_G1B_PREQUENTIAL_ERA_IDENTIFIABILITY_V1.md`
11. `project/reports/research/510300_stress_transmission_hazard_v2_g1b_historical_final_status_v1.json`
12. `project/reports/research/510300_stress_transmission_hazard_v2_g1b_prequential_event_audit_v1.csv`
13. `project/reports/research/510300_stress_transmission_hazard_v2_g1b_quarterly_vintage_audit_v1.csv`
14. `project/reports/research/510300_stress_transmission_hazard_v2_g1b_prequential_era_audit_v1.csv`
15. `05_DATA_SCOPE_AND_COMPLETENESS.md`
16. `06_DATA_FILE_INDEX.csv`
17. `07_INCLUDED_FILE_INDEX.csv`
18. `08_REPRODUCTION_AND_VERSION_MAP.md`
19. `09_VERIFICATION_BOUNDARY.md`

## 包规模

- 项目源文件：{len(source_rows):,}。
- 项目源文件字节：{sum(int(row['bytes']) for row in source_rows):,}。
- 数据及数据型结果文件：{len(data_rows):,}。
- 显式目标数据根：{len(explicit_rows)}，均逐文件完整覆盖。
- 三份外部专家附件均以原始字节纳入并单独核对 SHA-256。

## 不可越过的边界

- 本包完成不等于 GPT Pro 已经审阅。
- G1B 的 `NO_VIEW` 是正式研究结果，不是等待填充的预测。
- 不允许为了通过门槛补造事件、降低事件门、修改 BAD10、改变时代或让事件进入自身训练。
- 不运行正式、反事实或全样本 G2；不训练模型；不读取组合收益。
- 可执行资产仍仅为 `510300.SH` 和 `CASH_CNY`，且本包不授权任何执行。
- 本次未做安全审计；结构检查边界见 `09_VERIFICATION_BOUNDARY.md`。
"""


def review_prompt_text() -> str:
    return """# 给 GPT Pro 专家的终局战略审阅任务

你是一名严厉、证据优先的量化研究负责人。请审阅本包中的
`510300_STRESS_TRANSMISSION_HAZARD_V2` 完整证据链，并决定下一步资源战略。
不要做安全审计，不要给买卖指令、目标仓位、券商连接或实盘方案。每项实证判断
必须引用包内相对路径、字段、函数、审计表或回执。

## 必须先接受的状态边界

1. G0.1 已在本地提交后从独立干净 worktree 复跑，精确复现 31 个 B2 事件、
   1,102 个非事件风险日、1,279 个共同样本日。
2. G1A 只证明共同样本事件数量达到 B2 门槛，不证明任何特征具有预测增量。
3. G1B 在读取实际允许字段值之前冻结，不拟合模型；实际逐时代事件数为
   0、3、19、7，只有 ERA3 和 ERA4 达到每时代至少 5 个事件。
4. 正式 G2 要求至少 3 个合格评价时代，而当前只有 2 个，所以 G2 没有合法晋级路径。
5. B3 只有 31 个事件，低于冻结的 40 个事件门槛。
6. 正式、反事实和全样本 G2 均未运行；模型、概率、预测指标、收益和组合指标均未产生。
7. 历史分支状态为 `ARCHIVED_NO_RESCUE`；经济假设仍为 `UNTESTED`。
8. 任何降低门槛、修改标签/期限/时代、用未来事件训练过去、事件跨边界、增加代理事件、
   查看收益后改规则或参数搜索，均不属于允许的“继续”。

## 必须完成的审阅

### A. G0.1 是否真正形成可复现版本锁

- 核对锁定提交、Git 文件 manifest、320 个外部数据文件、净工作树物化和复跑回执。
- 检查 31 / 1,102 / 1,279 和 0/5/19/7 的 G1A 年代分布是否精确复现。
- 区分“文件/版本可复现”与“经济结论有效”。

### B. G1B 是否正确执行原有时间规则

- 核对季度 vintage、严格 `< model_vintage` 的整事件成熟规则、正负两类训练可识别性、
  事件不可进入自身训练、未来训练禁止和固定时代边界。
- 重点复核为什么 ERA2 表面有 5 个 G1A 事件，但逐时序只能评价 3 个。
- 判断是否存在足以推翻终局状态的可证明 P0 实现错误；“希望运行 G2”不是 P0。

### C. 战略资源配置

请比较并排序：

- `ACCEPT_ARCHIVE_AND_SHIFT_RESOURCES`：接受历史分支关闭，转向其他独立 510300 机制。
- `REOPEN_ONLY_FOR_PROVABLE_P0_ERROR`：仅当包内证据能证明实现错误时更正，不做结果性救援。
- `FORWARD_DATA_OBSERVATORY_ONLY`：只继续前向收集，不生成模型信号或仓位，不回填为历史可得。
- `NEW_PREREGISTERED_V3`：只有真正新的机制和独立数据契约才另立版本，不能复用当前失败样本调规则。
- `DATA_PLATFORM_WORK`：投入点时成分、停复牌、公司行动、供应商冲突和可用时钟基础设施。
- `STOP_THIS_MECHANISM_FAMILY_PERMANENTLY`：若信息增益相对成本已经不足，永久停止。

对每条路线给出信息增益、数据/工程成本、统计功效、主要失败方式、前置条件、停止条件、
是否会污染当前冻结结论，以及建议资源占比。

## 强制输出格式

1. 一句话主裁决：从 `ACCEPT_ARCHIVE_AND_SHIFT_RESOURCES`、
   `REOPEN_ONLY_FOR_PROVABLE_P0_ERROR`、`NEW_PREREGISTERED_V3_WITH_NEW_EVIDENCE`、
   `INSUFFICIENT_EVIDENCE_FOR_STRATEGIC_DECISION` 中选择一个。
2. 证据可靠性表：G0.1、G1A、G1B、G2 禁入、B3 禁入、归档状态逐项列证据和不确定性。
3. P0/P1/P2：每条必须给包内路径、影响和最小动作；没有 P0 时明确写“未发现 P0”。
4. ERA2 独立复核：逐事件解释 `NO_VIEW` 与首个可识别训练 vintage。
5. 停止/继续/资源矩阵：对上述六条路线评分和排序。
6. 30/90/180 天路线图：每阶段给唯一允许输出、验收门、停止条件和资源预算。
7. 明确“不再做”清单：至少覆盖补造事件、换代理事件、降门、改 BAD10、改时代、
   全样本回看、反事实 G2、模型网格、先看收益、扩大可执行资产。
8. 最终建议：最多 10 项，严格分开“已验证事实”“推断”“需要新增证据”。
"""


def stop_point_text(state: dict[str, Any]) -> str:
    final = state["g1b_final"]
    return f"""# 会话停止点与权限边界

## 本轮实际完成

- 新增并执行 `G0_1_POST_REMEDIATION_VERSION_LOCK`。
- 在锁定提交的独立干净 worktree 中复跑 G1 修复，精确复现 31 / 1,102 / 1,279。
- 在读取实际允许字段值前冻结 `G1B_PREQUENTIAL_ERA_IDENTIFIABILITY`。
- 只读取冻结允许的七个物理字段，并仅由 `origin_date` 派生固定 `era_id`。
- 生成逐事件、季度 vintage、逐时代审计和追加式终局状态。

## 当前停止点

- `G1B_PREQUENTIAL_ERA_IDENTIFIABILITY = {final['G1B_PREQUENTIAL_ERA_IDENTIFIABILITY']}`
- `G2_STRUCTURAL_INCREMENT = {final['G2_STRUCTURAL_INCREMENT']}`
- `G3_MACRO_INCREMENT = {final['G3_MACRO_INCREMENT']}`
- `G4_THROUGH_G7 = {final['G4_THROUGH_G7']}`
- `HISTORICAL_BRANCH = {final['HISTORICAL_BRANCH']}`
- `ECONOMIC_HYPOTHESIS = {final['ECONOMIC_HYPOTHESIS']}`

## 明确没有发生

- `formal_g2_model_fit = false`
- `counterfactual_or_full_sample_g2 = NOT_ALLOWED`
- `model_trained = false`
- `return_evaluation = NOT_ALLOWED`
- `portfolio_evaluation = NOT_ALLOWED`
- `paper_or_shadow_execution = false`
- `broker_connection = false`
- `position_or_order_generated = false`
- `position_impact = 0`

“继续”在本包中只表示完成证据归档并请求外部战略审阅，不表示继续被关闭的历史模型分支。
"""


def gate_timeline_text(state: dict[str, Any]) -> str:
    final = state["g1b_final"]
    g0 = state["g0_1"]
    eras = final["G1B_PREQUENTIAL_EVENTS_BY_ERA"]
    return f"""# 门槛时间线与当前权威状态

## 1. 初始 V2 与原始 G1

V2 协议、BAD10、四态收益、M/F/T 和原始 G1 证据全部保留。原始共同样本只有
23 个可识别事件，因此曾得到 G1 `NO_VIEW`。后续只允许修复可证明的历史数据问题，
不允许改变标签、阈值、期限、覆盖门或模型。

## 2. 可证明的历史停牌修复

官方资料被规范化为 14,205 条停复牌区间，15,340 行证据映射使 14,771 行四态记录
获得合法状态提升。重新构建特征与 G1 后，B2 共同样本变为：

- 可识别事件：{final['G1A_B2_IDENTIFIABLE_EVENTS']}。
- 非事件风险日：{final['G1A_B2_NON_EVENT_RISK_DAYS']}。
- 共同样本日：{final['G1A_B2_COMMON_SAMPLE_DAYS']}。
- 原始固定时代分布：0 / 5 / 19 / 7。

因此 G1A 的 B2 数量门通过；B3 仍为 `{final['B3_FULL_MODEL']}`。

## 3. G0.1 事后修复版本锁

- 状态：`{g0['G0_1_POST_REMEDIATION_VERSION_LOCK']}`。
- 锁定提交：`{g0['locked_commit_sha']}`。
- 锁定分支：`{g0['locked_branch']}`。
- 独立干净 worktree 复跑：精确复现 31 / 1,102 / 1,279 和 0 / 5 / 19 / 7。
- 正式 G2、模型、收益、组合和仓位均未触及。

## 4. G1B 前序预测时代可识别性

G1B 在实际行值读取前冻结。季度 vintage 只允许使用整事件 horizon 严格早于
模型 vintage 的历史，并要求训练侧至少有一个成熟正事件和一个成熟负风险日。

- ERA1：{eras['ERA_1_2015_2017']} 个逐时序事件。
- ERA2：{eras['ERA_2_2018_2020']} 个逐时序事件。
- ERA3：{eras['ERA_3_2021_2023']} 个逐时序事件。
- ERA4：{eras['ERA_4_2024_CUTOFF']} 个逐时序事件。
- 合格时代：{final['G1B_QUALIFIED_ERA_COUNT']}，要求至少
  {final['G1B_REQUIRED_QUALIFIED_ERA_COUNT']}。

终局：`{final['G1B_PREQUENTIAL_ERA_IDENTIFIABILITY']}`。

## 5. 顺序门终止

- G2：`{final['G2_STRUCTURAL_INCREMENT']}`。
- G3：`{final['G3_MACRO_INCREMENT']}`。
- G4-G7：`{final['G4_THROUGH_G7']}`。
- 历史分支：`{final['HISTORICAL_BRANCH']}`。

不得用早期状态文件覆盖这一追加式终局状态。
"""


def data_scope_text(
    data_rows: list[dict[str, Any]], explicit_rows: list[dict[str, Any]]
) -> str:
    explicit_lines = "\n".join(
        f"- `{row['logical_root']}`：{row['description']}；"
        f"{row['file_count']:,} 文件，{row['bytes']:,} 字节，{row['coverage']}。"
        for row in explicit_rows
    )
    direct_lines = "\n".join(
        f"- `{path}`：{description}。"
        for path, description in base.EXPLICIT_DIRECT_INPUTS.items()
    )
    return f"""# 数据范围与完整覆盖

## 范围定义

“本次 V2 包括数据”指：V2 目标专属原始、整理和派生数据目录全部文件；G0.1
manifest 锁定的全部 320 个外部数据文件；协议、代码和回执直接依赖的历史输入；
以及本次产生的数据型审计表。不会递归吞入其他策略、旧备份、虚拟环境、缓存、
历史交付包、pytest 临时目录或独立 worktree 的重复副本。

## 显式目标数据根

{explicit_lines}

## 强制纳入的直接历史输入

{direct_lines}

## 数据索引

- 数据及数据型结果文件：{len(data_rows):,}。
- 数据总字节：{sum(int(row['bytes']) for row in data_rows):,}。
- 每个文件的逻辑路径、大小、SHA-256、Git 状态和纳入理由见 `06_DATA_FILE_INDEX.csv`。
- 所有项目源文件见 `07_INCLUDED_FILE_INDEX.csv`。
- G0.1 manifest 中的 Git 与数据路径另有构包时强制覆盖检查，缺任一项即失败。

## 解释边界

文件覆盖完整不证明数据语义或经济结论正确。缺失、`NO_VIEW`、`NOT_ALLOWED`、
`BLOCKED` 和未运行状态不得解释为零收益、零风险、失败收益或空白待补。
"""


def reproduction_text() -> str:
    return r"""# 复现入口与版本地图

## 固定提交

- G0.1 锁定提交：`723aa67b67a85bc60c7f808f00d3294edf114836`。
- G0.1 状态提交：`53834701919eff757810620f7454fde03c370c06`。
- G1B 冻结提交：`2e26bc1c4f31f2ab3ad32072a65d674d6d737fb5`。
- G1B 终局提交：`4afc3f2a63f1ec70e7aa7a8e1ec320b151875349`。

各 manifest 内记录自身冻结文件和输入 SHA-256。必须按追加顺序解释，不能把旧失败
版本与修正版混成一次运行，也不能用构包时 HEAD 改写上述研究提交身份。

## 审计性测试入口

所有命令均在 ZIP 解压后的 `project/` 目录、Windows PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests\test_510300_stress_transmission_hazard_v2_g0_1_version_lock_v1.py `
  tests\test_510300_stress_transmission_hazard_v2_g1b_prequential_era_identifiability_v1.py
```

该测试只验证 G0.1/G1B 契约和实现，不授权或触发 G2。

## 构包器复现

只读盘点：

```powershell
.\.venv\Scripts\python.exe scripts\build_510300_stress_transmission_hazard_v2_g0_1_g1b_gpt_pro_review_package_20260904.py --inventory-only
```

正式构包要求同名交付物尚不存在，以防静默覆盖：

```powershell
.\.venv\Scripts\python.exe scripts\build_510300_stress_transmission_hazard_v2_g0_1_g1b_gpt_pro_review_package_20260904.py
```

## 禁止性边界

不要执行 G2 counterfactual 构建器，不要训练 B1/B2/B3，不要读取收益或组合输出。
若专家发现可证明的 P0，实现修正必须另立版本、重新冻结，并保持本终局不可变。
"""


def verification_boundary_text() -> str:
    performed = "\n".join(f"- `{item}=true`" for item in base.STRUCTURAL_CHECKS)
    performed += "\n- `SUPPLEMENTAL_ATTACHMENT_IDENTITY_CHECK=true`"
    performed += "\n- `G0_1_MANIFEST_COVERAGE_CHECK=true`"
    skipped = "\n".join(f"- `{item}=false`" for item in NOT_PERFORMED)
    return f"""# 交付检查边界

用户明确要求减少不必要审计并且“不需要安全审计”。本次只做交付所需的结构检查，
不把这些检查描述为安全、隐私、秘密、恶意文件、脱敏或研究正确性验证。

## 已执行的结构检查

{performed}

这些检查只说明：声明文件已索引、G0.1 manifest 路径全部入包、ZIP 成员可读、
CRC 正常、没有重复成员、索引大小一致、三份外部附件字节身份一致、整包有 SHA-256。

## 明确未执行

{skipped}

因此接收方应把本包视为用户授权的未脱敏研究快照。包完成不代表 GPT Pro 已审阅，
不代表数据和代码经济含义正确，也不授权 Paper/Shadow、仓位、订单、券商或实盘。
"""


def build_generated_files(
    builder: Any,
    source_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    explicit_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, str]],
    git_info: dict[str, Any],
    primary_bytes: bytes,
    supplemental: dict[str, bytes],
    state: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, bytes]:
    attachment_metadata = {
        "02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt": {
            "source_path": str(PRIMARY_ATTACHMENT),
            "bytes": len(primary_bytes),
            "sha256": hashlib.sha256(primary_bytes).hexdigest(),
        }
    }
    for archive_name, payload in supplemental.items():
        attachment_metadata[archive_name] = {
            "source_path": str(SUPPLEMENTAL_ATTACHMENTS[archive_name]),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    metadata = {
        "package_id": PACKAGE_BASENAME,
        "created_at": base.SNAPSHOT_AT.isoformat(),
        "purpose": "GPT_PRO_EXPERT_FINAL_STRATEGIC_REVIEW",
        "scope": "FULL_V2_G0_1_G1B_FINAL_WITH_TARGET_DATA_AND_DIRECT_INPUTS",
        "program_id": "510300_STRESS_TRANSMISSION_HAZARD_V2",
        "authoritative_current_state": AUTHORITATIVE_STATE,
        "g0_1": "PASS",
        "g1a": "PASS_B2_ONLY_31_EVENTS_1102_NEGATIVE_DAYS_1279_COMMON_DAYS",
        "g1b": "NO_VIEW_INSUFFICIENT_EVALUABLE_ERAS_2_LT_3",
        "g2": "NOT_RUN_BLOCKED_BY_G1B",
        "b3": "NO_VIEW_31_LT_40",
        "historical_branch": "ARCHIVED_NO_RESCUE",
        "economic_hypothesis": "UNTESTED",
        "model_trained": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
        "source_file_count": len(source_rows),
        "source_bytes": sum(int(row["bytes"]) for row in source_rows),
        "data_file_count": len(data_rows),
        "data_bytes": sum(int(row["bytes"]) for row in data_rows),
        "explicit_data_root_count": len(explicit_rows),
        "explicit_data_root_coverage": "COMPLETE",
        "g0_1_manifest_coverage": coverage,
        "missing_or_unresolved_reference_count": len(missing_rows),
        "external_attachments": attachment_metadata,
        "structural_checks": [
            *base.STRUCTURAL_CHECKS,
            "SUPPLEMENTAL_ATTACHMENT_IDENTITY_CHECK",
            "G0_1_MANIFEST_COVERAGE_CHECK",
        ],
        "not_performed": list(NOT_PERFORMED),
        "security_audit": False,
        "privacy_scan": False,
        "secret_scan": False,
        "malware_scan": False,
        "redaction": False,
        "fresh_extraction_replay": False,
        "live_trading_authorized": False,
        "git": git_info,
    }

    generated: dict[str, bytes] = {
        "00_README_FIRST.md": readme_text(
            source_rows, data_rows, explicit_rows, state
        ).encode("utf-8"),
        "01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md": review_prompt_text().encode(
            "utf-8"
        ),
        "02_ORIGINAL_GPT_PRO_REVIEW_AND_V2_SPEC.txt": primary_bytes,
        **supplemental,
        "03_SESSION_STOP_POINT_AND_AUTHORITY.md": stop_point_text(state).encode(
            "utf-8"
        ),
        "04_GATE_TIMELINE_AND_CURRENT_STATUS.md": gate_timeline_text(state).encode(
            "utf-8"
        ),
        "05_DATA_SCOPE_AND_COMPLETENESS.md": data_scope_text(
            data_rows, explicit_rows
        ).encode("utf-8"),
        "06_DATA_FILE_INDEX.csv": base.csv_bytes(
            data_rows,
            [
                "archive_path",
                "workspace_logical_path",
                "category",
                "bytes",
                "sha256",
                "git_state",
                "inclusion_reasons",
            ],
        ),
        "07_INCLUDED_FILE_INDEX.csv": base.csv_bytes(
            source_rows,
            [
                "archive_path",
                "workspace_logical_path",
                "category",
                "bytes",
                "sha256",
                "git_state",
                "inclusion_reasons",
            ],
        ),
        "08_REPRODUCTION_AND_VERSION_MAP.md": reproduction_text().encode("utf-8"),
        "09_VERIFICATION_BOUNDARY.md": verification_boundary_text().encode("utf-8"),
        "10_MISSING_OR_UNRESOLVED_REFERENCE_INDEX.csv": base.csv_bytes(
            missing_rows, ["reference", "reasons", "interpretation"]
        ),
        "11_EXPLICIT_DATA_ROOT_INDEX.csv": base.csv_bytes(
            explicit_rows,
            ["logical_root", "description", "file_count", "bytes", "coverage"],
        ),
        "12_GIT_SNAPSHOT.md": base.git_snapshot_text(
            git_info, source_rows
        ).encode("utf-8"),
        "13_PACKAGE_METADATA.json": (
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    missing_navigation = set(REQUIRED_NAVIGATION_FILES) - set(generated)
    if missing_navigation:
        raise RuntimeError(f"缺少导航文件：{sorted(missing_navigation)}")
    return generated


def verify_zip(
    source_rows: list[dict[str, Any]],
    primary_bytes: bytes,
    supplemental: dict[str, bytes],
) -> dict[str, Any]:
    verification = base.verify_zip(source_rows, primary_bytes)
    with base.zipfile.ZipFile(base.ZIP_PATH, "r") as archive:
        supplemental_results = {}
        for archive_name, expected in supplemental.items():
            member_name = f"{PACKAGE_BASENAME}/{archive_name}"
            actual = archive.read(member_name)
            matches = actual == expected
            if not matches:
                raise RuntimeError(f"ZIP 内补充附件身份不一致：{archive_name}")
            supplemental_results[archive_name] = {
                "identity_match": True,
                "sha256": hashlib.sha256(actual).hexdigest(),
            }
    verification["supplemental_attachment_identity"] = supplemental_results
    verification["supplemental_attachment_identity_match"] = all(
        item["identity_match"] for item in supplemental_results.values()
    )
    return verification


def upload_message(zip_sha256: str, zip_bytes: int) -> str:
    return f"""请将这个 ZIP 作为一个整体上传给 GPT Pro：

{base.ZIP_PATH.name}

大小：{zip_bytes:,} 字节
SHA-256：{zip_sha256}

上传后请直接发送以下指令：

请先阅读压缩包根目录的 00_README_FIRST.md，再严格执行
01_GPT_PRO_STRATEGIC_REVIEW_PROMPT.md。请以 G0.1 已通过、G1A B2 数量门已通过、
G1B 因只有 2/3 个合格时代而 NO_VIEW、正式 G2 未运行、历史分支已归档且禁止救援
为权威停止点。请依据包内路径给出批判性证据审阅、P0/P1/P2、资源路线矩阵以及
30/90/180 天战略；不要做安全审计，不要给买卖、仓位、券商或实盘指令。
"""


def inventory_payload(
    builder: Any,
    explicit_rows: list[dict[str, Any]],
    state: dict[str, Any],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    category_counts = Counter(base.classify_source(path) for path in builder.sources)
    data_count = sum(
        1
        for relative in builder.sources
        if relative.startswith("data/")
        or base.classify_source(relative) == "TARGET_DERIVED_DATA_OR_LEDGER"
    )
    return {
        "status": "PASS_INVENTORY_ONLY_NO_PACKAGE_WRITTEN",
        "package_id": PACKAGE_BASENAME,
        "authoritative_current_state": AUTHORITATIVE_STATE,
        "source_file_count": len(builder.sources),
        "source_bytes": builder.total_bytes,
        "data_file_count": data_count,
        "explicit_data_roots": explicit_rows,
        "g0_1_manifest_coverage": coverage,
        "missing_or_unresolved_reference_count": len(builder.missing_references),
        "missing_or_unresolved_references": sorted(builder.missing_references),
        "external_attachment_count": 1 + len(SUPPLEMENTAL_ATTACHMENTS),
        "g1b": state["g1b_final"]["G1B_PREQUENTIAL_ERA_IDENTIFIABILITY"],
        "formal_g2_executed": False,
        "model_trained": False,
        "security_audit": False,
    }


def build_package() -> dict[str, Any]:
    output_paths = (
        base.ZIP_PATH,
        base.SHA256_PATH,
        base.RECEIPT_PATH,
        base.UPLOAD_MESSAGE_PATH,
    )
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        raise RuntimeError(f"交付物已存在，拒绝覆盖：{existing_outputs}")

    builder = base.PackageBuilder()
    git_info, primary_bytes, supplemental = collect_sources(builder)
    explicit_rows = base.explicit_data_stats(builder)
    state = validate_state()
    coverage = manifest_coverage(builder, state)
    states = base.tracked_state_map(builder)
    source_rows = base.source_index_rows(builder, states)
    data_rows = base.data_index_rows(source_rows)
    missing_rows = base.missing_reference_rows(builder)
    generated = build_generated_files(
        builder,
        source_rows,
        data_rows,
        explicit_rows,
        missing_rows,
        git_info,
        primary_bytes,
        supplemental,
        state,
        coverage,
    )
    base.write_zip(builder, generated)
    verification = verify_zip(source_rows, primary_bytes, supplemental)
    zip_sha256 = base.sha256_file(base.ZIP_PATH)
    zip_bytes = base.ZIP_PATH.stat().st_size

    receipt = {
        "status": (
            "PASS_G0_1_G1B_FINAL_PACKAGE_CRC_DUPLICATE_INDEX_DATA_"
            "MANIFEST_ATTACHMENT_AND_WHOLE_ZIP_SHA256"
        ),
        "package_id": PACKAGE_BASENAME,
        "created_at": base.SNAPSHOT_AT.isoformat(),
        "zip_path": base.ZIP_PATH.relative_to(ROOT).as_posix(),
        "zip_bytes": zip_bytes,
        "zip_sha256": zip_sha256,
        "source_file_count": len(source_rows),
        "source_bytes": sum(int(row["bytes"]) for row in source_rows),
        "data_file_count": len(data_rows),
        "data_bytes": sum(int(row["bytes"]) for row in data_rows),
        "generated_navigation_file_count": len(generated),
        "explicit_data_root_count": len(explicit_rows),
        "explicit_data_root_coverage": "COMPLETE",
        "g0_1_manifest_coverage": coverage,
        "missing_or_unresolved_reference_count": len(missing_rows),
        "authoritative_current_state": AUTHORITATIVE_STATE,
        "g0_1": "PASS",
        "g1a": "PASS_B2_ONLY",
        "g1b": "NO_VIEW_INSUFFICIENT_EVALUABLE_ERAS",
        "qualified_era_count": 2,
        "required_qualified_era_count": 3,
        "formal_g2_executed": False,
        "counterfactual_g2_executed": False,
        "model_trained": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED",
        "historical_branch": "ARCHIVED_NO_RESCUE",
        "position_impact": 0,
        "verification": verification,
        "structural_checks": [
            *base.STRUCTURAL_CHECKS,
            "SUPPLEMENTAL_ATTACHMENT_IDENTITY_CHECK",
            "G0_1_MANIFEST_COVERAGE_CHECK",
        ],
        "not_performed": list(NOT_PERFORMED),
        "security_audit": False,
        "privacy_scan": False,
        "secret_scan": False,
        "malware_scan": False,
        "redaction": False,
        "fresh_extraction_replay": False,
    }

    base.atomic_write_bytes(
        base.SHA256_PATH,
        f"{zip_sha256}  {base.ZIP_PATH.name}\n".encode("ascii"),
    )
    base.atomic_write_bytes(
        base.RECEIPT_PATH,
        (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    base.atomic_write_bytes(
        base.UPLOAD_MESSAGE_PATH,
        upload_message(zip_sha256, zip_bytes).encode("utf-8"),
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="只读盘点目标范围，不计算逐文件哈希，也不生成交付物。",
    )
    return parser.parse_args()


def main() -> int:
    configure_base()
    args = parse_args()
    if args.inventory_only:
        builder = base.PackageBuilder()
        _, _, _ = collect_sources(builder)
        explicit_rows = base.explicit_data_stats(builder)
        state = validate_state()
        coverage = manifest_coverage(builder, state)
        print(
            json.dumps(
                inventory_payload(builder, explicit_rows, state, coverage),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    receipt = build_package()
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
