"""构建 A 股 Alpha/强 Beta 探索阶段的 GPT 独立审阅 ZIP。

本脚本不重新运行任何策略、不修改冻结协议、不把未完成研究改写为结果。
它只收集本轮策略报告、关键回执、协议、实现、测试和小型结果账本，
并生成阅读说明、目标变更说明、SHA-256 清单、隐私扫描与解压复核回执。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
PACKAGE_BASENAME = "A_SHARE_ALPHA_RESEARCH_GPT_REVIEW_20260824_ORJ_V2_FINAL"
INTERNAL_ROOT = "A_SHARE_ALPHA_RESEARCH_GPT_REVIEW"
ZIP_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
SHA256_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
RECEIPT_PATH = DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
TIME_ZONE = ZoneInfo("Asia/Shanghai")
EVIDENCE_CUTOFF = "2026-08-14"
MAX_SELECTED_SOURCE_BYTES = 2_000_000

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".py",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}

STRATEGY_TOKENS = (
    "extreme_compression_resolution_alpha_discovery_v2",
    "extreme_compression_conditional_alpha_discovery_v2_1",
    "extreme_compression_forecast_origin_rank_audit_v2_2",
    "extreme_compression_v2_v2_1_terminal_freeze",
    "pit_industry_neutral_residual_trend_v1",
    "monthly_lottery_max_effect_v1",
    "cross_sectional_return_seasonality_v1",
    "monthly_turnover_premium_v1",
    "market_price_delay_premium_v1",
    "ovp_daily_ivol_exclusion_v1",
    "rmb_fxv_positive_beta_exclusion_v1",
    "official_report_overnight_reaction_drift_v1",
    "official_report_orj_net_excess_execution_v2",
    "official_nonoverlapping_upward_profit_forecast_revision_v1",
    "official_unconditional_full_cash_tender_spread_v1",
    "official_partial_cash_tender_20x20_v1",
    "official_cash_option",
    "cb_priority_allocation_one_lot",
)

DATA_ROOT_NAMES = (
    "a_share_hs_extreme_compression_resolution_alpha_discovery_v2",
    "a_share_hs_extreme_compression_conditional_alpha_discovery_v2_1",
    "a_share_hs_extreme_compression_forecast_origin_rank_audit_v2_2",
    "a_share_hs_pit_industry_neutral_residual_trend_v1",
    "a_share_hs_monthly_lottery_max_effect_v1",
    "a_share_hs_cross_sectional_return_seasonality_v1",
    "a_share_hs_monthly_turnover_premium_v1",
    "a_share_hs_market_price_delay_premium_v1",
    "a_share_hs_ovp_daily_ivol_exclusion_v1",
    "a_share_hs_rmb_fxv_positive_beta_exclusion_v1",
    "a_share_hs_rmb_fxv_positive_beta_exclusion_v1_0_0_invalid_filter_mask",
    "a_share_hs_rmb_fxv_positive_beta_exclusion_v1_0_1",
    "a_share_hs_official_report_overnight_reaction_drift_v1",
    "a_share_hs_official_report_overnight_reaction_drift_v1_0_1",
    "a_share_hs_official_report_orj_net_excess_execution_v2",
    "a_share_hs_official_nonoverlapping_upward_profit_forecast_revision_v1",
    "a_share_hs_official_unconditional_full_cash_tender_spread_v1",
    "a_share_hs_official_partial_cash_tender_20x20_v1",
    "a_share_hs_official_cash_option_rights_v1",
    "a_share_hs_cb_priority_allocation_one_lot_v1",
    "a_share_hs_cb_priority_allocation_one_lot_v1_0_1",
    "a_share_hs_cb_priority_allocation_one_lot_v1_0_2",
    "a_share_hs_cb_priority_allocation_one_lot_v1_0_3",
)


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_stream(handle: Any) -> str:
    """分块计算二进制流 SHA-256。"""

    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    """以 UTF-8 原子写入文本。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    """以稳定格式原子写入 JSON。"""

    write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def safe_member(value: str) -> str:
    """拒绝绝对路径、空路径和路径穿越。"""

    member = PurePosixPath(value.replace("\\", "/"))
    if member.is_absolute() or not member.parts or ".." in member.parts:
        raise ValueError(f"非法包内路径：{value}")
    return member.as_posix()


def iter_formal_files(root: Path) -> Iterable[Path]:
    """枚举正式文件，排除缓存、临时文件和事件逐页长文本。"""

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(
            part in {"__pycache__", ".pytest_cache", ".venv"}
            or part.upper().endswith("_EVENTS")
            or part.endswith("_events")
            for part in relative_parts
        ):
            continue
        if path.suffix.lower() in {".pyc", ".pyo", ".tmp"}:
            continue
        yield path


class PackageBuilder:
    """维护源文件到包内成员的唯一映射，并执行文本脱敏。"""

    def __init__(self) -> None:
        self.mappings: dict[str, Path] = {}
        self.redaction_counts: Counter[str] = Counter()
        self.provenance: list[dict[str, Any]] = []
        self.large_data_exclusions: list[str] = []

    def add(self, source: Path, destination: str | None = None) -> None:
        source = source.resolve()
        source.relative_to(ROOT.resolve())
        if not source.is_file():
            raise FileNotFoundError(f"缺少打包源文件：{source}")
        if destination is None:
            destination = source.relative_to(ROOT).as_posix()
        destination = safe_member(destination)
        existing = self.mappings.get(destination)
        if existing is not None and existing != source:
            raise ValueError(f"包内目标冲突：{destination}")
        self.mappings[destination] = source

    @staticmethod
    def _redact(value: str) -> tuple[str, list[str]]:
        """删除本机目录和用户名；不改研究数值与状态。"""

        project = str(ROOT)
        home = str(Path.home())
        username = Path.home().name
        replacements = [
            ("PROJECT_ROOT_JSON_ESCAPED", project.replace("\\", "\\\\"), "<PROJECT_ROOT>"),
            ("PROJECT_ROOT_LITERAL", project, "<PROJECT_ROOT>"),
            ("PROJECT_ROOT_FORWARD", project.replace("\\", "/"), "<PROJECT_ROOT>"),
            ("USER_HOME_JSON_ESCAPED", home.replace("\\", "\\\\"), "<USER_HOME>"),
            ("USER_HOME_LITERAL", home, "<USER_HOME>"),
            ("USER_HOME_FORWARD", home.replace("\\", "/"), "<USER_HOME>"),
            ("USERNAME_LITERAL", username, "<REDACTED_USER>"),
        ]
        output = value
        applied: list[str] = []
        for rule_id, old, new in replacements:
            if old and old in output:
                output = output.replace(old, new)
                applied.append(rule_id)
        return output, applied

    def copy_all(self, package_root: Path) -> None:
        for destination, source in sorted(self.mappings.items()):
            target = package_root / Path(destination)
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha256_file(source)
            redacted = False
            if source.suffix.lower() in TEXT_SUFFIXES:
                raw = source.read_bytes()
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    shutil.copy2(source, target)
                else:
                    clean, rules = self._redact(text)
                    target.write_text(clean, encoding="utf-8")
                    for rule in rules:
                        self.redaction_counts[rule] += 1
                    redacted = bool(rules)
            else:
                shutil.copy2(source, target)
            self.provenance.append(
                {
                    "source_relative_path": source.relative_to(ROOT).as_posix(),
                    "package_member": destination,
                    "source_size_bytes": source.stat().st_size,
                    "source_sha256": source_hash,
                    "package_size_bytes": target.stat().st_size,
                    "package_sha256": sha256_file(target),
                    "text_redacted": redacted,
                }
            )


def collect_sources(builder: PackageBuilder) -> None:
    """按研究家族收集最小必要证据；不复制原始行情和公告 PDF。"""

    for base_name in ("config", "research", "scripts", "tests", "reports"):
        base = ROOT / base_name
        for path in iter_formal_files(base):
            relative_lower = path.relative_to(ROOT).as_posix().lower()
            include = any(token in relative_lower for token in STRATEGY_TOKENS)
            if "a_share_pead_data_feasibility_20260823" in relative_lower:
                include = True
            if not include:
                continue
            if path.stat().st_size > MAX_SELECTED_SOURCE_BYTES:
                builder.large_data_exclusions.append(path.relative_to(ROOT).as_posix())
                continue
            builder.add(path)

    builder.add(ROOT / "requirements.txt", "environment/requirements.txt")
    builder.add(ROOT / "pytest.ini", "environment/pytest.ini")
    builder.add(Path(__file__), "tools/build_review_package.py")

    for layer in ("staging", "curated"):
        base = ROOT / "data" / layer
        for root_name in DATA_ROOT_NAMES:
            source_root = base / root_name
            if not source_root.is_dir():
                continue
            for path in sorted(source_root.iterdir()):
                if not path.is_file():
                    continue
                if path.name.lower().startswith("attempt_"):
                    continue
                if path.stat().st_size > MAX_SELECTED_SOURCE_BYTES:
                    builder.large_data_exclusions.append(path.relative_to(ROOT).as_posix())
                    continue
                builder.add(path)

    required_high_value_files = (
        "data/curated/a_share_hs_cb_priority_allocation_one_lot_v1/discovery_candidate_set.parquet",
        "data/curated/a_share_hs_cb_priority_allocation_one_lot_v1_0_2/combined_official_positive_matches.parquet",
        "data/curated/a_share_hs_official_partial_cash_tender_20x20_v1/final_lifecycle_outcomes_v1.parquet",
        "data/curated/a_share_hs_official_partial_cash_tender_20x20_v1/pre_market_sufficiency_stop_v1_receipt.json",
        "data/curated/a_share_hs_official_cash_option_rights_v1/eligibility_outcomes_v1.parquet",
        "data/curated/a_share_hs_official_cash_option_rights_v1/cash_option_terms_v1.parquet",
        "data/staging/a_share_hs_official_report_orj_net_excess_execution_v2/event_outcomes.parquet",
    )
    for relative in required_high_value_files:
        builder.add(ROOT / relative)


def readme_text(source_count: int) -> str:
    return f"""# A股超额收益探索：GPT 独立审阅包

## 当前判断

- 生成日期：2026-08-24（Asia/Shanghai）
- 研究数据统一截止：{EVIDENCE_CUTOFF}
- 当前总状态：`NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET`
- 当前动作：`RESEARCH_PAUSED_FOR_EXTERNAL_REVIEW`
- 包内源证据文件：{source_count} 个，另含本包说明、校验与来源索引。
- 研究边界：全部属于历史发现/验收或数据可行性阶段；没有 Shadow、仓位映射、订单或实盘授权。

## 最新目标函数（覆盖今后的新研究，不追溯改写旧协议）

目标只剩一个：找到在点时、无未来泄漏、考虑可实现成本之后仍有正净超额收益的策略。

以下不再作为硬门槛：

- 高胜率；
- 高盈亏比或高利润因子；
- 高单笔利润幅度；
- 高频率；
- 大容量或高流动性。

低频、低容量、小幅优势、低胜率都可以；关键是总期望为正、超额来自可解释且可复核的规则，并能经受时间外验证。旧研究已经冻结，不能因为目标变化直接从“拒绝”改成“通过”；只能把其中证据作为新版本立项依据。

## 建议阅读顺序

1. `00_README_FIRST.md`
2. `01_GPT_REVIEW_PROMPT.md`
3. `02_STRATEGY_RESULTS_AND_CONCLUSIONS.md`
4. `03_OBJECTIVE_CHANGE_EXCLUSIONS_AND_BOUNDARIES.md`
5. 先读 ORJ V2 历史执行报告，再读 ORJ 父研究、低换手与 MAX 的 `reports/research` 报告。
6. 再看对应 `config`、`research`、`scripts`、`tests` 与小型结果账本。
7. 最后使用 `07_SOURCE_PROVENANCE.csv` 和 `04_FILE_MANIFEST_SHA256.csv` 追溯文件。

## 包内最重要的三点

1. 目前没有任何方向已经证明是可交易 Alpha；不要把“13/14”、正点估计或事件级平均值直接升级成交易结论。
2. ORJ 已完成独立 V2 历史执行测验：压力成本后行业日均净超额 -1.016%，沪深300全收益净超额 -0.564%，两条路径均拒绝；基础成本宽基点估计虽为 +0.233%，但区间跨零且前半段为负。
3. 可转债原股东一手优先配售研究尚未完成：939 个候选中 886 个有官方正面公告证据，895 份 PDF 已提取文本，但条款最终裁决、上市日与行情收益尚未完成。

## 如何验证

解压后在本目录执行：

```powershell
python .\verify_package.py
```

它会检查清单中每个文件的大小和 SHA-256，并拒绝额外文件。ZIP 外部另附 `.sha256` 与构建回执。
"""


def review_prompt_text() -> str:
    return """# 给 GPT 的独立审阅提示词

你是一名独立的 A 股量化研究负责人。请完整审阅本 ZIP，目标不是替历史回测调参，也不是寻找一个看起来最漂亮的数字，而是决定下一次最小、最有信息价值的研究应该做什么。

## 最新目标

寻找考虑真实可实现成本后仍有正净超额收益的 Alpha 或强 Beta。低频、低容量、低流动性、小幅利润、低胜率、低盈亏比都允许；不要求高胜率、高盈亏比、高利润幅度或高频。仍必须避免未来泄漏、幸存者偏差、结果后改规则和把缺失数据当作零。

## 不允许的做法

1. 不得回头修改旧版冻结协议并宣称旧研究通过。
2. 不得从大量窗口、阈值、年份或行业中事后挑赢家。
3. 不得把历史预测相关性直接叫作可交易 Alpha。
4. 不得把 `NO_VIEW`、网络失败、条款未裁决或行情未完成解释成负收益或零收益。
5. 不得输出仓位、订单或实盘建议。

## 请回答

1. 用“可直接淘汰 / 证据可复用但需真正不同的新问题 / NO_VIEW待补证 / 最值得下一轮验证”四类，重新归类包内全部研究方向。
2. 解释 ORJ 父研究的正 rank IC、正高低档差为何没有转化为 V2 下一开盘、一手、成本后的可靠正净超额；区分信号、入场时钟、最低佣金、压力滑点、资本加权和时间不稳定的作用。
3. 最多选择两个候选进入下一轮；给出明确排序和理由。不得以改 ORJ 持有期、成本、阈值、报告类型或年份来救援 V2；只有提出真正不同且可事前冻结的问题时，才可把 ORJ 证据列为先验。
4. 为第一名候选设计一个最小的新版本协议：固定信号、事件钟/调仓钟、入场、退出、基准、成本、不可成交处理、退市处理、时间顺序开发/验收/一次性未见数据，以及唯一终止条件。
5. 新协议只需要回答“净超额期望是否大于零且不是单一时期/少数事件驱动”，不要重新加入高胜率、高盈亏比、高频或高利润幅度硬门槛。
6. 说明可转债一手配售、全面/部分现金要约、现金选择权三个事件方向中，哪些是研究未完成，哪些已经因可证明的样本不足而不值得继续。
7. 给出一个一页式决策：`STOP`、`RETEST_AS_NEW_VERSION` 或 `COMPLETE_MISSING_EVIDENCE`，并列出下一步最多五项工作。

第一轮只输出独立审阅与下一版研究设计，不修改包内文件，不授权 Shadow 或实盘。
"""


def strategy_results_text() -> str:
    return """# 已完成策略、主要证据与当前结论

## 总结

本轮覆盖 15 个研究方向，并对最强的 ORJ 候选额外完成了一个独立 V2 历史执行测验。没有一个获得可交易 Alpha/强 Beta 资格。旧协议与 ORJ V2 都不能追溯翻案；后续只能基于现有证据提出真正不同、事前冻结的新研究问题。

| # | 方向 | 旧协议结果 | 主要历史证据 | 当前判断 |
|---:|---|---|---|---|
| 1 | 极端缩波 P1/P2/P3 与条件模型 | V2 0/6；V2.1 0/6；V2.2 6/8，冻结拒绝 | V2.2 M6 20日执行净 rank IC 0.0414，区间下界 0.0266；但前后档差 -0.16%，区间下界 -0.76%，五档不单调 | 压缩模型分支已关闭，不应优先重启 |
| 2 | 点时行业中性残差趋势 | 1/9，冻结拒绝 | 20日 rank IC -0.0463；前20%-后20%行业超额 -0.66% | 方向性反证强，直接淘汰 |
| 3 | 月度低 MAX | 9/11，冻结拒绝 | rank IC 0.0883；原始档差 1.01% [0.06%,1.46%]；波动匹配后 0.22% [-0.15%,0.47%] | 有原始效应，但大部分可能由波动解释；次优复核候选 |
| 4 | 横截面收益季节性 | 5/11，冻结拒绝 | rank IC 0.0119；原始档差 0.18% [-0.15%,0.57%]；动量匹配后 0.28% [-0.10%,0.62%] | 证据弱，不优先 |
| 5 | 上月低换手 | 10/12，冻结拒绝 | rank IC 0.0687 [0.0384,0.0887]；原始档差 0.59% [-0.49%,1.15%]；三重匹配后 0.59% [0.11%,0.99%] | 证据可复用；建议排在 ORJ 之后做新执行版 |
| 6 | 市场价格反应延迟 | 2/12，冻结拒绝 | rank IC -0.0199，区间 [-0.0395,-0.0001]；高延迟档差 -0.25% | 冻结方向与数据相反，淘汰 |
| 7 | 高估值×日频特质波动剔除 | 4/13，冻结拒绝 | 坏单元档差 +0.229%，剔除提升 -0.019%，置信区间均不支持冻结方向 | 淘汰 |
| 8 | 人民币 FXV 正向暴露尾部剔除 | 7/13，冻结拒绝 | Q5-Q1 -0.5065% [-1.8135%,0.9601%]；剔除提升 0.0792% [-0.0043%,0.1432%] | 有边缘剔除迹象但不确定，不优先 |
| 9 | 严格 PEAD | `NO_VIEW` | 缺少满足严格点时要求的财务公告 vintage/预期基准，无法构建无泄漏惊喜值 | 只有补齐数据后才能立项 |
| 10 | 官方定期报告 ORJ 延续 | 13/14，冻结拒绝 | rank IC 0.03117 [0.02149,0.03698]；高低档行业超额 1.062% [0.644%,1.310%]；控制后系数 0.00860 [0.00273,0.01131]；2021-2025 三项年度指标均正；唯一失败为五档不严格单调 | 本轮最强的新版本执行验证候选 |
| 10b | ORJ V2 下一开盘一手执行 | 两条压力成本路径均拒绝 | 2021-2025 共 192 个有效决策日；压力成本行业净超额 -1.016% [-1.685%,-0.500%]，0/5 年为正；沪深300全收益净超额 -0.564% [-1.532%,0.332%]，1/5 年为正；基础成本宽基 +0.233% 但区间跨零且前半段为负 | 经济结果失败，当前 ORJ 执行分支关闭 |
| 11 | 官方归母净利润新区间整体上移 | 证据门 11/11；经济门 4/16，冻结拒绝 | 71 事件；200bp 后行业超额胜率 49.30%；均值 1.56%，95% 下界 -2.20%；中位数 -0.47%；前后半不稳 | 新目标下仍缺乏正净超额的可靠证据 |
| 12 | 无条件全面现金要约 8% 价差 | `NO_VIEW` | 34 个冻结事件，0 个达到 8% 观察价差，0 个次日成交事件 | 此固定 8% 定义没有可评估样本，停止 |
| 13 | 部分现金要约 20%×20% | `NO_VIEW` | 17 事件，仅 1 个官方生命周期完整；协议至少需要 8 个，证明上限 1<8 | 可证明样本不足，停止，不读行情 |
| 14 | 重组现金选择权地板 | `NO_VIEW` | 81 候选；21 个允许公告后买入且条款完整；行情源失败，价格筛选与收益未完成 | 待补行情证据，不能判正负 |
| 15 | 可转债原股东一手优先配售 | `INCOMPLETE/NO_VIEW` | 939 候选；886 个有官方正面公告证据（94.36%）；895 份官方 PDF、23,073 页文本已归档；880 个候选的配售额度已在探索解析中与发现表一致，但最终条款账本、上市日和行情未完成 | 研究暂停；不能宣称收益结论 |

## 1. 极端缩波系列

研究从 64,729 个极端压缩 episode 出发，检验第一次升波、向上突破延续和向下破位收回。V2 六项预冻结政策检验全部失败；V2.1 的六个机制候选门也全部失败。V2.2 将预测对齐到真实预测原点，并加入下一开盘/固定退出的执行口径，rank IC 为正，但主要 20 日前后档净差为负且五档不单调，因此旧分支冻结终止。它说明“模型有微弱排序信息”不等于“顶档可实现净超额为正”。

## 2. 纯价格量横截面候选

- 残差趋势、价格延迟与 OVP 的冻结方向被数据直接反驳，继续投入的机会成本很高。
- 收益季节性只有微弱 rank IC，档差与控制后档差区间均跨零。
- 低 MAX 的原始效应显著，但经同月波动匹配后不显著，提示它可能主要是低波动代理。
- 低换手在规模、波动、动量三重匹配后仍有正档差和正区间下界；旧版失败来自原始档差区间跨零与五档不严格单调。它比 MAX 更适合在新目标下做独立执行版，但历史流通股本仍需点时刷新。
- FXV 尾部剔除的提升非常接近零，现阶段不是首选。

## 3. 官方事件方向

ORJ 父研究曾是现有证据最强的候选：210 个验收事件日、82,309 个观察；rank IC、尾档差、控制后增量系数的区间下界都大于零，两个时间半段为正，2021—2025 完整年度均为正。它唯一未过的是五档均值 `0.103%、1.017%、0.896%、0.805%、1.172%` 不严格单调。

因此另立 V2，严格按 `signal_eligible` 而不是未来标签完整性选股，在每个事件日取高 ORJ 前 20%，事件后下一市场日开盘买 100 股、第 20 市场日收盘卖出，并加入历史印花税、过户费、每单最低 5 元佣金、单边 10bp 基础或 50bp 压力滑点、涨跌停/停牌/同股重叠和 H00300 全收益基准。V2 共选 23,993 个事件，主检验 15,069 个；压力成本行业净超额 -1.016%，区间完全低于零；压力成本宽基净超额 -0.564%，两半均负。基础成本宽基只有 +0.233% 点估计，区间 `[-0.722%,1.112%]`，前半段 -0.055%。因此不要求高胜率、高盈亏比或高利润幅度也无法救援：核心失败是可靠正净超额本身不存在。

利润预告上修事件虽然均值为正，但置信下界为负、中位数为负、时间半段不稳且收益集中于少数赢家；即使移除高胜率/高盈亏比要求，仍不足以说明正净超额稳定存在。

## 4. 契约型小容量事件

- 全面现金要约的 8% 固定价差没有产生可评估事件。
- 部分现金要约缺失现金到账日/剩余股份解限日等关键生命周期证据，且即使所有完整事件都进入市场评估也只有 1 个，低于最低 8 个。
- 现金选择权已把 81 个候选缩到 21 个公告后买入仍享有权利且条款完整的事件，但行情获取失败，仍是 `NO_VIEW`。
- 可转债一手配售完成了候选集、官方公告搜集和全文提取，但没有完成最终条款裁决与市场回测。探索性解析显示官方配售额度与发现表高度一致，这只是数据工程进展，不是收益证据。

## 5. 新目标下的建议优先级

1. ORJ V2 已完成并拒绝，不应通过改持有期、阈值、成本或年份继续救援。
2. 低换手是剩余完成度最高的可复用证据，但若继续必须先刷新点时流通股本并另立执行协议。
3. 低 MAX 只可在明确控制低波动暴露的情况下作为备选。
4. 现金选择权或可转债一手配售属于补证分支，是否继续取决于数据获取成本；它们尚不能与已完成执行测验的方向直接比较。

以上排序是研究优先级，不是交易排名，也不是买卖建议。
"""


def boundaries_text(builder: PackageBuilder) -> str:
    excluded_count = len(builder.large_data_exclusions)
    return f"""# 目标变更、排除项与解释边界

## 目标变更

旧研究有些以高胜率、高盈亏比、利润因子、严格单调性或多项同时通过作为硬门。2026-08-24 用户将未来研究目标改为：只要求可验证、可实现的正净超额收益，不再要求高胜率、高盈亏比、高单笔利润、高频率，也不要求大容量或高流动性。

这个变化只影响未来新版本：

- 不修改旧配置、旧门槛或旧状态；
- 不把旧 `REJECTED` 直接翻成 `PASS`；
- 允许把旧结果作为新候选的先验依据；
- 新版本仍需点时数据、时间外验证、成本、不可成交与退市处理。

## 为控制体积没有复制

- 全市场逐日原始行情、供应商缓存和完整特征面板；
- 895 份可转债官方 PDF、670 份现金选择权来源文档及大批公告逐页文本；
- 大于 {MAX_SELECTED_SOURCE_BYTES:,} 字节的直接结果文件，共 {excluded_count} 个；
- 旧的 100MB 级外部审阅 ZIP；
- `.venv`、缓存、临时目录、`.env`、凭据与任何券商文件；
- 事件逐页长 Markdown 审阅材料。

包内保留了这些排除项的协议、哈希清单、回执或小型派生账本，因此足以审阅研究逻辑和结论，但不足以在一台新机器上从零重建全部原始数据。

## 明确状态

- `REJECTED_FROZEN_*`：旧协议下已经拒绝，不能调参救援。
- `NO_VIEW_*`：证据不足或来源失败，不等于收益为零或策略失败。
- `INCOMPLETE`：尚未进入收益裁决。
- `DISCOVERY_ONLY`：不能映射仓位、生成订单或连接实盘。

## 数据时间边界

所有研究结论以 {EVIDENCE_CUTOFF} 为统一证据截止日。2026 年多数年度统计只是截至该日的部分年度，不得与完整年度等同。
"""


VERIFY_SCRIPT = r'''"""验证解压后的 GPT 审阅包。"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "04_FILE_MANIFEST_SHA256.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["relative_path"] for row in rows}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"文件集合不一致：missing={missing}, extra={extra}")
    failures = []
    for row in rows:
        path = ROOT / Path(row["relative_path"])
        if path.stat().st_size != int(row["size_bytes"]):
            failures.append(f"大小不符：{row['relative_path']}")
        if sha256_file(path) != row["sha256"]:
            failures.append(f"哈希不符：{row['relative_path']}")
    if failures:
        raise RuntimeError("；".join(failures))
    print(f"验证通过：{len(rows)} 个清单文件，0 缺失，0 额外，SHA-256 全部一致。")


if __name__ == "__main__":
    main()
'''


def write_provenance(package_root: Path, builder: PackageBuilder) -> None:
    path = package_root / "07_SOURCE_PROVENANCE.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "source_relative_path",
            "package_member",
            "source_size_bytes",
            "source_sha256",
            "package_size_bytes",
            "package_sha256",
            "text_redacted",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(builder.provenance, key=lambda row: row["package_member"]))


def privacy_scan(package_root: Path) -> dict[str, Any]:
    """扫描用户目录、用户名、明显凭据与禁入文件。"""

    username = Path.home().name
    home = str(Path.home())
    windows_users_prefix = "C:" + chr(92) + "Users" + chr(92)
    byte_rules = {
        "USER_HOME_LITERAL": home.encode("utf-8"),
        "USERNAME_LITERAL": username.encode("utf-8"),
        "WINDOWS_USERS_ABSOLUTE_PREFIX": windows_users_prefix.encode("utf-8"),
        "PRIVATE_KEY_MARKER": b"BEGIN " + b"PRIVATE KEY",
    }
    credential_pattern = re.compile(
        r"(?i)(password|passwd|secret|api[_-]?key|access[_-]?token)"
        r"\s*[:=]\s*['\"][A-Za-z0-9_./+\-=]{8,}['\"]"
    )
    hits: list[dict[str, str]] = []
    scanned_files = 0
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root).as_posix()
        scanned_files += 1
        if path.name.lower() == ".env" or relative.lower().endswith("/.env"):
            hits.append({"rule_id": "ENV_FILE_FORBIDDEN", "relative_path": relative})
        raw = path.read_bytes()
        for rule_id, needle in byte_rules.items():
            if needle and needle in raw:
                hits.append({"rule_id": rule_id, "relative_path": relative})
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = ""
            if credential_pattern.search(text):
                hits.append(
                    {"rule_id": "LITERAL_CREDENTIAL_ASSIGNMENT", "relative_path": relative}
                )
    return {
        "status": "PASS_ZERO_PRIVACY_OR_CREDENTIAL_HITS" if not hits else "FAIL_PRIVACY_SCAN",
        "scanned_file_count": scanned_files,
        "hit_count": len(hits),
        "hits": hits,
        "rules": sorted([*byte_rules, "ENV_FILE_FORBIDDEN", "LITERAL_CREDENTIAL_ASSIGNMENT"]),
        "redaction_rule_file_counts": dict(sorted(Counter().items())),
    }


def write_manifest(package_root: Path) -> list[dict[str, Any]]:
    """清单覆盖除清单自身外的全部文件。"""

    manifest_path = package_root / "04_FILE_MANIFEST_SHA256.csv"
    rows: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(package_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def verify_tree(package_root: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = package_root / "04_FILE_MANIFEST_SHA256.csv"
    expected = {str(row["relative_path"]) for row in rows}
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise RuntimeError(
            f"文件集合不一致：missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    for row in rows:
        path = package_root / Path(str(row["relative_path"]))
        if path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"文件大小不一致：{row['relative_path']}")
        if sha256_file(path) != str(row["sha256"]):
            raise RuntimeError(f"文件哈希不一致：{row['relative_path']}")


def create_zip(package_root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package_root).as_posix()
            archive.write(path, f"{INTERNAL_ROOT}/{relative}")
    os.replace(temporary, destination)


def verify_zip_stream(zip_path: Path, package_root: Path) -> dict[str, Any]:
    expected = {
        f"{INTERNAL_ROOT}/{path.relative_to(package_root).as_posix()}": path
        for path in package_root.rglob("*")
        if path.is_file()
    }
    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        actual_names = {info.filename for info in infos}
        if actual_names != set(expected):
            raise RuntimeError("ZIP 成员集合与构包目录不一致")
        for info in infos:
            with archive.open(info, "r") as handle:
                archive_hash = sha256_stream(handle)
            if archive_hash != sha256_file(expected[info.filename]):
                raise RuntimeError(f"ZIP 字节流哈希不一致：{info.filename}")
    return {"status": "PASS_ZIP_STREAM_VERIFIED", "member_count": len(expected)}


def verify_fresh_extraction(zip_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="a_share_alpha_review_extract_") as temp_name:
        extraction_root = Path(temp_name)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                member = PurePosixPath(info.filename)
                if member.is_absolute() or ".." in member.parts:
                    raise RuntimeError(f"ZIP 含非法路径：{info.filename}")
            archive.extractall(extraction_root)
        extracted_package = extraction_root / INTERNAL_ROOT
        verify_tree(extracted_package, rows)
    return {
        "status": "PASS_FRESH_EXTRACTION_AND_MANIFEST_VERIFIED",
        "manifest_file_count": len(rows),
    }


def main() -> None:
    started_at = datetime.now(TIME_ZONE).isoformat()
    builder = PackageBuilder()
    collect_sources(builder)
    DELIVERABLES.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="a_share_alpha_review_build_") as temp_name:
        package_root = Path(temp_name) / INTERNAL_ROOT
        package_root.mkdir(parents=True, exist_ok=True)
        builder.copy_all(package_root)

        write_text(package_root / "00_README_FIRST.md", readme_text(len(builder.mappings)))
        write_text(package_root / "01_GPT_REVIEW_PROMPT.md", review_prompt_text())
        write_text(
            package_root / "02_STRATEGY_RESULTS_AND_CONCLUSIONS.md",
            strategy_results_text(),
        )
        write_text(
            package_root / "03_OBJECTIVE_CHANGE_EXCLUSIONS_AND_BOUNDARIES.md",
            boundaries_text(builder),
        )
        write_text(package_root / "verify_package.py", VERIFY_SCRIPT)
        write_provenance(package_root, builder)

        scan = privacy_scan(package_root)
        scan["redaction_rule_file_counts"] = dict(sorted(builder.redaction_counts.items()))
        if scan["hit_count"]:
            raise RuntimeError(
                "隐私扫描失败："
                + json.dumps(scan["hits"], ensure_ascii=False, sort_keys=True)
            )
        write_json(package_root / "05_PRIVACY_SCAN.json", scan)
        write_json(
            package_root / "06_VERIFICATION_SCOPE.json",
            {
                "status": "PASS_PREZIP_SOURCE_SELECTION_AND_PRIVACY_SCAN",
                "evidence_cutoff": EVIDENCE_CUTOFF,
                "source_file_count": len(builder.mappings),
                "large_direct_result_exclusion_count": len(builder.large_data_exclusions),
                "research_rerun_performed": False,
                "frozen_protocol_modified": False,
                "privacy_hit_count": 0,
                "verification_steps": [
                    "SOURCE_RELATIVE_PATH_AND_SHA256_INDEX",
                    "ZERO_HIT_PRIVACY_SCAN",
                    "ZIP_MEMBER_BYTE_STREAM_SHA256",
                    "FRESH_EXTRACTION_MANIFEST_SHA256",
                ],
            },
        )

        manifest_rows = write_manifest(package_root)
        verify_tree(package_root, manifest_rows)
        create_zip(package_root, ZIP_PATH)
        stream_result = verify_zip_stream(ZIP_PATH, package_root)
        extraction_result = verify_fresh_extraction(ZIP_PATH, manifest_rows)

    zip_hash = sha256_file(ZIP_PATH)
    write_text(SHA256_PATH, f"{zip_hash}  {ZIP_PATH.name}\n")
    receipt = {
        "package_id": PACKAGE_BASENAME,
        "status": "PASS_UPLOAD_READY_GPT_REVIEW_ZIP",
        "started_at": started_at,
        "completed_at": datetime.now(TIME_ZONE).isoformat(),
        "evidence_cutoff": EVIDENCE_CUTOFF,
        "zip_path": ZIP_PATH.relative_to(ROOT).as_posix(),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": zip_hash,
        "source_file_count": len(builder.mappings),
        "manifest_file_count": len(manifest_rows),
        "zip_member_count": stream_result["member_count"],
        "privacy_scan_status": "PASS_ZERO_PRIVACY_OR_CREDENTIAL_HITS",
        "privacy_hit_count": 0,
        "redaction_rule_file_counts": dict(sorted(builder.redaction_counts.items())),
        "large_direct_result_exclusion_count": len(builder.large_data_exclusions),
        "zip_stream_verification": stream_result,
        "fresh_extraction_verification": extraction_result,
        "research_rerun_performed": False,
        "frozen_protocol_modified": False,
        "shadow_authorized": False,
        "live_authorized": False,
    }
    write_json(RECEIPT_PATH, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
