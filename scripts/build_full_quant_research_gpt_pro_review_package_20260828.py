"""构建截至 2026-08-28 的全项目 GPT Pro 快速审阅包。

本脚本复用 2026-08-25 的稳定构包与 ZIP CRC 逻辑，更新当前状态、
审阅任务书和索引，并补充小型 ``data/research`` 派生证据。按用户要求，
本次不做数据安全、隐私、脱敏、逐文件哈希或全新解压审计。
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import build_full_quant_research_gpt_pro_review_package_20260825 as base


SNAPSHOT_DATE = "2026-08-28"
RECENT_CUTOFF = datetime(2026, 8, 25, tzinfo=base.TIME_ZONE)
PACKAGE_BASENAME = "QUANT_RESEARCH_ALL_WORK_GPT_PRO_REVIEW_20260828"
INTERNAL_ROOT = "QUANT_RESEARCH_ALL_WORK_GPT_PRO_REVIEW_20260828"

base.PACKAGE_BASENAME = PACKAGE_BASENAME
base.INTERNAL_ROOT = INTERNAL_ROOT
base.ZIP_PATH = base.DELIVERABLES / f"{PACKAGE_BASENAME}.zip"
base.SHA256_PATH = base.DELIVERABLES / f"{PACKAGE_BASENAME}.sha256"
base.RECEIPT_PATH = base.DELIVERABLES / f"{PACKAGE_BASENAME}_BUILD_RECEIPT.json"
base.UPLOAD_MESSAGE_PATH = (
    base.DELIVERABLES / f"{PACKAGE_BASENAME}_UPLOAD_MESSAGE.txt"
)
base.EXCLUSION_HASH_CACHE_PATH = (
    base.ROOT / "tmp" / f"{PACKAGE_BASENAME}_EXCLUSION_HASH_CACHE.csv"
)
base.MAX_INCLUDED_SOURCE_BYTES = 500 * 1024 * 1024
base.FAST_UNREDACTED_USER_REQUEST = True


STATUS_SOURCE_ROWS = (
    (
        "510300二元净超额20个百分点研究",
        "PAUSED_NO_VERIFIED_STRATEGY",
        "reports/research/510300_BINARY_EXCESS20_RESEARCH_PAUSE_SNAPSHOT_20260828_V2.md",
        "跨ETF官方周度份额流完成后暂停；没有候选通过双20硬门。",
    ),
    (
        "510300跨ETF被迫资金流V1",
        "REJECTED_FIXED_CROSS_ETF_FORCED_FLOW_BINARY_FAMILY_NO_RESCUE",
        "reports/research/510300_CROSS_ETF_FORCED_FLOW_BINARY_SCREEN_V1.md",
        "固定10只ETF、周度官方份额、T+2执行；禁止改池、阈值、周数或时钟救援。",
    ),
    (
        "510300前20名会员持仓V1",
        "REJECTED_FIXED_CFFEX_MEMBER_POSITION_BINARY_FAMILY_NO_RESCUE",
        "reports/research/510300_CFFEX_MEMBER_POSITION_BINARY_SCREEN_V1.md",
        "官方IF逐合约与会员排名库保留；固定家族未通过双20。",
    ),
    (
        "优先前瞻权威状态V1.6",
        "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET",
        "reports/audit/priority_forward_authoritative_status_v1_6.json",
        "总决策PAUSE_AND_FIX_FOUNDATION；需同时核对V1.8当日不可变运行回执。",
    ),
    (
        "PCF/IOPV V1.2.1当日采集",
        "EXTERNAL_FREE_SOURCE_FAILED",
        "reports/data_quality/510300_primary_market_task_status_v1_2_1.json",
        "2026-08-28免费外部源TLS校验失败；不是质量日，不得回填。",
    ),
    (
        "行业预期差前瞻流",
        "COLLECTING_FORWARD_WITH_STALE_OR_BLOCKED_OUTCOME_INPUT",
        "reports/audit/priority_forward_research_status_current.json",
        "1个原点、0个成熟原点、0个非重叠60日块；未到评价门。",
    ),
    (
        "V3_FORWARD_2",
        "NO_LATEST_STATUS_NO_SIGNAL",
        "paper/v3_forward_2_daily_run_status.json",
        "输入与模型状态需分别核对；缺latest_status不能解释为信号。",
    ),
    (
        "全A股点时质量-保守投资V1",
        "REJECTED_FROZEN_P0_QUALITY_INVESTMENT_MECHANISM_FAILED",
        "reports/research/A_SHARE_HS_PIT_QUALITY_CONSERVATIVE_INVESTMENT_V1.md",
        "P0单调性和对全部基准正超额门失败；停止该线，不加条件救援。",
    ),
    (
        "全A股基金宽度趋势确认AFHC",
        "REJECTED_FROZEN_P0_AFHC_MECHANISM_FAILED_STOP_FUND_LINE",
        "reports/research/A_SHARE_HS_FUND_BREADTH_TREND_CONFIRMATION_V1_0_1.md",
        "P0失败，P1/P2/P3未计算。",
    ),
    (
        "A股极端缩波V2/V2.1",
        "NO_REPEATABLE_CONDITIONAL_EXCESS_IDENTIFIED",
        "docs/A_SHARE_HS_EXTREME_COMPRESSION_V2_V2_1_TERMINAL_FREEZE_20260823.md",
        "冻结关闭，不得用相邻条件救援。",
    ),
    (
        "A股ORJ V2成本后执行",
        "REJECTED_ORJ_NET_EXCESS_EXECUTION_V2",
        "reports/research/A_SHARE_HS_OFFICIAL_REPORT_ORJ_NET_EXCESS_EXECUTION_V2.md",
        "父研究预测证据没有转化为下一开盘、一手、成本后可靠净超额。",
    ),
    (
        "多资产年化净超额40个百分点/高夏普",
        "NO_VERIFIED_40PCT_ANNUALIZED_NET_EXCESS_OR_HIGH_SHARPE_CANDIDATE",
        "reports/research/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_CLOSEOUT_20260827.md",
        "本轮结案；合同失败和定量拒绝均保留。",
    ),
    (
        "既有数字资产策略近期同口径比较",
        "NO_STRATEGY_PASSED_20PP_GATE",
        "reports/research/RECENT_DIGITAL_ASSET_STRATEGY_COMPARISON_2023_2026.md",
        "固定资产覆盖不强行对齐；报告成本、净超额、Sharpe、回撤和年度稳定性。",
    ),
    (
        "市场状态感知配置器V1",
        "NO_VIEW_CASH_ONLY",
        "reports/research/REGIME_AWARE_STRATEGY_ALLOCATOR_V1_STATUS.md",
        "缺环境证据且0个合格策略；不复活已拒绝组件。",
    ),
    (
        "国际化券商研究",
        "NO_VIEW_G3_PROFIT_BRIDGE_INCOMPLETE",
        "reports/research/international_broker_g2_g3_v1_3_status.md",
        "与510300隔离；利润桥未完成前不做收益、仓位或订单。",
    ),
    (
        "知乎可观察语料研究",
        "BOUNDED_OBSERVABLE_CORPUS_DISCOVERY_ONLY",
        "reports/research/ZHIHU_OLIVER_OBSERVABLE_CORPUS_V2_1_STATUS.md",
        "不是完整账户语料；只允许候选生成。",
    ),
)
base.STATUS_SOURCE_ROWS = STATUS_SOURCE_ROWS


def collect_sources(builder: base.PackageBuilder) -> None:
    """纳入正式证据层和小型研究派生表，排除大体量原始数据。"""

    base.collect_sources(builder)
    builder.add_tree(
        "data/research",
        lambda path: (
            path.suffix.lower()
            in {".json", ".jsonl", ".csv", ".tsv", ".md", ".yaml", ".yml"}
            and path.stat().st_size <= base.SMALL_STRUCTURED_LIMIT
        )
        or (
            path.suffix.lower() == ".parquet"
            and path.stat().st_size <= base.SMALL_BINARY_LIMIT
        ),
    )
    included_bytes = sum(path.stat().st_size for path in builder.mappings.values())
    if included_bytes > base.MAX_INCLUDED_SOURCE_BYTES:
        raise RuntimeError(
            "纳入证据层超过上传友好上限："
            f"{included_bytes} > {base.MAX_INCLUDED_SOURCE_BYTES} bytes"
        )


def readme_text(
    source_count: int,
    source_bytes: int,
    excluded_count: int,
    excluded_bytes: int,
) -> str:
    return f"""# 全项目量化研究：GPT Pro 独立审阅包

## 交付结论

这是截至 {SNAPSHOT_DATE}（Asia/Shanghai）的当前工作区审阅快照。包内直接纳入 {source_count:,} 个正式源文件和小型派生证据，原始合计 {source_bytes:,} 字节；另登记但不复制 {excluded_count:,} 个大体量原始、缓存、环境或重复文件，合计 {excluded_bytes:,} 字节。

“所有内容”在本包中的定义是：覆盖项目的代码、配置、冻结清单、测试、文档、报告、状态回执、小型账本、可上传派生结果和用户思考；不是工作区逐字节镜像。大体量行情、公告原文、逐页文本、中间面板、虚拟环境、缓存和旧ZIP只做索引，避免让GPT Pro在大量数据中丢失判断主线。

当前最高层结论仍是：`NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET`。其中510300二元净超额20个百分点历史机制搜索已暂停；多资产40个百分点/高夏普目标本轮结案未达标；A股质量—保守投资和AFHC均在冻结P0被拒绝；严格前瞻流仍未成熟或受数据/工程闸门阻塞。

## 必读顺序

1. `00_README_FIRST.md`
2. `01_GPT_PRO_REVIEW_PROMPT.md`
3. `18_CURRENT_STATUS_SNAPSHOT.md`
4. `03_PROJECT_MAP_AND_CURRENT_JUDGMENTS.md`
5. `02_USER_THOUGHTS_CONSTRAINTS_AND_OPEN_QUESTIONS.md`
6. `04_BRANCH_BOUNDARIES_AND_CONTRADICTION_RULES.md`
7. `17_RESEARCH_REPORT_INDEX.csv` 与 `12_STATUS_SOURCE_INDEX.csv`
8. `16_RECENT_WORK_INDEX.csv`，用于定位8月25日后新增或更新的工作
9. 再按需深入 `docs/`、`reports/`、`config/`、`research/`、`scripts/`、`tests/` 和 `data/research/`
10. 最后用 `06_INCLUDED_SOURCE_PROVENANCE.csv`、`07_EXCLUDED_SOURCE_INVENTORY.csv`、`08_INCLUDED_FILE_INDEX.csv` 核对覆盖范围

## 重要解释规则

- `NO_VIEW`、`FAILED`、`PROGRAM_FAILED`、`EXTERNAL_FREE_SOURCE_FAILED`、`SKIPPED`、`REJECTED_FROZEN`、`DISCOVERY_ONLY`、Paper/Shadow和实盘授权具有不同语义。
- 已拒绝版本不能通过换窗口、阈值、方向、市场、代理、成本、标签、权重或制度阶段进行事后救援。
- `docs/DECISIONS.md` 是追加式历史；按日期、版本、冻结协议和回执判断有效性，不得拼接出从未冻结的“最佳策略”。
- 研究包交付不等于GPT Pro已经审阅，也不等于任何策略获得信号、仓位、订单、券商连接或实盘资格。

## 本次校验边界

按用户要求，本次不做数据安全审计、隐私扫描、脱敏、秘密扫描、恶意文件扫描、逐文件SHA-256或全新解压复核。只做一次ZIP CRC/重复成员检查，并在ZIP外生成整包SHA-256；`.env`、虚拟环境和缓存仍按构包选择规则不纳入正文。
"""


def review_prompt_text() -> str:
    return """# 给 GPT Pro 的独立、全面审阅任务书

你是一名独立量化研究负责人、反方审计员、数据治理负责人和研究组合经理。请把本包作为一个包含大量负结论、冻结协议、程序失败、前瞻未成熟状态和少量开放假设的完整研究项目来审阅。目标不是从旧结果中挑一个最好看的曲线，而是判断：哪些结论可信，哪些结论可能由实现/数据/时钟错误导致，下一阶段有限资源应该投向哪里，哪些方向应永久或阶段性停止。

## 审阅方法

1. 先读00、18、03、02、04、17和12号文件，再进入具体证据；导航摘要不是最终证据。
2. 以冻结协议、不可变清单、运行回执、点时数据合同、事件/决策日账本、成本模型、测试和机器可读报告为准。
3. 对每个结论分别审查：数据可得时点、未来泄漏、幸存者偏差、样本选择、版本漂移、执行时钟、整手/最低佣金、基础与压力成本、不可成交、现金收益、基准、年度稳定性、Sharpe和回撤。
4. 区分历史开发、独立历史验证、封存复验、严格前瞻、Paper/Shadow和实盘；`HISTORICALLY_CONTAMINATED_DISCOVERY_ONLY`不得升级为严格验证。
5. 不得通过结果后改窗口、阈值、方向、资产池、市场、成本、标签、权重、制度阶段或代理变量救援任何冻结拒绝版本。真正不同的问题必须另立协议、事前冻结候选与唯一停止线。
6. 缺数据不是零收益；程序退出0不自动等于业务成功；`EXTERNAL_FREE_SOURCE_FAILED`不是策略失败；`NO_VIEW`不是空仓或看空。
7. 不输出买卖指令、目标仓位、个股/币种推荐或券商连接方案。这里只做研究裁决和资源配置建议。

## 必须覆盖的研究板块

- 510300历史研究全谱系：估值、宏观、趋势、MACD/背离、唐奇安、鱼中/板块传播、日内、ETF微观结构、期权/期指/宽度、一级市场、全球隔夜、北向、宏观意外、ETF份额/折溢价、IF期限结构、IF会员持仓、跨ETF官方周度份额流，以及净超额20个百分点的数学可行性与准确率前沿。
- 510300严格前瞻：PCF/IOPV V1.8单入口与V1.2.1质量日、行业预期差、T-only、V3_FORWARD_2、调度/回执/声明去重，以及权威状态与当日回执之间可能存在的口径差异。
- A股横截面与事件：低波/缩波、残差趋势、低MAX、低换手、季节性、价格延迟、OVP、FXV、ORJ、利润预告、现金要约/选择权、可转债配售、基金宽度AFHC、点时质量—保守投资等。
- 多资产/数字资产/美股/可转债/QDII等扩展目标：40个百分点高夏普结案、既有数字资产同口径近期比较、市场状态配置器及所有合同失败/冻结拒绝。
- 国际化券商和知乎可观察语料：严格保持为独立研究或候选生成分支，不得映射510300或仓位。
- 工程与治理：冻结哈希漂移、旧状态覆盖、免费源失败、TLS、模块导入、任务声明、不可变回执、数据覆盖和版本注册。

## 对每条实质性方向给出分类

使用以下分类之一，并引用直接证据路径：

1. `STOP_PERMANENTLY`
2. `STOP_CURRENT_VERSION_RETAIN_EVIDENCE`
3. `COMPLETE_MISSING_EVIDENCE`
4. `CONTINUE_STRICT_FORWARD_ONLY`
5. `ONE_NEW_FROZEN_TEST`
6. `OUT_OF_SCOPE_OR_SEPARATE_RESEARCH`
7. `RECHECK_IMPLEMENTATION_OR_DATA_CONTRACT`（仅当你能指出具体可证伪错误，不得用它泛化救援负结果）

## 重点问题

1. 哪些负结论在冻结合同下足够强，应永久停止；哪些只是当前版本失败但数据资产仍有复用价值？
2. 510300二元满仓/现金目标在现实信号强度、换手、回撤与年度稳定性上是否不合理？20个百分点门应保留为目标、改为筛选上限，还是终止该研究计划？
3. 为什么开发期最强候选会在独立期失效？请区分多重尝试、制度漂移、代理滞后、数据质量与执行成本。
4. PCF/IOPV、行业预期差和其他前瞻流最稀缺的是新信号、完整质量日、零成本数据、工程稳定性还是治理？是否值得继续等待？
5. A股质量—保守投资、AFHC、ORJ、缩波等P0/V2负结论是否存在实现或基准错误；若没有，请明确停止，不建议调参。
6. 数字资产策略2023—2026的固定覆盖、成本、年度稳定性、Sharpe与回撤是否支持继续；不要把2023—2024的强年掩盖2025—2026失效。
7. 项目是否因为同时研究太多方向而产生多重试验与注意力稀释？请提出项目级研究预算、并行流上限和停止机制。
8. 哪些原始数据确实需要从07号排除清单另行补给才能完成关键复核？只列高信息价值缺口，不要要求无差别上传全部数据。

## 输出格式

请用中文输出：

1. 一页执行结论：总判断、可信度、前三项继续事项、前三项停止事项；
2. 全方向裁决表：分支、版本、当前状态、直接证据、分类、理由、关键缺口、是否需原始数据；
3. 发现的问题清单：按P0/P1/P2严重度列出具体文件、字段/函数、影响和最小复核办法；若无证据，不要虚构问题；
4. 研究资产清单：哪些数据、框架、测试和治理组件值得保留，即使策略被拒绝；
5. 最多三个未来优先方向：按预期信息价值排序，给出为什么值得、为什么不是挑历史赢家；
6. 30/90/180天路线图：每阶段最多两个并行流，写输入、成本、交付物、成熟门和硬停止条件；
7. 明确不做清单：具体到策略家族、数据救援、调参方式和执行越权；
8. 只列真正需要用户决定且会改变范围、预算或授权的问题。

最后必须只选择一个项目级总建议：`PAUSE_AND_FIX_FOUNDATION`、`CONTINUE_FORWARD_ONLY`、`RUN_ONE_NEW_FROZEN_TEST` 或 `STOP_RESEARCH_PROGRAM`，并解释选择依据。
"""


def user_thoughts_text() -> str:
    inherited = base.user_thoughts_text().rstrip()
    return inherited + """

## E. 2026-08-26至28新增目标与边界

- 510300分支曾把“满仓510300或人民币现金”的历史成本后年化净超额20个百分点作为高门槛可证伪目标。数学神谕证明目标并非纯算术不可能，但现实固定机制全部未通过；用户接受暂停和负结论，不要求降低门槛来制造成功。
- 独立的多资产探索曾使用年化净超额40个百分点且基础/压力净Sharpe均不低于1.50的目标；本轮结案为未找到候选。该扩展研究不自动改变510300现货/现金交易主线，也不授权任何新资产执行。
- 用户希望GPT Pro结合已做的全部内容，决定后续应努力的方向与明确不做的方向；优先需要研究组合裁决，不是再生成一批未经冻结的策略点子。
- 本次为了便于上传，允许不附大量具体原始数据，也明确不要求数据安全审计；这不改变研究证据、状态语义和冻结拒绝边界。
"""


def project_map_text() -> str:
    return """# 项目地图与当前裁决导航

## 1. 510300现货/现金主线

最新总暂停快照是 `reports/research/510300_BINARY_EXCESS20_RESEARCH_PAUSE_SNAPSHOT_20260828_V2.md`。它汇总统一H00300全收益账本、14次分红、严格0/1状态、整手、T+1、基础/压力成本、固定时期和起点扰动，并记录ETF微观结构、日内趋势、期权IVS、一级市场、全球隔夜、北向旧口径、宏观意外、基金份额/折溢价、IF期限结构、IF会员持仓及跨ETF周度份额流均未形成通过双20门的可验证候选。

跨ETF官方份额流是当前最后完成的历史机制：287周、10只事前固定ETF、0缺失快照，最优规则只有弱信息且2024年后反转，冻结拒绝并暂停。数据资产与负结论均保留，不建立前瞻候选。

## 2. 严格前瞻与治理

优先读 `reports/audit/priority_forward_authoritative_status_v1_6.json`、`reports/audit/priority_forward_research_status_current.json`、V1.8不可变运行回执和 `reports/data_quality/510300_primary_market_task_status_v1_2_1.json`。当前只有2个完整PCF/IOPV质量日；2026-08-28免费源阶段为 `EXTERNAL_FREE_SOURCE_FAILED`。行业预期差仍为1个原点、0成熟原点、0个非重叠60日块。V3_FORWARD_2没有可用latest_status。工程修复或调度成功不能被写成策略通过。

## 3. A股横截面与事件

极端缩波V2/V2.1、ORJ V2、AFHC P0及点时质量—保守投资P0均已有冻结负结论；利润预告、现金要约/选择权、可转债配售等需要保留 `NO_VIEW`、数据不足和经济拒绝的差异。`17_RESEARCH_REPORT_INDEX.csv` 提供全部研究报告路由。

## 4. 多资产与数字资产扩展

多资产40个百分点/高Sharpe目标结案为未找到候选。数字资产同口径比较保留实际资产覆盖和区间，不把V15只有2023年的覆盖与2023—2026策略视为等价；五个完整比较策略均未过20个百分点门，2025—2026年度稳定性明显不足。市场状态配置器因缺环境证据且0个合格组件，输出 `NO_VIEW_CASH_ONLY`，不复活已拒绝策略。

## 5. 国际化券商与公开语料

国际化券商仍停在结构/利润桥补证；知乎仅为有限可观察语料和候选生成。两个分支都与510300执行隔离。

## 6. 证据层

- `reports/`：人读与机器读结论、数据质量、审计、前瞻回执；
- `config/`：协议、候选、清单和版本边界；
- `research/`、`scripts/`、`tests/`：实现、入口和回归证据；
- `data/research/`：小型派生面板和结果表；
- `07_EXCLUDED_SOURCE_INVENTORY.csv`：未复制的大型原始/中间证据；
- `16_RECENT_WORK_INDEX.csv`、`17_RESEARCH_REPORT_INDEX.csv`：近期增量和全部报告路由。
"""


def branch_boundaries_text() -> str:
    inherited = base.branch_boundaries_text().rstrip()
    return inherited + """

## 扩展分支补充

| 分支 | 允许输出 | 禁止跨越 |
|---|---|---|
| 510300二元20个百分点研究 | 数学可行性、历史筛选、独立验证、暂停/拒绝状态 | 降门槛冒充达标、把弱风险信息变为信号 |
| 多资产40个百分点/高Sharpe | 独立发现与结案证据 | 自动扩大现行交易授权或按历史赢家选资产 |
| 数字资产近期比较 | 固定资产与区间下的成本、净超额、Sharpe、回撤、年度稳定性 | 把不等覆盖当等价比较、把单一年份外推为当前资格 |
| 市场状态配置器 | 研究资格与风险预算视图 | 复活已冻结拒绝组件、缺证据时沿用旧环境 |
"""


def exclusions_text(
    excluded_rows: list[dict[str, Any]],
    excluded_directories: list[base.ExcludedDirectory],
) -> str:
    reasons = Counter(str(row["exclusion_reason"]) for row in excluded_rows)
    reason_lines = "\n".join(
        f"- `{reason}`：{count:,} 个文件" for reason, count in sorted(reasons.items())
    )
    total_bytes = sum(int(row.get("size_bytes") or 0) for row in excluded_rows)
    return f"""# 排除项、复现边界与本次检查范围

## 内容选择

本包优先纳入代码、协议、冻结清单、测试、报告、状态回执、小型账本、Excel成果以及不超过阈值的 `data/research` 派生表。另有 {len(excluded_rows):,} 个文件、约 {total_bytes / 1024 / 1024 / 1024:.2f} GiB 未直接复制，原因如下：

{reason_lines}

另有 {len(excluded_directories):,} 个环境、缓存、测试临时或重复解压目录只在 `11_EXCLUDED_DIRECTORY_SCOPES.csv` 汇总。

## 为什么省略大量具体数据

- 大型原始行情、官方公告PDF/逐页文本、原始响应库和中间面板会显著增加上传体积，但多数方向已有MD/JSON/CSV结论与质量报告可先审；
- 旧审阅ZIP不递归嵌套；
- `.env`、`.venv`、`.git`、缓存和临时目录不属于GPT Pro首轮研究判断正文；
- 需要重跑某一具体研究时，再按 `07_EXCLUDED_SOURCE_INVENTORY.csv` 精确补给对应源文件，而不是无差别补全部数据。

## 复现与完整性边界

1. `08_INCLUDED_FILE_INDEX.csv` 只记录包内路径和大小，不是逐文件完整性清单。
2. `06_INCLUDED_SOURCE_PROVENANCE.csv` 记录源到包的原样复制关系；不进行脱敏、重编码或逐文件哈希。
3. 排除清单只登记路径、大小和原因；按用户要求没有为排除项计算SHA-256。
4. 本次没有重跑研究、改冻结协议、重新估计模型或执行全量测试；只生成导航、索引和ZIP。
5. 文件系统快照没有暂停其他进程，属于构包时点的尽力快照，不是数据库式原子快照。
6. 本次明确跳过数据安全、隐私、秘密、恶意文件、脱敏和全新解压审计；只保留ZIP CRC和整包SHA-256。
"""


REPRODUCTION_NOTES = """# 独立复核说明

1. 先由17号报告索引定位研究，再读取对应配置、冻结清单、测试、机器可读报告和运行回执。
2. 若复核只涉及结论逻辑，包内小型证据通常足够；若要从零重跑，再从07号排除清单恢复该研究对应的大型输入。
3. 用 `requirements.txt` 重建环境，不复用本机 `.venv`。
4. 保留所有 `NO_VIEW`、`FAILED`、`PROGRAM_FAILED`、`EXTERNAL_FREE_SOURCE_FAILED`、`SKIPPED`、删失和未成熟状态。
5. 实现错误只能通过带差异说明的新版本更正，不能静默覆盖冻结结果。

本次构包未执行全量回归测试，也未做数据安全审计、隐私扫描、脱敏、逐文件哈希、ZIP逐成员SHA-256或全新解压复核。ZIP外 `.sha256` 只用于确认整包文件；ZIP CRC只确认压缩成员可读和无重复成员名。
"""


UPLOAD_MESSAGE = """我上传的是截至2026-08-28的全项目量化研究自包含审阅包。它详细覆盖代码、协议、测试、正负结果、当前状态、用户约束和近期新增研究，但为控制体积省略了大量原始/中间数据；排除项已在07号索引登记。本包按我的要求未做数据安全、隐私或脱敏审计，只做ZIP CRC和整包SHA-256。

请先读00_README_FIRST.md、01_GPT_PRO_REVIEW_PROMPT.md、18_CURRENT_STATUS_SNAPSHOT.md、03_PROJECT_MAP_AND_CURRENT_JUDGMENTS.md、02_USER_THOUGHTS_CONSTRAINTS_AND_OPEN_QUESTIONS.md、04_BRANCH_BOUNDARIES_AND_CONTRADICTION_RULES.md、17_RESEARCH_REPORT_INDEX.csv和12_STATUS_SOURCE_INDEX.csv，再按01号任务书完整审阅。不要调参救援冻结拒绝版本，不要把NO_VIEW/程序失败/外部源失败当收益结论，不要输出仓位或订单。我要你判断哪些结论可信、哪些需复核、后面最多投入哪三个方向、哪些方向明确停止，并给出30/90/180天路线图和硬停止条件。"""


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, base.TIME_ZONE).isoformat()


def _first_text_status(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")[:20000]
    except OSError:
        return ""
    patterns = (
        r"(?:主状态|正式状态|当前研究状态|状态|status)\s*[：:]\s*`?([A-Z][A-Z0-9_\-]+)",
        r"(?:总判断|结论)\s*[：:]\s*`?([A-Z][A-Z0-9_\-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _json_summary(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key in (
        "study_id",
        "report_version",
        "status",
        "overall_status",
        "overall_research_status",
        "decision",
        "next_step",
        "generated_at",
        "finished_at",
    ):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)):
            result[key] = str(item)
    return result


def write_recent_work_index(
    package_root: Path,
    builder: base.PackageBuilder,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for destination, source in sorted(builder.mappings.items()):
        modified_at = datetime.fromtimestamp(source.stat().st_mtime, base.TIME_ZONE)
        if modified_at < RECENT_CUTOFF:
            continue
        rows.append(
            {
                "source_relative_path": source.relative_to(base.ROOT).as_posix(),
                "package_member": destination,
                "modified_at": modified_at.isoformat(),
                "size_bytes": source.stat().st_size,
                "category": destination.split("/", 1)[0],
            }
        )
    base.write_csv(
        package_root / "16_RECENT_WORK_INDEX.csv",
        [
            "source_relative_path",
            "package_member",
            "modified_at",
            "size_bytes",
            "category",
        ],
        rows,
    )
    return rows


def write_research_report_index(
    package_root: Path,
    builder: base.PackageBuilder,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for destination, source in sorted(builder.mappings.items()):
        if not destination.startswith("reports/research/"):
            continue
        if source.suffix.lower() not in {".md", ".json"}:
            continue
        summary = _json_summary(source) if source.suffix.lower() == ".json" else {}
        status = (
            summary.get("status")
            or summary.get("overall_status")
            or summary.get("overall_research_status")
            or _first_text_status(source)
        )
        rows.append(
            {
                "relative_path": destination,
                "modified_at": _mtime_iso(source),
                "size_bytes": source.stat().st_size,
                "study_id": summary.get("study_id", ""),
                "report_version": summary.get("report_version", ""),
                "status": status,
                "decision": summary.get("decision", ""),
                "next_step": summary.get("next_step", ""),
                "generated_or_finished_at": summary.get("finished_at")
                or summary.get("generated_at", ""),
            }
        )
    base.write_csv(
        package_root / "17_RESEARCH_REPORT_INDEX.csv",
        [
            "relative_path",
            "modified_at",
            "size_bytes",
            "study_id",
            "report_version",
            "status",
            "decision",
            "next_step",
            "generated_or_finished_at",
        ],
        rows,
    )
    return rows


def write_category_coverage(
    package_root: Path,
    builder: base.PackageBuilder,
    excluded_rows: Iterable[dict[str, Any]],
) -> None:
    included_counts: Counter[str] = Counter()
    included_bytes: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    excluded_bytes: Counter[str] = Counter()
    for destination, source in builder.mappings.items():
        category = destination.split("/", 1)[0]
        included_counts[category] += 1
        included_bytes[category] += source.stat().st_size
    for row in excluded_rows:
        relative = str(row["source_relative_path"])
        category = relative.split("/", 1)[0]
        excluded_counts[category] += 1
        excluded_bytes[category] += int(row.get("size_bytes") or 0)
    categories = sorted(set(included_counts) | set(excluded_counts))
    rows = [
        {
            "category": category,
            "included_file_count": included_counts[category],
            "included_bytes": included_bytes[category],
            "excluded_file_count": excluded_counts[category],
            "excluded_bytes": excluded_bytes[category],
        }
        for category in categories
    ]
    base.write_csv(
        package_root / "19_CATEGORY_COVERAGE.csv",
        [
            "category",
            "included_file_count",
            "included_bytes",
            "excluded_file_count",
            "excluded_bytes",
        ],
        rows,
    )


def _nested(value: dict[str, Any], *keys: str, default: Any = "NO_VIEW") -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def current_status_snapshot() -> str:
    authority = base.load_json("reports/audit/priority_forward_authoritative_status_v1_6.json")
    current = base.load_json("reports/audit/priority_forward_research_status_current.json")
    pcf_task = base.load_json("reports/data_quality/510300_primary_market_task_status_v1_2_1.json")
    cross_etf = base.load_json("reports/research/510300_cross_etf_forced_flow_binary_screen_v1.json")
    quality = base.load_json("reports/research/a_share_hs_pit_quality_conservative_investment_v1.json")
    afhc = base.load_json("reports/research/a_share_hs_fund_breadth_trend_confirmation_v1_0_1.json")
    digital = base.load_json("reports/research/recent_digital_asset_strategy_comparison_2023_2026.json")
    regime = base.load_json("reports/research/regime_aware_strategy_allocator_v1_status.json")
    generated_at = datetime.now(base.TIME_ZONE).isoformat()
    return f"""# 当前状态快照

生成时间：{generated_at}

此文件只做导航；冲突时回到对应冻结协议、机器可读报告和同次运行回执。

| 分支 | 当前状态/判断 | 直接证据 | 关键解释 |
|---|---|---|---|
| 510300二元20个百分点研究 | `PAUSED_AFTER_CROSS_ETF_FORCED_FLOW_SCREEN_V1` / `NO_VERIFIED_STRATEGY` | `reports/research/510300_BINARY_EXCESS20_RESEARCH_PAUSE_SNAPSHOT_20260828_V2.md` | 历史机制搜索暂停；无信号、Paper/Shadow、仓位或订单 |
| 跨ETF官方周度份额流 | `{cross_etf.get('status', 'NO_VIEW')}` | `reports/research/510300_cross_etf_forced_flow_binary_screen_v1.json` | 0个双20通过者，不改规则救援 |
| 优先前瞻权威状态 | `{authority.get('overall_research_status', 'NO_VIEW')}` / `{authority.get('decision', 'NO_VIEW')}` | `reports/audit/priority_forward_authoritative_status_v1_6.json` | 生成于 `{authority.get('generated_at', 'NO_VIEW')}` |
| PCF/IOPV | `{_nested(current, 'directions', 'primary_market_pcf_iopv', 'status')}`；当日任务 `{pcf_task.get('collection_status', pcf_task.get('status', 'NO_VIEW'))}` | `reports/data_quality/510300_primary_market_task_status_v1_2_1.json` | 完整质量日 `{_nested(current, 'directions', 'primary_market_pcf_iopv', 'full_coverage_days')}/20`；外部源失败不是质量日 |
| 行业预期差 | `{_nested(current, 'directions', 'industry_expectation_gap', 'status')}` | `reports/audit/priority_forward_research_status_current.json` | 原点 `{_nested(current, 'directions', 'industry_expectation_gap', 'origin_cluster_count')}`、成熟 `{_nested(current, 'directions', 'industry_expectation_gap', 'mature_origin_cluster_count')}`、60日块 `{_nested(current, 'directions', 'industry_expectation_gap', 'non_overlapping_60d_block_count')}` |
| 全A股点时质量—保守投资 | `{quality.get('status', 'NO_VIEW')}` | `reports/research/a_share_hs_pit_quality_conservative_investment_v1.json` | P0失败，停止该线 |
| AFHC基金宽度 | `{afhc.get('status', 'NO_VIEW')}` | `reports/research/a_share_hs_fund_breadth_trend_confirmation_v1_0_1.json` | P0失败，P1/P2/P3不运行 |
| 多资产40个百分点/高Sharpe | `NO_VERIFIED_40PCT_ANNUALIZED_NET_EXCESS_OR_HIGH_SHARPE_CANDIDATE` | `reports/research/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_CLOSEOUT_20260827.md` | 合同失败与冻结拒绝均保留 |
| 数字资产近期比较 | `{digital.get('status', 'NO_STRATEGY_PASSED_20PP_GATE')}` | `reports/research/RECENT_DIGITAL_ASSET_STRATEGY_COMPARISON_2023_2026.md` | 保留真实资产覆盖、成本、年度稳定性、Sharpe和回撤 |
| 状态感知配置器 | `{regime.get('status', regime.get('overall_status', 'NO_VIEW_CASH_ONLY'))}` | `reports/research/REGIME_AWARE_STRATEGY_ALLOCATOR_V1_STATUS.md` | 缺当前环境证据，0个合格组件 |

## 需要GPT Pro特别核对的状态差异

`priority_forward_authoritative_status_v1_6.json` 在2026-08-28 19:03生成，但其PCF“latest_task”仍指向2026-08-26的 `PROGRAM_FAILED`；同日更直接的 `510300_primary_market_task_status_v1_2_1.json` 和 `priority_forward_research_status_current.json` 已记录2026-08-28 `EXTERNAL_FREE_SOURCE_FAILED`。这两者不能互相覆盖：前者是权威总览，后者是更具体的当日采集证据。请审查权威状态生成器为什么没有吸收V1.2.1最新任务，而不要把任一状态误写为采集成功或策略结果。

## 全局执行状态

研究包不授权执行。当前信号、Paper/Shadow、仓位映射、订单生成、券商连接和实盘均保持关闭；交付ZIP也不改变这些状态。
"""


def _git_value(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=base.ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "UNAVAILABLE"
    return completed.stdout.strip() if completed.returncode == 0 else "UNAVAILABLE"


def write_snapshot_metadata(
    package_root: Path,
    started_at: str,
    builder: base.PackageBuilder,
    recent_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
) -> None:
    changed = _git_value("diff", "--name-only")
    staged = _git_value("diff", "--cached", "--name-only")
    base.write_json(
        package_root / "20_SNAPSHOT_METADATA.json",
        {
            "schema_version": "1.0.0",
            "package_id": PACKAGE_BASENAME,
            "snapshot_date": SNAPSHOT_DATE,
            "capture_started_at": started_at,
            "metadata_generated_at": datetime.now(base.TIME_ZONE).isoformat(),
            "source": "CURRENT_WINDOWS_WORKTREE_INCLUDING_UNTRACKED_FILES_SELECTED_BY_POLICY",
            "snapshot_atomicity": "BEST_EFFORT_FILESYSTEM_SNAPSHOT_PROCESSES_NOT_FROZEN",
            "git_head": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_modified_file_count": 0 if changed == "" else len(changed.splitlines()),
            "staged_file_count": 0 if staged == "" else len(staged.splitlines()),
            "included_source_file_count": len(builder.mappings),
            "recent_included_file_count_since_2026_08_25": len(recent_rows),
            "research_report_index_count": len(report_rows),
            "research_rerun_performed": False,
            "frozen_protocol_modified_by_packager": False,
            "data_security_audit_performed": False,
        },
    )


def main() -> None:
    started_at = datetime.now(base.TIME_ZONE).isoformat()
    builder = base.PackageBuilder()
    collect_sources(builder)
    source_bytes = sum(path.stat().st_size for path in builder.mappings.values())
    print(
        f"纳入源文件 {len(builder.mappings)} 个，共 {source_bytes / 1024 / 1024:.2f} MiB。",
        flush=True,
    )
    excluded_rows, excluded_directories = base.build_exclusion_inventory(builder)
    excluded_bytes = sum(int(row.get("size_bytes") or 0) for row in excluded_rows)
    unhashed_secret_count = sum(
        row["exclusion_reason"] == "SECRET_ENV_FILE_EXCLUDED_NOT_HASHED"
        for row in excluded_rows
    )

    base.DELIVERABLES.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qra_20260828_build_") as temp_name:
        package_root = Path(temp_name) / INTERNAL_ROOT
        package_root.mkdir(parents=True, exist_ok=True)
        builder.copy_all(package_root)

        base.write_text(
            package_root / "00_README_FIRST.md",
            readme_text(
                len(builder.mappings),
                source_bytes,
                len(excluded_rows),
                excluded_bytes,
            ),
        )
        base.write_text(package_root / "01_GPT_PRO_REVIEW_PROMPT.md", review_prompt_text())
        base.write_text(
            package_root / "02_USER_THOUGHTS_CONSTRAINTS_AND_OPEN_QUESTIONS.md",
            user_thoughts_text(),
        )
        base.write_text(
            package_root / "03_PROJECT_MAP_AND_CURRENT_JUDGMENTS.md",
            project_map_text(),
        )
        base.write_text(
            package_root / "04_BRANCH_BOUNDARIES_AND_CONTRADICTION_RULES.md",
            branch_boundaries_text(),
        )
        base.write_text(
            package_root / "05_EXCLUSIONS_REPRODUCTION_AND_SAFETY.md",
            exclusions_text(excluded_rows, excluded_directories),
        )
        base.write_text(package_root / "14_REPRODUCTION_NOTES.md", REPRODUCTION_NOTES)
        base.write_text(package_root / "15_UPLOAD_MESSAGE.txt", UPLOAD_MESSAGE + "\n")
        base.write_text(
            package_root / "18_CURRENT_STATUS_SNAPSHOT.md",
            current_status_snapshot(),
        )

        base.write_provenance(package_root, builder)
        base.write_exclusions(package_root, excluded_rows, excluded_directories)
        base.write_status_index(package_root)
        base.write_prior_review_index(package_root, excluded_rows)
        recent_rows = write_recent_work_index(package_root, builder)
        report_rows = write_research_report_index(package_root, builder)
        write_category_coverage(package_root, builder, excluded_rows)
        write_snapshot_metadata(
            package_root,
            started_at,
            builder,
            recent_rows,
            report_rows,
        )

        base.write_json(
            package_root / "09_CHECK_SCOPE.json",
            {
                "status": "FAST_UNREDACTED_NO_DATA_SECURITY_AUDIT_BY_USER_REQUEST",
                "data_security_audit_performed": False,
                "privacy_scan_performed": False,
                "secret_scan_performed": False,
                "malware_scan_performed": False,
                "text_redaction_performed": False,
                "per_file_sha256_performed": False,
                "fresh_extraction_performed": False,
                "zip_crc_performed": True,
                "duplicate_member_check_performed": True,
                "whole_zip_sha256_performed": True,
            },
        )
        base.write_json(
            package_root / "10_BUILD_SCOPE_AND_COUNTS.json",
            {
                "schema_version": "2.0.0",
                "package_id": PACKAGE_BASENAME,
                "snapshot_date": SNAPSHOT_DATE,
                "status": "FAST_REVIEW_SOURCE_SELECTION_COMPLETE",
                "included_source_file_count": len(builder.mappings),
                "included_source_bytes": source_bytes,
                "included_categories": base.summarize_categories(builder),
                "recent_included_file_count": len(recent_rows),
                "research_report_index_count": len(report_rows),
                "excluded_file_count": len(excluded_rows),
                "excluded_file_bytes": excluded_bytes,
                "excluded_hashed_file_count": 0,
                "excluded_directory_scope_count": len(excluded_directories),
                "sensitive_env_file_count_excluded_unhashed": unhashed_secret_count,
                "data_security_audit_performed": False,
                "research_rerun_performed": False,
                "frozen_protocol_modified": False,
                "position_mapping_enabled": False,
                "order_generation_enabled": False,
                "broker_connection_enabled": False,
                "live_trading_enabled": False,
            },
        )

        file_index_rows = base.write_file_index(package_root)
        base.create_zip(package_root, base.ZIP_PATH)
        crc_result = base.verify_zip_crc_only(base.ZIP_PATH)

    zip_hash = base.sha256_file(base.ZIP_PATH)
    base.write_text(base.SHA256_PATH, f"{zip_hash}  {base.ZIP_PATH.name}\n")
    base.write_text(base.UPLOAD_MESSAGE_PATH, UPLOAD_MESSAGE + "\n")
    receipt = {
        "schema_version": "2.0.0",
        "package_id": PACKAGE_BASENAME,
        "status": "PASS_UPLOAD_READY_GPT_PRO_REVIEW_ZIP",
        "started_at": started_at,
        "completed_at": datetime.now(base.TIME_ZONE).isoformat(),
        "snapshot_date": SNAPSHOT_DATE,
        "zip_path": base.ZIP_PATH.relative_to(base.ROOT).as_posix(),
        "zip_internal_root": INTERNAL_ROOT,
        "zip_size_bytes": base.ZIP_PATH.stat().st_size,
        "zip_sha256": zip_hash,
        "included_source_file_count": len(builder.mappings),
        "included_source_bytes": source_bytes,
        "indexed_file_count_excluding_index": len(file_index_rows),
        "zip_member_count_including_index": crc_result["member_count"],
        "recent_included_file_count": len(recent_rows),
        "research_report_index_count": len(report_rows),
        "excluded_file_count": len(excluded_rows),
        "excluded_file_bytes": excluded_bytes,
        "excluded_hashed_file_count": 0,
        "excluded_directory_scope_count": len(excluded_directories),
        "data_security_audit_status": "SKIPPED_BY_USER_REQUEST",
        "privacy_scan_status": "SKIPPED_BY_USER_REQUEST",
        "secret_scan_status": "SKIPPED_BY_USER_REQUEST",
        "malware_scan_status": "SKIPPED_BY_USER_REQUEST",
        "text_redaction_performed": False,
        "per_file_sha256_performed": False,
        "zip_crc_verification": crc_result,
        "fresh_extraction_verification": "SKIPPED_BY_USER_REQUEST",
        "snapshot_atomicity": "BEST_EFFORT_FILESYSTEM_SNAPSHOT_PROCESSES_NOT_FROZEN",
        "research_rerun_performed": False,
        "frozen_protocol_modified": False,
        "shadow_authorized": False,
        "position_mapping_enabled": False,
        "orders_generated": False,
        "broker_connected": False,
        "live_authorized": False,
    }
    base.write_json(base.RECEIPT_PATH, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
