"""将已完成的日内隔夜研究及审核上下文打包；不训练、不回测。"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
STUDY = "510300_INTRADAY_OVERNIGHT_INCREMENT_V1"
RESULT_REL = "reports/research/510300_intraday_overnight_increment_v1"
RESULTS = ROOT / RESULT_REL
STEM = "510300_INTRADAY_OVERNIGHT_INCREMENT_V1_CHATGPT_REVIEW_20260905"
DEST = ROOT / "deliverables"
LATEST_TEXT = Path(r"E:\CodexData\.codex\attachments\ce324d3a-6779-47e7-9293-618526394f1c\pasted-text.txt")
FIRST_TEXT = Path(r"E:\CodexData\.codex\attachments\b33d10e4-d0c1-4cee-aa3a-6fe52b2806aa\pasted-text.txt")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    DEST.mkdir(exist_ok=True)
    zip_path = DEST / f"{STEM}.zip"
    require(not zip_path.exists(), "同名压缩包已存在，不能覆盖已交付版本")
    manifest = json.loads((RESULTS / "freeze_manifest.json").read_text(encoding="utf-8"))
    receipt = json.loads((RESULTS / "receipt.json").read_text(encoding="utf-8"))
    delivery = json.loads((RESULTS / "delivery_receipt.json").read_text(encoding="utf-8"))
    result = json.loads((RESULTS / "result.json").read_text(encoding="utf-8"))
    payload: dict[str, bytes] = {}
    roles: dict[str, str] = {}

    def add_bytes(name: str, content: bytes, role: str) -> None:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "包内路径必须为相对逻辑路径")
        if name in payload:
            require(payload[name] == content, f"同一路径有不同字节：{name}")
            return
        payload[name], roles[name] = content, role

    def add_project(relative: str, role: str) -> None:
        relative = PurePosixPath(relative.replace("\\", "/")).as_posix()
        source = ROOT / relative
        require(source.is_file(), f"本轮所需文件缺失：{relative}")
        add_bytes("project/" + relative, source.read_bytes(), role)

    def add_text(name: str, text: str, role: str = "导航与审核说明") -> None:
        add_bytes(name, (text.rstrip() + "\n").encode("utf-8"), role)

    for relative, identity in manifest["files"].items():
        add_project(relative, "本轮冻结的代码、协议、测试与直接输入")
        require(sha(payload["project/" + relative]) == identity["sha256"], f"冻结文件发生变化：{relative}")
    result_files = sorted(path for path in RESULTS.rglob("*") if path.is_file())
    for path in result_files:
        add_project(path.relative_to(ROOT).as_posix(), "本轮完整结果目录")
    for relative, identity in receipt["output_files"].items():
        require(sha(payload[f"project/{RESULT_REL}/{relative}"]) == identity["sha256"], f"主运行输出发生变化：{relative}")
    for relative, identity in delivery["supplement_files"].items():
        add_project(relative, "既有交付图表与核对代码")
        require(sha(payload["project/" + relative]) == identity["sha256"], f"既有交付文件变化：{relative}")
    require(sha(payload[f"project/{RESULT_REL}/receipt.json"]) == delivery["primary_receipt_sha256"], "主回执与既有交付回执不符")

    context_files = [
        "RESEARCH_STATUS.md", "docs/DECISIONS.md", "config/research_registry.jsonl",
        "config/510300_research_authority_v4.json",
        "docs/510300_DAILY_01_OVERNIGHT_ABSORPTION_V1_SPEC.md",
        "docs/510300_DAILY_01_OVERNIGHT_ABSORPTION_V1_EVALUATION_ADDENDUM.md",
        "config/daily_01_overnight_absorption_v1.yaml",
        "config/daily_01_overnight_absorption_v1_evaluation.yaml",
        "reports/discovery/daily_01_overnight_absorption_v1_result.md",
        "reports/discovery/daily_01_overnight_absorption_v1_result.json",
        "reports/data_quality/daily_01_overnight_absorption_v1_data_gate.json",
        "reports/research/510300_RETURN_ANATOMY_V1.md",
        "reports/research/510300_return_anatomy_v1.json",
        "data/reference/a_share_hs_trading_calendar_2010_2026_v1_receipt.json",
        "data/reference/a_share_hs_trading_calendar_2010_2026_v1.csv",
        "reports/data_quality/A_SHARE_HS_TRADING_CALENDAR_DATE_ONLY_V1.json",
        "reports/data_quality/510300_dividends_quality.json",
        Path(__file__).relative_to(ROOT).as_posix(),
    ]
    for relative in context_files:
        add_project(relative, "必要的来源、旧研究关联或打包上下文")
    # 来源证据仅沿本轮直接数据回执展开一层，不递归收录其他研究树。
    price_receipt = json.loads(payload["project/reports/data_quality/510300_downside_risk_inputs_v1.json"])
    source_checks = []
    for relative, expected in price_receipt["input_hashes"].items():
        add_project(relative, "已核验价格底座的直接交叉来源")
        actual = sha(payload["project/" + relative])
        source_checks.append({"path": "project/" + relative, "expected_sha256": expected, "actual_sha256": actual, "matches": expected == actual})
    coverage = json.loads(payload["project/data/reference/510300_dividends_coverage.json"])
    for snapshot in coverage["official_source_snapshots"]:
        relative = snapshot["saved_file"]
        add_project(relative, "现金分红官方公告原件")
        actual = sha(payload["project/" + relative])
        source_checks.append({"path": "project/" + relative, "expected_sha256": snapshot["sha256"], "actual_sha256": actual, "matches": actual == snapshot["sha256"]})

    add_bytes("user_context/01_EARLIER_REVIEW_SUPERSEDED_ON_DIRECTION.txt", FIRST_TEXT.read_bytes(), "第一文字框，研究方向冲突以第二文字框为准")
    add_bytes("user_context/02_AUTHORIZED_SECOND_TEXT.txt", payload["project/docs/510300_INTRADAY_OVERNIGHT_INCREMENT_V1_USER_TEXT_20260905.txt"], "第二文字框，本轮研究实施依据")
    add_bytes("user_context/03_POST_RESULT_COMMENTS_AND_NEXT_PROPOSALS.txt", LATEST_TEXT.read_bytes(), "本次用户附上的结果评论与未执行后续建议")

    dependencies = {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "pytest", "matplotlib", "tzdata"]}
    add_text("review_tools/requirements_observed.txt", "\n".join(f"{name}=={version}" for name, version in dependencies.items()), "已观察到的运行库版本")
    add_text("review_tools/pytest.ini", "[pytest]\naddopts =\n", "限定本轮合成测试的配置")

    readme = f"""# 510300 日内—隔夜条件增量：ChatGPT 审核包

本包是已经完成的 `{STUDY}` 研究材料。核心代码、冻结输入、全部原始结果、账户账本和用户意见均已收录，可在离线条件下阅读和核对本轮证据。它不是全项目所有历史研究的归档。

研究结论为 `{result['status']}`。本轮历史训练242个五日原点，评价320个成熟原点；训练截止2019-12-31，连续账户从2020-01-02开盘至2026-08-14开盘，期间不重训。

## 推荐阅读顺序

1. 阅读 [01_CHATGPT_REVIEW_PROMPT.md](01_CHATGPT_REVIEW_PROMPT.md)，明确审核问题与输出要求。
2. 阅读 [02_RESULTS_AND_READING_MAP.md](02_RESULTS_AND_READING_MAP.md) 和 [本轮完整报告](project/{RESULT_REL}/REPORT.md)。
3. 阅读 [第二文字框](user_context/02_AUTHORIZED_SECOND_TEXT.txt)、[实际冻结协议](project/docs/510300_INTRADAY_OVERNIGHT_INCREMENT_V1_PROTOCOL_20260905.md)、[实际配置](project/config/510300_intraday_overnight_increment_v1.json)。
4. 对照 [完整样本及预测](project/{RESULT_REL}/conditional_samples.csv)、[模型参数](project/{RESULT_REL}/models.json)、[代码](project/research/intraday_overnight_increment_v1.py) 以及 `project/{RESULT_REL}/ledgers/`、`trades/`、`decisions/`。
5. 阅读 [最新结果评论与后续建议](user_context/03_POST_RESULT_COMMENTS_AND_NEXT_PROPOSALS.txt) 及 [已完成/未完成边界](03_COMPLETED_AND_PENDING.md)。
6. 需要核对文件结构时使用 [只读校验器](review_tools/verify_package.py)；复核环境见 [04_REPRODUCIBILITY.md](04_REPRODUCIBILITY.md)。

`project/` 保留原项目逻辑相对路径，原冻结报告中的相对引用在此目录下解释。`FILE_INDEX.csv` 列出各文件的角色、字节数及SHA-256，索引自身不作递归自哈希。

## 当前事实与研究边界

- M1基础成本净累计-4.32%，净夏普0.018；压力成本净累计+9.29%，净夏普0.167。M1两种费用下都落后于对应买入持有。
- 基础M0净累计+18.54%，M1基础相对M0没有净增量；M1的预测MSE恶化1.6656%，训练与评价的D20条件方向反转。
- 没有重新拟合最终全历史模型。D20表征已按冻结规则停止。
- 本次包装没有运行最新文字提出的三项归因，也没有运行A/H来源核验或新策略。
- 所有历史均已被项目反复观察，不能称为真正未见样本外。`D20` 在本轮表示20日分歧特征，主要可交易标签是五日；旧DAILY_01里的D20目标含义不同，不要混淆。
- 当前模型ABSTAIN、目标UNSET、真实持仓未知，仓位影响0。只涉及510300与人民币现金的假想账户。

包内负结论需要接受审核，而不是要求审核者赞同。发现问题时请注明证据与影响，区分实现错误、研究设计不足、真实负结果和暂不能判定。
"""
    add_text("00_README_FIRST.md", readme)

    prompt = f"""# 请对这份510300研究做证据驱动的独立审核

请先完整读取本包导航、协议、代码、配置、样本、账本、报告和用户最新意见。需要你审查研究质量及结论是否成立，不是宣传策略，也不是继续给M0增加一个变量。

## 研究目标与已知边界

本研究只在假想账户中执行510300.SH与CASH_CNY。它检验：控制过去20日总收益及波动后，日内—隔夜分歧是否增加下一开盘到第六个开盘、五个完整交易日的含分红持有收益信息。M0与M1采用相同规则，M1只新增D20；2019年训练截止，2020年起固定参数回放。20万元、T+1、100份整手、基础及压力费用均事前固定。净夏普目标1.2。

当前裁决为STOP_REPRESENTATION_NO_PARAMETER_RESCUE，未最终全历史拟合。全部历史已被此前项目观察；不得称为独立未见样本外。用户希望继续利用既有历史研究，未来收益标签只用来检验最终冻结模型，不再训练或调参。

## 优先回答的问题

1. 第二文字框、实际协议、实现与结果是否一致？请检查20日特征、岭回归的mean-MSE惩罚0.1、训练段标准化、截距、标签成熟及截止隔离、评价期是否真正不重训。指出额外实施选择是否合理，不因事先固定就认可其经济合理性。
2. 信息与收益时钟是否准确？新买入是否排除了入场前跳空；登记、除息、付款与应收是否正确；日内/隔夜分解是否守恒；下一开盘固定份额、资金缩量、T+1、价格限制、整手和期末清算是否真实对应代码；买持分红现金留存与模型口径是否一致。
3. 核对预测和账户数字，判断当前负结论能支持多大范围。M1基础MSE较M0恶化1.6656%不是投资损失率；统计区间跨零与实际经济表现差须分别陈述。检查年度贡献、回撤、平均暴露、净夏普和20日区块不确定性，不要求每年都显著，也不把日数当独立事件。
4. M0是否只是比较用基准，还是已有证据支持的决策结构？基础M0/M1费用只差约47.49元，终值却相差约4.57万元；对失误来源能作哪些已证实或尚不能证实的判断？不要把全部差额未经归因就归为成本、择时协方差或某个年份。
5. 基础、压力及零费用情景均按各自费用生成动作，不能直接当成同一路径的纯费用对照。请核查这一点，并判断Q评分、离散档位和路径依赖是否可能放大预测误差。区别可以从现有数据确认的事实与需要新归因表才能确认的解释。
6. 对最新用户文字中的三项后续归因，判断是否必要、能否仅凭现有账本完成以及最小正确口径。固定订单的压力成本若使现金不足，应标明不可执行，不悄悄减量；平均暴露+择时协方差是算术归因，不冒充复利精确分解；边界分析只解释原决策，不生成优化策略。
7. 核查本轮与旧DAILY_01和无条件收益解剖的关系，以及重复使用同一历史的选择偏差。项目登记不是完整多重试验账本的证明。不要将D20换名、反向、换窗、换训练截止点或新增过滤器后作为成功结果。
8. A/H信息在最新文字中只是未核验建议。本包未提供完整A/H数据与可行性结论。可以评价它需要哪些证据，不得把该建议写成已验证可用来源或新Alpha；无需在本次审核中开展大型新数据工程。

## 希望你返回的审核结构

- 首先给出总判断：现有结果是否可复核、停止当前表征是否有依据、哪些结论表述过度。
- 给出问题表：优先级P0/P1/P2、文件及行号或CSV行/日期、观察证据、为什么影响结论、最小更正或补充工作。没有证据的问题写为疑问，不编造缺陷。
- 分别评价信息增量、预测模型、交易映射和统计可信度，避免用一个PASS/FAIL覆盖所有层。
- 按最新文字给出三项归因的必要性和边界；若建议实施，限定为现有订单/预测/账本诊断，不修改D20，不生成更好曲线。
- 明确列出实际读取、复算、运行过的文件或检查。仅阅读不能说完整复跑，包内旧核对通过也不能冒充你自己的独立验证。
- 提供一段可直接交回执行端的简短结论与下一步清单。不要输出当前买卖建议、仓位或实盘授权。

请先审阅已完成的研究，再评价后续建议。若包内资料不足，明确指出缺少什么以及受影响的具体判断，不用猜测补足。
"""
    add_text("01_CHATGPT_REVIEW_PROMPT.md", prompt)

    lines = ["# 结果与证据导航", "", "下表直接来自本轮result.json，不是新回测。", "",
             "| 情景 | 模型 | 净累计收益 | 年化收益 | 夏普 | 最大回撤 | 总费用(元) |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["comparisons"]:
        sharpe_value = "不可定义" if row["net_sharpe"] is None else f"{row['net_sharpe']:.3f}"
        lines.append(f"| {row['scenario']} | {row['model']} | {row['cumulative_return']:.2%} | {row['annualized_return']:.2%} | {sharpe_value} | {row['max_drawdown']:.2%} | {row['total_cost_cny']:,.2f} |")
    lines.extend(["", "| 审核问题 | 主要文件 |", "|---|---|",
                  f"| 研究结论和区间 | `project/{RESULT_REL}/REPORT.md`、`result.json` |",
                  f"| 完整同口径比较 | `project/{RESULT_REL}/comparison.csv` |",
                  f"| 年度表现及金额贡献 | `project/{RESULT_REL}/annual_contributions.csv` |",
                  f"| 标签、训练资格和预测 | `project/{RESULT_REL}/conditional_samples.csv` |",
                  f"| 特征和分红恒等式 | `project/{RESULT_REL}/feature_accounting.csv` |",
                  f"| 原始/修正价格、分红、日历 | `project/config/510300_intraday_overnight_increment_v1.json`中的inputs，文件均在包内 |",
                  f"| 冻结参数与预测函数 | `project/{RESULT_REL}/models.json`、`project/research/intraday_overnight_increment_v1.py` |",
                  f"| 每日现金、份额、应收、净值 | `project/{RESULT_REL}/ledgers/`，九个账户 |",
                  f"| 固定请求、实际模拟成交、费用 | `project/{RESULT_REL}/trades/` |",
                  f"| 收盘时选择的动作及Q分数 | `project/{RESULT_REL}/decisions/`，只保存最优动作，全部候选评分尚未另建表 |",
                  f"| 基础/压力路径差异 | `project/{RESULT_REL}/DELIVERY_NOTE.md`、`delivery_checks.json` |",
                  f"| 冻结与领用证据 | `project/{RESULT_REL}/freeze_manifest.json`、`run_claim.json`、`receipt.json` |",
                  "| 合成测试 | `project/tests/test_intraday_overnight_increment_v1.py`及原15项通过日志 |", "",
                  "![既有净值图](project/" + RESULT_REL + "/comparison_chart.png)", ""])
    add_text("02_RESULTS_AND_READING_MAP.md", "\n".join(lines))

    add_text("03_COMPLETED_AND_PENDING.md", """# 已完成结果与未执行建议

| 内容 | 本地状态 | 如何使用 |
|---|---|---|
| M0/M1单次训练、五日预测比较 | 已完成并冻结 | 审核原结论 |
| 基础、压力、零费用三种决策情景及买持 | 已完成，九个账户 | 可核对完整账户；各情景动作不同 |
| 原合成测试与账户算术核对 | 已完成，保留原日志和回执 | 原作者检查证据，不等于本次外部审核 |
| D20当前表征 | 已停止，不作参数营救 | 不反向，不改窗，不换目标 |
| 最终全历史模型 | 未拟合，因保留条件不满足 | 不补写不存在的模型 |
| fixed_order_cost_attribution.csv | NOT_RUN，尚未生成 | 最新文字建议；原零费用情景不替代它 |
| exposure_timing_attribution.csv | NOT_RUN，尚未生成 | 最新文字建议；不能提前声称择时协方差为负 |
| decision_boundary_attribution.csv | NOT_RUN，尚未生成 | 最新文字建议；现有decisions只保存最优动作 |
| A/H来源、权限、覆盖与去重 | NOT_RUN | 只是待核验的信息路线 |
| 新因子/新开发程序 | NOT_STARTED | 无新模型、新收益或新授权 |

本轮用户的直接任务是将做出来的结果打包交给ChatGPT审核。因此新增文字作为审核上下文原样保存，本次没有把上述后续工作写成已经完成。

资料优先级：历史研究按第二文字框及实际冻结协议解释；第一文字框中与之冲突的资源安排不再支配本轮；最新文字是结果评论及下一步建议。所有用户提供的外部评论均不等于本地独立复跑证据。
""")

    add_text("04_REPRODUCIBILITY.md", f"""# 离线阅读与复核入口

原执行环境：Python {platform.python_version()}，Windows/PowerShell；依赖的实际版本见 `review_tools/requirements_observed.txt`。库本体和虚拟环境没有放入ZIP。CSV可直接阅读，Parquet需要相应读取库。

## 仅检查包文件是否完整

从解压后的包根目录运行 `python review_tools/verify_package.py`。它只用标准库读文件，核对FILE_INDEX、原冻结清单、主回执；不写文件、不训练、不回测、不需要凭据或网络。本次包装仅进行ZIP内结构核验，没有声称完成新鲜解压端到端重放。

## 合成测试

切换到 `project/` 后，可用已安装好依赖的Python执行 `python -m pytest -c ../review_tools/pytest.ini tests/test_intraday_overnight_increment_v1.py -q -p no:cacheprovider`。这里只测试合成价格与可手算账户，不读取真实回测结果重新选模型。包内保存的是原先15项通过日志，本次打包没有重跑测试。

## 原主运行为什么不能再直接运行

原主程序 `project/research/intraday_overnight_increment_v1.py` 保留一次性领用保护，完整包包含run_claim.json，重复运行会被阻止。不要删除领用或修改冻结清单绕过保护。外部审核如果需要算术复核，可以只读已有样本、模型参数和账本，在自己的临时目录输出核对结果；必须标明这是独立审核，不是新增研究候选。

`project/scripts/verify_and_plot_intraday_overnight_increment_v1.py` 是既有交付核对代码，会重写交付补充文件，且绘图使用Windows微软雅黑字体；原样保留供审阅。Linux阅读已有PNG/PDF无需这些字体，不要在唯一原包副本上运行写出步骤。

打包器原样收录为 `project/scripts/{Path(__file__).name}`，它依赖当前任务的附件路径，仅用于解释本包是如何构建的，不是审核者必须运行的入口。

包内project保持逻辑路径，数据是实际文件，不是指向本机E盘的Junction或外部链接。本轮直接输入和45项结果文件均收录。旧项目状态文件内可能引用不在本包的其他研究，只作背景，不能推断已收录全项目。
""")

    add_text("05_SCOPE_AND_CHECKS.md", """# 收录范围与交付检查边界

收录：本轮16项冻结依赖、完整45项结果文件、既有交付核对脚本、直接价格交叉来源、分红覆盖及14份官方原件、日期日历、旧DAILY_01必要协议和终止结果、无条件收益解剖报告、当前与历史权限背景、三段用户文字、阅读导航、审核提示词及文件索引。

不递归收录与本轮无关的其他研究、其他原始数据树、虚拟环境、Git对象库、缓存、日志集合、旧大型审阅ZIP及临时凭据文件。本包采用明确路径清单构建；这些范围选择不构成安全扫描。

本次执行的检查仅为：ZIP CRC、重复成员、索引成员及字节数、成员SHA-256、核心冻结/原结果引用覆盖、整个ZIP的SHA-256。新增源证据的原回执哈希一致性见PACKAGE_METADATA.json；有不一致时保留事实，不改旧回执。

本次未执行安全性、隐私、凭据、恶意文件或脱敏扫描，未重拟合、未重跑策略、未作新鲜解压端到端重放。包构建完成只代表可交付，不能写成ChatGPT已经审核或认可，也不改变任何交易权限。
""")

    verifier_source = '''"""只读核对解压包、原冻结依赖和原输出回执。"""
from pathlib import Path
import csv
import hashlib
import json

root = Path(__file__).resolve().parents[1]
errors = []
with (root / "FILE_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
    entries = list(csv.DictReader(stream))
for row in entries:
    path = root / row["path"]
    if not path.is_file():
        errors.append("缺失：" + row["path"])
        continue
    content = path.read_bytes()
    if len(content) != int(row["bytes"]) or hashlib.sha256(content).hexdigest() != row["sha256"]:
        errors.append("字节或哈希不符：" + row["path"])
project = root / "project"
results = project / "reports/research/510300_intraday_overnight_increment_v1"
for file, base, key in [(results / "freeze_manifest.json", project, "files"), (results / "receipt.json", results, "output_files")]:
    receipt = json.loads(file.read_text(encoding="utf-8"))
    for name, identity in receipt[key].items():
        path = base / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            errors.append("原始引用不符：" + name)
print(json.dumps({"状态": "通过" if not errors else "失败", "索引条目": len(entries), "错误": errors, "重新拟合或回测": False}, ensure_ascii=False, indent=2))
raise SystemExit(1 if errors else 0)
'''
    compile(verifier_source, "review_tools/verify_package.py", "exec")
    add_text("review_tools/verify_package.py", verifier_source, "标准库只读文件校验器")

    upload = """请解压附件，先阅读00_README_FIRST.md，再按照01_CHATGPT_REVIEW_PROMPT.md独立审核。包内有本轮510300日内—隔夜条件增量的代码、冻结协议、原始输入、预测与完整账户账本，以及我的最新结果评论。请优先判断负结论是否成立、M0/M1决策映射与成本路径是否合理、哪些解释还缺少归因证据。最新文字提出的三项归因和A/H可行性核验目前尚未执行，请区别已完成结果与后续建议。不要通过反向、换窗或调参营救D20；请给出有文件/行号证据的问题表和最小必要的下一步工作。"""
    add_text("UPLOAD_MESSAGE.txt", upload, "可复制给ChatGPT的上传文字")
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    metadata = {"package_id": STEM, "created_at": timestamp, "scope": "本轮D20/M0/M1完整研究与直接来源证据及审核上下文",
                "study_id": STUDY, "study_status": result["status"], "result_directory_file_count": len(result_files),
                "frozen_dependencies_complete": len(manifest["files"]), "original_result_references_complete": len(receipt["output_files"]),
                "official_dividend_snapshots": len(coverage["official_source_snapshots"]),
                "runtime": {"python": platform.python_version(), "libraries": dependencies},
                "latest_user_text_sha256": sha(LATEST_TEXT.read_bytes()),
                "direct_source_receipt_checks": source_checks,
                "pending_attributions": ["fixed_order_cost_attribution.csv", "exposure_timing_attribution.csv", "decision_boundary_attribution.csv"],
                "pending_attributions_status": "NOT_RUN", "ah_feasibility_status": "NOT_RUN",
                "new_training_runs": 0, "new_backtest_runs": 0, "external_chatgpt_review_performed": False,
                "verification_scope": ["ZIP_CRC", "DUPLICATE_MEMBERS", "INDEX_COVERAGE_AND_BYTES", "MEMBER_SHA256", "CORE_REFERENCE_COVERAGE", "WHOLE_ZIP_SHA256"],
                "SECURITY_AUDIT": False, "PRIVACY_SCAN": False, "SECRET_SCAN": False, "MALWARE_SCAN": False, "REDACTION": False,
                "FRESH_EXTRACTION_REPLAY": False, "position_impact": 0,
                "index_self_excluded": True}
    add_bytes("PACKAGE_METADATA.json", encode_json(metadata), "包元数据与范围")
    index_stream = io.StringIO(newline="")
    writer = csv.DictWriter(index_stream, fieldnames=["path", "role", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(payload):
        writer.writerow({"path": name, "role": roles[name], "bytes": len(payload[name]), "sha256": sha(payload[name])})
    add_bytes("FILE_INDEX.csv", index_stream.getvalue().encode("utf-8-sig"), "全部载荷索引，不递归索引自身")

    temporary = zip_path.with_suffix(".building.zip")
    require(not temporary.exists(), "已有同名未完成构包文件，需要保留并检查")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(payload.items()):
            archive.writestr(name, content)
    with zipfile.ZipFile(temporary, "r") as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "ZIP存在重复成员")
        require(set(names) == set(payload), "ZIP成员范围与收录清单不一致")
        require(archive.testzip() is None, "ZIP CRC检查失败")
        index_rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require({row["path"] for row in index_rows} | {"FILE_INDEX.csv"} == set(names), "索引覆盖不完整")
        for row in index_rows:
            content = archive.read(row["path"])
            require(len(content) == int(row["bytes"]) and sha(content) == row["sha256"], f"索引字节或哈希不符：{row['path']}")
        for relative in manifest["files"]:
            require("project/" + relative in names, f"核心依赖未收录：{relative}")
        for relative in receipt["output_files"]:
            require(f"project/{RESULT_REL}/{relative}" in names, f"原始结果未收录：{relative}")
    temporary.replace(zip_path)
    zip_hash = sha(zip_path.read_bytes())
    project_count = sum(name.startswith("project/") for name in payload)
    summary = {"package_id": STEM, "created_at": timestamp, "status": "PASS_ZIP_CRC_INDEX_HASH_AND_CORE_COVERAGE",
               "zip_path": str(zip_path), "zip_bytes": zip_path.stat().st_size, "zip_sha256": zip_hash,
               "zip_members": len(payload), "project_files": project_count, "total_uncompressed_bytes": sum(map(len, payload.values())),
               "result_directory_files_included": len(result_files), "frozen_dependencies_included": len(manifest["files"]),
               "original_output_references_included": len(receipt["output_files"]), "official_dividend_snapshots": len(coverage["official_source_snapshots"]),
               "duplicates": 0, "index_missing_or_extra": 0, "index_size_or_hash_mismatches": 0, "crc_pass": True,
               "source_receipt_mismatches": sum(not row["matches"] for row in source_checks),
               "new_training_runs": 0, "new_backtest_runs": 0, "SECURITY_AUDIT": False,
               "FRESH_EXTRACTION_REPLAY": False, "external_chatgpt_review_performed": False}
    (DEST / f"{STEM}_BUILD_RECEIPT.json").write_bytes(encode_json(summary))
    (DEST / f"{STEM}.sha256").write_text(f"{zip_hash}  {zip_path.name}\n", encoding="ascii")
    (DEST / f"{STEM}_UPLOAD_MESSAGE.txt").write_text(upload + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
