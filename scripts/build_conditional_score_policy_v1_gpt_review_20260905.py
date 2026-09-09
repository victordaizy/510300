"""打包本轮完整研究和必要历史背景，供GPT审阅并修订下一步方向。"""
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
STUDY = "510300_CONDITIONAL_SCORE_POLICY_V1"
REPORT = "reports/research/510300_conditional_score_policy_v1"
PARENT = "reports/research/510300_intraday_overnight_increment_v1"
STEM = "510300_CONDITIONAL_SCORE_POLICY_V1_GPT_REVIEW_20260905"
DESTINATION = ROOT / "deliverables"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def encoded(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


READ_ONLY_REVIEWER = '''"""只读取已保存账本复算统计，不拟合、调参、下载行情或生成新政策。"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="复算包内已保存账户的绩效汇总")
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    archive = zipfile.ZipFile(args.zip) if args.zip else None
    def read(name):
        return archive.read(name) if archive else (base / name).read_bytes()
    prefix = "project/reports/research/510300_conditional_score_policy_v1/"
    result = json.loads(read(prefix + "result.json"))
    config = json.loads(read("project/config/510300_conditional_score_policy_v1.json"))
    annual = config["annual_days"]
    checks = {}
    try:
        for key, saved in result["economics"].items():
            rows = list(csv.DictReader(io.StringIO(read(prefix + "evaluation/" + key + "_ledger.csv").decode("utf-8-sig"))))
            returns = [float(row["net_return"]) for row in rows]
            mean = statistics.mean(returns) * annual
            volatility = statistics.stdev(returns) * math.sqrt(annual)
            wealth = peak = 1.0
            drawdown = 0.0
            for value in returns:
                wealth *= 1 + value
                peak = max(peak, wealth)
                drawdown = min(drawdown, wealth / peak - 1)
            eligible = statistics.mean(float(row["exposure"]) for row in rows) >= config["minimum_mean_exposure_for_sharpe"]
            eligible = eligible and sum(float(row["shares"]) > 0 for row in rows) >= config["minimum_exposed_days_for_sharpe"] and volatility > 1e-8
            values = {"cumulative_return": wealth - 1,
                      "annualized_return": wealth ** (annual / len(rows)) - 1,
                      "annualized_arithmetic_mean": mean,
                      "annualized_volatility": volatility,
                      "net_sharpe": mean / volatility if eligible else None,
                      "max_drawdown": drawdown,
                      "total_friction": sum(float(row["commission"]) + float(row["slippage_cost"]) for row in rows)}
            errors = {}
            for name, value in values.items():
                if value is None:
                    if saved[name] is not None:
                        raise ValueError("不可用夏普被替换为数值")
                else:
                    errors[name] = abs(value - saved[name])
                    if errors[name] > 1e-8:
                        raise ValueError(f"账本复算不符：{key}/{name}")
            checks[key] = {"交易日数": len(rows), "最大汇总差额": max(errors.values()), "复算指标": values}
        print(json.dumps({"状态": "通过_只复算已保存账本", "账户": checks}, ensure_ascii=False, indent=2))
    finally:
        if archive:
            archive.close()

if __name__ == "__main__":
    main()
'''


def main() -> None:
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    DESTINATION.mkdir(exist_ok=True)
    archive_path = DESTINATION / f"{STEM}.zip"
    require(not archive_path.exists(), "同名审阅包已存在，不能覆盖既有交付")
    report_dir = ROOT / REPORT
    manifest = json.loads((report_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    model_freeze = json.loads((report_dir / "model_freeze.json").read_text(encoding="utf-8"))
    execution = json.loads((report_dir / "execution_receipt.json").read_text(encoding="utf-8"))
    verification = json.loads((report_dir / "delivery_verification.json").read_text(encoding="utf-8"))
    result = json.loads((report_dir / "result.json").read_text(encoding="utf-8"))
    require(execution["status"] == "COMPLETED" and verification["status"] == "PASS_DELIVERY_CHECKS", "本轮完成与交付回执不齐全")
    require(result["state"] == "REJECTED_FROZEN_CONDITIONAL_SCORE_POLICY_V1", "本轮终态发生变化，必须重新核对包装说明")
    payload: dict[str, bytes] = {}
    roles: dict[str, list[str]] = {}
    generated: set[str] = set()
    checks: list[dict] = []

    def add(name: str, content: bytes, role: str, *, generated_file: bool = False) -> None:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, f"非法包内路径：{name}")
        if name in payload:
            require(payload[name] == content, f"同一路径存在不同内容：{name}")
            if role not in roles[name]:
                roles[name].append(role)
            return
        payload[name], roles[name] = content, [role]
        if generated_file:
            generated.add(name)

    def project(relative: str, role: str, expected: str | None = None) -> None:
        relative = PurePosixPath(relative.replace("\\", "/")).as_posix()
        source = ROOT / relative
        require(source.is_file(), f"必要文件不存在：{relative}")
        content = source.read_bytes()
        add("project/" + relative, content, role)
        if expected:
            actual = sha(content)
            checks.append({"path": "project/" + relative, "expected_sha256": expected, "actual_sha256": actual, "matches": actual == expected})
            require(actual == expected, f"冻结凭证与实际文件不符：{relative}")

    def note(name: str, content: str) -> None:
        add(name, (content.rstrip() + "\n").encode("utf-8"), "本次打包生成的阅读与审阅材料", generated_file=True)

    current_files = sorted(p for p in report_dir.rglob("*") if p.is_file())
    for path in current_files:
        project(path.relative_to(ROOT).as_posix(), "本轮成果目录完整收录")
    for item in [*manifest["implementation"], *manifest["inputs"].values()]:
        project(item["path"], "本轮冻结协议、实现、测试及直接输入", item["sha256"])
    for item in model_freeze["development_artifacts"]:
        project(item["path"], "主评价前冻结的开发证据", item["sha256"])
    project(execution["result"]["path"], "完成回执对应的原始结果", execution["result"]["sha256"])
    project("scripts/verify_510300_conditional_score_policy_v1_delivery.py", "日表时间展示与交付复核代码", verification["verifier_sha256"])

    parent_manifest = json.loads((ROOT / PARENT / "freeze_manifest.json").read_text(encoding="utf-8"))
    for relative, item in parent_manifest["files"].items():
        project(relative, "上一轮D20研究的冻结背景与共享账户依赖", item["sha256"])
    parent_files = sorted(p for p in (ROOT / PARENT).rglob("*") if p.is_file())
    for path in parent_files:
        project(path.relative_to(ROOT).as_posix(), "上一轮D20已完成结果，仅作历史背景")

    price_receipt = json.loads(payload["project/reports/data_quality/510300_downside_risk_inputs_v1.json"])
    for relative, expected in price_receipt["input_hashes"].items():
        project(relative, "沿既有价格凭证展开一层的直接来源", expected)
    coverage = json.loads(payload["project/data/reference/510300_dividends_coverage.json"])
    for item in coverage["official_source_snapshots"]:
        project(item["saved_file"], "现金分红官方公告原件", item["sha256"])

    extras = [
        "RESEARCH_STATUS.md", "docs/RESEARCH_REVIEW_ZIP_DELIVERY_RULE.md",
        "docs/R6_RESEARCH_SPEC.md", "docs/R6_REJECTION_DECISION.md", "docs/R6_POSTMORTEM_PROTOCOL.md",
        "reports/backtest/r6_model_selection.json", "reports/backtest/r6_postmortem.json",
        "scripts/build_510300_downside_risk_inputs_v1.py", "scripts/download_510300_daily.py",
        "data/reference/a_share_hs_trading_calendar_2010_2026_v1_receipt.json",
        Path(__file__).relative_to(ROOT).as_posix(),
    ]
    for relative in extras:
        project(relative, "直接相关历史背景、来源说明或交付实现")

    source_file_count = len(payload)
    add("user_context/01_用户完整候选策略原文.txt", payload["project/docs/510300_CONDITIONAL_SCORE_POLICY_V1_USER_TEXT_20260905.txt"], "本轮用户完整方案原文")
    note("user_context/02_本次打包与后续交付要求.txt", "用户于2026-09-05明确要求：\n\n每次把成果做成压缩包，我要交给gpt审阅，修改下一步研究的策略和方向\n\n本包据此交付；GPT尚未返回外部审阅意见。")
    note("00_README_FIRST.md", f'''# 510300连续条件评分：GPT审阅与下一步研究决策包

本包完整收录 `{STUDY}` 的本轮成果，并附上一轮日内—隔夜D20研究结果及R6拒绝证据。它是本轮独立审阅包，不是整个项目历次所有研究的归档。

当前结果为 **未达到净夏普1.2**：完整模型基础/压力净夏普0.353/0.351，复合年化净收益1.054%/1.045%。研究已完整运行一次并冻结拒绝；请审查结论与实现是否成立，进一步决定下一轮值得研究的策略结构和方向。

## 阅读顺序

1. [01_GPT_REVIEW_PROMPT.md](01_GPT_REVIEW_PROMPT.md)：审阅任务、重点问题与要求的下一步输出。
2. [02_核心结果与待解问题.md](02_核心结果与待解问题.md)及[完整研究报告](project/{REPORT}/研究报告.md)。
3. [用户完整方案](user_context/01_用户完整候选策略原文.txt)、[冻结协议](project/docs/510300_CONDITIONAL_SCORE_POLICY_V1_PROTOCOL.md)、[固定配置](project/config/510300_conditional_score_policy_v1.json)。
4. [开发与冻结地图](03_文件与证据地图.md)：全部54次内层拟合、九组配置排名、最终系数和主评价读取时序。
5. [每日决策表](project/{REPORT}/每日决策表.csv)、八条逐日账户、成交与决策文件，以及[图表](project/{REPORT}/连续账户比较.png)。
6. [历史背景与状态边界](04_历史背景与状态边界.md)，明确这轮与D20、R6的联系及不能混用的口径。
7. [复核方法](05_复核方法与环境.md)；完整成员见[FILE_INDEX.csv](FILE_INDEX.csv)。

`project/`保留原项目相对路径。包内包含全部直接数据及来源凭证，可以离线查阅本轮计算所需证据，不要求重新下载行情。CSV、Markdown、JSON可先读；Parquet提供同一数值证据的类型保留版本。

用户希望每轮成果都交付此类压缩包，并用GPT的审阅意见修订下一步方向。请不要只重复失败摘要，也不要未经证据支持就宣布新方向可达到1.2。
''')
    prompt = f'''# GPT审阅任务：检查已完成研究，并修订下一步策略与研究方向

你收到的是已经完成的 `{STUDY}` 研究包。请实际读取文件，先核对事实和实现，再帮助用户决定下一轮最值得做的研究。若无法解压、读取关键表格或运行核对，请先明确列出障碍和未读取的文件，不要声称已经复核。

用户最新要求是：“每次把成果做成压缩包，我要交给gpt审阅，修改下一步研究的策略和方向”。因此你的交付必须同时包含研究质量审阅和具体的下一步策略设计。不要把主要篇幅用于重复规则、泛泛建议或两天审计计划。

## 已执行范围与事实

- 交易范围仅510300.SH和CASH_CNY，初始20万元，无做空和杠杆；基础/压力佣金分别万二/万四且每笔至少5元，单边滑点5bp/10bp，100份整手、0.001元价位、T+1，独立分红权益及现金账本，现金收益为零。
- 五项连续分数T/Q/B/D/R，六个受约束系数，Sigmoid分数20—80线性映射持仓；10个百分点不交易带，加仓最多25个百分点，减仓直接向目标，20分以下退出。
- 同一九组gamma/lambda用于完整与简单模型，三个内层验证年2017/2018/2019，共54次内层拟合、两次2015—2019最终拟合。总候选评估291,014次，每次固定72成员、80代预算，不改种子。两个最终拟合均为预算耗尽后的最佳可行解，并非已证明全局最优。
- 主模型/对照冻结后才在程序中读取2020年后的行情；2020-01-02至2026-08-14共1,604个连续交易日不重训。历史此前已被多次研究观察，不能把这称为重新获得了全新盲测样本。
- 完整基础/压力净夏普0.353/0.351，复合年化1.054%/1.045%，最大回撤约-6.21%/-6.22%。同口径买入持有年化约3.60%。完整优于简单对照的点估计，但20日区块差额区间跨零。
- 完整模型平均暴露14.49%，评分范围23.87—31.94，原始目标6.45%—19.91%，仅7个成交日且从未触发20分退出。删除修复项年化差额仅约0.002个百分点。
- 完整模型两成本的实际成交日期和份额一致，额外118.44元可由纯费用解释。本轮没有训练最终全历史未来模型，没有Paper/Shadow、自动化或实盘订单。

## 请重点回答的问题

1. 实现有没有实质性错误或违反用户原文的地方？检查五项公式、分红信号序列、收盘决策/次开盘执行、T+1、无成交顺延、实际暴露、不交易带、份额舍入、样本末日估值、训练批量账户与既有账户一致性。按文件和行号/字段举证，区分程序缺陷、设计取舍和未验证假设。
2. 目标与学习结构是否匹配？用户验收是两成本夏普>=1.2且收益超过买入持有，训练是最坏成本下均值—方差效用加正则，内层再按固定gamma=5的跨年效用排序。这样的组合是否偏向低仓位或弱变化政策？数据、截距、符号/强度约束、lambda、10个百分点不交易带和仅5年开发史，各有何证据？不要把未经分解的直觉当成已证实因果。
3. 九组开发配置不等于只观察九条政策。请评估291,014次内部候选评价、80代未收敛、三个验证年和以往历史研究带来的局限。拒绝结论能支持“本次固定实现和预算失败”到什么程度，不能推广到哪些更宽泛主张？
4. 当前三个对照加固定消融是否足以回答问题？完整对照差额有多少可能来自暴露、风险管理或交易触发？是否缺少同平均暴露、同波动静态基准？如确有必要，明确这类诊断能回答什么、不能回答什么，而不是增加无限检查项目。
5. R项的设计没有带来明显经济差额：是系数收缩、政策变化被不交易带吸收、样本信息不足、还是机制本身无效？哪些能由现有表格辨别，哪些需要独立新实验？不能因这一轮微小消融差额就证明所有修复机制无效。
6. 与上一轮D20和R6失败记录相比，本轮增加了什么信息？下一轮应停止哪类重复尝试，保留哪个有具体理由的研究问题？不要把若干失败策略平均、只降低gamma/lambda或缩窄不交易带，就自动包装成已验证新方向。

## 你应给出的最终输出

1. 一段明确总判断：本轮结论是否可靠，最关键的不足是什么，是否值得继续在现有日线信息集上投入，以及理由。
2. 最多8条重要问题表，列优先级、证据文件/行号/字段、影响、最小处理方式；只列与科学结论和账户正确性直接相关的问题。
3. **最多一个优先研究方向、一个备选方向**。每个方向都要有明确机制假设、与D20/R6/本轮的差异、可用数据、预期解决的问题、最可能失败原因、最小验证步骤和停止条件。如果没有值得继续的方向，明确说没有，并指出具体缺少的信息；不要为了给方案而给方案。
4. 若推荐继续，给出下一轮可直接交给Codex实施的独立研究任务书：模型或规则的明确形式、数据字段及可得时钟、训练/验证/主评价边界、优化与尝试预算、交易账户约束、必要对照、验收条件和交付物。把确定项和待决定项区分清楚，避免让执行者在看完主评价后再补规则。
5. 简短的“继续/暂停/停止”决策表，说明先做哪一步、何时停止。不要给出脱离本轮证据的长期数据平台建设清单。

当前净夏普1.2和可交易范围保持不变；如认为需要调整研究目标或扩展数据/资产范围，必须明说这是另一个待用户决定的目标或授权，不能悄悄降低验收线。原V1证据和终态保持冻结；可以提出有明确理由的独立新版研究，但不能回改原结果或宣称原V1通过。未来新增收益只检验冻结模型，不自动参与持续再训练。

本次审阅可以读取、检查和复算已有账本统计，不要自行开展新参数网格、重新训练或把主评价期挑成更好看的子区间。你的下一步任务书是供用户选择的建议，本包交付本身不代表用户已批准执行新研究。
'''
    note("01_GPT_REVIEW_PROMPT.md", prompt)
    note("02_核心结果与待解问题.md", f'''# 核心结果与待解问题

正式状态：`{result['state']}`；程序运行和交付均已完成，研究验收失败。

| 模型 | 基础夏普 | 压力夏普 | 基础复合年化 | 压力复合年化 |
|---|---:|---:|---:|---:|
| 买入持有 | 0.287 | 0.286 | 3.60% | 3.60% |
| 简单评分 | 0.001 | -0.010 | -0.06% | -0.09% |
| 完整评分 | 0.353 | 0.351 | 1.05% | 1.05% |
| 完整去修复 | 0.352 | 0.349 | 1.05% | 1.04% |

完整模型选gamma=10、lambda=0.1；系数依次为截距-0.890730637、T 0.052188398、Q 0.101781114、B 0.004114209、D 0.180322946、R 0.029517038。简单模型选gamma=10、lambda=0.001。精确浮点值见[模型冻结记录](project/{REPORT}/model_freeze.json)。

完整评分只有7个成交日，评分从未到达退出线，目标持仓最高约19.91%。这是已有账本事实；“为何形成这种政策”尚不能只靠一句归因下结论。训练目标和风险收益验收是否一致、正则与执行带如何共同影响变化、现有信息是否足够，是希望外部审阅重点解决的问题。

完整相对简单的基础年化差约+1.11个百分点，但95%区块区间约[-0.022,2.371]个百分点。删除修复项的经济差额很小。这些结果不证明所有连续评分或修复机制都无效。

基础总摩擦148.74元，压力总摩擦267.17元；两种实际成交路径一致，差额118.44元归于费用。不是上一轮D20不同费用造成不同订单路径的同一情况，不能把上一轮解释直接套入。

所有统计使用完整交易日、含现金日的日净收益，年化242天。年化收益为CAGR；末日按收盘估值，未扣未来退出费用。更细的年度、回撤、换手、现金、分红和不确定性数据均在[完整结果](project/{REPORT}/result.json)及账本中。
''')
    note("03_文件与证据地图.md", f'''# 文件与证据地图

| 审阅问题 | 主要文件 |
|---|---|
| 用户原方案与实际落地差异 | `user_context/01_用户完整候选策略原文.txt`；`project/docs/510300_CONDITIONAL_SCORE_POLICY_V1_PROTOCOL.md` |
| 五项公式、请求、账户与批量训练 | `project/research/conditional_score_policy_v1.py`；`project/research/intraday_overnight_increment_v1.py` |
| 54次内层开发和参数选择 | `project/{REPORT}/development/all_inner_results.csv`和JSON；同目录54个拟合JSON与108个验证账本 |
| 九组配置排名与最终系数 | `project/{REPORT}/development/configuration_ranking.csv`；`model_freeze.json` |
| 是否先冻结后读主评价 | `freeze_manifest.json`、`run_claim.json`、`stage_events.jsonl`、`model_freeze.json`和入口代码 |
| 每日分数、目标及实际成交 | `project/{REPORT}/每日决策表.csv`；`evaluation/*_decisions.csv`、`*_trades.csv`、`*_ledger.csv` |
| 原始导出与展示整理 | `每日决策表_原始导出.csv`；`delivery_verification.json`；`project/scripts/verify_510300_conditional_score_policy_v1_delivery.py` |
| 绩效、年度、费用、区间 | `result.json`；`evaluation/economic_comparison.csv`；`annual_contributions.csv`；`*_fixed_path_costs.csv` |
| 输入、分红原件及日历 | `project/data/`下本包明确收录的文件；冻结配置`inputs`、价格凭证`input_hashes`、分红`official_source_snapshots` |
| 测试及既有执行证据 | `project/tests/test_510300_conditional_score_policy_v1.py`；`test_intraday_overnight_increment_v1.py`；`pre_freeze_validation.json` |

未写出完整前缀的本轮文件均位于`project/{REPORT}/`。原成果目录的239个文件全部包含在内。`frozen_sources/`和`frozen_inputs/`保留当时副本，与`project/`中逻辑路径版本相互校验。

日表共3,210行，包括两种费用各1,605个收盘决策记录，其中各有一个初始现金锚。正式绩效每天一行，每条账户为1,604个交易日；不能把初始锚或两个费用情景相加当成独立历史样本。
''')
    note("04_历史背景与状态边界.md", f'''# 历史背景与状态边界

本轮是用户完整方案明确授权的独立持仓政策研究。不是原D20或R6重新通过，也不是更换名称取消原拒绝结论。

- 上一轮D20日内—隔夜研究：完整成果目录`project/{PARENT}/`、原协议/配置/代码/测试及直接输入均随包提供。它解释本轮为何改为直接学习持仓。旧账户主评价末端是2026-08-14开盘，本轮末端为当日收盘；成本账户/决策结构也不同，不能直接拼接两轮收益。
- R6：`project/docs/R6_REJECTION_DECISION.md`、研究/归因协议，以及模型选择和归因JSON。它已含连续版本；相关文件是历史背景，不是完整R6原始数据归档。本轮“连续”本身不能证明新机制。
- `project/RESEARCH_STATUS.md`保留逐次历史状态，顶部本轮条目较新。V5授权JSON主要记载上一轮有限授权；本轮授权来自本轮用户完整原文和冻结协议。不要把历史V3/V4/V5说明与这轮混合成一个假想状态。
- 官方价格校验凭证中引用的H00300等文件只作为底座交叉验证来源被带入，未作为本轮交易资产或新增评分输入。

本轮未拟合全历史未来模型，未创建Paper/Shadow或自动化，未产生券商连接和实盘订单。实际用户持仓未知。GPT可以提出下一轮设计建议；收到建议后再决定独立实施范围，不回改原V1证据。
''')
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "pyarrow", "pytest", "matplotlib", "tzdata")}
    note("review_tools/requirements_observed.txt", "\n".join(f"{name}=={version}" for name, version in versions.items()))
    note("review_tools/pytest.ini", "[pytest]\naddopts =\n")
    note("review_tools/recompute_saved_metrics.py", READ_ONLY_REVIEWER)
    note("05_复核方法与环境.md", '''# 复核方法与环境

本轮原环境是Windows、Python 3.13；各库观察版本见`review_tools/requirements_observed.txt`。该文件记录版本，不保证审阅端可安装完全相同环境；直接阅读CSV/JSON/Markdown不需要这些库。

`review_tools/recompute_saved_metrics.py`仅用Python标准库，可以从解压目录或直接从ZIP读取八条已保存CSV账本，复算累计收益、CAGR、算术年化、波动、夏普、回撤和总摩擦，并与result.json核对。它不训练、不重新执行策略、不下载数据、不写入原始证据。

复核脚本所在路径：`review_tools/recompute_saved_metrics.py`

运行方式：在解压目录运行`python review_tools/recompute_saved_metrics.py`，或为脚本传入`--zip 压缩包绝对路径`。

原合成测试入口位于`project/tests/`。需复核测试时，在project目录中使用Python的pytest模块，指定`-c ../review_tools/pytest.ini`和两个测试文件：`tests/test_510300_conditional_score_policy_v1.py`、`tests/test_intraday_overnight_increment_v1.py`。合成测试不读取本轮真实市场收益。

`project/scripts/run_510300_conditional_score_policy_v1.py`是原正式一次性研究入口，冻结和一次运行声明已经存在。不要删除声明、清空结果或重跑优化器来“复现”或寻找更好成绩。代码审阅、现有账本复算和合成测试足以先检查大部分重要问题；新的研究实验需另立明确协议。

本次打包只核对已声明文件、CRC、重复成员、索引、字节数和哈希。不执行额外安全性审计，不重新训练，不做全流程解压重放。已有35项测试和研究运行证据保留；本次新增的标准库账本统计复算另有记录。
''')
    exclusions = {"scope": "本轮成果完整目录、直接依赖及来源，加上明确列出的D20和R6背景",
                  "excluded": [".env及凭据输入", "虚拟环境和缓存", "其他旧ZIP和重复备份树", "未被本轮依赖的其他研究原始数据", "操作系统或应用目录"],
                  "checks_performed": ["CRC", "duplicate_members", "file_index_membership", "size_and_sha256", "declared_scope_coverage"],
                  "security_audit": False, "privacy_scan": False, "secret_scan": False, "malware_scan": False,
                  "redaction": False, "fresh_extraction_full_replay": False, "new_training": False, "new_policy_evaluation": False}
    add("06_交付范围与检查.json", encoded(exclusions), "交付范围和实际检查项目", generated_file=True)
    upload = "请解压附件，先阅读00_README_FIRST.md和01_GPT_REVIEW_PROMPT.md，再核对本轮用户方案、冻结协议、代码、全部开发结果与连续账户。请审查结论和实现是否成立，并重点解释为何完整模型最终仅约14.5%平均仓位、7个成交日、修复项贡献很小。在现有510300/现金范围和净夏普1.2目标下，给出最多一个优先研究方向、一个备选方向及下一轮可直接交给Codex实施的独立任务书，写清机制、数据、验证预算和停止条件。不能只复述失败，也不要通过回改原参数或选择有利年份宣称原V1通过；如果没有值得继续的方向，请明确说明。"
    note("07_上传GPT时粘贴这段.txt", upload)
    metadata = {"package_id": STEM, "created_at": timestamp, "study_id": STUDY,
                "state": result["state"], "current_study_files": len(current_files),
                "parent_context_files": len(parent_files), "project_source_files": source_file_count,
                "observed_runtime": {"python": platform.python_version(), **versions},
                "main_result_sha256": sha(payload[f"project/{REPORT}/result.json"]),
                "declared_current_tree_coverage": "COMPLETE", "frozen_identity_checks": checks,
                "external_gpt_review_received": False, "next_research_executed": False}
    add("08_PACKAGE_METADATA.json", encoded(metadata), "构建来源、范围及冻结一致性记录", generated_file=True)

    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "role", "bytes", "sha256"])
    writer.writeheader()
    for name in sorted(payload):
        writer.writerow({"path": name, "role": "；".join(roles[name]), "bytes": len(payload[name]), "sha256": sha(payload[name])})
    add("FILE_INDEX.csv", index.getvalue().encode("utf-8-sig"), "全成员索引；索引自身不递归自哈希", generated_file=True)
    guide_directory = DESTINATION / STEM
    guide_directory.mkdir(exist_ok=False)
    for name in sorted(generated):
        path = guide_directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload[name])

    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(payload):
            archive.writestr(name, payload[name])
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)), "ZIP成员重复")
        require(archive.testzip() is None, "ZIP的CRC检查失败")
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(members) == {row["path"] for row in rows} | {"FILE_INDEX.csv"}, "索引与ZIP成员集合不一致")
        for row in rows:
            content = archive.read(row["path"])
            require(len(content) == int(row["bytes"]) and sha(content) == row["sha256"], f"包内文件大小或哈希不符：{row['path']}")
        included_current = [name for name in members if name.startswith(f"project/{REPORT}/")]
        require(len(included_current) == len(current_files), "本轮成果目录收录不完整")
    content = archive_path.read_bytes()
    digest = sha(content)
    receipt = {"created_at": timestamp, "status": "PASS_SINGLE_ZIP_CRC_INDEX_HASH_AND_SCOPE",
               "zip_path": str(archive_path), "bytes": len(content), "MiB": len(content) / 2**20,
               "members": len(payload), "current_study_files": len(current_files),
               "parent_context_files": len(parent_files), "sha256": digest,
               "duplicate_members": 0, "crc": "PASS", "index_size_hash": "PASS", "current_study_scope": "COMPLETE",
               "frozen_identity_checks": len(checks), "full_research_replayed": False,
               "external_review_received": False, "next_research_executed": False,
               "upload_prompt_path": str(guide_directory / "07_上传GPT时粘贴这段.txt")}
    (DESTINATION / f"{STEM}.sha256.txt").write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    (DESTINATION / f"{STEM}_receipt.json").write_bytes(encoded(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
