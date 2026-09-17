"""从已保存的IF持仓失败实验构建中文审阅包，不重跑研究。"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.intraday_overnight_increment_v1 import digest, now, require, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_if_open_interest_increment_v1"
ZIP = ROOT / "deliverables/510300_IF持仓信息增量_V1_GPT审阅_20260913.zip"


def write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def create_report(result: dict, comparison: pd.DataFrame, verification: dict) -> str:
    evaluation = result["evaluation"]
    rows = comparison.set_index("group")
    table = ["| 进入时期 | 成熟原点 | M0均方误差 | M1均方误差 | M1相对改善 |",
             "|---|---:|---:|---:|---:|"]
    for group in ["2020-2021", "2022-2023", "2024-2026", "ALL"]:
        row = rows.loc[group]
        label = "全评价期" if group == "ALL" else group
        table.append(f"| {label} | {int(row.paired_origins):,} | {row.M0_mse:.10f} | {row.M1_mse:.10f} | {row.relative_mse_improvement * 100:+.4f}% |")
    lower, upper = evaluation["paired_mse_improvement_95pct_ci"]
    return f"""# 510300 IF原始持仓信息增量：本轮结论与证据

结论：已完成一次冻结的配对历史实验，尚未确认持仓信息的预测增量。M1整体均方误差比M0高约{abs(rows.loc['ALL', 'relative_mse_improvement']) * 100:.4f}%，预先规定的三个时期只有{evaluation['positive_eras']}个改善，主门未通过。账户阶段没有运行。

终态：`{result['status']}`。用户的成本后复合年化10%、净夏普1.2目标仍未由本轮实现。此终态只适用于本轮变量、模型、五日期限及信息延迟，不能外推为所有期货信息必然无效。

## 类似研究是否已经做过

确实做过近邻，不能说原始持仓从未研究：

| 既有工作 | 与本轮的关系 | 原结果保留 |
|---|---|---|
| Round2 IF主连相对领先 | 主连存在换月跳变，当时未把原始持仓变化正式登记 | 环境探索，不能把旧领先改名 |
| 期货—期权—现货桥梁V1 | IF0持仓与价格、期权等7项输入一起使用，20日方向用途 | 开发期拒绝，非本轮消融 |
| IF强制资金流V1 | 已使用真实合约聚合持仓、基差压力和状态转移 | 每方向10个事件，事件门不足，收益有效性未检验 |
| 衍生品压力释放V2聚合OI | 已汇总真实IF持仓，同时依赖期权压力与盘中确认 | 压力分位无法构造，NO_VIEW，资源族关闭 |

本轮仅补定向检索未找到的严格M0/M1增量比较：相同模型、相同日期、相同训练，仅增加持仓变化与交互。V6续行授权允许有限新方法，用户本轮又明确提供了这一方案。旧研究代码、入口和终态没有重开。上述判断不证明全部项目历史记录绝无遗漏。

## 实际做了什么

1. 数据预检后、读取真实五日标签之前，冻结了配置、实现、原始建议、去重依据、测试与直接输入。冻结时间：{result['freeze_created_at']}。
2. 复用中金所197份月度原始压缩包、199个真实IF合约、15,860条日记录，以及ETF修正未复权价格、现货指数、独立日历和14次官方分红记录。IF数据截至2026-08-12；ETF标签截至2026-08-14。
3. 2016-02-01起扩展训练，2020-01-02起评价进入。进行了{result['refit_months']}次配对月首更新，共{result['model_fits']}个模型拟合，取得{evaluation['paired_origins']:,}个完整配对成熟预测，另有{evaluation['censored_origins']}个未成熟原点保留删失状态。
4. 主检验使用20交易日、2,000次成对循环区块重抽样。没有网格搜索、最终全历史模型、数据源切换或失败后反向使用。

M0使用ETF一日/五日总收益、20日方差、同一真实IF合约的相对五日价格收益以及到期/换月控制。M1只增加总持仓一日对数变化和其与ETF五日方向的交互。

历史每日发布时间没有得到证明，因此IF信息固定延迟一个交易日使用：源日s的数据假定至下一交易日t收盘可用，t+1开盘进入，t+6开盘退出。这个假设保留为限制，不是历史点时可得性的证明。入场前跳空另列，不进入五日持有收益。

## 主检验结果

相对改善=(M0均方误差−M1均方误差)/M0均方误差，正值才表示加入持仓信息更好。

{chr(10).join(table)}

M0−M1均方误差差值为{evaluation['overall']['mse_improvement']:.10g}，其95%成对区块区间为[{lower:.10g}, {upper:.10g}]，跨过零。完整覆盖门通过；区间下界为正和至少两个时期改善这两项门均未通过。

原点是逐日产生的五日预测，标签重叠；1,599条不是1,599次独立交易。区块推断仅处理本次时间相关性，没有消除项目此前多轮选题、模型和历史观察的选择偏差。

两组预测为正的原点比例均约{rows.loc['ALL', 'M0_positive_prediction_fraction'] * 100:.2f}%；这些原点的独立五日净标签均值，M0约{rows.loc['ALL', 'M0_positive_prediction_realized_mean'] * 100:.4f}%，M1约{rows.loc['ALL', 'M1_positive_prediction_realized_mean'] * 100:.4f}%。这只是重叠标签的描述，不是连续账户收益，更不能据此年化。

到期周和换月边界的分组结果均已保存，见`prediction_comparison.csv`；它们不能代替失败的整体主检验，也不能用于事后选择“只在有利日期交易”。

## 与10%／1.2目标的关系

| 项目 | 状态 |
|---|---|
| 持仓信息预测增量 | 本轮未确认 |
| 简单账户 | NOT_RUN_PRIMARY_INCREMENT_GATE_FAILED |
| 新策略净年化 | NOT_COMPUTED |
| 新策略净夏普 | NOT_COMPUTED |
| 独立未来验证 | NOT_ESTABLISHED |
| 最终目标 | 未实现 |

“未计算”不等于零收益，也不等于已经亏损。协议事先要求先有增量再进入账户，本轮按该顺序结案。代码中保留了通过主门后才能启用的简单账户实现，其风险和费用规则已事先写定；代码存在不代表该账户已运行或被验证。

## 完成检查与交付范围

冻结前10项合成数据测试通过，覆盖同合约、信息边界、分红、现金/T+1及账户财富。保存结果核对状态：`{verification['status']}`；160个模型的成熟时钟与保存预测、2,000个保存区块统计均核对通过，最大预测重算误差{verification['max_prediction_error']:.1f}。该核对没有新拟合、新账户、新随机样本或下载。

审阅包包含本轮完整输入快照、原始月档、原始分红凭据、方法和代码、模型与原点、失败状态和验证回执。仅做必要的数据合同、数值和包结构检查，没有另做安全审计，没有上传或声称外部GPT已经审阅。

## 下一步资源决定

本次日频总持仓增量用途关闭，不追加窗口、反向、第三个过滤器或更复杂模型。继续追求10%／1.2，需要另一项能在成交前到达、并与已试信息实质不同的证据；本包的审阅提示要求先给出候选差异、免费来源、时钟、有限预算、验收与停止条件，再提出下一项实验。

若未来考虑更早使用IF信息，必须先取得源日收市至下一开盘前的真实首次可得回执，并在观察结果前登记。它属于待设计的严格前向工作，本轮没有启动，不允许把现有历史时戳改成当年可得记录。
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    require(not ZIP.exists(), "目标ZIP已存在，禁止覆盖完成包")
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(result["status"] == "REJECTED_FROZEN_NO_INCREMENT_UNDER_REGISTERED_TEST", "本交付器只打包已完成的本轮终态")
    require(verification["new_fits"] == 0 and verification["new_accounts"] == 0, "保存核对引入了新研究活动")
    comparison = pd.read_csv(OUT / "prediction_comparison.csv")
    report = create_report(result, comparison, verification)
    write_text_exclusive(OUT / "研究结论与下一步.md", report)
    prompt = """# 可直接复制给GPT的审阅提示

请以严格的量化研究评审身份审阅本包。最终目标仅操作510300和人民币现金，成本后复合年化至少10%、净夏普至少1.2。

先读00_README_FIRST.md和02_研究结论与下一步.md，再按证据导航读取原始用户建议、去重记录、冻结配置/协议、代码、预测与模型、区块结果和验证回执。

本轮是IF真实合约总持仓增量的单次历史诊断。近邻已经用过聚合持仓；本次是否足够不同，必须从M0/M1消融问题判断，不能把“换名”当创新。既有强制资金流因事件门停止、衍生品压力V2因数据构造停止；不要说它们已证明持仓信息没有预测能力。

请依次给出：

1. 结论是否受证据支持；指出P0/P1/P2问题和具体文件/字段。重点核查共同信息集、同合约价格、到期移仓控制、假设的一日数据延迟、五日标签成熟清除、成本与分红权益。
2. MSE主门、固定0.1岭惩罚、逐日重叠标签、20日区块和三个时期划分是否合理；区分“未确认增量”“等效性已经证明”“所有持仓用途都无效”。不能依据结果推荐反向、窗口搜索、改时期或过滤器救援。
3. 本轮先预测门后账户的停止是否执行正确。主门失败，账户、年化和夏普未计算；不要从独立标签均值拼出账户指标，也不要把没有账户写成亏损。
4. 判断是否应关闭这一用途、保留何种有限前向问题。若提出更快的数据时钟，须说明如何获得真实首次可得回执，不能假装历史档案证明当时可用。
5. 对下一阶段只选择一个最高优先事项，并给出它与既有失败近邻的实质差异、免费直接来源、最小数据合同、有限研究预算、预先验证步骤和明确停止条件。再列最多两个低优先备选与明确不做事项；不要直接堆模型或参数。
6. 分别说明离年化10%和夏普1.2还缺什么证据。已被研究过的历史不是独立验证，任何历史点值达标也不能保证未来稳定表现。

本包只做本地数据/数值与ZIP结构检查，并非外部审阅结论。无券商、订单、期货或期权交易授权。请用中文输出可执行的质量改进和下一步资源决定。
"""
    write_text_exclusive(OUT / "GPT审阅提示.md", prompt)
    files: dict[str, Path] = {}

    def add(path: Path) -> None:
        require(path.is_file(), "交付缺少文件：" + str(path))
        name = path.relative_to(ROOT).as_posix()
        require(".." not in Path(name).parts, "压缩包成员路径非法")
        files[name] = path

    for path in OUT.rglob("*"):
        if path.is_file():
            add(path)
    freeze = json.loads((OUT / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in freeze["protected_files"] + freeze["input_snapshots"]:
        path = ROOT / item["path"]
        require(digest(path) == item["sha256"], "冻结文件漂移：" + item["path"])
        add(path)
    dedup = json.loads((OUT / "deduplication.json").read_text(encoding="utf-8"))
    for item in dedup["evidence"]:
        path = ROOT / item["path"]
        require(digest(path) == item["sha256"], "近邻证据漂移：" + item["path"])
        add(path)
    for rel in ["scripts/build_if_open_interest_increment_review_20260913.py",
                "scripts/download_cffex_if_true_term_structure_inputs_v1_0_1.py",
                "config/510300_if_true_term_structure_binary_screen_v1_0_1_candidates.yaml",
                "research/if_forced_flow_state_v1.py", "research/derivative_pressure_release_v2_aggregate_oi.py",
                "research/futures_option_bridge_v1.py", "research/direction_switch_common_v1.py", "backtest/engine.py",
                "reports/research/510300_IF_FORCED_FLOW_STATE_V1.md",
                "reports/data_quality/510300_if_forced_flow_state_v1.json",
                "reports/discovery/510300_futures_option_bridge_v1_dev.md",
                "docs/510300_DERIVATIVE_PRESSURE_RELEASE_V2_AGGREGATE_OI_SPEC.md",
                "docs/510300_SHARPE_1_2_RESEARCH_RESUMPTION_20260906.md",
                "data/raw/r6/510300_daily.parquet",
                "data/reference/510300_downside_risk_price_corrections_v1.csv"]:
        add(ROOT / rel)
    cfg = json.loads((ROOT / "config/510300_if_open_interest_increment_v1.json").read_text(encoding="utf-8"))
    for rel in cfg["inputs"].values():
        # 原路径复制也纳入，便于在解压目录阅读获取脚本和旧回执。
        snap = OUT / "frozen_inputs" / rel
        files[rel] = snap
    receipt = json.loads((OUT / "frozen_inputs" / cfg["inputs"]["if_receipt"]).read_text(encoding="utf-8"))
    raw = ROOT / "data/raw/futures/cffex_if_contract_history_v1_0_1"
    for month, sha in receipt["source"]["archive_sha256"].items():
        path = raw / "monthly_archives" / (month + ".zip")
        require(digest(path) == sha, "原始IF月档与获取回执不符：" + month)
        add(path)
    metadata = raw / "active_contract_metadata_20260812.xml"
    require(digest(metadata) == receipt["source"]["active_metadata_sha256"], "活动合约元数据不符")
    add(metadata)
    coverage = json.loads((OUT / "frozen_inputs" / cfg["inputs"]["dividend_coverage"]).read_text(encoding="utf-8"))

    def add_saved_sources(value) -> None:
        if isinstance(value, dict):
            if value.get("saved_file"):
                path = ROOT / value["saved_file"]
                if value.get("sha256"):
                    require(digest(path) == value["sha256"], "分红来源凭据不匹配")
                add(path)
            for nested in value.values():
                add_saved_sources(nested)
        elif isinstance(value, list):
            for nested in value:
                add_saved_sources(nested)

    add_saved_sources(coverage)
    requirements = "\n".join(name + "==" + importlib.metadata.version(name) for name in ["numpy", "pandas", "pyarrow", "pytest", "tzdata"]) + "\n"
    readme = """# 先读这里：IF持仓信息增量 V1

本轮完成，主预测门失败，账户未运行。整体MSE较基线略差，三个时期仅一个改善，95%区间跨零。不能据此声称年化10%／夏普1.2已实现，也不能说所有期货持仓研究都无效。

## 建议阅读顺序

1. `02_研究结论与下一步.md`：结论、数值、去重和资源决定。
2. `01_GPT_REVIEW_PROMPT.md`：可复制的质量审阅与下一步提示。
3. `reports/research/510300_if_open_interest_increment_v1/USER_REQUEST.md`和`USER_PROPOSAL.txt`：原始需求。
4. `docs/510300_IF_OPEN_INTEREST_INCREMENT_V1_PROTOCOL.md`和`config/510300_if_open_interest_increment_v1.json`：运行前冻结定义。
5. 本轮目录的`deduplication.json`、`data_preflight.json`、`freeze_manifest.json`：近邻、数据和冻结顺序。
6. `prediction_comparison.csv`、`predictions.parquet`、`samples.parquet`、`models.json`、`bootstrap_draws.parquet`与`bootstrap_indices.npz`：全部保存结果。
7. `result.json`、`execution_receipt.json`、`saved_verification_receipt.json`、`pre_freeze_tests.json`：终态与检查。

## 包含与排除

包含本轮完整输出、全部直接输入快照、197份中金所原始月档和活动合约XML、14次分红的官方PDF及覆盖来源、原价格与三处更正记录、入口/核心代码/合成测试、近邻的代码配置和结果。完整账户不存在，因此没有账户文件。

排除整个项目的其他研究、旧候选的庞大原始数据和账户、Python虚拟环境、缓存、机器配置及与此实验无关的附件。本包是本课题自包含审阅快照，不是项目磁盘镜像。旧近邻资料作为背景，未重跑其研究；其中旧仓位语义不构成本轮授权。

根`FILE_INDEX.csv`覆盖除索引自身外的全部成员，包含字节数与SHA-256。外部同名交付回执记录ZIP最终哈希和结构检查。结构通过不是科学结论、独立前向验证或外部GPT审阅。

## Windows只读核对

先在解压目录准备Python 3.13环境与`REQUIREMENTS_REVIEW.txt`中的包。下列命令只重算保存预测与保存抽样统计，不拟合、不生成账户、不下载。

📁 解压目录/scripts/run_510300_if_open_interest_increment_v1.py
```powershell
python scripts/run_510300_if_open_interest_increment_v1.py verify
```

📁 解压目录/tests/test_if_open_interest_increment_v1.py
```powershell
python -m pytest tests/test_if_open_interest_increment_v1.py -q
```

历史实验已经使用一次运行机会。`run_claim.json`保留，正常入口会阻止第二次run。不要删除它或改参数来重跑。本包的下载脚本只是来源构造说明，不应调用其下载入口来改写已冻结输入。
"""
    virtual = {"00_README_FIRST.md": readme.encode("utf-8"), "01_GPT_REVIEW_PROMPT.md": prompt.encode("utf-8"),
               "02_研究结论与下一步.md": report.encode("utf-8"), "REQUIREMENTS_REVIEW.txt": requirements.encode("utf-8")}
    virtual["DELIVERY_SCOPE.json"] = (json.dumps({"created_at": now(), "study_id": cfg["study_id"],
                                                "outcome": result["status"], "raw_cffex_month_archives": 197,
                                                "accounts": "NOT_COMPUTED", "external_review": "NOT_PERFORMED",
                                                "security_audit": False, "upload_performed": False}, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    entries = []
    for name, path in sorted(files.items()):
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": digest(path)})
    for name, payload in sorted(virtual.items()):
        entries.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    index_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(index_buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(sorted(entries, key=lambda item: item["path"]))
    virtual["FILE_INDEX.csv"] = index_buffer.getvalue().encode("utf-8-sig")
    temporary = ZIP.with_suffix(".building.zip")
    require(not temporary.exists(), "存在上次未完成构建，禁止静默覆盖")
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        for name, payload in sorted(virtual.items()):
            archive.writestr(name, payload)
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "ZIP成员重复")
        require(archive.testzip() is None, "ZIP的CRC检查失败")
        index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        require(set(names) - {"FILE_INDEX.csv"} == {item["path"] for item in index}, "索引覆盖不同")
        for item in index:
            payload = archive.read(item["path"])
            require(len(payload) == int(item["bytes"]) and hashlib.sha256(payload).hexdigest() == item["sha256"], "索引内容不匹配")
    os.replace(temporary, ZIP)
    delivery = {"status": "PASS_ZIP_CRC_INDEX_HASH_AND_DECLARED_COVERAGE", "created_at": now(),
                "zip": ZIP.relative_to(ROOT).as_posix(), "bytes": ZIP.stat().st_size, "sha256": digest(ZIP),
                "members": len(names), "indexed_files": len(index), "cffex_raw_archives": 197,
                "tests_passed": 10, "new_research_runs_by_builder": 0, "new_models_by_builder": 0,
                "security_audit": False, "external_review": "NOT_PERFORMED", "upload_performed": False}
    receipt_path = ZIP.with_suffix(".delivery.json")
    write_json(receipt_path, delivery, exclusive=True)
    print(json.dumps(delivery, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
