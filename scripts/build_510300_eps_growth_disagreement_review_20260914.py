"""生成本轮增长分歧失败结论的自包含中文审阅包。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_growth_disagreement_increment_v1 import OUT, CONFIG, frozen_paths, identity, read, save, now
from research.eps_growth_disagreement_source_feasibility_v1 import OUT as SOURCE_OUT, CONFIG as SOURCE_CONFIG, sources

VERIFY = ROOT / "reports/research/510300_eps_growth_disagreement_delivery_verification_20260914"
ZIP = ROOT / "deliverables/510300_EPS增长分歧增量_V1_GPT审阅_20260914.zip"
PREVIOUS_ZIP = ROOT / "deliverables/510300_EPS明示修正去重与增长分歧来源_V1_GPT审阅_20260914.zip"


def once(path, content):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def prepare():
    result = read(OUT / "result.json")
    source = read(OUT / "source_validation.json")
    evidence = identity(PREVIOUS_ZIP)
    if evidence["sha256"] != "6dcc45a3cb2078355a6863aabbc2fdbd872fb52a386c585d3b3982f39e817d7e" or evidence["size_bytes"] != 7349792:
        raise ValueError("上一轮实际交付ZIP身份改变")
    save(OUT / "prior_archive_identity.json", {"status": "PASS_PRIOR_DELIVERED_ARCHIVE_IDENTITY", "verified_at": now(), "archive": evidence})
    overall = result["evaluation"]["overall"]
    lo, hi = result["evaluation"]["paired_mse_improvement_95_interval"]
    report = f"""# 增长分歧未改善510300未来可成交收益预测，本用途冻结

只操作510300与人民币现金、完整账户成本后复合年化至少10%且净夏普至少1.2的共同目标仍未达成。本轮确实完成一个新变量的预测检验：72次模型拟合、36个共同预测原点，其中33个标签已成熟。新增增长分歧后，均方误差增加{-overall['relative_mse_improvement']:.2%}，没有通过事前固定的增量门。状态为REJECTED_FROZEN_NO_RELIABLE_GROWTH_DISAGREEMENT_INCREMENT，账户阶段NOT_RUN_PREDICTION_GATE，新增账户0；年化与夏普未计算，不能写成0或声称达标。

## 这次真正多使用了什么信息

M0用既有EPS增长、同年度利润修正、报告前瞻盈利收益率三项月度中位数，并控制同一配对公司的机构平均增长、配对公司占比、报告年龄差和平均年龄，共七项。M1完整保留这七项，只增加同公司两机构年度EPS对称增长差的绝对值之公司中位数。每公司各机构内的增长是2×(下一年度EPS−当年EPS)/(两者绝对值之和)，不是普通百分比增长或精确NTM。

定向查重未在已检查EPS实现中发现本次公司配对分歧的已完成同构实验。旧规划提出过分歧，其他分歧实现还包括估值情景宽度和模型版本差；这些不能冒充本次输入。该查重结论有范围，不宣称穷尽全项目所有可能别名。

两模型采用相同来源原点、同样的扩展训练、训练专属标准化、至少12个成熟月和岭回归alpha=10。alpha对应残差平方和加10倍系数平方和。目标为原点下一交易日开盘到其后第60个交易日开盘的含权益分红总收益；训练时只允许退出开盘已经发生的旧标签。固定行情截点为2026-08-14，没有因首个预测到2023年才出现而后移原评价起点2020-01-02。

## 来源覆盖、定义和时间

139个历史来源月中，50个月满足旧公共来源条件与至少30家配对公司；实际每月32至81家，中位数57.5家。这些合格月份包含2763个公司月、5526条机构公司月增长记录，涉及1906份不同报告。逐条从保存原始事实重新计算同报告两年度EPS增长，最大误差0；核对公司、机构、年度、同表EPS行与单位，并以目录、平台和正文日期的最晚值限制在原点之前、年龄不超过180日。

这里仍有明确限制：5526条记录中21条的原文EPS未明确注明基本或摊薄定义；其余报告也没有证明跨机构股本基础完全统一。因此只称“各报告口径的年度增长差异”，不称完全同口径的纯盈利不确定性。21条未因结果难解释而删去。

按事前固定的首条、末条、分歧最大公司月各取两机构报告，共6份完整原PDF。六份第1页均已渲染并目视核对公司、日期、年度、EPS行与负数。昆仑万维跨盈亏、东吴括号负数保留；中微国信正文2026-05-05、平台2026-05-06，昆仑东吴正文2025-05-01、平台2025-05-02，都取较晚可用日期。没有宣称1906份PDF均已重新目视核实。

## 预测结果

| 固定处理阶段 | 数量或状态 |
|---|---:|
| 历史来源月份 | 139 |
| 具备全部特征的来源月份 | 50 |
| 截点内已经成熟的合格来源收益标签 | 47 |
| 固定评价区间全部月末记录 | 79 |
| 因来源不足保持NO_VIEW | 29 |
| 因成熟训练月份不足保持NO_VIEW | 14 |
| 同时生成M0、M1预测 | 36 |
| 其中标签尚未成熟，不进入误差评价 | 3 |
| 两模型共同成熟评价原点 | 33 |

47个标签是既有60日定义的本轮重算，不是47个未经观察的新市场样本。33个评价原点从2023-07-31到2026-04-30，60日结果重叠。区块重采样采用连续34个日历月，其中1个缺失月保留；固定3个月区块、2000次和种子20260914，不把缺失前后月份直接拼成相邻。

| 主要比较 | M0：已有信息与控制 | M1：另加增长分歧 |
|---|---:|---:|
| 均方误差 | {overall['M0_mse']:.9f} | {overall['M1_mse']:.9f} |
| 平均绝对误差 | {overall['M0_mae']:.6f} | {overall['M1_mae']:.6f} |
| 方向判断正确 | 16/33（48.48%） | 16/33（48.48%） |

M0均方误差减M1均方误差为{overall['mse_improvement']:.9f}，95%区间[{lo:.9f}, {hi:.9f}]，改善区间下界没有大于0。2024年12个原点的改善为−0.000898439，2025年12个原点为+0.000009013；两个完整年同时改善的条件也未通过。

两个模型在33个成熟原点的预测正负方向完全相同。实际上涨18次、下跌15次；两模型都正确识别14次上涨、2次下跌，另有13次实际下跌被预测为上涨。这是用户关心的涨跌识别诊断，但不以方向准确率替代主要损失。实际下跌组的均方误差增加约12.79%；该事后分组也不能用于倒推一套新账户规则。

## 账户、独立性与停止

预测门要求至少24个共同成熟原点、误差改善95%区间下界大于0，且2024、2025各至少8个原点并各自改善。本轮只通过数量条件。依照用户附件提出的顺序，本用途到此冻结，不启动账户、不改方向、不换聚合器、窗口或正则化救援。历史训练与账户已有授权；本次账户未运行源于固定有限试验设计，并非缺少额外用户许可。

预先写好的账户条件仍原样保留：初始20万元、2020-01-02至2026-08-14完整区间、只买卖510300与人民币现金、现金利息0、年化242日、T+1、100股整手、0.001价格档、最低5元佣金、基础与压力费用、分红资格与到账、五档预算。账户收益、夏普和目标联合通过值均为空。缺少预测表示NO_VIEW、保留已有份额，不是把缺失解释成空仓信号。

本轮历史行情已被多项研究观察；顺序训练避免直接偷看尚未成熟标签，但不消除研究者多次试验的选择偏差，不是独立新样本验证。拒绝的是这个固定分歧变量在这个实验中的用途，不外推成“所有盈利信息永远无效”。

## 下一轮优先级

1. 停止从这项EPS分歧变量继续加工规则；保留本轮与旧EPS失败记录。
2. 转向不同信息机制，先查重与核实免费历史来源的可用时钟。候选仍须回答“下一开盘成交之后是否多知道了什么”，不能只解释已经发生的跳空。优先不恢复已暂停的大规模研报补源队列。
3. 只在输入与用途确有区别、来源足以支持可比检验时，冻结一个小型M0/M1试验；不依靠换名重做已拒绝的IF持仓、AH溢价或旧宏观用途。来源有缺口就保留NO_VIEW；预测没有增量就关闭该用途。历史点值同时达到10%与1.2也还需独立验证，不能降低目标。

## 审阅包范围和复算

本包包含1906份报告的保存事实JSON及采集元数据、6份固定样本完整PDF和对应HTML/目录原响应、6张已检查原页图，含83400行旧机构公司月输入、41700行配对结果、139行来源统计、本轮全部标签和预测、72个模型的训练收据、2000组区块结果及原始抽样索引，含ETF未复权价格、分红、交易日历与来源回执、协议、配置、代码、测试和所有未运行状态。

自包含范围为从包内冻结事实和机构月度输入复算本轮来源配对、时钟、标签、保存模型预测与统计。包内不是1906份全量PDF原件，也不重建上游完整券商事实库。仅样本原PDF可由审阅者直接翻页检查；其余保存事实的系统性提取错误仍是科学限制。全部哈希通过不等于这些经济定义无误。

合成测试4项通过，覆盖岭回归定义、未成熟/未来数据隔离、含分红开盘标签及连续日历区块。离线复核重算来源配对，核对79个全部月末及缺失状态；使用保存系数核验正规方程和预测，使用保存区块索引重算区间，0新增拟合、0新增随机抽样、0新增账户。结构核对与数值复算不等于收到外部GPT评审，也不构成目标已达成。
"""
    once(OUT / "研究结论与下一步.md", report)
    previous = ROOT / "reports/research/510300_eps_explicit_revision_novelty_diagnostic_v1"
    once(OUT / "原始用户附件.txt", (previous / "原始用户附件.txt").read_text(encoding="utf-8"))
    request = "用户目标：只操作510300与人民币现金，完整账户成本后年化至少10%、净夏普至少1.2。类似研究没有做过可以做；用户允许加入其他免费有效信息。\n本轮范围：一个报告增长分歧变量，相同原点M0/M1预测增量检验；预测门未通过则账户NOT_RUN，不调参救援。本轮不改变旧研究与实际持仓。\n"
    once(OUT / "用户请求与本轮范围.txt", request)
    once(OUT / "GPT审阅提示.md", """请围绕只操作510300和人民币现金、完整账户成本后年化10%与净夏普1.2的共同目标，独立批评本轮增长分歧试验，并给出下一研究方向与优先级。

先读研究结论、配置和冻结协议，再查原件样本、保存事实、来源配对、全部月末预测和模型训练收据。核心结论：33个共同成熟原点，M1均方误差比M0增加5.97%，两个模型方向判断相同、准确率均16/33，预测门失败，账户NOT_RUN。不要把未运行账户当0收益，也不要把历史顺序预测当独立新样本。

请检查M0七项是否充分控制均值、覆盖和报告年龄；M1是否只多一变量；EPS对称增长在跨盈亏、股本与报告异步下如何解释；21条未明示基本/摊薄定义的记录应如何限制结论；50个合格来源月最终只有33个成熟评价月、标签重叠和3个月区块的不确定性是否充分；连续34个月中的缺失是否保留；未来标签是否被隔离；固定惩罚与训练专属标准化是否正确。原点正负方向相同不能证明所有预测都相同，主要误差仍变差。

逐项区分已核实、计算错误、未建立与超出本包范围的事项。1906份报告的事实和采集元数据都在包内，但完整原PDF只有6份固定样本，不能把成员哈希当成全量EPS提取经济含义正确的证明。可从冻结输入复算来源配对、标签、保存模型和固定重采样，不需新拟合。

请提出下一轮最优先的一项不同信息机制，并说明与已拒绝用途如何区分、免费直接来源与可用时钟、最小公平对照、验证次序和停止条件。不要用更换方向、窗口、聚合器或费用来救本轮失败；若仍建议某种盈利信息，需明确新增观测究竟是什么。未来账户必须覆盖完整现金期、分红、T+1、整手和压力费用，且10%与1.2共同通过。本包未上传、未收到外部评审，不授权订单或实盘。
""")
    save(OUT / "goal_followup_and_next_route.json", {"recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "current_goal_turn_classification": "PROGRESS", "same_external_blocker_consecutive_turns": 0,
        "closed_local_hypothesis": result["status"], "next_route": "DIFFERENT_INFORMATION_MECHANISM_AFTER_EQUIVALENT_USE_AND_FREE_SOURCE_CLOCK_CHECK",
        "new_model_fits": 72, "new_accounts": 0, "prediction_increment_gate_pass": False,
        "goal_achieved": False, "old_frozen_results_preserved": True, "paused_source_queues_resumed": False})
    once(OUT / "requirements-review.txt", "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in
        ["numpy", "pandas", "pyarrow", "scikit-learn", "joblib", "threadpoolctl", "tzdata", "pytest", "matplotlib"]) + "\n")
    save(OUT / "delivery_scope.json", {"prepared_at": now(), "unique_reports_with_fact_and_metadata": source["unique_reports"],
        "complete_original_pdf_samples": len(source["raw_pdf_samples"]), "selected_pages_visually_checked": 6,
        "all_1906_original_pdfs_included": False, "all_upstream_fact_extraction_rebuilt": False,
        "complete_current_study_frozen_input_closure": True, "external_review_received": False})
    print("中文报告、审阅提示、用户目标、范围和下一步已保存。", flush=True)


def package():
    if ZIP.exists():
        raise FileExistsError("交付ZIP已经存在")
    receipt_dir = VERIFY / "workspace_packet_checks"
    if read(receipt_dir / "packet_verification.json")["status"] != "PASS_SOURCES_ALL_MONTHS_SAVED_MODELS_AND_FIXED_BOOTSTRAP":
        raise ValueError("工作区完整保存复核尚未通过")
    paths = set(frozen_paths(read(CONFIG)) + sources(read(SOURCE_CONFIG)))
    for folder in [OUT, SOURCE_OUT, receipt_dir]:
        paths.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.update([VERIFY / "workspace_verification.json", Path(__file__),
        ROOT / "scripts/verify_510300_eps_growth_disagreement_packet_20260914.py",
        ROOT / "scripts/register_510300_eps_growth_disagreement_20260914.py"])
    entries = [identity(p) for p in sorted(paths)]
    readme = """# 510300增长分歧增量试验：预测门失败，账户未运行

阅读顺序：reports/research/510300_eps_growth_disagreement_increment_v1/研究结论与下一步.md → GPT审阅提示.md → docs/510300_EPS_GROWTH_DISAGREEMENT_INCREMENT_V1_PROTOCOL.md → result.json、全部预测和训练收据。原始用户附件与本轮范围在同一结果目录。

一个新变量、72次拟合、36个共同预测、33个成熟评价原点。M1均方误差增加5.97%，没有改善涨跌识别，拒绝本用途。账户NOT_RUN_PREDICTION_GATE，年化和夏普未计算，总目标未达成。

包含1906份报告的事实及采集元数据，但完整原PDF仅6份固定样本；83400行旧机构输入与本轮全部来源配对、标签、保存模型、预测、固定抽样、ETF价格分红日历等均包含。能够从保存输入独立复算本轮，不重建上游全量原始研报提取。不要把事实文件哈希当成会计口径完全统一或预测有效的证明。

Windows离线复核：完整解压到新目录，用已经安装requirements-review.txt所列运行库的Python，在解压根目录PowerShell运行：python .\\scripts\\verify_510300_eps_growth_disagreement_packet_20260914.py --receipt-directory .\\本轮离线复核。该新回执目录必须不存在。脚本先核对FILE_INDEX，再重算来源配对、标签、全部月末状态、保存模型与固定区块结果；不联网、不重新拟合、不新增随机抽样、不运行账户。不要执行run模式或旧账户入口。

FILE_INDEX.csv列出除自身外全部成员的路径、大小和SHA-256；protocol_freeze.json另绑定研究开始前的输入、代码和协议。源检查、合成测试、未运行账户、局限与下一步都保留。本包尚未上传或获得外部GPT评审，复核通过不表示夏普1.2和年化10%目标实现。
""".encode("utf-8")
    entries.append({"path": "00_README_FIRST.md", "size_bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest()})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(sorted(entries, key=lambda e: e["path"]))
    temporary = ZIP.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("00_README_FIRST.md", readme)
        archive.writestr("FILE_INDEX.csv", buffer.getvalue().encode("utf-8-sig"))
    with zipfile.ZipFile(temporary) as archive:
        expected = {entry["path"]: entry for entry in entries}
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("ZIP CRC或成员重名检查失败")
        if set(archive.namelist()) != set(expected) | {"FILE_INDEX.csv"}:
            raise ValueError("ZIP成员范围与索引不一致")
        for relative, entry in expected.items():
            raw = archive.read(relative)
            if len(raw) != entry["size_bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError("ZIP成员大小或哈希不一致")
    if temporary.stat().st_size >= 512_000_000:
        raise ValueError("ZIP达到512MB限制，保留building文件")
    os.replace(temporary, ZIP)
    record = {"status": "PASS_REVIEW_ZIP_INDEX_CRC_AND_MEMBER_HASHES", "created_at": now(), "path": str(ZIP),
        "bytes": ZIP.stat().st_size, "sha256": identity(ZIP)["sha256"], "members": len(entries) + 1,
        "indexed_members": len(entries), "uncompressed_bytes_excluding_index": sum(e["size_bytes"] for e in entries),
        "unique_reports_with_fact_and_metadata": 1906, "complete_original_pdf_samples": 6,
        "new_model_fits_in_study": 72, "new_accounts_in_study": 0, "new_model_fits_in_packaging": 0,
        "external_review_received": False, "security_audit": False, "goal_achieved": False}
    save(ZIP.with_suffix(".delivery.json"), record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "package"], required=True)
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else package()
