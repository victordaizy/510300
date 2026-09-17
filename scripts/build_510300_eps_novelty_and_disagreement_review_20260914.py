"""生成明示修正去重与分歧来源检查的中文交付和审阅ZIP。"""
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
from research.eps_explicit_revision_novelty_diagnostic_v1 import OUT, CONFIG, source_paths, identity, read, save, now
from research.eps_growth_disagreement_source_feasibility_v1 import OUT as DISAGREEMENT, CONFIG as DISAGREEMENT_CONFIG, sources as disagreement_sources

VERIFY = ROOT / "reports/research/510300_eps_revision_novelty_delivery_verification_20260914"
ZIP = ROOT / "deliverables/510300_EPS明示修正去重与增长分歧来源_V1_GPT审阅_20260914.zip"
PREVIOUS_ZIP = ROOT / "deliverables/510300_EPS两对归母口径核实_V1_GPT审阅_20260914.zip"


def once(path, text):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def prepare():
    result = read(OUT / "result.json")
    disagreement = read(DISAGREEMENT / "result.json")
    if identity(PREVIOUS_ZIP)["sha256"] != "5b3cbbdff008027354725d4c6b37b990940249a48a71159908bd5ccaac26522a":
        raise ValueError("上一轮实际交付包身份改变")
    save(OUT / "prior_archive_identity.json", {"status": "PASS_PRIOR_DELIVERED_ARCHIVE_IDENTITY", "verified_at": now(), "archive": identity(PREVIOUS_ZIP)})
    report = f"""# 明示修正没有新增数字，增长分歧具备进一步检验所需的来源覆盖

只操作510300和现金、完整账户成本后年化至少10%且净夏普至少1.2的目标仍未达成。本轮完成两个有界步骤：查清两份研报明示前值与现值是否包含新的原始预测数字；按预先固定规则核对两机构增长分歧的覆盖和变化。没有新增收益标签、模型拟合或账户。

## 明示前值现值对应了哪些旧数字

候选集是已有国信归档选择表中，美的与兴业2024-08-01至2025-04-30的全部7份报告。完整PDF共94页，本轮重新提取用于核实的前两页合计14页；没有声称全部94页均目视审阅。此候选集也不是全球或未经选择的全部研报目录。

| 当前报告 | 叙述前值：2025/2026年，亿元整数 | 候选原表：2025/2026年，百万元 | 给定候选集中的对应报告 |
|---|---:|---:|---|
| 美的2025-03-31 | 424 / 466 | 42415 / 46635 | 2025-02-13，AP202502131643053017 |
| 兴业平台2025-03-31、正文2025-03-30 | 807 / 844 | 80687 / 84435 | 2024-08-26，AP202408261639463099 |

美的2024-11-04报告的2026年原值为46566百万元，2025-02-13为46635百万元，都可以舍入为466亿元。只匹配2026年会得到两个候选；联合2025和2026年，才排除11月报告。联合匹配证明给定候选集内数字可对应，不证明报告明文引用了该编号，也不证明全球唯一历史版本。2027年没有前值，保留NOT_COMPARABLE，不能补零。

美的从这组原表算出的2026年普通相对修正为+0.7248%，原对称修正为+0.007221605。旧系统2025-05-30已经保存同一报告对和同一对称修正。旧2025-04-30采用90日基准的2024-11-04报告，且前后利润标签不同，因而缺失。上一轮使用90日基准得到+0.8740%，与本轮不同是比较报告不同，不是改写或纠正上一轮算术。

兴业从原表算出的2026年普通相对修正为-4.9257%。旧2025-04-30已经选择相同前后报告，但受“净利润/归母净利润”标签不同影响，旧修正字段缺失。两份叙述均复述已有原始表格中的数字，本轮新增原始预测数字为0。

这只关闭“这两份文本提供新的原始预测数字”这一具体设想。事件表达、出现时间和旧月度因子不同，不能从来源去重直接推断更及时的表示一定无效；本轮没有检验这种交易方法，也不允许用它直接重开旧失败账户。

## 找到可以进入后续检验的来源候选

项目方向文档曾提出“预测分歧”。本轮定向检查EPS实现、配置和方向文档，已有实现使用公司内机构平均、覆盖、报告年龄和修正广度，未在这些已检查文件中找到本次固定的公司配对增长差异统计。官方业绩预告修正属于另一来源和事件定义，不是券商配对分歧。这里保留查重范围，不声称全项目任何同构实现都已穷尽。

在查看本次统计之前固定：同一公司、同一月末、同一目标财政年度，国信与东吴各自的年度EPS对称增长取差的绝对值，再按公司取中位数。两端日期必须严格早于原点、年龄不超过180日；至少30家公司配对，且满足原公共来源有效条件。只用一家机构时保持缺失，不能记为零分歧。

| 来源检查项 | 结果 |
|---|---:|
| 旧机构公司月度输入 | {disagreement['input_institution_company_month_rows']}行 |
| 公司月度宇宙 | {disagreement['company_month_universe']}行 |
| 可形成两机构配对的公司月度 | {disagreement['available_paired_company_months']}行 |
| 全部来源月份 | {disagreement['source_months']}个月 |
| 原公共来源有效月份 | {disagreement['old_common_valid_months']}个月 |
| 另满足至少30家公司配对的月份 | {disagreement['eligible_months']}个月 |
| 这些月份的配对公司数 | {disagreement['eligible_pair_count_min']}至{disagreement['eligible_pair_count_max']}家，中位数{disagreement['eligible_pair_count_median']}家 |
| 可用区间 | {disagreement['first_eligible_origin']}至{disagreement['last_eligible_origin']} |
| 月度分歧统计范围 | {disagreement['statistic_min']:.6f}至{disagreement['statistic_max']:.6f} |
| 不同统计值数目 | {disagreement['statistic_distinct_values']} |

固定来源门槛通过；这不是收益门槛通过。50个源月份不是50个独立收益样本，后续60日等成熟标签会减少可评价月份并存在重叠。可配对32至81家公司也不能被说成完整沪深300共识；两份预测的发布日可能不同，年龄与覆盖变化都可能混入分歧。

## 下一步与停止条件

优先核实“增长分歧”是否在其他命名下已经被检验，再固定一个增量变量的预测协议。比较基准应保留原盈利预测信息，并控制配对数量和报告年龄差异；不让较新观点的作用冒充机构分歧。训练只使用当时已成熟标签，使用相同来源可用原点比较基准和新增变量。预测比较的主要损失、区间、费用和完整账户门槛均须在读本次预测结果前固定。未通过预测门槛则不运行新账户，也不改取均值、分位数、方向或窗口救援。

若通过预测门槛，才检验只买卖510300与现金的完整账户：包含闲置现金、分红、T+1、整手和费用；共同满足净夏普1.2与复合年化10%才算达到目标。当前预测价值为NOT_COMPUTED，账户为NOT_RUN。

## 可复核范围

本包包含本轮7份原PDF、目录/HTML/事实、14页文本、两页新定位原图、原始用户附件、两套协议配置代码、22行旧公司记录、83400行保存机构输入、41700行配对结果和139行来源统计。它可独立复算本轮两个步骤；不重建上游全部券商原件事实库，不把保存输入的哈希当成全量原始来源已再次核实的证明。原始输入提取的系统性误差仍可能影响统计解释，后续协议需作针对性核对。

本轮唯一研究运行和保存核对未出现程序失败；早期检索有一次引用了不存在的文件名，随后通过文件列表定位了实际版本，没有据此判断研究不存在。来源缺失、候选边界和旧NO_VIEW全部保留。审阅ZIP、结构核对和数值复算不等于收到外部GPT评审或证明收益达标。
"""
    once(OUT / "研究结论与下一步.md", report)
    previous = ROOT / "reports/research/510300_eps_profit_attribution_pair_adjudication_v1"
    for filename in ["原始用户附件.txt", "用户请求与本轮范围.txt"]:
        original = (previous / filename).read_text(encoding="utf-8")
        if filename.startswith("用户请求"):
            original = original.split("本轮范围：")[0] + "本轮范围：两份券商叙述前后值的来源去重；两机构年度增长分歧的覆盖和变化检查。0新模型、0新账户。\n"
        once(OUT / filename, original)
    once(OUT / "GPT审阅提示.md", """请围绕只操作510300、完整账户成本后年化10%和净夏普1.2目标，独立批评本轮证据和下一步方向。先看研究结论、两套协议，再核对原件和保存输入。

重点检查：美的是否需要两个年度联合匹配；舍入区间与候选范围是否被过度解释；新原始数字为0能否被误读成所有事件表示都无效；旧4月缺失与5月已有记录的关系；新分歧统计是否只是年龄差或覆盖变化；139个月中只有50个月可用、仅32至81家公司配对是否导致选择偏差；原EPS年度与股本口径、报告异步和标签成熟如何验证。保存输入可以独立复算本轮统计，但本包不重建所有上游研报事实，不要把索引哈希当成其科学正确性证明。

请给出可接受结论、错误及未证事项，并为下一轮单一增量预测实验提出最少对照、来源核对、主要损失、时间区间处理、完整账户成本、验证顺序和停止条件。先核实是否有同构旧实验，不通过预测门槛则停止，不以换方向、阈值、均值/分位数、窗口或费用救援。请区分来源可行、预测有效和目标达成。
""")
    save(OUT / "source_visual_verification.json", {"completed_at": now(), "status": "TWO_NEWLY_LOCATED_SOURCE_PAGES_VISUALLY_INSPECTED",
        "images": [identity(p) for p in sorted((OUT / "source_images").glob("*.png"))],
        "scope": "美的2025-02-13原报告第1页身份和第2页精确归母预测表；两当前报告和两旧报告的原图上一轮已核实",
        "all_94_pdf_pages_visually_inspected": False})
    save(OUT / "dedup_inspection_and_search_receipt.json", {"completed_at": now(), "status": "BOUNDED_EXISTING_IMPLEMENTATIONS_INSPECTED",
        "inspected_paths": [x.relative_to(ROOT).as_posix() for x in source_paths(read(CONFIG)) if x.suffix in (".py", ".yaml", ".md")],
        "search_patterns": ["同报告 前值 现值 原预测 within.report explicit.revision", "分歧 离散 disagreement dispersion 机构间"],
        "missing_filename_during_initial_search": "research/forward_eps_revision_distribution_v1.py",
        "resolved_actual_filename": "research/forward_eps_revision_distribution_features_v1.py",
        "no_research_absence_claim_based_on_missing_filename": True,
        "global_equivalent_experiment_absence_proven": False,
        "finding": "官方修正、90日修正和广度已有；增长分歧见旧规划，未在已检查EPS实现中找到本次固定公司配对统计。"})
    save(OUT / "goal_followup_and_next_route.json", {"recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "current_goal_turn_classification": "PROGRESS", "same_external_blocker_consecutive_turns": 0,
        "closed_local_hypothesis": "TWO_EXPLICIT_NARRATIVES_AS_NEW_ORIGINAL_NUMERIC_SOURCE_NOT_SUPPORTED",
        "next_route": "ONE_INCREMENTAL_GROWTH_DISAGREEMENT_PREDICTION_PROTOCOL_AFTER_EQUIVALENT_STUDY_AND_SOURCE_CHECK",
        "source_screen": (DISAGREEMENT / "result.json").relative_to(ROOT).as_posix(),
        "source_eligible_months": disagreement["eligible_months"], "predictive_value": "NOT_COMPUTED",
        "must_control": ["既有盈利预测信息", "配对数量", "报告年龄差异", "原点相同及标签成熟"],
        "forbidden_rescue": ["结果后改变方向", "改汇总器", "改观察或评价窗口", "改费用"],
        "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False})
    once(OUT / "requirements-review.txt", "\n".join(f"{name}=={importlib.metadata.version(name)}" for name in ["numpy", "pandas", "pyarrow", "pdfplumber"]) + "\n")
    print("中文报告、范围说明、查重记录、源页记录与下一步已保存。", flush=True)


def package():
    if ZIP.exists():
        raise FileExistsError("交付ZIP已存在")
    checks = [("workspace_novelty_verification.json", "PASS_SEVEN_ORIGINAL_PDFS_TWO_YEAR_MATCHES_AND_OLD_RECORDS_RECOMPUTED"),
              ("workspace_disagreement_verification.json", "PASS_SAVED_PAIRING_CLOCKS_AND_SOURCE_STATISTICS_RECOMPUTED")]
    for filename, status in checks:
        if read(VERIFY / filename)["status"] != status:
            raise ValueError("工作区保存复算未通过")
    paths = set(source_paths(read(CONFIG)) + disagreement_sources(read(DISAGREEMENT_CONFIG)))
    for folder in (OUT, DISAGREEMENT):
        paths.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.update(VERIFY / filename for filename, _ in checks)
    paths.update([Path(__file__), ROOT / "scripts/verify_510300_eps_novelty_and_disagreement_packet_20260914.py",
                  ROOT / "scripts/register_510300_eps_novelty_and_disagreement_20260914.py"])
    entries = [identity(p) for p in sorted(paths)]
    readme = r"""# 510300 EPS明示修正去重与增长分歧来源检查

先读 reports/research/510300_eps_explicit_revision_novelty_diagnostic_v1/研究结论与下一步.md 和 GPT审阅提示.md。两份明示前值叙述未增加既有原表之外的预测数字；两机构增长分歧在50个源月份满足预定覆盖条件。预测价值未计算，0新拟合、0新账户，夏普1.2和年化10%的共同目标仍未达成。

包含7份完整原PDF（94页），本轮重新提取其中用于核实的14页；包含83400行保存机构输入、41700行配对结果和139行月度统计。自包含范围为本轮原件对应与从冻结保存输入重算统计，不重建全部上游券商原件事实库。

Windows离线复核：完整解压到新目录，在解压目录的PowerShell中，使用已安装numpy、pandas、pyarrow、pdfplumber的Python运行：python .\scripts\verify_510300_eps_novelty_and_disagreement_packet_20260914.py --receipt-directory .\本轮离线复核。回执目录必须尚不存在。脚本核对FILE_INDEX，复算两项研究，不联网、不拟合、不生成账户。不要执行run模式或旧模型入口。

FILE_INDEX.csv列出除自身外全部成员的路径、大小和SHA-256。两套代码与配置均由各自claim冻结，0新收益标签。原始用户附件、范围边界、旧缺失、查重检索缺失文件说明、下一步和停止条件均保留。未上传，未收到外部GPT评审；数值复算和ZIP检查不是交易目标证明。
""".encode("utf-8")
    entries.append({"path": "00_README_FIRST.md", "size_bytes": len(readme), "sha256": hashlib.sha256(readme).hexdigest()})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(sorted(entries, key=lambda x: x["path"]))
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
            raise ValueError("成员范围与索引不等")
        for relative, entry in expected.items():
            data = archive.read(relative)
            if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("ZIP成员与索引不符")
    if temporary.stat().st_size >= 512_000_000:
        raise ValueError("超出512MB，保留building对象")
    os.replace(temporary, ZIP)
    receipt = {"status": "PASS_REVIEW_ZIP_INDEX_CRC_AND_MEMBER_HASHES", "created_at": now(), "path": str(ZIP),
               "bytes": ZIP.stat().st_size, "sha256": identity(ZIP)["sha256"], "members": len(entries) + 1,
               "indexed_members": len(entries), "uncompressed_bytes": sum(x["size_bytes"] for x in entries),
               "original_pdfs": 7, "full_original_pdf_pages": 94, "reextracted_source_pages": 14,
               "new_model_fits": 0, "new_accounts": 0, "external_review_received": False, "security_audit": False, "goal_achieved": False}
    save(ZIP.with_suffix(".delivery.json"), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "package"], required=True)
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else package()
