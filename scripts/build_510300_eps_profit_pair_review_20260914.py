"""交付两对研报口径诊断及可离线复算的有明确范围的小型审阅包。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_profit_attribution_pair_adjudication_v1 import CONFIG, OUT, identity, now, read, save, source_paths

PREVIOUS = ROOT / "reports/research/510300_eps_disclosed_holdings_diagnostic_v1"
VERIFY = ROOT / "reports/research/510300_eps_profit_pair_delivery_verification_20260914"
ZIP = ROOT / "deliverables/510300_EPS两对归母口径核实_V1_GPT审阅_20260914.zip"
UPSTREAM = ROOT / "deliverables/510300_EPS实际持仓覆盖诊断_V1_GPT审阅_20260914.zip"
EXPECTED_UPSTREAM_SHA = "07dcc8e6bd327ba9be0f3111eb59f352d4658f0c6d5c2e36b58db90acb1b60dc"


def once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def prepare() -> None:
    config = read(CONFIG)
    result = read(OUT / "result.json")
    upstream_identity = identity(UPSTREAM)
    if upstream_identity["sha256"] != EXPECTED_UPSTREAM_SHA:
        raise ValueError("上一轮实际交付ZIP与冻结身份不符")
    save(OUT / "upstream_archive_identity.json", {
        "verified_at": now(), "status": "PASS_UPSTREAM_ARCHIVE_IDENTITY",
        "archive": upstream_identity, "full_upstream_original_pdfs": 659,
        "this_packet_scope": "独立复算四原件两对归母口径及从保存基线到条件覆盖的增量；不重建完整659份研报事实库",
        "upstream_full_recomputation_receipt": identity(ROOT / "reports/research/510300_eps_holdings_delivery_verification_20260914/extracted_packet_verification.json")})
    observations = []
    for pair in config["pairs"]:
        for role in ("current", "prior"):
            aid = pair[role + "_report_id"]
            page = read(OUT / "pages" / (aid + ".json"))
            facts = read(ROOT / config["original_fact_root"] / (aid + ".json"))
            metadata = read(ROOT / config["original_record_root"] / (aid + ".json"))
            date_match = re.search(r"证券研究报告\s*\|\s*(\d{4})年(\d{2})月(\d{2})日", page["pages"][0]["text"])
            if date_match is None:
                raise ValueError("首页署期无法唯一核对")
            internal = "-".join(date_match.groups())
            if internal != facts["report_internal_date"]:
                raise ValueError("原PDF署期与旧事实不一致")
            observations.append({"report_id": aid, "original_pdf": page["source"], "pdf_front_date": internal,
                                 "old_fact_internal_date": facts["report_internal_date"],
                                 "directory_date": metadata["directory_record"]["publishDate"][:10],
                                 "provider_notice_date": metadata["provider_metadata"]["notice_date"][:10],
                                 "provider_eitime": metadata["provider_metadata"]["eitime"],
                                 "conservative_information_date": facts["conservative_information_date"],
                                 "old_fact_date_error_found": False})
    save(OUT / "document_date_observation.json", {"completed_at": now(), "status": "PASS_PDF_DATES_AND_CONSERVATIVE_CLOCK_RECONCILED",
        "documents": observations, "specific_note": "兴业AP202503311649636738首页署期2025-03-30、平台日期2025-03-31；旧事实已正确保留差异，继续使用较晚3月31日。没有旧事实日期错误。",
        "historic_first_publication_instant_proven": False, "old_facts_changed": False})
    images = sorted((OUT / "source_images").glob("*.png"))
    if len(images) != 8:
        raise ValueError("原件目视核对页应为8页")
    save(OUT / "source_visual_verification.json", {"completed_at": now(), "status": "EIGHT_SOURCE_PAGES_VISUALLY_INSPECTED",
        "images": [identity(p) for p in images], "scope": "四份原PDF首页及详细利润表；核对公司、年度、单位、归母行、银行合并净利润行与署期",
        "all_twenty_pages_visually_inspected": False, "all_twenty_pages_text_reextracted": True})
    save(OUT / "frozen_verifier_failure_receipt.json", {"recorded_at": now(), "status": "FROZEN_VERIFY_FAILED_ON_NONPERSISTED_ROW_INDEX",
        "command": "python research/eps_profit_attribution_pair_adjudication_v1.py --mode verify --receipt reports/research/510300_eps_profit_pair_delivery_verification_20260914/workspace_verification.json",
        "exit_code": 1, "error": "AssertionError: DataFrame.index values are different (100.0 %); left RangeIndex(start=653, stop=990, step=1), right RangeIndex(start=0, stop=337, step=1)",
        "cause": "筛选保留原990行中的653至989内存索引，Parquet按原代码index=False不保存该索引。原核对器错误地比较了未持久化的行号。",
        "saved_result_json_comparison_reached_and_passed_before_failure": True,
        "remedy": "单独V1.1离线核对器规范内存索引后比较原顺序、业务键及所有字段精确值；冻结核心、原claim和输出全部保留。",
        "numerical_success_claimed_from_failed_command": False, "strategy_or_data_rules_changed": False})
    before, after = result["before"], result["conditional_after"]
    table = ["| 公司 | 固定比较报告（平台日期） | 2026年归母利润：前值→现值，百万元 | 普通相对变化 | 旧公式对称修正 |",
             "|---|---|---:|---:|---:|"]
    for pair in result["pairs"]:
        old = next(d for d in result["documents"] if d["report_id"] == pair["prior_report_id"])
        new = next(d for d in result["documents"] if d["report_id"] == pair["current_report_id"])
        table.append(f"| {pair['name']} | {old['conservative_information_date']}→{new['conservative_information_date']} | {pair['prior_reported_million']}→{pair['current_reported_million']} | {pair['relative_change']:+.4%} | {pair['symmetric_revision']:+.6f} |")
    report = f"""# 两处口径缺失已核实，补齐不足以改变整体修正方向不明的结论

只操作510300与现金、完整账户成本后夏普至少1.2且复合年化至少10%的目标仍未达成。本轮完成四份现成研报的具体归母利润口径核实及条件覆盖计算，0新联网、0收益标签、0拟合、0账户。不是一个新交易策略试验。

## 查明的事实

{chr(10).join(table)}

固定2025-04-30原点及2026财政年度，前次报告沿用原点减90日规则，两端均受180日年龄限制。美的首页简称“净利润”的旧表，与同报告第4页“归属于母公司净利润”的5列数值精确相同。兴业两份报告第2页分别列出净利润、归母净利润和普通股东净利润；首页精确百万元值与详细表十亿元整数符合舍入精度。2026年首页值对应归母84/80十亿元，并排除合并净利润85/81十亿元。归母与普通股东利润不混同。

共同历史2023年数值也对应：美的33720百万元、兴业77116百万元。它支持这四份文件的具体行归属；不能外推为所有“净利润”标签都等于归母利润。

兴业当前报告PDF署期为2025-03-30，平台日期为2025-03-31。旧事实正确保留了两者，保守可用日期继续采用3月31日；此次核对没有发现旧日期事实错误。历史首次公开的逐秒时点仍未证明。

## 对整体信息覆盖的实际影响

条件计算沿用上一轮2024年末基金完整股票持仓，分母为全部股票公允价值，不是当日沪深300权重。除这两家公司原缺失修正外，其他字段和公司均原样保留。

| 项目 | 保存基线 | 有条件纳入两项后 |
|---|---:|---:|
| 有修正事实公司数 | {before['covered_company_count']} | {after['covered_company_count']} |
| 覆盖已披露股票资产 | {before['covered_stock_weight']:.4%} | {after['covered_stock_weight']:.4%} |
| 未知股票资产 | {before['unknown_stock_weight']:.4%} | {after['unknown_stock_weight']:.4%} |
| 全股票修正符号加权边界 | [{before['lower_bound']:+.6f}, {before['upper_bound']:+.6f}] | [{after['lower_bound']:+.6f}, {after['upper_bound']:+.6f}] |

覆盖只增加{result['added_stock_weight']*100:.4f}个百分点。边界的算法是已知公司修正符号乘股票市值占比求和，再允许每份未知股票资产的符号在-1至+1之间变化；它是缺失值的最坏情形界，不是统计置信区间，也不是未来510300收益预测。

纳入后区间仍跨零。即使把这两项都按正修正作最有利反事实，下界也只有{result['both_recovered_revisions_positive_best_case_lower_bound']:+.6f}，仍不能确定全股票为正修正。因此，两处小补丁不足以支持整体方向主张，也不足以启动旧EPS账户的再回放。

“有条件”有具体含义：报告内单位和共同历史列已经核实，但旧事实缺少的独立ISO币种证明没有在本轮补成；报告特定候选记录不改写原因子、原NO_VIEW及旧权重准入。

## 研究去向与停止条件

本轮将“只修好这两处标签即可改变整体方向判断”这一局部设想关闭。旧修正广度、报告更新率、价格组合及趋势方法已经做过，不以新名字重跑；不恢复暂停的大型来源队列。

四份原文还出现同一份报告直接列出预测前值和现值的句子。这与按固定90日跨报告比较不完全相同，例如美的当前报告叙述的前值可能来自90日基准之后的另一次预测。它只能成为下一步有限来源可行性核对的候选。先查重、核实前值财政年度和所指历史版本、披露日期及已有事实是否已包含该事件；无法证明新增信息时停止，不新建账户。若存在可检验的独立信息，再固定少量规则和样本、先做预测比较，最后才允许完整账户的共同目标检验。此处尚未宣称该候选没做过或有预测力。

## 复核、范围和失败记录

原冻结核对入口曾因未持久化的DataFrame行索引653–989与0–336不同而失败，失败回执保留。新增V1.1离线核对器只规范该内存索引，严格核对原行顺序、业务键和所有数值；不更改冻结核心、claim或结果。

本包自包含本轮两对口径和从保存基线到条件增量的全部计算输入：四份原研报及目录、HTML、旧事实、新全文提取、8页原图、固定代码、配置、结果和基线快照。附三份基金原年报、上轮离线复算回执及上一轮完整ZIP身份。完整659份券商原件的全量事实重建由上一轮393925629字节包承担，本包不重复装入其余655份原件，不把上轮回执等同于本轮完整事实库重算。

可运行入口、依赖和文件索引说明见包根00_README_FIRST.md。本包的结构检查与数值复算不等于外部GPT评审；目标未达成，没有新增交易授权或仓位动作。
"""
    once(OUT / "研究结论与下一步.md", report)
    once(OUT / "用户请求与本轮范围.txt", "原始请求：我们类似的如果没做过可以做一下，最终目标是只操作510300，实现夏普1.2，年化10\n后续目标：实现夏普1.2，年化10，只操作510300，我们已经做过很多工作了，找到正确的方向，识别下跌，上涨，然后实现我们的目标，你可以加入其他免费有效信息\n本轮范围：四份现成原研报、两对归母利润、一个固定原点的条件来源覆盖诊断；不新增策略回测。\n")
    once(OUT / "原始用户附件.txt", (PREVIOUS / "原始用户附件.txt").read_text(encoding="utf-8"))
    once(OUT / "GPT审阅提示.md", """请独立审阅本包，目标是只操作510300与现金，完整账户成本后复合年化至少10%、净夏普至少1.2。先读研究结论、协议、原始用户附件，再查看四份原PDF首页和详细利润表。请重点批评：具体归母语义证据是否充分；银行精确值与整数表的舍入排除是否成立；报告署期与平台日期使用是否保守；单位一致是否被夸大为独立币种认证；条件覆盖分母与未知边界是否正确；原行索引修复是否仅影响验证表示；是否把盈利事实误当作未来价格方向。

请明确指出可接受结论、错误、未证明事项和重复研究。完整659份研报的基线事实重建不在本包独立复算范围内；本轮保存基线及上轮复算回执不能冒充全部原件在场。

请给下一轮研究优先级、最少新增证据、预测与完整账户验证条件、失败停止条件。评价“同报告明确前值/现值”是否仅重复已有修正事件，必须先核实已有代码、财政年度和所指版本。不要建议通过调整阈值、样本、方向、成本或挑选相对好看的旧策略救援冻结失败。请区分自包含增量交付、外部评审、统计有效性和达到交易目标。
""")
    versions = [f"{name}=={importlib.metadata.version(name)}" for name in ("numpy", "pandas", "pyarrow", "pdfplumber", "matplotlib", "pypdfium2")]
    once(OUT / "requirements-review.txt", "\n".join(versions) + "\n")
    save(OUT / "goal_followup_and_next_route.json", {"recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "current_goal_turn_classification": "PROGRESS", "same_external_blocker_consecutive_turns": 0,
        "completed_work": ["四份原PDF全文提取与两对归母行语义核对", "两项条件覆盖增量及最有利方向边界计算", "日期差异核对"],
        "local_route_status": "CLOSED_TWO_LABEL_FIXES_INSUFFICIENT_FOR_OVERALL_REVISION_DIRECTION",
        "next_candidate": "同报告明示前值现值的来源与既有事件去重检查，先证明财政年度及历史版本，不生成收益或账户",
        "candidate_not_yet_claimed_novel_or_predictive": True, "new_model_fits": 0, "new_accounts": 0,
        "old_rejected_branches_preserved": True, "goal_achieved": False})
    plot_bounds(result)
    print(json.dumps({"status": "DELIVERY_TEXT_DATE_EVIDENCE_AND_BOUNDS_FIGURE_PREPARED", "output": str(OUT)}, ensure_ascii=False), flush=True)


def plot_bounds(result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")
    for y, key, label, color in [(1, "before", "保存基线", "#697c91"), (0, "conditional_after", "有条件纳入两项", "#087f8c")]:
        value = result[key]
        ax.plot([value["lower_bound"], value["upper_bound"]], [y, y], color=color, linewidth=13, solid_capstyle="round")
        ax.scatter([value["known_signed_weight"]], [y], color="#172b4d", s=70, zorder=3)
        ax.text(0, y + 0.22, f"{label}：覆盖 {value['covered_stock_weight']:.2%}，边界 [{value['lower_bound']:+.3f}, {value['upper_bound']:+.3f}]", ha="center", fontproperties=font, fontsize=12)
    ax.axvline(0, color="#b44b42", linewidth=1.4, linestyle="--")
    ax.set(xlim=(-0.46, 0.46), ylim=(-0.6, 1.6), yticks=[])
    ax.set_xlabel("已知修正符号加权和 ± 未知股票资产占比", fontproperties=font, fontsize=11)
    ax.set_title("两处口径补齐后，整体修正方向仍不确定", fontproperties=font, fontsize=17, pad=20)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.14)
    fig.text(0.5, 0.035, "固定原点：2025-04-30；2024年末已披露股票资产。此为缺失值最坏情形边界，不是置信区间或股价预测。", ha="center", fontproperties=font, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "两项补齐与整体修正方向边界.png")
    plt.close(fig)


def package() -> None:
    if ZIP.exists():
        raise FileExistsError("本轮ZIP已存在，不能覆盖")
    verified = read(VERIFY / "workspace_verification_v1_1.json")
    if verified["status"] != "PASS_FOUR_ORIGINAL_PDFS_TWO_SEMANTIC_PAIRS_AND_CONDITIONAL_BOUNDS_RECOMPUTED":
        raise ValueError("工作区数值核对未通过")
    files: dict[str, Path] = {}
    def add(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        files[path.relative_to(ROOT).as_posix()] = path
    def tree(path: Path) -> None:
        for item in path.rglob("*"):
            if item.is_file() and "__pycache__" not in item.parts:
                add(item)
    tree(OUT)
    config = read(CONFIG)
    for path in source_paths(config):
        add(path)
    for pair in config["pairs"]:
        for role in ("current", "prior"):
            add(ROOT / "reports/research/510300_forward_eps_csi_originals_v1/page_texts" / (pair[role + "_report_id"] + ".json"))
    for name in ["annual_2017_pages.json", "annual_2020_pages.json", "annual_2024_pages.json", "source_collection_claim.json", "source_collection_result.json", "selected_earnings_source_closure.json"]:
        add(PREVIOUS / name)
    tree(ROOT / read(PREVIOUS / "source_collection_claim.json")["raw_directory"])
    for name in ["workspace_verification.json", "extracted_packet_verification.json", "extracted_offline_recomputation.json"]:
        add(ROOT / "reports/research/510300_eps_holdings_delivery_verification_20260914" / name)
    add(UPSTREAM.with_suffix(".delivery.json"))
    add(VERIFY / "workspace_verification_v1_1.json")
    add(Path(__file__))
    add(ROOT / "scripts/verify_510300_eps_profit_pair_packet_v1_1.py")
    add(ROOT / "scripts/register_510300_eps_profit_pair_20260914.py")
    readme = r"""# 510300两对研报归母口径核实V1

先读 reports/research/510300_eps_profit_attribution_pair_adjudication_v1/研究结论与下一步.md，再读同目录GPT审阅提示.md。两对特定归母语义可以核实，条件覆盖63.07%→66.20%，整体修正方向仍不确定。夏普1.2和年化10%的共同目标尚未达成。

本包自包含本轮四份原件、两对语义核对以及从保存基线到条件覆盖的增量计算。包含原PDF、HTML、平台目录、旧事实和日期、新20页文本、8页原图、337行条件结果、990行原保存基线、代码与冻结claim。附三份基金原年报及上轮实际解压复算回执。完整659份券商原件的整体基线重建不在本包范围；其完整上轮包名、SHA-256和大小见upstream_archive_identity.json，不将外部引用当作本包文件已经齐全。

Windows离线复核：完整解压到一个新目录，在解压目录打开PowerShell。使用已有包含numpy、pandas、pyarrow、pdfplumber的Python，实际版本在研究目录requirements-review.txt。运行 python .\scripts\verify_510300_eps_profit_pair_packet_v1_1.py --receipt .\本轮离线复核回执.json。此入口核对FILE_INDEX及冻结输入，重新提取四份原PDF共20页，按原顺序和业务键逐值核对337行及保存结果。不要调用run模式；它不是复核入口。原V1验证器的行索引失败和V1.1处理依据都保留。

FILE_INDEX.csv覆盖除自身外的所有成员，含相对路径、字节数、SHA-256。结构与数值复核不是外部GPT评审、交易收益证明或新的仓位操作。本包不包含虚拟环境、Git对象、凭据或其他无关项目材料。
"""
    memory = {"00_README_FIRST.md": readme.encode("utf-8")}
    entries = [identity(path) for _, path in sorted(files.items())]
    entries += [{"path": rel, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for rel, data in memory.items()]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(sorted(entries, key=lambda x: x["path"]))
    temporary = ZIP.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in sorted(files.items()):
            archive.write(path, relative)
        for relative, data in memory.items():
            archive.writestr(relative, data)
        archive.writestr("FILE_INDEX.csv", buffer.getvalue().encode("utf-8-sig"))
    with zipfile.ZipFile(temporary) as archive:
        expected = {entry["path"]: entry for entry in entries}
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("ZIP的CRC或重名检查失败")
        if set(archive.namelist()) != set(expected) | {"FILE_INDEX.csv"}:
            raise ValueError("ZIP成员与索引不一致")
        for relative, entry in expected.items():
            data = archive.read(relative)
            if len(data) != entry["size_bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"ZIP成员与索引哈希不一致：{relative}")
    if temporary.stat().st_size >= 512_000_000:
        raise ValueError("ZIP超过512MB，保留building对象")
    os.replace(temporary, ZIP)
    record = {"status": "PASS_REVIEW_ZIP_INDEX_CRC_AND_MEMBER_HASHES", "created_at": now(), "path": str(ZIP),
              "bytes": ZIP.stat().st_size, "sha256": identity(ZIP)["sha256"], "members": len(entries) + 1,
              "indexed_members": len(entries), "uncompressed_bytes": sum(x["size_bytes"] for x in entries),
              "earnings_original_pdfs": 4, "context_fund_original_pdfs": 3,
              "full_upstream_659_report_reconstruction_in_scope": False,
              "new_model_fits": 0, "new_accounts": 0, "external_review_received": False,
              "security_audit": False, "goal_achieved": False}
    save(ZIP.with_suffix(".delivery.json"), record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "package"], required=True)
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else package()
