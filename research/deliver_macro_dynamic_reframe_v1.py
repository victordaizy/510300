"""整理宏观研究重订阶段资料，并从新解压目录复核交付包。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_macro_dynamic_reframe_v1"
FUND = ROOT / "reports/research/510300_fund_information_source_extension_v2"
PARENT = ROOT / "deliverables/510300_货币预期差扩样与五日增量_V2_GPT审阅_20260921.zip"
PARENT_HASH = "d2c4a14173d0580b1a5ee62b1844a966869a263181f4efbbf1f079109e25eff1"
ARCHIVE = ROOT / "deliverables/510300_宏观研究重订与政策动态更新_V1_GPT审阅_20260922.zip"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare() -> None:
    if digest(PARENT) != PARENT_HASH:
        raise ValueError("父包身份与已交付回执不同")
    with zipfile.ZipFile(PARENT) as z:
        if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("父包结构复核未通过")
        parent_members = len(z.namelist())
    copy(PARENT, STUDY / "history" / PARENT.name)
    write_json(STUDY / "evidence/parent_archive_identity.json", {"file": PARENT.name, "sha256": PARENT_HASH,
        "bytes": PARENT.stat().st_size, "members": parent_members, "crc": "PASS", "duplicates": 0,
        "role": "旧五日子实验完整历史快照；其中的资金前置门仅属于该旧实验，不约束新的宏观通道。"})
    parent_study = ROOT / "reports/research/510300_money_consensus_increment_v2"
    for extension in ("png", "svg"):
        source = parent_study / "figures" / ("510300_M1M2_实际与预期走势对比." + extension)
        if source.exists():
            copy(source, STUDY / "history" / source.name)
    copy(ROOT / "reports/research/510300_post_information_capital_adjustment_v1/evidence/policy_context_source_receipts.json",
         STUDY / "evidence/背景政策历史获取回执.json")

    fund_facts = [
        {"source_key": "daily_recap_20241227", "source_name": "国元证券每日复盘", "page": 8, "cover_date": "2024-12-27",
         "stat_date": "2024-12-26", "latest_shares_100m": 901.84, "latest_share_change_100m": .07,
         "five_day_share_change_100m": -7.45, "five_day_change_ratio_from_rounded_numbers": -7.45 / (901.84 + 7.45),
         "status": "TRUE_FIVE_DAY_SHARE_FACT_FOUND_PUBLICATION_CLOCK_NOT_ESTABLISHED",
         "limitation": "份额列脚注截至12月26日，成交额列另有日期；报告封面和URL日期不证明历史首次公开时点。由四舍五入值推算的比例只作字段核对，未与宏观事件准入对接。"},
        {"source_key": "hx_20221114", "source_name": "宏信证券ETF日报", "page": 2, "cover_date": "2022-11-14",
         "stat_date": None, "latest_shares_100m": 161.94, "prior_friday_local_share_value_100m": 161.941877,
         "cover_day_local_share_value_100m": 162.481877, "status": "LATEST_SHARE_REFERENCE_STAT_DATE_UNRESOLVED",
         "limitation": "数值与前一周五接近，不能据此认定统计日期，也不能用封面日代替数据所属日。"},
        {"source_key": "hx_20191119", "source_name": "宏信证券ETF日报", "cover_date": "2019-11-19", "stat_date": None,
         "latest_shares_100m": 85.27, "status": "LATEST_SHARE_REFERENCE_STAT_DATE_UNRESOLVED",
         "limitation": "排名表中的最新份额参考，缺连续五日端点和对应历史公开时点。"},
        {"source_key": "lightfund_weekly_20190729", "source_name": "光大证券周报", "page": 11, "cover_date": "2019-07-29",
         "weekly_share_change_million": -84.59, "labeled_flow_100m_cny": 1.85, "fund_size_100m_cny": 361.67,
         "status": "SHARE_CHANGE_AND_ASSET_SIZE_CHANGE_MUST_NOT_BE_CONFLATED",
         "limitation": "份额变化为负、表内所谓资金流向为正；脚注使用净资产差且有重复周末字样，不能认定1.85亿元是净申购。"},
        {"source_key": "sina_weekly_20241215", "source_name": "21世纪经济报道新浪转载", "published_at": "2024-12-15T21:27:00+08:00",
         "stat_period": "2024-12-09至2024-12-13", "reported_net_flow_100m_cny": 36.90,
         "status": "WEEKLY_AMOUNT_CONTEXT_NOT_FIVE_DAY_SHARE_RATIO",
         "limitation": "金额口径缺期初份额和完整份额变化率，不能替代冻结的份额变量。"},
    ]
    for fact in fund_facts:
        receipt = read_json(FUND / "receipts" / (fact["source_key"] + ".json"))
        fact.update({"source_url": receipt["url"], "raw_sha256": receipt["sha256"]})
    write_json(FUND / "results/已取得字段与未准入原因.json", fund_facts)
    receipts = []
    for source in sorted((FUND / "receipts").glob("*.json")):
        record = read_json(source)
        receipts.append({"key": source.stem, "url": record.get("url"), "http": record.get("http"),
                         "transport_exit": record.get("transport_exit"), "bytes": record.get("bytes"),
                         "type": record.get("type"), "access_challenge": record.get("access_challenge"),
                         "error": record.get("error"), "receipt": "receipts/" + source.name})
    write_json(FUND / "results/访问清单.json", receipts)
    fund_status = {"status": "SOURCE_EXTENSION_PARTIAL_RETAINED_SCOPE_REDIRECTED", "saved_at": now(),
        "source_receipts": len(receipts), "numeric_fact_records": len(fund_facts), "qualifying_joined_five_day_events_established": False,
        "new_models": 0, "new_accounts": 0, "parent_C_D": "NOT_RUN_FUND_CLOCK_AND_COVERAGE",
        "blocks_other_macro_channels": False,
        "finding": "公开资料确有份额和五日变化事实；尚未形成原实验需要的完整时点配对。根据用户范围纠偏，将此来源扩展保存为附录，不作为整个宏观研究的前置门。",
        "access_limits": "管理人请求失败、部分研报入口需登录或只加载页面框架；原公众号全文没有连续取得。不能据此声称网上没有资料。"}
    write_json(FUND / "results/status.json", fund_status)
    for source in [FUND / "protocol.json", *(FUND / "receipts").glob("*.json"), *(FUND / "results").glob("*.json")]:
        copy(source, STUDY / "appendix/fund_sources" / source.relative_to(FUND))
    copy(ROOT / "research/collect_fund_report_sources_v2.py", STUDY / "appendix/fund_sources/code/collect_fund_report_sources_v2.py")

    channels = [
        ("增长与盈利", "新订单已公布状态", 139, "2015-01至2026-07", "当次水平相对50、月度变化", "无准入市场共识；其他增长与盈利指标未组成冻结样本", "READY_FOR_SEPARATE_EXECUTION_CONTRACT"),
        ("通胀与价格", "尚未为本分支准入", 0, "", "收入、成本与利率约束分通道", "原始当次值、分项、事前共识与历史时点", "NOT_PREPARED_IN_THIS_STAGE"),
        ("货币与融资", "操作利率记录；货币实际值及共识另存", 25, "含起始锚点；样本覆盖止于2026-08-14", "宣布、实施、预期分列", "25条记录不是完整首次宣布；各政策共识未建立", "PARTIAL_OPERATION_RECORDS_CASE_CLOCK_RECONSTRUCTED"),
        ("财政与债务", "8项背景清单中有财政及化债案例", 0, "背景案例而非完整母集", "新增支出、债务置换、执行进度分别记录", "完整政策链、事前预期、实施及实际支出", "BACKGROUND_CASES_ONLY"),
        ("地产与资本市场制度", "背景清单含保交房及资本市场措施", 0, "背景案例而非完整母集", "需求、资产负债表与投资约束", "细则、适用对象、执行额及首次可知时点", "BACKGROUND_CASES_ONLY"),
        ("外部与贸易", "背景清单含关税及联合声明", 0, "背景案例而非完整母集", "外需、成本、风险溢价及政策修订", "完整公告修订链及预期", "BACKGROUND_CASES_ONLY"),
    ]
    pd.DataFrame(channels, columns=["通道", "现有材料", "本分支已入164行账本数量", "覆盖", "研究角色", "缺口", "状态"]).to_csv(STUDY / "results/宏观六通道覆盖与缺口.csv", index=False, encoding="utf-8-sig")
    policy = read_json(STUDY / "inputs/policy_context.json")
    policy_table = []
    for event in policy:
        policy_table.append({"date": event["date"], "family": event["category"], "event": event["label"],
            "content": event["content"], "source_url": event["source_url"], "clock_precision": event["time_precision"],
            "expectation": None, "surprise": None, "selection": "INHERITED_ILLUSTRATIVE_CASES_NOT_COMPLETE_UNIVERSE",
            "use": "背景；除单独重建的九月案例外，不作为动态交易时钟输入"})
    pd.DataFrame(policy_table).to_csv(STUDY / "results/八项背景政策_范围说明.csv", index=False, encoding="utf-8-sig")

    csrc = STUDY / "raw/csrc_20240924_transcript.html"
    gov = STUDY / "raw/gov_pboc_wechat_20240927.html"
    csrc_text = "".join(BeautifulSoup(csrc.read_bytes(), "html.parser").get_text("", strip=True).split())
    start = csrc_text.index("09:10:58")
    end = csrc_text.index("09:19:36", start)
    block = csrc_text[start:end]
    if "1.5%" not in block or "0.5个百分点" not in block:
        raise ValueError("政策数字不在两个时间标记之间")
    facts = [
        {"source_key": "csrc_20240924_transcript", "sha256": digest(csrc), "segment_start": "2024-09-24T09:10:58+08:00",
         "segment_next_marker": "2024-09-24T09:19:36+08:00", "numeric_facts": {"proposed_rate_pct": 1.5, "proposed_rrr_cut_pp": .5},
         "fact_checked_inside_timestamp_segment": True, "admission": "CASE_HISTORICAL_CLOCK_RECONSTRUCTION_NOT_FORWARD_RECEIPT",
         "limitation": "采用后一标记作为保守上界，不声称具体政策句恰好在前一标记时刻说出。"},
        {"source_key": "gov_pboc_wechat_20240927", "sha256": digest(gov), "page_date": "2024-09-27",
         "attribution": "中国人民银行微信", "numeric_facts": {"rate_old_pct": 1.7, "rate_new_pct": 1.5},
         "effective_date": "2024-09-27", "available_upper_bound": "2024-09-27T23:59:59+08:00",
         "admission": "GOVERNMENT_SYNDICATED_WECHAT_ANNOUNCEMENT_DATE_PRECISION",
         "limitation": "政府网转载，不是直接取得微信后台历史；日期精度不能伪造分钟。"},
    ]
    for fact in facts:
        receipt = read_json(STUDY / "receipts" / (fact["source_key"] + ".json"))
        if receipt["sha256"] != fact["sha256"]:
            raise ValueError("原文与获取回执不同")
        fact["source_url"] = receipt["url"]
    write_json(STUDY / "evidence/政策案例原文数字与时点核对.json", facts)
    old_copy = STUDY / "code/macro_research_reframe_v1.py"
    original_copy = STUDY / "code/initial_freeze/macro_research_reframe_v1.py"
    if old_copy.exists() and not original_copy.exists():
        copy(old_copy, original_copy)
    for name in ("macro_research_reframe_v1.py", "plot_macro_dynamic_reframe_v1.py", "verify_macro_dynamic_reframe_v1.py", "deliver_macro_dynamic_reframe_v1.py"):
        copy(ROOT / "research" / name, STUDY / "code" / name)
    write_json(STUDY / "evidence/实现修正记录.json", {"scope_freeze_changed": False,
        "changes": ["网页时间数字分处多个span，去掉标签间空白后核对；按后一标记取保守上界。",
                    "保存原10000次固定区块抽样均值，以便只读复核区间，不新抽样。",
                    "图表标签缩短避免裁切，操作利率案例按自然日显示9月29日记录。",
                    "CSV与Parquet的日期统一为纳秒单位后比较实际日期，避免仅因存储精度不同误报。"],
        "new_models": 0, "new_20d_labels": 0, "scope_note": "初始代码快照与最终运行代码均保留；scope_freeze只冻结范围与输入，未声称代码最初已完整冻结。"})
    dependencies = [f"{name}=={importlib.metadata.version(name)}" for name in ("numpy", "pandas", "pyarrow", "matplotlib", "beautifulsoup4", "tzdata")]
    (STUDY / "requirements.txt").write_text("\n".join(dependencies) + "\n", encoding="utf-8")
    write_json(STUDY / "evidence/图表目视核对.json", {"checked_at": now(), "method": "逐张打开最终PNG，核对中文、标签、图例、日期与数值；SVG同次绘图生成。",
        "figures": [p.name for p in sorted((STUDY / "figures").glob("*.png"))], "status": "PASS_NO_OBSERVED_CLIPPING_OR_LABEL_COLLISION",
        "daily_points": "全景显示2015年起的全部交易日；完整3476日另附CSV/Parquet；案例保留窗口全部交易日。",
        "not_claimed": "未产生策略净值或新的买卖信号。"})

    (STUDY / "00_阅读导航.md").write_text("""# 510300 宏观研究重订与政策动态更新 V1

本包完成的是研究范围纠偏、信息账本、旧实验诊断与政策时序案例；新二十日模型和账户未运行。范围冻结于2026-09-21，跨日后于2026-09-22交付，行情仍截止2026-09-11。

1. 先读《研究结论.md》，再看《新研究立项.md》与《用户需求与本轮边界.md》。
2. figures含三张新图的PNG/SVG；history中另保留原M1/M2实际与预期对比图。
3. results含完整3476日数据、164条继承事件、六通道缺口、八项背景政策、旧24次预测诊断和九月时钟案例。
4. inputs和scope_freeze.json保存本轮直接计算输入及原始身份。evidence保存来源、时点和图表核对，receipts保存包括失败在内的联网回执。
5. appendix/fund_sources保存纠偏前来源扩展的真实发现和未准入原因，不能把它理解为其他宏观通道的阻断条件。
6. history中的旧V2 ZIP是独立历史快照；其旧范围和门槛不覆盖本次新立项。嵌套包有自己的历史索引；本轮根FILE_INDEX.csv覆盖当前交付成员。
7. GPT审阅提问.md可复制给另一位审阅者。没有进行外部GPT审阅。

离线数值复核：安装requirements.txt所列依赖后，在解压目录使用Python运行 code/verify_macro_dynamic_reframe_v1.py --study-dir . 。脚本只读保存结果，检查冻结输入、48条已冻结系数预测等式、24个成熟训练均值、四项RMSE、区间、逐日对齐和案例价格。不会重新拟合或创建二十日标签。

生成与采集脚本仍使用项目原目录布局；独立解压包支持的是保存结果复核，不声称可离线重新取得所有原始网页。原报告正文、139份NBS网页等未全部复制，详见《交付范围与排除.md》。来源历史真实性的完整重新认证仍需原文。
""", encoding="utf-8")
    (STUDY / "GPT审阅提问.md").write_text("""# 可复制的审阅请求

请先读00_阅读导航.md、研究结论.md、新研究立项.md与scope_freeze.json，再复核results和inputs。用户要研究持续更新的宏观信息如何改变510300持有价值，覆盖增长、通胀、货币、财政、地产、资本市场和外部政策，并考虑超预期、市场状态与动态调整。旧M1/M2五日实验失败保留，不能概括为宏观研究失败。

请给出：
- 当前范围是否仍偏离用户目标？增长状态、真实预期差与政策变化是否区分清楚？
- 已保存四基准诊断与政策案例是否支持报告结论？请优先复算，不把案例当因果或新策略收益。
- 宣布、实施、细则和执行进度的时钟合同是否可执行？操作利率25条记录有何缺口？
- 下一项增长通道应怎样形成最小、完整且可证伪的统计执行合同？请具体列出基准、共同目标、训练成熟规则、有效样本单位、验收与停止条件。
- 政策通道如何建立完整母集并检验状态条件及持续更新？不要用未来涨跌给历史政策贴方向标签。
- 如何分别评价收益预测、下行风险和仓位配置，使失败后不能临时换指标？哪些数据可先独立研究，哪些缺口只阻断自己的模块？
- 给出下一阶段策略研究优先级、必要验证、无效时的停止线；保留20万元主账户、2万元成本对照、既有失败与85/15终止。

新二十日模型、账户和独立前向事件均为0。139个月新订单只是已公布状态，25条利率不是完整首次政策宣布。八项政策为背景案例。原始券商PDF及部分官网全文未全部包含，不要宣称已逐份认证历史版本。结构验证不代表科学有效或外部审阅通过。
""", encoding="utf-8")
    (STUDY / "交付范围与排除.md").write_text("""# 交付范围与排除

本包包含当前阶段的用户需求摘要、范围合同、直接计算输入、全部结果、代码、图表、来源事实与访问回执、资金扩展附录、旧V2包及根文件索引。完整3476日价格保留；尚未开展的二十日模型与账户不以零收益填充。

新获取的网页全文和券商PDF、内部长篇文字提取、截图及139份NBS原HTML没有全部复制进本包；保留研究必需数值、URL、页码、时间精度及哈希，避免重复全文分发。没有这些原始全文仍可离线复核包内数值，不能据此独立认证全部历史首次发布版本。旧包保留其自身来源边界。

直接读取了证监会9月24日发布会逐字稿与政府网转载的央行微信9月27日公告。人民网另一篇9月27日页面的本地TLS请求失败，未作为分钟准入依据。资金扩展中的访问失败、加载框架、缺统计日与缺公开时点均保留；它们不等于资料不存在。没有绕过登录或验证码。

本轮仅采用市场历史观察、数据整理和保存结果复核。没有新策略模型、账户或独立前向信号。20万元/2万元、242日年化及费用口径为后续固定设定；不代表已通过验收。没有重新启动被终止的85/15策略。
""", encoding="utf-8")
    print(json.dumps({"状态": "交付材料准备完成", "资金来源回执": len(receipts), "背景政策": len(policy_table)}, ensure_ascii=False))


def package() -> None:
    if ARCHIVE.exists():
        raise FileExistsError("最终交付包已存在，禁止覆盖")
    verifier = STUDY / "code/verify_macro_dynamic_reframe_v1.py"
    run = subprocess.run([sys.executable, str(verifier), "--study-dir", str(STUDY)], capture_output=True, text=True, encoding="utf-8", check=True)
    local_result = json.loads(run.stdout)
    write_json(STUDY / "evidence/保存结果复核.json", local_result)
    files = sorted(p for p in STUDY.rglob("*") if p.is_file() and "raw" not in p.relative_to(STUDY).parts
                   and "__pycache__" not in p.parts and p.name not in ("FILE_INDEX.csv", "delivery_receipt.json"))
    entries = [{"path": p.relative_to(STUDY).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]
    index = io.StringIO(newline="")
    writer = csv.DictWriter(index, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(entries)
    index_bytes = index.getvalue().encode("utf-8-sig")
    building = ARCHIVE.with_suffix(".building.zip")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for path in files:
            z.write(path, path.relative_to(STUDY).as_posix())
        z.writestr("FILE_INDEX.csv", index_bytes)
    if building.stat().st_size > 80 * 1024 * 1024:
        raise ValueError("交付包超过本阶段80MiB范围上限")
    with tempfile.TemporaryDirectory(prefix="macro_reframe_verify_") as temp_dir:
        with zipfile.ZipFile(building) as z:
            names = z.namelist()
            if z.testzip() is not None or len(names) != len(set(names)):
                raise ValueError("ZIP的CRC或唯一性检查未通过")
            if set(names) != {e["path"] for e in entries} | {"FILE_INDEX.csv"}:
                raise ValueError("索引成员不一致")
            for entry in entries:
                content = z.read(entry["path"])
                if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError("成员身份不一致：" + entry["path"])
            z.extractall(temp_dir)
        extracted = subprocess.run([sys.executable, str(Path(temp_dir) / "code/verify_macro_dynamic_reframe_v1.py"), "--study-dir", temp_dir], capture_output=True, text=True, encoding="utf-8", check=True)
        extracted_result = json.loads(extracted.stdout)
        if extracted_result != local_result:
            raise ValueError("新解压目录复核与本地结果不同")
    building.replace(ARCHIVE)
    (STUDY / "FILE_INDEX.csv").write_bytes(index_bytes)
    receipt = {"archive": str(ARCHIVE), "created_at": now(), "bytes": ARCHIVE.stat().st_size,
        "sha256": digest(ARCHIVE), "members": len(entries) + 1, "indexed_members": len(entries),
        "crc": "PASS", "duplicates": 0, "index_size_hash": "PASS", "fresh_extraction_saved_recomputation": extracted_result,
        "status": "PASS_STRUCTURAL_AND_SAVED_NUMERIC_RECOMPUTATION", "new_models": 0, "new_accounts": 0,
        "external_review_completed": False, "whole_macro_program_complete": False,
        "limits": "结构及保存数值通过；完整政策母集、首次历史版本认证、二十日预测、动态账户与独立验证尚未建立。"}
    write_json(ARCHIVE.with_suffix(".receipt.json"), receipt)
    write_json(STUDY / "delivery_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "fresh_extraction_saved_recomputation"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="宏观研究重订阶段资料与交付")
    parser.add_argument("action", choices=["prepare", "package"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        package()
