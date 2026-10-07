"""只用已保存回执整理PR-E03结论；不请求网络，不读取策略收益。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/free_source_feasibility_20261002"
FOLLOWUP = OUT / "metadata_followup"
ZIP = OUT / "zip_schema_probe"
ORIGINAL = ROOT / "config/510300_pressure_recovery_v1.json"


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name: str, value: object) -> None:
    target = OUT / name
    if target.exists():
        raise RuntimeError(f"结果已存在，拒绝覆盖：{target}")
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    if (OUT / "summary.json").exists():
        raise SystemExit("PR-E03正式摘要已存在，拒绝重新生成。")
    base = read_json(OUT / "public_document_receipts.json")["receipts"]
    extra = read_json(FOLLOWUP / "receipts.json")["requests"]
    adaptive = read_json(FOLLOWUP / "adaptive_field_document_receipts.json")["requests"]
    documents = base + extra + adaptive
    zip_summary = read_json(ZIP / "summary.json")
    zip_ranges = read_json(ZIP / "range_receipts.json")["ranges"]
    checks = []
    for receipt in documents:
        path = Path(receipt["path"])
        checks.append({"check": "保存的公开响应内容", "source_id": receipt["source_id"],
                       "passed": path.exists() and sha(path) == receipt["sha256"]
                       and path.stat().st_size == receipt.get("saved_bytes", receipt["bytes_received"])})
    registrations = [OUT / "registration.json", FOLLOWUP / "registration.json", ZIP / "registration.json"]
    for registration_path in registrations:
        for item in read_json(registration_path)["inputs"]:
            path = Path(item["path"])
            checks.append({"check": "注册输入保持不变", "path": str(path),
                           "passed": path.exists() and sha(path) == item["sha256"]})
    document_bytes = sum(x["bytes_received"] for x in documents)
    zip_bytes = sum(x["bytes"] for x in zip_ranges) + (ZIP / "repo_tree.json").stat().st_size
    checks.extend([
        {"check": "公开文档联合预算不超过8MiB", "passed": document_bytes <= 8 * 1024**2},
        {"check": "ZIP目录及表头联合预算不超过1MiB", "passed": zip_bytes <= 1024**2},
        {"check": "ZIP摘要与分段回执字节一致", "passed": zip_bytes == zip_summary["response_bytes_received"]},
        {"check": "六个ZIP分段全部准确206", "passed": len(zip_ranges) == 6 and all(
            x["http_status"] == 206 and x["bytes"] == x["requested_bytes"]
            and x["status"] == "EXACT_RANGE_READ" for x in zip_ranges)},
        {"check": "ZIP未完整下载且没有解析市场行", "passed": not zip_summary["full_archive_downloaded"]
         and zip_summary["parsed_market_rows"] == 0},
    ])
    live_readme = FOLLOWUP / "venvoo_current_readme_authenticated.txt"
    old_readme = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo/README.md"
    readme_equal = sha(live_readme) == sha(old_readme)
    checks.append({"check": "现有认证可读的固定README与既有版本相同", "passed": readme_equal})
    if not all(x["passed"] for x in checks):
        raise SystemExit("保存回执、预算或冻结输入核对失败；不写正式来源结论。")

    metadata = {x["source_id"]: x for x in read_json(OUT / "public_metadata_summary.json")["sources"]}
    venvoo = metadata["hf_repo_metadata"]
    venvoo_names = [x["rfilename"] for x in venvoo["siblings"]]
    non_daily = [x for x in venvoo_names if not re.match(r"20\d{6}/", x)]
    alpha = []
    for number, year in enumerate([2023, 2024, 2025, 2026], 1):
        value = read_json(FOLLOWUP / f"alphat0{number}_metadata.json")
        archives = [x["rfilename"] for x in value["siblings"] if x["rfilename"].endswith(".7z")]
        dates = [re.search(r"20\d{6}", x).group() for x in archives]
        alpha.append({"repo": value["id"], "revision": value["sha"], "gated": value["gated"],
                      "year": year, "listed_archive_count": len(archives), "first_archive_date": min(dates),
                      "last_archive_date": max(dates), "partial_archive_names": [x for x in archives if "partial" in x],
                      "listed_dates_on_or_after_m1_regime": sum(x >= "20260706" for x in dates),
                      "actual_target_data_read": False, "source_independence_from_venvoo": "NOT_ESTABLISHED"})
    catalog = read_json(ZIP / "zip_member_catalog.json")["members"]
    target_names = [x["name"] for x in catalog if "510300" in x["name"]]
    checks.append({"check": "ZIP完整目录成员数与摘要相同且没有510300命名CSV",
                   "passed": len(catalog) == zip_summary["member_count"] and not any(
                       x.lower().endswith(".csv") for x in target_names)})
    if not checks[-1]["passed"]:
        raise SystemExit("ZIP目录摘要核对失败。")

    step_dir = ROOT / "reports/research/510300_pressure_recovery_v1/source_followup_20261001"
    step_pdf = step_dir / "step_0_63_20260918.pdf"
    step_pages = step_dir / "step_0_63_20260918.pages.json"
    contract = {
        "protocol_family": "上交所STEP竞价撮合平台行情，2026-09-18版本0.63",
        "primary_url": "https://www.sse.com.cn/services/tradingtech/data/c/10832589/files/5ea26a2949c943a7ae2c9838c4c252bf.pdf",
        "pdf_sha256": sha(step_pdf), "pages_json_sha256": sha(step_pages),
        "physical_pages": {
            "2": "2026-07-03的0.62修订下线旧竞价行情中的IOPV，相关字段无意义。",
            "13": "状态、快照和逐笔消息的SendingTime为交易所时间；它不是客户端接收时刻。",
            "22": "独立IOPV消息流为SecurityType=14、MDStreamID=MDE01。",
            "30": "MDE01中昨收盘、成交量/笔数/金额为0，TradingPhaseCode为空。",
            "34": "条目v的IOPV仅MDE01有意义，0无意义；条目w的T-1收盘IOPV无意义。",
            "34-35": "MD102只保留收盘价及买卖未成交总量；零量省略必须按已验证阶段、条目数及完整消息解释。"
        },
        "application_boundary": "不能把2026 STEP条款直接套到2024 LDDS或Wind等厂商CSV；必须证明导出接口、版本、流与字段映射。",
        "venvoo_conclusion": "目录没有单列MDE01文件不证明IOPV未合并；4日正字段也不能自动认证有效或作废。现有导出未建立映射。",
        "reference_economic_time_identified": False, "historical_receipt_time_identified": False,
        "individual_queue_fill_probability_identified": False,
        "frozen_PR_E01_E02_outputs_modified": False
    }
    save_json("current_step_iopv_contract.json", contract)
    review = {
        "experiment_id": "PR-E03", "reviewed_at": now(), "target_security": "510300.SH",
        "source_selection_used_return_labels": False,
        "existing_venvoo": {
            "revision": venvoo["revision"], "listed_paths": len(venvoo_names), "non_daily_paths": non_daily,
            "authenticated_current_readme_http": next(x["http_status"] for x in extra if x["source_id"] == "venvoo_current_readme_authenticated"),
            "current_readme_equals_pinned_local": readme_equal,
            "new_independent_reference_stream_file_listed": False,
            "limitation": "完整siblings目录未列独立IOPV流；不能从目录推断混合列的实际来源。原缺接收/流映射/队列证据仍在。",
            "admission": "NO_NEW_FIELD_OR_CLOCK_EVIDENCE"
        },
        "alphat_raw_csv_series": {
            "repositories": alpha,
            "public_layout": "按交易日存7z、证券目录下行情/逐笔委托/逐笔成交三CSV；公开卡片介绍十档和整数价格/10000。",
            "public_card_2026": "99个完整日档案和2026-02-24部分档案；作者称部分档案只缺301381.SZ，尚未以目标原始样本验证。",
            "actual_field_document_requests": [{"source_id": x["source_id"], "http_status": x["http_status"],
                                                 "authenticated": x["authenticated"]} for x in adaptive],
            "new_access_application_submitted": False,
            "new_access_approval_pending": False,
            "usage_contract": "人工审批的非商业研究/学术用途合同；未提交新的机构或个人资料。",
            "admission": "FIELD_DOCUMENT_ACCESS_UNAVAILABLE_NOT_ADMITTED",
            "limitation": "更老历史不增加7月6日后的同制度日；字段文档未读到，不能认证生成/接收时点或独立估值。也未证明与venvoo独立。"
        },
        "wind17_public_zip": {
            "revision": zip_summary["revision"], "archive_bytes": zip_summary["archive_bytes"],
            "member_count": len(catalog), "member_names_containing_target": target_names,
            "sample_header": zip_summary["schema_headers"], "read_bytes": zip_bytes,
            "target_coverage_verified": False, "admission": "TARGET_AND_REQUIRED_FIELDS_NOT_ESTABLISHED",
            "limitation": "完整目录没有510300命名CSV；只抽查另一证券表头，不能证明整库相同模式或所有编码形式都无目标。没有合格盘口/IOPV样本。"
        },
        "quantdb": {
            "primary_urls": ["https://www.quantdb.cn/pricing.html", "https://www.quantdb.cn/docs/fields.html", "https://www.quantdb.cn/docs/sdk.html"],
            "public_free_offer": "注册免费1GB下载额度，免费元数据和最后30行预览；API_KEY属于付费选项，不能说所有服务收费。",
            "public_tick_fields": "time是成交时刻毫秒Unix，买卖五档；金额万元、量按股规范化；未列IOPV、历史接收、通道连续性或盘后队列。",
            "account_created": False, "actual_target_sample_read": False,
            "admission": "PUBLIC_FREE_QUOTA_BUT_REQUIRED_HISTORICAL_FIELDS_NOT_ESTABLISHED",
            "limitation": "五档有时能覆盖10bp，不能一律判深度不足；公开字典和免费额度均不能认证未知历史字段或60合格日。"
        },
        "cifang": {
            "primary_urls": ["https://www.cifangquant.com/tool/data-api", "https://www.cifangquant.com/docs/data-api"],
            "public_fields": "历史ETF/LOF日线及分钟IOPV；分钟时间是bar开始，09:30代表09:30至09:31。",
            "access_contract": "hist_min/hist_intraday公开合同为SVIP，非SVIP返回403；未购买，也未取得免费历史合格样本。",
            "realtime_fields_are_historical_proof": False, "actual_target_sample_read": False,
            "admission": "NO_ZERO_COST_HISTORICAL_ADMISSIBLE_SAMPLE",
            "limitation": "实时trade_time/updated_at不证明历史IOPV经济时点或已知延迟；不能在分钟开始使用整根bar。此规则不直接移植给其他分钟源。"
        },
        "github_queries": {
            "bounded_results": {k: metadata[k]["total_count"] for k in ["github_find_china_l2", "github_find_ashare_l2", "github_find_510300_tick"]},
            "LITS": "公开页面是实时处理代码；README请求404不代表仓库不存在。没有取得510300历史数据，也没有执行外部代码。",
            "admission": "NO_VERIFIED_TARGET_DATA", "all_free_sources_nonexistent_proven": False
        },
        "next_source_sample_gate": [
            "先取得具体510300历史字段合同，证明接口/流映射、单位、经济时刻和历史可得性及误差界。",
            "只有上述证据有实际可读免费入口，才另注册一个证券、少数日期、最少列和存储/网络预算的小样本验证。",
            "M1还需2026-07-06以后独立参考与盘后执行证据；M2还需同制度同HH:MM的60个过去合格日。",
            "字段门槛未通过时不进入PR-E04；不看未来收益择源，不重跑不变543文件，不缩短60日或30独立日。"
        ]
    }
    save_json("source_contract_review.json", review)
    completion = {
        "original_goal_completed": False,
        "completed_this_turn": "免费来源的有界目录、权限及字段合同可行性检查；现行STEP接口解释。",
        "market_source_rows_already_available": 79288590,
        "new_market_table_rows_parsed": 0,
        "synchronous_reference_and_known_before_decision": "NOT_ADMITTED",
        "formal_M1_M2_event_test": "NOT_RUN_SOURCE_GATE_FAILED",
        "new_order_count": 0, "new_actual_fill_count": 0,
        "net_expectancy_pG_minus_loss_minus_cost": "NOT_COMPUTED",
        "full_account_sharpe_target_1_2": "NOT_COMPUTED",
        "tails_costs_T_plus_1_independent_financial_validation": "NOT_RUN_SOURCE_GATE_FAILED",
        "M1_M2_economic_mechanisms_rejected": False,
        "failed_RV20_proxy_reopened": False,
        "verified_external_job_waiting": False,
        "source_feasibility_is_financial_success": False
    }
    save_json("original_goal_remaining_requirements.json", completion)
    summary = {
        "study_id": "510300_PRESSURE_FREE_SOURCE_FEASIBILITY_V1", "experiment_id": "PR-E03",
        "status": "COMPLETED_BOUNDED_FREE_SOURCE_FEASIBILITY_NO_NEW_ADMISSIBLE_SOURCE", "completed_at": now(),
        "goal_achieved": False, "target_security": "510300.SH",
        "registered_parent_config_sha256": sha(ROOT / "config/510300_pressure_free_source_feasibility_v1.json"),
        "original_mechanism_config_sha256": sha(ORIGINAL),
        "new_admitted_source_count": 0, "new_market_table_rows_parsed": 0,
        "new_hf_raw_csv_repositories": len(alpha),
        "new_hf_raw_csv_archive_counts": [x["listed_archive_count"] for x in alpha],
        "new_hf_2026_raw_csv_last_listed_date": alpha[-1]["last_archive_date"],
        "new_hf_field_document_http_statuses": [x["http_status"] for x in adaptive],
        "zip_archive_bytes_not_downloaded_in_full": zip_summary["archive_bytes"],
        "zip_member_count": len(catalog), "zip_target_named_csv_found": zip_summary["target_named_csv_found"],
        "document_receipt_count": len(documents), "document_http_status_counts": dict(Counter(str(x["http_status"]) for x in documents)),
        "document_response_bytes": document_bytes, "zip_metadata_and_range_response_bytes": zip_bytes,
        "direct_probe_response_bytes": document_bytes + zip_bytes,
        "direct_probe_request_count": len(documents) + len(zip_ranges) + 1,
        "traffic_scope": "登记的直接HTTP探针响应；不包含浏览工具检索通信、TLS开销，也不是整个聊天网络流量。",
        "formal_M1_M2_event_test": "NOT_RUN_SOURCE_GATE_FAILED", "formal_event_count": None,
        "strategy_returns": "NOT_COMPUTED", "actual_orders": 0, "actual_fills": 0, "fee_usd": 0,
        "all_free_sources_nonexistent_proven": False,
        "next_step": "只有实质新增时点/估值/执行证据时定向验证小样本；当前输入不能启动PR-E04。"
    }
    save_json("summary.json", summary)
    save_json("verification.json", {
        "verified_at": now(), "status": "PASS_SAVED_SOURCE_RECEIPTS_AND_BUDGETS",
        "checks": checks, "check_count": len(checks), "passed_count": sum(x["passed"] for x in checks),
        "scope": "公开响应、注册输入、目录计数和预算的本地复核；分段原二进制未保存，不能据hash宣称独立重验远端内容。",
        "financial_validation": "NOT_RUN", "finalizer_path": str(Path(__file__)), "finalizer_sha256": sha(Path(__file__))
    })
    text = f"""# PR-E03：免费历史来源可行性结论

完成时间：{summary['completed_at']}。研究对象510300.SH，旁证仅观察。原机制配置SHA256：`{sha(ORIGINAL)}`。

**这轮完成了有限来源检查，没有取得新的合格同步估值或执行来源。原M1/M2正式检验保持NOT_RUN_SOURCE_GATE_FAILED，成交后净期望与账户夏普保持NOT_COMPUTED，原目标尚未完成。** 来源未准入不等于经济机制被拒绝，也没有证明所有免费来源不存在。

## 已查来源与实际结果

| 来源 | 本轮新增事实 | 准入结论与限制 |
|---|---|---|
| 已有venvoo | 完整目录{len(venvoo_names):,}路径；认证读取固定README成功，内容与已存版本一致 | 未列独立IOPV流文件，未新增流映射/接收/队列证据；目录不能证明混合列没有合并IOPV |
| alphat01—04原始CSV | 2023/24/25/26分别242/240/242/100日档案；2026止于6月5日，含2月24日partial；现有认证读取两个小字段文档均403 | 没有新申请/待审批作业，也未读到目标原始数据；不能增加7月6日后的同制度日。作者称partial只缺301381.SZ，此处未验证目标样本 |
| wind17免申请ZIP | 2026.zip约570MiB，完整目录{len(catalog):,}成员；没有510300命名CSV；抽查bj920000表头为股票代码、K线结束时间、OHLC、量、额 | 只读目录和另一个证券表头，未验证目标覆盖；不推断全库模式相同或所有编码形式都无目标 |
| QuantDB | 官方定价页公布注册免费1GB及免费预览；公开Tick字典为成交时刻、买卖五档，未列IOPV/历史接收/队列 | 未创建新账号或取得合格样本。五档不必然缺10bp深度，但额度与页面不补历史时点/估值证据 |
| 次方量化 | 历史分钟IOPV接口存在；hist_min/hist_intraday公开合同为SVIP，分钟时间为bar开始 | 未购买，未得到零费用合格历史样本；实时updated_at不能认证历史IOPV经济时点，不能在bar开始使用整根bar |
| 有界GitHub检索 | 三项检索分别3/0/0仓库；LITS为处理软件，未取得510300历史数据 | 未执行外部代码；搜索结果不证明免费数据在其他渠道不存在 |

数据卡片与接口页：[alphat04](https://huggingface.co/datasets/alphat04/Tick-by-Tick-Orders-China)、[wind17](https://huggingface.co/datasets/wind17/china-a-share-zip)、[QuantDB定价](https://www.quantdb.cn/pricing.html)、[QuantDB字段](https://www.quantdb.cn/docs/fields.html)、[次方字段及权限](https://www.cifangquant.com/docs/data-api)。逐条访问结果含失败响应正文见本目录JSON回执及metadata_followup目录。

## 现行IOPV合同的解释边界

上交所2026-09-18 STEP 0.63物理第2页记录7月3日已下线旧竞价行情IOPV；第22、34页把有效IOPV放入SecurityType14、MDE01流，v条目仅该流有意义，0无意义。SendingTime是交易所时间，不是客户端接收时刻。MD102未成交总量也不直接提供个人队列位置或填单概率。

这是**STEP版本合同**，不能直接套到旧LDDS或厂商重标度导出。当前数据没有证明iopv列的流映射：不能认证4日正值有效，也不能仅凭STEP规则判其无效。没有修改PR-E01/02冻结输出。具体物理页及源hash见[current_step_iopv_contract.json](current_step_iopv_contract.json)和[上交所原规范]({contract['primary_url']})。

## 完成范围、存储和验证

直接抓取公开文档28次；另有ZIP目录元数据1次和6个准确206分段，共{summary['direct_probe_request_count']}次。文档响应{document_bytes:,}字节，ZIP元数据/分段{zip_bytes:,}字节，合计{document_bytes+zip_bytes:,}字节，约{(document_bytes+zip_bytes)/1024**2:.3f}MiB。该数字不含浏览工具检索通信、TLS开销。570MiB ZIP没有完整下载，压缩前缀只供CSV表头解码，新增市场表及解析市场行0。没有费用、订单、成交、新账号或外部申请作业。

保存响应hash、注册输入、版本README一致性、目录计数与预算共{len(checks)}项通过，见[verification.json](verification.json)。这不是独立金融验证；分段二进制未保存，回执hash不是一次独立远端重验。FILE_INDEX.csv列本目录保存文件的大小和SHA256，索引不包含自身。

## 下一步的具体启动条件

1. 只有一个来源提供实际可免费读取的510300字段合同，并补接口/流映射、单位、经济时点、历史可得性及误差界，才注册少数日期、最少列的小样本验证；仅新增年数、IOPV列名或网页介绍不启动下载。
2. M1还需7月6日以后独立估值及盘后执行证据；M2还需同制度同HH:MM的60个过去合格日。原最低30独立日区间条件保留。
3. 通过后才运行PR-E04固定事件/匹配价格对照；实际成交或独立队列重放、T+1退出和费用成立后再运行20万元主账户、2万元敏感性。
4. 当前输入不满足启动条件。不得重复扫描不变543文件、用更早制度补其后基线、把当前下载时间回填历史接收、看收益择源或恢复已失败的RV20价格代理。

已完成的79,288,590行来源与47,784时点资格矩阵保留。本轮原目标的剩余证据见[original_goal_remaining_requirements.json](original_goal_remaining_requirements.json)；目录/字段工作完成不能记为成交后正净期望或夏普目标完成。
"""
    (OUT / "免费来源可行性结论.md").write_text(text, encoding="utf-8")
    paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "FILE_INDEX.csv")
    with (OUT / "FILE_INDEX.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows({"relative_path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths)
    print(f"PR-E03来源结论已保存：{len(paths)}个索引文件，{len(checks)}项核对通过，0个新准入来源；原目标未完成。")


if __name__ == "__main__":
    main()
