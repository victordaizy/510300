"""独立保存财务解析修复的输入、旧版复现及范围，不改任何既有冻结档案。"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

from research.factor96_earnings_cashflow_measurement_v1 import clean, digest, now, save


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_financial_parser_repair_v1"
PRIOR = ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1"
STUDY = "510300_FACTOR96_FINANCIAL_PARSER_REPAIR_V1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    paths = {
        "inputs/verified_facts_before.parquet": PRIOR / "verified_facts.parquet",
        "inputs/original_facts_before.parquet": PRIOR / "inputs/original_facts.parquet",
        "inputs/confirmed_contradictions.json": PRIOR / "anomaly_evidence/confirmed_field_contradictions.json",
        "inputs/prior_invalidation.json": PRIOR / "source_invalidation_addendum.json",
        "source_evidence/prior_delivery_receipt.json": PRIOR / "delivery_receipt.json",
        "source_evidence/prior_collection_receipt.json": PRIOR / "anomaly_evidence/collection_receipt.json",
        "source_evidence/prior_protocol.json": PRIOR / "protocol.json",
        "source_evidence/mandate_before.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
        "code/reproduction_test.py": ROOT / "tests/test_factor96_financial_parser_repair_v1.py",
        "code/initial_driver.py": Path(__file__),
    }
    for p in (PRIOR / "anomaly_evidence/pdf").glob("*.pdf"):
        paths["inputs/pdf/" + p.name] = p
    for name in ["csi300_pit_fundamental_underreaction_official_facts_v1.py", *[f"csi300_pit_fundamental_underreaction_official_facts_v1_{v}.py" for v in range(1, 8)]]:
        paths["code/legacy/" + name] = ROOT / "research" / name
    for name in ["csi300_pit_fundamental_underreaction_official_facts_v1_4_manual_appendix.json", "csi300_pit_fundamental_underreaction_official_facts_v1_5_quarterly_contamination_corrections.json"]:
        paths["code/legacy/config/" + name] = ROOT / "config" / name
    for p in (ROOT / "reports/research/510300_factor96_program_v1").iterdir():
        if p.is_file():
            paths["program_before/" + p.name] = p
    items = []
    for name, source in paths.items():
        target = OUT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        items.append({"original": source.relative_to(ROOT).as_posix(), "snapshot": name, "bytes": target.stat().st_size, "sha256": digest(target)})
    protocol = {"at": now(), "study_id": STUDY, "previous_turn_classification": "PROGRESS_MEASUREMENT_AND_SEVEN_SOURCE_CONTRADICTIONS",
        "question": "修复摘要财务字段的行列、单位、脚注和完整金额选择错误，形成独立来源修正版；不调整经济公式或策略阈值。",
        "known_regressions": "六份相同哈希原PDF中的七个错误字段，由前轮原页确定期望金额；先用旧公开提取API复现，再对新解析器运行同一测试。",
        "scope_before_fix": "三项T11基础字段：归母净利润YTD、经营现金流YTD、期末总资产；完整档案保留作来源范围，其他六项财务字段不据本轮测试声称修复。",
        "baseline": "原V1_6公开PDF接口，旧代码只读并按哈希记录；新实现放独立模块，不向旧全局模块注入变更。",
        "source_admission": "当前保存PDF可用且哈希与原档案一致时才能按原记录历史文档身份修复；缺全文、版本不同、行列/单位/当期口径不确定则保持未知。",
        "staging": "先六份原PDF回归；新范围由全部同一语义解析路径定义，排查标准不读取收益。新增公开请求另在下载前固定来源清单。",
        "research_boundary": "源字段修正不等于T11收益验证或夏普目标达成；旧数据、错误计数、测量结果和研究失败保留。",
        "diagnostic_context": "使用既有研究协议与本地记录，无工单发布或仓库技能配置变更；无需将工单系统设置作为财务修复的前置条件。",
        "network_requests_before_target_freeze": 0, "returns_read": False, "new_accounts": 0, "goal_achieved": False,
        "goal_status": "active", "orders_authorized": False, "external_review": "NOT_PERFORMED"}
    save(OUT / "protocol.json", protocol)
    save(OUT / "source_manifest.json", {"at": now(), "sources": items})
    save(OUT / "initial_freeze.json", {"at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(OUT.rglob("*")) if p.is_file()]})
    print("已保存旧解析器、六份原PDF及七个期望值；修复前输入已冻结。", flush=True)


def reproduce():
    assert not (OUT / "legacy_reproduction.json").exists()
    module = import_module("research.csi300_pit_fundamental_underreaction_official_facts_v1_6")
    cards = read(OUT / "inputs/confirmed_contradictions.json")
    docs = []
    for aid in sorted(set(c["announcement_id"] for c in cards)):
        selected = [c for c in cards if c["announcement_id"] == aid]
        pdf = OUT / "inputs/pdf" / (aid + ".pdf")
        assert digest(pdf) == selected[0]["pdf_sha256"]
        quarter = pd.Timestamp(selected[0]["report_period"]).quarter
        period_type = {1: "Q1", 2: "H1", 3: "Q3", 4: "FY"}[quarter]
        begin = time.monotonic()
        result = module.extract_official_pdf_facts(pdf.read_bytes(), period_type=period_type)
        values = {r["metric_id"]: r for r in result["metrics"]}
        save(OUT / "legacy_results" / (aid + ".json"), result)
        for c in selected:
            value = values.get(c["metric_id"], {}).get("metric_value_cny")
            docs.append({"announcement_id": aid, "metric_id": c["metric_id"], "legacy_value": value,
                "archived_wrong_value": c["archived_value_cny"], "expected_original_value": c["original_page_value_cny"],
                "reproduces_archived_wrong_value": value == c["archived_value_cny"],
                "agrees_with_original_page": value == c["original_page_value_cny"]})
        print(f"旧接口复现{aid}完成，耗时{time.monotonic()-begin:.1f}秒。", flush=True)
    save(OUT / "legacy_reproduction.json", {"at": now(), "rows": docs,
        "wrong_values_reproduced": sum(r["reproduces_archived_wrong_value"] for r in docs),
        "original_page_assertions_failed": sum(not r["agrees_with_original_page"] for r in docs),
        "network_requests": 0, "new_accounts": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "reproduce"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else reproduce()
