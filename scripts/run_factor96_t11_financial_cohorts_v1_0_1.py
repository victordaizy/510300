"""仅更新来源接入到V2.0.2，保留原V1披露篮子算法和时序定义。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.factor96_t11_financial_cohorts_v1 import measure_financial_cohorts

DEFINITION = ROOT / "reports/research/510300_factor96_t11_financial_cohorts_v1"
OUT = ROOT / "reports/research/510300_factor96_t11_financial_cohorts_v1_0_1"
SOURCE = ROOT / "reports/research/510300_factor96_financial_parser_scope_v2_0_2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now().astimezone().isoformat()


def prepare():
    assert not OUT.exists(), "已有定义不覆盖。"
    for item in read(DEFINITION / "definition_freeze.json")["files"]:
        assert digest(DEFINITION / item["path"]) == item["sha256"]
    for name, folder in [("factor96_t11_financial_cohorts_v1.py", "research"),
                         ("test_factor96_t11_financial_cohorts_v1.py", "tests")]:
        assert digest(ROOT / folder / name) == digest(DEFINITION / "code" / name)
    (OUT / "code").mkdir(parents=True)
    (OUT / "source_evidence").mkdir()
    shutil.copytree(DEFINITION, OUT / "original_definition")
    for path in [Path(__file__), ROOT / "research/factor96_t11_financial_cohorts_v1.py",
                 ROOT / "tests/test_factor96_t11_financial_cohorts_v1.py"]:
        shutil.copy2(path, OUT / "code" / path.name)
    references = {
        "factor_registry.json": ROOT / "reports/research/510300_factor96_mechanism_batch_v1/factor_registry.json",
        "strategy_registry.json": ROOT / "reports/research/510300_factor96_mechanism_batch_v1/strategy_registry.json",
        "financial_measurement_protocol.json": ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1/protocol.json",
        "clock_preflight_result.json": ROOT / "reports/research/510300_factor96_t11_clock_preflight_v1/result.json",
        "title_status_clock_addendum.json": ROOT / "reports/research/510300_factor96_t11_correction_catalogue_v1/title_status_clock_addendum.json",
        "mandate_before.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    for name, path in references.items():
        shutil.copy2(path, OUT / "source_evidence" / name)
    protocol = {"at": now(), "study_id": "510300_FACTOR96_T11_FINANCIAL_COHORTS_V1_0_1",
        "phase": "FINANCIAL_MEASUREMENT_ONLY_NO_PRICES_OR_ACCOUNTS",
        "known_before_freeze": "原V1篮子算法已在财报原文批次完成前冻结。本次只修改来源接入到V2.0.2，已知各版财务值、覆盖和错误，尚未执行篮子聚合或T11收益。全市场历史已在其他研究中观察，不能称独立前向样本。",
        "source": "只接入财报V2.0.2的repaired_member_report_measurements.parquet，须先有成功交付只读重算回执，输入哈希在执行前记录。原V1接入器因依赖失败的V2而未运行，保留。",
        "adapter_only": "原V1财务篮子算法和9项测试字节哈希必须保持相同；改变来源路径与验收版本，不改变阈值、行业参考、完整性、报告前沿或聚合方式。",
        "cohort": "每个名义可用日，纳入该日历史成员新公开的最新报告。迟到旧期报告不形成新事件，同公司同日仅取最新报告期；金融行业排除，行业身份未知使整篮子未知。",
        "prior_formula_unchanged": "继承已冻结L02与L04_change及200日报告年龄，不改变财报单季拆分、TTM、两年前同季意外或历史依赖。",
        "industry_reference": "对当前公司按当时行业、同季度选参考报告：可用日位于当前日减两日历年至当前日之前，报告期严格早于当前报告期；参考公司在其公告可用日为成员、非金融、报告年龄有效且L04_change已知。",
        "industry_normalization": "z=(当前L04_change-参考均值)/参考样本标准差(ddof=1)。少于2条、零波动或非有限值均未知，不加epsilon、不缩尾。两条只是数学可算下限，不代表推断证据充分。",
        "aggregation": "在各行业内分别对L02、现金质量z和原始L04_change取公司中位数，再对行业中位数取中位数。每行业一票，不是沪深300市值权重或指数EPS。",
        "completeness": "当天全部新披露非金融公司的当前字段及行业参考都须可算，才形成已知篮子。任一缺项保留NO_VIEW，不将未知改成零或退回旧报告。没有新非金融披露的日期不形成事件。",
        "positive_financial_measurement": "L02篮子>1、行业标准化现金质量篮子>=0、原始现金质量同比变化篮子>=0；只标记财务条件，不是入场。",
        "old_overlap": "旧240家公司每日存量广度模型及其失败保持；本轮是新披露事件篮子，不能称已补齐全指数覆盖或复活旧模型。",
        "point_in_time_limit": "名义日期代理、供应商行业时钟、原档案标题样本选择及历史首次版本局限全部保留。",
        "remaining_before_returns": "O02观察窗、同类首轮反应校准、入场、失效退出和完整账户规则尚须单独冻结。当前不计算任何市场收益。",
        "tests_passed_before_freeze": 9, "new_accounts": 0, "new_returns": 0,
        "new_network_requests": 0, "T11": "NOT_RUN", "O02": "NOT_COMPUTED", "goal_achieved": False}
    save(OUT / "protocol.json", protocol)
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "definition_freeze.json", {"at": now(), "files": files})
    print("T11披露日财务测量定义已冻结；真实输入尚未接入，收益与账户未计算。", flush=True)


def run():
    assert not (OUT / "run_started.json").exists()
    for item in read(OUT / "definition_freeze.json")["files"]:
        path = OUT / item["path"]
        assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
    for path in [Path(__file__), ROOT / "research/factor96_t11_financial_cohorts_v1.py"]:
        assert digest(path) == digest(OUT / "code" / path.name)
    receipt = read(SOURCE / "delivery_receipt.json")
    assert receipt["saved_output_recomputation"]["status"] == "PASS_SAVED_V2_0_2_BOUNDARY_REPAIR_AND_UNCHANGED_MEASUREMENT"
    assert read(SOURCE / "source_repair_result.json")["all_confirmed_regressions_passed"]
    assert read(SOURCE / "source_repair_result.json")["all_boundary_regressions_passed"]
    assert not (SOURCE / "parser_semantic_invalidation_addendum.json").exists()
    assert not (SOURCE / "legacy_source_invalidation_addendum.json").exists()
    name = "repaired_member_report_measurements.parquet"
    expected = next(item for item in read(SOURCE / "result_freeze.json")["files"] if item["path"] == name)
    assert digest(SOURCE / name) == expected["sha256"]
    (OUT / "inputs").mkdir()
    for file in [name, "source_repair_result.json", "measurement_replay_result.json", "delivery_receipt.json"]:
        shutil.copy2(SOURCE / file, OUT / "inputs" / file)
    save(OUT / "run_started.json", {"at": now(), "source_sha256": expected["sha256"],
        "financial_source_zip_sha256": receipt["sha256"], "definition_freeze_sha256": digest(OUT / "definition_freeze.json"),
        "new_accounts": 0, "returns_read": False})
    measured = pd.read_parquet(OUT / "inputs" / name)
    daily, companies, dependencies = measure_financial_cohorts(measured)
    for file, frame in [("daily_financial_cohorts.parquet", daily), ("company_financial_cohorts.parquet", companies),
                        ("industry_reference_dependencies.parquet", dependencies)]:
        frame.to_parquet(OUT / file, index=False)
    annual = daily.assign(year=daily.date.dt.year).groupby("year").agg(
        disclosure_days=("date", "size"), known_cohort_days=("cohort_known", "sum"),
        positive_financial_days=("positive_financial_measurement", "sum"),
        median_new_nonfinancial_reports=("nonfinancial_reports", "median")).reset_index()
    annual.to_csv(OUT / "yearly_cohort_measurement.csv", index=False, encoding="utf-8-sig")
    result = {"at": now(), "status": "COMPLETED_FINANCIAL_COHORT_MEASUREMENT_ONLY",
        "disclosure_days": len(daily), "known_cohort_days": int(daily.cohort_known.sum()),
        "positive_financial_days": int(daily.positive_financial_measurement.sum()),
        "company_rows": len(companies), "reference_dependency_rows": len(dependencies),
        "cohort_status_counts": daily.status.value_counts().to_dict(), "annual_measurement": annual.to_dict("records"),
        "O02": "NOT_COMPUTED", "T11": "NOT_RUN", "new_accounts": 0, "new_returns": 0,
        "new_network_requests": 0, "independent_forward_observations": 0, "goal_achieved": False}
    save(OUT / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run()
