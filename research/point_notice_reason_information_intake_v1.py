"""原央行操作理由对技术线全成员的时钟与成熟支持审核；不读收益目标。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json

CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
SOURCE = ROOT / "reports/research/510300_original_open_market_notice_reason_objects_v1"
OUT = ROOT / "reports/research/510300_point_notice_reason_information_intake_v1"
STUDY = "510300_POINT_NOTICE_REASON_INFORMATION_INTAKE_V1"
METADATA = ["cycle_id", "origin_index", "origin", "exit_index", "mature_date"]
CATEGORIES = {"TAX", "FISCAL_SPENDING", "CASH", "MATURITY_OFFSET", "LIQUIDITY_LEVEL", "OTHER_EXPLICIT"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(name: str, frame: pd.DataFrame) -> None:
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / (name + ".parquet"), index=False)
    frame.to_csv(folder / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def original_objects() -> list[dict]:
    source = read(SOURCE / "全部固定原公告_正文理由目的与对象.json")
    require(isinstance(source, list) and len(source) == 1823, "原1823公告对象发生变化。")
    require(len({row["raw_path"] for row in source}) == len(source), "原正文路径重复。")
    for row in source:
        date = pd.Timestamp(row["ledger_date"])
        require(pd.Timestamp("2018-12-01") <= date <= pd.Timestamp("2025-12-31"), "原固定公告范围变化。")
        require(row["raw_status"] == "ORIGINAL_BODY_AND_HASH_ESTABLISHED", "原正文资格未建立。")
        require(set(row["specific_reason_categories"]) <= CATEGORIES, "原词类体系发生变化。")
        require(isinstance(row["target_object"], bool), "原操作对象状态非法。")
        require(not row["first_vintage_authenticated"], "不得把新取得原件改称历史首版已认证。")
    require(sum(row["target_object"] for row in source) == 1767, "原常规及不操作公告母体变化。")
    require(sum(bool(row["specific_reason_categories"]) and row["target_object"] for row in source) == 232,
            "原232具体理由或状态公告发生变化。")
    return source


def support(states: pd.DataFrame, source: list[dict]) -> pd.DataFrame:
    """只用同日已经公布的明确原锚；多个常规对象不选择有利的一份。"""
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "原状态身份重复。")
    by_date = {}
    for row in source:
        if row["target_object"]:
            by_date.setdefault(pd.Timestamp(row["ledger_date"]).normalize(), []).append(row)
    rows = []
    for state in states.itertuples(index=False):
        date = pd.Timestamp(state.origin).normalize()
        origin_at = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
        objects = by_date.get(date, [])
        known, published, selected = False, pd.NaT, None
        status = "NO_VIEW_NO_SAME_DAY_TARGET_NOTICE"
        if len(objects) > 1:
            status = "NO_VIEW_MULTIPLE_SAME_DAY_TARGET_OBJECTS"
        elif len(objects) == 1:
            selected = objects[0]
            if not selected["specific_reason_categories"]:
                status = "NO_VIEW_SPECIFIC_REASON_UNKNOWN_PURPOSE_OR_UNWRITTEN"
            else:
                try:
                    published = pd.Timestamp(selected["published_at"])
                    if published.tzinfo is None:
                        published = published.tz_localize("Asia/Shanghai")
                    else:
                        published = published.tz_convert("Asia/Shanghai")
                except (ValueError, TypeError):
                    published = pd.NaT
                if pd.isna(published):
                    status = "NO_VIEW_ORIGINAL_PUBLICATION_CLOCK_UNKNOWN"
                elif published.normalize() != origin_at.normalize():
                    status = "NO_VIEW_PUBLICATION_DATE_DIFFERS_FROM_TARGET_NOTICE_DATE"
                elif published > origin_at:
                    status = "NO_VIEW_NOTICE_PUBLISHED_AFTER_ORIGINAL_DECISION"
                else:
                    known = True
                    status = "EXPLICIT_ORIGINAL_REASON_ANCHOR_AVAILABLE_BEFORE_DECISION"
        rows.append({
            "cycle_id": int(state.cycle_id), "origin_index": int(state.origin_index), "origin": state.origin,
            "origin_at": origin_at, "same_day_target_objects": len(objects), "reason_source_available": known,
            "availability_status": status, "published_at": published,
            "original_reason_status": selected["reason_status"] if selected is not None else None,
            "original_categories": json.dumps(selected["specific_reason_categories"], ensure_ascii=False) if selected else None,
            "original_source_url": selected["source_url"] if selected else None,
            "original_raw_path": selected["raw_path"] if selected else None,
            "original_raw_sha256": selected["ledger_sha256"] if selected else None,
            "ledger_clock_visible_in_original": selected["ledger_clock_visible_in_original"] if selected else None,
            "physical_first_vintage_verified": False,
        })
    frame = pd.DataFrame(rows)
    require(np.array_equal(states[["cycle_id", "origin_index"]].to_numpy(),
                           frame[["cycle_id", "origin_index"]].to_numpy()), "原成员被改换。")
    part = frame.loc[frame.reason_source_available]
    require((part.published_at <= part.origin_at).all(), "理由原钟晚于原决定钟。")
    return frame


def run() -> None:
    require(not (OUT / "protocol.json").exists(), "本来源资格已经登记，不重复。")
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "research/point_notice_reason_information_intake_v1.py",
              SOURCE / "protocol.json", SOURCE / "result.json", SOURCE / "实际旧用途与原因对象资格.json",
              SOURCE / "全部固定原公告_正文理由目的与对象.json",
              CURRENT / "results/training_reference/samples.parquet", CURRENT / "inputs/within_models.json",
              ROOT / "research/point_optional_residual_model_v1.py",
              ROOT / "research/point_funding_optional_correction_v1.py",
              ROOT / "reports/research/510300_original_operation_reasons_dr007_response_v1/result.json",
              ROOT / "reports/research/510300_fiscal_reason_month_end_dr007_comparison_v1/result.json",
              ROOT / "reports/research/510300_fiscal_reason_month_end_dr007_comparison_v1/next_experiment.json"]
    protocol = {
        "at": now(), "study": STUDY, "technical_decision": "TECH.R146",
        "question": "新E73原操作理由来源是否在技术线原全部状态和原成熟训练月具备进入当前可选模型的必要信息支持？",
        "different_use": "原E73为1212日五日联合评分的完整原因字段资格；本次只在技术线1507原状态/142月115成熟月核来源钟和支持上界，不改原标签/核心，不拟合原因或资金价格模型。",
        "fixed_source": "原1823对象、1767常规或明确不操作公告；原2018-12—2025-12范围，232具体原锚/其他目的与未知分别保持，多原因原样保留。",
        "clock": "原来源显示或历史重构的公告钟在原同日15:05前；日历日严格同日，无前填/后填/跨周末续用，多个同日常规对象保持歧义。首版/历史首次收件未认证。",
        "support_gate": "原115可用成熟月均需具备识别当前两系数可选函数的必要来源支持。只计算上界min(2,max(0,已知行数-1))；上界2不等于实际秩2。无理由行的可选设计为0，不以岭可逆冒充识别。",
        "no_numeric_function_registered": True,
        "gate_scope_limit": "只决定这一固定来源在当前完整115月可识别合同下的必要资格，不推断其金融收益失败或所有未来理由研究永久无效。",
        "original_members": "原全部1507状态、142版本115可用27未知，原成熟周期/行数和时钟保持；理由未知不能编码为真实原因0，目的模板不改成原因。",
        "source_verification_scope": "复用E73已保存原正文及hash资格，核本次源文件摘要和结构；不重复1823原HTML审核，不新宣称历史首版认证。",
        "target_column_read": False, "new_model_fits": 0, "new_stock_labels": 0,
        "new_accounts": 0, "new_network_requests": 0,
        "other_branch_E73_E74_E75_and_old_funding_failures_preserved": True,
        "E76_scope": "银行超额准备金率只为另一分支未执行的来源提案；不是当前技术候选或已取得29季度序列，不在本程序采集。",
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in inputs],
    }
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    states = pd.read_parquet(CURRENT / "results/training_reference/samples.parquet", columns=METADATA)
    require(len(states) == 1507, "原1507状态母集变化。")
    source = original_objects()
    mapped = support(states, source)
    originals = read(CURRENT / "inputs/within_models.json")["models"]
    require(len(originals) == 142 and sum(row["model"] is not None for row in originals) == 115,
            "原142月及115成熟版本变化。")
    receipts = []
    for model in originals:
        if model["model"] is None:
            receipts.append({"fit_origin": model["fit_origin"], "original_core_available": False,
                             "original_training_rows": 0, "reason_known_training_rows": None,
                             "optional_two_column_rank_upper_bound": None,
                             "necessary_support_passed": None})
            continue
        mask = states.cycle_id.isin(model["training_cycles"])
        selected = states.loc[mask]
        require(len(selected) == model["training_rows"] and (selected.exit_index <= model["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(model["fit_origin"])).all(),
                "原成熟成员和时钟不一致。")
        known = int(mapped.loc[mask.to_numpy(), "reason_source_available"].sum())
        upper = min(2, max(0, known-1))
        receipts.append({"fit_origin": model["fit_origin"], "original_core_available": True,
                         "original_training_rows": len(selected), "reason_known_training_rows": known,
                         "optional_two_column_rank_upper_bound": upper,
                         "necessary_support_passed": upper == 2})
    monthly = pd.DataFrame(receipts)
    ready = monthly.loc[monthly.original_core_available]
    blocked = ready.loc[~ready.necessary_support_passed.astype(bool)]
    require(len(blocked) > 0, "若必要上界全过，须另登记实际数值设计，不能自动准入金融模型。")
    result = {
        "at": now(), "study": STUDY, "technical_decision": "TECH.R146",
        "status": "REJECTED_CURRENT_115_MONTH_OPTIONAL_FUNCTION_SOURCE_SUPPORT_NECESSARY_GATE_FAILED",
        "original_natural_rows": len(mapped), "known_same_day_reason_rows": int(mapped.reason_source_available.sum()),
        "unknown_reason_rows_preserved": int((~mapped.reason_source_available).sum()),
        "original_availability_status_counts": mapped.availability_status.value_counts().to_dict(),
        "original_monthly_records": 142, "original_available_months": 115, "original_unknown_months_preserved": 27,
        "months_with_zero_known_reason_training_rows": int(ready.reason_known_training_rows.eq(0).sum()),
        "months_below_two_column_rank_necessary_upper_bound": len(blocked),
        "months_with_possible_two_column_rank_upper_bound": len(ready)-len(blocked),
        "first_blocked_month": blocked.iloc[0].to_dict(),
        "all_original_members_kept": True, "target_column_read": False,
        "numerical_function_registered": False, "new_model_fits": 0, "new_accounts": 0,
        "new_stock_labels": 0, "new_network_requests": 0, "economic_stage": "NOT_RUN_SOURCE_NOT_ADMITTED",
        "net_cagr": None, "net_sharpe": None,
        "hypothesis_result": "原2018年底才开始的具体理由来源不能提供最早原成熟月的识别信息；不能删早期月/回填理由/把目的或未知写成原因0。",
        "actual_rank_measured": False, "rank_upper_bound_is_not_actual_rank": True,
        "physical_first_vintage_verified": False, "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False,
        "other_branch_E73_E74_E75_and_old_failures_preserved": True,
    }
    for item in protocol["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "来源执行期间变化：" + item["path"])
    table("全部原状态_同日央行理由可知性", mapped)
    table("全部原月版本_央行理由成熟信息支持上界", monthly)
    write_json(OUT / "summary.json", result, exclusive=True)
    write_json(OUT / "completion_receipt.json", {
        "at": now(), "status": "PASS_ORIGINAL_1507_MEMBER_CLOCK_AND_142_MONTH_METADATA_SUPPORT",
        "sources_checked": len(protocol["sources"]), "natural_rows": len(mapped), "monthly_records": len(monthly),
        "new_model_fits": 0, "new_accounts": 0, "target_column_read": False,
        "result_sha256": digest(OUT / "summary.json"),
    }, exclusive=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    run()
