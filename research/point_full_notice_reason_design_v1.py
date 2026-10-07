"""原2773公告全账本的税期/财政原词状态资格；仅来源与设计，不读目标。"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.point_notice_reason_information_intake_v1 import support, METADATA

CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
OLD = ROOT / "reports/research/510300_original_open_market_notice_reason_objects_v1"
OUT = ROOT / "reports/research/510300_point_full_notice_reason_design_v1"
LEDGER = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc_open_market_notice_ledger_20150105_20260814.parquet"
FIELDS = ["ORIGINAL_TAX_WORD_PRESENT", "ORIGINAL_FISCAL_SPENDING_WORD_PRESENT"]
STUDY = "510300_POINT_FULL_NOTICE_REASON_DESIGN_V1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def extract(row: dict, raw: bytes | None, rules: dict) -> dict:
    """逐字继承E73原文本定位规则，其他工具/目的模板/缺原文保持未知。"""
    item = {"ledger_date": str(pd.Timestamp(row["notice_date"]).date()), "published_at": str(row["published_at"]),
            "title": str(row["title"]), "source_url": str(row["source_url"]), "raw_path": str(row["raw_path"]),
            "ledger_sha256": str(row["raw_sha256"]), "first_vintage_authenticated": False,
            "raw_status": "UNKNOWN", "target_object": False, "reason_status": "UNKNOWN",
            "specific_reason_categories": [], "specific_reason_anchors": [], "purpose_anchors": [],
            "other_preface_sentences": [], "body_full": "", "preface_before_operation_table": "",
            "visible_clock_strings": [], "ledger_clock_visible_in_original": False}
    if raw is None:
        item["raw_status"] = "UNKNOWN_RAW_FILE_MISSING"
        return item
    if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
        item["raw_status"] = "UNKNOWN_RAW_HASH_DIFFERS_FROM_ORIGINAL_LEDGER"
        return item
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        item["raw_status"] = "UNKNOWN_ORIGINAL_UTF8_BODY_DECODE_FAILED"
        return item
    soup = BeautifulSoup(decoded, "html.parser")
    body = soup.select_one("#zoom") or soup.select_one(".TRS_Editor")
    if body is None:
        item["raw_status"] = "UNKNOWN_ORIGINAL_BODY_NOT_LOCATED"
        return item
    text = re.sub(r"\s+", "", body.get_text(" ", strip=True))
    item["body_full"] = text
    page = soup.get_text(" ", strip=True)
    item["visible_clock_strings"] = sorted(set(a+" "+b for a,b in re.findall(
        r"(20\d{2}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", page)))
    item["ledger_clock_visible_in_original"] = pd.Timestamp(row["published_at"]).strftime(
        "%Y-%m-%d %H:%M:%S") in item["visible_clock_strings"]
    item["raw_status"] = "ORIGINAL_BODY_AND_HASH_ESTABLISHED"
    positions = [text.find(term) for term in ["逆回购操作情况", "MLF操作情况", "中期借贷便利操作情况",
                                             "正回购操作情况", "央行票据操作情况"] if text.find(term) >= 0]
    preface = text[:min(positions)] if positions else text
    item["preface_before_operation_table"] = preface
    target = bool(re.search(r"逆回购|不开展公开市场操作", preface)) and not bool(re.search(
        r"买断式逆回购|央行票据互换|CBS操作", preface))
    item["target_object"] = target
    if not target:
        item["reason_status"] = "UNKNOWN_DIFFERENT_OR_UNCONFIRMED_OPERATION_OBJECT"
        return item
    for sentence in re.findall(r"[^。！？]+[。！？]?", preface):
        if not sentence.strip():
            continue
        if re.search(rules["purpose_pattern"], sentence):
            item["purpose_anchors"].append(sentence)
        categories = [key for key,pattern in rules["patterns"].items() if re.search(pattern, sentence)]
        same = bool(re.search(rules["object_pattern"], sentence)) or bool(re.search(
            r"目前|近期|今日|今天|考虑到|鉴于|为对冲|随着|受", sentence))
        if categories and same:
            item["specific_reason_anchors"].append({"sentence": sentence, "categories": categories,
                "target": "本次常规操作或明确不操作的公告理由/银行流动性状态；非DR007或股市因果"})
            item["specific_reason_categories"].extend(categories)
        elif re.search(rules["other_explicit"], sentence) and same:
            item["specific_reason_anchors"].append({"sentence": sentence, "categories": ["OTHER_EXPLICIT"],
                "target": "本次原公告其他直接约束说明，不事后改成收益类别"})
            item["specific_reason_categories"].append("OTHER_EXPLICIT")
        else:
            item["other_preface_sentences"].append(sentence)
    item["specific_reason_categories"] = sorted(set(item["specific_reason_categories"]))
    if len(item["specific_reason_categories"]) > 1:
        item["reason_status"] = "MIXED_EXPLICIT_ORIGINAL_OPERATION_REASON"
    elif item["specific_reason_categories"]:
        item["reason_status"] = "EXPLICIT_ORIGINAL_OPERATION_REASON_OR_STATE"
    elif item["purpose_anchors"]:
        item["reason_status"] = "PURPOSE_ONLY_SPECIFIC_REASON_UNKNOWN"
    else:
        item["reason_status"] = "UNKNOWN_SPECIFIC_REASON_NOT_WRITTEN"
    return item


def fields(mapped: pd.DataFrame) -> pd.DataFrame:
    result = mapped.copy()
    values = np.full((len(mapped), 2), np.nan)
    for i,row in enumerate(mapped.itertuples(index=False)):
        if row.reason_source_available:
            tags = json.loads(row.original_categories)
            values[i] = [float("TAX" in tags), float("FISCAL_SPENDING" in tags)]
    result[FIELDS] = values
    result["auxiliary_available"] = np.isfinite(values).all(axis=1)
    require(np.array_equal(result.auxiliary_available.to_numpy(), mapped.reason_source_available.to_numpy()),
            "原词状态和来源已知状态不同。")
    return result


def monthly_design(states, mapped, models):
    raw = mapped[FIELDS].to_numpy(float)
    rows = []
    for model in models:
        if model["model"] is None:
            continue
        mask = states.cycle_id.isin(model["training_cycles"]).to_numpy()
        selected = states.loc[mask]
        require(len(selected) == model["training_rows"] and (selected.exit_index <= model["fit_index"]).all()
                and (pd.to_datetime(selected.mature_date) <= pd.Timestamp(model["fit_origin"])).all(),
                "原成熟成员与时钟变化。")
        ids = selected.cycle_id.to_numpy(int)
        unique, counts = np.unique(ids, return_counts=True)
        count = dict(zip(unique, counts, strict=True))
        weights = np.asarray([1/count[i] for i in ids], float)
        x = raw[mask]
        known = np.isfinite(x).all(axis=1)
        phi = np.zeros_like(x)
        if known.any():
            mean = np.average(x[known], axis=0, weights=weights[known])
            sd = np.sqrt(np.average((x[known]-mean)**2, axis=0, weights=weights[known]))
            z = np.clip((x[known]-mean)/np.where(sd>1e-12, sd, 1.), -5., 5.)
            phi[known] = z-np.average(z, axis=0, weights=weights[known])
        dx = np.empty_like(phi)
        for cycle in unique:
            c = ids == cycle
            dx[c] = phi[c]-np.average(phi[c], axis=0, weights=weights[c])
        matrix = np.sqrt(weights[:, None])*dx
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = float(max(matrix.shape)*np.finfo(float).eps*singular[0])
        rows.append({"fit_origin": model["fit_origin"], "original_training_rows": len(selected),
                     "known_optional_rows": int(known.sum()), "tax_present_rows": int(np.nansum(x[:, 0])),
                     "fiscal_present_rows": int(np.nansum(x[:, 1])),
                     "actual_rank": int((singular>tolerance).sum()),
                     "minimum_singular_value": float(singular[-1]), "standard_svd_tolerance": tolerance})
    require(len(rows) == 115, "原115成熟月变化。")
    return pd.DataFrame(rows)


def run():
    require(not (OUT/"protocol.json").exists(), "全账本原词设计已登记，不重复。")
    OUT.mkdir(parents=True, exist_ok=True)
    tests = read(OUT/"tests_receipt.json")
    require(tests["passed"] and tests["module_sha256"] == digest(ROOT/"research/point_full_notice_reason_design_v1.py"),
            "必要解析与时钟测试版本不符。")
    paths = [ROOT/"research/point_full_notice_reason_design_v1.py", ROOT/"tests/test_point_full_notice_reason_design_v1.py",
             OUT/"tests_receipt.json", LEDGER, OLD/"全部固定原公告_正文理由目的与对象.json",
             OLD/"冻结文本定位规则_只作对象资格.json", OLD/"protocol.json", OLD/"result.json",
             ROOT/"research/point_notice_reason_information_intake_v1.py",
             ROOT/"reports/research/510300_point_notice_reason_information_intake_v1/protocol.json",
             ROOT/"reports/research/510300_point_notice_reason_information_intake_v1/summary.json",
             CURRENT/"results/training_reference/samples.parquet", CURRENT/"inputs/within_models.json"]
    protocol = {"at": now(), "study": STUDY, "technical_decision": "TECH.R147",
        "question": "R146固定2018年后子集缺口，能否由原账本中已存较早/较晚原件提供完整技术时期的税期及财政原词设计信息？",
        "difference": "原E73/R146子集终态不改。使用技术研究本来需要的整个本地原账本，新增950原件首次理由抽取，不增网页/词类、目的解释、日历或原成员。该新增原文来源不是改已失败金融窗口或参数。",
        "source_scope": "完整原账本2773行，真实最早2014-01-21至2026-08-14；文件名不能替代经济日期。原E73的1823提取记录原样复用，另外950按完全相同原文本规则读本地原件。",
        "fields": FIELDS,
        "fixed_definition": "原同日明确理由可知时，TAX/FISCAL_SPENDING原词类是否出现，各0/1。真0只说明原公告其他具体词类未写本词；不代表现实税收/财政原因不存在。未知、目的模板、其他工具及晚公布均NaN，多原词并列保持，不预定输出符号。",
        "clock": "同原日原公布钟<=15:05；不同日不前填、未知不填0，多同日目标对象歧义保持。原显示钟为历史重构，首版/首次收件未认证。",
        "design_gate": "全部原1507状态/142版本115成熟月/27原未知保留，原周期总权重1、已知标准化clip正负5并中心化，未知phi0、周期内对比；各成熟月标准SVD秩必须2，tol=max(shape)*float64eps*最大奇异值。",
        "no_target_read": True, "no_financial_function_run": True, "new_fits": 0, "new_accounts": 0,
        "no_rescue": "任一成熟月不识别即该固定原词设计结束，不改税期/财政列、事件寿命、同日、窗口、成员、目的或未知后重试。来源通过才另登记唯一残差金融函数。",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]}
    write_json(OUT/"protocol.json", protocol, exclusive=True)
    ledger = pd.read_parquet(LEDGER, columns=["notice_date", "published_at", "title", "source_url", "raw_sha256", "raw_path"])
    require(len(ledger)==2773 and ledger.raw_path.is_unique, "原完整2773公告账本变化或路径重复。")
    old = {row["raw_path"]:row for row in read(OLD/"全部固定原公告_正文理由目的与对象.json")}
    require(len(old)==1823, "原1823对象变化。")
    rules = read(OLD/"冻结文本定位规则_只作对象资格.json")
    objects, receipts = [], []
    reused = 0
    for source in ledger.to_dict(orient="records"):
        if source["raw_path"] in old:
            item = old[source["raw_path"]]
            require(item["ledger_sha256"]==source["raw_sha256"]
                    and pd.Timestamp(item["published_at"])==pd.Timestamp(source["published_at"])
                    and item["ledger_date"]==str(pd.Timestamp(source["notice_date"]).date()), "原复用对象身份或钟变化。")
            reused += 1
        else:
            path = ROOT/source["raw_path"]
            raw = path.read_bytes() if path.exists() else None
            item = extract(source, raw, rules)
            receipts.append({"raw_path": source["raw_path"], "raw_sha256": source["raw_sha256"],
                             "bytes": len(raw) if raw is not None else 0, "raw_status": item["raw_status"]})
        objects.append(item)
    require(reused==1823 and len(receipts)==950, "原子集和新增原件身份数量变化。")
    states = pd.read_parquet(CURRENT/"results/training_reference/samples.parquet", columns=METADATA)
    require(len(states)==1507, "原1507状态变化。")
    mapped = fields(support(states, objects))
    models = read(CURRENT/"inputs/within_models.json")["models"]
    monthly = monthly_design(states, mapped, models)
    passed = monthly.actual_rank.eq(2).all()
    result = {"at": now(), "study": STUDY, "technical_decision": "TECH.R147",
        "status": "PASS_FULL_LOCAL_NOTICE_TAX_FISCAL_DESIGN_NO_TARGET" if passed
                  else "REJECTED_FIXED_FULL_NOTICE_TAX_FISCAL_DESIGN_NOT_IDENTIFIABLE",
        "original_ledger_rows": len(ledger), "old_E73_objects_reused": reused, "new_local_original_bodies_read": len(receipts),
        "new_original_body_and_hash_passed": sum(row["raw_status"]=="ORIGINAL_BODY_AND_HASH_ESTABLISHED" for row in receipts),
        "original_state_rows": len(mapped), "known_optional_rows": int(mapped.auxiliary_available.sum()),
        "unknown_rows_preserved": int((~mapped.auxiliary_available).sum()),
        "tax_word_present_rows": int(mapped[FIELDS[0]].eq(1).sum()),
        "fiscal_word_present_rows": int(mapped[FIELDS[1]].eq(1).sum()),
        "known_true_zero_both_words_rows": int(mapped[FIELDS].eq(0).all(axis=1).sum()),
        "original_mature_months": len(monthly), "rank_two_months": int(monthly.actual_rank.eq(2).sum()),
        "rank_counts": {str(k):int(v) for k,v in monthly.actual_rank.value_counts().items()},
        "first_deficient_month": monthly.loc[monthly.actual_rank.ne(2)].iloc[0].to_dict() if not passed else None,
        "design_gate_passed": bool(passed), "target_column_read": False, "new_model_fits": 0,
        "new_accounts": 0, "new_labels": 0, "new_network_requests": 0,
        "economic_stage": "NOT_RUN_SOURCE_DESIGN_NOT_ADMITTED" if not passed else "NOT_REGISTERED_UNIQUE_FINANCIAL_FUNCTION_REQUIRED",
        "net_cagr": None, "net_sharpe": None, "old_E73_R146_and_funding_failures_preserved": True,
        "physical_first_vintage_verified": False, "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False}
    for item in protocol["sources"]:
        require(digest(ROOT/item["path"])==item["sha256"], "执行期间父源或程序变化。")
    folder = OUT/"results"
    folder.mkdir(exist_ok=True)
    for name,frame in [("全部原状态_全账本税期财政原词",mapped),("全部115成熟月_税期财政设计识别",monthly)]:
        frame.to_parquet(folder/(name+".parquet"),index=False)
        frame.to_csv(folder/(name+".csv"),index=False,encoding="utf-8-sig",lineterminator="\n")
    write_json(OUT/"全部2773原公告_原文本规则对象.json",objects,exclusive=True)
    write_json(OUT/"950新增本地原件实际读取回执.json",receipts,exclusive=True)
    write_json(OUT/"summary.json",result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    run()
