"""原规则沿用、对象/目的边界和原决定钟的必要检查。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from research.point_full_notice_reason_design_v1 import extract, fields, OLD, FIELDS
from research.point_notice_reason_information_intake_v1 import support


def sample(text, time="09:10:24", container='id="zoom"'):
    raw=(f'<p>2020-01-02 {time}</p><div {container}>{text}</div>').encode("utf-8")
    source={"notice_date":"2020-01-02","published_at":f"2020-01-02 {time}","title":"公开市场业务交易公告",
            "source_url":"https://www.pbc.gov.cn/original-test", "raw_path":"original-test.html",
            "raw_sha256":hashlib.sha256(raw).hexdigest()}
    rules=json.loads((OLD/"冻结文本定位规则_只作对象资格.json").read_text(encoding="utf-8-sig"))
    return extract(source,raw,rules),source,raw,rules


def state(date="2020-01-02"):
    return pd.DataFrame({"cycle_id":[1],"origin_index":[7],"origin":[pd.Timestamp(date)]})


def test_purpose_is_not_specific_reason():
    item,_,_,_=sample("为维护银行体系流动性合理充裕，今日开展逆回购操作。")
    assert item["reason_status"]=="PURPOSE_ONLY_SPECIFIC_REASON_UNKNOWN"
    assert not item["specific_reason_categories"]
    assert not support(state(),[item]).reason_source_available.any()


def test_mixed_original_tax_and_fiscal_words_are_preserved():
    item,_,_,_=sample("考虑到税期和财政支出的影响，今日开展逆回购操作。")
    assert item["specific_reason_categories"]==["FISCAL_SPENDING","TAX"]
    assert item["reason_status"]=="MIXED_EXPLICIT_ORIGINAL_OPERATION_REASON"
    assert fields(support(state(),[item]))[FIELDS].iloc[0].tolist()==[1.,1.]


def test_tax_peak_passed_word_is_not_changed_to_opposite_category():
    item,_,_,_=sample("税期高峰已过，目前银行体系流动性总量较高，今日不开展逆回购操作。")
    assert "TAX" in item["specific_reason_categories"]
    assert "税期高峰已过" in item["specific_reason_anchors"][0]["sentence"]


def test_other_tool_is_not_substituted_for_regular_notice():
    item,_,_,_=sample("考虑到税期因素，今日开展买断式逆回购操作。")
    assert not item["target_object"] and not item["specific_reason_categories"]


def test_hash_difference_or_missing_body_stays_unknown():
    _,source,raw,rules=sample("考虑到税期因素，今日开展逆回购操作。")
    assert extract(source,raw+b" ",rules)["raw_status"]=="UNKNOWN_RAW_HASH_DIFFERS_FROM_ORIGINAL_LEDGER"
    raw2=b"<p>ordinary text</p>"
    source2=dict(source,raw_sha256=hashlib.sha256(raw2).hexdigest())
    assert extract(source2,raw2,rules)["raw_status"]=="UNKNOWN_ORIGINAL_BODY_NOT_LOCATED"
    assert extract(source,None,rules)["raw_status"]=="UNKNOWN_RAW_FILE_MISSING"


def test_original_alternative_body_container_is_allowed():
    item,_,_,_=sample("考虑到财政支出，今日不开展逆回购操作。",container='class="TRS_Editor"')
    assert item["specific_reason_categories"]==["FISCAL_SPENDING"]


def test_future_publication_and_previous_day_are_not_used():
    item,_,_,_=sample("考虑到税期因素，今日开展逆回购操作。",time="15:05:01")
    assert not support(state(),[item]).reason_source_available.any()
    assert not support(state("2020-01-03"),[item]).reason_source_available.any()


def test_exact_deadline_and_true_zero_word_absence_are_distinct_from_unknown():
    item,_,_,_=sample("目前银行体系流动性总量较高，今日不开展逆回购操作。",time="15:05:00")
    result=fields(support(state(),[item]))
    assert result.auxiliary_available.tolist()==[True]
    assert result[FIELDS].iloc[0].tolist()==[0.,0.]
    missing=fields(support(state(),[]))
    assert missing[FIELDS].isna().all().all()


def test_multiple_same_day_target_objects_remain_ambiguous():
    item,_,_,_=sample("考虑到税期因素，今日开展逆回购操作。")
    result=fields(support(state(),[item,dict(item,raw_path="second-original.html")]))
    assert not result.auxiliary_available.any() and result[FIELDS].isna().all().all()
