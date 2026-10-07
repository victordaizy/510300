"""必要测试覆盖目录边界、事件/公布时钟、重复及原链因果可用性。"""
import pandas as pd
import pytest

from research import official_policy_announcement_catalog_inputs_v1 as module


def item():
    return {"catalog_id": "测试", "catalog_year": 2024, "title": "2024年中国货币政策大事记",
        "listed_published_date": "2025-02-13", "url": module.BASE}


def document(body, clock="2025-02-13 17:05:00"):
    return f"<title>2024年中国货币政策大事记</title><div>{clock}</div><div id='zoom'>{body}</div>".encode()


def test_official_module_and_all_links():
    raw = b"<input name='article_paging_list_hidden' moduleid='17134' totalpage='5'>" + "总记录数:88".encode()
    plan = module.index_plan(raw)
    assert plan["advertised_records"] == 88 and len(plan["urls"]) == 5
    assert plan["urls"][-1].endswith("17134-5.html")
    with pytest.raises(ValueError):
        module.index_plan(raw.replace(b"17134", b"99999"))


def test_catalog_event_date_never_equals_original_availability():
    rows, meta = module.parse_document(document("<p>9月24日，宣布降准，9月27日实施。</p>"), item())
    assert rows[0]["catalog_event_date"] == pd.Timestamp("2024-09-24")
    assert rows[0]["source_available_upper"] == pd.Timestamp("2025-02-13 17:05:00", tz=module.TZ)
    assert rows[0]["multiple_date_tokens"] == 2
    assert rows[0]["record_role"] == "RETROSPECTIVE_LEAD_ONLY_NOT_ORIGINAL_ANNOUNCEMENT"
    assert not rows[0]["historical_first_vintage_authenticated"]


def test_source_date_only_upper_and_mismatch():
    rows, meta = module.parse_document(document("<p>1月15日，MLF操作。</p>", ""), item())
    assert meta["source_available_upper"] == pd.Timestamp("2025-02-13 23:59:59", tz=module.TZ)
    with pytest.raises(ValueError):
        module.parse_document(document("<p>1月15日，MLF操作。</p>", "2025-02-14 08:00:00"), item())


def test_multi_tool_navigation_never_invents_good_news():
    tags = module.tools("1年期LPR与前次持平，政策利率没有下调，准备金率保持。")
    assert tags == "准备金率|政策及操作利率|贷款报价"
    rows, meta = module.parse_document(document("<p>1月15日，利率持平。</p><p>2月5日，准备金率下调。</p>"), item())
    assert len(rows) == 2 and rows[0]["navigation_tool_tags"] == "其他"


def test_duplicate_identity_preserved_across_snapshots():
    a, _ = module.parse_document(document("<p>2月5日，准备金率下调。</p>"), item())
    other = {**item(), "catalog_id": "另快照"}
    b, _ = module.parse_document(document("<p>2月5日， 准备金率下调。</p>"), other)
    assert a[0]["logical_record_id"] == b[0]["logical_record_id"]
    assert a[0]["catalog_id"] != b[0]["catalog_id"]


def test_existing_chain_prefix_and_after_close_clock():
    nodes = pd.DataFrame([{"node_id": "早", "source_available_upper": "2024-09-24T09:19:36+08:00", "channel": "利率与流动性", "stage": "宣布"},
        {"node_id": "晚", "source_available_upper": "2024-09-24T16:00:01+08:00", "channel": "资本市场", "stage": "实施"}])
    dates = pd.date_range("2024-09-23", periods=3)
    observed = pd.DataFrame({"date": dates, "decision_time": dates.tz_localize(module.TZ)+pd.Timedelta(hours=16), "stage_entry_type": ["NONE", "REPRICING", "NONE"]})
    full = module.known_chain_daily(observed, nodes)
    cut = module.known_chain_daily(observed.iloc[:2], nodes)
    pd.testing.assert_frame_equal(full.iloc[:2].reset_index(drop=True), cut)
    pd.testing.assert_frame_equal(full.iloc[:1].reset_index(drop=True), module.known_chain_daily(observed.iloc[:1], nodes))
    assert full.recorded_new_chain_ids.tolist() == ["", "早", "晚"]
    assert not full.no_record_is_no_policy.any()
