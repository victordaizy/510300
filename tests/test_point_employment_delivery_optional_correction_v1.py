"""检验原月问项、延迟可得钟、未知与前缀稳定性，不估计金融收益。"""
import numpy as np
import pandas as pd
import pytest

from research.point_employment_delivery_optional_correction_v1 import (
    FIELDS, align, monthly_values, parse_current_month, validate_raw,
)


def page(month="2025年1月", employment="48.0", delivery="50.1", complete_header=False):
    head = ["生产", "新订单", "原材料库存", "从业人员", "供应商配送时间"]
    if complete_header:
        head = ["", "PMI"] + head
    cells = "".join("<td>" + item + "</td>" for item in head)
    current = [month, "49.8", "50.4", "50.2", "48.1", employment, delivery]
    return "<table><tr>" + cells + "</tr><tr>" + "".join("<td>" + item + "</td>" for item in current) + "</tr></table>"


def monthly(second_missing=False):
    rows = []
    for month, date, employment, delivery in [
        ("2025-01", "2025-02-01", 48.0, 50.1),
        ("2025-02", "2025-03-01", np.nan if second_missing else 48.2, 49.9),
    ]:
        rows.append({"stat_month": month, "published_at": date + "T09:30:00+08:00",
                     "known_at": date + "T23:59:59+08:00", "method_group": "原固定方法组",
                     FIELDS[0]: employment, FIELDS[1]: delivery})
    return monthly_values(rows)


def states(dates):
    return pd.DataFrame({"cycle_id": [1] * len(dates), "origin_index": range(len(dates)),
                         "origin": pd.to_datetime(dates)})


def test_current_month_only_and_two_header_layouts():
    for complete in [False, True]:
        raw = page(month="2024年12月", employment="44.0", delivery="40.0", complete_header=complete)
        current_cells = page(complete_header=complete).split("<tr>", 2)[2].split("</tr>", 1)[0]
        raw = raw.removesuffix("</table>") + "<tr>" + current_cells + "</tr></table>"
        pair, anchors = parse_current_month(raw.encode(), "2025-01")
        assert pair == [48.0, 50.1]
        assert anchors[0]["current_month_row"][0] == "2025年1月"


def test_duplicate_tables_agree_and_disagreement_rejected():
    pair, anchors = parse_current_month((page() + page(complete_header=True)).encode(), "2025-01")
    assert pair == [48.0, 50.1] and len(anchors) == 2
    with pytest.raises(ValueError):
        parse_current_month((page() + page(delivery="50.2")).encode(), "2025-01")


@pytest.mark.parametrize("employment", ["—", "-1", "101", "inf"])
def test_invalid_original_value_not_filled(employment):
    with pytest.raises(ValueError):
        parse_current_month(page(employment=employment).encode(), "2025-01")


def test_diffusion_endpoints_and_partial_unknown_preserved():
    pair, _ = parse_current_month(page(employment="0", delivery="100").encode(), "2025-01")
    assert pair == [0.0, 100.0]
    raw = np.array([[0., 100.], [np.nan, 50.2], [48., np.nan]])
    previous = raw.copy()
    assert validate_raw(raw).tolist() == [True, False, False]
    np.testing.assert_array_equal(raw, previous)


def test_publication_day_delay_and_microsecond_clocks():
    m = monthly()
    m["known_at"] = m.known_at.dt.as_unit("us")
    out = align(states(["2025-02-01", "2025-02-02", "2025-03-01", "2025-03-02"]), m)
    assert out.auxiliary_available.tolist() == [False, True, True, True]
    assert pd.isna(out.loc[0, FIELDS[0]])
    assert out.loc[2, FIELDS[0]] == 48.0
    assert out.loc[3, FIELDS[0]] == 48.2
    assert out.loc[1, "origin_at"].hour == 15 and out.loc[1, "origin_at"].minute == 5
    with pytest.raises(ValueError):
        monthly_values([{**m.iloc[0].to_dict(), "known_at": "2025-02-01T09:30:00+08:00"}])


def test_latest_missing_keeps_partial_value_no_older_backfill():
    out = align(states(["2025-02-02", "2025-03-02"]), monthly(second_missing=True))
    assert out.auxiliary_available.tolist() == [True, False]
    assert out.loc[1, "stat_month"] == "2025-02"
    assert pd.isna(out.loc[1, FIELDS[0]]) and out.loc[1, FIELDS[1]] == 49.9


def test_future_report_prefix_invariance():
    original = monthly()
    sample = states(["2025-02-01", "2025-02-02", "2025-03-02"])
    before = align(sample, original)
    future = {"stat_month": "2025-03", "published_at": "2025-03-31T09:30:00+08:00",
              "known_at": "2025-03-31T23:59:59+08:00", "method_group": "新样本组",
              FIELDS[0]: 70., FIELDS[1]: 20.}
    expanded = monthly_values(original.to_dict("records") + [future])
    pd.testing.assert_frame_equal(before, align(sample, expanded), check_exact=True)


def test_original_identity_and_clock_uniqueness():
    sample = states(["2025-02-02"])
    with pytest.raises(ValueError):
        align(pd.concat([sample, sample], ignore_index=True), monthly())
    m = monthly().to_dict("records")
    with pytest.raises(ValueError):
        monthly_values([m[0], m[0]])
    with pytest.raises(ValueError):
        validate_raw(np.array([[48.]]))


def test_method_group_and_raw_levels_are_retained():
    records = monthly().to_dict("records")
    records[1]["method_group"] = "另一原样本组"
    out = align(states(["2025-03-02"]), monthly_values(records))
    assert out.loc[0, "method_group"] == "另一原样本组"
    assert out.loc[0, FIELDS[0]] == 48.2 and out.loc[0, FIELDS[1]] == 49.9
