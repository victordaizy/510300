"""直接检验原表跨列风险、消息时钟及同版价格比方向。"""
import numpy as np
import pandas as pd

from research.copper_gold_monthly_vintage_v1 import information_clock, ratio_change
from scripts.collect_copper_gold_vintages_v1 import extract_price_values, extract_monthly_periods, month_from_url


def test_source_catalogue_excludes_other_monthly_named_presentations():
    assert month_from_url("https://example.org/Baffes-April-2022.pdf") is None
    assert month_from_url("https://example.org/CMO-Pink-Sheet-April-2022.pdf") == "2022-04"
    assert month_from_url("https://example.org/CMO28Pink28Sheet28January282017.pdf") == "2017-01"


def test_pdf_split_numerals_preserve_eleven_columns():
    line = "Copper $/mt b/ 6 ,530 6 ,010 6 ,174 5 ,634 5 ,351 6 ,525 7 ,185 8 ,477 8 ,988 9 ,325 1 0,162"
    values = extract_price_values(line)
    assert len(values) == 11
    assert values[-3:] == [8988., 9325., 10162.]
    assert values[:2] == [6530., 6010.]
    header = "Jan-Dec Jan-Dec Jan-Dec Oct-Dec Jan-Mar Apr-Jun Jul-Sep Oct-Dec NovemberDecember January\nUnit 2021 2022 2023 2022 2023 2023 2023 2023 2023 2023 2024"
    assert extract_monthly_periods(header) == ["2023-11", "2023-12", "2024-01"]


def test_release_date_delay_and_later_execution():
    dates = pd.bdate_range("2024-01-01", periods=12)
    cutoff, decision, entry = information_clock(dates, "2024-01-03")
    assert cutoff == pd.Timestamp("2024-01-05 23:59", tz="Asia/Shanghai")
    assert dates[decision] == pd.Timestamp("2024-01-05")
    assert dates[entry] == pd.Timestamp("2024-01-08")


def test_ratio_uses_same_vintage_and_is_invariant_to_constant_units():
    row = {"copper_latest_month": 9900., "gold_latest_month": 2200., "copper_previous_month": 9000., "gold_previous_month": 2000.}
    assert np.isclose(ratio_change(row), 0.)
    row["copper_latest_month"] = 10000.
    original = ratio_change(row)
    row["copper_latest_month"] *= 1000
    row["copper_previous_month"] *= 1000
    assert original > 0 and np.isclose(original, ratio_change(row))
