"""验证真实漏取报告、年度边界、股本口径及拒绝错误数字列。"""
from pathlib import Path
import json

import pytest

from research.forward_eps_soochow_latest_share_layout_v4 import parse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_forward_eps_soochow_originals_v1"


def original(rid="AP202301031581650009"):
    record = json.loads((SOURCE / "document_records" / (rid + ".json")).read_text(encoding="utf-8"))
    pages = json.loads((SOURCE / "page_texts" / (rid + ".json")).read_text(encoding="utf-8"))["pages"]
    return pages, record["provider_metadata"], record["directory_record"]


@pytest.mark.parametrize("rid,eps,profit,pe", [
    ("AP202301031581650009", ["1.39", "1.73", "2.15"], ["2382", "2951", "3671"], ["17.75", "14.33", "11.52"]),
    ("AP202301041581678346", ["1.41", "1.95", "2.56"], ["611", "845", "1108"], ["57.54", "41.63", "31.75"]),
    ("AP202301061581758608", ["2.52", "5.06", "7.60"], ["4030", "8082", "12140"], ["23.39", "11.66", "7.76"]),
])
def test_three_original_reports(rid, eps, profit, pe):
    pages, metadata, row = original(rid)
    facts = parse(pages, metadata, row)["facts"]
    assert [f["target_fiscal_year"] for f in facts] == [2022, 2023, 2024]
    for field, expected in [("eps", eps), ("net_profit", profit), ("pe", pe)]:
        assert [f[field + "_value_exact"] for f in facts] == expected
    assert all(f["latest_diluted_basis_explicit"] for f in facts)
    assert all(f["conservative_information_date"] >= str(metadata["eitime"])[:10] for f in facts)
    assert all(not f["historical_immutable_snapshot_proven"] for f in facts)


@pytest.mark.parametrize("replacement", ["1.09 1.39 1.73", "1.09 1.39 1.73 2.15 3.00"])
def test_incomplete_or_extra_columns_rejected(replacement):
    pages, metadata, row = original()
    pages[0] = pages[0].replace("1.09 1.39 1.73 2.15", replacement)
    with pytest.raises(ValueError):
        parse(pages, metadata, row)


@pytest.mark.parametrize("replacement", ["2021A 2022A 2023A 2024A", "2021A 2022E 2025E 2024E"])
def test_actual_and_discontinuous_years_rejected(replacement):
    pages, metadata, row = original()
    pages[0] = pages[0].replace("2021A 2022E 2023E 2024E", replacement)
    with pytest.raises(ValueError):
        parse(pages, metadata, row)


def test_ambiguous_source_or_wrong_identity_rejected():
    pages, metadata, row = original()
    with pytest.raises(ValueError):
        parse([pages[0], pages[0]], metadata, row)
    metadata["security"] = [{"stock": "600519"}]
    with pytest.raises(ValueError):
        parse(pages, metadata, row)


def test_negative_eps_and_missing_optional_pe_preserved():
    pages, metadata, row = original()
    pages[0] = pages[0].replace("1.09 1.39 1.73 2.15", "1.09 -1.39 (1.73) 0.00")
    pages[0] = pages[0].replace("P/E（现价&最新股本摊薄）", "P/E（目标价）")
    facts = parse(pages, metadata, row)["facts"]
    assert [f["eps_value_exact"] for f in facts] == ["-1.39", "-1.73", "0.00"]
    assert all(f["pe_value_exact"] is None for f in facts)
