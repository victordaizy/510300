"""重现混合时间字段的Arrow失败，验证统一类型后时点与资格不改变。"""
import pandas as pd
import pyarrow as pa
import pytest

from research.official_industry_full_sequence_parse_v1_0_1 import normalized_metadata


def test_mixed_old_json_and_new_timestamp_metadata_store_without_changing_publication_instants_or_eligibility():
    raw = pd.DataFrame({"available_at": ["2018-11-02T23:59:00+08:00", pd.Timestamp("2024-09-30 23:59", tz="Asia/Shanghai")],
        "complete_snapshot_passed": [False, True], "row_source_snapshot_eligible": [True, True]})
    with pytest.raises(pa.ArrowTypeError):
        pa.Table.from_pandas(raw, preserve_index=False)
    fixed = normalized_metadata(raw)
    stored = pa.Table.from_pandas(fixed, preserve_index=False).to_pandas()
    assert stored.available_at.iloc[0] == pd.Timestamp(raw.available_at.iloc[0])
    assert stored.available_at.iloc[1] == raw.available_at.iloc[1]
    assert str(stored.available_at.dtype) == "datetime64[ns, Asia/Shanghai]"
    assert stored.complete_snapshot_passed.tolist() == [False, True]
    assert stored.row_source_snapshot_eligible.tolist() == [True, True]
