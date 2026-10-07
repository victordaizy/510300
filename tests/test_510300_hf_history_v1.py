"""验证远程行组缓存保留原值并在超限前停止读取。"""

import io
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.acquire_510300_hf_history_v1 import CachedRangeFile, extract_selected
from scripts.acquire_510300_hf_l2_subset_v1 import candidate_row_groups


class RangeResponse:
    def __init__(self, body, start, end):
        self.status_code = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(body)}"}
        self.payload = body[start:end + 1]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def iter_content(self, size):
        for offset in range(0, len(self.payload), size):
            yield self.payload[offset:offset + size]


class RangeSession:
    def __init__(self, body):
        self.body, self.requests = body, []

    def get(self, url, headers, **kwargs):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        self.requests.append((start, end))
        return RangeResponse(self.body, start, end)


def fixture():
    rows = 8192
    columns = {"wind_code": ["510300.SZ"] * 4096 + ["600000.SH"] * 4096,
               "date": [20260930] * rows, "time": list(range(rows))}
    columns.update({f"value_{index}": [float(value + index) for value in range(rows)] for index in range(12)})
    table = pa.table(columns)
    output = io.BytesIO()
    pq.write_table(table, output, row_group_size=4096, compression=None, use_dictionary=False)
    return table, output.getvalue()


def test_cached_group_preserves_all_values_and_uses_one_data_request():
    original, content = fixture()
    session = RangeSession(content)
    config = {"symbol": "510300", "maximum_decoded_row_group_bytes": 8 * 1024 ** 2,
              "maximum_cached_row_group_bytes": 8 * 1024 ** 2}
    with CachedRangeFile("https://example.test/source", {"remaining": 8 * 1024 ** 2}, session=session) as remote:
        parquet = pq.ParquetFile(remote, pre_buffer=False, buffer_size=0)
        before = len(session.requests)
        result = extract_selected(parquet, remote, candidate_row_groups(parquet, ["510300"]), config)
        assert result.equals(original.slice(0, 4096))
        assert len(session.requests) - before == 1
        assert result.column_names == original.column_names


def test_cache_limit_rejects_before_network_request():
    _, content = fixture()
    session = RangeSession(content)
    with CachedRangeFile("https://example.test/source", {"remaining": 8 * 1024 ** 2}, session=session) as remote:
        parquet = pq.ParquetFile(remote, pre_buffer=False, buffer_size=0)
        before = len(session.requests)
        with pytest.raises(ValueError, match="内存缓存上限"):
            remote.cache_group(parquet.metadata.row_group(0), 1)
        assert len(session.requests) == before
