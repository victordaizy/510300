"""验证真实Parquet行组筛选与下载边界，防止按错证券或意外整文件读取。"""

import io
import json
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.acquire_510300_hf_l2_subset_v1 import RangeFile, candidate_row_groups, extract_table, prior_network_usage


class Response:
    def __init__(self, body, start, end, status=206, wrong_range=False, truncate=False):
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start + int(wrong_range)}-{end}/{len(body)}"}
        self.payload = body[start:end + 1]
        if truncate:
            self.payload = self.payload[:-1]
        self.read_started = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, size):
        self.read_started = True
        for start in range(0, len(self.payload), size):
            yield self.payload[start:start + size]


class Session:
    def __init__(self, body, **options):
        self.body, self.options, self.responses = body, options, []

    def get(self, url, headers, **kwargs):
        start, end = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups())
        response = Response(self.body, start, end, **self.options)
        self.responses.append(response)
        return response


def parquet_bytes(statistics=True):
    table = pa.table({"wind_code": ["159919.SZ", "510300.SZ", "510300.sh", "5103000.SZ", "510330.SH"],
                      "date": [20260812] * 5, "time": [150000000] * 5,
                      "price": [100, 200, 300, 400, 500]})
    sink = io.BytesIO()
    pq.write_table(table, sink, row_group_size=1, write_statistics=statistics)
    return sink.getvalue()


def test_remote_row_groups_preserve_raw_suffix_and_exclude_other_instruments():
    body = parquet_bytes()
    with RangeFile("https://example.test/file", {"remaining": 100000}, session=Session(body)) as remote:
        parquet = pq.ParquetFile(remote, pre_buffer=False)
        groups = candidate_row_groups(parquet, ["510300"])
        result = extract_table(parquet, groups, ["510300"])
        assert result["wind_code"].to_pylist() == ["510300.SZ", "510300.sh"]
        assert result["price"].to_pylist() == [200, 300]
        assert len(groups) < parquet.num_row_groups


def test_missing_statistics_keeps_groups_then_filters_exact_security():
    parquet = pq.ParquetFile(io.BytesIO(parquet_bytes(False)))
    groups = candidate_row_groups(parquet, ["510300"])
    assert len(groups) == 5
    assert extract_table(parquet, groups, ["510300"]).num_rows == 2


@pytest.mark.parametrize("status", [200, 401, 403])
def test_non_range_or_unauthorized_response_is_stopped_before_body(status):
    session = Session(parquet_bytes(), status=status)
    with pytest.raises(ValueError, match=f"HTTP {status}"):
        RangeFile("https://example.test/file", {"remaining": 100000}, session=session)
    assert not session.responses[0].read_started


@pytest.mark.parametrize("options", [{"wrong_range": True}, {"truncate": True}])
def test_mismatched_or_truncated_range_is_not_accepted(options):
    with pytest.raises(ValueError):
        RangeFile("https://example.test/file", {"remaining": 100000}, session=Session(parquet_bytes(), **options))


def test_budget_stops_before_next_request():
    session = Session(parquet_bytes())
    with RangeFile("https://example.test/file", {"remaining": 4}, session=session) as remote:
        with pytest.raises(ValueError, match="上限"):
            remote.read(1)
        assert len(session.responses) == 1


def test_absent_security_remains_missing():
    parquet = pq.ParquetFile(io.BytesIO(parquet_bytes()))
    with pytest.raises(ValueError, match="没有目标证券"):
        extract_table(parquet, candidate_row_groups(parquet, ["510999"]), ["510999"])


def test_previous_date_reads_reduce_next_request_budget(tmp_path):
    for name, value in (("day_one", 30), ("day_two", 66)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "receipt.json").write_text(json.dumps({"network_body_bytes": value}), encoding="utf-8")
    session = Session(parquet_bytes())
    with RangeFile("https://example.test/file", {"remaining": 100 - prior_network_usage(tmp_path)}, session=session) as remote:
        with pytest.raises(ValueError, match="上限"):
            remote.read(1)
    assert len(session.responses) == 1


def test_unfinished_previous_run_does_not_silently_restore_budget(tmp_path):
    folder = tmp_path / "unfinished"
    folder.mkdir()
    (folder / "receipt.json").write_text(json.dumps({"status": "STARTED"}), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少确定的网络用量"):
        prior_network_usage(tmp_path)
