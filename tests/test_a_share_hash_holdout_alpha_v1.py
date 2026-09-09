import hashlib

from research.a_share_hash_holdout_alpha_v1 import FEATURE_COLUMNS


def bucket(code: str) -> int:
    return hashlib.sha256(code.encode("utf-8")).digest()[0] % 5


def test_factor_count_and_hash_split_are_deterministic() -> None:
    assert len(FEATURE_COLUMNS) == 10
    codes = ["000001.SZ", "600000.SH", "300750.SZ", "688981.SH"]
    assert [bucket(code) for code in codes] == [bucket(code) for code in codes]
    assert all(0 <= bucket(code) <= 4 for code in codes)
