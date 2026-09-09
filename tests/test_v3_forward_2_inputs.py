from __future__ import annotations

import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import research.v3_forward_validation as base
from research.v3_forward_2_validation import CONFIG_FILE, activated_engine
from scripts.refresh_v3_forward_2_inputs import (
    append_financial_revisions,
    normalize_csindex_membership,
    normalize_csindex_weights,
)


NOW = datetime(2026, 8, 13, 18, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def _weights(count: int = 300) -> pd.DataFrame:
    rows = []
    for index in range(count):
        code = f"{index + 1:06d}"
        rows.append(
            {
                "日期": "2026-07-31",
                "指数代码": "000300",
                "成分券代码": code,
                "交易所": "上海证券交易所" if index >= 150 else "深圳证券交易所",
                "权重": 100.0 / count,
            }
        )
    return pd.DataFrame(rows)


def _membership(count: int = 300, date: str = "2026-08-13") -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "日期": date,
                "指数代码": "000300",
                "成分券代码": f"{index + 1:06d}",
                "交易所": "上海证券交易所" if index >= 150 else "深圳证券交易所",
            }
        )
    return pd.DataFrame(rows)


def test_csindex_weight_keeps_official_and_observed_dates_separate() -> None:
    canonical, snapshot = normalize_csindex_weights(
        _weights(), pd.Timestamp("2026-08-13"), NOW
    )
    assert canonical["trade_date"].nunique() == 1
    assert canonical["trade_date"].iloc[0] == pd.Timestamp("2026-07-31")
    assert snapshot["observed_date"].iloc[0] == pd.Timestamp("2026-08-13")
    assert canonical["con_code"].nunique() == 300
    assert canonical["weight"].sum() == pytest.approx(100.0)


def test_csindex_membership_must_be_observed_today() -> None:
    result = normalize_csindex_membership(
        _membership(), pd.Timestamp("2026-08-13"), NOW
    )
    assert result["con_code"].nunique() == 300
    with pytest.raises(ValueError, match="不是目标日"):
        normalize_csindex_membership(
            _membership(date="2026-08-12"), pd.Timestamp("2026-08-13"), NOW
        )


def _candidate(value: float, announcement: str = "2026-04-30") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "con_code": "600000.SH",
                "report_period": pd.Timestamp("2026-03-31"),
                "announcement_date": pd.Timestamp(announcement),
                "income_announcement_date": pd.Timestamp(announcement),
                "balance_announcement_date": pd.Timestamp(announcement),
                "revenue_cny": 100.0,
                "net_profit_parent_cny": value,
                "equity_parent_cny": 200.0,
                "total_shares": 10.0,
                "source": "eastmoney-test",
            }
        ]
    )


def test_financial_revision_uses_first_seen_date_and_is_idempotent() -> None:
    first, appended = append_financial_revisions(
        pd.DataFrame(), _candidate(10.0), pd.Timestamp("2026-08-13"), NOW
    )
    assert appended == 1
    assert first.loc[0, "available_at"] == pd.Timestamp("2026-08-13")
    assert not bool(first.loc[0, "revision_detected"])

    same, appended = append_financial_revisions(
        first, _candidate(10.0), pd.Timestamp("2026-08-14"), NOW
    )
    assert appended == 0
    assert len(same) == 1

    revised, appended = append_financial_revisions(
        first, _candidate(11.0), pd.Timestamp("2026-08-14"), NOW
    )
    assert appended == 1
    assert len(revised) == 2
    assert revised.iloc[-1]["available_at"] == pd.Timestamp("2026-08-14")
    assert bool(revised.iloc[-1]["revision_detected"])


def test_v2_engine_binding_restores_v1_globals() -> None:
    original = base.CONFIG_FILE
    with activated_engine():
        assert base.CONFIG_FILE == CONFIG_FILE
    assert base.CONFIG_FILE == original


def test_manifest_validation_fails_closed_after_frozen_bytes_change(
    tmp_path, monkeypatch
) -> None:
    frozen_file = tmp_path / "frozen.txt"
    frozen_file.write_text("冻结内容", encoding="utf-8")
    expected = hashlib.sha256(frozen_file.read_bytes()).hexdigest()
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({"frozen_files": {"frozen.txt": expected}}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(base, "ROOT", tmp_path)

    assert base.validate_frozen_manifest(manifest_file)["frozen_files"] == {
        "frozen.txt": expected
    }
    frozen_file.write_text("已改变的内容", encoding="utf-8")

    with pytest.raises(ValueError, match="冻结文件指纹不一致：frozen.txt"):
        base.validate_frozen_manifest(manifest_file)


def test_manifest_failure_happens_before_any_forward_output(
    tmp_path, monkeypatch
) -> None:
    output_paths = {
        "SIGNAL_LOG": tmp_path / "signal.parquet",
        "OUTCOME_LOG": tmp_path / "outcome.parquet",
        "SHADOW_LOG": tmp_path / "shadow.parquet",
        "STATUS_FILE": tmp_path / "status.json",
        "REPORT_DIR": tmp_path / "reports",
    }
    for name, path in output_paths.items():
        monkeypatch.setattr(base, name, path)

    def reject_manifest() -> dict[str, object]:
        raise ValueError("冻结文件指纹不一致：fixture；必须创建新模型版本")

    monkeypatch.setattr(base, "validate_frozen_manifest", reject_manifest)

    with pytest.raises(ValueError, match="冻结文件指纹不一致：fixture"):
        base.run_forward_cycle(
            signal_date=pd.Timestamp("2026-08-13"),
            generated_at=NOW,
            enforce_same_local_date=False,
        )

    assert all(not path.exists() for path in output_paths.values())
