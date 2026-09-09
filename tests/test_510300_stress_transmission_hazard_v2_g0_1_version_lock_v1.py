from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.stress_transmission_hazard_v2_g0_1_version_lock_v1 import (
    VersionLockError,
    file_evidence,
    frame_semantic_sha256,
    materialize_data_files,
    merge_expected_contract,
    normalize_relative_path,
    parquet_evidence,
    project_path,
    resolve_git_scope,
    validate_expected_replay_metrics,
)


def _expected_replay() -> dict[str, object]:
    return {
        "b2_identifiable_event_count": 31,
        "b2_eligible_non_event_risk_day_count": 1102,
        "b2_common_sample_day_count": 1279,
        "b2_view_day_count": 1279,
        "event_era_distribution": {
            "2015-2017": {
                "total_events": 10,
                "b2_identifiable_events": 0,
                "b3_identifiable_events": 0,
            },
            "2018-2020": {
                "total_events": 15,
                "b2_identifiable_events": 5,
                "b3_identifiable_events": 5,
            },
            "2021-2023": {
                "total_events": 22,
                "b2_identifiable_events": 19,
                "b3_identifiable_events": 19,
            },
            "2024-2026": {
                "total_events": 11,
                "b2_identifiable_events": 7,
                "b3_identifiable_events": 7,
            },
        },
    }


def _actual_metrics() -> dict[str, object]:
    expected = _expected_replay()
    return {
        "g1": {
            "b2_identifiable_event_count": 31,
            "b2_eligible_non_event_risk_day_count": 1102,
            "b2_common_sample_day_count": 1279,
        },
        "mft": {"b2_view_day_count": 1279},
        "event_era_distribution": expected["event_era_distribution"],
    }


def test_normalize_relative_path_accepts_windows_separator_and_blocks_escape(
    tmp_path: Path,
) -> None:
    assert normalize_relative_path(r"data\curated\sample.parquet") == (
        "data/curated/sample.parquet"
    )
    assert project_path(tmp_path, r"data\sample.csv") == (
        tmp_path / "data" / "sample.csv"
    ).resolve()
    for invalid in ("", "..", "../outside", r"C:\outside\file.csv"):
        with pytest.raises(VersionLockError):
            normalize_relative_path(invalid)


def test_project_path_and_file_evidence_do_not_resolve_junctions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "data" / "sample.csv"
    path.parent.mkdir()
    path.write_bytes(b"value\n1\n")

    def forbidden_resolve(*args: object, **kwargs: object) -> Path:
        raise AssertionError("项目相对身份不得解引用 Junction")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)

    lexical_path = project_path(tmp_path, "data/sample.csv")
    evidence = file_evidence(lexical_path, root=tmp_path)
    assert evidence["path"] == "data/sample.csv"


def test_parquet_evidence_records_bytes_rows_columns_and_semantics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data" / "sample.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "bad10": [0, 1],
            "event_id": pd.Series([pd.NA, "E001"], dtype="string"),
        }
    )
    frame.to_parquet(path, index=False)

    evidence = parquet_evidence(path, root=tmp_path)

    assert evidence["path"] == "data/sample.parquet"
    assert evidence["bytes"] == path.stat().st_size
    assert evidence["row_count"] == 2
    assert evidence["column_count"] == 3
    assert evidence["columns"] == ["origin_date", "bad10", "event_id"]
    assert evidence["persisted_semantic_sha256"] == frame_semantic_sha256(
        pd.read_parquet(path)
    )


def test_merge_expected_contract_deduplicates_sources_and_rejects_conflict() -> None:
    contracts: dict[str, dict[str, object]] = {}
    first = {"path": r"data\sample.csv", "bytes": 3, "sha256": "a" * 64}
    merge_expected_contract(contracts, first, source="配置甲")
    merge_expected_contract(contracts, first, source="配置乙")

    assert contracts["data/sample.csv"]["contract_sources"] == ["配置乙", "配置甲"]

    with pytest.raises(VersionLockError, match="冲突身份"):
        merge_expected_contract(
            contracts,
            {"path": "data/sample.csv", "bytes": 4, "sha256": "b" * 64},
            source="冲突配置",
        )


def test_resolve_git_scope_uses_name_rule_suffix_and_explicit_file(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    matched = config_dir / "510300_stress_transmission_hazard_v2.yaml"
    ignored_name = config_dir / "unrelated.yaml"
    ignored_suffix = config_dir / "stress_transmission_hazard_v2.parquet"
    explicit = tmp_path / ".gitattributes"
    matched.write_text("version: 1\n", encoding="utf-8")
    ignored_name.write_text("version: 1\n", encoding="utf-8")
    ignored_suffix.write_bytes(b"parquet")
    explicit.write_text("* -text\n", encoding="utf-8")

    paths = resolve_git_scope(
        tmp_path,
        roots=["config"],
        filename_regex=r"(?i)stress_transmission_hazard_v2",
        allowed_suffixes=[".yaml"],
        explicit_paths=[".gitattributes"],
    )

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        ".gitattributes",
        "config/510300_stress_transmission_hazard_v2.yaml",
    ]


def test_materialize_data_files_is_copy_or_exact_reuse_never_overwrite(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source = source_root / "data" / "raw" / "sample.csv"
    source.parent.mkdir(parents=True)
    target_root.mkdir()
    source.write_bytes(b"a,b\n1,2\n")
    contract = file_evidence(source, root=source_root)
    manifest = {"data_files": [contract]}

    first = materialize_data_files(
        source_root=source_root,
        target_root=target_root,
        manifest=manifest,
    )
    second = materialize_data_files(
        source_root=source_root,
        target_root=target_root,
        manifest=manifest,
    )

    assert first == {
        "copied_file_count": 1,
        "reused_file_count": 0,
        "copied_bytes": len(b"a,b\n1,2\n"),
    }
    assert second == {
        "copied_file_count": 0,
        "reused_file_count": 1,
        "copied_bytes": 0,
    }

    target = target_root / "data" / "raw" / "sample.csv"
    target.write_bytes(b"different")
    with pytest.raises(VersionLockError, match="禁止覆盖"):
        materialize_data_files(
            source_root=source_root,
            target_root=target_root,
            manifest=manifest,
        )


def test_validate_expected_replay_metrics_accepts_exact_contract() -> None:
    validate_expected_replay_metrics(_actual_metrics(), _expected_replay())


def test_validate_expected_replay_metrics_rejects_scalar_and_era_drift() -> None:
    scalar_drift = _actual_metrics()
    scalar_drift["g1"] = dict(scalar_drift["g1"])
    scalar_drift["g1"]["b2_eligible_non_event_risk_day_count"] = 1101
    with pytest.raises(VersionLockError, match="关键指标漂移"):
        validate_expected_replay_metrics(scalar_drift, _expected_replay())

    era_drift = _actual_metrics()
    era_drift["event_era_distribution"] = dict(
        era_drift["event_era_distribution"]
    )
    era_drift["event_era_distribution"]["2024-2026"] = {
        "total_events": 11,
        "b2_identifiable_events": 6,
        "b3_identifiable_events": 6,
    }
    with pytest.raises(VersionLockError, match="年代事件分布漂移"):
        validate_expected_replay_metrics(era_drift, _expected_replay())
