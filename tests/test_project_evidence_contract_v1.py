from __future__ import annotations

import ast
import json
import subprocess
from collections import deque
from pathlib import Path

import pytest

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    canonical_sha256,
    create_exclusive_claim,
    ensure_paths_committed_at_head,
    resolve_registered_input,
    sha256_bytes,
    strict_json_text,
    validate_authority_state,
    verify_manifest_files,
)


ROOT = Path(__file__).resolve().parents[1]


def _authority() -> dict[str, object]:
    return {
        "research_state": "DISCOVERY_ONLY",
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "actual_holdings_state": "UNKNOWN_OUT_OF_SCOPE",
        "position_impact": 0,
        "abstain_implies_cash_target": False,
        "paper_signal_authorized": False,
        "shadow_signal_authorized": False,
        "broker_connection_authorized": False,
        "live_trading_authorized": False,
    }


def test_authority_keeps_research_model_order_and_holdings_separate() -> None:
    validate_authority_state(_authority())
    invalid = _authority()
    invalid["current_holding_route"] = "CASH_CNY"
    with pytest.raises(EvidenceContractError, match="被禁止字段"):
        validate_authority_state(invalid)
    invalid = _authority()
    invalid["model_position_target"] = "CASH_CNY"
    with pytest.raises(EvidenceContractError, match="越过研究边界"):
        validate_authority_state(invalid)


def test_registered_input_accepts_explicit_physical_root_and_rejects_escape(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    source = data_root / "raw" / "sample.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen-input")
    root_contract = {
        "logical_path": "data",
        "physical_root_id": "TEST_DATA_ROOT_V1",
        "expected_resolved_root": str(data_root),
    }
    input_contract = {
        "logical_path": "data/raw/sample.bin",
        "physical_root_id": "TEST_DATA_ROOT_V1",
        "content_sha256": sha256_bytes(b"frozen-input"),
        "bytes": len(b"frozen-input"),
        "data_contract_version": "TEST_BINARY_V1",
    }
    evidence = resolve_registered_input(
        project_root=tmp_path,
        input_id="sample",
        root_contract=root_contract,
        input_contract=input_contract,
    )
    assert evidence.physical_root_id == "TEST_DATA_ROOT_V1"
    assert evidence.logical_path == "data/raw/sample.bin"

    escaped = dict(input_contract)
    escaped["logical_path"] = "data/../outside.bin"
    with pytest.raises(EvidenceContractError, match="路径段"):
        resolve_registered_input(
            project_root=tmp_path,
            input_id="escaped",
            root_contract=root_contract,
            input_contract=escaped,
        )


def test_strict_json_rejects_nonfinite_values() -> None:
    with pytest.raises(EvidenceContractError, match="NaN"):
        strict_json_text({"value": float("nan")})


def test_manifest_detects_implementation_drift(tmp_path: Path) -> None:
    implementation = tmp_path / "research" / "module.py"
    governance = tmp_path / "config" / "contract.json"
    implementation.parent.mkdir(parents=True)
    governance.parent.mkdir(parents=True)
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    governance.write_text("{}\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "implementation_files": {
            "research/module.py": sha256_bytes(implementation.read_bytes())
        },
        "governance_files": {
            "config/contract.json": sha256_bytes(governance.read_bytes())
        },
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    verify_manifest_files(manifest, project_root=tmp_path)
    implementation.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="漂移"):
        verify_manifest_files(manifest, project_root=tmp_path)


def test_committed_head_bytes_must_match_worktree(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "测试用户"], cwd=tmp_path, check=True
    )
    tracked = tmp_path / "contract.txt"
    tracked.write_text("冻结字节\n", encoding="utf-8")
    subprocess.run(["git", "add", "contract.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "test: freeze contract"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    evidence = ensure_paths_committed_at_head(tmp_path, ["contract.txt"])
    assert evidence["contract.txt"]["bytes"] > 0
    assert len(evidence["contract.txt"]["head_blob_sha256"]) == 64
    assert len(evidence["contract.txt"]["working_sha256"]) == 64
    tracked.write_text("漂移字节\n", encoding="utf-8")
    with pytest.raises(EvidenceContractError, match="漂移"):
        ensure_paths_committed_at_head(tmp_path, ["contract.txt"])


def test_exclusive_claim_cannot_be_reused(tmp_path: Path) -> None:
    claim = tmp_path / "claim.json"
    create_exclusive_claim(claim, {"claim_id": "ONE_SHOT"})
    assert json.loads(claim.read_text(encoding="utf-8"))["claim_id"] == "ONE_SHOT"
    with pytest.raises(EvidenceContractError, match="已存在"):
        create_exclusive_claim(claim, {"claim_id": "SECOND_ATTEMPT"})


def _internal_import_paths(path: str) -> set[str]:
    source = (ROOT / path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=path)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    resolved: set[str] = set()
    for module in modules:
        stem = module.replace(".", "/")
        for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
            if (ROOT / candidate).is_file():
                resolved.add(candidate)
                break
    return resolved


def test_tracked_python_dependency_closure_contains_no_untracked_module() -> None:
    tracked = {
        line.strip().replace("\\", "/")
        for line in subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True, encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    seeds = sorted(path for path in tracked if path.endswith(".py"))
    queue = deque(seeds)
    seen = set(seeds)
    missing: dict[str, set[str]] = {}
    while queue:
        source = queue.popleft()
        for target in _internal_import_paths(source):
            if target not in tracked:
                missing.setdefault(target, set()).add(source)
            if target not in seen:
                seen.add(target)
                queue.append(target)
    assert not missing, {
        target: sorted(importers) for target, importers in sorted(missing.items())
    }
