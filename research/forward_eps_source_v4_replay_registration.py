"""登记仅改变原件字段来源的重放，不重新选择模型、参数或进出场规则。"""
from pathlib import Path

from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "two": ("510300_forward_eps_two_institution_features_v3", "510300_forward_eps_two_institution_features_v2"),
    "pe": ("510300_forward_eps_valuation_consistency_features_v2", "510300_forward_eps_valuation_consistency_features_v1"),
}
POLICIES = {
    "two": ("510300_forward_eps_two_institution_policy_v3", "510300_forward_eps_two_institution_policy_v2"),
    "pe": ("510300_forward_eps_valuation_consistency_policy_v2", "510300_forward_eps_valuation_consistency_policy_v1"),
}
DOC = ROOT / "docs/510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY.md"


def files_with_ancestors(old_manifest: dict, extra: list[Path]) -> list[dict]:
    paths = {str(ROOT / row["path"]): ROOT / row["path"] for row in old_manifest["files"]}
    for path in extra:
        paths[str(path)] = path
    return [identity(path) for path in paths.values()]


def freeze_feature(kind: str, script: str):
    new, old = FEATURES[kind]
    out = ROOT / "reports/research" / new
    parent = ROOT / "reports/research" / old
    out.mkdir(parents=True, exist_ok=False)
    old_manifest = read(parent / "manifest.json")
    extra = [Path(script), Path(__file__), DOC, parent / "result.json", parent / "manifest.json",
             ROOT / "research/forward_eps_soochow_facts_v4.py",
             ROOT / "research/forward_eps_soochow_latest_share_layout_v4.py",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v4/result.json",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v4/manifest.json",
             ROOT / "reports/research/510300_forward_eps_soochow_facts_v4/saved_source_verification.json",
             ROOT / "tests/test_forward_eps_soochow_latest_share_layout_v4.py",
             ROOT / "research/forward_eps_two_institution_features_v3.py",
             ROOT / "tests/test_forward_eps_two_institution_features_v3.py",
             ROOT / "tests/test_forward_eps_valuation_consistency_features_v2.py"]
    if kind == "pe":
        extra.append(ROOT / "reports/research/510300_forward_eps_two_institution_features_v3/manifest.json")
    save(out / "manifest.json", {"registered_at": now(), "source_replay_of_completed_binding": old,
         "soochow_facts_version": 4, "prior_results_observed": True, "new_source_account_returns_observed": False,
         "feature_definition_aggregation_coverage_rules_unchanged": True,
         "new_distinct_methods": 0, "files": files_with_ancestors(old_manifest, extra)}, exclusive=True)
    print("来源重放因子已登记：", new, flush=True)


def freeze_policy(kind: str, script: str, models: dict):
    new, old = POLICIES[kind]
    config_path = ROOT / "config" / (new + ".json")
    parent_config = ROOT / "config" / (old + ".json")
    old_manifest = read(ROOT / "config" / (old + "_manifest.json"))
    original = read(parent_config)
    config = dict(original)
    config.update({"study_id": new.upper(), "source_layout_correction_only": True,
                   "soochow_source_fact_version": 4, "source_replay_of_completed_binding": old.upper(),
                   "new_source_originals_in_progress_at_registration": False,
                   "prior_account_results_already_observed": True})
    if "round_19_results_observed_at_registration" in config:
        config["round_19_results_observed_at_registration"] = True
    metadata_keys = {"study_id", "source_layout_correction_only", "soochow_source_fact_version",
                     "source_replay_of_completed_binding", "new_source_originals_in_progress_at_registration",
                     "prior_account_results_already_observed", "round_19_results_observed_at_registration"}
    assert {k: v for k, v in original.items() if k not in metadata_keys} == {k: v for k, v in config.items() if k not in metadata_keys}
    assert config["models"] == models
    old_out = ROOT / "reports/research" / old
    check = read(old_out / "saved_numerical_verification.json")
    assert check["complete_accounts_checked"] == 10
    feature = FEATURES[kind][0]
    verifier = ("verify_forward_eps_two_institution_source_v4_saved_20260907.py" if kind == "two"
                else "verify_forward_eps_valuation_consistency_source_v4_saved_20260907.py")
    extra = [Path(script), Path(__file__), DOC, parent_config, old_out / "result.json",
             old_out / "saved_numerical_verification.json", ROOT / "research" / (feature.removeprefix("510300_") + ".py"),
             ROOT / "reports/research" / feature / "manifest.json", ROOT / "scripts" / verifier,
             ROOT / "config/510300_research_authority_v6.json"]
    save(config_path, config, exclusive=True)
    save(ROOT / "config" / (new + "_manifest.json"), {"registered_at": now(),
         "new_candidate_returns_read_before_freeze": False, "prior_account_results_already_observed": True,
         "source_replay_of_completed_binding": old, "all_non_source_rules_unchanged": True,
         "new_distinct_methods": 0, "existing_methods_with_new_source_version": 3,
         "planned_new_accounts": 6, "planned_evaluation_accounts": 10, "goal_achieved": False,
         "files": files_with_ancestors(old_manifest, [config_path, *extra])}, exclusive=True)
    print("同方法新来源账户已登记：", new, flush=True)
