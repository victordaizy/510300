"""冻结板块评分卡 V1.3 微信公众号前瞻证据层。"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.industry_sector_scorecard_v1_3_wechat import (  # noqa: E402
    validate_wechat_snapshot,
)


CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1_3_wechat_forward.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_map(paths: list[Path], *, relative_to_root: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        key = path.relative_to(ROOT).as_posix() if relative_to_root else path.as_posix()
        result[key] = sha256(path)
    return result


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    selected_articles, corpus_audit = validate_wechat_snapshot(config)

    protocol_files = [
        CONFIG_FILE,
        ROOT / "docs" / "INDUSTRY_SECTOR_SCORECARD_V1_3_WECHAT_FORWARD_SPEC.md",
        ROOT / "research" / "industry_sector_scorecard_v1_3_wechat.py",
        ROOT / "scripts" / "build_industry_sector_scorecard_v1_3_wechat.py",
        ROOT / "scripts" / "freeze_industry_sector_scorecard_v1_3_wechat.py",
        ROOT / "tests" / "test_industry_sector_scorecard_v1_3_wechat.py",
    ]
    parent_input_files = [ROOT / value for value in config["parents"].values()]
    snapshot_root = Path(config["wechat_snapshot"]["root"])
    external_snapshot_files = [
        snapshot_root / config["wechat_snapshot"]["audit_json"],
        snapshot_root / config["wechat_snapshot"]["article_csv"],
        snapshot_root / config["wechat_snapshot"]["source_config"],
    ] + [Path(value) for value in selected_articles["markdown_path"].tolist()]
    output_files = [
        ROOT / value
        for key, value in config["outputs"].items()
        if key != "manifest"
    ]
    local_files = protocol_files + parent_input_files + output_files
    missing = [path.as_posix() for path in local_files + external_snapshot_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V1.3冻结缺少文件：{missing}")

    parent_manifest = ROOT / config["parents"]["v1_2_manifest"]
    expected_parent_hash = config["parent_integrity"][
        "industry_sector_scorecard_v1_2_manifest_sha256"
    ]
    actual_parent_hash = sha256(parent_manifest)
    if actual_parent_hash != expected_parent_hash:
        raise RuntimeError(
            f"V1.2父清单已变化：{actual_parent_hash} != {expected_parent_hash}"
        )

    report_path = ROOT / config["outputs"]["json"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["parent_files_changed"] is not False:
        raise RuntimeError("V1.3没有保持V1.2父文件不变")
    if report["corpus_audit"]["status"] != "PARTIAL_SUCCESS":
        raise RuntimeError("V1.3没有显式保留公众号语料的不完整状态")
    if report["corpus_audit"]["archive_partial_count"] != 22:
        raise RuntimeError("V1.3公众号不完整文章计数变化")
    if report["corpus_audit"]["selected_article_count"] != 15:
        raise RuntimeError("V1.3入选文章数量变化")
    if report["corpus_audit"]["raw_article_count_used_as_vote"] is not False:
        raise RuntimeError("V1.3违规把文章数量当投票")
    if (
        report["market_forward_context"]["national_team_holdings_state"]
        != "NO_VIEW_NO_AUDITABLE_POSITION_LEVEL_DATA"
    ):
        raise RuntimeError("V1.3违规代理推断国家队持仓")
    if report["synthetic_numeric_score"] is not None:
        raise RuntimeError("V1.3违规生成合成数值分数")
    if report["wechat_direction_is_probability"] is not False:
        raise RuntimeError("V1.3违规把公众号方向当作概率")
    if report["score_conditioned_win_rate"] != "UNAVAILABLE_AWAITING_TRUE_FORWARD":
        raise RuntimeError("V1.3违规生成高分条件胜率")
    if report["may_call_predictive_edge"] is not False:
        raise RuntimeError("V1.3违规宣称预测 Edge")
    if report["position_mapping_enabled"] is not False:
        raise RuntimeError("V1.3违规启用仓位映射")

    manifest = {
        "version": config["version"],
        "status": "FROZEN_RESEARCH_ONLY_WECHAT_FORWARD",
        "as_of_date": config["as_of_date"],
        "information_cutoff": config["information_cutoff"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_files": _hash_map(protocol_files, relative_to_root=True),
        "parent_input_files": _hash_map(parent_input_files, relative_to_root=True),
        "external_snapshot_files": _hash_map(
            external_snapshot_files, relative_to_root=False
        ),
        "output_files": _hash_map(output_files, relative_to_root=True),
        "parent_manifest_integrity": {
            parent_manifest.relative_to(ROOT).as_posix(): {
                "expected": expected_parent_hash,
                "actual": actual_parent_hash,
            }
        },
        "corpus_invariants": {
            "status": corpus_audit["status"],
            "article_count": corpus_audit["article_count"],
            "archive_complete_count": corpus_audit["archive_complete_count"],
            "archive_partial_count": corpus_audit["archive_partial_count"],
            "selected_article_count": int(len(selected_articles)),
            "selected_source_count": int(selected_articles["source_name"].nunique()),
            "raw_article_count_used_as_vote": False,
        },
        "invariants": {
            "wechat_direction_is_probability": False,
            "synthetic_numeric_score_allowed": False,
            "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
            "may_call_predictive_edge": False,
            "national_team_holdings_state": "NO_VIEW_NO_AUDITABLE_POSITION_LEVEL_DATA",
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
    }
    manifest_path = ROOT / config["outputs"]["manifest"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(manifest_path),
                "status": manifest["status"],
                "parent_manifest_unchanged": actual_parent_hash == expected_parent_hash,
                "selected_article_count": int(len(selected_articles)),
                "selected_source_count": int(selected_articles["source_name"].nunique()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
