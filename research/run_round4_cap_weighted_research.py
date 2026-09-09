"""运行第四轮流通市值近似加权宽度因子研究。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from research.run_registered_factor_research import (
    _assign_evidence,
    _benjamini_hochberg,
    _evaluate_hypothesis,
)


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"
REGISTRY_FILE = ROOT / "config" / "factor_registry_round4.yaml"
DATA_FILE = ROOT / "data" / "features" / "510300_round4_cap_weighted_dataset.parquet"
PRIOR_REPORTS = (
    ("round1", ROOT / "reports" / "research" / "registered_factor_research.json"),
    ("round2", ROOT / "reports" / "research" / "round2_mechanism_research.json"),
    ("round3", ROOT / "reports" / "research" / "round3_breadth_research.json"),
)
JSON_FILE = ROOT / "reports" / "research" / "round4_cap_weighted_research.json"
MARKDOWN_FILE = ROOT / "reports" / "research" / "round4_cap_weighted_research.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_effective_sample_guard(result: dict, minimum: int) -> None:
    pseudo = result["splits"]["pseudo_oos"]
    observations = int(pseudo["approx_independent_observations"])
    uncapped = result["evidence"]["rating"]
    audited = uncapped
    if observations < minimum and uncapped in {"WEAK", "STRONG"}:
        audited = "WEAK_SAMPLE_LIMITED"
    result["evidence"]["uncapped_rating"] = uncapped
    result["evidence"]["rating"] = audited
    result["evidence"]["effective_sample_guard"] = {
        "approx_independent_observations": observations,
        "minimum_required": minimum,
        "passes": observations >= minimum,
        "governance": "PREDECLARED_FOR_ROUND4_BY_REUSE_OF_EXISTING_AUDIT_RULE",
    }


def _studywide_q_values(round4_results: list[dict]) -> tuple[list[float | None], dict]:
    prior: list[tuple[str, dict]] = []
    source_hashes: dict[str, str] = {}
    for round_name, path in PRIOR_REPORTS:
        if not path.exists():
            raise FileNotFoundError(f"全研究多重检验缺少{round_name}报告：{path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        prior.extend((round_name, item) for item in report["results"])
        source_hashes[round_name] = _sha256(path)
    combined = prior + [("round4", item) for item in round4_results]
    p_values = [item["splits"]["pseudo_oos"]["hac_regression"].get("p_value") for _, item in combined]
    q_values = _benjamini_hochberg(p_values)
    audit_rows = [
        {
            "round": round_name,
            "hypothesis_id": item["hypothesis_id"],
            "factor_id": item["factor_id"],
            "target": item["target"],
            "pseudo_oos_hac_p_value": p_value,
            "studywide_bh_q_value": q_value,
        }
        for (round_name, item), p_value, q_value in zip(combined, p_values, q_values)
    ]
    round4_q = q_values[len(prior) :]
    counts = {name: sum(1 for round_name, _ in combined if round_name == name) for name in ("round1", "round2", "round3", "round4")}
    return round4_q, {
        "status": "PASS",
        "method": "Benjamini-Hochberg",
        "family": "ALL_REGISTERED_DAILY_HYPOTHESES_ROUNDS_1_TO_4",
        **{f"{name}_hypothesis_count": count for name, count in counts.items()},
        "studywide_hypothesis_count": len(combined),
        "evidence_scoring_uses_studywide_q": True,
        "prior_report_hashes": source_hashes,
        "results": audit_rows,
    }


def _format(value: float | None, percentage: bool = False) -> str:
    if value is None:
        return ""
    return f"{value:.2%}" if percentage else f"{value:.4f}"


def _render_markdown(report: dict) -> str:
    lines = [
        "# 第四轮流通市值近似加权宽度研究",
        "",
        "## 治理口径",
        "",
        "- 六个因子在第四轮结果计算前登记；复用历史使其仍是探索性研究，不是严格样本外确认。",
        "- 权重由当日点时成员的新浪流通市值归一化得到，不是中证自由流通调整权重。",
        "- 缺失流通股本的成员不估算、不倒填；每日成员覆盖至少99%，剩余权重重新归一化。",
        "- 证据评分使用第一至四轮全部日线假设的BH q值；D20独立观测不足20时统一标记样本受限。",
        "",
        "## 假设结果",
        "",
        "|ID|因子|目标|开发Spearman|pseudo-OOS Spearman|非重叠中位数|成本后有利-不利|全研究BH q|得分|评级|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        development = item["splits"]["development"]
        pseudo = item["splits"]["pseudo_oos"]
        lines.append(
            f"|{item['hypothesis_id']}|{item['factor_name_cn']}|{item['target']}|"
            f"{_format(development['spearman'])}|{_format(pseudo['spearman'])}|"
            f"{_format(pseudo['non_overlapping_spearman']['median'])}|"
            f"{_format(pseudo['conditional_returns']['favorable_minus_adverse_mean_net_return'], True)}|"
            f"{_format(item['multiple_testing']['studywide_bh_q_value_all_daily_hypotheses'])}|"
            f"{item['evidence']['score']}/7|{item['evidence']['rating']}|"
        )
    lines += [
        "",
        f"全研究多重检验族共{report['studywide_multiple_testing_audit']['studywide_hypothesis_count']}个日线假设。",
        "完整HAC、非重叠错位样本、移动区块Bootstrap、条件分桶和年度稳定性保存在同名JSON中。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    for path in (SETTINGS_FILE, REGISTRY_FILE, DATA_FILE, *(path for _, path in PRIOR_REPORTS)):
        if not path.exists():
            raise FileNotFoundError(f"第四轮研究输入缺失：{path}")
    settings = yaml.safe_load(SETTINGS_FILE.read_text(encoding="utf-8"))
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    dataset = pd.read_parquet(DATA_FILE)
    dataset["date"] = pd.to_datetime(dataset["date"])
    for horizon in (5, 20):
        dataset[f"label_end_date_{horizon}d"] = pd.to_datetime(dataset[f"label_end_date_{horizon}d"])
    definitions = {item["factor_id"]: item for item in registry["factor_definitions"]}
    results = [
        _evaluate_hypothesis(dataset, definitions[item["factor_id"]], item, settings["research_split"])
        for item in registry["hypotheses"]
    ]
    round4_target_q: dict[str, float | None] = {}
    for target in sorted({item["target"] for item in results}):
        indices = [index for index, item in enumerate(results) if item["target"] == target]
        p_values = [results[index]["splits"]["pseudo_oos"]["hac_regression"].get("p_value") for index in indices]
        q_values = _benjamini_hochberg(p_values)
        for index, q_value in zip(indices, q_values):
            round4_target_q[results[index]["hypothesis_id"]] = q_value
    studywide_q, audit = _studywide_q_values(results)
    minimum_independent = int(
        registry["evidence_governance"]["effective_sample_guard"][
            "minimum_pseudo_oos_approx_independent_observations"
        ]
    )
    for item, q_value in zip(results, studywide_q):
        _assign_evidence(item, q_value)
        item["multiple_testing"].update(
            {
                "round4_target_family_bh_q_value": round4_target_q[item["hypothesis_id"]],
                "studywide_bh_q_value_all_daily_hypotheses": q_value,
                "studywide_hypothesis_count": audit["studywide_hypothesis_count"],
                "evidence_scoring_basis": "STUDYWIDE_BH_Q_VALUE_ROUNDS_1_TO_4",
            }
        )
        _apply_effective_sample_guard(item, minimum_independent)
    report = {
        "status": "PASS_APPROXIMATION_NOT_OFFICIAL_WEIGHT",
        "generated_at": datetime.now(ZoneInfo(settings["project"]["timezone"])).isoformat(),
        "governance": registry["research_status"],
        "registration_timing": registry["registration_timing"],
        "derivation_note": registry["derivation_note"],
        "weighting_semantics": "DAILY_CIRCULATING_MARKET_CAP_APPROXIMATION_NOT_OFFICIAL_CSI_WEIGHT",
        "hypothesis_count": len(results),
        "results": results,
        "studywide_multiple_testing_audit": audit,
        "source_hashes": {"dataset": _sha256(DATA_FILE), "registry": _sha256(REGISTRY_FILE)},
    }
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    MARKDOWN_FILE.write_text(_render_markdown(report), encoding="utf-8")
    print(f"第四轮流通市值近似加权研究报告：{MARKDOWN_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
