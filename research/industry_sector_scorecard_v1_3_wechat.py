"""微信公众号前瞻证据审计与板块三轴整合。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class WechatForwardError(ValueError):
    """微信公众号前瞻证据不满足冻结数据契约。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wechat_snapshot(config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """校验公众号快照、文章清单和入选文章全文哈希。"""

    snapshot = config["wechat_snapshot"]
    root = Path(snapshot["root"])
    paths = {
        "audit_json": root / snapshot["audit_json"],
        "article_csv": root / snapshot["article_csv"],
        "source_config": root / snapshot["source_config"],
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise WechatForwardError(f"公众号快照缺少文件：{missing}")
    for key, path in paths.items():
        actual = sha256(path)
        expected = str(snapshot["expected_hashes"][key])
        if actual != expected:
            raise WechatForwardError(f"公众号快照哈希变化：{key}，{actual} != {expected}")

    audit = json.loads(paths["audit_json"].read_text(encoding="utf-8"))
    count_checks = {
        "expected_account_count": "expected_account_count",
        "expected_article_count": "article_count",
        "expected_archive_complete_count": "archive_complete_count",
        "expected_archive_partial_count": "archive_partial_count",
    }
    for expected_key, audit_key in count_checks.items():
        if int(snapshot[expected_key]) != int(audit[audit_key]):
            raise WechatForwardError(
                f"公众号快照计数变化：{audit_key}={audit[audit_key]}，"
                f"预期{snapshot[expected_key]}"
            )
    if audit["status"] != snapshot["accepted_corpus_status"]:
        raise WechatForwardError(f"公众号快照状态变化：{audit['status']}")

    articles = pd.read_csv(paths["article_csv"], encoding="utf-8-sig")
    if articles["id"].duplicated().any():
        raise WechatForwardError("公众号文章清单ID重复")
    articles["id"] = pd.to_numeric(articles["id"], errors="raise").astype(int)
    articles["published_at"] = pd.to_datetime(
        articles["published_at"], errors="coerce", utc=True
    )
    if articles["published_at"].isna().any():
        raise WechatForwardError("公众号文章存在非法发布时间")
    cutoff = pd.Timestamp(config["information_cutoff"]).tz_convert("UTC")
    selected_rows: list[dict[str, Any]] = []
    for selection in snapshot["selected_articles"]:
        article_id = int(selection["id"])
        matched = articles.loc[articles["id"].eq(article_id)]
        if len(matched) != 1:
            raise WechatForwardError(f"入选文章ID不存在或不唯一：{article_id}")
        row = matched.iloc[0]
        if row["published_at"] > cutoff:
            raise WechatForwardError(f"入选文章晚于信息截止时间：{article_id}")
        if str(row["archive_status"]) != "SUCCESS":
            raise WechatForwardError(f"入选文章不是完整正文：{article_id}")
        markdown_path = root / str(row["markdown_path"])
        if not markdown_path.exists():
            raise WechatForwardError(f"入选文章正文缺失：{markdown_path}")
        actual_hash = sha256(markdown_path)
        expected_hash = str(selection["markdown_sha256"])
        if actual_hash != expected_hash:
            raise WechatForwardError(
                f"入选文章正文哈希变化：{article_id}，{actual_hash} != {expected_hash}"
            )
        selected_rows.append(
            {
                "article_id": article_id,
                "source_name": str(row["source_name"]),
                "published_at": row["published_at"],
                "title": str(row["title"]),
                "classification_label": str(row["classification_label"]),
                "archive_status": str(row["archive_status"]),
                "text_length": int(row["text_length"]),
                "canonical_url": str(row["canonical_url"]),
                "markdown_path": str(markdown_path),
                "markdown_sha256": actual_hash,
            }
        )
    selected = pd.DataFrame(selected_rows).sort_values(
        ["published_at", "article_id"], ascending=[False, True]
    )
    if selected["article_id"].duplicated().any():
        raise WechatForwardError("入选公众号文章重复")
    return selected.reset_index(drop=True), audit


def build_evidence_frame(
    assessments: Sequence[Mapping[str, Any]],
    entity_type: str,
    selected_articles: pd.DataFrame,
    evidence_rules: Mapping[str, Any],
) -> pd.DataFrame:
    """把人工审计后的方向变成可追溯证据表。"""

    article_by_id = selected_articles.set_index("article_id")
    valid_directions = set(evidence_rules["direction_labels"])
    valid_confidence = set(evidence_rules["confidence_labels"])
    high_minimum = int(evidence_rules["minimum_independent_sources_for_high_confidence"])
    rows: list[dict[str, Any]] = []
    for assessment in assessments:
        support_ids = [int(value) for value in assessment["support_article_ids"]]
        adverse_ids = [int(value) for value in assessment["adverse_article_ids"]]
        reference_ids = sorted(set(support_ids + adverse_ids))
        if not reference_ids:
            raise WechatForwardError(f"{assessment['entity_id']}没有文章证据")
        missing = sorted(set(reference_ids).difference(article_by_id.index))
        if missing:
            raise WechatForwardError(f"{assessment['entity_id']}引用未入选文章：{missing}")
        direction = str(assessment["direction"])
        confidence = str(assessment["confidence"])
        if direction not in valid_directions or confidence not in valid_confidence:
            raise WechatForwardError(f"{assessment['entity_id']}方向或置信度非法")
        sources = sorted(
            article_by_id.loc[reference_ids, "source_name"].astype(str).unique().tolist()
        )
        if confidence == "HIGH" and len(sources) < high_minimum:
            raise WechatForwardError(
                f"{assessment['entity_id']}高置信但独立来源只有{len(sources)}个"
            )
        rows.append(
            {
                "entity_type": entity_type,
                "entity_id": str(assessment["entity_id"]),
                "entity_name_cn": str(assessment["entity_name_cn"]),
                "wechat_forward_direction": direction,
                "wechat_confidence": confidence,
                "independent_source_count": len(sources),
                "support_article_ids_json": json.dumps(support_ids, ensure_ascii=False),
                "adverse_article_ids_json": json.dumps(adverse_ids, ensure_ascii=False),
                "source_names_json": json.dumps(sources, ensure_ascii=False),
                "thesis": str(assessment["thesis"]),
                "risk_flags_json": json.dumps(
                    list(assessment["risk_flags"]), ensure_ascii=False
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result[["entity_type", "entity_id"]].duplicated().any():
        raise WechatForwardError("微信前瞻实体重复")
    return result


def _phase_guard(price_phase: str, config: Mapping[str, Any]) -> str:
    if price_phase in set(config["fish_middle_phases"]):
        return "FISH_MIDDLE_ELIGIBLE"
    if price_phase in set(config["tail_risk_phases"]):
        return "FISH_TAIL_RISK"
    if price_phase in set(config["wait_for_turn_phases"]):
        return "WAIT_FOR_TURN"
    return "PHASE_NO_VIEW"


def _integrated_state(row: Mapping[str, Any]) -> tuple[str, str]:
    direction = str(row["wechat_forward_direction"])
    phase_guard = str(row["phase_guard"])
    historical_favorable = bool(row["historical_odds_favorable"])
    current_positive = bool(row["current_forward_positive"])
    confidence = str(row["wechat_confidence"])
    if direction == "POSITIVE":
        if phase_guard == "FISH_TAIL_RISK":
            return "FISH_TAIL_WARNING_NO_CHASE", "NO_CHASE_TAIL_RISK_NO_POSITION"
        if (
            confidence == "HIGH"
            and historical_favorable
            and not current_positive
            and phase_guard == "FISH_MIDDLE_ELIGIBLE"
        ):
            return "QUALITATIVE_LEADS_QUANT", "RESEARCH_EARLY_QUALITATIVE_NO_POSITION"
        if (
            current_positive
            and historical_favorable
            and phase_guard == "FISH_MIDDLE_ELIGIBLE"
        ):
            return "FISH_MIDDLE_CONVERGENCE", "RESEARCH_FISH_MIDDLE_NO_POSITION"
        if not historical_favorable:
            return "FORWARD_POSITIVE_ODDS_CONFLICT", "MONITOR_CONFLICT_NO_POSITION"
        if phase_guard == "WAIT_FOR_TURN":
            return "WAIT_FOR_PRICE_TURN", "WAIT_NO_POSITION"
        return "POSITIVE_UNCONFIRMED", "MONITOR_NO_POSITION"
    if direction == "NEGATIVE":
        if not current_positive:
            return "NEGATIVE_CONVERGENCE", "NEGATIVE_VIEW_NO_POSITION"
        return "NEGATIVE_EVIDENCE_CONFLICT", "MONITOR_CONFLICT_NO_POSITION"
    return "MIXED_NO_HARD_VIEW", "NO_VIEW_NO_POSITION"


def integrate_forward_axes(
    parent_rows: pd.DataFrame,
    price_phases: pd.DataFrame,
    evidence: pd.DataFrame,
    integration_config: Mapping[str, Any],
) -> pd.DataFrame:
    """整合当前量化、历史赔率、价格阶段和微信前瞻方向。"""

    keys = ["entity_type", "entity_id", "entity_name_cn"]
    merged = parent_rows.merge(price_phases, on=keys, validate="one_to_one").merge(
        evidence, on=keys, validate="one_to_one"
    )
    if len(merged) != len(parent_rows) or len(merged) != len(evidence):
        raise WechatForwardError("三轴整合后实体覆盖不完整")
    merged["phase_guard"] = merged["price_phase"].map(
        lambda value: _phase_guard(str(value), integration_config)
    )
    states = merged.apply(_integrated_state, axis=1, result_type="expand")
    merged["integrated_forward_state"] = states[0]
    merged["action_state"] = states[1]
    state_order = {
        "FISH_MIDDLE_CONVERGENCE": 1,
        "QUALITATIVE_LEADS_QUANT": 2,
        "FORWARD_POSITIVE_ODDS_CONFLICT": 3,
        "WAIT_FOR_PRICE_TURN": 4,
        "POSITIVE_UNCONFIRMED": 5,
        "MIXED_NO_HARD_VIEW": 6,
        "NEGATIVE_EVIDENCE_CONFLICT": 7,
        "FISH_TAIL_WARNING_NO_CHASE": 8,
        "NEGATIVE_CONVERGENCE": 9,
    }
    merged["integrated_state_order"] = merged["integrated_forward_state"].map(
        state_order
    )
    if merged["integrated_state_order"].isna().any():
        raise WechatForwardError("出现未注册的三轴状态")
    merged = merged.sort_values(
        ["entity_type", "integrated_state_order", "current_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    merged["integrated_research_rank"] = merged.groupby("entity_type").cumcount() + 1
    merged["wechat_direction_is_probability"] = False
    merged["synthetic_numeric_score"] = np.nan
    merged["position_mapping_enabled"] = False
    return merged


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染微信公众号前瞻证据整合报告。"""

    def markdown_cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    market = report["market_forward_context"]
    lines = [
        "# 板块评分卡 V1.3：微信公众号前瞻证据整合",
        "",
        f"- 截止日：`{report['as_of_date']}`",
        f"- 公众号快照：{report['corpus_audit']['article_count']}篇，"
        f"{report['corpus_audit']['archive_complete_count']}篇完整，"
        f"状态`{report['corpus_audit']['status']}`。",
        f"- 市场流动性：`{market['liquidity_state']}`。{market['liquidity_summary']}",
        f"- 政策环境：`{market['policy_state']}`。{market['policy_summary']}",
        f"- 国家队持仓：`{market['national_team_holdings_state']}`。",
        "- 微信方向不是概率；文章数量不投票；没有合成新分数。",
    ]
    for title, key in (("互斥经济大类", "economic_buckets"), ("重叠主题", "themes")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| 研究序 | 板块 | 三轴状态 | 当前分 | 价格阶段 | 微信方向 | 置信度/来源 | 历史赔率轴 | 赔率 | 60日期望 | 关键判断 |",
                "|---:|---|---|---:|---|---|---|---|---:|---:|---|",
            ]
        )
        for row in report[key]:
            lines.append(
                f"| {row['integrated_research_rank']} | {row['entity_name_cn']} | "
                f"{row['integrated_forward_state']} | {row['current_score']:.1f} | "
                f"{row['price_phase']} | {row['wechat_forward_direction']} | "
                f"{row['wechat_confidence']}/{row['independent_source_count']} | "
                f"{row['historical_odds_axis']} | {row['average_payoff_ratio']:.2f} | "
                f"{row['mean_excess_return_60d']:.2%} | {markdown_cell(row['thesis'])} |"
            )
    lines.extend(
        [
            "",
            "## 入选文章证据",
            "",
            "| ID | 日期 | 公众号 | 标题 | 正文SHA-256 |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in report["selected_articles"]:
        lines.append(
            f"| {row['article_id']} | {row['published_at'][:10]} | "
            f"{markdown_cell(row['source_name'])} | {markdown_cell(row['title'])} | "
            f"`{row['markdown_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 单一作者公司研究只作二级确认，不能独立形成行业高置信方向。",
            "- 微信证据只允许形成研究状态，不允许覆盖缺失数据、生成概率、仓位或订单。",
            "- 真正的条件胜率和条件赔率需要冻结后的20日与60日前瞻样本成熟后再估计。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "WechatForwardError",
    "build_evidence_frame",
    "integrate_forward_axes",
    "render_markdown",
    "sha256",
    "validate_wechat_snapshot",
]
