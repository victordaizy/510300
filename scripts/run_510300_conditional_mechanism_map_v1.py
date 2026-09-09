"""运行 510300 条件机制地图 V1。

阶段分离：
1. freeze：冻结协议和全部输入哈希，只对收益文件做字节哈希；
2. build-states：只读取父项目 CF、DR、RC 状态，冻结条件面板；
3. evaluate：验证条件面板后，才读取 H00300 全收益并生成描述性地图。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.conditional_mechanism_map_v1 import (  # noqa: E402
    ARCHETYPE_FLAG_COLUMNS,
    QUADRANTS,
    RC_DIRECTIONS,
    assign_mechanism_archetypes,
    attach_total_return_context,
    build_archetype_conclusions,
    build_condition_state_panel,
    normalize_dates,
    summarize_condition_map,
)


PROGRAM_ID = "510300_CONDITIONAL_MECHANISM_MAP_V1"
CONFIG_PATH = ROOT / "config/510300_conditional_mechanism_map_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/conditional_mechanism_map_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("条件机制地图配置的 PROGRAM_ID 不匹配")
    return config


def project_path(relative: str) -> Path:
    return ROOT / Path(str(relative).replace("/", os.sep))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    atomic_bytes(path, (payload + "\n").encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _registered_input_roles(config: dict[str, Any]) -> dict[str, str]:
    parent = config["parent_contract"]
    states = config["state_inputs"]
    roles = {
        parent["manifest"]: "FROZEN_PARENT_PROTOCOL",
        parent["authoritative_status"]: "FROZEN_PARENT_STATUS",
        parent["state_freeze_receipt"]: "FROZEN_PARENT_STATE_RECEIPT",
        states["cashflow_panel"]["path"]: "OUTCOME_BLIND_PARENT_CF_STATE",
        states["present_value_panel"]["path"]: "OUTCOME_BLIND_PARENT_DR_STATE",
        states["risk_capacity_panel"]["path"]: "OUTCOME_BLIND_PARENT_RC_STATE",
        states["total_return_locked_until_condition_freeze"]["path"]: (
            "LOCKED_OUTCOME_NOT_VALUE_READ"
        ),
    }
    return roles


def _validate_parent_contract(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = config["parent_contract"]
    parent_manifest = json.loads(
        project_path(contract["manifest"]).read_text(encoding="utf-8")
    )
    parent_status = json.loads(
        project_path(contract["authoritative_status"]).read_text(encoding="utf-8")
    )
    parent_receipt = json.loads(
        project_path(contract["state_freeze_receipt"]).read_text(encoding="utf-8")
    )
    if (
        parent_manifest.get("manifest_payload_sha256")
        != contract["expected_parent_manifest_payload_sha256"]
    ):
        raise ValueError("父项目协议清单哈希与配置不一致")
    if parent_status.get("status") != contract["expected_parent_status"]:
        raise ValueError("父项目状态不是已冻结的第一阶段完成状态")
    if (
        parent_receipt.get("status")
        != contract["expected_parent_state_receipt_status"]
    ):
        raise ValueError("父项目状态产品回执状态不匹配")
    if (
        parent_receipt.get("receipt_payload_sha256")
        != contract["expected_parent_state_receipt_payload_sha256"]
    ):
        raise ValueError("父项目状态产品回执哈希与配置不一致")
    if parent_status.get("portfolio_evaluation_allowed") is not False:
        raise ValueError("父项目意外开放了组合评估")
    if parent_status.get("model_position_target") != "UNSET":
        raise ValueError("父项目意外生成了仓位目标")
    return parent_manifest, parent_status, parent_receipt


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"冻结清单已存在，禁止静默覆盖：{manifest_path}")
    parent_manifest, parent_status, parent_receipt = _validate_parent_contract(config)
    registered_inputs: dict[str, Any] = {}
    for relative, role in _registered_input_roles(config).items():
        path = project_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"冻结输入缺失：{path}")
        registered_inputs[Path(relative).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_CONDITION_BUILD_AND_TOTAL_RETURN_VALUE_READ",
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered_inputs,
        "parent_manifest_payload_sha256": parent_manifest[
            "manifest_payload_sha256"
        ],
        "parent_status_payload_sha256": parent_status["status_payload_sha256"],
        "parent_state_receipt_payload_sha256": parent_receipt[
            "receipt_payload_sha256"
        ],
        "condition_definitions_frozen": True,
        "future_return_values_read": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "sharpe_target_1_2_achieved": False,
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("协议尚未冻结，必须先运行 --phase freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("program_id") != PROGRAM_ID:
        raise ValueError("冻结清单 PROGRAM_ID 不匹配")
    expected_manifest_hash = manifest.get("manifest_payload_sha256")
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("manifest_payload_sha256", None)
    if canonical_hash(manifest_without_hash) != expected_manifest_hash:
        raise ValueError("冻结清单自身哈希校验失败")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("冻结后配置文件发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("冻结后配置语义发生漂移")
    for relative, expected_hash in manifest["implementation_files"].items():
        if sha256_file(project_path(relative)) != expected_hash:
            raise ValueError(f"冻结后实现文件发生漂移：{relative}")
    for relative, metadata in manifest["registered_inputs"].items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"冻结输入发生漂移或缺失：{relative}")
    _validate_parent_contract(config)
    return manifest


def build_and_freeze_condition_states(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    states = config["state_inputs"]
    cashflow = pd.read_parquet(project_path(states["cashflow_panel"]["path"]))
    present_value = pd.read_parquet(
        project_path(states["present_value_panel"]["path"])
    )
    risk_capacity = pd.read_parquet(
        project_path(states["risk_capacity_panel"]["path"])
    )
    condition = build_condition_state_panel(
        cashflow,
        present_value,
        risk_capacity,
        asof_tolerance_calendar_days=int(
            states["risk_capacity_panel"]["origin_asof_tolerance_calendar_days"]
        ),
        numerical_zero_tolerance=float(
            config["condition_definition"]["numerical_zero_tolerance"]
        ),
        rc_interaction_map=config["condition_definition"]["rc_interaction_map"],
    )
    condition["origin"] = normalize_dates(condition["origin"])
    condition = condition.loc[
        condition["origin"].between(
            pd.Timestamp(config["program"]["study_start"]),
            pd.Timestamp(config["program"]["state_origin_cutoff"]),
        )
    ].reset_index(drop=True)
    if condition.empty:
        raise ValueError("冻结区间内没有可用条件状态 origin")
    if bool(condition["future_return_values_read"].any()):
        raise ValueError("条件冻结阶段出现收益读取标记")

    condition_path = project_path(config["artifacts"]["condition_state_panel"])
    atomic_parquet(condition_path, condition)
    receipt = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "CONDITION_STATES_FROZEN_BEFORE_TOTAL_RETURN_VALUE_READ",
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "condition_state_panel": {
            "path": condition_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(condition_path),
            "rows": int(len(condition)),
            "first_origin": str(condition["origin"].min().date()),
            "last_origin": str(condition["origin"].max().date()),
        },
        "condition_quality_counts": {
            str(key): int(value)
            for key, value in condition["condition_quality_status"]
            .value_counts(dropna=False)
            .items()
        },
        "quadrant_counts": {
            str(key): int(value)
            for key, value in condition["cf_dr_quadrant"]
            .value_counts(dropna=False)
            .items()
        },
        "rc_direction_counts": {
            str(key): int(value)
            for key, value in condition["rc_direction"]
            .value_counts(dropna=False)
            .items()
        },
        "eligible_quadrant_origins": int(
            condition["condition_evaluation_eligible"].sum()
        ),
        "eligible_quadrant_rc_origins": int(
            condition["rc_interaction_evaluation_eligible"].sum()
        ),
        "future_return_values_read": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    receipt["receipt_payload_sha256"] = canonical_hash(receipt)
    atomic_json(
        project_path(config["artifacts"]["condition_state_receipt"]), receipt
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return receipt


def verify_condition_state_freeze(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    receipt_path = project_path(config["artifacts"]["condition_state_receipt"])
    if not receipt_path.is_file():
        raise FileNotFoundError("条件状态尚未冻结，必须先运行 --phase build-states")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_hash = receipt.get("receipt_payload_sha256")
    without_hash = dict(receipt)
    without_hash.pop("receipt_payload_sha256", None)
    if canonical_hash(without_hash) != expected_hash:
        raise ValueError("条件状态回执自身哈希校验失败")
    if (
        receipt.get("protocol_manifest_payload_sha256")
        != manifest["manifest_payload_sha256"]
    ):
        raise ValueError("条件状态回执未绑定当前协议")
    condition_path = project_path(config["artifacts"]["condition_state_panel"])
    if sha256_file(condition_path) != receipt["condition_state_panel"]["sha256"]:
        raise ValueError("冻结条件面板发生漂移")
    if receipt.get("future_return_values_read") is not False:
        raise ValueError("条件状态回执的收益读取状态不合法")
    return receipt


def _percent(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.2%}"


def _float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _markdown_summary_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| 样本 | 视图 | 条件 | 期限 | n | 均值 | 中位数 | 胜率 | 可靠性 |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| {scope} | {view} | {condition} | {horizon}D | {n} | {mean} | "
            "{median} | {positive} | {status} |".format(
                scope=row["sample_scope"],
                view=row["view_type"],
                condition=row["condition_id"],
                horizon=int(row["horizon_market_days"]),
                n=int(row["observed_origin_count"]),
                mean=_percent(row["mean_total_return"]),
                median=_percent(row["median_total_return"]),
                positive=_percent(row["positive_rate"]),
                status=row["reliability_status"],
            )
        )
    return lines


def _latest_state_payload(evaluated: pd.DataFrame) -> dict[str, Any]:
    ordered = evaluated.sort_values("origin")
    latest = ordered.iloc[-1]
    complete_eligible = ordered.loc[
        ordered["condition_evaluation_eligible"]
        & ordered["origin_completion_status"].eq(
            "COMPLETE_CALENDAR_MONTH_LAST_TRADING_SESSION"
        )
    ]
    latest_complete = complete_eligible.iloc[-1] if not complete_eligible.empty else None

    def payload(row: pd.Series | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "origin": str(pd.Timestamp(row["origin"]).date()),
            "origin_completion_status": str(row["origin_completion_status"]),
            "cf_dr_quadrant": str(row["cf_dr_quadrant"]),
            "rc_direction": str(row["rc_direction"]),
            "rc_interaction_role": str(row["rc_interaction_role"]),
            "condition_quality_status": str(row["condition_quality_status"]),
            "cf_news": (
                None if pd.isna(row["cf_news"]) else float(row["cf_news"])
            ),
            "discount_rate_easing_news": (
                None
                if pd.isna(row["discount_rate_easing_news"])
                else float(row["discount_rate_easing_news"])
            ),
            "risk_capacity_news": (
                None
                if pd.isna(row["risk_capacity_news"])
                else float(row["risk_capacity_news"])
            ),
        }

    return {
        "latest_state_origin": payload(latest),
        "latest_complete_eligible_origin": payload(latest_complete),
        "model_position_target": "UNSET",
    }


def build_report(
    config: dict[str, Any],
    manifest: dict[str, Any],
    state_receipt: dict[str, Any],
    evaluated: pd.DataFrame,
    summary: pd.DataFrame,
    conclusions: list[dict[str, Any]],
    latest_state: dict[str, Any],
) -> str:
    full_scope = "FULL_2015_TO_CUTOFF"
    user_scope = "USER_PRIMARY_2021_TO_CUTOFF"
    quadrant_table = summary.loc[
        summary["view_type"].eq("QUADRANT")
        & summary["sample_scope"].isin([full_scope, user_scope])
    ].sort_values(["sample_scope", "condition_id", "horizon_market_days"])
    rc_table = summary.loc[
        summary["view_type"].eq("QUADRANT_X_RC")
        & summary["sample_scope"].isin([full_scope, user_scope])
        & summary["horizon_market_days"].isin([60, 120])
        & summary["observed_origin_count"].gt(0)
    ].sort_values(["sample_scope", "condition_id", "horizon_market_days"])
    archetype_rows: list[pd.DataFrame] = []
    for archetype_id, specification in config["mechanism_archetypes"].items():
        horizon = int(specification["primary_evaluation_horizon_market_days"])
        archetype_rows.append(
            summary.loc[
                summary["view_type"].eq("ARCHETYPE")
                & summary["condition_id"].eq(archetype_id)
                & summary["horizon_market_days"].eq(horizon)
            ]
        )
    archetype_table = pd.concat(archetype_rows, ignore_index=True)

    quality_counts = state_receipt["condition_quality_counts"]
    latest = latest_state["latest_state_origin"]
    latest_complete = latest_state["latest_complete_eligible_origin"]
    pass_rows = int(summary["reliability_status"].eq("PASS_DESCRIPTIVE_COVERAGE").sum())
    partial_rows = int(summary["reliability_status"].str.startswith("PARTIAL").sum())
    no_view_rows = int(summary["reliability_status"].str.startswith("NO_VIEW").sum())
    repeated = [
        item["archetype_id"]
        for item in conclusions
        if item["conclusion"]
        == "REPEATED_DESCRIPTIVE_SUPPORT_FULL_AND_2021_PLUS"
    ]

    lines = [
        "# 510300 条件机制地图 V1",
        "",
        "## 结论先行",
        "",
        "第二阶段已经完成，但结果仍是**描述性条件地图，不是策略**。四类 `CF×DR` 状态、RC 交互关系、五类经济机制候选及 20D/60D/120D 后验全收益均按冻结规则计算。",
        "",
        f"- 条件状态总数：{state_receipt['condition_state_panel']['rows']}；四象限可评估 origin：{state_receipt['eligible_quadrant_origins']}；四象限×RC 可评估 origin：{state_receipt['eligible_quadrant_rc_origins']}。",
        f"- 状态质量：`PASS_FROZEN_PARENT_STATES`={quality_counts.get('PASS_FROZEN_PARENT_STATES', 0)}；其余 PARTIAL/NO_VIEW 保留原状态，没有补写成 PASS。",
        f"- 条件统计可靠性行：PASS={pass_rows}，PARTIAL={partial_rows}，NO_VIEW={no_view_rows}。每一行是一个冻结样本范围、视图、条件和期限的组合。",
        f"- 在全样本与 2021+ 样本的预注册主期限上重复获得描述性方向支持的机制：{', '.join(repeated) if repeated else '无'}。",
        "- `PORTFOLIO_EVALUATION_ALLOWED=false`，`MODEL_POSITION_TARGET=UNSET`，没有净值、仓位、交易次数或夏普率。夏普率 1.2 仍未被评估，更未达成。",
        "",
        "## 冻结边界",
        "",
        f"- 协议清单哈希：`{manifest['manifest_payload_sha256']}`。",
        f"- 条件状态回执哈希：`{state_receipt['receipt_payload_sha256']}`。",
        "- 条件只复用父项目相邻月度的一阶差分：`CF_NEWS=ΔCF_LEVEL`、`DR_EASING_NEWS=-ΔERP_Z`、`RC_NEWS=ΔRC_FACTOR`。",
        "- 数值零变化单列为中性，不进入四象限。没有按收益调整差分窗口、阈值、分组或 RC 方向。",
        "- 收益文件在条件状态面板冻结并校验后才读取；过去 20D/60D 仅用于定义价格语境，未来 20D/60D/120D 仅用于描述。",
        "",
        "## 四象限条件收益",
        "",
        *_markdown_summary_table(quadrant_table),
        "",
        "## RC 放大或削弱",
        "",
        "下表列出有观测的 60D/120D 四象限×RC 单元。RC 正向代表风险承载能力改善，负向代表约束增强；交互语义在收益读取前已经冻结。",
        "",
        *_markdown_summary_table(rc_table),
        "",
        "## 五类机制候选",
        "",
        *_markdown_summary_table(archetype_table),
        "",
        "| 机制候选 | 冻结主期限 | 跨样本结论 |",
        "|---|---:|---|",
    ]
    for item in conclusions:
        lines.append(
            f"| {item['archetype_id']} | {item['primary_evaluation_horizon_market_days']}D | {item['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "这些标签是经济机制候选：`TEMPORARY_LIQUIDITY_SHOCK_CANDIDATE` 尤其不是已经证实的暂时冲击；只有其冻结条件与后续恢复方向的描述性对应。",
            "",
            "## 最新状态快照",
            "",
            f"- 最新状态 origin：{latest['origin']}；月度完整性：`{latest['origin_completion_status']}`；状态：`{latest['cf_dr_quadrant']} × {latest['rc_direction']}`；质量：`{latest['condition_quality_status']}`。",
            f"- 最新完整且可评估 origin：{latest_complete['origin'] if latest_complete else '无'}；状态：`{latest_complete['cf_dr_quadrant'] if latest_complete else 'NO_VIEW'} × {latest_complete['rc_direction'] if latest_complete else 'NO_VIEW'}`。",
            "- 这只是状态识别，不对应任何仓位。",
            "",
            "## 数据可靠性与解释边界",
            "",
            "- 2015 年起滚动计算；同时固定报告 2021 年以来样本，避免只给出一个样本窗口。",
            "- 60D/120D 月度前瞻窗口高度重叠，因此统计行不被当作独立试验；不报告 p 值或显著性筛选。",
            "- PASS 门槛固定为至少 12 个已观察 origin、至少 3 个自然年、前瞻覆盖率至少 80%、父状态 PASS 占比至少 80%；6 个 origin/2 年以上仅为 PARTIAL。",
            "- 财务信息以 `NOTICE_DATE` 为主时钟，但当前二级聚合器的数值可能含后续修订；经营现金流没有独立本地交叉源。",
            "- 历史指数权重未获版本化 PIT 证明，继续使用市值代理；成分股 PIT 股息率仍为 NO_VIEW，现金流久期仍为代理。",
            "- 信用利差与 ETF 份额的历史发布时钟未完全版本证明，RC 中相应证据仍只用于发现和状态测量。",
            "",
            "## 下一阶段边界",
            "",
            "依据用户要求，无论本轮条件单元是否达到 PASS，都可进入第三阶段的**预注册**；但 PARTIAL/NO_VIEW 不会改写成合格结果。第三阶段只能预测 `EXPECTED_60D_TOTAL_RETURN`、`EXPECTED_120D_TOTAL_RETURN`、`20D_DRAWDOWN_PROBABILITY`，并必须分别检验 CF、DR、RC 的增量信息。当前文件没有执行第三阶段，也没有设计 510300/现金配置。",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_condition_map(
    config: dict[str, Any],
    manifest: dict[str, Any],
    state_receipt: dict[str, Any],
) -> dict[str, Any]:
    condition = pd.read_parquet(
        project_path(config["artifacts"]["condition_state_panel"])
    )
    outcome_contract = config["state_inputs"][
        "total_return_locked_until_condition_freeze"
    ]
    total_return = pd.read_parquet(project_path(outcome_contract["path"]))
    evaluated = attach_total_return_context(
        condition,
        total_return,
        horizons_market_days=config["program"]["horizons_market_days"],
        trailing_horizons_market_days=config["evaluation"][
            "trailing_price_context_market_days"
        ],
        observation_cutoff=config["program"]["outcome_observation_cutoff"],
    )
    evaluated = assign_mechanism_archetypes(
        evaluated,
        numerical_zero_tolerance=float(
            config["condition_definition"]["numerical_zero_tolerance"]
        ),
    )
    summary = summarize_condition_map(
        evaluated,
        horizons_market_days=config["program"]["horizons_market_days"],
        sample_scopes=config["evaluation"]["sample_scopes"],
        reliability_gate=config["evaluation"]["reliability_gate"],
        mechanism_archetypes=config["mechanism_archetypes"],
    )
    conclusions = build_archetype_conclusions(
        summary,
        mechanism_archetypes=config["mechanism_archetypes"],
        full_scope_id="FULL_2015_TO_CUTOFF",
        user_scope_id="USER_PRIMARY_2021_TO_CUTOFF",
    )
    latest_state = _latest_state_payload(evaluated)

    evaluated_path = project_path(config["artifacts"]["evaluated_observations"])
    map_path = project_path(config["artifacts"]["condition_map"])
    atomic_parquet(evaluated_path, evaluated)
    atomic_parquet(map_path, summary)

    map_payload = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "DESCRIPTIVE_CONDITIONAL_MECHANISM_MAP_ONLY",
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "condition_state_receipt_payload_sha256": state_receipt[
            "receipt_payload_sha256"
        ],
        "sample_scopes": config["evaluation"]["sample_scopes"],
        "summary_rows": dataframe_records(summary),
        "archetype_conclusions": conclusions,
        "latest_state": latest_state,
        "overlapping_forward_windows": True,
        "strategy_claim_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    map_payload["payload_sha256"] = canonical_hash(map_payload)
    atomic_json(project_path(config["artifacts"]["condition_map_json"]), map_payload)

    report = build_report(
        config,
        manifest,
        state_receipt,
        evaluated,
        summary,
        conclusions,
        latest_state,
    )
    atomic_text(project_path(config["artifacts"]["final_report"]), report)

    reliability_counts = {
        str(key): int(value)
        for key, value in summary["reliability_status"].value_counts().items()
    }
    archetype_counts = {
        archetype_id: int(evaluated[flag].sum())
        for archetype_id, flag in ARCHETYPE_FLAG_COLUMNS.items()
    }
    status = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "STAGE_2_COMPLETE_CONDITIONAL_MAP_DESCRIPTIVE_ONLY",
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "condition_state_receipt_payload_sha256": state_receipt[
            "receipt_payload_sha256"
        ],
        "condition_state_panel_sha256": state_receipt["condition_state_panel"][
            "sha256"
        ],
        "evaluated_observations": {
            "path": evaluated_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(evaluated_path),
            "rows": int(len(evaluated)),
            "future_return_observed_counts": {
                f"{horizon}D": int(
                    evaluated[f"forward_total_return_{horizon}d"].notna().sum()
                )
                for horizon in config["program"]["horizons_market_days"]
            },
        },
        "condition_map": {
            "path": map_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(map_path),
            "rows": int(len(summary)),
            "reliability_counts": reliability_counts,
        },
        "archetype_origin_counts": archetype_counts,
        "archetype_conclusions": conclusions,
        "latest_state": latest_state,
        "parent_limitations_propagated": config["limitations"],
        "next_stage_status": (
            "ALLOWED_TO_PREREGISTER_STAGE_3_PER_USER_DIRECTION_WITHOUT_"
            "PROMOTING_PARTIAL_OR_NO_VIEW_RESULTS"
        ),
        "predictive_model_validated": False,
        "strategy_validated": False,
        "return_evaluation": "DESCRIPTIVE_CONDITIONAL_TOTAL_RETURN_ONLY",
        "portfolio_evaluation_allowed": False,
        "nav_calculated": False,
        "sharpe_calculated": False,
        "sharpe_target_1_2_achieved": False,
        "model_position_target": "UNSET",
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json(project_path(config["artifacts"]["authoritative_status"]), status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 510300 条件机制地图 V1")
    parser.add_argument(
        "--phase",
        choices=["freeze", "build-states", "evaluate", "all"],
        default="all",
    )
    args = parser.parse_args()
    config = load_config()
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"build-states", "all"}:
        build_and_freeze_condition_states(config, manifest)
    if args.phase in {"evaluate", "all"}:
        state_receipt = verify_condition_state_freeze(config, manifest)
        evaluate_condition_map(config, manifest, state_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
