"""本地政策来源与既有评分同钟核对，保留全部节点、反例和未知。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_policy_announcement_score_intake_v1"
CLOCK = ROOT / "reports/research/510300_policy_information_clock_v1"
JOINT = ROOT / "reports/research/510300_macro_technical_first_passage_v1_clock_adapter"
POLICY = ROOT / "reports/research/510300_equity_support_policy_event_v1"
HIST = ROOT / "reports/research/510300_historical_policy_constraint_events_v1"
PRIOR = [
    ("跨通道政策时钟", CLOCK / "results/result.json", "来源与时钟；未拟合政策评分；24节点不是完整母集"),
    ("七次降息类型与时点", HIST / "result.json", "历史解释；既有7次，不是新独立样本，0新完整账户"),
    ("资本市场支持工具固定20日", POLICY / "summary.json", "7事实/6公告，完整账户失败；不改日期/退出救援"),
    ("操作量与资金压力", ROOT / "reports/research/510300_policy_liquidity_quantity_v1/result.json", "实际操作量与DR007；原固定预测/账户失败"),
    ("LPR预期识别", ROOT / "reports/research/510300_lpr_expectation_identification_v1/summary.json", "LPR调查不是7天逆回购调查"),
    ("LPR预期扩源", ROOT / "reports/research/510300_lpr_expectation_source_extension_v2/summary.json", "固定扩源仍有缺月；保持当时已知和金额识别"),
    ("LPR预期探索训练", ROOT / "reports/research/510300_lpr_expectation_exploratory_training_v1/summary.json", "已实际拟合，候选往返0，未达目标，不能降低阈值制造交易"),
    ("PMI预期与需求组合", ROOT / "reports/research/510300_historical_index_pmi_expectation_v1/result.json", "14发布/11有共识，两冻结表达与四账户拒绝"),
    ("LPR股债联合反应", ROOT / "reports/research/510300_lpr_joint_response_v1/result.json", "固定毛优势初筛拒绝；不能改过滤器或窗口救援"),
]
FILES = [
    ROOT / "docs/510300_POLICY_ANNOUNCEMENT_SCORE_INTAKE_V1.md",
    CLOCK / "results/跨通道政策链_完整事实与时钟.csv",
    CLOCK / "results/全部官方目录记录.csv",
    CLOCK / "results/catalog_summary.json",
    POLICY / "inputs/policy_facts.csv", POLICY / "results/账户指标.csv",
    HIST / "七次降息与剩余历史收益.csv", HIST / "protocol.json",
    JOINT / "results/全部3488当时已知技术与宏观_未知保留.parquet",
    JOINT / "results/四原上涨反弹案例_全部同钟宏观与技术.parquet",
    JOINT / "results/全部事前配对模型预测与实际宏观路径.parquet",
    ROOT / "reports/research/510300_upward_episode_anatomy_v1/results/上涨段全集.parquet",
    ROOT / "reports/research/510300_lpr_expectation_source_extension_v2/ledger.json",
    *[p for _, p, _ in PRIOR],
    ROOT / "research/factor96_funding_relief_v1.py",
    ROOT / "research/macro_technical_first_passage_inputs_v1.py",
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.absolute().relative_to(ROOT).as_posix()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value, exclusive=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def export(name, frame):
    frame = frame.copy()
    for column in frame.select_dtypes(include="object"):
        frame[column] = frame[column].map(lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
    frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)


def freeze():
    if OUT.exists():
        raise FileExistsError("已有本轮登记或结果，不覆盖、不重跑。")
    for path in FILES:
        if not path.is_file():
            raise FileNotFoundError(path)
    (OUT / "results").mkdir(parents=True)
    (OUT / "code").mkdir()
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "protocol.json", {
        "study": "510300_POLICY_ANNOUNCEMENT_SCORE_INTAKE_V1", "at": now(),
        "registration_decision": "TECH.R193", "type": "SOURCE_AND_COMPLETE_USE_INTAKE_ONLY",
        "identities": {rel(p): digest(p) for p in FILES},
        "code_sha256": digest(Path(__file__)),
        "sample": "原3488日/61分段/49上涨/四案例240行；24政策节点、7降息与全部固定两年目录",
        "decision_time": "Asia/Shanghai 16:00；只接入来源上界不晚于决策的信息",
        "announcement_pairs": [["R01", "R02"], ["R03", "R04"]],
        "pair_description_dates": [["2024-09-20", "2024-09-30"], ["2025-05-06", "2025-05-12"]],
        "prior_record_count": len(PRIOR), "new_accounts": 0, "new_fits": 0,
        "new_return_labels": 0, "new_downloads": 0, "new_numeric_candidates": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "historical_first_version": "NOT_CERTIFIED",
        "financial_gate": "先有覆盖完整的新输入与实质不同完整用途，才能另立数值协议；本核对不新增政策分值",
        "independent_validation": False, "goal_achieved": False,
    })
    print("已登记TECH.R193：仅做来源与用途核对，尚未产生新连接结果。")


def annotate(frame, facts):
    out = frame.copy()
    out["date"] = pd.to_datetime(out.date).astype("datetime64[ns]")
    out["decision_time"] = out.date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
    known = pd.to_datetime(facts.source_available_upper, utc=True)
    out["catalog_period_contains_date"] = out.date.between("2024-01-01", "2025-12-31")
    out["complete_policy_population"] = False
    out["policy_expectation_status"] = "UNKNOWN_NO_MATCHED_PREANNOUNCEMENT_REPO_CONSENSUS"
    out["policy_score_admission"] = "NOT_ADMITTED_INCOMPLETE_POLICY_POPULATION"
    out["saved_node_ids_known_to_date"] = [
        "|".join(facts.node_id.loc[known.le(t.tz_convert("UTC"))]) for t in out.decision_time
    ]
    out["nodes_newly_known_since_previous_16h"] = [
        "|".join(facts.node_id.loc[known.le(t.tz_convert("UTC")) & known.gt(prior.tz_convert("UTC"))])
        for t, prior in zip(out.decision_time, out.decision_time.shift().fillna(out.decision_time.iloc[0]))
    ]
    out["announced_repo_target_percent"] = np.nan
    out["selected_rate_announcement_known_at"] = pd.NaT
    out["selected_rate_announcement_node"] = ""
    out["selected_rate_implementation_node"] = ""
    out["selected_rate_information_status"] = "NOT_COMPUTED_OUTSIDE_FIXED_PAIR_DESCRIPTION_DATES"
    out["announcement_cut_pp"] = np.nan
    for announced, implemented in [("R01", "R02"), ("R03", "R04")]:
        a, e = facts.set_index("node_id").loc[[announced, implemented]].to_dict("index").values()
        start = pd.Timestamp(a["source_available_upper"])
        finish = pd.Timestamp(e["source_available_upper"])
        end = pd.Timestamp("2024-09-30T16:00:00+08:00" if announced == "R01" else "2025-05-12T16:00:00+08:00")
        active = out.decision_time.between(start, end)
        pending = active & out.decision_time.lt(finish)
        confirmed = active & ~pending
        out.loc[active, "announced_repo_target_percent"] = float(e["amount"])
        out.loc[active, "selected_rate_announcement_known_at"] = start.tz_localize(None)
        out.loc[active, "selected_rate_announcement_node"] = announced
        out.loc[active, "selected_rate_implementation_node"] = implemented
        out.loc[pending, "selected_rate_information_status"] = "ANNOUNCED_TARGET_IMPLEMENTATION_NOT_YET_KNOWN"
        out.loc[confirmed, "selected_rate_information_status"] = "IMPLEMENTATION_CONFIRMED_NOT_A_NEW_CUT_SURPRISE"
        out.loc[active, "announcement_cut_pp"] = .2 if announced == "R01" else .1
    out["lagged_dr_gap_to_announced_target_pp"] = out.dr007 - out.announced_repo_target_percent
    out["original_gap_minus_target_gap_pp"] = out.funding_gap_pp - out.lagged_dr_gap_to_announced_target_pp
    return out


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise FileExistsError("已有核对执行记录，不再执行。")
    cfg = load(OUT / "protocol.json")
    for path, sha in cfg["identities"].items():
        if digest(ROOT / path) != sha:
            raise ValueError("登记资料已改变：" + path)
    if digest(Path(__file__)) != cfg["code_sha256"]:
        raise ValueError("登记后核对代码改变。")
    save(OUT / "RUN_STARTED.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json")})
    facts = pd.read_csv(CLOCK / "results/跨通道政策链_完整事实与时钟.csv").fillna("")
    checks = []
    for row in facts.to_dict("records"):
        path = CLOCK / row["source_path"]
        text = re.sub(r"\s+", "", BeautifulSoup(path.read_bytes(), "html.parser").get_text("", strip=True))
        clauses = json.loads(row["source_checks"])
        passed = digest(path) == row["source_sha256"] and all(re.sub(r"\s+", "", c) in text for c in clauses)
        if not passed:
            raise ValueError("原件或定位不符：" + row["node_id"])
        checks.append({"node_id": row["node_id"], "source_path": rel(path), "sha256": digest(path),
                       "source_checks": len(clauses), "status": "PASS_ARCHIVED_CONTENT_NOT_FIRST_VINTAGE"})
    assert len(facts) == 24 and facts.node_id.is_unique
    export("全部24政策节点_原文钟与未知", facts)
    export("全部24节点原件定位核对", pd.DataFrame(checks))
    catalog = pd.read_csv(CLOCK / "results/全部官方目录记录.csv")
    assert len(catalog) == 490
    export("全部490目录线索_不可充当决策钟", catalog)
    prior_rows = []
    for title, path, purpose in PRIOR:
        item = load(path)
        prior_rows.append({"title": title, "result_path": rel(path), "sha256": digest(path),
                           "status": item.get("status", item.get("stage_status")),
                           "conclusion": item.get("discovery", item.get("conclusion", "原结果结构见绑定文件")),
                           "complete_use_and_disposition": purpose, "is_new_independent_evidence": False})
    export("九份既有政策预期用途与原裁决", pd.DataFrame(prior_rows))
    daily = pd.read_parquet(JOINT / "results/全部3488当时已知技术与宏观_未知保留.parquet")
    daily = annotate(daily, facts)
    assert len(daily) == 3488 and daily.date.is_unique
    export("全部3488日_已公布状态与公告覆盖未知", daily)
    cases = pd.read_parquet(JOINT / "results/四原上涨反弹案例_全部同钟宏观与技术.parquet")
    case_ids = cases[["date", "original_episode_id"]].copy()
    case_ids["date"] = pd.to_datetime(case_ids.date).astype("datetime64[ns]")
    case_rows = case_ids.merge(daily, on="date", how="left", validate="many_to_one")
    assert len(case_rows) == 240
    export("四原案例240行_政策可知与未知全保留", case_rows)
    selected = daily.loc[daily.date.between("2024-09-20", "2024-09-30") | daily.date.between("2025-05-06", "2025-05-12")].copy()
    predictions = pd.read_parquet(JOINT / "results/全部事前配对模型预测与实际宏观路径.parquet")
    predictions["date"] = pd.to_datetime(predictions.date).astype("datetime64[ns]")
    selected = selected.merge(predictions[["date", "policy", "status", "score", "predicted_p_times_b", "entry_event", "macro_features_on_path"]],
                              on="date", how="left", validate="one_to_many")
    export("两条宣布实施链_原保存评分不修改", selected)
    waves = pd.read_parquet(ROOT / "reports/research/510300_upward_episode_anatomy_v1/results/上涨段全集.parquet")
    waves["confirm_up_date"] = pd.to_datetime(waves.confirm_up_date).astype("datetime64[ns]")
    joined = waves.merge(daily[["date", "catalog_period_contains_date", "complete_policy_population", "policy_expectation_status",
                               "policy_score_admission", "saved_node_ids_known_to_date", "selected_rate_information_status"]],
                         left_on="confirm_up_date", right_on="date", how="left", validate="many_to_one")
    joined["retrospective_episode_role"] = "DESCRIPTIVE_ONLY_NOT_TRAINING_OR_ENTRY_FEATURE"
    assert len(joined) == 61 and int(joined.admitted.sum()) == 49
    export("原61分段及49正式上涨_确认日政策覆盖", joined)
    events = pd.read_csv(HIST / "七次降息与剩余历史收益.csv")
    assert len(events) == 7
    events["return_role"] = "REUSED_OLD_FIXED_5_20_DAY_EVENT_EXAMPLES_NOT_FULL_ACCOUNT_NOT_NEW_SAMPLE"
    export("七次降息全部旧结果与反例_不重跑", events)
    lpr = load(ROOT / "reports/research/510300_lpr_expectation_source_extension_v2/ledger.json")
    export("全部原LPR调查槽_不得转交逆回购预期", pd.DataFrame(lpr))
    summary = {"at": now(), "study": cfg["study"], "registration_decision": "TECH.R193", "decision": "TECH.R194",
               "status": "COMPLETED_BOUNDED_POLICY_INTAKE_NOT_ADMITTED_FOR_NEW_SCORE",
               "daily_origins": len(daily), "original_case_rows": len(case_rows), "original_segments": len(waves),
               "admitted_episodes": int(waves.admitted.sum()),
               "source_nodes_checked": len(checks), "catalog_leads": len(catalog), "prior_use_records": len(prior_rows),
               "repo_policy_consensus_nodes": int(facts.expectation_status.eq("QUALIFIED_PRE_RELEASE").sum()),
               "policy_population_complete": False, "catalog_event_clock_is_tradable": False,
               "source_contract_error_identified": False,
               "different_information_exists": True, "different_complete_numeric_use_admitted": False,
               "new_accounts": 0, "new_model_fits": 0, "new_return_labels": 0, "new_downloads": 0,
               "new_numeric_candidates": 0, "net_sharpe": None, "net_cagr": None,
               "new_financial_metrics_status": "NOT_COMPUTED", "goal_achieved": False,
               "independent_validation": False, "overfitting_removed": False,
               "admission_reason": "预告目标与实际资金状态确为不同信息；现有24节点为选择链、并非完整政策母集，逆回购事前共识没有。LPR调查不能替代。旧政策20日与预期用途失败保持，不能按2024行情追加分值救原树。",
               "next_action": "关闭本提案的数值准入；先补可覆盖母集/原件/同工具事前预期合同或真正新样本，成立后另行登记；不自动登记另一政策变体。"}
    save(OUT / "summary.json", summary)
    print("TECH.R194已完成：24原件/490目录/3488日与原240案例连接；没有准入新政策评分。")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="政策公告评分输入的本地有限核对")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
