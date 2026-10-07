"""将原65来源映射至共同评分和真实账户原点，检验不同催化用途的来源条件。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import all_factor_joint_path_attribution_v1 as saved

OUT = ROOT / "reports/research/510300_all_factor_catalyst_source_alignment_v1"
OLD_REVIEW = ROOT / "reports/research/510300_source_expectation_transmission_review_v1"
NODE_DIR = ROOT / "reports/research/510300_support_price_acceptance_v1/results"
QUALIFICATION = NODE_DIR / "全部65来源用途资格_操作前值与非激活保留.parquet"
FIRST = NODE_DIR / "全部来源首次原ETF可知_宣布与确认分开.parquet"
CASES = ROOT / "reports/research/510300_all_factor_joint_scorecard_v1/results/全部30历史案例_联合解释分与覆盖.parquet"
CREDIT = ROOT / "reports/research/510300_policy_expectation_thesis_exit_v1/implementation_v1_0_1/results/全部84月_双期限兑现与缺失不删.parquet"
OLD_TRACE = ROOT / "reports/research/510300_original_policy_announcement_trace_v1/implementation_v1_0_1/summary.json"
ROUTES = saved.OUT / "results/全部实际决策_质量门持仓风险与成交.parquet"
PURPOSE = saved.OUT / "next_source_purpose.json"
PRIMARY = saved.PRIMARY


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def normalized_calendar(frame):
    calendar = frame[["date", "decision_at"]].copy().reset_index(drop=True)
    calendar["date"] = pd.to_datetime(calendar.date).astype("datetime64[ns]")
    cuts = pd.DatetimeIndex(pd.to_datetime(calendar.decision_at, utc=True)).as_unit("ns")
    saved.require(calendar.date.is_unique and calendar.date.is_monotonic_increasing, "共同原日历重复或不顺序。")
    local = cuts.tz_convert("Asia/Shanghai")
    saved.require(local.tz_localize(None).normalize().equals(pd.DatetimeIndex(calendar.date)), "决策钟与实际ETF日期不同。")
    saved.require(bool(np.all((local.hour == 15) & (local.minute == 5) & (local.second == 0))), "共同决策不是15:05。")
    saved.require(cuts.is_monotonic_increasing and not cuts.hasnans, "共同决策钟为空或不顺序。")
    return calendar, cuts


def align_nodes(frame, nodes):
    """按真实日历搜索首次可知；保留原动作与未知钟，绝不推定无政策。"""
    calendar, cuts = normalized_calendar(frame)
    out = nodes.copy().reset_index(drop=True)
    saved.require(out.node_id.is_unique and out.common_source_id.notna().all(), "原节点或共同来源身份不完整。")
    public = pd.to_datetime(out.source_available_upper, utc=True)
    out["first_joint_origin"] = pd.NaT
    out["first_joint_origin"] = out.first_joint_origin.astype("datetime64[ns]")
    out["joint_mapping_status"] = "UNKNOWN_PUBLICATION_CLOCK"
    out["first_joint_index"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for i, timestamp in enumerate(public):
        if pd.isna(timestamp):
            continue
        index = int(cuts.searchsorted(timestamp, side="left"))
        if index >= len(calendar):
            out.loc[i, "joint_mapping_status"] = "AFTER_SAVED_CALENDAR"
            continue
        out.loc[i, "first_joint_index"] = index
        out.loc[i, "first_joint_origin"] = calendar.date.iloc[index]
        out.loc[i, "joint_mapping_status"] = "PRE_CALENDAR_KNOWN" if timestamp < cuts[0] else "FIRST_KNOWN_BY_JOINT_CUTOFF"
        saved.require(timestamp <= cuts[index], "公告在决策后仍被提前使用。")
        if index:
            saved.require(timestamp > cuts[index - 1], "首次可知不是首个合格原点。")
    out["first_joint_decision_at"] = out.first_joint_index.map(lambda value: cuts[int(value)].tz_convert("Asia/Shanghai") if pd.notna(value) else pd.NaT)
    out["complete_policy_calendar"] = "NOT_ESTABLISHED"
    out["is_independent_evidence_increment"] = False
    out["same_instrument_numeric_expectation"] = "NOT_ESTABLISHED_IN_NODE_SET"
    return out


def daily_recorded_view(frame, aligned):
    calendar, cuts = normalized_calendar(frame)
    public = pd.to_datetime(aligned.source_available_upper, utc=True)
    output = []
    for index, (date, decision_at) in enumerate(calendar.itertuples(index=False, name=None)):
        known = aligned.loc[public.le(cuts[index])]
        if index == 0:
            fresh = aligned.iloc[:0]
            fresh_status = "UNKNOWN_PRIOR_DECISION_BEFORE_SAVED_CALENDAR"
            fresh_count = None
        else:
            fresh = aligned.loc[public.gt(cuts[index - 1]) & public.le(cuts[index])]
            fresh_status = "NEW_RECORD_IN_FIXED_CATALOG" if len(fresh) else "NO_NEW_RECORD_IN_FIXED_CATALOG_NOT_NO_POLICY"
            fresh_count = len(fresh)
        announcement = fresh.loc[fresh.information_role.eq("ANNOUNCEMENT")]
        support = fresh.loc[fresh.registered_action.eq("RECORDED_SUPPORT_INFORMATION")]
        output.append({"date": date, "decision_at": decision_at, "recorded_known_nodes": len(known),
            "recorded_known_common_sources": known.common_source_id.nunique(),
            "recorded_new_nodes": fresh_count, "recorded_new_status": fresh_status,
            "recorded_new_node_ids": "|".join(fresh.node_id),
            "recorded_new_common_source_count": fresh.common_source_id.nunique() if index else None,
            "recorded_new_economic_identity_count": fresh.economic_identity.nunique() if index else None,
            "recorded_new_announcement_nodes": len(announcement) if index else None,
            "recorded_new_announcement_common_sources": announcement.common_source_id.nunique() if index else None,
            "recorded_new_support_nodes": len(support) if index else None,
            "recorded_new_actions": "|".join(sorted(set(fresh.registered_action))),
            "complete_policy_calendar": "NOT_ESTABLISHED", "all_policy_absence_known": False,
            "source_count_is_confidence": False, "numeric_catalyst_score": "NOT_COMPUTED_NOT_ADMITTED"})
    return pd.DataFrame(output)


def bind_scores(daily, prediction):
    prediction = prediction.copy()
    prediction["date"] = pd.to_datetime(prediction.date).astype("datetime64[ns]")
    saved.require(not prediction.duplicated(["date", "policy"]).any(), "共同模型原点评分重复。")
    out = prediction.merge(daily, on="date", how="left", validate="many_to_one")
    saved.require(len(out) == len(prediction) and out.complete_policy_calendar.notna().all(), "评分原点丢失或未知被删除。")
    return out


def family_rows():
    archived = saved.read(OLD_REVIEW / "saved_family_adjudications.json")
    rows = []
    for record in archived["families"]:
        path = ROOT / record["path"]
        actual = saved.read(path) if path.exists() else {}
        status = actual.get("status", actual.get("account_status", actual.get("research_round_status", "STATUS_NOT_EXPLICIT_AT_TOP_LEVEL")))
        rows.append({"name": record["name"], "path": record["path"], "saved_adjudication_status": record["status"],
            "current_file_exists": path.exists(), "current_top_level_status": str(status),
            "sha256": saved.sha(path) if path.exists() else None, "new_independent_evidence": False,
            "new_numeric_admission": False, "disposition": "保留各原用途的实际结果和拒绝；同资料不因改名再拟合。"})
    return pd.DataFrame(rows)


def freeze():
    purpose = saved.read(PURPOSE)
    saved.require(purpose["status"] == "NOT_ADMITTED_SOURCE_COVERAGE_AND_EXPECTATIONS_UNPROVEN", "来源用途状态改变。")
    paths = [PURPOSE, QUALIFICATION, FIRST, CASES, CREDIT, OLD_TRACE, ROUTES, saved.SCORES, saved.CONTEXT,
        OLD_REVIEW / "summary.json", OLD_REVIEW / "saved_family_adjudications.json",
        ROOT / "research/all_factor_joint_path_attribution_v1.py", ROOT / "tests/test_all_factor_catalyst_source_alignment_v1.py"]
    archived = saved.read(OLD_REVIEW / "saved_family_adjudications.json")
    paths += [ROOT / row["path"] for row in archived["families"] if (ROOT / row["path"]).exists()]
    saved.write(OUT / "protocol.json", {
        "study_id": "510300_ALL_FACTOR_CATALYST_SOURCE_ALIGNMENT_V1", "registered_at": saved.now(),
        "registration": "TECH.R259", "decision": "TECH.R260", "code_sha256": saved.sha(__file__),
        "sources": [{"path": p.absolute().relative_to(ROOT).as_posix(), "sha256": saved.sha(p)} for p in paths],
        "prior_results_observed": True, "web_reference_opened_before_registration": 1,
        "hypothesis": "旧来源是否能提供原联合模型未表示的进场前催化信息，同时满足完整覆盖和同工具预期；不预设来源门通过。",
        "scope": {"daily_origins": 3488, "original_source_nodes": 65, "original_cases": 30,
            "saved_prediction_rows": 13952, "saved_decision_rows": 22856, "old_families": 8, "credit_months": 84},
        "clock": "15:05，按原完整ETF日历映射上界；不得将夜间/周末/次日公布提前到前收盘。",
        "identity": "65节点、原动作与经济身份保留；共同来源去重数不算独立票；原16:00观察与现15:05分开。",
        "unknown": "没有新记录只说明固定目录无新增；不能推断无政策、无动作或预期为零。第一原点之前没有前决策，新增计数未知。",
        "expectation": "已有双期限LPR调查不自动匹配降准/逆回购；既有共识失败不重训，0新增同工具事前数量预期。",
        "source_admission": "来源范围/时间/同工具预期/不同完整用途不具备，则仅保存描述并关闭此原源评分用途。",
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "new_downloads": 0, "new_numeric_configurations": 0,
        "not_allowed": purpose["not_allowed"], "stop_condition": purpose["stop_condition"],
        "goal_achieved": False, "orders_authorized": False,
    }, exclusive=True)
    print("已冻结65来源、3488原点、30案例及全部保存评分/决策的来源匹配；0新模型。")


def report(summary, nodes, daily, paired, cases, gate):
    lines = ["# 新催化来源与全因素共同评分：完整原点匹配及用途裁定", "",
        "本轮是R258声明的有限来源实验。原65节点、原角色/资格、30案例、3488日、13952评分和22856实际决策全部保留，没有再下载同来源、拟合或重跑账户。原金融仍为R256固定拒绝。", "",
        "## 来源门与结果", "", "| 来源条件 | 当前证据 | 处理 |", "|---|---|---|"]
    for row in gate.to_dict("records"):
        lines.append(f"| {row['condition']} | {row['actual_evidence']} | {row['decision']} |")
    lines += ["", f"65节点来自{summary['common_source_count']}共同来源、{summary['economic_identity_count']}保存经济身份；不是65个独立冲击。宣布节点{summary['announcement_nodes']}，共同原文{summary['announcement_common_sources']}；原支持资格动作{summary['support_information_nodes']}，实际操作前值较低/较高、转载与非激活记录仍单列。",
        "", f"重新按15:05映射后，{summary['source_nodes_shifted_from_original_1600']}节点的首次原点晚于原16:00观察。完整3488日中有{summary['origins_with_recorded_arrival']}原点出现目录新记录、{summary['origins_with_recorded_announcement_arrival']}原点出现宣布节点；首行之前决策未知，不计为新消息。所有日期的完整政策无动作状态都未认证。", "",
        "## 每个共同模型实际可评分区间的宣布原点", "",
        "| 模型 | 可评分原点 | 同日目录新记录原点 | 同日宣布原点 | 同日支持资格原点 |", "|---|---:|---:|---:|---:|"]
    for row in summary["model_source_coverage"]:
        lines.append(f"| {row['policy']} | {row['available_origins']} | {row['available_with_recorded_arrival']} | {row['available_with_announcement_arrival']} | {row['available_with_support_arrival']} |")
    lines += ["", "‘同日宣布原点’是当前已保存目录的可知记录，不是完整事件分母，更没有事前超预期标签；不能据这几个原点建立稳定的新树或解释全部交易。", "",
        "## 全部记录中落在主模型可评分原点的宣布", "",
        "| 节点 | 原公布上界 | 15:05首次原点 | 共同来源 | 原动作 | 当时主模型质量门 |", "|---|---|---|---|---|---|"]
    for row in summary["available_announcement_nodes"]:
        lines.append(f"| {row['node_id']} | {row['source_available_upper']} | {row['first_joint_origin']} | {row['common_source_id'][:12]} | {row['registered_action']} | {row['candidate_quality_pass']} |")
    lines += ["", "2024/9/24公开的降准和政策利率信息在原点前可知；联合模型没有专门的该催化字段。央视当天09:08:37的发布会报道本轮再次打开核对，它只是案例复核，不是新增历史首版回执或完整事件台账：[原报道](https://news.cctv.cn/2024/09/24/ARTIwQJC9xG5fTUjpGt1K4g2240924.shtml)。",
        "2025/5/7的记录不能前移到5/6收盘，即使5/7开盘前已经可知。当前冻结策略的决定时点是前收盘；不改为临开盘重新决策以救结果。", "",
        "## 预期与既有研究的边界", "",
        "84月双期限LPR记录保留实际公布、同工具事前调查和全部缺失；LPR预期不能替代逆回购或降准预期。来源表不新增同工具预测，旧77月共识和非线性预期差等八家族的失败/未知按实际结果保留，不能因‘催化×阶段’的新名称重新拟合。", "",
        "假设→原联合模型遗漏公开催化，而旧资料是否足以建立不同可检验用途。",
        "方法→来源上界映射原15:05日历；共同来源/经济身份与动作分开；绑定全部共同评分、原30案例及实际决策；完整分母未知保持未知。",
        "结果→来源允许解释若干具体点位，尚不足以构造无动作/预期差和全样本政策阶段。",
        "接受/拒绝理由→接受全原点时间与来源对应为描述证据；拒绝以这批旧来源直接开启新数值评分，新增独立来源0，金融用途NOT_ADMITTED。",
        "重新验证→只有真正新增且有完整采样规则的来源、不同完整机制，或原协议下实际到达的新独立点位，才重新检验；不反复盘点此65目录、补零或调旧参数。",
        "", "本来源实验已经终止并归档。目标收益、净Sharpe与独立验证仍未达；本轮没有任何新策略金融收益。"]
    (OUT / "催化来源_完整共同原点与准入结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    protocol = saved.read(OUT / "protocol.json")
    saved.require(protocol["code_sha256"] == saved.sha(__file__), "登记后来源匹配代码改变。")
    for source in protocol["sources"]:
        saved.require(saved.sha(ROOT / source["path"]) == source["sha256"], "来源匹配固定输入改变。")
    saved.write(OUT / "run_started.json", {"at": saved.now(), "new_fits": 0, "new_accounts": 0}, exclusive=True)
    try:
        raw, first = pd.read_parquet(QUALIFICATION), pd.read_parquet(FIRST)
        saved.require(len(raw) == 65 and len(first) == 65, "原65节点母集改变。")
        nodes = raw.merge(first[["node_id", "first_original_decision_date", "actual_information_role"]], on="node_id", validate="one_to_one")
        calendar = pd.read_parquet(saved.CONTEXT)
        saved.require(len(calendar) == 3488, "原共同日历改变。")
        aligned = align_nodes(calendar, nodes)
        aligned["first_original_decision_date"] = pd.to_datetime(aligned.first_original_decision_date).astype("datetime64[ns]")
        aligned["first_origin_shifted_later_than_original_1600"] = aligned.first_joint_origin.gt(aligned.first_original_decision_date)
        daily = daily_recorded_view(calendar, aligned)
        prediction = pd.read_parquet(saved.SCORES)
        paired = bind_scores(daily, prediction)
        cases = pd.read_parquet(CASES).merge(daily.drop(columns="decision_at"), on="date", how="left", validate="one_to_one")
        saved.require(len(cases) == 30 and cases.complete_policy_calendar.notna().all(), "原全部案例丢失。")
        routes = pd.read_parquet(ROUTES)
        routes["origin"] = pd.to_datetime(routes.origin).astype("datetime64[ns]")
        bound_routes = routes.merge(daily.rename(columns={"date": "origin", "decision_at": "catalyst_decision_at"}), on="origin", how="left", validate="many_to_one")
        saved.require(len(bound_routes) == 22856 and bound_routes.complete_policy_calendar.notna().all(), "实际请求绑定不完整。")
        credit, families = pd.read_parquet(CREDIT), family_rows()
        old_review = saved.read(OLD_REVIEW / "summary.json")
        old_trace = saved.read(OLD_TRACE)
        saved.require(old_review["new_same_tool_prerelease_numeric_forecasts"] == 0, "旧同工具预期来源事实改变。")
        saved.require(old_trace["complete_policy_news_coverage"] == "NOT_ESTABLISHED", "原完整政策覆盖事实改变。")
        available = paired.status.eq("AVAILABLE")
        coverage = []
        for policy, group in paired.groupby("policy", sort=False):
            available_group = group[group.status.eq("AVAILABLE")]
            coverage.append({"policy": policy, "available_origins": len(available_group),
                "available_with_recorded_arrival": int(available_group.recorded_new_nodes.gt(0).sum()),
                "available_with_announcement_arrival": int(available_group.recorded_new_announcement_nodes.gt(0).sum()),
                "available_with_support_arrival": int(available_group.recorded_new_support_nodes.gt(0).sum())})
        main = prediction[prediction.policy.eq(PRIMARY)][["date", "status", "candidate_quality_pass", "score", "macro_fields_on_path", "earnings_fields_on_path"]]
        announced = aligned[aligned.information_role.eq("ANNOUNCEMENT")].merge(main, left_on="first_joint_origin", right_on="date", how="left", validate="many_to_one")
        gates = pd.DataFrame([
            {"condition": "原日历/原点时间", "actual_evidence": "65节点均按原15:05日历映射，保留原16:00对应、夜间/周末延后与原动作。", "decision": "PASS_DESCRIPTION"},
            {"condition": "完整事件与无动作分母", "actual_evidence": "原来源追溯明确NOT_ESTABLISHED；无新目录记录不能判无政策。", "decision": "NOT_ADMITTED"},
            {"condition": "同工具事前预期", "actual_evidence": "本轮新增同工具数量预期0；84月LPR不是逆回购/降准调查。", "decision": "NOT_ADMITTED"},
            {"condition": "与既有用途不同的完整依据", "actual_evidence": "旧65节点与八家族重新对应新共同评分；没有新增独立来源，不能以名称变更救旧用途。", "decision": "NOT_ADMITTED"},
            {"condition": "历史首版/新独立验证", "actual_evidence": "当前网页历史首版均未独立认证，真实新完成点位0。", "decision": "NOT_ESTABLISHED"},
        ])
        summary = {"study_id": protocol["study_id"], "completed_at": saved.now(), "decision": "TECH.R260",
            "status": "COMPLETED_ALL_JOINT_ORIGIN_SOURCE_ALIGNMENT_OLD_CATALYST_NUMERIC_USE_NOT_ADMITTED",
            "all_daily_origins": len(daily), "all_source_nodes": len(aligned), "all_case_rows": len(cases),
            "all_score_rows": len(paired), "all_actual_decision_rows": len(bound_routes),
            "common_source_count": int(aligned.common_source_id.nunique()), "economic_identity_count": int(aligned.economic_identity.nunique()),
            "announcement_nodes": int(aligned.information_role.eq("ANNOUNCEMENT").sum()),
            "announcement_common_sources": int(aligned.loc[aligned.information_role.eq("ANNOUNCEMENT"), "common_source_id"].nunique()),
            "support_information_nodes": int(aligned.registered_action.eq("RECORDED_SUPPORT_INFORMATION").sum()),
            "source_nodes_shifted_from_original_1600": int(aligned.first_origin_shifted_later_than_original_1600.sum()),
            "origins_with_recorded_arrival": int(daily.recorded_new_nodes.gt(0).sum()),
            "origins_with_recorded_announcement_arrival": int(daily.recorded_new_announcement_nodes.gt(0).sum()),
            "model_source_coverage": coverage,
            "available_announcement_nodes": announced[announced.status.eq("AVAILABLE")][["node_id", "source_available_upper", "first_joint_origin", "common_source_id", "registered_action", "candidate_quality_pass", "score"]].to_dict("records"),
            "complete_policy_calendar": "NOT_ESTABLISHED", "verified_all_policy_absence_origins": 0,
            "new_same_tool_prerelease_numeric_forecasts": 0, "old_credit_months_preserved": len(credit),
            "old_source_families_preserved": len(families), "new_independent_sources": 0,
            "new_fits": 0, "new_accounts": 0, "new_labels": 0, "new_downloads": 0, "new_numeric_configurations": 0,
            "new_independent_completed_points": 0, "financial_admission": "NOT_ADMITTED_SOURCE_COVERAGE_AND_EXPECTATIONS_UNPROVEN",
            "source_purpose_terminal": True, "repeat_same_source_scan_allowed": False,
            "new_strategy_CAGR": "NOT_COMPUTED_DIAGNOSTIC", "new_strategy_Sharpe": "NOT_COMPUTED_DIAGNOSTIC",
            "actual_latest_financial_decision": "TECH.R256", "independent_validation": "NOT_ESTABLISHED",
            "goal_achieved": False, "orders_authorized": False}
        for name, frame in [
            ("全部65原节点_15时05分首次可知与原动作", aligned), ("全部3488原点_有记录与全覆盖未知分开", daily),
            ("全部13952共同评分_催化记录与未知", paired), ("原全部30案例_催化钟与解释等级不变", cases),
            ("全部22856实际决策_催化当时可知", bound_routes), ("全部84月LPR_原预期与缺失不替代工具", credit),
            ("全部八旧家族_用途与实际状态", families), ("全部来源门_接受描述拒绝新数值用途", gates),
            ("全部宣布节点_共同源及模型原点", announced)]:
            table(name, frame)
        saved.require(all(saved.sha(ROOT / source["path"]) == source["sha256"] for source in protocol["sources"]), "原来源在绑定中改变。")
        saved.write(OUT / "summary.json", summary, exclusive=True)
        report(summary, aligned, daily, paired, cases, gates)
        saved.write(OUT / "run_completed.json", {"at": saved.now(), "terminal": True, "financial_admission": summary["financial_admission"]}, exclusive=True)
        print(f"全3488日、65来源、13952评分、22856请求已匹配；完整催化/预期来源门未通过，0新金融。")
    except Exception as error:
        saved.write(OUT / "implementation_failure.json", {"at": saved.now(), "terminal": True, "type": type(error).__name__, "error": str(error)}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="催化来源与已保存全因素模型的完整对应")
    parser.add_argument("action", choices=["freeze", "run"])
    action = parser.parse_args().action
    freeze() if action == "freeze" else run()


if __name__ == "__main__":
    main()
