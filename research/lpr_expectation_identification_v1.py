"""识别保存调查中的点值、区间与政策兑现差异；不读取证券行情。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "reports/research/510300_lpr_prior_expectations_v1"
OUT = ROOT / "reports/research/510300_lpr_expectation_identification_v1"
FILES = {
    "actual.csv": "facts/全部84月实际公告母集.csv",
    "survey_facts.csv": "facts/人工核对的分期限调查事实.csv",
    "candidates.csv": "facts/事前调查候选目录.csv",
    "parent_source_plan.json": "source_plan.json",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: str) -> float | None:
    return None if value == "" else float(value)


def initialize() -> None:
    require(not OUT.exists(), "本轮目录已存在，不能覆盖已保存资料。")
    (OUT / "inputs").mkdir(parents=True)
    inputs = {}
    for name, source in FILES.items():
        data = (PARENT / source).read_bytes()
        (OUT / "inputs" / name).write_bytes(data)
        inputs[name] = {"source": str(PARENT / source), "sha256": hashlib.sha256(data).hexdigest()}
    mandate = (ROOT / "config/510300_sparse_opportunity_mandate_v1.json").read_bytes()
    (OUT / "mandate.json").write_bytes(mandate)
    write_json(OUT / "protocol.json", {
        "study_id": "510300_LPR_EXPECTATION_IDENTIFICATION_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "SAVED_SOURCE_IDENTIFICATION_ONLY_NO_EQUITY_OUTCOMES",
        "inputs": inputs,
        "known_before_this_protocol": "已查看49条保存调查的数值、对应原文段落和84个月实际报价；已查看点值识别计数和三个月点值偏差。未查看任何新增证券收益标签。此协议不宣称数值发现为盲测。",
        "scope": "利用已有保存资料逐期限识别调查票数比例、报价中位数及其区间；保留全部84个月，不补选缺失月份，不更改父级原表。",
        "clock": "调查发布和当前版本修改时间均必须早于当月官方公告时钟；网页历史首版未独立认证。使用官方母表available_at，忽略人工备注中的旧09:15示例。",
        "definitions": {
            "actual_change_bp": "本月官方LPR减上月官方LPR，百分数差乘100；首月不可定义。",
            "cut_vote_share": "预计该期限下调人数除调查总人数；无法唯一识别则保留上下界，不解释为风险中性概率。",
            "cut_vote_realization_gap": "实际是否下调减事前下调票数比例；区间按反向端点计算。",
            "median_change_bp": "仅继承原文能推出的中位数或中位数区间；不取区间中点。",
            "surprise_bp": "实际调整bp减调查预期调整bp；负值为调整更偏宽松，正值为更偏紧。只表示兑现差异，不是结构性外生冲击。",
            "interval_direction": "偏差上界小于0则整段偏宽松，下界大于0则整段偏紧，两个端点均0则一致，其余为方向未唯一识别。只要任一期限区间缺失，即保持该期限缺失。",
        },
        "excluded_activity": "不建立预测模型、择时阈值、账户、退出规则或未来收益评价；不据本描述设定通过门槛。",
        "next_stage": "资料补齐或新收集后，另行冻结与价格基准比较的预测协议；本轮不把单月政策兑现差异升级成买入信号。",
        "new_equity_labels": 0, "new_models": 0, "new_accounts": 0,
        "net_sharpe_target": 1.5, "maximum_drawdown_target": 0.1, "capital_cny": 200000,
    })


def compute(root: Path) -> tuple[list[dict], dict]:
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8-sig"))
    for name, identity in protocol["inputs"].items():
        require(hashlib.sha256((root / "inputs" / name).read_bytes()).hexdigest() == identity["sha256"], f"输入身份改变：{name}")
    actual = read_csv(root / "inputs/actual.csv")
    facts = read_csv(root / "inputs/survey_facts.csv")
    candidates = read_csv(root / "inputs/candidates.csv")
    fact_by_month = {r["month"]: r for r in facts}
    candidate_by_uid = {r["uid"]: r for r in candidates}
    require(len(fact_by_month) == len(facts), "调查月份重复。")
    require(len(candidate_by_uid) == len(candidates), "来源目录身份重复。")
    expected = [f"{y:04d}-{m:02d}" for y in range(2019, 2027) for m in range(1, 13) if "2019-08" <= f"{y:04d}-{m:02d}" <= "2026-07"]
    require([r["month"] for r in actual] == expected, "84月母集不完整或顺序不符。")
    ledger = []
    for index, event in enumerate(actual):
        month = event["month"]
        row = dict(month=month, announcement_at=event["available_at"], official_url=event["source_url"], official_sha256=event["raw_sha256"])
        fact = fact_by_month.get(month)
        if fact is not None:
            source = candidate_by_uid[fact["uid"]]
            announcement = datetime.fromisoformat(event["available_at"])
            require(datetime.fromisoformat(source["published_at"]) < announcement and datetime.fromisoformat(source["modified_at"]) < announcement, f"调查时钟不合格：{month}")
            require(source["month"] == month and source["candidate_status"] == "CANDIDATE_PRE_RELEASE_POLL", "调查月份或来源资格不符。")
            later = [c for c in candidates if c["month"] == month and c["published_at"] > source["published_at"]]
            require(not later, f"存在更晚事前候选，须先核对版本：{month}")
            row.update(source_uid=fact["uid"], survey_url=source["source_url"], survey_published_at=source["published_at"], survey_modified_at=source["modified_at"], source_page_sha256=source["page_sha256"], source_content_sha256=source["content_sha256"], respondents=int(fact["n"]), original_explanation=fact["explanation"], status="ADMITTED_SAVED_HISTORICAL_SURVEY")
        else:
            row.update(status="NO_ADMITTED_NUMERIC_SURVEY_IN_SAVED_SOURCE_SET")
        for tenor in ("1", "5"):
            value = Decimal(event[f"lpr_{tenor}y_percent"])
            change = None if index == 0 else float((value - Decimal(actual[index - 1][f"lpr_{tenor}y_percent"])) * 100)
            prefix = f"t{tenor}_"
            row[prefix + "actual_change_bp"] = change
            if fact is None:
                row[prefix + "status"] = "NOT_IDENTIFIED_MISSING_SURVEY"
                continue
            n = row["respondents"]
            low, high = int(fact[f"cut{tenor}_min"]), int(fact[f"cut{tenor}_max"])
            hold = int(fact[f"hold{tenor}_min"])
            require(n > 0 and 0 <= low <= high <= n and high + hold <= n, f"票数上下界不闭合：{month}/{tenor}")
            median_low = number(fact[f"median{tenor}_min_bp"])
            median_high = number(fact[f"median{tenor}_max_bp"])
            require(median_low is None or median_high is None or median_low <= median_high, "中位数区间端点相反。")
            surprise_low = None if median_high is None or change is None else round(change - median_high, 10)
            surprise_high = None if median_low is None or change is None else round(change - median_low, 10)
            direction = "NOT_UNIQUELY_IDENTIFIED"
            if surprise_high is not None and surprise_high < 0:
                direction = "EASIER_THAN_IDENTIFIED_MEDIAN_RANGE"
            elif surprise_low is not None and surprise_low > 0:
                direction = "TIGHTER_THAN_IDENTIFIED_MEDIAN_RANGE"
            elif surprise_low == 0 and surprise_high == 0:
                direction = "MATCHES_IDENTIFIED_MEDIAN"
            row.update({
                prefix + "status": "IDENTIFIED_WITH_PRESERVED_BOUNDS", prefix + "cut_votes_min": low, prefix + "cut_votes_max": high,
                prefix + "cut_vote_share_min": low / n, prefix + "cut_vote_share_max": high / n,
                prefix + "cut_vote_share_exact": low == high,
                prefix + "cut_vote_realization_gap_min": None if change is None else int(change < 0) - high / n,
                prefix + "cut_vote_realization_gap_max": None if change is None else int(change < 0) - low / n,
                prefix + "median_change_min_bp": median_low, prefix + "median_change_max_bp": median_high,
                prefix + "median_change_exact": median_low is not None and median_low == median_high,
                prefix + "surprise_min_bp": surprise_low, prefix + "surprise_max_bp": surprise_high,
                prefix + "surprise_direction": direction,
            })
        ledger.append(row)
    counts = {}
    for tenor in ("1", "5"):
        prefix = f"t{tenor}_"
        counts[prefix + "exact_vote_months"] = sum(r.get(prefix + "cut_vote_share_exact", False) for r in ledger)
        counts[prefix + "exact_median_months"] = sum(r.get(prefix + "median_change_exact", False) for r in ledger)
    for item in ("cut_vote_share_exact", "median_change_exact"):
        counts["both_tenors_" + item] = sum(all(r.get(f"t{t}_" + item, False) for t in ("1", "5")) for r in ledger)
    easier = [r["month"] for r in ledger if any(r.get(f"t{t}_surprise_direction") == "EASIER_THAN_IDENTIFIED_MEDIAN_RANGE" for t in ("1", "5"))]
    tighter = [r["month"] for r in ledger if any(r.get(f"t{t}_surprise_direction") == "TIGHTER_THAN_IDENTIFIED_MEDIAN_RANGE" for t in ("1", "5"))]
    summary = {
        "study_id": "510300_LPR_EXPECTATION_IDENTIFICATION_V1", "status": "COMPLETED_SAVED_SOURCE_IDENTIFICATION_NO_PREDICTIVE_VIEW",
        "universe_months": len(actual), "saved_survey_months": len(facts), "missing_months": len(actual) - len(facts),
        "counts": counts, "at_least_one_tenor_robust_easier_months": easier, "at_least_one_tenor_robust_tighter_months": tighter,
        "direction_scope": "仅基于已保存49个月及原表数值约束，不代表完整84个月没有其他冲击，也不代表沪深300方向。",
        "new_equity_labels": 0, "new_models": 0, "new_accounts": 0, "strict_forward_events": 0,
        "net_sharpe": "NOT_COMPUTED", "max_drawdown": "NOT_COMPUTED", "account_stage": "NOT_RUN_NO_NEW_PREDICTION_PROTOCOL",
        "whole_goal_achieved": False, "goal_status": "ACTIVE", "position_impact": 0,
    }
    return ledger, summary


def save_results(root: Path) -> None:
    ledger, summary = compute(root)
    write_json(root / "ledger.json", ledger)
    write_json(root / "summary.json", summary)
    with (root / "全部84月事前预期识别.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(k for r in ledger for k in r)))
        writer.writeheader()
        writer.writerows(ledger)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只复算LPR保存来源的可识别范围，不读取510300收益")
    parser.add_argument("action", choices=["initialize", "compute", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == "initialize":
        initialize()
    elif args.action == "compute":
        save_results(args.root)
    else:
        ledger, summary = compute(args.root)
        require(ledger == json.loads((args.root / "ledger.json").read_text(encoding="utf-8")), "保存台账与复算不同。")
        require(summary == json.loads((args.root / "summary.json").read_text(encoding="utf-8")), "保存汇总与复算不同。")
        displayed = read_csv(args.root / "全部84月事前预期识别.csv")
        require(len(displayed) == len(ledger), "CSV月份数量不一致。")
        for saved, derived in zip(displayed, ledger):
            require(all(value == ("" if derived.get(key) is None else str(derived[key])) for key, value in saved.items()), "CSV展示内容与台账不一致。")
        print(json.dumps({"status": "PASS_SAVED_FACT_IDENTIFICATION_RECOMPUTATION", "months": len(ledger), "equity_outcomes_read": 0}, ensure_ascii=False))
