"""把固定央行报告完整范围对应至全部原点与原30案例，不产生交易分数。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from research.pbc_report_text_source_v1 import EXPECTED, first_known_origins

ROOT = Path(__file__).absolute().parents[1]
SOURCE = ROOT / "reports/research/510300_pbc_report_text_source_v1"
OUT = SOURCE / "case_binding"
CALENDAR = ROOT / "reports/research/510300_all_factor_macro_earnings_joint_v1/results/全部3488共同源视图_不足保留.parquet"
CASES = ROOT / "reports/research/510300_all_factor_joint_scorecard_v1/results/全部30历史案例_联合解释分与覆盖.parquet"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def clock_upper(row):
    """原件缺失时仍保留已确认的发布日期，防止沿用旧文本遮蔽缺口。"""
    dates = []
    for key in ("page_publication_at", "publication_upper"):
        if pd.notna(row.get(key)):
            value = pd.Timestamp(row[key])
            dates.append(value.tz_localize("Asia/Shanghai") if value.tzinfo is None else value.tz_convert("Asia/Shanghai"))
    for key in ("index_publication_date", "cover_date"):
        if pd.notna(row.get(key)):
            dates.append(pd.Timestamp(row[key]).normalize().tz_localize("Asia/Shanghai") + pd.Timedelta(hours=23, minutes=59, seconds=59))
    return max(dates).isoformat() if dates else None


def bind_origins(reports, dates):
    if reports.quarter.tolist() != EXPECTED:
        raise ValueError("须保留原59季度及顺序，不删未知季度。")
    reports = reports.copy()
    reports["publication_upper"] = [clock_upper(row) for row in reports.to_dict("records")]
    if reports.publication_upper.isna().any():
        raise ValueError("有报告公布钟未知，不能假定其未公布并沿用旧报告。")
    reports = first_known_origins(reports, dates)
    known = reports.dropna(subset=["first_known_origin"]).copy()
    if known.first_known_origin.duplicated().any():
        raise ValueError("多报告首次可知原点相同，须保留冲突，不自动选择。")
    known = known.sort_values("first_known_origin")
    fields = ["quarter", "status", "previous_quarter", "publication_upper", "first_known_origin",
              "economic_similarity", "guidance_similarity", "historical_first_vintage", "pdf_path"]
    for key in fields:
        if key not in known:
            known[key] = None
    right = known[fields].rename(columns={key: "pbc_" + key for key in fields})
    left = pd.DataFrame({"date": pd.to_datetime(dates)})
    result = pd.merge_asof(left, right, left_on="date", right_on="pbc_first_known_origin", direction="backward")
    result["pbc_two_channel_known"] = (result.pbc_status.eq("ORIGINAL_DOCUMENT_AND_TWO_SECTIONS")
                                       & result.pbc_economic_similarity.notna()
                                       & result.pbc_guidance_similarity.notna())
    for part in ("economic", "guidance"):
        result["pbc_" + part + "_change"] = (1 - result["pbc_" + part + "_similarity"]).where(result.pbc_two_channel_known)
    result["pbc_report_age_calendar_days"] = (result.date - result.pbc_first_known_origin).dt.days
    result["pbc_numeric_direction"] = "NONE_NOT_TRADING_SCORE"
    result["pbc_independent_validation"] = "NOT_ESTABLISHED"
    return reports, result


def freeze():
    save("protocol.json", {"at": now(), "registration": "TECH.R263", "decision": "TECH.R264",
          "question": "分开政策指引与经济描述的变化后，原全部30成功/失败案例在进场之前实际能看到哪份报告？",
          "original_cases": 30, "original_daily_origins": 3488,
          "all_cases_kept": True, "all_origins_kept": True,
          "source_range": EXPECTED, "missing": "最新已确认公布的季度缺原件/任一章节/上一季度即双通道未知；不沿用旧季度掩盖缺失，不补零。",
          "known_clock": "公布页完整时刻、目录/封面日期日末的最大上界；原3488日15:05首次可知，无新时钟优化。",
          "original_scores": "原全部30解释等级和所有字段逐列保存；新文本变化不加利好分或胜率。",
          "independence": "季度原件才是独立信息单位，3488日/30案例的重复携带不增加独立报告数；历史第一版未认证。",
          "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_trading_scores": 0,
          "financial_admission": "NOT_ADMITTED_METHOD_NOT_REGISTERED", "goal_achieved": False,
          "code_sha256": digest(Path(__file__)),
          "input_sha256": {str(path.absolute().relative_to(ROOT).as_posix()): digest(path)
                           for path in (SOURCE / "protocol.json", CALENDAR, CASES)}})
    print("已冻结全部原点/原30案例的文本对应，缺失不回退旧季度，0交易模型。", flush=True)


def run():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    if digest(Path(__file__)) != protocol["code_sha256"]:
        raise RuntimeError("对应登记后源码变化。")
    for name, expected in protocol["input_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("原固定输入变化：" + name)
    save("run_started.json", {"at": now()})
    dates = pd.read_parquet(CALENDAR, columns=["date"]).date
    cases = pd.read_parquet(CASES)
    reports = pd.read_parquet(SOURCE / "全部59季度_原件章节公布钟与未知.parquet")
    if len(cases) != 30 or len(dates) != 3488:
        raise ValueError("原案例或原点范围变化。")
    reports, origins = bind_origins(reports, dates)
    bound = cases.merge(origins, on="date", how="left", validate="one_to_one")
    if len(bound) != 30:
        raise ValueError("案例对应丢失行。")
    pd.testing.assert_frame_equal(cases, bound[cases.columns], check_exact=True)
    if len(origins) != 3488 or not origins.date.equals(dates.reset_index(drop=True)):
        raise ValueError("实际原点被删改。")
    for name, frame in (("全部59报告_缺件仍保留公布钟", reports), ("全部3488原点_最新已公布文本与未知", origins),
                        ("原全部30案例_解释等级不改与当时文本", bound)):
        frame.to_parquet(OUT / (name + ".parquet"), index=False)
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    status = "COMPLETED_ALL_ORIGINS_AND_ORIGINAL_CASES_TEXT_ALIGNMENT"
    save("summary.json", {"at": now(), "decision": "TECH.R264", "status": status,
          "original_cases": 30, "cases_two_channel_known": int(bound.pbc_two_channel_known.sum()),
          "original_daily_origins": 3488, "origins_two_channel_known": int(origins.pbc_two_channel_known.sum()),
          "unique_reports_carried": int(origins.pbc_quarter.nunique()),
          "unique_case_reports_carried": int(bound.pbc_quarter.nunique()),
          "original_case_values_and_explanatory_grades_preserved": True,
          "missing_current_quarter_not_backfilled": True, "historical_first_vintage_authenticated": False,
          "financial_admission": "NOT_ADMITTED_METHOD_NOT_REGISTERED",
          "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_trading_scores": 0,
          "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False})
    save("run_completed.json", {"at": now(), "terminal": True, "status": status})
    print(f"3488原点和30案例完整对应：双通道可见{int(origins.pbc_two_channel_known.sum())}原点、{int(bound.pbc_two_channel_known.sum())}案例，0新金融结果。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="央行文本与全部原案例的当时对应。")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()


if __name__ == "__main__":
    main()
