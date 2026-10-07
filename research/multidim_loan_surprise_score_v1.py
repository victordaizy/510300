"""在同一次金融数据公布中，固定比较贷款预期偏差的非线性增量。"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

import multidim_nonlinear_score_v1 as common
from multidim_money_surprise_score_v1 import TREE, BASE, NEWS, fit_tree, return_summary
from credit_recency_structure_v23 import Ledger


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_loan_surprise_score_v1"
SELECTION = ROOT / "reports/research/510300_money_consensus_increment_v2/inputs/104个月共识选择.csv"
NUMERIC = ROOT / "reports/research/510300_money_consensus_increment_v2/sources/selected_numeric_sources.json"
PDF_ROOT = ROOT / "reports/research/510300_money_consensus_source_extension_v1/raw"
ORIGINALS = ROOT / "reports/research/510300_macro_transmission_context_v4/inputs/loan_originals.json"
EVENTS = ROOT / "reports/research/510300_multidim_financing_composition_v1/data_correction/monthly_score/月度事件联合状态及标签.parquet"
LEDGER_ORIGINALS = ROOT / "reports/research/510300_credit_recency_structure_v23/inputs/originals.json"
LOAN = "贷款预期偏差超出显示精度的余额基点"
CONTROL = [*BASE, NEWS]
FEATURES = [*CONTROL, LOAN]
PATTERN = re.compile(r"New\s+(?:Yuan\s+)?Loans\s+CNY(?P<ytd>\s+YTD)?\s+(?P<month>[A-Za-z]{3})\s+(?P<survey>--|-?[\d,.]+b)\s+(?P<actual>--|-?[\d,.]+b)", re.I)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name: str, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(common.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本次贷款预期增量已固定，不覆盖。")
    if sha(ORIGINALS) != sha(LEDGER_ORIGINALS):
        raise ValueError("原始贷款账本副本不同，须先解决来源。")
    save("protocol.json", {
        "study_id": "510300_MULTIDIM_LOAN_SURPRISE_SCORE_V1", "frozen_at": common.now(),
        "previous_turn_classification": "PROGRESS_FINANCING_SCALE_INTENSITY_DECOMPOSITION",
        "question": "同一金融数据公布中的贷款预期偏差，是否在订单、资金、融资、指数量价及M2偏差之外，改善其后五日收益的联合历史评分？",
        "hypothesis": "M2变化同时包含多种资产负债表渠道；贷款净增相对公布前调查的偏差可能补充增长及政策反应信息，作用可能随指数状态不同。贷款净增不等于真实需求或新发放额，不预设正偏差必然利好。",
        "source_universe": "完整保留原104个月选择表；只解析原先已经选中的Bualuang报告同一中国日历页。截至2025年实际公布的此类报告52份，最早统计月2020年12月。其他来源及缺失月份保留未准入，不新增检索择取不同预测。",
        "provider_reason": "该来源有可直接对应的人民币贷款项目、统计月份、单月或YTD标记、Survey与Actual列。先固定统一表格来源，再做收益计算；这不是完整市场共识或独立验证。",
        "expectation": "匹配New Yuan Loans CNY或New Loans CNY，当月月份必须一致，Survey为数值，Actual为--。报告日期严格早于金融数据公布日期；同一已选日历页无唯一合格行则缺失。单位b为十亿元人民币，乘10转亿元。",
        "actual": "复用既有官方原文贷款账本，单月预期优先当月直接披露，否则用当时已经公布的累计端点作差；YTD预期只匹配相同年初至当月。当前和以前公告的舍入界及表达式全部保留，不用后来修订年表。",
        "new_feature": "先算同区间贷款实际减事前调查，扣去两者显示舍入误差合计的半宽；剩余有符号最小幅度除以上一统计月已公布人民币贷款余额，再乘10000。落在显示精度范围内记0，来源缺失仍缺失。此基点用于统一规模，不是贷款增速。",
        "scope_break": "保留2023年统计范围扩展及累计差混合版本标志。不将相对调查偏差称为真实需求冲击，也不把舍入界当作修订误差上限。",
        "information_clock": "决策在金融统计公开日23:59:59；状态及五日标签复用官方纠正后的原月度输入，资金和融资时钟不变。每次公布一条样本，不将同一月值向后复制成每日独立事件。",
        "features_control": CONTROL, "features_joint": FEATURES,
        "model": TREE, "training": {"lookback_sessions": 504, "minimum_mature_events": 12, "label_availability": "退出开盘不晚于当前决策截止"},
        "comparison": "同一合格事件、同一成熟训练池和同一树复杂度：九变量原状态与M2对照、增加贷款偏差的十变量树、十变量岭回归alpha10、同池历史均值。只比较这一次，不加深树、不换阈值。",
        "score": "当次模型预测相对训练池拟合值的中位百分位，0至100，不是上涨概率；五个固定20分档全部保留。",
        "opportunity": "分数>=80且预测净收益>0，前一机会退出之前不新增，固定10000份、公布日后的次开盘进入、5个开盘间隔后退出；原压力佣金和滑点标签原样沿用。",
        "primary_period": ["2024-01-01", "2025-12-31"], "earlier_context": ["2021-01-01", "2023-12-31"],
        "account_admission": "主要期联合高分机会净均值为正，且新增字段至少改变一个主要期预测时，才另行固定完整账户；否则停止该表达。报告误差和与对照的差异，不把高分数或正均值单独称为目标完成。",
        "new_parameter_searches": 0, "new_source_downloads": 0, "new_accounts_in_this_stage": 0,
        "old_research": ["八项状态加M2的月度树已完成但无M2增量", "三项价格加M1-M2预期差的旧岭回归已拒绝", "社融政府债构成加资金条件的旧近邻账户已拒绝", "贷款及社融预期的三病例此前仅作构成说明"],
        "selection_history": "多个历史结果与价格已经被研究，本轮是新增已选原报告字段的开发比较，不能称独立验证；原有失败规则、资金缺失和前瞻暂停不改动。",
        "inputs": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p)} for p in [SELECTION, NUMERIC, ORIGINALS, LEDGER_ORIGINALS, EVENTS]],
        "goal_achieved": False, "orders_authorized": False, "current_view": "NO_VIEW"
    })
    print("已固定单一来源52份报告、相同事件对照及原树复杂度。")


def sources():
    if (OUT / "source_result.json").exists():
        raise RuntimeError("贷款预期字段已经提取，不覆盖。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["inputs"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("冻结后的输入发生变化。")
    choices = pd.read_csv(SELECTION)
    chosen = {r["stat_month"]: r for r in read(NUMERIC)}
    originals = {r["stat_month"]: r for r in read(ORIGINALS)}
    ledger = Ledger()
    rows, evidence = [], []
    for item in choices.to_dict("records"):
        month = item["stat_month"]
        row = {k: item.get(k) for k in ["stat_month", "published_at", "available_at_upper_bound", "forecast_date", "forecast_source", "forecast_url", "forecast_file", "forecast_sha256", "consensus_admitted"]}
        if pd.Timestamp(item["published_at"]).year > 2025:
            row["loan_source_status"] = "AFTER_FIXED_PERIOD"
        elif item["forecast_source"] != "Bualuang":
            row["loan_source_status"] = "OUTSIDE_FIXED_PROVIDER_OR_MISSING"
        else:
            source = chosen[month]
            path = PDF_ROOT / source["filename"]
            if sha(path) != source["sha256"]:
                raise ValueError("报告文件与原先选择的副本不同。")
            page_number = int(source["M2"]["page"])
            with pdfplumber.open(path) as doc:
                text = doc.pages[page_number - 1].extract_text(x_tolerance=2, y_tolerance=3) or ""
            # 国家标题必须独占一行，不能误切在Caixin China或调查项目名称中。
            heading = re.search(r"(?m)^[ \t]*China[ \t]*$", text)
            country = text[heading.end():] if heading else ""
            country = re.split(r"(?m)^[A-Z][A-Za-z ]+\nEconomic Releases", country)[0]
            flat = re.sub(r"\s+", " ", country)
            month_name = calendar.month_abbr[int(month[-2:])]
            hits = [m for m in PATTERN.finditer(flat) if m.group("month").lower() == month_name.lower()]
            qualified = [m for m in hits if m.group("survey") != "--" and m.group("actual") == "--"]
            row.update(forecast_page=page_number, forecast_local_path=str(path.relative_to(ROOT)).replace("\\", "/"))
            evidence.append({"stat_month": month, "file": row["forecast_local_path"], "page": page_number, "country_text": country,
                             "matched_rows": [m.group(0) for m in hits], "sha256": source["sha256"]})
            if heading is None or not re.search(r"Survey\s+Actual\s+Prior", flat):
                row["loan_source_status"] = "UNRESOLVED_HEADER"
            elif len(qualified) != 1:
                row["loan_source_status"] = "MISSING_OR_AMBIGUOUS_LOAN_ROW"
            else:
                hit = qualified[0]
                literal = hit.group("survey")[:-1].replace(",", "")
                survey = float(literal) * 10
                precision = len(literal.split(".")[-1]) if "." in literal else 0
                survey_half = .5 * 10 ** (-precision) * 10
                window = "YTD" if hit.group("ytd") else "1"
                actual = ledger.row(month, "rmb_total", window)
                prior_month = str(pd.Period(month, freq="M") - 1)
                prior = originals.get(prior_month)
                cutoff = pd.Timestamp(item["available_at_upper_bound"])
                if pd.Timestamp(item["forecast_date"]).date() >= cutoff.date():
                    raise ValueError("贷款预期日期未早于公布日期。")
                if not np.isfinite(actual["value_yi"]) or prior is None:
                    row["loan_source_status"] = "MISSING_ACTUAL_OR_PRIOR_SCALE"
                elif pd.Timestamp(actual["known_at"]) > cutoff or pd.Timestamp(prior["conservative_known_at"]) >= cutoff:
                    raise ValueError("贷款实际或余额分母使用了后来信息。")
                else:
                    atoms = json.loads(actual["expression"])
                    raw_paths = sorted({originals[ledger.atoms[k]["source_month"]]["raw_path"] for k in atoms})
                    for p in raw_paths:
                        src = next(v for v in originals.values() if v["raw_path"] == p)
                        if sha(ROOT / p) != src["source_sha256"]:
                            raise ValueError("官方原文与金额账本来源不一致。")
                    raw = actual["value_yi"] - survey
                    half = actual["rounding_half_yi"] + survey_half
                    resolved = float(np.sign(raw) * max(abs(raw) - half, 0))
                    stock = float(prior["rmb_stock_yi"])
                    if not np.isfinite(stock) or stock <= 0:
                        raise ValueError("上一期贷款余额无效。")
                    row.update(loan_source_status="ADMITTED", forecast_loan_yi=survey, forecast_loan_rounding_half_yi=survey_half,
                               forecast_interval=window, forecast_numeric_row=hit.group(0), actual_loan_yi=actual["value_yi"],
                               actual_rounding_half_yi=actual["rounding_half_yi"], actual_interval_start=actual["start"],
                               actual_interval_end=actual["end"], actual_known_at=actual["known_at"], actual_method=actual["method"],
                               actual_expression=actual["expression"], actual_source_terms=actual["source_terms"],
                               actual_source_files=";".join(raw_paths), actual_source_url=originals[month]["source_url"],
                               statistical_regime=actual["regime"], loan_surprise_yi=raw, combined_rounding_half_yi=half,
                               loan_surprise_lower_yi=raw-half, loan_surprise_upper_yi=raw+half,
                               loan_surprise_precision_resolved_yi=resolved, prior_stock_yi=stock, prior_stock_month=prior_month,
                               prior_stock_known_at=prior["conservative_known_at"], **{LOAN: resolved / stock * 10000})
            print(f"贷款预期字段：{month}，{row['loan_source_status']}。", flush=True)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "全部104个月的贷款预期覆盖.csv", index=False, encoding="utf-8-sig")
    save("原报告贷款预期行与上下文.json", evidence)
    admitted = frame[frame.loan_source_status.eq("ADMITTED")]
    save("source_result.json", {"status": "COMPLETED_FIXED_PROVIDER_LOAN_SURPRISE_INPUTS", "at": common.now(),
                               "months": len(frame), "pdf_pages_read": len(evidence), "admitted": len(admitted),
                               "status_counts": frame.loan_source_status.value_counts().to_dict(),
                               "forecast_interval_counts": admitted.forecast_interval.value_counts().to_dict(),
                               "display_precision_unresolved": int(admitted.loan_surprise_precision_resolved_yi.eq(0).sum()),
                               "actual_method_counts": admitted.actual_method.value_counts().to_dict(),
                               "source_first_vintage_authenticated": False, "new_models": 0, "goal_achieved": False})
    print("原报告字段提取完成：" + str(len(admitted)) + "个月合格。", flush=True)


def fit_linear(pool, current):
    x, y = pool[FEATURES].to_numpy(float), pool.actual_net5.to_numpy(float)
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=1)
    scale[scale < 1e-12] = 1.
    z = np.clip((x-mean)/scale, -5, 5)
    zmean = z.mean(axis=0)
    centered = z-zmean
    coef = np.linalg.solve(centered.T @ centered + 10*np.eye(len(FEATURES)), centered.T @ (y-y.mean()))
    predicted = float(y.mean() + (np.clip((current[FEATURES].to_numpy(float)-mean)/scale, -5, 5)-zmean) @ coef)
    return predicted, {"mean": mean.tolist(), "scale": scale.tolist(), "zmean": zmean.tolist(), "coef": coef.tolist(), "ymean": float(y.mean())}


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("贷款预期联合评分已完成，不改变设定重跑。")
    protocol = read(OUT / "protocol.json")
    for item in protocol["inputs"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原固定输入已改变。")
    coverage = pd.read_csv(OUT / "全部104个月的贷款预期覆盖.csv")
    admitted = coverage[coverage.loan_source_status.eq("ADMITTED")]
    events = pd.read_parquet(EVENTS)
    extra = [c for c in admitted.columns if c not in events.columns or c == "stat_month"]
    events = events.merge(admitted[extra], on="stat_month", how="inner", validate="one_to_one").sort_values("decision_at").reset_index(drop=True)
    if not np.isfinite(events[FEATURES+['actual_net5']].to_numpy(float)).all():
        raise ValueError("共同事件存在缺失特征或标签。")
    for col in ["decision_at", "entry_at", "exit_at", "state_available_at"]:
        events[col] = pd.to_datetime(events[col], utc=True).dt.tz_convert("Asia/Shanghai")
    if not (events.state_available_at <= events.decision_at).all() or not (events.entry_at > events.decision_at).all():
        raise ValueError("原事件时钟不一致。")
    events.to_parquet(OUT / "共同事件的十变量与原标签.parquet", index=False)
    rows, models, excluded = [], [], []
    for _, current in events.iterrows():
        if current.decision_at.year < 2021:
            continue
        pool = events[events.state_idx.ge(current.state_idx-504) & events.exit_at.le(current.decision_at)]
        if len(pool) < 12:
            excluded.append({"stat_month": current.stat_month, "mature_events": len(pool), "reason": "固定训练池不足12次公布"})
            continue
        if pool.stat_month.eq(current.stat_month).any():
            raise ValueError("训练错误地包含当前公布结果。")
        control_prediction, control_score, control = fit_tree(pool, current, CONTROL)
        joint_prediction, joint_score, joint = fit_tree(pool, current, FEATURES)
        linear_prediction, linear = fit_linear(pool, current)
        row = current.to_dict()
        row.update(training_events=len(pool), training_months=";".join(pool.stat_month), training_max_exit_at=pool.exit_at.max(),
                   control_prediction=control_prediction, control_score=control_score, joint_prediction=joint_prediction,
                   joint_score=joint_score, linear_prediction=linear_prediction, mean_prediction=float(pool.actual_net5.mean()),
                   loan_used=LOAN in joint["used_features"], joint_features_used="；".join(joint["used_features"]),
                   prediction_changed=abs(joint_prediction-control_prediction)>1e-12)
        rows.append(row)
        models.append({"stat_month": current.stat_month, "training_months": pool.stat_month.to_list(),
                       "control_tree": control, "joint_tree": joint, "linear": linear})
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("共同事件不足以执行固定模型。")
    predictions.to_csv(OUT / "逐事件贷款预期联合评分.csv", index=False, encoding="utf-8-sig")
    save("saved_models.json", models)
    save("训练不足的月份.json", excluded)
    buckets, errors, opportunities = [], [], []
    for model in ["control", "joint"]:
        last_exit = pd.Timestamp("1900-01-01", tz="Asia/Shanghai")
        for _, row in predictions.iterrows():
            if row.decision_at < last_exit or row[model+"_score"] < 80 or row[model+"_prediction"] <= 0:
                continue
            opportunities.append({"model": model, "stat_month": row.stat_month, "era": row.era, "entry_date": row.entry_date,
                                  "exit_date": row.exit_date, "score": row[model+"_score"], "prediction": row[model+"_prediction"],
                                  "net_return": row.actual_net5, "loan_surprise_yi": row.loan_surprise_yi,
                                  "forecast_interval": row.forecast_interval, LOAN: row[LOAN]})
            last_exit = row.exit_at
        for era, block in predictions.groupby("era", sort=False):
            bands = np.minimum((block[model+"_score"] / 20).astype(int), 4)
            for band in range(5):
                buckets.append({"model": model, "era": era, "score_band": f"{band*20}—{band*20+20}",
                                **return_summary(block.loc[bands.eq(band), "actual_net5"])})
    for era, block in predictions.groupby("era", sort=False):
        for model in ["control", "joint", "linear", "mean"]:
            errors.append({"model": model, "era": era, "n": len(block), "mse": float(((block[model+"_prediction"]-block.actual_net5)**2).mean())})
    opp = pd.DataFrame(opportunities, columns=["model", "stat_month", "era", "entry_date", "exit_date", "score", "prediction", "net_return", "loan_surprise_yi", "forecast_interval", LOAN])
    summaries, annual = [], []
    for model in ["control", "joint"]:
        for era in ["2021—2023", "2024—2025"]:
            block = opp[opp.model.eq(model) & opp.era.eq(era)]
            summaries.append({"model": model, "era": era, **return_summary(block.net_return)})
        for year in range(2021, 2026):
            block = opp[opp.model.eq(model) & pd.to_datetime(opp.entry_date).dt.year.eq(year)]
            annual.append({"model": model, "year": year, **return_summary(block.net_return)})
    primary = predictions[predictions.era.eq("2024—2025")]
    main = next(x for x in summaries if x["model"] == "joint" and x["era"] == "2024—2025")
    admit = main["n"] > 0 and main["mean_net5"] > 0 and bool(primary.prediction_changed.any())
    for name, values in [("固定分层比较.csv", buckets), ("同事件模型误差.csv", errors), ("固定高分机会比较.csv", summaries), ("逐年自然机会.csv", annual)]:
        pd.DataFrame(values).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    opp.to_csv(OUT / "全部高分机会.csv", index=False, encoding="utf-8-sig")
    save("result.json", {"study_id": protocol["study_id"], "status": "COMPLETED_FIXED_LOAN_SURPRISE_JOINT_SCORE",
                         "at": common.now(), "common_events": len(events), "scored_events": len(predictions),
                         "primary_scored_events": len(primary), "earlier_scored_events": len(predictions)-len(primary),
                         "training_insufficient_events": len(excluded), "fits_per_family": len(predictions), "fitted_families": 3,
                         "loan_used_events": int(predictions.loan_used.sum()), "primary_loan_used_events": int(primary.loan_used.sum()),
                         "primary_predictions_changed": int(primary.prediction_changed.sum()), "buckets": buckets, "model_errors": errors,
                         "opportunity_summary": summaries, "annual_opportunities": annual, "full_account_admitted": admit,
                         "full_account_status": "PENDING_FIXED_ACCOUNT" if admit else "NOT_RUN_NO_ADMITTED_INCREMENT",
                         "new_accounts": 0, "goal_achieved": False, "independent_validation": False, "orders_authorized": False})
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(errors).to_string(index=False))
    print(f"共评分{len(predictions)}次，主要期{len(primary)}次；贷款字段改变主要期预测{int(primary.prediction_changed.sum())}次；完整账户准入：{admit}。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="贷款预期偏差与指数状态的固定联合比较")
    parser.add_argument("stage", choices=["prepare", "sources", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "sources": sources, "run": run}[args.stage]()
