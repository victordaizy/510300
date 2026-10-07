"""描述冻结联合模型的具体案例、概率误差和同完整年频率，不生成新策略。"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research import macro_technical_first_passage_study_v1 as study
from research.point_first_passage_study_v1 import read, write_json, digest, require, now
from research.point_account_nr7_complement_v1 import annual_rows

OUT = study.OUT.with_name(study.OUT.name + "_clock_adapter")


def save_table(name, frame):
    path = OUT / "results" / name
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def run():
    require(not (OUT / "description_receipt.json").exists(), "结果解释已归档，不追加模型或分组营救。")
    summary = read(OUT / "summary.json")
    require(read(OUT / "saved_result_verification.json")["status"] == "PASS_EIGHT_SAVED_ACCOUNTS_AND_METRICS", "保存账户未通过。")
    forecasts = pd.read_parquet(OUT / "results/全部事前配对模型预测与实际宏观路径.parquet")
    outcomes = pd.read_parquet(study.BASE / "results/原点首次边界参考结果.parquet")
    data = pd.read_parquet(OUT / "results/全部3488当时已知技术与宏观_未知保留.parquet")
    metrics = pd.read_parquet(OUT / "results/十二完整账户共同口径比较.parquet")
    points = pd.read_parquet(OUT / "results/全部实际进出点位与事前宏观评分.parquet")
    predictions = forecasts.merge(outcomes.rename(columns={"status": "outcome_status"}), on="origin_index", validate="many_to_one")
    errors = []
    for period, (start, end) in study.original.PERIODS.items():
        end_idx = int(data.index[data.date.le(end)][-1])
        first = int(data.index[data.date.ge(start)][0]) - 1
        for policy in study.model.POLICIES:
            f = predictions.loc[predictions.policy.eq(policy)]
            base = (f.date.ge(start) & f.date.le(end) & f.status.eq("AVAILABLE")
                    & f.outcome_status.eq("MATURE_REFERENCE") & f.mature_idx.le(end_idx))
            for name, mask in [("ALL_OVERLAPPING", base), ("ORIGINAL_FIXED_EVERY_TWENTY", base & ((f.origin_index - first) % 20 == 0))]:
                selected = f.loc[mask]
                probability = selected[["p_LOSS", "p_PROFIT", "p_TIMEOUT"]].to_numpy(float)
                truth = np.asarray([[float(label == x) for x in ["LOSS", "PROFIT", "TIMEOUT"]] for label in selected.event_class])
                errors.append({"period": period, "policy": policy, "sample": name, "mature_rows": len(selected),
                               "multiclass_logloss": float(-(truth * np.log(np.clip(probability, 1e-15, 1.))).sum(axis=1).mean()),
                               "multiclass_brier": float(((truth - probability) ** 2).sum(axis=1).mean()),
                               "net_win_brier": float(((selected.predicted_win_probability - selected.reference_net_return.gt(0)) ** 2).mean()),
                               "role": "原预测表现诊断，不选门槛、不当独立验证"})
    error_frame = pd.DataFrame(errors)
    save_table("两模型参考事件预测误差_全部与原20步抽样", error_frame)
    year_rows = pd.read_parquet(OUT / "results/逐年净收益与真实交易次数.parquet").to_dict("records")
    for period in study.original.PERIODS:
        for cost in study.original.COSTS:
            year_rows.extend(annual_rows(study.saved_baseline(period, cost), period, "A_SAVED_WEIGHT", cost))
    annual = pd.DataFrame(year_rows)
    save_table("三策略同口径逐年收益与次数", annual)
    frequency = annual.loc[annual.full_year].groupby(["period", "cost", "policy"]).agg(
        complete_years=("year", "size"), completed_cycles=("completed_cycles", "sum"),
        average_full_year_cycles=("completed_cycles", "mean")).reset_index()
    aligned = metrics.rename(columns={"average_full_year_cycles": "original_metrics_frequency_if_present"}).merge(
        frequency, on=["period", "cost", "policy"], validate="one_to_one", suffixes=("", "_full_year"))
    save_table("十二账户与统一完整年频率", aligned)
    save_table("三策略统一完整自然年交易频率", frequency)
    selected = pd.read_parquet(OUT / "results/事前固定17关键日期_宏观与技术.parquet")
    score_fields = ["date", "policy", "score", "entry_event", "predicted_p_times_b", "predicted_net_expectation", "macro_features_on_path"]
    snapshots = selected.merge(forecasts[score_fields], on="date", how="left", validate="one_to_many")
    save_table("原固定17关键日_两模型评分与实际宏观背景", snapshots)
    stress = aligned.loc[aligned.cost.eq("STRESS")].copy()
    text = "# 具体上涨段、宏观背景与联合评分的解读\n\n"
    text += "本次宏观接入和唯一金融检验已实际完成。以下是同钟资料与冻结结果的解释，0新增模型/标签/账户，不将好案例、误差分组或新阈值变成策略。\n\n"
    text += "2019年1月：1月8日日MACD柱先转正，1月14日上一完整周柱可用并转正，1月18日DIF转正；当时最新已发布的新订单49.7、环比下降0.7点，1月8日资金利率低于政策利率0.2692个百分点，前日可知融资五日变化约−0.558%。宽松资金、弱订单和融资收缩并存，不能要求宏观全部同向。联合模型1月8日估计净胜率29.79%、pB0.290，仍没有识别这次启动。\n\n"
    text += "2020年修复：4月1日可以看到3月新订单52.0及较2月上升22.7点，这是扩散指数点差，不能解释成订单金额增长22.7%。5月28/29日量价修复时最新订单50.2、较前月下降1.8点；日柱和上一完整周柱仍负，融资五日变化约−0.318%/−0.337%。此时两模型预测相同，pB约0.380，都未进入；5月的宏观字段虽已接入，但没有进入这些日期的实际树路径。4月22日实际盈利交易及5月7日实际亏损交易全部保留，不用7月最高价倒推退出。\n\n"
    text += "2024年9月：23日量准备、24日价格放量确认；24日DIF与上一完整周柱仍负。当时最新订单为8月48.9，较前月下降0.4点，资金政策利差+0.185个百分点，前日可知融资五日变化−0.871%。这段上涨可以在慢经济统计仍弱时出现。本次六宏观字段没有完整表达政策公告内容、预告目标与市场预期更新。24日纯技术模型估计净胜率57.80%、pB0.827，联合模型48.88%、pB0.684，二者均未通过原质量门；不能把漏检全部归因于新增宏观。该固定用途没有在9月启动进场，宏观联合账户2024全年没有新周期。\n\n"
    text += "2015年6月反弹：订单50.6、月度+0.4点不能抵消价格与动量走弱；融资五日变化从6月24日约+0.666%转到30日约−7.829%。该阶段资金政策差为未知，联合信息尚不完整，没有模型观点；不补成零分或认定宏观已证明下跌。\n\n"
    text += "压力费用完整账户：\n\n"
    text += stress[["period", "policy", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles", "win_rate", "payoff", "p_times_b", "average_full_year_cycles"]].to_markdown(index=False, floatfmt=".5f") + "\n\n"
    text += "早期联合模型2个完成周期全赢，但没有亏损样本，B和pB不可定义，不能当成无限盈亏比或高置信策略。近期12完成中4胜8负，压力B约1.0444、pB约0.3481，标准期望约−0.3185；联合收益、夏普和频率均未提高。早期夏普单项高于原A，但收益更低、只有2笔，不能用这个单项代替完整目标。\n\n"
    text += "两个模型只在共同完整资料日评分，共同可评分2256日，1867日预测不同且联合树实际使用宏观路径。不是把未知日期删出完整账户。2026年原资金/融资资料缺口仍保留，全年账户现金日继续计入年化和夏普。\n\n"
    text += error_frame.to_markdown(index=False, floatfmt=".5f") + "\n\n"
    text += "上表沿用原全部重叠标签与固定每20原点诊断；重叠、同月状态反复出现和旧历史筛选限制仍在，不把标签条数视为独立证据，不据误差更好的局部时期营救账户失败。\n\n"
    text += "来源接入状态：PMI订单、DR007/已知政策利率、修复融资已用于本次。M1/M2、社会融资/贷款、银行调查、工业经营及基金来源保留原各自口径、时钟、支持门与旧失败；不能从已有宏观叙事填出本次缺失字段。下一方向先核对公告时点的政策新信息/事前预期用途与原R19等已完成用途，建立不同、完整、可知的合同后才登记新实验；当前只是信息用途提案，没有新金融候选。\n\n"
    text += "本固定联合用途拒绝并关闭。历史开发/校准、first-vintage NOT_CERTIFIED、独立NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED；完整目标未完成。原20万元账户、原策略和12前瞻状态值保持，不通过换权重、阈值、树深或退出重跑。\n\n"
    text += "[主完整结果](研究结果与下一步.md)、[具体17日双模型与宏观](results/原固定17关键日_两模型评分与实际宏观背景.csv)、[全部点位](results/全部实际进出点位与事前宏观评分.csv)、[同完整年频率](results/三策略统一完整自然年交易频率.csv)。\n"
    path = OUT / "具体上涨段与宏观评分解读.md"
    path.write_text(text, encoding="utf-8")
    write_json(OUT / "description_receipt.json", {"at": now(), "status": "PASS_FROZEN_CASE_DESCRIPTION_AND_REFERENCE_DIAGNOSTICS",
        "report": path.relative_to(ROOT).as_posix(), "report_sha256": digest(path), "new_fits": 0, "new_labels": 0, "new_accounts": 0,
        "frequency_method": "统一annual_rows完整自然年，2026未完成年不计年均次数；不重算金融账户",
        "original_A_raw_frequency_difference_retained": True, "charts_visually_inspected": [18, 37, 42, 55],
        "primary_financial_values_unchanged": True, "independent_validation": "NOT_ESTABLISHED"}, exclusive=True)
    print("四具体上涨的宏观/量价解读、参考预测误差及统一完整年频率已归档，0新增模型或账户。", flush=True)


if __name__ == "__main__":
    run()
