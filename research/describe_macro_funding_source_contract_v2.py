"""解释已保存的资金源合同修正结果，不拟合、不重算账户。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research import funding_availability_semantics_v1 as diagnosis
from research import macro_funding_source_contract_v2 as version
from research import macro_technical_first_passage_study_v1 as study
from research.point_account_nr7_complement_v1 import annual_rows

OUT, OLD, ROOT = version.OUT, version.OLD, version.ROOT


def table(name, frame):
    p = OUT / "results" / name
    frame.to_parquet(p.with_suffix(".parquet"), index=False)
    frame.to_csv(p.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def run():
    receipt = OUT / "description_receipt.json"
    study.require(not receipt.exists(), "本次结果解释已完成，不重复覆盖。")
    study.require(study.read(OUT / "saved_result_verification.json")["status"] ==
                  "PASS_EIGHT_SAVED_ACCOUNTS_AND_METRICS", "八个保存账户未通过。")
    summary = study.read(OUT / "summary.json")
    sources = study.read(diagnosis.OUT / "summary.json")
    current = pd.read_parquet(OUT / "results/十二完整账户共同口径比较.parquet")
    prior = pd.read_parquet(OLD / "results/十二账户与统一完整年频率.parquet")
    annual = pd.read_parquet(OUT / "results/逐年净收益与真实交易次数.parquet").to_dict("records")
    for period in study.original.PERIODS:
        for cost in study.original.COSTS:
            annual.extend(annual_rows(study.saved_baseline(period, cost), period, "A_SAVED_WEIGHT", cost))
    annual = pd.DataFrame(annual)
    table("三策略同口径逐年收益与次数", annual)
    frequency = annual.loc[annual.full_year].groupby(["period", "cost", "policy"]).agg(
        complete_years=("year", "size"), completed_cycles_full_years=("completed_cycles", "sum"),
        average_full_year_cycles=("completed_cycles", "mean")).reset_index()
    keys = ["period", "cost", "policy"]
    aligned = current.rename(columns={"average_full_year_cycles": "original_metrics_frequency_if_present"}).merge(
        frequency, on=keys, validate="one_to_one")
    table("十二账户与统一完整年频率", aligned)
    table("三策略统一完整自然年交易频率", frequency)
    fields = ["net_cagr", "net_sharpe", "max_drawdown", "completed_cycles", "win_rate", "payoff",
              "p_times_b", "standard_expectancy_loss_units", "ending_equity", "average_full_year_cycles"]
    compared = prior[keys + fields].merge(aligned[keys + fields], on=keys, suffixes=("_V1", "_V2"), validate="one_to_one")
    for field in fields:
        compared[field + "_delta"] = compared[field + "_V2"] - compared[field + "_V1"]
    table("V1与V2全部十二账户固定比较", compared)
    points = pd.read_parquet(OUT / "results/全部实际进出点位与事前宏观评分.parquet")
    old_points = pd.read_parquet(OLD / "results/全部实际进出点位与事前宏观评分.parquet")
    identity = keys + ["entry_origin", "entry_date"]
    point_fields = ["exit_date", "status", "entry_quantity", "entry_equity", "exit_reason", "net_pnl",
                    "net_return", "predicted_p_times_b", "macro_features_on_path"]
    cycles = old_points[identity + point_fields].merge(points[identity + point_fields], on=identity,
            how="outer", suffixes=("_V1", "_V2"), indicator="membership", validate="one_to_one")
    cycles["net_pnl_delta"] = cycles.net_pnl_V2.fillna(0) - cycles.net_pnl_V1.fillna(0)
    table("V1与V2全部真实周期身份和净损益比较", cycles)
    f = pd.read_parquet(OUT / "results/全部事前配对模型预测与实际宏观路径.parquet")
    old_f = pd.read_parquet(OLD / "results/全部事前配对模型预测与实际宏观路径.parquet")
    ff = ["status", "entry_event", "predicted_p_times_b", "predicted_net_expectation", "macro_features_on_path"]
    prediction_change = old_f[["date", "origin_index", "policy"] + ff].merge(
        f[["date", "origin_index", "policy"] + ff], on=["date", "origin_index", "policy"],
        suffixes=("_V1", "_V2"), validate="one_to_one")
    prediction_change["entry_event_changed"] = prediction_change.entry_event_V1.ne(prediction_change.entry_event_V2)
    prediction_change["pB_changed_exactly"] = ~(
        prediction_change.predicted_p_times_b_V1.eq(prediction_change.predicted_p_times_b_V2) |
        (prediction_change.predicted_p_times_b_V1.isna() & prediction_change.predicted_p_times_b_V2.isna()))
    table("全部原点两版本预测和入场门变化", prediction_change)
    selected = pd.read_parquet(OUT / "results/事前固定17关键日期_宏观与技术.parquet")
    snapshots = selected.merge(f[["date", "policy", "status", "score", "entry_event", "predicted_p_times_b",
                                 "predicted_net_expectation", "macro_features_on_path"]], on="date", validate="one_to_many")
    table("原固定17关键日_两模型评分与实际宏观背景", snapshots)
    known = pd.read_parquet(diagnosis.OUT / "results/全部3488原点_源与统计训练支持分开.parquet")
    extra = known.loc[known.additional_joint_origin].copy()
    extra["year"] = extra.date.dt.year
    extra_years = extra.groupby("year").size().to_dict()
    stress_new = cycles.loc[cycles.period.eq("2020_2026") & cycles.cost.eq("STRESS") &
                            cycles.policy.eq("MACRO_TECH_TREE") & cycles.membership.eq("right_only")]
    study.require(len(stress_new) == 5 and stress_new.net_pnl_V2.lt(0).all(), "新增周期事实与已保存结果不一致。")
    pair = compared.loc[compared.period.eq("2020_2026") & compared.cost.eq("STRESS") &
                        compared.policy.eq("MACRO_TECH_TREE")].iloc[0]
    same = cycles.loc[cycles.period.eq("2020_2026") & cycles.cost.eq("STRESS") &
                     cycles.policy.eq("MACRO_TECH_TREE") & cycles.membership.eq("both")]
    study.require(np.isclose(stress_new.net_pnl_V2.sum() + same.net_pnl_delta.sum(), pair.ending_equity_delta,
                             atol=1e-7), "全体真实周期差与期末账户差不等。")
    erratum = {"at": study.now(), "affected_frozen_record": "TECH.R196 summary.json status文字",
               "preserved_original_status": sources["status"],
               "correct_interpretation": "COMPLETED_SOURCE_SEMANTICS_DIAGNOSTIC_TRAINING_AND_RECENT_INPUTS_CHANGED",
               "evidence": {k: sources[k] for k in ["changed_training_contracts", "recent_training_contracts_exactly_same",
                                                   "recent_forecast_input_origins_exactly_same"]},
               "original_numeric_fields_unchanged": True, "original_frozen_hashes_unchanged": True,
               "reason": "旧状态文字硬编码为无近期变化，与已保存的77变化合同及两个false字段冲突；仅此解释文字纠正。"}
    study.write_json(OUT / "source_diagnostic_status_erratum.json", erratum, exclusive=True)
    stress = aligned.loc[aligned.cost.eq("STRESS"), keys + fields]
    text = "# 多信息源评分：具体上涨、资金源可用性与完整账户结论\n\n"
    text += "TECH.R195—R196完成资金源语义核对；TECH.R197登记唯一金融用途；TECH.R198完成一次完整检验。宏观信息已进入模型并改变判断，但当前联合用途未改善完整账户收益与夏普，拒绝并关闭该配置。原TECH.R192结果和所有冻结策略保持。\n\n"
    text += "## 已实际接入哪些信息\n\n"
    text += "技术八项为日MACD柱/ATR、上一完整周柱/ATR、均线距离/ATR、五日动量、相对量对数、五日上涨成交量占比、实现波动率比对数、K线实体效率。宏观六项为已发布制造业PMI新订单相对50及连续月变化、前日DR007减统计日政策利率及其五日变化、前日可知融资余额五日变化、融资买入五日/六十日活动比。六字段来自三类宏观来源，相邻字段有相关性，不能算六份独立证据。\n\n"
    text += "采用相同成熟训练池的两棵固定浅树比较：八技术，与八技术加六宏观。分数为100×估计净胜率，未证明概率校准；入场须估计pB>1且估计净期望正，再执行原风险与交易约束。计划2ATR/1ATR不等于实际盈亏比。实际完成周期的p、B、pB及标准期望分别核对，不用分数代替实际成绩。\n\n"
    text += "## 源可用、指标支持、模型支持分别判断\n\n"
    text += f"旧fund_known还要求旧K05分位数统计成熟，将资金字段可用与旧信号支持混在一起。原账户按旧冻结规则正确执行，本轮改变的是输入合同。在相同3307行缓存中，有{sources['source_available_without_quantile_signal_support']}个时点原字段已满足源钟/源龄/有限原值，却旧分位支持不足。修正后联合完整原点{sources['old_joint_origins']}→{sources['source_joint_origins']}，新增{sources['additional_joint_origins']}个（按年{extra_years}）；142个月度训练合同有{sources['changed_training_contracts']}个变化，最后在2023年3月。源值、政策统计日配对和五日连续六槽规则均不变。\n\n"
    text += "源字段已经公布、六宏观特征能够计算、共同成熟样本足够拟合、分数通过入场门，是四个不同条件。2015年6月24日资金差实际为−0.5694个百分点，已可观察；仍无模型支持，不能把来源已知写成可以买入。R196原status文字与保存的数值相冲突，已另存文字更正，保留原件及哈希；77变化合同不是0变化。\n\n"
    text += "## 四个原案例：先解释同钟现象，再看事前点位\n\n"
    text += "2015年6月反弹：24日相对量0.935，日柱仍负，上一完整周柱正；当时新订单50.6、月差+0.4点，资金差−0.5694个百分点，融资五日变化+0.666%。随后融资变化到30日约−7.829%、日柱仍负。资金宽松和订单略扩张不能单独抵消价格与融资恶化；本例无模型观点，不给事后涨跌加分。\n\n"
    text += "2019年1月：8日日MACD柱先正，14日上一完整周柱已正，18日DIF转正；8日最新订单49.7、月差−0.7点，资金差−0.2692个百分点、融资五日约−0.558%。量价修复与偏弱慢宏观同时存在。联合模型8日估计胜率29.79%、pB0.2895，无入场；模型仍未识别该上涨启动。\n\n"
    text += "2020年修复：4月1日可看到3月订单52.0及月差+22.7点，扩散指数点差不是金额增长率。5月28/29日订单50.2、月差−1.8点；日DIF虽正而日柱/上一完整周柱仍负，融资五日约−0.318%/−0.337%。两模型这两日预测相同，估计pB约0.3819，均不入场。实际4月22日至5月6日压力净+1.639%，5月7日至25日−2.984%；全部赢亏保留，不用7月峰值倒推退出。\n\n"
    text += "2024年9月：24日相对量3.338、日柱正，而DIF及上一完整周柱负；最新订单仍8月48.9、月差−0.4点，资金差+0.185个百分点、融资五日约−0.871%。价格可以先于慢统计修复，本文不据此证明政策因果。24日技术估计pB0.8269、联合0.6839，均未过原门；联合2024年无新周期。遗漏不能全部归因于新增宏观。原政策公告接入裁决R194仍保持，预告目标、实际资金状态与事前预期不能混写。\n\n"
    for n in [18, 37, 42, 55]:
        text += f"![原案例{n}的量价及宏观](案例{n}_技术与宏观联合评分.png)\n\n"
    text += "## 实际完整账户\n\n"
    text += "20万元完整现金账户，原风险预算、T+1、100份、分红、费用、两时期及配对控制均不变；2026未知输入和空仓日仍计入账户日历。BASE和STRESS全部12账户在CSV，下面显示压力费用。年均完成次数使用完整自然年，2015—2019为5年、2020—2025为6年，未完成2026不用于年均次数。\n\n"
    text += stress.to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += "早期联合两笔全赢，没有亏损样本，B和pB不可估，不能记为无限盈亏比；收益0.4132%低于原A1.8371%。近期联合17完成4胜13负，胜率23.53%、B1.09085、pB0.25667，标准期望−0.50803；净年化−0.79645%、夏普−0.62898、回撤7.5665%。频率2→2.833次/完整年，但仍低于A4.1667。BASE近期同样为负。四经济门0/4、全部配对历史稳定门失败；拒绝此完整用途，不选择较早单项夏普作为成功。\n\n"
    text += "## 为什么次数增加而收益更低\n\n"
    text += "下表是两版全体周期身份比较中的全部新增近期压力周期，不用于筛选下一策略。原点资金字段增加后，训练成员、唯一性权重和类内净收益估计改变；2022年1月及7月的原估计pB跨过固定入场门，五笔新增真实周期均亏损。其余共同周期也因原净值风险预算出现投入数量差异，因此不能只相加新增收益代替完整账户。\n\n"
    text += stress_new[["entry_origin", "entry_date", "exit_date_V2", "net_pnl_V2", "net_return_V2",
                       "predicted_p_times_b_V2", "macro_features_on_path_V2"]].to_markdown(index=False, floatfmt=".6f") + "\n\n"
    text += f"五笔新增净损益合计{stress_new.net_pnl_V2.sum():.5f}元，共同周期实际差合计{same.net_pnl_delta.sum():.5f}元，与期末净值差{pair.ending_equity_delta:.5f}元相符。新近期压力期末189882.28916元，旧版194070.28848元。实际毛损益−7088.30元，负收益不能全部归因于费用；不据新增失败提高门槛或改变退出营救。\n\n"
    text += "## 限制与后续方向\n\n"
    text += "当前资金派生缓存和融资资料止2025年末，不等于所有原始资金源都缺2026。原始DR007本地源已到2026年8月14日，是否具有与政策钟一致的完整当时可知合同尚须另核；本次未扩缓存。PMI139月至2026年7月。M1/M2、社融贷款、工业经营、企业调查、基金申赎/折溢价及政策预期各有原合同和旧裁决，不能按涨跌叙事补缺失或把不同频率视为独立票。\n\n"
    text += "下一项具体工作只核对2026原始DR、实际政策利率和融资源覆盖/公布钟，查清未进派生缓存与真实源缺口，0金融拟合；形成可验证的不同完整用途或真正新样本后再登记。当前本配置已关闭，没有待跑金融候选。原E03账户前瞻保持，最早新合格收盘为2026年10月8日15:05。已观察历史属于DEVELOPMENT_CALIBRATION，first-vintage NOT_CERTIFIED，独立NOT_ESTABLISHED，global DSR/PBO NOT_COMPUTED；完整收益夏普目标和去除过拟合未实现。\n\n"
    text += "验证：五项新的源合同测试通过，四原A控制精确复现，八新保存账户和资金库存通过复算；原142月中125个月配对拟合、实际250次拟合，每模型2345可评分日，2007日预测不同且使用宏观路径，原61分段/49上涨及四案例240行保持。四图已实际查看。本报告和表格只解释保存结果，0新拟合/账户/股票标签/采集。\n\n"
    text += "[全部12账户及完整年次数](results/十二账户与统一完整年频率.csv)、[两版所有周期](results/V1与V2全部真实周期身份和净损益比较.csv)、[原17关键日双模型评分](results/原固定17关键日_两模型评分与实际宏观背景.csv)、[全部入场预测变化](results/全部原点两版本预测和入场门变化.csv)、[完整固定结果](summary.json)、[源诊断文字更正](source_diagnostic_status_erratum.json)。\n"
    report = OUT / "多信息源评分_具体上涨与完整结论.md"
    report.write_bytes(text.encode("utf-8"))
    study.write_json(receipt, {"at": study.now(), "status": "PASS_SAVED_SOURCE_AND_ACCOUNT_EXPLANATION",
        "report": report.relative_to(ROOT).as_posix(), "report_sha256": study.digest(report),
        "charts_visually_inspected": [18, 37, 42, 55], "all_original_episode_rows": 61,
        "all_original_admitted_waves": 49, "all_original_case_rows": 240,
        "additional_origins_by_year": extra_years, "all_cycle_memberships": cycles.membership.value_counts().to_dict(),
        "new_recent_stress_cycles": len(stress_new), "new_recent_stress_cycle_net_pnl": float(stress_new.net_pnl_V2.sum()),
        "common_recent_stress_cycle_net_pnl_delta": float(same.net_pnl_delta.sum()),
        "recent_stress_ending_equity_delta": float(pair.ending_equity_delta),
        "new_fits": 0, "new_accounts": 0, "new_labels": 0, "new_market_requests": 0,
        "primary_financial_values_unchanged": True, "candidate_status": summary["status"],
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False}, exclusive=True)
    print("源支持、四上涨及所有账户周期的解释已交付；五新增全亏与完整净值差一致。", flush=True)


if __name__ == "__main__":
    run()
