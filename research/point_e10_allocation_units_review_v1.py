"""E10有限资金单位/旧用途/保存方差资格核对，不生成预测或策略账户。"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_e10_allocation_units_review_v1"
NEXT = ROOT / "reports/research/510300_point_next_information_intake_20261002"
PLAN = NEXT / "E10_allocation_units_and_uncertainty_prior_review_plan.json"
MODELS = ROOT / "reports/research/510300_point_random_intercept_exit_v1/candidate_models.json"
ORIGINAL = ROOT / "reports/research/510300_point_current_observation_20261001/inputs/within_models.json"
STUDY = "510300_POINT_E10_ALLOCATION_UNITS_REVIEW_V1"
STATUS = "NOT_ADMITTED_E10_SAVED_VARIANCE_DIRECT_ALLOCATION_UNITS_AND_PURPOSE_GATE_FAILED"
META_KEYS = ["fit_index", "fit_origin", "fit_time", "status", "training_cycles", "training_cycle_count",
             "training_rows", "latest_exit_index", "latest_exit_date", "eligible_for_fit", "failure", "missing_feature_rows"]
VARIANCE_ROLE = "sigma_e_squared_times_cycle_size_identity_plus_sigma_b_squared_ones"
VARIANCE_FIT = "UNPENALIZED_REML_THEN_ORIGINAL_ALPHA_ONE_GLS_RIDGE"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stamp() -> str:
    return datetime.now().astimezone().isoformat()


def identity(path: Path) -> dict:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path: Path, value) -> None:
    require(not path.exists(), "不覆盖已保存的有限核对文件：" + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a: float, b: float) -> bool:
    """只核保存方差浮点恒等式，容差不用于交易信号或门槛。"""
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def source_excerpt(relative: str, function: str) -> dict:
    path = ROOT / relative
    source = path.read_text(encoding="utf-8-sig")
    candidates = [node for node in ast.walk(ast.parse(source))
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function]
    require(len(candidates) == 1, f"源码函数不再唯一：{relative}/{function}")
    node = candidates[0]
    return {"path": relative, "function": function, "start_line": node.lineno, "end_line": node.end_lineno,
            "source": "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])}


def monthly_metadata() -> list[dict]:
    """仅保存模型时点/成员数/方差角色；不读取样本标签，不重拟合或计算新预测。"""
    originals, candidates = read(ORIGINAL)["models"], read(MODELS)["models"]
    require(len(originals) == len(candidates) == 142, "原月度记录数改变，停止，不缩样本。")
    indexes = [r["fit_index"] for r in originals]
    require(indexes == sorted(set(indexes)), "原月度时点不唯一递增。")
    rows, first_by_identity = [], {}
    for original, record in zip(originals, candidates):
        require(all(original[k] == record[k] for k in META_KEYS), "E08与原月度时钟/训练元数据不同。")
        require(record["latest_exit_index"] <= record["fit_index"], "保存模型使用了未来自然退出。")
        require(record["latest_exit_date"] <= record["fit_origin"], "保存模型退出日期晚于拟合原点。")
        require(record["fit_time"] == record["fit_origin"] + "T15:05:00", "保存拟合时点不是原15:05。")
        row = {"fit_origin": record["fit_origin"], "fit_index": record["fit_index"], "status": record["status"],
               "training_cycles": record["training_cycle_count"], "training_rows": record["training_rows"],
               "latest_exit_date": record["latest_exit_date"], "saved_model_metadata_matches_original": True,
               "saved_variance_available": record["model"] is not None,
               "input_identity": None, "first_estimation_origin": None, "reused": None,
               "residual_variance_scale": None, "random_intercept_variance": None,
               "training_cycle_rows_min": None, "training_cycle_rows_max": None,
               "variance_role_checked": False, "coefficient_sampling_covariance_saved": False,
               "new_cycle_one_day_return_variance_identified": False}
        if record["model"] is None:
            require(original["model"] is None and record["status"] == "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS"
                    and not record["eligible_for_fit"], "原未知模型状态改变，不填方差。")
        else:
            model = record["model"]
            require(record["status"] == "FIT_COMPLETE" and record["eligible_for_fit"]
                    and record["training_cycle_count"] >= 10 and record["training_rows"] >= 100
                    and record["missing_feature_rows"] == 0, "旧成熟资格不符合原门。")
            require(model["residual_covariance_definition"] == VARIANCE_ROLE
                    and model["variance_fit"] == VARIANCE_FIT, "保存协方差或估计用途改变。")
            require(model["new_cycle_random_intercept"] == 0.0
                    and model["new_cycle_intercept_rule"] == "POPULATION_INTERCEPT_WITH_ZERO_UNKNOWN_RANDOM_EFFECT",
                    "不能将成熟训练周期条件随机效应赋给新周期。")
            noise, shared = model["residual_variance_scale"], model["random_intercept_variance"]
            k, ratio = model["between_cycle_mean_weight"], model["variance_ratio"]
            require(all(finite(v) for v in [noise, shared, k, ratio]) and noise > 0 and shared >= 0
                    and 0 < k <= 1 and ratio >= 0, "旧方差参数无效。")
            require(close(k, 1 / (1 + ratio)) and close(shared, ratio * noise), "旧方差保存恒等式失败。")
            groups = model["training_cycle_effects"]
            ids = [g["cycle_id"] for g in groups]
            counts = [g["rows"] for g in groups]
            require(len(set(ids)) == len(ids) == record["training_cycle_count"]
                    and set(ids) == set(record["training_cycles"]) and sum(counts) == record["training_rows"]
                    and all(type(n) is int and n > 0 for n in counts), "保存成熟周期权重/成员数不一致。")
            key = record["input_identity"]
            require(isinstance(key, str) and len(key) == 64, "旧输入身份缺失。")
            first = key not in first_by_identity
            if first:
                first_by_identity[key] = (record["fit_origin"], model)
            first_origin, first_model = first_by_identity[key]
            require(record["reused"] is (not first) and record["first_estimation_origin"] == first_origin
                    and first_origin <= record["fit_origin"] and model == first_model,
                    "旧模型缓存身份或首次估计时钟不一致。")
            require(not any("covariance" in name and name != "residual_covariance_definition" for name in model),
                    "新增保存协方差需分别辨认，不能套用原缺项裁决。")
            row.update(input_identity=key, first_estimation_origin=first_origin, reused=record["reused"],
                       residual_variance_scale=noise, random_intercept_variance=shared,
                       training_cycle_rows_min=min(counts), training_cycle_rows_max=max(counts), variance_role_checked=True)
        rows.append(row)
    require(sum(row["saved_variance_available"] for row in rows) == 115, "旧可用/未知记录数量改变。")
    require(len(first_by_identity) == 25, "旧不同训练输入数量改变。")
    return rows


def prior_quotes() -> dict:
    folder = ROOT / "reports/research"
    def take(name, keys):
        value = read(folder / name)
        require(all(k in value for k in keys), "旧结论字段改变，不猜测缺项。")
        return {k: value[k] for k in keys}
    return {
        "saved_weight": take("510300_point_weight_information_diagnostic_v1/summary.json",
                             ["status", "cross_period_return_and_sharpe_improvement", "comparisons", "independent_validation", "goal_achieved"]),
        "second_weight": take("510300_point_second_weight_comparison_v1/summary.json",
                              ["status", "all_four_economic_screens_pass", "economic_gates", "cross_period_return_and_sharpe_improvement"]),
        "adaptive": take("510300_adaptive_allocation_v1/result.json", ["status", "primary", "number_of_candidates", "independent_validation"]),
        "target_transmission": take("510300_point_exit_target_transmission_diagnostic_v1/summary.json",
                                    ["status", "all_decision_rows", "holding_origin_rows", "new_account_evaluations", "new_model_fits"]),
        "cashflow_state": take("510300_point_account_cashflow_state_v1/summary.json",
                                ["status", "field_definition_gate", "predictive_information_gate", "strategy_increment_status", "interpretation"]),
        "e08_mean_method": take("510300_point_random_intercept_exit_v1/prediction_summary.json",
                                 ["status", "prediction_gate_passed", "periods", "new_strategy_accounts", "account_return_sharpe"]),
        "e09_target": take("510300_point_e09_label_objective_review_v1/summary.json",
                            ["status", "metadata_groups", "original_target_error_established", "new_horizon_chosen", "new_return_labels", "new_model_fits"]),
    }


def units() -> list[dict]:
    return [
        {"name": "原自然继续目标Y", "unit": "参考份额单位净收益差",
         "numerator": "q*(自然退出卖出成交价-下一开盘卖出成交价)-自然退出佣金+下一开盘佣金+新增现金权益",
         "denominator": "q*下一开盘原始价", "horizon": "下一开盘至参考自然退出；变长，既有保存1—59区间",
         "clock": "标签只在原自然周期结束及权益成熟后可用于训练；下一开盘价格不是当前15:05已知值。",
         "allocation_limit": "预测单位优势不是下一日标的收益或全账户人民币期望；E[q*P_next*Y|I_t]一般不等于未知P_next乘E[Y|I_t]。"},
        {"name": "原cycle_return输入", "unit": "参考持仓累计收益",
         "numerator": "当前参考库存价值及已确认权益减原入场支出", "denominator": "参考原入场支出",
         "horizon": "参考进入至当前收盘", "clock": "当前收盘已知状态；与Y分母不同是用途差异，非标签错误。",
         "allocation_limit": "不能用未来完整周期结果修正当前状态或直接换成账户累计现金流。"},
        {"name": "真实现金流周期状态", "unit": "实际累计成交周期收益",
         "numerator": "累计卖出净收入+已确认股息+当前剩余份额市值-累计实际买入支出",
         "denominator": "截至当日累计实际买入支出", "horizon": "实际进入至当前收盘",
         "clock": "只使用已成交库存/已发生权益；费用和加减仓路径已经发生。",
         "allocation_limit": "E04定义核对通过，但预测增量未接纳；不同q和最低佣金下不能将参考标签简单线性缩放。"},
        {"name": "完整账户逐日净收益", "unit": "20万元账户净值收益率",
         "numerator": "当日净值减前日净值", "denominator": "前日净值",
         "horizon": "相邻完整股票交易日，含现金/应收/库存/佣金/滑点", "clock": "次开盘只可执行前收盘计划，开盘现金/风险仅缩减。",
         "allocation_limit": "收益及夏普由完整净值路径计算，不是事件毛收益/原Y或费用代理的年化。"},
        {"name": "E08两项保存方差", "unit": "原单位继续优势Y的拟合残差平方单位",
         "numerator": "同周期拟合协方差sigma_e^2*n_j*I+sigma_b^2*11'", "denominator": "没有全账户资金分母；n_j为成熟周期保存状态数",
         "horizon": "原变长自然继续目标及周期等权训练结构；不是固定次日标的区间",
         "clock": "115个成熟月的参数元数据可核；27无模型保留。新周期完整n_j及条件随机效应未知。",
         "allocation_limit": "没有保存系数/均值估计协方差或新周期下一日回报方差；不能直接sqrt后称95%预测区间、置信下界或Kelly分母。"},
        {"name": "旧adaptive效用", "unit": "同一预定h期的单位NAV分数",
         "numerator": "w*mu-(gamma/2)*w^2*(h*variance60)-friction/NAV", "denominator": "费用除以当前NAV",
         "horizon": "旧固定5/20日原开盘至原开盘含权益收益；h倍历史方差是旧假设",
         "clock": "用原成熟训练及当前历史60日方差，前收盘计划下一开盘",
         "allocation_limit": "旧主方案未达目标；E10不把同效用换名再跑、不调整gamma或权重网格救回。"},
    ]


def run() -> None:
    require(not OUT.exists(), "E10有限核对已建立，不覆盖或重复运行。")
    plan = read(PLAN)
    require(plan["question_id"] == "TECH.E10" and plan["new_method_bound"] is False, "E10提案身份或权限改变。")
    paths = {PLAN, Path(__file__).resolve()}
    for source in plan["sources"]:
        path = ROOT / source["path"]
        require(identity(path)["sha256"] == source["sha256"], "E10原提案来源改变：" + source["path"])
        paths.add(path)
    # 有限补齐直接依赖：当前原模型身份、版本选择函数及已完成E09原目标定义。
    paths.update([ORIGINAL, ROOT / "research/entry_vintage_exit_inputs_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py",
                  ROOT / "reports/research/510300_point_e09_label_objective_review_v1/protocol.json",
                  ROOT / "reports/research/510300_point_e09_label_objective_review_v1/summary.json",
                  ROOT / "reports/research/510300_point_e09_label_objective_review_v1/prior_definition_review.json"])
    manifest = [identity(path) for path in sorted(paths)]
    rows, quotes, unit_rows = monthly_metadata(), prior_quotes(), units()
    excerpts = [source_excerpt("research/learned_cycle_exit_v1.py", "continuation_label"),
                source_excerpt("research/learned_cycle_exit_v1.py", "state_values"),
                source_excerpt("research/intraday_overnight_increment_v1.py", "choose_order"),
                source_excerpt("research/point_weight_information_inputs_v1.py", "decide")]
    e09 = read(ROOT / "reports/research/510300_point_e09_label_objective_review_v1/prior_definition_review.json")
    require(e09["original_expression"] == "[q*(late_fill-early_fill)-late_commission+early_commission+q*新增权益]/(q*early_raw_open)",
            "已完成E09目标单位描述改变，不能据另一单位继续。")
    OUT.mkdir(parents=True)
    at = stamp()
    protocol = {"study": STUDY, "at": at, "question_id": "TECH.E10", "classification": "FINITE_SAVED_METADATA_AND_UNITS_QUALIFICATION",
                "prior_plan": PLAN.relative_to(ROOT).as_posix(), "old_results_already_observed": True,
                "evidence_identity_is_new_empirical_preregistration": False,
                "clock_checks_scope": "只核原保存月模型时点/完整周期数/缓存/方差定义；非训练逐行标签、物理首次发布、实际部署或误差校准验证。",
                "source_expansion_scope": "只补原142模型、版本选择及已完成E09的六份直接依赖，不扩市场数据。",
                "new_method_bound": False, "new_field_bound": False, "new_strategy_configurations": 0,
                "new_model_fits": 0, "new_return_labels": 0, "new_predictions": 0, "new_strategy_accounts": 0, "new_market_collection": 0,
                "original_target_and_e03_unchanged": True, "old_failures_reopened": False,
                "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    write(OUT / "protocol.json", protocol)
    write(OUT / "evidence_manifest.json", manifest)
    write(OUT / "monthly_variance_role_metadata.json", rows)
    write(OUT / "prior_method_quotes.json", quotes)
    write(OUT / "units_and_role_table.json", unit_rows)
    write(OUT / "source_function_excerpts.json", excerpts)
    with (OUT / "units_and_role_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(unit_rows[0]))
        writer.writeheader()
        writer.writerows(unit_rows)
    ready = [row for row in rows if row["saved_variance_available"]]
    summary = {**protocol, "status": STATUS, "monthly_records_checked": len(rows), "saved_variance_ready_months": len(ready),
               "original_no_model_months_preserved": len(rows) - len(ready),
               "distinct_saved_training_identities": len({r["input_identity"] for r in ready}),
               "metadata_matches_original_records": True, "saved_variance_definition_identity_checked_months": len(ready),
               "first_ready_month": ready[0]["fit_origin"], "last_ready_month": ready[-1]["fit_origin"],
               "full_original_training_labels_rechecked": False, "physical_first_publication_established": False,
               "new_cycle_one_day_return_variance_identified": False, "current_prediction_member_support": "NOT_COMPUTED",
               "current_financial_return_sharpe": "NOT_COMPUTED", "evidence_files": len(manifest),
               "direct_allocation_gate": "FAILED_NO_SAME_HORIZON_CAPITAL_MAPPING_OR_IDENTIFIED_NEW_CYCLE_RETURN_VARIANCE",
               "rejection_scope": "本批既有Y/E08残差方差直接当下一日资金配置均值/风险的用途；不否定所有资金分配或所有不确定性方法。",
               "next_research_direction": "不同融资需求信息，先有限核F01—F03旧用途和实际来源；尚未准入。"}
    write(OUT / "summary.json", summary)
    report = """# E10资金单位、旧用途与保存方差资格结论

E10有限核对完成：这批现有自然继续预测和E08方差，不能直接用作下一日资金配置的收益/风险参数。没有登记新仓位方法，也没有新收益或夏普结果。

原标签不是已发现的错误。它是下一开盘提前卖出与持有至参考自然退出的净差额，分母为参考份额乘下一开盘原价；不是原入场成本。原cycle_return输入才以参考入场支出为分母。真实账户状态另以累计实际买入支出为分母。三者服务不同问题，单位区别不能直接证明换标签或换输入会提高收益。

| 已有量 | 正确用途 | 不能直接做的替换 |
|---|---|---|
| 原继续优势Y | 参考自然退出相对下一开盘提前卖出的单位净优势，保存剩余区间1—59 | 下一日标的收益、完整账户CNY利润 |
| 真实现金流状态 | 已成交加减仓、已发生权益和当前库存的累计收益 | 替换参考状态后宣称模型已修复 |
| E08 sigma_e²/sigma_b² | 原继续目标的周期等权拟合协方差参数 | 下一日波动、系数估计误差、95%预测区间或Kelly分母 |
| 完整账户逐日收益 | 前后净值含现金/库存/应收/费用的变化 | 事件毛收益或原Y简单年化 |

全部142个月保存模型与原记录的时点、周期成员数和成熟上界元数据一致；115个成熟月的方差定义/恒等式及缓存身份核对通过，27个无模型保留，25个不同旧训练身份。这里只检查保存元数据，没有重算训练标签、REML、系数、预测或账户，也没有认证物理历史首次发布。

E08协方差为sigma_e²*n_j*I+sigma_b²*11'，n_j是已完成训练周期保存状态数。新周期完整n_j当时未知；训练周期条件随机效应不能赋给新周期。保存记录也没有系数/均值估计协方差。参数在模型时点可核，不代表某个新交易区间的风险已经识别。

从原Y换算人民币优势，需要参考资金基数及相同费用/权益/持有区间。当前收盘不知道下一开盘P_next，一般不能把E[q*P_next*Y|I_t]简化成q乘未知P_next再乘预测E[Y|I_t]。不同实际份额、最低佣金、加减仓及现金再部署也会改变目标。现有协议尚未建立这一资金映射，不临时选择近似式营救。

旧资金配置方法已经核实：保留原仓位强弱相对二元方案的四场景收益/夏普点值同向提高，但区间跨0且完整目标未达；B替代A固定比较被拒绝；旧adaptive已采用均值−风险−交易费用效用、主目标未达。E01已核库存传递，E04已核现金流定义，E09未接纳不同新目标；E08均值方法失败保持。不能通过重命名、调整gamma/权重网格/置信阈值或借用未来周期长度重开这些方向。

本裁决只拒绝“现有自然继续Y和E08残差方差直接用于次日资金配置”。它没有检验所有不确定性方法或证明资金配置永远无效。要进入实质不同的固定实验，须先有相同收益区间和资金单位、当时可知且完整支持的风险信息，保留原预测门及全部账户目标；本轮没有这样的已接纳候选。

下一步有限核F01—F03融资需求的真实旧用途及来源。这是外部融资活动信息的候选，不继续调旧仓位参数；目前只有旧用途/来源提案，未绑定字段、拟合残差或运行账户。原A、原失败、E03前瞻比较和其他研究线均保持；收益和夏普完整目标未达。

直接事实：[汇总](summary.json)、[六类资金单位](units_and_role_table.json)、[142月保存元数据](monthly_variance_role_metadata.json)、[旧方法原文摘录](prior_method_quotes.json)、[源码函数](source_function_excerpts.json)。
"""
    (OUT / "研究结论与下一步.md").write_text(report, encoding="utf-8")
    print(json.dumps({"裁决": STATUS, "月记录": len(rows), "保存方差可核": len(ready), "无模型保持": 27,
                      "旧不同训练身份": 25, "新拟合/标签/预测/账户": 0}, ensure_ascii=False))


def verify() -> None:
    path = OUT / "saved_output_verification_receipt.json"
    require(not path.exists(), "保存输出已核对，不重复执行。")
    manifest = read(OUT / "evidence_manifest.json")
    for source in manifest:
        require(identity(ROOT / source["path"]) == source, "有限来源身份改变：" + source["path"])
    require(read(OUT / "monthly_variance_role_metadata.json") == monthly_metadata(), "142月保存元数据不相同。")
    require(read(OUT / "prior_method_quotes.json") == prior_quotes(), "旧用途保存摘录不相同。")
    require(read(OUT / "units_and_role_table.json") == units(), "六类单位描述不相同。")
    with (OUT / "units_and_role_table.csv").open(encoding="utf-8-sig", newline="") as stream:
        require(list(csv.DictReader(stream)) == units(), "六类单位表格不相同。")
    for excerpt in read(OUT / "source_function_excerpts.json"):
        require(source_excerpt(excerpt["path"], excerpt["function"]) == excerpt, "源码摘录不相同。")
    summary = read(OUT / "summary.json")
    require(summary["status"] == STATUS and summary["monthly_records_checked"] == 142
            and summary["saved_variance_ready_months"] == 115 and summary["new_strategy_accounts"] == 0,
            "汇总身份或计数不同。")
    result = {"at": stamp(), "status": "PASS_SAVED_UNITS_PRIOR_QUOTES_AND_VARIANCE_METADATA",
              "source_files_checked": len(manifest), "monthly_records_checked": 142, "variance_definition_checks": 115,
              "unknown_original_model_records_preserved": 27, "old_distinct_training_identities": 25,
              "new_model_fits": 0, "new_return_labels": 0, "new_predictions": 0, "new_strategy_accounts": 0,
              "claim_limit": "仅保存元数据/单位/旧事实一致；非误差校准、物理首版、风险模型或金融验证。"}
    write(path, result)
    print(json.dumps({"保存核对": result["status"], "来源": len(manifest), "月记录": 142, "新拟合/账户": 0}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="E10有限单位和既有方差角色诊断，不运行模型或账户。")
    parser.add_argument("--verify", action="store_true", help="核对保存事实和142月元数据一次。")
    args = parser.parse_args()
    verify() if args.verify else run()


if __name__ == "__main__":
    main()
