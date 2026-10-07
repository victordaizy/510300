"""汇总十一份已经保存的可选修正结果，不重估、不重采样、不开账户。"""
import json

import pandas as pd

from research.point_account_cashflow_state_v1 import ROOT, now, digest, require, write_json


OUT = ROOT / "reports/research/510300_point_d02_optional_correction_v1"
CASES = [
    ("macro", "K04新订单与库存"), ("funding", "K05 DR007资金价格"),
    ("h03", "H03 IF持仓与基差"), ("j01", "J01股债联合"),
    ("c04", "C04回调成交额比"), ("a02", "A02突破后保留率"),
    ("c03", "C03十日量价效率差"), ("c01", "C01负正冲击中位数比"),
    ("b03", "B03破低后回收"), ("b04", "B04急跌后波动与低点"),
    ("d02", "D02低开缺口吸收与缺口大小"),
]


def read(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def main():
    require(not (OUT / "finite_optional_phase_summary.json").exists(), "本有限汇总已完成，不重复生成。")
    cases, sources, flat = [], [], []
    for key, title in CASES:
        folder = ROOT / "reports/research" / ("510300_point_" + key + "_optional_correction_v1")
        result, verification = read(folder / "prediction_summary.json"), read(folder / "verification.json")
        require(not result["prediction_gate_passed"] and result["economic_stage"] == "SKIPPED_PREDICTION_GATE_FAILED",
                "有限清单出现已通过模型或不同经济阶段，不能沿用拒绝汇总。")
        require(result["all_natural_rows"] == 1507 and result["paired_available_predictions"] == 1010
                and result["original_unknown_predictions_preserved"] == 497, "有限清单原成员发生变化。")
        require(verification["status"].startswith("PASS_") and verification["new_model_fits"] == 0
                and verification["new_accounts"] == 0, "有限清单复算凭据不完整。")
        accounting = result["accounting"]
        require(accounting["candidate_configurations"] == 1 and accounting["auxiliary_coefficient_estimations"] == 25
                and accounting["monthly_cache_reuses"] == 90 and accounting["core_model_reestimations"] == 0
                and accounting["new_accounts"] == 0 and accounting["new_return_labels"] == 0,
                "有限清单实际工作量不同，应重新区分计数。")
        periods = result["periods"]
        require([p["period"] for p in periods] == ["2015_2019", "2020_2026"], "固定双期发生变化。")
        for p, count, cycles, mse in zip(periods, [262, 748], [6, 18],
                                       [0.006796098625414917, 0.0055065266523754795]):
            require(p["paired_rows"] == count and p["cycles"] == cycles
                    and p["baseline_raw_return_mse"] == mse, "原配对、周期或基准误差不一致。")
        case = {"key": key, "title": title, "decision": result["technical_decision"],
                "status": result["status"], "prediction_gate_passed": False,
                "periods": periods, "accounting": accounting,
                "economic_stage": result["economic_stage"],
                "net_cagr": None, "net_sharpe": None,
                "result_path": (folder / "prediction_summary.json").relative_to(ROOT).as_posix(),
                "verification_path": (folder / "verification.json").relative_to(ROOT).as_posix()}
        cases.append(case)
        flat.append({"方向": title, "决策": case["decision"],
                     "早期MSE相对变化百分数": periods[0]["relative_mse_change"] * 100,
                     "早期预测门": periods[0]["prediction_gate_passed"],
                     "近期MSE相对变化百分数": periods[1]["relative_mse_change"] * 100,
                     "近期预测门": periods[1]["prediction_gate_passed"], "整体预测门": False,
                     "辅助估计次数": 25, "新账户数": 0})
        for filename in ["protocol.json", "freeze.json", "source_summary.json", "prediction_summary.json", "verification.json"]:
            path = folder / filename
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
    early_pass = [c["key"] for c in cases if c["periods"][0]["prediction_gate_passed"]]
    recent_pass = [c["key"] for c in cases if c["periods"][1]["prediction_gate_passed"]]
    require(early_pass == ["j01", "b04"] and not recent_pass, "有限阶段通过情况变化。")
    b02 = read(OUT / "next_B02_saved_field_design_preflight.json")
    require(not b02["design_gate_passed"] and b02["rank_counts"]["3"] == 3
            and b02["identifiable_months"] == 112 and not b02["target_column_read"], "B02资格事实不一致。")
    summary = {
        "at": now(), "technical_decision": "TECH.R139",
        "status": "COMPLETED_BOUNDED_ELEVEN_OPTIONAL_FUNCTION_COMPARISONS_NO_ADMISSION",
        "scope": "仅清单十一固定核心之外可选残差表达及B02原五列设计资格；不是全项目/全部技术分析/全部历史试验总表。",
        "cases": cases, "sources": sources, "function_configurations": len(cases),
        "accepted_function_configurations": 0, "early_only_gate_passed": early_pass,
        "recent_gate_passed": recent_pass,
        "total_auxiliary_coefficient_estimations": sum(c["accounting"]["auxiliary_coefficient_estimations"] for c in cases),
        "total_monthly_cache_reuses": sum(c["accounting"]["monthly_cache_reuses"] for c in cases),
        "original_core_reestimations": 0, "new_return_labels": 0, "new_accounts": 0,
        "original_natural_rows_shared_not_added": 1507,
        "paired_predictions_shared_not_added": 1010,
        "original_unknown_predictions_shared_not_added": 497,
        "natural_cycles_shared_by_period_not_added": [6, 18],
        "trial_interpretation": "11固定函数、275实际辅助方程估计、990月版本复用；同原样本/滚动版本，不是275独立策略或264独立周期。合成测试、区间抽样及保存复算不加入金融候选数。",
        "b02_design_result": "next_B02_saved_field_design_preflight.json",
        "phase_action": "结束这11个固定表达及B02五列资格路线。不按有利单期挑选、混合、换窗或加交互救回；其他独立分支授权保持。",
        "inference_limits": "拒绝这些固定表达，没有证明其经济机制或所有技术分析永远无效；MSE变化不是收益率变化，预测状态不是独立交易。无候选过门，不计算或倒推收益/夏普。",
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "physical_first_vintage_verified": False, "global_DSR_PBO": "NOT_COMPUTED",
        "overfit_removed": False, "goal_achieved": False,
        "new_model_fits_in_compilation": 0, "new_network_requests_in_compilation": 0,
    }
    write_json(OUT / "finite_optional_phase_summary.json", summary, exclusive=True)
    pd.DataFrame(flat).to_csv(OUT / "有限可选修正实际结果.csv", encoding="utf-8-sig", index=False)
    rows = []
    for c in cases:
        a, b = c["periods"]
        rows.append(f"| {c['title']} | {c['decision']} | {a['relative_mse_change'] * 100:+.6f}% | {'PASS' if a['prediction_gate_passed'] else 'FAIL'} | {b['relative_mse_change'] * 100:+.6f}% | {'PASS' if b['prediction_gate_passed'] else 'FAIL'} | FAIL |")
    text = """# 有限可选修正阶段结论

本批十一项实际模型均未通过原双期预测门槛，接受数量0，未进入新账户检验。TECH.R139是本有限阶段结论，最新实际金融模型仍为TECH.R137。结果来自各自已经冻结并保存的原比较，本汇总没有重估、重抽样或重新选择窗口。

两期分别为2015—2019及2020—2026-09-30。两期完整配对262/748状态，6/18自然周期；原基准周期等权MSE为0.006796098625414917及0.0055065266523754795。原门要求两期点估计均严格改善，而且5000次原入场年块的改善95%下界均严格为正。负百分数是预测误差下降，不能解释成账户收益下降或上涨。

| 固定方向 | 结果决策 | 早期MSE相对变化 | 早期门 | 近期MSE相对变化 | 近期门 | 整体门 |
|---|---|---:|---|---:|---|---|
""" + "\n".join(rows) + """

只有J01股债联合与B04急跌修正在早期通过；近期十一项全失败。不得把早期J01/B04与近期其他较好的点估计拼成新策略，也不能因为D02两期点估计都下降就降低原区间门。

B02另做原五列的事前设计资格：原3488日线43次完整记录；1507自然状态仅15完整、1492未知。原115成熟月中112个月秩5、最早3个月秩3，已知训练行仅4—12；标准SVD门事前冻结。原可用预测早期1/262、近期8/748具备五项。固定五列资格拒绝，目标列未读，金融模型/新账户0。不删除两列、跳过三个月或扩展二测窗口。原TECH.R102完整字段支持失败与旧T02/T13终态保持。

十一函数合计275次实际辅助估计、990次月版本复用，原核心重拟合0，新标签/账户0。共用的1507自然状态、1010配对与6/18周期不按函数数量累加；275不是独立策略数。所有历史仍是DEVELOPMENT_CALIBRATION，历史首次发布版本未认证，独立验证NOT_ESTABLISHED，全项目DSR/PBO NOT_COMPUTED。减少参数搜索不等于已经去除过拟合。

结束这批固定表达的历史开发。后续值得继续的是另一个有明确决策机制、与已有失败实质不同且能证明事前可得的隔离研究，或原E03两项登记的真实前瞻验证。新假设先核旧用途、来源及完整成员支持，再冻结唯一表达；不能从本表有利单期选函数或组合权重。现有E03登记、2026-10-08 15:05首个合法新收盘及0真实新账户日/闭合周期不重置、不回填。其他分支的实验及授权独立保留。

完整目标尚未完成。没有新的净收益或夏普优势，也没有晋升为独立验证通过的策略。
"""
    with (OUT / "有限可选修正阶段结论.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    route = {
        "at": now(), "status": "ROUTE_REVIEW_ONLY_NO_NEW_FINANCIAL_CANDIDATE_ADMITTED",
        "parent_phase_decision": "TECH.R139", "latest_actual_model_decision": "TECH.R137",
        "closed_routes": [c["key"] for c in cases] + ["b02_fixed_five_column_optional_function"],
        "next_concrete_experiment": {
            "name": "原E03 SAVED_WEIGHT与POINT_BINARY真实前瞻比较",
            "protocol": "reports/research/510300_point_weight_forward_v1/protocol.json",
            "registration_unchanged": True, "first_eligible_new_close_at": "2026-10-08T15:05:00+08:00",
            "activation": "真实合格新收盘与其完整本次来源回执到达，按原已登记协议；不从今天的历史开发反选候选。",
            "before_activation": "不制造新日期、回填历史或计算尚未发生账户收益。",
        },
        "parallel_research_route": "可在原授权范围内复核新的实质机制，核旧用途、源时钟与原成员；只有合格区别才另登记隔离方案。当前没有具体新收益模型被准入，其他分支权限不改。",
        "no_rescue": "不重跑本批失败表达或近邻参数，不按一时期选结果、混合或重标未知。",
        "new_model_fits": 0, "new_accounts": 0, "new_market_requests": 0,
        "goal_status": "active", "goal_achieved": False,
    }
    write_json(OUT / "next_research_route.json", route, exclusive=True)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["cases", "sources"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
