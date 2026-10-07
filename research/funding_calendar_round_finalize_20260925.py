"""保存利率可预测成分与意外变化研究的结果，不重估策略。"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now

OUT = ROOT / "reports/research/510300_funding_calendar_innovation_daily_v1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
PRIMARY = "PRICE_AND_FUNDING_INNOVATION"
RAW = "PRICE_AND_RAW_FDR"


def run():
    result = read(OUT / "result.json")
    assert result["status"] == "FROZEN_NO_QUALIFIED_FUNDING_INNOVATION_STRATEGY"
    metrics = {row["policy"]: row for row in result["all_accounts"] if row["cost"] == "STRESS"}
    primary, raw, price = metrics[PRIMARY], metrics[RAW], metrics["PRICE"]
    forecast = result["funding_forecast_evaluation"]
    mse_vs_persistence = 1-forecast["CALENDAR"]["MSE"]/forecast["PERSISTENCE"]["MSE"]
    mse_vs_ar = 1-forecast["CALENDAR"]["MSE"]/forecast["AR"]["MSE"]
    mae_vs_persistence = 1-forecast["CALENDAR"]["MAE"]/forecast["PERSISTENCE"]["MAE"]
    f_all, f_down = result["forecast_evaluation"]
    pairs = {row["control"]: row for row in result["comparisons"] if row["cost"] == "STRESS"}
    training = pd.read_parquet(OUT / "results/update_receipts.parquet")
    ready = training[training.status.eq("UPDATED")]
    funding = pd.read_parquet(OUT / "results/current_funding_predictions.parquet")
    unique = funding.sort_values("decision_idx").drop_duplicates("source_idx", keep="first")
    periods = []
    for lo, hi in [("2020-01-02", "2023-12-31"), ("2024-01-01", "2026-09-24")]:
        part = unique[unique.decision_date.between(lo, hi)]
        periods.append({"start": lo, "end": hi, "source_dates": len(part),
                        "MSE": {name: float(((part.actual-part[name])**2).mean()) for name in ["PERSISTENCE", "AR", "CALENDAR"]}})
    attribution = []
    for name in ["PRICE", RAW, PRIMARY]:
        ledger = pd.read_parquet(OUT / "accounts/STRESS" / name / "ledger.parquet")
        item = {"policy": name, "same_share_path_gross_pnl": float((ledger.price_pnl+ledger.dividend_recognized).sum()),
                "execution_cost": float((ledger.commission+ledger.slippage_cost).sum()),
                "terminal_reserve": float(ledger.terminal_exit_reserve.iloc[-1]),
                "net_profit": float(ledger.equity.iloc[-1]-200000)}
        assert abs(item["same_share_path_gross_pnl"]-item["execution_cost"]-item["terminal_reserve"]-item["net_profit"]) < 1e-6
        attribution.append(item)
    save(OUT / "results/saved_interpretation.json", {
        "funding_mse_improvement_vs_persistence": mse_vs_persistence,
        "funding_mse_improvement_vs_ar": mse_vs_ar,
        "funding_mae_improvement_vs_persistence": mae_vs_persistence,
        "funding_fixed_periods_descriptive": periods,
        "equity_training_rows_min": int(ready.n_train.min()), "equity_training_rows_max": int(ready.n_train.max()),
        "same_share_path_accounting_attribution_not_zero_cost_strategies": attribution,
        "new_fits_or_accounts_in_delivery": 0,
    })
    report = f"""本轮完成资金利率事前预测和6个完整账户，尚未找到符合要求的510300高夏普策略。日历信息使FDR007利率预测均方误差相对沿用上次利率下降{mse_vs_persistence:.2%}，但新增的利率模型意外没有形成可交易收益优势：主压力账户夏普{primary['sharpe']:.6f}、复合年化{primary['annual_return']:.4%}、最大回撤{abs(primary['max_drawdown']):.4%}，期末{primary['end_equity']:.2f}元。资金预测改善与账户未通过同时保留，高夏普目标仍active。

本轮检验的区别。

已有宏观持续状态、按状态配置策略、日历直接择时及原始回购差值实验均已做过，直接重复没有意义。本轮只问：资金利率中的可预期日历成分与其余变化分开后，是否更能区分同样下跌后的修复和延续。

央行报告说明银行体系资金需求受季末、税期和节假日等因素影响，为这个区分提供机制背景，不构成510300策略有效性证明。[中国人民银行2016年第三季度货币政策执行报告](https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3066656/61ed5a0548c542428584bccc244c9d5c/2016110818532756152.pdf)。本轮日历模型只包含自然月初3日、月末5日、季末月末5日，没有假设这些简单变量穷尽税期、春节、监管或政策冲击。

复用已取得的2328日官方FDR007上午定盘。它不是全天DR007，也不是中性利率。按源日12:00计划可用合并到A股09:00，来源最多7自然日；历史首版和首次送达仍未认证。没有新增市场下载，没有扩大可交易资产。

预测怎样避免引入未来或超过两年的记忆。

资金模型使用前一个定盘、前20个定盘均值和三项日历变量。固定训练内标准化岭回归、alpha为10、至少126条利率训练样本，预测与实际值之差以百分点表示。这个差只是模型意外，不是市场一致预期差，也不能识别纯粹资金供给冲击。

每个A股决策日都从最近两个日历年的原始利率重建历史预测。窗口最前20行只作滞后输入；窗内每个源日j只用j之前的样本拟合，当前待预测数值和后续数值不参与。历史训练特征同样嵌套重建，没有复用含两年前资料的旧参数。三个股票方案共享完整可用训练池，实际每个日更窗口有{int(ready.n_train.min())}—{int(ready.n_train.max())}条成熟股票收益标签，较完整两年更少，因为利率模型需要先建立历史。

股票层保持固定的3项价格特征和126近邻条件分布。三种方案分别为价格、价格加原始FDR007、价格加利率模型意外，每种BASE/STRESS两档成本，合计6个新完整账户。HISTORY为同池经验分布预测对照。没有搜索方向、窗口、k值、阈值或岭回归惩罚。

第一层结果：资金利率确有部分可预测成分。

每个源日只取第一次进入A股决策的预测，共{result['unique_funding_evaluation_days']}个不重复日期。加入日历后的利率均方误差，相对沿用上次利率下降{mse_vs_persistence:.2%}，相对仅用过去利率下降{mse_vs_ar:.2%}。平均绝对误差相对沿用上次利率仅下降{mae_vs_persistence:.2%}，因此不能把较大的均方误差改善解释成每一天都显著更准。固定两时期的误差另保存作描述，没有挑选表现较好区间。

为了严格满足两年限制，实际重建了{result['nested_funding_forecasts_each_model']}组历史源日预测，每组AR和日历两个模型，共{result['funding_regression_fits']}次回归。这些是重复嵌套计算，绝不是同等数量的独立市场样本。股票层保存{result['daily_distribution_updates']}份分布，来自{result['unique_prediction_days']}个日期、每日期4个模型，成熟评价日期{result['mature_prediction_days']}个。

第二层结果：股票预测误差改善很小。

全部成熟股票样本中，利率意外方案相对价格方案的MSE点值下降{f_all['primary_mse_improvement_vs_price']:.2%}，相对原始利率方案下降{f_all['primary_mse_improvement_vs_raw']:.2%}；在793个事前五日下跌样本中，分别下降{f_down['primary_mse_improvement_vs_price']:.2%}和{f_down['primary_mse_improvement_vs_raw']:.2%}。两口径下仍未超过同池历史均值的MSE，不能称为稳定的收益预测优势。

第三层结果：完整账户仍不达标。

全部账户覆盖2020-01-02至2026-09-24，共1633个交易日，20万元连续权益、全部空仓日均计入，不重置年度本金。现金收益和无风险基准按零、242日年化，与最近实验一致。压力结果为：

- 价格对照：夏普{price['sharpe']:.6f}，复合年化{price['annual_return']:.4%}，最大回撤{abs(price['max_drawdown']):.4%}，期末{price['end_equity']:.2f}元，36个闭合周期。
- 原始FDR007：夏普{raw['sharpe']:.6f}，复合年化{raw['annual_return']:.4%}，最大回撤{abs(raw['max_drawdown']):.4%}，期末{raw['end_equity']:.2f}元，29个闭合周期。
- 利率模型意外：夏普{primary['sharpe']:.6f}，复合年化{primary['annual_return']:.4%}，最大回撤{abs(primary['max_drawdown']):.4%}，期末{primary['end_equity']:.2f}元，36个闭合周期。

所有末端均无仓位。本轮价格对照特意使用与两阶段方案相同的训练资格，不能与上一轮不同训练池的价格对照直接混比，也不能从各轮选一个较差对照来宣称增量。

主方案相对原始利率的压力算术年化增量为{pairs[RAW]['annual_arithmetic_difference']*100:.4f}个百分点，95%区块区间[{pairs[RAW]['lower_95']*100:.4f}, {pairs[RAW]['upper_95']*100:.4f}]个百分点，两个固定时期点值均为负。相对价格的点值增量为{pairs['PRICE']['annual_arithmetic_difference']*100:.4f}个百分点，但95%区间[{pairs['PRICE']['lower_95']*100:.4f}, {pairs['PRICE']['upper_95']*100:.4f}]也包含零。以上区间未对项目全部尝试作选择校正。主方案滚动两年三目标联合通过次数为0。

固定已成交份额做会计归因，主方案毛损益为{attribution[2]['same_share_path_gross_pnl']:.2f}元，实际交易费用为{attribution[2]['execution_cost']:.2f}元。毛损益本身已负，费用并非唯一原因。此拆分不是重新运行的零成本策略，不据此调低费用挽救原方案。

现行目标保持为压力夏普至少1.2、复合年化至少10%、最大回撤不超过10%，可交易资产仅510300与现金。目标仓位50%、5日ES95为2.5%、10%跳空预算5%且受距90%权益峰值余量的一半限制；调仓及退出费用均纳入。T+1、整手、现金和分红已在六账户验证，全部保存分布已复算，两处未来标签隔离检查通过。原失败和参数保持固定。

本轮结论是：已经能够更好地预测部分资金利率波动，尚未证明这项改善能在次开盘之后提供510300收益优势。它更准确地定位了缺口，不支持把这个利率模型晋升为交易策略。新增独立前向观察仍为0，没有形成可宣称稳定达标的策略。总目标保持active；本轮没有制作审核包或用户表格。
"""
    report_path = OUT / "本轮研究结论.md"
    report_path.write_text(report, encoding="utf-8")
    completed = {
        "at": now(), "study_id": result["study_id"],
        "status": "FUNDING_FORECAST_IMPROVED_SIX_ACCOUNTS_COMPLETED_TARGET_UNMET",
        "new_unique_full_accounts": 6, "new_market_downloads": 0,
        "funding_source_days_reused": 2328, "unique_funding_evaluation_days": len(unique),
        "funding_mse_improvement_vs_persistence": mse_vs_persistence,
        "funding_mse_improvement_vs_ar": mse_vs_ar,
        "nested_funding_regression_fits": result["funding_regression_fits"],
        "nested_fits_are_independent_observations": False,
        "daily_distribution_updates": result["daily_distribution_updates"],
        "unique_equity_prediction_days": result["unique_prediction_days"],
        "primary": primary, "raw_rate_control_stress": raw, "price_control_stress": price,
        "goal_status": "active", "goal_achieved": False,
        "independent_forward_observations": 0, "orders_authorized": False, "review_package_created": False,
    }
    save(OUT / "completed_round.json", completed)
    relative = OUT.relative_to(ROOT).as_posix()
    classification = "PROGRESS_CAUSAL_FUNDING_FORECAST_AND_SIX_FULL_ACCOUNTS"
    path = MAIN / "current_status.json"
    status = read(path)
    if result["study_id"] not in status["completed_followup_studies"]:
        status["completed_followup_studies"].append(result["study_id"])
    status.update(
        updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification,
        same_condition_consecutive_no_progress_goal_turns=0,
        latest_funding_innovation_round=completed,
        latest_user_requested_study=relative,
        latest_user_requested_study_status=completed["status"],
        research_execution_state=completed["status"],
        latest_continuation_report=relative + "/本轮研究结论.md",
        active_blocker_id=None, active_blocker_description=None,
        goal_metadata_note="目标active；利率事前预测及6账户为实质进展，资金预测改善尚未转化为合格策略。",
        remaining_research_question="两年日更的资金利率预测有所改善，但次开盘之后的指数收益与完整账户优势尚未建立；高夏普目标和独立验证仍未完成。",
    )
    save(path, status)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update(
        current_round=result["study_id"], current_protocol=relative + "/protocol.json",
        latest_integrated_experiment=result["study_id"], latest_progress_receipt=relative + "/completed_round.json",
        last_research_result="资金日历预测误差改善，六个指数账户均未达标；主压力夏普-0.348，目标继续active。",
        latest_continuation_report=relative + "/本轮研究结论.md",
        latest_continuation_classification=classification,
        research_execution_state=completed["status"], goal_status="active", goal_achieved=False,
    )
    save(path, mandate)
    banner = (
        "> 2026-09-25 资金利率可预期成分与模型意外研究完成："
        f"1,624个源日利率MSE改善{mse_vs_persistence:.2%}，6个新完整账户仍未通过。"
        f"主压力夏普{primary['sharpe']:.3f}、年化{primary['annual_return']:.3%}、回撤{abs(primary['max_drawdown']):.2%}。"
        f"目标ACTIVE且未完成；见[本轮结论]({relative}/本轮研究结论.md)。\n\n"
    )
    path = ROOT / "RESEARCH_STATUS.md"
    old = path.read_text(encoding="utf-8-sig")
    if "资金利率可预期成分与模型意外研究完成" not in old:
        path.write_text(banner+old, encoding="utf-8")
    print("资金预测改善和六账户未通过的结果均已保存，主线目标仍active。", flush=True)


if __name__ == "__main__":
    run()
