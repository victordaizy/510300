"""保存资金与下午价格信息增量的结果，不重新拟合或改变账户。"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now

DATA = ROOT / "reports/research/510300_funding_afternoon_response_daily_v1"
OUT = ROOT / "reports/research/510300_funding_afternoon_response_saved_completion_v1"
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
PRIMARY = "PRICE_FUNDING_AND_AFTERNOON"
FUNDING = "PRICE_AND_FUNDING"
RESPONSE = "PRICE_AND_AFTERNOON"
NAMES = {"PRICE": "基础价格", FUNDING: "基础价格加资金意外", RESPONSE: "基础价格加下午价格", PRIMARY: "基础价格加资金意外和下午价格"}


def run():
    result = read(OUT / "result.json")
    primary = result["primary"]
    measures = {row["policy"]: row for row in result["all_accounts"] if row["cost"] == "STRESS"}
    pairs = {row["control"]: row for row in result["comparisons"] if row["cost"] == "STRESS"}
    f_all, f_down = result["forecast_evaluation"]
    predictions = pd.read_parquet(DATA / "results/predictions.parquet")
    primary_predictions = predictions[predictions.model.eq(PRIMARY)]
    primary_decisions = pd.read_parquet(DATA / "accounts/STRESS" / PRIMARY / "decisions.parquet")
    annual_activity = []
    for year, decisions in primary_decisions.groupby(primary_decisions.date.dt.year):
        forecasts = primary_predictions[primary_predictions.date.dt.year.eq(year)]
        annual_activity.append({"year": int(year), "account_days": len(decisions), "forecast_days": len(forecasts),
                                "positive_mean_forecasts": int(forecasts.mu5.gt(0).sum()),
                                "buy_days": int(decisions.filled_quantity.gt(0).sum()),
                                "sell_days": int(decisions.filled_quantity.lt(0).sum()),
                                "mean_forecast": float(forecasts.mu5.mean()) if len(forecasts) else None})
    cost_attribution = []
    for name in NAMES:
        ledger = pd.read_parquet(DATA / "accounts/STRESS" / name / "ledger.parquet")
        row = {"model": name, "same_share_path_gross_pnl": float((ledger.price_pnl+ledger.dividend_recognized).sum()),
               "execution_cost": float((ledger.commission+ledger.slippage_cost).sum()),
               "terminal_reserve": float(ledger.terminal_exit_reserve.iloc[-1]),
               "net_profit": float(ledger.equity.iloc[-1]-200000),
               "first_buy": ledger.loc[ledger.filled_quantity.gt(0), "date"].min(),
               "last_fill": ledger.loc[ledger.filled_quantity.ne(0), "date"].max()}
        assert abs(row["same_share_path_gross_pnl"]-row["execution_cost"]-row["terminal_reserve"]-row["net_profit"]) < 1e-6
        cost_attribution.append(row)
    save(OUT / "results/saved_account_interpretation.json", {
        "fixed_executed_share_path_attribution_not_new_zero_cost_strategies": cost_attribution,
        "primary_annual_prediction_and_execution_activity_descriptive": annual_activity,
        "new_fits": 0, "new_accounts": 0,
    })
    account_lines = []
    for name, description in NAMES.items():
        row = measures[name]
        cycle = next(v for v in result["completed_and_pending_cycles"] if v["model"] == name and v["cost"] == "STRESS")
        s = "未定义" if row["sharpe"] is None else f"{row['sharpe']:.6f}"
        account_lines.append(f"- {description}：夏普{s}，复合年化{row['annual_return']:.4%}，最大回撤{abs(row['max_drawdown']):.4%}，期末{row['end_equity']:.2f}元，闭合周期{cycle['closed']}个，未闭合周期{int(cycle['open'])}个。")
    account_text = "\n".join(account_lines)
    pair_lines = []
    for name in [RESPONSE, FUNDING, "PRICE"]:
        row = pairs[name]
        pair_lines.append(f"- 联合方案相对{NAMES[name]}：算术年化增量{row['annual_arithmetic_difference']*100:.4f}个百分点，95%区块区间[{row['lower_95']*100:.4f}, {row['upper_95']*100:.4f}]个百分点；两个固定时期点值分别为{row['fixed_periods'][0]['annual_arithmetic_difference']*100:.4f}、{row['fixed_periods'][1]['annual_arithmetic_difference']*100:.4f}个百分点。")
    pair_text = "\n".join(pair_lines)
    development = result["status"] == "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION"
    decision = "保留开发候选，但资料认证和独立验证仍未完成" if development else "固定方案不予晋升，失败结果保留"
    status_code = "FUNDING_RESPONSE_EIGHT_ACCOUNTS_COMPLETED_REQUIRES_INDEPENDENT_VALIDATION" if development else "FUNDING_RESPONSE_EIGHT_ACCOUNTS_COMPLETED_TARGET_UNMET"
    report = f"""本轮完成上午资金意外与下午价格反应的联合信息检验，新增8个完整账户。主压力账户夏普{primary['sharpe']:.6f}、复合年化{primary['annual_return']:.4%}、最大回撤{abs(primary['max_drawdown']):.4%}，期末{primary['end_equity']:.2f}元。{decision}。510300高夏普总目标仍未完成。

上轮资金利率日历模型的均方误差改善，并未转化为股票账户优势。本轮沿这条传导关系只增加一个具体信息：上午资金变化已经出现后，下午股价是继续下跌，还是企稳上涨。这个信息在下一开盘买入前已经观察到，其间涨幅不能算进策略收益。

新增信息与来源边界。

复用上轮在当前两年窗内嵌套构造的FDR007资金模型意外，不重估资金模型。价格信息固定为同源日13:05和14:55两个分钟标签close的对数比值，除以源日前20个交易日含分红日收益的均方根波动；不扫描其他时间窗口。上午定盘按12:00计划可用，下午信息按15:30以后可用，均合并至下一A股09:00决策，09:30模拟成交。等待观察的收益没有提前计入账户。

分钟文件来自已保存的第三方代理，2021-08-12至2026-08-12共1211个日期，两端标签都存在且价格为正，文件与既有覆盖检查的哈希一致。原供应商未明确标签属于bar开始还是结束，不能说这是秒级精确的公告反应；两个标签均位于午后，作为次日信息使用。历史首版与实际首次送达仍未认证。下午价格也受到其他消息影响，本轮不声称识别资金意外对股价的因果作用。

如何区分新增宏观信息与普通价格信息。

保留四组完全相同资料覆盖的方案：基础价格；基础价格加资金意外；基础价格加下午价格；三者共同使用。主比较是联合方案相对已经包含下午价格的方案，防止把普通价格确认带来的变化归功于资金信息。另两个直接对照同时保留。HISTORY是同池经验分布预测参考，不生成本轮第五个交易方案。

所有条件分布只用当前两个日历年内的成熟五日收益标签，退出开盘严格早于当前09:00；训练池完全相同、至少252行，固定126近邻和训练内标准化截断5。没有更改窗口、方向、阈值、风险预算或交易效用。只在已知五日价格弱势时允许新增库存，退出和费用沿用上一轮。

实际覆盖必须与整个账户区别开。

账户始终覆盖2020-01-02至2026-09-24，共1633个交易日、20万元连续本金。真正具备预测的日期为{str(result['first_prediction_date'])[:10]}至{str(result['last_prediction_date'])[:10]}，共{result['unique_prediction_days']}日；另有{result['schedule_statuses'].get('NO_VIEW_SOURCE',0)}日资料不足、{result['schedule_statuses'].get('NO_VIEW_TRAINING',0)}日训练不足。这些日期均留在账户中，未删去现金期来提高年化或夏普。已有库存缺信息达到原固定期限后退出，末端数据不足没有变成当前市场观点。

每日共同成熟训练池实际为{result['training_rows_min']}—{result['training_rows_max']}行。保存{result['daily_distribution_updates']}份新分布，来自{result['unique_prediction_days']}个日期、每日期5个模型；{result['mature_prediction_days']}个日期的五日结果已成熟。复用的{result['reused_nested_funding_predictions_each_model']}组历史嵌套资金预测不是新独立市场样本，本轮新增资金回归次数为0。

更具体地看，四组账户在2023年底前均没有发生买卖。主方案2022年已有70个预测日、2023年有237个预测日，其中正均值预测仅3日和17日，仍没有成交；第一笔买入直到2024-11-18。2020—2023的账户增量为零，不能据此证明已经跨时期重复获利。这里只描述事前模型判断与交易活动，没有据此放宽入场或费用门槛，也没有删除早期现金日。

完整压力账户结果。

{account_text}

基础和压力成本共8个账户全部保存，现金收益和无风险基准按零、242日年化。本轮训练资格与上一轮不同，不能跨轮混合挑选对照。现行目标仍为压力夏普至少1.2、复合年化至少10%、最大回撤不超过10%。主方案所有滚动两年三目标联合通过次数为{result['primary_rolling_two_year_joint_passes']}。

新增信息是否产生账户增量。

{pair_text}

区间使用预先固定的20日循环区块和2000次抽样，没有对全项目的历史选择作校正。另保存完整账户PRIMARY-RESPONSE-FUNDING+PRICE的四组差分，只作联合效果描述；各账户的仓位、费用及权益路径不同，不能把它当成结构性因果交互。

全部成熟预测中，联合方案相对下午价格对照的MSE点值改善为{f_all['primary_mse_improvement_vs_afternoon_only']:.2%}；在事前五日下跌样本中为{f_down['primary_mse_improvement_vs_afternoon_only']:.2%}。负值表示误差变大。均值MSE、尾部分位损失和最终账户分开保存，不用其中某个较好数字替代完整目标。

固定主方案实际成交份额做会计分解，毛损益为{cost_attribution[-1]['same_share_path_gross_pnl']:.2f}元，交易费用为{cost_attribution[-1]['execution_cost']:.2f}元，末端退出储备{cost_attribution[-1]['terminal_reserve']:.2f}元。这不是重新运行的零成本策略，不据此反调手续费或选择更有利的期间。

全部{result['saved_distributions_recomputed']}份保存分布已从原选择样本复算，两个时点的未来标签隔离通过；8个账户的现金、整手、T+1、分红及调仓费用后的尾部预算均通过一致性检查。50%目标上限、5日ES95预算2.5%、10%跳空预算5%且不超过距90%权益峰值余量一半，以及10%回撤退出规则均未改动。

原运行在八账户保存之后，最后核验将已移入索引的idx误当作普通列而报错。原失败文件与冻结程序均保留；独立完成程序只用行索引完成核验和已保存结果统计，未重跑账户、未修改策略参数或任何账户数值。本报告引用独立完成目录的result.json，原目录仍保留RUN_FAILURE.json，不把原失败覆盖成成功。

本轮{decision}。新增独立前向观察仍为0，总目标保持active。原资金模型和原失败账户均未修改，也没有恢复旧收盘30分钟成交压力策略。本轮没有制作审核包或用户表格。
"""
    (OUT / "本轮研究结论.md").write_text(report, encoding="utf-8")
    completed = {"at": now(), "study_id": result["study_id"], "completion_study_id": result["completion_study_id"], "status": status_code,
                 "new_unique_full_accounts": 8, "new_market_downloads": 0, "new_funding_regression_fits": 0,
                 "minute_response_days": 1211, "daily_distribution_updates": result["daily_distribution_updates"],
                 "unique_prediction_days": result["unique_prediction_days"],
                 "first_prediction_date": result["first_prediction_date"], "last_prediction_date": result["last_prediction_date"],
                 "primary": primary, "stress_controls": {k: v for k, v in measures.items() if k != PRIMARY},
                 "development_candidate": development, "goal_status": "active", "goal_achieved": False,
                 "independent_forward_observations": 0, "review_package_created": False, "orders_authorized": False}
    save(OUT / "completed_round.json", completed)
    relative = OUT.relative_to(ROOT).as_posix()
    classification = "PROGRESS_FUNDING_PRICE_RESPONSE_AND_EIGHT_FULL_ACCOUNTS"
    path = MAIN / "current_status.json"
    status = read(path)
    for study in [result["study_id"], result["completion_study_id"]]:
        if study not in status["completed_followup_studies"]:
            status["completed_followup_studies"].append(study)
    status.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification, same_condition_consecutive_no_progress_goal_turns=0,
        latest_funding_afternoon_response_round=completed,
        latest_user_requested_study=relative, latest_user_requested_study_status=status_code,
        research_execution_state=status_code, latest_continuation_report=relative+"/本轮研究结论.md",
        active_blocker_id=None, active_blocker_description=None,
        goal_metadata_note="资金意外与下午响应的8个账户为实际进展；高夏普目标及独立验证未完成。",
        remaining_research_question="资金预测与已知价格反应的传导检验已经完成；仍须在现行两年日更和尾部预算下建立合格收益优势及独立验证。")
    save(path, status)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(path)
    mandate.update(current_round=result["study_id"], current_protocol=DATA.relative_to(ROOT).as_posix()+"/protocol.json",
        latest_integrated_experiment=result["study_id"], latest_progress_receipt=relative+"/completed_round.json",
        last_research_result=f"资金与下午响应8账户完成；主压力夏普{primary['sharpe']:.3f}，{decision}，总目标active。",
        latest_continuation_report=relative+"/本轮研究结论.md", latest_continuation_classification=classification,
        research_execution_state=status_code, goal_status="active", goal_achieved=False)
    save(path, mandate)
    banner = (
        "> 2026-09-26 上午资金与下午价格反应研究完成：8个新完整账户，"
        f"主压力夏普{primary['sharpe']:.3f}、年化{primary['annual_return']:.3%}、回撤{abs(primary['max_drawdown']):.2%}。"
        f"{decision}，目标ACTIVE且未完成；见[本轮结论]({relative}/本轮研究结论.md)。\n\n")
    path = ROOT / "RESEARCH_STATUS.md"
    old = path.read_text(encoding="utf-8-sig")
    if "上午资金与下午价格反应研究完成" not in old:
        path.write_text(banner+old, encoding="utf-8")
    print("资金与下午价格反应的八账户结论已保存，主线目标仍active。", flush=True)


if __name__ == "__main__":
    run()
