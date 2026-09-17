"""交付概率幅度退出的完整失败结果和全部两类参数。"""
import json
import shutil
import pandas as pd
from research.probability_payoff_exit_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    counts = []
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in ["BASE", "STRESS"]:
            folder = OUT / period / cost
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            available = decisions[decisions.learning_status.eq("PREDICTION_AVAILABLE")]
            cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            counts.append({"period": period, "cost": cost, "two_day_cycles": int(cycles.holding_intervals.eq(2).sum()),
                "prediction_available": len(available), "negative_predictions": int(available.continuation_prediction.lt(0).sum()),
                "majority_advantage_but_negative_payoff": int((available.probability_advantage.gt(.5)&available.continuation_prediction.lt(0)).sum()),
                "probability_below01_or_above99": int((available.probability_advantage.lt(.01)|available.probability_advantage.gt(.99)).sum()),
                **{key: float(ledger[key].sum()) for key in ["price_pnl", "dividend_recognized", "commission", "slippage_cost"]}})
    write_json(OUT / "saved_prediction_path_summary.json", {"recorded_at": now(), "saved_only": True, "new_models": 0, "new_accounts": 0, "records": counts}, exclusive=True)
    decision = ("第158轮概率与盈亏幅度分离的退出方法完成。主基础／压力净夏普0.116／0.019，较早0.716／0.692；"
        "年化收益主0.50%／−0.05%、较早7.84%／7.54%。四项夏普均低于1.2、128及143，关闭这套固定方法，完整目标未实现。")
    detail = ("九项必要测试14.35秒；25组实际输入各拟合一次，后来89个月复用，保留141个月度时钟和114个可用月份。"
        "25组均同时具备两类样本并成功拟合，训练与四条真实资金账户核心计算5.574415秒，不含开发、测试、核对和交付。"
        "已独立复算原权重、两类均值方差、稳定项、先验及平均净增量，并用独立高斯密度函数复算532次实际预测。"
        "70个实际周期、940个持仓状态、5646个收盘判断和140次真实开盘成交，以及完整净值分红费用均已核对。"
        "\n\n每个模型有七项因素参加概率计算，恒定进入类别保留完整参数并标为不参与概率乘积。"
        "全部25组模型的八因素均值尺度、两类因素均值和原方差及稳定后方差、类别先验和类内平均净增量都在本文后部。"
        "这是非线性概率模型，没有用八个虚构线性系数替代其实际参数。"
        "\n\n主各26个实际周期、200个持仓收盘、52次成交；其中23周期触发学习退出，16周期仅持有两个交易日。"
        "较早各9周期、270个持仓收盘、18次成交，其中66个状态可预测、204个入场时没有成熟模型。"
        "较早各3周期触发学习退出，3周期仅持有两个交易日。四账户均完整清仓，没有受阻未成交请求。"
        "主最大回撤15.46%／17.15%，较早13.79%／13.92%。"
        "\n\n主基础价格利润4869.80元、分红10513.60元、佣金滑点8709.80元，净利润6673.60元；"
        "压力价格利润5038.60元、分红10209.40元、费用15940.39元，净亏692.39元。"
        "压力账户的算术平均日收益仍略为正，但复利累计收益略负，所以正的极低夏普和小额累计亏损可以同时出现。"
        "较早净利润92546.96／88450.54元。相对128，主终值少105879.68／104659.68元，较早少12520.21／19806.11元。"
        "主基础价格加分红已经少105806.40元，费用仅多73.28元；压力价格加分红少104490.90元、费用多168.78元。"
        "相对原策略的退步主要来自持仓路径，不能仅归因于费用。"
        "\n\n模型经常很快退出，也有应退出却继续持有的情况。例如基础账户2020年2月5日进入，2月7日即退出，原128到2月24日才退出；"
        "2021年2月3日进入，本轮2月26日退出并亏4307.88元，原模型2月19日退出时盈利11215.21元。"
        "2025年4月10日进入，则持有到7月9日期限退出，净赚16714.86元；这说明不能把所有长持有或所有提前退出统一判为坏事。"
        "本轮相对128共有64条进出日期变化记录，含新增或缺少的进入，不能视为64个互相独立的交易。单笔金额还受之前复利资金规模影响。"
        "\n\n在实际可用预测中，主基础有1个“占优概率大于一半但合成净增量为负”的状态，其余三个场景没有；这项结构差异确实发生，但没有带来足够的账户收益。"
        "主各仅2个概率低于1%或高于99%的预测，不能把本轮失败简单归因于所有概率都极端化。"
        "概率是假设模型的输出，尚无校准或真实胜率证明。"
        "\n\n主年化收益低于买入持有3.13／3.66个百分点，较早高3.95／3.67个百分点。"
        "四项夏普均低于143；相对143主年化收益较低、较早较高。相对157较早夏普小幅上升，但四项年化收益都下降。"
        "下一项只检验八个单因素保序曲线的等权预测，方向由当时成熟训练协方差决定，不假设正态分布，不增加数据或参数网格。")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        PRIMARY: "CLOSED_FOUR_SHARPE_BELOW12_128_143_MAIN_LOW_RETURN_AND_FREQUENT_TWO_DAY_EXIT"}, exclusive=True)
    next_path = ROOT / "docs/510300_MARGINAL_MONOTONE_EXIT_NEXT_20260910.md"
    document = ROOT / "deliverables/510300概率与盈亏幅度退出_第158轮_20260910/概率与盈亏幅度退出_结果及全部中文规则.md"
    delivery = deliver_round(ROOT, OUT, CONFIG, document, "原八因素概率与盈亏幅度分离及固定版本退出",
        "CLOSED_PROBABILITY_PAYOFF_EXIT_FULL_GOAL_NOT_MET", decision, detail, next_path,
        next_path.read_text(encoding="utf-8").splitlines(), "MARGINAL_MONOTONE_EXIT_FINITE_CANDIDATE_PREPARED",
        "原八因素各自周期加权保序曲线，成熟协方差定方向，八条预测固定等权")
    parameters = OUT / "全部已拟合模型参数.md"
    shutil.copy2(parameters, document.parent / parameters.name)
    with document.open("a", encoding="utf-8") as stream:
        stream.write("\n"+parameters.read_text(encoding="utf-8"))
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=159, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=25, planned_factor_curve_records=200, planned_new_reference_accounts=0, external_data_required=False)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=158,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO158")
    write_json(index_path, index)
    require(document.read_text(encoding="utf-8").count("## 首次拟合：") == 25, "中文报告未包含25组全部参数")
    require(len(pd.read_csv(document.parent / "saved_all_factor_class_parameters.csv")) == 400, "两类八因素参数行数不同")
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
