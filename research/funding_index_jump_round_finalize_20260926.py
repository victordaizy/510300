"""保存指数跳跃增量结果与主线进度，不改变冻结策略或旧结果。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.funding_index_jump_daily_v1 as study
from research.selected_mix_reappraisal_v1 import read, save, now, digest

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
NAMES = {"PRICE": "价格基准", study.RV: "价格＋盘中波动大小", study.JUMP: "价格＋盘中波动大小＋跳跃代理",
    study.FUNDING_RV: "价格＋资金意外＋盘中波动大小", study.PRIMARY: "价格＋资金意外＋盘中波动大小＋跳跃代理（预定主方案）"}


def run():
    result = read(study.OUT / "result.json")
    assert result["status"] == "FROZEN_NO_QUALIFIED_INDEX_JUMP_STRATEGY"
    frozen = read(study.OUT / "freeze.json")
    for relative, sha in frozen["sources"].items():
        assert digest(ROOT / relative) == sha, relative
    primary = result["primary"]
    assert result["new_full_accounts"] == 8 and result["reused_full_accounts"] == 2
    assert not any(row["historical_point_targets_met"] for row in result["all_accounts"])
    ledger = pd.read_parquet(study.OUT / "accounts/STRESS" / study.PRIMARY / "ledger.parquet")
    decisions = pd.read_parquet(study.OUT / "accounts/STRESS" / study.PRIMARY / "decisions.parquet")
    gross = result["primary_cash_attribution"]["actual_shares_gross_pnl"]
    expense = result["primary_cash_attribution"]["commission_slippage_and_terminal_reserve"]
    np.testing.assert_allclose(gross - expense, primary["profit"], atol=1e-6, rtol=0)
    buying = ledger[ledger.filled_quantity.gt(0)]
    early = ledger[ledger.date.le("2023-12-31")]
    assert early.shares.eq(0).all()
    checks = result["distribution_checks"]
    jump_comparison = next(row for row in result["comparisons"] if row["cost"] == "STRESS" and row["control"] == study.FUNDING_RV)
    funding_comparison = next(row for row in result["comparisons"] if row["cost"] == "STRESS" and row["control"] == study.JUMP)
    forecast = next(row for row in result["forecast_evaluation"] if row["subset"] == "ALL")
    accounts_text = []
    for name in ["PRICE", *study.NEW_MODELS]:
        base = next(row for row in result["all_accounts"] if row["policy"] == name and row["cost"] == "BASE")
        stress = next(row for row in result["all_accounts"] if row["policy"] == name and row["cost"] == "STRESS")
        reuse = "；复用旧账户" if name == "PRICE" else ""
        accounts_text.append(f"- {NAMES[name]}：压力夏普{stress['sharpe']:.6f}、年化{stress['annual_return']:.4%}、最大回撤{abs(stress['max_drawdown']):.4%}；BASE夏普{base['sharpe']:.6f}、年化{base['annual_return']:.4%}{reuse}。")
    closed = next(row["closed"] for row in result["completed_and_pending_cycles"] if row["model"] == study.PRIMARY and row["cost"] == "STRESS")
    report = f"""本轮已实现并完成“价格＋资金意外＋指数盘中波动结构”的固定增量检验，新增8个完整账户，另逐值核对并复用2个价格基准账户。没有产生达到当前目标的策略。预定主方案在压力成本下的夏普为{primary['sharpe']:.6f}，年化收益{primary['annual_return']:.4%}，最大回撤{abs(primary['max_drawdown']):.4%}，20万元最终为{primary['end_equity']:,.2f}元。总目标保持active，未完成。

这次补的证据是什么。

此前已经检验成分广度、日内隔夜、午后响应和多类资金信息，本轮没有换名重跑这些信息。新增测量把沪深300指数的盘中波动大小与少数时段集中变化分开，检验相同日线价格弱势是否能因此得到更有用的后续五日分布。没有提前规定“跳跃一定反转”或“跳跃一定延续”，也没有根据这次结果改方向、阈值或窗口。

原作者方法在对应过程假设下，使用实现方差与归一化双幂变差之差研究跳跃二次变差。[方法来源：Barndorff-Nielsen与Shephard]({study.METHOD_URL})。本轮只把它作为有限采样代理，不将其解释为已经识别新闻、知情交易或被迫卖出，也不从方法论文推断510300的收益有效性。

使用已有000300.SH指数分钟资料，源日2021-08-12至2026-08-12共1,211日。上午09:35—11:25、下午13:05—14:55各取23个五分钟端点，各形成22个收益，共44个；两段分别计算，不跨午休或隔夜。全部日期的46个所需端点完整，所选close均为正。原质量记录中的非close异常保留。

RV为44个收益平方之和；每段BV为相邻收益绝对值乘积之和乘以π/2及22/21，再把两段相加。22/21是固定的项数修正，只在对应独立同方差正态设定下补偿缺少一个相邻项，不保证真实市场无偏。跳跃代理份额为max(RV−BV,0)/RV。另加入log(RV/源日前20日ETF含分红日收益平方均值)作为波动大小对照。所选窗口没有包含全天所有收益，不能用它代表全部跳跃风险。

1,211日中{result['zero_jump_proxy_dates']}日的代理值为0，中位数{result['jump_fraction_summary']['50%']:.4%}，95分位{result['jump_fraction_summary']['95%']:.4%}。这是测量分布，不是行情分类正确率；零值也不证明当天没有消息或没有跳跃。

时间与训练如何实现。

每个源日按15:30后计划可用，最早下一A股交易日09:00判断、09:30开盘执行。分钟标签起止语义和历史首次送达仍未认证，因此全部结果属于历史重建。资金输入沿用原两年窗内嵌套重构的FDR007日历模型残差，不是市场一致预期差；本轮没有新增资金回归。

六个分布模型使用完全相同、至少252个已成熟原点的最近两日历年训练池；退出开盘严格早于当前判断。固定126近邻、训练内标准化和截断5，未搜索参数。共920个实际预测日，2022-09-16至2026-08-13；新增条件分布3,680个，另重建1,840个HISTORY/PRICE对照分布验证一致性。这些都来自同一批历史，不是新增独立样本。

账户从2020-01-02连续计算至2026-09-24，共1,633个交易日，保留456个源资料不可用日和257个训练不足日。现有现金、T+1、100份、原BASE/STRESS费用、50%仓位上限、五日ES95预算2.5%、10%跳空预算与回撤余量限制不变。使用原每日滚动库存引擎；上一轮五日期限账户的点值没有用于挑选本轮退出方式。

所有账户结果。

{chr(10).join(accounts_text)}

八个新账户和两个复用对照都没有同时满足压力夏普1.2、年化10%、回撤10%的目标，预定主方案全部滚动两年窗口的三项联合通过次数为0。没有从对照中挑一个较好点值改称主策略。

预定主压力账户完成{closed}个持仓周期，{len(buying)}个买入或增仓日，{int(ledger.shares.gt(0).sum())}个收盘有仓位日；首次买入是{buying.date.min().date()}，平均账户暴露{primary['mean_exposure']:.4%}。2020—2023全部空仓，这段收益差0不属于跨时期验证。按实际份额计算，毛损益{gross:,.2f}元，佣金及滑点等费用{expense:,.2f}元，净损益{primary['profit']:,.2f}元；不能把失败全部归因于成本。

新增信息的增量。

在已有价格、资金和波动大小上加入跳跃代理，完整账户年化算术收益差为{jump_comparison['annual_arithmetic_difference']:.4%}，20日循环区块2000次的95%区间为[{jump_comparison['lower_95']:.4%}, {jump_comparison['upper_95']:.4%}]。控制价格、波动大小和跳跃代理后加入资金信息，差为{funding_comparison['annual_arithmetic_difference']:.4%}，区间[{funding_comparison['lower_95']:.4%}, {funding_comparison['upper_95']:.4%}]。两者均未证明正增量。百分比这里表示年化收益差，不是策略自身CAGR。

全部920个成熟五日预测上，主方案相对不含跳跃的资金/RV模型，均方误差改善为{forecast['primary_mse_improvement'][study.FUNDING_RV]:.4%}；负数表示误差变大。相对共同训练池直接历史均值，改善为{forecast['primary_mse_improvement']['HISTORY']:.4%}。更细的波动结构并没有在本轮得到更准确的收益预测。区块区间未作历次研究选择调整，不能宣称已排除过拟合。

实施检查已完成：已知匀速路径、单跳路径、价格单位缩放、午间边界、零方差及缺端点NO_VIEW、未来分钟隔离、分母排除源日与未来；保存的{checks['saved_distributions_recomputed']}个分布全部复算；共同价格与历史对照的训练集、选中集、预测和状态逐值一致。扰动未来与过期标签不改变指定历史日判断。所有账户通过现金、股息、T+1与买入尾部预算检查。

规律监控继续单独保存：固定五日间隔成熟预测的60期q05跌穿率监控没有触发15%警报。这只说明该特定监控没有报警，不能证明优势存在或规律稳定；监控未参与事后改账户。独立前向观察新增0，当前市场视图NO_VIEW。

本轮的结果是排除这项固定代理在当前模型和成本下已足以改善账户的说法，并非证明所有盘中结构信息都无效。下一阶段仍需找到指数股票组合本身可执行的风险补偿；不能靠继续增加相似技术变量或放宽目标代替证据。旧失败、冻结定义和全部账户已保留。本轮没有制作审核包或用户表格。
"""
    report_path = study.OUT / "本轮研究结论.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    completed = {"at": now(), "study_id": study.STUDY, "status": result["status"],
        "result_sha256": digest(study.OUT / "result.json"), "report_sha256": digest(report_path),
        "new_full_accounts": 8, "reused_full_accounts": 2, "new_conditional_distributions": 3680,
        "recomputed_control_distributions": 1840, "new_funding_regression_fits": 0,
        "primary_stress_sharpe": primary["sharpe"], "primary_stress_cagr": primary["annual_return"],
        "primary_max_drawdown": primary["max_drawdown"], "primary_equity": primary["end_equity"],
        "goal_status": "active", "goal_achieved": False, "independent_forward_observations": 0,
        "orders_authorized": False, "review_package_created": False}
    save(study.OUT / "completed_round.json", completed, True)
    state_path = MAIN / "current_status.json"
    state = read(state_path)
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification="PROGRESS_FIXED_INDEX_JUMP_INCREMENT_8_NEW_ACCOUNTS",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=study.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=result["status"],
        latest_index_jump_increment_round=completed,
        remaining_research_question="价格弱势、日历资金残差与指数双幂差代理尚不能证明足够的收益补偿；仍需股票组合本身的可执行优势及独立验证。",
        goal_metadata_note="本轮完成预先固定的新信息增量与8个新账户，属于实质研究进展；目标未完成。")
    if study.STUDY not in state["completed_followup_studies"]:
        state["completed_followup_studies"].append(study.STUDY)
    save(state_path, state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=study.STUDY, latest_integrated_experiment=study.STUDY,
        current_protocol=(study.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(study.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=report_path.relative_to(ROOT).as_posix(),
        latest_continuation_classification=state["latest_goal_turn_classification"], research_execution_state=result["status"],
        last_research_result=f"指数跳跃增量及8新账户完成；主压力夏普{primary['sharpe']:.6f}、年化{primary['annual_return']:.4%}，无合格策略，总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    text = status_path.read_text(encoding="utf-8")
    marker = "<!-- INDEX_JUMP_INCREMENT_20260926 -->"
    if marker not in text:
        status_path.write_text(f"{marker}\n\n> 2026-09-26 指数跳跃代理增量完成，8个新账户与2个复用价格对照；主压力夏普{primary['sharpe']:.6f}、年化{primary['annual_return']:.4%}，目标ACTIVE且未完成；见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + text, encoding="utf-8")
    print("指数波动结构增量的结果、中文结论和主线进度已保存，目标仍为active。", flush=True)


if __name__ == "__main__":
    run()
