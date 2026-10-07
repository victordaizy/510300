"""整理已完成的分红事件训练与完整账户，不执行新拟合或回测。"""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_dividend_payment_event_training_v1"


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    accounts = pd.read_csv(OUT / "results/账户指标.csv")
    trades = pd.read_csv(OUT / "results/机会成交账簿.csv")
    metrics = pd.read_csv(OUT / "results/预测指标.csv")
    comparisons = pd.read_csv(OUT / "results/增量比较.csv")
    selected = trades[(trades.model == "B_DIVIDEND") & (trades.scenario == "STRESS")]
    decision = {
        "status": "STOP_CURRENT_PAYMENT_EVENT_REPRESENTATION_NO_PARAMETER_RESCUE",
        "basis": "全部规则两档成本均未达到全账户净夏普1.5；压力下B模型3次交易期末199922.50元，全部事件对照8次交易期末190698.57元。",
        "positive_but_insufficient": "8次逐期评价中B相对价格A的MSE改善17.60%，未产生可用账户净优势。",
        "mechanism_observed": False, "actual_reinvestment_flow_observed": False,
        "scope": "只否定当前发放后次开盘进入、固定五日持有及模型筛选表达；不宣称所有分红相关机会都不存在。",
        "prior_yield_funding_and_calendar_results_preserved": True,
        "goal_achieved": False, "new_collection_enabled": False, "orders_authorized": False,
    }
    (OUT / "branch_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 分红现金发放后的五日机会：实际训练与账户", "", summary["conclusion"], "",
        "压力成本下，分红模型仅覆盖费用的规则做3次交易，期末199,922.50元，净夏普0.0014、最大回撤3.35%；无模型筛选、每次事件都参与的对照做8次，期末190,698.57元，净夏普-0.316。增加历史预测误差缓冲的主规则无交易。当前固定表示停止，不改持有期或门槛继续挑选。", "",
        "本轮假设是：基金现金分红发放后，部分持有人可能再投资以恢复指数敞口。实际再投资资金流没有被观察到，付款日期确定不等于未来上涨确定。14次既有分红中11次发生在1月、2次在12月、1次在6月，还存在月份与春节时点等混杂。此次只检验可交易账户，没有建立分红导致上涨的因果关系。", "",
        "旧cash_distribution_funding_v1使用全年分红率与融资成本的连续利差状态；本次使用每次现金发放的事前日期和固定五日窗口，问题不同。旧研究及旧日历研究结果保留，均属于已经多次观察过的510300历史，不能重新称为未见保留样本。", "",
        "事件规则在读取本轮标签前固定：现金发放日收盘后形成决策，下一交易日实际开盘进入，持有5个交易日后收盘退出。若付款日非交易日，价格特征只用该日前最近收盘。不假定分红现金一定在发放日开盘前到账。事件账户买入时已过除息和付款，不能领取这次已发放的分红；没有把机械除息价格下降当成利润。", "",
        "使用14次事件，前6次训练预热，其后8次逐期评价。A为含分红过去5日收益和20日波动；B再加入当次分红率及除息前至付款日的含分红回报；另列成熟训练均值。岭惩罚固定1，标准化只用成熟训练样本。完成16次逐期岭拟合、2次期末岭拟合和9份均值快照，共27份参数或均值；期末模型未回填历史。", "",
        "|模型|评价事件|RMSE|方向命中|相对成熟均值MSE改善|", "|---|---:|---:|---:|---:|"]
    for row in metrics.itertuples():
        lines.append(f"|{row.model}|{row.n_predictions}|{row.rmse:.2%}|{row.direction_hit:.2%}|{row.mse_improvement_vs_mean:.2%}|")
    lines += ["", "|B相对A|事件数|MSE改善|", "|---|---:|---:|"]
    for row in comparisons.itertuples():
        lines.append(f"|{row.segment}|{row.n}|{row.mse_improvement:.2%}|")
    lines += ["", f"完整账户从{summary['account_start']}至{summary['account_end']}，共{summary['complete_calendar_sessions']}个交易日，包含全部空仓日。B费用覆盖规则只有15天有敞口，约99.19%的交易日空仓；次数少不代表风险收益质量足够高。全部事件对照只用相同的8次可预测事件，未将早期6次训练样本交易收益混入比较。", "",
        "账户本金20万元，执行资产仅510300和现金。佣金单边万2最低5元，基础/压力滑点单边万5/千1；100份整手、不融资、T+1，目标年化波动10%、仓位上限100%。242日年化，现金和无风险收益设为0。收盘回撤8%后次开盘清仓并停止，跳空时不保证回撤不超过10%。除息应收与付款可用现金分开核算；最小报价单位、涨跌停队列和实际冲击未模拟。", "",
        "误差缓冲规则要求预测收益超过压力往返费用加过去至少6个成熟逐期误差的RMSE。费用覆盖对照只要求覆盖压力费用。两档成本使用同一压力费用门槛，均在训练前固定。零交易账户夏普未定义，不算达标。", "",
        "|模型|规则|成本|净夏普|最大回撤|交易次数|期末资金|", "|---|---|---|---:|---:|---:|---:|"]
    for row in accounts.itertuples():
        sharpe = "未定义" if pd.isna(row.net_sharpe) else f"{row.net_sharpe:.4f}"
        lines.append(f"|{row.model}|{row.policy}|{row.scenario}|{sharpe}|{row.max_drawdown:.2%}|{row.opportunities}|{row.ending_equity_cny:,.2f}|")
    lines += ["", "B模型压力成本下的全部3次交易：", "", "|入场|退出|净损益|享有本次历史分红|", "|---|---|---:|---:|"]
    for row in selected.itertuples():
        lines.append(f"|{row.entry_date}|{row.exit_date}|{row.net_pnl_cny:,.2f}元|{row.dividends_cny:,.2f}元|")
    lines += ["", "累计亏损77.50元而日频夏普略高于零不是对账错误：夏普使用逐日算术收益均值，累计财富使用复利；这里差异极小，经济上均接近无优势。全部三笔损益与期末资金一致。", "",
        "既有完整覆盖回执截至2026-09-04，因此账户也截止此日。全部16份已存来源快照随包保留；公告日期由原来源URL目录及文件日期较晚者作保守日期代理，均早于决策日，不是历史首次收件的独立证明。没有新采集、当日信号、实盘操作或外部审阅。", ""]
    (OUT / "训练结果.md").write_text("\n".join(lines), encoding="utf-8")
    documents = {
        "00_阅读导航.md": """# 阅读导航

先读训练结果.md、summary.json及branch_decision.json。本轮完成18次岭拟合、9份均值快照及18组连续账户，没有达标组合。

- protocol.json和freeze.json：本轮标签、模型和账户前固定的规则及输入身份。
- inputs：完整行情、14次分红、覆盖回执、16份原始来源快照、旧分红利差和日历研究结果、用户要求。
- code：完整本轮训练、成熟事件训练辅助函数及现金/分红到账账户引擎。
- models：24份逐期参数或均值、3份期末参数或均值。
- results：14次事件特征和标签、24条逐期预测、预测指标、18组完整1849日账户、全部成交与事件判断。
- FILE_INDEX.csv：其他包内成员的字节数和SHA-256。

使用requirements.txt列出的环境，在包根目录运行 `python -X utf8 code/dividend_payment_event_training_v1.py verify --root .`。保存结果复核包括事件时钟、含分红特征与标签、训练成熟时间、模型方程和预测、完整现金流与空仓日夏普，不重新拟合或新建账户。

交付回执在研究目录包外，记录ZIP身份及新解压复算结果。ZIP可交GPT审阅；本次没有上传或获得外部审阅。
""",
        "用户需求与本轮范围.md": """# 用户需求与本轮范围

只操作510300和现金，本金20万元，全账户扣成本净夏普至少1.5、最大回撤10%；接受一年只有四五次机会，次数不是配额，其余时间空仓。

用户要求加速，直接训练，放宽不必要的研究准入限制，并明确不用采集。本轮仅使用已有14次分红和来源，固定付款后次开盘、五日持有，不联网，不搜索参数。

前一轮压力危险率完成87次拟合并形成无有效增量结论，属于实际进展。本轮继续独立事件表示，保留历史已研究边界和旧失败；总目标没有降低，也未宣布完成。
""",
        "GPT审阅提示词.md": """# 可复制审阅提示词

请评议这个510300分红现金发放事件训练包。目标为20万元、只做510300和现金、完整账户净夏普1.5和最大回撤10%，接受稀疏交易；用户明确不采集、直接训练。

请先判断是否得到可交易优势，并引用具体文件和数值。重点审查付款日期与公告日期代理、发放后次开盘执行、不能享有此前分红、价格特征是否正确消除机械除息、前6次训练与后8次评价是否分开、参数与误差校准是否只用成熟历史、完整1849日空仓收益和费用。请解释预测MSE改善17.60%为什么没有变成账户净优势。

区分资金再投资假设与真实资金流证据，检查1月事件集中和历史多轮搜索的影响。比较无筛选事件对照，检查3笔交易损益与期末资金。给出少量有优先级的下一步和停止条件，不建议事后挑持有期、删亏损年份或反转信号来挽救结果。不要将包结构检查等同于策略有效或实盘授权。
""",
        "EXCLUSIONS.md": """# 包范围与排除

包含所有本轮直接计算输入、既有分红原文来源快照、完整代码、参数、预测、日历账户和成交账簿。旧利差与日历研究作为历史背景保留结果及相关配置，不重新打包其完整历史仓库，也未运行这些旧研究。

不包含虚拟环境、无关研究、编译缓存、verification及新解压副本。交付回执留在包外以免自引用。公告日期来源为既有URL和已核对表，未重新认证历史第一版公开时刻；没有新增资料采集、真实资金再投资路径、现实成交或外部专家审阅。
""",
    }
    for name, text in documents.items():
        (OUT / name).write_text(text, encoding="utf-8")
    print("分红事件训练报告、全部交易解释及停止条件已保存。")


if __name__ == "__main__":
    main()
