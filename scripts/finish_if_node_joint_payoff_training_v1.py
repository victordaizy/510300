"""整理联合训练结果及图表；不新增拟合、账户或参数搜索。"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_if_node_joint_payoff_training_v1"


def write(name, content):
    (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    annual_summary = json.loads((OUT / "annual_frequency_summary.json").read_text(encoding="utf-8"))
    annual = pd.read_csv(OUT / "results/逐年完整持仓周期.csv")
    metrics = pd.read_csv(OUT / "results/账户联合指标.csv")
    predictions = pd.read_csv(OUT / "results/节点逐期预测.csv")
    trades = pd.read_csv(OUT / "results/机会成交账簿.csv")
    stress = metrics.loc[metrics.scenario == "STRESS"]
    primary = stress.loc[(stress.model == "B_IF_JOINT") & (stress.policy == "JOINT_PRIMARY")].iloc[0]
    control = stress.loc[stress.model == "EVENT_ONLY"].iloc[0]
    if len(summary["numeric_targets_pass_both_costs"]):
        raise ValueError("出现数值达标，需按实际结果重新组织本轮结论。")
    original_cycles = trades.loc[(trades.model == "EVENT_ONLY") & (trades.scenario == "STRESS"), ["event_month", "net_pnl_cny"]]
    event_detail = predictions.loc[predictions.model == "B_IF_JOINT"].merge(original_cycles, left_on="month", right_on="event_month", validate="one_to_one")
    event_detail = event_detail.rename(columns={"net_pnl_cny": "original_event_account_net_profit_cny"})
    event_detail.to_csv(OUT / "results/联合筛选与原节点对照.csv", index=False, encoding="utf-8-sig")
    omitted = event_detail.loc[~event_detail.joint_gate_pass.astype(bool)]
    retained = event_detail.loc[event_detail.joint_gate_pass.astype(bool)]
    removed_wins = int((omitted.original_event_account_net_profit_cny > 0).sum())
    removed_losses = int((omitted.original_event_account_net_profit_cny < 0).sum())
    probability_low, probability_high = retained.p_win.min(), retained.p_win.max()
    node_scores = pd.read_csv(OUT / "results/节点预测指标.csv")
    month_scores = pd.read_csv(OUT / "results/月末预测指标.csv")
    a = node_scores.loc[(node_scores.model == "A_PRICE_JOINT") & (node_scores.segment == "ALL")].iloc[0]
    b = node_scores.loc[(node_scores.model == "B_IF_JOINT") & (node_scores.segment == "ALL")].iloc[0]
    brier_improvement = 1 - b.brier_score / a.brier_score
    table = ["| 模型及固定规则 | 交易次数 | 胜率 | 实际金额盈亏比 | 全账户净夏普 | 最大回撤 | 净利润 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    labels = [("B_IF_JOINT", "JOINT_PRIMARY", "IF联合模型：胜面与幅度约束"),
              ("B_IF_JOINT", "EXPECTANCY_CONTROL", "IF联合模型：仅覆盖费用"),
              ("A_PRICE_JOINT", "JOINT_PRIMARY", "价格联合模型：胜面与幅度约束"),
              ("A_PRICE_JOINT", "EXPECTANCY_CONTROL", "价格联合模型：仅覆盖费用"),
              ("MATURE_EMPIRICAL", "JOINT_PRIMARY", "成熟经验：胜面与幅度约束"),
              ("MATURE_EMPIRICAL", "EXPECTANCY_CONTROL", "成熟经验：仅覆盖费用"),
              ("EVENT_ONLY", "EVENT_RULE_DIAGNOSTIC", "全部原节点")]
    for model, policy, name in labels:
        r = stress.loc[(stress.model == model) & (stress.policy == policy)].iloc[0]
        payoff = "未定义" if pd.isna(r.realized_cash_payoff_ratio) else f"{r.realized_cash_payoff_ratio:.2f}"
        table.append(f"| {name} | {r.opportunities} | {r.win_rate:.1%} | {payoff} | {r.net_sharpe:.3f} | {r.max_drawdown:.2%} | {r.net_profit_cny:,.2f}元 |")

    write("训练结果.md", f"""
# 少数IF节点的胜率与条件盈亏：实际训练完成，未达到夏普1.3

已完成222次逻辑回归、444次条件幅度回归及111份经验参考快照，并运行18组连续账户。主要IF联合规则在压力成本下6次交易、5次盈利，实际金额盈亏比{primary.realized_cash_payoff_ratio:.2f}、最大回撤{primary.max_drawdown:.2%}，但净夏普只有{primary.net_sharpe:.3f}，未超过全部原节点对照的{control.net_sharpe:.3f}，也未达到用户1.3目标。高历史胜率和高平均盈亏比尚未转化为足够的完整账户表现。本固定表示到此关闭，不根据结果继续调阈值。

训练完成后，用户新增“**一年至少五次交易**”。当前要求已更新为每个完整自然年至少5个完整持仓周期，按入场年归属，一买一卖计一次。IF原始节点池在完整年份每年也只有0—2次，本身无法满足新要求。追加逐年检查本轮及既有188条保存账户记录，同时满足年度频率、夏普1.3和回撤10%的记录为0。该检查不改变已经保存的模型和原研究协议。

## 固定研究对象与实际训练

交易标的是510300，IF期货数据只作辅助信号。沿用原卖压缓解状态机的全部10个节点，不改事件日期、十日持有、次开盘入场、20万元、波动目标仓位及8%账户收盘回撤后的次开盘退出规则。实际回撤阈值只是退出触发，不能保证最大回撤上限。IF信息按原规则滞后一交易日。

本轮仅改变预测表示：从单一收益回归，改为获利概率p、盈利条件下的平均幅度W、亏损条件下的平均幅度L。预计净收益代理为p×W−(1−p)×L。训练仍按原124个月末样本逐期扩展，至少24个成熟标签开始，每次只使用预测前已结束的标签；节点只有10个，月末样本不增加独立节点数。

三套模型分别是原4项价格特征、价格加原3项IF特征，以及成熟样本的经验频率和平均幅度。逻辑回归不使用类别加权；盈利和亏损幅度分别做对数岭回归，以训练残差还原条件均值。惩罚固定1，标准化只用当前成熟样本，标准化输入固定截断±3，不搜索参数。

本次保存330份逐期联合快照和3份最终快照，包含300条月末预测、30条节点预测。含最终参数共222次概率拟合、444次条件幅度拟合，111份经验参考更新。最终模型是事后历史研究快照，不是当年实际发出的预测，也不是当前交易信号。

## 同一节点和成本下的完整账户

所有账户都从2018-05-02至2026-08-18，含2016个交易日和全部空仓时间。STRESS单边佣金2bp、滑点10bp、最低佣金5元；BASE滑点5bp，其余一致。没有融资，没有交易其它资产。

{chr(10).join(table)}

主要IF联合规则期末权益{primary.ending_equity_cny:,.2f}元，累计净利润{primary.net_profit_cny:,.2f}元，年化收益{primary.annualized_return:.3%}；有持仓的交易日占{primary.exposure_day_fraction:.2%}。6次是整个八年多区间的总数，不是每年6次；按用户新要求，这套规则的频率也不达标。

联合主要规则要求预计净收益为正、p>0.5、W>L，同时接受原引擎按当时仓位估计的压力费用检查。0.5和幅度比1是本轮预先固定的方向性分界，不是用户指定的高胜率或高盈亏比验收数值。另一个固定对照只要求收益覆盖费用，用于观察联合约束本身的影响；没有比较一系列阈值后择优。

实际金额盈亏比是平均净盈利金额/平均净亏损金额绝对值。入场时预测的W/L、投入归一化的实际盈亏比、实际金额盈亏比在表中分别保存；都不是承诺的止损R倍数。

## 联合筛选究竟做了什么

联合规则保留6个节点、跳过4个。被跳过的节点在原全部节点账户中有{removed_wins}个盈利、{removed_losses}个亏损。它规避了一次亏损，也放弃了三次盈利；胜率和盈亏比上升，全部日历上的夏普没有改善。这是保存周期的归因，不是根据事后结果改写买卖规则。

更关键的是，6个入场节点当时估计的获利概率为{probability_low:.1%}—{probability_high:.1%}；事后的5胜1负不能改写成模型事前有83.3%的获利把握。盈亏比只有一个亏损样本作为分母，仍很不稳定。

加入IF信息后的节点Brier概率误差由价格模型的{a.brier_score:.6f}降到{b.brier_score:.6f}，相对下降{brier_improvement:.2%}；十日净收益代理RMSE由{a.net_proxy_rmse:.4%}降至{b.net_proxy_rmse:.4%}。这些都是10个已看过节点上的探索结果，不能据此认定概率已校准或建立独立优势。月末100次预测的概率误差和前后半段也完整报告，不把节点结果混入月末样本量。

## 成本代理和边界

训练标签是固定十日毛收益减24bp名义往返压力费用。它便于在各历史时点使用同一获利定义，但未包含最低佣金、未来收益变动造成的费用变化及账户提前退出状态；因此称为净收益代理。账户实际按原引擎逐笔计算滑点、最低佣金、分红权益与到账、整手和T+1，实际周期胜率与预测标签分开披露。

模型使用原事件特征，未重新搜索变盘点。所有历史都已经被前期研究查看过，本轮属于IF家族新的预测表示尝试；原失败与旧策略终止状态保持。此次样本没有独立前向事件，也没有今日行情结论。账户引擎仍沿用原连续价格滑点近似，未模拟涨跌停订单队列及真实成交冲击。

本轮没有达到净夏普1.3的模型账户。保留IF信息对预测评分的有限改善，同时关闭这次固定节点、固定十日退出和联合表示的组合；不通过事后抬高或降低门槛、替换窗口或延长持有来补救。
""")
    branch = {"status": "STOP_FIXED_IF_JOINT_REPRESENTATION_NO_ACCOUNT_INCREMENT",
              "basis": "联合主要规则6次5胜、实际金额盈亏比3.84，但压力净夏普0.361低于全部节点0.363且远低于1.3。",
              "trained_logistic_models": summary["logistic_fits_including_final"],
              "trained_conditional_magnitude_models": summary["conditional_magnitude_fits_including_final"],
              "positive_but_insufficient": "节点概率误差和净收益代理RMSE较价格基准改善，不能当作稳定账户优势。",
              "omitted_event_wins": removed_wins, "omitted_event_losses": removed_losses,
              "no_parameter_rescue": True, "previous_failures_preserved": True,
              "latest_minimum_complete_cycles_each_full_calendar_year": 5,
              "annual_frequency_pass": False,
              "node_pool_frequency_limit": "所有原节点在完整年份仅0至2次，无法支持每年至少5次。",
              "target_net_sharpe": 1.3, "new_collection_enabled": False, "goal_achieved": False, "orders_authorized": False}
    save("branch_decision.json", branch)
    write("00_README_FIRST.md", """
# 阅读导航

本轮已经实际训练，先读训练结果.md和summary.json，再看results/账户联合指标.csv、逐节点交易判断.csv和联合筛选与原节点对照.csv。

用户在训练完成后新增“每年至少5次交易”，当前以annual_frequency_requirement.json及最新频率要求与逐年评价.md为准。逐年完整持仓周期.csv和频率夏普回撤联合评价.csv包含188条保存账户的新增频率评价；首尾不完整年不年化凑数。

- protocol.json和freeze.json：本轮拟合前固定的模型、比较规则、成本及来源身份；明确历史早已被查看。
- inputs：原市场、分红、IF状态和上游已有输入；原关闭结果；父级月末/事件表与基准日线；最新用户要求。
- code：完整本轮训练代码和所用原账户、标签计算依赖。
- models：330份逐期快照、3份最终快照；每份包含训练月份、最晚标签时间和全部参数。
- results：124个月末样本、全部10节点、330条预测、概率分组、18账户及每个节点的进入/拒绝原因。
- figures：相同完整日历上的账户权益和回撤图。
- branch_decision.json：本固定表示停止，不进行结果驱动的参数补救。
- FILE_INDEX.csv：除自身外所有包内文件的大小与SHA-256。

用Python运行code/if_node_joint_payoff_training_v1.py的verify命令，--root指定解压目录，即可代入保存模型、检查成熟标签、最优化方程和完整账簿。verify新增拟合、账户、下载均为0。train入口拒绝覆盖已有summary.json。
""")
    write("用户要求.md", """
# 当前用户要求

只操作510300和现金，资金20万元，最大回撤10%，净夏普目标已调整为1.3。只做可识别节点，结合变盘依据、高胜率、高盈亏比和高夏普，拒绝过拟合。最新明确要求“一年至少五次交易”，按每个完整自然年不少于5个完整持仓周期评价，不能用多年平均代替；其余时间现金。

用户要求加速、直接训练、不采集，并在2026-09-23及2026-09-24明确“请继续”。本轮据此完成概率和条件盈亏的实际训练，使用已有本地数据，原事件和退出期限保持。0.5和幅度比1是本轮固定的比较分界，并非用户新增的高胜率硬门槛。
""")
    write("GPT审阅提示词.md", """
请审阅这一轮实际完成的510300联合训练。先给可用、不可用或证据不足的结论，再引用具体文件说明。

重点检查：

1. 标签成熟、训练标准化、逻辑回归无类别加权、条件幅度分组及指数残差均值校正是否正确？
2. 月末训练分布与10个IF节点分布有何差异？Brier改善是否被夸大为概率已经校准？
3. 24bp净收益代理与实际最低佣金、滑点、账户退出是否明确区分？完整空仓日是否计入夏普？
4. 联合筛选6次5胜、平均金额盈亏比3.84，但夏普0.361略低于全部节点0.363，应如何解释？被排除4个节点中的3胜1负是否完整保留？
5. 用户最新要求每个完整自然年至少5次交易。是否按完整持仓周期、入场年归属，保留零交易年并正确处理首尾不完整年份？旧14方案主历史的最少年度次数只有0至2，是否完整披露？
6. 已有历史被大量查看，本轮的新预测表示是否只是又一次同家族尝试？请不要从这批损益推荐最优门槛、持有期或窗口。

按优先级给出具体缺陷、最有价值的下一步验证及停止条件；区分本地已验证证据与外部未知。当前目标是1.3，父级1.5是历史记录。本包未经过外部审阅。
""")
    write("EXCLUSIONS.md", """
# 范围和排除

本包自含本轮直接使用的已有市场、分红、IF特征和上游本地输入，以及完整训练代码、保存参数、节点、预测和账户。父级原始获取回执保留，但未重新下载数据，不重复复制远端所有压缩包或虚拟环境。

新增频率评价直接使用上一轮已核对的170条账户范围和1496条含重复的周期记录，再加入本轮18条账户。频率复算所需直接表已纳入inputs/frequency_reference_accounts.csv和frequency_reference_cycles.csv，不重复打包上一轮所有原始账簿；本轮不重新声称验证那170条账户的训练过程。

verify重算本轮标签、代入已保存模型并检查模型方程与账簿；不重新拟合或重跑账户，不宣称重新认证所有原始网络来源。父级因子和节点沿用其保存输入与协议。

不包含缓存、解压验证目录和ZIP外最终回执。没有实盘订单、外部审阅或独立前向验证。未来实际损益、缺失时段和今日信号均不作推断。
""")

    nav = pd.read_csv(OUT / "results/完整逐日账户.csv", parse_dates=["date"])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    for model, policy, label, color in [
        ("B_IF_JOINT", "JOINT_PRIMARY", "联合规则：6次，胜率83.3%，夏普0.361", "#176787"),
        ("B_IF_JOINT", "EXPECTANCY_CONTROL", "仅预期收益：9次，夏普0.348", "#b67a25"),
        ("EVENT_ONLY", "EVENT_RULE_DIAGNOSTIC", "全部节点：10次，夏普0.363", "#59636f")]:
        d = nav.loc[(nav.model == model) & (nav.policy == policy) & (nav.scenario == "STRESS")]
        axes[0].plot(d.date, d.equity_cny/10000, label=label, color=color, linewidth=1.8)
        axes[1].plot(d.date, -d.drawdown*100, color=color, linewidth=1.6)
    axes[0].set_title("510300少数节点：联合筛选后的完整账户\n20万元，原压力成本，2018-05-02至2026-08-18", loc="left", fontsize=14)
    axes[0].set_ylabel("账户权益 / 万元")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set_ylabel("回撤幅度 / %")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", color="#d5dde2", alpha=.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("全部空仓交易日均计入；已有历史的探索训练，净夏普目标1.3尚未达到。", fontsize=10)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures/联合规则完整账户.png", dpi=150)
    plt.close(fig)
    frequency_table = ["| 完整自然年 | 全部IF原节点 | IF联合规则实际交易 | 最低要求 |", "|---|---:|---:|---:|"]
    scoped = annual.loc[(annual.source_family == "CURRENT_JOINT") & (annual.cost == "STRESS") & annual.complete_calendar_year]
    for year in sorted(scoped.year.unique()):
        all_nodes = scoped.loc[(scoped.model == "EVENT_ONLY") & (scoped.year == year), "complete_cycles_entered"].iloc[0]
        primary_nodes = scoped.loc[(scoped.model == "B_IF_JOINT") & (scoped.policy == "JOINT_PRIMARY") & (scoped.year == year), "complete_cycles_entered"].iloc[0]
        frequency_table.append(f"| {year} | {all_nodes} | {primary_nodes} | 5 |")
    write("最新频率要求与逐年评价.md", f"""
# 当前要求：每个完整自然年至少五次交易

用户在本轮联合训练完成后明确要求“一年至少五次交易”，已更新当前研究授权和节点训练口径。一次完整持仓周期计一次，按入场年归属；买入、卖出及期间加减仓不拆成多次机会。每个完整年都须满足，不能以高频年份补偿低频年份，也不能用多年平均替代。

本轮IF节点的逐年数量：

{chr(10).join(frequency_table)}

2018年的账户从5月开始、2026年只评价至8月18日，均单列为不完整年份，不用年化换算制造五次，也不凭不完整记录判断未来全年。完整年份根据本地市场日历该年的首末交易日与账户覆盖范围确定；保留零交易年。

此前14个主历史数值达到夏普1.3和回撤10%的旧模型也逐一检查：2020—2025各完整年中，每个方案最少年度周期数为0至2，**没有一个满足每个完整年至少5次**。例如RUNS_REFERENCE_BLEND依次为5、4、2、2、7、6次。过去“年均约四五次”不能等同于现在的逐年下限。

追加频率检查共覆盖{annual_summary['reviewed_account_records']}条账户记录、{annual_summary['full_year_record_count']}条完整年度记录；同时满足年度至少5次、完整账户夏普1.3和回撤10%的记录为{annual_summary['all_three_requirements_pass_records']}。其中含模型、成本和对照重复，不视为独立研究数量。

后续研究先检查完整候选节点母集是否能提供每年至少5次，再训练筛选和运行账户；最终仍以实际完整持仓周期逐年验收。当前IF节点母集本身不够，不能靠增加模型复杂度解决。新门槛不会改写已经完成的模型、交易或此前冻结记录。
""")
    status = ROOT / "RESEARCH_STATUS.md"
    old = status.read_text(encoding="utf-8-sig")
    marker = "# 510300 研究权威状态"
    entry = "> 2026-09-24 按用户“少数节点、胜率/盈亏比/变盘/夏普结合、拒绝过拟合；请继续”完成 `510300_IF_NODE_JOINT_PAYOFF_TRAINING_V1` 实际训练。124月末样本、10原节点，222次逻辑回归、444次条件幅度回归、111份经验参考，330条预测、18组完整2016日账户。压力IF联合规则6次5胜、金额盈亏比3.84、净夏普0.361、回撤2.85%、净利润9100.03元；全部节点10次8胜、夏普0.363。用户最新新增“一年至少五次交易”，当前按每完整自然年不少于5个完整持仓周期、按入场年计数，覆盖188条保存账户联合通过0；IF原母集每完整年0至2次，旧14方案主历史最少年度次数0至2，均不足。权威配置已替代旧非配额频率。本固定表示停止，不进行事后阈值/窗口补救；目标1.3和10%及新年度频率未完成，原失败保留，0下载/独立前向/订单。见[训练结果](reports/research/510300_if_node_joint_payoff_training_v1/训练结果.md)、[逐年结论](reports/research/510300_if_node_joint_payoff_training_v1/最新频率要求与逐年评价.md)。"
    if marker not in old:
        raise ValueError("未找到当前研究状态插入点。")
    if entry not in old:
        status.write_text(old.replace(marker, marker + "\n\n" + entry, 1), encoding="utf-8")
    print("实际训练结论、全部节点归因、停止条件和账户图已完成。")


if __name__ == "__main__":
    main()
