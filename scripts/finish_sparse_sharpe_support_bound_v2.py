"""整理用户1.3新目标下的保存账户复核结论。"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_sparse_sharpe_support_bound_v2"


def text(name, value):
    (OUT / name).write_text(value.strip() + "\n", encoding="utf-8")


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    results = pd.read_csv(OUT / "results/保存账户的稀疏上界.csv")
    scenarios = pd.read_csv(OUT / "results/机会频率数学示例.csv")
    chosen = [
        ("分红发放 B", "510300_dividend_payment_event_training_v1", "B_DIVIDEND"),
        ("成分吸收率 B", "510300_absorption_available_members_training_v1", "B_ABSORPTION"),
        ("IF 卖压缓解 B", "510300_if_exhaustion_existing_training_v1", "B_IF_EXHAUSTION"),
        ("资本市场支持政策主规则", "510300_equity_support_policy_event_v1", "REVIEW_CLOSE_NEXT_OPEN_PRIMARY"),
    ]
    table = ["| 保存账户 | 完整评价期 | 有敞口日 / 全部交易日 | 压力成本净夏普 | 最大回撤 | 给定参与天数的数学上限 |",
             "|---|---|---:|---:|---:|---:|"]
    for label, study, model in chosen:
        rows = results[(results.study == study) & (results.model == model) & (results.scenario == "STRESS") & (results.exposure_days > 0)]
        if len(rows) != 1:
            raise ValueError("报告账户匹配不唯一。")
        item = rows.iloc[0]
        table.append(f"| {label} | {item.start} 至 {item.end} | {item.exposure_days} / {item.n_calendar_days} | {item.actual_sharpe:.6f} | {item.actual_max_drawdown:.4%} | {item.cap_with_same_exposure_days:.6f} |")
    examples = ["| 年均机会数 | 每次持有交易日 | 十年内有敞口日 | 数学上限 | 达到 1.3 所需持有日均值 / 总体标准差 |",
                "|---:|---:|---:|---:|---:|"]
    for item in scenarios.itertuples():
        ratio = f"{item.required_active_mean_to_population_std:.4f}" if pd.notna(item.required_active_mean_to_population_std) else "此参与天数无法达到"
        examples.append(f"| {item.opportunities_per_year} | {item.holding_days_per_opportunity} | {item.exposure_days} | {item.theoretical_cap:.4f} | {ratio} |")
    dividend = results[(results.study == chosen[0][1]) & (results.model == "B_DIVIDEND") & (results.scenario == "STRESS") & (results.exposure_days > 0)].iloc[0]
    text("研究结论.md", rf"""
# 510300：净夏普目标改为 1.3 后的已有账户复核

用户最新指令“夏普1.3就行”已写入训练配置。20万元完整账户、最大回撤10%、只做510300与现金、允许长期空仓以及不用采集均保留。近期7项研究的114条保存账户记录按新目标重新判定，仍没有策略账户同时达到净夏普1.3和最大回撤10%。分红模型原先受1.5门槛限制的数学上界问题已解除，但其实际压力成本净夏普仅{dividend.actual_sharpe:.6f}，远未达标。

## 目标变化带来的具体区别

分红 B 在1849个完整交易日中仅15天有敞口。现金和无风险日收益都取0时，在这些参与天数固定的前提下，即使15天的净收益全部相等且为正，全账户年化净夏普也最多为1.406488。它低于旧门槛1.5，但高于新门槛1.3。新目标要求至少{dividend.minimum_nonzero_days_for_target}个非零收益日，旧目标要求18天；这只是极其乐观的必要条件，并不是增加交易次数的建议。

该模型实际期末资产为{dividend.ending_equity_cny:,.2f}元，压力成本最大回撤{dividend.actual_max_drawdown:.4%}。降低验收目标没有改变任何历史成交、净值、成本或预测，也没有把失败模型改判为可用。

## 几组已经训练或评估的账户

{chr(10).join(table)}

不同账户的评价日期、事件数量和机制不同，表格用于定位问题，不作同区间绩效排名。IF仅作信号输入，模拟可交易资产仍是510300与现金。政策项为固定事件规则，不是新增拟合。

## 完整范围

本次保存并计算7项研究的170550条逐日记录。114条账户记录包含28条基准记录、86条策略记录；成本情景、相同成交路径及不同预测版本均保留，合计55条唯一保存路径。因此，114不能解释为114项独立实验，也不能把多个相同路径当作独立证据。

按1.3新目标，86条策略记录中，46条全程空仓、夏普未定义；4条的固定参与天数数学上界仍低于1.3；36条没有被该上界排除，但实际净夏普和回撤联合达标数仍为0。4条上界不足记录对应吸收率成熟均值与分红价格基准两种账户，各含基础/压力成本；分红B不在其中。全空仓不记成夏普0或达标。

## 上界的定义和推导

设完整交易日数为n，最多k个日收益非零，年化因子A=242。采用日收益样本标准差（ddof=1），空仓收益和无风险收益均为0。本轮先由保存净值重新计算日收益，并核对全部无敞口日收益确实为0。

当0<k<n时：

$$|S|\leq\sqrt{{\frac{{A k(n-1)}}{{n(n-k)}}}}.$$

令s为日收益之和、q为日收益平方和。柯西不等式给出s²≤kq，而S²=A·s²·(n−1)/(n·(nq−s²))，代入即可得到上界。所有非零日收益相等时取等号，正收益对应正上界。这是代数性质，不是未来可实现收益预测。持有日中若还有零收益，实际非零天数给出的上界会更低。

目标为T时，必要非零日数为ceil(T²n²/[A(n−1)+T²n])。k=0时夏普未定义；k=n时该稀疏性关系不提供有限上界。上界足够高也不说明存在预测优势、独立验证或可交易机会。

## 一年四五次是否与1.3冲突

不冲突，具体还取决于每次持有天数及净收益的稳定程度。下表只作数学说明：固定10年、每年242日、各次持有不重叠、空仓收益为0。它没有生成行情或账户，不是推荐交易配额。

{chr(10).join(examples)}

“每年4次、每次5天”的假设并不因空仓太久而必然达不到1.3；真正需要证明的仍是这些机会在扣费后有足够稳定的收益。不会为了改善夏普而删除空仓日、缩短评价期、延长已失败规则的持有期或凑交易次数。

## 本轮完成与范围

已更新目标配置、复算近期保存账户指标、重判数值门槛并保存可复核计算。本轮新增模型拟合0、新增回测账户0、新下载0。用户先前要求的实际训练结果在来源目录及本包直接输入中保留；本轮只是目标变更后的计算，不冒称新训练。此前1.5协议与结果保留在V1和authority_update目录。

整体研究目标尚未实现。后续已有本地数据训练以1.3为目标；既有终止策略和已失败机制不因目标降低而自动重启。未建立外部审阅或独立达标证据。
""")
    text("用户需求.md", """
# 用户需求和本轮动作

原需求：只操作510300，20万元，最大回撤10%，寻找稀疏且有明确依据的机会，一年四五次可以接受，不是配额。用户要求加快、直接进入训练、不用采集。

最新明确变更：**“夏普1.3就行”**。净夏普验收目标从1.5调整为1.3，其他要求保持。本轮直接更新已有本地训练配置并重判保存账户，未请求额外确认。

原目标已保存在authority_update/previous_mandate.json；变更回执为authority_update/user_target_revision.json。原1.5分析不覆盖，新计算为V2。
""")
    text("交付导航.md", """
# 阅读顺序

1. 研究结论.md：新目标、目前结果和稀疏参与的数学含义。
2. 用户需求.md、protocol.json、authority_update：用户修改目标的依据及旧目标记录。
3. results/保存账户的稀疏上界.csv：114条账户及两套目标标记；当前达标列为current_numeric_targets_pass，旧标记单独保存。
4. inputs：七项研究完整逐日账户、原指标、协议、汇总和原交付回执；mandate.json是本轮配置快照。
5. code/sparse_sharpe_support_bound_v2.py：由原净值重算指标、上界和1.3判定；formula_checks.json是纯数学等式检查。
6. results/机会频率数学示例.csv：非市场模拟、非交易配额的数学例子。
7. FILE_INDEX.csv：ZIP内全部其他文件的大小与SHA-256。

解压目录中安装requirements.txt的本地Python环境可运行code/sparse_sharpe_support_bound_v2.py的verify子命令，并用--root指定解压目录。该命令不拟合模型、不生成新账户、不下载数据。不要对已完成目录再次运行freeze或calculate。

verify只需本包保存文件。prepare/finish/package脚本记录本轮工作流程，其中prepare和package依赖原工作区路径，不属于便携复算入口。
""")
    text("GPT审阅提示词.md", """
请审阅本包对用户最新目标的落实：仅510300与现金，20万元，净夏普至少1.3，最大回撤10%，稀疏交易可接受，不采集新数据。

请先读研究结论、protocol和目标变更回执，再核对以下问题：

- 是否从保存净值与全部交易日重新计算夏普、回撤并按1.3判定，而不是沿用1.5旧标记？
- 稀疏日收益夏普上界、取等条件和最少非零日数是否正确？0天和全持有的退化情形是否明确？
- 是否明确区分数学上界、实际收益、预测优势及独立证据？分红B已不受1.3上界排除，但是否仍明确未达标？
- 是否完整保留空仓日、成本情景和重复路径，避免把114条记录当独立实验？
- 包内直接账户足够复算哪些结论，哪些上游训练结论需要原包？

请给出具体问题及文件位置，并按重要性排序。不要以凑交易次数、删除空仓日或事后修改持有期作为达标方案。这个ZIP仅供审阅，不表示已经完成外部审阅或批准交易。
""")
    text("EXCLUSIONS.md", """
# 交付范围

本包自含当前计算所需的原逐日净值、暴露记录、原指标、参数、汇总、交付回执以及计算代码。不重复打包上游每个模型的全部市场输入、源码或PDF；因此可以复算本轮上界和目标判定，不能单靠本包完整重建所有上游模型训练。上游收据用于定位对应原交付，不表示本轮重新确认全部科学有效性。

不包括本机虚拟环境、缓存字节码、新解压复核目录和根目录交付回执。根目录回执在ZIP外生成以避免自引用；各来源输入的原回执保留。旧1.5结果仅作为历史说明保留，不作为当前通过标记。

数学示例使用构造的数列检查代数关系，没有生成510300市场路径、收益预测或交易账户。空仓收益和无风险收益均按0；若改变此口径，该上界的使用需重新推导。所有上界都只约束所给日期数量，不推断未来必然相同。
""")
    (OUT / "branch_status.json").write_text(json.dumps({
        "status": "COMPLETED_USER_TARGET_REVISION_AND_SAVED_ACCOUNT_REASSESSMENT",
        "target_net_sharpe": 1.3, "numeric_target_pass_strategy_records": summary["numeric_target_pass_strategy_records"],
        "new_model_fits": 0, "new_accounts": 0, "new_downloads": 0,
        "goal_achieved": False, "orders_authorized": False,
        "next_research_target": "在已有本地数据研究中使用净夏普1.3；不自动重启旧终止策略。"
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("1.3目标下的结论、全部结果导航及审阅说明已写入。")


if __name__ == "__main__":
    main()
