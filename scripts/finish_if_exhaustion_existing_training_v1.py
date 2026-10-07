"""整理已完成的 IF 训练结果、静态图和交付说明；不进行拟合或回测。"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_if_exhaustion_existing_training_v1"


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    nav = pd.read_csv(OUT / "results/完整逐日账户.csv", parse_dates=["date"])
    nav = nav[nav.scenario == "STRESS"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}, layout="constrained")
    curves = [
        ("B_IF_EXHAUSTION", "NET_POSITIVE_DIAGNOSTIC", "IF模型，仅覆盖费用：9次", "#145c80"),
        ("EVENT_ONLY", "EVENT_RULE_DIAGNOSTIC", "全部事件对照：10次", "#ba7930"),
        ("A_PRICE", "NET_POSITIVE_DIAGNOSTIC", "价格模型，仅覆盖费用：6次", "#778895"),
        ("B_IF_EXHAUSTION", "ERROR_BUFFER_PRIMARY", "IF模型，误差缓冲：0次", "#4b8466"),
    ]
    for model, policy, label, color in curves:
        frame = nav[(nav.model == model) & (nav.policy == policy)]
        axes[0].plot(frame.date, frame.equity_cny / 10000, label=label, color=color, linewidth=1.8)
        axes[1].plot(frame.date, frame.drawdown * 100, color=color, linewidth=1.5)
    axes[0].set_title("IF 卖压缓解：20万元账户完整日历（压力成本）\n2018-05-02—2026-08-18；全部空仓日计入", loc="left", fontsize=14)
    axes[0].set_ylabel("账户权益 / 万元")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[1].set_ylabel("账户回撤 / %")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
    for ax in axes:
        ax.grid(axis="y", color="#d9e1e7", alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("IF信号截至2026-08-12；末段仅完成已有交易。图示为历史探索，尚未达到净夏普1.5。", fontsize=9)
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures/完整账户与回撤.png", dpi=150)
    plt.close(fig)
    decision = {
        "status": "STOP_CURRENT_EVENT_AND_FORECAST_REPRESENTATION",
        "basis": "两档成本下所有训练规则和无筛选事件对照均未达到全账户净夏普1.5；误差缓冲主规则全部空仓。",
        "positive_but_insufficient": "B月末MSE相对价格基准改善8.20%，但九次事件账户压力夏普仅0.348，低于无筛选事件0.363。",
        "no_parameter_rescue": True, "old_rejections_unchanged": True, "goal_achieved": False,
        "new_collection_enabled": False, "orders_authorized": False,
    }
    (OUT / "branch_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    documents = {
        "用户需求与执行范围.md": """# 用户需求与执行范围

只操作510300与现金。本金20万元；全账户扣成本净夏普至少1.5、最大回撤目标10%。一年四五次机会可以接受，不凑交易次数，其余时间空仓。

最新明确要求：“请加速干”“不用那么严格，直接进入训练”“不用采集”。本轮因此不等待旧20事件准入门，不联网补数据，直接完成已有10个IF压力耗竭事件的训练和账户。

宽松探索只改变本轮研究准入，不改变真实时间顺序、费用、空仓日计入和如实报告。原冻结拒绝不改；SELECTED_MIX_BAND10_SIMPLE2保持终止。
""",
        "00_阅读导航.md": """# 阅读导航

本轮已经实际训练。首先读“训练结果.md”和“summary.json”，再看“results/账户指标.csv”“results/机会成交账簿.csv”。结论：本轮没有达到净夏普1.5的账户。

- protocol.json / freeze.json：标签和拟合前固定的规则、代码和本地输入身份。
- inputs：实际计算所需既有行情、分红、IF逐合约/到期/现货、保存状态、原回执和原拒绝。
- code：本轮完整训练代码、账户引擎、原纯计算函数及父代码参考。
- models：330份逐期参数或均值快照，3份截止信号末日的最终快照。
- results：124个月末样本、10个候选事件、330条逐期预测、18组账户及完整2016日历交易日。
- figures：压力成本下完整账户及回撤图。
- branch_decision.json：当前事件及预测表示停止，不做参数补救。
- FILE_INDEX.csv：所有其他包成员的字节数和SHA-256。交付回执在ZIP旁的研究目录，避免自引用。

在本目录安装requirements.txt列出的环境后，运行 `python -X utf8 code/if_exhaustion_existing_training_v1.py verify --root .`，可以从包内输入重算IF因子及状态、成熟标签、模型方程、保存预测和现金流。verify不重新拟合或新建回测账户。保存训练代码的train入口会拒绝覆盖既有summary.json。

从旧原始IF合约重算的3454日状态与原保存结果一致。模型目标为10日；这里的month字段为完整原点日期。最后IF信号日期8月12日，最后已有事件的10日持有在8月18日完成；8月12日后没有新IF信号输入。

这份ZIP可交由GPT审阅；本次没有上传，也没有声称已经获得外部审阅或独立前向验证。
""",
        "GPT审阅提示词.md": """# 可复制审阅提示词

请审阅这个510300本地训练包。用户要求20万元，完整账户扣成本净夏普至少1.5、最大回撤10%，允许很少交易、长期空仓；明确不采集新数据，允许放宽旧样本准入直接训练。

先给出可用、不可用或证据不足的结论，并用具体文件和数值支持。请检查：月末训练和事件预测的标签成熟时序；IF一日滞后；原状态和同一家族旧拒绝是否透明；IF基差/持仓是否被误称为真实投资者卖出；100个月末与10个事件是否混淆；2018年模型预热结束到2026年的所有空仓日；8月12日至18日仅清算已有事件；分红应收与实际到账；误差缓冲和无预测事件对照。

请解释MSE改善8.20%为什么没有带来1.5的全账户夏普；评价无筛选事件对照是否削弱模型筛选价值。不要把减少仓位降低回撤等同于提升夏普，也不要建议据历史表现反复挑选阈值、持有期或年份。按优先级给出最有价值的后续研究、可复核条件和停止条件；区分已有数据能回答的内容与实际缺失的数据。
""",
        "EXCLUSIONS.md": """# 包范围及排除

包含所有本轮计算直接使用的本地数据、已保存状态及上游IF逐合约和现货输入。原来源URL和获取回执保留。旧父代码作为参考，完整本轮计算入口为code/if_exhaustion_existing_training_v1.py。

不复制整个研究仓库、所有历史试验、虚拟环境和未使用的全部远端原始月度压缩包；既有逐合约表及其来源哈希足以复算本轮使用的IF表示，但不声称包内重建了整个抓取过程。没有网络采集或当前9月22日市场判断。

verification及解压副本不收入同一ZIP，避免循环；包外交付回执记录新解压保存结果复算。未模拟最小报价单位、涨跌停排队、真实冲击和独立成交；没有上传、外部专家审阅、实盘授权或未来表现保证。
""",
    }
    for name, content in documents.items():
        (OUT / name).write_text(content, encoding="utf-8")
    print("研究结果、停止条件、净值图与交付说明已整理。")


if __name__ == "__main__":
    main()
