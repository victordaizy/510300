"""呈现已经完成的月度分解，并更新当前历史研究导航。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from multidim_financing_turnover_decomposition_v1 import ROOT, OUT, LABELS, now, read, save, sha


FIGURE = "融资变化的成交规模与相对强度.png"
REPORT = "融资变化与市场成交规模_历史发现.md"


def draw(frame: pd.DataFrame, result):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10.5})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 10.5), sharey=True)
    fig.patch.set_facecolor("#f7f8fa")
    scale_color, intensity_color = "#dc9b3f", "#217e83"
    y = np.arange(len(frame))
    for ax, (key, title), total in zip(axes, LABELS.items(), result["summary"][:3]):
        a = frame[key + "_scale_component"].to_numpy()
        b = frame[key + "_intensity_component"].to_numpy()
        left = np.where(a * b > 0, a, 0)
        ax.set_facecolor("#ffffff")
        ax.barh(y, a, height=.62, color=scale_color)
        ax.barh(y, b, left=left, height=.62, color=intensity_color)
        ax.scatter(frame[key + "_change"], y, color="#1e293b", s=17, zorder=4)
        ax.axvline(0, color="#64748b", lw=.8)
        ax.axhline(11.5, color="#64748b", ls="--", alpha=.6, lw=.8)
        ax.grid(axis="x", color="#cbd5e1", alpha=.45)
        ax.set_axisbelow(True)
        ax.set_title(f"{title}\n规模分量更大：{total['scale_larger_months']}/24个月", loc="left", fontsize=12, pad=16)
        ax.set_xlabel("日均金额的月度变化（亿元／日）", labelpad=10)
        ax.set_yticks(y, frame.factor_month.to_list())
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.margins(x=.13)
    axes[0].invert_yaxis()
    fig.suptitle("融资买入与偿还多受共同成交规模影响，净融资变化有所不同", x=.075, y=.981, ha="left", fontsize=17, fontweight="bold")
    fig.text(.075, .94, "2024—2025全部24个月｜先按实际交易日计算日均，再与前月比较｜三个面板横轴尺度不同", fontsize=11, color="#475569")
    fig.legend(handles=[Patch(color=scale_color, label="成交规模分量"), Patch(color=intensity_color, label="相对强度分量"),
                        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1e293b", label="实际金额变化", markersize=5)],
               loc="upper left", bbox_to_anchor=(.071, .925), frameon=False, ncol=3, columnspacing=2)
    fig.text(.075, .046, "同号分量堆叠，异号分量分列零轴两侧；黑点等于两项之和。分解是会计恒等式，不是后续收益预测或因果识别。", fontsize=10, color="#475569")
    fig.text(.075, .023, "相对强度＝融资金额／股票成交额。分子与分母证券范围不完全一致；使用纠正后的融资副本及官方月度成交汇总。", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.075, right=.983, top=.84, bottom=.11, wspace=.13)
    fig.savefig(OUT / FIGURE, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def table_lines(frame: pd.DataFrame):
    lines = ["| 月份 | 日均买入变化 | 其中规模分量 | 其中强度分量 | 日均净融资变化 | 其中规模分量 | 其中强度分量 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"| {row.factor_month} | {row.buy_change:+.2f} | {row.buy_scale_component:+.2f} | {row.buy_intensity_component:+.2f} | {row.net_change:+.2f} | {row.net_scale_component:+.2f} | {row.net_intensity_component:+.2f} |")
    return lines


def write_report(frame: pd.DataFrame, result):
    july = frame.set_index("factor_month").loc["2024-07"]
    jan = frame.set_index("factor_month").loc["2025-01"]
    april = frame.set_index("factor_month").loc["2025-04"]
    summary_rows = []
    for item in result["summary"][:3]:
        summary_rows.append(f"| {item['amount']} | {item['scale_larger_months']}/24 | {item['intensity_larger_months']}/24 | {item['raw_and_intensity_opposite_months']}/24 |")
    content = [
        "# 融资变化与市场成交规模的历史拆解",
        "**2024—2025年的24个月中，融资买入与隐含偿还的月度日均变化全部同向。两者各有21个月是成交规模分量更大；净融资则有20个月是相对强度分量更大。** 这支持在多维评分中区分市场成交规模与借入相对偿还的强弱。本次没有建立新的收益优势，也没有改写旧交易规则。",
        "研究按事先固定的两年全部月份计算，只使用2023年12月作首月比较基准；融资采用已经纠正2024年8月8日缓存零值的副本。先除以实际交易日数，避免春节等假期使整月金额发生机械变化。",
        "| 金额 | 成交规模分量更大的月份 | 相对强度分量更大的月份 | 金额变化与强度变化反向的月份 |\n|---|---:|---:|---:|\n" + "\n".join(summary_rows),
        "2024年，买入、隐含偿还的规模分量分别在10/12、12/12个月较大，净融资的强度分量在9/12个月较大；2025年对应为11/12、9/12、11/12。这里只比较分量大小，不把月度观察当成独立冲击，也不把算术分解称为经济因果证明。",
        "## 为什么同样的金额下降可以代表不同状态",
        f"**2024年7月：买入金额下降，融资买入相对强度却略升。** 日均融资买入比6月减少{-july.buy_change:.2f}亿元，其中成交规模分量为{july.buy_scale_component:+.2f}亿元，相对强度分量为{july.buy_intensity_component:+.2f}亿元。直接把买入下降打成杠杆需求走弱，会遗漏全市场成交收缩这一背景。",
        f"**2025年1月：隐含偿还金额下降，相对强度却略升。** 日均隐含偿还下降{-jan.repay_change:.2f}亿元，成交规模分量为{jan.repay_scale_component:+.2f}亿元，强度分量为{jan.repay_intensity_component:+.2f}亿元。金额减少不能单独证明偿债约束减轻；隐含偿还也不能等同于强制卖出。",
        f"**2025年4月：净融资恶化主要体现在相对强度。** 日均净融资由前月的{april.previous_net_daily:+.2f}亿元变为{april.net_daily:+.2f}亿元，变化{april.net_change:+.2f}亿元。其中成交规模分量为{april.net_scale_component:+.2f}亿元，强度分量为{april.net_intensity_component:+.2f}亿元。成交收缩反而抵消了一小部分净融资恶化。",
        "这些是完整自然月的背景描述。原五个高分事件所在月份均另存于明细，不能把整月统计回填到月中。例如2024年9月的整月融资改善含9月下旬变化，不能据此解释9月18日入场前已知的信息。",
        "## 分解怎样计算",
        "令 A 为每月日均股票成交金额，B 为日均融资买入，R 为按余额恒等式推算的日均偿还；净融资 N＝B－R。令 q＝X/A，X 分别取 B、R、N。则相邻月份之间有：",
        "**金额变化＝两月平均相对强度 × 成交规模变化＋两月平均成交规模 × 相对强度变化。**",
        "这来自 X＝A×q 的乘法关系。采用两期对称分解，不拟合线性权重，也无需选择先改变规模还是先改变强度。规模与强度同时变化的交叉项平均分配到两项，二者允许同向或抵消。算术恒等式不能证明股市收益对这些变量的关系；后者仍可能依赖宏观、资金约束和价格已反映的程度。",
        "买入和偿还共有的规模变化会在相减时大量抵消，因此净融资不能直接套用两个总额的结论。另一方面，净融资相对强度仍受证券范围、融资标的和投资者构成影响，并非已经识别出的外生冲击。",
        "## 对联合评分的具体意义",
        "本次支持把融资描述拆为成交规模、融资买入相对强度、隐含偿还相对强度及二者之差。买入、偿还、净融资存在精确恒等关系，联合评分时应明确这种依赖，不把四个名称当作四份独立证据。是否保留其中哪一种表达，应由新增信息和已固定的历史比较决定。",
        "旧融资连锁研究已经使用过融资买入／成交、净变化／成交、偿还／成交及市场广度交互，其2019—2020开发选择期没有通过当时的原门槛。本次没有重跑、调节或恢复那套失败规则；该旧门槛也不等于当前夏普1.2目标。",
        "当前不追加新的融资收缩过滤器。月度分解纠正了输入含义，但尚不足以支持一个新的五日交易条件。后续历史发现回到指数共同驱动，先检查已有信用需求、政策原因与银行间资金条件是否提供独立的新信息，再决定是否进入一次固定的联合比较。",
        "## 全部月份与边界",
        "下表单位均为亿元／日，表示相对前月日均金额的变化。偿还完整分解、相对强度及官方源字段保留在CSV和Parquet中。",
        "\n".join(table_lines(frame)),
        f"![全部月份的融资变化分解]({(OUT / FIGURE).as_posix()})",
        "成交分母取上交所A股与深交所股票总计；深圳范围含B股和存托凭证。融资分子包含合资格ETF等证券，二者并非完全同一证券集合，故比值只称相对强度代理，不称精确融资成交占比。交易所[融资交易月报](https://www.sse.com.cn/aboutus/publication/monthly/documents/c/10061121/files/286d47959a4e4f45b406a108a9e4777b.pdf)列有ETF；[深交所2025年4月成交月报](https://docs.static.szse.cn/www/market/periodical/month/W020250513504876446032.html)分别列示股票、基金及债券成交，不能把这些列混为同一分母。",
        "月度分母复用官方历史年鉴、月报解析资料，未证明每条资料在月末当时可得。这里不计算随后收益，不把分解用作历史交易输入。原五日研究与完整账户结果保持原状态：已有主要期主账户夏普点值1.242、年化2.00%，仍未达到完整目标；本次没有新增账户或已获准使用的策略。",
        f"已完成三项必要计算核对：全部24个月完整；融资交易日与股票交易日历一致；两项分解之和与实际变化一致，最大浮点差{result['checks']['maximum_identity_error_cny100m_per_day']:.2g}亿元／日。",
        "依据文件：`protocol.json`固定口径；`全部24个月分解.csv`列出全部观测；`全部月度输入与分解.parquet`保留原始组成和基准月；`result.json`提供分年结果及旧研究去重记录。"
    ]
    (OUT / REPORT).write_text("\n\n".join(content) + "\n", encoding="utf-8")


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("月度分解已经交付，不覆盖结果。")
    result = read(OUT / "result.json")
    frame = pd.read_csv(OUT / "全部24个月分解.csv")
    draw(frame, result)
    write_report(frame, result)
    rel = str(OUT.relative_to(ROOT)).replace("\\", "/")
    summary = "2024—2025固定24个月：融资买入和隐含偿还变化24次同向，两者各21个月由成交规模分量占优；净融资20个月由相对强度分量占优。仅月度会计分解，没有新增收益优势或交易规则。"
    next_question = "停止追加融资收缩过滤器；回到指数共同驱动，先去重已有信用需求、政策原因与银行间资金条件联合研究，只在有独立新信息时进行一次固定低复杂度比较。月度强度分解不回填五日信号。"
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    config = read(path)
    config.update({"latest_completed_study": rel + "/result.json", "latest_report": rel + "/" + REPORT,
                   "current_study": rel + "/protocol.json", "latest_result_summary": summary,
                   "next_historical_question": next_question, "updated_at": now(),
                   "latest_completed_branch_boundary": "融资构成分解已完成；相对强度不是新增已验证信号，现阶段不继续从同一收缩分组增加过滤器。",
                   "goal_achieved": False})
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    config = read(path)
    config.update({"latest_historical_diagnostic_at": now(), "latest_historical_report": rel + "/" + REPORT,
                   "latest_financing_scale_intensity_diagnostic": rel + "/result.json",
                   "current_driver_continuation_classification": "PROGRESS_FINANCING_SCALE_INTENSITY_DECOMPOSITION",
                   "current_driver_consecutive_blocked_goal_turns": 0,
                   "latest_continuation_report": rel + "/" + REPORT,
                   "latest_continuation_classification": "PROGRESS_FINANCING_SCALE_INTENSITY_DECOMPOSITION",
                   "next_research_question": next_question, "goal_achieved": False})
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("delivery_receipt.json", {"completed_at": now(), "status": "COMPLETED_REVIEWABLE_MONTHLY_DECOMPOSITION",
                                   "report": rel + "/" + REPORT, "figure": rel + "/" + FIGURE,
                                   "result_sha256": sha(OUT / "result.json"), "new_model_fits": 0,
                                   "new_accounts": 0, "goal_achieved": False})
    print("已生成完整月度分解报告和图，并更新历史研究导航。")


if __name__ == "__main__":
    main()
