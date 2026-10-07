"""将选定原报告的经营解释接到财务指标构成，输出可复算桥表和图。"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_cash_quality_causes_v1"


def main():
    if (OUT / "case_bridges.json").exists():
        raise SystemExit("已保存案例调节，不覆盖既有结果。")
    source = json.loads((OUT / "case_source_values.json").read_text(encoding="utf-8"))
    sample = pd.read_parquet(OUT / "全部49条公司财务变化.parquet")
    sample["announcement_id"] = sample.announcement_id.astype(str)
    # 对照原页中的两期数字；该核对不替代报表范围和列顺序的阅读。
    for key in ["muyuan_2024h1", "commodity_city_2025q3", "shangji_2022q3"]:
        case = source[key]
        pages = json.loads((OUT / "sources" / (case["announcement_id"] + "_pages.json")).read_text(encoding="utf-8"))
        text = "".join(page["text"] for page in pages)
        for field, value in case.items():
            if isinstance(value, list) and len(value) == 2 and all(isinstance(v, float) for v in value):
                for number in value:
                    assert f"{number:,.2f}" in text, f"原文没有金额：{key} {field} {number}"
    case = source["muyuan_2024h1"]
    differences = {name: case[name][0] - case[name][1] for name in ["cfo", "consolidated_net_profit", "inventory_decrease", "operating_receivables_decrease", "operating_payables_increase"]}
    working_capital = sum(differences[name] for name in ["inventory_decrease", "operating_receivables_decrease", "operating_payables_increase"])
    residual = differences["cfo"] - differences["consolidated_net_profit"] - working_capital
    bridge = {
        "comparison": case["comparison"], "source_pdf_page": case["page"],
        "changes_cny": differences, "working_capital_change_cny": working_capital,
        "working_capital_fraction_of_cfo_change": working_capital / differences["cfo"],
        "remaining_adjustments_change_cny": residual,
        "not_causal_share_of_ttm_factor_or_share_return": True,
    }
    ids = ["1224710669", "1220788951", "1214728625"]
    cases = sample.set_index("announcement_id").loc[ids]
    compact = cases[["ts_code", "cash_component", "profit_component", "asset_component", "L04_change"]].copy()
    for col in ["cash_component", "profit_component", "asset_component", "L04_change"]:
        compact[col + "_pp"] = compact[col] * 100
    compact.to_csv(OUT / "三例指标变化的算术构成.csv", encoding="utf-8-sig")
    combined = {"muyuan_cfo_bridge": bridge,
                "muyuan_profit_component_fraction": float(cases.loc["1220788951", "profit_component"] / cases.loc["1220788951", "L04_change"]),
                "commodity_city_contract_liability_change_cny": source["commodity_city_2025q3"]["contract_liabilities"][0] - source["commodity_city_2025q3"]["contract_liabilities"][1],
                "case_selection_used_market_returns": False}
    (OUT / "case_bridges.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(cases, differences, residual)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


def plot(cases, differences, residual):
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    background = "#f5f3ed"
    fig.patch.set_facecolor(background)
    y = np.arange(3)
    positive, negative = np.zeros(3), np.zeros(3)
    for col, label, color in [("cash_component", "现金流变化", "#187b73"), ("profit_component", "利润变化", "#c18658"), ("asset_component", "资产分母", "#899ca9")]:
        values = cases[col].to_numpy() * 100
        left = np.where(values >= 0, positive, negative)
        axes[0].barh(y, values, left=left, label=label, color=color, height=.57)
        for i, value in enumerate(values):
            if abs(value) >= 1:
                axes[0].text(left[i] + value / 2, i, f"{value:+.2f}", ha="center", va="center", fontsize=10, color="white")
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    axes[0].set_yticks(y, ["小商品城 2025Q3", "牧原股份 2024H1", "上机数控 2022Q3"])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#a8a39a", lw=.8)
    axes[0].set(title="同一现金质量指标，改善来源不同", xlabel="指标同比变化的算术贡献（百分点）")
    axes[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=9)
    axes[0].set_ylim(2.65, -.65)
    components = [differences["consolidated_net_profit"], differences["inventory_decrease"], differences["operating_receivables_decrease"], differences["operating_payables_increase"], residual]
    labels = ["合并净利润", "存货调节", "经营性应收调节", "经营性应付调节", "其余调节项目"]
    values = np.array(components) / 1e8
    axes[1].barh(np.arange(5), values, color=["#187b73" if v >= 0 else "#c18658" for v in values], height=.57)
    for i, value in enumerate(values):
        axes[1].text(value + (1 if value >= 0 else -1), i, f"{value:+.2f}", va="center", ha="left" if value >= 0 else "right")
    axes[1].set_yticks(np.arange(5), labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#a8a39a", lw=.8)
    axes[1].set(title="牧原：上半年现金流增加165.18亿元", xlabel="2024H1相对2023H1的变化（亿元）")
    axes[1].margins(x=.2)
    for ax in axes:
        ax.set_facecolor(background)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c3c0b7")
        ax.grid(axis="x", alpha=.12)
    fig.suptitle("510300 历史因子研究｜从指标变化追到利润、现金与经营原因", x=.035, ha="left", fontsize=17)
    fig.text(.035, .025, "左图使用滚动四季与资产同比；右图使用半年度合并现金流调节。两种口径分别解释，不互相替代，也不代表股价因果贡献。", fontsize=10, color="#66635b")
    fig.tight_layout(rect=(0, .07, 1, .92))
    fig.savefig(OUT / "现金质量变化的来源.png", dpi=180, facecolor=background)
    plt.close(fig)


if __name__ == "__main__":
    main()
