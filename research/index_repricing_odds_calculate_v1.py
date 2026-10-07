"""计算同一估值序列的变化及报价条件下的费用边界，不拟合收益参数。"""

import ast
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from index_repricing_odds_sources_v1 import ROOT, OUT, now, save


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rounded(value, tick, direction):
    return (value / tick).to_integral_value(rounding=direction) * tick


def cost_example(capital, cost, quote):
    money = Decimal(str(capital))
    commission = Decimal(str(cost["commission"]))
    minimum = Decimal(str(cost["minimum"]))
    slip = Decimal(str(cost["slippage"]))
    tick = Decimal("0.001")
    bid, ask = Decimal(str(quote["bid"])), Decimal(str(quote["ask"]))
    mid, half_spread = (bid + ask) / 2, (ask - bid) / 2
    entry = rounded(ask * (1 + slip), tick, ROUND_CEILING)
    budget = money / 2
    shares = int(budget / entry / 100) * 100
    while shares * entry + max(minimum, shares * entry * commission) > budget:
        shares -= 100
    if shares <= 0:
        raise ValueError("费用算例预算不足一手。")
    paid = shares * entry + max(minimum, shares * entry * commission)

    def outcome(growth, multiple):
        gross_multiplier = (1 + Decimal(str(growth))) * (1 + Decimal(str(multiple)))
        future_bid = mid * gross_multiplier - half_spread
        exit_price = rounded(future_bid * (1 - slip), tick, ROUND_FLOOR)
        proceeds = shares * exit_price - max(minimum, shares * exit_price * commission)
        pnl = proceeds - paid
        return {"earnings_revision_pct": growth * 100, "multiple_change_pct": multiple * 100,
                "gross_price_change_pct": float((gross_multiplier - 1) * 100),
                "hypothetical_exit_price": float(exit_price), "pnl_cny": float(pnl),
                "net_return_on_paid_pct": float(pnl / paid * 100), "capital_change_pct": float(pnl / money * 100)}

    required_exit = rounded(max((paid + minimum) / shares, paid / (shares * (1 - commission))), tick, ROUND_CEILING)
    required_mid = required_exit / (1 - slip) + half_spread
    scenarios = [outcome(0, 0), outcome(.03, 0), outcome(.03, -.05), outcome(0, -.05), outcome(0, .05)]
    positive, negative = scenarios[1]["pnl_cny"], scenarios[3]["pnl_cny"]
    return {"capital_cny": capital, "cost_assumption": cost, "illustrative_budget_fraction": .5,
            "shares_for_cost_calculation_only": shares, "entry_price_with_assumed_slippage": float(entry),
            "paid_cny": float(paid), "break_even_future_mid": float(required_mid),
            "break_even_mid_change_pct": float((required_mid / mid - 1) * 100),
            "minimum_multiple_change_with_3pct_earnings_revision_pct": float((required_mid / mid / Decimal("1.03") - 1) * 100),
            "two_case_break_even_positive_probability_pct": -negative / (positive - negative) * 100,
            "two_case_probability_note": "只是假定+3%与-5%两个价格结果的费用后盈亏平衡概率；未估计真实概率，也未覆盖其他结果。",
            "scenarios": scenarios}


def main():
    if (OUT / "reviewed_facts.json").exists():
        render(read(OUT / "reviewed_facts.json")["cost_examples"])
        print("仅根据已保存的计算结果重绘图表；原事实、账户和来源未重跑。")
        return
    prices = {r["tradeDate"]: r for r in read(OUT / "sources/csi300_perf.json")["data"]}
    valuations = {str(r["日期Date"]): r for r in read(OUT / "sources/csi300_indicator.json")}
    start, end = "20260901", "20260928"
    price_ratio = prices[end]["close"] / prices[start]["close"]
    rows = []
    for key in ["市盈率1（总股本）P/E1", "市盈率2（计算用股本）P/E2"]:
        first, last = valuations[start][key], valuations[end][key]
        multiple_ratio = last / first
        rows.append({"series": key, "first": first, "last": last,
                     "multiple_change_pct": (multiple_ratio - 1) * 100,
                     "price_divided_by_multiple_proxy_change_pct": (price_ratio / multiple_ratio - 1) * 100,
                     "index_denominator_interpretation": "计算用股本口径下仍有盈利期、换样及维护变化；总股本口径不对应指数价格权重，不用于指数盈利分解。"})
    raw = read(OUT / "sources/etf_intraday.json")
    fields = raw.split('="', 1)[1].split('";', 1)[0].split(',')
    quote = {"symbol": "510300.SH", "name": fields[0], "date": fields[30], "time": fields[31],
             "previous_close": float(fields[2]), "last": float(fields[3]), "bid": float(fields[6]), "ask": float(fields[7]),
             "bid_quantity_vendor_units_unverified": fields[10], "ask_quantity_vendor_units_unverified": fields[20],
             "source_type": "新浪盘中供应商快照，非收盘，不保证能够成交。"}
    if not (0 < quote["bid"] <= quote["ask"]) or quote["date"] != "2026-09-29":
        raise SystemExit("报价时间或盘口不适合本轮费用例子。")
    tree = ast.parse((ROOT / "research/factor96_margin_repair_v1.py").read_text(encoding="utf-8-sig"))
    costs = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "COSTS" for t in node.targets))
    examples = [{"cost_name": name, **cost_example(capital, cost, quote)} for capital in [200000, 20000] for name, cost in costs.items()]
    facts = {"recorded_at": now(), "price_window": [start, end],
             "window_reason": "原8月31日起窗口保持；每日XLS不含8月31日，另外只在共同覆盖的9月1日至28日比较，不拼接月报与不同接口。",
             "original_window_full_decomposition": "NOT_COMPUTED",
             "index_price_first": prices[start]["close"], "index_price_last": prices[end]["close"],
             "index_price_change_pct": (price_ratio - 1) * 100, "valuation_rows": rows,
             "unresolved_scope": {"daily_xls_pe1_0928": 14.28, "daily_xls_pe2_0928": 16.29,
                                  "perf_peg_field_0928": prices[end]["peg"], "prior_factsheet_0831_rolling_pe": 14.65,
                                  "methodology_reconciled": False, "absolute_fair_value_or_equity_risk_premium_claim": False},
             "extra_valuation_date_without_price": sorted(set(valuations) - set(prices)),
             "quote": quote, "cost_source": "research/factor96_margin_repair_v1.py:COSTS",
             "cost_examples": examples,
             "example_assumptions": ["现有买卖报价基础上另加原费用假设的滑点，按0.001元向不利方向取整", "100份整手且含佣金的预算不超过资本50%", "假设退出时绝对价差仍0.001元；期间无分红、现金利息为0，折溢价与跟踪偏离不变", "同口径前瞻盈利修正和前瞻估值倍数相乘得到情景价格变化；不是将历史PE2视为前瞻PE", "并非实际账户仿真或建议仓位；没有估计五日ES、成交与退出路径"],
             "scenarios_are_forecasts": False, "fitted_parameters": 0, "new_accounts": 0, "new_strategy_return_tests": 0}
    save(OUT / "reviewed_facts.json", facts)
    render(examples)
    print(json.dumps({"同窗口价格变化": facts["index_price_change_pct"], "估值观察": rows, "费用例子": examples},ensure_ascii=False,indent=2))


def render(examples):
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    x = np.linspace(-5, 5, 301)
    for name, color in [("BASE", "#2378a0"), ("STRESS", "#b04b2f")]:
        item = next(r for r in examples if r["capital_cny"] == 200000 and r["cost_name"] == name)
        factor = 1 + item["break_even_mid_change_pct"] / 100
        y = (factor / (1 + x / 100) - 1) * 100
        ax.plot(x, y, label="基础费用" if name == "BASE" else "压力费用", color=color, linewidth=2.3)
    stress = next(r for r in examples if r["capital_cny"] == 200000 and r["cost_name"] == "STRESS")
    boundary = ((1 + stress["break_even_mid_change_pct"] / 100) / (1 + x / 100) - 1) * 100
    ax.fill_between(x, boundary, 6, alpha=.10, color="#27825a")
    ax.fill_between(x, -6, boundary, alpha=.08, color="#a23c38")
    ax.scatter([3,3], [0,-5], color=["#23784d", "#a5332b"], s=70, zorder=5)
    ax.annotate("盈利修正+3%，倍数不变", (3,0), xytext=(-190,22), textcoords="offset points", fontproperties=font, fontsize=10)
    ax.annotate("盈利修正+3%，倍数下降5%", (3,-5), xytext=(-190,18), textcoords="offset points", fontproperties=font, fontsize=10)
    ax.set(xlim=(-5,5), ylim=(-6,6))
    ax.set_xlabel("每份预期盈利修正（情景，%）", fontproperties=font)
    ax.set_ylabel("前瞻估值倍数变化（情景，%）", fontproperties=font)
    ax.set_title("经营改善需要抵过估值压缩和交易成本", fontproperties=font, fontsize=16, pad=15)
    ax.axhline(0, color="#777777", linewidth=.7); ax.axvline(0, color="#777777", linewidth=.7)
    ax.grid(alpha=.18); ax.legend(prop=font, loc="upper right")
    fig.text(.07,.025,"曲线上方为费用后盈利的条件区域；不是概率分布或交易信号。\n例：20万元资本、含买入费用的投入上限10万元；9月29日14:19:30报价，期间无分红。",fontproperties=font,fontsize=10,color="#555555")
    fig.subplots_adjust(left=.12,right=.96,bottom=.18,top=.87)
    fig.savefig(OUT / "盈利修正与估值压缩的费用边界.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
