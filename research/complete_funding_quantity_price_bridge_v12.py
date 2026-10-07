"""复核原始定盘、观察时钟与对照窗口，并输出融资价格机制报告。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_funding_quantity_price_bridge_v12"
BASE = "C:/Users/戴周阳/Documents/New project 8/"
REPORT = "第十二轮_货币数量与融资价格为何可能背离.md"
FIGURE = "figures/融资成本上升与定盘差收窄并存.png"


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def link(relative, title):
    return f"[{title}](<{BASE}{relative}>)"


def verify():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for f in frozen["files"]:
        assert digest(OUT / "inputs" / f["name"]) == f["sha256"]
    fixing = pd.read_parquet(OUT / "inputs/fixing.parquet").sort_values("date")
    fixing["available_at"] = pd.to_datetime(fixing.available_at, utc=True)
    policy = pd.read_parquet(OUT / "inputs/policy.parquet").sort_values("published_at")
    policy["known"] = pd.to_datetime(policy.published_at).dt.tz_localize("Asia/Shanghai")
    (OUT / "sources/original_fixings").mkdir(exist_ok=True)
    receipts, original_rows = [], {}
    for path, rows in fixing.groupby("raw_path"):
        original = ROOT / path
        assert len(rows.raw_sha256.unique()) == 1 and digest(original) == rows.raw_sha256.iloc[0]
        body = json.loads(original.read_text(encoding="utf-8"))
        records = {r["lfiProducDate"]: r["frValueMap"] for r in body["records"]}
        for r in rows.itertuples():
            date = pd.Timestamp(r.date).date().isoformat()
            item = records[date]
            np.testing.assert_allclose([r.fdr007_percent, r.fr007_percent], [float(item["FDR007"]), float(item["FR007"])], rtol=0, atol=1e-12)
            p = policy[policy.known.le(r.available_at)].iloc[-1]
            original_rows[date] = (r.fdr007_percent, r.fr007_percent, p.seven_day_rate_percent)
        destination = OUT / "sources/original_fixings" / original.name
        shutil.copy2(original, destination)
        receipts.append({"source": path, "saved": destination.relative_to(OUT).as_posix(), "sha256": digest(destination)})
    save("original_fixing_receipts.json", receipts)
    result = pd.read_csv(OUT / "results/104个月_货币数量与融资价格.csv")
    detail = pd.read_csv(OUT / "results/资金窗口明细.csv")
    parent = pd.read_csv(OUT / "inputs/monthly.csv")
    assert len(result) == 104 and len(detail) == 104 * 40
    bond = pd.read_parquet(OUT / "inputs/bond.parquet").sort_values("available_at")
    bond["available_at"] = pd.to_datetime(bond.available_at, utc=True)
    checked_bonds = 0
    for row in result.to_dict("records"):
        snapshot = pd.Timestamp(row["snapshot_at"])
        selected = fixing[fixing.available_at.le(snapshot)].tail(40)
        actual_dates = detail[detail.stat_month.eq(row["stat_month"])].date.tolist()
        expected_dates = [pd.Timestamp(d).date().isoformat() for d in selected.date]
        assert expected_dates == actual_dates
        values = np.asarray([original_rows[d] for d in expected_dates])
        features = {"fdr007_percent": values[:, 0], "fr007_percent": values[:, 1], "seven_day_rate_percent": values[:, 2], "fdr_policy_gap_bp": 100 * (values[:, 0] - values[:, 2]), "fr_fdr_gap_bp": 100 * (values[:, 1] - values[:, 0])}
        for name, sequence in features.items():
            expected = [sequence[-1], sequence[20:].mean(), sequence[:20].mean(), sequence[20:].mean() - sequence[:20].mean()]
            actual = [row[name + suffix] for suffix in ["_latest", "_recent20_mean", "_previous20_mean", "_mean_change"]]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
        b = bond[bond.available_at.le(snapshot)]
        if row["bond_status"] == "AVAILABLE_DIAGNOSTIC_ONLY":
            assert row["bond_date"] == pd.Timestamp(b.iloc[-1].observation_date).date().isoformat()
            np.testing.assert_allclose([row["bond_10y_percent"], row["bond_change20_bp"]], [b.iloc[-1].china_10y_yield, 100 * (b.iloc[-1].china_10y_yield - b.iloc[-21].china_10y_yield)], atol=1e-9)
            checked_bonds += 1
    for name in ["E0_20_return", "E1_20_return", "past_return60", "spread_pp"]:
        np.testing.assert_allclose(result[name], parent[name], equal_nan=True)
    # 金融统计公告中月度资金价格单独提取，不能拼入日度定盘。
    rate_facts = []
    for month in ["2020-12", "2021-01"]:
        path = OUT / "sources" / (month + "_金融统计原文.html")
        text = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)
        start = text.index("四、")
        section = text[start:text.index("五、", start)]
        for field, regex in [("同业拆借月加权平均", r"同业拆借(?:月)?加权平均利率为\s*([\d.]+)%"), ("质押式回购月加权平均", r"(?:质押式债券回购|质押式回购)(?:月)?加权平均利率为\s*([\d.]+)%")]:
            found = re.findall(regex, section)
            assert found and len(set(found)) == 1, (month, field, found)
            rate_facts.append({"month": month, "field": field, "percent": float(found[0]), "source_sha256": digest(path)})
    pd.DataFrame(rate_facts).to_csv(OUT / "results/公告月度资金价格_独立口径.csv", index=False, encoding="utf-8-sig")
    pages = json.loads((OUT / "sources/政策报告逐页文本.json").read_text(encoding="utf-8"))
    table = pages[10]["text"].replace(" ", "")
    assert "4.61" in table and "-0.02" in table and "-0.51" in table
    assert "不应过度关注公开市场" in pages[17]["text"].replace(" ", "")
    assert "DR007" in pages[18]["text"]
    verification = {"at": now(), "status": "PASS_RAW_FIXINGS_AND_SAVED_CONTEXT", "raw_fixings_checked": len(fixing), "monthly_windows": 104, "window_rows": len(detail), "bond_contexts": checked_bonds, "source_pdfs_visually_checked_pages": [11, 18, 19], "new_models": 0, "new_accounts": 0, "independent_validation": False, "first_vintage_authenticated": False}
    save("verification.json", verification)
    return result, detail, verification


def figure(detail, january):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 9))
    fig.set_facecolor("#faf9f5")
    grid = fig.add_gridspec(2, 2, left=.075, right=.97, top=.84, bottom=.15, height_ratios=[1.25, 1], hspace=.55, wspace=.28)
    ax = fig.add_subplot(grid[0, :])
    sample = detail[detail.stat_month.eq("2021-01")].copy()
    sample["date"] = pd.to_datetime(sample.date)
    navy, orange = "#287487", "#c37442"
    fig.text(.075, .948, "2021年2月9日：利差收窄，融资却未必更便宜", fontsize=21, weight="bold", color="#183a47")
    fig.text(.075, .902, "M1公告观察点之前的40条定盘记录；对照两个不重叠的20条窗口。", fontsize=12, color="#53626b")
    for field, label, color, style in [("fdr007_percent", "银银间定盘 FDR007", navy, "-"), ("fr007_percent", "银行间定盘 FR007", orange, "-"), ("policy_rate_percent", "7天逆回购政策利率", "#6d7478", "--")]:
        ax.plot(sample.date, sample[field], color=color, label=label, linestyle=style, linewidth=1.8)
    ax.axvline(pd.Timestamp("2021-01-14"), color="#899399", linestyle=":")
    ax.set(ylabel="年利率（%）", ylim=(1.35, 3.55))
    ax.set_title("A  政策利率未变，市场定盘利率已上移", loc="left", fontsize=14, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=3)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    labels = ["前20条\n12月16日—1月13日", "近20条\n1月14日—2月9日"]
    for pos, field, title in [(grid[1, 0], "fdr_policy_gap_bp", "B  银银间定盘相对政策利率的平均差"), (grid[1, 1], "fr_fdr_gap_bp", "C  两类定盘之间的平均差")]:
        panel = fig.add_subplot(pos)
        vals = [january[field + "_previous20_mean"], january[field + "_recent20_mean"]]
        bars = panel.bar([0, 1], vals, color=[navy, orange], width=.52)
        panel.bar_label(bars, labels=[f"{v:+.2f} bp" for v in vals], padding=7, fontsize=12)
        panel.axhline(0, color="#7f898e", linewidth=.8)
        panel.set(xticks=[0, 1], xticklabels=labels, ylabel="基点（bp）", ylim=(-35, 40) if field == "fdr_policy_gap_bp" else (0, 25))
        panel.set_title(title, loc="left", fontsize=13, pad=12)
    for panel in fig.axes:
        panel.set_facecolor("#faf9f5")
        panel.grid(axis="y", color="#deded8", alpha=.75, linewidth=.7)
        panel.set_axisbelow(True)
    fig.text(.075, .079, "FDR007和FR007是上午成交定盘，并非全天加权DR007和R007；两类定盘差也不等于纯非银融资溢价。", fontsize=10, color="#53626b")
    fig.text(.075, .045, "来源：中国货币网保存的原始定盘响应、央行政策利率公告。描述既有融资条件，不单独识别股价因果或买卖方向。", fontsize=10, color="#53626b")
    fig.savefig(OUT / FIGURE, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def publish(result, detail, verification):
    case = result.set_index("stat_month").loc["2021-01"]
    figure(detail, case)
    cases = result[result.stat_month.isin(["2020-06", "2021-01", "2024-08", "2025-07", "2025-08"])]
    lines = []
    for r in cases.to_dict("records"):
        lines.append(f"| {r['stat_month']} / {str(r['snapshot_at'])[:10]} | {r['fdr_policy_gap_bp_mean_change']:+.2f} | {r['fr_fdr_gap_bp_mean_change']:+.2f} | {r['bond_change20_bp']:+.2f} | {r['past_return60'] * 100:+.2f}% | {r['E0_20_return'] * 100:+.2f}% |")
    relative = OUT.relative_to(ROOT).as_posix()
    pdf = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/4021036/8d39aa9d730046c896289d17c09346d2/2021020821282167078.pdf"
    report = f"""**本轮的新结论：数量增加、贷款平均利率下降、机构之间利差收窄，是三件不同的事。2021年1月病例里，此前企业贷款平均利率降低与随后市场资金价格上升同时存在，不能用一个“宽松”标签合并。**

上一轮已经确认春节跨月、存款持有人和信贷期限结构会改变M1的含义。本轮继续连接货币数量、实体贷款成本、金融机构融资成本、长期利率与股票既有定价。沿用原104个月和五个既有病例，没有按本轮结果挑案例，没有新建预测模型或账户。

**2021年2月9日能看到的证据，既有改善，也有抵消。**

2021年2月8日22:20:39，央行发布2020年第四季度货币政策执行报告，先于本例2月9日21点观察。报告PDF第11页的表3显示：2020年12月企业贷款加权平均利率4.61%，较9月下降0.02个百分点，较上年同期下降0.51个百分点。较9月只降2bp与同比降51bp是不同参照，不能合并描述为最近融资成本快速下降。报告还显示2020年末制造业中长期贷款余额同比增长35.2%，证明此前确有结构性信贷支持，不能把所有改善都解释成统计假象。[央行报告原文]({pdf})。

这里的贷款利率统计期是2020年12月，制造业贷款是2020年末余额；不能把它们改称2021年1月的企业借款成本和资金用途。加权平均利率也没有固定借款人和贷款结构，不能推定每家公司的融资成本下降了相同幅度。报告当前下载版没有取得历史首次发布版本认证。

本轮重新连接的日度资金价格却显示，到了2月9日：

| 当时可见的不同层面 | 数值 | 可以说明什么 |
|---|---:|---|
| 7天逆回购政策利率 | 2.20% | 该比较期内政策利率未变 |
| 银银间上午定盘FDR007 | 2.45% | 高于同期政策利率25bp |
| 前20条FDR007减政策利率的平均差 | −23.30bp | 2020-12-16至2021-01-13 |
| 近20条相同口径平均差 | +25.69bp | 2021-01-14至2021-02-09；较前窗上移48.99bp |
| FR007减FDR007平均差 | 17.73bp降至7.82bp | 两类定盘差收窄9.90bp，但共同融资价格上移 |
| 最新按原可见时钟纳入的10年国债收益率 | 3.2370% | 2月8日值，较20条之前上升9.13bp |

**利差收窄不能自动解释为融资成本下降。**本例FR007−FDR007变小，同时FDR007相对政策利率明显上升。二者计算样本不同，这个差额不是纯非银融资溢价，更不能独立识别哪类机构受到何种限制。

FDR007、FR007是中国货币网上午9:00—11:30成交形成的定盘利率。央行报告强调的DR007则是银银间加权平均利率；本轮没有把定盘值改名为DR007，也没有把它与报道中的全天加权值拼接。历史档案使用当日12点的计划可见时点，尚未认证每一天的实际首次送达。[中国货币网方法说明](https://www.chinamoney.com.cn/chinese/bkfrr/)。

与日度定盘分开看，央行金融统计公告里的银行间质押式回购月加权利率从2020年12月1.36%升至2021年1月2.07%，提高71bp。这提供了不同口径的月度背景，不能把71bp与上述48.99bp相加，更不能把全市场月度回购均值当作FDR007。[央行2020年12月公告](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/f67a498ba4df43bdbc4cc3053c6bc168/index.html)、[2021年1月公告](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/de29861e7a2848368ce269dfc31e6be6/index.html)。

**把不同传导渠道接回510300。**

| 传导渠道 | 本例已有证据 | 仍需避免的推断 |
|---|---|---|
| 实体融资 | 此前企业贷款利率较上年降低，制造业中长期贷款扩张 | 不能据此推定2021年1月所有公司的新增项目、订单和利润同步改善 |
| 支付与货币结构 | 1月M1同比高增，公布余额环比基本持平，存在春节跨月与持有人变化 | 不能直接当成经营需求突然上升 |
| 金融机构融资 | 当时FDR007相对政策利率的平均差上升 | 不能因信贷多增就推定金融市场资金同步更便宜 |
| 长期贴现环境 | 当时可见的10年国债收益率上升 | 它只是贴现环境的一部分，不能单独确定股票估值方向 |
| 已有定价和内部风险 | 510300此前60日上涨15.95%，仅138/300只成分下行能量下降 | 不能把指数降波当作未被定价的普遍改善 |

政策报告PDF第18—19页也区分了操作数量、政策利率和市场利率：操作数量会响应现金、财政与市场需求等因素，不直接代表利率方向。它建议观察主要市场利率及一段时期的平均水平。这里用该说明界定经济机制，没有声称20条定盘均值是官方择时规则。

企业贷款利率与FDR007也不能直接相减充当银行净息差：银行资产、负债的期限、存量重定价、存款成本及其他来源均未被这个差值覆盖。这一轮只确认传导渠道可能相反，没有据此计算银行盈利损失。

本例原E0观察口径后续20日含分红毛收益为−11.09%，E1为−13.96%。这些是已经看过的历史结果，不是本轮新样本；它们不证明融资价格上升单独导致下跌。能够确认的是：当时的证据并非“真实经营改善、金融资金便宜、价格尚未反映、内部风险下降”四条同时成立。

**五个原有病例并列后，资金价格也没有给出一个单向答案。**

| 统计月 / 观察日 | FDR相对政策利率的20条均值变化，bp | FR−FDR的20条均值变化，bp | 10年国债20条间隔变化，bp | 此前60日510300收益 | 原E0后续20日毛收益 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

资金列比较两个不重叠的20条银行间定盘记录，国债列比较当时可见最新值与20条之前值；两列虽然都以bp计，观察方式不同。1bp=0.01个百分点。股票收益沿用原含分红固定股数口径，尚未扣除交易成本。

2020年6月与2021年1月病例都有资金价格相对上移、长期利率上升、股价此前已较大上涨的背景。这是相似机制的历史对照，不能据两个已知案例设卖出规则。2025年7、8月病例则是FDR相对政策利率的平均差小幅回落，而长期国债收益率上升；不能把短端改善直接推广到所有贴现条件。2024年8月病例的后续窗口跨过9月24日新政策，整段收益不能只归给9月13日已有的资金背景。

这些不是新增的牛熊分组或综合评分。原104个月的融资价格均已连接，103个月有符合原时钟和时效限制的国债背景；2026年8月公告对应的原行情仍超出截止日，新增资金背景不会把缺失的股票观察变成可用收益。全表保留，未筛选收益较好的资金区间。

**对当前研究的实际约束：以后解释剪刀差与降波，至少要同时问钱由谁创造和持有、谁的融资成本改变、哪些经营结果被支持、股票已经反映多少。**“数量好转但金融融资成本上升”和“短端缓和但长端利率上升”都是应保留的混合情形，不能为了方向明确而抹去其中一侧。

本轮完成的是机制与时序的可复查进展。新的独立方向优势及成本后账户收益仍未证明；原先冻结失败的资金策略不因此重启或改参。

{link(relative + '/results/104个月_货币数量与融资价格.csv', '104个月融资背景底表')} · {link(relative + '/results/资金窗口明细.csv', '逐条资金窗口')} · {link(relative + '/sources/2020年第四季度货币政策执行报告.pdf', '本地央行原始报告')} · {link(relative + '/' + FIGURE, '融资价格对照图')}
"""
    (OUT / REPORT).write_text(report, encoding="utf-8")
    save("result.json", {"at": now(), "study_id": "510300_FUNDING_QUANTITY_PRICE_BRIDGE_V12", "status": "CONTEXT_VERIFIED_VISUAL_REVIEW_PENDING", "continuation_classification": "PROGRESS", "main_conclusion": "真实信贷支持与市场资金价格上升可以并存；机构间定盘差收窄也可能发生在共同融资成本上升时。", "january2021": {"fdr_policy_gap_recent20_bp": float(case.fdr_policy_gap_bp_recent20_mean), "fdr_policy_gap_previous20_bp": float(case.fdr_policy_gap_bp_previous20_mean), "fdr_policy_gap_change_bp": float(case.fdr_policy_gap_bp_mean_change), "fr_fdr_gap_change_bp": float(case.fr_fdr_gap_bp_mean_change), "bond_20_change_bp": float(case.bond_change20_bp)}, "monthly_rows": 104, "verification_status": verification["status"], "report": relative + "/" + REPORT, "report_sha256": digest(OUT / REPORT), "figure": relative + "/" + FIGURE, "figure_sha256": digest(OUT / FIGURE), "visual_review_pending": True, "new_models": 0, "new_accounts": 0, "goal_achieved": False, "independent_validation": False, "goal_status": "active"})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps({"复核": verification, "报告": str(OUT / REPORT), "图": str(OUT / FIGURE)}, ensure_ascii=False))


if __name__ == "__main__":
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("已完成的研究不覆盖。")
    publish(*verify())
