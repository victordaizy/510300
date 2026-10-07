"""离线分析全国月度快照，分别输出原始事实、计算结果和解释边界。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "nbs_monthly"
CORE = [
    ("制造业PMI（数据库）", "a09aa989bdcf4cffa2021795722eb916", "扩散指数", 0),
    ("规上工业增加值", "ef1b1765960d45a29b4d7c4ca91be916", "当月实际同比", 0),
    ("规上工业增加值累计", "21e7072e9f384209aedb56e69a18216e", "累计实际同比", 0),
    ("服务业生产指数", "3fc5439f03ec46a5a76e8032036e8c17", "当月同比", 0),
    ("社会消费品零售总额", "aaac57d54d2e465d91bc9f3ea1a8618e", "当月名义同比", 0),
    ("固定资产投资（不含农户）", "7e570cf8071c4734a7d78d9f0a70fbe1", "累计名义同比", 0),
    ("房地产开发投资", "205e08cba8c2409980db58c98da91b6f", "累计名义同比", 0),
    ("新建商品房销售面积", "50a37fbef1d04be68f15d82b711783bf", "累计同比", 0),
    ("工业企业利润总额", "7f30595d047445e0a3880ab928795892", "累计可比同比", 0),
    ("工业企业营业收入", "84117bfe6f1b4a299d0db210deb5ad36", "累计可比同比", 0),
    ("CPI（2026年口径）", "53180dfb9c14411ba4b762307c85920c", "当月同比，原指数减100", 100),
    ("PPI", "150633e52b9a470a9a9fd1b296dd6c5b", "当月同比，原指数减100", 100),
    ("全国城镇调查失业率", "3888eac6062945a79c8a27e5f13d4953", "当月水平", 0),
    ("M2", "e03f2232631f41cd9d754a7d7feb4a81", "存量同比", 0),
    ("M1", "640401d3351b4b868dea28f89f410a54", "存量同比", 0),
    ("M0", "db7891fb8f3c4eb2a4d71a9955eba8c7", "存量同比", 0),
    ("货物进出口（美元口径）", "5143e29f77ee4d3489eaf46b901ba610", "当月名义同比", 0),
    ("货物出口（美元口径）", "788f44b0f310403fbd308b77d6f83890", "当月名义同比", 0),
    ("货物进口（美元口径）", "cc1ac699bc4f4cd7aa5c2b7a1e643259", "当月名义同比", 0),
    ("货运量", "3026a1f82c064d038062f42dba5409b0", "当月同比", 0),
    ("邮政行业业务收入", "b400e08c5eb1475594beba116f7c601c", "当月名义同比", 0),
    ("电信业务收入", "9e655be792d6479e99bb0ba6114bfdcb", "当月名义同比", 0),
    ("国家财政收入", "03e60611a6be446099bef411599bceb0", "累计名义同比", 0),
    ("原煤产量", "2f76fed8960547e0a69ab99917423fc0", "当月同比", 0),
    ("发电量", "5cc686f383d84ed9bd11f4293ab4170e", "当月同比", 0),
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalized(value):
    return re.sub(r"\s+", "", str(value)).replace("，", "、")


def number(value):
    candidate = pd.to_numeric(str(value).replace(",", ""), errors="coerce")
    return None if pd.isna(candidate) else float(candidate)


def sign(value):
    if value is None:
        return "不可比或缺失"
    return "增长" if value > 0 else "下降" if value < 0 else "持平"


def display(value, digits=1):
    return "未知" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def publication_month(title):
    match = re.search(r"(20\d{2})年(\d{1,2})月", normalized(title))
    return f"{match[1]}{int(match[2]):02d}" if match else None


def core_analysis(connection, cutoff):
    overview, details = [], []
    for label, iid, basis, subtract in CORE:
        definition = connection.execute("SELECT category,name,unit,period_basis FROM indicators WHERE indicator_id=?", [iid]).fetchone()
        if definition is None:
            raise RuntimeError(f"快照缺少指定指标：{label}。")
        data = pd.read_sql_query("SELECT period,value,raw_path,retrieved_at FROM observations WHERE indicator_id=? AND period<=? AND status='有值' ORDER BY period", connection, params=[iid, cutoff])
        if data.empty:
            raise RuntimeError(f"指定指标在截止月份前没有数值：{label}。")
        latest_period = pd.Period(data.period.iloc[-1], freq="M")
        window = pd.period_range(end=latest_period, periods=12, freq="M")
        indexed = data.set_index("period")
        values = {}
        for period in window:
            key = period.strftime("%Y%m")
            cell = indexed.loc[key] if key in indexed.index else None
            value = float(cell.value) - subtract if cell is not None else None
            values[key] = value
            details.append({"指标": label, "类别": definition[0], "指标ID": iid, "数据月份": key,
                            "口径": basis, "数值": value, "原始数值": float(cell.value) if cell is not None else None,
                            "计算": f"原值减{subtract}" if subtract else "原值",
                            "原始响应": cell.raw_path if cell is not None else None})
        last_three = [p.strftime("%Y%m") for p in window[-3:]]
        last, prior = values[last_three[2]], values[last_three[1]]
        # 年内累计区间跨年后不能作为连续累计增速变化处理。
        same_cumulative_year = "累计" not in basis or last_three[2][:4] == last_three[1][:4]
        change = last - prior if last is not None and prior is not None and same_cumulative_year else None
        overview.append({"指标": label, "类别": definition[0], "指标ID": iid, "口径": basis,
                         "前两个月份": last_three[0], "前两月数值": values[last_three[0]],
                         "前一月份": last_three[1], "前一月数值": prior,
                         "最新月份": last_three[2], "最新数值": last,
                         "较前一月数值差": change,
                         "差值单位": "指数点" if basis == "扩散指数" else "百分点",
                         "说明": "累计增速差不代表当月增速" if "累计" in basis else "同比增速差不代表环比增速" if "同比" in basis else "水平之差"})
    return pd.DataFrame(overview), pd.DataFrame(details)


def newest_release(releases, title_part):
    candidates = [r for r in releases if title_part in normalized(r["title"]) and r.get("published_at")]
    if not candidates:
        raise RuntimeError(f"没有带公布时间的官方原稿：{title_part}。")
    return max(candidates, key=lambda r: r["published_at"].replace("/", "-"))


def industry_analysis(connection, run, profit_release):
    title = normalized(profit_release["title"])
    match = re.search(r"(20\d{2})年1[—\-－](\d{1,2})月份", title)
    if not match:
        raise RuntimeError("工业利润原稿的累计区间无法确认。")
    period = f"{match[1]}{int(match[2]):02d}"
    tables = load(run / profit_release["table_path"])
    grid = next(t["grid"] for t in tables if normalized(t["grid"][0][0]) == "行业" and "利润总额" in "".join(t["grid"][0]))
    production = pd.read_sql_query("SELECT indicator_id,name FROM indicators WHERE catalogue_path LIKE '%工业分大类行业增加值增长速度 (2018-至今)%' AND period_basis='累计同比增减%'", connection)
    lookup = {normalized(row.name.split("增加值")[0]): row.indicator_id for row in production.itertuples(index=False)}
    monthly = pd.read_sql_query("SELECT indicator_id,name FROM indicators WHERE catalogue_path LIKE '%工业分大类行业增加值增长速度 (2018-至今)%' AND period_basis='同比增减%'", connection)
    monthly_ids = {normalized(row.name.split("增加值")[0]): row.indicator_id for row in monthly.itertuples(index=False)}
    bases = pd.read_sql_query("SELECT i.indicator_id,i.name,o.value,o.raw_path FROM indicators i JOIN observations o USING(indicator_id) WHERE i.catalogue_path='月度数据 / 工业 / 按行业分工业企业主要经济指标 (2018-至今) / 工业企业利润总额' AND i.period_basis='上年同期累计' AND o.period=? AND o.status='有值'", connection, params=[period])
    base_lookup = {normalized(row.name.split("利润总额")[0]): row for row in bases.itertuples(index=False)}
    rows = []
    for position, row in enumerate(grid):
        name = normalized(row[0])
        if name not in lookup:
            continue
        current = connection.execute("SELECT value,raw_path,status FROM observations WHERE indicator_id=? AND period=?", [lookup[name], period]).fetchone()
        current_month = connection.execute("SELECT value,status FROM observations WHERE indicator_id=? AND period=?", [monthly_ids[name], period]).fetchone()
        growth = float(current[0]) if current and current[2] == "有值" else None
        month_growth = float(current_month[0]) if current_month and current_month[1] == "有值" else None
        profit_growth = number(row[6])
        comparable_base = base_lookup.get(name)
        prior_profit = float(comparable_base.value) if comparable_base else None
        current_profit = number(row[5])
        rows.append({"行业": name, "累计截止月份": period, "增加值累计同比%": growth,
                     "增加值当月同比%": month_growth, "营业收入累计亿元": number(row[1]),
                     "营业收入累计同比%": number(row[2]), "营业成本累计亿元": number(row[3]),
                     "营业成本累计同比%": number(row[4]), "利润总额累计亿元": current_profit,
                     "利润上年同期可比基数亿元": prior_profit,
                     "利润净增额亿元": current_profit - prior_profit if current_profit is not None and prior_profit is not None else None,
                     "可比基数原始响应": comparable_base.raw_path if comparable_base else None,
                     "利润累计同比%": profit_growth, "利润增速原文": row[6],
                     "增加值方向": sign(growth), "利润方向": sign(profit_growth),
                     "生产指标ID": lookup[name], "生产原始响应": current[1] if current else None,
                     "利润官方来源": profit_release["url"], "利润原稿表序号": 2, "利润原稿行序号": position})
    if len(rows) != len(lookup) or len(rows) != 41:
        raise RuntimeError(f"工业大类映射不是完整41项：已匹配{len(rows)}，官方目录{len(lookup)}。")
    frame = pd.DataFrame(rows)
    production_counts = dict(Counter(frame["增加值方向"]))
    profit_counts = dict(Counter(frame["利润方向"]))
    joint = Counter(zip(frame["增加值方向"], frame["利润方向"]))
    distribution = {"累计截止月份": period, "行业总数": 41, "增加值累计方向": production_counts,
                    "利润累计方向": profit_counts,
                    "生产与利润联合分布": [{"增加值方向": p, "利润方向": r, "行业数": count} for (p, r), count in sorted(joint.items())],
                    "行业增加值增速中位数%": float(frame["增加值累计同比%"].median()),
                    "可比利润增速中位数%": float(frame["利润累计同比%"].median()),
                    "本期行业利润总额方向": dict(Counter("未知" if pd.isna(value) else "正值" if value > 0 else "负值" if value < 0 else "零" for value in frame["利润总额累计亿元"])),
                    "统计含义": "行业数量和中位数均不按行业经济规模加权，不能替代工业总体增速；注释项保留不可比。"}
    total = next(row for row in grid if normalized(row[0]) == "总计")
    total_profit, total_base = number(total[5]), float(base_lookup["工业企业"].value)
    missing_bases = frame.loc[frame["利润上年同期可比基数亿元"].isna(), "行业"].tolist()
    current_residual = total_profit - float(frame["利润总额累计亿元"].sum())
    prior_residual = total_base - float(frame["利润上年同期可比基数亿元"].sum()) if not missing_bases else None
    rounding_bound = (len(frame) + 1) * .05
    if abs(current_residual) > rounding_bound or (prior_residual is not None and abs(prior_residual) > rounding_bound):
        raise RuntimeError("行业利润与总量之差超出按一位小数逐项舍入的界限，暂停净增额占比计算。")
    net_increase = total_profit - total_base
    if net_increase == 0:
        raise RuntimeError("工业利润净增额为零，净增额占比没有定义。")
    frame["占全部工业利润净增额%"] = frame["利润净增额亿元"] / net_increase * 100
    selected_sectors = ["计算机、通信和其他电子设备制造业", "有色金属冶炼和压延加工业"]
    if frame.loc[frame["行业"].isin(selected_sectors), "利润净增额亿元"].isna().any():
        raise RuntimeError("用于单列净增额的两个行业缺少可比基数，不计算其净增额占比。")
    combined = float(frame.loc[frame["行业"].isin(selected_sectors), "利润净增额亿元"].sum())
    distribution["利润净增额分布"] = {"工业利润累计亿元": total_profit, "上年同期可比基数亿元": total_base,
        "工业利润净增额亿元": net_increase, "当期行业汇总与总量差亿元": current_residual,
        "可比基数行业汇总与总量差亿元": prior_residual, "电子与有色两行业净增额亿元": combined,
        "两行业净增额占工业净增额%": combined / net_increase * 100,
        "未提供上年同期可比基数的行业": missing_bases,
        "计算含义": "从当前快照提供的上年同期可比基数计算财务净增额，不用去年已公布旧值；占比是会计分布，不是GDP贡献或因果贡献。"}
    return frame, distribution


def pmi_analysis(run, release):
    period = publication_month(release["title"])
    if not period:
        raise RuntimeError("PMI原稿的数据月份无法确认。")
    tables = load(run / release["table_path"])
    window = {p.strftime("%Y%m") for p in pd.period_range(end=pd.Period(period, freq="M"), periods=3, freq="M")}
    result = []
    for index, labels in [(0, ["制造业PMI", "生产", "新订单", "原材料库存", "从业人员", "供应商配送时间"]),
                          (2, ["非制造业商务活动", "非制造业新订单", "非制造业投入品价格", "非制造业销售价格", "非制造业从业人员", "非制造业业务活动预期"])]:
        for row_index, row in enumerate(tables[index]["grid"]):
            match = re.fullmatch(r"(20\d{2})年(\d{1,2})月", normalized(row[0]))
            if not match:
                continue
            month = f"{match[1]}{int(match[2]):02d}"
            if month not in window:
                continue
            for column, label in enumerate(labels, 1):
                result.append({"指标": label, "数据月份": month, "数值": number(row[column]),
                               "单位": "指数点", "口径": "经季节调整的扩散指数", "官方来源": release["url"],
                               "原页公布时间": release["published_at"], "原稿表序号": index, "原稿行序号": row_index})
    frame = pd.DataFrame(result)
    if len(frame) != 36 or frame["数值"].isna().any():
        raise RuntimeError("PMI两张表的最近3个月没有完整对应36个数值。")
    return frame


def release_rows(run, release, pattern):
    tables = load(run / release["table_path"])
    return [row for row in tables[0]["grid"] if re.search(pattern, normalized(row[0]))]


def plot_industries(frame, output, period):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.8]}, facecolor="white")
    colors = {"增长": "#34627A", "下降": "#9AA4AE", "不可比或缺失": "#E5E9EC"}
    for y, field in enumerate(["增加值方向", "利润方向"]):
        left = 0
        counts = Counter(frame[field])
        for group in colors:
            count = counts.get(group, 0)
            if count:
                axes[0].barh(y, count, left=left, height=.42, color=colors[group], label=group if y == 1 else None)
                axes[0].text(left + count / 2, y, str(count), ha="center", va="center", color="white" if group == "增长" else "#283B45", fontsize=12)
                left += count
    axes[0].set_yticks([0, 1], ["增加值累计增速", "利润累计增速"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 41)
    axes[0].set_xlabel("行业数；全部41个行业，不按规模加权")
    axes[0].set_title("增长、下降和不可比的分布", loc="left", pad=16)
    handles, labels_for_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_for_legend, loc="lower left", bbox_to_anchor=(.13, .12), frameon=False, ncol=3, fontsize=9)
    valid = frame.dropna(subset=["增加值累计同比%", "利润累计同比%"])
    axes[1].scatter(valid["增加值累计同比%"], valid["利润累计同比%"], s=42, color="#34627A", alpha=.8, edgecolor="white", linewidth=.4)
    axes[1].axhline(0, color="#8B969D", linewidth=.9)
    axes[1].axvline(0, color="#8B969D", linewidth=.9)
    axes[1].set_xlabel("工业增加值累计同比（%）")
    axes[1].set_ylabel("利润累计同比（%）")
    axes[1].set_title("同期生产与利润；每个点代表一个行业", loc="left", pad=16)
    labels = {"汽车制造业": ("汽车", (9, -14)), "电气机械和器材制造业": ("电气机械", (9, 7)),
              "计算机、通信和其他电子设备制造业": ("计算机、通信和电子设备", (-12, -20)),
              "有色金属冶炼和压延加工业": ("有色金属冶炼", (9, 9))}
    for name, (label, offset) in labels.items():
        row = valid.loc[valid["行业"].eq(name)].iloc[0]
        axes[1].annotate(label, (row["增加值累计同比%"], row["利润累计同比%"]), xytext=offset,
                         textcoords="offset points", ha="right" if offset[0] < 0 else "left", fontsize=9)
    axes[1].grid(color="#EEF0F2", linewidth=.6)
    fig.suptitle(f"41个工业行业 · {period[:4]}年1—{int(period[4:])}月同期比较", x=.06, y=.98, ha="left", fontsize=19, fontweight="bold")
    missing = frame.loc[frame["利润累计同比%"].isna(), "行业"].tolist()
    fig.text(.06, .03, "散点保留所有可比行业；利润同比不可比的行业另列：" + "、".join(missing), fontsize=9, color="#596B76")
    fig.text(.06, .07, "正负表示同比增速方向；利润同比下降不等于行业亏损。生产为实际指标，利润为名义财务指标。", fontsize=9, color="#596B76")
    fig.subplots_adjust(left=.13, right=.97, top=.83, bottom=.25, wspace=.45)
    fig.savefig(output / "41行业同期生产与盈利.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="准确描述全国月度数据及41行业同期分布，不联网。")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.run_id and not re.fullmatch(r"[0-9A-Za-z_-]+", args.run_id):
        parser.error("快照名称格式不正确。")
    run = BASE / "runs" / args.run_id if args.run_id else Path(load(BASE / "LATEST.json")["path"])
    scope = load(run / "scope.json")
    releases = load(run / "releases_status.json")["releases"]
    output = BASE / "analyses" / (run.name + "_objective_v1")
    output.mkdir(parents=True, exist_ok=True)
    # 写出口径后再计算，所有行业和正负结果同时保留。
    save(output / "分析口径.json", {"数据快照": run.name, "采集业务日期": scope["as_of"],
         "宏观指标": [{"指标": label, "指标ID": iid, "口径": basis, "原值减数": subtract} for label, iid, basis, subtract in CORE],
         "比较方式": "各指标自己的最近3个月和12个月；相邻同比变化用百分点，PMI用指数点；累计区间跨年不计算连续变化。",
         "行业范围": "全部41个工业大类，生产、营业收入、利润均按同一1月至累计截止月比较。",
         "利润净增额": "本期累计利润减当前快照中的上年同期可比基数；检查41行业与总量的舍入差；固定列出电子和有色两行业的净增额占比。",
         "PMI分解": "按原稿公式：新订单30%、生产25%、从业人员20%、反向配送时间15%、原材料库存10%；保留公布数值舍入造成的残差。",
         "历史使用": "不将当前修订版本当成历史首次可得版本；不拼旧分类，不填补空值。",
         "判断规则": "只给出可重算事实与有限解释，不引用机构观点，不给出综合景气分数或由相关性直接推断因果。"})
    connection = sqlite3.connect(f"file:{(run / '国家统计局月度.sqlite').resolve().as_posix()}?mode=ro", uri=True)
    try:
        overview, details = core_analysis(connection, scope["end_request"])
        profit_release = newest_release(releases, "全国规模以上工业企业利润")
        industries, distribution = industry_analysis(connection, run, profit_release)
    finally:
        connection.close()
    pmi_release = newest_release(releases, "中国采购经理指数运行情况")
    pmi = pmi_analysis(run, pmi_release)
    retail_release = newest_release(releases, "社会消费品零售总额增长")
    retail_rows = release_rows(run, retail_release, "^社会消费品零售总额$|^其中：除汽车以外的消费品零售额$|^汽车类$")
    supplements = {"零售原稿": retail_release["url"], "零售原始行": retail_rows,
                   "工业利润原稿": profit_release["url"], "工业利润总量原始行": load(run / profit_release["table_path"])[0]["grid"][2],
                   "工业企业周转原始行": load(run / profit_release["table_path"])[1]["grid"][3]}
    save(output / "发布稿量化补充.json", supplements)
    overview.to_csv(output / "全国主要指标最近三个月.csv", index=False, encoding="utf-8-sig")
    details.to_csv(output / "全国主要指标近12个月.csv", index=False, encoding="utf-8-sig")
    industries.to_csv(output / "41行业同期生产收入利润.csv", index=False, encoding="utf-8-sig")
    pmi.to_csv(output / "PMI发布稿最近三个月.csv", index=False, encoding="utf-8-sig")
    save(output / "行业分布统计.json", distribution)
    plot_industries(industries, output, distribution["累计截止月份"])
    selected = overview.set_index("指标")
    index_pmi = pmi.pivot(index="数据月份", columns="指标", values="数值").sort_index()
    latest_pmi, prior_pmi = index_pmi.iloc[-1], index_pmi.iloc[-2]
    contributions = []
    for label, weight, direction in [("新订单", .30, 1), ("生产", .25, 1), ("从业人员", .20, 1),
                                     ("供应商配送时间", .15, -1), ("原材料库存", .10, 1)]:
        change = float(latest_pmi[label] - prior_pmi[label])
        contributions.append({"分项": label, "权重": weight, "原指数变化": change,
                              "方向系数": direction, "加权变化指数点": change * weight * direction})
    contribution_sum = sum(row["加权变化指数点"] for row in contributions)
    reported_change = float(latest_pmi["制造业PMI"] - prior_pmi["制造业PMI"])
    save(output / "PMI算术分解.json", {"最新月份": index_pmi.index[-1], "分项": contributions,
        "按已公布一位小数分项计算变化": contribution_sum, "PMI公布数值变化": reported_change,
        "舍入残差": reported_change - contribution_sum,
        "含义": "按官方计算公式作算术分解；分项及总指数已经舍入，合计不应强行调整为公布变化。"})
    currency = details.loc[details["指标"].isin(["M1", "M2"])].pivot(index="数据月份", columns="指标", values="数值").dropna().sort_index()
    gap_now = float(currency.iloc[-1]["M1"] - currency.iloc[-1]["M2"])
    gap_prior = float(currency.iloc[-2]["M1"] - currency.iloc[-2]["M2"])
    findings = {"业务日期": scope["as_of"], "宏观覆盖类别": sorted(overview["类别"].unique().tolist()),
                "核心指标数": len(overview), "行业统计": distribution,
                "PMI生产减新订单": {"最新月份": index_pmi.index[-1], "最新指数点差": float(latest_pmi["生产"]-latest_pmi["新订单"]),
                                      "前月指数点差": float(prior_pmi["生产"]-prior_pmi["新订单"])},
                "M1同比减M2同比": {"最新月份": currency.index[-1], "最新百分点差": gap_now,
                                      "前月百分点差": gap_prior, "变化百分点": gap_now-gap_prior}}
    save(output / "分析结果.json", findings)
    period = distribution["累计截止月份"]
    rows = ["# 国家统计局全国月度数据：客观分析", "",
            f"数据快照：{run.name}。采集业务日期：{scope['as_of']}。本报告只使用该快照中的数值、表格和统计口径，不引用官方文字解读或机构景气判断。",
            "", "## 分析方法", "",
            "宏观部分覆盖14类中的25项总量及主要指标，逐项展示最近3个月及原始口径；完整12个月数据另存。行业部分覆盖全部41个工业大类，增加值、营业收入、利润统一到同一累计区间。计数与中位数按行业数统计，不加经济规模权重。缺失、不可比和不同来源版本单列。",
            "", "## 宏观事实", "",
            "表中差值是原指标数值之差：同比及失业率以百分点表示，PMI以指数点表示。累计增速之差不是当月增速，同比增速之差不是环比增速。每行的月份单独列示。", "",
            "| 指标 | 口径 | 前两月 | 前一月 | 最新月 | 较前一月差值 |", "|---|---|---:|---:|---:|---:|"]
    for row in overview.to_dict("records"):
        rows.append(f"| {row['指标']} | {row['口径']} | {row['前两个月份']}：{display(row['前两月数值'])} | {row['前一月份']}：{display(row['前一月数值'])} | {row['最新月份']}：{display(row['最新数值'])} | {display(row['较前一月数值差'])} |")
    rows += ["", "### 数据可支持的描述", ""]
    for label in ["规上工业增加值", "服务业生产指数", "社会消费品零售总额", "固定资产投资（不含农户）", "房地产开发投资", "全国城镇调查失业率"]:
        row = selected.loc[label]
        rows.append(f"- {label}：{row['前一月份']}为{display(row['前一月数值'])}，{row['最新月份']}为{display(row['最新数值'])}，原指标变化{display(row['较前一月数值差'])}个百分点；口径为{row['口径']}。")
    rows += ["", "上述指标方向不一致。工业实际同比与零售名义同比的统计范围和价格处理不同，二者增速之差不能用来测量产需缺口。投资累计增速下降，能说明扩大累计区间后的可比同比读数下降；判断最新一个月投资量的变化应读取另行公布的当月或季调环比。",
             "", "## PMI：公布稿与数据库版本分别保留", "",
             f"最新原稿数据月份{index_pmi.index[-1]}，公布时间{pmi_release['published_at']}；官方来源：[{pmi_release['title']}]({pmi_release['url']})。数据库中PMI的最新月份仍单独列在前表。", "",
             "| 指标 | 前月 | 最新月 | 变化（指数点） |", "|---|---:|---:|---:|"]
    for label in ["制造业PMI", "生产", "新订单", "从业人员", "非制造业商务活动", "非制造业新订单"]:
        rows.append(f"| {label} | {display(prior_pmi[label])} | {display(latest_pmi[label])} | {display(latest_pmi[label]-prior_pmi[label])} |")
    rows += ["", f"生产指数减新订单指数，由{display(prior_pmi['生产']-prior_pmi['新订单'])}变为{display(latest_pmi['生产']-latest_pmi['新订单'])}点。这是两个扩散指数的算术差，不能换算成生产过剩数量。制造业PMI由五个分项加权构成，配送时间按逆指数处理；50为相应指标定义的临界点，指数水平不等于产出增长率，单月跨过50也不证明持续趋势。",
             "", f"按官方权重计算，生产分项对总指数变化的算术贡献为{contributions[1]['加权变化指数点']:.3f}点，新订单为{contributions[0]['加权变化指数点']:.3f}点，从业人员为{contributions[2]['加权变化指数点']:.3f}点。五项按公布数值计算合计{contribution_sum:.3f}点，总指数公布变化为{reported_change:.1f}点，舍入残差{reported_change-contribution_sum:.3f}点。数据支持总指数上升与订单、用工分项下降同时发生。完整权重和逆向处理见PMI算术分解.json。",
             "", "## 41行业：同期生产与盈利", "",
             f"累计区间：{period[:4]}年1—{int(period[4:])}月。利润来源：[{profit_release['title']}]({profit_release['url']})。生产来源为同一快照内2018年至今行业分类目录，逐项来源见CSV。", "",
             "| 同期增加值方向 | 同期利润方向 | 行业数 |", "|---|---|---:|"]
    for item in distribution["生产与利润联合分布"]:
        rows.append(f"| {item['增加值方向']} | {item['利润方向']} | {item['行业数']} |")
    rows += ["", f"累计增加值方向分布：{distribution['增加值累计方向']}；累计利润方向分布：{distribution['利润累计方向']}。可比行业利润增速中位数为{display(distribution['可比利润增速中位数%'])}%，工业总体利润增速为{display(selected.loc['工业企业利润总额','最新数值'])}%。中位数与总体增速的权重不同，不能互相替代。", "",
             f"本期行业利润总额正负分布：{distribution['本期行业利润总额方向']}。利润同比下降不等于利润总额为负；行业利润总额为正也不表示该行业每家企业均盈利。", "",
             "以下固定列出汽车、电气、电子、有色四个行业；完整41项和所有不可比项在CSV内保留。", "",
             "| 行业 | 增加值累计同比% | 收入累计同比% | 利润累计同比% |", "|---|---:|---:|---:|"]
    for label in ["汽车制造业", "电气机械和器材制造业", "计算机、通信和其他电子设备制造业", "有色金属冶炼和压延加工业"]:
        row = industries.loc[industries["行业"].eq(label)].iloc[0]
        rows.append(f"| {label} | {display(row['增加值累计同比%'])} | {display(row['营业收入累计同比%'])} | {display(row['利润累计同比%'])} |")
    net = distribution["利润净增额分布"]
    rows += ["", f"按当前快照的上年同期可比基数，工业利润净增额为{net['工业利润净增额亿元']:,.1f}亿元。电子与有色两个行业净增额合计{net['电子与有色两行业净增额亿元']:,.1f}亿元，相当于工业净增额的{net['两行业净增额占工业净增额%']:.1f}%。两个行业和工业总量的可比基数均直接提供。当期41行业利润之和与总计之差{net['当期行业汇总与总量差亿元']:.1f}亿元；未提供可比基数的行业为{net['未提供上年同期可比基数的行业']}，该行基数、净增额和占比均留空，全部41行业的可比基数加总核对未完成。净增额占比描述已知财务金额的分布，不是产出、GDP或因果贡献。"]
    rows += ["", "这些同期数据可以检验生产增长与利润增长是否同向；单凭这组数据不能把差异归因于价格、成本、产品组合、竞争或基数。工业增加值增速是实际增速，营业收入和利润金额是名义财务指标。两类资料均为行业汇总，尚未取得逐企业对应关系。",
             "", "## 消费、企业周转与货币", "",
             f"消费原稿：[{retail_release['title']}]({retail_release['url']})。原表各行按‘当月金额、当月同比、累计金额、累计同比’保留：", ""]
    for row in retail_rows:
        rows.append(f"- {normalized(row[0])}：当月{row[1]}亿元，同比{row[2]}%；累计{row[3]}亿元，同比{row[4]}%。")
    rows += ["", "总体零售、除汽车零售与汽车零售的数据能说明分项方向不同；零售总额不覆盖全部居民服务消费，不能直接等同于居民消费支出。分项贡献若计算，需使用可比的上年同期权重并处理范围调整和四舍五入，不能只用本年金额占比乘增速。",
             "", "工业原稿的总量周转表分别给出营业收入利润率、资产负债率、库存周转天数和应收回收期，原始行保存在发布稿量化补充.json。盈利率、回款速度和存货周转是不同维度，利润增长不能单独证明资金周转加快。",
             "", f"货币资料最新共同月份为{currency.index[-1]}。M1同比减M2同比由{display(gap_prior)}变为{display(gap_now)}个百分点，变化{display(gap_now-gap_prior)}个百分点；构成变化应分别看M1与M2原值。这个差值不能直接测量企业信用需求、资金价格或股票资金流。当前国家统计局金融目录没有银行间利率、社融结构或融资条件，金融市场流动性状态保持未判定。",
             "", "## 结论边界", "",
             "本报告描述已公布数据的水平、变化和行业分布。没有计算GDP贡献，没有建立经济周期综合分数，没有据此估计未来增长，也没有给出价格、政策或融资条件的因果贡献。短期同比变化同时受本期变化和去年基数影响，需要额外资料才能分解。",
             "", "更早历史保存在原始数据库，但本报告没有跨统计分类版本拼接后排序。货币口径、CPI分类基期、企业样本范围、网上消费范围调整等应按各项定义处理。所有月份、原始响应和计算字段可从对应CSV与JSON复算。"]
    (output / "客观数据分析.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    save(output / "来源与结果.json", {"生成时间": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
                                    "来源快照": str(run), "输出目录": str(output), "宏观指标数": len(overview),
                                    "宏观类别数": overview["类别"].nunique(), "行业数": len(industries),
                                    "PMI表单元格数": len(pmi), "源数据库访问模式": "只读"})
    print(json.dumps({"输出目录": str(output), "宏观指标": len(overview), "行业分布": distribution, "货币差值": findings["M1同比减M2同比"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
