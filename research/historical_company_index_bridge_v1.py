"""历史公司经营改善到指数覆盖、价格表现的有限诊断。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_company_index_bridge_v1"
sys.path.insert(0, str(ROOT / "research"))
from historical_price_gap_causes_v1 import round_trip

WINDOWS = {"pre20": (-21, -1), "response": (-1, 0), "post20": (0, 20)}
CASES = {"1220788951": "牧原股份", "1224710669": "小商品城", "1214728625": "上机数控"}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def describe(values):
    values = pd.Series(values, dtype=float).dropna()
    return {"n": len(values), "mean": values.mean(), "median": values.median(),
            "positive_count": int(values.gt(0).sum()), "minimum": values.min(),
            "maximum": values.max()}


def pct(value, digits=2):
    return "缺失" if pd.isna(value) else f"{value * 100:+.{digits}f}%"


def fraction_pct(value, digits=3):
    return "缺失" if pd.isna(value) else f"{value * 100:.{digits}f}%"


def plot(events):
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False,
                         "font.size": 10.5})
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.patch.set_facecolor("#f5f3ed")
    positions = np.arange(len(events))
    axes[0].barh(positions, events.reference_weight_sum * 100, color="#abc8ca",
                 label="当日财报篮子全部公司")
    axes[0].barh(positions, events.joint_positive_reference_weight_sum * 100,
                 color="#187e7a", label="其中公司自身同时通过原三条件")
    axes[0].set_yticks(positions, events.date.dt.strftime("%Y-%m-%d"), fontsize=8.7)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="占沪深300的参考权重（%）", title="财报篮子覆盖多少指数")
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    x = events.cohort_equal_mean_stock_post20 * 100
    y = events.etf_post20 * 100
    axes[1].scatter(x, y, s=30 + events.reference_weight_sum * 12000,
                    color="#187e7a", alpha=.72, edgecolors="white", linewidths=.7)
    axes[1].axhline(0, color="#aaa79f", lw=.8)
    axes[1].axvline(0, color="#aaa79f", lw=.8)
    for day in ["2022-10-11", "2024-08-05", "2025-10-16"]:
        row = events[events.date.eq(pd.Timestamp(day))].iloc[0]
        axes[1].annotate(day, (row.cohort_equal_mean_stock_post20 * 100,
                              row.etf_post20 * 100), xytext=(6, 7),
                         textcoords="offset points", fontsize=9)
    axes[1].set(xlabel="财报公司等权平均复权收益（%）", ylabel="同期510300含息收益（%）",
                title="同一20日窗口：公司表现与ETF表现")
    axes[1].margins(.20)
    for ax in axes:
        ax.set_facecolor("#fffdf8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#d8d5cd", alpha=.4, linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("经营改善的来源已找到，指数覆盖和剩余收益仍须分别看", fontsize=16, y=.985)
    fig.text(.04, .018, "全部23个日期均保留；权重来自严格早于测量日的月末参考快照。圆点大小表示覆盖权重。\n"
             "20日窗口为测量日收盘至其后第20个交易日收盘；重叠事件不独立，图中收益未扣交易费。",
             fontsize=9, color="#5f625e")
    fig.tight_layout(rect=[.02, .08, .99, .95])
    fig.savefig(OUT / "经营改善到指数覆盖.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    cfg = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    inputs = {k: ROOT / v for k, v in cfg["inputs"].items()}
    sample = pd.read_parquet(inputs["companies"]).copy()
    sample["announcement_id"] = sample.announcement_id.astype(str)
    names = pd.read_parquet(inputs["names"], columns=["announcement_id", "sec_name"])
    names["announcement_id"] = names.announcement_id.astype(str)
    names = names[names.announcement_id.isin(sample.announcement_id)]
    sample = sample.merge(names.drop_duplicates(), on="announcement_id", how="left", validate="one_to_one")
    if len(sample) != 49 or sample.date.nunique() != 23:
        raise ValueError("继承的49条、23个日期样本发生变化。")

    market = pd.read_parquet(inputs["etf_prices"]).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date)
    calendar = pd.DatetimeIndex(market.date)
    if calendar.has_duplicates:
        raise ValueError("ETF交易日期重复。")
    price_frame = pd.read_parquet(inputs["stock_prices"])
    price_frame = price_frame[price_frame.con_code.isin(sample.ts_code)]
    stock = price_frame.set_index(["date", "con_code"]).total_return_close
    if stock.index.has_duplicates:
        raise ValueError("股票同日价格重复。")
    weights = pd.read_parquet(inputs["weights"])
    weights["trade_date"] = pd.to_datetime(weights.trade_date)
    weight_dates = pd.DatetimeIndex(sorted(weights.trade_date.unique()))
    members = pd.read_parquet(inputs["membership"])
    membership = set(zip(pd.to_datetime(members.membership_date), members.symbol))
    dividends = pd.read_csv(inputs["dividends"])
    dividends = dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)

    companies, etf_events = [], {}
    for source in sample.sort_values(["date", "ts_code"]).to_dict("records"):
        day, code = pd.Timestamp(source["date"]), source["ts_code"]
        position = calendar.get_loc(day)
        if position < 21 or position + 21 >= len(market):
            raise ValueError("预先指定的价格窗口超出本地范围。")
        weight_position = weight_dates.searchsorted(day, side="left") - 1
        if weight_position < 0:
            weight_date, weight_age = pd.NaT, np.nan
            reference_weight = np.nan
        else:
            weight_date = weight_dates[weight_position]
            weight_age = (day - weight_date).days
            snapshot = weights[weights.trade_date.eq(weight_date)].set_index("con_code").weight
            reference_weight = float(snapshot.get(code, np.nan)) / 100 if weight_age <= 62 else np.nan
        row = {
            "date": day, "announcement_id": source["announcement_id"], "ts_code": code,
            "sec_name": source["sec_name"], "report_period": source["report_period"],
            "event_publication_date": source["event_publication_date"],
            "industry_l1_code": source["industry_l1_code"],
            "company_joint_positive": bool(source["company_joint_positive"]),
            "weight_snapshot_date": weight_date, "weight_age_calendar_days": weight_age,
            "member_on_measurement_day": (day, code) in membership,
            "reference_weight": reference_weight,
            "weight_status": "UNVERSIONED_REFERENCE_ONLY" if np.isfinite(reference_weight) else "MISSING",
        }
        if not row["member_on_measurement_day"]:
            row["reference_weight"] = np.nan
            row["weight_status"] = "NOT_MEMBER_ON_MEASUREMENT_DAY"
        for name, (start, end) in WINDOWS.items():
            left, right = calendar[position + start], calendar[position + end]
            stock_start = stock.get((left, code), np.nan)
            stock_end = stock.get((right, code), np.nan)
            row[f"{name}_start_date"], row[f"{name}_end_date"] = left, right
            row[f"stock_{name}_start_value"] = stock_start
            row[f"stock_{name}_end_value"] = stock_end
            row[f"stock_{name}"] = stock_end / stock_start - 1 if stock_start > 0 and stock_end > 0 else np.nan
            row[f"etf_{name}"] = market.iloc[position + end].wealth / market.iloc[position + start].wealth - 1
            row[f"stock_minus_etf_{name}"] = row[f"stock_{name}"] - row[f"etf_{name}"]
            row[f"reference_contribution_{name}"] = row["reference_weight"] * row[f"stock_{name}"]
        companies.append(row)

        if day not in etf_events:
            entry, exit_row = market.iloc[position + 1], market.iloc[position + 21]
            eligible = dividends.record_date.ge(entry.date) & dividends.record_date.lt(exit_row.date)
            div = float(dividends.loc[eligible, "cash_dividend_per_share"].sum())
            trade = round_trip(entry.open, exit_row.open, div, cfg["costs"])
            etf_events[day] = {
                "entry_date": entry.date, "exit_date": exit_row.date,
                "entry_open": entry.open, "exit_open": exit_row.open,
                "dividend_per_share": div,
                "etf_event_gross_return": (exit_row.open + div) / entry.open - 1,
                "etf_event_net_return": trade["net_return"],
                "illustrative_pnl_cny": trade["net_pnl_cny"],
                "illustrative_paid_cny": trade["paid_cny"],
                "illustrative_shares": trade["shares"],
                "illustrative_commissions_cny": trade["commissions_cny"],
                "buy_fill_reference": trade["buy_price"], "sell_fill_reference": trade["sell_price"],
                "etf_event_status": "SEPARATE_EVENT_EXAMPLE_NOT_FULL_ACCOUNT",
            }

    companies = pd.DataFrame(companies)
    events = []
    for day, group in companies.groupby("date", sort=True):
        passed = group[group.company_joint_positive]
        complete_weights = group.reference_weight.notna().all()
        event = {
            "date": day, "company_count": len(group), "joint_positive_company_count": len(passed),
            "company_names": "、".join(group.sec_name.astype(str)),
            "weight_snapshot_date": group.weight_snapshot_date.iloc[0],
            "weight_age_calendar_days": group.weight_age_calendar_days.max(),
            "missing_weight_count": int(group.reference_weight.isna().sum()),
            "reference_weight_sum": group.reference_weight.sum() if complete_weights else np.nan,
            "joint_positive_reference_weight_sum": passed.reference_weight.sum() if passed.reference_weight.notna().all() else np.nan,
            **etf_events[day],
        }
        for name in WINDOWS:
            values = group[f"stock_{name}"]
            event[f"stock_{name}_available_count"] = int(values.notna().sum())
            event[f"cohort_equal_mean_stock_{name}"] = values.mean() if values.notna().all() else np.nan
            event[f"cohort_reference_contribution_{name}"] = group[f"reference_contribution_{name}"].sum() if complete_weights and values.notna().all() else np.nan
            event[f"etf_{name}"] = group[f"etf_{name}"].iloc[0]
        event["reference_remainder_post20"] = event["etf_post20"] - event["cohort_reference_contribution_post20"]
        events.append(event)
    events = pd.DataFrame(events)
    events["overlapping_prior_event"] = False
    for index, row in events.iterrows():
        events.loc[index, "overlapping_prior_event"] = bool(events.iloc[:index].exit_date.gt(row.entry_date).any())
    for frame, name in [(companies, "49条公司与ETF的同窗比较"), (events, "23个日期的指数覆盖与剩余收益")]:
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
        frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    same_window = events.cohort_equal_mean_stock_post20.notna() & events.etf_post20.notna()
    case_rows = companies[companies.announcement_id.isin(CASES)].copy()
    for key in ["entry_date", "exit_date", "etf_event_net_return"]:
        case_rows[key] = case_rows.date.map(events.set_index("date")[key])
    save_json(OUT / "三例公司到指数.json", case_rows.to_dict("records"))
    result = {
        "study_id": cfg["study_id"], "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "HISTORICAL_COMPANY_EXPOSURE_AND_RETURN_BRIDGE_COMPLETED",
        "classification": "PROGRESS_HISTORICAL_COMPANY_INDEX_TRANSMISSION",
        "sample": {"company_reports": len(companies), "companies": companies.ts_code.nunique(),
                   "event_dates": len(events), "company_joint_positive": int(companies.company_joint_positive.sum()),
                   "single_company_dates": int(events.company_count.eq(1).sum()),
                   "missing_weights": int(companies.reference_weight.isna().sum()),
                   "nonmembers": int((~companies.member_on_measurement_day).sum()),
                   "missing_stock_return_endpoints": {name: int(companies[f"stock_{name}"].isna().sum()) for name in WINDOWS},
                   "events_overlapping_an_earlier_event": int(events.overlapping_prior_event.sum())},
        "reference_weights": {"status": cfg["weight_contract"]["status"],
                              "cohort_exposure": describe(events.reference_weight_sum),
                              "joint_positive_exposure": describe(events.joint_positive_reference_weight_sum),
                              "single_company_exposure": describe(events.loc[events.company_count.eq(1), "reference_weight_sum"]),
                              "max_snapshot_age_days": int(companies.weight_age_calendar_days.max())},
        "same_window_post20": {
            "date_count": int(same_window.sum()),
            "cohort_equal_mean_stock": describe(events.loc[same_window, "cohort_equal_mean_stock_post20"]),
            "etf": describe(events.loc[same_window, "etf_post20"]),
            "reference_contribution": describe(events.loc[same_window, "cohort_reference_contribution_post20"]),
            "cohort_up_etf_down_count": int(((events.cohort_equal_mean_stock_post20 > 0) & (events.etf_post20 < 0)).sum()),
            "cohort_down_etf_up_count": int(((events.cohort_equal_mean_stock_post20 < 0) & (events.etf_post20 > 0)).sum())},
        "etf_next_open_20_session_events": describe(events.etf_event_net_return),
        "etf_event_mean_without_best": events.etf_event_net_return.drop(events.etf_event_net_return.idxmax()).mean(),
        "full_causal_effect_on_equity_returns_established": False,
        "index_contribution_is_exact": False, "new_accounts": 0, "new_return_diagnostics": 1,
        "new_fitted_parameters": 0, "new_forecast_cards": 0, "goal_achieved": False,
        "old_account_unchanged": True, "old_account_path": cfg["inputs"]["old_account"],
        "wait_for_future_data": False, "independent_validation": False, "orders_authorized": False,
        "report": str((OUT / "历史发现_公司改善如何传到指数.md").relative_to(ROOT)).replace("\\", "/"),
    }
    save_json(OUT / "result.json", result)
    write_report(companies, events, case_rows, result)
    plot(events)
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


def write_report(companies, events, cases, result):
    weight = result["reference_weights"]["cohort_exposure"]
    single = result["reference_weights"]["single_company_exposure"]
    net = result["etf_next_open_20_session_events"]
    post = result["same_window_post20"]
    lines = [
        "# 历史发现：经营改善如何传到沪深300与510300", "",
        "2026年9月30日。沿用现金质量分解中的全部23个日期、49条公司报告、41家公司。这里只做已发生事件的覆盖与收益诊断，不建立前瞻判断。", "",
        f"**财报篮子当日覆盖的沪深300参考权重中位数为{fraction_pct(weight['median'])}；15个单公司日期的中位数为{fraction_pct(single['median'])}。改善涉及哪些公司、涉及多少指数，决定了公司故事能否代表宽基。**", "",
        "## 口径简化，但时间和对象必须对齐", "",
        "测量日d继承原T11记录，为名义公告日之后首个交易日，未证明历史精确首见时间。股票使用本地新浪后复权收盘价；ETF使用既有含息财富序列。同窗比较分成d−21收盘至d−1收盘、d−1收盘至d收盘、d收盘至d+20收盘。三个窗口事先固定，不尝试不同持有期。", "",
        "另外列示d+1开盘至d+21开盘的ETF事件收益，按10万元示例预算、每边万分之四佣金且最低5元、每边千分之一不利滑点、最小价位和100股一手计费，登记日分红权益计入。预算是开盘价格已知条件下的事件示例，未模拟预先定量下单、成交量、风险退出、20万元完整账户和现金时间线。", "",
        f"历史权重取严格早于测量日的最近月末快照，最大间隔{result['reference_weights']['max_snapshot_age_days']}个自然日。全部公司均在测量日历史成员表内，缺失权重{result['sample']['missing_weights']}条。这些供应商权重没有逐期首次版本证明，只用于参考量级；不将选中公司权重放大至100%。", "",
        "## 覆盖不足不能被因子分数掩盖", "",
        "| 项目 | 历史样本结果 |", "|---|---:|",
        f"| 全部篮子覆盖权重中位数 | {fraction_pct(weight['median'])} |",
        f"| 全部篮子覆盖权重范围 | {fraction_pct(weight['minimum'])}—{fraction_pct(weight['maximum'])} |",
        f"| 公司自身同时满足原三条件的覆盖中位数 | {fraction_pct(result['reference_weights']['joint_positive_exposure']['median'])} |",
        f"| 单公司日期 | {result['sample']['single_company_dates']}/23 |",
        f"| 单公司日期覆盖中位数 | {fraction_pct(single['median'])} |", "",
        "公司披露时间分散，这里的少量公司只是当日发生披露的篮子，并不描述其他成员最新已知基本面。不能由篮子覆盖低断言整个指数没有改善；同样不能由篮子为正直接断言指数盈利普遍恢复。", "",
        "## 三个经营来源案例，股价表现并不相同", "",
        "这三例沿用上一轮按财务构成选定的案例，没有根据本轮收益重新挑选。表中前三个收益均为公司后复权收益；ETF列与公司后20日使用完全相同收盘端点。", "",
        "| 报告与测量日 | 参考权重 | 公司此前20日 | 公司测量日 | 公司随后20日 | 同期ETF | 下一开盘后ETF净收益 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cases.sort_values("date").itertuples():
        lines.append(f"| {CASES[row.announcement_id]} {row.date:%Y-%m-%d} | {fraction_pct(row.reference_weight)} | {pct(row.stock_pre20)} | {pct(row.stock_response)} | {pct(row.stock_post20)} | {pct(row.etf_post20)} | {pct(row.etf_event_net_return)} |")
    lines += ["",
        "牧原的现金改善同时涉及猪价、成本、当季利润恢复和营运资本；小商品城涉及新市场招商与项目收款；上机数控涉及单晶硅业务规模扩张、回款和资产变化。来源解释见上一轮原报告。这里的股价结果不能反过来证明某一个经营解释具有独立因果效力。", "",
        "此前20日上涨可能包含预期调整，也可能来自市场及行业共同变化；单凭上涨不能证明财报已充分定价。要测量真正预期差仍需当时一致预期、预告或其他可定位的先行信息，不能把历史季节性分数直接当作市场预期。", "",
        "## 全部事件：经营解释正确仍不等于有ETF买入优势", "",
        f"相同随后20日窗口，23个日期的公司等权平均收益再按日期等权平均为{pct(post['cohort_equal_mean_stock']['mean'])}，同期ETF均值为{pct(post['etf']['mean'])}。其中{post['cohort_up_etf_down_count']}个日期出现公司均值上涨、ETF下跌，另有{post['cohort_down_etf_up_count']}个日期公司均值下跌、ETF上涨。这个差异说明不能用少量公司的涨跌替代指数收益，尚未隔离市场和行业共同冲击。", "",
        f"下一开盘后20交易日的23次ETF示例事件，费用后均值{pct(net['mean'])}、中位数{pct(net['median'])}，正收益{net['positive_count']}/23；去掉最好一次后均值{pct(result['etf_event_mean_without_best'])}。这些是全部原正向篮子的描述，不是新增选择规则、可复用因果规律或完整账户夏普。{result['sample']['events_overlapping_an_earlier_event']}次事件与更早事件区间重叠，样本不独立，也不能把各次10万元示例同时当作同一账户交易。", "",
        "固定参考权重乘公司收益再求和，只显示这些公司直接价格贡献的大致量级。其与ETF收益之差还包含其余股票、权重漂移、指数编制、复权口径、ETF跟踪和价格差异，不得称为已识别的‘其他公司精确贡献’，更不能称为财报冲击的因果分解。", "",
        "## 全部23个日期保留", "",
        "| 测量日 | 公司数 | 覆盖权重 | 公司均值随后20日 | 同期ETF | 下一开盘后ETF净收益 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in events.itertuples():
        lines.append(f"| {row.date:%Y-%m-%d} | {row.company_count} | {fraction_pct(row.reference_weight_sum)} | {pct(row.cohort_equal_mean_stock_post20)} | {pct(row.etf_post20)} | {pct(row.etf_event_net_return)} |")
    lines += ["",
        "## 对当前研究的实际含义", "",
        "研究顺序应保留三次连接：先从财务变化回到经营来源，再识别受影响公司及当时指数份量，最后比较信息可用后剩余收益。某一环成立不能替代其余环。当前样本能帮助发现传导缺口，没有建立新的510300择时优势。", "",
        "旧T11日期代理完整账户压力费用夏普−0.352、年化−0.27%的失败保留，本轮没有改其筛选、持有期或退出规则。原夏普1.2及年化10%目标未实现；研究保持历史范围，不等待未来数据，不把本轮已知历史当作独立验证。", "",
        "## 文件", "",
    ]
    for label, path in [
        ("49条公司比较", OUT / "49条公司与ETF的同窗比较.csv"),
        ("23个日期比较", OUT / "23个日期的指数覆盖与剩余收益.csv"),
        ("三例原始计算", OUT / "三例公司到指数.json"),
        ("事先固定的口径", OUT / "protocol.json"),
        ("计算脚本", ROOT / "research/historical_company_index_bridge_v1.py"),
        ("财务变化的上游经营解释", ROOT / "reports/research/510300_historical_cash_quality_causes_v1/历史发现_现金质量为何改善.md"),
    ]:
        lines.append(f"- [{label}](<{path.as_posix()}>)")
    lines += ["", f"![指数覆盖与同窗收益](<{(OUT / '经营改善到指数覆盖.png').as_posix()}>)", ""]
    (OUT / "历史发现_公司改善如何传到指数.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
