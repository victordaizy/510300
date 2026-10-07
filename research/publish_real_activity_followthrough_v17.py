"""复算后续经营时钟并发布第十七轮；不把后见确认变为起点信号。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager
import numpy as np
import pandas as pd

from real_activity_followthrough_v17 import OUT, ROOT, CUTOFF, SUPPORTED, clock


def load(name):
    return pd.read_csv(OUT / "results" / name)


def write_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for item in frozen["inputs"]:
        assert digest(OUT / "inputs" / item["name"]) == item["sha256"]
    origins = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    observations = load("104个月_后续经营结果与确认时钟.csv").set_index("stat_month")
    future = load("后续经营披露_逐条来源与时间.csv")
    assert len(observations) == 104 and len(future) == 700
    assert observations.joint_credit_orders_state.fillna("NO_VIEW").equals(origins.joint_credit_orders_state.fillna("NO_VIEW"))
    pmi = pd.read_parquet(OUT / "inputs/pmi_orders.parquet").rename(columns={"available_at": "known_at"})
    pmi = pd.concat([pmi, pd.DataFrame(json.loads((OUT / "inputs/pmi_extra.json").read_text(encoding="utf-8")))], ignore_index=True).set_index("reference_period")
    source = {"新订单": (pmi, ["first_release_value"]),
              "工业经营": (pd.DataFrame(json.loads((OUT / "inputs/industrial_profit.json").read_text(encoding="utf-8"))).set_index("stat_month"), ["receivable_days_published_yoy_change", "profit_ytd_reported_yoy_pct"]),
              "entrepreneur": (pd.read_parquet(OUT / "inputs/entrepreneur.parquet").set_index("quarter"), ["sales_revenue_collection_index", "fund_turnover_index"]),
              "banker": (pd.read_parquet(OUT / "inputs/banker.parquet").set_index("quarter"), ["loan_demand_index"])}
    checked_facts = 0
    for record in future.itertuples():
        raw, fields = source[record.channel]
        if record.status != "OBSERVED_BY_CUTOFF":
            assert record.status == "SOURCE_NOT_COVERED" and record.target_period not in raw.index
            continue
        original = raw.loc[record.target_period]
        assert clock(record.origin_snapshot_at) < clock(record.known_at) <= CUTOFF
        assert clock(record.known_at) == clock(original.known_at)
        for field in fields:
            assert abs(getattr(record, field) - original[field]) < 1e-9
            checked_facts += 1
        if record.channel != "新订单":
            prefix = "profit" if record.channel == "工业经营" else record.channel
            assert origins.loc[record.stat_month, prefix + "_status"] == "AVAILABLE_RECONSTRUCTED"
    # 过期起点仍保留，但不得从旧资料重新填出当前有效基准。
    for prefix, channel in [("profit", "工业经营"), ("entrepreneur", "entrepreneur"), ("banker", "banker")]:
        invalid = origins.index[origins[prefix + "_status"] != "AVAILABLE_RECONSTRUCTED"]
        assert not future[(future.channel == channel) & future.stat_month.isin(invalid)].shape[0]
    for month, observed in observations.iterrows():
        if observed.row_status != "ORIGIN_OBSERVED":
            assert not future.stat_month.eq(month).any()
            continue
        rows = future[(future.stat_month == month) & (future.channel == "新订单")].sort_values("horizon")
        assert len(rows) == 3
        expected_periods = [str(pd.Period(observed.orders_period, freq="M") + k) for k in [1, 2, 3]]
        assert rows.target_period.tolist() == expected_periods
        if rows.status.eq("OBSERVED_BY_CUTOFF").all():
            vals = rows.first_release_value.to_numpy()
            assert abs(vals.mean() - observed.orders_next3_mean) < 1e-8
            assert abs(vals.mean() - observed.orders_first_release_value - observed.orders_next3_mean_minus_origin) < 1e-8
            assert bool(observed.orders_all_next3_ge50) == bool((vals >= 50).all())
        else:
            assert pd.isna(observed.orders_next3_mean)
    market = pd.read_csv(OUT / "inputs/market.csv").set_index("date").sort_index()
    path_errors, path_count, risk_count = [], 0, 0
    for month, row in observations.iterrows():
        if row.row_status != "ORIGIN_OBSERVED":
            continue
        o = origins.loc[month]
        entry = o.E0_20_entry_date
        end = o.E0_20_exit_date
        for field, saved in [("E0_20_return", row.E0_return_percent), ("E1_20_return", row.E1_return_percent)]:
            if pd.isna(o[field]):
                assert pd.isna(saved)
            else:
                assert abs(saved - o[field] * 100) < 1e-8
        next_known = clock(pmi.loc[row.orders_next1_period, "known_at"])
        expected_confirmation = next_known.normalize() + pd.Timedelta(hours=21)
        if expected_confirmation < next_known:
            expected_confirmation += pd.Timedelta(days=1)
        assert expected_confirmation == clock(row.confirmation_snapshot_at)
        future_days = market.index[market.index > str(expected_confirmation.date())].tolist()
        for delay, prefix in [(0, "confirmation"), (1, "confirmation_delay1")]:
            status = row[prefix + "_path_status"]
            if status != "OBSERVED_WITHIN_ORIGINAL_WINDOW":
                assert len(future_days) <= delay or future_days[delay] > end
                continue
            start = future_days[delay]
            assert start == row[prefix + "_entry_date"]
            initial_price, start_price, end_price = market.loc[entry, "open"], market.loc[start, "open"], market.loc[end, "close"]
            cash_early = market.loc[(market.index > entry) & (market.index <= start), "dividend"].sum()
            cash_late = market.loc[(market.index > start) & (market.index <= end), "dividend"].sum()
            early = (start_price + cash_early - initial_price)/initial_price * 100
            late = (end_price + cash_late - start_price)/initial_price * 100
            fresh = (end_price + cash_late - start_price)/start_price * 100
            path_errors.extend([abs(early - row[prefix + "_before_contribution_pp"]), abs(late - row[prefix + "_after_contribution_pp"]),
                                abs(fresh - row[prefix + "_fresh_entry_return_percent"]), abs(early + late - row.E0_return_percent)])
            assert market.loc[start:end].shape[0] == row[prefix + "_remaining_sessions"]
            path_count += 1
        date = row.confirmation_observation_date
        r = market.loc[:date].total_simple.tail(20).to_numpy()
        assert len(r) == 20 and np.isfinite(r).all()
        rv = r.std(ddof=1) * np.sqrt(252) * 100
        downside = np.sqrt(np.minimum(r, 0).dot(np.minimum(r, 0)) / 20 * 252) * 100
        assert abs(rv - row.confirmation_rv20_percent) < 1e-8
        assert abs(downside - row.confirmation_downside20_percent) < 1e-8
        risk_count += 1
    assert np.isfinite(path_errors).all() and max(path_errors) < 1e-8
    survey = load("后续季度调查_月份重复关系.csv")
    keys = ["training_regime", "joint_state", "channel", "field", "target_quarter"]
    assert survey.groupby(keys, dropna=False).future_value.nunique().max() == 1
    dedup = survey.drop_duplicates(keys)
    qsummary = load("季度去重后的经营调查结果.csv")
    for row in qsummary.itertuples():
        a = dedup[(dedup.training_regime == row.training_regime) & (dedup.joint_state == row.joint_state) & (dedup.channel == row.channel) & (dedup.field == row.field)]
        assert a.target_quarter.nunique() == row.unique_quarters
        assert abs(a.future_qoq_change.mean() - row.qoq_change_mean) < 1e-8
    checks = {"status": "PASS_SAVED_REAL_ACTIVITY_CLOCK_PATH_RECOMPUTATION", "origins": len(observations), "future_source_rows": len(future),
              "raw_future_values_checked": checked_facts, "price_clock_splits_recomputed": path_count, "confirmation_risk_windows_recomputed": risk_count,
              "maximum_path_error_pp": float(max(path_errors)), "stale_baseline_excluded": True, "quarter_duplicate_means_checked": True,
              "all_original_groups_and_E0_E1_unchanged": True, "independent_validation": False}
    write_json("verification.json", checks)
    print("后续经营数值、公开先后、季度去重及价格贡献复算通过。")
    return checks


def chart(s):
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(), "font.size": 11, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(14, 8.6), facecolor="#f8f7f2")
    fig.text(.055, .948, "经营状态延续，股价仍有不同方向", fontsize=23, weight="bold", color="#183344")
    fig.text(.055, .902, "既有11个联合支持月全部保留：剪刀差改善、两类中长期信贷多增、新订单≥50", fontsize=12, color="#54646c")
    fig.text(.29, .862, "制造业新订单指数", ha="center", fontsize=14, color="#183344")
    fig.text(.745, .862, "510300后20日毛收益", ha="center", fontsize=14, color="#183344")
    ax = fig.add_axes([.12, .18, .34, .62])
    values = s[["orders_first_release_value", "orders_next1_value", "orders_next2_value", "orders_next3_value"]].to_numpy()
    ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=29, vcenter=50, vmax=54))
    ax.set_yticks(np.arange(len(s)), s.stat_month)
    ax.set_xticks(range(4), ["起点已知", "随后第1期", "随后第2期", "随后第3期"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=10)
    for i in range(len(s)):
        for j in range(4):
            val = values[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=12, color="white" if val < 40 or val >= 52.3 else "#183344", weight="bold")
    for y in np.arange(.5, len(s), 1):
        ax.axhline(y, color="#f8f7f2", linewidth=4)
    ax.axvline(.5, color="#f8f7f2", linewidth=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bx = fig.add_axes([.56, .18, .37, .62])
    y = np.arange(len(s))
    vals = s.E0_return_percent.to_numpy()
    bx.barh(y, vals, color=["#b85d43" if v >= 0 else "#37677f" for v in vals], height=.58, label="E0：原下一开盘起20日")
    bx.scatter(s.E1_return_percent, y-.1, color="#1f3441", marker="D", s=22, zorder=3, label="E1：再延迟一交易日")
    bx.set_ylim(len(s)-.5, -.5)
    bx.set_xlim(-16.5, 15)
    bx.set_yticks(y, [])
    bx.axvline(0, color="#9aa7ab", linewidth=.8)
    bx.grid(axis="x", color="#e0e2de", linewidth=.6)
    bx.set_axisbelow(True)
    bx.set_xlabel("收益率（%）", labelpad=10)
    bx.set_facecolor("#f8f7f2")
    for spine in ["top", "right", "left"]:
        bx.spines[spine].set_visible(False)
    bx.tick_params(axis="y", length=0)
    for pos, val in zip(y, vals):
        bx.text(val + (.3 if val >= 0 else -.3), pos+.35, f"{val:+.2f}%", ha="left" if val >= 0 else "right", fontsize=9, color="#183344")
    bx.legend(loc="upper center", bbox_to_anchor=(.48, -.12), frameon=False, fontsize=10)
    fig.text(.055, .096, "下一期全部在50以上；后三期连续≥50的有7个月，但只有5个月的后三期均值高于起点。", fontsize=11, color="#183344")
    fig.text(.055, .060, "后三列是事后经营结果，不能进入起点判断。11个月的33次后续观察只涉及19个不同统计月。", fontsize=10, color="#64747c")
    fig.text(.055, .033, "分红留现金，未扣费用；共同支持月集中在2019—2021年，未获得独立预测验证。", fontsize=10, color="#64747c")
    fig.savefig(OUT / "figures/订单延续与股价路径_全部11个月.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def report(checks):
    frame = load("104个月_后续经营结果与确认时钟.csv")
    s = frame[frame.joint_credit_orders_state == SUPPORTED].copy()
    groups = load("全部既有状态_经营后续与价格分布.csv")
    q = load("季度去重后的经营调查结果.csv")
    future = load("后续经营披露_逐条来源与时间.csv")
    supported_future = future[future.joint_state == SUPPORTED]
    company = s.set_index("stat_month").loc["2021-01"]
    def link(label, rel):
        return f"[{label}](<{(OUT / rel).as_posix()}>)"
    def pct(value):
        return "未计算" if pd.isna(value) else f"{value:+.2f}%"
    small_table = table(["原货币统计月", "后续三期新订单", "三期均值较起点", "此前60日", "原E0后20日", "原E1后20日"],
        [[r.stat_month, f"{r.orders_next1_value:.1f} / {r.orders_next2_value:.1f} / {r.orders_next3_value:.1f}", f"{r.orders_next3_mean_minus_origin:+.2f}", pct(r.pre_return60_pp), pct(r.E0_return_percent), pct(r.E1_return_percent)] for r in s.itertuples()])
    all_rows = []
    for r in groups.itertuples():
        label = "无可用起点" if pd.isna(r.joint_state) else r.joint_state.replace("剪刀差改善_", "改善／")
        all_rows.append(["旧M1" if r.training_regime.startswith("M1_OLD") else "新M1", label, r.origin_months,
                         f"{r.orders_all_next3_ge50_true_count}/{r.orders_all_next3_ge50_n}", pct(r.E0_return_percent_mean), pct(r.E1_return_percent_mean)])
    all_table = table(["制度", "既有状态", "月数", "后三期均≥50／完整三期", "原E0均值", "原E1均值"], all_rows)
    after_table = table(["原统计月", "下一期订单可用后的入场日", "至原终点剩余日数", "此前原本金贡献", "此后原本金贡献", "此后新起点毛收益", "再延迟一天"],
        [[r.stat_month, r.confirmation_entry_date, int(r.confirmation_remaining_sessions), f"{r.confirmation_before_contribution_pp:+.2f}pp", f"{r.confirmation_after_contribution_pp:+.2f}pp", pct(r.confirmation_fresh_entry_return_percent), pct(r.confirmation_delay1_fresh_entry_return_percent)] for r in s.itertuples()])
    quarterly_rows = []
    names = {"sales_revenue_collection_index": "企业家销货款回笼", "fund_turnover_index": "企业家资金周转", "loan_demand_index": "银行家贷款需求"}
    for r in q[q.joint_state == SUPPORTED].itertuples():
        quarterly_rows.append([names[r.field], r.unique_quarters, f"{r.qoq_positive_quarters}/{r.unique_quarters}", f"{r.qoq_change_mean:+.2f}", f"{r.yoy_change_mean:+.2f}"])
    q_table = table(["下一季度调查", "不同目标季度", "环比上升季度", "平均环比变化", "平均同比变化"], quarterly_rows)
    first_coverage = int(s.industrial_next1_collection_yoy_days.notna().sum())
    first_positive = int((s.industrial_next1_collection_yoy_days > 0).sum())
    first_narrowing = int((s.industrial_next1_collection_yoy_change_from_origin < 0).sum())
    delay1_positive = int((s.confirmation_delay1_fresh_entry_return_percent > 0).sum())
    timeline = table(["公开或执行时间", "当时可以知道什么", "原窗口中的位置"], [
        ["2021-02-09 21:00", "1月货币数据、当时信贷和新订单52.3；此前60日510300已涨15.95%", "原观察起点"],
        ["2021-02-10开盘", "按原E0开始记录", "至3月16日20日毛收益−11.09%"],
        ["2021-02-28公布，3月1日开盘起观察", "2月新订单51.5，仍扩张但比上月回落0.8", "剩12个交易日，毛收益−5.72%"],
        ["2021-03-25当日日末上界", "一季度企业家收款指数63.5，较上季提高0.8", "晚于原3月16日退出日"],
        ["2021-03-27当日日末上界", "1—2月工业应收账款平均回收期同比缩短14.5天", "晚于原退出日；含2020年疫情比较基数"],
        ["2021-04-30台账可用时间", "后续第三期新订单52.0；三期都在50以上至此才完整可知", "不能回填至2月起点"],
    ])
    note = f"""# 第十七轮：经营改善后来发生了，510300还剩什么收益？

**本轮得到更明确的约束：订单保持扩张是真实出现过的，但它既不等于扩张继续加速，也不足以推出510300随后20日上涨。原11个联合支持月，下一期订单全部高于50，7个月后三期始终不低于50；原E0平均毛收益仍只有+0.15%，最差单月途中到过−12.74%。**

这个结论把“经济传导没有发生”和“经济指标延续、但股价没有相应剩余收益”分开。它还不能识别股票折现率、盈利预期和新增消息各自造成了多少收益。研究目标保持进行中。

## 一、样本与判断保持原样

本轮连接原104个月的后续资料；沿第四轮已经确定的联合状态，旧、新M1分别报告。共同支持仍指：剪刀差三个月改善，企业和居民中长期累计新增贷款均同比多增且超过披露舍入界，起点已知制造业新订单不低于50。没有增加高位低位条件、改阈值或删除亏损月。原11个月全部集中在2019—2021年，不能将月份重复观察当作独立验证。

先保存不含新未来结果的原起点表，再连接后续资料。以起点已知订单统计月为基准，固定查看随后第1、2、3个月；三期完整才计算三期均值。回款与工业利润按后续三个实际披露期连接，1—2月合并不拆成两个月；企业家与银行家调查取下一季度。超过原截止日或来源缺失保留缺失。过期起点也保持NO_VIEW，不重新用旧表填成当前证据。

这不是新增模型或账户回测。原104个月的市场结果已经被研究过，本轮只增加经营后续和信息时序证据。

## 二、11个月中，经营延续具体到了什么程度

{small_table}

下一期新订单全部高于50，后三期均不低于50的有7个月；但后三期均值高于起点的只有5个月，下一期比起点上升的也只有5个月。**扩张状态、扩张的加速或减速、市场尚未预期的信息，是不同问题。** 本轮没有PMI市场预期差，不能把“仍高于50”自动命名为新利好。

国家统计局在[2021年2月原公告](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901006.html)中同时说明新订单高于临界点和环比回落。50是扩散指数的临界点，新订单指数不是订单金额同比增速，不能把53比51高两个点写成实际订单多增长2%。

33次后续月观察实际只涉及19个不同统计月，窗口有重叠。11个月又有起点订单本已较高的选择特征；之后仍较高不能证明相对可比状态存在增量预测能力。2019年末两行都包含2020年2月的29.3，这是一条共同冲击，不能当两次独立经济失败。

{link('全部11个月的订单与收益图', 'figures/订单延续与股价路径_全部11个月.png')}

## 三、回款和融资需求也不是一致向好

{q_table}

11个月对应的调查按目标季度去重后只有7个季度，表内均值按季度等权。收款、周转的环比上升季度占多数，但极端下降会拉低环比均值；同比与环比方向也不同。这些是企业家和银行家的调查感受，不能冒充公司现金流、贷款余额或银行实际批准率。季度名称不代表资料必须在季度末才公开，按原报告时钟连接。

工业回收期的下一披露期在11个月中有{first_coverage}个月可核对。其中{first_positive}个月回收期仍比上年同期更长；{first_narrowing}个月的同比变化较起点下降。**拖延程度收窄，与已恢复到上年同期的回款速度，应分开描述。** 部分2021年同比缩短又包含2020年疫情造成的高基数，不能把同比改善全部解释为当期新需求爆发。

制造业和规模以上工业也不是沪深300的全部行业。银行息差、保险新业务、消费渠道、企业成本的差异仍需沿第十六轮分别分析，本轮没有给所有层次打总分。

## 四、一个完整反例：经营指标后来改善，股价仍然先跌

2021年1月是第四轮已经固定过的反例，本轮保留全部其他月份，并把这一例的经营公布时钟补齐。

{timeline}

2、3、4月新订单分别51.5、53.6、52.0，三期均值略高于起点52.3。企业家回款感受和后出的工业回收期数据也有改善，但它们披露在不同日期，有些晚于原交易窗口结束。[企业家一季度原报告页面](https://www.pbc.gov.cn/diaochatongjisi/116219/116227/62297ae8d477435c9dd2ad30dfd184a4/index.html)、[工业1—2月原报告](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901035.html)的身份和本地原件位置保存在逐条来源表。

因此这个亏损病例不能一概解释为“后来订单全部恶化”。同时，后来经营改善不能回填成2月9日已知理由。此前60日上涨15.95%是可核对的价格背景，但仅凭随后下跌，仍无法确认当时已经充分定价，或计算估值过高的幅度。

在下一期订单公布后的确认快照，已知20日下行波动比原起点扩大了约{company.confirmation_downside_change_from_origin_pp:.2f}个百分点。经济仍处扩张区间时，市场下跌路径已经发生变化；它们属于同一时点的不同层面，不能只保留其中一个。

## 五、等待下一期订单之后，剩余路径也有正有负

固定等待下一期新订单公开后，第一个21点快照之后的下一交易日开盘，再观察至原E0终点。每月都采用相同的时钟，不依下一期数值好坏改变入场时刻。额外延迟一个交易日也完整列出。原终点不变，因此剩余日数是4至15日，与原20日不是同持有长度的策略比较。

{after_table}

两列本金贡献之和精确等于原20日E0收益；最后两列则按等待后的新开盘价归一化，不能与原本金贡献混加。确认当日除息权益归此前持有人，等待到该日才买入不享有已除息权益。所有收益都是分红留现金的毛观察收益，不包含费用和账户约束。

等待后这11个月只有5个月剩余收益为正，中位数−0.14%，均值+1.56%；再等一天有{delay1_positive}个月为正，中位数+0.65%、均值+1.24%。2019年12月案例等待后从2020-02-03开盘观察的剩余收益为+11.13%，不能删除，也不能把这个强反弹当普遍结果。原起点至等待开盘已贡献−10.16个百分点，随后贡献+10.00个百分点，合计仍是原窗口约−0.17%。这说明等待同时改变了所承担的路径，并非自动提高确定性。

11个月原E0均值+0.15%、中位数+0.80%，7次为正；原E1均值+0.48%。此前60日平均已上涨10.92%。数据呈现了“已发生上涨、经营状态延续、后续收益分散”并存的事实；它支持继续检查预期与定价时序，没有证明简单反向交易或高位卖出的规则。

## 六、其余月份与新口径全部保留

{all_table}

各组起点水平、年份和经济环境不同，这张表用于完整披露，不能把组间均值直接当因果差异或挑选最优规则。尤其新M1口径没有共同支持月份，不能用旧口径11个月替它验证；未支持组也存在上涨，不能把需求证据不足直接改写成看空信号。

本轮明确保留三类缺口：经营指标本身有范围和比较基数限制；公开经营状态尚不等于市场未预期的新信息；完整短期股价变化还受权重行业盈利、政策和外部折现条件影响。接下来应补的是相关历史事件当时的预期与新增信息，结合原价格路径和权重行业核对，不能继续将已知亏损月份通过加条件筛走。

## 七、完成范围与文件

共104个起点、{checks['future_source_rows']}条后续披露记录，核对{checks['raw_future_values_checked']}个未来经营原始数值、{checks['price_clock_splits_recomputed']}条价格时钟切分和{checks['confirmation_risk_windows_recomputed']}个确认时波动窗口。过期起点和未覆盖资料均保持缺失；首次计算中对过期调查期名的误用已按原冻结规则修正，修正前草稿留存，11个共同支持月的原有结果完全未变。

当前下载版和网页台账时间不等于不可修订的历史首版送达认证，次日延迟敏感性也不能代替首版证据。复算通过仅确认数值与时序一致，未构成独立预测验证。总目标尚未达成，新增模型0、策略账户0、订单0。

- {link('104个月完整后续表', 'results/104个月_后续经营结果与确认时钟.csv')}
- {link('原11个共同支持月全部证据', 'results/原11个共同支持月_全部后续证据.csv')}
- {link('后续披露逐条来源和时间', 'results/后续经营披露_逐条来源与时间.csv')}
- {link('全部状态逐年结果', 'results/全部既有状态_逐年结果.csv')}
- {link('季度去重后的调查结果', 'results/季度去重后的经营调查结果.csv')}
- {link('冻结方案', 'protocol.json')}、{link('复算回执', 'verification.json')}
"""
    (OUT / "第十七轮_经营传导兑现与剩余收益.md").write_text(note, encoding="utf-8")
    return s


def main():
    if (OUT / "completion_receipt.json").exists():
        raise RuntimeError("本轮已完成，不覆盖最终记录。")
    checks = verify()
    selected = report(checks)
    chart(selected)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("报告及全部11个月图已生成，待图像检查后记录完成状态。")


if __name__ == "__main__":
    main()
