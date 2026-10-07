"""保留全部就业事件，恢复国内状态与持有期间的新信息；不生成新交易规则。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import factor96_margin_repair_v1 as core

OUT = ROOT / "reports/research/510300_historical_index_domestic_revision_v1"
JOBS = ROOT / "reports/research/510300_historical_index_employment_surprise_v1"
DOMESTIC = ROOT / "reports/research/510300_historical_index_domestic_constraint_clock_v1"
LOGISTICS = ROOT / "reports/research/510300_historical_index_logistics_clock_v1"
MARKET = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/market.parquet"
TZ = ZoneInfo("Asia/Shanghai")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(name, value):
    core.save(OUT / name, value)


def source_plan():
    rows = [
        ("SH_PLAN_20220516", "2022-05-16", "2022-05-16", "上海公开三阶段恢复方向，目标6月进入恢复期", "SH_REOPEN_MAY2022",
         "https://www.shanghai.gov.cn/nw9822/20220516/b9e442c1930e42f8988de113165f0908.html", ["三个阶段", "6月1日"],
         "方向与条件", "恢复安排以风险可控为前提，尚非全面解除约束。"),
        ("PACKAGE_DIRECTION_20220523", "2022-05-23", "2022-05-24", "六方面33项稳经济措施的决策方向公开", "NATIONAL_PACKAGE_MAY2022",
         "https://jw.beijing.gov.cn/jyzy/gwyxx/202205/t20220524_2719122.html", ["33项", "留抵退税"],
         "方向与条件", "原文电头和标注为5月23日，转载URL为5月24日，保守取24日日末。不是5月31日才首次出现的政策方向。"),
        ("PACKAGE_IMPLEMENTATION_20220525", "2022-05-25", "2022-05-25", "稳经济大盘会议要求月底前出台可操作细则", "NATIONAL_PACKAGE_MAY2022",
         "https://app.www.gov.cn/govdata/gov/202205/25/485360/article.html", ["33条", "可操作"],
         "落实要求", "会议提出实施时限和督查，不等于企业已经收到资金或产生销售。"),
        ("SH_APPROVAL_20220529", "2022-05-29", "2022-05-29", "上海宣布6月1日起取消企业复工复产审批审核", "SH_REOPEN_MAY2022",
         "https://www.shanghai.gov.cn/nw9820/20220529/6401f75162334e14aa5d099029374b46.html", ["审批审核", "6月1日"],
         "明确执行约束", "降低复工程序约束，仍须符合防疫要求，不代表最终需求已经恢复。"),
        ("SH_MOBILITY_20220530", "2022-05-30", "2022-05-30", "上海公布6月1日起恢复出入、公交和机动车通行", "SH_REOPEN_MAY2022",
         "https://www.shanghai.gov.cn/nw12344/20220530/ccfdd516456b4c8b855f203d4de9016f.html", ["公共交通", "电子通行证"],
         "明确执行约束", "保留中高风险等区域例外，通行允许不等于订单和人员履约全部修复。"),
        ("PACKAGE_PRESS_20220531", "2022-05-31", "2022-05-31", "发改委发布会解释稳经济措施的投资和供应链传导", "NATIONAL_PACKAGE_MAY2022",
         "https://www.gov.cn/fuwu/2022-05/31/content_5693277.htm", ["复工达产", "有效需求"],
         "配套解释", "盘活存量资产所得资金需要再投入项目，资金回收不是当期新增利润。"),
        ("PACKAGE_TEXT_20220531", "2022-05-31", "2022-06-01", "国发2022年12号具体措施全文的政府转载", "NATIONAL_PACKAGE_MAY2022",
         "https://www.beijing.gov.cn/zhengce/gwywj/202206/t20220601_2725991.html", ["33项", "5月24日", "延期还本付息"],
         "措施全文", "成文5月24日、标注发布5月31日、转载URL6月1日。正文完整内容保守取6月1日日末，不能按签发日回填。"),
        ("SH_REMAINING_20220606", "2022-06-06", "2022-06-07", "上海后续发布会仍报告社会面感染和经营约束风险", "SH_REOPEN_MAY2022",
         "https://www.shanghai.gov.cn/nw9820/20220607/ae041ad920ce47b0888037a0a533d981.html", ["6月1日", "反弹风险"],
         "实施后反对证据", "内容对应6月6日发布会，网页为6月7日，取7日日末。恢复不是无条件、单向完成。")
    ]
    result = []
    for event_id, announced, usable, title, root, url, anchors, layer, boundary in rows:
        result.append({"id": event_id, "announced_date": announced, "known_at": usable + "T23:59:59+08:00", "title": title,
                       "root_policy_id": root, "url": url, "required_anchors": anchors, "layer": layer,
                       "boundary": boundary, "coverage_is_exhaustive": False, "kind": "政策或履约证据"})
    return result


def collect():
    plan = source_plan()
    save("new_source_plan.json", plan)
    manifest_path = OUT / "new_source_manifest.json"
    prior = {x["id"]: x for x in read(manifest_path)} if manifest_path.exists() else {}
    def fetch(row):
        if row["id"] in prior:
            return prior[row["id"]]
        receipt = {**row, "retrieved_at": datetime.now(TZ).isoformat()}
        target = OUT / "sources" / (row["id"] + ".html")
        try:
            response = requests.get(row["url"], timeout=25)
            response.raise_for_status()
            data = response.content
            soup = BeautifulSoup(data, "html.parser")
            for tag in soup.select("script,style"):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            assert all(word in text for word in row["required_anchors"]), "页面未通过正文关键事实定位"
            target.write_bytes(data)
            target.with_suffix(".txt").write_text(text, encoding="utf-8")
            receipt.update(status="FETCHED", http_status=response.status_code, path=target.relative_to(ROOT).as_posix(),
                           sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        except Exception as error:
            receipt.update(status="UNAVAILABLE", error=str(error))
        return receipt
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, row) for row in plan]):
            row = future.result()
            rows.append(row)
            save("new_source_manifest.json", sorted(rows, key=lambda x: x["known_at"]))
            print(row["id"], row["status"], flush=True)


def compute():
    pmi = read(DOMESTIC / "monthly_pmi.json")
    original = read(JOBS / "event_comparison.json")
    expected_dates = read(OUT / "protocol.json")["fixed_release_dates"]
    assert [r["event_date"] for r in original] == expected_dates
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    opens = pd.to_datetime(market.date).dt.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    news = read(DOMESTIC / "public_information_clock.json")
    new = read(OUT / "new_source_manifest.json")
    assert all(x["status"] in ["FETCHED", "OFFICIAL_WEB_CAPTURE_REVIEWED"] for x in new)
    news.extend({k:v for k,v in r.items() if k not in ["required_anchors"]} for r in new)
    weekly = pd.read_parquet(LOGISTICS / "weekly.parquet")
    weekly = weekly.loc[weekly.complete_week & weekly.publication_clock_valid & weekly.comparison_available_after.notna()].copy()
    for w in weekly.to_dict("records"):
        news.append({"id": "LOGISTICS_" + pd.Timestamp(w["week_end"]).strftime("%Y-%m-%d"), "kind": "物流完整周",
                     "known_at": pd.Timestamp(w["comparison_available_after"]).isoformat(), "week_end": w["week_end"],
                     "title": f"截至{pd.Timestamp(w['week_end']):%m-%d}完整周道路{w['trucks_10k_wow']:+.2%}、揽收{w['pickup_100m_wow']:+.2%}",
                     "holiday": w["holiday"], "source_urls": w["source_urls"], "scope": "既有完整周覆盖；数量同时含需求、供给、假期与基数"})
    for n in news:
        known = pd.Timestamp(n["known_at"])
        # 既有物流字段使用中国本地时间但未附时区；只补时区，不额外推迟一天。
        known = known.tz_localize(TZ) if known.tzinfo is None else known.tz_convert(TZ)
        n["known_at"] = known.isoformat()
        ix = np.flatnonzero((opens > known).to_numpy())
        assert len(ix), n["id"]
        n.update(first_open_idx=int(ix[0]), first_usable_open=opens.iloc[int(ix[0])].isoformat())
    news.sort(key=lambda n: (pd.Timestamp(n["known_at"]), n["id"]))
    save("public_information_clock.json", news)

    def gross(e, i):
        entitlement = market.dividend.iloc[e + 1:i + 1].sum() if i > e else 0.0
        return float((market.open.iloc[i] + entitlement) / market.open.iloc[e] - 1)

    rows, changes, paths = [], [], []
    pmi_positions = {r["stat_month"]:i for i,r in enumerate(pmi)}
    for event in original:
        entry_day = pd.Timestamp(event["entry_date"])
        e = int(np.flatnonzero(market.date.eq(entry_day).to_numpy())[0])
        last = e + 20
        latest = max([p for p in pmi if pd.Timestamp(p["known_at"]) < opens.iloc[e]], key=lambda p: pd.Timestamp(p["known_at"]))
        pos = pmi_positions[latest["stat_month"]]
        previous = pmi[pos - 1] if pos > 0 else None
        row = {k:event.get(k) for k in ["event_id", "event_date", "group", "expectation_available", "payroll_surprise", "earnings_mom_surprise", "nominal2_change_bp", "real10_change_bp", "dxy_change", "net_return5", "net_return20", "gross_return20", "pre_entry_gap"]}
        row.update(entry_date=entry_day, entry_idx=e, exit_date=market.date.iloc[last], exit_idx=last,
                   pmi_month=latest["stat_month"], pmi_known_at=latest["known_at"], pmi_url=latest["url"],
                   pmi_raw_path=latest["raw_path"], prior_20d_total_return=float(market.wealth.iloc[e - 1] / market.wealth.iloc[e - 21] - 1),
                   entry_known_new_policy_ids=[n["id"] for n in new if pd.Timestamp(n["known_at"]) < opens.iloc[e]],
                   prior_20d_from=market.date.iloc[e-21], prior_20d_to=market.date.iloc[e-1])
        for name in ["manufacturing_pmi", "new_orders", "production", "supplier_delivery", "services_activity", "input_price", "output_price"]:
            row[name] = latest[name]
            row[name + "_change"] = latest[name] - previous[name] if previous is not None else None
        row["pmi_state_boundary"] = "读数反映过去统计月；环比为与前次当月原文比较，不等于市场预期差，也不直接代表沪深300盈利。"
        matched = [n for n in news if e < n["first_open_idx"] <= last]
        row["subsequent_information_count"] = len(matched)
        row["subsequent_nonlogistics_ids"] = [n["id"] for n in matched if n["kind"] != "物流完整周"]
        for n in matched:
            i = n["first_open_idx"]
            marked = gross(e, i)
            end = gross(e, last)
            changes.append({"event_id": event["event_id"], "information_id": n["id"], "kind": n["kind"],
                            "title": n["title"], "known_at": n["known_at"], "first_usable_open": n["first_usable_open"],
                            "open_interval": i-e, "entry_to_information_open_gross_return": marked,
                            "remaining_gross_wealth_change_from_mark": (1+end)/(1+marked)-1,
                            "original_gross_return20": end, "available_before_original_entry": False,
                            "return_decomposition_is_causal": False, "url": n.get("url"), "source_urls": n.get("source_urls")})
        for i in range(e, last + 1):
            paths.append({"event_id": event["event_id"], "date": market.date.iloc[i], "open_interval": i-e,
                          "gross_return_from_entry": gross(e,i), "entry_open": market.open.iloc[e], "mark_open": market.open.iloc[i],
                          "dividend_entitlement_per_share": float(market.dividend.iloc[e+1:i+1].sum()) if i>e else 0.})
        assert abs(gross(e,last)-event["gross_return20"]) < 1e-12
        assert pd.Timestamp(latest["known_at"]) < opens.iloc[e]
        rows.append(row)
    save("event_contexts.json",rows)
    save("holding_period_revisions.json",changes)
    save("event_paths.json",paths)
    pd.DataFrame(rows).to_parquet(OUT / "event_contexts.parquet",index=False)
    save("necessary_checks.json",{"checked_at":datetime.now(TZ).isoformat(),"status":"PASS_FIXED_CLOCK_AND_RETURN_RECONCILIATION", "all_events_retained":len(rows),"PMI_before_entry":True,"original_20d_gross_reproduced":len(rows),"information_updates":len(changes),"new_accounts":0,"new_parameters":0})
    for r in rows:
        print(r['event_date'],r['pmi_month'],r['new_orders'],r['services_activity'],f"此前20日{r['prior_20d_total_return']:+.2%}",f"其后20日净{r['net_return20']:+.2%}",r['subsequent_nonlogistics_ids'])
    print('持有期信息更新',len(changes),'条；没有生成交易信号。')


def finish():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    rows = read(OUT / "event_contexts.json")
    changes = read(OUT / "holding_period_revisions.json")
    sources = read(OUT / "new_source_manifest.json")
    checks = read(OUT / "necessary_checks.json")
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.5, 8.8), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor("#f8fafc")
    for ax in (left, right):
        ax.set_facecolor("#f8fafc")
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.grid(axis="x" if ax is left else "y", alpha=.18)
        ax.set_axisbelow(True)
    yy = np.arange(len(rows))
    values = np.array([r["net_return20"] * 100 for r in rows])
    left.barh(yy, values, color=["#0b766e" if v >= 0 else "#ad4d48" for v in values], height=.62)
    left.set_yticks(yy, [r["event_date"] for r in rows])
    left.invert_yaxis()
    left.axvline(0, color="#67717e", linewidth=.8)
    left.set_xlim(-11, 11.5)
    for y, v in zip(yy, values):
        left.text(v + (.16 if v >= 0 else -.16), y, f"{v:+.2f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    left.set_title("全部14次事件保留\n公告后首个中国开盘起，固定20日成本后收益", loc="left", pad=17, fontsize=12)
    left.set_xlabel("事件收益（%）；不是完整账户年化收益")
    e = int(market.index[market.date.eq(pd.Timestamp("2022-05-09"))][0])
    last = int(market.index[market.date.eq(pd.Timestamp("2022-08-08"))][0])
    dates = market.date.iloc[e:last + 1]
    dividend = market.dividend.iloc[e:last + 1].copy()
    dividend.iloc[0] = 0
    wealth = (market.open.iloc[e:last + 1].to_numpy() + dividend.cumsum().to_numpy()) / market.open.iloc[e] * 100
    right.plot(dates, wealth, color="#174b70", linewidth=2.1)
    right.axhline(100, color="#87919f", linewidth=.8, linestyle="--")
    annotations = [
        ("2022-05-17", "5/16恢复安排公开\n保守可用开盘5/17", (10, 105)),
        ("2022-06-01", "5月PMI：订单48.2\n保守可用开盘6/1", (-36, -112)),
        ("2022-06-06", "6/6下一就业事件起点", (4, 31)),
        ("2022-07-01", "6月PMI：订单50.4\n保守可用开盘7/1", (-89, 32)),
        ("2022-07-11", "7/11下一就业事件起点", (3, -47))]
    for day, label, offset in annotations:
        i = int(market.index[market.date.eq(pd.Timestamp(day))][0]) - e
        right.scatter([pd.Timestamp(day)], [wealth[i]], s=26, color="#174b70", zorder=4)
        right.annotate(label, (pd.Timestamp(day), wealth[i]), xytext=offset, textcoords="offset points", fontsize=9,
                       bbox={"boxstyle": "round,pad=.25", "facecolor": "#f8fafc", "edgecolor": "none", "alpha": .95},
                       arrowprops={"arrowstyle": "-", "color": "#718091", "linewidth": .8})
    right.set_ylim(97.3, max(wealth) + 3.3)
    right.xaxis.set_major_locator(mdates.MonthLocator())
    right.xaxis.set_major_formatter(mdates.DateFormatter("%m月"))
    right.set_title("5—7月连续价格路径\n从5/9开盘计毛权益，起点=100", loc="left", pad=17, fontsize=12)
    right.set_ylabel("原持仓毛权益；含现金分红权益")
    fig.suptitle("指数整体：国内约束变化、信息确认与价格先后", x=.07, ha="left", fontsize=19, color="#173348", y=.96)
    fig.text(.07, .9, "历史原因发现｜日线口径｜已观察样本｜消息到达与涨跌相邻，不等于因果份额", fontsize=11, color="#5b6673")
    fig.text(.07, .045, "左：固定就业事件的既有成本后结果。右：已知结果后的阶段说明，不是新增交易规则。\n政策与PMI采用保守公开时钟；事件分组不构成独立验证。", fontsize=10, color="#5b6673")
    fig.subplots_adjust(left=.09, right=.975, top=.80, bottom=.15, wspace=.32)
    figure = OUT / "国内约束更新_指数历史路径.png"
    fig.savefig(figure, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)

    url = {r["id"]: r["url"] for r in sources}
    context = {r["event_id"]: r for r in rows}
    changed = {(r["event_id"], r["information_id"]): r for r in changes}
    def revision(event, information):
        return changed[(event, information)]
    may_plan = revision("JOBS_20220506", "SH_PLAN_20220516")
    may_pmi = revision("JOBS_20220506", "PMI_2022-05")
    june_pmi = revision("JOBS_20220603", "PMI_2022-06")
    july = context["JOBS_20220708"]
    report = OUT / "历史发现_国内约束更新与指数定价.md"
    def local_link(label, path):
        return f"[{label}](<{Path(path).absolute().as_posix()}>)"
    lines = [
        "# 国内约束更新与指数定价：14次固定就业事件的历史对照", "",
        "本轮发现是：同一类美国就业信息之后，沪深300所面对的国内经营约束与政策信息不同，而且在20日观察窗口内继续变化。2022年5月至7月的连续对照说明，经济读数水平、约束缓解的消息和价格已经反映的部分，必须分别观察。现有证据支持这条研究分解，尚未证明可以据此择时。", "",
        "研究对象始终为510300代表的指数整体：共同经营环境、整体风险折价和资金约束。没有通过挑选某家公司的故事解释整个指数。保留全部14个就业发布日期及原来的5/20日结果，没有增加过滤条件、调参或拼接账户。", "",
        "**目标状态：完整账户成本后夏普1.2仍未实现。本轮没有新增策略或完整账户，夏普与年化收益记为未计算。**", "",
        "## 一、为什么约束在5月发生变化", "",
        f"5月16日的上海官方说明同时提供了政策方向和它的前提：疫情传播压力已下降，随后才安排分阶段恢复活动；6月的恢复目标仍附带风险可控条件。因此，上游变化是防控条件和可执行约束发生变化，不能把它概括为一个无条件的‘宽松’标签。[上海5月16日说明]({url['SH_PLAN_20220516']})", "",
        f"5月25日全国会议给出的政策背景则是就业、工业生产、用电和货运等在3—4月明显走弱，要求稳经济措施尽快落地。这里的先后关系是经营冲击加深、政策响应加强、执行逐步推进；政策增强本身也携带经济弱的信息。[5月25日全国会议]({url['PACKAGE_IMPLEMENTATION_20220525']})", "",
        f"5月29日公布取消企业复工审批审核、恢复通勤的安排；5月30日公布恢复公共交通与机动车通行的安排，均仍有防疫范围限制。它们直接改变的是能否返岗、运输和经营，不直接证明最终需求和利润已经恢复。[5月29日实施说明]({url['SH_APPROVAL_20220529']})；[5月30日通行安排]({url['SH_MOBILITY_20220530']})", "",
        "从指数层面看，这些事实可能通过两个渠道共同作用：经营受阻概率下降，使未来现金流预期改善；极端损失的不确定性减轻，使风险折价收缩。这是与事实相容的机制解释。本研究没有估计两个渠道各贡献多少涨幅，也没有得到当时市场对政策的统一预期。", "",
        "## 二、同一段上涨中，早期与后期所知道的事实不同", "",
        f"5月9日原事件起点，最新公开的4月制造业新订单42.6、生产44.4、服务业活动40.0，均很弱。5月16日恢复安排公开后的保守可用开盘5月17日，原持仓已上涨{may_plan['entry_to_information_open_gross_return']:+.2%}。这份后来出现的材料无法作为5月9日已经具备的买入理由。", "",
        f"5月PMI公布后的保守可用开盘6月1日，原持仓毛收益已为{may_pmi['entry_to_information_open_gross_return']:+.2%}；到原定6月7日结束，之后毛权益又变动{may_pmi['remaining_gross_wealth_change_from_mark']:+.2%}。这是价格的时间拆分，不能说前一段或后一段由PMI造成。", "",
        f"6月6日下一次原事件起点，市场已经能够看到5月多数恢复安排和5月PMI。该窗口到7月4日原20日成本后收益为{context['JOBS_20220603']['net_return20']:+.2%}。到了6月PMI公布后的7月1日，原持仓毛收益已经是{june_pmi['entry_to_information_open_gross_return']:+.2%}，到窗口结束的剩余毛权益变化反而为{june_pmi['remaining_gross_wealth_change_from_mark']:+.2%}。", "",
        f"7月11日再进入下一次事件时，最新新订单已到50.4，服务业活动54.3，比5月、6月入场时更强；但之后20日成本后收益为{july['net_return20']:+.2%}。这足以否定‘已公布的经济读数越好，随后指数收益必然越高’这种简单解释；不足以证明经济读数转强后就应卖出。", "",
        "5月的低读数与后续上涨、7月的较高读数与后续下跌都来自已经看过结果的样本。不能据此发明‘低于50买入’或‘改善但未过50买入’的规则；PMI相关规则此前已有失败记录，本轮不重启。", "",
        "## 三、政策内容要分清作用对象", "",
        "| 事实或措施 | 直接缓解什么 | 从这里到指数盈利仍缺什么 |",
        "|---|---|---|",
        "| 恢复通勤、运输与经营许可 | 生产和交付的可达性、供给约束 | 员工实际到岗、订单履约、最终需求与收入 |",
        "| 留抵退税、社保缓缴、贷款延期 | 现金流时间压力、再融资和违约风险 | 新订单、盈利能力、持续现金创造；退税或延期不能统一当作新增利润 |",
        "| 专项债发行使用提速与项目融资 | 已有预算与融资向项目支出的转化 | 实际拨付、工程工作量、需求传导与回款；额度不能等同新增收入 |",
        "| 减免购置税等需求支持 | 特定需求的成交成本 | 真实销量、替代或提前消费、利润与指数覆盖范围 |", "",
        f"以上措施的事实依据为政策方向、具体文本和发布会解释；第三列是本研究对传导环节的区分，不是官方保证的效果。[5月政策方向]({url['PACKAGE_DIRECTION_20220523']})；[具体措施全文]({url['PACKAGE_TEXT_20220531']})；[5月31日发布会Q1—Q2]({url['PACKAGE_PRESS_20220531']})", "",
        "全国完整物流周数据提供了反例：截至5月22日道路货运和揽收分别环比+5.09%、+4.23%，但截至5月29日为−0.82%、+3.92%；截至6月5日为−9.16%、+5.53%，该周还包含端午假期。生产许可改善、物流数量和需求恢复并不同步。周环比也混有假期、供给与基数影响，不能当作纯需求。原始日份材料与周聚合来自既有物流研究，本轮不另选表现较好的周。", "",
        f"6月6日的后续发布会仍报告社会面感染与反弹风险。这里保守按网页6月7日公开、6月8日开盘才可用；它与恢复安排一起保留，不能删掉不支持单向修复的事实。[实施后反对证据]({url['SH_REMAINING_20220606']})", "",
        "## 四、14次事件的完整对照", "",
        "表中PMI均为原入场前已公开的最新当月原文。此前20日是入场前一交易日收盘向前20个间隔的总回报；之后20日为原固定开盘事件窗口成本后收益，两列计价端点不同，不能直接相减。", "",
        "| 就业发布日期 | 原入场 | 已知PMI统计月 | 新订单 | 服务业活动 | 此前20日总回报 | 之后20日净收益 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['event_date']} | {r['entry_date'][:10]} | {r['pmi_month']} | {r['new_orders']:.1f} | {r['services_activity']:.1f} | {r['prior_20d_total_return']:+.2%} | {r['net_return20']:+.2%} |")
    lines += ["", "5月、6月、7月均属于既有‘就业人数超预期、工资未超预期’类别，收益仍出现反转。不过它们的外部条件也并非完全相同，不能把收益差全部归给国内因素。11月7日起点之后还出现防疫政策与美国通胀等新信息，12月则同时出现进一步放开和履约压力；单条就业因子的20日收益混入多次信息更新。", "",
              "## 五、信息到达时价格已经走过多少", "",
              "以下为5月和6月两个连续事件的政策/PMI时点。先后毛权益变化采用乘法关系，不能直接相加；第二段也不是重新入场、另扣交易成本后的策略收益。", "",
              "| 原起点 | 新信息 | 保守可用开盘 | 原持仓到此毛收益 | 此后至原20日终点毛权益变化 |",
              "|---|---|---|---:|---:|"]
    for r in changes:
        if r["event_id"] in ["JOBS_20220506", "JOBS_20220603"] and r["kind"] != "物流完整周":
            lines.append(f"| {context[r['event_id']]['entry_date'][:10]} | {r['title']} | {r['first_usable_open'][:10]} | {r['entry_to_information_open_gross_return']:+.2%} | {r['remaining_gross_wealth_change_from_mark']:+.2%} |")
    lines += ["", "## 六、本轮边界与可继续使用的结论", "",
              "可继续使用的是原因拆分：先问共同经营或资金约束为何变化，再问变化是否成为可执行措施，最后对照当时已知的信息与价格。PMI读数、政策意图、经营许可、实物数量和价格反应是不同层次，不能互相代替。", "",
              "本轮新增8份有界官方材料，属于两个相关政策根事件的方向、落实与反对证据，不能数成8次独立机会。14次就业事件和61条持有期信息连接全部保留，但日历并不覆盖每条历史新闻，也没有统一政策预期分布。‘之后出现的利好解释之前买入’和‘好数据必然带来好收益’这两种推断均不成立。", "",
              "5月材料是为解释已知上涨追加的，整项研究属于历史发现，不能称为独立验证或确定性因果识别。供求和风险折价解释可以并存。没有新信号、没有新账户，也没有用20日收益冒充200,000元完整账户指标。原事件收益沿用10万元名义订单、100份整数手、佣金与滑点/T+1/现金分红口径；它不证明原开盘数量事前已知可成交。", "",
              "必要核对只保留时间和数值两类：14条入场前PMI均早于原入场，14条原20日毛收益重新计算一致；其余结果复用既有研究。没有新增长周期参数检验。", "",
              "本支线完成后，不把这些解释再改写成美国新闻买入变种或PMI阈值变种。后续研究必须先明确尚未被既有结果覆盖的指数整体问题；需要新的可判别证据，不能仅增加一个漂亮的历史故事。", "",
              "## 文件", "",
              f"- {local_link('运行脚本', ROOT / 'research/historical_index_domestic_revision_v1.py')}：已保存资料后依次运行compute、finish。",
              f"- {local_link('全部事件与入场前国内状态', OUT / 'event_contexts.json')}。",
              f"- {local_link('61条信息连接与原持仓毛权益拆分', OUT / 'holding_period_revisions.json')}。",
              f"- {local_link('既有信息及本轮新增信息的公开时钟', OUT / 'public_information_clock.json')}。",
              f"- {local_link('8份材料及来源形式', OUT / 'new_source_manifest.json')}：保留原HTTP失败。1份原始网页、6份网页工具文字、1份仅Q1—Q2搜索摘录，不将摘录称为完整网页。",
              f"- {local_link('研究范围', OUT / 'protocol.json')}：明确已观察样本和不加规则。",
              f"- {local_link('最低限度的时间与收益核对', OUT / 'necessary_checks.json')}。",
              f"- {local_link('既有就业研究', JOBS / 'result.json')}；{local_link('既有PMI原文与时钟', DOMESTIC / 'monthly_pmi.json')}；{local_link('既有完整物流周', LOGISTICS / 'weekly.parquet')}。",
              f"- {local_link('可分享的静态研究图', figure)}。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    summary = "14次就业事件与入场前国内状态及61条后续信息连接。5月恢复安排在原入场后出现，6月较强PMI确认时原窗口已上涨10.46%，之后毛权益-0.93%；7月已知读数更强却录得20日净-5.80%。明确经营约束变化、数据确认和价格反映的先后，尚无新可交易优势。"
    save("result.json", {"study_id": "510300_HISTORICAL_INDEX_DOMESTIC_REVISION_V1", "completed_at": datetime.now(TZ).isoformat(),
                         "status": "COMPLETED_HISTORICAL_CAUSE_DISCOVERY_NO_NEW_SIGNAL", "classification": "PROGRESS_INDEX_DOMESTIC_CONSTRAINT_REVISION_AND_PRICE_TIMING",
                         "report": report.relative_to(ROOT).as_posix(), "figure": figure.relative_to(ROOT).as_posix(),
                         "release_count": len(rows), "holding_period_information_connections": len(changes), "new_source_count": len(sources),
                         "new_policy_root_count": 2, "policy_roots_are_independent": False, "new_candidates": 0, "new_full_accounts": 0,
                         "net_sharpe": None, "net_cagr": None, "goal_achieved": False, "orders_authorized": False,
                         "causal_share_identified": False, "independent_validation": False,
                         "discovery": summary, "necessary_check_status": checks["status"],
                         "next_historical_question": None,
                         "continuation_boundary": "本支线完成。禁止从已观察窗口倒推新过滤器；没有尚未覆盖的新证据时，不继续扩展本支线。",
                         "report_status": "PENDING_REVIEW", "figure_visually_reviewed": False,
                         "limitations": ["价格已观察，5月材料为事后解释追加", "政策方向不等于统一预期差", "保守公开时钟并非当时实时接收凭证", "历史新闻覆盖不完整", "相关政策分层不独立", "事件窗口不等于完整账户"]})
    print("已生成全部事件对照、研究报告与图。完整账户目标尚未实现。")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['collect','compute','finish'])
    args=parser.parse_args()
    {'collect':collect,'compute':compute,'finish':finish}[args.action]()


if __name__ == '__main__':
    main()
