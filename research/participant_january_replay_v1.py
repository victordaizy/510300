"""2026年1月大资金行为线索回放；身份验证和可观察行为分开。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import requests

from participant_identity_clock_v1 import ROOT, OUT, OLD, HEADERS, now, dump, sha


def initialize():
    path = OUT / "january_replay_plan.json"
    if path.exists():
        raise ValueError("一月回放计划已经登记")
    dump(path, {
        "recorded_at": now(), "user_steering": "很多信息延后披露只能验证，重点预测大资金会怎么想怎么操作，再跟随；用户指出2026年1月国家队放量减仓。",
        "scope": "用户指定的已知历史案例，描述性回放，不能冒充事前策略或独立验证。",
        "holding_reports_role": "只作迟到的验证和当时已经公开的背景，不等待实名确认才研究供给压力。",
        "case_start": "2025-12-01", "case_end": "2026-02-27", "include_all_daily_points": True,
        "daily_share_clock": "历史统计日期的份额作图放所属日，但明确不证明当日公开。可决策栏只使用有日期证据的当时报导或官方公告，不用假定延迟覆盖原始缺口。",
        "behavior_proxies": ["当天成交额/严格此前20交易日成交额中位数", "当天含分红回报及收盘在日内区间位置", "ETF份额存量变化_不是现金流", "已公开政策的实际方向与生效日"],
        "participant_identity": "有条件推断，不输出未经标定的概率。以已知期末大持有人份额与新总份额形成库存上下界，不把每笔赎回当国家队卖出。",
        "january_known_anchors": ["2025中报2025-08-30公布的两家汇金直接份额", "2026-01-14融资保证金公告", "2026-01-22四季报匿名大持有人", "2026-01-26当时报导的1月23日总份额"],
        "dividend": "保留原价和含分红序列；1月19日除息0.123元。权益登记1月16日，现金1月27日发放。总回报图为研究指数，不是即时可再投资现金账户。",
        "target_for_next_study": "先预测未来资金行为的持续、衰减或反向，再检验在当前承接状态下对未来收益/风险的增量。不是单看资金净流入做多。",
        "future_path_selection": "只画全期日线，不选最优卖点、不计算本例策略收益或据本例冻结阈值。",
        "new_predictive_labels": 0, "new_model_fits": 0, "new_accounts": 0,
    })
    keep = ["date", "etf_open", "etf_high", "etf_low", "etf_close", "etf_volume", "etf_amount"]
    p = ROOT / "data/features/510300_registered_factor_dataset.parquet"
    pd.read_parquet(p, columns=keep).to_parquet(OUT / "inputs/daily_price_volume.parquet", index=False)
    p = ROOT / "data/raw/flow/510300_fund_share_daily_tushare.parquet"
    (OUT / "inputs/daily_fund_shares.parquet").write_bytes(p.read_bytes())
    q = next(x for x in json.loads((OLD / "quarterly_download_v1_1_result.json").read_text(encoding="utf-8"))["rows"] if x["period"] == "2025Q4")
    src = ROOT / q["raw_path"]
    if sha(src) != q["sha256"]:
        raise ValueError("四季报原始文件哈希不一致")
    (OUT / "raw/2025Q4_original.pdf").write_bytes(src.read_bytes())
    dump(OUT / "receipts/2025Q4_original.json", q)
    dump(OUT / "evidence/用户预测优先范围修正.json", {
        "recorded_at": now(), "original_source_plan_preserved": True,
        "change": "已完成的28份持有人报告作为验证附录。优先工作改为一月行为线索和可反驳的未来行为预测；实名未确认不阻止概率性研究。",
        "no_new_strategy_from_case": True,
    })
    print("一月行为回放范围已保存，保留全期日线和信息时间区别", flush=True)


def fetch_context():
    sources = [
        ("margin_rule_20260114", "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/specific/margin/c/c_20260114_10805174.shtml", False),
        ("margin_reason_20260114", "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260114_10805178.shtml", False),
        ("dividend_20260112", "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-01-12/510300_20260112_VTCZ.pdf", False),
        ("cnstock_20260126", "https://paper.cnstock.com/html/2026-01/26/content_2174084.htm", True),
        ("private_intentions_202408", "https://static.simuwang.com/Uploads/Documents/Research/Manager-Confidence-Index/2024/202408150005.pdf", True),
        ("ssf_2024_annual", "https://www.ssf.gov.cn/portal/yljjgl/webinfo/2025/09/1761903919491454.htm", False),
        ("szse_margin_clock", "https://investor.szse.cn/institute/video/studio/t20100818_538365.html", False),
    ]
    records = []
    for key, url, local_only in sources:
        folder = OUT / ("raw_local_only" if local_only else "raw")
        folder.mkdir(exist_ok=True)
        info = {"key": key, "url": url, "retrieved_at": now(), "local_only_full_text": local_only}
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            file = folder / (key + (".pdf" if r.content.startswith(b"%PDF") else ".html"))
            file.write_bytes(r.content)
            info.update(http_status=r.status_code, bytes=len(r.content), sha256=sha(file), raw_path=file.relative_to(OUT).as_posix())
            info["status"] = "HTTP200_SAVED" if r.status_code == 200 else "ACCESS_FAILED"
        except requests.RequestException as exc:
            info.update(status="ACCESS_FAILED", error=str(exc))
        dump(OUT / "receipts" / (key + ".json"), info)
        records.append(info)
        print(f"背景来源：{key} {info['status']}", flush=True)
    dump(OUT / "receipts/context_sources.json", records)


def build():
    d = pd.read_parquet(OUT / "inputs/daily_price_volume.parquet")
    s = pd.read_parquet(OUT / "inputs/daily_fund_shares.parquet")
    m = pd.read_parquet(OUT / "inputs/market.parquet")
    for frame in (d, s, m):
        frame["date"] = pd.to_datetime(frame.date)
    d = d.sort_values("date")
    d["amount_to_prior20_median"] = d.etf_amount / d.etf_amount.shift(1).rolling(20).median()
    d["close_location"] = (d.etf_close - d.etf_low) / (d.etf_high - d.etf_low).replace(0, np.nan)
    s = s.sort_values("date")
    s["units_change"] = s.fund_shares.diff()
    s["share_stat_date_is_publication_time"] = False
    frame = d.merge(s[["date", "fund_shares", "units_change", "share_stat_date_is_publication_time"]], on="date", how="left", validate="one_to_one")
    frame = frame.merge(m[["date", "close", "dividend", "total_simple", "wealth"]], on="date", validate="one_to_one")
    if not np.allclose(frame.etf_close, frame.close, atol=1e-9):
        raise ValueError("本轮两份历史价格的收盘价不一致")
    a = pd.read_parquet(OUT / "facts/持有结构与公开时钟.parquet")
    known = a[a.period_end.eq("2025-06-30")].iloc[0]
    assert known.huijin_direct_exact
    known_units = known.huijin_direct_lower
    frame["prior_disclosed_two_huijin_units"] = known_units
    frame["minimum_reduction_since_20250630"] = (known_units - frame.fund_shares).clip(lower=0)
    case = frame[frame.date.between("2025-12-01", "2026-02-27")].copy()
    base = frame[frame.date.eq("2025-12-31")].iloc[0]
    case["total_return_rebased100"] = 100 * case.wealth / base.wealth
    case["raw_price_rebased100"] = 100 * case.close / base.close
    case.to_csv(OUT / "facts/2026年1月行为回放全日线.csv", index=False, encoding="utf-8-sig")
    case.to_parquet(OUT / "facts/2026年1月行为回放全日线.parquet", index=False)
    def point(date):
        return frame.loc[frame.date.eq(date)].iloc[0]
    j13, j28, j19, j16, j23, j26 = [point(x) for x in ["2026-01-13", "2026-01-28", "2026-01-19", "2026-01-16", "2026-01-23", "2026-01-26"]]
    summary = {
        "case_daily_points": len(case), "price_start": str(case.date.min().date()), "price_end": str(case.date.max().date()),
        "known_huijin_snapshot_at_january": {"period_end": "2025-06-30", "published": "2025-08-30", "direct_units": known_units, "units_100m": known_units/1e8},
        "jan15_amount_cny_100m": float(point("2026-01-15").etf_amount/1e8),
        "jan16_amount_cny_100m": float(j16.etf_amount/1e8),
        "jan15_amount_ratio20": float(point("2026-01-15").amount_to_prior20_median),
        "jan16_amount_ratio20": float(j16.amount_to_prior20_median),
        "jan19_dividend_per_unit": float(j19.dividend), "jan19_raw_return": float(j19.close/j16.close-1),
        "jan19_total_return": float(j19.total_simple),
        "jan14_through_jan28_statistical_unit_change_100m": float((j28.fund_shares-j13.fund_shares)/1e8),
        "jan13close_to_jan28close_total_return": float(j28.wealth/j13.wealth-1),
        "jan23_total_units_100m": float(j23.fund_shares/1e8),
        "jan23_lower_bound_reduction_since_prior_disclosure_units_100m": float((known_units-j23.fund_shares)/1e8),
        "jan26_total_units_100m": float(j26.fund_shares/1e8),
        "jan26_lower_bound_reduction_units_100m": float((known_units-j26.fund_shares)/1e8),
        "bound_interpretation": "以2025-06-30已实名持有为起点，至新统计日两家直接登记持有的净减少至少达到该下界；不确定减少日期、二级卖出或直接赎回路线，也不等于退出全部A股。四季报匿名两户金额一致增加连续性证据，不能把匿名字段直接变成实名。",
        "first_share_crossing_stat_date": str(frame.loc[frame.date.ge("2026-01-01") & frame.minimum_reduction_since_20250630.gt(0), "date"].iloc[0].date()),
        "published_jan23_share_fact_date": "2026-01-26", "conservative_available_open": "2026-01-27",
        "quarterly_known_20260123": {"source": "2025Q4_original.pdf", "publication_date": "2026-01-22", "anonymous_institution_units": [37858474974,35654598859], "identity_exact": False},
        "new_prediction_models": 0, "new_accounts": 0, "case_is_ex_post": True,
    }
    dump(OUT / "january_case_summary.json", summary)
    stages = [
        {"asof": "2026-01-14收盘后", "observation": "上调新开融资合约保证金至100%的公告；当日成交额104.65亿元、约2.68倍此前20日中位数。", "inference": "逆周期降温约束已出现，提高对供给增加、上行受压的警惕；国家队卖方身份仍属待验证假说。", "future_test": "未来继续放量且含分红价格推进受阻，则增强持续供给解释；放量后有效突破且成交恢复常态，则减弱该解释。", "first_conservative_open": "2026-01-15"},
        {"asof": "2026-01-16收盘后", "observation": "连续两个交易日成交额约254和259亿元，均约6.46倍前序20日中位数；价格未随放量持续上冲。", "inference": "由一次异常升级为持续供给风险假说。份额快报只有实际公开后才可加入；不把尚未证实的当日份额放入收盘判断。", "future_test": "关注卖压延续还是被承接：份额继续减少而总回报稳定，说明供应量大但价格冲击有限，不能机械扩大看跌幅度。", "first_conservative_open": "2026-01-19"},
        {"asof": "2026-01-22收盘后", "observation": "四季报披露两名机构各持378.58474974、356.54598859亿份，与早前实名持有数量吻合，但本表匿名。", "inference": "对汇金仍为主要库存持有者的判断增加佐证，不伪称取得当日实名成交账。", "future_test": "比较新总份额能否容纳过去大持有人库存；仅能给下界，不能恢复逐日卖单。", "first_conservative_open": "2026-01-23"},
        {"asof": "2026-01-26日期公告上界之后", "observation": "当日报道1月23日总份额644.15亿份，低于此前两家实名持有735.13亿份。", "inference": "至少90.98亿份的库存减少已有强数量约束证据；其发生区间仍受此前持有快照时间限制。", "future_test": "方向判断仍需更新承接：确认卖过，并不意味着后面还有同样大卖量或价格必跌。", "first_conservative_open": "2026-01-27"},
    ]
    dump(OUT / "facts/一月逐节点信息与可反驳推断.json", stages)
    pd.DataFrame(stages).to_csv(OUT / "facts/一月逐节点信息与可反驳推断.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def style():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 11, "axes.titlesize": 12, "figure.facecolor": "#f8fafc", "axes.facecolor": "#ffffff",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "#f8fafc"})


def plots():
    style()
    c = pd.read_parquet(OUT / "facts/2026年1月行为回放全日线.parquet")
    a = pd.read_parquet(OUT / "facts/持有结构与公开时钟.parquet")
    summary = json.loads((OUT / "january_case_summary.json").read_text(encoding="utf-8"))
    x = pd.to_datetime(c.date)
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True, gridspec_kw={"height_ratios": [1.35, 1, 1, 1]})
    fig.subplots_adjust(top=.92, bottom=.1, left=.095, right=.97, hspace=.13)
    fig.suptitle("2026年1月：大额供给出现后，价格是否继续承压？", fontsize=20, fontweight="bold", x=.095, ha="left", y=.975)
    fig.text(.095,.943,"完整日线回放｜统计日与公开日分开｜用户指定的已知案例，不是事前策略验证", color="#536176", fontsize=11)
    axes[0].plot(x, c.total_return_rebased100, color="#1d4ed8", lw=2.4, label="含分红总回报（2025-12-31=100）")
    axes[0].plot(x, c.raw_price_rebased100, color="#94a3b8", lw=1.2, ls="--", label="未校正除息的价格，仅作对照")
    axes[0].set_ylabel("总回报指数")
    axes[0].legend(loc="upper left", fontsize=10, ncol=1)
    ix = c[c.date.eq("2026-01-19")].iloc[0]
    axes[0].annotate("1/19 除息0.123元\n含分红当日约+0.04%", xy=(pd.Timestamp("2026-01-19"),ix.total_return_rebased100), xytext=(pd.Timestamp("2026-02-04"),103.5),
                     arrowprops={"arrowstyle":"->","color":"#64748b"}, fontsize=10, color="#334155", va="top",
                     bbox={"facecolor":"white","edgecolor":"none","alpha":.88})
    axes[1].bar(x, c.etf_amount/1e8, width=1.2, color="#667eea", label="二级市场成交额")
    axes[1].set_ylabel("成交额（亿元）")
    axes[1].text(.015,.85,"1/15、1/16：均约为此前20日成交额中位数的6.46倍",transform=axes[1].transAxes,color="#4338ca",fontsize=10)
    axes[2].plot(x,c.fund_shares/1e8,color="#0f766e",lw=2,label="ETF总份额（所属统计日）")
    axes[2].axhline(summary["known_huijin_snapshot_at_january"]["units_100m"], color="#b45309", ls="--", lw=1.5,label="此前已公开两家汇金直接持有735.13亿份")
    axes[2].set_ylabel("持有份额（亿份）")
    axes[2].legend(loc="lower left", fontsize=9)
    axes[2].annotate("1/23总份额644.15亿份\n该数字见1/26当日报道",xy=(pd.Timestamp("2026-01-23"),644.152877),xytext=(pd.Timestamp("2025-12-05"),545),
                     arrowprops={"arrowstyle":"->","color":"#64748b"},fontsize=10)
    colors = np.where(c.units_change.ge(0),"#059669","#e07858")
    axes[3].bar(x,c.units_change/1e8,color=colors,width=1.2)
    axes[3].axhline(0,color="#94a3b8",lw=.8)
    axes[3].set_ylabel("份额净变化（亿份）")
    axes[3].text(.015,.08,"份额变化不是现金流；曲线按统计日绘制，不声称当日已公开",transform=axes[3].transAxes,color="#64748b",fontsize=10)
    for ax in axes:
        ax.grid(axis="y",alpha=.16)
        ax.axvline(pd.Timestamp("2026-01-14"),color="#b45309",ls=":",alpha=.7)
        ax.axvline(pd.Timestamp("2026-01-26"),color="#0f766e",ls=":",alpha=.7)
        ax.margins(x=.015)
    axes[1].annotate("1/14 保证金公告", xy=(pd.Timestamp("2026-01-14"), 170), xytext=(pd.Timestamp("2025-12-24"), 250),
                     fontsize=9, color="#92400e", arrowprops={"arrowstyle":"->", "color":"#b45309"})
    axes[2].text(pd.Timestamp("2026-01-27"), 855, "1/26 当时报导", fontsize=9, color="#0f766e")
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.text(.095,.047,"1/14—1/28：总份额减少约373.15亿份；同期含分红总回报约-0.99%。\n卖压规模与价格跌幅是两个预测对象：持续赎回也可能被承接，需动态更新。",fontsize=11,color="#334155",linespacing=1.6)
    fig.savefig(OUT / "figures/2026年1月_价格成交份额与政策对照.png",dpi=165)
    plt.close(fig)
    m = pd.read_parquet(OUT / "inputs/market.parquet")
    m.date = pd.to_datetime(m.date)
    dates = pd.to_datetime(a.period_end)
    fig, axes = plt.subplots(3,1,figsize=(14,10),sharex=True,gridspec_kw={"height_ratios":[1,1.2,1.1]})
    fig.subplots_adjust(top=.90,bottom=.11,left=.095,right=.97,hspace=.2)
    fig.suptitle("510300：价格、持有结构与迟到的身份验证",fontsize=19,fontweight="bold",x=.095,ha="left",y=.97)
    fig.text(.095,.927,"28份年报/中报；持有结构按所属期末，仅用于验证。2017年末—2022年中机构口径已扣除联接基金。",color="#64748b",fontsize=10)
    axes[0].plot(m.date,m.wealth,color="#1d4ed8",lw=1.3)
    axes[0].set_ylabel("含分红累计指数")
    vals=[a.institution_excluding_feeder_units/1e8,a.direct_personal_units/1e8,a.feeder_units/1e8]
    axes[1].stackplot(dates,*vals,colors=["#4f6bdc","#fb923c","#10b981"],labels=["机构（不含联接）","个人直接持有","联接基金"],alpha=.9)
    axes[1].legend(loc="upper left",ncol=3,fontsize=10)
    axes[1].set_ylabel("期末份额（亿份）")
    known = a.named_holder_count.gt(0)
    axes[2].plot(dates[known],a.loc[known,"huijin_direct_lower"]/1e8,"o-",color="#7c3aed",ms=3,label="两家汇金直接实名持有下界")
    axes[2].fill_between(dates[known],a.loc[known,"huijin_direct_lower"]/1e8,a.loc[known,"huijin_direct_upper"]/1e8,color="#a78bfa",alpha=.3,label="未进入前十的份额可能区间")
    axes[2].set_ylabel("两家实名主体（亿份）")
    axes[2].legend(loc="upper left",fontsize=9)
    axes[2].axvspan(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-11"), color="#94a3b8", alpha=.12)
    axes[2].annotate("2026中报没有实名表\n灰色区间不填数值",xy=(pd.Timestamp("2026-06-30"),380),xytext=(pd.Timestamp("2021-03-01"),480),arrowprops={"arrowstyle":"->","color":"#64748b"},fontsize=10)
    for ax in axes:
        ax.grid(axis="y",alpha=.16)
        ax.margins(x=.01)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.095,.042,"实际身份披露滞后56—90天。表格只能核对历史库存和约束持有上界，不能把半年末份额提前用于当时交易。\n机构、产品载体和最终出资人属于不同层次；持有变化不等于现金净流入。",fontsize=10,color="#475569",linespacing=1.6)
    fig.savefig(OUT / "figures/510300_资金身份与全历史走势.png",dpi=165)
    plt.close(fig)
    for key, page_indexes in [("20260829_0ed6535396",[59]),("20260331_e1ed47d747",[74]),("20180328_0f55abc3ac",[68]),("2025Q4_original",[12]),("20250830_33fe9c2203",[57,58]),("dividend_20260112",[0,1])]:
        doc=pdfium.PdfDocument(OUT/"raw"/(key+".pdf"))
        for i in page_indexes:
            page=doc[i]
            bitmap=page.render(scale=1.7)
            bitmap.to_pil().save(OUT/"figures"/(key+f"_原文第{i+1}页.png"))
            bitmap.close();page.close()
        doc.close()
    print("两张完整研究图和八张关键原文页已生成",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description="一月资金行为回放")
    p.add_argument("action",choices=["initialize","fetch_context","build","plots"])
    globals()[p.parse_args().action]()
