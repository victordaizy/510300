"""固定LPR公告节点后的股债联合反应；先做事件毛收益检验。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research import mechanism_odds_open_contract_v1 as shared

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_lpr_joint_response_v1"
HOLD = 10
PRIMARY = "YIELD_DOWN_EQUITY_UP"
CONTROLS = ["ALL_RELEASES", "EQUITY_UP_ONLY", "YIELD_DOWN_ONLY"]
QUADRANTS = [PRIMARY, "YIELD_DOWN_EQUITY_DOWN", "YIELD_UP_EQUITY_UP", "YIELD_UP_EQUITY_DOWN", "ZERO_RESPONSE"]
PERIODS = {"FULL": ("2019-08-20", "2025-12-31"), "EARLY": ("2019-08-20", "2022-12-31"), "LATE": ("2023-01-01", "2025-12-31")}
save, read, digest, now = shared.save, shared.read, shared.digest, shared.now


def classify(rate_change, equity_return):
    if not np.isfinite([rate_change, equity_return]).all():
        return "NO_VIEW"
    if rate_change == 0 or equity_return == 0:
        return "ZERO_RESPONSE"
    return ("YIELD_DOWN" if rate_change < 0 else "YIELD_UP") + ("_EQUITY_UP" if equity_return > 0 else "_EQUITY_DOWN")


def response_indices(dates, release_date):
    dates = pd.DatetimeIndex(dates)
    date = pd.Timestamp(release_date)
    before = int(dates.searchsorted(date, side="left")-1)
    observation = int(dates.searchsorted(date, side="right"))
    return before, observation, observation+1


def gross_return(market, entry, exit_idx):
    earned = float(market.dividend.iloc[entry+1:exit_idx+1].sum())
    return (float(market.open.iloc[exit_idx])+earned)/float(market.open.iloc[entry])-1


def directional_limit(market, i, side):
    row = market.iloc[i]
    basis = float(row.previous_close-row.dividend)
    lower = math.floor(basis*.9/.001+.5+1e-9)*.001
    upper = math.floor(basis*1.1/.001+.5+1e-9)*.001
    return bool(row.open >= upper-1e-9) if side > 0 else bool(row.open <= lower+1e-9)


def build_events(market, rates, releases):
    dates = pd.DatetimeIndex(market.date)
    rates = rates.set_index("date").cgb_1y
    log_returns = np.log((market.close+market.dividend)/market.close.shift())
    result = []
    for release in releases.to_dict("records"):
        published = pd.Timestamp(release["release_date"])
        base, observation, entry = response_indices(dates, published)
        row = {**release, "base_idx": base, "observation_idx": observation, "entry_idx": entry,
            "planned_exit_idx": entry+HOLD, "status": "NO_VIEW_INPUT", "quadrant": "NO_VIEW",
            "gross_return": np.nan, "stress_proportional_proxy_return": np.nan}
        if base < 0 or observation >= len(market):
            result.append(row)
            continue
        base_date, obs_date = dates[base], dates[observation]
        row.update(base_date=base_date, observation_date=obs_date,
            decision_at=obs_date.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=23, minutes=59))
        if base_date not in rates.index or obs_date not in rates.index:
            result.append(row)
            continue
        rate_change = float(rates.loc[obs_date]-rates.loc[base_date])
        equity_return = float(np.expm1(log_returns.iloc[base+1:observation+1].sum()))
        row.update(rate_before=float(rates.loc[base_date]), rate_after=float(rates.loc[obs_date]),
            rate_change_pp=rate_change, equity_response=equity_return,
            quadrant=classify(rate_change, equity_return))
        if entry >= len(market):
            row["status"] = "CENSORED_NO_ENTRY_PRICE"
            result.append(row)
            continue
        row["entry_date"] = dates[entry]
        if directional_limit(market, entry, 1):
            row.update(status="UNFILLED_UPPER_LIMIT", gross_return=0., stress_proportional_proxy_return=0., exit_idx=entry, exit_date=dates[entry])
            result.append(row)
            continue
        end = entry+HOLD
        while end < len(market) and directional_limit(market, end, -1):
            end += 1
        if end >= len(market):
            row["status"] = "CENSORED_EXIT_AFTER_SAMPLE"
            result.append(row)
            continue
        payoff = gross_return(market, entry, end)
        row.update(status="MATURE", exit_idx=end, exit_date=dates[end], exit_delay_sessions=end-entry-HOLD,
            gross_return=payoff, stress_proportional_proxy_return=payoff-.0028)
        assert row["decision_at"] < dates[entry].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=30)
        assert base_date < published < obs_date < dates[entry] <= dates[end]
        result.append(row)
    return pd.DataFrame(result)


def mask(frame, name):
    if name == "ALL_RELEASES":
        return pd.Series(True, index=frame.index)
    if name == "EQUITY_UP_ONLY":
        return frame.equity_response.gt(0)
    if name == "YIELD_DOWN_ONLY":
        return frame.rate_change_pp.lt(0)
    return frame.quadrant.eq(name)


def statistics(events):
    rows = []
    admitted = events.loc[events.gross_return.notna()].copy()
    for period, (start, end) in PERIODS.items():
        part = admitted.loc[admitted.release_date.between(start, end)]
        for name in CONTROLS+QUADRANTS:
            sample = part.loc[mask(part, name)]
            values = sample.gross_return
            rows.append({"period": period, "group": name, "n": len(sample),
                "mean_gross_return": values.mean(), "median_gross_return": values.median(),
                "mean_stress_proportional_proxy_return": sample.stress_proportional_proxy_return.mean(),
                "positive_fraction": float(values.gt(0).mean()) if len(values) else np.nan,
                "minimum": values.min(), "maximum": values.max(),
                "standard_error_iid_descriptive_only": values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.nan})
    return pd.DataFrame(rows)


def frozen_check():
    for row in read(OUT / "freeze.json")["files"]:
        assert digest(ROOT / row["path"]) == row["sha256"], row["path"]


def freeze():
    assert not (OUT / "freeze.json").exists(), "本版本已冻结"
    source = read(OUT / "source_finalization.json")
    assert source["status"] == "PASS_77_MONTHLY_RELEASE_DATES"
    assert read(OUT / "prefreeze_tests.json")["exit_code"] == 0
    copies = {
        "market.parquet": ROOT / "reports/research/510300_mechanism_odds_open_contract_v1/inputs/market.parquet",
        "dividends.csv": ROOT / "reports/research/510300_mechanism_odds_open_contract_v1/inputs/dividends.csv",
        "cgb_yields.parquet": ROOT / "data/raw/macro/china_government_bond_yields_daily.parquet",
        "authority_snapshot.json": ROOT / "config/510300_existing_data_training_mandate_v1.json"}
    for name, path in copies.items():
        target = OUT / "inputs" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    protocol = {**read(OUT / "design_registration.json"), "frozen_at": now(),
        "registered_code": "research/lpr_joint_response_v1.py", "source_finalization": source,
        "release_clock_admission": "公告目录和原文日期；两处英文日期错误以同次中文官方原文纠正，不按交易结果选日期。",
        "bond_source": "既有中债官方历史查询所得一年期国债收益率，按日期精确匹配；不是交易成交价。没有历史首版发布快照，作为历史重建筛查。",
        "publication_caution": "使用观察日23:59的信息截止，下一交易日开盘入场；实际首次获取尚未证明，不宣称前向实绩。",
        "execution_proxy": "当日开盘到方向涨停不模拟买入，该事件毛收益0且不扣交易代理费；卖出方向跌停则延后至首个允许开盘，末端未成熟保持CENSORED。",
        "annual_metrics_before_account": "NOT_COMPUTED", "account_status_before_screen": "NOT_RUN",
        "source_scope": "2019年8月至2025年12月77次定期公告；12月末标签若未成熟不填零。",
        "economic_boundary": "不把日频联动称为高频货币政策意外识别，不将短端估值曲线等同政策利率。"}
    save(OUT / "protocol.json", protocol)
    paths = [Path(__file__), ROOT / "research/mechanism_odds_open_contract_v1.py",
        ROOT / "scripts/collect_lpr_joint_response_source_v1.py", ROOT / "tests/test_lpr_joint_response_v1.py",
        OUT / "design_registration.json", OUT / "protocol.json", OUT / "prefreeze_tests.json",
        OUT / "lpr_releases.csv", OUT / "source_finalization.json", OUT / "lpr_original_notices.json"]
    paths += list((OUT / "inputs").glob("*"))+list((OUT / "sources").glob("*"))
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths if p.is_file()]})
    print("77次公告、一个主要联合反应和固定10日标签已冻结；尚未计算新事件收益。", flush=True)


def run():
    frozen_check()
    save(OUT / "run_started.json", {"at": now(), "stage": "FIXED_GROSS_SCREEN"})
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    market = market.loc[market.date.le("2025-12-31")].reset_index(drop=True)
    rates = pd.read_parquet(OUT / "inputs/cgb_yields.parquet")
    rates.date = pd.to_datetime(rates.date)
    assert rates.date.is_unique and np.isfinite(rates.cgb_1y).all()
    releases = pd.read_csv(OUT / "lpr_releases.csv")
    assert len(releases) == 77 and releases.release_date.is_unique
    events = build_events(market, rates, releases)
    events.to_parquet(OUT / "events.parquet", index=False)
    events.to_csv(OUT / "全部公告与事件收益.csv", index=False, encoding="utf-8-sig")
    stats = statistics(events)
    stats.to_csv(OUT / "group_statistics.csv", index=False, encoding="utf-8-sig")
    full = stats.loc[stats.period.eq("FULL")].set_index("group")
    partial = stats.loc[stats.group.eq(PRIMARY)].set_index("period")
    tests = {"primary_mean_above_cost_proxy": full.loc[PRIMARY,"mean_gross_return"] > .0028,
             "primary_mean_above_all_releases": full.loc[PRIMARY,"mean_gross_return"] > full.loc["ALL_RELEASES","mean_gross_return"],
             "early_positive": partial.loc["EARLY","mean_gross_return"] > 0,
             "late_positive": partial.loc["LATE","mean_gross_return"] > 0}
    passed = all(tests.values())
    admitted = events.loc[events.gross_return.notna()].copy().reset_index(drop=True)
    n = len(admitted)
    rng = np.random.default_rng(20260928)
    indices = ((rng.integers(0,n,size=(2000,math.ceil(n/4)))[:,:,None]+np.arange(4))%n).reshape(2000,-1)[:,:n]
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=indices.astype(np.int16))
    values = admitted.gross_return.to_numpy(float)
    selected = admitted.quadrant.eq(PRIMARY).to_numpy(bool)
    sampled_returns, sampled_selected = values[indices], selected[indices]
    counts = sampled_selected.sum(axis=1)
    conditional = np.divide((sampled_returns*sampled_selected).sum(axis=1),counts,out=np.full(2000,np.nan),where=counts>0)
    increments = conditional-sampled_returns.mean(axis=1)
    interval = {"method":"4个月循环区块，2000次，给定历史条件区间",
        "primary_mean_95_interval": np.nanquantile(conditional,[.025,.975]).tolist(),
        "primary_minus_all_95_interval": np.nanquantile(increments,[.025,.975]).tolist(),
        "selection_bias_adjusted": False}
    save(OUT / "uncertainty.json", interval)
    # 验证观察时点之后的行情变化不影响已形成的分组。
    origin = int(events.observation_idx.iloc[len(events)//2])
    altered = market.copy()
    altered.loc[origin+1:,["open","high","low","close","previous_close"]] *= 1.4
    altered_events = build_events(altered,rates,releases)
    prior = events.observation_idx.le(origin)
    pd.testing.assert_series_equal(events.loc[prior,"quadrant"],altered_events.loc[prior,"quadrant"])
    overlap_pairs = []
    mature = events.loc[events.status.eq("MATURE")].sort_values("entry_idx")
    for a,b in zip(mature.iloc[:-1].itertuples(),mature.iloc[1:].itertuples()):
        if b.entry_idx < a.exit_idx:
            overlap_pairs.append([a.release_date,b.release_date])
    result = {"at": now(), "study_id": "510300_LPR_JOINT_RESPONSE_V1",
        "status": "PASS_GROSS_SCREEN_ACCOUNT_REQUIRED" if passed else "REJECTED_FIXED_GROSS_SCREEN_NO_PARAMETER_RESCUE",
        "release_count": len(events), "statuses": events.status.value_counts().to_dict(),
        "primary": full.loc[PRIMARY].to_dict(), "all_releases": full.loc["ALL_RELEASES"].to_dict(),
        "stage_two_gate": tests, "overlapping_event_pairs": overlap_pairs,
        "account_status": "REQUIRED_NEXT_STAGE" if passed else "NOT_RUN_GROSS_SCREEN_FAILED",
        "net_sharpe": None, "net_cagr": None, "maximum_drawdown": None,
        "goal_achieved": False, "independent_forward_observations": 0,
        "new_fits": 0, "new_accounts": 0, "parameter_searches": 0,
        "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "result.json", result)
    save(OUT / "verification_receipt.json", {"at": now(), "status":"PASS_FROZEN_SOURCES_CLOCKS_AND_CAUSAL_TRIGGER_PREFIX",
        "announcements":77,"events":len(events),"causal_trigger_prefix_checks":1,"new_accounts":0,
        "boundary":"来源与计算检查不等于经济通过、独立验证或真实成交。"})
    rows=["|固定组别|成熟事件数|十日平均毛收益|减0.28%后的筛查代理|胜率|", "|---|---:|---:|---:|---:|"]
    for name in [PRIMARY]+CONTROLS:
        r=full.loc[name]
        rows.append(f"|{name}|{int(r['n'])}|{r.mean_gross_return:.2%}|{r.mean_stress_proportional_proxy_return:.2%}|{r.positive_fraction:.1%}|")
    conclusion="通过继续研究的毛收益初筛，仍须完整账户检验" if passed else "没有通过事先登记的毛收益初筛，本版本停止，完整账户未运行"
    text=f"""LPR公告后股债联合反应{conclusion}。夏普1.2目标仍未实现；本轮没有计算账户夏普，不能把事件收益当作完整账户表现。

本轮使用2019年8月至2025年12月77次实际定期公告，保留利率不变月份。以公告日之前最后一个A股收盘为基准，观察至公告日之后首个完整交易日收盘，再从下一开盘计算固定10个开盘间隔的收益。反应窗口中的涨幅没有计入可赚收益。

{chr(10).join(rows)}

主要规则固定为一年期中债国债收益率下降且510300含分红财富上升。其他象限只作描述，不能在主规则失败后替代。前段主组平均毛收益{partial.loc['EARLY','mean_gross_return']:.2%}，后段{partial.loc['LATE','mean_gross_return']:.2%}；主组相对全部公告的平均毛收益差95%区间[{interval['primary_minus_all_95_interval'][0]:.2%}, {interval['primary_minus_all_95_interval'][1]:.2%}]。区间不是对历史多重选择的校正。

0.28%只是双边比例佣金与滑点的粗筛代理，不包括最低佣金、价位、账户仓位及路径；表中该列不是实际账户净收益。末端未成熟事件保留为CENSORED，未运行账户的夏普、年化和回撤保持空值。

官方英文公告中2019年8月及10月的日期冲突已依据中文官方原文修正，旧英文原文保留。2024年7月带有发布时间调整的公告已纳入，没有因标题不同漏掉月份。来源为本次回取官方档案，不声称历史真实first_seen。

机制只提出可检验解释：政策消息可能同时反映融资条件与经济前景，但日频股债符号不是高频意外识别，也无法排除同期其他新闻。本轮不将LPR不变公告称为降息冲击，不将中债估值曲线称为政策利率或可成交债券报价。

旧政策目录、资金利率模型和本次固定检验分别保留。未搜索债券期限、观察日、持有期或阈值；没有订单、计划采集恢复或ZIP交付。

方法动机：[Jarociński与Karadi原论文](https://www.aeaweb.org/articles?id=10.1257/mac.20180090)；公告目录：[人民银行LPR官方档案](https://www.pbc.gov.cn/en/3688229/3688335/3730276/3883798/19e15ae1-4.html)。文献不提供本规则对510300有效的证明。
"""
    (OUT/"研究结论.md").write_text(text,encoding="utf-8")
    print(json.dumps(shared.clean(result),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="LPR公告后股债联合反应固定研究")
    parser.add_argument("stage",choices=["freeze","run"])
    args=parser.parse_args()
    {"freeze":freeze,"run":run}[args.stage]()
