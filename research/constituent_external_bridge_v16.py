"""以不再投资分红的持股账本，拆解两个病例的成分收益及下行贡献。"""
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_constituent_external_bridge_v16"
NAMES = {"600519.SH": "贵州茅台", "601318.SH": "中国平安", "000858.SZ": "五粮液", "600036.SH": "招商银行",
         "600276.SH": "恒瑞医药", "300750.SZ": "宁德时代", "600109.SH": "国金证券"}


def save_csv(name, frame):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("成分结果已完成，不重复覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha256((OUT / "protocol.json").read_bytes()).hexdigest() == frozen["protocol_sha256"]
    for item in frozen["inputs"]:
        assert sha256((OUT / "inputs" / item["name"]).read_bytes()).hexdigest() == item["sha256"]
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    original = pd.read_csv(OUT / "inputs/daily_paths.csv")
    segments = pd.read_csv(OUT / "inputs/news_segments.csv")
    selected = pd.read_csv(OUT / "inputs/固定前五名公司.csv")
    market = pd.read_csv(OUT / "inputs/market.csv").set_index("date")
    raw = pd.read_parquet(OUT / "inputs/returns.parquet")
    raw.date = pd.to_datetime(raw.date)
    members = pd.read_parquet(OUT / "inputs/membership.parquet")
    members.membership_date = pd.to_datetime(members.membership_date)
    weights = pd.read_parquet(OUT / "inputs/weights.parquet")
    weights.trade_date = pd.to_datetime(weights.trade_date)
    industry = pd.read_parquet(OUT / "inputs/industry.parquet", columns=["origin", "stock_code", "industry_name"])
    industry.origin = pd.to_datetime(industry.origin)
    # 只修复本轮结果账本的停牌日；旧收益原件和旧研究结论保持不变。
    suspension_src = OUT / "sources/600109_20201013_复牌_巨潮.pdf"
    with pdfplumber.open(suspension_src) as document:
        announcement = "".join(document.pages[0].extract_text().split())
    assert "2020年9月21日开市起停牌" in announcement and "2020年10月13日" in announcement and "开市起复牌" in announcement
    k = (raw.symbol == "600109.SH") & (raw.date == pd.Timestamp("2020-10-12"))
    assert k.sum() == 1 and not raw.loc[k, "return_is_usable"].iloc[0]
    before = raw[(raw.symbol == "600109.SH") & (raw.date == pd.Timestamp("2020-09-18"))].iloc[0]
    after = raw[(raw.symbol == "600109.SH") & (raw.date == pd.Timestamp("2020-10-13"))].iloc[0]
    assert before.return_is_usable and after.return_is_usable
    assert before.unadjusted_close == after.previous_unadjusted_close == after.daily_pre_close == 15.29
    override = {"symbol": "600109.SH", "date": "2020-10-12", "old_state": raw.loc[k, "constituent_return_state"].iloc[0],
                "new_state": "OFFICIAL_SUSPENSION_OUTCOME_RECONSTRUCTION", "marked_price": 15.29,
                "basis": "原公告确认9月21日起停牌、10月13日复牌；复牌日原始前收等于停牌前最后有效收盘，无除权除息价格断点。",
                "source": suspension_src.name, "source_sha256": sha256(suspension_src.read_bytes()).hexdigest(),
                "source_published_date": "2020-10-13", "role": "历史结果复原，非起点可用信息，未修改冻结父数据"}
    raw.loc[k, ["previous_unadjusted_close", "unadjusted_close"]] = 15.29
    raw.loc[k, "official_suspension"] = True
    raw.loc[k, ["cash_distribution_per_pre_event_share", "subscription_cash_outflow_per_pre_event_share", "daily_total_shareholder_return"]] = 0.0
    raw.loc[k, "post_to_pre_share_ratio"] = 1.0
    raw.loc[k, "return_is_usable"] = True
    save_json("results/唯一停牌补证_不修改父数据.json", override)
    summaries, stocks, ledgers, industries, stock_segments = [], [], [], [], []
    max_account_error = 0.0
    for case in ["2020-08", "2022-08"]:
        m = monthly.loc[case]
        day = pd.Timestamp(m.observation_date)
        symbols = sorted(members.loc[members.membership_date == day, "symbol"].tolist())
        assert len(symbols) == len(set(symbols)) == 300
        wdate = weights.loc[weights.trade_date < day, "trade_date"].max()
        wraw = weights[weights.trade_date == wdate].set_index("con_code").weight.reindex(symbols)
        assert wraw.notna().all() and (wraw > 0).all() and (day-wdate).days <= 62
        w = wraw.to_numpy() / wraw.sum()
        idate = industry.loc[industry.origin <= day, "origin"].max()
        inds = industry[industry.origin == idate].set_index("stock_code").industry_name.reindex(symbols).fillna("行业缺失")
        assert (day-idate).days <= 35
        p = original[original.stat_month == case].sort_values("day_number")
        dates = pd.to_datetime(p.date).tolist()
        assert len(dates) == 20
        rows = raw[raw.symbol.isin(symbols) & raw.date.isin(dates)].copy()
        assert len(rows) == 6000 and rows.return_is_usable.all()
        save_csv(f"{case}_实际采用的6000行成分原字段.csv", rows)
        def panel(field):
            return rows.pivot(index="date", columns="symbol", values=field).reindex(index=dates, columns=symbols).to_numpy()
        close = panel("unadjusted_close").astype(float)
        previous = panel("previous_unadjusted_close").astype(float)
        dividends = panel("cash_distribution_per_pre_event_share").astype(float)
        ratio = panel("post_to_pre_share_ratio").astype(float)
        subscription = panel("subscription_cash_outflow_per_pre_event_share").astype(float)
        suspension = panel("official_suspension").astype(bool)
        actual_r = panel("daily_total_shareholder_return").astype(float)
        assert np.isfinite(previous[0]).all() and np.isfinite(dividends).all() and np.isfinite(ratio).all()
        assert (previous[0] > 0).all() and (ratio > 0).all() and not np.any(subscription)
        for t in range(20):
            missing = ~np.isfinite(close[t])
            assert np.all(suspension[t, missing])
            close[t, missing] = previous[t, missing]
        assert np.isfinite(close).all()
        recomputed_r = (close * ratio + dividends - subscription) / previous - 1
        assert np.max(np.abs(recomputed_r - actual_r)) < 1e-10
        shares = np.ones(300) / previous[0]
        cash = np.zeros(300)
        wealth = np.zeros((21, 300))
        wealth[0] = 1
        shares_hist, cash_hist = [], []
        for t in range(20):
            cash += shares * (dividends[t] - subscription[t])
            shares *= ratio[t]
            wealth[t+1] = shares * close[t] + cash
            shares_hist.append(shares.copy())
            cash_hist.append(cash.copy())
        delta = np.diff(wealth, axis=0) * w
        basket = wealth @ w
        net = np.diff(basket) / basket[:-1]
        normalized = delta / basket[:-1, None]
        assert np.max(np.abs(normalized.sum(axis=1) - net)) < 1e-12
        var_contribution = ((normalized - normalized.mean(axis=0)) * (net - net.mean())[:, None]).sum(axis=0) / 19 * 252
        downside = (normalized * (net * (net < 0))[:, None]).mean(axis=0) * 252
        variance = float(net.var(ddof=1)*252)
        down2 = float(np.minimum(net, 0).dot(np.minimum(net, 0))/20*252)
        max_account_error = max(max_account_error, abs(var_contribution.sum()-variance), abs(downside.sum()-down2))
        top = selected[selected.stat_month == case].set_index("symbol")
        for j, symbol in enumerate(symbols):
            # 19日口径按各股原入场收盘重新设1元；不把第一天已有分红再带入新持有人收益。
            units19, cash19 = 1/close[0, j], 0.0
            for t in range(1, 20):
                cash19 += units19 * (dividends[t, j] - subscription[t, j])
                units19 *= ratio[t, j]
            return19 = units19 * close[-1, j] + cash19 - 1
            stocks.append({"stat_month": case, "symbol": symbol, "name": NAMES.get(symbol, symbol), "industry_reference": inds.iloc[j],
                           "weight_date": str(wdate.date()), "weight": w[j], "initial_close": previous[0, j],
                           "exit_close": close[-1, j], "cash_distributions_per_initial_yuan": cash_hist[-1][j],
                           "stock_return20_cc": wealth[-1, j]-1, "contribution20_pp": w[j]*(wealth[-1, j]-1)*100,
                           "stock_return19_entryclose": return19, "contribution19_pp": w[j]*return19*100,
                           "variance_contribution": var_contribution[j], "downside_squared_contribution": downside[j],
                           "selected_before_outcome": symbol in top.index,
                           "reference_rank": int(top.loc[symbol, "reference_rank"]) if symbol in top.index else None})
            for t, date in enumerate(dates):
                ledgers.append({"stat_month": case, "date": str(date.date()), "day_number": t+1, "symbol": symbol,
                                "industry_reference": inds.iloc[j], "initial_weight": w[j], "marked_close": close[t, j],
                                "units_per_initial_stock_yuan": shares_hist[t][j], "cash_per_initial_stock_yuan": cash_hist[t][j],
                                "stock_wealth": wealth[t+1, j], "daily_contribution_to_initial_basket": delta[t,j],
                                "daily_contribution_to_basket_return": normalized[t,j], "basket_wealth": basket[t+1],
                                "basket_daily_return": net[t]})
            for segment in segments[segments.stat_month == case].itertuples():
                mask = (pd.DatetimeIndex(dates) >= segment.start_date) & (pd.DatetimeIndex(dates) <= segment.end_date)
                stock_segments.append({"stat_month": case, "segment": segment.segment, "symbol": symbol,
                                       "industry_reference": inds.iloc[j], "contribution_pp": delta[mask,j].sum()*100})
        s = pd.DataFrame(stocks)
        s = s[s.stat_month == case]
        grouped = s.groupby("industry_reference", as_index=False).agg(stock_count=("symbol", "size"), weight=("weight", "sum"),
                            contribution20_pp=("contribution20_pp", "sum"), contribution19_pp=("contribution19_pp", "sum"),
                            variance_contribution=("variance_contribution", "sum"), downside_squared_contribution=("downside_squared_contribution", "sum"))
        grouped.insert(0, "stat_month", case)
        industries.extend(grouped.to_dict("records"))
        etf_initial = market.loc[m.observation_date, "close"]
        q = market.loc[p.date]
        etf20 = (q.close.iloc[-1] + q.dividend.sum()) / etf_initial - 1
        etf19 = (q.close.iloc[-1] + q.dividend.iloc[1:].sum()) / q.close.iloc[0] - 1
        top5 = s[s.selected_before_outcome]
        summaries.append({"stat_month": case, "origin_date": m.observation_date, "entry_date": m.E0_20_entry_date,
                          "exit_date": m.E0_20_exit_date, "members": len(s), "weight_date": str(wdate.date()),
                          "raw_weight_sum_percent": wraw.sum(), "industry_reference_date": str(idate.date()),
                          "basket20_cc_return_percent": (basket[-1]-1)*100, "etf20_cc_return_percent": etf20*100,
                          "basket_minus_etf20_pp": (basket[-1]-1-etf20)*100, "etf_original_E0_percent": m.E0_20_return*100,
                          "basket19_entryclose_return_percent": s.contribution19_pp.sum(), "etf19_entryclose_return_percent": etf19*100,
                          "basket_rv20_percent": variance**.5*100, "basket_downside20_percent": down2**.5*100,
                          "positive_stock_count": int((s.stock_return20_cc > 0).sum()),
                          "positive_stock_initial_weight_percent": s.loc[s.stock_return20_cc>0, "weight"].sum()*100,
                          "top5_initial_weight_percent": top5.weight.sum()*100, "top5_contribution_pp": top5.contribution20_pp.sum(),
                          "top5_downside_squared_share_percent": top5.downside_squared_contribution.sum()/down2*100,
                          "unknown_industry_weight_percent": s.loc[s.industry_reference == "行业缺失", "weight"].sum()*100,
                          "actual_etf_attribution": False})
    stocks, ledgers, stock_segments = pd.DataFrame(stocks), pd.DataFrame(ledgers), pd.DataFrame(stock_segments)
    save_csv("两个病例_完整篮子与510300同口径比较.csv", pd.DataFrame(summaries))
    save_csv("600个公司病例_全部收益及风险贡献.csv", stocks)
    save_csv("全部行业_收益与下行平方贡献.csv", pd.DataFrame(industries))
    save_csv("事前前五名_收益与风险贡献.csv", stocks[stocks.selected_before_outcome].sort_values(["stat_month", "reference_rank"]))
    ledgers.to_parquet(OUT / "results/12000个成分日_持股现金账本.parquet", index=False)
    save_csv("全部成分_消息分段贡献.csv", stock_segments)
    save_csv("全部行业_消息分段贡献.csv", stock_segments.groupby(["stat_month", "segment", "industry_reference"], as_index=False).contribution_pp.sum())
    receipt = {"status": "BUILT_COMPLETE_REFERENCE_BASKETS", "cases": 2, "stock_case_rows": len(stocks), "stock_daily_rows": len(ledgers),
               "local_suspension_overrides": 1, "parent_data_modified": False, "maximum_risk_identity_error": max_account_error,
               "goal_achieved": False, "independent_validation": False, "new_models": 0, "new_accounts": 0}
    save_json("results/build_receipt.json", receipt)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(stocks[stocks.selected_before_outcome][["stat_month", "name", "weight", "stock_return20_cc", "contribution20_pp"]].to_string(index=False))
    print(pd.DataFrame(industries).sort_values(["stat_month", "contribution20_pp"]).to_string(index=False))


if __name__ == "__main__":
    main()
