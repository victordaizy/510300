"""把指数降波拆为成分风险和涨跌抵消；固定观察篮子，不训练新策略。"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_constituent_risk_transmission_v10"
INPUTS = {
    "monthly.csv": "reports/research/510300_macro_transmission_context_v4/results/104个月_多层证据与原后续路径.csv",
    "lpr_cases.csv": "reports/research/510300_policy_information_timing_v9/results/七个调查偏差病例_完整时序与传导.csv",
    "market.csv": "reports/research/510300_macro_transmission_context_v4/inputs/market.csv",
    "returns.parquet": "data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/classified_constituent_returns_remediated.parquet",
    "membership.parquet": "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet",
    "weights.parquet": "data/raw/constituents/000300_historical_weights.parquet",
    "industry.parquet": "data/curated/510300_structural_equity_risk_premium_engine_v1/csi300_component_monthly_state.parquet",
    "old_dispersion_result.json": "reports/research/510300_csi300_return_dispersion_volatility_v1r1_result.json",
    "weight_source_manifest.json": "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/pit_weight_source_evidence_manifest.json",
}


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def save(relative, value):
    (OUT / relative).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    for name in ["inputs", "results", "figures", "code"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    if (OUT / "freeze.json").exists():
        return
    protocol = {"at": now(), "study_id": "510300_CONSTITUENT_RISK_TRANSMISSION_V10",
        "previous_turn_classification": "PROGRESS：V9已完成信息先后与信用背景的核对。",
        "question": "降波是成分股普遍缓和，还是涨跌抵消与共同涨跌程度改变？如何与既有货币及信贷背景连接？",
        "universe": "全部104个月货币观察点，另列原V9按调查选出的7个LPR病例，两类绝不合并统计。",
        "fixed_windows": "每个观察点回看25个交易日，当前20日与五日前20日；300只观察日成分均需25日有效收益。",
        "membership": "使用既有官方换样重建的观察日300只成分；回看这同一组股票，不用未来成员。",
        "equal_basket": "300只每日固定等权，解释成分普遍性；不宣称复制510300。",
        "reference_weight_basket": "使用观察日前最近且不超过62日的完整300只权重快照，归一化到1后在两个回看窗口固定；不填缺失权重、不以部分样本重标权重。首版来源未经认证，仅作结构观察。",
        "missing_rule": "缺失或未解决的收益保留未知；官方核对停牌收益复用既有分类；不把供应商缺失填零。",
        "identity1": "固定权重R=P−G；D_net²=252*mean(min(R,0)²)，D_gross²=252*mean(G²)，offset=D_gross²−D_net²。",
        "identity2": "Var(R)=A+B*rho；A=sum(w²*sigma²)，B=(sum(w*sigma))²−A，rho为波动和权重加权的平均相关。变化按对称代数分解为成分波动贡献dA+平均rho*dB与共同变动贡献平均B*drho。",
        "stock_contributions": "当前窗口各股方差贡献w*Cov(ri,R)与下行平方贡献252*mean(w*ri*R*I(R<0))；行业仅聚合该观察，不解释政策因果。",
        "no_lookahead_claim": "旧行情和结果已经看过；行业与权重是参考快照，不能晋升为已认证的历史可交易信号。",
        "outcomes": "月度E0/E1与LPR原20日标签各自保留，仅供完整路径阅读；不分组搜索后续收益、不调参、不复活旧离散度候选。",
        "reporting": "全部观察状态及缺失列出；正文沿用已看过的2020-06、2021-01、2024-08、2025-07、2025-08和七个LPR案例；任何新例只作当期恒等式说明，注明选择理由。",
        "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False}
    save("protocol.json", protocol)
    (ROOT / "config/510300_constituent_risk_transmission_v10.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    receipts = []
    for name, relative in INPUTS.items():
        source = ROOT / relative
        shutil.copy2(source, OUT / "inputs" / name)
        receipts.append({"name": name, "source": relative, "sha256": digest(source), "bytes": source.stat().st_size})
    save("freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "inputs": receipts})


def origins():
    monthly = pd.read_csv(OUT / "inputs/monthly.csv")
    lpr = pd.read_csv(OUT / "inputs/lpr_cases.csv")
    rows = []
    for old in monthly.to_dict("records"):
        row = {"origin_id": "monthly_" + old["stat_month"], "origin_kind": "MONTHLY", "origin_month": old["stat_month"],
            "observation_date": old["observation_date"], "snapshot_at": old["snapshot_at"], "money_month": old["stat_month"],
            "spread_pp": old["spread_pp"], "delta3_spread_pp": old["delta3_spread_pp"], "training_regime": old["training_regime"],
            "macro_state": old["arithmetic_mechanism_group"], "orders": old["orders_first_release_value"],
            "corporate_long_ytd_yoy_yi": old["loan_corporate_long_ytd_yoy_change_yi"], "household_long_ytd_yoy_yi": old["loan_household_long_ytd_yoy_change_yi"],
            "macro_status": old["macro_status"], "parent_E0_20_return": old["E0_20_return"], "parent_E1_20_return": old["E1_20_return"],
            "parent_E0_20_status": old["E0_20_status"], "parent_E1_20_status": old["E1_20_status"],
            "past_return20": old["past_return20"], "past_return60": old["past_return60"]}
        rows.append(row)
    for old in lpr.to_dict("records"):
        rows.append({"origin_id": "lpr_" + old["month"], "origin_kind": "LPR_CASE", "origin_month": old["month"],
            "observation_date": old["first_reaction_date"], "snapshot_at": old["decision_at"], "money_month": old["known_money_stat_month"],
            "spread_pp": old["known_money_spread_pp"], "delta3_spread_pp": old["known_money_delta3_spread_pp"],
            "orders": old["known_orders_first_release_value"], "corporate_long_ytd_yoy_yi": old["known_credit_corporate_long_ytd_yoy_change_yi"],
            "household_long_ytd_yoy_yi": old["known_credit_household_long_ytd_yoy_change_yi"], "parent_LPR_20_return": old["target20"],
            "past_return20": old["pre_return20"], "past_return60": old["pre_return60"]})
    return pd.DataFrame(rows)


def metrics(returns, weight):
    weighted = returns * weight
    net = weighted.sum(axis=1)
    gains = np.maximum(weighted, 0).sum(axis=1)
    losses = -np.minimum(weighted, 0).sum(axis=1)
    stock_var = returns.var(axis=0, ddof=1) * 252
    stock_down2 = np.minimum(returns, 0).square().mean(axis=0) * 252 if hasattr(returns, "square") else (np.minimum(returns, 0) ** 2).mean(axis=0) * 252
    variance = float(net.var(ddof=1) * 252)
    sigma = np.sqrt(stock_var)
    a = float(np.sum(weight ** 2 * stock_var))
    b = float(np.sum(weight * sigma) ** 2 - a)
    rho = (variance - a) / b if b > 1e-15 else np.nan
    net_down2 = float(np.mean(np.minimum(net, 0) ** 2) * 252)
    gross_down2 = float(np.mean(losses ** 2) * 252)
    covariance_contribution = weight * ((returns - returns.mean(axis=0)) * (net - net.mean())[:, None]).sum(axis=0) / (len(net) - 1) * 252
    downside_contribution = (weighted * net[:, None] * (net < 0)[:, None]).mean(axis=0) * 252
    values = {"variance": variance, "rv": np.sqrt(variance), "net_down2": net_down2, "net_down": np.sqrt(net_down2),
        "gross_down2": gross_down2, "gross_down": np.sqrt(gross_down2), "offset_energy": gross_down2 - net_down2,
        "offset_fraction": 1 - net_down2 / gross_down2 if gross_down2 > 0 else np.nan,
        "weighted_individual_down2": float(weight @ stock_down2), "weighted_individual_variance": float(weight @ stock_var),
        "weighted_average_stock_vol": float(weight @ sigma), "variance_diagonal_A": a, "variance_pair_scale_B": b,
        "weighted_mean_correlation": rho, "positive_20d_fraction": float(np.mean(np.prod(1 + returns, axis=0) > 1)),
        "positive_20d_weight": float(weight @ (np.prod(1 + returns, axis=0) > 1))}
    return values, {"returns": net, "gains": gains, "losses": losses, "stock_down2": stock_down2, "stock_variance": stock_var,
                    "covariance_contribution": covariance_contribution, "downside_contribution": downside_contribution}


def build():
    freeze()
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("本轮分解已保存，不重复覆盖。")
    all_origins = origins()
    all_origins.to_csv(OUT / "inputs/origins.csv", index=False, encoding="utf-8-sig")
    market = pd.read_csv(OUT / "inputs/market.csv")
    market.date = pd.to_datetime(market.date)
    calendar = pd.DatetimeIndex(market.date)
    etf_r = (market.close + market.dividend) / market.close.shift() - 1
    raw = pd.read_parquet(OUT / "inputs/returns.parquet", columns=["date", "symbol", "daily_total_shareholder_return", "return_is_usable"])
    raw.loc[~raw.return_is_usable, "daily_total_shareholder_return"] = np.nan
    assert not raw.duplicated(["date", "symbol"]).any()
    panel = raw.pivot(index="date", columns="symbol", values="daily_total_shareholder_return")
    membership = pd.read_parquet(OUT / "inputs/membership.parquet")
    member_sets = {pd.Timestamp(date): sorted(part.symbol.tolist()) for date, part in membership.groupby("membership_date")}
    weights = pd.read_parquet(OUT / "inputs/weights.parquet")
    weight_sets = {pd.Timestamp(date): part.set_index("con_code").weight for date, part in weights.groupby("trade_date")}
    weight_dates = pd.DatetimeIndex(sorted(weight_sets))
    industry = pd.read_parquet(OUT / "inputs/industry.parquet", columns=["origin", "stock_code", "industry_name"])
    industry_sets = {pd.Timestamp(date): part.set_index("stock_code").industry_name for date, part in industry.groupby("origin")}
    industry_dates = pd.DatetimeIndex(sorted(industry_sets))
    results, daily, stocks, gaps = [], [], [], []
    for meta in all_origins.to_dict("records"):
        date = pd.to_datetime(meta["observation_date"])
        base = dict(meta)
        if pd.isna(date) or date not in calendar or date not in member_sets:
            for basket in ["EQUAL", "REFERENCE_WEIGHT"]:
                results.append({**base, "basket": basket, "status": "NO_VIEW_DATE_OR_MEMBERSHIP"})
            continue
        members = member_sets[date]
        assert len(members) == len(set(members)) == 300
        end = int(calendar.get_loc(date))
        sessions = calendar[end - 24:end + 1]
        table = panel.reindex(index=sessions, columns=members)
        missing = table.isna()
        if missing.any().any():
            for code in table.columns[missing.any()]:
                gaps.append({"origin_id": meta["origin_id"], "symbol": code, "missing_dates": "|".join(str(t.date()) for t in sessions[missing[code]])})
            for basket in ["EQUAL", "REFERENCE_WEIGHT"]:
                results.append({**base, "basket": basket, "status": "NO_VIEW_INCOMPLETE_300_MEMBER_RETURNS", "missing_stock_days": int(missing.to_numpy().sum())})
            continue
        price = table.to_numpy(float)
        actual = etf_r.iloc[end - 24:end + 1].to_numpy(float)
        base.update(etf_down_current=np.sqrt(np.mean(np.minimum(actual[-20:], 0) ** 2) * 252),
                    etf_down_previous=np.sqrt(np.mean(np.minimum(actual[:20], 0) ** 2) * 252),
                    etf_rv_current=float(actual[-20:].std(ddof=1) * np.sqrt(252)), etf_rv_previous=float(actual[:20].std(ddof=1) * np.sqrt(252)))
        pos = int(weight_dates.searchsorted(date, side="left")) - 1
        wdate = weight_dates[pos] if pos >= 0 else None
        reference = weight_sets[wdate].reindex(members) if wdate is not None else pd.Series(np.nan, index=members)
        valid_weight = wdate is not None and (date - wdate).days <= 62 and reference.notna().all() and (reference > 0).all()
        ipos = int(industry_dates.searchsorted(date, side="right")) - 1
        idate = industry_dates[ipos] if ipos >= 0 else None
        industries = industry_sets[idate].reindex(members).fillna("行业缺失") if idate is not None and (date - idate).days <= 35 else pd.Series("行业缺失", index=members)
        for basket in ["EQUAL", "REFERENCE_WEIGHT"]:
            if basket == "REFERENCE_WEIGHT" and not valid_weight:
                results.append({**base, "basket": basket, "status": "NO_VIEW_INCOMPLETE_OR_STALE_WEIGHT", "weight_date": wdate, "weight_members_missing": int(reference.isna().sum())})
                continue
            weight = np.full(300, 1 / 300) if basket == "EQUAL" else reference.to_numpy(float) / reference.sum()
            old, old_parts = metrics(price[:20], weight)
            current, current_parts = metrics(price[-20:], weight)
            row = {**base, "basket": basket, "status": "COMPLETE_DIAGNOSTIC_ONLY", "member_count": 300,
                "weight_date": wdate if basket == "REFERENCE_WEIGHT" else pd.NaT,
                "reference_weight_sum_percent": reference.sum() if basket == "REFERENCE_WEIGHT" else np.nan,
                "industry_date": idate}
            row.update({"current_" + k: v for k, v in current.items()})
            row.update({"previous_" + k: v for k, v in old.items()})
            row["variance_change_stock_vol_component"] = current["variance_diagonal_A"] - old["variance_diagonal_A"] + .5 * (current["weighted_mean_correlation"] + old["weighted_mean_correlation"]) * (current["variance_pair_scale_B"] - old["variance_pair_scale_B"])
            row["variance_change_correlation_component"] = .5 * (current["variance_pair_scale_B"] + old["variance_pair_scale_B"]) * (current["weighted_mean_correlation"] - old["weighted_mean_correlation"])
            row["stock_fraction_downside_improved"] = float(np.mean(current_parts["stock_down2"] < old_parts["stock_down2"] - 1e-12))
            row["stock_weight_downside_improved"] = float(weight @ (current_parts["stock_down2"] < old_parts["stock_down2"] - 1e-12))
            row["net_downside_falls"] = current["net_down2"] < old["net_down2"] - 1e-12
            row["gross_downside_energy_rises"] = current["gross_down2"] > old["gross_down2"] + 1e-12
            row["individual_downside_energy_rises"] = current["weighted_individual_down2"] > old["weighted_individual_down2"] + 1e-12
            row["net_relief_with_gross_pressure_increase"] = row["net_downside_falls"] and row["gross_downside_energy_rises"]
            row["return_correlation_with_etf20"] = float(np.corrcoef(current_parts["returns"], actual[-20:])[0, 1])
            row["daily_basket_etf_rmse20"] = float(np.sqrt(np.mean((current_parts["returns"] - actual[-20:]) ** 2)))
            row["rv_identity_error"] = current["variance"] - current["variance_diagonal_A"] - current["variance_pair_scale_B"] * current["weighted_mean_correlation"]
            row["variance_change_identity_error"] = current["variance"] - old["variance"] - row["variance_change_stock_vol_component"] - row["variance_change_correlation_component"]
            assert abs(row["rv_identity_error"]) < 1e-10 and abs(row["variance_change_identity_error"]) < 1e-10
            assert abs(current_parts["downside_contribution"].sum() - current["net_down2"]) < 1e-10
            results.append(row)
            full_weighted = price * weight
            for k, session in enumerate(sessions):
                daily.append({"origin_id": meta["origin_id"], "basket": basket, "date": session, "relative_session": k - 24,
                    "net_return": full_weighted[k].sum(), "positive_contribution": np.maximum(full_weighted[k], 0).sum(),
                    "negative_magnitude": -np.minimum(full_weighted[k], 0).sum(), "etf_return": actual[k]})
            for k, code in enumerate(members):
                stocks.append({"origin_id": meta["origin_id"], "basket": basket, "symbol": code, "industry": industries.iloc[k], "weight": weight[k],
                    "current_down2": current_parts["stock_down2"][k], "previous_down2": old_parts["stock_down2"][k],
                    "current_variance": current_parts["stock_variance"][k], "previous_variance": old_parts["stock_variance"][k],
                    "variance_contribution": current_parts["covariance_contribution"][k], "downside_contribution": current_parts["downside_contribution"][k],
                    "past20_total_return": np.prod(1 + price[-20:, k]) - 1})
    full = pd.DataFrame(results)
    full.to_csv(OUT / "results/111个观察点_篮子风险与原后续路径.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(daily).to_csv(OUT / "results/固定篮子_25日上涨下跌分项.csv", index=False, encoding="utf-8-sig")
    stock_frame = pd.DataFrame(stocks)
    stock_frame.to_parquet(OUT / "results/成分股风险与贡献.parquet", index=False)
    grouped = stock_frame.groupby(["origin_id", "basket", "industry"], as_index=False).agg(stock_count=("symbol", "count"), weight=("weight", "sum"), variance_contribution=("variance_contribution", "sum"), downside_contribution=("downside_contribution", "sum"))
    grouped.to_csv(OUT / "results/行业风险贡献_仅观察篮子.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gaps, columns=["origin_id", "symbol", "missing_dates"]).to_csv(OUT / "results/未解决成分收益缺口.csv", index=False, encoding="utf-8-sig")
    summary = {"at": now(), "status": "CONSTITUENT_RISK_IDENTITIES_BUILT", "origins": len(all_origins),
        "status_counts": full.groupby(["origin_kind", "basket", "status"]).size().rename("count").reset_index().to_dict("records"),
        "daily_rows": len(daily), "stock_rows": len(stocks), "new_models": 0, "new_accounts": 0,
        "independent_validation": False, "goal_achieved": False, "historical_weights_are_diagnostic_only": True}
    summary["monthly_observations"] = []
    for basket in ["EQUAL", "REFERENCE_WEIGHT"]:
        part = full[(full.origin_kind == "MONTHLY") & (full.basket == basket) & (full.status == "COMPLETE_DIAGNOSTIC_ONLY")]
        falls = part[part.net_downside_falls.astype(bool)]
        summary["monthly_observations"].append({"basket": basket, "complete_months": len(part), "net_downside_falls": len(falls),
            "net_falls_gross_rises": int(falls.gross_downside_energy_rises.sum()),
            "net_falls_individual_rises": int(falls.individual_downside_energy_rises.sum()),
            "net_falls_less_than_half_stocks_improve": int((falls.stock_fraction_downside_improved < .5).sum()),
            "median_return_correlation_etf": part.return_correlation_with_etf20.median()})
    save("results/build_receipt.json", summary)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(clean(summary), ensure_ascii=False))


if __name__ == "__main__":
    build()
