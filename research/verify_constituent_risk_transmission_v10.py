"""从冻结成分收益直接复核篮子、相关分解与原有标签。"""
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


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def close(x, y):
    assert np.allclose(x, y, atol=1e-10, rtol=1e-10, equal_nan=True)


def main():
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha(OUT / "protocol.json") == freeze["protocol_sha256"]
    for rec in freeze["inputs"]:
        assert sha(OUT / "inputs" / rec["name"]) == rec["sha256"]
    origins = pd.read_csv(OUT / "inputs/origins.csv").set_index("origin_id")
    result = pd.read_csv(OUT / "results/111个观察点_篮子风险与原后续路径.csv")
    source = pd.read_parquet(OUT / "inputs/returns.parquet", columns=["date", "symbol", "return_is_usable", "daily_total_shareholder_return"])
    source = source.set_index(["date", "symbol"])
    members = pd.read_parquet(OUT / "inputs/membership.parquet")
    members["membership_date"] = pd.to_datetime(members.membership_date)
    weight_source = pd.read_parquet(OUT / "inputs/weights.parquet")
    daily = pd.read_csv(OUT / "results/固定篮子_25日上涨下跌分项.csv")
    stock = pd.read_parquet(OUT / "results/成分股风险与贡献.parquet")
    assert len(result) == 222 and len(origins) == 111
    assert result.groupby("origin_id").size().eq(2).all()
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    lpr = pd.read_csv(OUT / "inputs/lpr_cases.csv").set_index("month")
    valid = result[result.status == "COMPLETE_DIAGNOSTIC_ONLY"]
    errors, gaps, statuses = [], [], []
    for row in result.to_dict("records"):
        origin = origins.loc[row["origin_id"]]
        if row["origin_kind"] == "MONTHLY":
            parent = monthly.loc[row["origin_month"]]
            close(row["parent_E0_20_return"], parent.E0_20_return)
            close(row["parent_E1_20_return"], parent.E1_20_return)
        else:
            close(row["parent_LPR_20_return"], lpr.loc[row["origin_month"]].target20)
        if row["status"] != "COMPLETE_DIAGNOSTIC_ONLY":
            assert row["origin_id"] not in set(stock[stock.basket == row["basket"]].origin_id)
            continue
        stocks = stock[(stock.origin_id == row["origin_id"]) & (stock.basket == row["basket"])].sort_values("symbol")
        days = daily[(daily.origin_id == row["origin_id"]) & (daily.basket == row["basket"])].sort_values("relative_session")
        assert len(stocks) == 300 and len(days) == 25
        point = pd.Timestamp(row["observation_date"])
        expected = sorted(members[members.membership_date == point].symbol)
        assert stocks.symbol.tolist() == expected
        close(stocks.weight.sum(), 1)
        if row["basket"] == "EQUAL":
            close(stocks.weight, np.full(300, 1 / 300))
        else:
            date = pd.Timestamp(row["weight_date"])
            assert date < point and (point - date).days <= 62
            w = weight_source[weight_source.trade_date == date].set_index("con_code").weight.reindex(expected)
            assert w.notna().all()
            close(stocks.weight.to_numpy(), w.to_numpy() / w.sum())
        indices = pd.MultiIndex.from_product([pd.to_datetime(days.date), expected], names=["date", "symbol"])
        selected = source.reindex(indices)
        assert selected.return_is_usable.eq(True).all() and selected.daily_total_shareholder_return.notna().all()
        matrix = selected.daily_total_shareholder_return.to_numpy().reshape(25, 300)
        w = stocks.weight.to_numpy()
        basket = matrix @ w
        close(basket, days.net_return)
        close(np.maximum(matrix * w, 0).sum(axis=1), days.positive_contribution)
        close(-np.minimum(matrix * w, 0).sum(axis=1), days.negative_magnitude)
        close(days.positive_contribution - days.negative_magnitude, days.net_return)
        cov_current = np.cov(matrix[-20:], rowvar=False, ddof=1) * 252
        var_current = float(w @ cov_current @ w)
        close(var_current, row["current_variance"])
        close(w * (cov_current @ w), stocks.variance_contribution)
        close(stocks.variance_contribution.sum(), row["current_variance"])
        close(stocks.downside_contribution.sum(), row["current_net_down2"])
        close((np.minimum(matrix[-20:], 0) ** 2).mean(axis=0) * 252, stocks.current_down2)
        close((np.minimum(matrix[:20], 0) ** 2).mean(axis=0) * 252, stocks.previous_down2)
        close(float((stocks.current_down2 < stocks.previous_down2 - 1e-12).mean()), row["stock_fraction_downside_improved"])
        for prefix, window in [("current", slice(-20, None)), ("previous", slice(0, 20))]:
            net = days.net_return.to_numpy()[window]
            gross = days.negative_magnitude.to_numpy()[window]
            net_energy = 252 * np.mean(np.minimum(net, 0) ** 2)
            gross_energy = 252 * np.mean(gross ** 2)
            close(net_energy, row[prefix + "_net_down2"])
            close(gross_energy, row[prefix + "_gross_down2"])
            close(gross_energy - net_energy, row[prefix + "_offset_energy"])
            assert gross_energy + 1e-12 >= net_energy
        residual = row["current_variance"] - row["previous_variance"] - row["variance_change_stock_vol_component"] - row["variance_change_correlation_component"]
        assert abs(residual) < 1e-10
        errors.append(abs(residual))
    old = json.loads((OUT / "inputs/old_dispersion_result.json").read_text(encoding="utf-8"))
    assert old["status"] == "REJECTED_FROZEN_CSI300_RETURN_DISPERSION_VOLATILITY_GATE_FAILED_NO_RESCUE"
    assert old["mechanism_gate_passed"] is False
    record = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_CONSTITUENT_RISK_IDENTITIES",
        "origins_preserved": len(origins), "complete_baskets_recomputed": len(valid), "daily_rows_recomputed": len(daily),
        "stock_rows_checked": len(stock), "max_variance_change_identity_error": max(errors),
        "missing_statuses_preserved": len(result) - len(valid), "old_dispersion_rejection_preserved": True,
        "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False}
    (OUT / "verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
