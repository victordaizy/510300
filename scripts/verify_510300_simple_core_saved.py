"""仅用Python标准库复算已保存的简单核心信号、状态、仓位映射和账户指标。"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def close(a, b, label, tolerance=1e-7):
    assert abs(float(a)-float(b)) <= tolerance, (label, a, b)


def truth(value):
    return str(value).lower() == "true"


def verify(base):
    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    market = records(base / "signal_inputs.csv")
    deltas, returns, scores, vol_multiplier = [], [], {50: [], 60: [], 70: []}, []
    factor_checks = 0
    for i, row in enumerate(market):
        valid = row["previous_close"] != ""
        op, cl, div = (float(row[k]) for k in ["open", "close", "dividend"])
        overnight = math.log((op+div)/float(row["previous_close"])) if valid else None
        intraday = math.log((cl+div)/(op+div))
        if valid:
            close(overnight, row["overnight_log"], (row["date"], "隔夜"), 1e-12)
        close(intraday, row["intraday_log"], (row["date"], "日内"), 1e-12)
        deltas.append(intraday-overnight if valid else None)
        returns.append((cl+div)/float(row["previous_close"])-1 if valid else None)
        sample = returns[-20:]
        sigma = statistics.stdev(sample)*math.sqrt(242) if len(sample) == 20 and None not in sample else None
        if sigma is not None:
            close(sigma, row["vol20"], (row["date"], "ETF20日波动"), 1e-10)
        vol_multiplier.append(min(1, .1/sigma) if sigma and sigma > 0 else vol_multiplier[-1] if vol_multiplier else 1.0)
        for n in scores:
            sample = deltas[-n:]
            complete = len(sample) == n and None not in sample
            std = statistics.stdev(sample) if complete else 0
            scores[n].append(math.fsum(sample)/(std*math.sqrt(n)) if std else None)

    metrics = {row["account"]: row for row in records(base / "account_metrics.csv")}
    ledger_count, row_count, decision_count = 0, 0, 0
    account_metrics, reference_checks = {}, []
    for category in ["primary_accounts", "price_references", "comparators"]:
        for folder in sorted((base / category).iterdir()):
            ledger, decisions = records(folder / "ledger.csv"), records(folder / "decisions.csv")
            assert len(decisions) == len(ledger)+1
            previous, peak, drawdown, pnl_sum = 200000.0, 200000.0, 0.0, 0.0
            daily, exposures = [], []
            fills = 0
            for i, row in enumerate(ledger):
                equity, shares, mark = (float(row[k]) for k in ["equity", "shares", "mark"])
                label = (category, folder.name, row["date"])
                assert row["date"] == decisions[i]["execution_date"], label
                assert row["origin"] == decisions[i]["origin"], label
                assert row["date"] == decisions[i+1]["origin"], label
                assert row["mark_clock"] == "CLOSE", label
                close(row["requested_quantity"], decisions[i]["requested_quantity"], label)
                close(equity, float(row["cash"])+shares*mark+float(row["dividend_receivable"]), label)
                pnl = float(row["price_pnl"])+float(row["dividend_recognized"])-float(row["commission"])-float(row["slippage_cost"])
                close(equity-previous, pnl, label)
                close(row["pnl"], pnl, label)
                close(row["net_return"], equity/previous-1, label, 1e-12)
                close(row["exposure"], shares*mark/equity, label, 1e-12)
                assert shares >= 0 and shares % 100 == 0 and float(row["cash"]) >= -1e-7, label
                close(shares, float(row["shares_before"])+float(row["filled_quantity"]), label)
                daily.append(equity/previous-1)
                exposures.append(shares*mark/equity)
                fills += float(row["filled_quantity"]) != 0
                peak = max(peak, equity)
                drawdown = min(drawdown, equity/peak-1)
                previous = equity
                pnl_sum += pnl
            sigma = statistics.stdev(daily)*math.sqrt(242)
            actual = {"days": len(daily), "end_equity": previous, "annual_return": (previous/200000)**(242/len(daily))-1,
                      "sharpe": statistics.mean(daily)*242/sigma if sigma else None, "volatility": sigma,
                      "max_drawdown": drawdown, "mean_exposure": statistics.mean(exposures), "fills": fills}
            close(pnl_sum, previous-200000, folder.name)
            if category != "price_references":
                for k, value in actual.items():
                    close(value, metrics[folder.name][k], (folder.name, k))
                account_metrics[folder.name] = actual
            if category == "primary_accounts":
                price = records(base / "price_references" / folder.name / "decisions.csv")
                window = int(folder.name.split("_")[1][1:])
                for i, decision in enumerate(decisions):
                    t = int(decision["origin_index"])
                    assert decision["origin"] == market[t]["date"] == price[i]["origin"]
                    close(decision["score"], scores[window][t], (folder.name, "分数"), 1e-10)
                    close(decision["etf_volatility_multiplier"], vol_multiplier[t], (folder.name, "倍率"), 1e-10)
                    weight = float(price[i]["reference_weight"])
                    assert weight in [0, 1]
                    target = weight * vol_multiplier[t]
                    close(decision["reference_weight"], target, (folder.name, "单层目标"), 1e-10)
                    before = ledger[i-1] if i else None
                    held = int(before["shares"]) if before else 0
                    nav = float(before["equity"]) if before else 200000.0
                    mark = float(market[t]["close"])
                    desired = math.floor(target*nav/mark/100)*100
                    if target > 0 and held > 0 and abs(target-held*mark/nav) < 0.1:
                        desired = held
                    close(decision["requested_quantity"], desired-held, (folder.name, "带宽与次日数量"))
                    factor_checks += 1
            if category == "price_references":
                window = int(folder.name.split("_")[1][1:])
                armed, last_exit, active, accrued, high = True, -1000000, None, 0.0, None
                for i, decision in enumerate(decisions):
                    t = int(decision["origin_index"])
                    held = int(ledger[i-1]["shares"]) if i else 0
                    if i:
                        row = ledger[i-1]
                        old = int(row["shares_before"])
                        if old == 0 and held > 0:
                            active = {"entry_index": t, "quantity": held, "cost": float(row["notional"])+float(row["commission"])}
                            armed, accrued, high = False, 0.0, active["cost"]
                        elif old > 0 and held == 0:
                            last_exit, active, accrued, high = t, None, 0.0, None
                        elif held > 0:
                            assert float(row["filled_quantity"]) == 0, (folder.name, "价格参考周期中发生调仓")
                            accrued += float(row["dividend_recognized"])
                        if held:
                            high = max(high, held*float(market[t]["close"])+accrued)
                    current, prior = scores[window][t], scores[window][t-1]
                    entry = current is not None and prior is not None and current > 1 and prior > 1 and truth(market[t]["feature_valid"])
                    exit_score = current is not None and prior is not None and current < 0 and prior < 0
                    if not held and not entry:
                        armed = True
                    assert truth(decision["entry_rearmed"]) == armed, (folder.name, t, "再入场许可")
                    if held:
                        value = held*float(market[t]["close"])+accrued
                        exit_price = exit_score or value/active["cost"]-1 <= -.06 or value/high-1 <= -.08 or t-active["entry_index"]+1 >= 60
                        latched = bool(decisions[i-1]["exit_reasons"]) if i else False
                        expected_weight = 0 if exit_price or latched else 1
                    else:
                        expected_weight = int(entry and armed and t-last_exit >= 2)
                    close(decision["reference_weight"], expected_weight, (folder.name, t, "原价格状态"))
                reference_checks.append({"account": folder.name, "price_decisions_verified": len(decisions)})
            ledger_count += 1
            row_count += len(ledger)
            decision_count += len(decisions)
    assert ledger_count == 32 and len(account_metrics) == 20 and len(reference_checks) == 12
    concentration = records(base / "cycle_concentration.csv")
    completed = records(base / "completed_cycles.csv")
    for row in concentration:
        winners = sorted([float(c["profit"]) for c in completed if c["account"] == row["account"] and float(c["profit"]) > 0], reverse=True)
        total = float(row["total_net_profit"])
        if total > 0 and winners:
            close(row["top1_share"], winners[0]/total, (row["account"], "最大周期集中度"))
            close(row["top5_share"], sum(winners[:5])/total, (row["account"], "前五周期集中度"))
    return {"status": "PASS_OFFLINE_SAVED_ECONOMICS_SIGNALS_STATES_TARGETS", "ledger_accounts": ledger_count,
            "ledger_rows": row_count, "decision_rows": decision_count, "primary_score_and_target_checks": factor_checks,
            "primary_accounts": protocol["planned_primary_accounts"], "price_reference_checks": reference_checks,
            "network_requests": 0, "new_accounts": 0, "new_model_fits": 0, "random_samples": 0, "metrics": account_metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线复算已保存证据，不生成回测账户")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "reports/research/510300_simple_core_window_diagnostic_v1")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = verify(args.data)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["metrics", "price_reference_checks"]}, ensure_ascii=False))
