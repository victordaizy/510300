"""标准库离线复算窗口敏感性包内的因子、完整主账户与绩效。"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def verify(base):
    expected = records(base / "portfolio_metrics.csv")
    account_rows = 0
    metric_checks = []
    for item in expected:
        label = f'{item["period"]}_N{item["window"]}_{item["cost"]}'
        ledger = records(base / "master_accounts" / label / "ledger.csv")
        previous, peak, drawdown = 200000.0, 200000.0, 0.0
        returns = []
        fills = 0
        for row in ledger:
            equity = float(row["equity"])
            value = float(row["cash"]) + float(row["shares"]) * float(row["mark"]) + float(row["dividend_receivable"])
            pnl = float(row["price_pnl"]) + float(row["dividend_recognized"]) - float(row["commission"]) - float(row["slippage_cost"])
            assert abs(equity-value) < 1e-7, (label, row["date"], "财富恒等式")
            assert abs(equity-previous-pnl) < 1e-7, (label, row["date"], "盈亏分解")
            daily_return = equity/previous-1
            assert abs(daily_return-float(row["net_return"])) < 1e-12, (label, row["date"], "日收益")
            returns.append(daily_return)
            fills += float(row["filled_quantity"]) != 0
            peak = max(peak, equity)
            drawdown = min(drawdown, equity/peak-1)
            previous = equity
        vol = statistics.stdev(returns)*math.sqrt(242)
        actual = {"annual_return":(previous/200000)**(242/len(returns))-1,
                  "sharpe":statistics.mean(returns)*242/vol,"max_drawdown":drawdown,
                  "end_equity":previous,"fills":fills,"days":len(returns)}
        for key, value in actual.items():
            assert abs(value-float(item[key])) < 1e-8, (label,key,value,item[key])
        account_rows += len(ledger)
        metric_checks.append({"account":label, **actual})
    inputs = records(base / "signal_inputs.csv")
    saved = records(base / "all_factor_values.csv")
    differences = []
    factor_cells = 0
    max_error = 0.0
    for row, cached in zip(inputs,saved,strict=True):
        assert row["date"] == cached["date"]
        if row["previous_close"]:
            opening, close, previous, cash = (float(row[k]) for k in ["open","close","previous_close","dividend"])
            differences.append(math.log((close+cash)/(opening+cash))-math.log((opening+cash)/previous))
        else:
            differences.append(None)
        for n in [50,60,70]:
            for kind, width in [("MATCHED_N",n),("STD_FIXED_60",60)]:
                key=f"{kind}_{n}"
                signal_window, vol_window = differences[-n:], differences[-width:]
                complete = len(signal_window)==n and len(vol_window)==width and None not in signal_window and None not in vol_window
                if not complete:
                    assert cached[key] == ""
                    continue
                std=statistics.stdev(vol_window)
                if std == 0:
                    assert cached[key] == ""
                    continue
                value=math.fsum(signal_window)/(std*math.sqrt(n))
                error=abs(value-float(cached[key]))
                assert error < 1e-10,(row["date"],key,error)
                max_error=max(max_error,error)
                factor_cells+=1
    return {"status":"PASS_OFFLINE_SAVED_FACTOR_AND_MASTER_ACCOUNT_RECOMPUTATION","master_accounts":len(metric_checks),
            "master_ledger_rows":account_rows,"factor_cells":factor_cells,"factor_max_error":max_error,
            "network_requests":0,"new_accounts":0,"new_model_fits":0,"metrics":metric_checks}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="离线核对50／60／70日保存证据")
    parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parent / "数据")
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    result=verify(args.data)
    if args.receipt:
        args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="metrics"},ensure_ascii=False))
