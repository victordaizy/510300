"""复算V3补完的保存预测与账户，并写入纯文字研究结论。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_pattern_daily_state_learning_v3_saved_completion"
ORIGINAL = ROOT / "reports/research/510300_pattern_daily_state_learning_v3"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    if (OUT / "saved_verification.json").exists():
        raise RuntimeError("本轮保存结果已复算；如有实质问题，应单独登记修正。")
    frozen = load(OUT / "freeze.json")
    for relative, expected in frozen["hashes"].items():
        for base in (ORIGINAL, OUT):
            assert hashlib.sha256((base / relative).read_bytes()).hexdigest() == expected
    d = pd.read_parquet(OUT / "inputs/features.parquet")
    x = pd.read_parquet(OUT / "results/model_features.parquet")
    controls = pd.read_parquet(OUT / "inputs/controls.parquet").set_index("signal_idx", drop=False)
    forecasts = pd.read_parquet(OUT / "results/daily_forecasts.parquet")
    schedule = pd.read_parquet(OUT / "results/fit_schedule.parquet").set_index("fit_idx")
    models = {}
    training_records = 0
    for file in sorted((OUT / "models").glob("*.json")):
        model = load(file)
        indices = np.asarray(model["train_signal_indices"], dtype=int)
        labels = controls.loc[indices]
        assert len(indices) >= 252 and len(np.unique(indices)) == len(indices)
        assert int(labels.exit_idx.max()) <= model["fit_idx"]
        assert int(labels.signal_idx.min()) >= model["fit_idx"] - 504
        assert model["as_of_date"] == d.date.iloc[model["fit_idx"]]
        assert model["latest_mature_exit_idx"] == int(labels.exit_idx.max())
        training_records += len(indices)
        models[(model["model"], model["fit_idx"])] = model
    predicted_values = 0
    for row in forecasts.itertuples():
        assert row.fit_idx <= row.idx and row.date == d.date.iloc[row.idx]
        assert abs(row.MATURE_MEAN - schedule.loc[row.fit_idx, "mature_mean"]) < 1e-12
        for name in ("BACKGROUND", "PHASE_VOLUME"):
            model = models[(name, row.fit_idx)]
            z = np.clip((x.loc[row.idx, model["columns"]].to_numpy(float) - model["x_mean"]) / model["x_scale"], -5, 5)
            value = model["intercept"] + z @ model["coefficients"]
            assert np.isfinite(value) and abs(value - getattr(row, name)) < 1e-12
            predicted_values += 1
    account_rows = 0
    concentration = {}
    for folder in sorted((OUT / "results/accounts").glob("*/*")):
        ledger = pd.read_parquet(folder / "daily.parquet")
        trades = pd.read_parquet(folder / "trades.parquet")
        metrics = load(folder / "metrics.json")
        np.testing.assert_allclose(ledger.cash_cny + ledger.shares * ledger.close + ledger.receivable_cny, ledger.equity_cny, atol=1e-7, rtol=0)
        r = ledger.daily_return.to_numpy(float).copy()
        ending = ledger.equity_cny.iloc[-1] - metrics["terminal_haircut_cny"]
        r[-1] = ending / ledger.equity_cny.iloc[-2] - 1
        nav = np.r_[200000.0, 200000.0 * np.cumprod(1 + r)]
        expected = {
            "net_cagr": (ending / 200000.0) ** (252 / len(r)) - 1,
            "net_sharpe": r.mean() / r.std(ddof=1) * np.sqrt(252),
            "max_drawdown": -(nav / np.maximum.accumulate(nav) - 1).min(),
            "net_profit_cny": ending - 200000.0,
        }
        for key, value in expected.items():
            assert abs(metrics[key] - value) < (1e-6 if key == "net_profit_cny" else 1e-10)
        if len(trades):
            assert (trades.exit_idx > trades.entry_idx).all()
            assert ((trades.quantity % 100) == 0).all()
            assert trades.entry_date.ge("2021-01-04").all()
        if not metrics["open_position"]:
            assert abs(trades.net_pnl.sum() - metrics["net_profit_cny"]) < 1e-6
        if folder.name == "STRESS":
            maximum = trades.loc[trades.net_pnl.idxmax()]
            concentration[folder.parent.name] = {
                "largest_cycle": maximum.signal_id,
                "largest_cycle_net_pnl_cny": float(maximum.net_pnl),
                "largest_cycle_share_of_net_profit": float(maximum.net_pnl / metrics["net_profit_cny"]),
                "other_saved_cycle_profit_cny": float(trades.net_pnl.sum() - maximum.net_pnl),
                "diagnostic_only_not_rerun_without_cycle": True,
            }
        account_rows += len(ledger)
    summary = load(OUT / "summary.json")
    prediction = {item["model"]: item for item in summary["prediction_performance"]}
    deterioration = prediction["PHASE_VOLUME"]["mse"] / prediction["BACKGROUND"]["mse"] - 1
    save(OUT / "saved_verification.json", {
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "PASS_SAVED_PREDICTIONS_CLOCK_AND_ACCOUNT_IDENTITIES",
        "original_and_copied_freezes_unchanged": True,
        "fitted_models": len(models),
        "saved_prediction_values_recomputed": predicted_values,
        "training_records_including_reuse": training_records,
        "training_records_are_independent_events": False,
        "maturity_rule": "收盘判断可使用当日上午已完成退出的标签，故exit_idx <= fit_idx；不是09:00提前判断。",
        "accounts": 6,
        "account_rows": account_rows,
        "T_plus_1_and_integer_lots_pass": True,
        "new_model_fits_in_verification": 0,
        "new_accounts_in_verification": 0,
        "independent_forward_observations": 0,
    })
    save(OUT / "saved_concentration.json", concentration)
    phase = next(item for item in summary["phase_accounts"] if item["cost"] == "STRESS")
    c = concentration["PHASE_VOLUME"]
    report = f"""按原冻结协议补完普通日状态学习V3后，仍未达到高夏普目标。主样本2021-01-04至2026-09-16，20万元完整账户、252日年化，保留空仓日、原费用、滑点、T+1、整手和分红。压力成本下，成熟均值模型年化0.59%、夏普0.204；价格背景模型年化3.77%、夏普0.586；加入趋势变化与成交量后年化{phase['net_cagr']:.2%}、夏普{phase['net_sharpe']:.3f}、最大回撤{phase['max_drawdown']:.2%}。三者均未满足净年化10%、净夏普1.2、最大回撤10%的联合要求。

这是补完已登记实验。原V3在2026-09-24 12:39冻结，已有协议、代码与全部输入，此前没有结果。本次在独立目录运行，原文件字节不变；模型与账户代码不变，只把输出隔离并将内部数据存为Parquet。新增132次回归，对应66个更新时间、两个模型；成熟均值只更新均值。运行6条账户，包含三种表示与两档成本，不是六个独立策略。

三种表示使用完全相同的成熟普通日训练池；过去504日、至少252条、每21个交易日更新、岭惩罚10全部不变。量价扩展只增加三日趋势变化及成交量比率。每个确认日收盘判断，下一可执行开盘入场；持仓后预测转为非正时下一可卖开盘退出，原形态失效与最多五日仍执行。当天上午已经完成退出的标签可以进入当天收盘训练，不被挪到上午09:00使用。

共有1384个评价日、1377个已成熟预测目标；标签相互重叠，不能计作1377个独立机会。量价扩展预测均方误差比价格背景增加{deterioration:.2%}；其相对背景的误差差值95%区间为负。背景和量价两者的误差也均高于成熟训练均值。压力账户点值虽有改善，但量价相对背景的年化增量区间为-0.76至1.09个百分点，夏普增量区间为-0.149至0.238，均跨零。区间沿用原20日区块、2000次共同抽样，未改抽样方法或另选种子。

量价压力账户11个完整周期，净利润{phase['net_profit_cny']:.2f}元；最大一笔为{c['largest_cycle']}，盈利{c['largest_cycle_net_pnl_cny']:.2f}元，占净利润{c['largest_cycle_share_of_net_profit']:.2%}。其余保存周期净利润合计仅{c['other_saved_cycle_profit_cny']:.2f}元。这里是已保存周期的加总诊断，没有删除该周期重新复利回测。

同期间原形态保存对照为年化3.66%、夏普0.491、最大回撤11.19%。V3的回撤及夏普点值有所改善，但入场变少、优势仍依赖同一次2024年行情，不能据此认定可重复的信息优势。这个2026-09-16终点与此前宏观资金共同实验的2026-08-14终点不同，未跨终点拼接成排行榜。

本轮已复算2768个保存回归预测，核对6条账户共8304行及训练标签的成熟时点；复算不重新拟合或重跑账户。结果状态为FROZEN_V3_NUMERICAL_FAILURE。原冻结设置保持，不调整阈值、窗口、方向或表示救回。没有审核包、当前买卖判断、新资料采集或独立前向样本；整个高夏普目标仍未完成。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    print("V3保存预测、时钟及完整账户复算通过，纯文字结论已保存。")


if __name__ == "__main__":
    main()
