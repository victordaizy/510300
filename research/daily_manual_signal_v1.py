"""导出固定 D60 因子的每日人工核对表，并进行有界的公式对照。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.simple_session_divergence_v1 import make_rule

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_daily_manual_signal_v1"
SETTINGS = ROOT / "config/510300_new_daily_input_adapter_runtime_v1.json"
SPEC = {
    "study_id": "510300_D60_MANUAL_FORMULA_DIAGNOSTIC_V1",
    "scope": "仅比较因子和连续两日门槛，不回测账户、不选择新参数",
    "variants": ["CURRENT", "EQUIVALENT_MEAN", "REMOVE_SQRT", "DIVIDE_60", "WINDOW_20", "WINDOW_120", "PLUS_60", "POPULATION_STD"],
    "periods": [["主历史", "2020-01-01", "2026-09-11"], ["较早历史", "2015-01-01", "2019-12-31"]],
    "dispersion_window": 60,
    "entry_threshold": 1,
    "exit_threshold": 0,
    "confirmation_days": 2,
    "portfolio_replay": False,
    "model_fits": 0,
    "new_strategy_selection": False,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def resolve_source():
    settings = read(SETTINGS)
    completed = ROOT / settings["output_root"] / "latest_completed.json"
    if completed.exists():
        receipt = read(completed)
        if receipt["status"] != "COMPLETED_NEW_DAILY_INPUTS_AND_FIXED_CONTINUATION":
            raise ValueError("正式续算回执未完成，不能使用候选输入冒充正式日数据")
        return receipt["next_source"], completed
    return settings["initial_source"], SETTINGS


def conditions(factor, threshold=1.0):
    prior = factor.shift(1)
    complete = factor.notna() & prior.notna()
    return ((factor > threshold) & (prior > threshold) & complete), ((factor < 0) & (prior < 0) & complete)


def calculate(frame):
    data = frame.copy()
    if data.date.duplicated().any() or not data.date.is_monotonic_increasing:
        raise ValueError("交易日期必须唯一且升序")
    night = np.log((data.open + data.dividend) / data.previous_close)
    intraday = np.log((data.close + data.dividend) / (data.open + data.dividend))
    np.testing.assert_allclose(night, data.overnight_log, rtol=0, atol=1e-13, equal_nan=True)
    np.testing.assert_allclose(intraday, data.intraday_log, rtol=0, atol=1e-13, equal_nan=True)
    difference = intraday - night
    std60 = difference.rolling(60).std(ddof=1)
    total60 = difference.rolling(60).sum()
    score = total60 / (std60.replace(0, np.nan) * np.sqrt(60))
    original_rule, original = make_rule(data, {"kind": "DIFFERENCE", "window": 60, "direction": 1})
    np.testing.assert_allclose(score, original, rtol=0, atol=1e-12, equal_nan=True)
    entry, exit_flag = conditions(score)
    np.testing.assert_array_equal(entry & data.feature_valid.fillna(False), original_rule["entry"].astype(bool))
    np.testing.assert_array_equal(exit_flag, original_rule["exit"][1])
    return pd.DataFrame({
        "date": data.date.dt.strftime("%Y-%m-%d"), "previous_close": data.previous_close,
        "open": data.open, "close": data.close, "cash_dividend_per_share": data.dividend,
        "overnight_log": night, "intraday_log": intraday, "difference": difference,
        "sum60": total60, "sample_std60": std60, "score60": score,
        "entry_condition_2days": entry.astype(int), "difference_exit_condition_2days": exit_flag.astype(int),
        "feature_valid": data.feature_valid.fillna(False).astype(bool),
        "source_entry_with_feature_valid": (entry & data.feature_valid.fillna(False)).astype(int),
    })


def diagnostic(data, daily, folder):
    difference = daily.difference
    std60 = daily.sample_std60.replace(0, np.nan)
    addition = data.intraday_log + data.overnight_log
    factors = {
        "CURRENT": daily.score60,
        "EQUIVALENT_MEAN": difference.rolling(60).mean() / std60 * np.sqrt(60),
        "REMOVE_SQRT": daily.sum60 / std60,
        "DIVIDE_60": daily.sum60 / (std60 * 60),
        "WINDOW_20": difference.rolling(20).sum() / (std60 * np.sqrt(20)),
        "WINDOW_120": difference.rolling(120).sum() / (std60 * np.sqrt(120)),
        "PLUS_60": addition.rolling(60).sum() / (addition.rolling(60).std(ddof=1).replace(0, np.nan) * np.sqrt(60)),
        "POPULATION_STD": daily.sum60 / (difference.rolling(60).std(ddof=0).replace(0, np.nan) * np.sqrt(60)),
    }
    labels = {"CURRENT": "现版：差值加总 / 样本标准差 / √60", "EQUIVALENT_MEAN": "等价：差值均值 / 样本标准差 × √60",
              "REMOVE_SQRT": "去掉√60，仍用原入场阈值1", "DIVIDE_60": "把分母√60换成60，阈值仍为1",
              "WINDOW_20": "累计20日，波动窗口仍为60日", "WINDOW_120": "累计120日，波动窗口仍为60日",
              "PLUS_60": "差值换成日内加隔夜，并用加总序列标准差",
              "POPULATION_STD": "仅把样本标准差换成总体标准差"}
    baseline_entry, baseline_exit = conditions(factors["CURRENT"])
    rows = []
    for period, begin, end in SPEC["periods"]:
        within = (data.date >= begin) & (data.date <= end) & data.feature_valid.fillna(False)
        for key, factor in factors.items():
            entry, exits = conditions(factor)
            mask = within & factor.notna() & factor.shift(1).notna() & factors["CURRENT"].notna() & factors["CURRENT"].shift(1).notna()
            count = int(mask.sum())
            changed = (entry != baseline_entry) & mask
            changed_exit = (exits != baseline_exit) & mask
            rows.append({"period": period, "variant": key, "description": labels[key], "days": count,
                "entry_condition_days": int((entry & mask).sum()), "entry_changed_days": int(changed.sum()),
                "entry_changed_fraction": float(changed.sum() / count),
                "exit_condition_days": int((exits & mask).sum()), "exit_changed_days": int(changed_exit.sum()),
                "either_changed_days": int(((changed | changed_exit) & mask).sum()),
                "last_score": float(factor[mask].iloc[-1])})
    equivalent_error = float(np.nanmax(np.abs(factors["EQUIVALENT_MEAN"] - factors["CURRENT"])))
    assert equivalent_error < 1e-12
    for factor, threshold in [(factors["REMOVE_SQRT"], np.sqrt(60)), (factors["DIVIDE_60"], 1 / np.sqrt(60))]:
        adjusted_entry, adjusted_exit = conditions(factor, threshold)
        np.testing.assert_array_equal(adjusted_entry, baseline_entry)
        np.testing.assert_array_equal(adjusted_exit, baseline_exit)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(folder / "公式变化_门槛日期对照.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"date": daily.date, **factors}).to_csv(folder / "全部对照因子.csv", index=False, encoding="utf-8-sig")
    write_json(folder / "诊断结果.json", {"status": "COMPLETED_FACTOR_CONDITION_DIAGNOSTIC_ONLY", "comparisons": rows,
        "equivalent_mean_max_error": equivalent_error, "adjusted_threshold_entry_and_exit_identical": True,
        "account_returns_computed": False, "new_model_fits": 0, "new_selected_strategy": False})
    print(comparison[comparison.period == "主历史"].to_string(index=False), flush=True)


def portfolio_snapshot(source, cutoff, folder):
    records = []
    for cost in ["BASE", "STRESS"]:
        account = ROOT / source["source_folder"] / "accounts" / cost / "SELECTED_MIX_BAND10_SIMPLE2"
        ledger = pd.read_parquet(account / "ledger.parquet").iloc[-1]
        decision = pd.read_parquet(account / "decisions.parquet").iloc[-1]
        if str(ledger.date.date()) != cutoff or str(decision.origin.date()) != cutoff:
            raise ValueError("完整组合与因子截止日不一致，不能发布混合日期快照")
        records.append({"费用情景": cost, "研究收盘日期": cutoff,
            "研究下一执行日期": str(decision.execution_date.date()),
            "研究收盘持有份额": int(ledger.shares), "研究收盘仓位": float(ledger.exposure),
            "参考目标权重": float(decision.reference_weight), "研究下一日净申请份额": int(decision.requested_quantity),
            "研究动作": str(decision.action), "研究收盘权益": float(ledger.equity),
            "仅研究模拟": bool(decision.simulation_only), "依据文件": relative(account / "decisions.parquet"),
            "决定文件SHA256": digest(account / "decisions.parquet")})
    pd.DataFrame(records).to_csv(folder / "完整策略当前研究决定.csv", index=False, encoding="utf-8-sig")
    return records


def run(with_diagnostics=False):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "公式对照预定方案.json"
    if with_diagnostics and not protocol.exists():
        write_json(protocol, {**SPEC, "registered_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}, exclusive=True)
    source, source_receipt = resolve_source()
    input_file = ROOT / source["features"]
    data = pd.read_parquet(input_file)
    daily = calculate(data)
    cutoff = str(daily.date.iloc[-1])
    folder = OUT / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / "D60每日数据.csv"
    csv_bytes = daily.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    if csv_path.exists() and csv_path.read_bytes() != csv_bytes:
        raise ValueError("同日期导出与原保存记录不一致，保留原记录并停止覆盖")
    if not csv_path.exists():
        csv_path.write_bytes(csv_bytes)
    tail = daily.tail(260).replace({np.nan: None})
    payload = {"cutoff": cutoff, "rows": tail.to_dict("records"), "all_history_rows": len(data),
               "feature_source": relative(input_file), "feature_sha256": digest(input_file),
               "csv_source": relative(csv_path), "source_provider": str(data.source.iloc[-1])}
    write_json(folder / "工作簿输入.json", payload)
    portfolio = portfolio_snapshot(source, cutoff, folder)
    last = daily.iloc[-1].replace({np.nan: None}).to_dict()
    receipt = {"status": "COMPLETED_FIXED_D60_MANUAL_EXPORT", "cutoff": cutoff, "latest": last,
               "input_path": relative(input_file), "input_sha256": digest(input_file),
               "formal_source_receipt": relative(source_receipt), "csv_path": relative(csv_path),
               "full_history_rows": len(data), "workbook_seed_rows": len(tail),
               "matches_current_make_rule": True, "portfolio_signal": "NOT_REPLACED_BY_D60_CONDITION",
               "portfolio_snapshot": portfolio,
               "network_requests": 0, "new_accounts": 0, "new_model_fits": 0}
    write_json(folder / "每日导出回执.json", receipt)
    write_json(OUT / "latest.json", receipt)
    if with_diagnostics:
        diagnostic(data, daily, folder)
    print(json.dumps({"截止日": cutoff, "D60": last["score60"], "连续两日入场条件": last["entry_condition_2days"],
                      "连续两日差值退出条件": last["difference_exit_condition_2days"], "保存目录": str(folder)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从已完成的正式研究输入导出每日 D60 人工核对数值")
    parser.add_argument("--diagnostics", action="store_true", help="执行本次预定的八种公式对照，不改变现版")
    args = parser.parse_args()
    run(args.diagnostics)
