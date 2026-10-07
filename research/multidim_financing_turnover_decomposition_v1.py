"""把历史融资金额变化拆成成交规模和相对强度两项，不拟合交易策略。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_multidim_financing_turnover_decomposition_v1"
OVERRIDE = ROOT / "config/510300_financing_source_correction_20240808_v1.json"
COMPONENTS = ROOT / "data/raw/market/official_exchange_aggregate_turnover_v1_components.parquet"
CALENDAR = ROOT / "data/raw/market/510300_daily_downside_risk_v1.parquet"
OLD_RESULT = ROOT / "reports/discovery/510300_market_leverage_cascade_5d_discovery_v0.json"
LABELS = {"buy": "融资买入", "repay": "隐含偿还", "net": "融资净变化"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def save(name: str, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本次分解已经固定，禁止覆盖。")
    source = read(OVERRIDE)
    margin = ROOT / source["corrected_margin_path"]
    if sha(margin) != source["corrected_margin_sha256"]:
        raise ValueError("纠正后的融资输入与已确认副本不一致。")
    save("protocol.json", {
        "study_id": "510300_MULTIDIM_FINANCING_TURNOVER_DECOMPOSITION_V1",
        "frozen_at": now(),
        "question": "2024至2025年融资买入、隐含偿还及净融资的月度日均变化，有多少对应股票成交规模变化，有多少对应相对强度变化？",
        "primary_period": ["2024-01", "2025-12"],
        "baseline_month": "2023-12",
        "period_choice": "沿用当前主要历史期，逐月全部保留，不根据随后指数涨跌选月份。",
        "timing": "当月日均金额的历史会计分解。官方年鉴和月报为事后回取，不能称为当时已经收到的交易输入，也不回填到月内五日信号。",
        "scope": "融资汇总含合资格证券；成交分母为上交所A股与深交所股票总计，后者含B股和存托凭证。比值仅称相对强度代理，不能称为同口径融资成交占比，也不是沪深300专属资金。",
        "definitions": {
            "daily_average": "每月金额合计除以该月实际股票交易日数，统一单位亿元每日；2023年12月仅用于计算2024年1月变化。",
            "repay": "日隐含偿还=日融资买入-(日融资余额-前一交易日余额)，只作余额恒等式推算，不识别现金还款、卖券还款或强平。",
            "intensity": "qB=融资买入月合计/股票成交月合计；qR=隐含偿还月合计/股票成交月合计；qN=qB-qR。",
            "identity": "令A为日均股票成交，X为日均融资金额，q=X/A。两个相邻月份之间，ΔX=[(q今+q前)/2]×ΔA+[(A今+A前)/2]×Δq。",
            "components": "前项为成交规模分量，后项为相对强度分量，采用对称分解，无先后顺序参数；两项允许抵消。",
            "dominance": "仅当成交规模分量的绝对值大于相对强度分量的绝对值，计为规模分量较大；相等单列。",
            "absolute_component_share": "sum(abs(规模分量))/(sum(abs(规模分量))+sum(abs(强度分量)))，仅为算术分量大小比较，不是因果解释比例或预测解释率。"
        },
        "outputs": "全部24个月的输入、三个金额的两项分解、两年分别汇总、固定五个旧事件所属月的背景说明和图。",
        "fixed_context_months": ["2024-07", "2024-08", "2024-09", "2025-01", "2025-04"],
        "old_ratio_study": str(OLD_RESULT.relative_to(ROOT)).replace("\\", "/"),
        "old_study_disposition": "已有融资相对成交、偿还相对成交及市场广度交互的旧研究未过其原门槛。只去重，不恢复其规则，不把旧门槛等同于现目标。",
        "new_model_fits": 0,
        "new_strategy_tests": 0,
        "new_accounts": 0,
        "returns_loaded": False,
        "hypothesis_limit": "本次分解只能回答金额构成；相对强度是否预测后续收益仍未由本次建立，不能据此新加交易过滤器。",
        "inputs": {
            "margin": {"path": source["corrected_margin_path"], "sha256": sha(margin)},
            "components": {"path": str(COMPONENTS.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(COMPONENTS)},
            "calendar": {"path": str(CALENDAR.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(CALENDAR), "columns_loaded": ["date"]}
        },
        "goal_achieved": False
    })
    print("已固定24个月的金额分解；不读取收益、不新增模型或账户。")


def summarise(frame: pd.DataFrame, period: str):
    answer = []
    for name, label in LABELS.items():
        a, b = frame[name + "_scale_component"], frame[name + "_intensity_component"]
        actual = frame[name + "_change"]
        answer.append({
            "period": period, "amount": label, "months": len(frame),
            "scale_larger_months": int(a.abs().gt(b.abs()).sum()),
            "intensity_larger_months": int(b.abs().gt(a.abs()).sum()),
            "equal_magnitude_months": int(a.abs().eq(b.abs()).sum()),
            "components_opposite_months": int((a * b < 0).sum()),
            "raw_and_intensity_opposite_months": int((actual * b < 0).sum()),
            "scale_abs_share": a.abs().sum() / (a.abs().sum() + b.abs().sum()),
            "scale_abs_sum": a.abs().sum(), "intensity_abs_sum": b.abs().sum(),
            "maximum_identity_error_cny100m_per_day": (actual - a - b).abs().max()
        })
    return answer


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("已存在完成结果，禁止重复搜索或覆盖。")
    p = read(OUT / "protocol.json")
    for info in p["inputs"].values():
        if sha(ROOT / info["path"]) != info["sha256"]:
            raise ValueError("输入已改变，不能继续固定计算。")
    d = pd.read_parquet(ROOT / p["inputs"]["margin"]["path"], columns=["date", "market_rzye", "market_rzmre"])
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    d = d.sort_values("date").reset_index(drop=True)
    if d.date.duplicated().any():
        raise ValueError("融资日期重复。")
    d["net_total"] = d.market_rzye.diff() / 1e8
    d["buy_total"] = d.market_rzmre / 1e8
    d["repay_total"] = d.buy_total - d.net_total
    d["factor_month"] = d.date.dt.strftime("%Y-%m")
    d = d[d.factor_month.between("2023-12", "2025-12")].copy()
    calendar = pd.read_parquet(CALENDAR, columns=["date"])
    calendar["date"] = pd.to_datetime(calendar.date).dt.normalize()
    calendar["factor_month"] = calendar.date.dt.strftime("%Y-%m")
    calendar = calendar[calendar.factor_month.between("2023-12", "2025-12")]
    if calendar.date.duplicated().any() or list(d.date) != sorted(calendar.date.to_list()):
        raise ValueError("融资日期与股票交易日历不完整匹配。")
    if not np.isfinite(d[["buy_total", "repay_total", "net_total"]].to_numpy()).all() or d.repay_total.lt(0).any():
        raise ValueError("融资金额缺失或隐含偿还为负。")
    monthly = d.groupby("factor_month", sort=True).agg(
        trading_days=("date", "size"), first_date=("date", "min"), last_date=("date", "max"),
        buy_total=("buy_total", "sum"), repay_total=("repay_total", "sum"), net_total=("net_total", "sum")
    ).reset_index()
    c = pd.read_parquet(COMPONENTS)
    monthly = monthly.merge(c, on="factor_month", validate="one_to_one", how="left")
    required = ["sse_a_monthly_trade_amount_cny_100m", "szse_stock_monthly_trade_amount_cny_100m", "aggregate_monthly_trade_amount_cny_100m"]
    if monthly[required].isna().any().any() or monthly[required].le(0).any().any():
        raise ValueError("成交分母缺失或非正。")
    component_error = (monthly[required[0]] + monthly[required[1]] - monthly[required[2]]).abs().max()
    if component_error > 1e-6:
        raise ValueError("两市成交合计不一致。")
    monthly["turnover_daily"] = monthly.aggregate_monthly_trade_amount_cny_100m / monthly.trading_days
    monthly["previous_turnover_daily"] = monthly.turnover_daily.shift(1)
    monthly["turnover_change"] = monthly.turnover_daily.diff()
    for name in LABELS:
        monthly[name + "_daily"] = monthly[name + "_total"] / monthly.trading_days
        monthly[name + "_intensity"] = monthly[name + "_total"] / monthly.aggregate_monthly_trade_amount_cny_100m
        monthly["previous_" + name + "_daily"] = monthly[name + "_daily"].shift(1)
        monthly["previous_" + name + "_intensity"] = monthly[name + "_intensity"].shift(1)
        monthly[name + "_change"] = monthly[name + "_daily"].diff()
        monthly[name + "_intensity_change"] = monthly[name + "_intensity"].diff()
        mean_q = (monthly[name + "_intensity"] + monthly["previous_" + name + "_intensity"]) / 2
        mean_a = (monthly.turnover_daily + monthly.previous_turnover_daily) / 2
        monthly[name + "_scale_component"] = mean_q * monthly.turnover_change
        monthly[name + "_intensity_component"] = mean_a * monthly[name + "_intensity_change"]
        monthly[name + "_identity_error"] = monthly[name + "_change"] - monthly[name + "_scale_component"] - monthly[name + "_intensity_component"]
    kept = monthly[monthly.factor_month.ge("2024-01")].copy().reset_index(drop=True)
    expected = pd.period_range("2024-01", "2025-12", freq="M").astype(str).to_list()
    if kept.factor_month.to_list() != expected:
        raise ValueError("没有保留完整的24个月。")
    err = kept[[n + "_identity_error" for n in LABELS]].abs().to_numpy().max()
    net_err = max((kept.net_scale_component - kept.buy_scale_component + kept.repay_scale_component).abs().max(),
                  (kept.net_intensity_component - kept.buy_intensity_component + kept.repay_intensity_component).abs().max())
    if max(err, net_err) > 1e-7:
        raise ValueError("金额变化分解不满足恒等式。")
    summary = summarise(kept, "2024—2025")
    summary += summarise(kept[kept.factor_month.str.startswith("2024")], "2024")
    summary += summarise(kept[kept.factor_month.str.startswith("2025")], "2025")
    old = read(OLD_RESULT)
    monthly.to_parquet(OUT / "全部月度输入与分解.parquet", index=False)
    kept.to_csv(OUT / "全部24个月分解.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(OUT / "分量比较汇总.csv", index=False, encoding="utf-8-sig")
    context = kept[kept.factor_month.isin(p["fixed_context_months"])]
    context.to_csv(OUT / "原五个事件所属月背景.csv", index=False, encoding="utf-8-sig")
    result = {
        "study_id": p["study_id"], "status": "COMPLETED_MONTHLY_FINANCING_SCALE_INTENSITY_DECOMPOSITION",
        "completed_at": now(), "primary_months": len(kept), "trading_days": int(kept.trading_days.sum()),
        "buy_repay_same_direction_months": int((kept.buy_change * kept.repay_change > 0).sum()),
        "summary": summary,
        "fixed_context_months": context[["factor_month", "trading_days", "turnover_daily", "turnover_change"] + [n + suffix for n in LABELS for suffix in ["_daily", "_change", "_intensity", "_intensity_change", "_scale_component", "_intensity_component"]]].to_dict("records"),
        "checks": {"calendar_matches": True, "maximum_identity_error_cny100m_per_day": err,
                   "maximum_net_component_error_cny100m_per_day": net_err, "exchange_component_error_cny100m": component_error},
        "prior_ratio_study": {"path": p["old_ratio_study"], "status": old["status"], "development_contract": old["development_contract"],
                              "already_used": [s for s in old["feature_columns"] if "market_amount" in s or "interaction" in s]},
        "evidence_boundary": p["hypothesis_limit"], "scope": p["scope"], "timing": p["timing"],
        "new_model_fits": 0, "new_accounts": 0, "returns_loaded": False, "new_admitted_strategies": 0,
        "goal_achieved": False
    }
    save("result.json", result)
    print(json.dumps(clean({"status": result["status"], "primary_months": len(kept), "buy_repay_same_direction_months": result["buy_repay_same_direction_months"], "summary": summary[:3], "checks": result["checks"], "fixed_context_months": result["fixed_context_months"]}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定历史融资金额分解")
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.stage]()
