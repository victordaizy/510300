"""104个月货币变化的余额与隐含基数分解；只复用冻结行情标签。"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "reports/research/510300_macro_volatility_mechanisms_v2_run2"
PARENT = ROOT / "reports/research/510300_macro_volatility_observation_v2_run1"
CONFIG = ROOT / "config/510300_macro_volatility_mechanisms_v2_run2.json"
EPS = 1e-10
GROUP_CURRENT = "改善_当期相对余额同向"
GROUP_BASE = "改善_当期相对余额未同向"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(STUDY / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def currency(text: str, metric: str) -> tuple[float, float, str]:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    found = re.search(r"\(" + metric + r"\)余额([\d.]+)(万亿元|亿元)[,，][^();。]{0,45}?同比(增长|下降)(-?[\d.]+)%", text)
    if not found:
        raise ValueError(f"未找到{metric}余额及同比")
    balance = float(found[1]) * (10000 if found[2] == "万亿元" else 1)
    growth = float(found[4]) * (1 if found[3] == "增长" else -1)
    return round(balance, 10), growth, found[0]


def extract_facts() -> pd.DataFrame:
    releases = pd.read_csv(PARENT / "inputs/official_releases.csv")
    causes = pd.read_csv(PARENT / "results/104个月_原因证据台账.csv").set_index("stat_month")
    facts = []
    for release in releases.to_dict("records"):
        month = release["stat_month"]
        path = PARENT / "sources/official_money" / f"{month}.html"
        assert digest(path) == release["source_sha256"], month
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        row = {key: release[key] for key in ["stat_month", "training_regime", "definition_version", "source_url", "source_sha256", "available_at_upper_bound", "available_before_market_cutoff"]}
        for metric in ["M0", "M1", "M2"]:
            try:
                balance, growth, evidence = currency(text, metric)
            except ValueError as exc:
                normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
                matches = re.findall(r".{0,12}" + metric + r".{0,100}", normalized)
                raise ValueError(f"{month} {metric}: {matches}") from exc
            row[f"{metric.lower()}_balance_100m"] = balance
            row[f"{metric.lower()}_yoy_pp"] = growth
            row[f"{metric.lower()}_evidence"] = evidence
            row[f"{metric.lower()}_implied_base_100m"] = balance / (1 + growth / 100)
            if metric in ["M1", "M2"]:
                assert abs(balance - causes.loc[month, f"{metric.lower()}_balance_100m"]) < 1e-6, (month, metric, balance, causes.loc[month, f"{metric.lower()}_balance_100m"])
                assert abs(growth - release[f"{metric.lower()}_yoy_pp"]) < EPS
        row["m1_ex_m0_100m"] = round(row["m1_balance_100m"] - row["m0_balance_100m"], 10)
        row["m1_ex_m0_label"] = "单位活期余额近似" if str(release["training_regime"]).startswith("M1_OLD") else "多类高流动性账户合计"
        row["source_vintage"] = "RECONSTRUCTED_OFFICIAL_NOT_IMMUTABLE"
        row["economic_cause"] = "未知/可混合"
        facts.append(row)
    frame = pd.DataFrame(facts).sort_values("stat_month").reset_index(drop=True)
    frame["spread_pp"] = (frame.m1_yoy_pp - frame.m2_yoy_pp).round(10)
    periods = pd.PeriodIndex(frame.stat_month, freq="M").asi8
    for lag in [1, 3]:
        prefix = f"d{lag}"
        valid = (frame.definition_version == frame.definition_version.shift(lag)) & (pd.Series(periods).diff(lag) == lag)
        frame[f"{prefix}_eligible"] = valid
        for metric in ["m1", "m2"]:
            balance = frame[f"{metric}_balance_100m"]
            base = frame[f"{metric}_implied_base_100m"]
            growth = frame[f"{metric}_yoy_pp"]
            frame[f"{prefix}_{metric}_current_log_pp"] = (100 * np.log(balance / balance.shift(lag))).where(valid)
            frame[f"{prefix}_{metric}_base_revision_log_pp"] = (-100 * np.log(base / base.shift(lag))).where(valid)
            frame[f"{prefix}_{metric}_growth_log_pp"] = (100 * (np.log1p(growth / 100) - np.log1p(growth.shift(lag) / 100))).where(valid)
            frame[f"{prefix}_{metric}_ordinary_yoy_pp"] = growth.diff(lag).round(10).where(valid)
        frame[f"{prefix}_spread_ordinary_pp"] = frame.spread_pp.diff(lag).round(10).where(valid)
        for name in ["current_log_pp", "base_revision_log_pp", "growth_log_pp"]:
            frame[f"{prefix}_relative_{name}"] = frame[f"{prefix}_m1_{name}"] - frame[f"{prefix}_m2_{name}"]
        frame[f"{prefix}_m0_balance_change_100m"] = frame.m0_balance_100m.diff(lag).round(10).where(valid)
        frame[f"{prefix}_m1_ex_m0_change_100m"] = frame.m1_ex_m0_100m.diff(lag).round(10).where(valid)
    def label(row: pd.Series) -> str:
        if not row.d3_eligible:
            return "不可计算_口径或历史不足"
        if row.d3_spread_ordinary_pp <= EPS:
            return "剪刀差未改善"
        if row.d3_relative_growth_log_pp <= EPS:
            return "普通与对数方向不一致"
        return GROUP_CURRENT if row.d3_relative_current_log_pp > EPS else GROUP_BASE
    frame["arithmetic_mechanism_group"] = frame.apply(label, axis=1)
    return frame


def summaries(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, group), part in frame.groupby(["training_regime", "arithmetic_mechanism_group"], sort=True):
        for entry in ["E0", "E1"]:
            for horizon in [5, 20, 60]:
                prefix = f"{entry}_{horizon}"
                data = part[part[f"{prefix}_return"].notna()]
                row = {"training_regime": regime, "group": group, "entry": entry, "horizon": horizon, "all_months": len(part), "mature_months": len(data), "pending_months": len(part) - len(data), "stat_months": ";".join(data.stat_month)}
                for metric in ["return", "worst_close", "worst_path", "drawdown"]:
                    values = data[f"{prefix}_{metric}"]
                    for stat in ["mean", "median", "min", "max"]:
                        row[f"{metric}_{stat}"] = float(getattr(values, stat)()) if len(values) else np.nan
                for col in ["pre_publication_return20", "pre_publication_return60", "d3_relative_current_log_pp", "d3_relative_base_revision_log_pp"]:
                    row[f"{col}_mean"] = data[col].mean()
                row["status"] = "DESCRIPTIVE_ONLY_SMALL_SAMPLE" if len(data) < 10 else "DESCRIPTIVE_ONLY_DEPENDENT_MONTHS"
                rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    for folder in ["results", "inputs", "evidence", "sources", "code"]:
        (STUDY / folder).mkdir(parents=True, exist_ok=True)
    frozen = json.loads((STUDY / "freeze_receipt.json").read_text(encoding="utf-8-sig"))
    assert digest(CONFIG) == frozen["protocol_sha256"], "冻结协议不一致"
    sources = [PARENT / "inputs/official_releases.csv", PARENT / "results/104个月_原因证据台账.csv", PARENT / "results/104个月_完整观察.csv"]
    input_index = []
    for source in sources:
        destination = STUDY / "inputs" / source.name
        shutil.copy2(source, destination)
        input_index.append({"path": str(source), "sha256": digest(source), "copy": str(destination.relative_to(STUDY))})
    save_json(STUDY / "evidence/input_identity.json", {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "inputs": input_index})
    facts = extract_facts()
    save_csv(facts, "104个月_余额基数分解_不含未来标签.csv")
    fact_path = STUDY / "results/104个月_余额基数分解_不含未来标签.csv"
    save_json(STUDY / "evidence/feature_receipt.json", {"saved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "sha256": digest(fact_path), "rows": len(facts), "new_labels_not_yet_joined_in_this_run": True})
    parent = pd.read_csv(sources[-1])
    label_columns = [c for c in parent if c.startswith(("E0_", "E1_", "pre_publication_"))] + ["macro_status", "observation_date", "past_return20", "past_return60", "rv20", "downside20", "downside_change5", "internal_breadth20", "internal_breadth_change5", "delta3_spread_pp"]
    full = facts.merge(parent[["stat_month"] + label_columns], on="stat_month", how="left", validate="one_to_one")
    assert np.allclose(full.d3_spread_ordinary_pp, full.delta3_spread_pp, atol=EPS, rtol=0, equal_nan=True)
    save_csv(full, "104个月_机制分解与原冻结标签.csv")
    summary = summaries(full)
    save_csv(summary, "全部机制分组_描述分布.csv")
    primary = summary[(summary.horizon == 20) & summary.group.isin([GROUP_CURRENT, GROUP_BASE])]
    save_csv(primary, "主20日_当期余额支持与否.csv")
    checks = []
    for lag in [1, 3]:
        for metric in ["m1", "m2", "relative"]:
            lhs = full[f"d{lag}_{metric}_growth_log_pp"]
            rhs = full[f"d{lag}_{metric}_current_log_pp"] + full[f"d{lag}_{metric}_base_revision_log_pp"]
            residual = (lhs - rhs).abs().max()
            assert residual < EPS, (lag, metric, residual)
            checks.append({"identity": f"d{lag}_{metric}", "valid_months": int(lhs.notna().sum()), "max_abs_residual_log_pp": float(residual)})
    same = parent.set_index("stat_month").loc[full.stat_month]
    reused = [c for c in label_columns if c.startswith(("E0_", "E1_"))]
    for column in reused:
        pd.testing.assert_series_equal(full[column].reset_index(drop=True), same[column].reset_index(drop=True), check_names=False)
    save_json(STUDY / "verification_arithmetic.json", {"status": "PASS", "arithmetic_identities": checks, "monthly_sources_checked": len(full), "m0_extracted": int(full.m0_balance_100m.notna().sum()), "unchanged_parent_label_columns": len(reused), "vintage_note": "算术复核通过不等于旧时点不可修订快照成立。", "positions": "NOT_COMPUTED", "sharpe": "NOT_COMPUTED"})
    shutil.copy2(Path(__file__), STUDY / "code" / Path(__file__).name)
    print("104个月余额与基数分解完成，恒等式及原标签复用检查通过。")
    print(primary[["training_regime", "group", "entry", "mature_months", "return_mean", "return_median", "worst_path_mean", "worst_path_min"]].to_string(index=False))
    print("分组月份数：")
    print(full.groupby(["training_regime", "arithmetic_mechanism_group"]).size().to_string())


if __name__ == "__main__":
    run()
