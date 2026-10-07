"""510300宏观与波动首轮观察；仅生成历史描述、证据台账与可复算结果。"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
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
STUDY = ROOT if (ROOT / "inputs/market_daily.csv").exists() else ROOT / "reports/research/510300_macro_volatility_observation_v2_run1"
SOURCE = ROOT / "reports/research/510300_money_consensus_increment_v2"
CONFIG = ROOT / "config/510300_macro_volatility_observation_v2_run1.json"
REGIMES = ["M1_OLD_M2_MMF2018", "M1_NEW2025"]
EPS = 1e-10


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if x is pd.NA or x is pd.NaT:
        return None
    return x


def jsave(p: Path, x) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csave(d: pd.DataFrame, name: str) -> None:
    (STUDY / "results").mkdir(exist_ok=True)
    d.to_csv(STUDY / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def copy_file(src: Path, rel: str) -> None:
    dst = STUDY / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare() -> None:
    frozen = json.loads((STUDY / "freeze_receipt.json").read_text(encoding="utf-8-sig"))
    assert digest(CONFIG) == frozen["protocol_sha256"], "冻结方案发生变化"
    index = pd.read_csv(SOURCE / "FILE_INDEX.csv").set_index("path")
    selected = ["inputs/market_daily.csv", "inputs/official_releases.csv", "sources/104个月原始页面定位.csv"]
    loc = pd.read_csv(SOURCE / selected[-1])
    selected += loc.package_source_path.tolist()
    checks = []
    for rel in selected:
        src = SOURCE / rel
        assert src.stat().st_size == int(index.loc[rel, "bytes"]), rel
        assert digest(src) == index.loc[rel, "sha256"], rel
        copy_file(src, rel)
        checks.append({"path": rel, "sha256": digest(src), "bytes": src.stat().st_size})
    copy_file(CONFIG, "protocol.json")
    copy_file(SOURCE / "FILE_INDEX.csv", "evidence/parent_FILE_INDEX.csv")
    copy_file(SOURCE / "freeze_receipt.json", "evidence/parent_freeze_receipt.json")
    copy_file(ROOT / "config/510300_existing_data_training_mandate_v1.json", "evidence/current_authority_snapshot.json")
    copy_file(Path(__file__), "code/macro_volatility_observation_v2_run1.py")
    copy_file(Path(r"C:\Users\戴周阳\Downloads\104个月_观察底表.html"), "reference/104个月_观察底表.html")
    copy_file(Path(r"C:\Users\戴周阳\Downloads\510300_宏观与波动_规律观察_V2.md"), "reference/510300_宏观与波动_规律观察_V2.md")
    copy_file(Path(r"E:\CodexData\.codex\attachments\1e036087-e4f8-4ea4-8227-d0e1a20893e2\pasted-text-1.txt"), "reference/用户粘贴参考.txt")
    breadth = ROOT / "reports/research/510300_factor96_rapid_breadth_speed_v1"
    copy_file(breadth / "same_member_features.parquet", "inputs/internal_equal_weight_breadth.parquet")
    copy_file(breadth / "protocol.json", "evidence/internal_breadth_parent_protocol.json")
    jsave(STUDY / "evidence/input_identity.json", {
        "at": datetime.now(ZoneInfo("Asia/Shanghai")), "checked_files": checks,
        "original_archive_present": False,
        "scope": "旧ZIP当前路径未找到；直接验证保存目录107份输入和原文与旧FILE_INDEX一致。未宣称验证旧ZIP本体。",
        "new_price_downloads": 0, "reference_is_not_authority": True,
        "additional_input": {"path": "inputs/internal_equal_weight_breadth.parquet", "sha256": digest(STUDY / "inputs/internal_equal_weight_breadth.parquet")},
    })
    jsave(STUDY / "clock_clarification_before_run.json", {
        "at": datetime.now(ZoneInfo("Asia/Shanghai")), "new_results_read": False,
        "clarification": "除附件的21:00观察收盘前收益外，另存发布前最后收盘的20/60日收益及发布后第一收盘反应；H3分别披露，避免将当天公告反应算作提前反应。",
        "weekly_end": "沿用输入行情日历的每周末交易日；未重验完整交易日历。最后一个周为2026-09-11周五。"
    })
    print("输入快照已准备，107份源文件与旧索引一致。")


def body_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
    text = re.sub(r"\(\s*M\s*(\d)\s*\)", r"(M\1)", text)
    text = re.sub(r"([一二三四五六七八九十])\s+、", r"\1、", text)
    start = re.search(r"一[、.]", text)
    if start:
        text = text[start.start():]
    return re.split(r"打印本页|关闭窗口|法律声明", text)[0].strip()


def sections(text: str) -> list[str]:
    return [v.strip() for v in re.split(r"(?=[一二三四五六七八九十]+、)", text) if v.strip()]


def flow(block: str, label: str) -> tuple[float, str]:
    block = re.sub(r"\s+", "", block)
    found = re.search(label + r"(增加|减少)([\d.]+)(万亿元|亿元)", block)
    if not found:
        return np.nan, "未提取"
    value = float(found[2]) * (10000 if found[3] == "万亿元" else 1) * (1 if found[1] == "增加" else -1)
    return value, found[0]


def evidence() -> None:
    """该步骤只读公告，先于新价格标签生成；原因不会由后来涨跌倒推。"""
    d = pd.read_csv(STUDY / "inputs/official_releases.csv")
    rows = []
    for row in d.to_dict("records"):
        mon = row["stat_month"]
        text = body_text(STUDY / "sources/official_money" / f"{mon}.html")
        parts = sections(text)
        deposit = next((x for x in parts if re.match(r"[一二三四五六七八九十]+、[^。]{0,55}人民币存款", x)), "")
        loan = next((x for x in parts if re.match(r"[一二三四五六七八九十]+、[^。]{0,55}人民币贷款", x)), "")
        header = re.split(r"\s|。", deposit)[0]
        scope = "YTD" if re.search(r"前[一二三四五六七八九十\d]+个?月|上半年|全年|前三季度", header) else "MONTH"
        record = {"stat_month": mon, "available_at_upper_bound": row["available_at_upper_bound"],
                  "training_regime": row["training_regime"], "definition_version": row["definition_version"],
                  "source_url": row["source_url"], "source_sha256": row["source_sha256"],
                  "flow_period": scope, "flow_heading": header,
                  "deposit_context": deposit, "loan_context": loan,
                  "economic_cause": "未知/可混合", "cause_status": "NOT_IDENTIFIED",
                  "source_vintage": "RECONSTRUCTED_OFFICIAL_NOT_IMMUTABLE"}
        for code in ["M1", "M2"]:
            found = re.search(r"\(" + code + r"\)余额([\d.]+)(万亿元|亿元)", text)
            assert found, (mon, code)
            record[code.lower() + "_balance_100m"] = float(found[1]) * (10000 if found[2] == "万亿元" else 1)
            record[code.lower() + "_balance_evidence"] = found[0]
        for key, label, block in [
            ("household_deposit", r"住户存款", deposit),
            ("corporate_deposit", r"非金融企业存款", deposit),
            ("fiscal_deposit", r"财政性存款", deposit),
            ("nonbank_deposit", r"非银行业金融机构存款", deposit),
            ("household_loan", r"住户(?:部门)?贷款", loan),
            ("corporate_loan", r"(?:非金融企业及机关团体|企\(事\)业单位|企事业单位)贷款", loan),
        ]:
            record[key + "_flow_100m"], record[key + "_evidence"] = flow(block, label)
        enterprise = re.split(r"(?:非金融企业及机关团体|企\(事\)业单位|企事业单位)贷款", loan)
        enterprise_block = enterprise[1].split("非银行业金融机构")[0] if len(enterprise) > 1 else ""
        record["corporate_long_loan_flow_100m"], record["corporate_long_loan_evidence"] = flow(enterprise_block, "中长期贷款")
        record["methodology_notes"] = row["notes"]
        record["calendar_context"] = "一二月存在春节错位候选，未单因归因" if mon[-2:] in ["01", "02"] else ("季末资金结算候选，未单因归因" if mon[-2:] in ["03", "06", "09", "12"] else "无固定季节标签")
        record["mechanism_boundary"] = "企业存款含不同期限；贷款与存款增量只支持机制候选，不证明M1变化源自需求或资金进入股票。财政存款不是实际财政支出。"
        rows.append(record)
    e = pd.DataFrame(rows)
    e["h_log_m1_m2"] = np.log(e.m1_balance_100m / e.m2_balance_100m)
    e["balance_base_status"] = "NOT_COMPARABLE_OR_INSUFFICIENT_HISTORY"
    for i in range(13, len(e)):
        window = e.iloc[i - 13:i + 1]
        if window.definition_version.nunique() != 1:
            continue
        h = e.h_log_m1_m2
        current = h.iloc[i] - h.iloc[i - 1]
        base = -(h.iloc[i - 12] - h.iloc[i - 13])
        e.loc[i, "current_log_ratio_change"] = current
        e.loc[i, "prior_year_base_term"] = base
        e.loc[i, "delta12_log_ratio_change"] = current + base
        e.loc[i, "balance_base_status"] = "MIXED_RELEASE_VINTAGES_ARITHMETIC_ONLY"
    csave(e, "104个月_原因证据台账.csv")
    jsave(STUDY / "evidence/cause_extraction_receipt.json", {
        "at": datetime.now(ZoneInfo("Asia/Shanghai")), "rows": len(e),
        "future_market_labels_read": False, "unique_economic_causes_identified": 0,
        "flow_period_counts": e.flow_period.value_counts().to_dict(),
        "numeric_missing_counts": e.filter(regex="flow_100m$").isna().sum().to_dict(),
        "reason": "完成逐月原文事实首遍核对；未知不是遗漏月份，也不将分项同向当作唯一因果。"
    })
    print("104个月原因事实台账已生成，未来收益未用于贴原因标签。")


def features() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(STUDY / "inputs/market_daily.csv")
    m["date"] = pd.to_datetime(m.date)
    assert m.date.is_monotonic_increasing and not m.date.duplicated().any()
    assert (m[["open", "high", "low", "close"]] > 0).all().all()
    assert ((m.high >= m[["open", "close"]].max(axis=1)) & (m.low <= m[["open", "close"]].min(axis=1))).all()
    r = (m.close + m.dividend) / m.close.shift() - 1
    assert float((r - m.total_simple).abs().max()) < 1e-12
    m["recomputed_wealth"] = (1 + r.fillna(0)).cumprod()
    assert float((m.recomputed_wealth - m.wealth).abs().max()) < 1e-11
    for days in [20, 60]:
        m[f"past_return{days}"] = m.recomputed_wealth / m.recomputed_wealth.shift(days) - 1
    m["rv20"] = r.rolling(20).std(ddof=1) * np.sqrt(252)
    downside = r.clip(upper=0).pow(2).rolling(20).sum()
    m["downside20"] = np.sqrt(252 / 20 * downside)
    m["downside_change5"] = m.downside20 - m.downside20.shift(5)
    m["down_share20"] = downside / r.pow(2).rolling(20).sum()
    b = pd.read_parquet(STUDY / "inputs/internal_equal_weight_breadth.parquet")
    b["date"] = pd.to_datetime(b.date)
    b = b.set_index("date").reindex(m.date)
    b.loc[b.source_known.ne(True), ["E01_LEVEL", "E01_CHANGE5"]] = np.nan
    m["internal_breadth_source_date"] = pd.Series(m.date).shift(1)
    m["internal_breadth20"] = b.E01_LEVEL.shift(1).to_numpy()
    m["internal_breadth_change5"] = b.E01_CHANGE5.shift(1).to_numpy()
    m["internal_members"] = b.common_members.shift(1).to_numpy()
    a = pd.read_csv(STUDY / "inputs/official_releases.csv")
    # 公告为离散百分点值，先清除二进制尾差，避免把同值错误排成不同秩。
    for col in ["m1_yoy_pp", "m2_yoy_pp", "spread_pp"]:
        a[col] = a[col].round(10)
    for lag in [1, 3]:
        contiguous = pd.PeriodIndex(a.stat_month, freq="M").asi8 - pd.PeriodIndex(a.stat_month, freq="M").asi8[np.maximum(np.arange(len(a)) - lag, 0)] == lag
        strict = a.definition_version.eq(a.definition_version.shift(lag)) & contiguous
        legacy = a.training_regime.eq(a.training_regime.shift(lag)) & contiguous
        for key in ["m1", "m2"]:
            delta = a[key + "_yoy_pp"].diff(lag).round(10)
            a[f"delta{lag}_{key}_pp"] = delta.where(strict)
            a[f"reference_delta{lag}_{key}_pp"] = delta.where(legacy)
        a[f"delta{lag}_spread_pp"] = (a[f"delta{lag}_m1_pp"] - a[f"delta{lag}_m2_pp"]).round(10)
    return m, a


def macro_group(row) -> str:
    ds = row.get("delta3_spread_pp", np.nan)
    if pd.isna(ds):
        return "不可计算"
    if ds <= EPS:
        return "剪刀差未改善"
    one = max(row["delta3_m1_pp"], 0)
    two = max(-row["delta3_m2_pp"], 0)
    return "M1侧主导改善" if one > two + EPS else ("M2侧主导改善" if two > one + EPS else "两侧相等改善")


def snapshot(m: pd.DataFrame, arow: dict | None, snap: pd.Timestamp, origin: str, oid: str) -> dict:
    date = snap.tz_localize(None).normalize()
    i = int(m.date.searchsorted(date, side="right") - 1)
    assert i >= 0
    row = dict(arow or {})
    row.update({"origin": origin, "origin_id": oid, "snapshot_at": snap.isoformat(),
                "observation_date": m.at[i, "date"].strftime("%Y-%m-%d"), "market_index": i})
    for col in ["past_return20", "past_return60", "rv20", "downside20", "downside_change5", "down_share20", "internal_breadth20", "internal_breadth_change5", "internal_members"]:
        row[col] = m.at[i, col]
    source_date = m.at[i, "internal_breadth_source_date"]
    row["internal_breadth_source_date"] = source_date.strftime("%Y-%m-%d") if pd.notna(source_date) else None
    row["arithmetic_group"] = macro_group(row)
    row["volatility_group"] = ("此前上涨" if row["past_return20"] > 0 else "此前未涨") + ("/下行缓和" if row["downside_change5"] < -EPS else "/下行未缓和")
    row["improvement_prior_group"] = ("改善且此前上涨" if row["past_return20"] > 0 else "改善且此前未涨") if row.get("delta3_spread_pp", -1) > EPS else "其他或未知"
    row["macro_status"] = "RECONSTRUCTED_AVAILABLE" if arow else "NO_VIEW_NO_PUBLISHED_MACRO"
    row["period"] = "2018-2021" if row.get("stat_month", "9999") < "2022" else ("2022-2024" if row.get("stat_month", "9999") < "2025" else "2025-2026")
    if date > m.date.max():
        row["market_index"] = None
        row["observation_date"] = None
        row["macro_status"] = "NO_VIEW_SNAPSHOT_BEYOND_MARKET_CUTOFF"
        for col in ["past_return20", "past_return60", "rv20", "downside20", "downside_change5", "down_share20", "internal_breadth20", "internal_breadth_change5", "internal_members"]:
            row[col] = np.nan
        row["volatility_group"] = "不可计算"
        row["improvement_prior_group"] = "其他或未知"
    return row


def labels(m: pd.DataFrame, row: dict) -> dict:
    result = {}
    i = row["market_index"]
    for delay in [0, 1]:
        entry = i + 1 + delay if i is not None else len(m)
        for horizon in [5, 20, 60]:
            prefix = f"E{delay}_{horizon}"
            end = entry + horizon - 1
            result[prefix + "_entry_date"] = m.at[entry, "date"].strftime("%Y-%m-%d") if entry < len(m) else None
            result[prefix + "_exit_date"] = m.at[end, "date"].strftime("%Y-%m-%d") if end < len(m) else None
            result[prefix + "_status"] = "MATURE" if end < len(m) else "PENDING_MARKET_WINDOW"
            if end >= len(m):
                for key in ["return", "worst_close", "worst_path", "best_path", "drawdown"]:
                    result[prefix + "_" + key] = np.nan
                continue
            path = m.iloc[entry:end + 1]
            div = path.dividend.copy()
            div.iloc[0] = 0
            wealth = (path.close.to_numpy() + div.cumsum().to_numpy()) / m.at[entry, "open"]
            anchored = np.r_[1.0, wealth]
            result[prefix + "_return"] = wealth[-1] - 1
            result[prefix + "_worst_close"] = wealth.min() - 1
            result[prefix + "_worst_path"] = anchored.min() - 1
            result[prefix + "_best_path"] = anchored.max() - 1
            result[prefix + "_drawdown"] = np.min(anchored / np.maximum.accumulate(anchored) - 1)
    return result


def panels(m: pd.DataFrame, a: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly, weekly = [], []
    for ar in a.to_dict("records"):
        available = pd.Timestamp(ar["available_at_upper_bound"])
        snap = available.normalize() + pd.Timedelta(hours=21)
        if snap < available:
            snap += pd.Timedelta(days=1)
        row = snapshot(m, ar, snap, "月度", ar["stat_month"])
        pre = int((m.date + pd.Timedelta(hours=15)).searchsorted(available.tz_localize(None), side="left") - 1)
        for h in [20, 60]:
            row[f"pre_publication_return{h}"] = m.at[pre, f"past_return{h}"] if row["market_index"] is not None else np.nan
        row["pre_publication_close_date"] = m.at[pre, "date"].strftime("%Y-%m-%d") if row["market_index"] is not None else None
        row["first_reaction_close_date"] = m.at[pre + 1, "date"].strftime("%Y-%m-%d") if pre + 1 < len(m) else None
        row["first_reaction_return"] = m.at[pre + 1, "recomputed_wealth"] / m.at[pre, "recomputed_wealth"] - 1 if pre + 1 < len(m) else np.nan
        row.update(labels(m, row))
        monthly.append(row)
    origins = m.loc[m.date >= "2018-01-01"].groupby(m.loc[m.date >= "2018-01-01", "date"].dt.to_period("W-FRI")).tail(1)
    times = pd.to_datetime(a.available_at_upper_bound)
    for date in origins.date:
        snap = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21)
        j = int(times.searchsorted(snap, side="right") - 1)
        ar = a.iloc[j].to_dict() if j >= 0 else None
        row = snapshot(m, ar, snap, "周度", date.strftime("%Y-%m-%d"))
        row.update(labels(m, row))
        weekly.append(row)
    md, wd = pd.DataFrame(monthly), pd.DataFrame(weekly)
    for frame in [md, wd]:
        for h in [5, 20, 60]:
            frame[f"delay_change_{h}"] = frame[f"E1_{h}_return"] - frame[f"E0_{h}_return"]
    return md, wd


def reference_check(md: pd.DataFrame) -> dict:
    soup = BeautifulSoup((STUDY / "reference/104个月_观察底表.html").read_text(encoding="utf-8"), "html.parser")
    cells = [[x.get_text(strip=True) for x in tr.find_all("td")] for tr in soup.select("tbody tr")]
    errors = []
    count = 0
    for values in cells:
        row = md.set_index("stat_month").loc[values[0]]
        for idx, key in [(3, "m1_yoy_pp"), (4, "m2_yoy_pp"), (5, "spread_pp"), (6, "reference_delta3_m1_pp"), (7, "reference_delta3_m2_pp"), (9, "past_return20"), (10, "rv20"), (11, "downside20"), (13, "E0_20_return"), (14, "E0_20_worst_path"), (15, "E0_20_best_path"), (16, "E1_20_return")]:
            if values[idx] in ["—", "", "待成熟"]:
                if pd.notna(row[key]):
                    errors.append([values[0], key, "附件缺失", row[key]])
            else:
                percentage = "%" in values[idx]
                target = float(values[idx].replace("%", "")) / (100 if percentage else 1)
                tol = 0.0000501 if percentage else 0.00501
                if pd.isna(row[key]) or abs(row[key] - target) > tol:
                    errors.append([values[0], key, target, row[key]])
                count += 1
        for idx, key in [(8, "observation_date"), (12, "E0_20_entry_date")]:
            if values[idx] != row[key] and not (values[idx] == "—" and pd.isna(row[key])):
                errors.append([values[0], key, values[idx], row[key]])
    result = {"rows": len(cells), "numeric_cells_checked": count, "rounding_tolerance": "百分点0.00501、收益率0.0000501", "mismatches": errors,
              "strict_definition_delta_exclusions": md.loc[md.delta3_m1_pp.isna() & md.reference_delta3_m1_pp.notna(), "stat_month"].tolist()}
    jsave(STUDY / "results/reference_recomputation.json", result)
    assert not errors, "与参考HTML存在差异，先检查定义再比较结果"
    return result


def summaries(md: pd.DataFrame, wd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame in [md, wd]:
        frame = frame.loc[frame.training_regime.notna()].copy()
        for (regime, period), df in list(frame.groupby(["training_regime", "period"])) + [((r, "全期"), x) for r, x in frame.groupby("training_regime")]:
            for groupcol in ["arithmetic_group", "volatility_group", "improvement_prior_group"]:
                for name, group in df.groupby(groupcol):
                    for delay in [0, 1]:
                        for h in [5, 20, 60]:
                            prefix = f"E{delay}_{h}"
                            y = group.dropna(subset=[prefix + "_return"])
                            if y.empty:
                                continue
                            values = y[prefix + "_return"]
                            cyc = y.groupby("stat_month")[prefix + "_return"].mean()
                            positives = values.clip(lower=0)
                            rows.append({"origin": group.origin.iloc[0], "training_regime": regime, "period": period, "group_variable": groupcol, "group": name, "delay": delay, "horizon": h,
                                         "rows": len(y), "release_cycles": y.stat_month.nunique(), "mean": values.mean(), "median": values.median(), "q10": values.quantile(.1), "q25": values.quantile(.25), "q75": values.quantile(.75), "q90": values.quantile(.9), "positive_fraction": (values > 0).mean(), "cycle_equal_mean": cyc.mean(),
                                         "worst_close_median": y[prefix + "_worst_close"].median(), "worst_path_minimum": y[prefix + "_worst_path"].min(), "drawdown_median": y[prefix + "_drawdown"].median(), "delay_change_mean": y[f"delay_change_{h}"].mean(),
                                         "best_one_share_of_positive_sum": positives.max() / positives.sum() if positives.sum() > 0 else np.nan,
                                         "mean_without_best_one": values.drop(values.idxmax()).mean() if len(values) > 1 else np.nan})
    return pd.DataFrame(rows)


def correlations(md: pd.DataFrame, wd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame in [md, wd]:
        for regime, part in frame.groupby("training_regime"):
            for period in ["全期"] + list(part.period.unique()):
                df = part if period == "全期" else part.loc[part.period == period]
                xs = ["spread_pp", "delta3_spread_pp", "delta3_m1_pp", "delta3_m2_pp"] if frame is md else ["rv20", "downside20", "downside_change5", "down_share20"]
                ys = (["pre_publication_return20", "pre_publication_return60", "first_reaction_return"] if frame is md else []) + [f"E{d}_{h}_return" for d in [0, 1] for h in [5, 20, 60]] + ["E0_20_worst_close"]
                for x in xs:
                    for y in ys:
                        z = df[[x, y, "stat_month"]].dropna()
                        value = z[x].corr(z[y], method="spearman") if len(z) >= 3 and z[x].nunique() > 1 and z[y].nunique() > 1 else np.nan
                        rows.append({"origin": df.origin.iloc[0], "training_regime": regime, "period": period, "x": x, "y": y, "rows": len(z), "release_cycles": z.stat_month.nunique(), "spearman": value, "status": "DESCRIPTIVE_NO_IID_P_VALUE"})
    return pd.DataFrame(rows)


def match_history(wd: pd.DataFrame, question: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = json.loads((STUDY / "protocol.json").read_text(encoding="utf-8"))["matching"]
    columns = cfg[question + "_features"]
    group_feature = "downside_change5" if question == "H2" else "delta3_spread_pp"
    targets = wd[group_feature] < -EPS if question == "H2" else wd[group_feature] > EPS
    controls = wd[group_feature] >= -EPS if question == "H2" else wd[group_feature] <= EPS
    entries, matches = [], []
    for idx, row in wd.loc[targets & wd.training_regime.notna()].iterrows():
        item = {"question": question, "origin_id": row.origin_id, "stat_month": row.stat_month, "training_regime": row.training_regime, "period": row.period,
                "status": "NO_MATCH", "target_label_status": row.E0_20_status}
        pool = wd.loc[controls & wd.training_regime.eq(row.training_regime) & wd.stat_month.ne(row.stat_month)
                      & wd.E1_20_exit_date.notna() & (wd.E1_20_exit_date < row.observation_date)].dropna(subset=columns).copy()
        if len(pool) < 3 or row[columns].isna().any():
            entries.append(item)
            continue
        scales = pool[columns].std(ddof=1).replace(0, np.nan)
        if scales.isna().any():
            entries.append(item)
            continue
        pool["distance"] = np.sqrt(((pool[columns] - row[columns].astype(float)) / scales).pow(2).mean(axis=1))
        chosen = pool.loc[pool.distance <= 1].sort_values(["distance", "origin_id"]).drop_duplicates("stat_month").head(3)
        if len(chosen) < 3:
            entries.append(item)
            continue
        item.update({"status": "MATCHED", "control_cycles": len(chosen), "max_distance": chosen.distance.max(), "control_ids": "|".join(chosen.origin_id)})
        for delay in [0, 1]:
            for metric in ["return", "worst_close", "drawdown"]:
                col = f"E{delay}_20_{metric}"
                item[col + "_target"] = row[col]
                item[col + "_control"] = chosen[col].mean()
                item[col + "_difference"] = row[col] - chosen[col].mean()
        for c in chosen.to_dict("records"):
            matches.append({"question": question, "target_id": row.origin_id, "target_cycle": row.stat_month, "target_observation_date": row.observation_date,
                            "control_id": c["origin_id"], "control_cycle": c["stat_month"], "control_label_end": c["E1_20_exit_date"], "distance": c["distance"],
                            **{f"scale_{x}": scales[x] for x in columns}})
        entries.append(item)
    return pd.DataFrame(entries), pd.DataFrame(matches)


def uncertainty(md: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260928)
    rows, saved = [], {}
    for regime, universe in md.groupby("training_regime", sort=True):
        cycles = universe.stat_month.tolist()
        n = len(cycles)
        block = 6
        starts = rng.integers(0, max(n - block + 1, 1), size=(2000, math.ceil(n / block)))
        draw = (starts[:, :, None] + np.arange(block)).reshape(2000, -1)[:, :n]
        draw = np.minimum(draw, n - 1)
        saved[regime] = draw
        cases = []
        for question, groupcol, g1, g0 in [("H1", "arithmetic_group", "M1侧主导改善", "M2侧主导改善"), ("H3", "improvement_prior_group", "改善且此前上涨", "改善且此前未涨")]:
            for delay in [0, 1]:
                col = f"E{delay}_20_return"
                x = universe.set_index("stat_month")[col].where(universe.set_index("stat_month")[groupcol] == g1).reindex(cycles).to_numpy()
                y = universe.set_index("stat_month")[col].where(universe.set_index("stat_month")[groupcol] == g0).reindex(cycles).to_numpy()
                cases.append((question, f"E{delay}", "return", x, y))
        for question in ["H2", "H4"]:
            part = matched.loc[(matched.question == question) & (matched.training_regime == regime) & (matched.status == "MATCHED")]
            for delay in [0, 1]:
                for metric in ["return", "worst_close", "drawdown"]:
                    col = f"E{delay}_20_{metric}_difference"
                    series = part.groupby("stat_month")[col].mean() if not part.empty else pd.Series(dtype=float)
                    x = series.reindex(cycles).to_numpy()
                    y = np.where(np.isfinite(x), 0., np.nan)
                    cases.append((question, f"E{delay}", metric, x, y))
        for question, delay, metric, x, y in cases:
            nx, ny = int(np.isfinite(x).sum()), int(np.isfinite(y).sum())
            xd, yd = x[draw], y[draw]
            nxd, nyd = np.isfinite(xd).sum(axis=1), np.isfinite(yd).sum(axis=1)
            valid = (nxd > 0) & (nyd > 0)
            delta = np.nansum(xd[valid], axis=1) / nxd[valid] - np.nansum(yd[valid], axis=1) / nyd[valid]
            point = np.nanmean(x) - np.nanmean(y) if nx and ny else np.nan
            result = {"question": question, "training_regime": regime, "delay": delay, "horizon": 20, "metric": metric, "target_cycles": nx, "control_cycles": ny,
                      "mean_difference": point, "ci_low": np.quantile(delta, .025) if len(delta) else np.nan, "ci_high": np.quantile(delta, .975) if len(delta) else np.nan,
                      "bootstrap_valid_draws": len(delta), "status": "DESCRIPTIVE_DEPENDENT_HISTORY" if min(nx, ny) >= 12 else "SMALL_SAMPLE_DESCRIPTIVE_ONLY"}
            for period in universe.period.unique():
                mask = universe.period.eq(period).to_numpy()
                result["difference_" + period] = np.nanmean(x[mask]) - np.nanmean(y[mask]) if np.isfinite(x[mask]).any() and np.isfinite(y[mask]).any() else np.nan
            rows.append(result)
    np.savez_compressed(STUDY / "results/fixed_block_draws.npz", **saved)
    return pd.DataFrame(rows)


def selected_cases(md: pd.DataFrame, wd: pd.DataFrame) -> pd.DataFrame:
    records = []
    for frame, g in [(md, "arithmetic_group"), (wd, "volatility_group")]:
        for (regime, group), d in frame.groupby(["training_regime", g]):
            d = d.dropna(subset=["E0_20_return"]).copy()
            if d.empty:
                continue
            choices = [(i, "最差案例") for i in d.nsmallest(2, "E0_20_return").index]
            choices += [(d.E0_20_return.idxmax(), "最好案例"), ((d.E0_20_return - d.E0_20_return.median()).abs().idxmin(), "中位附近案例")]
            for i, role in choices:
                row = d.loc[i]
                records.append({"origin": row.origin, "training_regime": regime, "group": group, "selection": role,
                                **{k: row[k] for k in ["origin_id", "stat_month", "observation_date", "past_return20", "rv20", "downside20", "downside_change5", "delta3_spread_pp", "E0_20_return", "E0_20_worst_close", "E0_20_drawdown", "E1_20_return", "source_url"]},
                                "case_use": "事后解释和反例，未用于原因标签或排除样本"})
    return pd.DataFrame(records)


def run() -> None:
    m, a = features()
    md, wd = panels(m, a)
    check = reference_check(md)
    csave(m, "daily_features.csv")
    csave(md, "104个月_完整观察.csv")
    csave(wd, "固定周度_完整观察.csv")
    csave(summaries(md, wd), "全部分组分布.csv")
    csave(correlations(md, wd), "全部连续关联.csv")
    h2, p2 = match_history(wd, "H2")
    h4, p4 = match_history(wd, "H4")
    matched, pairs = pd.concat([h2, h4], ignore_index=True), pd.concat([p2, p4], ignore_index=True)
    csave(matched, "历史近邻_全部目标与缺口.csv")
    csave(pairs, "历史近邻_全部配对.csv")
    csave(uncertainty(md, matched), "主20日_固定比较与区间.csv")
    csave(selected_cases(md, wd), "典型与失败案例.csv")
    brief = {"study_id": "510300_MACRO_VOLATILITY_OBSERVATION_V2_RUN1", "status": "FIRST_PASS_OBSERVATION_COMPLETED_NO_STRATEGY_CLAIM",
             "market_rows": len(m), "market_cutoff": m.date.max().strftime("%Y-%m-%d"), "monthly_rows": len(md), "weekly_rows": len(wd), "regime_counts": md.training_regime.value_counts().to_dict(),
             "mature_monthly_E0_20": int(md.E0_20_return.notna().sum()), "mature_monthly_E1_20": int(md.E1_20_return.notna().sum()),
             "mature_weekly_E0_20": int(wd.E0_20_return.notna().sum()), "weekly_no_macro": int(wd.training_regime.isna().sum()),
             "strict_definition_delta_exclusions": check["strict_definition_delta_exclusions"],
             "weekly_internal_breadth_available": int(wd.internal_breadth20.notna().sum()),
             "matched_counts": matched.groupby(["question", "training_regime", "status"]).size().to_dict(),
             "accounts": 0, "sharpe": "NOT_COMPUTED", "new_market_downloads": 0, "independent_forward_events": 0,
             "current_view": "NO_VIEW_HISTORICAL_CUTOFF", "orders_authorized": False, "delivery_package_required": False}
    jsave(STUDY / "results/summary.json", brief)
    print(json.dumps(clean(brief), ensure_ascii=False, indent=2))


def main() -> None:
    global STUDY
    parser = argparse.ArgumentParser(description="510300宏观与波动观察复算")
    parser.add_argument("stage", choices=["prepare", "evidence", "run"])
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    STUDY = args.study.resolve()
    {"prepare": prepare, "evidence": evidence, "run": run}[args.stage]()


if __name__ == "__main__":
    main()
