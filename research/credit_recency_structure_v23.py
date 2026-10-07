"""按固定区间复原信贷净增，区别累计支持、近期变化与期限结构。"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_credit_recency_structure_v23"
TZ = ZoneInfo("Asia/Shanghai")
FIELDS = ["rmb_total", "corporate_total", "corporate_long", "corporate_short", "bills", "household_total", "household_long", "household_short", "nonbank_total"]
INPUTS = {
    "monthly.csv": "reports/research/510300_macro_transmission_context_v4/results/104个月_多层证据与原后续路径.csv",
    "credit.csv": "reports/research/510300_macro_transmission_context_v4/results/104个月_信贷分项与同区间比较.csv",
    "originals.json": "reports/research/510300_macro_transmission_context_v4/inputs/loan_originals.json",
    "money_causes.csv": "reports/research/510300_macro_transmission_context_v4/inputs/money_causes.csv",
    "funding.csv": "reports/research/510300_funding_quantity_price_bridge_v12/results/104个月_货币数量与融资价格.csv",
    "industrial.csv": "reports/research/510300_orders_cost_profit_bridge_v21/results/104个月_订单价格成本与原后续路径.csv",
    "previous_completion.json": "reports/research/510300_within_release_volatility_v22/completion.json",
    "authority_snapshot.json": "config/510300_existing_data_training_mandate_v1.json",
}
SUPPORTED = "剪刀差改善_信贷与订单共同支持"


def now():
    return datetime.now(TZ).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save(name, obj):
    (OUT / name).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame, name):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def freeze():
    if OUT.exists():
        raise RuntimeError("本轮目录已存在，不覆盖冻结输入。")
    for part in ["inputs", "sources", "results", "code", "figures"]:
        (OUT / part).mkdir(parents=True, exist_ok=True)
    protocol = {
        "at": now(), "study": "510300_CREDIT_RECENCY_STRUCTURE_V23", "previous_turn": "PROGRESS",
        "question": "累计中长期多增是否同时伴随最近三个月多增及企事业融资总额多增；与既有订单、利润、资金价格及已实现定价怎样并存。",
        "universe": "原104个月全部保留；原11个共同支持月份全部逐月列示，不删除亏损月份。",
        "outcomes_previously_seen": True, "independent_validation": False,
        "primary_window": "最近连续三个统计月，与M1三个月变化并列；非季节调整。单月和年内累计均保留，不选较优期限。",
        "construction": [
            "年内累计沿原公告优先直接累计，否则逐月累加至下次直接累计重置；每一原文数值保留唯一系数。",
            "当月优先使用本月原文明确单月区间，否则用本月累计减上月累计；不反用未来公告修补缺失。",
            "三个月按年内累计端点相减，跨年拆成两个年内区间；一月的零年初项不引入上年余额。",
            "同一原文项的系数先相消，再计算金额及显示舍入界；未披露项只在系数非零时造成缺失。",
            "同比为当前和去年对应区间的跨公告差额，不冒充同版本官方可比同比。2023扩围及混合区间拒绝跨范围比较。",
            "已知时点使用同源金融统计报告原可用上界；2026年8月在行情截止日后，结构原值保留，历史观察仍NO_VIEW。",
        ],
        "comparison": "9类贷款，单月、最近三月、年内累计；连续金额和全部方向交叉，不拟合新预测，不优化分组。方向只依显示舍入界。",
        "mechanism_limits": [
            "贷款增加是净增量，不能直接称新发放、未偿债前融资、实际投资、消费或收入。",
            "企事业中长期不等于民营制造业投资；住户中长期不等于全部住房新购支出。",
            "企事业未单列项与舍入残差完整保存，不把票据变化直接指定为置换。",
            "同一公告并不唯一识别供给与需求，也不证明贷款创造的存款实际流向。",
        ],
        "outcome_use": "原E0/E1与5/20/60日标签原样保留。20日主结果仅用于原11病例完整复盘；无新增收益分组排名、模型或账户。",
        "context": "所有原389列保留；补入第12轮资金价与第21轮当时可见利润率，未来工业报告不进入原点背景。",
        "stops": "不新增行情、不新设牛熊阈值、不改变旧失败规则；缺资料留缺失。", "goal_achieved": False,
    }
    save("protocol.json", protocol)
    receipts = []
    for name, source in INPUTS.items():
        dst = OUT / "inputs" / name
        shutil.copy2(ROOT / source, dst)
        receipts.append({"name": name, "source": source, "sha256": sha(dst)})
    records = json.loads((OUT / "inputs/originals.json").read_text("utf-8"))
    sources = []
    for r in records:
        src = ROOT / r["raw_path"]
        assert sha(src) == r["source_sha256"]
        dst = OUT / "sources" / (r["stat_month"] + ".html")
        shutil.copy2(src, dst)
        sources.append({"month": r["stat_month"], "source_url": r["source_url"], "source_sha256": sha(dst), "historical_first_vintage_verified": False})
    save("input_receipts.json", receipts)
    save("source_receipts.json", sources)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save("freeze.json", {"at": now(), "protocol_sha256": sha(OUT / "protocol.json"), "sources": len(sources), "input_files": len(receipts), "code_sha256": sha(__file__)})
    print("已冻结104个月、9类贷款和三种固定区间；本轮不建立交易条件。")


def add(*items):
    result = Counter()
    for coeff, expr in items:
        for key, value in expr.items():
            result[key] += coeff * value
    return {k: v for k, v in result.items() if v}


class Ledger:
    def __init__(self):
        self.originals = json.loads((OUT / "inputs/originals.json").read_text("utf-8"))
        self.by_month = {r["stat_month"]: r for r in self.originals}
        self.monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
        self.atoms, self.cum, self.direct_month, self.atom_rows = {}, {}, {}, []
        for r in self.originals:
            m = r["stat_month"]
            blocks = r["all_observed_blocks"]
            # 固定使用原提取层已选定的累计/单月区间。
            chosen = [i for i, b in enumerate(blocks) if b["reported_interval"] == r["reported_interval"]]
            assert len(chosen) == 1
            for i, b in enumerate(blocks):
                for f in FIELDS:
                    atom = b["fields"][f]
                    key = f"{m}__{i}__{f}"
                    row = {"key": key, "source_month": m, "block": i, "field": f,
                           "start": b["period_start"], "end": b["period_end"], "interval": b["reported_interval"],
                           "value": atom["value_yi"] if atom else np.nan,
                           "rounding_half": atom["display_rounding_half_yi"] if atom else np.nan,
                           "literal": atom["literal"] if atom else "原文未单列", "regime": r["statistical_regime"],
                           "known_at": self.monthly.loc[m, "available_at_upper_bound"], "source_sha256": r["source_sha256"]}
                    self.atoms[key] = row
                    self.atom_rows.append(row)
            for f in FIELDS:
                key = f"{m}__{chosen[0]}__{f}"
                if r["reported_interval"] == "YEAR_TO_DATE" or m.endswith("-01"):
                    self.cum[(m, f)] = {key: 1}
                else:
                    prev = str(pd.Period(m, "M") - 1)
                    self.cum[(m, f)] = add((1, self.cum[(prev, f)]), (1, {key: 1}))
                direct = [i for i, b in enumerate(blocks) if b["period_start"] == m and b["period_end"] == m]
                assert len(direct) <= 1
                if direct:
                    self.direct_month[(m, f)] = {f"{m}__{direct[0]}__{f}": 1}

    def span(self, m, f, window):
        end = pd.Period(m, "M")
        start = pd.Period(f"{end.year}-01", "M") if window == "YTD" else end - int(window) + 1
        members = [str(v) for v in pd.period_range(start, end, freq="M")]
        if any(v not in self.by_month for v in members):
            return None, members, "NO_VIEW_HISTORY"
        if window == "1" and (m, f) in self.direct_month:
            return self.direct_month[(m, f)], members, "原文直接当月"
        expr = {}
        for year in sorted({v.year for v in pd.period_range(start, end, freq="M")}):
            s = max(start, pd.Period(f"{year}-01", "M"))
            e = min(end, pd.Period(f"{year}-12", "M"))
            piece = self.cum[(str(e), f)]
            if s.month != 1:
                prior = str(s - 1)
                if (prior, f) not in self.cum:
                    return None, members, "NO_VIEW_HISTORY"
                piece = add((1, piece), (-1, self.cum[(prior, f)]))
            expr = add((1, expr), (1, piece))
        return expr, members, "累计端点差或跨年分段"

    def evaluate(self, expr):
        if expr is None:
            return np.nan, np.nan, None, "NO_VIEW_HISTORY"
        atoms = [self.atoms[k] for k in expr]
        if any(not np.isfinite(a["value"]) for a in atoms):
            return np.nan, np.nan, max(a["known_at"] for a in atoms), "NO_VIEW_COMPONENT_NOT_REPORTED"
        return (sum(expr[k] * self.atoms[k]["value"] for k in expr),
                sum(abs(expr[k]) * self.atoms[k]["rounding_half"] for k in expr),
                max((a["known_at"] for a in atoms), default=None), "AVAILABLE_CROSS_REPORT_RECONSTRUCTION")

    def row(self, m, f, window):
        expr, members, method = self.span(m, f, window)
        v, b, known, status = self.evaluate(expr)
        regimes = sorted({self.by_month[x]["statistical_regime"] for x in members if x in self.by_month})
        if len(regimes) > 1:
            status, v, b = "NO_VIEW_MIXED_SCOPE", np.nan, np.nan
        return {"stat_month": m, "field": f, "window": window, "start": members[0], "end": members[-1],
                "value_yi": v, "rounding_half_yi": b, "known_at": known, "status": status, "method": method,
                "regime": "|".join(regimes), "expression": json.dumps(expr, ensure_ascii=False, sort_keys=True),
                "source_terms": len(expr) if expr is not None else 0,
                "contains_subtraction": any(v < 0 for v in expr.values()) if expr is not None else False}


def direction(value, bound):
    if not np.isfinite(value + bound):
        return "NO_VIEW"
    return "多增" if value > bound else "少增" if value < -bound else "显示舍入界内"


def build():
    if (OUT / "build_receipt.json").exists():
        raise RuntimeError("已生成本轮金额与背景，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text("utf-8"))
    assert frozen["protocol_sha256"] == sha(OUT / "protocol.json")
    for r in json.loads((OUT / "input_receipts.json").read_text("utf-8")):
        assert sha(OUT / "inputs" / r["name"]) == r["sha256"]
    ledger = Ledger()
    rows = [ledger.row(m, f, w) for m in ledger.by_month for w in ["1", "3", "YTD"] for f in FIELDS]
    panel = pd.DataFrame(rows)
    lookup = {(r["stat_month"], r["field"], r["window"]): r for r in rows}
    comparisons = []
    for r in rows:
        prev = str(pd.Period(r["stat_month"], "M") - 12)
        p = lookup.get((prev, r["field"], r["window"]))
        value, bound, status = np.nan, np.nan, "NO_VIEW_PRIOR_YEAR_HISTORY"
        expr = None
        known = None
        if p is not None:
            if not r["status"].startswith("AVAILABLE") or not p["status"].startswith("AVAILABLE"):
                status = "NO_VIEW_INTERVAL_UNAVAILABLE"
            elif r["regime"] != p["regime"]:
                status = "NO_VIEW_2023_SCOPE_BREAK"
            else:
                expr = add((1, json.loads(r["expression"])), (-1, json.loads(p["expression"])))
                value, bound, known, status = ledger.evaluate(expr)
        comparisons.append({"stat_month": r["stat_month"], "field": r["field"], "window": r["window"],
                            "current_start": r["start"], "prior_start": p["start"] if p else None,
                            "current_yi": r["value_yi"], "prior_yi": p["value_yi"] if p else np.nan,
                            "yoy_change_yi": value, "rounding_bound_yi": bound, "direction": direction(value, bound),
                            "status": status, "known_at": known, "expression": json.dumps(expr, sort_keys=True)})
    comp = pd.DataFrame(comparisons)
    csv(pd.DataFrame(ledger.atom_rows), "963项_原文区间金额与显示精度.csv")
    csv(panel, "2808项_当月三月累计净增与公式.csv")
    csv(comp, "2808项_同区间跨公告同比比较.csv")
    # 宽表仅连接当时可见的新背景，不使用未来报告生成分类。
    original = pd.read_csv(OUT / "inputs/monthly.csv")
    wide = original.copy()
    indexed = comp.set_index(["stat_month", "window", "field"])
    newcols = {}
    for w in ["1", "3", "YTD"]:
        for f in FIELDS:
            for item in ["current_yi", "prior_yi", "yoy_change_yi", "rounding_bound_yi", "direction", "status", "known_at"]:
                newcols[f"credit_{w}_{f}_{item}"] = [indexed.loc[(m, w, f), item] for m in original.stat_month]
    wide = pd.concat([wide, pd.DataFrame(newcols)], axis=1)
    industrial = pd.read_csv(OUT / "inputs/industrial.csv").set_index("stat_month")
    fund = pd.read_csv(OUT / "inputs/funding.csv").set_index("stat_month")
    for c in [c for c in industrial if c.startswith("industrial_0_")]:
        wide[c] = [industrial.loc[m, c] for m in wide.stat_month]
    for c in ["funding_status", "fixing_date", "fixing_known_at", "fdr_policy_gap_bp_mean_change", "fr_fdr_gap_bp_mean_change", "bond_date", "bond_known_at", "bond_change20_bp"]:
        wide["context_" + c] = [fund.loc[m, c] for m in wide.stat_month]
    for w in ["1", "3", "YTD"]:
        a = wide[f"credit_{w}_corporate_long_direction"]
        b = wide[f"credit_{w}_household_long_direction"]
        wide[f"credit_{w}_both_long"] = np.where(a.eq("NO_VIEW") | b.eq("NO_VIEW"), "NO_VIEW", np.where(a.eq("多增") & b.eq("多增"), "两部门中长期共同多增", "至少一方未确认多增"))
    wide["credit_origin_admission"] = np.where(wide.available_before_market_cutoff, "原观察可用", "NO_VIEW_AFTER_MARKET_CUTOFF")
    csv(wide, "104个月_信用近期性与完整经营定价背景.csv")
    fixed = wide[wide.joint_credit_orders_state.eq(SUPPORTED)]
    assert len(fixed) == 11
    csv(fixed, "原11个共同支持月份_全部背景与原路径.csv")
    # 企事业余额净增的未单列项必须保留，不能把三类分项强制归一。
    residuals = []
    for m in ledger.by_month:
        for w in ["1", "3", "YTD"]:
            terms = [lookup[(m, f, w)] for f in ["corporate_total", "corporate_long", "corporate_short", "bills"]]
            expr = None
            if all(t["status"].startswith("AVAILABLE") for t in terms):
                expr = add(*[(1 if i == 0 else -1, json.loads(t["expression"])) for i, t in enumerate(terms)])
            v, b, known, status = ledger.evaluate(expr)
            residuals.append({"stat_month": m, "window": w, "corporate_unlisted_or_rounding_yi": v,
                              "display_rounding_bound_yi": b, "within_display_rounding": abs(v) <= b if np.isfinite(v+b) else None,
                              "status": status, "expression": json.dumps(expr, sort_keys=True)})
    csv(pd.DataFrame(residuals), "312组_企事业未单列项与显示舍入范围.csv")
    counts = []
    for f in FIELDS:
        for period, group in wide.groupby("analysis_period", dropna=False):
            cross = group.groupby([f"credit_YTD_{f}_direction", f"credit_3_{f}_direction"], dropna=False).size()
            for (cum, recent), n in cross.items():
                counts.append({"field": f, "analysis_period": period, "ytd_direction": cum, "recent3_direction": recent, "months": int(n)})
    csv(pd.DataFrame(counts), "全部贷款类型_累计与最近三月方向交叉.csv")
    # 原文同时报告月度与累计时，保留两种合法重建的差，说明版本/显示误差。
    dual = []
    for m, raw in ledger.by_month.items():
        if len(raw["all_observed_blocks"]) <= 1:
            continue
        prev = str(pd.Period(m, "M") - 1)
        for f in FIELDS:
            actual = ledger.direct_month[(m, f)]
            implied = add((1, ledger.cum[(m, f)]), (-1, ledger.cum[(prev, f)]))
            gap = add((1, actual), (-1, implied))
            v, b, _, status = ledger.evaluate(gap)
            dual.append({"stat_month": m, "field": f, "direct_month_yi": ledger.evaluate(actual)[0],
                         "cumulative_difference_yi": ledger.evaluate(implied)[0], "difference_yi": v,
                         "rounding_bound_yi": b, "status": status, "expression": json.dumps(gap, sort_keys=True)})
    csv(pd.DataFrame(dual), "27项_同文单月与累计差分并列.csv")
    save("build_receipt.json", {"at": now(), "status": "BUILT_FIXED_CREDIT_MECHANISM_LEDGER", "months": len(wide),
         "source_atoms": len(ledger.atom_rows), "interval_rows": len(panel), "comparison_rows": len(comp),
         "comparison_status": comp.status.value_counts().to_dict(), "original_supported_months": len(fixed),
         "old_columns_unchanged": len(original.columns), "outcomes_reestimated": False, "new_models": 0, "goal_achieved": False})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("已完成全部区间和原11个月连接；下一步独立复算原文、公式与时钟。")


if __name__ == "__main__":
    {"freeze": freeze, "build": build}[sys.argv[1]]()
