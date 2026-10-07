"""独立核对原文金额、区间覆盖、舍入范围、可见时钟与原观察保留。"""
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import shutil
import unicodedata

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_credit_recency_structure_v23"


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(text))).replace(",", "")


def main():
    if (OUT / "verification.json").exists():
        raise RuntimeError("已有本轮复算回执，不覆盖。")
    assert (OUT / "build_receipt.json").exists()
    atoms = pd.read_csv(OUT / "results/963项_原文区间金额与显示精度.csv").set_index("key")
    panel = pd.read_csv(OUT / "results/2808项_当月三月累计净增与公式.csv")
    comp = pd.read_csv(OUT / "results/2808项_同区间跨公告同比比较.csv")
    orig = pd.read_csv(OUT / "inputs/monthly.csv")
    wide = pd.read_csv(OUT / "results/104个月_信用近期性与完整经营定价背景.csv")
    credit = pd.read_csv(OUT / "inputs/credit.csv").set_index("stat_month")
    monthly = orig.set_index("stat_month")
    sources = json.loads((OUT / "source_receipts.json").read_text("utf-8"))
    frozen = json.loads((OUT / "freeze.json").read_text("utf-8"))
    assert frozen["protocol_sha256"] == hashlib.sha256((OUT / "protocol.json").read_bytes()).hexdigest()
    for r in json.loads((OUT / "input_receipts.json").read_text("utf-8")):
        assert r["sha256"] == hashlib.sha256((OUT / "inputs" / r["name"]).read_bytes()).hexdigest()
    pages = {}
    for r in sources:
        path = OUT / "sources" / (r["month"] + ".html")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r["source_sha256"]
        pages[r["month"]] = normalize(BeautifulSoup(path.read_text("utf-8"), "html.parser").get_text())
    raw_count = 0
    for key, a in atoms.iterrows():
        if pd.isna(a.value):
            assert a.literal == "原文未单列"
            continue
        literal = normalize(a.literal)
        assert literal in pages[a.source_month], (key, "原文证据未匹配", literal)
        matches = re.findall(r"(增加|减少)([\d.]+)(万亿元|亿元)", literal)
        assert len(matches) == 1, (key, matches)
        verb, number, unit = matches[0]
        scale = Decimal(10000) if unit == "万亿元" else Decimal(1)
        value = Decimal(number) * scale * (-1 if verb == "减少" else 1)
        digits = len(number.split(".")[1]) if "." in number else 0
        half = scale * Decimal(10) ** (-digits) / 2
        assert abs(float(value) - a.value) < 1e-8 and abs(float(half) - a.rounding_half) < 1e-8
        raw_count += 1

    def evaluate(expression):
        if not isinstance(expression, str):
            return np.nan, np.nan, {}, None
        expr = json.loads(expression)
        if expr is None:
            return np.nan, np.nan, {}, None
        total = Decimal(0)
        bound = Decimal(0)
        coverage = defaultdict(int)
        clocks = []
        missing = False
        for key, coefficient in expr.items():
            a = atoms.loc[key]
            assert coefficient and int(coefficient) == coefficient
            for month in pd.period_range(a.start, a.end, freq="M"):
                coverage[str(month)] += coefficient
            clocks.append(pd.Timestamp(a.known_at))
            if pd.isna(a.value):
                missing = True
            else:
                total += Decimal(str(a.value)) * coefficient
                bound += Decimal(str(a.rounding_half)) * abs(coefficient)
        return (np.nan if missing else float(total), np.nan if missing else float(bound),
                {m: n for m, n in coverage.items() if n}, max(clocks) if clocks else None)

    errors, admitted = [], 0
    for r in panel.itertuples():
        val, bound, coverage, known = evaluate(r.expression)
        if isinstance(r.expression, str) and r.expression != "null":
            expected = {str(m): 1 for m in pd.period_range(r.start, r.end, freq="M")}
            assert coverage == expected, (r.stat_month, r.field, r.window, coverage, expected)
        if r.status.startswith("AVAILABLE"):
            errors.extend([abs(r.value_yi - val), abs(r.rounding_half_yi - bound)])
            assert known <= pd.Timestamp(monthly.loc[r.stat_month, "available_at_upper_bound"])
            admitted += 1
        else:
            assert pd.isna(r.value_yi) and pd.isna(r.rounding_half_yi)
        if r.window == "YTD":
            old = credit.loc[r.stat_month, r.field + "_ytd_yi"]
            assert pd.isna(old) and pd.isna(r.value_yi) or abs(old - r.value_yi) < 1e-8
    for r in comp.itertuples():
        val, bound, coverage, known = evaluate(r.expression)
        if r.status.startswith("AVAILABLE"):
            current = pd.period_range(r.current_start, r.stat_month, freq="M")
            past = pd.period_range(r.prior_start, pd.Period(r.stat_month, "M") - 12, freq="M")
            expected = {str(m): 1 for m in current} | {str(m): -1 for m in past}
            assert coverage == expected
            scopes = {atoms.loc[k, "regime"] for k in json.loads(r.expression)}
            assert len(scopes) == 1
            errors.extend([abs(r.yoy_change_yi - val), abs(r.rounding_bound_yi - bound)])
            direct = "多增" if val > bound else "少增" if val < -bound else "显示舍入界内"
            assert r.direction == direct
            assert known <= pd.Timestamp(monthly.loc[r.stat_month, "available_at_upper_bound"])
            admitted += 1
        else:
            assert r.direction == "NO_VIEW" and pd.isna(r.yoy_change_yi)
    for name, value_col, bound_col in [
        ("312组_企事业未单列项与显示舍入范围.csv", "corporate_unlisted_or_rounding_yi", "display_rounding_bound_yi"),
        ("27项_同文单月与累计差分并列.csv", "difference_yi", "rounding_bound_yi"),
    ]:
        frame = pd.read_csv(OUT / "results" / name)
        for _, r in frame.iterrows():
            value, bound, coverage, _ = evaluate(r.expression)
            if r.status.startswith("AVAILABLE"):
                errors.extend([abs(r[value_col] - value), abs(r[bound_col] - bound)])
                # 净差的期数覆盖相消；分项资金并非相同对象，不据此声称余额为零。
                if name.startswith("27"):
                    assert not coverage
    assert max(errors) < 1e-8
    pd.testing.assert_frame_equal(orig, wide[orig.columns], check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
    fixed = pd.read_csv(OUT / "results/原11个共同支持月份_全部背景与原路径.csv")
    expected_members = orig.loc[orig.joint_credit_orders_state.eq("剪刀差改善_信贷与订单共同支持"), "stat_month"].tolist()
    assert fixed.stat_month.tolist() == expected_members and len(fixed) == 11
    clock_count = 0
    for r in wide.itertuples():
        if r.credit_origin_admission != "原观察可用":
            assert r.stat_month == "2026-08" and pd.isna(r.E0_20_return)
            continue
        snap = pd.Timestamp(r.snapshot_at)
        for field in ["industrial_0_known_at", "context_fixing_known_at", "context_bond_known_at"]:
            value = getattr(r, field)
            if pd.notna(value):
                assert pd.Timestamp(value) <= snap, (r.stat_month, field, value, snap)
                clock_count += 1
    result = {"at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "status": "PASS_RAW_AMOUNTS_INTERVAL_COVERAGE_ROUNDING_AND_ORIGINAL_CONTEXT",
              "source_pages": len(sources), "raw_nonmissing_atoms_reparsed": raw_count,
              "interval_rows_checked": len(panel), "comparison_rows_checked": len(comp),
              "available_interval_and_comparison_clocks": admitted, "extra_context_clocks_checked": clock_count,
              "original_columns_unchanged": len(orig.columns), "fixed_original_cases": len(fixed),
              "maximum_amount_error_yi": max(errors), "historical_first_vintage_verified": False,
              "independent_validation": False, "goal_achieved": False}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("原文金额、区间覆盖、显示舍入、当时可见时钟与原389列均已复算通过。")


if __name__ == "__main__":
    main()
