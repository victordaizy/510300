"""从原文数值和保存结果复核金额、累计口径、时钟及余额恒等式。"""
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_holiday_money_credit_bridge_v11"
FIELDS = ["loan_total", "loan_household", "loan_household_short", "loan_household_long",
          "loan_corporate", "loan_corporate_short", "loan_corporate_long", "loan_bills",
          "loan_nonbank", "deposit_total", "deposit_household", "deposit_corporate", "deposit_fiscal", "deposit_nonbank"]
# 对照原文逐项记录；不调用生成程序的正则或金额函数。
EXPECTED = {
    ("2020-01", "MONTH_REPORTED"): [33400, 6341, -1149, 7491, 28600, 7699, 16600, 3596, -1567, 28800, 42400, -16100, 4002, 5701],
    ("2020-02", "YEAR_TO_DATE_REPORTED"): [42400, 2209, -5653, 7862, 39900, 14200, 20800, 4231, 219, 39000, 41200, -13300, 4210, 10600],
    ("2020-02", "MONTH_REPORTED"): [9057, -4133, -4504, 371, 11300, 6549, 4157, 634, 1786, 10200, -1200, 2840, 208, 4924],
    ("2021-01", "MONTH_REPORTED"): [35800, 12700, 3278, 9448, 25500, 5755, 20400, -1405, -1992, 35700, 14800, 9484, 11700, -1120],
    ("2021-02", "MONTH_REPORTED"): [13600, 1421, -2691, 4113, 12000, 2497, 11000, -1855, 180, 11500, 32600, -24200, -8479, 16100],
}


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-8), (a, b)


def check_literal(row, key, raw):
    literal = row[key + "_literal"]
    assert literal in raw
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)(万亿元|亿元)$", literal)
    assert match, literal
    number = Decimal(match[1])
    unit = Decimal(10000 if match[2] == "万亿元" else 1)
    close(row[key + "_value_yi"], number * unit * (-1 if "减少" in literal else 1))
    close(row[key + "_rounding_half_yi"], Decimal("0.5") * Decimal(10) ** number.as_tuple().exponent * unit)


def verify():
    freeze = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha256((OUT / "protocol.json").read_bytes()).hexdigest() == freeze["protocol_sha256"]
    assert sha256((OUT / "inputs/monthly_context.csv").read_bytes()).hexdigest() == freeze["monthly_input_sha256"]
    raw = {}
    for source in freeze["sources"]:
        path = OUT / "sources" / source["name"]
        assert sha256(path.read_bytes()).hexdigest() == source["sha256"]
        if "month" in source:
            raw[source["month"]] = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser").get_text(" ", strip=True)
    money = pd.read_csv(OUT / "results/货币余额与发布时点.csv").set_index("month")
    flow = pd.read_csv(OUT / "results/原文贷款存款分项.csv").set_index(["month", "interval"])
    assert len(money) == 7 and len(flow) == 5
    for month, row in money.iterrows():
        for key in ["m0", "m1", "m2"]:
            check_literal(row, key, raw[month])
        close(row.noncash_m1_yi, row.m1_value_yi - row.m0_value_yi)
        close(row.spread_pp, row.m1_yoy_pp - row.m2_yoy_pp)
        assert row.available_at_20210209_21 == (pd.Timestamp(row.published_at) <= pd.Timestamp("2021-02-09T21:00:00+08:00"))
    for index, expected in EXPECTED.items():
        row = flow.loc[index]
        for key, value in zip(FIELDS, expected):
            close(row[key + "_value_yi"], value)
            check_literal(row, key, raw[index[0]])
    comparisons = pd.read_csv(OUT / "results/一月与两月累计_同区间结构对照.csv")
    assert len(comparisons) == 28
    for row in comparisons.to_dict("records"):
        idx = FIELDS.index(row["field"])
        jan = row["interval"] == "JAN"
        a = EXPECTED[("2020-01", "MONTH_REPORTED") if jan else ("2020-02", "YEAR_TO_DATE_REPORTED")][idx]
        b = EXPECTED[("2021-01", "MONTH_REPORTED")][idx]
        if not jan:
            b += EXPECTED[("2021-02", "MONTH_REPORTED")][idx]
        close(row["2020_value_yi"], a)
        close(row["2021_value_yi"], b)
        close(row["cross_release_difference_yi"], b - a)
        assert row["available_at_20210209_21"] == jan
    changes = pd.read_csv(OUT / "results/余额变动与剪刀差变动.csv")
    for row in changes.to_dict("records"):
        a, b = money.loc[row["start"]], money.loc[row["end"]]
        for key in ["m0", "m1", "m2"]:
            close(row[key + "_change_yi"], b[key + "_value_yi"] - a[key + "_value_yi"])
        close(row["noncash_m1_change_yi"], b.noncash_m1_yi - a.noncash_m1_yi)
        close(row["spread_change_pp"], b.spread_pp - a.spread_pp)
    logs = pd.read_csv(OUT / "results/一月与两月_当期基数对数分解.csv")
    for row in logs.to_dict("records"):
        end, prev = row["end"], row["end"].replace("2021", "2020")
        ratio = lambda m: money.loc[m].m1_value_yi / money.loc[m].m2_value_yi
        current = 100 * math.log(ratio(end) / ratio("2020-12"))
        base = -100 * math.log(ratio(prev) / ratio("2019-12"))
        close(row["current_relative_log_pp"], current)
        close(row["base_relative_log_pp"], base)
        close(row["total_relative_growth_log_pp"], current + base)
    residual = pd.read_csv(OUT / "results/官方同比与跨公告差_保留版本残差.csv").set_index("field")
    for field, official, difference in [("loan_total", 2252, 148), ("deposit_total", 6245, 655)]:
        close(residual.loc[field].official_reported_yoy_yi, official)
        close(residual.loc[field].difference_from_official_yoy_yi, difference)
        assert f"同比多增{official}亿元" in raw["2021-01"]
    assert "M1同比增长约2%" in "".join(raw["2022-01"].split())
    assert "单位活期存款会向个人存款转移" in raw["2022-01"]
    calendars = json.loads((OUT / "calendar_receipts.json").read_text(encoding="utf-8"))
    for rec in calendars:
        assert sha256((OUT / "sources" / (rec["name"] + ".html")).read_bytes()).hexdigest() == rec["sha256"]
        assert pd.Timestamp(rec["available_at"]) < pd.Timestamp("2021-02-09T21:00:00+08:00")
    demo = pd.read_csv(OUT / "results/100元划转_新旧口径示意.csv")
    np.testing.assert_allclose(demo[["old_m1_change_cny", "new_m1_change_cny", "m2_change_cny"]],
                               [[-100, 0, 0], [-100, -100, 0], [0, 0, 0], [100, 100, 100]])
    assert not demo.available_at_20210209_21.any()
    result = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_MONEY_CREDIT_RECOMPUTATION",
        "money_amounts": 21, "credit_deposit_amounts": 70, "same_interval_comparisons": 28,
        "balance_windows": 6, "log_identities": 2, "definition_examples": 4,
        "historical_first_version_authenticated": False, "economic_causality_identified": False,
        "independent_prediction_validation": False}
    (OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    verify()
