"""从固定原公告连接银行收入、减值、资本工具分配和每股收益。"""
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_bank_profit_credit_bridge_v13"
FACTS = []
PAGES = {}
RECEIPTS = {}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def add(company, document, page, key, label, current, previous, unit="人民币百万元", period="2020年/2019年", literal=None):
    text = PAGES[document][page - 1]["text"]
    compact = re.sub(r"\s+", "", text).replace(",", "")
    assert re.sub(r"\s+", "", label) in compact, (document, page, label)
    if literal:
        assert re.sub(r"\s+", "", literal).replace(",", "") in compact
    else:
        # 数值核对保留单元格间空白，避免把相邻年度的金额拼成一个数字。
        numeric_tokens = {Decimal(token) for token in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", text.replace(",", ""))}
        for value in [current, previous]:
            assert Decimal(str(value)) in numeric_tokens, (document, page, key, value)
    receipt = RECEIPTS[document + ".pdf"]
    assert pd.Timestamp(receipt["known_at"]) < pd.Timestamp("2021-02-09T21:00:00+08:00")
    FACTS.append({"company": company, "key": key, "current": current, "previous": previous, "unit": unit, "period": period,
        "document": document + ".pdf", "pdf_page": page, "label": label, "known_at": receipt["known_at"], "source_url": receipt["url"], "source_sha256": receipt["sha256"]})


def build():
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("本轮原表计算已保存，不覆盖。")
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert sha256((OUT / "protocol.json").read_bytes()).hexdigest() == frozen["protocol_sha256"]
    receipts = json.loads((OUT / "source_receipts.json").read_text(encoding="utf-8"))
    for r in receipts:
        assert sha256((OUT / "sources" / r["name"]).read_bytes()).hexdigest() == r["sha256"]
        if r["name"].endswith(".pdf"):
            RECEIPTS[r["name"]] = r
            PAGES[r["name"][:-4]] = json.loads((OUT / "sources" / (r["name"][:-4] + ".pages.json")).read_text(encoding="utf-8"))
    cmb = "招商银行_2020业绩快报"
    for key, label, a, b in [
        ("revenue", "营业收入", 290508, 269703), ("noninterest", "非利息净收入", 105493, 96613),
        ("operating_profit", "营业利润", 122680, 117047), ("pretax", "利润总额", 122478, 117132),
        ("parent_profit", "归属于本行股东的净利润", 97342, 92867),
        ("loan_end", "贷款和垫款总额", 5029128, 4490650), ("deposit_end", "客户存款总额", 5625793, 4844422)]:
        add("招商银行", cmb, 1, key, label, a, b)
    add("招商银行", cmb, 1, "eps", "每股收益", 3.79, 3.62, "人民币元")
    add("招商银行", "招商银行_2020三季报", 2, "parent_profit_9m", "归属于本行股东的净利润", 76603, 77239, period="2020年1—9月/2019年1—9月")
    cib = "兴业银行_2020业绩快报"
    for key, label, a, b in [
        ("revenue", "营业收入", 203137, 181308), ("preprovision", "拨备前利润", 151974, 132362),
        ("operating_profit", "营业利润", 76547, 74266), ("pretax", "利润总额", 76637, 74503),
        ("parent_profit", "归属于母公司股东的净利润", 66626, 65868)]:
        add("兴业银行", cib, 1, key, label, a, b)
    add("兴业银行", cib, 1, "eps", "基本每股收益", 3.08, 3.10, "人民币元")
    add("兴业银行", "兴业银行_2020三季报", 2, "parent_profit_9m", "归属于母公司股东的净利润", 51875, 54910, period="2020年1—9月/2019年1—9月")
    pab = "平安银行_2020年报"
    for key, label, a, b in [
        ("interest_income", "利息收入", 187187, 177549), ("interest_expense", "利息支出", 87537, 87588),
        ("net_interest", "利息净收入", 99650, 89961), ("net_fee", "手续费及佣金净收入", 43481, 36743),
        ("revenue", "营业收入合计", 153542, 137958), ("tax_surcharge", "税金及附加", 1525, 1290),
        ("opex", "业务及管理费", 44690, 40852), ("preprovision", "减值损失前营业利润", 107327, 95816),
        ("credit_impairment", "信用减值损失", 69611, 58471), ("other_impairment", "其他资产减值损失", 807, 1056),
        ("operating_profit", "营业利润", 36909, 36289), ("nonop_income", "营业外收入", 77, 99),
        ("nonop_expense", "营业外支出", 232, 148), ("pretax", "利润总额", 36754, 36240),
        ("income_tax", "所得税费用", 7826, 8045), ("parent_profit", "净利润", 28928, 28195)]:
        add("平安银行", pab, 148, key, label, a, b)
    for key, label, a, b in [
        ("earning_assets_avg", "生息资产总计", 3944430, 3433756), ("interest_liabilities_avg", "计息负债总计", 3776287, 3320408),
        ("loan_avg", "发放贷款和垫款", 2371043, 2025073), ("deposit_avg", "吸收存款", 2517798, 2274753)]:
        add("平安银行", pab, 30, key, label, a, b)
    add("平安银行", pab, 30, "nim", "净息差", 2.53, 2.62, "百分比")
    add("平安银行", pab, 31, "nii_quarters", "利息净收入", 24496, 24849, period="2020年第四季度/第三季度")
    add("平安银行", pab, 31, "nim_quarters", "净息差", 2.44, 2.48, "百分比", "2020年第四季度/第三季度")
    add("平安银行", pab, 34, "loan_impairment", "发放贷款和垫款", 43148, 53288)
    add("平安银行", pab, 34, "total_impairment", "合计", 70418, 59527)
    add("平安银行", pab, 240, "preferred_dividend", "母公司优先股宣告股息", 874, 874)
    add("平安银行", pab, 240, "perpetual_interest", "母公司永续债利息", 820, 0, literal="母公司永续债利息 (820) -")
    add("平安银行", pab, 240, "ordinary_profit", "归属于母公司普通股股东的本年净利润", 27234, 27321)
    add("平安银行", pab, 240, "weighted_shares", "已发行在外普通股的加权平均数", 19406, 17764, "百万股")
    add("平安银行", pab, 240, "eps", "基本每股收益", 1.40, 1.54, "人民币元")
    facts = pd.DataFrame(FACTS)
    facts.to_csv(OUT / "results/原表金额与期间来源.csv", index=False, encoding="utf-8-sig")
    def values(company, key):
        row = facts[facts.company.eq(company) & facts.key.eq(key)].iloc[0]
        return np.array([row.current, row.previous], dtype=float)
    p = lambda key: values("平安银行", key)
    delta = lambda arr: float(arr[0] - arr[1])
    np.testing.assert_allclose(p("interest_income") - p("interest_expense"), p("net_interest"))
    np.testing.assert_allclose(p("revenue") - p("tax_surcharge") - p("opex"), p("preprovision"))
    np.testing.assert_allclose(p("preprovision") - p("credit_impairment") - p("other_impairment"), p("operating_profit"))
    np.testing.assert_allclose(p("operating_profit") + p("nonop_income") - p("nonop_expense"), p("pretax"))
    np.testing.assert_allclose(p("pretax") - p("income_tax"), p("parent_profit"))
    np.testing.assert_allclose(p("parent_profit") - p("preferred_dividend") - p("perpetual_interest"), p("ordinary_profit"))
    np.testing.assert_allclose(np.round(p("ordinary_profit") / p("weighted_shares"), 2), p("eps"))
    bridge = []
    for label, amount in [("净利息收入增加", delta(p("net_interest"))), ("净手续费收入增加", delta(p("net_fee"))),
        ("其他收入变化", delta(p("revenue") - p("net_interest") - p("net_fee"))),
        ("业务费用与税金增加", -delta(p("opex") + p("tax_surcharge"))), ("信用及其他减值增加", -delta(p("total_impairment"))),
        ("营业外净额变化", delta(p("nonop_income") - p("nonop_expense"))), ("所得税费用减少", -delta(p("income_tax")))]:
        bridge.append({"item": label, "profit_change_yi": amount / 100})
    np.testing.assert_allclose(sum(r["profit_change_yi"] for r in bridge), delta(p("parent_profit")) / 100)
    pd.DataFrame(bridge).to_csv(OUT / "results/平安银行_净利润增量金额桥.csv", index=False, encoding="utf-8-sig")
    a, l, income, expense = p("earning_assets_avg"), p("interest_liabilities_avg"), p("interest_income"), p("interest_expense")
    y, c = income / a, expense / l
    nii_bridge = [{"item": "生息资产规模项", "change_yi": delta(a) * y.mean() / 100}, {"item": "资产综合收益率项", "change_yi": a.mean() * delta(y) / 100},
        {"item": "计息负债规模项", "change_yi": -delta(l) * c.mean() / 100}, {"item": "负债综合成本率项", "change_yi": -l.mean() * delta(c) / 100}]
    np.testing.assert_allclose(sum(r["change_yi"] for r in nii_bridge), delta(p("net_interest")) / 100)
    pd.DataFrame(nii_bridge).to_csv(OUT / "results/平安银行_净利息收入规模与综合比率分解.csv", index=False, encoding="utf-8-sig")
    summaries, q4 = [], []
    for name in ["招商银行", "兴业银行", "平安银行"]:
        revenue, profit, eps = [values(name, k) for k in ["revenue", "parent_profit", "eps"]]
        summaries.append({"company": name, "revenue_current_yi": revenue[0] / 100, "revenue_growth_percent": 100 * (revenue[0] / revenue[1] - 1), "profit_current_yi": profit[0] / 100,
            "profit_growth_percent": 100 * (profit[0] / profit[1] - 1), "eps_current": eps[0], "eps_previous": eps[1],
            "annual_nim_status": "DISCLOSED" if name == "平安银行" else "NO_VIEW_IN_PRELIMINARY_RELEASE",
            "annual_detailed_impairment_status": "DISCLOSED" if name == "平安银行" else "NO_VIEW_IN_PRELIMINARY_RELEASE"})
        if name != "平安银行":
            quarter = profit - values(name, "parent_profit_9m")
            q4.append({"company": name, "implied_2020_q4_yi": quarter[0] / 100, "implied_2019_q4_yi": quarter[1] / 100,
                "growth_percent": 100 * (quarter[0] / quarter[1] - 1), "status": "PRELIMINARY_YEAR_MINUS_REPORTED_NINE_MONTHS_NO_LATER_BACKFILL"})
    pd.DataFrame(summaries).to_csv(OUT / "results/三家银行_当时已知年度业绩.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(q4).to_csv(OUT / "results/快报减三季报_隐含第四季度.csv", index=False, encoding="utf-8-sig")
    cib_gap = values("兴业银行", "preprovision") - values("兴业银行", "pretax")
    cmb_nii = values("招商银行", "revenue") - values("招商银行", "noninterest")
    detail = {"pab": {"nii_growth_percent": 100 * (p("net_interest")[0] / p("net_interest")[1] - 1), "net_profit_change_yi": delta(p("parent_profit")) / 100,
        "loan_impairment_change_yi": delta(p("loan_impairment")) / 100,
        "nonloan_impairment_change_yi": delta(p("total_impairment") - p("loan_impairment")) / 100,
        "ordinary_profit_change_yi": delta(p("ordinary_profit")) / 100,
        "ordinary_profit_growth_percent": 100 * (p("ordinary_profit")[0] / p("ordinary_profit")[1] - 1),
        "share_growth_percent": 100 * (p("weighted_shares")[0] / p("weighted_shares")[1] - 1),
        "eps_unrounded_current": float(p("ordinary_profit")[0] / p("weighted_shares")[0]), "eps_unrounded_previous": float(p("ordinary_profit")[1] / p("weighted_shares")[1])},
        "cib": {"preprovision_growth_yi": delta(values("兴业银行", "preprovision")) / 100, "preprovision_to_pretax_gap_growth_yi": delta(cib_gap) / 100,
                "gap_meaning": "含拨备及其他差额，快报未充分分项，不冒称全是信用减值。"},
        "cmb": {"nii_current_yi": cmb_nii[0] / 100, "nii_previous_yi": cmb_nii[1] / 100, "nii_growth_percent": 100 * (cmb_nii[0] / cmb_nii[1] - 1)},
        "decomposition_meaning": "综合收益率/成本率由利息金额除以日均余额构造，包含结构变化，不等于纯利率定价因果；对称分解只保证金额恒等式。"}
    save("results/传导分解摘要.json", detail)
    verification = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_PRIMARY_STATEMENT_AND_PROFIT_IDENTITIES", "source_documents": 5,
        "paired_facts": len(facts), "source_dates_before_snapshot": True, "primary_visual_pages_checked": {"平安银行": [30, 31, 34, 148, 240], "招商银行快报": [1], "兴业银行快报": [1]},
        "future_annual_reports_backfilled": 0, "new_models": 0, "new_accounts": 0, "independent_validation": False}
    save("verification.json", verification)
    save("results/build_receipt.json", {**verification, "goal_achieved": False, "continuation_classification": "PROGRESS"})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps({"复算": verification, "传导": detail}, ensure_ascii=False))


if __name__ == "__main__":
    build()
