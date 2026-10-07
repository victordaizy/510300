"""连接原有货币观察、订单、价格扩散与实际工业利润；仅解释经营传导。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_orders_cost_profit_bridge_v21"
PARENT = ROOT / "reports/research/510300_real_activity_followthrough_v17"
PRICES = ROOT / "reports/research/510300_manufacturing_price_transmission_source_v1"
CUTOFF = pd.Timestamp("2026-09-11T21:00:00+08:00")
SUPPORTED = "剪刀差改善_信贷与订单共同支持"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(name, frame):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def clean(value):
    return re.sub(r"\s+", "", str(value)).replace("\\u3000", "").replace(",", "").replace("，", "，").replace("－", "-")


def number(value):
    value = clean(value)
    return float(value) if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value) else np.nan


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮方案已冻结，不覆盖。")
    for d in ["inputs", "sources", "results", "figures", "code"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    protocol = {
        "at": now(), "study_id": "510300_ORDERS_COST_PROFIT_BRIDGE_V21",
        "previous_turn_classification": "PROGRESS：V20全体利率背景及205条原路径已完成；整体目标未完成。",
        "question": "原联合支持月的订单扩张与成本传导、实际利润率怎样并存；工业经营事实与510300定价的范围和时间是否一致？",
        "selection": "原104个月、原11个联合支持月全部保留；不按本轮经营数据或股票收益选组。原收益已知，本轮为解释研究。方案前为确认原文结构读过2020年报与2021年1至2月报告的部分数字，不声称未见新结果。",
        "sources": "复用116份当期PMI价格记录与65份工业报告原件；不追补2019及2026工业缺口。只解析原报告当期行，不使用后来表格修订行。",
        "pmi": "起点沿原订单统计月，后续固定第1、2、3个统计月；价格使用同月当期原文。购进与出厂价格是涨跌普遍程度的扩散指数，差值不是成本金额、涨价幅度、毛利率或意外。沿原文行业分类和样本变化。",
        "industrial": "起点和后续三个工业报告严格沿V17已冻结时期与时效状态；1至2月保留为一披露期。不用缺失起点之后的资料反填已知背景。不以不同年份原公布额相除重算同比。",
        "amounts": "逐报告解析全部行业及总计、采矿、制造、公用三大类；所有制组有重叠，保存但不合计。每百元收入成本、费用及利润率按原表记录，工业利润总额不是上市公司归母净利润。",
        "bridge": "总计利润率同比变化=负的每百元收入成本同比变化+负的费用同比变化+其余净项及舍入残差变化。只在三项同比均由正文明确公布时计算。残差不指认为补贴、投资收益或某一原因。当前利润率亦按同分母记录恒等式。",
        "clocks": "沿原公布日日末上界，原104个月观察为21点。后续结果禁止进入起点输入。2026年8月货币公告晚于原截止保持NO_VIEW。原件没有不可修订历史首版认证。",
        "display": "11个月全部列出。重点延续已经固定的2021年1月病例，展示2020年全年、2021年1至2月以及下一三个PMI，不新增择优事件。全体104个月连接、65期完整行业均保存。",
        "limits": "PMI季调月度感受与工业累计同比不是同一期或同企业试验。工业不覆盖银行保险，也非沪深300权重篮子。不能将行业利润总额等权/金额权重替代指数权重，或以经营结果解释全部股票回报。",
        "no_rescue": "既有制造业价格传导模型及账户结果保持冻结，0个新拟合、0个新策略账户、0个新收益分组搜索。",
        "new_models": 0, "new_accounts": 0, "orders_authorized": False,
        "independent_validation": False, "goal_achieved": False,
    }
    save("protocol.json", protocol)
    originals = {
        "monthly.csv": PARENT / "results/104个月_后续经营结果与确认时钟.csv",
        "origin_context.csv": PARENT / "inputs/prior_context.csv",
        "future_records.csv": PARENT / "results/后续经营披露_逐条来源与时间.csv",
        "price_records.json": PRICES / "released_records.json",
        "industrial_records.json": PARENT / "inputs/industrial_profit.json",
    }
    receipts = []
    for name, path in originals.items():
        dest = OUT / "inputs" / name
        shutil.copy2(path, dest)
        receipts.append({"file": name, "source": str(path.relative_to(ROOT)), "sha256": digest(dest)})
    raw_receipts = []
    for family, name in [("pmi", "price_records.json"), ("industrial", "industrial_records.json")]:
        for row in json.loads((OUT / "inputs" / name).read_text("utf-8")):
            path = ROOT / row["raw_path"]
            assert digest(path) == row["raw_sha256"], path
            dest = OUT / "sources" / f"{family}_{row['stat_month']}.html"
            shutil.copy2(path, dest)
            raw_receipts.append({"family": family, "stat_month": row["stat_month"], "source": row["raw_path"], "file": dest.name, "sha256": digest(dest), "url": row["url"], "known_at": row["known_at"]})
    save("source_receipts.json", raw_receipts)
    save("freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "inputs": receipts, "original_source_files": len(raw_receipts)})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("第二十一轮范围已冻结：104个月、116份价格记录、65份工业原报告，经营解释与原模型分开。")


def unique_tables(soup, label):
    matches = []
    for table in soup.find_all("table"):
        rows = [[clean(c.get_text("", strip=True)) for c in tr.find_all(["td", "th"])] for tr in table.find_all("tr")]
        if not rows:
            continue
        h = rows[0]
        eligible = ((label == "financial" and h == ["分组", "营业收入", "营业成本", "利润总额"])
                    or (label == "efficiency" and "营业收入利润率" in h and h[0] == "分组")
                    or (label == "industries" and h == ["行业", "营业收入", "营业成本", "利润总额"]))
        if eligible and rows not in matches:
            matches.append(rows)
    if len(matches) != 1:
        raise ValueError(f"原表不唯一或缺失：{label}，数量{len(matches)}")
    return matches[0]


def published_change(text, field, unit):
    # 只从正文对应完整句提取，不从上年另一个版本的金额推导。
    label = {"margin": "营业收入利润率", "cost": "每百元营业收入中的成本", "fee": "每百元营业收入中的费用"}[field]
    pat = re.escape(label) + r"(?:为)?([\d.]+)(?:%|元)[，,](?:比上年同期|比上年|同比)(提高|上升|增加|下降|减少|降低)([\d.]+)" + unit
    m = re.search(pat, text)
    if m:
        return float(m.group(1)), float(m.group(3)) * (1 if m.group(2) in ["提高", "上升", "增加"] else -1), m.group(0)
    steady = re.search(re.escape(label) + r"(?:为)?([\d.]+)(?:%|元)[，,](?:与上年同期|与上年|同比)持平", text)
    if steady:
        return float(steady.group(1)), 0., steady.group(0)
    return np.nan, np.nan, "正文同比未提取，保留未知"


def parse_industrial(record):
    path = OUT / "sources" / f"industrial_{record['stat_month']}.html"
    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    tables = {kind: unique_tables(soup, kind) for kind in ["financial", "efficiency", "industries"]}
    eff = {r[0].replace("其中：", ""): r for r in tables["efficiency"] if len(r) == 9 and np.isfinite(number(r[1]))}
    metadata = {k: record[k] for k in ["stat_month", "known_at", "published_at", "url", "raw_sha256"]}
    metadata.update(source_file=path.name, historical_first_vintage_verified=False, period_type="年初至统计月累计；一月免报")
    frames = []
    for kind in ["financial", "industries"]:
        for raw in tables[kind]:
            if len(raw) != 7 or not np.isfinite(number(raw[1])):
                continue
            label = raw[0].replace("其中：", "")
            row = {**metadata, "table": kind, "group": label, "revenue": number(raw[1]), "revenue_yoy": number(raw[2]),
                   "cost": number(raw[3]), "cost_yoy": number(raw[4]), "profit": number(raw[5]), "profit_yoy": number(raw[6]),
                   "profit_yoy_raw": raw[6], "unit": "亿元", "amounts_same_report": True}
            row.update(margin_calculated=100*row["profit"]/row["revenue"], cost_per100_calculated=100*row["cost"]/row["revenue"])
            if kind == "financial":
                er = eff[label]
                row.update(margin=number(er[1]), cost_per100=number(er[2]), fee_per100=number(er[3]),
                           collection_days=number(er[8]), inventory_days=number(er[7]))
                row["other_net_per100"] = row["margin"] - (100 - row["cost_per100"] - row["fee_per100"])
            frames.append(row)
    allrows = pd.DataFrame(frames)
    assert len(allrows[allrows.table.eq("industries")]) == 42, (record["stat_month"], len(allrows))
    assert len(allrows[allrows.table.eq("financial")]) == 8
    text = clean(soup.get_text("", strip=True)).split("表1")[0]
    total = allrows[(allrows.table == "financial") & (allrows.group == "总计")].iloc[0].to_dict()
    for field, unit in [("margin", "个百分点"), ("cost", "元"), ("fee", "元")]:
        val, change, statement = published_change(text, field, unit)
        key = "margin" if field == "margin" else field + "_per100"
        if np.isfinite(val):
            assert abs(val - total[key]) < 1e-9, (record["stat_month"], key, val, total[key])
        total[field + "_yoy_change"] = change
        total[field + "_yoy_statement"] = statement
    vals = [total[f"{x}_yoy_change"] for x in ["margin", "cost", "fee"]]
    total["bridge_status"] = "COMPLETE_PUBLISHED_CHANGES" if np.isfinite(vals).all() else "MISSING_PUBLISHED_CHANGE"
    total["cost_contribution_pp"] = -vals[1]
    total["fee_contribution_pp"] = -vals[2]
    total["other_net_change_pp"] = vals[0] + vals[1] + vals[2]
    return allrows, total


def build():
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("本轮结果已经完成，不覆盖。")
    records = json.loads((OUT / "inputs/industrial_records.json").read_text("utf-8"))
    pairs = [parse_industrial(x) for x in records]
    industries = pd.concat([x[0] for x in pairs], ignore_index=True)
    totals = pd.DataFrame([x[1] for x in pairs])
    csv("65期_全部原表行业及经营金额.csv", industries)
    csv("65期_成本费用与利润率同比分解.csv", totals)
    monthly = pd.read_csv(OUT / "inputs/monthly.csv")
    origin = pd.read_csv(OUT / "inputs/origin_context.csv").set_index("stat_month")
    prices = pd.DataFrame(json.loads((OUT / "inputs/price_records.json").read_text("utf-8"))).set_index("stat_month")
    pkeys = ["input_price_diffusion", "output_price_diffusion", "output_minus_input_diffusion_pp", "method_group", "known_at", "url", "raw_sha256"]
    ikeys = ["known_at", "revenue_yoy", "cost_yoy", "profit_yoy", "margin", "cost_per100", "fee_per100", "collection_days", "margin_yoy_change", "cost_yoy_change", "fee_yoy_change", "other_net_change_pp", "url"]
    tm = totals.set_index("stat_month")
    additions, linked = [], []
    for base in monthly.itertuples(index=False):
        row = {"stat_month": base.stat_month}
        t0 = pd.Timestamp(base.snapshot_at)
        active = t0 <= CUTOFF and pd.notna(base.orders_period)
        for h in range(4):
            period = base.orders_period if h == 0 else getattr(base, f"orders_next{h}_period")
            p = {"stat_month": base.stat_month, "family": "PMI价格", "horizon": h, "target_period": period,
                 "origin_snapshot_at": base.snapshot_at, "status": "NO_VIEW_OR_SOURCE_NOT_COVERED", "role": "起点背景" if h == 0 else "后续披露，禁止回填起点"}
            if active and pd.notna(period) and period in prices.index:
                source = prices.loc[period]
                known = pd.Timestamp(source.known_at)
                valid = known <= t0 if h == 0 else t0 < known <= CUTOFF
                if valid:
                    p.update({key: source[key] for key in pkeys})
                    p["orders"] = base.orders_first_release_value if h == 0 else getattr(base, f"orders_next{h}_value")
                    p["status"] = "KNOWN_AT_ORIGIN" if h == 0 else "OBSERVED_AFTER_ORIGIN"
                    for key in pkeys + ["orders"]:
                        row[f"pmi_{h}_{key}"] = p[key]
            row[f"pmi_{h}_period"] = period
            row[f"pmi_{h}_status"] = p["status"]
            linked.append(p)
        original = origin.loc[base.stat_month]
        for h in range(4):
            if h == 0:
                period = original.get("profit_period")
                baseline_ok = getattr(base, "industrial_baseline_status") == "AVAILABLE_RECONSTRUCTED"
            else:
                period = getattr(base, f"industrial_next{h}_period", np.nan)
                baseline_ok = getattr(base, f"industrial_next{h}_status", "") == "OBSERVED_BY_CUTOFF"
            p = {"stat_month": base.stat_month, "family": "工业经营", "horizon": h, "target_period": period,
                 "origin_snapshot_at": base.snapshot_at, "status": "NO_VIEW_OR_SOURCE_NOT_COVERED", "role": "起点背景" if h == 0 else "后续披露，禁止回填起点"}
            if active and baseline_ok and pd.notna(period) and period in tm.index:
                source = tm.loc[period]
                known = pd.Timestamp(source.known_at)
                assert known <= t0 if h == 0 else t0 < known <= CUTOFF
                p.update({key: source[key] for key in ikeys})
                p["status"] = "KNOWN_AT_ORIGIN" if h == 0 else "OBSERVED_AFTER_ORIGIN"
                for key in ikeys:
                    row[f"industrial_{h}_{key}"] = p[key]
            row[f"industrial_{h}_period"] = period
            row[f"industrial_{h}_status"] = p["status"]
            linked.append(p)
        additions.append(row)
    result = monthly.merge(pd.DataFrame(additions), on="stat_month", validate="one_to_one")
    csv("104个月_订单价格成本与原后续路径.csv", result)
    csv("832条_经营信息角色与可见性.csv", pd.DataFrame(linked))
    csv("原11个月_完整联合支持背景.csv", result[result.joint_credit_orders_state.eq(SUPPORTED)])
    case = pd.DataFrame(linked)
    csv("2021年1月病例_全部起点与后续经营信息.csv", case[case.stat_month.eq("2021-01")])
    save("results/build_receipt.json", {"at": now(), "industrial_reports": len(totals), "full_table_rows": len(industries),
        "origin_months": len(result), "linked_rows": len(linked), "complete_margin_bridges": int(totals.bridge_status.eq("COMPLETE_PUBLISHED_CHANGES").sum()),
        "industrial_origin_statuses": result.industrial_0_status.value_counts().to_dict(),
        "pmi_origin_statuses": result.pmi_0_status.value_counts().to_dict(), "goal_achieved": False})
    print(json.dumps(json.loads((OUT / "results/build_receipt.json").read_text("utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    {"freeze": freeze, "build": build}[sys.argv[1]]()
