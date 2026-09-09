"""从已保存的金融年报提取合并利润及普通股每股收益组成，不读取价格或策略收益。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_financial_ttm_dependencies_v1"
OUT = ROOT / "reports/research/510300_financial_annual_components_v1"
MANIFEST = ROOT / "config/510300_financial_annual_components_v1_manifest.json"
BROKERS = {"000776.SZ", "600030.SH", "600999.SH", "601211.SH"}
CORE = ["OPERATING_REVENUE_YTD", "OPERATING_PROFIT_YTD", "PARENT_NET_PROFIT_YTD", "BASIC_EPS_YTD"]
TRAD = str.maketrans(dict(zip("營業總收入支淨利潤歸屬於東數幣萬元團併註減當權益發期終續釋銀額計時過程與別標", "营业总收入支净利润归属于东数币万元团并注减当权益发期终续释银额计时过程与别标")))
LABELS = {
    "OPERATING_REVENUE_YTD": ["营业收入合计", "营业总收入", "营业收入"],
    "OPERATING_EXPENSE_YTD": ["营业支出合计", "营业总支出", "营业支出"],
    "OPERATING_PROFIT_YTD": ["营业利润"],
    "TOTAL_NET_PROFIT_YTD": ["净利润"],
    "PARENT_NET_PROFIT_YTD": ["归属于母公司股东的净利润", "归属于母公司所有者(或股东)的净利润", "归属于本行股东的净利润", "本行股东的净利润"],
    "MINORITY_NET_PROFIT_YTD": ["少数股东损益", "少数股东的净利润"],
    "BASIC_EPS_YTD": ["基本及稀释每股收益", "基本和稀释每股收益", "基本每股收益"],
    "PRE_IMPAIRMENT_OPERATING_PROFIT_YTD": ["减值损失前营业利润"],
    "CREDIT_IMPAIRMENT_YTD": ["信用减值损失"],
    "OTHER_IMPAIRMENT_YTD": ["其他资产减值损失"],
}


def spec(pages, multiplier, eps_pages=None, **kwargs):
    return {"pages": pages, "multiplier": multiplier, "eps_pages": eps_pages or [], "widths": [2], "column": 0, **kwargs}


# 页码指物理 PDF 页；按正文表头选择，登记过程中不读任何账户收益。
PROFILES = {
    "1219306493": spec([126, 127], 1000000, [227], eps_start="(a)基本每股收益", eps_stop="(b)稀释每股收益", parent_from_note=True,
        ordinary="归属于母公司普通股股东的本年净利润", parent="归属于母公司股东的本年净利润", shares="已发行在外普通股的加权平均数", share_multiplier=1000000,
        deductions={"PREFERRED_DISTRIBUTION": "母公司优先股宣告股息", "PERPETUAL_DISTRIBUTION": "母公司永续债利息"}),
    "1212751519": spec([152], 1, [230], widths=[2, 4], eps_start="56、每股收益", eps_stop="57、现金流量表项目注释",
        ordinary="归属于普通股股东的当年净利润", parent="归属于母公司股东的当年净利润", shares="年末发行在外普通股的加权数", share_multiplier=1,
        deductions={"OTHER_EQUITY_DISTRIBUTION": "其他权益工具股息影响"}),
    "1212709852": spec([89, 90], 1, [158], eps_start="60.每股收益", eps_stop="本公司无稀释性潜在普通股",
        ordinary="归属于本公司普通股股东的当年净利润", ordinary_occurrence=1, parent="归属于本公司普通股股东的当年净利润", parent_occurrence=0,
        shares="本公司发行在外普通股的加权平均数", share_multiplier=1, deductions={"OTHER_EQUITY_DISTRIBUTION": "归属于本公司其他权益持有者的当年净利润"}),
    "1212626208": spec([136], 1000000, [261], eps_start="(a)每股收益", eps_stop="扣除非经常性损益后",
        ordinary="归属于本行普通股股东的净利润", parent="归属于本行股东的净利润", shares="加权平均普通股股本数", share_multiplier=1000000,
        eps_label="归属于本行普通股股东的基本和稀释每股收益", deductions={"PREFERRED_DISTRIBUTION": "归属于本行优先股股东的净利润", "PERPETUAL_DISTRIBUTION": "归属于本行永续债投资者的净利润"}),
    "1214964494": spec([8, 9], 1),
    "1216222871": spec([164], 1, [242], eps_start="54.每股收益", eps_stop="扣除非经常性损益的基本每股收益", note_formula=True,
        ordinary="归属于本公司普通股股东的当年净利润", parent="归属于母公司股东的净利润", shares="发行在外的普通股加权平均数", share_multiplier=1,
        deductions={"OTHER_EQUITY_DISTRIBUTION": "其他权益工具股息影响"}),
    "1212745096": spec([145, 146], 1, [228, 229], eps_start="(1)基本每股收益", eps_stop="(2)稀释每股收益",
        ordinary="归属于本公司普通股股东的合并净利润", parent="归属于母公司的合并净利润", shares="年末普通股的加权平均数", share_multiplier=1,
        deductions={"OTHER_EQUITY_DISTRIBUTION": "其他权益工具股息影响", "RESTRICTED_STOCK_DIVIDEND": "对限制性股票激励计划持有人的分红"},
        share_components={"OPENING_ORDINARY_SHARES": ["年初已发行普通股股数", 1], "REPURCHASE_WEIGHTED_SHARES": ["回购股份的影响", -1], "CONVERSION_WEIGHTED_SHARES": ["可转债持有人转股的影响", 1]}),
    "1208663570": spec([3], 1000000, widths=[4], column=2, summary=True),
    "1209490205": spec([190, 191], 1000000, [341], eps_start="44.每股收益", eps_stop="45.现金及现金等价物",
        ordinary="归属于母公司普通股股东的当年净利润", parent="归属于母公司股东的当年净利润", shares="当年发行在外普通股股数的加权平均数", share_multiplier=1000000,
        deductions={"OTHER_EQUITY_DISTRIBUTION": "归属于母公司其他权益持有者的当年净利润"}),
    "1219376072": spec([148, 149], 1000000, [269], eps_start="(1)基本每股收益", eps_stop="(2)稀释每股收益",
        ordinary="归属于母公司普通股股东的合并净利润", shares="当期发行在外普通股的加权平均数", share_multiplier=1000000,
        ordinary_equals_parent_explicit=True, deductions={},
        share_components={"OPENING_ORDINARY_SHARES": ["年初已发行的普通股数", 1], "CORE_EMPLOYEE_WEIGHTED_SHARES": ["核心人员持股计划所持股份加权平均数", 1], "SERVICE_PLAN_WEIGHTED_SHARES": ["长期服务计划所持股份加权平均数", 1], "CONSOLIDATED_PRODUCT_WEIGHTED_SHARES": ["合并资管产品所持股份加权平均数", 1], "CANCELLED_TREASURY_WEIGHTED_SHARES": ["注销库存股加权平均数", 1], "REPURCHASE_WEIGHTED_SHARES": ["股票回购股份加权平均数", 1]}),
    "1204547754": spec([140], 1000000, [239], eps_start="52每股收益",
        ordinary="归属于母公司普通股股东的当期净利润", parent="归属于母公司股东的当期净利润", shares="年末发行在外的普通股加权平均数", share_multiplier=None,
        deductions={"OTHER_EQUITY_DISTRIBUTION": "归属于母公司其他权益持有者的当期净利润"}),
    "1211437303": spec([10, 11], 1000000),
    "1212698832": spec([152, 153], 1000000, [223], eps_start="59.每股收益", eps_stop="(2)稀释每股收益",
        ordinary="归属于本公司股东的当期净利润", shares="本公司发行在外普通股的加权平均数", share_multiplier=1000000,
        ordinary_equals_parent_explicit=True, deductions={}),
    "1212669927": spec([119, 120], 1000000, [217], eps_start="53.每股收益", eps_stop="(2)稀释每股收益",
        ordinary="归属于母公司普通股股东的当期净利润", parent="归属于母公司股东的合并净利润", shares="本公司发行在外普通股的加权平均数", share_multiplier=None,
        deductions={"OTHER_EQUITY_DISTRIBUTION": "归属于母公司其他权益工具持有者的当期净利润"}, explicit_current_dash_zero=True),
    "1212730963": spec([178, 179, 180, 181], 1000000, [363], widths=[2, 4], eps_start="(1)每股收益", eps_stop="扣除非经常性损益后",
        ordinary="归属于本行普通股股东的净利润", parent="归属于本行股东的净利润", shares="加权平均普通股股数", share_multiplier=1000000,
        eps_label="归属于本行普通股股东的基本和稀释每股收益", deductions={"OTHER_EQUITY_DISTRIBUTION": "归属于本行其他权益工具持有者的净利润"}),
}


def norm(text):
    return unicodedata.normalize("NFKC", text).translate(TRAD).replace("損", "损").replace("\r", "").replace("−", "-").replace("—", "-").replace("–", "-").replace("\u3164", "").replace("\u1160", "")


def compact(text):
    return re.sub(r"\s+", "", norm(text))


def now():
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def read(path):
    return json.loads(path.read_text("utf-8"))


def save(path, data, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def decimal(token):
    v = norm(token).replace(",", "")
    v = re.sub(r"人民币|元/股|元", "", v)
    if v in {"-", ""}:
        return None
    return -Decimal(v[1:-1]) if v.startswith("(") else Decimal(v)


CELL = re.compile(r"(?:人民币)?(?:\(?[-+]?\d[\d,]*(?:\.\d+)?\)?|-)(?:元(?:/股)?)?\Z")
META = re.compile(r"(?:[一二三四五六七八九十]+、)?\d{1,3}(?:\(\d+\))?\Z")
FORMULA = re.compile(r"\d{1,2}=\d{1,2}[-+÷×]\d{1,2}\Z")


def parse_tail(tail, widths, allow_note, allow_formula=False):
    tokens = tail.strip().split()
    note = None
    if len(tokens) not in widths and len(tokens) - 1 in widths and allow_note:
        if META.fullmatch(tokens[0]) or (allow_formula and FORMULA.fullmatch(tokens[0])):
            note, tokens = tokens[0], tokens[1:]
    if len(tokens) not in widths or not all(CELL.fullmatch(t) for t in tokens):
        return None
    return {"cells": tokens, "values": [decimal(t) for t in tokens], "footnote_or_formula": note}


def rows_in_pages(pages, selected, start=None, stop=None, statement=False):
    lines = [{"page": p, "raw": s, "text": norm(s).strip()} for p in selected for s in pages[p-1].splitlines() if norm(s).strip()]
    if start:
        pos = next((i for i, r in enumerate(lines) if compact(start) in compact(r["text"])), None)
        if pos is None:
            raise ValueError("未找到已登记的附注起点: " + start)
        lines = lines[pos:]
    if statement:
        # 两个印刷页出现在同一物理页时，必须先定位合并表，再在公司表前结束。
        pos = next((i for i, r in enumerate(lines) if "合并" in compact(r["text"]) and "利润表" in compact(r["text"])), None)
        if pos is None:
            raise ValueError("已登记页面没有合并利润表")
        lines = lines[pos:]
        end = next((i for i, r in enumerate(lines[1:], 1) if "利润表" in compact(r["text"]) and "合并" not in compact(r["text"]) and re.search(r"(?:公司|银行|本行)利润表", compact(r["text"]))), None)
        if end is not None:
            lines = lines[:end]
    if stop:
        pos = next((i for i, r in enumerate(lines[1:], 1) if compact(stop) in compact(r["text"])), None)
        if pos is None:
            raise ValueError("未找到已登记的附注终点: " + stop)
        lines = lines[:pos]
    return lines


def row_matches(lines, label, widths, allow_note=True, allow_formula=False):
    lab = r"\s*".join(re.escape(c) for c in compact(label))
    prefix = r"[\s一二三四五六七八九十0-9、.()\-]*(?:[减加]\s*[:：]\s*)?"
    # 金额的括号表示负值，只有含文字的括号才是单位或标签注释。
    annotation = r"(?:\((?![-+]?\d[\d,.]*\))[^)]*\)\s*)?"
    pattern = re.compile(r"^" + prefix + lab + r"\s*" + annotation + r"\*?\s*(.*?)\s*$", re.S)
    matched = []
    for i in range(len(lines)):
        for n in range(1, min(4, len(lines)-i)+1):
            chunk = lines[i:i+n]
            # 只拼同一物理页上的换行；跨页表头不能混作数字。
            if len({r["page"] for r in chunk}) != 1:
                break
            m = pattern.fullmatch("\n".join(r["text"] for r in chunk))
            if m:
                data = parse_tail(m.group(1), widths, allow_note, allow_formula)
                if data:
                    matched.append({"label": label, "page": chunk[0]["page"], "line_index": i,
                                    "raw_row": "\n".join(r["raw"] for r in chunk), **data})
                    break
    return matched


def select_row(lines, labels, widths, occurrence=None, allow_note=True, allow_formula=False, required=True):
    found = []
    for label in ([labels] if isinstance(labels, str) else labels):
        found.extend(row_matches(lines, label, widths, allow_note, allow_formula))
    found.sort(key=lambda r: (r["page"], r["line_index"]))
    if not found:
        if required:
            raise ValueError("未识别表格行: " + str(labels))
        return None
    if occurrence is not None:
        if occurrence >= len(found):
            raise ValueError("登记的行次不存在: " + str(labels))
        return found[occurrence]
    unique = {tuple(r["values"]) for r in found}
    if len(unique) != 1:
        raise ValueError("同名行有不同金额，必须明确表格范围或行次: " + str(labels))
    return found[0]


def fact(metric, row, source, multiplier, column=0, unit="CNY", status="PASS_EXPLICIT_CURRENT_ORIGINAL"):
    value = row["values"][column]
    return {"announcement_id": source["announcement_id"], "ts_code": source["ts_code"], "sec_name": source["sec_name"],
        "report_period": str(source["report_period"])[:10], "event_publication_date": str(source["event_publication_date"])[:10],
        "official_pdf_url": source["official_pdf_url"], "source_sha256": source["sha256"], "metric_id": metric,
        "metric_value_exact": str(value * Decimal(multiplier)) if value is not None and multiplier is not None else None,
        "unit": unit, "source_unit_multiplier": multiplier, "source_page": row["page"], "source_label": row["label"],
        "source_raw_row": row["raw_row"], "source_raw_value": row["cells"][column], "all_numeric_cells": row["cells"],
        "selected_column_one_based": column+1, "footnote_or_formula": row["footnote_or_formula"],
        "statement_scope": "CONSOLIDATED", "period_scope": "FULL_YEAR" if str(source["report_period"])[5:7] == "12" else "YEAR_TO_DATE",
        "status": status, "comparative_is_earlier_original": False, "weighted_share_rounding": "REPORTED_MILLION_SHARES" if unit == "SHARES" and multiplier == 1000000 else None}


def relation(name, left, right, tolerance=Decimal(".02")):
    if len(left) != len(right) or any(v is None for v in left+right):
        raise ValueError("会计关系缺少明确数值: " + name)
    residuals = [a-b for a,b in zip(left,right)]
    if any(abs(x) > tolerance for x in residuals):
        raise ValueError(f"会计关系不一致: {name} {residuals}")
    return {"name": name, "left": list(map(str,left)), "right": list(map(str,right)), "residuals_reported_units": list(map(str,residuals)), "tolerance": str(tolerance)}


def process(aid, root=ROOT):
    d = read(root / "reports/research/510300_financial_ttm_dependencies_v1/page_texts" / f"{aid}.json")
    s, pages, p = d["source"], d["pages"], PROFILES[aid]
    raw = root / s["raw_path"]
    if hashlib.sha256(raw.read_bytes()).hexdigest() != s["sha256"]:
        raise ValueError("原始 PDF 与下载记录不一致: " + aid)
    if not s["subject"]["status"].startswith("PASS"):
        evidence = read(root / "reports/research/510300_financial_ttm_dependencies_v1/annual_subject_sections_supplement.json")
        receipt = next((r for r in evidence["rows"] if r["announcement_id"] == aid), None)
        if receipt is None or not receipt["status"].startswith("PASS") or receipt["source"]["sha256"] != s["sha256"]:
            raise ValueError("缺少全年报告主体补充证据")
        if receipt["cached_pages_sha256"] != hashlib.sha256((root / "reports/research/510300_financial_ttm_dependencies_v1/page_texts" / f"{aid}.json").read_bytes()).hexdigest():
            raise ValueError("主体证据引用的全文缓存变化")
    whole = compact("\n".join(pages[n-1] for n in p["pages"]))
    if "人民币" not in whole or str(s["report_period"])[:4] not in whole and not p.get("summary"):
        raise ValueError("报告期间或人民币单位未确认")
    if p["multiplier"] == 1000000 and "百万元" not in whole:
        raise ValueError("未找到已登记的百万元单位")
    lines = rows_in_pages(pages, p["pages"], statement=not p.get("summary"))
    rows = {}
    for metric, labels in LABELS.items():
        r = select_row(lines, labels, p["widths"], required=False)
        if r:
            rows[metric] = r
    facts, extra, checks = [], [], []
    ep = None
    if p["eps_pages"]:
        note = rows_in_pages(pages, p["eps_pages"], p.get("eps_start"), p.get("eps_stop"))
        get = lambda label, occurrence=None: select_row(note, label, [2], occurrence, allow_formula=p.get("note_formula", False))
        ep = {"ordinary": get(p["ordinary"], p.get("ordinary_occurrence")), "shares": get(p["shares"]),
              "eps": get(p.get("eps_label", LABELS["BASIC_EPS_YTD"]))}
        if p.get("parent"):
            ep["parent"] = get(p["parent"], p.get("parent_occurrence"))
        if p.get("parent_from_note"):
            rows["PARENT_NET_PROFIT_YTD"] = ep["parent"]
        ep["deductions"] = {k:get(label) for k,label in p.get("deductions",{}).items()}
        ep["share_components"] = {k:{"row":get(v[0]), "sign":v[1]} for k,v in p.get("share_components",{}).items()}
        ov = ep["ordinary"]["values"]
        parent = rows["PARENT_NET_PROFIT_YTD"]["values"][:2]
        if "parent" in ep:
            checks.append(relation("附注归母利润与合并表一致", ep["parent"]["values"], parent))
        if ep["deductions"]:
            ds = ep["deductions"].values()
            # 原始横杠仍保存为缺失。只在该附注的利润桥接中视为无本项扣除；不会写成零事实。
            after = [parent[i] - sum(abs(r["values"][i]) if r["values"][i] is not None else Decimal(0) for r in ds) for i in range(2)]
            checks.append(relation("归母利润减非普通股及限制性股票分配等于普通股利润", after, ov))
        elif p.get("ordinary_equals_parent_explicit"):
            checks.append(relation("附注明确基本收益分子等于归母利润", ov, parent))
        checks.append(relation("附注基本每股收益与主表一致", ep["eps"]["values"], rows["BASIC_EPS_YTD"]["values"][:2], Decimal(0)))
        if ep["share_components"]:
            sums = [sum((v["row"]["values"][i] or Decimal(0))*v["sign"] for v in ep["share_components"].values()) for i in range(2)]
            checks.append(relation("普通股加权股数变动桥接", sums, ep["shares"]["values"], Decimal(1)))
        mult = p.get("share_multiplier")
        if mult is not None:
            computed = [ov[i]*p["multiplier"]/(ep["shares"]["values"][i]*mult) for i in range(2)]
            checks.append(relation("普通股利润除加权股数与披露基本收益一致", computed, ep["eps"]["values"], Decimal(".0051")))
        extra.append(fact("ORDINARY_NET_PROFIT_YTD", ep["ordinary"], s, p["multiplier"]))
        extra.append(fact("WEIGHTED_ORDINARY_SHARES", ep["shares"], s, mult, unit="SHARES" if mult else "REPORTED_SHARE_COUNT_UNIT_UNPROVEN", status="PASS_REPORTED_WEIGHTED_SHARES" if mult else "NO_VIEW_SHARE_MULTIPLIER_NOT_EXPLICIT"))
        for metric,r in ep["deductions"].items():
            extra.append(fact(metric, r, s, p["multiplier"], status="NO_VIEW_REPORTED_DASH_NO_NUMERIC_FACT" if r["values"][0] is None else "PASS_EPS_NUMERATOR_ADJUSTMENT_NOT_ORDINARY_CASH_RETURN"))
        for metric,v in ep["share_components"].items():
            extra.append({**fact(metric, v["row"], s, mult, unit="SHARES", status="PASS_WEIGHTED_SHARE_COMPONENT_NOT_BUYBACK_CASH"), "bridge_sign":v["sign"]})
    for metric in CORE:
        if metric not in rows:
            if p.get("summary") and metric == "OPERATING_PROFIT_YTD":
                continue
            raise ValueError("核心指标缺失: " + metric)
        facts.append(fact(metric, rows[metric], s, 1 if metric == "BASIC_EPS_YTD" else p["multiplier"], p["column"], unit="CNY_PER_SHARE" if metric == "BASIC_EPS_YTD" else "CNY"))
    if not p.get("summary"):
        v = {k:r["values"][:2] for k,r in rows.items()}
        sign = -1 if s["ts_code"] in BROKERS else 1
        rp = [a+sign*b for a,b in zip(v["OPERATING_REVENUE_YTD"],v["OPERATING_EXPENSE_YTD"])]
        if s["ts_code"] == "000001.SZ":
            checks.append(relation("收入扣除减值前支出", rp, v["PRE_IMPAIRMENT_OPERATING_PROFIT_YTD"]))
            rp = [a+b+c for a,b,c in zip(rp,v["CREDIT_IMPAIRMENT_YTD"],v["OTHER_IMPAIRMENT_YTD"])]
        checks.append(relation("收入和全部营业支出至营业利润", rp, v["OPERATING_PROFIT_YTD"]))
        if "MINORITY_NET_PROFIT_YTD" in v:
            checks.append(relation("归母与少数股东利润至合并净利润", [a+b for a,b in zip(v["PARENT_NET_PROFIT_YTD"],v["MINORITY_NET_PROFIT_YTD"])],v["TOTAL_NET_PROFIT_YTD"]))
        elif s["ts_code"] == "000001.SZ":
            checks.append(relation("附注归母与合并净利润一致",v["PARENT_NET_PROFIT_YTD"],v["TOTAL_NET_PROFIT_YTD"]))
    return {"source":s, "profile":p, "status":"PASS_BOUNDED_ORIGINAL_COMPONENT_EXTRACTION", "core_facts":facts, "additional_facts":extra, "statement_rows":rows, "eps_note_rows":ep, "identities":checks,
            "strategy_return_read":False, "ttm_automatically_admitted":False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    if args.freeze:
        paths = [Path(__file__), ROOT / "docs/510300_FINANCIAL_ANNUAL_COMPONENTS_V1.md", ROOT / "tests/test_financial_annual_components_v1.py", SOURCE / "annual_subject_sections_supplement.json"]
        for aid in PROFILES:
            f = SOURCE / "page_texts" / f"{aid}.json"
            paths += [f, ROOT / read(f)["source"]["raw_path"]]
        save(MANIFEST, {"frozen_at":now(), "scope":"金融原始年报与同期报告来源提取，未读取策略价格和收益", "profiles":PROFILES,
            "files":[{"path":p.relative_to(ROOT).as_posix(), "bytes":p.stat().st_size, "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}, exclusive=True)
        print("年报来源提取协议与文件已冻结。")
        return
    if not args.preflight:
        for r in read(MANIFEST)["files"]:
            if hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest() != r["sha256"]:
                raise ValueError("冻结文件变化: "+r["path"])
        if (OUT/"result.json").exists():
            raise FileExistsError("已完成结果禁止覆盖")
    records, failures = [], []
    for aid in PROFILES:
        try:
            d = process(aid)
            records.append(d)
            print(d["source"]["sec_name"],aid,"核心",len(d["core_facts"]),"附注",len(d["additional_facts"]),"关系",len(d["identities"]))
        except (ValueError, KeyError) as exc:
            failures.append({"announcement_id":aid,"error":str(exc)})
            print("待修正版式",aid,str(exc))
    if args.preflight:
        print("预检查通过",len(records),"未通过",len(failures))
        return
    if failures:
        raise ValueError("冻结来源仍有未解决错误: " + str(failures))
    core = [f for d in records for f in d["core_facts"]]
    extras = [f for d in records for f in d["additional_facts"]]
    OUT.mkdir(parents=True, exist_ok=True)
    for d in records:
        save(OUT/"document_records"/(d["source"]["announcement_id"]+".json"),d,exclusive=True)
    for name, rows in [("current_original_core_facts",core),("ordinary_eps_components",extras)]:
        pd.DataFrame(rows).to_parquet(OUT/(name+".parquet"),index=False)
        pd.DataFrame(rows).to_csv(OUT/(name+".csv"),index=False,encoding="utf-8-sig")
    result = {"completed_at":now(),"status":"ORIGINAL_ANNUAL_AND_ORDINARY_EPS_COMPONENTS_EXTRACTED_TTM_NEXT", "documents":len(records),"core_facts":len(core),"additional_facts":len(extras),
              "additional_unresolved_numeric_or_unit":sum(f["metric_value_exact"] is None for f in extras), "identities":sum(len(d["identities"]) for d in records),
              "strategy_returns_read":False,"new_accounts":0,"goal_achieved":False}
    save(OUT/"result.json",result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
