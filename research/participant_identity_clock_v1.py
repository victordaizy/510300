"""整理510300原始定期报告中的投资者身份、持有份额和公开时钟。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import pypdfium2 as pdfium
import requests
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_participant_identity_clock_v1"
OLD = ROOT / "reports/research/510300_original_fund_subscription_reports_v1"
TZ = ZoneInfo("Asia/Shanghai")
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/"}


def now():
    return datetime.now(TZ).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def compact(text):
    return re.sub(r"\s+", "", str(text))


def initialize():
    path = OUT / "source_plan.json"
    if path.exists():
        raise ValueError("本轮来源计划已存在，不覆盖")
    for folder in ("inputs", "raw", "receipts", "facts", "figures", "evidence"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    plan = {
        "study_id": "510300_PARTICIPANT_IDENTITY_CLOCK_V1", "created_at": now(),
        "user_objective": "我们做过很多方向，还有一个是微观市场的资金面，哪些资金的进入可能带来上涨，可能在某些节点，公募 私募，国家队，社保基金包括散户，动量不一定一直有效，反转也不一定一直有效，我们最终的目标是实现夏普1.2，年化10或者夏普1.5",
        "stage": "IDENTITY_AND_INFORMATION_AVAILABILITY_BEFORE_NEW_PREDICTIVE_SPECIFICATION",
        "source_universe": "510300成立以来截至2026-09-22的全部年报/中报。先沿用已保存的上交所公告目录27份，再查询2026年官方目录。原始季度报告仅用于判断匿名大持有人披露是否能补足实名频率。",
        "fields": ["所属期末", "官方公告日期", "报告送出日期", "下一可用交易日", "机构总份额", "个人总份额", "实名前十持有人份额", "持有人名册及统计口径"],
        "identity_rules": [
            "机构/个人使用报告原始分类；机构包含联接基金、汇金等，不能互相作为独立资金票重复相加。",
            "实名汇金投资与汇金资管分列；合计仅称两家汇金主体，不外推为所有国家队。",
            "实名社保与基本养老金分列；其余名称没有身份依据时标记未分类。",
            "前十未出现不等于零；缺失保留区间上界或未知，不用零填补。",
            "个人直接持有ETF不代表全部散户权益资产；通过公募/私募持有者属于另一个观察层次。",
            "份额存量差不是以现金计的净流入；实名份额增加不证明购买都在一级申购环节发生。",
            "报告所属期末不能代替公告日期；仅日期的公告按当日结束后处理。",
            "历史文件当前重取不冒充历史收件快照；元数据晚于公告需另列。"
        ],
        "new_price_labels": 0, "new_model_fits": 0, "new_account_runs": 0,
        "chart_rule": "价格趋势仅作完整日频描述，不用图上拐点选窗口；持有结构与公开时点分别表达。",
        "prior_failures": ["DAILY_ETF_FLOW_DIRECTION_REJECTED", "510300_ORIGINAL_QUARTERLY_FLOW_POLICY_V1", "510300_MARKET_LEVERAGE_CASCADE_5D_DISCOVERY_V0", "SELECTED_MIX_BAND10_SIMPLE2"],
        "main_capital": 200000, "cost_comparison_capital": 20000, "annual_days": 242,
        "goal_complete": False, "position_impact": 0,
    }
    dump(path, plan)
    for name, source in {"previous_periodic_catalogue.parquet": OLD / "all_official_periodic_announcements.parquet",
                         "market.parquet": ROOT / "reports/research/510300_lpr_joint_response_20d_v1/inputs/market.parquet",
                         "calendar.parquet": ROOT / "reports/research/510300_lpr_joint_response_20d_v1/inputs/calendar.parquet"}.items():
        target = OUT / "inputs" / name
        target.write_bytes(source.read_bytes())
    dump(OUT / "receipts/initial_inputs.json", {
        "saved_at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": sha(p)} for p in (OUT / "inputs").iterdir()],
        "new_study_outcomes_observed": False})
    print("资金身份来源计划及输入快照已保存", flush=True)


def catalogue():
    old = pd.read_parquet(OUT / "inputs/previous_periodic_catalogue.parquet")
    params = {"isPagination": "true", "pageHelp.pageSize": 100, "pageHelp.pageNo": 1,
              "pageHelp.beginPage": 1, "pageHelp.endPage": 1, "pageHelp.cacheSize": 1,
              "type": "inParams", "sqlId": "COMMON_PL_JJXX_JJGG_NEW_L", "SECURITY_CODE": "510300",
              "TITLE": "", "BULLETIN_TYPE": "reits03,fund03", "START_DATE": "2026-01-01", "END_DATE": "2026-09-22"}
    r = requests.get("https://query.sse.com.cn/commonQuery.do", params=params, headers=HEADERS, timeout=35)
    raw = OUT / "raw/catalogue_2026.json"
    raw.write_bytes(r.content)
    dump(OUT / "receipts/catalogue_2026.json", {"url": r.url, "status": r.status_code, "retrieved_at": now(), "sha256": sha(raw)})
    r.raise_for_status()
    obj = r.json()
    rows = obj.get("result", obj.get("pageHelp", {}).get("data", []))
    if not rows or any(str(x.get("SECURITY_CODE")) != "510300" for x in rows):
        raise ValueError("官方目录为空或基金身份不一致")
    if int(obj.get("pageHelp", {}).get("total", len(rows))) > len(rows):
        raise ValueError("官方目录需分页，当前不得声明完整")
    current = pd.DataFrame(rows)
    combined = pd.concat([old, current], ignore_index=True).drop_duplicates(["SSEDATE", "TITLE", "URL"])
    chosen = combined[combined.ORG_BULLETIN_TYPE_DESC.str.contains("年度报告|中期报告|年报|半年报", regex=True)].copy()
    chosen = chosen.sort_values("SSEDATE").reset_index(drop=True)
    chosen["source_url"] = chosen.URL.map(lambda x: x if x.startswith("http") else "https://www.sse.com.cn" + x)
    chosen["key"] = chosen.apply(lambda x: x.SSEDATE.replace("-", "") + "_" + hashlib.sha256(x.URL.encode()).hexdigest()[:10], axis=1)
    chosen.to_csv(OUT / "facts/官方年报中报目录.csv", index=False, encoding="utf-8-sig")
    chosen.to_parquet(OUT / "facts/catalogue.parquet", index=False)
    print(f"官方年报中报目录：{len(chosen)}份，2026年新增查询{len(current)}条公告", flush=True)


def fetch_one(row):
    key = row["key"]
    pdf = OUT / "raw" / (key + ".pdf")
    receipt = OUT / "receipts" / (key + ".json")
    if pdf.exists() and receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old.get("status") == "HTTP_PDF_SAVED" and old["sha256"] == sha(pdf):
            return old
    info = {"key": key, "source_url": row["source_url"], "catalogue_date": row["SSEDATE"], "title": row["TITLE"], "retrieved_at": now()}
    try:
        r = requests.get(row["source_url"], headers=HEADERS, timeout=45)
        info.update(http_status=r.status_code, final_url=r.url, bytes=len(r.content))
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise ValueError("返回内容不是PDF")
        pdf.write_bytes(r.content)
        info.update(status="HTTP_PDF_SAVED", sha256=sha(pdf), raw_path=pdf.relative_to(OUT).as_posix())
    except (requests.RequestException, ValueError) as exc:
        info.update(status="SOURCE_ACCESS_FAILED", error=str(exc))
    dump(receipt, info)
    return info


def download():
    rows = pd.read_parquet(OUT / "facts/catalogue.parquet").to_dict("records")
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [executor.submit(fetch_one, row) for row in rows]
        for task in as_completed(tasks):
            result = task.result()
            results.append(result)
            print(f"原始报告{len(results)}/{len(rows)}：{result['key']} {result['status']}", flush=True)
    dump(OUT / "receipts/download_summary.json", sorted(results, key=lambda x: x["key"]))


def extract():
    all_records = []
    for source in json.loads((OUT / "receipts/download_summary.json").read_text(encoding="utf-8")):
        if source["status"] != "HTTP_PDF_SAVED":
            continue
        saved = OUT / "facts" / (source["key"] + "_extraction.json")
        if saved.exists():
            old = json.loads(saved.read_text(encoding="utf-8"))
            if old["source"]["sha256"] == source["sha256"]:
                all_records.append({"key": source["key"], "pages": old["page_count"], "holder_pages": [x["page"] for x in old["selected_pages"]]})
                continue
        path = OUT / source["raw_path"]
        if sha(path) != source["sha256"]:
            raise ValueError("PDF哈希与收据不符")
        pages = []
        doc = pdfium.PdfDocument(path)
        for index in range(len(doc)):
            page = doc[index]
            tp = page.get_textpage()
            pages.append(tp.get_text_range())
            tp.close()
            page.close()
        doc.close()
        (OUT / "raw" / (source["key"] + "_pages.json")).write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        selected = []
        for i, text in enumerate(pages):
            clean = compact(text)
            if ("期末基金份额持有人户数及持有人结构" in clean or "期末上市基金前十名持有人" in clean or
                    ("持有人名称" in clean and "持有份额" in clean) or "中央汇金" in clean or "全国社保" in clean):
                if clean.count("...") > 4 or "目录" in clean[:80]:
                    continue
                selected.extend([i, min(i + 1, len(pages) - 1)])
        selected = sorted(set(selected))
        with pdfplumber.open(path) as pdf:
            tables = [{"page": i + 1, "text": pages[i], "tables": pdf.pages[i].extract_tables()} for i in selected]
            metadata = pdf.metadata
        record = {"source": source, "cover": pages[0], "overview": pages[1:5], "page_count": len(pages), "metadata": metadata, "selected_pages": tables}
        dump(OUT / "facts" / (source["key"] + "_extraction.json"), record)
        all_records.append({"key": source["key"], "pages": len(pages), "holder_pages": [x + 1 for x in selected]})
        print(f"持有人表格抽取：{source['key']}，候选页{[x + 1 for x in selected]}", flush=True)
    dump(OUT / "facts/extraction_index.json", all_records)


def number(text):
    return float(compact(text).replace(",", "").replace("%", ""))


def investor_label(name):
    """名称只提供可证实的身份层次，保管/担保账户不归属最终出资人。"""
    if name == "中央汇金投资有限责任公司":
        return "汇金投资直接实名"
    if name == "中央汇金资产管理有限责任公司":
        return "汇金资管直接实名"
    if "汇金" in name:
        return "名称含汇金的资管计划_另列"
    if "全国社保基金" in name or "全国社会保障基金" in name:
        return "全国社保实名组合"
    if "基本养老保险基金" in name:
        return "基本养老实名组合"
    if "中央结算" in name or "担保证券账户" in name or "客户资金" in name:
        return "名义持有人或担保账户_最终身份未知"
    if "融券专用" in name:
        return "融券专用证券账户_非方向性买入证明"
    if "联接基金" in name:
        return "本基金联接基金_不得重复相加"
    if "私募" in name:
        return "显名私募载体_出资人属性另判"
    if "信托" in name or "资产管理计划" in name:
        return "信托或资管载体_最终身份未知"
    if "保险" in name or "人寿" in name:
        return "保险名称账户"
    if "证券" in name or "金融" in name:
        return "证券金融机构名称_资金用途未知"
    return "未分类实名持有人"


def build_facts():
    calendar = pd.read_parquet(OUT / "inputs/calendar.parquet")
    print("交易日历列：" + ",".join(calendar.columns), flush=True)
    date_column = "date" if "date" in calendar else "trade_date"
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.astype(bool), date_column]
                                         if "is_open" in calendar else calendar[date_column])).normalize().sort_values()
    aggregate, holders, issues = [], [], []
    for path in sorted((OUT / "facts").glob("*_extraction.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        source = obj["source"]
        key = source["key"]
        cover = compact(obj["cover"])
        dates = re.findall(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", cover)
        dates = [pd.Timestamp(int(y), int(m), int(d)) for y, m, d in dates]
        period_end = next((x for x in dates if (x.month, x.day) in ((6, 30), (12, 31))), None)
        if period_end is None or len(dates) < 2 or "华泰柏瑞沪深300交易型开放式指数证券投资基金" not in cover:
            raise ValueError(f"报告封面身份或日期无法确定：{key}")
        send_date = dates[-1]
        public_date = max(pd.Timestamp(source["catalogue_date"]), send_date)
        available_session = days[days > public_date][0]
        pages = json.loads((OUT / "raw" / (key + "_pages.json")).read_text(encoding="utf-8"))
        all_text = compact("".join(pages))
        total_matches = re.findall(r"本报告期期末基金份额总额([\d,]+\.\d{2})", all_text)
        if len(set(total_matches)) != 1:
            raise ValueError(f"期末总份额不唯一：{key} {total_matches}")
        total = number(total_matches[0])
        numeric = []
        feeder = []
        candidate_holders = []
        in_holder_table = False
        for pg in obj["selected_pages"]:
            for table in pg["tables"]:
                for row in table:
                    cells = [compact(s) for s in row if s and s.strip()]
                    if len(cells) in (6, 8) and all(re.fullmatch(r"[\d.,%\-]+", s) for s in cells):
                        numeric.append((pg["page"], cells))
                    if len(cells) == 3 and "联接基金" in cells[0] and re.fullmatch(r"[\d,.]+", cells[1]):
                        feeder.append((number(cells[1]), number(cells[2]), pg["page"]))
                    if any("持有人名称" in c for c in cells):
                        in_holder_table = True
                    if in_holder_table and len(cells) == 4 and re.fullmatch(r"\d{1,2}", cells[0]) and 1 <= int(cells[0]) <= 11:
                        try:
                            amount, percent = number(cells[2]), number(cells[3])
                        except ValueError:
                            continue
                        if amount > 0 and 0 <= percent <= 100 and not re.fullmatch(r"[\d,.]+", cells[1]):
                            candidate_holders.append({"rank": int(cells[0]), "name": cells[1], "units": amount, "reported_percent": percent, "source_page": pg["page"]})
                    elif in_holder_table and len(cells) == 1 and candidate_holders and "建设银行)" == cells[0]:
                        candidate_holders[-1]["name"] += cells[0]
        if len(numeric) != 1:
            raise ValueError(f"持有人结构数值行不唯一：{key}")
        page, cells = numeric[0]
        vals = list(map(number, cells))
        reported_inst, personal = vals[2], vals[4]
        if len(cells) == 8:
            feeder_units, feeder_pct = vals[6], vals[7]
        elif len(set((x[0], x[1]) for x in feeder)) == 1:
            feeder_units, feeder_pct, _ = feeder[0]
        else:
            raise ValueError(f"联接基金持有份额不唯一：{key} {feeder}")
        if abs(reported_inst + personal + feeder_units - total) < .02:
            treatment = "机构原列不含联接基金"
            canonical_inst = reported_inst
        elif abs(reported_inst + personal - total) < .02:
            treatment = "机构原列已含联接基金_统一口径需扣除"
            canonical_inst = reported_inst - feeder_units
        else:
            raise ValueError(f"三类持有人份额不能与总份额对齐：{key}")
        for value, pct in ((reported_inst, vals[3]), (personal, vals[5]), (feeder_units, feeder_pct)):
            if abs(100 * value / total - pct) > .0051:
                raise ValueError(f"报告份额百分比不一致：{key} {value} {pct}")
        if abs(vals[0] * vals[1] - total) > vals[0] * .0051:
            raise ValueError(f"户数与户均份额不一致：{key}")
        ranks = [x["rank"] for x in candidate_holders]
        if candidate_holders and ranks not in (list(range(1, 11)), list(range(1, 12))):
            raise ValueError(f"前十持有人名次不完整：{key} {ranks}")
        for h in candidate_holders:
            if abs(100 * h["units"] / total - h["reported_percent"]) > .0051:
                raise ValueError(f"实名份额和比例不一致：{key} {h}")
        metadata_late = []
        for field in ("CreationDate", "ModDate"):
            v = str(obj["metadata"].get(field, ""))
            match = re.search(r"(?:D:)?(20\d{2})(\d{2})(\d{2})", v)
            if match and pd.Timestamp(int(match[1]), int(match[2]), int(match[3])) > public_date:
                metadata_late.append(field)
        record = {
            "key": key, "period_end": str(period_end.date()), "catalogue_date": source["catalogue_date"],
            "cover_send_date": str(send_date.date()), "publication_date_upper_bound": str(public_date.date()),
            "first_available_session": str(available_session.date()), "publication_lag_days": int((public_date - period_end).days),
            "total_units": total, "holder_accounts": vals[0], "reported_institution_units": reported_inst,
            "reported_institution_percent": vals[3], "direct_personal_units": personal, "direct_personal_percent": vals[5],
            "feeder_units": feeder_units, "feeder_percent": feeder_pct,
            "institution_excluding_feeder_units": canonical_inst, "feeder_treatment": treatment,
            "holder_section_page": page, "named_holder_count": len(candidate_holders),
            "named_holder_status": "TOP10_OR_TOP10_PLUS_FEEDER" if candidate_holders else "NO_NAMED_HOLDER_TABLE_IN_REPORT",
            "source_url": source["source_url"], "source_sha256": source["sha256"], "source_pdf": source["raw_path"],
            "historical_first_version_authenticated": False,
            "metadata_after_catalogue": "|".join(metadata_late),
            "clock_status": "NO_VIEW_LATER_PDF_METADATA" if metadata_late else "CATALOGUE_AND_COVER_DATE_CONSERVATIVE_NEXT_SESSION",
        }
        top10 = [h for h in candidate_holders if h["rank"] <= 10]
        threshold = next((h["units"] for h in top10 if h["rank"] == 10), None)
        for field, name in (("huijin_investment", "中央汇金投资有限责任公司"), ("huijin_asset", "中央汇金资产管理有限责任公司")):
            found = [h["units"] for h in top10 if h["name"] == name]
            record[field + "_units"] = found[0] if found else None
            record[field + "_lower"] = found[0] if found else 0.0
            record[field + "_upper"] = found[0] if found else (threshold if threshold is not None else total)
            record[field + "_status"] = "EXACT_NAMED_DIRECT_ACCOUNT" if found else ("NOT_IN_TOP10_NOT_ZERO" if top10 else "HOLDER_TABLE_UNAVAILABLE")
        record["huijin_direct_lower"] = record["huijin_investment_lower"] + record["huijin_asset_lower"]
        record["huijin_direct_upper"] = min(canonical_inst, record["huijin_investment_upper"] + record["huijin_asset_upper"])
        if record["huijin_direct_lower"] > record["huijin_direct_upper"] + .01:
            raise ValueError("汇金直接实名份额超过机构份额")
        record["huijin_direct_exact"] = abs(record["huijin_direct_upper"] - record["huijin_direct_lower"]) < .01
        record["ssf_named_records"] = sum("全国社保" in h["name"] or "全国社会保障" in h["name"] for h in top10)
        record["ssf_total_units_status"] = "UNKNOWN_TOP10_IS_NOT_FULL_REGISTER"
        aggregate.append(record)
        for h in candidate_holders:
            holders.append({"key": key, "period_end": record["period_end"], "publication_date": record["publication_date_upper_bound"],
                            **h, "identity_label": investor_label(h["name"]), "source_url": source["source_url"], "source_sha256": source["sha256"]})
        if metadata_late:
            issues.append({"key": key, "reason": "PDF元数据晚于目录日", "fields": metadata_late})
    frame = pd.DataFrame(aggregate).sort_values("period_end").reset_index(drop=True)
    for col in ("total_units", "institution_excluding_feeder_units", "direct_personal_units", "feeder_units"):
        frame[col + "_change"] = frame[col].diff()
    frame["huijin_direct_change_lower"] = frame.huijin_direct_lower - frame.huijin_direct_upper.shift(1)
    frame["huijin_direct_change_upper"] = frame.huijin_direct_upper - frame.huijin_direct_lower.shift(1)
    for name, df in (("持有结构与公开时钟", frame), ("实名持有人逐项记录", pd.DataFrame(holders))):
        df.to_csv(OUT / "facts" / (name + ".csv"), index=False, encoding="utf-8-sig")
        df.to_parquet(OUT / "facts" / (name + ".parquet"), index=False)
    amendment = {
        "recorded_at": now(), "reason": "读取原始表格后修正来源计划对联接基金分类的先验描述；尚未生成预测标签或收益检验。",
        "correction": "不能预设机构均含联接。按原表机构+个人、机构+个人+联接分别与原报告期末总份额核对，统一为机构不含联接、个人、联接三栏，并保留原值。",
        "original_plan_preserved": True, "period_counts": frame.feeder_treatment.value_counts().to_dict(),
        "new_predictive_labels": 0, "model_fits": 0,
    }
    dump(OUT / "evidence/联接基金口径修正.json", amendment)
    summary = {"study_id": "510300_PARTICIPANT_IDENTITY_CLOCK_V1", "status": "COMPLETED_HOLDER_SOURCE_PANEL_NOT_PREDICTIVE_TEST",
               "report_count": len(frame), "first_period": frame.period_end.iloc[0], "last_period": frame.period_end.iloc[-1],
               "named_holder_rows": len(holders), "reports_with_named_holders": int(frame.named_holder_count.gt(0).sum()),
               "last_named_holder_period": frame.loc[frame.named_holder_count.gt(0), "period_end"].iloc[-1],
               "publication_lag_min": int(frame.publication_lag_days.min()), "publication_lag_max": int(frame.publication_lag_days.max()),
               "publication_lag_median": float(frame.publication_lag_days.median()), "source_issues": issues,
               "new_model_fits": 0, "account_runs": 0, "independent_forward_events": 0, "whole_objective_complete": False}
    dump(OUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="510300资金身份来源整理")
    parser.add_argument("action", choices=["initialize", "catalogue", "download", "extract", "build_facts"])
    args = parser.parse_args()
    globals()[args.action]()


if __name__ == "__main__":
    main()
