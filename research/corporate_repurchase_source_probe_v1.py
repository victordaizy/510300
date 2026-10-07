"""从交易所公开目录取得两家预定成分公司的回购原文，不生成交易信号。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
from urllib.parse import urljoin

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_corporate_repurchase_source_probe_v1"
WEIGHTS = ROOT / "data/raw/constituents/000300_historical_weights.parquet"
STUDY = "510300_CORPORATE_REPURCHASE_SOURCE_PROBE_V1"
START, END = "2024-09-01", "2026-09-24"
SSE_URL = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
SZSE_URL = "https://www.szse.cn/api/disc/announcement/annList"


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("来源探查范围已经固定，不能覆盖。")
    for folder in ["raw", "receipts", "results", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    weights = pd.read_parquet(WEIGHTS)
    when = weights.loc[weights.trade_date.lt(START), "trade_date"].max()
    selected = weights[weights.trade_date.eq(when)].sort_values(["weight", "con_code"], ascending=[False, True]).head(2)
    assert selected.con_code.tolist() == ["600519.SH", "300750.SZ"]
    protocol = {"at": now(), "study_id": STUDY, "window": [START, END],
        "question": "政策支持、回购计划、融资授信与实际股票买入能否按公告时点分开，还原为指数内可观察的真实需求信息？",
        "universe_rule": "研究起点前最近一份已保存指数权重的前两名，用于验证来源与解析流程；不按此后的涨跌、回购金额或方案成败选公司。",
        "weight_date": when, "companies": selected[["con_code", "weight"]].to_dict("records"),
        "weights_limit": "历史供应商权重为样本选择依据，不宣称当时首发权重时间已认证；两家公司不代表全部指数。",
        "collection": "两交易所按公司、日期及回购标题查询公开目录，最多每家5页、每页100条；仅下载命中目录的发行人PDF。",
        "catalogue_completeness": "按接口返回total及唯一文件数量检验分页；失败、过滤不符或缺页不得宣称完整。",
        "fact_rules": ["计划、授信、实际执行和注销分别记录，不相加", "限制性股票回购注销不视为二级市场买入", "同一方案的累计执行额不跨公告求和", "多方案及A/H股份分开", "经济截止日期不能当信息发布日期", "只有日期时保守按该日末已知，下一交易日才可研究使用"],
        "provider_transport": "本次核对已有Tushare代理凭据过期，没有向该接口发请求；改用公开交易所来源，不保存凭据。",
        "no_new_account_before_source_resolution": True, "new_market_returns_loaded": False,
        "new_accounts": 0, "new_fits": 0, "new_independent_forward_observations": 0,
        "goal_achieved": False, "orders_authorized": False, "review_package_created": False,
        "weights_sha256": digest(WEIGHTS), "code_sha256": digest(Path(__file__))}
    save(OUT / "protocol.json", protocol, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print("已固定两家事前权重最大的公司、公开来源范围及事实处理规则。", flush=True)


def request(key, url, *, params=None, payload=None):
    receipt = OUT / "receipts" / f"{key}.json"
    if receipt.exists():
        prior = read(receipt)
        return (OUT / prior["raw_path"]).read_bytes() if prior.get("raw_path") else None, prior
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/" if "sse.com.cn" in url else "https://www.szse.cn/"}
    record = {"key": key, "requested_at": now(), "url": url, "params": params, "payload": payload}
    try:
        if payload is None:
            response = requests.get(url, params=params, headers=headers, timeout=(10, 30))
        else:
            response = requests.post(url, json=payload, headers=headers, timeout=(10, 30))
        raw = OUT / "raw" / f"{key}.bin"
        raw.write_bytes(response.content)
        record.update(completed_at=now(), http_status=response.status_code,
            status="HTTP_OK_NOT_YET_ADMITTED" if response.status_code == 200 else "HTTP_ERROR",
            raw_path=raw.relative_to(OUT).as_posix(), sha256=digest(raw), bytes=len(response.content), final_url=response.url)
        content = response.content
    except requests.RequestException as exc:
        record.update(completed_at=now(), status="REQUEST_FAILED", error_type=type(exc).__name__)
        content = None
    save(receipt, record, True)
    return content, record


def catalogue(exchange, symbol):
    rows, total, complete, error = [], None, False, None
    for page in range(1, 6):
        if exchange == "SSE":
            params = {"isPagination": "true", "productId": symbol[:6], "keyWord": "回购", "securityType": "0101,120100,020100,020200,120200",
                "reportType2": "", "reportType": "ALL", "beginDate": START, "endDate": END,
                "pageHelp.pageSize": 100, "pageHelp.pageNo": page, "pageHelp.beginPage": page,
                "pageHelp.cacheSize": 1, "pageHelp.endPage": page}
            content, receipt = request(f"{symbol}_catalogue_{page}", SSE_URL, params=params)
        else:
            payload = {"seDate": [START, END], "stock": [symbol[:6]], "searchKey": ["回购"],
                "channelCode": ["listedNotice_disc"], "pageSize": 100, "pageNum": page}
            content, receipt = request(f"{symbol}_catalogue_{page}", SZSE_URL, payload=payload)
        if receipt["status"] != "HTTP_OK_NOT_YET_ADMITTED":
            error = receipt["status"]
            break
        try:
            body = json.loads(content)
            if exchange == "SSE":
                info = body["pageHelp"]
                batch, count = body.get("result", info.get("data", [])), int(info["total"])
                normalized = [{"symbol": symbol, "catalogue_date": row.get("SSEDATE"),
                    "title": row.get("TITLE", ""), "source_url": urljoin("https://www.sse.com.cn/", row["URL"]),
                    "catalogue_security_code": str(row.get("SECURITY_CODE", row.get("PRODUCTID", ""))),
                    "catalogue_page": page} for row in batch]
            else:
                batch, count = body["data"], int(body["announceCount"])
                normalized = [{"symbol": symbol, "catalogue_date": row.get("publishTime"),
                    "title": row.get("title", ""), "source_url": urljoin("https://disc.static.szse.cn/", row["attachPath"]),
                    "catalogue_security_code": "|".join(row.get("secCode", [])), "catalogue_page": page} for row in batch]
            if total is not None and total != count:
                raise ValueError("分页总量发生变化")
            total = count
            if any("回购" not in row["title"] for row in normalized):
                raise ValueError("目录标题过滤未生效")
            if any(symbol[:6] not in row["catalogue_security_code"] for row in normalized):
                raise ValueError("目录股票过滤未生效")
            rows.extend(normalized)
            if len(rows) >= total:
                complete = len(rows) == total
                break
            if not batch:
                raise ValueError("到达总量前返回空页")
        except (KeyError, TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
    unique = len({row["source_url"] for row in rows})
    complete &= unique == len(rows)
    return {"exchange": exchange, "symbol": symbol, "total": total, "rows": rows,
        "unique_urls": unique, "complete": complete, "error": error}


def run():
    protocol = read(OUT / "protocol.json")
    assert digest(Path(__file__)) == protocol["code_sha256"]
    assert digest(WEIGHTS) == protocol["weights_sha256"]
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    records = []
    for exchange, symbol in [("SSE", "600519.SH"), ("SZSE", "300750.SZ")]:
        record = catalogue(exchange, symbol)
        records.append(record)
        print(f"{symbol}公开目录：返回{len(record['rows'])}条，完整={record['complete']}，状态={record['error'] or '已取得'}。", flush=True)
    save(OUT / "results/catalogues.json", records, True)
    rows = [row for record in records if record["complete"] for row in record["rows"]]
    pd.DataFrame(rows).to_parquet(OUT / "results/announcement_catalogue.parquet", index=False)
    # 本阶段只下载和提取原文，不从标题猜测实际资金金额。
    def fetch(row):
        key = row["symbol"] + "_" + sha256(row["source_url"].encode()).hexdigest()[:16]
        content, receipt = request(key, row["source_url"])
        if receipt["status"] != "HTTP_OK_NOT_YET_ADMITTED" or not content.startswith(b"%PDF"):
            return {**row, "key": key, "status": "NO_USABLE_PDF", "receipt": f"receipts/{key}.json"}
        import pypdfium2 as pdfium
        # PDFium在同一进程内并非线程安全：提取阶段由下方主线程完成。
        return {**row, "key": key, "status": "PDF_DOWNLOADED", "raw_path": receipt["raw_path"],
            "sha256": receipt["sha256"], "receipt": f"receipts/{key}.json"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        downloaded = list(pool.map(fetch, rows))
    import pypdfium2 as pdfium
    for index, row in enumerate(downloaded, 1):
        if row["status"] == "PDF_DOWNLOADED":
            doc = pdfium.PdfDocument(OUT / row["raw_path"])
            pages = []
            for page in doc:
                text = page.get_textpage()
                pages.append(text.get_text_range())
                text.close()
                page.close()
            doc.close()
            text_path = OUT / "results" / f"{row['key']}_pages.json"
            save(text_path, pages, True)
            row.update(status="PDF_TEXT_SAVED_NOT_YET_STRUCTURED", text_path=text_path.relative_to(OUT).as_posix(), pages=len(pages))
        if index % 10 == 0:
            print(f"回购原文已处理{index}/{len(downloaded)}份。", flush=True)
    save(OUT / "results/downloaded_documents.json", downloaded, True)
    result = {"at": now(), "study_id": STUDY,
        "status": "CATALOGUES_AND_ORIGINALS_COLLECTED_FACTS_PENDING" if all(row["complete"] for row in records) else "PARTIAL_PUBLIC_SOURCE_COLLECTION_FACTS_PENDING",
        "catalogues": [{k: row[k] for k in ["exchange", "symbol", "total", "unique_urls", "complete", "error"]} for row in records],
        "pdfs_saved": sum(row["status"] == "PDF_TEXT_SAVED_NOT_YET_STRUCTURED" for row in downloaded),
        "new_accounts": 0, "new_fits": 0, "new_independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print("公开来源探查完成，原文与目录已保存；尚未把计划或累计金额当成交易信号。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="成分公司回购公开原文来源探查")
    parser.add_argument("command", choices=["freeze", "run"])
    command = parser.parse_args().command
    if command == "freeze":
        freeze()
    else:
        run()
