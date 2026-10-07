"""采集新版国家数据的全国月度目录、全部可返回历史和官方发布原文。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "nbs_monthly"
API = "https://data.stats.gov.cn/dg/website/publicrelease/web/external/"
HOME = "https://data.stats.gov.cn/dg/website/page.html#/pc/national/monthData"
RELEASES = "https://www.stats.gov.cn/sj/zxfb/"
SCHEDULE = "https://www.stats.gov.cn/xw/tjxw/tzgg/202512/t20251224_1962137.html"
CN_TZ = timezone(timedelta(hours=8))
EXPORT_NAMES = {"indicator_id": "指标ID", "category": "类别", "catalogue_path": "目录", "name": "指标",
                "unit": "单位", "period_basis": "口径", "period": "数据月份", "area_code": "地区代码",
                "value": "数值", "value_text": "原始数值", "status": "状态", "source_cell_present": "是否返回单元格",
                "raw_path": "原始响应", "retrieved_at": "下载时间", "definition": "指标定义", "annotation": "注释",
                "catalogue_explain": "目录说明", "previous_month": "上一有值月份", "previous_value": "上一有值数值",
                "consecutive_previous": "是否连续月份", "previous_reported_value_difference": "连续月原始数值差",
                "category_latest_month": "该类最新月份", "is_older_than_category_latest": "早于该类最新月份",
                "first_value_month": "最早有值月份", "last_value_month": "最新有值月份", "value_count": "有值月数",
                "blank_within_and_after_history": "历史内及末尾空值数", "non_numeric_count": "非数值单元格数",
                "omitted_cells": "接口省略单元格数", "returned_start": "接口返回起点", "returned_end": "接口返回终点"}


def now() -> str:
    return datetime.now(CN_TZ).isoformat(timespec="seconds")


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Client:
    """保存公开请求的原始响应；同一快照可从已有响应继续。"""

    def __init__(self, run: Path):
        self.run = run
        self.local = threading.local()
        self.lock = threading.Lock()
        self.blocked_hosts: set[str] = set()
        self.last_started = 0.0
        self.browser_pool = None
        self.browser = None
        self.browser_page = None
        self.playwright = None

    def request(self, url: str, params=None, payload=None, binary=False):
        method = "POST" if payload is not None else "GET"
        signature = json.dumps([method, url, params, payload], sort_keys=True, ensure_ascii=False)
        key = sha(signature.encode("utf-8"))
        receipt_path = self.run / "receipts" / (key + ".json")
        if receipt_path.exists():
            record = read(receipt_path)
            raw = self.run / record["raw_path"] if record.get("raw_path") else None
            if record["status"] == "OK" and raw and raw.exists():
                content = gzip.decompress(raw.read_bytes())
                if sha(content) != record["sha256"]:
                    raise RuntimeError(f"已保存响应内容不一致：{raw}")
                return content, record
        host = urlparse(url).hostname or ""
        if host not in {"www.stats.gov.cn", "data.stats.gov.cn"}:
            raise ValueError(f"采集来源必须属于国家统计局：{url}")
        record = {"url": url, "method": method, "params": params, "payload": payload,
                  "requested_at": now(), "status": "STARTED", "attempts": []}
        with self.lock:
            if host in self.blocked_hosts:
                record.update(status="HOST_ACCESS_STOPPED", retrieved_at=now())
                dump(receipt_path, record)
                return b"", record
            delay = max(0.0, 0.12 - (time.monotonic() - self.last_started))
            if delay:
                time.sleep(delay)
            self.last_started = time.monotonic()
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": HOME})
        content = b""
        for attempt in range(1, 3):
            started = now()
            try:
                response = self.local.session.request(method, url, params=params, json=payload,
                                                      timeout=(10, 100))
                content = response.content
                record["attempts"].append({"number": attempt, "at": started,
                                           "http_status": response.status_code, "bytes": len(content)})
                record.update(http_status=response.status_code, final_url=response.url)
                if response.status_code in (403, 429):
                    with self.lock:
                        self.blocked_hosts.add(host)
                    record["status"] = "ACCESS_DENIED"
                    break
                if response.status_code >= 500 and attempt == 1:
                    time.sleep(1)
                    continue
                record["status"] = "OK" if response.status_code == 200 else "HTTP_FAILURE"
                break
            except requests.RequestException as exc:
                record["attempts"].append({"number": attempt, "at": started,
                                           "error": f"{type(exc).__name__}: {exc}"})
                record["status"] = "TRANSPORT_FAILURE"
                if attempt == 1:
                    time.sleep(1)
        record.update(retrieved_at=now(), bytes=len(content), sha256=sha(content))
        raw = self.run / "raw" / (key + (".bin.gz" if binary else ".json.gz"))
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(gzip.compress(content, compresslevel=5, mtime=0))
        record["raw_path"] = raw.relative_to(self.run).as_posix()
        record["receipt_path"] = receipt_path.relative_to(self.run).as_posix()
        dump(receipt_path, record)
        return content, record

    def json(self, endpoint: str, params=None, payload=None):
        content, receipt = self.request(API + endpoint, params=params, payload=payload)
        if receipt["status"] != "OK":
            raise RuntimeError(f"官方接口访问未成功：{endpoint}，{receipt['status']}")
        if content.lstrip().startswith(b"<"):
            # 官方站点有时返回要求启用JavaScript的网页，交由普通浏览器访问。
            with self.lock:
                if self.browser_pool is None:
                    self.browser_pool = ThreadPoolExecutor(max_workers=1)
            content, receipt = self.browser_pool.submit(self.browser_request, API + endpoint,
                                                       params, payload, receipt).result()
        data = json.loads(content)
        if not isinstance(data, dict) or data.get("success") is not True:
            raise RuntimeError(f"官方接口没有成功数据：{endpoint}，{str(data)[:160]}")
        return data["data"], receipt

    def browser_request(self, url, params, payload, previous):
        if self.browser_page is None:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(channel="msedge", headless=True)
            self.browser_page = self.browser.new_page()
            self.browser_page.goto(HOME, wait_until="domcontentloaded", timeout=60_000)
            print("国家数据已切换到普通浏览器读取JavaScript网站响应。", flush=True)
        request_url = requests.Request("GET", url, params=params).prepare().url
        response = self.browser_page.evaluate(
            "async ({url,payload}) => {const response=await fetch(url,payload===null?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});return {status:response.status,url:response.url,body:await response.text()};}",
            {"url": request_url, "payload": payload})
        content = response["body"].encode("utf-8")
        if response["status"] != 200 or not content.lstrip().startswith(b"{"):
            raise RuntimeError("普通浏览器仍未收到官方JSON数据，需保留访问缺口。")
        raw = self.run / "raw" / (sha(content) + ".browser.json.gz")
        raw.write_bytes(gzip.compress(content, compresslevel=5, mtime=0))
        record = {**previous, "status": "OK", "transport": "正常Microsoft Edge浏览器",
                  "http_status": response["status"], "final_url": response["url"],
                  "previous_raw_path": previous["raw_path"], "raw_path": raw.relative_to(self.run).as_posix(),
                  "retrieved_at": now(), "bytes": len(content), "sha256": sha(content)}
        record["attempts"] = previous["attempts"] + [{"at": now(), "transport": "正常浏览器", "http_status": response["status"]}]
        dump(self.run / previous["receipt_path"], record)
        return content, record

    def close(self):
        if self.browser_pool:
            def finish():
                if self.browser:
                    self.browser.close()
                if self.playwright:
                    self.playwright.stop()
            self.browser_pool.submit(finish).result()
            self.browser_pool.shutdown()


def metadata(client: Client):
    path = client.run / "metadata.json"
    if path.exists():
        return read(path)
    nodes, queue, seen, leaves = [], [(None, [], "")], set(), []
    while queue:
        jobs, queue = queue, []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(client.json, "new/queryIndexTreeAsync",
                                      {"code": "1", **({"pid": pid} if pid else {})}): (pid, parent_path, category)
                       for pid, parent_path, category in jobs}
            for future in as_completed(futures):
                pid, parent_path, parent_category = futures[future]
                children, receipt = future.result()
                for child in children:
                    cid = child["_id"]
                    if cid in seen:
                        continue
                    seen.add(cid)
                    node_path = parent_path + [compact(child["name"])]
                    category = node_path[1] if len(node_path) >= 2 else parent_category
                    node = {**child, "path": node_path, "category": category,
                            "metadata_source": receipt["raw_path"]}
                    nodes.append(node)
                    if child["isLeaf"]:
                        leaves.append(node)
                    else:
                        queue.append((cid, node_path, category))
        print(f"全国月度目录：已发现 {len(nodes)} 个目录节点、{len(leaves)} 个末级数据表。", flush=True)
    if len([n for n in nodes if n.get("treeinfo_level") == 2]) != 1:
        raise RuntimeError("全国月度根目录不唯一。")
    root_id = next(n["_id"] for n in nodes if n.get("treeinfo_level") == 2)
    indicators, table_meta = [], []
    # 新版网站的全国月度检索接口可一次返回根目录下的完整指标定义。
    result, receipt = client.json("new/queryIndicatorsByCid", {"cid": "", "rootId": root_id, "dt": "", "name": ""})
    if len(result["list"]) != result["total"]:
        raise RuntimeError(f"全国指标检索被截断：{len(result['list'])}/{result['total']}")
    leaf_map = {n["_id"]: n for n in leaves}
    returned_catalogues = {i["catalogid"] for i in result["list"]}
    if returned_catalogues != set(leaf_map):
        raise RuntimeError("全国检索返回的目录与逐层遍历末级目录不一致。")
    basis = {"1": "本期", "3": "上年同期", "21": "本期累计", "24": "上年同期累计",
             "11": "同比增减%", "12": "累计同比增减%", "13": "环比增减%"}
    for indicator in result["list"]:
        node = leaf_map[indicator["catalogid"]]
        display = indicator["i_showname"].replace("（", "(").replace("）", ")")
        unit = re.search(r"\(([^()]*)\)\s*$", display)
        indicators.append({**indicator, "category": node["category"],
                           "catalogue_path": " / ".join(node["path"]),
                           "catalogue_explain": node.get("explain"),
                           "du_name": unit.group(1).strip() if unit else None,
                           "dp_name": basis.get(str(indicator["dp"]), "原始口径代码:" + str(indicator["dp"])),
                           "metadata_source": receipt["raw_path"]})
    counts = pd.Series([i["catalogid"] for i in indicators]).value_counts()
    table_meta = [{"catalogue_id": n["_id"], "count": int(counts.get(n["_id"], 0)),
                   "declared_start": n.get("sdate"), "declared_end": n.get("edate")} for n in leaves]
    ids = [x["_id"] for x in indicators]
    if len(ids) != len(set(ids)):
        raise RuntimeError("全国月度指标标识存在重复，需保留目录关联后处理。")
    result = {"root_id": root_id, "nodes": nodes, "leaves": leaves, "indicators": indicators,
              "tables": table_meta, "retrieved_at": now(), "source": HOME}
    dump(path, result)
    pd.DataFrame([{"指标ID": i["_id"], "类别": i["category"], "目录": i["catalogue_path"],
                   "指标": i["i_showname"], "单位": i.get("du_name"), "口径": i.get("dp_name"),
                   "指标定义": i.get("i_mark"), "注释": i.get("i_annotation"),
                   "目录说明": i.get("catalogue_explain"), "原始定义文件": i["metadata_source"]}
                  for i in indicators]).to_csv(client.run / "指标字典.csv", index=False, encoding="utf-8-sig")
    print(f"目录采集完成：{len(leaves)} 张末级数据表，{len(indicators)} 个指标。", flush=True)
    return result


def month_range(start, end):
    return [x.strftime("%Y%m") for x in pd.period_range(start=start[:4] + "-" + start[4:6],
                                                      end=end[:4] + "-" + end[4:6], freq="M")]


def history(client: Client, start: str, end: str):
    meta = metadata(client)
    groups = {n["_id"]: [i for i in meta["indicators"] if i["catalogid"] == n["_id"]] for n in meta["leaves"]}
    coverage, failed = [], []
    panels = client.run / "panels"
    panels.mkdir(exist_ok=True)

    def one(node):
        cid, expected = node["_id"], groups[node["_id"]]
        target, ledger = panels / (cid + ".parquet"), panels / (cid + ".coverage.json")
        if target.exists() and ledger.exists():
            return read(ledger)
        items, sources = [], []
        for offset in range(0, len(expected), 50):
            batch = expected[offset:offset + 50]
            payload = {"cid": cid, "rootId": meta["root_id"], "indicatorIds": [x["_id"] for x in batch],
                       "daCatalogId": "", "das": [{"text": "全国", "value": "000000000000"}],
                       "showType": 1, "dts": [start + "MM-" + end + "MM"]}
            data, receipt = client.json("stream/esData", payload=payload)
            if not data:
                raise RuntimeError(f"历史查询返回空时轴：{node['name']}")
            periods = [str(row["code"])[:6] for row in data]
            if len(periods) != len(set(periods)) or set(periods) != set(month_range(min(periods), max(periods))):
                raise RuntimeError(f"历史时轴不连续或重复：{node['name']}")
            if max(periods) != end:
                raise RuntimeError(f"返回历史时轴未到查询终点：{node['name']}，{max(periods)} != {end}")
            by_id = {i["_id"]: i for i in batch}
            for row in data:
                returned = {v["_id"]: v for v in row["values"]}
                if not set(returned).issubset(by_id):
                    raise RuntimeError(f"历史响应包含未请求的指标：{node['name']}，{row['code']}")
                for indicator_id in by_id:
                    # 旧口径表在有效年份之外可返回空values数组；与显式空字符串分别记录。
                    present = indicator_id in returned
                    value = returned.get(indicator_id, {"_id": indicator_id, "value": "", "da": "000000000000"})
                    if value.get("da") != "000000000000":
                        raise RuntimeError("历史响应包含非全国地区。")
                    text = str(value["value"]) if value.get("value") is not None else ""
                    number = pd.to_numeric(text.replace(",", ""), errors="coerce")
                    if text.strip() and pd.isna(number):
                        status = "非数值原文"
                    else:
                        status = "缺失" if pd.isna(number) else "有值"
                    items.append({"indicator_id": value["_id"], "period": str(row["code"])[:6],
                                  "area_code": value["da"], "value_text": text, "value": float(number),
                                  "status": status, "source_cell_present": present, "raw_path": receipt["raw_path"],
                                  "retrieved_at": receipt["retrieved_at"]})
            sources.append({"raw_path": receipt["raw_path"], "requested_start": start, "requested_end": end,
                            "returned_start": min(periods), "returned_end": max(periods),
                            "returned_months": len(periods), "indicators": len(batch)})
        frame = pd.DataFrame(items)
        if frame.empty:
            frame = pd.DataFrame(columns=["indicator_id", "period", "area_code", "value_text", "value", "status", "source_cell_present", "raw_path", "retrieved_at"])
        frame = frame.sort_values(["indicator_id", "period"])
        rows, retained = [], []
        for indicator in expected:
            subset = frame.loc[frame.indicator_id.eq(indicator["_id"])]
            valid = subset.loc[subset.status.eq("有值")]
            first = str(valid.period.min()) if len(valid) else None
            last = str(valid.period.max()) if len(valid) else None
            within = subset.loc[subset.period.ge(first) | subset.status.ne("缺失")] if first else subset
            retained.append(within)
            rows.append({"indicator_id": indicator["_id"], "category": node["category"],
                         "catalogue_path": indicator["catalogue_path"], "name": indicator["i_showname"],
                         "unit": indicator.get("du_name"), "period_basis": indicator.get("dp_name"),
                         "first_value_month": first, "last_value_month": last, "value_count": len(valid),
                         "blank_within_and_after_history": int(within.status.eq("缺失").sum()),
                         "non_numeric_count": int(subset.status.eq("非数值原文").sum()),
                         "omitted_cells": int((~subset.source_cell_present).sum()),
                         "returned_start": str(subset.period.min()) if len(subset) else None,
                         "returned_end": str(subset.period.max()) if len(subset) else None,
                         "status": "已采集" if len(valid) else "官方查询全空"})
        (pd.concat(retained, ignore_index=True) if retained else frame).to_parquet(target, index=False)
        result = {"catalogue_id": cid, "rows": rows, "sources": sources}
        dump(ledger, result)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(one, n): n for n in meta["leaves"]}
        for index, future in enumerate(as_completed(futures), 1):
            node = futures[future]
            try:
                coverage.extend(future.result()["rows"])
            except Exception as exc:
                failed.append({"catalogue_id": node["_id"], "name": node["name"], "error": str(exc)})
                if len(failed) <= 3:
                    print(f"待处理表：{node['name']}；原因：{exc}。", flush=True)
            if index % 10 == 0 or index == len(meta["leaves"]):
                print(f"历史采集 {index}/{len(meta['leaves'])} 张表，有数值指标 {sum(r['value_count'] > 0 for r in coverage)} 项，失败 {len(failed)} 张。", flush=True)
    dump(client.run / "history_status.json", {"at": now(), "start": start, "end": end, "coverage": coverage, "failed": failed})
    pd.DataFrame(coverage).rename(columns=EXPORT_NAMES).to_csv(client.run / "历史覆盖.csv", index=False, encoding="utf-8-sig")


def parse_listing(content: bytes, base: str):
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    found = {}
    for li in soup.find_all("li"):
        link = li.find("a", href=True)
        if link is None:
            continue
        date = re.search(r"(20\d{2})[-年](\d{2})[-月](\d{2})", li.get_text(" ", strip=True))
        url = urljoin(base, link["href"])
        if not date or not re.search(r"/t\d{8}_\d+\.html$", url):
            continue
        title = compact(link.get("title") or link.get_text(" ", strip=True))
        found[url] = {"url": url, "title": title,
                      "catalogue_date": "-".join(date.groups())}
    return list(found.values())


def release_kind(title):
    if "70个大中城市" in title:
        return "全国发布_城市房价附表"
    if any(x in title for x in ["上旬", "中旬", "下旬"]):
        return "旬度补充"
    if re.search(r"[一二三四1234]季度|前三季度|上半年|全年|统计公报|平均工资|粮食产量|创新指数|新动能|科技经费", title):
        return "季度年度补充"
    return "月度或累计发布"


def expand_table(table):
    occupied, origins = {}, []
    for ri, row in enumerate(table.find_all("tr")):
        ci = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (ri, ci) in occupied:
                ci += 1
            text = compact(cell.get_text(" ", strip=True))
            rowspan, colspan = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            origins.append({"row": ri, "column": ci, "rowspan": rowspan, "colspan": colspan,
                            "text": text, "tag": cell.name})
            for r in range(ri, ri + rowspan):
                for c in range(ci, ci + colspan):
                    occupied[r, c] = text
            ci += colspan
    height = max((r for r, _ in occupied), default=-1) + 1
    width = max((c for _, c in occupied), default=-1) + 1
    return [[occupied.get((r, c), "") for c in range(width)] for r in range(height)], origins


def parse_release(content: bytes, item, receipt, run: Path):
    soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    body = (soup.select_one(".TRS_Editor") or soup.select_one(".trs_editor_view")
            or soup.select_one(".TRS_PreAppend") or soup.select_one(".xxgk_cont"))
    if body is None:
        raise RuntimeError(f"官方发布正文未定位：{item['url']}")
    meta = soup.find("meta", attrs={"name": "PubDate"})
    published_at = meta.get("content") if meta else None
    if not published_at:
        date_node = soup.select_one(".detail-title-des h2 p")
        date_text = compact(date_node.get_text(" ", strip=True)) if date_node else ""
        if re.fullmatch(r"\d{4}[/\-]\d{2}[/\-]\d{2}\s+\d{2}:\d{2}(?::\d{2})?", date_text):
            published_at = date_text
    formed_on = None
    for field in soup.select(".content_top_box div"):
        if field.get_text(strip=True).startswith("成文日期"):
            match = re.search(r"(\d{4})年(\d{2})月(\d{2})日", field.get_text(strip=True))
            if match:
                formed_on = "-".join(match.groups())
                break
    title_meta = soup.find("meta", attrs={"name": "ArticleTitle"})
    title_node = soup.select_one(".xxgkNbXqTitle") or soup.select_one(".detail-title h1")
    title = (compact(title_meta["content"]) if title_meta else
             compact(title_node.get_text(" ", strip=True)) if title_node else item["title"])
    filename = re.search(r"(t\d{8}_\d+)\.html", item["url"]).group(1)
    text_path = run / "release_text" / (filename + ".txt")
    text_path.parent.mkdir(exist_ok=True)
    text = body.get_text("\n", strip=True)
    text_path.write_text(text + "\n", encoding="utf-8")
    tables, fingerprints = [], set()
    for table in body.find_all("table"):
        grid, origins = expand_table(table)
        signature = sha(json.dumps(grid, ensure_ascii=False).encode("utf-8"))
        if signature in fingerprints:
            continue
        fingerprints.add(signature)
        tables.append({"table_index": len(tables), "grid": grid, "cells": origins})
    table_path = run / "release_tables" / (filename + ".json")
    dump(table_path, tables)
    assets = []
    for tag, attr in [(a, "href") for a in body.select("a[href]")] + [(a, "src") for a in body.select("img[src]")]:
        url = urljoin(item["url"], tag[attr])
        extension = Path(urlparse(url).path).suffix.lower()
        if urlparse(url).hostname == "www.stats.gov.cn" and extension in {".pdf", ".xls", ".xlsx", ".csv", ".zip", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".gif"}:
            assets.append({"url": url, "extension": extension, "label": tag.get("alt") or tag.get_text(strip=True)})
    return {**item, "title": title, "published_at": published_at, "formed_on": formed_on, "parser_version": 2,
            "release_kind": release_kind(title), "raw_path": receipt["raw_path"],
            "retrieved_at": receipt["retrieved_at"], "sha256": receipt["sha256"],
            "text_path": text_path.relative_to(run).as_posix(), "table_path": table_path.relative_to(run).as_posix(),
            "table_count": len(tables), "text_characters": len(text), "assets": assets}


def releases(client: Client):
    content, receipt = client.request(RELEASES)
    if receipt["status"] != "OK":
        raise RuntimeError("国家统计局全国发布目录访问失败。")
    match = re.search(r'createPageHTML\((\d+),\s*0,\s*"index",\s*"html"', content.decode("utf-8"))
    if not match:
        raise RuntimeError("全国发布目录未提供可解析的分页总数。")
    page_count = int(match.group(1))
    items = {x["url"]: x for x in parse_listing(content, RELEASES)}
    page_failures = []

    def page(index):
        url = RELEASES + f"index_{index}.html"
        data, rec = client.request(url)
        if rec["status"] != "OK":
            return [], {"page": index, "status": rec["status"]}
        return parse_listing(data, url), None

    with ThreadPoolExecutor(max_workers=2) as executor:
        for index, (found, error) in enumerate(executor.map(page, range(1, page_count)), 1):
            items.update({x["url"]: x for x in found})
            if error:
                page_failures.append(error)
            if index % 10 == 0:
                print(f"发布稿目录：{index + 1}/{page_count} 页，{len(items)} 篇。", flush=True)
    # 目录较早部分被官网移出分页；补入本项目此前已定位的官方原页，单列覆盖身份。
    supplemental = set()
    pattern = re.compile(r"https://www\.stats\.gov\.cn/(?:sj/zxfb|xxgk/sjfb/zxfb2020)/\d{6}/t\d{8}_\d+\.html")
    for directory in (ROOT / "research", ROOT / "scripts", ROOT / "docs"):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md", ".json"} and path.stat().st_size < 2_000_000:
                supplemental.update(pattern.findall(path.read_text(encoding="utf-8", errors="replace")))
    for url in supplemental - items.keys():
        items[url] = {"url": url, "title": "待从官方原页读取", "catalogue_date": None, "discovery": "既有官方原页索引补充"}
    dump(client.run / "release_catalogue.json", {"pages": page_count, "page_failures": page_failures,
                                                "items": sorted(items.values(), key=lambda x: x["url"])})
    results, failures = [], []

    def one(item):
        key = sha(item["url"].encode("utf-8"))
        parsed_path = client.run / "parsed_releases" / (key + ".json")
        if parsed_path.exists():
            cached = read(parsed_path)
            if cached.get("parser_version") == 2:
                return cached
        data, rec = client.request(item["url"])
        if rec["status"] != "OK":
            raise RuntimeError(rec["status"])
        parsed = parse_release(data, item, rec, client.run)
        dump(parsed_path, parsed)
        return parsed

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(one, x): x for x in items.values()}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"url": futures[future]["url"], "error": str(exc)})
            if index % 25 == 0:
                print(f"发布原文：已处理 {index}/{len(items)} 篇，成功 {len(results)}，失败 {len(failures)}。", flush=True)
    assets = {a["url"]: a for item in results for a in item["assets"]}
    asset_results = []

    def asset(item):
        data, rec = client.request(item["url"], binary=True)
        if rec["status"] == "OK":
            target = client.run / "attachments" / (sha(item["url"].encode())[:16] + item["extension"])
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(data)
            return {**item, "status": "已保存", "path": target.relative_to(client.run).as_posix(), "sha256": sha(data),
                    "http_status": rec.get("http_status"), "receipt_path": rec["receipt_path"]}
        return {**item, "status": rec["status"], "http_status": rec.get("http_status"), "receipt_path": rec["receipt_path"]}

    with ThreadPoolExecutor(max_workers=6) as executor:
        for index, result in enumerate(executor.map(asset, assets.values()), 1):
            asset_results.append(result)
            if index % 25 == 0:
                print(f"附表与数据图原件：{index}/{len(assets)}。", flush=True)
    schedule, schedule_receipt = client.request(SCHEDULE)
    if schedule_receipt["status"] == "OK":
        (client.run / "官方2026发布日程.html").write_bytes(schedule)
    dump(client.run / "releases_status.json", {"at": now(), "directory_pages": page_count,
                                               "page_failures": page_failures, "releases": results,
                                               "failed": failures, "assets": asset_results})
    pd.DataFrame(asset_results).rename(columns={"url": "官方附件链接", "extension": "文件类型", "label": "原页标签",
        "status": "保存状态", "path": "文件位置", "sha256": "原件SHA256", "http_status": "HTTP状态", "receipt_path": "下载回执"}).to_csv(
        client.run / "附件来源索引.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in item.items() if k != "assets"} for item in results]).rename(columns={
        "url": "官方链接", "title": "标题", "catalogue_date": "目录日期", "published_at": "原页公布时间", "formed_on": "原页成文日期",
        "release_kind": "频率类型", "raw_path": "原始响应", "retrieved_at": "下载时间", "sha256": "原件SHA256",
        "text_path": "正文位置", "table_path": "附表位置", "table_count": "附表数量", "text_characters": "正文字符数",
        "discovery": "发现方式", "parser_version": "解析版本"}).to_csv(
        client.run / "官方发布索引.csv", index=False, encoding="utf-8-sig")


def finalize(run: Path):
    meta = read(run / "metadata.json")
    hist = read(run / "history_status.json")
    fingerprint = sha((run / "metadata.json").read_bytes() + (run / "history_status.json").read_bytes() + Path(__file__).read_bytes())
    build_receipt = run / "data_build_receipt.json"
    reuse_core = build_receipt.exists() and read(build_receipt).get("fingerprint") == fingerprint and (run / "全国月度数据.parquet").exists() and (run / "国家统计局月度.sqlite").exists()
    if reuse_core:
        frame = pd.read_parquet(run / "全国月度数据.parquet")
    else:
        frames = [pd.read_parquet(run / "panels" / (n["_id"] + ".parquet")) for n in meta["leaves"]
                  if (run / "panels" / (n["_id"] + ".parquet")).exists()]
        frame = pd.concat(frames, ignore_index=True)
    if "source_cell_present" in frame:
        frame["source_cell_present"] = frame.source_cell_present.fillna(True).astype(bool)
    if frame.duplicated(["indicator_id", "period", "area_code"]).any():
        raise RuntimeError("合并历史存在重复的指标、月份、地区。")
    if not reuse_core:
        frame.to_parquet(run / "全国月度数据.parquet", index=False)
        frame.to_csv(run / "全国月度数据.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    connection = sqlite3.connect(run / "国家统计局月度.sqlite")
    dictionary = pd.DataFrame([{"indicator_id": i["_id"], "category": i["category"], "catalogue_path": i["catalogue_path"],
                               "name": i["i_showname"], "unit": i.get("du_name"), "period_basis": i.get("dp_name"),
                               "definition": i.get("i_mark"), "annotation": i.get("i_annotation"),
                               "catalogue_explain": i.get("catalogue_explain"), "metadata_json": json.dumps(i, ensure_ascii=False)}
                              for i in meta["indicators"]])
    if not reuse_core:
        dictionary.to_sql("indicators", connection, if_exists="replace", index=False)
        frame.to_sql("observations", connection, if_exists="replace", index=False, chunksize=2000)
        pd.DataFrame(hist["coverage"]).to_sql("coverage", connection, if_exists="replace", index=False)
    pd.DataFrame(hist["coverage"]).rename(columns=EXPORT_NAMES).to_csv(run / "历史覆盖.csv", index=False, encoding="utf-8-sig")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS observation_key ON observations(indicator_id,period,area_code)")
    connection.execute("CREATE INDEX IF NOT EXISTS indicator_category ON indicators(category)")
    release = read(run / "releases_status.json") if (run / "releases_status.json").exists() else None
    if release:
        records = [{k: v for k, v in r.items() if k != "assets"} for r in release["releases"]]
        pd.DataFrame(records).to_sql("releases", connection, if_exists="replace", index=False)
        cells = []
        for item in release["releases"]:
            for table in read(run / item["table_path"]):
                for cell in table["cells"]:
                    value = pd.to_numeric(cell["text"].replace(",", ""), errors="coerce")
                    cells.append({"source_url": item["url"], "published_at": item["published_at"],
                                  "table_index": table["table_index"], "row_index": cell["row"],
                                  "column_index": cell["column"], "rowspan": cell["rowspan"],
                                  "colspan": cell["colspan"], "cell_text": cell["text"],
                                  "numeric_cell_value": float(value), "table_path": item["table_path"]})
        cell_frame = pd.DataFrame(cells)
        if len(cell_frame):
            cell_frame.to_sql("release_cells", connection, if_exists="replace", index=False, chunksize=2000)
            cell_frame.to_parquet(run / "发布附表单元格.parquet", index=False)
    connection.execute("DROP VIEW IF EXISTS monthly_with_definitions")
    connection.execute("CREATE VIEW monthly_with_definitions AS SELECT o.*,i.name,i.category,i.catalogue_path,i.unit,i.period_basis,i.annotation FROM observations o LEFT JOIN indicators i USING(indicator_id)")
    connection.commit()
    check = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    if check != "ok":
        raise RuntimeError("月度数据库结构检查未通过。")
    dump(build_receipt, {"fingerprint": fingerprint, "at": now(), "observations": len(frame), "numeric_observations": int(frame.status.eq("有值").sum())})
    panel = frame.merge(dictionary.drop(columns="metadata_json"), on="indicator_id", how="left", validate="many_to_one")
    valid = panel.loc[panel.status.eq("有值")].sort_values(["indicator_id", "period"])
    latest = valid.groupby("indicator_id", sort=False).tail(1).copy()
    previous = valid.groupby("indicator_id", sort=False)[["period", "value"]].shift(1)
    valid["previous_month"] = previous.period
    valid["previous_value"] = previous.value
    valid["consecutive_previous"] = (pd.PeriodIndex(valid.period, freq="M").asi8 - pd.PeriodIndex(valid.previous_month.fillna(valid.period), freq="M").asi8) == 1
    valid["previous_reported_value_difference"] = (valid.value - valid.previous_value).where(valid.consecutive_previous)
    latest = valid.groupby("indicator_id", sort=False).tail(1)
    latest = latest.copy()
    latest["category_latest_month"] = latest.groupby("category")["period"].transform("max")
    latest["is_older_than_category_latest"] = latest.period.lt(latest.category_latest_month)
    latest.rename(columns=EXPORT_NAMES).to_csv(run / "最新指标快照.csv", index=False, encoding="utf-8-sig")
    valid.to_parquet(run / "趋势查询数据.parquet", index=False)
    sectors = latest.loc[latest.category.eq("工业") & latest.name.str.contains("增加值") & latest.name.str.contains("同比|增长速度")]
    sectors.rename(columns=EXPORT_NAMES).to_csv(run / "工业行业最新观察.csv", index=False, encoding="utf-8-sig")
    liquidity = latest.loc[latest.category.eq("金融")]
    liquidity.rename(columns=EXPORT_NAMES).to_csv(run / "货币供应量最新观察.csv", index=False, encoding="utf-8-sig")
    category_summary = valid.groupby("category").agg(indicators=("indicator_id", "nunique"), observations=("value", "size"),
                                                     first_month=("period", "min"), last_month=("period", "max")).reset_index()
    category_summary.rename(columns={"category": "类别", "indicators": "有数值指标数", "observations": "数值记录数",
                                     "first_month": "最早月份", "last_month": "最新月份"}).to_csv(run / "分类覆盖.csv", index=False, encoding="utf-8-sig")
    summary = {"at": now(), "scope": "全国月度数据，全部官方查询可返回历史", "snapshot": run.name,
               "catalogue_nodes": len(meta["nodes"]), "leaf_tables": len(meta["leaves"]),
               "registered_indicators": len(meta["indicators"]), "indicators_with_values": valid.indicator_id.nunique(),
               "numeric_observations": len(valid), "retained_missing_observations": int(frame.status.eq("缺失").sum()),
               "first_value_month": valid.period.min(), "last_value_month": valid.period.max(),
               "history_failed": hist["failed"], "categories": category_summary.to_dict("records"),
               "release_count": len(release["releases"]) if release else None,
               "release_failed": release["failed"] if release else None,
               "attachments": sum(a["status"] == "已保存" for a in release["assets"]) if release else None,
               "attachment_links": len(release["assets"]) if release else None,
               "attachment_failed": [a for a in release["assets"] if a["status"] != "已保存"] if release else None,
               "first_vintage_verified": False, "database_history_publication_times": "未提供，保留未知",
               "new_strategy_tests": 0, "sqlite_integrity": check}
    summary["catalogued_indicators_without_numeric_values"] = len(meta["indicators"]) - int(valid.indicator_id.nunique())
    summary["release_table_cells"] = len(cells) if release else 0
    unavailable = [a for a in (summary["attachment_failed"] or []) if a.get("http_status") in {404, 410}]
    pending_assets = [a for a in (summary["attachment_failed"] or []) if a.get("http_status") not in {404, 410}]
    summary["official_unavailable_attachments"] = len(unavailable)
    if release and summary["attachment_failed"]:
        pd.DataFrame(summary["attachment_failed"]).rename(columns={"url": "官方附件链接", "extension": "文件类型",
            "label": "原页标签", "status": "返回状态", "http_status": "HTTP状态", "receipt_path": "下载回执"}).to_csv(
            run / "官方附件缺口.csv", index=False, encoding="utf-8-sig")
    completed = not hist["failed"] and release and not release["failed"] and not release["page_failures"] and not pending_assets
    summary["status"] = ("采集完成_已列示官方失效链接" if unavailable else "采集完成_按明确覆盖范围") if completed else "部分采集_存在明确缺口"
    dump(run / "result.json", summary)
    dump(BASE / "LATEST.json", {"run_id": run.name, "path": str(run), "at": now(), "status": summary["status"]})
    report = ["# 国家统计局全国月度数据库", "", f"快照：{run.name}。范围：全国，采集新版国家数据月度目录和全部可返回历史。", "",
              f"目录 {len(meta['leaves'])} 张表，注册 {len(meta['indicators'])} 个指标条目；有数值 {valid.indicator_id.nunique()} 项，共 {len(valid):,} 条数值记录。指标条目包括行业、金额、同比、累计、比较基数及不同年份口径，不代表这么多个独立经济变量。",
              f"实际数值历史 {valid.period.min()}—{valid.period.max()}，每个指标起止月份见历史覆盖.csv。", "",
              "| 类别 | 有数值指标 | 数值记录 | 最早月份 | 最新月份 |", "|---|---:|---:|---|---|"]
    report.extend(f"| {r.category} | {r.indicators} | {r.observations:,} | {r.first_month} | {r.last_month} |" for r in category_summary.itertuples())
    report += ["", "文件入口：国家统计局月度.sqlite 可直接按指标查询；全国月度数据.parquet 是完整数据；指标字典.csv 保存单位、口径及官方定义；最新指标快照.csv 用于查看各项最新读数；工业行业最新观察.csv、货币供应量最新观察.csv 供行业与货币观察。", "",
               "数据库中的历史数值是本次下载时官方返回的版本。历史发布日期、首次公布版本没有由该接口提供，不能将当前历史修订值当成当年已知值。发布原文的实际公布时间另外保存在官方发布索引.csv；部分旧信息公开页仅给出成文日期，单列保存，不替代未知的具体公布时间；URL路径中的202302迁移日期不替代原页公布日期。", "",
               "同比、环比、累计与当月都保留原始口径，不由累计增速反推当月增速，不填补未发布月份。趋势查询文件中的相邻读数差只表示原指标数值之差，只有连续月才计算；它不是统一意义的环比增长率。PMI指数、价格基期指数、累计增速与货币存量各按自己的定义读取。", "",
               "金融目录只包含官方平台提供的货币供应量。货币存量和增长速度不能完整代表银行间资金价格、社融结构或股票资金流；本次没有扩大采集机构。M1等口径变更须结合官方定义与各期原文观察，不直接当成连续同口径因果证据。", ""]
    if release:
        report += [f"已保存全国发布原文 {len(release['releases'])} 篇，以及附件和数据图原件 {summary['attachments']} 份；原稿列出附件链接 {len(release['assets'])} 个，其中 {len(unavailable)} 个官方链接返回404或410，见官方附件缺口.csv。当前官网分页历史覆盖有限；额外旧稿来自已定位的官方原页索引，不能声称历史每一次首次发布稿全部齐备。季度年度与旬度稿作为官网同一发布目录的补充保留，未改造成月度指标。全国房价公告含城市附表，原稿完整保存；未采集分省或城市数据库。", ""]
    report += [f"历史表失败：{len(hist['failed'])}；发布稿失败：{len(release['failed']) if release else '未运行'}；附件失败：{len(summary['attachment_failed']) if release else '未运行'}。详见result.json及对应状态文件。", "",
               "再次更新会创建独立快照，原始响应和旧值保留；中断后可用同一个run-id继续。公开目录、定义、时轴、数值、原文、附件均有原始响应可追溯。"]
    (run / "阅读说明.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"categories", "release_failed", "attachment_failed"}}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="采集国家统计局全国月度数据；仅做数据与经济观察。")
    parser.add_argument("--phase", choices=["metadata", "history", "releases", "finalize", "all"], default="all")
    parser.add_argument("--as-of", default=datetime.now(CN_TZ).strftime("%Y-%m-%d"), help="用户所在时区的采集业务日期")
    parser.add_argument("--run-id", help="独立快照名称；同一名称支持中断后继续")
    parser.add_argument("--start", default="190001", help="最早请求月，早于接口支持范围的空值不会伪造数据")
    args = parser.parse_args()
    date = datetime.strptime(args.as_of, "%Y-%m-%d")
    end = (date.replace(day=1) - timedelta(days=1)).strftime("%Y%m")
    run_id = args.run_id or date.strftime("%Y%m%d") + "_" + datetime.now(CN_TZ).strftime("%H%M%S")
    if not re.fullmatch(r"[0-9A-Za-z_-]+", run_id):
        parser.error("快照名称只能包含数字、字母、下划线和连接号。")
    run = BASE / "runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    scope = {"as_of": args.as_of, "scope": "全国月度", "start_request": args.start, "end_request": end,
             "source": HOME, "monthly_catalogue_code": "1", "updated_at": now(),
             "user_authorization": "全部可获取历史，先只做全国；本次只做官方数据采集与经济观察"}
    if (run / "scope.json").exists():
        old = read(run / "scope.json")
        if any(old.get(k) != scope[k] for k in ["as_of", "start_request", "end_request", "scope"]):
            raise RuntimeError("已有快照的日期或范围不同，请建立新的run-id。")
    else:
        dump(run / "scope.json", scope)
    client = Client(run)
    try:
        if args.phase in {"metadata", "all"}:
            metadata(client)
        if args.phase in {"history", "all"}:
            history(client, args.start, end)
        if args.phase in {"releases", "all"}:
            releases(client)
        if args.phase in {"finalize", "all"}:
            finalize(run)
    finally:
        client.close()


if __name__ == "__main__":
    main()
