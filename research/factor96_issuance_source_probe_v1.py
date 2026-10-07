"""固定小范围核对发行目录接口和自由流通量口径，不读取策略收益。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_issuance_source_probe_v1"
QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
FREE_FLOAT_RULE = "https://oss-ch.csindex.com.cn/contract/cms_add/20241122145338-《中证指数有限公司股票指数自由流通量规则》.pdf"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}


def now():
    return datetime.now().astimezone().isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    assert not (OUT / "protocol.json").exists(), "来源探查已存在，不能覆盖或无条件重试"
    probes = [("rights_title_2021", "配股", ""), ("rights_category_2021", "", "category_pg_szsh"),
              ("placements_category_2021", "", "category_zf_szsh"),
              ("union_category_2021", "", "category_pg_szsh;category_zf_szsh")]
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_FACTOR96_ISSUANCE_SOURCE_PROBE_V1",
         "purpose": "验证目录类别过滤与数量，不生成事件强度或策略收益；来源检查后另冻完整历史范围。",
         "query_window": ["2021-01-01", "2021-12-31"], "query_probes": probes,
         "requests": "主页、自由流通规则和四个目录首分页各一次；超时保留失败，403或429停止，不自动换源绕过。",
         "free_float_rule": FREE_FLOAT_RULE, "methodology_version": "2024-11 V1.1，只核对概念，不后填2015年历史方法",
         "new_accounts": 0, "new_returns": 0, "orders_authorized": False})
    (OUT / "code").mkdir(exist_ok=True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    requests_list = [("homepage", "https://www.cninfo.com.cn/new/index", None), ("free_float_rule", FREE_FLOAT_RULE, None)]
    for key, keyword, category in probes:
        payload = {"pageNum": "1", "pageSize": "30", "column": "szse", "tabName": "fulltext", "plate": "",
                   "stock": "", "searchkey": keyword, "secid": "", "category": category, "trade": "",
                   "seDate": "2021-01-01~2021-12-31", "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
        requests_list.append((key, QUERY, payload))
    summaries = []
    stopped = False
    for key, url, payload in requests_list:
        record = {"key": key, "requested_at": now(), "url": url, "payload": payload}
        if stopped:
            record["status"] = "NOT_REQUESTED_AFTER_SOURCE_LIMIT"
        else:
            try:
                response = requests.get(url, headers=HEADERS, timeout=(10, 25)) if payload is None else requests.post(url, data=payload, headers=HEADERS, timeout=(10, 25))
                content = response.content
                extension = ".pdf" if content.startswith(b"%PDF") else ".bin"
                raw = OUT / "raw" / (key + extension)
                raw.parent.mkdir(exist_ok=True)
                raw.write_bytes(content)
                record.update(http_status=response.status_code, raw_path=raw.relative_to(OUT).as_posix(), bytes=len(content),
                              sha256=hashlib.sha256(content).hexdigest(), status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR")
                stopped = response.status_code in (403, 429)
                if response.status_code == 200 and payload is not None:
                    body = response.json()
                    rows = body.get("announcements") or []
                    record.update(total=body.get("totalAnnouncement"), first_page_rows=len(rows),
                                  row_keys=sorted(rows[0]) if rows else [],
                                  sample_titles=[re.sub("<[^>]+>", "", r["announcementTitle"]) for r in rows[:8]],
                                  sample_security_codes=[r["secCode"] for r in rows[:8]])
                elif key == "free_float_rule" and extension == ".pdf":
                    with pdfium.PdfDocument(content) as pdf:
                        pages = []
                        for page in pdf:
                            with page.get_textpage() as textpage:
                                pages.append(textpage.get_text_range())
                            page.close()
                    save(OUT / "text/free_float_rule.json", pages)
                    record["pages"] = len(pages)
                elif key == "homepage" and response.status_code == 200:
                    record["scripts"] = re.findall(r'<script[^>]+src=["\x27]([^"\x27]+)', response.text)
            except requests.RequestException as exc:
                record.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
            except (ValueError, KeyError) as exc:
                record.update(status="RESPONSE_PARSE_FAILED", error_type=type(exc).__name__)
        record["completed_at"] = now()
        save(OUT / "receipts" / (key + ".json"), record)
        summaries.append(record)
        print(json.dumps({k: v for k, v in record.items() if k not in ["payload", "url", "row_keys", "scripts"]}, ensure_ascii=False), flush=True)
        time.sleep(.4)
    save(OUT / "result.json", {"at": now(), "requests": summaries, "new_accounts": 0,
         "free_float_denominator_admitted": False, "full_issuance_calendar_admitted": False,
         "goal_achieved": False, "goal_status": "active", "external_review": "NOT_PERFORMED"})


if __name__ == "__main__":
    main()
