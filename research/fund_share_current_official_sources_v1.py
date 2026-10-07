"""只获取当前官方来源与实现说明，不重扫已关闭的历史份额响应。"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_current_official_share_source_v1"
SOURCES = (
    ("FUND_HOME", "https://etf.sse.com.cn/home/"),
    ("UNITS_PAGE", "https://www.sse.com.cn/market/funddata/volumn/etfvolumn/"),
    ("FUND_DETAIL", "https://etf.sse.com.cn/fundlist/funddetail/index.shtml?fundCode=510300"),
    ("FUND_HOME_JS", "https://etf.sse.com.cn/xhtml/js/home.js"),
)
PRIOR = ROOT / "reports/research/510300_fund_share_publication_receipts_closure_v1/result.json"
ROUTES = ROOT / "reports/research/510300_point_financing_etf_prior_routes_v1/source_and_old_purpose_facts.json"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(sid, url):
    folder = OUT / "sources" / sid
    folder.mkdir(parents=True, exist_ok=False)
    record = {"id": sid, "url": url, "started_at": now(), "retries": 0,
              "historical_first_publication": "NOT_ESTABLISHED"}
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        response = session.get(url, timeout=(10, 25))
        data = response.content
        record.update(http_status=response.status_code, final_url=response.url,
                      headers={k: response.headers.get(k) for k in ["Content-Type", "Date", "Last-Modified"]})
        if len(data) > 10_000_000:
            raise ValueError("响应超过10MB本次上限。")
        with (folder / "response.body").open("xb") as stream:
            stream.write(data)
        record.update(bytes=len(data), sha256=sha(folder / "response.body"))
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        text = response.text
        soup = BeautifulSoup(text, "html.parser") if not sid.endswith("JS") else None
        if soup is not None:
            scripts = [{"url": urljoin(response.url, x["src"])} for x in soup.find_all("script", src=True)]
            record["linked_scripts"] = scripts
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
        (folder / "text.txt").write_text(text, encoding="utf-8")
        record.update(status="SAVED_CURRENT_OFFICIAL_RESPONSE", text_characters=len(text))
    except Exception as error:
        record.update(status="CURRENT_OFFICIAL_REQUEST_FAILED", error_type=type(error).__name__, error=str(error))
    finally:
        session.close()
        record["captured_at"] = now()
        write(folder / "receipt.json", record)
    print(f"{sid}：{record['status']}，字符{record.get('text_characters', 0)}。", flush=True)
    return record


def register():
    if (OUT / "source_protocol.json").exists():
        raise RuntimeError("本次当前来源检查已经登记。")
    old = read(PRIOR)
    if old["status"] != "CLOSED_NO_HISTORICAL_PUBLICATION_TIME_IN_DIRECTLY_LINKED_RECEIPTS":
        raise ValueError("旧份额来源终态与已读事实不同。")
    state_path = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
    state = read(state_path)
    from research.source_expectation_transmission_review_v1 import FORWARD, FINANCE
    write(OUT / "source_protocol.json", {
        "at": now(), "registration": "TECH.R243", "decision": "TECH.R244",
        "scope": "当前官方原文/字段用途核对，必要时建立独立真实时点的日频份额观察；不是新的盈利策略。",
        "already_seen": "R242、旧G01—G04与8规则固定拒绝、2220历史响应关闭结论、官网搜索与打开内容已知。",
        "source_urls": [{"id": sid, "url": url} for sid, url in SOURCES],
        "bounded_requests": "4固定网页/脚本GET，最多2个由当前官方原文链接的相关脚本，再最多2个当前API请求；失败保留，不重试、不历史批量查询。",
        "clock_question": "官网总规模23:00说明的适用字段，能否与单ETF清算后份额区分；实际采集钟是否足够形成新的可知版本？",
        "no_historical_clock_transfer": "当前总规模更新时间不认证历史TOT_VOL首次公开，也不把采集时间倒填历史。",
        "closed_response_rescans": 0, "old_rule_replays": 0, "new_fits": 0, "new_labels": 0,
        "new_accounts": 0, "new_financial_runs": 0, "new_minute_bars": 0,
        "collector_sha256": sha(Path(__file__)),
        "prior_sources": [{"path": str(p.absolute().relative_to(ROOT)), "sha256": sha(p)} for p in [PRIOR, ROUTES]],
        "forward_before": {k: state[k] for k in FORWARD}, "financial_before": {k: state[k] for k in FINANCE},
        "financial_admission": "NOT_ADMITTED", "goal_achieved": False,
    })
    print("R243当前官方来源核对已登记；旧历史响应不重扫，0新金融。", flush=True)


def collect():
    protocol = read(OUT / "source_protocol.json")
    if sha(Path(__file__)) != protocol["collector_sha256"]:
        raise ValueError("来源采集代码已改变。")
    write(OUT / "collection_started.json", {"at": now(), "fixed_requests": 4})
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda row: fetch(*row), SOURCES))
    write(OUT / "current_source_receipts.json", {"at": now(), "requests": 4, "results": receipts,
          "success": sum(x["status"] == "SAVED_CURRENT_OFFICIAL_RESPONSE" for x in receipts),
          "closed_response_rescans": 0, "new_accounts": 0})


def main():
    parser = argparse.ArgumentParser(description="当前官方份额来源一次核对")
    parser.add_argument("action", choices=("register", "collect"))
    args = parser.parse_args()
    {"register": register, "collect": collect}[args.action]()


if __name__ == "__main__":
    main()
