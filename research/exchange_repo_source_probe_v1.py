"""限定公开来源探查交易所回购资金价格，保留原文与字段语义。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_exchange_repo_source_probe_v1"
STUDY = "510300_EXCHANGE_REPO_SOURCE_PROBE_V1"
PAGES = {
    "repo_curve": "https://bond.sse.com.cn/data/standard/repocurve/sevenrepo/",
    "interest_and_close_rule": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20170519_4313753.shtml",
    "margin_definition": "https://www.sse.com.cn/market/othersdata/margin/sum/",
}


def fetch(url, name):
    receipt_path = OUT / "requests" / f"{name}.json"
    if receipt_path.exists():
        raise RuntimeError(f"请求记录{name}已存在，先使用已保存响应。")
    receipt = {"requested_at": now(), "url": url, "status": "REQUEST_STARTED"}
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://bond.sse.com.cn/"})
        raw_path = OUT / "raw" / f"{name}.bin"
        raw_path.write_bytes(response.content)
        receipt.update(received_at=now(), http_status=response.status_code,
                       content_type=response.headers.get("Content-Type"), http_date=response.headers.get("Date"),
                       bytes=len(response.content), raw_path=raw_path.relative_to(ROOT).as_posix(), sha256=digest(raw_path),
                       status="HTTP_OK_UNPARSED" if response.ok else "HTTP_FAILURE")
        if response.status_code in [403, 429]:
            receipt["stop_new_requests"] = True
    except requests.RequestException as exc:
        receipt.update(received_at=now(), status="TRANSPORT_FAILURE", error=f"{type(exc).__name__}: {exc}")
    save(receipt_path, receipt, True)
    return receipt


def inspect_saved(name):
    receipt = read(OUT / "requests" / f"{name}.json")
    if receipt["status"] != "HTTP_OK_UNPARSED":
        return {"name": name, "status": receipt["status"]}
    soup = BeautifulSoup((ROOT / receipt["raw_path"]).read_bytes(), "html.parser")
    scripts = [urljoin(receipt["url"], tag["src"]) for tag in soup.find_all("script", src=True)]
    embedded = []
    for tag in soup.find_all("script", src=False):
        text = tag.get_text()
        if any(term in text for term in ["204007", "ajax", "sqlId", "repocurve", "dataUrl"]):
            embedded.append(text)
    result = {"name": name, "status": "HTML_SAVED_FIELD_MAPPING_PENDING", "scripts": scripts,
              "embedded_relevant_scripts": embedded, "page_title": soup.title.get_text() if soup.title else None}
    save(OUT / "results" / f"{name}_structure.json", result, True)
    print({"来源": name, "状态": result["status"], "脚本链接": scripts, "相关内联脚本数": len(embedded)}, flush=True)
    return result


def initial():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("来源探查已开始，不得重复请求已有页面。")
    for directory in ["raw", "requests", "results", "code"]:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    old_margin = read(ROOT / "reports/discovery/510300_market_leverage_cascade_5d_discovery_v0.json")
    old_flow = read(ROOT / "reports/research/510300_daily_etf_flow_direction_report.json")
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "question": "证券交易所资金价格相对银行间七天定盘的差异，能否提供此前银行间内部利差以外的信息？本阶段只取得并辨认来源，不计算股票收益。",
        "proposed_measurement": "优先辨认GC007/204007的收盘或公开加权资金利率，与同日FDR007同名义期限相比较；明确计息、交易时钟、抵押品和参与者差异，不能称纯非银资金成本。",
        "source_priority": "交易所公开页面及页面实际引用的数据；如历史覆盖不足，再依据独立公开数据源交叉核对，不补写缺值或认证未知时钟。",
        "time_scope": "2017-05-22计息和收盘规则变更之后至2026-09-24，最终范围取真实可用数据；信息仅作510300研究输入，不增加可执行资产。",
        "old_research_boundary": {"leverage_cascade": old_margin["status"], "etf_margin_direction": old_flow["status"],
                                  "reason_not_repeated": "旧余额、偿还强度与卖压交互已被检验；偿还总额不能分离现金还款、卖券还款和强平。此次变为另一个市场的资金价格，不重新调旧融资规则。"},
        "initial_pages": PAGES, "request_limit_initial": 3, "transport_timeout_seconds": 30,
        "blocked_responses": "403/429停止新增请求并保留响应，不解挑战或绕过鉴权。",
        "new_strategy_returns": 0, "new_accounts": 0, "scheduled_pcf_iopv_resumed": False,
        "orders_authorized": False, "goal_status": "active", "goal_achieved": False}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    receipts = []
    for name, url in PAGES.items():
        receipt = fetch(url, name)
        receipts.append(receipt)
        print({"请求": name, "状态": receipt["status"], "字节": receipt.get("bytes")}, flush=True)
        if receipt.get("stop_new_requests"):
            break
        inspect_saved(name)
    save(OUT / "initial_result.json", {"at": now(), "requests": receipts, "new_accounts": 0,
                                        "status": "INITIAL_SOURCE_PROBE_SAVED"}, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交易所回购资金价格公开来源探查")
    parser.add_argument("command", choices=["initial"])
    args = parser.parse_args()
    initial()
