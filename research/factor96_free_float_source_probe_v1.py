"""固定少量自由流通字段请求，区分字段可得、口径和历史版本证据。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_csi300_all_etf_momentum_v1 import credentials

OUT = ROOT / "reports/research/510300_factor96_free_float_source_probe_v1"
EVENTS = ROOT / "reports/research/510300_factor96_repurchase_purpose_v1/inputs/execution_events.json"
MARKET = ROOT / "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/market.parquet"
FIELDS = "ts_code,trade_date,close,total_share,float_share,free_share,total_mv,circ_mv,turnover_rate,turnover_rate_f"


def now():
    return pd.Timestamp.now(tz="Asia/Shanghai").isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    events = json.loads(EVENTS.read_text(encoding="utf-8"))
    ordered = sorted(events, key=lambda r: (r["catalogue_date"], r["symbol"], r["document_id"]))
    dates = pd.read_parquet(MARKET, columns=["date"]).date
    targets = []
    for rank in [0, len(ordered)//2, len(ordered)-1]:
        event = ordered[rank]
        prior = dates.loc[dates.lt(event["catalogue_date"])].max()
        targets.append({"rank": rank, "document_id": event["document_id"], "symbol": event["symbol"],
                        "catalogue_date": event["catalogue_date"], "trade_date": prior.strftime("%Y%m%d")})
    save("protocol.json", {"at": now(), "study_id": "510300_FACTOR96_FREE_FLOAT_SOURCE_PROBE_V1",
        "previous_turn_classification": "PROGRESS_FIXED_T11_DATE_PROXY_REJECTED",
        "lead": "旧daily_basic下载FIELDS未请求free_share，缺列不证明供应商接口没有此字段。",
        "purpose": "仅检查回购事件前自由流通分母字段可得性及数量级；不以流通股本替代自由流通。",
        "targets": targets, "selection": "948执行公告按目录日、公司、ID排序取首、中、末三条；使用严格前一交易日。",
        "api_name": "daily_basic", "fields": FIELDS,
        "documentation_url": "https://tushare.pro/document/2?doc_id=32",
        "max_api_requests": 3, "attempts_per_target": 1,
        "transport": "只用项目当前已配置HTTPS端点与已有凭据；不归档或打印凭据，不购买或扩展权限。",
        "stop": "认证/额度/限流失败停止本组后续请求；空值、无该列、非正值与不一致值不准入。",
        "unit_checks": "普通流通市值应等于close*float_share；总市值应等于close*total_share；free_share不超过float_share；单位需官方文档确认。",
        "history_boundary": "当前回取历史值不能证明历史首次可得或无后修订；只改善字段证据，不自动准入原严格T12。",
        "input_identities": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in [EVENTS, MARKET, Path(__file__)]],
        "new_strategy_returns": 0, "new_accounts": 0, "delivery_package_required": False, "orders_authorized": False})


def run():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for row in protocol["input_identities"]:
        assert digest(ROOT / row["path"]) == row["sha256"]
    assert not (OUT / "run_started.json").exists(), "已请求的固定组不重跑"
    save("run_started.json", {"at": now()})
    session = requests.Session()
    doc = {"at": now(), "url": protocol["documentation_url"]}
    try:
        response = session.get(doc["url"], timeout=(10, 30))
        (OUT / "daily_basic_documentation.html").write_bytes(response.content)
        soup = BeautifulSoup(response.content, "html.parser")
        article = soup.select_one(".document") or soup.select_one(".content") or soup
        text = article.get_text(" ", strip=True)
        (OUT / "daily_basic_documentation.txt").write_text(text, encoding="utf-8")
        token_pos = text.find("free_share")
        doc.update(status_code=response.status_code, sha256=digest(OUT / "daily_basic_documentation.html"),
                   free_share_definition_found=token_pos >= 0,
                   relevant_excerpt=text[max(0, token_pos-180):token_pos+380] if token_pos >= 0 else text[:160])
    except requests.RequestException as error:
        doc.update(status="DOCUMENT_REQUEST_FAILED", error_type=type(error).__name__)
    save("documentation_receipt.json", doc)
    secret, endpoint = credentials()
    expiry = os.environ.get("TUSHARE_PROXY_TOKEN_EXPIRES_AT")
    if endpoint != "https://api.tushare.pro" and expiry and pd.Timestamp(expiry) <= pd.Timestamp.now(tz="Asia/Shanghai"):
        save("result.json", {"at": now(), "status": "NOT_RUN_CONFIGURED_CREDENTIAL_EXPIRED", "api_requests": 0,
                             "new_accounts": 0, "goal_achieved": False, "delivery_package_required": False})
        print("已配置数据凭据过期，本组未发送API请求。", flush=True)
        return
    results, frames, halted = [], [], False
    for target in protocol["targets"]:
        label = target["symbol"]+"_"+target["trade_date"]
        receipt = {"at": now(), "target": target, "endpoint_host": urlparse(endpoint).hostname,
                   "api_name": "daily_basic", "fields": FIELDS, "attempts": 0}
        if halted:
            receipt["status"] = "NOT_RUN_AFTER_ACCESS_OR_RATE_FAILURE"
        else:
            request = {"api_name": "daily_basic", "params": {"ts_code": target["symbol"], "trade_date": target["trade_date"]}, "fields": FIELDS}
            headers = {}
            if endpoint == "https://api.tushare.pro":
                request["token"] = secret
            else:
                headers["x-api-key"] = secret
            receipt["attempts"] = 1
            try:
                response = session.post(endpoint, json=request, headers=headers, timeout=(10, 35))
                # 接口正常响应不含凭据；若异常服务反射请求，只记录脱敏响应。
                raw = response.content
                reflected = secret.encode() in raw
                if reflected:
                    raw = raw.replace(secret.encode(), b"[REDACTED]")
                raw_path = OUT / (label+".json")
                raw_path.write_bytes(raw)
                receipt.update(status_code=response.status_code, response_sha256=digest(raw_path), response_redacted=reflected)
                body = json.loads(raw)
                receipt["provider_code"] = body.get("code")
                if response.status_code in (401, 403, 429) or body.get("code") not in (0, "0"):
                    halted = True
                    receipt.update(status="SOURCE_ACCESS_OR_PROVIDER_ERROR", provider_message=str(body.get("msg"))[:160])
                else:
                    data = body.get("data") or {}
                    frame = pd.DataFrame(data.get("items", []), columns=data.get("fields", []))
                    receipt.update(rows=len(frame), fields_returned=list(frame), status="RESPONSE_SAVED")
                    if not frame.empty:
                        frame["probe_id"] = label
                        frames.append(frame)
            except (requests.RequestException, ValueError) as error:
                receipt.update(status="REQUEST_OR_RESPONSE_FAILED", error_type=type(error).__name__)
        save(label+"_receipt.json", receipt)
        results.append(receipt)
        print("自由流通探测："+label+"，"+receipt["status"]+"，行数="+str(receipt.get("rows", 0)), flush=True)
        if not halted:
            time.sleep(.6)
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(OUT / "probe_rows.parquet", index=False)
    save("result.json", {"at": now(), "status": "FIXED_FIELD_PROBE_COMPLETE_SEMANTIC_CHECK_PENDING",
        "api_requests": sum(r["attempts"] for r in results), "returned_rows": sum(r.get("rows", 0) for r in results),
        "document_definition_found": bool(doc.get("free_share_definition_found")), "halted": halted,
        "new_accounts": 0, "goal_achieved": False, "delivery_package_required": False, "results": results})


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "run":
        run()
    else:
        raise SystemExit("仅支持prepare或run")
