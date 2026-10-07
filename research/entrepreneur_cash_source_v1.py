"""保存企业家季度原报告、公布时钟和两项现金状况字段的提取材料。"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import urljoin

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.banker_survey_source_probe_v1 as bank
from research.exchange_bank_funding_gap_daily_v1 import adapt
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_entrepreneur_cash_source_v1"
STUDY = "510300_ENTREPRENEUR_CASH_SOURCE_V1"
FETCH = adapt(bank.fetch, {"OUT": OUT})


def initial():
    if (OUT / "catalogue.json").exists():
        raise RuntimeError("企业家目录已经固定。")
    for folder in ["raw", "requests", "texts", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    records, receipts = {}, []
    for i in range(1, 6):
        receipt_path = bank.OUT / "requests" / f"catalogue_{i}.json"
        receipt = read(receipt_path)
        assert digest(ROOT / receipt["raw_path"]) == receipt["sha256"]
        receipts.append({"receipt": receipt_path.relative_to(ROOT).as_posix(), "sha256": digest(receipt_path)})
        soup = bank.soup_for(receipt)
        for a in soup.find_all("a", href=True):
            title = re.sub(r"\s+", "", a.get_text())
            m = re.fullmatch(r"(20\d{2})年第([一二三四])季度企业家问卷调查报告", title)
            if not m:
                continue
            quarter = f"{m.group(1)}Q{bank.QUARTERS[m.group(2)]}"
            if not "2018Q1" <= quarter <= "2026Q3":
                continue
            row = {"quarter": quarter, "title": title, "source_url": urljoin(receipt["url"], a["href"]),
                   "catalogue_source": receipt["raw_path"]}
            if quarter in records:
                assert records[quarter]["source_url"] == row["source_url"]
            else:
                records[quarter] = row
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "purpose": "与银行审批供给条件区分，核对企业经营现金回笼与周转状况的原文；目前只取得来源，不计算股票收益。",
        "selected_measurements": ["销货款回笼指数", "资金周转指数"],
        "semantic_limit": "两者是企业家对状况的扩散判断，不是财务现金流、应收账款真实周转天数或股市流量。资金周转仅作为分解解释候选，不默认再加入模型。",
        "duplicate_check": "已有PMI新订单、银行贷款需求及审批、贷款构成和住户存贷结果保留；本次是企业自身的收款状况，不改变旧策略。",
        "catalogue_reuse": receipts, "catalogued_quarters": len(records), "request_limit": 68,
        "acquisition": "只请求原目录实际链接的央行页面和PDF。25秒超时，403/429停止；一次传输失败最多额外普通重试一次，不关闭TLS验证。",
        "vintage": "每份原季度报告仅以后提取其当季值，后来附表中的旧值不回填。实际最早送达仍未认证。",
        "clock": "保存页面所载公布时间，保守采用公布日末，不能按季度末回填；同时补发的旧季不得反写此前决策。",
        "parser": "PDF同位置重复字形在1点坐标容差内去重；不按相邻字符删除数字。表头必须对齐，文本标题未识别保留待核。",
        "new_accounts": 0, "new_strategy_returns": 0, "orders_authorized": False, "goal_achieved": False}, True)
    save(OUT / "catalogue.json", sorted(records.values(), key=lambda r: r["quarter"]), True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "code_sha256": digest(Path(__file__)), "fetch_code_sha256": digest(Path(bank.__file__))}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"从已保存官方目录定位{len(records)}份企业家原报告，未重复下载目录。", flush=True)


def fetch_once(url, name):
    assert len(list((OUT / "requests").glob("*.json"))) < 68
    r = FETCH(url, name)
    if r["status"] == "TRANSPORT_FAILURE":
        r = FETCH(url, name + "_transport_retry")
    return r


def acquire():
    assert digest(Path(__file__)) == read(OUT / "freeze.json")["code_sha256"]
    if (OUT / "result.json").exists():
        raise RuntimeError("企业家来源已取得终态，不重复执行。")
    records, failures = [], []
    for item in read(OUT / "catalogue.json"):
        q = item["quarter"]
        try:
            html = fetch_once(item["source_url"], q + "_page")
            soup = bank.soup_for(html)
            clocks = re.findall(r"文章来源[：:]?\s*(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", soup.get_text(" ", strip=True))
            assert len(clocks) == 1, "原页面公布时间未唯一识别。"
            links = {urljoin(item["source_url"], a["href"]) for a in soup.find_all("a", href=True)
                     if re.search(r"\.pdf(?:$|[?#])", a["href"], flags=re.I)}
            assert len(links) == 1, "PDF链接不唯一。"
            pdf = fetch_once(next(iter(links)), q + "_pdf")
            assert pdf["status"] == "HTTP_OK", "PDF传输未成功。"
            with pdfplumber.open(ROOT / pdf["raw_path"]) as document:
                texts = [p.dedupe_chars(tolerance=1, extra_attrs=()).extract_text() or "" for p in document.pages]
            text_path = OUT / "texts" / (q + ".json")
            save(text_path, {"quarter": q, "pages": texts, "pdf_sha256": pdf["sha256"]}, True)
            compact = re.sub(r"\s+", "", "".join(texts))
            records.append({**item, "status": "ORIGINAL_SOURCE_SAVED_FIELDS_NOT_ADMITTED",
                "published_at": clocks[0] + "+08:00", "known_at": clocks[0][:10] + "T23:59:59+08:00",
                "page_raw_path": html["raw_path"], "page_sha256": html["sha256"],
                "pdf_url": pdf["url"], "pdf_raw_path": pdf["raw_path"], "pdf_sha256": pdf["sha256"],
                "text_path": text_path.relative_to(ROOT).as_posix(), "pages": len(texts),
                "title_text_matches": item["title"] in compact, "historical_first_vintage_verified": False})
        except (AssertionError, ValueError, RuntimeError) as exc:
            failures.append({**item, "error": str(exc)})
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
        if (len(records) + len(failures)) % 8 == 0:
            print(f"企业家原报告保存{len(records)}份，问题{len(failures)}份。", flush=True)
    save(OUT / "saved_originals.json", records, True)
    save(OUT / "unresolved_sources.json", failures, True)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "ORIGINAL_ENTREPRENEUR_REPORTS_SAVED_FIELD_ADMISSION_PENDING" if not failures else "PARTIAL_ORIGINAL_ENTREPRENEUR_REPORTS",
        "saved_quarters": len(records), "unresolved_quarters": len(failures),
        "unresolved_title_quarters": [r["quarter"] for r in records if not r["title_text_matches"]],
        "new_network_requests": len(list((OUT / "requests").glob("*.json"))),
        "new_strategy_returns": 0, "new_accounts": 0, "goal_status": "active", "goal_achieved": False}, True)
    print(f"企业家来源取得结束，{len(records)}份原报告已保存；尚未准入策略。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="企业家经营现金状况原始季度证据")
    parser.add_argument("command", choices=["initial", "acquire"])
    globals()[parser.parse_args().command]()
