"""补充原文“每股收益-最新股本摊薄”字段，不改变已成功提取的第三版事实。"""
from __future__ import annotations

from datetime import date
import re

from research.financial_annual_components_v1 import norm, compact
from research.forward_eps_guosen_history_v1 import exact_row
from research.forward_eps_soochow_facts_v1 import optional_snapshot
from research.forward_eps_soochow_layout_v3 import date_with_author, headers


def parse(pages: list[str], metadata: dict, row: dict) -> dict:
    """只识别有完整年度标题、明确股本口径的前三页预测摘要表。"""
    if metadata["info_code"] != row["infoCode"] or str(metadata["company_code"]) != "80000031":
        raise ValueError("原件编号或东吴机构身份不符")
    if row["ts_code"][:6] not in [str(item.get("stock")) for item in metadata["security"]]:
        raise ValueError("详情证券与目录证券不符")
    front = compact("\n".join(pages[:2]))
    if row["ts_code"][:6] not in front or not ("东吴证券" in front or "dwzq.com.cn" in front.lower()):
        raise ValueError("原件首页证券与机构身份未确认")
    internal = date_with_author(pages[0])
    clocks = [internal, str(metadata["notice_date"])[:10], str(metadata["eitime"])[:10], str(row["publishDate"])[:10]]
    for clock in clocks:
        date.fromisoformat(clock)
    tables = []
    for page_number, original in enumerate(pages[:3], 1):
        text = norm(original)
        for marker in re.finditer(r"盈利预测(?:与|预)估值", text):
            tail = text[marker.end():]
            metric = re.search(r"(?:营业总收入|营业收入|归属母公司净利润)\s*\(", tail)
            if metric is None:
                continue
            candidates = headers(tail[:metric.start()])
            if len(candidates) != 1:
                continue
            header = candidates[0]
            stop = re.search(r"\[[^\]\n]*Table_(?:Tag|Summary)[^\]\n]*\]|\n\s*(?:事件|投资要点|点评)", tail)
            body = tail[:stop.start()] if stop else tail
            try:
                eps = exact_row(body, ["每股收益-最新股本摊薄"], len(header["header"]), "元/股")
            except ValueError:
                continue
            optional, errors = {}, {}
            for key, labels, unit in [
                ("net_profit", ["归属母公司净利润"], "百万元"),
                ("pe", ["P/E"], "现价&最新股本摊薄"),
            ]:
                try:
                    optional[key] = exact_row(body, labels, len(header["header"]), unit)
                except ValueError as exc:
                    errors[key] = str(exc)
            tables.append({"page": page_number, "header": header["header"], "header_raw": header["header_raw"],
                           "eps": eps, "optional": optional, "optional_errors": errors})
    if len(tables) != 1:
        raise ValueError("最新股本摊薄明确摘要表未唯一识别：" + str(len(tables)))
    table = tables[0]
    shares = optional_snapshot(norm(pages[0]), "总股本", "百万股")
    quote = optional_snapshot(norm(pages[0]), "收盘价", "元")
    facts = []
    for i, (year, flag) in enumerate(table["header"]):
        if flag != "E":
            continue
        fact = {
            "report_id": row["infoCode"], "ts_code": row["ts_code"], "sec_name": row["stockName"],
            "institution_code": "80000031", "institution": "东吴证券", "target_fiscal_year": year,
            "eps_value_exact": table["eps"]["values"][i], "eps_unit_as_reported": "元/股",
            "source_eps_label": table["eps"]["label"], "eps_currency_iso_independently_proven": False,
            "eps_definition": "每股收益，按报告当时最新股本摊薄", "latest_diluted_basis_explicit": True,
            "source_page": table["page"], "source_eps_row": table["eps"]["raw"],
            "source_eps_cell": table["eps"]["cells"][i], "header": table["header"],
            "header_raw": table["header_raw"], "selected_column_one_based": i + 1,
            "report_internal_date": internal, "provider_notice_date": metadata["notice_date"],
            "provider_eitime": metadata["eitime"], "directory_publish_date": row["publishDate"],
            "conservative_information_date": max(clocks), "historical_immutable_snapshot_proven": False,
            "target_fiscal_year_already_ended": year < int(internal[:4]),
            "share_snapshot_million": shares["value"] if shares else None,
            "share_snapshot_raw": shares["raw"] if shares else None,
            "share_snapshot_is_rounded_report_value": True,
            "report_reference_close_exact": quote["value"] if quote else None,
            "report_reference_close_raw": quote["raw"] if quote else None,
            "actual_future_eps_used_as_predictor": False,
            "source_region": "完整页明确最新股本摊薄摘要表", "source_bbox_pdf_points": None,
        }
        for key in ["net_profit", "pe"]:
            value = table["optional"].get(key)
            fact[key + "_value_exact"] = value["values"][i] if value else None
            fact[key + "_source_raw"] = value["raw"] if value else None
            fact[key + "_source_label"] = ("归母净利润" if key == "net_profit" else "P/E") if value else None
            fact[key + "_source_label_as_reported"] = value["label"] if value else None
            fact[key + "_source_unit_as_reported"] = ("百万元" if key == "net_profit" else "现价&最新股本摊薄") if value else None
            fact[key + "_source_page"] = table["page"] if value else None
        facts.append(fact)
    return {"facts": facts, "table": table, "report_internal_date": internal,
            "conservative_information_date": max(clocks), "date_fields_differ": len(set(clocks)) > 1,
            "share_snapshot": shares, "reference_quote": quote,
            "adapter": "EXPLICIT_LATEST_SHARE_DILUTED_EPS_FIELD_V4"}
