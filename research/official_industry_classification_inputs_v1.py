"""官方行业表的行归属与公布钟；不补证券分类、不混合版本。"""
from __future__ import annotations

import re

import pandas as pd


def compact(value):
    return re.sub(r"\s+", "", value or "")


def publication_clock(date):
    return pd.Timestamp(date, tz="Asia/Shanghai").normalize() + pd.Timedelta(hours=23, minutes=59)


def symbol(code):
    if not re.fullmatch(r"\d{6}", code or ""):
        return None
    if code.startswith("6") or code.startswith("900"):
        return code + ".SH"
    if code.startswith(("0", "2", "3")):
        return code + ".SZ"
    if code.startswith(("4", "8", "9")):
        return code + ".BJ"
    return None


def choose_snapshot(metadata, decision):
    decision = pd.Timestamp(decision)
    if decision.tzinfo is None:
        decision = decision.tz_localize("Asia/Shanghai")
    published = [row for row in metadata if publication_clock(row["published"]) <= decision]
    latest = max(published, key=lambda row: (publication_clock(row["published"]), row["id"])) if published else None
    # 已知最新发布无法解析时保持未知，不换成旧版本制造可知覆盖。
    return latest if latest is not None and latest.get("parse_passed") else None


def parse_row(raw, taxonomy, state, origin):
    expected = 5 if taxonomy == "CSRC_QUARTERLY" else 8
    if len(raw) != expected:
        return None
    values = [compact(cell) for cell in raw]
    code = values[3] if expected == 5 else values[0]
    ticker = symbol(code)
    if ticker is None:
        return None
    inherited = []
    if expected == 5:
        if values[0]:
            match = re.search(r"[（(]([A-Z])[）)]", values[0])
            section = match.group(1) if match else None
            new_section = section is None or section != state.get("section")
            state["section"] = section
            state["section_name"] = re.sub(r"[（(][A-Z][）)]", "", values[0])
            state["section_origin"] = origin
            # 新门类没有大类时保持未知，不沿用上个门类。
            if new_section:
                state.pop("major", None)
                state.pop("major_name", None)
                state.pop("major_origin", None)
        else:
            inherited.append("section_merged_cell")
        if values[1]:
            state["major"] = values[1] if re.fullmatch(r"\d{2}", values[1]) else None
            state["major_name"] = values[2] or None
            state["major_origin"] = origin
        else:
            inherited.append("major_merged_cell")
        section, section_name = state.get("section"), state.get("section_name")
        major, major_name = state.get("major"), state.get("major_name")
        company = values[4]
        section_origin, major_origin = state.get("section_origin"), state.get("major_origin")
        subclass = None
    else:
        section = values[2] if re.fullmatch(r"[A-Z]", values[2]) else None
        section_name = values[3] or None
        major = values[6] if re.fullmatch(r"\d{2}", values[6]) else None
        major_name = values[7] or None
        company, subclass = values[1], values[4] or None
        section_origin = major_origin = origin
    known = bool(section and section_name and major and major_name)
    return {"symbol": ticker, "stock_code": code, "company_name": company, "taxonomy": taxonomy,
        "section_code": section, "section_name": section_name, "major_code": major, "major_name": major_name,
        "industry_key": section + major if section and major else None, "manufacturing_subclass": subclass,
        "classification_known": known, "row_origin": origin, "section_origin": section_origin,
        "major_origin": major_origin, "layout_inheritance": ";".join(inherited),
        "ordinary_a_share_sh_sz": code.startswith(("0", "3", "6"))}


def quality(rows, text_codes):
    frame = pd.DataFrame(rows)
    parsed = set(frame.stock_code) if len(frame) else set()
    duplicate = int(frame.stock_code.duplicated(keep=False).sum()) if len(frame) else 0
    unknown = int((~frame.classification_known).sum()) if len(frame) else 0
    missing, extra = sorted(text_codes - parsed), sorted(parsed - text_codes)
    return {"parse_passed": bool(len(rows) and not duplicate and not unknown and not missing and not extra),
        "rows": len(rows), "unique_securities": len(parsed), "duplicate_security_rows": duplicate,
        "unknown_classification_rows": unknown, "unparsed_text_codes": missing, "codes_absent_from_text": extra}


def classify_members(members, snapshot):
    if snapshot is None:
        result = members.copy()
        result["classification_known"] = False
        return result
    if snapshot.symbol.duplicated().any():
        raise ValueError("快照证券重复，不能合并成员分类。")
    result = members.merge(snapshot, on="symbol", how="left", validate="one_to_one")
    result["classification_known"] = result.classification_known.eq(True)
    return result
