"""从已保存央行当期月报提取政府债券社融占比及其同比变化。"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import sys
import unicodedata

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_tsf_government_composition_source_v1"
STUDY = "510300_TSF_GOVERNMENT_COMPOSITION_SOURCE_V1"
SOURCE = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/tsf_stock_yoy_release_vintage_2015_2026.parquet"
SHARE = re.compile(r"政府债券余额占比\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*[,，]\s*同比\s*(?:([高低])\s*([0-9]+(?:\.[0-9]+)?)\s*个百分点|(持平))")
TOTAL = re.compile(r"社会融资规模存量为\s*[0-9]+(?:\.[0-9]+)?\s*万亿元\s*[,，]\s*同比增长\s*([0-9]+(?:\.[0-9]+)?)\s*%")


def parse(raw):
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style"]):
        node.decompose()
    text = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
    matches, totals = list(SHARE.finditer(text)), list(TOTAL.finditer(text))
    result = {"status": "NO_VIEW_NO_UNIQUE_COMPOSITION", "share_percent": None,
              "government_share_yoy_change_pp": None, "total_tsf_yoy_percent": None,
              "matches": [m.group(0) for m in matches], "total_matches": [m.group(0) for m in totals]}
    unique = {(float(m.group(1)), 0. if m.group(4) else float(m.group(3)) * (1 if m.group(2) == "高" else -1)) for m in matches}
    total_values = {float(m.group(1)) for m in totals}
    if len(unique) != 1 or len(total_values) != 1:
        return result
    share, change = unique.pop()
    if not 0 <= share <= 100 or not -100 <= change <= 100:
        result["status"] = "NO_VIEW_VALUE_OUT_OF_RANGE"
        return result
    notes = re.findall(r"自\s*2019\s*年\s*12\s*月[^。]*。", text)
    result.update(status="EXTRACTED_FROM_SAVED_PBC_RELEASE", share_percent=share,
                  government_share_yoy_change_pp=change, total_tsf_yoy_percent=total_values.pop(),
                  methodology_notes=notes)
    return result


def run():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("社融构成提取已经固定，不能覆盖。")
    source = pd.read_parquet(SOURCE)
    chosen = source[source.reference_period.ge("2019-12")].copy()
    assert chosen.reference_period.is_unique
    assert pd.PeriodIndex(chosen.reference_period, freq="M").equals(pd.period_range(chosen.reference_period.min(), chosen.reference_period.max(), freq="M"))
    for folder in ["code", "results"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    files = {SOURCE.relative_to(ROOT).as_posix(): digest(SOURCE), Path(__file__).relative_to(ROOT).as_posix(): digest(Path(__file__))}
    for row in chosen.itertuples():
        path = ROOT / row.raw_path
        assert path.is_file() and digest(path) == row.source_hash, row.reference_period
        files[row.raw_path] = row.source_hash
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY,
        "question": "政府债券在社融存量中比重的当期同比变化，能否补充此前只用社融总量增速的信息？本阶段只提取来源，不计算策略收益。",
        "feature": "government_share_yoy_change_pp：当期月报直接披露的政府债券余额占比同比高/低的百分点；不从事后修订历史表回算。",
        "interpretation": "政府债券融资占总量比重的变化，不是政府买入股票，也不等同私人信贷；非政府债券项目仍可能含国企及融资平台。方向和可交易性尚待检验。",
        "scope": "2019-12至既有最后月报；国债及地方一般债纳入政府债券的统计口径变更从原文保留，不向更早期补造同口径数据。",
        "clock": "保留统计月份、来源发布日期、已保存时间戳和本次提取时点。研究最早在发布日期日末后的下一交易日使用；历史首次实际送达未认证。",
        "freshness_for_later_test": "最后一份月报最迟使用至其发布日加31自然日；超过该日期需要新月报，不能无限延用。",
        "no_discretionary_case_selection": "全部选定月份逐一提取；缺字段、冲突或与原社融同比不一致保留NO_VIEW。",
        "existing_research_difference": "此前总社融同比三月变化及信用利差不区分政府债券占比，本次仅新增当期政府债券比重同比变化；不扫描其他社融分项。",
        "preliminary_source_examples_seen": ["2019-12", "2020-01", "2023-01", "2025-08", "2026-07"],
        "new_candidate_returns_loaded": False, "new_accounts": 0, "new_model_fits": 0,
        "new_network_requests": 0, "sources": files, "orders_authorized": False}, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    rows = []
    for row in chosen.to_dict("records"):
        extracted = parse((ROOT / row["raw_path"]).read_bytes())
        if extracted["status"] == "EXTRACTED_FROM_SAVED_PBC_RELEASE":
            if not np.isclose(extracted["total_tsf_yoy_percent"], row["first_release_value"], atol=1e-12, rtol=0):
                extracted["status"] = "NO_VIEW_TOTAL_YOY_SOURCE_MISMATCH"
        published = pd.Timestamp(row["published_at"])
        conservative = max(published, pd.Timestamp(row["available_at"]))
        conservative = conservative.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
        rows.append({"reference_period": row["reference_period"], "published_at": published,
                     "saved_available_at": row["available_at"], "conservative_known_at": conservative,
                     "source_url": row["source_url"], "raw_path": row["raw_path"], "source_hash": row["source_hash"],
                     "source_retrieved_at": row["retrieved_at"], "extracted_at": now(), **extracted})
    save(OUT / "results/released_composition_rows.json", rows, True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "results/released_composition.parquet", index=False)
    valid = frame[frame.status.eq("EXTRACTED_FROM_SAVED_PBC_RELEASE")]
    result = {"at": now(), "study_id": STUDY,
        "status": "SAVED_RELEASE_COMPOSITION_READY" if len(valid) == len(frame) else "SAVED_RELEASE_COMPOSITION_PARTIAL",
        "selected_months": len(frame), "extracted_months": len(valid), "statuses": frame.status.value_counts().to_dict(),
        "first_reference_period": valid.reference_period.min() if len(valid) else None,
        "last_reference_period": valid.reference_period.max() if len(valid) else None,
        "first_conservative_known_at": valid.conservative_known_at.min() if len(valid) else None,
        "last_conservative_known_at": valid.conservative_known_at.max() if len(valid) else None,
        "missing_months": frame.loc[~frame.status.eq("EXTRACTED_FROM_SAVED_PBC_RELEASE"), "reference_period"].tolist(),
        "source_hashes_matched": len(chosen), "new_accounts": 0, "new_fits": 0,
        "new_independent_forward_observations": 0, "new_network_requests": 0,
        "historical_first_delivery_authenticated": False, "current_market_view": "NO_VIEW",
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False}
    save(OUT / "result.json", result, True)
    print({"社融构成来源完成": result["status"], "选定月份": len(frame), "已提取月份": len(valid), "未知月份": result["missing_months"]}, flush=True)


if __name__ == "__main__":
    run()
