"""补充既有LPR病例的先后信息、波动窗口构成与原文；不重做模型。"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_policy_information_timing_v9"
POPPLER = Path("E:/CodexData/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")
SOURCES = [
    {"key": "20230827_证券交易印花税公告", "suffix": ".html", "url": "https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm",
     "source_route_note": "财政部移动页及税政司页返回502，改用财政部主站同一公告，页面标注发布日期2023年8月27日；未改变公告内容。",
     "known_upper": "2023-08-27T23:59:59+08:00", "checks": ["2023年8月28日", "减半征收", "2023年8月27日"]},
    {"key": "20240722_090000_LPR公告", "suffix": ".html", "url": "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/5410019/index.html",
     "known_upper": "2024-07-22T09:00:00+08:00", "checks": ["2024-07-22 09:00:00", "3.35%", "3.85%"]},
    {"key": "2023年第二季度_货币政策执行报告", "suffix": ".pdf", "url": "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/4883187/e6dc129d48d9474cb6455691eee05766/2023081717054357626.pdf",
     "known_upper": "2023-08-17T23:59:59+08:00", "checks": ["合理看待我国商业银行利润水平", "1.74%", "6679"]},
]


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def acquire(spec):
    path = OUT / "sources" / (spec["key"] + spec["suffix"])
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding="utf-8"))
        assert digest(path) == prior["sha256"]
        return prior
    if not path.exists():
        response = requests.get(spec["url"], timeout=(15, 45), headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        path.write_bytes(response.content)
    if spec["suffix"] == ".pdf":
        assert path.read_bytes().startswith(b"%PDF")
        document = pdfium.PdfDocument(path)
        pages = []
        for i in range(len(document)):
            page = document[i]
            tp = page.get_textpage()
            pages.append(f"\n【PDF第{i + 1}页】\n" + tp.get_text_range())
            tp.close()
            page.close()
        document.close()
        text = "\n".join(pages)
        for number in [11, 12]:
            prefix = OUT / "sources" / (spec["key"] + f"_第{number}页")
            subprocess.run([str(POPPLER), "-f", str(number), "-l", str(number), "-singlefile", "-r", "125", "-png", str(path), str(prefix)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            raw = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            raw = path.read_bytes().decode("gb18030")
        text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    path.with_suffix(".txt").write_text(text, encoding="utf-8")
    compact = "".join(text.split())
    for check in spec["checks"]:
        assert "".join(check.split()) in compact, (spec["key"], check)
    record = {**spec, "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "retrieved_at": now(), "historical_first_version_authenticated": False}
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def build():
    if (OUT / "results/extension_receipt.json").exists():
        raise RuntimeError("补充结果已经完成，不覆盖。")
    if not (OUT / "extension_protocol.json").exists():
        save("extension_protocol.json", {"at": now(), "purpose": "补充已经看过的7个病例的机制事实，不新增预测检验。",
            "selection": "沿用主协议7例，全56个月统一补充波动和货币分解。",
            "source_extension": "仅补2023年二季度央行报告、2023年8月27日财政部公告、2024年7月22日9点LPR公告。",
            "past_results_seen": True, "old_policy_catalog_mutated": False,
            "chronology": "事后政策节点不回填为入场信息；先前报告中的机制说明不充当本次LPR选择的已证实原因。",
            "date_specific_paths": ["2023-08-28", "2024-07-22", "2024-09-23", "2024-09-24"],
            "new_models": 0, "new_accounts": 0})
    receipts = [acquire(spec) for spec in SOURCES]
    data = pd.read_csv(OUT / "results/56个月_原调查新信息时序与价格波动.csv")
    vol = pd.read_csv(OUT / "inputs/volatility.csv").set_index("observation_date")
    money = pd.read_csv(OUT / "inputs/monthly_context.csv").set_index("stat_month")
    market = pd.read_csv(OUT / "inputs/market.csv")
    records = []
    for row in data.to_dict("records"):
        rec = {"month": row["month"], "known_money_stat_month": row["known_money_stat_month"],
               "pre_date": row["pre_announcement_close_date"], "review_date": row["first_reaction_date"]}
        for prefix, date in [("pre", rec["pre_date"]), ("review", rec["review_date"])]:
            v = vol.loc[date]
            for key in ["v_down2", "v_d5_down2", "v_recent5_down_sum", "v_previous5_down_sum", "v_exited5_down_sum", "v_variance_change_up2", "v_variance_change_down2", "v_variance_change_mean_correction", "v_variance_change_total"]:
                rec[prefix + "_" + key] = v[key]
            rec[prefix + "_recent5_down_energy_pct_squared"] = v.v_recent5_down_sum * 10000
            rec[prefix + "_previous5_down_energy_pct_squared"] = v.v_previous5_down_sum * 10000
            rec[prefix + "_exited5_down_energy_pct_squared"] = v.v_exited5_down_sum * 10000
            rec[prefix + "_d20_now_pct"] = 100 * np.sqrt(v.v_down2)
            rec[prefix + "_d20_five_days_ago_pct"] = 100 * np.sqrt(v.v_down2 - v.v_d5_down2)
        m = money.loc[row["known_money_stat_month"]]
        for key in ["d3_relative_current_log_pp", "d3_relative_base_revision_log_pp", "d3_relative_growth_log_pp", "d3_eligible", "source_url", "source_sha256"]:
            rec["money_" + key] = m[key]
        records.append(rec)
    pd.DataFrame(records).to_csv(OUT / "results/56个月_货币基数与波动窗口构成.csv", index=False, encoding="utf-8-sig")
    node = {"node_id": "V9_MOF_20230827", "source_available_upper": receipts[0]["known_upper"],
            "title": "证券交易印花税自2023年8月28日起减半征收", "source_url": receipts[0]["url"],
            "source_path": receipts[0]["path"], "source_sha256": receipts[0]["sha256"],
            "interpretation": "LPR之后的资本市场政策；不折算为510300交易费率，不给LPR单项归因。"}
    links = []
    for row in data.to_dict("records"):
        if pd.Timestamp(row["announcement_at"]) < pd.Timestamp(node["source_available_upper"]) <= pd.Timestamp(row["exit_date"] + "T15:00:00+08:00"):
            links.append({"month": row["month"], **node})
    pd.DataFrame(links).to_csv(OUT / "results/定向补充的窗口内后续政策.csv", index=False, encoding="utf-8-sig")
    price_facts = []
    for date in ["2023-08-28", "2024-07-22", "2024-09-23", "2024-09-24"]:
        i = int(market.index[market.date == date][0])
        r, prev = market.iloc[i], market.iloc[i - 1]
        price_facts.append({"date": date, "previous_close": float(prev.close), "open": float(r.open), "close": float(r.close),
            "gap_return": float(r.open / prev.close - 1), "intraday_return": float(r.close / r.open - 1),
            "total_close_return": float((r.close + r.dividend) / prev.close - 1),
            "purpose": "已经发生的价格路径，不能把整日变化归于某一公告。"})
    sep = data.set_index("month").loc["2024-09"]
    ei, xi = int(sep.entry_i), int(sep.exit_i)
    cutoff = int(market.index[market.date == "2024-09-23"][0])
    price = float(market.iloc[ei].open)
    before = float((market.iloc[cutoff].close + market.iloc[ei + 1:cutoff + 1].dividend.sum()) / price - 1)
    parts = {"original_target20": float(sep.target20), "before_september24_node_return": before,
             "after_september23_close_contribution_same_initial_capital": float(sep.target20 - before),
             "meaning": "按同一初始资金拆金额时间段，后段包含多个新事件；不是9月24日政策的纯因果效应。"}
    save("results/日期价格事实.json", {"dates": price_facts, "september2024_path": parts})
    summary = {"at": now(), "source_count": len(receipts), "mechanism_rows": len(records), "added_later_policy_links": len(links),
               "new_models": 0, "new_accounts": 0, "pdf_visual_review_pending": True, "goal_achieved": False}
    save("results/extension_receipt.json", summary)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps({"summary": summary, "price_facts": price_facts, "september2024_path": parts}, ensure_ascii=False))


if __name__ == "__main__":
    build()
