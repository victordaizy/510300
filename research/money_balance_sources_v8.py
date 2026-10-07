"""取得固定历史窗口的央行货币资产负债表，保存原件和版本限制。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json
import shutil
import requests
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_money_balance_transmission_v8"
SOURCES = [
    ("dcs2024", "存款性公司概览_2024.pdf", "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111416432184461.pdf"),
    ("dcs2025", "存款性公司概览_2025.pdf", "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172611493.pdf"),
    ("cb2024", "货币当局资产负债表_2024.pdf", "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/11/2025111416404019802.pdf"),
    ("cb2025", "货币当局资产负债表_2025.pdf", "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172659799.pdf"),
    ("odcs2025", "其他存款性公司资产负债表_2025.pdf", "https://www.pbc.gov.cn/diaochatongjisi/attachDir/2025/12/2025121517172647559.pdf"),
    ("nbs_method", "统计局_货币供应量编制方法.html", "https://www.stats.gov.cn/zs/tjws/zytjzbqs/hbgyl/202410/t20241025_1957180.html"),
    ("pbc_definition", "央行_狭义货币与广义货币.html", "https://www.pbc.gov.cn/rmyh/109339/2025080818580470423/index.html"),
]


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fetch(source):
    key, name, url = source
    path = OUT / "sources" / name
    receipt = path.with_suffix(path.suffix + ".receipt.json")
    if receipt.exists():
        row = json.loads(receipt.read_text(encoding="utf-8"))
        assert row["sha256"] == digest(path)
        return row
    response = requests.get(url, timeout=(15, 45), headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    if path.suffix == ".pdf":
        assert response.content.startswith(b"%PDF"), "返回内容不是PDF。"
    path.write_bytes(response.content)
    row = {"id": key, "name": name, "url": url, "final_url": response.url,
           "retrieved_at": now(), "sha256": digest(path), "bytes": path.stat().st_size,
           "first_historical_vintage_verified": False,
           "available_at_original_2024_2025_decision": False,
           "version_note": "本次取得的历史表格版本，仅用于机制复盘；网址日期不冒充历史首次发布日期。"}
    save(receipt, row)
    return row


def run():
    if (OUT / "source_receipt.json").exists():
        raise RuntimeError("已有完整来源收据，不重复采集。")
    for folder in ["sources", "inputs", "results", "code", "figures"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protocol = {
        "at": now(), "study_id": "510300_MONEY_BALANCE_TRANSMISSION_V8",
        "prior_turn_classification": "PROGRESS：V7完成247家公司现金流来源和三家原公告核对。",
        "question": "M1/M2变化的存款结构与银行资产负债对应项，是否支持实体信用与需求普遍恢复的解释？",
        "window": "固定2024-01至2025-08，共20个月；保留窗口内全部月度差额，2025年全部可算的三个月差额及同区间前年比较。",
        "source_year_tables": [2024, 2025],
        "beyond_window": "2025年9月之后列不进入解释或特征；原PDF原样保存。",
        "balances": "M2按国外净资产、政府净债权、非金融部门债权、其他金融部门债权与非M2负债等拆开；所有项目保留，舍入残差单列。",
        "government": "政府净债权拆为货币当局债权、其他存款性公司债权减货币当局政府存款；不能等同政府融资增量、财政支出或企业收到的清欠款。",
        "m1": "2024旧M1与2025新M1分别保留。2025表脚注回溯值只用于2025可比基数复盘，不能回填2024历史信息集。活期/定期结构变化是净余额结果，不能识别同一账户之间的实际转换流量。",
        "comparison": "以原有月度报告的公布值、时点和市场路径连接；新的事后资产负债数据不进入历史可交易信号，也不覆盖任何旧列。",
        "known_outcomes_already_seen": True,
        "source_search_snippets_already_seen": True,
        "new_models": 0, "new_accounts": 0, "orders": 0,
        "goal_achieved": False,
    }
    if not (OUT / "protocol.json").exists():
        save(OUT / "protocol.json", protocol)
        save(ROOT / "config/510300_money_balance_transmission_v8.json", protocol)
        inputs = {
            "monthly_context.csv": "reports/research/510300_macro_transmission_context_v4/results/104个月_多层证据与原后续路径.csv",
            "tsf_sources.csv": "reports/research/510300_macro_transmission_context_v4/results/2025年6至8月_社融多增来源.csv",
            "money_decomposition.csv": "reports/research/510300_macro_transmission_context_v4/inputs/money_decomposition.csv",
            "v7_summary.json": "reports/research/510300_operating_cashflow_transmission_v7/goal_progress.json",
        }
        frozen = {}
        for name, relative in inputs.items():
            source = ROOT / relative
            shutil.copy2(source, OUT / "inputs" / name)
            frozen[relative] = digest(source)
        save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "inputs": frozen})
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(fetch, SOURCES))
    # PDFium顺序提取，避免线程间共享底层PDF状态。
    for row in rows:
        path = OUT / "sources" / row["name"]
        if path.suffix != ".pdf":
            continue
        doc = pdfium.PdfDocument(str(path))
        texts = []
        for page_no in range(len(doc)):
            page = doc[page_no]
            textpage = page.get_textpage()
            texts.append(textpage.get_text_range())
            textpage.close()
            page.close()
        row["pages"] = len(doc)
        doc.close()
        path.with_suffix(".txt").write_text("\n\f\n".join(texts), encoding="utf-8")
    save(OUT / "source_receipt.json", {"at": now(), "sources": rows, "status": "原文取得，仅机制复盘版本"})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps({"状态": "固定历史原表已保存", "来源数": len(rows), "PDF数": sum("pages" in r for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    run()
