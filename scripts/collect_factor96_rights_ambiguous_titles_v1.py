"""只补取已冻结717标题中的6份发行相关事项公告，判别是否包含实施日历。"""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research.factor96_rights_issue_documents_v1 as collector
from research.factor96_rights_intermediate_v1 import OUT
from research.factor96_rights_event_chains_v1 import read, save, digest, now


def main():
    source = OUT/"unselected_title_review.json"
    targets = [r for r in read(source) if r["title"] == "关于配股发行股票相关事项的公告"]
    assert len(targets) == 6
    directory = OUT/"ambiguous_title_supplement"
    save(directory/"protocol_addendum.json", {"at": now(), "scope": "全部6份发行股票相关事项标题；仅从文字标题不能判定实施或筹划，因此在读取其正文前固定补充范围。",
         "targets": targets, "parent_title_inventory_sha256": digest(source),
         "transport": "沿用父协议的有界公开PDF传输规则；不拓展关键词或日期窗。",
         "new_accounts": 0, "new_returns": 0, "new_models": 0, "orders_authorized": False})
    mandate = read(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    assert mandate["new_market_data_collection_enabled"] and not mandate["orders_authorized"]
    collector.OUT = directory
    rows = []
    for number, target in enumerate(targets, 1):
        row = collector.extract(collector.download(target))
        for key in ("raw_snapshot", "text_snapshot", "receipt_snapshot"):
            if row.get(key):
                row[key] = str(directory/row[key])
        save(directory/"documents"/f"{row['document_id']}.json", row)
        rows.append(row)
        print("模糊发行标题原文", number, "/6", row["document_id"], row["status"], flush=True)
    save(directory/"documents.json", rows)
    save(directory/"result.json", {"at": now(), "documents": len(rows),
         "statuses": dict(Counter(r["status"] for r in rows)), "requests": sum(r["attempts"] for r in rows),
         "new_accounts": 0, "new_returns": 0, "new_models": 0, "orders_authorized": False})


if __name__ == "__main__":
    main()
