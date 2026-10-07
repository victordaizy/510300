"""复用已验证的有界PDF传输器，收齐固定13份标题漏选的实际提示文件。"""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research.factor96_rights_issue_documents_v1 as collector
from research.factor96_rights_intermediate_v1 import OUT
from research.factor96_rights_event_chains_v1 import read, save, digest, now


def main():
    mandate = read(ROOT/"config/510300_existing_data_training_mandate_v1.json")
    assert mandate["new_market_data_collection_enabled"] and not mandate["orders_authorized"]
    for row in read(OUT/"freeze.json")["files"]:
        assert digest(row["path"]) == row["sha256"]
    targets = read(OUT/"additional_source_targets.json")
    directory = OUT/"supplement"
    directory.mkdir(parents=True, exist_ok=True)
    save(directory/"run_started.json", {"at": now(), "targets": len(targets),
         "collector_path": collector.__file__, "collector_sha256": digest(collector.__file__),
         "scope_freeze_sha256": digest(OUT/"freeze.json")})
    collector.OUT = directory
    rows = []
    for index, target in enumerate(targets, 1):
        row = collector.extract(collector.download(target))
        for field in ("raw_snapshot", "text_snapshot", "receipt_snapshot"):
            if row.get(field):
                row[field] = str(directory/row[field])
        save(directory/"documents"/f"{row['document_id']}.json", row)
        rows.append(row)
        print("固定配股提示原文", index, "/", len(targets), row["document_id"], row["status"], flush=True)
    save(OUT/"additional_documents.json", rows)
    save(directory/"result.json", {"at": now(), "statuses": dict(Counter(r["status"] for r in rows)),
         "source_requests": sum(r["attempts"] for r in rows), "pages": sum(r.get("pages",0) for r in rows),
         "new_accounts": 0, "new_models": 0, "new_returns": 0, "orders_authorized": False})


if __name__ == "__main__":
    main()
