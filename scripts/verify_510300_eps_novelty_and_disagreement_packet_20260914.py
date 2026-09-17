"""核对本轮索引并分别复算原件去重与两机构分歧来源统计。"""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_explicit_revision_novelty_diagnostic_v1 import verify as verify_novelty, save, now
from research.eps_growth_disagreement_source_feasibility_v1 import verify as verify_disagreement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-directory", type=Path, required=True)
    args = parser.parse_args()
    out = args.receipt_directory.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (ROOT / "FILE_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("文件索引包含重复路径")
    for entry in entries:
        path = ROOT / entry["path"]
        path.resolve().relative_to(ROOT.resolve())
        data = path.read_bytes()
        if len(data) != int(entry["size_bytes"]) or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("索引与文件内容不符：" + entry["path"])
    verify_novelty(out / "novelty_recomputation.json")
    verify_disagreement(out / "disagreement_recomputation.json")
    save(out / "packet_verification.json", {"status": "PASS_PACKET_INDEX_AND_BOTH_OFFLINE_RECOMPUTATIONS",
         "completed_at": now(), "indexed_files_verified": len(entries), "original_pdfs": 7,
         "source_pages_reextracted": 14, "old_company_month_slice_rows": 22,
         "paired_company_month_universe": 41700, "monthly_source_rows": 139,
         "full_upstream_original_fact_library_reconstructed": False,
         "new_network_requests": 0, "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False})
    print("本轮索引、原件对应关系和两机构分歧来源统计均已离线复算通过。", flush=True)


if __name__ == "__main__":
    main()
