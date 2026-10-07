"""生成并核验本轮信息研究交付包；不改写既有研究或运行交易账户。"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    study = root / "reports/research/510300_expectations_policy_evidence_v1"
    deliverables = root / "deliverables"
    deliverables.mkdir(exist_ok=True)
    final = deliverables / "510300_货币预期差与政策证据_V1_GPT审阅_20260917.zip"
    building = final.with_suffix(".building.zip")
    if final.exists() or building.exists():
        raise FileExistsError("目标交付文件已经存在；不覆盖已交付快照。")
    for name in ["analyze_money_expectations_policy_v1.py", "plot_money_expectations_policy_v1.py",
                 "verify_money_expectations_policy_v1.py", "collect_money_consensus_public_v1.py",
                 "package_money_expectations_policy_v1.py"]:
        shutil.copy2(root / "research" / name, study / "code" / name)
    stage = Path(tempfile.mkdtemp(prefix="money_policy_v1_build_", dir=deliverables))
    assert deliverables.resolve() in stage.resolve().parents
    for name in ["00_README_FIRST.md", "研究结论.md", "用户需求.md", "GPT审阅提示词.md",
                 "来源与未补齐部分.md", "requirements.txt", "protocol.json", "status.json", "visual_check.json"]:
        shutil.copy2(study / name, stage / name)
    for folder in ["inputs", "results", "figures", "code", "source_records", "prior_records"]:
        shutil.copytree(study / folder, stage / folder)
    manifest = {"scope": "本轮新生成的信息研究，不含其他策略树或原历史 ZIP",
                "input_scope": "保存的数字输入、官方月报原文、第三方来源与哈希",
                "exclusions": ["整份第三方报告及其完整页面截图", "未准入搜索缓存和错误页", "虚拟环境", "其他研究目录"],
                "strict_forward_validation": False, "external_review": False,
                "generated_at": datetime.now(timezone.utc).isoformat()}
    (stage / "PACKAGE_SCOPE.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            body = path.read_bytes()
            rows.append({"path": path.relative_to(stage).as_posix(), "bytes": len(body),
                         "sha256": hashlib.sha256(body).hexdigest()})
    pd.DataFrame(rows).to_csv(stage / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())
    extracted = Path(tempfile.mkdtemp(prefix="money_policy_v1_verify_", dir=deliverables))
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert archive.testzip() is None
        assert set(names) == set(pd.DataFrame(rows).path) | {"FILE_INDEX.csv"}
        for name in names:
            target = (extracted / name).resolve()
            assert extracted.resolve() in target.parents
        archive.extractall(extracted)
    check = subprocess.run([sys.executable, str(extracted / "code/verify_money_expectations_policy_v1.py"),
                            "--study-dir", str(extracted)], capture_output=True, text=True, encoding="utf-8", check=True)
    recompute = Path(tempfile.mkdtemp(prefix="money_policy_v1_recompute_", dir=deliverables))
    calculation = subprocess.run([sys.executable, str(extracted / "code/analyze_money_expectations_policy_v1.py"),
                                  "--study-dir", str(extracted), "--output-dir", str(recompute)],
                                 capture_output=True, text=True, encoding="utf-8", check=True)
    compared = []
    for path in sorted(recompute.glob("*.csv")):
        pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(extracted / "results" / path.name),
                                      check_exact=False, atol=2e-11, rtol=1e-10)
        compared.append(path.name)
    pd.testing.assert_frame_equal(pd.read_parquet(recompute / "510300与当时已知货币数据_全部日线.parquet"),
                                  pd.read_parquet(extracted / "results/510300与当时已知货币数据_全部日线.parquet"))
    assert json.loads((recompute / "summary.json").read_text(encoding="utf-8")) == json.loads((extracted / "results/summary.json").read_text(encoding="utf-8"))
    building.replace(final)
    receipt = {"status": "PASS_ZIP_STRUCTURE_AND_FRESH_EXTRACTION_NUMERIC_RECOMPUTATION",
               "zip_path": str(final), "bytes": final.stat().st_size,
               "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
               "members": len(names), "indexed_members": len(rows),
               "crc_ok": True, "no_duplicate_members": True, "index_all_sizes_and_hashes_checked": True,
               "saved_numeric_verifier": json.loads(check.stdout), "recomputed_csv_files": compared,
               "daily_parquet_equal": True, "summary_equal": True,
               "fresh_extraction": str(extracted), "recompute_directory": str(recompute),
               "new_downloads_in_verification": 0, "model_fits": 0, "account_runs": 0,
               "external_review": False, "third_party_full_reports_in_zip": False}
    (study / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    (deliverables / "510300_货币预期差与政策证据_V1_交付回执.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
