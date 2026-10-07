"""打包本轮保存结果，并在独立解压目录复算筛查；不重跑任何策略。"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STUDY = "510300_saved_candidates_sharpe13_v1"
REPORT = ROOT / "reports" / "research" / STUDY
NAME = "510300_夏普1.3早期训练档案重评_GPT审阅_20260922.zip"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    target = ROOT / "deliverables" / NAME
    building = target.with_suffix(".building.zip")
    if target.exists() or building.exists():
        raise FileExistsError("交付文件已经存在，禁止覆盖本轮已封存交付。")
    members: dict[str, bytes] = {}
    for path in sorted(REPORT.rglob("*")):
        if path.is_file() and "verification" not in path.relative_to(REPORT).parts and "__pycache__" not in path.relative_to(REPORT).parts and path.relative_to(REPORT).as_posix() != "delivery_receipt.json":
            members[path.relative_to(REPORT).as_posix()] = path.read_bytes()
    members["package_builder.py"] = Path(__file__).read_bytes()
    members["finish_saved_candidates_sharpe13_v1.py"] = (ROOT / "scripts/finish_saved_candidates_sharpe13_v1.py").read_bytes()
    index_stream = io.StringIO(newline="")
    writer = csv.DictWriter(index_stream, ["path", "bytes", "sha256"])
    writer.writeheader()
    for name, data in sorted(members.items()):
        writer.writerow({"path": name, "bytes": len(data), "sha256": digest(data)})
    members["FILE_INDEX.csv"] = index_stream.getvalue().encode("utf-8-sig")
    with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)

    extract_root = REPORT / "verification" / "fresh_extract"
    if extract_root.exists():
        raise FileExistsError("独立复算目录已存在，请使用新的交付版本。")
    extract_root.mkdir(parents=True)
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("压缩包重复路径或 CRC 检查失败。")
        index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if {r["path"] for r in index} != set(names) - {"FILE_INDEX.csv"}:
            raise ValueError("索引覆盖不完整。")
        for row in index:
            data = archive.read(row["path"])
            if len(data) != int(row["bytes"]) or digest(data) != row["sha256"]:
                raise ValueError(f"文件身份不一致：{row['path']}")
        for name in names:
            resolved = (extract_root / name).resolve()
            if not resolved.is_relative_to(extract_root.resolve()):
                raise ValueError("压缩包路径越界。")
        archive.extractall(extract_root)

    entrypoint = extract_root / "source_snapshot" / "scripts" / "review_510300_saved_candidates_sharpe13_v1.py"
    process = subprocess.run([sys.executable, "-X", "utf8", str(entrypoint)], cwd=extract_root, capture_output=True, text=True, encoding="utf-8", check=False)
    (REPORT / "verification" / "recompute_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (REPORT / "verification" / "recompute_stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"独立目录复算失败：{process.stderr}")
    replay = extract_root / "source_snapshot" / "reports" / "research" / STUDY
    verified_files = []
    for name in ("all_saved_metric_rows.csv", "metric_source_inventory.csv", "deduplicated_candidate_comparison.csv", "passing_model_saved_cycles.csv", "source_snapshot_manifest.json"):
        if (REPORT / name).read_bytes() != (replay / name).read_bytes():
            raise ValueError(f"独立目录复算不一致：{name}")
        verified_files.append(name)
    original_result = json.loads((REPORT / "result.json").read_text(encoding="utf-8"))
    replay_result = json.loads((replay / "result.json").read_text(encoding="utf-8"))
    original_result.pop("completed_at")
    replay_result.pop("completed_at")
    if original_result != replay_result:
        raise ValueError("独立目录汇总复算不一致。")
    target.parent.mkdir(parents=True, exist_ok=True)
    building.replace(target)
    receipt = {
        "status": "PASS_ZIP_STRUCTURE_AND_ISOLATED_SAVED_SCREEN_RECOMPUTATION",
        "zip": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size,
        "sha256": digest(target.read_bytes()), "members": len(members), "indexed_members": len(members)-1,
        "exactly_reproduced_files": verified_files + ["result.json_except_completed_at"],
        "new_fits": 0, "new_backtest_accounts": 0, "new_forward_events": 0,
        "new_downloads": 0, "current_target_net_sharpe": 1.3,
        "raw_market_to_account_replay_performed": False,
        "external_gpt_review_performed": False,
        "extra_security_audit_performed": False,
        "goal_achieved": False, "position_impact": 0,
    }
    (REPORT / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
