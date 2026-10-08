"""验证完整资料还原时不会同时积累多个下载附件。"""
import csv
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "tools/repository/snapshot.py"
SPEC = importlib.util.spec_from_file_location("snapshot_space", SCRIPT)
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class RestoreSpaceTests(unittest.TestCase):
    def test_multiple_chunks_and_bundles_use_one_download_at_a_time(self):
        original = b"\xff\x00\r\n" * 21
        materials = {"part-a.bin": original[:17], "part-b.bin": original[17:]}
        contents = {"data/分片原件.bin": original, "data/甲.txt": "证据甲\r\n".encode(), "data/乙.txt": "证据乙\r\n".encode()}
        assets = []
        for name, relative in (("bundle-a.zip", "data/甲.txt"), ("bundle-b.zip", "data/乙.txt")):
            memory = io.BytesIO()
            with zipfile.ZipFile(memory, "w") as archive:
                archive.writestr(relative, contents[relative])
            materials[name] = memory.getvalue()
        for name, data in materials.items():
            assets.append({"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": "zip" if name.endswith(".zip") else "chunk"})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / "catalog"
            catalog.mkdir()
            index = catalog / "files-001.csv"
            with index.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256", "storage", "assets"])
                writer.writeheader()
                for relative, names in (("data/分片原件.bin", ["part-a.bin", "part-b.bin"]), ("data/甲.txt", ["bundle-a.zip"]), ("data/乙.txt", ["bundle-b.zip"])):
                    writer.writerow({"path": relative, "bytes": len(contents[relative]), "sha256": hashlib.sha256(contents[relative]).hexdigest(), "storage": "release", "assets": json.dumps(names)})
            summary = {"indexes": [{"path": "catalog/files-001.csv", "sha256": snapshot.sha256(index)}], "source_file_count": 3, "source_bytes": sum(map(len, contents.values())), "release_asset_count": 4, "release_asset_bytes": sum(map(len, materials.values()))}
            (catalog / "snapshot.json").write_text(json.dumps(summary), encoding="utf-8")
            (catalog / "assets.json").write_text(json.dumps({"assets": assets}), encoding="utf-8")
            called = []

            def download(asset, cache):
                self.assertEqual(list(cache.iterdir()), [], "前一个附件应在下一个附件下载前清理")
                called.append(asset["name"])
                target = cache / asset["name"]
                target.write_bytes(materials[asset["name"]])
                return target

            with patch.object(snapshot, "download", side_effect=download):
                result = snapshot.restore(root, ["data/"])
            self.assertEqual(result["新还原文件数"], 3)
            self.assertEqual(set(called), set(materials))
            self.assertEqual(list((root / ".release-downloads").iterdir()), [])
            for relative, data in contents.items():
                self.assertEqual((root / relative).read_bytes(), data)
            self.assertEqual(snapshot.verify(root, git_only=False)["状态"], "通过")


if __name__ == "__main__":
    unittest.main()
