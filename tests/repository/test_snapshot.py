"""验证发布材料还原的关键行为，不运行历史研究。"""
import hashlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools/repository/snapshot.py"
SPEC = importlib.util.spec_from_file_location("snapshot", SCRIPT)
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class SnapshotTests(unittest.TestCase):
    def test_index_to_zip_to_original_end_to_end(self):
        content = '原始证据\r\n'.encode('utf-8') + b'\x00\xff'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            catalog = root / 'catalog'
            catalog.mkdir()
            package = root / 'fixture.zip'
            relative = 'data/research/原始文件.bin'
            with zipfile.ZipFile(package, 'w') as archive:
                archive.writestr(relative, content)
            index = catalog / 'files-001.csv'
            with index.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['path', 'bytes', 'sha256', 'storage', 'assets'])
                writer.writeheader()
                writer.writerow({'path': relative, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(), 'storage': 'release', 'assets': json.dumps(['fixture.zip'])})
            summary = {'indexes': [{'path': 'catalog/files-001.csv', 'sha256': snapshot.sha256(index)}], 'source_file_count': 1, 'source_bytes': len(content), 'release_asset_count': 1, 'release_asset_bytes': package.stat().st_size}
            (catalog / 'snapshot.json').write_text(json.dumps(summary), encoding='utf-8')
            assets = {'assets': [{'name': 'fixture.zip', 'kind': 'zip', 'bytes': package.stat().st_size, 'sha256': snapshot.sha256(package)}]}
            (catalog / 'assets.json').write_text(json.dumps(assets), encoding='utf-8')
            with patch.object(snapshot, 'download', return_value=package):
                result = snapshot.restore(root, ['data/'])
            self.assertEqual(result['新还原文件数'], 1)
            self.assertEqual((root / relative).read_bytes(), content)
            self.assertEqual(snapshot.verify(root, git_only=False)['状态'], '通过')

    def test_reject_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            for path in ("../outside", "/outside", "C:/outside", "a\\b", "a/../b", ".git/config"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    snapshot.safe_path(Path(folder), path)

    def test_chunk_restore_preserves_binary_and_crlf(self):
        content = b"\xff\xfe\x00\r\nraw\r\n" * 31
        row = {"path": "data/research/原始文件.bin", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertTrue(snapshot.write_verified(root, row, [lambda: io.BytesIO(content[:7]), lambda: io.BytesIO(content[7:])]))
            self.assertEqual((root / row["path"]).read_bytes(), content)
            self.assertFalse(snapshot.write_verified(root, row, []))

    def test_corruption_never_replaces_existing_file(self):
        row = {"path": "data/result.bin", "bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / row["path"]
            target.parent.mkdir()
            target.write_bytes(b"local edits")
            with self.assertRaises(FileExistsError):
                snapshot.write_verified(root, row, [lambda: io.BytesIO(b"good")])
            with self.assertRaises(ValueError):
                snapshot.write_verified(root, row, [lambda: io.BytesIO(b"bad!")], overwrite=True)
            self.assertEqual(target.read_bytes(), b"local edits")
            self.assertFalse(target.with_name("result.bin.restoring").exists())


if __name__ == "__main__":
    unittest.main()
