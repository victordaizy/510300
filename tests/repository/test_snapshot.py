"""验证发布材料还原的关键行为，不运行历史研究。"""
import hashlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools/repository/snapshot.py"
SPEC = importlib.util.spec_from_file_location("snapshot", SCRIPT)
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class SnapshotTests(unittest.TestCase):
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
