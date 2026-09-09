"""读取发布索引、验证原始字节，并从 GitHub Releases 还原研究材料。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
CHUNK = 4 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str, check_links: bool = True) -> Path:
    posix = PurePosixPath(relative)
    if not relative or "\\" in relative or ":" in relative or posix.is_absolute() or any(p in {"", ".", "..", ".git"} for p in relative.split("/")):
        raise ValueError(f"索引包含非法相对路径：{relative}")
    candidate = root.joinpath(*posix.parts)
    if check_links and not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"目标路径越过还原目录：{relative}")
    return candidate


def load_index(root: Path) -> tuple[dict, list[dict], dict[str, dict]]:
    summary = json.loads((root / "catalog/snapshot.json").read_text(encoding="utf-8"))
    rows = []
    for index in summary["indexes"]:
        path = safe_path(root, index["path"])
        if sha256(path) != index["sha256"]:
            raise ValueError(f"文件索引摘要不匹配：{index['path']}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row["bytes"] = int(row["bytes"])
                row["assets"] = json.loads(row["assets"])
                safe_path(root, row["path"], check_links=False)
                rows.append(row)
    paths = [row["path"] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("文件索引包含重复路径")
    assets_doc = json.loads((root / "catalog/assets.json").read_text(encoding="utf-8"))
    assets = {row["name"]: row for row in assets_doc["assets"]}
    if len(assets) != len(assets_doc["assets"]):
        raise ValueError("附件索引包含重复文件名")
    for row in rows:
        if row["storage"] not in {"git", "release"}:
            raise ValueError(f"未知存储方式：{row['path']}")
        if row["storage"] == "release" and not row["assets"]:
            raise ValueError(f"发布文件缺少附件映射：{row['path']}")
        for name in row["assets"]:
            if name not in assets:
                raise ValueError(f"缺少附件：{name}")
    if len(rows) != summary["source_file_count"]:
        raise ValueError("索引数量与快照摘要不一致")
    if sum(row["bytes"] for row in rows) != summary["source_bytes"]:
        raise ValueError("索引字节总量与快照摘要不一致")
    if len(assets) != summary["release_asset_count"] or sum(a["bytes"] for a in assets.values()) != summary["release_asset_bytes"]:
        raise ValueError("附件数量或总大小与快照摘要不一致")
    if {name for row in rows for name in row["assets"]} != set(assets):
        raise ValueError("附件映射存在遗漏或无对应源文件的附件")
    return summary, rows, assets


def verify(root: Path, git_only: bool = True) -> dict:
    summary, rows, assets = load_index(root)
    errors, checked, skipped = [], 0, 0
    for row in rows:
        if git_only and row["storage"] != "git":
            skipped += 1
            continue
        path = safe_path(root, row["path"])
        if not path.is_file():
            errors.append({"path": row["path"], "reason": "文件不存在"})
        elif path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            errors.append({"path": row["path"], "reason": "文件字节数或SHA-256不匹配"})
        checked += 1
    result = {"状态": "通过" if not errors else "失败", "检查文件数": checked, "本模式未检查的Release文件数": skipped, "附件数": len(assets), "错误": errors[:50]}
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def download(asset: dict, cache: Path) -> Path:
    target = safe_path(cache, asset["name"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == asset["bytes"] and sha256(target) == asset["sha256"]:
        return target
    expected_base = "https://github.com/victordaizy/510300/releases/download/"
    if not asset["url"].startswith(expected_base):
        raise ValueError("附件URL不属于本仓库的GitHub Releases")
    partial = target.with_name(target.name + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "510300-snapshot-restorer"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(asset["url"], headers=headers)
    print(f"下载附件：{asset['name']}，预期 {asset['bytes']:,} 字节", flush=True)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and partial.exists() and partial.stat().st_size == asset["bytes"] and sha256(partial) == asset["sha256"]:
            os.replace(partial, target)
            return target
        raise
    with response:
        mode = "ab" if offset and response.status == 206 else "wb"
        if mode == "ab" and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("续传响应的字节起点不一致")
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, CHUNK)
    if partial.stat().st_size != asset["bytes"] or sha256(partial) != asset["sha256"]:
        raise ValueError(f"附件未完整下载或摘要不匹配：{asset['name']}")
    os.replace(partial, target)
    return target


def write_verified(root: Path, row: dict, readers, overwrite: bool = False) -> bool:
    target = safe_path(root, row["path"])
    if target.exists():
        if target.is_file() and target.stat().st_size == row["bytes"] and sha256(target) == row["sha256"]:
            return False
        if not overwrite:
            raise FileExistsError(f"目标有不同内容；请先保留本地修改，或明确指定 --overwrite：{row['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".restoring")
    if partial.exists():
        raise FileExistsError(f"存在未完成的还原文件，请先检查：{partial}")
    digest, size = hashlib.sha256(), 0
    try:
        with partial.open("xb") as handle:
            for reader in readers:
                with reader() as source:
                    for block in iter(lambda: source.read(CHUNK), b""):
                        handle.write(block)
                        digest.update(block)
                        size += len(block)
        if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError(f"还原后的原文件摘要不匹配：{row['path']}")
        os.replace(partial, target)
        return True
    finally:
        if partial.exists():
            partial.unlink()


def restore(root: Path, prefixes: list[str], overwrite: bool = False) -> dict:
    _, rows, assets = load_index(root)
    selected = [row for row in rows if row["storage"] == "release" and (not prefixes or any(row["path"].startswith(p) for p in prefixes))]
    if not selected:
        raise ValueError("没有匹配的Release文件，请检查 --only 路径前缀")
    needed = sorted({name for row in selected for name in row["assets"]})
    cache = root / ".release-downloads"
    cache.mkdir(exist_ok=True)
    downloads = {name: download(assets[name], cache) for name in needed}
    completed = 0
    bundles: dict[str, list[dict]] = {}
    for row in selected:
        if len(row["assets"]) == 1 and assets[row["assets"][0]]["kind"] == "zip":
            bundles.setdefault(row["assets"][0], []).append(row)
        else:
            readers = [(lambda path=downloads[name]: path.open("rb")) for name in row["assets"]]
            completed += write_verified(root, row, readers, overwrite)
    for name, members in bundles.items():
        with zipfile.ZipFile(downloads[name]) as archive:
            names = archive.namelist()
            if len(set(names)) != len(names):
                raise ValueError(f"附件中有重复成员：{name}")
            for member in names:
                safe_path(root, member)
            for row in members:
                if archive.getinfo(row["path"]).file_size != row["bytes"]:
                    raise ValueError(f"ZIP成员大小不符：{row['path']}")
                completed += write_verified(root, row, [lambda path=row["path"]: archive.open(path)], overwrite)
        print(f"已还原附件：{name}", flush=True)
    return {"状态": "通过", "匹配文件数": len(selected), "新还原文件数": completed, "附件数": len(needed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    checking = sub.add_parser("verify", help="验证源文件大小、SHA-256和索引结构")
    checking.add_argument("--all", action="store_true", help="同时验证已还原的全部Release文件")
    restoring = sub.add_parser("restore", help="下载附件并还原原目录")
    choice = restoring.add_mutually_exclusive_group(required=True)
    choice.add_argument("--all", action="store_true", help="还原全部Release材料")
    choice.add_argument("--only", action="append", help="只还原指定相对路径前缀，可重复使用")
    restoring.add_argument("--overwrite", action="store_true", help="明确允许覆盖与快照不同的本地目标文件")
    args = parser.parse_args()
    try:
        result = verify(args.root, not args.all) if args.command == "verify" else restore(args.root, args.only or [], args.overwrite)
    except Exception as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
