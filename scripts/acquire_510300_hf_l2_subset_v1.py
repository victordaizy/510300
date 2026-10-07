"""获准后按远程行组提取目标ETF；默认只显示计划，不申请权限或整库下载。"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests


ROOT = Path(__file__).resolve().parents[1]
REPO = "venvoo/china-a-share-l2-level2-limit-order-book-tick-data"
REVISION = "8d942a74865a67de55ba01b0b982d7ac8743f456"
STREAMS = ("行情", "逐笔委托", "逐笔成交")
ALLOWED_SYMBOLS = {"510300": "510300.SH", "510330": "510330.SH", "159919": "159919.SZ"}
LIMIT = 256 * 1024 * 1024
OUT = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo_selected"
CN = timezone(timedelta(hours=8))


class RangeFile(io.RawIOBase):
    """只接受精确的HTTP范围响应；正文不写缓存文件，认证信息不进入日志。"""

    def __init__(self, url, budget, token=None, session=None):
        super().__init__()
        self.url = url
        self.budget = budget
        self.position = 0
        self.size = None
        self.receipts = []
        self.session = session or requests.Session()
        self.owns_session = session is None
        self.auth = {"Authorization": "Bearer " + token} if token else {}
        try:
            if self._fetch(0, 4) != b"PAR1":
                raise ValueError("远程文件没有Parquet文件头。")
        except Exception:
            if self.owns_session:
                self.session.close()
            raise

    def _fetch(self, offset, count):
        if count > self.budget["remaining"]:
            raise ValueError("远程读取将超过本批256MiB上限，已停止。")
        headers = dict(self.auth, Range=f"bytes={offset}-{offset + count - 1}",
                       **{"Accept-Encoding": "identity"})
        try:
            with self.session.get(self.url, headers=headers, stream=True, timeout=(8, 30)) as response:
                if response.status_code != 206:
                    raise ValueError(f"服务返回HTTP {response.status_code}；未读取整文件或绕过权限。")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or (int(match[1]), int(match[2])) != (offset, offset + count - 1):
                    raise ValueError("服务器返回的字节范围不符，已停止。")
                total = int(match[3])
                if self.size is not None and self.size != total:
                    raise ValueError("同一固定版本的文件大小发生变化。")
                self.size = total
                body = bytearray()
                for chunk in response.iter_content(min(65536, count)):
                    self.budget["remaining"] -= len(chunk)
                    body.extend(chunk)
                    if len(body) > count or self.budget["remaining"] < 0:
                        raise ValueError("服务器响应超出请求范围，已停止。")
                if len(body) != count:
                    raise ValueError("范围响应提前截断，已停止。")
        except requests.RequestException as error:
            raise ValueError(f"网络请求失败：{type(error).__name__}，未保存认证信息。") from None
        self.receipts.append({"offset": offset, "bytes": len(body),
                              "sha256": hashlib.sha256(body).hexdigest(),
                              "received_at": datetime.now(CN).isoformat()})
        return bytes(body)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("未知定位方式。")
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("不能定位到文件开头之前。")
        self.position = position
        return position

    def read(self, size=-1):
        self._checkClosed()
        count = max(0, self.size - self.position)
        if size is not None and size >= 0:
            count = min(count, size)
        if count == 0:
            return b""
        result = self._fetch(self.position, count)
        self.position += len(result)
        return result

    def readinto(self, buffer):
        result = self.read(len(buffer))
        buffer[:len(result)] = result
        return len(result)

    def close(self):
        if self.owns_session:
            self.session.close()
        super().close()


def candidate_row_groups(parquet, symbols):
    names = parquet.schema.names
    if "wind_code" not in names:
        raise ValueError("文件缺少原始证券代码，不能按后缀推测身份。")
    col = names.index("wind_code")
    selected = []
    for i in range(parquet.num_row_groups):
        stat = parquet.metadata.row_group(i).column(col).statistics
        if stat is None or not stat.has_min_max:
            selected.append(i)
            continue
        lower, upper = str(stat.min), str(stat.max)
        if any(lower < symbol + ":" and upper >= symbol for symbol in symbols):
            selected.append(i)
    return selected


def extract_table(parquet, groups, symbols):
    fragments = []
    retained_bytes = 0
    for group in groups:
        if parquet.metadata.row_group(group).total_byte_size > 512 * 1024 * 1024:
            raise ValueError("单行组解压后超过512MiB内存预算，停止读取。")
        table = parquet.read_row_group(group, use_threads=False)
        mask = pc.match_substring_regex(table["wind_code"], "^(" + "|".join(symbols) + r")(?:\.|$)")
        fragment = table.filter(pc.fill_null(mask, False))
        if fragment.num_rows:
            retained_bytes += fragment.nbytes
            if retained_bytes > 256 * 1024 * 1024:
                raise ValueError("目标记录超过256MiB内存预算，停止拼接。")
            fragments.append(fragment)
    if not fragments:
        raise ValueError("已检查候选行组，但没有目标证券记录；不解释为零成交。")
    return pa.concat_tables(fragments)


def load_local_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # 官方客户端负责刷新浏览器授权得到的短期令牌；旧环境仍兼容标准令牌文件。
    try:
        from huggingface_hub import get_token
    except ImportError:
        get_token = None
    if get_token is not None:
        return get_token() or ""
    hf_cache = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    token_path = Path(os.environ.get("HF_TOKEN_PATH", str(hf_cache / "token")))
    if token_path.is_file():
        return token_path.read_text(encoding="utf-8").strip()
    return ""


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prior_network_usage(folder):
    """累计本次来源目录的提取回执，避免换日期后重新获得完整预算。"""
    total = 0
    for receipt in folder.glob("*/receipt.json"):
        record = json.loads(receipt.read_text(encoding="utf-8"))
        value = record.get("network_body_bytes")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("已有提取批次缺少确定的网络用量；未启动新的读取。")
        total += value
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260812", help="首批选择与已存分钟及IOPV档案重叠的历史日期")
    parser.add_argument("--symbols", nargs="+", default=["510300"], choices=sorted(ALLOWED_SYMBOLS))
    parser.add_argument("--download", action="store_true", help="使用本地已配置的获准账号只读令牌执行提取")
    args = parser.parse_args()
    datetime.strptime(args.date, "%Y%m%d")
    metadata_file = ROOT / "data/raw/510300_free_channels_v1/20261001/venvoo/dataset_info.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    present = {row["rfilename"] for row in metadata["siblings"]}
    if metadata["sha"] != REVISION or any(f"{args.date}/{stream}.parquet" not in present for stream in STREAMS):
        raise SystemExit("固定版本或所选日期文件清单不符。")
    plan = {"repo": REPO, "revision": REVISION, "date": args.date, "symbols": args.symbols,
            "streams": STREAMS, "network_body_limit_bytes": LIMIT,
            "all_selected_files_disk_limit_bytes": LIMIT, "disk_cache": False,
            "output_directory": str(OUT.resolve()), "m1_m2_admitted": False,
            "strategy_returns_computed": False, "status": "PLAN_ONLY"}
    if not args.download:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    token = load_local_token()
    if not token:
        raise SystemExit("尚无本地只读认证；先完成网页访问申请及本地认证。无需在聊天中发送密码或令牌。")
    OUT.mkdir(parents=True, exist_ok=True)
    previous_network_bytes = prior_network_usage(OUT)
    if previous_network_bytes >= LIMIT:
        raise SystemExit("本次来源的累计网络读取已到256MiB上限，未发出新请求。")
    used = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    if used >= LIMIT or shutil.disk_usage(OUT.resolve()).free < LIMIT + 1024 ** 3:
        raise SystemExit("提取目录已到上限，或目标盘剩余空间不足，未发出下载请求。")
    run_dir = OUT / (args.date + "_" + datetime.now(CN).strftime("%Y%m%dT%H%M%S%f"))
    run_dir.mkdir()
    plan["status"] = "STARTED"
    plan["prior_network_body_bytes"] = previous_network_bytes
    plan["remaining_network_body_bytes_at_start"] = LIMIT - previous_network_bytes
    write_json(run_dir / "receipt.json", plan)
    budget = {"remaining": LIMIT - previous_network_bytes}
    plan["files"] = []
    try:
        for stream in STREAMS:
            relative = f"{args.date}/{stream}.parquet"
            url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(relative)}"
            entry = {"source_path": relative, "source_url": url, "status": "READING"}
            plan["files"].append(entry)
            with RangeFile(url, budget, token) as remote:
                try:
                    parquet = pq.ParquetFile(remote, pre_buffer=False, buffer_size=0)
                    groups = candidate_row_groups(parquet, args.symbols)
                    entry.update(source_bytes=remote.size, source_row_groups=parquet.num_row_groups,
                                 selected_row_groups=groups, schema=str(parquet.schema_arrow))
                    table = extract_table(parquet, groups, args.symbols)
                    entry.update(rows=table.num_rows,
                                 source_codes=pc.unique(table["wind_code"]).to_pylist(),
                                 source_dates=pc.unique(table["date"]).to_pylist(),
                                 first_time=pc.min(table["time"]).as_py(),
                                 last_time=pc.max(table["time"]).as_py(),
                                 rows_after_1505=pc.sum(pc.cast(pc.greater_equal(table["time"], 150500000), pa.int64())).as_py())
                    if entry["source_dates"] != [int(args.date)]:
                        raise ValueError("目标记录日期与请求日期不符，停止保存。")
                    sink = io.BytesIO()
                    pq.write_table(table, sink, compression="zstd")
                    content = sink.getvalue()
                    used = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
                    if used + len(content) + 1024 ** 2 > LIMIT:
                        raise ValueError("目标子集与回执将超过磁盘256MiB上限，未保存新文件。")
                    target = run_dir / (stream + ".parquet")
                    target.write_bytes(content)
                    entry.update(status="SAVED_RAW_VALUES_SUBSET", output_path=str(target.resolve()),
                                 bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                finally:
                    entry["range_receipts"] = remote.receipts
            print(f"已保存{stream}：{entry['rows']}条，{entry['bytes']}字节。", flush=True)
        plan["status"] = "EXTRACTED_PENDING_SEMANTIC_VALIDATION"
    except (ValueError, OSError, pa.ArrowException) as error:
        plan.update(status="STOPPED", reason=str(error))
        print(f"提取停止：{error}")
    finally:
        plan.update(network_body_bytes=LIMIT - previous_network_bytes - budget["remaining"],
                    cumulative_network_body_bytes=LIMIT - budget["remaining"],
                    finished_at=datetime.now(CN).isoformat())
        write_json(run_dir / "receipt.json", plan)
    if plan["status"] == "STOPPED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
