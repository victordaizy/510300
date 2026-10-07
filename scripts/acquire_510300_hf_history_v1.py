"""分批提取510300完整日期区间的三流原值，远程数据块只在内存中读取。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.acquire_510300_hf_l2_subset_v1 import RangeFile, candidate_row_groups, load_local_token

CONFIG = ROOT / "config/510300_hf_history_expansion_v1.json"
BASE = ROOT / "data/raw/510300_free_channels_v1/20261001"
OUT = BASE / "venvoo_history_20261001"
CN = ZoneInfo("Asia/Shanghai")
DECODE_LOCK = threading.Lock()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path: Path, value: dict) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


class CachedRangeFile(RangeFile):
    """一次读取同一行组的连续压缩数据，避免逐列重复经过远程认证跳转。"""

    def __init__(self, *args, **kwargs):
        self.metadata_cache = []
        self.group_cache = None
        super().__init__(*args, **kwargs)

    def read(self, size=-1):
        self._checkClosed()
        count = max(0, self.size - self.position)
        if size is not None and size >= 0:
            count = min(count, size)
        if count == 0:
            return b""
        caches = ([self.group_cache] if self.group_cache else []) + self.metadata_cache
        for offset, content in caches:
            if offset <= self.position and self.position + count <= offset + len(content):
                start = self.position - offset
                self.position += count
                return content[start:start + count]
        offset = self.position
        result = super().read(count)
        if count <= 8 * 1024 * 1024 and len(self.metadata_cache) < 6:
            self.metadata_cache.append((offset, result))
        return result

    def cache_group(self, group, limit: int) -> None:
        spans = []
        for number in range(group.num_columns):
            column = group.column(number)
            candidates = [value for value in [column.dictionary_page_offset, column.data_page_offset]
                          if value is not None and value > 0]
            if not candidates:
                raise ValueError("列数据缺少可定位偏移。")
            offset = min(candidates)
            spans.append((offset, offset + column.total_compressed_size))
        low, high = min(item[0] for item in spans), max(item[1] for item in spans)
        if high - low > limit:
            raise ValueError("目标行组超过本批内存缓存上限。")
        if high - low > sum(end - start for start, end in spans) + 1024 * 1024:
            raise ValueError("行组数据块并不连续，停止扩大远程读取范围。")
        if high - low > self.budget["remaining"]:
            raise ValueError("目标行组将超过本文件网络预算。")
        self.group_cache = (low, self._fetch(low, high - low))


def extract_selected(parquet, remote, groups, config):
    fragments = []
    for index in groups:
        group = parquet.metadata.row_group(index)
        if group.total_byte_size > config["maximum_decoded_row_group_bytes"]:
            raise ValueError("行组解码量超过内存上限。")
        remote.cache_group(group, config["maximum_cached_row_group_bytes"])
        with DECODE_LOCK:
            table = parquet.read_row_group(index, use_threads=False)
            mask = pc.match_substring_regex(table["wind_code"], "^" + config["symbol"] + r"(?:\.|$)")
            selected = table.filter(pc.fill_null(mask, False))
            if selected.num_rows:
                fragments.append(selected)
            del table
            pa.default_memory_pool().release_unused()
        remote.group_cache = None
    if not fragments:
        raise ValueError("目标证券没有源记录；保持缺失，不填零。")
    return pa.concat_tables(fragments)


def original_profile(table, day, stream):
    dates = pc.unique(table["date"]).to_pylist()
    if dates != [int(day)]:
        raise ValueError("原值日期与目标日期不一致。")
    result = {
        "rows": table.num_rows,
        "source_codes": pc.unique(table["wind_code"]).to_pylist(),
        "columns": table.column_names,
        "first_time": pc.min(table["time"]).as_py(),
        "last_time": pc.max(table["time"]).as_py(),
        "rows_at_or_after_1505": pc.sum(pc.cast(pc.greater_equal(table["time"], 150500000), pa.int64())).as_py(),
    }
    if stream == "行情":
        result["positive_iopv_rows"] = pc.sum(pc.cast(pc.fill_null(pc.greater(table["iopv"], 0), False), pa.int64())).as_py()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = json.loads((BASE / "venvoo/dataset_info.json").read_text(encoding="utf-8"))
    if source["sha"] != config["revision"]:
        raise SystemExit("源清单的固定版本不一致。")
    paths = {item["rfilename"] for item in source["siblings"]}
    days = sorted({path.split("/")[0] for path in paths if len(path.split("/")) == 2
                   and config["date_start"] <= path.split("/")[0] <= config["date_end"]
                   and path.endswith("/行情.parquet")})
    for day in days:
        if any(f"{day}/{stream}.parquet" not in paths for stream in config["streams"]):
            raise SystemExit("区间内有缺失流，必须先明确源缺口。")
    import pandas as pd
    calendar = pd.read_csv(ROOT / "data/reference/sse_trade_calendar_2026.csv")
    expected = [value.replace("-", "") for value in calendar.trade_date if config["date_start"] <= value.replace("-", "") <= config["date_end"]]
    if days != expected:
        raise SystemExit("源目录日期与当前已存交易日历不一致。")
    if not args.download:
        print(json.dumps({"日期数": len(days), "流文件数": len(days) * len(config["streams"]), "配置": config}, ensure_ascii=False))
        return
    token = load_local_token()
    if not token:
        raise SystemExit("本地Hugging Face认证不可用；不会输出或索取明文令牌。")
    OUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUT.resolve()).free < config["new_disk_limit_bytes"] + config["minimum_free_disk_bytes"]:
        raise SystemExit("目标盘空闲空间不足，未启动下载。")
    registration_path = OUT / "registration.json"
    receipt = {
        "registered_at": datetime.now(CN).isoformat(), "config": config, "dates": days,
        "source_manifest_sha256": digest(BASE / "venvoo/dataset_info.json"),
        "calendar_sha256": digest(ROOT / "data/reference/sse_trade_calendar_2026.csv"),
        "config_sha256": digest(CONFIG), "script_sha256": digest(Path(__file__)),
    }
    if registration_path.exists():
        existing = json.loads(registration_path.read_text(encoding="utf-8"))
        if existing["config_sha256"] != receipt["config_sha256"] or existing["script_sha256"] != receipt["script_sha256"]:
            raise SystemExit("登记配置或代码变化，拒绝沿用未完成批次。")
    else:
        save_json(registration_path, receipt)
    reuse = {}
    for path in (BASE / "venvoo_selected").glob("*/receipt.json"):
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("revision") != config["revision"]:
            continue
        for item in old.get("files", []):
            if item.get("status") != "SAVED_RAW_VALUES_SUBSET":
                continue
            parts = item["source_path"].split("/")
            stream = Path(parts[-1]).stem
            if parts[0] not in days or stream not in config["streams"]:
                continue
            local = path.parent / parts[-1]
            if not local.is_file() or digest(local) != item["sha256"]:
                raise SystemExit("已存原值子集发生变化，停止复用。")
            reuse[(parts[0], stream)] = {
                "status": "REUSED_VERIFIED_ORIGINAL", "date": parts[0], "stream": stream,
                "path": str(local.resolve()), "sha256": item["sha256"], "bytes": item["bytes"],
                **original_profile(pq.read_table(local), parts[0], stream),
            }
    previous_attempts = []
    for path in OUT.glob("*/attempt_*.json"):
        previous_attempts.append(json.loads(path.read_text(encoding="utf-8")))
    probe_path = OUT / "header_probe_20260930.json"
    probe_bytes = json.loads(probe_path.read_text(encoding="utf-8"))["metadata_network_bytes"] if probe_path.exists() else 0
    charged = probe_bytes + sum(record.get("network_body_bytes", config["per_file_network_limit_bytes"])
                                for record in previous_attempts)
    disk_used = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    completed = dict(reuse)
    attempts = {}
    for record in previous_attempts:
        key = (record["date"], record["stream"])
        attempts[key] = max(attempts.get(key, 0), record["attempt"])
        if record["status"] == "SAVED_RAW_VALUES_SUBSET":
            local = Path(record["path"])
            if not local.is_file() or digest(local) != record["sha256"]:
                raise SystemExit("本批已完成文件发生变化，拒绝覆盖。")
            completed[key] = record
    lock = threading.Lock()

    def download(key, number):
        nonlocal disk_used
        day, stream = key
        folder = OUT / day
        folder.mkdir(exist_ok=True)
        attempt_path = folder / f"attempt_{stream}_{number}.json"
        record = {"date": day, "stream": stream, "attempt": number, "status": "READING",
                  "started_at": datetime.now(CN).isoformat(),
                  "reserved_network_bytes": config["per_file_network_limit_bytes"]}
        save_json(attempt_path, record)
        budget = {"remaining": config["per_file_network_limit_bytes"]}
        remote = None
        try:
            url = f"https://huggingface.co/datasets/{config['repo']}/resolve/{config['revision']}/" + quote(f"{day}/{stream}.parquet")
            with CachedRangeFile(url, budget, token) as remote:
                parquet = pq.ParquetFile(remote, pre_buffer=False, buffer_size=0)
                groups = candidate_row_groups(parquet, [config["symbol"]])
                record.update(source_bytes=remote.size, selected_row_groups=groups)
                table = extract_selected(parquet, remote, groups, config)
                record.update(original_profile(table, day, stream))
                sink = io.BytesIO()
                pq.write_table(table, sink, compression="zstd")
                content = sink.getvalue()
                target = folder / f"{stream}.parquet"
                with lock:
                    if target.exists():
                        raise ValueError("目标原件已存在，拒绝覆盖。")
                    if disk_used + len(content) + 4 * 1024 * 1024 > config["new_disk_limit_bytes"]:
                        raise ValueError("本批新增落盘将超过上限。")
                    if shutil.disk_usage(OUT.resolve()).free < len(content) + config["minimum_free_disk_bytes"]:
                        raise ValueError("目标盘空闲空间不足。")
                    temporary = target.with_name(target.name + ".writing")
                    temporary.write_bytes(content)
                    temporary.replace(target)
                    disk_used += len(content)
                record.update(status="SAVED_RAW_VALUES_SUBSET", path=str(target.resolve()), bytes=len(content),
                              sha256=hashlib.sha256(content).hexdigest())
        except Exception as error:
            # 网络异常可能含签名地址，只保存类型及明确不含凭据的本地错误。
            safe_message = str(error) if isinstance(error, ValueError) and "http" not in str(error).lower() else type(error).__name__
            record.update(status="FAILED_RETAINED", reason=safe_message[:300])
        finally:
            record.update(finished_at=datetime.now(CN).isoformat(),
                          network_body_bytes=config["per_file_network_limit_bytes"] - budget["remaining"],
                          range_receipts=remote.receipts if remote is not None else [])
            save_json(attempt_path, record)
        return record

    ordered_days = [day for day in config["priority_dates_for_access_and_recent_coverage_check"] if day in days]
    ordered_days += [day for day in days if day not in ordered_days]
    pending = [(day, stream) for day in ordered_days for stream in config["streams"]
               if (day, stream) not in completed and attempts.get((day, stream), 0) < config["maximum_attempts_per_file"]]
    active, reserved = {}, 0
    failed = []
    save_json(OUT / "progress.json", {"status": "RUNNING", "target_dates": len(days), "target_files": len(days) * 3,
                                      "completed_files": len(completed), "reused_files": len(reuse), "network_body_bytes": charged})
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        while pending or active:
            while pending and len(active) < config["workers"]:
                if charged + reserved + config["per_file_network_limit_bytes"] > config["new_network_request_upper_limit_bytes"]:
                    break
                key = pending.pop(0)
                number = attempts.get(key, 0) + 1
                attempts[key] = number
                reserved += config["per_file_network_limit_bytes"]
                active[pool.submit(download, key, number)] = key
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                key = active.pop(future)
                record = future.result()
                reserved -= config["per_file_network_limit_bytes"]
                charged += record["network_body_bytes"]
                if record["status"] == "SAVED_RAW_VALUES_SUBSET":
                    completed[key] = record
                else:
                    failed.append({"date": key[0], "stream": key[1], "attempt": record["attempt"], "reason": record.get("reason")})
                    if attempts[key] < config["maximum_attempts_per_file"]:
                        pending.append(key)
                save_json(OUT / "progress.json", {
                    "status": "RUNNING", "updated_at": datetime.now(CN).isoformat(),
                    "target_dates": len(days), "target_files": len(days) * len(config["streams"]),
                    "completed_files": len(completed), "reused_files": len(reuse),
                    "network_body_bytes": charged, "active_file_reservations": reserved,
                    "new_data_bytes": sum(value["bytes"] for key, value in completed.items() if key not in reuse),
                    "last_file": {"date": record["date"], "stream": record["stream"], "status": record["status"], "rows": record.get("rows")},
                    "failed_attempts": failed,
                })
                print(json.dumps({"已完成文件": len(completed), "目标文件": len(days) * 3,
                                  "日期": record["date"], "流": record["stream"], "状态": record["status"],
                                  "本批网络MiB": round(charged / 1024 ** 2, 2)}, ensure_ascii=False), flush=True)
    manifest = [dict(completed[key], date=key[0], stream=key[1]) for key in sorted(completed)]
    save_json(OUT / "coverage_manifest.json", {"repo": config["repo"], "revision": config["revision"], "files": manifest})
    missing = [{"date": day, "stream": stream} for day in days for stream in config["streams"] if (day, stream) not in completed]
    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    summary = {
        "status": "ALL_TARGET_SOURCE_SUBSETS_SAVED" if not missing else "PARTIAL_SOURCE_SUBSETS_RETAINED",
        "finished_at": datetime.now(CN).isoformat(), "date_start": days[0], "date_end": days[-1],
        "target_dates": len(days), "target_files": len(days) * len(config["streams"]),
        "saved_or_reused_files": len(completed), "reused_files": len(reuse), "missing": missing,
        "complete_three_stream_days": sum(all((day, stream) in completed for stream in config["streams"]) for day in days),
        "source_rows": sum(item["rows"] for item in completed.values()), "new_network_body_bytes": charged,
        "included_metadata_probe_bytes": probe_bytes,
        "new_directory_bytes_before_final_summary": size, "failed_attempts": failed,
        "source_message_completeness_independently_verified": False,
        "historical_receive_clock_established": False, "strategy_returns_computed": False,
        "actual_orders": 0, "paid_services_used": False,
    }
    save_json(OUT / "summary.json", summary)
    save_json(OUT / "progress.json", summary)
    print(json.dumps({"本批结果": summary}, ensure_ascii=False), flush=True)
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
