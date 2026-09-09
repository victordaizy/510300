"""补齐2015—2020官方宏观首发数据并与冻结的2021+输入无缝合并。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time as time_module
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from scripts import download_510300_macro_stress_inputs_v1 as base


ROOT = BOOTSTRAP_ROOT
STUDY_ID = "510300_MACRO_STRESS_AVOIDANCE_2015_START_V2"
EARLY_START = date(2015, 1, 1)
EARLY_END = date(2020, 12, 31)
FULL_START = date(2015, 1, 1)
FULL_END = date(2026, 8, 25)
EARLY_MONTH_START = "2015-01"
EARLY_MONTH_END = "2020-12"
EARLY_TSF_MONTH_START = "2016-01"
FULL_MONTH_END = "2026-07"

OUTPUT_DIR = ROOT / "data" / "raw" / "macro" / "510300_macro_stress_2015_v2"
FDR_PATH = OUTPUT_DIR / "fdr007_daily_2015_2026.parquet"
FX_PATH = OUTPUT_DIR / "usdcny_midpoint_daily_2015_2026.parquet"
PMI_PATH = OUTPUT_DIR / "pmi_new_orders_release_vintage_2015_2026.parquet"
TSF_PATH = OUTPUT_DIR / "tsf_stock_yoy_release_vintage_2015_2026.parquet"
AUDIT_PATH = ROOT / "reports" / "data_quality" / "510300_macro_stress_inputs_2015_v2.json"
ATTEMPT_DIR = (
    ROOT / "reports" / "data_quality" / "510300_macro_stress_inputs_2015_v2_attempts"
)
CHECKPOINT_DIR = OUTPUT_DIR / "acquisition_checkpoints"
MANIFEST_PATH = ROOT / "config" / "510300_macro_stress_avoidance_2015_v2_manifest.json"

FROZEN_V1_INPUTS = {
    "data/raw/macro/fdr007_daily.parquet": (
        "cf46d66001dc15570b0d90dd1a2ad1c03d0c8cdf8d35c97c8f78855f72cb66dc"
    ),
    "data/raw/macro/usdcny_midpoint_daily.parquet": (
        "a3779e83148ea1de13c4755cc8c63cde5e184d6a4983494c8fb7b7a7283e829a"
    ),
    "data/raw/macro/pmi_new_orders_release_vintage.parquet": (
        "ec72ffc6d095eb2430c809d4458e77c227d463474b31ca21b1e3dee3ab929304"
    ),
    "data/raw/macro/tsf_stock_yoy_release_vintage.parquet": (
        "846550bfc7302b6d274d53bdbc3d116e21fed08c1e12b7535f50b092d9d6b83f"
    ),
    "reports/data_quality/510300_macro_stress_inputs_v1.json": (
        "70f9fa5e4ba8a50e829cd5e77c69d480c1d5d7df6c4ec36a2c734d7610a8bd83"
    ),
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@contextmanager
def _base_early_scope(evidence_dir: Path) -> Iterator[None]:
    """仅在本进程内把V1日频采集器约束到2015—2020及独立证据目录。"""

    original = (base.DATA_START, base.DATA_END, base.RAW_EVIDENCE_DIR)
    base.DATA_START = EARLY_START
    base.DATA_END = EARLY_END
    base.RAW_EVIDENCE_DIR = evidence_dir
    try:
        yield
    finally:
        base.DATA_START, base.DATA_END, base.RAW_EVIDENCE_DIR = original


def _expected_months(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start=start, end=end, freq="M")]


def _save_checkpoint(
    stage: str,
    frames: dict[str, pd.DataFrame],
    stage_evidence: list[base.RawEvidence],
    extras: dict[str, Any] | None = None,
) -> None:
    artifacts: dict[str, dict[str, str]] = {}
    for name, frame in frames.items():
        path = CHECKPOINT_DIR / f"{stage}_{name}.parquet"
        base._atomic_parquet(path, frame)
        artifacts[name] = {
            "path": _relative(path),
            "sha256": base.sha256_file(path),
        }
    payload = {
        "status": "PASS",
        "stage": stage,
        "artifacts": artifacts,
        "raw_evidence": [item.__dict__ for item in stage_evidence],
        "extras": extras or {},
        "performance_metrics_computed": False,
        "backtest_run": False,
    }
    base._atomic_json(CHECKPOINT_DIR / f"{stage}.json", payload)


def _load_checkpoint(
    stage: str,
    names: list[str],
) -> tuple[dict[str, pd.DataFrame], list[base.RawEvidence], dict[str, Any]] | None:
    path = CHECKPOINT_DIR / f"{stage}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("stage") != stage:
        raise ValueError(f"采集检查点状态异常：{stage}")
    frames: dict[str, pd.DataFrame] = {}
    for name in names:
        contract = payload.get("artifacts", {}).get(name)
        if not isinstance(contract, dict):
            raise ValueError(f"采集检查点缺少序列：{stage}/{name}")
        artifact_path = ROOT / contract["path"]
        if base.sha256_file(artifact_path) != contract["sha256"]:
            raise ValueError(f"采集检查点哈希漂移：{stage}/{name}")
        frames[name] = pd.read_parquet(artifact_path)
    evidence = [base.RawEvidence(**item) for item in payload.get("raw_evidence", [])]
    return frames, evidence, dict(payload.get("extras", {}))


def _archive_request(
    session: requests.Session,
    *,
    method: str,
    url: str,
    path: Path,
    evidence: list[base.RawEvidence],
    allowed_hosts: set[str],
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    transient_errors = (
        requests.exceptions.SSLError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
    )
    for attempt in range(1, 4):
        try:
            return base._request_and_archive(
                session,
                method=method,
                url=url,
                evidence_path=path,
                evidence=evidence,
                allowed_hosts=allowed_hosts,
                params=params,
                data=data,
                headers=headers,
            )
        except transient_errors:
            if attempt >= 3:
                raise
            print(
                f"官方来源瞬时传输失败，保持TLS校验并进行第{attempt + 1}/3次连接：{url}",
                flush=True,
            )
            time_module.sleep(float(attempt))
    raise AssertionError("官方请求重试流程异常")


def collect_early_fdr007(
    session: requests.Session,
    evidence: list[base.RawEvidence],
    retrieved_at: str,
    evidence_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """采集FDR007；正式推出前的官方---保留为左删失，不作替代或插值。"""

    rows: list[dict[str, Any]] = []
    placeholders: list[date] = []
    per_year: dict[str, dict[str, int]] = {}
    for chunk_start, chunk_end in base._year_chunks(EARLY_START, EARLY_END):
        path = evidence_dir / (
            f"chinamoney_fdr007_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json"
        )
        response = _archive_request(
            session,
            method="POST",
            url=base.CHINAMONEY_FDR_URL,
            path=path,
            evidence=evidence,
            allowed_hosts={"www.chinamoney.com.cn"},
            params={
                "lang": "CN",
                "startDate": chunk_start.isoformat(),
                "endDate": chunk_end.isoformat(),
            },
            headers={
                "Referer": base.CHINAMONEY_FDR_REFERER,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json, text/javascript, */*; q=0.01",
            },
        )
        payload = response.json()
        if str(payload.get("head", {}).get("rep_code")) != "200":
            raise ValueError("中国货币网FDR007接口返回非成功状态")
        records = list(payload.get("records", []))
        if not records:
            raise ValueError(f"中国货币网FDR007分段无记录：{chunk_start}至{chunk_end}")
        raw_hash = base.sha256_bytes(response.content)
        year_key = str(chunk_start.year)
        per_year[year_key] = {"official_rows": len(records), "valid_values": 0, "placeholders": 0}
        for record in records:
            observation_date = pd.Timestamp(record["lfiProducDate"]).date()
            if not EARLY_START <= observation_date <= EARLY_END:
                continue
            raw_value = record.get("frValueMap", {}).get("FDR007")
            if raw_value in (None, "", "---"):
                placeholders.append(observation_date)
                per_year[year_key]["placeholders"] += 1
                continue
            value = float(raw_value)
            per_year[year_key]["valid_values"] += 1
            rows.append(
                {
                    "date": pd.Timestamp(observation_date),
                    "published_at": base._local_timestamp(
                        observation_date, base.time(11, 30)
                    ),
                    "available_at": base._local_timestamp(
                        observation_date, base.time(11, 30)
                    ),
                    "first_release_value": value,
                    "latest_revised_value": value,
                    "revision_number": 0,
                    "source": "中国外汇交易中心暨全国银行间同业拆借中心",
                    "source_url": base.CHINAMONEY_FDR_URL,
                    "source_hash": raw_hash,
                    "raw_path": _relative(path),
                    "retrieved_at": retrieved_at,
                }
            )
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError("FDR007在2015—2020没有任何官方有效值")
    first_valid = pd.Timestamp(frame["date"].min()).date()
    if first_valid != date(2017, 5, 31):
        raise ValueError(f"FDR007首个官方有效日漂移：{first_valid}")
    if any(day >= first_valid for day in placeholders):
        bad = sorted(day for day in placeholders if day >= first_valid)
        raise ValueError(f"FDR007推出后仍有官方占位缺测：{bad[:10]}")
    if len(frame) < 895:
        raise ValueError(f"FDR007推出后的官方有效行数不足：{len(frame)}")
    if not frame["first_release_value"].between(0.1, 20.0).all():
        raise ValueError("FDR007超出冻结的合理值范围")
    censoring = {
        "status": "LEFT_CENSORED_BEFORE_OFFICIAL_LAUNCH",
        "requested_start": EARLY_START.isoformat(),
        "first_official_valid_date": first_valid.isoformat(),
        "pre_launch_placeholder_rows": len(placeholders),
        "pre_launch_placeholder_value": "---",
        "zero_fill_used": False,
        "interpolation_used": False,
        "FR007_proxy_used": False,
        "per_year": per_year,
    }
    return frame, censoring


def collect_early_pmi_new_orders(
    session: requests.Session,
    evidence: list[base.RawEvidence],
    retrieved_at: str,
    evidence_dir: Path,
) -> pd.DataFrame:
    """按月精确检索国家统计局首发正文，采集制造业PMI新订单指数。"""

    rows: list[dict[str, Any]] = []
    expected = _expected_months(EARLY_MONTH_START, EARLY_MONTH_END)
    for number, period in enumerate(expected, start=1):
        month_stage = f"pmi_month_{period}"
        month_checkpoint = _load_checkpoint(month_stage, ["row"])
        if month_checkpoint is not None:
            frames, checkpoint_evidence, _ = month_checkpoint
            rows.append(frames["row"].iloc[0].to_dict())
            evidence.extend(checkpoint_evidence)
            if number % 12 == 0:
                print(
                    f"国家统计局PMI检查点已恢复：{period}（{number}/{len(expected)}）",
                    flush=True,
                )
            continue
        month_evidence_start = len(evidence)
        year, month = (int(value) for value in period.split("-"))
        exact_title = f"{year}年{month}月中国采购经理指数运行情况"
        candidates: list[dict[str, Any]] = []
        target_title_found = False
        queries = [
            (exact_title, 1),
            (f"{year}年{month}月 制造业采购经理指数", 8),
        ]
        for query_number, (query, maximum_pages) in enumerate(queries, start=1):
            for page in range(1, maximum_pages + 1):
                search_path = evidence_dir / (
                    f"nbs_pmi_targeted_{period}_{query_number}_{page:02d}.json"
                )
                response = _archive_request(
                    session,
                    method="POST",
                    url=base.NBS_SEARCH_URL,
                    path=search_path,
                    evidence=evidence,
                    allowed_hosts={"api.so-gov.cn"},
                    data={
                        "siteCode": "bm36000002",
                        "tab": "",
                        "qt": query,
                        "page": str(page),
                        "pageSize": "50",
                        "sort": "relevance",
                        "keyPlace": "1",
                    },
                    headers={
                        "Origin": "https://www.stats.gov.cn",
                        "Referer": base.NBS_SEARCH_REFERER,
                        "Accept": "application/json, text/plain, */*",
                    },
                )
                payload = response.json()
                if not bool(payload.get("ok")):
                    raise ValueError(f"国家统计局官方搜索失败：{period}")
                for rank, item in enumerate(payload.get("resultDocs", []), start=1):
                    data = item.get("data", {})
                    title = base._strip_title(
                        str(data.get("titleO", data.get("title", "")))
                    )
                    valid_title = bool(
                        (
                            "中国制造业采购经理指数为" in title
                            or "中国采购经理指数运行情况" in title
                        )
                        and "解读" not in title
                    )
                    if not valid_title:
                        continue
                    url = str(data.get("url", ""))
                    if url.startswith("http://www.stats.gov.cn/"):
                        url = "https://www.stats.gov.cn/" + url.removeprefix(
                            "http://www.stats.gov.cn/"
                        )
                    if urlparse(url).hostname != "www.stats.gov.cn":
                        continue
                    candidates.append(
                        {
                            "url": url,
                            "timestamp": int(
                                item.get("timestamp", data.get("dreDate", 0)) or 0
                            ),
                            "title": title,
                            "search_order": query_number * 100000 + page * 100 + rank,
                        }
                    )
                    if (
                        title.startswith(f"{year}年{month}月")
                        or title.startswith(f"{month}月中国制造业采购经理指数")
                    ):
                        target_title_found = True
                if target_title_found:
                    break
            if target_title_found:
                break
        if not candidates:
            raise ValueError(f"国家统计局官方搜索缺少PMI首发页：{period}")
        ordered_with_duplicates = sorted(
            candidates,
            key=lambda item: (
                (
                    0
                    if item["title"] == exact_title
                    else 1
                    if item["title"].startswith(f"{year}年{month}月")
                    else 2
                    if item["title"].startswith(
                        f"{month}月中国制造业采购经理指数"
                    )
                    else 9
                ),
                item["search_order"],
                base._nbs_url_priority(item["url"]),
                item["url"],
            ),
        )
        ordered: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in ordered_with_duplicates:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            ordered.append(item)
        selected_payload: dict[str, Any] | None = None
        for candidate_number, selected in enumerate(ordered, start=1):
            url_hash = hashlib.sha256(selected["url"].encode("utf-8")).hexdigest()[:12]
            page_path = evidence_dir / (
                f"nbs_pmi_{period}_{candidate_number:02d}_{url_hash}.html"
            )
            article = _archive_request(
                session,
                method="GET",
                url=selected["url"],
                path=page_path,
                evidence=evidence,
                allowed_hosts={"www.stats.gov.cn"},
                headers={"Referer": "https://www.stats.gov.cn/sj/zxfb/"},
            )
            published = base._parse_nbs_publish_time(
                article.content, int(selected["timestamp"])
            )
            soup = BeautifulSoup(article.content, "lxml")
            text = re.sub(r"\s+", "", soup.get_text(" "))
            explicit_periods = re.findall(
                r"(20\d{2})年(\d{1,2})月份?，?中国制造业采购经理指数",
                text,
            )
            title_has_period = selected["title"].startswith(f"{year}年{month}月")
            content_has_target = bool(
                explicit_periods
                and explicit_periods[0] == (str(year), str(month))
            )
            if not title_has_period and not content_has_target:
                continue
            manufacturing = text.split("二、中国非制造业", maxsplit=1)[0]
            matches = re.findall(
                r"新订单指数(?:为)?([0-9]+(?:\.[0-9]+)?)%", manufacturing
            )
            if not matches:
                continue
            value = float(matches[0])
            # 采购经理指数的定义域是 0—100。极端冲击期可低于 30；
            # 例如国家统计局2020年2月制造业新订单指数首发值为29.3。
            if not 0.0 <= value <= 100.0:
                raise ValueError(
                    f"制造业新订单指数超出定义域：{period}，{value}"
                )
            selected_payload = {
                "article": article,
                "published": published,
                "page_path": page_path,
                "value": value,
            }
            break
        if selected_payload is None:
            raise ValueError(f"国家统计局PMI候选页均非目标首发正文：{period}")
        article = selected_payload["article"]
        published = selected_payload["published"]
        page_path = selected_payload["page_path"]
        value = float(selected_payload["value"])
        row = {
                "reference_period": period,
                "published_at": published.isoformat(),
                "available_at": published.isoformat(),
                "first_release_value": value,
                "latest_revised_value": np.nan,
                "latest_revision_status": "NOT_SEPARATELY_OBSERVED_NOT_USED",
                "revision_number": 0,
                "source": "国家统计局服务业调查中心",
                "source_url": article.url,
                "source_hash": base.sha256_bytes(article.content),
                "raw_path": _relative(page_path),
                "retrieved_at": retrieved_at,
            }
        rows.append(row)
        _save_checkpoint(
            month_stage,
            {"row": pd.DataFrame([row])},
            evidence[month_evidence_start:],
        )
        if number % 12 == 0:
            print(f"国家统计局PMI已完成：{period}（{number}/{len(expected)}）", flush=True)
    frame = pd.DataFrame(rows).sort_values("reference_period").reset_index(drop=True)
    base._validate_monthly_frame(frame, "2015—2020 PMI新订单", expected)
    return frame


def _pbc_search_titles(year: int, month: int) -> list[str]:
    titles = [
        f"{year}年{month}月社会融资规模存量统计数据报告",
        f"{year}年{month}月末社会融资规模存量统计数据报告",
        f"{year}年{month}月金融统计数据报告",
    ]
    period_titles = {
        3: (f"{year}年一季度",),
        6: (f"{year}年上半年",),
        9: (f"{year}年前三季度",),
        12: (f"{year}年",),
    }
    for prefix in period_titles.get(month, ()):
        titles.extend(
            [
                f"{prefix}社会融资规模存量统计数据报告",
                f"{prefix}金融统计数据报告",
            ]
        )
    return titles


def _search_pbc_candidates(
    session: requests.Session,
    evidence: list[base.RawEvidence],
    evidence_dir: Path,
    period: str,
) -> list[dict[str, str]]:
    year, month = (int(value) for value in period.split("-"))
    candidates: dict[str, dict[str, str]] = {}
    for query_number, exact_title in enumerate(_pbc_search_titles(year, month), start=1):
        search_path = evidence_dir / f"pbc_tsf_targeted_{period}_{query_number}.html"
        response = _archive_request(
            session,
            method="GET",
            url=base.PBOC_SEARCH_URL,
            path=search_path,
            evidence=evidence,
            allowed_hosts={"wzdig.pbc.gov.cn"},
            params={
                "pageId": base.PBOC_SEARCH_PAGE_ID,
                "sr": "score desc",
                "pNo": "1",
                "q": exact_title,
            },
            headers={"Referer": "https://www.pbc.gov.cn/"},
        )
        soup = BeautifulSoup(response.content, "lxml")
        for anchor in soup.find_all("a", href=True):
            title = re.sub(r"\s+", "", anchor.get_text("", strip=True))
            if title != exact_title:
                continue
            article_url = urljoin(response.url, str(anchor["href"]))
            if urlparse(article_url).hostname != "www.pbc.gov.cn":
                continue
            candidates[article_url] = {"url": article_url, "title": title}
    return list(candidates.values())


def collect_early_tsf_stock_yoy(
    session: requests.Session,
    evidence: list[base.RawEvidence],
    retrieved_at: str,
    evidence_dir: Path,
) -> pd.DataFrame:
    """按月精确检索人民银行首发正文，采集社融存量同比。"""

    rows: list[dict[str, Any]] = []
    # 人民银行2015年按季公布社融存量，2016年1月起才提高为月度。
    # 冻结因子要求月度参考期，因此不把2015年季度值展开为月度值。
    expected = _expected_months(EARLY_TSF_MONTH_START, EARLY_MONTH_END)
    for month_number, period in enumerate(expected, start=1):
        month_stage = f"tsf_month_{period}"
        month_checkpoint = _load_checkpoint(month_stage, ["row"])
        if month_checkpoint is not None:
            frames, checkpoint_evidence, _ = month_checkpoint
            rows.append(frames["row"].iloc[0].to_dict())
            evidence.extend(checkpoint_evidence)
            if month_number % 12 == 0:
                print(
                    f"人民银行社融检查点已恢复：{period}（{month_number}/{len(expected)}）",
                    flush=True,
                )
            continue
        month_evidence_start = len(evidence)
        candidates = _search_pbc_candidates(session, evidence, evidence_dir, period)
        if not candidates:
            raise ValueError(f"人民银行官方搜索缺少社融首发页：{period}")
        parsed: list[dict[str, Any]] = []
        ordered = sorted(
            candidates,
            key=lambda item: (
                0 if "社会融资规模存量统计数据报告" in item["title"] else 1,
                item["url"],
            ),
        )
        for candidate_number, item in enumerate(ordered, start=1):
            url_hash = hashlib.sha256(item["url"].encode("utf-8")).hexdigest()[:12]
            page_path = evidence_dir / (
                f"pbc_tsf_{period}_{candidate_number:02d}_{url_hash}.html"
            )
            article = _archive_request(
                session,
                method="GET",
                url=item["url"],
                path=page_path,
                evidence=evidence,
                allowed_hosts={"www.pbc.gov.cn"},
                headers={"Referer": base.PBOC_INDEX_URL},
            )
            try:
                published = base._parse_pbc_publish_time(article.content)
                value = base._parse_tsf_stock_yoy(article.content)
            except ValueError:
                continue
            parsed.append(
                {
                    "title": item["title"],
                    "published": published,
                    "value": value,
                    "url": article.url,
                    "hash": base.sha256_bytes(article.content),
                    "path": page_path,
                }
            )
        if not parsed:
            raise ValueError(f"人民银行社融候选页均无法解析：{period}")
        values = {round(float(item["value"]), 10) for item in parsed}
        if len(values) != 1:
            raise ValueError(
                f"人民银行社融首发候选值冲突：{period}，"
                f"{[(item['title'], item['value']) for item in parsed]}"
            )
        selected = sorted(
            parsed,
            key=lambda item: (
                0 if "社会融资规模存量统计数据报告" in item["title"] else 1,
                item["published"],
            ),
        )[0]
        row = {
                "reference_period": period,
                "published_at": selected["published"].isoformat(),
                "available_at": selected["published"].isoformat(),
                "first_release_value": float(selected["value"]),
                "latest_revised_value": np.nan,
                "latest_revision_status": "NOT_SEPARATELY_OBSERVED_NOT_USED",
                "revision_number": 0,
                "source": "中国人民银行调查统计司",
                "source_url": selected["url"],
                "source_hash": selected["hash"],
                "raw_path": _relative(selected["path"]),
                "retrieved_at": retrieved_at,
            }
        rows.append(row)
        _save_checkpoint(
            month_stage,
            {"row": pd.DataFrame([row])},
            evidence[month_evidence_start:],
        )
        if month_number % 12 == 0:
            print(
                f"人民银行社融已完成：{period}（{month_number}/{len(expected)}）",
                flush=True,
            )
    frame = pd.DataFrame(rows).sort_values("reference_period").reset_index(drop=True)
    base._validate_monthly_frame(frame, "2016—2020社融存量同比", expected)
    return frame


def _combine_daily(
    early: pd.DataFrame,
    frozen: pd.DataFrame,
    label: str,
    *,
    latest_allowed_first_date: date,
    minimum_rows: int,
) -> pd.DataFrame:
    left = early.copy()
    right = frozen.copy()
    left["date"] = pd.to_datetime(left["date"]).dt.normalize()
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()
    if left["date"].max() >= right["date"].min():
        raise ValueError(f"{label}早期与冻结区间意外重叠")
    combined = pd.concat([left, right], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    if combined["date"].duplicated().any():
        raise ValueError(f"{label}合并后日期重复")
    if combined["date"].min().date() > latest_allowed_first_date:
        raise ValueError(f"{label}首个有效日过晚")
    if combined["date"].max().date() < FULL_END:
        raise ValueError(f"{label}未覆盖冻结结束日")
    if len(combined) < minimum_rows:
        raise ValueError(f"{label}合并后有效行数不足：{len(combined)}")
    if combined["first_release_value"].isna().any():
        raise ValueError(f"{label}合并后存在首发值空缺")
    return combined


def _combine_monthly(
    early: pd.DataFrame,
    frozen: pd.DataFrame,
    label: str,
    *,
    expected_start: str = EARLY_MONTH_START,
) -> pd.DataFrame:
    combined = pd.concat([early, frozen], ignore_index=True)
    combined = combined.sort_values("reference_period").reset_index(drop=True)
    expected = _expected_months(expected_start, FULL_MONTH_END)
    base._validate_monthly_frame(combined, label, expected)
    return combined


def _series_audit(frame: pd.DataFrame, key: str) -> dict[str, Any]:
    if key == "date":
        first = str(pd.Timestamp(frame[key].min()).date())
        last = str(pd.Timestamp(frame[key].max()).date())
    else:
        first = str(frame[key].min())
        last = str(frame[key].max())
    return {
        "rows": int(len(frame)),
        "first": first,
        "last": last,
        "duplicate_keys": int(frame[key].duplicated().sum()),
        "missing_first_release_values": int(frame["first_release_value"].isna().sum()),
        "minimum": float(frame["first_release_value"].min()),
        "maximum": float(frame["first_release_value"].max()),
    }


def _verify_v1_inputs() -> None:
    actual = {
        relative: base.sha256_file(ROOT / relative)
        for relative in FROZEN_V1_INPUTS
    }
    if actual != FROZEN_V1_INPUTS:
        raise ValueError("冻结的2021+宏观输入或审计哈希漂移")
    audit = json.loads(
        (ROOT / "reports" / "data_quality" / "510300_macro_stress_inputs_v1.json")
        .read_text(encoding="utf-8")
    )
    if audit.get("status") != "PASS":
        raise ValueError("冻结的2021+宏观输入审计不是PASS")


def main() -> int:
    if MANIFEST_PATH.exists():
        raise FileExistsError("2015敏感性研究已经冻结，禁止重建宏观输入")
    for path in (FDR_PATH, FX_PATH, PMI_PATH, TSF_PATH, AUDIT_PATH):
        if path.exists():
            raise FileExistsError(f"通过态输出已存在，拒绝覆盖：{path}")
    _verify_v1_inputs()

    retrieved_at = datetime.now(base.TIME_ZONE).isoformat()
    attempt_id = datetime.now(base.TIME_ZONE).strftime("%Y%m%dT%H%M%S_%z")
    evidence_dir = OUTPUT_DIR / "sources_2015_2020" / attempt_id
    failure_path = ATTEMPT_DIR / f"{attempt_id}.json"
    evidence: list[base.RawEvidence] = []
    session = base._session()
    try:
        daily_checkpoint = _load_checkpoint("daily", ["fdr", "fx"])
        if daily_checkpoint is None:
            stage_start = len(evidence)
            print("开始采集2015—2020 FDR007官方日频数据", flush=True)
            early_fdr, fdr_censoring = collect_early_fdr007(
                session, evidence, retrieved_at, evidence_dir
            )
            with _base_early_scope(evidence_dir):
                print("开始采集2015—2020 USD/CNY官方中间价", flush=True)
                early_fx = base.collect_usdcny_midpoint(session, evidence, retrieved_at)
            _save_checkpoint(
                "daily",
                {"fdr": early_fdr, "fx": early_fx},
                evidence[stage_start:],
                {"fdr_censoring": fdr_censoring},
            )
        else:
            frames, checkpoint_evidence, extras = daily_checkpoint
            early_fdr = frames["fdr"]
            early_fx = frames["fx"]
            fdr_censoring = extras["fdr_censoring"]
            evidence.extend(checkpoint_evidence)
            print("复用已通过的FDR007与USD/CNY采集检查点", flush=True)

        pmi_checkpoint = _load_checkpoint("pmi", ["pmi"])
        if pmi_checkpoint is None:
            stage_start = len(evidence)
            print("开始逐月采集2015—2020国家统计局PMI新订单首发值", flush=True)
            early_pmi = collect_early_pmi_new_orders(
                session, evidence, retrieved_at, evidence_dir
            )
            _save_checkpoint(
                "pmi", {"pmi": early_pmi}, evidence[stage_start:]
            )
        else:
            frames, checkpoint_evidence, _ = pmi_checkpoint
            early_pmi = frames["pmi"]
            evidence.extend(checkpoint_evidence)
            print("复用已通过的PMI采集检查点", flush=True)

        tsf_checkpoint = _load_checkpoint("tsf", ["tsf"])
        if tsf_checkpoint is None:
            stage_start = len(evidence)
            print(
                "2015年社融存量仅按季公布，保持月度因子左删失；"
                "开始逐月采集2016—2020首发值",
                flush=True,
            )
            early_tsf = collect_early_tsf_stock_yoy(
                session, evidence, retrieved_at, evidence_dir
            )
            _save_checkpoint(
                "tsf", {"tsf": early_tsf}, evidence[stage_start:]
            )
        else:
            frames, checkpoint_evidence, _ = tsf_checkpoint
            early_tsf = frames["tsf"]
            evidence.extend(checkpoint_evidence)
            print("复用已通过的社融采集检查点", flush=True)
    except Exception as error:
        failure = {
            "status": "EXTERNAL_FREE_SOURCE_FAILED",
            "study_id": STUDY_ID,
            "attempt_id": attempt_id,
            "data_start": FULL_START.isoformat(),
            "data_end": FULL_END.isoformat(),
            "retrieved_at": retrieved_at,
            "tls_verification": "ENABLED_NO_BYPASS",
            "error_type": type(error).__name__,
            "error": str(error),
            "raw_evidence_count": len(evidence),
            "raw_evidence": [item.__dict__ for item in evidence],
            "governance": {
                "proxy_substitution_used": False,
                "backtest_run": False,
                "live_trading_authorized": False,
            },
        }
        base._atomic_json(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        raise
    finally:
        session.close()

    frozen_fdr = pd.read_parquet(ROOT / "data" / "raw" / "macro" / "fdr007_daily.parquet")
    frozen_fx = pd.read_parquet(
        ROOT / "data" / "raw" / "macro" / "usdcny_midpoint_daily.parquet"
    )
    frozen_pmi = pd.read_parquet(
        ROOT / "data" / "raw" / "macro" / "pmi_new_orders_release_vintage.parquet"
    )
    frozen_tsf = pd.read_parquet(
        ROOT / "data" / "raw" / "macro" / "tsf_stock_yoy_release_vintage.parquet"
    )
    fdr = _combine_daily(
        early_fdr,
        frozen_fdr,
        "FDR007",
        latest_allowed_first_date=date(2017, 5, 31),
        minimum_rows=2200,
    )
    fx = _combine_daily(
        early_fx,
        frozen_fx,
        "USD/CNY中间价",
        latest_allowed_first_date=date(2015, 1, 8),
        minimum_rows=2750,
    )
    pmi = _combine_monthly(early_pmi, frozen_pmi, "PMI新订单")
    tsf = _combine_monthly(
        early_tsf,
        frozen_tsf,
        "社融存量同比",
        expected_start=EARLY_TSF_MONTH_START,
    )

    base._atomic_parquet(FDR_PATH, fdr)
    base._atomic_parquet(FX_PATH, fx)
    base._atomic_parquet(PMI_PATH, pmi)
    base._atomic_parquet(TSF_PATH, tsf)
    output_hashes = {
        _relative(path): base.sha256_file(path)
        for path in (FDR_PATH, FX_PATH, PMI_PATH, TSF_PATH)
    }
    audit = {
        "status": "PASS",
        "study_id": STUDY_ID,
        "evidence_class": "POST_REJECTION_WINDOW_SENSITIVITY_ONLY",
        "scope": "2015-01-01至2026-08-25；2015—2020官方补采加冻结2021+输入",
        "data_start": FULL_START.isoformat(),
        "data_end": FULL_END.isoformat(),
        "latest_complete_month": FULL_MONTH_END,
        "retrieved_at": retrieved_at,
        "successful_attempt_id": attempt_id,
        "tls_verification": "ENABLED_NO_BYPASS",
        "proxy_substitution_used": False,
        "fdr007_left_censoring": fdr_censoring,
        "tsf_stock_yoy_left_censoring": {
            "official_2015_frequency": "QUARTERLY",
            "frozen_factor_required_frequency": "MONTHLY",
            "first_monthly_reference_period": EARLY_TSF_MONTH_START,
            "handling": (
                "LEFT_CENSORED_NO_QUARTERLY_TO_MONTHLY_EXPANSION_"
                "NO_INTERPOLATION_NO_PROXY"
            ),
        },
        "frozen_2021_plus_input_hashes": FROZEN_V1_INPUTS,
        "series": {
            "fdr007": _series_audit(fdr, "date"),
            "usdcny_midpoint": _series_audit(fx, "date"),
            "pmi_new_orders": _series_audit(pmi, "reference_period"),
            "tsf_stock_yoy": _series_audit(tsf, "reference_period"),
        },
        "output_hashes": output_hashes,
        "new_raw_evidence_count": len(evidence),
        "new_raw_evidence": [item.__dict__ for item in evidence],
        "governance": {
            "future_return_labels_computed": False,
            "performance_metrics_computed": False,
            "backtest_run": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    base._atomic_json(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "series": audit["series"],
                "output_hashes": output_hashes,
                "audit": _relative(AUDIT_PATH),
                "new_raw_evidence_count": len(evidence),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
