"""下载并审计510300宏观压力规避V1所需的2021年以来官方数据。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DATA_START = date(2021, 1, 1)
DATA_END = date(2026, 8, 25)
LATEST_COMPLETE_MONTH = "2026-07"
TIME_ZONE = ZoneInfo("Asia/Shanghai")

RAW_MACRO_DIR = ROOT / "data" / "raw" / "macro"
RAW_EVIDENCE_DIR = RAW_MACRO_DIR / "510300_macro_stress_avoidance_v1_sources"
FDR_PATH = RAW_MACRO_DIR / "fdr007_daily.parquet"
PMI_PATH = RAW_MACRO_DIR / "pmi_new_orders_release_vintage.parquet"
TSF_PATH = RAW_MACRO_DIR / "tsf_stock_yoy_release_vintage.parquet"
FX_PATH = RAW_MACRO_DIR / "usdcny_midpoint_daily.parquet"
AUDIT_PATH = ROOT / "reports" / "data_quality" / "510300_macro_stress_inputs_v1.json"
FREEZE_MANIFEST_PATH = ROOT / "config" / "510300_macro_stress_avoidance_v1_manifest.json"

CHINAMONEY_FDR_URL = (
    "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-currency/FrrHis"
)
CHINAMONEY_FX_URL = (
    "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew"
)
CHINAMONEY_FDR_REFERER = "https://www.chinamoney.com.cn/chinese/bkfrr/"
CHINAMONEY_FX_REFERER = "https://www.chinamoney.com.cn/chinese/bkccpr/"
NBS_SEARCH_URL = "https://api.so-gov.cn/query/s"
NBS_SEARCH_REFERER = "https://www.stats.gov.cn/search/s"
PBOC_INDEX_URL = (
    "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/index.html"
)
PBOC_PAGE_URL = (
    "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/11871-{page}.html"
)
PBOC_SEARCH_URL = "https://wzdig.pbc.gov.cn/search/pcRender"
PBOC_SEARCH_PAGE_ID = "c177a85bd02b4114bebebd210809f691"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/139.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class RawEvidence:
    """单个官方响应的不可变证据摘要。"""

    path: str
    url: str
    method: str
    http_status: int
    content_type: str
    size_bytes: int
    sha256: str


def sha256_bytes(content: bytes) -> str:
    """计算字节内容的SHA-256。"""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """在同一目录中原子写入文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        file.write(content)
        file.flush()
    temporary_path.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入UTF-8 JSON。"""

    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, content)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """原子写入Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp.parquet",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
    try:
        frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _json_default(value: Any) -> Any:
    """将Pandas、NumPy和日期类型转换为JSON可序列化值。"""

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"不支持的JSON类型：{type(value)!r}")


def _relative(path: Path) -> str:
    """返回项目根目录相对POSIX路径。"""

    return path.relative_to(ROOT).as_posix()


def _session() -> requests.Session:
    """创建启用系统证书校验的HTTP会话。"""

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.6",
        }
    )
    return session


def _validate_official_url(url: str, allowed_hosts: Iterable[str]) -> None:
    """禁止采集器被重定向到未冻结的非官方域名。"""

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in set(allowed_hosts):
        raise ValueError(f"非冻结官方地址：{url}")


def _request_and_archive(
    session: requests.Session,
    *,
    method: str,
    url: str,
    evidence_path: Path,
    evidence: list[RawEvidence],
    allowed_hosts: Iterable[str],
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """请求官方来源、验证最终域名并保存原始响应。"""

    _validate_official_url(url, allowed_hosts)
    response = session.request(
        method=method,
        url=url,
        params=params,
        data=data,
        headers=headers,
        timeout=45,
        allow_redirects=True,
    )
    response.raise_for_status()
    _validate_official_url(response.url, allowed_hosts)
    content = response.content
    _atomic_write_bytes(evidence_path, content)
    evidence.append(
        RawEvidence(
            path=_relative(evidence_path),
            url=response.url,
            method=method.upper(),
            http_status=int(response.status_code),
            content_type=str(response.headers.get("Content-Type", "")),
            size_bytes=len(content),
            sha256=sha256_bytes(content),
        )
    )
    return response


def _local_timestamp(day: date, clock: time) -> str:
    """形成带上海时区的ISO时间戳。"""

    return datetime.combine(day, clock, tzinfo=TIME_ZONE).isoformat()


def collect_fdr007(
    session: requests.Session,
    evidence: list[RawEvidence],
    retrieved_at: str,
) -> pd.DataFrame:
    """从中国货币网官方接口采集FDR007。"""

    rows: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks(DATA_START, DATA_END):
        path = RAW_EVIDENCE_DIR / (
            f"chinamoney_fdr007_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json"
        )
        response = _request_and_archive(
            session,
            method="POST",
            url=CHINAMONEY_FDR_URL,
            evidence_path=path,
            evidence=evidence,
            allowed_hosts={"www.chinamoney.com.cn"},
            params={
                "lang": "CN",
                "startDate": chunk_start.isoformat(),
                "endDate": chunk_end.isoformat(),
            },
            headers={
                "Referer": CHINAMONEY_FDR_REFERER,
                "X-Requested-With": "XMLHttpRequest",
                "Accept": "application/json, text/javascript, */*; q=0.01",
            },
        )
        payload = response.json()
        if str(payload.get("head", {}).get("rep_code")) != "200":
            raise ValueError("中国货币网FDR007接口返回非成功状态")
        records = list(payload.get("records", []))
        if not records:
            message = str(payload.get("data", {}).get("message", "无错误说明"))
            raise ValueError(
                f"中国货币网FDR007分段无记录：{chunk_start}至{chunk_end}，{message}"
            )
        raw_hash = sha256_bytes(response.content)
        for record in records:
            observation_date = pd.Timestamp(record["lfiProducDate"]).date()
            if not DATA_START <= observation_date <= DATA_END:
                continue
            raw_value = record.get("frValueMap", {}).get("FDR007")
            if raw_value in (None, ""):
                raise ValueError(f"FDR007缺值：{observation_date}")
            value = float(raw_value)
            rows.append(
                {
                    "date": pd.Timestamp(observation_date),
                    "published_at": _local_timestamp(observation_date, time(11, 30)),
                    "available_at": _local_timestamp(observation_date, time(11, 30)),
                    "first_release_value": value,
                    "latest_revised_value": value,
                    "revision_number": 0,
                    "source": "中国外汇交易中心暨全国银行间同业拆借中心",
                    "source_url": CHINAMONEY_FDR_URL,
                    "source_hash": raw_hash,
                    "raw_path": _relative(path),
                    "retrieved_at": retrieved_at,
                }
            )
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    _validate_daily_frame(frame, "FDR007", minimum_rows=1300)
    if not frame["first_release_value"].between(0.1, 20.0).all():
        raise ValueError("FDR007超出冻结的合理值范围")
    return frame


def collect_usdcny_midpoint(
    session: requests.Session,
    evidence: list[RawEvidence],
    retrieved_at: str,
) -> pd.DataFrame:
    """从中国货币网官方接口采集美元兑人民币中间价。"""

    session.get(CHINAMONEY_FX_REFERER, timeout=45).raise_for_status()
    # 官方网页默认10行；实测官方WAF允许50行、拒绝100行及以上。
    page_size = 50
    rows: list[dict[str, Any]] = []
    for chunk_start, chunk_end in _year_chunks(DATA_START, DATA_END):
        page = 1
        while True:
            path = RAW_EVIDENCE_DIR / (
                f"chinamoney_usdcny_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}_"
                f"page_{page:02d}.json"
            )
            response = _request_and_archive(
                session,
                method="POST",
                url=CHINAMONEY_FX_URL,
                evidence_path=path,
                evidence=evidence,
                allowed_hosts={"www.chinamoney.com.cn"},
                params={
                    "startDate": chunk_start.isoformat(),
                    "endDate": chunk_end.isoformat(),
                    "pageNum": page,
                    "pageSize": page_size,
                },
                headers={
                    "Referer": CHINAMONEY_FX_REFERER,
                    "Origin": "https://www.chinamoney.com.cn",
                    "X-Requested-With": "XMLHttpRequest",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                },
            )
            payload = response.json()
            if str(payload.get("head", {}).get("rep_code")) != "200":
                raise ValueError("中国货币网汇率接口返回非成功状态")
            data = payload.get("data", {})
            head = list(data.get("head", []))
            if "USD/CNY" not in head:
                raise ValueError("中国货币网汇率接口缺少USD/CNY列")
            usd_index = head.index("USD/CNY")
            records = list(payload.get("records", []))
            if page == 1 and not records:
                message = str(data.get("flagMessage", "无错误说明"))
                raise ValueError(
                    f"中国货币网汇率分段无记录：{chunk_start}至{chunk_end}，{message}"
                )
            raw_hash = sha256_bytes(response.content)
            for record in records:
                observation_date = pd.Timestamp(record["date"]).date()
                if not DATA_START <= observation_date <= DATA_END:
                    continue
                values = list(record.get("values", []))
                if usd_index >= len(values) or values[usd_index] in (None, ""):
                    raise ValueError(f"USD/CNY中间价缺值：{observation_date}")
                value = float(values[usd_index])
                rows.append(
                    {
                        "date": pd.Timestamp(observation_date),
                        "published_at": _local_timestamp(observation_date, time(9, 15)),
                        "available_at": _local_timestamp(observation_date, time(9, 15)),
                        "first_release_value": value,
                        "latest_revised_value": value,
                        "revision_number": 0,
                        "source": "中国外汇交易中心暨全国银行间同业拆借中心",
                        "source_url": CHINAMONEY_FX_URL,
                        "source_hash": raw_hash,
                        "raw_path": _relative(path),
                        "retrieved_at": retrieved_at,
                    }
                )
            page_total = int(data.get("pageTotal", 1))
            if page >= page_total:
                break
            page += 1
    frame = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    _validate_daily_frame(frame, "USD/CNY中间价", minimum_rows=1300)
    if not frame["first_release_value"].between(4.0, 10.0).all():
        raise ValueError("USD/CNY中间价超出冻结的合理值范围")
    return frame


def _strip_title(value: str) -> str:
    """移除搜索结果标题中的高亮标签和空白。"""

    return re.sub(r"\s+", "", BeautifulSoup(value, "lxml").get_text(" "))


def _parse_nbs_publish_time(content: bytes, fallback_millis: int) -> datetime:
    """从国家统计局首发页提取发布时间。"""

    text = content.decode("utf-8", errors="replace")
    match = re.search(
        r"(20\d{2})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})",
        text,
    )
    if match:
        values = [int(value) for value in match.groups()]
        return datetime(*values, tzinfo=TIME_ZONE)
    match = re.search(
        r'<meta\s+name=["\']?createDate["\']?\s+content=["\']([^"\']+)',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        parsed = pd.Timestamp(match.group(1)).to_pydatetime()
        return parsed.replace(tzinfo=TIME_ZONE)
    if fallback_millis <= 0:
        raise ValueError("国家统计局页面缺少可用发布时间")
    return datetime.fromtimestamp(fallback_millis / 1000.0, tz=TIME_ZONE)


def _parse_pmi_new_orders(content: bytes) -> float:
    """从国家统计局首发正文提取制造业新订单指数。"""

    soup = BeautifulSoup(content, "lxml")
    text = re.sub(r"\s+", "", soup.get_text(" "))
    manufacturing = text.split("二、中国非制造业", maxsplit=1)[0]
    matches = re.findall(r"新订单指数为([0-9]+(?:\.[0-9]+)?)%", manufacturing)
    if not matches:
        raise ValueError("国家统计局首发页未找到制造业新订单指数")
    value = float(matches[0])
    if not 30.0 <= value <= 70.0:
        raise ValueError("制造业新订单指数超出冻结的合理值范围")
    return value


def _nbs_url_priority(url: str) -> tuple[int, str]:
    """优先使用统计局数据发布正文，其次使用聚合和信息公开副本。"""

    if "/sj/zxfb/" in url:
        return (0, url)
    if "/sj/zxfbhjd/" in url:
        return (1, url)
    if "/xxgk/sjfb/zxfb2020/" in url:
        return (2, url)
    return (9, url)


def collect_pmi_new_orders(
    session: requests.Session,
    evidence: list[RawEvidence],
    retrieved_at: str,
) -> pd.DataFrame:
    """从国家统计局逐月首发页采集制造业PMI新订单指数。"""

    candidates: dict[str, list[dict[str, Any]]] = {}
    page = 1
    page_size = 100
    total_hits = 1
    while (page - 1) * page_size < total_hits:
        path = RAW_EVIDENCE_DIR / f"nbs_pmi_search_page_{page:02d}.json"
        response = _request_and_archive(
            session,
            method="POST",
            url=NBS_SEARCH_URL,
            evidence_path=path,
            evidence=evidence,
            allowed_hosts={"api.so-gov.cn"},
            data={
                "siteCode": "bm36000002",
                "tab": "",
                "qt": "中国采购经理指数运行情况",
                "page": str(page),
                "pageSize": str(page_size),
                "sort": "CUSTOM:DOCOPENDATE:DESC",
                "keyPlace": "1",
            },
            headers={
                "Origin": "https://www.stats.gov.cn",
                "Referer": NBS_SEARCH_REFERER,
                "Accept": "application/json, text/plain, */*",
            },
        )
        payload = response.json()
        if not bool(payload.get("ok")):
            raise ValueError("国家统计局官方搜索接口返回失败")
        total_hits = int(payload.get("totalHits", 0))
        for item in payload.get("resultDocs", []):
            data = item.get("data", {})
            title = _strip_title(str(data.get("titleO", data.get("title", ""))))
            match = re.fullmatch(r"(20\d{2})年(\d{1,2})月中国采购经理指数运行情况", title)
            if not match:
                continue
            year, month = int(match.group(1)), int(match.group(2))
            period = f"{year:04d}-{month:02d}"
            if period < "2021-01" or period > LATEST_COMPLETE_MONTH:
                continue
            url = str(data.get("url", ""))
            if url.startswith("http://www.stats.gov.cn/"):
                url = "https://www.stats.gov.cn/" + url.removeprefix(
                    "http://www.stats.gov.cn/"
                )
            if urlparse(url).hostname != "www.stats.gov.cn":
                continue
            candidates.setdefault(period, []).append(
                {
                    "url": url,
                    "timestamp": int(item.get("timestamp", data.get("dreDate", 0)) or 0),
                    "title": title,
                }
            )
        page += 1
    expected = _expected_months("2021-01", LATEST_COMPLETE_MONTH)
    # 批量按发布日期排序时，少数历史正文会被大量解读文章挤出结果窗口。
    # 对明确缺失月使用同一官方接口做标题精确、相关度优先的定向检索。
    for period in expected:
        if period in candidates:
            continue
        year, month = (int(value) for value in period.split("-"))
        exact_title = f"{year}年{month}月中国采购经理指数运行情况"
        path = RAW_EVIDENCE_DIR / f"nbs_pmi_targeted_{period}.json"
        response = _request_and_archive(
            session,
            method="POST",
            url=NBS_SEARCH_URL,
            evidence_path=path,
            evidence=evidence,
            allowed_hosts={"api.so-gov.cn"},
            data={
                "siteCode": "bm36000002",
                "tab": "",
                "qt": exact_title,
                "page": "1",
                "pageSize": "50",
                "sort": "relevance",
                "keyPlace": "1",
            },
            headers={
                "Origin": "https://www.stats.gov.cn",
                "Referer": NBS_SEARCH_REFERER,
                "Accept": "application/json, text/plain, */*",
            },
        )
        payload = response.json()
        for item in payload.get("resultDocs", []):
            data = item.get("data", {})
            title = _strip_title(str(data.get("titleO", data.get("title", ""))))
            if title != exact_title:
                continue
            url = str(data.get("url", ""))
            if url.startswith("http://www.stats.gov.cn/"):
                url = "https://www.stats.gov.cn/" + url.removeprefix(
                    "http://www.stats.gov.cn/"
                )
            if urlparse(url).hostname != "www.stats.gov.cn":
                continue
            candidates.setdefault(period, []).append(
                {
                    "url": url,
                    "timestamp": int(item.get("timestamp", data.get("dreDate", 0)) or 0),
                    "title": title,
                }
            )
    missing_candidates = sorted(set(expected) - set(candidates))
    if missing_candidates:
        raise ValueError(f"国家统计局搜索缺少PMI月份：{missing_candidates}")

    rows: list[dict[str, Any]] = []
    for period in expected:
        selected = sorted(candidates[period], key=lambda item: _nbs_url_priority(item["url"]))[0]
        url_hash = hashlib.sha256(selected["url"].encode("utf-8")).hexdigest()[:12]
        path = RAW_EVIDENCE_DIR / f"nbs_pmi_{period}_{url_hash}.html"
        response = _request_and_archive(
            session,
            method="GET",
            url=selected["url"],
            evidence_path=path,
            evidence=evidence,
            allowed_hosts={"www.stats.gov.cn"},
            headers={"Referer": "https://www.stats.gov.cn/sj/zxfb/"},
        )
        published = _parse_nbs_publish_time(response.content, selected["timestamp"])
        value = _parse_pmi_new_orders(response.content)
        raw_hash = sha256_bytes(response.content)
        rows.append(
            {
                "reference_period": period,
                "published_at": published.isoformat(),
                "available_at": published.isoformat(),
                "first_release_value": value,
                "latest_revised_value": np.nan,
                "latest_revision_status": "NOT_SEPARATELY_OBSERVED_NOT_USED",
                "revision_number": 0,
                "source": "国家统计局服务业调查中心",
                "source_url": response.url,
                "source_hash": raw_hash,
                "raw_path": _relative(path),
                "retrieved_at": retrieved_at,
            }
        )
    frame = pd.DataFrame(rows).sort_values("reference_period").reset_index(drop=True)
    _validate_monthly_frame(frame, "PMI新订单", expected)
    return frame


def _infer_reference_period_from_title(title: str) -> str | None:
    """从央行月报标题推断其统计参考月。"""

    compact = re.sub(r"\s+", "", title)
    year_match = re.search(r"(20\d{2})年", compact)
    if not year_match:
        return None
    year = int(year_match.group(1))
    month_match = re.search(r"(\d{1,2})月", compact)
    if month_match:
        month = int(month_match.group(1))
    elif "一季度" in compact:
        month = 3
    elif "上半年" in compact:
        month = 6
    elif "前三季度" in compact:
        month = 9
    else:
        month = 12
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}"


def _parse_pbc_publish_time(content: bytes) -> datetime:
    """从人民银行文章提取精确发布时间。"""

    text = content.decode("utf-8", errors="replace")
    patterns = [
        r'id=["\']shijian["\'][^>]*>\s*(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
        (
            r'<!--\s*20\d{2}年\d{2}月\d{2}日\s*-->\s*'
            r'(20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
        ),
        (
            r'<meta\s+name=["\']?createDate["\']?\s+'
            r'content=["\']([^"\']+)'
        ),
    ]
    match = None
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            break
    if not match:
        raise ValueError("人民银行文章缺少精确发布时间")
    parsed = pd.Timestamp(match.group(1)).to_pydatetime()
    return parsed.replace(tzinfo=TIME_ZONE)


def _parse_tsf_stock_yoy(content: bytes) -> float:
    """从人民银行首发正文提取社会融资规模存量同比。"""

    soup = BeautifulSoup(content, "lxml")
    zoom = soup.select_one("#zoom")
    text = (zoom or soup).get_text(" ")
    compact = re.sub(r"\s+", "", text)
    patterns = [
        r"社会融资规模存量(?:为[0-9.]+万亿元)?[，,]同比增长([0-9]+(?:\.[0-9]+)?)%",
        r"社会融资规模存量同比增长([0-9]+(?:\.[0-9]+)?)%",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            value = float(match.group(1))
            if not 0.0 <= value <= 30.0:
                raise ValueError("社会融资规模存量同比超出冻结的合理值范围")
            return value
    raise ValueError("人民银行首发页未找到社会融资规模存量同比")


def collect_tsf_stock_yoy(
    session: requests.Session,
    evidence: list[RawEvidence],
    retrieved_at: str,
) -> pd.DataFrame:
    """从人民银行历史栏目逐月采集社融存量同比首发值。"""

    candidates: dict[str, list[dict[str, str]]] = {}
    for page in range(1, 13):
        url = PBOC_INDEX_URL if page == 1 else PBOC_PAGE_URL.format(page=page)
        path = RAW_EVIDENCE_DIR / f"pbc_statistics_index_page_{page:02d}.html"
        response = _request_and_archive(
            session,
            method="GET",
            url=url,
            evidence_path=path,
            evidence=evidence,
            allowed_hosts={"www.pbc.gov.cn"},
            headers={"Referer": PBOC_INDEX_URL},
        )
        soup = BeautifulSoup(response.content, "lxml")
        for anchor in soup.find_all("a", href=True):
            title = str(anchor.get("title") or anchor.get_text(" ", strip=True))
            if not (
                "社会融资规模存量统计数据报告" in title
                or "金融统计数据报告" in title
            ):
                continue
            period = _infer_reference_period_from_title(title)
            if period is None or period < "2021-01" or period > LATEST_COMPLETE_MONTH:
                continue
            article_url = urljoin("https://www.pbc.gov.cn", str(anchor["href"]))
            if urlparse(article_url).hostname != "www.pbc.gov.cn":
                continue
            candidates.setdefault(period, []).append(
                {"url": article_url, "title": title}
            )
    expected = _expected_months("2021-01", LATEST_COMPLETE_MONTH)
    missing_candidates = sorted(set(expected) - set(candidates))
    for period in missing_candidates:
        year, month = (int(part) for part in period.split("-"))
        exact_title = f"{year}年{month}月社会融资规模存量统计数据报告"
        path = RAW_EVIDENCE_DIR / f"pbc_tsf_targeted_search_{period}.html"
        response = _request_and_archive(
            session,
            method="GET",
            url=PBOC_SEARCH_URL,
            evidence_path=path,
            evidence=evidence,
            allowed_hosts={"wzdig.pbc.gov.cn"},
            params={
                "pageId": PBOC_SEARCH_PAGE_ID,
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
            candidates.setdefault(period, []).append(
                {"url": article_url, "title": title}
            )
    missing_candidates = sorted(set(expected) - set(candidates))
    if missing_candidates:
        raise ValueError(
            f"人民银行历史栏目与官方搜索均缺少社融月份：{missing_candidates}"
        )

    rows: list[dict[str, Any]] = []
    for period in expected:
        unique: dict[str, dict[str, str]] = {
            item["url"]: item for item in candidates[period]
        }
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                0 if "社会融资规模存量统计数据报告" in item["title"] else 1,
                item["url"],
            ),
        )
        parsed_candidates: list[dict[str, Any]] = []
        for number, item in enumerate(ordered, start=1):
            url_hash = hashlib.sha256(item["url"].encode("utf-8")).hexdigest()[:12]
            path = RAW_EVIDENCE_DIR / f"pbc_tsf_{period}_{number:02d}_{url_hash}.html"
            response = _request_and_archive(
                session,
                method="GET",
                url=item["url"],
                evidence_path=path,
                evidence=evidence,
                allowed_hosts={"www.pbc.gov.cn"},
                headers={"Referer": PBOC_INDEX_URL},
            )
            try:
                published = _parse_pbc_publish_time(response.content)
                value = _parse_tsf_stock_yoy(response.content)
            except ValueError:
                continue
            parsed_candidates.append(
                {
                    "title": item["title"],
                    "published": published,
                    "value": value,
                    "url": response.url,
                    "hash": sha256_bytes(response.content),
                    "path": path,
                }
            )
        if not parsed_candidates:
            raise ValueError(f"社融月份{period}的候选文章均无法解析")
        values = {round(float(item["value"]), 10) for item in parsed_candidates}
        if len(values) != 1:
            details = [(item["title"], item["value"]) for item in parsed_candidates]
            raise ValueError(f"社融月份{period}的同日首发值冲突：{details}")
        selected = sorted(
            parsed_candidates,
            key=lambda item: (
                0 if "社会融资规模存量统计数据报告" in item["title"] else 1,
                item["published"],
            ),
        )[0]
        rows.append(
            {
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
        )
    frame = pd.DataFrame(rows).sort_values("reference_period").reset_index(drop=True)
    _validate_monthly_frame(frame, "社会融资规模存量同比", expected)
    return frame


def _expected_months(start: str, end: str) -> list[str]:
    """构造闭区间自然月列表。"""

    return [str(period) for period in pd.period_range(start=start, end=end, freq="M")]


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """按自然年切分中国货币网最长一年查询区间。"""

    chunks: list[tuple[date, date]] = []
    current = start
    while current <= end:
        chunk_end = min(date(current.year, 12, 31), end)
        chunks.append((current, chunk_end))
        current = date(current.year + 1, 1, 1)
    return chunks


def _validate_daily_frame(
    frame: pd.DataFrame,
    label: str,
    *,
    minimum_rows: int,
) -> None:
    """执行日频官方数据的结构与覆盖硬门。"""

    required = {
        "date",
        "published_at",
        "available_at",
        "first_release_value",
        "source_hash",
        "raw_path",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")
    if len(frame) < minimum_rows:
        raise ValueError(f"{label}有效行数不足：{len(frame)} < {minimum_rows}")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError(f"{label}日期重复或未递增")
    if frame["first_release_value"].isna().any():
        raise ValueError(f"{label}存在空值")
    first_date = pd.Timestamp(frame["date"].min()).date()
    last_date = pd.Timestamp(frame["date"].max()).date()
    if first_date > date(2021, 1, 8) or last_date < DATA_END:
        raise ValueError(f"{label}覆盖不足：{first_date}至{last_date}")


def _validate_monthly_frame(
    frame: pd.DataFrame,
    label: str,
    expected: list[str],
) -> None:
    """执行月频首发数据的结构与完整月份硬门。"""

    required = {
        "reference_period",
        "published_at",
        "available_at",
        "first_release_value",
        "source_hash",
        "raw_path",
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"{label}缺少字段：{sorted(missing)}")
    actual = frame["reference_period"].astype(str).tolist()
    if actual != expected:
        raise ValueError(f"{label}月份不完整或顺序错误")
    if frame["reference_period"].duplicated().any():
        raise ValueError(f"{label}参考月重复")
    if frame["first_release_value"].isna().any():
        raise ValueError(f"{label}首发值存在空值")
    published = pd.to_datetime(frame["published_at"], utc=True, errors="raise")
    periods = pd.PeriodIndex(frame["reference_period"], freq="M")
    local_published = published.dt.tz_convert(TIME_ZONE).dt.tz_localize(None)
    if any(
        timestamp.to_period("M") < period
        for timestamp, period in zip(local_published, periods, strict=True)
    ):
        raise ValueError(f"{label}发布时间早于参考月")


def _series_audit(frame: pd.DataFrame, date_column: str) -> dict[str, Any]:
    """形成单序列覆盖审计。"""

    if date_column == "date":
        first = str(pd.Timestamp(frame[date_column].min()).date())
        last = str(pd.Timestamp(frame[date_column].max()).date())
    else:
        first = str(frame[date_column].min())
        last = str(frame[date_column].max())
    return {
        "rows": int(len(frame)),
        "first": first,
        "last": last,
        "duplicate_keys": int(frame[date_column].duplicated().sum()),
        "missing_first_release_values": int(frame["first_release_value"].isna().sum()),
        "minimum": float(frame["first_release_value"].min()),
        "maximum": float(frame["first_release_value"].max()),
    }


def main() -> int:
    """执行一次官方采集并写入审计工件。"""

    if FREEZE_MANIFEST_PATH.exists():
        raise FileExistsError("策略已经冻结，禁止重跑或覆盖宏观输入")
    retrieved_at = datetime.now(TIME_ZONE).isoformat()
    evidence: list[RawEvidence] = []
    session = _session()
    try:
        fdr = collect_fdr007(session, evidence, retrieved_at)
        fx = collect_usdcny_midpoint(session, evidence, retrieved_at)
        pmi = collect_pmi_new_orders(session, evidence, retrieved_at)
        tsf = collect_tsf_stock_yoy(session, evidence, retrieved_at)
    except Exception as error:
        failure = {
            "status": "EXTERNAL_FREE_SOURCE_FAILED",
            "study_id": "510300_MACRO_STRESS_AVOIDANCE_V1",
            "data_start": DATA_START.isoformat(),
            "data_end": DATA_END.isoformat(),
            "retrieved_at": retrieved_at,
            "tls_verification": "ENABLED_NO_BYPASS",
            "error_type": type(error).__name__,
            "error": str(error),
            "raw_evidence": [item.__dict__ for item in evidence],
        }
        _atomic_json(AUDIT_PATH, failure)
        raise
    finally:
        session.close()

    _atomic_parquet(FDR_PATH, fdr)
    _atomic_parquet(FX_PATH, fx)
    _atomic_parquet(PMI_PATH, pmi)
    _atomic_parquet(TSF_PATH, tsf)

    output_hashes = {
        _relative(path): sha256_file(path)
        for path in (FDR_PATH, FX_PATH, PMI_PATH, TSF_PATH)
    }
    audit = {
        "status": "PASS",
        "study_id": "510300_MACRO_STRESS_AVOIDANCE_V1",
        "scope": "仅2021-01-01至2026-08-25；未采集2016-2020",
        "data_start": DATA_START.isoformat(),
        "data_end": DATA_END.isoformat(),
        "latest_complete_month": LATEST_COMPLETE_MONTH,
        "retrieved_at": retrieved_at,
        "tls_verification": "ENABLED_NO_BYPASS",
        "daily_publication_clock": {
            "FDR007": "11:30 Asia/Shanghai，来自中国货币网页面发布说明",
            "USD/CNY": "09:15 Asia/Shanghai，冻结为开盘前官方中间价发布时间",
        },
        "monthly_revision_policy": (
            "交易研究只使用逐月首发正文；未单独取得的后续修订值保留为空，"
            "不得用最新修订表回填首发值"
        ),
        "series": {
            "fdr007": _series_audit(fdr, "date"),
            "usdcny_midpoint": _series_audit(fx, "date"),
            "pmi_new_orders": _series_audit(pmi, "reference_period"),
            "tsf_stock_yoy": _series_audit(tsf, "reference_period"),
        },
        "output_hashes": output_hashes,
        "raw_evidence_count": len(evidence),
        "raw_evidence": [item.__dict__ for item in evidence],
        "governance": {
            "performance_returns_read": False,
            "backtest_run": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    _atomic_json(AUDIT_PATH, audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "series": audit["series"],
                "output_hashes": output_hashes,
                "audit": _relative(AUDIT_PATH),
                "raw_evidence_count": len(evidence),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
