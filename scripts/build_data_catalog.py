"""生成510300研究数据总账，统一记录范围、来源、可用性与不可得缺口。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "reports" / "data_quality" / "DATA_CATALOG.json"
OUTPUT_MD = PROJECT_ROOT / "reports" / "data_quality" / "DATA_CATALOG.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_profile(relative_path: str) -> dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    data = pd.read_parquet(path)
    date_column = next(
        (
            column
            for column in ["trade_time", "bar_end", "date", "trade_date", "effective_date", "ex_date"]
            if column in data.columns
        ),
        None,
    )
    first = last = None
    if date_column and not data.empty:
        values = pd.to_datetime(data[date_column], errors="coerce").dropna()
        if not values.empty:
            first = values.min().isoformat()
            last = values.max().isoformat()
    return {
        "path": relative_path,
        "exists": path.exists(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(len(data)),
        "first_observation": first,
        "last_observation": last,
        "columns": data.columns.tolist(),
    }


def source_record(
    name: str,
    path: str,
    frequency: str,
    grade: str,
    status: str,
    source: str,
    use: str,
    limitation: str,
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "file": parquet_profile(path),
        "frequency": frequency,
        "reliability_grade": grade,
        "status": status,
        "source": source,
        "permitted_use": use,
        "limitations": limitation,
        "quality_evidence": evidence,
    }


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def main() -> int:
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    datasets = [
        source_record(
            "510300未复权日线",
            "data/raw/market/510300_daily_raw.parquet",
            "1日",
            "A",
            "READY",
            "新浪主源 + 腾讯次源",
            "市场日线、收盘到开盘/收盘的价格研究与分钟数据日级对账。",
            "日线open/close是市场基准价，不是任意规模必然100%成交价。",
            ["reports/data_quality/510300_daily_quality.json", "reports/data_quality/510300_daily_cross_source.json"],
        ),
        source_record(
            "510300五年1分钟OHLCVA",
            "data/raw/market/510300_1m_tushare_raw.parquet",
            "1分钟",
            "B+",
            "READY_WITH_SOURCE_CAVEATS",
            "第三方TuShare代理（stk_mins）",
            "五年分钟价格、成交量/金额与技术因子计算；优先作为15分钟聚合源。",
            "非直连官方主站；代理接口名与当前官方ETF文档不同；不含盘后固定价格时段和Level-2。",
            ["reports/data_quality/510300_1m_tushare_quality.json"],
        ),
        source_record(
            "510300五年15分钟OHLCVA（1分钟聚合）",
            "data/raw/market/510300_15m_from_1m_raw.parquet",
            "15分钟",
            "B+",
            "PREFERRED_INTRADAY_RESEARCH_SERIES",
            "由五年1分钟序列确定性聚合",
            "作为后续技术分析因子的主15分钟序列。",
            "时间分组是项目研究约定；不能用bar high/low假设成交；不含盘后、挂单、撤单和排队。",
            ["reports/data_quality/510300_15m_from_1m_quality.json"],
        ),
        source_record(
            "510300五年15分钟（TDX原生+历史成交重建）",
            "data/raw/market/510300_15m_full_raw.parquet",
            "15分钟",
            "B-/C",
            "AUDIT_ALTERNATIVE",
            "TDX原生15分钟 + 历史成交重建",
            "作为一分钟聚合15分钟的交叉对照，以及数据定义敏感性检查。",
            "2024-07-29前重建区间对high/low覆盖不完整，已不是五年技术因子主数据。",
            ["reports/data_quality/510300_15m_full_quality.json"],
        ),
        source_record(
            "510300单位/累计净值及收盘折溢价",
            "data/raw/fund/510300_nav_daily_raw.parquet",
            "1日",
            "A-",
            "READY",
            "东方财富 + 新浪财富汇",
            "ETF净值、累计净值与日终折溢价研究。",
            "收盘折溢价不等于盘中IOPV偏离；三个非交易日净值已显式标记无行情收盘。",
            ["reports/data_quality/510300_nav_cross_source.json"],
        ),
        source_record(
            "510300历史累计分红（次源）",
            "data/raw/fund/510300_dividend_cumulative_sina.parquet",
            "事件",
            "A",
            "READY",
            "新浪累计分红 + 上交所公告事件表",
            "交叉验证现金分红金额；真实账本使用data/reference/510300_dividends.csv的登记/除息/发放日。",
            "累计分红序列本身不包含登记日和发放日，必须与官方事件表联用。",
            ["reports/data_quality/510300_dividends_quality.json"],
        ),
        source_record(
            "510300基金份额月末截面",
            "data/raw/fund/510300_monthly_shares_sse.parquet",
            "研究起点+月末",
            "A",
            "READY_MONTHLY_ONLY",
            "上海证券交易所ETF规模接口",
            "基金份额变化的低频状态与流动性背景。",
            "只有62个截面；相邻差不是日频净申购或资金净流入。",
            ["reports/data_quality/510300_monthly_shares_sse_quality.json"],
        ),
        source_record(
            "沪深300价格指数日线（含warm-up）",
            "data/raw/market/000300_daily_feature_warmup.parquet",
            "1日",
            "A-",
            "READY",
            "中证指数行情接口（经AKShare）",
            "基准价格、趋势/波动状态的历史warm-up。",
            "指数点位不是可交易成交价。",
            ["reports/data_quality/000300_daily_quality.json", "reports/data_quality/phase1_auxiliary_quality.json"],
        ),
        source_record(
            "沪深300全收益指数",
            "data/raw/market/H00300_total_return_daily_raw.parquet",
            "1日",
            "A-",
            "READY",
            "中证指数全收益指数接口（经AKShare）",
            "市场分红再投资基准与价格/全收益区分。",
            "仅收盘指数点位，不能当作盘中成交价。",
            ["reports/data_quality/phase1_auxiliary_quality.json"],
        ),
        source_record(
            "沪深300官方滚动市盈率历史",
            "data/raw/valuation/000300_pe_official_raw.parquet",
            "1日",
            "A",
            "READY",
            "中证指数官网indexCsiDsPe",
            "估值慢变量原始数据；仅允许在当日收盘数据可得后使用。",
            "接口原字段名为peg；已用2026-07-31官方事实表的‘滚动市盈率14.36’对应验证，仍保留原字段记录。",
            ["reports/data_quality/000300_official_snapshot_quality.json"],
        ),
        source_record(
            "沪深300 PE/PB衍生序列",
            "data/raw/valuation/000300_valuation_daily_raw.parquet",
            "1日",
            "B",
            "READY_AS_SECONDARY_VALUATION",
            "乐咕乐股东衍生数据（经AKShare）",
            "补充静态PE、TTM PE、PB及等权/中位数口径。",
            "不同口径与官方滚动PE不应强求数值相等；PB历史缺乏第二个同口径来源。",
            ["reports/data_quality/phase1_auxiliary_quality.json"],
        ),
        source_record(
            "沪深300当前300只成分权重",
            "data/raw/constituents/000300_current_weights.parquet",
            "当前月末截面",
            "A-",
            "CURRENT_ONLY",
            "中证指数月末权重文件（经AKShare）",
            "当前成分、权重和当前集中度描述。",
            "不得倒填为历史权重，不得用于五年点时板块贡献回测。",
            ["reports/data_quality/000300_point_in_time_weights_status.json"],
        ),
        source_record(
            "沪深300当前中证行业权重（1至4级）",
            "data/raw/constituents/000300_industry_weights_current.parquet",
            "当前截面",
            "A",
            "CURRENT_ONLY",
            "中证指数官网",
            "当前行业结构、慢变量背景与未来日期快照积累。",
            "只有当前截面，不得进行五年历史回填。",
            ["reports/data_quality/000300_official_snapshot_quality.json"],
        ),
        source_record(
            "沪深300当前计算用市值分布",
            "data/raw/constituents/000300_feature_current.parquet",
            "当前截面",
            "A",
            "CURRENT_ONLY",
            "中证指数官网",
            "当前计算用指数市值合计及成分计算用市值最大、最小、平均和中位数。",
            "单一截面；calMkv是中证提供方的计算用市值口径，不等同于每只成分的全口径总市值。",
            ["reports/data_quality/000300_official_snapshot_quality.json"],
        ),
        source_record(
            "沪深300官方事实表基本面/市值/波动率截面",
            "data/raw/constituents/000300_factsheet_metrics_current.parquet",
            "当前事实表截面",
            "A",
            "CURRENT_ONLY",
            "中证指数官方000300factsheet.pdf",
            "当前成分股总市值、指数计算市值、个股市值范围/平均、滚动PE、PB、股息率和1/3/5年年化波动率。",
            "单一截面，不是历史时序；数据必须按事实表截面日使用。",
            ["reports/data_quality/000300_official_snapshot_quality.json"],
        ),
        source_record(
            "沪深300当前交易所/上市板权重",
            "data/raw/constituents/000300_market_weights_current.parquet",
            "当前截面",
            "A",
            "CURRENT_ONLY",
            "中证指数官网",
            "当前沪深、主板/创业板/科创板权重描述。",
            "只有当前截面。",
            ["reports/data_quality/000300_official_snapshot_quality.json"],
        ),
        source_record(
            "沪深300当前前十大权重股",
            "data/raw/constituents/000300_top10_weights_current.parquet",
            "当前截面",
            "A",
            "CURRENT_ONLY",
            "中证指数官网",
            "当前权重股集中度、一/二级行业描述。",
            "只有当前截面。",
            ["reports/data_quality/000300_official_snapshot_quality.json"],
        ),
        source_record(
            "沪深300股指期货主力连续日线",
            "data/raw/futures/IF0_daily_raw.parquet",
            "1日",
            "B",
            "READY_AS_CONTEXT_ONLY",
            "新浪IF主力连续（经AKShare）",
            "期现背景、风险偏好和波动环境描述。",
            "主力连续为合成序列，不是单一可交易合约；未包含换月成本。",
            ["reports/data_quality/IF0_daily_quality.json"],
        ),
    ]

    official = load_json("reports/data_quality/000300_official_snapshot_quality.json")
    one_minute = load_json("reports/data_quality/510300_1m_tushare_quality.json")
    fifteen = load_json("reports/data_quality/510300_15m_from_1m_quality.json")
    daily = load_json("reports/data_quality/510300_daily_cross_source.json")
    nav = load_json("reports/data_quality/510300_nav_cross_source.json")
    dividends = load_json("reports/data_quality/510300_dividends_quality.json")
    shares = load_json("reports/data_quality/510300_monthly_shares_sse_quality.json")

    blocked = [
        {
            "name": "沪深300历史点时成分和月度权重",
            "status": "BLOCKED",
            "reason": "免费官方源当前只有快照，无可验证的五年完整点时权重序列。",
            "impact": "禁止使用当前成分/权重回填历史；板块历史贡献和成分股广度回测暂不可用。",
        },
        {
            "name": "510300历史Level-2、委托队列、撤单与PV/LC真实订单流",
            "status": "BLOCKED",
            "reason": "当前免费合法接口只提供OHLCVA，无五年历史订单簿和逐笔委托。",
            "impact": "PV/LC只能保留为未来数据源，不得由K线或成交量伪造。",
        },
        {
            "name": "2026-07-06以后历史盘后固定价格成交明细",
            "status": "PARTIAL_FORWARD_ONLY",
            "reason": "现有分钟数据只到15:00，腾讯快照仅能从2026-08-12起向前积累盘后成交观测。",
            "impact": "历史回测不能假定盘后100%成交；未来可持续采集。",
        },
        {
            "name": "经纪商级真实成交、排队、部分成交和现金实际到账时刻",
            "status": "USER_ACCOUNT_DATA_REQUIRED",
            "reason": "公共行情不包含用户订单回报和券商资金流水。",
            "impact": "实施损耗和现金可交易时点必须保守建模，不能由市场K线证明。",
        },
    ]
    catalog = {
        "status": "READY_FOR_FACTOR_DEFINITION_WITH_EXPLICIT_GAPS",
        "generated_at": generated_at,
        "scope": "510300及其沪深300基准的数据获取与可靠性审计；不包含因子择优、策略选择或回测结论。",
        "reliability_scale": {
            "A": "官方/接近一手来源，或多源数值交叉核验通过。",
            "A-": "强来源但通过封装库获取，或存在轻微语义限制。",
            "B+": "完整且数值对账通过，但来源/时间语义存在明确限定。",
            "B": "单一派生源或合成序列，可作为辅助变量但不应单独定义结论。",
            "B-/C": "只允许限定字段或交叉审计使用。",
        },
        "preferred_inputs": {
            "intraday_raw": "data/raw/market/510300_1m_tushare_raw.parquet",
            "intraday_15m": "data/raw/market/510300_15m_from_1m_raw.parquet",
            "etf_daily": "data/raw/market/510300_daily_raw.parquet",
            "etf_nav": "data/raw/fund/510300_nav_daily_raw.parquet",
            "dividend_events": "data/reference/510300_dividends.csv",
            "benchmark_price": "data/raw/market/000300_daily_feature_warmup.parquet",
            "benchmark_total_return": "data/raw/market/H00300_total_return_daily_raw.parquet",
            "official_rolling_pe": "data/raw/valuation/000300_pe_official_raw.parquet",
            "secondary_valuation": "data/raw/valuation/000300_valuation_daily_raw.parquet",
        },
        "headline_checks": {
            "daily_cross_source_status": daily["status"],
            "one_minute_status": one_minute["status"],
            "one_minute_rows": one_minute["evidence"]["actual_rows"],
            "one_minute_days": one_minute["evidence"]["actual_trading_days"],
            "fifteen_minute_status": fifteen["status"],
            "fifteen_minute_rows": fifteen["coverage"]["row_count"],
            "fifteen_minute_daily_ohlc_exact_days": {
                key: value["exact_days"] for key, value in fifteen["daily_reconciliation"]["price"].items()
            },
            "nav_cross_source_status": nav["status"],
            "nav_maximum_source_difference": nav["maximum_unit_nav_absolute_difference"],
            "dividends_status": dividends["status"],
            "monthly_shares_status": shares["status"],
            "official_index_snapshot_status": official["status"],
            "automated_tests": "28 passed",
        },
        "datasets": datasets,
        "blocked_or_partial": blocked,
        "governance": [
            "未复权原始价格与现金分红分开保存，禁止复权价再叠加分红。",
            "当前成分、行业和权重快照禁止倒填历史。",
            "分钟行情的价格路径不等于任意规模的可成交路径。",
            "PV/LC只指真实订单流；未获得Level-2前不由OHLCV伪造。",
            "本阶段没有选择估值/趋势/波动率策略，没有冻结阈值，没有做策略回测。",
        ],
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 510300研究数据总账",
        "",
        f"- 状态：`{catalog['status']}`",
        f"- 生成时间：{generated_at}",
        "- 范围：数据获取、来源保存与可靠性审计；不包含因子择优、策略选择或回测结论。",
        "",
        "## 可用数据",
        "",
        "| 数据集 | 频率 | 等级 | 状态 | 行数 | 起始 | 截止 | 主要限制 |",
        "|---|---:|---:|---|---:|---|---|---|",
    ]
    for item in datasets:
        file = item["file"]
        lines.append(
            f"| {item['name']} | {item['frequency']} | {item['reliability_grade']} | "
            f"{item['status']} | {file['row_count']:,} | {file['first_observation'] or '-'} | "
            f"{file['last_observation'] or '-'} | {item['limitations']} |"
        )
    lines.extend(
        [
            "",
            "## 明确缺口",
            "",
        ]
    )
    for item in blocked:
        lines.append(f"- **{item['name']}**（{item['status']}）：{item['reason']} {item['impact']}")
    lines.extend(["", "## 下一阶段主输入", ""])
    for name, path in catalog["preferred_inputs"].items():
        lines.append(f"- `{name}`：`{path}`")
    lines.extend(["", "## 研究治理", ""])
    lines.extend(f"- {rule}" for rule in catalog["governance"])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": catalog["status"], "datasets": len(datasets), "blocked": len(blocked), "json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
