"""只读已取得的官方简表，记录滞后披露上下文，不替代同日数据。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from research.all_factor_source_binding_v1 import OUT, digest, now, parse_factsheet, read, write

SECTORS = {
    "能源": 0.3, "原材料": 9.6, "工业": 16.2, "可选消费": 5.4, "主要消费": 6.4,
    "医药卫生": 4.3, "金融": 19.9, "信息技术": 22.6, "通信服务": 9.9, "公用事业": 2.8, "房地产": 2.7,
}


def build_context(text, captured_at, normalized_at, original_quote):
    # 统计日来自简表页首；不因为目标日9月30日未匹配而改写原统计日。
    match = re.match(r"\s*(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not match:
        raise ValueError("无法从官方简表页首唯一识别统计日。")
    date = pd.Timestamp(year=int(match[1]), month=int(match[2]), day=int(match[3]))
    fields = parse_factsheet(text, date.strftime("%Y%m%d"))
    if "计算用股本" not in text:
        raise ValueError("当前简表的指标计算说明缺失。")
    holdings = []
    for line in text.splitlines():
        row = re.match(r"^(\d{6})\s+(\S+)\s+(\S+)\s+(上海|深圳|北京)\s+(\d+(?:\.\d+)?)%$", line.strip())
        if row:
            holdings.append({"code": row[1], "name": row[2], "reported_industry": row[3],
                             "exchange": row[4], "reported_weight_percent": float(row[5])})
    if len(holdings) != 10 or len({row["code"] for row in holdings}) != 10:
        raise ValueError("前十大披露成员不唯一或数量不符，不补成员。")
    capture, normalization = pd.Timestamp(captured_at), pd.Timestamp(normalized_at)
    if capture.tzinfo is None or normalization.tzinfo is None:
        raise ValueError("实际取得和处理时间必须包含时区。")
    return {
        "document_statistics_date": date.strftime("%Y-%m-%d"), "original_received_at": capture.isoformat(),
        "normalized_at": normalization.isoformat(), "available_at": max(capture, normalization).isoformat(),
        "reported_metrics": fields, "metric_definition_as_reported": "当前简表标示滚动市盈率/市净率/股息率，指标采用个股计算用股本；不外推全历史定义。",
        "reported_top10": holdings,
        "reported_top10_weight_percent": round(sum(row["reported_weight_percent"] for row in holdings), 2),
        "source_publication_first_vintage": "NOT_ESTABLISHED",
        "role": "LAGGED_OFFICIALLY_DISCLOSED_CONTEXT_KNOWN_ONLY_AFTER_ACTUAL_CAPTURE_AND_PROCESSING",
        "weight_role": "REPORTED_STATISTICS_DATE_NOT_CURRENT_DAILY_MEMBERSHIP_OR_ETF_HOLDINGS",
        "original_quote": original_quote,
        "quote_and_factsheet_same_statistics_date": original_quote.get("statistics_date") == date.strftime("%Y-%m-%d"),
        "api_peg_current_semantics_verified": False,
        "historical_use_authorized": False, "financial_model_admission": "NOT_ADMITTED",
        "forecast_success_or_trading_authority": False,
    }


def freeze():
    path = OUT / "disclosed_context_protocol_addendum.json"
    if path.exists():
        raise FileExistsError("滞后披露上下文用途已经登记，不覆盖。")
    source = OUT / "current_official_sources/factsheet.pdf"
    text = OUT / "current_official_sources/factsheet.txt"
    receipt = OUT / "current_official_sources/factsheet.receipt.json"
    quote = OUT / "current_official_snapshot.json"
    write(path, {
        "at": now(), "purpose": "新增已披露滞后上下文角色；原9月30日同日校验失败保持，非改门或重跑来源请求。",
        "known_content": "官方当前取得原件页首2026-08-31，已读原文并查看两页；不是盲检。",
        "source_files": [{"path": p.absolute().relative_to(OUT.absolute()).as_posix(), "sha256": digest(p)} for p in (source, text, receipt, quote)],
        "code_sha256": digest(Path(__file__).absolute()),
        "dependency_code_sha256": digest(Path(__file__).absolute().with_name("all_factor_source_binding_v1.py")),
        "sector_transcription_from_viewed_page_two": SECTORS,
        "weight_rounding": "原一位小数行业权重保留，允许11×0.05个百分点舍入误差，不归一化100.1。",
        "not_allowed": ["行业主线收益预测", "将8月31日估值套到9月30日", "将当今分类回填2019", "以P/PE假造盈利", "新请求或金融账户"],
        "new_network": 0, "new_models": 0, "new_accounts": 0,
    })
    print("已登记滞后官方披露上下文；原同日来源失败保持。")


def run():
    protocol = read(OUT / "disclosed_context_protocol_addendum.json")
    assert protocol["code_sha256"] == digest(Path(__file__).absolute())
    assert protocol["dependency_code_sha256"] == digest(Path(__file__).absolute().with_name("all_factor_source_binding_v1.py"))
    for source in protocol["source_files"]:
        assert digest(OUT / source["path"]) == source["sha256"]
    target = OUT / "current_disclosed_context.json"
    if target.exists():
        raise FileExistsError("滞后上下文已经保存，不覆盖。")
    text = (OUT / "current_official_sources/factsheet.txt").read_text(encoding="utf-8-sig")
    receipt = read(OUT / "current_official_sources/factsheet.receipt.json")
    assert receipt["http_status"] == 200
    original = read(OUT / "current_official_snapshot.json")
    context = build_context(text, receipt["received_at"], now(), original["quote"])
    total = sum(SECTORS.values())
    assert abs(total - 100.) <= len(SECTORS) * .05 + 1e-8
    context.update(reported_sector_weights_percent=SECTORS, reported_sector_weight_total_percent=round(total, 1),
                   sector_weight_rounding_residual_percentage_points=round(total - 100., 1),
                   current_valuation_and_weight_view_verified_pages=2)
    write(target, context)
    date, available = context["document_statistics_date"], context["available_at"]
    sectors = pd.DataFrame([{"industry": k, "reported_weight_percent": v, "statistics_date": date, "available_at": available,
                             "source_role": context["weight_role"]} for k, v in SECTORS.items()])
    holdings = pd.DataFrame(context["reported_top10"])
    holdings["statistics_date"], holdings["available_at"] = date, available
    for name, frame in [("官方披露行业权重_原日期与舍入保留", sectors), ("官方披露前十大_不冒充当日成员", holdings)]:
        frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)
        frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig")
    print(json.dumps({"统计日": date, "实际处理后可用时点": available, "官方原指标": context["reported_metrics"],
                      "前十大权重合计百分比": context["reported_top10_weight_percent"],
                      "行业权重原和百分比": context["reported_sector_weight_total_percent"],
                      "与9月30日同日": context["quote_and_factsheet_same_statistics_date"], "金融准入": "NOT_ADMITTED"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只读官方滞后披露，保留原同日失败，无新请求或回测。")
    parser.add_argument("action", choices=["freeze", "run"])
    freeze() if parser.parse_args().action == "freeze" else run()
