"""仅核对本地期权资料能否描述两腿认沽价差；不生成收益、标签或账户。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_continuation_20260925/option_source_feasibility"
EOD = ROOT / "data/raw/return_tail/options/510300_tushare_eod.parquet"
RISK = ROOT / "data/raw/return_tail/options/510300_sse_risk_indicators.parquet"
MANDATE = ROOT / "config/510300_existing_data_training_mandate_v1.json"


def sha(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "result.json").exists() or (OUT / "method.json").exists():
        raise RuntimeError("本次来源检查已有记录；请读取已保存结果，不覆盖。")
    method = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "只检查当日合约身份及单日横截面覆盖，不计算未来收益或选取最优结构",
        "selection_authorized": False,
        "new_strategy_labels": 0,
        "new_fits": 0,
        "new_accounts": 0,
        "new_market_downloads": 0,
        "criteria": {
            "historical_standard_id": "^510300([CP])(\\d{4})M(\\d{5})$；行权价以当日代码为准",
            "option_type": "P",
            "days_to_expiry": [30, 75],
            "minimum_volume": 100,
            "minimum_open_interest": 500,
            "close": "有限且大于零",
            "candidate_pair": "同一日、同一到期、至少两个不同合格行权价；不规定新策略选腿",
            "reason_for_thresholds": "沿用已有期权资料检查的流动性口径，非按新策略收益筛选",
        },
        "inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in [EOD, RISK, MANDATE]},
        "script_sha256": sha(Path(__file__)),
    }
    write_json(OUT / "method.json", method)
    eod = pd.read_parquet(EOD)
    risk = pd.read_parquet(RISK)
    for frame in [eod, risk]:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        if frame.duplicated(["trade_date", "contract_code"]).any():
            raise RuntimeError("日期与合约代码重复，不能进行一对一覆盖检查。")
    panel = eod.merge(
        risk[["trade_date", "contract_code", "exchange_contract_id"]],
        on=["trade_date", "contract_code"], how="left", validate="one_to_one",
    )
    panel["expiry_date"] = pd.to_datetime(panel["expiry_date"]).dt.normalize()
    parts = panel["exchange_contract_id"].fillna("").str.extract(r"^510300([CP])(\d{4})M(\d{5})$")
    panel["historical_standard"] = parts[0].eq(panel["option_type"]) & parts[0].notna()
    panel["historical_strike"] = pd.to_numeric(parts[2], errors="coerce") / 1000
    panel["dte"] = (panel["expiry_date"] - panel["trade_date"]).dt.days
    panel["liquid_put"] = (
        panel["historical_standard"] & panel["option_type"].eq("P")
        & panel["dte"].between(30, 75) & panel["volume"].ge(100)
        & panel["open_interest"].ge(500) & np.isfinite(panel["close"])
        & panel["close"].gt(0)
    )
    calendar = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
    daily_records = []
    for day, rows in panel.groupby("trade_date", sort=True):
        eligible = rows.loc[rows["liquid_put"]]
        strike_counts = eligible.groupby("expiry_date")["historical_strike"].nunique()
        daily_records.append({
            "observation_date": day,
            "all_eod_contracts": len(rows),
            "missing_official_id_contracts": int(rows["exchange_contract_id"].isna().sum()),
            "standard_put_contracts": int((rows["historical_standard"] & rows["option_type"].eq("P")).sum()),
            "liquid_put_contracts": len(eligible),
            "eligible_expiries": int(strike_counts.ge(2).sum()),
            "possible_strike_pairs": int(((strike_counts * (strike_counts - 1)) // 2).sum()),
            "pair_source_available": bool(strike_counts.ge(2).any()),
        })
    daily = pd.DataFrame(daily_records)
    daily.to_parquet(OUT / "daily_source_coverage.parquet", index=False)
    mismatch = panel["historical_standard"] & (
        (panel["historical_strike"] - pd.to_numeric(panel["strike"], errors="coerce")).abs() > 1e-8
    )
    keyed_risk = risk.set_index(["trade_date", "contract_code"]).index
    keyed_eod = eod.set_index(["trade_date", "contract_code"]).index
    yearly = []
    for year, rows in daily.groupby(daily["observation_date"].dt.year):
        yearly.append({"year": int(year), "days": len(rows),
                       "days_with_two_eligible_put_strikes": int(rows["pair_source_available"].sum())})
    quote_fields = [x for x in eod.columns if any(s in x.lower() for s in ["bid", "ask", "quote_time"])]
    mandate = json.loads(MANDATE.read_text(encoding="utf-8"))
    result = {
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "status": "LOCAL_DAILY_HISTORY_AVAILABLE_SYNCHRONOUS_EXECUTION_NOT_ESTABLISHED",
        "eod_rows": len(eod), "risk_rows": len(risk), "source_sessions": len(calendar),
        "start_date": str(calendar.min().date()), "end_date": str(calendar.max().date()),
        "eod_rows_missing_official_id": int(panel["exchange_contract_id"].isna().sum()),
        "official_risk_rows_without_eod": len(keyed_risk.difference(keyed_eod)),
        "historical_standard_rows": int(panel["historical_standard"].sum()),
        "snapshot_strike_disagrees_with_historical_standard_rows": int(mismatch.sum()),
        "historically_standard_but_current_metadata_adjusted_rows": int((panel["historical_standard"] & panel["is_adjusted"].fillna(False)).sum()),
        "days_with_two_eligible_put_strikes": int(daily["pair_source_available"].sum()),
        "days_without_two_eligible_put_strikes": int((~daily["pair_source_available"]).sum()),
        "yearly_source_coverage": yearly,
        "available_bid_ask_timestamp_fields": quote_fields,
        "eod_retrieved_at_min": str(eod["retrieved_at"].min()),
        "eod_retrieved_at_max": str(eod["retrieved_at"].max()),
        "risk_retrieved_at_min": str(risk["retrieved_at"].min()),
        "risk_retrieved_at_max": str(risk["retrieved_at"].max()),
        "limitations": [
            "单日有两腿资料不等于下一个开盘同时可成交。",
            "非同步日开盘和收盘只能构成历史价格代理；不能宣称成交或可部署。",
            "交易代码M可还原标准合约身份；持仓后除息调整须另外处理单位、行权价及保证金。",
            "取得时间不能替代历史首次公布时间；两日滞后仅为研究假设。",
            "没有用后续是否缺价筛掉事前候选，没有生成后续收益标签。",
            "未评估某一价差翼宽、Delta、持有期、收益、胜率或夏普。",
        ],
        "scope_confirmation_pending": True,
        "current_task_executable_assets": mandate["executable_assets"],
        "current_task_asset_scope_changed": False,
        "new_labels": 0, "new_fits": 0, "new_accounts": 0, "new_market_downloads": 0,
        "goal_achieved": False,
        "sources": [
            {"title": "上交所期权组合策略业务指引，现行页面核对",
             "url": "https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250610_10781452.shtml"},
            {"title": "上交所2026年1月16日沪深300ETF期权合约调整公告",
             "url": "https://www.sse.com.cn/assortment/options/disclo/update/c/c_20260116_10805396.shtml"},
        ],
    }
    for rel, checksum in method["inputs"].items():
        if sha(ROOT / rel) != checksum:
            raise RuntimeError("检查期间输入发生变化，不能保存为同一来源版本。")
    write_json(OUT / "result.json", result)
    report = (
        "本次只检查已存在的510300期权日线资料，没有新增收益标签、拟合、账户或市场数据采集。"
        "有保护认沽价差的研究范围仍等待本任务用户确认。\n\n"
        f"现存数据从{result['start_date']}至{result['end_date']}，共{len(calendar)}个交易日，"
        f"日行情{len(eod):,}条、官方风险指标{len(risk):,}条。沿用旧来源检查的30—75日到期、"
        "当日成交至少100张、持仓至少500张条件，"
        f"有{result['days_with_two_eligible_put_strikes']}天至少存在一组同到期、不同执行价的标准认沽资料，"
        f"另{result['days_without_two_eligible_put_strikes']}天不满足。这里是资料覆盖，并未固定新策略选腿。\n\n"
        f"晚期合约元数据的行权价与当日标准合约代码不符共{int(mismatch.sum()):,}行。"
        "因此历史选腿必须按当日官方M代码还原，不能用最新元数据回写过去；"
        "已持有合约的后续除息调整则须单独核算。"
        "[上交所2026年1月调整公告](https://www.sse.com.cn/assortment/options/disclo/update/c/c_20260116_10805396.shtml)"
        "明确给出了行权价、合约单位和标识的调整。\n\n"
        "本地文件没有同步买卖报价及其时钟。不同腿的日开盘价不能直接当成同步成交，"
        "当前资料至多支持注明价格代理假设的历史模拟，不能建立实盘成交证据。"
        "保证金减免还应按[上交所组合策略业务指引](https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250610_10781452.shtml)"
        "核对构建、解除及逐腿过渡，不能把理论到期最大损失当成全部资金需求。\n\n"
        "未检验收益，不能据此给出盈利结论或高夏普承诺；既有卖跨式失败结论保持。"
        "当前研究账户资产仍为510300.SH和CASH_CNY。\n"
    )
    (OUT / "资料可行性结论.md").write_text(report, encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "source_sessions", "days_with_two_eligible_put_strikes",
                      "days_without_two_eligible_put_strikes", "snapshot_strike_disagrees_with_historical_standard_rows",
                      "current_task_asset_scope_changed", "new_labels", "new_accounts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
