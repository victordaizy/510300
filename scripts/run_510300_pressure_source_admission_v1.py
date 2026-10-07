"""逐日读取已存三流，完成PR-E01/02；不下载数据，不计算未来收益。"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.pressure_source_admission_v1 import (  # noqa: E402
    add_previous_day_counts, conditional_prefix_windows, make_grid, observe_grid,
    observed_book_features, session_labels,
)
from research.quote_state_envelope_v1 import clock_ms  # noqa: E402

CONFIG = ROOT / "config/510300_pressure_source_admission_v1.json"
CN = timezone(timedelta(hours=8))


def now() -> str:
    return datetime.now(CN).isoformat()


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(json_value(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity_check(frame: pd.DataFrame, date: str, source_security: str) -> None:
    if frame.date.isna().any() or not frame.date.eq(int(date)).all():
        raise ValueError(f"{date}源日期不一致。")
    if frame.wind_code.isna().any() or not frame.wind_code.eq(source_security).all():
        raise ValueError(f"{date}原证券标签不一致。")
    if frame.ex_code.isna().any() or not frame.ex_code.eq("510300").all():
        raise ValueError(f"{date}交易所证券代码不一致。")
    clock_ms(frame.time.to_numpy())


def scan_ticks(item: dict, source_security: str, batch_size: int) -> dict:
    stream, date = item["stream"], item["date"]
    columns = ["wind_code", "ex_code", "date", "time", "price", "volume"]
    columns += (["order_type", "order_code", "order_id", "ex_order_id"] if stream == "逐笔委托"
                else ["trade_code", "order_code", "bs_flag", "trade_id"])
    session_counts, type_counts, after_type_counts, after_bs_counts, after_prices = Counter(), Counter(), Counter(), Counter(), Counter()
    rows = bad_grid = after_rows = adds_deletes = usable_adds = cancellation_rows = non_execution = 0
    after_samples, times, volumes = [], [], []
    source_first, source_last = 240000000, 0
    for batch in pq.ParquetFile(item["path"]).iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        identity_check(frame, date, source_security)
        t = frame.time.to_numpy(dtype=np.int64)
        phases = session_labels(t)
        session_counts.update(Counter(phases))
        rows += len(frame)
        source_first, source_last = min(source_first, int(t.min())), max(source_last, int(t.max()))
        bad_grid += int((t % 10 != 0).sum())
        after = frame.time.ge(150500000)
        after_rows += int(after.sum())
        if stream == "逐笔委托":
            tags = frame.order_type.fillna("<NULL>")
            type_counts.update(tags.value_counts().to_dict())
            after_type_counts.update(tags[after].value_counts().to_dict())
            adds_deletes += int((after & tags.isin(["A", "D"])).sum())
            usable_adds += int((after & tags.eq("A") & frame.price.gt(0)
                                & frame.volume.gt(0) & frame.order_code.isin(["B", "S"])
                                & frame.ex_order_id.gt(0)).sum())
            remaining = max(0, 12 - len(after_samples))
            after_samples.extend(frame.loc[after].head(remaining).to_dict(orient="records"))
        else:
            tags = frame.trade_code.fillna("<NULL>")
            type_counts.update(tags.value_counts().to_dict())
            after_type_counts.update(tags[after].value_counts().to_dict())
            after_bs_counts.update(frame.loc[after, "bs_flag"].fillna("<NULL>").value_counts().to_dict())
            after_prices.update(frame.loc[after, "price"].value_counts().to_dict())
            execution = frame.volume.gt(0) & frame.price.gt(0) & ~tags.eq("C")
            cancellation_rows += int(tags.eq("C").sum())
            non_execution += int((~execution).sum())
            regular = execution & frame.time.lt(150500000)
            times.append(t[regular])
            volumes.append(frame.loc[regular, "volume"].to_numpy(dtype=float))
    return {"rows": rows, "first_time": source_first, "last_time": source_last,
            "non_centisecond_rows": bad_grid, "session_counts": dict(session_counts),
            "type_counts": dict(type_counts), "after_1505_rows": after_rows,
            "after_1505_type_counts": dict(after_type_counts), "after_1505_bs_counts": dict(after_bs_counts),
            "after_1505_price_counts": dict(after_prices), "after_1505_add_delete_rows": adds_deletes,
            "after_1505_usable_add_rows": usable_adds, "after_1505_samples": after_samples,
            "trade_code_C_rows": cancellation_rows, "non_execution_rows": non_execution,
            "prefix_trade_times": np.concatenate(times) if times else np.array([], dtype=np.int64),
            "prefix_trade_volumes": np.concatenate(volumes) if volumes else np.array([], dtype=float)}


def summarize_iopv(date: str, quotes: pd.DataFrame, orders: dict, trades: dict, grid: pd.DataFrame) -> dict:
    positive = quotes.loc[quotes.iopv.gt(0)].sort_values("time", kind="stable")
    after = positive.loc[positive.time.ge(150500000)]
    changed = positive.loc[positive.iopv.ne(positive.iopv.shift())]
    ratios = positive.loc[positive.price.gt(0), "iopv"] / positive.loc[positive.price.gt(0), "price"]
    probes = grid.loc[grid.m1_probe, ["hhmm", "source_quote_row", "source_time", "source_iopv_raw",
                                    "source_price_raw", "quote_age_nominal_ms", "source_nominal_fresh",
                                    "source_positive_iopv", "local_diagnostic_reasons"]].copy()
    for scale in [1000, 10000, 100000]:
        probes[f"iopv_cny_if_raw_divided_by_{scale}"] = probes.source_iopv_raw / scale
    return {"date": date, "quote_rows": len(quotes), "positive_iopv_rows": len(positive),
            "zero_iopv_rows": int(quotes.iopv.eq(0).sum()), "null_iopv_rows": int(quotes.iopv.isna().sum()),
            "negative_iopv_rows": int(quotes.iopv.lt(0).sum()),
            "first_positive_time": int(positive.time.min()), "last_positive_time": int(positive.time.max()),
            "raw_iopv_min": float(positive.iopv.min()), "raw_iopv_max": float(positive.iopv.max()),
            "raw_iopv_distinct": int(positive.iopv.nunique()),
            "last_observed_value_change_time": int(changed.time.iloc[-1]),
            "median_raw_iopv_over_raw_price": float(ratios.median()) if len(ratios) else None,
            "after_1505_positive_rows": len(after), "after_1505_iopv_values": sorted(after.iopv.unique().tolist()),
            "after_1505_order_rows": orders["after_1505_rows"],
            "after_1505_order_type_counts": orders["after_1505_type_counts"],
            "after_1505_add_delete_order_rows": orders["after_1505_add_delete_rows"],
            "after_1505_order_samples": orders["after_1505_samples"],
            "after_1505_trade_rows": trades["after_1505_rows"],
            "after_1505_trade_bs_counts": trades["after_1505_bs_counts"],
            "after_1505_trade_price_counts": trades["after_1505_price_counts"],
            "frozen_time_probes": probes.to_dict(orient="records"),
            "conclusion": "数值与现价相近只是尺度候选；恒定尾段不能区分真实稳定、携带或未更新；S是产品状态而非排队申报",
            "unit_mapping_verified": False, "reference_time_verified": False,
            "received_before_1506_verified": False, "after_hours_queue_reconstructable_verified": False}


def official_contract(config: dict) -> dict:
    pages = json.loads((ROOT / config["official_pages"]).read_text(encoding="utf-8"))
    evidence = [{"physical_page": p["physical_page"], "text_sha256": hashlib.sha256(p["text"].encode("utf-8")).hexdigest(),
                 "saved_text_path": str(ROOT / config["official_pages"])}
                for p in pages if p["physical_page"] in [6, 7, 10, 11, 12, 15, 17, 18, 19, 20, 21, 22]]
    return {"status": "OFFICIAL_FEED_SEMANTICS_VERIFIED_VENDOR_MAPPING_UNVERIFIED",
            "official_version": "LDDS Level-2 2.0.10", "official_url": config["official_url"],
            "official_pdf_sha256": digest(ROOT / config["official_pdf"]),
            "readme_sha256": digest(ROOT / config["source_readme"]),
            "readme_revision": config["source_revision"],
            "verified_facts": [
                {"fact": "UA3202/DataTimeStamp是最新订单时间，秒精度；不是接收时间", "physical_page": 10},
                {"fact": "ClosePx与LastPx是两个字段，导出price不能自动认证今日收盘成分", "physical_page": 10},
                {"fact": "UA3202 IOPV有tag10057三位及tag10140五位精度；有效精度由实际业务决定", "physical_page": 15},
                {"fact": "UA3108盘后行情的DataTimeStamp是最新订单时间，毫秒精度；该消息没有IOPV字段", "physical_pages": [17, 18]},
                {"fact": "快照和逐笔属于不同类别，没有相对先后次序；检查丢包需要通道及连续序号", "physical_page": 19},
                {"fact": "合并逐笔A为新增、D为删除、S为产品状态、T为成交；S的价格和数量无订单含义", "physical_page": 21},
                {"fact": "合并逐笔TickTime为百分之一秒；不能据同一时间自行排列订单次序", "physical_page": 21},
                {"fact": "源README仅承诺价格/10000，明确未核实IOPV等单位，且没有交易所数据接收时间", "source": "pinned README"}],
            "unidentified_items": ["导出time对应哪个模板/字段及是否被厂商改写", "IOPV高/低精度字段及重标度映射",
                                   "IOPV经济时点、独立计算误差和15:06前的历史可得性", "price对应ClosePx还是LastPx",
                                   "盘后逐笔通道连续性、完整申撤与前方队列", "连续行情盘口与累计量是否同次更新"],
            "do_not_infer": ["盘后恒定IOPV不等于已证实陈旧，也不等于15:00同步参考",
                             "逐笔成交存在不等于个人订单能成交", "整秒time不等于快照生成在该秒内",
                             "条件累计量区间<=1秒不等于误差已校准", "源文件齐备不等于通道无丢包"],
            "legacy_filename_correction": "official_sources/sse_md102_spec.txt是较早接口文本；本实验使用此处保存的LDDS2.0.10直接模板，不改写旧冻结输出",
            "official_page_evidence": evidence,
            "strict_source_contract": config["verified_source_contract"]}


def write_summary(matrix: pd.DataFrame, daily: pd.DataFrame, session_rows: pd.DataFrame,
                  iopv: list[dict], receipts: list[dict], config: dict) -> dict:
    continuous = matrix.loc[matrix.grid_kind.eq("M2_CONTINUOUS")]
    baseline = []
    candidate_dates = []
    for regime, part in continuous.groupby("regime", sort=True):
        rec = {"regime": regime, "source_dates": int(part.date.nunique()), "continuous_grid_rows": len(part)}
        for column in ["calendar_prior_days", "prior_source_m2_pair_days", "prior_conditional_m2_pair_days", "prior_strict_m2_pair_days"]:
            eligible = part.loc[part[column].ge(60)]
            rec[f"{column}_ge60_rows"] = len(eligible)
            rec[f"{column}_ge60_dates"] = int(eligible.date.nunique())
            rec[f"{column}_first_ge60_date"] = eligible.date.min() if len(eligible) else None
            rec[f"{column}_maximum"] = int(part[column].max())
        for label in ["source_60day_m2_candidate", "conditional_60day_m2_candidate", "strict_60day_m2_candidate"]:
            selected = part.loc[part[label]]
            rec[f"{label}_rows"] = len(selected)
            rec[f"{label}_dates"] = int(selected.date.nunique())
            rec[f"{label}_first_date"] = selected.date.min() if len(selected) else None
        baseline.append(rec)
    for date, part in continuous.groupby("date", sort=True):
        candidate_dates.append({"date": date, "regime": part.regime.iloc[0], "calendar_prior_days": int(part.calendar_prior_days.iloc[0]),
                                "source_m2_candidate_slots": int(part.source_60day_m2_candidate.sum()),
                                "conditional_m2_candidate_slots": int(part.conditional_60day_m2_candidate.sum()),
                                "strict_m2_candidate_slots": int(part.strict_60day_m2_candidate.sum())})
    out = ROOT / config["output_directory"]
    pd.DataFrame(candidate_dates).to_csv(out / "06_逐日可评价候选.csv", index=False, encoding="utf-8-sig")
    strict_receipt = {"status": "PASS_SAVED_MATRIX_AND_INPUT_IDENTITY_RECOMPUTATION",
                      "verified_files": len(receipts), "verified_source_rows": int(sum(r["rows"] for r in receipts)),
                      "source_hash_mismatch_count": sum(not r["hash_matches_manifest"] for r in receipts),
                      "matrix_rows": len(matrix), "unique_date_minute_rows": int(matrix[["date", "minute"]].drop_duplicates().shape[0]),
                      "same_regime_current_day_excluded": True, "strict_rows_zero_from_missing_contract": not bool(matrix.strict_point_qualified.any()),
                      "conditional_claim_is_not_receive_proof": True, "strategy_returns_computed": False}
    saved = pd.read_parquet(out / "03_日期时间资格矩阵.parquet")
    recomputed = add_previous_day_counts(saved.drop(columns=[c for c in saved if c.startswith("prior_")
                                                            or c.startswith("m2_") or c.startswith("calendar_prior")
                                                            or "60day_m2_candidate" in c]), 60, 5)
    checked_columns = ["date", "minute", "calendar_prior_days", "prior_source_m2_pair_days", "prior_conditional_m2_pair_days",
                       "prior_strict_m2_pair_days", "source_60day_m2_candidate", "conditional_60day_m2_candidate", "strict_60day_m2_candidate"]
    pd.testing.assert_frame_equal(saved[checked_columns].reset_index(drop=True), recomputed[checked_columns], check_dtype=False)
    assert len(matrix) == len(daily) * len(make_grid(config["measurement"]))
    assert len(receipts) == config["expected_files"] and strict_receipt["source_hash_mismatch_count"] == 0
    assert strict_receipt["matrix_rows"] == strict_receipt["unique_date_minute_rows"]
    assert not matrix.strict_point_qualified.any() and not matrix.strict_60day_m2_candidate.any()
    save_json(out / "08_保存结果复算.json", strict_receipt)
    return {"study_id": config["study_id"], "completed_at": now(),
            "status": "COMPLETED_SOURCE_DIAGNOSTICS_ORIGINAL_M1_M2_NOT_ADMITTED",
            "experiment_status": {"PR-E01": "COMPLETED_FIELD_AND_AFTER_HOURS_CONTRACT",
                                  "PR-E02": "COMPLETED_ALL_181_DAY_ADMISSION_MATRIX"},
            "source_dates": len(daily), "source_files": len(receipts),
            "source_rows_by_stream": dict(session_rows.groupby("stream").rows.sum().astype(int)),
            "grid_rows": len(matrix), "continuous_grid_rows": len(continuous),
            "nominal_fresh_book_continuous_slots": int(continuous.source_nominal_book_observed.sum()),
            "nominal_fixed_band_continuous_slots": int(continuous.source_nominal_band_observed.sum()),
            "source_m2_pair_slots": int(continuous.m2_source_field_pair.sum()),
            "conditional_m2_pair_slots": int(continuous.m2_source_conditional_pair.sum()),
            "baseline_by_regime": baseline,
            "positive_iopv_dates": [x["date"] for x in iopv],
            "positive_iopv_rows": sum(x["positive_iopv_rows"] for x in iopv),
            "positive_iopv_days_after_1505_add_delete_orders": sum(x["after_1505_add_delete_order_rows"] for x in iopv),
            "all_days_after_1505_orders": int(daily.after_1505_orders.sum()),
            "all_days_after_1505_status_S_orders": int(daily.after_1505_status_S_orders.sum()),
            "all_days_after_1505_add_delete_orders": int(daily.after_1505_add_delete_orders.sum()),
            "all_days_after_1505_trades": int(daily.after_1505_trades.sum()),
            "continuous_quotes_total": int(daily.continuous_quotes.sum()),
            "continuous_non_second_quote_rows": int(daily.continuous_non_second_quotes.sum()),
            "quote_trade_flag_nonnull_rows": int(daily.quote_trade_flag_nonnull_rows.sum()),
            "continuous_exact_prefix_quotes": int(daily.continuous_exact_prefix_quotes.sum()),
            "continuous_quotes_with_later_nominal_trade": int(daily.continuous_quotes_with_later_nominal_trade.sum()),
            "continuous_legacy_conditional_compatible_quotes": int(daily.continuous_legacy_conditional_compatible_quotes.sum()),
            "strict_M1_reference_days": 0, "strict_M2_field_slots": int(continuous.m2_strict_field_pair.sum()),
            "strict_M2_60day_candidate_slots": int(continuous.strict_60day_m2_candidate.sum()),
            "formal_strategy_event_count": None, "formal_strategy_event_status": "NOT_RUN_SOURCE_GATE_FAILED",
            "actual_orders": 0, "confirmed_fills": 0,
            "returns_status": "NOT_COMPUTED", "net_expectancy": None, "net_sharpe": None,
            "mechanism_rejected": False, "fee_usd": 0, "new_raw_data_download_bytes": 0,
            "source_scope": "冻结181日全部三流，不读取策略未来标签；只使用行情字段与消息计数/累计量",
            "conditional_scope": "旧截断秒假设下的相容性诊断，最新订单时间到导出time的映射未证实，不能认证真实快照或接收时点",
            "next_work": "PR-E03只补具体字段合同：免费IOPV映射/经济时点/历史可得性、带类型/通道的盘后申撤与接收时间；未取得则保持NOT_COMPUTED，不能仅靠旁证ETF或更多同类日期解除全部门槛",
            "original_m1_minimum_independent_days": 30,
            "verification": strict_receipt}


def report_markdown(summary: dict, iopv: list[dict], config: dict) -> str:
    lines = ["# PR-E01/PR-E02：免费历史盘口来源资格结论", "",
             f"完成时间：{summary['completed_at']}。本次按固定合同核对全部181日，不计算策略未来收益。", "",
             "**两项来源实验已完成，原M1/M2仍不能进入正式成交后收益检验。** 原因是IOPV字段/经济时点/可得性和接收时钟未识别；盘后委托文件里的S消息不能提供排队证据。这是来源准入结论，不是拒绝压力修复机制。", "",
             "## PR-E01：字段与盘后消息", "",
             "| 日期 | 正IOPV行 | 15:05后正值行 | 尾段原值 | 状态S消息 | 新增/删除委托 | 盘后成交行 |", "|---|---:|---:|---|---:|---:|---:|"]
    for day in iopv:
        lines.append(f"| {day['date']} | {day['positive_iopv_rows']:,} | {day['after_1505_positive_rows']} | {day['after_1505_iopv_values']} | {day['after_1505_order_type_counts'].get('S', 0)} | {day['after_1505_add_delete_order_rows']} | {day['after_1505_trade_rows']} |")
    lines += ["", "IOPV原值/现价相近只支持一个重标度候选；源README明确未建立IOPV单位，官方字段有不同精度而导出缺少字段映射。尾段恒定不能区分实际稳定、值的携带和未更新，更不能认证15:00公允价值。四日亦不足原30个独立日的区间条件。", "",
              f"上交所LDDS2.0.10物理第21页定义S为产品状态，第17—18页盘后行情模板无IOPV字段；第10页连续模板的时间是最新订单时间。出处：[上交所官方接口规范]({config['official_url']})。保留原导出字段，不把厂商混合66列替换成已确认的模板类型。", "",
              "全181日15:05后的委托记录："
              f"{summary['all_days_after_1505_orders']:,}行，其中S {summary['all_days_after_1505_status_S_orders']:,}行，A/D {summary['all_days_after_1505_add_delete_orders']:,}行；成交{summary['all_days_after_1505_trades']:,}行。成交存在不能认证个人填单概率。", "",
              "## PR-E02：同时间、同制度的先前60日", "",
              f"保存{summary['grid_rows']:,}个日期×HH:MM来源点，其中连续会话{summary['continuous_grid_rows']:,}点。保留原09:30—11:29及13:00—14:56网格，附加15:00和15:05—15:30覆盖探针。连续源{summary['continuous_quotes_total']:,}条整秒报价，非整秒{summary['continuous_non_second_quote_rows']}条；全行情trade_flag非空{summary['quote_trade_flag_nonnull_rows']}条。", "",
              "| 制度 | 源日期 | 日历上够60过去日的日期 | 有本地字段及60日的候选日期/分钟 | 条件时钟诊断候选日期/分钟 | 严格准入日期/分钟 |", "|---|---:|---:|---|---|---|"]
    for row in summary["baseline_by_regime"]:
        lines.append(f"| {row['regime']} | {row['source_dates']} | {row['calendar_prior_days_ge60_dates']} | {row['source_60day_m2_candidate_dates']}/{row['source_60day_m2_candidate_rows']} | {row['conditional_60day_m2_candidate_dates']}/{row['conditional_60day_m2_candidate_rows']} | {row['strict_60day_m2_candidate_dates']}/{row['strict_60day_m2_candidate_rows']} |")
    lines += ["", "本地字段候选仅要求当前固定10bp盘口、同会话五分钟前报价及其过去同时间覆盖。条件候选另加旧截断秒/百分之一秒假设的累计量范围完全落在端点前。两者均缺同步参考和接收证据，所以不是合格M2事件，也不是正式60日基线。先前日计数严格排除当前日及另一制度。", "",
              f"连续行情累计量可对应逐笔前缀{summary['continuous_exact_prefix_quotes']:,}条；其中已包含名义上更晚成交{summary['continuous_quotes_with_later_nominal_trade']:,}条。在旧时间截断假设下范围相容{summary['continuous_legacy_conditional_compatible_quotes']:,}条。这些数字只诊断导出相容性，不能认证快照、盘口与累计量同次更新。", "",
              "## 处置与下一步", "",
              "PR-E01/02结案；M1合格独立参考日0，M2严格字段及60日候选0。正式事件检验未启动，期望/盈亏比/夏普NOT_COMPUTED；不是算出收益等于零。没有新订单、新账户、费用或原始数据下载。", "",
              "PR-E03若继续，优先取得免费字段字典/映射和带类型、通道、时钟证据的具体原始样本。旁证ETF价格只可支持价格共振诊断；更多同类历史不会补出源README明确缺失的历史接收时钟，也不能把7月6日前日期填进其后基线。无法取得时关闭这套导出对原定义的准入路线，保留机制未识别状态。PR-E04保持NOT_RUN，不缩短60日或30独立日，不调旧RV20表达。", "",
              "## 可复查文件", "",
              "- registration.json：运行前合同、代码/源说明hash与权限。",
              "- 01_字段合同.json：官方物理页、保存原文的入口与hash、导出未识别项。",
              "- 02_IOPV与盘后逐日证据.json：四日值变化、固定时点引用及状态消息。",
              "- 03_日期时间资格矩阵.parquet：全部来源点、引用源行、条件范围及严格失败原因。",
              "- 04_逐日来源核对.csv、05_三流会话覆盖.csv、06_逐日可评价候选.csv：181日完整计数。",
              "- 07_输入文件核对.csv、08_保存结果复算.json：543原文件hash、行/schema检查及保存矩阵计数复算。",
              "- summary.json、验证日志.txt、FILE_INDEX.csv：结果、边界测试与输出身份。", ""]
    return "\n".join(lines)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    original = json.loads((ROOT / config["original_protocol"]).read_text(encoding="utf-8"))
    assert original["measurement"]["baseline_previous_qualified_days"] == config["measurement"]["baseline_previous_qualified_days"] == 60
    assert original["measurement"]["maximum_continuous_quote_age_seconds"] * 1000 == config["measurement"]["maximum_quote_age_ms"]
    assert original["measurement"]["depth_band_bps"] == config["measurement"]["band_bps"]
    assert original["m2"]["shock_window_minutes"] == config["measurement"]["m2_lag_minutes"] == 5
    assert not any(config["verified_source_contract"].values())
    manifest_path = ROOT / config["source_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["revision"] == config["source_revision"]
    files = manifest["files"]
    by_day = {}
    for item in files:
        if item["stream"] in by_day.setdefault(item["date"], {}):
            raise ValueError("同日期同流重复，拒绝任意选择。")
        by_day[item["date"]][item["stream"]] = item
    assert len(by_day) == config["expected_dates"] and len(files) == config["expected_files"]
    assert all(set(streams) == {"行情", "逐笔委托", "逐笔成交"} for streams in by_day.values())
    out = ROOT / config["output_directory"]
    if out.exists():
        raise SystemExit("该实验输出已存在；拒绝覆盖冻结批次。")
    out.mkdir(parents=True)
    tests = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "tests/test_pressure_source_admission_v1.py",
                            "tests/test_quote_state_envelope_v1.py", "-q", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    (out / "验证日志.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    if tests.returncode:
        raise RuntimeError("来源资格边界验证失败，未启动实样本。")
    frozen_inputs = [CONFIG, ROOT / config["original_protocol"], ROOT / config["source_manifest"],
                     ROOT / config["source_readme"], ROOT / config["official_pdf"], ROOT / config["official_pages"],
                     Path(__file__), ROOT / "research/pressure_source_admission_v1.py",
                     ROOT / "research/quote_state_envelope_v1.py", ROOT / "tests/test_pressure_source_admission_v1.py",
                     ROOT / "tests/test_quote_state_envelope_v1.py"]
    registration = {"study_id": config["study_id"], "registered_at": now(),
                    "status": "FROZEN_BEFORE_ALL_DAY_DIAGNOSTICS_NO_RETURN_LABELS",
                    "authorization": "用户在阅读长期事实和下一步后明确要求：请继续完成；本次只完成PR-E01/02",
                    "source_dates": sorted(by_day), "configuration": config,
                    "inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in frozen_inputs],
                    "boundary_validation_exit_code": tests.returncode,
                    "source_hashes_from": str(manifest_path), "strategy_returns_read": False}
    save_json(out / "registration.json", registration)
    save_json(out / "01_字段合同.json", official_contract(config))
    grid = make_grid(config["measurement"])
    matrix_parts, daily_rows, session_rows, iopv_rows, receipts = [], [], [], [], []
    for number, date in enumerate(sorted(by_day), 1):
        streams = by_day[date]
        for stream, item in streams.items():
            path = Path(item["path"])
            pf = pq.ParquetFile(path)
            current_hash = digest(path)
            if current_hash != item["sha256"] or pf.metadata.num_rows != item["rows"] or pf.schema_arrow.names != item["columns"]:
                raise ValueError(f"{date}/{stream}内容或schema与manifest不一致。")
            receipts.append({"date": date, "stream": stream, "path": str(path), "bytes": path.stat().st_size,
                             "rows": pf.metadata.num_rows, "column_count": len(pf.schema_arrow), "sha256": current_hash,
                             "hash_matches_manifest": True, "rows_and_schema_match_manifest": True})
        quotes = pq.read_table(streams["行情"]["path"]).to_pandas().reset_index(drop=True)
        identity_check(quotes, date, config["preserved_source_security"])
        orders = scan_ticks(streams["逐笔委托"], config["preserved_source_security"], config["batch_size"])
        trades = scan_ticks(streams["逐笔成交"], config["preserved_source_security"], config["batch_size"])
        witnesses = conditional_prefix_windows(quotes.time, quotes.cum_volume,
                                               trades["prefix_trade_times"], trades["prefix_trade_volumes"],
                                               config["measurement"]["conditional_quote_floor_precision_ms"],
                                               config["measurement"]["conditional_tick_floor_precision_ms"])
        part = observe_grid(quotes, witnesses, grid, config["measurement"]["maximum_quote_age_ms"], config["measurement"]["band_bps"])
        part.insert(0, "date", date)
        regime = "PRE_20260706" if date < config["regime_start"] else "POST_20260706"
        part.insert(1, "regime", regime)
        part["source_quote_sha256"] = streams["行情"]["sha256"]
        matrix_parts.append(part)
        phase = session_labels(quotes.time)
        continuous = np.isin(phase, ["AM", "PM"])
        for stream, counts in [("行情", Counter(phase)), ("逐笔委托", orders["session_counts"]), ("逐笔成交", trades["session_counts"])]:
            for session in ["PREOPEN", "AM", "MIDDAY", "PM", "CLOSE_WINDOW", "AFTER_HOURS", "POST_1530"]:
                session_rows.append({"date": date, "regime": regime, "stream": stream, "session": session, "rows": counts.get(session, 0)})
        book = observed_book_features(quotes, config["measurement"]["band_bps"])
        daily_rows.append({"date": date, "regime": regime, "quote_rows": len(quotes), "order_rows": orders["rows"], "trade_rows": trades["rows"],
                           "continuous_quotes": int(continuous.sum()), "continuous_non_second_quotes": int(((quotes.time % 1000 != 0) & continuous).sum()),
                           "quote_trade_flag_nonnull_rows": int(quotes.trade_flag.notna().sum()),
                           "order_non_centisecond_rows": orders["non_centisecond_rows"], "trade_non_centisecond_rows": trades["non_centisecond_rows"],
                           "source_quote_duplicate_time_rows": int(quotes.time.duplicated(keep=False).sum()),
                           "continuous_exact_prefix_quotes": int((witnesses.exact_cumulative_prefix & continuous).sum()),
                           "continuous_quotes_with_later_nominal_trade": int((witnesses.last_trade_after_nominal & continuous).sum()),
                           "continuous_legacy_conditional_compatible_quotes": int((witnesses.conditional_clock_compatible & continuous).sum()),
                           "continuous_fixed_band_quotes": int((book.fixed_10bp_band_visible & continuous).sum()),
                           "positive_iopv_rows": int(quotes.iopv.gt(0).sum()),
                           "after_1505_orders": orders["after_1505_rows"], "after_1505_status_S_orders": orders["after_1505_type_counts"].get("S", 0),
                           "after_1505_add_delete_orders": orders["after_1505_add_delete_rows"], "after_1505_trades": trades["after_1505_rows"],
                           "order_type_counts_json": json.dumps(orders["type_counts"], ensure_ascii=False),
                           "trade_code_counts_json": json.dumps(trades["type_counts"], ensure_ascii=False),
                           "strict_m1_reference_qualified": False, "strict_m2_field_qualified": False})
        if quotes.iopv.gt(0).any():
            iopv_rows.append(summarize_iopv(date, quotes, orders, trades, part))
        if number % 10 == 0 or number == len(by_day):
            print(f"已核对{number}/{len(by_day)}日；累计原文件{len(receipts)}个；日期{date}。", flush=True)
    if [x["date"] for x in iopv_rows] != config["positive_iopv_dates"]:
        raise ValueError("正IOPV日期与冻结输入清点不一致，不允许更换日期。")
    matrix = add_previous_day_counts(pd.concat(matrix_parts, ignore_index=True), 60, 5)
    matrix.to_parquet(out / "03_日期时间资格矩阵.parquet", index=False, compression="zstd")
    # csv只输出必要字段，完整条件范围仍在parquet；节省空间并便于无需代码查看。
    selected_columns = ["date", "regime", "hhmm", "grid_kind", "source_quote_row", "source_time", "quote_age_nominal_ms",
                        "source_nominal_band_observed", "conditional_clock_compatible", "conditional_range_before_endpoint",
                        "m2_source_field_pair", "calendar_prior_days", "prior_source_m2_pair_days", "prior_conditional_m2_pair_days",
                        "source_60day_m2_candidate", "conditional_60day_m2_candidate", "strict_point_qualified",
                        "strict_60day_m2_candidate", "local_diagnostic_reasons"]
    matrix[selected_columns].to_csv(out / "03_日期时间资格摘要.csv", index=False, encoding="utf-8-sig")
    daily, sessions = pd.DataFrame(daily_rows), pd.DataFrame(session_rows)
    daily.to_csv(out / "04_逐日来源核对.csv", index=False, encoding="utf-8-sig")
    sessions.to_csv(out / "05_三流会话覆盖.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(receipts).to_csv(out / "07_输入文件核对.csv", index=False, encoding="utf-8-sig")
    save_json(out / "02_IOPV与盘后逐日证据.json", {"status": "COMPLETED_SOURCE_FIELD_PROFILE_NOT_REFERENCE_ADMISSION", "days": iopv_rows})
    summary = write_summary(matrix, daily, sessions, iopv_rows, receipts, config)
    (out / "来源资格结论.md").write_text(report_markdown(summary, iopv_rows, config), encoding="utf-8")
    for frozen in registration["inputs"]:
        if digest(Path(frozen["path"])) != frozen["sha256"]:
            raise RuntimeError("运行期间冻结代码/配置/来源说明发生变化，结果不得接受。")
    summary["output_bytes_before_summary_index"] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if summary["output_bytes_before_summary_index"] > config["output_budget_bytes"]:
        raise RuntimeError("来源摘要超出20MiB预算，不允许自动删结果或继续扩大。")
    save_json(out / "summary.json", summary)
    save_json(out / "run_state.json", {"status": summary["status"], "completed_at": now(), "returns_status": "NOT_COMPUTED"})
    index = [{"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(out.iterdir()) if p.is_file() and p.name != "FILE_INDEX.csv"]
    pd.DataFrame(index).to_csv(out / "FILE_INDEX.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(json_value({"状态": summary["status"], "来源日期": summary["source_dates"],
                                 "资格矩阵行": summary["grid_rows"], "基线": summary["baseline_by_regime"],
                                 "严格M1参考日": 0, "严格M2候选": 0, "策略收益": "NOT_COMPUTED",
                                 "输出目录": str(out)}), ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        directory = ROOT / cfg["output_directory"]
        if directory.exists():
            failure_file = directory / "run_failure.json"
            if not failure_file.exists():
                save_json(failure_file, {"status": "RUN_FAILED_OUTPUTS_NOT_ACCEPTED", "failed_at": now(),
                                         "error": str(error), "traceback": traceback.format_exc(), "returns_computed": False})
        raise
