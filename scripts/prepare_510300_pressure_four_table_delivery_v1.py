"""从已存资格矩阵生成181日四表中间数据，不扫描原始三流或计算收益。"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_pressure_four_table_delivery_v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_file(path: Path, headers: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> None:
    cfg = read(CONFIG)
    out = ROOT / cfg["report_directory"]
    artifact = ROOT / cfg["artifact_directory"]
    if out.exists() or artifact.exists():
        raise SystemExit("181日四表交付目录已存在，拒绝覆盖。")
    inputs = [ROOT / name for name in cfg["inputs"]]
    records = [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)} for p in [CONFIG, Path(__file__), *inputs]]
    original = read(inputs[0])
    matrix = pq.read_table(inputs[2]).to_pylist()
    admission = read(inputs[4])
    feasibility = read(inputs[5])
    metadata = read(inputs[6])
    manifest = read(inputs[7])
    if len(matrix) != cfg["scope"]["expected_grid_rows"]:
        raise SystemExit("资格矩阵行数不符合已登记覆盖，需先查明输入版本。")
    dates = sorted({row["date"] for row in matrix})
    if len(dates) != cfg["scope"]["expected_source_days"] or any(row["strict_point_qualified"] for row in matrix):
        raise SystemExit("来源资格状态已改变，不能继续以原缺口制作交付。")
    if admission["returns_status"] != "NOT_COMPUTED" or feasibility["new_admitted_source_count"] != 0:
        raise SystemExit("原研究状态已改变，需按新实验结果另登记交付。")
    inventory = manifest["files"]
    if len(inventory) != 543 or sum(x["rows"] for x in inventory) != 79288590:
        raise SystemExit("来源清单与已完成181日三流不一致。")
    by_source = {(x["date"], x["stream"]): x for x in inventory}
    repo_metadata = next(x for x in metadata["sources"] if x["source_id"] == "hf_repo_metadata")
    remote_names = {x["rfilename"] for x in repo_metadata["siblings"]}
    if repo_metadata["revision"] != manifest["revision"]:
        raise SystemExit("目录版本与已存三流版本不同。")
    def source_url(day: str, stream: str) -> str:
        name = f"{day}/{stream}.parquet"
        if name not in remote_names:
            raise ValueError(f"固定仓库目录没有预期来源：{name}")
        return f"https://huggingface.co/datasets/{manifest['repo']}/blob/{manifest['revision']}/{quote(name)}"
    stages = {"M2_CONTINUOUS": "连续竞价网格", "CLOSE_PROBE": "收盘覆盖探针", "AFTER_HOURS_PROBE": "盘后覆盖探针"}
    sync_headers = [
        "日期", "网格时间（北京）", "观察ID", "网格类别", "制度阶段", "时点资格", "缺失原因",
        "来源ID", "源名义时间原值", "子集源行号（0起）", "最新价原值", "IOPV原值（未认证）",
        "名义中间价（元）", "名义价差（bp）", "10bp可见买盘（元）", "累计量原值",
        "名义报价年龄（秒）", "固定价带完整", "本地五分钟字段配对", "过去本地配对日数",
        "过去严格配对日数", "价格经济时间", "参考经济时间", "历史接收时间",
        "参考价值下界（元）", "参考价值上界（元）", "参考误差已界定", "对应PCF已验证", "来源URL"
    ]
    sync = []
    for row in matrix:
        source = by_source[(row["date"], "行情")]
        if row["source_quote_sha256"] != source["sha256"]:
            raise SystemExit("资格矩阵引用的源hash与来源清单不一致。")
        day = datetime.strptime(row["date"], "%Y%m%d").date().isoformat()
        age = row["quote_age_nominal_ms"]
        sync.append([day, row["minute"] / 1440, f"GRID_{row['date']}_{row['hhmm'].replace(':', '')}",
                     stages[row["grid_kind"]], "制度后" if row["regime"] == "POST_20260706" else "制度前",
                     "NO_VIEW", row["strict_reasons"], f"行情_{row['date']}", row["source_time"],
                     row["source_quote_row"], clean(row["source_price_raw"]), clean(row["source_iopv_raw"]),
                     clean(row["source_mid_cny"]), clean(row["source_spread_bps"]), clean(row["source_band_bid_depth_cny"]),
                     clean(row["source_cum_volume"]), age / 1000 if age is not None else None,
                     row["fixed_10bp_band_visible"], row["m2_source_field_pair"], row["prior_source_m2_pair_days"],
                     row["prior_strict_m2_pair_days"], None, None, None, None, None, False, None, source_url(row["date"], "行情")])
    event_headers = ["事件ID", "机制", "日期", "冲击起点", "固定观察终点", "恢复分类", "支持证据", "反对证据", "引用观察ID", "剩余净空间", "入场资格"]
    order_headers = ["订单ID", "事件ID", "机制", "证据类型", "方向", "提交时间", "限价（元）", "请求数量（份）", "确认时间", "成交数量（份）", "成交均价（元）", "撤单请求时间", "撤单确认时间", "状态", "证据来源"]
    result_headers = ["版本", "研究账户规模（元）", "来源日期数", "正式事件数", "已提交订单数", "确认成交数", "完成交易数", "成交率", "净期望", "胜率", "实际盈亏比", "尾部损失", "资金占用", "全账户夏普", "最大回撤", "计算状态", "未运行原因"]
    results = []
    for variant in [*original["m1"]["variants"], original["m2"]["variant"]]:
        for account in [original["account_cny"], original["sensitivity_account_cny"]]:
            results.append([variant, account, len(dates), None, 0, 0, 0, None, None, None, None, None, None, None, None,
                            "NOT_COMPUTED", "估值时点、历史可得性与执行证据未准入"])
    with inputs[3].open(encoding="utf-8-sig", newline="") as handle:
        daily_input = list(csv.DictReader(handle))
    if {x["date"] for x in daily_input} != set(dates):
        raise SystemExit("每日资格表与完整网格日期不一致。")
    source_days = {day: {s: by_source[(day, s)]["rows"] for s in ["行情", "逐笔委托", "逐笔成交"]} for day in dates}
    grids_per_day = Counter(x["date"] for x in matrix)
    daily_headers = ["日期", "制度阶段", "行情源行数", "委托源行数", "成交源行数", "测量网格数", "此前同制度源日期", "本地60日候选分钟", "条件60日候选分钟", "严格60日候选分钟", "M1参考状态", "M2事件检验状态", "缺失等于无事件"]
    daily = []
    for row in daily_input:
        day = row["date"]
        daily.append([datetime.strptime(day, "%Y%m%d").date().isoformat(), "制度后" if row["regime"] == "POST_20260706" else "制度前",
                      source_days[day]["行情"], source_days[day]["逐笔委托"], source_days[day]["逐笔成交"], grids_per_day[day],
                      int(row["calendar_prior_days"]), int(row["source_m2_candidate_slots"]), int(row["conditional_m2_candidate_slots"]),
                      int(row["strict_m2_candidate_slots"]), "NO_VIEW", "NOT_RUN_SOURCE_GATE_FAILED", False])
    source_headers = ["来源ID", "消息流", "日期", "源证券标签", "来源文件名", "原值子集SHA256", "行数", "固定版本来源URL"]
    sources = []
    for item in sorted(inventory, key=lambda x: (x["date"], x["stream"])):
        sources.append([f"{item['stream']}_{item['date']}", item["stream"], datetime.strptime(item["date"], "%Y%m%d").date().isoformat(),
                        ";".join(item["source_codes"]), Path(item["path"]).name, item["sha256"], item["rows"], source_url(item["date"], item["stream"])])
    dictionary = [
        ["观察ID", "日期与HH:MM形成唯一网格ID；不同网格可以引用同一源行。"],
        ["NO_VIEW", "原时点/估值合同不满足，不能由名义报价字段形成原策略观点。"],
        ["NOT_RUN_SOURCE_GATE_FAILED", "原事件检验尚未启动；正式事件数为空，不等于市场没有事件。"],
        ["NOT_COMPUTED", "净收益和账户指标没有计算；空白不能按零收益或零风险解释。"],
        ["名义中间价/价差/买盘", "沿用保存矩阵的源行测量值；不能证明该网格时刻已收到或同步可交易。"],
        ["最新价原值", "源价格整数尺度，已做收盘/10000交叉核对；本列保持原值。"],
        ["IOPV原值", "保持源值和零值；单位/流映射/经济时刻未认证，零不是有效参考价值。"],
        ["名义报价年龄", "仅按未认证的厂商time计算；盘后探针该值只描述，不用5秒规则否定M1。"],
        ["本地配对/60日候选", "只反映源行字段和来源日期；不等于严格合格基线或正式事件。"],
        ["条件60日候选", "只描述最后名义源行及旧截断秒假设，不是所有更早行的可用性上界。"],
        ["价格/参考经济时间与历史接收", "当前未识别，明确留空；不把源名义time或现在下载时间填入。"],
        ["对应PCF已验证", "没有181日逐时点对应证据，留空；旧单日PCF核对不推广到全部日期。"],
        ["空事件/订单表", "已登记正式事件0行、提交订单0行；没有运行正式事件检验，也没有验证填单概率。"],
        ["研究账户规模", "20万元/2万元是原研究参数，不是账户余额或当前仓位。"],
        ["原M2恢复基线", "固定合同使用冲击时已知中位数，未改成终点中位数。"],
        ["NO_RECEIVE_CLOCK", "历史接收时钟缺失。"],
        ["UNVERIFIED_VENDOR_TIME_MAPPING", "厂商名义时间与经济时刻映射未验证。"],
        ["NO_SYNC_REFERENCE_CONTRACT", "没有同步参考估值合同。"],
        ["NO_REFERENCE_ERROR_BOUND", "参考估值误差界缺失。"],
        ["NO_AFTER_HOURS_QUEUE_REPLAY", "盘后队列重放或个人执行证据未验证。"],
        ["IOPV_MAPPING_UNVERIFIED", "IOPV流、单位及经济时点映射未验证。"],
        ["CLOSE_COMPONENTS_UNVERIFIED", "同步收盘成分估值未验证。"],
    ]
    public_sources = [
        ["历史Level-2原件", f"https://huggingface.co/datasets/{manifest['repo']}", manifest["revision"], "181日三流已存；接收/估值/队列未准入"],
        ["LDDS2.0.10字段合同", "https://www.sseinfo.com/services/assortment/document/interface/c/10759998/files/f3ca62e905764efaa3983a7c20d9e1d9.pdf", "2024接口族", "不能与2026 STEP自动混用"],
        ["STEP0.63字段合同", "https://www.sse.com.cn/services/tradingtech/data/c/10832589/files/5ea26a2949c943a7ae2c9838c4c252bf.pdf", "2026-09-18", "MDE01有效IOPV；导出映射尚未证明"],
        ["alphat04候选", "https://huggingface.co/datasets/alphat04/Tick-by-Tick-Orders-China", "2026截止6月5日", "小字段文件403，没有新增合格样本"],
        ["QuantDB字段/额度", "https://www.quantdb.cn/docs/fields.html", "官网公开页", "有免费额度，未列必要历史IOPV/接收字段"],
        ["次方字段/权限", "https://www.cifangquant.com/docs/data-api", "官网公开页", "历史分钟SVIP；分钟时间是bar开始"],
    ]
    tables = {
        "同步行情": {"headers": sync_headers, "rows": sync, "date_columns": [0], "time_columns": [1], "note": "2026-01-05至09-30，全部264格/日；名义报价未准入原时点合同。"},
        "事件": {"headers": event_headers, "rows": [], "date_columns": [2], "time_columns": [], "note": "正式事件检验未运行；0条登记记录不能解释为没有市场事件。"},
        "订单": {"headers": order_headers, "rows": [], "date_columns": [], "time_columns": [], "note": "没有提交订单或合格成交回执；未估计成交率，未把未知成交作未成交。"},
        "结果": {"headers": result_headers, "rows": results, "date_columns": [], "time_columns": [], "note": "3版本×2账户规模；全部收益字段未计算，账户规模仅为研究参数。"},
        "日期覆盖": {"headers": daily_headers, "rows": daily, "date_columns": [0], "time_columns": [], "note": "181日来源覆盖；本地/条件候选不是严格合格日，不将缺失日作无事件。"},
        "数据来源": {"headers": source_headers, "rows": sources, "date_columns": [2], "time_columns": [], "note": "543份目标原值子集；源标签保留510300.SZ，规范研究对象为510300.SH。"},
        "字段说明": {"headers": ["字段或状态", "含义与限制"], "rows": dictionary, "date_columns": [], "time_columns": [], "note": "原协议、缺失值、时点与计数的解释。"},
        "公开来源": {"headers": ["来源", "URL", "版本或范围", "本轮结论"], "rows": public_sources, "date_columns": [], "time_columns": [], "note": "公开合同与候选来源；不是准入或收益证明。"}
    }
    out.mkdir(parents=True)
    artifact.mkdir(parents=True)
    registered_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    save(out / "registration.json", {"registered_at": registered_at, "configuration": cfg, "inputs": records,
                                       "builder": "build_pressure_recovery_181day_workbook.mjs", "original_results_overwritten": False})
    for name, filename in {"同步行情": "01_同步行情表.csv", "事件": "02_事件表.csv", "订单": "03_订单表.csv", "结果": "04_结果表.csv", "日期覆盖": "05_日期覆盖.csv", "数据来源": "06_原值来源.csv", "字段说明": "07_字段说明.csv"}.items():
        csv_file(out / filename, tables[name]["headers"], tables[name]["rows"])
    payload = {"delivery_id": cfg["delivery_id"], "registered_at": registered_at, "target_security": cfg["target_security"],
               "source_revision": manifest["revision"], "tables": tables,
               "state": {"source_days": len(dates), "source_rows": 79288590, "grid_rows": len(matrix), "source_files": len(inventory),
                         "formal_event_status": "NOT_RUN_SOURCE_GATE_FAILED", "formal_event_count": None,
                         "strict_qualified_grid_rows": 0, "actual_orders": 0, "actual_fills": 0,
                         "net_expectancy": None, "net_sharpe": None, "goal_achieved": False}}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with (out / "workbook_data.json.gz").open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
            compressed.write(data)
    save(out / "prepared_summary.json", {"status": "FOUR_TABLE_DATA_PREPARED_NOT_FINANCIAL_VALIDATION", "state": payload["state"],
                                          "table_row_counts": {name: len(value["rows"]) for name, value in tables.items()},
                                          "csv_file_count": 7, "new_network_requests": 0, "fee_usd": 0,
                                          "original_goal_achieved": False, "matrix_read_only": True, "raw_three_streams_rescanned": False})
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if size > cfg["scope"]["maximum_new_output_bytes"]:
        raise SystemExit("交付中间资料超预算，停止工作簿创建。")
    print(f"四表数据已准备：{len(sync):,}网格、{len(daily)}日期、{len(sources)}源文件；正式事件未运行、收益未计算。报告{size:,}字节。")


if __name__ == "__main__":
    main()
