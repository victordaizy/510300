"""重建宏观研究范围、信息更新账本和说明图；不运行新收益模型或账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_macro_dynamic_reframe_v1"
PARENT = ROOT / "reports/research/510300_money_consensus_increment_v2"
STUDY = "510300_MACRO_DYNAMIC_REFRAME_V1"
TZ = ZoneInfo("Asia/Shanghai")
INPUTS = {
    "market.parquet": PARENT / "inputs/market_daily.parquet",
    "money_104.csv": PARENT / "inputs/104个月共识选择.csv",
    "old_predictions.csv": PARENT / "results/A_B全部逐期预测.csv",
    "old_events.csv": PARENT / "results/全部准入事件_固定五日.csv",
    "old_models.json": PARENT / "results/全部模型训练记录.json",
    "policy_context.json": PARENT / "inputs/policy_context_events.json",
    "pmi_new_orders.parquet": ROOT / "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet",
    "operation_rate_records.parquet": ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc_7d_reverse_repo_rate_changes_anchor_20150105_20260814.parquet",
}
SOURCES = {
    "csrc_20240924_transcript": "https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml",
    "gov_pboc_wechat_20240927": "https://app.www.gov.cn/govdata/gov/202409/27/519932/article.html",
    "people_20240927_0914": "https://finance.people.com.cn/n1/2024/0927/c1004-40329480.html",
    "fed_transmission_2025": "https://www.federalreserve.gov/econres/feds/decoding-equity-market-reactions-to-macroeconomic-news.htm",
}


def now() -> str:
    return datetime.now(TZ).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def prepare() -> None:
    for sub in ("inputs", "evidence", "results", "figures", "raw", "receipts", "code"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    if (OUT / "scope_freeze.json").exists():
        raise RuntimeError("研究范围已保存，禁止覆盖冻结记录")
    identities = []
    for name, src in INPUTS.items():
        dst = OUT / "inputs" / name
        shutil.copy2(src, dst)
        identities.append({"input": name, "original": src.relative_to(ROOT).as_posix(), "sha256": digest(dst), "bytes": dst.stat().st_size})
    write_json(OUT / "scope_freeze.json", {
        "study_id": STUDY, "created_at": now(),
        "user_correction": "宏观主线不局限M1/M2或固定五日；包括经济传导、不同政策、当时市场状态与持续更新。资金字段缺口不阻断独立宏观假设。",
        "this_stage": "重订立项、核对旧实验诊断、清点直接数据、构建逐日信息账本及政策时序案例；不是新二十日策略检验。",
        "first_independent_channel": "增长状态：PMI新订单的当次水平与月度变化；定位为经济状态，不冒充市场预期差。政策宣布与落地另建事件链。",
        "subsequent_common_horizon": 20, "horizon_status": "用户建议采纳的研究期限，无最优性声明，需独立执行合同后读取对应标签",
        "update_clock": "每周最后交易日收盘复核；已准入的新数据或政策在能够核实内容的首个收盘额外复核，下一开盘执行；发布会开始不等于全部内容已知。",
        "cash_capital": {"main": 200000, "comparison": 20000},
        "old_five_day_result": "REJECTED_FROZEN_NO_RELIABLE_MESSAGE_INCREMENT",
        "old_five_day_scope": "只拒绝该剪刀差预期差表达与固定五日模型；不推导宏观整体无效或稳定负作用。",
        "baselines": ["零收益预测", "当时成熟训练标签均值", "固定简单价格模型", "相同模型与时钟下移除新增宏观信息"],
        "future_tests_separate": ["二十日收益均值增量", "下行风险预测", "每周与事件触发更新的实际决策差异", "资金与退出模块"],
        "no_wait_for_all_data": True, "new_model_fits": 0, "new_20d_returns_read": 0, "new_accounts": 0,
        "history_is_independent_holdout": False,
        "known_case_after_review": "2024-08所属数据的9月19日至25日窗口，是用户已指出的事后案例，只演示时钟，不用于选择规则或证明政策收益。",
        "inputs": identities,
    })
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("宏观研究范围、输入快照和阶段边界已保存。")


def fetch(key: str) -> None:
    path = OUT / "raw" / (key + ".html")
    receipt = OUT / "receipts" / (key + ".json")
    if receipt.exists():
        print("来源回执已存在：" + key)
        return
    url = SOURCES[key]
    proc = subprocess.run(["curl.exe", "--silent", "--show-error", "--location", "--connect-timeout", "10", "--max-time", "40", "--output", str(path), "--write-out", "%{json}", url], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
    meta = json.loads(proc.stdout) if proc.stdout.strip() else {}
    ok = proc.returncode == 0 and meta.get("http_code") == 200 and path.exists()
    write_json(receipt, {"key": key, "url": url, "retrieved_at": now(), "http": meta.get("http_code"), "transport_exit": proc.returncode, "status": "RETRIEVED_NOT_YET_ADJUDICATED" if ok else "FETCH_FAILED", "bytes": path.stat().st_size if path.exists() else 0, "sha256": digest(path) if path.exists() else None, "error": proc.stderr[:400]})
    print(key + ("：已取得原文" if ok else "：未取得原文"))


def old_diagnostic() -> dict:
    pred = pd.read_csv(OUT / "inputs/old_predictions.csv")
    events = pd.read_csv(OUT / "inputs/old_events.csv").set_index("event_id")
    models = json.loads((OUT / "inputs/old_models.json").read_text(encoding="utf-8"))
    lookup = {r["event_id"]: r for r in models if r["arm"] == "A"}
    means = []
    for r in pred.itertuples():
        ids = lookup[r.event_id]["training_events"]
        tr = events.loc[ids]
        assert (pd.to_datetime(tr.exit_at, utc=True) < pd.to_datetime(r.observation_at, utc=True)).all()
        means.append(float(tr.residual_5d_gross_return.mean()))
    pred["prediction_mature_mean"] = means
    pred["prediction_zero"] = 0.0
    metrics = []
    for model in ("A", "B", "mature_mean", "zero"):
        values = pred["prediction_" + model]
        mse = float(np.mean((values - pred.actual_5d_return) ** 2))
        metrics.append({"model": model, "n": len(pred), "mse": mse, "rmse_pp": np.sqrt(mse)*100, "positive_predictions": int((values>0).sum()), "negative_predictions": int((values<0).sum())})
    pred.to_csv(OUT / "results/旧五日实验_补充基准逐事件.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(OUT / "results/旧五日实验_补充基准汇总.csv", index=False, encoding="utf-8-sig")
    # 同一已冻结区块方法增加双侧描述区间，不改变原单侧门槛。
    diff = pred.MSE_improvement_A_minus_B.to_numpy()
    rng = np.random.default_rng(5103005)
    draws = []
    for _ in range(10000):
        starts = rng.integers(0, len(diff)-4+1, size=int(np.ceil(len(diff)/4)))
        idx = np.concatenate([np.arange(s,s+4) for s in starts])[:len(diff)]
        draws.append(float(diff[idx].mean()))
    interval = np.quantile(draws, [0.05,0.95]).tolist()
    pd.DataFrame({"A_minus_B_mean_squared_error_gain":draws}).to_csv(OUT/"results/旧五日诊断_固定抽样均值.csv",index=False,encoding="utf-8-sig")
    case = events.loc["MONEY_2024-08"]
    result = {"status": "POST_HOC_DIAGNOSTIC_ONLY_OLD_VERDICT_UNCHANGED", "models": metrics,
        "both_A_B_negative_all_24": bool((pred.prediction_A<0).all() and (pred.prediction_B<0).all()),
        "relative_B_MSE_increase": metrics[1]["mse"]/metrics[0]["mse"]-1,
        "A_minus_B_error_90pct_two_sided": interval,
        "stable_negative_effect_established": bool(interval[1]<0),
        "case": {k: str(case[k]) for k in ["stat_month","published_at","observation_at","entry_date","exit_date"]},
        "case_gross_return": float(case.residual_5d_gross_return),
        "new_fits": 0, "new_accounts": 0}
    write_json(OUT / "results/旧五日诊断.json", result)
    return result


def derive() -> None:
    freeze = json.loads((OUT / "scope_freeze.json").read_text(encoding="utf-8"))
    for item in freeze["inputs"]:
        assert digest(OUT/"inputs"/item["input"]) == item["sha256"]
    diagnostic = old_diagnostic()
    pmi = pd.read_parquet(OUT / "inputs/pmi_new_orders.parquet").sort_values("available_at")
    rates = pd.read_parquet(OUT / "inputs/operation_rate_records.parquet").sort_values("published_at")
    market = pd.read_parquet(OUT / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    money = pd.read_csv(OUT / "inputs/money_104.csv")
    evidence = []
    for kind, frame, pathcol, hashcol in [("PMI_NEW_ORDERS",pmi,"raw_path","source_hash"),("RATE_OBSERVED_NOTICE",rates,"raw_path","raw_sha256")]:
        for row in frame.to_dict("records"):
            path = ROOT / row[pathcol]
            evidence.append({"kind":kind,"source_url":row["source_url"],"raw_path":row[pathcol],"expected_sha256":row[hashcol],"current_sha256":digest(path) if path.exists() else None,"hash_match":path.exists() and digest(path)==row[hashcol]})
    assert all(r["hash_match"] for r in evidence)
    pd.DataFrame(evidence).to_csv(OUT/"evidence/继承原文身份核对.csv",index=False,encoding="utf-8-sig")
    pmi["available_at_local"] = pd.to_datetime(pmi.available_at,utc=True).dt.tz_convert(TZ).dt.tz_localize(None)
    pmi["pmi_change_1m"] = pmi.first_release_value.diff()
    pmi["pmi_above50"] = pmi.first_release_value-50
    market["date"] = pd.to_datetime(market.date)
    market["decision_at"] = market.date + pd.Timedelta(hours=15)
    panel = pd.merge_asof(market, pmi[["available_at_local","reference_period","first_release_value","pmi_change_1m","pmi_above50"]],left_on="decision_at",right_on="available_at_local",direction="backward")
    rates["rate_notice_at"] = pd.to_datetime(rates.published_at)
    panel = pd.merge_asof(panel,rates[["rate_notice_at","seven_day_rate_percent"]],left_on="decision_at",right_on="rate_notice_at",direction="backward")
    # 继承表只被用作“本地样本已记录的操作利率”，不得当作完整的政策首次宣布链。
    panel.loc[panel.date>pd.Timestamp("2026-08-14"), "seven_day_rate_percent"] = np.nan
    panel.loc[panel.date>pd.Timestamp("2026-07-31"), ["first_release_value","pmi_change_1m","pmi_above50"]] = np.nan
    panel["rate_status"] = "INHERITED_OPERATION_RECORD_NOT_FIRST_POLICY_ANNOUNCEMENT"
    panel["close_drawdown"] = panel.close/panel.close.cummax()-1
    panel["total_return_drawdown"] = panel.wealth/panel.wealth.cummax()-1
    panel.to_csv(OUT/"results/全部日线_增长与操作利率已知记录.csv",index=False,encoding="utf-8-sig")
    panel.to_parquet(OUT/"results/全部日线_增长与操作利率已知记录.parquet",index=False)
    events = []
    for row in pmi.to_dict("records"):
        events.append({"event_id":"PMI_NEW_ORDERS_"+row["reference_period"],"family":"增长","information_kind":"已公布经济状态","available_at":row["available_at_local"].isoformat(),"reference_period":row["reference_period"],"value":float(row["first_release_value"]),"unit":"扩散指数","expectation":None,"surprise":None,"admission":"INHERITED_SOURCE_CLOCK_RECONSTRUCTION","source_url":row["source_url"]})
    for row in rates.to_dict("records"):
        events.append({"event_id":"RATE_RECORD_"+str(pd.Timestamp(row["notice_date"]).date()),"family":"利率条件","information_kind":"已记录操作利率","available_at":row["rate_notice_at"].isoformat(),"reference_period":str(pd.Timestamp(row["notice_date"]).date()),"value":float(row["seven_day_rate_percent"]),"unit":"%","expectation":None,"surprise":None,"admission":"NOT_ADMITTED_AS_FIRST_POLICY_ANNOUNCEMENT","source_url":row["source_url"]})
    pd.DataFrame(events).sort_values("available_at").to_csv(OUT/"results/宏观信息事件账本_继承来源.csv",index=False,encoding="utf-8-sig")
    # 用户指出的案例是已知事后样本；这里只保存真实公开内容和可执行时钟。
    csrc = OUT/"raw/csrc_20240924_transcript.html"
    govt = OUT/"raw/gov_pboc_wechat_20240927.html"
    if not csrc.exists() or not govt.exists():
        raise RuntimeError("政策原文未取得，不能生成已核实的政策链")
    # 原网页将时、分、秒分放在多个 span 内；核对数字时去掉标签间空白。
    csrc_text = "".join(BeautifulSoup(csrc.read_bytes(),"html.parser").get_text("",strip=True).split())
    govt_text = "".join(BeautifulSoup(govt.read_bytes(),"html.parser").get_text("",strip=True).split())
    assert "09:10:58" in csrc_text and "09:19:36" in csrc_text and "1.5%" in csrc_text and "0.5个百分点" in csrc_text
    assert "9月27日" in govt_text and "1.50%" in govt_text and "中国人民银行微信" in govt_text
    policy_case = [
        {"event_id":"POLICY_20240924_RATE_ANNOUNCED","available_at":"2024-09-24T09:19:36+08:00","announcement_segment_start":"2024-09-24T09:10:58+08:00","kind":"宣布","label":"7天逆回购利率将从1.70%降至1.50%","rate_old":1.7,"rate_new":1.5,"amount":None,"unit":"%","effective_date":None,"expectation":None,"surprise":None,"source_url":SOURCES["csrc_20240924_transcript"],"clock_evidence":"发言段从09:10:58开始；采用下一时间标记09:19:36作为内容可用保守上界；历史重建"},
        {"event_id":"POLICY_20240924_RRR_ANNOUNCED","available_at":"2024-09-24T09:19:36+08:00","announcement_segment_start":"2024-09-24T09:10:58+08:00","kind":"宣布","label":"近期降准0.5个百分点，约1万亿元长期流动性","rate_old":None,"rate_new":None,"amount":10000,"unit":"亿元，公告估计","effective_date":None,"expectation":None,"surprise":None,"source_url":SOURCES["csrc_20240924_transcript"],"clock_evidence":"按发言段后的时间标记取上界；不得把拟释放金额当成已买入股票资金"},
        {"event_id":"POLICY_20240927_RATE_EFFECTIVE","available_at":"2024-09-27T23:59:59+08:00","kind":"实施","label":"7天逆回购利率降至1.50%","rate_old":1.7,"rate_new":1.5,"amount":None,"unit":"%","effective_date":"2024-09-27","expectation":None,"surprise":None,"source_url":SOURCES["gov_pboc_wechat_20240927"],"clock_evidence":"政府网转载央行微信，页面为日期精度；保守上界不是实际首发时间"},
    ]
    pd.DataFrame(policy_case).to_csv(OUT/"results/政策宣布与实施_九月案例.csv",index=False,encoding="utf-8-sig")
    case = diagnostic["case"]
    window = market[(market.date>=pd.Timestamp("2024-09-13")) & (market.date<=pd.Timestamp("2024-09-30"))].copy()
    window["event"] = ""
    annotations={"2024-09-13":"8月M1/M2公布","2024-09-18":"旧实验观察收盘","2024-09-19":"旧实验标签开始","2024-09-24":"新政策09:10:58宣布；收盘复核","2024-09-25":"新判断最早次日开盘执行；旧标签结束","2024-09-26":"政治局经济部署","2024-09-27":"利率正式实施"}
    for date, label in annotations.items():
        window.loc[window.date==pd.Timestamp(date),"event"] = label
    window.to_csv(OUT/"results/九月案例_全部交易日.csv",index=False,encoding="utf-8-sig")
    # 成交时钟演示，不是新策略收益；同样的旧退出点只是为了排除已经发生的涨幅。
    prices = market.set_index("date")
    old_price = float(prices.loc[pd.Timestamp(case["entry_date"]),"open"])
    update_price = float(prices.loc[pd.Timestamp("2024-09-25"),"open"])
    exit_price = float(prices.loc[pd.Timestamp("2024-09-25"),"close"])
    case_clock = {"old_entry_open":old_price,"old_exit_close":exit_price,"old_window_return":exit_price/old_price-1,
        "update_observation":"2024-09-24T15:00:00+08:00","update_first_execution":"2024-09-25T09:30:00+08:00","update_open":update_price,
        "illustrative_update_open_to_old_exit":exit_price/update_price-1,"is_strategy_return":False,"is_policy_causal_contribution":False,
        "timing_gap":{"old_table_first_record":"2024-09-29","policy_announcement":"2024-09-24T09:10:58+08:00","effective_date":"2024-09-27"}}
    write_json(OUT/"results/九月案例_时钟与价格核对.json",case_clock)
    summary={"study_id":STUDY,"status":"COMPLETED_SCOPE_REFRAME_DATA_LEDGER_AND_DIAGNOSTIC_ONLY",
        "created_at":now(),"daily_prices":len(market),"price_end":str(market.date.max().date()),"pmi_months":len(pmi),"pmi_first_month":str(pmi.reference_period.iloc[0]),"pmi_last_month":str(pmi.reference_period.iloc[-1]),
        "operation_rate_records":len(rates),"first_policy_announcement_coverage_established":False,"inherited_raw_hashes_checked":len(evidence),"macro_information_ledger_rows":len(events),
        "money_104_months":len(money),"money_pair_months":int(money.consensus_admitted.sum()),"case_policy_rows":len(policy_case),
        "old_diagnostic":diagnostic,"new_twenty_day_models":0,"new_accounts":0,"new_20d_labels":0,"whole_macro_program_complete":False,
        "capital_main":200000,"capital_cost_comparison":20000,"full_account_target_established":False,"independent_forward_events":0}
    write_json(OUT/"results/summary.json",summary)
    print(json.dumps({"日线":len(market),"增长月份":len(pmi),"操作利率记录":len(rates),"旧基准诊断":diagnostic["models"],"九月案例":case_clock,"新模型":0,"新账户":0},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="宏观研究重订与信息账本")
    parser.add_argument("action",choices=["prepare","fetch","derive"])
    parser.add_argument("--source",choices=list(SOURCES))
    args=parser.parse_args()
    if args.action=="prepare":
        prepare()
    elif args.action=="fetch":
        if not args.source:
            parser.error("获取来源时必须提供 --source")
        fetch(args.source)
    else:
        derive()
