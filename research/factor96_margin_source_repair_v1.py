"""定点补齐11个深市官方历史记录；不改变任何交易规则或旧输入。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import pandas as pd
import requests

from research import factor96_margin_repair_v1 as base

PARENT = base.OUT
REPAIR = PARENT / "data_repair"
ACTIVE = PARENT / "completed_run"
URL = "https://www.szse.cn/api/report/ShowReport/data"
FIELDS = {"rzmre":"jrrzmr", "rzye":"jrrzye", "rqmcl":"jrrjmc", "rqyl":"jrrjyl", "rqye":"jrrjye", "rzrqye":"jrrzrjye"}


def fetch_day(day):
    path = REPAIR / ("szse_"+day.replace("-", "")+".json")
    retrieved = base.now()
    if not path.exists():
        session = requests.Session()
        session.trust_env = False
        r = session.get(URL, params={"SHOWTYPE":"JSON", "CATALOGID":"1837_xxpl", "txtDate":day, "tab1PAGENO":"1"},
            headers={"Referer":"https://www.szse.cn/disclosure/margin/margin/index.html", "User-Agent":"Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        path.write_bytes(r.content)
    j = json.loads(path.read_text(encoding="utf-8"))[0]
    assert j["metadata"]["subname"] == day and len(j["data"]) == 1
    labels = j["metadata"]["cols"]
    assert all("亿" in labels[key] for key in FIELDS.values())
    row = {"date":pd.Timestamp(day)}
    for name, key in FIELDS.items():
        row[name] = float(j["data"][0][key].replace(",", ""))*1e8
    assert row["rzye"] > 0 and row["rzmre"] > 0
    return row, {"stat_date":day, "retrieved_at":retrieved, "source_url":URL,
        "query":{"CATALOGID":"1837_xxpl","txtDate":day,"SHOWTYPE":"JSON"},
        "file":path.name, "sha256":base.digest(path), "source_unit":"亿元；显示两位小数，人民币精度约100万元",
        "historical_first_publication_receipt":False}


def collect():
    REPAIR.mkdir(exist_ok=True)
    failure = json.loads((PARENT/"run_failure_01.json").read_text(encoding="utf-8"))
    assert failure["generated_accounts"] == 0 and len(failure["missing"]) == 11
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch_day, failure["missing"]))
    sz = pd.DataFrame([r[0] for r in results]).set_index("date")
    sse_path = base.ROOT/"data/raw/market_margin_leverage_v0/sse_margin_history_official.parquet"
    sh = pd.read_parquet(sse_path).set_index("date").loc[sz.index]
    sh.reset_index().to_parquet(REPAIR/"sse_local_official_matching_days.parquet", index=False)
    old = pd.read_parquet(PARENT/"inputs/margin.parquet")
    added = []
    for day in sz.index:
        row = {"date":day, "source":"SSE_LOCAL_OFFICIAL_PLUS_SZSE_OFFICIAL_11_DAY_REPAIR",
            "publication_rule":old.publication_rule.iloc[0]}
        for col in FIELDS:
            row[col+"_sse"], row[col+"_szse"] = sh.loc[day,col], sz.loc[day,col]
            row["market_"+col] = sh.loc[day,col]+sz.loc[day,col]
        added.append(row)
    combined = pd.concat([old,pd.DataFrame(added)],ignore_index=True).sort_values("date").reset_index(drop=True)
    combined["market_rzye_change"] = combined.market_rzye.diff()
    combined["market_financing_balance_identity_residual"] = combined.market_rzye_change-combined.market_rzmre
    prices = pd.read_parquet(PARENT/"inputs/market.parquet")
    expected = prices.loc[prices.date.between("2015-01-05","2025-12-31"),"date"].tolist()
    assert combined.date.tolist() == expected and len(combined) == 2674
    pd.testing.assert_frame_equal(old.set_index("date")[["market_rzye","market_rzmre"]],
        combined.set_index("date").loc[old.date,["market_rzye","market_rzmre"]])
    combined.to_parquet(REPAIR/"margin_complete.parquet", index=False)
    base.save(REPAIR/"source_receipt.json", {"at":base.now(),"missing_dates":failure["missing"],
        "records_added":11,"unchanged_existing_balance_and_buy_rows":len(old),"complete_days":2674,
        "old_file_sha256":base.digest(PARENT/"inputs/margin.parquet"),
        "new_file_sha256":base.digest(REPAIR/"margin_complete.parquet"),
        "sse_original_path":str(sse_path),"sse_original_sha256":base.digest(sse_path),
        "sources":[r[1] for r in results],"preliminary_request_failure":"默认代理连接发生TLS EOF，改为不读取代理环境后正常取得官方HTTPS响应；未关闭证书校验。",
        "strategy_or_parameter_changes":0,"new_strategy_results_seen_before_repair":0}, True)
    print("官方缺口补齐完成：11日，完整融资2674日，既有余额和买入值未改动。",flush=True)


def configure():
    base.OUT = ACTIVE
    base.MARGIN = REPAIR/"margin_complete.parquet"
    base.STUDY = "510300_FACTOR96_MARGIN_REPAIR_V1_DATA_COMPLETION"


def freeze():
    base.save(REPAIR/"metadata.json", {"status":"COMPLETE_CALENDAR_AFTER_OFFICIAL_11_DAY_REPAIR",
        "original_metadata":json.loads((PARENT/"inputs/margin_metadata.json").read_text(encoding="utf-8")),
        "repair":json.loads((REPAIR/"source_receipt.json").read_text(encoding="utf-8")),
        "historical_first_publication_receipts":"NOT_ESTABLISHED"},True)
    configure()
    ACTIVE.mkdir(exist_ok=True)
    for name in ["factor_registry.json","strategy_registry.json"]:
        shutil.copy2(PARENT/name,ACTIVE/name)
    base.freeze()
    base.save(ACTIVE/"data_completion_freeze.json", {"at":base.now(),
        "reason":"旧融资合并表缺11个深市日；前次运行在标签和账户前停止，官方补齐后重新冻结数据，交易代码完全相同。",
        "unchanged_core_code_sha256":base.digest(Path(base.__file__)),
        "driver_sha256":base.digest(Path(__file__)),
        "original_freeze_sha256":base.digest(PARENT/"freeze.json"),
        "source_receipt_sha256":base.digest(REPAIR/"source_receipt.json"),
        "actual_margin_source":str(base.MARGIN),"strategy_or_parameter_changes":0}, True)


def run():
    configure()
    receipt = json.loads((ACTIVE/"data_completion_freeze.json").read_text(encoding="utf-8"))
    assert receipt["driver_sha256"] == base.digest(Path(__file__))
    assert receipt["source_receipt_sha256"] == base.digest(REPAIR/"source_receipt.json")
    base.run()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["collect","freeze","run"])
    args=parser.parse_args()
    {"collect":collect,"freeze":freeze,"run":run}[args.action]()
