"""固定两个病例的成分贡献范围，并保存指定公司半年度报告。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_constituent_external_bridge_v16"
P10 = "reports/research/510300_constituent_risk_transmission_v10/inputs/"
P15 = "reports/research/510300_external_discount_clock_v15/"
INPUTS = {"returns.parquet": P10 + "returns.parquet", "membership.parquet": P10 + "membership.parquet",
          "weights.parquet": P10 + "weights.parquet", "industry.parquet": P10 + "industry.parquet",
          "monthly.csv": P15 + "inputs/monthly.csv", "market.csv": P15 + "inputs/market.csv",
          "daily_paths.csv": P15 + "inputs/daily_paths.csv", "news_segments.csv": P15 + "results/全部消息分段_同一入场本金贡献.csv"}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("本轮范围已固定，不覆盖。")
    for d in ["inputs", "sources", "results", "figures", "code"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    protocol = {"at": datetime.now().astimezone().isoformat(), "study_id": "510300_CONSTITUENT_EXTERNAL_BRIDGE_V16",
        "previous_turn_classification": "PROGRESS：外部利率、汇率与公告时钟已经落盘并复算。",
        "cases": ["2020-08", "2022-08"], "question": "外部压力背景中，哪些成分实际贡献收益和下行波动；当时已知的公司暴露与盈利是否支持单一解释？",
        "membership_and_weights": "观察日官方换样重建的300只股票；沿V10用此前最近、62日内的完整权重快照，归一化后作为参考本金份额。不是ETF真实日持仓或认证历史首版。",
        "industry": "沿已保存月度参考行业标签并列未知；不把当前版本标签提升为已认证的历史行业分类。跨年行业变更不强行拼接。",
        "price_window": "保持两个原E0入场日与退出日；成分只有可靠收盘账本，以起点收盘至原退出收盘构建全部20个日贡献，同时列原入场日收盘至退出的19日贡献。原E0开盘收益单列，绝不冒充相同口径。",
        "holding_identity": "起点1元、按参考权重买入各成分；分红留现金、不再投资，送转只改变股数。每日贡献与行业贡献直接相加；不每天重置权重。",
        "risk_identity": "每日个股本金贡献除以上日参考篮子净值，求各股与篮子收益的协方差贡献及下行平方贡献；行业加总回完整篮子。",
        "missing": "不填供应商缺失；2020-10-12的600109仅在官方停复牌和无公司行动证据吻合后建立本轮覆盖记录，原冻结数据不改。否则全篮子保持未完成，已知部分不重标权重。",
        "company_selection": "每个病例按起点参考权重前五名，合计10个公司报告期、6家不同公司。选例不使用后续涨跌。",
        "company_evidence": "仅取起点前已披露的2020或2022半年度报告；比较收入、归母利润、现金流以及境外收入、外币净敞口和利率敏感性中实际披露的字段。缺失保留未知，银行保险不与制造业现金流机械比较。",
        "source_budget": "10份半年度报告，另最多2份用于600109停复牌核对的原公告。目录仅用于找原件；不以分析师研报代替市场一致预期。",
        "information_change": "报告期数据是起点背景，非起点当天的新消息；原窗口内仍有后续消息，本轮不建立完备公司消息目录。",
        "new_models": 0, "new_accounts": 0, "orders_authorized": False, "independent_validation": False, "goal_achieved": False}
    save("protocol.json", protocol)
    shutil.copy2(OUT / "protocol.json", ROOT / "config/510300_constituent_external_bridge_v16.json")
    items = []
    for name, source in INPUTS.items():
        raw = ROOT / source
        shutil.copy2(raw, OUT / "inputs" / name)
        items.append({"name": name, "source": source, "sha256": sha256(raw.read_bytes()).hexdigest()})
    weights = pd.read_parquet(OUT / "inputs/weights.parquet")
    weights.trade_date = pd.to_datetime(weights.trade_date)
    selected = []
    for case, day in [("2020-08", "2020-09-11"), ("2022-08", "2022-09-09")]:
        date = weights.loc[weights.trade_date < pd.Timestamp(day), "trade_date"].max()
        w = weights[weights.trade_date == date].sort_values(["weight", "con_code"], ascending=[False, True]).head(5)
        for rank, r in enumerate(w.itertuples(), 1):
            selected.append({"stat_month": case, "origin_date": day, "symbol": r.con_code, "reference_rank": rank,
                             "weight_date": str(date.date()), "raw_weight_percent": r.weight, "required_report": case[:4] + "-06-30"})
    pd.DataFrame(selected).to_csv(OUT / "inputs/固定前五名公司.csv", index=False, encoding="utf-8-sig")
    save("freeze.json", {"at": protocol["at"], "protocol_sha256": sha256((OUT / "protocol.json").read_bytes()).hexdigest(),
                         "inputs": items, "company_selection_sha256": sha256((OUT / "inputs/固定前五名公司.csv").read_bytes()).hexdigest()})
    print(json.dumps(selected, ensure_ascii=False))


def fetch(item):
    path = OUT / "sources" / item["name"]
    rp = path.with_suffix(".receipt.json")
    if rp.exists():
        return json.loads(rp.read_text(encoding="utf-8"))
    receipt = {**item, "retrieved_at": datetime.now().astimezone().isoformat(), "first_vintage_authenticated": False}
    try:
        r = requests.get(item["url"], timeout=(12, 45), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update(http_status=r.status_code, final_url=r.url)
        if r.status_code == 200 and r.content.startswith(b"%PDF"):
            path.write_bytes(r.content)
            receipt.update(status="SAVED_PENDING_CONTENT_REVIEW", bytes=len(r.content), sha256=sha256(r.content).hexdigest())
        else:
            receipt["status"] = "FAILED_HTTP_OR_NOT_PDF"
    except Exception as e:
        receipt.update(status="FAILED_REQUEST", error=str(e)[:250])
    rp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def collect():
    docs = json.loads((OUT / "source_catalog.json").read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, docs))
    save("source_receipts.json", receipts)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps([{k: r.get(k) for k in ["name", "status", "http_status", "bytes"]} for r in receipts], ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["freeze", "collect"]:
        raise SystemExit("请指定freeze或collect。")
    freeze() if sys.argv[1] == "freeze" else collect()
