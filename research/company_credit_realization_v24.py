"""固定2021年5月货币病例的公司集合、报告期与价格时序范围。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import sys

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_company_credit_realization_v24"
V5 = ROOT / "reports/research/510300_macro_earnings_pricing_bridge_v5"
V16 = ROOT / "reports/research/510300_constituent_external_bridge_v16"
V23 = ROOT / "reports/research/510300_credit_recency_structure_v23"
NAMES = {"600519.SH": "贵州茅台", "601318.SH": "中国平安", "600036.SH": "招商银行", "000858.SZ": "五粮液", "000333.SZ": "美的集团", "600276.SH": "恒瑞医药"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def freeze():
    if OUT.exists():
        raise RuntimeError("本轮目录已存在，不覆盖。")
    for d in ["inputs", "sources", "results", "figures", "code"]:
        (OUT / d).mkdir(parents=True)
    protocol = {
        "at": now(), "study": "510300_COMPANY_CREDIT_REALIZATION_V24", "previous_turn_classification": "PROGRESS",
        "question": "原2021年5月信用结构分化、订单扩张、利润率改善和降波并存时，权重公司已公布的最新一季报如何兑现；哪些高同比受2020低基数影响，公告距货币观察有多远。",
        "origin": "原2021年5月货币公告，2021年6月10日21:00；原ETF的E0/E1、5/20/60日完全不改。",
        "selection": "按起点之前最近的原权重前五公司，再保留V19既定医药公司恒瑞；本次查看权重发现仍为原六家公司。病例与ETF历史结果已知，不称独立验证。",
        "companies": list(NAMES), "reports": ["2021Q1", "2020Q1"],
        "source_budget": "六家公司各两份一季报，共12份；优先交易所/巨潮或公司原件。已有本地原件可复用。2020Q1的2019比较列用于两年基数核对，不能假定跨版本无修订。",
        "fields": "收入原科目、归母与扣非利润、经营净现金、原文解释。非金融额外核对销售收款、税费和财务子公司；金融核对净利息收入/息差、减值或新业务价值。只记录实际披露项目。",
        "comparability": "各公司单独保留元/百万元和收入科目；不加总六家公司利润为指数EPS，不比较金融与制造业经营现金。负数基期不计算传统增长率，保留金额差和原报告显示值。",
        "base": "2021Q1对2020Q1采用同份原表；2019Q1来自2020Q1原表。先比较两份2020值，一致才计算跨表两年比，否则保留版本差与NOT_COMPUTED。不年化季度收益或利润。",
        "clocks": "原报告正式公告日仅有日期时，保守日末可用。年度末或季度末不是公开日；原件当前取得不证明历史首版不可变。",
        "prices": "全部六家公司：2021Q1公告后首个可观察收盘至原6月10日收盘、原E0/E1二十日的对应收盘参考路径。逐日固定股数、现金分红留存、送转调整股数；不可用公司行动保持缺失。成分收盘路径不冒充ETF开盘收益。",
        "price_gap": "公告后的首个收盘已包含首日反应，不据区间涨跌估计公告因果影响。同期宏观/行业消息未穷尽。没有当时合格市场预期时，盈利预期差保持NO_VIEW。",
        "weights": "原历史参考权重用于固定选择和披露覆盖；不是ETF真实持仓，也不假定窗口内指数不换样。",
        "limits": "全国贷款分项不追踪六家公司借款流水；公司利润增长不证明是某类贷款造成。价格收益与经营更新并列，不能反推唯一原因。",
        "new_models": 0, "new_accounts": 0, "orders_authorized": False, "independent_validation": False, "goal_achieved": False,
        "stop": "不加公司、不改报告期或持有期限来修救原失败；缺失留缺失，不新增行情。",
    }
    save("protocol.json", protocol)
    copies = {
        "monthly.csv": V23 / "results/104个月_信用近期性与完整经营定价背景.csv",
        "market.csv": V16 / "inputs/market.csv",
        "weights.parquet": V5 / "inputs/weights.parquet",
        "previous_completion.json": V23 / "completion.json",
        "authority_snapshot.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }
    receipts = []
    for name, path in copies.items():
        shutil.copy2(path, OUT / "inputs" / name)
        receipts.append({"name": name, "source": str(path.relative_to(ROOT)), "source_sha256": sha(path), "sha256": sha(OUT / "inputs" / name)})
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    r = monthly.loc["2021-05"]
    day = pd.Timestamp(r.observation_date)
    w = pd.read_parquet(OUT / "inputs/weights.parquet")
    w = w[pd.to_datetime(w.trade_date) < day]
    date = pd.to_datetime(w.trade_date).max()
    w = w[pd.to_datetime(w.trade_date).eq(date)].sort_values(["weight", "con_code"], ascending=[False, True])
    selected = w.head(5).copy()
    assert set(selected.con_code) == set(NAMES) - {"600276.SH"}
    selected = pd.concat([selected, w[w.con_code.eq("600276.SH")]])
    selected["name"] = selected.con_code.map(NAMES)
    selected["selection"] = ["原前五权重"] * 5 + ["沿用原医药病例"]
    selected.to_csv(OUT / "inputs/固定六家公司.csv", index=False, encoding="utf-8-sig")
    for name, source, datecol in [
        ("membership.parquet", V5 / "inputs/membership.parquet", "membership_date"),
        ("returns.parquet", V16 / "inputs/returns.parquet", "date"),
    ]:
        data = pd.read_parquet(source)
        dates = pd.to_datetime(data[datecol])
        if name == "membership.parquet":
            subset = data[dates.between(day, pd.Timestamp(r.E1_20_exit_date))].copy()
            assert set(NAMES).issubset(set(subset.loc[pd.to_datetime(subset[datecol]).eq(day), "symbol"]))
        else:
            subset = data[data.symbol.isin(NAMES) & dates.between(pd.Timestamp("2021-01-01"), pd.Timestamp(r.E1_20_exit_date))].copy()
        subset.to_parquet(OUT / "inputs" / name, index=False)
        receipts.append({"name": name, "source": str(source.relative_to(ROOT)), "source_sha256": sha(source), "sha256": sha(OUT / "inputs" / name), "rows": len(subset), "rule": "固定日期与成员截取，不按收益选择"})
    save("input_receipts.json", receipts)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save("freeze.json", {"at": now(), "protocol_sha256": sha(OUT / "protocol.json"), "selected_companies": len(selected),
                         "reference_weight_date": str(date.date()), "reference_weight_percent": float(selected.weight.sum()),
                         "selection_sha256": sha(OUT / "inputs/固定六家公司.csv")})
    print("已固定六家公司、两个原报告期和原价格时序，ETF结果不变。")


def fetch(item):
    dest = OUT / "sources" / item["name"]
    rp = dest.with_suffix(".receipt.json")
    if rp.exists():
        return json.loads(rp.read_text("utf-8"))
    r = {**item, "retrieved_at": now(), "historical_first_vintage_verified": False}
    try:
        response = requests.get(item["url"], timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
        r.update(status_code=response.status_code, final_url=response.url)
        response.raise_for_status()
        assert response.content.startswith(b"%PDF"), "非PDF原件"
        dest.write_bytes(response.content)
        r.update(status="SAVED_PENDING_FACT_REVIEW", bytes=len(response.content), sha256=sha(dest))
    except Exception as e:
        r.update(status="FAILED_REQUEST_OR_CONTENT", error=str(e)[:300])
    rp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return r


def collect():
    catalog = json.loads((OUT / "source_catalog.json").read_text("utf-8"))
    assert len(catalog) == 12
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, catalog))
    save("source_receipts.json", results)
    for r in results:
        print(r["name"], r["status"], r.get("bytes", r.get("error")))


def extract():
    receipts = json.loads((OUT / "source_receipts.json").read_text("utf-8"))
    for r in receipts:
        if not r["status"].startswith("SAVED"):
            continue
        path = OUT / "sources" / r["name"]
        assert sha(path) == r["sha256"]
        doc = pdfium.PdfDocument(path)
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            textpage = page.get_textpage()
            pages.append({"page": i + 1, "text": textpage.get_text_range()})
            textpage.close()
            page.close()
        doc.close()
        path.with_suffix(".pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
        path.with_suffix(".txt").write_text("\n\n".join(f"第{p['page']}页\n{p['text']}" for p in pages), encoding="utf-8")
        print(f"已提取{path.name}，{len(pages)}页。", flush=True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


if __name__ == "__main__":
    {"freeze": freeze, "collect": collect, "extract": extract}[sys.argv[1]]()
