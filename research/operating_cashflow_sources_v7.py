"""为固定247家公司取得指定五个报告期的现金流明细，不改变原研究成员。"""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import shutil

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_operating_cashflow_transmission_v7"
V5 = ROOT / "reports/research/510300_macro_earnings_pricing_bridge_v5"


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fetch(number, codes, dates):
    dest = OUT / "sources" / f"fixed_cohort_batch_{number:02d}.json"
    receipt_path = dest.with_suffix(".receipt.json")
    quoted_codes = ",".join('"' + code + '"' for code in codes)
    quoted_dates = ",".join("'" + date + "'" for date in dates)
    params = {"type": "RPT_F10_FINANCE_GCASHFLOW", "sty": "ALL", "filter": f"(SECUCODE in ({quoted_codes}))(REPORT_DATE in ({quoted_dates}))", "p": "1", "ps": "500", "sr": "1,1", "st": "SECURITY_CODE,REPORT_DATE", "source": "HSF10", "client": "PC"}
    if dest.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert sha(dest) == receipt["sha256"]
        document = json.loads(dest.read_bytes())
    else:
        r = requests.get("https://datacenter.eastmoney.com/securities/api/data/get", params=params, timeout=(10,40))
        r.raise_for_status()
        dest.write_bytes(r.content)
        receipt = {"at": now(), "url": r.url, "source": "公开二次汇编财务数据，当前版本", "code_count": len(codes), "codes": codes, "report_dates": dates, "http_status": r.status_code, "sha256": sha(dest)}
        save(receipt_path, receipt)
        document = r.json()
    assert document.get("success") is True, document.get("message")
    result = document["result"]
    assert result["pages"] == 1
    rows = result["data"]
    assert len(rows) == result["count"]
    assert {x["SECUCODE"] for x in rows}.issubset(set(codes))
    for row in rows:
        row["source_file"] = dest.name
        row["source_sha256"] = receipt["sha256"]
        row["retrieved_at"] = receipt["at"]
    return number, rows, receipt


def run():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert sha(OUT / "protocol.json") == json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))["sha256"]
    original = pd.read_parquet(V5 / "results/全部观察日_公司财务重建与版本标记.parquet")
    cohort = original[original.observation_date.eq(pd.Timestamp("2025-09-12")) & original.cash_matched_valid].copy().sort_values("stock_code")
    assert len(cohort) == cohort.stock_code.nunique() == 247
    assert cohort.cash_report_end.eq(pd.Timestamp("2025-06-30")).all()
    cohort.to_parquet(OUT / "inputs/fixed_247_company_cohort.parquet", index=False)
    shutil.copy2(V5 / "inputs/cashflow.parquet", OUT / "inputs/parent_cashflow.parquet")
    shutil.copy2(V5 / "inputs/income.parquet", OUT / "inputs/parent_income.parquet")
    save(OUT / "inputs/source_identity.json", {"at": now(), "parent_company_file": str((V5 / "results/全部观察日_公司财务重建与版本标记.parquet").relative_to(ROOT)), "parent_company_sha256": sha(V5 / "results/全部观察日_公司财务重建与版本标记.parquet"), "cohort_sha256": sha(OUT / "inputs/fixed_247_company_cohort.parquet"), "parent_cashflow_sha256": sha(OUT / "inputs/parent_cashflow.parquet"), "parent_income_sha256": sha(OUT / "inputs/parent_income.parquet")})
    codes, dates = cohort.stock_code.tolist(), protocol["periods"]
    tasks = [(i // 50 + 1, codes[i:i+50], dates) for i in range(0, len(codes), 50)]
    # 先确认代码和报告期过滤语法，再并行读取互不依赖的其余批次。
    first = fetch(*tasks[0])
    outputs = [first]
    print(f"固定公司批次 {first[0]} 已保存 {len(first[1])} 行。", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(fetch, *task) for task in tasks[1:]]
        for future in as_completed(futures):
            result = future.result()
            outputs.append(result)
            print(f"固定公司批次 {result[0]} 已保存 {len(result[1])} 行。", flush=True)
    raw = pd.DataFrame([row for _, rows, _ in sorted(outputs) for row in rows])
    raw["report_end"] = pd.to_datetime(raw.REPORT_DATE).dt.normalize()
    assert not raw.duplicated(["SECUCODE", "report_end"]).any()
    raw.to_parquet(OUT / "results/固定247家公司_五期现金流完整原字段.parquet", index=False)
    missing = [(code, date) for code in codes for date in dates if not ((raw.SECUCODE == code) & (raw.report_end == pd.Timestamp(date))).any()]
    save(OUT / "source_receipt.json", {"at": now(), "company_count": len(codes), "report_periods": dates, "expected_rows": len(codes) * len(dates), "saved_rows": len(raw), "missing_company_periods": missing, "sources": [receipt for _, _, receipt in sorted(outputs)], "source_vintage": "CURRENT_SECONDARY_VALUES_NOT_STRICT_PIT", "historical_first_version_authenticated": False})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(f"明细来源完成：{len(raw)} 行，缺失公司报告期 {len(missing)} 个。")


if __name__ == "__main__":
    run()
