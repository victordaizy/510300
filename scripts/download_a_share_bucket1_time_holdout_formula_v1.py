"""冻结四因子公式后下载桶1未见的2023—2026日线。"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import tushare as ts


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_a_share_hash_holdout_training_v1 import download_benchmark, download_daily  # noqa: E402
from scripts.download_csi300_all_etf_momentum_v1 import credentials  # noqa: E402
from scripts.freeze_a_share_bucket1_time_holdout_formula_v1 import sha256, tree_sha256, verify_protocol  # noqa: E402


def main() -> int:
    contract, protocol = verify_protocol()
    paths, periods = contract["paths"], contract["periods"]
    master_path = ROOT / contract["inputs"]["master"]
    master = pd.read_parquet(master_path)
    holdout = master.loc[
        master["split_bucket"].eq(int(contract["split"]["future_holdout_bucket"]))
        & master["list_date"].le(pd.Timestamp(periods["evaluation_end"]))
        & (master["delist_date"].isna() | master["delist_date"].ge(pd.Timestamp(periods["evaluation_start"])))
    ].copy()
    secret, default_endpoint = credentials()
    ts.set_token(secret)
    candidates = [default_endpoint, "https://tt.xiaodefa.cn", "https://fast.xiaodefa.cn"]
    api = None
    selected_endpoint = None
    for endpoint in dict.fromkeys(candidates):
        candidate = ts.pro_api()
        candidate._DataApi__http_url = endpoint
        try:
            probe = candidate.trade_cal(exchange="SSE", start_date="20240102", end_date="20240102", fields="exchange,cal_date,is_open")
            if probe is not None and not probe.empty:
                api = candidate
                selected_endpoint = endpoint
                break
        except Exception:
            continue
    if api is None:
        raise RuntimeError("两个已审核代理节点均不可用")
    checkpoint_dir = ROOT / paths["checkpoint_directory"]
    results = {}
    failures = {}
    print(f"公式已冻结；开始下载桶1未来盲测{len(holdout)}只股票。", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_daily, api, code, periods["warmup_start"], periods["evaluation_end"], checkpoint_dir): code
            for code in holdout["ts_code"].astype(str)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                results[code] = future.result()
            except Exception as exc:
                failures[code] = str(exc)
            if completed % 50 == 0 or completed == len(futures):
                print(f"桶1数据 {completed}/{len(futures)}，成功 {len(results)}，失败 {len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"桶1未来数据不完整：失败{len(failures)}只，样例={dict(list(failures.items())[:10])}")
    panel = pd.concat(results.values(), ignore_index=True).sort_values(["con_code", "date"]).reset_index(drop=True)
    if panel[["date", "con_code"]].duplicated().any():
        raise ValueError("桶1未来面板存在重复证券日期")
    prices = panel[["raw_close", "pre_close", "total_return_close"]].apply(pd.to_numeric, errors="coerce")
    if prices.isna().any().any() or prices.le(0).any().any():
        raise ValueError("桶1未来原始价格、前收或总收益价格无效")
    observed = panel.groupby("con_code")["total_return_close"].pct_change(fill_method=None)
    expected = panel["raw_close"] / panel["pre_close"] - 1.0
    age = panel.groupby("con_code").cumcount()
    comparable = age.gt(0) & observed.notna() & expected.notna()
    difference = (observed - expected).abs()
    unexplained = comparable & difference.gt(1e-8)
    if unexplained.any():
        sample = panel.loc[unexplained, ["date", "con_code", "raw_close", "pre_close", "total_return_close"]].head(20)
        raise ValueError(f"存在{int(unexplained.sum())}个无法由原始收盘/前收解释的收益：{sample.to_dict('records')}")
    extreme = comparable & observed.abs().gt(0.30)
    panel_path = ROOT / paths["holdout_panel"]
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = panel_path.with_suffix(".parquet.tmp")
    panel.to_parquet(temporary, index=False)
    temporary.replace(panel_path)
    benchmark_path = ROOT / paths["benchmark"]
    download_benchmark(periods["warmup_start"], periods["evaluation_end"], benchmark_path)
    status = {
        "status": "PASS",
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "api_endpoint": selected_endpoint,
        "protocol_frozen_before_download": True,
        "protocol_manifest_sha256": sha256(ROOT / paths["protocol_manifest"]),
        "holdout_bucket": 1,
        "holdout_master_count": int(len(holdout)),
        "downloaded_symbols": int(panel["con_code"].nunique()),
        "panel_rows": int(len(panel)),
        "extreme_returns_over_30pct": int(extreme.sum()),
        "unexplained_returns": int(unexplained.sum()),
        "maximum_return_identity_difference": float(difference.loc[comparable].max()),
        "hashes": {
            "master": sha256(master_path),
            "panel": sha256(panel_path),
            "benchmark": sha256(benchmark_path),
            "checkpoint_tree": tree_sha256(checkpoint_dir),
        },
    }
    status_path = ROOT / paths["data_status"]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
