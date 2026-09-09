"""模型封存后只下载永久盲测代码组2023—2026日线。"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import tushare as ts
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_a_share_hash_holdout_training_v1 import download_benchmark, download_daily, sha256  # noqa: E402
from scripts.download_csi300_all_etf_momentum_v1 import credentials  # noqa: E402
from scripts.freeze_a_share_hash_holdout_model_v1 import MODEL_MANIFEST  # noqa: E402
from scripts.freeze_a_share_hash_holdout_protocol_v1 import CONFIG_FILE  # noqa: E402


def main() -> int:
    contract = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    paths, periods = contract["paths"], contract["periods"]
    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if model_manifest.get("state") != "TRAINED_MODEL_FROZEN_BEFORE_HOLDOUT_RETURN_DOWNLOAD":
        raise RuntimeError("训练模型尚未在盲测下载前封存")
    master_path = ROOT / paths["master"]
    if sha256(master_path) != model_manifest["master_sha256"]:
        raise RuntimeError("模型封存后股票母表变化")
    master = pd.read_parquet(master_path)
    holdout = master.loc[
        master["split_group"].eq("HOLDOUT")
        & master["list_date"].le(pd.Timestamp(periods["holdout_evaluation_end"]))
        & (master["delist_date"].isna() | master["delist_date"].ge(pd.Timestamp(periods["holdout_evaluation_start"])))
    ].copy()
    training_status = json.loads((ROOT / paths["training_status"]).read_text(encoding="utf-8"))
    secret, default_endpoint = credentials()
    endpoint = training_status.get("api_endpoint") or default_endpoint
    ts.set_token(secret)
    candidates = [endpoint]
    alternate = "https://tt.xiaodefa.cn" if endpoint == "https://fast.xiaodefa.cn" else "https://fast.xiaodefa.cn"
    if alternate not in candidates:
        candidates.append(alternate)
    api = None
    selected_endpoint = None
    for candidate in candidates:
        candidate_api = ts.pro_api()
        candidate_api._DataApi__http_url = candidate
        try:
            probe = candidate_api.trade_cal(exchange="SSE", start_date="20240102", end_date="20240102", fields="exchange,cal_date,is_open")
            if probe is not None and not probe.empty:
                api = candidate_api
                selected_endpoint = candidate
                break
        except Exception:
            continue
    if api is None:
        raise RuntimeError("模型封存后两个已审核代理节点均不可用")
    directory = ROOT / paths["holdout_checkpoint_directory"]
    results: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    print(f"模型已封存；开始下载永久盲测组{len(holdout)}只股票收益。", flush=True)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_daily, api, code, periods["holdout_warmup_start"], periods["holdout_evaluation_end"], directory): code for code in holdout["ts_code"].astype(str)}
        for completed, future in enumerate(as_completed(futures), start=1):
            code = futures[future]
            try:
                results[code] = future.result()
            except Exception as exc:
                failures[code] = str(exc)
            if completed % 50 == 0 or completed == len(futures):
                print(f"盲测数据 {completed}/{len(futures)}，成功 {len(results)}，失败 {len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"盲测组数据不完整：失败{len(failures)}只，样例={dict(list(failures.items())[:10])}")
    panel = pd.concat(results.values(), ignore_index=True).sort_values(["date", "con_code"]).reset_index(drop=True)
    if panel[["date", "con_code"]].duplicated().any():
        raise ValueError("盲测面板存在重复键")
    abnormal = panel.groupby("con_code")["total_return_close"].pct_change(fill_method=None).abs().gt(0.30)
    age = panel.groupby("con_code").cumcount()
    unexplained = abnormal & age.gt(20)
    if unexplained.any():
        sample = panel.loc[unexplained.to_numpy(), ["date", "con_code", "raw_close", "pre_close", "total_return_close"]].head(20)
        raise ValueError(f"盲测面板上市20日后仍有{int(unexplained.sum())}个单日超过30%的收益：{sample.to_dict('records')}")
    panel_path = ROOT / paths["holdout_panel"]
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    benchmark_path = ROOT / paths["holdout_benchmark"]
    download_benchmark(periods["holdout_warmup_start"], periods["holdout_evaluation_end"], benchmark_path)
    status = {
        "status": "PASS", "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "api_endpoint": selected_endpoint,
        "model_frozen_before_download": True, "holdout_master_count": int(len(holdout)),
        "holdout_downloaded_symbols": int(panel["con_code"].nunique()), "holdout_panel_rows": int(len(panel)),
        "unexplained_returns_over_30pct": int(unexplained.sum()),
        "hashes": {"master": sha256(master_path), "holdout_panel": sha256(panel_path), "holdout_benchmark": sha256(benchmark_path), "model_manifest": sha256(MODEL_MANIFEST)},
    }
    status_path = ROOT / paths["holdout_status"]
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
