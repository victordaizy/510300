"""股东公告覆盖差的固定可行性检验；严格区别普通股东与内部人。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import requests

from research.mechanism_odds_open_contract_v1 import now, read, save, digest
from research.constituent_dividend_calendar_v1 import weekly_origins, label_origins

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_shareholder_disclosure_breadth_v1"
MEMBERS = ROOT / "reports/research/510300_factor96_internal_reclaim_v1_0_1/inputs/membership.parquet"
MARKET = ROOT / "reports/research/510300_mechanism_odds_open_contract_v1/inputs/market.parquet"
PERIODS = {"FULL": ("2017-01-01", "2025-12-31"), "EARLY": ("2017-01-01", "2020-12-31"),
           "LATE": ("2021-01-01", "2025-12-31")}
ENDPOINT = "https://datacenter-web.eastmoney.com/api/data/v1/get"
HOLD = 20


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "at": now(), "study_id": "510300_SHAREHOLDER_DISCLOSURE_BREADTH_V1",
        "prior_goal_turn": "PROGRESS_DIVIDEND_CALENDAR_GROSS_PASS_FULL_ACCOUNT_REJECTED",
        "objective": "完整账户成本后夏普至少1.2、CAGR至少10%，原风险合同及资产范围不变。",
        "hypothesis": "实际股东增持的公告覆盖超过减持覆盖，可能表明所有者对价格的认同更广，随后仍有重定价。",
        "counterexamples": ["普通机构调仓不代表内部信息", "公告滞后且已买入不是未来买盘", "增持可能持续买早，减持可能为个人资金需求", "监管与披露习惯变化影响覆盖差"],
        "novelty": "不同于已失败的公司回购/发行数量代理；M03旁支的全类股东方向覆盖代理，原内部人金额/自由流通市值定义仍NOT_RUN。",
        "sources": ["https://data.eastmoney.com/executive/gdzjc.html", "https://raw.githubusercontent.com/akfamily/akshare/main/akshare/stock_feature/stock_gdzjc_em.py"],
        "source_contract": {"reportName": "RPT_SHARE_HOLDER_INCREASE", "notice_min": "2016-11-01", "notice_max": "2025-12-31",
            "universe": "既有官方重建历史CSI300成员所有证券；每日仅按该日成员筛选。",
            "collection": "证券代码排序后每50个一批，pageSize=5000，最多40次公开请求；无重试，无凭据。",
            "known_at": "供应商NOTICE_DATE日末23:59:59（北京时间），当日16时不可用；EITIME仅保存诊断，不宣称历史first_seen。",
            "event": "正数CHANGE_NUM且DIRECTION为增持/减持、END_DATE<=NOTICE_DATE、MARKET含二级市场或大宗交易；排除单纯协议/非交易过户、激励、增发等。",
            "dedup": "同证券、同股东、同开始/结束/公告日、同方向及变动数量去重；方向集合内同发行人计一次，不累加累计金额。",
            "missing": "任一代码批次未完整取得即源门失败；无法解释的记录保留为排除/未知，不能默认为零事实。",
            "historical_first_vintage_verified": False},
        "signal": {"decision_at": "每ISO周第一个A股交易日16:00；输入每日可更新。",
            "window": "截至决策时已知、首次按NOTICE_DATE日末进入的此前20个A股交易日公告；在(第i-20日16:00,第i日16:00]内。",
            "formula": "当日300成员中有增持公告的公司数减有减持公告的公司数，再除300；两边均出现的公司互相抵消。",
            "primary": "覆盖差>0；无价格、估值、阈值排名、方向反转或期限搜索。",
            "model_fits": 0, "training_window": "无拟合模型，无额外训练参数。"},
        "execution": {"entry": "决策后下一个A股开盘", "hold_open_intervals": 20,
            "overlap": "主检验只接受空仓时信号，不加仓、不重置期限；持有期内的信号保留为被忽略。",
            "unfilled": "方向涨停不买，收益及成本为零；跌停卖出顺延；末端未成熟不填零。",
            "dividend": "复用原始ETF价格及现金分红标签，买入除息日不享权，卖出除息日仍享权。"},
        "evaluation": {"periods": PERIODS, "primary": "无重叠入场机会的20开盘毛收益（仍不是账户夏普）。",
            "controls": ["同年同月全部可评估周原点均值，按主机会月份数量加权", "全周无重叠机械持有", "全部周信号重叠事件仅诊断"],
            "screen_gates": ["全部代码批次/分页完整", "主可评估机会至少20次且前后段各至少5次", "主毛均值>0.28%", "主毛均值>同年同月周对照", "前后两段主均值均>0"],
            "uncertainty": "按用户减少繁琐检验的新指令，本轮仅做固定规则与前后段、收益集中度筛查；区块重采样留待有用候选。",
            "concentration": "最大机会对毛收益算术和贡献及去除后均值，仅诊断；不删交易救策略。"},
        "stop": "初筛任一门失败即固定版本停止，不改方向/窗口/阈值；通过才准许同规则完整账户，仍不能达成目标声明。",
        "orders_authorized": False, "new_returns_computed_before_registration": False,
    }
    save(OUT / "design_registration.json", protocol)
    (OUT / "inputs").mkdir(exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)
    shutil.copy2(MEMBERS, OUT / "inputs/membership.parquet")
    shutil.copy2(MARKET, OUT / "inputs/market.parquet")
    shutil.copy2(ROOT / "config/510300_existing_data_training_mandate_v1.json", OUT / "inputs/authority_snapshot.json")
    print("固定规则已登记；先取公告来源，不读取未来收益。", flush=True)


def fetch_batch(batch_id, codes):
    params = {"reportName": "RPT_SHARE_HOLDER_INCREASE", "columns": "ALL", "pageSize": 5000,
        "sortTypes": "1,1,1,1", "sortColumns": "NOTICE_DATE,SECURITY_CODE,EITIME,HOLDER_NAME",
        "filter": '(NOTICE_DATE>=\'2016-11-01\')(NOTICE_DATE<=\'2025-12-31\')(SECURITY_CODE in (' +
                  ",".join('"' + c + '"' for c in codes) + '))', "source": "WEB", "client": "WEB"}
    rows, receipts, count, pages = [], [], None, None
    for page in range(1, 41):
        params["pageNumber"] = page
        path = OUT / f"raw/batch_{batch_id:02d}_page_{page:02d}.json"
        if path.exists():
            receipt = read(path.with_suffix(".receipt.json"))
            assert digest(path) == receipt["sha256"]
            value = read(path)
        else:
            assert len(list((OUT / "raw").glob("batch_*_page_*.receipt.json"))) < 40, "已到总请求边界。"
            request_at = now()
            response = requests.get(ENDPOINT, params=params, timeout=30)
            path.write_bytes(response.content)
            receipt = {"batch": batch_id, "page": page, "codes": codes, "request_at": request_at,
                       "received_at": now(), "http_status": response.status_code, "path": str(path.relative_to(ROOT)),
                       "sha256": digest(path), "url": response.url}
            save(path.with_suffix(".receipt.json"), receipt)
            response.raise_for_status()
            value = response.json()
        assert value.get("success"), str(value.get("message"))
        result = value["result"]
        if page == 1:
            count, pages = int(result["count"]), int(result["pages"])
            assert pages <= 40, "该批超过注册总请求边界。"
        assert int(result["count"]) == count and int(result["pages"]) == pages, "分页总数发生变化。"
        new_rows = result.get("data") or []
        assert all(r["SECURITY_CODE"] in codes for r in new_rows), "服务端代码筛选未生效。"
        rows.extend(new_rows)
        receipts.append(receipt)
        if page >= pages:
            break
    assert len(rows) == count, "分页行数不完整。"
    return {"batch": batch_id, "codes": codes, "count": count, "pages": pages, "receipts": receipts}, rows


def collect():
    assert not (OUT / "source_collection.json").exists()
    members = pd.read_parquet(OUT / "inputs/membership.parquet")
    codes = sorted({s[:6] for s in members.symbol})
    batches = [codes[i:i + 50] for i in range(0, len(codes), 50)]
    summaries, frames, failures = [], [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch_batch, i + 1, c): i + 1 for i, c in enumerate(batches)}
        for future in as_completed(futures):
            try:
                summary, rows = future.result()
                summaries.append(summary)
                frames.extend(rows)
                print(f"公告来源批次{summary['batch']}/{len(batches)}完整：{summary['count']}行。", flush=True)
            except Exception as exc:
                failures.append({"batch": futures[future], "error": f"{type(exc).__name__}: {exc}"})
                print(f"来源批次{futures[future]}失败，不重试。", flush=True)
    save(OUT / "source_collection.json", {"at": now(), "status": "COMPLETE" if not failures else "SOURCE_GATE_NOT_PASSED",
        "expected_codes": len(codes), "expected_batches": len(batches), "completed_batches": len(summaries),
        "source_rows": len(frames), "summaries": sorted(summaries, key=lambda x: x["batch"]), "failures": failures,
        "requests": sum(s["pages"] for s in summaries) + len(failures), "historical_first_vintage_verified": False})
    pd.DataFrame(frames).to_parquet(OUT / "inputs/provider_records.parquet", index=False)
    assert not failures, "来源不完整，禁止收益检验。"


def finish_pagination():
    """修正服务端实际500行分页的处理；不重复请求已有页面。"""
    initial = read(OUT / "source_collection.json")
    assert initial["failures"] == [{"batch": 4, "error": "AssertionError: 该批超过3页，超过注册请求边界。"}]
    assert not (OUT / "freeze.json").exists()
    first = read(OUT / "raw/batch_04_page_01.receipt.json")
    summary, missing_batch_rows = fetch_batch(4, first["codes"])
    raw = pd.read_parquet(OUT / "inputs/provider_records.parquet")
    raw.to_parquet(OUT / "inputs/provider_records_initial_partial.parquet", index=False)
    full = pd.concat([raw, pd.DataFrame(missing_batch_rows)], ignore_index=True)
    full.to_parquet(OUT / "inputs/provider_records.parquet", index=False)
    result = {**initial, "at": now(), "status": "COMPLETE", "completed_batches": 14, "source_rows": len(full),
              "summaries": sorted(initial["summaries"] + [summary], key=lambda x: x["batch"]), "failures": [],
              "requests": len(list((OUT / "raw").glob("batch_*_page_*.receipt.json"))),
              "source_only_correction": "pageSize=5000请求被服务端按500行分页；删除代码中未由协议要求的每批3页限制，补取3个未请求页面；总31次低于原40次上限。",
              "new_return_reads_before_correction": 0}
    save(OUT / "source_collection_completion.json", result)
    print(f"来源完整：14批、{len(full)}行、{result['requests']}次请求。", flush=True)


def collection_receipt():
    path = OUT / "source_collection_completion.json"
    return read(path if path.exists() else OUT / "source_collection.json")


def normalize(raw):
    frame = raw.copy()
    frame["notice_date"] = pd.to_datetime(frame.NOTICE_DATE, errors="coerce")
    frame["start_date"] = pd.to_datetime(frame.START_DATE, errors="coerce")
    frame["end_date"] = pd.to_datetime(frame.END_DATE, errors="coerce")
    frame["quantity"] = pd.to_numeric(frame.CHANGE_NUM, errors="coerce")
    frame["symbol"] = frame.SECURITY_CODE.astype(str) + np.where(frame.SECURITY_CODE.str.startswith("6"), ".SH", ".SZ")
    market = frame.MARKET.fillna("").astype(str)
    frame["eligible_market"] = market.str.contains("二级市场|大宗交易", regex=True)
    frame["valid_clock"] = frame.notice_date.notna() & frame.end_date.notna() & frame.end_date.le(frame.notice_date)
    frame["valid_direction"] = frame.DIRECTION.isin(["增持", "减持"]) & frame.quantity.gt(0)
    frame["eligible"] = frame.eligible_market & frame.valid_clock & frame.valid_direction
    frame["known_at"] = frame.notice_date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    keys = ["SECURITY_CODE", "HOLDER_NAME", "NOTICE_DATE", "START_DATE", "END_DATE", "DIRECTION", "CHANGE_NUM"]
    frame["duplicate"] = frame.duplicated(keys, keep="first")
    good = frame.loc[frame.eligible & ~frame.duplicate].copy()
    good["known_at"] = good.notice_date.dt.tz_localize("Asia/Shanghai") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return good, frame


def build_daily(calendar, membership, events, all_queries_complete=True, uncertain=None):
    calendar = pd.DatetimeIndex(calendar)
    members = membership.copy()
    members["membership_date"] = pd.to_datetime(members.membership_date)
    groups = {date: set(g.symbol) for date, g in members.groupby("membership_date")}
    known = pd.DatetimeIndex(events.known_at)
    rows = []
    for i, day in enumerate(calendar):
        if day < pd.Timestamp("2017-01-01") or day > pd.Timestamp("2025-12-31"):
            continue
        current = groups.get(day, set())
        row = {"date": day, "idx": i, "members": len(current), "status": "NO_VIEW_SOURCE",
               "selected": False, "buy_count": np.nan, "sell_count": np.nan, "breadth": np.nan,
               "buy_symbols": "", "sell_symbols": ""}
        if len(current) == 300 and all_queries_complete and i >= 20:
            cutoff = day.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
            start = calendar[i - 20].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
            if uncertain is not None and not uncertain.empty:
                unknown_now = uncertain.symbol.isin(current) & (
                    uncertain.known_at.isna() | (uncertain.known_at.gt(start) & uncertain.known_at.le(cutoff)))
                if unknown_now.any():
                    row["status"] = "NO_VIEW_CONFLICTING_SOURCE_DATES"
                    rows.append(row)
                    continue
            recent = events.loc[(known > start) & (known <= cutoff) & events.symbol.isin(current)]
            buys = set(recent.loc[recent.DIRECTION.eq("增持"), "symbol"])
            sells = set(recent.loc[recent.DIRECTION.eq("减持"), "symbol"])
            row.update(status="READY", buy_count=len(buys), sell_count=len(sells), breadth=(len(buys) - len(sells)) / 300.,
                       selected=len(buys) > len(sells), buy_symbols=";".join(sorted(buys)), sell_symbols=";".join(sorted(sells)),
                       latest_known_at=None if recent.empty else str(recent.known_at.max()), recent_rows=len(recent))
        rows.append(row)
    return pd.DataFrame(rows)


def nonoverlap(labels, only_selected=True):
    chosen = np.zeros(len(labels), dtype=bool)
    free_idx = -1
    for i, row in labels.iterrows():
        if row.information_status != "READY" or (only_selected and not row.selected) or row.idx < free_idx:
            continue
        chosen[i] = True
        if row.status == "MATURE":
            free_idx = int(row.exit_idx)
        elif row.status.startswith("CENSORED"):
            free_idx = 10 ** 12
    return chosen


def statistics(frame):
    available = frame.gross_return.notna()
    sub = frame.loc[available]
    values = sub.gross_return.to_numpy(float)
    return {"opportunities": len(frame), "evaluated": len(sub), "filled": int(sub.status.eq("MATURE").sum()),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "positive_fraction": float((values > 0).mean()) if len(values) else None,
            "stress_cost_proxy_mean": float(sub.stress_proportional_proxy_return.mean()) if len(sub) else None,
            "min": float(values.min()) if len(values) else None, "max": float(values.max()) if len(values) else None}


def matched_comparison(labels, mask):
    work = labels.loc[labels.gross_return.notna()].copy()
    work["selected_for_comparison"] = mask[labels.gross_return.notna().to_numpy()]
    means = work.groupby(work.date.dt.to_period("M")).gross_return.transform("mean")
    work["monthly_control"] = means
    sub = work.loc[work.selected_for_comparison]
    return {"count": len(sub), "primary_mean": float(sub.gross_return.mean()) if len(sub) else None,
            "matched_month_control_mean": float(sub.monthly_control.mean()) if len(sub) else None,
            "increment": float((sub.gross_return - sub.monthly_control).mean()) if len(sub) else None}


def freeze():
    assert collection_receipt()["status"] == "COMPLETE"
    good, annotated = normalize(pd.read_parquet(OUT / "inputs/provider_records.parquet"))
    good.to_parquet(OUT / "inputs/events.parquet", index=False)
    uncertain = annotated.loc[annotated.eligible_market & (~annotated.valid_clock | ~annotated.valid_direction)]
    uncertain.to_parquet(OUT / "inputs/uncertain_events.parquet", index=False)
    annotated.to_parquet(OUT / "inputs/annotated_source.parquet", index=False)
    by_year = annotated.groupby(annotated.notice_date.dt.year).agg(source_rows=("SECURITY_CODE", "size"), eligible=("eligible", "sum"))
    save(OUT / "source_fields.json", {"at": now(), "rows": len(annotated), "eligible_events": len(good),
        "excluded_nonmarket": int((~annotated.eligible_market).sum()), "invalid_clock": int((~annotated.valid_clock).sum()),
        "invalid_direction_quantity": int((~annotated.valid_direction).sum()), "duplicate_rows": int(annotated.duplicate.sum()),
        "market_values": annotated.MARKET.fillna("MISSING").value_counts().to_dict(),
        "by_year": by_year.reset_index().to_dict("records"), "historical_first_vintage_verified": False})
    paths = [Path(__file__), ROOT / "research/constituent_dividend_calendar_v1.py",
             ROOT / "research/lpr_joint_response_v1.py", ROOT / "research/mechanism_odds_open_contract_v1.py",
             ROOT / "tests/test_shareholder_disclosure_breadth_v1.py", OUT / "design_registration.json",
             OUT / "source_collection.json", OUT / "source_fields.json", *sorted((OUT / "inputs").glob("*"))]
    if (OUT / "source_collection_completion.json").exists():
        paths.append(OUT / "source_collection_completion.json")
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths],
                              "new_return_tests_before_freeze": 0})
    print(f"代码、规则和来源已冻结：{len(good)}条合格公告事件。", flush=True)


def run():
    assert not (OUT / "result.json").exists(), "已存在结果，不重复检验。"
    frozen = read(OUT / "freeze.json")
    assert all(digest(ROOT / x["path"]) == x["sha256"] for x in frozen["files"])
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market = market.loc[market.date.le("2025-12-31")].reset_index(drop=True)
    members = pd.read_parquet(OUT / "inputs/membership.parquet")
    events = pd.read_parquet(OUT / "inputs/events.parquet")
    uncertain = pd.read_parquet(OUT / "inputs/uncertain_events.parquet")
    daily = build_daily(market.date, members, events, uncertain=uncertain)
    labels = label_origins(weekly_origins(daily), market)
    labels["primary_nonoverlap"] = nonoverlap(labels, True)
    labels["all_nonoverlap"] = nonoverlap(labels, False)
    daily.to_parquet(OUT / "daily_features.parquet", index=False)
    labels.to_parquet(OUT / "origins.parquet", index=False)
    labels.to_csv(OUT / "周原点与执行标签.csv", index=False, encoding="utf-8-sig")
    rows = []
    for name, period in PERIODS.items():
        part = labels.loc[labels.date.between(*period)].copy()
        for group, mask in [("PRIMARY_NONOVERLAP", part.primary_nonoverlap), ("ALL_NONOVERLAP", part.all_nonoverlap),
                            ("PRIMARY_ALL_WEEKLY_DIAGNOSTIC", part.selected), ("ALL_WEEKLY", part.information_status.eq("READY"))]:
            rows.append({"period": name, "group": group, **statistics(part.loc[mask]),
                         **matched_comparison(part, mask.to_numpy(bool))})
    save(OUT / "summary.json", rows)
    primary = {r["period"]: r for r in rows if r["group"] == "PRIMARY_NONOVERLAP"}
    full = primary["FULL"]
    gate = {"complete_provider_queries": collection_receipt()["status"] == "COMPLETE",
            "enough_nonoverlap": full["evaluated"] >= 20 and all(primary[p]["evaluated"] >= 5 for p in ["EARLY", "LATE"]),
            "mean_above_cost_proxy": full["mean"] is not None and full["mean"] > .0028,
            "increment_above_matched_month": full["increment"] is not None and full["increment"] > 0,
            "both_subperiods_positive": all(primary[p]["mean"] is not None and primary[p]["mean"] > 0 for p in ["EARLY", "LATE"])}
    selected = labels.loc[labels.primary_nonoverlap & labels.gross_return.notna()]
    top = selected.sort_values("gross_return", ascending=False).iloc[0]
    others = selected.drop(top.name)
    total = float(selected.gross_return.sum())
    concentration = {"largest_date": str(top.date.date()), "largest_gross_return": float(top.gross_return),
                     "largest_share_of_arithmetic_sum": float(top.gross_return / total) if total > 0 else None,
                     "without_largest_count": len(others), "without_largest_mean": float(others.gross_return.mean())}
    save(OUT / "concentration.json", concentration)
    ci = {"status": "NOT_RUN_QUICK_FEASIBILITY_FIRST", "replicates": 0,
          "boundary": "本轮只筛规律可行性，不声称统计确定性；历史已反复研究，前后切分不构成独立验证。"}
    save(OUT / "uncertainty.json", ci)
    # 真正截断未来公告后，过去输入必须完全一致。
    cutoff = pd.Timestamp("2021-12-31")
    past_events = events.loc[events.notice_date.le(cutoff)].copy()
    past_unknown = uncertain.loc[uncertain.notice_date.le(cutoff) | uncertain.notice_date.isna()]
    past = build_daily(market.loc[market.date.le(cutoff), "date"], members, past_events, uncertain=past_unknown)
    pd.testing.assert_frame_equal(daily.loc[daily.date.le(cutoff)].reset_index(drop=True), past.reset_index(drop=True))
    saved = pd.read_parquet(OUT / "origins.parquet")
    errors = []
    for row in saved.loc[saved.status.eq("MATURE")].itertuples():
        a, b = int(row.entry_idx), int(row.exit_idx)
        value = (float(market.open.iloc[b]) + math.fsum(market.dividend.iloc[a + 1:b + 1])) / float(market.open.iloc[a]) - 1
        errors.append(abs(value - row.gross_return))
    assert max(errors, default=0) < 1e-12
    save(OUT / "verification_receipt.json", {"at": now(), "status": "PASS_FIXED_GROSS_IMPLEMENTATION_CHECKS",
        "freeze_hashes": len(frozen["files"]), "future_prefix_checks": 1, "saved_labels_recomputed": len(errors),
        "max_label_error": max(errors, default=0), "new_model_fits": 0, "new_accounts": 0,
        "independent_forward_observations": 0, "source_first_vintage_proven": False})
    result = {"at": now(), "study_id": "510300_SHAREHOLDER_DISCLOSURE_BREADTH_V1",
        "status": "PASS_GROSS_SCREEN_ACCOUNT_REQUIRED" if all(gate.values()) else "REJECTED_FIXED_GROSS_SCREEN_NO_PARAMETER_RESCUE",
        "gates": gate, "weekly_origins": len(labels), "status_counts": labels.status.value_counts().to_dict(),
        "primary": primary, "concentration": concentration, "uncertainty": ci,
        "account_status": "NOT_RUN_GROSS_SCREEN_PASSED" if all(gate.values()) else "NOT_RUN_GROSS_SCREEN_FAILED",
        "net_sharpe": None, "net_cagr": None, "max_drawdown": None, "new_accounts": 0,
        "new_fixed_proxy_questions": 1, "strict_M03_run": False, "goal_achieved": False,
        "historical_first_vintage_verified": False, "orders_authorized": False}
    save(OUT / "result.json", result)
    print(f"固定无重叠主机会{full['evaluated']}次，毛均值{full['mean']:.4%}，同月增量{full['increment']:.4%}；{result['status']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="股东公告覆盖差固定研究")
    parser.add_argument("phase", choices=["register", "collect", "finish-pagination", "freeze", "run"])
    args = parser.parse_args()
    {"register": register, "collect": collect, "finish-pagination": finish_pagination, "freeze": freeze, "run": run}[args.phase]()


if __name__ == "__main__":
    main()
