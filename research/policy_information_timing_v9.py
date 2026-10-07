"""复用固定LPR母集，连接当时货币背景、第一段反应与原有后续标签。"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_policy_information_timing_v9"
INPUTS = {
    "survey84.csv": "reports/research/510300_lpr_expectation_source_extension_v2/全部84月事前预期识别.csv",
    "parent56.csv": "reports/research/510300_lpr_expectation_exploratory_training_v1/results/事件特征与20日标签.csv",
    "parent_summary.json": "reports/research/510300_lpr_expectation_exploratory_training_v1/summary.json",
    "market.csv": "reports/research/510300_macro_transmission_context_v4/inputs/market.csv",
    "volatility.csv": "reports/research/510300_volatility_money_decomposition_v3/results/每日波动精确分解.csv",
    "monthly_context.csv": "reports/research/510300_macro_transmission_context_v4/results/104个月_当时可见多层证据_不含未来标签.csv",
    "orders.parquet": "reports/research/510300_macro_transmission_context_v4/inputs/pmi_orders.parquet",
    "orders_completion.json": "reports/research/510300_macro_transmission_context_v4/results/新订单既有原文补全.json",
    "credit.csv": "reports/research/510300_macro_transmission_context_v4/results/104个月_信贷分项与同区间比较.csv",
    "policy_nodes.csv": "reports/research/510300_information_change_transmission_v6/inputs/policy_nodes.csv",
    "operation_rate.parquet": "reports/research/510300_macro_transmission_context_v4/inputs/policy_rate.parquet",
}


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(data, name):
    data.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig")


def freeze():
    for folder in ["inputs", "results", "sources", "code", "figures"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    if (OUT / "freeze.json").exists():
        return
    protocol = {"at": now(), "study_id": "510300_POLICY_INFORMATION_TIMING_V9",
        "previous_turn_classification": "PROGRESS：V8完成20个月货币来源与35个期间金额分解。",
        "question": "调查与公告之间的信息变化，及公告后的第一段反应，是否改变政策宽松与510300剩余收益的含义？",
        "universe": "复用84个月LPR母集和56个月已准入调查；28个月缺失不补零。全部56行旧收益标签原样保留。",
        "cases": "按已保存调查选出至少一个期限的bp偏差区间严格同侧，或多数预计降息却不降的全部月份；在读取本轮股票结果前已核对共7月。不是按收益选案例。",
        "case_months": ["2022-05", "2022-08", "2023-06", "2023-08", "2024-02", "2024-07", "2024-09"],
        "first_reaction": "沿用父级公告上界之后第一个开盘严格更晚的完整交易日。保存前收盘到当日开盘、日内和分红分项；单独标明此起点不是盘中精确政策效果。",
        "residual_return": "主要结果原样复用父级：第一完整反应日收盘复核后，下一交易日开盘起20个交易日固定股数含分红毛收益。另展示同一终点下首反应日开盘起的金额路径，不择优换起点。",
        "macro_context": "每个事件按自己的公告时点连接此前已公布的M1/M2、同区间信贷和新订单；价格与波动在公告前和首次复核日分别保存。",
        "intervening_information": "只对既有7病例作定向时序核对。查明新信息不等于取得更新后的数值调查；其余未查明者为未核对，不能称没有其他信息。",
        "policy_catalog": "复用既有24节点不完整目录；后续节点只解释路径，不进入起点或删掉失败。",
        "already_seen": "全部旧行情与旧模型结果此前已见；本轮不构成独立验证。",
        "existing_results": "旧LPR训练和账户终态、阈值、样本及标签不改，0个新模型、0个新账户。",
        "historical_first_vintage_authenticated": False, "goal_achieved": False}
    save(OUT / "protocol.json", protocol)
    save(ROOT / "config/510300_policy_information_timing_v9.json", protocol)
    frozen = {}
    for name, relative in INPUTS.items():
        source = ROOT / relative
        shutil.copy2(source, OUT / "inputs" / name)
        frozen[relative] = digest(source)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "inputs": frozen})


def acquire_intervening_notice():
    path = OUT / "sources/20240722_080000_政策利率与操作机制公告.html"
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))
    url = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/125469/5409998/index.html"
    response = requests.get(url, timeout=(15, 40), headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    path.write_bytes(response.content)
    text = BeautifulSoup(response.content.decode("utf-8"), "html.parser").get_text(" ", strip=True)
    for check in ["2024-07-22 08:00:00", "1.80%", "1.70%", "固定利率", "数量招标"]:
        assert check in text, check
    path.with_suffix(".txt").write_text(text, encoding="utf-8")
    record = {"url": url, "path": path.relative_to(ROOT).as_posix(), "sha256": digest(path), "retrieved_at": now(),
              "published_at_current_page": "2024-07-22T08:00:00+08:00", "old_rate_pct": 1.8, "new_rate_pct": 1.7,
              "new_information": "7天逆回购降10bp，并改为固定利率、数量招标。",
              "not_same_notice_as": "09:20:30的当日操作交易公告；两条来源各自保留。",
              "historical_first_version_authenticated": False}
    save(receipt, record)
    return record


def latest(data, at):
    matched = data[data["known"] <= at].sort_values("known")
    return None if matched.empty else matched.iloc[-1]


def build():
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("本轮已完成，不覆盖旧结果。")
    freeze()
    notice = acquire_intervening_notice()
    parent = pd.read_csv(OUT / "inputs/parent56.csv")
    universe = pd.read_csv(OUT / "inputs/survey84.csv")
    market = pd.read_csv(OUT / "inputs/market.csv")
    market["date"] = pd.to_datetime(market.date)
    opens = pd.DatetimeIndex(market.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    vols = pd.read_csv(OUT / "inputs/volatility.csv").set_index("observation_date")
    monetary = pd.read_csv(OUT / "inputs/monthly_context.csv")
    monetary["known"] = pd.to_datetime(monetary.published_at, utc=True)
    credit = pd.read_csv(OUT / "inputs/credit.csv")
    credit["known"] = pd.to_datetime(credit.known_at, utc=True)
    orders = pd.read_parquet(OUT / "inputs/orders.parquet")
    # 主输入中的缺失沿用，补全原文若在单独文件中，先核对格式再另行连接。
    orders["known"] = pd.to_datetime(orders.available_at, utc=True)
    policies = pd.read_csv(OUT / "inputs/policy_nodes.csv")
    policies["known"] = pd.to_datetime(policies.source_available_upper, utc=True)
    results, paths, policy_links = [], [], []
    for old in parent.to_dict("records"):
        row = dict(old)
        at = pd.Timestamp(old["announcement_at"])
        ri = int(np.flatnonzero(opens > at)[0])
        pi = ri - 1
        entry = int(old["entry_i"])
        end = int(old["exit_i"])
        assert ri == int(old["review_i"]) and entry == ri + 1 and end == entry + 19
        first = market.iloc[ri]
        previous = market.iloc[pi]
        review_date = str(first.date.date())
        pre_date = str(previous.date.date())
        row.update(pre_announcement_close_date=pre_date, first_reaction_date=review_date,
                   first_gap_return=float(first.open / previous.close - 1),
                   first_intraday_contribution=float((first.close - first.open) / previous.close),
                   first_dividend_contribution=float(first.dividend / previous.close),
                   first_total_return=float((first.close + first.dividend) / previous.close - 1),
                   pre_return20=float(previous.wealth / market.iloc[pi - 20].wealth - 1),
                   pre_return60=float(previous.wealth / market.iloc[pi - 60].wealth - 1))
        for label, date in [("pre", pre_date), ("review", review_date)]:
            for field in ["v_upside20", "v_downside20", "v_rv20", "volatility_source_state", "downside_window_state"]:
                row[label + "_" + field] = vols.loc[date, field]
        money = latest(monetary, at)
        loan = latest(credit, at)
        order = latest(orders, at)
        if money is not None:
            for field in ["stat_month", "published_at", "m1_yoy_pp", "m2_yoy_pp", "spread_pp", "delta3_spread_pp", "definition_version"]:
                row["known_money_" + field] = money[field]
            row["known_money_age_days"] = (at - money.known).total_seconds() / 86400
        if loan is not None:
            for field in ["stat_month", "known_at", "corporate_long_ytd_yoy_change_yi", "household_long_ytd_yoy_change_yi", "bills_ytd_yoy_change_yi",
                          "corporate_long_ytd_yoy_direction", "household_long_ytd_yoy_direction", "source_url", "source_sha256"]:
                row["known_credit_" + field] = loan[field]
        if order is not None:
            for field in ["reference_period", "available_at", "first_release_value", "source_url", "source_hash"]:
                row["known_orders_" + field] = order[field]
            row["known_orders_age_days"] = (at - order.known).total_seconds() / 86400
        easier = any(pd.notna(old[f"{t}_surprise_max_bp"]) and old[f"{t}_surprise_max_bp"] < 0 for t in ["t1", "t5"])
        tighter = any(pd.notna(old[f"{t}_surprise_min_bp"]) and old[f"{t}_surprise_min_bp"] > 0 for t in ["t1", "t5"])
        vote_miss = any(pd.notna(old[f"{t}_cut_vote_share_min"]) and old[f"{t}_cut_vote_share_min"] > .5 and old[f"{t}_actual_change_bp"] >= 0 for t in ["t1", "t5"])
        row["at_least_one_tenor_easier_than_old_survey"] = easier
        row["at_least_one_tenor_tighter_than_old_survey"] = tighter
        row["majority_cut_expectation_not_realized"] = vote_miss
        row["case_selected_by_survey_only"] = easier or tighter or vote_miss
        row["intervening_information_status"] = "NOT_SYSTEMATICALLY_CHECKED_NOT_ASSUMED_ABSENT"
        row["updated_numeric_consensus_after_intervening_notice"] = "UNKNOWN"
        if old["month"] == "2024-07":
            survey_upper = max(pd.Timestamp(old["survey_published_at"]), pd.Timestamp(old["survey_modified_at"]))
            intermediate = pd.Timestamp(notice["published_at_current_page"])
            assert survey_upper < intermediate < at < opens[ri]
            row["intervening_information_status"] = "POLICY_RATE_CUT_OBSERVED_AFTER_SURVEY_BEFORE_LPR"
            row["intervening_notice_at"] = intermediate.isoformat()
            row["intervening_notice_url"] = notice["url"]
            row["minutes_notice_to_lpr"] = (at - intermediate).total_seconds() / 60
        entry_price = float(market.iloc[entry].open)
        dividend = float(market.iloc[entry + 1:end + 1].dividend.sum())
        rebuilt = (market.iloc[end].close + dividend) / entry_price - 1
        assert abs(rebuilt - old["target20"]) < 1e-11, (old["month"], rebuilt, old["target20"])
        row["parent20_recomputed"] = rebuilt
        row["same_end_first_open_return"] = float((market.iloc[end].close + market.iloc[ri + 1:end + 1].dividend.sum()) / first.open - 1)
        row["same_end_entry_delay_price_ratio"] = entry_price / first.open
        balances = market.iloc[entry:end + 1].close.to_numpy(float) + np.r_[0, market.iloc[entry + 1:end + 1].dividend.to_numpy(float).cumsum()]
        normalized = np.r_[1.0, balances / entry_price]
        row["parent20_worst_close_from_entry"] = float(np.min(normalized) - 1)
        row["parent20_drawdown"] = float(np.min(normalized / np.maximum.accumulate(normalized) - 1))
        for index in range(ri, end + 1):
            day = market.iloc[index]
            elapsed = index - ri
            cash = float(market.iloc[ri + 1:index + 1].dividend.sum())
            paths.append({"month": old["month"], "date": str(day.date.date()), "session_from_reaction": elapsed,
                          "first_open_price": float(first.open), "close": float(day.close), "dividend_after_first_open": cash,
                          "first_open_to_close_return": float((day.close + cash) / first.open - 1),
                          "role": "固定起点的事后路径，不是公告单项因果贡献"})
        end_time = pd.Timestamp(market.iloc[end].date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
        future = policies[(policies.known > at) & (policies.known <= end_time)]
        row["recorded_later_policy_nodes"] = len(future)
        for node in future.to_dict("records"):
            policy_links.append({"month": old["month"], **{k: v for k, v in node.items() if k != "known"},
                                 "role": "只解释路径；目录不完整，节点不等于独立事件"})
        results.append(row)
    full = pd.DataFrame(results)
    selected = full[full.case_selected_by_survey_only].copy()
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    assert selected.month.tolist() == protocol["case_months"]
    pd.testing.assert_frame_equal(parent, full[parent.columns], check_dtype=False, atol=1e-12, rtol=1e-12)
    csv(full, "56个月_原调查新信息时序与价格波动.csv")
    csv(selected, "七个调查偏差病例_完整时序与传导.csv")
    csv(pd.DataFrame(paths), "56个月_第一完整反应日至原终点的逐日路径.csv")
    csv(pd.DataFrame(policy_links), "原观察窗口内_已记录的后续政策节点.csv")
    csv(universe[~universe.month.isin(parent.month)], "28个月_调查缺失原状.csv")
    summary = {"at": now(), "status": "FULL_56_EVENT_TIMELINES_BUILT_OLD_LABELS_PRESERVED", "universe": len(universe),
               "survey_events": len(parent), "missing_surveys": len(universe) - len(parent), "cases_by_survey": len(selected),
               "daily_path_rows": len(paths), "parent_labels_recomputed": len(parent), "later_policy_links": len(policy_links),
               "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False}
    save(OUT / "results/build_receipt.json", summary)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(summary, ensure_ascii=False))
    fields = ["month", "known_money_stat_month", "known_money_spread_pp", "known_money_delta3_spread_pp", "known_orders_reference_period", "known_orders_first_release_value",
              "known_credit_corporate_long_ytd_yoy_change_yi", "known_credit_household_long_ytd_yoy_change_yi", "pre_return60", "pre_v_downside20",
              "review_v_downside20", "first_total_return", "target20", "recorded_later_policy_nodes"]
    print(selected[fields].to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    build()
