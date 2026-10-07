"""修正已核实的深市零值缓存错误，独立保存并按原参数重算受影响评分。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import multidim_financing_composition_v1 as composition
import multidim_money_surprise_score_v1 as monthly
import multidim_event_state_control_v1 as control


ROOT = composition.ROOT
OUT = composition.OUT / "data_correction"
SOURCE = composition.OUT / "sources/szse_20240808.json"
FIELDS = {"rzmre": "jrrzmr", "rzye": "jrrzye", "rqmcl": "jrrjmc", "rqyl": "jrrjyl", "rqye": "jrrjye", "rzrqye": "jrrzrjye"}


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(composition.common.clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def repair():
    if (OUT / "correction_receipt.json").exists():
        raise RuntimeError("本次单日数据纠正已保存，不覆盖。")
    payload = composition.read(SOURCE)[0]
    assert payload["metadata"]["subname"] == "2024-08-08" and len(payload["data"]) == 1
    raw = payload["data"][0]
    values = {name: float(raw[key].replace(",", ""))*1e8 for name, key in FIELDS.items()}
    assert values["rzye"] == 664954000000 and values["rzmre"] == 22088000000
    old = pd.read_parquet(composition.MARGIN).sort_values("date").reset_index(drop=True)
    m = old.copy(deep=True)
    target = m.date.eq("2024-08-08")
    assert target.sum() == 1 and (m.loc[target, ["rzye_szse", "rzmre_szse"]] == 0).all().all()
    assert not (old.loc[~target, ["rzye_sse", "rzye_szse", "rzmre_sse", "rzmre_szse"]] <= 0).any().any()
    for name, value in values.items():
        m.loc[target, name+"_szse"] = value
        m.loc[target, "market_"+name] = m.loc[target, name+"_sse"] + value
    m.loc[target, "source"] = "SSE_ORIGINAL_PLUS_SZSE_OFFICIAL_ZERO_ROW_CORRECTION_20240808"
    m["market_rzye_change"] = m.market_rzye.diff()
    m["market_financing_balance_identity_residual"] = m.market_rzye_change - m.market_rzmre
    unchanged_cols = [c for c in old.columns if c not in ["market_rzye_change", "market_financing_balance_identity_residual"]]
    pd.testing.assert_frame_equal(old.loc[~target, unchanged_cols], m.loc[~target, unchanged_cols])
    assert (m.market_rzmre - m.market_rzye.diff()).dropna().ge(0).all()
    OUT.mkdir(parents=True, exist_ok=True)
    fixed_margin = OUT / "margin_corrected_20240808.parquet"
    m.to_parquet(fixed_margin, index=False)
    original = pd.read_parquet(composition.DAILY)
    d = original.copy(deep=True)
    raw_features = m[["date"]].copy()
    raw_features["融资五日净变化"] = m.market_rzye.pct_change(5, fill_method=None)
    raw_features["融资买入活跃度"] = np.log(m.market_rzmre.rolling(5).mean()/m.market_rzmre.rolling(60).mean())
    names = ["融资五日净变化", "融资买入活跃度"]
    aligned = d[["date"]].merge(raw_features, on="date", how="left", validate="one_to_one")
    d[names] = aligned[names].shift(1)
    d["valid_features"] = np.isfinite(d[composition.common.FEATURES].to_numpy(float)).all(axis=1)
    other_cols = [c for c in d.columns if c not in names+["valid_features"]]
    pd.testing.assert_frame_equal(original[other_cols], d[other_cols])
    changed = []
    for name in names:
        mask = ~np.isclose(original[name], d[name], rtol=1e-12, atol=1e-14, equal_nan=True)
        for index in d.index[mask]:
            changed.append({"date": d.at[index, "date"], "feature": name,
                            "old_value": original.at[index, name], "corrected_value": d.at[index, name]})
    changed_frame = pd.DataFrame(changed)
    changed_frame.to_csv(OUT / "受影响的原评分输入.csv", index=False, encoding="utf-8-sig")
    fixed_daily = OUT / "daily_inputs_corrected_20240808.parquet"
    d.to_parquet(fixed_daily, index=False)
    receipt = {
        "at": composition.common.now(), "status": "OFFICIAL_SINGLE_ROW_CORRECTION_SAVED",
        "trigger": "融资构成分解在产生任何新收益汇总前发现2024-08-09隐含偿还为负；追溯到前日深市所有金额字段为0。",
        "bad_stat_date": "2024-08-08", "old_transport": "jin10.cdn.history历史副本中的零行，未被旧官方抽样覆盖。",
        "old_values": {name: float(old.loc[target, name+"_szse"].iloc[0]) for name in FIELDS},
        "corrected_values": values,
        "official_url": "https://www.szse.cn/api/report/ShowReport/data?SHOWTYPE=JSON&CATALOGID=1837_xxpl&txtDate=2024-08-08&tab1PAGENO=1",
        "official_file": SOURCE.relative_to(ROOT).as_posix(), "official_sha256": composition.sha(SOURCE),
        "source_precision": "官网以亿元保留两位小数；约100万元显示单位，非逐元精确值。",
        "first_vintage": False, "tls_verified": True,
        "transport_note": "直接连接出现TLS EOF；保留默认环境网络设置的HTTPS请求成功，未关闭证书检查。",
        "original_margin_path": composition.MARGIN.relative_to(ROOT).as_posix(), "original_margin_sha256": composition.sha(composition.MARGIN),
        "corrected_margin_path": fixed_margin.relative_to(ROOT).as_posix(), "corrected_margin_sha256": composition.sha(fixed_margin),
        "original_daily_path": composition.DAILY.relative_to(ROOT).as_posix(), "original_daily_sha256": composition.sha(composition.DAILY),
        "corrected_daily_path": fixed_daily.relative_to(ROOT).as_posix(), "corrected_daily_sha256": composition.sha(fixed_daily),
        "changed_input_cells": len(changed_frame), "changed_input_dates": int(changed_frame.date.nunique()),
        "changed_feature_counts": changed_frame.feature.value_counts().to_dict(),
        "prices_labels_macro_and_dividends_unchanged": True,
        "old_files_overwritten": False, "parameter_changes": 0,
        "old_monthly_and_ordinary_findings": "SOURCE_CORRECTION_RECOMPUTATION_PENDING",
        "old_factor96_failed_rules": "旧失败保留；本次未重新搜索旧策略，也不将其错误输入结果提升为有效证据。",
        "new_composition_run": "输入错误发现于结果之前；首次停止后只纠正这一条官方确认的输入，不改变任何分组、窗口或成本。",
    }
    save("correction_receipt.json", receipt)
    print("官方单日纠正已保存；旧文件不变。", flush=True)
    print("受影响输入：", receipt["changed_feature_counts"], "日期数", receipt["changed_input_dates"])


def recalculate():
    receipt = composition.read(OUT / "correction_receipt.json")
    if (OUT / "recalculation_result.json").exists():
        raise RuntimeError("已完成原评分纠正对照，不重复运行。")
    corrected_daily = ROOT / receipt["corrected_daily_path"]
    assert composition.sha(corrected_daily) == receipt["corrected_daily_sha256"]
    original_monthly = monthly.OUT
    original_control = control.OUT
    monthly.OUT = OUT / "monthly_score"
    monthly.DAILY = corrected_daily
    monthly.prepare()
    monthly.run()
    control.OUT = OUT / "event_state_control"
    control.PREVIOUS = monthly.OUT
    control.DAILY = corrected_daily
    control.prepare()
    control.run()
    before = pd.read_csv(original_monthly / "逐事件联合评分.csv").set_index("stat_month")
    after = pd.read_csv(monthly.OUT / "逐事件联合评分.csv").set_index("stat_month")
    assert before.index.tolist() == after.index.tolist()
    changed = []
    for name in ["state_prediction", "state_score", "joint_prediction", "joint_score", "linear_prediction", "mean_prediction"]:
        for month in before.index[~np.isclose(before[name], after[name], rtol=1e-12, atol=1e-12)]:
            changed.append({"stat_month": month, "field": name, "old_value": before.at[month, name], "corrected_value": after.at[month, name]})
    pd.DataFrame(changed, columns=["stat_month", "field", "old_value", "corrected_value"]).to_csv(OUT / "原月度评分纠正差异.csv", index=False, encoding="utf-8-sig")
    opportunities_old = pd.read_csv(original_monthly / "全部固定高分机会.csv")
    opportunities_new = pd.read_csv(monthly.OUT / "全部固定高分机会.csv")
    keys = ["model", "stat_month", "entry_date", "exit_date"]
    signals_identical = opportunities_old[keys].equals(opportunities_new[keys])
    ordinary_old = pd.read_csv(original_control / "公布与普通日期比较.csv")
    ordinary_new = pd.read_csv(control.OUT / "公布与普通日期比较.csv")
    paired = ordinary_old.merge(ordinary_new, on=["era", "group", "selection"], suffixes=("_old", "_corrected"), validate="one_to_one")
    paired.to_csv(OUT / "普通日期纠正前后比较.csv", index=False, encoding="utf-8-sig")
    result = {"completed_at": composition.common.now(), "status": "SAME_PARAMETER_SOURCE_CORRECTION_RECOMPUTED",
              "monthly_recomputed_rows": len(after), "monthly_changed_fields": len(changed),
              "monthly_changed_events": len({r["stat_month"] for r in changed}),
              "all_state_and_joint_entry_exit_events_identical": bool(signals_identical),
              "parameter_changes": 0, "new_candidate_families": 0,
              "account_recalculation_required": not signals_identical,
              "monthly_corrected_path": monthly.OUT.relative_to(ROOT).as_posix(),
              "ordinary_corrected_path": control.OUT.relative_to(ROOT).as_posix(),
              "old_files_unchanged": True, "goal_achieved": False}
    save("recalculation_result.json", result)
    print("原模型同参数纠正比较：", result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="深市单日零值错误的官方纠正及影响检查")
    parser.add_argument("stage", choices=["repair", "recalculate"])
    {"repair": repair, "recalculate": recalculate}[parser.parse_args().stage]()
