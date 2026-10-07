"""补全固定分钟研究的数据清单与保存结果解释，不生成新信号或重跑账户。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_intraday_process_increment_v1"
MARKER = "\n## 保存结果的进一步解释（无新拟合或账户）\n"


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def complete_inventory() -> dict:
    inventory = pd.read_csv(OUT / "01_实际数据清单.csv").fillna("")
    records = inventory.to_dict("records")
    roles = {row["用途"] for row in records}
    additions = {
        "restricted_nbs_minutes": "data/curated/510300_stk_mins_source_admission_v1/510300_1min.parquet",
        "restricted_nbs_contract": "config/510300_nbs_fixed_5min_usage_v2_1.json",
        "breadth_historical_cache": "data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/internal_F_T_features.parquet",
        "breadth_existing_use": "research/breadth_majority_inputs_v1.py",
        "breadth_existing_contract": "config/510300_breadth_majority_trend_v1.json",
        "breadth_equal_recent": "data/features/000300_equal_weight_breadth_daily.parquet",
        "breadth_equal_earlier": "data/features/000300_external_equal_weight_breadth_2015_2021.parquet",
        "breadth_official_weighted": "data/features/000300_official_weighted_breadth_daily.parquet",
        "breadth_official_warmup": "data/features/000300_official_weighted_breadth_daily_v1_0_1_warmup.parquet",
        "breadth_cap_weighted": "data/features/000300_circulating_cap_weighted_breadth_daily.parquet",
        "breadth_labeled_old_dataset": "data/features/510300_round3_breadth_dataset.parquet",
    }
    for role, relative in additions.items():
        if role not in roles:
            records.append({"用途": role, "路径": str(ROOT / relative)})
    for row in records:
        path = Path(row["路径"])
        row.update({"SHA256": sha(path), "字节": path.stat().st_size})
        row.setdefault("使用权限", "现有本地项目研究；本轮没有取得新增许可或再分发授权")
        if not row["使用权限"]:
            row["使用权限"] = "现有本地项目研究；来源回执不等于再分发授权"
        if path.suffix == ".parquet":
            table = pq.ParquetFile(path)
            row.update({"行数": table.metadata.num_rows, "字段": ", ".join(table.schema.names)})
            date_col = next((x for x in ("trade_time", "date", "trade_date") if x in table.schema.names), None)
            if date_col:
                dates = pd.to_datetime(pd.read_parquet(path, columns=[date_col])[date_col]).dt.normalize()
                row.update({"开始": str(dates.min().date()), "结束": str(dates.max().date()), "日期数": int(dates.nunique())})
        else:
            for field, value in {"字段": "非行情表；原格式完整保留", "时间标签": "不适用：协议、来源回执或代码", "单位": "不适用：非行情数值表"}.items():
                if not row.get(field):
                    row[field] = value
        role = row["用途"]
        if role.startswith("breadth_"):
            row.update({"本轮用途": "仅盘点既有缓存与旧用途；未作为A/B/C输入", "已有准入用途": "依原字段/成员时钟及独立覆盖规则；不得把整表存在视为字段准入", "时间标签": "date是日频观察日期，不等于逐日实际接收时间", "单位": "比例、收益与相关系数为无量纲；成员数量为只；状态为枚举"})
            if role == "breadth_labeled_old_dataset":
                row["异常及限制"] = "包含未来20日标签，不能将整表直接投入特征；本轮不使用"
        if role == "restricted_nbs_minutes":
            row.update({"单位": "价格元、数量份、金额元", "时间标签": "旧NBS用途合同BAR_END；不移植到主分钟来源", "已有准入用途": "EXACT_EIGHT_FIXED_5MIN_WINDOWS_ONLY", "本轮用途": "仅盘点，不扩展为本轮全分钟用途", "异常及限制": "263根正成交分钟VWAP越界；原分钟硬门失败仍保留"})
        if role == "minutes":
            row.update({"已有准入用途": "新冻结协议的盘后量价过程代理", "异常及限制": "严格VWAP越界5489根1166日；超一档42根38日；原值及逐日状态见02/03，不作行情修复"})
        if role == "prices":
            row.update({"单位": "价格元、数量份、金额元，以文件volume_unit/amount_unit为准", "时间标签": "date为日收盘观察日期；retrieved_at为本地回执，不倒造历史送达", "本轮用途": "日线对照、合法后续价格、完整账户计价", "异常及限制": "来源修正记录保留；日聚合一致不证明分钟准确"})
        if role == "calendar":
            row.update({"单位": "日期、开市布尔值", "时间标签": "trade_date为交易日期", "本轮用途": "完整交易日对齐，不压缩缺失日"})
        if role == "old_c_features":
            row.update({"单位": "价格/金额为元，收益与比例无量纲", "时间标签": "date及signal_asof，按旧协议盘后可知", "本轮用途": "核对同一尾盘字段，保留旧拒绝；新用途仅对日线作条件增量", "异常及限制": "非独立新来源、非新特征；依新共同质量门"})
        if role == "dividends":
            d = pd.read_csv(path)
            row.update({"行数": len(d), "字段": ", ".join(d.columns), "开始": d.ex_date.min(), "结束": d.ex_date.max(), "日期数": d.ex_date.nunique(), "单位": "每份现金分红为元", "时间标签": "登记、除息及支付日分别处理；覆盖核查截至2026-09-04", "本轮用途": "合法权益、应收和到账现金", "异常及限制": "事件日期范围不等于覆盖回执范围"})
    pd.DataFrame(records).to_csv(OUT / "01_实际数据清单.csv", index=False, encoding="utf-8-sig")

    nbs = pd.read_parquet(ROOT / additions["restricted_nbs_minutes"])
    nbs_vwap = nbs.amount / nbs.vol.replace(0, np.nan)
    bad = nbs.vol.gt(0) & ((nbs_vwap < nbs.low - .001 - 1e-12) | (nbs_vwap > nbs.high + .001 + 1e-12))
    old_receipt = json.loads((ROOT / "reports/data_quality/510300_nbs_fixed_5min_usage_v2_1/source_adjudication.json").read_text(encoding="utf-8"))
    assert sha(ROOT / additions["restricted_nbs_minutes"]) == old_receipt["candidate"]["sha256"]
    assert int(bad.sum()) == old_receipt["bar_vwap_out_of_range_rows"] == 263

    breadth = pd.read_parquet(ROOT / additions["breadth_historical_cache"])
    exact_gate = (breadth.breadth20.between(0, 1) & breadth.point_in_time_member_count.eq(300)
                  & breadth.return20_scoreable_member_count.between(294, 300)
                  & breadth.return20_scoreable_member_count.mod(1).eq(0)
                  & breadth.return20_coverage_ratio.ge(.98)
                  & np.isclose(breadth.return20_scoreable_member_count / breadth.point_in_time_member_count,
                               breadth.return20_coverage_ratio, rtol=0, atol=1e-12)
                  & breadth.four_state_daily_coverage_state.eq("VIEW_ALLOWED"))
    coverage = breadth[["date", "breadth20", "return20_coverage_ratio", "internal_feature_state", "four_state_daily_coverage_state"]].copy()
    coverage["旧原始breadth20独立覆盖门"] = exact_gate
    coverage["本轮进入模型"] = False
    coverage.to_csv(OUT / "19_旧广度逐日字段可用性.csv", index=False, encoding="utf-8-sig")
    result = {
        "数据清单记录数": len(records),
        "长分钟": {"rows": len(nbs), "days": int(nbs.trade_time.dt.normalize().nunique()), "first": nbs.trade_time.min().isoformat(), "last": nbs.trade_time.max().isoformat(), "identical_to_old_receipt_candidate": True, "bad_rows": int(bad.sum()), "bad_dates": sorted(nbs.loc[bad, "trade_time"].dt.strftime("%Y-%m-%d").unique().tolist()), "scope": old_receipt["scope"], "used_for_new_model": False},
        "旧广度": {"rows": len(breadth), "breadth20_nonmissing": int(breadth.breadth20.notna().sum()), "breadth20_missing": int(breadth.breadth20.isna().sum()), "old_raw_breadth_gate_passes": int(exact_gate.sum()), "composite_feature_states": breadth.internal_feature_state.value_counts().to_dict(), "all_column_missing_counts": {k: int(v) for k, v in breadth.isna().sum().items()}, "used_for_new_model": False},
        "输入范围": "本次补充只盘点；新实验仍使用freeze.json原输入，没有引入广度或延长分钟历史",
    }
    write_json(OUT / "18_相关缓存覆盖与异常.json", result)
    return result


def interpret_saved() -> dict:
    cycles = pd.read_csv(OUT / "11_完成持有周期.csv")
    metrics = pd.read_csv(OUT / "12_全部账户指标.csv")
    ledgers = pd.read_parquet(OUT / "08_完整账户逐日账本.parquet")
    rows = []
    for key, part in cycles.groupby(["capital", "cost", "model"]):
        ordered = part.sort_values("net_pnl", ascending=False)
        total = float(part.net_pnl.sum())
        best = ordered.iloc[0]
        metric = metrics.loc[metrics.capital.eq(key[0]) & metrics.cost.eq(key[1]) & metrics.model.eq(key[2])].iloc[0]
        assert np.isclose(total, metric.completed_net_pnl, rtol=0, atol=1e-6)
        assert np.isclose(part.gross_quote_pnl.sum() - part.commission.sum() - part.slippage.sum(), total, rtol=0, atol=1e-6)
        rows.append({"capital": int(key[0]), "cost": key[1], "model": key[2], "cycles": len(part), "net_pnl": total, "gross_quote_pnl": float(part.gross_quote_pnl.sum()), "commission": float(part.commission.sum()), "slippage": float(part.slippage.sum()), "best_net_pnl": float(best.net_pnl), "best_entry_date": best.entry_date, "best_exit_date": best.exit_date, "best_share_of_total_net": float(best.net_pnl / total) if total else None, "top3_share_of_total_net": float(ordered.head(3).net_pnl.sum() / total) if total else None, "median_cycle_net_return": float(part.net_return.median()), "worst_cycle_net_return": float(part.net_return.min())})
    concentration = pd.DataFrame(rows)
    concentration.to_csv(OUT / "16_全账户收益集中度.csv", index=False, encoding="utf-8-sig")
    monthly = []
    for (capital, cost), part in ledgers.groupby(["capital", "cost"]):
        wide = part.pivot(index="date", columns="model", values="net_return").sort_index()
        for month, group in wide.groupby(wide.index.to_period("M")):
            for model in ("A", "B", "C"):
                monthly.append({"capital": capital, "cost": cost, "month": str(month), "model": model, "days": len(group), "sum_daily_return_difference": float((group[model] - group.D).sum()), "model_month_return": float(np.prod(1 + group[model]) - 1), "baseline_month_return": float(np.prod(1 + group.D) - 1)})
    pd.DataFrame(monthly).to_csv(OUT / "17_逐月相对日线归因.csv", index=False, encoding="utf-8-sig")
    main = concentration.loc[concentration.capital.eq(200000) & concentration.cost.eq("BASE")].set_index("model")
    unknown = ledgers.loc[ledgers.capital.eq(200000) & ledgers.cost.eq("BASE") & ledgers.model.eq("D") & ledgers.prediction_state.eq("NO_VIEW")]
    text = [MARKER, "以下仅解释原保存预测、周期与账户，不改变冻结代码、窗口、方向、标签、成员或任何仓位。", "", "**主要瓶颈在增量信号。** A/B的毛收益预测MSE已比日线增加约1.72%/1.67%，且两个半段均无改善；费用不是这一结果的原因。C误差增加约0.11%，小幅账户收益改善仍没有正的增量下界，不能靠调低手续费宣布机制通过。", "", "C的20万元基础账户CAGR比D高约0.8580个百分点；另外，逐日配对的年化算术收益差约+0.7605个百分点，校正区间[-0.0683,+2.5083]个百分点。这两个数采用不同定义，区间只对应后者。", "", "|20万元基础模型|同路径毛报价损益（元）|佣金（元）|滑点（元）|净损益（元）|最大单周期占净利润|前三周期占净利润|", "|---|---:|---:|---:|---:|---:|---:|"]
    for model in ("D", "A", "B", "C"):
        row = main.loc[model]
        text.append(f"|{model}|{row.gross_quote_pnl:,.2f}|{row.commission:,.2f}|{row.slippage:,.2f}|{row.net_pnl:,.2f}|{row.best_share_of_total_net:.2%}|{row.top3_share_of_total_net:.2%}|")
    text += ["", "四组最大贡献均来自2024-09-27开盘至2024-09-30收盘的同一市场阶段。分母是包含亏损后的总净利润，因此前三占比可以超过100%；这是集中度，不是概率。不删除最大交易、不按月筛选或重跑账户。完整16账户集中度见16表，全部月份配对贡献见17表。", "", f"**缺失会影响表面平稳。** 471个完整账户日中有{len(unknown)}个NO_VIEW，范围{unknown.date.min().date()}至{unknown.date.max().date()}。此时仍保留库存、应收、现金和市值变化；该样本后段恰为空仓，不能归因于择时成功。所有日历均保留，不能将364个预测日误写为471日持续有信号。", "", "额外缓存已经实际读取：旧长分钟566,350行/2,350日，哈希与旧来源裁决中的候选相同，263根异常仍在；准入继续仅八个固定五分钟窗口。旧广度2,823日中breadth20非缺失2,350日、缺失473日；综合internal_feature_state只有1,280日VIEW_ALLOWED，1,543日NO_VIEW。二者是不同字段，不能互相代替。本轮没有使用它们增补输入。", "", "**本轮处置与下一步。** 冻结A/B/C负结果，完成当前新目标所要求的数据、过程、预测和完整账户交付。当前没有已通过的下一策略实验，不立即叠加广度、改持有期、改变信号方向或按波动状态重试。本轮已完成信号、费用、容量/缺失和贡献集中度解释；任何后续新实验先说明新增信息用途，核对旧同构研究和独立样本，冻结单一比较后才能计算。若仍是同一信息与已暴露样本，则停止该提案。盘口/IOPV/队列继续为储备，不阻塞分钟研究，也不从分钟表伪造这些字段。", "", "当前研究交付完成不代表盈利目标通过：A/B/C净夏普均低于1.2，预测及收益增量未通过，独立验证NOT_ESTABLISHED，真实成交验证数0。"]
    report = OUT / "研究结论与下一步.md"
    report.write_text(report.read_text(encoding="utf-8").split(MARKER)[0].rstrip() + "\n" + "\n".join(text) + "\n", encoding="utf-8")
    return {"account_scenarios": len(rows), "monthly_rows": len(monthly), "unknown_account_days": len(unknown), "new_fits": 0, "new_accounts": 0, "new_thresholds": 0}


def main() -> None:
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    for item in frozen["files"]:
        assert sha(Path(item["path"])) == item["sha256"], "冻结文件已变化：" + item["path"]
    immutable_names = ["06_滚动预测.parquet", "07_全部模型参数.json", "08_完整账户逐日账本.parquet", "09_全部模拟请求.csv", "10_逐日决策.parquet", "11_完成持有周期.csv", "12_全部账户指标.csv", "13_账户相对日线增量.csv", "summary.json", "saved_output_verification.json"]
    before = {name: sha(OUT / name) for name in immutable_names}
    inventory = complete_inventory()
    interpretation = interpret_saved()
    objective = next(Path(x["path"]) for x in frozen["files"] if Path(x["path"]).name == "goal-objective.md")
    (OUT / "goal_objective.md").write_bytes(objective.read_bytes())
    assert all(sha(OUT / name) == value for name, value in before.items()), "原预测或账户意外变化"
    receipt = {"completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "status": "COMPLETED_SAVED_OUTPUT_INTERPRETATION_NO_NEW_EXPERIMENT", "script_sha256": sha(Path(__file__)), "frozen_files_unchanged": len(frozen["files"]), "saved_core_hashes_unchanged": before, "inventory_records": inventory["数据清单记录数"], "breadth20_nonmissing": inventory["旧广度"]["breadth20_nonmissing"], **interpretation}
    write_json(OUT / "post_result_interpretation_receipt.json", receipt)
    files = [{"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "file_index.json"]
    total = sum(x["bytes"] for x in files)
    assert total <= 30 * 1024 * 1024, "超出固定存储预算"
    write_json(OUT / "file_index.json", {"files": files, "total_bytes": total, "index_excludes_itself": True})
    print(json.dumps({"状态": receipt["status"], "数据清单项": inventory["数据清单记录数"], "账户解释数": interpretation["account_scenarios"], "报告总字节": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
