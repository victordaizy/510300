"""归纳贷款预期偏差的固定联合评分与账户结果，保留全部反例。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import multidim_loan_surprise_score_v1 as study


ROOT, OUT = study.ROOT, study.OUT
REPORT = "贷款预期偏差与指数联合评分_历史结果.md"
FIGURE = "贷款预期偏差_分层与连续账户.png"


def tree_path(model, row):
    node, path = 0, []
    while model["children_left"][node] != -1:
        feature = model["features"][model["feature"][node]]
        threshold = model["threshold"][node]
        value = float(np.float32(row[feature]))
        left = value <= threshold
        path.append({"feature": feature, "value": float(row[feature]), "threshold": threshold, "branch": "<=" if left else ">"})
        node = model["children_left"][node] if left else model["children_right"][node]
    return {"conditions": path, "leaf_training_events": model["samples"][node], "prediction": model["value"][node]}


def calculate_review(predictions, result, accounts):
    models = {r["stat_month"]: r for r in study.read(OUT / "saved_models.json")}
    paths, error = [], 0.
    for _, row in predictions.iterrows():
        entry = {"stat_month": row.stat_month, "entry_date": row.entry_date, "actual_net5": row.actual_net5}
        for label in ["control", "joint"]:
            path = tree_path(models[row.stat_month][label + "_tree"], row)
            error = max(error, abs(path["prediction"] - row[label + "_prediction"]))
            entry[label] = path
        paths.append(entry)
    if error > 1e-12:
        raise ValueError("保存树与保存预测不一致。")
    primary = predictions[predictions.era.eq("2024—2025")]
    err = {(x["era"], x["model"]): x["mse"] for x in result["model_errors"]}
    opportunity = pd.read_csv(OUT / "全部高分机会.csv")
    selected = opportunity[opportunity.era.eq("2024—2025")]
    left = selected[selected.model.eq("control")].entry_date.to_list()
    right = selected[selected.model.eq("joint")].entry_date.to_list()
    cycle = pd.DataFrame(accounts["cycles"])
    current = cycle[cycle.case_id.eq("joint_200000")]
    extra = current[~current.entry.isin(cycle[cycle.case_id.eq("control_200000")].entry)]
    loan = study.read(study.ORIGINALS)
    actual = next(x for x in loan if x["stat_month"] == "2023-01")
    row = predictions[predictions.stat_month.eq("2023-01")].iloc[0]
    override = study.read(ROOT / "config/510300_financing_source_correction_20240808_v1.json")
    daily = pd.read_parquet(ROOT / override["corrected_daily_path"])
    background = daily.iloc[int(row.state_idx)]
    review = {
        "completed_at": study.common.now(), "saved_tree_prediction_max_error": error,
        "primary_mse_relative_change": err["2024—2025", "joint"] / err["2024—2025", "control"] - 1,
        "earlier_mse_relative_change": err["2021—2023", "joint"] / err["2021—2023", "control"] - 1,
        "primary_high_entries_identical": left == right, "primary_high_entries": right,
        "primary_changed_predictions": int(primary.prediction_changed.sum()),
        "primary_high_current_path_features": {r["stat_month"]: [c["feature"] for c in r["joint"]["conditions"]]
                                               for r in paths if r["entry_date"] in right},
        "new_main_account_cycles": extra.to_dict("records"),
        "extra_case": {
            "stat_month": "2023-01", "forecast_yi": float(row.forecast_loan_yi), "actual_yi": float(row.actual_loan_yi),
            "surprise_yi": float(row.loan_surprise_yi), "surprise_display_lower_yi": float(row.loan_surprise_lower_yi),
            "surprise_display_upper_yi": float(row.loan_surprise_upper_yi), "forecast_file": row.forecast_file,
            "forecast_page": int(row.forecast_page), "forecast_date": row.forecast_date,
            "official_source_url": actual["source_url"], "department_evidence": actual["department_evidence"],
            "state_date": str(background.date), "pre20_wealth_return": float(background.mom20),
            "pre60_wealth_return": float(background.mom60), "orders_diffusion_index": float(background["订单水平"] + 50),
            "orders_monthly_change_pp": float(background["订单月度变化"]),
            "net5_label": float(row.actual_net5), "score": float(row.joint_score),
            "background_role": "结果完成后的病例解释；这些背景没有用于新增筛选或改变本次账户。部门净增不等于部门预期差，缺少各部门调查值。"
        },
        "source_page_visual_review": ["Bls210208.pdf 第12页", "Bls230209.pdf 第10页", "Bls240314.pdf 第14页", "Bls240411.pdf 第15页", "Bls250911.pdf 第15页"],
        "new_parameters_fitted_in_review": 0, "new_candidates_admitted": 0, "goal_achieved": False
    }
    study.save("逐事件实际决策路径.json", paths)
    study.save("结果解释与新增机会.json", review)
    return review


def draw(result, review):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.patch.set_facecolor("#f7f8fa")
    colors = {"control": "#8094a9", "joint": "#177f83"}
    names = {"control": "同事件九变量对照", "joint": "增加贷款预期偏差"}
    frame = pd.DataFrame(result["buckets"])
    frame = frame[frame.era.eq("2024—2025")]
    for label, offset in [("control", -.18), ("joint", .18)]:
        part = frame[frame.model.eq(label)]
        y = part.mean_net5.fillna(0).to_numpy() * 100
        axes[0].bar(np.arange(5)+offset, y, width=.34, color=colors[label], label=names[label])
        for i, (mean, count) in enumerate(zip(y, part.n)):
            axes[0].text(i+offset, mean + (.06 if mean >= 0 else -.06), f"n={count}" if count else "空", fontsize=9,
                         ha="center", va="bottom" if mean >= 0 else "top")
    axes[0].set_xticks(np.arange(5), ["0—20", "20—40", "40—60", "60—80", "80—100"])
    axes[0].set_title("主要期各分数档的实际五日净收益", loc="left", fontsize=12, pad=14)
    axes[0].set_ylabel("固定份额事件净收益均值（%）")
    axes[0].set_xlabel("分数是模型内的相对排序，不是上涨概率")
    axes[0].set_ylim(-1.65, 2.35)
    axes[0].axhline(0, lw=.8, color="#64748b")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    for label in ["control", "joint"]:
        ledger = pd.read_parquet(OUT / "accounts" / (label + "_200000.parquet"))
        axes[1].plot(ledger.date, ledger.equity / 200000, color=colors[label], label=names[label], lw=1.8)
    axes[1].axvline(pd.Timestamp("2023-02-13"), ls="--", color="#be7d36", lw=1)
    axes[1].text(pd.Timestamp("2023-02-13"), 1.041, "新增机会\n净亏1,072元", fontsize=9, ha="center", color="#9b5d1f")
    axes[1].set_title("20万元账户沿原日历连续计算", loc="left", fontsize=12, pad=14)
    axes[1].set_ylabel("净值（包括全部空仓日）")
    axes[1].set_ylim(.98, 1.052)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlabel("2021年处于训练预热，主要比较期为2024—2025年")
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#cbd5e1", alpha=.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#cbd5e1")
    fig.suptitle("贷款预期偏差进入了模型，但没有增加主要期高分机会", x=.06, y=.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(.06, .915, f"52个月来源完成对齐，40次按时序评分｜主要期20次：6次预测改变，五个高分日期相同，均方误差增加{review['primary_mse_relative_change']:.2%}", color="#475569", fontsize=10.5)
    fig.text(.06, .061, "两棵树采用相同事件、相同训练池和相同复杂度；账户沿用原压力费用、50%上限和尾部预算，不按结果更换持有期。", fontsize=10, color="#475569")
    fig.text(.06, .027, "主要期联合账户：净夏普1.226，年化1.92%，五次交易；最大一笔占净利润83.55%。完整目标未完成。", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.06, right=.985, bottom=.2, top=.815, wspace=.2)
    fig.savefig(OUT / FIGURE, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("贷款增量报告已交付，不覆盖。")
    result = study.read(OUT / "result.json")
    accounts = study.read(OUT / "account_result.json")
    predictions = pd.read_csv(OUT / "逐事件贷款预期联合评分.csv")
    review = calculate_review(predictions, result, accounts)
    draw(result, review)
    lookup = {(r["case_id"], r["period"]): r for r in accounts["accounts"]}
    main_account = lookup["joint_200000", "主要期2024—2025"]
    control = lookup["control_200000", "主要期2024—2025"]
    full = lookup["joint_200000", "完整期2021—2025"]
    extra = review["new_main_account_cycles"][0]
    case = review["extra_case"]
    lines = ["| 账户 | 净夏普 | 年化收益 | 最大回撤 | 净利润 | 完成交易 |", "|---|---:|---:|---:|---:|---:|"]
    for label, capital in [("control", 200000), ("joint", 200000), ("control", 20000), ("joint", 20000)]:
        row = lookup[f"{label}_{capital}", "主要期2024—2025"]
        name = ("九变量对照" if label == "control" else "加入贷款偏差") + ("，20万元" if capital == 200000 else "，2万元")
        lines.append(f"| {name} | {row['net_sharpe']:.3f} | {row['cagr']:.2%} | {row['max_drawdown']:.2%} | {row['net_profit']:,.2f}元 | {row['completed_cycles']} |")
    changed = predictions[predictions.prediction_changed & predictions.era.eq("2024—2025")]
    changes = ["| 入场观察日 | 对照分数 | 联合分数 | 固定五日净收益 |", "|---|---:|---:|---:|"]
    for row in changed.itertuples():
        changes.append(f"| {row.entry_date} | {row.control_score:.1f} | {row.joint_score:.1f} | {row.actual_net5:.2%} |")
    contents = [
        "# 贷款预期偏差与指数状态联合评分的历史结果",
        f"**新增贷款预期偏差没有改善本次固定评分的可交易结果。** 2024—2025年的20次评分中，6次预测改变，五个高分机会仍与同事件对照完全相同；均方误差增加{review['primary_mse_relative_change']:.2%}。20万元联合账户该段净夏普{main_account['net_sharpe']:.3f}、年化{main_account['cagr']:.2%}，低于对照，且仍只有2024年3次、2025年2次交易。当前表达停止调整，夏普1.2、年化10%及自然交易频率的完整目标未完成。",
        "## 这次增加了什么信息",
        "旧模型包含订单水平、订单月度变化、资金利率与政策利率差、融资五日净变化、融资买入活跃度、指数趋势强度、短长波动比、成交活跃度及M2相对事前预期的偏差。本轮只增加贷款净增相对同一期事前调查的偏差，检验它能否补充增长及政策反应信息；不预设超预期一定利好。",
        "原104个月来源选择表全部保留。本次固定单一Bualuang来源，52份原先已选报告在2025年末前对应有效公布，取得36个单月、16个年内累计调查值。单月实际值中24个可由原文直接读取，12个用当时已经公开的累计端点计算。每条都保存金额区间、单位、报告页、原文表达式及显示精度。其他来源和缺失月份没有换报告补齐。",
        "贷款实际减调查之后，扣除两者显示舍入误差合计的半宽，保留有符号的最小剩余差，再除以上一期已公布贷款余额统一规模。1个差异落在显示精度范围内记为0；来源缺失仍然是缺失。该归一化不是贷款增速，显示误差界也不是历史修订误差上限。",
        "52次公布中前12次用于训练预热，实际完成40次评分，首个入场观察日为2022年1月13日，末个为2025年11月14日。每次只用此前504个交易日内、五日收益已经完整发生的公布事件。九变量树、十变量树和十变量岭回归使用同一事件与同一成熟训练池；两棵树最多两层，每叶至少6个公布事件，没有搜索深度、门槛或持有期。",
        "## 主要期的固定比较",
        "\n".join(lines),
        "以上均为原连续账户在2024—2025年的区间结果，继承此前现金、历史权益峰值和风险预算，未在2024年初重置。2万元账户独立计入最低佣金。沿用前收盘已知价格和风险确定份额、次开盘成交、固定五日退出、T+1、分红及原尾部风险预算。",
        f"联合主账户最大一笔占该段净利润{main_account['largest_cycle_share_of_net_profit']:.2%}。五个高分事件的固定份额标签均值仍为1.90%，中位数仅0.23%，胜率60%；这些是五个观察，不能替代账户年化或稳定性。",
        "六次改变预测的日期全部列出，未只保留有利改变：\n\n" + "\n".join(changes),
        "例如2024年3月18日，新增字段把分数从75降到15，随后五日确实为负；但2024年4月15日也被压到15分，随后五日为正。主要期总体误差上升，且两个模型在这些日期都没有达到80分入场线，不能把某一次方向改善宣传为新增可交易收益。",
        f"较早阶段的20次评分，联合模型均方误差降低{-review['earlier_mse_relative_change']:.2%}，这是保留的有利结果；但高分机会从2次变为3次，固定五日净收益均值由−0.79%变为−1.09%。预测误差的改善未转化为高分入场优势。",
        "## 新增高分机会为什么需要保留为反例",
        f"新增机会是2023年2月13日至20日：当时报告中的1月贷款调查为4.20万亿元，央行公布4.90万亿元，相差7000亿元，连同显示精度后的偏差区间仍为{case['surprise_display_lower_yi']:.1f}至{case['surprise_display_upper_yi']:.1f}亿元。模型因此给出{case['score']:.1f}分，但固定份额五日净收益为{case['net5_label']:.2%}，实际20万元账户该笔净亏{-extra['profit']:,.2f}元，其中费用{-extra['profit']+extra['gross_same_holdings_profit']:,.2f}元。毛损益也为负，费用不是这笔失败的全部原因。",
        "调查来自[Bualuang 2023年2月9日原报告第10页](https://research2.bualuang.co.th/upload/Bls230209.pdf)，实际来自[央行2023年1月金融统计报告](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/1421ef5774954fd795d2e2312daf0470/index.html)。官方分项中，企事业单位贷款净增4.68万亿元、住户2572亿元、非银行业金融机构减少585亿元。总量强并不表示所有借款部门同样强；没有各部门事前调查，不能把7000亿元总量偏差直接分配到企业中长期、居民或票据。",
        f"在公告前最后一个收盘快照，510300此前20及60个交易日的含息累计收益已分别为{case['pre20_wealth_return']:.2%}和{case['pre60_wealth_return']:.2%}，当时已公布的新订单指数为{case['orders_diffusion_index']:.1f}、环比提高{case['orders_monthly_change_pp']:.1f}个百分点。这些背景与贷款上行并存，但不能据此识别该笔亏损的单一原因，也不能证明此前涨幅已经计入全部贷款意外。它们是结果后的解释材料，没有用于新增价格过滤器。",
        "保存树的实际路径显示，这次新增高分直接由贷款偏差越过训练内分界触发，路径上没有第二项条件。主要期五个高分的实际路径仍只经过融资五日净变化。模型允许非线性交互，并不意味着每次高分都已经获得多维共同确认。",
        f"沿2021—2025原日历完整计算，联合20万元账户净利润{full['net_profit']:,.2f}元、净夏普{full['net_sharpe']:.3f}、年化{full['cagr']:.2%}；2021年处于训练预热。较早期新增亏损及其后风险预算变化，也使主要期相同五个入场日期的份额与利润低于对照。该长期记录用于展示连续账户，没有把因子必须跨五年有效设为本次研究要求。",
        f"![分层及账户比较](<{(OUT / FIGURE).as_posix()}>)",
        "## 保留的限制与本轮处置",
        "52份报告是单一机构日历中的调查代理，不能代表完整市场共识。报告日期与官方发布日期可核对，但事后回取不是历史当时已经留存的首次版本认证。2023年贷款统计扩围、12个月累计作差的混合版本风险均保留；没有把央行所说的企事业单位扩大解释成纯民营需求。",
        "原始输入已被其他研究使用，本次新增字段的模型比较仍是历史开发。完成了52页字段提取及5个代表页面的可视核对，修正了把项目名称中的China误认作国家标题的解析问题，全部修正发生在评分之前。保存的80棵树预测与评分一致；四个账户均完成现金、持仓、费用、分红和自然周期核对，未新增复杂检验。",
        "本轮处置：贷款原始预期、实际值及反例保留为历史资料；当前十变量评分未带来可用增量，不继续调深度、换入场阈值、延迟入场或挑持有期。下一轮选择不同收益来源前，先核对是否有独立的新证据，避免继续堆叠同一信用消息的表达。原有暂停任务、交易权限和失败研究保持原状态。",
        "文件导航：`全部104个月的贷款预期覆盖.csv`为来源台账；`逐事件贷款预期联合评分.csv`列出全部评分；`逐事件实际决策路径.json`解释每条预测；`完整账户指标.csv`与`accounts`保留全部结果。"
    ]
    (OUT / REPORT).write_text("\n\n".join(contents) + "\n", encoding="utf-8")
    result.update(status="COMPLETED_LOAN_SURPRISE_SCORE_AND_ACCOUNTS_NO_INCREMENT", finalized_at=study.common.now(),
                  full_account_status="COMPLETED_FOUR_FIXED_ACCOUNTS", new_accounts=4,
                  candidate_disposition="REJECTED_FIXED_INCREMENT_NO_PARAMETER_RESCUE", new_admitted_strategies=0,
                  primary_mse_relative_change=review["primary_mse_relative_change"],
                  primary_high_entries_identical=review["primary_high_entries_identical"],
                  report=str((OUT / REPORT).relative_to(ROOT)).replace("\\", "/"),
                  goal_achieved=False)
    study.save("result.json", result)
    rel = str(OUT.relative_to(ROOT)).replace("\\", "/")
    summary = "新增52个月贷款事前调查偏差，固定40次十变量评分及4个账户。主要期6次预测改变但五个高分日期不变，误差增加5.98%；20万元主账户净夏普1.226、年化1.92%，完整目标未达到。新增2023年2月机会净亏1071.67元，该表达封存。"
    next_question = "贷款消息增量封存；选择不同的指数级收益来源，要求有独立于当前信用消息和融资状态的新证据。先核对既有定义与结果，避免增加本树字段、变更阈值、延迟入场或调整持有期营救当前失败。"
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    config = study.read(path)
    config.update(latest_completed_study=rel + "/result.json", latest_report=rel + "/" + REPORT,
                  current_study=rel + "/protocol.json", latest_result_summary=summary, next_historical_question=next_question,
                  updated_at=study.common.now(), goal_achieved=False,
                  latest_completed_branch_boundary="本次贷款偏差评分和四账户完成，主要期无新增高分机会，当前表达拒绝封存。")
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    config = study.read(path)
    config.update(current_round=result["study_id"], latest_progress_receipt=rel + "/result.json", last_research_result=summary,
                  latest_historical_report=rel + "/" + REPORT, latest_historical_diagnostic_at=study.common.now(),
                  latest_continuation_report=rel + "/" + REPORT,
                  latest_continuation_classification="PROGRESS_LOAN_SURPRISE_JOINT_SCORE_AND_FOUR_ACCOUNTS",
                  current_driver_continuation_classification="PROGRESS_LOAN_SURPRISE_JOINT_SCORE_AND_FOUR_ACCOUNTS",
                  current_driver_consecutive_blocked_goal_turns=0, next_research_question=next_question,
                  goal_achieved=False, latest_loan_surprise_increment=rel + "/result.json")
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    study.save("delivery_receipt.json", {"status": "COMPLETED_FIXED_LOAN_INCREMENT_AND_ACCOUNT_DELIVERY",
                                        "at": study.common.now(), "report": rel + "/" + REPORT,
                                        "new_source_months": 52, "new_fit_families": 3, "new_fits": 120,
                                        "new_accounts": 4, "new_admitted_strategies": 0, "goal_achieved": False})
    print("已交付贷款预期偏差评分、四账户和完整反例；当前表达封存，目标未完成。")


if __name__ == "__main__":
    main()
