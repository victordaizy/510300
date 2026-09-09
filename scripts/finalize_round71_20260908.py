"""交付条件中位数退出失败归因，保留全部系数与完整账户。"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from research.median_continuation_v1 import ROOT, OUT as RESEARCH, CONFIG, PRIMARY
from research.learned_cycle_exit_v1 import FEATURES, CN
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from scripts.finalize_round60_20260907 import metric, table

INDEX = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
OUT = ROOT / "deliverables/510300中位数继续价值退出_第71轮_20260908"
DOCUMENT = OUT / "中位数继续价值退出_全部因子规则和历史表现.md"
NEXT = ROOT / "docs/510300_AFTER_MEDIAN_DELETED_CYCLE_STABILITY_20260908.md"


def main():
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    require([r["round"] for r in index["completed_rounds"]] == list(range(1, 71)), "索引不是截至70轮，不能重复收尾")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "中位数冻结文件改变")
    result = json.loads((RESEARCH / "result.json").read_text(encoding="utf-8"))
    receipt = json.loads((RESEARCH / "saved_verification_receipt.json").read_text(encoding="utf-8"))
    require(not result["historical_point_target_met"], "不能按失败关闭已经出现1.2的候选")
    status = "COMPLETED_REJECTED_MEDIAN_EXIT_MAIN_DETERIORATION"
    write_json(RESEARCH / "acceptance_outcome.json", {"recorded_at": now(), "status": status, "goal_achieved": False,
        "decision": "主评价基础夏普由原0.704868降至0.470093；较早仅0.747698升至0.756780，且两段两费用终值均低于原模型。中位数让实际持有更短并丢掉部分大盈利，结束此损失函数，不扫描分位点、惩罚和确认次数。",
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# 510300中位数继续价值退出：第71轮", "", "## 结果与决定", "",
        "**本轮没有达到夏普1.2，结束这一改动。** 主评价基础净夏普从原平均收益模型的0.705降至0.470，压力从0.641降至0.410；较早基础仅从0.748升至0.757，压力从0.725升至0.733。新方法在两段、两档费用下的最终资产都低于原方法，不能仅凭较早夏普略高就称为改善。", "",
        "这次没有补EPS、财报、公募或行情。复用相同的已结束持仓样本、八个因子及训练时点，只改为估计继续持有收益的条件中位数。完成114次真实月度拟合，均成功；27个较早时点因成熟样本不足没有模型。旧参考账户、旧平均收益模型和对照账户全部复用。", "",
        "## 用中文说明策略怎样做", "",
        "进入继续使用原日内相对隔夜强弱信号：近六十日日内表现明显强于隔夜，连续两天满足条件，才在下一开盘买入。持有后，模型用当时已结束的历史交易判断继续持有通常是否合算；中位数连续两天为负，就在下一开盘退出。原固定止损、回撤保护、价格转弱和最长持有期仍有效。卖出后要等旧进入条件先消失、再重新出现，才能再买。", "",
        "中位数与平均数估计的内容不同。中位数为正不表示平均盈利必然为正，也不保证更高夏普；本轮也移除了原岭回归的系数惩罚，以无惩罚绝对误差求解，不能将效果说成只改了一个原模型系数。", "",
        "## 两段历史的完整表现", "", "### 主评价：2020年1月2日至2026年8月14日开盘", ""]
    lines += table(result["all_metrics"])
    lines += ["### 较早诊断：2015年1月5日至2019年12月31日开盘", ""] + table(result["earlier_diagnostics"])
    lines += ["两段各自从20万元开始，完整纳入1604日和1219日，包括全部空仓日和终点。基础净累计收益分别19.56%和50.56%，压力为16.42%和48.45%。历史已经被多轮研究使用，不构成新的独立样本外证据。", "",
        "## 改动为何没有带来跨时期改善", "",
        "主评价基础新账户少赚44,015.53元：价格盈亏比原模型少42,796.10元，分红多633.80元，佣金和滑点合计多1,853.23元。主要差异来自持仓路径和价格收益，并非只有手续费。较早基础少赚2,452.01元，年化从8.64%降至8.46%；其夏普略高主要伴随波动下降，不能代替收益改善。", "",
        "主评价从原24个周期、252个持仓收盘，变为30个周期、130个持仓收盘；平均仅持有4.33个收盘，30个周期都由学习条件触发退出。原24个进入原点全部仍出现，另外新增6个进入原点。提前卖出让账户更早空仓，也改变后续可以进入的机会。较早9个进入原点完全相同，持仓收盘由299减到255，学习退出由原2个周期变为4个。", "",
        "在新账户实际走到的同一持仓状态上比较两个保存模型：主基础130个有效预测中，有52个状态是中位数为负、原平均预测不为负，反方向仅2个；较早51个预测中分别为5个和0个。这是同一输入上的事后诊断，支持本轮退出更积极的观察，不能当作一套额外交易策略。", "",
        "相同进入原点既有提前退出获益，也有提前退出损失，完整匹配表全部保留。例如：", "",
        "|进入决定日|新方法卖出日|原方法卖出日|新周期净收益率|原周期净收益率|观察|", "|---|---|---|---:|---:|---|"]
    matches = pd.read_csv(RESEARCH / "saved_entry_matched_cycle_comparison.csv")
    main = matches[matches.period.eq("evaluation") & matches.cost.eq("BASE")]
    for origin, reason in [("2024-09-24", "过早退出，少获得随后上涨"), ("2026-06-04", "提前止于亏损，原持仓后来转盈"), ("2021-02-02", "较早退出避开随后下跌")]:
        r = main[main.entry_origin.eq(origin)].iloc[0]
        lines.append(f"|{origin}|{r.exit_date_new}|{r.exit_date_old}|{r.cycle_net_return_new:.2%}|{r.cycle_net_return_old:.2%}|{reason}|")
    lines += ["", "周期收益率以各自含买入佣金的实际成本为分母，包含归属该持仓的分红权益。不能把个别周期收益率差直接换成整个账户的因果损失，也不能依据上表日期反向选交易。", "",
        "## 全部周期经济结果", "", "|历史|费用|周期数|盈利／亏损周期|价格损益（元）|分红（元）|佣金（元）|滑点（元）|净损益（元）|持仓收盘数|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pd.read_csv(RESEARCH / "saved_cycle_profit_groups.csv").itertuples():
        lines.append(f"|{'主评价' if r.period=='evaluation' else '较早诊断'}|{'基础' if r.cost=='BASE' else '压力'}|{r.cycles}|{r.positive_cycles}／{r.negative_cycles}|{r.gross_price_profit:,.2f}|{r.dividend_recognized:,.2f}|{r.commission:,.2f}|{r.slippage:,.2f}|{r.net_profit:,.2f}|{r.held_closes}|")
    lines += ["", "压力主评价持仓125个收盘，比基础少5个。两档费用共享同一批训练模型，但实际买入成本影响浮盈浮亏及回撤因子，所以退出时点可以不同；压力回撤略小并不说明更高费用有利。两段两费用均无整笔未成交请求。", "",
        "## 基础费用逐年表现", "", "|历史|年份|该年净收益|净夏普|年内最大回撤|成交笔数|", "|---|---|---:|---:|---:|---:|"]
    years = pd.read_csv(RESEARCH / "yearly_metrics.csv")
    for r in years[years.model.eq(PRIMARY) & years.cost.eq("BASE")].itertuples():
        sharpe = f"{r.net_sharpe:.3f}" if pd.notna(r.net_sharpe) else "未定义"
        lines.append(f"|{'主评价' if r.period=='evaluation' else '较早诊断'}|{r.year}|{r.cumulative_return:.2%}|{sharpe}|{-r.max_drawdown:.2%}|{r.trade_count}|")
    lines += ["", "2023年全年空仓、收益波动为零，夏普未定义，不能写成0或从完整评价中删掉。2026年只到固定终点；单年超过1.2不代表整体达到目标。年内回撤从该年自身起点计算，完整账户回撤以此前完整高点计算。", "",
        "## 全部因子与完整中文进出场规则", ""]
    protocol = (ROOT / cfg["rules"]).read_text(encoding="utf-8").splitlines()
    lines += [line.replace("## ", "### ", 1) if line.startswith("## ") else line for line in protocol[1:]]
    lines += ["", "## 交付和核对", "",
        "同目录的“每月中位数退出模型中文规则.md”列出全部114次成功模型的八个均值、标准差、系数及截距，并逐次说明无模型月份。四份“逐日全部因子和进出场.csv”保存真实持仓因子、模型日期、预测、连续计数、请求和退出原因；另有四份完整账户、成交和含分红周期归因。", "",
        "10项关键测试一次通过。已核对141个训练时点、90,217条带月度归属的训练成员记录、3456日原进入因子、20项账户指标、16组账户经济差额、5646个进入退出请求、357个有效持仓预测和78个含分红周期。357个预测复算只存在浮点精度范围内的差异。较早两档费用各204个无模型持仓日，合计408个，预测保留缺失且继续原价格退出。以上核对没有重新拟合或新模拟账户，也不代表策略有效性通过。", "",
        "## 下一项", "",
        "下一项回到原平均收益模型，检验退出决定是否过度依赖某一个已结束交易周期：每次训练时依次拿掉一个完整旧周期，再估计其余周期上的模型；只有原模型与这些模型都认为继续持有收益为负，才增加学习退出。原价格退出始终保留。这是一个尚待登记的样本稳定性规则，不是已经验证有效，也不是调本轮中位数的分位点。", "",
        "它可能减少脆弱的提前退出，也可能延迟必要的止损，下一轮仍直接检验完整账户。只用现有成熟样本，不增加数据补齐任务。", ""]
    DOCUMENT.write_text("\n".join(lines), encoding="utf-8")
    files = ["metrics.csv", "earlier_diagnostics.csv", "yearly_metrics.csv", "era_metrics.csv", "model_coverage.csv", "result.json", "training_receipts.csv",
        "每月中位数退出模型中文规则.md", "saved_models.json", "saved_all_entry_factors.csv", "saved_training_support_replay.csv", "saved_account_metrics.csv",
        "saved_account_differences.csv", "saved_actual_cycles.csv", "saved_cycle_profit_groups.csv", "saved_prediction_comparison.csv", "saved_entry_matched_cycle_comparison.csv",
        "saved_verification_receipt.json", "acceptance_outcome.json", "tests_receipt.json"]
    for name in files:
        shutil.copy2(RESEARCH / name, OUT / name)
    shutil.copy2(CONFIG, OUT / "冻结设置.json")
    factors = pd.read_csv(RESEARCH / "saved_all_entry_factors.csv", parse_dates=["date"])
    factor_fields = factors[["date", "d60_factor", "entry_condition", "original_price_exit"]]
    for period, label in [("evaluation", "主评价"), ("earlier_diagnostic", "较早历史")]:
        for cost, cost_label in [("BASE", "基础费用"), ("STRESS", "压力费用")]:
            folder = RESEARCH / period / cost
            ledger = pd.read_parquet(folder / f"{PRIMARY}_ledger.parquet")
            ledger.to_csv(OUT / f"{label}_{cost_label}_完整账户.csv", index=False, encoding="utf-8-sig")
            decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
            decisions = decisions.merge(factor_fields.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            decisions.rename(columns={"origin": "收盘决定日", "execution_date": "实际计划执行日", "action": "中文动作", "requested_quantity": "请求份额",
                "exit_reasons": "退出原因", "entry_rearmed": "再次进入资格", "continuation_prediction": "继续价值中位数预测", "learning_fit_origin": "模型训练日",
                "negative_confirmation_count": "连续负值次数", "learned_exit_requested": "学习退出请求", "learning_status": "模型状态",
                "d60_factor": "日内相对隔夜六十日强弱", "entry_condition": "原进入条件", "original_price_exit": "原价格退出", **dict(zip(FEATURES, CN))}).to_csv(
                OUT / f"{label}_{cost_label}_逐日全部因子和进出场.csv", index=False, encoding="utf-8-sig")
            shutil.copy2(folder / f"{PRIMARY}_trades.csv", OUT / f"{label}_{cost_label}_全部成交.csv")
    NEXT.write_text("""# 第71轮后：逐个删除已结束周期的退出稳定性

第71轮510300_MEDIAN_CONTINUATION_V1完成且拒绝，成果在reports/research/510300_median_continuation_v1；中文交付deliverables/510300中位数继续价值退出_第71轮_20260908。原始helper research/median_continuation_inputs_v1.py、runner research/median_continuation_v1.py、tests/test_median_continuation_v1.py、config/510300_median_continuation_v1.json、docs/510300_MEDIAN_CONTINUATION_V1.md。10测试3.67秒一次通过，冻结和运行均一次完成、无修正版；run的exec session93134已经确认exit0，savedreview session2157也exit0，不等待或重启这些旧句柄。

固定条件中位数q0.5、alpha0、solver highs，无惩罚绝对误差，8原因子、周期等权、训练均值/标准差、clip±5，原月度成熟10周期100行及最近20周期。141时点114拟合全成功、27旧早期样本不足；无新参考账户，无新数据下载。四新账户、16旧对照。主BASE/STRESS净夏普0.470092714487/0.410009181994，基础年化2.7318408%、回撤10.1776558%、累计19.5590250%；早0.756780212891/0.732920137729，基础年化8.4618229%、回撤13.7916444%、累计50.5554383%。主原R32 0.704868/0.641136，早0.747698/0.724697。所有新终值都低于原R32，结束中位数，不改分位数/惩罚/窗口/确认次数救回。

主30周期60成交，基础15盈15亏，130持仓收盘、平均4.333，压力125；原24周期48成交252收盘。新原有24进入原点全部相同，另外新增6进入；30周期全由学习条件退出，0未成交，433原点等旧进入条件消失。早9周期18成交255持仓、6盈3亏，原9周期299收盘，9进入原点全同；4学习退出而原2，418等待旧条件原点。主有效预测130/125，早51/51另各204无模型；压力成本改变实际输入，主少5持仓日不是核算错误。独立357预测误差1.11e-16，408无模型预测保留缺失；同新实际持仓输入比较原均值，主BASE中位数负原均值非负52次，反向2次；压力50/2；早各5/0。

主基础价格40864.3+分红9671.4−佣金2883.85002−滑点8533.8=净39118.04998；比原少44015.52572，价格少42796.1、分红多633.8、费用多1853.22572。压力净32847.90124，比原少40982.74948。早基础价格102799.1+分红1939−佣金927.0235−滑点2700.2=净101110.8765，比原少2452.01488；压力净96896.40884，比原少2427.15232。早夏普略升不能掩盖年化与终值更低。例2024-09-24进入原点，新9/27卖5.8872%，原9/30卖14.6491%；2026-06-04新6/9卖负3.1129%、原6/23卖正3.2974%；2021-02-02新2/24卖1.6157%、原2/26卖负2.0050%。这些是保存周期归因，不可据日期调整策略。主2023全年空仓，夏普未定义，不填0或删日。

scripts/review_round71_saved.py已完成原始分红日线进入/价格退出和四价格模型因子、141时点90217训练成员、20指标16差额5646全部份额请求357预测78分红周期复算，无新拟合或账户。实际持仓8因子从真实买入金额、持仓现金流和逐日价格独立重建。finalize_round71_20260908.py完成后勿重跑。索引截至71应345不同设置、359已评价来源版本、364登记含5旧未运行、1132主评价记录。此目标回合71完成属于PROGRESS，目标未实现不标记complete或blocked。

下一72仅有方向与旧同类方法辨析，未代码/协议/登记/拟合/回测。检验原均值模型的逐周期删除稳定性，沿用原R32的进入、再次进入、原止损/回撤/最长持有和两次负值确认，不用第71中位数。

拟一个新策略：原全训练样本模型预测为负，同时删除任一单个已结束参考周期后重新拟合的全部模型也预测为负，才将本日计为学习退出负值；连续两天才请求下一开盘退出。原已冻Ridge模型复用，不重拟合全样本；删周期模型重新按余下周期等权、训练均值标准差、clip±5、原岭惩罚1、svd拟合截距，所有删法全列，不挑最好子样本。相等、任何一个非负、模型/资料缺失则学习计数归零；已经形成退出请求仍保持到卖出。每个真实账户用自身因子，不共享持仓路径。

训练限定原141月度时点及原最近20完整周期窗口，原月度10周期100行最低支持不变。删一个周期的扰动子集会剩9至19周期；这是同一已合格原样本的删除稳定性检查，需要在下一协议明确说明，不偷偷提高/降低主样本成熟门槛或追加历史周期。子模型不独立重做一个成熟窗口、不补入第21周期；删除后只保留有限非空原记录，子模型失败则整月没有稳健学习判断，原价格退出继续。所有子模型必须只用拟合日已经自然结束的周期。先精确计预定子拟合数、不读新模型预测/收益；必要分组删除、成熟时间、原模型保留、全部负值、计数/受阻退出/实际成交测试通过后再冻结。

这是样本影响稳定性规则，不是置信区间、独立样本外证明或另一历史评分门，不阻止完整账户运行。不选择删除哪段行情来抬高结果，不删除评价日期。失败直接换方向，不把全部同意改成多数或调整百分位/阈值继续救回。不会给中位数再加规则，是回到原均值模型的一项明确新机制。

查到两个旧同类词：research/regime_transition_router_v1.py约1231的leave_one_cycle_out_gate是旧状态机制结构方向检查，research/liquidity_stress_long_cycle_replication_v1.py:746是删除事先指定压力周期的评价收益日再重算整体指标。本候选在当时成熟训练集中按真实自然交易周期删除、重拟合并产生之后的每日退出条件，保留全部评价日；不重启这两个旧研究。方法分组定义参考 https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html 已查，文档只支持按组留出；不把来源当本规则盈利或稳健保证。

本机sklearn1.9.0、scipy1.18.0已确认。不补EPS、财报、估值、公募及其他慢源，不新下载行情，不建立新参考账户，不做GPT数值包，不创建子任务，交易范围仍只510300.SH和CASH_CNY；v6允许新历史方法，完整账户口径与两窗口费用全部保留。
""", encoding="utf-8")
    record = {"round": 71, "study": result["study_id"], "title": "条件中位数继续价值学习退出", "status": status,
        "result": str((RESEARCH / "result.json").relative_to(ROOT)), **{k: result[k] for k in ["candidate_configurations", "evaluation_accounts", "new_accounts_generated", "reused_control_accounts",
        "earlier_diagnostic_accounts", "new_earlier_diagnostic_accounts", "new_model_fits", "new_reference_accounts"]}, "evaluated_candidate_source_runs": 1,
        "primary_base": metric(result, PRIMARY), "primary_stress": metric(result, PRIMARY, cost="STRESS"), "post_selected_best_base": metric(result, PRIMARY)}
    index["completed_rounds"].append(record)
    for key in ["registered_configurations_in_this_resumption", "evaluated_configurations_in_this_resumption", "evaluated_candidate_source_runs_including_corrected_replays", "registered_candidate_source_runs_including_unrun_legacy_bindings"]:
        index[key] += 1
    index["evaluation_accounts_in_this_resumption"] += result["evaluation_accounts"]
    index.update(updated_at=now(), status="ROUND71_COMPLETE_MEDIAN_CONTINUATION_REJECTED", running_studies=[], goal_achieved=False, latest_completed_round=record,
        count_warning="累计71轮，345不同设置，359已评价来源版本，364登记含5旧未运行，1132主评价记录。第71轮114拟合、27原样本不足时点另计，不算114个策略。",
        checks="第71轮10测试、141训练时点90217训练成员、20指标16差额5646请求357预测及78含分红周期复算完成。",
        process_state_note="第71轮训练与完整账户、失败归因和中文交付完成；第72轮只有逐周期删除稳定性方向，尚未登记或拟合。",
        next_work={"status": "DELETED_CYCLE_EXIT_STABILITY_DIRECTION_NOT_REGISTERED", "focus": "原均值模型的学习退出是否对删除任一成熟周期保持负向，单一完整策略检验", "source": str(NEXT.relative_to(ROOT))},
        latest_saved_median_continuation_diagnostic=str((RESEARCH / "saved_verification_receipt.json").relative_to(ROOT)))
    index["deliveries"].append({"created_at": now(), "type": "MEDIAN_CONTINUATION_ROUND71_CHINESE_RESULTS", "rounds": [71], "directory": str(OUT), "main_document": str(DOCUMENT), "new_gpt_review_archive_created": False})
    write_json(INDEX, index)
    delivery = {"created_at": now(), "status": "MEDIAN_CONTINUATION_RULES_MODELS_AND_FULL_ACCOUNTS_DELIVERED", "main_document": str(DOCUMENT),
        "document_characters": len(DOCUMENT.read_text(encoding="utf-8")), "ordinary_files_excluding_this_receipt": len(list(OUT.iterdir())),
        "new_gpt_review_archive_created": False, "security_audit_performed": False, "goal_achieved": False, "next_round_registered": False}
    write_json(OUT / "交付回执.json", delivery, exclusive=True)
    print(json.dumps({"交付": delivery, "最新状态": index["status"], "核对": receipt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
