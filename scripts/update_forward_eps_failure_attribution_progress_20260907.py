"""记录已核对的失败归因和普通中文交付，保持第十九轮原进程。"""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.forward_eps_guosen_history_v1 import identity


def main():
    source = ROOT / "reports/research/510300_forward_eps_corrected_account_attribution_v1_1"
    result = read(source / "result.json")
    checked = read(source / "saved_numerical_verification.json")
    assert checked["status"].startswith("PASS_") and checked["all_original_v1_numerical_components_preserved"]
    delivery = ROOT / "deliverables/510300前瞻EPS失败归因与进出场影响_20260907"
    document = delivery / "前瞻EPS失败归因_预测仓位与退出.md"
    text = document.read_text(encoding="utf-8")
    assert "```" not in text and "![完整账户表现与预测偏差](<" in text
    assert text.count("| 原前瞻EPS三因子 |") == 1
    for phrase in ["714日", "1604", "十八条", "重新进入", "目标保持未完成", "不准备GPT数值审阅包"]:
        assert phrase in text, phrase
    expected_rows = {"账户规模与时变损益分解.csv": 36, "同期间策略增量分解.csv": 28,
        "逐年持仓和损益.csv": 126, "已有预测与兑现结果.csv": 108, "不同市场状态下预测误差.csv": 15,
        "盈利集中程度.csv": 18, "已有账户全部进出场.csv": result["saved_filled_trades"]}
    for filename, expected in expected_rows.items():
        frame = pd.read_csv(delivery / filename)
        assert len(frame) == expected, filename
        assert identity(delivery / filename)["sha256"] == identity(source / filename)["sha256"]
    save(delivery / "visual_and_document_verification.json", {"checked_at": now(),
        "status": "PASS_CHINESE_TEXT_SOURCE_NUMBERS_CSV_IDENTITY_AND_CHART_VISUAL_CHECK",
        "csv_rows": expected_rows, "chart_dimensions": [2240, 992], "chart_manually_viewed": True,
        "target_label_no_longer_overlaps_axis": True, "all_rule_and_image_links_absolute": True,
        "new_strategy_accounts_created": 0, "gpt_review_package_created": False,
        "files": [identity(p) for p in [document, delivery / "完整账户表现与预测偏差.png"]]}, exclusive=True)
    receipt = read(delivery / "delivery_receipt.json")
    receipt["visual_check_pending"] = False
    receipt["verified_at"] = now()
    receipt["verification"] = "visual_and_document_verification.json"
    (delivery / "delivery_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    index_path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    original = index_path.read_text(encoding="utf-8")
    index = json.loads(original)
    assert len(index["completed_rounds"]) == 18 and index["evaluation_accounts_in_this_resumption"] == 438
    assert index["registered_configurations_in_this_resumption"] == 189
    diagnostic = {"study": result["study_id"], "result": source.relative_to(ROOT).as_posix() + "/result.json",
        "status": result["status"], "source_accounts": 18, "account_period_decompositions": 36,
        "paired_decompositions": 28, "saved_forecasts": 108, "new_account_evaluations": 0,
        "saved_verification": checked, "goal_achieved": False}
    index.setdefault("completed_diagnostics", []).append(diagnostic)
    index["updated_at"] = now()
    index["new_evidence"].extend([
        "正确来源已有18账户完成36期间拆分、28组比较和108条预测核对，没有新模型或新账户。费用支付正额与收益负贡献另列，全部旧分解数值保持。",
        "原EPS首次预测2023-08-31，首次可用模型下一交易日2023-09-01；2020至2023实际零持仓。2024单年夏普1.308、2025约1.197，完整0.619；714日诊断0.928不替代1604日验收。",
        "原EPS33个已兑现月末方向正确18、实际上涨19；平均绝对误差8.60个百分点，均方误差0.010481高于当时训练均值基准0.007629。价格加EPS预测更准，但账户少赚18156.6879元。",
        "EPS在判断时位于120日均线上方的20次中预测平均约3.98%、实际0.52%；不高于均线13次预计约0.23%、实际5.35%。标签重叠、样本有限，不宣布新状态切换策略有效。",
        "增加期限与每日趋势退出后比原EPS少87575.85432元，其中额外费用1705.95432元，差距主要来自持仓变化的市场损益；进出场需同时检验。",
        "原EPS首次预测后714日的平均份额损益约47025元、时点变化48618元、费用负1058元，合计94585元；事后诊断量不能当可实施静态账户或独立择时证据。",
    ])
    index["deliveries"].append({"created_at": now(), "type": "CHINESE_FAILURE_ATTRIBUTION_MD_7_CSV_AND_CHART_NO_GPT_PACKAGE",
        "directory": str(delivery), "main_document": str(document), "csv_files": 7,
        "new_account_evaluations": 0, "new_gpt_review_archive_created": False,
        "visual_and_document_check": "PASS_CHINESE_TEXT_SOURCE_NUMBERS_CSV_IDENTITY_AND_CHART_VISUAL_CHECK"})
    index["next_work"].append("第十九轮完成后，同时比较预测误差与真实持仓损益，检查状态相关的预测偏差、早期输入覆盖、退出后再入；不能只按均方误差或单年夏普选赢家。")
    index["latest_corrected_eps_failure_attribution"] = diagnostic
    index["checks"] = "此前来源31项测试及492EPS事实核对保留；本轮18原账户、36期间、28比较、108预测时钟核对通过，七CSV一致及图文目视通过；没有新训练、账户、随机抽样或额外安全审计。"
    with (ROOT / "reports/research/510300_sharpe_1_2_before_failure_attribution_update_20260907.json").open("x", encoding="utf-8") as handle:
        handle.write(original)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    continuation = ROOT / index["latest_continuation_note"]
    addition = """

## 后续已完成：正确来源账户的失败归因

`reports/research/510300_forward_eps_corrected_account_attribution_v1_1` 已完成并核对。读取原十八条账户，形成三十六个完整及首次EPS模型后期间、二十八组比较、一百零八条原预测记录，不新增候选、账户、模型或标签，策略计数保持18轮、438评价账户、189登记方法。第一版诊断的费用支出与负贡献列名冲突已另存 `verification_issue.json`；第一点一版新增正额支付字段，全部原分解数值逐项保持，不能说原交易成本或策略绩效被改写。保存核对通过。

原EPS首次预测2023-08-31、下一交易日2023-09-01，共36次预测，33次已兑现；2020至2023实际账户没有持仓。2024单年夏普约1.308，但完整0.619；首次模型后714日诊断0.928，不能用714日取代1604日验收。原EPS33次方向正确18、实际上涨19，均方误差0.010481高于训练均值基准0.007629。价格加EPS预测误差更小、方向正确20，但完整少赚18156.6879元，必须检查预测如何转为仓位。

判断时位于120日均线上方20次，原EPS平均预测3.98%、实际0.52%；不高于均线13次，预测0.23%、实际5.35%。60日标签重叠，样本小，分组只是诊断；不要按这份已观察结果直接宣布一个新行情切换有效。一月仅三次，不能归因年度切换。

增加期限与每日趋势退出，完整比EPS少87575.85432元，其中额外费用1705.95432元，持仓变化造成的市场损益是主要差距。原EPS714日平均份额损益47025.133053元、时点变化48618.266947元、费用负1058.47866元，合计94584.92134元；都是保存路径诊断，均值由事后整个期间算出，不当实际静态策略或独立择时证明。

中文交付 `deliverables/510300前瞻EPS失败归因与进出场影响_20260907/前瞻EPS失败归因_预测仓位与退出.md`，附七份原数值CSV和图，图文核对已通过。第十九轮47679、2240不受影响，继续按原五步顺序；新模型未运行前不改已登记参数。后续结合新源早期覆盖、预测偏差、仓位和退出后重入共同判断。无新GPT包，目标保持进行中。
"""
    with continuation.open("a", encoding="utf-8") as handle:
        handle.write(addition)
    print("已记录失败归因与七CSV图文交付；模型、账户和第十九轮规则保持不变。")


if __name__ == "__main__":
    main()
