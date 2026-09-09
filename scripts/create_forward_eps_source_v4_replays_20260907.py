"""建立完整来源重放文件，原运行函数只替换来源与输出标识。"""
from pathlib import Path
import ast
import json

ROOT = Path(__file__).resolve().parents[1]
REPLACEMENTS = {
    "forward_eps_two_institution_features_v2": "forward_eps_two_institution_features_v3",
    "FORWARD_EPS_TWO_INSTITUTION_FEATURES_V2": "FORWARD_EPS_TWO_INSTITUTION_FEATURES_V3",
    "forward_eps_two_institution_policy_v2": "forward_eps_two_institution_policy_v3",
    "FORWARD_EPS_TWO_INSTITUTION_POLICY_V2": "FORWARD_EPS_TWO_INSTITUTION_POLICY_V3",
    "forward_eps_valuation_consistency_features_v1": "forward_eps_valuation_consistency_features_v2",
    "FORWARD_EPS_VALUATION_CONSISTENCY_FEATURES_V1": "FORWARD_EPS_VALUATION_CONSISTENCY_FEATURES_V2",
    "forward_eps_valuation_consistency_policy_v1": "forward_eps_valuation_consistency_policy_v2",
    "FORWARD_EPS_VALUATION_CONSISTENCY_POLICY_V1": "FORWARD_EPS_VALUATION_CONSISTENCY_POLICY_V2",
}


def replace_identifiers(text):
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def write_exclusive(path: Path, text: str):
    compile(text, str(path), "exec") if path.suffix == ".py" else None
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main():
    receipts = []
    for old, new, kind, is_feature in [
        ("forward_eps_two_institution_features_v2", "forward_eps_two_institution_features_v3", "two", True),
        ("forward_eps_valuation_consistency_features_v1", "forward_eps_valuation_consistency_features_v2", "pe", True),
        ("forward_eps_two_institution_policy_v2", "forward_eps_two_institution_policy_v3", "two", False),
        ("forward_eps_valuation_consistency_policy_v1", "forward_eps_valuation_consistency_policy_v2", "pe", False),
    ]:
        original = (ROOT / "research" / (old + ".py")).read_text(encoding="utf-8")
        expected = replace_identifiers(original)
        text = expected
        if new == "forward_eps_two_institution_features_v3":
            text = text.replace('"facts": "510300_forward_eps_soochow_facts_v3"', '"facts": "510300_forward_eps_soochow_facts_v4"')
            text = text.replace('verified["status"] != "PASS_SAVED_SOURCE_ROWS_CLOCKS_AND_PREVIOUS_FACTS" or verified["first_250_only"]',
                                'verified["status"] != "PASS_ALL_V3_FACTS_PRESERVED_NEW_RAW_ROWS_YEARS_AND_CLOCKS"')
            text = text.replace("东吴完整第三版事实尚未完成保存核对", "东吴完整第四版事实尚未完成保存核对")
        start, end = text.index("def freeze("), text.index("def run(")
        call = f'freeze_feature("{kind}", __file__)' if is_feature else f'freeze_policy("{kind}", __file__, MODELS)'
        text = text[:start] + "def freeze() -> None:\n    from research.forward_eps_source_v4_replay_registration import freeze_feature, freeze_policy\n    " + call + "\n\n\n" + text[end:]
        before_run = next(n for n in ast.parse(expected).body if isinstance(n, ast.FunctionDef) and n.name == "run")
        after_run = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "run")
        assert ast.dump(before_run, include_attributes=False) == ast.dump(after_run, include_attributes=False), "来源重放不得修改运行算法"
        target = ROOT / "research" / (new + ".py")
        write_exclusive(target, text)
        receipts.append({"old": old, "new": new, "run_function_unchanged_except_identifiers": True})
    for old, new in [
        ("tests/test_forward_eps_two_institution_features_v2.py", "tests/test_forward_eps_two_institution_features_v3.py"),
        ("tests/test_forward_eps_valuation_consistency_features_v1.py", "tests/test_forward_eps_valuation_consistency_features_v2.py"),
        ("scripts/verify_forward_eps_two_institution_saved_20260907.py", "scripts/verify_forward_eps_two_institution_source_v4_saved_20260907.py"),
        ("scripts/verify_forward_eps_valuation_consistency_saved_20260907.py", "scripts/verify_forward_eps_valuation_consistency_source_v4_saved_20260907.py"),
    ]:
        write_exclusive(ROOT / new, replace_identifiers((ROOT / old).read_text(encoding="utf-8")))
    doc = """# 第二十二轮：修正原件字段后，完整重算原六种方法

2026年9月7日登记。前十九、二十轮旧来源结果均已观察，全部未达夏普1.2。本轮只更正原件字段遗漏，不改变六种方法的因子、模型、训练、进出场、费用、全期评价和统计规则。新增的是六个来源版本，新增独立方法为零；两组各六条新账户加四条保存对照，总计十二条新账户、二十条评价账户。

东吴固定6734份原件已全部保存，第四版在原4490份成功报告全部原样保留的基础上，新提取1245份，合计5735份报告、17194条年度预测。按目录实际发布日期，2022年由111增至606份，2023年由零增至698份，2024年由673增至725份。2023年目录为863份；此前按报告编号开头计为864份，编号年份不能代替实际发布日期。本版仍有999份不能使用，主要是日期不明确，不能声称所有缺口已解决。

新字段为原文明确的“每股收益-最新股本摊薄（元/股）”。保留其股本定义，只有预测年度可用；报告PE原值保留，数学关系条件仍沿用。这里只完成新增字段覆盖，不等于当前价格、当前股份或历史不可变版本全部证明。

两机构组沿用十九轮的三因子、东吴单机构和六因子模型；PE组沿用二十轮的关系相容PE、共同月份原PE和完全去PE模型。新增数据会改变符合原门槛的月份、训练样本和预测，必须如实报告，不能事后把月份删回旧版本。

以下保留完整中文规则。新来源收益尚未读取，结果必须另存并与旧版本逐项比较。没有GPT数值审阅包。

"""
    old_s = (ROOT / "docs/510300_FORWARD_EPS_TWO_INSTITUTION_POLICY_V2.md").read_text(encoding="utf-8")
    old_s = old_s[old_s.index("## 来源及三种固定方案"):old_s.index("## 账户运行前更正来源版本")]
    old_q = (ROOT / "docs/510300_FORWARD_EPS_VALUATION_CONSISTENCY_POLICY_V1.md").read_text(encoding="utf-8")
    old_q = old_q[old_q.index("## 三种策略分别使用哪些因子"):]
    old_q = old_q.replace("登记时本轮尚未训练、尚无历史成绩，目标仍未完成。", "原版本结果已经保存；本次来源重放登记时尚无新结果，目标仍未完成。")
    write_exclusive(ROOT / "docs/510300_FORWARD_EPS_SOURCE_V4_SIX_METHOD_REPLAY.md", doc + old_s + "\n" + old_q)
    receipt_path = ROOT / "reports/research/510300_forward_eps_soochow_latest_share_field_diagnostic_v1/replay_code_generation.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump({"bindings": receipts, "new_distinct_methods": 0, "new_source_versions": 6,
                   "same_run_algorithm_ast_checked": True}, handle, ensure_ascii=False, indent=2)
    print("六种同方法来源重放代码和全部中文进出场规则已建立。", flush=True)


if __name__ == "__main__":
    main()
