"""保留首次源比对程序，仅修正不同实验的既定日期范围校验。"""
from pathlib import Path
import ast

ROOT = Path.cwd()
OUT = ROOT / "reports/research/510300_point_relative_price_optional_correction_v1"
DEST = ROOT / "research/point_relative_price_optional_correction_v1_0_1.py"
HELPER = '''def compare_saved_source(daily, saved, scope_end):
    """共同范围字段严格比对，另一个实验的主动屏蔽保持原样。"""
    overlap = daily.set_index("date").loc[pd.DatetimeIndex(saved.date)]
    require(np.array_equal(overlap.original_510300_close.to_numpy(float), saved.original_510300_close.to_numpy(float)),
            "另一用途与原技术线510300原价不一致，不拼接来源。")
    np.testing.assert_allclose(overlap.original_510500_close.to_numpy(float), saved.original_510500_close.to_numpy(float),
                               rtol=0, atol=0, equal_nan=True)
    columns = ["510300相对510500原价格五日变化差", "510300相对510500原价格二十日变化差"]
    common = pd.DatetimeIndex(saved.date) <= pd.Timestamp(scope_end)
    require(saved.loc[~common, columns].isna().all().all(), "原E70范围外字段主动屏蔽发生变化。")
    np.testing.assert_allclose(overlap.loc[common, FIELDS].to_numpy(float), saved.loc[common, columns].to_numpy(float),
                               rtol=0, atol=1e-13, equal_nan=True)
    return int(common.sum())

'''


def main():
    if DEST.exists():
        raise RuntimeError("完整修正版已存在，不覆盖。")
    original = (ROOT / "research/point_relative_price_optional_correction_v1.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(original)
    load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_fields")
    old_load = ast.get_source_segment(original, load)
    begin = old_load.index('    overlap = daily.set_index("date")')
    end = old_load.index('    summary = {', begin)
    replacement = '''    other_scope_end = read(OTHER / "source_result.json")["last_used_date"]
    common_rows = compare_saved_source(daily, other, other_scope_end)
'''
    new_load = old_load[:begin] + replacement + old_load[end:]
    anchor = '"same_fixed_E70_fields_overlap_rows": len(other),'
    new_load = new_load.replace(anchor, '"same_fixed_E70_fields_overlap_rows": common_rows,\n'
                               '               "E70_field_comparison_scope_end": other_scope_end,\n'
                               '               "E70_outside_scope_mask_preserved_rows": len(other)-common_rows,')
    code = original.replace(old_load, HELPER + new_load)
    replacements = {
        "research/point_relative_price_optional_correction_v1.py": "research/point_relative_price_optional_correction_v1_0_1.py",
        "tests/test_point_relative_price_optional_correction_v1.py": "tests/test_point_relative_price_optional_correction_v1_0_1.py",
        'OUT / "build_isolated_module.py"': 'OUT / "build_isolated_module_v1_0_1.py"',
        'OUT / "tests_receipt.json"': 'OUT / "tests_receipt_v1_0_1.json"',
        'OUT / "source_protocol.json"': 'OUT / "source_protocol_v1_0_1.json"',
    }
    for old, new in replacements.items():
        code = code.replace(old, new)
    anchor = '    return sorted(set(own))'
    parents = '''    own += [ROOT / "research/point_relative_price_optional_correction_v1.py",
            ROOT / "tests/test_point_relative_price_optional_correction_v1.py",
            OUT / "build_isolated_module.py", OUT / "tests_receipt.json", OUT / "source_protocol.json",
            OUT / "initial_source_scope_failure_and_repair_registration.json"]
'''
    if code.count(anchor) != 1:
        raise RuntimeError("来源清单插入位置不唯一。")
    code = code.replace(anchor, parents + anchor)
    ast.parse(code)
    with DEST.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(code)
    print("完整源范围修正版已生成；原程序、原协议和原数据均保持，尚无金融拟合。")


if __name__ == "__main__":
    main()
