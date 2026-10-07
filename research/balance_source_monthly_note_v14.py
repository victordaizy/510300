"""补充累计贷款与当月边际变化，保留已完成研究的原始结果。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_balance_source_comparability_v14"


def main():
    receipt = OUT / "monthly_note_receipt.json"
    if receipt.exists():
        raise RuntimeError("累计与单月说明已保存，不覆盖。")
    source = OUT / "results/104个月_完整输入与原结果.csv"
    a = pd.read_csv(source).set_index("stat_month")
    rows = []
    for month, previous in [("2020-08", "2019-08"), ("2022-08", "2021-08")]:
        for key, label in [("corporate_long", "企事业中长期"), ("household_long", "住户中长期"), ("corporate_short", "企事业短期"), ("bills", "票据融资")]:
            field = f"loan_{key}_reported_month_or_difference_yi"
            current, prior = float(a.loc[month, field]), float(a.loc[previous, field])
            rows.append({"stat_month": month, "item": label, "monthly_current_yi": current, "monthly_previous_yi": prior,
                         "monthly_yoy_difference_yi": current-prior, "ytd_yoy_difference_yi": float(a.loc[month, f"loan_{key}_ytd_yoy_change_yi"]),
                         "current_source_url": a.loc[month, "source_url"], "previous_source_url": a.loc[previous, "source_url"],
                         "status": "SAVED_ORIGINAL_RELEASE_COMPARISON_NOT_REVISED_COMMON_VINTAGE"})
    frame = pd.DataFrame(rows)
    np.testing.assert_allclose(frame.monthly_current_yi-frame.monthly_previous_yi, frame.monthly_yoy_difference_yi)
    frame.to_csv(OUT / "results/两次八月_累计与当月方向.csv", index=False, encoding="utf-8-sig")
    table = "\n".join(f"| {r.stat_month} | {r.item} | {r.ytd_yoy_difference_yi:+,.0f} | {r.monthly_previous_yi:,.0f} | {r.monthly_current_yi:,.0f} | {r.monthly_yoy_difference_yi:+,.0f} |" for r in frame.itertuples())
    report = "补充_累计偏弱与当月改善可以并存.md"
    text = f"""**2022年8月累计企业中长期少增，与8月单月中长期多增同时存在。** 第十四轮主表使用年初累计口径，不能据它概括当月所有信用的方向。这里只补齐同一原数据中已经保存的当月金额，不改变分组、系数、窗口或研究结果。

单位均为亿元；同比差是两年相同区间的原公告相减，未冒充后来同版本官方同比。

| 月份 | 项目 | 年初累计同比差 | 上年8月单月 | 本年8月单月 | 单月同比差 |
|---|---|---:|---:|---:|---:|
{table}

2022年8月企事业中长期当月7353亿元，相比2021年8月5215亿元多2138亿元；但当年前八个月累计仍少3340亿元。住户中长期当月2658亿元，相比上年4259亿元少1601亿元，累计也少22789亿元。票据融资前八个月累计多增19032亿元，但8月单月同比少增1222亿元。因此“累计新增信用更多落在短贷、票据”和“当月企业中长期出现修复”可以同时成立。

2020年8月则是企业中长期累计与单月都多增，住户中长期累计与单月也都多增。部门、期限、累计水平、最近变化四个维度都需要保留；不能把它们压成“信用强/弱”一个标签。上述事实没有识别贷款最终用途、是否超出当时预期、或随后股票收益的因果来源。

原公告：[2019年8月]({a.loc['2019-08','source_url']})、[2020年8月]({a.loc['2020-08','source_url']})、[2021年8月]({a.loc['2021-08','source_url']})、[2022年8月]({a.loc['2022-08','source_url']})。累计重建和缺失说明沿主报告及V4原来源台账。

本补充在主结果完成后检查，属于完整性说明，不是新的假设检验。未据此改分组、删病例或重新估计；目标仍未完成。
"""
    (OUT / report).write_text(text, encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    record = {"at": datetime.now().astimezone().isoformat(), "status": "PASS_SAVED_MONTHLY_VERSUS_YTD_DIFFERENCES", "source_sha256": sha256(source.read_bytes()).hexdigest(),
              "rows": len(frame), "report": report, "report_sha256": sha256((OUT / report).read_bytes()).hexdigest(),
              "main_result_changed": False, "new_hypothesis_tests": 0, "goal_achieved": False,
              "meaning": "累计偏弱不等于当月各部门均恶化；该补充不改变已保存条件关联结果。"}
    receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"补充": report, "状态": record["status"], "事实行数": len(frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
