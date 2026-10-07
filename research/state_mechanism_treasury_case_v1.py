"""用一个有公开解释的短阶段展示美债分解，不创建交易信号。"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_state_mechanism_expectation_v1"
NOMINAL = ROOT / "reports/research/510300_rmb_residual_state_v1/sources/treasury_2023.xml"


def rows(raw, real=False):
    values = []
    for el in ET.fromstring(raw).iter():
        if el.tag.split("}")[-1] != "properties":
            continue
        d = {c.tag.split("}")[-1]: c.text for c in el}
        name = "TC_10YEAR" if real else "BC_10YEAR"
        if d.get(name):
            v = {"date": d["NEW_DATE"][:10], "real_10y" if real else "nominal_10y": float(d[name])}
            if not real:
                v["nominal_2y"] = float(d["BC_2YEAR"])
            values.append(v)
    return pd.DataFrame(values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources").mkdir(exist_ok=True)
    path = OUT / "sources/treasury_real_2023.xml"
    url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_real_yield_curve&field_tdr_date_value=2023"
    if not path.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
        (path.with_suffix(".receipt.json")).write_text(json.dumps({"at": datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            "url": url, "status": response.status_code, "use": "2023年7至10月分解的事后机制示例，不能当作当时可交易预测。"}, ensure_ascii=False, indent=2), encoding="utf-8")
    data = rows(NOMINAL.read_bytes()).merge(rows(path.read_bytes(), True), on="date", how="inner", validate="one_to_one")
    data["inflation_compensation_10y"] = data.nominal_10y - data.real_10y
    endpoints = data.loc[data.date.isin(["2023-06-30", "2023-07-31", "2023-08-31", "2023-09-29", "2023-10-31"])].copy()
    assert len(endpoints) == 5
    beginning, ending = endpoints.iloc[0], endpoints.iloc[-1]
    changes = {c + "_change_bp": round(float((ending[c] - beginning[c]) * 100), 6)
               for c in ["nominal_10y", "real_10y", "inflation_compensation_10y", "nominal_2y"]}
    assert abs(changes['nominal_10y_change_bp'] - changes['real_10y_change_bp'] - changes['inflation_compensation_10y_change_bp']) < 1e-8
    endpoints.to_csv(OUT / "美债短阶段分解.csv", index=False, encoding="utf-8-sig")
    result = {"case_period": ["2023-06-30", "2023-10-31"], "observed_endpoints": endpoints.to_dict("records"),
        **changes, "source_explanation_publication": "2023-11-16",
        "source_explanation": "纽约联储SOMA负责人当时指出，其跟踪模型平均与调查分解均认为期限溢价占7至10月收益率上升的主要部分；模型存在分歧。",
        "source_url": "https://www.newyorkfed.org/newsevents/speeches/2023/per231116",
        "causal_limits": ["实际收益率上涨不能单独识别增长、政策或真实期限溢价。", "名义减TIPS是通胀补偿，不能直接命名为纯通胀预期。", "两个分解角度不能重复相加。", "11月的解释不能前置到7月成为信号。"],
        "new_strategy_return_tests": 0, "new_accounts": 0, "current_market_forecast": "NOT_MADE",
        "goal_achieved": False}
    (OUT / "treasury_case_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(changes, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
