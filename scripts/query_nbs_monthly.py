"""从本地国家统计局月度数据库搜索指标、查看原始历史或导出CSV。"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "nbs_monthly"


def main():
    parser = argparse.ArgumentParser(description="查询全国月度数据，不联网。")
    parser.add_argument("--run-id", help="不填则读取最近完成的快照")
    parser.add_argument("--keyword", help="按指标名称检索，如新订单、汽车、货币")
    parser.add_argument("--category", help="限制官方类别，如工业、金融、价格指数")
    parser.add_argument("--indicator-id", help="按指标ID查看完整历史，不混合不同口径")
    parser.add_argument("--start", default="190001")
    parser.add_argument("--end", default="999912")
    parser.add_argument("--output", help="导出结果的CSV路径")
    args = parser.parse_args()
    run = BASE / "runs" / args.run_id if args.run_id else Path(json.loads((BASE / "LATEST.json").read_text(encoding="utf-8"))["path"])
    connection = sqlite3.connect(run / "国家统计局月度.sqlite")
    if args.indicator_id:
        query = "SELECT period AS 月份,name AS 指标,unit AS 单位,period_basis AS 口径,value AS 数值,value_text AS 原始数值,status AS 状态,source_cell_present AS 是否返回单元格,raw_path AS 原始响应 FROM monthly_with_definitions WHERE indicator_id=? AND period>=? AND period<=? ORDER BY period"
        frame = pd.read_sql_query(query, connection, params=[args.indicator_id, args.start, args.end])
    else:
        query = "SELECT i.indicator_id AS 指标ID,i.category AS 类别,i.name AS 指标,i.unit AS 单位,i.period_basis AS 口径,i.catalogue_path AS 目录,c.first_value_month AS 最早月份,c.last_value_month AS 最新月份,c.value_count AS 有值月数,c.status AS 采集状态 FROM indicators i LEFT JOIN coverage c USING(indicator_id) WHERE 1=1"
        params = []
        if args.keyword:
            query += " AND i.name LIKE ?"
            params.append("%" + args.keyword + "%")
        if args.category:
            query += " AND i.category=?"
            params.append(args.category)
        query += " ORDER BY i.category,i.catalogue_path,i.name"
        frame = pd.read_sql_query(query, connection, params=params)
    connection.close()
    if args.output:
        target = Path(args.output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        print(f"已导出 {len(frame)} 行：{target}")
    else:
        print(frame.head(80).to_string(index=False))
        if len(frame) > 80:
            print(f"共 {len(frame)} 行，终端展示前80行；指定 --output 可导出全部。")


if __name__ == "__main__":
    main()
