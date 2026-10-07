"""比较全国月度数据的全部历史、多年同月、官方同比和真实环比。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "nbs_monthly"
SOURCE = "https://data.stats.gov.cn/dg/website/page.html#/pc/national/monthData"

# 每个系列保留原有指标身份；表中的同比和环比不会从同比读数相除得到。
MAIN = [
    ("工业增加值", "当月实际同比", "ef1b1765960d45a29b4d7c4ca91be916", 0),
    ("工业增加值", "累计实际同比", "21e7072e9f384209aedb56e69a18216e", 0),
    ("服务业生产", "当月实际同比", "3fc5439f03ec46a5a76e8032036e8c17", 0),
    ("服务业生产", "累计实际同比", "6efd45f6154649adb4f0176f016d3a36", 0),
    ("社会消费品零售", "当月名义同比", "aaac57d54d2e465d91bc9f3ea1a8618e", 0),
    ("社会消费品零售", "累计名义同比", "e3ca151b53d347b78d1e179e5ebf1d33", 0),
    ("固定资产投资", "累计名义同比", "7e570cf8071c4734a7d78d9f0a70fbe1", 0),
    ("民间固定资产投资", "累计名义同比", "287f55a9520d43a9b3e045860a49fedd", 0),
    ("房地产开发投资", "累计名义同比", "205e08cba8c2409980db58c98da91b6f", 0),
    ("新建商品房销售面积", "累计同比", "50a37fbef1d04be68f15d82b711783bf", 0),
    ("新建商品房销售额", "累计名义同比", "4617f4bb61234af9ab075b3695a1f0bf", 0),
    ("工业企业利润", "累计可比同比", "7f30595d047445e0a3880ab928795892", 0),
    ("工业企业营业收入", "累计可比同比", "84117bfe6f1b4a299d0db210deb5ad36", 0),
    ("工业应收账款", "期末可比同比", "01d4fd4cd8ec4be5bf836d89cb2ebe0e", 0),
    ("工业产成品库存", "期末可比同比", "e00c23ba6ffc45ae86757ce81f1c1997", 0),
    ("PPI", "当月同比", "150633e52b9a470a9a9fd1b296dd6c5b", 100),
    ("PPI", "当月环比，未季调", "e64079bae9064aebad1c4c5fe0c8a6ef", 100),
    ("全国城镇调查失业率", "水平", "3888eac6062945a79c8a27e5f13d4953", 0),
    ("企业就业人员周平均工时", "小时", "40ab91b1ef4948e89633c5c7f55b9713", 0),
    ("M2", "期末同比", "e03f2232631f41cd9d754a7d7feb4a81", 0),
    ("M1", "期末同比，2025年起新口径", "640401d3351b4b868dea28f89f410a54", 0),
    ("M0", "期末同比", "db7891fb8f3c4eb2a4d71a9955eba8c7", 0),
    ("货物出口", "当月名义同比，美元", "788f44b0f310403fbd308b77d6f83890", 0),
    ("货物进口", "当月名义同比，美元", "cc1ac699bc4f4cd7aa5c2b7a1e643259", 0),
    ("货物进出口", "当月名义同比，美元", "5143e29f77ee4d3489eaf46b901ba610", 0),
    ("货运量", "当月同比", "3026a1f82c064d038062f42dba5409b0", 0),
    ("邮政行业业务收入", "当月名义同比", "b400e08c5eb1475594beba116f7c601c", 0),
    ("电信业务收入", "当月名义同比", "9e655be792d6479e99bb0ba6114bfdcb", 0),
    ("国家财政收入", "累计名义同比", "03e60611a6be446099bef411599bceb0", 0),
    ("国家财政支出", "累计名义同比", "", 0),
    ("原煤产量", "当月同比", "2f76fed8960547e0a69ab99917423fc0", 0),
    ("发电量", "当月同比", "5cc686f383d84ed9bd11f4293ab4170e", 0),
]

PMI_MAPPING = [
    (0, 1, "制造业PMI", "a09aa989bdcf4cffa2021795722eb916"),
    (0, 2, "制造业生产指数", "6729aa00f9ed46d8b30c5d2312214b89"),
    (0, 3, "制造业新订单指数", "4151df33b53f4d02ae9f51fe402f1a50"),
    (0, 4, "制造业原材料库存指数", "c149709d0c48422d83a59d4b94d03bbb"),
    (0, 5, "制造业从业人员指数", "24454731f2fd46f1850da13fe6f39263"),
    (0, 6, "制造业供应商配送时间指数", "23a80d4340314e45ab0cc0ce69f3eeec"),
    (2, 1, "非制造业商务活动指数", "88a150208f6e4a1db8babe41ae700f66"),
    (2, 2, "非制造业新订单指数", "e64aa8133ca647da8f75893583a5bb24"),
    (2, 3, "非制造业投入品价格指数", "61d7f87361374c7e9f9b811a69e5e5c9"),
    (2, 4, "非制造业销售价格指数", "60c688be2e284933986a6f42fd65b366"),
    (2, 5, "非制造业从业人员指数", "bbe0067c5fbb49eda66f8fd8d60811cf"),
    (2, 6, "非制造业业务活动预期指数", "7f8ebe0d686142cbaccf5b2ee450c50c"),
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False, double_precision=10))


def norm(text):
    return re.sub(r"\s+", "", str(text)).replace("，", "、")


def numeric(text):
    result = pd.to_numeric(str(text).replace(",", ""), errors="coerce")
    return None if pd.isna(result) else float(result)


def fmt(value, digits=1):
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def yearmonth(year, month):
    return f"{int(year):04d}{int(month):02d}"


def calendar_index(period):
    return period.str[:4].astype(int) * 12 + period.str[4:].astype(int) - 1


def classify(metadata):
    name, unit, basis = str(metadata["name"]), str(metadata["unit"]), str(metadata["period_basis"])
    if "上月=100" in unit:
        return "官方未季调环比指数", 100
    if "上年同月=100" in unit:
        return "官方当月同比指数", 100
    if "上年同期=100" in unit:
        return "官方累计同比指数", 100
    if "累计同比" in basis:
        return "官方累计同比", 0
    if "同比" in basis and "上年" not in basis:
        return "官方同比", 0
    if metadata["category"] == "采购经理指数":
        return "扩散指数水平", 0
    if "上年同期" in basis:
        return "上年同期可比基数", 0
    if unit == "%" or any(word in name for word in ["失业率", "利润率", "资产负债率", "周转", "回收期"]):
        return "比例或时间水平", 0
    if basis == "本期累计":
        return "累计总量", 0
    if basis == "本期":
        return "当月总量或期末存量", 0
    return "其他原始指标", 0


def universal_comparison(connection, metadata, output, end_request, first_display_year):
    """在原有系列内部计算，缺失月份通过日期键保持缺失。"""
    meta = metadata.copy()
    types = meta.apply(classify, axis=1)
    meta["指标类型"] = [x[0] for x in types]
    meta["原值减数"] = [x[1] for x in types]
    data = pd.read_sql_query(
        "SELECT indicator_id,period,value,raw_path FROM observations WHERE status='有值' AND period<=? ORDER BY indicator_id,period",
        connection, params=[end_request],
    )
    data["月序号"] = calendar_index(data.period)
    data = data.merge(meta[["indicator_id", "指标类型", "原值减数"]], on="indicator_id", how="left", validate="many_to_one")
    data["分析值"] = data.value - data["原值减数"]
    index = pd.MultiIndex.from_frame(data[["indicator_id", "月序号"]])
    value_lookup = pd.Series(data.value.to_numpy(), index=index)
    prior_month_key = pd.MultiIndex.from_arrays([data.indicator_id, data["月序号"] - 1])
    prior_year_key = pd.MultiIndex.from_arrays([data.indicator_id, data["月序号"] - 12])
    prior_month = value_lookup.reindex(prior_month_key).to_numpy()
    prior_year = value_lookup.reindex(prior_year_key).to_numpy()
    data["上月原值"] = prior_month
    data["上年同月原值"] = prior_year
    same_year = data.period.str[4:] != "01"
    cumulative = data["指标类型"].eq("累计总量")
    data["累计差分当月原量"] = np.where(cumulative & same_year, data.value - prior_month, np.nan)
    data["原值较上月差"] = data.value - prior_month
    data.loc[data["指标类型"].isin(["官方累计同比", "官方累计同比指数", "累计总量"]) & ~same_year, "原值较上月差"] = np.nan
    quantitative = data["指标类型"].eq("当月总量或期末存量")
    data["原值比值环比%"] = np.where(quantitative & (prior_month > 0), (data.value / prior_month - 1) * 100, np.nan)
    data["原值比值同比%"] = np.where((quantitative | cumulative) & (prior_year > 0), (data.value / prior_year - 1) * 100, np.nan)
    # M1统计口径于2025年调整，跨断点的原值比值不能作为同口径同比。
    m1 = data.indicator_id.eq("add08d4a1ca049158166f126e169edde")
    data.loc[m1 & data.period.ge("202501") & data.period.lt("202601"), "原值比值同比%"] = np.nan
    data.loc[m1 & data.period.eq("202501"), "原值比值环比%"] = np.nan
    data.loc[m1 & data.period.eq("202501"), "原值较上月差"] = np.nan
    data["同月原值差"] = data.value - prior_year
    data.to_parquet(output / "全指标全历史比较.parquet", index=False)
    coverage = data.groupby("indicator_id").agg(历史首月=("period", "min"), 历史末月=("period", "max"), 历史有效值数=("value", "size"))
    latest = data.groupby("indicator_id", sort=False).tail(1).set_index("indicator_id")
    summary = meta.merge(coverage, on="indicator_id", how="left").merge(
        latest[["period", "value", "分析值", "原值较上月差", "原值比值环比%", "原值比值同比%"]],
        on="indicator_id", how="left",
    ).rename(columns={"period": "最新月份", "value": "最新原值"})
    summary["比较月份"] = summary["最新月份"].str[4:]
    years = range(first_display_year, int(end_request[:4]) + 1)
    for year in years:
        target = str(year) + summary["比较月份"].fillna("")
        key = pd.MultiIndex.from_arrays([summary.indicator_id, target.str[:4].where(target.str.len().eq(6), "0").astype(int) * 12 + pd.to_numeric(target.str[4:], errors="coerce").fillna(0).astype(int) - 1])
        summary[f"{year}年同月原值"] = value_lookup.reindex(key).to_numpy()
    summary["历史有效值数"] = summary["历史有效值数"].fillna(0).astype(int)
    summary.to_csv(output / "全指标历史覆盖与多年同月.csv", index=False, encoding="utf-8-sig")
    print(f"已比较全部 {len(meta):,} 个指标条目，共 {len(data):,} 个原始有效数值。", flush=True)
    return data, summary


def seasonal_releases(connection, run):
    """保存所有找到的修订值，并为每个数据月份选择快照内最新公布值。"""
    release_list = pd.read_sql_query("SELECT title,published_at,formed_on,url,table_path FROM releases WHERE table_count>0", connection)
    rows = []
    for release in release_list.itertuples(index=False):
        title = norm(release.title)
        if "规模以上工业增加值" in title:
            label = "工业增加值"
        elif "社会消费品零售总额" in title:
            label = "社会消费品零售"
        elif "全国固定资产投资" in title:
            label = "固定资产投资"
        else:
            continue
        tables = read_json(run / release.table_path)
        for table_number, table in enumerate(tables):
            grid = table["grid"]
            if not grid or "环比" not in "".join(str(s) for s in grid[0]):
                continue
            if not any("月份" in norm(s) for s in grid[0]):
                continue
            year = None
            for row_number, row in enumerate(grid[1:], 1):
                if len(row) < 3:
                    continue
                y = re.search(r"(20\d{2})年", norm(row[0]))
                if y:
                    year = int(y[1])
                m = re.fullmatch(r"(\d{1,2})月", norm(row[1]))
                value = numeric(row[-1])
                if year is None or m is None or value is None or not 1 <= int(m[1]) <= 12:
                    continue
                published = release.published_at or release.formed_on or ""
                rows.append({"指标": label, "数据月份": yearmonth(year, m[1]), "季调环比%": value,
                             "公布时间": published.replace("/", "-"), "官方来源": release.url,
                             "原稿标题": release.title, "表序号": table_number, "行序号": row_number})
    versions = pd.DataFrame(rows).sort_values(["指标", "数据月份", "公布时间", "官方来源"])
    if versions.empty:
        raise RuntimeError("未找到官方季调环比表。")
    selected = versions.drop_duplicates(["指标", "数据月份"], keep="last").copy()
    selected["月序号"] = calendar_index(selected["数据月份"])
    selected = selected.sort_values(["指标", "月序号"])
    for label, indexes in selected.groupby("指标").groups.items():
        part = selected.loc[indexes]
        sequence = pd.Series(part["季调环比%"].to_numpy(), index=part["月序号"])
        complete = sequence.reindex(range(sequence.index.min(), sequence.index.max() + 1))
        compound = ((1 + complete / 100).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1) * 100
        selected.loc[indexes, "三个月季调累计变化%"] = compound.reindex(part["月序号"]).to_numpy()
    return versions, selected


def pmi_extension(connection, run, data, metadata):
    release = connection.execute("SELECT title,published_at,url,table_path FROM releases WHERE title LIKE '%中国采购经理指数运行情况%' AND published_at IS NOT NULL ORDER BY REPLACE(published_at,'/','-') DESC LIMIT 1").fetchone()
    if release is None:
        return []
    title, published, url, table_path = release
    tables = read_json(run / table_path)
    extensions = []
    for table_index, column, label, iid in PMI_MAPPING:
        if table_index >= len(tables):
            continue
        latest_database_month = data.loc[data.indicator_id.eq(iid), "period"].max()
        for row in tables[table_index]["grid"]:
            match = re.fullmatch(r"(20\d{2})年(\d{1,2})月", norm(row[0]))
            if match is None or len(row) <= column:
                continue
            month = yearmonth(match[1], match[2])
            value = numeric(row[column])
            if value is not None and month > latest_database_month:
                extensions.append({"indicator_id": iid, "period": month, "value": value, "分析值": value,
                                   "raw_path": table_path, "来源类别": "发布稿新增月份", "官方来源": url,
                                   "公布时间": published, "指标": label})
    return extensions


def main_series(metadata):
    specifications = []
    for label, basis, iid, subtract in MAIN:
        if not iid:
            match = metadata.loc[metadata.category.eq("财政") & metadata.period_basis.eq("累计同比增减%") & metadata.name.str.contains("支出")]
            if len(match) != 1:
                raise RuntimeError("财政支出指标不能唯一识别。")
            iid = match.indicator_id.iloc[0]
        specifications.append((label, basis, iid, subtract))
    # CPI总指数跨基期的公布读数可以并列展示，不能当作固定权重的同一定基指数。
    for basis, ids in [
        ("当月同比", ["4c1065dd4e984b25a21190c843551697", "e5c318ffdbbc4d38898e52b52267eb25", "4ae9047687934a6390984c21d6ddab96", "53180dfb9c14411ba4b762307c85920c"]),
        ("当月环比，未季调", ["384ddbda2edc47969caa98263f16231b", "0dc091b5194c46afaf10369d5c55676a", "e437d965279d41ceb9ace591b62f6ffc", "f3904a1f5a384d54a3944ec6e2df3d1c"]),
    ]:
        specifications.extend(("CPI", basis, iid, 100) for iid in ids)
    for row in metadata.loc[metadata.category.eq("采购经理指数")].itertuples(index=False):
        label = next((r[2] for r in PMI_MAPPING if r[3] == row.indicator_id), None)
        if label is None:
            prefix = "制造业" if row.catalogue_path.endswith("/ 制造业采购经理指数") else "非制造业" if row.catalogue_path.endswith("/ 非制造业采购经理指数") else ""
            label = prefix + re.sub(r"\s*\(%\)\s*$", "", row.name).strip()
        specifications.append((label, "季调扩散指数", row.indicator_id, 0))
    return specifications


def macro_comparison(metadata, data, extension, seasonal, first_year, end_request):
    selected = []
    meta = metadata.set_index("indicator_id")
    for label, basis, iid, subtract in main_series(metadata):
        part = data.loc[data.indicator_id.eq(iid), ["period", "value", "raw_path"]].copy()
        if part.empty:
            continue
        part["数值"] = part.value - subtract
        part["指标"] = label
        part["口径"] = basis
        part["指标ID"] = iid
        part["类别"] = meta.loc[iid, "category"]
        part["统计版本"] = meta.loc[iid, "catalogue_path"].split(" / ")[-1]
        part["官方来源"] = SOURCE
        part["来源类别"] = "数据库"
        extras = [row for row in extension if row["indicator_id"] == iid]
        if extras:
            extra = pd.DataFrame(extras).rename(columns={"分析值": "数值"})
            extra["指标"] = label
            extra["口径"] = basis
            extra["指标ID"] = iid
            extra["类别"] = meta.loc[iid, "category"]
            extra["统计版本"] = part["统计版本"].iloc[0]
            part = pd.concat([part, extra], ignore_index=True)
        selected.append(part[["指标", "口径", "指标ID", "类别", "统计版本", "period", "数值", "value", "raw_path", "官方来源", "来源类别"]])
    macro = pd.concat(selected, ignore_index=True).rename(columns={"period": "数据月份", "value": "原值", "raw_path": "原始来源文件"})
    macro = macro.sort_values(["指标", "口径", "数据月份", "指标ID"])
    if macro.duplicated(["指标", "口径", "数据月份"]).any():
        raise RuntimeError("主要指标版本在同一个月份重叠，不能自动选择。")
    macro["年"] = macro["数据月份"].str[:4].astype(int)
    macro["月"] = macro["数据月份"].str[4:].astype(int)
    macro["月序号"] = calendar_index(macro["数据月份"])
    summaries, annual = [], []
    for (label, basis), part in macro.groupby(["指标", "口径"], sort=False):
        part = part.sort_values("数据月份")
        latest = part.iloc[-1]
        lookup = part.set_index("月序号")["数值"]
        now = float(latest["数值"])
        previous = lookup.get(latest["月序号"] - 1)
        last_year = lookup.get(latest["月序号"] - 12)
        contiguous = lookup.reindex(range(int(latest["月序号"]) - 5, int(latest["月序号"]) + 1))
        recent3 = contiguous.iloc[-3:].mean() if contiguous.iloc[-3:].notna().all() else np.nan
        prior3 = contiguous.iloc[:3].mean() if contiguous.iloc[:3].notna().all() else np.nan
        same_month = part.loc[part["月"].eq(latest["月"]) & part["年"].ge(first_year) & part["年"].lt(latest["年"]), "数值"]
        rank_note = "不同年份同月的公布指标读数；同比变动仍受去年基数影响"
        if label == "M1":
            same_month = part.loc[part["月"].eq(latest["月"]) & part["年"].ge(2025) & part["年"].lt(latest["年"]), "数值"]
            rank_note = "历史定位仅用2025年起新口径；更早旧口径另列"
        if label == "CPI":
            rank_note = "公布总指数读数并列；基期和权重更新，非固定权重指数"
        if "累计" in basis:
            recent3, prior3 = np.nan, np.nan
            if latest["月"] == 1:
                previous = None
        summary = {"指标": label, "口径": basis, "最新月份": latest["数据月份"], "最新数值": now,
                   "前月数值": previous, "较前月差": now - previous if previous is not None and pd.notna(previous) else None,
                   "上年同月数值": last_year, "较上年同月差": now - last_year if last_year is not None and pd.notna(last_year) else None,
                   "最近三月平均": recent3, "之前三月平均": prior3, "三月均值差": recent3 - prior3,
                   "此前同月样本数": len(same_month), "此前同月中位数": same_month.median(),
                   "此前同月最小值": same_month.min(), "此前同月最大值": same_month.max(),
                   "低于最新值的同月样本数": int(same_month.lt(now).sum()), "历史比较说明": rank_note,
                   "官方来源": latest["官方来源"]}
        for year in range(first_year, int(end_request[:4]) + 1):
            hit = part.loc[part["年"].eq(year) & part["月"].eq(latest["月"]), "数值"]
            summary[f"{year}年同月"] = float(hit.iloc[0]) if len(hit) else None
        summaries.append(summary)
        for year, year_part in part.groupby("年"):
            compare = year_part.loc[year_part["月"].le(latest["月"])]
            record = {"指标": label, "口径": basis, "年份": int(year), "比较截至月": int(latest["月"]),
                      "有值月份数": len(compare), "同月数值": float(compare.loc[compare["月"].eq(latest["月"]), "数值"].iloc[0]) if compare["月"].eq(latest["月"]).any() else None,
                      "年内至今月度读数均值": compare["数值"].mean() if "累计" not in basis else None,
                      "说明": "月度同比的均值不是累计增速" if "同比" in basis and "累计" not in basis else "累计采用同一截止月" if "累计" in basis else "指数或水平读数按实际有值月份求均值"}
            record.update({f"{month}月": float(year_part.loc[year_part["月"].eq(month), "数值"].iloc[0]) if year_part["月"].eq(month).any() else None for month in range(1, 13)})
            annual.append(record)
    return macro, pd.DataFrame(summaries), pd.DataFrame(annual)


def amounts_comparison(metadata, data, end_request):
    ids = [
        ("社会消费品零售总额", "1142a3a03e9045959e606a21822641ac", "当月总量"),
        ("社会消费品零售总额", "260a1794443b43dd93a59928b12f38af", "累计总量"),
        ("房地产开发投资", "bfb626c0dfa04afab67937c452ca9f50", "累计总量"),
        ("新建商品房销售面积", "d353226cf0434c929b6299f8d4987754", "累计总量"),
        ("新建商品房销售额", "090ed0c087024014a7fe903e409958df", "累计总量"),
        ("工业企业利润", "5d98d24036dd4dbebf7f5c0a80753df6", "累计总量"),
        ("工业企业营业收入", "f3fbbb5fbb28401bb6351039a171df5a", "累计总量"),
        ("M2", "f3c0ae453a54424489af41de315ec592", "期末存量"),
        ("M1", "add08d4a1ca049158166f126e169edde", "期末存量"),
        ("M0", "bd67997414b147a08d4aa03d146f4486", "期末存量"),
        ("货物出口", "9e38b39f55a7461ea195508c1bb7dbdc", "当月总量"),
        ("货物进口", "86a340fee806409ebdf4d0069bd23f29", "当月总量"),
        ("货物进出口", "6770f815eded431a93bf422d38ab488c", "当月总量"),
        ("货运量", "82483e0ae1574e87b89cb693edc143b2", "当月总量"),
        ("原煤产量", "8f384e0069b34e5c89def94fdc58272e", "当月总量"),
        ("发电量", "baafe3a9a09d4b39a366e5b625574aea", "当月总量"),
    ]
    lookup = metadata.set_index("indicator_id")
    output = []
    for label, iid, basis in ids:
        part = data.loc[data.indicator_id.eq(iid)].copy()
        if part.empty:
            continue
        sequence = part.set_index("月序号").value
        for row in part.itertuples(index=False):
            result = {"指标": label, "口径": basis, "月份": row.period, "单位": lookup.loc[iid, "unit"], "原始总量": row.value,
                      "指标ID": iid, "来源": SOURCE}
            for years in [1, 2, 3, 5]:
                prior = sequence.get(row.月序号 - years * 12)
                comparable = not (label == "M1" and row.period >= "202501" and row.period[:4] < str(2025 + years))
                result[f"{years}年前同月总量"] = prior
                result[f"{years}年复合增速%"] = ((row.value / prior) ** (1 / years) - 1) * 100 if comparable and prior is not None and prior > 0 and row.value > 0 else None
            result["说明"] = "原始金额或数量之比；样本范围及修订影响未剔除，不能替代官方可比增速"
            output.append(result)
    frame = pd.DataFrame(output)
    if not frame.empty:
        raw_lookup = data.set_index(["indicator_id", "period"])
        keys = pd.MultiIndex.from_arrays([frame["指标ID"], frame["月份"]])
        for original, target in [("原值比值环比%", "原值比值环比%"), ("上月原值", "上月原始总量"), ("原值较上月差", "原始月度差额")]:
            frame[target] = raw_lookup[original].reindex(keys).to_numpy()
    return frame


def industry_comparison(metadata, data, first_year, latest_period):
    production = metadata.loc[metadata.catalogue_path.str.contains("工业分大类行业增加值增长速度", regex=False) & metadata.catalogue_path.str.contains("2018", regex=False) & metadata.period_basis.isin(["同比增减%", "累计同比增减%"])]
    current = metadata.loc[metadata.catalogue_path.str.startswith("月度数据 / 工业 / 按行业分工业企业主要经济指标 (2018-至今) / ")]
    definitions = {}
    for row in production.itertuples(index=False):
        sector = norm(row.name.split("增加值")[0])
        kind = "增加值累计同比%" if row.period_basis == "累计同比增减%" else "增加值当月同比%"
        definitions[(sector, kind)] = row.indicator_id
    allowed = {
        ("工业企业营业收入", "本期累计"): "营业收入累计亿元",
        ("工业企业营业收入", "累计同比增减%"): "营业收入累计同比%",
        ("工业企业营业收入", "上年同期累计"): "营业收入可比基数亿元",
        ("工业企业利润总额", "本期累计"): "利润累计亿元",
        ("工业企业利润总额", "累计同比增减%"): "利润累计同比%",
        ("工业企业利润总额", "上年同期累计"): "利润可比基数亿元",
        ("工业企业应收账款", "同比增减%"): "应收账款期末同比%",
        ("工业企业产成品存货", "同比增减%"): "产成品库存期末同比%",
        ("工业企业资产总计", "本期"): "资产期末亿元",
        ("工业企业负债合计", "本期"): "负债期末亿元",
    }
    markers = {
        "工业企业营业收入": "营业收入", "工业企业利润总额": "利润总额", "工业企业应收账款": "应收账款",
        "工业企业产成品存货": "产成品存货", "工业企业资产总计": "资产总计", "工业企业负债合计": "负债合计",
    }
    sector_names = sorted({key[0] for key in definitions})
    if len(sector_names) != 41:
        raise RuntimeError("当前增加值分类没有完整41行业。")
    for row in current.itertuples(index=False):
        metric = row.catalogue_path.split(" / ")[-1]
        kind = allowed.get((metric, row.period_basis))
        if kind is None:
            continue
        sector = norm(row.name.split(markers[metric])[0])
        if sector in sector_names:
            definitions[(sector, kind)] = row.indicator_id
    parts = []
    metric_names = sorted({key[1] for key in definitions})
    for (sector, metric), iid in definitions.items():
        subset = data.loc[data.indicator_id.eq(iid), ["period", "value"]].copy()
        subset["行业"] = sector
        subset["指标"] = metric
        subset["指标ID"] = iid
        parts.append(subset)
    long = pd.concat(parts, ignore_index=True).rename(columns={"period": "数据月份", "value": "数值"})
    panel = long.pivot(index=["行业", "数据月份"], columns="指标", values="数值").reset_index()
    panel["利润数据库原始同比%"] = panel["利润累计同比%"]
    panel["利润比较说明"] = np.where(panel["利润可比基数亿元"].gt(0), "上年可比基数为正",
                                      np.where(panel["利润可比基数亿元"].le(0), "上年可比利润非正，普通同比不作增长下降解释", "可比利润基数未提供"))
    panel["利润累计同比%"] = panel["利润累计同比%"].where(panel["利润可比基数亿元"].gt(0))
    panel["营业收入利润率%"] = np.where(panel["营业收入累计亿元"].gt(0), panel["利润累计亿元"] / panel["营业收入累计亿元"] * 100, np.nan)
    panel["可比上年营业收入利润率%"] = np.where(panel["营业收入可比基数亿元"].gt(0), panel["利润可比基数亿元"] / panel["营业收入可比基数亿元"] * 100, np.nan)
    panel["利润率同比变化百分点"] = panel["营业收入利润率%"] - panel["可比上年营业收入利润率%"]
    panel["资产负债率%"] = np.where(panel["资产期末亿元"].gt(0), panel["负债期末亿元"] / panel["资产期末亿元"] * 100, np.nan)
    panel["利润可比净增额亿元"] = panel["利润累计亿元"] - panel["利润可比基数亿元"]
    period_month = latest_period[4:]
    same_month = panel.loc[panel["数据月份"].str.endswith(period_month) & panel["数据月份"].str[:4].astype(int).ge(first_year)].copy()
    breadth = []
    for month, group in panel.groupby("数据月份"):
        production = group["增加值累计同比%"]
        profits = group["利润累计同比%"]
        breadth.append({"数据月份": month, "生产有值行业数": int(production.notna().sum()), "增加值累计增长行业数": int(production.gt(0).sum()),
                        "利润可比行业数": int(profits.notna().sum()), "利润增长行业数": int(profits.gt(0).sum()),
                        "产增利降行业数": int((production.gt(0) & profits.lt(0)).sum()),
                        "利润增速中位数%": profits.median(), "增加值增速中位数%": production.median(),
                        "行业利润率中位数%": group["营业收入利润率%"].median()})
    compare = []
    for sector, group in panel.groupby("行业"):
        lookup = group.set_index("数据月份")
        if latest_period not in lookup.index:
            continue
        row = {"行业": sector, "最新月份": latest_period, "官方来源": SOURCE}
        for metric in ["增加值累计同比%", "营业收入累计同比%", "利润累计同比%", "营业收入利润率%", "利润率同比变化百分点", "应收账款期末同比%", "产成品库存期末同比%"]:
            row[metric] = lookup.loc[latest_period, metric]
            for years in [1, 3, 5]:
                past_month = yearmonth(int(latest_period[:4]) - years, int(period_month))
                past = lookup.loc[past_month, metric] if past_month in lookup.index else np.nan
                row[f"{metric}较{years}年前同月差"] = row[metric] - past
            for year in range(max(2018, first_year), int(latest_period[:4]) + 1):
                key = yearmonth(year, int(period_month))
                row[f"{metric}_{year}年同月"] = lookup.loc[key, metric] if key in lookup.index else np.nan
        compare.append(row)
    return long, panel, same_month, pd.DataFrame(breadth), pd.DataFrame(compare)


def price_base_decomposition(macro):
    """从同比与环比的数学关系观察本期变动和滚动基数，保留舍入残差。"""
    rows = []
    for label in ["CPI", "PPI"]:
        year_on_year = macro.loc[macro["指标"].eq(label) & macro["口径"].eq("当月同比")].set_index("月序号")
        monthly = macro.loc[macro["指标"].eq(label) & macro["口径"].eq("当月环比，未季调")].set_index("月序号")
        for position, row in year_on_year.iterrows():
            if position - 1 not in year_on_year.index or position not in monthly.index or position - 12 not in monthly.index:
                continue
            previous_yoy = float(year_on_year.loc[position - 1, "数值"])
            now_mom = float(monthly.loc[position, "数值"])
            old_mom = float(monthly.loc[position - 12, "数值"])
            model_yoy = ((1 + previous_yoy / 100) * (1 + now_mom / 100) / (1 + old_mom / 100) - 1) * 100
            current_log = np.log1p(now_mom / 100) * 100
            base_log = -np.log1p(old_mom / 100) * 100
            actual_log = (np.log1p(float(row["数值"]) / 100) - np.log1p(previous_yoy / 100)) * 100
            rows.append({"指标": label, "月份": row["数据月份"], "当月同比%": row["数值"], "前月同比%": previous_yoy,
                         "当月未季调环比%": now_mom, "上年同月未季调环比%": old_mom,
                         "按环比关系计算同比%": model_yoy, "与公布同比残差百分点": float(row["数值"]) - model_yoy,
                         "当月变化对数项": current_log, "上年同月退出对数项": base_log, "同比变化对数项": actual_log,
                         "对数舍入残差": actual_log - current_log - base_log,
                         "说明": "公布读数已经舍入；跨CPI基期更新还可能有链式口径差，分项是算术关系，不是经济原因"})
    return pd.DataFrame(rows)


def real_mom_summary(seasonal):
    rows = []
    for label, frame in seasonal.groupby("指标"):
        part = frame.sort_values("数据月份")
        latest = part.iloc[-1]
        history = part.set_index("月序号")
        previous_three_endpoint = int(latest["月序号"]) - 3
        prior_three = history.loc[previous_three_endpoint, "三个月季调累计变化%"] if previous_three_endpoint in history.index else None
        positive_run, negative_run = 0, 0
        for back in range(len(history)):
            key = int(latest["月序号"]) - back
            if key not in history.index or float(history.loc[key, "季调环比%"])*float(latest["季调环比%"]) <= 0:
                break
            if float(latest["季调环比%"] ) > 0:
                positive_run += 1
            else:
                negative_run += 1
        rows.append({"指标": label, "最新月份": latest["数据月份"], "最新官方季调环比%": latest["季调环比%"],
                     "最近三个月累计变化%": latest["三个月季调累计变化%"], "之前三个月累计变化%": prior_three,
                     "三个月变化差百分点": latest["三个月季调累计变化%"] - prior_three if prior_three is not None else None,
                     "连续正环比月数": positive_run, "连续负环比月数": negative_run})
    return pd.DataFrame(rows)


def multiyear_yoy_mom(macro, seasonal, first_year):
    rows = []
    definitions = [("工业增加值", "当月实际同比", "累计实际同比"),
                   ("社会消费品零售", "当月名义同比", "累计名义同比"),
                   ("固定资产投资", None, "累计名义同比"),
                   ("CPI", "当月同比", None), ("PPI", "当月同比", None)]
    for label, monthly_basis, cumulative_basis in definitions:
        source = macro.loc[macro["指标"].eq(label)]
        latest_period = source["数据月份"].max()
        month = int(latest_period[4:])
        for year in range(first_year, int(latest_period[:4]) + 1):
            period = yearmonth(year, month)
            values = source.loc[source["数据月份"].eq(period)].set_index("口径")["数值"]
            seasonal_row = seasonal.loc[seasonal["指标"].eq(label) & seasonal["数据月份"].eq(period)]
            mom = float(seasonal_row.iloc[0]["季调环比%"]) if len(seasonal_row) else values.get("当月环比，未季调")
            rolling_three = float(seasonal_row.iloc[0]["三个月季调累计变化%"]) if len(seasonal_row) else None
            rows.append({"指标": label, "年份": year, "同月月份": period,
                         "官方当月同比%": values.get(monthly_basis), "官方累计同比%": values.get(cumulative_basis),
                         "官方真实环比%": mom, "环比口径": "未季调" if label in ["CPI", "PPI"] else "季调",
                         "三个月季调累计变化%": rolling_three,
                         "官方来源": seasonal_row.iloc[0]["官方来源"] if len(seasonal_row) else SOURCE})
    return pd.DataFrame(rows)


def plot_results(macro, seasonal, industry_same_month, output, first_year):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    choices = [("工业增加值", "当月实际同比"), ("社会消费品零售", "当月名义同比"),
               ("固定资产投资", "累计名义同比"), ("房地产开发投资", "累计名义同比"),
               ("工业企业利润", "累计可比同比"), ("制造业PMI", "季调扩散指数")]
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), facecolor="white")
    for ax, (label, basis) in zip(axes.flat, choices):
        part = macro.loc[macro["指标"].eq(label) & macro["口径"].eq(basis) & macro["年"].ge(first_year)]
        ax.plot(pd.to_datetime(part["数据月份"], format="%Y%m"), part["数值"], color="#315E7A", linewidth=1.3)
        ax.axhline(50 if label == "制造业PMI" else 0, color="#939DA7", linewidth=.7)
        ax.set_title(f"{label}：{basis}", loc="left", fontsize=12)
        ax.set_ylabel("指数点" if label == "制造业PMI" else "%")
        ax.grid(axis="y", color="#E6E9ED")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(f"全国主要指标：{first_year}年至最新月份", x=.07, ha="left", fontsize=20)
    fig.text(.07, .025, "各指标原始口径分别标明；不同最新月份；累计同比保留原定义；历史为本次下载版本。", fontsize=10, color="#536371")
    fig.subplots_adjust(top=.91, bottom=.09, hspace=.34, wspace=.18)
    fig.savefig(output / "多年宏观走势.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), facecolor="white")
    colors = plt.get_cmap("tab10")
    for ax, (label, basis) in zip(axes.flat, choices):
        part = macro.loc[macro["指标"].eq(label) & macro["口径"].eq(basis)]
        last_year = int(part["年"].max())
        for year in range(last_year - 5, last_year + 1):
            values = part.loc[part["年"].eq(year)].set_index("月")["数值"].reindex(range(1, 13))
            ax.plot(range(1, 13), values, label=str(year), color="#1D3E57" if year == last_year else colors((year - last_year + 5) % 10), linewidth=2.7 if year == last_year else 1.0, alpha=1 if year == last_year else .70)
        ax.set_title(f"{label}：{basis}", loc="left", fontsize=12)
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("月份")
        ax.set_ylabel("指数点" if label == "制造业PMI" else "%")
        ax.grid(axis="y", color="#E6E9ED")
        ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle("各年份月度路径：同月比较与年内变化", x=.07, ha="left", fontsize=20)
    fig.text(.07, .02, "最新年份用粗线；未公布或未单独调查的月份留空。月度同比的平均数不等于累计同比。", fontsize=10, color="#536371")
    fig.subplots_adjust(top=.92, bottom=.085, hspace=.42, wspace=.18)
    fig.savefig(output / "六年同月与年内路径.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="white")
    for ax, label in zip(axes, ["工业增加值", "社会消费品零售", "固定资产投资"]):
        part = seasonal.loc[seasonal["指标"].eq(label) & seasonal["数据月份"].ge("202401")]
        dates = pd.to_datetime(part["数据月份"], format="%Y%m")
        ax.bar(dates, part["季调环比%"], width=18, color="#8EA9B9", label="官方季调环比")
        ax.plot(dates, part["三个月季调累计变化%"], color="#2A465C", linewidth=1.7, label="三个月环比复合变化")
        ax.axhline(0, color="#89939E", linewidth=.8)
        ax.set_title(label, loc="left", fontsize=12)
        ax.set_ylabel("%")
        ax.grid(axis="y", color="#E6E9ED")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle("官方季调环比与三个月复合变化", x=.07, ha="left", fontsize=20)
    fig.text(.07, .02, "每月采用快照内保存的最新公布修订值；三个月变化为三个月(1+环比)连乘后减1。", fontsize=10, color="#536371")
    fig.subplots_adjust(top=.91, bottom=.085, hspace=.35)
    fig.savefig(output / "真实环比与三个月变化.png", dpi=160)
    plt.close(fig)
    names = sorted(industry_same_month["行业"].unique())
    years = sorted(industry_same_month["数据月份"].str[:4].unique())
    fig, axes = plt.subplots(1, 2, figsize=(15, 14), facecolor="white")
    image = None
    for ax, metric, label in zip(axes, ["增加值累计同比%", "利润累计同比%"], ["增加值累计同比", "利润累计同比"]):
        matrix = industry_same_month.assign(年份=industry_same_month["数据月份"].str[:4]).pivot(index="行业", columns="年份", values=metric).reindex(index=names, columns=years)
        palette = plt.get_cmap("RdBu_r").copy()
        palette.set_bad("#E5E8EC")
        image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect="auto", cmap=palette, vmin=-100, vmax=100)
        ax.set_yticks(range(len(names)), names if metric == "增加值累计同比%" else [], fontsize=9)
        ax.set_xticks(range(len(years)), years, rotation=45, fontsize=9)
        ax.set_title(label, loc="left", fontsize=14)
    fig.suptitle("41行业多年同期：各年1月至同一截止月", x=.05, ha="left", fontsize=20)
    fig.text(.05, .025, "灰色为空值或不可比；颜色范围固定为±100%，超出范围只饱和颜色，完整数值保留在明细。", fontsize=10, color="#536371")
    fig.subplots_adjust(left=.28, right=.91, top=.94, bottom=.075, wspace=.10)
    color_axes = fig.add_axes([.93, .12, .015, .76])
    fig.colorbar(image, cax=color_axes, label="累计同比 %")
    fig.savefig(output / "41行业多年同期.png", dpi=150)
    plt.close(fig)


def write_report(output, scope, macro_summary, seasonal, breadth, industry_panel, amounts, first_year, momentum, prices):
    latest = macro_summary.set_index(["指标", "口径"])
    latest_seasonal = seasonal.sort_values("数据月份").groupby("指标").tail(1).set_index("指标")
    years = list(range(max(first_year, int(scope["end_request"][:4]) - 5), int(scope["end_request"][:4]) + 1))
    choices = [("工业增加值", "当月实际同比"), ("服务业生产", "当月实际同比"), ("社会消费品零售", "当月名义同比"),
               ("固定资产投资", "累计名义同比"), ("房地产开发投资", "累计名义同比"), ("工业企业利润", "累计可比同比"),
               ("CPI", "当月同比"), ("PPI", "当月同比"), ("全国城镇调查失业率", "水平"), ("制造业PMI", "季调扩散指数")]
    current_momentum = momentum.set_index("指标")
    latest_breadth = breadth.sort_values("数据月份").iloc[-1]
    industry_end = latest_breadth["数据月份"]
    automotive_start = yearmonth(int(industry_end[:4]) - 5, int(industry_end[4:]))
    automotive = industry_panel.loc[industry_panel["行业"].eq("汽车制造业") & industry_panel["数据月份"].str.endswith(industry_end[4:]) & industry_panel["数据月份"].ge(automotive_start)].sort_values("数据月份")
    previous_breadth = breadth.loc[breadth["数据月份"].eq(yearmonth(int(latest_breadth["数据月份"][:4]) - 1, int(latest_breadth["数据月份"][4:])))].iloc[0]
    industrial = latest.loc[("工业增加值", "当月实际同比")]
    retail = latest.loc[("社会消费品零售", "当月名义同比")]
    investment = latest.loc[("固定资产投资", "累计名义同比")]
    property_investment = latest.loc[("房地产开发投资", "累计名义同比")]
    profits = latest.loc[("工业企业利润", "累计可比同比")]
    pmi = latest.loc[("制造业PMI", "季调扩散指数")]
    retail_mom = latest_seasonal.loc["社会消费品零售"]
    below_retail_history = pd.notna(retail["此前同月最小值"]) and retail["最新数值"] < retail["此前同月最小值"]
    below_investment_history = pd.notna(investment["此前同月最小值"]) and investment["最新数值"] < investment["此前同月最小值"]
    retail_position = f"低于此前{int(retail['此前同月样本数'])}个年份同月公布增速" if below_retail_history else f"此前同月中位数为{fmt(retail['此前同月中位数'])}%"
    investment_position = f"低于此前{int(investment['此前同月样本数'])}个年份同一累计区间读数" if below_investment_history else f"此前同期读数中位数为{fmt(investment['此前同月中位数'])}%"
    lines = ["# 国家统计局全国月度：多年同比、环比与行业比较", "",
             f"数据快照：{scope['as_of']}。多年展示窗口：{first_year}年至最新月份。所有指标的完整可获取历史另存，不以展示窗口裁剪计算明细。",
             "", "## 数据显示的主要变化", "",
             f"1. 工业增加值{industrial['最新月份']}当月同比{fmt(industrial['最新数值'])}%，上年同月为{fmt(industrial['上年同月数值'])}%，此前{int(industrial['此前同月样本数'])}个年份同月中位数为{fmt(industrial['此前同月中位数'])}%。最近三个月官方季调环比复合变化{fmt(current_momentum.loc['工业增加值', '最近三个月累计变化%'], 2)}%，之前三个月为{fmt(current_momentum.loc['工业增加值', '之前三个月累计变化%'], 2)}%。",
             f"2. 零售{retail['最新月份']}当月同比{fmt(retail['最新数值'])}%，{retail_position}。最新月季调环比{fmt(retail_mom['季调环比%'], 2)}%，最近三个月复合变化{fmt(current_momentum.loc['社会消费品零售', '最近三个月累计变化%'], 2)}%，之前三个月{fmt(current_momentum.loc['社会消费品零售', '之前三个月累计变化%'], 2)}%。同月同比位置、单月环比与三个月累计变化分别报告。",
             f"3. 固定资产投资截至{investment['最新月份']}累计同比{fmt(investment['最新数值'])}%，{investment_position}。当前连续负季调环比月数为{int(current_momentum.loc['固定资产投资', '连续负环比月数'])}；最近三个月复合变化{fmt(current_momentum.loc['固定资产投资', '最近三个月累计变化%'], 2)}%，之前三个月为{fmt(current_momentum.loc['固定资产投资', '之前三个月累计变化%'], 2)}%。房地产投资各年同期增速依次为" + "、".join(f"{year}年{fmt(property_investment.get(f'{year}年同月'))}%" for year in years) + "。",
             f"4. 工业总体利润截至{profits['最新月份']}累计同比{fmt(profits['最新数值'])}%，上年同期同比为{fmt(profits['上年同月数值'])}%。利润同比增长行业数从{int(previous_breadth['利润增长行业数'])}/{int(previous_breadth['利润可比行业数'])}变为{int(latest_breadth['利润增长行业数'])}/{int(latest_breadth['利润可比行业数'])}，行业利润增速中位数从{fmt(previous_breadth['利润增速中位数%'])}%变为{fmt(latest_breadth['利润增速中位数%'])}%。同期间生产增长行业数从{int(previous_breadth['增加值累计增长行业数'])}/41变为{int(latest_breadth['增加值累计增长行业数'])}/41。盈利分布与生产分布分别比较。",
             f"5. 制造业PMI{pmi['最新月份']}为{fmt(pmi['最新数值'])}，此前同月中位数为{fmt(pmi['此前同月中位数'])}。最近三个月均值{fmt(pmi['最近三月平均'])}，之前三个月均值{fmt(pmi['之前三月平均'])}；生产与订单分项的三个月均值分别列在明细。单月读数与三个月均值描述不同时间范围。",
             f"6. 汽车行业各年1—{int(industry_end[4:])}月按公布金额计算的营业收入利润率依次为" + "、".join(f"{row['数据月份'][:4]}年{fmt(row['营业收入利润率%'], 2)}%" for row in automotive.to_dict('records')) + "。这是多年行业汇总利润率路径；价格、成本、产品组合各自贡献尚未分解。",
             "", "## 比较方法", "",
             "同比回答相对上年同月或同一累计区间的变化；环比回答相对上月的变化。同比增速相邻月份之差称为增速变化，不能称为环比。工业、零售、投资使用官方季调环比；CPI、PPI使用官方上月=100指数减100，保持未季调口径。",
             "", "同月横比采用同一指标各年相同月份，累计横比采用各年1月至同一截止月。年内路径保留每个月的原始同比、指数或水平。月度同比的三个月均值描述增速路径，不冒充累计增速。官方季调环比三个月连乘计算三个月累计变化。",
             "", "累计总量允许在同年相邻月份差分，输出明确标为机械差分，不能从累计增速相减反推出当月同比。企业样本和比较基数调整时，原始金额比值及其多年复合增速可能与官方可比增速不同，二者分别保存。",
             "", "CPI各基期总指数读数带版本标识并列，不能作为固定权重价格指数。M1在2025年启用新口径，历史定位和跨期金额比值避开断点。旧行业分类保留独立指标，41行业同期分析采用2018年至今分类。",
             "", "## 多年同月与同一累计区间", "",
             "数值为原口径读数；同比与失业率单位为%，PMI单位为指数点。每行的比较月份单列，货币数据最新为7月，PMI发布稿最新为9月。", "",
             "| 指标与口径 | 比较月 | " + " | ".join(str(y) for y in years) + " |",
             "|---|---|" + "---:|" * len(years)]
    for label, basis in choices:
        row = latest.loc[(label, basis)]
        lines.append(f"| {label}：{basis} | {int(row['最新月份'][4:])}月 | " + " | ".join(fmt(row.get(f"{y}年同月")) for y in years) + " |")
    lines += ["", "## 官方环比与短期变化", "", "| 指标 | 最新月 | 上月季调环比% | 最新季调环比% | 三个月累计变化% |", "|---|---|---:|---:|---:|"]
    for label in ["工业增加值", "社会消费品零售", "固定资产投资"]:
        row = latest_seasonal.loc[label]
        previous = seasonal.loc[seasonal["指标"].eq(label) & seasonal["月序号"].eq(row["月序号"] - 1), "季调环比%"]
        lines.append(f"| {label} | {row['数据月份']} | {fmt(previous.iloc[0] if len(previous) else None, 2)} | {fmt(row['季调环比%'], 2)} | {fmt(row['三个月季调累计变化%'], 2)} |")
    for label in ["CPI", "PPI"]:
        row = latest.loc[(label, "当月环比，未季调")]
        lines.append(f"\n- {label}官方未季调环比：{row['最新月份']}为{fmt(row['最新数值'])}%，前月为{fmt(row['前月数值'])}%。")
    lines += ["", "工业、零售和投资的近期三个月变化与之前三个月分别比较：", "",
              "| 指标 | 最近三个月累计% | 之前三个月累计% | 差值，百分点 | 连续正环比月数 | 连续负环比月数 |", "|---|---:|---:|---:|---:|---:|"]
    for row in momentum.to_dict("records"):
        lines.append(f"| {row['指标']} | {fmt(row['最近三个月累计变化%'], 2)} | {fmt(row['之前三个月累计变化%'], 2)} | {fmt(row['三个月变化差百分点'], 2)} | {row['连续正环比月数']} | {row['连续负环比月数']} |")
    lines += ["", "## 多年事实与当前变化", ""]
    for label, basis in choices:
        row = latest.loc[(label, basis)]
        unit = "指数点" if "PMI" in label else "个百分点"
        lines.append(f"- {label}（{basis}）：最新{row['最新月份']}为{fmt(row['最新数值'])}；上年同月为{fmt(row['上年同月数值'])}，相差{fmt(row['较上年同月差'])}{unit}；前月为{fmt(row['前月数值'])}，相差{fmt(row['较前月差'])}{unit}。此前同月{int(row['此前同月样本数'])}个读数的中位数为{fmt(row['此前同月中位数'])}，范围{fmt(row['此前同月最小值'])}至{fmt(row['此前同月最大值'])}。")
        if pd.notna(row["三月均值差"]):
            lines.append(f"  最近三个月读数均值{fmt(row['最近三月平均'])}，此前三个月{fmt(row['之前三月平均'])}，均值差{fmt(row['三月均值差'])}{unit}。")
    industry_month = industry_panel["数据月份"].max()
    month = industry_month[4:]
    b = breadth.loc[breadth["数据月份"].str.endswith(month)]
    lines += ["", "## 41行业的多年同期分布", "", f"各年比较1—{int(month)}月；行业数量统计不按规模加权。", "",
              "| 年份 | 生产增长行业数 | 利润增长行业数 | 可比利润行业数 | 产增利降行业数 | 利润增速中位数% |", "|---|---:|---:|---:|---:|---:|"]
    for row in b.itertuples(index=False):
        lines.append(f"| {row.数据月份[:4]} | {row.增加值累计增长行业数} | {row.利润增长行业数} | {row.利润可比行业数} | {row.产增利降行业数} | {fmt(getattr(row, '_6'))} |")
    lines += ["", "行业营业收入利润率=本期累计利润/本期累计营业收入；同比利润率变化使用本次快照提供的上年同期可比利润与可比收入基数。利润金额为负、同比下降、缺少可比基数分别处理。生产实际增速与财务名义增速的差异本身不提供因果解释。",
              "", "数据库原始利润百分比另列保留；上年可比利润非正或基数未提供时，不纳入普通同比增长/下降分布。最新燃料加工行业数据库为-569.9%，但发布稿注1因上年亏损不列普通同比，故其分析同比留空。",
              "", "## 总量、多年复合变化与口径差异", "",
              "主要金额和数量的1、2、3、5年同月比值及复合增速保存在主要总量多年复合变化.csv。原始金额同比与官方可比同比分列，不能优先选用较大者。M1跨2025年口径断点的比值留空。"]
    lines += ["", "## CPI和PPI的当月变化与滚动基数", "",
              "同一价格序列的关系为：(1+本月同比)=(1+上月同比)×(1+本月环比)/(1+上年同月环比)。使用已公布舍入读数计算时，残差保留；跨基期时还可能出现链式口径差。", "",
              "| 指标 | 月份 | 本月同比% | 上月同比% | 本月环比% | 上年同月环比% | 按关系计算同比% | 残差，百分点 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in prices.sort_values("月份").groupby("指标").tail(1).to_dict("records"):
        lines.append(f"| {row['指标']} | {row['月份']} | {fmt(row['当月同比%'])} | {fmt(row['前月同比%'])} | {fmt(row['当月未季调环比%'])} | {fmt(row['上年同月未季调环比%'])} | {fmt(row['按环比关系计算同比%'], 3)} | {fmt(row['与公布同比残差百分点'], 3)} |")
    lines += ["", "## 货币的多年同比与原始存量", "",
              "| 指标 | 比较月 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label in ["M2", "M1", "M0"]:
        row = macro_summary.loc[macro_summary["指标"].eq(label)].iloc[0]
        lines.append(f"| {label}同比% | {row['最新月份']} | " + " | ".join(fmt(row.get(f'{year}年同月')) for year in range(2021, 2027)) + " |")
    lines += ["", "M1的2025年前读数为旧口径，2025年起为新口径，分段展示。"]
    for label in ["M2", "M1"]:
        row = macro_summary.loc[macro_summary["指标"].eq(label)].iloc[0]
        lines.append(f"{label}最新{row['最新月份']}同比{fmt(row['最新数值'])}%，上年同月读数{fmt(row['上年同月数值'])}%。最近三个月同比均值{fmt(row['最近三月平均'])}%，之前三个月{fmt(row['之前三月平均'])}%。")
    currency_latest = amounts.loc[amounts["指标"].isin(["M2", "M1", "M0"])].sort_values("月份").groupby("指标").tail(1)
    lines += ["", "| 指标 | 月份 | 期末存量，亿元 | 上月存量，亿元 | 原始月度差额，亿元 | 未季调原值环比% |", "|---|---|---:|---:|---:|---:|"]
    for row in currency_latest.to_dict("records"):
        lines.append(f"| {row['指标']} | {row['月份']} | {fmt(row['原始总量'], 2)} | {fmt(row['上月原始总量'], 2)} | {fmt(row['原始月度差额'], 2)} | {fmt(row['原值比值环比%'], 2)} |")
    lines += ["原始货币存量、月度净增额及未季调环比保留在总量与全历史文件内。这些值描述货币数量及其增速，不能单独认定市场资金价格或信用需求的变化。"]
    lines += ["", "## 数据文件", "",
              "- 全国主要指标多年同月.csv：主要指标最近十年同月读数、上年同月及前月变化、此前同月范围。",
              "- 全国主要指标各年月度路径.csv：每年1—12月读数及同一截至月比较。",
              "- 官方季调环比最新修订.csv：所有已解析月份的最新公布版本、三个月累计变化及来源。",
              "- 官方季调环比所有保存版本.csv：修订前后值与公布时间。",
              "- 41行业多年同期.csv、41行业月度生产盈利.csv：完整行业历史与同月横比。",
              "- 全指标全历史比较.parquet：全部有值指标的原始值、真实日期匹配和适用的比值、差分。",
              "- 全指标历史覆盖与多年同月.csv：完整目录，包括全空条目及不同统计版本。",
              "", "## 解释范围", "",
              "报告使用国家统计局原始数据和本地计算，不引用景气解读。同比、季调环比、同月历史位置和行业分布分别回答不同问题；共同变化可以作为描述证据，不能直接认定政策、价格或融资为原因。缺少资金价格和信用结构数据，金融市场流动性不能由M1/M2单独确定。历史数值为当前下载或保存的最新公布版本，不能当作当时首次可得值。"]
    (output / "多年同比环比分析.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="全国月度全部历史与多年同月、官方同比、真实环比比较。")
    parser.add_argument("--run-id", default="20261004_national_full")
    parser.add_argument("--display-years", type=int, default=10)
    parser.add_argument("--workbook-dir", default="outputs/nbs_multiyear_20261004")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id) or not 3 <= args.display_years <= 40:
        parser.error("快照名称或多年展示长度不正确。")
    run = BASE / "runs" / args.run_id
    output = BASE / "analyses" / (args.run_id + "_multiyear_v2")
    output.mkdir(parents=True, exist_ok=True)
    workbook_dir = ROOT / args.workbook_dir
    workbook_dir.mkdir(parents=True, exist_ok=True)
    scope = read_json(run / "scope.json")
    first_year = int(scope["end_request"][:4]) - args.display_years + 1
    save_json(output / "分析方案.json", {"快照": args.run_id, "展示首年": first_year, "计算历史": "全部可获取历史，不按展示窗口裁剪",
              "同比": "优先保留官方同比；原始总量比值单列，不能替代可比口径",
              "环比": "工业、零售、投资采用官方季调环比表；价格采用官方上月=100；其他适用总量计算未季调原值比值",
              "多年": "同一个月份横比、同一累计截止期横比、逐年1至12月路径、1/2/3/5年原始总量复合变化",
              "缺失": "保留缺失与不可比；不拆分1—2月、不从同比推环比、不跨年差分累计值",
              "行业": "2018年至今相同分类的全部41行业；其他版本在全指标历史结果中独立保留"})
    connection = sqlite3.connect(f"file:{(run / '国家统计局月度.sqlite').resolve().as_posix()}?mode=ro", uri=True)
    try:
        metadata = pd.read_sql_query("SELECT * FROM indicators", connection)
        data, full_summary = universal_comparison(connection, metadata, output, scope["end_request"], first_year)
        versions, seasonal = seasonal_releases(connection, run)
        extensions = pmi_extension(connection, run, data, metadata)
        macro, summary, annual = macro_comparison(metadata, data, extensions, seasonal, first_year, scope["end_request"])
        amounts = amounts_comparison(metadata, data, scope["end_request"])
        industrial_latest = data.loc[data.indicator_id.eq("7f30595d047445e0a3880ab928795892"), "period"].max()
        industry_long, industry_panel, industry_same, breadth, industry_latest = industry_comparison(metadata, data, first_year, industrial_latest)
        prices = price_base_decomposition(macro)
        momentum = real_mom_summary(seasonal)
        true_mom_years = multiyear_yoy_mom(macro, seasonal, first_year)
    finally:
        connection.close()
    for name, frame in [
        ("全国主要指标完整历史.csv", macro), ("全国主要指标多年同月.csv", summary), ("全国主要指标各年月度路径.csv", annual),
        ("官方季调环比所有保存版本.csv", versions), ("官方季调环比最新修订.csv", seasonal),
        ("主要总量多年复合变化.csv", amounts), ("41行业月度生产盈利.csv", industry_panel),
        ("41行业多年同期.csv", industry_same), ("41行业各月分布.csv", breadth), ("41行业最新与往年同期比较.csv", industry_latest),
        ("CPI与PPI滚动基数分解.csv", prices), ("真实环比三个月比较.csv", momentum),
        ("同比环比多年同月综合比较.csv", true_mom_years),
    ]:
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")
    industry_long.to_parquet(output / "41行业指标来源.parquet", index=False)
    save_json(output / "发布稿新增月份.json", extensions)
    plot_results(macro, seasonal, industry_same, output, first_year)
    write_report(output, scope, summary, seasonal, breadth, industry_panel, amounts, first_year, momentum, prices)
    latest_amounts = amounts.sort_values("月份").groupby(["指标", "口径"]).tail(1)
    amount_rows = records(latest_amounts)
    seasonal_same = seasonal.assign(年份=seasonal["数据月份"].str[:4].astype(int), 月=seasonal["数据月份"].str[4:].astype(int))
    workbook_payload = {
        "日期": scope["as_of"], "首年": first_year, "末年": int(scope["end_request"][:4]), "结果目录": str(output),
        "宏观同月": records(summary), "年内路径": records(annual.loc[annual["年份"].ge(first_year)]),
        "季调环比": records(seasonal), "行业同期": records(industry_same), "行业分布": records(breadth.loc[breadth["数据月份"].str.endswith(industrial_latest[4:])]),
        "环比三个月比较": records(momentum), "价格滚动基数": records(prices.sort_values("月份").groupby("指标").tail(1)),
        "同比环比多年同月": records(true_mom_years),
        "金额复合": amount_rows, "宏观月度": records(macro.loc[macro["年"].ge(first_year)]),
        "来源": records(metadata.loc[metadata.indicator_id.isin(macro["指标ID"].unique()), ["indicator_id", "category", "name", "unit", "period_basis", "catalogue_path", "annotation"]]),
    }
    save_json(workbook_dir / "workbook_input.json", workbook_payload)
    stats = {"生成时间": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"), "结果目录": str(output),
             "全部指标条目": len(metadata), "全部有效历史数值": len(data), "宏观系列": len(summary),
             "宏观类别数": macro["类别"].nunique(), "宏观历史读数": len(macro), "行业数": industry_panel["行业"].nunique(),
             "行业历史月份数": industry_panel["数据月份"].nunique(), "官方季调环比保存值数": len(versions), "官方季调环比最新值数": len(seasonal),
             "最新季调环比": records(seasonal.sort_values("数据月份").groupby("指标").tail(1))}
    save_json(output / "分析结果.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
