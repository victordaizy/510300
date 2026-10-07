"""保存本轮结论、复核既有数值，并生成可独立阅读的审阅包。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = Path("reports/research/510300_participant_identity_clock_v1")
OUT = ROOT / REL
ZIP_NAME = "510300_一月资金行为与持有验证_V1_GPT审阅_20260922.zip"


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    summary = read(OUT / "summary.json")
    summary.update(status="COMPLETED_JANUARY_CASE_AND_HOLDER_VALIDATION_NOT_PREDICTIVE_TEST",
                   user_priority="预测资金持续性与承接，持有报告只作验证和当时已知背景",
                   case_daily_points=57, new_prediction_models=0, current_signal="NOT_ESTABLISHED",
                   next_task="资金行为预测任务书.md", numerical_protocol_frozen=False,
                   price_data_end="2026-09-11", price_volume_data_end="2026-08-12")
    dump(OUT / "summary.json", summary)
    dump(OUT / "receipts/本轮输入快照.json", {
        "recorded_at": stamp(), "role": "记录现有本轮输入，不声称原始历史收件凭证",
        "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)}
                  for p in sorted((OUT / "inputs").glob("*")) if p.is_file()]})
    dump(OUT / "facts/其他参与者来源摘要.json", {
        "private_intentions": {
            "source": "https://static.simuwang.com/Uploads/Documents/Research/Manager-Confidence-Index/2024/202408150005.pdf",
            "source_sha256": "2b0318f22be7769590bc730c595aa396a14d2e8958db76b266ed35cd71878b34",
            "forecast_month": "2024-08", "source_page": 3,
            "planned_exposure_index": 111.64, "market_expectation_index": 119.83,
            "planned_large_increase_percent": 5.2, "planned_increase_percent": 21.6,
            "planned_unchanged_percent": 66.4, "planned_decrease_percent": 5.2,
            "planned_large_decrease_percent": 1.7,
            "public_timestamp": "UNKNOWN_NOT_EQUAL_TO_COVER_OR_FILE_PATH_DATE",
            "usage": "独立操作意向来源线索，尚未建连续母集或用于预测；舍入比例合计100.1%"},
        "ssf_annual": {"status": "LOCAL_DOWNLOAD_FAILED_SOURCE_REMAINS_LEAD",
                       "source": "https://www.ssf.gov.cn/portal/yljjgl/webinfo/2025/09/1761903919491454.htm"},
        "margin_public_clock": {"status": "LOCAL_DOWNLOAD_FAILED_SOURCE_REMAINS_LEAD",
                                "source": "https://investor.szse.cn/institute/video/studio/t20100818_538365.html"},
        "wechat": "已作公开索引检索，未取得足够连续、带原始时点的该类意向母集；不推断微信不存在资料。"})
    views = ["2026年1月_价格成交份额与政策对照.png", "510300_资金身份与全历史走势.png",
             "20250830_33fe9c2203_原文第58页.png", "2025Q4_original_原文第13页.png",
             "20180328_0f55abc3ac_原文第69页.png", "20260331_e1ed47d747_原文第75页.png",
             "20260829_0ed6535396_原文第60页.png", "dividend_20260112_原文第2页.png"]
    dump(OUT / "receipts/图形与原表目视核对.json", {
        "recorded_at": stamp(), "status": "PASS_CHART_LAYOUT_AND_KEY_SOURCE_PAGES",
        "checks": ["57日与3476日价格未抽样", "中文无缺字、图例和标注可读",
                   "2026中报身份缺失保留空白", "分红日期和金额原表一致",
                   "中报实名与四季报匿名分开", "联接基金口径原表可区分"],
        "files": [{"path": "figures/" + name, "sha256": digest(OUT / "figures" / name)} for name in views]})
    prior = {
        "日频份额方向旧失败.json": "reports/research/510300_daily_etf_flow_direction_report.json",
        "季度申赎旧失败.json": "reports/research/510300_original_quarterly_flow_policy_v1/result.json",
        "融资压力旧失败.json": "reports/discovery/510300_market_leverage_cascade_5d_discovery_v0.json",
        "反转固定诊断.json": "reports/research/510300_reversal_monthly_diagnostic_v1/summary.json",
    }
    records = []
    for name, src in prior.items():
        target = OUT / "evidence" / name
        target.write_bytes((ROOT / src).read_bytes())
        records.append({"path": "evidence/" + name, "original": src, "sha256": digest(target),
                        "scope": "继承保存的原裁决，本轮不重新运行原模型或账户"})
    dump(OUT / "receipts/既有裁决来源.json", records)
    (OUT / "用户要求.md").write_text(
        "# 用户要求与本轮边界\n\n"
        + read(OUT / "source_plan.json")["user_objective"] + "\n\n"
        "最新纠偏原话：很多信息延后披露的，这个只能做到验证。就比如很多是很明显的减仓，国家队，比如26年1月，这么大的放量就是国家队在放量，在卖啊，灵活变通一点，我们重点是预测，大资金会怎么想，怎么操作，然后跟随。\n\n"
        "沿用：需要图形；涨幅较大后结合回撤决定退出；宏观政策、预期差和资金行为共同研究；20万元主账户、2万元成本对照；微信公众号等公开材料可以检索。\n\n"
        "本轮完成一月案例、持有验证层和新的预测任务定义；没有完成新模型、连续账户或独立达标。\n", encoding="utf-8")
    (OUT / "审阅提示词.md").write_text(
        "请阅读研究结论、资金行为预测任务书、两张图及保存数据。用户要根据公开信息预测大资金下一步操作并跟随；延后持仓仅用于验证。请先指出论证错误，再给出下一步研究优先级、具体策略假说、对照、验证与停止条件。\n\n"
        "重点检查：一月供给/承接是否支持所述推断；匿名四季报是否被误写为实名；中期快照下界能否归因到一月；除息是否正确；份额统计日与公开日是否混用；怎样避免把净流入方向旧失败改名再试；如何把资金意向、约束和价格承接转成有限、可证伪的预测。\n\n"
        "请区分科学证据与文件完整性。28份报告和一个已知案例不等于预测有效。不要将本包说成已经通过外部审阅或达成夏普目标，也不要直接授权实盘。给出可执行的下一项数值协议，保留所有失败和已终止85/15，不围绕旧达标线调参。\n", encoding="utf-8")
    (OUT / "交付导航.md").write_text(
        "# 阅读顺序\n\n1. 用户要求.md与研究结论.md：先看一月主图与结论。\n"
        "2. 资金行为预测任务书.md：后续预测对象、动态更新、比较与停止条件；尚未冻结数值模型。\n"
        "3. january_case_summary.json与facts目录：57日案例、28期持有结构、274条实名资料及信息时钟。\n"
        "4. raw目录：28份年报/中报、四季报、分红与官方政策原文；商业新闻和调查全文仅本地保存，见来源摘要。\n"
        "5. source_plan.json、january_replay_plan.json及evidence目录：原计划、用户纠偏、口径修正和旧裁决。\n"
        "6. research目录中三份脚本及审阅提示词：重画与离线复核。\n\n"
        "离线复核：在解压根目录运行 `python research/participant_delivery_v1.py verify --root .`；依赖pandas、numpy、pyarrow。"
        "重画用 `python research/participant_january_replay_v1.py plots`，另需matplotlib、pypdfium2、pdfplumber、requests。"
        "验证不联网、不拟合模型、不运行账户。前序研究只附裁决记录，不能声称本包重新复现了其完整实验。\n", encoding="utf-8")
    print("结论、任务书、来源摘要和交付导航已保存", flush=True)


def verify(root):
    out = Path(root).resolve() / REL
    count = 0
    def check(value, label):
        nonlocal count
        count += 1
        if not bool(value):
            raise AssertionError("保存数值复核失败：" + label)
    def near(actual, expected, label, tol=1e-7):
        check(np.isclose(actual, expected, rtol=0, atol=tol), label)

    a = pd.read_parquet(out / "facts/持有结构与公开时钟.parquet")
    holders = pd.read_parquet(out / "facts/实名持有人逐项记录.parquet")
    check(len(a) == 28 and len(holders) == 274, "持有记录规模")
    check(a.period_end.is_unique and a.key.is_unique, "所属期与来源唯一")
    calendar = pd.read_parquet(out / "inputs/calendar.parquet")
    days = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open.astype(bool), "trade_date"]))
    for r in a.itertuples():
        check(digest(out / r.source_pdf) == r.source_sha256, "原PDF " + r.key)
        text = re.sub(r"\s+", "", "".join(read(out / "raw" / (r.key + "_pages.json"))))
        match = re.findall(r"本报告期期末基金份额总额([\d,]+\.\d{2})", text)
        check(len(set(match)) == 1, "总份额原文唯一")
        near(float(match[0].replace(",", "")), r.total_units, "原文总份额", .01)
        near(r.institution_excluding_feeder_units + r.direct_personal_units + r.feeder_units,
             r.total_units, "三类不重叠加总", .01)
        for v, p in [(r.reported_institution_units, r.reported_institution_percent),
                     (r.direct_personal_units, r.direct_personal_percent), (r.feeder_units, r.feeder_percent)]:
            near(100*v/r.total_units, p, "原比例舍入", .0051)
        h = holders[holders.key.eq(r.key)]
        check(len(h) == r.named_holder_count, "实名条数")
        for z in h.itertuples():
            near(100*z.units/r.total_units, z.reported_percent, "实名比例", .0051)
            check(f"{z.units:,.2f}" in text, "实名份额见原文")
        exact = h[h.name.isin(["中央汇金投资有限责任公司", "中央汇金资产管理有限责任公司"])]
        near(exact.units.sum(), r.huijin_direct_lower, "两家实名合计", .01)
        check(r.huijin_direct_lower <= r.huijin_direct_upper <= r.institution_excluding_feeder_units, "汇金库存界限")
        public = max(pd.Timestamp(r.catalogue_date), pd.Timestamp(r.cover_send_date))
        check(str(days[days > public][0].date()) == r.first_available_session, "下一交易日")
        check((public - pd.Timestamp(r.period_end)).days == r.publication_lag_days, "披露滞后")
    check(a.iloc[-1].named_holder_count == 0 and pd.isna(a.iloc[-1].huijin_asset_units), "中报身份未填零")
    check((a.clock_status == "NO_VIEW_LATER_PDF_METADATA").sum() == 1, "保留一个元数据时钟疑点")
    for receipt in ["initial_inputs.json", "本轮输入快照.json"]:
        for r in read(out / "receipts" / receipt)["files"]:
            check(digest(out / r["path"]) == r["sha256"], "输入快照")
    q = read(out / "receipts/2025Q4_original.json")
    check(digest(out / "raw/2025Q4_original.pdf") == q["sha256"], "四季报继承身份")
    for r in read(out / "receipts/context_sources.json"):
        if r.get("raw_path") and not r["local_only_full_text"]:
            check(digest(out / r["raw_path"]) == r["sha256"], "背景原文")

    price = pd.read_parquet(out / "inputs/daily_price_volume.parquet").sort_values("date")
    shares = pd.read_parquet(out / "inputs/daily_fund_shares.parquet").sort_values("date")
    market = pd.read_parquet(out / "inputs/market.parquet").sort_values("date")
    case = pd.read_parquet(out / "facts/2026年1月行为回放全日线.parquet")
    for f in [price, shares, market, case]:
        f["date"] = pd.to_datetime(f.date)
        check(f.date.is_unique, "日频日期唯一")
    price["ratio"] = price.etf_amount / price.etf_amount.shift(1).rolling(20).median()
    shares["delta"] = shares.fund_shares.diff()
    merged = price.merge(shares[["date", "fund_shares", "delta"]], on="date").merge(market, on="date")
    full = merged[merged.date.between("2025-12-01", "2026-02-27")]
    check(len(case) == 57 and case.date.tolist() == full.date.tolist(), "全案例日期未抽样")
    base = merged.loc[merged.date.eq("2025-12-31")].iloc[0]
    for c, r in zip(case.itertuples(), full.itertuples()):
        near(c.etf_amount, r.etf_amount, "成交额")
        near(c.amount_to_prior20_median, r.ratio, "量能比较不含当日")
        near(c.units_change, r.delta, "份额差分", .01)
        near(c.total_return_rebased100, 100*r.wealth/base.wealth, "含分红指数")
        near(c.raw_price_rebased100, 100*r.close/base.close, "原价指数")
        check(not c.share_stat_date_is_publication_time, "份额时钟未冒充")
    s = read(out / "january_case_summary.json")
    at = lambda date: merged.loc[merged.date.eq(date)].iloc[0]
    j13, j16, j19, j23, j28 = [at(d) for d in ["2026-01-13", "2026-01-16", "2026-01-19", "2026-01-23", "2026-01-28"]]
    known = a[a.period_end.eq("2025-06-30")].iloc[0]
    check(known.publication_date_upper_bound == "2025-08-30", "一月背景使用此前中报")
    near(s["jan23_lower_bound_reduction_since_prior_disclosure_units_100m"],
         (known.huijin_direct_lower-j23.fund_shares)/1e8, "库存减少下界")
    near(s["jan19_total_return"], (j19.close+j19.dividend)/j16.close-1, "除息收益")
    near(s["jan19_dividend_per_unit"], .123, "分红金额")
    near(s["jan14_through_jan28_statistical_unit_change_100m"], (j28.fund_shares-j13.fund_shares)/1e8, "整段份额变化")
    near(s["jan13close_to_jan28close_total_return"], j28.wealth/j13.wealth-1, "整段总回报")
    stages = read(out / "facts/一月逐节点信息与可反驳推断.json")
    check([x["first_conservative_open"] for x in stages] == ["2026-01-15", "2026-01-19", "2026-01-23", "2026-01-27"], "信息更新节点")
    summary = read(out / "summary.json")
    check(summary["new_model_fits"] == summary["account_runs"] == summary["independent_forward_events"] == 0,
          "未冒充预测与账户完成")
    check(not summary["whole_objective_complete"], "整体目标未完成")
    for r in read(out / "receipts/图形与原表目视核对.json")["files"]:
        check(digest(out / r["path"]) == r["sha256"], "已目视图形身份")
    result = {"status": "PASS_SAVED_PARTICIPANT_CASE_RECOMPUTATION", "checks": count,
              "holder_reports": len(a), "holder_rows": len(holders), "case_rows": len(case),
              "new_downloads": 0, "new_model_fits": 0, "new_accounts": 0,
              "historical_first_versions_authenticated": False}
    return result


def package():
    result = verify(ROOT)
    dump(OUT / "saved_verification.json", result)
    mapping = {}
    excluded = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file():
            continue
        if "raw_local_only" in p.parts or p.name == "delivery_receipt.json":
            excluded.append({"path": p.relative_to(ROOT).as_posix(), "reason": "商业原文仅本地研究或包外最终回执", "sha256": digest(p)})
        else:
            mapping[p.relative_to(ROOT).as_posix()] = p.read_bytes()
    for name in ["participant_identity_clock_v1.py", "participant_january_replay_v1.py", "participant_delivery_v1.py"]:
        mapping["research/"+name] = (ROOT/"research"/name).read_bytes()
    mapping["README.md"] = ("# 本轮 GPT 审阅包\n\n请从 `" + REL.as_posix() + "/交付导航.md` 开始。\n\n"
                             "本包范围是一月资金行为案例、持有验证资料与下一步任务定义，非已建立的策略。商业新闻和调查全文未收入；所有核心一月数量结论有独立数据和官方报告支撑。"
                             "原数据获取时间不等于历史首次公开时间。旧研究裁决仅作背景；没有外部审阅、实盘或达标证明。\n").encode("utf-8")
    mapping["EXCLUSIONS.json"] = (json.dumps(excluded, ensure_ascii=False, indent=2)+"\n").encode("utf-8")
    index = io.StringIO(newline="")
    writer = csv.writer(index)
    writer.writerow(["path", "bytes", "sha256"])
    for name, data in sorted(mapping.items()):
        writer.writerow([name, len(data), hashlib.sha256(data).hexdigest()])
    mapping["FILE_INDEX.csv"] = index.getvalue().encode("utf-8-sig")
    target = ROOT / "deliverables" / ZIP_NAME
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError("本轮审阅包已存在，不覆盖")
    building = target.with_suffix(".building.zip")
    with zipfile.ZipFile(building, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(mapping.items()):
            archive.writestr(name, data)
    extract = ROOT / "reports/research/510300_participant_delivery_checks" / datetime.now().strftime("%Y%m%d_%H%M%S")
    extract.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(building) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("压缩包CRC或重复成员失败")
        rows = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if set(names) != {r["path"] for r in rows} | {"FILE_INDEX.csv"}:
            raise ValueError("文件索引覆盖不一致")
        for r in rows:
            data = archive.read(r["path"])
            if len(data) != int(r["bytes"]) or hashlib.sha256(data).hexdigest() != r["sha256"]:
                raise ValueError("文件索引大小或哈希失败")
        for name in names:
            dest = (extract/name).resolve()
            if not dest.is_relative_to(extract.resolve()):
                raise ValueError("包成员路径越界")
        archive.extractall(extract)
    completed = subprocess.run([sys.executable, str(extract/"research/participant_delivery_v1.py"), "verify", "--root", str(extract)],
                               check=True, capture_output=True, text=True, encoding="utf-8")
    extracted_result = json.loads(completed.stdout)
    if extracted_result != result:
        raise ValueError("新解压目录数值复核不一致")
    building.replace(target)
    receipt = {"created_at": stamp(), "status": "PASS_STRUCTURAL_AND_SAVED_PARTICIPANT_RECOMPUTATION",
               "zip": target.relative_to(ROOT).as_posix(), "sha256": digest(target), "bytes": target.stat().st_size,
               "members": len(mapping), "indexed_members": len(mapping)-1,
               "saved_verification": result, "fresh_extracted_verification": extracted_result,
               "excluded_count": len(excluded), "external_review_completed": False, "whole_objective_complete": False}
    dump(OUT / "delivery_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def finalize():
    receipt = read(OUT / "delivery_receipt.json")
    path = ROOT / "reports/research/510300_macro_research_program_status_v1.json"
    program = read(path)
    program.update(
        program_status="ACTIVE_CAPITAL_BEHAVIOR_AND_MACRO_DYNAMIC_HOLDING_RESEARCH",
        user_scope="510300宏观政策、预期与微观资金行为如何改变未来持有价值；预测大资金的约束、持续操作及市场承接，公募、私募、国家队、社保和个人资金分层，延迟身份资料作验证。",
        latest_user_objective=read(OUT / "source_plan.json")["user_objective"],
        whole_objective_complete=False, latest_completed_study="510300_PARTICIPANT_IDENTITY_CLOCK_V1",
        latest_review_zip=receipt["zip"], latest_review_zip_sha256=receipt["sha256"],
        latest_delivery_status=receipt["status"], latest_saved_verification_checks=receipt["saved_verification"]["checks"],
        current_view="NO_VIEW_NO_VALIDATED_CAPITAL_BEHAVIOR_RULE", current_view_is_cash_target=False,
        next_priority="依据已知政策约束、资金操作意向和价量承接，预测资金持续或衰减及后续持有价值。定期持有表作为完成的验证附录，不等待实名确认才预测。先冻结一项独立通道和一个承接交互的数值协议；原日频份额/季度净流入/融资/跨ETF失败不重命名救援。",
        capital_behavior_case={"days": 57, "holder_reports": 28, "named_holder_rows": 274,
                               "new_models": 0, "new_accounts": 0, "independent_forward_events": 0,
                               "case_is_ex_post": True, "numerical_protocol_frozen": False},
        target_reporting={"joint_target": "net_sharpe>=1.2 AND cagr>=0.10", "alternative_sharpe_target": 1.5,
                          "alternative_requires_full_return_and_risk_reporting": True},
        previous_lpr_expectation_source_work="PARTIAL_PRESERVED_CURRENT_PRIORITY_REDIRECTED_BY_USER")
    program["remaining_requirements"] = [
        "资金行为主线：选定一个资金约束或独立操作意向通道，以及一个供给与承接交互，冻结数值模型前说明新增信息和继承的历史选择",
        "持续预测资金供给/需求及未来持有收益和风险；日常与新公告更新，记录失效条件；主体实名未确认不阻断及时行为推断",
        "价格量能可当时观察；实际申赎、份额、意向调查只按各自已公开时间准入，字段缺口仅约束相应模块",
        "宏观政策与预期研究保留；原LPR、新订单、财政、净申购、融资方向等具体失败不反号或改窗口救援",
        "有可靠增量后进行连续账户及上涨后回撤退出对照，20万元主账户/2万元成本对照，基础与压力成本，242日口径",
        "分别报告净夏普1.2且年化10%的联合目标，以及净夏普1.5方案的实际收益与风险；独立前向证据从冻结后的新数据累计"
    ]
    entry = {"id": "510300_PARTICIPANT_IDENTITY_CLOCK_V1", "status": read(OUT / "summary.json")["status"],
             "scope": "28份原始报告、274条实名资料；2025-12至2026-02完整57日一月案例与逐节点可反驳推断。用户要求预测优先，持有报告为验证层。",
             "new_model_fits": 0, "accounts": "NOT_RUN_CASE_AND_TASK_DEFINITION_ONLY", "whole_objective_complete": False}
    program["completed_substudies"] = [x for x in program["completed_substudies"] if x["id"] != entry["id"]]+[entry]
    dump(path, program)
    status = ROOT / "RESEARCH_STATUS.md"
    line = "> 2026-09-22 资金行为预测主线纠偏与一月案例完成：`510300_PARTICIPANT_IDENTITY_CLOCK_V1`。按用户要求，延后持仓转为验证层，优先预测大资金持续操作与市场承接。保存28份年报/中报、274条实名资料和完整57日案例；1/15、1/16成交额均约前20日中位数6.46倍，1/14—1/28份额减少373.15亿份但含分红回报约-0.99%；1/19除息0.123元已校正。国家队减持有联合证据，未据此宣称后市必跌。新模型/账户/前向均0，下一步任务已重订但数值协议未冻结；整体资金与宏观目标保持ACTIVE，旧失败和85/15终止保留。见[研究结论](reports/research/510300_participant_identity_clock_v1/研究结论.md)、[预测任务书](reports/research/510300_participant_identity_clock_v1/资金行为预测任务书.md)及[审阅包](deliverables/" + ZIP_NAME + ")。\n\n"
    old = status.read_text(encoding="utf-8")
    if not old.startswith(line):
        status.write_text(line + old, encoding="utf-8")
    print("主线状态已更新，整体目标保持进行中", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本轮研究保存数据复核与交付")
    parser.add_argument("action", choices=["prepare", "verify", "package", "finalize"])
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.action == "verify":
        print(json.dumps(verify(args.root), ensure_ascii=False), flush=True)
    else:
        globals()[args.action]()
