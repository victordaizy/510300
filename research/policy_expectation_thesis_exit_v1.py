"""双期限预期兑现与支持持仓取消：隔离、一次性探索实验。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import FunctionType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

from research import shared_cash_source_ownership_study_v1 as parent
from research import source_expectation_transmission_review_v1 as facts

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_policy_expectation_thesis_exit_v1"
STATE = facts.STATE
SURVEYS = ROOT / "reports/research/510300_lpr_expectation_source_extension_v2/全部84月事前预期识别.csv"
REGISTER, DECISION = "TECH.R245", "TECH.R246"
POLICY = "COMPLEMENT_DUAL_CREDIT_UNDERDELIVERY_EXIT"
BASELINE = parent.rules.POLICIES[0]
EXIT_REASON = "双期限信用预期兑现不足_支持持仓取消"
SOURCES = (
    ("OCBC_2018_OUTLOOK", "2018-12-31", "2019展望的中国章节", "https://www.ocbc.com/assets/pdf/regional%20focus/global%20outlook/ocbc%20global%20outlook%202019.pdf"),
    ("OCBC_20190107", "2019-01-07", "降准后流动性传导周报", "https://www.ocbc.com/assets/pdf/regional%20focus/china/week%20in%20review/2019/week%20in%20review%2007jan19.pdf"),
    ("ING_20230613", "2023-06-13", "降息后MLF与LPR跟进预期", "https://think.ing.com/snaps/china-cuts-rates-ahead-of-monthly-data-dump/"),
    ("MUFG_20230613", "2023-06-13", "后续政策利率预期及传导限制", "https://www.mufgresearch.com/fx/fx-special-focus-230613/"),
)
OLD_LPR = ROOT / "reports/research/510300_lpr_expectation_exploratory_training_v1/summary.json"
OLD_JOINT = ROOT / "reports/research/510300_lpr_joint_response_20d_v1/results/summary.json"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def rel(path):
    return path.absolute().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(frame, name):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".parquet", ".csv"):
        if path.with_suffix(suffix).exists():
            raise FileExistsError("禁止覆盖已有结果：" + name)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def tenor_state(row, prefix):
    """区间一侧足以证明方向；不能为缺失端点补零或取中点。"""
    lo, hi = row[prefix + "_surprise_min_bp"], row[prefix + "_surprise_max_bp"]
    if pd.notna(lo) and pd.notna(hi) and float(lo) > float(hi):
        raise ValueError("兑现差异区间上下界颠倒。")
    if pd.notna(lo) and float(lo) > 0:
        return "TIGHTER"
    if pd.notna(hi) and float(hi) < 0:
        return "EASIER"
    if pd.notna(lo) and pd.notna(hi) and float(lo) == 0 and float(hi) == 0:
        return "MATCH"
    return "UNIDENTIFIED"


def joint_state(first, fifth):
    """两个贷款期限共同判断；不设置大小阈值、权重或线性分数。"""
    if "UNIDENTIFIED" in (first, fifth):
        return "ABSTAIN_UNIDENTIFIED_TENOR"
    if {first, fifth} == {"TIGHTER", "EASIER"}:
        return "MIXED_DIRECTION_NO_CANCELLATION"
    if "TIGHTER" in (first, fifth) and {first, fifth} <= {"MATCH", "TIGHTER"}:
        return "DUAL_CREDIT_UNDERDELIVERY"
    if "EASIER" in (first, fifth) and {first, fifth} <= {"MATCH", "EASIER"}:
        return "DUAL_CREDIT_OVERDELIVERY"
    if first == fifth == "MATCH":
        return "MATCHED_BOTH_TENORS"
    raise ValueError("未知期限状态。")


def classify_surveys(frame):
    result = frame.copy()
    if result.month.duplicated().any():
        raise ValueError("月份母集重复。")
    states, firsts, fifths, valid = [], [], [], []
    for row in result.to_dict("records"):
        admitted = row["status"] == "ADMITTED_SAVED_HISTORICAL_SURVEY"
        clock = False
        if admitted:
            dates = [pd.Timestamp(row[k]) for k in ("announcement_at", "survey_published_at", "survey_modified_at")]
            if any(pd.isna(x) or x.tzinfo is None for x in dates):
                raise ValueError("准入调查缺少带时区的公布/版本时钟。")
            clock = dates[1] < dates[0] and dates[2] < dates[0]
        if not admitted or not clock:
            first = fifth = "UNIDENTIFIED"
            state = "ABSTAIN_MISSING_SURVEY" if not admitted else "ABSTAIN_NON_PRIOR_VERSION"
        else:
            first, fifth = tenor_state(row, "t1"), tenor_state(row, "t5")
            state = joint_state(first, fifth)
        firsts.append(first)
        fifths.append(fifth)
        states.append(state)
        valid.append(clock)
    result["t1_bound_state"] = firsts
    result["t5_bound_state"] = fifths
    result["joint_credit_state"] = states
    result["same_instrument_prior_version_clock_valid"] = valid
    result["historical_first_vintage_independently_authenticated"] = False
    return result


def attach_events(data, surveys):
    """只在公告可知后的第一个收盘产生一次事件；不跨日无限沿用。"""
    out = data.copy()
    out["credit_cancellation_event"] = False
    out["credit_review_month"] = ""
    out["credit_joint_state"] = "NO_NEW_LPR_REVIEW"
    out["credit_announcement_at"] = ""
    close_times = pd.DatetimeIndex(out.date).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15, minutes=5)
    used = set()
    for row in surveys.to_dict("records"):
        at = pd.Timestamp(row["announcement_at"])
        if at.tzinfo is None:
            raise ValueError("官方公告时钟没有时区。")
        eligible = np.flatnonzero(close_times >= at)
        if not len(eligible):
            continue
        idx = int(eligible[0])
        if idx in used:
            raise ValueError("同一个收盘对应多个月度政策版本，不能覆盖。")
        used.add(idx)
        out.loc[out.index[idx], ["credit_review_month", "credit_joint_state", "credit_announcement_at"]] = [row["month"], row["joint_credit_state"], row["announcement_at"]]
        out.loc[out.index[idx], "credit_cancellation_event"] = row["joint_credit_state"] == "DUAL_CREDIT_UNDERDELIVERY"
    return out


class SupportBridge:
    """仅为当前实例接入支持持仓失效信息；不修改冻结模块全局变量。"""
    def __getattr__(self, name):
        return getattr(parent.rules.support, name)

    def exit_decision(self, active, row, idx, policy):
        if active["owner"] == "COMPLEMENT" and bool(row.credit_cancellation_event):
            return EXIT_REASON
        return parent.rules.support.exit_decision(active, row, idx, policy)


class RulesBridge:
    support = SupportBridge()

    def __getattr__(self, name):
        return getattr(parent.rules, name)


def account(data, dividends, parents, risks, cost, start):
    original = parent.execution.account
    if original.__closure__ is not None:
        raise RuntimeError("原账户新增了闭包，必须先解释兼容性。")
    namespace = dict(original.__globals__)
    namespace["rules"] = RulesBridge()
    isolated = FunctionType(original.__code__, namespace, "隔离预期兑现账户", original.__defaults__)
    return isolated(data, dividends, parents, risks, BASELINE, cost, start)


def source_files():
    return [Path(__file__), SURVEYS, OLD_LPR, OLD_JOINT, parent.OUT / "summary.json", parent.OUT / "protocol.json",
            Path(parent.__file__), Path(parent.execution.__file__), Path(parent.rules.__file__),
            ROOT / "tests/test_policy_expectation_thesis_exit_v1.py", OUT / "tests_receipt.json",
            *parent.source_paths()]


def register():
    if (OUT / "protocol.json").exists():
        raise FileExistsError("本实验已经登记。")
    state = read(STATE)
    tests = read(OUT / "tests_receipt.json")
    if tests["exit_code"] != 0 or tests["passed"] != 8 or tests["code_sha256"] != digest(Path(__file__)):
        raise ValueError("八项必要测试未通过或版本不一致。")
    write(OUT / "protocol.json", {
        "at": now(), "registration": REGISTER, "decision": DECISION, "status": "ONE_COMPLETE_EXPLORATORY_PURPOSE_REGISTERED",
        "hypothesis": "支持性信息与价格接受产生的持仓，在新公布的两个贷款期限共同明确少于事前预期时，其信用宽松论据减弱；一次取消支持持仓可能提高同资金完整账户的净收益和Sharpe。",
        "different_use": "旧LPR五模型用调查/实际/幅度预测20日收益并受误差阈值决定进入；旧联合四模型用实际LPR与股债反应预测均值/下行方差。本实验无训练/新进入/幅度阈值，只对共同现金账户中实际归属COMPLEMENT的当前持仓做新消息的一次失效。",
        "old_family_terminal_statuses_retained": {"expectation_training": read(OLD_LPR)["status"], "joint_information_verdicts": read(OLD_JOINT)["verdicts"]},
        "historical_outcomes_already_seen": "R240全部周期、五补充盈亏，84月调查含2023-06及2022/2023-08方向、旧LPR训练/联合结果和本轮原网页均已查看；不宣称预注册盲测或独立验证。",
        "web_preflight": {"search_queries": 6, "open_calls": 4, "find_targets": 4, "followup_open_targets": 4, "role": "浏览探索，不等于本机归档请求为零"},
        "source_requests": [{"id": sid, "internal_date": date, "title": title, "url": url} for sid, date, title, url in SOURCES],
        "archive_rule": "4个固定原作者地址各一次GET；无重试。新增银行研究只解释机制；内部日期不认证历史公开上界，不进入本次数值模型。",
        "numeric_source": "复用84月完整母集/56保存调查，含28缺失和不能唯一识别的区间；两个期限和原公布/修订钟原样保留；不把LPR调查传给7天逆回购。",
        "primary": POLICY, "new_primary_full_accounts": 4, "periods": parent.PERIODS, "costs": parent.COSTS,
        "saved_controls": [BASELINE, parent.SAVED_CORE], "saved_control_accounts": 8,
        "state_rule": "至少一期限区间能严格证明兑现偏紧，另一期限为一致或也偏紧；方向混合、未识别/缺失均不取消。零仅是方向边界，无参数扫描。",
        "action": "仅公告可知后的第一个15:05收盘事件；只有当时COMPLEMENT真实持仓才请求全部退出，下一实际开盘依原T+1/涨跌停/整手/最低佣金/现金规则执行；被阻退出锁定。只在该原周期取消，不制造新入场、持久宏观禁入或同开盘重入。",
        "unchanged": "原CORE目标/支持接受事件与失效、风险预算、20万元共同现金、历史分红、两时期两费用、费用后账户口径不变；退出改变之后的现金/数量由完整账户自然反馈。",
        "check": "关闭新事件必须精确复现R240四账户daily/orders/trades；完整八项必要测试覆盖区间、两期限、时钟、事件不沿用、原模块隔离和CORE不取消。",
        "judgement": "所有四场景完整披露；近期BASE/STRESS对原共同账户及原A净CAGR与Sharpe均须提高、净pB>1、标准EV>0、DD<=10%，逐年/周期/来源集中度同时报告；早期缺少新数据不能称有效适应，独立样本0禁止推广。",
        "uncertainty": "真实因果、概率校准与历史原首版未建立；这是机制一致的探索退出，不能证明政策兑现差异造成股价下跌。",
        "no_rescue": "固定一次主用途；不改一期/五期优先级、阈值、窗口、份额、旧训练、旧R240或分期拼接。",
        "frozen_files": [{"path": rel(p), "sha256": digest(p)} for p in source_files()],
        "forward_before": {k: state[k] for k in facts.FORWARD}, "financial_before": {k: state[k] for k in facts.FINANCE},
        "new_fits": 0, "new_equity_labels": 0, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    })
    print("R245唯一双期限兑现退出用途已登记；四新账户、八保存对照，旧终态保持。", flush=True)


def fetch(source):
    sid, date, title, url = source
    folder = OUT / "sources" / sid
    folder.mkdir(parents=True, exist_ok=False)
    receipt = {"id": sid, "url": url, "internal_date": date, "title": title, "started_at": now(),
               "retries": 0, "historical_publication_upper": "NOT_ESTABLISHED", "first_vintage": "NOT_AUTHENTICATED", "numeric_model_role": "EXCLUDED_BACKGROUND_ONLY"}
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        response = session.get(url, timeout=(8, 25))
        raw = response.content
        (folder / "response.body").write_bytes(raw)
        receipt.update(status=response.status_code, final_url=response.url, bytes=len(raw), sha256=digest(folder / "response.body"), captured_at=now(), content_type=response.headers.get("Content-Type", ""))
        if response.status_code == 200:
            if raw.startswith(b"%PDF"):
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    selected = list(range(11, 13)) if sid == "OCBC_2018_OUTLOOK" else list(range(len(pdf.pages)))
                    text = "\n\n".join(f"第{i+1}页\n{pdf.pages[i].extract_text() or ''}" for i in selected)
                receipt["extracted_pages_one_based"] = [i+1 for i in selected]
            else:
                soup = BeautifulSoup(raw, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text("\n", strip=True)
            (folder / "text.txt").write_text(text, encoding="utf-8")
            receipt["text_sha256"] = digest(folder / "text.txt")
    except (requests.RequestException, ValueError, OSError) as error:
        receipt.update(error_type=type(error).__name__, error=str(error), failed_at=now())
    finally:
        session.close()
        write(folder / "receipt.json", receipt)
    return {"id": sid, "status": receipt.get("status", "FAILED"), "bytes": receipt.get("bytes"), "error_type": receipt.get("error_type")}


def collect():
    read(OUT / "protocol.json")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, SOURCES):
            print(json.dumps(result, ensure_ascii=False), flush=True)


def frozen_exact():
    protocol = read(OUT / "protocol.json")
    for row in protocol["frozen_files"]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("已冻结文件改变：" + row["path"])
    state = read(STATE)
    for names, key in ((facts.FORWARD, "forward_before"), (facts.FINANCE, "financial_before")):
        if {k: state[k] for k in names} != protocol[key]:
            raise ValueError("原独立前瞻或金融身份改变。")
    return protocol


def run():
    protocol = frozen_exact()
    write(OUT / "run_started.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json")})
    for source in SOURCES:
        read(OUT / "sources" / source[0] / "receipt.json")
    surveys = classify_surveys(pd.read_csv(SURVEYS))
    if len(surveys) != 84 or int(surveys.same_instrument_prior_version_clock_valid.sum()) != 56:
        raise ValueError("原调查全体/钟范围不一致。")
    data, _, dividends, risks, parents = parent.load()
    data = attach_events(data, surveys)
    export(surveys, "全部84月_双期限兑现与缺失不删")
    export(data[["date", "credit_review_month", "credit_joint_state", "credit_announcement_at", "credit_cancellation_event"]], "全部日线_一次可知事件而非持续禁入")
    control_checks, metrics, cycle_tables, order_tables, yearly, cancellation = [], [], [], [], [], []
    for period, (start, end) in parent.PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        for cost in parent.COSTS:
            folder = parent.OUT / "accounts" / period / cost / BASELINE
            saved = {name: pd.read_parquet(folder / f"{name}.parquet") for name in parent.ACCOUNT_TABLES}
            saved["terminal"] = read(folder / "terminal.json")
            muted = local.copy()
            muted["credit_cancellation_event"] = False
            control = account(muted, dividends, parents[period], risks, cost, start)
            for name in parent.ACCOUNT_TABLES[:3]:
                parent.previous.exact_saved_table(control[name], saved[name])
            control_checks.append({"period": period, "cost": cost, "R240_muted_event_account_exact": True})
            candidate = account(local, dividends, parents[period], risks, cost, start)
            verification = parent.previous.parent.verify_account(candidate)
            destination = OUT / "accounts" / period / cost / POLICY
            destination.mkdir(parents=True, exist_ok=False)
            for name in parent.ACCOUNT_TABLES:
                candidate[name].to_parquet(destination / f"{name}.parquet", index=False)
            write(destination / "terminal.json", candidate["terminal"])
            write(destination / "verification.json", verification)
            stats, years = parent.previous.controls.statistics(candidate, period, cost, POLICY)
            metric = {"period": period, "cost": cost, "policy": POLICY, **stats}
            metrics.append(metric)
            for control_policy, ctrl in ((BASELINE, saved), (parent.SAVED_CORE, parent.saved_account(period, cost, parent.SAVED_CORE))):
                control_stats, _ = parent.previous.controls.statistics(ctrl, period, cost, control_policy)
                metrics.append({"period": period, "cost": cost, "policy": control_policy, **control_stats})
            cycle_tables.append(candidate["trades"].assign(period=period, cost=cost, policy=POLICY))
            order_tables.append(candidate["orders"].assign(period=period, cost=cost, policy=POLICY))
            requests_frame = candidate["decisions"]
            for row in requests_frame.loc[requests_frame.reason.eq(EXIT_REASON)].to_dict("records"):
                cancellation.append({"period": period, "cost": cost, **row})
            yearly.extend(years)
            print(f"{period}/{cost}唯一新账户已完成，{len(requests_frame.loc[requests_frame.reason.eq(EXIT_REASON)])}次消息取消请求。", flush=True)
    export(pd.DataFrame(metrics), "全部12完整账户_四新与八保存对照")
    export(pd.concat(cycle_tables, ignore_index=True), "全部新账户周期与开放_费用复本非独立")
    export(pd.concat(order_tables, ignore_index=True), "全部真实订单_完整现金反馈")
    export(pd.DataFrame(cancellation), "全部取消请求_真实来源归属与执行钟")
    export(pd.DataFrame(yearly), "逐年完整净收益_缺失年不删")
    write(OUT / "control_verification.json", {"at": now(), "all_four_muted_accounts_exact": control_checks})
    summary = {"at": now(), "registration": REGISTER, "decision": DECISION, "status": "COMPLETED_FIXED_EXPLORATORY_FINANCIAL_EXPERIMENT_NOT_YET_ADJUDICATED",
               "new_candidate_accounts": 4, "muted_adapter_check_accounts": 4, "saved_control_accounts": 8, "new_fits": 0, "new_equity_labels": 0,
               "all_months": len(surveys), "saved_prior_surveys": int(surveys.same_instrument_prior_version_clock_valid.sum()),
               "joint_states": surveys.joint_credit_state.value_counts().to_dict(),
               "cancellation_event_months": surveys.loc[surveys.joint_credit_state.eq("DUAL_CREDIT_UNDERDELIVERY"), "month"].tolist(),
               "primary_metrics": [m for m in metrics if m["policy"] == POLICY],
               "archive_requests": 4, "archive_receipts": [read(OUT / "sources" / source[0] / "receipt.json") for source in SOURCES],
               "new_bank_reports_in_numeric_model": 0, "historical_source_first_versions_authenticated": False,
               "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "forward_before": protocol["forward_before"]}
    write(OUT / "financial_raw_summary.json", summary)
    print("四个唯一新账户、八原对照和全部月/日/周期/取消请求已完整保存。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="双期限预期兑现与支持持仓一次失效研究")
    parser.add_argument("action", choices=("register", "collect", "run"))
    args = parser.parse_args()
    {"register": register, "collect": collect, "run": run}[args.action]()


if __name__ == "__main__":
    main()
