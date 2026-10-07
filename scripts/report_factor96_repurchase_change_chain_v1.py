"""写出回购变更来源整理结果与后续条件，不追加策略账户。"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_repurchase_change_chain_v1"
DAILY = ROOT / "reports/research/510300_factor96_daily_state_shrink_v1"
STATUS = ROOT / "reports/research/510300_factor96_program_v1/status.json"
MANDATE = ROOT / "config/510300_existing_data_training_mandate_v1.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value, exclusive=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "round_status.json").exists()
    result, ledger = read(OUT / "result.json"), read(OUT / "change_ledger.json")
    program, mandate = read(STATUS), read(MANDATE)
    assert program == read(OUT / "inputs/program_before.json")
    assert mandate == read(OUT / "inputs/mandate_before.json")
    command = [sys.executable, "-X", "utf8", str(ROOT / "scripts/verify_factor96_repurchase_change_chain_v1.py"), "--root", str(OUT)]
    process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
    assert process.returncode == 0, process.stdout + process.stderr
    verified = json.loads(process.stdout.strip().splitlines()[-1])
    save(OUT / "saved_verification_receipt.json", verified)
    prior_files = []
    for name in ["result.json", "metrics.csv", "delivery_receipt.json"]:
        source, target = DAILY / name, OUT / "prior_account_evidence" / name
        target.parent.mkdir(exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)
        assert digest(source) == digest(target)
        prior_files.append({"path": target.relative_to(OUT).as_posix(), "bytes": target.stat().st_size,
                            "sha256": digest(target), "original_path": source.relative_to(ROOT).as_posix()})
    save(OUT / "prior_account_evidence/manifest.json", {"files": prior_files, "accounts_replayed": 0})
    annual = {}
    for key, rows in [("all_candidates", ledger), ("reviewed", [r for r in ledger if r["action"] != "UNRESOLVED"]),
                      ("confirmed_root_documents", [r for r in ledger if r["confirmed_target_roots"]])]:
        annual[key] = dict(sorted(Counter(r["known_at"][:4] for r in rows).items()))
    save(OUT / "calendar_scope.json", {"annual_document_counts": annual,
         "known_at_is_proven_first_publication": False,
         "primary_window": "2021-01-01/2025-12-31", "allow_2026_notices_backfilled_into_primary_window": False,
         "comment": "按既有保守公告日期计数，不证明首发历史版本；2026只作后期字段研究。"})
    missing = [{"change_document_id": r["document_id"], "symbol": r["symbol"],
                "known_at": r["known_at"], "target": t,
                "next_action": "先查已有元数据与原文；仅当出现可定位的缺失原方案，再执行有限公开补证。"}
               for r in ledger for t in r.get("targets", []) if t["original_id"] is None]
    save(OUT / "identified_missing_originals.json", missing)
    clock = datetime.now().astimezone().isoformat()
    round_status = {"at": clock, "study_id": result["study_id"],
         "previous_turn_classification": "PROGRESS_BOUNDED_SOURCE_CHECK_AND_QUANTIFIED_GAPS",
         "this_turn_classification": "PROGRESS_EXPLICIT_CHANGE_TARGETS_AND_APPROVAL_STAGES",
         "progress": "54候选全保留，23份逐条复核；新增16条公告至原方案关联，排除4个仅作背景的方案引用；保留7个明确目标缺原件和同日双版本歧义。",
         "new_accounts": 0, "cumulative_admitted_account_scenarios": 432,
         "cumulative_invalid_implementation_accounts": 152, "cumulative_executed_account_scenarios": 584,
         "new_admitted_trading_features": 0, "qualified_candidates": [],
         "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW",
         "external_review": "NOT_PERFORMED", "orders_authorized": False}
    save(OUT / "round_status.json", round_status)
    accounts = read(DAILY / "result.json")["primary_accounts"]
    account_table = "\n".join(f"| {r['capital']:,} | {r['net_sharpe']:.4f} | {r['cagr'] * 100:.4f}% | {r['max_drawdown'] * 100:.4f}% |" for r in accounts)
    report = f"""# 510300：回购变更对象与审批阶段

目标尚未实现。本轮补的是来源字段：完整保留54份变更标题候选，对23份公告逐条复核，17份能够对应到15个既有原方案，相比上一版新增16条公告至原方案关联。3份公告中提到的4个既有方案实际并非变更对象，已明确排除。没有新增交易因子准入、账户或收益结果，T12继续NOT_RUN。

## 本轮实际完成

- 20份存在原方案候选关系的公告全部复核；另选3份标题含“终止”的公告核对终止对象。其余31份仍明确标为未完成完整对象复核，不能计作已复核。
- 使用80份已保存来源PDF及其逐页文本、原收据；91条短语对应135处页内定位。仅对三页存在数量、对象或预算疑点的原PDF做渲染视觉核对。
- 17份公告的确认关联覆盖15个原方案：2024年2份、2025年5份、2026年10份。23份复核卡中15份属于2026年；2026信息不会倒填2021至2025主检验窗口。
- 4份多目标公告只找齐部分原方案；总计7个已明确对象的原方案文件缺失。同一方案、同一保守时钟的泰格医药两份文件保留版本歧义，不任选一份、不重复汇总股份。
- 原有948条执行用途的569条已知、379条未知计数保持不变。本轮23份复核卡与这些执行记录不是同一口径，不能相加。

## 会改变因子含义的具体区别

| 公告案例 | 原文支持的记录 | 不具备的含义 |
|---|---|---|
| 光启技术 1222223042 | 拟变更1,258,707股，另3,818,103股保留激励用途，仍待股东会 | 跨页页码串接产生的21,258,707股；已完成注销 |
| 天山铝业 1224735615 | 实际变更第一轮2,314.80万股；第一轮原方案缺失 | 同文中第三轮2025年方案也被变更 |
| 歌尔股份 1224960309 | 回购预算5至10亿元上调至10至15亿元 | 将预算增量记作新发生的回购现金流 |
| 中科创达 1225080515、1225140967 | 董事长提议后董事会通过，仍待股东会 | 从提议日就把用途切换为已生效 |
| 泰格医药 1225525350、1225530137 | 同编号、同日两份不同文件，修改顺序未明 | 按文件编号排序任选最新值或将5,883,780股重复计算 |
| 海天味业、长城汽车、阿特斯 | 分别涉及员工持股计划或建设项目终止 | 仅凭“终止”二字认定市场回购方案停止 |

审批状态按原文保存：4份董事会通过且公告说明无需股东会；14份仍待股东会；1份董事长提议待董事会及股东会；1份还待债券持有人会议；3份是其他对象的终止；31份未完成状态复核。这些是公告条款整理，未将拟注销记成已经完成。

## 时序与复核范围

34个查询分别在相应公告保守已知时钟之前一秒、到达该时钟时读取已复核变更。其中2个查询因同一时钟多版本而返回歧义。每个查询只回答本轮已复核变更，不声明完整当前方案状态。沿用的known_at未独立证明首次历史发布时间，因此54条记录的trading_feature_admitted均为false。

冻结前9项真实边界测试通过。独立只读脚本核对259个冻结文件、80份源文件身份、54条台账、91条短语、135处页内定位和34个查询。脚本证明保存证据与输出一致，不代替全部原文语义、首发时钟、外部审阅或策略有效性验证。冻结后没有修改冻结文件。

## 与夏普目标的距离

最近一次已完成的账户检验仍是固定日更变体，主期2021至2025年、压力成本、延迟一日、完整策略。下表沿用其保存结果，本轮未重跑：

| 初始资金（元） | 成本后净夏普 | 净复合年化 | 最大回撤 |
|---|---:|---:|---:|
{account_table}

该变体未达到净夏普1.2、净年化10%的目标。累计仍为432个正式账户情景与152个否定实现情景，共584个执行情景；它们不是584种独立策略。原库完成7个固定问题，另有1个授权日更变体，合格候选0，独立前向观察0。

## 下一步与停止条件

identified_missing_originals.json列出7个可定位原方案缺口，应先查已有元数据，再按明确公告对象有限补证。同时仍需补齐后续批准、登记与修订关系、可证明的可用时钟，以及自由流通分母。缺分母或完整版本链时，不计算T12收益、不使用普通流通口径替代、不把2026材料回填历史。候选关联只有原批准日或编号相同、而对象条款不一致时，保持拒绝。不存在具体新线索时不重复相同来源轮询。

当前范围仅510300.SH与CASH_CNY研究模拟；采集计划继续PAUSED_BY_USER。当前市场NO_VIEW、外部审阅NOT_PERFORMED、订单权限false。本轮使用本地保存资料，网络请求0、收益计算0、新账户0。目标保持active，未标记完成。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    prompt = """请对这份510300本地研究包做批判性审阅。用户目标是仅510300.SH/CASH_CNY，20万元完整账户成本后净夏普至少1.2，现行净年化目标10%、目标最大回撤10%、两年训练、日更及账户尾部约束保持不变。附件96因子18策略是研究参考，不是授权或效果证明。

本轮只整理回购变更来源，T12未运行。请先核对23份复核卡对实际变更对象、原方案根、待批准状态、多方案剩余股份、预算与实际执行、终止对象的解释；重点检查3份公告中的4个背景方案是否正确排除、4份多目标公告是否被误称完整、泰格同日双文档是否应保持未知，以及2026材料是否被倒填2021至2025。

请区分：代码与保存输出一致、来源语义可信、历史首次发布时间可证、全体事件版本完整、独立预测和账户收益有效。这些层级不能互相替代。当前所有记录未准入交易因子，7个已识别目标缺原方案，31份未完整复核，自由流通分母仍缺，既有执行用途569已知/379未知口径不变。累计432正式+152否定实现=584账户情景不是独立策略数。最近固定日更变体失败已保留。

请输出具体错误及路径/页码、影响范围、优先级；再给出最小下一步和停止条件。只有当缺失数据、时序和版本满足条件，才提出一个可冻结的新假设与验证方案；不要按已有收益调参、拼接盈利区间或用现金稀释掩饰失败。这个包尚未外部审阅，也不授权上传、自动任务、Paper/Shadow、券商或订单。
"""
    (OUT / "01_GPT_REVIEW_PROMPT.txt").write_text(prompt, encoding="utf-8")
    program.update(at=clock, latest_round=result["study_id"], latest_result=(OUT / "round_status.json").relative_to(ROOT).as_posix(),
                   admitted_account_scenarios_this_round=0, invalid_implementation_accounts_this_round=0,
                   new_archived_source_http_responses_this_round=0, new_source_documents_this_round=0,
                   new_searchable_text_documents_this_round=0, reused_repurchase_pdf_documents_this_round=80,
                   source_field_candidates_this_round=91, source_field_candidate_kind="23份人工复核卡的91条原文定位短语，尚未准入交易因子",
                   new_explicit_version_links_this_round=16, new_version_roles_this_round=0,
                   latest_repurchase_change_source_check=(OUT / "result.json").relative_to(ROOT).as_posix(),
                   next_candidates=["T12_SEVEN_IDENTIFIED_MISSING_ORIGINALS_AND_FREE_FLOAT", "T13_EVENT_IDENTITIES_AND_FREE_FLOAT", "T04_HISTORICAL_WEIGHTS"])
    mandate.update(current_round=result["study_id"], current_protocol=(OUT / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(OUT / "round_status.json").relative_to(ROOT).as_posix(),
                   latest_continuation_report=(OUT / "研究结论.md").relative_to(ROOT).as_posix(),
                   latest_continuation_classification=round_status["this_turn_classification"],
                   research_execution_state="CHANGE_FIELDS_REVIEWED_NO_TRADING_FEATURE_ADMISSION",
                   last_source_result="23份复核卡新增16条公告至原方案关联，排除4个背景方案引用；完整版本链与自由流通仍缺，T12继续NOT_RUN。")
    save(STATUS, program, exclusive=False)
    save(MANDATE, mandate, exclusive=False)
    save(OUT / "program_after.json", program)
    save(OUT / "mandate_after.json", mandate)
    print(json.dumps(round_status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
