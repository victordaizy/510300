"""在新的延迟收益计算前冻结事件、时钟、比较组与停止条件。"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_post_information_capital_adjustment_v1"
PRIOR = ROOT / "reports/research/510300_expectations_policy_evidence_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError("本研究目录已存在，禁止覆盖冻结记录。")
    OUT.mkdir(parents=True)
    for directory in ("inputs", "evidence", "code", "results", "figures", "sources"):
        (OUT / directory).mkdir()
    archive = ROOT / "deliverables/510300_货币预期差与政策证据_V1_GPT审阅_20260917.zip"
    expected = "d8db078f71418244c6174bf198aa48abab55956428c1b5fadbf92d64de25ed28"
    assert digest(archive) == expected, "先前研究包身份不符。"
    copies = {
        PRIOR / "inputs/market_daily.parquet": "inputs/market_daily.parquet",
        PRIOR / "inputs/official_releases.csv": "inputs/official_releases.csv",
        PRIOR / "results/104个月信息覆盖.csv": "inputs/monthly_coverage.csv",
        PRIOR / "results/事前预期差与510300全部事件.csv": "inputs/prior_six_cases.csv",
        PRIOR / "inputs/policy_context_events.json": "inputs/policy_context_events.json",
        PRIOR / "results/510300与当时已知货币数据_全部日线.parquet": "inputs/money_market_daily.parquet",
        PRIOR / "results/政策冲击与510300全部事件.csv": "evidence/prior_policy_events.csv",
        PRIOR / "source_records/wgc_archive_inventory.json": "evidence/wgc_archive_inventory.json",
        PRIOR / "source_records/wgc_forecast_sources.json": "evidence/wgc_forecast_sources.json",
        PRIOR / "source_records/policy_data_receipt.json": "evidence/policy_data_receipt.json",
        PRIOR / "来源与未补齐部分.md": "evidence/此前来源与缺口.md",
        PRIOR / "protocol.json": "evidence/prior_protocol.json",
        PRIOR / "研究结论.md": "evidence/此前研究结论.md",
        ROOT / "config/510300_research_authority_v5.json": "evidence/research_authority_snapshot.json",
        ROOT / "reports/research/510300_nbs_fixed_5min_usage_v2_1/G2_adjudication.json": "evidence/NBS_G2_adjudication.json",
        ROOT / "docs/510300_NBS_V2_FIXED_5MIN_G2_FINAL_RESULT_20260905.md": "evidence/NBS最终结果.md",
        ROOT / "reports/research/510300_fund_share_publication_receipts_closure_v1/result.json": "evidence/fund_share_clock_closure.json",
        ROOT / "reports/research/510300_eight_round_failure_attribution_20260906/result.json": "evidence/prior_failure_attribution.json",
        ROOT / "reports/research/510300_eight_round_failure_attribution_20260906/二十四个既有候选的日收益相关性.csv": "evidence/prior_24_correlation.csv",
        ROOT / "reports/research/510300_eight_round_failure_attribution_20260906/过去排名与下一阶段表现.csv": "evidence/prior_rank_continuation.csv",
        Path("E:/CodexData/.codex/attachments/9985585b-6e77-41bf-b927-0e02209a9c57/pasted-text-1.txt"): "用户原始附件.md",
    }
    inventory = []
    for source, destination in copies.items():
        target = OUT / destination
        shutil.copy2(source, target)
        inventory.append({"source_path": str(source), "package_path": destination,
                          "sha256": digest(target), "bytes": target.stat().st_size})
    shutil.copytree(PRIOR / "source_records/official_money", OUT / "sources/official_money")
    # 只复制信息字段；原报告的已知收益另存，以免误称这些历史是未见样本。
    old = pd.read_csv(OUT / "inputs/prior_six_cases.csv")
    columns = [x for x in old.columns if not x.startswith(("return_", "end_")) and x != "base_date"]
    old[columns].to_csv(OUT / "inputs/event_information.csv", index=False, encoding="utf-8-sig")
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    protocol = {
        "study_id": "510300_POST_INFORMATION_CAPITAL_ADJUSTMENT_V1",
        "frozen_at": frozen_at,
        "user_capital_confirmation": "20 万元为主，2 万元作成本对照",
        "main_capital_cny": 200000, "cost_comparison_capital_cny": 20000,
        "scope": "信息更新后的持续调整；暂时压力回归单独保留，不混入本研究",
        "primary_question": "首轮价格反应已知后，消息及一项真实资金变量能否增量预测后续五交易日收益",
        "event_family": "央行月度金融统计首次公布的M1/M2同比配对预期差",
        "event_universe": "2018-01至2026-08全部104个所属月；2025新口径单列，缺少配对共识不补造",
        "policy_role": "八项背景事件只作图示背景；非完整政策样本，不能据此检验政策效应",
        "prior_policy_factor_status": "全样本PCA不准入逐期交易预测，保留446条原政策记录及限制",
        "observation_clock": "已知公布时刻严格早于交易日09:30则该日为首个完整日，否则取下一交易日；仅日期则按23:59:59上界",
        "observation_at": "第一完整交易日15:00收盘",
        "entry_clock": "观察结束后的下一交易日09:30开盘；不赚取此前跳空或首日反应",
        "primary_horizon_trading_days": 5,
        "fixed_exit_clock": "入场当日为第1日，第5日15:00预定收盘卖出；非看见该收盘才决定",
        "case_return": "(第5日收盘价+入场后至退出日获得的每份现金分红)/入场开盘价-1；不再投资分红；入场当日除息不享有；本六例窗口无分红",
        "comparison_clock": "同时报告原报告公告后前5日收益；另以相同新退出日拆分旧收盘到新开盘已发生收益与余下收益，避免不同终点误当分解",
        "prior_price_features": ["公告前20日总回报", "公告后首完整交易日收盘相对公告前收盘总回报", "观察日已知20日日收益标准差的对数"],
        "message_feature": "(M1实际-M1事前共识)-(M2实际-M2事前共识)，百分点；分开报告两部分",
        "fund_feature": "截至观察收盘已证明公开的最近五个交易日ETF份额净变化/期初份额；每个观察值须有所属日、发布时间上界、来源与原始内容；不以成交量替代",
        "arms": {"A": "价格三变量", "B": "A+预期差", "C": "A+份额净变化", "D": "A+预期差+份额净变化+预期差乘份额净变化"},
        "support": "仅记录公告前20交易日最低价；用公告前已知20日日收益标准差归一化距离；不增加A/D主模型特征或另选窗口",
        "model_if_admitted": "各组训练样本内标准化；带截距ridge，目标平均平方误差加1倍斜率L2范数；截距不惩罚；扩展训练，不调参",
        "data_gate": "同口径同事件类别至少60个独立事件群，前36训练后至少24评价；是预定最低可执行门，不宣称足够统计功效；共同完整样本上比较A/B/C/D",
        "fund_gate": "逐事件资金公开时点及份额口径均合格才运行C/D；不能假定固定一天延迟等于历史公开证明",
        "label_maturity": "每次训练只纳入退出时点严格早于当前观察时点的事件；重叠五日持有区间并为一事件群",
        "increment_gates": ["B对A、C对A、D对B及C全部报告，不择优隐藏", "配对MSE差大于0且事件群顺序4群移动块Bootstrap的单侧90%下界大于0；10000次固定种子5103005", "评价前后两半均有改善；不因失败改变窗口或方向", "压力成本后预测为正的入场至少12个独立事件群，均值下界大于0，单一事件正收益贡献不超过30%"],
        "entry_after_gates": "D通过主门才进入主账户；预测五日收益大于预计压力往返成本则持有，无合格模型则不产生仓位意见；重叠事件不加仓",
        "exit_comparison": "仅同一批已冻结合格入场比较固定5日与一项回撤退出；入场前20日波动sigma冻结，持有收盘峰值相对入场上涨>=2sigma后启动，从该收盘峰值回撤>=sigma时次日开盘退出，最迟仍第5日预定收盘；T+1、实际跳空、无保证止损价；该规则当前NOT_RUN",
        "account": {"assets": ["510300.SH", "CASH_CNY"], "annualization": 242,
                    "commission_rate": 0.0002, "minimum_commission_per_side_cny": 5,
                    "lot_shares": 100, "base_slippage_per_side": 0.0005, "stress_slippage_per_side": 0.001,
                    "cash_return": 0, "risk_free_return": 0, "shorting": False,
                    "cash_days_included": True, "dividend_record_ex_pay_separate": True,
                    "target_cagr": 0.10, "target_net_sharpe": 1.2,
                    "benchmark": "同日同价同费用BUY_HOLD，以及平均仓位/实现风险相近的固定基准；在预测门通过后执行"},
        "risk_role": "波动用于风险测量和10%年化波动预算上限，不声称方向预测；账户开始后冻结同一预算给各组",
        "cost_example_only": "以4元不变参考价、20万/2万各100%及25%预算做整手双边费用示例，不做任何信号账户",
        "selection_history": "继承先前全部宏观/价格/资金开发事实，保留原失败及85/15终止；DSR没有完整上游试验分布和新账户时NOT_COMPUTED，不用四组数量替代全部选择次数",
        "independence": "当前六例及原前5/20日走势已被看过，本次先冻结新执行时钟仍只是历史案例复核；严格前向样本0",
        "no_new_sample_rescue": "本轮不切换到NBS、季度流、其他ETF或政策混合类来凑样本；新证据补足前保持缺失",
        "run_order": ["冻结与旧包身份核对", "数据门判定", "六例新执行时钟描述收益与图", "合格时才模型与账户；不合格明确NOT_RUN", "复算与完整审阅包"],
        "position_impact": 0, "live_trading_authorized": False,
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    frozen = [p for p in OUT.rglob("*") if p.is_file()]
    receipt = {"frozen_at": frozen_at, "new_residual_return_code_has_not_run": True,
               "earlier_price_outcomes_already_known": True,
               "prior_archive": {"path": str(archive), "sha256": expected, "bytes": archive.stat().st_size},
               "original_input_map": inventory,
               "frozen_files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size} for p in frozen]}
    (OUT / "freeze_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"状态": "冻结完成", "冻结时间": frozen_at, "文件数": len(frozen)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
