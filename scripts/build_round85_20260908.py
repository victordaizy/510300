"""复用单次进入和完整退出账户，新增一个价格恐慌信息设置。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / "research/synthetic_fear_recovery_v1.py"
    require(not p.exists(), "85来源已建立")
    s = (ROOT / "research/composite_streak_reversal_v1.py").read_text(encoding="utf-8")
    s = s.replace("composite_streak_reversal", "synthetic_fear_recovery").replace("COMPOSITE_STREAK_REVERSAL", "SYNTHETIC_FEAR_RECOVERY")
    s = s.replace('NAME = "三项短期反转与独立退出"', 'NAME = "价格恐慌从波动带上方回落后进入"')
    s = s.replace('P76 = ROOT /', 'P82 = ROOT / "reports/research/510300_two_policy_min_variance_v1"\nP76 = ROOT /', 1)
    s = s.replace('"BUY_HOLD": "买入持有"}', '"BUY_HOLD": "买入持有", "PANIC_ONLY": "原短期高波动急跌回升", "TWO_POLICY_MIN_VARIANCE": "第82轮最小方差预算"}')
    s = s.replace('(P76 if model == "TWO_POLICY_RISK_BUDGET" else P46)', '(P82 if model == "TWO_POLICY_MIN_VARIANCE" else P76 if model == "TWO_POLICY_RISK_BUDGET" else P46)')
    s = s.replace('round=78,', 'round=85,')
    s = s.replace('price_rsi_period=3, streak_rsi_period=2, prior_rank_window=100, trend_window=200,\n        entry_score_strictly_below=10, exit_score_strictly_above=70,',
        'fear_window=22, band_window=20, band_standard_deviations=2, previous_above_strict=True, current_inside_inclusive=True, price_must_rise=True,')
    s = s.replace('policy_spec={"cooldown": 1, "modes": {1: {"loss": .04, "trail": None, "take": None, "days": 5}}}', 'policy_spec={"cooldown": 2, "modes": {1: {"loss": .04, "trail": None, "take": .06, "days": 10}}}')
    s = s.replace('"PROGRESS_ROUND76_AND_77_COMPLETED_AND_DELIVERED"', '"PROGRESS_ROUND84_COMPLETE"')
    s = s.replace('"evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8', '"evaluation_accounts": 14, "new_accounts_generated": 2, "reused_control_accounts": 12')
    s = s.replace('"earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8', '"earlier_diagnostic_accounts": 14, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 12')
    s = s.replace('"保存对照": 16', '"保存对照": 24')
    s = s.replace("第78轮三项反转唯一设置已冻结，尚未运行新账户。", "第85轮单项价格恐慌回落进入已冻结，尚无新账户收益。")
    s = s.replace("三项短期反转", "价格恐慌回落").replace("三项反转", "价格恐慌回落")
    p.write_text(s, encoding="utf-8")
    (ROOT / "reports/research/510300_synthetic_fear_recovery_v1").mkdir(parents=True, exist_ok=True)
    print("第85轮价格恐慌回落单项运行来源已建立。", flush=True)


if __name__ == "__main__":
    main()
