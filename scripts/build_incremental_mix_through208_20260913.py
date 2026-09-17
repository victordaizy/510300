"""为五十个新增候选生成一次增量比较，不重新读取旧258条路径。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = ROOT/'scripts/incremental_saved_mix_through208_20260913.py'
    if dest.exists():
        raise ValueError('本次增量入口已存在')
    text = (ROOT/'scripts/incremental_saved_mix_through199_20260913.py').read_text(encoding='utf-8')
    text = text.replace('510300_incremental_saved_mix_through199', '510300_incremental_saved_mix_through208')
    text = text.replace("OLD = ROOT/'reports/research/510300_saved_combination_fast_screen_20260913'", "OLD = ROOT/'reports/research/510300_incremental_saved_mix_through199'")
    text = text.replace('saved_net_return_matrices.npz', 'saved_return_matrices.npz')
    text = text.replace('224', '258').replace('34', '50').replace('截至199轮', '截至208轮')
    old = "source_folders = ['510300_risk_window_clock_batch_v1', '510300_target_band_partial_rebalance_v1',\n                      '510300_two_close_zero_exit_v1', '510300_confirmed_exit_existing_budget_batch_v1']"
    new = "source_folders = ['510300_three_source_order_intent_mix_v1', '510300_finite_rebalance_band_batch_v1',\n        '510300_account_cycle_loss_exit_batch_v1', '510300_close_return_buy_gate_batch_v1',\n        '510300_close_return_buy_strength_gate_batch_v1', '510300_addition_only_return_gate_batch_v1',\n        '510300_addition_gate_exposure_batch_v1', '510300_positive_target_reduction_batch_v1',\n        '510300_gated_source_ablation_batch_v1']"
    if text.count(old) != 1:
        raise ValueError('旧增量来源列表不同')
    text = text.replace(old, new).replace('原198单独', '原205单独').replace("item['model'] == 'TWO_CLOSE_ZERO_EXIT'", "item['model'] == 'ADD_GATE_BAND20_LOWER01'")
    with dest.open('x', encoding='utf-8') as stream:
        stream.write(text)
    print('增量入口已生成：复用258条矩阵、仅新增五十候选四场景；尚未读取新结果或优化。', flush=True)


if __name__ == '__main__':
    main()
