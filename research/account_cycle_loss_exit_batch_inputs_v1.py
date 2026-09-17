"""以实际首次成交对应的事前净值为基准，维护完整账户亏损退出状态。"""
from dataclasses import dataclass

import numpy as np

from research.adaptive_allocation_v1 import target_request
from research.event_account_indexed_request_v1 import simulate_indexed_request_account
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

MODELS = ['INTENT_MIX_BAND_00', 'INTENT_MIX_BAND_20']
SETTINGS, CANDIDATES = {}, {}
for band in [0, 20]:
    for loss in [1, 2, 3]:
        model = f'CYCLE_LOSS_BAND{band:02d}_{loss:02d}'
        SETTINGS[model] = {'parent': f'INTENT_MIX_BAND_{band:02d}', 'band': band/100, 'loss_fraction': loss/100}
        CANDIDATES[model] = f'{band}个百分点调仓、账户周期亏损{loss}%触发退出'
PRIMARY = 'CYCLE_LOSS_BAND00_02'


def loss_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['candidate_settings'] == SETTINGS, '固定候选或亏损门槛改变')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    summaries = []
    for cost, frame in frames.items():
        for model, setting in SETTINGS.items():
            raw = frame[setting['parent']+'_parent_target'].to_numpy(float)
            frame[model+'_target'] = raw
            values = raw[first-1:-1]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(values),
                'positive_source_origins': int((values > 0).sum()), 'zero_source_origins': int((values == 0).sum()),
                'unknown_source_origins': int(np.isnan(values).sum()), 'mean_source_target': float(np.nanmean(values)),
                'rebalance_band': setting['band'], 'loss_fraction': setting['loss_fraction']})
    return frames, summaries


@dataclass
class CycleLossPolicy:
    model: str
    had_shares: bool = False
    pending_entry_nav: float = np.nan
    basis: float = np.nan
    protection: bool = False
    zero_seen: bool = False
    cycle_number: int = 0

    def request(self, account, price, source, cfg):
        require(self.model in SETTINGS and cfg['candidate_settings'] == SETTINGS, '亏损保护候选身份不同')
        require(np.isnan(source) or (np.isfinite(source) and 0 <= source <= 1), '原目标超出范围')
        setting = SETTINGS[self.model]
        nav, holding = account.value(price), account.shares > 0
        require(np.isfinite(nav) and nav > 0, '亏损保护需要有效的当前净资产')
        if holding and not self.had_shares:
            require(np.isfinite(self.pending_entry_nav) and self.pending_entry_nav > 0 and not self.protection, '实际首次买入缺少有效事前基准')
            self.basis = self.pending_entry_nav
            self.cycle_number += 1
        if not holding:
            self.basis = np.nan
            if self.protection and self.zero_seen:
                self.protection, self.zero_seen = False, False
        cycle_return = nav/self.basis-1 if holding else np.nan
        triggered = holding and not self.protection and nav <= self.basis*(1-setting['loss_fraction'])
        if triggered:
            self.protection, self.zero_seen = True, False
        if self.protection and source == 0:
            self.zero_seen = True
        if not holding and self.protection and self.zero_seen:
            self.protection, self.zero_seen = False, False
        effective = 0. if self.protection else source
        if np.isnan(effective):
            result = {'requested_quantity': 0, 'reference_weight': np.nan, 'action': '来源未知且没有保护退出，维持实际份额'}
        else:
            result = target_request(account, price, effective, {**cfg, 'weight_band': setting['band']})
            if self.protection:
                result['action'] = '账户周期亏损保护，等待实际卖完及来源明确归零'
        result.update(output_candidate=self.model, budget_source=setting['parent'], source_original_target=source,
            used_band=float(setting['band']), loss_fraction=float(setting['loss_fraction']),
            cycle_basis=float(self.basis), cycle_return=float(cycle_return),
            cycle_number=float(self.cycle_number if holding else 0), triggered_protection=float(triggered),
            pending_protective_exit=float(self.protection and holding), waiting_source_zero=float(self.protection and not self.zero_seen),
            protection_active=float(self.protection), own_close_shares=float(account.shares), own_close_equity=float(nav))
        self.had_shares = holding
        self.pending_entry_nav = nav if not holding and result['requested_quantity'] > 0 else np.nan
        return result


def simulate_loss_account(data, dividends, cfg, cost, start, model, targets=None, prediction=None, horizon=1, event_mask=None):
    require(model in SETTINGS and targets is not None and prediction is None, '本批仅使用原保存目标')
    require(event_mask is not None and np.asarray(event_mask).all(), '风险保护必须逐收盘检查')
    raw = np.asarray(targets, dtype=float)
    require(raw.shape == (len(data),) and (np.isnan(raw) | (np.isfinite(raw) & (raw >= 0) & (raw <= 1))).all(), '原目标长度或范围无效')
    policy = CycleLossPolicy(model)

    def request(account, price, unused_value, settings, model_id, origin_index):
        return policy.request(account, price, float(raw[origin_index]), settings)

    # 共用引擎在数值缺失时跳过回调；有限占位只用于确保每日调用风险检查。
    # 回调始终重新读取原目标，并独立返回明确保护或原未知状态，不把未知目标补成空仓。
    proxy = np.where(np.isnan(raw), 0., raw)
    ledger, decisions = simulate_indexed_request_account(data, dividends, cfg, cost, start, model,
        targets=proxy, event_mask=event_mask, request_policy=request)
    decisions['signal_state'] = np.where(decisions.reference_weight.isna(), 'NO_VIEW_KEEP_EXISTING_SHARES', 'MODEL_VIEW_AVAILABLE')
    return ledger, decisions
