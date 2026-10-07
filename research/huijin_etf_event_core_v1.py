"""汇金ETF公开披露的因果事件状态与账户决策；只观察公告，不推断真实买入路径。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.selected_mix_risk_before_band_v1 import fixed_choice
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account


PRIMARY = "DISCLOSURE_PRICE_CONFIRMATION"
CONTROL = "DISCLOSURE_ONLY"


def public_states(data, facts):
    dates = pd.DatetimeIndex(data.date)
    closes = dates.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)
    scheduled, receipts = {}, []
    for number, fact in enumerate(facts):
        available = pd.Timestamp(fact["available_at_upper"])
        index = int(closes.searchsorted(available, side="right"))
        if index >= len(data):
            continue
        publication_day = pd.Timestamp(fact["publication_date"])
        anchor = int(dates.searchsorted(publication_day, side="left")) - 1
        if anchor < 0:
            raise ValueError("公告前参考价格缺失")
        record = {"fact_index": number, "review_index": index, "review_date": dates[index],
                  "available_at_upper": available, "anchor_index": anchor,
                  "anchor_date": dates[anchor], "anchor_wealth": float(data.wealth.iloc[anchor]), **fact}
        assert available < closes[index] and dates[anchor] < publication_day
        scheduled.setdefault(index, []).append(record)
    active, episodes, states = None, [], []
    for t, day in enumerate(dates):
        if active is not None and t > active["expiry_index"]:
            active = None
        todays = scheduled.get(t, [])
        sell = any(row["action"] == "SELL" for row in todays)
        if sell:
            active = None
        for record in todays:
            row = {**record, "episode_action": "NO_NEW_BUY_EVENT"}
            if record["action"] == "SELL":
                row["episode_action"] = "CANCEL_PUBLIC_SUPPORT_AND_EXIT"
            elif record["action"] == "BUY" and not sell:
                if active is not None:
                    row["episode_action"] = "REPEATED_BUY_DISCLOSURE_NO_EXTENSION"
                    row["episode_id"] = active["episode_id"]
                else:
                    active = {"episode_id": len(episodes), "publication_date": record["publication_date"],
                              "review_index": t, "review_date": day, "expiry_index": t + 20,
                              "anchor_index": record["anchor_index"], "anchor_date": record["anchor_date"],
                              "anchor_wealth": record["anchor_wealth"], "source_url": record["url"]}
                    episodes.append(active.copy())
                    row.update(episode_action="NEW_BUY_DISCLOSURE_EPISODE", episode_id=active["episode_id"])
            receipts.append(row)
        states.append({"date": day, "origin_index": t, "episode_id": active["episode_id"] if active else -1,
                       "known_sell_today": sell, "anchor_wealth": active["anchor_wealth"] if active else np.nan,
                       "review_index": active["review_index"] if active else -1,
                       "expiry_index": active["expiry_index"] if active else -1})
    return pd.DataFrame(states), episodes, receipts


class DisclosurePolicy:
    """持仓是否成交与公开事件分开；任何实际退出后不在同一事件重入。"""

    def __init__(self, data, states, episodes, tails, config, cost_name, name):
        self.data, self.states = data, states
        self.episodes = {row["episode_id"]: row for row in episodes}
        self.tails = {row["origin_index"]: row for row in tails}
        self.config, self.cost_name, self.name = config, cost_name, name
        self.peak, self.stopped = float(config["initial_capital"]), False
        self.current_episode, self.entered, self.completed, self.last_shares = -1, False, False, 0

    def __call__(self, account, price, unused, config, model, t):
        nav = account.value(price)
        self.peak = max(self.peak, nav)
        drawdown = 1. - nav / self.peak
        self.stopped = self.stopped or drawdown >= .1
        public_id = int(self.states.episode_id.iloc[t])
        reason = "没有可用的买入披露事件"
        if account.shares == 0 and public_id != self.current_episode:
            self.current_episode = public_id
            self.entered, self.completed, self.last_shares = False, False, 0
        if account.shares > 0:
            self.entered = True
        elif self.entered and self.last_shares > 0:
            self.completed = True
        raw = 0.
        episode = self.episodes.get(self.current_episode)
        anchor = episode["anchor_wealth"] if episode else np.nan
        price_valid = bool(episode and self.data.wealth.iloc[t] > anchor)
        if episode and self.current_episode == public_id and not self.completed and not self.stopped:
            if t >= episode["expiry_index"]:
                self.completed = True
                reason = "二十交易日事件窗口到期"
            elif account.shares > 0:
                if price_valid:
                    raw, reason = 1., "参考位以上持有，逐日受风险预算约束"
                else:
                    self.completed = True
                    reason = "收盘未维持公告前参考位，申请退出"
            elif self.name == CONTROL or price_valid:
                raw, reason = 1., "公告参与" if self.name == CONTROL else "已收复参考位，申请下一开盘参与"
            else:
                reason = "等待收盘收复参考位"
        elif account.shares > 0:
            self.completed = True
            reason = "公开支持取消、事件结束或已有退出状态，继续请求卖出"
        elif self.completed:
            reason = "本事件已退出，不重复进入"
        if self.stopped:
            raw, reason = 0., "账户回撤达到停止线，退出且不再恢复"
        tail = self.tails[t]
        chosen = fixed_choice(account.shares, nav, self.peak, price, raw, self.stopped,
                              tail["es95"], config["costs"][self.cost_name], config["tick"], config["lot"])
        if account.shares > 0 and chosen["target_shares"] == 0:
            self.completed = True
        self.last_shares = account.shares
        q = chosen["target_shares"]
        return {**chosen, "requested_quantity": q-account.shares, "reference_weight": q*price/nav,
                "action": reason, "decision_drawdown": drawdown, "tail_sample_rows": tail["training_rows"],
                "tail_es95": tail["es95"], "tail_q05": tail["q05"], "conditional_tail": tail["conditional"],
                "public_episode_id": public_id, "account_episode_id": self.current_episode,
                "reference_wealth": anchor, "price_confirmed": price_valid,
                "episode_entered": self.entered, "episode_completed": self.completed,
                "known_sell_today": bool(self.states.known_sell_today.iloc[t])}


def simulate(data, dividends, config, states, episodes, tails, policy_name, cost_name, stop_index=None,
             next_execution_date="2026-09-28"):
    policy = DisclosurePolicy(data, states, episodes, tails, config, cost_name, policy_name)
    ledger, decisions, checkpoint = simulate_indexed_request_account(
        data, dividends, config, config["costs"][cost_name], "2015-01-05", policy_name,
        targets=np.zeros(len(data)), event_mask=np.ones(len(data), bool), request_policy=policy,
        stop_index=stop_index, next_execution_date=next_execution_date)
    checkpoint["risk_governor"] = {"peak": policy.peak, "stopped": policy.stopped}
    checkpoint["disclosure_state"] = {"current_episode": policy.current_episode, "entered": policy.entered,
                                       "completed": policy.completed, "last_shares": policy.last_shares}
    return ledger, decisions, checkpoint
