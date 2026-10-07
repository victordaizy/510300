"""固定高点下的累计成交量先恢复、价格后确认；严格因果状态。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.full_daily_gap_inputs_v1 import gap_states, TICK
from research.confirmed_structure_inputs_v1 import structure_states

PRIMARY = "VOLUME_LEAD_PRICE_CONFIRM"


def volume_lead_states(data):
    """返回全部原点、全部参考锚生命周期及原枢轴到达，不读未来结果。"""
    if "volume" not in data:
        raise ValueError("量先恢复需要真实成交份额。")
    d = data.reset_index(drop=True)
    coordinates = gap_states(d)
    _, pivots = structure_states(d)
    accepted = pivots.loc[pivots.action.ne("IGNORED_SAME_KIND_LESS_EXTREME")]
    arrivals = {int(p.confirmation_index): p for p in accepted.itertuples(index=False)}
    volume = pd.to_numeric(d.volume, errors="coerce").to_numpy(float)
    observations, cumulative, continuous_known = [], 0, True
    for i, q in enumerate(coordinates.itertuples(index=False)):
        valid_volume = bool(np.isfinite(volume[i]) and volume[i] >= 0 and volume[i].is_integer()
                            and volume[i] <= 2**53)
        continuous_known = bool(continuous_known and q.current_quote_known and valid_volume)
        direction, contribution = 0, 0
        if continuous_known and i:
            difference = int(q.known_cash_close_ticks)-int(coordinates.known_cash_close_ticks.iloc[i-1])
            direction = (difference > 0)-(difference < 0)
            contribution = direction*int(volume[i])
            cumulative += contribution
        observations.append({"volume_input_known": valid_volume, "cumulative_volume_known": continuous_known,
                             "cash_close_direction": direction if continuous_known else None,
                             "signed_volume": contribution if continuous_known else None,
                             "cumulative_signed_volume": cumulative if continuous_known else None})
    obs = pd.DataFrame(observations)
    rows, anchors, anchor, latest_low = [], [], None, None
    for i, q in enumerate(coordinates.itertuples(index=False)):
        new_anchor, lead, confirmed = False, False, False
        current = obs.iloc[i]
        arrival = arrivals.get(i)
        if arrival is not None:
            center = int(arrival.center_index)
            if coordinates.current_quote_known.iloc[center]:
                if arrival.kind == "LOW":
                    latest_low = {"index": center, "confirmation_index": i,
                                  "price_ticks": int(coordinates.known_cash_close_ticks.iloc[center])}
                else:
                    if anchor is not None and anchor["terminal_index"] < 0:
                        anchor["status"] = "REPLACED_AFTER_LEAD" if anchor["lead_index"] >= 0 else "REPLACED_BEFORE_LEAD"
                        anchor["terminal_index"], anchor["terminal_date"] = i, q.date
                    reference_known = bool(obs.cumulative_volume_known.iloc[center] and current.cumulative_volume_known)
                    anchor = {"anchor_id": f"VOLUME_PRICE_HIGH_{center}_KNOWN_{i}", "center_index": center,
                              "center_date": coordinates.date.iloc[center], "birth_index": i, "birth_date": q.date,
                              "reference_high_ticks": int(coordinates.known_cash_close_ticks.iloc[center]),
                              "reference_cumulative_volume": int(obs.cumulative_signed_volume.iloc[center]) if reference_known else None,
                              "lead_index": -1, "lead_date": pd.NaT, "fixed_low_ticks": -1,
                              "fixed_low_center_index": -1, "confirmation_index": -1, "confirmation_date": pd.NaT,
                              "terminal_index": -1 if reference_known else i,
                              "terminal_date": pd.NaT if reference_known else q.date,
                              "status": "UNARMED" if reference_known else "UNKNOWN_REFERENCE"}
                    anchors.append(anchor)
                    new_anchor = True
        if anchor is not None and anchor["terminal_index"] < 0:
            if not current.cumulative_volume_known:
                anchor["status"] = "UNKNOWN_INPUT_NO_BRIDGE"
                anchor["terminal_index"], anchor["terminal_date"] = i, q.date
            else:
                price, amount = int(q.known_cash_close_ticks), int(current.cumulative_signed_volume)
                reference_price, reference_amount = anchor["reference_high_ticks"], anchor["reference_cumulative_volume"]
                if anchor["lead_index"] < 0:
                    if latest_low is not None and latest_low["price_ticks"] < price < reference_price and amount > reference_amount:
                        anchor.update(lead_index=i, lead_date=q.date, fixed_low_ticks=latest_low["price_ticks"],
                                      fixed_low_center_index=latest_low["index"], status="ARMED_VOLUME_FIRST")
                        lead = True
                elif price <= anchor["fixed_low_ticks"] or amount <= reference_amount:
                    anchor["status"] = "FIXED_LOW_FAILED_BEFORE_CONFIRM" if price <= anchor["fixed_low_ticks"] else "VOLUME_LEAD_LOST_BEFORE_CONFIRM"
                    anchor["terminal_index"], anchor["terminal_date"] = i, q.date
                elif i > anchor["lead_index"] and price > reference_price:
                    anchor.update(status="CONFIRMED_PRICE_AFTER_VOLUME", confirmation_index=i,
                                  confirmation_date=q.date, terminal_index=i, terminal_date=q.date)
                    confirmed = True
        detail = {"reference_anchor_id": anchor["anchor_id"] if anchor is not None else None,
                  "reference_center_index": anchor["center_index"] if anchor is not None else -1,
                  "reference_confirmation_index": anchor["birth_index"] if anchor is not None else -1,
                  "reference_high_ticks": anchor["reference_high_ticks"] if anchor is not None else -1,
                  "reference_cumulative_volume": anchor["reference_cumulative_volume"] if anchor is not None else None,
                  "reference_status": anchor["status"] if anchor is not None else "NO_CONFIRMED_HIGH",
                  "lead_index": anchor["lead_index"] if anchor is not None else -1,
                  "fixed_low_ticks": anchor["fixed_low_ticks"] if anchor is not None else -1,
                  "fixed_low_center_index": anchor["fixed_low_center_index"] if anchor is not None else -1,
                  "latest_low_index": latest_low["index"] if latest_low is not None else -1,
                  "latest_low_confirmation_index": latest_low["confirmation_index"] if latest_low is not None else -1,
                  "latest_low_ticks": latest_low["price_ticks"] if latest_low is not None else -1}
        rows.append({**q._asdict(), **current.to_dict(), **detail,
                     "new_reference_high": new_anchor, "volume_lead_event": lead,
                     "price_after_volume_event": confirmed})
    frame = pd.DataFrame(rows)
    for name in ("cash_close_direction", "signed_volume", "cumulative_signed_volume", "reference_cumulative_volume"):
        frame[name] = pd.array(frame[name], dtype="Int64")
    paths = pd.DataFrame(anchors)
    if len(paths):
        paths.loc[paths.terminal_index.lt(0) & paths.lead_index.lt(0), "status"] = "OPEN_UNARMED"
        paths.loc[paths.terminal_index.lt(0) & paths.lead_index.ge(0), "status"] = "OPEN_ARMED"
        paths["reference_cumulative_volume"] = pd.array(paths.reference_cumulative_volume, dtype="Int64")
    return frame, paths, pivots


def account_signals(data, states, policy):
    if policy not in (PRIMARY, "PRICE_CONFIRMATION") or not data.date.reset_index(drop=True).equals(states.date):
        raise ValueError("量价顺序政策或原点日期不一致。")
    if policy == "PRICE_CONFIRMATION":
        from research import point_fresh_repair_order_inputs_v1 as old
        frame = old.account_signals(data, old.signals(data), policy)
        for name in ("cash_shift_ticks", "cash_shift_known"):
            frame[name] = states[name].to_numpy()
        return frame
    frame = states.copy()
    frame["entry_event"] = frame.price_after_volume_event
    frame["event_id"] = pd.Series([f"{PRIMARY}_{i}" if event else None
                                  for i, event in enumerate(frame.entry_event)], dtype=object)
    frame["stop_index"] = np.where(frame.entry_event, frame.fixed_low_ticks*TICK, np.nan)
    frame["target_index"] = np.nan
    frame["atr"] = data.atr20.to_numpy(float)
    frame["rule_exit"] = False
    return frame
