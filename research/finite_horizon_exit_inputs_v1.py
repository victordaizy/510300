"""按持仓年龄倒推已成熟路径的可成交退出现金流，同年龄固定目标作对照。"""
from bisect import bisect_right

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import Account, execute_order, require
from research.learned_cycle_exit_v1 import state_values
from research.market_path_exit_inputs_v1 import FEATURES, CN
from research.market_path_state_v1 import MarketPathTracker
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit

PRIMARY = "AGE_BACKWARD_EXIT"
CONTROL = "AGE_NATURAL_TARGET"
METHODS = [PRIMARY, CONTROL]


def liquidation_curve(data, dividends, entry, terminal, quantity, cfg, cost):
    """对完整历史周期逐日检查可卖性；收入包括已取得的分红权益一次。"""
    require(0 <= entry < terminal < len(data) and quantity > 0, "成熟参考的日期或份额无效")
    require(quantity % cfg["lot"] == 0, "成熟参考份额不是整手")
    values, details = {}, []
    for day in range(entry+1, terminal+1):
        raw = data.iloc[day]
        require(np.isfinite([raw.open, raw.previous_close, raw.dividend]).all(), "可成交现金流的必要价格缺失")
        inventory = Account(cash=0., shares=quantity, purchase_lots=[(entry, quantity)])
        order = execute_order(inventory, -quantity, float(raw.open), float(raw.previous_close), float(raw.dividend), day, cost, cfg)
        require(order["filled_quantity"] in [0, -quantity], "单一成熟参考退出发生部分成交")
        rights = dividends[(dividends.record_date >= data.date.iloc[entry]) & (dividends.record_date < raw.date)]
        require(rights.empty or rights.ex_date.le(data.date.iloc[terminal]).all(), "分红权益在原完整周期终点仍未确认，不能沿用原成熟时点")
        earned = quantity*float(rights.cash_dividend_per_share.sum())
        total = inventory.cash+earned if order["filled_quantity"] else None
        if total is not None:
            values[day] = float(total)
        details.append({"entry_index": entry, "sale_index": day, "natural_exit_index": terminal,
                        "quantity": quantity, "raw_open": float(raw.open), "status": order["status"],
                        "filled_quantity": order["filled_quantity"], "fill_price": order["fill_price"],
                        "commission": order["commission"], "slippage_cost": order["slippage_cost"],
                        "net_sale_cash": inventory.cash if order["filled_quantity"] else None,
                        "earned_dividend_cny": earned, "liquidation_value": total})
    require(terminal in values, "原自然退出开盘不能完成卖出，原参考成熟边界失效")
    return values, details


def prepare_cashflows(data, dividends, samples, cycles, cfg):
    require(cycles.cycle_id.is_unique, "原参考周期编号重复")
    by_cycle = cycles.set_index("cycle_id")
    values, checks, additions = {}, [], []
    for cycle_id, rows in samples.groupby("cycle_id", sort=True):
        cycle_id = int(cycle_id)
        original = by_cycle.loc[cycle_id]
        require(rows.exit_index.nunique() == 1 and rows.reference_quantity.nunique() == 1, "同周期的自然终点或份额不唯一")
        entry, terminal = int(original.entry_index), int(rows.exit_index.iloc[0])
        quantity = int(rows.reference_quantity.iloc[0])
        require(quantity == original.entry_quantity, "状态份额与实际原参考入场不同")
        curve, detail = liquidation_curve(data, dividends, entry, terminal, quantity, cfg, cfg["costs"]["BASE"])
        values[cycle_id] = curve
        checks.extend({"cycle_id": cycle_id, **d} for d in detail)
        feasible = np.array(sorted(curve), dtype=int)
        for row in rows.itertuples():
            first = int(feasible[np.searchsorted(feasible, int(row.origin_index)+1)])
            age = int(row.origin_index)-entry+1
            require(first <= terminal and 1 <= age <= cfg["maximum_age"], "训练状态超出原自然退出或持仓年龄边界")
            additions.append({"cycle_id": cycle_id, "origin_index": int(row.origin_index), "holding_age": age,
                              "entry_index": entry, "immediate_exit_index": first,
                              "immediate_exit_value": curve[first], "natural_exit_value": curve[terminal],
                              "target_denominator": quantity*float(data.open.iloc[first])})
    out = samples.merge(pd.DataFrame(additions), on=["cycle_id", "origin_index"], how="left", validate="one_to_one", sort=False)
    require(len(out) == len(samples) and not out.duplicated(["cycle_id", "holding_age"]).any(), "逐年龄现金流删除或重复原状态")
    require(np.isfinite(out[["immediate_exit_value", "natural_exit_value", "target_denominator"]]).all().all(), "逐年龄现金流缺失")
    return out, values, pd.DataFrame(checks)


def fit_pair(x, targets, cfg):
    x, targets = np.asarray(x, float), np.asarray(targets, float)
    require(x.ndim == 2 and x.shape[1] == len(FEATURES) and targets.shape == (len(x), 2), "同年龄两个目标的维度不同")
    require(np.isfinite(x).all() and np.isfinite(targets).all(), "逐年龄训练输入或目标缺失，禁止删行")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    design = np.column_stack([np.ones(len(x)), z])
    penalty = np.diag([0.]+[cfg["ridge_alpha"]]*len(FEATURES))
    coefficients = np.linalg.solve(design.T@design+penalty, design.T@targets)
    require(np.isfinite(coefficients).all(), "逐年龄岭回归系数无效")
    return {"kind": "AGE_PAIRED_RIDGE", "features": FEATURES.copy(), "methods": METHODS.copy(),
            "mean": mean.tolist(), "scale": scale.tolist(), "feature_clip": cfg["feature_clip"],
            "intercepts": coefficients[0].tolist(), "coefficients": coefficients[1:].tolist()}


def predict_pair(model, x):
    require(model["kind"] == "AGE_PAIRED_RIDGE" and model["features"] == FEATURES and model["methods"] == METHODS, "逐年龄模型身份不同")
    x = np.asarray(x, float)
    require(x.shape[-1] == len(FEATURES) and np.isfinite(x).all(), "逐年龄预测状态缺失")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return np.asarray(model["intercepts"])+z@np.asarray(model["coefficients"])


def fit_backward_month(rows, curves, cfg):
    """未来退出仅由已拟合预测决定；目标对照始终使用原自然退出。"""
    require(not rows.duplicated(["cycle_id", "holding_age"]).any(), "同年龄一个周期出现多行，不能宣称等周期权重")
    require(np.isfinite(rows[FEATURES].to_numpy(float)).all(), "整月必要状态缺失，禁止局部删行")
    chosen = {int(cid): int(group.exit_index.iloc[0]) for cid, group in rows.groupby("cycle_id")}
    models, receipts, members = {}, [], []
    groups = {int(age): group for age, group in rows.groupby("holding_age")}
    for age in range(cfg["maximum_age"], 0, -1):
        group = groups.get(age, rows.iloc[:0])
        record = {"holding_age": age, "cycles": len(group), "status": "NO_VIEW_MINIMUM_AGE_CYCLES", "failure": None}
        if len(group) < cfg["minimum_age_cycles"]:
            receipts.append(record)
            continue
        ids = group.cycle_id.to_numpy(int)
        future_exits = np.array([chosen[cid] for cid in ids], int)
        early = group.immediate_exit_index.to_numpy(int)
        require(np.all(early <= future_exits) and np.all(future_exits <= group.exit_index.to_numpy(int)), "倒推未来退出跨越立即可成交时点或自然边界")
        future_values = np.array([curves[cid][day] for cid, day in zip(ids, future_exits, strict=True)], float)
        immediate, denominator = group.immediate_exit_value.to_numpy(float), group.target_denominator.to_numpy(float)
        targets = np.column_stack([(future_values-immediate)/denominator,
                                   (group.natural_exit_value.to_numpy(float)-immediate)/denominator])
        try:
            model = fit_pair(group[FEATURES].to_numpy(float), targets, cfg)
            predictions = predict_pair(model, group[FEATURES].to_numpy(float))
            negative = predictions[:, 0] < 0
            for cid, first, leave in zip(ids, early, negative, strict=True):
                if leave:
                    chosen[int(cid)] = int(first)
            models[str(age)] = model
            record["status"] = "FIT_COMPLETE"
            for k, row in enumerate(group.itertuples()):
                members.append({"holding_age": age, "cycle_id": int(row.cycle_id), "origin_index": int(row.origin_index),
                                "exit_index": int(row.exit_index), "sample_weight": 1., "immediate_exit_index": int(early[k]),
                                "future_exit_before_decision": int(future_exits[k]), "future_exit_after_decision": chosen[int(row.cycle_id)],
                                "future_liquidation_value": float(future_values[k]), "backward_target": float(targets[k, 0]),
                                "natural_target": float(targets[k, 1]), "backward_prediction": float(predictions[k, 0]),
                                "natural_prediction": float(predictions[k, 1]), "backward_requests_exit": bool(negative[k])})
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            record["status"], record["failure"] = "NO_VIEW_MODEL_FIT_FAILED", str(error)
        receipts.append(record)
    return models, receipts, members


class FiniteHorizonExitController:
    def __init__(self, data, models, method):
        require(method in METHODS, "逐年龄退出方法未登记")
        self.data, self.models, self.method = data, models, method
        self.column = METHODS.index(method)
        self.indexes = [m["fit_index"] for m in models]
        require(self.indexes == sorted(set(self.indexes)), "逐年龄模型月份必须唯一递增")
        self.market_path, self.cycle_id, self.negative_count = MarketPathTracker(data), None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = np.asarray(state_values(self.data, t, cycle, current_value, peak_value), float)
        account_return, account_drawdown = float(x[1]), float(x[2])
        market = self.market_path.observe(t, cycle, current_value)
        x[1:3] = [market["market_cycle_return"], market["market_cycle_drawdown"]]
        age, k = t-int(cycle["entry_index"])+1, bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "MONTH_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "逐年龄退出读取未来周期或模型")
            model = record["age_models"].get(str(age))
            if model is None:
                status = record["age_statuses"].get(str(age), "NO_VIEW_OUTSIDE_REGISTERED_AGE")
            elif not np.isfinite(x).all():
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
            else:
                value, status = float(predict_pair(model, x)[self.column]), "PREDICTION_AVAILABLE"
        elif record and record["status"] == "NO_VIEW_INCOMPLETE_TRAINING_FEATURES":
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "holding_age": age, "learning_method": self.method, "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": value is not None and value < 0, "account_cycle_return": account_return,
                "account_cycle_drawdown": account_drawdown, **market, **dict(zip(FEATURES, x))}


def simulate_finite_exit(data, dividends, cfg, cost, start, rule, specification, controller):
    ledger, decisions, cycles = simulate_rearmed_exit(data, dividends, cfg, cost, start, rule, specification, controller)
    before = "连续两个收盘预测继续持有收益为负，学习条件请求退出"
    after = "当日收盘预测继续持有收益为负，学习条件请求退出"
    for frame, column in [(ledger, "execution_reasons"), (decisions, "exit_reasons"), (cycles, "exit_reasons")]:
        if column in frame:
            frame[column] = frame[column].str.replace(before, after, regex=False)
    return ledger, decisions, cycles
