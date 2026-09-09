"""以成熟单日成交情景估计十二状态模型，联合选择进入、持有和退出。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import Account, affordable_quantity, execute_order, fill_price, require

PRIMARY = "JOINT_STATE_POLICY"
CONTROL = "JOINT_STATE_MYOPIC"
LEGAL = np.array([[True, state % 3 != 2] for state in range(12)])
INITIAL = np.array([int(state % 3 == 1) for state in range(12)])


def one_step_case(data, dividends, states, origin, mode, action, cfg):
    """名义收盘账户只模拟下一日，成交请求不使用下一开盘的价格。"""
    require(mode in (0, 1, 2) and action in (0, 1) and not (mode == 2 and action == 1), "不合法的账户状态或动作")
    require(0 <= origin < len(data)-1, "一步起点必须有下一交易日")
    source, next_row = data.iloc[origin], data.iloc[origin+1]
    market, next_market = states.market_state.iloc[origin:origin+2].to_numpy(float)
    values = [source.close, next_row.open, next_row.close, next_row.previous_close, next_row.dividend, market, next_market]
    require(np.isfinite(values).all() and min(values[:4]) > 0, "一步必要价格或状态缺失，不允许补零")
    require(abs(float(next_row.previous_close)-float(source.close)) < 1e-8, "一步原收盘连接不一致")
    nominal, cost, lot = cfg["initial_capital"], cfg["costs"]["BASE"], cfg["lot"]
    shares = int(np.floor(nominal / float(source.close) / lot)) * lot if mode else 0
    require(mode == 0 or shares > 0, "名义规模不能形成持仓状态")
    account = Account(nominal-shares*float(source.close), shares=shares,
                      purchase_lots=[(origin, shares)] if shares else [])
    if mode == 0 and action == 1:
        requested = affordable_quantity(account.cash, fill_price(float(source.close), 1, cost, cfg["tick"]), cost, lot)
    else:
        requested = -shares if mode and action == 0 else 0
    recognized, paid, due_at_close = 0., 0., []
    for key, event in enumerate(dividends.itertuples()):
        if pd.Timestamp(event.ex_date) != pd.Timestamp(next_row.date):
            continue
        require(pd.Timestamp(event.record_date) == pd.Timestamp(source.date), "一步除息需要未知的起点之前登记权益")
        value = shares * float(event.cash_dividend_per_share)
        account.entitlements[key] = shares
        account.receivables[key] = value
        recognized += value
        if pd.Timestamp(event.payment_date) < pd.Timestamp(next_row.date):
            paid += account.receivables.pop(key)
            account.cash += value
        elif pd.Timestamp(event.payment_date) == pd.Timestamp(next_row.date):
            due_at_close.append(key)
    execution = execute_order(account, requested, float(next_row.open), float(next_row.previous_close),
                              float(next_row.dividend), origin+1, cost, cfg)
    for key in due_at_close:
        value = account.receivables.pop(key)
        account.cash += value
        paid += value
    end_nav = account.value(float(next_row.close))
    price_pnl = shares*(float(next_row.open)-float(source.close)) + account.shares*(float(next_row.close)-float(next_row.open))
    error = end_nav-nominal-price_pnl-recognized+execution["commission"]+execution["slippage_cost"]
    require(np.isfinite(end_nav) and end_nav > 0 and abs(error) < 1e-6, "名义单日财富核算不符")
    account.assert_valid()
    next_mode = 0 if not account.shares else (2 if mode and action == 0 else 1)
    return {"origin_index": origin, "origin": source.date, "maturity_index": origin+1, "maturity_date": next_row.date,
            "market_state": int(market), "account_mode": mode, "state": int(market)*3+mode, "action": action,
            "next_market_state": int(next_market), "next_account_mode": next_mode, "next_state": int(next_market)*3+next_mode,
            "initial_cash": nominal-shares*float(source.close), "initial_shares": shares, "nominal_nav": nominal,
            "cash": account.cash, "shares": account.shares, "dividend_recognized": recognized, "dividend_paid": paid,
            "dividend_receivable": account.receivable(), "end_nav": end_nav, "reward": float(np.log(end_nav/nominal)),
            "simple_return": end_nav/nominal-1, "accounting_error": error, **execution}


def make_cases(data, dividends, states, clocks, cfg):
    origins = sorted({i for t in clocks for i in range(int(t)-cfg["training_days"], int(t))})
    rows = [one_step_case(data, dividends, states, t, mode, action, cfg)
            for t in origins for mode, action in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]]
    return pd.DataFrame(rows)


def empirical_model(cases, fit_index, cfg):
    left = fit_index-cfg["training_days"]
    chosen = cases[cases.origin_index.ge(left) & cases.origin_index.lt(fit_index)].copy()
    failed = {"status": "NO_VIEW_INCOMPLETE_MATURE_WINDOW", "fit_index": fit_index, "training_left": left}
    if len(chosen) != cfg["training_days"]*5 or chosen.duplicated(["origin_index", "account_mode", "action"]).any():
        return failed
    if set(chosen.origin_index) != set(range(left, fit_index)) or not chosen.maturity_index.eq(chosen.origin_index+1).all() or not chosen.maturity_index.le(fit_index).all():
        return failed
    r, p, counts = np.full((12, 2), np.nan), np.zeros((12, 2, 12)), np.zeros((12, 2), int)
    for state in range(12):
        for action in np.flatnonzero(LEGAL[state]):
            part = chosen[chosen.state.eq(state) & chosen.action.eq(action)]
            if len(part) < cfg["minimum_market_observations"] or not np.isfinite(part.reward).all() or not part.next_state.isin(range(12)).all():
                return {**failed, "status": "NO_VIEW_STATE_ACTION_SUPPORT_OR_REWARD", "failed_state": state, "failed_action": int(action)}
            counts[state, action] = len(part)
            r[state, action] = part.reward.mean()
            p[state, action] = np.bincount(part.next_state.to_numpy(int), minlength=12)/len(part)
    return {"status": "EMPIRICAL_MODEL_COMPLETE", "fit_index": fit_index, "training_left": left,
            "latest_maturity_index": int(chosen.maturity_index.max()), "reward": r, "transition": p, "counts": counts}


def solve_policy(reward, transition, cfg):
    r, p = np.asarray(reward, float), np.asarray(transition, float)
    require(r.shape == (12, 2) and p.shape == (12, 2, 12), "有限模型维度不符")
    require(np.isfinite(r[LEGAL]).all() and np.isfinite(p[LEGAL]).all() and (p[LEGAL] >= 0).all(), "合法动作模型非法")
    require(np.allclose(p.sum(axis=2)[LEGAL], 1., atol=1e-12, rtol=0), "合法动作转移概率之和不等于一")
    gamma, tol = cfg["discount"], cfg["improvement_tolerance"]
    require(0 < gamma < 1, "继续价值折现必须在零和一之间")
    policy, seen = INITIAL.copy(), set()
    idx = np.arange(12)
    for iteration in range(1, cfg["maximum_policy_iterations"]+1):
        signature = tuple(policy)
        if signature in seen:
            return {"status": "NO_VIEW_POLICY_CYCLE", "linear_solves": iteration-1}
        seen.add(signature)
        a, b = np.eye(12)-gamma*p[idx, policy], r[idx, policy]
        try:
            value = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return {"status": "NO_VIEW_POLICY_EVALUATION", "linear_solves": iteration}
        residual = float(np.max(np.abs(a@value-b)))
        if not np.isfinite(value).all() or residual > cfg["equation_residual_limit"]:
            return {"status": "NO_VIEW_POLICY_EQUATION_RESIDUAL", "linear_solves": iteration, "residual": residual}
        q = r + gamma*np.einsum("sak,k->sa", p, value)
        improved = policy.copy()
        for state in range(12):
            for action in np.flatnonzero(LEGAL[state]):
                if q[state, action] > q[state, improved[state]]+tol:
                    improved[state] = action
        if np.array_equal(improved, policy):
            return {"status": "POLICY_COMPLETE", "policy": policy.tolist(), "value": value.tolist(),
                    "q": [[float(q[s, a]) if LEGAL[s, a] else None for a in range(2)] for s in range(12)],
                    "linear_solves": iteration, "residual": residual}
        policy = improved
    return {"status": "NO_VIEW_POLICY_ITERATION_LIMIT", "linear_solves": cfg["maximum_policy_iterations"]}


def myopic_policy(reward, cfg):
    policy = INITIAL.copy()
    for state in range(12):
        for action in np.flatnonzero(LEGAL[state]):
            if reward[state, action] > reward[state, policy[state]]+cfg["improvement_tolerance"]:
                policy[state] = action
    return policy.tolist()


def fit_month(cases, fit_index, cfg):
    fitted = empirical_model(cases, fit_index, cfg)
    if fitted["status"] != "EMPIRICAL_MODEL_COMPLETE":
        return fitted
    reward, transition, counts = fitted["reward"], fitted["transition"], fitted["counts"]
    solution = solve_policy(reward, transition, cfg)
    fitted.update(reward=[[float(reward[s, a]) if LEGAL[s, a] else None for a in range(2)] for s in range(12)],
                  transition=transition.tolist(), counts=counts.tolist(), solution=solution)
    if solution["status"] != "POLICY_COMPLETE":
        fitted["status"] = "NO_VIEW_SHARED_POLICY_ADMISSION"
    else:
        fitted.update(status="FIT_COMPLETE", policies={PRIMARY: solution["policy"], CONTROL: myopic_policy(reward, cfg)})
    return fitted


class JointController:
    def __init__(self, states, models, method):
        self.states, self.models, self.method = states.market_state.to_numpy(float), models, method
        self.clocks = [m["fit_index"] for m in models]
        require(self.clocks == sorted(set(self.clocks)), "模型时钟必须唯一递增")

    def __call__(self, t, mode):
        position = bisect_right(self.clocks, t)-1
        market = float(self.states[t])
        row = {"market_state": market, "account_mode": mode, "policy_action": None, "fit_index": None,
               "joint_state": int(market)*3+mode if np.isfinite(market) else None, "model_status": "NO_VIEW_NO_MATURE_MODEL"}
        if position >= 0:
            model = self.models[position]
            row.update(fit_index=model["fit_index"], model_status=model["status"])
        if mode == 2:
            return {**row, "policy_action": 0, "decision_status": "LOCKED_EXIT_CONTINUES"}
        if not np.isfinite(market):
            return {**row, "decision_status": "NO_VIEW_MARKET_STATE"}
        if position < 0 or model["status"] != "FIT_COMPLETE":
            return {**row, "decision_status": "NO_VIEW_MODEL"}
        require(model["latest_maturity_index"] <= model["fit_index"] <= t, "联合模型偷看未来")
        action = model["policies"][self.method][row["joint_state"]]
        require(LEGAL[row["joint_state"], action], "保存动作不合法")
        return {**row, "policy_action": action, "decision_status": "POLICY_ACTION_AVAILABLE"}
