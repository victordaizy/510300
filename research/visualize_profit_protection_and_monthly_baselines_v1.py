"""读取已保存账本绘图，并用人工路径解释先启动、后回撤退出的机制。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/research/510300_profit_protection_visuals_v1"
ACCOUNT_SIZE = 200_000.0
ANNUAL_DAYS = 242
SOURCES = {
    "A": "reports/research/510300_monthly_single_factor_walkforward_v1/accounts/main/BASE/MONTHLY_VOL10",
    "C": "reports/research/510300_monthly_downside_forecast_v1/accounts/main/BASE/LOG_RISK_RIDGE",
    "BUY_HOLD": "reports/research/510300_simple_core_window_diagnostic_v1/comparators/main_BUY_HOLD_BASE",
}
LABELS = {"A": "A  简单目标波动率配置", "C": "C  下行风险预测配置", "BUY_HOLD": "510300 买入持有"}
COLORS = {"A": "#197a8a", "C": "#bd7040", "BUY_HOLD": "#727d90"}


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    # 优先使用系统中文字体；在其他电脑复核时允许常见中文字体回退。
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        plt.rcParams["font.family"] = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "sans-serif"]
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#ccd3da",
        "axes.labelcolor": "#374456",
        "xtick.color": "#596575",
        "ytick.color": "#596575",
        "text.color": "#1a2c40",
        "figure.facecolor": "#f7f8fa",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#f7f8fa",
        "path.simplify": False,
    })


def load_accounts(source_root: Path) -> tuple[pd.DataFrame, dict, list]:
    merged = None
    metrics = {}
    receipts = []
    for key, folder in SOURCES.items():
        ledger_path = source_root / folder / "ledger.parquet"
        completed_path = source_root / folder / "completed.json"
        df = pd.read_parquet(ledger_path)
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].duplicated().any() or not df["date"].is_monotonic_increasing:
            raise ValueError(f"{key} 日期必须严格递增且不能重复")
        if df.isna()[["equity", "net_return", "exposure"]].any().any():
            raise ValueError(f"{key} 关键账户字段出现缺失")
        nav = df["equity"].to_numpy(dtype=float) / ACCOUNT_SIZE
        peaks = np.maximum.accumulate(np.r_[1.0, nav])[1:]
        dd = nav / peaks - 1
        ret = df["net_return"].to_numpy(dtype=float)
        expected_ret = df["equity"].to_numpy(dtype=float) / np.r_[ACCOUNT_SIZE, df["equity"].to_numpy(dtype=float)[:-1]] - 1
        np.testing.assert_allclose(ret, expected_ret, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            df["equity"], df["cash"] + df["shares"] * df["mark"] + df["dividend_receivable"], atol=1e-6, rtol=0
        )
        trough = int(np.argmin(dd))
        peak = int(np.argmax(nav[:trough + 1]))
        result = {
            "label": LABELS[key],
            "days": len(df),
            "start_date": str(df["date"].iloc[0].date()),
            "end_date": str(df["date"].iloc[-1].date()),
            "end_nav": float(nav[-1]),
            "end_equity": float(df["equity"].iloc[-1]),
            "annual_return": float(nav[-1] ** (ANNUAL_DAYS / len(df)) - 1),
            "sharpe": float(ret.mean() / ret.std(ddof=1) * np.sqrt(ANNUAL_DAYS)),
            "max_drawdown": float(dd[trough]),
            "mean_exposure": float(df["exposure"].mean()),
            "peak_date": str(df["date"].iloc[peak].date()),
            "peak_nav": float(nav[peak]),
            "trough_date": str(df["date"].iloc[trough].date()),
            "trough_nav": float(nav[trough]),
        }
        saved = json.loads(completed_path.read_text(encoding="utf-8-sig"))
        for name in ["annual_return", "sharpe", "max_drawdown", "mean_exposure", "end_equity"]:
            np.testing.assert_allclose(result[name], saved[name], atol=1e-10, rtol=0)
        metrics[key] = result
        part = pd.DataFrame({
            "date": df["date"], f"{key}_equity": df["equity"], f"{key}_nav": nav,
            f"{key}_drawdown": dd, f"{key}_exposure": df["exposure"], f"{key}_net_return": ret,
        })
        if merged is None:
            merged = part
        else:
            if not merged["date"].equals(part["date"]):
                raise ValueError("三个账本日期并非逐日完全一致")
            merged = merged.merge(part, on="date", validate="one_to_one")
        for p in [ledger_path, completed_path]:
            receipts.append({"path": p.relative_to(source_root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)})
    return merged, metrics, receipts


def artificial_path() -> tuple[pd.DataFrame, dict]:
    close = np.array([100, 102, 104, 108, 110, 115, 120, 122, 121, 120, 117, 114, 112, 109, 108, 111, 116, 124, 130], dtype=float)
    entry, activation_gain, trailing_dd = 100.0, 0.10, 0.05
    armed = False
    pending = False
    high = entry
    armed_day = None
    trigger_day = None
    records = []
    for day, price in enumerate(close):
        if pending:
            state = "次日开盘执行后持币" if day == trigger_day + 1 else "已退出；未定义重新入场"
            threshold = np.nan
            shown_high = np.nan
        else:
            high = max(high, price)
            if not armed and price >= entry * (1 + activation_gain) - 1e-10:
                armed = True
                armed_day = day
            shown_high = high if armed else np.nan
            threshold = high * (1 - trailing_dd) if armed else np.nan
            state = "已启动保护" if armed else "尚未启动保护"
            if armed and price <= threshold + 1e-10:
                trigger_day = day
                pending = True
                state = "收盘触发；等待下一交易日开盘"
        records.append({"day": day, "close": price, "rolling_close_high_while_armed": shown_high, "trigger_line": threshold, "state": state})
    info = {
        "type": "SYNTHETIC_MECHANISM_ONLY_NOT_A_BACKTEST",
        "entry_price": entry, "activation_gain": activation_gain, "trailing_drawdown": trailing_dd,
        "activation_day": armed_day, "trigger_day": trigger_day,
        "trigger_close": float(close[trigger_day]),
        "trigger_high": float(max(close[:trigger_day + 1])),
        "trigger_threshold": float(max(close[:trigger_day + 1]) * (1 - trailing_dd)),
        "fill_day": trigger_day + 1, "assumed_next_open_fill": 112.6,
        "fill_is_illustrative": True,
        "parameters_frozen_for_research": False,
        "account_backtest": "NOT_RUN",
    }
    return pd.DataFrame(records), info


def annotation(ax, text, xy, xytext, color="#1a2c40", **kwargs):
    return ax.annotate(text, xy=xy, xytext=xytext, fontsize=11, color=color,
                       arrowprops={"arrowstyle": "-", "color": color, "lw": 1.1},
                       bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "none", "alpha": 0.94}, **kwargs)


def mechanism_plot(out: Path, df: pd.DataFrame, info: dict) -> None:
    fig = plt.figure(figsize=(16, 9))
    fig.text(.055, .95, "先涨到门槛，再等回撤退出", fontsize=25, weight="bold")
    fig.text(.055, .905, "人工价格路径  ·  +10% 启动 / 从滚动高点回撤 5% 触发，仅用于解释机制", fontsize=12, color="#596575")
    ax = fig.add_axes([.065, .29, .87, .56])
    ax.axvspan(-.2, 4, color="#f0f3f7", alpha=.8)
    ax.axvspan(4, 11, color="#e6f3f0", alpha=.65)
    ax.axvspan(11, 12, color="#fff0d6", alpha=.7)
    ax.axvspan(12, 18.3, color="#f1f2f4", alpha=.8)
    ax.plot(df.loc[:11, "day"], df.loc[:11, "close"], color="#197a8a", lw=2.8, marker="o", ms=4, label="持仓期间收盘价")
    ax.plot(df.loc[11:, "day"], df.loc[11:, "close"], color="#8d98a8", lw=2, ls="--", label="信号后及退出后的价格路径")
    ax.step(df["day"], df["rolling_close_high_while_armed"], where="post", color="#4a8c67", lw=1.6, ls=":", label="当时已知的滚动收盘高点")
    ax.step(df["day"], df["trigger_line"], where="post", color="#cc7840", lw=2, label="回撤触发线 = 高点 × 95%")
    ax.scatter([4, 7, 11], [110, 122, 114], s=[100, 65, 100], color=["#197a8a", "#4a8c67", "#cc7840"], edgecolors="white", zorder=5)
    ax.scatter([12], [112.6], marker="X", s=150, color="#b54240", edgecolors="white", zorder=6)
    annotation(ax, "示例买入 100", (0, 100), (.4, 97.9))
    annotation(ax, "① 涨到 110\n启动利润保护", (4, 110), (1.2, 120.5), color="#197a8a")
    annotation(ax, "② 高点升到 122\n保护线随之上移", (7, 122), (5.8, 130.0), color="#347052")
    annotation(ax, "触发线 115.9\n只上移，不下调", (10, 115.9), (12.4, 121.4), color="#ac612e")
    annotation(ax, "③ 收盘 114 跌破线\n此时才产生卖出信号", (11, 114), (7.0, 101.0), color="#ac612e")
    annotation(ax, "④ 次日开盘示例成交 112.6\n成交价可能低于触发线", (12, 112.6), (12.25, 97.9), color="#a13d3b")
    annotation(ax, "卖出后也可能重新上涨\n何时买回，需要另定规则", (17, 124), (14.25, 130.0), color="#647083")
    ax.set_xlim(-.25, 18.4)
    ax.set_ylim(96, 137)
    ax.set_xticks(np.arange(0, 19, 2))
    ax.set_xlabel("示例交易日（0 为买入日）", labelpad=10)
    ax.set_ylabel("价格指数（买入 = 100）", labelpad=12)
    ax.grid(axis="y", color="#d9e0e7", lw=.7, alpha=.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.15), ncol=2, frameon=False, fontsize=10)
    fig.text(.06, .13, "状态顺序", fontsize=12, weight="bold")
    fig.text(.17, .13, "未启动  →  达到涨幅门槛  →  持续更新高点  →  回撤触发  →  下一交易时点执行", fontsize=13)
    fig.text(.06, .076, "一旦启动就保持保护状态；高点只用当时已知信息。图中数值不是推荐参数，也没有证明这种退出能提高收益。", fontsize=10.7, color="#596575")
    fig.text(.06, .046, "这张图只解释退出动作。启动前如何控制亏损、退出多少、何时重新买入，仍需单独规定。", fontsize=10.7, color="#596575")
    fig.savefig(out / "01_先上涨后回撤退出_机制示意.png", dpi=160)
    fig.savefig(out / "01_先上涨后回撤退出_机制示意.svg")
    plt.close(fig)


def account_plot(out: Path, data: pd.DataFrame, metrics: dict) -> None:
    fig = plt.figure(figsize=(16, 11))
    fig.text(.055, .953, "实际历史：回撤小一些，收益也可能少很多", fontsize=24, weight="bold")
    fig.text(.055, .916, "2020-01-02 — 2026-09-11  ·  全部 1,624 个交易日  ·  20 万元独立账户  ·  基础成本", fontsize=12, color="#596575")
    ax_nav = fig.add_axes([.065, .59, .665, .285])
    ax_dd = fig.add_axes([.065, .355, .665, .185], sharex=ax_nav)
    ax_exp = fig.add_axes([.065, .155, .665, .145], sharex=ax_nav)
    dates = data["date"]
    for key in ["BUY_HOLD", "C", "A"]:
        ax_nav.plot(dates, data[f"{key}_nav"], color=COLORS[key], lw=1.5, label=LABELS[key])
        ax_dd.plot(dates, data[f"{key}_drawdown"], color=COLORS[key], lw=1.25)
        ax_exp.plot(dates, data[f"{key}_exposure"], color=COLORS[key], lw=1.1, alpha=.95)
        m = metrics[key]
        ax_nav.scatter(pd.Timestamp(m["peak_date"]), m["peak_nav"], marker="^", color=COLORS[key], s=38, edgecolors="white", zorder=5)
        ax_nav.scatter(pd.Timestamp(m["trough_date"]), m["trough_nav"], marker="v", color=COLORS[key], s=38, edgecolors="white", zorder=5)
        ax_dd.scatter(pd.Timestamp(m["trough_date"]), m["max_drawdown"], color=COLORS[key], s=36, edgecolors="white", zorder=5)
        ax_nav.text(dates.iloc[-1] + pd.Timedelta(days=35), m["end_nav"], f"{m['end_nav']:.3f}", color=COLORS[key], va="center", fontsize=11, weight="bold")
    ax_nav.axhline(1, color="#b2bac4", lw=.7, ls="--")
    ax_nav.set_ylabel("账户净值（初始 = 1）")
    ax_nav.set_ylim(.69, 1.61)
    fig.legend(*ax_nav.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(.061, .879), fontsize=10, ncol=3, frameon=False)
    annotation(ax_nav, "共同回撤起点\n2021-02-10", (pd.Timestamp("2021-02-10"), metrics["BUY_HOLD"]["peak_nav"]), (pd.Timestamp("2022-01-15"), 1.45), color="#596575")
    ax_dd.set_ylabel("相对历史高点回撤")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_dd.set_ylim(-.485, .016)
    ax_dd.set_yticks([0, -.1, -.2, -.3, -.4])
    annotation(ax_dd, "C  −25.89%", (pd.Timestamp("2024-02-02"), metrics["C"]["max_drawdown"]), (pd.Timestamp("2023-03-01"), -.185), color=COLORS["C"])
    annotation(ax_dd, "A  −31.11%", (pd.Timestamp("2024-09-13"), metrics["A"]["max_drawdown"]), (pd.Timestamp("2025-03-01"), -.325), color=COLORS["A"])
    annotation(ax_dd, "买入持有  −40.95%", (pd.Timestamp("2024-02-02"), metrics["BUY_HOLD"]["max_drawdown"]), (pd.Timestamp("2021-11-01"), -.454), color=COLORS["BUY_HOLD"])
    ax_exp.set_ylabel("收盘股票仓位")
    ax_exp.set_ylim(-.025, 1.045)
    ax_exp.set_yticks([0, .5, 1])
    ax_exp.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax_exp.xaxis.set_major_locator(mdates.YearLocator())
    ax_exp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_exp.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2027-01-01"))
    for ax in [ax_nav, ax_dd, ax_exp]:
        ax.grid(axis="y", color="#e1e5ea", lw=.65)
        ax.tick_params(axis="both", length=0)
    plt.setp(ax_nav.get_xticklabels(), visible=False)
    plt.setp(ax_dd.get_xticklabels(), visible=False)
    for i, key in enumerate(["A", "C", "BUY_HOLD"]):
        m = metrics[key]
        top = .85 - i * .216
        fig.text(.772, top, LABELS[key], fontsize=13, weight="bold", color=COLORS[key])
        lines = [
            f"年化收益    {m['annual_return']:.2%}",
            f"净夏普       {m['sharpe']:.3f}",
            f"最大回撤    {m['max_drawdown']:.2%}",
            f"平均仓位    {m['mean_exposure']:.2%}",
            f"回撤谷底    {m['trough_date']}",
        ]
        for j, line in enumerate(lines):
            fig.text(.774, top - .034 - j * .027, line, fontsize=11.5, color="#374456")
    fig.text(.772, .18, "三角标记：最大回撤的峰与谷\n曲线保留每一个日度观察", fontsize=10, color="#687483")
    fig.text(.06, .078, "图中 A / C 是已保存的历史配置结果；尚未加入“先涨后回撤退出”规则。B / D 因宏观预测门槛未通过，账户未运行。", fontsize=10.5, color="#596575")
    fig.text(.06, .05, "净值包含现金、分红及应收分红、交易成本；242 日年化，现金和无风险收益率为 0。回撤含初始净值 1，期末不强制卖出。", fontsize=10.5, color="#596575")
    fig.savefig(out / "02_实际历史_完整日度净值回撤仓位.png", dpi=160)
    fig.savefig(out / "02_实际历史_完整日度净值回撤仓位.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制完整日度历史与人工退出机制，或仅复核图表数据。")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data, metrics, sources = load_accounts(args.source_root)
    artificial, info = artificial_path()
    if args.verify_only:
        saved = pd.read_parquet(args.output / "完整日度净值回撤仓位.parquet")
        pd.testing.assert_frame_equal(data, saved, check_exact=True)
        saved_metrics = json.loads((args.output / "账户指标.json").read_text(encoding="utf-8"))
        assert metrics == saved_metrics, "指标与保存结果不一致"
        saved_toy = pd.read_csv(args.output / "人工路径与逐日状态.csv")
        pd.testing.assert_frame_equal(artificial, saved_toy, check_dtype=False, atol=1e-12)
        assert info == json.loads((args.output / "机制示意设定.json").read_text(encoding="utf-8"))
        print("复核通过：日度数据、账户指标和人工状态路径一致；新增账户、拟合、下载均为 0。")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    data.to_parquet(args.output / "完整日度净值回撤仓位.parquet", index=False)
    data.to_csv(args.output / "完整日度净值回撤仓位.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    artificial.to_csv(args.output / "人工路径与逐日状态.csv", index=False, encoding="utf-8-sig")
    dump(args.output / "账户指标.json", metrics)
    dump(args.output / "机制示意设定.json", info)
    dump(args.output / "来源与数值复核.json", {
        "status": "PASS_SAVED_LEDGER_ALIGNMENT_IDENTITIES_AND_METRICS",
        "daily_rows_per_account": len(data), "source_files": sources,
        "new_accounts": 0, "new_fits": 0, "new_downloads": 0,
        "illustrated_exit_rule_backtest": "NOT_RUN", "parameter_optimization": False,
        "max_drawdown_includes_initial_nav_one": True, "all_daily_points_plotted": True,
        "source_completed_json_metrics_equal": True,
    })
    mechanism_plot(args.output, artificial, info)
    account_plot(args.output, data, metrics)
    print(f"已保存图形与完整日度数据：{args.output}")
    print("人工路径只解释机制；真实历史图未加入新的退出规则。")


if __name__ == "__main__":
    main()
