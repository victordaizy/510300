"""整理固定风险情景的保存结果并打包；不拟合、不重新生成账户。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_option_volatility_account_risk_v1"
PARENT = ROOT / "reports/research/510300_option_volatility_probability_payoff_v1"
OUTPUT = ROOT / "deliverables/510300_期权10万元_三档回撤与一年目标_20260924.zip"
MODEL = "MODEL_POSITIVE_EXPECTANCY"
REFERENCE = "ALWAYS_PROTECTED_SHORT_REFERENCE"
NAMES = {"R1_D10": "单次风险1% / 回撤退出线10%", "R2_D20": "单次风险2% / 回撤退出线20%", "R4_D30": "单次风险4% / 回撤退出线30%"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def describe_results() -> None:
    metrics = pd.read_csv(SOURCE / "results/全部账户指标.csv")
    years = pd.read_csv(SOURCE / "results/完整自然年与部分年.csv")
    cycles = pd.read_csv(SOURCE / "results/完整持仓周期.csv")
    predictions = pd.read_csv(SOURCE / "inputs/predictions.csv")
    primary = metrics[metrics.policy.eq(MODEL) & metrics.scenario.eq("STRESS")]
    short = metrics[metrics.policy.eq(REFERENCE) & metrics.scenario.eq("STRESS")]
    positive = predictions[predictions.model.eq("VOLATILITY_LOGIT_GAMMA") & predictions.expected_net_risk_unit.gt(0)]
    conclusion = {
        "status": "REJECTED_CURRENT_FIXED_MODELS_NOT_SCALABLE_TO_TARGET",
        "meaning": "本轮固定模型、结构与三档账户均未满足目标；不据此断言所有510300期权策略都不可能盈利。",
        "total_accounts": int(len(metrics)),
        "accounts_with_any_full_year_doubling": int(metrics.complete_year_doubling_count.gt(0).sum()),
        "accounts_with_any_rolling_year_doubling": int(metrics.rolling_year_doubling_windows.gt(0).sum()),
        "accounts_with_sharpe_at_least_1_3": int(metrics.net_sharpe.ge(1.3).sum()),
        "primary_accounts_with_at_least_five_each_complete_year": int(primary.minimum_completed_cycles_full_year.ge(5).sum()),
        "zero_volatility_sharpe": "未定义，不作达标",
        "fresh_independent_holdout": False,
        "future_success_probability": "NOT_COMPUTED",
        "next_step_not_run": "若继续研究，先预先定义能解释未来实现波动、方向变盘或期权价格重估的新节点，再检查扣费后的预测增量；不在本次失败曲线上加杠杆、挑窗口或调阈值。",
    }
    (SOURCE / "研究结论.json").write_bytes(encode(conclusion))
    lines = [
        "# 10万元510300期权：不同风险预算的结果",
        "",
        "本轮完成18个固定账户。没有账户达到净夏普1.3，也没有完整自然年或滚动一年翻倍的账户。主模型中实际交易的两档均亏损，1%单次风险档因整张合约的最低资金需求而持续空仓。当前这套信号不支持用提高风险预算来实现一年10万元到20万元。",
        "",
        "本金10万元；观察期2021-03-02至2026-08-14，共1325个交易日。以下使用压力成本：每张每腿每侧5元，另扣max(0.0005元,1%权利金)的不利价格变化。所有空仓日计入全账户日收益；现金及夏普基准利率假设为0。",
        "",
        "|研究情景|全期期末净值|年化净收益|净夏普|实际最大回撤|交易总数|最好滚动一年|最差滚动一年|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in primary.itertuples(index=False):
        sharpe = "未定义（未交易）" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        lines.append(f"|{NAMES[r.profile]}|{r.ending_equity_cny:,.2f}元|{r.annualized_return:.2%}|{sharpe}|{abs(r.max_drawdown):.2%}|{r.completed_cycles}|{r.best_rolling_year_return:.2%}|{r.worst_rolling_year_return:.2%}|")
    lines += [
        "",
        "期末金额是约五年半的累计净值，不能解释为一年结果。完整自然年为2022—2025；滚动一年使用日历周年日之后第一个交易日，可能因节假日略超365/366天。每档1083个滚动窗口高度重叠，不是1083次独立试验，更不是未来成功概率。",
        "",
        "## 为什么还没有正收益优势",
        "",
        f"- 主模型保存的正期望信号共{len(positive)}个，全部属于买入平值跨式；保护性卖跨式没有预测净期望大于0的信号。此处没有追加胜率阈值或选择事后赢家。",
        f"- 这些信号的事前一套风险尺度最低{positive.known_risk_scale_cny.min():,.0f}元，高于10万元账户的1%预算1,000元。因此1%档无成交；这不是低风险盈利策略。",
    ]
    for r in primary.itertuples(index=False):
        chosen = cycles[cycles.account_id.eq(r.account_id)]
        if chosen.empty:
            continue
        gains = chosen.loc[chosen.net_pnl_cny.gt(0), "net_pnl_cny"]
        losses = -chosen.loc[chosen.net_pnl_cny.lt(0), "net_pnl_cny"]
        average_gain, average_loss = gains.mean(), losses.mean()
        break_even = average_loss / (average_gain + average_loss)
        gross = r.net_profit_cny + r.fees_cny + r.slippage_cost_cny
        lines.append(f"- {NAMES[r.profile]}：实现胜率{r.win_rate:.2%}，平均盈利{average_gain:,.2f}元、平均亏损{average_loss:,.2f}元，平均盈亏比{r.cash_payoff_ratio:.2f}。按这组已实现平均幅度计算的盈亏平衡胜率为{break_even:.2%}，该数只是事后描述。累计净损益{r.net_profit_cny:,.2f}元，加回费用和假设滑点后仍为{gross:,.2f}元，当前亏损不只是由本轮成本扣减造成。")
    lines += [
        "",
        "## 每年不少于5次交易的检查",
        "",
        "|完整自然年|1%风险档：收益 / 交易数|2%风险档：收益 / 交易数|4%风险档：收益 / 交易数|",
        "|---|---:|---:|---:|",
    ]
    for year in [2022, 2023, 2024, 2025]:
        cells = []
        for account in primary.account_id:
            row = years[years.account_id.eq(account) & years.year.eq(year)].iloc[0]
            cells.append(f"{row.net_return:.2%} / {row.completed_cycles_by_entry_year}次")
        lines.append(f"|{year}|" + "|".join(cells) + "|")
    lines += [
        "",
        "交易数按入场年份归属且必须最终完成平仓。三档均不满足每个完整自然年至少5次交易；不能用五年总次数除以年份替代这一要求。",
        "",
        "## 同时卖出认购与认沽的对照",
        "",
        "为观察收权利金与保证金的效果，另保留有保护翼的卖跨式常开账户。它只是一项固定对照，主模型没有产生对它的正期望入场判断。压力成本结果如下：",
        "",
        "|风险情景|全期期末净值|净夏普|实际最大回撤|完整交易|",
        "|---|---:|---:|---:|---:|",
    ]
    for r in short.itertuples(index=False):
        lines.append(f"|{NAMES[r.profile]}|{r.ending_equity_cny:,.2f}元|{r.net_sharpe:.3f}|{r.max_drawdown:.2%}|{r.completed_cycles}|")
    lines += [
        "",
        "4%单次风险、30%退出线的卖跨式对照在2025-12-16收盘触发停止开仓，并在下一交易日开盘平仓，最终最大回撤30.34%。退出线不是保证上限。三个主模型账户都未触及各自退出线，因此其差异主要由风险预算和整张取整所导致，不能把它们视为实际承受10%、20%、30%最大回撤后的收益前沿。",
        "",
        "## 执行假设与条款还原",
        "",
        "- 入场信号只读取事前可用字段，按已保存的下一交易日开盘代理入场，原第10个持有交易日收盘代理退出；事前按单腿最小成交量和持仓量的1%限制张数。一套组合未平仓时不再开下一套。",
        "- 若次日开盘代理下整组最大到期损失超过此前的资金预算，整组取消；未用未来标签是否完整来决定是否交易。基准与压力成本可能因该预算检查产生不同成交路径。",
        "- 两种成本情景下所有实际交易账户的累计净利润均为负。未交易的账户为0收益，夏普未定义；没有把它们当成合格方案。",
        "- 卖出所得权利金同时对应期权负债；保证金只计冻结占用。组合持有保证金保守相加两个信用价差宽度，并增加20%假设余量；入场另检查裸空短腿的过渡准备金。它是统一研究假设，不是历年券商实际收取的保证金记录。",
        "- 分红调整按原单位×前收盘价/(前收盘价−每份现金红利)取整数，再调整行权价；交易代码的原始行权价后缀不能代替调整后的真实行权价。[上交所调整说明](https://star.sse.com.cn/assortment/options/guide/c/c_20161121_4201435.shtml)",
        "- 共还原610条合约调整，与27,870行当日官方调整简称的行权价相符。6行上市首日的代码信息缺失，保留为未知；本轮实际持仓未依赖这6行，所有账户均无缺失估值日。",
        "- 保存结果复算已核对逐腿报价、有效单位、费用、现金流、持仓数量、逐日净值、周期损益、年度收益及滚动一年指标。账目一致不等于预测有效。",
        "- 日线开收盘不是同步组合买卖报价；四腿一次完成、假设滑点与保证金安排没有实盘验证。日收盘最大回撤也没有覆盖盘中最大损失。",
        "- 这段历史已经被多次研究，不能当成全新独立验证，更不能据此保证将来稳定盈利或一年翻倍。未增加行情采集，未重新拟合模型。",
        "",
        "## 后续边界",
        "",
        "当前固定版本保留为未达标结果，不用扩大仓位、挑选少数行情窗口或反复调整阈值挽救。若继续增加节点，应先定义新的、事前可解释的变盘或波动重估机制，检验预测的扣费增量后再做新的账户；该新节点研究本轮尚未运行。",
        "",
        "参考：[上交所组合保证金规则](https://www.sse.com.cn/lawandrules/sselawsrules2025/option/c/c_20250610_10781452.shtml)，[2026年1月510300期权实际调整公告](https://www.sse.com.cn/assortment/options/disclo/update/c/c_20260116_10805396.shtml)。",
    ]
    (SOURCE / "风险比较结论.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_archive(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    if len(names) != len(set(names)) or archive.testzip() is not None:
        raise RuntimeError("压缩包重复成员或CRC检查失败。")
    index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
    if len(index) != len({row["path"] for row in index}):
        raise RuntimeError("压缩包索引出现重复路径。")
    if set(row["path"] for row in index) != set(names) - {"FILE_INDEX.csv"}:
        raise RuntimeError("压缩包索引覆盖不完整。")
    for row in index:
        content = archive.read(row["path"])
        if len(content) != int(row["size_bytes"]) or sha(content) != row["sha256"]:
            raise RuntimeError(f"成员大小或指纹不符：{row['path']}")


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("交付包已存在，禁止覆盖已交付快照。")
    verified = json.loads((SOURCE / "verification/保存结果复算.json").read_text(encoding="utf-8"))
    if verified["status"] != "PASS_SAVED_CASHFLOWS_NAV_CYCLES_AND_ONE_YEAR_METRICS":
        raise RuntimeError("保存账户尚未复核通过。")
    freeze = json.loads((SOURCE / "freeze.json").read_text(encoding="utf-8"))
    for name, expected in freeze["files"].items():
        if sha((SOURCE / name).read_bytes()) != expected:
            raise RuntimeError(f"冻结文件不符：{name}")
    parent_receipt = json.loads((PARENT / "delivery_receipt.json").read_text(encoding="utf-8"))
    parent_archive = Path(parent_receipt["path"])
    if sha(parent_archive.read_bytes()) != parent_receipt["sha256"]:
        raise RuntimeError("前轮原始交付包身份不符，不能引用其快照。")
    describe_results()
    records: dict[str, bytes] = {}
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file() and path.name != "delivery_receipt.json" and "__pycache__" not in path.parts:
            records[path.relative_to(SOURCE).as_posix()] = path.read_bytes()
    with zipfile.ZipFile(parent_archive) as archive:
        check_archive(archive)
        for name in archive.namelist():
            records["前轮模型研究/" + name] = archive.read(name)
        for name in ["option_eod.parquet", "option_risk.parquet", "etf_market.parquet"]:
            if sha(records["inputs/" + name]) != sha(archive.read("inputs/" + name)):
                raise RuntimeError(f"前轮与本轮原始数据身份不一致：{name}")
        if sha(records["inputs/predictions.csv"]) != sha(archive.read("results/完整滚动预测.csv")):
            raise RuntimeError("本轮使用的预测与前轮原始快照不一致。")
    records["前轮模型研究/原交付收据.json"] = encode(parent_receipt)
    records["code/package_option_volatility_account_risk_v1.py"] = Path(__file__).read_bytes()
    records["用户当前指令.md"] = (
        "# 本轮用户指令\n\n"
        "- 我只做510300期权，高夏普，稳定盈利，可以多空双开，波动率赚钱。\n"
        "- 继续，我要一年从期权10万做到二十万。\n"
        "- 对最大回撤的进一步答复：先看不同回撤下的结果。\n\n"
        "沿用最新夏普至少1.3、每完整自然年至少5次交易、先训练胜率与盈亏幅度、拒绝过拟合、不新增采集。"
        "本轮本金10万元取代此前20万元；风险档仅用于研究比较，10%、20%、30%退出线并非保证回撤上限。"
        "前轮模型研究目录内的20万元旧指令是不可变历史背景，当前以本文件及inputs/mandate.json为准。\n"
    ).encode("utf-8")
    records["00_README_FIRST.md"] = (
        "# 510300期权10万元账户：三档风险与一年目标\n\n"
        "结论：18个固定账户中，0个达到夏普1.3，0个出现完整自然年或滚动一年翻倍。"
        "主模型有交易的两档均亏损；1%风险档因整数组合最低资金需求而未交易。扩大风险未解决信号的盈利问题。\n\n"
        "阅读顺序：用户当前指令.md → 风险比较结论.md → results/三档风险结果_中文.csv → "
        "results/完整自然年与部分年.csv → results/完整滚动一年.csv → protocol.json → verification/保存结果复算.json。\n\n"
        "inputs包含直接日线、每日官方合约代码、ETF历史行情与分红、前轮预测/合约腿/模型参数和新资金约束。"
        "code包含冻结账户实现和本次打包代码。results包含18个账户的完整日净值、逐腿现金流、持仓、完成周期、"
        "决策、重建逐日条款、逐次分红调整和指标。freeze.json记录首次生成新账户前的协议、代码与直接输入指纹。\n\n"
        "前轮模型研究/是先核对SHA-256后从原交付包读取的完整历史快照，保留首轮模型训练及既有失败背景。"
        "该目录中的旧20万元目标不替代本轮10万元目标，根目录FILE_INDEX.csv是本包权威成员索引。\n\n"
        "只读复算：用安装了pandas、numpy和parquet读取引擎的Python，运行code/option_volatility_account_risk_v1.py "
        "verify --root 解压目录。只核对保存现金流与指标，不训练、不生成新账户、不下载行情；写入verification结果记录。"
        "前轮模型也保留独立verify入口，可对其保存参数、标签和预测做复核。\n\n"
        "结果依赖日线成交代理、滑点、费用和统一保证金假设；没有同步历史组合盘口或全新独立留出验证。"
        "本包结构检查与账务复算不代表外部GPT已审阅，不证明未来盈利，也不构成下单授权。\n\n"
        "没有指定上传字节上限；本包按完整直接证据打包。未上传外部服务。排除项见02_范围与排除项.md。\n"
    ).encode("utf-8")
    records["01_给GPT的复核提示.md"] = (
        "请审阅这个510300 ETF期权研究包。用户要求10万元本金、一年20万元、净夏普至少1.3、"
        "每个完整自然年至少5次交易，并希望比较不同回撤。请先读00_README_FIRST.md、用户当前指令.md和风险比较结论.md。\n\n"
        "本轮18个账户由已保存的预测映射而来，没有重新训练；全部实际交易账户累计净亏损。请核对：\n"
        "1. 信号时钟、未来标签隔离、完整未知样本、入场数量取整及次日超预算整组拒绝是否正确。\n"
        "2. 每日合约单位/行权价、现金分红、权利金现金与期权负债、双边费用、保证金及逐日现金账是否一致。\n"
        "3. 实际回撤与名义退出线、无交易账户未定义夏普、完整自然年交易次数是否恰当报告。\n"
        "4. 最好滚动一年是否被误当未来翻倍概率，窗口重叠或历史反复研究是否影响证据强度。\n"
        "5. 正期望筛选为何没有实现盈利，高盈亏比与胜率的组合是否支持扩大风险。\n\n"
        "请给出最重要的错误/反证、需要保留的结论、下一阶段节点研究优先级、明确验证条件与停止条件。"
        "不要通过挑年份、提高杠杆、改变翼宽/持有期/阈值来救回当前失败；提出新研究时标为尚未运行。"
        "请明确区分账务核对通过、模拟执行假设、预测能力和未来盈利。此包不授权交易，也不表示你此前已审阅。\n"
    ).encode("utf-8")
    records["02_范围与排除项.md"] = (
        "只包含本轮账户的全部直接输入、冻结实现、全账户结果，以及已验证身份的前轮完整训练快照。\n\n"
        "排除：无关全仓库、其他资产、IO指数期权、各合约仅末几日的分钟数据、零散盘口快照、采集服务、"
        "券商凭据及实盘系统。没有借助新行情补齐结果；6行无法还原的首次上市条款保留未知，本轮没有实际持仓依赖它们。\n\n"
        "读过的官方规则网址保留在protocol.json及风险比较结论.md，网页正文不属于本包的离线镜像。"
        "分红CSV源链接保留，未把券商历年真实保证金或同步期权盘口当成已拥有的数据。\n\n"
        "本轮没有独立留出样本；新变盘或波动重估节点尚未运行。未上传，外部评审状态为NOT_PERFORMED。\n"
    ).encode("utf-8")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["path", "size_bytes", "sha256"])
    writer.writeheader()
    for name, content in sorted(records.items()):
        writer.writerow({"path": name, "size_bytes": len(content), "sha256": sha(content)})
    records["FILE_INDEX.csv"] = stream.getvalue().encode("utf-8-sig")
    temporary = OUTPUT.with_suffix(".building.zip")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(records.items()):
            archive.writestr(name, content)
    with zipfile.ZipFile(temporary) as archive:
        check_archive(archive)
        for name, expected in freeze["files"].items():
            if sha(archive.read(name)) != expected:
                raise RuntimeError(f"压缩包冻结成员不符：{name}")
    os.replace(temporary, OUTPUT)
    receipt = {
        "path": str(OUTPUT), "size_bytes": OUTPUT.stat().st_size, "sha256": sha(OUTPUT.read_bytes()),
        "members": len(records), "index_rows": len(records)-1, "zip_structure": "PASS",
        "frozen_member_coverage": "PASS", "parent_archive_identity": "PASS",
        "saved_account_recomputation": verified["status"], "saved_account_count": verified["accounts"],
        "fills_recomputed": verified["fills"], "complete_cycles_recomputed": verified["completed_cycles"],
        "new_accounts_during_packaging": 0, "new_fits_during_packaging": 0, "new_downloads_during_packaging": 0,
        "external_review": "NOT_PERFORMED", "goal_achieved": False,
    }
    (SOURCE / "delivery_receipt.json").write_bytes(encode(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
