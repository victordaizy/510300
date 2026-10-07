"""写入D02实证、B02设计拒绝及十一函数有限结论；保留其他分支和原前瞻登记。"""
import json
import os
import re

from research.point_account_cashflow_state_v1 import ROOT, now, digest, require, write_json
from research.point_d02_optional_correction_v1 import check


OUT = ROOT / "reports/research/510300_point_d02_optional_correction_v1"
REL = OUT.relative_to(ROOT).as_posix()
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
DOCS = ["docs/PROJECT_STATE.md", "docs/RESEARCH_DECISIONS.md",
        "docs/PROJECT_STATE_TECHNICAL_LINE.md", "docs/RESEARCH_DECISIONS_TECHNICAL_LINE.md"]


def read(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def protected(key):
    return "forward" in key or "prospective" in key or key in {
        "earliest_future_exchange_session", "next_new_close_eligible_at", "registered_candidate_intents",
        "accepted_independent_candidates", "current_validated_candidates",
    }


def replace_bytes(path, before, after):
    require(path.read_bytes() == before, "共享事实在准备后已改变，停止覆盖：" + str(path))
    temporary = path.with_name(path.name + ".tech_R139.tmp")
    with temporary.open("xb") as stream:
        stream.write(after)
    os.replace(temporary, path)


def main():
    require(not (OUT / "facts_update_receipt.json").exists(), "本轮事实已经提交，不重复写入。")
    result, verify = read(OUT / "prediction_summary.json"), read(OUT / "verification.json")
    source, tests = read(OUT / "source_summary.json"), read(OUT / "tests_receipt.json")
    pre = read(OUT / "next_B02_saved_field_design_preflight.json")
    finite, route = read(OUT / "finite_optional_phase_summary.json"), read(OUT / "next_research_route.json")
    require(result["technical_decision"] == "TECH.R137" and not result["prediction_gate_passed"]
            and not any(p["prediction_gate_passed"] for p in result["periods"]), "D02原双期拒绝终态不同。")
    require(check() == 49 and verify["maximum_prediction_error"] == 0 and tests["tests"] == 15
            and tests["passed"], "D02必要测试、冻结或复算凭据不一致。")
    require(not pre["design_gate_passed"] and not pre["target_column_read"]
            and finite["function_configurations"] == 11 and finite["accepted_function_configurations"] == 0,
            "B02或有限阶段裁决不同。")
    overview = (
        "D02低开缺口吸收/有符号缺口可选两系数实际比较拒绝，最新实际模型TECH.R137。"
        "原762/1507联合已知、745联合不完整及其中已知gap保留，362吸收真0/327大于1保持；"
        "原八项/142月115可用27未知不变，25辅助估计/90复用、核心0重拟合，"
        "1010完整配对/497原未知、533活动/477精确回退、4负与非负预测判断改变而非4交易。"
        "早期MSE下降0.789156%、近期下降0.194511%，改善95%区间均跨零，整体FAIL；"
        "经济SKIPPED、收益夏普NOT_COMPUTED、新标签/账户0。15必要测试首次通过，"
        "49原冻结对象/3488字段/142版本/25方程/1507预测/24周期误差复算通过，预测差0。"
        "TECH.R138另拒绝B02原五列可选设计资格：43/3488日线、15/1507自然状态有值，"
        "115成熟月112秩5、2017年3/4/5月三份秩3，每月完整记录4—12；"
        "可用预测早期1/262、近期8/748有五项，未读目标、0金融拟合。"
        "TECH.R139有限汇总十一固定函数全部整体门拒绝，仅J01/B04早期通过、近期全部失败；"
        "合计275辅助估计/990复用，不是独立策略数，0新账户。结束这些固定表达，"
        "不按有利时期挑选或拼接；原R94/R102与旧账户失败、E03登记及其他分支保持，0行情/联网。"
        "下一具体实验为原SAVED_WEIGHT/POINT_BINARY真实前瞻比较，首个合法新收盘仍2026-10-08 15:05，"
        "当前0真实新账户日/闭合周期；新实质机制须另核旧用途、来源和支持再冻结，当前没有新金融候选准入。"
        "独立验证、去过拟合和完整收益夏普目标未达。"
    )
    rows = []
    for p in result["periods"]:
        ci = p["improvement_interval"]
        rows.append(f"| {p['period']} | {p['paired_rows']} / {p['cycles']} | {p['optional_input_known_rows']} / {p['exact_fallback_rows']} | {p['baseline_raw_return_mse']:.12f} | {p['candidate_raw_return_mse']:.12f} | {p['relative_mse_change'] * 100:+.6f}% | [{ci['low']:.12f}, {ci['high']:.12f}] | FAIL |")
    table = "\n".join(rows)
    report = f"""# D02低开缺口的日内吸收与缺口大小：实际结果和有限阶段结论

实际完成：{result['at']}。TECH.R136为唯一事前模型协议，TECH.R137为实际结果，TECH.R138为另立的B02无标签设计资格，TECH.R139为十一函数的有限阶段总结。代码为隔离新增，原冻结策略和前瞻登记保持。

## 结论

D02状态为{result['status']}。两期MSE点估计下降0.789156%与0.194511%，但原入场年块改进95%区间均跨零，整体拒绝。经济阶段{result['economic_stage']}，新账户0，净CAGR/净夏普NOT_COMPUTED。四个预测负与非负判断改变是参考状态变化，不能称四次新增交易或收益提高。独立验证、去过拟合和完整目标未达。

| 固定时期 | 完整配对 / 周期 | 两项已知 / 精确回退 | 原MSE | D02 MSE | 相对变化 | 改进95%区间 | 门 |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

区间沿用5000次/seed51030099原入场年块，早期2017/2018/2019三块，近期2020/2021/2022/2024/2025/2026六块，是开发历史敏感性，独立性未建立。1010参考状态不是1010独立交易，只有原6/18自然周期；原目标是自然终点继续退出相对下一合法开盘提前退出的BASE扣费增量，以当前剩余持有价值为分母，不是完整账户收益，也不是A实际剩余现金流。

## 原字段和单独可选函数

0.001整数报价/现金单位按原规则验收，不修复离格值。deltaCO=open_units+cash_units−previous_units；rCO=log1p(deltaCO/previous_units)，rOC=log1p((close_units−open_units)/(open_units+cash_units))，其和等于原现金经济log收益。只在严格deltaCO<0时，negative_gap_absorption=max(rOC,0)/abs(rCO)有定义；signed_overnight_gap始终是原有符号rCO。日内平/跌时吸收真0有效，大于1保留，不对原比例先log或截断，不补epsilon。完整T收盘15:05可知，下一合法开盘应用。

正/零隔夜时吸收保持NaN，原已知gap分别保留，不强制两列都NaN、不把未知补0。首个市场区间两列原未知、坏源保持未知；当前快照源合同原失败则拒绝，不按未来修复删除成员。3488日线1791联合已知，1507自然状态762联合有值/745联合不完整；原gap在全部1507自然状态已知，745不完整行的gap未删。已知自然状态362吸收真0、327吸收>1。

原TECH.R94仅测试两列直接追加的完整原成员支持，115原可用训练月完整0，来源拒绝保持。旧HIGH/LOW八保存账户、gap_recovery、DAILY01及其他日内隔夜用途失败不重跑，不混原2万/20万与当前252日双期口径。PCA的absorption是成分股协方差特征值比例，不是本缺口吸收，原同名纠正保持。本模型没有新增物理信息或证明首次发布版本。

另立的固定两系数修正保留原八项参数/尺度/截距、全部原成熟行/目标、周期总权重1、原最近20/至少10周期100行、入场锁定与两负确认。联合已知原训练行按原周期权重标准化、clip正负5并重新中心化；联合不完整原值仍未知，仅修正设计不活动。设计与原目标减固定核心的残差分别周期内去均值，beta=(Dx.T W Dx+I2)^(-1)Dx.T W Dy，alpha1、新截距0；不完整输入精确返回原核心浮点值，原核心未知时双方未知。没有缺失指示、条件子样本、换窗口/方向/标签。

115成熟月两列设计资格原预检通过，最小特征值2.1711608409818477。实际142原模型记录、115可用27未知；25次不同训练输入辅助估计、90月版本复用、核心0重拟合。1010完整配对/497原未知；早期133活动129回退，近期400活动348回退，合计533活动477精确回退。新标签、策略账户、行情与联网均0。

## 必要验证

15必要测试首次通过，15 passed in 6.70s。覆盖原严格负缺口现金时钟、真0与>1、坏源和未来前缀、原单位和经济身份、开盘不同而收盘同、原成熟成员；新测试覆盖原身份15:05、原部分未知gap保留、两列闭式方程、全部原行/原核心固定、联合不完整精确回退、全未知训练、周期共同误差消除、未知原成员对周期对比保留，以及原比例与缺口域。

原49冻结对象保持。3488日线原字段与R94精确相同，142原版本、25不同方程、1507原状态预测与24周期误差复算通过；最大预测差0，最大方程梯度1.3010426069826053e−17，最大全局设计均值9.251858538542972e−17。复算0拟合/0账户。这些验证说明保存实现与协议一致，不证明金融优势或独立验证。

## B02五项记录的下一资格检查

另在读取设计前冻结TECH.R138无目标合同，检查原二测金额比、首次/本次经济log收益、首次/本次CLV五列。原20日低点、首次后两日确认、a+3..a+10认领、严格0.5ATR、均额与原未知/真0和1均保持。核22旧B02冻结源与49父D02源，日线字段与R102原保存值完全相同。

43/3488日线有五项，15/1507自然状态完整、1492未知。原115成熟月中112秩5，2017-03-01、2017-04-05、2017-05-02仅4完整训练行/3有记录周期，设计秩3；全部成熟月完整训练记录4—12。标准SVD容差在测量前固定为max(shape)*机器精度*最大奇异值；岭惩罚可逆不等于数据提供五个识别方向。原可用预测早期1/262、近期8/748具备完整五项。

本固定五列可选函数资格拒绝，收益目标列未读取，金融拟合/账户0。不得删两列、跳过原前三月、改窗口或只看后半期救回。原R102完整成员支持失败、T02旧失败和T13源门NOT_RUN分别保持，不把设计拒绝写成已回测收益失败。首次技术运行在生成资格协议前因reports联接到E盘路径标签而退出；修正工作区逻辑路径后登记并测量，没有修改来源、定义、容差或结果门。

## 有限阶段和下一步

十一固定可选函数全部整体门失败，仅J01/B04早期单期通过，近期全部失败。完整比较见[有限阶段结论](有限可选修正阶段结论.md)与[逐项保存结果](finite_optional_phase_summary.json)。累计275辅助估计/990月复用，同原样本/滚动版本，不是275独立策略，不累计共用周期。当前没有新账户收益或夏普结果，也不能从有利单期挑选函数、混合或调参。

这批固定表达开发结束。下一具体实验沿原E03 SAVED_WEIGHT/POINT_BINARY登记开展真实前瞻比较；首个合法新收盘仍2026-10-08 15:05，真实新账户日/闭合周期当前0，不回填、不重置、不从本批结果反选。可在原授权内核对另一个实质不同的决策机制、旧用途、事前来源和完整成员，再另冻结隔离实验；当前[下一路线](next_research_route.json)只登记研究路线，没有新收益模型准入。其他分支的历史研发授权不因本有限阶段结束改变。

长期事实见项目docs/PROJECT_STATE.md、docs/RESEARCH_DECISIONS.md及配套技术线两文件。整个项目的高收益、高夏普、独立验证目标仍未完成。
"""
    with (OUT / "研究结果与下一步.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
    entries = f"""
## TECH.R136 — D02严格负缺口原两项的单独可选函数协议

- 假设：完整当日负缺口吸收比例及有符号缺口可补充原八项的周期内继续价值；这是另立的可选预测函数，不是原R94完整字段支持晋升或新物理信息。
- 验证方法：原26来源/3488字段/1507身份、115月两列识别；15必要测试后冻结唯一alpha1两系数、无新截距，全部原行/目标/权重/八项/月钟/入场锁定与原NaN保留。原双期完整配对MSE均改善且原5000年块95%下界均>0才进入完整20万252日BASE/STRESS账户。
- 结果：协议冻结49对象，真实金融运行一次，详见TECH.R137。必要测试首次15 passed in 6.70s。
- 接受/拒绝原因：接受的是事前定义和完整回退函数具备可运行资格，尚非金融优势；原R94直接支持拒绝及HIGH/LOW、gap_recovery、DAILY01、PCA同名区别保持。
- 是否需要重新验证：不重跑或近邻参数救回本版本；只有实质新机制、源或真实独立数据另立协议。历史首版/独立验证未建立。
- 直接证据：[协议](../{REL}/protocol.json)、[定义区别](../{REL}/prior_definition_and_function_addendum.json)、[测试](../{REL}/tests_receipt.json)。

## TECH.R137 — D02实际两系数比较：双期区间跨零，整体拒绝

- 假设：TECH.R136固定函数在原双期完整成员上形成稳健预测改进，再可能改进净收益与夏普。
- 验证方法：唯一配置，25实际辅助估计/90复用、原核心0重拟合；142月115可用27未知，1010完整配对/497原未知、6/18周期等权原MSE与原入场年块5000次/seed51030099。
- 结果：早期原MSE0.006796098625414917→0.006742466796448296，下降0.789156%，改善95%[-0.000011815290849140,+0.000345755102002143]；近期0.0055065266523754795→0.005495815842882907，下降0.194511%，95%[-0.000003708510797143,+0.000025993746217583]。两期门FAIL、整体REJECTED；762原字段完整745不完整保持、533配对活动477精确回退，4负与非负判断改变不等于4交易。15测试通过，49源/3488字段/142版本/25方程/1507预测/24周期损失保存复算通过、预测差0、复算0拟合。
- 接受/拒绝原因：点估计两期下降不足满足事前区间门，不选时期或弱化门；经济SKIPPED，新账户/标签0，收益夏普NOT_COMPUTED。金融增量拒绝，实现复算接受，旧R94及相关失败保持。
- 是否需要重新验证：本固定开发版本不复跑、不改原正/零缺口NaN、原已知gap、真0或>1；未来新假设或真实独立数据另立。DEVELOPMENT_CALIBRATION、首版未认证、DSR/PBO NOT_COMPUTED、去过拟合/完整目标未达。
- 直接证据：[结果](../{REL}/prediction_summary.json)、[原字段](../{REL}/source_summary.json)、[复算](../{REL}/verification.json)、[完整报告](../{REL}/研究结果与下一步.md)。

## TECH.R138 — B02原五项二次试低的无标签设计资格拒绝

- 假设：原五项记录可在全部115原可用成熟月提供五个可识别设计方向，形成另一个完整核心回退函数；不重开R102全字段支持失败。
- 验证方法：先冻结无目标合同，核22旧B02/49父D02源，仅读PRICE_COLUMNS/身份/成熟模型元数据，字段与原R102精确比较。全部原行/周期权重保留，已知五项标准化clip正负5中心化再周期内对比，以事前标准SVD容差测rank，115个月均需rank5；岭可逆不是源识别。
- 结果：43/3488日线、15/1507状态完整/1492未知；115月112秩5、2017-03-01/04-05/05-02三月仅4完整行/3记录周期、秩3，完整训练记录4—12。原可用预测早期1/262、近期8/748有五项。设计资格REJECTED，目标列未读，金融拟合/新标签/账户0。协议生成前一次联接路径标签技术退出已说明，仅修正逻辑路径，源/定义/秩规则不变。
- 接受/拒绝原因：原前三可用月无足够五列识别，不能删除两项、跳月或扩窗后宣称同版本可行；接受源同值及记录事实，拒绝该固定五列可选函数资格，不宣称其收益已回测失败。旧R102、T02及T13各自终态保留。
- 是否需要重新验证：当前固定成员与五列不重复；新真实成熟数据或实质不同事前机制另立。112后续月份识别不构成单独截取许可。
- 直接证据：[事前资格协议](../{REL}/next_B02_design_protocol.json)、[实际资格](../{REL}/next_B02_saved_field_design_preflight.json)、[月度设计](../{REL}/next_B02_monthly_design_support.csv)。

## TECH.R139 — 十一固定可选修正的有限阶段结束，0晋升

- 假设：单独的低维日线量价/资金/股债字段可在固定原核心外形成完整回退函数，改善完整双期预测并进一步改善账户。
- 验证方法：只读取K04/K05/H03/J01/C04/A02/C03/C01/B03/B04/D02十一份原冻结实际结果及PASS保存复算，不重新估计、抽样、删成员或挑选时期；并单列B02设计资格。
- 结果：十一函数整体全FAIL、0晋升；仅J01/B04早期通过，近期全失败。累计275实际辅助估计/990月复用，核心0重拟合、新账户/标签0；共用1507自然状态/1010配对/6与18周期不累计为独立证据。B02资格另拒绝、无金融拟合。
- 接受/拒绝原因：结束这些固定表达的开发，拒绝金融增量，不从单期优点拼接或调参。该有限结论不证明所有技术分析或经济机制永远无效，不取消其他分支独立授权。独立验证NOT_ESTABLISHED、全球DSR/PBO NOT_COMPUTED、首版未认证、去过拟合与完整目标未达。
- 是否需要重新验证：本表失败版本不再重跑。下一具体实验为原E03 SAVED_WEIGHT/POINT_BINARY真实前瞻，原注册/首个2026-10-08 15:05/0新账户日与闭合周期保持；新实质机制须核旧用途、事前来源和完整成员后另冻结。当前无新收益模型准入，目标仍active且未完成。
- 直接证据：[完整有限结论](../{REL}/有限可选修正阶段结论.md)、[十一原结果清单](../{REL}/finite_optional_phase_summary.json)、[CSV](../{REL}/有限可选修正实际结果.csv)、[下一路线](../{REL}/next_research_route.json)。
"""
    state_entry = f"""
## 技术线接续：TECH.R136协议 / R137实际拒绝 / R138资格拒绝 / R139有限结论

{overview}

| 固定时期 | 完整配对 / 周期 | 两项已知 / 精确回退 | 原MSE | D02 MSE | 相对变化 | 改进95%区间 | 门 |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

当前核心判断是：这十一项可选修正未提供跨期稳健增量，继续从这些结果选单期、混合或换窗不能作为去过拟合或高夏普证据。B02只是设计资格失败，不是已经运行的金融收益失败。原参考目标仍以剩余持有价值为分母，MSE不是账户净收益；没有新收益或夏普结果，既有A跨期不足的结论保持。新机制资格与原真实前瞻分开，其他分支继续按自身协议处理。

直接证据：[D02报告](../{REL}/研究结果与下一步.md)、[协议](../{REL}/protocol.json)、[真实结果](../{REL}/prediction_summary.json)、[保存复算](../{REL}/verification.json)、[B02事前设计资格](../{REL}/next_B02_saved_field_design_preflight.json)、[有限阶段结论](../{REL}/有限可选修正阶段结论.md)、[下一路线](../{REL}/next_research_route.json)。
"""
    snapshots = OUT / "facts_before_R139"
    snapshots.mkdir(exist_ok=False)
    changes, original_docs = {}, {}
    for filename in DOCS:
        path = ROOT / filename
        before = path.read_bytes()
        original_docs[filename] = before
        with (snapshots / path.name).open("xb") as stream:
            stream.write(before)
        bom = before.startswith(b"\xef\xbb\xbf")
        text = before.decode("utf-8-sig")
        require("最新TECH.R135" in text and not any("TECH.R" + str(i) in text for i in range(136, 140)),
                "技术线入口已经由其他写入更新，不能覆盖：" + filename)
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.splitlines(keepends=True)
        changed = 0
        for i, line in enumerate(lines):
            if line.startswith("> ") and "最新TECH.R135" in line:
                prefix, remainder = line.split("最新TECH.R135", 1)
                require("：" in remainder, "旧技术线顶部行格式不明。")
                lines[i] = prefix + "最新TECH.R139：" + overview + newline
                changed += 1
        require(changed == 1, "技术线顶部入口匹配数量不为1：" + filename)
        appendix = entries if "RESEARCH_DECISIONS" in filename else state_entry
        text = "".join(lines).rstrip("\r\n") + newline + appendix.replace("\n", newline)
        changes[filename] = (b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8")

    state_before = STATE.read_bytes()
    state = json.loads(state_before.decode("utf-8-sig"))
    require(state["latest_actual_model_decision"] == "TECH.R135", "目标当前模型状态已经变化。")
    frozen_forward = {k: v for k, v in state.items() if protected(k)}
    with (OUT / "state_before_R139.json").open("xb") as stream:
        stream.write(state_before)
    previous = read(ROOT / "reports/research/510300_point_b04_optional_correction_v1/prediction_summary.json")["accounting"]
    for key, value in list(state.items()):
        if "this_continuation" in key and not key.startswith("prior_") and not protected(key):
            if isinstance(value, bool):
                state[key] = False
            elif isinstance(value, (int, float)):
                state[key] = 0
    state.update({
        "updated_at": now(), "status": "research_in_progress", "goal_status": "active", "goal_achieved": False,
        "current_phase": "FINITE_ELEVEN_OPTIONAL_FUNCTIONS_AND_B02_DESIGN_CLOSED_NO_ADMISSION",
        "latest_model_decision": "TECH.R137", "latest_actual_model_decision": "TECH.R137",
        "latest_technical_decision": "TECH.R139", "latest_actual_prediction_result": REL + "/prediction_summary.json",
        "latest_D02_optional_verification": REL + "/verification.json",
        "latest_d02_optional_source_summary": REL + "/source_summary.json",
        "latest_D02_design_identifiability": REL + "/design_identifiability_preflight.json",
        "latest_B02_optional_design_protocol": REL + "/next_B02_design_protocol.json",
        "latest_B02_optional_metadata_design_preflight": REL + "/next_B02_saved_field_design_preflight.json",
        "latest_finite_optional_phase_summary": REL + "/finite_optional_phase_summary.json",
        "latest_finite_optional_phase_report": REL + "/有限可选修正阶段结论.md",
        "latest_research_report": REL + "/研究结果与下一步.md",
        "next_information_intake_plan": REL + "/next_research_route.json",
        "next_research_action": REL + "/next_research_route.json",
        "next_experiment_status": route["status"],
        "next_experiment": route["next_concrete_experiment"],
        "goal_turn_classification": "progress",
        "goal_turn_progress_classification": "ACTUAL_D02_COMPARISON_REJECTED_B02_DESIGN_REJECTED_FINITE_PHASE_FACTS_SAVED",
        "blocked_audit_count": 0, "blocking_decision": None,
        "previous_actual_candidate_trials_before_R137": previous,
        "current_candidate_trials": result["accounting"], "actual_candidate_trials": result["accounting"],
        "current_phase_trial_accounting": result["accounting"],
        "finite_phase_trial_accounting": {
            "fixed_function_configurations": 11, "auxiliary_coefficient_estimations": 275,
            "monthly_cache_reuses": 990, "core_model_reestimations": 0,
            "new_accounts": 0, "new_return_labels": 0, "accepted_function_configurations": 0,
            "B02_design_financial_fits": 0,
        },
        "new_model_fits_this_continuation": 25, "new_candidate_coefficient_estimations_this_continuation": 25,
        "auxiliary_coefficient_estimations_this_continuation": 25,
        "total_estimation_calls_this_continuation": 25, "total_return_model_fit_calls_this_continuation": 25,
        "total_fit_calls_this_continuation": 25,
        "reused_monthly_model_records_this_continuation": 90,
        "control_monthly_parameter_checks_this_continuation": 142,
        "original_monthly_membership_checks_this_continuation": 142,
        "saved_field_rows_recomputed_this_continuation": 3488,
        "saved_natural_member_rows_recomputed_this_continuation": 1507,
        "saved_prediction_rows_recomputed_this_continuation": 1507,
        "preserved_no_view_prediction_rows_this_continuation": 497,
        "original_no_model_months_preserved_this_continuation": 27,
        "saved_daily_information_rows_this_continuation": 3488,
        "saved_natural_member_support_rows_this_continuation": 1507,
        "saved_prediction_rows_this_continuation": 1507,
        "empirical_strategy_configurations_this_continuation": 1,
        "new_strategy_configurations_this_continuation": 1,
        "necessary_tests_passed_this_continuation": 15, "necessary_tests_passed_in_current_phase": 15,
        "source_freeze_count_this_continuation": 49,
        "current_phase_source_freeze_count": 49, "current_phase_field_source_freeze_count": 49,
        "code_files_added_this_continuation": ["research/point_d02_optional_correction_v1.py",
                                               "tests/test_point_d02_optional_correction_v1.py"],
        "report_programs_added_this_continuation": [REL + "/next_B02_design_preflight.py",
                                                  REL + "/compile_finite_optional_phase.py",
                                                  REL + "/finalize_research_state.py"],
        "current_phase_original_state_rows": 1507, "current_phase_optional_known_rows": 762,
        "current_phase_optional_unknown_rows": 745, "current_phase_available_paired_predictions": 1010,
        "current_phase_original_unknown_predictions": 497, "current_phase_active_optional_predictions": 533,
        "current_phase_exact_core_fallback_predictions": 477, "current_phase_prediction_sign_changes": 4,
        "current_phase_known_genuine_zero_fields": 362, "current_phase_known_genuine_one_fields": None,
        "current_phase_genuine_zero_volatility_rows": None, "current_phase_known_no_new_low_rows": None,
        "current_phase_known_new_low_rows": None, "current_phase_raw_gap_known_rows": 1507,
        "current_phase_raw_gap_preserved_in_incomplete_rows": 745, "current_phase_above_one_absorption_rows": 327,
        "current_phase_next_B02_old_sources_checked": 22,
        "current_phase_field_scope": "原R94负隔夜吸收与原有符号gap；762联合有值745不完整，745原已知gap保留；另两系数固定函数整体拒绝。",
        "current_phase_prior_source_coverage_role": "3488日线与R94同值/1507身份/115原成熟月设计；非物理首次发布认证，原完整两列支持拒绝。",
        "current_member_support_this_continuation": "全部1010原可用预测完整配对/497原未知保留；762/1507原两项完整、745部分未知保留。",
        "validation_method_this_continuation": "15必要测试首次通过；49源/3488原字段/142版本/25方程/1507预测/24周期损失复算，无验证期新拟合；B02仅读元数据的事前标准SVD门拒绝。",
        "source_freeze_count_interpretation": "D02唯一金融函数49冻结对象/25实际辅助估计/90复用；B02原22源资格另核、0金融拟合；十一结果汇总不是新候选。",
        "historical_refit_counter_interpretation": "原核心重拟合0，本次25金融估计仅D02辅助两系数；复算/设计资格/十一结果汇总0金融估计。",
        "account_return_sharpe_this_continuation": "NOT_COMPUTED",
        "account_return_sharpe_reason": "D02原两期改进区间跨零、整体门失败；有限十一项0晋升，未进账户阶段。",
        "complete_D02_all_member_field_admission": False, "old_R94_direct_two_field_support_gate_passed": False,
        "complete_B02_all_member_field_admission": False, "old_R102_direct_five_field_support_gate_passed": False,
        "B02_optional_five_column_design_gate_passed": False,
        "original_strategy_source_files_changed_this_continuation": 0, "original_frozen_models_changed": False,
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "overfit_removed": False,
    })
    require({k: v for k, v in state.items() if protected(k)} == frozen_forward, "前瞻或已验证候选状态被改变。")
    state_after = (json.dumps(state, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for filename in DOCS:
        require((ROOT / filename).read_bytes() == original_docs[filename], "其他分支在准备中更新事实，停止覆盖。")
    require(STATE.read_bytes() == state_before, "目标状态在准备中变化，停止覆盖。")
    for filename, after in changes.items():
        replace_bytes(ROOT / filename, original_docs[filename], after)
    replace_bytes(STATE, state_before, state_after)
    for filename in DOCS:
        require((ROOT / filename).read_bytes() == changes[filename], "事实落盘不一致。")
    saved_state = read(STATE)
    require({k: v for k, v in saved_state.items() if protected(k)} == frozen_forward,
            "落盘前瞻状态不一致。")
    receipt = {
        "at": now(), "status": "PASS_R136_R137_R138_R139_REPORT_FACTS_STATE_UPDATE",
        "updated_facts": [{"path": f, "sha256": digest(ROOT / f)} for f in DOCS],
        "state_sha256": digest(STATE), "latest_actual_model_decision": "TECH.R137",
        "latest_technical_decision": "TECH.R139", "protected_forward_keys": len(frozen_forward),
        "forward_values_preserved": True, "original_frozen_strategy_changed": False,
        "new_model_fits_in_finalization": 0, "new_accounts": 0, "new_network_requests": 0,
        "goal_achieved": False, "blocked_audit_count": 0,
    }
    write_json(OUT / "facts_update_receipt.json", receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
