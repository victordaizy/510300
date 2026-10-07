"""打包官方政策目录、六类信息链、图表及离线复核入口。"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from policy_information_clock_v1 import ROOT, OUT, now, save, sha

ARCHIVE = ROOT / "deliverables/510300_官方政策目录与六类信息更新_V1_GPT审阅_20260922.zip"
PARENT = ROOT / "deliverables/510300_增长状态二十日增量与政策时钟_V1_GPT审阅_20260922.zip"
PARENT_SHA = "24c37d4ffb218f0d60b4518c0fa971aec4bd5d52a747ca6019e8ba0bb7ef2a58"


def prepare() -> None:
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("前轮结果包身份不同")
    with zipfile.ZipFile(PARENT) as z:
        if z.testzip() is not None:
            raise ValueError("前轮包CRC失败")
    shutil.copy2(PARENT, OUT / "history" / PARENT.name)
    save(OUT / "evidence/父阶段身份.json", {"name": PARENT.name, "sha256": PARENT_SHA, "bytes": PARENT.stat().st_size,
        "role": "完整保留前一增长状态失败、宏观主线重订、旧货币实验及资金附录；当前资料不改变旧裁决。"})
    for name in ("policy_information_clock_v1.py", "report_policy_information_clock_v1.py", "verify_policy_information_clock_v1.py", "package_policy_information_clock_v1.py"):
        shutil.copy2(ROOT / "research" / name, OUT / "code" / ("current_" + name if name == "policy_information_clock_v1.py" else name))
    original = Path("E:/CodexData/.codex/attachments/9985585b-6e77-41bf-b927-0e02209a9c57/pasted-text-1.txt")
    if original.exists():
        shutil.copy2(original, OUT / "inputs/最初研究目标_用户附件.txt")
    (OUT / "用户纠正与本轮对应.md").write_text("""# 用户要求与本轮对应

用户的最新纠正：研究不能局限M1/M2，也不能把完整宏观方向缩成公告后剩余五日预测。需要判断持续更新的信息怎样影响未来持有价值，覆盖经济传导、市场状态和新公告后的重新判断。货币实际值、事前预期、价格先行反应、资金持续调整、支撑与退出各自需要验证。用户倾向涨幅较大后结合回撤退出；20万元为主账户，2万元为成本对照；允许查询网络和微信公众号。

用户指出旧五日模型未包括中期经济状态，价格基准没有证明预测价值；24次A/B预测全部为负，新增剪刀差预期差没有产生不同入场。旧失败结论仅限具体模型，不能扩大到全部宏观。用户建议新研究固定20日并按周和公告更新，停止围绕旧失败改参数。上述纠正继续生效。

当前完成四份官方年度目录的全部固定入口提取、六类政策24个来源节点及其复核时钟、当时可观察的货币和价格状态、四张图、完整日频数据。不同通道不压成一个利好分数，政策预期缺失保留UNKNOWN；依据新的文章数量不直接跑账户。

尚未完成全部政策首发母集与历史版本、完整通胀预期资料、政策模型增量、真实资金模块、连续账户、固定入场后的退出比较。这些未完成项保留在完整宏观主线，资金缺口不封锁独立通道。本轮不是整个目标完成，也不是当前市场买卖建议。
""", encoding="utf-8")
    (OUT / "00_阅读导航.md").write_text("""# 官方政策目录与六类信息更新 V1

先读研究结论.md，再看figures中的四张图。用户纠正与本轮对应.md解释完整任务，protocol.json界定当前数据阶段。

results/全部官方目录记录.csv有490行：2024/2025国新办年度目录186/167个入口，央行年度大事记72/65条。它们是目录记录，不等于独立政策数；所有记录均为追溯线索，不能直接当历史交易输入。2024目录入口少于官方回顾所称全年发布会数量，不宣称全集。

results/跨通道政策链_完整事实与时钟.csv保存六条链、24个来源节点。每个节点分列此前已知、本次新增、未知、经济发生日期、来源可用上界、固定复核收盘、规则执行开盘、上界之后首个日频开盘。政策共识字段全部UNKNOWN，不是零意外。

results/政策复核时点_当时可观察状态.csv和results/两年全日频走势与当时已知货币数据.csv是独立复核的状态数据。完整3476日行情在inputs/market.parquet，走势图保留2024—2025全部485日；未复权价格不等于含费用净值。M1新旧口径分开显示。

raw和receipts保留本轮网页、访问失败、既有原文复用和网页工具文本来源。Web工具文本不是原始HTTP文件。来源哈希与正文定位不等于认证历史最早版本。协议、24节点冻结与最终代码版本分别保留，没有产生新预测模型、账户或退出检验。

离线复核：安装requirements.txt后，在任意解压目录用Python运行 code/verify_policy_information_clock_v1.py --study-dir . 。该入口独立提取固定目录、检查来源定位，重算时钟及全部逐日已知值；不下载、不训练、不生成账户或随机样本。原始采集及绘图代码仍按项目目录布局运行，不能当成联网一键复现承诺。

history内嵌前一阶段完整审阅包，原字节身份已核对。前阶段有408次增长模型训练及其失败；当前阶段新训练为0。两个数字属于不同阶段，不可互相覆盖。

根FILE_INDEX.csv列明本包成员、大小与SHA-256，外部receipt记录包自身身份。该包用于外部审阅准备，不表示已经完成外部GPT审阅。完整宏观目标仍在进行。
""", encoding="utf-8")
    (OUT / "GPT审阅提问.md").write_text("""# 可复制的审阅请求

请把本包视为宏观主线的政策数据阶段。用户要判断不断更新的信息如何改变510300未来持有价值，范围包括增长、通胀、利率、财政、地产、资本市场和外部政策，不能缩成一个货币预期差五日实验。请先读研究结论和用户纠正，再查看图表和完整CSV，可运行独立离线复核。

请具体回答：
1. 490条目录与24个来源节点是否被正确区分？年度回顾、英文目录、会议日期、文件落款、首次公开和执行日期有没有混淆？哪些节点还不能作为严格历史输入？
2. 各政策金额是否混淆额度、申请、当次操作、实际投放、债务置换或新增需求？6万亿化债与3000亿两新是否给出正确传导解释？
3. 2025年5月7日、5月12日盘前和15:00的不同消息，是否正确限制了可用信息和可交易起点？固定等待收盘的成本怎样与信息本身的价值分开验证？
4. 源上界、M1口径、月度重复填充和状态成熟性是否正确？图表是否只支持时序描述，是否有越界的因果或预测主张？
5. 对尚缺的通胀、完整政策共识、全国财政执行等信息，提出最小且可取得的原始资料优先级。资金数据不足不能锁死独立通道。
6. 明确下一项独立研究到底预测20日收益、下行风险还是改变持有条件；给出机制、最小变量、可观察状态交互、简单基准、更新时钟、接受与停止条件。不要用24个案例的收益选规则，也不要把旧失败模型换参数救回。
7. 只有模块通过自己的目标门后，才比较周度/公告更新连续账户、20万元/2万元成本和涨幅后回撤退出。指出何种证据足以进入下一步，何种结果应停止。

当前新模型0、新账户0、独立前向事件0。NO_VIEW表示没有已验证规则，不表示预测下跌或目标现金仓位。旧货币、增长和NBS失败以及85/15终止保留。此包未建立年化10%/净夏普1.2，也不授权订单或实盘。
""", encoding="utf-8")
    (OUT / "交付范围与限制.md").write_text("""# 交付范围与限制

包含：四份固定官方目录网页、全部490条目录记录、24次LPR含不变记录、六类政策24个来源节点与直接来源、公布时钟、金额含义、24次状态快照、2024—2025全部485日绘图对齐值、完整3476日价格、104个月货币与139个月新订单输入、旧25条操作记录、四张PNG/SVG、协议与代码、离线复核、原用户附件和前阶段完整包。

不把固定目录说成全国政策全集，不把目录行数说成独立事件数。没有重新认证每个原始网页的历史首发版本，也未补齐所有CPI/PPI共识、财政支出、贸易后续实施/延期和机构资金交易。继承月度原始券商PDF及全部NBS网页未逐份重复打包；前阶段来源清单、身份记录与边界保留。

访问情况：国新办英文HTTPS证书主机名校验失败，采用同域公开HTTP目录，未关闭TLS校验。部分部委网页返回403或502，失败响应和回执保留；三条贸易信息使用网页工具取得的官方正文文本，日内瓦条款说明复用前阶段已核对SHA的官方HTML。一个英文历史详情返回200但只有页面壳，未用作事实节点。HTTP200与成功取得政策内容不同。

数值复核仅证明包内目录、金额定位、时钟及状态与输入一致，不证明预测价值、因果关系、独立前向有效或交易可执行。没有新的拟合、回测账户、订单或服务。没有实施额外安全性审计，未宣称外部审阅已完成。
""", encoding="utf-8")
    requirements = [f"{n}=={importlib.metadata.version(n)}" for n in ("pandas", "numpy", "pyarrow", "matplotlib", "beautifulsoup4", "tzdata")]
    (OUT / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    probes = []
    for path in sorted((OUT / "receipts").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        probes.append({"key": r.get("key", path.stem), "http_code": r.get("http_code"), "format": r.get("format", "HTML_OR_HTTP_ERROR"),
                       "url": r.get("url"), "raw_path": r.get("raw_path"), "sha256": r.get("sha256")})
    save(OUT / "evidence/全部来源取得方式.json", probes)
    print("阅读导航、源文件、历史包与审阅提问已准备。")


def package() -> None:
    if ARCHIVE.exists():
        raise FileExistsError("最终审阅包已存在，禁止覆盖")
    verifier = OUT / "code/verify_policy_information_clock_v1.py"
    run = subprocess.run([sys.executable, str(verifier), "--study-dir", str(OUT)], check=True, capture_output=True, text=True, encoding="utf-8")
    checked = json.loads(run.stdout)
    save(OUT / "evidence/保存目录时钟状态独立复核.json", checked)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in ("FILE_INDEX.csv", "delivery_receipt.json"))
    entries = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(entries)
    index = buffer.getvalue().encode("utf-8-sig")
    building = ARCHIVE.with_suffix(".building.zip")
    with zipfile.ZipFile(building, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for p in files:
            z.write(p, p.relative_to(OUT).as_posix())
        z.writestr("FILE_INDEX.csv", index)
    if building.stat().st_size > 80 * 1024 * 1024:
        raise ValueError("超出80MiB交付范围")
    with tempfile.TemporaryDirectory(prefix="policy_clock_verify_") as temp:
        with zipfile.ZipFile(building) as z:
            names = z.namelist()
            if z.testzip() is not None or len(names) != len(set(names)):
                raise ValueError("ZIP的CRC或重复成员检查失败")
            if set(names) != {r["path"] for r in entries} | {"FILE_INDEX.csv"}:
                raise ValueError("索引成员与ZIP不一致")
            for row in entries:
                binary = z.read(row["path"])
                if len(binary) != row["bytes"] or hashlib.sha256(binary).hexdigest() != row["sha256"]:
                    raise ValueError("索引大小或哈希不一致")
            z.extractall(temp)
        extracted = subprocess.run([sys.executable, str(Path(temp) / "code/verify_policy_information_clock_v1.py"), "--study-dir", temp], check=True, capture_output=True, text=True, encoding="utf-8")
        if json.loads(extracted.stdout) != checked:
            raise ValueError("新解压目录的保存数据复算结果不同")
    building.replace(ARCHIVE)
    (OUT / "FILE_INDEX.csv").write_bytes(index)
    receipt = {"created_at": now(), "archive": str(ARCHIVE), "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE),
        "members": len(entries) + 1, "indexed_members": len(entries), "crc": "PASS", "duplicates": 0, "index_size_hash": "PASS",
        "fresh_extraction_recomputation": checked, "status": "PASS_STRUCTURAL_AND_SAVED_RECOMPUTATION",
        "new_accounts": 0, "new_research_fits": 0, "external_review_completed": False, "whole_macro_objective_complete": False}
    save(ARCHIVE.with_suffix(".receipt.json"), receipt)
    save(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "fresh_extraction_recomputation"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="政策信息研究审阅包")
    parser.add_argument("action", choices=["prepare", "package"])
    a = parser.parse_args()
    prepare() if a.action == "prepare" else package()
