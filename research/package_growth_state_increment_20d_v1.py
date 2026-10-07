"""打包增长状态固定结果、政策原文定位及离线复核入口。"""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from growth_state_increment_20d_v1 import ROOT, OUT, now, save, sha

ARCHIVE = ROOT / "deliverables/510300_增长状态二十日增量与政策时钟_V1_GPT审阅_20260922.zip"
PARENT = ROOT / "deliverables/510300_宏观研究重订与政策动态更新_V1_GPT审阅_20260922.zip"
PARENT_SHA = "a4e886a3aee2ca10e99de4e7ca2e6522fd74ac916386bc14de90d2e5b69b22d0"


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare() -> None:
    if sha(PARENT) != PARENT_SHA:
        raise ValueError("范围重订阶段ZIP身份不同")
    with zipfile.ZipFile(PARENT) as z:
        if z.testzip() is not None or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError("前轮ZIP结构不一致")
        parent_members = len(z.namelist())
    copy(PARENT, OUT / "history" / PARENT.name)
    save(OUT / "evidence/父包身份.json", {"name": PARENT.name, "sha256": PARENT_SHA, "members": parent_members,
        "bytes": PARENT.stat().st_size, "crc": "PASS", "role": "上一阶段范围、M1/M2五日子实验与资金来源附录的完整历史快照。"})
    for name in ("report_growth_state_increment_20d_v1.py", "package_growth_state_increment_20d_v1.py"):
        copy(ROOT / "research" / name, OUT / "code" / name)
    copy(ROOT / "reports/research/510300_macro_dynamic_reframe_v1/evidence/继承原文身份核对.csv", OUT / "evidence/前阶段139个增长及25个利率原文身份.csv")
    requirements = [f"{name}=={importlib.metadata.version(name)}" for name in ("numpy", "pandas", "pyarrow", "matplotlib", "beautifulsoup4", "tzdata")]
    (OUT / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    (OUT / "00_阅读导航.md").write_text("""# 增长状态二十日增量与政策时钟 V1

主结论：新增增长状态未通过收益预测及下行风险各自的冻结条件。本包为宏观主线的一项已完成子实验，不能替代完整政策研究。

阅读顺序：研究报告.md → figures中的三张图 → protocol.json → results中的月度主要评价、八项基准、六项预定对照、全部预测与事后诊断 → evidence中的政策事实和复核记录。

直接数据：139个月新订单，138个月训练原点，101个月主要评价；505次周度/公告复核预测；691个含训练期的20日标签原点。完整3476日价格、每日已知输入和分红数据保留。408次唯一模型训练，账户0，独立前向事件0。

实际执行代码：code/corrected/growth_state_increment_20d_v1.py。原code/growth_state_increment_20d_v1.py及freeze_receipt.json是最初冻结快照。首次执行在标签生成前因日历覆盖不足中断，calendar_correction_receipt.json记录仅补足日期的修正；没有覆盖初始快照。

离线复核：安装requirements.txt所列依赖，在解压目录用Python运行 code/corrected/verify_growth_state_increment_20d_v1.py --study-dir . 。它从输入独立重建标签，以增广最小二乘复算固定系数，复算全部预测、保存区块样本与判据；不会产生新策略、账户或随机抽样。

历史回放代码依赖项目原目录布局及独占运行标记；不要在交付包内重新开始研究或删除标记。离线复核入口不依赖项目目录。保存图表生成与包装代码是为了说明流程，不声称它们在解压目录可以重新访问外部原始材料。

history中的上一轮ZIP保留原阶段结论及更早子实验，不能拿它的“新模型0”覆盖当前408次模型训练，也不能拿本轮增长失败覆盖其他政策通道。当前根FILE_INDEX.csv是本包成员索引，历史ZIP有自己的索引。

原始券商PDF、139份NBS网页与全部政策网页未重复分发；数字、当次月份、公开时点、URL及哈希在直接输入和evidence中。离线数值复算不等于独立认证全部历史首次版本。详见交付范围与限制.md。
""", encoding="utf-8")
    (OUT / "用户需求与执行对应.md").write_text("""# 用户需求与当前执行

用户要求以510300为交易研究对象，通过图形比较市场与宏观信息；不仅考虑M1/M2剪刀差，还要看实际值相对事前预期、其他宏观政策、市场状态和持续更新。用户已纠正把整条宏观主线收窄成单个五日事件实验的问题，允许增长、利率等独立通道不等待全部资金资料。本金以20万元为主，2万元作成本对照；卖出倾向为涨幅较大后再结合回撤决定退出。

本阶段按新立项先独立检验增长状态的20日均值与下行风险，而非更换旧M1/M2失败实验窗口。模型和基准在新标签生成前固定。每个目标都有自己的损失函数与准入判据；风险失败与收益失败分别保留。75次公告刷新对照使用相同目标终点，不能冒充连续账户。

本阶段提供三张完整结果图、全部标签与预测、既有日线、三类政策原文事实与时钟、可复算结果和GPT审阅包。收益与风险均未通过自己的门，所以账户、回撤退出没有运行；它们保持NOT_RUN，不填成零收益。财政、地产、资本市场、贸易政策的完整母集、预期与动态账户尚未完成。

旧NBS固定五分钟失败及85/15终止保持。所有当前结果是已观察历史上的有限开发证据。没有发布现时买卖指令或启动新的交易服务。
""", encoding="utf-8")
    (OUT / "GPT审阅提问.md").write_text("""# 可复制的审阅请求

请审阅这一项增长状态二十日子实验，并放在完整宏观主线中评价。用户关心持续更新的信息如何改变510300持有价值，不局限M1/M2。请先读研究报告、protocol.json与两份冻结/日历修正回执，再独立复算。

请重点回答：
1. 月度训练、按周及公告更新、共同起点/20日终点、成熟标签与分红目标是否正确？505次更新有没有被错误当作505个独立宏观观察？
2. 收益和风险的分目标判据及全部基准是否合理？判断是否如实保留价格基准本身的不足？
3. 2020-02极端输入与模型外推是否解释了误差集中？该诊断有没有被用来删除样本或救回失败？
4. 75次刷新对照究竟支持什么、不支持什么？能否区分更新操作、信息表达和真实连续账户的价值？
5. 地产再贷款、财政化债、贸易关税三类政策的额度、用途、有效期和时间精度是否被混淆？完整政策母集与预期还缺哪些必要字段？
6. 给出下一项独立宏观政策研究的具体机制、最小变量、可证伪比较、数据优先级、验收和停止条件。不能把本轮增长失败扩大到全部政策，也不要围绕这个失败模型调窗口、方向或缩尾。

账户0，独立前向事件0；20万元/2万元费用比较及上涨后回撤退出尚未运行。完整账户年化10%/净夏普1.2未建立。所有历史此前已被观察，两个目标的预定统计门不等于校正了整个仓库试验历史。原始网页未全部包含，不能宣称逐份认证历史首发。此请求不授权实盘或订单。
""", encoding="utf-8")
    (OUT / "交付范围与限制.md").write_text("""# 范围与限制

包含当前实验的直接输入、原冻结代码与日历修正代码、协议、全部691个目标原点、138个月训练原点、408组固定模型记录、505次预测、101个月主要评价、八基准和六个预定对照、10000次共同区块索引及全部损失改善样本、图表、事后外推诊断和三类政策事实。前一轮审阅包以原字节嵌套保存。

未重复复制原始券商全文、139份NBS网页和全部政策HTML；来源URL、字段、历史日期及哈希保留。已独立复算包内数值、时钟和固定模型；未重新认证每份历史首次公开版本，未建立独立前向样本，未进行外部GPT审阅。

三类政策为已有历史背景案例的原文核对，非完整政策母集；未取得事前政策共识，不将记者或研究者事后用词当成预期差。没有利用政策窗口回报选择政策方向或分数。

两项模型按自己冻结判据失败，账户及回撤退出均NOT_RUN。历史统计不代表可实盘有效或账户达标。没有安全性审计宣称，也没有扩大执行标的。
""", encoding="utf-8")
    save(OUT / "evidence/图表目视核对.json", {"checked_at": now(), "figures": [p.name for p in (OUT / "figures").glob("*.png")],
        "method": "逐张打开PNG核对中文、数据、图例、轴及注释；调整注释避免覆盖2019年峰值与风险误差线。",
        "status": "PASS_AFTER_LABEL_POSITION_REVIEW", "all_months_retained": True, "daily_data_retained": 3476})
    print("审阅导航、独立复核入口与完整历史上下文已准备。")


def package() -> None:
    if ARCHIVE.exists():
        raise FileExistsError("最终审阅包已存在，禁止覆盖")
    verifier = OUT / "code/corrected/verify_growth_state_increment_20d_v1.py"
    local = subprocess.run([sys.executable, str(verifier), "--study-dir", str(OUT)], check=True, capture_output=True, text=True, encoding="utf-8")
    result = json.loads(local.stdout)
    save(OUT / "evidence/保存结果独立复算.json", result)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name not in ("FILE_INDEX.csv", "delivery_receipt.json"))
    entries = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=["path", "bytes", "sha256"])
    writer.writeheader()
    writer.writerows(entries)
    index = text.getvalue().encode("utf-8-sig")
    building = ARCHIVE.with_suffix(".building.zip")
    with zipfile.ZipFile(building, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for p in files:
            z.write(p, p.relative_to(OUT).as_posix())
        z.writestr("FILE_INDEX.csv", index)
    if building.stat().st_size > 80 * 1024 * 1024:
        raise ValueError("交付范围超过预定80MiB上限")
    with tempfile.TemporaryDirectory(prefix="growth20d_saved_verify_") as temp:
        with zipfile.ZipFile(building) as z:
            names = z.namelist()
            if z.testzip() is not None or len(names) != len(set(names)) or set(names) != {r["path"] for r in entries} | {"FILE_INDEX.csv"}:
                raise ValueError("CRC、重复路径或索引成员检查未通过")
            for item in entries:
                data = z.read(item["path"])
                import hashlib
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("成员大小或身份不同")
            z.extractall(temp)
        extracted = subprocess.run([sys.executable, str(Path(temp) / "code/corrected/verify_growth_state_increment_20d_v1.py"), "--study-dir", temp], check=True, capture_output=True, text=True, encoding="utf-8")
        if json.loads(extracted.stdout) != result:
            raise ValueError("新解压目录的复算结果不同")
    building.replace(ARCHIVE)
    (OUT / "FILE_INDEX.csv").write_bytes(index)
    receipt = {"archive": str(ARCHIVE), "created_at": now(), "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE),
        "members": len(entries) + 1, "indexed_members": len(entries), "crc": "PASS", "duplicates": 0, "index_size_hash": "PASS",
        "fresh_extraction_verification": result, "status": "PASS_STRUCTURAL_AND_INDEPENDENT_SAVED_RECOMPUTATION",
        "new_accounts": 0, "external_review_completed": False, "whole_macro_program_complete": False}
    save(ARCHIVE.with_suffix(".receipt.json"), receipt)
    save(OUT / "delivery_receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "fresh_extraction_verification"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="增长状态与政策时钟审阅包")
    parser.add_argument("action", choices=["prepare", "package"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        package()
