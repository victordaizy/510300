"""打包本轮期权波动率训练的已有结果，不训练、不生成账户。"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/research/510300_option_volatility_probability_payoff_v1"
OUTPUT = ROOT / "deliverables/510300_期权波动率_首轮训练_20260924.zip"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    result = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    verified = json.loads((SOURCE / "verification/验证结果.json").read_text(encoding="utf-8"))
    if verified["status"] != "PASS_SAVED_LABELS_PREDICTIONS_AND_METRICS":
        raise RuntimeError("保存的样本、预测与指标尚未核对通过。")
    records: dict[str, bytes] = {}
    for p in sorted(SOURCE.rglob("*")):
        if p.is_file() and p.name != "delivery_receipt.json":
            records[p.relative_to(SOURCE).as_posix()] = p.read_bytes()
    records["code/package_option_volatility_probability_payoff_v1.py"] = Path(__file__).read_bytes()
    records["用户当前指令.md"] = (
        "用户最新指令：我只做510300期权，高夏普，稳定盈利，可以多空双开，波动率赚钱。\n\n"
        "沿用同一任务此前的20万元本金、成本后净夏普至少1.3、最大回撤10%、每完整自然年至少5次交易、"
        "先训练胜率与盈亏幅度、不新增行情采集的要求。新指令明确更新期权研究范围；不含实盘、下单或券商连接授权。\n"
    ).encode("utf-8")
    records["00_README_FIRST.md"] = (
        "# 510300期权波动率：首轮训练资料\n\n"
        "阅读顺序：第一轮训练结果.md → summary.json → protocol.json → results/固定经验基线对照.csv → "
        "results/完整滚动预测.csv → models/全部月度参数.json。\n\n"
        "本轮完成买入跨式、保护性卖出跨式的概率与盈亏幅度模型。没有资金账户回测，"
        "没有新的账户夏普、回撤或年度完成交易次数。盈利能力与用户目标尚未得到证明。\n\n"
        "完整输入位于inputs；code为冻结实现；freeze.json记录首次生成本轮盈亏标签之前的协议与文件指纹。"
        "历史合约用每日官方交易代码的M标识还原标准行权价与单位，避免后来合约快照造成的筛选前视。\n\n"
        "候选样本保留未知标签和原因；完整预测也保留未来标签不可评价的候选。"
        "未知标签不能在将来的资金回测里事后作为拒绝交易条件。日线开收盘及高低价不是同步组合成交报价。\n\n"
        "旧研究/目录中保留此前方向策略2024年失败的结果及研究状态清单，作为避免重复试验的背景，"
        "不把旧开发期通过作为本轮的盈利证据。\n\n"
        "只读复核命令：项目Python运行 code/option_volatility_probability_payoff_v1.py verify --root 解压目录。"
        "复核不训练、不下载、不生成账户，但会写入单独的verification结果记录。\n\n"
        "本包只有本轮直接输入及必要背景，不含其他策略全仓库、IO指数期权、终端最后若干日分钟数据、"
        "零散盘口快照、券商凭据或实盘系统。FILE_INDEX.csv为本包文件清单。\n"
    ).encode("utf-8")
    records["01_给GPT的复核提示.md"] = (
        "请先阅读00_README_FIRST.md及用户当前指令。核对当前研究是否真正只使用510300 ETF期权，"
        "并评估已完成的两个固定波动率结构的预测结果。\n\n"
        "请区分：本地已核对的标签和模型参数、日线成交代理假设、未计算的账户绩效，以及未证实的实盘盈利。"
        "重点检查历史M合约标识、后来快照的前视风险、标签未知样本、月度拟合时钟、标签重叠、固定经验基线、"
        "Gamma条件幅度、期权组合方向敞口及保护翼。不要将候选平均损益当作独立交易账户。\n\n"
        "给出：1.最有影响的错误或反证；2.概率、盈利幅度、亏损幅度是否稳定改善；3.后续优先工作；"
        "4.保证金、逐日估值、调整与缺失数据需要怎样验证；5.防止过拟合的明确停止条件。"
        "请勿通过反复改变持有期、翼宽、方向、阈值来救回失败；本包不是外部审阅已通过的证据，也不授权交易。\n"
    ).encode("utf-8")
    old_results = []
    for p in sorted((ROOT / "reports/discovery").glob("*option*json")):
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        old_results.append({"file":p.relative_to(ROOT).as_posix(),"status":d.get("status"),"sha256":sha(p.read_bytes())})
    records["旧研究/已查阅研究状态.json"] = json.dumps(old_results,ensure_ascii=False,indent=2).encode("utf-8")
    old_validation = ROOT / "reports/validation/510300_option_ivs_transfer_v1_validation_2024.json"
    records["旧研究/510300_option_ivs_transfer_v1_validation_2024.json"] = old_validation.read_bytes()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream,fieldnames=["path","size_bytes","sha256"])
    writer.writeheader()
    for name,data in sorted(records.items()):
        writer.writerow({"path":name,"size_bytes":len(data),"sha256":sha(data)})
    records["FILE_INDEX.csv"] = stream.getvalue().encode("utf-8-sig")
    temporary = OUTPUT.with_suffix(".building.zip")
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data in sorted(records.items()):
            archive.writestr(name,data)
    with zipfile.ZipFile(temporary) as archive:
        names = archive.namelist()
        if len(names)!=len(set(names)) or archive.testzip() is not None:
            raise RuntimeError("压缩包成员或CRC检查失败。")
        index = list(csv.DictReader(io.StringIO(archive.read("FILE_INDEX.csv").decode("utf-8-sig"))))
        if set(x["path"] for x in index)!=(set(names)-{"FILE_INDEX.csv"}):
            raise RuntimeError("文件索引覆盖不完整。")
        for row in index:
            content = archive.read(row["path"])
            if len(content)!=int(row["size_bytes"]) or sha(content)!=row["sha256"]:
                raise RuntimeError(f"文件大小或指纹不一致：{row['path']}")
    os.replace(temporary,OUTPUT)
    receipt = {"path":str(OUTPUT),"size_bytes":OUTPUT.stat().st_size,"sha256":sha(OUTPUT.read_bytes()),
               "members":len(records),"index_rows":len(records)-1,"zip_structure":"PASS",
               "saved_model_and_label_verification":verified["status"],"research_status":result["status"],
               "new_fits_during_packaging":0,"new_accounts_during_packaging":0,"new_downloads":0,
               "external_review":"NOT_PERFORMED","goal_achieved":False}
    (SOURCE/"delivery_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
