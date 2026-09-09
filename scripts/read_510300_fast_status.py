"""只输出继续研究所需的最新字段，避免反复展开全部历史索引。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / "reports/research/510300_sharpe_1_2_latest_research.json").read_text(encoding="utf-8"))
    recent = []
    for record in index["completed_rounds"][-3:]:
        result = json.loads((ROOT / record["result"]).read_text(encoding="utf-8"))
        cfg_model = record["primary_base"]["model"]
        recent.append({"轮次": record["round"], "研究": record["study"], "结论": record["status"],
            "主基础夏普": record["primary_base"]["net_sharpe"], "主压力夏普": record["primary_stress"]["net_sharpe"],
            "较早夏普": {m["cost"]: m["net_sharpe"] for m in result["earlier_diagnostics"] if m["model"] == cfg_model},
            "核心秒数": result.get("run_seconds"), "结果": record["result"]})
    output = {"更新时间": index["updated_at"], "完整目标实现": index["goal_achieved"], "计数": index["count_warning"],
        "运行中": index["running_studies"], "最近三轮": recent, "下一项": index["next_work"],
        "四情景均衡比较": index.get("current_best_four_scenario_comparison_candidate"),
        "主历史两档费用达到点值的新增比较": index.get("additional_covariance_main_sharpe_target_candidate"),
        "合成下行的新增比较": index.get("additional_joint_downside_main_sharpe_target_candidate"),
        "最近交付": index["deliveries"][-1]}
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
